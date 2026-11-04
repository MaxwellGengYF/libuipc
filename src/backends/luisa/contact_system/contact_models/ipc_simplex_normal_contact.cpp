/********************************************************************
 * @file contact_models/ipc_simplex_normal_contact.cpp
 * @brief T8.1b — `IPCSimplexNormalContact`: the concrete IPC simplex normal contact model
 * (port of src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu,
 * 576 reference lines).
 *
 * It supplies the four energies (PT / EE( + mollifier) / PE / PP), and the
 * gradient+Hessian assembly, through `SimplexNormalContact`'s two virtuals:
 *   do_compute_energy -> four `do_compute_energy_k{1..4}_kernel` launches, one per candidate
 *                        array (the reference's `if(count > 0)` guard, verbatim);
 *   do_assemble       -> **four** launches, one per candidate type
 *                        (`do_assemble_{PT,EE,PE,PP}_kernel<GradientOnly>`), each over its own
 *                        `count` threads.
 *
 * DELIBERATE DEVIATION (rule 14), measured: the reference fuses all four types into **one**
 * launch, but that kernel holds four `make_spd` chains (12x12 PT, 12x12 EE, 9x9 PE, 6x6 PP)
 * and its generated CUDA source takes over ten minutes in NVRTC -- with two or more of them
 * pending in one command batch the wait is unbounded in practice (measured: > 900 s of wall
 * clock, the host blocked inside `Device::compile` with its NVRTC helper process still
 * running).  Splitting into four kernels and committing each with a host
 * `frame_stream().synchronize()` after its dispatch keeps the compile serial and finite.
 * The split's one contract: every per-type kernel rebuilds its device views with
 * `from_handles_full` from the reporter subview's `*_index_offset()`/`total_*_count()`,
 * because `get_index()` adds the offset — the offset-less `from_handles` would make all four
 * types write from slot 0 and the last writer would win.
 *
 * MAPPINGS (all mechanical, cookbook §3):
 *  * `cuda_tool::CDense2D<ContactCoeff> table` -> a flat `lcrt::Buffer<ContactCoeff>` + the
 *    `contact_tabular_size()` extent (cookbook §7.35 rule 1); the `PT_kappa(table, cids)`
 *    family of `codim_ipc_simplex_normal_contact_function.h` takes both.
 *  * `Vector4i/Vector3i/Vector2i` reads -> `load_vec4i/load_vec3i/load_vec2i`, positions ->
 *    `load_vec3`, `Vector12`/`Matrix12x12` locals -> `lps::LVec<Var<Float>,12>` /
 *    `lps::LMat<Var<Float>,12,12>`.
 *  * `Distance::point_triangle_distance_flag(...)` -> `ldist::point_triangle_distance_flag`.
 *  * `make_spd(H)` -> `lps::make_spd_inplace(H, kMakeSpdSweeps)` (the reference's
 *    `utils/make_spd.h` is a tolerance-driven Jacobi loop, so the port fixes the sweep
 *    count; the T5.2 `hs1d_detail::kMakeSpdSweeps` choice, which reaches the exact eigenvalue
 *    clamp for N = 6/9/12 — the same 8.
 *  * `DoubletVectorAssembler DVA{PT_Gs}` / `TripletMatrixAssembler TMA{PT_Hs}` -> the port's
 *    own assemblers over the *device* views built in-kernel with
 *    `lla::LDoubletVectorDeviceView<Var<Float>,3>::from_handles_full(...)` /
 *    `lla::LTripletMatrixDeviceView<...>::from_handles_full(...)` from the flat slice handles
 *    the host passes (`indices/values/rows/cols` + the slice's element counts *and* its
 *    index offsets, see the deviation note above).  The
 *    `half_block<4>` upper-triangle write order and the `upper_LR` select are the ported
 *    assembler's, i.e. the reference's.
 *  * `UIPC_KERNEL_ASSERT(cond, "fmt", args...)` -> `lcrt::device_assert(cond, "literal")`
 *    (LC's `device_assert` takes a message literal only, see the T7.2b kernels' note).
 *    The guards stay inside the reference's `if constexpr(RUNTIME_CHECK)` block.
 *********************************************************************/
#include <contact_system/simplex_normal_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <luisa_device/math/distance/distance.h>
#include <utils/codim_thickness.h>
#include <kernel_cout.h>
#include <utils/matrix_assembler.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace (utils/make_spd.h)
#include <utils/primitive_d_hat.h>
#include <pipeline/ipc_pipeline_flag.h>

namespace uipc::backend::luisa
{
namespace
{
namespace ldist = luisa_device::math::distance;
namespace lla   = luisa_device::linalg;

/// The sweep count passed to `lps::make_spd_inplace` for the 12x12/9x9/6x6 pair Hessians
/// (the reference's tolerance-driven `utils/make_spd.h` loop has no device equivalent; the
/// T5.2 measurement — `hs1d_detail::kMakeSpdSweeps` — is the port's 8).
inline constexpr int kMakeSpdSweeps = 8;

/// The four half-Hessian block sizes as *host* `int32` constants: the assembler's index is a
/// DSL value, and LC's `*` requires both operands to have the same element type (`SizeT` is
/// 64-bit), so the literal has to be narrowed once, here.
constexpr std::int32_t kPTHalfHessianSize =
    static_cast<std::int32_t>(SimplexNormalContact::PTHalfHessianSize);
constexpr std::int32_t kEEHalfHessianSize =
    static_cast<std::int32_t>(SimplexNormalContact::EEHalfHessianSize);
constexpr std::int32_t kPEHalfHessianSize =
    static_cast<std::int32_t>(SimplexNormalContact::PEHalfHessianSize);
constexpr std::int32_t kPPHalfHessianSize =
    static_cast<std::int32_t>(SimplexNormalContact::PPHalfHessianSize);

/// `do_compute_energy_k1_kernel` (PT).
auto& do_compute_energy_k1_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector4i>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff>        table,
               lcrt::Var<std::int32_t>              table_size,
               lcrt::BufferVar<IndexT>              contact_ids,
               lcrt::BufferVar<LVector4i>           PTs,
               lcrt::BufferVar<Float>               Es,
               lcrt::BufferVar<LVector3>            Ps,
               lcrt::BufferVar<Float>               thicknesses,
               lcrt::BufferVar<Float>               d_hats,
               lcrt::Var<Float>                     dt,
               lcrt::Var<std::uint32_t>             n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    using namespace sym::codim_ipc_simplex_contact;

                    auto PT   = load_vec4i(PTs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 4>{contact_ids->read(PT(0)),
                                                                contact_ids->read(PT(1)),
                                                                contact_ids->read(PT(2)),
                                                                contact_ids->read(PT(3))};
                    auto kt2  = lps::detail::mul(
                        lps::detail::mul(PT_kappa(table, table_size, cids), dt), dt);

                    const auto P  = load_vec3(Ps->read(PT(0)));
                    const auto T0 = load_vec3(Ps->read(PT(1)));
                    const auto T1 = load_vec3(Ps->read(PT(2)));
                    const auto T2 = load_vec3(Ps->read(PT(3)));

                    auto thickness = PT_thickness(thicknesses->read(PT(0)),
                                                  thicknesses->read(PT(1)),
                                                  thicknesses->read(PT(2)),
                                                  thicknesses->read(PT(3)));

                    auto d_hat = PT_d_hat(d_hats->read(PT(0)),
                                          d_hats->read(PT(1)),
                                          d_hats->read(PT(2)),
                                          d_hats->read(PT(3)));

                    auto flag = ldist::point_triangle_distance_flag(P, T0, T1, T2);

                    if constexpr(uipc::RUNTIME_CHECK)
                    {
                        lcrt::Var<Float> D;
                        ldist::point_triangle_distance2(flag, P, T0, T1, T2, D);

                        auto range = D_range(thickness, d_hat);

                        lcrt::device_assert(is_active_D(range, D),
                                            "PT d^2 out of range");
                    }

                    Es->write(i,
                              PT_barrier_energy(flag, kt2, d_hat, thickness, P, T0, T1, T2));
                };
            });
    return k;
}

/// `do_compute_energy_k2_kernel` (EE, mollified).
auto& do_compute_energy_k2_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector4i>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff>    table,
               lcrt::Var<std::int32_t>          table_size,
               lcrt::BufferVar<IndexT>          contact_ids,
               lcrt::BufferVar<LVector4i>       EEs,
               lcrt::BufferVar<Float>           Es,
               lcrt::BufferVar<LVector3>        Ps,
               lcrt::BufferVar<Float>           thicknesses,
               lcrt::BufferVar<LVector3>        rest_Ps,
               lcrt::BufferVar<Float>           d_hats,
               lcrt::Var<Float>                 dt,
               lcrt::Var<std::uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    using namespace sym::codim_ipc_simplex_contact;

                    auto EE   = load_vec4i(EEs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 4>{contact_ids->read(EE(0)),
                                                                contact_ids->read(EE(1)),
                                                                contact_ids->read(EE(2)),
                                                                contact_ids->read(EE(3))};
                    auto kt2  = lps::detail::mul(
                        lps::detail::mul(EE_kappa(table, table_size, cids), dt), dt);

                    const auto E0 = load_vec3(Ps->read(EE(0)));
                    const auto E1 = load_vec3(Ps->read(EE(1)));
                    const auto E2 = load_vec3(Ps->read(EE(2)));
                    const auto E3 = load_vec3(Ps->read(EE(3)));

                    const auto t0_Ea0 = load_vec3(rest_Ps->read(EE(0)));
                    const auto t0_Ea1 = load_vec3(rest_Ps->read(EE(1)));
                    const auto t0_Eb0 = load_vec3(rest_Ps->read(EE(2)));
                    const auto t0_Eb1 = load_vec3(rest_Ps->read(EE(3)));

                    auto thickness = EE_thickness(thicknesses->read(EE(0)),
                                                  thicknesses->read(EE(1)),
                                                  thicknesses->read(EE(2)),
                                                  thicknesses->read(EE(3)));

                    auto d_hat = EE_d_hat(d_hats->read(EE(0)),
                                          d_hats->read(EE(1)),
                                          d_hats->read(EE(2)),
                                          d_hats->read(EE(3)));

                    auto flag = ldist::edge_edge_distance_flag(E0, E1, E2, E3);

                    if constexpr(uipc::RUNTIME_CHECK)
                    {
                        lcrt::Var<Float> D;
                        ldist::edge_edge_distance2(flag, E0, E1, E2, E3, D);
                        auto range = D_range(thickness, d_hat);
                        lcrt::device_assert(is_active_D(range, D),
                                            "EE d^2 out of range");
                    }

                    Es->write(i,
                              mollified_EE_barrier_energy(flag,
                                                          // coefficients
                                                          kt2,
                                                          d_hat,
                                                          thickness,
                                                          // positions
                                                          t0_Ea0,
                                                          t0_Ea1,
                                                          t0_Eb0,
                                                          t0_Eb1,
                                                          E0,
                                                          E1,
                                                          E2,
                                                          E3));
                };
            });
    return k;
}

/// `do_compute_energy_k3_kernel` (PE).
auto& do_compute_energy_k3_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector3i>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff>    table,
               lcrt::Var<std::int32_t>          table_size,
               lcrt::BufferVar<IndexT>          contact_ids,
               lcrt::BufferVar<LVector3i>       PEs,
               lcrt::BufferVar<Float>           Es,
               lcrt::BufferVar<LVector3>        Ps,
               lcrt::BufferVar<Float>           thicknesses,
               lcrt::BufferVar<Float>           d_hats,
               lcrt::Var<Float>                 dt,
               lcrt::Var<std::uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    using namespace sym::codim_ipc_simplex_contact;

                    auto PE = load_vec3i(PEs->read(i));

                    auto cids = lps::LVec<lcrt::Var<IndexT>, 3>{
                        contact_ids->read(PE(0)),
                        contact_ids->read(PE(1)),
                        contact_ids->read(PE(2))};
                    auto kt2 = lps::detail::mul(
                        lps::detail::mul(PE_kappa(table, table_size, cids), dt), dt);

                    const auto P  = load_vec3(Ps->read(PE(0)));
                    const auto E0 = load_vec3(Ps->read(PE(1)));
                    const auto E1 = load_vec3(Ps->read(PE(2)));

                    auto thickness = PE_thickness(thicknesses->read(PE(0)),
                                                  thicknesses->read(PE(1)),
                                                  thicknesses->read(PE(2)));

                    auto d_hat = PE_d_hat(d_hats->read(PE(0)),
                                          d_hats->read(PE(1)),
                                          d_hats->read(PE(2)));

                    auto flag = ldist::point_edge_distance_flag(P, E0, E1);

                    if constexpr(uipc::RUNTIME_CHECK)
                    {
                        lcrt::Var<Float> D;
                        ldist::point_edge_distance2(flag, P, E0, E1, D);

                        auto range = D_range(thickness, d_hat);

                        lcrt::device_assert(is_active_D(range, D),
                                            "PE d^2 out of range");
                    }

                    Es->write(i, PE_barrier_energy(flag, kt2, d_hat, thickness, P, E0, E1));
                };
            });
    return k;
}

/// `do_compute_energy_k4_kernel` (PP).
auto& do_compute_energy_k4_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector2i>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff>    table,
               lcrt::Var<std::int32_t>          table_size,
               lcrt::BufferVar<IndexT>          contact_ids,
               lcrt::BufferVar<LVector2i>       PPs,
               lcrt::BufferVar<Float>           Es,
               lcrt::BufferVar<LVector3>        Ps,
               lcrt::BufferVar<Float>           thicknesses,
               lcrt::BufferVar<Float>           d_hats,
               lcrt::Var<Float>                 dt,
               lcrt::Var<std::uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    using namespace sym::codim_ipc_simplex_contact;

                    auto PP = load_vec2i(PPs->read(i));

                    auto cids = lps::LVec<lcrt::Var<IndexT>, 2>{contact_ids->read(PP(0)),
                                                                contact_ids->read(PP(1))};
                    auto kt2  = lps::detail::mul(
                        lps::detail::mul(PP_kappa(table, table_size, cids), dt), dt);

                    const auto Pa = load_vec3(Ps->read(PP(0)));
                    const auto Pb = load_vec3(Ps->read(PP(1)));

                    auto thickness =
                        PP_thickness(thicknesses->read(PP(0)), thicknesses->read(PP(1)));

                    auto d_hat = PP_d_hat(d_hats->read(PP(0)), d_hats->read(PP(1)));

                    auto flag = ldist::point_point_distance_flag(Pa, Pb);

                    if constexpr(uipc::RUNTIME_CHECK)
                    {
                        lcrt::Var<Float> D;
                        ldist::point_point_distance2(flag, Pa, Pb, D);

                        auto range = D_range(thickness, d_hat);

                        lcrt::device_assert(is_active_D(range, D),
                                            "PP d^2 out of range");
                    }

                    Es->write(i, PP_barrier_energy(flag, kt2, d_hat, thickness, Pa, Pb));
                };
            });
    return k;
}

/// `do_assemble_PT_kernel<GradientOnly>`: the PT quarter of the reference's **one**
/// fused assemble launch (see this file's header for the four-way split and for its
/// offset contract).
///
/// The body is the reference's PT block verbatim, restricted to PT's own thread range
/// (12x12 gradient/Hessian), and it rebuilds its device views with
/// `from_handles_full` from the reporter subview's `PT_G_offset`/`PT_G_total`/
/// `PT_H_offset`/`PT_H_total` plus the global segment count (`n_segments`), so the
/// four types write into their own slices of the shared buffers instead of all at slot 0.
template <bool GradientOnly>
auto& do_assemble_PT_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   lcrt::Buffer<LVector4i>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff>    table,
               lcrt::Var<std::int32_t>          table_size,
               lcrt::BufferVar<IndexT>          contact_ids,
               lcrt::BufferVar<LVector3>        Ps,
               lcrt::BufferVar<LVector3>        rest_Ps,
               lcrt::BufferVar<Float>           thicknesses,
               lcrt::BufferVar<Float>           d_hats,
               lcrt::Var<Float>                 dt,
               lcrt::BufferVar<LVector4i>       PTs,
               lcrt::BufferVar<IndexT>          PT_G_indices,
               lcrt::BufferVar<Float>           PT_G_values,
               lcrt::Var<std::uint32_t>         PT_G_count,
               lcrt::BufferVar<IndexT>          PT_H_rows,
               lcrt::BufferVar<IndexT>          PT_H_cols,
               lcrt::BufferVar<Float>           PT_H_values,
               lcrt::Var<std::uint32_t>         PT_H_count,
               lcrt::Var<std::uint32_t>         n_segments,
               lcrt::Var<std::uint32_t>         PT_G_offset,
               lcrt::Var<std::uint32_t>         PT_G_total,
               lcrt::Var<std::uint32_t>         PT_H_offset,
               lcrt::Var<std::uint32_t>         PT_H_total,
               lcrt::Var<std::uint32_t>         n) noexcept
            {
                auto idx = lcrt::dispatch_x();
                                using namespace sym::codim_ipc_simplex_contact;
                $if(idx < n)
                {
                    auto i    = lcrt::cast<std::int32_t>(idx);
                    auto PT   = load_vec4i(PTs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 4>{
                        contact_ids->read(PT(0)),
                        contact_ids->read(PT(1)),
                        contact_ids->read(PT(2)),
                        contact_ids->read(PT(3))};
                    auto kt2 = lps::detail::mul(
                        lps::detail::mul(PT_kappa(table, table_size, cids), dt), dt);

                    const auto P  = load_vec3(Ps->read(PT(0)));
                    const auto T0 = load_vec3(Ps->read(PT(1)));
                    const auto T1 = load_vec3(Ps->read(PT(2)));
                    const auto T2 = load_vec3(Ps->read(PT(3)));

                    auto thickness = PT_thickness(thicknesses->read(PT(0)),
                                                  thicknesses->read(PT(1)),
                                                  thicknesses->read(PT(2)),
                                                  thicknesses->read(PT(3)));
                    auto d_hat = PT_d_hat(d_hats->read(PT(0)),
                                          d_hats->read(PT(1)),
                                          d_hats->read(PT(2)),
                                          d_hats->read(PT(3)));
                    auto flag  = ldist::point_triangle_distance_flag(P, T0, T1, T2);

                    auto G = lps::LVec<lcrt::Var<Float>, 12>();
                    auto D_G = lla::LDoubletVectorDeviceView<lcrt::Var<Float>, 3>::
                        from_handles_full(PT_G_indices,
                                          PT_G_values,
                                          n_segments,
                                          PT_G_offset,
                                          PT_G_count,
                                          PT_G_total,
                                          0u,
                                          n_segments);
                    if constexpr(GradientOnly)
                    {
                        PT_barrier_gradient(G, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<4>(i * 4).write(PT, G);
                    }
                    else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 12, 12>();
                        PT_barrier_gradient_hessian(
                            G, H, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                        lps::make_spd_inplace(H, kMakeSpdSweeps);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<4>(i * 4).write(PT, G);
                        auto T_H = lla::LTripletMatrixDeviceView<lcrt::Var<Float>, 3, 3>::
                            from_handles_full(PT_H_rows,
                                              PT_H_cols,
                                              PT_H_values,
                                              n_segments,
                                              n_segments,
                                              PT_H_offset,
                                              PT_H_count,
                                              PT_H_total,
                                              0u,
                                              0u,
                                              n_segments,
                                              n_segments);
                        TripletMatrixAssembler TMA{T_H};
                        TMA.half_block<4>(i * kPTHalfHessianSize)
                            .write(PT, H);
                    }
                }
                ;  // terminates the `$if` expression statement (the reference's block
                   // closed its whole `$if`/`$else` chain with one `;`)
            });
    return k;
}

/// `do_assemble_EE_kernel<GradientOnly>`: the EE quarter of the reference's **one**
/// fused assemble launch (see this file's header for the four-way split and for its
/// offset contract).
///
/// The body is the reference's EE block verbatim, restricted to EE's own thread range
/// (12x12 gradient/Hessian), and it rebuilds its device views with
/// `from_handles_full` from the reporter subview's `EE_G_offset`/`EE_G_total`/
/// `EE_H_offset`/`EE_H_total` plus the global segment count (`n_segments`), so the
/// four types write into their own slices of the shared buffers instead of all at slot 0.
template <bool GradientOnly>
auto& do_assemble_EE_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   lcrt::Buffer<LVector4i>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff>    table,
               lcrt::Var<std::int32_t>          table_size,
               lcrt::BufferVar<IndexT>          contact_ids,
               lcrt::BufferVar<LVector3>        Ps,
               lcrt::BufferVar<LVector3>        rest_Ps,
               lcrt::BufferVar<Float>           thicknesses,
               lcrt::BufferVar<Float>           d_hats,
               lcrt::Var<Float>                 dt,
               lcrt::BufferVar<LVector4i>       EEs,
               lcrt::BufferVar<IndexT>          EE_G_indices,
               lcrt::BufferVar<Float>           EE_G_values,
               lcrt::Var<std::uint32_t>         EE_G_count,
               lcrt::BufferVar<IndexT>          EE_H_rows,
               lcrt::BufferVar<IndexT>          EE_H_cols,
               lcrt::BufferVar<Float>           EE_H_values,
               lcrt::Var<std::uint32_t>         EE_H_count,
               lcrt::Var<std::uint32_t>         n_segments,
               lcrt::Var<std::uint32_t>         EE_G_offset,
               lcrt::Var<std::uint32_t>         EE_G_total,
               lcrt::Var<std::uint32_t>         EE_H_offset,
               lcrt::Var<std::uint32_t>         EE_H_total,
               lcrt::Var<std::uint32_t>         n) noexcept
            {
                auto idx = lcrt::dispatch_x();
                                using namespace sym::codim_ipc_simplex_contact;
                $if(idx < n)
                {
                    auto i    = lcrt::cast<std::int32_t>(idx);
                    auto EE   = load_vec4i(EEs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 4>{
                        contact_ids->read(EE(0)),
                        contact_ids->read(EE(1)),
                        contact_ids->read(EE(2)),
                        contact_ids->read(EE(3))};
                    auto kt2 = lps::detail::mul(
                        lps::detail::mul(EE_kappa(table, table_size, cids), dt), dt);

                    const auto E0     = load_vec3(Ps->read(EE(0)));
                    const auto E1     = load_vec3(Ps->read(EE(1)));
                    const auto E2     = load_vec3(Ps->read(EE(2)));
                    const auto E3     = load_vec3(Ps->read(EE(3)));
                    const auto t0_Ea0 = load_vec3(rest_Ps->read(EE(0)));
                    const auto t0_Ea1 = load_vec3(rest_Ps->read(EE(1)));
                    const auto t0_Eb0 = load_vec3(rest_Ps->read(EE(2)));
                    const auto t0_Eb1 = load_vec3(rest_Ps->read(EE(3)));

                    auto thickness = EE_thickness(thicknesses->read(EE(0)),
                                                  thicknesses->read(EE(1)),
                                                  thicknesses->read(EE(2)),
                                                  thicknesses->read(EE(3)));
                    auto d_hat = EE_d_hat(d_hats->read(EE(0)),
                                          d_hats->read(EE(1)),
                                          d_hats->read(EE(2)),
                                          d_hats->read(EE(3)));
                    auto flag  = ldist::edge_edge_distance_flag(E0, E1, E2, E3);

                    auto G = lps::LVec<lcrt::Var<Float>, 12>();
                    auto D_G = lla::LDoubletVectorDeviceView<lcrt::Var<Float>, 3>::
                        from_handles_full(EE_G_indices,
                                          EE_G_values,
                                          n_segments,
                                          EE_G_offset,
                                          EE_G_count,
                                          EE_G_total,
                                          0u,
                                          n_segments);
                    if constexpr(GradientOnly)
                    {
                        mollified_EE_barrier_gradient(G,
                                                      flag,
                                                      kt2,
                                                      d_hat,
                                                      thickness,
                                                      t0_Ea0,
                                                      t0_Ea1,
                                                      t0_Eb0,
                                                      t0_Eb1,
                                                      E0,
                                                      E1,
                                                      E2,
                                                      E3);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<4>(i * 4).write(EE, G);
                    }
                    else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 12, 12>();
                        mollified_EE_barrier_gradient_hessian(G,
                                                              H,
                                                              flag,
                                                              kt2,
                                                              d_hat,
                                                              thickness,
                                                              t0_Ea0,
                                                              t0_Ea1,
                                                              t0_Eb0,
                                                              t0_Eb1,
                                                              E0,
                                                              E1,
                                                              E2,
                                                              E3);
                        lps::make_spd_inplace(H, kMakeSpdSweeps);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<4>(i * 4).write(EE, G);
                        auto T_H =
                            lla::LTripletMatrixDeviceView<lcrt::Var<Float>, 3, 3>::
                                from_handles_full(EE_H_rows,
                                                  EE_H_cols,
                                                  EE_H_values,
                                                  n_segments,
                                                  n_segments,
                                                  EE_H_offset,
                                                  EE_H_count,
                                                  EE_H_total,
                                                  0u,
                                                  0u,
                                                  n_segments,
                                                  n_segments);
                        TripletMatrixAssembler TMA{T_H};
                        TMA.half_block<4>(i
                                          * kEEHalfHessianSize)
                            .write(EE, H);
                    }
                }
                ;  // terminates the `$if` expression statement (the reference's block
                   // closed its whole `$if`/`$else` chain with one `;`)
            });
    return k;
}

/// `do_assemble_PE_kernel<GradientOnly>`: the PE quarter of the reference's **one**
/// fused assemble launch (see this file's header for the four-way split and for its
/// offset contract).
///
/// The body is the reference's PE block verbatim, restricted to PE's own thread range
/// (9x9 gradient/Hessian), and it rebuilds its device views with
/// `from_handles_full` from the reporter subview's `PE_G_offset`/`PE_G_total`/
/// `PE_H_offset`/`PE_H_total` plus the global segment count (`n_segments`), so the
/// four types write into their own slices of the shared buffers instead of all at slot 0.
template <bool GradientOnly>
auto& do_assemble_PE_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   lcrt::Buffer<LVector3i>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff>    table,
               lcrt::Var<std::int32_t>          table_size,
               lcrt::BufferVar<IndexT>          contact_ids,
               lcrt::BufferVar<LVector3>        Ps,
               lcrt::BufferVar<LVector3>        rest_Ps,
               lcrt::BufferVar<Float>           thicknesses,
               lcrt::BufferVar<Float>           d_hats,
               lcrt::Var<Float>                 dt,
               lcrt::BufferVar<LVector3i>       PEs,
               lcrt::BufferVar<IndexT>          PE_G_indices,
               lcrt::BufferVar<Float>           PE_G_values,
               lcrt::Var<std::uint32_t>         PE_G_count,
               lcrt::BufferVar<IndexT>          PE_H_rows,
               lcrt::BufferVar<IndexT>          PE_H_cols,
               lcrt::BufferVar<Float>           PE_H_values,
               lcrt::Var<std::uint32_t>         PE_H_count,
               lcrt::Var<std::uint32_t>         n_segments,
               lcrt::Var<std::uint32_t>         PE_G_offset,
               lcrt::Var<std::uint32_t>         PE_G_total,
               lcrt::Var<std::uint32_t>         PE_H_offset,
               lcrt::Var<std::uint32_t>         PE_H_total,
               lcrt::Var<std::uint32_t>         n) noexcept
            {
                auto idx = lcrt::dispatch_x();
                                using namespace sym::codim_ipc_simplex_contact;
                $if(idx < n)
                {
                    auto i  = lcrt::cast<std::int32_t>(idx);
                    auto PE = load_vec3i(PEs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 3>{
                        contact_ids->read(PE(0)),
                        contact_ids->read(PE(1)),
                        contact_ids->read(PE(2))};
                    auto kt2 = lps::detail::mul(
                        lps::detail::mul(PE_kappa(table, table_size, cids), dt),
                        dt);

                    const auto P  = load_vec3(Ps->read(PE(0)));
                    const auto E0 = load_vec3(Ps->read(PE(1)));
                    const auto E1 = load_vec3(Ps->read(PE(2)));

                    auto thickness = PE_thickness(thicknesses->read(PE(0)),
                                                  thicknesses->read(PE(1)),
                                                  thicknesses->read(PE(2)));
                    auto d_hat = PE_d_hat(d_hats->read(PE(0)),
                                          d_hats->read(PE(1)),
                                          d_hats->read(PE(2)));
                    auto flag  = ldist::point_edge_distance_flag(P, E0, E1);

                    auto G = lps::LVec<lcrt::Var<Float>, 9>();
                    auto D_G = lla::LDoubletVectorDeviceView<lcrt::Var<Float>, 3>::
                        from_handles_full(PE_G_indices,
                                          PE_G_values,
                                          n_segments,
                                          PE_G_offset,
                                          PE_G_count,
                                          PE_G_total,
                                          0u,
                                          n_segments);
                    if constexpr(GradientOnly)
                    {
                        PE_barrier_gradient(
                            G, flag, kt2, d_hat, thickness, P, E0, E1);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<3>(i * 3).write(PE, G);
                    }
                    else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 9, 9>();
                        PE_barrier_gradient_hessian(
                            G, H, flag, kt2, d_hat, thickness, P, E0, E1);
                        lps::make_spd_inplace(H, kMakeSpdSweeps);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<3>(i * 3).write(PE, G);
                        auto T_H =
                            lla::LTripletMatrixDeviceView<lcrt::Var<Float>, 3, 3>::
                                from_handles_full(PE_H_rows,
                                                  PE_H_cols,
                                                  PE_H_values,
                                                  n_segments,
                                                  n_segments,
                                                  PE_H_offset,
                                                  PE_H_count,
                                                  PE_H_total,
                                                  0u,
                                                  0u,
                                                  n_segments,
                                                  n_segments);
                        TripletMatrixAssembler TMA{T_H};
                        TMA.half_block<3>(
                               i * kPEHalfHessianSize)
                            .write(PE, H);
                    }
                }
                ;  // terminates the `$if` expression statement (the reference's block
                   // closed its whole `$if`/`$else` chain with one `;`)
            });
    return k;
}

/// `do_assemble_PP_kernel<GradientOnly>`: the PP quarter of the reference's **one**
/// fused assemble launch (see this file's header for the four-way split and for its
/// offset contract).
///
/// The body is the reference's PP block verbatim, restricted to PP's own thread range
/// (6x6 gradient/Hessian), and it rebuilds its device views with
/// `from_handles_full` from the reporter subview's `PP_G_offset`/`PP_G_total`/
/// `PP_H_offset`/`PP_H_total` plus the global segment count (`n_segments`), so the
/// four types write into their own slices of the shared buffers instead of all at slot 0.
template <bool GradientOnly>
auto& do_assemble_PP_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   lcrt::Buffer<LVector2i>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff>    table,
               lcrt::Var<std::int32_t>          table_size,
               lcrt::BufferVar<IndexT>          contact_ids,
               lcrt::BufferVar<LVector3>        Ps,
               lcrt::BufferVar<LVector3>        rest_Ps,
               lcrt::BufferVar<Float>           thicknesses,
               lcrt::BufferVar<Float>           d_hats,
               lcrt::Var<Float>                 dt,
               lcrt::BufferVar<LVector2i>       PPs,
               lcrt::BufferVar<IndexT>          PP_G_indices,
               lcrt::BufferVar<Float>           PP_G_values,
               lcrt::Var<std::uint32_t>         PP_G_count,
               lcrt::BufferVar<IndexT>          PP_H_rows,
               lcrt::BufferVar<IndexT>          PP_H_cols,
               lcrt::BufferVar<Float>           PP_H_values,
               lcrt::Var<std::uint32_t>         PP_H_count,
               lcrt::Var<std::uint32_t>         n_segments,
               lcrt::Var<std::uint32_t>         PP_G_offset,
               lcrt::Var<std::uint32_t>         PP_G_total,
               lcrt::Var<std::uint32_t>         PP_H_offset,
               lcrt::Var<std::uint32_t>         PP_H_total,
               lcrt::Var<std::uint32_t>         n) noexcept
            {
                auto idx = lcrt::dispatch_x();
                                using namespace sym::codim_ipc_simplex_contact;
                $if(idx < n)
                {
                    auto i  = lcrt::cast<std::int32_t>(idx);
                    auto PP = load_vec2i(PPs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 2>{
                        contact_ids->read(PP(0)),
                        contact_ids->read(PP(1))};
                    auto kt2 = lps::detail::mul(
                        lps::detail::mul(PP_kappa(table, table_size, cids), dt),
                        dt);

                    const auto P0 = load_vec3(Ps->read(PP(0)));
                    const auto P1 = load_vec3(Ps->read(PP(1)));

                    auto thickness = PP_thickness(thicknesses->read(PP(0)),
                                                  thicknesses->read(PP(1)));
                    auto d_hat = PP_d_hat(d_hats->read(PP(0)),
                                          d_hats->read(PP(1)));
                    auto flag  = ldist::point_point_distance_flag(P0, P1);

                    auto G = lps::LVec<lcrt::Var<Float>, 6>();
                    auto D_G = lla::LDoubletVectorDeviceView<lcrt::Var<Float>, 3>::
                        from_handles_full(PP_G_indices,
                                          PP_G_values,
                                          n_segments,
                                          PP_G_offset,
                                          PP_G_count,
                                          PP_G_total,
                                          0u,
                                          n_segments);
                    if constexpr(GradientOnly)
                    {
                        PP_barrier_gradient(
                            G, flag, kt2, d_hat, thickness, P0, P1);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<2>(i * 2).write(PP, G);
                    }
                    else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 6, 6>();
                        PP_barrier_gradient_hessian(
                            G, H, flag, kt2, d_hat, thickness, P0, P1);
                        lps::make_spd_inplace(H, kMakeSpdSweeps);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<2>(i * 2).write(PP, G);
                        auto T_H =
                            lla::LTripletMatrixDeviceView<lcrt::Var<Float>, 3, 3>::
                                from_handles_full(PP_H_rows,
                                                  PP_H_cols,
                                                  PP_H_values,
                                                  n_segments,
                                                  n_segments,
                                                  PP_H_offset,
                                                  PP_H_count,
                                                  PP_H_total,
                                                  0u,
                                                  0u,
                                                  n_segments,
                                                  n_segments);
                        TripletMatrixAssembler TMA{T_H};
                        TMA.half_block<2>(
                               i * kPPHalfHessianSize)
                            .write(PP, H);
                    }
                };
                ;  // terminates the `$if` expression statement (the reference's block
                   // closed its whole `$if`/`$else` chain with one `;`)
            });
    return k;
}

}  // namespace

class IPCSimplexNormalContact final : public SimplexNormalContact
{
  public:
    using SimplexNormalContact::SimplexNormalContact;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace sym::codim_ipc_simplex_contact;

        auto& stream = frame_stream();

        auto table      = info.contact_tabular();
        auto table_size = static_cast<std::int32_t>(info.contact_tabular_size());

        // Compute Point-Triangle energy
        auto PT_count = info.PTs().size();
        if(PT_count > 0)
        {
            auto& k = do_compute_energy_k1_kernel();
            auto  n = static_cast<std::uint32_t>(PT_count);
            stream.record(k(table,
                            table_size,
                            info.contact_element_ids(),
                            info.PTs(),
                            info.PT_energies(),
                            info.positions(),
                            info.thicknesses(),
                            info.d_hats(),
                            info.dt(),
                            n)
                              .dispatch(n));
        }

        // Compute Edge-Edge energy
        auto EE_count = info.EEs().size();
        if(EE_count > 0)
        {
            auto& k = do_compute_energy_k2_kernel();
            auto  n = static_cast<std::uint32_t>(EE_count);
            stream.record(k(table,
                            table_size,
                            info.contact_element_ids(),
                            info.EEs(),
                            info.EE_energies(),
                            info.positions(),
                            info.thicknesses(),
                            info.rest_positions(),
                            info.d_hats(),
                            info.dt(),
                            n)
                              .dispatch(n));
        }

        // Compute Point-Edge energy
        auto PE_count = info.PEs().size();
        if(PE_count > 0)
        {
            auto& k = do_compute_energy_k3_kernel();
            auto  n = static_cast<std::uint32_t>(PE_count);
            stream.record(k(table,
                            table_size,
                            info.contact_element_ids(),
                            info.PEs(),
                            info.PE_energies(),
                            info.positions(),
                            info.thicknesses(),
                            info.d_hats(),
                            info.dt(),
                            n)
                              .dispatch(n));
        }

        // Compute Point-Point energy
        auto PP_count = info.PPs().size();
        if(PP_count > 0)
        {
            auto& k = do_compute_energy_k4_kernel();
            auto  n = static_cast<std::uint32_t>(PP_count);
            stream.record(k(table,
                            table_size,
                            info.contact_element_ids(),
                            info.PPs(),
                            info.PP_energies(),
                            info.positions(),
                            info.thicknesses(),
                            info.d_hats(),
                            info.dt(),
                            n)
                              .dispatch(n));
        }
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace sym::codim_ipc_simplex_contact;

        auto pt_count = static_cast<IndexT>(info.PTs().size());
        auto ee_count = static_cast<IndexT>(info.EEs().size());
        auto pe_count = static_cast<IndexT>(info.PEs().size());
        auto pp_count = static_cast<IndexT>(info.PPs().size());
        auto total    = pt_count + ee_count + pe_count + pp_count;

        if(total == 0)
            return;

        IndexT ee_offset = pt_count;
        IndexT pe_offset = ee_offset + ee_count;
        IndexT pp_offset = pe_offset + pe_count;
        // The global (all-dof) segment count: the block dimension of every device view
        // built below, `doublet_index_offset + doublet_count <= total_doublet_count`.
        auto n_segments = static_cast<std::uint32_t>(info.PT_gradients().total_extent());

        // The reference's "keep all contact types in one launch" comment, and its
        // `[&]<bool GradientOnly>()` launch lambda specialized on the manager's branch.
        // Each primitive type's gradient/Hessian slice travels as its flat buffers (LC has no
        // pass-by-value view argument -- a view holding buffers cannot be a kernel argument),
        // so the kernel rebuilds the device views with `from_handles_full`, which needs the
        // slice's index offsets and the global counts gathered below (see the kernel note).
        auto PT_G_indices = info.PT_gradients().indices();
        auto PT_G_values  = info.PT_gradients().values();
        auto PT_G_count   = static_cast<std::uint32_t>(info.PT_gradients().doublet_count());
        auto PT_H_rows    = info.PT_hessians().row_indices();
        auto PT_H_cols    = info.PT_hessians().col_indices();
        auto PT_H_values  = info.PT_hessians().values();
        auto PT_H_count   = static_cast<std::uint32_t>(info.PT_hessians().triplet_count());
        // the per-type slice metadata the kernel needs to rebuild its device
        // views with `from_handles_full` (the four types share the launch buffers
        // and differ only in their offsets -- dropping them makes every type write
        // from slot 0, i.e. the last writer wins).
        auto PT_G_offset = static_cast<std::uint32_t>(info.PT_gradients().doublet_index_offset());
        auto PT_G_total  = static_cast<std::uint32_t>(info.PT_gradients().total_doublet_count());
        auto PT_H_offset = static_cast<std::uint32_t>(info.PT_hessians().triplet_index_offset());
        auto PT_H_total  = static_cast<std::uint32_t>(info.PT_hessians().total_triplet_count());

        auto EE_G_indices = info.EE_gradients().indices();
        auto EE_G_values  = info.EE_gradients().values();
        auto EE_G_count   = static_cast<std::uint32_t>(info.EE_gradients().doublet_count());
        auto EE_H_rows    = info.EE_hessians().row_indices();
        auto EE_H_cols    = info.EE_hessians().col_indices();
        auto EE_H_values  = info.EE_hessians().values();
        auto EE_H_count   = static_cast<std::uint32_t>(info.EE_hessians().triplet_count());
        auto EE_G_offset = static_cast<std::uint32_t>(info.EE_gradients().doublet_index_offset());
        auto EE_G_total  = static_cast<std::uint32_t>(info.EE_gradients().total_doublet_count());
        auto EE_H_offset = static_cast<std::uint32_t>(info.EE_hessians().triplet_index_offset());
        auto EE_H_total  = static_cast<std::uint32_t>(info.EE_hessians().total_triplet_count());

        auto PE_G_indices = info.PE_gradients().indices();
        auto PE_G_values  = info.PE_gradients().values();
        auto PE_G_count   = static_cast<std::uint32_t>(info.PE_gradients().doublet_count());
        auto PE_H_rows    = info.PE_hessians().row_indices();
        auto PE_H_cols    = info.PE_hessians().col_indices();
        auto PE_H_values  = info.PE_hessians().values();
        auto PE_H_count   = static_cast<std::uint32_t>(info.PE_hessians().triplet_count());
        auto PE_G_offset = static_cast<std::uint32_t>(info.PE_gradients().doublet_index_offset());
        auto PE_G_total  = static_cast<std::uint32_t>(info.PE_gradients().total_doublet_count());
        auto PE_H_offset = static_cast<std::uint32_t>(info.PE_hessians().triplet_index_offset());
        auto PE_H_total  = static_cast<std::uint32_t>(info.PE_hessians().total_triplet_count());

        auto PP_G_indices = info.PP_gradients().indices();
        auto PP_G_values  = info.PP_gradients().values();
        auto PP_G_count   = static_cast<std::uint32_t>(info.PP_gradients().doublet_count());
        auto PP_H_rows    = info.PP_hessians().row_indices();
        auto PP_H_cols    = info.PP_hessians().col_indices();
        auto PP_H_values  = info.PP_hessians().values();
        auto PP_H_count   = static_cast<std::uint32_t>(info.PP_hessians().triplet_count());
        auto PP_G_offset = static_cast<std::uint32_t>(info.PP_gradients().doublet_index_offset());
        auto PP_G_total  = static_cast<std::uint32_t>(info.PP_gradients().total_doublet_count());
        auto PP_H_offset = static_cast<std::uint32_t>(info.PP_hessians().triplet_index_offset());
        auto PP_H_total  = static_cast<std::uint32_t>(info.PP_hessians().total_triplet_count());

        // Four launches instead of the reference's one: see the per-type kernels above.
        auto launch = [&]<bool GradientOnly>()
        {

            if(pt_count > 0)
            {
                auto& k = do_assemble_PT_kernel<GradientOnly>();
                auto  n = static_cast<std::uint32_t>(pt_count);
                frame_stream().record(
                  k(info.contact_tabular(),
                    static_cast<std::int32_t>(info.contact_tabular_size()),
                    info.contact_element_ids(),
                    info.positions(),
                    info.rest_positions(),
                    info.thicknesses(),
                    info.d_hats(),
                    info.dt(),
                    info.PTs(),
                    PT_G_indices,
                    PT_G_values,
                    PT_G_count,
                    PT_H_rows,
                    PT_H_cols,
                    PT_H_values,
                    PT_H_count,
                    n_segments,
                    PT_G_offset,
                    PT_G_total,
                    PT_H_offset,
                    PT_H_total,
                    n)
                      .dispatch(n));
                // Commit (and therefore *compile*) this kernel on its own: this build's
                // `luisa::compute::Device::compile` deadlocks when two or more of these big
                // `make_spd` kernels are pending in one batch -- measured by bisection: the PT
                // kernel alone runs, PT+EE in one batch never returns (the recorded T6.4a
                // harness hazard; cookbook 7.36).
                frame_stream().synchronize();
            }

            if(ee_count > 0)
            {
                auto& k = do_assemble_EE_kernel<GradientOnly>();
                auto  n = static_cast<std::uint32_t>(ee_count);
                frame_stream().record(
                  k(info.contact_tabular(),
                    static_cast<std::int32_t>(info.contact_tabular_size()),
                    info.contact_element_ids(),
                    info.positions(),
                    info.rest_positions(),
                    info.thicknesses(),
                    info.d_hats(),
                    info.dt(),
                    info.EEs(),
                    EE_G_indices,
                    EE_G_values,
                    EE_G_count,
                    EE_H_rows,
                    EE_H_cols,
                    EE_H_values,
                    EE_H_count,
                    n_segments,
                    EE_G_offset,
                    EE_G_total,
                    EE_H_offset,
                    EE_H_total,
                    n)
                      .dispatch(n));
                // Commit (and therefore *compile*) this kernel on its own: this build's
                // `luisa::compute::Device::compile` deadlocks when two or more of these big
                // `make_spd` kernels are pending in one batch -- measured by bisection: the PT
                // kernel alone runs, PT+EE in one batch never returns (the recorded T6.4a
                // harness hazard; cookbook 7.36).
                frame_stream().synchronize();
            }

            if(pe_count > 0)
            {
                auto& k = do_assemble_PE_kernel<GradientOnly>();
                auto  n = static_cast<std::uint32_t>(pe_count);
                frame_stream().record(
                  k(info.contact_tabular(),
                    static_cast<std::int32_t>(info.contact_tabular_size()),
                    info.contact_element_ids(),
                    info.positions(),
                    info.rest_positions(),
                    info.thicknesses(),
                    info.d_hats(),
                    info.dt(),
                    info.PEs(),
                    PE_G_indices,
                    PE_G_values,
                    PE_G_count,
                    PE_H_rows,
                    PE_H_cols,
                    PE_H_values,
                    PE_H_count,
                    n_segments,
                    PE_G_offset,
                    PE_G_total,
                    PE_H_offset,
                    PE_H_total,
                    n)
                      .dispatch(n));
                // Commit (and therefore *compile*) this kernel on its own: this build's
                // `luisa::compute::Device::compile` deadlocks when two or more of these big
                // `make_spd` kernels are pending in one batch -- measured by bisection: the PT
                // kernel alone runs, PT+EE in one batch never returns (the recorded T6.4a
                // harness hazard; cookbook 7.36).
                frame_stream().synchronize();
            }

            if(pp_count > 0)
            {
                auto& k = do_assemble_PP_kernel<GradientOnly>();
                auto  n = static_cast<std::uint32_t>(pp_count);
                frame_stream().record(
                  k(info.contact_tabular(),
                    static_cast<std::int32_t>(info.contact_tabular_size()),
                    info.contact_element_ids(),
                    info.positions(),
                    info.rest_positions(),
                    info.thicknesses(),
                    info.d_hats(),
                    info.dt(),
                    info.PPs(),
                    PP_G_indices,
                    PP_G_values,
                    PP_G_count,
                    PP_H_rows,
                    PP_H_cols,
                    PP_H_values,
                    PP_H_count,
                    n_segments,
                    PP_G_offset,
                    PP_G_total,
                    PP_H_offset,
                    PP_H_total,
                    n)
                      .dispatch(n));
                // Commit (and therefore *compile*) this kernel on its own: this build's
                // `luisa::compute::Device::compile` deadlocks when two or more of these big
                // `make_spd` kernels are pending in one batch -- measured by bisection: the PT
                // kernel alone runs, PT+EE in one batch never returns (the recorded T6.4a
                // harness hazard; cookbook 7.36).
                frame_stream().synchronize();
            }

        };

        if(info.gradient_only())
            launch.operator()<true>();
        else
            launch.operator()<false>();
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexNormalContact);
}  // namespace uipc::backend::luisa
