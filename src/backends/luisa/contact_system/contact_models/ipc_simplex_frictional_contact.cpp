/********************************************************************
 * @file contact_models/ipc_simplex_frictional_contact.cpp
 * @brief T8.1b — `IPCSimplexFrictionalContact`: the concrete IPC simplex frictional contact
 * model (port of
 * src/backends/cuda/contact_system/contact_models/ipc_simplex_frictional_contact.cu, 643
 * reference lines).
 *
 * The frictional twin of ipc_simplex_normal_contact.cpp: the same four per-type energy kernels
 * and the same four per-type assemble kernels (`do_assemble_{PT,EE,PE,PP}_kernel<
 * GradientOnly>`, the four-way split the normal model's header documents), but each energy is
 * the *friction* law of `codim_ipc_simplex_frictional_contact_function.h`
 * (`friction_energy(mu, f, eps_velocity * dt, tan_rel_dx)`, with the previous-step projection
 * `*_friction_basis` and the normal force `f = normal_force(kappa, d_hat, thickness, prev_D)`),
 * and the candidates are the filters' `friction_*` arrays.  Three reference details are
 * transcribed exactly:
 *  * the **EE** pair is skipped (energy 0, gradient/Hessian 0) when the *previous* configuration
 * is inside the mollified (near-parallel) region — `distance::need_mollify(prev_Ea0,
 * prev_Ea1, prev_Eb0, prev_Eb1, eps_x)`, a *device* predicate, so the port spells the
 * reference's `if/else` as `$if/$else` (an LC branch, not host control flow);
 *  * the barrier `kappa` and the friction `mu` both come from the *contact table*:
 * `PT_contact_coeff(kappa, mu, table, extent, cids)` (the reference's
 * `auto coeff = PT_contact_coeff(table, cids); kt2 = coeff.kappa * dt * dt; mu = coeff.mu;`);
 *  * `kt2 = kappa * dt * dt` and `eps_vh = eps_velocity * dt` are the reference's own scalings
 * (the "eps_velocity" the manager exposes is a *rate*).
 * Everything else (the flat-table mapping, `load_vec*`, `ldist::*_flag`,
 * `lps::make_spd_inplace(H, kMakeSpdSweeps)`, the device views rebuilt with
 * `from_handles_full` + the slice metadata, and the in-kernel assemblers) is shared with the
 * normal model — see that file's header for the argument.
 *********************************************************************/
#include <contact_system/simplex_frictional_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
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

/// See ipc_simplex_normal_contact.cpp: the reference's tolerance-driven `utils/make_spd.h`
/// loop is replaced by the port's fixed-sweep `lps::make_spd_inplace`.
inline constexpr int kMakeSpdSweeps = 8;

constexpr std::int32_t kPTHalfHessianSize =
    static_cast<std::int32_t>(SimplexFrictionalContact::PTHalfHessianSize);
constexpr std::int32_t kEEHalfHessianSize =
    static_cast<std::int32_t>(SimplexFrictionalContact::EEHalfHessianSize);
constexpr std::int32_t kPEHalfHessianSize =
    static_cast<std::int32_t>(SimplexFrictionalContact::PEHalfHessianSize);
constexpr std::int32_t kPPHalfHessianSize =
    static_cast<std::int32_t>(SimplexFrictionalContact::PPHalfHessianSize);

/// `do_compute_energy_k1_kernel` (PT friction).
auto& do_compute_energy_k1_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector4i>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector4i>    PTs,
               lcrt::BufferVar<Float>        Es,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              eps_v,
               lcrt::Var<Float>              dt,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    using namespace sym::codim_ipc_contact;

                    const auto PT = load_vec4i(PTs->read(i));

                    auto cids = lps::LVec<lcrt::Var<IndexT>, 4>{contact_ids->read(PT(0)),
                                                                contact_ids->read(PT(1)),
                                                                contact_ids->read(PT(2)),
                                                                contact_ids->read(PT(3))};

                    lcrt::Var<Float> kappa;
                    lcrt::Var<Float> mu;
                    PT_contact_coeff(kappa, mu, table, table_size, cids);
                    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);

                    const auto prev_P  = load_vec3(prev_Ps->read(PT(0)));
                    const auto prev_T0 = load_vec3(prev_Ps->read(PT(1)));
                    const auto prev_T1 = load_vec3(prev_Ps->read(PT(2)));
                    const auto prev_T2 = load_vec3(prev_Ps->read(PT(3)));

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

                    Es->write(i,
                              PT_friction_energy(kt2,
                                                 d_hat,
                                                 thickness,
                                                 mu,
                                                 lps::detail::mul(eps_v, dt),
                                                 // previous positions
                                                 prev_P,
                                                 prev_T0,
                                                 prev_T1,
                                                 prev_T2,
                                                 // current positions
                                                 P,
                                                 T0,
                                                 T1,
                                                 T2));
                };
            });
    return k;
}

/// `do_compute_energy_k2_kernel` (EE friction, with the previous-configuration mollifier
/// skip).
auto& do_compute_energy_k2_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector4i>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   Float,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector4i>    EEs,
               lcrt::BufferVar<Float>        Es,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<LVector3>     rest_Ps,
               lcrt::Var<Float>              eps_v,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              dt,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    using namespace sym::codim_ipc_contact;

                    const auto EE = load_vec4i(EEs->read(i));

                    auto cids = lps::LVec<lcrt::Var<IndexT>, 4>{contact_ids->read(EE(0)),
                                                                contact_ids->read(EE(1)),
                                                                contact_ids->read(EE(2)),
                                                                contact_ids->read(EE(3))};

                    lcrt::Var<Float> kappa;
                    lcrt::Var<Float> mu;
                    EE_contact_coeff(kappa, mu, table, table_size, cids);
                    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);

                    const auto rest_Ea0 = load_vec3(rest_Ps->read(EE(0)));
                    const auto rest_Ea1 = load_vec3(rest_Ps->read(EE(1)));
                    const auto rest_Eb0 = load_vec3(rest_Ps->read(EE(2)));
                    const auto rest_Eb1 = load_vec3(rest_Ps->read(EE(3)));

                    const auto prev_Ea0 = load_vec3(prev_Ps->read(EE(0)));
                    const auto prev_Ea1 = load_vec3(prev_Ps->read(EE(1)));
                    const auto prev_Eb0 = load_vec3(prev_Ps->read(EE(2)));
                    const auto prev_Eb1 = load_vec3(prev_Ps->read(EE(3)));

                    const auto Ea0 = load_vec3(Ps->read(EE(0)));
                    const auto Ea1 = load_vec3(Ps->read(EE(1)));
                    const auto Eb0 = load_vec3(Ps->read(EE(2)));
                    const auto Eb1 = load_vec3(Ps->read(EE(3)));

                    auto thickness = EE_thickness(thicknesses->read(EE(0)),
                                                  thicknesses->read(EE(1)),
                                                  thicknesses->read(EE(2)),
                                                  thicknesses->read(EE(3)));
                    auto d_hat = EE_d_hat(d_hats->read(EE(0)),
                                          d_hats->read(EE(1)),
                                          d_hats->read(EE(2)),
                                          d_hats->read(EE(3)));

                    lcrt::Var<Float> eps_x;
                    ldist::edge_edge_mollifier_threshold(rest_Ea0,
                                                        rest_Ea1,
                                                        rest_Eb0,
                                                        rest_Eb1,
                                                        static_cast<double>(1e-3),
                                                        eps_x);
                    auto mollified =
                        ldist::need_mollify(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, eps_x);
                    // almost parallel, don't compute energy
                    $if(mollified)
                    {
                        Es->write(i, Float{0.0});
                    }
                    $else
                    {
                        Es->write(i,
                                  EE_friction_energy(kt2,
                                                     d_hat,
                                                     thickness,
                                                     mu,
                                                     lps::detail::mul(eps_v, dt),
                                                     // previous positions
                                                     prev_Ea0,
                                                     prev_Ea1,
                                                     prev_Eb0,
                                                     prev_Eb1,
                                                     // current positions
                                                     Ea0,
                                                     Ea1,
                                                     Eb0,
                                                     Eb1));
                    };
                };
            });
    return k;
}

/// `do_compute_energy_k3_kernel` (PE friction).
auto& do_compute_energy_k3_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector3i>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector3i>    PEs,
               lcrt::BufferVar<Float>        Es,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              eps_v,
               lcrt::Var<Float>              dt,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    using namespace sym::codim_ipc_contact;

                    const auto PE = load_vec3i(PEs->read(i));

                    auto cids = lps::LVec<lcrt::Var<IndexT>, 3>{contact_ids->read(PE(0)),
                                                                contact_ids->read(PE(1)),
                                                                contact_ids->read(PE(2))};

                    lcrt::Var<Float> kappa;
                    lcrt::Var<Float> mu;
                    PE_contact_coeff(kappa, mu, table, table_size, cids);
                    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);

                    const auto prev_P  = load_vec3(prev_Ps->read(PE(0)));
                    const auto prev_E0 = load_vec3(prev_Ps->read(PE(1)));
                    const auto prev_E1 = load_vec3(prev_Ps->read(PE(2)));

                    const auto P  = load_vec3(Ps->read(PE(0)));
                    const auto E0 = load_vec3(Ps->read(PE(1)));
                    const auto E1 = load_vec3(Ps->read(PE(2)));

                    auto thickness = PE_thickness(thicknesses->read(PE(0)),
                                                  thicknesses->read(PE(1)),
                                                  thicknesses->read(PE(2)));
                    auto d_hat = PE_d_hat(d_hats->read(PE(0)),
                                          d_hats->read(PE(1)),
                                          d_hats->read(PE(2)));

                    Es->write(i,
                              PE_friction_energy(kt2,
                                                 d_hat,
                                                 thickness,
                                                 mu,
                                                 lps::detail::mul(eps_v, dt),
                                                 // previous positions
                                                 prev_P,
                                                 prev_E0,
                                                 prev_E1,
                                                 // current positions
                                                 P,
                                                 E0,
                                                 E1));
                };
            });
    return k;
}

/// `do_compute_energy_k4_kernel` (PP friction).
auto& do_compute_energy_k4_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector2i>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector2i>    PPs,
               lcrt::BufferVar<Float>        Es,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              eps_v,
               lcrt::Var<Float>              dt,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    using namespace sym::codim_ipc_contact;

                    const auto PP = load_vec2i(PPs->read(i));

                    auto cids = lps::LVec<lcrt::Var<IndexT>, 2>{contact_ids->read(PP(0)),
                                                                contact_ids->read(PP(1))};
                    lcrt::Var<Float> kappa;
                    lcrt::Var<Float> mu;
                    PP_contact_coeff(kappa, mu, table, table_size, cids);
                    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);

                    const auto prev_P0 = load_vec3(prev_Ps->read(PP(0)));
                    const auto prev_P1 = load_vec3(prev_Ps->read(PP(1)));

                    const auto P0 = load_vec3(Ps->read(PP(0)));
                    const auto P1 = load_vec3(Ps->read(PP(1)));

                    auto thickness =
                        PP_thickness(thicknesses->read(PP(0)), thicknesses->read(PP(1)));
                    auto d_hat = PP_d_hat(d_hats->read(PP(0)), d_hats->read(PP(1)));

                    Es->write(i,
                              PP_friction_energy(kt2,
                                                 d_hat,
                                                 thickness,
                                                 mu,
                                                 lps::detail::mul(eps_v, dt),
                                                 // previous positions
                                                 prev_P0,
                                                 prev_P1,
                                                 // current positions
                                                 P0,
                                                 P1));
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
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
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
            [](lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<LVector3>     rest_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              eps_v,
               lcrt::Var<Float>              dt,
               lcrt::BufferVar<LVector4i>    PTs,
               lcrt::BufferVar<IndexT>       PT_G_indices,
               lcrt::BufferVar<Float>        PT_G_values,
               lcrt::Var<std::uint32_t>      PT_G_count,
               lcrt::BufferVar<IndexT>       PT_H_rows,
               lcrt::BufferVar<IndexT>       PT_H_cols,
               lcrt::BufferVar<Float>        PT_H_values,
               lcrt::Var<std::uint32_t>      PT_H_count,
               lcrt::Var<std::uint32_t>      n_segments,
               lcrt::Var<std::uint32_t>      PT_G_offset,
               lcrt::Var<std::uint32_t>      PT_G_total,
               lcrt::Var<std::uint32_t>      PT_H_offset,
               lcrt::Var<std::uint32_t>      PT_H_total,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto idx = lcrt::dispatch_x();
                                using namespace sym::codim_ipc_contact;
                $if(idx < n)
                {
                    // PT friction
                    auto i = lcrt::cast<std::int32_t>(idx);
                    const auto PT = load_vec4i(PTs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 4>{
                        contact_ids->read(PT(0)),
                        contact_ids->read(PT(1)),
                        contact_ids->read(PT(2)),
                        contact_ids->read(PT(3))};
                    lcrt::Var<Float> kappa;
                    lcrt::Var<Float> mu;
                    PT_contact_coeff(kappa, mu, table, table_size, cids);
                    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);

                    const auto prev_P  = load_vec3(prev_Ps->read(PT(0)));
                    const auto prev_T0 = load_vec3(prev_Ps->read(PT(1)));
                    const auto prev_T1 = load_vec3(prev_Ps->read(PT(2)));
                    const auto prev_T2 = load_vec3(prev_Ps->read(PT(3)));
                    const auto P       = load_vec3(Ps->read(PT(0)));
                    const auto T0      = load_vec3(Ps->read(PT(1)));
                    const auto T1      = load_vec3(Ps->read(PT(2)));
                    const auto T2      = load_vec3(Ps->read(PT(3)));

                    auto thickness = PT_thickness(thicknesses->read(PT(0)),
                                                  thicknesses->read(PT(1)),
                                                  thicknesses->read(PT(2)),
                                                  thicknesses->read(PT(3)));
                    auto d_hat = PT_d_hat(d_hats->read(PT(0)),
                                          d_hats->read(PT(1)),
                                          d_hats->read(PT(2)),
                                          d_hats->read(PT(3)));

                    auto G   = lps::LVec<lcrt::Var<Float>, 12>();
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
                        PT_friction_gradient(G,
                                             kt2,
                                             d_hat,
                                             thickness,
                                             mu,
                                             lps::detail::mul(eps_v, dt),
                                             prev_P,
                                             prev_T0,
                                             prev_T1,
                                             prev_T2,
                                             P,
                                             T0,
                                             T1,
                                             T2);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<4>(i * 4).write(PT, G);
                    }
                    else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 12, 12>();
                        PT_friction_gradient_hessian(G,
                                                     H,
                                                     kt2,
                                                     d_hat,
                                                     thickness,
                                                     mu,
                                                     lps::detail::mul(eps_v, dt),
                                                     prev_P,
                                                     prev_T0,
                                                     prev_T1,
                                                     prev_T2,
                                                     P,
                                                     T0,
                                                     T1,
                                                     T2);
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
                        TMA.half_block<4>(i * kPTHalfHessianSize).write(PT, H);
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
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
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
            [](lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<LVector3>     rest_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              eps_v,
               lcrt::Var<Float>              dt,
               lcrt::BufferVar<LVector4i>    EEs,
               lcrt::BufferVar<IndexT>       EE_G_indices,
               lcrt::BufferVar<Float>        EE_G_values,
               lcrt::Var<std::uint32_t>      EE_G_count,
               lcrt::BufferVar<IndexT>       EE_H_rows,
               lcrt::BufferVar<IndexT>       EE_H_cols,
               lcrt::BufferVar<Float>        EE_H_values,
               lcrt::Var<std::uint32_t>      EE_H_count,
               lcrt::Var<std::uint32_t>      n_segments,
               lcrt::Var<std::uint32_t>      EE_G_offset,
               lcrt::Var<std::uint32_t>      EE_G_total,
               lcrt::Var<std::uint32_t>      EE_H_offset,
               lcrt::Var<std::uint32_t>      EE_H_total,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto idx = lcrt::dispatch_x();
                                using namespace sym::codim_ipc_contact;
                $if(idx < n)
                {
                    // EE friction
                    auto i = lcrt::cast<std::int32_t>(idx);
                    const auto EE = load_vec4i(EEs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 4>{
                        contact_ids->read(EE(0)),
                        contact_ids->read(EE(1)),
                        contact_ids->read(EE(2)),
                        contact_ids->read(EE(3))};
                    lcrt::Var<Float> kappa;
                    lcrt::Var<Float> mu;
                    EE_contact_coeff(kappa, mu, table, table_size, cids);
                    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);

                    const auto rest_Ea0 = load_vec3(rest_Ps->read(EE(0)));
                    const auto rest_Ea1 = load_vec3(rest_Ps->read(EE(1)));
                    const auto rest_Eb0 = load_vec3(rest_Ps->read(EE(2)));
                    const auto rest_Eb1 = load_vec3(rest_Ps->read(EE(3)));
                    const auto prev_Ea0 = load_vec3(prev_Ps->read(EE(0)));
                    const auto prev_Ea1 = load_vec3(prev_Ps->read(EE(1)));
                    const auto prev_Eb0 = load_vec3(prev_Ps->read(EE(2)));
                    const auto prev_Eb1 = load_vec3(prev_Ps->read(EE(3)));
                    const auto Ea0      = load_vec3(Ps->read(EE(0)));
                    const auto Ea1      = load_vec3(Ps->read(EE(1)));
                    const auto Eb0      = load_vec3(Ps->read(EE(2)));
                    const auto Eb1      = load_vec3(Ps->read(EE(3)));

                    auto thickness = EE_thickness(thicknesses->read(EE(0)),
                                                  thicknesses->read(EE(1)),
                                                  thicknesses->read(EE(2)),
                                                  thicknesses->read(EE(3)));
                    auto d_hat = EE_d_hat(d_hats->read(EE(0)),
                                          d_hats->read(EE(1)),
                                          d_hats->read(EE(2)),
                                          d_hats->read(EE(3)));

                    lcrt::Var<Float> eps_x;
                    ldist::edge_edge_mollifier_threshold(rest_Ea0,
                                                        rest_Ea1,
                                                        rest_Eb0,
                                                        rest_Eb1,
                                                        static_cast<double>(1e-3),
                                                        eps_x);
                    auto mollified = ldist::need_mollify(
                        prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, eps_x);

                    auto G   = lps::LVec<lcrt::Var<Float>, 12>();
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
                        $if(mollified)
                        {
                            G.setZero();
                        }
                        $else
                        {
                            EE_friction_gradient(G,
                                                 kt2,
                                                 d_hat,
                                                 thickness,
                                                 mu,
                                                 lps::detail::mul(eps_v, dt),
                                                 prev_Ea0,
                                                 prev_Ea1,
                                                 prev_Eb0,
                                                 prev_Eb1,
                                                 Ea0,
                                                 Ea1,
                                                 Eb0,
                                                 Eb1);
                        };
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<4>(i * 4).write(EE, G);
                    }
                    else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 12, 12>();
                        $if(mollified)
                        {
                            G.setZero();
                            H.setZero();
                        }
                        $else
                        {
                            EE_friction_gradient_hessian(G,
                                                         H,
                                                         kt2,
                                                         d_hat,
                                                         thickness,
                                                         mu,
                                                         lps::detail::mul(eps_v, dt),
                                                         prev_Ea0,
                                                         prev_Ea1,
                                                         prev_Eb0,
                                                         prev_Eb1,
                                                         Ea0,
                                                         Ea1,
                                                         Eb0,
                                                         Eb1);
                            lps::make_spd_inplace(H, kMakeSpdSweeps);
                        };
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
                        TMA.half_block<4>(i * kEEHalfHessianSize).write(EE, H);
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
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
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
            [](lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<LVector3>     rest_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              eps_v,
               lcrt::Var<Float>              dt,
               lcrt::BufferVar<LVector3i>    PEs,
               lcrt::BufferVar<IndexT>       PE_G_indices,
               lcrt::BufferVar<Float>        PE_G_values,
               lcrt::Var<std::uint32_t>      PE_G_count,
               lcrt::BufferVar<IndexT>       PE_H_rows,
               lcrt::BufferVar<IndexT>       PE_H_cols,
               lcrt::BufferVar<Float>        PE_H_values,
               lcrt::Var<std::uint32_t>      PE_H_count,
               lcrt::Var<std::uint32_t>      n_segments,
               lcrt::Var<std::uint32_t>      PE_G_offset,
               lcrt::Var<std::uint32_t>      PE_G_total,
               lcrt::Var<std::uint32_t>      PE_H_offset,
               lcrt::Var<std::uint32_t>      PE_H_total,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto idx = lcrt::dispatch_x();
                                using namespace sym::codim_ipc_contact;
                $if(idx < n)
                {
                    // PE friction
                    auto i = lcrt::cast<std::int32_t>(idx);
                    const auto PE = load_vec3i(PEs->read(i));
                    auto cids     = lps::LVec<lcrt::Var<IndexT>, 3>{
                        contact_ids->read(PE(0)),
                        contact_ids->read(PE(1)),
                        contact_ids->read(PE(2))};
                    lcrt::Var<Float> kappa;
                    lcrt::Var<Float> mu;
                    PE_contact_coeff(kappa, mu, table, table_size, cids);
                    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);

                    const auto prev_P  = load_vec3(prev_Ps->read(PE(0)));
                    const auto prev_E0 = load_vec3(prev_Ps->read(PE(1)));
                    const auto prev_E1 = load_vec3(prev_Ps->read(PE(2)));
                    const auto P       = load_vec3(Ps->read(PE(0)));
                    const auto E0      = load_vec3(Ps->read(PE(1)));
                    const auto E1      = load_vec3(Ps->read(PE(2)));

                    auto thickness = PE_thickness(thicknesses->read(PE(0)),
                                                  thicknesses->read(PE(1)),
                                                  thicknesses->read(PE(2)));
                    auto d_hat = PE_d_hat(d_hats->read(PE(0)),
                                          d_hats->read(PE(1)),
                                          d_hats->read(PE(2)));

                    auto G   = lps::LVec<lcrt::Var<Float>, 9>();
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
                        PE_friction_gradient(G,
                                             kt2,
                                             d_hat,
                                             thickness,
                                             mu,
                                             lps::detail::mul(eps_v, dt),
                                             prev_P,
                                             prev_E0,
                                             prev_E1,
                                             P,
                                             E0,
                                             E1);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<3>(i * 3).write(PE, G);
                    }
                    else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 9, 9>();
                        PE_friction_gradient_hessian(G,
                                                     H,
                                                     kt2,
                                                     d_hat,
                                                     thickness,
                                                     mu,
                                                     lps::detail::mul(eps_v, dt),
                                                     prev_P,
                                                     prev_E0,
                                                     prev_E1,
                                                     P,
                                                     E0,
                                                     E1);
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
                        TMA.half_block<3>(i * kPEHalfHessianSize).write(PE, H);
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
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   Float,
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
            [](lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<LVector3>     rest_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              eps_v,
               lcrt::Var<Float>              dt,
               lcrt::BufferVar<LVector2i>    PPs,
               lcrt::BufferVar<IndexT>       PP_G_indices,
               lcrt::BufferVar<Float>        PP_G_values,
               lcrt::Var<std::uint32_t>      PP_G_count,
               lcrt::BufferVar<IndexT>       PP_H_rows,
               lcrt::BufferVar<IndexT>       PP_H_cols,
               lcrt::BufferVar<Float>        PP_H_values,
               lcrt::Var<std::uint32_t>      PP_H_count,
               lcrt::Var<std::uint32_t>      n_segments,
               lcrt::Var<std::uint32_t>      PP_G_offset,
               lcrt::Var<std::uint32_t>      PP_G_total,
               lcrt::Var<std::uint32_t>      PP_H_offset,
               lcrt::Var<std::uint32_t>      PP_H_total,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto idx = lcrt::dispatch_x();
                                using namespace sym::codim_ipc_contact;
                $if(idx < n)
                {
                    // PP friction
                    auto i = lcrt::cast<std::int32_t>(idx);
                    const auto PP = load_vec2i(PPs->read(i));
                    auto cids = lps::LVec<lcrt::Var<IndexT>, 2>{
                        contact_ids->read(PP(0)),
                        contact_ids->read(PP(1))};
                    lcrt::Var<Float> kappa;
                    lcrt::Var<Float> mu;
                    PP_contact_coeff(kappa, mu, table, table_size, cids);
                    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);

                    const auto prev_P0 = load_vec3(prev_Ps->read(PP(0)));
                    const auto prev_P1 = load_vec3(prev_Ps->read(PP(1)));
                    const auto P0      = load_vec3(Ps->read(PP(0)));
                    const auto P1      = load_vec3(Ps->read(PP(1)));

                    auto thickness = PP_thickness(thicknesses->read(PP(0)),
                                                  thicknesses->read(PP(1)));
                    auto d_hat =
                        PP_d_hat(d_hats->read(PP(0)), d_hats->read(PP(1)));

                    auto G   = lps::LVec<lcrt::Var<Float>, 6>();
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
                        PP_friction_gradient(G,
                                             kt2,
                                             d_hat,
                                             thickness,
                                             mu,
                                             lps::detail::mul(eps_v, dt),
                                             prev_P0,
                                             prev_P1,
                                             P0,
                                             P1);
                        DoubletVectorAssembler DVA{D_G};
                        DVA.segment<2>(i * 2).write(PP, G);
                    }
                    else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 6, 6>();
                        PP_friction_gradient_hessian(G,
                                                     H,
                                                     kt2,
                                                     d_hat,
                                                     thickness,
                                                     mu,
                                                     lps::detail::mul(eps_v, dt),
                                                     prev_P0,
                                                     prev_P1,
                                                     P0,
                                                     P1);
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
                        TMA.half_block<2>(i * kPPHalfHessianSize).write(PP, H);
                    }
                };
                ;  // terminates the `$if` expression statement (the reference's block
                   // closed its whole `$if`/`$else` chain with one `;`)
            });
    return k;
}

}  // namespace

class IPCSimplexFrictionalContact final : public SimplexFrictionalContact
{
  public:
    using SimplexFrictionalContact::SimplexFrictionalContact;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace sym::codim_ipc_contact;

        auto& stream = frame_stream();

        auto table      = info.contact_tabular();
        auto table_size = static_cast<std::int32_t>(info.contact_tabular_size());

        // Compute Point-Triangle energy
        auto PT_count = info.friction_PTs().size();
        if(PT_count > 0)
        {
            auto& k = do_compute_energy_k1_kernel();
            auto  n = static_cast<std::uint32_t>(PT_count);
            stream.record(k(table,
                            table_size,
                            info.contact_element_ids(),
                            info.friction_PTs(),
                            info.friction_PT_energies(),
                            info.positions(),
                            info.prev_positions(),
                            info.thicknesses(),
                            info.d_hats(),
                            info.eps_velocity(),
                            info.dt(),
                            n)
                              .dispatch(n));
        }

        // Compute Edge-Edge energy
        auto EE_count = info.friction_EEs().size();
        if(EE_count > 0)
        {
            auto& k = do_compute_energy_k2_kernel();
            auto  n = static_cast<std::uint32_t>(EE_count);
            stream.record(k(table,
                            table_size,
                            info.contact_element_ids(),
                            info.friction_EEs(),
                            info.friction_EE_energies(),
                            info.positions(),
                            info.prev_positions(),
                            info.rest_positions(),
                            info.eps_velocity(),
                            info.thicknesses(),
                            info.d_hats(),
                            info.dt(),
                            n)
                              .dispatch(n));
        }

        // Compute Point-Edge energy
        auto PE_count = info.friction_PEs().size();
        if(PE_count > 0)
        {
            auto& k = do_compute_energy_k3_kernel();
            auto  n = static_cast<std::uint32_t>(PE_count);
            stream.record(k(table,
                            table_size,
                            info.contact_element_ids(),
                            info.friction_PEs(),
                            info.friction_PE_energies(),
                            info.positions(),
                            info.prev_positions(),
                            info.thicknesses(),
                            info.d_hats(),
                            info.eps_velocity(),
                            info.dt(),
                            n)
                              .dispatch(n));
        }

        // Compute Point-Point energy
        auto PP_count = info.friction_PPs().size();
        if(PP_count > 0)
        {
            auto& k = do_compute_energy_k4_kernel();
            auto  n = static_cast<std::uint32_t>(PP_count);
            stream.record(k(table,
                            table_size,
                            info.contact_element_ids(),
                            info.friction_PPs(),
                            info.friction_PP_energies(),
                            info.positions(),
                            info.prev_positions(),
                            info.thicknesses(),
                            info.d_hats(),
                            info.eps_velocity(),
                            info.dt(),
                            n)
                              .dispatch(n));
        }
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace sym::codim_ipc_contact;

        auto pt_count = static_cast<IndexT>(info.friction_PTs().size());
        auto ee_count = static_cast<IndexT>(info.friction_EEs().size());
        auto pe_count = static_cast<IndexT>(info.friction_PEs().size());
        auto pp_count = static_cast<IndexT>(info.friction_PPs().size());
        auto total    = pt_count + ee_count + pe_count + pp_count;

        if(total == 0)
            return;

        IndexT ee_offset = pt_count;
        IndexT pe_offset = ee_offset + ee_count;
        IndexT pp_offset = pe_offset + pe_count;
        // The global (all-dof) segment count: the block dimension of every device view
        // built below, `doublet_index_offset + doublet_count <= total_doublet_count`.
        auto n_segments = static_cast<std::uint32_t>(info.friction_PT_gradients().total_extent());

        auto PT_G_indices = info.friction_PT_gradients().indices();
        auto PT_G_values  = info.friction_PT_gradients().values();
        auto PT_G_count = static_cast<std::uint32_t>(info.friction_PT_gradients().doublet_count());
        auto PT_H_rows    = info.friction_PT_hessians().row_indices();
        auto PT_H_cols    = info.friction_PT_hessians().col_indices();
        auto PT_H_values  = info.friction_PT_hessians().values();
        auto PT_H_count = static_cast<std::uint32_t>(info.friction_PT_hessians().triplet_count());
        // the per-type slice metadata the kernel needs to rebuild its device
        // views with `from_handles_full` (the four types share the launch buffers
        // and differ only in their offsets -- dropping them makes every type write
        // from slot 0, i.e. the last writer wins).
        auto PT_G_offset = static_cast<std::uint32_t>(info.friction_PT_gradients().doublet_index_offset());
        auto PT_G_total  = static_cast<std::uint32_t>(info.friction_PT_gradients().total_doublet_count());
        auto PT_H_offset = static_cast<std::uint32_t>(info.friction_PT_hessians().triplet_index_offset());
        auto PT_H_total  = static_cast<std::uint32_t>(info.friction_PT_hessians().total_triplet_count());

        auto EE_G_indices = info.friction_EE_gradients().indices();
        auto EE_G_values  = info.friction_EE_gradients().values();
        auto EE_G_count = static_cast<std::uint32_t>(info.friction_EE_gradients().doublet_count());
        auto EE_H_rows    = info.friction_EE_hessians().row_indices();
        auto EE_H_cols    = info.friction_EE_hessians().col_indices();
        auto EE_H_values  = info.friction_EE_hessians().values();
        auto EE_H_count = static_cast<std::uint32_t>(info.friction_EE_hessians().triplet_count());
        auto EE_G_offset = static_cast<std::uint32_t>(info.friction_EE_gradients().doublet_index_offset());
        auto EE_G_total  = static_cast<std::uint32_t>(info.friction_EE_gradients().total_doublet_count());
        auto EE_H_offset = static_cast<std::uint32_t>(info.friction_EE_hessians().triplet_index_offset());
        auto EE_H_total  = static_cast<std::uint32_t>(info.friction_EE_hessians().total_triplet_count());

        auto PE_G_indices = info.friction_PE_gradients().indices();
        auto PE_G_values  = info.friction_PE_gradients().values();
        auto PE_G_count = static_cast<std::uint32_t>(info.friction_PE_gradients().doublet_count());
        auto PE_H_rows    = info.friction_PE_hessians().row_indices();
        auto PE_H_cols    = info.friction_PE_hessians().col_indices();
        auto PE_H_values  = info.friction_PE_hessians().values();
        auto PE_H_count = static_cast<std::uint32_t>(info.friction_PE_hessians().triplet_count());
        auto PE_G_offset = static_cast<std::uint32_t>(info.friction_PE_gradients().doublet_index_offset());
        auto PE_G_total  = static_cast<std::uint32_t>(info.friction_PE_gradients().total_doublet_count());
        auto PE_H_offset = static_cast<std::uint32_t>(info.friction_PE_hessians().triplet_index_offset());
        auto PE_H_total  = static_cast<std::uint32_t>(info.friction_PE_hessians().total_triplet_count());

        auto PP_G_indices = info.friction_PP_gradients().indices();
        auto PP_G_values  = info.friction_PP_gradients().values();
        auto PP_G_count = static_cast<std::uint32_t>(info.friction_PP_gradients().doublet_count());
        auto PP_H_rows    = info.friction_PP_hessians().row_indices();
        auto PP_H_cols    = info.friction_PP_hessians().col_indices();
        auto PP_H_values  = info.friction_PP_hessians().values();
        auto PP_H_count = static_cast<std::uint32_t>(info.friction_PP_hessians().triplet_count());
        auto PP_G_offset = static_cast<std::uint32_t>(info.friction_PP_gradients().doublet_index_offset());
        auto PP_G_total  = static_cast<std::uint32_t>(info.friction_PP_gradients().total_doublet_count());
        auto PP_H_offset = static_cast<std::uint32_t>(info.friction_PP_hessians().triplet_index_offset());
        auto PP_H_total  = static_cast<std::uint32_t>(info.friction_PP_hessians().total_triplet_count());

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
                    info.prev_positions(),
                    info.rest_positions(),
                    info.thicknesses(),
                    info.d_hats(),
                    info.eps_velocity(),
                    info.dt(),
                    info.friction_PTs(),
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
                    info.prev_positions(),
                    info.rest_positions(),
                    info.thicknesses(),
                    info.d_hats(),
                    info.eps_velocity(),
                    info.dt(),
                    info.friction_EEs(),
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
                    info.prev_positions(),
                    info.rest_positions(),
                    info.thicknesses(),
                    info.d_hats(),
                    info.eps_velocity(),
                    info.dt(),
                    info.friction_PEs(),
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
                    info.prev_positions(),
                    info.rest_positions(),
                    info.thicknesses(),
                    info.d_hats(),
                    info.eps_velocity(),
                    info.dt(),
                    info.friction_PPs(),
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

REGISTER_SIM_SYSTEM(IPCSimplexFrictionalContact);
}  // namespace uipc::backend::luisa
