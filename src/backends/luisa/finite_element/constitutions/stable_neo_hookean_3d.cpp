/********************************************************************
 * @file stable_neo_hookean_3d.cpp
 * @brief T5.2 (batch a) — the stable Neo-Hookean 3D constitution: kernels,
 *        out-of-line members and the system registration (reference:
 *        src/backends/cuda/finite_element/constitutions/stable_neo_hookean_3d.cu).
 *
 * The class itself is declared in `stable_neo_hookean_3d.h` (see the deviation
 * note there).  The kernel mapping is the same as `arap_3d.cpp`'s (see the file
 * comment there): subviews travel as kernel arguments (LC binds them at
 * `base + offset_bytes`), the doublet/triplet views are rebuilt inside the kernel
 * from their pre-offset handles, and the per-element math is the generic
 * `snh_3d_detail::{tet_energy,tet_gradient,tet_hessian}` the unit test also
 * instantiates.
 *
 * The reference's gradient/Hessian kernel writes
 * `DVA(I * StencilSize + i).write(tet(i), G)` and
 * `H3x3s(hessian_offset++).write(tet(left), tet(right), H)` with the
 * `tet(left) > tet(right)` swap inlined; the port writes the same doublets with
 * `DoubletVectorAssembler::segment<4>` and the same triplets with
 * `TripletMatrixAssembler::half_block<4>` (whose `upper_LR` is that swap, rewritten
 * as a select) -- identical (row, col) sequences, identical blocks.
 ********************************************************************/
#include <finite_element/constitutions/stable_neo_hookean_3d.h>
#include <utils/matrix_assembler.h>
#include <frame_stream.h>

namespace uipc::backend::luisa
{
namespace
{
// `int`, not `SizeT`: they are used as *template* arguments (see arap_3d.cpp).
constexpr int StencilSize     = static_cast<int>(StableNeoHookean3D::StencilSize);
constexpr int HalfHessianSize = static_cast<int>(StableNeoHookean3D::HalfHessianSize);

using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `StableNeoHookean3D_do_compute_energy_kernel`
auto& snh3d_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,      // mus
                                                lcrt::Buffer<double>,      // lambdas
                                                lcrt::Buffer<double>,      // energies
                                                lcrt::Buffer<LVector4i>,   // indices
                                                lcrt::Buffer<LVector3>,    // xs
                                                lcrt::Buffer<LMatrix3x3>,  // Dm_invs
                                                lcrt::Buffer<double>,      // volumes
                                                double,                    // dt
                                                uint32_t>(                 // n
        [](lcrt::BufferVar<double>     mus,
           lcrt::BufferVar<double>     lambdas,
           lcrt::BufferVar<double>     energies,
           lcrt::BufferVar<LVector4i>  indices,
           lcrt::BufferVar<LVector3>   xs,
           lcrt::BufferVar<LMatrix3x3> Dm_invs,
           lcrt::BufferVar<double>     volumes,
           lcrt::Var<double>           dt,
           lcrt::Var<uint32_t>         n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto tet    = load_vec4i(indices->read(i));
                auto Dm_inv = load_mat3x3(Dm_invs->read(i));
                auto mu     = mus->read(i);
                auto lambda = lambdas->read(i);

                auto x0 = load_vec3(xs->read(tet(0)));
                auto x1 = load_vec3(xs->read(tet(1)));
                auto x2 = load_vec3(xs->read(tet(2)));
                auto x3 = load_vec3(xs->read(tet(3)));

                auto E = snh_3d_detail::tet_energy<lcrt::Var<double>>(
                    mu, lambda, dt, volumes->read(i), x0, x1, x2, x3, Dm_inv);

                energies->write(i, E);
            };
        });
    return k;
}

/// `StableNeoHookean3D_do_compute_gradient_hessian_kernel`
auto& snh3d_gradient_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,      // mus
                                   lcrt::Buffer<double>,      // lambdas
                                   lcrt::Buffer<LVector4i>,   // indices
                                   lcrt::Buffer<LVector3>,    // xs
                                   lcrt::Buffer<LMatrix3x3>,  // Dm_invs
                                   // gradients (the doublet slice)
                                   lcrt::Buffer<int32_t>,
                                   lcrt::Buffer<double>,
                                   int32_t,  // subvector_offset
                                   // hessians (the triplet slice)
                                   lcrt::Buffer<int32_t>,
                                   lcrt::Buffer<int32_t>,
                                   lcrt::Buffer<double>,
                                   int32_t,  // submatrix_offset.x
                                   int32_t,  // submatrix_offset.y
                                   int32_t,  // total_block_rows
                                   int32_t,  // total_block_cols
                                   lcrt::Buffer<double>,  // volumes
                                   double,                // dt
                                   uint32_t,              // gradient_only
                                   uint32_t>(             // n
            [](lcrt::BufferVar<double>     mus,
               lcrt::BufferVar<double>     lambdas,
               lcrt::BufferVar<LVector4i>  indices,
               lcrt::BufferVar<LVector3>   xs,
               lcrt::BufferVar<LMatrix3x3> Dm_invs,
               lcrt::BufferVar<int32_t>    g_indices,
               lcrt::BufferVar<double>     g_values,
               lcrt::Var<int32_t>          g_subvector_offset,
               lcrt::BufferVar<int32_t>    h_row_indices,
               lcrt::BufferVar<int32_t>    h_col_indices,
               lcrt::BufferVar<double>     h_values,
               lcrt::Var<int32_t>          h_submatrix_offset_x,
               lcrt::Var<int32_t>          h_submatrix_offset_y,
               lcrt::Var<int32_t>          h_total_block_rows,
               lcrt::Var<int32_t>          h_total_block_cols,
               lcrt::BufferVar<double>     volumes,
               lcrt::Var<double>           dt,
               lcrt::Var<uint32_t>         gradient_only,
               lcrt::Var<uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto tet    = load_vec4i(indices->read(i));
                    auto Dm_inv = load_mat3x3(Dm_invs->read(i));
                    auto mu     = mus->read(i);
                    auto lambda = lambdas->read(i);

                    auto x0 = load_vec3(xs->read(tet(0)));
                    auto x1 = load_vec3(xs->read(tet(1)));
                    auto x2 = load_vec3(xs->read(tet(2)));
                    auto x3 = load_vec3(xs->read(tet(3)));

                    auto v = volumes->read(i);

                    auto G12 = snh_3d_detail::tet_gradient<lcrt::Var<double>>(
                        mu, lambda, dt, v, x0, x1, x2, x3, Dm_inv);

                    auto G3s = DoubletDeviceView::from_handles_full(
                        g_indices,
                        g_values,
                        n,                   // total_segments
                        uint32_t{0},         // doublet_index_offset (lives in the handles)
                        n,                   // doublet_count
                        n,                   // total_doublet_count
                        g_subvector_offset,  // subvector_offset
                        n);                  // subvector_extent

                    DoubletVectorAssembler DVA{G3s};
                    DVA.segment<StencilSize>(i * StencilSize).write(tet, G12);

                    // reference: `if(gradient_only) return;` (a device predicate,
                    // so the Hessian stage is predicated -- RULE M2)
                    $if(gradient_only == 0u)
                    {
                        auto H12x12 = snh_3d_detail::tet_hessian<lcrt::Var<double>>(
                            mu, lambda, dt, v, x0, x1, x2, x3, Dm_inv);

                        auto H3x3s = TripletDeviceView::from_handles_full(
                            h_row_indices,
                            h_col_indices,
                            h_values,
                            h_total_block_rows,
                            h_total_block_cols,
                            uint32_t{0},          // triplet_index_offset (in the handles)
                            n * HalfHessianSize,  // triplet_count
                            n * HalfHessianSize,  // total_triplet_count
                            h_submatrix_offset_x,
                            h_submatrix_offset_y,
                            h_total_block_rows,   // submatrix_extent.x
                            h_total_block_cols);  // submatrix_extent.y

                        TripletMatrixAssembler TMA{H3x3s};
                        TMA.half_block<StencilSize>(i * HalfHessianSize).write(tet, H12x12);
                    };
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(StableNeoHookean3D);

U64 StableNeoHookean3D::get_uid() const noexcept
{
    return ConstitutionUID;
}

void StableNeoHookean3D::do_build(BuildInfo& info)
{
    (void)info;
}

void StableNeoHookean3D::do_report_extent(ReportExtentInfo& info)
{
    info.energy_count(mus.size());
    info.gradient_count(mus.size() * StencilSize);

    if(info.gradient_only())
        return;

    info.hessian_count(mus.size() * HalfHessianSize);
}

void StableNeoHookean3D::do_init(FiniteElementMethod::FilteredInfo& info)
{
    using ForEachInfo = FiniteElementMethod::ForEachInfo;

    auto geo_slots = world().scene().geometries();

    auto N = info.primitive_count();

    h_mus.resize(N);
    h_lambdas.resize(N);

    info.for_each(
        geo_slots,
        [](geometry::SimplicialComplex& sc) -> auto
        {
            auto mu     = sc.tetrahedra().find<Float>("mu");
            auto lambda = sc.tetrahedra().find<Float>("lambda");

            return zip(mu->view(), lambda->view());
        },
        [&](const ForEachInfo& I, auto mu_and_lambda)
        {
            auto&& [mu, lambda] = mu_and_lambda;

            auto vI = I.global_index();

            h_mus[vI]     = mu;
            h_lambdas[vI] = lambda;
        });

    mus.resize(N);
    mus.view().copy_from(span<const Float>{h_mus});

    lambdas.resize(N);
    lambdas.view().copy_from(span<const Float>{h_lambdas});
}

void StableNeoHookean3D::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.indices().size());
    if(n > 0)
    {
        auto& k = snh3d_energy_kernel();
        frame_stream().record(k(mus.cview(),
                                lambdas.cview(),
                                info.energies(),
                                info.indices(),
                                info.xs(),
                                info.Dm_invs(),
                                info.rest_volumes(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}

void StableNeoHookean3D::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.indices().size());
    if(n == 0)
        return;

    auto gradients = info.gradients();

    // The handles are pre-offset to this constitution's slice (see arap_3d.cpp).
    const auto g_offset = static_cast<size_t>(gradients.doublet_index_offset());
    const auto g_count  = static_cast<size_t>(gradients.doublet_count());
    auto       g_indices = gradients.indices().subview(g_offset, g_count);
    auto       g_values  = gradients.values().subview(g_offset * 3u, g_count * 3u);

    // Hessian slice; empty on a *gradient-only* pass -- see arap_3d.cpp for why the
    // placeholder substitution is required (an LC empty view's handle is null).
    const bool gradient_only = info.gradient_only();
    lud::LCBufferView<int32_t> h_row_indices;
    lud::LCBufferView<int32_t> h_col_indices;
    lud::LCBufferView<double>  h_values;
    int32_t h_submatrix_offset_x = 0;
    int32_t h_submatrix_offset_y = 0;
    int32_t h_total_block_rows   = 0;
    int32_t h_total_block_cols   = 0;
    if(gradient_only)
    {
        h_row_indices = g_indices;
        h_col_indices = g_indices;
        h_values      = g_values;
        h_total_block_rows = h_total_block_cols = static_cast<int32_t>(g_count);
    }
    else
    {
        auto       hessians = info.hessians();
        const auto h_offset = static_cast<size_t>(hessians.triplet_index_offset());
        const auto h_count  = static_cast<size_t>(hessians.triplet_count());
        h_row_indices = hessians.row_indices().subview(h_offset, h_count);
        h_col_indices = hessians.col_indices().subview(h_offset, h_count);
        h_values      = hessians.values().subview(h_offset * 9u, h_count * 9u);
        h_submatrix_offset_x = static_cast<int32_t>(hessians.submatrix_offset().x);
        h_submatrix_offset_y = static_cast<int32_t>(hessians.submatrix_offset().y);
        h_total_block_rows   = static_cast<int32_t>(hessians.total_rows());
        h_total_block_cols   = static_cast<int32_t>(hessians.total_cols());
    }

    auto& k = snh3d_gradient_hessian_kernel();
    frame_stream()
        .record(k(mus.cview(),
                  lambdas.cview(),
                  info.indices(),
                  info.xs(),
                  info.Dm_invs(),
                  g_indices,
                  g_values,
                  static_cast<int32_t>(gradients.subvector_offset()),
                  h_row_indices,
                  h_col_indices,
                  h_values,
                  h_submatrix_offset_x,
                  h_submatrix_offset_y,
                  h_total_block_rows,
                  h_total_block_cols,
                  info.rest_volumes(),
                  info.dt(),
                  gradient_only ? uint32_t{1} : uint32_t{0},
                  n)
                  .dispatch(n));
}
}  // namespace uipc::backend::luisa
