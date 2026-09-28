/********************************************************************
 * @file arap_3d.cpp
 * @brief T5.2 (batch a) — the ARAP 3D constitution: kernels, out-of-line members
 *        and the system registration (reference:
 *        src/backends/cuda/finite_element/constitutions/arap_3d.cu).
 *
 * The class itself is declared in `arap_3d.h` (see the deviation note there).
 *
 * Kernel mapping (cookbook §7.9/§7.10/§7.13):
 *   * `cuda_tool::CBufferView<T>` -> `lud::LCBufferView<T>`; a *subview*
 *     (`info.indices()`, `info.Dm_invs()`, ...) is passed as a kernel argument and
 *     LC binds it at `base + offset_bytes` (verified in LC's CUDA backend:
 *     `CUDABuffer::binding(offset, size)` -> `device_address() + offset`), so the
 *     per-element index arithmetic is identical to the reference's.
 *   * the doublet/triplet *views* cannot be kernel arguments (a captured view
 *     becomes a compile-time-frozen shader argument, §7.9 rule 2).  Their flat
 *     buffers are passed instead, pre-offset to the constitution's slice, and the
 *     device view is rebuilt inside the kernel with `from_handles_full`; the one
 *     piece of index arithmetic that cannot travel inside a buffer handle -- the
 *     dof-vector `subvector_offset` resp. the `submatrix_offset` -- is passed as a
 *     scalar.
 *   * the per-element math is the *same* generic routine the unit test
 *     instantiates (`arap_3d_detail::{tet_energy,tet_gradient,tet_hessian}`).
 ********************************************************************/
#include <finite_element/constitutions/arap_3d.h>
#include <utils/matrix_assembler.h>
#include <frame_stream.h>

namespace uipc::backend::luisa
{
namespace
{
// `int`, not `SizeT`: they are used as *template* arguments (`segment<StencilSize>`)
// and a size_t -> int conversion is not a valid converted constant expression.
constexpr int StencilSize     = static_cast<int>(ARAP3D::StencilSize);
constexpr int HalfHessianSize = static_cast<int>(ARAP3D::HalfHessianSize);

using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `ARAP3D_do_compute_energy_kernel`
auto& arap3d_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,      // kappas
                                                lcrt::Buffer<double>,      // energies
                                                lcrt::Buffer<LVector4i>,   // indices
                                                lcrt::Buffer<LVector3>,    // xs
                                                lcrt::Buffer<LMatrix3x3>,  // Dm_invs
                                                lcrt::Buffer<double>,      // volumes
                                                double,                    // dt
                                                uint32_t>(                 // n
        [](lcrt::BufferVar<double>     kappas,
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

                auto x0 = load_vec3(xs->read(tet(0)));
                auto x1 = load_vec3(xs->read(tet(1)));
                auto x2 = load_vec3(xs->read(tet(2)));
                auto x3 = load_vec3(xs->read(tet(3)));

                auto E = arap_3d_detail::tet_energy<lcrt::Var<double>>(
                    kappas->read(i), dt, volumes->read(i), x0, x1, x2, x3, Dm_inv);

                energies->write(i, E);
            };
        });
    return k;
}

/// `ARAP3D_do_compute_gradient_hessian_kernel`
auto& arap3d_gradient_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,      // kappas
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
            [](lcrt::BufferVar<double>     kappas,
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

                    auto x0 = load_vec3(xs->read(tet(0)));
                    auto x1 = load_vec3(xs->read(tet(1)));
                    auto x2 = load_vec3(xs->read(tet(2)));
                    auto x3 = load_vec3(xs->read(tet(3)));

                    auto kappa = kappas->read(i);
                    auto v     = volumes->read(i);

                    auto G12 = arap_3d_detail::tet_gradient<lcrt::Var<double>>(
                        kappa, dt, v, x0, x1, x2, x3, Dm_inv);

                    // the handles are pre-offset to this constitution's slice, so the
                    // device view starts at its own index 0
                    auto G3s = DoubletDeviceView::from_handles_full(
                        g_indices,
                        g_values,
                        n,                      // total_segments
                        uint32_t{0},            // doublet_index_offset (lives in the handles)
                        n,                      // doublet_count
                        n,                      // total_doublet_count
                        g_subvector_offset,     // subvector_offset
                        n);                     // subvector_extent

                    DoubletVectorAssembler DVA{G3s};
                    DVA.segment<StencilSize>(i * StencilSize).write(tet, G12);

                    // reference: `if(gradient_only) return;` -- a *device* predicate
                    // cannot drive host control flow (RULE M2), so the Hessian stage
                    // is predicated instead.
                    $if(gradient_only == 0u)
                    {
                        auto H12x12 = arap_3d_detail::tet_hessian<lcrt::Var<double>>(
                            kappa, dt, v, x0, x1, x2, x3, Dm_inv);

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

REGISTER_SIM_SYSTEM(ARAP3D);

U64 ARAP3D::get_uid() const noexcept
{
    return ConstitutionUID;
}

void ARAP3D::do_build(BuildInfo& info)
{
    (void)info;
}

void ARAP3D::do_report_extent(ReportExtentInfo& info)
{
    info.energy_count(kappas.size());
    info.gradient_count(kappas.size() * StencilSize);
    if(info.gradient_only())
        return;
    info.hessian_count(kappas.size() * HalfHessianSize);
}

void ARAP3D::do_init(FiniteElementMethod::FilteredInfo& info)
{
    using ForEachInfo = FiniteElementMethod::ForEachInfo;

    auto geo_slots = world().scene().geometries();

    auto N = info.primitive_count();

    h_kappas.resize(N);

    info.for_each(
        geo_slots,
        [](geometry::SimplicialComplex& sc) -> auto
        {
            auto kappa = sc.tetrahedra().find<Float>("kappa");
            UIPC_ASSERT(kappa, "Can't find attribute `kappa` on tetrahedra, why can it happen?");
            return kappa->view();
        },
        [&](const ForEachInfo& I, Float kappa) { h_kappas[I.global_index()] = kappa; });

    kappas.resize(N);
    kappas.view().copy_from(span<const Float>{h_kappas});
}

void ARAP3D::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.indices().size());
    if(n > 0)
    {
        auto& k = arap3d_energy_kernel();
        frame_stream().record(k(kappas.cview(),
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

void ARAP3D::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.indices().size());
    if(n == 0)
        return;

    auto gradients = info.gradients();
    auto hessians  = info.hessians();

    // The views' handles are pre-offset to this constitution's slice: LC binds a
    // subview at `base + offset_bytes`, so the device view inside the kernel can
    // start at its own index 0.
    const auto g_offset = static_cast<size_t>(gradients.doublet_index_offset());
    const auto g_count  = static_cast<size_t>(gradients.doublet_count());
    auto       g_indices = gradients.indices().subview(g_offset, g_count);
    auto       g_values  = gradients.values().subview(g_offset * 3u, g_count * 3u);

    const auto h_offset = static_cast<size_t>(hessians.triplet_index_offset());
    const auto h_count  = static_cast<size_t>(hessians.triplet_count());
    auto       h_row_indices = hessians.row_indices().subview(h_offset, h_count);
    auto       h_col_indices = hessians.col_indices().subview(h_offset, h_count);
    auto       h_values      = hessians.values().subview(h_offset * 9u, h_count * 9u);

    auto& k = arap3d_gradient_hessian_kernel();
    frame_stream()
        .record(k(kappas.cview(),
                  info.indices(),
                  info.xs(),
                  info.Dm_invs(),
                  g_indices,
                  g_values,
                  static_cast<int32_t>(gradients.subvector_offset()),
                  h_row_indices,
                  h_col_indices,
                  h_values,
                  static_cast<int32_t>(hessians.submatrix_offset().x),
                  static_cast<int32_t>(hessians.submatrix_offset().y),
                  static_cast<int32_t>(hessians.total_rows()),
                  static_cast<int32_t>(hessians.total_cols()),
                  info.rest_volumes(),
                  info.dt(),
                  info.gradient_only() ? uint32_t{1} : uint32_t{0},
                  n)
                  .dispatch(n));
}
}  // namespace uipc::backend::luisa
