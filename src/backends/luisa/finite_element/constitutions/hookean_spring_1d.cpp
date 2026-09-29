/********************************************************************
 * @file hookean_spring_1d.cpp
 * @brief T5.2 (batch c) — the 1d Hookean-spring constitution: kernels, out-of-line
 *        members and the system registration (reference:
 *        src/backends/cuda/finite_element/constitutions/hookean_spring_1d.cu).
 *
 * The class itself is declared in `hookean_spring_1d.h` (see the deviation note there).
 *
 * Kernel mapping (cookbook §7.9/§7.10/§7.17):
 *   * the per-element math is the *same* generic routine the unit test instantiates
 *     (`hs1d_detail::{stencil_energy,stencil_gradient,stencil_hessian}`), so the oracle
 *     and the shipped kernel run identical code (§7.4);
 *   * `cuda_tool::CBufferView<T>` -> `lud::LCBufferView<T>`; the constitution's own
 *     device buffers and the method's (possibly sub-viewed) buffers are passed as
 *     kernel arguments, never captured (§7.9 rule 2);
 *   * the doublet/triplet *views* cannot be kernel arguments, so their flat buffers are
 *     passed pre-offset to this constitution's slice and the device view is rebuilt
 *     inside the kernel with `from_handles_full`; the pieces of index arithmetic that
 *     cannot travel inside a buffer handle (the dof-vector `subvector_offset` resp. the
 *     `submatrix_offset`) travel as scalars;
 *   * the reference's `if(!gradient_only) { … }` (a *device* predicate) becomes
 *     `$if(gradient_only == 0u)` (RULE M2), with the empty-Hessian placeholder of
 *     §7.17 rule 6 (a gradient-only pass hands over a null-handle Hessian view).
 ********************************************************************/
#include <finite_element/constitutions/hookean_spring_1d.h>
#include <utils/matrix_assembler.h>
#include <frame_stream.h>

#include <numbers>

namespace uipc::backend::luisa
{
namespace
{
// `int`, not `SizeT`: they are used as *template* arguments (`segment<StencilSize>`)
// and a size_t -> int conversion is not a valid converted constant expression.
constexpr int StencilSize     = static_cast<int>(HookeanSpring1D::StencilSize);
constexpr int HalfHessianSize = static_cast<int>(HookeanSpring1D::HalfHessianSize);

using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `HookeanSpring1D_do_compute_energy_kernel`
auto& hs1d_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,     // kappas
                                   lcrt::Buffer<double>,     // rest_lengths
                                   lcrt::Buffer<double>,     // thicknesses
                                   lcrt::Buffer<double>,     // energies
                                   lcrt::Buffer<LVector2i>,  // indices
                                   lcrt::Buffer<LVector3>,   // xs
                                   double,                   // dt
                                   double,                   // Pi
                                   uint32_t>(                // n
            [](lcrt::BufferVar<double>    kappas,
               lcrt::BufferVar<double>    rest_lengths,
               lcrt::BufferVar<double>    thicknesses,
               lcrt::BufferVar<double>    energies,
               lcrt::BufferVar<LVector2i> indices,
               lcrt::BufferVar<LVector3>  xs,
               lcrt::Var<double>          dt,
               lcrt::Var<double>          Pi,
               lcrt::Var<uint32_t>        n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto idx = load_vec2i(indices->read(i));
                    auto x0  = load_vec3(xs->read(idx(0)));
                    auto x1  = load_vec3(xs->read(idx(1)));

                    auto L0    = rest_lengths->read(i);
                    auto r     = edge_thickness(thicknesses->read(idx(0)),
                                            thicknesses->read(idx(1)));
                    auto kappa = kappas->read(i);

                    auto E = hs1d_detail::stencil_energy<lcrt::Var<double>>(
                        kappa, L0, r, Pi, dt, x0, x1);
                    energies->write(i, E);
                };
            });
    return k;
}

/// `HookeanSpring1D_do_compute_gradient_hessian_kernel`
auto& hs1d_gradient_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,     // kappas
                                   lcrt::Buffer<double>,     // rest_lengths
                                   lcrt::Buffer<double>,     // thicknesses
                                   lcrt::Buffer<LVector2i>,  // indices
                                   lcrt::Buffer<LVector3>,   // xs
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
                                   double,   // dt
                                   double,   // Pi
                                   uint32_t, // gradient_only
                                   uint32_t>(// n
            [](lcrt::BufferVar<double>    kappas,
               lcrt::BufferVar<double>    rest_lengths,
               lcrt::BufferVar<double>    thicknesses,
               lcrt::BufferVar<LVector2i> indices,
               lcrt::BufferVar<LVector3>  xs,
               lcrt::BufferVar<int32_t>   g_indices,
               lcrt::BufferVar<double>    g_values,
               lcrt::Var<int32_t>         g_subvector_offset,
               lcrt::BufferVar<int32_t>   h_row_indices,
               lcrt::BufferVar<int32_t>   h_col_indices,
               lcrt::BufferVar<double>    h_values,
               lcrt::Var<int32_t>         h_submatrix_offset_x,
               lcrt::Var<int32_t>         h_submatrix_offset_y,
               lcrt::Var<int32_t>         h_total_block_rows,
               lcrt::Var<int32_t>         h_total_block_cols,
               lcrt::Var<double>          dt,
               lcrt::Var<double>          Pi,
               lcrt::Var<uint32_t>        gradient_only,
               lcrt::Var<uint32_t>        n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto idx = load_vec2i(indices->read(i));
                    auto x0  = load_vec3(xs->read(idx(0)));
                    auto x1  = load_vec3(xs->read(idx(1)));

                    auto L0    = rest_lengths->read(i);
                    auto r     = edge_thickness(thicknesses->read(idx(0)),
                                            thicknesses->read(idx(1)));
                    auto kappa = kappas->read(i);

                    auto G6 = hs1d_detail::stencil_gradient<lcrt::Var<double>>(
                        kappa, L0, r, Pi, dt, x0, x1);

                    // the handles are pre-offset to this constitution's slice, so the
                    // device view starts at its own index 0
                    auto G3s = DoubletDeviceView::from_handles_full(
                        g_indices,
                        g_values,
                        n,                   // total_segments
                        uint32_t{0},         // doublet_index_offset (lives in the handles)
                        n,                   // doublet_count
                        n,                   // total_doublet_count
                        g_subvector_offset,  // subvector_offset
                        n);                  // subvector_extent

                    DoubletVectorAssembler VA{G3s};
                    VA.segment<StencilSize>(i * StencilSize).write(idx, G6);

                    // reference: `if(!gradient_only) { … }`
                    $if(gradient_only == 0u)
                    {
                        auto H6x6 = hs1d_detail::stencil_hessian<lcrt::Var<double>>(
                            kappa, L0, r, Pi, dt, x0, x1);

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

                        TripletMatrixAssembler MA{H3x3s};
                        MA.half_block<StencilSize>(i * HalfHessianSize).write(idx, H6x6);
                    };
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(HookeanSpring1D);

U64 HookeanSpring1D::get_uid() const noexcept
{
    return ConstitutionUID;
}

void HookeanSpring1D::do_build(BuildInfo& info)
{
    (void)info;
}

void HookeanSpring1D::do_report_extent(ReportExtentInfo& info)
{
    info.energy_count(kappas.size());
    info.gradient_count(kappas.size() * StencilSize);

    if(info.gradient_only())
        return;

    info.hessian_count(kappas.size() * HalfHessianSize);
}

void HookeanSpring1D::do_init(FiniteElementMethod::FilteredInfo& info)
{
    using ForEachInfo = FiniteElementMethod::ForEachInfo;

    auto geo_slots = world().scene().geometries();

    auto N = info.primitive_count();

    h_kappas.resize(N);

    info.for_each(
        geo_slots,
        [](geometry::SimplicialComplex& sc) -> auto
        {
            auto kappa = sc.edges().find<Float>("kappa");
            UIPC_ASSERT(kappa, "Can't find attribute `kappa` on edges, why can it happen?");
            return kappa->view();
        },
        [&](const ForEachInfo& I, Float kappa)
        {
            auto vI = I.global_index();
            // retrieve material parameters
            h_kappas[vI] = kappa;
        });

    kappas.resize(N);
    kappas.view().copy_from(span<const Float>{h_kappas});
}

void HookeanSpring1D::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.indices().size());
    if(n > 0)
    {
        auto& k = hs1d_energy_kernel();
        frame_stream().record(k(kappas.cview(),
                                info.rest_lengths(),
                                info.thicknesses(),
                                info.energies(),
                                info.indices(),
                                info.xs(),
                                info.dt(),
                                std::numbers::pi,
                                n)
                                .dispatch(n));
    }
}

void HookeanSpring1D::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.indices().size());
    if(n == 0)
        return;

    auto gradients = info.gradients();

    // The views' handles are pre-offset to this constitution's slice: LC binds a
    // subview at `base + offset_bytes`, so the device view inside the kernel can
    // start at its own index 0.
    const auto g_offset  = static_cast<size_t>(gradients.doublet_index_offset());
    const auto g_count   = static_cast<size_t>(gradients.doublet_count());
    auto       g_indices = gradients.indices().subview(g_offset, g_count);
    auto       g_values  = gradients.values().subview(g_offset * 3u, g_count * 3u);

    // Hessian slice.  A *gradient-only* pass hands over an **empty** Hessian view whose
    // buffer handle is null (the reference's elastics allocates no Hessian buffer then),
    // and `LBufferView::subview()` on a null handle crashes; the port substitutes the
    // (valid) doublet buffers as placeholders when the flag is set (cookbook §7.17
    // rule 6).  The kernel's Hessian stage is predicated and never reads them.
    const bool gradient_only = info.gradient_only();

    lud::LCBufferView<int32_t> h_row_indices;
    lud::LCBufferView<int32_t> h_col_indices;
    lud::LCBufferView<double>  h_values;
    int32_t                    h_submatrix_offset_x = 0;
    int32_t                    h_submatrix_offset_y = 0;
    int32_t                    h_total_block_rows   = 0;
    int32_t                    h_total_block_cols   = 0;

    if(gradient_only)
    {
        h_row_indices = g_indices;
        h_col_indices = g_indices;
        h_values      = g_values;
        h_total_block_rows = h_total_block_cols = static_cast<int32_t>(g_count);
    }
    else
    {
        auto hessians = info.hessians();

        const auto h_offset = static_cast<size_t>(hessians.triplet_index_offset());
        const auto h_count  = static_cast<size_t>(hessians.triplet_count());
        h_row_indices       = hessians.row_indices().subview(h_offset, h_count);
        h_col_indices       = hessians.col_indices().subview(h_offset, h_count);
        h_values            = hessians.values().subview(h_offset * 9u, h_count * 9u);
        h_submatrix_offset_x = static_cast<int32_t>(hessians.submatrix_offset().x);
        h_submatrix_offset_y = static_cast<int32_t>(hessians.submatrix_offset().y);
        h_total_block_rows   = static_cast<int32_t>(hessians.total_rows());
        h_total_block_cols   = static_cast<int32_t>(hessians.total_cols());
    }

    auto& k = hs1d_gradient_hessian_kernel();
    frame_stream()
        .record(k(kappas.cview(),
                  info.rest_lengths(),
                  info.thicknesses(),
                  info.indices(),
                  info.xs(),
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
                  info.dt(),
                  std::numbers::pi,
                  gradient_only ? uint32_t{1} : uint32_t{0},
                  n)
                  .dispatch(n));
}
}  // namespace uipc::backend::luisa
