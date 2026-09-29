/********************************************************************
 * @file strain_limiting_baraff_witkin_shell_2d.cpp
 * @brief T5.2 (batch b) — the strain-limiting Baraff-Witkin membrane constitution
 *        (reference: src/backends/cuda/finite_element/constitutions/
 *        strain_limiting_baraff_witkin_shell_2d.cu; `Codim2DConstitution` base).
 *
 * Kernel mapping (cookbook §7.9/§7.10/§7.17):
 *   * the per-element math is the *same* generic routine the unit test instantiates
 *     (`bw_shell_detail::{rest_metric_inv,element_energy,element_gradient,
 *     element_hessian}`);
 *   * the per-triangle `Dm2x2^-1` is computed by a device kernel inside `do_init`,
 *     exactly like the reference (the reference calls `cuda_tool::eigen::inverse`,
 *     the port `lps::inverse` — T2.2's closed-form 2x2 adjugate, the same routine);
 *   * the doublet/triplet views cannot be kernel arguments, so their flat buffers are
 *     passed pre-offset and the device view is rebuilt inside the kernel with
 *     `from_handles_full`, with the two offsets that cannot travel inside a buffer
 *     handle passed as scalars;
 *   * the reference's `half_hessian_size` kernel argument is a *host* constant, so the
 *     port uses its compile-time value, and `if(gradient_only) return;` becomes a
 *     predicate (RULE M2).
 *
 * Do not "fix" the reference's energy kernel: it receives the vertex thickness buffer
 * and does not use it (the element weight is the rest *area*; the stiffness attributes
 * carry the thickness convention).  The port keeps the argument and the behaviour.
 ********************************************************************/
#include <finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.h>
#include <finite_element/codim_2d_constitution.h>
#include <utils/matrix_assembler.h>
#include <frame_stream.h>
#include <uipc/common/zip.h>

namespace uipc::backend::luisa
{
namespace
{
// `int`, not `SizeT`: they are used as *template* arguments (`segment<StencilSize>`)
// and a size_t -> int conversion is not a valid converted constant expression.
constexpr int StencilSize = static_cast<int>(StrainLimitingBaraffWitkinShell2D::StencilSize);
constexpr int HalfHessianSize =
    static_cast<int>(StrainLimitingBaraffWitkinShell2D::HalfHessianSize);

using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `StrainLimitingBaraffWitkinShell2D_do_init_kernel`
auto& bw_shell_init_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector3i>,   // indices
                                   lcrt::Buffer<LVector3>,    // x_bars
                                   lcrt::Buffer<LMatrix2x2>,  // inv_Bs
                                   uint32_t>(                 // n
            [](lcrt::BufferVar<LVector3i>  indices,
               lcrt::BufferVar<LVector3>   x_bars,
               lcrt::BufferVar<LMatrix2x2> inv_Bs,
               lcrt::Var<uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto tri = load_vec3i(indices->read(i));

                    auto Dm = bw_shell_detail::rest_metric_inv<lcrt::Var<double>>(
                        load_vec3(x_bars->read(tri(0))),
                        load_vec3(x_bars->read(tri(1))),
                        load_vec3(x_bars->read(tri(2))));

                    lcrt::Var<LMatrix2x2> inv_B;
                    store_mat2x2(inv_B, Dm);
                    inv_Bs->write(i, inv_B);
                };
            });
    return k;
}

/// `StrainLimitingBaraffWitkinShell2D_do_compute_energy_kernel`
auto& bw_shell_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,      // mus
                                   lcrt::Buffer<double>,      // lambdas
                                   lcrt::Buffer<double>,      // strain_rates
                                   lcrt::Buffer<double>,      // rest_areas
                                   lcrt::Buffer<double>,      // thicknesses (unused)
                                   lcrt::Buffer<double>,      // energies
                                   lcrt::Buffer<LVector3i>,   // indices
                                   lcrt::Buffer<LVector3>,    // xs
                                   lcrt::Buffer<LMatrix2x2>,  // IBs
                                   double,                    // dt
                                   uint32_t>(                 // n
            [](lcrt::BufferVar<double>     mus,
               lcrt::BufferVar<double>     lambdas,
               lcrt::BufferVar<double>     strain_rates,
               lcrt::BufferVar<double>     rest_areas,
               lcrt::BufferVar<double>     thicknesses,
               lcrt::BufferVar<double>     energies,
               lcrt::BufferVar<LVector3i>  indices,
               lcrt::BufferVar<LVector3>   xs,
               lcrt::BufferVar<LMatrix2x2> IBs,
               lcrt::Var<double>           dt,
               lcrt::Var<uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    // reference parity: the kernel takes the thickness buffer and does
                    // not use it (the element weight is the rest area)
                    (void)thicknesses;
                    auto idx = load_vec3i(indices->read(i));
                    auto IB  = load_mat2x2(IBs->read(i));

                    auto E = bw_shell_detail::element_energy<lcrt::Var<double>>(
                        lambdas->read(i),
                        mus->read(i),
                        strain_rates->read(i),
                        rest_areas->read(i),
                        load_vec3(xs->read(idx(0))),
                        load_vec3(xs->read(idx(1))),
                        load_vec3(xs->read(idx(2))),
                        IB,
                        dt);
                    energies->write(i, E);
                };
            });
    return k;
}

/// `StrainLimitingBaraffWitkinShell2D_do_compute_gradient_hessian_kernel`
auto& bw_shell_gradient_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,      // mus
                                   lcrt::Buffer<double>,      // lambdas
                                   lcrt::Buffer<double>,      // strainRates
                                   lcrt::Buffer<LVector3i>,   // indices
                                   lcrt::Buffer<LVector3>,    // xs
                                   lcrt::Buffer<double>,      // thicknesses (unused)
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
                                   lcrt::Buffer<double>,  // rest_areas
                                   double,                // dt
                                   lcrt::Buffer<LMatrix2x2>,  // IBs
                                   uint32_t,              // gradient_only
                                   uint32_t>(             // n
            [](lcrt::BufferVar<double>     mus,
               lcrt::BufferVar<double>     lambdas,
               lcrt::BufferVar<double>     strainRates,
               lcrt::BufferVar<LVector3i>  indices,
               lcrt::BufferVar<LVector3>   xs,
               lcrt::BufferVar<double>     thicknesses,
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
               lcrt::BufferVar<double>     rest_areas,
               lcrt::Var<double>           dt,
               lcrt::BufferVar<LMatrix2x2> IBs,
               lcrt::Var<uint32_t>         gradient_only,
               lcrt::Var<uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    (void)thicknesses;
                    auto idx = load_vec3i(indices->read(i));
                    auto IB  = load_mat2x2(IBs->read(i));

                    auto x0 = load_vec3(xs->read(idx(0)));
                    auto x1 = load_vec3(xs->read(idx(1)));
                    auto x2 = load_vec3(xs->read(idx(2)));

                    auto lambda      = lambdas->read(i);
                    auto mu          = mus->read(i);
                    auto strain_rate = strainRates->read(i);
                    auto rest_area   = rest_areas->read(i);

                    auto G = bw_shell_detail::element_gradient<lcrt::Var<double>>(
                        lambda, mu, strain_rate, rest_area, x0, x1, x2, IB, dt);

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
                    DVA.segment<StencilSize>(i * StencilSize).write(idx, G);

                    // reference: `if(gradient_only) return;`
                    $if(gradient_only == 0u)
                    {
                        auto H = bw_shell_detail::element_hessian<lcrt::Var<double>>(
                            lambda, mu, strain_rate, rest_area, x0, x1, x2, IB, dt);

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
                        TMA.half_block<StencilSize>(i * HalfHessianSize).write(idx, H);
                    };
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(StrainLimitingBaraffWitkinShell2D);

U64 StrainLimitingBaraffWitkinShell2D::get_uid() const noexcept
{
    return ConstitutionUID;
}

void StrainLimitingBaraffWitkinShell2D::do_build(BuildInfo& info)
{
    (void)info;
    fem = require<FiniteElementMethod>();
}

void StrainLimitingBaraffWitkinShell2D::do_init(FiniteElementMethod::FilteredInfo& info)
{
    using ForEachInfo = FiniteElementMethod::ForEachInfo;

    auto geo_slots = world().scene().geometries();

    auto N = info.primitive_count();

    h_mus.resize(N);
    h_lambdas.resize(N);
    h_strain_rates.resize(N);

    info.for_each(
        geo_slots,
        [](geometry::SimplicialComplex& sc) -> auto
        {
            auto lambda = sc.triangles().find<Float>("lambda");
            auto mu     = sc.triangles().find<Float>("mu");

            // strain_rate is user-settable via the constitution's apply_to;
            // scenes saved before it existed default to the legacy 100.
            auto strain_rate = sc.triangles().find<Float>("strain_rate");
            if(!strain_rate)
            {
                strain_rate = sc.triangles().create<Float>("strain_rate");
                for(auto& v : geometry::view(*strain_rate))
                    v = 100.0;
            }

            return zip(lambda->view(), mu->view(), strain_rate->view());
        },
        [&](const ForEachInfo& I, auto lambda_mu_strain_rate)
        {
            auto vI = I.global_index();

            auto&& [lambda, mu, strain_rate] = lambda_mu_strain_rate;

            h_lambdas[vI]      = lambda;
            h_mus[vI]          = mu;
            h_strain_rates[vI] = strain_rate;
        });

    mus.resize(N);
    mus.view().copy_from(span<const Float>{h_mus});

    lambdas.resize(N);
    lambdas.view().copy_from(span<const Float>{h_lambdas});

    strain_rates.resize(N);
    strain_rates.view().copy_from(span<const Float>{h_strain_rates});

    auto& cinfo       = info.constitution_info();
    auto  prim_offset = cinfo.primitive_offset;
    auto  prim_count  = cinfo.primitive_count;

    auto prims  = fem->codim_2ds().subview(prim_offset, prim_count);
    auto x_bars = fem->x_bars();

    inv_B_matrices.resize(N);

    // Precompute inverse of rest shape matrix for each triangle
    auto n = static_cast<uint32_t>(prims.size());
    if(n > 0)
    {
        auto& kernel = bw_shell_init_kernel();
        frame_stream().record(
            kernel(prims, x_bars, inv_B_matrices.view(), n).dispatch(n));
    }
}

void StrainLimitingBaraffWitkinShell2D::do_report_extent(ReportExtentInfo& info)
{
    info.energy_count(h_mus.size());
    info.gradient_count(h_mus.size() * StencilSize);

    if(info.gradient_only())
        return;

    info.hessian_count(h_mus.size() * HalfHessianSize);
}

void StrainLimitingBaraffWitkinShell2D::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.indices().size());
    if(n > 0)
    {
        auto& k = bw_shell_energy_kernel();
        frame_stream().record(k(mus.cview(),
                                lambdas.cview(),
                                strain_rates.cview(),
                                info.rest_areas(),
                                info.thicknesses(),
                                info.energies(),
                                info.indices(),
                                info.xs(),
                                inv_B_matrices.cview(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}

void StrainLimitingBaraffWitkinShell2D::do_compute_gradient_hessian(
    ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.indices().size());
    if(n == 0)
        return;

    auto gradients = info.gradients();

    const auto g_offset  = static_cast<size_t>(gradients.doublet_index_offset());
    const auto g_count   = static_cast<size_t>(gradients.doublet_count());
    auto       g_indices = gradients.indices().subview(g_offset, g_count);
    auto       g_values  = gradients.values().subview(g_offset * 3u, g_count * 3u);

    // Hessian slice.  A *gradient-only* pass hands over an **empty** Hessian view (the
    // elastics allocates no Hessian buffer then) whose null buffer handle
    // `LBufferView::subview()` would dereference -- the port substitutes the (valid)
    // doublet buffers as placeholders, and the kernel's Hessian stage is predicated on
    // `gradient_only == 0u` and never reads them.
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

    auto& k = bw_shell_gradient_hessian_kernel();
    frame_stream()
        .record(k(mus.cview(),
                  lambdas.cview(),
                  strain_rates.cview(),
                  info.indices(),
                  info.xs(),
                  info.thicknesses(),
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
                  info.rest_areas(),
                  info.dt(),
                  inv_B_matrices.cview(),
                  gradient_only ? uint32_t{1} : uint32_t{0},
                  n)
                  .dispatch(n));
}
}  // namespace uipc::backend::luisa
