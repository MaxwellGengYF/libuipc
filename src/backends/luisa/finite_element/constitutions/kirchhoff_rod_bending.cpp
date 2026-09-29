/********************************************************************
 * @file kirchhoff_rod_bending.cpp
 * @brief T5.2 (batch c) — the Kirchhoff-rod bending constitution: kernels, out-of-line
 *        members and the system registration (reference:
 *        src/backends/cuda/finite_element/constitutions/kirchhoff_rod_bending.cu).
 *
 * The class itself is declared in `kirchhoff_rod_bending.h` (see the deviation note
 * there).
 *
 * Kernel mapping (cookbook §7.9/§7.10/§7.17):
 *   * the per-element math is the *same* generic routine the unit test instantiates
 *     (`krb_detail::{stencil_energy,stencil_gradient,stencil_hessian}`), so the oracle
 *     and the shipped kernel run identical code (§7.4);
 *   * `cuda_tool::CBufferView<T>` -> `lud::LCBufferView<T>`; the constitution's own
 *     device buffers and the method's (possibly sub-viewed) buffers are passed as
 *     kernel arguments, never captured (§7.9 rule 2);
 *   * the doublet/triplet *views* cannot be kernel arguments, so their flat buffers are
 *     passed pre-offset to this constitution's slice and the device view is rebuilt
 *     inside the kernel with `from_handles_full`; the pieces of index arithmetic that
 *     cannot travel inside a buffer handle (the dof-vector `subvector_offset` resp. the
 *     `submatrix_offset`) travel as scalars;
 *   * the reference's `if(gradient_only) return;` is a *device* predicate, so the
 *     Hessian stage is wrapped in `$if(gradient_only == 0u)` (RULE M2).  A
 *     gradient-only pass hands over an **empty** Hessian view whose buffer handle is
 *     null (the reference's elastics allocates no Hessian buffer then), so the caller
 *     substitutes the valid doublet buffers as placeholders — `LBufferView::subview()`
 *     on a null handle crashes (cookbook §7.17 rule 6, the same guard batch a's
 *     cd5f8bc9 fix and batch b's files carry).
 *   * `thicknesses` is indexed by *global* vertex id, not hinge id (the reference's own
 *     comment), which is why the hinge indices are stored already offset by the
 *     geometry's `backend_fem_vertex_offset` (see `do_init`).
 ********************************************************************/
#include <finite_element/constitutions/kirchhoff_rod_bending.h>
#include <utils/matrix_assembler.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>

#include <algorithm>
#include <numbers>

namespace uipc::backend::luisa
{
namespace
{
// `int`, not `SizeT`: they are used as *template* arguments (`segment<StencilSize>`)
// and a size_t -> int conversion is not a valid converted constant expression.
constexpr int StencilSize     = static_cast<int>(KirchhoffRodBending::StencilSize);
constexpr int HalfHessianSize = static_cast<int>(KirchhoffRodBending::HalfHessianSize);

using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `KirchhoffRodBending_do_compute_energy_kernel`
auto& krb_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector3i>,  // hinges
                                   lcrt::Buffer<double>,      // bending_stiffnesses
                                   lcrt::Buffer<double>,      // thicknesses
                                   lcrt::Buffer<LVector3>,    // xs
                                   lcrt::Buffer<LVector3>,    // x_bars
                                   lcrt::Buffer<double>,      // energies
                                   double,                    // dt
                                   double,                    // Pi
                                   uint32_t>(                 // n
            [](lcrt::BufferVar<LVector3i> hinges,
               lcrt::BufferVar<double>    bending_stiffnesses,
               lcrt::BufferVar<double>    thicknesses,
               lcrt::BufferVar<LVector3>  xs,
               lcrt::BufferVar<LVector3>  x_bars,
               lcrt::BufferVar<double>    energies,
               lcrt::Var<double>          dt,
               lcrt::Var<double>          Pi,
               lcrt::Var<uint32_t>        n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto hinge = load_vec3i(hinges->read(i));
                    // reference: `thicknesses is indexed by global vertex id, not hinge id.`
                    auto r = thicknesses->read(hinge(1));

                    auto x0 = load_vec3(xs->read(hinge(0)));
                    auto x1 = load_vec3(xs->read(hinge(1)));
                    auto x2 = load_vec3(xs->read(hinge(2)));

                    auto x0_bar = load_vec3(x_bars->read(hinge(0)));
                    auto x1_bar = load_vec3(x_bars->read(hinge(1)));
                    auto x2_bar = load_vec3(x_bars->read(hinge(2)));

                    // Rest length of the two edges
                    auto L0 = krb_detail::rest_length<lcrt::Var<double>>(x0_bar, x1_bar, x2_bar);

                    // reference: `Float k = bending_stiffnesses(I) * dt * dt;`
                    // (`stencil_energy` applies that scale itself, in the reference's
                    //  association order)
                    auto E = krb_detail::stencil_energy<lcrt::Var<double>>(
                        bending_stiffnesses->read(i), dt, x0, x1, x2, L0, r, Pi);

                    energies->write(i, E);
                };
            });
    return k;
}

/// `KirchhoffRodBending_do_compute_gradient_hessian_kernel`
auto& krb_gradient_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector3i>,  // hinges
                                   lcrt::Buffer<double>,      // bending_stiffnesses
                                   lcrt::Buffer<double>,      // thicknesses
                                   lcrt::Buffer<LVector3>,    // xs
                                   lcrt::Buffer<LVector3>,    // x_bars
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
            [](lcrt::BufferVar<LVector3i> hinges,
               lcrt::BufferVar<double>    bending_stiffnesses,
               lcrt::BufferVar<double>    thicknesses,
               lcrt::BufferVar<LVector3>  xs,
               lcrt::BufferVar<LVector3>  x_bars,
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
                    auto hinge = load_vec3i(hinges->read(i));
                    auto k     = bending_stiffnesses->read(i);
                    // thicknesses is indexed by global vertex id, not hinge id.
                    auto r = thicknesses->read(hinge(1));

                    auto x0 = load_vec3(xs->read(hinge(0)));
                    auto x1 = load_vec3(xs->read(hinge(1)));
                    auto x2 = load_vec3(xs->read(hinge(2)));

                    auto x0_bar = load_vec3(x_bars->read(hinge(0)));
                    auto x1_bar = load_vec3(x_bars->read(hinge(1)));
                    auto x2_bar = load_vec3(x_bars->read(hinge(2)));

                    // Rest length of the two edges
                    auto L0 = krb_detail::rest_length<lcrt::Var<double>>(x0_bar, x1_bar, x2_bar);

                    auto G9 = krb_detail::stencil_gradient<lcrt::Var<double>>(
                        k, dt, x0, x1, x2, L0, r, Pi);

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

                    DoubletVectorAssembler DVA{G3s};
                    DVA.segment<StencilSize>(i * StencilSize).write(hinge, G9);

                    // reference: `if(gradient_only) return;`
                    $if(gradient_only == 0u)
                    {
                        auto H9x9 = krb_detail::stencil_hessian<lcrt::Var<double>>(
                            k, dt, x0, x1, x2, L0, r, Pi);

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
                        TMA.half_block<StencilSize>(i * HalfHessianSize).write(hinge, H9x9);
                    };
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(KirchhoffRodBending);

U64 KirchhoffRodBending::get_uid() const noexcept
{
    return ConstitutionUID;
}

void KirchhoffRodBending::do_build(BuildInfo& info)
{
    (void)info;
}

void KirchhoffRodBending::do_init(FilteredInfo& info)
{
    using ForEachInfo = FiniteElementMethod::ForEachInfo;
    auto geo_slots    = world().scene().geometries();

    list<Vector3i> hinge_list;  // X0, X1, X2
    list<Float>    bending_stiffness_list;

    info.for_each(  //
        geo_slots,
        [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
        {
            (void)I;
            unordered_map<IndexT, set<IndexT>> hinge_map;  // Vertex -> Connected Vertices

            auto vertex_offset = sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
            UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
            auto vertex_offset_v = vertex_offset->view().front();

            auto edges = sc.edges().topo().view();

            for(auto e : edges)
            {
                auto v0 = e[0];
                auto v1 = e[1];

                hinge_map[v0].insert(v1);
                hinge_map[v1].insert(v0);
            }

            auto bending_stiffnesses = sc.vertices().find<Float>("bending_stiffness");
            UIPC_ASSERT(bending_stiffnesses, "Bending stiffness not found, why?");

            auto bs_view = bending_stiffnesses->view();

            for(auto&& [v, connected] : hinge_map)
            {
                auto bs = bs_view[v];

                if(connected.size() < 2)  // Not a hinge
                    continue;

                for(auto v1 : connected)
                    for(auto v2 : connected)
                    {
                        if(v1 >= v2)  // Avoid duplicate
                            continue;

                        hinge_list.push_back({vertex_offset_v + v1,
                                              vertex_offset_v + v,  // center vertex
                                              vertex_offset_v + v2});
                        bending_stiffness_list.push_back(bs);
                    }
            }
        });

    // Setup data
    h_hinges.resize(hinge_list.size());
    h_bending_stiffness.resize(bending_stiffness_list.size());
    std::ranges::move(hinge_list, h_hinges.begin());
    std::ranges::move(bending_stiffness_list, h_bending_stiffness.begin());

    // Copy to device
    // host `Vector3i` (Eigen) -> the device element type `LVector3i`
    copy_host_to_device(hinges, h_hinges);

    bending_stiffnesses.resize(h_bending_stiffness.size());
    bending_stiffnesses.view().copy_from(span<const Float>{h_bending_stiffness});
}

void KirchhoffRodBending::do_report_extent(ReportExtentInfo& info)
{
    info.energy_count(hinges.size());                    // Each hinge has 1 energy
    info.gradient_count(hinges.size() * StencilSize);    // Each hinge has 3 vertices

    if(info.gradient_only())
        return;

    info.hessian_count(hinges.size() * HalfHessianSize);
}

void KirchhoffRodBending::do_compute_energy(ComputeEnergyInfo& info)
{
    constexpr Float Pi = std::numbers::pi;

    auto n = static_cast<uint32_t>(info.energies().size());
    if(n > 0)
    {
        auto& k = krb_energy_kernel();
        frame_stream().record(k(hinges.cview(),
                                bending_stiffnesses.cview(),
                                info.thicknesses(),
                                info.xs(),
                                info.x_bars(),
                                info.energies(),
                                info.dt(),
                                Pi,
                                n)
                                .dispatch(n));
    }
}

void KirchhoffRodBending::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    constexpr Float Pi = std::numbers::pi;

    auto n = static_cast<uint32_t>(hinges.size());
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

    // Hessian slice.  A *gradient-only* pass hands over an **empty** Hessian view: the
    // reference's elastics allocates no Hessian buffer then, and `hessians()` returns a
    // default view whose buffer handle is null -- `LBufferView::subview()` dereferences
    // it (measured: a hard crash), unlike cuda_tool's plain-value view.  The port
    // therefore substitutes the (valid) doublet buffers as placeholders when the flag is
    // set; the kernel's Hessian stage is predicated on `gradient_only == 0u` and never
    // reads them.
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

    auto& k = krb_gradient_hessian_kernel();
    frame_stream()
        .record(k(hinges.cview(),
                  bending_stiffnesses.cview(),
                  info.thicknesses(),
                  info.xs(),
                  info.x_bars(),
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
                  Pi,
                  gradient_only ? uint32_t{1} : uint32_t{0},
                  n)
                  .dispatch(n));
}
}  // namespace uipc::backend::luisa
