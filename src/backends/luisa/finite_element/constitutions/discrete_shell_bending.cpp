/********************************************************************
 * @file discrete_shell_bending.cpp
 * @brief T5.2 (batch b) — the discrete-shells bending constitution: kernels,
 *        out-of-line members and the system registration (reference:
 *        src/backends/cuda/finite_element/constitutions/discrete_shell_bending.cu).
 *
 * The class itself is declared in `discrete_shell_bending.h` (see the deviation note
 * there).
 *
 * Kernel mapping (cookbook §7.9/§7.10/§7.17):
 *   * the per-element math is the *same* generic routine the unit test instantiates
 *     (`dsb_detail::{stencil_energy,stencil_gradient,stencil_hessian}`), so the
 *     oracle and the shipped kernel run identical code (§7.4);
 *   * `xcuda_tool::CBufferView<T>` -> `lud::LCBufferView<T>`; the constitution's own
 *     device buffers are passed as kernel arguments (never captured, §7.9 rule 2);
 *   * the doublet/triplet *views* cannot be kernel arguments, so their flat buffers
 *     are passed pre-offset to this constitution's slice and the device view is
 *     rebuilt inside the kernel with `from_handles_full`; the two pieces of index
 *     arithmetic that cannot travel inside a buffer handle (the dof-vector
 *     `subvector_offset` resp. the `submatrix_offset`) travel as scalars;
 *   * the reference's `if(gradient_only) return;` is a *device* predicate, so the
 *     Hessian stage is wrapped in `$if(gradient_only == 0u)` (RULE M2).
 ********************************************************************/
#include <finite_element/constitutions/discrete_shell_bending.h>
#include <utils/matrix_assembler.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>

namespace uipc::backend::luisa
{
namespace
{
// `int`, not `SizeT`: they are used as *template* arguments (`segment<StencilSize>`)
// and a size_t -> int conversion is not a valid converted constant expression.
constexpr int StencilSize     = static_cast<int>(DiscreteShellBending::StencilSize);
constexpr int HalfHessianSize = static_cast<int>(DiscreteShellBending::HalfHessianSize);

using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

/// The reference's `struct Vector2iHash` (identical body to the `std::hash`
/// specialization the reference's elastic file declares).
struct Vector2iHash
{
    size_t operator()(const Vector2i& v) const
    {
        size_t front = v[0];
        size_t end   = v[1];
        return front << 32 | end;
    }
};

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `DiscreteShellBending_do_compute_energy_kernel`
auto& dsb_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector4i>,  // stencils
                                   lcrt::Buffer<double>,      // bending_stiffnesses
                                   lcrt::Buffer<double>,      // theta_bars
                                   lcrt::Buffer<double>,      // h_bars
                                   lcrt::Buffer<double>,      // V_bars
                                   lcrt::Buffer<double>,      // L0s
                                   lcrt::Buffer<LVector3>,    // xs
                                   lcrt::Buffer<double>,      // energies
                                   double,                    // dt
                                   uint32_t>(                 // n
            [](lcrt::BufferVar<LVector4i> stencils,
               lcrt::BufferVar<double>    bending_stiffnesses,
               lcrt::BufferVar<double>    theta_bars,
               lcrt::BufferVar<double>    h_bars,
               lcrt::BufferVar<double>    V_bars,
               lcrt::BufferVar<double>    L0s,
               lcrt::BufferVar<LVector3>  xs,
               lcrt::BufferVar<double>    energies,
               lcrt::Var<double>          dt,
               lcrt::Var<uint32_t>        n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto stencil   = load_vec4i(stencils->read(i));
                    auto kappa     = bending_stiffnesses->read(i);
                    auto L0        = L0s->read(i);
                    auto h_bar     = h_bars->read(i);
                    auto theta_bar = theta_bars->read(i);
                    auto V_bar     = V_bars->read(i);

                    auto x0 = load_vec3(xs->read(stencil(0)));
                    auto x1 = load_vec3(xs->read(stencil(1)));
                    auto x2 = load_vec3(xs->read(stencil(2)));
                    auto x3 = load_vec3(xs->read(stencil(3)));

                    auto E = dsb_detail::stencil_energy<lcrt::Var<double>>(
                        x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, V_bar, dt);
                    energies->write(i, E);
                };
            });
    return k;
}

/// `DiscreteShellBending_do_compute_gradient_hessian_kernel`
auto& dsb_gradient_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector4i>,  // stencils
                                   lcrt::Buffer<double>,      // bending_stiffnesses
                                   lcrt::Buffer<double>,      // theta_bars
                                   lcrt::Buffer<double>,      // h_bars
                                   lcrt::Buffer<double>,      // V_bars
                                   lcrt::Buffer<double>,      // L0s
                                   lcrt::Buffer<LVector3>,    // xs
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
                                   uint32_t, // gradient_only
                                   uint32_t>(// n
            [](lcrt::BufferVar<LVector4i> stencils,
               lcrt::BufferVar<double>    bending_stiffnesses,
               lcrt::BufferVar<double>    theta_bars,
               lcrt::BufferVar<double>    h_bars,
               lcrt::BufferVar<double>    V_bars,
               lcrt::BufferVar<double>    L0s,
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
               lcrt::Var<uint32_t>        gradient_only,
               lcrt::Var<uint32_t>        n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto stencil   = load_vec4i(stencils->read(i));
                    auto kappa     = bending_stiffnesses->read(i);
                    auto L0        = L0s->read(i);
                    auto h_bar     = h_bars->read(i);
                    auto theta_bar = theta_bars->read(i);
                    auto V_bar     = V_bars->read(i);

                    auto x0 = load_vec3(xs->read(stencil(0)));
                    auto x1 = load_vec3(xs->read(stencil(1)));
                    auto x2 = load_vec3(xs->read(stencil(2)));
                    auto x3 = load_vec3(xs->read(stencil(3)));

                    auto G12 = dsb_detail::stencil_gradient<lcrt::Var<double>>(
                        x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, V_bar, dt);

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
                    DVA.segment<StencilSize>(i * StencilSize).write(stencil, G12);

                    // reference: `if(gradient_only) return;`
                    $if(gradient_only == 0u)
                    {
                        auto H12x12 = dsb_detail::stencil_hessian<lcrt::Var<double>>(
                            x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, V_bar, dt);

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
                        TMA.half_block<StencilSize>(i * HalfHessianSize).write(stencil, H12x12);
                    };
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(DiscreteShellBending);

U64 DiscreteShellBending::get_uid() const noexcept
{
    return ConstitutionUID;
}

void DiscreteShellBending::do_build(BuildInfo& info)
{
    (void)info;
}

void DiscreteShellBending::do_init(FilteredInfo& info)
{
    namespace DSB = sym::discrete_shell_bending;

    using ForEachInfo = FiniteElementMethod::ForEachInfo;
    auto geo_slots    = world().scene().geometries();

    list<Vector4i> stencil_list;
    list<Float>    bending_stiffness_list;

    // 1) Retrieve Quad Stencils
    info.for_each(  //
        geo_slots,
        [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
        {
            (void)I;
            unordered_map<Vector2i, InitInfo, Vector2iHash> stencil_map;  // Edge -> opposite vertices

            auto vertex_offset = sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
            UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
            auto vertex_offset_v = vertex_offset->view().front();

            auto edges = sc.edges().topo().view();

            for(auto&& [i, e] : enumerate(edges))
            {
                Vector2i E = e;
                std::sort(E.begin(), E.end());

                stencil_map[E].edge_index = static_cast<IndexT>(i);
            }

            auto triangles = sc.triangles().topo().view();
            for(auto&& t : triangles)
            {
                Vector3i T = t;
                std::sort(T.begin(), T.end());

                Vector2i E01 = {T[0], T[1]};
                Vector2i E02 = {T[0], T[2]};
                Vector2i E12 = {T[1], T[2]};

                // insert opposite vertices
                stencil_map[E01].oppo_verts.insert(T[2]);
                stencil_map[E02].oppo_verts.insert(T[1]);
                stencil_map[E12].oppo_verts.insert(T[0]);
            }

            auto bending_stiffnesses = sc.edges().find<Float>("bending_stiffness");
            UIPC_ASSERT(bending_stiffnesses, "Bending stiffness not found, why?");
            auto bs_view = bending_stiffnesses->view();

            for(auto&& [E, stencil_info] : stencil_map)
            {
                if(stencil_info.valid_bending())
                {
                    // X0, X1, X2, X3; (X1, X2) is middle edge
                    Vector4i stencil{*stencil_info.oppo_verts.begin(),    // X0
                                     E(0),                                // X1
                                     E(1),                                // X2
                                     *stencil_info.oppo_verts.rbegin()};  // X3

                    // convert to fem vertex index
                    stencil_list.push_back(stencil.array() + vertex_offset_v);

                    Float bs = bs_view[stencil_info.edge_index];
                    bending_stiffness_list.push_back(bs);
                }
            }
        });

    // 2) Setup Invariant Data
    h_stencils.resize(stencil_list.size());
    h_bending_stiffness.resize(stencil_list.size());
    std::ranges::move(stencil_list, h_stencils.begin());
    std::ranges::move(bending_stiffness_list, h_bending_stiffness.begin());

    // 3) Setup Related Data
    span x_bars      = info.rest_positions();
    span thicknesses = info.thicknesses();
    h_rest_lengths.resize(h_stencils.size());
    h_h_bars.resize(h_stencils.size());
    h_theta_bars.resize(h_stencils.size());
    h_V_bars.resize(h_stencils.size());

    for(auto&& [i, stencil] : enumerate(h_stencils))
    {
        // the generic reference routine works on the numerics layer's LVec
        // (byte-identical to the reference's `Vector3`)
        const auto X0 = to_luisa(x_bars[stencil[0]]);
        const auto X1 = to_luisa(x_bars[stencil[1]]);
        const auto X2 = to_luisa(x_bars[stencil[2]]);
        const auto X3 = to_luisa(x_bars[stencil[3]]);
        const auto thickness0 = thicknesses[stencil[0]];
        const auto thickness1 = thicknesses[stencil[1]];
        const auto thickness2 = thicknesses[stencil[2]];
        const auto thickness3 = thicknesses[stencil[3]];

        Float L0, V_bar, h_bar, theta_bar;
        DSB::compute_constants(L0,
                               h_bar,
                               theta_bar,
                               V_bar,  //
                               X0,
                               X1,
                               X2,
                               X3,  //
                               thickness0,
                               thickness1,
                               thickness2,
                               thickness3);


        h_rest_lengths[i] = L0;
        h_h_bars[i]       = h_bar;
        h_theta_bars[i]   = theta_bar;
        h_V_bars[i]       = V_bar;
    }

    // 4) Copy to Device
    // host `Vector4i` (Eigen) -> the device element type `LVector4i`
    copy_host_to_device(stencils, h_stencils);

    bending_stiffnesses.resize(h_bending_stiffness.size());
    bending_stiffnesses.view().copy_from(span<const Float>{h_bending_stiffness});

    rest_lengths.resize(h_rest_lengths.size());
    rest_lengths.view().copy_from(span<const Float>{h_rest_lengths});

    h_bars.resize(h_h_bars.size());
    h_bars.view().copy_from(span<const Float>{h_h_bars});

    theta_bars.resize(h_theta_bars.size());
    theta_bars.view().copy_from(span<const Float>{h_theta_bars});

    V_bars.resize(h_V_bars.size());
    V_bars.view().copy_from(span<const Float>{h_V_bars});
}

void DiscreteShellBending::do_report_extent(ReportExtentInfo& info)
{
    info.energy_count(stencils.size());  // Each quad has 1 energy
    info.gradient_count(stencils.size() * StencilSize);  // Each quad has 4 vertices

    if(info.gradient_only())
        return;

    info.hessian_count(stencils.size() * HalfHessianSize);
}

void DiscreteShellBending::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.energies().size());
    if(n > 0)
    {
        auto& k = dsb_energy_kernel();
        frame_stream().record(k(stencils.cview(),
                                bending_stiffnesses.cview(),
                                theta_bars.cview(),
                                h_bars.cview(),
                                V_bars.cview(),
                                rest_lengths.cview(),
                                info.xs(),
                                info.energies(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}

void DiscreteShellBending::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(stencils.size());
    if(n == 0)
        return;

    auto gradients = info.gradients();

    // The views' handles are pre-offset to this constitution's slice: LC binds a
    // subview at `base + offset_bytes`, so the device view inside the kernel can
    // start at its own index 0.
    const auto g_offset = static_cast<size_t>(gradients.doublet_index_offset());
    const auto g_count  = static_cast<size_t>(gradients.doublet_count());
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

    auto& k = dsb_gradient_hessian_kernel();
    frame_stream()
        .record(k(stencils.cview(),
                  bending_stiffnesses.cview(),
                  theta_bars.cview(),
                  h_bars.cview(),
                  V_bars.cview(),
                  rest_lengths.cview(),
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
                  gradient_only ? uint32_t{1} : uint32_t{0},
                  n)
                  .dispatch(n));
}
}  // namespace uipc::backend::luisa
