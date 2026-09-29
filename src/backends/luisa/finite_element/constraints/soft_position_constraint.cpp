#include <finite_element/constraints/soft_position_constraint.h>
#include <uipc/builtin/attribute_name.h>
#include <frame_stream.h>
#include <uipc/common/zip.h>
#include <utils/scalar_util.h>

/***************************************************************
 * T5.3 (pulled forward from T5.4) — SoftPositionConstraint
 * (reference:
 * src/backends/cuda/finite_element/constraints/soft_position_constraint.cu).
 *
 * Port notes:
 *  * `lerp(x_prev, aim_positions(I), substep_ratio)` becomes
 *    `animator::lerp(...)`: the reference's unqualified `lerp` is
 *    `cuda/animator/utils.h`'s clamped law, which the port had to move into
 *    `uipc::backend::luisa::animator` because `lps::lerp` (unclamped) is found by
 *    ADL for the same arguments and wins overload resolution (cookbook §7.14
 *    rule 4).
 *  * The doublet/triplet views cannot be kernel arguments, so their flat buffers
 *    are passed pre-offset to this constraint's slice and the device views are
 *    rebuilt inside the kernel with `from_handles_full`; the
 *    `subvector_offset`/`submatrix_offset` travel as scalars (cookbook §7.17
 *    rule 2).  The device views' `write` takes the numerics layer's `LVec`/`LMat`
 *    by value, so `G3s(I).write(i, G)` / `H3x3s(I).write(i, i, H)` are literally
 *    the reference's statements (segment index = the *global* vertex index `i`,
 *    view index = the constraint-local index `I`).
 *  * `is_fixed(i)` is device data, so the reference's three `if`s become
 *    `select`s of the already computed candidates (RULE M2) -- for the Hessian
 *    this is the *guarded* form, unlike the kinetic systems'.
 *  * `constraint_strength_ratio` is the `"strength_ratio"` vertex attribute
 *    (a plain string name in the reference, not a `builtin::` one).
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

/// `SoftPositionConstraint_do_compute_energy_kernel`
auto& soft_position_constraint_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,   // indices
                                                lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // x_prevs
                                                lcrt::Buffer<LVector3>,  // aim_positions
                                                lcrt::Buffer<double>,    // strength_ratios
                                                lcrt::Buffer<double>,    // masses
                                                lcrt::Buffer<double>,    // energies
                                                lcrt::Buffer<int32_t>,   // is_fixed
                                                double,                  // substep_ratio
                                                uint32_t>(               // n
        [](lcrt::BufferVar<int32_t>  indices,
           lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> x_prevs,
           lcrt::BufferVar<LVector3> aim_positions,
           lcrt::BufferVar<double>   strength_ratios,
           lcrt::BufferVar<double>   masses,
           lcrt::BufferVar<double>   energies,
           lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::Var<double>         substep_ratio,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto i = indices->read(I);

                auto x      = load_vec3(xs->read(i));
                auto x_prev = load_vec3(x_prevs->read(i));
                auto aim_x  = animator::lerp(x_prev, load_vec3(aim_positions->read(I)), substep_ratio);
                auto m      = masses->read(i);
                auto s      = strength_ratios->read(I);
                auto dx     = x - aim_x;

                auto E = scalar_detail::select_scalar(
                    lps::detail::mul(lps::detail::mul(lps::detail::mul(0.5, s), m),
                                     lps::dot(dx, dx)),
                    lcrt::Var<double>{0.0},
                    is_fixed->read(i) != 0);

                energies->write(I, E);
            };
        });
    return k;
}

/// `SoftPositionConstraint_do_compute_gradient_hessian_k1_kernel`
auto& soft_position_constraint_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,   // indices
                                                lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // x_prevs
                                                lcrt::Buffer<LVector3>,  // aim_positions
                                                lcrt::Buffer<double>,    // strength_ratios
                                                lcrt::Buffer<double>,    // masses
                                                // gradients (this constraint's slice)
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<double>,
                                                int32_t,  // subvector_offset
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                double,                 // substep_ratio
                                                uint32_t>(              // n
        [](lcrt::BufferVar<int32_t>  indices,
           lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> x_prevs,
           lcrt::BufferVar<LVector3> aim_positions,
           lcrt::BufferVar<double>   strength_ratios,
           lcrt::BufferVar<double>   masses,
           lcrt::BufferVar<int32_t>  g_indices,
           lcrt::BufferVar<double>   g_values,
           lcrt::Var<int32_t>        g_subvector_offset,
           lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::Var<double>         substep_ratio,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto i = indices->read(I);

                auto x      = load_vec3(xs->read(i));
                auto x_prev = load_vec3(x_prevs->read(i));
                auto aim_x  = animator::lerp(x_prev, load_vec3(aim_positions->read(I)), substep_ratio);
                auto m      = masses->read(i);
                auto s      = strength_ratios->read(I);
                auto dx     = x - aim_x;

                auto zero = lps::make_vec<lcrt::Var<double>, 3>(
                    [](int) noexcept { return lcrt::Var<double>{0.0}; });
                auto G = scalar_detail::select_container(
                    lps::detail::mul(s, m) * dx, zero, is_fixed->read(i) != 0);

                auto ii = lcrt::cast<std::int32_t>(I);
                auto G3s = DoubletDeviceView::from_handles_full(g_indices,
                                                                g_values,
                                                                n,           // total_segments
                                                                int32_t{0},  // doublet_index_offset
                                                                n,           // doublet_count
                                                                n,           // total_doublet_count
                                                                g_subvector_offset,
                                                                n);          // subvector_extent
                G3s(ii).write(i, G);
            };
        });
    return k;
}

/// `SoftPositionConstraint_do_compute_gradient_hessian_k2_kernel`
auto& soft_position_constraint_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // indices
                                                lcrt::Buffer<double>,   // strength_ratios
                                                lcrt::Buffer<double>,   // masses
                                                // hessians (this constraint's slice)
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<double>,
                                                int32_t,  // submatrix_offset.x
                                                int32_t,  // submatrix_offset.y
                                                int32_t,  // submatrix_extent.x
                                                int32_t,  // submatrix_extent.y
                                                int32_t,  // total_block_rows
                                                int32_t,  // total_block_cols
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t>(              // n
        [](lcrt::BufferVar<int32_t>  indices,
           lcrt::BufferVar<double>   strength_ratios,
           lcrt::BufferVar<double>   masses,
           lcrt::BufferVar<int32_t>  h_row_indices,
           lcrt::BufferVar<int32_t>  h_col_indices,
           lcrt::BufferVar<double>   h_values,
           lcrt::Var<int32_t>        h_submatrix_offset_x,
           lcrt::Var<int32_t>        h_submatrix_offset_y,
           lcrt::Var<int32_t>        h_submatrix_extent_x,
           lcrt::Var<int32_t>        h_submatrix_extent_y,
           lcrt::Var<int32_t>        h_total_block_rows,
           lcrt::Var<int32_t>        h_total_block_cols,
           lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto i = indices->read(I);
                auto m = masses->read(i);
                auto s = strength_ratios->read(I);

                auto sm   = lps::detail::mul(s, m);
                auto zero = lps::make_mat<lcrt::Var<double>, 3, 3>(
                    [](int, int) noexcept { return lcrt::Var<double>{0.0}; });
                auto H = scalar_detail::select_container(
                    sm * lps::LMat<lcrt::Var<double>, 3, 3>::Identity(),
                    zero,
                    is_fixed->read(i) != 0);

                auto ii = lcrt::cast<std::int32_t>(I);
                auto H3x3s = TripletDeviceView::from_handles_full(h_row_indices,
                                                                  h_col_indices,
                                                                  h_values,
                                                                  h_total_block_rows,
                                                                  h_total_block_cols,
                                                                  int32_t{0},  // triplet_index_offset
                                                                  n,           // triplet_count
                                                                  n,           // total_triplet_count
                                                                  h_submatrix_offset_x,
                                                                  h_submatrix_offset_y,
                                                                  h_submatrix_extent_x,
                                                                  h_submatrix_extent_y);
                H3x3s(ii).write(i, i, H);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(SoftPositionConstraint);

void SoftPositionConstraint::do_build(BuildInfo& info)
{
    (void)info;
}

void SoftPositionConstraint::do_init(FiniteElementAnimator::FilteredInfo& info)
{
    do_step(info);  // do the same thing as do_step
}

void SoftPositionConstraint::do_step(FiniteElementAnimator::FilteredInfo& info)
{
    using ForEachInfo = FiniteElementMethod::ForEachInfo;

    auto geo_slots = world().scene().geometries();

    // clear
    h_constrained_vertices.clear();
    h_aim_positions.clear();
    h_strength_ratios.clear();

    IndexT current_vertex_offset = 0;
    info.for_each(
        geo_slots,
        [&](geometry::SimplicialComplex& sc)
        {
            auto vertex_offset =
                sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
            current_vertex_offset = vertex_offset->view().front();

            auto is_constrained = sc.vertices().find<IndexT>(builtin::is_constrained);
            auto aim_pos        = sc.vertices().find<Vector3>(builtin::aim_position);
            auto strength_ratio = sc.vertices().find<Float>("strength_ratio");

            return zip(is_constrained->view(),
                       aim_pos->view(),
                       strength_ratio->view());
        },
        [&](const ForEachInfo& I, auto&& values)
        {
            auto vI = I.local_index() + current_vertex_offset;

            auto&& [is_constrained, aim_pos, strength] = values;

            if(is_constrained)
            {
                h_constrained_vertices.push_back(static_cast<IndexT>(vI));
                h_aim_positions.push_back(aim_pos);
                h_strength_ratios.push_back(strength);
            }
        });

    auto n = h_constrained_vertices.size();
    constrained_vertices.resize(n);
    strength_ratios.resize(n);
    if(n != 0)
    {
        constrained_vertices.view().copy_from(
            span<const IndexT>{h_constrained_vertices});
        strength_ratios.view().copy_from(span<const Float>{h_strength_ratios});
    }
    // the host `Vector3` (Eigen) -> device `LVector3` crossing is element-wise
    copy_host_to_device(aim_positions, h_aim_positions);
}

void SoftPositionConstraint::do_report_extent(FiniteElementAnimator::ReportExtentInfo& info)
{
    info.energy_count(h_constrained_vertices.size());
    info.gradient_count(h_constrained_vertices.size());

    if(info.gradient_only())
        return;

    info.hessian_count(h_constrained_vertices.size());
}

void SoftPositionConstraint::do_compute_energy(FiniteElementAnimator::ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(constrained_vertices.size());
    if(n > 0)
    {
        auto& k = soft_position_constraint_energy_kernel();
        frame_stream().record(k(constrained_vertices.cview(),
                                info.xs(),
                                info.x_prevs(),
                                aim_positions.cview(),
                                strength_ratios.cview(),
                                info.masses(),
                                info.energies(),
                                info.is_fixed(),
                                info.substep_ratio(),
                                n)
                                  .dispatch(n));
    }
}

void SoftPositionConstraint::do_compute_gradient_hessian(FiniteElementAnimator::ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(constrained_vertices.size());
    if(n > 0)
    {
        auto gradients = info.gradients();
        const auto g_offset = static_cast<size_t>(gradients.doublet_index_offset());
        const auto g_count  = static_cast<size_t>(gradients.doublet_count());
        auto       g_indices = gradients.indices().subview(g_offset, g_count);
        auto       g_values  = gradients.values().subview(g_offset * 3u, g_count * 3u);

        auto& k = soft_position_constraint_gradient_kernel();
        frame_stream().record(k(constrained_vertices.cview(),
                                info.xs(),
                                info.x_prevs(),
                                aim_positions.cview(),
                                strength_ratios.cview(),
                                info.masses(),
                                g_indices,
                                g_values,
                                static_cast<int32_t>(gradients.subvector_offset()),
                                info.is_fixed(),
                                info.substep_ratio(),
                                n)
                                  .dispatch(n));
    }

    if(info.gradient_only())
        return;

    if(n > 0)
    {
        auto       hessians = info.hessians();
        const auto h_offset = static_cast<size_t>(hessians.triplet_index_offset());
        const auto h_count  = static_cast<size_t>(hessians.triplet_count());

        auto& k = soft_position_constraint_hessian_kernel();
        frame_stream().record(k(constrained_vertices.cview(),
                                strength_ratios.cview(),
                                info.masses(),
                                hessians.row_indices().subview(h_offset, h_count),
                                hessians.col_indices().subview(h_offset, h_count),
                                hessians.values().subview(h_offset * 9u, h_count * 9u),
                                static_cast<int32_t>(hessians.submatrix_offset().x),
                                static_cast<int32_t>(hessians.submatrix_offset().y),
                                static_cast<int32_t>(hessians.extent().x),
                                static_cast<int32_t>(hessians.extent().y),
                                static_cast<int32_t>(hessians.total_rows()),
                                static_cast<int32_t>(hessians.total_cols()),
                                info.is_fixed(),
                                n)
                                  .dispatch(n));
    }
}
}  // namespace uipc::backend::luisa
