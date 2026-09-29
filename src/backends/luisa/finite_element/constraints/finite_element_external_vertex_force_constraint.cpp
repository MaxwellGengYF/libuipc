#include <finite_element/constraints/finite_element_external_vertex_force_constraint.h>
#include <finite_element/finite_element_method.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>
#include <uipc/builtin/attribute_name.h>

/*************************************************************************************************
* T5.4b — FiniteElementExternalVertexForceConstraint (port of
* src/backends/cuda/finite_element/constraints/finite_element_external_vertex_force_constraint.cu).
*
* `Impl::step` is the reference's body verbatim:
*   * per geometry, read `builtin::backend_fem_vertex_offset` and the per-vertex
*     `builtin::is_constrained` / `external_force` attributes;
*   * keep the marked entries as (global vertex id, force) pairs;
*   * **always** resize + re-upload the device buffers, even when empty — the
*     reference's own comment: without it, stale forces from a previous frame
*     persist in the device buffer and get scatter-added every advance() forever.
*
* Host-vector element type: the reference's `vector<Vector3>` (Eigen) maps onto the
* device `LVector3` element type with the explicit element-wise conversion
* (`copy_host_to_device`, type_define.h — identical bytes, unrelated types).
*
* `do_report_extent` declares 0/0/0, which is the reference's contract (forces are
* applied through x_tilde, not through the linear system); `do_compute_energy` and
* `do_compute_gradient_hessian` are empty for the same reason.
*************************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(FiniteElementExternalVertexForceConstraint);

void FiniteElementExternalVertexForceConstraint::do_build(BuildInfo& info)
{
    (void)info;
}

U64 FiniteElementExternalVertexForceConstraint::get_uid() const noexcept
{
    return UID;
}

void FiniteElementExternalVertexForceConstraint::do_init(FiniteElementAnimator::FilteredInfo& info)
{
    do_step(info);
}

lud::LCBufferView<LVector3> FiniteElementExternalVertexForceConstraint::forces() const noexcept
{
    return m_impl.forces.cview();
}

lud::LCBufferView<IndexT> FiniteElementExternalVertexForceConstraint::vertex_ids() const noexcept
{
    return m_impl.vertex_ids.cview();
}

void FiniteElementExternalVertexForceConstraint::Impl::step(backend::WorldVisitor& world,
                                                            FiniteElementAnimator::FilteredInfo& info)
{
    using ForEachInfo = FiniteElementMethod::ForEachInfo;

    h_forces.clear();
    h_vertex_ids.clear();

    auto   geo_slots             = world.scene().geometries();
    IndexT current_vertex_offset = 0;

    info.for_each(
        geo_slots,
        [&](geometry::SimplicialComplex& sc)
        {
            auto vertex_offset = sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
            UIPC_ASSERT(vertex_offset, "`backend_fem_vertex_offset` attribute not found");
            current_vertex_offset = vertex_offset->view().front();

            auto is_constrained = sc.vertices().find<IndexT>(builtin::is_constrained);
            UIPC_ASSERT(is_constrained, "`is_constrained` attribute not found");
            auto external_force = sc.vertices().find<Vector3>("external_force");
            UIPC_ASSERT(external_force, "`external_force` attribute not found");

            return zip(is_constrained->view(), external_force->view());
        },
        [&](const ForEachInfo& I, auto&& values)
        {
            auto vI = I.local_index() + current_vertex_offset;
            auto&& [is_constrained, force] = values;
            if(is_constrained)
            {
                h_vertex_ids.push_back(vI);
                h_forces.push_back(force);
            }
        });

    // Always sync device buffers — even when empty.
    // Without this, stale forces from a previous frame persist in the device
    // buffer and get scatter-added every advance() call forever.
    vertex_ids.resize(h_vertex_ids.size());
    if(!h_vertex_ids.empty())
    {
        vertex_ids.view().copy_from(span<const IndexT>{h_vertex_ids});
    }
    // the host `Vector3` (Eigen) -> device `LVector3` crossing is element-wise
    copy_host_to_device(forces, h_forces);
}

void FiniteElementExternalVertexForceConstraint::do_step(FiniteElementAnimator::FilteredInfo& info)
{
    m_impl.step(world(), info);
}

void FiniteElementExternalVertexForceConstraint::do_report_extent(
    FiniteElementAnimator::ReportExtentInfo& info)
{
    info.energy_count(0);
    info.gradient_count(0);
    info.hessian_count(0);
}

void FiniteElementExternalVertexForceConstraint::do_compute_energy(
    FiniteElementAnimator::ComputeEnergyInfo& info)
{
    (void)info;
}

void FiniteElementExternalVertexForceConstraint::do_compute_gradient_hessian(
    FiniteElementAnimator::ComputeGradientHessianInfo& info)
{
    (void)info;
}
}  // namespace uipc::backend::luisa
