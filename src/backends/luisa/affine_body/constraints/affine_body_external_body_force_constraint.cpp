#include <affine_body/constraints/affine_body_external_body_force_constraint.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>

/***********************************************************************************************
* T6.4b — AffineBodyExternalBodyForceConstraint (port of
* src/backends/cuda/affine_body/constraints/affine_body_external_body_force_constraint.cu).
*
* `Impl::step` is the reference's body verbatim:
*   * per geometry, read `builtin::backend_abd_body_offset` and the per-instance
*     `builtin::is_constrained` / `external_force` (**Vector12**) attributes;
*   * keep the marked entries as (global body id, generalized force) pairs.
* The reference's `if(!h_forces.empty())` guard around the two uploads is kept as-is
* (the LC `copy_from` of a zero-size span is not a no-op -- the T6.3b1 port-necessity
* note), and the host `Vector12` (Eigen) -> `LVector12` crossing is element-wise
* (the T6.3b2 rule-15 trap: the ABD header's `to_luisa(Vector12)` is invisible to the
* global `copy_host_to_device`'s dependent unqualified call).
*
* `do_report_extent` declares 0/0/0, which is the reference's contract (the forces are
* applied through the reporter into `body_id_to_external_force`, not through the linear
* system); `do_compute_energy` and `do_compute_gradient_hessian` are empty for the same
* reason.
***********************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(AffineBodyExternalBodyForceConstraint);

void AffineBodyExternalBodyForceConstraint::do_build(BuildInfo& info)
{
    (void)info;
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
}

U64 AffineBodyExternalBodyForceConstraint::get_uid() const noexcept
{
    return UID;
}

void AffineBodyExternalBodyForceConstraint::do_init(AffineBodyAnimator::FilteredInfo& info)
{
    // Initial read of external forces
    do_step(info);
}

lud::LCBufferView<LVector12> AffineBodyExternalBodyForceConstraint::forces() const noexcept
{
    return m_impl.forces.cview();
}

lud::LCBufferView<IndexT> AffineBodyExternalBodyForceConstraint::body_ids() const noexcept
{
    return m_impl.body_ids.cview();
}

void AffineBodyExternalBodyForceConstraint::Impl::step(backend::WorldVisitor& world,
                                                       AffineBodyAnimator::FilteredInfo& info)
{
    // Clear host buffers
    h_forces.clear();
    h_body_ids.clear();
    // Read external forces from geometry attributes
    auto   geo_slots           = world.scene().geometries();
    IndexT current_body_offset = 0;
    info.for_each(
        geo_slots,
        [&](geometry::SimplicialComplex& sc)
        {
            auto body_offset = sc.meta().find<IndexT>(builtin::backend_abd_body_offset);
            UIPC_ASSERT(body_offset,
                        "`backend_abd_body_offset` attribute not found in geometry simplicial complex");
            current_body_offset = body_offset->view().front();
            auto is_constrained = sc.instances().find<IndexT>(builtin::is_constrained);
            UIPC_ASSERT(is_constrained,
                        "`is_constrained` attribute not found in geometry simplicial complex");
            auto external_force = sc.instances().find<Vector12>("external_force");
            UIPC_ASSERT(external_force,
                        "`external_force` attribute not found in geometry simplicial complex");
            return zip(is_constrained->view(), external_force->view());
        },
        [&](const AffineBodyDynamics::ForEachInfo& I, auto&& values)
        {
            SizeT body_id = I.local_index() + current_body_offset;
            auto&& [is_constrained, force] = values;
            if(is_constrained)
            {
                h_forces.push_back(force);
                h_body_ids.push_back(static_cast<IndexT>(body_id));
            }
        });

    // Copy from host to device
    //
    // REFERENCE QUIRK (transcribed, not fixed, cookbook "transcribe + pin"): the guard
    // is the reference's own, so a frame whose host table is *empty* leaves the device
    // buffers at their previous contents.  That is a no-op in the reference (its
    // `copy_from` of an empty cuda_tool buffer does nothing and the reporter's `n` is
    // the *device* buffer's size, which therefore keeps the stale count and re-applies
    // the previous frame's forces).  The FEM sibling (uid 671,
    // finite_element_external_vertex_force_constraint.cpp) deliberately fixed exactly
    // this by always resizing; the port does NOT import that fix here -- the ABD
    // behaviour is the reference's, and this batch's test drives the table explicitly.
    if(!h_forces.empty())
    {
        // the host `Vector12` (Eigen) -> device `LVector12` crossing is element-wise
        std::vector<LVector12> staging;
        staging.reserve(h_forces.size());
        for(auto& v : h_forces)
            staging.push_back(to_luisa(v));
        forces.resize(staging.size());
        forces.view().copy_from(span<const LVector12>{staging});

        body_ids.resize(h_body_ids.size());
        body_ids.view().copy_from(span<const IndexT>{h_body_ids});
    }
}

void AffineBodyExternalBodyForceConstraint::do_step(AffineBodyAnimator::FilteredInfo& info)
{
    m_impl.step(world(), info);
}

// No energy/gradient/hessian for external forces (they are applied directly in do_predict_dof)
void AffineBodyExternalBodyForceConstraint::do_report_extent(
    AffineBodyAnimator::ReportExtentInfo& info)
{
    // No contribution to energy/gradient/hessian
    info.energy_count(0);
    info.gradient_count(0);
    info.hessian_count(0);
}

void AffineBodyExternalBodyForceConstraint::do_compute_energy(
    AffineBodyAnimator::ComputeEnergyInfo& info)
{
    (void)info;
    // No energy computation
}

void AffineBodyExternalBodyForceConstraint::do_compute_gradient_hessian(
    AffineBodyAnimator::ComputeGradientHessianInfo& info)
{
    (void)info;
    // No gradient/hessian computation
}
}  // namespace uipc::backend::luisa
