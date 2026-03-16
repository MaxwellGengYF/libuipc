#include <affine_body/constraints/affine_body_external_body_force_constraint.h>
#include <affine_body/affine_body_constraint.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>
#include <uipc/builtin/attribute_name.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class AffineBodyExternalBodyForceConstraint final : public AffineBodyConstraint
{
  public:
    using AffineBodyConstraint::AffineBodyConstraint;

    static constexpr U64 UID = 666;  // Same UID as AffineBodyExternalForce

    class Impl
    {
      public:
        // Host-side buffers for collecting data before copying to device
        vector<Vector12>             h_forces;
        vector<IndexT>               h_body_ids;
        luisa::compute::Buffer<Vector12> forces;
        luisa::compute::Buffer<IndexT>   body_ids;
        AffineBodyDynamics*          affine_body_dynamics = nullptr;

        void step(backend::WorldVisitor& world, AffineBodyAnimator::FilteredInfo& info);
    };

    luisa::compute::BufferView<const Vector12> forces() const noexcept;
    luisa::compute::BufferView<const IndexT>   body_ids() const noexcept;

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual U64  get_uid() const noexcept override;
    virtual void do_init(AffineBodyAnimator::FilteredInfo& info) override;
    virtual void do_step(AffineBodyAnimator::FilteredInfo& info) override;
    virtual void do_report_extent(AffineBodyAnimator::ReportExtentInfo& info) override;
    virtual void do_compute_energy(AffineBodyAnimator::ComputeEnergyInfo& info) override;
    virtual void do_compute_gradient_hessian(AffineBodyAnimator::ComputeGradientHessianInfo& info) override;

    Impl m_impl;
};

REGISTER_SIM_SYSTEM(AffineBodyExternalBodyForceConstraint);

void AffineBodyExternalBodyForceConstraint::do_build(BuildInfo& info)
{
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

luisa::compute::BufferView<const Vector12> AffineBodyExternalBodyForceConstraint::forces() const noexcept
{
    return m_impl.forces.view();
}

luisa::compute::BufferView<const IndexT> AffineBodyExternalBodyForceConstraint::body_ids() const noexcept
{
    return m_impl.body_ids.view();
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
            UIPC_ASSERT(body_offset, "`backend_abd_body_offset` attribute not found in geometry simplicial complex");
            current_body_offset = body_offset->view().front();
            auto is_constrained = sc.instances().find<IndexT>(builtin::is_constrained);
            UIPC_ASSERT(is_constrained, "`is_constrained` attribute not found in geometry simplicial complex");
            auto external_force = sc.instances().find<Vector12>("external_force");
            UIPC_ASSERT(external_force, "`external_force` attribute not found in geometry simplicial complex");
            return zip(is_constrained->view(), external_force->view());
        },
        [&](const AffineBodyDynamics::ForEachInfo& I, auto&& values)
        {
            SizeT body_id = I.local_index() + current_body_offset;
            auto&& [is_constrained, force] = values;
            if(is_constrained)
            {
                h_forces.push_back(force);
                h_body_ids.push_back(body_id);
            }
        });

    // Copy from host to device
    auto& device = static_cast<SimEngine&>(world.sim_engine()).device();
    if(!h_forces.empty())
    {
        forces = device.create_buffer<Vector12>(h_forces.size());
        body_ids = device.create_buffer<IndexT>(h_body_ids.size());
        forces.view().copy_from(h_forces.data());
        body_ids.view().copy_from(h_body_ids.data());
    }
}

void AffineBodyExternalBodyForceConstraint::do_step(AffineBodyAnimator::FilteredInfo& info)
{
    m_impl.step(world(), info);
}

// No energy/gradient/hessian for external forces (they are applied directly in do_predict_dof)
void AffineBodyExternalBodyForceConstraint::do_report_extent(AffineBodyAnimator::ReportExtentInfo& info)
{
    // No contribution to energy/gradient/hessian
    info.energy_count(0);
    info.gradient_count(0);
    info.hessian_count(0);
}

void AffineBodyExternalBodyForceConstraint::do_compute_energy(AffineBodyAnimator::ComputeEnergyInfo& info)
{
    // No energy computation
}

void AffineBodyExternalBodyForceConstraint::do_compute_gradient_hessian(
    AffineBodyAnimator::ComputeGradientHessianInfo& info)
{
    // No gradient/hessian computation
}
}  // namespace uipc::backend::luisa
