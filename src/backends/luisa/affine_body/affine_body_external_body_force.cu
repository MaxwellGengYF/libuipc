#include <affine_body/affine_body_external_force_reporter.h>
#include <affine_body/constraints/affine_body_external_body_force_constraint.h>
#include <affine_body/affine_body_dynamics.h>
#include <luisa/dsl/syntax.h>

namespace uipc::backend::luisa
{
/**
 * @brief Get external forces from ExternalForceConstraint and apply them to Affine Bodies
 *
 * This reporter add forces to Affine Bodies in the AffineBodyDynamics system.
 *
 * This is the "body force" implementation - forces are applied directly to bodies.
 * Future implementations like AffineBodyExternalVertexForce may apply forces to vertices.
 */
class AffineBodyExternalBodyForce final : public AffineBodyExternalForceReporter
{
  public:
    static constexpr U64 UID = 666;  // Same UID as ExternalForceConstraint

    using AffineBodyExternalForceReporter::AffineBodyExternalForceReporter;

    SimSystemSlot<AffineBodyDynamics>                    affine_body_dynamics;
    SimSystemSlot<AffineBodyExternalBodyForceConstraint> constraint;

    virtual void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = require<AffineBodyDynamics>();
        constraint           = require<AffineBodyExternalBodyForceConstraint>();
    }

    U64 get_uid() const noexcept override { return UID; }

    void do_init() override
    {
        // Nothing to do
    }

    void do_step(ExternalForceInfo& info) override
    {
        SizeT force_count = constraint->forces().size();

        auto& device = engine().luisa_device();
        auto& stream = engine().compute_stream();

        // Create kernel for scattering external forces to bodies
        Kernel1D scatter_forces_kernel = [&](BufferVar<Vector12> forces,
                                              BufferVar<IndexT> body_ids,
                                              BufferVar<Vector12> body_forces) noexcept
        {
            auto i = dispatch_id().x;
            if_(i < force_count, [&] {
                // Scatter add the external forces to the corresponding bodies
                auto body_id = body_ids.read(i);
                auto force = body_forces.read(i);
                
                // Atomic add to the forces buffer
                forces.atomic(body_id).fetch_add(force);
            });
        };

        auto shader = device.compile(scatter_forces_kernel);
        stream << shader(info.external_forces(), 
                         constraint->body_ids(),
                         constraint->forces()).dispatch(force_count);
    }
};

REGISTER_SIM_SYSTEM(AffineBodyExternalBodyForce);
}  // namespace uipc::backend::luisa
