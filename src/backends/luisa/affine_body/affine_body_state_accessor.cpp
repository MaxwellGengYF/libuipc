#include <sim_system.h>
#include <affine_body/affine_body_state_accessor_feature.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <joint_dof_system/global_joint_dof_manager.h>

/********************************************************************
 * T6.1a — AffineBodyStateAccessor (reference:
 * src/backends/cuda/affine_body/affine_body_state_accessor.cu).
 *
 * A SimSystem whose only job is to add the AffineBodyStateAccessorFeature to the
 * engine at build time.  `require<GlobalJointDofManager>()` is kept verbatim: the
 * ABD state accessor must re-sync the joint DOF attributes after a `copy_from`
 * (the joint layer is T6.3b/T6.4), and a missing joint manager invalidates the
 * system rather than silently skipping the re-sync.
 *********************************************************************/
namespace uipc::backend::luisa
{
// A SimSystem to add AffineBodyStateAccessorFeature to the engine
class AffineBodyStateAccessor final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    virtual void do_build() override
    {
        auto& affine_body_dynamics        = require<AffineBodyDynamics>();
        auto& affine_body_vertex_reporter = require<AffineBodyVertexReporter>();
        auto& global_joint_dof_manager    = require<GlobalJointDofManager>();

        // Register the AffineBodyStateAccessorFeature
        auto overrider = std::make_shared<AffineBodyStateAccessorFeatureOverrider>(
            affine_body_dynamics, affine_body_vertex_reporter, global_joint_dof_manager);
        auto feature = std::make_shared<core::AffineBodyStateAccessorFeature>(overrider);
        features().insert(feature);
    }
};

REGISTER_SIM_SYSTEM(AffineBodyStateAccessor);
}  // namespace uipc::backend::luisa
