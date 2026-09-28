#pragma once
/********************************************************************
 * @file joint_dof_reporter.h
 * @brief Port of src/backends/cuda/joint_dof_system/joint_dof_reporter.h
 *        (T4.2). Identical shape; the only change is the namespace.
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
class GlobalJointDofManager;

/**
 * @brief Abstract reporter that any joint-like SimSystem can inherit to
 * participate in the GlobalJointDofManager's update-dof-attributes protocol.
 *
 * A joint typically maintains persistent DOF state (e.g. a revolute joint
 * tracks `current_angles`). When the underlying geometry is reset or
 * topologically changed, those persistent states must be updated to match
 * the new attributes before the next frame — see GlobalVertexManager's
 * `require_discard_friction` for the analogous pattern on friction.
 */
class JointDofReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
        // reserved for future extensibility
    };

    class UpdateDofAttributesInfo
    {
        // reserved for future extensibility
    };

  protected:
    virtual void do_build(BuildInfo& info)                     = 0;
    virtual void do_update_dof_attributes(UpdateDofAttributesInfo& info) = 0;

  private:
    virtual void do_build() override final;

    friend class GlobalJointDofManager;
    void update_dof_attributes(UpdateDofAttributesInfo& info);
};
}  // namespace uipc::backend::luisa
