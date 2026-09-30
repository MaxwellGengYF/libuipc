#pragma once
/********************************************************************
 * @file adaptive_contact_parameter_reporter.h
 * @brief Base of the adaptive-contact-parameter reporters (port of
 * src/backends/cuda/contact_system/adaptive_contact_parameter_reporter.h, 35 lines).
 *
 * There is at most **one** such reporter per scene (`GlobalContactManager::add_reporter`
 * asserts it), and only in a scene that opts into adaptive kappa (some contact model
 * with `resistance <= 0`); `do_build` throws `SimSystemException` otherwise — which is
 * exactly how the engine invalidates `GIPCAdaptiveParameterStrategy` in a non-adaptive
 * scene.  `AdaptiveParameterInfo` aliases the manager's nested type.
 *********************************************************************/
#include <sim_system.h>
#include <contact_system/global_contact_manager.h>

namespace uipc::backend::luisa
{
class AdaptiveContactParameterReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class InitInfo
    {
      public:
    };

    class BuildInfo
    {
      public:
    };

    using AdaptiveParameterInfo = GlobalContactManager::AdaptiveParameterInfo;

  protected:
    virtual void do_init(InitInfo& info)                            = 0;
    virtual void do_build(BuildInfo& info)                          = 0;
    virtual void do_compute_parameters(AdaptiveParameterInfo& info) = 0;

  private:
    virtual void do_build() override;

    friend class GlobalContactManager;
    void init();
    void compute_parameters(AdaptiveParameterInfo& info);
};
}  // namespace uipc::backend::luisa
