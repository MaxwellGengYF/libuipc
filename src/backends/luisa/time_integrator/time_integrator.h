#pragma once
/********************************************************************
 * @file time_integrator.h
 * @brief Port of src/backends/cuda/time_integrator/time_integrator.h (T4.2).
 *
 * Identical shape: the base class resolves the TimeIntegratorManager, calls the
 * subclass Build hook and registers itself; the manager drives init() once and
 * predict_dof()/update_state() once per frame, forwarding the scene `dt`.
 *********************************************************************/
#include <sim_system.h>
#include <time_integrator/time_integrator_manager.h>

namespace uipc::backend::luisa
{
class TimeIntegrator : public SimSystem
{
  public:
    using PredictDofInfo     = TimeIntegratorManager::PredictDofInfo;
    using UpdateVelocityInfo = TimeIntegratorManager::UpdateVelocityInfo;

    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

  protected:
    virtual void do_init(InitInfo& info)            = 0;
    virtual void do_build(BuildInfo& info)          = 0;
    virtual void do_predict_dof(PredictDofInfo& info) = 0;
    virtual void do_update_state(UpdateVelocityInfo& info) = 0;

  private:
    friend class TimeIntegratorManager;
    void do_build() override final;
    void init();
    void predict_dof(PredictDofInfo&);
    void update_state(UpdateVelocityInfo&);
};
}  // namespace uipc::backend::luisa
