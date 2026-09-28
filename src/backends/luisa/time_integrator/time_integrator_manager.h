#pragma once
/********************************************************************
 * @file time_integrator_manager.h
 * @brief Port of
 *        src/backends/cuda/time_integrator/time_integrator_manager.h (T4.2).
 *
 * Identical shape. The manager reads the scene's `dt` attribute in do_build and
 * forwards it to every registered TimeIntegrator: init() once during scene
 * initialization, predict_dof() in the PredictMotion phase (after the animator
 * and the external forces have stepped) and update_state() at the end of the
 * frame (the reference's "5. Update Velocity => v = (x - x_0) / dt").
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
class TimeIntegrator;

class TimeIntegratorManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class PredictDofInfo
    {
      public:
        auto dt() const noexcept { return m_dt; }

      private:
        friend class TimeIntegratorManager;
        Float m_dt;
    };

    class UpdateVelocityInfo
    {
      public:
        auto dt() const noexcept { return m_dt; };

      private:
        friend class TimeIntegratorManager;
        Float m_dt;
    };

    class Impl
    {
      public:
        SimSystemSlotCollection<TimeIntegrator>          time_integrators;
        S<const geometry::AttributeSlot<Float>>          dt_attr;
    };

  private:
    friend class SimEngine;
    virtual void do_build() override;
    void init();

    void predict_dof();
    void update_state();

    friend class TimeIntegrator;
    void add_integrator(TimeIntegrator* integrator);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
