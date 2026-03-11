#pragma once
#include <sim_system.h>
#include <time_integrator/time_integrator_manager.h>

namespace uipc::backend::luisa
{
/**
 * @brief Base class for time integrators in the LuisaCompute backend
 * 
 * Time integrators handle the prediction of DOF values and update of velocities
 * during the simulation loop. Different integration schemes (implicit Euler,
 * BDF2, etc.) can be implemented by subclassing this interface.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class TimeIntegrator : public SimSystem
{
  public:
    using PredictDofInfo     = TimeIntegratorManager::PredictDofInfo;
    using UpdateVelocityInfo = TimeIntegratorManager::UpdateVelocityInfo;

    using SimSystem::SimSystem;

    /**
     * @brief Build information passed to do_build()
     */
    class BuildInfo
    {
      public:
    };

    /**
     * @brief Initialization information passed to do_init()
     */
    class InitInfo
    {
      public:
    };

  protected:
    /**
     * @brief Initialize the time integrator
     * 
     * Called once during scene initialization.
     * 
     * @param info Initialization information
     */
    virtual void do_init(InitInfo& info) = 0;

    /**
     * @brief Build the time integrator
     * 
     * Called during system build to set up resources and register callbacks.
     * 
     * @param info Build information
     */
    virtual void do_build(BuildInfo& info) = 0;

    /**
     * @brief Predict the DOF values for the next time step
     * 
     * Called during each Newton iteration to compute predicted DOF values
     * based on current state and external forces.
     * 
     * @param info Prediction information including time step size
     */
    virtual void do_predict_dof(PredictDofInfo& info) = 0;

    /**
     * @brief Update velocities after the Newton solve converges
     * 
     * Called after the Newton solve to update velocities based on
     * the new DOF values.
     * 
     * @param info Update information including time step size
     */
    virtual void do_update_state(UpdateVelocityInfo& info) = 0;

  private:
    friend class TimeIntegratorManager;
    void do_build() override final;
    void init();
    void predict_dof(PredictDofInfo&);
    void update_state(UpdateVelocityInfo&);
};
}  // namespace uipc::backend::luisa
