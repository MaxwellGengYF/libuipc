#pragma once
#include <affine_body/affine_body_dynamics.h>
#include <time_integrator/time_integrator.h>

namespace uipc::backend::luisa
{
/**
 * @brief Time integrator for Affine Body Dynamics (ABD)
 * 
 * This class provides time integration capabilities specific to affine body dynamics,
 * including DOF prediction and velocity updates for rigid and deformable bodies
 * represented using affine transformations.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 * 
 * The affine body state is represented as a 12-dimensional vector:
 * q = [p; a1; a2; a3]
 * where p is the translation and a1, a2, a3 are the columns of the affine matrix.
 */
class ABDTimeIntegrator : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    /**
     * @brief Implementation details (Pimpl pattern)
     */
    class Impl
    {
      public:
        SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
    };

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

    /**
     * @brief Base information class providing access to ABD data
     * 
     * Provides convenient accessors to affine body state and properties.
     */
    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        /**
         * @brief Get view of fixed body flags
         * @return Buffer view of fixed flags (1 = fixed, 0 = free)
         */
        auto is_fixed() const noexcept
        {
            return m_impl->affine_body_dynamics->body_is_fixed();
        }

        /**
         * @brief Get view of dynamic body flags
         * @return Buffer view of dynamic flags (1 = dynamic, 0 = kinematic)
         */
        auto is_dynamic() const noexcept
        {
            return m_impl->affine_body_dynamics->body_is_dynamic();
        }

        /**
         * @brief Get view of Jacobi matrices for all vertices
         * @return Buffer view of ABD Jacobi matrices
         */
        auto Js() const noexcept { return m_impl->affine_body_dynamics->Js(); }

        /**
         * @brief Get view of current DOF values
         * @return Buffer view of q values (12D per body)
         */
        auto qs() const noexcept { return m_impl->affine_body_dynamics->qs(); }

        /**
         * @brief Get view of predicted DOF values
         * @return Buffer view of q_tilde values (12D per body)
         */
        auto q_tildes() const noexcept
        {
            return m_impl->affine_body_dynamics->q_tildes();
        }

        /**
         * @brief Get view of current velocities
         * @return Buffer view of q_v values (12D per body)
         */
        auto q_vs() const noexcept
        {
            return m_impl->affine_body_dynamics->q_vs();
        }

        /**
         * @brief Get view of previous DOF values
         * @return Buffer view of q_prev values (12D per body)
         */
        auto q_prevs() const noexcept
        {
            return m_impl->affine_body_dynamics->q_prevs();
        }

        /**
         * @brief Get view of body masses
         * @return Buffer view of mass matrices (12x12 blocks)
         */
        auto masses() const noexcept
        {
            return m_impl->affine_body_dynamics->body_masses();
        }

        /**
         * @brief Get view of gravity accelerations
         * @return Buffer view of gravity vectors (12D per body)
         */
        auto gravities() const noexcept
        {
            return m_impl->affine_body_dynamics->body_gravities();
        }

        /**
         * @brief Get view of external force accelerations
         * @return Buffer view of external acceleration vectors (12D per body)
         */
        auto external_force_accs() const noexcept
        {
            return m_impl->affine_body_dynamics->body_external_force_accs();
        }

      protected:
        Impl* m_impl = nullptr;
    };

    /**
     * @brief Information for the DOF prediction stage
     * 
     * Extends BaseInfo with time step information.
     */
    class PredictDofInfo : public BaseInfo
    {
      public:
        PredictDofInfo(Impl* impl, TimeIntegrator::PredictDofInfo* base_info)
            : BaseInfo(impl)
            , base_info(base_info)
        {
        }

        /**
         * @brief Get the current time step size
         */
        Float dt() const noexcept { return base_info->dt(); }

      private:
        TimeIntegrator::PredictDofInfo* base_info = nullptr;
    };

    /**
     * @brief Information for the velocity update stage
     * 
     * Extends BaseInfo with time step information.
     */
    class UpdateVelocityInfo : public BaseInfo
    {
      public:
        UpdateVelocityInfo(Impl* impl, TimeIntegrator::UpdateVelocityInfo* base_info)
            : BaseInfo(impl)
            , base_info(base_info)
        {
        }

        /**
         * @brief Get the current time step size
         */
        Float dt() const noexcept { return base_info->dt(); }

      private:
        TimeIntegrator::UpdateVelocityInfo* base_info = nullptr;
    };

  protected:
    /**
     * @brief Build the time integrator
     * 
     * Derived classes implement this to set up resources and register
     * callbacks specific to their integration scheme.
     * 
     * @param info Build information
     */
    virtual void do_build(BuildInfo& info) = 0;

    /**
     * @brief Initialize the time integrator
     * 
     * Called once during scene initialization.
     * 
     * @param info Initialization information
     */
    virtual void do_init(InitInfo& info) = 0;

    /**
     * @brief Predict the DOF values for the next time step
     * 
     * Computes predicted DOF values (q_tilde) based on current state,
     * velocities, and external forces using the formula:
     * q_tilde = q + dt * v + dt^2 * M^-1 * f_ext
     * 
     * @param info Prediction information including time step and buffer access
     */
    virtual void do_predict_dof(PredictDofInfo& info) = 0;

    /**
     * @brief Update velocities after the Newton solve converges
     * 
     * Updates velocities based on the converged DOF values:
     * v = (q - q_prev) / dt
     * 
     * @param info Update information including time step and buffer access
     */
    virtual void do_update_state(UpdateVelocityInfo& info) = 0;

  private:
    void do_build(TimeIntegrator::BuildInfo& info) override;

    void do_init(TimeIntegrator::InitInfo& info) override;
    void do_predict_dof(TimeIntegrator::PredictDofInfo& info) override;
    void do_update_state(TimeIntegrator::UpdateVelocityInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
