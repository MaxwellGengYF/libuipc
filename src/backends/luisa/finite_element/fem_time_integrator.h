#pragma once
#include <finite_element/finite_element_method.h>
#include <time_integrator/time_integrator.h>

namespace uipc::backend::luisa
{
/**
 * @brief Time integrator for Finite Element Method (FEM)
 * 
 * This class provides time integration capabilities specific to finite element dynamics,
 * including DOF prediction and velocity updates for deformable bodies represented
 * using volumetric meshes.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 * 
 * The FEM state is represented as 3D positions for each vertex:
 * x = [x; y; z]
 * where x, y, z are the coordinates of the vertex in world space.
 */
class FEMTimeIntegrator : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    /**
     * @brief Implementation details (Pimpl pattern)
     */
    class Impl
    {
      public:
        SimSystemSlot<FiniteElementMethod> finite_element_method;
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
     * @brief Base information class providing access to FEM data
     * 
     * Provides convenient accessors to finite element state and properties.
     */
    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        /**
         * @brief Get view of fixed vertex flags
         * @return Buffer view of fixed flags (1 = fixed, 0 = free)
         */
        auto is_fixed() const noexcept
        {
            return m_impl->finite_element_method->is_fixed();
        }

        /**
         * @brief Get view of dynamic vertex flags
         * @return Buffer view of dynamic flags (1 = dynamic, 0 = kinematic)
         */
        auto is_dynamic() const noexcept
        {
            return m_impl->finite_element_method->is_dynamic();
        }

        /**
         * @brief Get view of rest positions
         * @return Buffer view of rest positions (x_bar)
         */
        auto x_bars() const noexcept
        {
            return m_impl->finite_element_method->x_bars();
        }

        /**
         * @brief Get view of current positions
         * @return Buffer view of positions (x)
         */
        auto xs() const noexcept { return m_impl->finite_element_method->xs(); }

        /**
         * @brief Get view of current velocities
         * @return Buffer view of velocities (v)
         */
        auto vs() const noexcept { return m_impl->finite_element_method->vs(); }

        /**
         * @brief Get view of predicted positions
         * @return Buffer view of predicted positions (x_tilde)
         */
        auto x_tildes() const noexcept
        {
            return m_impl->finite_element_method->x_tildes();
        }

        /**
         * @brief Get view of previous positions
         * @return Buffer view of previous positions (x_prev)
         */
        auto x_prevs() const noexcept
        {
            return m_impl->finite_element_method->x_prevs();
        }

        /**
         * @brief Get view of vertex masses
         * @return Buffer view of masses
         */
        auto masses() const noexcept
        {
            return m_impl->finite_element_method->masses();
        }

        /**
         * @brief Get view of gravity accelerations
         * @return Buffer view of gravity vectors
         */
        auto gravities() const noexcept
        {
            return m_impl->finite_element_method->gravities();
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

        /**
         * @brief Get view of previous positions
         * @return Buffer view of previous positions (x_prev)
         */
        auto x_prevs() const noexcept
        {
            return m_impl->finite_element_method->x_prevs();
        }

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

        /**
         * @brief Get view of velocities for modification
         * @return Buffer view of velocities (v)
         */
        auto vs() noexcept
        {
            return m_impl->finite_element_method->vs();
        }

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
     * Computes predicted DOF values (x_tilde) based on current state,
     * velocities, and external forces using the formula:
     * x_tilde = x + dt * v + dt^2 * M^-1 * f_ext
     * 
     * @param info Prediction information including time step and buffer access
     */
    virtual void do_predict_dof(PredictDofInfo& info) = 0;

    /**
     * @brief Update velocities after the Newton solve converges
     * 
     * Updates velocities based on the converged DOF values:
     * v = (x - x_prev) / dt
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
