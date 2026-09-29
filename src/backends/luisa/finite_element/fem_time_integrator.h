#pragma once
/********************************************************************
 * @file fem_time_integrator.h
 * @brief T5.3 — the FEM time-integrator bridge (port of
 * src/backends/cuda/finite_element/fem_time_integrator.h).
 *
 * A *pure forwarding shim* over the already ported
 * `time_integrator/time_integrator.h`: `TimeIntegrator`'s hooks take the
 * manager's info types, this class inserts the FEM-specific `BaseInfo`
 * (is_fixed / is_dynamic / xs / x_tildes / x_prevs / masses / gravities /
 * external_force_accs) between them and the concrete BDF integrators.
 *
 * Port mapping (cookbook §3.2/§7.13):
 *   cuda_tool::CBufferView<Vector3> -> lud::LCBufferView<LVector3>  (const views)
 *   cuda_tool::BufferView<Vector3>  -> lud::LBufferView<LVector3>   (mutable views)
 * `Vector3` is an Eigen type and Eigen types cannot be LC buffer elements, so
 * every view is over `lud::LVector3` (byte-identical to `uipc::Vector3`, see
 * type_define.h).  `auto` is used where the reference uses `auto`, so the
 * accessors keep returning exactly what the method's own accessors return.
 *
 * The reference's `Impl::finite_element_method` is a `SimSystemSlot` and
 * `require<FiniteElementMethod>()` is resolved in `do_build()`, i.e. a scene
 * without a FiniteElement constitution invalidates this system (and, through
 * `require<>`, its concrete subclasses) -- the reference's shutdown idiom.
 *********************************************************************/
#include <finite_element/finite_element_method.h>
#include <time_integrator/time_integrator.h>

namespace uipc::backend::luisa
{
class FEMTimeIntegrator : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    class Impl
    {
      public:
        SimSystemSlot<FiniteElementMethod> finite_element_method;
    };

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        auto is_fixed() const noexcept
        {
            return m_impl->finite_element_method->is_fixed();
        }

        auto is_dynamic() const noexcept
        {
            return m_impl->finite_element_method->is_dynamic();
        }

        auto x_bars() const noexcept
        {
            return m_impl->finite_element_method->x_bars();
        }

        auto xs() const noexcept { return m_impl->finite_element_method->xs(); }

        auto vs() const noexcept { return m_impl->finite_element_method->vs(); }

        lud::LBufferView<LVector3> x_tildes() const noexcept
        {
            return m_impl->finite_element_method->m_impl.x_tildes.view();
        }

        auto x_prevs() const noexcept
        {
            return m_impl->finite_element_method->x_prevs();
        }

        auto masses() const noexcept
        {
            return m_impl->finite_element_method->masses();
        }

        auto gravities() const noexcept
        {
            return m_impl->finite_element_method->gravities();
        }

        auto external_force_accs() const noexcept
        {
            return m_impl->finite_element_method->vertex_external_force_accs();
        }

      protected:
        Impl* m_impl = nullptr;
    };

    class PredictDofInfo : public BaseInfo
    {
      public:
        PredictDofInfo(Impl* impl, TimeIntegrator::PredictDofInfo* base_info)
            : BaseInfo(impl)
            , base_info(base_info)
        {
        }

        auto dt() const noexcept { return base_info->dt(); }

        /// The reference *shadows* `BaseInfo::x_prevs()` with a mutable view: the
        /// BDF integrators record `x_n := xs` through it.
        lud::LBufferView<LVector3> x_prevs() const noexcept
        {
            return m_impl->finite_element_method->m_impl.x_prevs.view();
        }

      private:
        TimeIntegrator::PredictDofInfo* base_info = nullptr;
    };

    class UpdateVelocityInfo : public BaseInfo
    {
      public:
        UpdateVelocityInfo(Impl* impl, TimeIntegrator::UpdateVelocityInfo* base_info)
            : BaseInfo(impl)
            , base_info(base_info)
        {
        }

        auto dt() const noexcept { return base_info->dt(); }

        lud::LBufferView<LVector3> vs() noexcept
        {
            return m_impl->finite_element_method->m_impl.vs.view();
        }

      private:
        TimeIntegrator::UpdateVelocityInfo* base_info = nullptr;
    };


  protected:
    virtual void do_build(BuildInfo& info) = 0;

    virtual void do_init(InitInfo& info) = 0;
    virtual void do_predict_dof(PredictDofInfo& info) = 0;
    virtual void do_update_state(UpdateVelocityInfo& info) = 0;

  private:
    void do_build(TimeIntegrator::BuildInfo& info) override;

    void do_init(TimeIntegrator::InitInfo& info) override;
    void do_predict_dof(TimeIntegrator::PredictDofInfo& info) override;
    void do_update_state(TimeIntegrator::UpdateVelocityInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
