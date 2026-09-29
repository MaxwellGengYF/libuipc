#pragma once
/********************************************************************
 * @file finite_element_kinetic.h
 * @brief T5.3 — the FEM kinetic system (port of
 * src/backends/cuda/finite_element/finite_element_kinetic.h).
 *
 * The kinetic energy/gradient/Hessian *provider*: its concrete subclasses
 * (`FiniteElementBDF1Kinetic`, `FiniteElementBDF2Kinetic`) are the two ends of the
 * BDF integrators, and both the line search (through
 * `FEMLineSearchReporter::ComputeEnergyInfo`) and the linear subsystem (through
 * `FEMLinearSubsystem::ComputeGradientHessianInfo`) call into it.
 *
 * This is the file the T5.3 dependency cycle runs through: the `ComputeEnergyInfo`
 * wrapper needs `FEMLineSearchReporter::ComputeEnergyInfo` (T5.5's task, ported in
 * T5.3 -- see fem_line_search_reporter.h) and `ComputeGradientHessianInfo` needs
 * `FEMLinearSubsystem::ComputeGradientHessianInfo` (T5.4's task, whose *inline*
 * nested class T5.3 pulled forward into the T5.1 scaffold -- see the scope note in
 * fem_linear_subsystem.h).
 *
 * `do_build()` is `final` and does the reference's three-step wiring: require the
 * method, then `require<>` both the linear subsystem and the line-search reporter
 * (`add_kinetic(this)` on each).  A scene without a FiniteElement constitution,
 * or one whose config selects neither `bdf1` nor `bdf2`, never builds a concrete
 * subclass.
 *
 * Deviation from the reference's .cu (reported): the reference's
 * `finite_element_kinetic.cu` includes `<finite_element/finite_element_diff_dof_reporter.h>`
 * (T5.5) without using anything from it; the port drops that include, because the
 * diff-dof reporter is not ported and would drag in `<diff_sim/...>`.
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/fem_line_search_reporter.h>
#include <finite_element/fem_linear_subsystem.h>

namespace uipc::backend::luisa
{
class FiniteElementKinetic : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        SimSystemSlot<FiniteElementMethod> finite_element_method;
    };

    class BuildInfo
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

        auto x_tildes() const noexcept
        {
            return m_impl->finite_element_method->x_tildes();
        }

        auto x_prevs() const noexcept
        {
            return m_impl->finite_element_method->x_prevs();
        }

        auto masses() const noexcept
        {
            return m_impl->finite_element_method->masses();
        }


      protected:
        Impl* m_impl = nullptr;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Impl* impl, FEMLineSearchReporter::ComputeEnergyInfo* base_info) noexcept
            : BaseInfo(impl)
            , base_info(base_info)
        {
        }

        auto energies() const noexcept { return base_info->energies(); }
        auto dt() const noexcept { return base_info->dt(); }

      private:
        FEMLineSearchReporter::ComputeEnergyInfo* base_info = nullptr;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(Impl* impl,
                                   FEMLinearSubsystem::ComputeGradientHessianInfo* base_info) noexcept
            : BaseInfo(impl)
            , base_info(base_info)
        {
        }

        auto gradient_only() const noexcept { return base_info->gradient_only(); }
        auto gradients() const noexcept { return base_info->gradients(); }
        auto hessians() const noexcept { return base_info->hessians(); }
        auto dt() const noexcept { return base_info->dt(); }

      private:
        FEMLinearSubsystem::ComputeGradientHessianInfo* base_info = nullptr;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) = 0;

  private:
    virtual void do_build() override final;

    friend class FEMLineSearchReporter;
    void compute_energy(FEMLineSearchReporter::ComputeEnergyInfo& info);

    friend class FEMLinearSubsystem;
    void compute_gradient_hessian(FEMLinearSubsystem::ComputeGradientHessianInfo& info);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
