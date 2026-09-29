#pragma once
/********************************************************************
 * @file affine_body/affine_body_constitution.h
 * @brief T6.1a — the ABD constitution *interface* (port of
 * src/backends/cuda/affine_body/affine_body_constitution.{h,cu}).
 *
 * This is the interface every backend ABD constitution derives from: it resolves
 * `AffineBodyDynamics` in `do_build()`, checks that the scene really carries its
 * constitution uid (the frontend's `ConstitutionTabular` is built from the
 * geometries' `constitution_uid` meta attribute), calls the subclass hook and
 * registers itself with the dynamics.  The concrete constitutions
 * (`constitutions/arap`, `constitutions/ortho_potential`) are **T6.3a**'s.
 *
 * The reference's header includes abd_line_search_reporter.h and
 * abd_linear_subsystem.h because the two `*Info` nested classes wrap
 * `ABDLineSearchReporter::ComputeEnergyInfo` /
 * `ABDLinearSubsystem::ComputeGradientHessianInfo`; both interfaces are ported (as
 * declaration surfaces) by T6.1a, so these parameter types exist (the T5.3 §7.18
 * rule 3 policy).
 *
 * Port mapping: `cuda_tool::CBufferView<Vector12>` -> `lud::LCBufferView<LVector12>`
 * (both are windowed views; `Impl::subview` returns the constitution's body
 * window).
 *********************************************************************/
#include <sim_system.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/abd_linear_subsystem.h>

namespace uipc::backend::luisa
{
class AffineBodyConstitution : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    U64 uid() const;

    class BuildInfo
    {
      public:
    };

    class Impl
    {
      public:
        SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
        AffineBodyDynamics::Impl& abd() noexcept
        {
            return affine_body_dynamics->m_impl;
        }
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl, IndexT index) noexcept
            : m_impl(impl)
            , m_index(index)
        {
        }

        auto is_fixed() const noexcept
        {
            return m_impl->affine_body_dynamics->body_is_fixed();
        }

        lud::LCBufferView<LVector12> qs() const noexcept;

        lud::LCBufferView<LVector12> q_prevs() const noexcept;

        lud::LCBufferView<Float> volumes() const noexcept;

      protected:
        Impl*  m_impl  = nullptr;
        IndexT m_index = -1;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Impl*  impl,
                          IndexT index,
                          ABDLineSearchReporter::ComputeEnergyInfo* base_info) noexcept
            : BaseInfo(impl, index)
            , base_info(base_info)
        {
        }

        auto energies() const noexcept { return base_info->energies(); }

        auto dt() const noexcept { return base_info->dt(); }

      private:
        ABDLineSearchReporter::ComputeEnergyInfo* base_info = nullptr;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(Impl*  impl,
                                   IndexT index,
                                   ABDLinearSubsystem::ComputeGradientHessianInfo* base_info) noexcept
            : BaseInfo(impl, index)
            , base_info(base_info)
        {
        }

        auto gradient_only() const noexcept
        {
            return base_info->gradient_only();
        }
        auto gradients() const noexcept { return base_info->gradients(); }
        auto hessians() const noexcept { return base_info->hessians(); }
        auto dt() const noexcept { return base_info->dt(); }

      private:
        ABDLinearSubsystem::ComputeGradientHessianInfo* base_info = nullptr;
    };

  protected:
    virtual U64  get_uid() const = 0;
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(AffineBodyDynamics::FilteredInfo& info) = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) = 0;

  private:
    friend class AffineBodyDynamics;
    friend class ABDLineSearchReporter;
    friend class ABDLinearSubsystem;

    virtual void do_build() override final;

    void init(AffineBodyDynamics::FilteredInfo& info);
    void compute_energy(ABDLineSearchReporter::ComputeEnergyInfo& info);
    void compute_gradient_hessian(ABDLinearSubsystem::ComputeGradientHessianInfo& info);

    IndexT m_index = -1;
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
