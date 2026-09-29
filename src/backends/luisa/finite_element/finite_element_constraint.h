#pragma once
/********************************************************************
 * @file finite_element_constraint.h
 * @brief T5.3 (pulled forward from T5.4) — the FEM constraint base (port of
 * src/backends/cuda/finite_element/finite_element_constraint.h).
 *
 * Why it is here: `FiniteElementAnimator::Impl::init()` walks every FEM
 * geometry's `builtin::constraint_uids` and aborts on a uid no registered
 * `FiniteElementConstraint` claims, and the animator needs a constraint class to
 * declare its `FilteredInfo`-taking hooks against (the T5.3 brief's §1.2
 * dependency-cycle policy).  The class is a 40-line interface + 60-line
 * plumbing file with no T5.4 dependency, so it is ported in full -- T5.4 still
 * owns the concrete constraints other than `SoftPositionConstraint`
 * (`constraints/finite_element_external_vertex_force_constraint` keeps uid 671),
 * and T5.5 owns the diff-dof/diff-parm reporters.
 *
 * `do_build()` is `final` and does the reference's three-step check:
 *  1. the uid must appear in the scene's ConstitutionTabular (otherwise
 *     `SimSystemException`, i.e. the system is simply not built for that scene),
 *  2. the subclass's own `do_build(info)`,
 *  3. `FiniteElementAnimator::add_constraint(this)`.
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/finite_element_animator.h>
#include <animator/utils.h>

namespace uipc::backend::luisa
{
class FiniteElementConstraint : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    U64 uid() const noexcept;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual U64  get_uid() const noexcept = 0;
    virtual void do_init(FiniteElementAnimator::FilteredInfo& info) = 0;
    virtual void do_step(FiniteElementAnimator::FilteredInfo& info) = 0;
    virtual void do_report_extent(FiniteElementAnimator::ReportExtentInfo& info) = 0;
    virtual void do_compute_energy(FiniteElementAnimator::ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(FiniteElementAnimator::ComputeGradientHessianInfo& info) = 0;

  private:
    friend class FiniteElementAnimator;
    virtual void do_build() override final;

    void init(FiniteElementAnimator::FilteredInfo& info);
    void step(FiniteElementAnimator::FilteredInfo& info);
    void report_extent(FiniteElementAnimator::ReportExtentInfo& info);
    void compute_energy(FiniteElementAnimator::ComputeEnergyInfo& info);
    void compute_gradient_hessian(FiniteElementAnimator::ComputeGradientHessianInfo& info);
    SizeT m_index = ~0ull;
};
}  // namespace uipc::backend::luisa
