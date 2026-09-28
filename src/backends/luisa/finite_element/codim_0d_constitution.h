/********************************************************************
 * @file codim_0d_constitution.h
 * @brief T5.1/T5.2 — Codim0DConstitution, the dimension-0d (particle) constitution base (port of
 * src/backends/cuda/finite_element/codim_0d_constitution.h).
 *
 * SCOPE NOTE (T5.1): ported in T5.1 together with finite_element_constitution.h
 * because `FiniteElementMethod::_classify_base_constitutions()` `dynamic_cast`s
 * every constitution to its dimension base and the base chain's virtual overrides
 * must be *defined* (they are what fills the vtable of a concrete constitution).
 * The bodies are the reference's pure plumbing (type dispatch + the BaseInfo
 * accessors); T5.2 adds the concrete constitutions on top.
 *********************************************************************/
#pragma once
#include <finite_element/finite_element_constitution.h>

namespace uipc::backend::luisa
{
class Codim0DConstitution : public FiniteElementConstitution
{
  public:
    using FiniteElementConstitution::FiniteElementConstitution;

    class BuildInfo
    {
      public:
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;

  private:
    friend class FiniteElementMethod;

    virtual void do_report_extent(ReportExtentInfo& info) override final;
    virtual void do_build(FiniteElementConstitution::BuildInfo& info) override final;
    virtual IndexT get_dim() const noexcept override final;
    virtual void do_compute_energy(FiniteElementConstitution::ComputeEnergyInfo& info) override final;
    virtual void do_compute_gradient_hessian(
        FiniteElementConstitution::ComputeGradientHessianInfo& info) override final;
};
}  // namespace uipc::backend::luisa
