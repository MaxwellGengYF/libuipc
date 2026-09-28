#pragma once
/********************************************************************
 * @file finite_element_constitution.h
 * @brief T5.1/T5.2 — the FEM constitution base class (port of
 * src/backends/cuda/finite_element/finite_element_constitution.h).
 *
 * SCOPE NOTE (T5.1): the reference plan assigns this file to T5.2 ("constitution
 * base"), but `FiniteElementMethod` cannot work without it: the method
 * classifies the registered constitutions by (dim, uid), `dynamic_cast`s them to
 * the dimension bases and calls `init()`, so the *interface* (and its
 * dimension-independent plumbing, which is exactly what the reference's .cu
 * contains) is a T5.1 prerequisite. T5.2 adds the concrete constitutions
 * (arap_3d, stable_neo_hookean_3d, the shells, ... and their sym/*.inl) on top of
 * this, unchanged. See the T5.1 report.
 *
 * A concrete constitution declares only its three build/compute hooks plus
 * `get_uid()`/`get_dim()`; the dimension base (`FEM3DConstitution`,
 * `Codim2DConstitution`, ...) provides the 2nd-level Info objects and the type
 * dispatch. A system that is not used by the current scene throws
 * SimSystemException from `do_build()`, which invalidates it (the engine catches
 * it), so an unused constitution simply disappears from the system graph.
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_elastics.h>

namespace uipc::backend::luisa
{
class FiniteElementConstitution : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    using ReportExtentInfo           = FiniteElementElastics::ReportExtentInfo;
    using ComputeEnergyInfo          = FiniteElementElastics::ComputeEnergyInfo;
    using ComputeGradientHessianInfo = FiniteElementElastics::ComputeGradientHessianInfo;

    U64 uid() const noexcept;
    IndexT dim() const noexcept;

  protected:
    virtual U64 get_uid() const noexcept = 0;
    virtual IndexT get_dim() const noexcept = 0;

    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(FiniteElementMethod::FilteredInfo& info) = 0;
    virtual void do_report_extent(ReportExtentInfo& info) = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) = 0;
    const FiniteElementMethod::DimInfo& dim_info() const noexcept;
    const FiniteElementMethod::ConstitutionInfo& constitution_info() const noexcept;

  private:
    friend class Codim0DConstitution;
    friend class Codim1DConstitution;
    friend class Codim2DConstitution;
    friend class FEM3DConstitution;
    friend class FiniteElementMethod;
    friend class FiniteElementElastics;

    friend class FiniteElementConstitutionDiffParmReporter;           // T5.5
    friend class FiniteElementConstitutionLinearSubsystemReporter;    // T5.4
    friend class FiniteElementConstitutionLineSearchSubreporter;      // T5.5

    void init();  // only be called by FiniteElementMethod

    virtual void do_build() override final;
    void report_extent(ReportExtentInfo& info);
    void compute_energy(ComputeEnergyInfo& info);
    void compute_gradient_hessian(ComputeGradientHessianInfo& info);

    SizeT m_index = 0;
    SizeT m_index_in_dim = 0;
    FiniteElementMethod* m_finite_element_method = nullptr;
    FiniteElementMethod::Impl& fem() const noexcept;
};
}  // namespace uipc::backend::luisa
