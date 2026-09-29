#pragma once
/********************************************************************
 * @file fem_line_search_subreporter.h
 * @brief T5.3 (pulled forward from T5.5) — the FEM line-search sub-reporter base
 * (port of src/backends/cuda/finite_element/fem_line_search_subreporter.h).
 *
 * A pure reporter base: the elastics and the animator derive from it and declare
 * their energy extent, which `FEMLineSearchReporter` lays out.  Pulled forward
 * with its manager because `FiniteElementAnimator` and `FiniteElementElastics`
 * both need the class for their `LineSearchSubreporter` members (the T5.3 brief's
 * §1.2 dependency-cycle policy).  Nothing is left for T5.5 here.
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/fem_line_search_reporter.h>

namespace uipc::backend::luisa
{
class FEMLineSearchSubreporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    using ReportExtentInfo  = FEMLineSearchReporter::ReportExtentInfo;
    using ComputeEnergyInfo = FEMLineSearchReporter::ComputeEnergyInfo;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo& info) = 0;
    virtual void do_report_extent(ReportExtentInfo& info) = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info) = 0;

  private:
    virtual void do_build() override final;

    friend class FEMLineSearchReporter;
    void init();

    void report_extent(ReportExtentInfo& info);
    void compute_energy(ComputeEnergyInfo& info);

    IndexT m_index = -1;
};
}  // namespace uipc::backend::luisa
