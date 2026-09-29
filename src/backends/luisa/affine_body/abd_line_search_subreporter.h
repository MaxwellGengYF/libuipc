#pragma once
/********************************************************************
 * @file affine_body/abd_line_search_subreporter.h
 * @brief SCOPE NOTE (T6.4) — the ABD line-search *sub*-reporter's declaration
 * surface (port of src/backends/cuda/affine_body/abd_line_search_subreporter.h).
 *
 * The sibling of abd_line_search_reporter.h: `ABDLineSearchReporter::Impl` holds a
 * `SimSystemSlotCollection<ABDLineSearchSubreporter>`, so the type must be
 * complete.  Ported here: the class declaration with the reference's two
 * Ported here: the class declaration with the reference's two `using`-aliases, both
 * info classes, the hook set and the private plumbing declarations.  ITS BODIES
 * ARE PULLED FORWARD BY T6.1b (abd_line_search_subreporter.cpp: the four
 * forwarding hooks) because the ABD animator's and the inter-body manager's
 * line-search subreporters derive from it and a vtable needs them (the T5.3 /
 * fem_line_search_subreporter.cpp precedent).  NOT defined here (T6.4):
 * `ABDLineSearchReporter`'s own bodies, so `do_build()`'s
 * `require<ABDLineSearchReporter>()` invalidates every subreporter until then.
 *********************************************************************/
#include <sim_system.h>
#include <affine_body/abd_line_search_reporter.h>

namespace uipc::backend::luisa
{
class ABDLineSearchSubreporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    using ReportExtentInfo  = ABDLineSearchReporter::ReportExtentInfo;
    using ComputeEnergyInfo = ABDLineSearchReporter::ComputeEnergyInfo;

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

    friend class ABDLineSearchReporter;
    void init();

    void report_extent(ReportExtentInfo& info);
    void compute_energy(ComputeEnergyInfo& info);

    IndexT m_index = -1;
};
}  // namespace uipc::backend::luisa
