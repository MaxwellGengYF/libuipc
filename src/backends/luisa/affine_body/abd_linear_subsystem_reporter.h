#pragma once
/********************************************************************
 * @file affine_body/abd_linear_subsystem_reporter.h
 * @brief T6.1b — the ABD linear-subsystem reporter base (port of
 *        src/backends/cuda/affine_body/abd_linear_subsystem_reporter.h).
 *
 * A pure reporter base: the ABD animator (this batch), the inter-affine-body
 * constitution manager (this batch) and the T6.3 constitutions all derive from it
 * and declare a gradient/Hessian extent, which `ABDLinearSubsystem` uses to lay
 * out its shared 12x12 assembly buffers.
 *
 * The reference's four methods are defined in its `abd_linear_subsystem_reporter.cu`
 * and are ported body-for-body in the .cpp below.  The interface is a
 * declaration-only scaffold since T6.1a (`ABDLinearSubsystem`'s nested
 * `ReportExtentInfo`/`AssembleInfo` are named in it), so nothing here is new
 * surface — only the bodies.
 *********************************************************************/
#include <sim_system.h>
#include <affine_body/abd_linear_subsystem.h>

namespace uipc::backend::luisa
{
class ABDLinearSubsystemReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    using ReportExtentInfo = ABDLinearSubsystem::ReportExtentInfo;
    using AssembleInfo     = ABDLinearSubsystem::AssembleInfo;

  protected:
    virtual void do_build(BuildInfo& info)                = 0;
    virtual void do_init(InitInfo& info)                  = 0;
    virtual void do_report_extent(ReportExtentInfo& info) = 0;
    virtual void do_assemble(AssembleInfo& info)          = 0;

  private:
    friend class ABDLinearSubsystem;
    virtual void do_build() final override;
    void         init();  // only by ABDLinearSubsystem
    void         report_extent(ReportExtentInfo& info);
    void         assemble(AssembleInfo& info);

    IndexT m_index = -1;
};
}  // namespace uipc::backend::luisa
