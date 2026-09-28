#pragma once
/********************************************************************
 * @file fem_linear_subsystem_reporter.h
 * @brief T5.1 — FEM linear-subsystem reporter base (port of
 * src/backends/cuda/finite_element/fem_linear_subsystem_reporter.h).
 *
 * A pure reporter base: every concrete reporter (the elastics reporter, the
 * contact reporters, ...) derives from it and declares a gradient/hessian extent,
 * which the subsystem uses to lay out its shared assembly buffers. The subsystem
 * itself (FEMLinearSubsystem) is T5.4; see fem_linear_subsystem.h.
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/fem_linear_subsystem.h>

namespace uipc::backend::luisa
{
class FEMLinearSubsystemReporter : public SimSystem
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

    using ReportExtentInfo = FEMLinearSubsystem::ReportExtentInfo;
    using AssembleInfo     = FEMLinearSubsystem::AssembleInfo;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo& info) = 0;
    virtual void do_report_extent(ReportExtentInfo& info) = 0;
    virtual void do_assemble(AssembleInfo& info) = 0;

  private:
    friend class FEMLinearSubsystem;
    virtual void do_build() final override;
    void init();
    void report_extent(ReportExtentInfo& info);
    void assemble(AssembleInfo& info);

    IndexT m_index = -1;
};
}  // namespace uipc::backend::luisa
