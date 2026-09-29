#include <finite_element/fem_linear_subsystem_reporter.h>
#include <utils/report_extent_check.h>

/*************************************************************************************************
* T5.4a — FEMLinearSubsystemReporter (port of the reporter base; the reference's
* four methods live in src/backends/cuda/finite_element/fem_linear_subsystem_reporter.h
* and are defined at the top of its fem_linear_subsystem.cu).
*
* The four definitions T5.1 kept here (the reporter registry + the extent
* setters/check of `FEMLinearSubsystem`) moved into fem_linear_subsystem.cpp in
* T5.4a, exactly as T5.1's scope note said they would, so this file is again only
* the reporter base.
*************************************************************************************************/
namespace uipc::backend::luisa
{
void FEMLinearSubsystemReporter::do_build()
{
    auto& fem_linear_subsystem = require<FEMLinearSubsystem>();
    // let subclass do the actual build
    BuildInfo info;
    do_build(info);

    fem_linear_subsystem.add_reporter(this);
}

void FEMLinearSubsystemReporter::init()
{
    InitInfo info;
    do_init(info);
}

void FEMLinearSubsystemReporter::report_extent(ReportExtentInfo& info)
{
    do_report_extent(info);
    info.check(name());
    UIPC_ASSERT(!(info.gradient_only() && info.m_hessian_count != 0),
                "When gradient_only is true, hessian_count must be 0, but {} provides hessian count={}.\n"
                "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                name(),
                info.m_hessian_count);
}

void FEMLinearSubsystemReporter::assemble(AssembleInfo& info)
{
    do_assemble(info);
}
}  // namespace uipc::backend::luisa
