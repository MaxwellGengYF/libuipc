#include <affine_body/abd_linear_subsystem_reporter.h>
#include <utils/report_extent_check.h>
/*************************************************************************************************
* T6.1b — ABDLinearSubsystemReporter (port of the reporter base; the reference's
* four methods live in src/backends/cuda/affine_body/abd_linear_subsystem_reporter.cu).
*
* `report_extent` forwards to the subclass and then runs the *shared*
* `check_report_extent` guard (cookbook §7.17 rule 7): a reporter that reported a
* Hessian block while `gradient_only` is set trips it.  The reference has no extra
* `UIPC_ASSERT` here (unlike the FEM reporter), so neither has the port.
*************************************************************************************************/
namespace uipc::backend::luisa
{
void ABDLinearSubsystemReporter::do_build()
{
    auto& abd_linear_subsystem = require<ABDLinearSubsystem>();

    // let subclass do the actual build
    BuildInfo info;
    do_build(info);

    abd_linear_subsystem.add_reporter(this);
}

void ABDLinearSubsystemReporter::init()
{
    InitInfo info;
    do_init(info);
}

void ABDLinearSubsystemReporter::report_extent(ReportExtentInfo& info)
{
    do_report_extent(info);
    info.check(name());
}

void ABDLinearSubsystemReporter::assemble(AssembleInfo& info)
{
    do_assemble(info);
}
}  // namespace uipc::backend::luisa
