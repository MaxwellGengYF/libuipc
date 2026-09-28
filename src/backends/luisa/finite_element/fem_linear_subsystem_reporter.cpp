#include <finite_element/fem_linear_subsystem_reporter.h>
#include <utils/report_extent_check.h>

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

// ---------------------------------------------------------------------------
// T5.1 provides the four *trivial* members of the T5.4-owned subsystem surface
// (see fem_linear_subsystem.h): the reporter registry and the extent setters/check.
// T5.4 moves these definitions into its fem_linear_subsystem.cpp and adds the
// system class itself.
// ---------------------------------------------------------------------------
void FEMLinearSubsystem::add_reporter(FEMLinearSubsystemReporter* reporter)
{
    m_impl.reporters.register_sim_system(*reporter);
}

void FEMLinearSubsystem::ReportExtentInfo::gradient_count(SizeT size)
{
    m_gradient_count = size;
}

void FEMLinearSubsystem::ReportExtentInfo::hessian_count(SizeT size)
{
    m_hessian_count = size;
}

void FEMLinearSubsystem::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}
}  // namespace uipc::backend::luisa
