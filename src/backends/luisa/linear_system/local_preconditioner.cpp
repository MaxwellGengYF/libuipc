#include <linear_system/local_preconditioner.h>
#include <uipc/common/log.h>

/*************************************************************************************************
* T5.4a — LocalPreconditioner (port of
* src/backends/cuda/linear_system/local_preconditioner.cu — namespace only).
*
* All four forwarding members are defined here (T5.4a already defined them, so
* T5.4b only had to add the two concrete preconditioners): `apply()` is called by
* `GlobalLinearSystem::Impl::apply_preconditioner()`, which T5.4b defined.
*************************************************************************************************/
namespace uipc::backend::luisa
{
void LocalPreconditioner::do_build()
{
    auto& global_linear_system = require<GlobalLinearSystem>();

    BuildInfo info;
    do_build(info);

    m_subsystem = info.m_subsystem;
    UIPC_ASSERT(m_subsystem != nullptr, "The depend_subsystem should not be nullptr.");

    global_linear_system.add_preconditioner(this);
}

void LocalPreconditioner::init()
{
    InitInfo info;
    do_init(info);
}

void LocalPreconditioner::assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info)
{
    do_assemble(info);
}

void LocalPreconditioner::apply(GlobalLinearSystem::ApplyPreconditionerInfo& info)
{
    do_apply(info);
}

void LocalPreconditioner::BuildInfo::connect(DiagLinearSubsystem* system)
{
    m_subsystem = system;
}
}  // namespace uipc::backend::luisa
