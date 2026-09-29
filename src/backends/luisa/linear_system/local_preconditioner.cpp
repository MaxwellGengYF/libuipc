#include <linear_system/local_preconditioner.h>
#include <uipc/common/log.h>

/*************************************************************************************************
* T5.4a — LocalPreconditioner (port of
* src/backends/cuda/linear_system/local_preconditioner.cu — namespace only).
*
* `apply()` has no caller in T5.4a: the reference's only caller is
* `GlobalLinearSystem::Impl::apply_preconditioner()`, which is part of the T8.3
* solve path (see global_linear_system.h's scope note).  It is defined here so the
* class has no undefined member at all -- T5.4b only has to add the two concrete
* preconditioners and T8.3 only has to call it.
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
