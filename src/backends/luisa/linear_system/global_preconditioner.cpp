#include <linear_system/global_preconditioner.h>

/*************************************************************************************************
* T8.3a-3 — GlobalPreconditioner (port of
* src/backends/cuda/linear_system/global_preconditioner.cu — namespace only).
*
* Four bodies, exactly the reference's three:
*  * `do_build()`       — `require<GlobalLinearSystem>()` -> `do_build(BuildInfo&)` ->
*                         `add_preconditioner(this)`, the only writer of the
*                         `Impl::global_preconditioner` slot;
*  * `assemble()/apply()` — one-line thunks to the concrete preconditioner.
*
* Deliberately *not* here (both measured traps, research §7 R3/R4):
*  * no `check_state(SimEngineState::SolveGlobalLinearSystem, ...)`: the reference
*    has none, and a standalone-`Impl` test (`impl().apply_preconditioner(...)`) is
*    in no engine state, so the guard would abort it (`UIPC_ASSERT` abort()s);
*  * no second-registration assert: `SimSystemSlot::register_sim_system()` silently
*    overwrites (`src/backends/common/details/sim_system_slot.inl:7-13`) and the
*    reference's `add_preconditioner(GlobalPreconditioner*)` has no assert either, so
*    adding one would be a new cross-backend divergence, not reference fidelity.
*************************************************************************************************/
namespace uipc::backend::luisa
{
void GlobalPreconditioner::do_build()
{
    auto& global_linear_system = require<GlobalLinearSystem>();

    BuildInfo info;
    do_build(info);

    global_linear_system.add_preconditioner(this);
}

void GlobalPreconditioner::assemble(GlobalLinearSystem::GlobalPreconditionerAssemblyInfo& info)
{
    do_assemble(info);
}

void GlobalPreconditioner::apply(GlobalLinearSystem::ApplyPreconditionerInfo& info)
{
    do_apply(info);
}
}  // namespace uipc::backend::luisa
