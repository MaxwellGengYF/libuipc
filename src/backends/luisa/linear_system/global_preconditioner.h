#pragma once
/********************************************************************
 * @file global_preconditioner.h
 * @brief T8.3a-3 — the global-preconditioner base (port of
 *        src/backends/cuda/linear_system/global_preconditioner.h — namespace only).
 *
 * A `GlobalPreconditioner` is the *whole-matrix* counterpart of
 * `LocalPreconditioner`: it is not bound to a `DiagLinearSubsystem`, it registers
 * itself with the `GlobalLinearSystem` (one optional `SimSystemSlot`), and it sees
 * the assembled matrix `A` with its off-diagonal blocks, so it can implement
 * incomplete-Cholesky-like or multi-block schemes.  `LocalPreconditioner` may only
 * touch one diagonal block — that contrast is the point of the two classes.
 *
 * Call order (the reference's, `global_linear_system.cu:422-435` / `:466-510`, see
 * `GlobalLinearSystem::Impl::_assemble_preconditioner` /
 * `apply_preconditioner`):
 *   * `assemble()` runs *first*, once per solve, before the local fan-out;
 *   * `apply()` runs *first* too, over the **whole** `z`/`r`, and the local
 *     fan-out then overwrites its own dof windows.  Both run — the global one is
 *     not an alternative to the local one;
 *   * `!global_preconditioner` gates **only** the `z <- r` copy fallback for the
 *     diag subsystems that have no local preconditioner.
 *
 * There is no `InitInfo` here (contrast `LocalPreconditioner`): the reference's
 * `do_build()` is `require<GlobalLinearSystem>() -> do_build(info) ->
 * add_preconditioner(this)` and nothing else, and the thunks have no
 * `check_state()` either — they run inside the solver's iteration, i.e. during
 * `SolveGlobalLinearSystem`, and the reference does not test for it (a test that
 * drives a standalone `impl().apply_preconditioner()` is in no engine state).
 *
 * `GlobalPreconditionerAssemblyInfo` / `ApplyPreconditionerInfo` are already the
 * reference's complete types (T5.4a ported them with the rest of the header);
 * this class adds no payload of its own.
 *********************************************************************/
#include <sim_system.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::luisa
{
class GlobalPreconditioner : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_assemble(GlobalLinearSystem::GlobalPreconditionerAssemblyInfo& info) = 0;
    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) = 0;

  private:
    friend class GlobalLinearSystem;
    virtual void do_build() final override;
    void assemble(GlobalLinearSystem::GlobalPreconditionerAssemblyInfo& info);
    void apply(GlobalLinearSystem::ApplyPreconditionerInfo& info);
};
}  // namespace uipc::backend::luisa
