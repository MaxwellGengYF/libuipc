#pragma once
/********************************************************************
 * @file local_preconditioner.h
 * @brief T5.4a — the local-preconditioner base (port of
 *        src/backends/cuda/linear_system/local_preconditioner.h — namespace only).
 *
 * A `LocalPreconditioner` is bound to exactly one `DiagLinearSubsystem` (through
 * `BuildInfo::connect()`), registers itself with the `GlobalLinearSystem` and
 * applies a preconditioner to that subsystem's dof window.
 *
 * T5.4b derives the two concrete preconditioners (`fem_diag_preconditioner`,
 * `fem_mas_preconditioner`) from this class; its interface is exactly the
 * reference's, so they plug in without reworking anything here.
 *
 * The apply path (`do_apply(GlobalLinearSystem::ApplyPreconditionerInfo&)`) is
 * called by `GlobalLinearSystem::Impl::apply_preconditioner()`, which T5.4b
 * defined (the local fan-out + the no-local-preconditioner copy fallback); only
 * the *global*-preconditioner branch of that entry stays T8.3's.
 *********************************************************************/
#include <sim_system.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::luisa
{
class DiagLinearSubsystem;

class LocalPreconditioner : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class InitInfo
    {
      public:
    };

    class BuildInfo
    {
      public:
        void connect(DiagLinearSubsystem* system);

      private:
        friend class LocalPreconditioner;
        DiagLinearSubsystem* m_subsystem = nullptr;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo& info) = 0;
    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) = 0;
    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) = 0;

  private:
    friend class GlobalLinearSystem;

    virtual void do_build() final override;
    virtual void init();

    void assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info);
    void apply(GlobalLinearSystem::ApplyPreconditionerInfo& info);
    DiagLinearSubsystem* m_subsystem = nullptr;
};
}  // namespace uipc::backend::luisa
