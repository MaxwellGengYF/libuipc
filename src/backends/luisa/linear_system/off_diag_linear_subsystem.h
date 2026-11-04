#pragma once
/********************************************************************
 * @file off_diag_linear_subsystem.h
 * @brief T8.3a-1 — the off-diagonal linear-subsystem base (port of
 *        src/backends/cuda/linear_system/off_diag_linear_subsystem.h —
 *        namespace + doc comment only).
 *
 * An `OffDiagLinearSubsystem` represents the block *between* two diagonal blocks
 * of the global Hessian: it names the two `DiagLinearSubsystem`s it couples in
 * `BuildInfo::connect(l, r)` (called from the derived `do_build(BuildInfo&)`),
 * registers itself with the `GlobalLinearSystem`, reports an
 * `(extent_lr, extent_rl)` block budget (`report_extent`) and writes both blocks
 * in `assemble(GlobalLinearSystem::OffDiagInfo&)`.
 *
 * WHY THE PORT COULD NOT DEFER IT ANY LONGER (the T5.4a/T5.4b scope note)
 * ----------------------------------------------------------------------
 * `SimSystemSlotCollection<T>::register_sim_system()` instantiates `T::is_valid()`
 * and `view()` instantiates `lazy_init()` (details/sim_system_slot.inl:82-89,
 * :107-133), so it needs a *complete* `T`.  This header being the first complete
 * declaration of the type is exactly what un-defers the four off-diag branches of
 * the global linear system (`add_subsystem`, `Impl::init`,
 * `Impl::_update_subsystem_extent`, `Impl::_assemble_linear_system`).
 *
 * SURFACE PARITY (plan 14 §3.1, rule 3)
 * ------------------------------------
 * The class surface is the reference's verbatim: `uid()` returns
 * `std::tuple<U64, U64>` = `(m_l->uid(), m_r->uid())` (the deterministic
 * block-ordering key, *not* a bare `U64`), `do_build()` stays `final override`
 * (the `BuildInfo` handshake is the base's), `report_extent`/`assemble` are the
 * two protected pure virtuals and `m_index` does *not* exist here — the index
 * lives in `GlobalLinearSystem::Impl::subsystem_infos[i].index` only (the T5.4a
 * comment claiming otherwise was wrong and is corrected).
 *********************************************************************/
#include <sim_system.h>
#include <linear_system/diag_linear_subsystem.h>
namespace uipc::backend::luisa
{
/**
 * @brief An off diag linear subsystem represents a submatrix of the global hessian
 */
class OffDiagLinearSubsystem : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
        void connect(DiagLinearSubsystem* l, DiagLinearSubsystem* r)
        {
            m_diag_l = l;
            m_diag_r = r;
        }

      private:
        friend class OffDiagLinearSubsystem;
        DiagLinearSubsystem* m_diag_l = nullptr;
        DiagLinearSubsystem* m_diag_r = nullptr;
    };

    class InitInfo
    {
      public:
    };

    std::tuple<U64, U64> uid() const noexcept;

  protected:
    virtual void report_extent(GlobalLinearSystem::OffDiagExtentInfo& info) = 0;
    virtual void assemble(GlobalLinearSystem::OffDiagInfo&)                 = 0;
    virtual void do_build(BuildInfo& info)                                  = 0;
    virtual void do_init(InitInfo& info);

  private:
    friend class GlobalLinearSystem;
    DiagLinearSubsystem* m_l = nullptr;
    DiagLinearSubsystem* m_r = nullptr;
    virtual void         do_build() final override;
    void                 init();  // only be called by GlobalLinearSystem
};
}  // namespace uipc::backend::luisa
