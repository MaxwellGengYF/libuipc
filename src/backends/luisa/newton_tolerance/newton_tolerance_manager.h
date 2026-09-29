#pragma once
/********************************************************************
 * @file newton_tolerance/newton_tolerance_manager.h
 * @brief T6.4a — the Newton-tolerance manager (port of
 *        src/backends/cuda/newton_tolerance/newton_tolerance_manager.h).
 *
 * This is the subsystem T6.4a had to port *before* the ABD tolerance checker:
 * `ABDToleranceChecker` derives from `NewtonToleranceChecker`, and the checker
 * registers itself with this manager.  The reference's whole subsystem is small
 * (307 lines: this pair + the checker pair + `max_translation_checker.cu`), and it
 * is the Newton loop's convergence contract:
 *
 *   SimEngine::pre_newton(frame) -> every registered checker's `do_pre_newton`
 *   SimEngine::check(ResultInfo&) -> every registered checker's `do_check`
 *      (a `CheckResultInfo` carrying the finished iteration count), the first
 *      non-converged checker **short-circuits** the loop, and the whole report is
 *      logged through `logger::info`.
 *
 * The contract is exactly the reference's:
 *  * `ResultInfo::m_converged` starts **true** ("all checkers converged") and is
 *    only turned false by a checker that reports false;
 *  * `check()` **breaks** out of the iteration on the first non-converged checker
 *    (so later checkers do not even get their `do_check` called -- a test can
 *    observe this ordering);
 *  * the per-checker report line is `\n    [*|x] <name> <report>` and the header
 *    is the reference's own multi-line format string;
 *  * `report_buffer` is a *reused* std::string, cleared per call (the reference
 *    reserves 1024).
 *
 * DEVIATIONS (both reported in the batch's FINDINGS, none behavioural):
 *  1. `SimSystemSlotCollection<NewtonToleranceChecker>` is the port's T3/T4
 *     collection (the reference's is its own); it is the same shape it has in
 *     `ABDLineSearchReporter::Impl` and every other ported fan-out.
 *  2. `init()`/`pre_newton()`/`check()` are **public**: the reference keeps them
 *     private with `friend class SimEngine`, and the port's `SimEngine` is the only
 *     production caller, but the *unit test* has to drive the contract the way the
 *     reference's Newton loop does (`pre_newton` then `check` per iteration).  This
 *     is the T5.4b "pure test seam" precedent (cookbook §7.20 rule 9: the FEM
 *     animator's three count accessors and T5.4a's public `GlobalLinearSystem::init()`)
 *     -- the `friend class SimEngine` declaration is kept as the reference spells it.
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
/// The reference's `newton_tolerance_manager.h` forward-declares this type: the
/// manager only ever holds a `SimSystemSlotCollection<NewtonToleranceChecker>`
/// (a list of pointers), while `newton_tolerance_checker.h` includes *this* header
/// for the nested info classes -- the reference's include cycle, preserved.
class NewtonToleranceChecker;

class NewtonToleranceManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class ResultInfo
    {
      public:
        bool converged() const { return m_converged; }
        void frame(SizeT frame) { m_frame = frame; }
        void newton_iter(IndexT count) { m_newton_iter = count; }

      private:
        friend class NewtonToleranceManager;
        bool   m_converged   = true;  // true if all checkers converged
        IndexT m_newton_iter = 0;
        SizeT  m_frame       = 0;  // current frame
    };

    class CheckResultInfo
    {
      public:
        // Finished iteration count
        IndexT newton_iter() const { return m_newton_iter; }
        void   converged(bool converged) { m_converged = converged; }

      private:
        friend class NewtonToleranceManager;
        bool   m_converged   = true;  // true if this checker converged
        IndexT m_newton_iter = 0;
    };

    class PreNewtonInfo
    {
      public:
        SizeT frame() const { return m_frame; }

      private:
        friend class NewtonToleranceManager;
        SizeT m_frame = 0;  // current frame
    };

    class Impl
    {
      public:
        void init();

        std::string                                     report_buffer;
        SimSystemSlotCollection<NewtonToleranceChecker> tolerance_checkers;
    };

    virtual void do_build() override;

    /// TEST SEAM (T6.4a, see the file header): the reference keeps the three
    /// engine-facing members private (`friend class SimEngine`); the unit test
    /// drives the same contract the reference's Newton loop does.
    /// `init()` is only be called by SimEngine.
    void init();
    void pre_newton(SizeT frame);
    void check(ResultInfo& info);

    friend class NewtonToleranceChecker;
    void add_checker(NewtonToleranceChecker* checker);
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
