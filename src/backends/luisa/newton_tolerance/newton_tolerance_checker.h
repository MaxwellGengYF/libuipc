#pragma once
/********************************************************************
 * @file newton_tolerance/newton_tolerance_checker.h
 * @brief T6.4a — the Newton-tolerance checker base (port of
 *        src/backends/cuda/newton_tolerance/newton_tolerance_checker.h).
 *
 * A `NewtonToleranceChecker` is the reference's `IToleranceChecker`-style
 * contract for the Newton loop's convergence test: the derived class implements
 * `do_check(CheckResultInfo&)` (decide convergence, report it through
 * `info.converged(...)`) and `do_report()` (a one-line human readable residual),
 * the base registers it with the `NewtonToleranceManager` in `do_build()` and
 * forwards the manager's `pre_newton`/`check` fan-out.
 *
 * `Impl` is empty in the reference and is kept (the subclasses are free to use it;
 * neither of this batch's two concrete checkers does).
 *
 * DEVIATIONS (both test seams, reported):
 *  1. `init()`, `pre_newton()`, `check()` and `report()` are **public** (the
 *     reference keeps them private with `friend class NewtonToleranceManager`):
 *     the unit test drives a checker directly to pin the one-checker contract
 *     (`check()` -> `do_check` -> `info.converged()`), which the manager's
 *     short-circuit fan-out otherwise hides.  `do_build()` keeps the reference's
 *     `virtual void do_build() override;` declaration.
 *  2. `do_report()` stays exactly as the reference splits it: declared here
 *     (protected, default body `return "";`) and overridden by the concrete
 *     checkers.  The base's own default body is in the `.cpp`.
 *********************************************************************/
#include <sim_system.h>
#include <newton_tolerance/newton_tolerance_manager.h>

namespace uipc::backend::luisa
{
class NewtonToleranceChecker : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    using CheckResultInfo = NewtonToleranceManager::CheckResultInfo;
    using PreNewtonInfo   = NewtonToleranceManager::PreNewtonInfo;

    class Impl
    {
      public:
    };

  protected:
    virtual void        do_build(BuildInfo& info)          = 0;
    virtual void        do_init(InitInfo& info)            = 0;
    virtual void        do_pre_newton(PreNewtonInfo& info) = 0;
    virtual void        do_check(CheckResultInfo& info)    = 0;
    virtual std::string do_report();

  public:
    /// TEST SEAM (T6.4a, see the file header): private + `friend
    /// class NewtonToleranceManager` in the reference.
    virtual void do_build() override;
    void         init();  // only called by NewtonToleranceManager

    void        pre_newton(PreNewtonInfo& info);
    void        check(CheckResultInfo& info);
    std::string report();
    Impl        m_impl;
};
}  // namespace uipc::backend::luisa
