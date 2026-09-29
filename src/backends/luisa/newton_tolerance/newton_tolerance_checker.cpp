#include <newton_tolerance/newton_tolerance_checker.h>
#include <newton_tolerance/newton_tolerance_manager.h>

/*************************************************************************************************
* T6.4a — `NewtonToleranceChecker`'s five forwarding bodies (port of
* src/backends/cuda/newton_tolerance/newton_tolerance_checker.cu, 40 lines).
*
* Everything here is the reference's, statement for statement:
*  * `do_build()` resolves the manager, lets the subclass build (`do_build(info)`)
*    and *then* registers (`manager.add_checker(this)`) -- the registration is
*    deliberately last, so a subclass whose `do_build` throws (`require<>` of a
*    missing system) never enters the manager's collection;
*  * `init()`/`pre_newton()`/`check()`/`report()` are one-line forwards.
*
* The reference's `pre_newton(PreNewtonInfo& info)` ignores its argument and
* constructs a *fresh* default `PreNewtonInfo` for the subclass (the manager fills
* its own copy) -- transcribed verbatim, including that unused argument.
*************************************************************************************************/
namespace uipc::backend::luisa
{
std::string NewtonToleranceChecker::do_report()
{
    return "";
}

void NewtonToleranceChecker::do_build()
{
    auto& manager = require<NewtonToleranceManager>();

    BuildInfo info;
    do_build(info);  // let derived classes build their specific info

    manager.add_checker(this);  // Register this checker with the manager
}

void NewtonToleranceChecker::init()
{
    InitInfo info;
    do_init(info);  // let derived classes initialize their specific info
}

void NewtonToleranceChecker::pre_newton(PreNewtonInfo& info)
{
    (void)info;  // the reference's parameter is unused (a fresh info is built)
    PreNewtonInfo pre_info;
    do_pre_newton(pre_info);  // let derived classes prepare for newton step
}

void NewtonToleranceChecker::check(CheckResultInfo& info)
{
    do_check(info);
}

std::string NewtonToleranceChecker::report()
{
    return do_report();
}
}  // namespace uipc::backend::luisa
