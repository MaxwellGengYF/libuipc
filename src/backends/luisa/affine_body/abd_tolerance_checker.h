#pragma once
/********************************************************************
 * @file affine_body/abd_tolerance_checker.h
 * @brief T6.4a — the ABD Newton tolerance checker (port of
 *        src/backends/cuda/affine_body/abd_tolerance_checker.cu, 90 lines).
 *
 * WHAT IT IS
 * ----------
 * A `NewtonToleranceChecker` (the base this batch ports first) that converges the
 * Newton loop when **no ABD body's rotational/scale/shear dof** exceeded the
 * frame's absolute tolerance:
 *     abs_tol = newton/transrate_tol * dt
 *     fail    = OR over bodies, over components 3..11, of |dq_i(c)| > abs_tol
 * The three translational components (0..2) are **ignored on purpose** (the
 * reference's comment: "the first 3 components are translation, ignore; the rest 9
 * components are rotation/scaling/shear, take").
 *
 * The `dq` array is `AffineBodyDynamics::dqs()`; the per-thread decision and the
 * one-element `success` flag's "already failed stays failed" atomic are factored in
 * `abd_tolerance_checker_function.h` so the shipped shader and the unit test run the
 * same code.
 *
 * TWO PINNED REFERENCE QUIRKS (transcribed verbatim, asserted by the unit test):
 *  1. `do_check` declares a **local** `bool h_success`, which *shadows* the member
 *     `IndexT h_success = 1`.  The local feeds `info.converged(...)`; the member is
 *     never written after construction, so `do_report()` -- which reads the
 *     *member* -- always prints `Tol: < abs_tol`, even on a failed check.  The port
 *     keeps both names and the shadowing (MSVC's C4458 is a level-4 warning and this
 *     build is at the default level), so a reader comparing the reference with the
 *     port sees the same thing.
 *  2. `abs_tol` is recomputed at the top of every `do_check` from `transrate_tol`
 *     and the `dt` attribute (so a *stale* `abs_tol` is impossible, and `do_report`
 *     after a check reports the tolerance the check actually used).
 *
 * DEVIATIONS (reported):
 *  * the reference defines the class inside its `.cu`; the port splits the class
 *    into this header and `abd_tolerance_checker.cpp` so the unit test can
 *    `find<ABDToleranceChecker>()` and drive `do_check` directly with a hand-built
 *    `CheckResultInfo` (the T5.4b "class in the header, kernels in the .cpp" /
 *    "hooks are public" test-seam precedent, cookbook §7.20 rule 9).
 *  * the reference's implicit `bool h_success = success;` read-back is spelled
 *    `success.copy_to(&frame_stream()) != 0` -- the same value, but read on the
 *    stream the checker's kernel was recorded on (the §7.22 rule 7 hazard: a
 *    `LDeviceVar`'s implicit conversion reads on the *main* stream and may see a
 *    stale flag).
 *********************************************************************/
#include <newton_tolerance/newton_tolerance_checker.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/geometry/attribute_slot.h>
#include <luisa_device/buffer.h>

namespace uipc::backend::luisa
{
class ABDToleranceChecker final : public NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    SimSystemSlot<AffineBodyDynamics>       affine_body_dynamics;
    S<const geometry::AttributeSlot<Float>> dt_attr;
    Float                                   transrate_tol = 0.0;
    Float                                   abs_tol       = 0.0;
    lud::LDeviceVar<IndexT>                 success;
    IndexT                                  h_success = 1;  // 1 means success, 0 means failure

    // Inherited via NewtonToleranceChecker
    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override {}
    void do_pre_newton(PreNewtonInfo& info) override {}
    void do_check(CheckResultInfo& info) override;

    std::string do_report() override;
};
}  // namespace uipc::backend::luisa
