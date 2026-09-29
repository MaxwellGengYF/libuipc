#pragma once
/********************************************************************
 * @file newton_tolerance/max_translation_checker.h
 * @brief T6.4a — the reference's whole-system Newton tolerance checker (port of
 *        src/backends/cuda/newton_tolerance/max_translation_checker.cu, 67 lines).
 *
 * WHAT IT IS
 * ----------
 * A `NewtonToleranceChecker` that converges when the frame's **maximum axis
 * displacement** of the global vertex set is within a velocity tolerance scaled by
 * `dt`:
 *     vel_tol = (newton/velocity_tol_relative > 0)
 *                 ? newton/velocity_tol_relative * scene_diagonal   (Stiff-GIPC)
 *                 : newton/velocity_tol
 *     abs_tol = vel_tol * dt
 *     res     = GlobalVertexManager::compute_axis_max_displacement()
 *     converged = res <= abs_tol
 * It also records `res0` (the residual at `newton_iter == 0`) -- the reference's
 * hand-off to a future step-relative criterion; nothing consumes it yet, exactly as
 * in the reference.
 *
 * DEVIATION (a test seam, reported): the reference defines this class inside its
 * `.cu` (nothing else can name it).  The port splits the class into this header and
 * `max_translation_checker.cpp` so the unit test can `find<MaxTranslationChecker>()`
 * and pin the comparison above (`res`, `abs_tol`, `res0`, `frame`) against a
 * hand-derived oracle -- the T5.4b "class + Impl in the header, kernels in the .cpp"
 * precedent (cookbook §7.20).  Every member is public here, exactly as the
 * reference's class has it (its members precede no access specifier change).
 *
 * The one infrastructure hook the class needs already exists in the port: the
 * T1.4/T4.1 `GlobalVertexManager` declares `friend class MaxTranslationChecker;`
 * and `compute_axis_max_displacement()` (a private engine-facing member), so
 * `do_check` reaches it without any new API.
 *********************************************************************/
#include <newton_tolerance/newton_tolerance_checker.h>
#include <uipc/geometry/attribute_slot.h>
#include <global_geometry/global_vertex_manager.h>

namespace uipc::backend::luisa
{
class MaxTranslationChecker final : public NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    SimSystemSlot<GlobalVertexManager>      vertex_manager;
    S<const geometry::AttributeSlot<Float>> dt_attr;
    Float                                   newton_velocity_tol   = 0.0;
    Float                                   velocity_tol_relative = 0.0;
    Float                                   res0                  = 0.0;
    Float                                   res                   = 0.0;
    Float                                   rel_tol               = 0.0;
    Float                                   abs_tol               = 0.0;
    SizeT                                   frame                 = 0;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override {}
    void do_pre_newton(PreNewtonInfo& info) override;
    void do_check(CheckResultInfo& info) override;

    std::string do_report() override;
};
}  // namespace uipc::backend::luisa
