#pragma once
/********************************************************************
 * @file soft_position_constraint.h
 * @brief T5.3 (pulled forward from T5.4) — the soft-position constraint (port of
 * src/backends/cuda/finite_element/constraints/soft_position_constraint.cu).
 *
 * MANDATORY in T5.3 (the brief's §1.1 warning): `SoftPositionConstraintUID = 14`
 * and `luisa_half_plane_animator_test.cpp` applies this constraint to four of its
 * meshes (`mesh.with_constraint = true`).  Once `FiniteElementAnimator` exists, a
 * scene that has *both* a `FiniteElement` and a `Constraint` constitution builds
 * the animator and its `Impl::init()` walks every FEM geometry's
 * `builtin::constraint_uids` -- aborting on a uid no `FiniteElementConstraint`
 * claims.  (In the half-plane scenes there is no `FiniteElement` constitution at
 * all, so `FiniteElementMethod` -- and therefore the animator and this constraint
 * -- are invalidated before that walk; the port is nevertheless required, and its
 * numbers must not move: this constraint only refreshes its own
 * `aim_positions`/`strength_ratios` buffers in `do_step`, it never writes
 * `xs`/`x_tildes`, so it is inert until a Phase-8 solver asks it for an
 * energy/gradient/Hessian.)
 *
 * DEVIATION (reported): the reference declares the class *and* its kernels inside
 * the `.cu`; the port keeps the reference's public surface (`get_uid`, the three
 * host staging vectors, the three device buffers, the five hooks) in this header
 * and the kernels + definitions in the `.cpp`, exactly like the T5.2 constitutions
 * (cookbook §7.17 rule 11), so the unit test can drive the hooks.
 *
 * The per-vertex law (the `lerp` is the *clamped* `animator::lerp`, cookbook
 * §7.14 rule 4 -- the clamped law is what holds the aim position once
 * `substep_ratio` reaches 1):
 *   aim_x = lerp(x_prev, aim_positions(I), substep_ratio)
 *   dx    = x - aim_x
 *   E     = is_fixed(i) ? 0 : 0.5 * s * m * dot(dx, dx)
 *   G     = is_fixed(i) ? 0 : s * m * dx
 *   H     = is_fixed(i) ? 0 : (s * m) * I
 * NOTE the difference from the kinetic systems: here the Hessian *is* guarded by
 * `is_fixed`.  That is the reference's kernel, verbatim, and the unit test asserts
 * it (`fem_dynamics_soft_position_constraint`).
 *********************************************************************/
#include <finite_element/finite_element_constraint.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class SoftPositionConstraint final : public FiniteElementConstraint
{
    static constexpr U64 SoftPositionConstraintUID = 14ull;

  public:
    using FiniteElementConstraint::FiniteElementConstraint;

    vector<IndexT>  h_constrained_vertices;
    vector<Vector3> h_aim_positions;
    vector<Float>   h_strength_ratios;

    lud::LDeviceBuffer<IndexT>   constrained_vertices;
    lud::LDeviceBuffer<LVector3> aim_positions;
    lud::LDeviceBuffer<Float>    strength_ratios;

    void do_build(BuildInfo& info) override;
    U64  get_uid() const noexcept override { return SoftPositionConstraintUID; }
    void do_init(FiniteElementAnimator::FilteredInfo& info) override;
    void do_step(FiniteElementAnimator::FilteredInfo& info) override;
    void do_report_extent(FiniteElementAnimator::ReportExtentInfo& info) override;
    void do_compute_energy(FiniteElementAnimator::ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(FiniteElementAnimator::ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
