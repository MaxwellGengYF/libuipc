#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_driving_revolute_joint.h
 * @brief T6.3b3 — the ABD driving revolute joint (`ConstraintUID = 19`).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu
 * (class `AffineBodyDrivingRevoluteJoint`).
 *
 * WHAT THIS SYSTEM IS.  An `InterAffineBodyConstraint` (so `InterAffineBodyAnimator`
 * drives it) that *drives* the revolute joint's relative angle about its axle to
 * `aim_angle`.  It owns **no** geometry data of its own beyond the four driving
 * attributes the frontend writes on the *joint's* edges
 * (`uipc::constitution::AffineBodyDrivingRevoluteJoint::apply_to`):
 *
 *   driving/is_constrained   IndexT   (0 -> the pair contributes nothing)
 *   driving/strength_ratio   Float    (the kappa factor; 0 in the frontend's default)
 *   is_passive               IndexT   (1 -> aim at the *current* angle instead)
 *   aim_angle                Float    (the target angle)
 *
 * and it **reuses the joint's** `body_ids` / `l_basis` / `r_basis` / `init_angles` /
 * `current_angles` device buffers under a strict *index alignment* assumption: both
 * systems enumerate exactly the same `(geometry, edge)` sequence, which the reference
 * asserts in `do_init` (and so does the port — that assert is the contract, not a
 * convenience).
 *
 * THE ANGLE SEMANTICS (the one real difference from the prismatic sibling).  The
 * energy is `1/2 kappa (theta - theta_tilde)^2`, where `theta = atan2(s, c)` is the
 * pair's relative rotation about the axle *as measured this frame* — i.e. wrapped into
 * (-pi, pi] — and `theta_tilde` is a *parameter* produced by the ported
 * `driving_theta_tilde(F01_q, aim_angle, init_angle, current_angle)`.  That helper
 * unwraps the freshly measured angle against the joint's stored multi-turn angle
 * (`current_angle - init_angle`) and re-folds it into `theta_tilde`, so the difference
 * `theta - theta_tilde` equals `aim_angle - init_angle - (unwrapped current)` — the
 * graded quantity is the *unwrapped* angle difference, while `theta`'s own derivative
 * is the wrapped one (the fold is a constant within a frame, so the reference's
 * comment that "the gradient is unaffected" holds).
 *
 * DEVIATION (the T6.3b1/T6.3b2 file split): the class declaration lives here, the
 * kernels + the member definitions + `REGISTER_SIM_SYSTEM` in the `.cpp`, so the unit
 * test can drive the reference's public hooks on the real device buffers (it is also
 * the only way to reach a driving kernel: it has no constitution-side energy hook at
 * all).
 *********************************************************************/
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/constitutions/affine_body_revolute_joint.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
class AffineBodyDrivingRevoluteJoint final : public InterAffineBodyConstraint
{
  public:
    using InterAffineBodyConstraint::InterAffineBodyConstraint;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;
    static constexpr SizeT StencilSize     = 2;

    static constexpr U64 ConstraintUID = 19;

    SimSystemSlot<AffineBodyRevoluteJoint> revolute_joint;

    // Host
    // Note: body_ids / l_basis / r_basis / init_angles / current_angles are reused
    // directly from AffineBodyRevoluteJoint under a strict index-alignment assumption
    // (see sanity check in do_init).
    OffsetCountCollection<IndexT> h_geo_joint_offsets_counts;

    vector<IndexT> h_is_constrained;
    vector<Float>  h_strength_ratios;
    vector<IndexT> h_is_passive;
    vector<Float>  h_aim_angles;

    // Device
    lud::LDeviceBuffer<IndexT> is_constrained;
    lud::LDeviceBuffer<Float>  strength_ratios;
    lud::LDeviceBuffer<IndexT> is_passive;
    lud::LDeviceBuffer<Float>  aim_angles;

    void do_build(BuildInfo& info) override;
    void do_init(InterAffineBodyAnimator::FilteredInfo& info) override;
    void do_step(InterAffineBodyAnimator::FilteredInfo& info) override;

    void do_report_extent(InterAffineBodyAnimator::ReportExtentInfo& info) override;
    void do_compute_energy(InterAffineBodyAnimator::ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(
        InterAffineBodyAnimator::GradientHessianInfo& info) override;

    U64 get_uid() const noexcept override;
};
}  // namespace uipc::backend::luisa
