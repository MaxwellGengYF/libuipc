#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_driving_prismatic_joint.h
 * @brief T6.3b2 — the ABD driving prismatic joint (`ConstraintUID = 21`).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu
 * (class `AffineBodyDrivingPrismaticJoint`).
 *
 * WHAT THIS SYSTEM IS.  An `InterAffineBodyConstraint` (so `InterAffineBodyAnimator`
 * drives it) that *drives* the relative distance of a prismatic joint along its axis
 * to `aim_distance`.  It owns **no** geometry data of its own beyond the four
 * driving attributes the frontend writes on the *joint's* edges
 * (`uipc::constitution::AffineBodyDrivingPrismaticJoint::apply_to`):
 *
 *   driving/is_constrained   IndexT   (0 -> the pair contributes nothing)
 *   driving/strength_ratio   Float    (the kappa factor; 0 in the frontend's default)
 *   is_passive               IndexT   (1 -> aim at the *current* distance instead)
 *   aim_distance             Float    (the target distance)
 *
 * and it **reuses the joint's** `body_ids` / `rest_cs` / `rest_ts` /
 * `init_distances` / `current_distances` device buffers under a strict *index
 * alignment* assumption: both systems enumerate exactly the same `(geometry, edge)`
 * sequence, which the reference asserts in `do_init` (and so does the port — that
 * assert is the contract, not a convenience).
 *
 * DEVIATION (the T6.3b1 file split): the class declaration lives here, the kernels +
 * the member definitions + `REGISTER_SIM_SYSTEM` in the `.cpp`, so the unit test can
 * drive the reference's public hooks on the real device buffers (it is also the only
 * way to reach a driving kernel: it has no constitution-side energy hook at all).
 *********************************************************************/
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/constitutions/affine_body_prismatic_joint.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
class AffineBodyDrivingPrismaticJoint final : public InterAffineBodyConstraint
{
  public:
    using InterAffineBodyConstraint::InterAffineBodyConstraint;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;
    static constexpr SizeT StencilSize     = 2;

    static constexpr U64 ConstraintUID = 21;

    SimSystemSlot<AffineBodyPrismaticJoint> prismatic_joint;

    // Host
    // Note: body_ids / rest_cs / rest_ts / init_distances are reused directly
    // from AffineBodyPrismaticJoint under a strict index-alignment assumption
    // (see sanity check in do_init).
    OffsetCountCollection<IndexT> h_geo_joint_offsets_counts;

    vector<IndexT> h_is_constrained;
    vector<Float>  h_strength_ratios;
    vector<IndexT> h_is_passive;
    vector<Float>  h_aim_distances;

    // Device
    lud::LDeviceBuffer<IndexT> is_constrained;
    lud::LDeviceBuffer<Float>  strength_ratios;
    lud::LDeviceBuffer<IndexT> is_passive;
    lud::LDeviceBuffer<Float>  aim_distances;

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
