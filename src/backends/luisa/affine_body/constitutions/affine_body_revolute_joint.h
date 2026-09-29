#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_revolute_joint.h
 * @brief T6.3b3 — the ABD revolute joint (`ConstitutionUID = 18`) and its two
 *        satellite systems (the dof reporter and the time integrator).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu
 * (three of its seven classes: `AffineBodyRevoluteJoint`,
 * `AffineBodyRevoluteJointDofReporter`, `AffineBodyRevoluteJointTimeIntegrator`).
 *
 * DEVIATION (the T6.3b1/T6.3b2 file split, reported): the *class declarations* live in
 * this header and the kernels + the out-of-line member definitions + the
 * `REGISTER_SIM_SYSTEM` calls stay in the `.cpp`.  It is a pure file split whose only
 * motivation is that the unit test can reach the reference's *public* surface (the
 * joint's `h_*` host records, its device buffers and its `compute_current_angles` /
 * `adopt_scene_angles`) and its two satellite systems by type.  The seven classes of
 * the reference's `.cu` are split over four header/`.cpp` pairs, one per *uid*:
 *
 *   affine_body_revolute_joint.{h,cpp}                  uid 18 + dof reporter + integrator
 *   affine_body_driving_revolute_joint.{h,cpp}           uid 19
 *   affine_body_revolute_joint_external_force.{h,cpp}    uid 668 (constraint + reporter)
 *   affine_body_revolute_joint_limit.{h,cpp}             uid 670
 *
 * (The reference keeps them in one translation unit because its class bodies are
 * literally adjacent; every `REGISTER_SIM_SYSTEM` still appears exactly once.)
 *
 * WHAT THIS SYSTEM IS.  An `InterAffineBodyConstitution` that owns one *per-edge*
 * record — the two body ids, the four body-local rest points
 * (`rest_positions = [l_pos0 | l_pos1 | r_pos0 | r_pos1]`), the `strength_ratio`, the
 * two body-local axle-frame pairs `l_basis`/`r_basis` (`[n_bar | b_bar]`, the images
 * of the *world* `orthonormal_basis(t)` under `LT.rotation()^{-1}` — the T6.3b1 §7.25
 * rule 3 polar convention, re-pinned by this batch's scene case) and `init_angle` —
 * and, unlike T6.3b1's joints, a *persistent scalar state per joint*:
 * `current_angles`, maintained by `compute_current_angles()` and re-derived whenever
 * the geometry layout is rebuilt (`AdoptSceneAngles`/`compute_current_angles` from the
 * dof reporter's `do_update_dof_attributes`) or the state has been integrated (the
 * time integrator's `do_update_state`).  The frontend's `angle` edge attribute is the
 * *observable* form of that state; `init_angle` shifts it
 * (`current = relative + init`), and the joint-limit constitution's bounds are shifted
 * by the same `init_angle`.
 *
 * The angle is *unwrapped* against the previous stored value
 * (`unwrap_angle(theta, current - init)`), so per-frame rotations of less than pi are
 * tracked across turns; the reference warns when a frame's rotation approaches the pi
 * aliasing limit (the port routes that through the device-log facade — see the `.cpp`).
 *********************************************************************/
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constitutions/affine_body_revolute_joint_function.h>
#include <joint_dof_system/joint_dof_reporter.h>
#include <time_integrator/time_integrator.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class AffineBodyRevoluteJoint final : public InterAffineBodyConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 18ull;
    /// The reference's `2 * (2 + 1) / 2`: the 12x12-triplet run length of a two-body
    /// `TMA.half_block<2>` (the upper triangle of a 2x2 block of 12x12 blocks).
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;
    /// The reference's `StencilSize`: the number of `Vector12` doublets per joint.
    static constexpr SizeT StencilSize = 2;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    // [    body0   |   body1    ]
    lud::LDeviceBuffer<LVector2i> body_ids;
    // [l_pos0 | l_pos1 | r_pos0 | r_pos1] -- the reference's `X_bar` per pair
    lud::LDeviceBuffer<LVector12> rest_positions;
    lud::LDeviceBuffer<Float>     strength_ratio;

    /// [n_L, b_L] per joint -- the left body's body-local images of the pair's world
    /// orthonormal basis (the `t` direction is the axle, so only `n`/`b` are stored).
    lud::LDeviceBuffer<LVector6> l_basis;
    /// [n_R, b_R] per joint.
    lud::LDeviceBuffer<LVector6> r_basis;
    lud::LDeviceBuffer<Float>    init_angles;
    lud::LDeviceBuffer<Float>    current_angles;

    vector<Vector2i> h_body_ids;
    vector<Vector12> h_rest_positions;
    vector<Float>    h_strength_ratio;

    // Angle tracking (for all revolute joints)
    OffsetCountCollection<IndexT> h_geo_joint_offsets_counts;
    vector<Vector6>               h_l_basis;  // [n_L, b_L] per joint
    vector<Vector6>               h_r_basis;  // [n_R, b_R] per joint
    vector<Float>                 h_init_angles;
    vector<Float>                 h_current_angles;
    vector<Float>                 h_adopted_angles;

    BufferDump curr_angles_dump;

    /// The reference's public `compute_current_angles()`: `current = unwrap(theta,
    /// current - init) + init`, recomputed from the dynamics' `qs`.  Called by the
    /// joint's own `do_init`, by `write_scene`'s satellites below, and by the unit test.
    void compute_current_angles();

    /// The reference's public `adopt_scene_angles()`: overlay any *user-authored*
    /// `angle` attribute (a reset/teleport whose > pi jump cannot be unwrapped) onto
    /// the device state; `adopted` drives the upload.  Called by the dof reporter.
    void adopt_scene_angles();

    U64 get_uid() const noexcept override;

    void do_build(BuildInfo& info) override;
    void do_init(FilteredInfo& info) override;

    void do_report_energy_extent(EnergyExtentInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;

    bool do_dump(DumpInfo& info) override;
    bool do_try_recover(RecoverInfo& info) override;
    void do_apply_recover(RecoverInfo& info) override;
    void do_clear_recover(RecoverInfo& info) override;

    /// The reference's `write_scene()`: `current_angles` is copied back to the host and
    /// scattered into each geometry's `angle` edge attribute (the per-geometry slices
    /// come from `h_geo_joint_offsets_counts`).  Registered with `on_write_scene` in
    /// `do_build`, exactly like the reference.
    ///
    /// NOTE (unlike the prismatic sibling's, which the port made private): this one is
    /// **public** because the reference's `AffineBodyRevoluteJointDofReporter`
    /// calls it and the reference declares it in the class's public section — the
    /// DofReporter re-publishes the angles after adopting a user-authored value.
    void write_scene();
};

/// `JointDofReporter` (uid-less): re-syncs the joint's persistent per-joint state
/// (`current_angles`) with the geometry/body attribute layout.  The reference's
/// `do_update_dof_attributes` adopts user-authored angles, re-derives `current_angles`
/// from the new pose and republishes — all three steps, in that order.
class AffineBodyRevoluteJointDofReporter final : public JointDofReporter
{
  public:
    using JointDofReporter::JointDofReporter;

    SimSystemSlot<AffineBodyRevoluteJoint> revolute_joint;

    void do_build(BuildInfo& info) override;
    void do_update_dof_attributes(UpdateDofAttributesInfo& info) override;
};

/// `TimeIntegrator` (uid-less): the same re-derivation, once per frame after the dofs
/// have been integrated (`do_update_state`), with the reference's empty
/// `do_init`/`do_predict_dof` overrides kept.
class AffineBodyRevoluteJointTimeIntegrator final : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<AffineBodyRevoluteJoint> revolute_joint;
    SimSystemSlot<AffineBodyDynamics>      affine_body_dynamics;

    void do_init(InitInfo& info) override;
    void do_build(BuildInfo& info) override;
    void do_predict_dof(PredictDofInfo& info) override;
    void do_update_state(UpdateVelocityInfo& info) override;
};
}  // namespace uipc::backend::luisa
