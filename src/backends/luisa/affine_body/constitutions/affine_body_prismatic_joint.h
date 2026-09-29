#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_prismatic_joint.h
 * @brief T6.3b2 — the ABD prismatic joint (`ConstitutionUID = 20`) and its two
 *        satellite systems (the dof reporter and the time integrator).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu
 * (three of its seven classes: `AffineBodyPrismaticJoint`,
 * `AffineBodyPrismaticJointDofReporter`, `AffineBodyPrismaticJointTimeIntegrator`).
 *
 * DEVIATION (the T6.3b1 precedent, reported): the *class declarations* live in this
 * header and the kernels + the out-of-line member definitions + the
 * `REGISTER_SIM_SYSTEM` calls stay in the `.cpp`.  It is a pure file split whose only
 * motivation is that the unit test can reach the reference's *public* surface (the
 * joint's `h_*` host records, its device buffers and its `compute_current_distances`)
 * and its two satellite systems by type.  The seven classes of the reference's `.cu`
 * are split over four header/`.cpp` pairs, one per *uid*:
 *
 *   affine_body_prismatic_joint.{h,cpp}           uid 20  + dof reporter + integrator
 *   affine_body_driving_prismatic_joint.{h,cpp}   uid 21
 *   affine_body_prismatic_joint_external_force.{h,cpp}   uid 667 (constraint + reporter)
 *   affine_body_prismatic_joint_limit.{h,cpp}     uid 669
 *
 * (The reference keeps them in one translation unit only because its class bodies are
 * literally adjacent; every `REGISTER_SIM_SYSTEM` still appears exactly once.)
 *
 * WHAT THIS SYSTEM IS.  An `InterAffineBodyConstitution` that owns one *per-edge*
 * record (the two body ids, the four body-local `Vector6` rest rows, the frontend's
 * `strength_ratio` and its `init_distance`) and — unlike T6.3b1's joints — a
 * *persistent scalar state per joint*: `current_distances`, maintained by
 * `compute_current_distances()` and re-derived whenever the geometry layout is
 * rebuilt (the dof reporter's `do_update_dof_attributes`) or the state is written
 * back (the time integrator's `do_update_state`).  The frontend's `distance` edge
 * attribute is the *observable* form of that state; `init_distance` shifts it
 * (`current = absolute + init`), and the joint-limit constitution's bounds are
 * shifted by the same `init_distance`.
 *
 * The per-joint rest data is the frontend's convention
 * (`uipc::constitution::AffineBodyPrismaticJoint::create_geometry`/`apply_to`):
 * `rest_cs` holds the two attachment points in each body's *local* frame
 * (`[ci_bar | cj_bar]` — from the `l_position0`/`r_position0` edge attributes when
 * the frontend wrote them, otherwise `LT.inverse() * P0` from the scene edge's first
 * vertex), and `rest_ts/rest_ns/rest_bs` hold the body-local images of the edge
 * direction and its two orthonormal companions (`orthonormal_basis`), i.e.
 * `[L^{-1} t | L^{-1} t]` etc. with `L = LT.rotation()` — the **polar** rotation of
 * the body's initial transform (T6.3b1 §7.25 rule 3's finding, re-pinned by this
 * batch's scene case).
 *********************************************************************/
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constitutions/affine_body_prismatic_joint_function.h>
#include <joint_dof_system/joint_dof_reporter.h>
#include <time_integrator/time_integrator.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class AffineBodyPrismaticJoint final : public InterAffineBodyConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 20ull;
    /// The reference's `2 * (2 + 1) / 2`: the 12x12-triplet run length of a two-body
    /// `TMA.half_block<2>` (the upper triangle of a 2x2 block of 12x12 blocks).
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    // [    body0   |   body1    ]
    lud::LDeviceBuffer<LVector2i> body_ids;
    lud::LDeviceBuffer<LVector6>  rest_cs;  // c_bar
    lud::LDeviceBuffer<LVector6>  rest_ts;  // t_bar
    lud::LDeviceBuffer<LVector6>  rest_ns;  // n_bar
    lud::LDeviceBuffer<LVector6>  rest_bs;  // b_bar
    lud::LDeviceBuffer<Float>     strength_ratios;

    vector<Vector2i> h_body_ids;
    vector<Vector6>  h_rest_cs;
    vector<Vector6>  h_rest_ts;
    vector<Vector6>  h_rest_ns;
    vector<Vector6>  h_rest_bs;
    vector<Float>    h_strength_ratios;

    // Distance tracking (for all prismatic joints)
    OffsetCountCollection<IndexT> h_geo_joint_offsets_counts;
    vector<Float>                 h_init_distances;
    vector<Float>                 h_current_distances;

    lud::LDeviceBuffer<Float> init_distances;
    lud::LDeviceBuffer<Float> current_distances;

    BufferDump curr_distances_dump;

    /// The reference's public `compute_current_distances()`: `current = absolute +
    /// init`, recomputed from the dynamics' `qs`.  Called by the joint's own
    /// `do_init` and `write_scene`, and by the two satellite systems below.
    void compute_current_distances();

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

  private:
    /// The reference's private `write_scene()`: `current_distances` is copied back to
    /// the host and scattered into each geometry's `distance` edge attribute (the
    /// per-geometry slices come from `h_geo_joint_offsets_counts`).  Registered with
    /// `on_write_scene` in `do_build`, exactly like the reference.
    void write_scene();
};

/// `JointDofReporter` (uid-less): re-syncs the joint's persistent per-joint state
/// (`current_distances`) with the geometry/body attribute layout.  The reference's
/// class has exactly this one hook plus the `require<>`.
class AffineBodyPrismaticJointDofReporter final : public JointDofReporter
{
  public:
    using JointDofReporter::JointDofReporter;

    SimSystemSlot<AffineBodyPrismaticJoint> prismatic_joint;

    void do_build(BuildInfo& info) override;
    void do_update_dof_attributes(UpdateDofAttributesInfo& info) override;
};

/// `TimeIntegrator` (uid-less): the same re-sync, once per frame after the dofs have
/// been integrated (`do_update_state`), with the reference's empty
/// `do_init`/`do_predict_dof` overrides kept.
class AffineBodyPrismaticJointTimeIntegrator final : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<AffineBodyPrismaticJoint> prismatic_joint;
    SimSystemSlot<AffineBodyDynamics>       affine_body_dynamics;

    void do_init(InitInfo& info) override;
    void do_build(BuildInfo& info) override;
    void do_predict_dof(PredictDofInfo& info) override;
    void do_update_state(UpdateVelocityInfo& info) override;
};
}  // namespace uipc::backend::luisa
