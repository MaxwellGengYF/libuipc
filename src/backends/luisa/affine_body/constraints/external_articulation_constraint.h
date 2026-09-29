#pragma once
/********************************************************************
 * @file affine_body/constraints/external_articulation_constraint.h
 * @brief T6.4b — the ABD **external articulation** constraint: the empty
 * constitution (uid 23) it is paired with, the constraint itself (uid 24) and its
 * time integrator.
 *
 * REFERENCE: src/backends/cuda/affine_body/constraints/external_articulation_constraint.cu
 * (three of its classes: `ExternalArticulationConstituion`, `ExternalArticulationConstraint`,
 * `ExternalArticulationConstraintTimeIntegrator`) plus
 * `external_articulation_constraint_function.h`.
 *
 * WHAT THIS SYSTEM IS.  UIPC couples a *constraint* to a *constitution* (the constraint
 * is only built when the scene's constitution table carries its uid), and the external
 * articulation's constitution is deliberately **empty** (`uid 23`,
 * `InterAffineBody`-typed): the real material is the joint constitutions
 * (`AffineBodyRevoluteJoint` uid 18, `AffineBodyPrismaticJoint` uid 20) the articulation
 * *cross-references* by uid, exactly the way the joint-limit constitutions
 * cross-reference their base joint's `JointUID`.
 *
 * The constraint (uid 24) therefore does not constrain two bodies: it constrains the
 * **relative motion of joints**.  Its geometry (the frontend's
 * `ExternalArticulationConstraint::create_geometry`, uid 24 / constitution uid 23) owns
 * two attribute collections:
 *   `joint`        one entry per referenced (joint geometry, edge index) pair, with
 *                  `geo_id`, `index`, `delta_theta_tilde` (the animator's prediction) and
 *                  an output `delta_theta`;
 *   `joint_joint`  `n_joint^2` entries, all-pairs (i,j) with `i` **slower** than `j`
 *                  (`FastSegmentalReduce`'s ordering requirement), carrying the symmetric
 *                  coupling `mass`.
 * and it penalises
 *     E = 1/2 sum_ij (DeltaTheta_i - delta_theta_tilde_i) m_ij
 *                            (DeltaTheta_j - delta_theta_tilde_j)
 * where `DeltaTheta_i` is the wrapped relative-state change of joint `i` between the
 * previous and the current dofs (the generated bodies in
 * `external_articulation_constraint_function.h`).
 *
 * THE TWO UIDS, AND THE LOOKUP THIS CLASS PERFORMS.  `ConstraintUID = 24` (an
 * `InterAffineBodyConstraint`) and `ConstitutionUID = 23` (an `InterAffineBodyConstitution`)
 * are *different numbers on purpose* — the frontend registers both, the constraint is
 * filtered by the `constraint_uids` entry, the constitution is the (empty) owner of the
 * geometry group.  The articulation's own constitution `do_*` hooks report **zero**
 * energy/gradient/Hessian extents: all of the work is a *constraint*.
 *
 * The per-joint cross-reference is the data-level lookup the reference asserts in
 * `build_joint_info`: the joint geometry's `constitution_uid` meta must be 18 or 20
 * (`UIPC_ASSERT`), and the edge must carry the joint's own
 * `l_geo_id`/`r_geo_id`/`l_inst_id`/`r_inst_id` (and optionally the four
 * `l_position0/l_position1/r_position0/r_position1` body-local rest points, which
 * select the frontend's "local rest data" path over the "scene edge" path).
 *
 * DEVIATION (reported, the T6.3b1 file-split precedent): the class declarations live
 * here and the kernels + the out-of-line member definitions + the three
 * `REGISTER_SIM_SYSTEM` calls stay in the `.cpp`, so the unit test can reach the
 * reference's public surface (the `h_*` host records, the device buffers, the public
 * `joint_id_to_delta_theta`) and can drive the real hooks.
 *
 * SCOPE NOTE (T6.4b): this file owns *both* the articulation and its time integrator
 * because the reference's `do_update_state` writes the constraint's own
 * `joint_id_to_delta_theta` from the same (joint, dof) data its Hessian chain uses —
 * one uid, one pair, the T6.3b2 rule 1 convention.
 *********************************************************************/
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/constraints/external_articulation_constraint_function.h>
#include <time_integrator/time_integrator.h>
#include <utils/dump_utils.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
/**
 * @brief The (deliberately empty) `InterAffineBodyConstitution` the external
 * articulation constraint is paired with.
 *
 * NOTE (transcribed, not fixed): the reference's own class name is
 * `ExternalArticulationConstituion` (sic).  The typo is kept so that a grep of the
 * reference and of the port reads the same — the cookbook's "never fix a reference
 * inconsistency; transcribe + pin" rule.  Its *uid* (23) is the honest identifier, and
 * the frontend registers it as `"ExternalArticulationConstitution"`.
 */
class ExternalArticulationConstituion final : public InterAffineBodyConstitution
{
  public:
    static constexpr U64 ConstitutionUID = 23ull;

    /// The two joint constitutions this constitution/constraint pair cross-references
    /// by uid (the joint-limit constitutions' `JointUID` precedent).
    static constexpr U64 RevoluteJointConstitutionUID  = 18ull;
    static constexpr U64 PrismaticJointConstitutionUID = 20ull;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    void do_build(BuildInfo& info) override;
    void do_init(FilteredInfo& info) override;
    void do_report_energy_extent(EnergyExtentInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;
    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;

    U64 get_uid() const noexcept override;
};

/**
 * @brief The external articulation constraint (uid 24).
 *
 * See the file banner for the model.  Every buffer below is the reference's, in the
 * reference's order; the two additions the port needs are marked `PORT ADDITION`.
 */
class ExternalArticulationConstraint final : public InterAffineBodyConstraint
{
  public:
    static constexpr U64   ConstraintUID   = 24ull;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstraint::InterAffineBodyConstraint;

    /// The two joint constitutions the `joint` entries cross-reference (see
    /// `ExternalArticulationConstituion`).
    static constexpr U64 RevoluteJointConstitutionUID  = 18ull;
    static constexpr U64 PrismaticJointConstitutionUID = 20ull;

    /// The per-joint `delta_theta_tilde` / `mass` attribute slots, kept so `do_step`
    /// can re-read them every frame (they are the animator's outputs).
    vector<S<geometry::AttributeSlot<Float>>> h_art_joint_joint_mass;
    vector<S<geometry::AttributeSlot<Float>>> h_art_joint_delta_theta_tilde;

    // ---- per joint ---------------------------------------------------------
    OffsetCountCollection<IndexT> h_art_id_to_joint_offsets_counts;
    vector<IndexT>                h_joint_id_to_art_id;
    vector<U64>                   h_joint_id_to_uid;
    vector<Vector2i>              h_joint_id_to_body_ids;
    vector<Float>                 h_joint_id_to_delta_theta;
    vector<Float>                 h_joint_id_to_delta_theta_tilde;

    // ---- per joint-pair ----------------------------------------------------
    OffsetCountCollection<IndexT> h_art_id_to_joint_joint_offsets_counts;
    vector<IndexT>                h_joint_joint_id_to_art_id;
    vector<Vector2i>              h_joint_joint_id_to_joint_ij;
    vector<Float>                 h_joint_joint_id_to_mass;

    lud::LDeviceBuffer<IndexT>   art_id_to_joint_offsets;
    lud::LDeviceBuffer<IndexT>   art_id_to_joint_counts;
    lud::LDeviceBuffer<IndexT>   joint_id_to_art_id;
    lud::LDeviceBuffer<U64>      joint_id_to_uid;
    lud::LDeviceBuffer<LVector2i> joint_id_to_body_ids;
    lud::LDeviceBuffer<Float>    joint_id_to_delta_theta;
    lud::LDeviceBuffer<Float>    joint_id_to_delta_theta_tilde;

    lud::LDeviceBuffer<IndexT>   art_id_to_joint_joint_offsets;
    lud::LDeviceBuffer<IndexT>   art_id_to_joint_joint_counts;
    lud::LDeviceBuffer<LVector2i> joint_joint_id_to_joint_ij;
    lud::LDeviceBuffer<Float>    joint_joint_id_to_mass;

    /// PORT ADDITION (T6.4b): the segmented reduce's *keys* — `ij(0)` of every
    /// joint-joint entry, which are static (the joint layout never changes) and
    /// therefore materialised once in `do_init`.  The reference's
    /// `FastSegmentalReduce` computes its key with an in-kernel functor and needs no
    /// buffer; `lla::segmented_reduce` reads the sorted keys from a buffer.  Ascending
    /// by construction (see the `_function.h` banner).
    vector<std::int32_t>             h_joint_joint_id_to_key;
    lud::LDeviceBuffer<std::int32_t> joint_joint_id_to_key;

    /// PORT ADDITION (T6.4b): the segmented reduce's *values*, one per joint-joint
    /// entry — `m_ij * (DeltaTheta_j - delta_theta_tilde_j)`, written by
    /// `..._compute_reduce_value_kernel` (the reference's `GetValue` functor, which
    /// `lla::segmented_reduce` cannot take).
    lud::LDeviceBuffer<Float> joint_joint_id_to_value;

    // ---- the per-body reference state (see the reference's `get_ref_q_prev`) ----
    unordered_map<IndexT, IndexT> h_body_id_to_ref_q_prev_id;
    using AttrRefQPrev = S<const geometry::AttributeSlot<Vector12>>;
    vector<std::tuple<AttrRefQPrev, IndexT>> h_attr_ref_q_prevs;

    /// `joint_id -> [l_ref_q_prev_id, r_ref_q_prev_id]` (`-1` = "use the dynamics'
    /// `q_prevs` for that side").
    vector<Vector2i>             h_joint_id_to_ref_q_prev_ids;
    lud::LDeviceBuffer<LVector2i> joint_id_to_ref_q_prev_ids;

    vector<Vector12>             h_ref_q_prevs;
    lud::LDeviceBuffer<LVector12> ref_q_prevs;

    /// Joint basis:
    ///   prismatic: `[c_bar ; t_bar]` (a body-local point and a direction)
    ///   revolute : `[bn_bar ; tn_bar]` (two body-local frame rows)
    vector<Vector6>            h_joint_id_to_L_basis;
    vector<Vector6>            h_joint_id_to_R_basis;
    lud::LDeviceBuffer<LVector6> joint_id_to_L_basis;
    lud::LDeviceBuffer<LVector6> joint_id_to_R_basis;

    /// `G^theta` per joint: the segmented reduce's output and the gradient/Hessian
    /// kernels' input.
    lud::LDeviceBuffer<Float> joint_id_to_G_theta;

    /// The reference's own dump/recover state (its name kept verbatim).
    BufferDump dump_delta_theta;

    void do_build(BuildInfo& info) override;
    void do_init(InterAffineBodyAnimator::FilteredInfo& info) override;
    void do_step(InterAffineBodyAnimator::FilteredInfo& info) override;
    void do_report_extent(InterAffineBodyAnimator::ReportExtentInfo& info) override;
    void do_compute_energy(InterAffineBodyAnimator::ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(
        InterAffineBodyAnimator::GradientHessianInfo& info) override;
    bool do_dump(DumpInfo& info) override;
    bool do_try_recover(RecoverInfo& info) override;
    void do_apply_recover(RecoverInfo& info) override;
    void do_clear_recover(RecoverInfo& info) override;

    U64 get_uid() const noexcept override;

    /// The reference's two static basis builders (the "no local rest data" path: the
    /// axle is taken from the *joint mesh* edge, mapped into each body's initial
    /// frame).  Public because the unit test derives the same basis and compares.
    static auto get_revolute_basis(const geometry::SimplicialComplex* L,
                                   IndexT                           L_inst_id,
                                   const geometry::SimplicialComplex* R,
                                   IndexT                           R_inst_id,
                                   const geometry::SimplicialComplex* joint_mesh,
                                   IndexT                           joint_index);

    static auto get_prismatic_basis(const geometry::SimplicialComplex* L,
                                    IndexT                           L_inst_id,
                                    const geometry::SimplicialComplex* R,
                                    IndexT                           R_inst_id,
                                    const geometry::SimplicialComplex* joint_mesh,
                                    IndexT                           joint_index);

    /// The reference's `get_ref_q_prev` (public for the test: it is the only place the
    /// `ref_dof_prev` attribute protocol is visible).
    IndexT get_ref_q_prev(const geometry::SimplicialComplex* geo,
                          IndexT                           inst_id,
                          IndexT                           body_id);

  private:
    /// The reference's private `write_scene`: `joint_id_to_delta_theta` is copied back
    /// to the host and scattered into each articulation geometry's `delta_theta`
    /// attribute (registered with `on_write_scene` in `do_build`).
    void write_scene();

    /// The reference's `build_joint_info` (one `joint` entry).
    void build_joint_info(span<S<geometry::GeometrySlot>>        geo_slots,
                          InterAffineBodyAnimator::FilteredInfo& info,
                          const geometry::Geometry&              mesh,
                          IndexT                                 index);
};

/**
 * @brief The articulation's time integrator (uid-less): after the dofs have been
 * integrated it re-derives every joint's `delta_theta` from the current/previous state
 * and stores it in the constraint's `joint_id_to_delta_theta` (which `write_scene`
 * then publishes as the frontend's `delta_theta` attribute).
 */
class ExternalArticulationConstraintTimeIntegrator final : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<ExternalArticulationConstraint> constraint;
    SimSystemSlot<AffineBodyDynamics>             affine_body_dynamics;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_predict_dof(PredictDofInfo& info) override;
    void do_update_state(UpdateVelocityInfo& info) override;
};
}  // namespace uipc::backend::luisa
