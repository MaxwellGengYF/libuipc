#pragma once
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/abd_time_integrator.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
/**
 * @brief ExternalArticulationConstituion
 * 
 * Define an empty constitution for ExternalArticulationConstraint, 
 * because Constraint should be associated with a Constitution in UIPC design.
 * 
 * The actual constitutions are based on
 * - AffineBodyRevoluteJointConstitution (UID=18)
 * - AffineBodyPrismaticJointConstitution (UID=20)
 * 
 */
class ExternalArticulationConstituion final : public InterAffineBodyConstitution
{
  public:
    static constexpr U64 ConstitutionUID = 23ull;
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
 * @brief ExternalArticulationConstraint
 * 
 * Constraint for external articulation joints (revolute and prismatic).
 * Computes energy, gradient, and hessian for joint constraints.
 */
class ExternalArticulationConstraint final : public InterAffineBodyConstraint
{
  public:
    static constexpr U64   ConstraintUID   = 24ull;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstraint::InterAffineBodyConstraint;

    // Host-side attribute slots for per-step updates
    vector<S<geometry::AttributeSlot<Float>>> h_art_joint_joint_mass;
    vector<S<geometry::AttributeSlot<Float>>> h_art_joint_delta_theta_tilde;

    // Host-side data structures
    OffsetCountCollection<IndexT> h_art_id_to_joint_offsets_counts;
    vector<IndexT>                h_joint_id_to_art_id;
    vector<U64>                   h_joint_id_to_uid;
    vector<Vector2i>              h_joint_id_to_body_ids;
    vector<Float>                 h_joint_id_to_delta_theta;
    vector<Float>                 h_joint_id_to_delta_theta_tilde;

    OffsetCountCollection<IndexT> h_art_id_to_joint_joint_offsets_counts;
    vector<IndexT>                h_joint_joint_id_to_art_id;
    vector<Vector2i>              h_joint_joint_id_to_joint_ij;
    vector<Float>                 h_joint_joint_id_to_mass;

    // Device buffers
    luisa::compute::Buffer<IndexT>   art_id_to_joint_offsets;
    luisa::compute::Buffer<IndexT>   art_id_to_joint_counts;
    luisa::compute::Buffer<IndexT>   joint_id_to_art_id;
    luisa::compute::Buffer<U64>      joint_id_to_uid;
    luisa::compute::Buffer<Vector2i> joint_id_to_body_ids;
    luisa::compute::Buffer<Float>    joint_id_to_delta_theta;
    luisa::compute::Buffer<Float>    joint_id_to_delta_theta_tilde;

    luisa::compute::Buffer<IndexT>   art_id_to_joint_joint_offsets;
    luisa::compute::Buffer<IndexT>   art_id_to_joint_joint_counts;
    luisa::compute::Buffer<Vector2i> joint_joint_id_to_joint_ij;
    luisa::compute::Buffer<Float>    joint_joint_id_to_mass;

    // Reference q_prev data
    unordered_map<IndexT, IndexT> h_body_id_to_ref_q_prev_id;
    using AttrRefQPrev = S<const geometry::AttributeSlot<Vector12>>;
    vector<std::tuple<AttrRefQPrev, IndexT>> h_attr_ref_q_prevs;

    vector<Vector2i>             h_joint_id_to_ref_q_prev_ids;
    luisa::compute::Buffer<Vector2i> joint_id_to_ref_q_prev_ids;

    vector<Vector12>             h_ref_q_prevs;
    luisa::compute::Buffer<Vector12> ref_q_prevs;

    // Joint Basis
    vector<Vector6>             h_joint_id_to_L_basis;
    vector<Vector6>             h_joint_id_to_R_basis;
    luisa::compute::Buffer<Vector6> joint_id_to_L_basis;
    luisa::compute::Buffer<Vector6> joint_id_to_R_basis;

    // G^theta for each joint
    luisa::compute::Buffer<Float> joint_id_to_G_theta;

    void do_build(BuildInfo& info) override;
    U64 get_uid() const noexcept override;
    void do_init(InterAffineBodyAnimator::FilteredInfo& info) override;
    void do_step(InterAffineBodyAnimator::FilteredInfo& info) override;
    void do_report_extent(InterAffineBodyAnimator::ReportExtentInfo& info) override;
    void do_compute_energy(InterAffineBodyAnimator::ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(InterAffineBodyAnimator::GradientHessianInfo& info) override;

    bool do_dump(DumpInfo& info) override;
    bool do_try_recover(RecoverInfo& info) override;
    void do_apply_recover(RecoverInfo& info) override;
    void do_clear_recover(RecoverInfo& info) override;

  private:
    static auto get_revolute_basis(const geometry::SimplicialComplex* L,
                                   IndexT                             L_inst_id,
                                   const geometry::SimplicialComplex* R,
                                   IndexT                             R_inst_id,
                                   const geometry::SimplicialComplex* joint_mesh,
                                   IndexT joint_index);

    static auto get_prismatic_basis(const geometry::SimplicialComplex* L,
                                    IndexT L_inst_id,
                                    const geometry::SimplicialComplex* R,
                                    IndexT R_inst_id,
                                    const geometry::SimplicialComplex* joint_mesh,
                                    IndexT joint_index);

    auto get_ref_q_prev(const geometry::SimplicialComplex* geo, IndexT inst_id, IndexT body_id);

    void build_joint_info(span<S<geometry::GeometrySlot>>        geo_slots,
                          InterAffineBodyAnimator::FilteredInfo& info,
                          const geometry::Geometry&              mesh,
                          IndexT                                 index);

    void write_scene();

    BufferDump dump_delta_theta;
};

/**
 * @brief ExternalArticulationConstraintTimeIntegrator
 * 
 * Time integrator for external articulation constraints.
 * Computes delta_theta during velocity update.
 */
class ExternalArticulationConstraintTimeIntegrator final : public ABDTimeIntegrator
{
  public:
    using ABDTimeIntegrator::ABDTimeIntegrator;

    SimSystemSlot<ExternalArticulationConstraint> constraint;
    SimSystemSlot<AffineBodyDynamics>             affine_body_dynamics;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_predict_dof(PredictDofInfo& info) override;
    void do_update_state(UpdateVelocityInfo& info) override;
};
}  // namespace uipc::backend::luisa
