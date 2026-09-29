#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_revolute_joint_external_force.h
 * @brief T6.3b3 — the revolute joint's external-force pair: the constraint
 *        (`ConstraintUID = 668`) and the reporter (`ReporterUID = 668`).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu
 * (classes `AffineBodyRevoluteJointExternalForceConstraint` and
 * `AffineBodyRevoluteJointExternalForce`).
 *
 * THE 668 SHARING IS DELIBERATE (the T5.4b `671` / T6.3b2 `667` precedent, cookbook
 * §7.20/§7.26): the constraint uid and the reporter uid are *different registries* —
 * the constraint's 668 is a `constraint_uids` entry that the frontend's
 * `uipc::constitution::AffineBodyRevoluteJointExternalForce::apply_to` writes on the
 * joint geometry, while the reporter's 668 is the `ExternalForceReporter` registry uid
 * of the `GlobalExternalForceManager` chain.  Reproducing the same number twice is what
 * the reference does, and the two lookups never collide; the unit test pins both.
 *
 * WHAT THE PAIR IS.  The constraint reads the two per-edge attributes
 * `external_torque/is_constrained` (IndexT) and `external_torque` (Float) from the
 * *revolute joint's* geometry and mirrors them into its own device buffers; it
 * contributes no energy/gradient/Hessian to the animator (its `do_report_extent`
 * reports zeros and both compute hooks are empty — the reference says so explicitly).
 * The reporter derives the actual force from them: for every constrained pair, the
 * joint's axle direction is taken from the *edge's own two local endpoints*
 * (`e_bar_i = l_pos1 - l_pos0`, `e_bar_j = r_pos1 - r_pos0` — the same `rest_positions`
 * rows the joint energy uses), rotated into the world by each body
 * (`e = A x_bar`, then `.normalized()`), symmetrised (`e_j = (e_i + e_j)/2`,
 * `e_i = -e_j` — the reference's own "avoid the near-singularity" step, so the two
 * bodies get exactly anti-parallel unit axes), and the scalar torque is applied
 * through the shared `torque_to_F` (`affine_body/utils.h`, the *same* helper the
 * reference's `affine_body/utils.cu` provides) as `F_i = torque_to_F(tau, e_i, q_i)`,
 * `F_j = torque_to_F(tau, e_j, q_j)`, *atomically* (several joints may address the
 * same body).
 *
 * DEVIATION (port necessity, the T5.4b/T6.3b2 idiom): the reference's
 * `eigen::atomic_add` on the AoS `Vector12` element becomes the *scalar alias* of the
 * same buffer (`BufferView<T>::as<double>()`, stride-checked in `do_step`) plus twelve
 * per-component atomic adds — see `affine_body_revolute_joint_external_force.cpp`.
 *********************************************************************/
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/affine_body_external_force_reporter.h>
#include <affine_body/constitutions/affine_body_revolute_joint.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
class AffineBodyRevoluteJointExternalForceConstraint final : public InterAffineBodyConstraint
{
  public:
    using InterAffineBodyConstraint::InterAffineBodyConstraint;
    static constexpr U64 ConstraintUID = 668;

    SimSystemSlot<AffineBodyRevoluteJoint> revolute_joint;

    vector<Float>  h_torques;
    vector<IndexT> h_is_constrained;

    lud::LDeviceBuffer<Float>  torques;
    lud::LDeviceBuffer<IndexT> is_constrained;

    void do_build(BuildInfo& info) override;

    U64 get_uid() const noexcept override;

    void do_init(InterAffineBodyAnimator::FilteredInfo& info) override;
    void do_step(InterAffineBodyAnimator::FilteredInfo& info) override;
    void do_report_extent(InterAffineBodyAnimator::ReportExtentInfo& info) override;
    void do_compute_energy(InterAffineBodyAnimator::ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(
        InterAffineBodyAnimator::GradientHessianInfo& info) override;
};

/**
 * @brief Applies external torques about revolute joint axles.
 *
 * An `AffineBodyExternalForceReporter`: the ABD external-force manager owns its
 * lifecycle and calls `do_step(ExternalForceInfo&)`, in which the force is added to
 * the dynamics' per-body `external_force` window (`info.external_forces()`), which is
 * the *same* buffer the FEM/ABD reporters write and the manager's own `step()`
 * afterwards turns into an acceleration.
 */
class AffineBodyRevoluteJointExternalForce final : public AffineBodyExternalForceReporter
{
  public:
    static constexpr U64 ReporterUID = 668;
    using AffineBodyExternalForceReporter::AffineBodyExternalForceReporter;

    SimSystemSlot<AffineBodyRevoluteJointExternalForceConstraint> constraint;

    void do_build(BuildInfo& info) override;

    U64 get_uid() const noexcept override;

    void do_init() override;
    void do_step(ExternalForceInfo& info) override;
};
}  // namespace uipc::backend::luisa
