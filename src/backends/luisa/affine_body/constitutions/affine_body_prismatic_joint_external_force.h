#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_prismatic_joint_external_force.h
 * @brief T6.3b2 — the prismatic joint's external-force pair: the constraint
 *        (`ConstraintUID = 667`) and the reporter (`ReporterUID = 667`).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu
 * (classes `AffineBodyPrismaticJointExternalForceConstraint` and
 * `AffineBodyPrismaticJointExternalForce`).
 *
 * THE 667 SHARING IS DELIBERATE (the T5.4b `671` precedent, cookbook §7.20): the
 * constraint uid and the reporter uid are *different registries* — the constraint's
 * 667 is a `constraint_uids` entry that the frontend's
 * `uipc::constitution::AffineBodyPrismaticJointExternalForce::apply_to` writes on the
 * joint geometry, while the reporter's 667 is the `ExternalForceReporter` registry
 * uid of the `GlobalExternalForceManager` chain.  Reproducing the same number twice
 * is what the reference does, and the two lookups never collide; the unit test pins
 * both (the constraint's `is_constrained` host record *and* the reporter's uid).
 *
 * WHAT THE PAIR IS.  The constraint reads the two per-edge attributes
 * `external_force/is_constrained` (IndexT) and `external_force` (Float) from the
 * *prismatic joint's* geometry and mirrors them into its own device buffers; it
 * contributes no energy/gradient/Hessian to the animator (its
 * `do_report_extent` reports zeros and both compute hooks are empty — the reference
 * says so explicitly).  The reporter derives the actual force from them: for every
 * constrained pair, the joint's *world* axis directions `t_i = J(t_i_bar) q_i` and
 * `t_j` are replaced by their symmetrised pair (`t_j = (t_i + t_j)/2`,
 * `t_i = -t_j` — the reference's own "avoid the near-singularity" step), and the
 * per-edge scalar force `f` is applied as `-f t` on body i and `+f t` on body j,
 * *atomically* (several joints may address the same body).
 *
 * DEVIATION (port necessity, the T5.4b idiom): the reference's `eigen::atomic_add`
 * on the AoS `Vector12` element becomes the *scalar alias* of the same buffer
 * (`BufferView<T>::as<double>()`, stride-checked in `do_step`) plus twelve
 * per-component atomic adds — see `affine_body_prismatic_joint_external_force.cpp`.
 *********************************************************************/
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/affine_body_external_force_reporter.h>
#include <affine_body/constitutions/affine_body_prismatic_joint.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
class AffineBodyPrismaticJointExternalForceConstraint final : public InterAffineBodyConstraint
{
  public:
    using InterAffineBodyConstraint::InterAffineBodyConstraint;
    static constexpr U64 ConstraintUID = 667;

    SimSystemSlot<AffineBodyPrismaticJoint> prismatic_joint;

    vector<Float>  h_forces;
    vector<IndexT> h_is_constrained;

    lud::LDeviceBuffer<Float>  forces;
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
 * @brief Applies external forces along prismatic joint axes.
 *
 * An `AffineBodyExternalForceReporter`: the ABD external-force manager owns its
 * lifecycle and calls `do_step(ExternalForceInfo&)`, in which the force is added to
 * the dynamics' per-body `external_force` window (`info.external_forces()`), which is
 * the *same* buffer the FEM/ABD reporters write and the manager's own `step()`
 * afterwards turns into an acceleration.
 */
class AffineBodyPrismaticJointExternalForce final : public AffineBodyExternalForceReporter
{
  public:
    static constexpr U64 ReporterUID = 667;
    using AffineBodyExternalForceReporter::AffineBodyExternalForceReporter;

    SimSystemSlot<AffineBodyPrismaticJointExternalForceConstraint> constraint;

    void do_build(BuildInfo& info) override;

    U64 get_uid() const noexcept override;

    void do_init() override;
    void do_step(ExternalForceInfo& info) override;
};
}  // namespace uipc::backend::luisa
