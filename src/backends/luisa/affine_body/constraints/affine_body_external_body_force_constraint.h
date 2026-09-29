#pragma once
/********************************************************************
 * @file affine_body/constraints/affine_body_external_body_force_constraint.h
 * @brief T6.4b — the ABD per-**body** external-force constraint (uid 666), the ABD
 *        sibling of the ported FEM `FiniteElementExternalVertexForceConstraint`
 *        (uid 671).
 *
 * REFERENCE: src/backends/cuda/affine_body/constraints/
 * affine_body_external_body_force_constraint.{h,cu}.
 *
 * WHAT THIS SYSTEM IS.  An `AffineBodyConstraint` that reads the per-instance
 * `external_force` (**Vector12**) attribute of every ABD geometry whose
 * `constraint_uids` contains 666, keeps the marked (`builtin::is_constrained`)
 * entries as (global body id, 12-dof generalized force) pairs in its own host
 * staging vectors and mirrors them into its own device buffers.  It contributes
 * **no** energy/gradient/Hessian — the reference's own comment says why: "forces are
 * applied directly in do_predict_dof", i.e. through the paired reporter
 * `AffineBodyExternalBodyForce` (uid 666 too, a *different* registry) which
 * scatter-adds them into the dynamics' `body_id_to_external_force`, from where the
 * external-force manager turns them into `a_ext = M^-1 F`.
 *
 * `UID = 666` is the reference's number, and it is the *same* number the reference
 * uses for the reporter and for the frontend `uipc::constitution::AffineBodyExternalForce`
 * (the constitution).  These are three *different* registries (constraint uids, the
 * external-force reporter uids, constitution uids) so the collision is by design —
 * the T5.4b 671 / T6.3b2 667 / T6.3b3 668 precedent.
 *
 * DEVIATIONS (reported, both test seams):
 *  1. the reference declares the class *and* its kernels inside the `.cu`; here the
 *     class + `Impl` declaration live in this header (the kernels stay in the `.cpp`),
 *     exactly like T5.2's constitutions (cookbook §7.17 rule 11) and T5.4b's pair;
 *  2. the hooks and `m_impl` are public so the unit test can drive `do_report_extent`
 *     and can write a *synthetic* (body_id, force) table with duplicate body ids into
 *     the constraint's device buffers (the duplicate case is what makes the reporter's
 *     vector atomic observable — see `affine_body_external_body_force.h`).
 *********************************************************************/
#include <affine_body/affine_body_constraint.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
/**
 * @brief Constraint that reads external forces from geometry attributes
 *
 * This constraint reads "external_force" attributes from geometries
 * and stores them inside its local buffers.
 */
class AffineBodyExternalBodyForceConstraint final : public AffineBodyConstraint
{
  public:
    using AffineBodyConstraint::AffineBodyConstraint;

    static constexpr U64 UID = 666;  // Same UID as AffineBodyExternalForce

    class Impl
    {
      public:
        // Host-side buffers for collecting data before copying to device
        vector<Vector12>              h_forces;
        vector<IndexT>                h_body_ids;
        lud::LDeviceBuffer<LVector12> forces;
        lud::LDeviceBuffer<IndexT>    body_ids;
        AffineBodyDynamics*           affine_body_dynamics = nullptr;

        void step(backend::WorldVisitor& world, AffineBodyAnimator::FilteredInfo& info);
    };

    lud::LCBufferView<LVector12> forces() const noexcept;
    lud::LCBufferView<IndexT>    body_ids() const noexcept;

    /// The reference's five hooks (private there; public here for the unit test --
    /// deviation 2 above).
    void do_build(BuildInfo& info) override;
    U64  get_uid() const noexcept override;
    void do_init(AffineBodyAnimator::FilteredInfo& info) override;
    void do_step(AffineBodyAnimator::FilteredInfo& info) override;
    void do_report_extent(AffineBodyAnimator::ReportExtentInfo& info) override;
    void do_compute_energy(AffineBodyAnimator::ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(
        AffineBodyAnimator::ComputeGradientHessianInfo& info) override;

    /// public test seam: the unit test writes synthetic (body_id, force) tables
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
