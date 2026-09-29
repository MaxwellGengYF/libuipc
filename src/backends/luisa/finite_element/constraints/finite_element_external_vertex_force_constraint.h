#pragma once
/********************************************************************
 * @file finite_element_external_vertex_force_constraint.h
 * @brief T5.4b — the per-vertex external-force constraint (port of
 *        src/backends/cuda/finite_element/constraints/finite_element_external_vertex_force_constraint.{h,cu}).
 *
 * Reads the per-vertex `external_force` (Vector3) geometry attribute of every FEM
 * geometry whose `builtin::constraint_uids` contains this constraint's uid, keeps
 * the marked (`builtin::is_constrained`) entries in its own host staging vectors
 * and mirrors them into its own device buffers.  It contributes **no**
 * energy/gradient/Hessian — the reference's own comment says why: "forces are
 * applied through x_tilde in the time integrator".
 *
 * `UID = 671` is the reference's number, and it is the *same* number the reference
 * uses for `FiniteElementExternalVertexForce` (the reporter) and the frontend
 * `uipc::constitution::FiniteElementExternalForce` (the constitution).  These are
 * three *different* registries (constraint uids, external-force-reporter uids and
 * constitution uids) so the collision is by design, not a bug — cookbook §7.20.
 *
 * DEVIATIONS (reported, both test seams):
 *  1. the reference declares the class *and* its kernels inside the `.cu`; here the
 *     class + `Impl` declaration live in this header (the kernels stay in the
 *     `.cpp`), exactly like T5.2's constitutions (cookbook §7.17 rule 11) and
 *     T5.3's `SoftPositionConstraint`;
 *  2. the hooks and `m_impl` are public so the unit test can drive `do_report_extent`
 *     and can write a *synthetic* (vertex_id, force) table with duplicate vertex
 *     ids into the constraint's device buffers (the duplicate case is what makes
 *     the reporter's vector atomic observable — see `finite_element_external_vertex_force.h`).
 *********************************************************************/
#include <finite_element/finite_element_constraint.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
/**
 * @brief Constraint that reads per-vertex external forces from geometry attributes
 *
 * This constraint reads "external_force" (Vector3) attributes from vertices
 * and stores them inside its local buffers. No energy/gradient/hessian --
 * forces are applied through x_tilde in the time integrator.
 */
class FiniteElementExternalVertexForceConstraint final : public FiniteElementConstraint
{
  public:
    using FiniteElementConstraint::FiniteElementConstraint;

    static constexpr U64 UID = 671;

    class Impl
    {
      public:
        vector<Vector3>              h_forces;
        vector<IndexT>               h_vertex_ids;
        lud::LDeviceBuffer<LVector3> forces;
        lud::LDeviceBuffer<IndexT>   vertex_ids;

        void step(backend::WorldVisitor& world, FiniteElementAnimator::FilteredInfo& info);
    };

    lud::LCBufferView<LVector3> forces() const noexcept;
    lud::LCBufferView<IndexT>   vertex_ids() const noexcept;

    /// The reference's five hooks (private there; public here for the unit test
    /// — deviation 2 above).
    void do_build(BuildInfo& info) override;
    U64  get_uid() const noexcept override;
    void do_init(FiniteElementAnimator::FilteredInfo& info) override;
    void do_step(FiniteElementAnimator::FilteredInfo& info) override;
    void do_report_extent(FiniteElementAnimator::ReportExtentInfo& info) override;
    void do_compute_energy(FiniteElementAnimator::ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(FiniteElementAnimator::ComputeGradientHessianInfo& info) override;

    /// public test seam: the unit test writes synthetic (vertex_id, force) tables
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
