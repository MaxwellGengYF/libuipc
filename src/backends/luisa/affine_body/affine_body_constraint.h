#pragma once
/********************************************************************
 * @file affine_body/affine_body_constraint.h
 * @brief T6.1a — the ABD constraint *interface* (port of
 * src/backends/cuda/affine_body/affine_body_constraint.{h,cu}).
 *
 * Mirrors T5.3's finite_element_constraint.{h,cpp}: `do_build()` checks the scene's
 * constraint-uid table (the frontend's `constraint_uids` meta attribute), resolves
 * the `AffineBodyAnimator` and registers itself with it.  The concrete constraints
 * (T6.4: soft_transform_constraint, external_articulation_constraint, ...) are
 * T6.4's.
 *
 * `report_extent()` keeps the reference's `info.check(name())` and the
 * `UIPC_ASSERT(!(gradient_only && hessian_block_count != 0), ...)` verbatim
 * (cookbook §7.16 rule 7: a backend UIPC_ASSERT aborts, so a test cannot exercise
 * the invalid combination -- the T6.1a test pins the legal one only).
 *
 * Port mapping: `cuda_tool::DoubletVectorView<Float,12>` ->
 * `lla::LDoubletVectorView<Float,12>`, `TripletMatrixView<Float,12>` ->
 * `lla::LTripletMatrixView<Float,12,12>`.
 *********************************************************************/
#include <sim_system.h>
#include <affine_body/affine_body_animator.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
class AffineBodyConstraint : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    U64 uid() const noexcept;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual U64  get_uid() const noexcept = 0;
    virtual void do_init(AffineBodyAnimator::FilteredInfo& info) = 0;
    virtual void do_step(AffineBodyAnimator::FilteredInfo& info) = 0;
    virtual void do_report_extent(AffineBodyAnimator::ReportExtentInfo& info) = 0;
    virtual void do_compute_energy(AffineBodyAnimator::ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(
        AffineBodyAnimator::ComputeGradientHessianInfo& info) = 0;

  private:
    friend class AffineBodyAnimator;
    virtual void do_build() override final;

    void init(AffineBodyAnimator::FilteredInfo& info);
    void step(AffineBodyAnimator::FilteredInfo& info);
    void report_extent(AffineBodyAnimator::ReportExtentInfo& info);
    void compute_energy(AffineBodyAnimator::ComputeEnergyInfo& info);
    void compute_gradient_hessian(AffineBodyAnimator::ComputeGradientHessianInfo& info);

    SizeT m_index = ~0ull;
};
}  // namespace uipc::backend::luisa
