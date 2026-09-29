#pragma once
/********************************************************************
 * @file affine_body/inter_affine_body_constraint.h
 * @brief T6.1b — the inter-affine-body constraint *interface* (port of
 *        src/backends/cuda/affine_body/inter_affine_body_constraint.{h,cu}).
 *
 * The inter-body sibling of `AffineBodyConstraint`: it checks the scene's
 * constraint-uid table, resolves the `InterAffineBodyAnimator`, runs its own
 * `do_build` hook and registers itself with the animator.  The concrete
 * constraints (the joints and the articulation constraint) are T6.4's.
 *
 * The reference `.cu` is ported body-for-body: the uid check + registration in
 * `do_build()`, `uid()`, the five hook forwards, and `report_extent`'s
 * `check()` + gradient-only assertion (issue #295).
 *********************************************************************/
#include <sim_system.h>
#include <affine_body/inter_affine_body_animator.h>

namespace uipc::backend::luisa
{
class InterAffineBodyConstraint : public SimSystem
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
    virtual U64  get_uid() const noexcept  = 0;
    virtual void do_init(InterAffineBodyAnimator::FilteredInfo& info)                            = 0;
    virtual void do_step(InterAffineBodyAnimator::FilteredInfo& info)                            = 0;
    virtual void do_report_extent(InterAffineBodyAnimator::ReportExtentInfo& info)               = 0;
    virtual void do_compute_energy(InterAffineBodyAnimator::ComputeEnergyInfo& info)             = 0;
    virtual void do_compute_gradient_hessian(InterAffineBodyAnimator::GradientHessianInfo& info) = 0;

    template <typename ForEachGeometry>
    void for_each(span<S<geometry::GeometrySlot>> geo_slots, ForEachGeometry&& for_every_geometry)
    {
        InterAffineBodyAnimator::_for_each(geo_slots,
                                           animated_inter_geo_info(),
                                           std::forward<ForEachGeometry>(for_every_geometry));
    }

    span<const InterAffineBodyAnimator::AnimatedInterGeoInfo> animated_inter_geo_info() const noexcept;

  private:
    friend class InterAffineBodyAnimator;
    virtual void do_build() override final;

    void init(InterAffineBodyAnimator::FilteredInfo& info);
    void step(InterAffineBodyAnimator::FilteredInfo& info);
    void report_extent(InterAffineBodyAnimator::ReportExtentInfo& info);
    void compute_energy(InterAffineBodyAnimator::ComputeEnergyInfo& info);
    void compute_gradient_hessian(InterAffineBodyAnimator::GradientHessianInfo& info);

    SizeT                                  m_index = ~0ull;
    SimSystemSlot<InterAffineBodyAnimator> m_animator;
};
}  // namespace uipc::backend::luisa
