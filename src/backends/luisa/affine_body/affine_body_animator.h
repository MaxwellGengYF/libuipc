#pragma once
/********************************************************************
 * @file affine_body/affine_body_animator.h
 * @brief SCOPE NOTE (T6.1b) — the ABD animator's *declaration surface*
 * (port of src/backends/cuda/affine_body/affine_body_animator.h).
 *
 * WHY THIS FILE EXISTS IN T6.1a
 * -----------------------------
 * `AffineBodyConstraint` (in scope here) declares hooks whose parameters are
 * `AffineBodyAnimator::{FilteredInfo, ComputeEnergyInfo, ComputeGradientHessianInfo,
 * ReportExtentInfo}` and calls `AffineBodyAnimator::add_constraint` in its
 * `do_build()`, so the animator class must be complete and that one member must
 * link (the T5.3 §7.18 rule 3 dependency-cycle policy; the FEM analogue is
 * finite_element_animator.h, ported by T5.3 because it owned it there).
 *
 * WHAT IS PORTED HERE
 *  * the class declaration (an `Animator`), `Impl`'s member list, the nested
 *    `FilteredInfo`/`BaseInfo`/`ComputeEnergyInfo`/`ComputeGradientHessianInfo`/
 *    `ReportExtentInfo` classes with the reference's access surface, the `friend`
 *    set and the private hook declarations;
 *  * `add_constraint()` (affine_body_animator.cpp) — the one member
 *    `AffineBodyConstraint::do_build()` needs, so the base links;
 *  * `BaseInfo::dt()` — the reference *declares* it and never defines it (the same
 *    finding T5.3 recorded for `FiniteElementAnimator::BaseInfo::dt()`); the port
 *    defines it as `return m_dt;`, the only sensible reading of the member it
 *    stores, so T6.4's constraints cannot hit a link error.
 *
 * WHAT IS **NOT** DEFINED HERE (T6.1b's, none of it stubbed):
 *  * `Impl::{init, step}`, `do_init()`, `do_step()`, `do_build(BuildInfo&)`,
 *    `compute_energy(...)`, `compute_gradient_hessian(...)`,
 *    `BaseInfo::{substep_ratio, qs, q_prevs, body_masses, is_fixed}`,
 *    `ComputeEnergyInfo::energies()`,
 *    `ComputeGradientHessianInfo::{gradients, hessians, gradient_only}`,
 *    `ReportExtentInfo::{hessian_count, gradient_count, energy_count, check}`
 *    (`check()` is the `check_report_extent` pattern — never weakened),
 *    `FilteredInfo::{anim_geo_infos, for_each}` (the two templates are declared
 *    here, defined in details/affine_body_animator.inl by T6.1b),
 *  * `REGISTER_SIM_SYSTEM(AffineBodyAnimator)` and the two sub-reporters the
 *    reference defines at the bottom of its .cu
 *    (`AffineBodyAnimatorLinearSubsystemReporter` needs T6.1b's
 *    abd_linear_subsystem_reporter.h; `AffineBodyAnimatorLineSearchSubreporter`
 *    needs T6.4's abd_line_search_subreporter.h) plus the inter-affine-body layer.
 *
 * Consequence, stated for the parent: until T6.1b, no AffineBodyAnimator exists in
 * any scene, so `AffineBodyConstraint::do_build()`'s `require<AffineBodyAnimator>()`
 * throws `SimSystemException` and invalidates the constraint.  That is the correct
 * "not implemented yet" signal (T5.1 rule 13) -- not a silent no-op.
 *********************************************************************/
#include <animator/animator.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/abd_line_search_reporter.h>
#include <line_search/line_searcher.h>

namespace uipc::backend::luisa
{
class AffineBodyConstraint;  // T6.4
class ABDLineSearchReporter;  // T6.4
class ABDGradientHessianComputer;  // not ported (unused in the reference's .cu)

class AffineBodyAnimator final : public Animator
{
  public:
    using Animator::Animator;

    class Impl;

    using AnimatedGeoInfo = AffineBodyDynamics::GeoInfo;

    class FilteredInfo
    {
      public:
        FilteredInfo(Impl* impl, SizeT index)
            : m_impl(impl)
            , m_index(index)
        {
        }

        span<const AnimatedGeoInfo> anim_geo_infos() const noexcept;

        template <typename ViewGetterF, typename ForEachF>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ViewGetterF&&                   getter,
                      ForEachF&&                      for_each);

        template <typename ForEachGeometry>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ForEachGeometry&&               for_every_geometry);

      private:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl, SizeT index, Float dt)
            : m_impl(impl)
            , m_index(index)
            , m_dt(dt)
        {
        }

        Float                            substep_ratio() const noexcept;
        Float                            dt() const noexcept;
        lud::LCBufferView<LVector12>     qs() const noexcept;
        lud::LCBufferView<LVector12>     q_prevs() const noexcept;
        lud::LCBufferView<ABDJacobiDyadicMass> body_masses() const noexcept;
        lud::LCBufferView<IndexT>        is_fixed() const noexcept;

      protected:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
        Float m_dt    = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Impl*                     impl,
                          SizeT                     index,
                          Float                     dt,
                          lud::LBufferView<Float>   energy)
            : BaseInfo(impl, index, dt)
            , m_energies(energy)
        {
        }
        lud::LBufferView<Float> energies() const noexcept;

      private:
        friend class AffineBodyAnimator;
        lud::LBufferView<Float> m_energies;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(Impl*                                  impl,
                                   SizeT                                  index,
                                   Float                                  dt,
                                   lla::LDoubletVectorView<Float, 12>     gradients,
                                   lla::LTripletMatrixView<Float, 12, 12> hessians,
                                   bool                                   gradient_only)
            : BaseInfo(impl, index, dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
        {
        }
        lla::LDoubletVectorView<Float, 12>     gradients() const noexcept;
        lla::LTripletMatrixView<Float, 12, 12> hessians() const noexcept;
        bool                                   gradient_only() const noexcept;

      private:
        friend class AffineBodyAnimator;
        lla::LDoubletVectorView<Float, 12>     m_gradients;
        lla::LTripletMatrixView<Float, 12, 12> m_hessians;
        bool                                   m_gradient_only = false;
    };

    class ReportExtentInfo
    {
      public:
        void hessian_count(SizeT count) noexcept;
        void gradient_count(SizeT count) noexcept;
        void energy_count(SizeT count) noexcept;
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class AffineBodyAnimator;
        friend class AffineBodyConstraint;
        SizeT        m_hessian_block_count    = 0;
        SizeT        m_gradient_segment_count = 0;
        SizeT        m_energy_count           = 0;
        bool         m_gradient_only          = false;
        mutable bool m_gradient_only_checked  = false;
    };

    class Impl
    {
      public:
        void init(backend::WorldVisitor& world);
        void step();

        S<const geometry::AttributeSlot<Float>> dt_attr;

        AffineBodyDynamics* affine_body_dynamics = nullptr;
        GlobalAnimator*     global_animator      = nullptr;
        SimSystemSlotCollection<AffineBodyConstraint> constraints;
        unordered_map<U64, SizeT>                     uid_to_constraint_index;

        vector<AnimatedGeoInfo>       anim_geo_infos;
        OffsetCountCollection<IndexT> constraint_geo_info_offsets_counts;

        OffsetCountCollection<IndexT> constraint_energy_offsets_counts;
        OffsetCountCollection<IndexT> constraint_gradient_offsets_counts;
        OffsetCountCollection<IndexT> constraint_hessian_offsets_counts;
    };

  private:
    friend class AffineBodyConstraint;
    void add_constraint(AffineBodyConstraint* constraint);  // only by AffineBodyConstraint

    friend class AffineBodyAnimatorLineSearchSubreporter;  // T6.4
    void compute_energy(ABDLineSearchReporter::ComputeEnergyInfo& info);

    friend class AffineBodyAnimatorLinearSubsystemReporter;  // T6.1b
    void compute_gradient_hessian(ABDLinearSubsystem::AssembleInfo& info);

    Impl m_impl;

    virtual void do_init() override;
    virtual void do_step() override;
    virtual void do_build(BuildInfo& info) override;
};
}  // namespace uipc::backend::luisa
