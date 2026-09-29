#pragma once
/********************************************************************
 * @file affine_body/inter_affine_body_animator.h
 * @brief T6.1b — the inter-affine-body animator (port of
 *        src/backends/cuda/affine_body/inter_affine_body_animator.h).
 *
 * The second half of the ABD animation pipeline: where `AffineBodyAnimator` fans
 * the single-body constraints out, this one fans the *inter-body* constraints
 * (`InterAffineBodyConstraint`: joints, articulations) out, over the geometries
 * the `InterAffineBodyConstitutionManager` collected.  It keeps the same
 * uid-sorted constraint -> geometry-range offset tables and the same
 * energy/gradient/Hessian offset tables, so the `InterAffineBodyConstraint` hooks
 * index their own slices exactly like `AffineBodyConstraint` does.
 *
 * `FilteredInfo` exposes the manager's own `FilteredInfo` surface (geo_infos,
 * geo_info(geo_id), body_id(...), body_geo(...), for_each) plus the animator's
 * per-constraint `anim_inter_geo_infos()` slice.
 *
 * The two sub-reporters of the reference's .cu live at the bottom of the .cpp
 * (linear-subsystem half: this batch; line-search half: invalidated until T6.4
 * registers `ABDLineSearchReporter`).
 *********************************************************************/
#include <animator/animator.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/inter_affine_body_constitution_manager.h>

namespace uipc::backend::luisa
{
class InterAffineBodyConstraint;  // T6.4 (the joints are T6.3b's constitutions)
class ABDLineSearchReporter;      // T6.4
class ABDGradientHessianComputer;  // not ported (unused in the reference's .cu)

class InterAffineBodyAnimator final : public Animator
{
  public:
    using Animator::Animator;

    class Impl;

    using AnimatedInterGeoInfo = InterAffineBodyConstitutionManager::InterGeoInfo;
    using ForEachInfo          = InterAffineBodyConstitutionManager::ForEachInfo;
    using InterGeoInfo         = InterAffineBodyConstitutionManager::InterGeoInfo;

    class FilteredInfo
    {
      public:
        FilteredInfo(Impl* impl, SizeT index)
            : m_impl(impl)
            , m_index(index)
        {
        }

        span<const AnimatedInterGeoInfo> anim_inter_geo_infos() const noexcept;
        span<const AffineBodyDynamics::GeoInfo> geo_infos() const noexcept;
        const AffineBodyDynamics::GeoInfo& geo_info(IndexT geo_id) const noexcept;
        IndexT body_id(IndexT geo_id) const noexcept;
        IndexT body_id(IndexT geo_id, IndexT instance_id) const noexcept;
        geometry::SimplicialComplex* body_geo(span<S<geometry::GeometrySlot>> geo_slots,
                                              IndexT geo_id) const noexcept;

        template <typename ForEachGeometry>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots, ForEachGeometry&& for_every_geometry)
        {
            InterAffineBodyAnimator::_for_each(geo_slots,
                                               this->anim_inter_geo_infos(),
                                               std::forward<ForEachGeometry>(for_every_geometry));
        }

      private:
        Impl*  m_impl  = nullptr;
        SizeT  m_index = ~0ull;
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

        Float                                 substep_ratio() const noexcept;
        Float                                 dt() const noexcept;
        lud::LCBufferView<LVector12>          qs() const noexcept;
        lud::LCBufferView<LVector12>          q_prevs() const noexcept;
        lud::LCBufferView<ABDJacobiDyadicMass> body_masses() const noexcept;
        lud::LCBufferView<IndexT>             is_fixed() const noexcept;

      protected:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
        Float m_dt    = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Impl* impl, SizeT index, Float dt, lud::LBufferView<Float> energy)
            : BaseInfo(impl, index, dt)
            , m_energies(energy)
        {
        }
        lud::LBufferView<Float> energies() const noexcept;

      private:
        friend class InterAffineBodyAnimator;
        lud::LBufferView<Float> m_energies;
    };

    class GradientHessianInfo : public BaseInfo
    {
      public:
        GradientHessianInfo(Impl*                                  impl,
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
        friend class InterAffineBodyAnimator;
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

        /// DEVIATION (T6.1b): read-only accessors for the three counts (the
        /// reference has none), so a test can observe a constraint's reported
        /// extent -- the T5.4b precedent for the FEM animator's `ReportExtentInfo`.
        [[nodiscard]] SizeT energy_count() const noexcept { return m_energy_count; }
        [[nodiscard]] SizeT gradient_count() const noexcept { return m_gradient_segment_count; }
        [[nodiscard]] SizeT hessian_count() const noexcept { return m_hessian_block_count; }

      private:
        friend class InterAffineBodyAnimator;
        friend class InterAffineBodyConstraint;
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

        AffineBodyDynamics*                 affine_body_dynamics = nullptr;
        InterAffineBodyConstitutionManager* manager              = nullptr;
        GlobalAnimator*                     global_animator      = nullptr;

        SimSystemSlotCollection<InterAffineBodyConstraint> constraints;
        unordered_map<U64, SizeT>                          uid_to_constraint_index;

        vector<AnimatedInterGeoInfo>  anim_geo_infos;
        OffsetCountCollection<IndexT> constraint_geo_info_offsets_counts;

        // Constraints

        OffsetCountCollection<IndexT> constraint_energy_offsets_counts;
        OffsetCountCollection<IndexT> constraint_gradient_offsets_counts;
        OffsetCountCollection<IndexT> constraint_hessian_offsets_counts;
    };

    template <typename ForEachGeometry>
    static void _for_each(span<S<geometry::GeometrySlot>> geo_slots,
                          span<const InterGeoInfo>        geo_infos,
                          ForEachGeometry&&               for_every_geometry)
    {
        InterAffineBodyConstitutionManager::_for_each(geo_slots, geo_infos, for_every_geometry);
    }

  private:
    friend class InterAffineBodyConstraint;
    void add_constraint(InterAffineBodyConstraint* constraint);  // only by InterAffineBodyConstraint

    friend class InterAffineBodyAnimatorLineSearchSubreporter;
    void compute_energy(ABDLineSearchReporter::ComputeEnergyInfo& info);

    friend class InterAffineBodyAnimatorLinearSubsystemReporter;
    void compute_gradient_hessian(ABDLinearSubsystem::AssembleInfo& info);

    Impl m_impl;

    virtual void do_init() override;
    virtual void do_step() override;
    virtual void do_build(BuildInfo& info) override;
};
}  // namespace uipc::backend::luisa
