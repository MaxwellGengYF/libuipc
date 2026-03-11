#pragma once
#include <animator/animator.h>
#include <affine_body/affine_body_dynamics.h>
#include <line_search/line_searcher.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
class InterAffineBodyConstraint;
class ABDLineSearchReporter;
class ABDGradientHessianComputer;

class InterAffineBodyAnimator final : public Animator
{
  public:
    using Animator::Animator;

    class Impl;

    using AnimatedInterGeoInfo = InterAffineBodyConstitutionManager::InterGeoInfo;
    using ForEachInfo  = InterAffineBodyConstitutionManager::ForEachInfo;
    using InterGeoInfo = InterAffineBodyConstitutionManager::InterGeoInfo;

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

        Float                                           substep_ratio() const noexcept;
        Float                                           dt() const noexcept;
        luisa::compute::BufferView<Vector12>            qs() const noexcept;
        luisa::compute::BufferView<Vector12>            q_prevs() const noexcept;
        luisa::compute::BufferView<ABDJacobiDyadicMass> body_masses() const noexcept;
        luisa::compute::BufferView<IndexT>              is_fixed() const noexcept;

      protected:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
        Float m_dt    = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Impl* impl, SizeT index, Float dt, luisa::compute::BufferView<Float> energy)
            : BaseInfo(impl, index, dt)
            , m_energies(energy)
        {
        }
        luisa::compute::BufferView<Float> energies() const noexcept;

      private:
        friend class InterAffineBodyAnimator;
        luisa::compute::BufferView<Float> m_energies;
    };

    /**
     * @brief Gradient and Hessian computation info for inter-affine body constraints.
     * 
     * In luisa-compute backend, we use raw buffers for sparse linear algebra data.
     * - gradients: Buffer of Vector12 representing gradient segments
     * - hessians: Buffer of Matrix12x12 representing Hessian blocks
     * - gradient_indices: Buffer of indices for gradient segments (for sparse storage)
     * - hessian_row_indices/hessian_col_indices: Buffers for Hessian block coordinates
     */
    class GradientHessianInfo : public BaseInfo
    {
      public:
        GradientHessianInfo(Impl*                                  impl,
                            SizeT                                  index,
                            Float                                  dt,
                            luisa::compute::BufferView<Vector12>   gradients,
                            luisa::compute::BufferView<IndexT>     gradient_indices,
                            luisa::compute::BufferView<Matrix12x12> hessians,
                            luisa::compute::BufferView<IndexT>     hessian_row_indices,
                            luisa::compute::BufferView<IndexT>     hessian_col_indices,
                            bool                                   gradient_only)
            : BaseInfo(impl, index, dt)
            , m_gradients(gradients)
            , m_gradient_indices(gradient_indices)
            , m_hessians(hessians)
            , m_hessian_row_indices(hessian_row_indices)
            , m_hessian_col_indices(hessian_col_indices)
            , m_gradient_only(gradient_only)
        {
        }
        
        luisa::compute::BufferView<Vector12>    gradients() const noexcept;
        luisa::compute::BufferView<IndexT>      gradient_indices() const noexcept;
        luisa::compute::BufferView<Matrix12x12> hessians() const noexcept;
        luisa::compute::BufferView<IndexT>      hessian_row_indices() const noexcept;
        luisa::compute::BufferView<IndexT>      hessian_col_indices() const noexcept;
        bool                                    gradient_only() const noexcept;

      private:
        friend class InterAffineBodyAnimator;
        luisa::compute::BufferView<Vector12>    m_gradients;
        luisa::compute::BufferView<IndexT>      m_gradient_indices;
        luisa::compute::BufferView<Matrix12x12> m_hessians;
        luisa::compute::BufferView<IndexT>      m_hessian_row_indices;
        luisa::compute::BufferView<IndexT>      m_hessian_col_indices;
        bool                                    m_gradient_only = false;
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
        friend class InterAffineBodyAnimator;
        friend class InterAffineBodyConstraint;
        SizeT m_hessian_block_count    = 0;
        SizeT m_gradient_segment_count = 0;
        SizeT m_energy_count           = 0;
        bool  m_gradient_only          = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl
    {
      public:
        void init(backend::WorldVisitor& world);
        void step();

        Float dt = 0.0;

        AffineBodyDynamics*                 affine_body_dynamics = nullptr;
        InterAffineBodyConstitutionManager* manager              = nullptr;
        GlobalAnimator*                     global_animator      = nullptr;

        SimSystemSlotCollection<InterAffineBodyConstraint> constraints;
        unordered_map<U64, SizeT> uid_to_constraint_index;

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
    void add_constraint(InterAffineBodyConstraint* constraint);  // only be called by AffineBodyConstraint

    friend class InterAffineBodyAnimatorLineSearchSubreporter;
    void compute_energy(ABDLineSearchReporter::ComputeEnergyInfo& info);  // only be called by AffineBodyAnimatorLineSearchSubreporter

    friend class InterAffineBodyAnimatorLinearSubsystemReporter;
    void compute_gradient_hessian(ABDLinearSubsystem::AssembleInfo& info);  // only be called by AffineBodyAnimatorLinearSubsystemReporter

    Impl m_impl;

    virtual void do_init() override;
    virtual void do_step() override;
    virtual void do_build(BuildInfo& info) override;
};
}  // namespace uipc::backend::luisa
