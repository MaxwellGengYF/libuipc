#pragma once
#include <animator/animator.h>
#include <finite_element/fem_linear_subsystem_reporter.h>
#include <finite_element/fem_line_search_subreporter.h>
#include <utils/offset_count_collection.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
class FiniteElementConstraint;
class FiniteElementAnimator final : public Animator
{
  public:
    using Animator::Animator;

    // Forward declaration - GeoInfo is defined in FiniteElementMethod
    // We use a placeholder that will be resolved when FiniteElementMethod is included
    struct AnimatedGeoInfo
    {
        IndexT geo_slot_index = -1;
        struct DimUID
        {
            IndexT dim = -1;
            U64    uid = ~0ull;
        } dim_uid;
        SizeT vertex_offset    = ~0ull;
        SizeT vertex_count     = 0ull;
        SizeT primitive_offset = ~0ull;
        SizeT primitive_count  = 0ull;
    };

    class Impl;

    class FilteredInfo
    {
      public:
        FilteredInfo(Impl* impl, SizeT index)
            : m_impl(impl)
            , m_index(index)
        {
        }

        span<const AnimatedGeoInfo> anim_geo_infos() const;

        template <typename ForEach, typename ViewGetter>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ViewGetter&&                    view_getter,
                      ForEach&&                       for_each_action) noexcept;

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
        Float                              substep_ratio() const noexcept;
        Float                              dt() const noexcept;
        luisa::compute::BufferView<Vector3> xs() const noexcept;
        luisa::compute::BufferView<Vector3> x_prevs() const noexcept;
        luisa::compute::BufferView<Float>   masses() const noexcept;
        luisa::compute::BufferView<IndexT>  is_fixed() const noexcept;

      protected:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
        Float m_dt    = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Impl* impl, SizeT index, Float dt, luisa::compute::BufferView<Float> energies)
            : BaseInfo(impl, index, dt)
            , m_energies(energies)
        {
        }

        luisa::compute::BufferView<Float> energies() const noexcept;

      private:
        luisa::compute::BufferView<Float> m_energies;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        using DoubletVector3    = FEMLinearSubsystem::MutableDoubletVector3;
        using TripletMatrix3x3  = FEMLinearSubsystem::MutableTripletMatrix3x3;

        ComputeGradientHessianInfo(Impl*              impl,
                                   SizeT              index,
                                   Float              dt,
                                   DoubletVector3     gradients,
                                   TripletMatrix3x3   hessians,
                                   bool               gradient_only)
            : BaseInfo(impl, index, dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
        {
        }
        DoubletVector3   gradients() const noexcept;
        TripletMatrix3x3 hessians() const noexcept;
        bool             gradient_only() const noexcept;

      private:
        DoubletVector3   m_gradients;
        TripletMatrix3x3 m_hessians;
        bool             m_gradient_only = false;
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
        friend class FiniteElementAnimator;
        friend class FiniteElementConstraint;
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

        // Forward declaration - FiniteElementMethod will be defined in the implementation
        struct FEMImpl;
        FEMImpl* finite_element_method = nullptr;
        
        // Accessor for FEM implementation
        FEMImpl& fem() const noexcept
        {
            return *finite_element_method;
        }

        struct GlobalAnimatorImpl;
        GlobalAnimatorImpl* global_animator = nullptr;
        
        SimSystemSlotCollection<FiniteElementConstraint> constraints;
        unordered_map<U64, SizeT> uid_to_constraint_index;

        vector<AnimatedGeoInfo>       anim_geo_infos;
        OffsetCountCollection<IndexT> constraint_geo_info_offsets_counts;

        OffsetCountCollection<IndexT> constraint_energy_offsets_counts;
        OffsetCountCollection<IndexT> constraint_gradient_offsets_counts;
        OffsetCountCollection<IndexT> constraint_hessian_offsets_counts;
    };

  private:
    friend class FiniteElementConstraint;
    void add_constraint(FiniteElementConstraint* constraint);

    friend class FiniteElementAnimatorLineSearchSubreporter;
    void compute_energy(FEMLineSearchSubreporter::ComputeEnergyInfo& info);

    friend class FiniteElementAnimatorLinearSubsystemReporter;
    void assemble(FEMLinearSubsystemReporter::AssembleInfo& info);

    Impl m_impl;

    virtual void do_init() override;
    virtual void do_step() override;
    virtual void do_build(BuildInfo& info) override;
};
}  // namespace uipc::backend::luisa

#include "details/finite_element_animator.inl"
