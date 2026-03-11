#pragma once
#include <algorithm/matrix_converter.h>
#include <linear_system/diag_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <utils/offset_count_collection.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
class FiniteElementKinetic;
class FEMDyTopoEffectReceiver;
class FEMLinearSubsystemReporter;

/**
 * @brief Linear subsystem for Finite Element Method (FEM)
 * 
 * Manages the linear system assembly and solution for FEM simulations.
 * Integrates kinetic energy, constitution terms, and dynamic topology effects.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class FEMLinearSubsystem final : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    // Forward declarations for sparse matrix/vector types compatible with LuisaCompute
    template<typename T, int N>
    struct DoubletVectorView
    {
        luisa::compute::BufferView<const luisa::uint> indices;
        luisa::compute::BufferView<const luisa::Vector<T, N>> values;
        SizeT count;
    };

    template<typename T, int N>
    struct MutableDoubletVectorView
    {
        luisa::compute::BufferView<luisa::uint> indices;
        luisa::compute::BufferView<luisa::Vector<T, N>> values;
        SizeT count;
    };

    template<typename T, int M, int N>
    struct TripletMatrixView
    {
        luisa::compute::BufferView<const luisa::uint> row_indices;
        luisa::compute::BufferView<const luisa::uint> col_indices;
        luisa::compute::BufferView<const luisa::Matrix<T, N, M>> values;
        SizeT count;
    };

    template<typename T, int M, int N>
    struct MutableTripletMatrixView
    {
        luisa::compute::BufferView<luisa::uint> row_indices;
        luisa::compute::BufferView<luisa::uint> col_indices;
        luisa::compute::BufferView<luisa::Matrix<T, N, M>> values;
        SizeT count;
    };

    // Type aliases for FEM 3D (Vector3)
    using DoubletVector3 = DoubletVectorView<Float, 3>;
    using MutableDoubletVector3 = MutableDoubletVectorView<Float, 3>;
    using TripletMatrix3x3 = TripletMatrixView<Float, 3, 3>;
    using MutableTripletMatrix3x3 = MutableTripletMatrixView<Float, 3, 3>;

    /**
     * @brief Information for computing gradient and Hessian
     */
    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(bool                     gradient_only,
                                   MutableDoubletVector3    gradients,
                                   MutableTripletMatrix3x3  hessians,
                                   Float                    dt) noexcept
            : m_gradient_only(gradient_only)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_dt(dt)
        {
        }

        auto gradient_only() const noexcept { return m_gradient_only; }
        auto gradients() const noexcept { return m_gradients; }
        auto hessians() const noexcept { return m_hessians; }
        auto dt() const noexcept { return m_dt; }

      private:
        bool                    m_gradient_only = false;
        MutableDoubletVector3   m_gradients;
        MutableTripletMatrix3x3 m_hessians;
        Float                   m_dt = 0.0;
    };

    /**
     * @brief Information for reporting extent of gradient and Hessian
     */
    class ReportExtentInfo
    {
      public:
        // DoubletVector3 count
        void gradient_count(SizeT size);
        // TripletMatrix3x3 count
        void hessian_count(SizeT size);
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class FEMLinearSubsystem;
        friend class FEMLinearSubsystemReporter;
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count  = 0;
        bool  m_gradient_only  = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl;

    /**
     * @brief Information for assembling the linear system
     */
    class AssembleInfo
    {
      public:
        AssembleInfo(Impl*                   impl,
                     IndexT                  index,
                     bool                    gradient_only) noexcept;
        MutableDoubletVector3   gradients() const;
        MutableTripletMatrix3x3 hessians() const;
        Float                   dt() const noexcept;
        bool                    gradient_only() const noexcept;

      private:
        friend class FEMLinearSubsystem;

        Impl*                   m_impl          = nullptr;
        IndexT                  m_index         = ~0;
        bool                    m_gradient_only = false;
    };

    /**
     * @brief Implementation details (PIMPL pattern)
     */
    class Impl
    {
      public:
        void init();
        void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
        void receive_init_dof_info(WorldVisitor& w, GlobalLinearSystem::InitDofInfo& info);

        void report_extent(GlobalLinearSystem::DiagExtentInfo& info);

        void assemble(GlobalLinearSystem::DiagInfo& info);
        void _assemble_kinetic(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_reporters(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_dytopo_effect(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);

        void accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
        void retrieve_solution(GlobalLinearSystem::SolutionInfo& info);

        SimEngine* sim_engine = nullptr;

        SimSystemSlot<FiniteElementMethod> finite_element_method;
        FiniteElementMethod::Impl&         fem() noexcept
        {
            return finite_element_method->m_impl;
        }

        SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;

        SimSystemSlot<FEMDyTopoEffectReceiver> dytopo_effect_receiver;
        SimSystemSlotCollection<FEMLinearSubsystemReporter> reporters;

        SimSystemSlot<FiniteElementKinetic> kinetic;

        Float dt            = 0.0;
        Float reserve_ratio = 1.5;

        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        // LuisaCompute buffers for sparse data
        // DeviceTripletMatrix equivalent using LuisaCompute buffers
        luisa::compute::Buffer<luisa::uint> reporter_hessian_row_indices;
        luisa::compute::Buffer<luisa::uint> reporter_hessian_col_indices;
        luisa::compute::Buffer<Matrix3x3>   reporter_hessian_values;
        SizeT                               reporter_hessian_count = 0;

        // DeviceDoubletVector equivalent using LuisaCompute buffers  
        luisa::compute::Buffer<luisa::uint> reporter_gradient_indices;
        luisa::compute::Buffer<Vector3>     reporter_gradient_values;
        SizeT                               reporter_gradient_count = 0;

        // intermediate gradient/hessian buffers for kinetic
        luisa::compute::Buffer<Matrix3x3>   body_id_to_kinetic_hessian;
        luisa::compute::Buffer<Vector3>     body_id_to_kinetic_gradient;

        // diag hessian for preconditioner
        luisa::compute::Buffer<Matrix3x3>   diag_hessian;

        void loose_resize_entries(luisa::compute::Buffer<luisa::uint>& indices,
                                  luisa::compute::Buffer<Vector3>&     values,
                                  SizeT                                size);
    };

  private:
    virtual void do_build(DiagLinearSubsystem::BuildInfo& info) override;
    virtual void do_init(DiagLinearSubsystem::InitInfo& info) override;
    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) override;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info) override;
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) override;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) override;
    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) override;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) override;
    virtual U64 get_uid() const noexcept override;

    friend class FEMLinearSubsystemReporter;
    void add_reporter(FEMLinearSubsystemReporter* reporter);

    friend class FiniteElementKinetic;
    void add_kinetic(FiniteElementKinetic* kinetic);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
