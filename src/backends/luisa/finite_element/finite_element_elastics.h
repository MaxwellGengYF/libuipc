#pragma once
#include <sim_system.h>
#include <luisa/runtime/buffer.h>
#include <finite_element/finite_element_method.h>
#include <utils/offset_count_collection.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/fem_line_search_reporter.h>

namespace uipc::backend::luisa
{
class FiniteElementMethod;

/**
 * @brief Doublet vector view for sparse gradient storage (LuisaCompute version)
 * 
 * Represents a sparse vector where each entry is (index, value) pair.
 * Used for storing gradients in FEM computations.
 */
template<typename T, int N>
class DoubletVectorView
{
public:
    using IndexType = luisa::compute::BufferView<int>;  // indices
    using ValueType = luisa::compute::BufferView<luisa::Vector<T, N>>;  // values
    
    DoubletVectorView() = default;
    DoubletVectorView(IndexType indices, ValueType values, size_t count)
        : m_indices(indices)
        , m_values(values)
        , m_count(count)
    {}
    
    [[nodiscard]] auto indices() const noexcept { return m_indices; }
    [[nodiscard]] auto values() const noexcept { return m_values; }
    [[nodiscard]] size_t count() const noexcept { return m_count; }
    
    [[nodiscard]] DoubletVectorView subview(size_t offset, size_t subcount) const noexcept
    {
        return DoubletVectorView{
            m_indices.subview(offset, subcount),
            m_values.subview(offset, subcount),
            subcount};
    }
    
private:
    IndexType m_indices;
    ValueType m_values;
    size_t m_count = 0;
};

/**
 * @brief Triplet matrix view for sparse Hessian storage (LuisaCompute version)
 * 
 * Represents a sparse matrix where each entry is (row, col, value) triplet.
 * Used for storing Hessians in FEM computations.
 */
template<typename T, int M, int N>
class TripletMatrixView
{
public:
    using RowType = luisa::compute::BufferView<int>;     // row indices
    using ColType = luisa::compute::BufferView<int>;     // col indices  
    using ValueType = luisa::compute::BufferView<luisa::Matrix<T, M, N>>;  // values
    
    TripletMatrixView() = default;
    TripletMatrixView(RowType row_indices, ColType col_indices, ValueType values, size_t count)
        : m_row_indices(row_indices)
        , m_col_indices(col_indices)
        , m_values(values)
        , m_count(count)
    {}
    
    [[nodiscard]] auto row_indices() const noexcept { return m_row_indices; }
    [[nodiscard]] auto col_indices() const noexcept { return m_col_indices; }
    [[nodiscard]] auto values() const noexcept { return m_values; }
    [[nodiscard]] size_t count() const noexcept { return m_count; }
    
    [[nodiscard]] TripletMatrixView subview(size_t offset, size_t subcount) const noexcept
    {
        return TripletMatrixView{
            m_row_indices.subview(offset, subcount),
            m_col_indices.subview(offset, subcount),
            m_values.subview(offset, subcount),
            subcount};
    }
    
private:
    RowType m_row_indices;
    ColType m_col_indices;
    ValueType m_values;
    size_t m_count = 0;
};

class FiniteElementElastics final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class ReportExtentInfo
    {
      public:
        void energy_count(SizeT count) noexcept { m_energy_count = count; }
        void gradient_count(SizeT count) noexcept { m_gradient_count = count; }
        void hessian_count(SizeT count) noexcept { m_hessian_count = count; }
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class FiniteElementElastics;
        friend class FiniteElementConstitution;
        friend class FiniteElementExtraConstitution;

        SizeT m_energy_count   = 0;
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count  = 0;
        bool  m_gradient_only  = false;
        mutable bool m_gradient_only_checked = false;
    };

    class ComputeEnergyInfo
    {
      public:
        ComputeEnergyInfo(Impl* impl, SizeT index, Float dt, luisa::compute::BufferView<Float> energies)
            : m_impl(impl)
            , m_index(index)
            , m_dt(dt)
            , m_energies(energies)
        {
        }

        auto energies() const noexcept
        {
            auto [offset, count] = m_impl->constitution_energy_offsets_counts[m_index];
            return m_energies.subview(offset, count);
        }
        auto dt() const noexcept { return m_dt; }

      private:
        Impl*                                 m_impl  = nullptr;
        SizeT                                 m_index = 0;
        Float                                 m_dt    = 0.0;
        luisa::compute::BufferView<Float> m_energies;
    };

    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(Impl* impl,
                                   SizeT index,
                                   bool  gradient_only,
                                   Float dt,
                                   DoubletVectorView<Float, 3> gradients,
                                   TripletMatrixView<Float, 3, 3> hessians)
            : m_impl(impl)
            , m_index(index)
            , m_gradient_only(gradient_only)
            , m_dt(dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
        {
        }

        auto gradient_only() const noexcept { return m_gradient_only; }
        DoubletVectorView<Float, 3> gradients() const noexcept;
        TripletMatrixView<Float, 3, 3> hessians() const noexcept;

        auto dt() const noexcept { return m_dt; }

      private:
        Impl*                          m_impl          = nullptr;
        SizeT                          m_index         = 0;
        bool                           m_gradient_only = false;
        Float                          m_dt            = 0.0;
        DoubletVectorView<Float, 3>    m_gradients;
        TripletMatrixView<Float, 3, 3> m_hessians;
    };

    class Impl
    {
      public:
        SimSystemSlot<FiniteElementMethod> finite_element_method;
        FiniteElementMethod::Impl&         fem()
        {
            return finite_element_method->m_impl;
        }

        OffsetCountCollection<IndexT> constitution_energy_offsets_counts;
        OffsetCountCollection<IndexT> constitution_gradient_offsets_counts;
        OffsetCountCollection<IndexT> constitution_hessian_offsets_counts;

        void assemble(FEMLinearSubsystem::AssembleInfo& info);
        void compute_energy(FEMLineSearchReporter::ComputeEnergyInfo& info);
    };

  private:
    friend class FiniteElementElasticsLinearSubsystemReporter;
    friend class FiniteElementElasticsLineSearchSubreporter;

    virtual void do_build() override;

    friend class FiniteElementMethod;
    void init();  // only be called in FiniteElementMethod

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
