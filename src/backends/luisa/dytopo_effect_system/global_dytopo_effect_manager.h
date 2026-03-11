#pragma once
#include <sim_system.h>
#include <energy_component_flags.h>
#include <global_geometry/global_vertex_manager.h>
#include <dytopo_effect_system/dytopo_classify_info.h>
#include <luisa/runtime/buffer.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
class DyTopoEffectReporter;
class DyTopoEffectReceiver;

// Forward declarations for sparse matrix/vector types compatible with LuisaCompute
// DoubletVector: sparse vector with (index, value) pairs, N is the dimension of the value vector
/**
 * @brief View of a sparse vector in doublet format (index, value pairs)
 * 
 * Equivalent to muda::DoubletVectorView<T, N>
 */
template<typename T, int N>
struct DoubletVectorView
{
    luisa::compute::BufferView<const luisa::uint> indices;
    luisa::compute::BufferView<const luisa::Vector<T, N>> values;
    SizeT count = 0;
    SizeT capacity = 0;
};

/**
 * @brief Mutable view of a sparse vector in doublet format
 */
template<typename T, int N>
struct MutableDoubletVectorView
{
    luisa::compute::BufferView<luisa::uint> indices;
    luisa::compute::BufferView<luisa::Vector<T, N>> values;
    SizeT count = 0;
    SizeT capacity = 0;
};

/**
 * @brief View of a sparse matrix in triplet format (row, col, value)
 * 
 * Equivalent to muda::TripletMatrixView<T, N>
 * For block matrices, N is the dimension of the block (e.g., 3 for 3x3 blocks)
 */
template<typename T, int N>
struct TripletMatrixView
{
    luisa::compute::BufferView<const luisa::uint> row_indices;
    luisa::compute::BufferView<const luisa::uint> col_indices;
    luisa::compute::BufferView<const luisa::Matrix<T, N, N>> values;
    SizeT count = 0;
    SizeT capacity = 0;
};

/**
 * @brief Mutable view of a sparse matrix in triplet format
 */
template<typename T, int N>
struct MutableTripletMatrixView
{
    luisa::compute::BufferView<luisa::uint> row_indices;
    luisa::compute::BufferView<luisa::uint> col_indices;
    luisa::compute::BufferView<luisa::Matrix<T, N, N>> values;
    SizeT count = 0;
    SizeT capacity = 0;
};

// Type aliases for 3D (common case)
using DoubletVector3 = DoubletVectorView<Float, 3>;
using MutableDoubletVector3 = MutableDoubletVectorView<Float, 3>;
using TripletMatrix3 = TripletMatrixView<Float, 3>;
using MutableTripletMatrix3 = MutableTripletMatrixView<Float, 3>;

/**
 * @brief Block COO matrix view
 * 
 * Equivalent to muda::BCOOMatrixView
 */
template<typename T, int N>
struct BCOOMatrixView
{
    luisa::compute::BufferView<const luisa::uint> block_row_indices;
    luisa::compute::BufferView<const luisa::uint> block_col_indices;
    luisa::compute::BufferView<const luisa::Matrix<T, N, N>> block_values;
    SizeT block_count = 0;
};

/**
 * @brief Const block COO matrix view
 */
template<typename T, int N>
struct CBCOOMatrixView
{
    luisa::compute::BufferView<const luisa::uint> block_row_indices;
    luisa::compute::BufferView<const luisa::uint> block_col_indices;
    luisa::compute::BufferView<const luisa::Matrix<T, N, N>> block_values;
    SizeT block_count = 0;
};

/**
 * @brief Block COO vector view
 */
template<typename T, int N>
struct BCOOVectorView
{
    luisa::compute::BufferView<const luisa::uint> block_indices;
    luisa::compute::BufferView<const luisa::Vector<T, N>> block_values;
    SizeT block_count = 0;
};

/**
 * @brief Const block COO vector view
 */
template<typename T, int N>
struct CBCOOVectorView
{
    luisa::compute::BufferView<const luisa::uint> block_indices;
    luisa::compute::BufferView<const luisa::Vector<T, N>> block_values;
    SizeT block_count = 0;
};

// Type aliases for 3D
using BCOOMatrix3 = BCOOMatrixView<Float, 3>;
using CBCOOMatrix3 = CBCOOMatrixView<Float, 3>;
using BCOOVector3 = BCOOVectorView<Float, 3>;
using CBCOOVector3 = CBCOOVectorView<Float, 3>;

/**
 * @brief Global manager for dynamic topology effects
 * 
 * Coordinates reporters (sources) and receivers (destinations) of dynamic topology effects.
 * Handles collection, classification, and distribution of gradient and Hessian contributions.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class GlobalDyTopoEffectManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class GradientHessianExtentInfo
    {
      public:
        bool gradient_only() const { return m_gradient_only; }
        void gradient_count(SizeT count) noexcept { m_gradient_count = count; }
        void hessian_count(SizeT count) noexcept { m_hessian_count = count; }

      private:
        friend class Impl;
        friend class DyTopoEffectReporter;

        bool  m_gradient_only  = false;
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count  = 0;
    };

    class GradientHessianInfo
    {
      public:
        bool gradient_only() const noexcept { return m_gradient_only; }
        MutableDoubletVector3 gradients() const noexcept
        {
            return m_gradients;
        }
        MutableTripletMatrix3 hessians() const noexcept
        {
            return m_hessians;
        }


      private:
        friend class Impl;
        bool                  m_gradient_only = false;
        MutableDoubletVector3 m_gradients;
        MutableTripletMatrix3 m_hessians;
    };

    class EnergyExtentInfo
    {
      public:
        void energy_count(SizeT count) noexcept { m_energy_count = count; }

      private:
        friend class Impl;
        friend class DyTopoEffectLineSearchReporter;
        SizeT m_energy_count = 0;
    };

    class EnergyInfo
    {
      public:
        luisa::compute::BufferView<Float> energies() const { return m_energies; }
        bool                              is_initial() const { return m_is_initial; }

      private:
        friend class DyTopoEffectLineSearchReporter;
        luisa::compute::BufferView<Float> m_energies;
        bool                              m_is_initial = false;
    };

    using ClassifyInfo = DyTopoClassifyInfo;

    class ClassifiedDyTopoEffectInfo
    {
      public:
        DoubletVector3 gradients() const noexcept
        {
            return m_gradients;
        }
        TripletMatrix3 hessians() const noexcept
        {
            return m_hessians;
        }

      private:
        friend class Impl;
        DoubletVector3 m_gradients;
        TripletMatrix3 m_hessians;
    };

    class ComputeDyTopoEffectInfo
    {
      public:
        void gradient_only(bool v) noexcept { m_gradient_only = v; }
        void component_flags(EnergyComponentFlags v) noexcept
        {
            m_component_flags = v;
        }

      private:
        friend class Impl;
        bool                 m_gradient_only   = false;
        EnergyComponentFlags m_component_flags = EnergyComponentFlags::All;
    };

    /**
     * @brief Container for triplet format sparse matrix data (device buffers)
     * 
     * Equivalent to muda::DeviceTripletMatrix<Float, 3>
     */
    struct DeviceTripletMatrix3
    {
        luisa::compute::Buffer<luisa::uint> row_indices;
        luisa::compute::Buffer<luisa::uint> col_indices;
        luisa::compute::Buffer<Matrix3x3> values;
        SizeT count = 0;
        SizeT capacity = 0;

        MutableTripletMatrix3 view() noexcept
        {
            return MutableTripletMatrix3{
                row_indices.view(),
                col_indices.view(),
                values.view(),
                count,
                capacity};
        }

        TripletMatrix3 view() const noexcept
        {
            return TripletMatrix3{
                row_indices.view(),
                col_indices.view(),
                values.view(),
                count,
                capacity};
        }

        void clear() noexcept { count = 0; }
    };

    /**
     * @brief Container for doublet format sparse vector data (device buffers)
     * 
     * Equivalent to muda::DeviceDoubletVector<Float, 3>
     */
    struct DeviceDoubletVector3
    {
        luisa::compute::Buffer<luisa::uint> indices;
        luisa::compute::Buffer<Vector3> values;
        SizeT count = 0;
        SizeT capacity = 0;

        MutableDoubletVector3 view() noexcept
        {
            return MutableDoubletVector3{
                indices.view(),
                values.view(),
                count,
                capacity};
        }

        DoubletVector3 view() const noexcept
        {
            return DoubletVector3{
                indices.view(),
                values.view(),
                count,
                capacity};
        }

        void clear() noexcept { count = 0; }
    };

    /**
     * @brief Container for block COO format sparse matrix data (device buffers)
     * 
     * Equivalent to muda::DeviceBCOOMatrix<Float, 3>
     */
    struct DeviceBCOOMatrix3
    {
        luisa::compute::Buffer<luisa::uint> block_row_indices;
        luisa::compute::Buffer<luisa::uint> block_col_indices;
        luisa::compute::Buffer<Matrix3x3> block_values;
        SizeT block_count = 0;
        SizeT block_capacity = 0;

        BCOOMatrix3 view() noexcept
        {
            return BCOOMatrix3{
                block_row_indices.view(),
                block_col_indices.view(),
                block_values.view(),
                block_count};
        }

        CBCOOMatrix3 view() const noexcept
        {
            return CBCOOMatrix3{
                block_row_indices.view(),
                block_col_indices.view(),
                block_values.view(),
                block_count};
        }

        void clear() noexcept { block_count = 0; }
    };

    /**
     * @brief Container for block COO format sparse vector data (device buffers)
     * 
     * Equivalent to muda::DeviceBCOOVector<Float, 3>
     */
    struct DeviceBCOOVector3
    {
        luisa::compute::Buffer<luisa::uint> block_indices;
        luisa::compute::Buffer<Vector3> block_values;
        SizeT block_count = 0;
        SizeT block_capacity = 0;

        BCOOVector3 view() noexcept
        {
            return BCOOVector3{
                block_indices.view(),
                block_values.view(),
                block_count};
        }

        CBCOOVector3 view() const noexcept
        {
            return CBCOOVector3{
                block_indices.view(),
                block_values.view(),
                block_count};
        }

        void clear() noexcept { block_count = 0; }
    };

    class Impl
    {
      public:
        void init(WorldVisitor& world);
        void compute_dytopo_effect(ComputeDyTopoEffectInfo& info);
        void _assemble(ComputeDyTopoEffectInfo& info);
        void _convert_matrix();
        void _distribute(ComputeDyTopoEffectInfo& info);

        SimSystemSlot<GlobalVertexManager> global_vertex_manager;

        Float reserve_ratio = 1.1;


        /***********************************************************************
        *                              Reporter                                *
        ***********************************************************************/

        SimSystemSlotCollection<DyTopoEffectReporter> dytopo_effect_reporters;
        SimSystemSlotCollection<DyTopoEffectReporter> contact_reporters;
        SimSystemSlotCollection<DyTopoEffectReporter> non_contact_reporters;

        OffsetCountCollection<IndexT> reporter_energy_offsets_counts;
        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        DeviceTripletMatrix3 collected_dytopo_effect_hessian;
        DeviceDoubletVector3 collected_dytopo_effect_gradient;

        // MatrixConverter<Float, 3>        matrix_converter;
        DeviceBCOOMatrix3 sorted_dytopo_effect_hessian;
        DeviceBCOOVector3 sorted_dytopo_effect_gradient;

        /***********************************************************************
        *                               Receiver                               *
        ***********************************************************************/

        SimSystemSlotCollection<DyTopoEffectReceiver> dytopo_effect_receivers;

        luisa::compute::Buffer<Vector2i> gradient_range;
        luisa::compute::Buffer<IndexT> selected_hessian;
        luisa::compute::Buffer<IndexT> selected_hessian_offsets;

        luisa::vector<DeviceTripletMatrix3> classified_dytopo_effect_hessians;
        luisa::vector<DeviceDoubletVector3> classified_dytopo_effect_gradients;

        void loose_resize_entries(DeviceTripletMatrix3& m, SizeT size);
        void loose_resize_entries(DeviceDoubletVector3& v, SizeT size);
        
        template <typename T>
        void loose_resize(luisa::compute::Buffer<T>& buffer, SizeT size)
        {
            if(size > buffer.size())
            {
                // LuisaCompute Buffer doesn't have reserve, so we recreate
                // The actual recreation happens in the .cpp file where device is available
                // Here we just mark that resize is needed
            }
        }
    };

    CBCOOVector3 gradients() const noexcept;
    CBCOOMatrix3 hessians() const noexcept;

    void compute_dytopo_effect(ComputeDyTopoEffectInfo& info);

  protected:
    virtual void do_build() override;

  private:
    friend class DyTopoEffectLineSearchReporter;
    void init();

    friend class SimEngine;
    // only be called by SimEngine
    void compute_dytopo_effect();

    friend class DyTopoEffectReporter;
    void add_reporter(DyTopoEffectReporter* reporter);
    friend class DyTopoEffectReceiver;
    void add_receiver(DyTopoEffectReceiver* receiver);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
