#pragma once
#include <sim_system.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <utils/offset_count_collection.h>
#include <algorithm/matrix_converter.h>
#include <uipc/diff_sim/sparse_coo_view.h>

namespace uipc::backend::luisa
{
class DiffDofReporter;
class DiffParmReporter;
class GlobalLinearSystem;

/**
 * @brief Differentiable Simulation Subsystem for LuisaCompute backend
 * 
 * This class manages the differentiable simulation components including
 * parameter and DOF (Degrees of Freedom) reporting for gradient computation.
 * 
 * Refactored from CUDA backend to use LuisaCompute runtime APIs:
 * - luisa::compute::Buffer<T> instead of muda::DeviceBuffer<T>
 * - luisa::compute::Stream instead of CUDA streams
 * - Custom sparse matrix structures compatible with LuisaCompute
 */
class GlobalDiffSimManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    /**
     * @brief Sparse COO matrix structure on host
     */
    class SparseCOO
    {
      public:
        luisa::vector<IndexT> row_indices;
        luisa::vector<IndexT> col_indices;
        luisa::vector<Float>  values;
        luisa::Vector2i       shape;

        diff_sim::SparseCOOView view() const;
    };

    /**
     * @brief Implementation details (PIMPL idiom)
     */
    class Impl
    {
      public:
        using BufferFloat = luisa::compute::Buffer<Float>;
        using BufferIndexT = luisa::compute::Buffer<IndexT>;
        
        void init(WorldVisitor& world);
        void update();
        void assemble();
        void write_scene(WorldVisitor& world);

        GlobalLinearSystem* global_linear_system = nullptr;
        SimEngine*          sim_engine           = nullptr;

        SimSystemSlotCollection<DiffDofReporter>  diff_dof_reporters;
        SimSystemSlotCollection<DiffParmReporter> diff_parm_reporters;

        OffsetCountCollection<IndexT> diff_dof_triplet_offset_count;
        OffsetCountCollection<IndexT> diff_parm_triplet_offset_count;

        luisa::vector<IndexT> dof_offsets;
        luisa::vector<IndexT> dof_counts;

        SizeT total_parm_count = 0;

        // Device buffer for parameters using LuisaCompute
        luisa::compute::Buffer<Float> parameters;

        /**
         * @brief Triplet matrix structure for sparse matrix assembly
         * 
         * Replaces muda::DeviceTripletMatrix with LuisaCompute-compatible structure
         */
        struct TripletMatrix
        {
            luisa::compute::Buffer<IndexT> row_indices;
            luisa::compute::Buffer<IndexT> col_indices;
            luisa::compute::Buffer<Float>  values;
            SizeT                          triplet_count = 0;
            SizeT                          rows = 0;
            SizeT                          cols = 0;

            void reshape(SizeT m, SizeT n)
            {
                rows = m;
                cols = n;
            }

            void resize_triplets(SizeT count, luisa::compute::Device& device)
            {
                triplet_count = count;
                if(count > 0)
                {
                    row_indices = device.create_buffer<IndexT>(count);
                    col_indices = device.create_buffer<IndexT>(count);
                    values = device.create_buffer<Float>(count);
                }
            }

            auto view() const
            {
                return TripletMatrixView{row_indices.view(), col_indices.view(), values.view(), triplet_count};
            }

            auto viewer() const
            {
                return view();
            }

            auto cviewer() const
            {
                return view();
            }
        };

        /**
         * @brief View into a triplet matrix
         */
        struct TripletMatrixView
        {
            luisa::compute::BufferView<IndexT> row_indices;
            luisa::compute::BufferView<IndexT> col_indices;
            luisa::compute::BufferView<Float>  values;
            SizeT                              triplet_count;

            TripletMatrixView subview(SizeT offset, SizeT count) const
            {
                return TripletMatrixView{
                    row_indices.subview(offset, count),
                    col_indices.subview(offset, count),
                    values.subview(offset, count),
                    count
                };
            }
        };

        /**
         * @brief COO matrix structure for sparse matrix storage
         * 
         * Replaces muda::DeviceCOOMatrix with LuisaCompute-compatible structure
         */
        struct COOMatrix
        {
            luisa::compute::Buffer<IndexT> row_indices;
            luisa::compute::Buffer<IndexT> col_indices;
            luisa::compute::Buffer<Float>  values;
            SizeT                          non_zeros_count = 0;
            SizeT                          rows = 0;
            SizeT                          cols = 0;

            void reshape(SizeT m, SizeT n)
            {
                rows = m;
                cols = n;
            }

            void resize_non_zeros(SizeT count, luisa::compute::Device& device)
            {
                non_zeros_count = count;
                if(count > 0)
                {
                    row_indices = device.create_buffer<IndexT>(count);
                    col_indices = device.create_buffer<IndexT>(count);
                    values = device.create_buffer<Float>(count);
                }
            }

            SizeT non_zeros() const { return non_zeros_count; }

            auto row_indices_view() const { return row_indices.view(); }
            auto col_indices_view() const { return col_indices.view(); }
            auto values_view() const { return values.view(); }

            auto cviewer() const
            {
                return COOMatrixViewer{row_indices.view(), col_indices.view(), values.view(), non_zeros_count};
            }
        };

        struct COOMatrixViewer
        {
            luisa::compute::BufferView<IndexT> row_indices;
            luisa::compute::BufferView<IndexT> col_indices;
            luisa::compute::BufferView<Float>  values;
            SizeT                              non_zeros;

            __device__ auto operator()(SizeT i) const
            {
                return luisa::make_tuple(row_indices[i], col_indices[i], values[i]);
            }
        };

        // NOTE:
        // local_triplet_pGpP only consider the triplet at current frame.
        // So the triplet count = current_frame_triplet_count
        // The shape of the matrix = (total_frame_dof_count, total_parm_count)
        //tex:
        //$$
        //T = \begin{bmatrix}
        // 0   \\
        // ... \\
        // 0   \\
        // T^{[i]} \\
        //\end{bmatrix}
        //$$
        TripletMatrix local_triplet_pGpP;
        //tex:
        //$$
        //T = \begin{bmatrix}
        // T^{[1]}   \\
        // ... \\
        // T^{[i]} \\
        //\end{bmatrix}
        //$$
        TripletMatrix total_triplet_pGpP;
        COOMatrix     total_coo_pGpP;

        // NOTE:
        // local_triplet_H only consider the triplet at current frame.
        // So the triplet count = current_frame_triplet_count
        // The shape of the matrix = (total_frame_dof_count, total_frame_dof_count)
        //tex:
        //$$
        //T = \begin{bmatrix}
        // 0   \\
        // ... \\
        // 0   \\
        // H^{[i]} \\
        //\end{bmatrix}
        //$$
        TripletMatrix local_triplet_H;
        //tex:
        //$$
        //T = \begin{bmatrix}
        // H^{[1]}   \\
        // ... \\
        // H^{[i]} \\
        //\end{bmatrix}
        //$$
        TripletMatrix total_triplet_H;
        COOMatrix     total_coo_H;

        SizeT current_frame_dof_count = 0;
        SizeT total_frame_dof_count   = 0;

        SparseCOO host_coo_pGpP;
        SparseCOO host_coo_H;

        // Stream for async operations
        luisa::compute::Stream compute_stream;
    };

    /**
     * @brief Base info class for DOF/Parameter info access
     */
    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl, SizeT index)
            : m_impl(impl)
            , m_index(index)
        {
        }

        SizeT  frame() const;
        IndexT dof_offset(SizeT frame) const;
        IndexT dof_count(SizeT frame) const;

      protected:
        friend class Impl;
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
    };

    /**
     * @brief Info for reporting parameter extent
     */
    class DiffParmExtentInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        void triplet_count(SizeT N) { m_triplet_count = N; }

      private:
        friend class Impl;
        SizeT m_triplet_count = 0;
    };

    /**
     * @brief Info for accessing parameter gradient information
     */
    class DiffParmInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        Impl::TripletMatrixView pGpP() const;
    };

    /**
     * @brief Info for reporting DOF extent
     */
    class DiffDofExtentInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        void triplet_count(SizeT N) { m_triplet_count = N; }

      private:
        friend class Impl;
        SizeT m_triplet_count = 0;
    };

    /**
     * @brief Info for accessing DOF Hessian information
     */
    class DiffDofInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        friend class Impl;
        Impl::TripletMatrixView H() const;
    };

    /**
     * @brief Info for parameter updates
     */
    class DiffParmUpdateInfo
    {
      public:
        DiffParmUpdateInfo(Impl* impl)
            : m_impl(impl)
        {
        }

        luisa::compute::BufferView<Float> parameters() const noexcept;

      private:
        friend class Impl;
        Impl* m_impl = nullptr;
    };

  private:
    friend class SimEngine;
    void init();      // only be called by SimEngine
    void assemble();  // only be called by SimEngine
    void update();    // only be called by SimEngine

    virtual void do_build() override;

    friend class DiffDofReporter;
    friend class DiffParmReporter;

    void add_reporter(DiffDofReporter* subsystem);   // only be called by DiffDofReporter
    void add_reporter(DiffParmReporter* subsystem);  // only be called by DiffParmReporter

    Impl m_impl;
};

/**
 * @brief Base class for DOF reporters in differentiable simulation
 */
class DiffDofReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  protected:
    class BuildInfo
    {
      public:
    };

    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info) = 0;
    virtual void do_assemble(GlobalDiffSimManager::DiffDofInfo& info) = 0;

  private:
    friend class GlobalDiffSimManager;
    virtual void do_build() override final;
    void         report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info);
    void         assemble_diff_dof(GlobalDiffSimManager::DiffDofInfo& info);
    SizeT        m_index = ~0ull;
};

/**
 * @brief Base class for parameter reporters in differentiable simulation
 */
class DiffParmReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_report_extent(GlobalDiffSimManager::DiffParmExtentInfo& info) = 0;
    virtual void do_assemble(GlobalDiffSimManager::DiffParmInfo& info)     = 0;
    virtual void do_update(GlobalDiffSimManager::DiffParmUpdateInfo& info) = 0;

  private:
    friend class GlobalDiffSimManager;
    virtual void do_build() override final;
    void         report_extent(GlobalDiffSimManager::DiffParmExtentInfo& info);
    void         assemble_diff_parm(GlobalDiffSimManager::DiffParmInfo& info);
    void         update_diff_parm(GlobalDiffSimManager::DiffParmUpdateInfo& info);
    SizeT        m_index = ~0ull;
};

}  // namespace uipc::backend::luisa
