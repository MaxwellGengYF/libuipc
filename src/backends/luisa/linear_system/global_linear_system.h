#pragma once
#include <sim_system.h>
#include <functional>
#include <uipc/common/list.h>
#include <uipc/common/vector.h>
#include <algorithm/matrix_converter.h>
#include <linear_system/spmv.h>
#include <utils/offset_count_collection.h>
#include <energy_component_flags.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>

namespace uipc::backend::luisa
{
// Define a simple POD to avoid constructing CUDA's built-in vector type with pmr allocators in host code
struct SizeT2
{
    SizeT x;
    SizeT y;
};

class DiagLinearSubsystem;
class OffDiagLinearSubsystem;
class IterativeSolver;
class LocalPreconditioner;
class GlobalPreconditioner;

class GlobalLinearSystem : public SimSystem
{
    static constexpr SizeT DoFBlockSize = 3;

  public:
    using SimSystem::SimSystem;
    using TripletMatrixView3 = TripletMatrixView<Float, 3>;
    using CBCOOMatrixView3   = CBCOOMatrixView<Float, 3>;
    using DenseVectorView    = luisa::compute::BufferView<Float>;
    using CDenseVectorView   = luisa::compute::BufferView<const Float>;
    using ComponentFlags     = EnergyComponentFlags;

    class Impl;

    class InitDofExtentInfo
    {
      public:
        void extent(SizeT dof_count) noexcept { m_dof_count = dof_count; }

      private:
        friend class Impl;
        SizeT m_dof_count = 0;
    };

    class InitDofInfo
    {
      public:
        IndexT dof_offset() const { return m_dof_offset; }
        IndexT dof_count() const { return m_dof_count; }

      private:
        friend class Impl;
        IndexT m_dof_offset = 0;
        IndexT m_dof_count  = 0;
    };

    class DiagExtentInfo
    {
      public:
        bool           gradient_only() const { return m_gradient_only; }
        ComponentFlags component_flags() const { return m_component_flags; }
        void extent(SizeT hessian_count, SizeT dof_count) noexcept;

      private:
        friend class Impl;
        ComponentFlags m_component_flags = ComponentFlags::All;
        SizeT          m_dof_count       = 0;
        SizeT          m_block_count     = 0;
        bool           m_gradient_only   = false;
    };

    class ComputeGradientInfo
    {
      public:
        /**
         * Output gradient vector view
         */
        void buffer_view(luisa::compute::BufferView<Float> grad) noexcept;
        /**
         * Specify which component to be taken into account during gradient computation
         * - Contact: only consider contact part
         * - Complement: only consider non-contact part
         */
        void flags(ComponentFlags component) noexcept;

      private:
        friend class Impl;
        luisa::compute::BufferView<Float> m_gradients;
        ComponentFlags                    m_flags = ComponentFlags::All;
    };

    class DiagInfo
    {
      public:
        DiagInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        TripletMatrixView3 hessians() const { return m_hessians; }
        DenseVectorView    gradients() const { return m_gradients; }
        bool               gradient_only() const { return m_gradient_only; }
        ComponentFlags     component_flags() const { return m_component_flags; }

      private:
        friend class Impl;
        SizeT             m_index = ~0ull;
        TripletMatrixView3 m_hessians;
        DenseVectorView   m_gradients;
        bool              m_gradient_only   = false;
        ComponentFlags    m_component_flags = ComponentFlags::All;

        Impl* m_impl = nullptr;
    };

    class OffDiagExtentInfo
    {
      public:
        void extent(SizeT lr_hessian_block_count, SizeT rl_hessian_block_count) noexcept;

      private:
        friend class Impl;
        SizeT m_lr_block_count = 0;
        SizeT m_rl_block_count = 0;
    };

    class OffDiagInfo
    {
      public:
        OffDiagInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        TripletMatrixView3 lr_hessian() const { return m_lr_hessian; }
        TripletMatrixView3 rl_hessian() const { return m_rl_hessian; }

      private:
        friend class Impl;
        SizeT             m_index = ~0ull;
        TripletMatrixView3 m_lr_hessian;
        TripletMatrixView3 m_rl_hessian;
        Impl*             m_impl = nullptr;
    };

    class AssemblyInfo
    {
      public:
        AssemblyInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        CBCOOMatrixView3 A() const;

      protected:
        friend class Impl;
        Impl* m_impl = nullptr;
    };

    class GlobalPreconditionerAssemblyInfo : public AssemblyInfo
    {
      public:
        using AssemblyInfo::AssemblyInfo;
    };

    class LocalPreconditionerAssemblyInfo : public AssemblyInfo
    {
      public:
        LocalPreconditionerAssemblyInfo(Impl* impl, SizeT index) noexcept
            : AssemblyInfo(impl)
            , m_index(index)
        {
        }

        SizeT dof_offset() const;
        SizeT dof_count() const;

      private:
        SizeT m_index;
    };

    class ApplyPreconditionerInfo
    {
      public:
        ApplyPreconditionerInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        DenseVectorView  z() { return m_z; }
        CDenseVectorView r() { return m_r; }

      private:
        friend class Impl;
        DenseVectorView  m_z;
        CDenseVectorView m_r;
        Impl*            m_impl = nullptr;
    };

    class AccuracyInfo
    {
      public:
        AccuracyInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        CDenseVectorView r() const { return m_r; }

        void satisfied(bool statisfied) { m_statisfied = statisfied; }

      private:
        friend class Impl;
        CDenseVectorView m_r;
        Impl*            m_impl       = nullptr;
        bool             m_statisfied = true;
    };

    class SolvingInfo
    {
      public:
        SolvingInfo(Impl* impl)
            : m_impl(impl)
        {
        }

        DenseVectorView  x() { return m_x; }
        CDenseVectorView b() { return m_b; }
        void iter_count(SizeT iter_count) { m_iter_count = iter_count; }

      private:
        friend class Impl;
        DenseVectorView  m_x;
        CDenseVectorView m_b;
        SizeT            m_iter_count = 0;
        Impl*            m_impl       = nullptr;
    };

    class SolutionInfo
    {
      public:
        SolutionInfo(Impl* impl)
            : m_impl(impl)
        {
        }

        CDenseVectorView solution() { return m_solution; }

      private:
        friend class Impl;
        CDenseVectorView m_solution;
        Impl*            m_impl = nullptr;
    };

  private:
    class LinearSubsytemInfo
    {
      public:
        bool  is_diag                  = false;
        bool  has_local_preconditioner = false;
        SizeT local_index              = ~0ull;
        SizeT index                    = ~0ull;
    };

  public:
    class Impl
    {
      public:
        void init();

        void build_linear_system();
        bool _update_subsystem_extent();
        void _assemble_linear_system();
        void _assemble_preconditioner();
        void solve_linear_system();
        void distribute_solution();

        Float reserve_ratio = 1.1;

        std::vector<LinearSubsytemInfo> subsystem_infos;

        OffsetCountCollection<IndexT> diag_dof_offsets_counts;
        OffsetCountCollection<IndexT> subsystem_triplet_offsets_counts;

        std::vector<SizeT2> off_diag_lr_triplet_counts;


        std::vector<int> accuracy_statisfied_flags;
        std::vector<int> no_precond_diag_subsystem_indices;

        // Containers
        SimSystemSlotCollection<DiagLinearSubsystem>    diag_subsystems;
        SimSystemSlotCollection<OffDiagLinearSubsystem> off_diag_subsystems;
        SimSystemSlotCollection<LocalPreconditioner>    local_preconditioners;

        SimSystemSlot<IterativeSolver>      iterative_solver;
        SimSystemSlot<GlobalPreconditioner> global_preconditioner;

        // Linear System - Using LuisaCompute buffers
        luisa::compute::Buffer<Float> x;
        luisa::compute::Buffer<Float> b;
        
        // Triplet matrix storage using DeviceTripletMatrix3 from global_dytopo_effect_manager
        DeviceTripletMatrix3 triplet_A;
        DeviceBCOOMatrix3    bcoo_A;
        
        // Dense matrix for debug (if needed)
        luisa::compute::Buffer<Float> debug_A;  // dense A for debug

        Spmv                      spmver;
        MatrixConverter<Float, 3> converter;

        bool empty_system = true;

        void apply_preconditioner(luisa::compute::BufferView<Float>       z,
                                  luisa::compute::BufferView<const Float> r);

        void spmv(Float a, luisa::compute::BufferView<const Float> x, Float b, luisa::compute::BufferView<Float> y);
        void spmv_dot(luisa::compute::BufferView<const Float> x, 
                      luisa::compute::BufferView<Float> y, 
                      luisa::compute::BufferView<Float> d_dot);

        bool accuracy_statisfied(luisa::compute::BufferView<Float> r);

        void compute_gradient(ComputeGradientInfo& info);

        bool        need_debug_dump = false;
        std::string debug_dump_path;
    };

    SizeT dof_count() const;

    /**
     * @brief Interface to compute the gradient of the system.
     * 
     * The size of the gradient buffer should be equal to `dof_count()`.
     */
    void compute_gradient(ComputeGradientInfo& info);

  protected:
    void do_build() override;

  private:
    friend class SimEngine;
    friend class IterativeSolver;
    friend class DiagLinearSubsystem;
    friend class OffDiagLinearSubsystem;
    friend class LocalPreconditioner;
    friend class GlobalPreconditioner;
    friend class GlobalDiffSimManager;
    friend class CurrentFrameDiffDofReporter;

    void add_subsystem(DiagLinearSubsystem* subsystem);
    void add_subsystem(OffDiagLinearSubsystem* subsystem);
    void add_solver(IterativeSolver* solver);
    void add_preconditioner(LocalPreconditioner* preconditioner);
    void add_preconditioner(GlobalPreconditioner* preconditioner);

    // only be called by SimEngine::do_init();
    void init();

    // only be called by SimEngine::do_advance()
    void solve();

    Impl m_impl;

    // local debug dump functions
    void _dump_A_b();
    void _dump_x();
};
}  // namespace uipc::backend::luisa
