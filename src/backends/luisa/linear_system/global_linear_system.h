#pragma once
/********************************************************************
 * @file global_linear_system.h
 * @brief T5.4a — the global linear system, ported as "everything except the
 *        solve path" (port of
 *        src/backends/cuda/linear_system/global_linear_system.h).
 *
 * ---------------------------------------------------------------------------
 * THE T5.4a SPLIT (the brief's §1.2) — this is the one file that must not be
 * ported blindly
 * ---------------------------------------------------------------------------
 * `GlobalLinearSystem` is the hub every diag/off-diag subsystem feeds, and in the
 * reference it is *also* the solver: `SimEngine::do_init` requires it and calls
 * `init()`, and `SimEngine::advance` calls `solve()`, which triggers the whole
 * assembly pipeline (`_update_subsystem_extent` -> `_assemble_linear_system` ->
 * `_assemble_preconditioner` -> `solve_linear_system`).
 *
 * `solve()` needs `IterativeSolver` + `LinearPCG` (+ `LinearFusedPCG`/CUDA
 * graphs), `GlobalPreconditioner`, `OffDiagLinearSubsystem` and
 * `CurrentFrameDiffDofReporter` — none of which exist in this backend yet.  They
 * are T8.3's task.  The port therefore keeps the *whole public surface* and the
 * *whole assembly/layout half*, and leaves the solve path declared but
 * **undefined**, each one marked `SCOPE NOTE (T8.3)` below, so nothing is
 * silently dropped (T5.1's rule 13: never invent a placeholder type — an
 * undefined member is fine as long as nothing ODR-uses it).
 *
 * PORTED NOW
 *   * the header's public surface: every `*Info` class (`InitDofExtentInfo`,
 *     `InitDofInfo`, `DiagExtentInfo`, `ComputeGradientInfo`, `DiagInfo`,
 *     `OffDiagExtentInfo`, `OffDiagInfo`, `AssemblyInfo`,
 *     `GlobalPreconditionerAssemblyInfo`, `LocalPreconditionerAssemblyInfo`,
 *     `DiagNormInfo`, `ApplyPreconditionerInfo`, `AccuracyInfo`, `SolvingInfo`,
 *     `SolutionInfo`, the private `LinearSubsytemInfo`), the `SimSystem` base,
 *     the friend set, the `add_*()` registration functions, `dof_count()`,
 *     `last_solve_iterations()`, `compute_gradient()`, `ctx()` and the type
 *     aliases;
 *   * `Impl::{init,build_linear_system,_update_subsystem_extent,
 *     _assemble_linear_system,distribute_solution,accuracy_statisfied,
 *     compute_gradient,diag_norm,mass_norm,_assemble_preconditioner}`,
 *     `DiagExtentInfo::extent`, `OffDiagExtentInfo::extent`,
 *     `AssemblyInfo::A`, `LocalPreconditionerAssemblyInfo::{dof_offset,dof_count}`,
 *     `ComputeGradientInfo::{flags,buffer_view}` and the `add_*()` definitions.
 *
 * DEFERRED TO T8.3 (declared here, not defined — `SCOPE NOTE (T8.3)`)
 * * `do_build()` — only defined as an empty body, because it is a pure-virtual
 *   override (a system must be constructible) and the reference's whole body is
 *   the *debug dump* config (`extras/debug/dump_linear_system`), which is
 *   decidable only together with `solve()`;
 * * `solve()`, `Impl::solve_linear_system()`, `Impl::spmv()`,
 *   `Impl::spmv_dot()`, `matrix_data_ptrs()`;
 * * `_dump_A_b()` / `_dump_x()` (they need `utils/matrix_market` + the
 *   `BackendPathTool` workspace plumbing, i.e. the debug path) — the references
 *   to them are inside `solve()`, which is itself deferred;
 * * `need_debug_dump` / `debug_dump_path` and the graph-stability key
 *   `triplet_count_dev` (whose only purpose is a launch shape that does not
 *   change when nnz varies inside the reserved capacity).
 *
 * COMPLETED BY T5.4b: `Impl::apply_preconditioner()` — the local-preconditioner
 * fan-out plus the no-local-preconditioner `z <- r` copy fallback.  It was under
 * this file's SCOPE NOTE (T8.3) because T5.4a had no `LocalPreconditioner`
 * implementation to call and no `LocalPreconditioner` static apply helper; the
 * *global* preconditioner branch inside it stays deferred (that type is T8.3's).
 *
 * Container mapping onto T3.1/T3.3/T3.4 (cookbook §7.9/§7.11/§7.12)
 * ----------------------------------------------------------------
 *   cuda_tool::DeviceTripletMatrix<Float,3>  -> lla::LTripletMatrix<Float,3,3>
 *   cuda_tool::DeviceBCOOMatrix<Float,3>     -> lla::LBCOOMatrix<Float,3,3>
 *   cuda_tool::DeviceDenseVector<Float>      -> lla::LDenseVector<Float>
 *   cuda_tool::DeviceDenseMatrix<Float>      -> lla::LDenseMatrix<Float>
 *   cuda_tool::DeviceVar<IndexT>             -> lud::LDeviceVar<IndexT>
 *   cuda_tool::LinearSystemContext           -> lla::LinearSystemContext
 *   cuda_tool::*View<...>                    -> lla::L*View<...>
 *   Spmv / MatrixConverter<Float,3>          -> lla::Spmv / lla::MatrixConverter<Float,3>
 *   cuda_tool::VarView<T> / CVarView<T>      -> lud::LBufferView<T> / lud::LCBufferView<T>
 *   cudaStream_t                             -> lud::LStream*
 *
 * TEST-REACHABLE VS SOLVER-ONLY (the brief's §2)
 * ---------------------------------------------
 * reachable from a unit test / a scene:
 *   `init()`, `dof_count()`, `ctx()`, `compute_gradient()` (the *gradient-only*
 *   assembly: `DiagExtentInfo{gradient_only=true}` + `DiagInfo` +
 *   `DiagSubsystem::assemble`, i.e. `FEMLinearSubsystem::Impl::assemble` and the
 *   whole reporter/constitution fan-out), and — through a test-local `Impl`
 * instance, which is a *public* nested class — `_update_subsystem_extent()`,
 * `_assemble_linear_system()`, `distribute_solution()`, `accuracy_statisfied()`,
 * `diag_norm()`, `mass_norm()`; and, since T5.4b, `build_linear_system()` (the
 * extent -> assemble -> convert -> `_assemble_preconditioner()` chain) and
 * `apply_preconditioner()` on such an instance, once a `FEMDiagPreconditioner` is
 * registered in it;
 * solver-only (T8.3): the `solve()` chain, the *global*-preconditioner branch of
 * `apply_preconditioner`, `spmv`, `spmv_dot`, the debug dump and the CUDA-graph
 * validity key.
 *
 * DEVIATION (T5.4a, reported): `init()` is declared **public** here.  The
 * reference keeps it private + `friend class SimEngine`, because only
 * `SimEngine::do_init()` may call it.  This port does not wire the engine (per the
 * brief), so with the reference's access level the layout half (dof offsets,
 * `dof_count()`, the gradient-only assembly through `compute_gradient()`) would be
 * unreachable from a test, which the brief's §2 forbids; the alternatives were a
 * test-only friend (T5.1 rule 9 forbids widening the friend list) or a test-only
 * hook.  Everything else keeps the reference's access levels.
 *********************************************************************/
#include <sim_system.h>
#include <functional>
#include <array>
#include <uipc/common/list.h>
#include <uipc/common/vector.h>
#include <luisa_device/linear/linear.h>
#include <utils/offset_count_collection.h>
#include <energy_component_flags.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

/// Define a simple POD to avoid constructing a device-vector type with pmr
/// allocators in host code (verbatim from the reference).
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
// T8.3: declared for the reference's `friend` list (they do not exist in this
// backend yet; a friend declaration may introduce the name).
class GlobalDiffSimManager;
class CurrentFrameDiffDofReporter;

class GlobalLinearSystem : public SimSystem
{
    static constexpr SizeT DoFBlockSize = 3;

  public:
    using SimSystem::SimSystem;
    using TripletMatrixView = lla::LTripletMatrixView<Float, 3, 3>;
    using CBCOOMatrixView   = lla::CBCOOMatrixView<Float, 3, 3>;
    using DenseVectorView   = lla::LDenseVectorView<Float>;
    using CDenseVectorView  = lla::LCDenseVectorView<Float>;
    using ComponentFlags    = EnergyComponentFlags;

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
        void extent(SizeT hessian_block_count, SizeT dof_count) noexcept;

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
        // - Contact: only consider contact part
        // - Complement: only consider non-contact part
        void flags(ComponentFlags component) noexcept;
        void buffer_view(DenseVectorView grad) noexcept;

      private:
        friend class Impl;
        DenseVectorView m_gradients;
        ComponentFlags  m_flags = ComponentFlags::All;
    };

    class DiagInfo
    {
      public:
        DiagInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        TripletMatrixView hessians() { return m_hessians; }
        DenseVectorView   gradients() { return m_gradients; }
        bool              gradient_only() const { return m_gradient_only; }
        ComponentFlags    component_flags() const { return m_component_flags; }

      private:
        friend class Impl;
        SizeT             m_index = ~0ull;
        TripletMatrixView m_hessians;
        DenseVectorView   m_gradients;
        bool              m_gradient_only   = false;
        ComponentFlags    m_component_flags = ComponentFlags::All;
        Impl*             m_impl            = nullptr;
    };

    class OffDiagExtentInfo
    {
      public:
        void extent(SizeT lr_hessian_block_count, SizeT rl_hassian_block_count) noexcept;

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

        TripletMatrixView lr_hessian() const { return m_lr_hessian; }
        TripletMatrixView rl_hessian() const { return m_rl_hessian; }

      private:
        friend class Impl;
        SizeT             m_index = ~0ull;
        TripletMatrixView m_lr_hessian;
        TripletMatrixView m_rl_hessian;
        Impl*             m_impl = nullptr;
    };

    class AssemblyInfo
    {
      public:
        AssemblyInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        CBCOOMatrixView A() const;

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

    using DiagNormInfo = LocalPreconditionerAssemblyInfo;

    class ApplyPreconditionerInfo
    {
      public:
        ApplyPreconditionerInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        DenseVectorView  z() { return m_z; }
        CDenseVectorView r() { return m_r; }
        // cuda_tool::CVarView<IndexT> (T1.2's replacement, see the file header)
        lud::LCBufferView<IndexT> converged() { return m_converged; }
        // launch stream for the apply kernels (the capture stream while recording
        // a CUDA graph; the legacy default stream otherwise).  `cudaStream_t` has
        // no LuisaCompute equivalent -> the port's stream handle.
        lud::LStream* stream() const noexcept { return m_stream; }

      private:
        friend class Impl;
        DenseVectorView           m_z;
        CDenseVectorView          m_r;
        lud::LCBufferView<IndexT> m_converged;
        lud::LStream*             m_stream = nullptr;
        Impl*                     m_impl   = nullptr;
    };

    class AccuracyInfo
    {
      public:
        AccuracyInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        CDenseVectorView r() const { return m_r; }

        void satisfied(bool satisfied) { m_statisfied = satisfied; }

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
        // SCOPE NOTE (T8.3): declared, deliberately not defined here.
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

        // Linear System
        lla::LinearSystemContext                ctx;
        lla::LDenseVector<Float>                x;
        lla::LDenseVector<Float>                b;
        lla::LTripletMatrix<Float, 3, 3>        triplet_A;
        lla::LBCOOMatrix<Float, 3, 3>           bcoo_A;
        lla::LDenseMatrix<Float>                debug_A;  // dense A for debug
        // SCOPE NOTE (T8.3): device-side copy of bcoo_A's triplet count, for the
        // graph-stable SpMV launch (its only consumer is `spmv_dot`).  Declared so
        // the interface is complete; `build_linear_system()` does not fill it.
        lud::LDeviceVar<IndexT> triplet_count_dev;

        lla::Spmv spmver;
        lla::MatrixConverter<Float, 3> converter;

        bool  initialized           = false;
        bool  empty_system          = true;
        SizeT last_solve_iterations = 0;

        // SCOPE NOTE (T8.3): the remaining apply/spmv entries of the solve path —
        // only the *global*-preconditioner branch of `apply_preconditioner` is left
        // to them (T5.4b defined the local fan-out + the copy fallback).  The
        // `cudaStream_t` parameters map onto the port's `lud::LStream*`.
        void apply_preconditioner(lla::LDenseVectorView<Float>  z,
                                  lla::LCDenseVectorView<Float> r,
                                  lud::LCBufferView<IndexT>     converged,
                                  lud::LStream*                 stream = nullptr);

        void spmv(Float                        a,
                  lla::LCDenseVectorView<Float> x,
                  Float                        b,
                  lla::LDenseVectorView<Float>  y);
        void spmv_dot(lla::LCDenseVectorView<Float> x,
                      lla::LDenseVectorView<Float>  y,
                      lud::LBufferView<Float>       d_dot,
                      lud::LStream*                 stream = nullptr);

        // SCOPE NOTE (T8.3): data pointers of the assembled matrix buffers
        // (row/col/value), used by the FusedPCG CUDA-graph validity key.  The
        // reference's body is `bcoo_A.cview().{row_indices,col_indices,values()}.data()`
        // — a raw device pointer, which a buffer DSL does not expose (cookbook
        // §7.9, T3.1's "no raw element pointers"); it needs the graph-validity
        // reformulation (a generation/epoch key) that T8.3 owns.  Declared, not
        // defined.
        std::array<const void*, 3> matrix_data_ptrs() const;


        bool accuracy_statisfied(lla::LDenseVectorView<Float> r);
        void compute_gradient(ComputeGradientInfo& info);

        Float diag_norm();
        Float mass_norm();

        // SCOPE NOTE (T8.3): the debug dump configuration, read by the reference's
        // `do_build()` and consumed by `solve()`.
        bool        need_debug_dump = false;
        std::string debug_dump_path;
    };

    /// The number of dofs the layout assigned (`Impl::init()` may have run).
    ///
    /// DEVIATION (T5.4a): public, so the layout half is reachable from a test
    /// (see the file header).  `SimEngine::do_init()` is its reference caller.
    void init();

    SizeT dof_count() const;
    SizeT last_solve_iterations() const noexcept;
    void  compute_gradient(ComputeGradientInfo& info);

    lla::LinearSystemContext& ctx() noexcept { return m_impl.ctx; }

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

    // SCOPE NOTE (T8.3): `solve()` (the only caller of the whole assembly
    // pipeline in the reference) and the two norm entries `SimEngine::do_advance`
    // calls.  They are declared-but-undefined; nothing in T5.4a ODR-uses them.
    // The assembly they trigger *is* ported (see `Impl`), and a test reaches it
    // through `init()` + `compute_gradient()`, or through a test-local `Impl`.
    void solve();

    // only be called by SimEngine::do_advance()
    Float diag_norm();
    Float mass_norm();

    Impl m_impl;

    // local debug dump functions (T8.3, with the debug path of `solve()`)
    void _dump_A_b();
    void _dump_x();
};
}  // namespace uipc::backend::luisa
