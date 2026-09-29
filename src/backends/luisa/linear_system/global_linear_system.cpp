#include <linear_system/global_linear_system.h>
#include <linear_system/global_linear_system_function.h>
#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/local_preconditioner.h>
#include <uipc/common/range.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/log.h>
#include <uipc/common/timer.h>
#include <algorithm>

/*************************************************************************************************
* T5.4a — GlobalLinearSystem, the assembly/layout half.
*
* See the big scope note in global_linear_system.h for the exact now/defer split.
* Everything below is the reference's
* src/backends/cuda/linear_system/global_linear_system.cu minus:
*   - solve(), _dump_A_b(), _dump_x()  (the debug/UDA path; they live next to
*     solve()'s `need_debug_dump` branch and drag in <utils/matrix_market.h> +
*     BackendPathTool -- T8.3),
*   - Impl::solve_linear_system(), Impl::apply_preconditioner(), Impl::spmv(),
*     Impl::spmv_dot(), matrix_data_ptrs()  (T8.3: they need IterativeSolver,
*     GlobalPreconditioner, the CUDA-graph validity key),
*   - the `triplet_count_dev = ...` upload in `build_linear_system()` (its only
*     consumer is the graph-stable spmv_dot).
* All of those are declared-but-undefined in the header, each with a
* `SCOPE NOTE (T8.3)`.
*
* ONE MORE SCOPE NOTE (T8.3) — the OFF-DIAG subsystem
* ---------------------------------------------------
* `OffDiagLinearSubsystem` does not exist in this backend yet (it is T8.3's, and
* T5.1's rule 13 forbids inventing a placeholder type), so the reference's
* off-diag branches cannot be transcribed: `Impl::init()`'s second sort,
* `_update_subsystem_extent()`'s `SizeT2{lr, rl}` fill,
* `_assemble_linear_system()`'s two `.submatrix(...)` views and
* `_assemble_preconditioner()`'s global-preconditioner block all dereference
* types that only T8.3 defines.  They are marked `SCOPE NOTE (T8.3)` at each site.
* The port keeps `off_diag_subsystems` / `off_diag_lr_triplet_counts` /
* `off_diag_span` sized from the *host* subsystem table and treats the off-diag
* count as zero; since the reference's collection is empty for every scene of
* this backend, the observable behaviour is identical (`subsystem_infos` holds
* exactly the diag entries, in the same order, with the same `index`).
*************************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(GlobalLinearSystem);

SizeT GlobalLinearSystem::dof_count() const
{
    UIPC_ASSERT(m_impl.initialized,
                "GlobalLinearSystem::dof_count() is called before GlobalLinearSystem::init().");
    return m_impl.diag_dof_offsets_counts.total_count();
}

void GlobalLinearSystem::compute_gradient(ComputeGradientInfo& info)
{
    m_impl.compute_gradient(info);
}

void GlobalLinearSystem::init()
{
    m_impl.init();
}

void GlobalLinearSystem::do_build()
{
    // SCOPE NOTE (T8.3): the reference's whole body is
    //   auto attr = world().scene().config().find<IndexT>("extras/debug/dump_linear_system");
    //   m_impl.need_debug_dump = attr ? attr->view()[0] : false;
    // i.e. the *debug dump* switch, whose only consumer is solve()'s
    // `_dump_A_b()`/`_dump_x()` pair.  `do_build()` must still exist (it is a
    // pure-virtual override and the system has to be constructible), so it is a
    // no-op here and the config read lands with the dump path in T8.3.
}

void GlobalLinearSystem::Impl::init()
{
    // 1) Init all diag subsystems and off-diag subsystems

    auto diag_subsystem_view = diag_subsystems.view();

    {
        // Sort the diag subsystems by their UIDs to ensure the order is consistent
        // ref: https://github.com/spiriMirror/libuipc/issues/271
        std::ranges::sort(diag_subsystem_view,
                          [](const DiagLinearSubsystem* a, const DiagLinearSubsystem* b)
                          { return a->uid() < b->uid(); });

        // SCOPE NOTE (T8.3): the reference sorts `off_diag_subsystems.view()` right
        // here as well (same uid predicate).  `OffDiagLinearSubsystem` is T8.3's
        // type, so the collection cannot be viewed yet; it is empty in every scene
        // of this backend, so the order and the counts below are identical.
    }

    auto total_count = diag_subsystem_view.size();  // + off_diag_subsystem_view.size(): T8.3
    subsystem_infos.resize(total_count);

    // Diag System Always Go First
    auto diag_span = std::span{subsystem_infos}.subspan(0, diag_subsystem_view.size());
    // Off Diag System Always After Diag System (T8.3; empty here)
    auto off_diag_span =
        std::span{subsystem_infos}.subspan(diag_subsystem_view.size(), SizeT{0});
    {
        auto offset = 0;
        for(auto i : range(diag_span.size()))
        {
            auto& dst_diag                  = diag_span[i];
            dst_diag.is_diag                = true;
            dst_diag.local_index            = i;
            auto index                      = offset + i;
            dst_diag.index                  = index;
            diag_subsystem_view[i]->m_index = index;
        }

        offset += diag_subsystem_view.size();
        for(auto i : range(off_diag_span.size()))
        {
            auto& dst_off_diag       = off_diag_span[i];
            dst_off_diag.is_diag     = false;
            dst_off_diag.local_index = i;
            dst_off_diag.index       = offset + i;
            // SCOPE NOTE (T8.3): the reference also writes `->m_index` here (via
            // the off-diag subsystem's own `init()` below).
        }

        for(auto diag_subsystem : diag_subsystem_view)
            diag_subsystem->init();

        // SCOPE NOTE (T8.3): `for(auto off_diag : off_diag_subsystem_view) off_diag->init();`
    }

    // 2) DoF Offsets/Counts
    //
    // The reference computes `counts[i] = report_init_extent(...)` and then lets
    // `OffsetCountCollection::scan()` prefix-sum them.  The port derives the
    // offsets with the *factored* layout routine
    // (`global_linear_system_detail::plan_dof_layout`), which is the same
    // uid-ascending exclusive scan and is therefore also drivable by a unit test
    // (the T5.4a brief's §2); the collection is still filled and scanned so every
    // existing accessor (`counts()`, `offsets()`, `total_count()`, `operator[]`)
    // behaves exactly as before.
    {
        diag_dof_offsets_counts.resize(diag_subsystem_view.size());
        auto               diag_dof_counts = diag_dof_offsets_counts.counts();
        std::vector<U64>   uids(diag_subsystem_view.size());
        std::vector<SizeT> counts(diag_subsystem_view.size());
        for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
        {
            InitDofExtentInfo info;
            diag_subsystem->report_init_extent(info);
            uids[i]            = diag_subsystem->uid();
            counts[i]          = info.m_dof_count;
            diag_dof_counts[i] = info.m_dof_count;
        }
        diag_dof_offsets_counts.scan();

        auto layout = global_linear_system_detail::plan_dof_layout(
            std::span<const U64>(uids), std::span<const SizeT>(counts));
        UIPC_ASSERT(layout.total_count() == diag_dof_offsets_counts.total_count(),
                    "dof layout mismatch: factored layout total {} vs collection total {}",
                    layout.total_count(),
                    diag_dof_offsets_counts.total_count());

        for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
        {
            InitDofInfo info;
            info.m_dof_offset = static_cast<IndexT>(layout.offsets[i]);
            info.m_dof_count  = static_cast<IndexT>(layout.counts[i]);
            diag_subsystem->receive_init_dof_info(info);
        }
    }
    accuracy_statisfied_flags.resize(diag_subsystem_view.size());

    // 3) Triplet Offsets/Counts
    subsystem_triplet_offsets_counts.resize(total_count);
    // SCOPE NOTE (T8.3): the reference sizes this from the off-diag count
    // (`off_diag_lr_triplet_counts.resize(off_diag_subsystem_view.size())`); the
    // off-diag collection is empty here, so the size is 0 -- the member is kept
    // (and stays empty) for the T8.3 branch.
    off_diag_lr_triplet_counts.resize(0);

    // 4) Preconditioner
    // find out diag systems that don't have preconditioner
    auto local_preconditioner_view = local_preconditioners.view();

    for(auto precond : local_preconditioner_view)
    {
        auto index = precond->m_subsystem->m_index;
        diag_span[index].has_local_preconditioner = true;
    }

    no_precond_diag_subsystem_indices.reserve(diag_span.size());
    for(auto&& [i, diag_info] : enumerate(diag_span))
    {
        if(!diag_info.has_local_preconditioner)
        {
            no_precond_diag_subsystem_indices.push_back(static_cast<int>(i));
        }
    }

    for(auto precond : local_preconditioner_view)
    {
        precond->init();
    }

    initialized = true;
}

void GlobalLinearSystem::Impl::build_linear_system()
{
    Timer timer{"Build Linear System"};
    empty_system = !_update_subsystem_extent();

    if(empty_system) [[unlikely]]
    {
        logger::warn(
            "The global linear system is empty, skip *assembling, *solving and *solution distributing phase.");
        return;
    }

    {
        Timer t{"Assemble Subsystems"};
        _assemble_linear_system();
    }

    {
        Timer t{"Convert To BCOO"};
        converter.ge2sym(triplet_A);
        converter.convert(triplet_A, bcoo_A);
        // SCOPE NOTE (T8.3): the reference uploads the nnz count here
        //   `triplet_count_dev = (IndexT)bcoo_A.triplet_count();`
        // for the graph-stable `spmv_dot` launch.  Its only consumer is the
        // (deferred) spmv_dot, so the upload lands with T8.3.
    }

    {
        Timer t{"Assemble Preconditioner"};
        _assemble_preconditioner();
    }

    logger::info("GlobalLinearSystem has {} DoFs, Unique Triplet Count: {}",
                 b.size(),
                 bcoo_A.triplet_count());
}

bool GlobalLinearSystem::Impl::_update_subsystem_extent()
{
    bool dof_count_changed     = false;
    bool triplet_count_changed = false;

    auto diag_subsystem_view       = diag_subsystems.view();
    auto diag_dof_counts           = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets          = diag_dof_offsets_counts.offsets();
    auto subsystem_triplet_counts  = subsystem_triplet_offsets_counts.counts();
    auto subsystem_triplet_offsets = subsystem_triplet_offsets_counts.offsets();

    for(const auto& subsystem_info : subsystem_infos)
    {
        if(subsystem_info.is_diag)
        {
            auto  dof_i          = subsystem_info.local_index;
            auto  triplet_i      = subsystem_info.index;
            auto& diag_subsystem = diag_subsystem_view[dof_i];
            DiagExtentInfo info;
            diag_subsystem->report_extent(info);

            dof_count_changed |= diag_dof_counts[dof_i] != info.m_dof_count;
            diag_dof_counts[dof_i] = info.m_dof_count;


            triplet_count_changed |= subsystem_triplet_counts[triplet_i] != info.m_block_count;
            subsystem_triplet_counts[triplet_i] = info.m_block_count;
        }
        else
        {
            // SCOPE NOTE (T8.3): the reference's off-diag branch
            //   OffDiagExtentInfo info; off_diag_subsystem->report_extent(info);
            //   auto total = info.m_lr_block_count + info.m_rl_block_count;
            //   subsystem_triplet_counts[triplet_i] = total;
            //   off_diag_lr_triplet_counts[local_index] = SizeT2{lr, rl};
            // is unreachable here: `subsystem_infos` holds only diag entries
            // (`OffDiagLinearSubsystem` is T8.3's type).
        }
    }

    SizeT total_dof     = 0;
    SizeT total_triplet = 0;

    if(dof_count_changed)
    {
        diag_dof_offsets_counts.scan();
    }
    total_dof = diag_dof_offsets_counts.total_count();
    if(x.capacity() < total_dof)
    {
        auto reserve_count = static_cast<SizeT>(static_cast<Float>(total_dof) * reserve_ratio);
        x.reserve(reserve_count);
        b.reserve(reserve_count);
    }
    auto blocked_dof = total_dof / DoFBlockSize;
    triplet_A.reshape(static_cast<int>(blocked_dof), static_cast<int>(blocked_dof));
    x.resize(total_dof);
    b.resize(total_dof);

    if(triplet_count_changed) [[likely]]
    {
        subsystem_triplet_offsets_counts.scan();
    }
    total_triplet = subsystem_triplet_offsets_counts.total_count();

    if(triplet_A.triplet_capacity() < total_triplet)
    {
        auto reserve_count =
            static_cast<SizeT>(static_cast<Float>(total_triplet) * reserve_ratio);
        triplet_A.reserve_triplets_discard(reserve_count);
        bcoo_A.reserve_triplets_discard(reserve_count);
    }
    triplet_A.resize_triplets_discard(total_triplet);

    if(total_dof == 0 || total_triplet == 0) [[unlikely]]
    {
        return false;
    }

    return true;
}

void GlobalLinearSystem::Impl::_assemble_linear_system()
{
    auto HA = triplet_A.view();

    // Clear and invalidate previous values
    // (the reference fills the *block* values with `Matrix3x3::Zero()`; T3.1's
    //  flat scalar storage makes that a `fill(0.0)` over nnz * 9 scalars)
    triplet_A.fill(0.0);
    triplet_A.row_indices().fill(-1);
    triplet_A.col_indices().fill(-1);

    auto B = b.view();
    B.buffer_view().fill(0.0);

    auto diag_subsystem_view = diag_subsystems.view();

    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    auto subsystem_triplet_counts  = subsystem_triplet_offsets_counts.counts();
    auto subsystem_triplet_offsets = subsystem_triplet_offsets_counts.offsets();

    for(const auto& subsystem_info : subsystem_infos)
    {
        if(!subsystem_info.is_diag)
        {
            // SCOPE NOTE (T8.3): the reference's off-diag branch builds the two
            // `.submatrix(...)` windows from `off_diag_subsystem->m_l->m_index` /
            // `->m_r->m_index` and calls `off_diag_subsystem->assemble(info)`.
            // `OffDiagLinearSubsystem` is T8.3's type (no placeholder type --
            // T5.1 rule 13), so it lands there; the branch is unreachable here.
            continue;
        }

        auto  dof_i          = subsystem_info.local_index;
        auto  triplet_i      = subsystem_info.index;
        auto& diag_subsystem = diag_subsystem_view[dof_i];

        auto dof_offset         = diag_dof_offsets[dof_i];
        auto dof_count          = diag_dof_counts[dof_i];
        auto blocked_dof_offset = dof_offset / DoFBlockSize;
        auto blocked_dof_count  = dof_count / DoFBlockSize;
        auto ij_offset =
            lla::index_pair<std::int32_t>{static_cast<std::int32_t>(blocked_dof_offset),
                                          static_cast<std::int32_t>(blocked_dof_offset)};
        auto ij_count =
            lla::index_pair<std::int32_t>{static_cast<std::int32_t>(blocked_dof_count),
                                          static_cast<std::int32_t>(blocked_dof_count)};

        DiagInfo info{this};

        info.m_index     = triplet_i;
        info.m_gradients = B.subview(dof_offset, dof_count);
        info.m_hessians  = HA.subview(subsystem_triplet_offsets[triplet_i],
                                     subsystem_triplet_counts[triplet_i])
                              .submatrix(ij_offset, ij_count);

        diag_subsystem->assemble(info);
    }
}

void GlobalLinearSystem::Impl::_assemble_preconditioner()
{
    // SCOPE NOTE (T8.3): the reference first fans out to the *global*
    // preconditioner (`GlobalPreconditionerAssemblyInfo info{this};
    // global_preconditioner->assemble(info);`).  `GlobalPreconditioner` is T8.3's
    // type (forward-declared here, exactly as in the reference header); the slot
    // stays declared and is empty in every scene of this backend.
    for(auto&& preconditioner : local_preconditioners.view())
    {
        LocalPreconditionerAssemblyInfo info{this, preconditioner->m_subsystem->m_index};
        preconditioner->assemble(info);
    }
}

void GlobalLinearSystem::Impl::distribute_solution()
{
    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    // distribute the solution to all diag subsystems
    for(auto&& [i, diag_subsystem] : enumerate(diag_subsystems.view()))
    {
        SolutionInfo info{this};
        info.m_solution = x.view().subview(diag_dof_offsets[i], diag_dof_counts[i]);
        diag_subsystem->retrieve_solution(info);
    }
}

bool GlobalLinearSystem::Impl::accuracy_statisfied(lla::LDenseVectorView<Float> r)
{
    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    for(auto&& [i, diag_subsystem] : enumerate(diag_subsystems.view()))
    {
        AccuracyInfo info{this};
        info.m_r = r.subview(diag_dof_offsets[i], diag_dof_counts[i]);
        diag_subsystem->accuracy_check(info);

        accuracy_statisfied_flags[i] = info.m_statisfied ? 1 : 0;
    }

    return std::ranges::all_of(accuracy_statisfied_flags,
                               [](bool flag) { return flag; });
}

void GlobalLinearSystem::Impl::compute_gradient(ComputeGradientInfo& info)
{
    auto diag_subsystem_view = diag_subsystems.view();

    // report extent first (gradient only mode)
    for(auto diag_subsystem : diag_subsystem_view)
    {
        DiagExtentInfo diag_info;
        diag_info.m_gradient_only   = true;
        diag_info.m_component_flags = info.m_flags;
        diag_subsystem->report_extent(diag_info);
    }

    // assemble gradient only
    for(auto diag_subsystem : diag_subsystem_view)
    {
        DiagInfo diag_info{this};
        diag_info.m_index           = diag_subsystem->m_index;
        diag_info.m_gradients       = info.m_gradients;
        diag_info.m_hessians        = TripletMatrixView{};
        diag_info.m_gradient_only   = true;
        diag_info.m_component_flags = info.m_flags;
        diag_subsystem->assemble(diag_info);
    }
}

Float GlobalLinearSystem::Impl::diag_norm()
{
    Float norm = 0;

    for(auto diag_subsystem : diag_subsystems.view())
    {
        DiagNormInfo info(this, diag_subsystem->m_index);
        norm = std::max(norm, diag_subsystem->diag_norm(info));
    }

    return norm;
}

Float GlobalLinearSystem::Impl::mass_norm()
{
    Float norm = 0;

    for(auto diag_subsystem : diag_subsystems.view())
    {
        DiagNormInfo info(this, diag_subsystem->m_index);
        norm = std::max(norm, diag_subsystem->mass_norm(info));
    }

    return norm;
}

void GlobalLinearSystem::DiagExtentInfo::extent(SizeT hessian_block_count,
                                                SizeT dof_count) noexcept
{
    m_block_count = hessian_block_count;
    UIPC_ASSERT(dof_count % DoFBlockSize == 0,
                "dof_count must be multiple of {}, yours {}.",
                DoFBlockSize,
                dof_count);
    m_dof_count = dof_count;
}

void GlobalLinearSystem::OffDiagExtentInfo::extent(SizeT lr_hessian_block_count,
                                                   SizeT rl_hassian_block_count) noexcept
{
    m_lr_block_count = lr_hessian_block_count;
    m_rl_block_count = rl_hassian_block_count;
}

auto GlobalLinearSystem::AssemblyInfo::A() const -> CBCOOMatrixView
{
    return m_impl->bcoo_A.cview();
}

SizeT GlobalLinearSystem::LocalPreconditionerAssemblyInfo::dof_offset() const
{
    auto diag_dof_offsets = m_impl->diag_dof_offsets_counts.offsets();
    return diag_dof_offsets[m_index];
}

SizeT GlobalLinearSystem::LocalPreconditionerAssemblyInfo::dof_count() const
{
    auto diag_dof_counts = m_impl->diag_dof_offsets_counts.counts();
    return diag_dof_counts[m_index];
}

SizeT GlobalLinearSystem::last_solve_iterations() const noexcept
{
    return m_impl.last_solve_iterations;
}

void GlobalLinearSystem::add_subsystem(DiagLinearSubsystem* subsystem)
{
    check_state(SimEngineState::BuildSystems, "add_subsystem()");
    UIPC_ASSERT(subsystem != nullptr, "The subsystem should not be nullptr.");
    m_impl.diag_subsystems.register_sim_system(*subsystem);
}

void GlobalLinearSystem::add_subsystem(OffDiagLinearSubsystem* subsystem)
{
    check_state(SimEngineState::BuildSystems, "add_subsystem()");
    UIPC_ASSERT(subsystem != nullptr, "The subsystem should not be nullptr.");
    // SCOPE NOTE (T8.3): `m_impl.off_diag_subsystems.register_sim_system(*subsystem);`
    // -- `SimSystemSlotCollection<OffDiagLinearSubsystem>::register_sim_system`
    // instantiates `T::is_valid()`, so it needs the (T8.3) complete type.  The
    // entry point is kept (the reference's public surface) and the registration
    // lands with the off-diag subsystem.
}

void GlobalLinearSystem::add_solver(IterativeSolver* solver)
{
    check_state(SimEngineState::BuildSystems, "add_solver()");
    UIPC_ASSERT(solver != nullptr, "The solver should not be nullptr.");
    // SCOPE NOTE (T8.3): `m_impl.iterative_solver.register_sim_system(*solver);`
    // -- `SimSystemSlot<T>::register_sim_system` instantiates `T::is_valid()`, so
    // it needs the (T8.3) complete `IterativeSolver` type.  The entry point is kept
    // (the reference's public surface) and the registration lands with the solver.
}

void GlobalLinearSystem::add_preconditioner(LocalPreconditioner* preconditioner)
{
    check_state(SimEngineState::BuildSystems, "add_preconditioner()");
    UIPC_ASSERT(preconditioner != nullptr, "The preconditioner should not be nullptr.");
    m_impl.local_preconditioners.register_sim_system(*preconditioner);
}

void GlobalLinearSystem::add_preconditioner(GlobalPreconditioner* preconditioner)
{
    check_state(SimEngineState::BuildSystems, "add_preconditioner()");
    UIPC_ASSERT(preconditioner != nullptr, "The global preconditioner should not be nullptr.");
    // SCOPE NOTE (T8.3): see `add_solver` / `add_subsystem(OffDiagLinearSubsystem*)`
    // -- the registration needs the complete `GlobalPreconditioner` type.
}

void GlobalLinearSystem::ComputeGradientInfo::flags(ComponentFlags flags) noexcept
{
    m_flags = flags;
}

void GlobalLinearSystem::ComputeGradientInfo::buffer_view(DenseVectorView grad) noexcept
{
    m_gradients = grad;
}

// ===========================================================================
// SCOPE NOTE (T8.3) — the solve path, deliberately left undefined here.
//
// Declared in global_linear_system.h and not defined (nothing in T5.4a ODR-uses
// them, which is exactly what keeps the interface complete without a stub):
//
//   void  GlobalLinearSystem::solve();
//   void  GlobalLinearSystem::Impl::solve_linear_system();
//   void  GlobalLinearSystem::Impl::apply_preconditioner(...);
//   void  GlobalLinearSystem::Impl::spmv(...);
//   void  GlobalLinearSystem::Impl::spmv_dot(...);
//   std::array<const void*, 3> GlobalLinearSystem::Impl::matrix_data_ptrs() const;
//   void  GlobalLinearSystem::_dump_A_b();
//   void  GlobalLinearSystem::_dump_x();
//   Float GlobalLinearSystem::diag_norm();   // only called by SimEngine::do_advance
//   Float GlobalLinearSystem::mass_norm();   // only called by SimEngine::do_advance
//   void  GlobalLinearSystem::add_subsystem(OffDiagLinearSubsystem*);
//   void  GlobalLinearSystem::add_preconditioner(GlobalPreconditioner*);
//
// T8.3 adds them together with IterativeSolver / LinearPCG (+ the CUDA-graph
// fused variant, whose validity key is what matrix_data_ptrs() and
// triplet_count_dev exist for), GlobalPreconditioner and
// OffDiagLinearSubsystem, plus the SimEngine::do_init / do_advance wiring.
// ===========================================================================
}  // namespace uipc::backend::luisa
