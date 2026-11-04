#include <linear_system/global_linear_system.h>
#include <linear_system/global_linear_system_function.h>
#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/off_diag_linear_subsystem.h>
#include <linear_system/local_preconditioner.h>
#include <linear_system/global_preconditioner.h>
#include <linear_system/iterative_solver.h>
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
* - solve(), _dump_A_b(), _dump_x()  (the debug/dump path; they live next to
* solve()'s `need_debug_dump` branch and drag in <utils/matrix_market.h> +
* BackendPathTool -- plan 11),
* - `matrix_data_ptrs()`'s raw-pointer body (it is an identity token here:
* there is no graph to key, plan §3.3(3)).
* The remaining declared-but-undefined entries are listed at the end of this file
* under `SCOPE NOTE (T8.3 → plan 11)`.
*
* T8.3a-3 — THE GLOBAL-PRECONDITIONER BRANCH IS LIVE (the last SCOPE NOTE resolved)
* ---------------------------------------------------------------------------------
* `GlobalPreconditioner` exists (linear_system/global_preconditioner.{h,cpp}), so the
* three places that were waiting for the complete type are the reference's again:
*   * `add_preconditioner(GlobalPreconditioner*)` is a real
*     `m_impl.global_preconditioner.register_sim_system(*preconditioner);`,
*   * `_assemble_preconditioner()` fans out to the global preconditioner **first**
*     (global_linear_system.cu:422-435) and then runs the local loop
*     unconditionally,
*   * `apply_preconditioner()` applies the global preconditioner to the **whole**
*     z/r first, then the unconditional local fan-out, and `!global_preconditioner`
*     gates **only** the `z <- r` fallback (.cu:466-510).
* Nothing in the local path changed: [mas_preconditioner] and [fem_precond] are the
* same numbers, with or without a registered global preconditioner (the slot is empty
* in every scene of this backend — there is no shipped concrete implementation,
* exactly like the reference).
*
* T8.3a-2 — THE SOLVER SUBSTRATE IS LIVE (the seam agreed with plan 11, research §4.3)
* ------------------------------------------------------------------------------------
* `IterativeSolver`'s utility forwards dereference `m_system->m_impl`, so defining
* them ODR-uses these four bodies — they are the reference's, transcribed here now:
*   * `Impl::solve_linear_system()` (global_linear_system.cu:437-447),
*   * `Impl::spmv()`           (".cu:512-522"),
*   * `Impl::spmv_dot()`       (".cu:524-537") — `matrix_data_ptrs()` is *not* a raw
*     pointer triple here, see its body,
*   * the `triplet_count_dev = (IndexT)bcoo_A.triplet_count();` upload in
*     `build_linear_system()` (".cu:237"),
*   * `add_solver()` (".cu:672-679"), a real `register_sim_system`.
* `solve()` itself, the two dumps and `diag_norm`/`mass_norm` stay plan 11's.
*
* T8.3a-1 — THE OFF-DIAG SUBSYSTEM IS LIVE (the previous scope note is resolved)
* -----------------------------------------------------------------------------
* T5.4a could not transcribe the reference's four off-diag branches, because
* `OffDiagLinearSubsystem` did not exist and `SimSystemSlotCollection<T>` needs a
* *complete* `T`.  With linear_system/off_diag_linear_subsystem.h in the tree
* they are transcribed verbatim now:
*   * `Impl::init()`'s second (uid-ascending) sort, the
*     `diag + off_diag` subsystem table with `index = diag_count + i`,
*     `off_diag->init()` and `off_diag_lr_triplet_counts.resize(off_diag_count)`;
*   * `_update_subsystem_extent()`'s `else` branch (`{lr, rl}` per off-diag
*     subsystem, `subsystem_triplet_counts[idx] = lr + rl`);
*   * `_assemble_linear_system()`'s two `.submatrix(...)` windows
*     (`lr` at `(l_off, r_off)`, `rl` at `(r_off, l_off)`);
*   * `add_subsystem(OffDiagLinearSubsystem*)`.
* The one addition (not in the reference): the duplicate/self-pair rejection,
* see `off_diag_layout_detail::check_unique_pairs()` +
* global_linear_system_function.h.
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

    auto diag_subsystem_view     = diag_subsystems.view();
    auto off_diag_subsystem_view = off_diag_subsystems.view();

    {
        // Sort the diag subsystems by their UIDs to ensure the order is consistent
        // ref: https://github.com/spiriMirror/libuipc/issues/271
        std::ranges::sort(diag_subsystem_view,
                          [](const DiagLinearSubsystem* a, const DiagLinearSubsystem* b)
                          { return a->uid() < b->uid(); });
        // ... and the off-diag subsystems by their (l, r) uid pair: the block
        // order of the global matrix *is* this order (plan 14 §4: never reorder
        // the block layout to make a dump nicer -- BSR/PCG depend on it).
        std::ranges::sort(off_diag_subsystem_view,
                          [](const OffDiagLinearSubsystem* a,
                             const OffDiagLinearSubsystem* b) -> bool
                          { return a->uid() < b->uid(); });
    }

    {
        // Reject duplicate / self-referential off-diag pairs BEFORE anything is
        // sized from the off-diag count.  A deliberate deviation from the
        // reference (it has no such check): two subsystems owning the same block
        // are silently *summed* by the BCOO->BSR segmentation reduction, and a
        // self pair writes a block the diag subsystem already owns.
        std::vector<U64> off_diag_uid_l(off_diag_subsystem_view.size());
        std::vector<U64> off_diag_uid_r(off_diag_subsystem_view.size());
        for(auto&& [i, off_diag_subsystem] : enumerate(off_diag_subsystem_view))
        {
            auto uid           = off_diag_subsystem->uid();
            off_diag_uid_l[i] = std::get<0>(uid);
            off_diag_uid_r[i] = std::get<1>(uid);
        }
        auto pair_check = off_diag_layout_detail::check_unique_pairs(
            std::span<const U64>{off_diag_uid_l}, std::span<const U64>{off_diag_uid_r});

        // the diagnostic is built only on the failing path (UIPC_ASSERT formats
        // lazily, but the subsystem names have to be resolved here)
        std::string pair_error;
        if(!pair_check.ok)
        {
            auto& first  = *off_diag_subsystem_view[pair_check.first];
            auto& second = *off_diag_subsystem_view[pair_check.second];
            pair_error   = pair_check.self
                               ? fmt::format("Off-diag linear subsystem {} connects the diagonal "
                                             "subsystem {} to itself; that diag block is already "
                                             "owned by the diag subsystem.",
                                             first.name(),
                                             first.name())
                               : fmt::format("Off-diag linear subsystems {} and {} connect the same "
                                             "(left, right) pair of diagonal subsystems; their "
                                             "blocks would overlap in the global matrix.",
                                             first.name(),
                                             second.name());
        }
        UIPC_ASSERT(pair_check.ok, "{}", pair_error);
    }

    auto total_count = diag_subsystem_view.size() + off_diag_subsystem_view.size();
    subsystem_infos.resize(total_count);

    // Diag System Always Go First
    auto diag_span = std::span{subsystem_infos}.subspan(0, diag_subsystem_view.size());
    // Off Diag System Always After Diag System
    auto off_diag_span = std::span{subsystem_infos}.subspan(diag_subsystem_view.size(),
                                                            off_diag_subsystem_view.size());
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
            // NOTE: an off-diag subsystem has no `m_index` member (the reference's
            // header has none either) -- its index lives *only* here, in
            // `subsystem_infos[i].index`, and is passed to it through
            // `OffDiagInfo::m_index` at assembly time.
        }

        for(auto diag_subsystem : diag_subsystem_view)
            diag_subsystem->init();

        for(auto off_diag_subsystem : off_diag_subsystem_view)
            off_diag_subsystem->init();
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
    // sized from the off-diag count *before* `_update_subsystem_extent()` writes
    // `off_diag_lr_triplet_counts[local_index]` (an out-of-bounds write otherwise)
    off_diag_lr_triplet_counts.resize(off_diag_subsystem_view.size());

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
        // the reference's `triplet_count_dev = (IndexT)bcoo_A.triplet_count();`
        // (global_linear_system.cu:237): `spmv_dot` reads the count *on the device*
        // so the launch shape does not depend on it.
        triplet_count_dev.copy_from(static_cast<IndexT>(bcoo_A.triplet_count()));
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
    auto off_diag_subsystem_view   = off_diag_subsystems.view();
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
            auto  triplet_i = subsystem_info.index;
            auto& off_diag_subsystem = off_diag_subsystem_view[subsystem_info.local_index];
            OffDiagExtentInfo info;
            off_diag_subsystem->report_extent(info);

            auto total_block_count = info.m_lr_block_count + info.m_rl_block_count;

            triplet_count_changed |= subsystem_triplet_counts[triplet_i] != total_block_count;
            subsystem_triplet_counts[triplet_i] = total_block_count;
            off_diag_lr_triplet_counts[subsystem_info.local_index] =
                SizeT2{info.m_lr_block_count, info.m_rl_block_count};
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

    auto diag_subsystem_view     = diag_subsystems.view();
    auto off_diag_subsystem_view = off_diag_subsystems.view();

    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    auto subsystem_triplet_counts  = subsystem_triplet_offsets_counts.counts();
    auto subsystem_triplet_offsets = subsystem_triplet_offsets_counts.offsets();

    for(const auto& subsystem_info : subsystem_infos)
    {
        if(!subsystem_info.is_diag)
        {
            auto  triplet_i          = subsystem_info.index;
            auto  local_index        = subsystem_info.local_index;
            auto& off_diag_subsystem = off_diag_subsystem_view[local_index];
            // the two ends' *diag* indices (assigned by `Impl::init()` above)
            auto l_diag_index = off_diag_subsystem->m_l->m_index;
            auto r_diag_index = off_diag_subsystem->m_r->m_index;

            auto l_blocked_dof_offset = diag_dof_offsets[l_diag_index] / DoFBlockSize;
            auto l_blocked_dof_count  = diag_dof_counts[l_diag_index] / DoFBlockSize;
            auto r_blocked_dof_offset = diag_dof_offsets[r_diag_index] / DoFBlockSize;
            auto r_blocked_dof_count  = diag_dof_counts[r_diag_index] / DoFBlockSize;

            // the (lr, rl) split of this subsystem's triplet slice: `lr` first,
            // `rl` immediately after (that is how `_update_subsystem_extent()`
            // budgets it, so the two windows tile the slice exactly)
            auto lr_triplet_offset = subsystem_triplet_offsets[triplet_i];
            auto lr_triplet_count  = off_diag_lr_triplet_counts[local_index].x;
            auto rl_triplet_offset = lr_triplet_offset + lr_triplet_count;
            auto rl_triplet_count  = off_diag_lr_triplet_counts[local_index].y;

            OffDiagInfo info{this};
            info.m_index = triplet_i;

            // an off-diag subsystem writes both sides from one stiffness
            // expression (plan 14 §3.2): `A(l, r)` at (l_off, r_off) and
            // `A(r, l) = A(l, r)^T` at (r_off, l_off).  Writing only `lr` leaves
            // A non-symmetric -- and ge2sym would silently drop it whenever the
            // pair is written in the *other* order (cookbook §7.39).
            info.m_lr_hessian =
                HA.subview(lr_triplet_offset, lr_triplet_count)
                    .submatrix(lla::index_pair<std::int32_t>{
                                   static_cast<std::int32_t>(l_blocked_dof_offset),
                                   static_cast<std::int32_t>(r_blocked_dof_offset)},
                               lla::index_pair<std::int32_t>{
                                   static_cast<std::int32_t>(l_blocked_dof_count),
                                   static_cast<std::int32_t>(r_blocked_dof_count)});

            info.m_rl_hessian =
                HA.subview(rl_triplet_offset, rl_triplet_count)
                    .submatrix(lla::index_pair<std::int32_t>{
                                   static_cast<std::int32_t>(r_blocked_dof_offset),
                                   static_cast<std::int32_t>(l_blocked_dof_offset)},
                               lla::index_pair<std::int32_t>{
                                   static_cast<std::int32_t>(r_blocked_dof_count),
                                   static_cast<std::int32_t>(l_blocked_dof_count)});

            off_diag_subsystem->assemble(info);
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
    // The reference's order (global_linear_system.cu:422-435): the *global*
    // preconditioner is assembled FIRST, then the local fan-out runs
    // unconditionally.  Nothing is allocated here and `reserve_ratio` is not used
    // (`_update_subsystem_extent()` owns it, .cu:305/:322): a concrete global
    // preconditioner sizes its own storage in `do_assemble()`.
    if(global_preconditioner)
    {
        GlobalPreconditionerAssemblyInfo info{this};
        global_preconditioner->assemble(info);
    }

    for(auto&& preconditioner : local_preconditioners.view())
    {
        LocalPreconditionerAssemblyInfo info{this, preconditioner->m_subsystem->m_index};
        preconditioner->assemble(info);
    }
}

void GlobalLinearSystem::Impl::solve_linear_system()
{
    Timer timer{"Solve Linear System"};

    if(iterative_solver)
    {
        SolvingInfo info{this};
        info.m_b = b.cview();
        info.m_x = x.view();
        iterative_solver->solve(info);

        last_solve_iterations = info.m_iter_count;
        logger::info("Iterative linear solver iteration count: {}", info.m_iter_count);
    }
}

void GlobalLinearSystem::Impl::apply_preconditioner(lla::LDenseVectorView<Float>  z,
                                                    lla::LCDenseVectorView<Float> r,
                                                    lud::LCBufferView<IndexT>     converged,
                                                    lud::LStream*                 stream)
{
    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    // The reference's precedence (global_linear_system.cu:466-510), transcribed:
    //  1) a registered *global* preconditioner is applied FIRST, to the whole
    //     z/r, in place;
    //  2) the local fan-out then runs UNCONDITIONALLY and overwrites its own dof
    //     windows (it is *not* an alternative to the global one -- applying only
    //     one of them yields a differently preconditioned, silently different
    //     iteration);
    //  3) `!global_preconditioner` gates ONLY the `z <- r` fallback below, i.e. a
    //     registered global preconditioner is responsible for every dof it does
    //     not leave to a local one.
    if(global_preconditioner)
    {
        ApplyPreconditionerInfo info{this};
        info.m_z         = z;
        info.m_r         = r;
        info.m_converged = converged;
        info.m_stream    = stream;
        global_preconditioner->apply(info);
    }

    for(auto& preconditioner : local_preconditioners.view())
    {
        ApplyPreconditionerInfo info{this};
        auto                    index  = preconditioner->m_subsystem->m_index;
        auto                    offset = diag_dof_offsets[index];
        auto                    count  = diag_dof_counts[index];
        info.m_z                       = z.subview(offset, count);
        info.m_r                       = r.subview(offset, count);
        info.m_converged               = converged;
        info.m_stream                  = stream;
        preconditioner->apply(info);
    }

    if(!global_preconditioner)
    {
        // For diag subsystems without local preconditioner, just copy r to z
        for(auto i : no_precond_diag_subsystem_indices)
        {
            auto offset = diag_dof_offsets[i];
            auto count  = diag_dof_counts[i];
            auto z_sub  = z.subview(offset, count);
            auto r_sub  = r.subview(offset, count);
            // the reference's `cuda_tool::BufferLaunch(stream).copy(z_sub.buffer_view(),
            // r_sub.buffer_view())`: a device->device copy of the two *windowed*
            // views (each LBufferView carries its own offset, so the subview windows
            // are copied, not the whole buffers).
            z_sub.buffer_view().copy_from(r_sub.buffer_view(), stream);
        }
    }
}

void GlobalLinearSystem::Impl::spmv(Float                        a,
                                    lla::LCDenseVectorView<Float> x,
                                    Float                         b,
                                    lla::LDenseVectorView<Float>  y)
{
    spmver.rbk_sym_spmv(a, bcoo_A.cview(), x, b, y);
    // Just some debug options
    //  * spmver.sym_spmv(a, bcoo_A.cview(), x, b, y); // Slightly slower
    //  * spmver.cpu_sym_spmv(a, bcoo_A.cview(), x, b, y);  // Much slower
}

void GlobalLinearSystem::Impl::spmv_dot(lla::LCDenseVectorView<Float> x,
                                        lla::LDenseVectorView<Float>  y,
                                        lud::LBufferView<Float>       d_dot,
                                        lud::LStream*                 stream)
{
    // The reference's `cudaStream_t stream = nullptr` means "the legacy default
    // stream"; T3.1's primitives default to `default_blas_stream()` == the frame
    // command list, which is what `nullptr` maps onto here.
    auto& s = stream ? *stream : lla::default_blas_stream();
    spmver.rbk_sym_spmv_dot(1.0,
                            bcoo_A.cview(),
                            x,
                            0.0,
                            y,
                            d_dot,
                            triplet_count_dev.cview(),
                            bcoo_A.triplet_capacity(),
                            s);
}

std::array<const void*, 3> GlobalLinearSystem::Impl::matrix_data_ptrs() const
{
    // DEVIATION (T8.3a-2, reported): the reference returns
    //   {v.row_indices().data(), v.col_indices().data(), v.values().data()}
    // — three raw device pointers, used only as the FusedPCG CUDA-graph validity
    // key.  Neither exists here: a buffer DSL exposes no element pointer (cookbook
    // §7.9), and there is no graph to key (plan §3.3(3)).  The returned triple is an
    // identity token over the *containers*; nothing may depend on its value.
    return {static_cast<const void*>(&bcoo_A),
            static_cast<const void*>(&triplet_count_dev),
            nullptr};
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
    m_impl.off_diag_subsystems.register_sim_system(*subsystem);
}

void GlobalLinearSystem::add_solver(IterativeSolver* solver)
{
    check_state(SimEngineState::BuildSystems, "add_solver()");
    UIPC_ASSERT(solver != nullptr, "The solver should not be nullptr.");
    m_impl.iterative_solver.register_sim_system(*solver);
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
    // The reference's message (global_linear_system.cu:678-683) is the same text as
    // the local overload's; the port matches it word for word.
    UIPC_ASSERT(preconditioner != nullptr, "The preconditioner should not be nullptr.");
    // One optional slot, not a collection: the reference allows at most one global
    // preconditioner, and a second registration silently overwrites the first
    // (`SimSystemSlot::register_sim_system()`, src/backends/common/details/
    // sim_system_slot.inl:7-13).  Deliberately no assert here -- see the file-header
    // note in global_preconditioner.cpp (research §7 R3).
    m_impl.global_preconditioner.register_sim_system(*preconditioner);
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
// SCOPE NOTE (T8.3 → plan 11) — what is still deliberately undefined here.
//
// Declared in global_linear_system.h and not defined (nothing ODR-uses them,
// which is exactly what keeps the interface complete without a stub):
//
//   void  GlobalLinearSystem::solve();
//   void  GlobalLinearSystem::_dump_A_b();
//   void  GlobalLinearSystem::_dump_x();
//   Float GlobalLinearSystem::diag_norm();   // only called by SimEngine::do_advance
//   Float GlobalLinearSystem::mass_norm();   // only called by SimEngine::do_advance
//
// T8.3a-2 landed the solver substrate those need (`Impl::spmv`, `Impl::spmv_dot`,
// `matrix_data_ptrs`, `Impl::solve_linear_system`, the `triplet_count_dev` upload
// and a real `add_solver()`), together with `IterativeSolver` / `LinearPCG`.
//
// RESOLVED BY T8.3a-1: `add_subsystem(OffDiagLinearSubsystem*)` is a real
// registration, together with the three off-diag branches inside `Impl` (see the
// file header).
//
// RESOLVED BY T8.3a-3: `add_preconditioner(GlobalPreconditioner*)` is a real
// registration, and `_assemble_preconditioner()` / `apply_preconditioner()` have the
// reference's global branch.  Nothing in this file is now "declared but undefined"
// because of plan 13; the five above are plan 11's (the engine wiring + the dumps).
// ===========================================================================
}  // namespace uipc::backend::luisa
