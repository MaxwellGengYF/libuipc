#pragma once
/********************************************************************
 * @file global_linear_system_function.h
 * @brief T5.4a — the host-callable, test-reachable part of
 *        `GlobalLinearSystem`'s layout half (port of the ordering/dof-layout
 *        arithmetic of src/backends/cuda/linear_system/global_linear_system.cu
 *        `Impl::init()`, step 1 + 2).
 *
 * WHY THIS HEADER EXISTS (the T5.4a brief's §2)
 * --------------------------------------------
 * In the reference, `GlobalLinearSystem::Impl::init()` is the *only* place that
 * decides the subsystem order (ascending uid — issue #271) and hands out the dof
 * offsets/counts, and nothing but `SimEngine::do_init()` ever calls it.  The
 * T5.4a port deliberately does not wire the engine (the solve path is T8.3), so
 * the arithmetic is factored here as a plain host routine over
 * `(uid, dof_count)` arrays: the shipped `Impl::init()` calls it *and* a unit
 * test can drive it directly against hand-derived offsets (cookbook §7.19).
 *
 * The routines are host-only on purpose: the reference's ordering/scan run on
 * host vectors (`OffsetCountCollection` is host-only in both backends).
 *********************************************************************/
#include <type_define.h>
#include <uipc/common/vector.h>

#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace uipc::backend::luisa
{
namespace global_linear_system_detail
{
/**
 * @brief The diag-subsystem dof layout: ascending-uid order + exclusive-scan
 * offsets.  Mirrors the reference's `diag_dof_offsets_counts` after
 * `Impl::init()`'s steps 1 and 2.
 */
struct DofLayout
{
    /// subsystem uids in ascending order == the layout order (`order[i]` is the
    /// i-th subsystem of the sorted collection)
    std::vector<U64> uids;
    /// dof counts, in the same order (the reference's `diag_dof_offsets_counts.counts()`)
    std::vector<SizeT> counts;
    /// exclusive-scan offsets, in the same order (`...::offsets()`)
    std::vector<SizeT> offsets;

    [[nodiscard]] SizeT total_count() const noexcept
    {
        return offsets.empty() ? SizeT{0} : offsets.back() + counts.back();
    }
};

/**
 * @brief The reference's `std::ranges::sort(view, [](a, b){ return a->uid() < b->uid(); })`
 * followed by the exclusive scan of the dof counts.
 *
 * `uids` and `dof_counts` are parallel arrays of the *unsorted* subsystem list.
 * The port sorts with `std::stable_sort` (the reference's `std::ranges::sort` is
 * unstable, which is unobservable: `DiagLinearSubsystem::uid()` values are unique
 * — the registry hands a `(dim, uid)` geometry group to the first system with that
 * uid, and uids are the systems' identity).
 */
[[nodiscard]] inline DofLayout plan_dof_layout(std::span<const U64>   uids,
                                               std::span<const SizeT> dof_counts)
{
    if(uids.size() != dof_counts.size())
    {
        throw std::invalid_argument{"plan_dof_layout: uid/count size mismatch"};
    }

    DofLayout layout;
    layout.uids.resize(uids.size());
    layout.counts.resize(uids.size());
    layout.offsets.resize(uids.size());

    std::vector<SizeT> order(uids.size());
    std::iota(order.begin(), order.end(), SizeT{0});
    std::stable_sort(order.begin(),
                     order.end(),
                     [&](SizeT a, SizeT b) noexcept { return uids[a] < uids[b]; });

    SizeT offset = 0;
    for(SizeT i = 0; i < order.size(); ++i)
    {
        layout.uids[i]    = uids[order[i]];
        layout.counts[i]  = dof_counts[order[i]];
        layout.offsets[i] = offset;
        offset += layout.counts[i];
    }
    return layout;
}
}  // namespace global_linear_system_detail
}  // namespace uipc::backend::luisa
