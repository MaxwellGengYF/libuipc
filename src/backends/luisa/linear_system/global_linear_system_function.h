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
 *
 * T8.3a-1 adds `off_diag_layout_detail::check_unique_pairs()` next to it, for the
 * same reason: the reference has no duplicate/self off-diag rejection at all, so
 * the port's version is factored into a *predicate* that `Impl::init()` turns into
 * a `UIPC_ASSERT` and that a test can drive directly (an assert aborts the test
 * process -- cookbook 7.22 "harness hazards").
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

namespace off_diag_layout_detail
{
/**
 * @brief The verdict of `check_unique_pairs()`.
 *
 * `first`/`second` are indices into the caller's arrays: the two colliding
 * entries for a duplicate pair, `(first, first)` for a self pair.
 */
struct PairCheck
{
    bool  ok     = true;
    SizeT first  = ~SizeT{0};
    SizeT second = ~SizeT{0};
    /// the offending entry couples a diagonal block to itself (`uid_l == uid_r`)
    bool self = false;
};

/**
 * @brief Reject two off-diag subsystems owning the same block, and an off-diag
 * subsystem coupling a diagonal block to itself.
 *
 * A DELIBERATE DEVIATION (T8.3a-1, research 4.4): the reference has no such
 * check.  Two off-diag subsystems registered on the same `(l, r)` pair write the
 * same block coordinates of `triplet_A`, and the BCOO conversion's
 * segmentation-reduction then *silently sums* them -- a wrong matrix with no
 * diagnostic.  Likewise `l == r` writes a block that the diagonal subsystem
 * already owns.
 *
 * This is a *predicate* (no assert, no throw except the size-mismatch guard) on
 * purpose: `UIPC_ASSERT` aborts the process, so `GlobalLinearSystem::Impl::init()`
 * is the only caller that turns it into a diagnostic, and a unit test can drive
 * every verdict here without killing the test binary.
 */
[[nodiscard]] inline PairCheck check_unique_pairs(std::span<const U64> uid_l,
                                                  std::span<const U64> uid_r)
{
    if(uid_l.size() != uid_r.size())
    {
        throw std::invalid_argument{"check_unique_pairs: uid_l/uid_r size mismatch"};
    }

    PairCheck result;

    for(SizeT i = 0; i < uid_l.size(); ++i)
    {
        // (b) a self pair: the two ends are the same diagonal block
        if(uid_l[i] == uid_r[i])
        {
            result.ok     = false;
            result.first  = i;
            result.second = i;
            result.self   = true;
            return result;
        }

        // (a) a duplicate pair: two subsystems own the same block
        for(SizeT j = i + 1; j < uid_l.size(); ++j)
        {
            if(uid_l[i] == uid_l[j] && uid_r[i] == uid_r[j])
            {
                result.ok     = false;
                result.first  = i;
                result.second = j;
                result.self   = false;
                return result;
            }
        }
    }

    return result;
}
}  // namespace off_diag_layout_detail
}  // namespace uipc::backend::luisa
