#pragma once
/********************************************************************
 * @file aabb_reduce.h
 * @brief The AABB *union* reduction — the port's stand-in for the reference's
 *        `DeviceReduce(...).Reduce(in, out, n, merge, default_aabb)`.
 *
 * Where it is used in T7.1a:
 *   - `LinearBVH::build` step 1 ("get max aabb"): the union box of all input AABBs,
 *     which is what normalizes the Morton codes;
 *   - `StacklessBVH::Impl::calcMaxBVFromBox`: the scene box (the reference does it
 *     with a warp/block shuffle tree + `atomicMinf`/`atomicMaxf`).
 *
 * WHY IT IS NOT A `linalg` PRIMITIVE
 * ----------------------------------
 * `linalg::device_reduce` (the ported CUB `DeviceReduce`) supports only
 * sum/min/max over *scalars*: a custom merge functor (the reference passes
 * `[](const AABB& a, const AABB& b) { return a.merged(b); }`) does not fit its
 * `Plus`/`Min`/`Max` op list, and an `LAABB` buffer element is a registered *struct*,
 * so its components are not addressable as `AtomicRef<float>` (LC's struct `AtomicRef`
 * reaches the members through `LUISA_STRUCT`'s per-member `AtomicRef<member_type>`,
 * which for `LVec<float,3>` would need an `AtomicRef<std::array<float,3>>` that does
 * not exist — that is why the reference's `atomicMinf(&box.min().x(), ...)` cannot be
 * transcribed either).  The two-pass shape below is the same one
 * `linalg::device_reduce` uses (per-block tree reduction, then one block that folds
 * the partials), so it is deterministic and its result is exact: component-wise
 * min/max is associative and commutative for floats, and the empty box
 * (`min = +FLT_MAX`, `max = -FLT_MAX`) is its identity, so the *value* is identical to
 * the reference's CUB reduction whatever the association order.
 *
 * The block reduction is the reference's own shape: a `__shfl_down_sync` tree over
 * the 6 box scalars, one wrapped in a bit-cast per-warp `__shared__` staging array,
 * then a second shuffle tree over the per-warp results (the stackless scene-box
 * kernel's algorithm), generalised to the whole-buffer reduce.
 *********************************************************************/
#include <type_define.h>

#include <collision_detection/aabb.h>
#include <luisa_device/buffer.h>
#include <luisa_device/stream.h>
#include <luisa_device/view.h>

#include <cstddef>
#include <cstdint>

namespace uipc::backend::luisa::detail
{
/// The box reduction's device block size (a multiple of 32 — LC aborts otherwise).
inline constexpr std::uint32_t kAabbReduceBlock = 256u;

/// Number of `LAABB` scratch entries `reduce_aabb_union` needs for `n` inputs.
[[nodiscard]] inline constexpr std::size_t aabb_reduce_scratch_size(std::size_t n) noexcept
{
    return (n + kAabbReduceBlock - 1u) / kAabbReduceBlock;
}

/**
 * @brief `out[0] = union(in[0 .. n-1])`.
 *
 * `n == 0` writes the empty box (CUB's `Reduce` semantics with an empty range: the
 * initial value is written).
 *
 * @param partials caller-owned scratch of `aabb_reduce_scratch_size(n)` elements
 *        (`LDeviceBuffer`, so it survives a `resize` of the caller's other buffers).
 */
void reduce_aabb_union(lud::LStream&            stream,
                       lud::LCBufferView<LAABB> in,
                       lud::LBufferView<LAABB>  out,
                       std::size_t              n,
                       lud::LDeviceBuffer<LAABB>& partials);
}// namespace uipc::backend::luisa::detail
