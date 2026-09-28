#pragma once
/********************************************************************
 * @file builtins.h
 * @brief Warp/bit intrinsic shims (plan risk R9).
 *
 * LuisaCompute exposes *higher level* warp primitives (warp_active_sum,
 * warp_prefix_sum, warp_read_lane, warp_active_bit_mask, ...) but no raw
 * `__shfl_*_sync` / `__ballot_sync`. The reference backend uses the raw CUDA
 * intrinsics heavily, so this header maps the CUDA spelling onto the LC
 * primitives, giving the ported kernels the exact call shape they had before.
 *
 * Mapping:
 *   __shfl_down_sync(m, v, d) -> warp_read_lane(v, lane + d)
 *   __shfl_up_sync(m, v, d)   -> warp_read_lane(v, lane - d)
 *   __shfl_xor_sync(m, v, x)  -> warp_read_lane(v, lane ^ x)
 *   __shfl_sync(m, v, src)    -> warp_read_lane(v, src)
 *   __ballot_sync(m, p)       -> warp_active_bit_mask(p)
 *   __any_sync / __all_sync   -> warp_active_any / warp_active_all
 *   __popc / __clz / __ctz    -> popcount / clz / ctz
 *   __ffs(x)                  -> ctz(x) + 1
 *   __brev(x)                 -> reverse_bits (emulated)
 *   __syncthreads()           -> sync_block()
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

#include <luisa_device/atomic.h>

namespace uipc::backend::luisa::luisa_device
{
namespace warp
{
using lcrt::def;
using lcrt::Expr;

// Lane position of the current thread inside its warp.
[[nodiscard]] inline auto lane_id() noexcept
{
    return lcrt::warp_lane_id();
}

[[nodiscard]] inline auto lane_count() noexcept
{
    return lcrt::warp_lane_count();
}

// Equivalent of __shfl_down_sync(mask, v, delta).
template <typename V, typename D>
[[nodiscard]] inline auto shfl_down(V&& v, D&& delta) noexcept
{
    return lcrt::warp_read_lane(std::forward<V>(v), lane_id() + lcrt::def(std::forward<D>(delta)));
}

// Equivalent of __shfl_up_sync(mask, v, delta).
template <typename V, typename D>
[[nodiscard]] inline auto shfl_up(V&& v, D&& delta) noexcept
{
    return lcrt::warp_read_lane(std::forward<V>(v), lane_id() - lcrt::def(std::forward<D>(delta)));
}

// Equivalent of __shfl_xor_sync(mask, v, lane_mask).
template <typename V, typename D>
[[nodiscard]] inline auto shfl_xor(V&& v, D&& lane_mask) noexcept
{
    return lcrt::warp_read_lane(std::forward<V>(v), lane_id() ^ lcrt::def(std::forward<D>(lane_mask)));
}

// Equivalent of __shfl_sync(mask, v, src).
template <typename V, typename I>
[[nodiscard]] inline auto shfl(V&& v, I&& src) noexcept
{
    return lcrt::warp_read_lane(std::forward<V>(v), lcrt::def(std::forward<I>(src)));
}

// ---- high level reductions (prefer these over hand-rolled shuffles) -------
template <typename V>
[[nodiscard]] inline auto reduce_sum(V&& v) noexcept
{
    return lcrt::warp_active_sum(std::forward<V>(v));
}
template <typename V>
[[nodiscard]] inline auto reduce_min(V&& v) noexcept
{
    return lcrt::warp_active_min(std::forward<V>(v));
}
template <typename V>
[[nodiscard]] inline auto reduce_max(V&& v) noexcept
{
    return lcrt::warp_active_max(std::forward<V>(v));
}
template <typename V>
[[nodiscard]] inline auto prefix_sum(V&& v) noexcept
{
    return lcrt::warp_prefix_sum(std::forward<V>(v));
}

// ---- ballot / vote --------------------------------------------------------
[[nodiscard]] inline auto ballot(lcrt::Expr<bool> pred) noexcept
{
    return lcrt::warp_active_bit_mask(pred);
}
[[nodiscard]] inline auto any(lcrt::Expr<bool> pred) noexcept
{
    return lcrt::warp_active_any(pred);
}
[[nodiscard]] inline auto all(lcrt::Expr<bool> pred) noexcept
{
    return lcrt::warp_active_all(pred);
}
[[nodiscard]] inline auto lane_mask() noexcept
{
    // All lanes of the (full) warp active: 0xFFFFFFFF.
    return lcrt::warp_active_bit_mask(true);
}
}// namespace warp

namespace bit
{
// __popc
template <typename X>
[[nodiscard]] inline auto popc(X&& x) noexcept
{
    return lcrt::popcount(std::forward<X>(x));
}
// __clz
template <typename X>
[[nodiscard]] inline auto clz(X&& x) noexcept
{
    return lcrt::clz(std::forward<X>(x));
}
// __ffs(x) == ctz(x) + 1 (0 for x == 0)
template <typename X>
[[nodiscard]] inline auto ffs(X&& x) noexcept
{
    auto v = lcrt::def(std::forward<X>(x));
    return lcrt::select(0u, lcrt::ctz(v) + 1u, v != 0u);
}
// __brev: bit reverse (emulated; no LC primitive exists).
template <typename X>
[[nodiscard]] inline auto brev(X&& x) noexcept
{
    auto v0 = lcrt::def(std::forward<X>(x));
    auto v1 = ((v0 & 0x55555555u) << 1u) | ((v0 >> 1u) & 0x55555555u);
    auto v2 = ((v1 & 0x33333333u) << 2u) | ((v1 >> 2u) & 0x33333333u);
    auto v3 = ((v2 & 0x0F0F0F0Fu) << 4u) | ((v2 >> 4u) & 0x0F0F0F0Fu);
    auto v4 = ((v3 & 0x00FF00FFu) << 8u) | ((v3 >> 8u) & 0x00FF00FFu);
    return (v4 << 16u) | (v4 >> 16u);
}
}// namespace bit

// __syncthreads()
inline void syncthreads() noexcept
{
    lcrt::sync_block();
}

// Bit reinterpret casts (__float_as_int / __int_as_float / ...).
template <typename To, typename From>
[[nodiscard]] inline auto bit_cast(From&& x) noexcept
{
    return lcrt::as<To>(std::forward<From>(x));
}
}// namespace uipc::backend::luisa::luisa_device
