/********************************************************************
 * @file stackless_bvh.inl
 * @brief StacklessBVH — the parts that must be visible to a kernel body (port of
 *        src/backends/cuda/collision_detection/details/stackless_bvh.inl).
 *
 * Included at the end of stackless_bvh.h.  Contents:
 *   1. `namespace culbvh` — the reference's `uipc::culbvh` device helpers (the
 *      constants, `expandBits`, `morton3D`, the pair helpers and `safe_copy_to`);
 *   2. the two traversal kernels (`StacklessCDSharedSelf` / `SharedOther`), templated
 *      on the record-time predicate, and the `detect`/`query` bodies that dispatch them.
 *
 * The build kernels and the host-side step list live in stackless_bvh.cpp.
 *********************************************************************/
#pragma once

#include <luisa_device/builtins.h>

#include <frame_stream.h>
#include <uipc/common/log.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace uipc::backend::luisa::culbvh
{
using uipc::backend::luisa::LAABB;
using uipc::backend::luisa::LVector2i;
using uipc::backend::luisa::StacklessBVH;
using uipc::Vector2i;

using uint   = std::uint32_t;
using ullint = unsigned long long int;

constexpr int K_THREADS = 256;
constexpr int K_WARPS   = K_THREADS >> 5;

constexpr int K_REDUCTION_LAYER  = 5;
constexpr int K_REDUCTION_NUM    = 1 << K_REDUCTION_LAYER;
constexpr int K_REDUCTION_MODULO = K_REDUCTION_NUM - 1;

// The reference declares these quantisation constants but never uses them (the
// `intAABB`/`lessThan`/`PlainAABB` helpers that would have used them are dead code in
// the reference: the quantised traversal never shipped).  They are kept so that a later
// batch (info_stackless_bvh / info_stackless_bvh_v0) finds them where the reference has
// them.
constexpr int    aabbBits  = 15;
constexpr int    aabbRes   = (1 << aabbBits) - 2;
constexpr int    indexBits = 64 - 3 * aabbBits;
constexpr int    offset3   = aabbBits * 3;
constexpr int    offset2   = aabbBits * 2;
constexpr int    offset1   = aabbBits * 1;
constexpr ullint indexMask = 0xFFFFFFFFFFFFFFFFu << offset3;
constexpr uint   aabbMask  = 0xFFFFFFFFu >> (32 - aabbBits);
constexpr uint   MaxIndex  = 0xFFFFFFFFFFFFFFFFu >> offset3;

constexpr uint MAX_CD_NUM_PER_VERT = 64;
constexpr int  MAX_RES_PER_BLOCK   = 1024;

/// `(uint)x` for a host or a DSL float (`static_cast` / `lcrt::cast`).
template <typename T>
[[nodiscard]] inline auto cast_to_uint(const T& x) noexcept
{
    if constexpr(lps::is_dsl_value_v<T>)
        return lcrt::cast<std::uint32_t>(x);
    else
        return static_cast<std::uint32_t>(x);
}

/// The reference's `__mm_min` / `__mm_max` (`a > b ? b : a`) — `fminf`/`fmaxf` on the
/// device, which is what the reference's helpers compile to for the ±0/NaN cases that
/// reach them.  `lps::min`/`lps::max` are the port's spelling.
template <typename T>
[[nodiscard]] inline auto mm_min(T a, T b) noexcept
{
    return lps::min(a, b);
}
template <typename T>
[[nodiscard]] inline auto mm_max(T a, T b) noexcept
{
    return lps::max(a, b);
}

/// `expandBits`: a 10-bit integer expanded to 30 bits (2 zeros after each bit).
///
/// The reference has this arithmetic **twice** (linear_bvh's `detail::expand_bits` and
/// this `culbvh::expandBits`, two independently vendored implementations); the port
/// keeps both spellings so that neither file depends on the other's internals.
template <typename V>
[[nodiscard]] inline auto expandBits(V&& v) noexcept
{
    if constexpr(lps::is_dsl_value_v<V>)
    {
        // LC's Expr is single-assignment: the running value is a Var local.
        lcrt::Var<std::uint32_t> x = std::forward<V>(v);
        x                          = (x * 0x00010001u) & 0xFF0000FFu;
        x                          = (x * 0x00000101u) & 0x0F00F00Fu;
        x                          = (x * 0x00000011u) & 0xC30C30C3u;
        x                          = (x * 0x00000005u) & 0x49249249u;
        return x;
    }
    else
    {
        std::uint32_t x = static_cast<std::uint32_t>(std::forward<V>(v));
        x               = (x * 0x00010001u) & 0xFF0000FFu;
        x               = (x * 0x00000101u) & 0x0F00F00Fu;
        x               = (x * 0x00000011u) & 0xC30C30C3u;
        x               = (x * 0x00000005u) & 0x49249249u;
        return x;
    }
}

/// The reference's `int2{a, b}` (the port's pair element is `LVector2i`).
template <typename A, typename B>
[[nodiscard]] inline lcrt::Var<LVector2i> make_pair_var(const A& a, const B& b) noexcept
{
    lps::LVec<lcrt::Var<IndexT>, 2> p;
    p(0) = lcrt::cast<IndexT>(a);
    p(1) = lcrt::cast<IndexT>(b);
    lcrt::Var<LVector2i> v;
    store_vec2i(v, p);
    return v;
}

/// The reference's `make_ordered_pair(int a, int b)`: `a < b ? {a, b} : {b, a}`.
template <typename A, typename B>
[[nodiscard]] inline lcrt::Var<LVector2i> make_ordered_pair(const A& a, const B& b) noexcept
{
    if constexpr(lps::is_dsl_value_v<A> || lps::is_dsl_value_v<B>)
    {
        auto ai = lcrt::cast<IndexT>(a);
        auto bi = lcrt::cast<IndexT>(b);
        // `select(false_value, true_value, pred)`
        auto lo = lcrt::select(bi, ai, ai < bi);  // a < b ? a : b
        auto hi = lcrt::select(ai, bi, ai < bi);  // a < b ? b : a
        return make_pair_var(lo, hi);
    }
    else
    {
        auto ai = static_cast<IndexT>(a);
        auto bi = static_cast<IndexT>(b);
        return make_pair_var(a < b ? ai : bi, a < b ? bi : ai);
    }
}

/**
 * @brief `morton3D`: a 30-bit Morton code for a point in the unit cube.
 *
 * NOTE the *arithmetic*: the reference clamps `x * 1024.0f` (a **float** multiply) into
 * `[0, 1023]` and truncates to `uint`.  `linear_bvh`'s `detail::morton_code` instead
 * clamps the coordinate into `[0, 1]` and multiplies by the **double** literal `1024.0`
 * (a double multiply before the truncation).  The two differ whenever `x * 1024.0f` is
 * not exactly representable; the port transcribes each as it is.
 */
template <typename T>
[[nodiscard]] inline auto morton3D(const T& x, const T& y, const T& z) noexcept
{
    auto cx = lps::min(lps::max(x * 1024.0f, 0.0f), 1023.0f);
    auto cy = lps::min(lps::max(y * 1024.0f, 0.0f), 1023.0f);
    auto cz = lps::min(lps::max(z * 1024.0f, 0.0f), 1023.0f);
    auto xx = expandBits(cast_to_uint(cx));
    auto yy = expandBits(cast_to_uint(cy));
    auto zz = expandBits(cast_to_uint(cz));
    return (xx * 4u + yy * 2u + zz);
}

/**
 * @brief `SafeCopyTo`: flush `total_res` shared results to the global pair array.
 *
 * The reference's early `return` guards become `$if`-guards (the helper has no barrier,
 * but a whole-body `$if` is the port's standard shape); `blockDim.x` is the
 * compile-time `K_THREADS` (the port always sets that block size); `res.total_size()`
 * is the pair *capacity*, which the kernel reads back from the buffer handle
 * (`BufferVar::size()`).
 */
inline void safe_copy_to(const lcrt::Shared<LVector2i>&     shared_res,
                         const lcrt::Var<std::int32_t>&    total_res,
                         const lcrt::BufferVar<LVector2i>& global_res,
                         const lcrt::Var<std::int32_t>&    global_idx,
                         const lcrt::Var<std::int32_t>&    max_res) noexcept
{
    $if((global_idx < max_res) & (total_res > 0))
    {
        const auto copy_count = lcrt::min(total_res, max_res - global_idx);
        // Copy full blocks
        const auto full_blocks = (copy_count - 1) / std::int32_t(K_THREADS);
        $for(i, 0, full_blocks)
        {
            const auto offset =
                i * std::int32_t(K_THREADS) + lcrt::cast<std::int32_t>(lcrt::thread_x());
            global_res->write(global_idx + offset, shared_res[offset]);
        };
        // Copy the rest
        const auto offset =
            full_blocks * std::int32_t(K_THREADS) + lcrt::cast<std::int32_t>(lcrt::thread_x());
        $if(offset < copy_count)
        {
            global_res->write(global_idx + offset, shared_res[offset]);
        };
    };
}
}// namespace uipc::backend::luisa::culbvh

namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;
using culbvh::MAX_RES_PER_BLOCK;
// The kernel lambdas below live in this anonymous namespace (the nested node type is
// only found by enclosing-class lookup inside `StacklessBVH::Impl`'s members).
using Node = StacklessBVH::Node;

// ============================================================================
//  StacklessBVH_StacklessCDSharedSelf_kernel
// ============================================================================
/// NOTE (the port's record-time predicate): `pred` is *baked into* the cached shader
/// (the reference passes its device functor as a kernel argument, which an LC kernel
/// signature cannot carry).  The cache key is the predicate's type, so the predicate
/// must be value-semantics-stable: stateless, or with captures identical on every call.
template <typename Pred>
auto& stackless_cd_shared_self_kernel(Pred pred)
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LAABB>,
                                               lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<Node>,
                                               lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<LVector2i>, std::int32_t,
                                               std::int32_t, std::int32_t>(
        [pred](lcrt::BufferVar<LAABB> box, lcrt::BufferVar<std::int32_t> lvs_idx,
           lcrt::BufferVar<Node> nodes, lcrt::BufferVar<std::int32_t> res_counter,
           lcrt::BufferVar<LVector2i> res, lcrt::Var<std::int32_t> size,
           lcrt::Var<std::int32_t> int_size, lcrt::Var<std::int32_t> num_objs) noexcept
        {
            lcrt::set_block_size(std::uint32_t(culbvh::K_THREADS));

            const auto tid    = lcrt::cast<std::int32_t>(lcrt::dispatch_x());
            const auto active = tid < size;

            lcrt::Var<std::int32_t> idx{0};
            LBox<lcrt::Var<float>>  bv = box_empty<lcrt::Var<float>>();
            $if(active)
            {
                idx = lvs_idx->read(tid);
                bv  = load_aabb(box->read(idx));
            };

            $shared<LVector2i>    shared_res{std::uint32_t(MAX_RES_PER_BLOCK)};
            $shared<std::int32_t> shared_counter{1u};     // results cached in shared memory
            $shared<std::int32_t> shared_global_idx{1u};  // where to write in global memory
            $if(lcrt::thread_x() == 0u)
            {
                shared_counter[0] = 0;
            };

            lcrt::Var<std::int32_t> st{0};
            // Upper bound of iterations to avoid an infinite loop
            const auto max_iter = num_objs * 2;

            $while(true)
            {
                lcrt::sync_block();
                $if(active)
                {
                    // The reference's inner `for(; inner_I < MaxIter; inner_I++)` with
                    // its two `break`s; the counter survives the loop so the
                    // reference's `UIPC_KERNEL_ASSERT(inner_I < MaxIter, ...)` (an
                    // LC_LOG here) keeps its meaning.
                    lcrt::Var<std::int32_t> inner_i{0};
                    $while(inner_i < max_iter)
                    {
                        $if(st == -1)
                        {
                            $break;
                        };
                        // `node = _nodes[st]` -- the reference loads the three members
                        // separately; LC reads a whole struct element.
                        const auto node = nodes->read(st);
                        $if(box_intersects(load_aabb(node.bound), bv))
                        {
                            $if(node.lc == -1)
                            {
                                // only the thread with the smaller *sorted* position
                                // reports the pair
                                $if(tid < st - int_size)
                                {
                                    const auto other = lvs_idx->read(st - int_size);
                                    const auto pair  = culbvh::make_ordered_pair(idx, other);
                                    $if(pred(idx, other))
                                    {
                                        const auto s_idx =
                                            shared_counter.atomic(0u).fetch_add(1);
                                        $if(s_idx >= std::int32_t(MAX_RES_PER_BLOCK))
                                        {
                                            $break;
                                        };
                                        shared_res[s_idx] = pair;
                                    };
                                };
                                st = node.escape;
                            }
                            $else
                            {
                                st = node.lc;
                            };
                        }
                        $else
                        {
                            st = node.escape;
                        };
                        inner_i = inner_i + 1;
                    };
                    LC_LOG("Exceeded max iteration in stackless traversal, {} (Max={}), numObj=({})",
                           inner_i,
                           max_iter,
                           num_objs);
                };

                // Flush whatever we have
                lcrt::sync_block();
                const auto total_res_in_block =
                    lcrt::min(shared_counter[0], std::int32_t(MAX_RES_PER_BLOCK));

                $if(lcrt::thread_x() == 0u)
                {
                    // this block starts writing at shared_global_idx
                    shared_global_idx[0] =
                        res_counter.atomic(0u).fetch_add(total_res_in_block);
                };
                lcrt::sync_block();

                // Make sure we don't write out of bounds
                const auto global_idx =
                    lcrt::Var<std::int32_t>{shared_global_idx[0]};

                $if(lcrt::thread_x() == 0u)
                {
                    shared_counter[0] = 0;
                };

                // if there is at least one element empty it means we have found all
                // collisions for this block
                const auto done = total_res_in_block < std::int32_t(MAX_RES_PER_BLOCK);

                culbvh::safe_copy_to(shared_res, total_res_in_block, res, global_idx,
                                     lcrt::cast<std::int32_t>(res->size()));

                $if(done)
                {
                    $break;
                };
            };
        });
    return k;
}

// ============================================================================
//  StacklessBVH_StacklessCDSharedOther_kernel
// ============================================================================
/// See `stackless_cd_shared_self_kernel`'s note.
template <typename Pred>
auto& stackless_cd_shared_other_kernel(Pred pred)
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LAABB>,
                                               lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<Node>,
                                               lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<LVector2i>, std::int32_t,
                                               std::int32_t, std::int32_t>(
        [pred](lcrt::BufferVar<LAABB> box, lcrt::BufferVar<std::int32_t> sorted_idx,
           lcrt::BufferVar<std::int32_t> lvs_idx, lcrt::BufferVar<Node> nodes,
           lcrt::BufferVar<std::int32_t> res_counter, lcrt::BufferVar<LVector2i> res,
           lcrt::Var<std::int32_t> size, lcrt::Var<std::int32_t> int_size,
           lcrt::Var<std::int32_t> num_objs) noexcept
        {
            lcrt::set_block_size(std::uint32_t(culbvh::K_THREADS));

            const auto tid    = lcrt::cast<std::int32_t>(lcrt::dispatch_x());
            const auto active = tid < size;

            lcrt::Var<std::int32_t> idx{0};
            LBox<lcrt::Var<float>>  bv = box_empty<lcrt::Var<float>>();
            $if(active)
            {
                idx = sorted_idx->read(tid);
                bv  = load_aabb(box->read(idx));
            };

            $shared<LVector2i>    shared_res{std::uint32_t(MAX_RES_PER_BLOCK)};
            $shared<std::int32_t> shared_counter{1u};
            $shared<std::int32_t> shared_global_idx{1u};
            $if(lcrt::thread_x() == 0u)
            {
                shared_counter[0] = 0;
            };

            lcrt::Var<std::int32_t> st{0};
            const auto              max_iter = num_objs * 2;

            $while(true)
            {
                lcrt::sync_block();
                $if(active)
                {
                    lcrt::Var<std::int32_t> inner_i{0};
                    $while(inner_i < max_iter)
                    {
                        $if(st == -1)
                        {
                            $break;
                        };
                        const auto node = nodes->read(st);
                        $if(box_intersects(load_aabb(node.bound), bv))
                        {
                            $if(node.lc == -1)
                            {
                                const auto other = lvs_idx->read(st - int_size);
                                $if(pred(idx, other))
                                {
                                    const auto s_idx =
                                        shared_counter.atomic(0u).fetch_add(1);
                                    $if(s_idx >= std::int32_t(MAX_RES_PER_BLOCK))
                                    {
                                        $break;
                                    };
                                    shared_res[s_idx] = culbvh::make_pair_var(idx, other);
                                };
                                st = node.escape;
                            }
                            $else
                            {
                                st = node.lc;
                            };
                        }
                        $else
                        {
                            st = node.escape;
                        };
                        inner_i = inner_i + 1;
                    };
                    LC_LOG("Exceeded max iteration in stackless traversal, {} (Max={}), numObj=({})",
                           inner_i,
                           max_iter,
                           num_objs);
                };

                lcrt::sync_block();
                const auto total_res_in_block =
                    lcrt::min(shared_counter[0], std::int32_t(MAX_RES_PER_BLOCK));

                $if(lcrt::thread_x() == 0u)
                {
                    shared_global_idx[0] =
                        res_counter.atomic(0u).fetch_add(total_res_in_block);
                };
                lcrt::sync_block();

                const auto global_idx = lcrt::Var<std::int32_t>{shared_global_idx[0]};

                $if(lcrt::thread_x() == 0u)
                {
                    shared_counter[0] = 0;
                };
                lcrt::sync_block();

                const auto done = total_res_in_block < std::int32_t(MAX_RES_PER_BLOCK);

                culbvh::safe_copy_to(shared_res, total_res_in_block, res, global_idx,
                                     lcrt::cast<std::int32_t>(res->size()));

                $if(done)
                {
                    $break;
                };
            };
        });
    return k;
}
}// namespace

// ===========================================================================
//  The dispatch bodies (templated on the record-time predicate)
// ===========================================================================
template <typename Pred>
void StacklessBVH::Impl::StacklessCDSharedSelf(Pred                        pred,
                                               lud::LBufferView<int>       cpNum,
                                               lud::LBufferView<LVector2i> buffer,
                                               lud::LStream&               stream)
{
    const auto num_query = static_cast<std::int32_t>(ext_aabb.size());
    const auto num_objs  = num_query;
    const auto block_dim = std::uint32_t(culbvh::K_THREADS);
    const auto grid_dim =
        (num_query + std::int32_t(block_dim) - 1) / std::int32_t(block_dim);

    if(grid_dim > 0)
    {
        stream.record(stackless_cd_shared_self_kernel<Pred>(pred)(
                          objs, ext_idx.cview(), nodes.view(), cpNum, buffer, num_query,
                          num_objs - 1, num_objs)
                          .dispatch(static_cast<std::uint32_t>(grid_dim) * block_dim));
    }
}

template <typename Pred>
void StacklessBVH::Impl::StacklessCDSharedOther(
    Pred                            pred,
    lud::LCBufferView<LAABB>        query_aabbs,
    lud::LCBufferView<std::int32_t> query_sorted_id,
    lud::LBufferView<int>           cpNum,
    lud::LBufferView<LVector2i>     buffer,
    lud::LStream&                   stream)
{
    const auto num_query = static_cast<std::int32_t>(query_aabbs.size());
    const auto num_objs  = static_cast<std::int32_t>(ext_aabb.size());
    const auto block_dim = std::uint32_t(culbvh::K_THREADS);
    const auto grid_dim =
        (num_query + std::int32_t(block_dim) - 1) / std::int32_t(block_dim);

    if(grid_dim > 0)
    {
        stream.record(stackless_cd_shared_other_kernel<Pred>(pred)(
                          query_aabbs, query_sorted_id, ext_idx.cview(), nodes.view(),
                          cpNum, buffer, num_query, num_objs - 1, num_objs)
                          .dispatch(static_cast<std::uint32_t>(grid_dim) * block_dim));
    }
}

// ===========================================================================
//  detect / query
// ===========================================================================
template <std::invocable<IndexT, IndexT> Pred>
void StacklessBVH::detect(Pred callback, QueryBuffer& qbuffer)
{
    if(m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    auto& stream = frame_stream();

    auto do_query = [&]
    {
        // clear the counter
        qbuffer.m_cpNum.view().fill(0);
        m_impl.StacklessCDSharedSelf(callback, qbuffer.m_cpNum.view(),
                                     qbuffer.m_pairs.view(), stream);
    };

    do_query();

    // get the total number of pairs (the reference's implicit synchronous read)
    auto h_cp_num = lud::read_host_sync(stream, qbuffer.m_cpNum.cview());

    // if it failed, resize and retry
    if(h_cp_num > static_cast<std::int32_t>(qbuffer.m_pairs.size()))
    {
        const auto new_size =
            static_cast<size_t>(static_cast<Float>(h_cp_num) * m_impl.config.reserve_ratio);
        qbuffer.m_pairs.resize(new_size);
        do_query();
    }

    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = static_cast<SizeT>(h_cp_num);
}

template <std::invocable<IndexT, IndexT> Pred>
void StacklessBVH::query(lud::LCBufferView<LAABB> aabbs, Pred callback, QueryBuffer& qbuffer)
{
    if(aabbs.size() == 0 || m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    auto& stream = frame_stream();

    qbuffer.build(aabbs, stream);

    auto do_query = [&]
    {
        // clear the counter
        qbuffer.m_cpNum.view().fill(0);
        m_impl.StacklessCDSharedOther(callback, aabbs, qbuffer.m_querySortedId.cview(),
                                      qbuffer.m_cpNum.view(), qbuffer.m_pairs.view(),
                                      stream);
    };

    do_query();

    auto h_cp_num = lud::read_host_sync(stream, qbuffer.m_cpNum.cview());

    if(h_cp_num > static_cast<std::int32_t>(qbuffer.m_pairs.size()))
    {
        const auto new_size =
            static_cast<size_t>(static_cast<Float>(h_cp_num) * m_impl.config.reserve_ratio);
        qbuffer.m_pairs.resize(new_size);
        do_query();
    }

    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = static_cast<SizeT>(h_cp_num);
}
}// namespace uipc::backend::luisa
