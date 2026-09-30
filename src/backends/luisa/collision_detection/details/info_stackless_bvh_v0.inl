/********************************************************************
 * @file info_stackless_bvh_v0.inl
 * @brief InfoStacklessBVHV0 — the parts that must be visible to a kernel body (port of
 *        src/backends/cuda/collision_detection/details/info_stackless_bvh_v0.inl).
 *
 * Included at the end of info_stackless_bvh_v0.h.  The two traversal kernels are v1's
 * minus the shared-memory bid/cid pre-load (and with the pair predicates called with
 * the reference's raw `(i, j)` arguments); the `detect`/`query` bodies are the
 * reference's inline ones (no `launch_*` split, no `prepare_query_result`).
 *
 * The reference's `InfoStacklessBVHV0NodeCullAdapter` / `SelfPairPredAdapter` /
 * `OtherPairPredAdapter` exist only because CUDA passes a device functor as a kernel
 * argument; in the port the predicate *is* the recorded kernel body, so the adapters
 * collapse into the call sites below (`NodePredInfo{i, node_bid, node_cid}` and the
 * self flavour's `if(j <= i) return false;` guard).
 *********************************************************************/
#pragma once

#include <luisa_device/builtins.h>

#include <frame_stream.h>
#include <uipc/common/log.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;
using culbvh::K_THREADS;
using culbvh::MAX_RES_PER_BLOCK;
using InfoNodeV0 = InfoStacklessBVHV0::Node;

// ============================================================================
//  InfoStacklessBVHV0_stacklessSelf_kernel
// ============================================================================
/// NOTE: see info_stackless_bvh.inl — the predicates are *record-time* host callables
/// baked into the cached shader (the cache key is their type).
template <typename NodeCull, typename PairPred>
auto& info_v0_stackless_self_kernel(NodeCull node_cull, PairPred pair_pred)
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LAABB>, lcrt::Buffer<std::int32_t>, lcrt::Buffer<InfoNodeV0>,
        lcrt::Buffer<std::int32_t>, lcrt::Buffer<LVector2i>, std::int32_t, std::int32_t,
        std::int32_t>(
        [node_cull, pair_pred](lcrt::BufferVar<LAABB> box,
                               lcrt::BufferVar<std::int32_t> lvs_idx,
                               lcrt::BufferVar<InfoNodeV0> nodes,
                               lcrt::BufferVar<std::int32_t> res_counter,
                               lcrt::BufferVar<LVector2i> res, lcrt::Var<std::int32_t> size,
                               lcrt::Var<std::int32_t> int_size,
                               lcrt::Var<std::int32_t> num_objs) noexcept
        {
            lcrt::set_block_size(std::uint32_t(K_THREADS));

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
                            // the reference's NodeCullAdapter:
                            // `NodePredInfo{i, node_bid, node_cid}` -> np(info)
                            const typename InfoStacklessBVHV0::template NodePredInfoT<
                                lcrt::Var<IndexT>>
                                node_info{idx, node.bid, node.cid};
                            $if(node_cull(node_info))
                            {
                                $if(node.lc == -1)
                                {
                                    $if(tid < st - int_size)
                                    {
                                        const auto leaf_raw = lvs_idx->read(st - int_size);
                                        const auto q_first  = idx < leaf_raw;
                                        // the reference's `ordered_pair(idx, leaf_raw)`
                                        const auto pair_i =
                                            lcrt::select(leaf_raw, idx, q_first);
                                        const auto pair_j =
                                            lcrt::select(idx, leaf_raw, q_first);
                                        // the reference's SelfPairPredAdapter:
                                        // `if(j <= i) return false;`
                                        $if(pair_i < pair_j)
                                        {
                                            $if(pair_pred(pair_i, pair_j))
                                            {
                                                const auto s_idx =
                                                    shared_counter.atomic(0u).fetch_add(1);
                                                $if(s_idx
                                                    >= std::int32_t(MAX_RES_PER_BLOCK))
                                                {
                                                    $break;
                                                };
                                                shared_res[s_idx] =
                                                    culbvh::make_pair_var(pair_i, pair_j);
                                            };
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
//  InfoStacklessBVHV0_stacklessOther_kernel
// ============================================================================
template <typename NodeCull, typename PairPred>
auto& info_v0_stackless_other_kernel(NodeCull node_cull, PairPred pair_pred)
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LAABB>, lcrt::Buffer<std::int32_t>, lcrt::Buffer<std::int32_t>,
        lcrt::Buffer<InfoNodeV0>, lcrt::Buffer<std::int32_t>, lcrt::Buffer<LVector2i>,
        std::int32_t, std::int32_t, std::int32_t>(
        [node_cull, pair_pred](lcrt::BufferVar<LAABB> box,
                               lcrt::BufferVar<std::int32_t> sorted_idx,
                               lcrt::BufferVar<std::int32_t> lvs_idx,
                               lcrt::BufferVar<InfoNodeV0> nodes,
                               lcrt::BufferVar<std::int32_t> res_counter,
                               lcrt::BufferVar<LVector2i> res, lcrt::Var<std::int32_t> size,
                               lcrt::Var<std::int32_t> int_size,
                               lcrt::Var<std::int32_t> num_objs) noexcept
        {
            lcrt::set_block_size(std::uint32_t(K_THREADS));

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
                            const typename InfoStacklessBVHV0::template NodePredInfoT<
                                lcrt::Var<IndexT>>
                                node_info{idx, node.bid, node.cid};
                            $if(node_cull(node_info))
                            {
                                $if(node.lc == -1)
                                {
                                    const auto other = lvs_idx->read(st - int_size);
                                    // the reference's OtherPairPredAdapter:
                                    // `lp(i, j)` with NO guard
                                    $if(pair_pred(idx, other))
                                    {
                                        const auto s_idx =
                                            shared_counter.atomic(0u).fetch_add(1);
                                        $if(s_idx >= std::int32_t(MAX_RES_PER_BLOCK))
                                        {
                                            $break;
                                        };
                                        shared_res[s_idx] =
                                            culbvh::make_pair_var(idx, other);
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
//  The dispatch bodies
// ===========================================================================
template <typename NodeCull, typename PairPred>
void InfoStacklessBVHV0::Impl::stacklessSelf(NodeCull                       node_cull,
                                             PairPred                       pair_pred,
                                             lud::LBufferView<std::int32_t> cpNum,
                                             lud::LBufferView<LVector2i>    buffer,
                                             lud::LStream&                  stream)
{
    const auto num_query = static_cast<std::int32_t>(ext_aabb.size());
    const auto num_objs  = num_query;
    const auto block_dim = std::uint32_t(K_THREADS);
    const auto grid_dim =
        (num_query + std::int32_t(block_dim) - 1) / std::int32_t(block_dim);

    if(grid_dim <= 0)
        return;

    stream.record(info_v0_stackless_self_kernel<NodeCull, PairPred>(node_cull, pair_pred)(
                      objs, ext_idx.cview(), nodes.view(), cpNum, buffer, num_query,
                      num_objs - 1, num_objs)
                      .dispatch(static_cast<std::uint32_t>(grid_dim) * block_dim));
}

template <typename NodeCull, typename PairPred>
void InfoStacklessBVHV0::Impl::stacklessOther(
    NodeCull                        node_cull,
    PairPred                        pair_pred,
    lud::LCBufferView<LAABB>        query_aabbs,
    lud::LCBufferView<std::int32_t> query_sorted_id,
    lud::LBufferView<std::int32_t>  cpNum,
    lud::LBufferView<LVector2i>     buffer,
    lud::LStream&                   stream)
{
    const auto num_query = static_cast<std::int32_t>(query_aabbs.size());
    const auto num_objs  = static_cast<std::int32_t>(ext_aabb.size());
    const auto block_dim = std::uint32_t(K_THREADS);
    const auto grid_dim =
        (num_query + std::int32_t(block_dim) - 1) / std::int32_t(block_dim);

    if(grid_dim <= 0)
        return;

    stream.record(info_v0_stackless_other_kernel<NodeCull, PairPred>(node_cull, pair_pred)(
                      query_aabbs, query_sorted_id, ext_idx.cview(), nodes.view(), cpNum,
                      buffer, num_query, num_objs - 1, num_objs)
                      .dispatch(static_cast<std::uint32_t>(grid_dim) * block_dim));
}

// ===========================================================================
//  detect / query (the reference's inline bodies)
// ===========================================================================
template <typename NodePred, typename LeafPred>
void InfoStacklessBVHV0::detect(MaskView  cmts,
                                NodePred  np,
                                LeafPred  lp,
                                QueryBuffer& qbuffer)
{
    if(m_aabbs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }
    UIPC_ASSERT(m_aabbs.size() == m_BIDs.size(),
                "AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                m_aabbs.size(),
                m_BIDs.size());
    UIPC_ASSERT(m_aabbs.size() == m_CIDs.size(),
                "AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                m_aabbs.size(),
                m_CIDs.size());

    auto& stream = frame_stream();
    // the reference's `do_query` lambda: the counter is cleared *before every* pass
    // (including the retry — the port's first attempt missed that and the second pass
    // appended to the first pass's total)
    qbuffer.m_cpNum.view().fill(0);
    m_impl.stacklessSelf(np, lp, qbuffer.m_cpNum.view(), qbuffer.m_pairs.view(), stream);

    const auto h_cp_num =
        static_cast<int>(lud::read_host_sync(stream, qbuffer.m_cpNum.cview()));
    if(h_cp_num > static_cast<int>(qbuffer.m_pairs.size()))
    {
        const auto new_size =
            static_cast<size_t>(static_cast<Float>(h_cp_num) * m_impl.config.reserve_ratio);
        qbuffer.m_pairs.resize(new_size);
        qbuffer.m_cpNum.view().fill(0);
        m_impl.stacklessSelf(np, lp, qbuffer.m_cpNum.view(), qbuffer.m_pairs.view(),
                             stream);
    }
    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = static_cast<SizeT>(h_cp_num);
}

template <typename NodePred, typename LeafPred>
void InfoStacklessBVHV0::query(lud::LCBufferView<LAABB>   query_aabbs,
                               lud::LCBufferView<IndexT> query_BIDs,
                               lud::LCBufferView<IndexT> query_CIDs,
                               MaskView                  cmts,
                               NodePred                  np,
                               LeafPred                  lp,
                               QueryBuffer&              qbuffer)
{
    if(m_aabbs.size() == 0 || query_aabbs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }
    UIPC_ASSERT(query_aabbs.size() == query_BIDs.size(),
                "Query AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                query_aabbs.size(),
                query_BIDs.size());
    UIPC_ASSERT(query_aabbs.size() == query_CIDs.size(),
                "Query AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                query_aabbs.size(),
                query_CIDs.size());

    auto& stream = frame_stream();
    qbuffer.build(query_aabbs, stream);

    qbuffer.m_cpNum.view().fill(0);
    m_impl.stacklessOther(np, lp, query_aabbs, qbuffer.m_querySortedId.cview(),
                          qbuffer.m_cpNum.view(), qbuffer.m_pairs.view(), stream);

    const auto h_cp_num =
        static_cast<int>(lud::read_host_sync(stream, qbuffer.m_cpNum.cview()));
    if(h_cp_num > static_cast<int>(qbuffer.m_pairs.size()))
    {
        const auto new_size =
            static_cast<size_t>(static_cast<Float>(h_cp_num) * m_impl.config.reserve_ratio);
        qbuffer.m_pairs.resize(new_size);
        qbuffer.m_cpNum.view().fill(0);
        m_impl.stacklessOther(np, lp, query_aabbs, qbuffer.m_querySortedId.cview(),
                              qbuffer.m_cpNum.view(), qbuffer.m_pairs.view(), stream);
    }
    UIPC_ASSERT(h_cp_num >= 0, "fatal error");
    qbuffer.m_size = static_cast<SizeT>(h_cp_num);
}
}// namespace uipc::backend::luisa
