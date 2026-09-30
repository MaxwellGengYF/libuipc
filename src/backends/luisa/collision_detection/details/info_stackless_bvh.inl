/********************************************************************
 * @file info_stackless_bvh.inl
 * @brief InfoStacklessBVH — the parts that must be visible to a kernel body (port of
 *        src/backends/cuda/collision_detection/details/info_stackless_bvh.inl).
 *
 * Included at the end of info_stackless_bvh.h.  The build kernels and the host-side
 * step list live in info_stackless_bvh.cpp; this file holds
 *   1. the two traversal kernels (the reference's `stacklessSelf` / `stacklessOther`
 *      __global__ bodies), templated on the record-time predicates,
 *   2. the `Impl::stacklessSelf` / `Impl::stacklessOther` dispatch bodies,
 *   3. the public `detect` / `launch_detect` / `prepare_query_result` / `query` /
 *      `launch_query` bodies.
 *
 * The reference vendors its own copy of the culbvh helpers under
 * `uipc::info_stackless_detail`; the port reuses T7.1a's `culbvh` (the values are the
 * same arithmetic — see stackless_bvh.h's PORT DEVIATIONS), so `morton3D`,
 * `make_pair_var`, `safe_copy_to`, `K_THREADS` and `MAX_RES_PER_BLOCK` all come from
 * there.
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
// The kernel lambdas below live in this anonymous namespace, so the nested node type
// needs an alias (`InfoStacklessBVH::Node` is only found by enclosing-class lookup
// inside Impl's members).
using InfoNode = InfoStacklessBVH::Node;

// ============================================================================
//  InfoStacklessBVH_stacklessSelf_kernel
// ============================================================================
/// NOTE (the port's record-time predicate, T7.1a's rule 13): `node_cull` / `pair_pred`
/// are *baked into* the cached shader (the reference passes its device functors as
/// kernel arguments, which an LC kernel signature cannot carry).  The cache key is the
/// predicate's type, so the predicates must be value-semantics-stable: stateless, or
/// with captures identical on every call.
///
/// The reference's shared-memory pre-load is transcribed: `s_qbid`/`s_qcid` are filled
/// once per thread before the hot loop, and `NodePredInfo::query_bid`/`query_cid` come
/// from them (this is the *only* difference from T7.1a's
/// `stackless_cd_shared_self_kernel` in this file's kernel body).
template <typename NodeCull, typename PairPred>
auto& info_stackless_self_kernel(NodeCull node_cull, PairPred pair_pred)
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LAABB>, lcrt::Buffer<std::int32_t>, lcrt::Buffer<InfoNode>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<std::int32_t>,
        lcrt::Buffer<LVector2i>, std::int32_t, std::int32_t, std::int32_t, std::int32_t>(
        [node_cull, pair_pred](lcrt::BufferVar<LAABB> box,
                               lcrt::BufferVar<std::int32_t> lvs_idx,
                               lcrt::BufferVar<InfoNode> nodes, lcrt::BufferVar<IndexT> bids,
                               lcrt::BufferVar<IndexT> cids,
                               lcrt::BufferVar<std::int32_t> res_counter,
                               lcrt::BufferVar<LVector2i> res, lcrt::Var<std::int32_t> size,
                               lcrt::Var<std::int32_t> int_size,
                               lcrt::Var<std::int32_t> num_objs,
                               lcrt::Var<std::int32_t> has_info) noexcept
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

            // -----------------------------------------------------------------
            // SMem: pre-load query bid/cid once per thread, before the hot loop.
            // The reference's `s_qbid[threadIdx.x] = (active && has_info) ?
            // _bids(idx) : invalid;` — predicated so that no read happens for an
            // inactive thread (`idx == -1`).
            // -----------------------------------------------------------------
            $shared<IndexT> s_qbid{std::uint32_t(K_THREADS)};
            $shared<IndexT> s_qcid{std::uint32_t(K_THREADS)};

            lcrt::Var<IndexT> h_qbid{IndexT(-1)};
            lcrt::Var<IndexT> h_qcid{IndexT(-1)};
            $if(active & (has_info != 0))
            {
                h_qbid = bids->read(idx);
                h_qcid = cids->read(idx);
            };
            s_qbid[lcrt::thread_x()] = h_qbid;
            s_qcid[lcrt::thread_x()] = h_qcid;

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
                // The reference's first __syncthreads also makes the SMem pre-loads
                // above visible to the block.
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
                            // The reference calls node_cull on *every* intersected
                            // node, leaves included (so a node predicate that rejects
                            // a leaf node rejects that primitive too).
                            const typename InfoStacklessBVH::template NodePredInfoT<
                                lcrt::Var<IndexT>>
                                node_info{idx,
                                          s_qbid[lcrt::thread_x()],
                                          s_qcid[lcrt::thread_x()],
                                          node.bid,
                                          node.cid};
                            $if(node_cull(node_info))
                            {
                                $if(node.lc == -1)
                                {
                                    // only the thread with the smaller *sorted*
                                    // position reports the pair
                                    $if(tid < st - int_size)
                                    {
                                        const auto leaf_raw = lvs_idx->read(st - int_size);
                                        const auto q_first  = idx < leaf_raw;
                                        // the reference's `ordered_pair(idx, leaf_raw)`;
                                        // the components are kept so that the
                                        // LeafPredInfo can be built from them
                                        const auto pair_i =
                                            lcrt::select(leaf_raw, idx, q_first);
                                        const auto pair_j =
                                            lcrt::select(idx, leaf_raw, q_first);
                                        const auto pair = culbvh::make_pair_var(pair_i, pair_j);

                                        const typename InfoStacklessBVH::template LeafPredInfoT<
                                            lcrt::Var<IndexT>>
                                            leaf_info{
                                                pair_i,
                                                pair_j,
                                                lcrt::select(node.bid, h_qbid, q_first),
                                                lcrt::select(node.cid, h_qcid, q_first),
                                                lcrt::select(h_qbid, node.bid, q_first),
                                                lcrt::select(h_qcid, node.cid, q_first)};

                                        $if(pair_pred(leaf_info))
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
                const auto global_idx = lcrt::Var<std::int32_t>{shared_global_idx[0]};

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
//  InfoStacklessBVH_stacklessOther_kernel
// ============================================================================
/// See `info_stackless_self_kernel`'s note.
template <typename NodeCull, typename PairPred>
auto& info_stackless_other_kernel(NodeCull node_cull, PairPred pair_pred)
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LAABB>, lcrt::Buffer<std::int32_t>, lcrt::Buffer<std::int32_t>,
        lcrt::Buffer<InfoNode>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<std::int32_t>, lcrt::Buffer<LVector2i>, std::int32_t, std::int32_t,
        std::int32_t, std::int32_t>(
        [node_cull, pair_pred](lcrt::BufferVar<LAABB> box,
                               lcrt::BufferVar<std::int32_t> sorted_idx,
                               lcrt::BufferVar<std::int32_t> lvs_idx,
                               lcrt::BufferVar<InfoNode> nodes,
                               lcrt::BufferVar<IndexT> qbids, lcrt::BufferVar<IndexT> qcids,
                               lcrt::BufferVar<std::int32_t> res_counter,
                               lcrt::BufferVar<LVector2i> res, lcrt::Var<std::int32_t> size,
                               lcrt::Var<std::int32_t> int_size,
                               lcrt::Var<std::int32_t> num_objs,
                               lcrt::Var<std::int32_t> qhas_info) noexcept
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

            // -----------------------------------------------------------------
            // SMem: pre-load the per-query bid/cid before the traversal loop.
            // -----------------------------------------------------------------
            $shared<IndexT> s_qbid{std::uint32_t(K_THREADS)};
            $shared<IndexT> s_qcid{std::uint32_t(K_THREADS)};

            lcrt::Var<IndexT> h_qbid{IndexT(-1)};
            lcrt::Var<IndexT> h_qcid{IndexT(-1)};
            $if(active & (qhas_info != 0))
            {
                h_qbid = qbids->read(idx);
                h_qcid = qcids->read(idx);
            };
            s_qbid[lcrt::thread_x()] = h_qbid;
            s_qcid[lcrt::thread_x()] = h_qcid;

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
                            const typename InfoStacklessBVH::template NodePredInfoT<
                                lcrt::Var<IndexT>>
                                node_info{idx,
                                          s_qbid[lcrt::thread_x()],
                                          s_qcid[lcrt::thread_x()],
                                          node.bid,
                                          node.cid};
                            $if(node_cull(node_info))
                            {
                                $if(node.lc == -1)
                                {
                                    const auto other = lvs_idx->read(st - int_size);
                                    // the reference's `int2{idx, _lvs_idx(st - intSize)}`
                                    // (query index first, NO ordering)
                                    const auto pair = culbvh::make_pair_var(idx, other);

                                    const typename InfoStacklessBVH::template LeafPredInfoT<
                                        lcrt::Var<IndexT>>
                                        leaf_info{idx,
                                                  other,
                                                  h_qbid,
                                                  h_qcid,
                                                  node.bid,
                                                  node.cid};

                                    $if(pair_pred(leaf_info))
                                    {
                                        const auto s_idx =
                                            shared_counter.atomic(0u).fetch_add(1);
                                        $if(s_idx >= std::int32_t(MAX_RES_PER_BLOCK))
                                        {
                                            $break;
                                        };
                                        shared_res[s_idx] = pair;
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
//  The dispatch bodies (templated on the record-time predicates)
// ===========================================================================
template <typename NodeCull, typename PairPred>
void InfoStacklessBVH::Impl::stacklessSelf(NodeCull                       node_cull,
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
    {
        return;
    }

    // the reference's `bool has_info = bids.size() == (size_t)num_objs && ...`
    const auto has_info =
        (bids.size() == static_cast<size_t>(num_objs)
         && cids.size() == static_cast<size_t>(num_objs)) ?
            std::int32_t(1) :
            std::int32_t(0);

    // LC rejects a zero-length buffer view as a kernel argument; when `build(aabbs)`
    // was used the (never-read, see `has_info`) dummy span is passed instead.
    const auto bid_arg = has_info != 0 ? bids : info_dummy.cview();
    const auto cid_arg = has_info != 0 ? cids : info_dummy.cview();

    stream.record(info_stackless_self_kernel<NodeCull, PairPred>(node_cull, pair_pred)(
                      objs, ext_idx.cview(), nodes.view(), bid_arg, cid_arg, cpNum,
                      buffer, num_query, num_objs - 1, num_objs, has_info)
                      .dispatch(static_cast<std::uint32_t>(grid_dim) * block_dim));
}

template <typename NodeCull, typename PairPred>
void InfoStacklessBVH::Impl::stacklessOther(
    NodeCull                       node_cull,
    PairPred                       pair_pred,
    lud::LCBufferView<LAABB>       query_aabbs,
    lud::LCBufferView<IndexT>      query_bids,
    lud::LCBufferView<IndexT>      query_cids,
    lud::LCBufferView<std::int32_t> query_sorted_id,
    lud::LBufferView<std::int32_t> cpNum,
    lud::LBufferView<LVector2i>    buffer,
    lud::LStream&                  stream)
{
    const auto num_query = static_cast<std::int32_t>(query_aabbs.size());
    const auto num_objs  = static_cast<std::int32_t>(ext_aabb.size());
    const auto block_dim = std::uint32_t(K_THREADS);
    const auto grid_dim =
        (num_query + std::int32_t(block_dim) - 1) / std::int32_t(block_dim);

    if(grid_dim <= 0)
    {
        return;
    }

    const auto qhas_info =
        (query_bids.size() == static_cast<size_t>(num_query)
         && query_cids.size() == static_cast<size_t>(num_query)) ?
            std::int32_t(1) :
            std::int32_t(0);

    // see stacklessSelf: a zero-length view is not a kernel argument
    const auto qbid_arg = qhas_info != 0 ? query_bids : info_dummy.cview();
    const auto qcid_arg = qhas_info != 0 ? query_cids : info_dummy.cview();

    stream.record(info_stackless_other_kernel<NodeCull, PairPred>(node_cull, pair_pred)(
                      query_aabbs, query_sorted_id, ext_idx.cview(), nodes.view(),
                      qbid_arg, qcid_arg, cpNum, buffer, num_query, num_objs - 1,
                      num_objs, qhas_info)
                      .dispatch(static_cast<std::uint32_t>(grid_dim) * block_dim));
}

// ===========================================================================
//  detect / query (the reference's public API)
// ===========================================================================
template <typename NodePred, typename LeafPred>
void InfoStacklessBVH::launch_detect(MaskView  cmts,
                                     NodePred  np,
                                     LeafPred  lp,
                                     QueryBuffer& qbuffer)
{
    // The reference's `BufferLaunch().fill(qbuffer.m_cpNum.view(), 0)` then the
    // m_aabbs/bids/cids asserts — in that order.
    auto& stream = frame_stream();
    qbuffer.m_cpNum.view().fill(0);

    if(m_aabbs.size() == 0)
        return;

    UIPC_ASSERT(m_aabbs.size() == m_BIDs.size(),
                "AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                m_aabbs.size(),
                m_BIDs.size());
    UIPC_ASSERT(m_aabbs.size() == m_CIDs.size(),
                "AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                m_aabbs.size(),
                m_CIDs.size());

    m_impl.stacklessSelf(np, lp, qbuffer.m_cpNum.view(), qbuffer.m_pairs.view(), stream);
}

// The `cmts` argument is the reference's `CBuffer2DView<IndexT>`; its own bodies never
// read it (see the header note), so the port keeps it for signature parity only.
template <typename NodePred, typename LeafPred>
void InfoStacklessBVH::detect(MaskView  cmts,
                              NodePred  np,
                              LeafPred  lp,
                              QueryBuffer& qbuffer)
{
    auto& stream = frame_stream();

    launch_detect(cmts, np, lp, qbuffer);

    // the reference's implicitly-synchronous `int h_cp_num = qbuffer.m_cpNum;`
    const auto h_cp_num =
        static_cast<int>(lud::read_host_sync(stream, qbuffer.m_cpNum.cview()));

    if(prepare_query_result(qbuffer, h_cp_num))
        launch_detect(cmts, np, lp, qbuffer);
}

// Publish a device-produced count and grow the output if a retry is required.
inline bool InfoStacklessBVH::prepare_query_result(QueryBuffer& qbuffer, int count)
{
    UIPC_ASSERT(count >= 0, "BVH query count must be non-negative, got %d", count);

    const bool retry = static_cast<size_t>(count) > qbuffer.m_pairs.size();
    if(retry)
    {
        const auto new_size =
            static_cast<size_t>(static_cast<Float>(count) * m_impl.config.reserve_ratio);
        qbuffer.m_pairs.resize(new_size);
    }
    qbuffer.m_size = static_cast<SizeT>(count);
    return retry;
}

template <typename NodePred, typename LeafPred>
void InfoStacklessBVH::launch_query(lud::LCBufferView<LAABB>   query_aabbs,
                                    lud::LCBufferView<IndexT> query_BIDs,
                                    lud::LCBufferView<IndexT> query_CIDs,
                                    MaskView                  cmts,
                                    NodePred                  np,
                                    LeafPred                  lp,
                                    QueryBuffer&              qbuffer,
                                    bool                      rebuild_query)
{
    auto& stream = frame_stream();
    qbuffer.m_cpNum.view().fill(0);

    if(m_aabbs.size() == 0 || query_aabbs.size() == 0)
        return;

    UIPC_ASSERT(query_aabbs.size() == query_BIDs.size(),
                "Query AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                query_aabbs.size(),
                query_BIDs.size());
    UIPC_ASSERT(query_aabbs.size() == query_CIDs.size(),
                "Query AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                query_aabbs.size(),
                query_CIDs.size());

    if(rebuild_query)
        qbuffer.build(query_aabbs, stream);

    m_impl.stacklessOther(np, lp, query_aabbs, query_BIDs, query_CIDs,
                          qbuffer.m_querySortedId.cview(), qbuffer.m_cpNum.view(),
                          qbuffer.m_pairs.view(), stream);
}

template <typename NodePred, typename LeafPred>
void InfoStacklessBVH::query(lud::LCBufferView<LAABB>   query_aabbs,
                             lud::LCBufferView<IndexT> query_BIDs,
                             lud::LCBufferView<IndexT> query_CIDs,
                             MaskView                  cmts,
                             NodePred                  np,
                             LeafPred                  lp,
                             QueryBuffer&              qbuffer)
{
    auto& stream = frame_stream();

    launch_query(query_aabbs, query_BIDs, query_CIDs, cmts, np, lp, qbuffer, true);

    const auto h_cp_num =
        static_cast<int>(lud::read_host_sync(stream, qbuffer.m_cpNum.cview()));

    if(prepare_query_result(qbuffer, h_cp_num))
    {
        // the reference's `launch_query(..., false)`: the query Morton order is
        // already built for this input.
        launch_query(query_aabbs, query_BIDs, query_CIDs, cmts, np, lp, qbuffer, false);
    }
}
}// namespace uipc::backend::luisa
