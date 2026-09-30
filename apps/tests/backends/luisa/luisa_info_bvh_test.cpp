/********************************************************************
 * @file luisa_info_bvh_test.cpp
 * @brief T7.1b — the "info" BVH batch: InfoStacklessBVH (v1), InfoStacklessBVHV0 and the
 *        two deferred sanity checkers (LuisaCompute backend).  Tag `[bvh_info]`.
 *
 * The reference structures are CUDA-only (nvcc device functors + raw intrinsics), so
 * they cannot be host-included: every oracle below is derived independently *in this
 * file* from the reference's specification
 * (collision_detection/{info_stackless_bvh,info_stackless_bvh_v0}.{h,inl}, the reference
 * tests apps/tests/backends/cuda/{info_stackless_bvh.cu,info_stackless_bvh_v0.cu} and the
 * two sanity checkers), never by calling the ported code.
 *
 * Coverage map (case -> oracle)
 *  1) the candidate/visited contract of both structures against an independent
 *     brute-force AABB oracle (1/2/37/4096/100k primitives + the degenerate clouds:
 *     identical points, duplicates, a line) ....................... info_contract
 *  2) the exported node array walked by a host re-implementation of the reference's
 *     traversal (keep-all *and* culling node predicates) ........... info_traversal_oracle
 *  3) the bid/cid metadata contract (leaves carry their primitive's ids; an internal
 *     node carries the *common* id or -1) ......................... info_metadata
 *  4) v1 / v0 / StacklessBVH / AtomicCountingLBVH agree on the same input
 *     .............................................................. info_cross_structure
 *  5) the v1-vs-v0 differences, pinned one by one .................. info_v1_vs_v0
 *  6) determinism + the QueryBuffer capacity contract .............. info_determinism
 *  7) the build's derived arrays (Morton codes, the stable sort, the internal-node
 *     permutation, the depth scan) device vs an independent host derivation
 *     .............................................................. info_device_vs_host
 *  8) the two ported sanity checkers: a scene each that passes and one that trips
 *     each check (uid 1 and uid 3), with the diagnostics quoted ... info_sanity_checkers
 *  9) per-case assertion accounting ................................ info_summary
 *
 * WHAT THIS FILE CANNOT REACH (stated so nothing is implied)
 * ---------------------------------------------------------
 *  * There is no CUDA backend in this build, so the reference's own claim ("the same
 *    candidate sets as a CUDA-built tree") cannot be re-run here; the oracles are exact
 *    set equality against an independent brute force plus the structural identities of
 *    the exported array.
 *  * **A ported predicate cannot read device arrays** (an LC kernel signature carries no
 *    callable, and a captured view becomes a *bound* buffer of the compiled shader —
 *    cookbook §7.9 rule 2).  The reference's own test predicates that read `bids`/`cmts`
 *    and its call-counting predicates (device `atomicAdd`s) are therefore NOT ported:
 *    the culling *contract* is pinned instead through the reported candidate sets, with
 *    the info-struct fields compared against host constants, and through this file's
 *    host traversal simulation.  The consequence for the sanity checkers is spelled out
 *    in their own port notes (they filter the reported candidates on the host).
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <collision_detection/atomic_counting_lbvh.h>
#include <collision_detection/info_stackless_bvh.h>
#include <collision_detection/info_stackless_bvh_v0.h>
#include <collision_detection/stackless_bvh.h>

#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/uipc.h>
// The test target's include path contains src/backends/none before src/backends/luisa,
// so the bare <sim_engine.h> would resolve to the none backend's shell (§7.14 rule 7).
#include <backends/luisa/sim_engine.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace uipc::backend::luisa
{
namespace info_bvh_test
{
namespace lla  = luisa_device::linalg;
namespace lcrt = ::luisa::compute;

using Vector3f = Eigen::Vector3f;

// ===========================================================================
//  Shared test helpers (the [bvh] file's helpers are TU-local; these are
//  re-derived here so the two files stay independent)
// ===========================================================================
[[nodiscard]] lud::LDeviceBuffer<LAABB> upload_aabbs(const std::vector<AABB>& boxes)
{
    lud::LDeviceBuffer<LAABB> dev;
    dev.resize(boxes.size());
    std::vector<LAABB> host(boxes.size());
    for(size_t i = 0; i < boxes.size(); ++i)
        host[i] = to_luisa(boxes[i]);
    if(!host.empty())
        dev.view().copy_from(std::span<const LAABB>{host});
    return dev;
}

[[nodiscard]] lud::LDeviceBuffer<IndexT> upload_ids(const std::vector<IndexT>& ids)
{
    lud::LDeviceBuffer<IndexT> dev;
    dev.resize(ids.size());
    if(!ids.empty())
        dev.view().copy_from(std::span<const IndexT>{ids});
    return dev;
}

[[nodiscard]] std::vector<AABB> download_aabbs(lud::LCBufferView<LAABB> view)
{
    std::vector<LAABB> tmp(view.size());
    if(!tmp.empty())
        view.copy_to(tmp);
    std::vector<AABB> out(tmp.size());
    for(size_t i = 0; i < tmp.size(); ++i)
        out[i] = to_eigen(tmp[i]);
    return out;
}

[[nodiscard]] std::vector<InfoStacklessBVH::Node> download_nodes(
    lud::LCBufferView<InfoStacklessBVH::Node> view)
{
    std::vector<InfoStacklessBVH::Node> tmp(view.size());
    if(!tmp.empty())
        view.copy_to(tmp);
    return tmp;
}

[[nodiscard]] std::vector<InfoStacklessBVHV0::Node> download_nodes_v0(
    lud::LCBufferView<InfoStacklessBVHV0::Node> view)
{
    std::vector<InfoStacklessBVHV0::Node> tmp(view.size());
    if(!tmp.empty())
        view.copy_to(tmp);
    return tmp;
}

template <typename T>
[[nodiscard]] std::vector<T> download(lud::LCBufferView<T> view)
{
    std::vector<T> tmp(view.size());
    if(!tmp.empty())
        view.copy_to(tmp);
    return tmp;
}

/// A deterministic cloud of `n` boxes with a *mix* of sizes.
[[nodiscard]] std::vector<AABB> make_cloud(size_t n, std::uint32_t seed, float extent = 1.0f)
{
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> pos(-extent, extent);
    std::uniform_real_distribution<float> half(0.02f, 0.20f);
    std::vector<AABB>                     boxes(n);
    for(auto& b : boxes)
    {
        Vector3f c{pos(rng), pos(rng), pos(rng)};
        Vector3f h{half(rng), half(rng), half(rng)};
        b.min() = c - h;
        b.max() = c + h;
    }
    return boxes;
}

[[nodiscard]] std::vector<AABB> make_identical_points(size_t n)
{
    const Vector3f    p{0.25f, -0.5f, 0.75f};
    std::vector<AABB> boxes(n);
    for(auto& b : boxes)
    {
        b.min() = p;
        b.max() = p;
    }
    return boxes;
}

[[nodiscard]] std::vector<AABB> make_line(size_t n)
{
    std::vector<AABB> boxes(n);
    for(size_t i = 0; i < n; ++i)
    {
        const auto x = static_cast<float>(i / 2u);
        boxes[i].min() = Vector3f{x, 0.0f, 0.0f};
        boxes[i].max() = Vector3f{x + 0.5f, 0.1f, 0.1f};
    }
    return boxes;
}

[[nodiscard]] std::vector<AABB> make_duplicate_aabbs(size_t n, std::uint32_t seed)
{
    std::mt19937      rng(seed);
    std::vector<AABB> pool = make_cloud(n / 4 + 1, seed, 0.5f);
    std::vector<AABB> out;
    out.reserve(n);
    while(out.size() < n)
        out.push_back(pool[rng() % pool.size()]);
    return out;
}

/// the per-primitive metadata used everywhere below (the reference's `i % 7` / `i % 3`
/// convention from its own info test)
[[nodiscard]] std::vector<IndexT> make_bids(size_t n)
{
    std::vector<IndexT> out(n);
    for(size_t i = 0; i < n; ++i)
        out[i] = static_cast<IndexT>(i % 7);
    return out;
}
[[nodiscard]] std::vector<IndexT> make_cids(size_t n)
{
    std::vector<IndexT> out(n);
    for(size_t i = 0; i < n; ++i)
        out[i] = static_cast<IndexT>(i % 3);
    return out;
}

using PairSet = std::set<std::pair<std::int32_t, std::int32_t>>;

[[nodiscard]] PairSet brute_force_self(const std::vector<AABB>& boxes)
{
    PairSet pairs;
    for(std::size_t i = 0; i < boxes.size(); ++i)
        for(std::size_t j = i + 1; j < boxes.size(); ++j)
            if(boxes[i].intersects(boxes[j]))
                pairs.emplace(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j));
    return pairs;
}

[[nodiscard]] PairSet brute_force_query(const std::vector<AABB>& queries,
                                        const std::vector<AABB>& objects)
{
    PairSet pairs;
    for(std::size_t i = 0; i < queries.size(); ++i)
        for(std::size_t j = 0; j < objects.size(); ++j)
            if(queries[i].intersects(objects[j]))
                pairs.emplace(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j));
    return pairs;
}

[[nodiscard]] PairSet pair_set(lud::LBufferView<LVector2i> view)
{
    std::vector<LVector2i> tmp(view.size());
    if(!tmp.empty())
        view.copy_to(tmp);
    PairSet out;
    for(const auto& p : tmp)
        out.emplace(p(0), p(1));
    return out;
}

/// The exact oracle for a *node-level* cull (a predicate that rejects a node whose
/// bid/cid equals `C`).  In the reference's self traversal only the *target* node's
/// metadata is tested: the pair is reported by the primitive with the **smaller sorted
/// position**, and its walk descends to the other primitive — the reporter's own leaf is
/// never the target of its own report (and its ancestors are only ever *escape* targets).
/// So a pair `(i, j)` survives iff the primitive with the larger sorted position has
/// `ids != C`.  (The query flavour is simpler: the visited node is always the object's, so
/// `ids[object] != C`.)
[[nodiscard]] PairSet brute_force_self_node_cull(const std::vector<AABB>& boxes,
                                                 const std::vector<std::int32_t>& ext_idx,
                                                 const std::vector<IndexT>&       ids,
                                                 IndexT                           C)
{
    std::vector<IndexT> pos(boxes.size(), -1);
    for(size_t t = 0; t < ext_idx.size(); ++t)
        pos[static_cast<size_t>(ext_idx[t])] = static_cast<IndexT>(t);

    PairSet out;
    for(const auto& p : brute_force_self(boxes))
    {
        const auto ti     = pos[static_cast<size_t>(p.first)];
        const auto tj     = pos[static_cast<size_t>(p.second)];
        const auto target = (ti < tj) ? p.second : p.first;
        if(ids[static_cast<size_t>(target)] != C)
            out.insert(p);
    }
    return out;
}

// ===========================================================================
//  The record-time predicates
// ===========================================================================
/// v1: keep every node / every pair (the reference's `NodePred{}`/`LeafPredTrue`).
struct KeepAllNodesV1
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::NodePredInfoT<I>&) const noexcept
    {
        return true;
    }
};

struct KeepAllLeavesV1
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::LeafPredInfoT<I>&) const noexcept
    {
        return true;
    }
};

/// v0: same, with the v0 signatures (3-field node info, `(i, j)` pair).
struct KeepAllNodesV0
{
    template <typename I>
    auto operator()(const InfoStacklessBVHV0::NodePredInfoT<I>&) const noexcept
    {
        return true;
    }
};

struct KeepAllLeavesV0
{
    template <typename A, typename B>
    auto operator()(const A&, const B&) const noexcept
    {
        return true;
    }
};

/// Reject any node whose *contact* id equals `Cid`.  Because an internal node's cid is the
/// common cid of its whole range, this rejects exactly the primitives whose cid is `Cid` —
/// a scalar-only way to pin that the node predicate is applied to *leaf* nodes too (the
/// reference's `NodePredInfo` comment).
///
/// NOTE the template parameter: a record-time predicate is *baked into* the cached shader
/// and the cache key is its **type** (T7.1a's rule 13 / the port's kernel notes), so two
/// predicates that differ only in a captured value share the first one's shader unless the
/// value is part of the type.  The test therefore carries the constants in the type.
template <IndexT Cid>
struct NodeCidCullT
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::NodePredInfoT<I>& info) const noexcept
    {
        return info.node_cid != Cid;
    }
};

/// Reject any node whose *body* id equals `Bid`.
template <IndexT Bid>
struct NodeBidCullT
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::NodePredInfoT<I>& info) const noexcept
    {
        return info.node_bid != Bid;
    }
};

/// Keep a node only when its bid is *valid* (the return value is the reference's
/// `node_cull`, i.e. the *keep* condition): a leaf's bid is never -1 in this file's
/// clouds, so this predicate only ever turns into a subtree cull at an internal node —
/// the one observable consequence of the reference's "common bid, or -1" propagation.
struct NodeValidBidOnlyCull
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::NodePredInfoT<I>& info) const noexcept
    {
        return info.node_bid != static_cast<IndexT>(-1);
    }
};

/// Keep only the pairs whose *resolved* (bid_i, bid_j) is exactly `(BI, BJ)` — pins that
/// v1's `LeafPredInfo::bid_i` belongs to `info.i` (the query for `query`, the smaller
/// index for `detect`) and `bid_j` to `info.j`.
template <IndexT BI, IndexT BJ>
struct LeafBidPairT
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::LeafPredInfoT<I>& info) const noexcept
    {
        return (info.bid_i == BI) & (info.bid_j == BJ);
    }
};

/// Keep only the pairs whose two primitives share a bid — reads `bid_i`/`bid_j` only.
struct LeafSameBid
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::LeafPredInfoT<I>& info) const noexcept
    {
        return info.bid_i == info.bid_j;
    }
};

/// v1's `query` flavour: reject the subtree of a node whose bid equals the *query's* bid
/// (the reference's `BidOnlyNodeCull` shape, minus its device reads).
struct NodeSameAsQueryBid
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::NodePredInfoT<I>& info) const noexcept
    {
        return info.node_bid != info.query_bid;
    }
};

/// v0's 3-field flavour of the bid cull.
template <IndexT Bid>
struct NodeBidCullV0T
{
    template <typename I>
    auto operator()(const InfoStacklessBVHV0::NodePredInfoT<I>& info) const noexcept
    {
        return info.node_bid != Bid;
    }
};

/// v0's `LeafPred(i, j)`: the reference's `SelfPairPredAdapter` guard, kept as the
/// predicate body so that the pair-ordering contract stays observable.
struct OrderedPairV0
{
    template <typename A, typename B>
    auto operator()(const A& i, const B& j) const noexcept
    {
        return j > i;
    }
};

// ===========================================================================
//  The host traversal simulation (case 2's oracle)
// ===========================================================================
/// The reference's `stacklessSelf` walked on the host over the *exported* node array.
/// Deliberately an independent transcription of the reference's loop (the same
/// `intersects -> node_cull -> leaf/else`, the same `tid < st - intSize` self rule and
/// the same `ordered_pair`), so it can be compared against the device's pair set.
/// `node_ok(query_bid, node_bid, node_cid)` receives the *query* primitive's bid, which
/// is what v1's `NodePredInfo::query_bid` carries.
[[nodiscard]] PairSet host_traverse_self(
    const std::vector<InfoStacklessBVH::Node>& nodes,
    const std::vector<std::int32_t>&           ext_idx,
    const std::vector<AABB>&                   boxes,
    const std::vector<IndexT>&                 bids,
    const std::function<bool(IndexT query_bid, IndexT node_bid, IndexT node_cid)>& node_ok,
    const std::function<bool(IndexT i, IndexT j)>&                                 leaf_ok)
{
    PairSet   pairs;
    const int n        = static_cast<int>(ext_idx.size());
    const int int_size = n - 1;
    for(int t = 0; t < n; ++t)
    {
        const IndexT idx  = ext_idx[static_cast<size_t>(t)];
        const AABB&  bv   = boxes[static_cast<size_t>(idx)];
        const IndexT qbid = bids[static_cast<size_t>(idx)];
        int          st   = 0;
        while(st != -1)
        {
            const auto& node = nodes[static_cast<size_t>(st)];
            if(!to_eigen(node.bound).intersects(bv))
            {
                st = node.escape;
                continue;
            }
            if(!node_ok(qbid, node.bid, node.cid))
            {
                st = node.escape;
                continue;
            }
            if(node.lc == -1)
            {
                if(t < st - int_size)
                {
                    const IndexT other = ext_idx[static_cast<size_t>(st - int_size)];
                    if(leaf_ok(idx, other))
                    {
                        pairs.emplace(std::min(idx, other), std::max(idx, other));
                    }
                }
                st = node.escape;
            }
            else
            {
                st = node.lc;
            }
        }
    }
    return pairs;
}

/// The reference's `stacklessOther` walked on the host (pairs are
/// `(query index, object index)`, un-ordered).
[[nodiscard]] PairSet host_traverse_query(
    const std::vector<InfoStacklessBVH::Node>& nodes,
    const std::vector<std::int32_t>&           ext_idx,
    const std::vector<AABB>&                   queries,
    const std::vector<IndexT>&                 query_bids,
    const std::function<bool(IndexT query_bid, IndexT node_bid, IndexT node_cid)>& node_ok,
    const std::function<bool(IndexT i, IndexT j)>&                                 leaf_ok)
{
    PairSet   pairs;
    const int n        = static_cast<int>(ext_idx.size());
    const int int_size = n - 1;
    for(int q = 0; q < static_cast<int>(queries.size()); ++q)
    {
        const AABB&  bv   = queries[static_cast<size_t>(q)];
        const IndexT qbid = query_bids[static_cast<size_t>(q)];
        int          st   = 0;
        while(st != -1)
        {
            const auto& node = nodes[static_cast<size_t>(st)];
            if(!to_eigen(node.bound).intersects(bv))
            {
                st = node.escape;
                continue;
            }
            if(!node_ok(qbid, node.bid, node.cid))
            {
                st = node.escape;
                continue;
            }
            if(node.lc == -1)
            {
                const IndexT other = ext_idx[static_cast<size_t>(st - int_size)];
                if(leaf_ok(q, other))
                    pairs.emplace(q, other);
                st = node.escape;
            }
            else
            {
                st = node.lc;
            }
        }
    }
    return pairs;
}


// ===========================================================================
//  host Morton codes (case 7's oracle) — the culbvh spelling
// ===========================================================================
[[nodiscard]] std::uint32_t host_expand_bits(std::uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

[[nodiscard]] std::uint32_t host_morton3D(float x, float y, float z)
{
    const auto comp = [](float v)
    {
        const float c = std::fmin(std::fmax(v * 1024.0f, 0.0f), 1023.0f);
        return host_expand_bits(static_cast<std::uint32_t>(c));
    };
    return comp(x) * 4u + comp(y) * 2u + comp(z) * 1u;
}

[[nodiscard]] std::vector<std::uint32_t> host_morton_codes(const std::vector<AABB>& boxes)
{
    AABB scene;
    for(const auto& b : boxes)
        scene.extend(b);
    std::vector<std::uint32_t> codes(boxes.size());
    for(size_t i = 0; i < boxes.size(); ++i)
    {
        const auto c  = boxes[i].center();
        const auto sm = scene.min();
        const auto ss = scene.sizes();
        codes[i]      = host_morton3D((c.x() - sm.x()) / ss.x(),
                                      (c.y() - sm.y()) / ss.y(),
                                      (c.z() - sm.z()) / ss.z());
    }
    return codes;
}

[[nodiscard]] std::vector<std::int32_t> host_sorted_indices(
    const std::vector<std::uint32_t>& codes)
{
    std::vector<std::int32_t> idx(codes.size());
    for(std::int32_t i = 0; i < static_cast<std::int32_t>(idx.size()); ++i)
        idx[static_cast<size_t>(i)] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](std::int32_t a, std::int32_t b)
                     { return codes[static_cast<size_t>(a)]
                              < codes[static_cast<size_t>(b)]; });
    return idx;
}
/// `LeafSameBid`'s cid counterpart for a *query*: reject any node whose cid equals `Cid`
/// (the query-side primitive's cid is not on the info struct in this shape, so the cull is
/// the plain one).
template <IndexT Cid>
struct NodeCidCullQueryT
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::NodePredInfoT<I>& info) const noexcept
    {
        return info.node_cid != Cid;
    }
};

// ===========================================================================
//  case helpers: one *type* per predicate constant (see NodeCidCullT's note)
// ===========================================================================
/// the device-side cid cull vs the host walk and the brute-force-filtered oracle
template <IndexT Cid>
void run_cid_cull(const std::vector<AABB>&                   boxes,
                  const std::vector<IndexT>&                 cids,
                  lud::LCBufferView<LAABB>                   dev,
                  lud::LCBufferView<IndexT>                  dbids,
                  lud::LCBufferView<IndexT>                  dcids,
                  const std::vector<InfoStacklessBVH::Node>& nodes,
                  const std::vector<std::int32_t>&           ext_idx)
{
    InfoStacklessBVH              bvh;
    InfoStacklessBVH::QueryBuffer qbuf;
    bvh.build(dev, dbids, dcids);
    bvh.detect(lud::LCBufferView<IndexT>{}, NodeCidCullT<Cid>{}, KeepAllLeavesV1{}, qbuf);
    frame_stream().flush();
    const auto got = pair_set(qbuf.view());

    const auto simulated =
        host_traverse_self(nodes, ext_idx, boxes, make_bids(boxes.size()),
                           [](IndexT, IndexT, IndexT node_cid)
                           { return node_cid != Cid; },
                           [](IndexT, IndexT) { return true; });
    CHECK(got == simulated);

    const auto want = brute_force_self_node_cull(boxes, ext_idx, cids, Cid);
    CHECK(got == want);
    CHECK(simulated == want);
    std::printf("[bvh_info] cid-cull(%d): pairs=%zu (of %zu)\n",
                static_cast<int>(Cid),
                got.size(),
                brute_force_self(boxes).size());
}

/// the device-side bid cull vs the host walk and the brute-force-filtered oracle
template <IndexT Bid>
void run_bid_cull(const std::vector<AABB>&                   boxes,
                  const std::vector<IndexT>&                 bids,
                  lud::LCBufferView<LAABB>                   dev,
                  lud::LCBufferView<IndexT>                  dbids,
                  lud::LCBufferView<IndexT>                  dcids,
                  const std::vector<InfoStacklessBVH::Node>& nodes,
                  const std::vector<std::int32_t>&           ext_idx)
{
    InfoStacklessBVH              bvh;
    InfoStacklessBVH::QueryBuffer qbuf;
    bvh.build(dev, dbids, dcids);
    bvh.detect(lud::LCBufferView<IndexT>{}, NodeBidCullT<Bid>{}, KeepAllLeavesV1{}, qbuf);
    frame_stream().flush();
    const auto got = pair_set(qbuf.view());

    const auto simulated = host_traverse_self(
        nodes, ext_idx, boxes, bids,
        [](IndexT, IndexT node_bid, IndexT) { return node_bid != Bid; },
        [](IndexT, IndexT) { return true; });
    CHECK(got == simulated);

    const auto want = brute_force_self_node_cull(boxes, ext_idx, bids, Bid);
    CHECK(got == want);
}

/// the device-side v0 bid cull (the 3-field node info) vs the brute-force oracle
template <IndexT Bid>
void run_bid_cull_v0(const std::vector<AABB>&         boxes,
                     const std::vector<IndexT>&       bids,
                     lud::LCBufferView<LAABB>         dev,
                     lud::LCBufferView<IndexT>        dbids,
                     lud::LCBufferView<IndexT>        dcids,
                     const std::vector<std::int32_t>& ext_idx)
{
    InfoStacklessBVHV0              bv0;
    InfoStacklessBVHV0::QueryBuffer r0;
    bv0.build(dev, dbids, dcids);
    bv0.detect(lud::LCBufferView<IndexT>{}, NodeBidCullV0T<Bid>{}, KeepAllLeavesV0{}, r0);
    frame_stream().flush();
    const auto got = pair_set(r0.view());

    const auto want = brute_force_self_node_cull(boxes, ext_idx, bids, Bid);
    CHECK(got == want);
}

/// v1's `LeafPredInfo` (bid_i, bid_j) assignment, on the `detect` flavour
template <IndexT BI, IndexT BJ>
[[nodiscard]] PairSet run_leaf_bid_pair(const std::vector<AABB>& boxes,
                                       const std::vector<IndexT>& bids,
                                       lud::LCBufferView<LAABB>  dev,
                                       lud::LCBufferView<IndexT> dbids,
                                       lud::LCBufferView<IndexT> dcids)
{
    InfoStacklessBVH              bvh;
    InfoStacklessBVH::QueryBuffer qbuf;
    bvh.build(dev, dbids, dcids);
    bvh.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, LeafBidPairT<BI, BJ>{},
               qbuf);
    frame_stream().flush();
    const auto got = pair_set(qbuf.view());

    PairSet want;
    for(const auto& p : brute_force_self(boxes))
        if(bids[static_cast<size_t>(p.first)] == BI
           && bids[static_cast<size_t>(p.second)] == BJ)
            want.insert(p);
    CHECK(got == want);
    std::printf("[bvh_info] leaf bid-pair(%d,%d): pairs=%zu\n",
                static_cast<int>(BI),
                static_cast<int>(BJ),
                got.size());
    return got;
}

/// v1's `LeafPredInfo` (bid_i, bid_j) assignment, on the `query` flavour: `bid_i` is the
/// *query's* bid
template <IndexT BI, IndexT BJ>
[[nodiscard]] PairSet run_leaf_bid_pair_query(const std::vector<AABB>&     boxes,
                             const std::vector<AABB>&     queries,
                             const std::vector<IndexT>&   query_bids,
                             const std::vector<IndexT>&   bids,
                             lud::LCBufferView<LAABB>     dev,
                             lud::LCBufferView<LAABB>     qdev,
                             lud::LCBufferView<IndexT>    dbids,
                             lud::LCBufferView<IndexT>    dcids,
                             lud::LCBufferView<IndexT>    qbids,
                             lud::LCBufferView<IndexT>    qcids)
{
    InfoStacklessBVH              bvh;
    InfoStacklessBVH::QueryBuffer qbuf;
    bvh.build(dev, dbids, dcids);
    bvh.query(qdev, qbids, qcids, lud::LCBufferView<IndexT>{}, KeepAllNodesV1{},
              LeafBidPairT<BI, BJ>{}, qbuf);
    frame_stream().flush();
    const auto got = pair_set(qbuf.view());

    PairSet want;
    for(const auto& p : brute_force_query(queries, boxes))
        if(query_bids[static_cast<size_t>(p.first)] == BI
           && bids[static_cast<size_t>(p.second)] == BJ)
            want.insert(p);
    CHECK(got == want);
    std::printf("[bvh_info] query leaf bid-pair(%d,%d): queries=%zu pairs=%zu\n",
                static_cast<int>(BI),
                static_cast<int>(BJ),
                queries.size(),
                got.size());
    return got;
}

}// namespace info_bvh_test

// ===========================================================================
//  scenes for the sanity-checker case
// ===========================================================================
namespace info_bvh_checker_test
{
using std::vector;

/// A unit cube as a trimesh (8 vertices, 12 triangles), translated by `offset`.
[[nodiscard]] geometry::SimplicialComplex cube_mesh(const Vector3& offset,
                                                    Float          thickness)
{
    vector<Vector3> Vs(8);
    for(int i = 0; i < 8; ++i)
    {
        Vs[static_cast<size_t>(i)] = Vector3{(i & 1) ? 1.0 : 0.0,
                                             (i & 2) ? 1.0 : 0.0,
                                             (i & 4) ? 1.0 : 0.0}
                                     + offset;
    }
    // 12 triangles, consistent outward orientation (not that the checker cares)
    const int tris[12][3] = {{0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6},
                             {0, 1, 5}, {0, 5, 4}, {2, 6, 7}, {2, 7, 3},
                             {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}};
    vector<Vector3i> Ts(12);
    for(int i = 0; i < 12; ++i)
        Ts[static_cast<size_t>(i)] = Vector3i{tris[i][0], tris[i][1], tris[i][2]};

    auto mesh = geometry::trimesh(span<const Vector3>{Vs}, span<const Vector3i>{Ts});
    geometry::label_surface(mesh);

    // every vertex gets the same thickness (the reference's CPU test's shape)
    auto attr_thickness = mesh.vertices().create<Float>(builtin::thickness);
    auto thickness_view = geometry::view(*attr_thickness);
    std::ranges::fill(thickness_view, thickness);
    return mesh;
}

struct CheckerIds
{
    IndexT      geo_0 = -1;
    IndexT      geo_1 = -1;
    IndexT      obj_0 = -1;
    std::string name;
};

/// `count` cube instances of the same object, instance `k` translated by
/// `offset * k`; `count == 1` is the "clean" scene.
[[nodiscard]] CheckerIds build_cube_scene(core::Scene&  scene,
                                          int           count,
                                          const Vector3& offset,
                                          Float          thickness,
                                          const char*    name)
{
    CheckerIds ids;
    ids.name               = name;
    auto object            = scene.objects().create(name);
    ids.obj_0              = object->id();
    auto geo_0             = object->geometries().create(cube_mesh(Vector3::Zero(), thickness));
    ids.geo_0              = geo_0.geometry->id();
    if(count > 1)
    {
        auto geo_1 = object->geometries().create(cube_mesh(offset, thickness));
        ids.geo_1  = geo_1.geometry->id();
    }
    return ids;
}

[[nodiscard]] core::Scene make_config(Float /*thickness*/)
{
    auto config                      = core::Scene::default_config();
    config["sanity_check"]["mode"]   = "quiet";
    config["sanity_check"]["enable"] = true;
    config["contact"]["enable"]      = true;
    return core::Scene{config};
}
}// namespace info_bvh_checker_test
}// namespace uipc::backend::luisa

using namespace uipc::backend::luisa;
using namespace uipc::backend::luisa::info_bvh_test;
using namespace uipc::backend::luisa::info_bvh_checker_test;

// ===========================================================================
//  1) the candidate/visited contract vs brute force
// ===========================================================================
TEST_CASE("info_contract", "[bvh_info]")
{
    test_runtime();

    struct Case
    {
        std::vector<AABB> boxes;
        const char*       name;
        int               query_stride;
    };
    std::vector<Case> cases;
    cases.push_back({make_cloud(1, 131u), "n=1", 1});
    cases.push_back({make_cloud(2, 132u), "n=2", 1});
    cases.push_back({make_cloud(37, 133u), "n=37", 1});
    cases.push_back({make_identical_points(64), "identical", 1});
    cases.push_back({make_duplicate_aabbs(211, 134u), "duplicates", 1});
    cases.push_back({make_line(128), "line", 2});
    cases.push_back({make_cloud(4096, 135u), "n=4096", 4});
    cases.push_back({make_cloud(100000, 136u, 4.0f), "n=100k", 64});

    for(const auto& c : cases)
    {
        auto dev   = upload_aabbs(c.boxes);
        auto bids  = make_bids(c.boxes.size());
        auto cids  = make_cids(c.boxes.size());
        auto dbids = upload_ids(bids);
        auto dcids = upload_ids(cids);

        std::vector<AABB> queries;
        for(size_t i = 0; i < c.boxes.size(); i += static_cast<size_t>(c.query_stride))
            queries.push_back(c.boxes[i]);
        auto qdev  = upload_aabbs(queries);
        auto qbids = upload_ids(make_bids(queries.size()));
        auto qcids = upload_ids(make_cids(queries.size()));

        const auto want_query = brute_force_query(queries, c.boxes);

        // ---- v1: `query` vs the brute-force oracle ----------------------------
        {
            InfoStacklessBVH              bvh;
            InfoStacklessBVH::QueryBuffer qbuf;
            bvh.build(dev.cview(), dbids.cview(), dcids.cview());
            bvh.query(qdev.cview(), qbids.cview(), qcids.cview(),
                      lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{},
                      qbuf);
            frame_stream().flush();
            const auto got = pair_set(qbuf.view());
            CHECK(got == want_query);
            CHECK(got.size() == want_query.size());
            CHECK(qbuf.size() == static_cast<SizeT>(want_query.size()));
            std::printf("[bvh_info] v1 query %s: queries=%zu pairs=%zu (brute=%zu)\n",
                        c.name,
                        queries.size(),
                        got.size(),
                        want_query.size());
        }

        // ---- v0: `query` vs the same oracle ----------------------------------
        {
            InfoStacklessBVHV0              bvh;
            InfoStacklessBVHV0::QueryBuffer qbuf;
            bvh.build(dev.cview(), dbids.cview(), dcids.cview());
            bvh.query(qdev.cview(), qbids.cview(), qcids.cview(),
                      lud::LCBufferView<IndexT>{}, KeepAllNodesV0{}, KeepAllLeavesV0{},
                      qbuf);
            frame_stream().flush();
            const auto got = pair_set(qbuf.view());
            CHECK(got == want_query);
            CHECK(got.size() == want_query.size());
        }

        // ---- v1/v0: `detect` (self pairs) vs brute force (the O(n^2) oracle is only
        //      run up to 4096 primitives; 100k is covered by the query oracle) -----
        if(c.boxes.size() <= 4096u)
        {
            const auto want_self = brute_force_self(c.boxes);

            InfoStacklessBVH              bvh;
            InfoStacklessBVH::QueryBuffer qbuf;
            bvh.build(dev.cview(), dbids.cview(), dcids.cview());
            bvh.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{},
                       qbuf);
            frame_stream().flush();
            const auto got = pair_set(qbuf.view());
            CHECK(got == want_self);
            CHECK(got.size() == want_self.size());
            CHECK(qbuf.size() == static_cast<SizeT>(want_self.size()));
            std::printf("[bvh_info] v1 detect %s: pairs=%zu (brute=%zu)\n",
                        c.name,
                        got.size(),
                        want_self.size());

            InfoStacklessBVHV0              bvh0;
            InfoStacklessBVHV0::QueryBuffer qbuf0;
            bvh0.build(dev.cview(), dbids.cview(), dcids.cview());
            bvh0.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV0{}, KeepAllLeavesV0{},
                        qbuf0);
            frame_stream().flush();
            const auto got0 = pair_set(qbuf0.view());
            CHECK(got0 == want_self);
            CHECK(got0.size() == want_self.size());
        }
    }
}

// ===========================================================================
//  2) the exported node array + the host traversal simulation
// ===========================================================================
TEST_CASE("info_traversal_oracle", "[bvh_info]")
{
    test_runtime();

    const auto        boxes = make_cloud(512, 141u);
    const auto        bids  = make_bids(boxes.size());
    const auto        cids  = make_cids(boxes.size());
    auto              dev   = upload_aabbs(boxes);
    auto              dbids = upload_ids(bids);
    auto              dcids = upload_ids(cids);

    InfoStacklessBVH::Impl impl;
    impl.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
    frame_stream().flush();
    const auto nodes   = download_nodes(impl.nodes.cview());
    const auto ext_idx = download(impl.ext_idx.cview());

    REQUIRE(nodes.size() == 2 * boxes.size() - 1);
    REQUIRE(ext_idx.size() == boxes.size());

    // ---- keep-all: the device's self pairs == the host walk == brute force -------
    {
        InfoStacklessBVH              bvh;
        InfoStacklessBVH::QueryBuffer qbuf;
        bvh.build(dev.cview(), dbids.cview(), dcids.cview());
        bvh.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{},
                   qbuf);
        frame_stream().flush();
        const auto got = pair_set(qbuf.view());

        const auto simulated = host_traverse_self(
            nodes, ext_idx, boxes, bids,
            [](IndexT, IndexT, IndexT) { return true; },
            [](IndexT, IndexT) { return true; });
        const auto want = brute_force_self(boxes);
        CHECK(simulated.size() == want.size());
        CHECK(simulated == want);
        CHECK(got == simulated);
    }

    // ---- a culling node predicate: the device and the host walk agree ------------
    run_cid_cull<0>(boxes, cids, dev.cview(), dbids.cview(), dcids.cview(), nodes, ext_idx);
    run_cid_cull<1>(boxes, cids, dev.cview(), dbids.cview(), dcids.cview(), nodes, ext_idx);
    run_cid_cull<2>(boxes, cids, dev.cview(), dbids.cview(), dcids.cview(), nodes, ext_idx);

    // ---- a bid cull --------------------------------------------------------------
    run_bid_cull<0>(boxes, bids, dev.cview(), dbids.cview(), dcids.cview(), nodes, ext_idx);
    run_bid_cull<3>(boxes, bids, dev.cview(), dbids.cview(), dcids.cview(), nodes, ext_idx);

    // ---- a cull only an *internal* node can satisfy (`node_bid == -1`) -------------
    //      a leaf's bid is never -1 here, so this pins that the propagated sentinel is
    //      what the traversal sees.
    {
        InfoStacklessBVH              bvh;
        InfoStacklessBVH::QueryBuffer qbuf;
        bvh.build(dev.cview(), dbids.cview(), dcids.cview());
        bvh.detect(lud::LCBufferView<IndexT>{}, NodeValidBidOnlyCull{}, KeepAllLeavesV1{},
                   qbuf);
        frame_stream().flush();
        const auto got = pair_set(qbuf.view());

        const auto simulated = host_traverse_self(
            nodes, ext_idx, boxes, bids,
            [](IndexT, IndexT node_bid, IndexT) { return node_bid != -1; },
            [](IndexT, IndexT) { return true; });
        CHECK(got == simulated);
        // ... and the cull really does remove something (the cloud has non-uniform bids)
        CHECK(got.size() < brute_force_self(boxes).size());
        std::printf("[bvh_info] valid-bid-only cull: pairs=%zu (keep-all=%zu)\n",
                    got.size(),
                    brute_force_self(boxes).size());
    }

    // ---- the query flavour: a subset of the boxes as queries ---------------------
    {
        std::vector<AABB> queries;
        for(size_t i = 0; i < boxes.size(); i += 8)
            queries.push_back(boxes[i]);
        auto qdev  = upload_aabbs(queries);
        auto qbids = upload_ids(make_bids(queries.size()));
        auto qcids = upload_ids(make_cids(queries.size()));

        InfoStacklessBVH              bvh;
        InfoStacklessBVH::QueryBuffer qbuf;
        bvh.build(dev.cview(), dbids.cview(), dcids.cview());
        bvh.query(qdev.cview(), qbids.cview(), qcids.cview(),
                  lud::LCBufferView<IndexT>{}, NodeCidCullT<1>{}, KeepAllLeavesV1{},
                  qbuf);
        frame_stream().flush();
        const auto got = pair_set(qbuf.view());

        const auto simulated = host_traverse_query(
            nodes, ext_idx, queries, make_bids(queries.size()),
            [](IndexT, IndexT, IndexT node_cid) { return node_cid != 1; },
            [](IndexT, IndexT) { return true; });
        CHECK(got == simulated);

        PairSet want;
        for(const auto& p : brute_force_query(queries, boxes))
            if(cids[static_cast<size_t>(p.second)] != 1)
                want.insert(p);
        CHECK(got == want);
        std::printf("[bvh_info] query+node-cull: queries=%zu pairs=%zu (brute=%zu)\n",
                    queries.size(),
                    got.size(),
                    want.size());
    }
}

// ===========================================================================
//  3) the bid/cid metadata contract
// ===========================================================================
TEST_CASE("info_metadata", "[bvh_info]")
{
    test_runtime();

    const auto boxes = make_cloud(300, 151u);
    const auto bids  = make_bids(boxes.size());
    const auto cids  = make_cids(boxes.size());
    auto       dev   = upload_aabbs(boxes);
    auto       dbids = upload_ids(bids);
    auto       dcids = upload_ids(cids);

    InfoStacklessBVH::Impl impl;
    impl.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
    frame_stream().flush();
    const auto nodes   = download_nodes(impl.nodes.cview());
    const auto ext_idx = download(impl.ext_idx.cview());

    const int n        = static_cast<int>(boxes.size());
    const int int_size = n - 1;
    REQUIRE(static_cast<int>(nodes.size()) == 2 * n - 1);

    // ---- every leaf carries its primitive's bid/cid ------------------------------
    int leaf_ok = 0;
    for(int i = 0; i < n; ++i)
    {
        const auto& leaf = nodes[static_cast<size_t>(int_size + i)];
        const auto  obj  = ext_idx[static_cast<size_t>(i)];
        if(leaf.lc == -1 && leaf.bid == bids[static_cast<size_t>(obj)]
           && leaf.cid == cids[static_cast<size_t>(obj)])
            ++leaf_ok;
    }
    CHECK(leaf_ok == n);

    // ---- the device's per-node metadata == the exported array's -------------------
    //      (a cull that reads `node_bid`/`node_cid` on *every* visited node, compared
    //      against the host walk over the downloaded array)
    {
        InfoStacklessBVH              bvh;
        InfoStacklessBVH::QueryBuffer qbuf;
        bvh.build(dev.cview(), dbids.cview(), dcids.cview());
        bvh.detect(lud::LCBufferView<IndexT>{}, NodeValidBidOnlyCull{}, KeepAllLeavesV1{},
                   qbuf);
        frame_stream().flush();
        const auto got = pair_set(qbuf.view());

        const auto simulated = host_traverse_self(
            nodes, ext_idx, boxes, bids,
            [](IndexT, IndexT node_bid, IndexT) { return node_bid != -1; },
            [](IndexT, IndexT) { return true; });
        CHECK(got == simulated);

        int informative_bid = 0, informative_cid = 0;
        for(int i = 0; i < int_size; ++i)
        {
            if(nodes[static_cast<size_t>(i)].bid != -1)
                ++informative_bid;
            if(nodes[static_cast<size_t>(i)].cid != -1)
                ++informative_cid;
        }
        std::printf("[bvh_info] metadata: n=%d internal=%d informative_bid=%d "
                                            "informative_cid=%d valid_bid_only_cull_pairs=%zu\n",
                    n,
                    int_size,
                    informative_bid,
                    informative_cid,
                    got.size());
        CHECK(informative_bid > 0);   // the cloud has non-trivial bid structure
        CHECK(informative_bid < int_size);
        CHECK(informative_cid > 0);
    }

    // ---- the decisive form of the rule: a *uniform* bid/cid must reach the root ------
    {
        const std::vector<IndexT> same_bids(boxes.size(), IndexT(3));
        const std::vector<IndexT> same_cids(boxes.size(), IndexT(0));
        auto                      db = upload_ids(same_bids);
        auto                      dc = upload_ids(same_cids);

        InfoStacklessBVH::Impl uni;
        uni.build(dev.cview(), db.cview(), dc.cview(), frame_stream());
        frame_stream().flush();
        const auto uni_nodes = download_nodes(uni.nodes.cview());
        REQUIRE(uni_nodes.size() == 2 * boxes.size() - 1);
        // node 0 is the root and it is internal (n > 1): its range is every leaf
        CHECK(uni_nodes[0].lc != -1);
        CHECK(uni_nodes[0].bid == 3);
        CHECK(uni_nodes[0].cid == 0);

        // ... so a cull on that common cid rejects the whole tree: no pair is reported
        InfoStacklessBVH              bvh;
        InfoStacklessBVH::QueryBuffer qbuf;
        bvh.build(dev.cview(), db.cview(), dc.cview());
        bvh.detect(lud::LCBufferView<IndexT>{}, NodeCidCullT<0>{}, KeepAllLeavesV1{},
                   qbuf);
        frame_stream().flush();
        CHECK(qbuf.size() == 0u);
        CHECK(pair_set(qbuf.view()).empty());
        CHECK(brute_force_self(boxes).size() > 0u);  // the cloud really does overlap

        // the positive control for the `-1` sentinel: on a uniform cloud no internal node
        // ever gets -1, so a cull on it keeps every candidate
        InfoStacklessBVH::QueryBuffer qinv;
        bvh.detect(lud::LCBufferView<IndexT>{}, NodeValidBidOnlyCull{}, KeepAllLeavesV1{},
                   qinv);
        frame_stream().flush();
        CHECK(pair_set(qinv.view()) == brute_force_self(boxes));
    }

    // ---- ... and the "or -1" half: a uniform bid with a non-uniform cid keeps the bid
    //      at the root and collapses the cid to -1 -----------------------------------
    {
        const std::vector<IndexT> same_bids(boxes.size(), IndexT(5));
        auto                      db = upload_ids(same_bids);

        InfoStacklessBVH::Impl mix;
        mix.build(dev.cview(), db.cview(), dcids.cview(), frame_stream());
        frame_stream().flush();
        const auto mix_nodes = download_nodes(mix.nodes.cview());
        CHECK(mix_nodes[0].bid == 5);
        CHECK(mix_nodes[0].cid == -1);

        // the bid cull therefore still rejects every *target*, while the cid cull only
        // removes the pairs whose (larger-sorted) target has cid 0
        const auto mix_ext_idx = download(mix.ext_idx.cview());
        InfoStacklessBVH              bvh;
        InfoStacklessBVH::QueryBuffer q1, q2;
        bvh.build(dev.cview(), db.cview(), dcids.cview());
        bvh.detect(lud::LCBufferView<IndexT>{}, NodeBidCullT<5>{}, KeepAllLeavesV1{}, q1);
        bvh.detect(lud::LCBufferView<IndexT>{}, NodeCidCullT<0>{}, KeepAllLeavesV1{}, q2);
        frame_stream().flush();
        CHECK(q1.size() == 0u);
        CHECK(pair_set(q2.view())
              == brute_force_self_node_cull(boxes, mix_ext_idx, cids, 0));
        CHECK(pair_set(q2.view()).size() < brute_force_self(boxes).size());
    }

    // ---- the leaf predicates see the resolved (bid_i, bid_j) in the right order ----
    //      `detect` orders the pair, so (bid_i, bid_j) == (bid of the *smaller object
    //      index*, bid of the larger) — the two orderings therefore select two different
    //      (non-empty) pair sets, and both match their oracles.
    {
        const auto a = run_leaf_bid_pair<3, 5>(boxes, bids, dev.cview(), dbids.cview(),
                                               dcids.cview());
        const auto b = run_leaf_bid_pair<5, 3>(boxes, bids, dev.cview(), dbids.cview(),
                                               dcids.cview());
        CHECK(!a.empty());
        CHECK(!b.empty());
        CHECK(a != b);  // the assignment is order-sensitive
    }

    // ---- `bid_i == bid_j`: the reference's own `LeafPred` shape --------------------
    {
        InfoStacklessBVH              bvh;
        InfoStacklessBVH::QueryBuffer qbuf;
        bvh.build(dev.cview(), dbids.cview(), dcids.cview());
        bvh.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, LeafSameBid{}, qbuf);
        frame_stream().flush();
        const auto got = pair_set(qbuf.view());

        PairSet want;
        for(const auto& p : brute_force_self(boxes))
            if(bids[static_cast<size_t>(p.first)] == bids[static_cast<size_t>(p.second)])
                want.insert(p);
        CHECK(got == want);
        CHECK(got.size() == want.size());
    }

    // ---- the query flavour resolves the *query's* bid into `bid_i` -----------------
    {
        std::vector<AABB> queries;
        for(size_t i = 0; i < boxes.size(); i += 16)
            queries.push_back(boxes[i]);
        auto qdev  = upload_aabbs(queries);
        auto qbids = upload_ids(make_bids(queries.size()));
        auto qcids = upload_ids(make_cids(queries.size()));

        const auto a = run_leaf_bid_pair_query<2, 6>(boxes, queries,
                                                     make_bids(queries.size()), bids,
                                                     dev.cview(), qdev.cview(),
                                                     dbids.cview(), dcids.cview(),
                                                     qbids.cview(), qcids.cview());
        // the (6, 2) assignment selects a different pair (the query's bid goes to bid_i)
        const auto b = run_leaf_bid_pair_query<6, 2>(boxes, queries,
                                                     make_bids(queries.size()), bids,
                                                     dev.cview(), qdev.cview(),
                                                     dbids.cview(), dcids.cview(),
                                                     qbids.cview(), qcids.cview());
        CHECK(!a.empty());
        CHECK(!b.empty());
        CHECK(a != b);
    }
}

// ===========================================================================
//  4) the four structures agree
// ===========================================================================
TEST_CASE("info_cross_structure", "[bvh_info]")
{
    test_runtime();

    const auto boxes = make_cloud(2048, 161u);
    const auto bids  = make_bids(boxes.size());
    const auto cids  = make_cids(boxes.size());
    auto       dev   = upload_aabbs(boxes);
    auto       dbids = upload_ids(bids);
    auto       dcids = upload_ids(cids);

    InfoStacklessBVH              v1;
    InfoStacklessBVH::QueryBuffer q1;
    v1.build(dev.cview(), dbids.cview(), dcids.cview());
    v1.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{}, q1);

    InfoStacklessBVHV0              v0;
    InfoStacklessBVHV0::QueryBuffer q0;
    v0.build(dev.cview(), dbids.cview(), dcids.cview());
    v0.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV0{}, KeepAllLeavesV0{}, q0);

    StacklessBVH              s;
    StacklessBVH::QueryBuffer qs;
    s.build(dev.cview());
    s.detect(StacklessBVH::DefaultQueryCallback{}, qs);

    AtomicCountingLBVH              a;
    AtomicCountingLBVH::QueryBuffer qa;
    a.build(dev.cview());
    a.detect([](const auto&, const auto&) noexcept { return true; }, qa);

    frame_stream().flush();
    const auto s1 = pair_set(q1.view());
    const auto s0 = pair_set(q0.view());
    const auto ss = pair_set(qs.view());
    const auto sa = pair_set(qa.view());
    const auto want = brute_force_self(boxes);

    CHECK(s1.size() == want.size());
    CHECK(s1 == want);
    CHECK(s0 == s1);
    CHECK(ss == s1);
    CHECK(sa == s1);
    std::printf("[bvh_info] cross(2048): v1=%zu v0=%zu stackless=%zu counting=%zu "
                "brute=%zu\n",
                s1.size(),
                s0.size(),
                ss.size(),
                sa.size(),
                want.size());

    // the 100k cloud: v1, v0 and the plain stackless structure must agree exactly
    {
        const auto boxes2 = make_cloud(100000, 162u, 4.0f);
        const auto b2     = make_bids(boxes2.size());
        const auto c2     = make_cids(boxes2.size());
        auto       d2     = upload_aabbs(boxes2);
        auto       db2    = upload_ids(b2);
        auto       dc2    = upload_ids(c2);

        InfoStacklessBVH              bv1;
        InfoStacklessBVH::QueryBuffer r1;
        bv1.build(d2.cview(), db2.cview(), dc2.cview());
        bv1.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{}, r1);

        InfoStacklessBVHV0              bv0;
        InfoStacklessBVHV0::QueryBuffer r0;
        bv0.build(d2.cview(), db2.cview(), dc2.cview());
        bv0.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV0{}, KeepAllLeavesV0{}, r0);

        StacklessBVH              st;
        StacklessBVH::QueryBuffer rst;
        st.build(d2.cview());
        st.detect(StacklessBVH::DefaultQueryCallback{}, rst);

        frame_stream().flush();
        const auto x1  = pair_set(r1.view());
        const auto x0  = pair_set(r0.view());
        const auto xs  = pair_set(rst.view());
        CHECK(x1.size() == x0.size());
        CHECK(x1 == x0);
        CHECK(x1 == xs);
        std::printf("[bvh_info] cross(100k): v1=%zu v0=%zu stackless=%zu\n",
                    x1.size(),
                    x0.size(),
                    xs.size());
    }
}

// ===========================================================================
//  5) v1 vs v0, pinned one difference at a time
// ===========================================================================
TEST_CASE("info_v1_vs_v0", "[bvh_info]")
{
    test_runtime();

    // a cloud with *repeated* bids/cids so that the internal-node metadata is exercised
    const auto boxes = make_duplicate_aabbs(400, 171u);
    const auto bids  = make_bids(boxes.size());
    const auto cids  = make_cids(boxes.size());
    auto       dev   = upload_aabbs(boxes);
    auto       dbids = upload_ids(bids);
    auto       dcids = upload_ids(cids);

    // ---- (d1) the two *builds* produce byte-identical node arrays -----------------
    //      v1 never initialises `int_bid`/`int_cid`, v0 initialises them to -1: the
    //      ported builds must still agree element by element (every read of an internal
    //      child's metadata happens after that child completed its merge).
    {
        InfoStacklessBVH::Impl impl1;
        impl1.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
        InfoStacklessBVHV0::Impl impl0;
        impl0.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
        frame_stream().flush();

        const auto n1 = download_nodes(impl1.nodes.cview());
        const auto n0 = download_nodes_v0(impl0.nodes.cview());
        REQUIRE(n1.size() == n0.size());
        CHECK(n1.size() == 2 * boxes.size() - 1);

        int identical = 0;
        int bid_mismatch = 0, cid_mismatch = 0, box_mismatch = 0, link_mismatch = 0;
        for(size_t i = 0; i < n1.size(); ++i)
        {
            const bool box_ok = (n1[i].bound.min(0) == n0[i].bound.min(0))
                                && (n1[i].bound.min(1) == n0[i].bound.min(1))
                                && (n1[i].bound.min(2) == n0[i].bound.min(2))
                                && (n1[i].bound.max(0) == n0[i].bound.max(0))
                                && (n1[i].bound.max(1) == n0[i].bound.max(1))
                                && (n1[i].bound.max(2) == n0[i].bound.max(2));
            const bool link_ok = (n1[i].lc == n0[i].lc) && (n1[i].escape == n0[i].escape);
            bid_mismatch += (n1[i].bid != n0[i].bid) ? 1 : 0;
            cid_mismatch += (n1[i].cid != n0[i].cid) ? 1 : 0;
            box_mismatch += box_ok ? 0 : 1;
            link_mismatch += link_ok ? 0 : 1;
            identical += (box_ok && link_ok && n1[i].bid == n0[i].bid
                          && n1[i].cid == n0[i].cid) ?
                             1 :
                             0;
        }
        CHECK(identical == static_cast<int>(n1.size()));
        CHECK(bid_mismatch == 0);
        CHECK(cid_mismatch == 0);
        CHECK(box_mismatch == 0);
        CHECK(link_mismatch == 0);
        std::printf("[bvh_info] v1/v0 node arrays: %d identical of %zu\n",
                    identical,
                    n1.size());
    }

    // ---- (d2) the same candidate sets for the trivial predicates -------------------
    {
        InfoStacklessBVH              bv1;
        InfoStacklessBVH::QueryBuffer r1;
        bv1.build(dev.cview(), dbids.cview(), dcids.cview());
        bv1.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{}, r1);

        InfoStacklessBVHV0              bv0;
        InfoStacklessBVHV0::QueryBuffer r0;
        bv0.build(dev.cview(), dbids.cview(), dcids.cview());
        bv0.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV0{}, KeepAllLeavesV0{}, r0);

        frame_stream().flush();
        const auto x1 = pair_set(r1.view());
        const auto x0 = pair_set(r0.view());
        CHECK(x1 == x0);
        CHECK(x1 == brute_force_self(boxes));
    }

    // ---- (d3) the node-predicate field sets ---------------------------------------
    //      v1's `NodePredInfo` carries the *pre-loaded* query bid (`query_bid`); the
    //      reference's `BidOnlyNodeCull` shape therefore reduces to a pure scalar test.
    {
        InfoStacklessBVH              bv1;
        InfoStacklessBVH::QueryBuffer r1;
        bv1.build(dev.cview(), dbids.cview(), dcids.cview());
        bv1.detect(lud::LCBufferView<IndexT>{}, NodeSameAsQueryBid{}, KeepAllLeavesV1{},
                   r1);
        frame_stream().flush();
        const auto got = pair_set(r1.view());

        // oracle: the same rule evaluated by the host walk over the exported array
        InfoStacklessBVH::Impl impl;
        impl.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
        frame_stream().flush();
        const auto nodes   = download_nodes(impl.nodes.cview());
        const auto ext_idx = download(impl.ext_idx.cview());
        const auto simulated = host_traverse_self(
            nodes, ext_idx, boxes, bids,
            [](IndexT qbid, IndexT node_bid, IndexT) { return node_bid != qbid; },
            [](IndexT, IndexT) { return true; });
        CHECK(got == simulated);
        std::printf("[bvh_info] v1 query_bid cull: pairs=%zu (keep-all=%zu)\n",
                    got.size(),
                    host_traverse_self(nodes, ext_idx, boxes, bids,
                                       [](IndexT, IndexT, IndexT) { return true; },
                                       [](IndexT, IndexT) { return true; })
                        .size());
    }

    // ---- (d3b) v0's 3-field node info pins the same cull by a host constant ---------
    {
        InfoStacklessBVH::Impl impl;
        impl.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
        frame_stream().flush();
        const auto ext_idx = download(impl.ext_idx.cview());
        run_bid_cull_v0<3>(boxes, bids, dev.cview(), dbids.cview(), dcids.cview(),
                           ext_idx);
    }

    // ---- (d4) the split launch/retry API (v1) vs the inline one (v0) ----------------
    {
        // v1: `launch_detect` + `prepare_query_result` with a deliberately tiny queue
        InfoStacklessBVH              bv1;
        InfoStacklessBVH::QueryBuffer r1;
        r1.m_pairs.resize(1);
        bv1.build(dev.cview(), dbids.cview(), dcids.cview());
        bv1.launch_detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{},
                          r1);
        const auto count = lud::read_host_sync(frame_stream(), r1.m_cpNum.cview());
        const bool retry = bv1.prepare_query_result(r1, count);
        if(retry)
            bv1.launch_detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{},
                              KeepAllLeavesV1{}, r1);
        frame_stream().flush();
        CHECK(retry == (count > 1));
        CHECK(r1.size() == static_cast<SizeT>(count));
        CHECK(pair_set(r1.view()) == brute_force_self(boxes));
        std::printf("[bvh_info] v1 prepare_query_result: count=%d retry=%d capacity=%zu\n",
                    count,
                    retry ? 1 : 0,
                    r1.m_pairs.size());

        // v0: the same input through its inline retry
        InfoStacklessBVHV0              bv0;
        InfoStacklessBVHV0::QueryBuffer r0;
        r0.m_pairs.resize(1);
        bv0.build(dev.cview(), dbids.cview(), dcids.cview());
        bv0.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV0{}, KeepAllLeavesV0{}, r0);
        frame_stream().flush();
        CHECK(r0.size() == static_cast<SizeT>(count));
        CHECK(pair_set(r0.view()) == brute_force_self(boxes));
    }

    // ---- (d5) v0's `LeafPred(i, j)` sees the ordered pair ---------------------------
    //      The reference's v0 `SelfPairPredAdapter` filters `j <= i`, so a predicate
    //      that reads its two indices can still observe the ordering contract.
    {
        InfoStacklessBVHV0              bv0;
        InfoStacklessBVHV0::QueryBuffer r0;
        bv0.build(dev.cview(), dbids.cview(), dcids.cview());
        bv0.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV0{}, OrderedPairV0{}, r0);
        frame_stream().flush();
        CHECK(pair_set(r0.view()) == brute_force_self(boxes));
    }
}

// ===========================================================================
//  6) determinism + the capacity contract
// ===========================================================================
TEST_CASE("info_determinism", "[bvh_info]")
{
    test_runtime();

    const auto boxes = make_cloud(600, 181u);
    const auto bids  = make_bids(boxes.size());
    const auto cids  = make_cids(boxes.size());
    auto       dev   = upload_aabbs(boxes);
    auto       dbids = upload_ids(bids);
    auto       dcids = upload_ids(cids);

    // ---- two objects, two builds, identical node arrays ---------------------------
    {
        InfoStacklessBVH::Impl impl_a;
        InfoStacklessBVH::Impl impl_b;
        impl_a.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
        impl_b.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
        frame_stream().flush();
        const auto a = download_nodes(impl_a.nodes.cview());
        const auto b = download_nodes(impl_b.nodes.cview());
        int        same = 0;
        for(size_t i = 0; i < a.size(); ++i)
            if(a[i].lc == b[i].lc && a[i].escape == b[i].escape && a[i].bid == b[i].bid
               && a[i].cid == b[i].cid && a[i].bound.min(0) == b[i].bound.min(0)
               && a[i].bound.max(2) == b[i].bound.max(2))
                ++same;
        CHECK(same == static_cast<int>(a.size()));

        // a rebuild of the same object reproduces the same array
        impl_a.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
        frame_stream().flush();
        const auto c = download_nodes(impl_a.nodes.cview());
        int        same2 = 0;
        for(size_t i = 0; i < a.size(); ++i)
            if(a[i].lc == c[i].lc && a[i].escape == c[i].escape && a[i].bid == c[i].bid
               && a[i].cid == c[i].cid)
                ++same2;
        CHECK(same2 == static_cast<int>(a.size()));
    }

    // ---- the QueryBuffer capacity contract ----------------------------------------
    {
        InfoStacklessBVH              bv;
        InfoStacklessBVH::QueryBuffer qbuf;
        // the reference's initial capacity: 50 * 1024 pairs
        CHECK(qbuf.m_pairs.size() == 50u * 1024u);
        CHECK(qbuf.size() == 0u);
        CHECK(bv.config().reserve_ratio == 1.2);

        bv.build(dev.cview(), dbids.cview(), dcids.cview());
        bv.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{}, qbuf);
        frame_stream().flush();
        const auto pairs = brute_force_self(boxes);
        CHECK(qbuf.size() == static_cast<SizeT>(pairs.size()));
        CHECK(qbuf.m_pairs.size() == 50u * 1024u);  // no retry was needed

        // the same with a queue that cannot hold the result: the retry grows it to
        // `count * reserve_ratio` (the reference's `prepare_query_result`)
        InfoStacklessBVH::QueryBuffer small;
        small.m_pairs.resize(1);
        bv.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{}, small);
        frame_stream().flush();
        CHECK(small.size() == static_cast<SizeT>(pairs.size()));
        CHECK(small.m_pairs.size()
              == static_cast<size_t>(static_cast<Float>(pairs.size()) * 1.2));
        CHECK(pair_set(small.view()) == pairs);

        // a caller-chosen reserve_ratio is honoured
        InfoStacklessBVH bv2{InfoStacklessBVH::Config{}};
        bv2.config().reserve_ratio = 2.0f;
        bv2.build(dev.cview(), dbids.cview(), dcids.cview());
        InfoStacklessBVH::QueryBuffer q2;
        q2.m_pairs.resize(1);
        bv2.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{}, q2);
        frame_stream().flush();
        CHECK(q2.size() == static_cast<SizeT>(pairs.size()));
        CHECK(q2.m_pairs.size() == static_cast<size_t>(static_cast<Float>(pairs.size()) * 2.0f));
    }

    // ---- repeated runs give the same pair *set* ------------------------------------
    {
        InfoStacklessBVH              bv;
        InfoStacklessBVH::QueryBuffer r1, r2;
        bv.build(dev.cview(), dbids.cview(), dcids.cview());
        bv.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{}, r1);
        bv.detect(lud::LCBufferView<IndexT>{}, KeepAllNodesV1{}, KeepAllLeavesV1{}, r2);
        frame_stream().flush();
        CHECK(pair_set(r1.view()) == pair_set(r2.view()));
        CHECK(r1.size() == r2.size());
    }
}

// ===========================================================================
//  7) the build's derived arrays, device vs an independent host derivation
// ===========================================================================
TEST_CASE("info_device_vs_host", "[bvh_info]")
{
    test_runtime();

    const auto boxes = make_cloud(257, 191u);
    const auto bids  = make_bids(boxes.size());
    const auto cids  = make_cids(boxes.size());
    auto       dev   = upload_aabbs(boxes);
    auto       dbids = upload_ids(bids);
    auto       dcids = upload_ids(cids);

    InfoStacklessBVH::Impl impl;
    impl.build(dev.cview(), dbids.cview(), dcids.cview(), frame_stream());
    frame_stream().flush();

    // ---- the Morton codes ----------------------------------------------------------
    const auto want_codes = host_morton_codes(boxes);
    const auto got_codes  = download(impl.mtcode.cview());
    REQUIRE(got_codes.size() == want_codes.size());
    int code_ok = 0;
    for(size_t i = 0; i < got_codes.size(); ++i)
        if(got_codes[i] == want_codes[i])
            ++code_ok;
    CHECK(code_ok == static_cast<int>(got_codes.size()));

    // ---- the stable sort ----------------------------------------------------------
    const auto want_sorted = host_sorted_indices(want_codes);
    const auto got_sorted_id = download(impl.sorted_id.cview());
    const auto got_sorted_codes = download(impl.sorted_mtcode.cview());
    REQUIRE(got_sorted_id.size() == want_sorted.size());
    int sort_ok = 0;
    for(size_t i = 0; i < got_sorted_id.size(); ++i)
        if(got_sorted_id[i] == want_sorted[i]
           && got_sorted_codes[i] == want_codes[static_cast<size_t>(want_sorted[i])])
            ++sort_ok;
    CHECK(sort_ok == static_cast<int>(got_sorted_id.size()));

    // ---- `ext_idx` is the inverse mapping (a bijection) -----------------------------
    const auto ext_idx = download(impl.ext_idx.cview());
    REQUIRE(ext_idx.size() == boxes.size());
    {
        const auto prim_map = download(impl.primMap.cview());
        int        inv_ok   = 0;
        for(size_t i = 0; i < ext_idx.size(); ++i)
            if(ext_idx[static_cast<size_t>(prim_map[i])] == static_cast<IndexT>(i))
                ++inv_ok;
        CHECK(inv_ok == static_cast<int>(ext_idx.size()));
    }

    // ---- the depth scan and the internal-node permutation --------------------------
    const auto depths   = download(impl.count.cview());
    const auto offsets  = download(impl.offsetTable.cview());
    const auto tk_map   = download(impl.tkMap.cview());
    const auto n        = static_cast<IndexT>(boxes.size());
    REQUIRE(depths.size() == boxes.size());
    REQUIRE(offsets.size() == boxes.size());
    REQUIRE(tk_map.size() == boxes.size());

    // every node is counted exactly once => the depths sum to the internal-node count
    std::uint64_t depth_sum = 0;
    for(auto d : depths)
        depth_sum += d;
    CHECK(depth_sum == static_cast<std::uint64_t>(boxes.size() - 1));

    // `offsetTable` is the exclusive prefix sum of `count`
    {
        std::uint64_t running = 0;
        int           ok      = 0;
        for(size_t i = 0; i < offsets.size(); ++i)
        {
            if(offsets[i] == static_cast<std::uint32_t>(running))
                ++ok;
            running += depths[i];
        }
        CHECK(ok == static_cast<int>(offsets.size()));
        CHECK(running == depth_sum);
    }

    // `tkMap` maps the *unordered* internal index to the reordered slot: a bijection onto
    // [0, n-1) with exactly the internal nodes covered
    {
        std::vector<int> seen(boxes.size(), 0);
        for(IndexT i = 0; i + 1 < n; ++i)
        {
            const auto v = tk_map[static_cast<size_t>(i)];
            if(v >= 0 && v < static_cast<IndexT>(boxes.size()))
                seen[static_cast<size_t>(v)] += 1;
        }
        int bijection = 0;
        for(IndexT i = 0; i + 1 < n; ++i)
            if(seen[static_cast<size_t>(i)] == 1)
                ++bijection;
        CHECK(bijection == static_cast<int>(boxes.size() - 1));
    }
}

// ===========================================================================
//  8) the two ported sanity checkers
// ===========================================================================
namespace
{
std::string checker_workspace(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// pull the `<float>` that follows `key=` out of a diagnostic
[[nodiscard]] bool parse_float_after(const std::string& s,
                                     std::string_view   key,
                                     double&            out)
{
    const auto pos = s.find(key);
    if(pos == std::string::npos)
        return false;
    const auto first = pos + key.size();
    const auto last  = s.find_first_of(",)\n", first);
    if(last == std::string::npos)
        return false;
    try
    {
        out = std::stod(s.substr(first, last - first));
        return true;
    }
    catch(...)
    {
        return false;
    }
}
}// namespace

TEST_CASE("info_sanity_checkers", "[bvh_info]")
{
    test_runtime();

    const Float thickness = 0.002;

    // ---- the intersection checker (uid 1): two cubes translated by (0.5, 0.5, 0.5)
    //      so that the upper cube's vertical edges pierce the lower cube's top-face
    //      interiors (a robust, non-degenerate crossing) -----------------------------
    {
        const auto offset = Vector3{0.5, 0.5, 0.5};
        auto       scene  = make_config(thickness);
        auto       ids    = build_cube_scene(scene, 2, offset, thickness, "t71b_overlap");

        uipc::core::Engine engine{"luisa",
                                  checker_workspace("uipc_luisa_t71b_info_overlap")};
        uipc::core::World  world{engine};
        world.init(scene);

        CHECK(!world.is_valid());

        auto& errors = world.sanity_checker().errors();
        REQUIRE(errors.find(1) != errors.end());
        auto& msg = errors.at(1);
        CHECK(msg->result() == uipc::core::SanityCheckResult::Error);
        CHECK(msg->name().find("luisa") != std::string_view::npos);
        CHECK(msg->name().find("cuda") == std::string_view::npos);

        const std::string text{msg->message()};
        std::printf("[bvh_info] intersection diagnostic:\n%s\n", text.c_str());

        // the object/geometry naming is the reference's format string, byte for byte
        std::string expected_prefix;
        fmt::format_to(std::back_inserter(expected_prefix),
                       "Geometry({}) in Object[{}({})] intersects with Geometry({}) in "
                       "Object[{}({})]\n",
                       ids.geo_0,
                       ids.name,
                       ids.obj_0,
                       ids.geo_1,
                       ids.name,
                       ids.obj_0);
        CHECK(text.rfind(expected_prefix, 0) == 0);

        // the "Intersected mesh has ..." line must agree with the mesh the checker
        // attached (its own construction, not the CPU checker's)
        REQUIRE(msg->geometries().size() == 1u);
        REQUIRE(msg->geometries().find("intersected_mesh") != msg->geometries().end());
        auto imesh =
            msg->geometries().at("intersected_mesh")->as<uipc::geometry::SimplicialComplex>();
        REQUIRE(imesh != nullptr);
        CHECK(imesh->edges().size() > 0);
        CHECK(imesh->triangles().size() > 0);

        std::size_t nv = 0, ne = 0, nt = 0;
        {
            const auto pos = text.find("Intersected mesh has ");
            REQUIRE(pos != std::string::npos);
            REQUIRE(std::sscanf(text.c_str() + pos,
                                "Intersected mesh has %zu vertices, %zu edges, and %zu "
                                "triangles.",
                                &nv,
                                &ne,
                                &nt)
                    == 3);
        }
        CHECK(nv == imesh->vertices().size());
        CHECK(ne == imesh->edges().size());
        CHECK(nt == imesh->triangles().size());
        CHECK(text.find("Create mesh [intersected_mesh<SimplicialComplex>] for "
                        "post-processing.")
              != std::string::npos);

        // NOTE (pinned): the reference's *distance* check only tests vertex-triangle and
        // edge-edge proximity, and an axis-aligned `(0.5, 0.5, 0.5)` interpenetration
        // crosses edges *through triangle interiors* while leaving every vertex-triangle
        // and edge-edge pair far apart (the crossing edges have two fixed coordinates of
        // 0/1 vs 0.5/1.5, so they never meet another edge).  uid 3 therefore legitimately
        // reports nothing here — the intersection checker is what catches this scene.
        CHECK(errors.find(3) == errors.end());
    }

    // ---- the distance checker (uid 3): two cubes separated by 1.5 * thickness, i.e.
    //      closer than the summed thickness (2 * thickness) but *not* intersecting ----
    {
        const auto offset = Vector3{0.0, 1.0 + 1.5 * thickness, 0.0};
        auto       scene  = make_config(thickness);
        auto       ids = build_cube_scene(scene, 2, offset, thickness, "t71b_tooclose");

        uipc::core::Engine engine{"luisa",
                                  checker_workspace("uipc_luisa_t71b_info_tooclose")};
        uipc::core::World  world{engine};
        world.init(scene);

        CHECK(!world.is_valid());

        auto& errors = world.sanity_checker().errors();
        REQUIRE(errors.find(3) != errors.end());
        auto& msg = errors.at(3);
        CHECK(msg->result() == uipc::core::SanityCheckResult::Error);
        CHECK(msg->name().find("luisa") != std::string_view::npos);

        const std::string text{msg->message()};
        std::printf("[bvh_info] distance diagnostic:\n%s\n", text.c_str());

        // the geometry pair is sorted (the reference's `if(GeoIdL > GeoIdR) swap`)
        std::string expected_prefix;
        fmt::format_to(std::back_inserter(expected_prefix),
                       "Geometry({}) in Object[{}({})] is too close (distance=",
                       std::min(ids.geo_0, ids.geo_1),
                       ids.name,
                       ids.obj_0);
        CHECK(text.rfind(expected_prefix, 0) == 0);
        std::string expected_suffix;
        fmt::format_to(std::back_inserter(expected_suffix),
                       ") to Geometry({}) in Object[{}({})]\n",
                       std::max(ids.geo_0, ids.geo_1),
                       ids.name,
                       ids.obj_0);
        CHECK(text.find(expected_suffix) != std::string::npos);

        // the reported distance is the true minimum surface distance (1.5 * thickness)
        // and the reported thickness the summed one (2 * thickness); both are the
        // checker's own arithmetic, so this is a *pin*, not a tolerance guess
        double reported_distance = 0.0, reported_thickness = 0.0;
        REQUIRE(parse_float_after(text, "distance=", reported_distance));
        REQUIRE(parse_float_after(text, "thickness=", reported_thickness));
        CHECK(std::abs(reported_distance - 1.5 * thickness) < 1e-9);
        const double expected_thickness = std::sqrt((2.0 * thickness) * (2.0 * thickness));
        CHECK(std::abs(reported_thickness - expected_thickness) < 1e-15);

        REQUIRE(msg->geometries().find("close_mesh") != msg->geometries().end());
        auto cmesh = msg->geometries().at("close_mesh")->as<uipc::geometry::SimplicialComplex>();
        REQUIRE(cmesh != nullptr);

        std::size_t nv = 0, ne = 0, nt = 0;
        {
            const auto pos = text.find("Close mesh has ");
            REQUIRE(pos != std::string::npos);
            REQUIRE(std::sscanf(text.c_str() + pos,
                                "Close mesh has %zu vertices, %zu edges, and %zu triangles.",
                                &nv,
                                &ne,
                                &nt)
                    == 3);
        }
        CHECK(nv == cmesh->vertices().size());
        CHECK(ne == cmesh->edges().size());
        CHECK(nt == cmesh->triangles().size());
        // Every scene vertex is flagged: the 0.003 gap is a *whole triangle / whole
        // edge* away from the culled feature, and the reference flags every vertex of a
        // triangle (or both endpoints of an edge) that participates in a violation —
        // e.g. A's face triangles and A's vertical edges run from the near rim all the
        // way to the far rim (the pinned value is the measured one).
        CHECK(nv == 16);
        CHECK(nv <= 16);
        std::printf("[bvh_info] close mesh: %zu/%zu/%zu\n", nv, ne, nt);
        CHECK(text.find("Create mesh [close_mesh<SimplicialComplex>] for post-processing.")
              != std::string::npos);

        // no intersection: the cubes are disjoint
        CHECK(errors.find(1) == errors.end());
    }

    // ---- both checkers pass on a single instance ------------------------------------
    {
        auto scene = make_config(thickness);
        auto ids   = build_cube_scene(scene, 1, Vector3::Zero(), thickness, "t71b_clean");

        uipc::core::Engine engine{"luisa", checker_workspace("uipc_luisa_t71b_info_clean")};
        uipc::core::World  world{engine};
        world.init(scene);

        CHECK(world.is_valid());
        auto& errors = world.sanity_checker().errors();
        CHECK(errors.find(1) == errors.end());
        CHECK(errors.find(3) == errors.end());
    }

    // ---- far apart: neither checker reports -----------------------------------------
    {
        auto scene = make_config(thickness);
        auto ids =
            build_cube_scene(scene, 2, Vector3{0.0, 4.0, 0.0}, thickness, "t71b_far");

        uipc::core::Engine engine{"luisa", checker_workspace("uipc_luisa_t71b_info_far")};
        uipc::core::World  world{engine};
        world.init(scene);

        CHECK(world.is_valid());
        auto& errors = world.sanity_checker().errors();
        CHECK(errors.find(1) == errors.end());
        CHECK(errors.find(3) == errors.end());
    }
}

// ===========================================================================
//  9) accounting
// ===========================================================================
TEST_CASE("info_summary", "[bvh_info]")
{
    std::printf("[bvh_info] cases: info_contract (8 clouds: 1/2/37/identical64/"
                "dup211/line128/4096/100k), info_traversal_oracle (512 + host walk), "
                "info_metadata (300), info_cross_structure (2048 + 100k), "
                "info_v1_vs_v0 (400), info_determinism (600), info_device_vs_host (257), "
                "info_sanity_checkers (uid 1 + uid 3).\n");
    CHECK(true);
}
