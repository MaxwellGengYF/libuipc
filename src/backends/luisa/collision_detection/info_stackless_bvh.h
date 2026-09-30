#pragma once
/********************************************************************
 * @file info_stackless_bvh.h
 * @brief InfoStacklessBVH — the "info" traversal of T7.1b (port of
 *        src/backends/cuda/collision_detection/info_stackless_bvh.h (238 lines) +
 *        details/info_stackless_bvh.inl (1154 lines)).
 *
 * The structure is T7.1a's `StacklessBVH` (read details/stackless_bvh.inl: the same
 * culbvh helpers, the same build pipeline, the same node array and the same
 * stackless walk) **plus per-leaf and per-internal-node body/contact metadata** and a
 * two-predicate query contract:
 *
 *  * every `Node` carries `bid`/`cid`; a leaf's are the primitive's, an internal
 *    node's are the *common* bid/cid of its whole range, or `-1` when the range
 *    spans more than one body/contact ("invalid" — the reference's `invalid`).
 *  * `detect`/`query` take a `NodePred` (called on every *intersected* node, leaf or
 *    internal, BEFORE the leaf test — so a node predicate that rejects a leaf also
 *    rejects that primitive) and a `LeafPred` (called on every candidate pair).
 *  * the reference's two "info" flavours differ only in what the predicates see:
 *    v1 (`InfoStacklessBVH`, this file) pre-loads the per-query bid/cid into shared
 *    memory once per thread and hands the predicates a `NodePredInfo` with
 *    `query_bid`/`query_cid` and a `LeafPredInfo` with the pair's *resolved*
 *    bid/cid; v0 (`InfoStacklessBVHV0`, info_stackless_bvh_v0.h) does neither and
 *    hands the predicates the raw `(query_id, node_bid, node_cid)` / `(i, j)`.
 *
 * PORT DEVIATIONS (forced by the DSL; each is repeated at its call site)
 * ----------------------------------------------------------------------
 *  * `AABB` -> `LAABB` and the `culbvh` helpers / `LBox` / `make_pair_var` are
 *    **reused from T7.1a's stackless_bvh.{h,inl}** (the reference vendors a second,
 *    identical copy under `uipc::info_stackless_detail`; the port keeps one shared
 *    copy and says so here). `Node` is a registered LC struct with two extra 4-byte
 *    members (40 bytes).
 *  * **A predicate is a *record-time* host callable, not a CUDA kernel argument.** LC
 *    kernel signatures cannot carry a callable, and a captured `Buffer`/`BufferView`
 *    becomes a *bound* argument of the compiled shader (cookbook §7.9 rule 2), so a
 *    cached shader would keep pointing at the first call's allocation. The reference's
 *    device functors therefore become host functors that are *baked into* the cached
 *    shader (its type is the cache key: keep them stateless, or with captures that are
 *    identical on every call — T7.1a's rule 13), and they are invoked with **LC
 *    expressions**: `NodePredInfo`/`LeafPredInfo` are *templates over the scalar
 *    type* (`IndexT` on the host, `lcrt::Var<IndexT>` inside a kernel), because a
 *    device value is a DSL object and cannot be stored in an `IndexT` field.
 *    CONSEQUENCE: a predicate cannot read device arrays (the reference's do: they
 *    capture `cuda_tool::CBufferView`s). The ported callers (the two sanity checkers)
 *    therefore apply their per-pair filters to the reported candidate pairs instead;
 *    see sanity_check/simplicial_surface_distance_check.cpp.
 *  * `cuda_tool::CBuffer2DView<IndexT> cmts`: the port has no 2-D buffer view (masks
 *    are flat, `row * width + col` — cookbook §7.13), so the parameter is a flat
 *    `LCBufferView<IndexT>`. It is **unused in the reference's own bodies** (both
 *    `launch_detect` and `launch_query` never touch it — the callers' predicates own
 *    their tables); the port keeps the argument for signature parity and ignores it.
 *  * `QueryBuffer::build` / the build pipeline / `read_host_sync` are T7.1a's,
 *    unchanged; the scene-box reduction is `detail::reduce_aabb_union`.
 *  * the reference's `propagateInformativeMetadata(int)` is declared and defined as an
 *    **empty** function (it is dead code in the reference too) — transcribed.
 *  * the reference's `build(aabbs)` overload leaves `m_BIDs`/`m_CIDs` empty views, but
 *    both `launch_detect` and `launch_query` then assert `m_aabbs.size() ==
 *    m_BIDs.size()` — i.e. the two-argument build is only usable with an empty input
 *    set. That is a reference inconsistency; the port transcribes the asserts
 *    verbatim (cookbook §7.31 rule 17).
 *********************************************************************/
#include <type_define.h>

#include <collision_detection/aabb.h>
// The culbvh helpers (K_THREADS, MAX_RES_PER_BLOCK, morton3D, make_pair_var,
// make_ordered_pair, make_pair_var, safe_copy_to), LBox, the box routines and the
// pair element are T7.1a's; the reference's info files are near-duplicates of its
// stackless files, and the port shares the one copy (see the header note).
#include <collision_detection/stackless_bvh.h>
#include <luisa_device/buffer.h>
#include <luisa_device/kernel.h>
#include <luisa_device/linear/algorithms.h>
#include <luisa_device/stream.h>
#include <luisa_device/view.h>

#include <cstdint>
#include <functional>
#include <type_traits>

namespace uipc::backend::luisa
{
class InfoStacklessBVH
{
  public:
    /// The port's spelling of the reference's `cuda_tool::CBuffer2DView<IndexT> cmts`
    /// argument (a flat mask; ignored by the traversal — see the header note).
    using MaskView = lud::LCBufferView<IndexT>;

    /// The reference's `NodePredInfo`: the pre-loaded query bid/cid plus the node's.
    /// Templated over the scalar type so that a *record-time* predicate can be called
    /// with either host integers or LC expressions (`I` = `IndexT` /
    /// `lcrt::Var<IndexT>`); see the header note.
    template <typename I = IndexT>
    struct NodePredInfoT
    {
        I query_id;
        I query_bid;  // pre-loaded from shared memory (v1 only)
        I query_cid;  // pre-loaded from shared memory (v1 only)
        I node_bid;
        I node_cid;

        NodePredInfoT() = default;
        NodePredInfoT(I query_id, I query_bid, I query_cid, I node_bid, I node_cid)
            : query_id(query_id)
            , query_bid(query_bid)
            , query_cid(query_cid)
            , node_bid(node_bid)
            , node_cid(node_cid)
        {
        }
    };
    using NodePredInfo = NodePredInfoT<IndexT>;

    /// The reference's `LeafPredInfo`: the pair *and* its resolved bid/cid (the
    /// query side from shared memory, the leaf side from the node). NOTE (the
    /// reference's own comment): the bid/cid check is not redundant with the node
    /// predicate — an internal node whose bid/cid is `-1` cannot be culled by
    /// `NodePred`, so the exact check must happen at the leaf.
    template <typename I = IndexT>
    struct LeafPredInfoT
    {
        I i;
        I j;
        I bid_i;
        I cid_i;
        I bid_j;
        I cid_j;

        LeafPredInfoT() = default;
        LeafPredInfoT(I i, I j, I bid_i, I cid_i, I bid_j, I cid_j)
            : i(i)
            , j(j)
            , bid_i(bid_i)
            , cid_i(cid_i)
            , bid_j(bid_j)
            , cid_j(cid_j)
        {
        }
    };
    using LeafPredInfo = LeafPredInfoT<IndexT>;

    class QueryBuffer
    {
      public:
        QueryBuffer()
        {
            // initial capacity (the reference reserves 50 * 1024 pairs)
            m_pairs.resize(50 * 1024);
        }

        [[nodiscard]] auto view() const noexcept { return m_pairs.cview(0, m_size); }
        void reserve(size_t size) { m_pairs.resize(size); }
        [[nodiscard]] SizeT size() const noexcept { return m_size; }
        [[nodiscard]] auto  viewer() const noexcept { return view().viewer(); }

      public:
        friend class InfoStacklessBVH;
        SizeT m_size = 0;
        lud::LDeviceBuffer<LVector2i> m_pairs;

        lud::LDeviceBuffer<std::uint32_t> m_queryMtCode;
        lud::LDeviceBuffer<std::uint32_t> m_querySortedMtCode;
        lud::LDeviceVar<LAABB>            m_querySceneBox;
        lud::LDeviceBuffer<std::int32_t>  m_queryId;
        lud::LDeviceBuffer<std::int32_t>  m_querySortedId;
        lud::LDeviceVar<std::int32_t>     m_cpNum;

        /// The port's scratch for the query-side union-box reduction (the reference's
        /// `calcMaxBVFromBox` uses `atomicMinf` instead; see details/aabb_reduce.h).
        lud::LDeviceBuffer<LAABB> m_queryReducePartials;

        void  build(lud::LCBufferView<LAABB> aabbs, lud::LStream& stream);
        SizeT query_count() { return m_queryMtCode.size(); }
    };

    /// The info node: T7.1a's `{lc, escape, bound}` plus the leaf/common bid/cid.
    /// `lc == -1` is a leaf, `escape == -1` ends the traversal, `bid`/`cid` are `-1`
    /// when the node's range spans more than one body / contact element.
    struct Node
    {
        IndexT lc;
        IndexT escape;
        LAABB  bound;
        IndexT bid;
        IndexT cid;
    };

    class Config
    {
      public:
        Float reserve_ratio;
        Config()
            : reserve_ratio(1.2)
        {
        }
    };

    InfoStacklessBVH(Config config = Config{}) { m_impl.config = config; }
    ~InfoStacklessBVH() = default;

    Config&       config() noexcept { return m_impl.config; }
    const Config& config() const noexcept { return m_impl.config; }

    /// Build with per-primitive body/contact ids (the reference asserts both spans
    /// match the AABB span).
    void build(lud::LCBufferView<LAABB>   aabbs,
               lud::LCBufferView<IndexT> BIDs,
               lud::LCBufferView<IndexT> CIDs,
               lud::LStream*             stream = nullptr);

    /// The reference's info-less overload (see the header note: the reference's own
    /// `detect`/`query` then assert against the empty BID/CID spans).
    void build(lud::LCBufferView<LAABB> aabbs, lud::LStream* stream = nullptr);

    /**
     * @brief Detect overlapping pairs inside the structure (each unordered pair
     *        once), with the two info predicates.
     *
     * `NodePred` is called on every intersected node with a `NodePredInfo`, `LeafPred`
     * on every candidate pair with a `LeafPredInfo`; both are *record-time* host
     * callables over LC expressions.
     */
    template <typename NodePred, typename LeafPred>
    void detect(MaskView cmts, NodePred np, LeafPred lp, QueryBuffer& qbuffer);

    /// The split half of `detect`: record one traversal, then decide whether the
    /// output had to grow (`prepare_query_result`) and, if so, re-record it.
    template <typename NodePred, typename LeafPred>
    void launch_detect(MaskView cmts, NodePred np, LeafPred lp, QueryBuffer& qbuffer);

    /**
     * @brief Query overlapping primitives from an external AABB set; each pair is
     *        reported as `(query_index, object_index)` (no ordering).
     */
    template <typename NodePred, typename LeafPred>
    void query(lud::LCBufferView<LAABB>   query_aabbs,
               lud::LCBufferView<IndexT> query_BIDs,
               lud::LCBufferView<IndexT> query_CIDs,
               MaskView                  cmts,
               NodePred                  np,
               LeafPred                  lp,
               QueryBuffer&              qbuffer);

    template <typename NodePred, typename LeafPred>
    void launch_query(lud::LCBufferView<LAABB>   query_aabbs,
                      lud::LCBufferView<IndexT> query_BIDs,
                      lud::LCBufferView<IndexT> query_CIDs,
                      MaskView                  cmts,
                      NodePred                  np,
                      LeafPred                  lp,
                      QueryBuffer&              qbuffer,
                      bool                      rebuild_query = true);

    /// Publish a device-produced count and grow the output if a retry is required.
    /// The caller re-launches the same query when this returns true.
    bool prepare_query_result(QueryBuffer& qbuffer, int count);

  public:
    class Impl
    {
      public:
        void build(lud::LCBufferView<LAABB>   aabbs,
                   lud::LCBufferView<IndexT> bids,
                   lud::LCBufferView<IndexT> cids,
                   lud::LStream&             stream);

        static void calcMaxBVFromBox(lud::LCBufferView<LAABB> aabbs,
                                     lud::LBufferView<LAABB>  scene_box,
                                     lud::LStream&            stream,
                                     lud::LDeviceBuffer<LAABB>& partials);
        static void calcMCsFromBox(lud::LCBufferView<LAABB>        aabbs,
                                   lud::LCBufferView<LAABB>        scene_box,
                                   lud::LBufferView<std::uint32_t> codes,
                                   lud::LStream&                   stream);
        void        calcInverseMapping(lud::LStream& stream);
        void        buildPrimitivesFromBox(lud::LCBufferView<LAABB> aabbs,
                                           lud::LStream&            stream);
        void        calcExtNodeSplitMetrics(lud::LStream& stream);
        void        buildIntNodes(int size, lud::LStream& stream);
        void        calcIntNodeOrders(int size, lud::LStream& stream);
        void        updateBvhExtNodeLinks(int size, lud::LStream& stream);
        void        reorderNode(int intSize, lud::LStream& stream);
        /// The reference's `propagateInformativeMetadata` — an empty function in the
        /// reference too (dead code); transcribed so a reader can diff the two files.
        void        propagateInformativeMetadata(int int_size) { (void)int_size; }

        /// The info traversal: pre-loads the per-query bid/cid into shared memory
        /// once per thread before the hot loop (the reference's "optimized" variant).
        template <typename NodeCull, typename PairPred>
        void stacklessSelf(NodeCull                        node_cull,
                           PairPred                        pair_pred,
                           lud::LBufferView<std::int32_t>  cpNum,
                           lud::LBufferView<LVector2i>     buffer,
                           lud::LStream&                   stream);

        template <typename NodeCull, typename PairPred>
        void stacklessOther(NodeCull                        node_cull,
                            PairPred                        pair_pred,
                            lud::LCBufferView<LAABB>        query_aabbs,
                            lud::LCBufferView<IndexT>       query_bids,
                            lud::LCBufferView<IndexT>       query_cids,
                            lud::LCBufferView<std::int32_t> query_sorted_id,
                            lud::LBufferView<std::int32_t>  cpNum,
                            lud::LBufferView<LVector2i>     buffer,
                            lud::LStream&                   stream);

        lud::LCBufferView<LAABB>  objs;  // external AABBs, kept alive by the view
        lud::LCBufferView<IndexT> bids;
        lud::LCBufferView<IndexT> cids;

        lud::LDeviceVar<LAABB>            scene_box;
        lud::LDeviceBuffer<std::uint32_t> flags;
        lud::LDeviceBuffer<std::uint32_t> mtcode;
        lud::LDeviceBuffer<std::uint32_t> sorted_mtcode;
        lud::LDeviceBuffer<std::int32_t>  sorted_id;
        lud::LDeviceBuffer<std::int32_t>  primMap;
        lud::LDeviceBuffer<std::int32_t>  metric;
        lud::LDeviceBuffer<std::uint32_t> count;  // == the reference's `depths`
        lud::LDeviceBuffer<std::int32_t>  tkMap;
        lud::LDeviceBuffer<std::uint32_t> offsetTable;

        lud::LDeviceBuffer<LAABB>         ext_aabb;
        lud::LDeviceBuffer<std::int32_t>  ext_idx;
        lud::LDeviceBuffer<std::int32_t>  ext_lca;
        lud::LDeviceBuffer<std::uint32_t> ext_par;

        lud::LDeviceBuffer<std::int32_t>  int_lc;
        lud::LDeviceBuffer<std::int32_t>  int_rc;
        lud::LDeviceBuffer<std::int32_t>  int_par;
        lud::LDeviceBuffer<std::int32_t>  int_range_x;
        lud::LDeviceBuffer<std::int32_t>  int_range_y;
        lud::LDeviceBuffer<std::uint32_t> int_mark;
        lud::LDeviceBuffer<LAABB>         int_aabb;

        // the info metadata (v0 initialises these in its init kernel, v1 does not —
        // see info_stackless_bvh_v0.h and the cookbook rule)
        lud::LDeviceBuffer<IndexT> ext_bid;
        lud::LDeviceBuffer<IndexT> ext_cid;
        lud::LDeviceBuffer<IndexT> int_bid;
        lud::LDeviceBuffer<IndexT> int_cid;

        lud::LDeviceBuffer<Node> nodes;

        /// A one-element stand-in for the BID/CID spans when `build(aabbs)` was used:
        /// LC rejects a zero-length buffer view as a kernel argument (T7.1a rule 9),
        /// and the kernels never read it when `has_info == 0`.
        lud::LDeviceBuffer<IndexT> info_dummy;

        /// The port's scratch for the union-box reduction (the reference's scene-box
        /// kernel uses `atomicMinf` instead; see details/aabb_reduce.h).
        lud::LDeviceBuffer<LAABB> reduce_partials;

        Config config;
    };

  private:
    lud::LCBufferView<LAABB>  m_aabbs;
    lud::LCBufferView<IndexT> m_BIDs;
    lud::LCBufferView<IndexT> m_CIDs;
    Impl                      m_impl;
};
}// namespace uipc::backend::luisa

// ---------------------------------------------------------------------------
//  LuisaCompute struct registration (global scope, before the first
//  `Expr<Node>`/`Var<Node>` use in details/info_stackless_bvh.inl).
// ---------------------------------------------------------------------------
LUISA_STRUCT(uipc::backend::luisa::InfoStacklessBVH::Node, lc, escape, bound, bid, cid){};

#include "details/info_stackless_bvh.inl"
