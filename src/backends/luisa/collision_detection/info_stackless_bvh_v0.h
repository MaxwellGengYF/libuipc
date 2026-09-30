#pragma once
/********************************************************************
 * @file info_stackless_bvh_v0.h
 * @brief InfoStacklessBVHV0 — the *baseline* "info" flavour (port of
 *        src/backends/cuda/collision_detection/info_stackless_bvh_v0.h (178 lines) +
 *        details/info_stackless_bvh_v0.inl (1134 lines)).
 *
 * The V0 (baseline) variant of T7.1b's info traversal: the two files are
 * near-duplicates and the port keeps them apart (the reference does), so that a reader
 * can diff them.  Measured differences from `InfoStacklessBVH` (v1), all of them
 * transcribed and pinned by the [bvh_info] test:
 *
 *  | | v1 (InfoStacklessBVH) | v0 (InfoStacklessBVHV0) |
 *  |---|---|---|
 *  | `NodePredInfo` | 5 fields, `query_bid`/`query_cid` pre-loaded into SMem | **3 fields** (`query_id`, `node_bid`, `node_cid`) |
 *  | `LeafPredInfo` | 6 fields, the pair's bid/cid resolved | **absent** — `LeafPred(i, j)` |
 *  | traversal SMem | `s_qbid`/`s_qcid` pre-load once per thread | **none** |
 *  | self pair | `LeafPredInfo` with the query side first (ordered by `q_first`) | `LeafPred(ordered_pair)` + the reference's `j <= i` adapter guard |
 *  | `detect`/`query` | split into `launch_*` + `prepare_query_result` (public) | one inline body each, retry in place |
 *  | `Impl::build` | no `ext_mark` | `ext_mark` allocated + initialised to 7 |
 *  | init kernel | `ext_lca(idx) = -1` for `idx <= num_objs`; no `int_bid`/`int_cid` init | `ext_lca = 0` / `ext_par = 0` / `ext_mark = 7` per leaf, `int_bid`/`int_cid = -1` per internal node |
 *  | `buildIntNodes` | no leaf prologue | `_lvs_lca(idx) = -1; _depths(idx) = 0;` prologue |
 *  | `stacklessOther` | takes the query BID/CID spans | takes neither |
 *
 * Everything else — the culbvh helpers, the build pipeline, the node layout, the
 * traversal order, the shared-cache flush protocol — is shared with v1 and reused from
 * `info_stackless_bvh.h` / T7.1a's `stackless_bvh.{h,inl}`.  Read
 * `info_stackless_bvh.h`'s header note for the port deviations; they apply verbatim.
 *
 * The one *behavioural* difference the port can observe is the missing
 * `int_bid`/`int_cid` initialisation (v0 writes -1, v1 does not): the [bvh_info] test
 * pins that the two builds still produce element-identical node arrays (every read of
 * an internal child's bid/cid happens after that child completed its own merge).
 *********************************************************************/
#include <type_define.h>

#include <collision_detection/aabb.h>
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
class InfoStacklessBVHV0
{
  public:
    /// The port's spelling of the reference's `cuda_tool::CBuffer2DView<IndexT> cmts`
    /// argument (a flat mask; ignored by the traversal).
    using MaskView = lud::LCBufferView<IndexT>;

    /// The reference's v0 `NodePredInfo`: **no** pre-loaded query bid/cid (the user's
    /// predicate would read them from global memory through `query_id`; a ported
    /// predicate cannot read device arrays — see info_stackless_bvh.h).  Templated
    /// over the scalar type, like v1's.
    template <typename I = IndexT>
    struct NodePredInfoT
    {
        I query_id;
        I node_bid;
        I node_cid;

        NodePredInfoT() = default;
        NodePredInfoT(I query_id, I node_bid, I node_cid)
            : query_id(query_id)
            , node_bid(node_bid)
            , node_cid(node_cid)
        {
        }
    };
    using NodePredInfo = NodePredInfoT<IndexT>;

    class QueryBuffer
    {
      public:
        QueryBuffer()
        {
            m_pairs.resize(50 * 1024);
        }

        [[nodiscard]] auto view() const noexcept { return m_pairs.cview(0, m_size); }
        void reserve(size_t size) { m_pairs.resize(size); }
        [[nodiscard]] SizeT size() const noexcept { return m_size; }
        [[nodiscard]] auto  viewer() const noexcept { return view().viewer(); }

      public:
        friend class InfoStacklessBVHV0;
        SizeT m_size = 0;
        lud::LDeviceBuffer<LVector2i> m_pairs;

        lud::LDeviceBuffer<std::uint32_t> m_queryMtCode;
        lud::LDeviceBuffer<std::uint32_t> m_querySortedMtCode;
        lud::LDeviceVar<LAABB>            m_querySceneBox;
        lud::LDeviceBuffer<std::int32_t>  m_queryId;
        lud::LDeviceBuffer<std::int32_t>  m_querySortedId;
        lud::LDeviceVar<std::int32_t>     m_cpNum;

        lud::LDeviceBuffer<LAABB> m_queryReducePartials;

        void  build(lud::LCBufferView<LAABB> aabbs, lud::LStream& stream);
        SizeT query_count() { return m_queryMtCode.size(); }
    };

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

    InfoStacklessBVHV0(Config config = Config{}) { m_impl.config = config; }
    ~InfoStacklessBVHV0() = default;

    Config&       config() noexcept { return m_impl.config; }
    const Config& config() const noexcept { return m_impl.config; }

    void build(lud::LCBufferView<LAABB>   aabbs,
               lud::LCBufferView<IndexT> BIDs,
               lud::LCBufferView<IndexT> CIDs,
               lud::LStream*             stream = nullptr);
    void build(lud::LCBufferView<LAABB> aabbs, lud::LStream* stream = nullptr);

    /// Detect overlapping pairs inside the structure (each unordered pair once).
    template <typename NodePred, typename LeafPred>
    void detect(MaskView cmts, NodePred np, LeafPred lp, QueryBuffer& qbuffer);

    /// Query overlapping primitives from an external AABB set; each pair is reported
    /// as `(query_index, object_index)`.
    template <typename NodePred, typename LeafPred>
    void query(lud::LCBufferView<LAABB>   query_aabbs,
               lud::LCBufferView<IndexT> query_BIDs,
               lud::LCBufferView<IndexT> query_CIDs,
               MaskView                  cmts,
               NodePred                  np,
               LeafPred                  lp,
               QueryBuffer&              qbuffer);

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
        /// the reference's `propagateInformativeMetadata` — empty there too
        void        propagateInformativeMetadata(int int_size) { (void)int_size; }

        template <typename NodeCull, typename PairPred>
        void stacklessSelf(NodeCull                       node_cull,
                           PairPred                       pair_pred,
                           lud::LBufferView<std::int32_t> cpNum,
                           lud::LBufferView<LVector2i>    buffer,
                           lud::LStream&                  stream);

        template <typename NodeCull, typename PairPred>
        void stacklessOther(NodeCull                        node_cull,
                            PairPred                        pair_pred,
                            lud::LCBufferView<LAABB>        query_aabbs,
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
        lud::LDeviceBuffer<std::uint32_t> count;
        lud::LDeviceBuffer<std::int32_t>  tkMap;
        lud::LDeviceBuffer<std::uint32_t> offsetTable;

        lud::LDeviceBuffer<LAABB>         ext_aabb;
        lud::LDeviceBuffer<std::int32_t>  ext_idx;
        lud::LDeviceBuffer<std::int32_t>  ext_lca;
        lud::LDeviceBuffer<std::uint32_t> ext_mark;
        lud::LDeviceBuffer<std::uint32_t> ext_par;

        lud::LDeviceBuffer<std::int32_t>  int_lc;
        lud::LDeviceBuffer<std::int32_t>  int_rc;
        lud::LDeviceBuffer<std::int32_t>  int_par;
        lud::LDeviceBuffer<std::int32_t>  int_range_x;
        lud::LDeviceBuffer<std::int32_t>  int_range_y;
        lud::LDeviceBuffer<std::uint32_t> int_mark;
        lud::LDeviceBuffer<LAABB>         int_aabb;

        lud::LDeviceBuffer<IndexT> ext_bid;
        lud::LDeviceBuffer<IndexT> ext_cid;
        lud::LDeviceBuffer<IndexT> int_bid;
        lud::LDeviceBuffer<IndexT> int_cid;

        lud::LDeviceBuffer<Node> nodes;

        /// see InfoStacklessBVH::Impl::info_dummy
        lud::LDeviceBuffer<IndexT> info_dummy;

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
//  `Expr<Node>`/`Var<Node>` use in details/info_stackless_bvh_v0.inl).
// ---------------------------------------------------------------------------
LUISA_STRUCT(uipc::backend::luisa::InfoStacklessBVHV0::Node, lc, escape, bound, bid, cid){};

#include "details/info_stackless_bvh_v0.inl"
