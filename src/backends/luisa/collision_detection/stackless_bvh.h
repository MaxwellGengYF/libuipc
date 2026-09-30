#pragma once
/********************************************************************
 * @file stackless_bvh.h
 * @brief Stackless BVH for AABB overlap detection (port of
 *        src/backends/cuda/collision_detection/stackless_bvh.h +
 *        details/stackless_bvh.inl, 195 + 1217 lines).
 *
 * "Stackless": the tree is stored so that a traversal needs no explicit stack — each
 * node carries an `escape` link to the node to visit when its subtree does not
 * intersect, and the traversal is a `while(st != -1)` walk.  The hierarchy is built
 * bottom-up from the Morton-sorted leaf order (the culbvh algorithm: the reference
 * credits https://github.com/ZiXuanVickyLu/culbvh and
 * https://github.com/jerry060599/KittenGpuLBVH).
 *
 * STRUCTURE (the contract M8 depends on)
 * --------------------------------------
 *  * `Node {int32 lc; int32 escape; AABB bound;}` is 32 bytes: `lc == -1` marks a leaf
 *    (then the "object" of the node is `ext_idx[st - intSize]`), `escape == -1` marks
 *    the end of the traversal.
 *  * `nodes[0 .. intSize)` are the internal nodes in *reordered* order,
 *    `nodes[intSize .. intSize + numObjs)` the leaves (`nodes[intSize + i]` is the
 *    i-th *sorted* box).
 *  * `detect(pred, qbuffer)` reports **ordered** pairs `(i, j)` with `i < j` from
 *    `make_ordered_pair`; `query(aabbs, pred, qbuffer)` reports `(query_index,
 *    object_index)` without the ordering rule.  Both report through a per-block shared
 *    cache flushed with an atomic ticket on the pair counter, so the *order* of the
 *    pair array is not deterministic — the reference's own test compares the pair
 *    *sets* (`check_cp_conservative`), and the [bvh] test pins exactly that.
 *  * `MAX_RES_PER_BLOCK = 1024` caps one block's batch; a block that fills the cache
 *    flushes and continues (`while(true)` loop), so a block may emit several batches.
 *
 * PORT DEVIATIONS (forced by the DSL; all repeated at the call site)
 * ------------------------------------------------------------------
 *  * `AABB` -> `LAABB` (aabb.h; same bytes).  `Node` is a registered LC struct.
 *  * `__shfl_down_sync` + `atomicMinf/atomicMaxf` in the scene-box kernel ->
 *    `detail::reduce_aabb_union` (see details/aabb_reduce.h: the struct element has no
 *    component atomics, and min/max over floats is order-independent so the *value* is
 *    identical).
 *  * the scene-box kernel's `if(idx >= size) return;` sits *before* its
 *    `__syncthreads()`, which is a CUDA UB for a partial last block; the port
 *    dispatches whole blocks, clamps the index and lets the surplus lanes contribute
 *    the identity box (the reference's early return around a barrier cannot be
 *    ported — cookbook §7.30 rule 16).
 *  * `__threadfence()` -> LC's `volatile_read`/`volatile_write` on the buffers the
 *    protocol orders (their CUDA codegen is `__threadfence()` + the access), see
 *    linear_bvh.h's note.
 *  * the Impl's kernels write *whole* struct elements (LC has no per-member buffer
 *    store), so the reference's `_tks_box(cur) = _lvs_box(chl)` style merges are
 *    load-merge-store; where the reference relied on a *word*-granular store to avoid
 *    a lost update, the port reorders the work instead (see `Impl::build`).
 *  * `StacklessBVH::QueryBuffer::query_count()` etc. are unchanged; the reference's
 *    implicitly-synchronous scalar reads (`int h = qbuffer.m_cpNum;`) become an
 *    explicit `stream.flush()` + read (the port's LStream batches into a frame
 *    CommandList, so an unordered read would be wrong — cookbook §7.10).
 *  * `quantNode` (declared, never used in the reference) is not ported;
 *    `StacklessBVHFriend` (declared, never instantiated) is not ported.
 *********************************************************************/
#include <type_define.h>

#include <collision_detection/aabb.h>
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
class StacklessBVH
{
  public:
    class Config
    {
      public:
        Float reserve_ratio;
        Config()
            : reserve_ratio(1.2)
        {
        }
    };

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
        friend class StacklessBVH;
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

    /// The stackless node: `lc == -1` is a leaf, `escape == -1` ends the traversal.
    struct Node
    {
        IndexT lc;
        IndexT escape;
        LAABB  bound;
    };

    StacklessBVH(Config config = Config{}) { m_impl.config = config; }
    ~StacklessBVH() = default;

    /// The reference's `DefaultQueryCallback` (`bool operator()(IndexT, IndexT) const
    /// { return true; }`).  A record-time predicate is called with *DSL* components
    /// inside the kernel, so the port's spelling is generic over its arguments and
    /// returns the host `true` (a `$if` condition).
    struct DefaultQueryCallback
    {
        [[nodiscard]] bool operator()(const auto& /*i*/, const auto& /*j*/) const noexcept
        {
            return true;
        }
    };

    /**
     * @brief Build the Stackless BVH from given AABBs
     *
     * @param aabbs Input AABBs; the view keeps the buffer alive for the BVH's lifetime
     */
    void build(lud::LCBufferView<LAABB> aabbs, lud::LStream* stream = nullptr);

    /**
     * @brief Detect overlapping AABB pairs in the BVH
     *
     * @param callback f: (int i, int j) -> bool   (a *record-time* predicate: it
     *        returns a DSL expression and may read device data)
     */
    template <std::invocable<IndexT, IndexT> Pred = DefaultQueryCallback>
    void detect(Pred callback, QueryBuffer& qbuffer);

    /**
     * @brief Query overlapping AABBs from external AABBs
     */
    template <std::invocable<IndexT, IndexT> Pred = DefaultQueryCallback>
    void query(lud::LCBufferView<LAABB> aabbs, Pred callback, QueryBuffer& qbuffer);

  public:
    class Impl
    {
      public:
        void build(lud::LCBufferView<LAABB> aabbs, lud::LStream& stream);

        template <typename Pred>
        void StacklessCDSharedSelf(Pred                    pred,
                                   lud::LBufferView<int>   cpNum,
                                   lud::LBufferView<LVector2i> buffer,
                                   lud::LStream&           stream);
        template <typename Pred>
        void StacklessCDSharedOther(Pred                           pred,
                                    lud::LCBufferView<LAABB>       query_aabbs,
                                    lud::LCBufferView<std::int32_t> query_sorted_id,
                                    lud::LBufferView<int>          cpNum,
                                    lud::LBufferView<LVector2i>    buffer,
                                    lud::LStream&                  stream);

        static void calcMaxBVFromBox(lud::LCBufferView<LAABB> aabbs,
                                     lud::LBufferView<LAABB>  scene_box,
                                     lud::LStream&            stream,
                                     lud::LDeviceBuffer<LAABB>& partials);
        static void calcMCsFromBox(lud::LCBufferView<LAABB>          aabbs,
                                   lud::LCBufferView<LAABB>          scene_box,
                                   lud::LBufferView<std::uint32_t>   codes,
                                   lud::LStream&                     stream);
        void        calcInverseMapping(lud::LStream& stream);
        void        buildPrimitivesFromBox(lud::LCBufferView<LAABB> aabbs,
                                           lud::LStream&            stream);
        void        calcExtNodeSplitMetrics(lud::LStream& stream);
        void        buildIntNodes(int size, lud::LStream& stream);
        void        calcIntNodeOrders(int size, lud::LStream& stream);
        void        updateBvhExtNodeLinks(int size, lud::LStream& stream);
        void        reorderNode(int intSize, lud::LStream& stream);

        lud::LCBufferView<LAABB> objs;  // external AABBs, kept alive by the view

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

        lud::LDeviceBuffer<Node> nodes;

        /// The port's scratch for the union-box reduction (the reference's
        /// scene-box kernel uses `atomicMinf` instead; see details/aabb_reduce.h).
        lud::LDeviceBuffer<LAABB> reduce_partials;

        Config config;
    };

  private:
    Impl m_impl;
};
}// namespace uipc::backend::luisa

// ---------------------------------------------------------------------------
//  LuisaCompute struct registration (global scope, before the first
//  `Expr<Node>`/`Var<Node>` use in details/stackless_bvh.inl).
// ---------------------------------------------------------------------------
LUISA_STRUCT(uipc::backend::luisa::StacklessBVH::Node, lc, escape, bound){};

#include "details/stackless_bvh.inl"
