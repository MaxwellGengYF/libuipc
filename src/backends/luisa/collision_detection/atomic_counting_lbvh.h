#pragma once
/********************************************************************
 * @file atomic_counting_lbvh.h
 * @brief AtomicCountingLBVH: an LBVH with an atomic-ticket candidate-pair output
 *        (port of src/backends/cuda/collision_detection/atomic_counting_lbvh.{h,cu} +
 *        details/atomic_counting_lbvh.inl, 48 + 16 + 154 lines).
 *
 * WHAT IT IS
 * ----------
 * A thin wrapper around `LinearBVH` that answers "all overlapping pairs" instead of
 * "all overlaps of one query": one thread per primitive traverses the tree and, for
 * every reported object id, appends `(i, id)` to a flat pair array through a single
 * global atomic counter.  `detect` uses the input AABBs themselves as queries (and
 * only reports `id > i`, so a self-collision pass yields each pair once, ordered by
 * *input index*); `query` runs the same traversal against an external AABB set and
 * reports every `(query_index, object_index)` hit.
 *
 * THE CAPACITY CONTRACT (what the test pins)
 * -----------------------------------------
 * The counter counts *every* candidate, even one that did not fit, so after the first
 * pass the host knows the exact number of pairs `h_cp_num`.  If it exceeds the
 * buffer's size, the buffer is grown to `h_cp_num * m_reserve_ratio` (1.1 by default)
 * and the query is **redone** — the pair array is not appended to, it is rewritten
 * from ticket 0, so the second pass is the authoritative one.  `QueryBuffer::size()`
 * then returns `h_cp_num` and `view()` is the first `h_cp_num` entries.
 *
 * PORT DEVIATIONS
 * ---------------
 *  * `Vector2i` -> `LVector2i` (the registered device pair element).
 *  * the reference's `Pred p` is a CUDA *device* functor carried as a kernel argument;
 *    the port's predicate is a record-time callable baked into the cached shader (see
 *    details/atomic_counting_lbvh.inl) — it must be stateless or have identical
 *    captures on every call.
 *  * the host reads `int h = m_cp_num;` (an implicitly synchronous read in the
 *    reference) through `lud::read_host_sync`, which commits the frame CommandList
 *    first; that is also what makes the resize safe (the recorded dispatches that
 *    reference the old pair buffer have completed before it is released).
 *  * the `build()` entry point keeps the reference's `CBufferView` signature; the port
 *    stores an owning `LCBufferView` (the reference's non-owning view is the caller's
 *    responsibility).
 *********************************************************************/
#include <type_define.h>

#include <collision_detection/linear_bvh.h>
#include <luisa_device/buffer.h>
#include <luisa_device/stream.h>
#include <luisa_device/view.h>

#include <cstdint>

namespace uipc::backend::luisa
{
class AtomicCountingLBVH
{
  public:
    class QueryBuffer
    {
      public:
        [[nodiscard]] auto view() const noexcept { return m_pairs.cview(0, m_size); }
        void reserve(size_t size) { m_pairs.resize(size); }
        [[nodiscard]] SizeT size() const noexcept { return m_size; }
        [[nodiscard]] auto  viewer() const noexcept { return view().viewer(); }

      private:
        friend class AtomicCountingLBVH;
        SizeT                         m_size = 0;
        lud::LDeviceBuffer<LVector2i> m_pairs;
    };

    explicit AtomicCountingLBVH(lud::LStream* stream = nullptr) noexcept;

    void build(lud::LCBufferView<LinearBVHAABB> aabbs);

    template <typename Pred>
    void detect(Pred p, QueryBuffer& out_pairs);

    template <typename Pred>
    void query(lud::LCBufferView<LinearBVHAABB> query_aabbs, Pred p, QueryBuffer& out_pairs);

    // ---- test seam (the reference keeps the same state private; the [bvh] test reads
    //      the tree through LinearBVHVisitor instead, so this is only a convenience)
    [[nodiscard]] LinearBVH& lbvh() noexcept { return m_lbvh; }

  private:
    [[nodiscard]] lud::LStream& stream() noexcept;

    lud::LCBufferView<LinearBVHAABB> m_aabbs;
    lud::LDeviceVar<std::int32_t>    m_cp_num;
    LinearBVH                        m_lbvh;
    Float                            m_reserve_ratio = 1.1;
    lud::LStream*                    m_stream        = nullptr;
};
}// namespace uipc::backend::luisa

#include "details/atomic_counting_lbvh.inl"
