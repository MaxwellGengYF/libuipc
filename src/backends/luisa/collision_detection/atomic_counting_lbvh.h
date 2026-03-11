#pragma once
#include <type_define.h>
#include <collision_detection/linear_bvh.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class AtomicCountingLBVH
{
  public:
    class QueryBuffer
    {
      public:
        luisa::compute::BufferView<Vector2i> view() const noexcept { return m_pairs.view(0, m_size); }
        void  reserve(size_t size, luisa::compute::Device& device);
        SizeT size() const noexcept { return m_size; }
        auto  viewer() const noexcept { return view(); }

      private:
        friend class AtomicCountingLBVH;
        SizeT                              m_size = 0;
        luisa::compute::Buffer<Vector2i>   m_pairs;
    };

    AtomicCountingLBVH(luisa::compute::Device& device, luisa::compute::Stream& stream) noexcept;

    void build(luisa::compute::BufferView<LinearBVHAABB> aabbs);

    template <typename Pred>
    void detect(Pred p, QueryBuffer& out_pairs);

    template <typename Pred>
    void query(luisa::compute::BufferView<LinearBVHAABB> query_aabbs, Pred p, QueryBuffer& out_pairs);

  private:
    luisa::compute::Device&            m_device;
    luisa::compute::Stream&            m_stream;
    luisa::compute::BufferView<LinearBVHAABB> m_aabbs;
    luisa::compute::Buffer<IndexT>     m_cp_num;  // size 1 buffer for atomic counter
    LinearBVH                          m_lbvh;
    Float                              m_reserve_ratio = 1.1;
};
}  // namespace uipc::backend::luisa

#include "details/atomic_counting_lbvh.inl"
