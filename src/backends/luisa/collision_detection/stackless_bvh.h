/**
 * @file stackless_bvh.h
 * 
 * @brief LuisaCompute Refactored Version of Stackless BVH for AABB overlap detection
 * 
 * References:
 * 
 * Thanks to the original authors of the following repositories for their excellent implementations of Stackless BVH!
 * 
 * - https://github.com/ZiXuanVickyLu/culbvh
 * - https://github.com/jerry060599/KittenGpuLBVH
 * 
 */

#pragma once
#include <uipc/common/logger.h>
#include <backends/luisa/type_define.h>
#include <backends/luisa/collision_detection/aabb.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
/**
 * @brief Friend class to access private members of StacklessBVH (for internal use only)
 */
template <typename T>
class StacklessBVHFriend;

/**
 * @brief Stackless Bounding Volume Hierarchy for AABB overlap detection
 */
class StacklessBVH
{
  public:
    template <typename T>
    friend class StacklessBVHFriend;

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
        QueryBuffer(luisa::compute::Device& device);

        auto  view() const noexcept { return luisa::compute::BufferView<Vector2i>{m_pairs, 0, m_size}; }
        void  reserve(size_t size) { m_pairs.reserve(size); }
        SizeT size() const noexcept { return m_size; }
        auto  viewer() const noexcept { return view(); }

      public:
        friend class StacklessBVH;
        SizeT                       m_size = 0;
        luisa::compute::Buffer<Vector2i> m_pairs;

        luisa::compute::Buffer<uint32_t> m_queryMtCode;
        luisa::compute::Buffer<AABB>     m_querySceneBox;
        luisa::compute::Buffer<int32_t>  m_querySortedId;
        luisa::compute::Buffer<int32_t>  m_cpNum;

        void  build(luisa::compute::BufferView<AABB> aabbs);
        SizeT query_count() { return m_queryMtCode.size(); }
    };

    struct /*__align__(16) */ Node
    {
        IndexT lc;
        IndexT escape;
        AABB   bound;
    };

    StacklessBVH(luisa::compute::Device& device, Config config = Config{});

    ~StacklessBVH() = default;

    struct DefaultQueryCallback
    {
        LUISA_GENERIC bool operator()(IndexT i, IndexT j) const { return true; }
    };

    /**
     * @brief Build the Stackless BVH from given AABBs
     * 
     * @param aabbs Input AABBs, aabbs must be kept valid during the lifetime of this BVH
     */
    void build(luisa::compute::BufferView<AABB> aabbs);

    /**
     * @brief Detect overlapping AABB pairs in the BVH
     * 
     * @param callback f: (int i, int j) -> bool Callback predicate to filter overlapping pairs
     * @param qbuffer Output buffer to store detected overlapping pairs
     */
    template <std::invocable<IndexT, IndexT> Pred = DefaultQueryCallback>
    void detect(Pred callback, QueryBuffer& qbuffer);


    /*
    * @brief Query overlapping AABBs from external AABBs
    * 
    * @param aabbs Input external AABBs to query, aabbs must be kept valid during the lifetime of this BVH
    * @param callback f: (int i, int j) -> bool Callback predicate to filter overlapping pairs
    * @param qbuffer Output buffer to store detected overlapping pairs
    */
    template <std::invocable<IndexT, IndexT> Pred = DefaultQueryCallback>
    void query(luisa::compute::BufferView<AABB> aabbs, Pred callback, QueryBuffer& qbuffer);


  public:
    class Impl
    {
      public:
        Impl(luisa::compute::Device& d) : device{d} {}
        
        void build(luisa::compute::BufferView<AABB> aabbs);
        
        template <typename Pred>
        void StacklessCDSharedSelf(Pred                            pred,
                                   luisa::compute::BufferView<int32_t>  cpNum,
                                   luisa::compute::BufferView<Vector2i> buffer);
        template <typename Pred>
        void StacklessCDSharedOther(Pred                            pred,
                                    luisa::compute::BufferView<AABB>     query_aabbs,
                                    luisa::compute::BufferView<int32_t>  query_sorted_id,
                                    luisa::compute::BufferView<int32_t>  cpNum,
                                    luisa::compute::BufferView<Vector2i> buffer);


        static void calcMaxBVFromBox(luisa::compute::BufferView<AABB> aabbs,
                                     luisa::compute::BufferView<AABB> scene_box);
        static void calcMCsFromBox(luisa::compute::BufferView<AABB>     aabbs,
                                   luisa::compute::BufferView<AABB>     scene_box,
                                   luisa::compute::BufferView<uint32_t> codes);
        void        calcInverseMapping();
        void        buildPrimitivesFromBox(luisa::compute::BufferView<AABB> aabbs);
        void        calcExtNodeSplitMetrics();
        void        buildIntNodes(int size);
        void        calcIntNodeOrders(int size);
        void        updateBvhExtNodeLinks(int size);
        void        reorderNode(int intSize);

        luisa::compute::Device& device;
        luisa::compute::Stream  stream;

        luisa::compute::BufferView<AABB> objs;  // external AABBs, should be kept valid
        luisa::compute::Buffer<AABB>     scene_box;  // external bounding boxes
        luisa::compute::Buffer<uint32_t> flags;
        luisa::compute::Buffer<uint32_t> mtcode;  // external morton codes
        luisa::compute::Buffer<int32_t>  sorted_id;
        luisa::compute::Buffer<int32_t>  primMap;
        luisa::compute::Buffer<int32_t>  metric;
        luisa::compute::Buffer<uint32_t> count;
        luisa::compute::Buffer<int32_t>  tkMap;
        luisa::compute::Buffer<uint32_t> offsetTable;

        luisa::compute::Buffer<AABB>     ext_aabb;
        luisa::compute::Buffer<int32_t>  ext_idx;
        luisa::compute::Buffer<int32_t>  ext_lca;
        luisa::compute::Buffer<uint32_t> ext_mark;
        luisa::compute::Buffer<uint32_t> ext_par;

        luisa::compute::Buffer<int32_t>  int_lc;
        luisa::compute::Buffer<int32_t>  int_rc;
        luisa::compute::Buffer<int32_t>  int_par;
        luisa::compute::Buffer<int32_t>  int_range_x;
        luisa::compute::Buffer<int32_t>  int_range_y;
        luisa::compute::Buffer<uint32_t> int_mark;
        luisa::compute::Buffer<AABB>     int_aabb;

        luisa::compute::Buffer<luisa::ulonglong2> quantNode;
        luisa::compute::Buffer<Node>              nodes;

        Config config;
    };

  private:
    Impl m_impl;
};

}  // namespace uipc::backend::luisa

#include "details/stackless_bvh.inl"
