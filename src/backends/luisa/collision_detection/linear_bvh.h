/*****************************************************************/ /**
 * \file   linear_bvh.h
 * \brief  The LinearBVH class and its viewer class (LuisaCompute version).
 * 
 * Refactored from CUDA/muda backend to LuisaCompute backend.
 * 
 * \author MuGdxy (original), refactored for LuisaCompute
 * \date   September 2024
 *********************************************************************/

#pragma once
#include <type_define.h>
#include <collision_detection/aabb.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/device.h>
#include <luisa/dsl/sugar.h>

namespace uipc::backend::luisa
{
namespace detail
{
    struct LinearBVHMortonIndex
    {
        UIPC_GENERIC LinearBVHMortonIndex(uint32_t m, uint32_t idx) noexcept;

        UIPC_GENERIC LinearBVHMortonIndex() noexcept = default;

        UIPC_GENERIC operator uint64_t() const noexcept;

      private:
        friend UIPC_GENERIC bool operator==(const LinearBVHMortonIndex& lhs,
                                            const LinearBVHMortonIndex& rhs) noexcept;
        uint64_t                 m_morton_index = 0;
    };

    UIPC_GENERIC bool operator==(const LinearBVHMortonIndex& lhs,
                                 const LinearBVHMortonIndex& rhs) noexcept;
}  // namespace detail

class LinearBVHNode
{
  public:
    uint32_t parent_idx = 0xFFFFFFFF;  // parent node
    uint32_t left_idx   = 0xFFFFFFFF;  // index of left  child node
    uint32_t right_idx  = 0xFFFFFFFF;  // index of right child node
    uint32_t object_idx = 0xFFFFFFFF;  // == 0xFFFFFFFF if internal node.

    UIPC_GENERIC bool is_leaf() const noexcept;
    UIPC_GENERIC bool is_internal() const noexcept;
    UIPC_GENERIC bool is_top() const noexcept;
};

using LinearBVHAABB = AABB;

class LinearBVH;

/**
 * @brief Viewer class for LinearBVH, used within kernels for querying.
 */
class LinearBVHViewer
{
    constexpr static uint32_t DEFAULT_STACK_SIZE = 64;

  public:
    struct DefaultQueryCallback
    {
        UIPC_GENERIC void operator()(uint32_t obj_idx) const noexcept {}
    };

    LinearBVHViewer() noexcept = default;
    
    LinearBVHViewer(uint32_t                                          num_nodes,
                    uint32_t                                          num_objects,
                    const luisa::compute::BufferView<LinearBVHNode>&  nodes,
                    const luisa::compute::BufferView<LinearBVHAABB>&  aabbs) noexcept;

    UIPC_GENERIC auto num_nodes() const noexcept { return m_num_nodes; }
    UIPC_GENERIC auto num_objects() const noexcept { return m_num_objects; }

    /**
     * @brief query AABBs that intersect with the given point q.
     * 
     * @param q query point
     * @param callback callback function that is called when an AABBs is found
     * @return the number of found AABBs
     */
    template <uint32_t StackNum = DEFAULT_STACK_SIZE, typename CallbackF = DefaultQueryCallback>
    UIPC_DEVICE uint32_t query(luisa::float3 q, CallbackF callback = DefaultQueryCallback{}) const noexcept
    {
        uint32_t stack[StackNum];
        return this->query(q, stack, StackNum, callback);
    }

    template <typename CallbackF = DefaultQueryCallback>
    UIPC_DEVICE uint32_t query(luisa::float3 q,
                               uint32_t*      stack,
                               uint32_t       stack_num,
                               CallbackF callback = DefaultQueryCallback{}) const noexcept
    {
        return this->query(
            q,
            [](const LinearBVHAABB& node, luisa::float3 q)
            { return node.contains(q); },
            stack,
            stack_num,
            callback);
    }

    /**
     * @brief query AABBs that intersect with the given AABB.
     * 
     * @param aabb query AABB
     * @param callback callback function that is called when an AABBs is found
     * @return the number of found AABBs
     */
    template <uint32_t StackNum = DEFAULT_STACK_SIZE, typename CallbackF = DefaultQueryCallback>
    UIPC_DEVICE uint32_t query(const LinearBVHAABB& aabb,
                               CallbackF callback = DefaultQueryCallback{}) const noexcept
    {
        uint32_t stack[StackNum];
        return this->query(aabb, stack, StackNum, callback);
    }

    template <typename CallbackF = DefaultQueryCallback>
    UIPC_DEVICE uint32_t query(const LinearBVHAABB& aabb,
                               uint32_t*            stack,
                               uint32_t             stack_num,
                               CallbackF callback = DefaultQueryCallback{}) const noexcept
    {
        return this->query(
            aabb,
            [](const LinearBVHAABB& node, const LinearBVHAABB& Q)
            { return node.intersects(Q); },
            stack,
            stack_num,
            callback);
    }

    /**
     * @brief check if the stack overflow occurs during the query.
     */
    UIPC_DEVICE bool stack_overflow() const noexcept;

  private:
    uint32_t m_num_nodes    = 0;   // (# of internal node) + (# of leaves), 2N+1
    uint32_t m_num_objects  = 0;   // (# of leaves), the same as the number of objects

    luisa::compute::BufferView<LinearBVHAABB> m_aabbs;
    luisa::compute::BufferView<LinearBVHNode> m_nodes;

    mutable int m_stack_overflow = false;

    UIPC_DEVICE void check_index(const uint32_t idx) const noexcept;
    UIPC_DEVICE void report_stack_overflow(uint32_t num_found, uint32_t stack_num) const noexcept;

    template <typename QueryType, typename IntersectF, typename CallbackF>
    UIPC_DEVICE uint32_t query(const QueryType& Q,
                               IntersectF       Intersect,
                               uint32_t*        stack,
                               uint32_t         stack_num,
                               CallbackF        Callback) const noexcept;
};

/**
 * @brief Configuration for LinearBVH Tree.
 */
class LinearBVHConfig
{
  public:
    Float buffer_resize_factor = 1.5;
};

/**
 * @brief LinearBVH Tree class using LuisaCompute.
 */
class LinearBVH
{
    friend class LinearBVHVisitor;

  public:
    using MortonIndex = detail::LinearBVHMortonIndex;

    LinearBVH(luisa::compute::Device& device, luisa::compute::Stream& stream, const LinearBVHConfig& config = {}) noexcept;
    LinearBVH(const LinearBVH&)            = delete;
    LinearBVH(LinearBVH&&)                 = default;
    LinearBVH& operator=(const LinearBVH&) = delete;
    LinearBVH& operator=(LinearBVH&&)      = default;

    /**
     * @brief Construct LinearBVH Tree of given AABBs.
     * 
     * @param aabbs The array of AABBs
     * @param stream The stream to execute the construction
     */
    void build(luisa::compute::BufferView<LinearBVHAABB> aabbs);

    /**
     * @brief Keep the constructed LinearBVH Tree and update the AABBs.
     * 
     * The `update()` performs better than `build()` because it reuses the constructed tree.
     * 
     * @param aabbs The array of AABBs
     * @param stream The stream to execute the update
     */
    void update(luisa::compute::BufferView<LinearBVHAABB> aabbs);

    /**
     * @brief Get a query handler for the constructed LinearBVH tree.
     */
    LinearBVHViewer viewer() const noexcept;
    
    /**
     * @brief Get the node buffer view.
     */
    luisa::compute::BufferView<LinearBVHNode> nodes_view() const noexcept;
    
    /**
     * @brief Get the AABB buffer view.
     */
    luisa::compute::BufferView<LinearBVHAABB> aabbs_view() const noexcept;

  private:
    template <typename T>
    void resize(luisa::compute::Stream& stream, luisa::compute::Buffer<T>& buffer, size_t size);

    luisa::compute::Device&  m_device;
    luisa::compute::Stream&  m_stream;
    
    luisa::compute::Buffer<LinearBVHAABB> m_aabbs;
    luisa::compute::Buffer<uint32_t>      m_mortons;
    luisa::compute::Buffer<uint32_t>      m_sorted_mortons;
    luisa::compute::Buffer<uint32_t>      m_indices;
    luisa::compute::Buffer<uint32_t>      m_new_to_old;
    luisa::compute::Buffer<MortonIndex>   m_morton_idx;
    luisa::compute::Buffer<int>           m_flags;
    luisa::compute::Buffer<LinearBVHNode> m_nodes;
    luisa::compute::Buffer<LinearBVHAABB> m_max_aabb;

    LinearBVHConfig m_config;

    void build_internal_aabbs(luisa::compute::Stream& stream);
};

/**
 * @brief Visitor class for LinearBVH, which provides advanced information of the constructed tree.
 */
class LinearBVHVisitor
{
  public:
    LinearBVHVisitor(LinearBVH& bvh) noexcept;
    LinearBVHVisitor(const LinearBVHVisitor&)            = default;
    LinearBVHVisitor(LinearBVHVisitor&&)                 = default;
    LinearBVHVisitor& operator=(const LinearBVHVisitor&) = default;

    luisa::compute::BufferView<LinearBVHNode> nodes() const noexcept;
    luisa::compute::BufferView<LinearBVHNode> object_nodes() const noexcept;
    luisa::compute::BufferView<LinearBVHAABB> aabbs() const noexcept;
    luisa::compute::BufferView<LinearBVHAABB> top_aabb() const noexcept;

  private:
    LinearBVH& m_bvh;
};

}  // namespace uipc::backend::luisa

#include "details/linear_bvh.inl"
