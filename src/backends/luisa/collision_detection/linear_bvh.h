#pragma once
/********************************************************************
 * @file linear_bvh.h
 * @brief LinearBVH: a Morton-code LBVH over AABBs (port of
 *        src/backends/cuda/collision_detection/linear_bvh.h +
 *        details/linear_bvh.inl, 268 + 716 lines).
 *
 * THE REFERENCE'S NOTE, AND WHY THE PORT SPLITS DIFFERENTLY
 * --------------------------------------------------------
 * The reference is header-only on purpose: "We make the LinearBVH header-only to
 * make unit testing easier. Otherwise, if using dllexport/dllimport, cuda will throw
 * 'InvalidDeviceFunction' error" (MSVC + CUDA >= 12.4). That is a CUDA constraint —
 * the port has no nvcc, an LC Shader is compiled once on the host and dispatched
 * through an LStream, so the kernel bodies live in `linear_bvh.cpp` (the established
 * shape: mas_preconditioner_engine.{h,cpp}, cookbook §7.16). `details/linear_bvh.inl`
 * keeps everything that must be visible to a *kernel body*: the generic Morton-code
 * arithmetic, the node predicates and the device-side traversal.
 *
 * WHAT THE STRUCTURE IS (contract for M8's contact detection)
 * ----------------------------------------------------------
 *  1. `m_aabbs` holds `2n-1` boxes: the internal nodes in `[0, n-1)` and the leaves in
 *     `[n-1, 2n-1)`, i.e. "leaf i" of the sorted order lives at `aabbs[n-1+i]`.
 *  2. `LinearBVHNode` is 16 bytes: `{parent_idx, left_idx, right_idx, object_idx}`,
 *     with `0xFFFFFFFF` meaning "none" (a leaf has `object_idx != 0xFFFFFFFF`, an
 *     internal node has `object_idx == 0xFFFFFFFF` and `0xFFFFFFFF` children slots
 *     never occur).  Children are leaf slots in `[n-1, 2n-1)` or internal slots in
 *     `[0, n-2]`; the root is node 0 and `nodes(0).parent_idx == 0xFFFFFFFF`.
 *  3. The build is Karras' radix-tree construction over 30-bit Morton codes of the
 *     *normalized* box centres (normalized by the union box of all inputs,
 *     `m_max_aabb`), sorted **stably** by the 32-bit code (CUB's
 *     `DeviceRadixSort::SortPairs`); `m_new_to_old[i]` is the input index of the
 *     i-th sorted element and `m_morton_idx[i] = (sorted_morton[i] << 32) | i` —
 *     note the low word is the **sorted position**, not the object index (the
 *     reference's `linear_bvh_build_expand_morton64_kernel` ignores its `indices`
 *     argument; transcribed and pinned).
 *  4. The internal boxes are computed by the reference's bottom-up,
 *     one-merge-per-node atomic protocol (`linear_bvh_build_internal_aabbs_kernel`),
 *     not by a per-node top-down aggregation: each leaf thread walks to the root and
 *     the *second* thread to arrive at a node merges its two children.
 *  5. `viewer()` is the query handle: a stack-based traversal that reports, in
 *     traversal order, every object whose box passes the node test.  The port keeps
 *     the traversal (the candidate *order* is part of the contract) and the
 *     reference's two node tests: `query(point)` (box contains the point) and
 *     `query(aabb)` (box overlaps).
 *
 * PORT DEVIATIONS (all forced by the DSL, each documented again where it bites)
 * ---------------------------------------------------------------------------
 *  * `AABB` is host-only; device buffers hold `LAABB` (see aabb.h) — same bytes.
 *  * `LinearBVHMortonIndex` (a 64-bit word) is stored as a plain `uint64_t`: the port
 *    keeps the reference's `detail::morton_index(morton, idx)` construction.
 *  * A kernel cannot take the viewer *struct* as an argument (the reference passes
 *    `LinearBVHViewer` by value); the port passes the two buffers + the two counts and
 *    builds `LinearBVHDeviceViewer` inside the kernel body from them.
 *  * `__threadfence()` / `::cuda::atomic_thread_fence(...)` have no LC spelling, but
 *    LC's `BufferVar::volatile_read`/`volatile_write` *are* `__threadfence()` + the
 *    access in the CUDA codegen (`cuda_device_resource.h`:
 *    `lc_buffer_volatile_read` = `__threadfence(); read;`,
 *    `lc_buffer_volatile_write` = `write; __threadfence();`).  The internal-AABB
 *    kernel therefore reads its children with `volatile_read` (the reference's
 *    "atomic fetch" spin) and writes its merged box with `volatile_write` (the
 *    reference's store + the acquire fence before the next `atomicAdd`).
 *  * the reference's `LinearBVHViewer::query(..., uint32_t* stack, uint32_t stack_num)`
 *    overload exists so a *caller* can own the stack; the port's stack is an LC
 *    `Local<uint32_t>` created inside the query (a `Local` is per-thread storage), and
 *    the explicit-stack overload takes a `Local<uint32_t>&` plus the stack pointer.
 *  * `LinearBVHViewer::stack_overflow()` is a `mutable int` *member* of a viewer that
 *    is copied into the kernel, so a kernel's write is invisible on the host — it is
 *    not ported (the *traversal* consequence is: a stack that is too small truncates
 *    the candidate set, which the [bvh] test pins).
 *  * the reference's kernel assertions are `LC_LOG` (compiled out unless
 *    UIPC_LUISA_KERNEL_PRINT, exactly like the reference's release build).
 *********************************************************************/
#include <type_define.h>

#include <collision_detection/aabb.h>
#include <luisa_device/buffer.h>
#include <luisa_device/kernel.h>
#include <luisa_device/stream.h>
#include <luisa_device/view.h>

#include <cstdint>

namespace uipc::backend::luisa
{
// ===========================================================================
//  detail: the reference's `namespace detail` helpers (bodies in the .inl)
// ===========================================================================
namespace detail
{
/// The reference's `LinearBVHMortonIndex`: one 64-bit word `(morton << 32) | idx`.
/// Host (`uint64_t` from two `uint32_t`) and device (from two `Expr<uint32_t>`).
template <typename M, typename I>
[[nodiscard]] inline auto morton_index(const M& morton, const I& idx) noexcept;

/// `common_upper_bits(lhs, rhs)` == the number of leading equal bits of the two
/// 64-bit words == `__clzll(lhs ^ rhs)`.  `V` is `uint64_t` (host) or
/// `lcrt::Expr<uint64_t>`/`Var<uint64_t>` (device); the reference rejects the host
/// path (`UIPC_KERNEL_ASSERT(false, "common_upper_bits is not supported on the
/// host.")`) — the port supports it because the [bvh] oracles are host-side.
template <typename V>
[[nodiscard]] auto common_upper_bits(const V& lhs, const V& rhs) noexcept;

/// The reference's `detail::expand_bits` (a 10-bit integer expanded to 30 bits).
template <typename V>
[[nodiscard]] auto expand_bits(V&& v) noexcept;

/// The reference's `detail::morton_code(Eigen::Vector3f xyz)`.
template <typename T>
[[nodiscard]] auto morton_code(const lps::LVec<T, 3>& xyz) noexcept;

/// A u32 pair for a *device* range (`uint2` in the reference; a Var-pair here, so
/// that the two ends can be DSL values).
struct LinearBVHMortonRange
{
    lcrt::Var<std::uint32_t> x;
    lcrt::Var<std::uint32_t> y;
};

/// The reference's `detail::determine_range` (Karras): the leaf range `[x, y]` of the
/// internal node `idx`.  Device-only (`$if`/`$while`); `codes` is the
/// `m_morton_idx` buffer.
template <typename IdxT>
[[nodiscard]] LinearBVHMortonRange determine_range(
    const lcrt::BufferVar<std::uint64_t>& codes,
    lcrt::Var<std::uint32_t>              num_leaves,
    const IdxT&                           idx) noexcept;

/// The reference's `detail::find_split`: the split leaf index of the range
/// `[first, last]` (a binary search on the longest common prefix).
template <typename IdxT>
[[nodiscard]] lcrt::Var<std::uint32_t> find_split(
    const lcrt::BufferVar<std::uint64_t>& codes,
    lcrt::Var<std::uint32_t>              num_leaves,
    const IdxT&                           first,
    const IdxT&                           last) noexcept;
}// namespace detail

// ===========================================================================
//  LinearBVHNode
// ===========================================================================
class LinearBVHNode
{
  public:
    std::uint32_t parent_idx = 0xFFFFFFFFu;  // parent node
    std::uint32_t left_idx   = 0xFFFFFFFFu;  // index of left  child node
    std::uint32_t right_idx  = 0xFFFFFFFFu;  // index of right child node
    std::uint32_t object_idx = 0xFFFFFFFFu;  // == 0xFFFFFFFF if internal node.

    // Host-side predicates on a node *value* (the reference's member functions).
    // A kernel reads the fields and uses the field-level spellings below.
    [[nodiscard]] bool is_leaf() const noexcept { return object_idx != 0xFFFFFFFFu; }
    [[nodiscard]] bool is_internal() const noexcept { return object_idx == 0xFFFFFFFFu; }
    [[nodiscard]] bool is_top() const noexcept { return parent_idx == 0xFFFFFFFFu; }
};

/// Kernel-side predicates on a node *field* (`nodes->read(i).object_idx`).
[[nodiscard]] inline auto node_is_leaf(const auto& object_idx) noexcept
{
    return object_idx != 0xFFFFFFFFu;
}
[[nodiscard]] inline auto node_is_internal(const auto& object_idx) noexcept
{
    return object_idx == 0xFFFFFFFFu;
}
[[nodiscard]] inline auto node_is_top(const auto& parent_idx) noexcept
{
    return parent_idx == 0xFFFFFFFFu;
}
}// namespace uipc::backend::luisa

// ---------------------------------------------------------------------------
//  LuisaCompute struct registration (global scope, before the first
//  `Expr<LinearBVHNode>`/`Var<LinearBVHNode>` use in details/linear_bvh.inl).
//  The class is not an aggregate (it has member functions and NSDMIs), which
//  `LUISA_STRUCT` does not need: it registers the member offsets explicitly.
// ---------------------------------------------------------------------------
LUISA_STRUCT(uipc::backend::luisa::LinearBVHNode, parent_idx, left_idx, right_idx, object_idx){};

namespace uipc::backend::luisa
{

/// The reference's `using LinearBVHAABB = AABB;` -- the *device* element type here.
using LinearBVHAABB = LAABB;

/**
 * @brief Configuration for LinearBVH Tree.
 */
class LinearBVHConfig
{
  public:
    Float buffer_resize_factor = 1.5;
};

class LinearBVH;
class LinearBVHDeviceViewer;

/**
 * @brief The host-side query handle (the reference's `LinearBVHViewer`).
 *
 * The reference passes this struct *by value* into kernels (it holds raw device
 * pointers); an LC kernel argument list cannot carry a struct of buffer handles, so
 * the port splits the role: this class is the host handle handed out by
 * `LinearBVH::viewer()` (it owns the buffer views so a caller can read them back),
 * and `LinearBVHDeviceViewer` is what a kernel body constructs from the same buffers
 * to run the traversal.
 */
class LinearBVHViewer
{
  public:
    static constexpr std::uint32_t DEFAULT_STACK_SIZE = 64u;

    LinearBVHViewer(std::uint32_t                    num_nodes,
                    std::uint32_t                    num_objects,
                    lud::LCBufferView<LinearBVHNode> nodes,
                    lud::LCBufferView<LinearBVHAABB> aabbs) noexcept
        : m_num_nodes{num_nodes}
        , m_num_objects{num_objects}
        , m_nodes{nodes}
        , m_aabbs{aabbs}
    {
    }

    [[nodiscard]] std::uint32_t num_nodes() const noexcept { return m_num_nodes; }
    [[nodiscard]] std::uint32_t num_objects() const noexcept { return m_num_objects; }

    /// The kernel-argument views of the two buffers (both convert to LC's
    /// `BufferView<T>` implicitly, so `k(viewer.nodes(), viewer.aabbs(), ...)` works).
    [[nodiscard]] const lud::LCBufferView<LinearBVHNode>& nodes() const noexcept
    {
        return m_nodes;
    }
    [[nodiscard]] const lud::LCBufferView<LinearBVHAABB>& aabbs() const noexcept
    {
        return m_aabbs;
    }

  private:
    std::uint32_t                    m_num_nodes;
    std::uint32_t                    m_num_objects;
    lud::LCBufferView<LinearBVHNode> m_nodes;
    lud::LCBufferView<LinearBVHAABB> m_aabbs;
};

/**
 * @brief LinearBVH Tree class.
 */
class LinearBVH
{
    friend class LinearBVHVisitor;

  public:
    /// The reference's `detail::LinearBVHMortonIndex`, flattened to its 64 bits.
    using MortonIndex = std::uint64_t;

    /// NOTE (reference inconsistency, transcribed): the reference's
    /// `LinearBVH::LinearBVH(const LinearBVHConfig& config) noexcept {}` *ignores* the
    /// argument, so `m_config` always keeps its default member value
    /// (`buffer_resize_factor = 1.5`).  The [bvh] test pins that a non-default config
    /// changes nothing.
    explicit LinearBVH(const LinearBVHConfig& config = {}) noexcept;
    LinearBVH(const LinearBVH&)            = default;
    LinearBVH(LinearBVH&&)                 = default;
    LinearBVH& operator=(const LinearBVH&) = default;
    LinearBVH& operator=(LinearBVH&&)      = default;

    /**
     * @brief Construct LinearBVH Tree of given AABBs.
     */
    void build(lud::LCBufferView<LinearBVHAABB> aabbs, lud::LStream* s = nullptr);

    /**
     * @brief Keep the constructed LinearBVH Tree and update the AABBs.
     *
     * The `update()` performs better than `build()` because it reuses the constructed
     * LinearBVH Tree.  However, if the AABBs are significantly changed (bad time
     * coherence), you may get a Tree with bad query performance.  So, if the AABBs are
     * significantly changed, you should call `build()` instead of `update()`.
     */
    void update(lud::LCBufferView<LinearBVHAABB> aabbs, lud::LStream* s = nullptr);

    /**
     * @brief Get a query handler for the constructed LinearBVH tree.
     */
    [[nodiscard]] LinearBVHViewer viewer() const noexcept;

    // ---- test seams (the reference's state is private here too, but the unit test
    //      reaches it through the visitor; these two are the counts the contract
    //      quotes)
    [[nodiscard]] std::size_t num_objects() const noexcept { return m_indices.size(); }
    [[nodiscard]] std::size_t num_nodes() const noexcept { return m_nodes.size(); }

  private:
    template <typename T>
    void resize(lud::LStream& s, lud::LDeviceBuffer<T>& V, size_t size);

    void build_internal_aabbs(lud::LStream& s);

    lud::LDeviceBuffer<LinearBVHAABB>        m_aabbs;
    lud::LDeviceBuffer<std::uint32_t>        m_mortons;
    lud::LDeviceBuffer<std::uint32_t>        m_sorted_mortons;
    lud::LDeviceBuffer<std::uint32_t>        m_indices;
    lud::LDeviceBuffer<std::uint32_t>        m_new_to_old;
    lud::LDeviceBuffer<std::uint64_t>        m_morton_idx;
    lud::LDeviceBuffer<std::int32_t>         m_flags;
    lud::LDeviceBuffer<LinearBVHNode>        m_nodes;
    lud::LDeviceVar<LinearBVHAABB>           m_max_aabb;
    // The port's scratch for the union-box reduction (the reference's CUB
    // `DeviceReduce` owns its own temporary; `detail::reduce_aabb_union` takes a
    // caller-owned buffer so that no recorded handle can dangle -- a scratch allocated
    // inside a call would die while the frame's CommandList still references it).
    lud::LDeviceBuffer<LinearBVHAABB>        m_reduce_partials;

    LinearBVHConfig m_config;
};

/**
 * @brief Visitor class for LinearBVH, which provides the advanced information of the
 *        constructed LinearBVH tree.
 */
class LinearBVHVisitor
{
  public:
    LinearBVHVisitor(LinearBVH& bvh) noexcept
        : m_bvh{bvh}
    {
    }
    LinearBVHVisitor(const LinearBVHVisitor&)            = default;
    LinearBVHVisitor(LinearBVHVisitor&&)                 = default;
    LinearBVHVisitor& operator=(const LinearBVHVisitor&) = default;

    [[nodiscard]] lud::LCBufferView<LinearBVHNode> nodes() const noexcept;
    [[nodiscard]] lud::LCBufferView<LinearBVHNode> object_nodes() const noexcept;
    [[nodiscard]] lud::LCBufferView<LinearBVHAABB> aabbs() const noexcept;
    [[nodiscard]] lud::LCBufferView<LinearBVHAABB> top_aabb() const noexcept;

  private:
    LinearBVH& m_bvh;
};
}// namespace uipc::backend::luisa

#include "details/linear_bvh.inl"
