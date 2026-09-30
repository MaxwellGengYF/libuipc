#include <collision_detection/linear_bvh.h>

#include <collision_detection/details/aabb_reduce.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

#include <uipc/common/log.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

/*************************************************************************************************
 * T7.1a — LinearBVH (port of src/backends/cuda/collision_detection/linear_bvh.h +
 * details/linear_bvh.inl).  This file holds the *build* kernels and the host-side
 * bookkeeping; the kernel-visible parts (Morton arithmetic, the traversal) are in
 * details/linear_bvh.inl.
 *
 * KERNEL MAP (reference __global__ -> here)
 *   linear_bvh_build_morton_code_kernel          -> morton_code_kernel
 *   linear_bvh_build_init_indices_kernel         -> init_indices_kernel
 *   linear_bvh_build_expand_morton64_kernel      -> expand_morton64_kernel
 *   linear_bvh_build_leaf_nodes_kernel           -> leaf_nodes_kernel
 *   linear_bvh_build_internal_nodes_kernel       -> internal_nodes_kernel
 *   linear_bvh_build_internal_aabbs_kernel       -> internal_aabbs_kernel
 *   linear_bvh_update_leaf_aabbs_kernel          -> update_leaf_aabbs_kernel
 *   DeviceReduce(merge)/DeviceRadixSort          -> detail::reduce_aabb_union /
 *                                                   lla::sort_pairs_by_key
 *
 * Host deviations, each documented at the call site:
 *   - the reference passes *offset views* of m_nodes/m_aabbs to the leaf kernel; the
 *     port passes `leaf_start` as a scalar kernel argument (same stores);
 *   - the leaf/internal kernels dispatch whole blocks and predicate (`$if(gid < n)`),
 *     the port's standard shape;
 *   - the node array is filled with `default_node` on *every* build (the reference's
 *     `DeviceBuffer::resize(n, init)` only fills the grown tail; `LDeviceBuffer::resize`
 *     zero-initialises the tail, so without the fill a rebuild after a resize would
 *     leave `0` instead of `0xFFFFFFFF` in the root's parent slot — the fill makes the
 *     reference's first-build value the value of every build);
 *   - `build_internal_aabbs` is skipped for `num_objects <= 1` (there is no internal
 *     node; the reference launches a kernel that returns immediately, but it would have
 *     to pass a *zero-length* `flags` view, which is not a valid LC kernel argument —
 *     cookbook §7.30 rule 16).
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

constexpr std::uint32_t kBlock = 256u;

[[nodiscard]] constexpr std::uint32_t round_up(std::uint64_t n, std::uint32_t block) noexcept
{
    return static_cast<std::uint32_t>((n + block - 1u) / block * block);
}

// ============================================================================
//  linear_bvh_build_morton_code_kernel
// ============================================================================
auto& morton_code_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LAABB>, lcrt::Buffer<LAABB>,
                                               lcrt::Buffer<std::uint32_t>, std::uint32_t>(
        [](lcrt::BufferVar<LAABB> max_aabb, lcrt::BufferVar<LAABB> aabbs,
           lcrt::BufferVar<std::uint32_t> mortons, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                const auto box = load_aabb(aabbs->read(gid));
                // UIPC_KERNEL_ASSERT(aabbs(i).volume() >= 0, ...) -- debug-only in the
                // reference, LC_LOG here (compiled out unless UIPC_LUISA_KERNEL_PRINT)
                LC_LOG("Invalid AABB({}), volume {}", gid, box_volume(box));

                const auto scene = load_aabb(max_aabb->read(0u));
                const auto p     = box_center(box);
                const auto s     = box_sizes(scene);

                lps::LVec<lcrt::Var<float>, 3> q;
                for(int c = 0; c < 3; ++c)
                    q(c) = (p(c) - scene.min(c)) / s(c);

                mortons->write(gid, detail::morton_code(q));
            };
        });
    return k;
}

// ============================================================================
//  linear_bvh_build_init_indices_kernel
// ============================================================================
auto& init_indices_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::uint32_t>, std::uint32_t>(
            [](lcrt::BufferVar<std::uint32_t> indices, lcrt::Var<std::uint32_t> n) noexcept
            {
                lcrt::set_block_size(kBlock);
                auto gid = lcrt::dispatch_x();
                $if(gid < n)
                {
                    indices->write(gid, gid);
                };
            });
    return k;
}

// ============================================================================
//  linear_bvh_build_expand_morton64_kernel
// ============================================================================
auto& expand_morton64_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::uint64_t>,
                                               lcrt::Buffer<std::uint32_t>,
                                               lcrt::Buffer<std::uint32_t>, std::uint32_t>(
        [](lcrt::BufferVar<std::uint64_t> morton64s, lcrt::BufferVar<std::uint32_t> mortons,
           lcrt::BufferVar<std::uint32_t> indices, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                // NOTE (reference quirk, transcribed): the low word is the *sorted
                // position* `i`, not `indices(i)` -- the reference's kernel assigns
                // `uint32_t idx = i;` and never reads its `indices` argument.
                (void)indices;
                const auto idx = gid;
                morton64s->write(gid, detail::morton_index(mortons->read(gid), idx));
            };
        });
    return k;
}

// ============================================================================
//  linear_bvh_build_leaf_nodes_kernel
// ============================================================================
auto& leaf_nodes_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LinearBVHNode>,
                                               lcrt::Buffer<std::uint32_t>,
                                               lcrt::Buffer<LAABB>, lcrt::Buffer<LAABB>,
                                               std::uint32_t, std::uint32_t>(
        [](lcrt::BufferVar<LinearBVHNode> nodes, lcrt::BufferVar<std::uint32_t> indices,
           lcrt::BufferVar<LAABB> aabbs, lcrt::BufferVar<LAABB> sorted_aabbs,
           lcrt::Var<std::uint32_t> n, lcrt::Var<std::uint32_t> leaf_start) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                const auto object_idx = indices->read(gid);
                lcrt::Var<LinearBVHNode> node;
                node.parent_idx = 0xFFFFFFFFu;
                node.left_idx   = 0xFFFFFFFFu;
                node.right_idx  = 0xFFFFFFFFu;
                node.object_idx = object_idx;
                nodes->write(leaf_start + gid, node);
                sorted_aabbs->write(leaf_start + gid, aabbs->read(object_idx));
            };
        });
    return k;
}

// ============================================================================
//  linear_bvh_build_internal_nodes_kernel -- split into two dispatches
// ============================================================================
// The reference's kernel does three things in one __global__ body: it writes the
// internal node's own `object_idx`/`left_idx`/`right_idx`, and it writes the two
// children's `parent_idx` (`nodes(nodes(idx).left_idx).parent_idx = idx;` -- a
// single 4-byte store to a *different word* of the child's node, which the child's
// own thread may be writing concurrently: different words, so neither store is lost).
//
// LC cannot write one member of a buffer element: a store is a whole-struct store, so
// the child's `parent_idx` update is a read-modify-write of the child node.  Fused
// into one dispatch that would race with the child's own node write and could lose
// it, so the port splits the kernel into the reference's two *phases*:
//   K1 (`internal_nodes_kernel`): every internal node's object_idx/left/right_idx;
//   K2 (`internal_parents_kernel`): every internal node's two children's parent_idx.
// Same-stream dispatches are ordered, so K2's read-modify-write cannot race with K1.
// The resulting node array is bit-identical to the reference's.
auto& internal_nodes_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LinearBVHNode>,
                                               lcrt::Buffer<std::uint64_t>,
                                               std::uint32_t, std::uint32_t>(
        [](lcrt::BufferVar<LinearBVHNode> nodes, lcrt::BufferVar<std::uint64_t> morton_idx,
           lcrt::Var<std::uint32_t> num_objects, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                const auto idx = gid;
                lcrt::Var<LinearBVHNode> node;
                // `nodes(idx).object_idx = 0xFFFFFFFF;` in the reference leaves the
                // other three members at the buffer's value; an LC `Var<Node>` is
                // default-initialised to ZERO, so the parent slot has to be spelled out
                // (the root is nobody's child, so 0xFFFFFFFF is what the reference's
                // default-node fill leaves there; K2 overwrites it for the others).
                node.parent_idx = 0xFFFFFFFFu;
                node.object_idx = 0xFFFFFFFFu;  // internal node

                const auto ij    = detail::determine_range(morton_idx, num_objects, idx);
                const auto gamma = detail::find_split(morton_idx, num_objects, ij.x, ij.y);

                lcrt::Var<std::uint32_t> left  = gamma;
                lcrt::Var<std::uint32_t> right = gamma + 1u;
                // `((ij.x < ij.y) ? ij.x : ij.y) == gamma` -- a DSL ternary is not a
                // C++ `?:` (RULE M2), so the reference's min/max spelling is used
                $if(lcrt::min(ij.x, ij.y) == gamma)
                {
                    left = left + num_objects - 1u;
                };
                $if(lcrt::max(ij.x, ij.y) == gamma + 1u)
                {
                    right = right + num_objects - 1u;
                };
                node.left_idx  = left;
                node.right_idx = right;
                nodes->write(idx, node);
            };
        });
    return k;
}

auto& internal_parents_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LinearBVHNode>, std::uint32_t>(
        [](lcrt::BufferVar<LinearBVHNode> nodes, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                const auto idx = gid;
                auto left      = nodes->read(idx).left_idx;
                auto right     = nodes->read(idx).right_idx;

                lcrt::Var<LinearBVHNode> lnode = nodes->read(left);
                lnode.parent_idx               = idx;
                nodes->write(left, lnode);

                lcrt::Var<LinearBVHNode> rnode = nodes->read(right);
                rnode.parent_idx               = idx;
                nodes->write(right, rnode);
            };
        });
    return k;
}

// ============================================================================
//  linear_bvh_build_internal_aabbs_kernel
// ============================================================================
auto& internal_aabbs_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LinearBVHNode>,
                                               lcrt::Buffer<LAABB>,
                                               lcrt::Buffer<std::int32_t>, std::uint32_t,
                                               std::uint32_t>(
        [](lcrt::BufferVar<LinearBVHNode> nodes, lcrt::BufferVar<LAABB> aabbs,
           lcrt::BufferVar<std::int32_t> flags, lcrt::Var<std::uint32_t> leaf_start,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                const auto leaf_idx = gid + leaf_start;
                auto       parent   = nodes->read(leaf_idx).parent_idx;

                $while(parent != 0xFFFFFFFFu)
                {
                    // the reference: `const int old = atomic_add(&flags(parent), 1);`
                    // followed by an acquire fence.  The port keeps the fence on the
                    // (volatile) child reads below -- LC's volatile_read is
                    // `__threadfence(); read;` in its CUDA codegen.
                    const auto old = flags.atomic(parent).fetch_add(1);
                    $if(old == 0)
                    {
                        // first thread at this node: the other child has not merged yet
                        $break;
                    };
                    // LC_LOG("old={}", old);  // UIPC_KERNEL_ASSERT(old == 1, ...)

                    const auto lidx = nodes->read(parent).left_idx;
                    const auto ridx = nodes->read(parent).right_idx;

                    // the reference's `atomic_fetch`: spin until the child box is
                    // non-empty (the sibling's merge is only ordered by its own
                    // atomicAdd on this node's flag)
                    lcrt::Var<LAABB> lbox = aabbs->volatile_read(lidx);
                    $while(box_is_empty(load_aabb(lbox)))
                    {
                        lbox = aabbs->volatile_read(lidx);
                    };
                    lcrt::Var<LAABB> rbox = aabbs->volatile_read(ridx);
                    $while(box_is_empty(load_aabb(rbox)))
                    {
                        rbox = aabbs->volatile_read(ridx);
                    };

                    const auto merged = box_merged(load_aabb(lbox), load_aabb(rbox));
                    lcrt::Var<LAABB> out_box;
                    store_aabb(out_box, merged);
                    // the reference's plain store, fenced so that the next atomicAdd
                    // (on this node's parent) cannot be observed before the box is
                    aabbs->volatile_write(parent, out_box);

                    parent = nodes->read(parent).parent_idx;
                };
            };
        });
    return k;
}

// ============================================================================
//  linear_bvh_update_leaf_aabbs_kernel
// ============================================================================
auto& update_leaf_aabbs_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::uint32_t>,
                                               lcrt::Buffer<LAABB>, lcrt::Buffer<LAABB>,
                                               std::uint32_t, std::uint32_t>(
        [](lcrt::BufferVar<std::uint32_t> indices, lcrt::BufferVar<LAABB> aabbs,
           lcrt::BufferVar<LAABB> leaf_aabbs, lcrt::Var<std::uint32_t> n,
           lcrt::Var<std::uint32_t> leaf_start) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                leaf_aabbs->write(leaf_start + gid, aabbs->read(indices->read(gid)));
            };
        });
    return k;
}
}// namespace

// ===========================================================================
//  Host side
// ===========================================================================
template <typename T>
void LinearBVH::resize(lud::LStream& s, lud::LDeviceBuffer<T>& V, size_t size)
{
    if(size > V.capacity())
        V.reserve(static_cast<size_t>(size * m_config.buffer_resize_factor));
    V.resize(size);
}

LinearBVH::LinearBVH(const LinearBVHConfig& config) noexcept
{
    // NOTE (reference inconsistency, transcribed + pinned): the reference ignores the
    // argument, so `m_config` keeps its default member initialiser.
    (void)config;
}

void LinearBVH::build(lud::LCBufferView<LinearBVHAABB> aabbs, lud::LStream* s)
{
    auto& stream = s ? *s : frame_stream();

    if(aabbs.size() == 0)
        return;

    const std::uint32_t num_objects        = static_cast<std::uint32_t>(aabbs.size());
    const std::uint32_t num_internal_nodes = num_objects - 1u;
    const std::uint32_t leaf_start         = num_internal_nodes;
    const std::uint32_t num_nodes          = num_objects * 2u - 1u;

    const auto default_aabb = empty_laabb();

    resize(stream, m_aabbs, num_nodes);
    m_aabbs.view().fill(default_aabb);

    resize(stream, m_sorted_mortons, num_objects);

    resize(stream, m_indices, num_objects);
    resize(stream, m_new_to_old, num_objects);

    resize(stream, m_mortons, num_objects);
    resize(stream, m_morton_idx, num_objects);

    const LinearBVHNode default_node;
    resize(stream, m_nodes, num_nodes);
    m_nodes.view().fill(default_node);

    // 1) get max aabb -- the reference's
    //    DeviceReduce(s).Reduce(aabbs, m_max_aabb, size, merge, default_aabb)
    detail::reduce_aabb_union(stream, aabbs, m_max_aabb.view(), aabbs.size(),
                              m_reduce_partials);

    // 2) calculate the morton code
    stream.record(morton_code_kernel()(m_max_aabb.view(), aabbs, m_mortons.view(), num_objects)
                      .dispatch(round_up(num_objects, kBlock)));

    // 3) init the sort payload
    stream.record(init_indices_kernel()(m_indices.view(), num_objects)
                      .dispatch(round_up(num_objects, kBlock)));

    // 4) sort the morton codes (CUB's stable DeviceRadixSort::SortPairs)
    lla::sort_pairs_by_key<std::uint32_t, std::uint32_t>(stream,
                                                         m_mortons.cview(),
                                                         m_sorted_mortons.view(),
                                                         m_indices.cview(),
                                                         m_new_to_old.view(),
                                                         num_objects,
                                                         32u);

    // 5) expand the morton code to 64 bit (the low word is the *sorted position*)
    stream.record(
        expand_morton64_kernel()(m_morton_idx.view(), m_sorted_mortons.cview(),
                                 m_new_to_old.cview(), num_objects)
            .dispatch(round_up(num_objects, kBlock)));

    // 6) set up the leaf nodes (the reference passes offset views of m_nodes/m_aabbs;
    //    the port passes `leaf_start`)
    stream.record(leaf_nodes_kernel()(m_nodes.view(), m_new_to_old.cview(), aabbs,
                                      m_aabbs.view(), num_objects, leaf_start)
                      .dispatch(round_up(num_objects, kBlock)));

    // 7) construct the internal nodes (K1: the node fields, K2: the children's parents)
    if(num_internal_nodes > 0u)
    {
        stream.record(internal_nodes_kernel()(m_nodes.view(), m_morton_idx.cview(),
                                              num_objects, num_internal_nodes)
                          .dispatch(round_up(num_internal_nodes, kBlock)));
        stream.record(internal_parents_kernel()(m_nodes.view(), num_internal_nodes)
                          .dispatch(round_up(num_internal_nodes, kBlock)));
    }

    // 8) calculate the AABB of the internal nodes
    build_internal_aabbs(stream);
}

void LinearBVH::update(lud::LCBufferView<LinearBVHAABB> aabbs, lud::LStream* s)
{
    auto& stream = s ? *s : frame_stream();

    UIPC_ASSERT(aabbs.size() == m_indices.size(),
                "To update the tree, the number of input AABBs({}) must be the same as "
                "the original number of objects ({}) in the BVH.",
                aabbs.size(),
                m_indices.size());

    if(aabbs.size() == 0)
        return;

    const std::uint32_t num_objects = static_cast<std::uint32_t>(aabbs.size());
    const std::uint32_t leaf_start  = num_objects - 1u;

    // 1) update the leaf aabbs
    stream.record(update_leaf_aabbs_kernel()(m_new_to_old.cview(), aabbs, m_aabbs.view(),
                                             num_objects, leaf_start)
                      .dispatch(round_up(num_objects, kBlock)));

    // 2) update the internal aabbs
    build_internal_aabbs(stream);
}

void LinearBVH::build_internal_aabbs(lud::LStream& stream)
{
    const auto num_objects        = m_indices.size();
    const auto num_internal_nodes = m_nodes.size() - num_objects;

    if(num_objects <= 1u || num_internal_nodes == 0u)
        return;  // no internal node (a single leaf needs no aggregation)

    resize(stream, m_flags, num_internal_nodes);
    m_flags.view().fill(0);

    const auto leaf_start = static_cast<std::uint32_t>(num_internal_nodes);
    const auto n          = static_cast<std::uint32_t>(num_objects);

    stream.record(internal_aabbs_kernel()(m_nodes.view(), m_aabbs.view(), m_flags.view(),
                                          leaf_start, n)
                      .dispatch(round_up(n, kBlock)));
}
}// namespace uipc::backend::luisa
