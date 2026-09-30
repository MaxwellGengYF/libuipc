/********************************************************************
 * @file atomic_counting_lbvh.inl
 * @brief AtomicCountingLBVH — the two ticket-counting kernels and the dispatch bodies
 *        (port of src/backends/cuda/collision_detection/details/atomic_counting_lbvh.inl).
 *
 * Included at the end of atomic_counting_lbvh.h.  `AtomicCountingLBVH::build` is the
 * only non-template body and lives in atomic_counting_lbvh.cpp.
 *
 * TRANSCRIPTION NOTES
 *  * the reference's kernels are templated on the CUDA device predicate `Pred p`, which
 *    it passes as a kernel argument; the port's predicate is a *record-time* callable
 *    (it runs while the shader is being recorded and returns a DSL expression), so it
 *    is baked into the cached shader.  The shader cache key is the predicate's type,
 *    so a predicate must be stateless or carry identical captures on every call.
 *  * the reference's `auto N = aabbs.total_size();` is computed and never used in both
 *    kernels; it is not transcribed (nothing observes it).
 *  * `cuda_tool::atomic_add(cp_num.data(), 1)` -> `cp_num.atomic(0).fetch_add(1)`;
 *    `last < pairs.total_size()` -> `last < pairs_capacity` (the pair *capacity* is a
 *    kernel scalar because an LC kernel reads a buffer's own size, and the reference's
 *    `viewer()`-of-the-whole-buffer argument is what carries it).
 *  * `p(i, id)` is called only when `id > i` (detect) — inside the callback, exactly
 *    like the reference, so the traversal cost is unchanged.
 *********************************************************************/
#pragma once

#include <luisa_device/linear/algorithms.h>

#include <cstdint>

namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

inline constexpr std::uint32_t kAtomicCountingBlock = 256u;

[[nodiscard]] inline lcrt::Var<LVector2i> atomic_counting_pair(const auto& i, const auto& j) noexcept
{
    lps::LVec<lcrt::Var<IndexT>, 2> p;
    p(0) = lcrt::cast<IndexT>(i);
    p(1) = lcrt::cast<IndexT>(j);
    lcrt::Var<LVector2i> v;
    store_vec2i(v, p);
    return v;
}

// ============================================================================
//  atomic_counting_lbvh_detect_kernel
// ============================================================================
template <typename Pred>
auto& atomic_counting_lbvh_detect_kernel(Pred pred)
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LinearBVHNode>, lcrt::Buffer<LAABB>,
                                   lcrt::Buffer<LAABB>, lcrt::Buffer<std::int32_t>,
                                   lcrt::Buffer<LVector2i>, std::uint32_t, std::uint32_t,
                                   std::uint32_t, std::uint32_t>(
            [pred](lcrt::BufferVar<LinearBVHNode> nodes, lcrt::BufferVar<LAABB> tree_aabbs,
                   lcrt::BufferVar<LAABB> query_aabbs, lcrt::BufferVar<std::int32_t> cp_num,
                   lcrt::BufferVar<LVector2i> pairs, lcrt::Var<std::uint32_t> num_nodes,
                   lcrt::Var<std::uint32_t> num_objects,
                   lcrt::Var<std::uint32_t> pairs_capacity,
                   lcrt::Var<std::uint32_t> n) noexcept
            {
                lcrt::set_block_size(kAtomicCountingBlock);
                const auto gid = lcrt::dispatch_x();
                $if(gid < n)
                {
                    const auto i = gid;
                    // the reference passes `aabbs(i)` -- the *input* AABBs are the query
                    // set, the tree's own (sorted) boxes feed the traversal
                    const auto query_box = load_aabb(query_aabbs->read(i));
                    auto       view = LinearBVHDeviceViewer{num_nodes, num_objects, nodes,
                                                            tree_aabbs};
                    [[maybe_unused]] auto found = view.query_aabb(
                        query_box,
                        [&](const auto& id) noexcept
                        {
                                        $if(id > i)
                                        {
                                            const auto ok = pred(i, id);
                                            $if(ok)
                                            {
                                                const auto last =
                                                    cp_num.atomic(0u).fetch_add(1);
                                                $if(last < pairs_capacity)
                                                {
                                                    pairs->write(last,
                                                                 atomic_counting_pair(i, id));
                                                };
                                            };
                                        };
                                    });
                    (void)found;
                };
            });
    return k;
}

// ============================================================================
//  atomic_counting_lbvh_query_kernel
// ============================================================================
template <typename Pred>
auto& atomic_counting_lbvh_query_kernel(Pred pred)
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LinearBVHNode>, lcrt::Buffer<LAABB>,
                                   lcrt::Buffer<LAABB>, lcrt::Buffer<std::int32_t>,
                                   lcrt::Buffer<LVector2i>, std::uint32_t, std::uint32_t,
                                   std::uint32_t, std::uint32_t>(
            [pred](lcrt::BufferVar<LinearBVHNode> nodes, lcrt::BufferVar<LAABB> tree_aabbs,
                   lcrt::BufferVar<LAABB> query_aabbs, lcrt::BufferVar<std::int32_t> cp_num,
                   lcrt::BufferVar<LVector2i> pairs, lcrt::Var<std::uint32_t> num_nodes,
                   lcrt::Var<std::uint32_t> num_objects,
                   lcrt::Var<std::uint32_t> pairs_capacity,
                   lcrt::Var<std::uint32_t> n) noexcept
            {
                lcrt::set_block_size(kAtomicCountingBlock);
                const auto gid = lcrt::dispatch_x();
                $if(gid < n)
                {
                    const auto i        = gid;
                    const auto query_box = load_aabb(query_aabbs->read(i));
                    auto       view = LinearBVHDeviceViewer{num_nodes, num_objects, nodes,
                                                            tree_aabbs};
                    [[maybe_unused]] auto found = view.query_aabb(
                        query_box,
                        [&](const auto& id) noexcept
                        {
                                        const auto ok = pred(i, id);
                                        $if(ok)
                                        {
                                            const auto last = cp_num.atomic(0u).fetch_add(1);
                                            $if(last < pairs_capacity)
                                            {
                                                pairs->write(last,
                                                             atomic_counting_pair(i, id));
                                            };
                                        };
                        });
                    (void)found;
                };
            });
    return k;
}
}// namespace

template <typename Pred>
void AtomicCountingLBVH::detect(Pred p, QueryBuffer& qbuffer)
{
    if(m_aabbs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    auto& s = stream();

    // LC rejects a zero-length buffer view as a kernel argument, and the reference's
    // pair buffer starts empty (ticket 0 overflows it, which is the point of the retry
    // below).  One element keeps the argument valid; the surplus tickets are dropped by
    // the kernel's own `last < pairs_capacity` guard.
    if(qbuffer.m_pairs.size() == 0u)
        qbuffer.m_pairs.resize(1u);

    auto do_query = [&]
    {
        m_cp_num.view().fill(0);

        const auto n = static_cast<std::uint32_t>(m_aabbs.size());
        if(n > 0u)
        {
            auto viewer = m_lbvh.viewer();
            s.record(atomic_counting_lbvh_detect_kernel<Pred>(p)(
                         viewer.nodes(), viewer.aabbs(), m_aabbs, m_cp_num.view(),
                         qbuffer.m_pairs.view(), viewer.num_nodes(), viewer.num_objects(),
                         static_cast<std::uint32_t>(qbuffer.m_pairs.size()), n)
                         .dispatch(n));
        }
    };

    do_query();

    // get the total number of pairs (the reference's implicit synchronous read)
    auto h_cp_num = lud::read_host_sync(s, m_cp_num.cview());

    // if it failed, resize and retry
    if(h_cp_num > static_cast<std::int32_t>(qbuffer.m_pairs.size()))
    {
        const auto new_size =
            static_cast<size_t>(static_cast<Float>(h_cp_num) * m_reserve_ratio);
        qbuffer.m_pairs.resize(new_size);
        do_query();
    }

    qbuffer.m_size = static_cast<SizeT>(h_cp_num);
}

template <typename Pred>
void AtomicCountingLBVH::query(lud::LCBufferView<LinearBVHAABB> query_aabbs,
                               Pred                            p,
                               QueryBuffer&                    qbuffer)
{
    if(m_aabbs.size() == 0 || query_aabbs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    auto& s = stream();

    // see `detect`: a zero-length pair buffer is not a valid LC kernel argument
    if(qbuffer.m_pairs.size() == 0u)
        qbuffer.m_pairs.resize(1u);

    auto do_query = [&]
    {
        m_cp_num.view().fill(0);

        const auto n = static_cast<std::uint32_t>(query_aabbs.size());
        if(n > 0u)
        {
            auto viewer = m_lbvh.viewer();
            s.record(atomic_counting_lbvh_query_kernel<Pred>(p)(
                         viewer.nodes(), viewer.aabbs(), query_aabbs, m_cp_num.view(),
                         qbuffer.m_pairs.view(), viewer.num_nodes(), viewer.num_objects(),
                         static_cast<std::uint32_t>(qbuffer.m_pairs.size()), n)
                         .dispatch(n));
        }
    };

    do_query();

    // get the total number of pairs
    auto h_cp_num = lud::read_host_sync(s, m_cp_num.cview());

    // if it failed, resize and retry
    if(h_cp_num > static_cast<std::int32_t>(qbuffer.m_pairs.size()))
    {
        const auto new_size =
            static_cast<size_t>(static_cast<Float>(h_cp_num) * m_reserve_ratio);
        qbuffer.m_pairs.resize(new_size);
        do_query();
    }

    qbuffer.m_size = static_cast<SizeT>(h_cp_num);
}
}// namespace uipc::backend::luisa
