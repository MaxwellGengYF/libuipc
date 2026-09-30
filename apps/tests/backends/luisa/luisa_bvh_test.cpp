/********************************************************************
 * @file luisa_bvh_test.cpp
 * @brief T7.1a — the BVH batch: AtomicCountingLBVH, LinearBVH and StacklessBVH
 *        (LuisaCompute backend).
 *
 * The reference structures are CUDA-only (nvcc device lambdas + raw intrinsics), so
 * they cannot be host-included: every oracle below is derived independently *in this
 * file* from the reference's *specification* (collision_detection/{aabb,linear_bvh,
 * stackless_bvh,atomic_counting_lbvh}.{h,inl/cu} and the reference tests
 * apps/tests/backends/cuda/{lbvh.cu,stackless_bvh.cu}), never by calling the ported
 * code.
 *
 * Coverage map (case -> oracle)
 *  1) Morton codes, device vs an independent host bit-interleave .. bvh_morton_codes
 *  2) the tree: internal AABBs == the exact union of the children, the leaf
 *     permutation is a bijection *and* the stable Morton order, the
 *     node/depth/parent contract ................................. bvh_tree
 *  3) traversal equivalence against brute force (self pairs up to 4096
 *     primitives; a 100k-primitive tree against a per-query brute
 *     force) ..................................................... bvh_traversal
 *  4) AtomicCountingLBVH vs StacklessBVH on the same primitive set .. bvh_cross_structure
 *  5) determinism + the QueryBuffer capacity contract ............. bvh_determinism
 *  6) the factored generic routines, device vs host .............. bvh_device_vs_host
 *  7) the traversal stack-overflow contract ...................... bvh_stack_overflow
 *  8) per-case assertion accounting .............................. bvh_summary
 *
 * WHAT THIS FILE CANNOT REACH (stated so nothing is implied)
 * ---------------------------------------------------------
 * There is no CUDA backend in this build, so the reference's own claim ("the same
 * candidate sets as a CUDA-built tree") cannot be re-run here.  The oracles are
 * therefore (a) exact set equality against an independent brute force over the
 * primitives and (b) structural identities of the exported node array.  The
 * `h_cp_num > capacity` retry of both QueryBuffers is exercised, but `reserve()`'s
 * *capacity* is only observable through its effect (the pair set stays complete) — see
 * bvh_determinism's printed numbers.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <collision_detection/atomic_counting_lbvh.h>
#include <collision_detection/linear_bvh.h>
#include <collision_detection/stackless_bvh.h>

#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace uipc::backend::luisa
{
namespace bvh_test
{
namespace lla  = luisa_device::linalg;
namespace lcrt = ::luisa::compute;

using Vector3f = Eigen::Vector3f;

// ===========================================================================
//  Shared test helpers
// ===========================================================================
[[nodiscard]] lud::LDeviceBuffer<LAABB> upload_aabbs(const std::vector<AABB>& boxes)
{
    lud::LDeviceBuffer<LAABB> dev;
    dev.resize(boxes.size());
    std::vector<LAABB> host(boxes.size());
    for(size_t i = 0; i < boxes.size(); ++i)
        host[i] = to_luisa(boxes[i]);
    if(!host.empty())
        dev.view().copy_from(host);
    return dev;
}

[[nodiscard]] std::vector<AABB> download_aabbs(lud::LCBufferView<LAABB> view)
{
    std::vector<LAABB> tmp(view.size());
    if(!tmp.empty())
        view.copy_to(tmp);
    std::vector<AABB> out(tmp.size());
    for(size_t i = 0; i < tmp.size(); ++i)
        out[i] = to_eigen(tmp[i]);
    return out;
}

[[nodiscard]] std::vector<LinearBVHNode> download_nodes(lud::LCBufferView<LinearBVHNode> view)
{
    std::vector<LinearBVHNode> tmp(view.size());
    if(!tmp.empty())
        view.copy_to(tmp);
    return tmp;
}

/// A deterministic cloud of `n` boxes with a *mix* of sizes (so that overlaps are
/// common and the Morton codes are spread).
[[nodiscard]] std::vector<AABB> make_cloud(size_t n, std::uint32_t seed, float extent = 1.0f)
{
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> pos(-extent, extent);
    std::uniform_real_distribution<float> half(0.02f, 0.20f);
    std::vector<AABB>                     boxes(n);
    for(auto& b : boxes)
    {
        Vector3f c{pos(rng), pos(rng), pos(rng)};
        Vector3f h{half(rng), half(rng), half(rng)};
        b.min() = c - h;
        b.max() = c + h;
    }
    return boxes;
}

/// `n` identical *point* boxes (the degenerate case: the union box has zero size, so the
/// Morton normalization divides by zero).
[[nodiscard]] std::vector<AABB> make_identical_points(size_t n)
{
    const Vector3f    p{0.25f, -0.5f, 0.75f};
    std::vector<AABB> boxes(n);
    for(auto& b : boxes)
    {
        b.min() = p;
        b.max() = p;
    }
    return boxes;
}

/// `n` boxes on a line, every second one an exact duplicate of its neighbour.
[[nodiscard]] std::vector<AABB> make_line(size_t n)
{
    std::vector<AABB> boxes(n);
    for(size_t i = 0; i < n; ++i)
    {
        const auto x = static_cast<float>(i / 2u);
        boxes[i].min() = Vector3f{x, 0.0f, 0.0f};
        boxes[i].max() = Vector3f{x + 0.5f, 0.1f, 0.1f};
    }
    return boxes;
}

/// A cloud with many exact duplicates.
[[nodiscard]] std::vector<AABB> make_duplicate_aabbs(size_t n, std::uint32_t seed)
{
    std::mt19937      rng(seed);
    std::vector<AABB> pool = make_cloud(n / 4 + 1, seed, 0.5f);
    std::vector<AABB> out;
    out.reserve(n);
    while(out.size() < n)
        out.push_back(pool[rng() % pool.size()]);
    return out;
}

// ===========================================================================
//  Independent host oracles
// ===========================================================================
/// The classic bit-interleave spelling of `expand_bits` (10 bits, 2 zeros each) —
/// deliberately *not* the reference's magic-constant form.
[[nodiscard]] std::uint32_t host_part1by2(std::uint32_t x)
{
    std::uint32_t r = 0;
    for(int b = 0; b < 10; ++b)
        r |= ((x >> b) & 1u) << (3 * b);
    return r;
}

/// The reference's `expand_bits` (magic constants): the cross-check of the loop above.
[[nodiscard]] std::uint32_t host_expand_bits(std::uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/// `linear_bvh`'s `detail::morton_code(Eigen::Vector3f xyz)`:
/// clamp into [0, 1], then `uint32(x * 1024.0)` (**double** multiply), interleaved.
[[nodiscard]] std::uint32_t host_morton_code(const AABB& box, const AABB& scene)
{
    Vector3f p = box.center();
    p -= scene.min();
    p.array() /= scene.sizes().array();

    const auto comp = [](float x)
    {
        const float c = std::fmax(std::fmin(x, 1.0f), 0.0f);
        return host_part1by2(static_cast<std::uint32_t>(static_cast<double>(c) * 1024.0));
    };
    return comp(p.x()) * 4u + comp(p.y()) * 2u + comp(p.z()) * 1u;
}

/// `culbvh::morton3D(x, y, z)`: clamp `x * 1024.0f` (**float** multiply) into
/// [0, 1023], truncate, interleave.
[[nodiscard]] std::uint32_t host_morton3D(float x, float y, float z)
{
    const auto comp = [](float v)
    {
        const float c = std::fmin(std::fmax(v * 1024.0f, 0.0f), 1023.0f);
        return host_part1by2(static_cast<std::uint32_t>(c));
    };
    return comp(x) * 4u + comp(y) * 2u + comp(z) * 1u;
}

/// The exact union of a box set (component-wise min/max).
[[nodiscard]] AABB host_union(const std::vector<AABB>& boxes)
{
    AABB r;  // empty
    for(const auto& b : boxes)
        r.extend(b);
    return r;
}

/// The expected `new_to_old`: a **stable** sort of 0..n-1 by the 32-bit Morton code
/// (CUB's `DeviceRadixSort::SortPairs` is stable, and the port's `sort_pairs_by_key`
/// documents the same stability by construction).
[[nodiscard]] std::vector<std::uint32_t> host_sorted_indices(const std::vector<std::uint32_t>& codes)
{
    std::vector<std::uint32_t> idx(codes.size());
    for(std::uint32_t i = 0; i < static_cast<std::uint32_t>(idx.size()); ++i)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](std::uint32_t a, std::uint32_t b)
                     { return codes[a] < codes[b]; });
    return idx;
}

using PairSet = std::set<std::pair<std::int32_t, std::int32_t>>;

/// Brute-force self pairs `(i, j)`, `i < j` (the `detect` contract).
[[nodiscard]] PairSet brute_force_self(const std::vector<AABB>& boxes)
{
    PairSet pairs;
    for(std::size_t i = 0; i < boxes.size(); ++i)
        for(std::size_t j = i + 1; j < boxes.size(); ++j)
            if(boxes[i].intersects(boxes[j]))
                pairs.emplace(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j));
    return pairs;
}

/// Brute-force query pairs `(query i, object j)` (the `query` contract: no ordering).
[[nodiscard]] PairSet brute_force_query(const std::vector<AABB>& queries,
                                       const std::vector<AABB>& objects)
{
    PairSet pairs;
    for(std::size_t i = 0; i < queries.size(); ++i)
        for(std::size_t j = 0; j < objects.size(); ++j)
            if(queries[i].intersects(objects[j]))
                pairs.emplace(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j));
    return pairs;
}

/// Read a pair buffer back as a set (the reported *order* is not part of the contract:
/// the kernels write through an atomic ticket).
[[nodiscard]] PairSet pair_set(lud::LBufferView<LVector2i> view)
{
    std::vector<LVector2i> tmp(view.size());
    if(!tmp.empty())
        view.copy_to(tmp);
    PairSet out;
    for(const auto& p : tmp)
        out.emplace(p(0), p(1));
    return out;
}

// ===========================================================================
//  The recursive structural check of the exported node array
// ===========================================================================
struct TreeStats
{
    int depth          = 0;
    int internal_nodes = 0;
    int leaves         = 0;
    int roots          = 0;  // nodes whose parent_idx == 0xFFFFFFFF
    int bad_boxes      = 0;  // an internal box != the union of its children's boxes
    int bad_links      = 0;  // a child's parent_idx != its parent
    int bad_leaves     = 0;  // leaf box/object_idx mismatch
};

[[nodiscard]] TreeStats check_tree(const std::vector<LinearBVHNode>& nodes,
                                   const std::vector<AABB>&          tree_aabbs,
                                   const std::vector<AABB>&          input_boxes,
                                   const std::vector<std::uint32_t>& order)
{
    TreeStats st;
    const int n = static_cast<int>(order.size());
    if(n == 0)
        return st;

    for(const auto& nd : nodes)
    {
        if(nd.parent_idx == 0xFFFFFFFFu)
            ++st.roots;
        else if(nd.parent_idx >= nodes.size())
            ++st.bad_links;
    }

    // leaves: object_idx == order[i] and the leaf box == the input box
    for(int i = 0; i < n; ++i)
    {
        const auto& nd = nodes[n - 1 + i];
        if(nd.object_idx != order[i] || nd.left_idx != 0xFFFFFFFFu
           || nd.right_idx != 0xFFFFFFFFu)
            ++st.bad_leaves;
        const auto& want = input_boxes[order[i]];
        const auto& got  = tree_aabbs[n - 1 + i];
        if(!(want.min() == got.min()) || !(want.max() == got.max()))
            ++st.bad_leaves;
    }

    // internal nodes: box == the exact union of the two children boxes, recursively
    std::function<int(int)> visit = [&](int idx) -> int
    {
        const auto& nd = nodes[idx];
        if(nd.object_idx != 0xFFFFFFFFu)
        {
            ++st.leaves;
            return 0;
        }
        ++st.internal_nodes;
        const int dl = visit(static_cast<int>(nd.left_idx));
        const int dr = visit(static_cast<int>(nd.right_idx));

        const AABB merged = tree_aabbs[nd.left_idx].merged(tree_aabbs[nd.right_idx]);
        const AABB got    = tree_aabbs[idx];
        if(!(merged.min() == got.min()) || !(merged.max() == got.max()))
            ++st.bad_boxes;

        if(nodes[nd.left_idx].parent_idx != static_cast<std::uint32_t>(idx)
           || nodes[nd.right_idx].parent_idx != static_cast<std::uint32_t>(idx))
            ++st.bad_links;
        return std::max(dl, dr) + 1;
    };
    st.depth = visit(0);
    return st;
}

// ===========================================================================
//  Test-local kernels
// ===========================================================================
/// `detail::morton_code` over the build's own normalization.
auto& morton_code_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LAABB>, lcrt::Buffer<LAABB>,
                                               lcrt::Buffer<std::uint32_t>, std::uint32_t>(
        [](lcrt::BufferVar<LAABB> aabbs, lcrt::BufferVar<LAABB> scene_box,
           lcrt::BufferVar<std::uint32_t> codes, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto box   = load_aabb(aabbs->read(i));
                const auto scene = load_aabb(scene_box->read(0u));
                const auto p     = box_center(box);
                const auto s     = box_sizes(scene);
                lps::LVec<lcrt::Var<float>, 3> q;
                for(int c = 0; c < 3; ++c)
                    q(c) = (p(c) - scene.min(c)) / s(c);
                codes->write(i, detail::morton_code(q));
            };
        });
    return k;
}

/// `culbvh::morton3D` on a flat `(x, y, z)` triple array.
auto& morton3d_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<float>,
                                               lcrt::Buffer<std::uint32_t>, std::uint32_t>(
        [](lcrt::BufferVar<float> xyz, lcrt::BufferVar<std::uint32_t> out,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                out->write(i,
                           culbvh::morton3D(xyz->read(i * 3u), xyz->read(i * 3u + 1u),
                                            xyz->read(i * 3u + 2u)));
            };
        });
    return k;
}

/// `detail::common_upper_bits` on 64-bit words (`words(i)` vs `words(i+1)`).
auto& common_upper_bits_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::uint64_t>,
                                               lcrt::Buffer<std::uint32_t>, std::uint32_t>(
        [](lcrt::BufferVar<std::uint64_t> words, lcrt::BufferVar<std::uint32_t> out,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                out->write(i, detail::common_upper_bits(words->read(i), words->read(i + 1u)));
            };
        });
    return k;
}

/// The device viewer's AABB query, written into a ticket-counted pair array.
/// `counts[0]` is the total candidate count, `counts[n + i]` the per-query hit count.
template <std::uint32_t StackNum = LinearBVHDeviceViewer::DEFAULT_STACK_SIZE>
auto& viewer_query_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LinearBVHNode>,
                                               lcrt::Buffer<LAABB>, lcrt::Buffer<LAABB>,
                                               lcrt::Buffer<std::uint32_t>,
                                               lcrt::Buffer<LVector2i>, std::uint32_t,
                                               std::uint32_t, std::uint32_t>(
        [](lcrt::BufferVar<LinearBVHNode> nodes, lcrt::BufferVar<LAABB> tree_aabbs,
           lcrt::BufferVar<LAABB> queries, lcrt::BufferVar<std::uint32_t> counts,
           lcrt::BufferVar<LVector2i> pair_out, lcrt::Var<std::uint32_t> num_nodes,
           lcrt::Var<std::uint32_t> num_objects, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto view = LinearBVHDeviceViewer{num_nodes, num_objects, nodes,
                                                        tree_aabbs};
                lcrt::Var<std::uint32_t> hits{0u};
                [[maybe_unused]] auto    found =
                    view.template query_aabb<StackNum>(load_aabb(queries->read(i)),
                                                       [&](const auto& id) noexcept
                                                       {
                                                       hits = hits + 1u;
                                                       auto last =
                                                           counts.atomic(0u).fetch_add(1u);
                                                       $if(last < pair_out->size())
                                                       {
                                                           pair_out->write(
                                                               last,
                                                               culbvh::make_pair_var(i, id));
                                                       };
                                                       });
                (void)found;
                counts->write(n + i, hits);
            };
        });
    return k;
}
}// namespace bvh_test
}// namespace uipc::backend::luisa

using namespace uipc::backend::luisa;
using namespace uipc::backend::luisa::bvh_test;

// ===========================================================================
//  1) Morton codes, device vs an independent host interleave
// ===========================================================================
TEST_CASE("bvh_morton_codes", "[bvh]")
{
    test_runtime();

    // the two spellings of expand_bits agree on all 1024 representable inputs
    {
        int agree = 0;
        for(std::uint32_t v = 0; v < 1024u; ++v)
            if(host_part1by2(v) == host_expand_bits(v))
                ++agree;
        CHECK(agree == 1024);
    }
    CHECK(host_part1by2(0u) == 0u);
    CHECK(host_part1by2(1u) == 1u);
    CHECK(host_part1by2(2u) == 8u);
    CHECK(host_part1by2(1023u) == 0x09249249u);

    struct Case
    {
        std::vector<AABB> boxes;
        const char*       name;
    };
    std::vector<Case> cases;
    cases.push_back({make_cloud(300, 11u), "cloud300"});
    cases.push_back({make_identical_points(64), "identical_points"});
    cases.push_back({make_line(64), "line"});
    cases.push_back({make_duplicate_aabbs(128, 12u), "duplicates"});
    cases.push_back({make_cloud(1, 13u), "one"});

    for(const auto& c : cases)
    {
        auto dev   = upload_aabbs(c.boxes);
        auto scene = upload_aabbs({host_union(c.boxes)});

        lud::LDeviceBuffer<std::uint32_t> codes;
        codes.resize(c.boxes.size());
        auto& stream = frame_stream();
        stream.record(morton_code_kernel()(dev.view(), scene.view(), codes.view(),
                                           static_cast<std::uint32_t>(c.boxes.size()))
                          .dispatch(static_cast<std::uint32_t>(c.boxes.size())));
        stream.flush();

        std::vector<std::uint32_t> got(c.boxes.size());
        codes.cview().copy_to(got);

        const auto scene_box = host_union(c.boxes);
        int        mismatch  = 0;
        std::set<std::uint32_t> uniq;
        for(size_t i = 0; i < c.boxes.size(); ++i)
        {
            if(got[i] != host_morton_code(c.boxes[i], scene_box))
                ++mismatch;
            uniq.insert(got[i]);
        }
        CHECK(mismatch == 0);
        std::printf("[bvh] morton %s: n=%zu distinct_codes=%zu first=%u\n",
                    c.name,
                    c.boxes.size(),
                    uniq.size(),
                    got.empty() ? 0u : got.front());
    }

    // the degenerate point cloud: the union box is a point, the normalization is
    // 0/0 = NaN, and `fminf(NaN, 1)` is 1 while `fmaxf(1, 0)` is 1 -> ONE code
    {
        const auto boxes = make_identical_points(64);
        CHECK(host_morton_code(boxes[0], host_union(boxes))
              == host_morton_code(boxes[63], host_union(boxes)));
    }
}

// ===========================================================================
//  2) the tree
// ===========================================================================
TEST_CASE("bvh_tree", "[bvh]")
{
    test_runtime();

    struct Case
    {
        std::vector<AABB> boxes;
        const char*       name;
    };
    std::vector<Case> cases;
    cases.push_back({make_cloud(1, 21u), "n=1"});
    cases.push_back({make_cloud(2, 22u), "n=2"});
    cases.push_back({make_cloud(3, 23u), "n=3"});
    cases.push_back({make_cloud(1024, 24u), "n=1024"});
    cases.push_back({make_identical_points(37), "identical"});
    cases.push_back({make_line(129), "line"});

    for(const auto& c : cases)
    {
        auto dev = upload_aabbs(c.boxes);

        LinearBVH lbvh;
        lbvh.build(dev.view());
        frame_stream().flush();

        LinearBVHVisitor visitor{lbvh};
        const auto       nodes  = download_nodes(visitor.nodes());
        const auto       aabbs  = download_aabbs(visitor.aabbs());
        const auto       leaves = download_nodes(visitor.object_nodes());
        std::vector<std::uint32_t> order;
        for(const auto& nd : leaves)
            order.push_back(nd.object_idx);

        const auto n = c.boxes.size();

        // the node-count rule: 2n-1 nodes, n leaves at [n-1, 2n-1)
        CHECK(nodes.size() == 2 * n - 1);
        CHECK(order.size() == n);
        CHECK(lbvh.num_objects() == n);
        CHECK(lbvh.num_nodes() == 2 * n - 1);

        // the leaf permutation is a bijection *and* the stable Morton order
        {
            std::set<std::uint32_t> seen(order.begin(), order.end());
            CHECK(seen.size() == n);

            const auto        scene = host_union(c.boxes);
            std::vector<std::uint32_t> codes(n);
            for(size_t i = 0; i < n; ++i)
                codes[i] = host_morton_code(c.boxes[i], scene);
            const auto want = host_sorted_indices(codes);
            int        same = 0;
            for(size_t i = 0; i < n; ++i)
                if(order[i] == want[i])
                    ++same;
            CHECK(same == static_cast<int>(n));
        }

        // every internal box is the exact union of its children (recursively), the
        // links agree, exactly the root has no parent
        const auto stats = check_tree(nodes, aabbs, c.boxes, order);
        CHECK(stats.bad_boxes == 0);
        CHECK(stats.bad_links == 0);
        CHECK(stats.bad_leaves == 0);
        CHECK(stats.internal_nodes == static_cast<int>(n) - 1);
        CHECK(stats.leaves == static_cast<int>(n));
        CHECK(stats.roots == 1);
        if(n > 1)
        {
            CHECK(nodes[0].object_idx == 0xFFFFFFFFu);
            CHECK(nodes[0].parent_idx == 0xFFFFFFFFu);
        }

        std::printf("[bvh] tree %s: n=%zu nodes=%zu depth=%d internal=%d roots=%d\n",
                    c.name,
                    n,
                    nodes.size(),
                    stats.depth,
                    stats.internal_nodes,
                    stats.roots);

        // the union box (the tree's top_aabb) is the exact union of the inputs
        {
            const auto top  = download_aabbs(visitor.top_aabb());
            const auto want = host_union(c.boxes);
            CHECK(top.size() == 1);
            CHECK(top[0].min() == want.min());
            CHECK(top[0].max() == want.max());
        }

        // `update()` with the same boxes reproduces the identical node array
        lbvh.update(dev.view());
        frame_stream().flush();
        const auto nodes2 = download_nodes(LinearBVHVisitor(lbvh).nodes());
        int        equal  = 0;
        for(size_t i = 0; i < nodes.size(); ++i)
            if(nodes[i].parent_idx == nodes2[i].parent_idx
               && nodes[i].left_idx == nodes2[i].left_idx
               && nodes[i].right_idx == nodes2[i].right_idx
               && nodes[i].object_idx == nodes2[i].object_idx)
                ++equal;
        CHECK(equal == static_cast<int>(nodes.size()));

        // the config argument is ignored (the reference's constructor body is empty)
        LinearBVHConfig cfg;
        cfg.buffer_resize_factor = 7.0;
        LinearBVH lbvh2{cfg};
        lbvh2.build(dev.view());
        frame_stream().flush();
        const auto nodes3 = download_nodes(LinearBVHVisitor(lbvh2).nodes());
        int        equal3 = 0;
        for(size_t i = 0; i < nodes.size(); ++i)
            if(nodes[i].parent_idx == nodes3[i].parent_idx
               && nodes[i].left_idx == nodes3[i].left_idx
               && nodes[i].right_idx == nodes3[i].right_idx
               && nodes[i].object_idx == nodes3[i].object_idx)
                ++equal3;
        CHECK(equal3 == static_cast<int>(nodes.size()));
    }
}

// ===========================================================================
//  3) traversal equivalence against brute force
// ===========================================================================
TEST_CASE("bvh_traversal", "[bvh]")
{
    test_runtime();

    struct Case
    {
        std::vector<AABB> boxes;
        const char*       name;
        int               query_stride;
    };
    std::vector<Case> cases;
    cases.push_back({make_cloud(1, 31u), "n=1", 1});
    cases.push_back({make_cloud(2, 32u), "n=2", 1});
    cases.push_back({make_cloud(37, 33u), "n=37", 1});
    cases.push_back({make_identical_points(64), "identical", 1});
    cases.push_back({make_duplicate_aabbs(211, 34u), "duplicates", 1});
    cases.push_back({make_cloud(4096, 35u), "n=4096", 4});
    // the plan's 100k-primitive case: a *sparse* cloud so that the pair counts stay
    // bounded (extent 4.0 instead of 1.0)
    cases.push_back({make_cloud(100000, 36u, 4.0f), "n=100k", 64});

    for(const auto& c : cases)
    {
        auto dev = upload_aabbs(c.boxes);

        // the query set: every `query_stride`-th box (a *subset* of the primitives, so
        // hits and misses both occur)
        std::vector<AABB> queries;
        for(size_t i = 0; i < c.boxes.size(); i += static_cast<size_t>(c.query_stride))
            queries.push_back(c.boxes[i]);
        auto qdev = upload_aabbs(queries);

        // ---- AtomicCountingLBVH::query vs brute force
        {
            AtomicCountingLBVH              bvh;
            AtomicCountingLBVH::QueryBuffer qbuffer;
            bvh.build(dev.view());
            bvh.query(qdev.view(), [](const auto&, const auto&) noexcept { return true; },
                      qbuffer);

            const auto got  = pair_set(qbuffer.view());
            const auto want = brute_force_query(queries, c.boxes);
            CHECK(got == want);
            CHECK(got.size() == want.size());
            CHECK(qbuffer.size() == static_cast<SizeT>(want.size()));
            std::printf("[bvh] query %s: queries=%zu pairs=%zu (brute=%zu)\n",
                        c.name,
                        queries.size(),
                        got.size(),
                        want.size());
        }

        // ---- the device viewer, driven from a test-local kernel (the same traversal,
        //      without the AtomicCountingLBVH wrapper)
        {
            LinearBVH lbvh;
            lbvh.build(dev.view());
            const auto viewer = lbvh.viewer();

            lud::LDeviceBuffer<std::uint32_t> counts;
            // counts[0] is the ticket total and counts[n + i] the per-query hit count, so
            // the buffer needs 2n + 1 entries (an out-of-range kernel store here corrupts
            // whatever follows the allocation -- it cost a 10-pair mismatch when the whole
            // suite ran in one process)
            counts.resize(2 * queries.size() + 1);
            counts.fill(0u);
            const auto capacity = std::max<size_t>(1u, queries.size() * 256u);
            lud::LDeviceBuffer<LVector2i> pairs;
            pairs.resize(capacity);

            auto& stream = frame_stream();
            stream.record(viewer_query_kernel<>()(viewer.nodes(), viewer.aabbs(),
                                                  qdev.view(), counts.view(), pairs.view(),
                                                  viewer.num_nodes(), viewer.num_objects(),
                                                  static_cast<std::uint32_t>(queries.size()))
                              .dispatch(static_cast<std::uint32_t>(queries.size())));
            stream.flush();

            const auto total = counts.cview(0, 1).read_host();
            CHECK(total <= static_cast<std::uint32_t>(capacity));
            const auto got = pair_set(pairs.view(0, std::min<size_t>(total, capacity)));
            const auto want = brute_force_query(queries, c.boxes);
            CHECK(got.size() == want.size());
            CHECK(got == want);
            std::printf("[bvh] viewer %s: queries=%zu pairs=%zu (brute=%zu)\n",
                        c.name,
                        queries.size(),
                        got.size(),
                        want.size());
        }

        // ---- self pairs (`detect`) vs brute force; the O(n^2) oracle is only run up to
        //      4096 primitives (100k is covered by the per-query oracle above and by the
        //      cross-structure comparison)
        if(c.boxes.size() <= 4096u)
        {
            AtomicCountingLBVH              bvh;
            AtomicCountingLBVH::QueryBuffer qbuffer;
            bvh.build(dev.view());
            bvh.detect([](const auto&, const auto&) noexcept { return true; }, qbuffer);

            const auto got  = pair_set(qbuffer.view());
            const auto want = brute_force_self(c.boxes);
            CHECK(got == want);
            CHECK(got.size() == want.size());
            CHECK(qbuffer.size() == static_cast<SizeT>(want.size()));
            std::printf("[bvh] detect %s: pairs=%zu (brute=%zu)\n",
                        c.name,
                        got.size(),
                        want.size());
        }
    }
}

// ===========================================================================
//  4) the two structures agree
// ===========================================================================
TEST_CASE("bvh_cross_structure", "[bvh]")
{
    test_runtime();

    // a small cloud: both structures vs brute force
    {
        const auto boxes = make_cloud(2048, 41u);
        auto       dev   = upload_aabbs(boxes);

        AtomicCountingLBVH              counting;
        AtomicCountingLBVH::QueryBuffer cbuf;
        counting.build(dev.view());
        counting.detect([](const auto&, const auto&) noexcept { return true; }, cbuf);

        StacklessBVH              stackless;
        StacklessBVH::QueryBuffer sbuf;
        stackless.build(dev.view());
        stackless.detect(StacklessBVH::DefaultQueryCallback{}, sbuf);

        const auto cset = pair_set(cbuf.view());
        const auto sset = pair_set(sbuf.view());
        const auto want = brute_force_self(boxes);

        CHECK(cset.size() == want.size());
        CHECK(sset.size() == want.size());
        CHECK(cset == sset);
        CHECK(sset == want);
        std::printf("[bvh] cross(2048): counting=%zu stackless=%zu brute=%zu\n",
                    cset.size(),
                    sset.size(),
                    want.size());
    }

    // the 100k cloud: the two independent structures must agree *exactly* (the
    // reference's "same candidate sets" statement, cross-checked between the two
    // algorithms rather than against CUDA)
    {
        const auto boxes = make_cloud(100000, 42u, 4.0f);
        auto       dev   = upload_aabbs(boxes);

        AtomicCountingLBVH              counting;
        AtomicCountingLBVH::QueryBuffer cbuf;
        counting.build(dev.view());
        counting.detect([](const auto&, const auto&) noexcept { return true; }, cbuf);

        StacklessBVH              stackless;
        StacklessBVH::QueryBuffer sbuf;
        stackless.build(dev.view());
        stackless.detect(StacklessBVH::DefaultQueryCallback{}, sbuf);

        const auto cset = pair_set(cbuf.view());
        const auto sset = pair_set(sbuf.view());
        CHECK(cset.size() == sset.size());
        CHECK(cset == sset);
        std::printf("[bvh] cross(100k): counting=%zu stackless=%zu\n",
                    cset.size(),
                    sset.size());
    }

    // the stackless `query` flavour against the brute-force query oracle
    {
        const auto boxes = make_cloud(1024, 43u);
        auto       dev   = upload_aabbs(boxes);
        std::vector<AABB> queries;
        for(size_t i = 0; i < boxes.size(); i += 8)
            queries.push_back(boxes[i]);
        auto qdev = upload_aabbs(queries);

        StacklessBVH              bvh;
        StacklessBVH::QueryBuffer qbuf;
        bvh.build(dev.view());
        bvh.query(qdev.view(), StacklessBVH::DefaultQueryCallback{}, qbuf);
        const auto got = pair_set(qbuf.view());
        const auto exp = brute_force_query(queries, boxes);
        CHECK(got.size() == exp.size());
        CHECK(got == exp);
    }
}

// ===========================================================================
//  5) determinism + the capacity contract
// ===========================================================================
TEST_CASE("bvh_determinism", "[bvh]")
{
    test_runtime();

    const auto boxes = make_cloud(600, 51u);
    auto       dev   = upload_aabbs(boxes);

    // two builds of the same input on two objects are bit-identical
    LinearBVH a;
    LinearBVH b;
    a.build(dev.view());
    b.build(dev.view());
    frame_stream().flush();
    const auto na = download_nodes(LinearBVHVisitor(a).nodes());
    const auto nb = download_nodes(LinearBVHVisitor(b).nodes());
    const auto aa = download_aabbs(LinearBVHVisitor(a).aabbs());
    const auto ab = download_aabbs(LinearBVHVisitor(b).aabbs());
    int        same = 0;
    for(size_t i = 0; i < na.size(); ++i)
        if(na[i].parent_idx == nb[i].parent_idx && na[i].left_idx == nb[i].left_idx
           && na[i].right_idx == nb[i].right_idx && na[i].object_idx == nb[i].object_idx
           && (aa[i].min() == ab[i].min()) && (aa[i].max() == ab[i].max()))
            ++same;
    CHECK(same == static_cast<int>(na.size()));

    // a second build of the same object reproduces the same nodes
    a.build(dev.view());
    frame_stream().flush();
    const auto nc = download_nodes(LinearBVHVisitor(a).nodes());
    int        same2 = 0;
    for(size_t i = 0; i < na.size(); ++i)
        if(na[i].left_idx == nc[i].left_idx && na[i].right_idx == nc[i].right_idx
           && na[i].object_idx == nc[i].object_idx)
            ++same2;
    CHECK(same2 == static_cast<int>(na.size()));

    // AtomicCountingLBVH: the pair buffer starts empty; the first pass counts *all*
    // candidates, so the caller learns the exact pair count and the retry grows the
    // buffer by reserve_ratio = 1.1 before rewriting it
    {
        AtomicCountingLBVH              bvh;
        AtomicCountingLBVH::QueryBuffer qbuffer;
        CHECK(qbuffer.size() == 0);
        CHECK(qbuffer.view().size() == 0);
        bvh.build(dev.view());
        bvh.detect([](const auto&, const auto&) noexcept { return true; }, qbuffer);
        const auto want = brute_force_self(boxes);
        CHECK(qbuffer.size() == static_cast<SizeT>(want.size()));
        CHECK(qbuffer.view().size() == static_cast<SizeT>(want.size()));
        CHECK(pair_set(qbuffer.view()) == want);
        std::printf("[bvh] capacity atomic_counting: pairs=%zu, retry grows to "
                    "ceil(pairs*1.1)=%zu\n",
                    want.size(),
                    static_cast<size_t>(static_cast<double>(want.size()) * 1.1));
    }

    // StacklessBVH: the QueryBuffer constructor reserves 50*1024 pairs; `reserve(n)`
    // overrides it and an undersized buffer is grown by the config's 1.2 ratio.  A
    // deliberate 1-pair reserve must still produce the complete pair set.
    {
        StacklessBVH::QueryBuffer qbuffer;
        CHECK(qbuffer.size() == 0);
        CHECK(qbuffer.query_count() == 0);

        StacklessBVH bvh;
        bvh.build(dev.view());
        qbuffer.reserve(1);
        bvh.detect(StacklessBVH::DefaultQueryCallback{}, qbuffer);
        const auto want = brute_force_self(boxes);
        CHECK(qbuffer.size() == static_cast<SizeT>(want.size()));
        CHECK(pair_set(qbuffer.view()) == want);
        std::printf("[bvh] capacity stackless: reserved 1, pairs=%zu, retry grows to "
                    "ceil(pairs*1.2)=%zu\n",
                    want.size(),
                    static_cast<size_t>(static_cast<double>(want.size()) * 1.2));
    }

    // a repeat run reproduces the same pair set
    {
        StacklessBVH              s1;
        StacklessBVH::QueryBuffer q1;
        s1.build(dev.view());
        s1.detect(StacklessBVH::DefaultQueryCallback{}, q1);
        StacklessBVH              s2;
        StacklessBVH::QueryBuffer q2;
        s2.build(dev.view());
        s2.detect(StacklessBVH::DefaultQueryCallback{}, q2);
        CHECK(pair_set(q1.view()) == pair_set(q2.view()));
        CHECK(q1.size() == q2.size());
    }
}

// ===========================================================================
//  6) the factored generic routines, device vs host
// ===========================================================================
TEST_CASE("bvh_device_vs_host", "[bvh]")
{
    test_runtime();
    auto& stream = frame_stream();

    // detail::morton_code (linear_bvh's spelling), bit-for-bit
    {
        const auto boxes = make_cloud(257, 61u);
        const auto scene = host_union(boxes);
        auto       dev   = upload_aabbs(boxes);
        auto       sdev  = upload_aabbs({scene});

        lud::LDeviceBuffer<std::uint32_t> codes;
        codes.resize(boxes.size());
        stream.record(morton_code_kernel()(dev.view(), sdev.view(), codes.view(),
                                           static_cast<std::uint32_t>(boxes.size()))
                          .dispatch(static_cast<std::uint32_t>(boxes.size())));
        stream.flush();

        std::vector<std::uint32_t> got(boxes.size());
        codes.cview().copy_to(got);
        int mismatch = 0;
        for(size_t i = 0; i < boxes.size(); ++i)
            if(got[i] != host_morton_code(boxes[i], scene))
                ++mismatch;
        CHECK(mismatch == 0);
        std::printf("[bvh] device_vs_host morton_code: n=%zu mismatches=%d\n",
                    boxes.size(),
                    mismatch);
    }

    // culbvh::morton3D (the *other* spelling: a float multiply, a clamp to 1023)
    {
        const std::vector<float> probes{0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 0.999f, 1.0f, 1.5f};
        std::vector<float>       flat;
        for(float x : probes)
        {
            flat.push_back(x);
            flat.push_back(x * 2.0f);
            flat.push_back(x * 4.0f);
        }
        lud::LDeviceBuffer<float> in;
        in.resize(flat.size());
        in.view().copy_from(flat);
        lud::LDeviceBuffer<std::uint32_t> out;
        out.resize(probes.size());
        stream.record(morton3d_kernel()(in.view(), out.view(),
                                        static_cast<std::uint32_t>(probes.size()))
                          .dispatch(static_cast<std::uint32_t>(probes.size())));
        stream.flush();
        std::vector<std::uint32_t> got(probes.size());
        out.cview().copy_to(got);

        int mismatch = 0;
        int differing = 0;
        for(size_t i = 0; i < probes.size(); ++i)
        {
            const auto want = host_morton3D(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
            if(got[i] != want)
                ++mismatch;
            // the two reference spellings are *different functions*: compare the
            // culbvh spelling of one component with linear_bvh's
            const auto a = host_part1by2(static_cast<std::uint32_t>(
                static_cast<double>(std::fmax(std::fmin(probes[i], 1.0f), 0.0f)) * 1024.0));
            const auto b = host_part1by2(
                static_cast<std::uint32_t>(std::fmin(std::fmax(probes[i] * 1024.0f, 0.0f), 1023.0f)));
            if(a != b)
                ++differing;
        }
        CHECK(mismatch == 0);
        CHECK(differing > 0);  // x >= 1.0 clamps to 1024 vs 1023 -> different codes
        std::printf("[bvh] device_vs_host morton3D: probes=8 mismatches=%d "
                    "spellings_differing=%d\n",
                    mismatch,
                    differing);
    }

    // detail::common_upper_bits (the Karras common-prefix count), bit-for-bit
    {
        std::vector<std::uint64_t> words;
        std::mt19937_64            rng(62u);
        for(int i = 0; i < 64; ++i)
            words.push_back((rng() >> (i % 33)) | (static_cast<std::uint64_t>(i) << 32));
        words.push_back(0u);
        words.push_back(0u);
        words.push_back(0x8000000000000000ull);
        words.push_back(0x8000000000000001ull);

        lud::LDeviceBuffer<std::uint64_t> wdev;
        wdev.resize(words.size());
        wdev.view().copy_from(words);
        lud::LDeviceBuffer<std::uint32_t> od;
        od.resize(words.size());
        od.fill(12345u);
        stream.record(common_upper_bits_kernel()(wdev.view(), od.view(),
                                                 static_cast<std::uint32_t>(words.size() - 1))
                          .dispatch(static_cast<std::uint32_t>(words.size())));
        stream.flush();
        std::vector<std::uint32_t> got(words.size());
        od.cview().copy_to(got);
        int mismatch = 0;
        for(size_t i = 0; i + 1 < words.size(); ++i)
        {
            const auto want =
                static_cast<std::uint32_t>(std::countl_zero(words[i] ^ words[i + 1]));
            if(got[i] != want)
                ++mismatch;
        }
        CHECK(mismatch == 0);
        CHECK(got.back() == 12345u);  // the surplus lane wrote nothing
        std::printf("[bvh] device_vs_host common_upper_bits: n=%zu mismatches=%d\n",
                    words.size() - 1,
                    mismatch);
    }

    // the host oracles' own sanity against Eigen
    {
        const auto boxes = make_cloud(64, 63u);
        const auto u     = host_union(boxes);
        AABB       e;  // Eigen's union
        for(const auto& b : boxes)
            e.extend(b);
        CHECK(u.min() == e.min());
        CHECK(u.max() == e.max());
        const auto merged = boxes[0].merged(boxes[1]);
        CHECK(merged.min() == boxes[0].min().cwiseMin(boxes[1].min()));
        CHECK(merged.max() == boxes[0].max().cwiseMax(boxes[1].max()));
    }
}

// ===========================================================================
//  7) the traversal stack-overflow contract
// ===========================================================================
TEST_CASE("bvh_stack_overflow", "[bvh]")
{
    test_runtime();

    // A deep tree (a 512-box line) queried with a box that overlaps everything, with
    // the default 64-entry stack and with a deliberately tiny 5-entry stack.
    // The reference breaks out of the traversal when the stack is full and warns that
    // "the intersection count may be smaller than the ground truth"; the port cannot
    // report that warning back to the host (`stack_overflow()` was a mutable member of
    // a viewer copied *into* the kernel), so the contract pinned here is: the tiny
    // stack finds no more than the full stack, and fewer when the tree is deeper.
    const auto boxes = make_line(512);
    auto       dev   = upload_aabbs(boxes);

    LinearBVH lbvh;
    lbvh.build(dev.view());
    frame_stream().flush();
    const auto viewer = lbvh.viewer();

    const auto u = host_union(boxes);
    auto       qdev = upload_aabbs({u});

    const auto run = [&](bool tiny) -> std::uint32_t
    {
        lud::LDeviceBuffer<LVector2i> pairs;
        pairs.resize(boxes.size() + 1);
        lud::LDeviceBuffer<std::uint32_t> counts;
        counts.resize(3);  // counts[0] total, counts[1] per-query (n == 1)
        counts.fill(0u);

        auto& stream = frame_stream();
        if(tiny)
            stream.record(viewer_query_kernel<5u>()(viewer.nodes(), viewer.aabbs(),
                                                    qdev.view(), counts.view(), pairs.view(),
                                                    viewer.num_nodes(),
                                                    viewer.num_objects(), 1u)
                              .dispatch(32u));
        else
            stream.record(viewer_query_kernel<>()(viewer.nodes(), viewer.aabbs(),
                                                  qdev.view(), counts.view(), pairs.view(),
                                                  viewer.num_nodes(), viewer.num_objects(),
                                                  1u)
                              .dispatch(32u));
        stream.flush();
        return counts.cview(1, 1).read_host();  // the per-query hit count
    };

    const auto full = run(false);
    const auto tiny = run(true);
    std::printf("[bvh] stack_overflow: full_stack_found=%u tiny_stack_found=%u n=%zu\n",
                full,
                tiny,
                boxes.size());
    CHECK(full == static_cast<std::uint32_t>(boxes.size()));
    CHECK(tiny <= full);
    CHECK(tiny < full);  // the tiny stack really does truncate

    // the same 5-entry stack on a *shallow* tree (a 37-box cloud, depth 6) is exact:
    // the truncation above is a property of the deep tree, not a broken traversal
    {
        const auto shallow = make_cloud(37, 71u);
        auto       sdev    = upload_aabbs(shallow);
        LinearBVH  slbvh;
        slbvh.build(sdev.view());
        frame_stream().flush();
        const auto sviewer = slbvh.viewer();
        const auto su      = host_union(shallow);
        auto       squery  = upload_aabbs({su});

        const auto run_shallow = [&](bool tiny_stack) -> std::uint32_t
        {
            lud::LDeviceBuffer<LVector2i> pairs;
            pairs.resize(shallow.size() + 1);
            lud::LDeviceBuffer<std::uint32_t> counts;
            counts.resize(3);  // counts[0] total, counts[1] per-query (n == 1)
            counts.fill(0u);
            auto& stream = frame_stream();
            if(tiny_stack)
                stream.record(viewer_query_kernel<5u>()(
                                  sviewer.nodes(), sviewer.aabbs(), squery.view(),
                                  counts.view(), pairs.view(), sviewer.num_nodes(),
                                  sviewer.num_objects(), 1u)
                                  .dispatch(32u));
            else
                stream.record(viewer_query_kernel<>()(
                                  sviewer.nodes(), sviewer.aabbs(), squery.view(),
                                  counts.view(), pairs.view(), sviewer.num_nodes(),
                                  sviewer.num_objects(), 1u)
                                  .dispatch(32u));
            stream.flush();
            return counts.cview(1, 1).read_host();
        };

        const auto sfull = run_shallow(false);
        const auto stiny = run_shallow(true);
        std::printf("[bvh] stack_overflow shallow: full=%u tiny=%u n=%zu\n",
                    sfull,
                    stiny,
                    shallow.size());
        CHECK(sfull == static_cast<std::uint32_t>(shallow.size()));
        CHECK(stiny == sfull);
    }
}

// ===========================================================================
//  8) accounting
// ===========================================================================
TEST_CASE("bvh_summary", "[bvh]")
{
    std::printf("[bvh] cases: morton_codes (5 clouds), tree (6 clouds, incl. 1024), "
                "traversal (7 clouds, incl. 100k), cross_structure (2048 + 100k), "
                "determinism (600), device_vs_host (257 + 8 probes + 68 words), "
                "stack_overflow (512)\n");
    CHECK(true);
}
