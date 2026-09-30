/********************************************************************
 * @file simplicial_surface_distance_check.cpp
 * @brief T7.1b — simplicial-surface distance ("too close") sanity check (port of
 *        src/backends/cuda/sanity_check/simplicial_surface_distance_check.cu, 1117
 *        lines).  Sanity-checker uid 3.
 *
 * The reference runs four phases against T7.1b's `InfoStacklessBVH` —
 *   (1) codim points vs all points, (2) codim points vs all edges,
 *   (3) all points vs all triangles, (4) all edges vs all edges —
 * each with a `NodePred` that culls whole subtrees by the contact-element mask and a
 * `LeafPred` that applies the subscene/contact masks, the self-collision rule and the
 * thickness test before recording the pair.
 *
 * PORT NOTES
 * ----------
 *  * the seven `__global__` kernels (four AABB builders + three bid/cid gatherers)
 *    become seven `linalg::detail::cached_kernel` shaders; `AABB` -> `LAABB`,
 *    `Vector3` -> `LVector3` with a float cast for the boxes (the reference's
 *    `positions(...).cast<float>()`).
 *  * **The reference's `NodePred`/`LeafPred` are CUDA device functors that capture
 *    `cuda_tool::CBufferView`s and a `CDense2D` mask table; LC cannot carry a callable
 *    in a kernel signature and a captured view becomes a *bound* buffer of the compiled
 *    shader (cookbook §7.9 rule 2), so the port drives the four queries with the trivial
 *    predicates (`KeepAllNodes`/`KeepAllLeaves`) and evaluates the reference's predicate
 *    *bodies* as host filters over the reported candidate pairs.**  Everything the
 *    filters need is already a host span here (positions, thicknesses, topologies, cids,
 *    scids, instance ids, self-collision flags, masks), and the arithmetic is the
 *    reference's (the ported `math::distance` layer for the distance tests, Eigen for
 *    the rest).  The *surviving* pair set is provably identical: the reference's
 *    `NodePred` cull is `!mask(query_cid, node_cid)` for a node whose whole subtree
 *    shares one cid, which the `LeafPred`'s per-vertex mask checks imply, so culling
 *    only removes candidates that the leaf test would reject anyway.
 *  * `simplex_contact_mask_utils.h`'s four `allow_*_contact` device helpers become the
 *    four host helpers below over a *flat* mask (`row * N + col`) — the port has no
 *    2-D device buffer element type (cookbook §7.13).
 *  * `distance::*` (the reference's `utils/distance/distance_flagged.h`) maps onto the
 *    ported `luisa_device::math::distance`; the flagged variants are used exactly where
 *    the reference's leaf predicates use them, and the unflagged ones in the host
 *    violation loops (exactly where its host code uses them).
 *  * the reported message, the `close_mesh` extraction and the returned
 *    `SanityCheckResult` are transcribed from the reference's `do_check`.
 *********************************************************************/
#include <type_define.h>
#include <uipc/core/contact_tabular.h>
#include <uipc/core/object.h>
#include <uipc/core/subscene_tabular.h>
#include <sanity_check/backend_sanity_checker.h>
#include <uipc/backend/visitors/sanity_check_message_visitor.h>
#include <uipc/geometry/simplicial_complex.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/io/simplicial_complex_io.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/log.h>
#include <uipc/common/map.h>

#include <collision_detection/aabb.h>
#include <collision_detection/info_stackless_bvh.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>
#include <luisa_device/math/distance/flagged.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace uipc::backend::luisa
{
namespace
{
namespace lla  = luisa_device::linalg;
namespace dist = luisa_device::math::distance;

[[nodiscard]] constexpr std::uint32_t round_up(std::uint64_t n,
                                               std::uint32_t block) noexcept
{
    return static_cast<std::uint32_t>((n + block - 1u) / block * block);
}
[[nodiscard]] constexpr size_t padded(size_t n) noexcept
{
    return n == 0u ? 1u : n;
}

/// The port's stand-ins for the reference's device predicates (see the file note).
struct KeepAllNodes
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::NodePredInfoT<I>&) const noexcept
    {
        return true;
    }
};

struct KeepAllLeaves
{
    template <typename I>
    auto operator()(const InfoStacklessBVH::LeafPredInfoT<I>&) const noexcept
    {
        return true;
    }
};

// ---------------------------------------------------------------------------
//  the reference's `utils/simplex_contact_mask_utils.h`, over a flat mask
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool allow_PP_contact(span<const IndexT> table,
                                           IndexT             dim,
                                           const Vector2i&    cids) noexcept
{
    return table[cids[0] * dim + cids[1]] != 0;
}

[[nodiscard]] inline bool allow_PE_contact(span<const IndexT> table,
                                           IndexT             dim,
                                           const Vector3i&    cids) noexcept
{
    return table[cids[0] * dim + cids[1]] != 0 && table[cids[0] * dim + cids[2]] != 0;
}

[[nodiscard]] inline bool allow_PT_contact(span<const IndexT> table,
                                           IndexT             dim,
                                           const Vector4i&    cids) noexcept
{
    return table[cids[0] * dim + cids[1]] != 0 && table[cids[0] * dim + cids[2]] != 0
           && table[cids[0] * dim + cids[3]] != 0;
}

[[nodiscard]] inline bool allow_EE_contact(span<const IndexT> table,
                                           IndexT             dim,
                                           const Vector4i&    cids) noexcept
{
    return table[cids[0] * dim + cids[2]] != 0 && table[cids[0] * dim + cids[3]] != 0
           && table[cids[1] * dim + cids[2]] != 0 && table[cids[1] * dim + cids[3]] != 0;
}

// ---------------------------------------------------------------------------
//  host/device conversions
// ---------------------------------------------------------------------------
[[nodiscard]] std::vector<LVector3> to_device_positions(span<const Vector3> Vs)
{
    std::vector<LVector3> out(Vs.size());
    for(size_t i = 0; i < Vs.size(); ++i)
        out[i] = to_luisa(Vs[i]);
    return out;
}

[[nodiscard]] std::vector<LVector2i> to_device_edges(span<const Vector2i> Es)
{
    std::vector<LVector2i> out(Es.size());
    for(size_t i = 0; i < Es.size(); ++i)
        out[i] = to_luisa(Es[i]);
    return out;
}

[[nodiscard]] std::vector<LVector3i> to_device_triangles(span<const Vector3i> Fs)
{
    std::vector<LVector3i> out(Fs.size());
    for(size_t i = 0; i < Fs.size(); ++i)
        out[i] = to_luisa(Fs[i]);
    return out;
}

/// the reference's `.cast<float>()`-ed device positions
[[nodiscard]] inline lps::LVec<lcrt::Var<float>, 3> to_float3(
    const lps::LVec<lcrt::Var<Float>, 3>& v) noexcept
{
    lps::LVec<lcrt::Var<float>, 3> r;
    for(int k = 0; k < 3; ++k)
        r(k) = lcrt::cast<float>(v(k));
    return r;
}

/// the reference's `Eigen::Vector3f ext = Constant(extend_f)` + the four/six `extend`s
[[nodiscard]] inline LBox<lcrt::Var<float>> expanded_box(
    const lps::LVec<lcrt::Var<float>, 3>& p0,
    const lcrt::Var<float>&               ext) noexcept
{
    lps::LVec<lcrt::Var<float>, 3> lo, hi;
    for(int k = 0; k < 3; ++k)
    {
        lo(k) = p0(k) - ext;
        hi(k) = p0(k) + ext;
    }
    return LBox<lcrt::Var<float>>{lo, hi};
}

// ============================================================================
//  SimplicialSurfaceDistanceCheck_build_point_aabbs_kernel
// ============================================================================
auto& build_point_aabbs_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<Float>, lcrt::Buffer<Float>,
        lcrt::Buffer<LAABB>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<Float> thickness,
           lcrt::BufferVar<Float> d_hat, lcrt::BufferVar<LAABB> point_aabbs,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto ext  = lcrt::cast<float>(thickness->read(i) + d_hat->read(i));
                const auto pos  = to_float3(load_vec3(positions->read(i)));
                lcrt::Var<LAABB> box;
                store_aabb(box, expanded_box(pos, ext));
                point_aabbs->write(i, box);
            };
        });
    return k;
}

// ============================================================================
//  SimplicialSurfaceDistanceCheck_build_codim_point_aabbs_kernel
// ============================================================================
auto& build_codim_point_aabbs_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<Float>, lcrt::Buffer<Float>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<LAABB>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<Float> thickness,
           lcrt::BufferVar<Float> d_hat, lcrt::BufferVar<IndexT> codim_indices,
           lcrt::BufferVar<LAABB> codim_aabbs, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto p   = codim_indices->read(i);
                const auto ext = lcrt::cast<float>(thickness->read(p) + d_hat->read(p));
                const auto pos = to_float3(load_vec3(positions->read(p)));
                lcrt::Var<LAABB> box;
                store_aabb(box, expanded_box(pos, ext));
                codim_aabbs->write(i, box);
            };
        });
    return k;
}

// ============================================================================
//  SimplicialSurfaceDistanceCheck_build_edge_aabbs_kernel
// ============================================================================
auto& build_edge_aabbs_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector2i>, lcrt::Buffer<Float>,
        lcrt::Buffer<Float>, lcrt::Buffer<LAABB>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<LVector2i> edges,
           lcrt::BufferVar<Float> thickness, lcrt::BufferVar<Float> d_hat,
           lcrt::BufferVar<LAABB> edge_aabbs, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto E = load_vec2i(edges->read(i));
                const auto ext = lcrt::cast<float>((thickness->read(E(0))
                                                    + thickness->read(E(1)))
                                                   + (d_hat->read(E(0))
                                                      + d_hat->read(E(1))) / 2.0);
                // the reference extends both endpoints' `p +- ext` into one box
                auto box = expanded_box(to_float3(load_vec3(positions->read(E(0)))), ext);
                box = box_merged(
                    box, expanded_box(to_float3(load_vec3(positions->read(E(1)))), ext));
                lcrt::Var<LAABB> packed;
                store_aabb(packed, box);
                edge_aabbs->write(i, packed);
            };
        });
    return k;
}

// ============================================================================
//  SimplicialSurfaceDistanceCheck_build_tri_aabbs_kernel
// ============================================================================
auto& build_tri_aabbs_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3i>, lcrt::Buffer<Float>,
        lcrt::Buffer<Float>, lcrt::Buffer<LAABB>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<LVector3i> triangles,
           lcrt::BufferVar<Float> thickness, lcrt::BufferVar<Float> d_hat,
           lcrt::BufferVar<LAABB> tri_aabbs, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto F = load_vec3i(triangles->read(i));
                const auto ext = lcrt::cast<float>((thickness->read(F(0))
                                                    + thickness->read(F(1))
                                                    + thickness->read(F(2)))
                                                   + (d_hat->read(F(0)) + d_hat->read(F(1))
                                                      + d_hat->read(F(2)))
                                                         / 3.0);
                auto box = expanded_box(to_float3(load_vec3(positions->read(F(0)))), ext);
                box = box_merged(
                    box, expanded_box(to_float3(load_vec3(positions->read(F(1)))), ext));
                box = box_merged(
                    box, expanded_box(to_float3(load_vec3(positions->read(F(2)))), ext));
                lcrt::Var<LAABB> packed;
                store_aabb(packed, box);
                tri_aabbs->write(i, packed);
            };
        });
    return k;
}

// ============================================================================
//  the three gather kernels (bid/cid per primitive)
// ============================================================================
auto& gather_codim_bids_cids_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>, std::uint32_t>(
        [](lcrt::BufferVar<IndexT> codim_indices, lcrt::BufferVar<IndexT> vert_bids,
           lcrt::BufferVar<IndexT> vert_cids, lcrt::BufferVar<IndexT> codim_bids,
           lcrt::BufferVar<IndexT> codim_cids, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto p = codim_indices->read(i);
                codim_bids->write(i, vert_bids->read(p));
                codim_cids->write(i, vert_cids->read(p));
            };
        });
    return k;
}

auto& gather_edge_bids_cids_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>, std::uint32_t>(
        [](lcrt::BufferVar<LVector2i> edges, lcrt::BufferVar<IndexT> vert_bids,
           lcrt::BufferVar<IndexT> vert_cids, lcrt::BufferVar<IndexT> edge_bids,
           lcrt::BufferVar<IndexT> edge_cids, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto E = load_vec2i(edges->read(i));
                edge_bids->write(i, vert_bids->read(E(0)));
                edge_cids->write(i, vert_cids->read(E(0)));
            };
        });
    return k;
}

auto& gather_tri_bids_cids_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3i>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>,
                                               lcrt::Buffer<IndexT>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3i> triangles, lcrt::BufferVar<IndexT> vert_bids,
           lcrt::BufferVar<IndexT> vert_cids, lcrt::BufferVar<IndexT> tri_bids,
           lcrt::BufferVar<IndexT> tri_cids, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto F = load_vec3i(triangles->read(i));
                tri_bids->write(i, vert_bids->read(F(0)));
                tri_cids->write(i, vert_cids->read(F(0)));
            };
        });
    return k;
}
}// namespace

class SimplicialSurfaceDistanceCheck final : public BackendSanityChecker
{
  public:
    constexpr static U64 SanityCheckerUID = 3;
    using BackendSanityChecker::BackendSanityChecker;

    enum ViolationType : IndexT
    {
        PP = 0,
        PE = 1,
        PT = 2,
        EE = 3
    };

    struct ViolationInfo
    {
        IndexT   type              = 0;
        Vector2i primitive_indices = {-1, -1};
        Vector2i geo_ids           = {-1, -1};
        Vector2i obj_ids           = {-1, -1};
        Vector2  distance          = {0, 0};
    };

    struct Vector2iLess
    {
        bool operator()(const Vector2i& lhs, const Vector2i& rhs) const
        {
            return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
        }
    };

    struct ViolationResult
    {
        bool                                  is_too_close = false;
        vector<ViolationInfo>                 violations;
        map<Vector2i, Vector2i, Vector2iLess> close_geo_ids;
        map<Vector2i, Vector2, Vector2iLess>  close_geo_distances;
        IndexT                                total_violations = 0;
    };

    struct Impl
    {
        ViolationResult check(span<const Vector3>  Vs,
                              span<const Vector2i> Es,
                              span<const Vector3i> Fs,
                              span<const IndexT>   CodimIndices,
                              span<const Float>    h_thickness_span,
                              span<const Float>    h_d_hat_span,
                              span<const IndexT>   h_vert_bids,
                              span<const IndexT>   h_vert_cids,
                              span<const IndexT>   h_vert_scids,
                              span<const IndexT>   h_vert_self_collision,
                              span<const IndexT>   h_vert_geo_ids,
                              span<const IndexT>   h_vert_object_ids,
                              span<const IndexT>   h_contact_mask,
                              SizeT                contact_element_count,
                              span<const IndexT>   h_subscene_mask,
                              SizeT                subscene_element_count)
        {
            SizeT num_verts = Vs.size();
            SizeT num_edges = Es.size();
            SizeT num_tris  = Fs.size();
            SizeT num_codim = CodimIndices.size();

            auto& stream = frame_stream();

            // ---- the device inputs ------------------------------------------------
            // NOTE: the upload commands reference the host memory, so every host span
            // must outlive the synchronisation that commits them (the read-backs below).
            auto h_positions  = to_device_positions(Vs);
            auto h_edges      = to_device_edges(Es);
            auto h_tris       = to_device_triangles(Fs);
            auto h_codim_idx  = std::vector<IndexT>(CodimIndices.begin(), CodimIndices.end());
            auto h_thickness  = std::vector<Float>(h_thickness_span.begin(),
                                                   h_thickness_span.end());
            auto h_d_hat      = std::vector<Float>(h_d_hat_span.begin(),
                                                   h_d_hat_span.end());
            auto h_bid_vec    = std::vector<IndexT>(h_vert_bids.begin(), h_vert_bids.end());
            auto h_cid_vec    = std::vector<IndexT>(h_vert_cids.begin(), h_vert_cids.end());
            auto h_scid_vec   = std::vector<IndexT>(h_vert_scids.begin(),
                                                    h_vert_scids.end());
            auto h_self_vec   = std::vector<IndexT>(h_vert_self_collision.begin(),
                                                    h_vert_self_collision.end());

            lud::LDeviceBuffer<LVector3> positions;
            positions.copy_from(span<const LVector3>{h_positions});
            lud::LDeviceBuffer<LVector2i> edges;
            edges.resize(padded(num_edges));
            if(num_edges > 0)
                edges.view().copy_from(span<const LVector2i>{h_edges});
            lud::LDeviceBuffer<LVector3i> triangles;
            triangles.resize(padded(num_tris));
            if(num_tris > 0)
                triangles.view().copy_from(span<const LVector3i>{h_tris});
            lud::LDeviceBuffer<IndexT> codim_indices;
            codim_indices.resize(padded(num_codim));
            if(num_codim > 0)
                codim_indices.view().copy_from(span<const IndexT>{h_codim_idx});
            lud::LDeviceBuffer<Float> thickness;
            thickness.copy_from(span<const Float>{h_thickness});
            lud::LDeviceBuffer<Float> d_hat;
            d_hat.copy_from(span<const Float>{h_d_hat});

            lud::LDeviceBuffer<IndexT> vert_bids;
            vert_bids.resize(num_verts);
            vert_bids.view().copy_from(span<const IndexT>{h_bid_vec});
            lud::LDeviceBuffer<IndexT> vert_cids;
            vert_cids.resize(num_verts);
            vert_cids.view().copy_from(span<const IndexT>{h_cid_vec});
            lud::LDeviceBuffer<IndexT> vert_scids;
            vert_scids.resize(num_verts);
            vert_scids.view().copy_from(span<const IndexT>{h_scid_vec});
            lud::LDeviceBuffer<IndexT> self_collision;
            self_collision.resize(num_verts);
            self_collision.view().copy_from(span<const IndexT>{h_self_vec});

            // ---- the AABB stages --------------------------------------------------
            lud::LDeviceBuffer<LAABB> point_aabbs;
            point_aabbs.resize(num_verts);
            {
                const auto n = static_cast<std::uint32_t>(num_verts);
                if(n > 0u)
                    stream.record(build_point_aabbs_kernel()(positions.view(),
                                                             thickness.view(),
                                                             d_hat.view(),
                                                             point_aabbs.view(), n)
                                      .dispatch(round_up(n, 256u)));
            }

            lud::LDeviceBuffer<LAABB> codim_point_aabbs;
            codim_point_aabbs.resize(padded(num_codim));
            if(num_codim > 0)
            {
                const auto n = static_cast<std::uint32_t>(num_codim);
                stream.record(build_codim_point_aabbs_kernel()(positions.view(),
                                                               thickness.view(),
                                                               d_hat.view(),
                                                               codim_indices.view(),
                                                               codim_point_aabbs.view(), n)
                                  .dispatch(round_up(n, 256u)));
            }

            lud::LDeviceBuffer<LAABB> edge_aabbs;
            edge_aabbs.resize(padded(num_edges));
            if(num_edges > 0)
            {
                const auto n = static_cast<std::uint32_t>(num_edges);
                stream.record(build_edge_aabbs_kernel()(positions.view(), edges.view(),
                                                        thickness.view(), d_hat.view(),
                                                        edge_aabbs.view(), n)
                                  .dispatch(round_up(n, 256u)));
            }

            lud::LDeviceBuffer<LAABB> tri_aabbs;
            tri_aabbs.resize(padded(num_tris));
            if(num_tris > 0)
            {
                const auto n = static_cast<std::uint32_t>(num_tris);
                stream.record(build_tri_aabbs_kernel()(positions.view(),
                                                       triangles.view(), thickness.view(),
                                                       d_hat.view(), tri_aabbs.view(), n)
                                  .dispatch(round_up(n, 256u)));
            }

            auto& point_bids = vert_bids;
            auto& point_cids = vert_cids;

            lud::LDeviceBuffer<IndexT> codim_bids;
            lud::LDeviceBuffer<IndexT> codim_cids;
            codim_bids.resize(padded(num_codim));
            codim_cids.resize(padded(num_codim));
            if(num_codim > 0)
            {
                const auto n = static_cast<std::uint32_t>(num_codim);
                stream.record(gather_codim_bids_cids_kernel()(codim_indices.view(),
                                                              vert_bids.view(),
                                                              vert_cids.view(),
                                                              codim_bids.view(),
                                                              codim_cids.view(), n)
                                  .dispatch(round_up(n, 256u)));
            }

            lud::LDeviceBuffer<IndexT> edge_bids;
            lud::LDeviceBuffer<IndexT> edge_cids;
            edge_bids.resize(padded(num_edges));
            edge_cids.resize(padded(num_edges));
            if(num_edges > 0)
            {
                const auto n = static_cast<std::uint32_t>(num_edges);
                stream.record(gather_edge_bids_cids_kernel()(edges.view(),
                                                             vert_bids.view(),
                                                             vert_cids.view(),
                                                             edge_bids.view(),
                                                             edge_cids.view(), n)
                                  .dispatch(round_up(n, 256u)));
            }

            lud::LDeviceBuffer<IndexT> tri_bids;
            lud::LDeviceBuffer<IndexT> tri_cids;
            tri_bids.resize(padded(num_tris));
            tri_cids.resize(padded(num_tris));
            if(num_tris > 0)
            {
                const auto n = static_cast<std::uint32_t>(num_tris);
                stream.record(gather_tri_bids_cids_kernel()(triangles.view(),
                                                            vert_bids.view(),
                                                            vert_cids.view(),
                                                            tri_bids.view(),
                                                            tri_cids.view(), n)
                                  .dispatch(round_up(n, 256u)));
            }

            // ---------------------------------------------------------------------
            //  the reference's four phases, each: query the structure, then evaluate
            //  its LeafPred on the host (see the file note)
            // ---------------------------------------------------------------------
            const IndexT CN = static_cast<IndexT>(contact_element_count);
            const IndexT SN = static_cast<IndexT>(subscene_element_count);

            vector<ViolationInfo> pp_violations, pe_violations, pt_violations,
                ee_violations;

            const auto copy_pairs = [&](InfoStacklessBVH::QueryBuffer& qbuf)
            {
                vector<LVector2i> tmp(qbuf.size());
                if(!tmp.empty())
                    qbuf.view().copy_to(tmp);
                vector<Vector2i> out(tmp.size());
                for(size_t i = 0; i < tmp.size(); ++i)
                    out[i] = to_eigen(tmp[i]);
                return out;
            };

            // ---- Phase 1: CodimP vs AllP -----------------------------------------
            if(num_codim > 0 && num_verts > 0)
            {
                InfoStacklessBVH              point_bvh;
                InfoStacklessBVH::QueryBuffer pp_qbuf;
                point_bvh.build(point_aabbs.cview(), point_bids.cview(),
                                point_cids.cview());
                point_bvh.query(codim_point_aabbs.cview(), codim_bids.cview(),
                                codim_cids.cview(), lud::LCBufferView<IndexT>{},
                                KeepAllNodes{}, KeepAllLeaves{}, pp_qbuf);

                for(const auto& p : copy_pairs(pp_qbuf))
                {
                    IndexT   codim_idx = p[0];
                    IndexT   P         = p[1];
                    IndexT   CodimP    = CodimIndices[codim_idx];

                    if(CodimP == P)
                        continue;

                    Vector2i scids = {h_vert_scids[CodimP], h_vert_scids[P]};
                    if(!allow_PP_contact(h_subscene_mask, SN, scids))
                        continue;

                    Vector2i cids = {h_vert_cids[CodimP], h_vert_cids[P]};
                    if(!allow_PP_contact(h_contact_mask, CN, cids))
                        continue;

                    if(h_vert_bids[CodimP] == h_vert_bids[P] && h_vert_bids[CodimP] != -1
                       && !h_vert_self_collision[CodimP])
                        continue;

                    Float D;
                    dist::point_point_distance2(to_luisa(Vs[CodimP]), to_luisa(Vs[P]), D);

                    Float thickness  = h_thickness_span[CodimP] + h_thickness_span[P];
                    Float thickness2 = thickness * thickness;

                    if(D <= thickness2)
                    {
                        pp_violations.push_back(
                            {PP,
                             p,
                             {h_vert_geo_ids[CodimP], h_vert_geo_ids[P]},
                             {h_vert_object_ids[CodimP], h_vert_object_ids[P]},
                             {D, thickness2}});
                    }
                }
            }

            // ---- Phase 2: CodimP vs AllE -----------------------------------------
            if(num_codim > 0 && num_edges > 0)
            {
                InfoStacklessBVH              edge_bvh;
                InfoStacklessBVH::QueryBuffer pe_qbuf;
                edge_bvh.build(edge_aabbs.cview(), edge_bids.cview(), edge_cids.cview());
                edge_bvh.query(codim_point_aabbs.cview(), codim_bids.cview(),
                               codim_cids.cview(), lud::LCBufferView<IndexT>{},
                               KeepAllNodes{}, KeepAllLeaves{}, pe_qbuf);

                for(const auto& p : copy_pairs(pe_qbuf))
                {
                    IndexT   codim_idx = p[0];
                    IndexT   edge_idx  = p[1];
                    IndexT   CodimP    = CodimIndices[codim_idx];
                    Vector2i E         = Es[edge_idx];

                    if(CodimP == E[0] || CodimP == E[1])
                        continue;

                    Vector3i scids = {h_vert_scids[CodimP], h_vert_scids[E[0]],
                                      h_vert_scids[E[1]]};
                    if(!allow_PE_contact(h_subscene_mask, SN, scids))
                        continue;

                    Vector3i cids = {h_vert_cids[CodimP], h_vert_cids[E[0]],
                                     h_vert_cids[E[1]]};
                    if(!allow_PE_contact(h_contact_mask, CN, cids))
                        continue;

                    if(h_vert_bids[CodimP] == h_vert_bids[E[0]] && h_vert_bids[CodimP] != -1
                       && !h_vert_self_collision[CodimP])
                        continue;

                    auto  pe_flag = dist::point_edge_distance_flag(
                        to_luisa(Vs[CodimP]), to_luisa(Vs[E[0]]), to_luisa(Vs[E[1]]));
                    Float D;
                    dist::point_edge_distance2(pe_flag, to_luisa(Vs[CodimP]),
                                               to_luisa(Vs[E[0]]), to_luisa(Vs[E[1]]), D);

                    Float thickness  = h_thickness_span[CodimP] + h_thickness_span[E[0]];
                    Float thickness2 = thickness * thickness;

                    if(D <= thickness2)
                    {
                        pe_violations.push_back(
                            {PE,
                             p,
                             {h_vert_geo_ids[CodimP], h_vert_geo_ids[E[0]]},
                             {h_vert_object_ids[CodimP], h_vert_object_ids[E[0]]},
                             {D, thickness2}});
                    }
                }
            }

            // ---- Phase 3: AllP vs AllT -------------------------------------------
            if(num_verts > 0 && num_tris > 0)
            {
                InfoStacklessBVH              tri_bvh;
                InfoStacklessBVH::QueryBuffer pt_qbuf;
                tri_bvh.build(tri_aabbs.cview(), tri_bids.cview(), tri_cids.cview());
                tri_bvh.query(point_aabbs.cview(), point_bids.cview(), point_cids.cview(),
                              lud::LCBufferView<IndexT>{}, KeepAllNodes{},
                              KeepAllLeaves{}, pt_qbuf);

                for(const auto& p : copy_pairs(pt_qbuf))
                {
                    IndexT   P       = p[0];
                    IndexT   tri_idx = p[1];
                    Vector3i T       = Fs[tri_idx];

                    if(P == T[0] || P == T[1] || P == T[2])
                        continue;

                    Vector4i scids = {h_vert_scids[P], h_vert_scids[T[0]],
                                      h_vert_scids[T[1]], h_vert_scids[T[2]]};
                    if(!allow_PT_contact(h_subscene_mask, SN, scids))
                        continue;

                    Vector4i cids = {h_vert_cids[P], h_vert_cids[T[0]], h_vert_cids[T[1]],
                                     h_vert_cids[T[2]]};
                    if(!allow_PT_contact(h_contact_mask, CN, cids))
                        continue;

                    if(h_vert_bids[P] == h_vert_bids[T[0]] && h_vert_bids[P] != -1
                       && !h_vert_self_collision[P])
                        continue;

                    auto pt_flag = dist::point_triangle_distance_flag(
                        to_luisa(Vs[P]), to_luisa(Vs[T[0]]), to_luisa(Vs[T[1]]),
                        to_luisa(Vs[T[2]]));
                    Float D;
                    dist::point_triangle_distance2(pt_flag, to_luisa(Vs[P]),
                                                   to_luisa(Vs[T[0]]),
                                                   to_luisa(Vs[T[1]]),
                                                   to_luisa(Vs[T[2]]), D);

                    Float thickness  = h_thickness_span[P] + h_thickness_span[T[0]];
                    Float thickness2 = thickness * thickness;

                    if(D <= thickness2)
                    {
                        pt_violations.push_back(
                            {PT,
                             p,
                             {h_vert_geo_ids[P], h_vert_geo_ids[T[0]]},
                             {h_vert_object_ids[P], h_vert_object_ids[T[0]]},
                             {D, thickness2}});
                    }
                }
            }

            // ---- Phase 4: AllE vs AllE -------------------------------------------
            if(num_edges > 1)
            {
                InfoStacklessBVH              edge_bvh;
                InfoStacklessBVH::QueryBuffer ee_qbuf;
                edge_bvh.build(edge_aabbs.cview(), edge_bids.cview(), edge_cids.cview());
                edge_bvh.detect(lud::LCBufferView<IndexT>{}, KeepAllNodes{},
                                KeepAllLeaves{}, ee_qbuf);

                for(const auto& p : copy_pairs(ee_qbuf))
                {
                    IndexT   ei = p[0];
                    IndexT   ej = p[1];
                    Vector2i E0 = Es[ei];
                    Vector2i E1 = Es[ej];

                    if(E0[0] == E1[0] || E0[0] == E1[1] || E0[1] == E1[0]
                       || E0[1] == E1[1])
                        continue;

                    Vector4i scids = {h_vert_scids[E0[0]], h_vert_scids[E0[1]],
                                      h_vert_scids[E1[0]], h_vert_scids[E1[1]]};
                    if(!allow_EE_contact(h_subscene_mask, SN, scids))
                        continue;

                    Vector4i cids = {h_vert_cids[E0[0]], h_vert_cids[E0[1]],
                                     h_vert_cids[E1[0]], h_vert_cids[E1[1]]};
                    if(!allow_EE_contact(h_contact_mask, CN, cids))
                        continue;

                    if(h_vert_bids[E0[0]] == h_vert_bids[E1[0]] && h_vert_bids[E0[0]] != -1
                       && !h_vert_self_collision[E0[0]])
                        continue;

                    auto ee_flag = dist::edge_edge_distance_flag(
                        to_luisa(Vs[E0[0]]), to_luisa(Vs[E0[1]]), to_luisa(Vs[E1[0]]),
                        to_luisa(Vs[E1[1]]));
                    Float D;
                    dist::edge_edge_distance2(ee_flag, to_luisa(Vs[E0[0]]),
                                              to_luisa(Vs[E0[1]]),
                                              to_luisa(Vs[E1[0]]),
                                              to_luisa(Vs[E1[1]]), D);

                    Float thickness  = h_thickness_span[E0[0]] + h_thickness_span[E1[0]];
                    Float thickness2 = thickness * thickness;

                    if(D <= thickness2)
                    {
                        ee_violations.push_back(
                            {EE,
                             p,
                             {h_vert_geo_ids[E0[0]], h_vert_geo_ids[E1[0]]},
                             {h_vert_object_ids[E0[0]], h_vert_object_ids[E1[0]]},
                             {D, thickness2}});
                    }
                }
            }

            // ---- the reference's host bookkeeping (same order: pp, pe, pt, ee) ----
            ViolationResult result;
            result.total_violations = static_cast<IndexT>(pp_violations.size()
                                                          + pe_violations.size()
                                                          + pt_violations.size()
                                                          + ee_violations.size());
            result.is_too_close = result.total_violations > 0;
            result.violations.reserve(result.total_violations);

            for(auto& info : pp_violations)
                result.violations.push_back(info);
            for(auto& info : pe_violations)
                result.violations.push_back(info);
            for(auto& info : pt_violations)
                result.violations.push_back(info);
            for(auto& info : ee_violations)
                result.violations.push_back(info);

            auto set_geo_distance = [&](const Vector2i& geo_ids, const Vector2& distance)
            {
                if(auto it = result.close_geo_distances.find(geo_ids);
                   it == result.close_geo_distances.end())
                {
                    result.close_geo_distances[geo_ids] = distance;
                }
                else if(distance[0] < it->second[0])
                {
                    it->second = distance;
                }
            };

            for(auto info : result.violations)
            {
                if(info.geo_ids[0] > info.geo_ids[1])
                {
                    std::swap(info.geo_ids[0], info.geo_ids[1]);
                    std::swap(info.obj_ids[0], info.obj_ids[1]);
                }

                result.close_geo_ids[info.geo_ids] = info.obj_ids;
                set_geo_distance(info.geo_ids, info.distance);
            }

            return result;
        }
    };

    static geometry::SimplicialComplex extract_close_mesh(
        const geometry::SimplicialComplex& scene_surface,
        span<const IndexT>                 vertex_too_close,
        span<const IndexT>                 edge_too_close,
        span<const IndexT>                 tri_too_close)
    {
        geometry::SimplicialComplex mesh;

        vector<SizeT> close_verts;
        vector<SizeT> close_edges;
        vector<SizeT> close_tris;

        for(SizeT i = 0; i < vertex_too_close.size(); i++)
            if(vertex_too_close[i])
                close_verts.push_back(i);
        for(SizeT i = 0; i < edge_too_close.size(); i++)
            if(edge_too_close[i])
                close_edges.push_back(i);
        for(SizeT i = 0; i < tri_too_close.size(); i++)
            if(tri_too_close[i])
                close_tris.push_back(i);

        mesh.vertices().resize(close_verts.size());
        mesh.vertices().copy_from(scene_surface.vertices(),
                                  geometry::AttributeCopy::pull(close_verts));
        mesh.edges().resize(close_edges.size());
        mesh.edges().copy_from(scene_surface.edges(),
                               geometry::AttributeCopy::pull(close_edges));
        mesh.triangles().resize(close_tris.size());
        mesh.triangles().copy_from(scene_surface.triangles(),
                                   geometry::AttributeCopy::pull(close_tris));

        vector<IndexT> vertex_remap(scene_surface.vertices().size(), -1);
        for(auto [i, v] : enumerate(close_verts))
            vertex_remap[v] = i;

        auto Map = [&]<IndexT N>(const Eigen::Vector<IndexT, N>& V) -> Eigen::Vector<IndexT, N>
        {
            auto ret = V;
            for(auto& v : ret)
                v = vertex_remap[v];
            return ret;
        };

        auto edge_topo_view = view(mesh.edges().topo());
        std::ranges::transform(edge_topo_view, edge_topo_view.begin(), Map);

        auto tri_topo_view = view(mesh.triangles().topo());
        std::ranges::transform(tri_topo_view, tri_topo_view.begin(), Map);

        return mesh;
    }

  protected:
    virtual void build() override
    {
        auto enable_contact = context().config().find<IndexT>("contact/enable");
        if(!enable_contact->view()[0])
        {
            throw BackendSanityCheckerException("Contact is not enabled");
        }
    }

    virtual U64 get_id() const noexcept override { return SanityCheckerUID; }

    virtual SanityCheckResult do_check(core::SanityCheckMessage& msg) override
    {
        auto& ctx = context();

        const geometry::SimplicialComplex& scene_surface = ctx.scene_simplicial_surface();

        auto Vs = scene_surface.vertices().size() ? scene_surface.positions().view() :
                                                    span<const Vector3>{};
        auto Es = scene_surface.edges().size() ? scene_surface.edges().topo().view() :
                                                 span<const Vector2i>{};
        auto Fs = scene_surface.triangles().size() ?
                      scene_surface.triangles().topo().view() :
                      span<const Vector3i>{};

        if(Vs.size() == 0)
            return SanityCheckResult::Success;

        SizeT num_verts = Vs.size();

        vector<IndexT> h_codim_indices;
        {
            auto attr_dim = scene_surface.vertices().find<IndexT>("sanity_check/dim");
            UIPC_ASSERT(attr_dim,
                        "`sanity_check/dim` is not found in scene surface, why can it happen?");
            auto dim = attr_dim->view();
            h_codim_indices.reserve(num_verts);
            for(SizeT i = 0; i < num_verts; ++i)
            {
                if(dim[i] <= 1)
                    h_codim_indices.push_back(static_cast<IndexT>(i));
            }
        }

        auto attr_v_thickness = scene_surface.vertices().find<Float>(builtin::thickness);
        auto VThickness =
            attr_v_thickness ? attr_v_thickness->view() : span<const Float>{};

        auto attr_v_d_hat = scene_surface.vertices().find<Float>("sanity_check/d_hat");
        UIPC_ASSERT(attr_v_d_hat, "`sanity_check/d_hat` is not found in scene surface");
        auto Vd_hats = attr_v_d_hat->view();

        vector<Float> h_thickness(num_verts, 0.0);
        vector<Float> h_d_hat(num_verts, 0.0);
        for(SizeT i = 0; i < num_verts; ++i)
        {
            h_thickness[i] = VThickness.empty() ? 0.0 : VThickness[i];
            h_d_hat[i]     = Vd_hats[i];
        }

        auto attr_v_instance_id =
            scene_surface.vertices().find<IndexT>("sanity_check/instance_id");
        UIPC_ASSERT(attr_v_instance_id,
                    "`sanity_check/instance_id` is not found in scene surface");
        auto VInstanceIds = attr_v_instance_id->view();

        auto attr_v_geo_ids =
            scene_surface.vertices().find<IndexT>("sanity_check/geometry_id");
        UIPC_ASSERT(attr_v_geo_ids,
                    "`sanity_check/geometry_id` is not found in scene surface");
        auto VGeoIds = attr_v_geo_ids->view();

        auto attr_v_object_id =
            scene_surface.vertices().find<IndexT>("sanity_check/object_id");
        UIPC_ASSERT(attr_v_object_id,
                    "`sanity_check/object_id` is not found in scene surface");
        auto VObjectIds = attr_v_object_id->view();

        auto attr_cids =
            scene_surface.vertices().find<IndexT>("sanity_check/contact_element_id");
        UIPC_ASSERT(attr_cids,
                    "`sanity_check/contact_element_id` is not found in scene surface");
        auto CIds = attr_cids->view();

        auto attr_scids = scene_surface.vertices().find<IndexT>(
            "sanity_check/subscene_contact_element_id");
        UIPC_ASSERT(attr_scids,
                    "`sanity_check/subscene_contact_element_id` is not found in scene surface");
        auto SCIds = attr_scids->view();

        auto attr_self_collision =
            scene_surface.vertices().find<IndexT>("sanity_check/self_collision");
        UIPC_ASSERT(attr_self_collision,
                    "`sanity_check/self_collision` is not found in scene surface");
        auto SelfCollision = attr_self_collision->view();

        auto& contact_tabular = ctx.contact_tabular();
        SizeT CN              = contact_tabular.element_count();

        vector<IndexT> h_contact_mask(CN * CN);
        for(IndexT i = 0; i < (IndexT)CN; ++i)
            for(IndexT j = 0; j < (IndexT)CN; ++j)
                h_contact_mask[i * CN + j] =
                    contact_tabular.at(i, j).is_enabled() ? 1 : 0;

        auto& subscene_tabular = ctx.subscene_tabular();
        SizeT SN               = subscene_tabular.element_count();

        vector<IndexT> h_subscene_mask(SN * SN);
        for(IndexT i = 0; i < (IndexT)SN; ++i)
            for(IndexT j = 0; j < (IndexT)SN; ++j)
                h_subscene_mask[i * SN + j] =
                    subscene_tabular.at(i, j).is_enabled() ? 1 : 0;

        auto result =
            m_impl.check(Vs,
                         Es,
                         Fs,
                         span<const IndexT>{h_codim_indices},
                         span<const Float>{h_thickness},
                         span<const Float>{h_d_hat},
                         span<const IndexT>{VInstanceIds.data(), VInstanceIds.size()},
                         span<const IndexT>{CIds.data(), CIds.size()},
                         span<const IndexT>{SCIds.data(), SCIds.size()},
                         span<const IndexT>{SelfCollision.data(), SelfCollision.size()},
                         span<const IndexT>{VGeoIds.data(), VGeoIds.size()},
                         span<const IndexT>{VObjectIds.data(), VObjectIds.size()},
                         span<const IndexT>{h_contact_mask},
                         CN,
                         span<const IndexT>{h_subscene_mask},
                         SN);

        if(!result.is_too_close)
            return SanityCheckResult::Success;

        SizeT num_edges = Es.size();
        SizeT num_tris  = Fs.size();

        vector<IndexT> vertex_too_close(num_verts, 0);
        vector<IndexT> edge_too_close(num_edges, 0);
        vector<IndexT> tri_too_close(num_tris, 0);

        auto mark_vert = [&](IndexT v) { vertex_too_close[v] = 1; };

        for(auto& info : result.violations)
        {
            switch(info.type)
            {
                case PP: {
                    IndexT CodimP = h_codim_indices[info.primitive_indices[0]];
                    mark_vert(CodimP);
                    mark_vert(info.primitive_indices[1]);
                    break;
                }
                case PE: {
                    IndexT   CodimP = h_codim_indices[info.primitive_indices[0]];
                    mark_vert(CodimP);
                    Vector2i E = Es[info.primitive_indices[1]];
                    mark_vert(E[0]);
                    mark_vert(E[1]);
                    edge_too_close[info.primitive_indices[1]] = 1;
                    break;
                }
                case PT: {
                    mark_vert(info.primitive_indices[0]);
                    Vector3i F = Fs[info.primitive_indices[1]];
                    mark_vert(F[0]);
                    mark_vert(F[1]);
                    mark_vert(F[2]);
                    tri_too_close[info.primitive_indices[1]] = 1;
                    break;
                }
                case EE: {
                    Vector2i E0 = Es[info.primitive_indices[0]];
                    Vector2i E1 = Es[info.primitive_indices[1]];
                    mark_vert(E0[0]);
                    mark_vert(E0[1]);
                    mark_vert(E1[0]);
                    mark_vert(E1[1]);
                    edge_too_close[info.primitive_indices[0]] = 1;
                    edge_too_close[info.primitive_indices[1]] = 1;
                    break;
                }
                default:
                    UIPC_ASSERT(false, "Unknown violation type: {}", info.type);
            }
        }

        ::uipc::backend::SanityCheckMessageVisitor scmv{msg};
        auto& buffer = scmv.message();

        for(auto& [GeoIds, ObjIds] : result.close_geo_ids)
        {
            auto obj_0 = find_object(ObjIds[0]);
            auto obj_1 = find_object(ObjIds[1]);

            UIPC_ASSERT(obj_0 != nullptr, "Object[{}] not found", ObjIds[0]);
            UIPC_ASSERT(obj_1 != nullptr, "Object[{}] not found", ObjIds[1]);

            std::string name_0{obj_0->name()};
            std::string name_1{obj_1->name()};

            fmt::format_to(std::back_inserter(buffer),
                           "Geometry({}) in Object[{}({})] is too close (distance={}, thickness={}) to Geometry({}) in "
                           "Object[{}({})]\n",
                           GeoIds[0],
                           name_0,
                           obj_0->id(),
                           std::sqrt(result.close_geo_distances[GeoIds][0]),
                           std::sqrt(result.close_geo_distances[GeoIds][1]),
                           GeoIds[1],
                           name_1,
                           obj_1->id());
        }

        auto close_mesh =
            extract_close_mesh(scene_surface, vertex_too_close, edge_too_close,
                               tri_too_close);

        fmt::format_to(std::back_inserter(buffer),
                       "Close mesh has {} vertices, {} edges, and {} triangles.\n",
                       close_mesh.vertices().size(),
                       close_mesh.edges().size(),
                       close_mesh.triangles().size());

        std::string name = "close_mesh";

        auto sanity_check_mode = ctx.config().find<std::string>("sanity_check/mode");
        if(sanity_check_mode->view()[0] == "normal")
        {
            auto output_path = this_output_path();
            namespace fs     = std::filesystem;
            fs::path path{output_path};
            path /= fmt::format("{}.obj", name);
            auto path_str = path.string();

            geometry::SimplicialComplexIO io;
            io.write(path_str, close_mesh);
            fmt::format_to(std::back_inserter(buffer),
                           "Close mesh is saved at {}.\n",
                           path_str);
        }

        fmt::format_to(std::back_inserter(buffer),
                       "Create mesh [{}<{}>] for post-processing.",
                       name,
                       close_mesh.type());

        scmv.geometries()[name] =
            uipc::make_shared<geometry::SimplicialComplex>(std::move(close_mesh));

        return SanityCheckResult::Error;
    }

  private:
    Impl m_impl;
};

REGISTER_BACKEND_SANITY_CHECKER(SimplicialSurfaceDistanceCheck);
}// namespace uipc::backend::luisa
