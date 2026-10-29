/********************************************************************
 * @file simplicial_surface_intersection_check.cpp
 * @brief T7.1b — edge/triangle intersection sanity check (port of
 *        src/backends/cuda/sanity_check/simplicial_surface_intersection_check.cu, 603
 *        lines).  Sanity-checker uid 1.
 *
 * PORT NOTES (the two checkers share them; see also
 * simplicial_surface_distance_check.cpp)
 * ---------------------------------------------------------------------------
 *  * The reference's `SimplicialSurfaceIntersectionCheck_build_{tri,edge}_aabbs_kernel`
 *    become two `linalg::detail::cached_kernel` shaders (compile once, dispatch per
 *    call).  `AABB` -> `LAABB` (T7.1a's device element type); `Vector3` -> `LVector3`
 *    on the device side, `to_luisa()` at the host/device boundary.
 *  * The structure is T7.1b's `InfoStacklessBVH` (the reference's choice: uid 1 builds
 *    the *triangle* BVH and queries it with the edges, i.e. the "Other" flavour with
 *    the edge set as queries).
 *  * **The reference's device predicates (which capture `cuda_tool::CBufferView`s and
 *    a `CDense2D` mask) cannot be ported as LC predicates** — an LC kernel cannot carry
 *    a callable, and a captured view becomes a *bound* buffer of the compiled shader
 *    (cookbook §7.9 rule 2), so a cached shader would keep reading the first call's
 *    allocation.  The port therefore drives the query with the trivial predicates
 *    (`KeepAllNodes` / `KeepAllLeaves`) and evaluates the reference's
 *    `NodePred`/`LeafPred` **body** as a host filter over the reported candidate pairs.
 *    The ported filter is the same arithmetic on the same data: the masks, the
 *    topologies, the cids/scids/instance-ids and the self-collision flags are all host
 *    spans here, and the surviving pair set is provably identical because the reference's
 *    `NodePred` cull (`cmts(query_cid, node_cid) == 0` for a node whose whole subtree
 *    shares one cid) is implied by the `LeafPred`'s per-vertex `cmts(cid_e, cid_f)`
 *    checks.  Only the number of *candidates* differs, never the verdict.
 *  * the reference's `tri_edge_intersect_device` (`UIPC_GENERIC`, Eigen arithmetic) is
 *    transcribed verbatim as a host function — the port's filter runs on the host, so
 *    the reference's own host instantiation is what executes.
 *  * the reported message, the `intersected_mesh` (with the reference's
 *    `builtin::is_facet` fill) and the returned `SanityCheckResult` are transcribed
 *    from the reference's `do_check`.
 *********************************************************************/
#include <type_define.h>
#include <uipc/core/object.h>
#include <uipc/core/contact_tabular.h>
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

#include <algorithm>
#include <utility>
#include <vector>

namespace uipc::backend::luisa
{
/// The reference's `tri_edge_intersect_device` (a `UIPC_GENERIC` function; the port's
/// filter is a host filter, so this is its host instantiation).
inline bool tri_edge_intersect(const Vector3& t0,
                               const Vector3& t1,
                               const Vector3& t2,
                               const Vector3& e0,
                               const Vector3& e1)
{
    constexpr Float eps = 1e-12;

    Vector3 dir = e1 - e0;
    Float   len = dir.norm();
    if(len < eps)
        return false;
    dir /= len;

    Vector3 edge1 = t1 - t0;
    Vector3 edge2 = t2 - t0;
    Vector3 h     = dir.cross(edge2);
    Float   a     = edge1.dot(h);

    if(a > -eps && a < eps)
        return false;

    Float   f = 1.0 / a;
    Vector3 s = e0 - t0;
    Float   u = f * s.dot(h);
    if(u < 0.0 || u > 1.0)
        return false;

    Vector3 q = s.cross(edge1);
    Float   v = f * dir.dot(q);
    if(v < 0.0 || u + v > 1.0)
        return false;

    Float t = f * edge2.dot(q);
    if(t < 0.0 || t > len)
        return false;

    return true;
}

namespace
{
namespace lla = luisa_device::linalg;

/// the port's padded-buffer helpers (LC rejects a zero-length buffer view as a kernel
/// argument — T7.1a rule 9)
[[nodiscard]] constexpr std::uint32_t round_up(std::uint64_t n,
                                               std::uint32_t block) noexcept
{
    return static_cast<std::uint32_t>((n + block - 1u) / block * block);
}
[[nodiscard]] constexpr size_t padded(size_t n) noexcept
{
    return n == 0u ? 1u : n;
}

/// The reference's `std::less<uipc::Vector2i>` specialisation (see the ported half-plane
/// checker: specialising `std::less` for an Eigen type is not permitted, and the
/// lexicographic *order* is what fixes the reported message order).
struct Vec2iLess
{
    bool operator()(const Vector2i& lhs, const Vector2i& rhs) const noexcept
    {
        return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
    }
};

/// The port's stand-ins for the reference's device predicates (see the file note):
/// record-time host callables over the templated info structs that keep every node and
/// every pair.  The reference's actual filters run on the host below.
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

/// the host/device conversion of a position span (identical bytes)
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

/// The reference's `positions(F[0]).cast<float>()`: the device AABB element is float
/// while a `Vector3` is double.
[[nodiscard]] inline lps::LVec<lcrt::Var<float>, 3> to_float3(
    const lps::LVec<lcrt::Var<Float>, 3>& v) noexcept
{
    lps::LVec<lcrt::Var<float>, 3> r;
    for(int k = 0; k < 3; ++k)
        r(k) = lcrt::cast<float>(v(k));
    return r;
}

// ============================================================================
//  SimplicialSurfaceIntersectionCheck_build_tri_aabbs_kernel
// ============================================================================
auto& build_tri_aabbs_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3i>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<LAABB>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<LVector3i> triangles,
           lcrt::BufferVar<IndexT> vert_bids, lcrt::BufferVar<IndexT> vert_cids,
           lcrt::BufferVar<LAABB> tri_aabbs, lcrt::BufferVar<IndexT> tri_bids,
           lcrt::BufferVar<IndexT> tri_cids, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto F = load_vec3i(triangles->read(i));
                auto box = box_extend_point(box_empty<lcrt::Var<float>>(),
                                            to_float3(load_vec3(positions->read(F(0)))));
                box      = box_extend_point(box, to_float3(load_vec3(positions->read(F(1)))));
                box      = box_extend_point(box, to_float3(load_vec3(positions->read(F(2)))));
                lcrt::Var<LAABB> packed;
                store_aabb(packed, box);
                tri_aabbs->write(i, packed);
                tri_bids->write(i, vert_bids->read(F(0)));
                tri_cids->write(i, vert_cids->read(F(0)));
            };
        });
    return k;
}

// ============================================================================
//  SimplicialSurfaceIntersectionCheck_build_edge_aabbs_kernel
// ============================================================================
auto& build_edge_aabbs_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<LAABB>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<LVector2i> edges,
           lcrt::BufferVar<IndexT> vert_bids, lcrt::BufferVar<IndexT> vert_cids,
           lcrt::BufferVar<LAABB> edge_aabbs, lcrt::BufferVar<IndexT> edge_bids,
           lcrt::BufferVar<IndexT> edge_cids, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto E = load_vec2i(edges->read(i));
                auto box = box_extend_point(box_empty<lcrt::Var<float>>(),
                                            to_float3(load_vec3(positions->read(E(0)))));
                box      = box_extend_point(box, to_float3(load_vec3(positions->read(E(1)))));
                lcrt::Var<LAABB> packed;
                store_aabb(packed, box);
                edge_aabbs->write(i, packed);
                edge_bids->write(i, vert_bids->read(E(0)));
                edge_cids->write(i, vert_cids->read(E(0)));
            };
        });
    return k;
}
}// namespace

class SimplicialSurfaceIntersectionCheck final : public BackendSanityChecker
{
  public:
    constexpr static U64 SanityCheckerUID = 1;
    using BackendSanityChecker::BackendSanityChecker;

    struct Impl
    {
        vector<Vector2i> detect_intersections(span<const Vector3>  Vs,
                                              span<const Vector2i> Es,
                                              span<const Vector3i> Fs,
                                              span<const IndexT>   h_vert_bids,
                                              span<const IndexT>   h_vert_cids,
                                              span<const IndexT>   h_vert_scids,
                                              span<const IndexT> h_vert_self_collision,
                                              span<const IndexT> h_contact_mask,
                                              SizeT contact_element_count,
                                              span<const IndexT> h_subscene_mask,
                                              SizeT subscene_element_count)
        {
            SizeT num_verts = Vs.size();
            SizeT num_edges = Es.size();
            SizeT num_tris  = Fs.size();

            auto& stream = frame_stream();

            // ---- the device inputs (the reference's DeviceBuffer copies) ----------
            // NOTE: the upload commands reference the host memory, so every host span
            // must outlive the synchronisation that commits them (the read-back below).
            auto h_positions = to_device_positions(Vs);
            auto h_edges     = to_device_edges(Es);
            auto h_tris      = to_device_triangles(Fs);

            lud::LDeviceBuffer<LVector3> positions;
            positions.copy_from(span<const LVector3>{h_positions});
            lud::LDeviceBuffer<LVector2i> edges;
            edges.resize(num_edges);
            if(num_edges > 0)
                edges.view().copy_from(span<const LVector2i>{h_edges});
            lud::LDeviceBuffer<LVector3i> triangles;
            triangles.resize(num_tris);
            if(num_tris > 0)
                triangles.view().copy_from(span<const LVector3i>{h_tris});
            lud::LDeviceBuffer<IndexT> vert_bids;
            vert_bids.resize(num_verts);
            if(num_verts > 0)
                vert_bids.view().copy_from(h_vert_bids);
            lud::LDeviceBuffer<IndexT> vert_cids;
            vert_cids.resize(num_verts);
            if(num_verts > 0)
                vert_cids.view().copy_from(h_vert_cids);
            lud::LDeviceBuffer<IndexT> vert_scids;
            vert_scids.resize(num_verts);
            if(num_verts > 0)
                vert_scids.view().copy_from(h_vert_scids);
            lud::LDeviceBuffer<IndexT> self_collision;
            self_collision.resize(num_verts);
            if(num_verts > 0)
                self_collision.view().copy_from(h_vert_self_collision);

            lud::LDeviceBuffer<LAABB> tri_aabbs;
            lud::LDeviceBuffer<IndexT> tri_bids;
            lud::LDeviceBuffer<IndexT> tri_cids;
            lud::LDeviceBuffer<LAABB> edge_aabbs;
            lud::LDeviceBuffer<IndexT> edge_bids;
            lud::LDeviceBuffer<IndexT> edge_cids;

            // LC rejects a zero-length buffer view as a kernel argument: the arrays are
            // padded to one element and the kernels' `i < n` guard keeps them untouched.
            tri_aabbs.resize(padded(num_tris));
            tri_bids.resize(padded(num_tris));
            tri_cids.resize(padded(num_tris));
            edge_aabbs.resize(padded(num_edges));
            edge_bids.resize(padded(num_edges));
            edge_cids.resize(padded(num_edges));

            if(num_tris > 0)
            {
                const auto n = static_cast<std::uint32_t>(num_tris);
                stream.record(build_tri_aabbs_kernel()(positions.view(), triangles.view(),
                                                       vert_bids.view(), vert_cids.view(),
                                                       tri_aabbs.view(), tri_bids.view(),
                                                       tri_cids.view(), n)
                                  .dispatch(round_up(n, 256u)));
            }
            if(num_edges > 0)
            {
                const auto n = static_cast<std::uint32_t>(num_edges);
                stream.record(build_edge_aabbs_kernel()(positions.view(), edges.view(),
                                                        vert_bids.view(), vert_cids.view(),
                                                        edge_aabbs.view(), edge_bids.view(),
                                                        edge_cids.view(), n)
                                  .dispatch(round_up(n, 256u)));
            }

            // ---- the structure: query the triangle BVH with the edges ------------
            InfoStacklessBVH              bvh;
            InfoStacklessBVH::QueryBuffer qbuffer;
            bvh.build(tri_aabbs.cview(), tri_bids.cview(), tri_cids.cview());

            // the reference's `cmts.view()` (a `CDense2D` the traversal never reads; the
            // port has no 2-D view, so an empty mask view is passed — see the header of
            // info_stackless_bvh.h)
            bvh.query(edge_aabbs.cview(), edge_bids.cview(), edge_cids.cview(),
                      lud::LCBufferView<IndexT>{}, KeepAllNodes{}, KeepAllLeaves{},
                      qbuffer);

            vector<LVector2i> h_pairs(qbuffer.size());
            if(!h_pairs.empty())
            {
                qbuffer.view().copy_to(h_pairs);
                stream.synchronize();
            }

            // ---- the reference's LeafPred, evaluated on the host -------------------
            vector<Vector2i> result;
            result.reserve(h_pairs.size());
            for(const auto& pair : h_pairs)
            {
                const IndexT   edge_idx = pair(0);
                const IndexT   tri_idx  = pair(1);
                const Vector2i E        = Es[edge_idx];
                const Vector3i F        = Fs[tri_idx];

                if(E[0] == F[0] || E[0] == F[1] || E[0] == F[2] || E[1] == F[0]
                   || E[1] == F[1] || E[1] == F[2])
                    continue;

                bool ok = true;
                for(int ei = 0; ei < 2 && ok; ++ei)
                    for(int fi = 0; fi < 3 && ok; ++fi)
                    {
                        IndexT scid_e = h_vert_scids[E[ei]];
                        IndexT scid_f = h_vert_scids[F[fi]];
                        if(h_subscene_mask[scid_e * subscene_element_count + scid_f] == 0)
                            ok = false;
                    }
                if(!ok)
                    continue;

                ok = true;
                for(int ei = 0; ei < 2 && ok; ++ei)
                    for(int fi = 0; fi < 3 && ok; ++fi)
                    {
                        IndexT cid_e = h_vert_cids[E[ei]];
                        IndexT cid_f = h_vert_cids[F[fi]];
                        if(h_contact_mask[cid_e * contact_element_count + cid_f] == 0)
                            ok = false;
                    }
                if(!ok)
                    continue;

                if(h_vert_bids[E[0]] == h_vert_bids[F[0]] && h_vert_bids[E[0]] != -1
                   && !h_vert_self_collision[E[0]])
                    continue;

                if(!tri_edge_intersect(Vs[F[0]], Vs[F[1]], Vs[F[2]], Vs[E[0]],
                                       Vs[E[1]]))
                    continue;

                result.push_back(to_eigen(pair));
            }

            return result;
        }
    };

    static geometry::SimplicialComplex extract_intersected_mesh(
        const geometry::SimplicialComplex& scene_surface,
        span<const IndexT>                 vert_intersected,
        span<const IndexT>                 edge_intersected,
        span<const IndexT>                 tri_intersected)
    {
        geometry::SimplicialComplex i_mesh;

        vector<SizeT> intersected_verts;
        vector<SizeT> intersected_edges;
        vector<SizeT> intersected_tris;

        for(SizeT i = 0; i < vert_intersected.size(); i++)
            if(vert_intersected[i])
                intersected_verts.push_back(i);
        for(SizeT i = 0; i < edge_intersected.size(); i++)
            if(edge_intersected[i])
                intersected_edges.push_back(i);
        for(SizeT i = 0; i < tri_intersected.size(); i++)
            if(tri_intersected[i])
                intersected_tris.push_back(i);

        i_mesh.vertices().resize(intersected_verts.size());
        i_mesh.vertices().copy_from(scene_surface.vertices(),
                                    geometry::AttributeCopy::pull(intersected_verts));
        i_mesh.edges().resize(intersected_edges.size());
        i_mesh.edges().copy_from(scene_surface.edges(),
                                 geometry::AttributeCopy::pull(intersected_edges));
        i_mesh.triangles().resize(intersected_tris.size());
        i_mesh.triangles().copy_from(scene_surface.triangles(),
                                     geometry::AttributeCopy::pull(intersected_tris));

        vector<IndexT> vertex_remap(scene_surface.vertices().size(), -1);
        for(auto [i, v] : enumerate(intersected_verts))
            vertex_remap[v] = i;

        auto Map = [&]<IndexT N>(const Eigen::Vector<IndexT, N>& V) -> Eigen::Vector<IndexT, N>
        {
            auto ret = V;
            for(auto& v : ret)
                v = vertex_remap[v];
            return ret;
        };

        auto edge_topo_view = view(i_mesh.edges().topo());
        std::ranges::transform(edge_topo_view, edge_topo_view.begin(), Map);

        auto tri_topo_view = view(i_mesh.triangles().topo());
        std::ranges::transform(tri_topo_view, tri_topo_view.begin(), Map);

        return i_mesh;
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

        if(Vs.size() == 0 || Es.size() == 0 || Fs.size() == 0)
            return SanityCheckResult::Success;

        auto& contact_tabular  = ctx.contact_tabular();
        auto& subscene_tabular = ctx.subscene_tabular();

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

        auto attr_v_instance_id =
            scene_surface.vertices().find<IndexT>("sanity_check/instance_id");
        UIPC_ASSERT(attr_v_instance_id,
                    "`sanity_check/instance_id` is not found in scene surface");
        auto VInstanceIds = attr_v_instance_id->view();

        auto attr_self_collision =
            scene_surface.vertices().find<IndexT>("sanity_check/self_collision");
        UIPC_ASSERT(attr_self_collision,
                    "`sanity_check/self_collision` is not found in scene surface");
        auto SelfCollision = attr_self_collision->view();

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

        SizeT          CN = contact_tabular.element_count();
        vector<IndexT> h_contact_mask(CN * CN);
        for(IndexT i = 0; i < (IndexT)CN; ++i)
            for(IndexT j = 0; j < (IndexT)CN; ++j)
                h_contact_mask[i * CN + j] =
                    contact_tabular.at(i, j).is_enabled() ? 1 : 0;

        SizeT          SN = subscene_tabular.element_count();
        vector<IndexT> h_subscene_mask(SN * SN);
        for(IndexT i = 0; i < (IndexT)SN; ++i)
            for(IndexT j = 0; j < (IndexT)SN; ++j)
                h_subscene_mask[i * SN + j] =
                    subscene_tabular.at(i, j).is_enabled() ? 1 : 0;

        auto pairs = m_impl.detect_intersections(
            Vs,
            Es,
            Fs,
            span<const IndexT>{VInstanceIds.data(), VInstanceIds.size()},
            span<const IndexT>{CIds.data(), CIds.size()},
            span<const IndexT>{SCIds.data(), SCIds.size()},
            span<const IndexT>{SelfCollision.data(), SelfCollision.size()},
            span<const IndexT>{h_contact_mask},
            CN,
            span<const IndexT>{h_subscene_mask},
            SN);

        if(pairs.empty())
            return SanityCheckResult::Success;

        SizeT num_verts = Vs.size();
        SizeT num_edges = Es.size();
        SizeT num_tris  = Fs.size();

        vector<IndexT> vertex_intersected(num_verts, 0);
        vector<IndexT> edge_intersected(num_edges, 0);
        vector<IndexT> tri_intersected(num_tris, 0);

        map<Vector2i, Vector2i, Vec2iLess> intersected_geo_ids;

        for(auto& pair : pairs)
        {
            IndexT edge_idx = pair[0];
            IndexT tri_idx  = pair[1];

            Vector2i E = Es[edge_idx];
            Vector3i F = Fs[tri_idx];

            edge_intersected[edge_idx] = 1;
            tri_intersected[tri_idx]   = 1;

            vertex_intersected[E[0]] = 1;
            vertex_intersected[E[1]] = 1;
            vertex_intersected[F[0]] = 1;
            vertex_intersected[F[1]] = 1;
            vertex_intersected[F[2]] = 1;

            auto GeoIdL = VGeoIds[E[0]];
            auto GeoIdR = VGeoIds[F[0]];

            auto ObjIdL = VObjectIds[E[0]];
            auto ObjIdR = VObjectIds[F[0]];

            auto InstIdL = VInstanceIds[E[0]];
            auto InstIdR = VInstanceIds[F[0]];

            auto SelfCollL = SelfCollision[E[0]];
            auto SelfCollR = SelfCollision[F[0]];

            logger::error(
                "Intersection detected between Edge({},{}) in Geometry({}) "
                "Instance({}) Object[{}] and Triangle({},{},{}) in "
                "Geometry({}) Instance({}) Object[{}], SelfColl({},{})",
                E[0],
                E[1],
                GeoIdL,
                InstIdL,
                ObjIdL,
                F[0],
                F[1],
                F[2],
                GeoIdR,
                InstIdR,
                ObjIdR,
                SelfCollL,
                SelfCollR);

            if(GeoIdL > GeoIdR)
            {
                std::swap(GeoIdL, GeoIdR);
                std::swap(ObjIdL, ObjIdR);
            }
            intersected_geo_ids[{GeoIdL, GeoIdR}] = {ObjIdL, ObjIdR};
        }

        ::uipc::backend::SanityCheckMessageVisitor scmv{msg};
        auto&                                      buffer = scmv.message();

        for(auto& [GeoIds, ObjIds] : intersected_geo_ids)
        {
            auto obj_0 = find_object(ObjIds[0]);
            auto obj_1 = find_object(ObjIds[1]);

            UIPC_ASSERT(obj_0 != nullptr, "Object[{}] not found", ObjIds[0]);
            UIPC_ASSERT(obj_1 != nullptr, "Object[{}] not found", ObjIds[1]);

            std::string name_0{obj_0->name()};
            std::string name_1{obj_1->name()};

            fmt::format_to(std::back_inserter(buffer),
                           "Geometry({}) in Object[{}({})] intersects with Geometry({}) in "
                           "Object[{}({})]\n",
                           GeoIds[0],
                           name_0,
                           obj_0->id(),
                           GeoIds[1],
                           name_1,
                           obj_1->id());
        }

        auto intersected_mesh = extract_intersected_mesh(
            scene_surface, vertex_intersected, edge_intersected, tri_intersected);

        fmt::format_to(std::back_inserter(buffer),
                       "Intersected mesh has {} vertices, {} edges, and {} triangles.\n",
                       intersected_mesh.vertices().size(),
                       intersected_mesh.edges().size(),
                       intersected_mesh.triangles().size());

        std::string name = "intersected_mesh";

        auto sanity_check_mode = ctx.config().find<std::string>("sanity_check/mode");
        if(sanity_check_mode->view()[0] == "normal")
        {
            auto output_path = this_output_path();
            namespace fs     = std::filesystem;
            fs::path path{output_path};
            path /= fmt::format("{}.obj", name);
            auto path_str = path.string();

            geometry::SimplicialComplexIO io;
            {
                auto is_facet = intersected_mesh.edges().find<IndexT>(builtin::is_facet);
                if(!is_facet)
                    intersected_mesh.edges().create<IndexT>(builtin::is_facet, 0);
                auto is_facet_view = view(*is_facet);
                std::ranges::fill(is_facet_view, 1);
            }
            io.write(path_str, intersected_mesh);
            fmt::format_to(std::back_inserter(buffer),
                           "Intersected mesh is saved at {}.\n",
                           path_str);
        }

        fmt::format_to(std::back_inserter(buffer),
                       "Create mesh [{}<{}>] for post-processing.",
                       name,
                       intersected_mesh.type());

        scmv.geometries()[name] =
            uipc::make_shared<geometry::SimplicialComplex>(std::move(intersected_mesh));

        return SanityCheckResult::Error;
    }

  private:
    Impl m_impl;
};

REGISTER_BACKEND_SANITY_CHECKER(SimplicialSurfaceIntersectionCheck);
}// namespace uipc::backend::luisa
