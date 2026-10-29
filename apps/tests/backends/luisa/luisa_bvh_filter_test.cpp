/********************************************************************
 * @file luisa_bvh_filter_test.cpp
 * @brief T7.2b — the four BVH simplex trajectory filters (LuisaCompute backend).
 *        Tag [bvh_filter].
 *
 * The batch's four deliverables are
 *   collision_detection/filters/{lbvh,stackless_bvh,info_stackless_bvh,
 *                                info_stackless_bvh_v0}_simplex_trajectory_filter.{h,cpp}
 * and the *batch's most valuable test* is the four-way table: which of the four
 * produces which candidate set, and which differences are semantic.
 *
 * Coverage map (brief §Tests):
 * 1) the per-method registration gating, the base accessor surface, the scene
 *    contracts and the two `QueryBuffer` capacity contracts, plus the four-way
 *    DCD table and the info variants' bid/cid arrays ... t72b_registration_and_contracts
 * 2) the LBVH   variant's candidate production vs the host oracle (all four pair
 *    kinds, at alpha != 0 through the pipeline and at alpha == 0 through its `Impl`)
 *    and its degenerate cases ............................. t72b_oracle_lbvh
 * 3) the same battery for the stackless variant .......... t72b_oracle_stackless
 * 4) the same battery for info_stackless_bvh (v1) ........ t72b_oracle_infov1
 * 5) the same battery for info_stackless_bvh_v0 (v0) + the v1-vs-v0 relation
 *    ..................................................... t72b_oracle_infov0
 * 6) the mask semantics (contact + subscene, both directions), the empty/zero
 *    candidate path and the pair-buffer capacity/retry contract
 *    ..................................................... t72b_masks_empty_capacity
 * 7) determinism + device-vs-host for the batch's factored routines (the layer's
 *    struct compaction, the ported distance/primitive_d_hat layers)
 *    ..................................................... t72b_determinism_and_device_vs_host
 * 8) the quoted numbers ................................. t72b_summary
 *
 * HOW THE SCENES AND THE ORACLES WORK
 * -----------------------------------
 * The scene is the ordinary frontend one (the sibling luisa tests' convention): one
 * or more `SimplicialComplex` "plates" carrying the shared per-vertex `test_*`
 * attributes (dimension / contact id / subscene id / d_hat / thickness /
 * displacement) and a per-geometry `test_self_collision`.  The *reports* into the
 * global vertex / surface / body managers come from the sibling test files'
 * registered reporters (every geometry is reported), which is what makes the global
 * arrays readable here through the ported managers (t72a's convention).
 *
 * Two independent ways into the filters:
 *
 *  (a) **the pipeline**: `GlobalTrajectoryFilter::run_stage(Stage, alpha)` (the T7.2a
 *      seam) drives the *registered* filter for the scene's
 *      `collision_detection/method`, and the test reads the reference's four exports
 *      (`candidate_PTs()/candidate_EEs()/toi_PTs()/toi_EEs()`) plus the active sets
 *      (`PTs()/EEs()/PEs()/PPs()`).  This is the only route that can set `alpha != 0`
 *      (`DetectInfo::m_alpha` is private to the filter base), and it is the route the
 *      reference's own consumers use.
 *  (b) **the `Impl` directly**: `SimplexTrajectoryFilter::Impl` and the four filters'
 *      `Impl`s are public nested classes with public members (the reference's own
 *      layout), so the test constructs them, wires the probe's managers and calls
 *      `detect()`; the `DetectInfo` it can build has `alpha == 0` (the private
 *      default), which is exactly the DCD pass where the four variants differ.  This
 *      route exposes the *raw* (keep-all traversal) pairs, the *selected* candidate
 *      pairs of all four kinds -- including the two the reference does not export --
 *      the swept AABBs and the info variants' bid/cid arrays.
 *
 * Every expectation is an independent host oracle over the *device* arrays the test
 * read back through the managers: the reference's four swept-AABB builders, a
 * brute-force box-overlap enumeration, and the reference's own per-pair predicate
 * bodies re-implemented here (the `ccd_broadphase` tests are spelled out
 * componentwise, and the ported templated distance layer is cross-checked against
 * them device-vs-host).  Counts are quoted for every case.
 *
 * HAZARD (pinned, not worked around): the reference's `filter_active`
 * `UIPC_KERNEL_ASSERT(D > thickness^2)` *aborts the process* when a candidate pair is
 * closer than the thickness (the reference's own "this should not happen" invariant).
 * The degenerate scenes in this file therefore run `detect` only, and the scenes that
 * run `filter_active` keep every pair beyond `thickness`.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/unit.h>
#include <uipc/core/contact_tabular.h>
#include <uipc/core/subscene_tabular.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>

#include <type_define.h>
#include <collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.h>
#include <collision_detection/filters/info_stackless_bvh_v0_simplex_trajectory_filter.h>
#include <collision_detection/filters/lbvh_simplex_trajectory_filter.h>
#include <collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <frame_stream.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <luisa_device/linear/algorithms.h>
#include <luisa_device/math/distance/distance.h>
#include <utils/codim_thickness.h>
#include <utils/primitive_d_hat.h>
// the test target's include path puts src/backends/none first (cookbook §7.14 rule 7)
#include <backends/luisa/sim_engine.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
namespace gg  = uipc::backend::luisa;
namespace lud = uipc::backend::luisa::luisa_device;

namespace t72b
{
namespace lcrt   = ::luisa::compute;
namespace lla    = uipc::backend::luisa::luisa_device::linalg;
namespace ldist  = uipc::backend::luisa::luisa_device::math::distance;
namespace lps    = uipc::backend::luisa::luisa_device::math;
namespace lprim  = uipc::backend::luisa;  // the ported primitive_d_hat layer

using PairSet = std::set<std::pair<IndexT, IndexT>>;

// ===========================================================================
//  read-back helpers (the T7.2a spellings)
// ===========================================================================
std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

template <typename T>
std::vector<T> read_device(const lud::LCBufferView<T>& view)
{
    std::vector<T> host(view.size());
    gg::frame_stream().synchronize();
    if(!host.empty())
    {
        view.copy_to(span<T>{host});
        lud::main_stream().synchronize();
    }
    return host;
}

std::vector<Vector3> read_vector3(const lud::LCBufferView<gg::LVector3>& view)
{
    std::vector<Vector3> host;
    auto                 raw = read_device(view);
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

std::vector<Vector2i> read_vector2i(const lud::LCBufferView<gg::LVector2i>& view)
{
    std::vector<Vector2i> host;
    auto                  raw = read_device(view);
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

std::vector<Vector3i> read_vector3i(const lud::LCBufferView<gg::LVector3i>& view)
{
    std::vector<Vector3i> host;
    auto                  raw = read_device(view);
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

std::vector<Vector4i> read_vector4i(const lud::LCBufferView<gg::LVector4i>& view)
{
    std::vector<Vector4i> host;
    auto                  raw = read_device(view);
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

PairSet to_pair_set(const std::vector<Vector2i>& raw)
{
    PairSet out;
    for(auto& p : raw)
        out.emplace(p[0], p[1]);
    return out;
}

/// The device's toi slices are *parallel* to its candidate slices (the traversal /
/// compaction order), while the oracles are pair-keyed maps; re-key the device's values
/// so the comparison is order-independent.  (The earlier element-wise walk silently
/// passed while every value was the same `1.1` sentinel, and reported bogus mismatches as
/// soon as the closing probe made the pairs really collide -- found in T7.2b.)
inline std::map<std::pair<IndexT, IndexT>, Float> toi_by_pair(
    const std::vector<Vector2i>& pairs,
    const std::vector<Float>&    values)
{
    REQUIRE(pairs.size() == values.size());
    std::map<std::pair<IndexT, IndexT>, Float> out;
    for(size_t i = 0; i < pairs.size(); ++i)
        out[std::make_pair(static_cast<IndexT>(pairs[i][0]),
                           static_cast<IndexT>(pairs[i][1]))] = values[i];
    return out;
}

PairSet to_pair_set4_first2(const std::vector<Vector4i>& raw)
{
    PairSet out;
    for(auto& p : raw)
        out.emplace(p[0], p[1]);
    return out;
}

PairSet to_pair_set3(const std::vector<Vector3i>& raw)
{
    PairSet out;
    for(auto& p : raw)
        out.emplace(p[0], p[1]);
    return out;
}

std::string set_to_string(const PairSet& s)
{
    std::string out = "{";
    for(auto& p : s)
        out += std::to_string(p.first) + "-" + std::to_string(p.second) + " ";
    return out + "}";
}

// ===========================================================================
//  The reference's four swept-AABB builders, in the reference's float arithmetic
// ===========================================================================
/// Eigen's `AlignedBox<float,3>` is the reference's AABB type (the port keeps it as
/// the host type; the device element is `LAABB`, byte-identical).
using FAABB = gg::AABB;  // Eigen::AlignedBox<float,3> (the reference host type)

FAABB host_swept_box(std::initializer_list<Vector3> pos,
                     std::initializer_list<Vector3> disp,
                     Float                          alpha,
                     Float                          thickness,
                     Float                          d_hat_expansion)
{
    FAABB box;
    box.setEmpty();
    auto  n = pos.size();
    auto  p = pos.begin();
    auto  d = disp.begin();
    for(size_t k = 0; k < n; ++k, ++p, ++d)
    {
        // the reference: `Vector3 pos_t = pos + dxs(vI) * alpha;` in double, then
        // `aabb.extend(pos.cast<float>()).extend(pos_t.cast<float>())`
        Vector3 pos_t = *p + (*d) * alpha;
        box.extend(p->cast<float>());
        box.extend(pos_t.cast<float>());
    }
    // the reference: `float expand = d_hat_expansion + thickness;`
    const float expand = static_cast<float>(d_hat_expansion + thickness);
    box.min().array() -= expand;
    box.max().array() += expand;
    return box;
}

bool host_box_intersects(const FAABB& a, const FAABB& b)
{
    return (a.min().array() <= b.max().array()).all()
           && (b.min().array() <= a.max().array()).all();
}

// ===========================================================================
//  The reference's per-pair predicate bodies, re-implemented on the host
// ===========================================================================
/// `distance::point_point_ccd_broadphase` etc., spelled out componentwise
/// (`array_max(p, p + dp)` and `!(any(min_a - max_b > dist) || any(min_b - max_a > dist))`).
struct SweptBox
{
    Vector3 lo;
    Vector3 hi;
};

SweptBox swept_box(std::initializer_list<Vector3> p, std::initializer_list<Vector3> d)
{
    SweptBox b;
    b.lo = *p.begin();
    b.hi = *p.begin();
    auto pv = p.begin();
    auto dv = d.begin();
    for(size_t k = 0; k < p.size(); ++k, ++pv, ++dv)
    {
        Vector3 moved = *pv + *dv;
        for(int c = 0; c < 3; ++c)
        {
            b.lo[c] = std::min(b.lo[c], std::min((*pv)[c], moved[c]));
            b.hi[c] = std::max(b.hi[c], std::max((*pv)[c], moved[c]));
        }
    }
    return b;
}

bool host_boxes_far(const SweptBox& a, const SweptBox& b, Float dist)
{
    return ((a.lo - b.hi).array() > dist).any() || ((b.lo - a.hi).array() > dist).any();
}

bool host_pp_broadphase(const Vector3& p0,
                        const Vector3& p1,
                        const Vector3& dp0,
                        const Vector3& dp1,
                        Float          dist)
{
    return !host_boxes_far(swept_box({p0}, {dp0}), swept_box({p1}, {dp1}), dist);
}

bool host_pe_broadphase(const Vector3& p,
                        const Vector3& e0,
                        const Vector3& e1,
                        const Vector3& dp,
                        const Vector3& de0,
                        const Vector3& de1,
                        Float          dist)
{
    return !host_boxes_far(swept_box({p}, {dp}), swept_box({e0, e1}, {de0, de1}), dist);
}

bool host_pt_broadphase(const Vector3& p,
                        const Vector3& t0,
                        const Vector3& t1,
                        const Vector3& t2,
                        const Vector3& dp,
                        const Vector3& dt0,
                        const Vector3& dt1,
                        const Vector3& dt2,
                        Float          dist)
{
    return !host_boxes_far(swept_box({p}, {dp}),
                           swept_box({t0, t1, t2}, {dt0, dt1, dt2}),
                           dist);
}

bool host_ee_broadphase(const Vector3& ea0,
                        const Vector3& ea1,
                        const Vector3& eb0,
                        const Vector3& eb1,
                        const Vector3& dea0,
                        const Vector3& dea1,
                        const Vector3& deb0,
                        const Vector3& deb1,
                        Float          dist)
{
    return !host_boxes_far(swept_box({ea0, ea1}, {dea0, dea1}),
                           swept_box({eb0, eb1}, {deb0, deb1}),
                           dist);
}

/// The reference's `utils/simplex_contact_mask_utils.h` over the port's flat mask
/// (`index = row * extent + col`, cookbook §7.33 rule 5).
bool allow_PP(const std::vector<IndexT>& t, IndexT dim, IndexT a, IndexT b)
{
    return t[a * dim + b] != 0;
}
bool allow_PE(const std::vector<IndexT>& t, IndexT dim, IndexT a, IndexT b, IndexT c)
{
    return allow_PP(t, dim, a, b) && allow_PP(t, dim, a, c);
}
bool allow_PT(const std::vector<IndexT>& t, IndexT dim, IndexT a, IndexT b, IndexT c, IndexT d)
{
    return allow_PE(t, dim, a, b, c) && allow_PP(t, dim, a, d);
}
bool allow_EE(const std::vector<IndexT>& t, IndexT dim, IndexT a, IndexT b, IndexT c, IndexT d)
{
    return allow_PP(t, dim, a, c) && allow_PP(t, dim, a, d) && allow_PP(t, dim, b, c)
           && allow_PP(t, dim, b, d);
}

// ===========================================================================
//  The gathered device data the oracles run on
// ===========================================================================
struct SceneData
{
    std::vector<IndexT>  surf_vertices;
    std::vector<IndexT>  codim_vertices;
    std::vector<Vector2i> surf_edges;
    std::vector<Vector3i> surf_triangles;

    std::vector<Vector3> positions;
    std::vector<Vector3> rest_positions;
    std::vector<Vector3> displacements;
    std::vector<Float>   thicknesses;
    std::vector<Float>   d_hats;
    std::vector<IndexT>  dimensions;
    std::vector<IndexT>  contact_ids;
    std::vector<IndexT>  subscene_ids;
    std::vector<IndexT>  v2b;
    std::vector<IndexT>  body_self_collision;

    std::vector<IndexT> contact_mask;
    IndexT              contact_mask_size = 0;
    std::vector<IndexT> subscene_mask;
    IndexT              subscene_mask_size = 0;

    /// the reference's four swept-AABB sets
    std::vector<FAABB> codim_point_aabbs;
    std::vector<FAABB> point_aabbs;
    std::vector<FAABB> edge_aabbs;
    std::vector<FAABB> triangle_aabbs;

    void build_aabbs(Float alpha)
    {
        codim_point_aabbs.clear();
        point_aabbs.clear();
        edge_aabbs.clear();
        triangle_aabbs.clear();

        for(auto vI : codim_vertices)
            codim_point_aabbs.push_back(
                host_swept_box({positions[vI]},
                               {displacements[vI]},
                               alpha,
                               thicknesses[vI],
                               lprim::point_dcd_expansion(d_hats[vI])));

        for(auto vI : surf_vertices)
            point_aabbs.push_back(host_swept_box({positions[vI]},
                                                 {displacements[vI]},
                                                 alpha,
                                                 thicknesses[vI],
                                                 lprim::point_dcd_expansion(d_hats[vI])));

        for(auto& E : surf_edges)
            edge_aabbs.push_back(host_swept_box({positions[E[0]], positions[E[1]]},
                                                {displacements[E[0]], displacements[E[1]]},
                                                alpha,
                                                gg::edge_thickness(thicknesses[E[0]],
                                                               thicknesses[E[1]]),
                                                lprim::edge_dcd_expansion(d_hats[E[0]],
                                                                   d_hats[E[1]])));

        for(auto& F : surf_triangles)
            triangle_aabbs.push_back(
                host_swept_box({positions[F[0]], positions[F[1]], positions[F[2]]},
                               {displacements[F[0]], displacements[F[1]], displacements[F[2]]},
                               alpha,
                               gg::triangle_thickness(thicknesses[F[0]],
                                                  thicknesses[F[1]],
                                                  thicknesses[F[2]]),
                               lprim::triangle_dcd_expansion(d_hats[F[0]],
                                                      d_hats[F[1]],
                                                      d_hats[F[2]])));
    }
};

// ===========================================================================
//  The oracle: the raw (keep-all traversal) pair sets and the predicate bodies
// ===========================================================================
PairSet host_raw_allp_codimp(const SceneData& d)
{
    PairSet out;
    for(size_t i = 0; i < d.surf_vertices.size(); ++i)
        for(size_t j = 0; j < d.codim_vertices.size(); ++j)
            if(host_box_intersects(d.point_aabbs[i], d.codim_point_aabbs[j]))
                out.emplace(static_cast<IndexT>(i), static_cast<IndexT>(j));
    return out;
}

PairSet host_raw_codimp_alle(const SceneData& d)
{
    PairSet out;
    for(size_t i = 0; i < d.codim_vertices.size(); ++i)
        for(size_t j = 0; j < d.surf_edges.size(); ++j)
            if(host_box_intersects(d.codim_point_aabbs[i], d.edge_aabbs[j]))
                out.emplace(static_cast<IndexT>(i), static_cast<IndexT>(j));
    return out;
}

PairSet host_raw_alle_alle(const SceneData& d)
{
    PairSet out;
    for(size_t i = 0; i < d.surf_edges.size(); ++i)
        for(size_t j = i + 1; j < d.surf_edges.size(); ++j)
            if(host_box_intersects(d.edge_aabbs[i], d.edge_aabbs[j]))
                out.emplace(static_cast<IndexT>(i), static_cast<IndexT>(j));
    return out;
}

PairSet host_raw_allp_allt(const SceneData& d)
{
    PairSet out;
    for(size_t i = 0; i < d.surf_vertices.size(); ++i)
        for(size_t j = 0; j < d.surf_triangles.size(); ++j)
            if(host_box_intersects(d.point_aabbs[i], d.triangle_aabbs[j]))
                out.emplace(static_cast<IndexT>(i), static_cast<IndexT>(j));
    return out;
}

/// The reference's `*_detect_AllP_CodimP_pred` body (`dcd` = the info v1 flavour's
/// `alpha == 0` exact-distance branch).
bool host_pred_allp_codimp(const SceneData& d, IndexT i, IndexT j, Float alpha, bool dcd)
{
    const auto V      = d.surf_vertices[i];
    const auto codimV = d.codim_vertices[j];

    const IndexT cids[2]  = {d.contact_ids[V], d.contact_ids[codimV]};
    const IndexT scids[2] = {d.subscene_ids[V], d.subscene_ids[codimV]};
    if(!allow_PP(d.subscene_mask, d.subscene_mask_size, scids[0], scids[1]))
        return false;
    if(!allow_PP(d.contact_mask, d.contact_mask_size, cids[0], cids[1]))
        return false;

    if((d.dimensions[V] <= 2) && (V >= codimV))
        return false;

    const auto body_i = d.v2b[V];
    const auto body_j = d.v2b[codimV];
    if((body_i == body_j) && (d.body_self_collision[body_i] == 0))
        return false;

    const Vector3 P0 = d.positions[V];
    const Vector3 P1 = d.positions[codimV];

    const Float thickness = d.thicknesses[V] + d.thicknesses[codimV];
    const Float d_hat     = (d.d_hats[V] + d.d_hats[codimV]) / 2.0;
    const Float expand    = d_hat + thickness;

    if(dcd && alpha == 0.0)
    {
        const Float D2 = (P0 - P1).squaredNorm();
        return D2 < expand * expand;
    }

    const Vector3 dP0 = alpha * d.displacements[V];
    const Vector3 dP1 = alpha * d.displacements[codimV];
    return host_pp_broadphase(P0, P1, dP0, dP1, expand);
}

/// The reference's `*_detect_CodimP_AllE_pred` body.
bool host_pred_codimp_alle(const SceneData& d, IndexT i, IndexT j, Float alpha, bool dcd)
{
    const auto codimV = d.codim_vertices[i];
    const auto E      = d.surf_edges[j];

    const IndexT cids[3] = {d.contact_ids[codimV], d.contact_ids[E[0]], d.contact_ids[E[1]]};
    const IndexT scids[3] = {d.subscene_ids[codimV], d.subscene_ids[E[0]], d.subscene_ids[E[1]]};
    if(!allow_PE(d.subscene_mask, d.subscene_mask_size, scids[0], scids[1], scids[2]))
        return false;
    if(!allow_PE(d.contact_mask, d.contact_mask_size, cids[0], cids[1], cids[2]))
        return false;

    if(E[0] == codimV || E[1] == codimV)
        return false;

    const auto body_i = d.v2b[codimV];
    const auto body_j = d.v2b[E[0]];
    if((body_i == body_j) && (d.body_self_collision[body_i] == 0))
        return false;

    const Vector3 P  = d.positions[codimV];
    const Vector3 E0 = d.positions[E[0]];
    const Vector3 E1 = d.positions[E[1]];

    const Float thickness =
        d.thicknesses[codimV] + d.thicknesses[E[0]];  // PE_thickness == add(P, E0)
    const Float d_hat = (d.d_hats[codimV] + d.d_hats[E[0]]) / 2.0;
    const Float expand = d_hat + thickness;

    if(dcd && alpha == 0.0)
    {
        // the ported *host* instantiation of the reference's flagged distance (the
        // flag picks the active sub-simplex exactly as the reference does)
        const auto flag = ldist::point_edge_distance_flag<Float>(gg::to_luisa(P),
                                                                 gg::to_luisa(E0),
                                                                 gg::to_luisa(E1));
        Float      D2{};
        ldist::point_edge_distance2(flag,
                                    gg::to_luisa(P),
                                    gg::to_luisa(E0),
                                    gg::to_luisa(E1),
                                    D2);
        return D2 < expand * expand;
    }

    const Vector3 dP  = alpha * d.displacements[codimV];
    const Vector3 dE0 = alpha * d.displacements[E[0]];
    const Vector3 dE1 = alpha * d.displacements[E[1]];
    return host_pe_broadphase(P, E0, E1, dP, dE0, dE1, expand);
}

/// The reference's `*_detect_AllE_AllE_pred` body (the reported pairs are the ordered
/// `i < j` ones).
bool host_pred_alle_alle(const SceneData& d, IndexT i, IndexT j, Float alpha, bool dcd)
{
    const auto E0 = d.surf_edges[i];
    const auto E1 = d.surf_edges[j];

    const IndexT cids[4] = {d.contact_ids[E0[0]],
                            d.contact_ids[E0[1]],
                            d.contact_ids[E1[0]],
                            d.contact_ids[E1[1]]};
    const IndexT scids[4] = {d.subscene_ids[E0[0]],
                             d.subscene_ids[E0[1]],
                             d.subscene_ids[E1[0]],
                             d.subscene_ids[E1[1]]};
    if(!allow_EE(d.subscene_mask, d.subscene_mask_size, scids[0], scids[1], scids[2], scids[3]))
        return false;
    if(!allow_EE(d.contact_mask, d.contact_mask_size, cids[0], cids[1], cids[2], cids[3]))
        return false;

    if(E0[0] == E1[0] || E0[0] == E1[1] || E0[1] == E1[0] || E0[1] == E1[1])
        return false;

    const auto body_i = d.v2b[E0[0]];
    const auto body_j = d.v2b[E1[0]];
    if((body_i == body_j) && (d.body_self_collision[body_i] == 0))
        return false;

    const Vector3 E0_0 = d.positions[E0[0]];
    const Vector3 E0_1 = d.positions[E0[1]];
    const Vector3 E1_0 = d.positions[E1[0]];
    const Vector3 E1_1 = d.positions[E1[1]];

    const Float thickness = d.thicknesses[E0[0]] + d.thicknesses[E1[0]];  // EE_thickness
    const Float d_hat     = (d.d_hats[E0[0]] + d.d_hats[E1[0]]) / 2.0;    // EE_d_hat
    const Float expand    = d_hat + thickness;

    if(dcd && alpha == 0.0)
    {
        const auto flag = ldist::edge_edge_distance_flag<Float>(gg::to_luisa(E0_0),
                                                                gg::to_luisa(E0_1),
                                                                gg::to_luisa(E1_0),
                                                                gg::to_luisa(E1_1));
        Float      D2{};
        ldist::edge_edge_distance2(flag,
                                   gg::to_luisa(E0_0),
                                   gg::to_luisa(E0_1),
                                   gg::to_luisa(E1_0),
                                   gg::to_luisa(E1_1),
                                   D2);
        return D2 < expand * expand;
    }

    const Vector3 dE0_0 = alpha * d.displacements[E0[0]];
    const Vector3 dE0_1 = alpha * d.displacements[E0[1]];
    const Vector3 dE1_0 = alpha * d.displacements[E1[0]];
    const Vector3 dE1_1 = alpha * d.displacements[E1[1]];
    return host_ee_broadphase(E0_0, E0_1, E1_0, E1_1, dE0_0, dE0_1, dE1_0, dE1_1, expand);
}

/// The reference's `*_detect_AllP_AllT_pred` body.
bool host_pred_allp_allt(const SceneData& d, IndexT i, IndexT j, Float alpha, bool dcd)
{
    const auto V = d.surf_vertices[i];
    const auto F = d.surf_triangles[j];

    const IndexT cids[4] = {d.contact_ids[V],
                            d.contact_ids[F[0]],
                            d.contact_ids[F[1]],
                            d.contact_ids[F[2]]};
    const IndexT scids[4] = {d.subscene_ids[V],
                             d.subscene_ids[F[0]],
                             d.subscene_ids[F[1]],
                             d.subscene_ids[F[2]]};
    if(!allow_PT(d.subscene_mask, d.subscene_mask_size, scids[0], scids[1], scids[2], scids[3]))
        return false;
    if(!allow_PT(d.contact_mask, d.contact_mask_size, cids[0], cids[1], cids[2], cids[3]))
        return false;

    if(F[0] == V || F[1] == V || F[2] == V)
        return false;

    const auto body_i = d.v2b[V];
    const auto body_j = d.v2b[F[0]];
    if((body_i == body_j) && (d.body_self_collision[body_i] == 0))
        return false;

    const Vector3 P  = d.positions[V];
    const Vector3 F0 = d.positions[F[0]];
    const Vector3 F1 = d.positions[F[1]];
    const Vector3 F2 = d.positions[F[2]];

    const Float thickness = d.thicknesses[V] + d.thicknesses[F[0]];      // PT_thickness
    const Float d_hat     = (d.d_hats[V] + d.d_hats[F[0]]) / 2.0;        // PT_d_hat
    const Float expand    = d_hat + thickness;

    if(dcd && alpha == 0.0)
    {
        const auto flag = ldist::point_triangle_distance_flag<Float>(gg::to_luisa(P),
                                                                    gg::to_luisa(F0),
                                                                    gg::to_luisa(F1),
                                                                    gg::to_luisa(F2));
        Float      D2{};
        ldist::point_triangle_distance2(flag,
                                        gg::to_luisa(P),
                                        gg::to_luisa(F0),
                                        gg::to_luisa(F1),
                                        gg::to_luisa(F2),
                                        D2);
        return D2 < expand * expand;
    }

    const Vector3 dP  = alpha * d.displacements[V];
    const Vector3 dF0 = alpha * d.displacements[F[0]];
    const Vector3 dF1 = alpha * d.displacements[F[1]];
    const Vector3 dF2 = alpha * d.displacements[F[2]];
    return host_pt_broadphase(P, F0, F1, F2, dP, dF0, dF1, dF2, expand);
}

/// The four *selected* candidate sets: the reference's `filter`-time view.
struct OracleSets
{
    PairSet allp_codimp;
    PairSet codimp_alle;
    PairSet alle_alle;
    PairSet allp_allt;

    [[nodiscard]] size_t total() const
    {
        return allp_codimp.size() + codimp_alle.size() + alle_alle.size() + allp_allt.size();
    }
};

OracleSets host_candidates(const SceneData& d, Float alpha, bool dcd)
{
    OracleSets out;
    for(auto& p : host_raw_allp_codimp(d))
        if(host_pred_allp_codimp(d, p.first, p.second, alpha, dcd))
            out.allp_codimp.insert(p);
    for(auto& p : host_raw_codimp_alle(d))
        if(host_pred_codimp_alle(d, p.first, p.second, alpha, dcd))
            out.codimp_alle.insert(p);
    for(auto& p : host_raw_alle_alle(d))
        if(host_pred_alle_alle(d, p.first, p.second, alpha, dcd))
            out.alle_alle.insert(p);
    for(auto& p : host_raw_allp_allt(d))
        if(host_pred_allp_allt(d, p.first, p.second, alpha, dcd))
            out.allp_allt.insert(p);
    return out;
}

// ===========================================================================
//  The scene: `SimplicialComplex` plates with the shared test_ attributes
// ===========================================================================
constexpr std::string_view attr_dimension    = "test_dimension";
constexpr std::string_view attr_contact_id   = "test_contact_element_id";
constexpr std::string_view attr_subscene_id  = "test_subscene_element_id";
constexpr std::string_view attr_d_hat        = "test_d_hat";
constexpr std::string_view attr_thickness    = "test_thickness";
constexpr std::string_view attr_displacement = "test_displacement";
constexpr std::string_view attr_self_collision = "test_self_collision";

struct MeshDesc
{
    std::vector<Vector3>  positions;
    std::vector<Vector3i> triangles;
    std::vector<IndexT>   dimensions;
    std::vector<IndexT>   contact_ids;
    std::vector<IndexT>   subscene_ids;
    std::vector<Float>    d_hats;
    std::vector<Float>    thicknesses;
    std::vector<Vector3>  displacements;
    IndexT                self_collision = 0;
};

/// A unit quad (4 vertices / 2 triangles), scaled by `scale` and translated to
/// `origin`.
///
/// The second plate is *scaled down and offset* on purpose: two congruent, axis-aligned
/// plates make a vertex's projection land exactly on the other plate's triangle diagonal,
/// where the reference's `point_triangle_distance_flag` classification sits on an FP
/// boundary (the ported device/host instantiations then legitimately disagree about the
/// degenerate sub-simplex -- a property of the reference's flag chain, recorded in the
/// cookbook).  An asymmetric pair of plates keeps every projection strictly inside a
/// triangle, so the classification is not boundary-sensitive.
MeshDesc plate(const Vector3& origin, Float scale = 1.0)
{
    MeshDesc m;
    m.positions = {{origin.x(), origin.y(), origin.z()},
                   {origin.x() + scale, origin.y(), origin.z()},
                   {origin.x(), origin.y(), origin.z() + scale},
                   {origin.x() + scale, origin.y(), origin.z() + scale}};
    m.triangles = {{0, 1, 2}, {1, 3, 2}};
    return m;
}

/// Does this (gathered) vertex belong to the *upper* plate -- the only geometry the
/// scene builders give a non-zero `test_displacement`?  The upper plate sits 0.15 above
/// the lower one, so the test must be `y > 0.05`; the earlier spelling (`y > 1.0`)
/// matched *no* vertex at all, which silently made `alpha` invisible to every
/// scene-level pin (found by the T7.2b closing probe).
inline bool is_upper_plate(const Vector3& p)
{
    return p.y() > 0.05;
}

/// The half-plane geometry every scene carries: the T7.2a half-plane filters
/// (`easy/al_vertex_half_plane_trajectory_filter`) register in *every* ipc scene, and
/// their `filter_active` reads the plane's vertex slice, so a scene without a half-plane
/// aborts inside the manager's stage loop (measured: a SIGSEGV in
/// `EasyVertexHalfPlaneTrajectoryFilter`'s `filter_active` on a plane-less scene).  The
/// plane sits 100 units below the scene, so it never contributes a candidate and its
/// TOI stays the 1.1f sentinel.
struct HPDesc
{
    Vector3 P{0.0, -100.0, 0.0};
    Vector3 N{0.0, 1.0, 0.0};
    IndexT  contact_id  = 0;
    IndexT  subscene_id = 0;
};

struct SceneDesc
{
    std::vector<MeshDesc> meshes;
    std::vector<HPDesc>   half_planes = {HPDesc{}};
    /// The test-local variant filters register only for a scene whose geometries carry
    /// their own `test_t72b_variant` meta uid (see `T72bVariantFilter`).  The scene's
    /// `collision_detection/method` is always the reference's default
    /// (`info_stackless_bvh`): this build's scene schema lists *only* that method
    /// (`UIPC_WITH_CUDA_LEGACY_COLLISION == 0` makes the schema's `conditionalValues`
    /// drop the other three), so the other three filters cannot be selected through the
    /// frontend -- the reference's own gating, transcribed.
    IndexT variant_uid = -1;
    /// the `collision_detection/method` value the scene is built with (this build's
    /// schema lists only this one, see the note above)
    std::string           method = "info_stackless_bvh";
    bool                  two_contact_elements    = false;
    bool                  contact_pair_disabled   = false;
    bool                  two_subscenes            = false;
    bool                  subscene_pair_enabled    = false;
    Float                 dt                       = 0.25;
};

Scene build_scene(const SceneDesc& desc)
{
    auto config                       = Scene::default_config();
    config["sanity_check"]["enable"]  = 0;
    config["dt"]                      = desc.dt;
    config["contact"]["constitution"] = std::string{"ipc"};
    config["collision_detection"]["method"] = desc.method;
    Scene scene{config};

    if(desc.two_contact_elements)
    {
        auto& ct = scene.contact_tabular();
        auto  e1 = ct.create("t72b_b");
        // the pair is disabled in *both* directions by the frontend's own insert
        ct.insert(e1, e1, 0.5, 1.0_GPa, !desc.contact_pair_disabled, Json::object());
    }
    if(desc.two_subscenes)
    {
        auto& st = scene.subscene_tabular();
        auto  s1 = st.create("t72b_s1");
        if(desc.subscene_pair_enabled)
            st.insert(st.default_element(), s1, true, Json::object());
    }

    for(size_t i = 0; i < desc.half_planes.size(); ++i)
    {
        auto        object = scene.objects().create("t72b_ground_" + std::to_string(i));
        const auto& hp     = desc.half_planes[i];
        auto        ig     = halfplane(hp.P, hp.N);
        ig.meta().create<IndexT>(builtin::contact_element_id, hp.contact_id);
        ig.meta().create<IndexT>(builtin::subscene_element_id, hp.subscene_id);
        object->geometries().create(ig);
    }

    IndexT mesh_index = 0;
    for(auto& m : desc.meshes)
    {
        auto complex = trimesh(span<const Vector3>{m.positions},
                               span<const Vector3i>{m.triangles});
        label_surface(complex);

        auto verts        = complex.vertices();
        auto dimension    = verts.create<IndexT>(attr_dimension, 0);
        auto contact_id   = verts.create<IndexT>(attr_contact_id, 0);
        auto subscene_id  = verts.create<IndexT>(attr_subscene_id, 0);
        auto d_hat        = verts.create<Float>(attr_d_hat, 0.0);
        auto thickness    = verts.create<Float>(attr_thickness, 0.0);
        auto displacement = verts.create<Vector3>(attr_displacement, Vector3::Zero());

        auto assign = [&](auto& attr, const auto& values, auto fallback)
        {
            auto v = view(*attr);
            for(size_t i = 0; i < v.size(); ++i)
                v[i] = i < values.size() ? values[i] : fallback;
        };
        assign(dimension, m.dimensions, IndexT{2});
        assign(contact_id, m.contact_ids, IndexT{0});
        assign(subscene_id, m.subscene_ids, IndexT{0});
        assign(d_hat, m.d_hats, Float{0.0});
        assign(thickness, m.thicknesses, Float{0.0});
        assign(displacement, m.displacements, Vector3::Zero());

        complex.meta().create<IndexT>(attr_self_collision, m.self_collision);

        complex.meta().create<IndexT>("test_t72b_variant", desc.variant_uid);

        auto object = scene.objects().create("t72b_mesh_" + std::to_string(mesh_index++));
        object->geometries().create(complex);
    }

    return scene;
}

// ===========================================================================
//  The probe (the sibling tests' pattern): the managers + the filter manager
// ===========================================================================
class BvhFilterTestProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    static gg::GlobalVertexManager*            vertex_manager;
    static gg::GlobalBodyManager*              body_manager;
    static gg::GlobalSimplicialSurfaceManager* surface_manager;
    static gg::GlobalContactManager*           contact_manager;
    static gg::GlobalTrajectoryFilter*         trajectory_filter;
    static gg::HalfPlaneVertexReporter*        half_plane_reporter;

  protected:
    void do_build() override
    {
        vertex_manager    = find<gg::GlobalVertexManager>();
        body_manager      = find<gg::GlobalBodyManager>();
        surface_manager   = find<gg::GlobalSimplicialSurfaceManager>();
        contact_manager   = find<gg::GlobalContactManager>();
        trajectory_filter = find<gg::GlobalTrajectoryFilter>();
        half_plane_reporter = find<gg::HalfPlaneVertexReporter>();
    }
};
REGISTER_SIM_SYSTEM(BvhFilterTestProbe);

gg::GlobalVertexManager*            BvhFilterTestProbe::vertex_manager    = nullptr;
gg::GlobalBodyManager*              BvhFilterTestProbe::body_manager      = nullptr;
gg::GlobalSimplicialSurfaceManager* BvhFilterTestProbe::surface_manager   = nullptr;
gg::GlobalContactManager*           BvhFilterTestProbe::contact_manager   = nullptr;
gg::GlobalTrajectoryFilter*         BvhFilterTestProbe::trajectory_filter = nullptr;
gg::HalfPlaneVertexReporter*        BvhFilterTestProbe::half_plane_reporter = nullptr;

/// Everything the oracles need, read back from the ported managers after `init()`.
SceneData gather()
{
    auto* vm = BvhFilterTestProbe::vertex_manager;
    auto* sm = BvhFilterTestProbe::surface_manager;
    auto* cm = BvhFilterTestProbe::contact_manager;

    SceneData d;
    d.surf_vertices       = read_device(sm->surf_vertices());
    d.codim_vertices      = read_device(sm->codim_vertices());
    d.surf_edges          = read_vector2i(sm->surf_edges());
    d.surf_triangles      = read_vector3i(sm->surf_triangles());
    d.positions           = read_vector3(vm->positions());
    d.rest_positions      = read_vector3(vm->rest_positions());
    d.displacements       = read_vector3(vm->displacements());
    d.thicknesses         = read_device(vm->thicknesses());
    d.d_hats              = read_device(vm->d_hats());
    d.dimensions          = read_device(vm->dimensions());
    d.contact_ids         = read_device(vm->contact_element_ids());
    d.subscene_ids        = read_device(vm->subscene_element_ids());
    d.v2b                 = read_device(vm->body_ids());
    d.body_self_collision = read_device(BvhFilterTestProbe::body_manager->self_collision());
    d.contact_mask        = read_device(cm->contact_mask_tabular());
    d.contact_mask_size   = cm->contact_mask_size();
    d.subscene_mask       = read_device(cm->subscene_mask_tabular());
    d.subscene_mask_size  = cm->subscene_mask_size();
    return d;
}

/// The test plays the (Phase-8) fan-out that fills the global displacement array.
void set_displacements(const std::vector<Vector3>& per_vertex)
{
    auto* vm   = BvhFilterTestProbe::vertex_manager;
    auto  view = vm->displacements();
    REQUIRE(view.size() == per_vertex.size());
    std::vector<gg::LVector3> host;
    host.reserve(per_vertex.size());
    for(auto& v : per_vertex)
        host.push_back(gg::to_luisa(v));
    view.copy_from(span<const gg::LVector3>{host}, &gg::frame_stream());
    gg::frame_stream().flush();
}

/// Write the whole position array (the test's way of building degenerate layouts
/// without touching the scene).
void set_positions(const std::vector<Vector3>& per_vertex)
{
    auto* vm   = BvhFilterTestProbe::vertex_manager;
    auto  view = vm->positions();
    REQUIRE(view.size() == per_vertex.size());
    std::vector<gg::LVector3> host;
    host.reserve(per_vertex.size());
    for(auto& v : per_vertex)
        host.push_back(gg::to_luisa(v));
    view.copy_from(span<const gg::LVector3>{host}, &gg::frame_stream());
    gg::frame_stream().flush();
}

/// Restore the positions the managers reported at init (a fresh `gather()`).
void restore_positions(const SceneData& d)
{
    set_positions(d.positions);
}

void set_contact_ids(const std::vector<IndexT>& per_vertex)
{
    auto* vm   = BvhFilterTestProbe::vertex_manager;
    auto  view = vm->contact_element_ids();
    REQUIRE(view.size() == per_vertex.size());
    view.copy_from(span<const IndexT>{per_vertex}, &gg::frame_stream());
    gg::frame_stream().flush();
}

void set_subscene_ids(const std::vector<IndexT>& per_vertex)
{
    auto* vm   = BvhFilterTestProbe::vertex_manager;
    auto  view = vm->subscene_element_ids();
    REQUIRE(view.size() == per_vertex.size());
    view.copy_from(span<const IndexT>{per_vertex}, &gg::frame_stream());
    gg::frame_stream().flush();
}

/// The base's `Impl` wired to the probe's managers (the reference's own layout: the
/// derived filters' `detect` receives the *base's* `DetectInfo`).
gg::SimplexTrajectoryFilter::Impl make_simplex_impl(Float toi_safety_margin = 0.1)
{
    gg::SimplexTrajectoryFilter::Impl impl;
    impl.global_vertex_manager             = BvhFilterTestProbe::vertex_manager;
    impl.global_simplicial_surface_manager = BvhFilterTestProbe::surface_manager;
    impl.global_contact_manager            = BvhFilterTestProbe::contact_manager;
    impl.global_body_manager               = BvhFilterTestProbe::body_manager;
    impl.toi_safety_margin                 = toi_safety_margin;
    return impl;
}

// ===========================================================================
//  The white-box route: `Impl::detect` at alpha == 0 (the DCD pass)
// ===========================================================================
struct ImplDetectResult
{
    PairSet raw_allp_codimp;
    PairSet raw_codimp_alle;
    PairSet raw_alle_alle;
    PairSet raw_allp_allt;

    PairSet cand_allp_codimp;
    PairSet cand_codimp_alle;
    PairSet cand_alle_alle;
    PairSet cand_allp_allt;

    // the info variants' per-primitive body/contact ids (empty for the plain ones)
    std::vector<IndexT> codim_point_bids, codim_point_cids;
    std::vector<IndexT> point_bids, point_cids;
    std::vector<IndexT> edge_bids, edge_cids;
    std::vector<IndexT> triangle_bids, triangle_cids;

    [[nodiscard]] size_t total_candidates() const
    {
        return cand_allp_codimp.size() + cand_codimp_alle.size() + cand_alle_alle.size()
               + cand_allp_allt.size();
    }
};

/// Drive one variant's `Impl::detect` with the base `DetectInfo` the test can build
/// (`alpha == 0`) and read everything it produced.
template <typename FilterT>
ImplDetectResult impl_detect(gg::SimplexTrajectoryFilter::Impl& base_impl)
{
    // NOTE (measured): destroying a white-box-constructed variant `Impl` (whose BVHs own
    // device buffers created outside a scene's build) crashes the process at the end of
    // its scope in this port, while the same `Impl` owned by a registered filter is torn
    // down cleanly; the test therefore leaks the probe instances (a test-only workaround,
    // reported as a finding -- see the test header).
    using ImplT = typename FilterT::Impl;
    auto* impl  = new ImplT();

    // the v1 filter's `do_build` sizes these two four-slot count buffers; a
    // default-constructed `Impl` would hand LC a zero-length view (the T7.1a rule 9
    // hazard), so the white-box route does what `do_build` does
    if constexpr(requires { impl->query_counts; })
    {
        impl->query_counts.resize(4u);
        impl->selected_counts.resize(4u);
    }

    gg::SimplexTrajectoryFilter::DetectInfo info{&base_impl};
    REQUIRE(info.alpha() == 0.0);  // the port's only settable alpha on this route (DCD)

    impl->detect(info);
    gg::frame_stream().synchronize();

    ImplDetectResult r;
    r.raw_allp_codimp  = to_pair_set(read_vector2i(impl->raw_AllP_CodimP_pairs.view()));
    r.raw_codimp_alle  = to_pair_set(read_vector2i(impl->raw_CodimP_AllE_pairs.view()));
    r.raw_alle_alle    = to_pair_set(read_vector2i(impl->raw_AllE_AllE_pairs.view()));
    r.raw_allp_allt    = to_pair_set(read_vector2i(impl->raw_AllP_AllT_pairs.view()));
    r.cand_allp_codimp = to_pair_set(read_vector2i(impl->candidate_AllP_CodimP_pairs.view()));
    r.cand_codimp_alle = to_pair_set(read_vector2i(impl->candidate_CodimP_AllE_pairs.view()));
    r.cand_alle_alle   = to_pair_set(read_vector2i(impl->candidate_AllE_AllE_pairs.view()));
    r.cand_allp_allt   = to_pair_set(read_vector2i(impl->candidate_AllP_AllT_pairs.view()));

    if constexpr(requires { impl->point_bids; })
    {
        r.codim_point_bids = read_device(impl->codim_point_bids.cview());
        r.codim_point_cids = read_device(impl->codim_point_cids.cview());
        r.point_bids       = read_device(impl->point_bids.cview());
        r.point_cids       = read_device(impl->point_cids.cview());
        r.edge_bids        = read_device(impl->edge_bids.cview());
        r.edge_cids        = read_device(impl->edge_cids.cview());
        r.triangle_bids    = read_device(impl->triangle_bids.cview());
        r.triangle_cids    = read_device(impl->triangle_cids.cview());
    }
    return r;
}

// ===========================================================================
//  The pipeline route: `run_stage` + the reference's four exports
// ===========================================================================
struct PipelineResult
{
    std::vector<Vector2i> pt_candidates;
    std::vector<Vector2i> ee_candidates;
    std::vector<Vector4i> PTs;
    std::vector<Vector4i> EEs;
    std::vector<Vector3i> PEs;
    std::vector<Vector2i> PPs;
    std::vector<Float>    toi_pt;
    std::vector<Float>    toi_ee;
    Float                 step = 1.0;

    [[nodiscard]] size_t toi_total() const { return toi_pt.size() + toi_ee.size(); }
};

/// Run detect(alpha) -> filter_active -> filter_toi(alpha) on the scene's registered
/// filter and read its four exports + its active sets (this is the only route with a
/// non-zero alpha; `DetectInfo::m_alpha` is private to the filter base).
template <typename FilterT>
PipelineResult run_pipeline(Float alpha)
{
    auto* gtf = BvhFilterTestProbe::trajectory_filter;
    REQUIRE(gtf != nullptr);
    auto slot = gtf->find<FilterT>();
    REQUIRE(static_cast<bool>(slot));
    auto* filter = slot.view();

    PipelineResult r;

    CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::Detect, alpha) == 1.0);
    CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive) == 1.0);
    r.step = gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, alpha);

    r.pt_candidates = read_vector2i(filter->candidate_PTs());
    r.ee_candidates = read_vector2i(filter->candidate_EEs());
    r.toi_pt        = read_device(filter->toi_PTs());
    r.toi_ee        = read_device(filter->toi_EEs());
    r.PTs           = read_vector4i(filter->PTs());
    r.EEs           = read_vector4i(filter->EEs());
    r.PEs           = read_vector3i(filter->PEs());
    r.PPs           = read_vector2i(filter->PPs());
    return r;
}

// ===========================================================================
//  The host `filter_active` oracle: the reference's four classifier kernels
//  (`filter_active_k1..k4`) with the degeneracy classification, as *multisets* of
//  index tuples (the reference does not deduplicate, and the device's order is the
//  compaction's, so multiplicities are the comparable quantity).
// ===========================================================================
using IndexTuple4 = std::array<IndexT, 4>;
using IndexMultiset = std::map<IndexTuple4, size_t>;

IndexTuple4 tuple2(IndexT a, IndexT b) { return {a, b, -1, -1}; }
IndexTuple4 tuple3(IndexT a, IndexT b, IndexT c) { return {a, b, c, -1}; }
IndexTuple4 tuple4(IndexT a, IndexT b, IndexT c, IndexT d) { return {a, b, c, d}; }

struct ActiveOracle
{
    /// the reference's `UIPC_KERNEL_ASSERT(D > range.x())` invariant violations the host
    /// oracle saw (0 for every scene of this file: a mis-designed scene fails loudly)
    int violations = 0;

    IndexMultiset PPs;
    IndexMultiset PEs;
    IndexMultiset PTs;
    IndexMultiset EEs;

    [[nodiscard]] size_t total() const
    {
        return total(PPs) + total(PEs) + total(PTs) + total(EEs);
    }
    static size_t total(const IndexMultiset& m)
    {
        size_t n = 0;
        for(auto& [k, v] : m)
            n += v;
        return n;
    }
    void insert(IndexMultiset& m, const IndexTuple4& t) { ++m[t]; }
};

/// `D_range` / `is_active_D` spelled out (the ported header's own formula).
std::array<Float, 2> host_D_range(Float xi, Float d_hat)
{
    auto upper = xi + d_hat;
    auto lower = xi;
    return {lower * lower, upper * upper};
}

bool host_is_active_D(const std::array<Float, 2>& r, Float D)
{
    return r[0] < D && D < r[1];
}

/// The reference's `distance::degenerate_*(flag, offsets)` rule, spelled out on the
/// host: `dim` = the number of active flags, `offsets(k)` = the k-th active index in
/// *ascending* order (and `n` -- the "none" sentinel -- for the unused slots), with the
/// reference's `pe_from_ee` permutation applied for the edge-edge flavour's `dim == 3`.
///
/// This is deliberately *not* a call into the ported `active_offsets` gather: the oracle
/// must be an independent re-implementation, and the reference's rule is a one-line rank
/// order (`offsets(k)` = the k-th active flag's index).
int host_degenerate(int                       n,
                    const std::array<int, 4>& flag,
                    std::array<int, 4>&       offsets,
                    bool                      ee_permute = false)
{
    int dim = 0;
    for(int i = 0; i < n; ++i)
        dim += (flag[static_cast<size_t>(i)] != 0) ? 1 : 0;

    int asc[4] = {0, 0, 0, 0};
    int k      = 0;
    for(int i = 0; i < n; ++i)
        if(flag[static_cast<size_t>(i)] != 0)
            asc[k++] = i;

    for(int i = 0; i < 4; ++i)
        offsets[static_cast<size_t>(i)] = (i < dim) ? asc[i] : n;

    if(ee_permute && dim == 3)
    {
        // the reference's `pe_from_ee`, indexed by the *inactive* index.  Its entries are
        // **direct** indices into the four vertices `{ea0, ea1, eb0, eb1}` -- the
        // reference does `Eigen::Vector<T,3> P[] = {ea0,ea1,eb0,eb1};
        // Vector3i offsets = pe_from_ee(flag); P[offsets[j]]` -- and *not* ranks into the
        // ascending layout.  Re-gathering them through `asc` (which this oracle used to
        // do, mirroring the same bug in the ported `degenerate_edge_edge`) permuted three
        // of the four `dim == 3` flags: active {0,1,3} (inactive 2, row {3,0,1}) came out
        // as {2,0,1}, i.e. the point became eb0 instead of the free endpoint eb1.  Every
        // entry is in 0..3, so no out-of-range read is involved.
        static constexpr int pe_from_ee[4][3] = {{1, 2, 3}, {0, 2, 3}, {3, 0, 1}, {2, 0, 1}};
        for(int kk = 0; kk < 4; ++kk)
        {
            if(flag[static_cast<size_t>(kk)] != 0)
                continue;
            for(int j = 0; j < 3; ++j)
                offsets[static_cast<size_t>(j)] = pe_from_ee[kk][j];
        }
    }
    else if(ee_permute)
    {
        for(int i = 0; i < 4; ++i)
            offsets[static_cast<size_t>(i)] = asc[i];
    }
    return dim;
}

ActiveOracle host_active(const SceneData& d, const OracleSets& cand)
{
    ActiveOracle out;

    // AllP-CodimP -> PP (k1)
    for(auto& [i, j] : cand.allp_codimp)
    {
        const auto P0 = d.surf_vertices[i];
        const auto P1 = d.codim_vertices[j];

        const auto thickness = d.thicknesses[P0] + d.thicknesses[P1];        // PP_thickness
        const auto d_hat     = (d.d_hats[P0] + d.d_hats[P1]) / 2.0;          // PP_d_hat
        const auto range     = host_D_range(thickness, d_hat);
        const auto D         = (d.positions[P0] - d.positions[P1]).squaredNorm();
        out.violations += (D > range[0]) ? 0 : 1;  // the reference's device assert
        if(host_is_active_D(range, D))
            out.insert(out.PPs, tuple2(P0, P1));
    }

    // CodimP-AllE -> PP / PE (k2)
    for(auto& [i, j] : cand.codimp_alle)
    {
        const auto V = d.codim_vertices[i];
        const auto E = d.surf_edges[j];

        const IndexT  vIs[3] = {V, E[0], E[1]};
        const Vector3 x[3]   = {d.positions[vIs[0]], d.positions[vIs[1]], d.positions[vIs[2]]};

        const auto thickness = d.thicknesses[V] + d.thicknesses[E[0]];
        const auto d_hat     = (d.d_hats[V] + d.d_hats[E[0]]) / 2.0;
        const auto range     = host_D_range(thickness, d_hat);

        const auto flag = ldist::point_edge_distance_flag<Float>(gg::to_luisa(x[0]),
                                                                 gg::to_luisa(x[1]),
                                                                 gg::to_luisa(x[2]));
        Float      D{};
        ldist::point_edge_distance2(flag,
                                    gg::to_luisa(x[0]),
                                    gg::to_luisa(x[1]),
                                    gg::to_luisa(x[2]),
                                    D);
        out.violations += (D > range[0]) ? 0 : 1;
        if(!host_is_active_D(range, D))
            continue;

        std::array<int, 4> h_flag{};
        std::array<int, 4> h_offsets{};
        for(int k = 0; k < 3; ++k)
            h_flag[static_cast<size_t>(k)] = static_cast<int>(flag(k));
        const auto dim = host_degenerate(3, h_flag, h_offsets);
        if(dim == 2)
            out.insert(out.PPs, tuple2(vIs[h_offsets[0]], vIs[h_offsets[1]]));
        else if(dim == 3)
            out.insert(out.PEs, tuple3(vIs[0], vIs[1], vIs[2]));
        else
            CHECK((dim == 2 || dim == 3));
    }

    // AllP-AllT -> PP / PE / PT (k3)
    for(auto& [i, j] : cand.allp_allt)
    {
        const auto V = d.surf_vertices[i];
        const auto F = d.surf_triangles[j];

        const IndexT  vIs[4] = {V, F[0], F[1], F[2]};
        const Vector3 x[4]   = {d.positions[vIs[0]],
                                d.positions[vIs[1]],
                                d.positions[vIs[2]],
                                d.positions[vIs[3]]};

        const auto thickness = d.thicknesses[V] + d.thicknesses[F[0]];
        const auto d_hat     = (d.d_hats[V] + d.d_hats[F[0]]) / 2.0;
        const auto range     = host_D_range(thickness, d_hat);

        const auto flag = ldist::point_triangle_distance_flag<Float>(gg::to_luisa(x[0]),
                                                                    gg::to_luisa(x[1]),
                                                                    gg::to_luisa(x[2]),
                                                                    gg::to_luisa(x[3]));
        Float      D{};
        ldist::point_triangle_distance2(flag,
                                        gg::to_luisa(x[0]),
                                        gg::to_luisa(x[1]),
                                        gg::to_luisa(x[2]),
                                        gg::to_luisa(x[3]),
                                        D);
        out.violations += (D > 0.0) ? 0 : 1;
        out.violations += (D > range[0]) ? 0 : 1;
        if(!host_is_active_D(range, D))
            continue;

        std::array<int, 4> h_flag{};
        std::array<int, 4> h_offsets{};
        for(int k = 0; k < 4; ++k)
            h_flag[static_cast<size_t>(k)] = static_cast<int>(flag(k));
        const auto dim = host_degenerate(4, h_flag, h_offsets);
        if(dim == 2)
            out.insert(out.PPs, tuple2(vIs[h_offsets[0]], vIs[h_offsets[1]]));
        else if(dim == 3)
            out.insert(out.PEs,
                       tuple3(vIs[h_offsets[0]], vIs[h_offsets[1]], vIs[h_offsets[2]]));
        else if(dim == 4)
            out.insert(out.PTs, tuple4(vIs[0], vIs[1], vIs[2], vIs[3]));
        else
            CHECK((dim == 2 || dim == 3 || dim == 4));
    }

    // AllE-AllE -> PP / PE / EE (k4, with the mollifier escape)
    for(auto& [i, j] : cand.alle_alle)
    {
        const auto E0 = d.surf_edges[i];
        const auto E1 = d.surf_edges[j];

        const IndexT  vIs[4] = {E0[0], E0[1], E1[0], E1[1]};
        const Vector3 x[4]   = {d.positions[vIs[0]],
                                d.positions[vIs[1]],
                                d.positions[vIs[2]],
                                d.positions[vIs[3]]};

        const auto thickness = d.thicknesses[E0[0]] + d.thicknesses[E1[0]];
        const auto d_hat     = (d.d_hats[E0[0]] + d.d_hats[E1[0]]) / 2.0;
        const auto range     = host_D_range(thickness, d_hat);

        const auto flag = ldist::edge_edge_distance_flag<Float>(gg::to_luisa(x[0]),
                                                                gg::to_luisa(x[1]),
                                                                gg::to_luisa(x[2]),
                                                                gg::to_luisa(x[3]));
        Float      D{};
        ldist::edge_edge_distance2(flag,
                                   gg::to_luisa(x[0]),
                                   gg::to_luisa(x[1]),
                                   gg::to_luisa(x[2]),
                                   gg::to_luisa(x[3]),
                                   D);
        out.violations += (D > range[0]) ? 0 : 1;
        if(!host_is_active_D(range, D))
            continue;

        Float eps_x{};
        ldist::edge_edge_mollifier_threshold(gg::to_luisa(d.rest_positions[vIs[0]]),
                                             gg::to_luisa(d.rest_positions[vIs[1]]),
                                             gg::to_luisa(d.rest_positions[vIs[2]]),
                                             gg::to_luisa(d.rest_positions[vIs[3]]),
                                             1e-3,
                                             eps_x);

        if(ldist::need_mollify<Float>(gg::to_luisa(x[0]),
                                      gg::to_luisa(x[1]),
                                      gg::to_luisa(x[2]),
                                      gg::to_luisa(x[3]),
                                      eps_x))
        {
            out.insert(out.EEs, tuple4(vIs[0], vIs[1], vIs[2], vIs[3]));
            continue;
        }

          std::array<int, 4> h_flag{};
          std::array<int, 4> h_offsets{};
          for(int k = 0; k < 4; ++k)
              h_flag[static_cast<size_t>(k)] = static_cast<int>(flag(k));
          const auto dim = host_degenerate(4, h_flag, h_offsets, /*ee_permute*/ true);
          if(dim == 2)
            out.insert(out.PPs, tuple2(vIs[h_offsets[0]], vIs[h_offsets[1]]));
        else if(dim == 3)
            out.insert(out.PEs,
                       tuple3(vIs[h_offsets[0]], vIs[h_offsets[1]], vIs[h_offsets[2]]));
        else if(dim == 4)
            out.insert(out.EEs, tuple4(vIs[0], vIs[1], vIs[2], vIs[3]));
        else
            CHECK((dim == 2 || dim == 3 || dim == 4));
    }

    return out;
}

// ===========================================================================
//  The host TOI oracle: the reference's four `filter_toi` kernels over the ported
//  CCD (`*_ccd`), with the reference's `max_iter` budget told apart from the port's
//  convergence-sized unroll bound.
// ===========================================================================
struct ToiOracle
{
    std::map<std::pair<IndexT, IndexT>, Float> pt;  // keyed by the AllP-AllT pair
    std::map<std::pair<IndexT, IndexT>, Float> ee;  // keyed by the AllE-AllE pair
    Float                                      min_all = static_cast<Float>(1.1);
    size_t                                     n       = 0;

    void note(Float toi)
    {
        ++n;
        min_all = std::min(min_all, toi);
    }
};

/// `if(faraway) toi = 1.1; else { hit = *_ccd(...); if(!hit) toi = 1.1; }`
Float toi_from_ccd(bool faraway, bool hit, Float toi)
{
    if(faraway)
        return static_cast<Float>(1.1);
    if(!hit)
        return static_cast<Float>(1.1);
    return toi;
}

ToiOracle host_toi(const SceneData&  d,
                   const OracleSets& cand,
                   Float             alpha,
                   Float             eta,
                   int               max_iter)
{
    ToiOracle out;

    // AllP-CodimP -> point-point (k1)
    for(auto& [i, j] : cand.allp_codimp)
    {
        const auto V0 = d.surf_vertices[i];
        const auto V1 = d.codim_vertices[j];

        const Float thickness = d.thicknesses[V0] + d.thicknesses[V1];
        const Float d_hat     = (d.d_hats[V0] + d.d_hats[V1]) / 2.0;

        const Vector3 VP0  = d.positions[V0];
        const Vector3 VP1  = d.positions[V1];
        const Vector3 dVP0 = alpha * d.displacements[V0];
        const Vector3 dVP1 = alpha * d.displacements[V1];

        const bool faraway = !host_pp_broadphase(VP0, VP1, dVP0, dVP1, d_hat + thickness);
        Float      toi     = static_cast<Float>(1.1);
        bool       hit     = false;
        if(!faraway)
        {
            hit = ldist::point_point_ccd<Float>(gg::to_luisa(VP0),
                                                gg::to_luisa(VP1),
                                                gg::to_luisa(dVP0),
                                                gg::to_luisa(dVP1),
                                                eta,
                                                thickness,
                                                max_iter,
                                                toi);
        }
        out.note(toi = toi_from_ccd(faraway, hit, toi));
    }

    // CodimP-AllE -> point-edge (k2)
    for(auto& [i, j] : cand.codimp_alle)
    {
        const auto V = d.codim_vertices[i];
        const auto E = d.surf_edges[j];

        const Float thickness = d.thicknesses[V] + d.thicknesses[E[0]];
        const Float d_hat     = (d.d_hats[V] + d.d_hats[E[0]]) / 2.0;

        const Vector3 VP  = d.positions[V];
        const Vector3 dVP = alpha * d.displacements[V];
        const Vector3 EP0 = d.positions[E[0]];
        const Vector3 EP1 = d.positions[E[1]];
        const Vector3 dEP0 = alpha * d.displacements[E[0]];
        const Vector3 dEP1 = alpha * d.displacements[E[1]];

        const bool faraway =
            !host_pe_broadphase(VP, EP0, EP1, dVP, dEP0, dEP1, d_hat + thickness);
        Float toi = static_cast<Float>(1.1);
        bool  hit = false;
        if(!faraway)
        {
            hit = ldist::point_edge_ccd<Float>(gg::to_luisa(VP),
                                               gg::to_luisa(EP0),
                                               gg::to_luisa(EP1),
                                               gg::to_luisa(dVP),
                                               gg::to_luisa(dEP0),
                                               gg::to_luisa(dEP1),
                                               eta,
                                               thickness,
                                               max_iter,
                                               toi);
        }
        out.note(toi = toi_from_ccd(faraway, hit, toi));
    }

    // AllP-AllT -> point-triangle (k3)
    for(auto& [i, j] : cand.allp_allt)
    {
        const auto V = d.surf_vertices[i];
        const auto F = d.surf_triangles[j];

        const Float thickness = d.thicknesses[V] + d.thicknesses[F[0]];
        const Float d_hat     = (d.d_hats[V] + d.d_hats[F[0]]) / 2.0;

        const Vector3 VP  = d.positions[V];
        const Vector3 dVP = alpha * d.displacements[V];
        const Vector3 FP0 = d.positions[F[0]];
        const Vector3 FP1 = d.positions[F[1]];
        const Vector3 FP2 = d.positions[F[2]];
        const Vector3 dFP0 = alpha * d.displacements[F[0]];
        const Vector3 dFP1 = alpha * d.displacements[F[1]];
        const Vector3 dFP2 = alpha * d.displacements[F[2]];

        const bool faraway = !host_pt_broadphase(VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2,
                                                 d_hat + thickness);
        Float toi = static_cast<Float>(1.1);
        bool  hit = false;
        if(!faraway)
        {
            hit = ldist::point_triangle_ccd<Float>(gg::to_luisa(VP),
                                                   gg::to_luisa(FP0),
                                                   gg::to_luisa(FP1),
                                                   gg::to_luisa(FP2),
                                                   gg::to_luisa(dVP),
                                                   gg::to_luisa(dFP0),
                                                   gg::to_luisa(dFP1),
                                                   gg::to_luisa(dFP2),
                                                   eta,
                                                   thickness,
                                                   max_iter,
                                                   toi);
        }
        toi          = toi_from_ccd(faraway, hit, toi);
        out.pt[{i, j}] = toi;
        out.note(toi);
    }

    // AllE-AllE -> edge-edge (k4)
    for(auto& [i, j] : cand.alle_alle)
    {
        const auto E0 = d.surf_edges[i];
        const auto E1 = d.surf_edges[j];

        const Float thickness = d.thicknesses[E0[0]] + d.thicknesses[E1[0]];
        const Float d_hat     = (d.d_hats[E0[0]] + d.d_hats[E1[0]]) / 2.0;

        const Vector3 EP0 = d.positions[E0[0]];
        const Vector3 EP1 = d.positions[E0[1]];
        const Vector3 EP2 = d.positions[E1[0]];
        const Vector3 EP3 = d.positions[E1[1]];
        const Vector3 dEP0 = alpha * d.displacements[E0[0]];
        const Vector3 dEP1 = alpha * d.displacements[E0[1]];
        const Vector3 dEP2 = alpha * d.displacements[E1[0]];
        const Vector3 dEP3 = alpha * d.displacements[E1[1]];

        const bool faraway = !host_ee_broadphase(EP0, EP1, EP2, EP3, dEP0, dEP1, dEP2,
                                                 dEP3, d_hat + thickness);
        Float toi = static_cast<Float>(1.1);
        bool  hit = false;
        if(!faraway)
        {
            hit = ldist::edge_edge_ccd<Float>(gg::to_luisa(EP0),
                                              gg::to_luisa(EP1),
                                              gg::to_luisa(EP2),
                                              gg::to_luisa(EP3),
                                              gg::to_luisa(dEP0),
                                              gg::to_luisa(dEP1),
                                              gg::to_luisa(dEP2),
                                              gg::to_luisa(dEP3),
                                              eta,
                                              thickness,
                                              max_iter,
                                              toi);
        }
        toi            = toi_from_ccd(faraway, hit, toi);
        out.ee[{i, j}] = toi;
        out.note(toi);
    }

    return out;
}

// ===========================================================================
//  multiset read-back of the device's active sets (the four `Impl` outputs)
// ===========================================================================
IndexMultiset to_multiset2(const std::vector<Vector2i>& v)
{
    IndexMultiset m;
    for(auto& p : v)
        ++m[tuple2(p[0], p[1])];
    return m;
}

IndexMultiset to_multiset3(const std::vector<Vector3i>& v)
{
    IndexMultiset m;
    for(auto& p : v)
        ++m[tuple3(p[0], p[1], p[2])];
    return m;
}

IndexMultiset to_multiset4(const std::vector<Vector4i>& v)
{
    IndexMultiset m;
    for(auto& p : v)
        ++m[tuple4(p[0], p[1], p[2], p[3])];
    return m;
}

std::string multiset_to_string(const IndexMultiset& m)
{
    std::string out;
    for(auto& [k, c] : m)
    {
        out += "(";
        for(auto x : k)
            out += std::to_string(x) + ",";
        out += "x" + std::to_string(c) + ") ";
    }
    return out;
}

// ===========================================================================
//  The variant battery, shared by the four variant cases
// ===========================================================================
/// Print a `[bvh_filter]`-prefixed line (every quoted number comes from here).
template <typename... Args>
void note(const char* fmt, Args&&... args)
{
    std::printf("[bvh_filter] ");
    std::printf(fmt, std::forward<Args>(args)...);
    std::printf("\n");
}

/// The device AABB buffers vs the host oracle (a bit-exact float comparison: both
/// sides run the reference's own `float` arithmetic).
template <typename ImplT>
void check_aabbs(const ImplT& impl, const SceneData& d)
{
    auto codim = read_device(impl.codim_point_aabbs.cview());
    auto point = read_device(impl.point_aabbs.cview());
    auto edge  = read_device(impl.edge_aabbs.cview());
    auto tri   = read_device(impl.triangle_aabbs.cview());

    REQUIRE(codim.size() == d.codim_point_aabbs.size());
    REQUIRE(point.size() == d.point_aabbs.size());
    REQUIRE(edge.size() == d.edge_aabbs.size());
    REQUIRE(tri.size() == d.triangle_aabbs.size());

    int mismatches = 0;
    auto same_box  = [&](const gg::LAABB& dev, const FAABB& host)
    {
        auto box = gg::to_eigen(dev);
        return (box.min().array() == host.min().array()).all()
               && (box.max().array() == host.max().array()).all();
    };
    for(size_t i = 0; i < codim.size(); ++i)
        mismatches += same_box(codim[i], d.codim_point_aabbs[i]) ? 0 : 1;
    for(size_t i = 0; i < point.size(); ++i)
        mismatches += same_box(point[i], d.point_aabbs[i]) ? 0 : 1;
    for(size_t i = 0; i < edge.size(); ++i)
        mismatches += same_box(edge[i], d.edge_aabbs[i]) ? 0 : 1;
    for(size_t i = 0; i < tri.size(); ++i)
        mismatches += same_box(tri[i], d.triangle_aabbs[i]) ? 0 : 1;

    CHECK(mismatches == 0);
    note("AABB builders vs host: %zu+%zu+%zu+%zu boxes, %d mismatches",
         codim.size(),
         point.size(),
         edge.size(),
         tri.size(),
         mismatches);
}

/// The four variants' candidate production, active sets and TOI values vs the host
/// oracles, for one scene (the *registered* variant's pipeline at `alpha`, and the
/// white-box `Impl::detect` at the DCD `alpha == 0`).
template <typename PipelineFilterT, typename ImplOwnerT>
void check_variant_battery(SceneData& d, Float alpha, bool dcd, std::string_view variant)
{
    // ---- (a) the pipeline route (alpha != 0) ------------------------------
    {  // the gathered index sets (quoted by the summary; also a sanity probe)
        std::string sv, cv, se, st;
        for(auto v : d.surf_vertices)
            sv += std::to_string(v) + " ";
        for(auto v : d.codim_vertices)
            cv += std::to_string(v) + " ";
        for(auto& e : d.surf_edges)
            se += "(" + std::to_string(e[0]) + "," + std::to_string(e[1]) + ") ";
        for(auto& f : d.surf_triangles)
            st += "(" + std::to_string(f[0]) + "," + std::to_string(f[1]) + ","
                  + std::to_string(f[2]) + ") ";
        note("[%s] gathered: surf_vertices={%s} codim_vertices={%s} surf_edges={%s} surf_triangles={%s}",
             std::string{variant}.c_str(), sv.c_str(), cv.c_str(), se.c_str(), st.c_str());
    }
    d.build_aabbs(alpha);
    {
        const auto oracle = host_candidates(d, alpha, dcd);
        const auto p      = run_pipeline<PipelineFilterT>(alpha);

        const auto pt_cand = to_pair_set(p.pt_candidates);
        const auto ee_cand = to_pair_set(p.ee_candidates);

        note("[%s] alpha=%.3f candidates: PP+PE=(not exported) PT=%zu EE=%zu (oracle PP+PE=%zu PT=%zu EE=%zu)",
             std::string{variant}.c_str(),
             alpha,
             pt_cand.size(),
             ee_cand.size(),
             oracle.allp_codimp.size() + oracle.codimp_alle.size(),
             oracle.allp_allt.size(),
             oracle.alle_alle.size());
        if(pt_cand != oracle.allp_allt)
            note("[%s] PT candidate mismatch: device=%s oracle=%s",
                 std::string{variant}.c_str(),
                 set_to_string(pt_cand).c_str(),
                 set_to_string(oracle.allp_allt).c_str());
        if(ee_cand != oracle.alle_alle)
            note("[%s] EE candidate mismatch: device=%s oracle=%s",
                 std::string{variant}.c_str(),
                 set_to_string(ee_cand).c_str(),
                 set_to_string(oracle.alle_alle).c_str());

        CHECK(pt_cand == oracle.allp_allt);
        CHECK(ee_cand == oracle.alle_alle);

        // NOTE: the reference exports only the PT and EE toi *slices* of its tois
        // array (`toi_PTs()` starts at `pp + pe`), so the PP/PE candidate *counts* are
        // not observable through the pipeline; they are pinned by the white-box DCD
        // pass below (`cand_allp_codimp` / `cand_codimp_alle`) and by the active sets
        // (which derive from them).

        // ---- the active sets (the reference's filter_active output) ---------
        const auto active = host_active(d, oracle);
        note("[%s] active: PP=%zu PE=%zu PT=%zu EE=%zu (oracle %zu/%zu/%zu/%zu)",
             std::string{variant}.c_str(),
             p.PPs.size(),
             p.PEs.size(),
             p.PTs.size(),
             p.EEs.size(),
             ActiveOracle::total(active.PPs),
             ActiveOracle::total(active.PEs),
             ActiveOracle::total(active.PTs),
             ActiveOracle::total(active.EEs));

        if(to_multiset2(p.PPs) != active.PPs)
            note("[%s] PP multiset difference: device=%s oracle=%s",
                 std::string{variant}.c_str(),
                 multiset_to_string(to_multiset2(p.PPs)).c_str(),
                 multiset_to_string(active.PPs).c_str());
        if(to_multiset3(p.PEs) != active.PEs)
            note("[%s] PE multiset difference: device=%s oracle=%s",
                 std::string{variant}.c_str(),
                 multiset_to_string(to_multiset3(p.PEs)).c_str(),
                 multiset_to_string(active.PEs).c_str());
        if(to_multiset4(p.EEs) != active.EEs)
            note("[%s] EE multiset difference: device=%s oracle=%s",
                 std::string{variant}.c_str(),
                 multiset_to_string(to_multiset4(p.EEs)).c_str(),
                 multiset_to_string(active.EEs).c_str());
        if(to_multiset4(p.PTs) != active.PTs)
            note("[%s] PT multiset difference: device=%s oracle=%s",
                 std::string{variant}.c_str(),
                 multiset_to_string(to_multiset4(p.PTs)).c_str(),
                 multiset_to_string(active.PTs).c_str());
        CHECK(active.violations == 0);  // the reference's per-pair invariant
        CHECK(to_multiset2(p.PPs) == active.PPs);
        CHECK(to_multiset3(p.PEs) == active.PEs);
        CHECK(to_multiset4(p.PTs) == active.PTs);
        CHECK(to_multiset4(p.EEs) == active.EEs);

        // ---- the TOI path ---------------------------------------------------
        // the device kernels pass the port's convergence-sized bound (the port's CCD
        // unrolls once per iteration), not the reference's 1000
        const auto toi =
            host_toi(d, oracle, alpha, /*eta*/ Float{0.1}, gg::sftd::kToiMaxIter());
        REQUIRE(p.toi_pt.size() == oracle.allp_allt.size());
        REQUIRE(p.toi_ee.size() == oracle.alle_alle.size());
        // per *pair*, not per index: see `toi_by_pair`
        const auto got_pt = toi_by_pair(p.pt_candidates, p.toi_pt);
        const auto got_ee = toi_by_pair(p.ee_candidates, p.toi_ee);
        int    mismatches = 0;
        double max_diff   = 0.0;
        for(auto& pair : oracle.allp_allt)
        {
            const auto want = toi.pt.at(pair);
            const auto got  = got_pt.at(pair);
            max_diff = std::max(max_diff, std::abs(static_cast<double>(want - got)));
            mismatches += (std::abs(static_cast<double>(want - got)) < 1e-12) ? 0 : 1;
        }
        for(auto& pair : oracle.alle_alle)
        {
            const auto want = toi.ee.at(pair);
            const auto got  = got_ee.at(pair);
            max_diff = std::max(max_diff, std::abs(static_cast<double>(want - got)));
            mismatches += (std::abs(static_cast<double>(want - got)) < 1e-12) ? 0 : 1;
        }
        CHECK(mismatches == 0);

        // the manager's step fraction is the min over *all four* kinds (so this pins
        // the PP/PE tois too), clamped to 1
        const auto want_step = std::min(static_cast<Float>(1.0), toi.min_all);
        CHECK(std::abs(static_cast<double>(want_step - p.step)) < 1e-12);

        note("[%s] toi: min=%.17g step=%.17g (PT=%zu EE=%zu values, %d mismatches, "
             "max|d|=%.3g)",
             std::string{variant}.c_str(),
             static_cast<double>(toi.min_all),
             static_cast<double>(p.step),
             p.toi_pt.size(),
             p.toi_ee.size(),
             mismatches,
             max_diff);

        // the convergence-sized bound measurement: the ported CCD unrolls once per
        // iteration, so the kernels pass 16 instead of the reference's 1000 -- prove
        // the tested regime is unaffected
        {
            const auto toi_ref = host_toi(d, oracle, alpha, Float{0.1}, 1000);
            int        diff    = 0;
            for(auto& [pair, want] : toi_ref.pt)
                diff += (want == toi.pt.at(pair)) ? 0 : 1;
            for(auto& [pair, want] : toi_ref.ee)
                diff += (want == toi.ee.at(pair)) ? 0 : 1;
            CHECK(diff == 0);
            note("[%s] CCD max_iter 1000 vs %d: %d differing toi values",
             std::string{variant}.c_str(), gg::sftd::kToiMaxIter(), diff);
        }

        // at least one reported toi is the 1.1 sentinel (an active pair that does not
        // collide within the step), and every value is positive (the manager's assert)
        bool has_sentinel = false;
        bool all_positive = true;
        for(auto v : p.toi_pt)
        {
            has_sentinel |= (v == static_cast<Float>(1.1));
            all_positive &= (v > 0.0);
        }
        for(auto v : p.toi_ee)
        {
            has_sentinel |= (v == static_cast<Float>(1.1));
            all_positive &= (v > 0.0);
        }
          CHECK(has_sentinel);
        CHECK(all_positive);
      // the *production* filter of this variant, when the scene's
        // `collision_detection/method` selects it (`info_stackless_bvh` is the only
        // selectable one in this build, so only its production filter is registered next
        // to the test-local mirror).  The manager's stage loop drives *every* registered
        // filter, so the production filter's own `Impl` is filled by the same run --
        // reading its four exported slices through the *production* accessors pins the
        // offset arithmetic of those accessors themselves.  (The mirror above re-derives
        // the same slices, so without this block a wrong offset in e.g.
        // `LBVHSimplexTrajectoryFilter::toi_EEs()` is invisible: measured in T7.2b.)
        auto* gtf = BvhFilterTestProbe::trajectory_filter;
        REQUIRE(gtf != nullptr);
        if(auto prod_slot = gtf->find<ImplOwnerT>(); static_cast<bool>(prod_slot))
        {
            auto* prod = prod_slot.view();
            CHECK(read_vector2i(prod->candidate_PTs()) == p.pt_candidates);
            CHECK(read_vector2i(prod->candidate_EEs()) == p.ee_candidates);
            CHECK(read_device(prod->toi_PTs()) == p.toi_pt);
            CHECK(read_device(prod->toi_EEs()) == p.toi_ee);
        }
    }

    // ---- (a2) the *closing* probe: the same scene with a displacement large enough
    //      that pairs actually collide inside the step.  In the scene above every pair
    //      reports the `1.1` sentinel (the 0.15 gap never closes), so the TOI pins below
    //      would compare the sentinel against itself -- a slice-offset, an `eta` or a
    //      CCD-iteration perturbation would stay invisible.  This probe re-runs the whole
    //      pipeline on a copy whose upper plate drops 0.35 (through the lower one) and
    //      re-derives every oracle for it.
    {
        auto closing = d;
        for(size_t i = 0; i < closing.displacements.size(); ++i)
            if(is_upper_plate(closing.positions[i]))
                closing.displacements[i] = Vector3{0.0, -0.35, 0.0};
        closing.build_aabbs(alpha);

        const auto oracle = host_candidates(closing, alpha, dcd);
        set_displacements(closing.displacements);
        const auto p = run_pipeline<PipelineFilterT>(alpha);
        set_displacements(d.displacements);  // restore the scene's own displacement

        CHECK(to_pair_set(p.pt_candidates) == oracle.allp_allt);
        CHECK(to_pair_set(p.ee_candidates) == oracle.alle_alle);

        const auto active = host_active(closing, oracle);
        CHECK(active.violations == 0);
        CHECK(to_multiset2(p.PPs) == active.PPs);
        CHECK(to_multiset3(p.PEs) == active.PEs);
        CHECK(to_multiset4(p.PTs) == active.PTs);
        CHECK(to_multiset4(p.EEs) == active.EEs);

        const auto toi =
            host_toi(closing, oracle, alpha, /*eta*/ Float{0.1}, gg::sftd::kToiMaxIter());
        REQUIRE(p.toi_pt.size() == oracle.allp_allt.size());
        REQUIRE(p.toi_ee.size() == oracle.alle_alle.size());
        // per *pair*, not per index: see `toi_by_pair`
        const auto got_pt = toi_by_pair(p.pt_candidates, p.toi_pt);
        const auto got_ee = toi_by_pair(p.ee_candidates, p.toi_ee);
        int    mismatches   = 0;
        double max_diff     = 0.0;
        size_t non_sentinel = 0;
        for(auto& pair : oracle.allp_allt)
        {
            const auto want = toi.pt.at(pair);
            const auto got  = got_pt.at(pair);
            max_diff = std::max(max_diff, std::abs(static_cast<double>(want - got)));
            mismatches += (std::abs(static_cast<double>(want - got)) < 1e-12) ? 0 : 1;
            non_sentinel += (got == static_cast<Float>(1.1)) ? 0 : 1;
        }
        for(auto& pair : oracle.alle_alle)
        {
            const auto want = toi.ee.at(pair);
            const auto got  = got_ee.at(pair);
            max_diff = std::max(max_diff, std::abs(static_cast<double>(want - got)));
            mismatches += (std::abs(static_cast<double>(want - got)) < 1e-12) ? 0 : 1;
            non_sentinel += (got == static_cast<Float>(1.1)) ? 0 : 1;
        }
        const auto want_step = std::min(static_cast<Float>(1.0), toi.min_all);
        note("[%s] closing: PT=%zu EE=%zu toi=%zu (oracle %zu/%zu), min=%.17g step=%.17g, "
             "%d mismatches, max|d|=%.3g, %zu non-sentinel",
             std::string{variant}.c_str(),
             p.pt_candidates.size(),
             p.ee_candidates.size(),
             p.toi_pt.size() + p.toi_ee.size(),
             oracle.allp_allt.size(),
             oracle.alle_alle.size(),
             static_cast<double>(toi.min_all),
             static_cast<double>(p.step),
             mismatches,
             max_diff,
             non_sentinel);
        CHECK(mismatches == 0);
        CHECK(std::abs(static_cast<double>(want_step - p.step)) < 1e-12);
        // the probe is only worth its runtime if it *does* collide: some pair must hit
        // inside the step and the manager's step must be clamped below 1
        CHECK(non_sentinel > 0);
        CHECK(p.step < 1.0);
        CHECK(toi.min_all < 1.0);
    }

    // ---- (b) the white-box route: the DCD pass (alpha == 0), all four kinds --
    d.build_aabbs(0.0);
    {
        auto base = make_simplex_impl();
        auto r    = impl_detect<ImplOwnerT>(base);

        const auto raw_oracle = [&]
        {
            struct Raw
            {
                PairSet allp_codimp;
                PairSet codimp_alle;
                PairSet alle_alle;
                PairSet allp_allt;
            } raw;
            raw.allp_codimp = host_raw_allp_codimp(d);
            raw.codimp_alle = host_raw_codimp_alle(d);
            raw.alle_alle   = host_raw_alle_alle(d);
            raw.allp_allt   = host_raw_allp_allt(d);
            return raw;
        }();

        note("[%s] DCD(alpha=0) raw: PP=%zu PE=%zu EE=%zu PT=%zu",
             std::string{variant}.c_str(),
             r.raw_allp_codimp.size(),
             r.raw_codimp_alle.size(),
             r.raw_alle_alle.size(),
             r.raw_allp_allt.size());

        CHECK(r.raw_allp_codimp == raw_oracle.allp_codimp);
        CHECK(r.raw_codimp_alle == raw_oracle.codimp_alle);
        CHECK(r.raw_alle_alle == raw_oracle.alle_alle);
        CHECK(r.raw_allp_allt == raw_oracle.allp_allt);

        const auto oracle = host_candidates(d, 0.0, dcd);
        note("[%s] DCD(alpha=0) candidates: PP=%zu PE=%zu EE=%zu PT=%zu (oracle %zu/%zu/%zu/%zu)",
             std::string{variant}.c_str(),
             r.cand_allp_codimp.size(),
             r.cand_codimp_alle.size(),
             r.cand_alle_alle.size(),
             r.cand_allp_allt.size(),
             oracle.allp_codimp.size(),
             oracle.codimp_alle.size(),
             oracle.alle_alle.size(),
             oracle.allp_allt.size());

        CHECK(r.cand_allp_codimp == oracle.allp_codimp);
        CHECK(r.cand_codimp_alle == oracle.codimp_alle);
        CHECK(r.cand_alle_alle == oracle.alle_alle);
        CHECK(r.cand_allp_allt == oracle.allp_allt);
    }
}

// ===========================================================================
//  The test-local variant filters
//
//  This build's scene schema only lists `info_stackless_bvh` (the reference's
//  `conditionalValues` drops the other three methods for a backend without the CUDA
//  legacy collision build option), so the *other* three filters cannot be selected by
//  a frontend scene.  Their `detect`/`filter_active`/`filter_toi` are still driven
//  here by a test-local `SimplexTrajectoryFilter` subclass that owns the variant's
//  `Impl` -- the reference's own call shape, with the *base* building the
//  `DetectInfo`/`FilterTOIInfo` (their `alpha`/`toi` slots are private to
//  `SimplexTrajectoryFilter`, which is exactly why the filters cannot be driven
//  directly from the test).
//
//  Registration is gated on the scene's object-name marker, so these systems exist
//  only in this file's scenes (no other test's scene has a `t72b_*` object) and the
//  `[filter]` suite's expectations are untouched.
// ===========================================================================
template <typename ImplT>
class T72bVariantFilter : public gg::SimplexTrajectoryFilter
{
  public:
    using SimplexTrajectoryFilter::SimplexTrajectoryFilter;

    lud::LCBufferView<gg::LVector2i> candidate_PTs() const noexcept override
    {
        return m_impl->candidate_AllP_AllT_pairs.view();
    }
    lud::LCBufferView<gg::LVector2i> candidate_EEs() const noexcept override
    {
        return m_impl->candidate_AllE_AllE_pairs.view();
    }
    lud::LCBufferView<Float> toi_PTs() const noexcept override
    {
        auto pp = m_impl->candidate_AllP_CodimP_pairs.size();
        auto pe = m_impl->candidate_CodimP_AllE_pairs.size();
        auto pt = m_impl->candidate_AllP_AllT_pairs.size();
        return m_impl->tois.cview(pp + pe, pt);
    }
    lud::LCBufferView<Float> toi_EEs() const noexcept override
    {
        auto pp = m_impl->candidate_AllP_CodimP_pairs.size();
        auto pe = m_impl->candidate_CodimP_AllE_pairs.size();
        auto pt = m_impl->candidate_AllP_AllT_pairs.size();
        auto ee = m_impl->candidate_AllE_AllE_pairs.size();
        return m_impl->tois.cview(pp + pe + pt, ee);
    }

  protected:
    /// the `test_t72b_variant` meta uid this variant's scenes carry
    [[nodiscard]] virtual IndexT variant_uid() const noexcept = 0;

    void do_build(BuildInfo& info) override;
    void do_detect(DetectInfo& info) override { m_impl->detect(info); }
    void do_filter_active(FilterActiveInfo& info) override { m_impl->filter_active(info); }
    void do_filter_toi(FilterTOIInfo& info) override { m_impl->filter_toi(info); }

  private:
    /// NOTE (measured): *destroying* an `LBVHSimplexTrajectoryFilter::Impl` or an
    /// `InfoStacklessBVHV0SimplexTrajectoryFilter::Impl` (whose BVHs own device buffers
    /// built outside a scene's own build) crashes the process in this port, while the
    /// `StacklessBVH`/`InfoStacklessBVH` (v1) flavours tear down cleanly.  The test
    /// therefore leaks its probe `Impl`s (a test-only workaround; the two unreachable
    /// flavours are latent-only for the port's own scenes -- reported as a finding).
    ImplT* m_impl = new ImplT();
};

template <typename ImplT>
void T72bVariantFilter<ImplT>::do_build(BuildInfo& info)
{
    (void)info;
    // only this file's scenes carry a geometry with this variant's `test_t72b_variant`
    // meta uid; every other scene's build invalidates this system (the reference's own
    // `SimSystemException` gating shape)
    bool found = false;
    for(auto& slot : world().scene().geometries())
    {
        auto* sc = slot->geometry().as<SimplicialComplex>();
        if(sc == nullptr)
            continue;
        auto attr = sc->meta().find<IndexT>("test_t72b_variant");
        if(attr && view(*attr)[0] == variant_uid())
        {
            found = true;
            break;
        }
    }
    if(!found)
        throw uipc::backend::SimSystemException(
            "t72b test variant filter: not a t72b scene");
}

class T72bLbvhVariantFilter final
    : public T72bVariantFilter<gg::LBVHSimplexTrajectoryFilter::Impl>
{
  public:
    using T72bVariantFilter::T72bVariantFilter;

  protected:
    IndexT variant_uid() const noexcept override { return 1; }  // `linear_bvh`
};
class T72bStacklessVariantFilter final
    : public T72bVariantFilter<gg::StacklessBVHSimplexTrajectoryFilter::Impl>
{
  public:
    using T72bVariantFilter::T72bVariantFilter;

  protected:
    IndexT variant_uid() const noexcept override { return 2; }  // `stackless_bvh`
};
class T72bInfov0VariantFilter final
    : public T72bVariantFilter<gg::InfoStacklessBVHV0SimplexTrajectoryFilter::Impl>
{
  public:
    using T72bVariantFilter::T72bVariantFilter;

  protected:
    IndexT variant_uid() const noexcept override { return 3; }  // `info_stackless_bvh_v0`
};

REGISTER_SIM_SYSTEM(T72bLbvhVariantFilter);
REGISTER_SIM_SYSTEM(T72bStacklessVariantFilter);
REGISTER_SIM_SYSTEM(T72bInfov0VariantFilter);

// ===========================================================================
//  The registration gate (each scene selects one of the four methods; the engine's
//  `SimSystemException` handler invalidates the other three before they register)
// ===========================================================================
template <typename FilterT>
void check_registration(gg::GlobalTrajectoryFilter* gtf)
{
    // the pipeline filter is registered ...
    CHECK(static_cast<bool>(gtf->find<FilterT>()));

    // ... and the other three *scene* filters are not: this build's scene schema lists
    // only `info_stackless_bvh` (the reference's `conditionalValues` gates the other
    // three on the CUDA legacy-collision build option), so their `do_build` gate can
    // never pass (the reference's own gating, transcribed)
    CHECK(!static_cast<bool>(gtf->find<gg::LBVHSimplexTrajectoryFilter>()));
    CHECK(!static_cast<bool>(gtf->find<gg::StacklessBVHSimplexTrajectoryFilter>()));
    CHECK(!static_cast<bool>(gtf->find<gg::InfoStacklessBVHV0SimplexTrajectoryFilter>()));

    // the schema's default method is `info_stackless_bvh`, so its filter is
    // registered in *every* scene of this file (also in the scenes that drive another
    // variant through a test-local filter)
    CHECK(static_cast<bool>(gtf->find<gg::InfoStacklessBVHSimplexTrajectoryFilter>()));
}

/// The scene's global vertex offsets (the sibling tests' reporters report every
/// geometry in scene order, so geometry k owns vertices
/// `[offset_k, offset_k + its vertex count)`).
std::vector<IndexT> mesh_offsets(const SceneDesc& desc)
{
    std::vector<IndexT> out;
    // the half plane's reporter is the first vertex slice (its own `vertex_offset()`),
    // so the geometries' global vertex indices start after it
    auto*  hr      = BvhFilterTestProbe::half_plane_reporter;
    IndexT running = hr ? (hr->vertex_offset() + hr->vertex_count()) : 0;
    for(auto& m : desc.meshes)
    {
        out.push_back(running);
        running += static_cast<IndexT>(m.positions.size());
    }
    out.push_back(running);
    return out;
}

std::vector<Vector3> shifted(const std::vector<Vector3>& pos, IndexT first, IndexT last, const Vector3& d)
{
    auto out = pos;
    for(IndexT i = first; i < last; ++i)
        out[i] += d;
    return out;
}

/// The degenerate candidate-production battery (the brief's "identical geometry, a
/// single primitive, a vertex on the boundary"), driven through the white-box
/// `Impl::detect` so that all four candidate kinds are checked.
template <typename FilterT>
void check_degenerate_battery(const SceneDesc&                       desc,
                              const SceneData&                       original,
                              const std::vector<IndexT>&             offsets,
                              bool                                   dcd,
                              std::string_view                       variant)
{
    auto base = make_simplex_impl();

    const auto run = [&](std::string_view what)
    {
        auto d = gather();
        d.build_aabbs(0.0);
        auto r     = impl_detect<FilterT>(base);
        auto oracle = host_candidates(d, 0.0, dcd);
        note("[%s] %s: raw PP=%zu PE=%zu EE=%zu PT=%zu | cand PP=%zu PE=%zu EE=%zu PT=%zu "
             "(oracle %zu/%zu/%zu/%zu)",
             std::string{variant}.c_str(),
             std::string{what}.c_str(),
             r.raw_allp_codimp.size(),
             r.raw_codimp_alle.size(),
             r.raw_alle_alle.size(),
             r.raw_allp_allt.size(),
             r.cand_allp_codimp.size(),
             r.cand_codimp_alle.size(),
             r.cand_alle_alle.size(),
             r.cand_allp_allt.size(),
             oracle.allp_codimp.size(),
             oracle.codimp_alle.size(),
             oracle.alle_alle.size(),
             oracle.allp_allt.size());

        CHECK(r.raw_allp_codimp == host_raw_allp_codimp(d));
        CHECK(r.raw_codimp_alle == host_raw_codimp_alle(d));
        CHECK(r.raw_alle_alle == host_raw_alle_alle(d));
        CHECK(r.raw_allp_allt == host_raw_allp_allt(d));
        CHECK(r.cand_allp_codimp == oracle.allp_codimp);
        CHECK(r.cand_codimp_alle == oracle.codimp_alle);
        CHECK(r.cand_alle_alle == oracle.alle_alle);
        CHECK(r.cand_allp_allt == oracle.allp_allt);
        return r;
    };

    REQUIRE(desc.meshes.size() == 2);
    const IndexT a0 = offsets[0];
    const IndexT b0 = offsets[1];
    const IndexT b1 = offsets[2];

    // (i) identical geometry: mesh 1's four vertices sit on mesh 0's
    set_positions(shifted(original.positions, b0, b1, Vector3::Zero()
                                                         + (original.positions[a0]
                                                            - original.positions[b0])));
    {  // positive control: the write must be visible to a fresh gather()
        auto after = gather();
        CHECK(after.positions[b0] == original.positions[a0]);
    }
    run("identical-geometry");

    // (ii) a vertex exactly on the predicate's boundary: mesh 1 displaced so that
    //      |a0 - b0| == expand (the reference's `> dist` is strict, so `==` is kept)
    set_positions(shifted(original.positions,
                          b0,
                          b1,
                          Vector3{0.0, 0.1875, 0.0}
                              + (original.positions[a0] - original.positions[b0])));
    run("boundary-equal");

    // (iii) a hair beyond the boundary: the same pair is rejected by the predicate
    //       (and disappears from the candidate set while staying in the raw set)
    set_positions(shifted(original.positions,
                          b0,
                          b1,
                          Vector3{0.0, 0.1875 + 1.0 / 1024.0, 0.0}
                              + (original.positions[a0] - original.positions[b0])));
    run("boundary-beyond");

    // (iv) a single primitive in play: mesh 1 goes far away, so every candidate kind
    //      is empty (the intra-body pairs stay in the raw set and are rejected by the
    //      body self-collision check)
    set_positions(shifted(original.positions, b0, b1, Vector3{0.0, 10.0, 0.0}));
    {
        auto r = run("single-primitive");
        CHECK(r.total_candidates() == 0);
    }

    restore_positions(original);
}

/// The mask battery: mutate the per-vertex contact / subscene ids and see the pair
/// leave the *candidate* set (it stays in the raw set).  `contact_pair_disabled` and
/// `cross_subscene_enabled` are the two directions the scene's tabulars were built
/// with (the positive controls).
template <typename FilterT>
void check_mask_battery(const SceneData&           original,
                        const std::vector<IndexT>& offsets,
                        bool                       dcd,
                        bool                       contact_pair_disabled,
                        bool                       cross_subscene_enabled,
                        std::string_view           variant)
{
    auto base = make_simplex_impl();

    const IndexT a0 = offsets[0];
    const IndexT b0 = offsets[1];
    const IndexT b1 = offsets[2];

    const auto run = [&](std::string_view what)
    {
        auto d = gather();
        d.build_aabbs(0.0);
        auto r      = impl_detect<FilterT>(base);
        auto oracle = host_candidates(d, 0.0, dcd);
        note("[%s] masks %s: raw PP=%zu PE=%zu EE=%zu PT=%zu | cand PP=%zu PE=%zu EE=%zu PT=%zu "
             "(oracle %zu/%zu/%zu/%zu)",
             std::string{variant}.c_str(),
             std::string{what}.c_str(),
             r.raw_allp_codimp.size(),
             r.raw_codimp_alle.size(),
             r.raw_alle_alle.size(),
             r.raw_allp_allt.size(),
             r.cand_allp_codimp.size(),
             r.cand_codimp_alle.size(),
             r.cand_alle_alle.size(),
             r.cand_allp_allt.size(),
             oracle.allp_codimp.size(),
             oracle.codimp_alle.size(),
             oracle.alle_alle.size(),
             oracle.allp_allt.size());

        CHECK(r.cand_allp_codimp == oracle.allp_codimp);
        CHECK(r.cand_codimp_alle == oracle.codimp_alle);
        CHECK(r.cand_alle_alle == oracle.alle_alle);
        CHECK(r.cand_allp_allt == oracle.allp_allt);
        return r;
    };

    // NOTE: the *contact*-id direction is exercised by the dedicated
    // `t72b_contact_mask_disabled` case (a scene whose vertices all sit on a disabled
    // contact entry), not by mutating the global contact-id array here: `set_contact_ids`
    // (a mid-case write to the global per-vertex attribute) leaves the following
    // teardown in a state that crashes the process (measured; reported as a finding), and
    // a scene-level assignment pins the same rule without that hazard.
    (void)contact_pair_disabled;
    (void)a0;

    // ---- the subscene mask, both directions -------------------------------
    if(original.subscene_mask_size >= 2)
    {
        auto ids = original.subscene_ids;
        for(IndexT i = b0; i < b1; ++i)
            ids[i] = 1;
        set_subscene_ids(ids);
        {
            auto r = run(cross_subscene_enabled ? "subscene-cross-enabled"
                                               : "subscene-cross-disabled");
            if(cross_subscene_enabled)
                CHECK(r.total_candidates() > 0);  // the scene enabled (0,1)
            else
                CHECK(r.total_candidates() == 0);  // identity mask: cross pairs are off
        }
        set_subscene_ids(original.subscene_ids);
    }
}

/// One variant's case body (the four `t72b_oracle_*` cases share it).
template <typename PipelineFilterT, typename ImplOwnerT>
void run_variant_case(SceneDesc        desc,
                      bool             dcd,
                      Float            alpha,
                      std::string_view variant)
{
    test_runtime();

    auto scene = build_scene(desc);
    Engine engine{"luisa",
                  workspace_path(std::string{"uipc_luisa_t72b_"} + std::string{variant})};
    World world{engine};
    world.init(scene);

    auto* gtf = BvhFilterTestProbe::trajectory_filter;
    REQUIRE(gtf != nullptr);
    check_registration<PipelineFilterT>(gtf);

    auto       d       = gather();
    const auto offsets = mesh_offsets(desc);
    REQUIRE(d.positions.size() == offsets.back());

    // the test plays the Phase-8 fan-out that fills the global displacement array
    // (the reporters already reported `test_displacement`; this is the explicit form)
    set_displacements(d.displacements);
    d = gather();

    check_variant_battery<PipelineFilterT, ImplOwnerT>(d, alpha, dcd, variant);
    check_degenerate_battery<ImplOwnerT>(desc, d, offsets, dcd, variant);
    check_mask_battery<ImplOwnerT>(d,
                                   offsets,
                                   dcd,
                                   desc.contact_pair_disabled,
                                   desc.subscene_pair_enabled,
                                   variant);
}

}// namespace t72b

using namespace t72b;

// ===========================================================================
//  2) the LBVH variant: the candidate production vs the host oracle
// ===========================================================================
TEST_CASE("t72b_oracle_lbvh", "[bvh_filter]")
{
    SceneDesc desc;
    desc.two_contact_elements  = true;
    desc.contact_pair_disabled = true;  // the (1,1) contact entry is off

    // two quads, one 0.15 above the other and offset by (0.1, 0.1): every candidate kind
    // has a pair and the intra-body pairs are rejected by the self-collision check (both
    // geometries default to `self_collision == 0`).
    //
    // THE BAND ARITHMETIC (why 0.15): the reference's per-pair band is
    // `(thickness, thickness + d_hat)` with the *pairwise* values --
    // `PP_thickness = t0 + t1` (0.125 here, NOT 0.0625), `PP_d_hat = (h0 + h1)/2`
    // (0.125) -- i.e. `(0.125, 0.25)`.  A candidate closer than 0.125 would trip the
    // reference's `UIPC_KERNEL_ASSERT(D > thickness^2)` in `filter_active` (the host
    // oracle checks the same invariant, so a mis-designed scene fails loudly instead of
    // silently), and 0.15-0.21 keeps every cross pair inside the band.
    desc.variant_uid = 0;
    desc.meshes.push_back(plate(Vector3{0.0, 0.0, 0.0}));
    desc.meshes.push_back(plate(Vector3{0.1, 0.15, 0.1}, 0.6));
    for(auto& m : desc.meshes)
        for(size_t i = 0; i < m.positions.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.d_hats.push_back(0.125);
            m.thicknesses.push_back(0.0625);
            m.displacements.push_back(is_upper_plate(m.positions[0])
                                          ? Vector3{0.0, -0.01, 0.0}
                                          : Vector3::Zero());
        }
    desc.meshes[1].dimensions[0] = 1;  // one codim (1D) vertex -> AllP-CodimP / CodimP-AllE

    desc.variant_uid = 1;
    run_variant_case<T72bLbvhVariantFilter, gg::LBVHSimplexTrajectoryFilter>(
        desc, /*dcd*/ false, /*alpha*/ 0.5, "lbvh");
}

// ===========================================================================
//  3) the stackless variant
// ===========================================================================
TEST_CASE("t72b_oracle_stackless", "[bvh_filter]")
{
    SceneDesc desc;
    desc.two_subscenes = true;  // two subscenes -> the cross-subscene mask is the identity

    desc.variant_uid = 0;
    desc.meshes.push_back(plate(Vector3{0.0, 0.0, 0.0}));
    desc.meshes.push_back(plate(Vector3{0.1, 0.15, 0.1}, 0.6));
    for(auto& m : desc.meshes)
        for(size_t i = 0; i < m.positions.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.d_hats.push_back(0.125);
            m.thicknesses.push_back(0.0625);
            m.displacements.push_back(is_upper_plate(m.positions[0])
                                          ? Vector3{0.0, -0.01, 0.0}
                                          : Vector3::Zero());
        }
    desc.meshes[1].dimensions[0] = 1;

    desc.variant_uid = 2;
    run_variant_case<T72bStacklessVariantFilter, gg::StacklessBVHSimplexTrajectoryFilter>(
        desc, false, 0.5, "stackless");
}

// ===========================================================================
//  4) the info v1 variant (the only one with the `alpha == 0` DCD branch)
// ===========================================================================
TEST_CASE("t72b_oracle_infov1", "[bvh_filter]")
{
    SceneDesc desc;
    desc.two_subscenes         = true;
    desc.subscene_pair_enabled = true;  // (s0, s1) enabled: the positive direction

    desc.variant_uid = 0;
    desc.meshes.push_back(plate(Vector3{0.0, 0.0, 0.0}));
    desc.meshes.push_back(plate(Vector3{0.1, 0.15, 0.1}, 0.6));
    for(auto& m : desc.meshes)
        for(size_t i = 0; i < m.positions.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.d_hats.push_back(0.125);
            m.thicknesses.push_back(0.0625);
            m.displacements.push_back(is_upper_plate(m.positions[0])
                                          ? Vector3{0.0, -0.01, 0.0}
                                          : Vector3::Zero());
        }
    desc.meshes[1].dimensions[0] = 1;

    run_variant_case<gg::InfoStacklessBVHSimplexTrajectoryFilter,
                     gg::InfoStacklessBVHSimplexTrajectoryFilter>(desc, true, 0.5, "infov1");
}

// ===========================================================================
//  5) the info v0 variant (the box-flavour predicates, like lbvh/stackless)
// ===========================================================================
TEST_CASE("t72b_oracle_infov0", "[bvh_filter]")
{
    SceneDesc desc;
    // the (1,1) contact entry stays *enabled* (the positive control for the contact
    // mask direction; the disabled direction has its own case)
    desc.two_contact_elements  = true;
    desc.contact_pair_disabled = false;

    desc.variant_uid = 0;
    desc.meshes.push_back(plate(Vector3{0.0, 0.0, 0.0}));
    desc.meshes.push_back(plate(Vector3{0.1, 0.15, 0.1}, 0.6));
    for(auto& m : desc.meshes)
        for(size_t i = 0; i < m.positions.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.d_hats.push_back(0.125);
            m.thicknesses.push_back(0.0625);
            m.displacements.push_back(is_upper_plate(m.positions[0])
                                          ? Vector3{0.0, -0.01, 0.0}
                                          : Vector3::Zero());
        }
    desc.meshes[1].dimensions[0] = 1;

    desc.variant_uid = 3;
    run_variant_case<T72bInfov0VariantFilter, gg::InfoStacklessBVHV0SimplexTrajectoryFilter>(
        desc, false, 0.5, "infov0");
}

// ===========================================================================
//  5b) the contact mask: a scene whose vertices all sit on a *disabled* entry
// ===========================================================================
TEST_CASE("t72b_contact_mask_disabled", "[bvh_filter]")
{
    test_runtime();

    SceneDesc desc;
    desc.variant_uid          = 0;  // the schema's default method registers v1
    desc.two_contact_elements = true;
    desc.contact_pair_disabled = true;  // the (1,1) entry is off

    // every mesh vertex carries contact element 1, so *every* pair is (1,1): the raw
    // (AABB) overlap sets are unchanged while every candidate set must be empty
    desc.meshes.push_back(plate(Vector3{0.0, 0.0, 0.0}));
    desc.meshes.push_back(plate(Vector3{0.1, 0.15, 0.1}, 0.6));
    for(auto& m : desc.meshes)
    {
        for(size_t i = 0; i < m.positions.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.contact_ids.push_back(1);
            m.d_hats.push_back(0.125);
            m.thicknesses.push_back(0.0625);
            m.displacements.push_back(Vector3::Zero());
        }
        m.self_collision = 1;  // keep the body check out of the way
    }
    desc.meshes[1].dimensions[0] = 1;

    // the half plane sits on contact element 0 (the plane's own entry is enabled)
    desc.half_planes[0].contact_id = 0;

    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t72b_contact_mask")};
    World  world{engine};
    world.init(scene);

    auto* gtf = BvhFilterTestProbe::trajectory_filter;
    REQUIRE(gtf != nullptr);
    check_registration<gg::InfoStacklessBVHSimplexTrajectoryFilter>(gtf);

    auto d = gather();
    REQUIRE(d.contact_mask_size == 2);
    REQUIRE(d.contact_mask[1 * d.contact_mask_size + 1] == 0);  // (1,1) disabled
    REQUIRE(d.contact_mask[0 * d.contact_mask_size + 0] == 1);  // (0,0) enabled

    d.build_aabbs(0.0);
    const auto raw = host_raw_allp_allt(d);
    const auto oracle_exact = host_candidates(d, 0.0, true);
    const auto oracle_box   = host_candidates(d, 0.0, false);
    CHECK(oracle_exact.total() == 0);
    CHECK(oracle_box.total() == 0);
    CHECK(!raw.empty());  // the *traversal* still reports the overlapping pairs

    auto base = make_simplex_impl();
    {
        auto r = impl_detect<gg::LBVHSimplexTrajectoryFilter>(base);
        CHECK(r.raw_allp_allt == raw);   // the mask is a selection, not a traversal cull
        CHECK(r.total_candidates() == 0);
    }
    {
        auto r = impl_detect<gg::StacklessBVHSimplexTrajectoryFilter>(base);
        CHECK(r.total_candidates() == 0);
    }
    {
        auto r = impl_detect<gg::InfoStacklessBVHSimplexTrajectoryFilter>(base);
        CHECK(r.total_candidates() == 0);
    }
    {
        auto r = impl_detect<gg::InfoStacklessBVHV0SimplexTrajectoryFilter>(base);
        CHECK(r.total_candidates() == 0);
    }
    note("contact mask (1,1) disabled: raw AABB overlaps=%zu in the AllP-AllT query, "
         "0 candidates in all four variants",
         raw.size());
}

// ===========================================================================
//  6) the empty path, the buffer capacities and the pair-buffer retry contract
// ===========================================================================
TEST_CASE("t72b_empty_and_capacity", "[bvh_filter]")
{
    test_runtime();

    SceneDesc desc;
    desc.variant_uid = 0;  // the schema's default method drives the pipeline
    desc.meshes.push_back(plate(Vector3{0.0, 0.0, 0.0}));
    desc.meshes.push_back(plate(Vector3{0.1, 0.15, 0.1}, 0.6));
    for(auto& m : desc.meshes)
        for(size_t i = 0; i < m.positions.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.d_hats.push_back(0.125);
            m.thicknesses.push_back(0.0625);
            m.displacements.push_back(Vector3::Zero());
        }
    desc.meshes[1].dimensions[0] = 1;  // one codim (1D) vertex

    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t72b_empty")};
    World  world{engine};
    world.init(scene);

    auto* gtf = BvhFilterTestProbe::trajectory_filter;
    REQUIRE(gtf != nullptr);
    const auto offsets = mesh_offsets(desc);

    auto         d     = gather();
    const IndexT b0    = offsets[1];
    const IndexT b1    = offsets[2];

    // ---- (a) the QueryBuffer capacity contract ------------------------------
    // The reference's `AtomicCountingLBVH::QueryBuffer` starts *empty* (the T7.1a port
    // pads it to one element so it stays a legal LC kernel argument) while the three
    // stackless-family ones reserve 50 * 1024 pairs up front, and all four grow by
    // their `Config::reserve_ratio` (1.2) / `m_reserve_ratio` (1.1) on a retry.
    {
        gg::StacklessBVH::QueryBuffer stackless_qb;
        CHECK(stackless_qb.m_pairs.size() == 50 * 1024);
        CHECK(gg::StacklessBVH::Config{}.reserve_ratio == 1.2);

        gg::InfoStacklessBVH::QueryBuffer v1_qb;
        CHECK(v1_qb.m_pairs.size() == 50 * 1024);
        CHECK(gg::InfoStacklessBVH::Config{}.reserve_ratio == 1.2);

        gg::InfoStacklessBVHV0::QueryBuffer v0_qb;
        CHECK(v0_qb.m_pairs.size() == 50 * 1024);
        CHECK(gg::InfoStacklessBVHV0::Config{}.reserve_ratio == 1.2);

        note("QueryBuffer initial capacities: stackless/info(reserve_ratio 1.2) = %zu "
             "pairs; the AtomicCountingLBVH's starts empty and is padded to 1 by its "
             "detect/query (its private m_reserve_ratio is 1.1)",
             stackless_qb.m_pairs.size());
    }

    // ---- (b) the pair-buffer retry contract, on the scene's own AABBs --------
    // The LBVH's QueryBuffer starts empty, so *every* detect takes the reference's
    // "count > capacity -> grow by the reserve ratio and re-run the query from ticket
    // 0" path; the reported set must still be exactly the brute-force AABB overlap set.
    {
        auto keep_all = [](const auto&, const auto&) noexcept { return true; };

        d.build_aabbs(0.0);

        lud::LDeviceBuffer<gg::LAABB> edge_aabbs;
        edge_aabbs.resize(d.edge_aabbs.size());
        {
            std::vector<gg::LAABB> host(d.edge_aabbs.size());
            for(size_t i = 0; i < d.edge_aabbs.size(); ++i)
                host[i] = gg::to_luisa(d.edge_aabbs[i]);
            edge_aabbs.view().copy_from(span<const gg::LAABB>{host});
        }

        gg::AtomicCountingLBVH lbvh;
        lbvh.build(edge_aabbs.view());
        gg::AtomicCountingLBVH::QueryBuffer qb;
        lbvh.detect(keep_all, qb);
        gg::frame_stream().synchronize();

        auto reported = to_pair_set(read_vector2i(qb.view()));
        auto brute    = host_raw_alle_alle(d);

        CHECK(qb.size() == reported.size());
        CHECK(reported == brute);
        note("LBVH retry contract: %zu reported edge pairs == %zu brute-force overlap "
             "pairs (the buffer started empty, so the query ran twice)",
             reported.size(),
             brute.size());
    }

    // ---- (c) the empty path -------------------------------------------------
    // every primitive of mesh 1 moves 10 units away -> no AABB overlap at all:
    // `tois.size() == 0`, so `filter_toi` takes the reference's `else
    // info.toi().fill(large_enough_toi)` branch (the ported reduce would otherwise
    // *write* the reduce identity for n == 0), and the manager returns the clamped 1.0.
    {
        set_positions(shifted(d.positions, b0, b1, Vector3{0.0, 10.0, 0.0}));
        set_displacements(std::vector<Vector3>(d.positions.size(), Vector3::Zero()));

        auto slot = gtf->find<gg::InfoStacklessBVHSimplexTrajectoryFilter>();
        REQUIRE(static_cast<bool>(slot));
        auto* filter = slot.view();

        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::Detect, 0.5) == 1.0);
        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive) == 1.0);
        CHECK(filter->candidate_PTs().size() == 0);
        CHECK(filter->candidate_EEs().size() == 0);
        CHECK(filter->toi_PTs().size() == 0);
        CHECK(filter->toi_EEs().size() == 0);

        const auto step = gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, 0.5);
        CHECK(step == 1.0);
        CHECK(filter->PTs().size() == 0);
        CHECK(filter->EEs().size() == 0);
        CHECK(filter->PEs().size() == 0);
        CHECK(filter->PPs().size() == 0);
        note("empty path: 0 candidates, 0 tois, manager step=%.1f (the fill(1.1) branch)",
             static_cast<double>(step));

        restore_positions(d);
        set_displacements(d.displacements);
    }

    // ---- (d) back to a non-empty scene: the same filter reports again --------
    {
        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::Detect, 0.5) == 1.0);
        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive) == 1.0);
        // the displacements are zero here, so every candidate's CCD finds no collision
        // and reports the 1.1 sentinel -> the manager clamps the step to 1.0
        const auto static_step =
            gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, 0.5);
        CHECK(static_step == 1.0);
        note("static scene: every toi is the 1.1 sentinel, manager step=%.1f",
             static_cast<double>(static_step));
    }
}

// ===========================================================================
//  7) determinism + the batch's factored routines, device vs host
// ===========================================================================
TEST_CASE("t72b_determinism_and_device_vs_host", "[bvh_filter]")
{
    test_runtime();

    SceneDesc desc;
    desc.variant_uid = 2;
    desc.meshes.push_back(plate(Vector3{0.0, 0.0, 0.0}));
    desc.meshes.push_back(plate(Vector3{0.1, 0.15, 0.1}, 0.6));
    for(auto& m : desc.meshes)
        for(size_t i = 0; i < m.positions.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.d_hats.push_back(0.125);
            m.thicknesses.push_back(0.0625);
            m.displacements.push_back(is_upper_plate(m.positions[0])
                                          ? Vector3{0.0, -0.01, 0.0}
                                          : Vector3::Zero());
        }
    desc.meshes[1].dimensions[0] = 1;

    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t72b_determinism")};
    World  world{engine};
    world.init(scene);

    auto* gtf = BvhFilterTestProbe::trajectory_filter;
    REQUIRE(gtf != nullptr);
    auto d    = gather();
    auto base = make_simplex_impl();

    // ---- (a) determinism: two pipeline runs and two white-box detects --------
    {
        auto p1 = run_pipeline<T72bStacklessVariantFilter>(0.5);
        auto p2 = run_pipeline<T72bStacklessVariantFilter>(0.5);

        CHECK(to_pair_set(p1.pt_candidates) == to_pair_set(p2.pt_candidates));
        CHECK(to_pair_set(p1.ee_candidates) == to_pair_set(p2.ee_candidates));
        CHECK(to_multiset2(p1.PPs) == to_multiset2(p2.PPs));
        CHECK(to_multiset3(p1.PEs) == to_multiset3(p2.PEs));
        CHECK(to_multiset4(p1.PTs) == to_multiset4(p2.PTs));
        CHECK(to_multiset4(p1.EEs) == to_multiset4(p2.EEs));
        REQUIRE(p1.toi_pt.size() == p2.toi_pt.size());
        REQUIRE(p1.toi_ee.size() == p2.toi_ee.size());
        int diff = 0;
        for(size_t i = 0; i < p1.toi_pt.size(); ++i)
            diff += (p1.toi_pt[i] == p2.toi_pt[i]) ? 0 : 1;
        for(size_t i = 0; i < p1.toi_ee.size(); ++i)
            diff += (p1.toi_ee[i] == p2.toi_ee[i]) ? 0 : 1;
        CHECK(diff == 0);
        CHECK(p1.step == p2.step);

        auto r1 = impl_detect<gg::StacklessBVHSimplexTrajectoryFilter>(base);
        auto r2 = impl_detect<gg::StacklessBVHSimplexTrajectoryFilter>(base);
        CHECK(r1.raw_allp_codimp == r2.raw_allp_codimp);
        CHECK(r1.cand_alle_alle == r2.cand_alle_alle);
        CHECK(r1.cand_allp_allt == r2.cand_allp_allt);
        note("determinism: PT=%zu EE=%zu candidates, PP=%zu PE=%zu PT=%zu EE=%zu active, "
             "%zu + %zu toi values, %d toi differences",
             p1.pt_candidates.size(),
             p1.ee_candidates.size(),
             p1.PPs.size(),
             p1.PEs.size(),
             p1.PTs.size(),
             p1.EEs.size(),
             p1.toi_pt.size(),
             p1.toi_ee.size(),
             diff);
    }

    // ---- (b) the layer's struct compaction, device vs host -------------------
    // `lla::select_valid_pairs<LVector2i>` / `compact_by_flags` is the primitive the
    // four filters' `filter_active` uses for the reference's
    // `DeviceSelect::If(..., {pair(0) != -1})`; it is exercised here on a random array
    // with sentinels and compared against a strided host implementation.
    {
        std::mt19937                          rng(0x72b1u);
        std::uniform_int_distribution<int>    value(-2, 7);
        std::uniform_int_distribution<int>    sentinel(0, 3);
        const size_t                          n = 257;
        std::vector<gg::LVector2i>            host(n);
        for(size_t i = 0; i < n; ++i)
        {
            if(sentinel(rng) == 0)
            {
                host[i](0) = -1;
                host[i](1) = -1;
            }
            else
            {
                host[i](0) = value(rng);
                host[i](1) = value(rng);
            }
        }

        lud::LDeviceBuffer<gg::LVector2i> in;
        in.resize(n);
        in.view().copy_from(span<const gg::LVector2i>{host});
        lud::LDeviceBuffer<gg::LVector2i> out;
        out.resize(n);
        lud::LDeviceVar<std::uint32_t> count;
        auto&                          stream = gg::frame_stream();
        lla::select_valid_pairs<gg::LVector2i>(stream, in.cview(), out.view(), count.view(), n);
        auto kept = read_vector2i(out.cview());
        auto h_count = lud::read_host_sync(stream, count.cview());

        std::vector<Vector2i> want;
        for(auto& p : host)
            if(p(0) != -1)
                want.push_back(gg::to_eigen(p));

        CHECK(h_count == want.size());
        REQUIRE(kept.size() == n);
        int mismatches = 0;
        for(size_t i = 0; i < want.size(); ++i)
            mismatches += (kept[i] == want[i]) ? 0 : 1;
        CHECK(mismatches == 0);
        // stable: the input order survives
        if(!want.empty())
            CHECK(kept[0] == want[0]);
        CHECK(!want.empty());
        // n == 0 writes 0 and dispatches nothing
        {
            lud::LDeviceVar<std::uint32_t> zero_count;
            zero_count.copy_from(std::uint32_t{7u});
            lla::select_valid_pairs<gg::LVector2i>(stream,
                                                  lud::LCBufferView<gg::LVector2i>{},
                                                  lud::LBufferView<gg::LVector2i>{},
                                                  zero_count.view(),
                                                  0u);
            CHECK(lud::read_host_sync(stream, zero_count.cview()) == 0u);
        }
        note("select_valid_pairs<LVector2i>: %zu -> %zu kept (stable), %d mismatches; "
             "n==0 writes 0",
             n,
             want.size(),
             mismatches);
    }

    // ---- (c) the ported distance/ccd broadphase, device vs host --------------
    {
        struct Case
        {
            Vector3 p0, p1, dp0, dp1;
            Float   dist;
        };
        std::vector<Case> cases;
        std::mt19937      rng(0x72b2u);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for(int i = 0; i < 64; ++i)
        {
            cases.push_back(Case{{dist(rng), dist(rng), dist(rng)},
                                 {dist(rng), dist(rng), dist(rng)},
                                 {dist(rng), dist(rng), dist(rng)},
                                 {dist(rng), dist(rng), dist(rng)},
                                 static_cast<Float>(std::abs(dist(rng))) * 0.5});
        }
        // the exact boundary: |p0 - p1| == dist on one axis
        cases.push_back(Case{{0.0, 0.0, 0.0},
                             {0.5, 0.0, 0.0},
                             {0.0, 0.0, 0.0},
                             {0.0, 0.0, 0.0},
                             0.5});

        // the kernel reads `offset + 0..2` = p0, `+3..5` = p1, `+6..8` = dp0,
        // `+9..11` = dp1 (a *blocked* layout, one case per 12 doubles)
        std::vector<Float> coords;
        std::vector<Float> dists;
        for(auto& c : cases)
        {
            for(const Vector3* v : {&c.p0, &c.p1, &c.dp0, &c.dp1})
                for(int k = 0; k < 3; ++k)
                    coords.push_back((*v)[k]);
            dists.push_back(c.dist);
        }
        lud::LDeviceBuffer<Float> coords_buf, dist_buf;
        coords_buf.resize(coords.size());
        dist_buf.resize(dists.size());
        coords_buf.view().copy_from(span<const Float>{coords});
        dist_buf.view().copy_from(span<const Float>{dists});

        lud::LDeviceBuffer<IndexT> flag_buf;
        flag_buf.resize(cases.size());

        auto& stream = gg::frame_stream();
        auto& k      = lla::detail::cached_kernel<lcrt::Buffer<double>, lcrt::Buffer<double>,
                                                  lcrt::Buffer<std::int32_t>, std::uint32_t>(
            [](lcrt::BufferVar<double> c, lcrt::BufferVar<double> d,
               lcrt::BufferVar<std::int32_t> out, lcrt::Var<std::uint32_t> n) noexcept
            {
                lcrt::set_block_size(32u);
                const auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    const auto offset = i * 12u;
                    lps::LVec<lcrt::Var<Float>, 3> p0;
                  lps::LVec<lcrt::Var<Float>, 3> p1;
                  lps::LVec<lcrt::Var<Float>, 3> dp0;
                  lps::LVec<lcrt::Var<Float>, 3> dp1;
                    for(int j = 0; j < 3; ++j)
                    {
                        p0(j)  = c->read(offset + 0u + static_cast<std::uint32_t>(j));
                        p1(j)  = c->read(offset + 3u + static_cast<std::uint32_t>(j));
                        dp0(j) = c->read(offset + 6u + static_cast<std::uint32_t>(j));
                        dp1(j) = c->read(offset + 9u + static_cast<std::uint32_t>(j));
                    }
                    const auto hit = ldist::point_point_ccd_broadphase(p0, p1, dp0, dp1,
                                                                       d->read(i));
                    out->write(i, lcrt::select(std::int32_t{0}, std::int32_t{1}, hit));
                };
            });
        auto n = static_cast<std::uint32_t>(cases.size());
        stream.record(k(coords_buf.view(), dist_buf.view(), flag_buf.view(), n)
                          .dispatch(((n + 31u) / 32u) * 32u));
        auto host_flags = read_device(flag_buf.cview());

        int mismatches = 0;
        for(size_t i = 0; i < cases.size(); ++i)
        {
            auto want = host_pp_broadphase(
                cases[i].p0, cases[i].p1, cases[i].dp0, cases[i].dp1, cases[i].dist);
            mismatches += (static_cast<bool>(host_flags[i]) == want) ? 0 : 1;
        }
        CHECK(mismatches == 0);
        note("point_point_ccd_broadphase device-vs-host: %zu cases, %d mismatches",
             cases.size(),
             mismatches);
    }

    // ---- (d) the ported primitive_d_hat layer, device vs host ---------------
    {
        std::vector<Float> d_hats   = {0.125, 0.25, 0.0625, 0.5};
        std::vector<Float> d_hats2  = {0.375, 0.125, 0.5, 0.25};
        lud::LDeviceBuffer<Float> a, b;
        a.resize(4);
        b.resize(4);
        a.view().copy_from(span<const Float>{d_hats});
        b.view().copy_from(span<const Float>{d_hats2});
        lud::LDeviceBuffer<Float> out_pp, out_pe, out_ee, out_pt, out_edge, out_tri;
        out_pp.resize(4);
        out_pe.resize(4);
        out_ee.resize(4);
        out_pt.resize(4);
        out_edge.resize(4);
        out_tri.resize(4);

        auto& stream = gg::frame_stream();
        auto& k      = lla::detail::cached_kernel<lcrt::Buffer<double>, lcrt::Buffer<double>,
                                                  lcrt::Buffer<double>, lcrt::Buffer<double>,
                                                  lcrt::Buffer<double>, lcrt::Buffer<double>,
                                                  lcrt::Buffer<double>, lcrt::Buffer<double>,
                                                  std::uint32_t>(
            [](lcrt::BufferVar<double> a, lcrt::BufferVar<double> b, lcrt::BufferVar<double> pp,
               lcrt::BufferVar<double> pe, lcrt::BufferVar<double> ee, lcrt::BufferVar<double> pt,
               lcrt::BufferVar<double> edge, lcrt::BufferVar<double> tri,
               lcrt::Var<std::uint32_t> n) noexcept
            {
                lcrt::set_block_size(32u);
                const auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    pp->write(i, lprim::PP_d_hat(a->read(i), b->read(i)));
                    pe->write(i, lprim::PE_d_hat(a->read(i), b->read(i), b->read(i)));
                    ee->write(i, lprim::EE_d_hat(a->read(i), a->read(i), b->read(i), b->read(i)));
                    pt->write(i, lprim::PT_d_hat(a->read(i), b->read(i), b->read(i), b->read(i)));
                    edge->write(i, lprim::edge_dcd_expansion(a->read(i), a->read(i)));
                    tri->write(i, lprim::triangle_dcd_expansion(a->read(i), a->read(i), a->read(i)));
                };
            });
        std::uint32_t n = 4u;
        stream.record(k(a.view(), b.view(), out_pp.view(), out_pe.view(), out_ee.view(),
                        out_pt.view(), out_edge.view(), out_tri.view(), n)
                          .dispatch(32u));

        auto h_pp   = read_device(out_pp.cview());
        auto h_pe   = read_device(out_pe.cview());
        auto h_ee   = read_device(out_ee.cview());
        auto h_pt   = read_device(out_pt.cview());
        auto h_edge = read_device(out_edge.cview());
        auto h_tri  = read_device(out_tri.cview());

        int mismatches = 0;
        for(int i = 0; i < 4; ++i)
        {
            mismatches += (h_pp[i] == lprim::PP_d_hat(d_hats[i], d_hats2[i])) ? 0 : 1;
            mismatches += (h_pe[i] == lprim::PE_d_hat(d_hats[i], d_hats2[i], d_hats2[i])) ? 0 : 1;
            mismatches += (h_ee[i] == lprim::EE_d_hat(d_hats[i], d_hats[i], d_hats2[i], d_hats2[i])) ? 0 : 1;
            mismatches += (h_pt[i] == lprim::PT_d_hat(d_hats[i], d_hats2[i], d_hats2[i], d_hats2[i])) ? 0 : 1;
            mismatches += (h_edge[i] == lprim::edge_dcd_expansion(d_hats[i], d_hats[i])) ? 0 : 1;
            mismatches += (h_tri[i] == lprim::triangle_dcd_expansion(d_hats[i], d_hats[i], d_hats[i])) ? 0 : 1;
        }
        CHECK(mismatches == 0);
        note("primitive_d_hat device-vs-host: 4 d_hat pairs x 6 routines, %d mismatches",
             mismatches);
    }
}

// ===========================================================================
//  8) the quoted numbers
// ===========================================================================
TEST_CASE("t72b_summary", "[bvh_filter]")
{
    test_runtime();
    std::printf(
        "[bvh_filter] T7.2b: filters/{lbvh,stackless_bvh,info_stackless_bvh,"
        "info_stackless_bvh_v0}_simplex_trajectory_filter.\n");
    std::printf(
        "[bvh_filter] four-way table: at alpha != 0 all four variants' candidate sets, "
        "active sets and TOI values equal the same host oracle (the predicates of the "
        "lbvh/stackless/info-v0 flavours are the same bodies; the info-v1 flavour uses "
        "the same bodies plus its `alpha == 0` exact-distance branch, which is the "
        "batch's one behavioural difference).\n");
    std::printf(
        "[bvh_filter] the predicates are applied as device *selection passes* over the "
        "keep-all traversal's output (the reference applies them inside the traversal "
        "callback, after the fixed node test, so the surviving set is identical); the "
        "info variants' node-level cull is implied by their leaf predicate, so no "
        "device-data hook was needed.\n");
    CHECK(true);
}


// ===========================================================================
//  1) registration, the scene/accessor contracts, the capacity contracts and the
//     four-way DCD table
// ===========================================================================
TEST_CASE("t72b_registration_and_contracts", "[bvh_filter]")
{
    test_runtime();

    SceneDesc desc;
    desc.variant_uid = 0;
    desc.meshes.push_back(plate(Vector3{0.0, 0.0, 0.0}));
    desc.meshes.push_back(plate(Vector3{0.1, 0.15, 0.1}, 0.6));
    for(auto& m : desc.meshes)
        for(size_t i = 0; i < m.positions.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.d_hats.push_back(0.125);
            m.thicknesses.push_back(0.0625);
            m.displacements.push_back(Vector3::Zero());
        }
    desc.meshes[1].dimensions[0] = 1;  // one codim (1D) vertex

    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t72b_contracts")};
    World  world{engine};
    world.init(scene);

    auto* gtf = BvhFilterTestProbe::trajectory_filter;
    auto* vm  = BvhFilterTestProbe::vertex_manager;
    auto* sm  = BvhFilterTestProbe::surface_manager;
    auto* cm  = BvhFilterTestProbe::contact_manager;
    REQUIRE(gtf != nullptr);
    REQUIRE(vm != nullptr);
    REQUIRE(sm != nullptr);
    REQUIRE(cm != nullptr);

    // ---- the method gate ---------------------------------------------------
    // the scene's `collision_detection/method` can only be the schema's default in this
    // build, so `info_stackless_bvh`'s filter is the only *scene* filter registered
    check_registration<gg::InfoStacklessBVHSimplexTrajectoryFilter>(gtf);

    // ---- the surface/vertex contracts --------------------------------------
    auto d = gather();
    CHECK(d.surf_vertices.size() == 8);
    CHECK(d.surf_edges.size() == 10);       // two quads: 5 edges each
    CHECK(d.surf_triangles.size() == 4);    // two quads: 2 triangles each
    CHECK(d.codim_vertices.size() == 1);    // the one dimension-1 vertex
    CHECK(d.codim_vertices[0] == 5);        // mesh 1's first vertex
    CHECK(d.positions.size() == 9);  // 8 mesh vertices + the half plane's vertex
    CHECK(d.v2b.size() == 9);  // the half plane's vertex is reported too
    CHECK(d.body_self_collision.size() == 2);  // two geometries -> two bodies
    CHECK(d.body_self_collision[0] == 0);      // `test_self_collision` defaults to 0
    CHECK(d.contact_mask_size == 1);           // one contact element
    CHECK(d.subscene_mask_size == 1);          // one subscene

    // the accessor surface of the *BaseInfo* (both bases' accessors mirror the
    // managers -- the batch's filters read exactly these)
    {
        gg::SimplexTrajectoryFilter::Impl impl = make_simplex_impl();
        gg::SimplexTrajectoryFilter::BaseInfo base{&impl};
        CHECK(base.d_hat() == cm->d_hat());
        CHECK(base.toi_safety_margin() == 0.1);
        CHECK(base.d_hats().size() == vm->d_hats().size());
        CHECK(base.v2b().size() == vm->body_ids().size());
        CHECK(base.positions().size() == vm->positions().size());
        CHECK(base.rest_positions().size() == vm->rest_positions().size());
        CHECK(base.thicknesses().size() == vm->thicknesses().size());
        CHECK(base.dimensions().size() == vm->dimensions().size());
        CHECK(base.contact_element_ids().size() == vm->contact_element_ids().size());
        CHECK(base.subscene_element_ids().size() == vm->subscene_element_ids().size());
        CHECK(base.contact_mask_tabular().size() == d.contact_mask.size());
        CHECK(base.subscene_mask_tabular().size() == d.subscene_mask.size());
        CHECK(base.contact_mask_size() == cm->contact_mask_size());
        CHECK(base.subscene_mask_size() == cm->subscene_mask_size());
        CHECK(base.body_self_collision().size() == 2);
        CHECK(base.codim_vertices().size() == d.codim_vertices.size());
        CHECK(base.surf_vertices().size() == d.surf_vertices.size());
        CHECK(base.surf_edges().size() == d.surf_edges.size());
        CHECK(base.surf_triangles().size() == d.surf_triangles.size());
        gg::SimplexTrajectoryFilter::DetectInfo dinfo{&impl};
        CHECK(dinfo.alpha() == 0.0);
        CHECK(dinfo.displacements().size() == vm->displacements().size());
    }

    // ---- the four filters' own `Impl` contracts ----------------------------
    // (a) the initial pair-buffer capacities: the reference's AtomicCountingLBVH
    // QueryBuffer starts empty (the T7.1a port pads it to one element so it is a
    // valid LC kernel argument), the three stackless-family ones reserve 50*1024.
    {
        gg::LBVHSimplexTrajectoryFilter::Impl lbvh;
        CHECK(lbvh.raw_AllP_CodimP_pairs.size() == 0);  // nothing ran yet
        CHECK(lbvh.candidate_AllP_CodimP_pairs.size() == 0);

        gg::StacklessBVHSimplexTrajectoryFilter::Impl stackless;
        CHECK(stackless.raw_AllP_CodimP_pairs.m_pairs.size() == 50 * 1024);
        CHECK(gg::StacklessBVH::Config{}.reserve_ratio == 1.2);

        gg::InfoStacklessBVHSimplexTrajectoryFilter::Impl v1;
        CHECK(v1.raw_AllP_CodimP_pairs.m_pairs.size() == 50 * 1024);
        CHECK(gg::InfoStacklessBVH::Config{}.reserve_ratio == 1.2);

        gg::InfoStacklessBVHV0SimplexTrajectoryFilter::Impl v0;
        CHECK(v0.raw_AllP_CodimP_pairs.m_pairs.size() == 50 * 1024);
        CHECK(gg::InfoStacklessBVHV0::Config{}.reserve_ratio == 1.2);
        note("initial QueryBuffer capacities: lbvh=%zu (padded on first use), "
             "stackless/info=%zu, reserve_ratio=%.1f",
             static_cast<size_t>(0),
             static_cast<size_t>(50 * 1024),
             1.2);
    }

    // ---- the four-way DCD table (white-box, alpha == 0) ---------------------
    // v1 is the only flavour with the `alpha == 0` exact-distance branch: its
    // candidate sets are the *Euclidean* ones, while the other three keep the
    // conservative swept-box (per-axis, L-infinity) selection.
    d.build_aabbs(0.0);
    const auto dcd_exact = host_candidates(d, 0.0, true);
    const auto box_sets  = host_candidates(d, 0.0, false);

    {
        auto base = make_simplex_impl();
        auto r    = impl_detect<gg::LBVHSimplexTrajectoryFilter>(base);
        CHECK(r.cand_allp_codimp == box_sets.allp_codimp);
        CHECK(r.cand_codimp_alle == box_sets.codimp_alle);
        CHECK(r.cand_alle_alle == box_sets.alle_alle);
        CHECK(r.cand_allp_allt == box_sets.allp_allt);
        note("DCD lbvh: %zu candidates (box oracle %zu, exact oracle %zu)",
             r.total_candidates(),
             box_sets.total(),
             dcd_exact.total());
    }
    {
        auto base = make_simplex_impl();
        auto r    = impl_detect<gg::StacklessBVHSimplexTrajectoryFilter>(base);
        CHECK(r.cand_allp_codimp == box_sets.allp_codimp);
        CHECK(r.cand_codimp_alle == box_sets.codimp_alle);
        CHECK(r.cand_alle_alle == box_sets.alle_alle);
        CHECK(r.cand_allp_allt == box_sets.allp_allt);
        note("DCD stackless: %zu candidates", r.total_candidates());
    }
    {
        auto base = make_simplex_impl();
        auto r    = impl_detect<gg::InfoStacklessBVHV0SimplexTrajectoryFilter>(base);
        CHECK(r.cand_allp_codimp == box_sets.allp_codimp);
        CHECK(r.cand_codimp_alle == box_sets.codimp_alle);
        CHECK(r.cand_alle_alle == box_sets.alle_alle);
        CHECK(r.cand_allp_allt == box_sets.allp_allt);
        note("DCD info_stackless_bvh_v0: %zu candidates (== the box oracle, i.e. == the "
             "lbvh/stackless flavours)",
             r.total_candidates());

        // the info variants' per-primitive bid/cid arrays (the node metadata the
        // reference's node-level cull consumed; the port still builds and pins them:
        // a point's is its vertex's, an edge's/triangle's is its FIRST vertex's)
        REQUIRE(r.point_bids.size() == d.surf_vertices.size());
        REQUIRE(r.edge_bids.size() == d.surf_edges.size());
        REQUIRE(r.triangle_bids.size() == d.surf_triangles.size());
        int bid_mismatches = 0;
        for(size_t i = 0; i < r.point_bids.size(); ++i)
        {
            auto v        = d.surf_vertices[i];
            bid_mismatches += (r.point_bids[i] == d.v2b[v]) ? 0 : 1;
            bid_mismatches += (r.point_cids[i] == d.contact_ids[v]) ? 0 : 1;
        }
        for(size_t i = 0; i < r.edge_bids.size(); ++i)
        {
            auto v0 = d.surf_edges[i][0];
            bid_mismatches += (r.edge_bids[i] == d.v2b[v0]) ? 0 : 1;
            bid_mismatches += (r.edge_cids[i] == d.contact_ids[v0]) ? 0 : 1;
        }
        for(size_t i = 0; i < r.triangle_bids.size(); ++i)
        {
            auto v0 = d.surf_triangles[i][0];
            bid_mismatches += (r.triangle_bids[i] == d.v2b[v0]) ? 0 : 1;
            bid_mismatches += (r.triangle_cids[i] == d.contact_ids[v0]) ? 0 : 1;
        }
        for(size_t i = 0; i < r.codim_point_bids.size(); ++i)
        {
            auto v = d.codim_vertices[i];
            bid_mismatches += (r.codim_point_bids[i] == d.v2b[v]) ? 0 : 1;
            bid_mismatches += (r.codim_point_cids[i] == d.contact_ids[v]) ? 0 : 1;
        }
        CHECK(bid_mismatches == 0);
        note("info bid/cid arrays: %zu point + %zu edge + %zu triangle + %zu codim entries, "
             "%d mismatches vs v2b/contact_element_ids",
             r.point_bids.size(),
             r.edge_bids.size(),
             r.triangle_bids.size(),
             r.codim_point_bids.size(),
             bid_mismatches);
    }
    {
        auto base = make_simplex_impl();
        auto r    = impl_detect<gg::InfoStacklessBVHSimplexTrajectoryFilter>(base);
        CHECK(r.cand_allp_codimp == dcd_exact.allp_codimp);
        CHECK(r.cand_codimp_alle == dcd_exact.codimp_alle);
        CHECK(r.cand_alle_alle == dcd_exact.alle_alle);
        CHECK(r.cand_allp_allt == dcd_exact.allp_allt);
        note("DCD info_stackless_bvh (v1): %zu candidates (exact oracle %zu, box oracle "
             "%zu)",
             r.total_candidates(),
             dcd_exact.total(),
             box_sets.total());

        // v1 is a *subset* of the box flavour (the exact test is strictly tighter),
        // and the two differ on this scene -- the batch's one behavioural difference
        CHECK(dcd_exact.total() <= box_sets.total());
        CHECK(dcd_exact.allp_allt.size() <= box_sets.allp_allt.size());
        CHECK(dcd_exact.alle_alle.size() <= box_sets.alle_alle.size());
    }
}
