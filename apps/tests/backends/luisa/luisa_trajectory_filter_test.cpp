/********************************************************************
 * @file luisa_trajectory_filter_test.cpp
 * @brief T7.2a — the trajectory filters, batch 1: the two bases
 *        (`simplex_trajectory_filter`, `vertex_half_plane_trajectory_filter`), the
 *        manager (`global_trajectory_filter`) and the two concrete vertex-half-plane
 *        filters (LuisaCompute backend). Tag [filter].
 *
 * Coverage map (brief §Tests):
 * 1) scene/flag contracts + the base accessor surface ... t72a_filter_scene_contracts
 * 2) the Easy candidate production vs an independent host oracle (the band, the
 *    degenerate vertices, the two masks) .......... t72a_easy_candidate_oracle
 * 3) the half-plane verdicts: at / before / beyond / inside the thickness, the
 *    signed case and the two mask rejections ...... t72a_half_plane_verdicts
 * 4) the TOI stage, Easy vs AL (eta, the grid, the two 1.1 spellings)
 *    ................................................ t72a_toi_easy_and_al
 * 5) the base-class contracts: the pair-buffer capacity/retry, the friction copies,
 *    the FilteredInfo iteration order and the manager's global filter run
 *    ................................................ t72a_base_contracts
 * 6) the predicate-shape argument (selection vs pruning) + determinism + the
 *    device-vs-host round for the ported codim-thickness helpers
 *    ................................................ t72a_predicate_shape_and_determinism
 * 7) the quoted numbers ..................................... t72a_summary
 *
 * HOW THE SCENES ARE READ
 * -----------------------
 * Every expectation is computed independently in this file from data the test itself
 * fed into the scene, cross-checked against the *device* arrays read back through the
 * ported managers (never through the filter's own outputs).  The scene is the ordinary
 * frontend one: a `HalfPlane` implicit geometry + a `SimplicialComplex` mesh carrying
 * the shared `test_*` vertex attributes that the sibling luisa tests' reporters consume
 * (`test_dimension`, `test_contact_element_id`, `test_subscene_element_id`, `test_d_hat`,
 * `test_thickness`, `test_displacement`) — the same convention luisa_global_geometry_test
 * and luisa_half_plane_animator_test use, which is what makes the global vertex arrays
 * readable here without registering a reporter of our own.
 *
 * The filter pipeline itself is driven through `GlobalTrajectoryFilter::run_stage()`
 * (the documented T7.2a test seam: the reference's six stage entry points are private
 * and are called by the Newton loop, which the port's engine shell does not run yet --
 * Phase 8 / T8.4).
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <uipc/core/contact_tabular.h>
#include <uipc/core/subscene_tabular.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>

#include <type_define.h>
#include <frame_stream.h>
#include <collision_detection/atomic_counting_lbvh.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <collision_detection/filters/al_vertex_half_plane_trajectory_filter.h>
#include <collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <utils/codim_thickness.h>
// The test target's include path contains src/backends/none before src/backends/luisa,
// so the bare <sim_engine.h> would silently resolve to the none backend's shell
// (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
namespace gg = uipc::backend::luisa;
namespace lud = uipc::backend::luisa::luisa_device;

namespace t72a
{
namespace lcrt = ::luisa::compute;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;

// ===========================================================================
//  Helpers: read-back, sets, the mask/predicate oracles
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

using PairSet    = std::set<std::pair<IndexT, IndexT>>;
using PairVec    = std::vector<std::pair<IndexT, IndexT>>;
using PairVector = std::vector<Vector2i>;

PairVec to_pairs(const std::vector<Vector2i>& raw)
{
    PairVec out;
    out.reserve(raw.size());
    for(auto& p : raw)
        out.emplace_back(p[0], p[1]);
    return out;
}

PairSet to_pair_set(const std::vector<Vector2i>& raw)
{
    PairSet out;
    for(auto& p : raw)
        out.emplace(p[0], p[1]);
    return out;
}

/// The host re-implementation of the reference's `utils/codim_thickness.h`
/// `D_range` / `is_active_D` -- written out here on purpose (the ported header's own
/// device instantiation is cross-checked against this same formula in
/// t72a_predicate_shape_and_determinism).
struct DRange
{
    Float lo = 0.0;
    Float hi = 0.0;
};

DRange host_D_range(Float xi, Float d_hat)
{
    auto upper = xi + d_hat;
    auto lower = xi;
    return DRange{lower * lower, upper * upper};
}

bool host_is_active_D(const DRange& r, Float D)
{
    return r.lo < D && D < r.hi;
}

/// The three mask/band conditions of the reference's Easy `filter_active` kernel, over
/// the same *device* inputs the kernel consumed.
struct SceneData
{
    std::vector<IndexT>  surf_vertices;
    std::vector<Vector3> positions;
    std::vector<Float>   thicknesses;
    std::vector<IndexT>  contact_ids;
    std::vector<IndexT>  subscene_ids;
    std::vector<Float>   d_hats;
    std::vector<Vector3> displacements;

    std::vector<Vector3> plane_positions;
    std::vector<Vector3> plane_normals;
    IndexT               plane_vertex_offset = 0;

    std::vector<IndexT> contact_mask;
    IndexT              contact_mask_size = 0;
    std::vector<IndexT> subscene_mask;
    IndexT              subscene_mask_size = 0;

    [[nodiscard]] bool in_scope(IndexT vI, IndexT j) const
    {
        auto vJ = plane_vertex_offset + j;
        auto L  = contact_ids[vI];
        auto R  = contact_ids[vJ];
        auto sL = subscene_ids[vI];
        auto sR = subscene_ids[vJ];
        if(subscene_mask[sL * subscene_mask_size + sR] == 0)
            return false;
        if(contact_mask[L * contact_mask_size + R] == 0)
            return false;
        return true;
    }
};

/// `EasyVertexHalfPlaneTrajectoryFilter_filter_active_kernel`'s surviving set.
PairSet easy_candidate_oracle(const SceneData& d)
{
    PairSet out;
    for(auto vI : d.surf_vertices)
    {
        for(size_t j = 0; j < d.plane_positions.size(); ++j)
        {
            if(!d.in_scope(vI, static_cast<IndexT>(j)))
                continue;
            auto dst = (d.positions[vI] - d.plane_positions[j]).dot(d.plane_normals[j]);
            auto D   = dst * dst;
            auto band = host_D_range(d.thicknesses[vI], d.d_hats[vI]);
            if(host_is_active_D(band, D))
                out.emplace(vI, static_cast<IndexT>(j));
        }
    }
    return out;
}

/// The reference's dot product order (`s = a0*b0; s += a1*b1; s += a2*b2`), spelled out
/// so the oracle cannot be re-associated by an Eigen expression template.
Float dot3(const Vector3& a, const Vector3& b)
{
    Float s = a[0] * b[0];
    s += a[1] * b[1];
    s += a[2] * b[2];
    return s;
}

/// `EasyVertexHalfPlaneTrajectoryFilter_filter_toi_kernel`: one per-surface-vertex
/// minimum, `eta = 0.1`, the reference's `Float min_toi = 1.1f` initial value.
std::vector<Float> easy_toi_oracle(const SceneData& d, Float alpha)
{
    const Float eta    = 0.1;
    const Float init   = static_cast<Float>(1.1f);
    std::vector<Float> out;
    out.reserve(d.surf_vertices.size());
    for(auto vI : d.surf_vertices)
    {
        Float min_toi = init;
        for(size_t j = 0; j < d.plane_positions.size(); ++j)
        {
            if(!d.in_scope(vI, static_cast<IndexT>(j)))
                continue;
            auto x   = d.positions[vI];
            auto dx  = d.displacements[vI] * alpha;
            auto P   = d.plane_positions[j];
            auto N   = d.plane_normals[j];
            auto thickness = d.thicknesses[vI];

            auto t = -dot3(N, dx);
            if(!(t > 0.0))
                continue;  // moving away from the plane, no collision
            auto t0       = -dot3(N, P - x) - thickness;
            auto this_toi = t0 / t * (1.0 - eta);
            min_toi       = std::min(min_toi, this_toi);
        }
        out.push_back(min_toi);
    }
    return out;
}

/// `ALVertexHalfPlaneTrajectoryFilter_filter_toi_kernel`: the full
/// `surf_vertices x plane_instances` grid, `eta = 0.001`; a mask-skipped pair keeps the
/// caller's `tois.fill(1.1f)` value (the *float* literal), while a pair that moves away
/// is written as the *double* `1.1`.
struct AlOracle
{
    PairVector          phs;
    std::vector<Float>  tois;
};

AlOracle al_toi_oracle(const SceneData& d, Float alpha)
{
    const Float eta  = 0.001;
    const Float fill = static_cast<Float>(1.1f);  // `tois.fill(1.1f)`
    AlOracle    out;
    for(size_t i = 0; i < d.surf_vertices.size(); ++i)
    {
        auto vI = d.surf_vertices[i];
        for(size_t j = 0; j < d.plane_positions.size(); ++j)
        {
            out.phs.push_back(Vector2i{static_cast<int>(vI), static_cast<int>(j)});
            if(!d.in_scope(vI, static_cast<IndexT>(j)))
            {
                out.tois.push_back(fill);  // the reference's two `continue`s
                continue;
            }
            auto dx = d.displacements[vI] * alpha;
            auto N  = d.plane_normals[j];
            auto t  = -dot3(N, dx);
            if(!(t > 0.0))
            {
                out.tois.push_back(static_cast<Float>(1.1));  // the kernel's double 1.1
                continue;
            }
            auto x  = d.positions[vI];
            auto P  = d.plane_positions[j];
            auto t0 = -dot3(N, P - x) - d.thicknesses[vI];
            out.tois.push_back(t0 / t * (1.0 - eta));
        }
    }
    return out;
}

// ===========================================================================
//  Scene description + builder
// ===========================================================================
struct HPDesc
{
    Vector3 P{0.0, 0.0, 0.0};
    Vector3 N{0.0, 1.0, 0.0};
    IndexT  contact_id = 0;
    IndexT  subscene_id = 0;
};

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
};

struct SceneDesc
{
    std::vector<HPDesc>  half_planes;
    MeshDesc             mesh;
    bool                 has_mesh        = true;
    std::string          constitution    = "ipc";
    bool                 two_contact_elements  = false;
    bool                 contact_pair_disabled = false;
    bool                 two_subscenes         = false;
    bool                 cfl_enable            = false;
    Float                dt                    = 0.25;
};

constexpr std::string_view attr_dimension   = "test_dimension";
constexpr std::string_view attr_contact_id  = "test_contact_element_id";
constexpr std::string_view attr_subscene_id = "test_subscene_element_id";
constexpr std::string_view attr_d_hat       = "test_d_hat";
constexpr std::string_view attr_thickness   = "test_thickness";
constexpr std::string_view attr_displacement = "test_displacement";

Scene build_scene(const SceneDesc& desc)
{
    auto config                 = Scene::default_config();
    config["sanity_check"]["enable"]   = 0;
    config["dt"]                        = desc.dt;
    config["contact"]["constitution"]   = desc.constitution;
    config["cfl"]["enable"]             = desc.cfl_enable ? 1 : 0;
    Scene scene{config};

    if(desc.two_contact_elements)
    {
        auto& ct = scene.contact_tabular();
        auto  e1 = ct.create("t72a_b");
        if(desc.contact_pair_disabled)
            ct.insert(e1, e1, 0.5, 1.0_GPa, false, Json::object());
        else
            ct.insert(e1, e1, 0.5, 1.0_GPa, true, Json::object());
    }
    if(desc.two_subscenes)
    {
        auto& st = scene.subscene_tabular();
        st.create("t72a_s1");
        // the reference's default subscene mask is the identity: different subscenes
        // are disabled unless an entry says otherwise.
    }

    for(size_t i = 0; i < desc.half_planes.size(); ++i)
    {
        auto        object = scene.objects().create("t72a_ground_" + std::to_string(i));
        const auto& hp     = desc.half_planes[i];
        auto        ig     = halfplane(hp.P, hp.N);
        ig.meta().create<IndexT>(builtin::contact_element_id, hp.contact_id);
        ig.meta().create<IndexT>(builtin::subscene_element_id, hp.subscene_id);
        object->geometries().create(ig);
    }

    if(desc.has_mesh)
    {
        auto& m = desc.mesh;
        auto  complex = trimesh(span<const Vector3>{m.positions},
                               span<const Vector3i>{m.triangles});
        label_surface(complex);

        auto verts = complex.vertices();
        auto dimension   = verts.create<IndexT>(attr_dimension, 0);
        auto contact_id  = verts.create<IndexT>(attr_contact_id, 0);
        auto subscene_id = verts.create<IndexT>(attr_subscene_id, 0);
        auto d_hat       = verts.create<Float>(attr_d_hat, 0.0);
        auto thickness   = verts.create<Float>(attr_thickness, 0.0);
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

        auto object = scene.objects().create("t72a_mesh");
        object->geometries().create(complex);
    }

    return scene;
}

// ===========================================================================
//  Probe: hand the ported systems to the test (the sibling tests' pattern)
// ===========================================================================
class TrajectoryFilterTestProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    static gg::GlobalVertexManager*            vertex_manager;
    static gg::GlobalBodyManager*              body_manager;
    static gg::GlobalSimplicialSurfaceManager* surface_manager;
    static gg::GlobalContactManager*           contact_manager;
    static gg::HalfPlane*                      half_plane;
    static gg::HalfPlaneVertexReporter*        half_plane_vertex_reporter;
    static gg::GlobalTrajectoryFilter*         trajectory_filter;

  protected:
    void do_build() override
    {
        vertex_manager             = find<gg::GlobalVertexManager>();
        body_manager               = find<gg::GlobalBodyManager>();
        surface_manager            = find<gg::GlobalSimplicialSurfaceManager>();
        contact_manager            = find<gg::GlobalContactManager>();
        half_plane                 = find<gg::HalfPlane>();
        half_plane_vertex_reporter = find<gg::HalfPlaneVertexReporter>();
        trajectory_filter          = find<gg::GlobalTrajectoryFilter>();
    }
};
REGISTER_SIM_SYSTEM(TrajectoryFilterTestProbe);

gg::GlobalVertexManager*            TrajectoryFilterTestProbe::vertex_manager             = nullptr;
gg::GlobalBodyManager*              TrajectoryFilterTestProbe::body_manager               = nullptr;
gg::GlobalSimplicialSurfaceManager* TrajectoryFilterTestProbe::surface_manager            = nullptr;
gg::GlobalContactManager*           TrajectoryFilterTestProbe::contact_manager            = nullptr;
gg::HalfPlane*                      TrajectoryFilterTestProbe::half_plane                 = nullptr;
gg::HalfPlaneVertexReporter*        TrajectoryFilterTestProbe::half_plane_vertex_reporter = nullptr;
gg::GlobalTrajectoryFilter*         TrajectoryFilterTestProbe::trajectory_filter          = nullptr;

/// Everything the oracles need, read back from the ported managers after `world.init()`.
SceneData gather(const gg::GlobalTrajectoryFilter& gtf)
{
    auto* vm = TrajectoryFilterTestProbe::vertex_manager;
    auto* sm = TrajectoryFilterTestProbe::surface_manager;
    auto* cm = TrajectoryFilterTestProbe::contact_manager;
    auto* hp = TrajectoryFilterTestProbe::half_plane;
    auto* hr = TrajectoryFilterTestProbe::half_plane_vertex_reporter;

    SceneData d;
    d.surf_vertices   = read_device(sm->surf_vertices());
    d.positions       = read_vector3(vm->positions());
    d.thicknesses     = read_device(vm->thicknesses());
    d.contact_ids     = read_device(vm->contact_element_ids());
    d.subscene_ids    = read_device(vm->subscene_element_ids());
    d.d_hats          = read_device(vm->d_hats());
    d.displacements   = read_vector3(vm->displacements());
    d.plane_positions = read_vector3(hp->positions());
    d.plane_normals   = read_vector3(hp->normals());
    d.plane_vertex_offset = hr->vertex_offset();

    d.contact_mask      = read_device(cm->contact_mask_tabular());
    d.contact_mask_size = cm->contact_mask_size();
    d.subscene_mask      = read_device(cm->subscene_mask_tabular());
    d.subscene_mask_size = cm->subscene_mask_size();
    return d;
}

/// The base's own `Impl`, wired to the probe's managers, so the test can drive the
/// derived filters' public `Impl::filter_active()/filter_toi()` directly (the ported
/// `Impl` structs are public nested classes, exactly as in the reference).
gg::VertexHalfPlaneTrajectoryFilter::Impl make_vhp_impl()
{
    gg::VertexHalfPlaneTrajectoryFilter::Impl impl;
    impl.global_vertex_manager = TrajectoryFilterTestProbe::vertex_manager;
    impl.global_simplicial_surface_manager = TrajectoryFilterTestProbe::surface_manager;
    impl.global_contact_manager = TrajectoryFilterTestProbe::contact_manager;
    impl.half_plane = TrajectoryFilterTestProbe::half_plane;
    impl.half_plane_vertex_reporter = TrajectoryFilterTestProbe::half_plane_vertex_reporter;
    return impl;
}

/// The contact manager's CFL activation buffer, read through the *public*
/// `LabelActiveVerticesInfo` (its `Impl` pointer is a public member of a public nested
/// class, so the test can hand it its own `GlobalTrajectoryFilter::Impl`).
std::vector<IndexT> read_active_vertices()
{
    gg::GlobalTrajectoryFilter::Impl gtf_impl;
    gtf_impl.global_contact_manager = TrajectoryFilterTestProbe::contact_manager;
    gg::GlobalTrajectoryFilter::LabelActiveVerticesInfo info{&gtf_impl};
    return read_device(info.vert_is_active());
}

/// The test plays the role of the (Phase-8) reporters'/solver's fan-out that fills the
/// *global* displacement array the filters read through `DetectInfo::displacements()`
/// (`GlobalVertexManager::collect_vertex_displacements()` is private, and the ported
/// engine shell does not call it yet).  The buffer is otherwise zero at init time.
void set_displacements(const std::vector<Vector3>& per_vertex)
{
    auto* vm   = TrajectoryFilterTestProbe::vertex_manager;
    auto  view = vm->displacements();
    REQUIRE(view.size() == per_vertex.size());
    std::vector<gg::LVector3> host;
    host.reserve(per_vertex.size());
    for(auto& v : per_vertex)
        host.push_back(gg::to_luisa(v));
    view.copy_from(span<const gg::LVector3>{host}, &gg::frame_stream());
    gg::frame_stream().flush();
}

std::vector<Vector2i> read_phs(const gg::VertexHalfPlaneTrajectoryFilter& f)
{
    return read_vector2i(const_cast<gg::VertexHalfPlaneTrajectoryFilter&>(f).PHs());
}

std::vector<Float> read_tois(const lud::LCBufferView<Float>& view)
{
    return read_device(view);
}
}// namespace t72a

using namespace t72a;

// ===========================================================================
//  1) the scene, the flags and the base accessor surface
// ===========================================================================
TEST_CASE("t72a_filter_scene_contracts", "[filter]")
{
    test_runtime();

    SceneDesc desc;
    desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
    desc.mesh.positions = {{0.0, 0.0625, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.25, 0.0}, {1.0, 0.25, 0.0}};
    desc.mesh.triangles = {{0, 1, 2}, {1, 3, 2}};
    for(int i = 0; i < 4; ++i)
    {
        desc.mesh.thicknesses.push_back(0.0625);
        desc.mesh.d_hats.push_back(0.125);
    }

    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t72a_scene_contracts")};
    World  world{engine};
    world.init(scene);

    auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
    REQUIRE(gtf != nullptr);
    auto* vm = TrajectoryFilterTestProbe::vertex_manager;
    auto* sm = TrajectoryFilterTestProbe::surface_manager;
    auto* cm = TrajectoryFilterTestProbe::contact_manager;
    auto* hp = TrajectoryFilterTestProbe::half_plane;
    auto* hr = TrajectoryFilterTestProbe::half_plane_vertex_reporter;
    REQUIRE(vm != nullptr);
    REQUIRE(sm != nullptr);
    REQUIRE(cm != nullptr);
    REQUIRE(hp != nullptr);
    REQUIRE(hr != nullptr);

    // ---- the scene's implicit geometry ------------------------------------
    CHECK(hp->positions().size() == 1);
    CHECK(hp->normals().size() == 1);
    auto plane_positions = read_vector3(hp->positions());
    auto plane_normals   = read_vector3(hp->normals());
    REQUIRE(plane_positions.size() == 1);
    CHECK(plane_positions[0] == Vector3(0.0, 0.0, 0.0));
    CHECK(plane_normals[0] == Vector3(0.0, 1.0, 0.0));
    // the half plane's own vertex is the first slice of the global vertex arrays
    CHECK(hr->vertex_offset() == 0);
    CHECK(hr->vertex_count() == 1);

    // ---- the surface manager's vertex set ---------------------------------
    auto d = gather(*gtf);
    CHECK(d.surf_vertices.size() == 4);  // the mesh is the only SimplicialComplex
    std::set<IndexT> surf(d.surf_vertices.begin(), d.surf_vertices.end());
    CHECK(surf.size() == 4);
    CHECK(surf.count(0) == 0);  // the half plane's vertex is not a surface vertex
    for(size_t i = 0; i < 4; ++i)
        CHECK(surf.count(static_cast<IndexT>(1 + i)) == 1);
    CHECK(sm->surf_edges().size() == 5);      // the quad's 4 boundary + 1 diagonal
    CHECK(sm->surf_triangles().size() == 2);
    CHECK(sm->codim_vertices().size() == 0);

    // ---- the contact manager's attribute surface --------------------------
    CHECK(cm->d_hat() == 0.01);                    // the scene default
    CHECK(cm->contact_mask_size() == 1);           // one contact element
    CHECK(cm->subscene_mask_size() == 1);          // one subscene
    auto cmask = read_device(cm->contact_mask_tabular());
    auto smask = read_device(cm->subscene_mask_tabular());
    REQUIRE(cmask.size() == 1);
    REQUIRE(smask.size() == 1);
    CHECK(cmask[0] == 1);
    CHECK(smask[0] == 1);
    CHECK(cm->cfl_enabled() == false);  // `cfl/enable` defaults to 0

    // ---- the filter graph -------------------------------------------------
    auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
    REQUIRE(static_cast<bool>(easy_slot));
    auto al_slot = gtf->find<gg::ALVertexHalfPlaneTrajectoryFilter>();
    // the AL filter's do_build requires ALIPCPipelineFlag: in an "ipc" scene the engine's
    // SimSystemException handler invalidates it *before* it registers, so the manager's
    // registry holds exactly one filter (the reference's own scene-gating mechanism).
    CHECK(!static_cast<bool>(al_slot));
    auto base_slot = gtf->find<gg::VertexHalfPlaneTrajectoryFilter>();
    REQUIRE(static_cast<bool>(base_slot));
    CHECK(base_slot.view() == static_cast<gg::VertexHalfPlaneTrajectoryFilter*>(easy_slot.view()));

    // nothing has run yet
    CHECK(static_cast<gg::VertexHalfPlaneTrajectoryFilter*>(easy_slot.view())->PHs().empty());
    CHECK(static_cast<gg::VertexHalfPlaneTrajectoryFilter*>(easy_slot.view())->friction_PHs().empty());
    CHECK(easy_slot.view()->candidate_PHs().empty());
    CHECK(easy_slot.view()->toi_PHs().empty());

    // ---- the empty-candidate path through the real filter ------------------
    // No vertex of this scene is inside the band, so the stage produces nothing while
    // the kernel still runs (the padded one-element pair buffer is never written).
    CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive) == 1.0);
    {
        auto empty_oracle = easy_candidate_oracle(d);
        CHECK(empty_oracle.empty());
        CHECK(to_pair_set(read_phs(*easy_slot.view())).empty());
        // NOTE the reference's own asymmetry, transcribed: `candidate_PHs()` returns the
        // filter's *whole* pair buffer (whose size is the retry-grown capacity), while the
        // candidate view the base stored in `PHs()` is exactly the reported count.
        CHECK(easy_slot.view()->PHs().empty());          // reported: 0 candidates
        CHECK(easy_slot.view()->candidate_PHs().size() == 1);  // the padded buffer (T7.1a rule 9)
    }

    // ---- the two bases' `BaseInfo` accessors mirror the managers ------------
    // Both `Impl` structs are public nested classes (the reference's own layout), so a
    // standalone instance wired to the probe's managers exercises the whole accessor
    // surface without a scene-side seam.
    {
        auto vhp = make_vhp_impl();
        gg::VertexHalfPlaneTrajectoryFilter::BaseInfo base{&vhp};
        CHECK(base.d_hat() == cm->d_hat());
        CHECK(base.d_hats().size() == vm->d_hats().size());
        CHECK(base.positions().size() == vm->positions().size());
        CHECK(base.thicknesses().size() == vm->thicknesses().size());
        CHECK(base.contact_element_ids().size() == vm->contact_element_ids().size());
        CHECK(base.subscene_element_ids().size() == vm->subscene_element_ids().size());
        CHECK(base.surf_vertices().size() == sm->surf_vertices().size());
        CHECK(base.half_plane_vertex_offset() == hr->vertex_offset());
        CHECK(base.plane_positions().size() == 1);
        CHECK(base.plane_normals().size() == 1);
        CHECK(base.contact_mask_tabular().size() == cmask.size());
        CHECK(base.subscene_mask_tabular().size() == smask.size());
        CHECK(base.contact_mask_size() == cm->contact_mask_size());
        CHECK(base.subscene_mask_size() == cm->subscene_mask_size());

        gg::SimplexTrajectoryFilter::Impl simplex;
        simplex.global_vertex_manager = *vm;
        simplex.global_simplicial_surface_manager = *sm;
        simplex.global_contact_manager = *cm;
        simplex.global_body_manager = *TrajectoryFilterTestProbe::body_manager;
        CHECK(simplex.toi_safety_margin == 0.1);  // the reference's default
        gg::SimplexTrajectoryFilter::BaseInfo sbase{&simplex};
        CHECK(sbase.d_hat() == cm->d_hat());
        CHECK(sbase.toi_safety_margin() == 0.1);
        CHECK(sbase.d_hats().size() == vm->d_hats().size());
        CHECK(sbase.v2b().size() == vm->body_ids().size());
        CHECK(sbase.positions().size() == vm->positions().size());
        CHECK(sbase.rest_positions().size() == vm->rest_positions().size());
        CHECK(sbase.thicknesses().size() == vm->thicknesses().size());
        CHECK(sbase.dimensions().size() == vm->dimensions().size());
        CHECK(sbase.contact_element_ids().size() == vm->contact_element_ids().size());
        CHECK(sbase.subscene_element_ids().size() == vm->subscene_element_ids().size());
        CHECK(sbase.contact_mask_tabular().size() == cmask.size());
        CHECK(sbase.subscene_mask_tabular().size() == smask.size());
        CHECK(sbase.contact_mask_size() == cm->contact_mask_size());
        CHECK(sbase.subscene_mask_size() == cm->subscene_mask_size());
        CHECK(sbase.body_self_collision().size()
              == TrajectoryFilterTestProbe::body_manager->self_collision().size());
        CHECK(sbase.codim_vertices().size() == sm->codim_vertices().size());
        CHECK(sbase.surf_vertices().size() == sm->surf_vertices().size());
        CHECK(sbase.surf_edges().size() == sm->surf_edges().size());
        CHECK(sbase.surf_triangles().size() == sm->surf_triangles().size());
        gg::VertexHalfPlaneTrajectoryFilter::DetectInfo dinfo{&vhp};
        CHECK(dinfo.alpha() == 0.0);
        CHECK(dinfo.displacements().size() == vm->displacements().size());
        gg::SimplexTrajectoryFilter::DetectInfo sinfo{&simplex};
        CHECK(sinfo.alpha() == 0.0);
        CHECK(sinfo.displacements().size() == vm->displacements().size());
    }
}

// ===========================================================================
//  2) the Easy candidate production vs the independent host oracle
// ===========================================================================
TEST_CASE("t72a_easy_candidate_oracle", "[filter]")
{
    test_runtime();

    // thickness = 1/16, d_hat = 1/8 -> the active band is |dst| in (0.0625, 0.1875):
    //  *  y =  0.0625        -> D == thickness^2 exactly: NOT active (strict)
    //  *  y =  0.125         -> inside the band                      : active
    //  *  y =  0.0           -> D == 0 < thickness^2                 : NOT active
    //  *  y =  0.25          -> D == 0.0625 > (0.1875)^2             : NOT active
    //  *  y = -0.125         -> the signed case, D == dst^2          : active
    //  *  y =  0.0625+1/1024 -> just inside the band                 : active
    SceneDesc desc;
    desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
    desc.mesh.positions = {{0.0, 0.0625, 0.0},
                           {1.0, 0.125, 0.0},
                           {0.0, 0.0, 0.0},
                           {1.0, 0.25, 0.0},
                           {0.0, -0.125, 0.0},
                           {1.0, 0.0625 + 1.0 / 1024.0, 0.0}};
    desc.mesh.triangles = {{0, 1, 2}, {1, 3, 2}, {2, 1, 4}, {3, 5, 2}};
    for(int i = 0; i < 6; ++i)
    {
        desc.mesh.thicknesses.push_back(0.0625);
        desc.mesh.d_hats.push_back(0.125);
    }

    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t72a_easy_oracle")};
    World  world{engine};
    world.init(scene);

    auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
    REQUIRE(gtf != nullptr);
    auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
    REQUIRE(static_cast<bool>(easy_slot));

    auto d = gather(*gtf);
    std::set<IndexT> surf(d.surf_vertices.begin(), d.surf_vertices.end());
    CHECK(surf.size() == 6);
    REQUIRE(d.plane_positions.size() == 1);
    // the mesh's vertices are global ids 1..6 (0 is the half plane's own vertex)
    for(size_t i = 0; i < 6; ++i)
        CHECK(surf.count(static_cast<IndexT>(1 + i)) == 1);
    // the attribute surface the oracle reads
    CHECK(d.thicknesses[1] == 0.0625);
    CHECK(d.d_hats[1] == 0.125);
    CHECK(d.contact_ids[1] == 0);
    CHECK(d.subscene_ids[1] == 0);

    // ---- drive the stage through the manager ------------------------------
    CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive) == 1.0);

    auto got  = to_pair_set(read_phs(*easy_slot.view()));
    auto want = easy_candidate_oracle(d);
    CHECK(got.size() == want.size());
    CHECK(got == want);

    // The hand-computed verdicts the oracle encodes, spelled out (global vertex id 1+idx):
    //   idx 0: y=+0.0625  ==  thickness         -> rejected (D == thickness^2)
    //   idx 1: y=+0.125   inside the band       -> accepted
    //   idx 2: y= 0       on the plane          -> rejected (D == 0)
    //   idx 3: y=+0.25    beyond thickness+d_hat -> rejected
    //   idx 4: y=-0.125   before the plane      -> accepted (signed case)
    //   idx 5: y=+0.0625+1/1024 just inside     -> accepted
    auto sv = d.surf_vertices;  // sorted: the surface reporter pushes in geometry order
    std::sort(sv.begin(), sv.end());
    REQUIRE(sv.size() == 6);
    CHECK(want.count({sv[0], 0}) == 0);  // y=+0.0625 == thickness  -> rejected
    CHECK(want.count({sv[1], 0}) == 1);  // y=+0.125  inside      -> accepted
    CHECK(want.count({sv[2], 0}) == 0);  // y= 0      on the plane -> rejected
    CHECK(want.count({sv[3], 0}) == 0);  // y=+0.25   beyond       -> rejected
    CHECK(want.count({sv[4], 0}) == 1);  // y=-0.125  before       -> accepted
    CHECK(want.count({sv[5], 0}) == 1);  // just inside the band   -> accepted
    CHECK(want.size() == 3);
    // the reported count is the *selected* count; the filter's own buffer was grown to
    // `static_cast<size_t>(count * reserve_ratio(1.1)) == 3` by the retry...
    CHECK(easy_slot.view()->candidate_PHs().size() == 3);
    // ... while the *stored* candidate view is exactly `h_num_collisions` entries
    CHECK(easy_slot.view()->PHs().size() == want.size());

    // ---- the distance values the verdicts follow --------------------------
    CHECK(std::abs(d.positions[2][1] - 0.125) < 1e-15);
    CHECK(std::abs(d.positions[5][1] + 0.125) < 1e-15);
    auto band1 = host_D_range(d.thicknesses[1], d.d_hats[1]);
    CHECK(band1.lo == 0.0625 * 0.0625);
    CHECK(band1.hi == (0.0625 + 0.125) * (0.0625 + 0.125));
    CHECK(host_is_active_D(band1, 0.125 * 0.125) == true);
    CHECK(host_is_active_D(band1, 0.0625 * 0.0625) == false);
    CHECK(host_is_active_D(band1, 0.0) == false);

    std::printf("[filter] easy candidates: %zu (brute oracle %zu) over %zu surf vertices\n",
                got.size(),
                want.size(),
                d.surf_vertices.size());
}

// ===========================================================================
//  3) the half-plane verdicts: signed, thickness, and the two mask rejections
// ===========================================================================
TEST_CASE("t72a_half_plane_verdicts", "[filter]")
{
    test_runtime();

    // (a) the band with two half planes at different offsets: the candidate set is the
    //     union over planes, and every pair carries the *plane index*, not a vertex id
    {
        SceneDesc desc;
        desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
        desc.half_planes.push_back(HPDesc{.P = {0.0, 1.0, 0.0}, .N = {0.0, 1.0, 0.0}});
        desc.mesh.positions = {{0.0, 0.125, 0.0}, {1.0, 1.125, 0.0}, {0.0, 0.5, 0.0}, {1.0, 2.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}, {1, 3, 2}};
        for(int i = 0; i < 4; ++i)
        {
            desc.mesh.thicknesses.push_back(0.0625);
            desc.mesh.d_hats.push_back(0.125);
        }

        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_t72a_verdicts_a")};
        World  world{engine};
        world.init(scene);

        auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
        REQUIRE(gtf != nullptr);
        auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
        REQUIRE(static_cast<bool>(easy_slot));

        auto d = gather(*gtf);
        REQUIRE(d.plane_positions.size() == 2);
        CHECK(d.plane_positions[0] == Vector3(0.0, 0.0, 0.0));
        CHECK(d.plane_positions[1] == Vector3(0.0, 1.0, 0.0));
        CHECK(d.surf_vertices.size() == 4);

        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        auto got  = to_pair_set(read_phs(*easy_slot.view()));
        auto want = easy_candidate_oracle(d);
        CHECK(got.size() == want.size());
        CHECK(got == want);
        // vertex 1 (y=0.125) is in plane 0's band and *outside* plane 1's band
        // (|0.125-1| = 0.875); vertex 2 (y=1.125) is in plane 1's band; vertex 3
        // (y=0.5) is in neither (0.5 and 0.5 -> both == 0.5, outside (0.0625,0.1875))
        // and vertex 4 (y=2.0) is in neither.
        auto sv = d.surf_vertices;
        std::sort(sv.begin(), sv.end());
        REQUIRE(sv.size() == 4);
        // the two half-plane vertices occupy global ids 0 and 1, the mesh follows
        CHECK(sv[0] == 2);
        CHECK(d.plane_vertex_offset == 0);
        CHECK(want.count({sv[0], 0}) == 1);  // y=0.125 in plane 0's band
        CHECK(want.count({sv[0], 1}) == 0);  //      ... 0.875 from plane 1
        CHECK(want.count({sv[1], 1}) == 1);  // y=1.125 in plane 1's band
        CHECK(want.count({sv[1], 0}) == 0);
        CHECK(want.count({sv[2], 0}) == 0);  // y=0.5 in neither
        CHECK(want.count({sv[2], 1}) == 0);
        CHECK(want.count({sv[3], 0}) == 0);  // y=2.0 in neither
        CHECK(want.count({sv[3], 1}) == 0);
        CHECK(want.size() == 2);
        std::printf("[filter] two-plane band: %zu candidates over %zu planes\n",
                    got.size(),
                    d.plane_positions.size());
    }

    // (b) a disabled contact-model pair: the vertex's contact element id matches the
    //     half plane's but the mask entry is 0 -> no candidate at all
    {
        SceneDesc desc;
        desc.two_contact_elements    = true;
        desc.contact_pair_disabled   = true;
        desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}, .contact_id = 1});
        desc.mesh.positions = {{0.0, 0.125, 0.0}, {1.0, 0.125, 0.0}, {0.0, 0.5, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        for(int i = 0; i < 3; ++i)
        {
            desc.mesh.thicknesses.push_back(0.0625);
            desc.mesh.d_hats.push_back(0.125);
            desc.mesh.contact_ids.push_back(1);
        }

        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_t72a_verdicts_b")};
        World  world{engine};
        world.init(scene);

        auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
        REQUIRE(gtf != nullptr);
        auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
        REQUIRE(static_cast<bool>(easy_slot));

        auto d = gather(*gtf);
        CHECK(d.contact_mask_size == 2);
        CHECK(d.contact_mask.size() == 4);
        CHECK(d.contact_mask[0 * 2 + 0] == 1);  // the default pair stays enabled
        CHECK(d.contact_mask[1 * 2 + 1] == 0);  // the disabled pair
        // every *other* entry keeps the default model's flag (the reference's
        // `resize(N*N, enabled_view[0])` seed), i.e. the (0,1) pair is still enabled
        CHECK(d.contact_mask[0 * 2 + 1] == 1);
        CHECK(d.contact_mask[1 * 2 + 0] == 1);
        CHECK(d.contact_ids[d.plane_vertex_offset] == 1);  // the plane's own vertex
        CHECK(d.contact_ids[d.surf_vertices[0]] == 1);     // and the mesh's

        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        auto got  = to_pair_set(read_phs(*easy_slot.view()));
        auto want = easy_candidate_oracle(d);
        CHECK(want.empty());  // the mask rejects every pair
        CHECK(got.empty());
        CHECK(got == want);
        // control: the vertex really is inside the band (the mask is what rejected it)
        CHECK(d.positions[d.surf_vertices[0]][1] == 0.125);
    }

    // (c) two subscenes: the reference's default subscene mask is the identity, so a
    //     plane in subscene 1 and a mesh in subscene 0 never meet
    {
        SceneDesc desc;
        desc.two_subscenes = true;
        desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}, .subscene_id = 1});
        desc.mesh.positions = {{0.0, 0.125, 0.0}, {1.0, 0.125, 0.0}, {0.0, 0.5, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        for(int i = 0; i < 3; ++i)
        {
            desc.mesh.thicknesses.push_back(0.0625);
            desc.mesh.d_hats.push_back(0.125);
            desc.mesh.subscene_ids.push_back(0);  // the mesh stays in subscene 0
        }

        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_t72a_verdicts_c")};
        World  world{engine};
        world.init(scene);

        auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
        REQUIRE(gtf != nullptr);
        auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
        REQUIRE(static_cast<bool>(easy_slot));

        auto d = gather(*gtf);
        CHECK(d.subscene_mask_size == 2);
        CHECK(d.subscene_mask.size() == 4);
        CHECK(d.subscene_mask[0 * 2 + 0] == 1);  // self subscene contact is enabled
        CHECK(d.subscene_mask[0 * 2 + 1] == 0);  // different subscenes are not
        CHECK(d.subscene_ids[d.plane_vertex_offset] == 1);  // the plane
        CHECK(d.subscene_ids[d.surf_vertices[0]] == 0);     // the mesh

        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        auto got  = to_pair_set(read_phs(*easy_slot.view()));
        auto want = easy_candidate_oracle(d);
        CHECK(want.empty());
        CHECK(got.empty());
        CHECK(got == want);
    }
}

// ===========================================================================
//  4) the TOI stage: Easy vs AL
// ===========================================================================
TEST_CASE("t72a_toi_easy_and_al", "[filter]")
{
    test_runtime();

    const Float alpha = 0.5;

    // (a) the IPC pipeline builds the Easy filter; every vertex moves down by a
    //     dyadic amount so the arithmetic below is exact
    {
        SceneDesc desc;
        desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
        desc.mesh.positions = {{0.0, 0.25, 0.0},
                               {1.0, 0.5, 0.0},
                               {0.0, 0.125, 0.0},
                               {1.0, 1.0, 0.0},
                               {0.0, 0.75, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}, {1, 3, 2}, {2, 3, 4}};
        // four vertices move towards the plane at different speeds, one moves away, and
        // one arrives so slowly that its raw toi (19.75) is above the kernel's 1.1f
        // sentinel -- which is the value the report then shows (a reference quirk the
        // hand-computed checks below pin).
        std::vector<Vector3> disp = {{0.0, -0.5, 0.0},
                                     {0.0, -1.0, 0.0},
                                     {0.0, -0.125, 0.0},
                                     {0.0, 0.125, 0.0},
                                     {0.0, -0.0625, 0.0}};
        for(int i = 0; i < 5; ++i)
        {
            desc.mesh.thicknesses.push_back(0.0625);
            desc.mesh.d_hats.push_back(0.125);
            desc.mesh.displacements.push_back(disp[i]);
        }

        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_t72a_toi_easy")};
        World  world{engine};
        world.init(scene);

        auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
        REQUIRE(gtf != nullptr);
        auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
        REQUIRE(static_cast<bool>(easy_slot));
        {
            // global order: the half plane's vertex first, then the mesh's four
            std::vector<Vector3> gdisp = {Vector3::Zero()};
            for(auto& v : disp)
                gdisp.push_back(v);
            set_displacements(gdisp);
        }
        auto d = gather(*gtf);

        REQUIRE(d.surf_vertices.size() == 5);
        // the displacement array the kernels consume is the *global* one
        CHECK(std::abs(d.displacements[1][1] + 0.5) < 1e-15);
        CHECK(std::abs(d.displacements[2][1] + 1.0) < 1e-15);
        CHECK(std::abs(d.displacements[3][1] + 0.125) < 1e-15);
        CHECK(std::abs(d.displacements[4][1] - 0.125) < 1e-15);
        CHECK(std::abs(d.displacements[5][1] + 0.0625) < 1e-15);

        auto stepped = gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, alpha);
        auto want    = easy_toi_oracle(d, alpha);
        auto got     = read_tois(easy_slot.view()->toi_PHs());

        REQUIRE(got.size() == want.size());
        CHECK(got.size() == d.surf_vertices.size());
        for(size_t i = 0; i < got.size(); ++i)
        {
            INFO("vertex index " << i);
            CHECK(got[i] == want[i]);
        }

        // The hand-computed verdicts, looked up by the vertex's plane distance (the
        // surface array's order is the reporters', not the test's):
        //   y=0.25, dx=-0.25    -> t = 0.25,   t0 = 0.1875 -> 0.75  * 0.9 = 0.675
        //   y=0.5,  dx=-0.5     -> t = 0.5,    t0 = 0.4375 -> 0.875 * 0.9 = 0.7875
        //   y=0.125,dx=-0.0625  -> t = 0.0625, t0 = 0.0625 -> 1.0   * 0.9 = 0.9
        //   y=1.0,  dx=+0.0625  -> moves away: the kernel's `t <= 0` skip keeps 1.1f
        //   y=0.75, dx=-0.03125 -> raw toi 19.8 > 1.1f: the *sentinel* wins
        auto index_of_y = [&](Float y) -> size_t
        {
            for(size_t i = 0; i < d.surf_vertices.size(); ++i)
                if(d.positions[d.surf_vertices[i]][1] == y)
                    return i;
            return d.surf_vertices.size();
        };
        REQUIRE(index_of_y(0.25) < d.surf_vertices.size());
        REQUIRE(index_of_y(0.5) < d.surf_vertices.size());
        REQUIRE(index_of_y(0.125) < d.surf_vertices.size());
        REQUIRE(index_of_y(1.0) < d.surf_vertices.size());
        REQUIRE(index_of_y(0.75) < d.surf_vertices.size());
        CHECK(want[index_of_y(0.25)] == (0.1875 / 0.25) * (1.0 - 0.1));
        CHECK(want[index_of_y(0.5)] == (0.4375 / 0.5) * (1.0 - 0.1));
        CHECK(want[index_of_y(0.125)] == (0.0625 / 0.0625) * (1.0 - 0.1));
        CHECK(want[index_of_y(1.0)] == static_cast<Float>(1.1f));
        // the sentinel: the raw toi is 19.8, but `min_toi` starts at the reference's own
        // `1.1f`, so any larger toi is reported as 1.1f
        CHECK(want[index_of_y(0.75)] == static_cast<Float>(1.1f));
        CHECK((0.6875 / 0.03125) * (1.0 - 0.1) > 1.1);

        // the manager reduces the per-filter minima and floors at 1.0
        CHECK(stepped == std::min(want[index_of_y(0.25)], Float{1.0}));
        // while the *filter's* own reduced slot holds the unfloored minimum
        CHECK(want[index_of_y(0.25)] < 1.0);
        std::printf("[filter] easy toi: y=0.25 %g, y=0.5 %g, y=0.125 %g, y=1.0(away) %g, "
                    "y=0.75(sentinel) %g, manager step %g\n",
                    want[index_of_y(0.25)],
                    want[index_of_y(0.5)],
                    want[index_of_y(0.125)],
                    want[index_of_y(1.0)],
                    want[index_of_y(0.75)],
                    stepped);
    }

    // (b) the AL pipeline: the AL filter is the live one, its candidate grid is the full
    //     `surf x plane` product, and its slackness is 0.001 instead of 0.1
    {
        SceneDesc desc;
        desc.constitution = "al-ipc";
        desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
        desc.mesh.positions = {{0.0, 0.25, 0.0}, {1.0, 0.5, 0.0}, {0.0, 0.125, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        std::vector<Vector3> disp = {{0.0, -0.5, 0.0}, {0.0, -0.25, 0.0}, {0.0, 0.0625, 0.0}};
        for(int i = 0; i < 3; ++i)
        {
            desc.mesh.thicknesses.push_back(0.0625);
            desc.mesh.d_hats.push_back(0.125);
            desc.mesh.displacements.push_back(disp[i]);
        }

        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_t72a_toi_al")};
        World  world{engine};
        world.init(scene);

        auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
        REQUIRE(gtf != nullptr);
        {
            std::vector<Vector3> gdisp = {Vector3::Zero()};
            for(auto& v : disp)
                gdisp.push_back(v);
            set_displacements(gdisp);
        }
        auto al_slot = gtf->find<gg::ALVertexHalfPlaneTrajectoryFilter>();
        REQUIRE(static_cast<bool>(al_slot));
        auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
        CHECK(!static_cast<bool>(easy_slot));  // the mirror of case 1's flag check

        auto d = gather(*gtf);
        REQUIRE(d.surf_vertices.size() == 3);
        REQUIRE(d.plane_positions.size() == 1);

        // the AL filter's filter_active is the reference's explicit no-op: the candidate
        // grid is *not* produced by the active stage
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        CHECK(al_slot.view()->candidate_PHs().empty());

        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, alpha);
        auto got_phs = to_pairs(read_vector2i(al_slot.view()->candidate_PHs()));
        auto want    = al_toi_oracle(d, alpha);
        REQUIRE(got_phs.size() == want.phs.size());
        CHECK(got_phs.size() == 3);  // 3 surface vertices x 1 plane
        for(size_t i = 0; i < got_phs.size(); ++i)
        {
            CHECK(got_phs[i].first == want.phs[i][0]);
            CHECK(got_phs[i].second == want.phs[i][1]);
        }
        auto got_tois = read_tois(al_slot.view()->toi_PHs());
        REQUIRE(got_tois.size() == want.tois.size());
        for(size_t i = 0; i < got_tois.size(); ++i)
        {
            INFO("pair index " << i);
            CHECK(got_tois[i] == want.tois[i]);
        }
        // hand-computed: vertex 0: t = 0.25, t0 = 0.1875 -> 0.75 * 0.999 = 0.74925
        CHECK(want.tois[0] == (0.1875 / 0.25) * (1.0 - 0.001));
        // vertex 2 moves away (dx.y = +0.03125): t < 0 -> the kernel's double 1.1
        CHECK(want.tois[2] == static_cast<Float>(1.1));
        CHECK(want.tois[2] != static_cast<Float>(1.1f));
        std::printf("[filter] al toi grid: %zu pairs, toi[0]=%g toi[2]=%g (1.1f=%g)\n",
                    got_tois.size(),
                    want.tois[0],
                    want.tois[2],
                    static_cast<double>(1.1f));
    }
}

// ===========================================================================
//  5) the base-class contracts: capacity/retry, the friction copies, the manager's run
// ===========================================================================
TEST_CASE("t72a_base_contracts", "[filter]")
{
    test_runtime();

    SceneDesc desc;
    desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
    desc.mesh.positions = {{0.0, 0.125, 0.0}, {1.0, 0.125, 0.0}, {0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}};
    desc.mesh.triangles = {{0, 1, 2}, {1, 3, 2}};
    for(int i = 0; i < 4; ++i)
    {
        desc.mesh.thicknesses.push_back(0.0625);
        desc.mesh.d_hats.push_back(0.125);
    }

    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t72a_base_contracts")};
    World  world{engine};
    world.init(scene);

    auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
    REQUIRE(gtf != nullptr);
    auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
    REQUIRE(static_cast<bool>(easy_slot));
    auto d  = gather(*gtf);
    auto want = easy_candidate_oracle(d);
    REQUIRE(want.size() == 2);  // the two y=0.125 vertices sit in the band

    // ---- the manager's global filter run over its registry -----------------
    // `Stage::FilterActive` fans out over exactly the registered filters (here: one).
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
    auto manager_view = to_pair_set(read_phs(*easy_slot.view()));
    CHECK(manager_view == want);
    CHECK(easy_slot.view()->candidate_PHs().size() == want.size());

    // ---- the standalone `Impl` path: the capacity/retry contract -------------
    // The kernel takes its ticket *before* the `last < max_count` guard, so
    // `h_num_collisions` counts every surviving pair; the host therefore knows the true
    // count and, when it exceeds `PHs.size()`, grows the buffer to
    // `h_num_collisions * reserve_ratio` and re-runs the *same* kernel from ticket 0.
    // The test drives `reserve_ratio = 2.0` so every quoted size below is exact.
    auto vhp = make_vhp_impl();  // the *base's* Impl, wired to the probe's managers
    gg::VertexHalfPlaneTrajectoryFilter::FilterActiveInfo active_info{&vhp};
    {
        // (i) a buffer that already covers the count: no growth, stable repeat calls
        gg::EasyVertexHalfPlaneTrajectoryFilter::Impl easy;
        easy.reserve_ratio = 2.0;
        easy.PHs.resize(8);
        easy.filter_active(active_info);
        CHECK(easy.h_num_collisions == 2);
        CHECK(easy.PHs.size() == 8);  // untouched: 2 <= 8
        CHECK(vhp.PHs.size() == 2);   // the reported view is (0, h_num_collisions)
        CHECK(to_pair_set(read_vector2i(vhp.PHs)) == want);
        easy.filter_active(active_info);
        CHECK(easy.h_num_collisions == 2);
        CHECK(easy.PHs.size() == 8);
        CHECK(vhp.PHs.size() == 2);
        CHECK(to_pair_set(read_vector2i(vhp.PHs)) == want);
    }
    {
        // (ii) the reference's empty-buffer start: the retry
        gg::EasyVertexHalfPlaneTrajectoryFilter::Impl easy;
        easy.reserve_ratio = 2.0;
        CHECK(easy.PHs.empty());
        easy.filter_active(active_info);
        CHECK(easy.h_num_collisions == 2);
        CHECK(easy.PHs.size() == 4);  // 2 * reserve_ratio(2.0)
        CHECK(vhp.PHs.size() == 2);
        // the retry *rewrote* the buffer from ticket 0: two distinct pairs, no loss
        auto retried = to_pair_set(read_vector2i(vhp.PHs));
        CHECK(retried == want);
        CHECK(retried.size() == 2);
        // and the second call finds the grown buffer sufficient
        easy.filter_active(active_info);
        CHECK(easy.PHs.size() == 4);
        CHECK(easy.h_num_collisions == 2);
    }

    // ---- the friction-candidate contract (VertexHalfPlaneTrajectoryFilter) ----
    // `run_stage(RecordFrictionCandidates)`: the manager copies every registered
    // filter's candidate view into its friction buffer.
    CHECK(easy_slot.view()->friction_PHs().empty());
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::RecordFrictionCandidates);
    auto friction = to_pair_set(read_vector2i(easy_slot.view()->friction_PHs()));
    CHECK(friction == want);
    CHECK(easy_slot.view()->friction_PHs().size() == want.size());

    // `require_discard_friction()` (the reference's issue #303 path): the *next* stage
    // clears the friction candidates, resets the flag and records nothing.
    gtf->require_discard_friction();
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::RecordFrictionCandidates);
    CHECK(static_cast<gg::VertexHalfPlaneTrajectoryFilter*>(easy_slot.view())->friction_PHs().empty());
    // and the *second* call records again (the flag was consumed)
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::RecordFrictionCandidates);
    CHECK(easy_slot.view()->friction_PHs().size() == want.size());
    CHECK(to_pair_set(read_vector2i(easy_slot.view()->friction_PHs())) == want);

    // ---- `Stage::ClearFrictionCandidates` ---------------------------------
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::ClearFrictionCandidates);
    CHECK(easy_slot.view()->friction_PHs().empty());

    // ---- the CFL active-vertex labelling ----------------------------------
    // `cfl/enable` is 0 by default, so `filter_active` does *not* clear the contact
    // manager's activation buffer (the reference clears it only when CFL is on).
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::LabelActiveVertices);
    auto active = read_active_vertices();
    std::set<IndexT> active_set;
    for(size_t i = 0; i < active.size(); ++i)
        if(active[i] != 0)
            active_set.insert(static_cast<IndexT>(i));
    for(auto& p : want)
        CHECK(active_set.count(p.first) == 1);

    // ---- the SimplexTrajectoryFilter::Impl friction/recovery contract ------
    // (the base's Impl is a public nested class; the four candidate arrays are views the
    // derived filter would have produced)
    gg::SimplexTrajectoryFilter::Impl simplex;
    lud::LDeviceBuffer<gg::LVector4i> pts;
    lud::LDeviceBuffer<gg::LVector3i> pes;
    pts.resize(3);
    pes.resize(2);
    {
        std::vector<gg::LVector4i> h_pts(3);
        for(int i = 0; i < 3; ++i)
        {
            h_pts[i](0) = i;
            h_pts[i](1) = 100 + i;
            h_pts[i](2) = 200 + i;
            h_pts[i](3) = 300 + i;
        }
        pts.view().copy_from(span<const gg::LVector4i>{h_pts});
        std::vector<gg::LVector3i> h_pes(2);
        for(int i = 0; i < 2; ++i)
        {
            h_pes[i](0) = 10 + i;
            h_pes[i](1) = 20 + i;
            h_pes[i](2) = 30 + i;
        }
        pes.view().copy_from(span<const gg::LVector3i>{h_pes});
    }
    simplex.PTs = pts.cview();
    simplex.PEs = pes.cview();
    simplex.reserve_ratio = 2.0;
    gg::GlobalTrajectoryFilter::RecordFrictionCandidatesInfo record_info;
    simplex.record_friction_candidates(record_info);
    CHECK(simplex.friction_PT.size() == 3);
    CHECK(simplex.friction_PE.size() == 2);
    CHECK(simplex.friction_EE.size() == 0);  // EEs/PPs were never set by the "filter"
    CHECK(simplex.friction_PP.size() == 0);
    {
        auto h = read_device(simplex.friction_PT.cview());
        REQUIRE(h.size() == 3);
        for(int i = 0; i < 3; ++i)
        {
            CHECK(h[i](0) == i);
            CHECK(h[i](1) == 100 + i);
            CHECK(h[i](2) == 200 + i);
            CHECK(h[i](3) == 300 + i);
        }
    }
    // the recovery views start empty and `clear_recover` only drops the dumps
    CHECK(simplex.recovered_PT.size() == 0);
    CHECK(simplex.recovered_EE.size() == 0);
    CHECK(simplex.recovered_PE.size() == 0);
    CHECK(simplex.recovered_PP.size() == 0);
    // (`dump`/`try_recover`/`apply_recover`/`clear_recover` take the engine's
    // `DumpInfo`/`RecoverInfo`, which only the engine's dump path can build -- Phase 8.
    // The friction copies above are the part of the contract this batch can drive.)
    std::printf("[filter] base contracts: candidates=%zu friction=%zu\n",
                want.size(),
                easy_slot.view()->friction_PHs().size());
}

// ===========================================================================
//  6) the predicate-shape argument, determinism, device-vs-host
// ===========================================================================
namespace t72a
{
/// `AtomicCountingLBVH::detect` with a keep-all predicate: the traversal's candidate set.
PairSet lbvh_detect_all(const std::vector<gg::AABB>& boxes, lud::LDeviceBuffer<gg::LAABB>& dev)
{
    gg::AtomicCountingLBVH bvh;
    gg::AtomicCountingLBVH::QueryBuffer qbuffer;
    bvh.build(dev.view());
    bvh.detect([](const auto&, const auto&) noexcept { return true; }, qbuffer);
    gg::frame_stream().flush();
    return to_pair_set(read_vector2i(qbuffer.view()));
}

/// The same traversal with a *selective* predicate: the surviving set must be exactly
/// the keep-all set filtered by the predicate, because the predicate is applied inside
/// the traversal's *callback* -- after the fixed AABB-AABB node test -- so it cannot
/// change which nodes are visited (it is a selection, not a pruning).
PairSet lbvh_detect_even(const std::vector<gg::AABB>& boxes, lud::LDeviceBuffer<gg::LAABB>& dev)
{
    gg::AtomicCountingLBVH bvh;
    gg::AtomicCountingLBVH::QueryBuffer qbuffer;
    bvh.build(dev.view());
    bvh.detect([](const auto& i, const auto&) noexcept { return (i % 2u) == 0u; }, qbuffer);
    gg::frame_stream().flush();
    return to_pair_set(read_vector2i(qbuffer.view()));
}

/// The brute-force overlap oracle (the same one the [bvh] test uses).
PairSet brute_force(const std::vector<gg::AABB>& boxes)
{
    PairSet out;
    for(size_t i = 0; i < boxes.size(); ++i)
        for(size_t j = i + 1; j < boxes.size(); ++j)
            if(boxes[i].intersects(boxes[j]))
                out.emplace(static_cast<IndexT>(i), static_cast<IndexT>(j));
    return out;
}

std::vector<gg::AABB> make_cloud(size_t n, std::uint32_t seed, float extent = 1.0f)
{
    std::mt19937                           rng(seed);
    std::uniform_real_distribution<float>  pos(-extent, extent);
    std::uniform_real_distribution<float>  half(0.02f, 0.20f);
    std::vector<gg::AABB>                  boxes(n);
    for(auto& b : boxes)
    {
        Eigen::Vector3f c{pos(rng), pos(rng), pos(rng)};
        Eigen::Vector3f h{half(rng), half(rng), half(rng)};
        b.min() = c - h;
        b.max() = c + h;
    }
    return boxes;
}

lud::LDeviceBuffer<gg::LAABB> upload_aabbs(const std::vector<gg::AABB>& boxes)
{
    lud::LDeviceBuffer<gg::LAABB> dev;
    dev.resize(boxes.size());
    std::vector<gg::LAABB> host(boxes.size());
    for(size_t i = 0; i < boxes.size(); ++i)
        host[i] = gg::to_luisa(boxes[i]);
    if(!host.empty())
        dev.view().copy_from(span<const gg::LAABB>{host});
    return dev;
}
}// namespace t72a

TEST_CASE("t72a_predicate_shape_and_determinism", "[filter]")
{
    test_runtime();

    // ---- (a) the design argument, pinned on the port's own structures --------
    // The reference's LBVH/StacklessBVH trajectory filters pass their predicate into
    // `detect`/`query`, where it is called from the traversal *callback* -- only for
    // objects whose AABB already passed the fixed `node.intersects(Q)` test (see
    // `LinearBVHViewer::query`'s AABB overload, which hard-wires that Intersect, and
    // `atomic_counting_lbvh.inl`'s `if(id > i && p(i, id))`). The predicate therefore
    // cannot change the visited set: it is a *selection*. The InfoStacklessBVH family
    // is the exception (its NodePred prunes subtrees) and its cull is pinned *sound*
    // by the [bvh_info] test's node-cull oracle (cookbook §7.32 rules 5/12).
    {
        auto boxes = make_cloud(512, 0x72a2u);
        auto dev   = upload_aabbs(boxes);

        auto all  = lbvh_detect_all(boxes, dev);
        auto even = lbvh_detect_even(boxes, dev);
        auto bf   = brute_force(boxes);

        CHECK(all.size() == bf.size());
        CHECK(all == bf);

        PairSet want_even;
        for(auto& p : all)
            if((p.first % 2) == 0)
                want_even.insert(p);
        CHECK(even.size() == want_even.size());
        CHECK(even == want_even);
        CHECK(even.size() < all.size());  // a real selection, not a no-op

        std::printf("[filter] predicate shape: keep-all=%zu even-i=%zu (brute=%zu)\n",
                    all.size(),
                    even.size(),
                    bf.size());
    }

    // ---- (b) determinism + device-vs-host for the ported codim-thickness layer ---
    {
        SceneDesc desc;
        desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
        desc.mesh.positions = {{0.0, 0.125, 0.0}, {1.0, 0.25, 0.0}, {0.0, 0.5, 0.0}, {1.0, 1.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}, {1, 3, 2}};
        for(int i = 0; i < 4; ++i)
        {
            desc.mesh.thicknesses.push_back(0.0625);
            desc.mesh.d_hats.push_back(0.125);
            desc.mesh.displacements.push_back(Vector3{0.0, -0.25, 0.0});
        }

        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_t72a_determinism")};
        World  world{engine};
        world.init(scene);

        {
            std::vector<Vector3> gdisp(5, Vector3{0.0, -0.25, 0.0});
            gdisp[0] = Vector3::Zero();  // the half plane's own vertex does not move
            set_displacements(gdisp);
        }
        auto* gtf = TrajectoryFilterTestProbe::trajectory_filter;
        REQUIRE(gtf != nullptr);
        auto easy_slot = gtf->find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
        REQUIRE(static_cast<bool>(easy_slot));
        auto d = gather(*gtf);

        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive) == 1.0);
        auto phs1 = to_pair_set(read_phs(*easy_slot.view()));
        // every vertex is above the plane and moves down by 0.25 at alpha = 0.5:
        // y=0.125 -> t = 0.125, t0 = 0.0625 -> the step fraction is 0.5 * 0.9 = 0.45
        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, 0.5) == 0.45);
        auto tois1 = read_tois(easy_slot.view()->toi_PHs());

        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive) == 1.0);
        auto phs2 = to_pair_set(read_phs(*easy_slot.view()));
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, 0.5);
        auto tois2 = read_tois(easy_slot.view()->toi_PHs());

        CHECK(phs1 == phs2);
        CHECK(phs1 == easy_candidate_oracle(d));
        REQUIRE(tois1.size() == tois2.size());
        for(size_t i = 0; i < tois1.size(); ++i)
            CHECK(tois1[i] == tois2[i]);  // bit-identical across runs
        CHECK(tois1 == easy_toi_oracle(d, 0.5));

        // the managed step fraction: every vertex is above the plane and moves down, so
        // the minimum is below 1 and the manager returns it unclamped
        auto stepped = gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, 0.5);
        CHECK(stepped == 0.45);
        CHECK(*std::min_element(tois2.begin(), tois2.end()) == 0.45);

        // ---- device-vs-host for the ported `D_range`/`is_active_D` ---------------
        // A test-local kernel evaluates the *ported* helpers on device values and the
        // test compares against the host re-implementation of the same formula.
        struct Case
        {
            Float xi;
            Float d_hat;
            Float D;
        };
        std::vector<Case> cases;
        std::mt19937                      rng(0x72a3u);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for(int i = 0; i < 64; ++i)
        {
            Float xi    = static_cast<Float>(dist(rng)) * 0.5;
            Float d_hat = static_cast<Float>(dist(rng)) * 0.5;
            Float D     = static_cast<Float>(std::pow(static_cast<double>(xi + d_hat) * dist(rng) * 1.2, 2.0));
            cases.push_back(Case{xi, d_hat, D});
        }
        // the exact boundary cases
        cases.push_back(Case{0.0625, 0.125, 0.0625 * 0.0625});
        cases.push_back(Case{0.0625, 0.125, (0.0625 + 0.125) * (0.0625 + 0.125)});
        cases.push_back(Case{0.0625, 0.125, 0.125 * 0.125});
        cases.push_back(Case{0.0, 0.125, 0.0});

        lud::LDeviceBuffer<Float> xi_buf, dh_buf, D_buf;
        xi_buf.resize(cases.size());
        dh_buf.resize(cases.size());
        D_buf.resize(cases.size());
        std::vector<Float> h_xi, h_dh, h_D;
        for(auto& c : cases)
        {
            h_xi.push_back(c.xi);
            h_dh.push_back(c.d_hat);
            h_D.push_back(c.D);
        }
        xi_buf.view().copy_from(span<const Float>{h_xi});
        dh_buf.view().copy_from(span<const Float>{h_dh});
        D_buf.view().copy_from(span<const Float>{h_D});

        lud::LDeviceBuffer<Float> lo_buf, hi_buf;
        lud::LDeviceBuffer<std::int32_t> active_buf;
        lo_buf.resize(cases.size());
        hi_buf.resize(cases.size());
        active_buf.resize(cases.size());

        auto& stream = gg::frame_stream();
        auto& k      = lla::detail::cached_kernel<lcrt::Buffer<double>,
                                                  lcrt::Buffer<double>,
                                                  lcrt::Buffer<double>,
                                                  lcrt::Buffer<double>,
                                                  lcrt::Buffer<double>,
                                                  lcrt::Buffer<std::int32_t>,
                                                  std::uint32_t>(
            [](lcrt::BufferVar<double> xi,
               lcrt::BufferVar<double> dh,
               lcrt::BufferVar<double> D,
               lcrt::BufferVar<double> lo,
               lcrt::BufferVar<double> hi,
               lcrt::BufferVar<std::int32_t> active,
               lcrt::Var<std::uint32_t> n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto a = gg::D_range(xi->read(i), dh->read(i));
                    lo->write(i, a(0));
                    hi->write(i, a(1));
                    active->write(i, lcrt::select(std::int32_t{0}, std::int32_t{1}, gg::is_active_D(a, D->read(i))));
                };
            });
        auto n = static_cast<std::uint32_t>(cases.size());
        stream.record(k(xi_buf.view(), dh_buf.view(), D_buf.view(), lo_buf.view(), hi_buf.view(), active_buf.view(), n).dispatch(n));
        auto h_lo     = read_device(lo_buf.cview());
        auto h_hi     = read_device(hi_buf.cview());
        auto h_active = read_device(active_buf.cview());
        REQUIRE(h_lo.size() == cases.size());
        int mismatches = 0;
        for(size_t i = 0; i < cases.size(); ++i)
        {
            auto band = host_D_range(cases[i].xi, cases[i].d_hat);
            CHECK(h_lo[i] == band.lo);
            CHECK(h_hi[i] == band.hi);
            if(static_cast<bool>(h_active[i]) != host_is_active_D(band, cases[i].D))
                ++mismatches;
        }
        CHECK(mismatches == 0);
        std::printf("[filter] device-vs-host D_range/is_active_D: %zu cases, %d mismatches\n",
                    cases.size(),
                    mismatches);
    }
}

// ===========================================================================
//  7) the quoted numbers
// ===========================================================================
TEST_CASE("t72a_summary", "[filter]")
{
    test_runtime();
    std::printf(
        "[filter] T7.2a: trajectory_filter, simplex_trajectory_filter, "
        "vertex_half_plane_trajectory_filter, global_trajectory_filter, "
        "filters/easy_vertex_half_plane_trajectory_filter, "
        "filters/al_vertex_half_plane_trajectory_filter.\n");
    std::printf("[filter] the manager's stages are driven through the documented "
                "GlobalTrajectoryFilter::run_stage() test seam (no Newton loop yet).\n");
    CHECK(true);
}
