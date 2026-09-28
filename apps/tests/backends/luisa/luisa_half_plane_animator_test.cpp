/********************************************************************
 * @file luisa_half_plane_animator_test.cpp
 * @brief T4.2 — implicit geometry (half plane) + animator + external force +
 *        joint dof + pipeline/time-integrator flags (LuisaCompute backend).
 *
 * Drives the ported systems through the public uipc::Engine/World API (like
 * backend_shell_test.cpp / luisa_global_geometry_test.cpp) and reaches into the
 * backend only through public interfaces:
 *   - test-local systems (VertexReporter / SimplicialSurfaceReporter / Animator /
 *     ExternalForceReporter / JointDofReporter / TimeIntegrator) registered with
 *     REGISTER_SIM_SYSTEM, exactly like the real FEM/ABD ones;
 *   - a test-local probe SimSystem that caches the managers, the *ported* half
 *     plane reporters and the flag systems;
 *   - the backend's public classes (HalfPlane::positions()/normals()/geo_infos(),
 *     GlobalVertexManager, GlobalBodyManager, GlobalSimplicialSurfaceManager,
 *     TimeIntegratorManager, GlobalJointDofManager, GlobalAnimator, the four flag
 *     systems and SimEngine::pipeline_type()).
 *
 * Two constraints shape this file, and both are properties of a *shared* test
 * binary (every luisa test source is linked into one executable, and every
 * registered SimSystem is built for every scene of every test case):
 *
 *  1. The test reporters only report geometries that carry the test's marker meta
 *     attribute (test_attr::marker). The other luisa test sources register their
 *     own reporters, which report *every* SimplicialComplex of the scene; without
 *     the marker this file would double every vertex/body/surface array of
 *     `luisa_global_geometry_test.cpp` (measured: 16 vertex entries instead of 8).
 *  2. The global manager arrays therefore contain slices in *reporter uid order*
 *     (the half plane reporter's uid 2 sorts first, this test's reporter second),
 *     so a check reads its own slice through the reporters' public
 *     `vertex_offset()`/`body_offset()`/`vertex_count()`/`body_count()`
 *     accessors instead of assuming it owns the whole array. The slice contents
 *     are still hand-computed from the scene alone.
 *
 * Every expectation is computed independently in this file: a hand-written
 * interpolation law, a host model derived from the scene, or a value this test
 * fed in. Nothing is taken from the CUDA backend.
 *
 * Coverage map (brief §3):
 *   1) half plane geometry / vertex sets / body set ... half_plane_geometry_and_vertex_sets
 *   2) animator keyframes (hand-computed) ........... half_plane_animator_keyframes
 *      animator law on the host and on the device ..... half_plane_animator_lerp_law
 *   3) pipeline + BDF flags ......................... half_plane_pipeline_and_flags
 *   4) external force ............................... half_plane_external_force
 *   5) joint dof .................................... half_plane_joint_dof
 *   6) edge cases ................................... half_plane_animator_edge_cases
 *********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/constitution/soft_position_constraint.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>

#include <type_define.h>
#include <frame_stream.h>
#include <animator/animator.h>
#include <animator/global_animator.h>
#include <animator/utils.h>
#include <external_force/external_force_reporter.h>
#include <external_force/global_external_force_manager.h>
#include <global_geometry/body_reporter.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/simplicial_surface_reporter.h>
#include <global_geometry/vertex_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_body_reporter.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <joint_dof_system/global_joint_dof_manager.h>
#include <joint_dof_system/joint_dof_reporter.h>
#include <luisa_device/linear/algorithms.h>
#include <pipeline/al_ipc_pipeline_flag.h>
#include <pipeline/ipc_pipeline_flag.h>
// NOTE: the test target's include path contains src/backends/none before
// src/backends/luisa, so the ambiguous form <sim_engine.h> would pick the none
// backend's header; the fully qualified path is unambiguous.
#include <backends/luisa/sim_engine.h>
#include <time_integrator/bdf1_flag.h>
#include <time_integrator/bdf2_flag.h>
#include <time_integrator/time_integrator.h>
#include <time_integrator/time_integrator_manager.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;

namespace gg  = uipc::backend::luisa;
namespace lud = uipc::backend::luisa::luisa_device;
namespace lps = uipc::backend::luisa::luisa_device::math;
namespace lla = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;

// ---------------------------------------------------------------------------
//  Attribute names this test feeds through its reporters
// ---------------------------------------------------------------------------
namespace test_attr
{
constexpr std::string_view marker       = "test_t42_half_plane_scene";  // IndexT, per geometry meta
constexpr std::string_view dimension    = "test_dimension";             // IndexT, per vertex
constexpr std::string_view contact_id   = "test_contact_element_id";    // IndexT, per vertex
constexpr std::string_view subscene_id  = "test_subscene_element_id";   // IndexT, per vertex
constexpr std::string_view d_hat        = "test_d_hat";                 // Float, per vertex
constexpr std::string_view thickness    = "test_thickness";             // Float, per vertex
constexpr std::string_view displacement = "test_displacement";          // Vector3, per vertex
constexpr std::string_view force        = "test_external_force";        // Vector3, per vertex
constexpr std::string_view force_on     = "test_external_force_enable"; // IndexT, per vertex
constexpr std::string_view joint_count  = "test_joint_dof_count";       // IndexT, per geometry meta
constexpr std::string_view joint_angle  = "test_joint_angle";           // Float, per instance
}// namespace test_attr

// ---------------------------------------------------------------------------
//  The animator law, written out independently of the port
// ---------------------------------------------------------------------------
// src/backends/cuda/animator/utils.h:
//     alpha = clamp(alpha, 0, 1);  return src * (1 - alpha) + dst * alpha;
[[nodiscard]] Float hand_lerp(Float src, Float dst, Float alpha)
{
    const Float a = std::clamp(alpha, Float{0.0}, Float{1.0});
    return src * (1.0 - a) + dst * a;
}

[[nodiscard]] Vector3 hand_lerp(const Vector3& src, const Vector3& dst, Float alpha)
{
    Vector3 out;
    for(int i = 0; i < 3; ++i)
    {
        out[i] = hand_lerp(src[i], dst[i], alpha);
    }
    return out;
}

namespace
{
// ---------------------------------------------------------------------------
//  Event log: the per-frame call order of the ported systems
// ---------------------------------------------------------------------------
struct EventLog
{
    std::vector<std::string> events;

    void push(std::string_view e) { events.emplace_back(e); }
    void clear() { events.clear(); }
    [[nodiscard]] int count_of(std::string_view e) const
    {
        return static_cast<int>(std::count(events.begin(), events.end(), e));
    }
};

EventLog& event_log()
{
    static EventLog log;
    return log;
}

/// `a` is a subsequence of `b` (each event of `a` appears in `b`, in order).
bool is_subsequence(const std::vector<std::string>& a, const std::vector<std::string>& b)
{
    size_t j = 0;
    for(const auto& e : a)
    {
        while(j < b.size() && b[j] != e)
        {
            ++j;
        }
        if(j == b.size())
        {
            return false;
        }
        ++j;
    }
    return true;
}

// ---------------------------------------------------------------------------
//  Comparison / read-back helpers
// ---------------------------------------------------------------------------
template <typename T>
bool eigen_equal(const std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
    {
        return false;
    }
    for(size_t i = 0; i < a.size(); ++i)
    {
        if(!(a[i].array() == b[i].array()).all())
        {
            return false;
        }
    }
    return true;
}

/// [offset, offset + count) of a container, clamped to its size.
template <typename T>
std::vector<T> slice(const std::vector<T>& v, size_t offset, size_t count)
{
    if(offset > v.size())
    {
        return {};
    }
    const size_t n = std::min(count, v.size() - offset);
    return std::vector<T>(v.begin() + static_cast<long>(offset), v.begin() + static_cast<long>(offset + n));
}

/// Element equality that works for Eigen vectors too (Eigen's `==` is
/// coefficient-wise and is not convertible to bool).
template <typename T>
bool value_equal(const T& a, const T& b)
{
    if constexpr(requires { a.array(); })
    {
        return (a.array() == b.array()).all();
    }
    else
    {
        return a == b;
    }
}

/// `needle` appears as a contiguous subsequence of `hay` (the manager arrays are
/// a concatenation of the reporters' slices, so a reporter's own slice is
/// contiguous but not necessarily at the front).
template <typename T>
bool contains_subsequence(const std::vector<T>& hay, const std::vector<T>& needle)
{
    if(needle.empty())
    {
        return true;
    }
    if(needle.size() > hay.size())
    {
        return false;
    }
    for(size_t i = 0; i + needle.size() <= hay.size(); ++i)
    {
        bool all = true;
        for(size_t j = 0; j < needle.size(); ++j)
        {
            if(!value_equal(hay[i + j], needle[j]))
            {
                all = false;
                break;
            }
        }
        if(all)
        {
            return true;
        }
    }
    return false;
}

/// Flush whatever is recorded, copy the slice to the host and wait (the same
/// helper the T4.1 test uses).
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
    {
        host.push_back(gg::to_eigen(v));
    }
    return host;
}

std::vector<Vector2i> read_vector2i(const lud::LCBufferView<gg::LVector2i>& view)
{
    std::vector<Vector2i> host;
    auto                  raw = read_device(view);
    host.reserve(raw.size());
    for(auto& v : raw)
    {
        host.push_back(gg::to_eigen(v));
    }
    return host;
}

std::vector<Vector3i> read_vector3i(const lud::LCBufferView<gg::LVector3i>& view)
{
    std::vector<Vector3i> host;
    auto                  raw = read_device(view);
    host.reserve(raw.size());
    for(auto& v : raw)
    {
        host.push_back(gg::to_eigen(v));
    }
    return host;
}

std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}
}// namespace

// ---------------------------------------------------------------------------
//  Scene description
// ---------------------------------------------------------------------------
namespace
{
struct HalfPlaneDesc
{
    Vector3 P{0.0, 0.0, 0.0};
    Vector3 N{0.0, 1.0, 0.0};
    IndexT  contact_id    = 0;
    IndexT  subscene_id   = 0;
    bool    with_meta_ids = true;
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
    std::vector<Vector3>  forces;
    std::vector<IndexT>   force_enabled;
    bool                  with_constraint = false;  // SoftPositionConstraint -> "Constraint"
};

struct JointDesc
{
    IndexT dof_count   = 1;
    Float  angle       = 0.0;
    IndexT contact_id  = 0;
    IndexT subscene_id = 1;
};

struct SceneDesc
{
    std::vector<HalfPlaneDesc> half_planes;
    std::vector<MeshDesc>      meshes;
    std::vector<JointDesc>     joints;
    std::size_t                empty_objects = 0;  // objects without any geometry

    std::string integrator_type      = "bdf1";
    std::string contact_constitution = "ipc";
    Float       dt                   = 0.25;
    bool        enable_sanity_check  = false;
};

/// Fill every per-vertex test attribute a case did not set with its default, so
/// a MeshDesc can be described piecewise.
void normalize(MeshDesc& mesh)
{
    const size_t n   = mesh.positions.size();
    auto         fit = [n](auto& v, auto value)
    {
        if(v.empty())
        {
            v.resize(n, value);
        }
    };
    fit(mesh.dimensions, IndexT{2});
    fit(mesh.contact_ids, IndexT{0});
    fit(mesh.subscene_ids, IndexT{0});
    fit(mesh.d_hats, Float{0.01});
    fit(mesh.thicknesses, Float{0.0});
    fit(mesh.displacements, Vector3::Zero());
    fit(mesh.forces, Vector3::Zero());
    fit(mesh.force_enabled, IndexT{0});
}

/// Build the scene: one object per half plane / mesh / joint, plus
/// `empty_objects` objects without any geometry. `mesh_objects` /
/// `empty_object_handles` (both optional) collect the created objects so a test
/// can attach an animation to them. Every geometry carries the test marker, so
/// the reporters of *other* test sources ignore it (see the file header).
Scene build_scene(const SceneDesc&        desc,
                  std::vector<S<Object>>* mesh_objects         = nullptr,
                  std::vector<S<Object>>* empty_object_handles = nullptr)
{
    auto config                       = Scene::default_config();
    config["sanity_check"]["enable"]  = desc.enable_sanity_check;
    config["dt"]                      = desc.dt;
    config["integrator"]["type"]      = desc.integrator_type;
    config["contact"]["constitution"] = desc.contact_constitution;
    Scene scene{config};

    for(size_t i = 0; i < desc.half_planes.size(); ++i)
    {
        const auto& hp     = desc.half_planes[i];
        auto        object = scene.objects().create("ground_" + std::to_string(i));
        auto        ig     = halfplane(hp.P, hp.N);
        ig.meta().create<IndexT>(test_attr::marker, 1);
        if(hp.with_meta_ids)
        {
            ig.meta().create<IndexT>(builtin::contact_element_id, hp.contact_id);
            ig.meta().create<IndexT>(builtin::subscene_element_id, hp.subscene_id);
        }
        object->geometries().create(ig);
    }

    for(size_t i = 0; i < desc.meshes.size(); ++i)
    {
        auto mesh = desc.meshes[i];
        normalize(mesh);

        SimplicialComplex sc = [&]() -> SimplicialComplex
        {
            if(mesh.positions.empty())
            {
                return SimplicialComplex{};
            }
            if(mesh.triangles.empty())
            {
                auto complex = pointcloud(span<const Vector3>{mesh.positions});
                label_surface(complex);
                return complex;
            }
            auto complex = trimesh(span<const Vector3>{mesh.positions},
                                   span<const Vector3i>{mesh.triangles});
            label_surface(complex);
            return complex;
        }();

        sc.meta().create<IndexT>(test_attr::marker, 1);

        if(!mesh.positions.empty() && !mesh.triangles.empty())
        {
            auto vertices  = sc.vertices();
            auto dimension = vertices.create<IndexT>(test_attr::dimension, 0);
            auto contact   = vertices.create<IndexT>(test_attr::contact_id, 0);
            auto subscene  = vertices.create<IndexT>(test_attr::subscene_id, 0);
            auto d_hat     = vertices.create<Float>(test_attr::d_hat, 0.0);
            auto thickness = vertices.create<Float>(test_attr::thickness, 0.0);
            auto displace  = vertices.create<Vector3>(test_attr::displacement, Vector3::Zero());
            auto force     = vertices.create<Vector3>(test_attr::force, Vector3::Zero());
            auto force_on  = vertices.create<IndexT>(test_attr::force_on, 0);

            std::ranges::copy(mesh.dimensions, view(*dimension).begin());
            std::ranges::copy(mesh.contact_ids, view(*contact).begin());
            std::ranges::copy(mesh.subscene_ids, view(*subscene).begin());
            std::ranges::copy(mesh.d_hats, view(*d_hat).begin());
            std::ranges::copy(mesh.thicknesses, view(*thickness).begin());
            std::ranges::copy(mesh.displacements, view(*displace).begin());
            std::ranges::copy(mesh.forces, view(*force).begin());
            std::ranges::copy(mesh.force_enabled, view(*force_on).begin());
        }

        if(mesh.with_constraint)
        {
            uipc::constitution::SoftPositionConstraint spc;
            spc.apply_to(sc, 100.0);
        }

        auto object = scene.objects().create("mesh_" + std::to_string(i));
        object->geometries().create(sc);
        if(mesh_objects)
        {
            mesh_objects->push_back(object);
        }
    }

    for(size_t i = 0; i < desc.joints.size(); ++i)
    {
        const auto&          joint = desc.joints[i];
        std::vector<Vector3> vs    = {Vector3::Zero()};
        auto                 sc    = pointcloud(span<const Vector3>{vs});
        sc.meta().create<IndexT>(test_attr::marker, 1);
        sc.meta().create<IndexT>(test_attr::joint_count, joint.dof_count);
        auto angle = sc.instances().create<Float>(test_attr::joint_angle, joint.angle);
        view(*angle)[0] = joint.angle;

        auto object = scene.objects().create("joint_" + std::to_string(i));
        object->geometries().create(sc);
    }

    for(std::size_t i = 0; i < desc.empty_objects; ++i)
    {
        auto object = scene.objects().create("empty_" + std::to_string(i));
        if(empty_object_handles)
        {
            empty_object_handles->push_back(object);
        }
    }

    return scene;
}

/// The scene's implicit geometries in scene order.
std::vector<ImplicitGeometry*> scene_implicit_geometries(uipc::backend::WorldVisitor& world)
{
    std::vector<ImplicitGeometry*> result;
    for(auto& slot : world.scene().geometries())
    {
        auto ig = slot->geometry().as<ImplicitGeometry>();
        if(ig)
        {
            result.push_back(ig);
        }
    }
    return result;
}

/// Only the SimplicialComplexes this test fed in (they carry the test marker) —
/// every reporter / model in this file walks exactly this list, in scene order.
std::vector<SimplicialComplex*> test_complexes(uipc::backend::WorldVisitor& world)
{
    std::vector<SimplicialComplex*> result;
    for(auto& slot : world.scene().geometries())
    {
        auto sc = slot->geometry().as<SimplicialComplex>();
        if(!sc)
        {
            continue;
        }
        auto marker = sc->meta().find<IndexT>(test_attr::marker);
        if(marker && view(*marker)[0] == 1)
        {
            result.push_back(sc);
        }
    }
    return result;
}

/// The hand-computed host model of this test's *own* contribution to the global
/// arrays: the half plane slice (one vertex per instance) followed by the
/// complexes this test fed in, in scene order.
struct VertexHostModel
{
    std::vector<IndexT>  half_plane_coindices;
    std::vector<Vector3> half_plane_positions;
    std::vector<IndexT>  half_plane_contact_ids;
    std::vector<IndexT>  half_plane_subscene_ids;

    std::vector<IndexT>  complex_coindices;
    std::vector<IndexT>  complex_body_ids;  // this test reports -1 for its complexes
    std::vector<Vector3> complex_positions;
    std::vector<IndexT>  complex_contact_ids;
    std::vector<IndexT>  complex_subscene_ids;
    std::vector<IndexT>  complex_dimensions;
    std::vector<Float>   complex_d_hats;
    std::vector<Float>   complex_thicknesses;

    SizeT half_plane_count() const { return half_plane_coindices.size(); }
    SizeT complex_count() const { return complex_coindices.size(); }
};

VertexHostModel compute_vertex_model(uipc::backend::WorldVisitor& world)
{
    VertexHostModel model;

    for(auto* ig : scene_implicit_geometries(world))
    {
        auto uid = ig->meta().find<U64>(builtin::implicit_geometry_uid);
        if(!uid || uid->view()[0] != gg::HalfPlane::ImplicitGeometryUID)
        {
            continue;
        }
        auto contact  = ig->meta().find<IndexT>(builtin::contact_element_id);
        auto subscene = ig->meta().find<IndexT>(builtin::subscene_element_id);
        auto P        = ig->instances().find<Vector3>("P");
        for(size_t i = 0; i < ig->instances().size(); ++i)
        {
            model.half_plane_coindices.push_back(static_cast<IndexT>(i));
            model.half_plane_positions.push_back(view(*P)[i]);
            model.half_plane_contact_ids.push_back(contact ? view(*contact)[0] : 0);
            model.half_plane_subscene_ids.push_back(subscene ? view(*subscene)[0] : 0);
        }
    }

    for(auto* sc : test_complexes(world))
    {
        auto positions = sc->positions().view();
        auto dimension = sc->vertices().find<IndexT>(test_attr::dimension);
        auto contact   = sc->vertices().find<IndexT>(test_attr::contact_id);
        auto subscene  = sc->vertices().find<IndexT>(test_attr::subscene_id);
        auto d_hat     = sc->vertices().find<Float>(test_attr::d_hat);
        auto thickness = sc->vertices().find<Float>(test_attr::thickness);
        for(size_t v = 0; v < sc->vertices().size(); ++v)
        {
            model.complex_coindices.push_back(static_cast<IndexT>(v));
            model.complex_body_ids.push_back(-1);  // this test's convention
            model.complex_positions.push_back(positions[v]);
            model.complex_dimensions.push_back(dimension ? view(*dimension)[v] : 3);
            model.complex_contact_ids.push_back(contact ? view(*contact)[v] : 0);
            model.complex_subscene_ids.push_back(subscene ? view(*subscene)[v] : 0);
            model.complex_d_hats.push_back(d_hat ? view(*d_hat)[v] : 0.0);
            model.complex_thicknesses.push_back(thickness ? view(*thickness)[v] : 0.0);
        }
    }

    return model;
}

/// The host model of this test's surface slice (global vertex indices): the
/// surface elements of the complexes this test fed in, offset by the slice
/// offset of the reporter that reports them.
struct SurfaceHostModel
{
    std::vector<IndexT>   surf_vertices;
    std::vector<Vector2i> surf_edges;
    std::vector<Vector3i> surf_triangles;
};

SurfaceHostModel compute_surface_model(uipc::backend::WorldVisitor& world, SizeT slice_offset)
{
    SurfaceHostModel model;

    IndexT offset = static_cast<IndexT>(slice_offset);
    for(auto* sc : test_complexes(world))
    {
        auto v_is_surf = sc->vertices().find<IndexT>(builtin::is_surf);
        if(v_is_surf)
        {
            for(size_t i = 0; i < view(*v_is_surf).size(); ++i)
            {
                if(view(*v_is_surf)[i])
                {
                    model.surf_vertices.push_back(offset + static_cast<IndexT>(i));
                }
            }
        }
        auto e_is_surf = sc->edges().find<IndexT>(builtin::is_surf);
        if(e_is_surf)
        {
            auto edges = sc->edges().topo().view();
            for(size_t i = 0; i < view(*e_is_surf).size(); ++i)
            {
                if(view(*e_is_surf)[i])
                {
                    model.surf_edges.push_back(Vector2i{edges[i][0] + offset, edges[i][1] + offset});
                }
            }
        }
        auto f_is_surf = sc->triangles().find<IndexT>(builtin::is_surf);
        if(f_is_surf)
        {
            auto triangles = sc->triangles().topo().view();
            for(size_t i = 0; i < view(*f_is_surf).size(); ++i)
            {
                if(view(*f_is_surf)[i])
                {
                    model.surf_triangles.push_back(Vector3i{triangles[i][0] + offset,
                                                            triangles[i][1] + offset,
                                                            triangles[i][2] + offset});
                }
            }
        }
        offset += static_cast<IndexT>(sc->vertices().size());
    }

    return model;
}
}// namespace

// ---------------------------------------------------------------------------
//  Test reporters (registered systems, exactly like the real FEM/ABD ones).
//  They report only the geometries carrying test_attr::marker, which keeps them
//  from changing the arrays the other test sources assert on.
// ---------------------------------------------------------------------------

/// Reports this test's complexes (one slice, scene order). This is the geometry
/// the animation writes into; the ported HalfPlaneVertexReporter reports the
/// half plane slice.
class HalfPlaneAnimatorTestVertexReporter final : public gg::VertexReporter
{
  public:
    using VertexReporter::VertexReporter;

  protected:
    void do_build(BuildInfo& info) override { (void)info; }

    void do_report_count(VertexCountInfo& info) override
    {
        SizeT count = 0;
        for(auto* sc : test_complexes(world()))
        {
            count += sc->vertices().size();
        }
        info.count(count);
    }

    void do_report_attributes(VertexAttributeInfo& info) override
    {
        std::vector<IndexT>       coindices;
        std::vector<gg::LVector3> positions;
        std::vector<IndexT>       body_ids;
        std::vector<IndexT>       dimensions;
        std::vector<IndexT>       contact_ids;
        std::vector<IndexT>       subscene_ids;
        std::vector<Float>        d_hats;
        std::vector<Float>        thicknesses;

        for(auto* sc : test_complexes(world()))
        {
            auto positions_view = sc->positions().view();
            auto dimension      = sc->vertices().find<IndexT>(test_attr::dimension);
            auto contact        = sc->vertices().find<IndexT>(test_attr::contact_id);
            auto subscene       = sc->vertices().find<IndexT>(test_attr::subscene_id);
            auto d_hat          = sc->vertices().find<Float>(test_attr::d_hat);
            auto thickness      = sc->vertices().find<Float>(test_attr::thickness);

            for(size_t v = 0; v < positions_view.size(); ++v)
            {
                coindices.push_back(static_cast<IndexT>(v));
                positions.push_back(gg::to_luisa(positions_view[v]));
                // this test's complexes are not dynamics bodies: -1 is the
                // reference's "no body id" default
                body_ids.push_back(-1);
                dimensions.push_back(dimension ? view(*dimension)[v] : 3);
                contact_ids.push_back(contact ? view(*contact)[v] : 0);
                subscene_ids.push_back(subscene ? view(*subscene)[v] : 0);
                d_hats.push_back(d_hat ? view(*d_hat)[v] : 0.0);
                thicknesses.push_back(thickness ? view(*thickness)[v] : 0.0);
            }
        }

        if(info.frame() == 0)
        {
            m_rest_positions = positions;  // the reference reports rest at frame 0
        }

        info.coindices().copy_from(span<const IndexT>{coindices});
        info.positions().copy_from(span<const gg::LVector3>{positions});
        info.rest_positions().copy_from(span<const gg::LVector3>{m_rest_positions});
        info.body_ids().copy_from(span<const IndexT>{body_ids});
        info.dimensions().copy_from(span<const IndexT>{dimensions});
        info.contact_element_ids().copy_from(span<const IndexT>{contact_ids});
        info.subscene_element_ids().copy_from(span<const IndexT>{subscene_ids});
        info.d_hats().copy_from(span<const Float>{d_hats});
        info.thicknesses().copy_from(span<const Float>{thicknesses});

        if(info.frame() > 0)
        {
            // frame > 0 is the engine's per-frame update_attributes() call
            event_log().push("vertex_attributes");
        }
    }

    void do_report_displacements(VertexDisplacementInfo& info) override
    {
        std::vector<gg::LVector3> displacements;
        for(auto* sc : test_complexes(world()))
        {
            auto displace = sc->vertices().find<Vector3>(test_attr::displacement);
            for(size_t v = 0; v < sc->vertices().size(); ++v)
            {
                displacements.push_back(displace ? gg::to_luisa(view(*displace)[v]) :
                                                   gg::LVector3{});
            }
        }
        info.displacements().copy_from(span<const gg::LVector3>{displacements});
    }

    U64 get_uid() const noexcept override { return 0x543432414E494Dull; }  // "T42ANIM"

  private:
    std::vector<gg::LVector3> m_rest_positions;
};
REGISTER_SIM_SYSTEM(HalfPlaneAnimatorTestVertexReporter);

template <typename AttributesT>
SizeT count_surf_elements(AttributesT attributes)
{
    auto is_surf = attributes.template find<IndexT>(builtin::is_surf);
    if(!is_surf)
    {
        return 0;
    }
    auto v = view(*is_surf);
    return static_cast<SizeT>(std::ranges::count_if(v, [](IndexT x) { return x != 0; }));
}

/// Reports the surface of this test's complexes with *global* vertex indices (the
/// "[0, 1, 2]-style" convention of the reference reporters): the global vertex
/// offset of a geometry is this reporter's slice offset plus the vertex counts of
/// the previous geometries of its own list.
class HalfPlaneAnimatorTestSurfaceReporter final : public gg::SimplicialSurfaceReporter
{
  public:
    using SimplicialSurfaceReporter::SimplicialSurfaceReporter;

    static std::vector<IndexT>   reported_surf_vertices;
    static std::vector<Vector2i> reported_surf_edges;
    static std::vector<Vector3i> reported_surf_triangles;

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        m_vertex_reporter = &require<HalfPlaneAnimatorTestVertexReporter>();
    }

    void do_init(SurfaceInitInfo& info) override { (void)info; }

    void do_report_count(SurfaceCountInfo& info) override
    {
        SizeT v = 0, e = 0, f = 0;
        for(auto* sc : test_complexes(world()))
        {
            v += count_surf_elements(sc->vertices());
            e += count_surf_elements(sc->edges());
            f += count_surf_elements(sc->triangles());
        }
        info.surf_vertex_count(v);
        info.surf_edge_count(e);
        info.surf_triangle_count(f);
    }

    void do_report_attributes(SurfaceAttributeInfo& info) override
    {
        auto complexes = test_complexes(world());

        std::vector<IndexT> vertex_offsets(complexes.size() + 1,
                                           m_vertex_reporter->vertex_offset());
        for(size_t g = 0; g < complexes.size(); ++g)
        {
            vertex_offsets[g + 1] =
                vertex_offsets[g] + static_cast<IndexT>(complexes[g]->vertices().size());
        }

        std::vector<IndexT>        surf_vertices;
        std::vector<gg::LVector2i> surf_edges;
        std::vector<gg::LVector3i> surf_triangles;

        size_t geometry_index = 0;
        for(auto* sc : complexes)
        {
            const IndexT global_vertex_offset = vertex_offsets[geometry_index++];

            auto v_is_surf = sc->vertices().find<IndexT>(builtin::is_surf);
            if(v_is_surf)
            {
                for(size_t i = 0; i < view(*v_is_surf).size(); ++i)
                {
                    if(view(*v_is_surf)[i])
                    {
                        surf_vertices.push_back(global_vertex_offset + static_cast<IndexT>(i));
                    }
                }
            }

            auto e_is_surf = sc->edges().find<IndexT>(builtin::is_surf);
            if(e_is_surf)
            {
                auto edges = sc->edges().topo().view();
                for(size_t i = 0; i < view(*e_is_surf).size(); ++i)
                {
                    if(view(*e_is_surf)[i])
                    {
                        surf_edges.push_back(
                            gg::to_luisa(Vector2i{edges[i][0] + global_vertex_offset,
                                                  edges[i][1] + global_vertex_offset}));
                    }
                }
            }

            auto f_is_surf = sc->triangles().find<IndexT>(builtin::is_surf);
            if(f_is_surf)
            {
                auto triangles = sc->triangles().topo().view();
                for(size_t i = 0; i < view(*f_is_surf).size(); ++i)
                {
                    if(view(*f_is_surf)[i])
                    {
                        surf_triangles.push_back(
                            gg::to_luisa(Vector3i{triangles[i][0] + global_vertex_offset,
                                                  triangles[i][1] + global_vertex_offset,
                                                  triangles[i][2] + global_vertex_offset}));
                    }
                }
            }
        }

        reported_surf_vertices = surf_vertices;
        reported_surf_edges.clear();
        for(auto& e : surf_edges)
        {
            reported_surf_edges.push_back(gg::to_eigen(e));
        }
        reported_surf_triangles.clear();
        for(auto& f : surf_triangles)
        {
            reported_surf_triangles.push_back(gg::to_eigen(f));
        }

        info.surf_vertices().copy_from(span<const IndexT>{surf_vertices});
        info.surf_edges().copy_from(span<const gg::LVector2i>{surf_edges});
        info.surf_triangles().copy_from(span<const gg::LVector3i>{surf_triangles});
    }

  private:
    HalfPlaneAnimatorTestVertexReporter* m_vertex_reporter = nullptr;
};
REGISTER_SIM_SYSTEM(HalfPlaneAnimatorTestSurfaceReporter);

std::vector<IndexT>   HalfPlaneAnimatorTestSurfaceReporter::reported_surf_vertices;
std::vector<Vector2i> HalfPlaneAnimatorTestSurfaceReporter::reported_surf_edges;
std::vector<Vector3i> HalfPlaneAnimatorTestSurfaceReporter::reported_surf_triangles;

// ---------------------------------------------------------------------------
//  Animator
// ---------------------------------------------------------------------------
/// A minimal backend Animator. Its do_step() observes the substep ratio *inside*
/// the fan-out: the reference's GlobalAnimator::step() resets the ratio to 0
/// BEFORE calling the animators, so do_step() must observe exactly 0.
class HalfPlaneAnimatorTestAnimator final : public gg::Animator
{
  public:
    using Animator::Animator;

    static bool  built;
    static int   init_count;
    static int   step_count;
    static Float ratio_during_step;

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        built = true;
    }
    void do_init() override { ++init_count; }
    void do_step() override
    {
        ++step_count;
        ratio_during_step = substep_ratio();
        event_log().push("animator_step");
    }
};
REGISTER_SIM_SYSTEM(HalfPlaneAnimatorTestAnimator);

bool  HalfPlaneAnimatorTestAnimator::built             = false;
int   HalfPlaneAnimatorTestAnimator::init_count        = 0;
int   HalfPlaneAnimatorTestAnimator::step_count        = 0;
Float HalfPlaneAnimatorTestAnimator::ratio_during_step = -1.0;

// ---------------------------------------------------------------------------
//  External force reporter
// ---------------------------------------------------------------------------
/// Synthetic external force: the reporter owns the "dynamics" accumulation
/// buffers and follows the reference's lifecycle
///   init()  : build the vertex set (indices) from the scene's attributes
///   clear() : zero the force buffer BEFORE the animation writes it
///   step()  : consume the (possibly animated) force values AFTER the animation
class HalfPlaneAnimatorTestExternalForceReporter final : public gg::ExternalForceReporter
{
  public:
    using ExternalForceReporter::ExternalForceReporter;

    static int init_count;
    static int clear_count;
    static int step_count;

    [[nodiscard]] lud::LCBufferView<IndexT> vertex_indices() const { return m_indices.cview(); }
    [[nodiscard]] lud::LCBufferView<gg::LVector3> forces() const { return m_forces.cview(); }

  protected:
    void do_build(BuildInfo& info) override { (void)info; }
    U64  get_uid() const noexcept override { return 0x543432464F5243ull; }  // "T42FORC"

    void do_init() override
    {
        ++init_count;
        event_log().push("force_init");

        m_indices_host.clear();
        for(auto* sc : test_complexes(world()))
        {
            auto on = sc->vertices().find<IndexT>(test_attr::force_on);
            if(!on)
            {
                continue;
            }
            for(size_t v = 0; v < view(*on).size(); ++v)
            {
                if(view(*on)[v])
                {
                    m_indices_host.push_back(static_cast<IndexT>(v));
                }
            }
        }
        m_indices.resize(m_indices_host.size());
        if(!m_indices_host.empty())
        {
            m_indices.view().copy_from(span<const IndexT>{m_indices_host}, &gg::frame_stream());
        }
        consume_forces();
    }

    void do_clear() override
    {
        ++clear_count;
        event_log().push("force_clear");
        // "Clear all force buffers BEFORE constraints write to them"
        m_forces.resize(m_indices_host.size());
        m_forces.view().fill(gg::LVector3{}, &gg::frame_stream());
    }

    void do_step() override
    {
        ++step_count;
        event_log().push("force_step");
        consume_forces();
    }

  private:
    /// Read the (possibly animated) per-vertex force values of the reported
    /// vertex set, in the order of the set.
    void consume_forces()
    {
        std::vector<gg::LVector3> values;
        values.reserve(m_indices_host.size());

        for(auto* sc : test_complexes(world()))
        {
            auto force = sc->vertices().find<Vector3>(test_attr::force);
            if(!force)
            {
                continue;
            }
            for(size_t v = 0; v < view(*force).size(); ++v)
            {
                if(std::find(m_indices_host.begin(), m_indices_host.end(),
                             static_cast<IndexT>(v)) != m_indices_host.end())
                {
                    values.push_back(gg::to_luisa(view(*force)[v]));
                }
            }
        }

        m_forces.resize(values.size());
        if(!values.empty())
        {
            m_forces.view().copy_from(span<const gg::LVector3>{values}, &gg::frame_stream());
        }
    }

    std::vector<IndexT>              m_indices_host;
    lud::LDeviceBuffer<IndexT>       m_indices;
    lud::LDeviceBuffer<gg::LVector3> m_forces;
};
REGISTER_SIM_SYSTEM(HalfPlaneAnimatorTestExternalForceReporter);

int HalfPlaneAnimatorTestExternalForceReporter::init_count  = 0;
int HalfPlaneAnimatorTestExternalForceReporter::clear_count = 0;
int HalfPlaneAnimatorTestExternalForceReporter::step_count  = 0;

// ---------------------------------------------------------------------------
//  Joint dof reporter
// ---------------------------------------------------------------------------
/// Models a joint's persistent DOF state: each joint geometry declares its dof
/// count (test_joint_dof_count) and its current scalar state (test_joint_angle).
/// update_dof_attributes() re-derives the persistent device state from the
/// current attributes and publishes the dof sets through the reference's
/// `dof_offset`/`dof_count` geometry-meta convention (see abd_linear_subsystem.cu).
class HalfPlaneAnimatorTestJointDofReporter final : public gg::JointDofReporter
{
  public:
    using JointDofReporter::JointDofReporter;

    static bool built;
    static int  update_count;

    [[nodiscard]] std::vector<IndexT> dof_offsets_per_joint() const { return m_dof_offsets; }
    [[nodiscard]] std::vector<IndexT> dof_counts_per_joint() const { return m_dof_counts; }
    [[nodiscard]] lud::LCBufferView<Float> dof_values() const { return m_dof_values.cview(); }

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        built = true;
    }

    void do_update_dof_attributes(UpdateDofAttributesInfo& info) override
    {
        (void)info;
        ++update_count;
        event_log().push("joint_dof_update");

        m_dof_offsets.clear();
        m_dof_counts.clear();

        std::vector<Float> values;
        IndexT             offset = 0;
        for(auto& slot : world().scene().geometries())
        {
            auto sc = slot->geometry().as<SimplicialComplex>();
            if(!sc)
            {
                continue;
            }
            auto count_attr = sc->meta().find<IndexT>(test_attr::joint_count);
            if(!count_attr)
            {
                continue;
            }
            const IndexT count = view(*count_attr)[0];
            auto         d_off = sc->meta().find<IndexT>(builtin::dof_offset);
            if(!d_off)
            {
                d_off = sc->meta().create<IndexT>(builtin::dof_offset, 0);
            }
            auto d_cnt = sc->meta().find<IndexT>(builtin::dof_count);
            if(!d_cnt)
            {
                d_cnt = sc->meta().create<IndexT>(builtin::dof_count, 0);
            }
            view(*d_off)[0] = offset;
            view(*d_cnt)[0] = count;

            m_dof_offsets.push_back(offset);
            m_dof_counts.push_back(count);

            auto  angle_attr = sc->instances().find<Float>(test_attr::joint_angle);
            Float angle      = angle_attr ? view(*angle_attr)[0] : 0.0;
            // one persistent scalar per dof: dof k of a joint holds angle + k
            for(IndexT k = 0; k < count; ++k)
            {
                values.push_back(angle + static_cast<Float>(k));
            }
            offset += count;
        }

        m_dof_values.resize(values.size());
        if(!values.empty())
        {
            m_dof_values.view().copy_from(span<const Float>{values}, &gg::frame_stream());
        }
    }

  private:
    std::vector<IndexT>       m_dof_offsets;
    std::vector<IndexT>       m_dof_counts;
    lud::LDeviceBuffer<Float> m_dof_values;
};
REGISTER_SIM_SYSTEM(HalfPlaneAnimatorTestJointDofReporter);

bool HalfPlaneAnimatorTestJointDofReporter::built        = false;
int  HalfPlaneAnimatorTestJointDofReporter::update_count = 0;

// ---------------------------------------------------------------------------
//  Time integrators: one per BDF flag, so the flag decides which one survives
// ---------------------------------------------------------------------------
template <typename FlagT>
class HalfPlaneAnimatorTestTimeIntegrator final : public gg::TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    static bool               built;
    static int                init_count;
    static std::vector<Float> predict_dt;
    static std::vector<Float> update_dt;

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        // The reference's shutdown idiom: requiring a flag system that does not
        // exist throws SimSystemException, which build_systems() turns into
        // "this system is invalid" (and, transitively, everything requiring it).
        require<FlagT>();
        built = true;
    }

    void do_init(InitInfo& info) override
    {
        (void)info;
        ++init_count;
        event_log().push("integrator_init");
    }

    void do_predict_dof(PredictDofInfo& info) override
    {
        predict_dt.push_back(info.dt());
        event_log().push("integrator_predict_dof");
    }

    void do_update_state(UpdateVelocityInfo& info) override
    {
        update_dt.push_back(info.dt());
        event_log().push("integrator_update_state");
    }
};

using HalfPlaneAnimatorTestBDF1Integrator = HalfPlaneAnimatorTestTimeIntegrator<gg::BDF1Flag>;
using HalfPlaneAnimatorTestBDF2Integrator = HalfPlaneAnimatorTestTimeIntegrator<gg::BDF2Flag>;

REGISTER_SIM_SYSTEM(HalfPlaneAnimatorTestBDF1Integrator);
REGISTER_SIM_SYSTEM(HalfPlaneAnimatorTestBDF2Integrator);

template <typename FlagT>
bool HalfPlaneAnimatorTestTimeIntegrator<FlagT>::built = false;
template <typename FlagT>
int HalfPlaneAnimatorTestTimeIntegrator<FlagT>::init_count = 0;
template <typename FlagT>
std::vector<Float> HalfPlaneAnimatorTestTimeIntegrator<FlagT>::predict_dt;
template <typename FlagT>
std::vector<Float> HalfPlaneAnimatorTestTimeIntegrator<FlagT>::update_dt;

// ---------------------------------------------------------------------------
//  Probe system: exposes the ported systems to the test
// ---------------------------------------------------------------------------
class HalfPlaneAnimatorTestProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    static gg::GlobalVertexManager*            vertex_manager;
    static gg::GlobalBodyManager*              body_manager;
    static gg::GlobalSimplicialSurfaceManager* surface_manager;
    static gg::TimeIntegratorManager*          time_integrator_manager;
    static gg::GlobalJointDofManager*          joint_dof_manager;
    static gg::GlobalExternalForceManager*     external_force_manager;
    static gg::GlobalAnimator*                 global_animator;
    static gg::HalfPlane*                      half_plane;
    static gg::HalfPlaneBodyReporter*          half_plane_body_reporter;
    static gg::HalfPlaneVertexReporter*        half_plane_vertex_reporter;
    static gg::IPCPipelineFlag*                ipc_flag;
    static gg::ALIPCPipelineFlag*              al_ipc_flag;
    static gg::BDF1Flag*                       bdf1_flag;
    static gg::BDF2Flag*                       bdf2_flag;
    static gg::SimEngine*                      sim_engine;

    static HalfPlaneAnimatorTestAnimator*             test_animator;
    static HalfPlaneAnimatorTestExternalForceReporter* force_reporter;
    static HalfPlaneAnimatorTestJointDofReporter*      joint_dof_reporter;
    static HalfPlaneAnimatorTestVertexReporter*        test_vertex_reporter;

  private:
    void do_build() override
    {
        vertex_manager          = &require<gg::GlobalVertexManager>();
        body_manager            = &require<gg::GlobalBodyManager>();
        time_integrator_manager = &require<gg::TimeIntegratorManager>();

        surface_manager        = find<gg::GlobalSimplicialSurfaceManager>();
        joint_dof_manager      = find<gg::GlobalJointDofManager>();
        external_force_manager = find<gg::GlobalExternalForceManager>();
        global_animator        = find<gg::GlobalAnimator>();
        half_plane             = find<gg::HalfPlane>();
        half_plane_body_reporter   = find<gg::HalfPlaneBodyReporter>();
        half_plane_vertex_reporter = find<gg::HalfPlaneVertexReporter>();
        ipc_flag               = find<gg::IPCPipelineFlag>();
        al_ipc_flag            = find<gg::ALIPCPipelineFlag>();
        bdf1_flag              = find<gg::BDF1Flag>();
        bdf2_flag              = find<gg::BDF2Flag>();
        sim_engine             = &engine();

        test_animator        = find<HalfPlaneAnimatorTestAnimator>();
        force_reporter       = find<HalfPlaneAnimatorTestExternalForceReporter>();
        joint_dof_reporter   = find<HalfPlaneAnimatorTestJointDofReporter>();
        test_vertex_reporter = find<HalfPlaneAnimatorTestVertexReporter>();
    }
};
REGISTER_SIM_SYSTEM(HalfPlaneAnimatorTestProbe);

gg::GlobalVertexManager*            HalfPlaneAnimatorTestProbe::vertex_manager          = nullptr;
gg::GlobalBodyManager*              HalfPlaneAnimatorTestProbe::body_manager            = nullptr;
gg::GlobalSimplicialSurfaceManager* HalfPlaneAnimatorTestProbe::surface_manager         = nullptr;
gg::TimeIntegratorManager*          HalfPlaneAnimatorTestProbe::time_integrator_manager = nullptr;
gg::GlobalJointDofManager*          HalfPlaneAnimatorTestProbe::joint_dof_manager       = nullptr;
gg::GlobalExternalForceManager*     HalfPlaneAnimatorTestProbe::external_force_manager  = nullptr;
gg::GlobalAnimator*                 HalfPlaneAnimatorTestProbe::global_animator         = nullptr;
gg::HalfPlane*                      HalfPlaneAnimatorTestProbe::half_plane              = nullptr;
gg::HalfPlaneBodyReporter*          HalfPlaneAnimatorTestProbe::half_plane_body_reporter = nullptr;
gg::HalfPlaneVertexReporter*        HalfPlaneAnimatorTestProbe::half_plane_vertex_reporter = nullptr;
gg::IPCPipelineFlag*                HalfPlaneAnimatorTestProbe::ipc_flag                = nullptr;
gg::ALIPCPipelineFlag*              HalfPlaneAnimatorTestProbe::al_ipc_flag             = nullptr;
gg::BDF1Flag*                       HalfPlaneAnimatorTestProbe::bdf1_flag               = nullptr;
gg::BDF2Flag*                       HalfPlaneAnimatorTestProbe::bdf2_flag               = nullptr;
gg::SimEngine*                      HalfPlaneAnimatorTestProbe::sim_engine              = nullptr;

HalfPlaneAnimatorTestAnimator*              HalfPlaneAnimatorTestProbe::test_animator        = nullptr;
HalfPlaneAnimatorTestExternalForceReporter* HalfPlaneAnimatorTestProbe::force_reporter      = nullptr;
HalfPlaneAnimatorTestJointDofReporter*      HalfPlaneAnimatorTestProbe::joint_dof_reporter   = nullptr;
HalfPlaneAnimatorTestVertexReporter*        HalfPlaneAnimatorTestProbe::test_vertex_reporter = nullptr;

namespace
{
/// Reset every test-system counter; call before creating an engine.
void reset_test_systems()
{
    event_log().clear();
    HalfPlaneAnimatorTestAnimator::built             = false;
    HalfPlaneAnimatorTestAnimator::init_count        = 0;
    HalfPlaneAnimatorTestAnimator::step_count        = 0;
    HalfPlaneAnimatorTestAnimator::ratio_during_step = -1.0;
    HalfPlaneAnimatorTestExternalForceReporter::init_count  = 0;
    HalfPlaneAnimatorTestExternalForceReporter::clear_count = 0;
    HalfPlaneAnimatorTestExternalForceReporter::step_count  = 0;
    HalfPlaneAnimatorTestJointDofReporter::built        = false;
    HalfPlaneAnimatorTestJointDofReporter::update_count = 0;
    HalfPlaneAnimatorTestBDF1Integrator::built      = false;
    HalfPlaneAnimatorTestBDF1Integrator::init_count = 0;
    HalfPlaneAnimatorTestBDF1Integrator::predict_dt.clear();
    HalfPlaneAnimatorTestBDF1Integrator::update_dt.clear();
    HalfPlaneAnimatorTestBDF2Integrator::built      = false;
    HalfPlaneAnimatorTestBDF2Integrator::init_count = 0;
    HalfPlaneAnimatorTestBDF2Integrator::predict_dt.clear();
    HalfPlaneAnimatorTestBDF2Integrator::update_dt.clear();
}

/// The per-frame event order the reference's advance() implies for the ported
/// systems (see SimEngine::advance()). `frontend_animation` only exists when the
/// scene actually has an animation callback (GlobalAnimator::step() first
/// updates the frontend animator, then fans out to the backend animators).
bool frame_event_order_ok(bool has_animator, bool has_animation_callback = false)
{
    std::vector<std::string> expected{"vertex_attributes", "force_clear"};
    if(has_animator)
    {
        if(has_animation_callback)
        {
            expected.push_back("frontend_animation");
        }
        expected.push_back("animator_step");
    }
    expected.push_back("force_step");
    expected.push_back("integrator_predict_dof");
    expected.push_back("integrator_update_state");
    return is_subsequence(expected, event_log().events);
}
}// namespace

// ===========================================================================
//  1) half plane geometry / vertex sets / body set
// ===========================================================================
TEST_CASE("half_plane_geometry_and_vertex_sets", "[luisa][half_plane]")
{
    reset_test_systems();

    SceneDesc desc;
    // two half planes (in scene order), then one triangle mesh
    desc.half_planes = {
        HalfPlaneDesc{.P = {0.0, -1.0, 0.0}, .N = {0.0, 1.0, 0.0}, .contact_id = 11, .subscene_id = 21},
        HalfPlaneDesc{.P = {0.0, -3.0, 0.0}, .N = {0.0, 1.0, 0.0}, .contact_id = 12, .subscene_id = 22}};

    MeshDesc mesh;
    mesh.positions    = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    mesh.triangles    = {{0, 1, 2}};
    mesh.dimensions   = {2, 2, 0};
    mesh.contact_ids  = {31, 32, 33};
    mesh.subscene_ids = {41, 42, 43};
    mesh.d_hats       = {0.02, 0.03, 0.04};
    mesh.thicknesses  = {0.1, 0.2, 0.3};
    desc.meshes.push_back(mesh);

    auto   scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_test")};
    World  world{engine};
    world.init(scene);

    auto& vm = *HalfPlaneAnimatorTestProbe::vertex_manager;
    auto& bm = *HalfPlaneAnimatorTestProbe::body_manager;
    auto& sm = *HalfPlaneAnimatorTestProbe::surface_manager;

    uipc::backend::WorldVisitor visitor{world};
    auto                        model = compute_vertex_model(visitor);

    auto* hp_vr = HalfPlaneAnimatorTestProbe::half_plane_vertex_reporter;
    auto* hp_br = HalfPlaneAnimatorTestProbe::half_plane_body_reporter;
    auto* my_vr = HalfPlaneAnimatorTestProbe::test_vertex_reporter;
    REQUIRE(hp_vr != nullptr);
    REQUIRE(hp_br != nullptr);
    REQUIRE(my_vr != nullptr);

    // ---- the half plane's own reported sets --------------------------------
    // the ported HalfPlaneVertexReporter's uid is 2, so its slice starts at 0
    CHECK(hp_vr->vertex_offset() == 0);
    CHECK(hp_vr->vertex_count() == 2);  // one vertex per half plane instance
    // The reference's init_scene() calls GlobalBodyManager::init() *before*
    // event_init_scene(), and the HalfPlane fills its instance arrays (the ones
    // HalfPlaneBodyReporter::report_count() reads) from its on_init_scene()
    // action — so the half plane contributes **0 bodies**: its body slice is
    // empty and only body_offset (its would-be start) is set. Reference parity:
    // the port keeps the reference's call order and the same host-side code.
    CHECK(hp_br->body_count() == 0);
    const IndexT hp_body_offset = hp_br->body_offset();

    // ---- the half plane slice of the global vertex arrays ------------------
    auto coindices = read_device(vm.coindices());
    auto body_ids  = read_device(vm.body_ids());
    auto positions = read_vector3(vm.positions());
    REQUIRE(coindices.size() >= 5);
    REQUIRE(positions.size() >= 5);

    CHECK(slice(coindices, 0, 2) == std::vector<IndexT>{0, 1});
    CHECK(eigen_equal(slice(positions, 0, 2),
                      std::vector<Vector3>{Vector3{0.0, -1.0, 0.0}, Vector3{0.0, -3.0, 0.0}}));
    // each vertex *is* a body, so the reporter writes body_offset + local index
    // (with the 0-body quirk above these ids are beyond the reported body array;
    // the reference's reporter does exactly the same)
    CHECK(slice(body_ids, 0, 2)
          == std::vector<IndexT>{hp_body_offset, hp_body_offset + 1});
    // the per-instance contact/subscene ids come from the geometry metas
    CHECK(slice(read_device(vm.contact_element_ids()), 0, 2) == std::vector<IndexT>{11, 12});
    CHECK(slice(read_device(vm.subscene_element_ids()), 0, 2) == std::vector<IndexT>{21, 22});
    // the reference reporter leaves the other per-vertex attributes at the
    // manager's defaults
    CHECK(slice(read_device(vm.dimensions()), 0, 2) == std::vector<IndexT>{3, 3});
    CHECK(slice(read_device(vm.d_hats()), 0, 2) == std::vector<Float>{0.01, 0.01});
    CHECK(slice(read_device(vm.thicknesses()), 0, 2) == std::vector<Float>{0.0, 0.0});
    CHECK(eigen_equal(slice(read_vector3(vm.displacements()), 0, 2),
                      std::vector<Vector3>{Vector3::Zero(), Vector3::Zero()}));

    // ---- the half plane contributes no body to the body arrays -------------
    // (see the body_count() note above: its slice is empty, so the body arrays
    // hold only the other body reporters' bodies)
    CHECK(slice(read_device(bm.coindices()), hp_body_offset, 2).empty());
    CHECK(slice(read_device(bm.self_collision()), hp_body_offset, 2).empty());

    // ---- this test's own slice (the triangle mesh) -------------------------
    const IndexT my_offset = my_vr->vertex_offset();
    CHECK(my_vr->vertex_count() == static_cast<IndexT>(model.complex_count()));
    CHECK(my_offset == static_cast<IndexT>(model.half_plane_count()));
    CHECK(slice(coindices, my_offset, model.complex_count()) == model.complex_coindices);
    CHECK(slice(body_ids, my_offset, model.complex_count()) == model.complex_body_ids);
    CHECK(eigen_equal(slice(positions, my_offset, model.complex_count()),
                      model.complex_positions));
    CHECK(slice(read_device(vm.dimensions()), my_offset, model.complex_count())
          == model.complex_dimensions);
    CHECK(slice(read_device(vm.contact_element_ids()), my_offset, model.complex_count())
          == model.complex_contact_ids);
    CHECK(slice(read_device(vm.subscene_element_ids()), my_offset, model.complex_count())
          == model.complex_subscene_ids);
    CHECK(slice(read_device(vm.d_hats()), my_offset, model.complex_count())
          == model.complex_d_hats);
    CHECK(slice(read_device(vm.thicknesses()), my_offset, model.complex_count())
          == model.complex_thicknesses);

    // ---- the half plane's own device buffers -------------------------------
    auto& hp = *HalfPlaneAnimatorTestProbe::half_plane;
    REQUIRE(hp.positions().size() == 2);
    CHECK(eigen_equal(read_vector3(hp.positions()), model.half_plane_positions));
    CHECK(eigen_equal(read_vector3(hp.normals()),
                      std::vector<Vector3>{Vector3{0.0, 1.0, 0.0}, Vector3{0.0, 1.0, 0.0}}));

    // ---- geo_infos: slot index / id / vertex count / offset ----------------
    auto geo_infos = hp.geo_infos();
    REQUIRE(geo_infos.size() == 2);
    {
        std::vector<IndexT> slot_indices;
        std::vector<IndexT> geo_ids;
        std::vector<IndexT> offsets;
        std::vector<IndexT> counts;
        for(auto& info : geo_infos)
        {
            slot_indices.push_back(info.geo_slot_index);
            geo_ids.push_back(info.geo_id);
            offsets.push_back(info.vertex_offset);
            counts.push_back(info.vertex_count);
        }
        // hand-computed from the scene: the two half planes are the first two
        // geometries created, one instance each
        auto slots = visitor.scene().geometries();
        CHECK(slot_indices == std::vector<IndexT>{0, 1});
        CHECK(geo_ids == std::vector<IndexT>{slots[0]->id(), slots[1]->id()});
        CHECK(counts == std::vector<IndexT>{1, 1});
        CHECK(offsets == std::vector<IndexT>{0, 1});
    }

    // ---- the reporter writes the global vertex offset on the geometry meta -
    {
        auto igs = scene_implicit_geometries(visitor);
        REQUIRE(igs.size() == 2);
        for(size_t i = 0; i < igs.size(); ++i)
        {
            auto gvo = igs[i]->meta().find<IndexT>(builtin::global_vertex_offset);
            REQUIRE(gvo != nullptr);
            // [half plane local offset] + [half plane slice offset (0)]
            CHECK(view(*gvo)[0] == static_cast<IndexT>(i));
        }
    }

    // ---- the surface: this test's slice, with "[0, 1, 2]-style" indices ----
    auto surf = compute_surface_model(visitor, my_vr->vertex_offset());
    REQUIRE(surf.surf_vertices.size() == 3);
    REQUIRE(surf.surf_triangles.size() == 1);
    // the triangle's global vertex indices are the slice offset + {0, 1, 2}
    CHECK(surf.surf_triangles[0] == Vector3i{my_offset + 0, my_offset + 1, my_offset + 2});
    // the reporter's own computation matches this file's host model ...
    CHECK(HalfPlaneAnimatorTestSurfaceReporter::reported_surf_vertices == surf.surf_vertices);
    CHECK(eigen_equal(HalfPlaneAnimatorTestSurfaceReporter::reported_surf_edges, surf.surf_edges));
    CHECK(eigen_equal(HalfPlaneAnimatorTestSurfaceReporter::reported_surf_triangles,
                      surf.surf_triangles));
    // ... and its slice is present in the manager's arrays
    CHECK(contains_subsequence(read_device(sm.surf_vertices()), surf.surf_vertices));
    CHECK(contains_subsequence(read_vector2i(sm.surf_edges()), surf.surf_edges));
    CHECK(contains_subsequence(read_vector3i(sm.surf_triangles()), surf.surf_triangles));

    // ---- state at frame 0 (before any advance) ----------------------------
    CHECK(world.frame() == 0);
    CHECK(HalfPlaneAnimatorTestBDF1Integrator::built);  // config default: bdf1
    CHECK(!HalfPlaneAnimatorTestBDF2Integrator::built);
    CHECK(HalfPlaneAnimatorTestBDF1Integrator::init_count == 1);
    CHECK(HalfPlaneAnimatorTestBDF1Integrator::predict_dt.empty());  // no frame yet
    // no Constraint constitution in this scene -> no GlobalAnimator
    CHECK(HalfPlaneAnimatorTestProbe::global_animator == nullptr);
    CHECK(!HalfPlaneAnimatorTestAnimator::built);
}

// ===========================================================================
//  2) animator: hand-computed keyframes, frame by frame
// ===========================================================================
TEST_CASE("half_plane_animator_keyframes", "[luisa][half_plane]")
{
    reset_test_systems();

    // initial scene positions (deliberately different from the first keyframe)
    const std::vector<Vector3> P_init = {{0.5, 0.5, 0.5}, {1.5, 0.5, 0.5}, {0.5, 1.5, 0.5}};
    // two keyframes: t = 0 -> P0, t = 1 -> P1 (all coordinates exact in binary,
    // so the interpolation is exactly representable at every alpha below)
    const std::vector<Vector3> P0 = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    const std::vector<Vector3> P1 = {{4.0, 0.0, 2.0}, {7.0, 3.0, 2.0}, {2.0, 6.0, 2.0}};
    const SizeT                N  = 3;

    SceneDesc desc;
    desc.dt = 0.25;  // t = dt * frame, so alpha = t at frames 1..4 -> 0.25/0.5/0.75/1
    desc.half_planes.push_back(HalfPlaneDesc{.P = {0.0, -2.0, 0.0}});

    MeshDesc mesh;
    mesh.positions       = P_init;
    mesh.triangles       = {{0, 1, 2}};
    mesh.dimensions      = {2, 2, 2};
    mesh.d_hats          = {0.1, 0.2, 0.3};
    mesh.with_constraint = true;  // -> "Constraint" -> GlobalAnimator exists
    desc.meshes.push_back(mesh);

    std::vector<S<Object>> mesh_objects;
    auto                   scene = build_scene(desc, &mesh_objects);
    REQUIRE(mesh_objects.size() == 1);

    // the animation: evaluate the two keyframes at t = dt * frame through the
    // *ported* animator law (animator::lerp = clamp + src*(1-a) + dst*a)
    std::vector<SizeT> animated_frames;
    auto               animate = [&, P0, P1](Animation::UpdateInfo& info)
    {
        animated_frames.push_back(info.frame());
        event_log().push("frontend_animation");
        auto  sc  = info.geo_slots()[0]->geometry().as<SimplicialComplex>();
        auto& pos = sc->positions();
        const Float t     = info.dt() * static_cast<Float>(info.frame());
        const Float alpha = (t - 0.0) / (1.0 - 0.0);  // keyframes at t = 0 and t = 1
        for(SizeT v = 0; v < N; ++v)
        {
            auto r       = gg::animator::lerp(gg::to_luisa(P0[v]), gg::to_luisa(P1[v]), alpha);
            view(pos)[v] = gg::to_eigen(r);
        }
    };
    scene.animator().insert(*mesh_objects[0], animate);

    Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_animator_test")};
    World  world{engine};
    world.init(scene);

    auto& vm = *HalfPlaneAnimatorTestProbe::vertex_manager;
    auto* my_vr = HalfPlaneAnimatorTestProbe::test_vertex_reporter;
    REQUIRE(my_vr != nullptr);
    const SizeT my_offset = static_cast<SizeT>(my_vr->vertex_offset());

    uipc::backend::WorldVisitor visitor{world};

    REQUIRE(HalfPlaneAnimatorTestProbe::global_animator != nullptr);
    CHECK(HalfPlaneAnimatorTestAnimator::built);
    CHECK(HalfPlaneAnimatorTestAnimator::init_count == 1);
    CHECK(HalfPlaneAnimatorTestAnimator::step_count == 0);  // frame 0: not stepped yet
    // the animator's ratio starts at the reference's initial value
    CHECK(HalfPlaneAnimatorTestProbe::global_animator->substep_ratio() == 1.0);
    CHECK(animated_frames.empty());
    // nothing has animated yet: the device holds the initial scene state
    CHECK(eigen_equal(slice(read_vector3(vm.positions()), my_offset, N), P_init));

    std::vector<Vector3> last_scene = P_init;

    for(SizeT k = 1; k <= 6; ++k)
    {
        // The state this frame's update_attributes() must report is the scene as
        // the *previous* frame's animation left it (the reference calls
        // update_attributes() in the frame prologue, before the animation step).
        auto model_before = compute_vertex_model(visitor);

        event_log().clear();
        world.advance();
        world.sync();

        CHECK(world.frame() == k);

        // (a) the device positions of this test's slice are exactly the
        //     pre-advance scene state ...
        CHECK(eigen_equal(slice(read_vector3(vm.positions()), my_offset, N),
                          model_before.complex_positions));
        // ... and the half plane's slice is never animated
        CHECK(eigen_equal(slice(read_vector3(vm.positions()), 0, 1),
                          std::vector<Vector3>{Vector3{0.0, -2.0, 0.0}}));

        // (b) the scene now holds the hand-computed keyframe evaluation A(k)
        const Float          alpha_k = static_cast<Float>(k) * desc.dt;
        std::vector<Vector3> expect_scene;
        for(SizeT v = 0; v < N; ++v)
        {
            expect_scene.push_back(hand_lerp(P0[v], P1[v], alpha_k));
        }
        auto model_after = compute_vertex_model(visitor);
        CHECK(eigen_equal(model_after.complex_positions, expect_scene));
        last_scene = model_after.complex_positions;

        // (c) the device mesh slice is the previous frame's animation state:
        //     frame 1 still holds the initial positions, frame k holds A(k-1)
        std::vector<Vector3> expect_device;
        if(k == 1)
        {
            expect_device = P_init;
        }
        else
        {
            const Float alpha_prev = static_cast<Float>(k - 1) * desc.dt;
            for(SizeT v = 0; v < N; ++v)
            {
                expect_device.push_back(hand_lerp(P0[v], P1[v], alpha_prev));
            }
        }
        CHECK(eigen_equal(slice(read_vector3(vm.positions()), my_offset, N), expect_device));

        // the law, restated at the interesting frame boundaries
        if(k == 2)  // t = 0.5
            CHECK(eigen_equal(model_after.complex_positions,
                              std::vector<Vector3>{hand_lerp(P0[0], P1[0], 0.5),
                                                   hand_lerp(P0[1], P1[1], 0.5),
                                                   hand_lerp(P0[2], P1[2], 0.5)}));
        if(k == 4)  // t = 1
            CHECK(eigen_equal(model_after.complex_positions, P1));
        if(k == 5 || k == 6)  // past the end: clamped, not extrapolated
            CHECK(eigen_equal(model_after.complex_positions, P1));

        // (d) the animation callback ran exactly once per frame with frame() == k
        REQUIRE(animated_frames.size() == k);
        CHECK(animated_frames.back() == k);

        // (e) the animator's per-frame state machine
        CHECK(HalfPlaneAnimatorTestAnimator::step_count == static_cast<int>(k));
        CHECK(HalfPlaneAnimatorTestAnimator::ratio_during_step == 0.0);  // reset by step()
        // ... and recomputed for the frame's (only) Newton iteration 0:
        //     min((0 + 1) / substep, 1). The frontend substep is 1 for the first
        //     three frames and 4 afterwards (changed below), so the hand-computed
        //     ratio flips from 1.0 to 0.25 from frame 4 on.
        const Float substep     = (k <= 3) ? Float{1.0} : Float{4.0};
        const Float expect_ratio = std::min(Float{1.0} / substep, Float{1.0});
        CHECK(HalfPlaneAnimatorTestProbe::global_animator->substep_ratio() == expect_ratio);

        // (f) the rest state is not animated
        CHECK(eigen_equal(slice(read_vector3(vm.rest_positions()), my_offset, N), P_init));

        // (g) the frame's event order is the reference's
        CHECK(frame_event_order_ok(true, true));

        // (h) change the frontend substep mid-run for the next frames
        if(k == 3)
        {
            scene.animator().substep(4);
            CHECK(scene.animator().substep() == 4);
        }
    }

    // the last frame's scene state (past the end of the keyframes) is the last
    // keyframe, held - not extrapolated
    CHECK(eigen_equal(last_scene, P1));
    CHECK(animated_frames == std::vector<SizeT>{1, 2, 3, 4, 5, 6});
    CHECK(HalfPlaneAnimatorTestExternalForceReporter::clear_count == 6);
    CHECK(HalfPlaneAnimatorTestExternalForceReporter::step_count == 6);
    CHECK(HalfPlaneAnimatorTestBDF1Integrator::predict_dt
          == std::vector<Float>{0.25, 0.25, 0.25, 0.25, 0.25, 0.25});
    CHECK(HalfPlaneAnimatorTestBDF1Integrator::update_dt.size() == 6);
}

// ===========================================================================
//  2b) the animator law itself: host containers and the device (DSL) instantiation
// ===========================================================================
TEST_CASE("half_plane_animator_lerp_law", "[luisa][half_plane]")
{
    test_runtime();  // the device part needs the LC runtime

    const std::vector<Float> alphas = {-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};

    // ---- (a) scalar -------------------------------------------------------
    for(auto a : alphas)
    {
        CHECK(gg::animator::lerp(0.0, 8.0, a) == hand_lerp(0.0, 8.0, a));
        CHECK(gg::animator::lerp(-4.0, 4.0, a) == hand_lerp(-4.0, 4.0, a));
        // a single keyframe stays put for every alpha
        CHECK(gg::animator::lerp(3.0, 3.0, a) == 3.0);
    }

    // ---- (b) LVec (the animator's Vector3/Vector12 analogue) --------------
    const Vector3 s3 = {0.0, -1.0, 2.0};
    const Vector3 d3 = {4.0, 3.0, -6.0};
    for(auto a : alphas)
    {
        auto got = gg::animator::lerp(gg::to_luisa(s3), gg::to_luisa(d3), a);
        auto exp = hand_lerp(s3, d3, a);
        bool ok  = true;
        for(int c = 0; c < 3; ++c)
        {
            ok = ok && (got(c) == exp[c]);
        }
        CHECK(ok);
    }

    // ---- (c) LVec<T,4> and LMat<T,4,4> (the transform-shaped containers) ---
    {
        lps::LVec<double, 4> s4{1.0, 2.0, 3.0, 4.0};
        lps::LVec<double, 4> d4{5.0, 6.0, 7.0, 8.0};
        auto                 got = gg::animator::lerp(s4, d4, 0.25);
        for(int c = 0; c < 4; ++c)
        {
            CHECK(got(c) == hand_lerp(s4(c), d4(c), 0.25));
        }
        auto clamped = gg::animator::lerp(s4, d4, 5.0);
        for(int c = 0; c < 4; ++c)
        {
            CHECK(clamped(c) == d4(c));  // alpha clamped to 1
        }

        lps::LMat<double, 4, 4> m0{};
        lps::LMat<double, 4, 4> m1{};
        for(int r = 0; r < 4; ++r)
        {
            for(int c = 0; c < 4; ++c)
            {
                m0(r, c) = static_cast<double>(r * 4 + c);
                m1(r, c) = static_cast<double>(100 + r * 4 + c);
            }
        }
        auto m = gg::animator::lerp(m0, m1, 0.5);
        for(int r = 0; r < 4; ++r)
        {
            for(int c = 0; c < 4; ++c)
            {
                CHECK(m(r, c) == hand_lerp(m0(r, c), m1(r, c), 0.5));
            }
        }
        auto m_clamped = gg::animator::lerp(m0, m1, -3.0);
        CHECK(m_clamped(3, 3) == m0(3, 3));  // alpha clamped to 0
    }

    // ---- (d) the device (DSL) instantiation, one dispatch -----------------
    {
        const SizeT n = 7;
        // integer-valued inputs: every intermediate is exact, so an exact
        // comparison is valid even if the device contracts the multiply-add
        std::vector<gg::LVector3> s(n);
        std::vector<gg::LVector3> d(n);
        std::vector<double>       a(n);
        for(SizeT i = 0; i < n; ++i)
        {
            const double fi = static_cast<double>(i);
            s[i]            = gg::to_luisa(Vector3{fi, 2.0 * fi, -fi});
            d[i]            = gg::to_luisa(Vector3{3.0 * fi, fi + 4.0, 2.0 * fi});
            a[i]            = alphas[i % alphas.size()];
        }

        lud::LDeviceBuffer<gg::LVector3> s_buf(span<const gg::LVector3>{s});
        lud::LDeviceBuffer<gg::LVector3> d_buf(span<const gg::LVector3>{d});
        lud::LDeviceBuffer<double>       a_buf(span<const double>{a});
        lud::LDeviceBuffer<gg::LVector3> out_buf;
        out_buf.resize(n);

        auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector3>,
                                             lcrt::Buffer<gg::LVector3>,
                                             lcrt::Buffer<gg::LVector3>,
                                             lcrt::Buffer<double>,
                                             uint32_t>(
            [](lcrt::BufferVar<gg::LVector3> out,
               lcrt::BufferVar<gg::LVector3> src,
               lcrt::BufferVar<gg::LVector3> dst,
               lcrt::BufferVar<double>       alpha,
               lcrt::Var<uint32_t>           count) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < count)
                {
                    auto                    s = gg::load_vec3(src->read(i));
                    auto                    d = gg::load_vec3(dst->read(i));
                    auto                    t = alpha->read(i);
                    auto                    r = gg::animator::lerp(s, d, t);
                    lcrt::Var<gg::LVector3> o;
                    gg::store_vec3(o, r);
                    out->write(i, o);
                };
            });

        const uint32_t count = static_cast<uint32_t>(n);
        gg::frame_stream().record(
            k(out_buf.view(), s_buf.cview(), d_buf.cview(), a_buf.cview(), count)
                .dispatch(count));

        auto got = read_vector3(out_buf.cview());
        REQUIRE(got.size() == n);
        for(SizeT i = 0; i < n; ++i)
        {
            const auto exp = hand_lerp(gg::to_eigen(s[i]), gg::to_eigen(d[i]), a[i]);
            CHECK((got[i].array() == exp.array()).all());
        }
    }
}

// ===========================================================================
//  3) pipeline + time-integrator flags
// ===========================================================================
namespace
{
struct FlagExpectation
{
    bool bdf1   = true;
    bool bdf2   = false;
    bool ipc    = true;
    bool al_ipc = false;
};

/// The flags the scene config selects, computed on the host from the config
/// alone (the ported SimSystemCreators test the same two keys).
FlagExpectation expected_flags(const SceneDesc& desc)
{
    FlagExpectation e;
    e.bdf1   = desc.integrator_type == "bdf1";
    e.bdf2   = desc.integrator_type == "bdf2";
    e.ipc    = desc.contact_constitution == "ipc";
    e.al_ipc = desc.contact_constitution == "al-ipc";
    return e;
}

void check_probe_flags(const FlagExpectation& e)
{
    CHECK((HalfPlaneAnimatorTestProbe::bdf1_flag != nullptr) == e.bdf1);
    CHECK((HalfPlaneAnimatorTestProbe::bdf2_flag != nullptr) == e.bdf2);
    CHECK((HalfPlaneAnimatorTestProbe::ipc_flag != nullptr) == e.ipc);
    CHECK((HalfPlaneAnimatorTestProbe::al_ipc_flag != nullptr) == e.al_ipc);
}

/// One scene with a half plane + a constrained mesh, so the animator and the
/// integrator take part in the frame as well.
SceneDesc flag_scene_desc(const std::string& integrator_type,
                          const std::string& contact_constitution,
                          Float              dt)
{
    SceneDesc desc;
    desc.integrator_type      = integrator_type;
    desc.contact_constitution = contact_constitution;
    desc.dt                   = dt;
    desc.half_planes.push_back(HalfPlaneDesc{.P = {0.0, -1.0, 0.0}});
    MeshDesc mesh;
    mesh.positions       = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    mesh.triangles       = {{0, 1, 2}};
    mesh.with_constraint = true;
    desc.meshes.push_back(mesh);
    return desc;
}
}// namespace

TEST_CASE("half_plane_pipeline_and_flags", "[luisa][half_plane]")
{
    // (a) the default config: integrator/type = "bdf1", contact/constitution = "ipc"
    {
        reset_test_systems();

        auto desc = flag_scene_desc("bdf1", "ipc", 0.25);
        std::vector<S<Object>> mesh_objects;
        auto                   scene = build_scene(desc, &mesh_objects);
        REQUIRE(mesh_objects.size() == 1);
        int  animation_calls = 0;
        auto animate         = [&](Animation::UpdateInfo& info)
        {
            (void)info;
            ++animation_calls;
            event_log().push("frontend_animation");
        };
        scene.animator().insert(*mesh_objects[0], animate);

        Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_flags_test")};
        World  world{engine};
        world.init(scene);

        auto expected = expected_flags(desc);
        check_probe_flags(expected);
        CHECK(HalfPlaneAnimatorTestProbe::sim_engine->pipeline_type()
              == gg::SimEngine::PipelineType::Basic);
        // exactly the integrator gated on the surviving flag is built
        CHECK(HalfPlaneAnimatorTestBDF1Integrator::built == expected.bdf1);
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::built == expected.bdf2);
        CHECK(HalfPlaneAnimatorTestBDF1Integrator::init_count == 1);
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::init_count == 0);
        CHECK(HalfPlaneAnimatorTestBDF1Integrator::predict_dt.empty());

        for(SizeT k = 1; k <= 3; ++k)
        {
            event_log().clear();
            world.advance();
            world.sync();

            // the flag systems are *scene* predicates: the host-recomputed
            // expectation must hold on every frame (they do not flip per frame)
            check_probe_flags(expected);
            CHECK(HalfPlaneAnimatorTestProbe::sim_engine->pipeline_type()
                  == gg::SimEngine::PipelineType::Basic);

            // the manager forwards the scene's dt to the surviving integrator,
            // once per frame in each phase
            REQUIRE(HalfPlaneAnimatorTestBDF1Integrator::predict_dt.size() == k);
            CHECK(HalfPlaneAnimatorTestBDF1Integrator::predict_dt.back() == desc.dt);
            REQUIRE(HalfPlaneAnimatorTestBDF1Integrator::update_dt.size() == k);
            CHECK(HalfPlaneAnimatorTestBDF1Integrator::update_dt.back() == desc.dt);
            CHECK(HalfPlaneAnimatorTestBDF2Integrator::predict_dt.empty());
            CHECK(HalfPlaneAnimatorTestBDF2Integrator::update_dt.empty());

            // the per-frame state that *does* flip: the animator's substep ratio
            // (0 inside GlobalAnimator::step(), then min((0+1)/substep, 1))
            REQUIRE(HalfPlaneAnimatorTestAnimator::step_count == static_cast<int>(k));
            CHECK(HalfPlaneAnimatorTestAnimator::ratio_during_step == 0.0);
            CHECK(HalfPlaneAnimatorTestProbe::global_animator->substep_ratio() == 1.0);

            CHECK(animation_calls == static_cast<int>(k));
            CHECK(frame_event_order_ok(true, true));
        }
    }

    // (b) the mirror config: integrator/type = "bdf2", contact/constitution = "al-ipc"
    {
        reset_test_systems();

        auto desc = flag_scene_desc("bdf2", "al-ipc", 0.4);
        std::vector<S<Object>> mesh_objects;
        auto                   scene = build_scene(desc, &mesh_objects);
        REQUIRE(mesh_objects.size() == 1);
        int  animation_calls = 0;
        auto animate         = [&](Animation::UpdateInfo& info)
        {
            (void)info;
            ++animation_calls;
            event_log().push("frontend_animation");
        };
        scene.animator().insert(*mesh_objects[0], animate);

        Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_flags_test")};
        World  world{engine};
        world.init(scene);

        auto expected = expected_flags(desc);
        check_probe_flags(expected);
        CHECK(HalfPlaneAnimatorTestProbe::sim_engine->pipeline_type()
              == gg::SimEngine::PipelineType::AugmentedLagrangian);
        CHECK(!HalfPlaneAnimatorTestBDF1Integrator::built);
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::built);
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::init_count == 1);
        CHECK(HalfPlaneAnimatorTestBDF1Integrator::init_count == 0);

        for(SizeT k = 1; k <= 2; ++k)
        {
            event_log().clear();
            world.advance();
            world.sync();
            check_probe_flags(expected);
            REQUIRE(HalfPlaneAnimatorTestBDF2Integrator::predict_dt.size() == k);
            CHECK(HalfPlaneAnimatorTestBDF2Integrator::predict_dt.back() == 0.4);
            CHECK(HalfPlaneAnimatorTestBDF1Integrator::predict_dt.empty());
        }
    }
}

// ===========================================================================
//  4) external force: the clear -> animate -> step lifecycle
// ===========================================================================
TEST_CASE("half_plane_external_force", "[luisa][half_plane]")
{
    // (a) a synthetic force on a known vertex set, animated per frame
    {
        reset_test_systems();

        SceneDesc desc;
        desc.dt = 0.25;
        desc.half_planes.push_back(HalfPlaneDesc{.P = {0.0, -1.0, 0.0}});

        MeshDesc mesh;
        mesh.positions       = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        mesh.triangles       = {{0, 1, 2}};
        mesh.forces          = {{10.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, -20.0, 0.0}};
        mesh.force_enabled   = {1, 0, 1};  // vertices 0 and 2
        mesh.with_constraint = true;       // so an animation can drive the force
        desc.meshes.push_back(mesh);

        std::vector<S<Object>> mesh_objects;
        auto                   scene = build_scene(desc, &mesh_objects);
        REQUIRE(mesh_objects.size() == 1);

        // the animation writes the *force attribute* of vertex 0 (the reference's
        // external-force constitution does the same): F(frame) = (10 + frame, 0, 0)
        auto animate = [](Animation::UpdateInfo& info)
        {
            event_log().push("frontend_animation");
            auto sc    = info.geo_slots()[0]->geometry().as<SimplicialComplex>();
            auto force = sc->vertices().find<Vector3>(test_attr::force);
            view(*force)[0] = Vector3{10.0 + static_cast<Float>(info.frame()), 0.0, 0.0};
        };
        scene.animator().insert(*mesh_objects[0], animate);

        Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_force_test")};
        World  world{engine};
        world.init(scene);

        auto* reporter = HalfPlaneAnimatorTestProbe::force_reporter;
        REQUIRE(reporter != nullptr);
        CHECK(HalfPlaneAnimatorTestProbe::external_force_manager != nullptr);

        // the manager ran the reporters' init during init_scene
        CHECK(reporter->init_count == 1);
        CHECK(reporter->clear_count == 0);
        CHECK(reporter->step_count == 0);
        // ... and the reported set/values are what the test fed in
        CHECK(read_device(reporter->vertex_indices()) == std::vector<IndexT>{0, 2});
        CHECK(eigen_equal(read_vector3(reporter->forces()),
                          std::vector<Vector3>{Vector3{10.0, 0.0, 0.0},
                                               Vector3{0.0, -20.0, 0.0}}));

        for(SizeT k = 1; k <= 3; ++k)
        {
            event_log().clear();
            world.advance();
            world.sync();

            CHECK(reporter->clear_count == static_cast<int>(k));
            CHECK(reporter->step_count == static_cast<int>(k));
            // The force consumed in frame k is the value the animation wrote in
            // frame k: clear() runs before the animation and step() after it. A
            // step-before-animation order would leave the previous frame's value
            // here (11,0,0 at frame 2 instead of 12,0,0).
            CHECK(eigen_equal(read_vector3(reporter->forces()),
                              std::vector<Vector3>{Vector3{10.0 + static_cast<Float>(k), 0.0, 0.0},
                                                   Vector3{0.0, -20.0, 0.0}}));
            CHECK(read_device(reporter->vertex_indices()) == std::vector<IndexT>{0, 2});

            // the whole frame's event sequence is the reference's order
            std::vector<std::string> expected{"vertex_attributes",
                                              "force_clear",
                                              "frontend_animation",
                                              "animator_step",
                                              "force_step",
                                              "integrator_predict_dof",
                                              "integrator_update_state"};
            CHECK(event_log().events == expected);
        }
    }

    // (b) a force on an empty set: no enabled vertex at all
    {
        reset_test_systems();

        SceneDesc desc;
        desc.dt = 0.25;
        MeshDesc mesh;
        mesh.positions     = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        mesh.triangles     = {{0, 1, 2}};
        mesh.forces        = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        mesh.force_enabled = {0, 0, 0};
        desc.meshes.push_back(mesh);

        auto   scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_force_test")};
        World  world{engine};
        world.init(scene);

        auto* reporter = HalfPlaneAnimatorTestProbe::force_reporter;
        REQUIRE(reporter != nullptr);
        CHECK(reporter->init_count == 1);
        CHECK(reporter->vertex_indices().size() == 0);
        CHECK(reporter->forces().size() == 0);

        // the lifecycle still runs, on an empty set
        event_log().clear();
        world.advance();
        world.sync();
        CHECK(reporter->clear_count == 1);
        CHECK(reporter->step_count == 1);
        CHECK(reporter->forces().size() == 0);
        CHECK(read_device(reporter->forces()).empty());

        // no Constraint in this scene -> no animator in the frame
        CHECK(HalfPlaneAnimatorTestProbe::global_animator == nullptr);
        CHECK(!HalfPlaneAnimatorTestAnimator::built);
        std::vector<std::string> expected{"vertex_attributes",
                                          "force_clear",
                                          "force_step",
                                          "integrator_predict_dof",
                                          "integrator_update_state"};
        CHECK(event_log().events == expected);
        CHECK(frame_event_order_ok(false));
    }
}

// ===========================================================================
//  5) joint dof
// ===========================================================================
TEST_CASE("half_plane_joint_dof", "[luisa][half_plane]")
{
    reset_test_systems();

    SceneDesc desc;
    desc.dt = 0.25;
    desc.half_planes.push_back(HalfPlaneDesc{.P = {0.0, -1.0, 0.0}});
    // three joints: one, two and zero dofs (the last one is the empty dof set)
    desc.joints = {JointDesc{.dof_count = 1, .angle = 0.5},
                   JointDesc{.dof_count = 2, .angle = 1.5},
                   JointDesc{.dof_count = 0, .angle = -2.5}};

    auto   scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_joint_test")};
    World  world{engine};
    world.init(scene);

    auto* manager = HalfPlaneAnimatorTestProbe::joint_dof_manager;
    REQUIRE(manager != nullptr);                          // the manager resolved
    CHECK(HalfPlaneAnimatorTestJointDofReporter::built);  // ... and its reporter

    auto* reporter = HalfPlaneAnimatorTestProbe::joint_dof_reporter;
    REQUIRE(reporter != nullptr);
    // The reference only calls update_dof_attributes() from the state-restore
    // path, never from the frame loop, so nothing has been updated yet.
    CHECK(HalfPlaneAnimatorTestJointDofReporter::update_count == 0);
    CHECK(reporter->dof_offsets_per_joint().empty());

    uipc::backend::WorldVisitor visitor{world};

    // ---- first sync: the dof sets/indices and the persistent values --------
    manager->update_dof_attributes();
    CHECK(HalfPlaneAnimatorTestJointDofReporter::update_count == 1);
    // hand-computed: dof sets [0, 1), [1, 3), [3, 3)
    CHECK(reporter->dof_offsets_per_joint() == std::vector<IndexT>{0, 1, 3});
    CHECK(reporter->dof_counts_per_joint() == std::vector<IndexT>{1, 2, 0});
    // persistent per-dof state: dof k of a joint holds angle + k
    CHECK(read_device(reporter->dof_values()) == std::vector<Float>{0.5, 1.5, 2.5});

    // ... and the dof sets are published through the reference's
    // dof_offset/dof_count geometry-meta convention
    {
        std::vector<SimplicialComplex*> joints;
        for(auto& slot : visitor.scene().geometries())
        {
            auto sc = slot->geometry().as<SimplicialComplex>();
            if(sc && sc->meta().find<IndexT>(test_attr::joint_count))
            {
                joints.push_back(sc);
            }
        }
        REQUIRE(joints.size() == 3);  // the three joint geometries, in scene order
        const std::vector<IndexT> expect_offset{0, 1, 3};
        const std::vector<IndexT> expect_count{1, 2, 0};
        for(size_t i = 0; i < joints.size(); ++i)
        {
            auto d_off = joints[i]->meta().find<IndexT>(builtin::dof_offset);
            auto d_cnt = joints[i]->meta().find<IndexT>(builtin::dof_count);
            REQUIRE(d_off != nullptr);
            REQUIRE(d_cnt != nullptr);
            CHECK(view(*d_off)[0] == expect_offset[i]);
            CHECK(view(*d_cnt)[0] == expect_count[i]);
        }
        // the joint geometries are reported as vertices too (one point each,
        // plus the half plane's own vertex)
        CHECK(HalfPlaneAnimatorTestProbe::test_vertex_reporter->vertex_count() == 3);
        CHECK(HalfPlaneAnimatorTestProbe::half_plane_vertex_reporter->vertex_count() == 1);
    }

    // ---- a layout change: re-sync the persistent state ---------------------
    {
        const std::vector<Float> new_angles{9.5, -1.25, 7.0};
        size_t                   i = 0;
        for(auto& slot : visitor.scene().geometries())
        {
            auto sc = slot->geometry().as<SimplicialComplex>();
            if(!sc)
            {
                continue;
            }
            auto angle = sc->instances().find<Float>(test_attr::joint_angle);
            if(!angle)
            {
                continue;
            }
            view(*angle)[0] = new_angles[i++];
        }
        CHECK(i == 3);
    }
    manager->update_dof_attributes();
    CHECK(HalfPlaneAnimatorTestJointDofReporter::update_count == 2);
    // hand-computed: {9.5}, {-1.25, -1.25 + 1}, {}
    CHECK(read_device(reporter->dof_values()) == std::vector<Float>{9.5, -1.25, -0.25});
    CHECK(reporter->dof_offsets_per_joint() == std::vector<IndexT>{0, 1, 3});
    CHECK(reporter->dof_counts_per_joint() == std::vector<IndexT>{1, 2, 0});

    // ---- the frame loop still runs with joints present ---------------------
    event_log().clear();
    world.advance();
    world.sync();
    CHECK(world.frame() == 1);
    // the manager is not part of the frame loop
    CHECK(HalfPlaneAnimatorTestJointDofReporter::update_count == 2);
    CHECK(frame_event_order_ok(false));
}

// ===========================================================================
//  6) edge cases
// ===========================================================================
TEST_CASE("half_plane_animator_edge_cases", "[luisa][half_plane]")
{
    // (a) 0 half planes + a force on an empty set + the mirror flag config
    {
        reset_test_systems();

        SceneDesc desc;
        desc.integrator_type      = "bdf2";
        desc.contact_constitution = "al-ipc";
        desc.dt                   = 0.25;
        MeshDesc mesh;
        mesh.positions     = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        mesh.triangles     = {{0, 1, 2}};
        mesh.force_enabled = {0, 0, 0};
        desc.meshes.push_back(mesh);

        auto   scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_edge_test")};
        World  world{engine};
        world.init(scene);

        // ---- 0 half planes -------------------------------------------------
        auto& hp = *HalfPlaneAnimatorTestProbe::half_plane;
        CHECK(hp.positions().size() == 0);
        CHECK(hp.normals().size() == 0);
        CHECK(hp.geo_infos().empty());
        // the half plane reporters contribute nothing
        CHECK(HalfPlaneAnimatorTestProbe::half_plane_vertex_reporter->vertex_count() == 0);
        CHECK(HalfPlaneAnimatorTestProbe::half_plane_body_reporter->body_count() == 0);

        auto& vm = *HalfPlaneAnimatorTestProbe::vertex_manager;
        auto& sm = *HalfPlaneAnimatorTestProbe::surface_manager;
        auto* my_vr = HalfPlaneAnimatorTestProbe::test_vertex_reporter;
        REQUIRE(my_vr != nullptr);
        CHECK(my_vr->vertex_offset() == 0);  // nothing precedes it
        CHECK(my_vr->vertex_count() == 3);
        CHECK(slice(read_device(vm.coindices()), 0, 3) == std::vector<IndexT>{0, 1, 2});
        CHECK(slice(read_device(vm.body_ids()), 0, 3) == std::vector<IndexT>{-1, -1, -1});
        CHECK(contains_subsequence(read_device(sm.surf_vertices()),
                                   std::vector<IndexT>{0, 1, 2}));

        // ---- flags at frame 0 ---------------------------------------------
        auto expected = expected_flags(desc);
        check_probe_flags(expected);
        CHECK(HalfPlaneAnimatorTestProbe::sim_engine->pipeline_type()
              == gg::SimEngine::PipelineType::AugmentedLagrangian);
        CHECK(!HalfPlaneAnimatorTestBDF1Integrator::built);
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::built);
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::init_count == 1);
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::predict_dt.empty());
        CHECK(HalfPlaneAnimatorTestProbe::global_animator == nullptr);  // no constraint

        // ---- a force on an empty set --------------------------------------
        auto* force_reporter = HalfPlaneAnimatorTestProbe::force_reporter;
        REQUIRE(force_reporter != nullptr);
        CHECK(force_reporter->vertex_indices().size() == 0);
        CHECK(force_reporter->forces().size() == 0);

        event_log().clear();
        world.advance();
        world.sync();
        CHECK(force_reporter->clear_count == 1);
        CHECK(force_reporter->step_count == 1);
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::predict_dt == std::vector<Float>{0.25});
        CHECK(HalfPlaneAnimatorTestBDF2Integrator::update_dt == std::vector<Float>{0.25});
        std::vector<std::string> frame_events{"vertex_attributes",
                                              "force_clear",
                                              "force_step",
                                              "integrator_predict_dof",
                                              "integrator_update_state"};
        CHECK(event_log().events == frame_events);
    }

    // (b) an animator with a single keyframe + an animation with 0 geometry slots
    {
        reset_test_systems();

        const std::vector<Vector3> P_init = {{0.5, 0.5, 0.5}, {1.5, 0.5, 0.5}, {0.5, 1.5, 0.5}};
        const std::vector<Vector3> K      = {{5.0, 0.0, 0.0}, {5.0, 1.0, 0.0}, {5.0, 0.0, 1.0}};

        SceneDesc desc;
        desc.dt = 0.25;
        desc.half_planes.push_back(HalfPlaneDesc{.P = {0.0, -1.0, 0.0}});
        MeshDesc mesh;
        mesh.positions       = P_init;
        mesh.triangles       = {{0, 1, 2}};
        mesh.with_constraint = true;
        desc.meshes.push_back(mesh);
        desc.empty_objects = 1;  // an object with no geometry at all

        std::vector<S<Object>> mesh_objects;
        std::vector<S<Object>> empty_objects;
        auto                   scene = build_scene(desc, &mesh_objects, &empty_objects);
        REQUIRE(mesh_objects.size() == 1);
        REQUIRE(empty_objects.size() == 1);

        int  single_calls = 0;
        int  empty_calls  = 0;
        auto single       = [&, K](Animation::UpdateInfo& info)
        {
            ++single_calls;
            event_log().push("frontend_animation");
            auto  sc  = info.geo_slots()[0]->geometry().as<SimplicialComplex>();
            auto& pos = sc->positions();
            const Float t = info.dt() * static_cast<Float>(info.frame());
            for(SizeT v = 0; v < K.size(); ++v)
            {
                // one keyframe: the law holds it for every t (src == dst), and
                // the clamp keeps a past-the-end t from extrapolating
                auto r       = gg::animator::lerp(gg::to_luisa(K[v]), gg::to_luisa(K[v]), t);
                view(pos)[v] = gg::to_eigen(r);
            }
        };
        // an animation whose object has no geometry: the callback is still driven,
        // but it has no vertices to animate
        auto empty = [&](Animation::UpdateInfo& info)
        {
            ++empty_calls;
            event_log().push("frontend_animation");
            CHECK(info.geo_slots().empty());
        };
        scene.animator().insert(*mesh_objects[0], single);
        scene.animator().insert(*empty_objects[0], empty);

        Engine engine{"luisa", workspace_path("uipc_luisa_half_plane_edge_test")};
        World  world{engine};
        world.init(scene);

        auto& vm    = *HalfPlaneAnimatorTestProbe::vertex_manager;
        auto* my_vr = HalfPlaneAnimatorTestProbe::test_vertex_reporter;
        REQUIRE(my_vr != nullptr);
        const SizeT my_offset = static_cast<SizeT>(my_vr->vertex_offset());

        uipc::backend::WorldVisitor visitor{world};

        CHECK(HalfPlaneAnimatorTestAnimator::built);
        REQUIRE(HalfPlaneAnimatorTestProbe::global_animator != nullptr);
        CHECK(single_calls == 0);
        CHECK(empty_calls == 0);

        for(SizeT k = 1; k <= 3; ++k)
        {
            auto model_before = compute_vertex_model(visitor);
            event_log().clear();
            world.advance();
            world.sync();

            CHECK(single_calls == static_cast<int>(k));
            CHECK(empty_calls == static_cast<int>(k));

            // the device holds what the previous frame reported: at frame 1 the
            // initial scene state, afterwards the single keyframe
            CHECK(eigen_equal(slice(read_vector3(vm.positions()), my_offset, 3),
                              model_before.complex_positions));

            auto model_after = compute_vertex_model(visitor);
            CHECK(eigen_equal(model_after.complex_positions, K));

            // 2 animations -> 2 frontend_animation events per frame
            CHECK(event_log().count_of("frontend_animation") == 2);
            CHECK(frame_event_order_ok(true));
        }
    }
}
