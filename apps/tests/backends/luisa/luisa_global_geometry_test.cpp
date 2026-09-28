/********************************************************************
 * @file luisa_global_geometry_test.cpp
 * @brief T4.1 — global geometry managers (LuisaCompute backend).
 *
 * Drives the ported systems through the public uipc::Engine/World API, exactly
 * like backend_shell_test.cpp, and reaches into the backend only through public
 * interfaces:
 *   - three test-local reporter systems (VertexReporter / BodyReporter /
 *     SimplicialSurfaceReporter subclasses) that report the scene's
 *     SimplicialComplex geometries, one geometry = one body — the same contract
 *     the FEM/ABD reporters use;
 *   - a test-local probe SimSystem that caches the manager pointers
 *     (`require<GlobalVertexManager>()` etc.) so the test can read the manager
 *     buffers;
 *   - the managers' public accessors (`positions()`, `surf_triangles()`, ...) and
 *     `GlobalVertexManager::Impl`'s public device methods, which are exercised
 *     directly on synthetic input against a host model computed here.
 *
 * Every expectation is an independently computed value (a host model, or a value
 * this test fed in), never a number taken from the CUDA backend.
 *
 * Coverage map (brief §3):
 *   1) scene layout + hand-computable values ....... global_geometry_scene_layout
 *   2) advance/sync keeps the state ................ global_geometry_advance_keeps_state
 *   3) dump/recover byte-identical ................. global_geometry_dump_recover
 *   4) offsets/ranges on changed scenes ............ global_geometry_offsets_on_changed_scene
 *   5) adversarial inputs .......................... global_geometry_edge_cases
 *   6) kernels vs a host model (synthetic) ......... global_geometry_manager_kernels
 *   7) kernel edge cases (0 / 1 elements) .......... global_geometry_kernels_edge_cases
 *********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>

#include <type_define.h>
#include <frame_stream.h>
#include <global_geometry/body_reporter.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/simplicial_surface_reporter.h>
#include <global_geometry/vertex_reporter.h>
#include <utils/offset_count_collection.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;

namespace gg  = uipc::backend::luisa;
namespace lud = uipc::backend::luisa::luisa_device;

// ---------------------------------------------------------------------------
//  Attribute names this test feeds through the reporters
// ---------------------------------------------------------------------------
namespace test_attr
{
// per-vertex attributes created by this test on every geometry
constexpr std::string_view dimension    = "test_dimension";            // IndexT
constexpr std::string_view contact_id   = "test_contact_element_id";   // IndexT
constexpr std::string_view subscene_id  = "test_subscene_element_id";  // IndexT
constexpr std::string_view d_hat        = "test_d_hat";                // Float
constexpr std::string_view thickness    = "test_thickness";            // Float
constexpr std::string_view displacement = "test_displacement";         // Vector3
// per-geometry (meta) attribute
constexpr std::string_view self_collision = "test_self_collision";     // IndexT
}// namespace test_attr

namespace
{
// ---------------------------------------------------------------------------
//  Comparison / read-back helpers
// ---------------------------------------------------------------------------
/// Eigen's `==` is coefficient-wise, so it is not a bool. Two type parameters so
/// lazy Eigen expressions (e.g. `a + alpha * b`) can be compared as well.
template <typename A, typename B>
bool element_equal(const A& a, const B& b)
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

template <typename T>
bool arrays_equal(const std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
    {
        return false;
    }
    for(size_t i = 0; i < a.size(); ++i)
    {
        if(!element_equal(a[i], b[i]))
        {
            return false;
        }
    }
    return true;
}

/// Byte-identical comparison (the dump/recover requirement).
template <typename T>
bool byte_equal(const std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
    {
        return false;
    }
    return a.empty() || (std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

/// Flush whatever is recorded (nothing in T4.1 opens a frame, but be exact),
/// copy the slice to the host and wait.
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

template <typename T>
std::vector<Vector3> to_eigen_vector3(const std::vector<T>& raw)
{
    std::vector<Vector3> host;
    host.reserve(raw.size());
    for(auto& v : raw)
    {
        host.push_back(gg::to_eigen(v));
    }
    return host;
}

std::vector<Vector3> read_vector3(const lud::LCBufferView<gg::LVector3>& view)
{
    return to_eigen_vector3(read_device(view));
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

/// The scene's simplicial complexes, in scene order. Every reporter in this file
/// agrees on this order, which is what makes "one geometry = one body" work.
std::vector<SimplicialComplex*> scene_complexes(uipc::backend::WorldVisitor& world)
{
    std::vector<SimplicialComplex*> result;
    for(auto& slot : world.scene().geometries())
    {
        auto sc = slot->geometry().as<SimplicialComplex>();
        if(sc)
        {
            result.push_back(sc);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
//  Test scene description: one geometry = one body
// ---------------------------------------------------------------------------
struct BodyDesc
{
    std::vector<Vector3>  positions;
    std::vector<Vector3i> triangles;
    std::vector<IndexT>   dimensions;
    std::vector<Float>    thicknesses;
    std::vector<Float>    d_hats;
    std::vector<IndexT>   contact_ids;
    std::vector<IndexT>   subscene_ids;
    std::vector<Vector3>  velocities;      // extra frontend attribute (see the report)
    std::vector<Vector3>  displacements;   // reported through report_displacements()
    std::vector<IndexT>   is_surf_vertices;
    std::vector<IndexT>   is_surf_edges;
    std::vector<IndexT>   is_surf_triangles;
    IndexT                self_collision = 0;
};

/// 2 geometries (bodies), 8 vertices. Vertex ranges: body 0 owns [0,3), body 1
/// owns [3,8) — uneven ranges, one starts at 0 and the last ends at the last
/// vertex.
std::vector<BodyDesc> two_body_scene()
{
    std::vector<BodyDesc> bodies(2);

    auto& a = bodies[0];
    a.positions    = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    a.triangles    = {{0, 1, 2}};
    a.dimensions   = {2, 2, 0};
    a.thicknesses  = {0.1, 0.2, 0.3};
    a.d_hats       = {0.01, 0.02, 0.03};
    a.contact_ids  = {10, 11, 12};
    a.subscene_ids = {20, 21, 22};
    a.velocities   = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    a.displacements = {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}};
    a.is_surf_vertices  = {1, 1, 1};
    a.is_surf_triangles = {1};
    a.self_collision    = 1;

    auto& b = bodies[1];
    b.positions    = {{2.0, 0.0, 0.0},
                      {3.0, 0.0, 0.0},
                      {4.0, 0.0, 0.0},
                      {3.0, 1.0, 0.0},
                      {3.0, 0.0, 1.0}};
    b.triangles    = {{0, 1, 3}, {1, 2, 4}, {3, 4, 0}};
    b.dimensions   = {3, 3, 3, 3, 3};
    b.thicknesses  = {0.5, 0.6, 0.7, 0.8, 0.9};
    b.d_hats       = {0.04, 0.05, 0.06, 0.07, 0.08};
    b.contact_ids  = {30, 31, 32, 33, 34};
    b.subscene_ids = {40, 41, 42, 43, 44};
    b.velocities   = {{-1.0, 0.0, 0.0},
                      {-2.0, 0.0, 0.0},
                      {-3.0, 0.0, 0.0},
                      {-4.0, 0.0, 0.0},
                      {-5.0, 0.0, 0.0}};
    b.displacements = {{0.25, 0.0, 0.0},
                       {0.5, 0.0, 0.0},
                       {0.75, 0.0, 0.0},
                       {1.0, 0.0, 0.0},
                       {1.25, 0.0, 0.0}};
    b.is_surf_vertices  = {1, 1, 0, 1, 1};  // local vertex 2 is not on the surface
    b.is_surf_triangles = {1, 0, 1};        // local triangle 1 is not on the surface
    b.self_collision    = 0;

    return bodies;
}

/// Same layout as two_body_scene() but with shifted positions: proves that
/// recover() really overwrites the freshly reported state.
std::vector<BodyDesc> two_body_scene_shifted(Float dx, Float dy)
{
    auto bodies = two_body_scene();
    for(auto& body : bodies)
    {
        for(auto& p : body.positions)
        {
            p += Vector3{dx, dy, 0.0};
        }
    }
    return bodies;
}

/// One empty geometry (a body with 0 vertices) + one normal geometry.
std::vector<BodyDesc> zero_vertex_body_scene()
{
    std::vector<BodyDesc> bodies(2);

    // bodies[0]: 0 vertices, 0 triangles, nothing else

    auto& b = bodies[1];
    b.positions     = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    b.triangles     = {{0, 1, 2}, {0, 2, 3}};
    b.dimensions    = {2, 2, 2, 2};
    b.thicknesses   = {0.01, 0.02, 0.03, 0.04};
    b.d_hats        = {0.1, 0.1, 0.1, 0.1};
    b.contact_ids   = {1, 1, 1, 1};
    b.subscene_ids  = {2, 2, 2, 2};
    b.velocities    = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    b.displacements = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    b.is_surf_vertices  = {1, 1, 1, 1};
    b.is_surf_triangles = {1, 1};
    b.self_collision    = 0;

    return bodies;
}

/// Build a scene: one object per body, one simplicial complex per object, with
/// the per-vertex attributes the test reporters read.
Scene build_scene(const std::vector<BodyDesc>& bodies, Float d_hat_relative = 0.0)
{
    auto config                      = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    if(d_hat_relative > 0.0)
    {
        // the scene-relative d_hat override the vertex manager propagates into the
        // per-vertex d_hats (Stiff-GIPC convention, see GlobalVertexManager::init())
        config["contact"]["d_hat_relative"] = d_hat_relative;
    }
    Scene scene{config};

    for(size_t i = 0; i < bodies.size(); ++i)
    {
        const auto& body = bodies[i];

        // SimplicialComplex has no assignment operator, so build it in place.
        SimplicialComplex sc = [&]() -> SimplicialComplex
        {
            if(body.positions.empty())
            {
                // a simplicial complex with no vertices at all
                return SimplicialComplex{};
            }
            if(body.triangles.empty())
            {
                auto complex = pointcloud(span<const Vector3>{body.positions});
                label_surface(complex);
                return complex;
            }
            auto complex = trimesh(span<const Vector3>{body.positions},
                                   span<const Vector3i>{body.triangles});
            label_surface(complex);
            return complex;
        }();

        auto vertices = sc.vertices();
        if(!body.positions.empty())
        {
            // per-vertex attributes fed by the test
            auto dimension    = vertices.create<IndexT>(test_attr::dimension, 0);
            auto contact      = vertices.create<IndexT>(test_attr::contact_id, 0);
            auto subscene     = vertices.create<IndexT>(test_attr::subscene_id, 0);
            auto d_hat        = vertices.create<Float>(test_attr::d_hat, 0.0);
            auto thickness    = vertices.create<Float>(test_attr::thickness, 0.0);
            auto displacement = vertices.create<Vector3>(test_attr::displacement, Vector3::Zero());
            auto velocity     = vertices.create<Vector3>(builtin::velocity, Vector3::Zero());

            std::ranges::copy(body.dimensions, view(*dimension).begin());
            std::ranges::copy(body.contact_ids, view(*contact).begin());
            std::ranges::copy(body.subscene_ids, view(*subscene).begin());
            std::ranges::copy(body.d_hats, view(*d_hat).begin());
            std::ranges::copy(body.thicknesses, view(*thickness).begin());
            std::ranges::copy(body.displacements, view(*displacement).begin());
            std::ranges::copy(body.velocities, view(*velocity).begin());

            // surface flags (label_surface() sets them all to 1 for a <=2D complex)
            auto v_is_surf = vertices.find<IndexT>(builtin::is_surf);
            if(v_is_surf && body.is_surf_vertices.size() == view(*v_is_surf).size())
            {
                std::ranges::copy(body.is_surf_vertices, view(*v_is_surf).begin());
            }
            if(!body.is_surf_edges.empty())
            {
                auto e_is_surf = sc.edges().find<IndexT>(builtin::is_surf);
                if(e_is_surf && body.is_surf_edges.size() == view(*e_is_surf).size())
                {
                    std::ranges::copy(body.is_surf_edges, view(*e_is_surf).begin());
                }
            }
            if(!body.is_surf_triangles.empty())
            {
                auto f_is_surf = sc.triangles().find<IndexT>(builtin::is_surf);
                if(f_is_surf && body.is_surf_triangles.size() == view(*f_is_surf).size())
                {
                    std::ranges::copy(body.is_surf_triangles, view(*f_is_surf).begin());
                }
            }
        }

        // per-geometry (meta) attribute read by the body reporter
        sc.meta().create<IndexT>(test_attr::self_collision, body.self_collision);

        auto object = scene.objects().create("body_" + std::to_string(i));
        object->geometries().create(sc);
    }

    return scene;
}

// ---------------------------------------------------------------------------
//  Host model: the expected manager contents, computed from the scene alone
// ---------------------------------------------------------------------------
struct HostModel
{
    std::vector<Vector3>  positions;
    std::vector<IndexT>   coindices;
    std::vector<IndexT>   body_ids;
    std::vector<IndexT>   dimensions;
    std::vector<Float>    thicknesses;
    std::vector<Float>    d_hats;
    std::vector<IndexT>   contact_ids;
    std::vector<IndexT>   subscene_ids;
    std::vector<Vector3>  displacements;
    std::vector<Vector3>  velocities;

    std::vector<IndexT> body_coindices;
    std::vector<IndexT> body_self_collision;

    std::vector<IndexT>   surf_vertices;
    std::vector<Vector2i> surf_edges;
    std::vector<Vector3i> surf_triangles;
    std::vector<IndexT>   codim_vertices;

    gg::AABB bounding_box;
};

HostModel compute_host_model(uipc::backend::WorldVisitor& world)
{
    HostModel model;

    const Float max_float = std::numeric_limits<Float>::max();
    Vector3     min_pos{max_float, max_float, max_float};
    Vector3     max_pos{-max_float, -max_float, -max_float};

    IndexT index         = 0;
    IndexT vertex_offset = 0;
    for(auto& sc : scene_complexes(world))
    {
        model.body_coindices.push_back(index);
        auto self_collision = sc->meta().find<IndexT>(test_attr::self_collision);
        model.body_self_collision.push_back(self_collision ? view(*self_collision)[0] : 0);

        auto positions = sc->positions().view();
        auto dimension = sc->vertices().find<IndexT>(test_attr::dimension);
        auto contact   = sc->vertices().find<IndexT>(test_attr::contact_id);
        auto subscene  = sc->vertices().find<IndexT>(test_attr::subscene_id);
        auto d_hat     = sc->vertices().find<Float>(test_attr::d_hat);
        auto thickness = sc->vertices().find<Float>(test_attr::thickness);
        auto displace  = sc->vertices().find<Vector3>(test_attr::displacement);
        auto velocity  = sc->vertices().find<Vector3>(builtin::velocity);

        for(size_t v = 0; v < positions.size(); ++v)
        {
            model.coindices.push_back(static_cast<IndexT>(v));
            model.body_ids.push_back(index);
            model.positions.push_back(positions[v]);
            model.dimensions.push_back(dimension ? view(*dimension)[v] : 3);
            model.contact_ids.push_back(contact ? view(*contact)[v] : 0);
            model.subscene_ids.push_back(subscene ? view(*subscene)[v] : 0);
            model.d_hats.push_back(d_hat ? view(*d_hat)[v] : 0.0);
            model.thicknesses.push_back(thickness ? view(*thickness)[v] : 0.0);
            model.displacements.push_back(displace ? view(*displace)[v] : Vector3::Zero());
            model.velocities.push_back(velocity ? view(*velocity)[v] : Vector3::Zero());

            min_pos = min_pos.cwiseMin(positions[v]);
            max_pos = max_pos.cwiseMax(positions[v]);

            if(model.dimensions.back() <= 1)
            {
                model.codim_vertices.push_back(vertex_offset + static_cast<IndexT>(v));
            }
        }

        // surface elements in element order, with global vertex indices
        auto v_is_surf = sc->vertices().find<IndexT>(builtin::is_surf);
        if(v_is_surf)
        {
            for(size_t v = 0; v < view(*v_is_surf).size(); ++v)
            {
                if(view(*v_is_surf)[v])
                {
                    model.surf_vertices.push_back(vertex_offset + static_cast<IndexT>(v));
                }
            }
        }
        auto e_is_surf = sc->edges().find<IndexT>(builtin::is_surf);
        if(e_is_surf)
        {
            auto edges = sc->edges().topo().view();
            for(size_t e = 0; e < view(*e_is_surf).size(); ++e)
            {
                if(view(*e_is_surf)[e])
                {
                    model.surf_edges.push_back(edges[e].array() + vertex_offset);
                }
            }
        }
        auto f_is_surf = sc->triangles().find<IndexT>(builtin::is_surf);
        if(f_is_surf)
        {
            auto triangles = sc->triangles().topo().view();
            for(size_t f = 0; f < view(*f_is_surf).size(); ++f)
            {
                if(view(*f_is_surf)[f])
                {
                    model.surf_triangles.push_back(triangles[f].array() + vertex_offset);
                }
            }
        }

        vertex_offset += static_cast<IndexT>(positions.size());
        ++index;
    }

    // the reference's CUB reduce identities (cwiseMin/cwiseMax over [0, n) with
    // the init value used when the range is empty)
    model.bounding_box = gg::AABB{min_pos.cast<float>(), max_pos.cast<float>()};
    return model;
}
}// namespace

// ---------------------------------------------------------------------------
//  Test reporters (registered systems, like the FEM/ABD reporters)
// ---------------------------------------------------------------------------
class GlobalGeometryTestBodyReporter final : public gg::BodyReporter
{
  public:
    using BodyReporter::BodyReporter;

  protected:
    void do_init(InitInfo& info) override { (void)info; }
    void do_build(BuildInfo& info) override { (void)info; }

    void do_report_count(BodyCountInfo& info) override
    {
        info.count(scene_complexes(world()).size());
    }

    void do_report_attributes(BodyAttributeInfo& info) override
    {
        std::vector<IndexT> coindices;
        std::vector<IndexT> self_collision;

        IndexT index = 0;
        for(auto& sc : scene_complexes(world()))
        {
            coindices.push_back(index);
            auto flag = sc->meta().find<IndexT>(test_attr::self_collision);
            self_collision.push_back(flag ? view(*flag)[0] : 0);
            ++index;
        }

        info.coindices().copy_from(span<const IndexT>{coindices});
        info.self_collision().copy_from(span<const IndexT>{self_collision});
    }
};
REGISTER_SIM_SYSTEM(GlobalGeometryTestBodyReporter);

class GlobalGeometryTestVertexReporter final : public gg::VertexReporter
{
  public:
    using VertexReporter::VertexReporter;

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        m_body_reporter = &require<GlobalGeometryTestBodyReporter>();
    }

    void do_report_count(VertexCountInfo& info) override
    {
        SizeT count = 0;
        for(auto& sc : scene_complexes(world()))
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

        IndexT geometry_index = 0;
        for(auto& sc : scene_complexes(world()))
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
                // the global body index: the manager's body range plus this
                // geometry's place in the report (the FEM reporters do the same)
                body_ids.push_back(m_body_reporter->body_offset() + geometry_index);
                dimensions.push_back(dimension ? view(*dimension)[v] : 3);
                contact_ids.push_back(contact ? view(*contact)[v] : 0);
                subscene_ids.push_back(subscene ? view(*subscene)[v] : 0);
                d_hats.push_back(d_hat ? view(*d_hat)[v] : 0.0);
                thicknesses.push_back(thickness ? view(*thickness)[v] : 0.0);
            }
            ++geometry_index;
        }

        if(info.frame() == 0)
        {
            m_rest_positions = positions;
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
    }

    void do_report_displacements(VertexDisplacementInfo& info) override
    {
        std::vector<gg::LVector3> displacements;
        for(auto& sc : scene_complexes(world()))
        {
            auto displace = sc->vertices().find<Vector3>(test_attr::displacement);
            for(size_t v = 0; v < sc->vertices().size(); ++v)
            {
                displacements.push_back(displace ? gg::to_luisa(view(*displace)[v]) : gg::LVector3{});
            }
        }
        info.displacements().copy_from(span<const gg::LVector3>{displacements});
    }

    U64 get_uid() const noexcept override { return 0x5445535456455254ull; }  // "TESTVERT"

  private:
    GlobalGeometryTestBodyReporter* m_body_reporter = nullptr;
    std::vector<gg::LVector3>       m_rest_positions;
};
REGISTER_SIM_SYSTEM(GlobalGeometryTestVertexReporter);

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

class GlobalGeometryTestSurfaceReporter final : public gg::SimplicialSurfaceReporter
{
  public:
    using SimplicialSurfaceReporter::SimplicialSurfaceReporter;

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        m_vertex_reporter = &require<GlobalGeometryTestVertexReporter>();
    }

    void do_init(SurfaceInitInfo& info) override { (void)info; }

    void do_report_count(SurfaceCountInfo& info) override
    {
        SizeT v = 0, e = 0, f = 0;
        for(auto& sc : scene_complexes(world()))
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
        auto complexes = scene_complexes(world());

        // The vertex reporter reports every geometry in one slice, so a geometry's
        // global vertex offset is the running sum of the vertex counts of the
        // geometries before it, plus the reporter's own slice offset.
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

        size_t       geometry_index       = 0;
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

        info.surf_vertices().copy_from(span<const IndexT>{surf_vertices});
        info.surf_edges().copy_from(span<const gg::LVector2i>{surf_edges});
        info.surf_triangles().copy_from(span<const gg::LVector3i>{surf_triangles});
    }

  private:
    GlobalGeometryTestVertexReporter* m_vertex_reporter = nullptr;
};
REGISTER_SIM_SYSTEM(GlobalGeometryTestSurfaceReporter);

// ---------------------------------------------------------------------------
//  Probe system: exposes the manager instances to the test
// ---------------------------------------------------------------------------
class GlobalGeometryTestProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    static gg::GlobalVertexManager*            vertex_manager;
    static gg::GlobalBodyManager*              body_manager;
    static gg::GlobalSimplicialSurfaceManager* surface_manager;

  private:
    void do_build() override
    {
        vertex_manager  = &require<gg::GlobalVertexManager>();
        body_manager    = &require<gg::GlobalBodyManager>();
        surface_manager = &require<gg::GlobalSimplicialSurfaceManager>();
    }
};
REGISTER_SIM_SYSTEM(GlobalGeometryTestProbe);

gg::GlobalVertexManager*            GlobalGeometryTestProbe::vertex_manager  = nullptr;
gg::GlobalBodyManager*              GlobalGeometryTestProbe::body_manager    = nullptr;
gg::GlobalSimplicialSurfaceManager* GlobalGeometryTestProbe::surface_manager = nullptr;

// ---------------------------------------------------------------------------
//  1) scene layout: init must reproduce exactly what the test fed in
// ---------------------------------------------------------------------------
TEST_CASE("global_geometry_scene_layout", "[luisa][global_geometry]")
{
    auto bodies = two_body_scene();
    auto scene  = build_scene(bodies);

    Engine engine{"luisa", workspace_path("uipc_luisa_global_geometry_test")};
    World  world{engine};
    world.init(scene);

    uipc::backend::WorldVisitor visitor{world};
    auto model = compute_host_model(visitor);

    // sanity: the test scene really is what the assertions below assume
    REQUIRE(model.positions.size() == 8);
    REQUIRE(model.body_coindices.size() == 2);
    REQUIRE(model.surf_vertices.size() == 7);
    REQUIRE(model.codim_vertices.size() == 1);

    REQUIRE(GlobalGeometryTestProbe::vertex_manager != nullptr);
    REQUIRE(GlobalGeometryTestProbe::body_manager != nullptr);
    REQUIRE(GlobalGeometryTestProbe::surface_manager != nullptr);

    auto& vm = *GlobalGeometryTestProbe::vertex_manager;
    auto& bm = *GlobalGeometryTestProbe::body_manager;
    auto& sm = *GlobalGeometryTestProbe::surface_manager;

    // ---- vertex manager ----------------------------------------------------
    CHECK(arrays_equal(read_vector3(vm.positions()), model.positions));
    CHECK(read_device(vm.coindices()) == model.coindices);
    CHECK(read_device(vm.body_ids()) == model.body_ids);
    CHECK(read_device(vm.dimensions()) == model.dimensions);
    CHECK(read_device(vm.contact_element_ids()) == model.contact_ids);
    CHECK(read_device(vm.subscene_element_ids()) == model.subscene_ids);
    CHECK(read_device(vm.d_hats()) == model.d_hats);
    CHECK(read_device(vm.thicknesses()) == model.thicknesses);

    // rest / prev / safe positions are all the frame-0 state; displacements stay
    // zero until the frame loop reports them (the reference calls
    // collect_vertex_displacements() in the update phase, not in init()).
    CHECK(arrays_equal(read_vector3(vm.rest_positions()), model.positions));
    CHECK(arrays_equal(read_vector3(vm.prev_positions()), model.positions));
    CHECK(arrays_equal(read_vector3(vm.safe_positions()), model.positions));
    {
        auto displacements = read_vector3(vm.displacements());
        REQUIRE(displacements.size() == model.positions.size());
        for(auto& d : displacements)
        {
            CHECK(element_equal(d, Vector3::Zero()));
        }
    }

    // the cached AABB that init() computed with the pack kernel + 6 scalar
    // reductions, against the host model (the reduction kernels themselves are
    // exercised in global_geometry_manager_kernels below)
    auto box = vm.vertex_bounding_box();
    CHECK(element_equal(box.min(), model.bounding_box.min()));
    CHECK(element_equal(box.max(), model.bounding_box.max()));

    // scene_diagonal() is the float norm of the rest-position box diagonal
    const Float expected_diagonal =
        (model.bounding_box.max() - model.bounding_box.min()).norm();
    CHECK(vm.scene_diagonal() == expected_diagonal);

    // ---- body manager ------------------------------------------------------
    CHECK(read_device(bm.coindices()) == model.body_coindices);
    CHECK(read_device(bm.self_collision()) == model.body_self_collision);

    // ---- simplicial surface manager ---------------------------------------
    CHECK(read_device(sm.surf_vertices()) == model.surf_vertices);
    CHECK(arrays_equal(read_vector2i(sm.surf_edges()), model.surf_edges));
    CHECK(arrays_equal(read_vector3i(sm.surf_triangles()), model.surf_triangles));
    CHECK(read_device(sm.codim_vertices()) == model.codim_vertices);

    // ---- the scene's own attributes survived the reporter round trip -------
    auto complexes = scene_complexes(visitor);
    REQUIRE(complexes.size() == 2);
    auto velocity = complexes[1]->vertices().find<Vector3>(builtin::velocity);
    REQUIRE(velocity != nullptr);
    REQUIRE(view(*velocity).size() == bodies[1].velocities.size());
    for(size_t v = 0; v < view(*velocity).size(); ++v)
    {
        CHECK(element_equal(view(*velocity)[v], bodies[1].velocities[v]));
    }
}

// ---------------------------------------------------------------------------
//  2) advance: the frame counter moves, the attributes stay consistent
// ---------------------------------------------------------------------------
TEST_CASE("global_geometry_advance_keeps_state", "[luisa][global_geometry]")
{
    auto bodies = two_body_scene();
    auto scene  = build_scene(bodies);

    Engine engine{"luisa", workspace_path("uipc_luisa_global_geometry_test")};
    World  world{engine};
    world.init(scene);

    auto& vm = *GlobalGeometryTestProbe::vertex_manager;

    auto positions_before = read_vector3(vm.positions());
    auto rest_before      = read_vector3(vm.rest_positions());
    auto prev_before      = read_vector3(vm.prev_positions());
    auto safe_before      = read_vector3(vm.safe_positions());

    const auto frame_before = world.frame();
    world.advance();
    world.sync();
    world.retrieve();

    CHECK(world.frame() == frame_before + 1);

    // no dynamics system is ported yet, so the geometry must be untouched
    CHECK(byte_equal(read_vector3(vm.positions()), positions_before));
    CHECK(byte_equal(read_vector3(vm.rest_positions()), rest_before));
    CHECK(byte_equal(read_vector3(vm.prev_positions()), prev_before));
    CHECK(byte_equal(read_vector3(vm.safe_positions()), safe_before));
}

// ---------------------------------------------------------------------------
//  3) dump / recover round trip: byte-identical attribute arrays
// ---------------------------------------------------------------------------
TEST_CASE("global_geometry_dump_recover", "[luisa][global_geometry]")
{
    const std::string workspace = workspace_path("uipc_luisa_global_geometry_dump_test");

    struct Snapshot
    {
        std::vector<Vector3> positions;
        std::vector<Vector3> prev_positions;
        std::vector<Vector3> rest_positions;
        std::vector<Vector3> safe_positions;
        std::vector<IndexT>  coindices;
        std::vector<IndexT>  body_ids;
        std::vector<IndexT>  dimensions;
        std::vector<Float>   d_hats;
        std::vector<Float>   thicknesses;
    } snapshot;

    // --- engine 1: dump the scene at frame 0 -------------------------------
    {
        auto   bodies = two_body_scene();
        auto   scene  = build_scene(bodies);
        Engine engine{"luisa", workspace};
        World  world{engine};
        world.init(scene);

        auto& vm = *GlobalGeometryTestProbe::vertex_manager;
        snapshot.positions      = read_vector3(vm.positions());
        snapshot.prev_positions = read_vector3(vm.prev_positions());
        snapshot.rest_positions = read_vector3(vm.rest_positions());
        snapshot.safe_positions = read_vector3(vm.safe_positions());
        snapshot.coindices      = read_device(vm.coindices());
        snapshot.body_ids       = read_device(vm.body_ids());
        snapshot.dimensions     = read_device(vm.dimensions());
        snapshot.d_hats         = read_device(vm.d_hats());
        snapshot.thicknesses    = read_device(vm.thicknesses());

        CHECK(world.dump());
    }

    // --- engine 2: same scene with *different* positions, then recover ------
    {
        auto   bodies = two_body_scene_shifted(10.0, -3.0);
        auto   scene  = build_scene(bodies);
        Engine engine{"luisa", workspace};
        World  world{engine};
        world.init(scene);

        auto& vm = *GlobalGeometryTestProbe::vertex_manager;

        // the fresh initialization really took the different scene
        auto shifted = read_vector3(vm.positions());
        REQUIRE(shifted.size() == snapshot.positions.size());
        bool any_difference = false;
        for(size_t i = 0; i < shifted.size(); ++i)
        {
            if(!element_equal(shifted[i], snapshot.positions[i]))
            {
                any_difference = true;
            }
        }
        CHECK(any_difference);

        CHECK(world.recover());

        // The reference dumps `positions` and `prev_positions` only (see
        // GlobalVertexManager::Impl::dump), so `rest_positions` (reporter-owned)
        // and `safe_positions` (written by init from `positions`) keep the fresh
        // initialization's values: assert that explicitly rather than expecting a
        // recovery that the reference does not perform.
        CHECK(byte_equal(read_vector3(vm.positions()), snapshot.positions));
        CHECK(byte_equal(read_vector3(vm.prev_positions()), snapshot.prev_positions));
        CHECK(byte_equal(read_device(vm.coindices()), snapshot.coindices));
        CHECK(byte_equal(read_device(vm.body_ids()), snapshot.body_ids));
        CHECK(byte_equal(read_device(vm.dimensions()), snapshot.dimensions));
        CHECK(byte_equal(read_device(vm.d_hats()), snapshot.d_hats));
        CHECK(byte_equal(read_device(vm.thicknesses()), snapshot.thicknesses));
        CHECK(arrays_equal(read_vector3(vm.rest_positions()), shifted));
        CHECK(arrays_equal(read_vector3(vm.safe_positions()), shifted));

        // ... and the same data seen per vertex / per body
        CHECK(arrays_equal(read_vector3(vm.positions()), snapshot.positions));
        auto& bm = *GlobalGeometryTestProbe::body_manager;
        CHECK(read_device(bm.coindices()) == std::vector<IndexT>{0, 1});
    }

    // the dump file itself: 8-byte magic, 8-byte payload size, raw element bytes
    // (byte-exact against the pre-dump positions, so the file format and the
    // LVector3/Vector3 element layout are verified independently of the loader)
    {
        auto path = std::filesystem::path{workspace} / "dump" / "luisa" / "global_geometry"
                    / "global_vertex_manager.cpp" / "positions.0";
        std::ifstream ifs(path, std::ios::binary);
        REQUIRE(ifs.is_open());

        std::size_t magic      = 0;
        std::size_t size_bytes = 0;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char*>(&size_bytes), sizeof(size_bytes));

        CHECK(magic == 0xc2663291fdf3ull);
        CHECK(size_bytes == snapshot.positions.size() * sizeof(Vector3));

        std::vector<char> payload(size_bytes);
        ifs.read(payload.data(), static_cast<std::streamsize>(size_bytes));
        CHECK(std::memcmp(payload.data(), snapshot.positions.data(), size_bytes) == 0);
    }
}

// ---------------------------------------------------------------------------
//  4) offsets/ranges on changed scenes (the OffsetCountCollection::scan() path)
// ---------------------------------------------------------------------------
TEST_CASE("global_geometry_offsets_on_changed_scene", "[luisa][global_geometry]")
{
    // (a) three bodies with uneven vertex counts
    {
        auto bodies = two_body_scene();

        BodyDesc third;
        third.positions        = {{9.0, 9.0, 9.0}};
        third.dimensions       = {0};
        third.thicknesses      = {0.9};
        third.d_hats           = {0.09};
        third.contact_ids      = {99};
        third.subscene_ids     = {99};
        third.velocities       = {{0.0, 0.0, 0.0}};
        third.displacements    = {{0.0, 0.0, 0.0}};
        third.is_surf_vertices = {1};
        third.self_collision   = 1;
        bodies.push_back(third);

        auto   scene = build_scene(bodies);
        Engine engine{"luisa", workspace_path("uipc_luisa_global_geometry_test")};
        World  world{engine};
        world.init(scene);

        auto& vm = *GlobalGeometryTestProbe::vertex_manager;
        auto& bm = *GlobalGeometryTestProbe::body_manager;
        auto& sm = *GlobalGeometryTestProbe::surface_manager;

        // vertex counts 3 + 5 + 1 -> contiguous ranges [0,3) [3,8) [8,9)
        std::vector<IndexT> expected_body_ids(3, 0);
        expected_body_ids.insert(expected_body_ids.end(), 5, 1);
        expected_body_ids.push_back(2);

        CHECK(vm.positions().size() == 9);
        CHECK(read_device(vm.body_ids()) == expected_body_ids);
        // coindices are the reporters' own (per-geometry local) vertex indices,
        // exactly like the reference reporters fill them
        CHECK(read_device(vm.coindices())
              == std::vector<IndexT>{0, 1, 2, 0, 1, 2, 3, 4, 0});
        CHECK(read_device(bm.coindices()) == std::vector<IndexT>{0, 1, 2});
        CHECK(read_device(bm.self_collision()) == std::vector<IndexT>{1, 0, 1});

        // the surface ranges follow the same vertex offsets
        auto surf_vertices = read_device(sm.surf_vertices());
        REQUIRE(surf_vertices.size() == 8);
        CHECK(surf_vertices[7] == 8);  // the third body's only vertex
        CHECK(surf_vertices[0] == 0);
    }

    // (b) a single body: its range starts at 0 and ends at the last vertex
    {
        auto bodies = two_body_scene();
        bodies.resize(1);

        auto   scene = build_scene(bodies);
        Engine engine{"luisa", workspace_path("uipc_luisa_global_geometry_test")};
        World  world{engine};
        world.init(scene);

        auto& vm = *GlobalGeometryTestProbe::vertex_manager;
        auto& bm = *GlobalGeometryTestProbe::body_manager;

        CHECK(vm.positions().size() == 3);
        CHECK(read_device(vm.body_ids()) == std::vector<IndexT>{0, 0, 0});
        CHECK(read_device(vm.coindices()) == std::vector<IndexT>{0, 1, 2});
        CHECK(read_device(bm.coindices()) == std::vector<IndexT>{0});
    }

    // (c) the host-only scan() path, exercised directly
    {
        gg::OffsetCountCollection<IndexT> collection;
        collection.resize(0);
        collection.scan();
        CHECK(collection.total_count() == 0);
        CHECK(collection.counts().empty());
        CHECK(collection.offsets().empty());

        collection.resize(3);
        auto counts = collection.counts();
        counts[0]   = 4;
        counts[1]   = 0;
        counts[2]   = 7;
        collection.scan();
        CHECK(collection.total_count() == 11);
        CHECK(collection.offsets()[0] == 0);
        CHECK(collection.offsets()[1] == 4);
        CHECK(collection.offsets()[2] == 4);
        CHECK(collection[0] == std::tuple<IndexT, IndexT>{0, 4});
        CHECK(collection[2] == std::tuple<IndexT, IndexT>{4, 7});

        // scan() with a non-zero init offset shifts every offset
        collection.scan(100);
        CHECK(collection.offsets()[0] == 100);
        CHECK(collection.offsets()[1] == 104);
        CHECK(collection.total_count() == 111);

        // re-scanning with different counts (the "changed scene" bookkeeping)
        counts[0] = 0;
        counts[1] = 2;
        counts[2] = 0;
        collection.scan();
        CHECK(collection.total_count() == 2);
        CHECK(collection.offsets()[1] == 0);
        CHECK(collection.offsets()[2] == 2);
    }
}

// ---------------------------------------------------------------------------
//  5) adversarial inputs
// ---------------------------------------------------------------------------
TEST_CASE("global_geometry_edge_cases", "[luisa][global_geometry]")
{
    // (a) 0 geometries: 0 vertices, 0 bodies, 0 surfaces
    {
        auto   scene = build_scene({});
        Engine engine{"luisa", workspace_path("uipc_luisa_global_geometry_edge_test")};
        World  world{engine};
        world.init(scene);

        auto& vm = *GlobalGeometryTestProbe::vertex_manager;
        auto& bm = *GlobalGeometryTestProbe::body_manager;
        auto& sm = *GlobalGeometryTestProbe::surface_manager;

        CHECK(vm.positions().size() == 0);
        CHECK(vm.coindices().size() == 0);
        CHECK(vm.body_ids().size() == 0);
        CHECK(vm.displacements().size() == 0);
        CHECK(bm.coindices().size() == 0);
        CHECK(bm.self_collision().size() == 0);
        CHECK(sm.surf_vertices().size() == 0);
        CHECK(sm.surf_edges().size() == 0);
        CHECK(sm.surf_triangles().size() == 0);
        CHECK(sm.codim_vertices().size() == 0);

        // the reference's CUB reduce over an empty range returns its init value:
        // (+max_float, ...) for the min and (-max_float, ...) for the max, which
        // is +inf / -inf once cast to the AABB's float.
        auto box = vm.vertex_bounding_box();
        CHECK(std::isinf(box.min().x()));
        CHECK(box.min().x() > 0.0f);
        CHECK(std::isinf(box.max().x()));
        CHECK(box.max().x() < 0.0f);
    }

    // (b) one body with 0 vertices + one normal body
    {
        auto   bodies = zero_vertex_body_scene();
        auto   scene  = build_scene(bodies);
        Engine engine{"luisa", workspace_path("uipc_luisa_global_geometry_edge_test")};
        World  world{engine};
        world.init(scene);

        uipc::backend::WorldVisitor visitor{world};
    auto model = compute_host_model(visitor);

        auto& vm = *GlobalGeometryTestProbe::vertex_manager;
        auto& bm = *GlobalGeometryTestProbe::body_manager;
        auto& sm = *GlobalGeometryTestProbe::surface_manager;

        REQUIRE(model.body_coindices.size() == 2);
        REQUIRE(model.positions.size() == 4);

        CHECK(read_device(bm.coindices()) == model.body_coindices);
        CHECK(arrays_equal(read_vector3(vm.positions()), model.positions));
        // body 0 owns an empty range; the body id is the global body index
        CHECK(read_device(vm.body_ids()) == std::vector<IndexT>{1, 1, 1, 1});
        CHECK(read_device(vm.coindices()) == std::vector<IndexT>{0, 1, 2, 3});

        CHECK(read_device(sm.surf_vertices()) == model.surf_vertices);
        CHECK(arrays_equal(read_vector3i(sm.surf_triangles()), model.surf_triangles));
        CHECK(sm.codim_vertices().size() == 0);
    }

    // (c) a single vertex, a single body, no surface
    {
        BodyDesc only;
        only.positions        = {{5.0, 6.0, 7.0}};
        only.dimensions       = {0};
        only.thicknesses      = {0.5};
        only.d_hats           = {0.05};
        only.contact_ids      = {7};
        only.subscene_ids     = {8};
        only.velocities       = {{0.0, 0.0, 0.0}};
        only.displacements    = {{0.0, 0.0, 0.0}};
        only.is_surf_vertices = {1};

        auto   scene = build_scene({only});
        Engine engine{"luisa", workspace_path("uipc_luisa_global_geometry_edge_test")};
        World  world{engine};
        world.init(scene);

        auto& vm = *GlobalGeometryTestProbe::vertex_manager;
        auto& sm = *GlobalGeometryTestProbe::surface_manager;

        REQUIRE(vm.positions().size() == 1);
        CHECK(element_equal(read_vector3(vm.positions())[0], Vector3{5.0, 6.0, 7.0}));
        CHECK(read_device(vm.body_ids()) == std::vector<IndexT>{0});
        CHECK(read_device(vm.dimensions()) == std::vector<IndexT>{0});
        // a point cloud is a surface whose only element is a codim vertex
        CHECK(read_device(sm.surf_vertices()) == std::vector<IndexT>{0});
        CHECK(read_device(sm.codim_vertices()) == std::vector<IndexT>{0});
        CHECK(sm.surf_edges().size() == 0);
        CHECK(sm.surf_triangles().size() == 0);

        auto box = vm.vertex_bounding_box();
        CHECK(element_equal(box.min(), Vector3{5.0, 6.0, 7.0}.cast<float>()));
        CHECK(element_equal(box.max(), Vector3{5.0, 6.0, 7.0}.cast<float>()));
    }
}

// ---------------------------------------------------------------------------
//  5b) the scene-relative d_hat propagation (contact/d_hat_relative > 0)
// ---------------------------------------------------------------------------
TEST_CASE("global_geometry_relative_d_hat", "[luisa][global_geometry]")
{
    auto bodies = two_body_scene();
    // body 0's first d_hat is exactly the absolute default (0.01), the others are
    // not: after init() only the entry still holding the default is rewritten.
    auto   scene  = build_scene(bodies, 0.5);
    Engine engine{"luisa", workspace_path("uipc_luisa_global_geometry_test")};
    World  world{engine};
    world.init(scene);

    uipc::backend::WorldVisitor visitor{world};
    auto                        model = compute_host_model(visitor);

    auto& vm = *GlobalGeometryTestProbe::vertex_manager;

    auto  default_attr = visitor.scene().config().find<Float>("contact/d_hat");
    REQUIRE(default_attr != nullptr);
    const Float default_d_hat = default_attr->view()[0];

    // the manager computes scene_diagonal as the *float* norm of the AABB diagonal
    const Float scene_diagonal =
        (model.bounding_box.max() - model.bounding_box.min()).norm();
    const Float rel_d_hat = 0.5 * scene_diagonal;

    auto d_hats = read_device(vm.d_hats());
    REQUIRE(d_hats.size() == model.d_hats.size());
    bool any_rewritten = false;
    for(size_t i = 0; i < d_hats.size(); ++i)
    {
        const Float expected = (model.d_hats[i] == default_d_hat) ? rel_d_hat : model.d_hats[i];
        if(model.d_hats[i] == default_d_hat)
        {
            any_rewritten = true;
        }
        CHECK(d_hats[i] == expected);
    }
    CHECK(any_rewritten);
}

// ---------------------------------------------------------------------------
//  6) device kernels vs a host model, on synthetic input
//     (`GlobalVertexManager::Impl` is a public nested class, so its device
//     methods can be exercised without an engine or a scene.)
// ---------------------------------------------------------------------------
TEST_CASE("global_geometry_manager_kernels", "[luisa][global_geometry]")
{
    test_runtime();  // the device layer needs the LC runtime

    const std::vector<Vector3> safe_positions = {{1.0, 2.0, 3.0},
                                                 {4.0, 5.0, 6.0},
                                                 {-1.0, -2.0, -3.0}};
    const std::vector<Vector3> displacements  = {{0.5, 0.0, -1.0},
                                                 {2.0, -1.0, 0.0},
                                                 {0.0, 0.0, 0.0}};
    const Float                alpha          = 0.25;

    auto to_device = [](const std::vector<Vector3>& src)
    {
        std::vector<gg::LVector3> dst;
        dst.reserve(src.size());
        for(auto& v : src)
        {
            dst.push_back(gg::to_luisa(v));
        }
        return dst;
    };

    gg::GlobalVertexManager::Impl impl;
    impl.positions.resize(safe_positions.size());
    impl.safe_positions.resize(safe_positions.size());
    impl.displacements.resize(displacements.size());
    impl.prev_positions.resize(safe_positions.size());
    impl.min_max_components.resize(6);

    auto safe_dev = to_device(safe_positions);
    auto disp_dev = to_device(displacements);
    auto pos_dev  = to_device(safe_positions);

    impl.safe_positions.view().copy_from(span<const gg::LVector3>{safe_dev}, &gg::frame_stream());
    impl.displacements.view().copy_from(span<const gg::LVector3>{disp_dev}, &gg::frame_stream());
    impl.positions.view().copy_from(span<const gg::LVector3>{pos_dev}, &gg::frame_stream());
    gg::frame_stream().synchronize();

    // ---- step_forward: pos = safe_pos + alpha * disp -----------------------
    impl.step_forward(alpha);
    auto stepped = read_vector3(impl.positions.cview());
    REQUIRE(stepped.size() == safe_positions.size());
    for(size_t i = 0; i < stepped.size(); ++i)
    {
        CHECK(element_equal(stepped[i], safe_positions[i] + alpha * displacements[i]));
    }

    // ---- compute_axis_max_displacement: max over all |components| ----------
    {
        Float expected = 0.0;
        for(auto& d : displacements)
        {
            expected = std::max({expected, std::abs(d.x()), std::abs(d.y()), std::abs(d.z())});
        }
        CHECK(impl.compute_axis_max_displacement() == expected);
    }

    // ---- compute_vertex_bounding_box: per-component min/max ----------------
    {
        auto    box = impl.compute_vertex_bounding_box();
        Vector3 min_pos{std::numeric_limits<Float>::max(),
                        std::numeric_limits<Float>::max(),
                        std::numeric_limits<Float>::max()};
        Vector3 max_pos{-std::numeric_limits<Float>::max(),
                        -std::numeric_limits<Float>::max(),
                        -std::numeric_limits<Float>::max()};
        for(auto& p : stepped)
        {
            min_pos = min_pos.cwiseMin(p);
            max_pos = max_pos.cwiseMax(p);
        }
        CHECK(element_equal(box.min(), min_pos.cast<float>()));
        CHECK(element_equal(box.max(), max_pos.cast<float>()));
    }

    // ---- setup_ccd / restore_ccd / overwrite_positions ---------------------
    {
        std::vector<Vector3> base     = {{0.5, 0.5, 0.5}, {1.5, 1.5, 1.5}, {2.5, 2.5, 2.5}};
        auto                 base_dev = to_device(base);

        lud::LDeviceBuffer<gg::LVector3> base_buffer;
        base_buffer.copy_from(span<const gg::LVector3>{base_dev});

        auto positions_before_ccd = read_vector3(impl.positions.cview());

        // setup_ccd(base): disp = pos - base; safe_positions = pos; pos = base
        impl.setup_ccd(base_buffer.cview());
        auto after = read_vector3(impl.positions.cview());
        auto disp  = read_vector3(impl.displacements.cview());
        auto safe  = read_vector3(impl.safe_positions.cview());
        REQUIRE(after.size() == base.size());
        for(size_t i = 0; i < after.size(); ++i)
        {
            CHECK(element_equal(after[i], base[i]));
            CHECK(element_equal(safe[i], positions_before_ccd[i]));
            CHECK(element_equal(disp[i], positions_before_ccd[i] - base[i]));
        }

        // restore_ccd(): positions = safe_positions
        impl.restore_ccd();
        CHECK(arrays_equal(read_vector3(impl.positions.cview()), positions_before_ccd));

        // overwrite_positions(src)
        impl.overwrite_positions(base_buffer.cview());
        CHECK(arrays_equal(read_vector3(impl.positions.cview()), base));
    }

    // ---- record_prev_positions / record_start_point ------------------------
    {
        auto expected = read_vector3(impl.positions.cview());
        impl.record_prev_positions();
        impl.record_start_point();
        CHECK(arrays_equal(read_vector3(impl.prev_positions.cview()), expected));
        CHECK(arrays_equal(read_vector3(impl.safe_positions.cview()), expected));
    }
}

// ---------------------------------------------------------------------------
//  7) empty / single-element edge cases of the same kernels
// ---------------------------------------------------------------------------
TEST_CASE("global_geometry_kernels_edge_cases", "[luisa][global_geometry]")
{
    test_runtime();

    gg::GlobalVertexManager::Impl impl;
    impl.positions.resize(0);
    impl.safe_positions.resize(0);
    impl.displacements.resize(0);
    impl.prev_positions.resize(0);
    impl.min_max_components.resize(6);

    // n == 0: nothing is dispatched, the AABB is the reduce identity
    impl.step_forward(1.0);
    impl.record_prev_positions();
    impl.record_start_point();
    gg::frame_stream().synchronize();
    CHECK(impl.positions.size() == 0);
    CHECK(impl.compute_axis_max_displacement() == 0.0);

    {
        auto box = impl.compute_vertex_bounding_box();
        CHECK(std::isinf(box.min().x()));
        CHECK(box.min().x() > 0.0f);
        CHECK(std::isinf(box.max().x()));
        CHECK(box.max().x() < 0.0f);
    }

    // a single vertex (the smallest non-empty input)
    std::vector<gg::LVector3> one_dev = {gg::to_luisa(Vector3{-2.0, 0.5, 4.0})};
    impl.positions.resize(1);
    impl.safe_positions.resize(1);
    impl.displacements.resize(1);
    impl.positions.view().copy_from(span<const gg::LVector3>{one_dev}, &gg::frame_stream());
    impl.safe_positions.view().copy_from(span<const gg::LVector3>{one_dev}, &gg::frame_stream());
    impl.displacements.view().fill(gg::LVector3{}, &gg::frame_stream());
    gg::frame_stream().synchronize();

    auto one = read_vector3(impl.positions.cview());
    REQUIRE(one.size() == 1);
    CHECK(element_equal(one[0], Vector3{-2.0, 0.5, 4.0}));

    {
        auto box = impl.compute_vertex_bounding_box();
        CHECK(element_equal(box.min(), one[0].cast<float>()));
        CHECK(element_equal(box.max(), one[0].cast<float>()));
    }
    CHECK(impl.compute_axis_max_displacement() == 0.0);

    // step_forward with a zero displacement is the identity
    impl.step_forward(0.5);
    CHECK(arrays_equal(read_vector3(impl.positions.cview()), one));
}
