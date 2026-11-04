/********************************************************************
 * @file parity_scene.cpp
 * @brief T9.3a — the two scenes of the parity harness.
 *
 * Scope (see .kimix_cache/plans/research/03-parity-harness-research.md §1): the
 * plan's P0.3 asks for nine sim_case re-statements, but every one of them needs the
 * Newton/line-search loop that the Luisa backend does not have yet
 * (src/backends/luisa/sim_engine.cpp advance() = prologue -> predict_dof ->
 * update_state). Two scenes are enough to prove that the registry is not
 * hard-coded to one, and they cover the two probe routes the port publishes live
 * state through:
 *
 *   gravity_abd — one affine-body cube    -> AffineBodyStateAccessorFeature
 *                                             (transforms / velocities probes)
 *   gravity_fem — one finite-element tet  -> FiniteElementStateAccessorFeature
 *                                             (fe_positions / fe_velocities probes)
 *
 * Both bodies are re-statements, not inventions: gravity_abd is the abd-cube case of
 * apps/benchmarks/engine/ab_main.cpp:219-237 (the scene that is *measured* to run
 * 40 frames on the Luisa backend, benchmarks/results/2026-10-29/engine_ab.md), and
 * gravity_fem is apps/tests/sim_case/13_fem_3d_gravity.cpp:49-74. The geometry
 * construction is kept verbatim so a change here is always a deliberate change.
 *
 * NOTE: no `#ifdef`, no backend name, no asset file. Anything that depends on the
 * backend would break the "one scene definition drives both legs" rule.
 *********************************************************************/
#include <parity_scene.h>

// The umbrella header, like every apps/tests/sim_case/*.cpp case: the scene bodies
// below are re-statements of those cases, and the template instantiation of
// Object::Geometries::create<SimplicialComplex> needs the geometry/core details
// headers that only <uipc/uipc.h> pulls in for a plain executable.
#include <uipc/uipc.h>

// Constitutions are not part of the umbrella header (sim_case cases include them the
// same way).
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/stable_neo_hookean.h>

#include <array>
#include <cmath>

namespace uipc::parity
{
using namespace uipc;

namespace
{
    // ---------------------------------------------------------------------------
    //  gravity_abd — the tilted affine-body cube of ab_main.cpp
    // ---------------------------------------------------------------------------
    /// The six-tet decomposition of the cube [0,L]^3 (ab_main.cpp:192-209): every tet is
    /// a monotone path from v0 to v7 with determinant +L^3.
    std::array<Vector4i, 6> cube_tets() noexcept
    {
        return {{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}}};
    }

    /// Vertex index = x + 2y + 4z.
    std::array<Vector3, 8> cube_vertices(Float L) noexcept
    {
        std::array<Vector3, 8> vertices;
        for(int i = 0; i < 8; ++i)
        {
            const Float x = (i & 1) ? L : 0.0;
            const Float y = (i & 2) ? L : 0.0;
            const Float z = (i & 4) ? L : 0.0;
            vertices[i]   = Vector3{x, y, z};
        }
        return vertices;
    }

    Matrix4x4 make_transform(const Matrix3x3& A, const Vector3& t) noexcept
    {
        Matrix4x4 transform         = Matrix4x4::Identity();
        transform.block<3, 3>(0, 0) = A;
        transform.block<3, 1>(0, 3) = t;
        return transform;
    }

    geometry::SimplicialComplex make_abd_cube(Float L, Float density, Float kappa, const Matrix4x4& transform)
    {
        const auto vertices = cube_vertices(L);
        const auto tets     = cube_tets();

        auto sc = geometry::tetmesh(span<const Vector3>{vertices},
                                    span<const Vector4i>{tets});
        // `label_surface` writes `is_surf`, which the affine-body surface reporter reads.
        geometry::label_surface(sc);
        sc.instances().resize(1);

        constitution::AffineBodyConstitution abd;
        abd.apply_to(sc, kappa, density);

        // The instance transform carries the deformed affine state; gravity then acts on
        // it every frame, so the frame checksums are not the rest configuration.
        view(sc.transforms())[0] = transform;
        return sc;
    }

    void build_gravity_abd(core::Scene& scene)
    {
        constexpr Float kCubeSize  = 0.6;
        constexpr Float kDensity   = 800.0;
        constexpr Float kKappa     = 5.0e7;
        constexpr Float kTiltAngle = 0.35;

        const Float cos_tilt = std::cos(kTiltAngle);
        const Float sin_tilt = std::sin(kTiltAngle);

        Matrix3x3 tilt = Matrix3x3::Identity();
        tilt(0, 0)     = cos_tilt;
        tilt(0, 2)     = sin_tilt;
        tilt(2, 0)     = -sin_tilt;
        tilt(2, 2)     = cos_tilt;

        auto object = scene.objects().create("abd_cube");
        object->geometries().create(make_abd_cube(
            kCubeSize, kDensity, kKappa, make_transform(tilt, Vector3{0.0, 0.4, 0.0})));
    }

    Json describe_gravity_abd()
    {
        Json d = Json::object();
        d["mesh"] = "tetmesh of the six-tet decomposition of the cube [0, 0.6]^3";
        d["vertices"]     = 8;
        d["tetrahedra"]   = 6;
        d["instances"]    = 1;
        d["constitution"] = "affine_body";
        d["kappa"]        = 5.0e7;
        d["density"]      = 800.0;
        d["tilt_angle_y"] = 0.35;
        d["translation"]  = Json::array({0.0, 0.4, 0.0});
        return d;
    }

    // ---------------------------------------------------------------------------
    //  gravity_fem — the finite-element tet of 13_fem_3d_gravity.cpp:49-74
    // ---------------------------------------------------------------------------
    void build_gravity_fem(core::Scene& scene)
    {
        constitution::StableNeoHookean snh;

        auto object = scene.objects().create("tets");

        const std::array<Vector4i, 1> Ts = {Vector4i{0, 1, 2, 3}};
        const std::array<Vector3, 4>  Vs = {Vector3{0, 1, 0},
                                            Vector3{0, 0, 1},
                                            Vector3{-std::sqrt(3.0) / 2.0, 0, -0.5},
                                            Vector3{std::sqrt(3.0) / 2.0, 0, -0.5}};

        auto mesh =
            geometry::tetmesh(span<const Vector3>{Vs}, span<const Vector4i>{Ts});

        geometry::label_surface(mesh);
        geometry::label_triangle_orient(mesh);

        auto parm = constitution::ElasticModuli::youngs_poisson(1e5, 0.499);
        snh.apply_to(mesh, parm, 1e3);

        object->geometries().create(mesh);
    }

    Json describe_gravity_fem()
    {
        Json d          = Json::object();
        d["mesh"]       = "tetmesh of the regular tet of 13_fem_3d_gravity.cpp";
        d["vertices"]   = 4;
        d["tetrahedra"] = 1;
        d["constitution"]   = "stable_neo_hookean";
        d["youngs_modulus"] = 1e5;
        d["poisson_ratio"]  = 0.499;
        d["density"]        = 1e3;
        return d;
    }

    // ---------------------------------------------------------------------------
    //  harness_smoke — a geometry with no constitution
    //
    //  This is the harness's own proof-of-life scene, and it exists because of a
    //  measured, pre-existing defect of this tree: `World::init()` aborts for every
    //  scene that registers a dof-producing system (ABD or FEM), with
    //      Assertion '*type == *t' failed: Type mismatch: declared as double, got byte
    //      [LuisaCompute/src/ast/function_builder.cpp:436]
    //  raised *inside* init, right after the systems.json dump. The same abort is
    //  reproduced by the independent [abd_base] tag (apps/tests/backends/luisa/
    //  luisa_abd_base_test.cpp "abd_base_scene"), i.e. it is not this harness's scene.
    //  Fixing it belongs to the engine/advance work (plan 10), not to S0 -- but a
    //  harness that can never produce two artifacts cannot be validated either, so this
    //  scene keeps the *plumbing* end-to-end exercisable: it is a real geometry (so the
    //  vertices probe, the raw f64 writer, the per-frame checksum and the comparator all
    //  run) without a constitution (so no dof-based system is built).
    // ---------------------------------------------------------------------------
    void build_harness_smoke(core::Scene& scene)
    {
        const auto vertices = cube_vertices(0.6);
        const auto tets     = cube_tets();

        auto mesh = geometry::tetmesh(span<const Vector3>{vertices},
                                      span<const Vector4i>{tets});
        geometry::label_surface(mesh);

        auto object = scene.objects().create("smoke_box");
        object->geometries().create(mesh);
    }

    Json describe_harness_smoke()
    {
        Json d = Json::object();
        d["mesh"] = "tetmesh of the six-tet decomposition of the cube [0, 0.6]^3";
        d["vertices"]   = 8;
        d["tetrahedra"] = 6;
        d["constitution"] = "none (harness plumbing only, see the comment in parity_scene.cpp)";
        return d;
    }

    const std::array<SceneEntry, 3> kScenes = {{
        SceneEntry{"gravity_abd",
                   "apps/benchmarks/engine/ab_main.cpp:219-237 (abd-cube case)",
                   &build_gravity_abd,
                   &describe_gravity_abd},
        SceneEntry{"gravity_fem", "apps/tests/sim_case/13_fem_3d_gravity.cpp:49-74", &build_gravity_fem, &describe_gravity_fem},
        SceneEntry{"harness_smoke",
                   "T9.3a harness self-test: a geometry with no constitution",
                   &build_harness_smoke,
                   &describe_harness_smoke},
    }};
}  // namespace

span<const SceneEntry> scenes() noexcept
{
    return span<const SceneEntry>{kScenes.data(), kScenes.size()};
}

const SceneEntry* find_scene(std::string_view name) noexcept
{
    for(const auto& entry : kScenes)
    {
        if(entry.name == name)
            return &entry;
    }
    return nullptr;
}
}  // namespace uipc::parity
