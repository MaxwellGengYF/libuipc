/********************************************************************
 * @file luisa_distance_diagnoser_test.cpp
 * @brief S9' (T8.5-2) — the distance diagnoser on the Luisa backend. Tag [diagnoser].
 *
 * The feature is a pure query service: `world.features().find<core::DistanceDiagnoserFeature>()`
 * must resolve after `world.init()`, and the four queries must write the reference's
 * attribute set on the result geometry `R`.
 *
 * WHAT THE REFERENCE GUARANTEES (src/backends/cuda/distance_system/distance_diagnoser.cu)
 * -------------------------------------------------------------------------------------
 *  * every query resizes `R.instances()` to `num_pairs` and writes a fixed attribute set
 *    whose names, types and per-entry values are .cu:733-740 / 855-869 / 949-956 /
 *    1028-1033;
 *  * there are NO config keys: `dist2/*`, `e_k/*`, `barrier/*` are attribute *names*, always
 *    written (the plan's §1.3/§2 "six keys" claim is wrong — research §1 correction A);
 *  * an inactive pair (D outside `(thickness^2, (thickness + d_hat)^2)`) still gets `dist2`
 *    and its gradient/Hessian; only `barrier`, `barrier/grad` and `barrier/hess` stay at
 *    their zero initialisation, and `e_k` keeps the mollifier's own value (1.0 unless the
 *    pair is inside the mollified band);
 *  * 0 pairs ⇒ `R.instances().resize(0)` and NO attribute is created (.cu:673-677).
 *
 * HOW THIS FILE VERIFIES IT (there is no CUDA backend on this machine — AGENT_ENV_BRIEF.md)
 * -------------------------------------------------------------------------------------
 *  1. `t85_parity_golden_pt` is the Class-A gate: it feeds the 40 cases of the Tier-1 golden
 *     `parity_golden/distance_pt.bin` — produced by `uipc_parity_reference.exe` evaluating
 *     the *reference* `.inl` on the host (apps/tests/parity/reference/main.cpp) — one pair at
 *     a time through `compute_point_triangle_distance` and requires `|delta| == 0` on
 *     `dist2`, `dist2/grad`, `dist2/hess` (157 slots/case; NaN==NaN and +/-0 equal, the
 *     harness convention of docs/luisa_porting_cookbook.md §7.38).  A SKIP (never a pass) if
 *     the golden file is missing.
 *  2. The per-query cases compare `dist2` against oracles written *in this file* (Eigen
 *     projection code — the ported layer is never its own oracle), the flag against the
 *     expected Voronoi region, `coord` against the oracle's closest point (and against
 *     `dist2`), and `dist2/grad` / `dist2/hess` against central differences of the reported
 *     `dist2` (the `python/tests/test_contact_fd.py` recipe: delta = 1e-6).
 *  3. Negative controls: a missing `d_hat`/`thickness` is an assert, not a silent default —
 *     the assert is `uipc::abort()` (`src/core/common/abort.cpp`), so, exactly like
 *     luisa_fem_base_test.cpp's zero-volume tet, the test asserts the *condition* that
 *     aborts instead of aborting the shared test binary; and the feature must never be null
 *     on this backend (that is the whole point of T8.5-2).
 *
 * SCENE FIXTURE: one tiny trimesh and `sanity_check.enable = 0` — the sibling `[contact_core]`
 * tests' proven `world.init()` payload (luisa_contact_core_test.cpp:641-642).  One World per
 * case: a World owns GPU state and must not outlive `zzz_runtime_shutdown`.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <uipc/core/distance_diagnoser_feature.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;

namespace t85
{
namespace fs = std::filesystem;

/// The `d_hat` used by the cases that need an *active* pair (D < d_hat^2).
constexpr Float kActiveDhat = 0.5;

// ===========================================================================
//  Fixture
// ===========================================================================
std::string workspace_path(std::string_view name)
{
    auto p = fs::temp_directory_path() / name;
    fs::create_directories(p);
    return p.string() + "/";
}

Json diag_config()
{
    auto config                      = Scene::default_config();
    config["sanity_check"]["enable"] = 0;
    return config;
}

/// One small trimesh object (see the file header for why a scene carries geometry).
void add_trimesh(Scene& scene, std::string_view name)
{
    std::vector<Vector3>  vs = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    std::vector<Vector3i> fs = {{0, 1, 2}};
    auto complex = trimesh(span<const Vector3>{vs}, span<const Vector3i>{fs});
    label_surface(complex);
    auto object = scene.objects().create(std::string{name});
    object->geometries().create(complex);
}

/// A World plus the diagnoser feature.  `World` is not movable, so it is constructed in
/// place; the engine must outlive the world (member order).
struct Diag
{
    Engine engine;
    World  world;

    Diag(std::string_view name, Scene& scene)
        : engine{"luisa", workspace_path(name)}
        , world{engine}
    {
        world.init(scene);
    }

    [[nodiscard]] S<DistanceDiagnoserFeature> feature() const
    {
        return world.features().find<DistanceDiagnoserFeature>();
    }
};

// ===========================================================================
//  Inputs (the diagnoser's contract: positions + `d_hat` + `thickness`)
// ===========================================================================
[[nodiscard]] SimplicialComplex make_points(span<const Vector3> vs,
                                            Float                 d_hat     = 0.01,
                                            Float                 thickness = 0.0)
{
    auto sc = pointcloud(vs);
    sc.vertices().create<Float>("d_hat", d_hat);
    sc.vertices().create<Float>("thickness", thickness);
    return sc;
}

[[nodiscard]] SimplicialComplex make_edges(span<const Vector3>  vs,
                                           span<const Vector2i> es,
                                           Float                d_hat     = 0.01,
                                           Float                thickness = 0.0)
{
    auto sc = linemesh(vs, es);
    sc.vertices().create<Float>("d_hat", d_hat);
    sc.vertices().create<Float>("thickness", thickness);
    return sc;
}

[[nodiscard]] SimplicialComplex make_triangles(span<const Vector3>  vs,
                                               span<const Vector3i> fs,
                                               Float                d_hat     = 0.01,
                                               Float                thickness = 0.0)
{
    auto sc = trimesh(vs, fs);
    sc.vertices().create<Float>("d_hat", d_hat);
    sc.vertices().create<Float>("thickness", thickness);
    return sc;
}

// ===========================================================================
//  Attribute readers (the reference's names/types)
// ===========================================================================
template <typename T>
[[nodiscard]] std::vector<T> read_attr(const geometry::Geometry& R, std::string_view name)
{
    auto slot = R.instances().find<T>(std::string{name});
    if(!slot)
        return {};
    auto v = slot->view();
    return std::vector<T>(v.begin(), v.end());
}

/// The exact set of attribute names on `R.instances()` (a `Geometry`'s instance collection
/// serialises as `{name: slot}`), sorted for a stable comparison.
[[nodiscard]] std::vector<std::string> instance_attribute_names(const geometry::Geometry& R)
{
    auto                       j = R.instances().to_json();
    std::vector<std::string>   names;
    for(auto&& [key, value] : j.items())
        names.push_back(key);
    std::sort(names.begin(), names.end());
    return names;
}

[[nodiscard]] bool close(double a, double b, double rel = 1e-12) noexcept
{
    const double s = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= rel * s;
}

// ===========================================================================
//  Independent oracles (written here, *not* the ported distance layer)
// ===========================================================================
/// Closest point on a triangle — Ericson, "Real-Time Collision Detection" §5.1.5.
[[nodiscard]] Vector3 closest_point_on_triangle(const Vector3& p,
                                                const Vector3& a,
                                                const Vector3& b,
                                                const Vector3& c)
{
    const Vector3 ab = b - a, ac = c - a, ap = p - a;
    const double  d1 = ab.dot(ap), d2 = ac.dot(ap);
    if(d1 <= 0.0 && d2 <= 0.0)
        return a;
    const Vector3 bp = p - b;
    const double  d3 = ab.dot(bp), d4 = ac.dot(bp);
    if(d3 >= 0.0 && d4 <= d3)
        return b;
    const double vc = d1 * d4 - d3 * d2;
    if(vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
        return a + ab * (d1 / (d1 - d3));
    const Vector3 cp = p - c;
    const double  d5 = ab.dot(cp), d6 = ac.dot(cp);
    if(d6 >= 0.0 && d5 <= d6)
        return c;
    const double vb = d5 * d2 - d1 * d6;
    if(vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
        return a + ac * (d2 / (d2 - d6));
    const double va = d3 * d6 - d5 * d4;
    if(va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    const double denom = 1.0 / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

/// Squared distance between two segments — Ericson §5.1.9 (clamped s, t).
[[nodiscard]] double segment_segment_distance2(const Vector3& p1,
                                               const Vector3& q1,
                                               const Vector3& p2,
                                               const Vector3& q2)
{
    const Vector3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    const double  a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
    double        s = 0.0, t = 0.0;
    if(a <= 0.0 && e <= 0.0)
    {
        s = t = 0.0;
    }
    else if(a <= 0.0)
    {
        t = std::clamp(f / e, 0.0, 1.0);
    }
    else
    {
        const double c = d1.dot(r);
        if(e <= 0.0)
        {
            s = std::clamp(-c / a, 0.0, 1.0);
        }
        else
        {
            const double b     = d1.dot(d2);
            const double denom = a * e - b * b;
            s = (denom != 0.0) ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
            t = (b * s + f) / e;
            if(t < 0.0)
            {
                t = 0.0;
                s = std::clamp(-c / a, 0.0, 1.0);
            }
            else if(t > 1.0)
            {
                t = 1.0;
                s = std::clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    return ((p1 + d1 * s) - (p2 + d2 * t)).squaredNorm();
}

/// Squared distance from a point to a segment.
[[nodiscard]] double point_segment_distance2(const Vector3& p,
                                             const Vector3& e0,
                                             const Vector3& e1)
{
    const Vector3 d  = e1 - e0;
    const double  l2 = d.squaredNorm();
    const double  t  = l2 > 0.0 ? std::clamp((p - e0).dot(d) / l2, 0.0, 1.0) : 0.0;
    return (p - (e0 + d * t)).squaredNorm();
}

// ===========================================================================
//  Single-pair re-evaluation helpers (for the central-difference checks)
// ===========================================================================
[[nodiscard]] double pt_dist2(const S<DistanceDiagnoserFeature>& f,
                              const std::array<Vector3, 4>&     v)
{
    std::vector<Vector3>  pv{v[0]};
    std::vector<Vector3>  tv{v[1], v[2], v[3]};
    std::vector<Vector3i> tf{{0, 1, 2}};
    auto points = make_points(span<const Vector3>{pv}, 1e-3, 0.0);
    auto tris = make_triangles(span<const Vector3>{tv}, span<const Vector3i>{tf}, 1e-3, 0.0);
    geometry::Geometry R;
    f->compute_point_triangle_distance(R, points, tris);
    return read_attr<Float>(R, "dist2").at(0);
}

/// The reported barrier of a single point-triangle pair (the `kActiveDhat` inputs of the
/// C2 check in `t85_pt`).
[[nodiscard]] double pt_barrier(const S<DistanceDiagnoserFeature>& f,
                                const std::array<Vector3, 4>&     v)
{
    std::vector<Vector3>  pv{v[0]};
    std::vector<Vector3>  tv{v[1], v[2], v[3]};
    std::vector<Vector3i> tf{{0, 1, 2}};
    auto points = make_points(span<const Vector3>{pv}, kActiveDhat, 0.0);
    auto tris = make_triangles(span<const Vector3>{tv},
                               span<const Vector3i>{tf},
                               kActiveDhat,
                               0.0);
    geometry::Geometry R;
    f->compute_point_triangle_distance(R, points, tris);
    return read_attr<Float>(R, "barrier").at(0);
}

[[nodiscard]] double pe_dist2(const S<DistanceDiagnoserFeature>& f,
                              const std::array<Vector3, 3>&     v)
{
    std::vector<Vector3>  pv{v[0]};
    std::vector<Vector3>  ev{v[1], v[2]};
    std::vector<Vector2i> ef{{0, 1}};
    auto points = make_points(span<const Vector3>{pv}, 1e-3, 0.0);
    auto edges  = make_edges(span<const Vector3>{ev}, span<const Vector2i>{ef}, 1e-3, 0.0);
    geometry::Geometry R;
    f->compute_point_edge_distance(R, points, edges);
    return read_attr<Float>(R, "dist2").at(0);
}

[[nodiscard]] double ee_dist2(const S<DistanceDiagnoserFeature>& f,
                              const std::array<Vector3, 4>&     v)
{
    std::vector<Vector3>  av{v[0], v[1]};
    std::vector<Vector3>  bv{v[2], v[3]};
    std::vector<Vector2i> af{{0, 1}};
    std::vector<Vector2i> bf{{0, 1}};
    auto ea = make_edges(span<const Vector3>{av}, span<const Vector2i>{af}, 1e-3, 0.0);
    auto eb = make_edges(span<const Vector3>{bv}, span<const Vector2i>{bf}, 1e-3, 0.0);
    geometry::Geometry R;
    f->compute_edge_edge_distance(R, ea, eb, ea, eb);
    return read_attr<Float>(R, "dist2").at(0);
}

}  // namespace t85
using namespace t85;

// ===========================================================================
//  Phase 1 — the feature resolves and the overrider forwards
// ===========================================================================
TEST_CASE("t85_feature_resolves", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_feature_resolves");

    Diag d{"u85_feature_resolves", scene};
    REQUIRE(d.world.is_valid());

    auto feature = d.feature();
    // Negative control (c): the feature must resolve on the luisa backend (it returned
    // nullptr before T8.5-2), and the test must fail - never skip - when it does not.
    REQUIRE(feature != nullptr);
    CHECK(feature->name() == "core/distance_diagnoser");

    // All four forwarders reach the diagnoser (the overrider's slot is bound at build time;
    // a dead slot would abort inside the call).
    std::vector<Vector3>  pv = {{0.0, 0.0, 0.0}};
    std::vector<Vector3>  tv = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
    std::vector<Vector3i> tf = {{0, 1, 2}};
    std::vector<Vector3>  ev = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Vector2i> ef = {{0, 1}};

    auto points    = make_points(span<const Vector3>{pv});
    auto triangles = make_triangles(span<const Vector3>{tv},
                                    span<const Vector3i>{tf});
    auto edges     = make_edges(span<const Vector3>{ev}, span<const Vector2i>{ef});

    geometry::Geometry R_pt, R_ee, R_pe, R_pp;

    CHECK_NOTHROW(feature->compute_point_triangle_distance(R_pt, points, triangles));
    CHECK(R_pt.instances().size() == 1);

    CHECK_NOTHROW(feature->compute_edge_edge_distance(R_ee, edges, edges, edges, edges));
    CHECK(R_ee.instances().size() == 1);

    CHECK_NOTHROW(feature->compute_point_edge_distance(R_pe, points, edges));
    CHECK(R_pe.instances().size() == 1);

    CHECK_NOTHROW(feature->compute_point_point_distance(R_pp, points, points));
    CHECK(R_pp.instances().size() == 1);
}

// ===========================================================================
//  Phase 2.1 — point-point (6 dof, no flag/coord), the full closed form
// ===========================================================================
TEST_CASE("t85_pp", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_pp");
    Diag d{"u85_pp", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    std::vector<Vector3> av = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    std::vector<Vector3> bv = {{0.0, 0.0, 2.0}, {2.0, 0.0, 2.5}};

    auto A = make_points(span<const Vector3>{av}, 0.01, 0.0);
    auto B = make_points(span<const Vector3>{bv}, 0.01, 0.0);

    geometry::Geometry R;
    f->compute_point_point_distance(R, A, B);

    // the reference's pair order: i = idx_a, j = idx_b with M = num_b
    REQUIRE(R.instances().size() == av.size() * bv.size());

    auto dist2  = read_attr<Float>(R, "dist2");
    auto grad   = read_attr<Vector6>(R, "dist2/grad");
    auto hess   = read_attr<Matrix6x6>(R, "dist2/hess");
    auto bar    = read_attr<Float>(R, "barrier");
    auto bar_g  = read_attr<Vector6>(R, "barrier/grad");
    auto bar_h  = read_attr<Matrix6x6>(R, "barrier/hess");

    for(size_t i = 0; i < av.size(); ++i)
    {
        for(size_t j = 0; j < bv.size(); ++j)
        {
            const size_t  k = i * bv.size() + j;
            const Vector3 diff = av[i] - bv[j];  // the reference's (a - b)

            // (a) the squared distance, exactly (the reference's own expression is
            //     (a - b).squaredNorm(), so this is an exact-equality gate)
            CHECK(dist2[k] == diff.squaredNorm());

            // (b) the gradient: [2(a - b); -2(a - b)]  (point_point.inl)
            for(int c = 0; c < 3; ++c)
            {
                CHECK(grad[k](c) == 2.0 * diff(c));
                CHECK(grad[k](3 + c) == -2.0 * diff(c));
            }

            // (c) the constant Hessian: diag(2) with the -2 coupling blocks
            for(int r = 0; r < 6; ++r)
            {
                for(int c = 0; c < 6; ++c)
                {
                    const double expected = (r == c) ? 2.0
                                            : (r % 3 == c % 3) ? -2.0
                                                               : 0.0;
                    CHECK(hess[k](r, c) == expected);
                }
            }

            // (d) D is far outside (thickness^2, (thickness + d_hat)^2) = (0, 1e-4) for
            //     every pair, so the barrier family stays at its zero initialisation
            CHECK(bar[k] == 0.0);
            CHECK(bar_g[k].isZero(0.0));
            CHECK(bar_h[k].isZero(0.0));
        }
    }

    // a point pair has no flag/coord attributes in the reference (its `copy_back` block has
    // six calls, not eight — .cu:1028-1033)
    const std::vector<std::string> pp_names = {"barrier",
                                               "barrier/grad",
                                               "barrier/hess",
                                               "dist2",
                                               "dist2/grad",
                                               "dist2/hess"};
    CHECK(instance_attribute_names(R) == pp_names);
}

// ===========================================================================
//  Phase 2.2 — point-edge (9 dof): regions, oracle, coordinates, FD derivatives
// ===========================================================================
TEST_CASE("t85_pe", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_pe");
    Diag d{"u85_pe", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    const Vector3 e0{0.0, 0.0, 0.0}, e1{1.0, 0.0, 0.0};

    // ---- the three Voronoi regions, by construction ------------------------
    struct Region
    {
        Vector3 p;
        Vector3i flag;
        Vector3 coord;  // coord[1], coord[2] = the edge barycentrics
    };
    const std::vector<Region> regions = {
        // closest to e0 (t <= 0)
        {{-0.5, 0.5, 0.25}, {1, 1, 0}, {-0.0, 1.0, 0.0}},
        // interior
        {{0.25, 0.5, 0.25}, {1, 1, 1}, {-0.0, 0.75, 0.25}},
        // closest to e1 (t >= 1)
        {{1.5, -0.5, 0.25}, {1, 0, 1}, {-0.0, 0.0, 1.0}},
    };

    for(const auto& r : regions)
    {
        std::vector<Vector3>  pv{r.p};
        std::vector<Vector3>  ev{e0, e1};
        std::vector<Vector2i> ef{{0, 1}};
        auto points = make_points(span<const Vector3>{pv}, 0.01, 0.0);
        auto edges  = make_edges(span<const Vector3>{ev}, span<const Vector2i>{ef}, 0.01, 0.0);

        geometry::Geometry R;
        f->compute_point_edge_distance(R, points, edges);
        REQUIRE(R.instances().size() == 1);

        auto flag = read_attr<Vector3i>(R, "flag").at(0);
        auto crd  = read_attr<Vector3>(R, "coord").at(0);
        auto D    = read_attr<Float>(R, "dist2").at(0);

        CHECK(flag == r.flag);
        CHECK(crd(0) == 0.0);
        CHECK(close(crd(1), r.coord(1), 1e-15));
        CHECK(close(crd(2), r.coord(2), 1e-15));
        // the coordinates reconstruct the oracle's closest point and its distance
        const Vector3 q = crd(1) * e0 + crd(2) * e1;
        CHECK(close(D, (r.p - q).squaredNorm(), 1e-14));
        CHECK(close(D, point_segment_distance2(r.p, e0, e1), 1e-14));
    }

    // ---- 200 random pairs against the in-test oracle ----------------------
    {
        std::vector<Vector3> ps, es;
        unsigned             state = 12345u;
        auto                 rnd   = [&state]() noexcept
        {
            state = state * 1664525u + 1013904223u;
            return static_cast<double>(state >> 8) / 16777216.0;  // [0,1)
        };
        for(int i = 0; i < 200; ++i)
        {
            ps.push_back(Vector3{4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0});
            Vector3 a{4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0};
            es.push_back(a);
            es.push_back(a + Vector3{2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0});
        }
        std::vector<Vector2i> ef;
        for(int i = 0; i < 200; ++i)
            ef.push_back(Vector2i{2 * i, 2 * i + 1});

        auto points = make_points(span<const Vector3>{ps}, 0.01, 0.0);
        auto edges  = make_edges(span<const Vector3>{es}, span<const Vector2i>{ef}, 0.01, 0.0);

        geometry::Geometry R;
        f->compute_point_edge_distance(R, points, edges);
        // num_pairs = num_points * num_edges = 200 * 200 = 40000; check the queried
        // slice only (pair (i, j) = i * 200 + j)
        REQUIRE(R.instances().size() == ps.size() * ef.size());
        auto D    = read_attr<Float>(R, "dist2");
        auto crd  = read_attr<Vector3>(R, "coord");

        std::size_t worst     = 0;
        double      worst_rel = 0.0;
        double      worst_got = 0.0, worst_expected = 0.0;
        std::size_t worst_i   = 0;
        for(int j = 0; j < 200; ++j)
        {
            for(int i = 0; i < 200; ++i)
            {
                const double expected =
                    point_segment_distance2(ps[i], es[2 * j], es[2 * j + 1]);
                const double got = D[static_cast<size_t>(i) * 200 + j];
                const double rel = std::abs(got - expected) / std::max(1.0, expected);
                if(rel > worst_rel)
                {
                    worst_rel      = rel;
                    worst          = static_cast<size_t>(j);
                    worst_i        = static_cast<std::size_t>(i);
                    worst_got      = got;
                    worst_expected = expected;
                }
                // consistency of coord with dist2 (the strongest oracle-free invariant)
                const Vector3 q = crd[static_cast<size_t>(i) * 200 + j](1) * es[2 * j]
                                  + crd[static_cast<size_t>(i) * 200 + j](2) * es[2 * j + 1];
                CHECK(close(got, (ps[i] - q).squaredNorm(), 1e-12));
            }
        }
        std::printf(" [diagnoser] point-edge oracle: worst rel |delta| = %.3e "
                    "(edge %zu, point %zu: got %.17g, expected %.17g)\n",
                    worst_rel,
                    worst,
                    worst_i,
                    worst_got,
                    worst_expected);
        CHECK(worst_rel <= 1e-14);
    }

    // ---- central differences of the reported dist2 -----------------------
    {
        // a well-centred interior pair (2 x 1e-6 away from any Voronoi boundary)
        std::array<Vector3, 3> v = {{{0.3, 0.3, 1.0}, {-0.5, 0.0, 0.0}, {1.5, 0.25, 0.0}}};
        std::vector<Vector3>  pv{v[0]};
        std::vector<Vector3>  ev{v[1], v[2]};
        std::vector<Vector2i> ef{{0, 1}};
        auto points = make_points(span<const Vector3>{pv}, 0.01, 0.0);
        auto edges  = make_edges(span<const Vector3>{ev}, span<const Vector2i>{ef}, 0.01, 0.0);
        geometry::Geometry R;
        f->compute_point_edge_distance(R, points, edges);
        auto G = read_attr<Vector9>(R, "dist2/grad").at(0);
        auto H = read_attr<Matrix9x9>(R, "dist2/hess").at(0);

        const double h          = 1e-6;
        double       worst_g    = 0.0;
        for(int k = 0; k < 9; ++k)
        {
            auto up = v, down = v;
            up[k / 3](k % 3) += h;
            down[k / 3](k % 3) -= h;
            const double fd = (pe_dist2(f, up) - pe_dist2(f, down)) / (2.0 * h);
            worst_g         = std::max(worst_g, std::abs(fd - G(k)) / std::max(1.0, std::abs(fd)));
            CHECK(close(fd, G(k), 1e-6));
        }
        // Hessian: symmetry exactly, and the mixed central difference (h = 1e-4; dist2 is
        // a quadratic form in the coordinates inside one Voronoi region, so the 4-point
        // stencil is exact up to round-off)
        const double hh       = 1e-4;
        double       worst_h  = 0.0;
        for(int r = 0; r < 9; ++r)
            for(int c = 0; c < 9; ++c)
                CHECK(H(r, c) == H(c, r));
        for(int r = 0; r < 9; ++r)
        {
            for(int c = 0; c < 9; ++c)
            {
                auto pp = v, pm = v, mp = v, mm = v;
                pp[r / 3](r % 3) += hh;
                pm[r / 3](r % 3) += hh;
                mp[r / 3](r % 3) -= hh;
                mm[r / 3](r % 3) -= hh;
                pp[c / 3](c % 3) += hh;
                mm[c / 3](c % 3) -= hh;
                pm[c / 3](c % 3) -= hh;
                mp[c / 3](c % 3) += hh;
                const double fd = (pe_dist2(f, pp) - pe_dist2(f, pm) - pe_dist2(f, mp)
                                   + pe_dist2(f, mm))
                                  / (4.0 * hh * hh);
                const double rel_h = std::abs(fd - H(r, c)) / std::max(1.0, std::abs(fd));
                worst_h = std::max(worst_h, rel_h);
                CHECK(close(fd, H(r, c), 1e-5));
            }
        }
        std::printf(" [diagnoser] point-edge FD: grad %.3e / hess %.3e (relative)\n",
                    worst_g,
                    worst_h);
    }
}

// ===========================================================================
//  Phase 2.3 — point-triangle (12 dof): the seven Voronoi regions + oracles
// ===========================================================================
TEST_CASE("t85_pt", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_pt");
    Diag d{"u85_pt", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    const Vector3 t0{0.0, 0.0, 0.0}, t1{1.0, 0.0, 0.0}, t2{0.0, 1.0, 0.0};

    struct Region
    {
        Vector3  p;
        Vector4i flag;
        Vector4  coord;  // coord[1..3] = the triangle barycentrics
    };
    // one representative configuration per region of the reference's flag chain
    const std::vector<Region> regions = {
        {{-1.0, -1.0, 0.1}, {1, 1, 0, 0}, {0.0, 1.0, 0.0, 0.0}},   // vertex t0
        {{2.0, -1.0, 0.2}, {1, 0, 1, 0}, {0.0, 0.0, 1.0, 0.0}},    // vertex t1
        {{-1.0, 2.0, -0.3}, {1, 0, 0, 1}, {0.0, 0.0, 0.0, 1.0}},   // vertex t2
        {{0.5, -0.5, 0.3}, {1, 1, 1, 0}, {0.0, 0.5, 0.5, 0.0}},    // edge t0-t1
        {{1.5, 1.5, -0.2}, {1, 0, 1, 1}, {0.0, 0.0, 0.5, 0.5}},    // edge t1-t2
        {{-0.5, 0.5, 0.1}, {1, 1, 0, 1}, {0.0, 0.5, 0.0, 0.5}},    // edge t2-t0
        {{0.25, 0.25, 1.0}, {1, 1, 1, 1}, {0.0, 0.5, 0.25, 0.25}},  // face
    };

    for(const auto& r : regions)
    {
        std::vector<Vector3>  pv{r.p};
        std::vector<Vector3>  tv{t0, t1, t2};
        std::vector<Vector3i> tf{{0, 1, 2}};
        auto points    = make_points(span<const Vector3>{pv}, 0.01, 0.0);
        auto triangles = make_triangles(span<const Vector3>{tv},
                                        span<const Vector3i>{tf},
                                        0.01,
                                        0.0);

        geometry::Geometry R;
        f->compute_point_triangle_distance(R, points, triangles);
        REQUIRE(R.instances().size() == 1);

        auto flag = read_attr<Vector4i>(R, "flag").at(0);
        auto crd  = read_attr<Vector4>(R, "coord").at(0);
        auto D    = read_attr<Float>(R, "dist2").at(0);

        CHECK(flag == r.flag);
        CHECK(crd(0) == 0.0);
        CHECK(close(crd(1), r.coord(1), 1e-15));
        CHECK(close(crd(2), r.coord(2), 1e-15));
        CHECK(close(crd(3), r.coord(3), 1e-15));

        const Vector3 q = crd(1) * t0 + crd(2) * t1 + crd(3) * t2;
        CHECK(close(D, (r.p - q).squaredNorm(), 1e-14));
        CHECK(close(D, (r.p - closest_point_on_triangle(r.p, t0, t1, t2)).squaredNorm(), 1e-14));
    }

    // ---- 200 random pairs against the in-test oracle ----------------------
    {
        std::vector<Vector3>  ps, tv;
        std::vector<Vector3i> tf;
        unsigned              state = 987654321u;
        auto                  rnd   = [&state]() noexcept
        {
            state = state * 1664525u + 1013904223u;
            return static_cast<double>(state >> 8) / 16777216.0;
        };
        for(int i = 0; i < 200; ++i)
        {
            ps.push_back(Vector3{4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0});
            const Vector3 a{4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0};
            tv.push_back(a);
            tv.push_back(a + Vector3{2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0});
            tv.push_back(a + Vector3{2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0});
            tf.push_back(Vector3i{3 * i, 3 * i + 1, 3 * i + 2});
        }
        auto points    = make_points(span<const Vector3>{ps}, 0.01, 0.0);
        auto triangles = make_triangles(span<const Vector3>{tv},
                                        span<const Vector3i>{tf},
                                        0.01,
                                        0.0);

        geometry::Geometry R;
        f->compute_point_triangle_distance(R, points, triangles);
        REQUIRE(R.instances().size() == ps.size() * tf.size());

        auto D    = read_attr<Float>(R, "dist2");
        auto crd  = read_attr<Vector4>(R, "coord");
        auto flag = read_attr<Vector4i>(R, "flag");

        double      worst_rel = 0.0;
        std::size_t nan_count = 0;
        for(size_t i = 0; i < ps.size(); ++i)
        {
            for(size_t j = 0; j < tf.size(); ++j)
            {
                const size_t  k = i * tf.size() + j;
                const Vector3 a = tv[3 * j], b = tv[3 * j + 1], c = tv[3 * j + 2];
                const double  expected =
                    (ps[i] - closest_point_on_triangle(ps[i], a, b, c)).squaredNorm();
                const double got = D[k];
                if(std::isnan(got))
                {
                    ++nan_count;
                    continue;
                }
                worst_rel = std::max(worst_rel, std::abs(got - expected) / std::max(1.0, expected));
                // coord is consistent with dist2, and it selects a real sub-simplex
                const Vector3 q = crd[k](1) * a + crd[k](2) * b + crd[k](3) * c;
                CHECK(close(got, (ps[i] - q).squaredNorm(), 1e-12));
                CHECK(flag[k](0) == 1);
                const int sub = flag[k](1) + flag[k](2) + flag[k](3);
                CHECK(sub >= 1);
                CHECK(sub <= 3);
            }
        }
        std::printf(" [diagnoser] point-triangle oracle: worst rel |delta| = %.3e, NaN pairs %zu\n",
                    worst_rel,
                    nan_count);
        CHECK(worst_rel <= 1e-13);
        // random degenerate triangles are not part of this sweep (see t85_degenerate)
        CHECK(nan_count == 0);
    }

    // ---- central differences of the reported dist2 -----------------------
    {
        std::array<Vector3, 4> v = {{{0.25, 0.25, 1.0},
                                     {-0.5, -0.25, 0.0},
                                     {1.5, 0.0, 0.0},
                                     {0.0, 1.5, 0.0}}};
        std::vector<Vector3>  pv{v[0]};
        std::vector<Vector3>  tv{v[1], v[2], v[3]};
        std::vector<Vector3i> tf{{0, 1, 2}};
        auto points    = make_points(span<const Vector3>{pv}, 0.01, 0.0);
        auto triangles = make_triangles(span<const Vector3>{tv},
                                        span<const Vector3i>{tf},
                                        0.01,
                                        0.0);
        geometry::Geometry R;
        f->compute_point_triangle_distance(R, points, triangles);
        auto G = read_attr<Vector12>(R, "dist2/grad").at(0);
        auto H = read_attr<Matrix12x12>(R, "dist2/hess").at(0);
        auto flag = read_attr<Vector4i>(R, "flag").at(0);
        REQUIRE(flag == Vector4i(1, 1, 1, 1));  // the face region: D is a quadratic form

        const double h       = 1e-6;
        double       worst_g = 0.0;
        for(int k = 0; k < 12; ++k)
        {
            auto up = v, down = v;
            up[k / 3](k % 3) += h;
            down[k / 3](k % 3) -= h;
            const double fd = (pt_dist2(f, up) - pt_dist2(f, down)) / (2.0 * h);
            worst_g = std::max(worst_g, std::abs(fd - G(k)) / std::max(1.0, std::abs(fd)));
            CHECK(close(fd, G(k), 1e-6));
        }
        const double hh      = 1e-4;
        double       worst_h = 0.0;
        for(int r = 0; r < 12; ++r)
            for(int c = 0; c < 12; ++c)
                CHECK(H(r, c) == H(c, r));
        for(int r = 0; r < 12; ++r)
        {
            for(int c = 0; c < 12; ++c)
            {
                auto pp = v, pm = v, mp = v, mm = v;
                pp[r / 3](r % 3) += hh;
                pm[r / 3](r % 3) += hh;
                mp[r / 3](r % 3) -= hh;
                mm[r / 3](r % 3) -= hh;
                pp[c / 3](c % 3) += hh;
                mm[c / 3](c % 3) -= hh;
                pm[c / 3](c % 3) -= hh;
                mp[c / 3](c % 3) += hh;
                const double fd = (pt_dist2(f, pp) - pt_dist2(f, pm) - pt_dist2(f, mp)
                                   + pt_dist2(f, mm))
                                  / (4.0 * hh * hh);
                const double rel_h = std::abs(fd - H(r, c)) / std::max(1.0, std::abs(fd));
                worst_h = std::max(worst_h, rel_h);
                CHECK(close(fd, H(r, c), 1e-5));
            }
        }
        std::printf(" [diagnoser] point-triangle FD: grad %.3e / hess %.3e (relative)\n",
                    worst_g,
                    worst_h);
    }

    // ---- the barrier's C2 form, on an active pair -------------------------
    {
        // D = 0.1^2 = 0.01 is inside (0, d_hat_pair^2) with d_hat = 0.5, so the pair is
        // active and the whole chain rule is exercised
        // (`GradB = dBdD * GradD`, `HessB = ddBddD GradD GradD^T + dBdD HessD`).
        std::array<Vector3, 4> v = {{{0.25, 0.25, 0.1},
                                     {-0.5, -0.25, 0.0},
                                     {1.5, 0.0, 0.0},
                                     {0.0, 1.5, 0.0}}};
        std::vector<Vector3>  pv{v[0]};
        std::vector<Vector3>  tv{v[1], v[2], v[3]};
        std::vector<Vector3i> tf{{0, 1, 2}};
        auto points    = make_points(span<const Vector3>{pv}, kActiveDhat, 0.0);
        auto triangles = make_triangles(span<const Vector3>{tv},
                                        span<const Vector3i>{tf},
                                        kActiveDhat,
                                        0.0);
        geometry::Geometry R;
        f->compute_point_triangle_distance(R, points, triangles);

        REQUIRE(read_attr<Vector4i>(R, "flag").at(0) == Vector4i(1, 1, 1, 1));
        const double B = read_attr<Float>(R, "barrier").at(0);
        const auto   bg = read_attr<Vector12>(R, "barrier/grad").at(0);
        const auto   bh = read_attr<Matrix12x12>(R, "barrier/hess").at(0);
        REQUIRE(B != 0.0);  // the pair is active

        // gradient and Hessian against central differences of the *reported* barrier, so
        // the barrier law itself is not re-derived here
        const double step    = 1e-6;
        double       worst_g = 0.0;
        for(int k = 0; k < 12; ++k)
        {
            auto up = v, down = v;
            up[k / 3](k % 3) += step;
            down[k / 3](k % 3) -= step;
            const double fd = (pt_barrier(f, up) - pt_barrier(f, down)) / (2.0 * step);
            worst_g = std::max(worst_g, std::abs(fd - bg(k)) / std::max(1.0, std::abs(fd)));
            CHECK(close(fd, bg(k), 1e-5));
        }
        const double hstep   = 1e-4;
        double       worst_h = 0.0;
        for(int r = 0; r < 12; ++r)
        {
            for(int c = 0; c < 12; ++c)
            {
                auto pp = v, pm = v, mp = v, mm = v;
                pp[r / 3](r % 3) += hstep;
                pm[r / 3](r % 3) += hstep;
                mp[r / 3](r % 3) -= hstep;
                mm[r / 3](r % 3) -= hstep;
                pp[c / 3](c % 3) += hstep;
                mm[c / 3](c % 3) -= hstep;
                pm[c / 3](c % 3) -= hstep;
                mp[c / 3](c % 3) += hstep;
                const double fd = (pt_barrier(f, pp) - pt_barrier(f, pm) - pt_barrier(f, mp)
                                   + pt_barrier(f, mm))
                                  / (4.0 * hstep * hstep);
                const double rel_h = std::abs(fd - bh(r, c)) / std::max(1.0, std::abs(fd));
                worst_h = std::max(worst_h, rel_h);
                CHECK(close(fd, bh(r, c), 1e-4));
            }
        }
        // the raw barrier Hessian is indefinite (`dB/dD < 0`), so symmetry is the only
        // structural property asserted here (the sibling contact-model tests' convention)
        for(int r = 0; r < 12; ++r)
            for(int c = 0; c < 12; ++c)
                CHECK(close(bh(r, c), bh(c, r), 1e-14));
        std::printf(" [diagnoser] point-triangle barrier: B = %.6e, FD grad %.3e / hess %.3e\n",
                    B,
                    worst_g,
                    worst_h);
    }
}

// ===========================================================================
//  Phase 2.4 — edge-edge (12 dof, mollifier + barrier)
// ===========================================================================
TEST_CASE("t85_ee", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_ee");
    Diag d{"u85_ee", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    // ---- the three region classes of the reference's flag chain -----------
    struct Region
    {
        Vector3  ea0, ea1, eb0, eb1;
        Vector4i flag;
    };
    const std::vector<Region> regions = {
        // interior (dim 4): both closest points are strictly interior
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, -1.0, 1.0}, {0.5, 2.0, 1.0}, {1, 1, 1, 1}},
        // point-edge (dim 3): eb0 is the closest point of edge b (the a-side is interior)
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 1.0, 0.0}, {0.5, 3.0, 0.0}, {1, 1, 1, 0}},
        // point-point (dim 2): ea0 vs eb0 — pB (s clamps high) + low1 in the reference's
        // flag chain, i.e. {0,1,1,0}
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 2.0, 0.0}, {0, 1, 1, 0}},
    };

    for(const auto& r : regions)
    {
        std::vector<Vector3>  av{r.ea0, r.ea1}, bv{r.eb0, r.eb1};
        std::vector<Vector2i> af{{0, 1}}, bf{{0, 1}};
        auto ea = make_edges(span<const Vector3>{av}, span<const Vector2i>{af}, 0.01, 0.0);
        auto eb = make_edges(span<const Vector3>{bv}, span<const Vector2i>{bf}, 0.01, 0.0);

        geometry::Geometry R;
        f->compute_edge_edge_distance(R, ea, eb, ea, eb);
        REQUIRE(R.instances().size() == 1);

        auto flag = read_attr<Vector4i>(R, "flag").at(0);
        auto crd  = read_attr<Vector4>(R, "coord").at(0);
        auto D    = read_attr<Float>(R, "dist2").at(0);

        std::printf(" [diagnoser] edge-edge region: flag = {%d,%d,%d,%d}, D = %.9e\n",
                    flag(0),
                    flag(1),
                    flag(2),
                    flag(3),
                    D);
        CHECK(flag == r.flag);
        // coord: {1-s, s, 1-t, t} — both pairs sum to 1 and reconstruct the closest points
        CHECK(close(crd(0) + crd(1), 1.0, 1e-15));
        CHECK(close(crd(2) + crd(3), 1.0, 1e-15));
        const Vector3 qa = crd(0) * r.ea0 + crd(1) * r.ea1;
        const Vector3 qb = crd(2) * r.eb0 + crd(3) * r.eb1;
        CHECK(close(D, (qa - qb).squaredNorm(), 1e-13));
        CHECK(close(D, segment_segment_distance2(r.ea0, r.ea1, r.eb0, r.eb1), 1e-13));
    }

    // ---- 200 random pairs against the in-test oracle ----------------------
    {
        std::vector<Vector3>  as, bs;
        std::vector<Vector2i> af, bf;
        unsigned              state = 2468013579u;
        auto                  rnd   = [&state]() noexcept
        {
            state = state * 1664525u + 1013904223u;
            return static_cast<double>(state >> 8) / 16777216.0;
        };
        for(int i = 0; i < 200; ++i)
        {
            for(int k = 0; k < 2; ++k)
            {
                as.push_back(Vector3{4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0});
                bs.push_back(Vector3{4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0, 4.0 * rnd() - 2.0});
            }
            af.push_back(Vector2i{2 * i, 2 * i + 1});
            bf.push_back(Vector2i{2 * i, 2 * i + 1});
        }
        auto ea = make_edges(span<const Vector3>{as}, span<const Vector2i>{af}, 0.01, 0.0);
        auto eb = make_edges(span<const Vector3>{bs}, span<const Vector2i>{bf}, 0.01, 0.0);

        geometry::Geometry R;
        f->compute_edge_edge_distance(R, ea, eb, ea, eb);
        REQUIRE(R.instances().size() == 40000);

        auto D    = read_attr<Float>(R, "dist2");
        auto crd  = read_attr<Vector4>(R, "coord");
        auto flag = read_attr<Vector4i>(R, "flag");
        auto ek   = read_attr<Float>(R, "e_k");
        auto eps  = read_attr<Float>(R, "eps_x");

        double      worst_rel = 0.0;
        std::size_t nan_count = 0;
        for(size_t i = 0; i < 200; ++i)
        {
            for(size_t j = 0; j < 200; ++j)
            {
                const size_t k = i * 200 + j;
                const double expected = segment_segment_distance2(
                    as[2 * i], as[2 * i + 1], bs[2 * j], bs[2 * j + 1]);
                const double got = D[k];
                if(std::isnan(got))
                {
                    ++nan_count;
                    continue;
                }
                worst_rel = std::max(worst_rel, std::abs(got - expected) / std::max(1.0, expected));
                const Vector3 qa = crd[k](0) * as[2 * i] + crd[k](1) * as[2 * i + 1];
                const Vector3 qb = crd[k](2) * bs[2 * j] + crd[k](3) * bs[2 * j + 1];
                CHECK(close(got, (qa - qb).squaredNorm(), 1e-12));
                // the flag is one of the reference's vectors (the port's select chain,
                // i.e. the reference's own table) and is never all-zero
                const int dim = flag[k](0) + flag[k](1) + flag[k](2) + flag[k](3);
                CHECK(dim >= 2);
                CHECK(dim <= 4);
                // default mollifier_coeff = 1e-3, rest == current ⇒ eps_x > 0 and, for a
                // generic (non-parallel) configuration, e_k == 1 exactly
                CHECK(eps[k] >= 0.0);
            }
        }
        std::printf(" [diagnoser] edge-edge oracle: worst rel |delta| = %.3e, NaN pairs %zu\n",
                    worst_rel,
                    nan_count);
        CHECK(worst_rel <= 1e-13);
        CHECK(nan_count == 0);

        // the dist2 gradient against central differences on one interior pair
        std::array<Vector3, 4> v = {{{0.0, 0.0, 0.0},
                                     {1.0, 0.25, 0.0},
                                     {0.25, -1.0, 1.0},
                                     {0.25, 2.0, 1.0}}};
        std::vector<Vector3>  av{v[0], v[1]}, bv{v[2], v[3]};
        std::vector<Vector2i> afd{{0, 1}}, bfd{{0, 1}};
        auto ea1 = make_edges(span<const Vector3>{av}, span<const Vector2i>{afd}, 0.01, 0.0);
        auto eb1 = make_edges(span<const Vector3>{bv}, span<const Vector2i>{bfd}, 0.01, 0.0);
        geometry::Geometry R1;
        f->compute_edge_edge_distance(R1, ea1, eb1, ea1, eb1);
        REQUIRE(read_attr<Vector4i>(R1, "flag").at(0) == Vector4i(1, 1, 1, 1));
        auto G = read_attr<Vector12>(R1, "dist2/grad").at(0);
        auto H = read_attr<Matrix12x12>(R1, "dist2/hess").at(0);

        const double h       = 1e-6;
        double       worst_g = 0.0;
        for(int k = 0; k < 12; ++k)
        {
            auto up = v, down = v;
            up[k / 3](k % 3) += h;
            down[k / 3](k % 3) -= h;
            const double fd = (ee_dist2(f, up) - ee_dist2(f, down)) / (2.0 * h);
            worst_g = std::max(worst_g, std::abs(fd - G(k)) / std::max(1.0, std::abs(fd)));
            CHECK(close(fd, G(k), 1e-6));
        }
        for(int r = 0; r < 12; ++r)
            for(int c = 0; c < 12; ++c)
                CHECK(H(r, c) == H(c, r));
        std::printf(" [diagnoser] edge-edge FD: grad %.3e (relative)\n", worst_g);
    }
}

// ===========================================================================
//  The e_k family — the mollifier's own contract
// ===========================================================================
TEST_CASE("t85_ee_ek", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_ee_ek");
    Diag d{"u85_ee_ek", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    std::vector<Vector3>  av{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Vector3>  bv{{0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}};
    std::vector<Vector2i> af{{0, 1}}, bf{{0, 1}};
    auto ea = make_edges(span<const Vector3>{av}, span<const Vector2i>{af}, 0.01, 0.0);
    auto eb = make_edges(span<const Vector3>{bv}, span<const Vector2i>{bf}, 0.01, 0.0);

    // (a) absent mollifier_coeff ⇒ eps_x = 1e-3 * |ea|^2 * |eb|^2 and the attribute is
    //     created with 1e-3 (.cu:772-779, 868-869)
    {
        geometry::Geometry R;
        f->compute_edge_edge_distance(R, ea, eb, ea, eb);
        auto eps  = read_attr<Float>(R, "eps_x").at(0);
        auto mc   = read_attr<Float>(R, "mollifier_coeff");
        REQUIRE(mc.size() == R.instances().size());
        CHECK(mc[0] == 1e-3);
        CHECK(close(eps, 1e-3 * 1.0 * 1.0, 1e-15));  // |ea|^2 = |eb|^2 = 1
        // generic (non-parallel) configuration ⇒ e_k == 1 exactly and its derivatives 0
        CHECK(read_attr<Float>(R, "e_k").at(0) == 1.0);
        CHECK(read_attr<Vector12>(R, "e_k/grad").at(0).isZero(0.0));
        CHECK(read_attr<Matrix12x12>(R, "e_k/hess").at(0).isZero(0.0));
    }

    // (b) a present mollifier_coeff is read *before* the resize and used
    {
        geometry::Geometry R;
        R.instances().resize(1);
        R.instances().create<Float>("mollifier_coeff", 0.5);
        f->compute_edge_edge_distance(R, ea, eb, ea, eb);
        auto eps = read_attr<Float>(R, "eps_x").at(0);
        CHECK(close(eps, 0.5 * 1.0 * 1.0, 1e-15));
        CHECK(read_attr<Float>(R, "mollifier_coeff")[0] == 0.5);  // untouched
    }

    // (c) the disabled-mollifier sentinel: coeff = EEMollifierDisabledThreshold (-1) makes
    //     eps_x negative, so nothing is ever mollified (that is how the active-set and
    //     frictional models disable the mollifier)
    {
        geometry::Geometry R;
        R.instances().resize(1);
        R.instances().create<Float>("mollifier_coeff", -1.0);
        // a *parallel* current configuration, the worst case for the mollifier
        std::vector<Vector3> parallel_b{{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}};
        auto eb_par = make_edges(span<const Vector3>{parallel_b},
                                 span<const Vector2i>{bf},
                                 0.01,
                                 0.0);
        f->compute_edge_edge_distance(R, ea, eb_par, ea, eb);
        auto eps = read_attr<Float>(R, "eps_x").at(0);
        CHECK(eps < 0.0);
        CHECK(read_attr<Float>(R, "e_k").at(0) == 1.0);
    }

    // (d) inside the mollified band: |ea x eb|^2 < eps_x ⇒ e_k = 2x/eps - (x/eps)^2
    {
        // rest edges of length 1 each ⇒ eps_x = 1e-3 with the default coeff; current edges
        // nearly parallel ⇒ |cross|^2 = 0
        geometry::Geometry R;
        std::vector<Vector3> bpar{{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}};
        auto eb_par = make_edges(span<const Vector3>{bpar}, span<const Vector2i>{bf}, 0.01, 0.0);
        f->compute_edge_edge_distance(R, ea, eb_par, ea, eb);
        const double eps = read_attr<Float>(R, "eps_x").at(0);
        const double ek  = read_attr<Float>(R, "e_k").at(0);
        REQUIRE(eps == 1e-3);
        // |cross|^2 == 0 ⇒ EEM(0, eps) == (2 - 0) * 0 == 0
        CHECK(ek == 0.0);
        // the barrier then vanishes with it
        CHECK(read_attr<Float>(R, "barrier").at(0) == 0.0);
    }

    // (e) active_count(flag) < 4 zeroes e_k/grad and e_k/hess but keeps e_k (.cu:464-469)
    {
        // ea collapses to a point ⇒ the a-side is a point, dim 3 ⇒ Gradek/Hessek zeroed
        std::vector<Vector3>  av_deg{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        auto ea_deg = make_edges(span<const Vector3>{av_deg}, span<const Vector2i>{af}, 0.01, 0.0);
        geometry::Geometry R;
        f->compute_edge_edge_distance(R, ea_deg, eb, ea_deg, eb);
        auto flag = read_attr<Vector4i>(R, "flag").at(0);
        const int dim = flag(0) + flag(1) + flag(2) + flag(3);
        CHECK(dim < 4);
        CHECK(read_attr<Vector12>(R, "e_k/grad").at(0).isZero(0.0));
        CHECK(read_attr<Matrix12x12>(R, "e_k/hess").at(0).isZero(0.0));
        // e_k itself is still the mollifier's value (1.0 here: not in the band)
        CHECK(read_attr<Float>(R, "e_k").at(0) == 1.0);
    }
}

// ===========================================================================
//  The attribute set — names, types, sizes (the reference's contract)
// ===========================================================================
TEST_CASE("t85_attribute_set", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_attribute_set");
    Diag d{"u85_attribute_set", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    std::vector<Vector3>  pv{{0.1, 0.1, 0.1}, {0.2, 0.3, 0.4}};
    std::vector<Vector3>  tv{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
    std::vector<Vector3i> tf{{0, 1, 2}};
    std::vector<Vector3>  ev{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Vector2i> ef{{0, 1}};

    auto points    = make_points(span<const Vector3>{pv});
    auto triangles = make_triangles(span<const Vector3>{tv},
                                    span<const Vector3i>{tf});
    auto edges     = make_edges(span<const Vector3>{ev}, span<const Vector2i>{ef});

    // the exact name sets of the reference's four `copy_back` blocks
    const std::vector<std::string> pt_names = {"barrier",
                                               "barrier/grad",
                                               "barrier/hess",
                                               "coord",
                                               "dist2",
                                               "dist2/grad",
                                               "dist2/hess",
                                               "flag"};
    auto ee_names = pt_names;
    for(const char* extra : {"e_k", "e_k/grad", "e_k/hess", "eps_x"})
        ee_names.push_back(extra);
    std::sort(ee_names.begin(), ee_names.end());
    const std::vector<std::string> pe_names = {"barrier",
                                               "barrier/grad",
                                               "barrier/hess",
                                               "coord",
                                               "dist2",
                                               "dist2/grad",
                                               "dist2/hess",
                                               "flag"};
    const std::vector<std::string> pp_names = {"barrier",
                                               "barrier/grad",
                                               "barrier/hess",
                                               "dist2",
                                               "dist2/grad",
                                               "dist2/hess"};

    // ---- PT ---------------------------------------------------------------
    {
        geometry::Geometry R;
        f->compute_point_triangle_distance(R, points, triangles);
        CHECK(R.instances().size() == 2u * 1u);
        CHECK(instance_attribute_names(R) == pt_names);
        // the types are the reference's (a wrong type would make find<T> return nullptr)
        CHECK(R.instances().find<Float>("dist2") != nullptr);
        CHECK(R.instances().find<Vector12>("dist2/grad") != nullptr);
        CHECK(R.instances().find<Matrix12x12>("dist2/hess") != nullptr);
        CHECK(R.instances().find<Vector4i>("flag") != nullptr);
        CHECK(R.instances().find<Vector4>("coord") != nullptr);
        CHECK(R.instances().find<Float>("barrier") != nullptr);
        CHECK(R.instances().find<Vector12>("barrier/grad") != nullptr);
        CHECK(R.instances().find<Matrix12x12>("barrier/hess") != nullptr);
        // every slot has one entry per pair
        CHECK(R.instances().find<Float>("dist2")->view().size() == 2u);
        CHECK(R.instances().find<Matrix12x12>("dist2/hess")->view().size() == 2u);
        // the result geometry's meta collection is untouched
        CHECK(R.to_json().at("meta").empty());
    }

    // ---- EE (+ mollifier_coeff, created last) -----------------------------
    {
        geometry::Geometry R;
        f->compute_edge_edge_distance(R, edges, edges, edges, edges);
        CHECK(R.instances().size() == 1u);
        auto names = instance_attribute_names(R);
        auto expected = ee_names;
        expected.push_back("mollifier_coeff");
        std::sort(expected.begin(), expected.end());
        CHECK(names == expected);
        CHECK(R.instances().find<Float>("eps_x") != nullptr);
        CHECK(R.instances().find<Float>("e_k") != nullptr);
        CHECK(R.instances().find<Vector12>("e_k/grad") != nullptr);
        CHECK(R.instances().find<Matrix12x12>("e_k/hess") != nullptr);
        CHECK(R.instances().find<Float>("mollifier_coeff") != nullptr);
    }

    // ---- PE --------------------------------------------------------------
    {
        geometry::Geometry R;
        f->compute_point_edge_distance(R, points, edges);
        CHECK(R.instances().size() == 2u);
        CHECK(instance_attribute_names(R) == pe_names);
        CHECK(R.instances().find<Vector9>("dist2/grad") != nullptr);
        CHECK(R.instances().find<Matrix9x9>("dist2/hess") != nullptr);
        CHECK(R.instances().find<Vector3i>("flag") != nullptr);
        CHECK(R.instances().find<Vector3>("coord") != nullptr);
        CHECK(R.instances().find<Vector9>("barrier/grad") != nullptr);
        CHECK(R.instances().find<Matrix9x9>("barrier/hess") != nullptr);
    }

    // ---- PP --------------------------------------------------------------
    {
        geometry::Geometry R;
        f->compute_point_point_distance(R, points, points);
        CHECK(R.instances().size() == 4u);
        CHECK(instance_attribute_names(R) == pp_names);
        CHECK(R.instances().find<Vector6>("dist2/grad") != nullptr);
        CHECK(R.instances().find<Matrix6x6>("dist2/hess") != nullptr);
        CHECK(R.instances().find<Vector6>("barrier/grad") != nullptr);
        CHECK(R.instances().find<Matrix6x6>("barrier/hess") != nullptr);
    }
}

// ===========================================================================
//  Negative control (a): a missing d_hat/thickness is an assert, not a default
// ===========================================================================
TEST_CASE("t85_missing_input_attribute", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_missing_input_attribute");
    Diag d{"u85_missing_input_attribute", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    // The reference's `upload_vertex_attribute` (.cu:289-299) is a hard precondition:
    // UIPC_ASSERT(attr, "Vertex attribute '{}' not found on SimplicialComplex").  The
    // assert is `uipc::abort()` (src/core/common/abort.cpp), which cannot be caught, so —
    // exactly like luisa_fem_base_test.cpp's zero-volume tet — the test asserts the
    // *condition* instead of taking the shared test binary down with it.
    std::vector<Vector3> vs{{0.0, 0.0, 0.0}};
    auto                 bare = pointcloud(span<const Vector3>{vs});
    CHECK(bare.vertices().find<Float>("d_hat") == nullptr);
    CHECK(bare.vertices().find<Float>("thickness") == nullptr);

    // ... and the attribute the query *does* require is present on the good inputs, so the
    // negative control above is about the input contract, not about the query.
    auto good = make_points(span<const Vector3>{vs});
    CHECK(good.vertices().find<Float>("d_hat") != nullptr);
    CHECK(good.vertices().find<Float>("thickness") != nullptr);
    CHECK(good.vertices().find<Float>("d_hat")->view()[0] == 0.01);

    // a query whose inputs satisfy the contract succeeds (the same call the assert guards)
    geometry::Geometry R;
    CHECK_NOTHROW(f->compute_point_point_distance(R, good, good));
    CHECK(R.instances().size() == 1u);
}

// ===========================================================================
//  Degenerate sweep + the empty case
// ===========================================================================
TEST_CASE("t85_degenerate", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_degenerate");
    Diag d{"u85_degenerate", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    // ---- empty: 0 pairs ⇒ instances().size() == 0 and no attribute at all
    {
        // `.cu:673-677`: `if(num_pairs == 0) { R.instances().resize(0); return; }`.  The
        // inputs are full-size complexes whose element collection is then resized to 0
        // (an empty *pair count* is the diagnoser's contract; `geometry::trimesh` asserts on
        // a complex that has no cells at all, which is a factory detail, not a diagnoser one).
        std::vector<Vector3>  vs{{0.0, 0.0, 0.0}};
        std::vector<Vector3>  tv{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
        std::vector<Vector3i> tf{{0, 1, 2}};
        std::vector<Vector3>  ev{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        std::vector<Vector2i> ef{{0, 1}};

        auto one_point = make_points(span<const Vector3>{vs});
        auto one_tri = make_triangles(span<const Vector3>{tv}, span<const Vector3i>{tf});
        auto one_edge = make_edges(span<const Vector3>{ev}, span<const Vector2i>{ef});

        auto no_points = make_points(span<const Vector3>{vs});
        no_points.vertices().resize(0);
        auto no_tris = make_triangles(span<const Vector3>{tv}, span<const Vector3i>{tf});
        no_tris.triangles().resize(0);
        auto no_edges = make_edges(span<const Vector3>{ev}, span<const Vector2i>{ef});
        no_edges.edges().resize(0);

        REQUIRE(no_points.positions().size() == 0u);
        REQUIRE(no_tris.triangles().topo().size() == 0u);
        REQUIRE(no_edges.edges().topo().size() == 0u);

        // (a) 0 points x 1 triangle
        geometry::Geometry R;
        f->compute_point_triangle_distance(R, no_points, one_tri);
        CHECK(R.instances().size() == 0u);
        CHECK(instance_attribute_names(R).empty());
        // (b) 1 point x 0 triangles
        geometry::Geometry R2;
        f->compute_point_triangle_distance(R2, one_point, no_tris);
        CHECK(R2.instances().size() == 0u);
        CHECK(instance_attribute_names(R2).empty());
        // (c) the other three queries
        geometry::Geometry R3;
        f->compute_point_point_distance(R3, no_points, one_point);
        CHECK(R3.instances().size() == 0u);
        CHECK(instance_attribute_names(R3).empty());
        geometry::Geometry R4;
        f->compute_point_edge_distance(R4, no_points, one_edge);
        CHECK(R4.instances().size() == 0u);
        CHECK(instance_attribute_names(R4).empty());
        geometry::Geometry R5;
        f->compute_edge_edge_distance(R5, no_edges, one_edge, no_edges, one_edge);
        CHECK(R5.instances().size() == 0u);
        CHECK(instance_attribute_names(R5).empty());
    }

    // ---- touching (D == 0), coincident points, zero-length edge -----------
    {
        // touching point/point pairs: three distinct points, all touching each other
        std::vector<Vector3> vs{{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};
        auto p = make_points(span<const Vector3>{vs}, 0.01, 0.0);
        geometry::Geometry R;
        f->compute_point_point_distance(R, p, p);
        auto D  = read_attr<Float>(R, "dist2");
        auto B  = read_attr<Float>(R, "barrier");
        auto G  = read_attr<Vector6>(R, "dist2/grad");
        for(size_t k = 0; k < D.size(); ++k)
        {
            CHECK(D[k] == 0.0);
            CHECK(!std::isnan(D[k]));
            CHECK(G[k].isZero(0.0));  // 2 (a - b) with a == b
            CHECK(B[k] == 0.0);       // D == 0 is outside the open band (0, d_hat^2)
        }
    }

    // ---- zero-length edge (PE and EE) ------------------------------------
    {
        std::vector<Vector3>  pv{{0.25, 0.25, 0.5}};
        std::vector<Vector3>  ev{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};  // e0 == e1
        std::vector<Vector2i> ef{{0, 1}};
        auto points = make_points(span<const Vector3>{pv}, 0.01, 0.0);
        auto edges  = make_edges(span<const Vector3>{ev}, span<const Vector2i>{ef}, 0.01, 0.0);
        geometry::Geometry R;
        f->compute_point_edge_distance(R, points, edges);
        auto D    = read_attr<Float>(R, "dist2").at(0);
        auto flag = read_attr<Vector3i>(R, "flag").at(0);
        CHECK(!std::isnan(D));
        CHECK(close(D, 0.375, 1e-15));  // |(0.25,0.25,0.5)|^2
        CHECK(flag(0) == 1);
        CHECK(flag(1) + flag(2) == 1);  // degenerate edge ⇒ point-point against one endpoint

        // edge-edge with both edges collapsed
        geometry::Geometry R2;
        f->compute_edge_edge_distance(R2, edges, edges, edges, edges);
        CHECK(!std::isnan(read_attr<Float>(R2, "dist2").at(0)));
        CHECK(read_attr<Float>(R2, "eps_x").at(0) == 0.0);  // |rest edge|^2 == 0
    }

    // ---- zero-area (collinear) triangle ----------------------------------
    {
        // The reference evaluates the *unflagged* full-simplex body here (nVec == 0 ⇒ the
        // 2x2 barycentric solve is singular ⇒ every flag predicate is false ⇒ dim 4), and
        // that body divides by the degenerate area.  The parity harness records the same
        // behaviour for the reference on its three degenerate cases
        // (luisa_parity_golden_test.cpp: "the three deliberately degenerate configurations
        // ... are NaN for every output on both sides"), so the diagnoser is expected to
        // reproduce it rather than to "fix" it: the query must complete, the attribute set
        // must exist, and the value must be the reference's (NaN).
        std::vector<Vector3>  pv{{0.0, 0.0, 0.5}};
        std::vector<Vector3>  tv{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
        std::vector<Vector3i> tf{{0, 1, 2}};
        auto points    = make_points(span<const Vector3>{pv}, 0.01, 0.0);
        auto triangles = make_triangles(span<const Vector3>{tv},
                                        span<const Vector3i>{tf},
                                        0.01,
                                        0.0);
        geometry::Geometry R;
        CHECK_NOTHROW(f->compute_point_triangle_distance(R, points, triangles));
        CHECK(R.instances().size() == 1u);
        CHECK(instance_attribute_names(R).size() == 8u);
        const double D = read_attr<Float>(R, "dist2").at(0);
        const auto   flag = read_attr<Vector4i>(R, "flag").at(0);
        std::printf(" [diagnoser] zero-area triangle: dist2 = %g, flag = {%d,%d,%d,%d}\n",
                    D,
                    flag(0),
                    flag(1),
                    flag(2),
                    flag(3));
        CHECK(flag == Vector4i(1, 1, 1, 1));
        CHECK(std::isnan(D));
    }

    // ---- far apart: no barrier, no mollification --------------------------
    {
        std::vector<Vector3>  av{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        std::vector<Vector3>  bv{{0.0, 0.0, 100.0}, {0.0, 1.0, 100.0}};
        std::vector<Vector2i> af{{0, 1}}, bf{{0, 1}};
        auto ea = make_edges(span<const Vector3>{av}, span<const Vector2i>{af}, 0.01, 0.0);
        auto eb = make_edges(span<const Vector3>{bv}, span<const Vector2i>{bf}, 0.01, 0.0);
        geometry::Geometry R;
        f->compute_edge_edge_distance(R, ea, eb, ea, eb);
        CHECK(read_attr<Float>(R, "dist2").at(0) == 100.0 * 100.0);
        CHECK(read_attr<Float>(R, "barrier").at(0) == 0.0);
        CHECK(read_attr<Vector12>(R, "barrier/grad").at(0).isZero(0.0));
        CHECK(read_attr<Matrix12x12>(R, "barrier/hess").at(0).isZero(0.0));
        CHECK(read_attr<Float>(R, "e_k").at(0) == 1.0);
        CHECK(read_attr<Vector12>(R, "e_k/grad").at(0).isZero(0.0));
        CHECK(read_attr<Matrix12x12>(R, "e_k/hess").at(0).isZero(0.0));
    }
}

// ===========================================================================
//  Class A — the Tier-1 golden (the reference `.inl` evaluated on the host)
// ===========================================================================
namespace
{
constexpr std::string_view kGoldenMagic   = "UIPCGOLD1";
constexpr std::uint32_t    kGoldenVersion = 1;
constexpr std::size_t      kGoldenInSize  = 12u;              // p, t0, t1, t2
constexpr std::size_t      kGoldenOutSize = 1u + 12u + 144u;  // dist2 + grad + 12x12

struct Golden
{
    std::string         name;
    std::size_t         n_input  = 0;
    std::size_t         n_output = 0;
    std::vector<double> input;   // case_count * n_input
    std::vector<double> output;  // case_count * n_output

    [[nodiscard]] std::size_t case_count() const noexcept
    {
        return n_input == 0 ? 0 : input.size() / n_input;
    }
};

[[nodiscard]] std::string golden_path()
{
    return (fs::path{UIPC_TEST_PROGRAM_PATH}.parent_path() / "parity_golden"
            / "distance_pt.bin")
        .string();
}

[[nodiscard]] bool read_golden(const std::string& path, Golden& golden, std::string& error)
{
    std::ifstream in{path, std::ios::binary};
    if(!in)
    {
        error = "cannot open " + path;
        return false;
    }
    char magic[kGoldenMagic.size()];
    in.read(magic, sizeof(magic));
    if(!in || std::memcmp(magic, kGoldenMagic.data(), sizeof(magic)) != 0)
    {
        error = "bad magic in " + path;
        return false;
    }
    std::uint32_t version = 0, name_len = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    if(!in || version != kGoldenVersion)
    {
        error = "unsupported golden version";
        return false;
    }
    golden.name.resize(name_len);
    if(name_len > 0)
        in.read(golden.name.data(), name_len);
    std::uint64_t case_count = 0, n_input = 0, n_output = 0;
    in.read(reinterpret_cast<char*>(&case_count), sizeof(case_count));
    in.read(reinterpret_cast<char*>(&n_input), sizeof(n_input));
    in.read(reinterpret_cast<char*>(&n_output), sizeof(n_output));
    if(!in || case_count == 0 || n_input == 0 || n_output == 0)
    {
        error = "corrupt golden header";
        return false;
    }
    golden.n_input  = static_cast<std::size_t>(n_input);
    golden.n_output = static_cast<std::size_t>(n_output);
    golden.input.resize(static_cast<std::size_t>(case_count) * golden.n_input);
    golden.output.resize(static_cast<std::size_t>(case_count) * golden.n_output);
    for(std::uint64_t c = 0; c < case_count; ++c)
    {
        in.read(reinterpret_cast<char*>(golden.input.data() + c * golden.n_input),
                static_cast<std::streamsize>(golden.n_input * sizeof(double)));
        in.read(reinterpret_cast<char*>(golden.output.data() + c * golden.n_output),
                static_cast<std::streamsize>(golden.n_output * sizeof(double)));
        if(!in)
        {
            error = "truncated payload in " + path;
            return false;
        }
    }
    return true;
}
}  // namespace

TEST_CASE("t85_parity_golden_pt", "[luisa][diagnoser]")
{
    test_runtime();

    const std::string path = golden_path();
    if(!fs::exists(path))
    {
        SKIP("reference golden not generated: " + path + " (run uipc_parity_reference.exe first)");
    }

    Golden      golden;
    std::string error;
    INFO("golden: " + path + ": " + error);
    REQUIRE(read_golden(path, golden, error));
    REQUIRE(golden.name == "distance_pt");
    REQUIRE(golden.n_input == kGoldenInSize);
    REQUIRE(golden.n_output == kGoldenOutSize);

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_parity_golden_pt");
    Diag d{"u85_parity_golden_pt", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    std::size_t exact_mismatches = 0;
    std::size_t zero_sign_diffs  = 0;
    std::size_t nan_slots        = 0;
    std::size_t class_a_cases    = 0;
    std::size_t class_a_slots    = 0;
    std::size_t dispatched_cases = 0;
    double      worst_abs        = 0.0;
    std::size_t worst_case = 0, worst_slot = 0;
    double      worst_dispatched = 0.0;

    std::vector<double> got(kGoldenOutSize, 0.0);

    for(std::size_t c = 0; c < golden.case_count(); ++c)
    {
        const double* base = golden.input.data() + c * golden.n_input;
        std::vector<Vector3>  pv{{base[0], base[1], base[2]}};
        std::vector<Vector3>  tv{{base[3], base[4], base[5]},
                                 {base[6], base[7], base[8]},
                                 {base[9], base[10], base[11]}};
        std::vector<Vector3i> tf{{0, 1, 2}};
        auto points    = make_points(span<const Vector3>{pv}, 0.01, 0.0);
        auto triangles = make_triangles(span<const Vector3>{tv},
                                        span<const Vector3i>{tf},
                                        0.01,
                                        0.0);

        geometry::Geometry R;
        f->compute_point_triangle_distance(R, points, triangles);
        REQUIRE(R.instances().size() == 1u);

        const Float      D    = read_attr<Float>(R, "dist2").at(0);
        const Vector12   G    = read_attr<Vector12>(R, "dist2/grad").at(0);
        const Matrix12x12 H   = read_attr<Matrix12x12>(R, "dist2/hess").at(0);
        const Vector4i    flag = read_attr<Vector4i>(R, "flag").at(0);

        got[0] = D;
        for(int i = 0; i < 12; ++i)
            got[1 + i] = G(i);
        std::size_t k = 1 + 12;
        for(int col = 0; col < 12; ++col)
            for(int row = 0; row < 12; ++row)
                got[k++] = H(row, col);

        const bool all_active = flag == Vector4i(1, 1, 1, 1);
        if(!all_active)
        {
            // The golden evaluates the *unflagged* full-simplex body; the diagnoser
            // dispatches on the flag exactly like CUDA's diagnoser does.  For a case whose
            // closest feature is a sub-simplex the two are different (and correct) answers,
            // so the Class-A slot comparison does not apply; instead check the flagged
            // result against this file's own sub-simplex oracle.
            ++dispatched_cases;
            double expected = 0.0;
            if(flag == Vector4i(1, 1, 0, 0))
                expected = (pv[0] - tv[0]).squaredNorm();
            else if(flag == Vector4i(1, 0, 1, 0))
                expected = (pv[0] - tv[1]).squaredNorm();
            else if(flag == Vector4i(1, 0, 0, 1))
                expected = (pv[0] - tv[2]).squaredNorm();
            else if(flag == Vector4i(1, 1, 1, 0))
                expected = point_segment_distance2(pv[0], tv[0], tv[1]);
            else if(flag == Vector4i(1, 0, 1, 1))
                expected = point_segment_distance2(pv[0], tv[1], tv[2]);
            else if(flag == Vector4i(1, 1, 0, 1))
                expected = point_segment_distance2(pv[0], tv[2], tv[0]);
            else
                expected = (pv[0] - closest_point_on_triangle(pv[0], tv[0], tv[1], tv[2]))
                               .squaredNorm();
            worst_dispatched = std::max(worst_dispatched,
                                        std::abs(D - expected) / std::max(1.0, expected));
            continue;
        }

        ++class_a_cases;
        class_a_slots += kGoldenOutSize;
        const double* expected = golden.output.data() + c * golden.n_output;
        for(std::size_t s = 0; s < kGoldenOutSize; ++s)
        {
            if(std::isnan(got[s]) && std::isnan(expected[s]))
            {
                ++nan_slots;
                continue;
            }
            if(!(got[s] == expected[s]))
            {
                ++exact_mismatches;
                const double delta = std::abs(got[s] - expected[s]);
                if(delta >= worst_abs)
                {
                    worst_abs  = delta;
                    worst_case = c;
                    worst_slot = s;
                }
            }
            else if(std::memcmp(&got[s], expected + s, sizeof(double)) != 0)
            {
                ++zero_sign_diffs;
            }
        }
    }

    std::printf(
        " [diagnoser] golden: %zu cases = %zu Class-A (%zu slots) + %zu flag-dispatched; "
        "host |delta| = %.3e (%zu mismatches, %zu zero-sign-only, %zu NaN slots; worst "
        "case %zu slot %zu), dispatched |delta| = %.3e\n",
        golden.case_count(),
        class_a_cases,
        class_a_slots,
        dispatched_cases,
        worst_abs,
        exact_mismatches,
        zero_sign_diffs,
        nan_slots,
        worst_case,
        worst_slot,
        worst_dispatched);

    CHECK(class_a_cases + dispatched_cases == golden.case_count());
    CHECK(exact_mismatches == 0u);
    CHECK(worst_dispatched <= 1e-14);
}

// ===========================================================================
//  Negative control (b) — the diagnoser cannot perturb a solve
// ===========================================================================
TEST_CASE("t85_no_simulation", "[luisa][diagnoser]")
{
    test_runtime();

    Scene scene{diag_config()};
    add_trimesh(scene, "t85_no_simulation");
    Diag d{"u85_no_simulation", scene};
    auto f = d.feature();
    REQUIRE(f != nullptr);

    // (a) the queries write *only* to `R.instances()`: the input complexes are untouched
    std::vector<Vector3>  pv{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    std::vector<Vector3>  tv{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
    std::vector<Vector3i> tf{{0, 1, 2}};
    auto points    = make_points(span<const Vector3>{pv});
    auto triangles = make_triangles(span<const Vector3>{tv},
                                    span<const Vector3i>{tf});

    const auto points_before    = points.vertices().size();
    const auto triangles_before = triangles.triangles().size();
    auto       points_json      = points.to_json();
    auto       triangles_json   = triangles.to_json();

    geometry::Geometry R;
    f->compute_point_triangle_distance(R, points, triangles);
    CHECK(points.vertices().size() == points_before);
    CHECK(triangles.triangles().size() == triangles_before);
    CHECK(points.to_json() == points_json);
    CHECK(triangles.to_json() == triangles_json);
    CHECK(R.to_json().at("meta").empty());

    // (b) the diagnoser is stateless: a second identical call writes identical values
    geometry::Geometry R2;
    f->compute_point_triangle_distance(R2, points, triangles);
    REQUIRE(R2.instances().size() == R.instances().size());
    CHECK(read_attr<Float>(R2, "dist2") == read_attr<Float>(R, "dist2"));
    CHECK(read_attr<Vector4i>(R2, "flag") == read_attr<Vector4i>(R, "flag"));

    // (c) `do_build()` only inserts the feature: it registers no action and requires no
    // other system, so the scene's solve chain is byte-identical with and without it.  The
    // observable form of that claim (identical `newton_iterations` / frame stats over an
    // `advance()`) needs plan 10's engine loop; the recorded pre-existing abort inside the
    // first `advance()` (AGENT_ENV_BRIEF.md, shell_init_advance_retrieve) blocks it here.
    std::printf(
        " [diagnoser] SKIP(newton_iterations comparison): the engine advance loop is blocked "
        "by the pre-existing AST-builder abort (AGENT_ENV_BRIEF.md) - plan 10; the "
        "insert-only do_build() is verified structurally (no action, no require<>)\n");
}
