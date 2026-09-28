/********************************************************************
 * @file luisa_sanity_and_dytopo_test.cpp
 * @brief T4.3 — kernel_cout device print, the half-plane vertex distance
 *        sanity checker and the dytopo effect system (LuisaCompute backend).
 *
 * Coverage map (brief §4):
 *   1) kernel_cout: device print captured on sync (values of several types, the
 *      exact expected text/order, a 0-line case and a single-line case)
 *      ........................................................ t43_kernel_cout_capture
 *   2) half_plane_vertex_distance_check driven on a hand-built scene with a known
 *      violation (and no report for a clean scene / no half plane / no vertices /
 *      contact disabled) ................................. t43_half_plane_vertex_distance_check
 *   3) dytopo classify info (host) .......................... t43_dytopo_classify_info
 *      dytopo reporter/receiver round trip through the ported manager, both the
 *      single-diagonal-receiver fast path and the sorted/distributed path, plus
 *      the empty-set case ................................. t43_dytopo_round_trip
 *
 * Every expectation here is computed independently: hand-written text, hand-built
 * geometry, hand-computed sorted/deduplicated effects, or values this file fed in.
 * Nothing comes from the CUDA backend.
 *
 * Two shared-binary constraints (cookbook §7.14 rule 6): this file's systems are
 * built for *every* scene of the test binary, so (a) the test vertex reporter only
 * reports geometries carrying this file's marker meta attribute, and (b) the dytopo
 * reporter/receiver do nothing unless the scene config carries this file's mode key
 * — and the dytopo manager is only ever *driven* from this file (the engine shell
 * does not call compute_dytopo_effect() yet; that is the Phase-8 Newton loop).
 *********************************************************************/
// The kernel-print path is compiled out unless the *defining* translation unit opts
// in (luisa_device/log.h). Must come before any header that includes it.
#define UIPC_LUISA_KERNEL_PRINT 1

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>

#include <type_define.h>
#include <kernel_cout.h>
#include <frame_stream.h>
#include <energy_component_flags.h>

#include <dytopo_effect_system/dytopo_classify_info.h>
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/vertex_reporter.h>
#include <line_search/line_searcher.h>

// The test target's include path contains src/backends/none before src/backends/luisa,
// so the bare <sim_engine.h> would silently resolve to the none backend's shell
// (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;

namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;

namespace
{
    std::string t43_workspace(std::string_view name)
    {
        auto p = std::filesystem::temp_directory_path() / name;
        std::filesystem::create_directories(p);
        return p.string() + "/";
    }
}// namespace

// ===========================================================================
//  1) kernel_cout — the device print captured on sync
// ===========================================================================
namespace
{
    /// A dedicated LC stream, so the capture cannot observe any other test's
    /// device prints (the module's own sources are compiled with the print macro
    /// off, but a dedicated stream makes the test independent of that too).
    lud::LStream& kernel_cout_stream()
    {
        // Leaked on purpose (cookbook §7.10 trap 1): destroying an LC resource after
        // Runtime::shutdown() aborts the process (0xC0000409).
        static auto* lc   = new lcrt::Stream{lud::device().create_stream()};
        static auto* wrap = new lud::LStream{*lc};
        return *wrap;
    }

    /// Three print items, several argument types, in one kernel (one thread):
    /// the reference's `cout << "..." << x` streaming is one LC_LOG per item.
    auto& cout_multi_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>>(
            [](lcrt::BufferVar<std::int32_t> src) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i == 0u)
                {
                    auto v = src->read(0u);  // a genuine device read
                    LC_LOG("int={} uint={}", v, 7u);
                    LC_LOG("double={}", 0.5);
                    LC_LOG("mixed {} {} {}", -3, 42u, 2.25);
                };
            });
        return k;
    }

    /// A single print item.
    auto& cout_single_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>>(
            [](lcrt::BufferVar<std::int32_t> src) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i == 0u) { LC_LOG("only {}", src->read(0u)); };
            });
        return k;
    }

    /// No print item at all (the "0 lines" case).
    auto& cout_silent_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>>(
            [](lcrt::BufferVar<std::int32_t> src) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i == 0u) { src->write(0u, 1); };
            });
        return k;
    }

    /// Prints without an active capture: the text must reach the process log
    /// (LC's own `[DEVICE] …` path, which the port's trampoline reproduces).
    auto& cout_log_fallback_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>>(
            [](lcrt::BufferVar<std::int32_t> src) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i == 0u) { LC_LOG("t43_no_capture {}", src->read(0u)); };
            });
        return k;
    }

    void upload_one(lud::LStream& stream, lud::LDeviceBuffer<std::int32_t>& buf, std::int32_t v)
    {
        buf.resize(1u);
        stream.stream() << buf.view().native().copy_from(lcrt::span<const std::int32_t>{&v, 1u});
    }
}// namespace

TEST_CASE("t43_kernel_cout_capture", "[luisa][t43]")
{
    test_runtime();

    auto& stream = kernel_cout_stream();

    lud::LDeviceBuffer<std::int32_t> src;
    upload_one(stream, src, -7);

    // ---- three items, several types, in dispatch order --------------------
    {
        gg::KernelCoutCapture capture{stream};
        stream.record(cout_multi_kernel()(src.view()).dispatch(1u));
        stream.synchronize();

        REQUIRE(capture.line_count() == 3u);
        CHECK(capture.lines()[0] == "int=-7 uint=7");
        CHECK(capture.lines()[1] == "double=0.5");
        CHECK(capture.lines()[2] == "mixed -3 42 2.25");
    }

    // ---- a single item ----------------------------------------------------
    {
        gg::KernelCoutCapture capture{stream};
        stream.record(cout_single_kernel()(src.view()).dispatch(1u));
        stream.synchronize();

        REQUIRE(capture.line_count() == 1u);
        CHECK(capture.lines()[0] == "only -7");
    }

    // ---- 0 lines ----------------------------------------------------------
    {
        gg::KernelCoutCapture capture{stream};
        stream.record(cout_silent_kernel()(src.view()).dispatch(1u));
        stream.synchronize();

        CHECK(capture.line_count() == 0u);

        // the capture is per stream and installed by the RAII guard; after it the
        // default path (the process log) is restored, which must not throw.
        std::int32_t host = 0;
        stream.stream() << src.view().native().copy_to(lcrt::span<std::int32_t>{&host, 1u});
        stream.synchronize();
        CHECK(host == 1);
    }

    // The captured text is the *only* thing the two RAII guard scopes above ever
    // asserted; a second capture on the same stream must start empty.
    {
        gg::KernelCoutCapture capture{stream};
        CHECK(capture.line_count() == 0u);
    }

    // ---- with no capture installed the print goes to the process log ---------
    // (LC's own path, `[DEVICE] <text>`). The run output is grepped for the line
    // (see the validation evidence); the CHECKs below assert that the fallback
    // keeps dispatch and the following transfers working.
    {
        stream.record(cout_log_fallback_kernel()(src.view()).dispatch(1u));
        stream.synchronize();

        std::int32_t after = -1;
        upload_one(stream, src, 123);
        stream.stream() << src.view().native().copy_to(lcrt::span<std::int32_t>{&after, 1u});
        stream.synchronize();
        CHECK(after == 123);
    }
}

// ===========================================================================
//  2) half_plane_vertex_distance_check
// ===========================================================================
namespace
{
    /// One mesh + one half plane; `y` places every mesh vertex at that height, so
    /// the half plane's signed distance is `y - thickness` (the plane is
    /// {P = 0, N = +Y} and the checker's test is d <= 0).
    struct SanityIds
    {
        IndexT      mesh_geo_id = -1;
        IndexT      hp_geo_id   = -1;
        IndexT      mesh_obj_id = -1;
        IndexT      hp_obj_id   = -1;
        std::string mesh_name;
        std::string hp_name;
        // the mesh's own topology, for the expected close-mesh counts
        std::size_t close_vertices = 0;
        std::size_t close_edges    = 0;
        std::size_t close_tris     = 0;
    };

    struct BuiltSanityScene
    {
        Scene     scene;
        SanityIds ids;
    };

    /// A unit square (4 vertices, 2 triangles) at height `y`, plus a half plane
    /// at the origin with normal +Y. `sanity_check/enable` is on, `mode` quiet.
    BuiltSanityScene build_half_plane_scene(Float y, bool with_half_plane, bool contact_enabled)
    {
        auto config                   = Scene::default_config();
        config["sanity_check"]["mode"]   = "quiet";
        config["sanity_check"]["enable"] = true;
        config["contact"]["enable"]      = contact_enabled;
        Scene scene{config};

        BuiltSanityScene built{std::move(scene)};
        built.ids.mesh_name = "t43_mesh";
        built.ids.hp_name   = "t43_halfplane";

        const std::vector<Vector3> ps = {{0.0, y, 0.0},
                                         {1.0, y, 0.0},
                                         {1.0, y, 1.0},
                                         {0.0, y, 1.0}};
        const std::vector<Vector3i> ts = {{0, 1, 2}, {0, 2, 3}};
        auto mesh = trimesh(span<const Vector3>{ps}, span<const Vector3i>{ts});
        label_surface(mesh);

        auto mesh_object = built.scene.objects().create(built.ids.mesh_name);
        auto mesh_slots  = mesh_object->geometries().create(mesh);
        built.ids.mesh_geo_id = mesh_slots.geometry->id();
        built.ids.mesh_obj_id = mesh_object->id();

        // The expected close-mesh sizes come from the geometry this test created
        // (not from the checker): every vertex is close (they all share the same
        // height), so the close sets are the full sets.
        built.ids.close_vertices = (y <= 0.0) ? mesh.vertices().size() : 0u;
        built.ids.close_edges    = (y <= 0.0) ? mesh.edges().size() : 0u;
        built.ids.close_tris     = (y <= 0.0) ? mesh.triangles().size() : 0u;

        if(with_half_plane)
        {
            auto hp_object = built.scene.objects().create(built.ids.hp_name);
            auto hp        = halfplane(Vector3::Zero(), Vector3::UnitY());
            auto hp_slots  = hp_object->geometries().create(hp);
            built.ids.hp_geo_id = hp_slots.geometry->id();
            built.ids.hp_obj_id = hp_object->id();
        }

        return built;
    }
}// namespace

TEST_CASE("t43_half_plane_vertex_distance_check", "[luisa][t43]")
{
    // ---- a known violation: the whole square is behind the plane -----------
    {
        auto built = build_half_plane_scene(-0.5, true, true);
        Engine engine{"luisa", t43_workspace("uipc_luisa_t43_sanity_bad")};
        World  world{engine};
        world.init(built.scene);

        CHECK(!world.is_valid());

        auto& errors = world.sanity_checker().errors();
        REQUIRE(errors.find(2) != errors.end());  // uid 2 == half plane distance
        auto& msg = errors.at(2);

        // The reported message is built from the frontend's own ids/names, and the
        // text is the reference's format string byte for byte.
        std::string expected;
        fmt::format_to(std::back_inserter(expected),
                       "Geometry({}) in Object[{}({})] is too close (distance <= 0) to HalfPlane({}) in "
                       "Object[{}({})]\n",
                       built.ids.mesh_geo_id,
                       built.ids.mesh_name,
                       built.ids.mesh_obj_id,
                       built.ids.hp_geo_id,
                       built.ids.hp_name,
                       built.ids.hp_obj_id);
        fmt::format_to(std::back_inserter(expected),
                       "Close mesh has {} vertices, {} edges, {} triangles.\n",
                       built.ids.close_vertices,
                       built.ids.close_edges,
                       built.ids.close_tris);
        expected += "Create mesh [close_mesh<SimplicialComplex>] for post-processing.";

        CHECK(std::string{msg->message()} == expected);

        // ... and the checker that produced it is the *ported* one (the frontend
        // replaces the CPU checker of the same uid 2 by this backend's).
        CHECK(msg->name().find("luisa") != std::string_view::npos);
        CHECK(msg->name().find("cuda") == std::string_view::npos);
        CHECK(msg->result() == SanityCheckResult::Error);

        // one reported geometry pair, one "close_mesh"
        CHECK(msg->geometries().size() == 1u);
        REQUIRE(msg->geometries().find("close_mesh") != msg->geometries().end());
        auto close_mesh = msg->geometries().at("close_mesh")->as<SimplicialComplex>();
        REQUIRE(close_mesh != nullptr);
        CHECK(close_mesh->vertices().size() == built.ids.close_vertices);
        CHECK(close_mesh->edges().size() == built.ids.close_edges);
        CHECK(close_mesh->triangles().size() == built.ids.close_tris);
    }

    // ---- a clean scene: the square is above the plane ----------------------
    {
        auto built = build_half_plane_scene(1.0, true, true);
        Engine engine{"luisa", t43_workspace("uipc_luisa_t43_sanity_clean")};
        World  world{engine};
        world.init(built.scene);

        auto& errors = world.sanity_checker().errors();
        CHECK(errors.find(2) == errors.end());
    }

    // ---- edge case: no half plane at all ----------------------------------
    {
        auto built = build_half_plane_scene(-0.5, false, true);
        Engine engine{"luisa", t43_workspace("uipc_luisa_t43_sanity_nohp")};
        World  world{engine};
        world.init(built.scene);

        auto& errors = world.sanity_checker().errors();
        CHECK(errors.find(2) == errors.end());
    }

    // ---- edge case: a half plane but no surface vertices (0 geometries) ----
    {
        auto config                    = Scene::default_config();
        config["sanity_check"]["mode"] = "quiet";
        config["sanity_check"]["enable"] = true;
        Scene scene{config};

        auto config_hp = halfplane(Vector3::Zero(), Vector3::UnitY());
        scene.objects().create("t43_empty_hp")->geometries().create(config_hp);

        Engine engine{"luisa", t43_workspace("uipc_luisa_t43_sanity_noverts")};
        World  world{engine};
        world.init(scene);

        auto& errors = world.sanity_checker().errors();
        CHECK(errors.find(2) == errors.end());
    }

    // ---- the checker's build() gate: contact disabled -> not inserted ------
    {
        auto built = build_half_plane_scene(-0.5, true, false);
        Engine engine{"luisa", t43_workspace("uipc_luisa_t43_sanity_nocontact")};
        World  world{engine};
        world.init(built.scene);

        auto& errors = world.sanity_checker().errors();
        CHECK(errors.find(2) == errors.end());
    }
}

// ===========================================================================
//  3) dytopo classify info (pure host)
// ===========================================================================
TEST_CASE("t43_dytopo_classify_info", "[luisa][t43]")
{
    gg::DyTopoClassifyInfo info;

    // default: both ranges are empty [0, 0)
    CHECK(info.is_empty());
    CHECK(!info.is_diag());
    CHECK(info.hessian_i_range() == Vector2i{0, 0});
    CHECK(info.hessian_j_range() == Vector2i{0, 0});
    CHECK(info.gradient_i_range() == Vector2i{0, 0});

    // diagonal: gradient range == hessian ranges
    info.range(Vector2i{2, 5});
    CHECK(info.is_diag());
    CHECK(!info.is_empty());
    CHECK(info.gradient_i_range() == Vector2i{2, 5});
    CHECK(info.hessian_i_range() == Vector2i{2, 5});
    CHECK(info.hessian_j_range() == Vector2i{2, 5});
    if constexpr(uipc::RUNTIME_CHECK)
    {
        info.sanity_check();  // must not fire for a well-formed diagonal range
    }

    // off-diagonal: rectangular hessian range, *no* gradient part
    info.range(Vector2i{0, 3}, Vector2i{2, 5});
    CHECK(!info.is_diag());
    CHECK(!info.is_empty());
    CHECK(info.gradient_i_range() == Vector2i::Zero());
    CHECK(info.hessian_i_range() == Vector2i{0, 3});
    CHECK(info.hessian_j_range() == Vector2i{2, 5});
    if constexpr(uipc::RUNTIME_CHECK)
    {
        info.sanity_check();
    }

    // an empty diagonal range is both diag-free and empty
    info.range(Vector2i{3, 3});
    CHECK(!info.is_diag());
    CHECK(info.is_empty());

    // an empty off-diagonal range (i is empty) is empty too
    info.range(Vector2i{4, 4}, Vector2i{2, 5});
    CHECK(!info.is_diag());
    CHECK(info.is_empty());

    // and the diagonal range [0, 0) (no hessian, no gradient) is empty
    info.range(Vector2i{0, 0});
    CHECK(info.is_empty());
}

// ===========================================================================
//  4) dytopo reporter / receiver round trip through the manager
// ===========================================================================
namespace t43_dytopo
{
    constexpr std::string_view marker = "test_t43_dytopo_scene";  // IndexT, meta

    /// Which range/effect shape this file's reporter and receiver should use. This
    /// is the *test's* channel between the test body and its own systems:
    /// Scene::validate_config rejects unknown config groups (a "test_t43/mode" key
    /// throws in World::init), and a config key would also be visible to the other
    /// test sources' scenes. It is only ever read while this file drives the
    /// manager explicitly.
    inline std::string g_mode = "none";

    /// The synthetic effect every mode distributes (hand-computed, exact integers).
    struct Gradient
    {
        IndexT index;
        double v[3];
    };
    struct Triplet
    {
        IndexT i, j;
        double base;  // block value c is base + c (9 values, column-major)
    };

    // Written by the reporter, in this order (duplicates included on purpose:
    // index 5 twice and (0,0) twice, so the conversion must sum them).
    const Gradient kGradients[4] = {{0, {1, 0, 0}},
                                    {5, {0, 2, 0}},
                                    {2, {0, 0, 3}},
                                    {5, {0, 0, 4}}};
    const Triplet kHessians[4]  = {{0, 0, 1.0}, {4, 4, 300.0}, {2, 3, 200.0}, {0, 0, 100.0}};

    // After the BCOO conversion: sorted by index / by (row, col), duplicates summed.
    const Gradient kSortedGradients[3] = {{0, {1, 0, 0}}, {2, {0, 0, 3}}, {5, {0, 2, 4}}};
    const Triplet kSortedHessians[3]   = {{0, 0, 101.0}, {2, 3, 200.0}, {4, 4, 300.0}};
    // (0,0) is the sum of bases 1 and 100: value c is (1 + c) + (100 + c) = 101 + 2c.

    struct Capture
    {
        std::size_t              receives = 0;
        std::vector<IndexT>      g_index;
        std::vector<double>      g_value;  // 3 per doublet
        std::vector<IndexT>      h_row, h_col;
        std::vector<double>      h_value;  // 9 per triplet
    };

    Capture& capture()
    {
        static Capture c;
        return c;
    }

    void clear_capture()
    {
        capture() = Capture{};
    }

    std::vector<double> block_values(double base)
    {
        std::vector<double> v(9);
        for(int c = 0; c < 9; ++c)
            v[c] = base + c;
        return v;
    }
}// namespace t43_dytopo

namespace
{
    /// A vertex reporter for this file's marked geometries (the global vertex array
    /// is the concatenation of the reporters' slices, and the dytopo manager's
    /// fast path compares the receiver's range against that array's size).
    class T43VertexReporter final : public gg::VertexReporter
    {
      public:
        using VertexReporter::VertexReporter;

      protected:
        void do_build(BuildInfo& info) override { (void)info; }

        std::vector<SimplicialComplex*> marked(uipc::backend::WorldVisitor& world)
        {
            std::vector<SimplicialComplex*> result;
            for(auto& slot : world.scene().geometries())
            {
                auto sc = slot->geometry().as<SimplicialComplex>();
                if(sc && sc->meta().find<IndexT>(t43_dytopo::marker))
                {
                    result.push_back(sc);
                }
            }
            return result;
        }

        void do_report_count(VertexCountInfo& info) override
        {
            SizeT count = 0;
            for(auto* sc : marked(world()))
                count += sc->vertices().size();
            info.count(count);
        }

        void do_report_attributes(VertexAttributeInfo& info) override
        {
            std::vector<IndexT>      coindices;
            std::vector<gg::LVector3> positions;
            std::vector<IndexT>      body_ids, dimensions, contact_ids, subscene_ids;
            std::vector<Float>       d_hats, thicknesses;

            for(auto* sc : marked(world()))
            {
                auto positions_view = sc->positions().view();
                for(size_t v = 0; v < positions_view.size(); ++v)
                {
                    coindices.push_back(static_cast<IndexT>(v));
                    positions.push_back(gg::to_luisa(positions_view[v]));
                    body_ids.push_back(-1);
                    dimensions.push_back(0);
                    contact_ids.push_back(0);
                    subscene_ids.push_back(0);
                    d_hats.push_back(0.01);
                    thicknesses.push_back(0.0);
                }
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
            for(auto* sc : marked(world()))
            {
                for(size_t v = 0; v < sc->vertices().size(); ++v)
                    displacements.push_back(gg::LVector3{});
            }
            info.displacements().copy_from(span<const gg::LVector3>{displacements});
        }

        U64 get_uid() const noexcept override { return 0x54343344594E54ull; }  // "T43DYNT"

      private:
        std::vector<gg::LVector3> m_rest_positions;
    };
    REGISTER_SIM_SYSTEM(T43VertexReporter);

    /// The synthetic dytopo reporter: reports the hand-computed effect, gated on the
    /// scene config's mode key (absent => reports nothing for other test scenes).
    class T43DyTopoReporter final : public gg::DyTopoEffectReporter
    {
      public:
        using DyTopoEffectReporter::DyTopoEffectReporter;

      protected:
        std::string_view mode() { return t43_dytopo::g_mode; }

        bool active()
        {
            return mode() == "fast" || mode() == "diag" || mode() == "offdiag";
        }

        void do_build(BuildInfo& info) override { (void)info; }

        void do_report_energy_extent(gg::GlobalDyTopoEffectManager::EnergyExtentInfo& info) override
        {
            info.energy_count(0);
        }

        void do_report_gradient_hessian_extent(
            gg::GlobalDyTopoEffectManager::GradientHessianExtentInfo& info) override
        {
            if(!active())
            {
                info.gradient_count(0);
                info.hessian_count(0);
                return;
            }
            info.gradient_count(4);
            info.hessian_count(4);
        }

        void do_assemble(gg::GlobalDyTopoEffectManager::GradientHessianInfo& info) override
        {
            if(!active())
                return;

            auto gradients = info.gradients();
            auto hessians  = info.hessians();
            REQUIRE(gradients.doublet_count() == 4);
            REQUIRE(hessians.triplet_count() == 4);

            std::vector<IndexT> g_index;
            std::vector<double> g_value;
            for(auto& g : t43_dytopo::kGradients)
            {
                g_index.push_back(g.index);
                for(int c = 0; c < 3; ++c)
                    g_value.push_back(g.v[c]);
            }
            gradients.indices().copy_from(span<const IndexT>{g_index});
            gradients.values().copy_from(span<const double>{g_value});

            std::vector<IndexT> h_row, h_col;
            std::vector<double> h_value;
            for(auto& t : t43_dytopo::kHessians)
            {
                h_row.push_back(t.i);
                h_col.push_back(t.j);
                for(double v : t43_dytopo::block_values(t.base))
                    h_value.push_back(v);
            }
            hessians.row_indices().copy_from(span<const IndexT>{h_row});
            hessians.col_indices().copy_from(span<const IndexT>{h_col});
            hessians.values().copy_from(span<const double>{h_value});
        }

        void do_compute_energy(gg::GlobalDyTopoEffectManager::EnergyInfo& info) override
        {
            (void)info;
        }

        gg::EnergyComponentFlags component_flags() override
        {
            // a single valid flag (the manager asserts exactly one bit)
            return gg::EnergyComponentFlags::Complement;
        }
    };
    REGISTER_SIM_SYSTEM(T43DyTopoReporter);

    /// The synthetic receiver: reports a range that depends on the scene config's
    /// mode key, and copies exactly what the manager hands it back to the host.
    class T43DyTopoReceiver final : public gg::DyTopoEffectReceiver
    {
      public:
        using DyTopoEffectReceiver::DyTopoEffectReceiver;

      protected:
        std::string_view mode() { return t43_dytopo::g_mode; }

        void do_build(BuildInfo& info) override { (void)info; }

        void do_report(gg::GlobalDyTopoEffectManager::ClassifyInfo& info) override
        {
            auto m = mode();
            if(m == "fast")
            {
                // [0, 5): with a single receiver that owns the whole prefix this is
                // the reference's fast path (no conversion, no classification).
                info.range(Vector2i{0, 5});
            }
            else if(m == "diag")
            {
                // diagonal, but not the whole prefix -> the distributed path
                info.range(Vector2i{2, 5});
            }
            else if(m == "offdiag")
            {
                // rectangular hessian range, no gradient part
                info.range(Vector2i{0, 3}, Vector2i{2, 5});
            }
            else if(m == "empty")
            {
                info.range(Vector2i{2, 5});
            }
            else
            {
                info.range(Vector2i{0, 0});  // empty: no report, nothing received
            }
        }

        void do_receive(gg::GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) override
        {
            auto& cap = t43_dytopo::capture();
            cap.receives++;

            auto gradients = info.gradients();
            auto hessians  = info.hessians();

            cap.g_index.resize(gradients.doublet_count());
            cap.g_value.resize(gradients.doublet_count() * 3u);
            cap.h_row.resize(hessians.triplet_count());
            cap.h_col.resize(hessians.triplet_count());
            cap.h_value.resize(hessians.triplet_count() * 9u);

            if(!cap.g_index.empty())
            {
                gradients.indices().copy_to(span<IndexT>{cap.g_index});
                gradients.values().copy_to(span<double>{cap.g_value});
            }
            if(!cap.h_row.empty())
            {
                hessians.row_indices().copy_to(span<IndexT>{cap.h_row});
                hessians.col_indices().copy_to(span<IndexT>{cap.h_col});
                hessians.values().copy_to(span<double>{cap.h_value});
            }
        }
    };
    REGISTER_SIM_SYSTEM(T43DyTopoReceiver);

    /// Caches the ported systems this test drives directly.
    class T43Probe final : public gg::SimSystem
    {
      public:
        using SimSystem::SimSystem;

        static gg::GlobalDyTopoEffectManager* dytopo_manager;
        static gg::GlobalVertexManager*       vertex_manager;
        static gg::LineSearcher*              line_searcher;

      private:
        void do_build() override
        {
            dytopo_manager = find<gg::GlobalDyTopoEffectManager>();
            vertex_manager = &require<gg::GlobalVertexManager>();
            line_searcher  = &require<gg::LineSearcher>();
        }
    };
    REGISTER_SIM_SYSTEM(T43Probe);

    gg::GlobalDyTopoEffectManager* T43Probe::dytopo_manager = nullptr;
    gg::GlobalVertexManager*       T43Probe::vertex_manager = nullptr;
    gg::LineSearcher*              T43Probe::line_searcher  = nullptr;
}// namespace

namespace
{
    std::vector<double> read_block_values(const lud::LCBufferView<double>& values,
                                          std::size_t                     triplet_index)
    {
        std::vector<double> host(9u, 0.0);
        values.subview(triplet_index * 9u, 9u).copy_to(span<double>{host});
        lud::main_stream().synchronize();
        return host;
    }

    void check_sorted_effect(gg::GlobalDyTopoEffectManager* manager)
    {
        REQUIRE(manager != nullptr);

        // the sorted (BCOO) gradient the manager exposes
        auto gradients = manager->gradients();
        CHECK(gradients.doublet_count() == 3);
        std::vector<IndexT> g_index(3u, 0);
        std::vector<double> g_value(9u, 0.0);
        gradients.indices().copy_to(span<IndexT>{g_index});
        gradients.values().copy_to(span<double>{g_value});
        lud::main_stream().synchronize();
        for(int k = 0; k < 3; ++k)
        {
            CHECK(g_index[k] == t43_dytopo::kSortedGradients[k].index);
            for(int c = 0; c < 3; ++c)
                CHECK(g_value[k * 3 + c] == t43_dytopo::kSortedGradients[k].v[c]);
        }

        // the sorted (BCOO) hessian, duplicates summed
        auto hessians = manager->hessians();
        CHECK(hessians.triplet_count() == 3);
        std::vector<IndexT> h_row(3u, 0), h_col(3u, 0);
        hessians.row_indices().copy_to(span<IndexT>{h_row});
        hessians.col_indices().copy_to(span<IndexT>{h_col});
        lud::main_stream().synchronize();
        for(int k = 0; k < 3; ++k)
        {
            CHECK(h_row[k] == t43_dytopo::kSortedHessians[k].i);
            CHECK(h_col[k] == t43_dytopo::kSortedHessians[k].j);
        }
        // (0,0) sums bases 1 and 100 -> value c == 101 + 2c
        auto b0 = read_block_values(hessians.values(), 0u);
        for(int c = 0; c < 9; ++c)
            CHECK(b0[c] == 101.0 + 2.0 * c);
        auto b1 = read_block_values(hessians.values(), 1u);
        for(int c = 0; c < 9; ++c)
            CHECK(b1[c] == 200.0 + c);
        auto b2 = read_block_values(hessians.values(), 2u);
        for(int c = 0; c < 9; ++c)
            CHECK(b2[c] == 300.0 + c);
    }
}// namespace

TEST_CASE("t43_dytopo_round_trip", "[luisa][t43]")
{
    // ---- the scene: 6 marked points + the config-driven mode key -----------
    auto config                       = Scene::default_config();
    config["sanity_check"]["enable"]  = false;
    config["line_search"]["max_iter"] = 16;
    Scene scene{config};

    const std::vector<Vector3> ps = {{0.0, 0.0, 0.0},
                                     {1.0, 0.0, 0.0},
                                     {1.0, 1.0, 0.0},
                                     {0.0, 1.0, 0.0},
                                     {2.0, 0.0, 0.0},
                                     {2.0, 1.0, 0.0}};
    auto points = pointcloud(span<const Vector3>{ps});
    label_surface(points);
    points.meta().create<IndexT>(t43_dytopo::marker, 1);
    scene.objects().create("t43_points")->geometries().create(points);

    Engine engine{"luisa", t43_workspace("uipc_luisa_t43_dytopo")};
    World  world{engine};
    world.init(scene);

    REQUIRE(T43Probe::dytopo_manager != nullptr);  // contact/enable defaults to 1
    REQUIRE(T43Probe::vertex_manager != nullptr);
    REQUIRE(T43Probe::line_searcher != nullptr);
    auto* manager = T43Probe::dytopo_manager;

    // the ported vertex reporter really fed the global array (the fast path's
    // "range <= vertex_count" test needs it), and the mark isolates this test's
    // geometries from the other test sources' reporters.
    const auto vertex_count = T43Probe::vertex_manager->positions().size();
    CHECK(vertex_count >= 6u);

    // the engine's init_scene() ran LineSearcher::init(), which reads the config
    CHECK(T43Probe::line_searcher->max_iter() == 16u);

    auto set_mode = [&](std::string_view m)
    {
        t43_dytopo::g_mode = std::string{m};
        t43_dytopo::clear_capture();
        gg::GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo info;
        info.gradient_only(false);
        info.component_flags(gg::EnergyComponentFlags::All);
        manager->compute_dytopo_effect(info);
        lud::main_stream().synchronize();
    };

    // ---- (a) the single-diagonal-receiver fast path: the receiver gets the
    //          *collected* (unsorted, duplicates kept) effect -----------------
    {
        set_mode("fast");
        auto& cap = t43_dytopo::capture();
        REQUIRE(cap.receives == 1u);

        REQUIRE(cap.g_index.size() == 4u);
        REQUIRE(cap.g_value.size() == 12u);
        for(int k = 0; k < 4; ++k)
        {
            CHECK(cap.g_index[k] == t43_dytopo::kGradients[k].index);
            for(int c = 0; c < 3; ++c)
                CHECK(cap.g_value[k * 3 + c] == t43_dytopo::kGradients[k].v[c]);
        }
        REQUIRE(cap.h_row.size() == 4u);
        REQUIRE(cap.h_col.size() == 4u);
        REQUIRE(cap.h_value.size() == 36u);
        for(int k = 0; k < 4; ++k)
        {
            CHECK(cap.h_row[k] == t43_dytopo::kHessians[k].i);
            CHECK(cap.h_col[k] == t43_dytopo::kHessians[k].j);
            for(int c = 0; c < 9; ++c)
                CHECK(cap.h_value[k * 9 + c] == t43_dytopo::kHessians[k].base + c);
        }
        // The reference's _assemble() sizes the *sorted* containers as well, so the
        // fast path leaves them sized but unconverted; what identifies the fast path
        // is the received data above (collected order, duplicates kept, unlike the
        // sorted/deduplicated distributed path checked in the next block).
        CHECK(manager->hessians().triplet_count() == 4u);
        CHECK(manager->gradients().doublet_count() == 4u);
    }

    // ---- (b) a diagonal range that is not the whole prefix: the sorted,
    //          deduplicated effect is partitioned to the receiver ------------
    {
        set_mode("diag");
        auto& cap = t43_dytopo::capture();
        REQUIRE(cap.receives == 1u);

        // gradient: only the sorted doublet with index in [2, 5)
        REQUIRE(cap.g_index.size() == 1u);
        REQUIRE(cap.g_value.size() == 3u);
        CHECK(cap.g_index[0] == 2);
        CHECK(cap.g_value[0] == 0.0);
        CHECK(cap.g_value[1] == 0.0);
        CHECK(cap.g_value[2] == 3.0);

        // hessian: the sorted triplets with i and j in [2, 5) -> (2,3) and (4,4)
        REQUIRE(cap.h_row.size() == 2u);
        REQUIRE(cap.h_col.size() == 2u);
        REQUIRE(cap.h_value.size() == 18u);
        CHECK(cap.h_row[0] == 2);
        CHECK(cap.h_col[0] == 3);
        CHECK(cap.h_row[1] == 4);
        CHECK(cap.h_col[1] == 4);
        for(int c = 0; c < 9; ++c)
            CHECK(cap.h_value[c] == 200.0 + c);
        for(int c = 0; c < 9; ++c)
            CHECK(cap.h_value[9 + c] == 300.0 + c);

        // ... and the manager's own sorted views hold the whole effect
        check_sorted_effect(manager);
    }

    // ---- (c) an off-diagonal range: rectangular hessian, no gradient --------
    {
        set_mode("offdiag");
        auto& cap = t43_dytopo::capture();
        REQUIRE(cap.receives == 1u);

        CHECK(cap.g_index.empty());  // the reference leaves it empty for off-diagonal
        REQUIRE(cap.h_row.size() == 1u);
        REQUIRE(cap.h_col.size() == 1u);
        REQUIRE(cap.h_value.size() == 9u);
        CHECK(cap.h_row[0] == 2);
        CHECK(cap.h_col[0] == 3);
        for(int c = 0; c < 9; ++c)
            CHECK(cap.h_value[c] == 200.0 + c);
    }

    // ---- (d) an empty effect: the reporter reports nothing -----------------
    {
        set_mode("empty");
        auto& cap = t43_dytopo::capture();
        REQUIRE(cap.receives == 1u);
        CHECK(cap.g_index.empty());
        CHECK(cap.h_row.empty());
        CHECK(cap.h_col.empty());
        CHECK(cap.h_value.empty());
    }

    // ---- (e) back to the fast path after the containers were shrunk --------
    {
        set_mode("fast");
        auto& cap = t43_dytopo::capture();
        REQUIRE(cap.receives == 1u);
        REQUIRE(cap.h_row.size() == 4u);
        for(int k = 0; k < 4; ++k)
        {
            CHECK(cap.h_row[k] == t43_dytopo::kHessians[k].i);
            CHECK(cap.h_col[k] == t43_dytopo::kHessians[k].j);
        }
    }
}
