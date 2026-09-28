/********************************************************************
 * @file backend_shell_test.cpp
 * @brief T1.4 — backend shell + SimSystem plumbing test.
 *
 * Loads the `luisa` backend through the public engine API (the same path the
 * sim-case suite uses), initializes a minimal scene and advances one frame.
 * That exercises: module loading, SimEngine construction, LC runtime bring-up,
 * system registration/build (SimEngine::build_systems -> REGISTER_SIM_SYSTEM),
 * find()/require() resolution inside LuisaBackendSimSystem::do_build and the
 * init/advance/sync/retrieve pipeline hooks.
 *********************************************************************/
#include "luisa_test_shim.h"

#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>

#include <filesystem>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

namespace
{
std::string shell_output_path()
{
    auto p = std::filesystem::temp_directory_path() / "uipc_luisa_shell_test";
    std::filesystem::create_directories(p);
    return p.string() + "/";
}
}// namespace

TEST_CASE("backend_module_loads", "[luisa][backend]")
{
    // Engine construction loads the backend module and creates the SimEngine.
    CHECK_NOTHROW(([&]() {
        Engine engine{"luisa", shell_output_path()};
        (void)engine;
    })());
}

TEST_CASE("shell_init_advance_retrieve", "[luisa][backend]")
{
    Engine engine{"luisa", shell_output_path()};
    World  world{engine};

    auto config                  = Scene::default_config();
    config["gravity"]            = Vector3{0.0, -9.8, 0.0};
    config["contact"]["enable"]  = false;

    Scene scene{config};

    // Minimal geometry: a single tetrahedron carrying an affine-body constitution.
    auto object = scene.objects().create("tet");

    std::vector<Vector3>  Vs = {Vector3{0.0, 0.0, 0.0},
                                Vector3{1.0, 0.0, 0.0},
                                Vector3{0.0, 1.0, 0.0},
                                Vector3{0.0, 0.0, 1.0}};
    std::vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};
    auto                  mesh = tetmesh(Vs, Ts);
    label_surface(mesh);
    label_triangle_orient(mesh);

    AffineBodyConstitution abd;
    abd.apply_to(mesh, 100.0_MPa);
    object->geometries().create(mesh);

    // init -> build systems on the backend, bring up the LC runtime
    CHECK_NOTHROW(world.init(scene));

    const auto frame_before = world.frame();
    CHECK_NOTHROW(world.advance());
    CHECK_NOTHROW(world.sync());
    CHECK_NOTHROW(world.retrieve());

    // The shell must advance the frame counter.
    CHECK(world.frame() == frame_before + 1);
}
