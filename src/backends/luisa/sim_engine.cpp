#include <sim_engine.h>

#include <sim_system.h>
#include <uipc/common/log.h>
#include <uipc/common/log_pattern_guard.h>

#include <backends/common/module.h>

// T4.1: the engine drives the global geometry managers (see init_scene()).
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>

namespace uipc::backend::luisa
{
SimEngine::SimEngine(EngineCreateInfo* info)
    : backend::SimEngine{info}
{
    LogPatternGuard guard("Luisa");
    logger::info("Constructor called.");
    logger::info(
        "LuisaCompute backend (LC device backend 'cuda' by default). "
        "The full advance pipeline is assembled in Phase 8.");
}

SimEngine::~SimEngine()
{
    LogPatternGuard guard("Luisa");
    logger::info("Destructor called.");
    // Tear the LC runtime down only after every engine-owned resource is gone.
    if(luisa_device::Runtime::instance().initialized())
    {
        luisa_device::Runtime::instance().shutdown();
    }
}

void SimEngine::init_device(const Json& config)
{
    // Device selection: engine config "luisa_backend" (default "cuda").
    std::string backend = "cuda";
    if(config.is_object() && config.contains("luisa_backend") &&
       config["luisa_backend"].is_string())
    {
        backend = config["luisa_backend"].get<std::string>();
    }
    auto program_path = luisa_device::default_program_path();
    logger::info("LuisaCompute device: backend='{}', program='{}'", backend, program_path);
    luisa_device::Runtime::instance().init(backend, program_path);
}

void SimEngine::do_init(InitInfo& info)
{
    LogPatternGuard guard("Luisa");
    logger::info("do_init() called.");

    init_device(info.config());

    m_state = SimEngineState::BuildSystems;
    build_systems();
    // T4.1: the reference resolves its engine-aware topo systems inside build(),
    // i.e. still in the BuildSystems state.
    build_geometry_systems();
    m_state = SimEngineState::None;

    dump_system_info();

    init_scene();

    // Match the frontend mutation lifecycle used by other backends: changes made
    // after World::init() are deferred until the next advance.
    world().scene().begin_pending();
}

void SimEngine::build_systems()
{
    backend::SimEngine::build_systems();
}

void SimEngine::build_geometry_systems()
{
    // Mirrors SimEngine::build() of the reference backend
    // (src/backends/cuda/engine/sim_engine_do_init.cu):
    //   m_global_vertex_manager = &require<GlobalVertexManager>();
    //   m_global_body_manager   = &require<GlobalBodyManager>();
    //   m_global_simplicial_surface_manager = find<GlobalSimplicialSurfaceManager>();
    // All three are always registered, so require<> is used for the two the
    // reference requires and find<> for the surface manager (like the reference).
    m_global_vertex_manager = &require<GlobalVertexManager>();
    m_global_body_manager   = &require<GlobalBodyManager>();
    m_global_simplicial_surface_manager = find<GlobalSimplicialSurfaceManager>();
}

void SimEngine::init_scene()
{
    m_state = SimEngineState::InitScene;

    // 1. Before the common scene initialization: the body manager hands out the
    //    global body ranges that the vertex reporters use while filling body_ids.
    if(m_global_body_manager)
    {
        m_global_body_manager->init();
    }

    // 2. Common scene initialization phase.
    event_init_scene();

    // 3. After the common scene initialization (same order as the reference).
    if(m_global_vertex_manager)
    {
        m_global_vertex_manager->init();
    }
    if(m_global_simplicial_surface_manager)
    {
        m_global_simplicial_surface_manager->init();
    }

    event_rebuild_scene();
    m_state = SimEngineState::None;
}

void SimEngine::event_init_scene()
{
    for(auto& action : m_on_init_scene.view())
    {
        action();
    }
}

void SimEngine::event_rebuild_scene()
{
    for(auto& action : m_on_rebuild_scene.view())
    {
        action();
    }
}

void SimEngine::event_write_scene()
{
    for(auto& action : m_on_write_scene.view())
    {
        action();
    }
}

void SimEngine::do_advance()
{
    LogPatternGuard guard("Luisa");
    world().scene().solve_pending();

    m_current_frame++;
    m_newton_iter      = 0;
    m_line_search_iter = 0;

    logger::info("do_advance() called, frame = {}", m_current_frame);
}

void SimEngine::do_sync()
{
    // Phase 8 wires the per-frame pipeline here. The only synchronization the
    // shell performs is flushing the LC frame command list.
    luisa_device::main_stream().synchronize();
}

void SimEngine::do_retrieve()
{
    // nothing to retrieve yet (Phase 8)
}

SizeT SimEngine::get_frame() const
{
    return m_current_frame;
}

Json SimEngine::do_frame_stats() const
{
    Json stats = Json::object();
    stats["frame"]            = m_current_frame;
    stats["newton_iters"]     = m_newton_iter;
    stats["line_search_iters"] = m_line_search_iter;
    return stats;
}

bool SimEngine::do_dump(DumpInfo& info)
{
    // No state to dump yet; Phase 8 implements the real dump/recover paths.
    event_write_scene();
    return true;
}

bool SimEngine::do_try_recover(RecoverInfo& info)
{
    return true;
}

void SimEngine::do_apply_recover(RecoverInfo& info)
{
    m_current_frame = info.frame();
}

void SimEngine::do_clear_recover(RecoverInfo& info)
{
    // nothing to clear
}
}// namespace uipc::backend::luisa
