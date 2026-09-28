#include <sim_engine.h>

#include <sim_system.h>
#include <uipc/common/log.h>
#include <uipc/common/log_pattern_guard.h>

#include <backends/common/module.h>

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

void SimEngine::init_scene()
{
    m_state = SimEngineState::InitScene;
    event_init_scene();
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
