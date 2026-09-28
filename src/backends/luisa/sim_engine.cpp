#include <sim_engine.h>

#include <sim_system.h>
#include <uipc/common/log.h>
#include <uipc/common/log_pattern_guard.h>

#include <backends/common/module.h>
// T4.3: the constructor touches EngineCreateInfo::config (EngineCreateInfo is
// only forward-declared by the base engine header).
#include <uipc/backend/engine_create_info.h>

// T4.1/T4.2: the engine drives the engine-aware systems (see init_scene() /
// do_advance()); the include order mirrors the reference's sim_engine_do_init.cu.
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <animator/global_animator.h>
#include <external_force/global_external_force_manager.h>
#include <pipeline/ipc_pipeline_flag.h>
#include <pipeline/al_ipc_pipeline_flag.h>
#include <time_integrator/time_integrator_manager.h>
// T4.3
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <line_search/line_searcher.h>
// T5.1
#include <finite_element/finite_element_method.h>


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
    // T4.3: bring the LC device up here, like the reference backend does in its
    // constructor. `World::init` runs the sanity check *before* it calls
    // engine->init() (i.e. before do_init()), and the backend's checkers are
    // device code, so the device must already exist at that point.
    // Runtime::init() is idempotent (a no-op when the runtime is up), so the
    // do_init() call below is unchanged.
    init_device(info->config);
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

    init_device(info.config());  // idempotent; the constructor already did this

    m_state = SimEngineState::BuildSystems;
    build_systems();
    // T4.1/T4.2: the reference resolves its engine-aware systems inside build(),
    // i.e. still in the BuildSystems state.
    build_engine_systems();
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

void SimEngine::build_engine_systems()
{
    // Mirrors SimEngine::build() of the reference backend
    // (src/backends/cuda/engine/sim_engine_do_init.cu):
    //   require<>: GlobalVertexManager, GlobalBodyManager, TimeIntegratorManager,
    //              LineSearcher, GlobalLinearSystem, NewtonToleranceManager
    //   find<>   : GlobalSimplicialSurfaceManager, GlobalDyTopoEffectManager,
    //              GlobalContactManager, GlobalTrajectoryFilter, GlobalAnimator,
    //              GlobalExternalForceManager, GlobalDiffSimManager,
    //              AffineBodyDynamics, ..., GlobalActiveSetManager
    // Only the systems T4.1/T4.2 ported are resolved here; the rest are the
    // subject of Phase 5/6/8 and are simply absent from the graph (the reference
    // guards every optional one with `if(ptr)`, so the behaviour is identical).
    m_global_vertex_manager = &require<GlobalVertexManager>();
    m_global_body_manager   = &require<GlobalBodyManager>();
    m_time_integrator_manager = &require<TimeIntegratorManager>();

    m_line_searcher = &require<LineSearcher>();

    m_global_simplicial_surface_manager = find<GlobalSimplicialSurfaceManager>();
    m_global_dytopo_effect_manager = find<GlobalDyTopoEffectManager>();
    m_global_animator                   = find<GlobalAnimator>();
    m_global_external_force_manager     = find<GlobalExternalForceManager>();
    // T5.1: the reference `find<>`s the FEM method alongside AffineBodyDynamics
    // (optional: only a scene with a FiniteElement constitution has one).
    m_finite_element_method = find<FiniteElementMethod>();
}

void SimEngine::set_pipeline_type()
{
    // The reference does this at the top of init_scene(): exactly one of the two
    // pipeline flag systems exists, because each is created by a
    // SimSystemCreator specialization that tests the scene config's
    // `contact/constitution` value.
    auto alipc = find<ALIPCPipelineFlag>();
    auto ipc   = find<IPCPipelineFlag>();
    if(alipc)
    {
        logger::info("Pipeline: Augmented Lagrangian IPC");
        m_pipeline_type = PipelineType::AugmentedLagrangian;
    }
    else if(ipc)
    {
        m_pipeline_type = PipelineType::Basic;
    }
    else
    {
        throw SimEngineException("No valid pipeline flag found in the scene!");
    }
}

void SimEngine::init_scene()
{
    m_state = SimEngineState::InitScene;

    set_pipeline_type();

    // 1. Before the common scene initialization: the FEM method builds its
    //    geometry/constitution infos, uploads the rest state and hands the
    //    constitutions their init(); then the body manager hands out the global
    //    body ranges that the vertex reporters use while filling body_ids.
    //    (Reference order, sim_engine_do_init.cu step 1:
    //     affine_body_dynamics -> inter_affine_body_constitution_manager ->
    //     finite_element_method -> global_body_manager.)
    if(m_finite_element_method)
    {
        m_finite_element_method->init();
    }
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
    // 3.1 Forwards (reference order): dytopo effect manager -> contact manager
    //     (Phase 8) -> animator -> external force -> active set (Phase 8) ->
    //     line searcher -> linear system (Phase 8) -> time integrator -> ...
    if(m_global_dytopo_effect_manager)
    {
        m_global_dytopo_effect_manager->init();
    }
    if(m_global_animator)
    {
        m_global_animator->init();
    }
    if(m_global_external_force_manager)
    {
        m_global_external_force_manager->init();
    }
    // (GlobalActiveSetManager -> Phase 8; GlobalLinearSystem -> Phase 8)
    m_line_searcher->init();
    if(m_time_integrator_manager)
    {
        m_time_integrator_manager->init();
    }
    // (NewtonToleranceManager -> Phase 8)

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

void SimEngine::step_animation_and_external_forces()
{
    // Verbatim port of the reference's SimEngine::step_animation_and_external_forces()
    // (src/backends/cuda/engine/sim_engine.cu): external-force reporters accumulate
    // into shared dynamics buffers, so the previous frame is cleared *before* user
    // animation mutates the force attributes and the freshly animated values are
    // consumed *after* the animation step, both before the DOF prediction.
    if(m_global_external_force_manager)
    {
        m_global_external_force_manager->clear();
    }

    if(m_global_animator)
    {
        m_global_animator->step();
    }

    if(m_global_external_force_manager)
    {
        m_global_external_force_manager->step();
    }
}

void SimEngine::advance()
{
    // Scope note (T4.2): the reference's advance() is a Newton/line-search
    // pipeline (advance_ipc.cu) that only exists once the solver systems are
    // ported (Phase 8). What is ported here is the frame prologue and epilogue
    // that the T4.2 systems participate in, in the reference's exact order:
    //
    //   ++frame
    //   RebuildScene event + solve_pending()
    //   0. Process External Changes : update_attributes() + bounding box
    //   1. record_prev_positions()
    //   2. Predict Motion           : step_animation_and_external_forces()
    //                                 time_integrator_manager->predict_dof()
    //   4. Newton loop              : compute_animation_substep_ratio()
    //   5. Update Velocity          : time_integrator_manager->update_state()
    ++m_current_frame;
    m_newton_iter      = 0;
    m_line_search_iter = 0;

    logger::info("do_advance() called, frame = {}", m_current_frame);

    // Rebuild Scene
    {
        m_state = SimEngineState::RebuildScene;
        event_rebuild_scene();
        // After the rebuild_scene event, the pending creation or deletion can be
        // solved.
        world().scene().solve_pending();
    }

    // 0. Process External Changes
    if(m_global_vertex_manager)
    {
        m_global_vertex_manager->update_attributes();
        [[maybe_unused]] AABB bbox = m_global_vertex_manager->compute_vertex_bounding_box();

        // 1. Record the friction candidates' reference state at the frame start
        m_global_vertex_manager->record_prev_positions();
    }

    // 2. Predict Motion => x_tilde = x + v * dt
    m_state = SimEngineState::PredictMotion;
    // MUST step animation before predicting dof: some animations provide
    // information for the DOF prediction.
    step_animation_and_external_forces();
    if(m_time_integrator_manager)
    {
        m_time_integrator_manager->predict_dof();
    }

    // 4. Nonlinear-Newton iteration. The reference recomputes the animation
    //    substep ratio at the top of every iteration and refuses to converge
    //    while the animation has not reached its target
    //    (`animation_reach_target()`), i.e. at least the newton_iter = 0
    //    bookkeeping happens in every frame. The solver loop itself is Phase 8,
    //    so the shell performs that single iteration-0 step here; without it the
    //    ratio would stay at the 0 that GlobalAnimator::step() resets it to.
    if(m_global_animator)
    {
        m_global_animator->compute_substep_ratio(0);
        logger::info("Animation Substep Ratio: {}", m_global_animator->substep_ratio());
    }

    // 5. Update Velocity => v = (x - x_0) / dt
    m_state = SimEngineState::UpdateVelocity;
    if(m_time_integrator_manager)
    {
        m_time_integrator_manager->update_state();
    }

    logger::info("<<< End Frame: {}", m_current_frame);
}

void SimEngine::do_advance()
{
    LogPatternGuard guard("Luisa");

    switch(m_pipeline_type)
    {
        case PipelineType::Basic:
        case PipelineType::AugmentedLagrangian:
            // The AL pipeline differs from the basic one only in the systems it
            // drives (active set / AL contact), none of which are ported yet; the
            // ported frame structure is identical for both.
            advance();
            break;
        default:
            UIPC_ERROR_WITH_LOCATION("Unknown pipeline type");
            break;
    }
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
