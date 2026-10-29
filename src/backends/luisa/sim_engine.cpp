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
// T6.1a: the engine drives the ABD dynamics too (the reference's
// sim_engine_do_init.cu step 1 resolves and initialises it *before* the FEM
// method and before the global body manager).
#include <affine_body/affine_body_dynamics.h>
// T6.1b: the inter-affine-body constitution manager is engine-aware too (the
// reference's step 1 initialises it between the ABD dynamics and the FEM method).
#include <affine_body/inter_affine_body_constitution_manager.h>
// T6.4a: the Newton-tolerance manager (the reference's step-1 tail: the line
// searcher, the linear system, the tolerance manager and the time integrator are
// the last four `init()`s of init_scene()).
#include <newton_tolerance/newton_tolerance_manager.h>
// T7.2a: the contact manager (the M7 slice) and the trajectory-filter manager. The
// reference resolves both in SimEngine::build() (`find<>`) and the contact manager's
// init() runs in its step-3.1 forwards.
#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>

// OP-A1/A6: the frame command-list bracket and the process-wide frame stream.
#include <frame_stream.h>
#include <luisa_device/tuning.h>
#include <uipc/common/timer.h>

#include <algorithm>

namespace uipc::backend::luisa
{
namespace
{
/**
 * @brief RAII bracket for the frame command list (optimize_plan.md OP-A1, risk R-5).
 *
 * `advance()` opens the list and the destructor commits it, so an exception thrown
 * anywhere in the frame (SimSystemException for "already converged" is normal
 * control flow here) cannot leave a non-empty list open. That matters twice:
 * LC asserts on destroying a non-empty uncommitted list, and a list left open
 * would be submitted at the *next* frame's commit, mixed with the new frame's
 * commands. `commit()` is idempotent, so the explicit commit at the end of the
 * frame is safe with the destructor as the fallback.
 */
class FrameBracket
{
  public:
    FrameBracket(luisa_device::LStream& stream, bool enabled) noexcept
        : m_stream{stream}
        , m_enabled{enabled}
    {
        if(m_enabled)
        {
            m_stream.begin();
        }
    }

    FrameBracket(const FrameBracket&)            = delete;
    FrameBracket& operator=(const FrameBracket&) = delete;

    ~FrameBracket()
    {
        if(m_enabled)
        {
            m_stream.commit();
        }
    }

  private:
    luisa_device::LStream& m_stream;
    bool                   m_enabled;
};

Json counters_to_json(const luisa_device::LStreamCounters& c)
{
    Json j = Json::object();
    j["commands"]       = c.commands;
    j["direct_submits"] = c.direct_submits;
    j["commits"]        = c.commits;
    j["frame_opens"]    = c.frame_opens;
    j["uploads"]        = c.uploads;
    j["syncs"]          = c.syncs;
    j["sync_ms"]        = c.sync_ms;
    return j;
}
}  // namespace

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
    // OP-A1: the runtime A/B switch from optimize_plan.md §7. Both settings come
    // from the same binary, so the A/B runs in one session on one scene.
    if(config.is_object() && config.contains("luisa") && config["luisa"].is_object())
    {
        const auto& lc = config["luisa"];
        if(lc.contains("frame_batching") && lc["frame_batching"].is_boolean())
        {
            m_frame_batching = lc["frame_batching"].get<bool>();
        }
        // `Timer` is a process-wide switch in the core library, so this is honoured
        // only when the scenario asks for it; a throughput run leaves it alone.
        if(lc.contains("timers") && lc["timers"].is_boolean())
        {
            if(lc["timers"].get<bool>())
            {
                Timer::enable_all();
            }
            else
            {
                Timer::disable_all();
            }
        }
        // OP-C1 / OP-C3: the trace-time knobs (luisa_device/tuning.h).  They are read
        // while a shader is traced -- i.e. after this point -- and baked into the shader,
        // so a new value needs a fresh process, which is exactly the A/B protocol of
        // optimize_plan.md §7.  The environment (`UIPC_LUISA_*`) is applied on the first
        // `tuning()` access and therefore wins over the config: it is the sweep channel.
        auto& tune = luisa_device::tuning();
        if(lc.contains("block_dim") && lc["block_dim"].is_number_integer())
        {
            const auto v = lc["block_dim"].get<std::uint32_t>();
            if(v >= 32u && v <= 1024u && (v % 32u) == 0u)
                tune.block_dim = v;
            else
                logger::warn("config luisa.block_dim = {} rejected (32..1024, multiple of 32)",
                             v);
        }
        if(lc.contains("ccd_max_iter") && lc["ccd_max_iter"].is_number_integer())
        {
            auto v = lc["ccd_max_iter"].get<int>();
            v      = std::clamp(v, luisa_device::kCcdMaxIterFloor, luisa_device::kCcdMaxIterLimit);
            tune.ccd_max_iter = v;
        }
        if(lc.contains("block_dim_table") && lc["block_dim_table"].is_object())
        {
            for(const auto& [kernel, value] : lc["block_dim_table"].items())
            {
                if(!value.is_number_integer())
                    continue;
                const auto v = value.get<std::uint32_t>();
                if(v >= 32u && v <= 1024u && (v % 32u) == 0u)
                    luisa_device::detail::set_named_block_dim(kernel, v);
                else
                    logger::warn("config luisa.block_dim_table['{}'] rejected "
                                 "(32..1024, multiple of 32)",
                                 kernel);
            }
        }
    }

    auto program_path = luisa_device::default_program_path();
    logger::info("LuisaCompute device: backend='{}', program='{}'", backend, program_path);
    luisa_device::Runtime::instance().init(backend, program_path);

    // OP-A6.1 (the asymmetry bench_plan.md B7 calls out): without a sync function
    // `Timer` scopes measure when the host *enqueued* the work, not when the device
    // finished it, which flatters the port against the CUDA backend -- that one
    // installs `cuda_tool::wait_device()` in the same place. Timers are off unless a
    // scenario enables them (`Timer::enable_all()`, wired through
    // `config["luisa"]["timers"]`), so this costs nothing in a throughput run.
    Timer::set_sync_func([] { frame_stream().synchronize(); });

    logger::info("Frame command-list batching: {}",
                 m_frame_batching ? "on" : "off (A/B baseline)");

    // Resolve the trace-time knobs now, so the environment override is applied before the
    // first shader is traced and this log describes the values that will really be baked
    // into the shaders.
    luisa_device::detail::apply_env_once();
    const auto& tune = luisa_device::tuning();
    logger::info("Trace-time tuning: block_dim = {}, ccd_max_iter = {} "
                 "(UIPC_LUISA_BLOCK_DIM / UIPC_LUISA_BLOCK_DIM_TABLE / "
                 "UIPC_LUISA_CCD_MAX_ITER override the config)",
                 tune.block_dim,
                 tune.ccd_max_iter);
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
    // T6.4a: the reference `require<>`s the NewtonToleranceManager too: it is the
    // registry every tolerance checker registers with (`MaxTranslationChecker` in
    // every scene, `ABDToleranceChecker` in an ABD scene), and it is registered
    // unconditionally, so `require<>` never throws.
    m_newton_tolerance_manager = &require<NewtonToleranceManager>();

    m_global_simplicial_surface_manager = find<GlobalSimplicialSurfaceManager>();
    m_global_dytopo_effect_manager = find<GlobalDyTopoEffectManager>();
    // T7.2a: the reference's order here is dytopo -> contact manager ->
    // trajectory filter -> animator (sim_engine_do_init.cu step 2). Both are created
    // only when the scene enables contact.
    m_global_contact_manager = find<GlobalContactManager>();
    m_global_trajectory_filter = find<GlobalTrajectoryFilter>();
    m_global_animator = find<GlobalAnimator>();
    m_global_external_force_manager     = find<GlobalExternalForceManager>();
    // T5.1: the reference `find<>`s the FEM method alongside AffineBodyDynamics
    // (optional: only a scene with a FiniteElement constitution has one).
    m_finite_element_method = find<FiniteElementMethod>();
    // T6.1a: ABD dynamics (optional: only a scene with an AffineBody constitution
    // type has one).  The reference resolves it *before* the FEM method.
    m_affine_body_dynamics = find<AffineBodyDynamics>();
    // T6.1b: the inter-affine-body constitution manager (optional: its
    // SimSystemCreator only creates it when the scene has an `InterAffineBody`
    // constitution type).  The reference resolves it right after the dynamics and
    // *before* the FEM method (sim_engine_do_init.cu step 1).
    m_inter_affine_body_constitution_manager = find<InterAffineBodyConstitutionManager>();
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

    // 1. Before the common scene initialization: the ABD dynamics builds its
    //    geometry/constitution infos and hands the constitutions their init();
    //    then the inter-affine-body manager groups the joint geometries and hands
    //    its constitutions theirs; then the FEM method does the same, and finally
    //    the body manager hands out the global body ranges that the vertex
    //    reporters use while filling body_ids.
    //    (Reference order, sim_engine_do_init.cu step 1:
    //     affine_body_dynamics -> inter_affine_body_constitution_manager ->
    //     finite_element_method -> global_body_manager.)
    if(m_affine_body_dynamics)
    {
        m_affine_body_dynamics->init();
    }
    if(m_inter_affine_body_constitution_manager)
    {
        m_inter_affine_body_constitution_manager->init();
    }
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
    // -> animator -> external force -> active set (Phase 8) ->
    // line searcher -> linear system (Phase 8) -> time integrator -> ...
    if(m_global_dytopo_effect_manager)
        m_global_dytopo_effect_manager->init();
    // T7.2a: the contact manager's init() builds the contact/subscene mask tabulars
    // and the vertex contact buffers (the trajectory filters' `BaseInfo` reads them).
    if(m_global_contact_manager)
        m_global_contact_manager->init();
    if(m_global_animator)
        m_global_animator->init();
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
    // T6.4a: the tolerance manager's init() hands every registered checker its
    // init().  The reference calls it unguarded (it is `require<>`d, like the line
    // searcher) and after the time-integrator manager's.
    m_newton_tolerance_manager->init();

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
    m_frame_completed  = false;

    logger::info("do_advance() called, frame = {}", m_current_frame);

    // OP-A1: open the frame command list. Every `frame_stream().record(...)` in the
    // systems below is then *appended* instead of submitted, so one frame costs
    // O(commits) submissions instead of O(commands). The bracket commits at scope
    // exit (exception safe) and the explicit commit before the counters covers the
    // normal path.
    auto&      stream      = frame_stream();
    const auto counters_at = stream.counters();
    // OP-A4: compile *calls* per frame.  Every shader creation in this backend goes
    // through `luisa_device::compile1/2/3`, so this delta is the exact JIT count of the
    // frame; "0 after frame 1" is the acceptance of optimize_plan.md OP-A4, and it is
    // what catches a `static` lost during a refactor (which would re-JIT every frame).
    const auto compiles_at = luisa_device::compile_stats::total();
    FrameBracket bracket{stream, m_frame_batching};

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

    // Commit explicitly (the bracket's destructor is the exception path) so that the
    // counters below describe a frame that was really submitted.
    if(m_frame_batching)
    {
        stream.commit();
    }
    m_frame_counters  = counters_to_json(stream.counters() - counters_at);
    m_frame_counters["compiles"] = luisa_device::compile_stats::total() - compiles_at;
    m_frame_completed = true;

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
    // OP-A1 (risk R-3): this used to synchronize the *raw* lcrt::Stream. With a frame
    // list open that is a no-op -- the work is still sitting in the list, so
    // `World::sync()` returned while the GPU had queued work and the following
    // `retrieve()` could read stale state. `LStream::synchronize()` commits the list
    // first, then waits (and re-opens the list for the next frame).
    frame_stream().synchronize();
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
    // OP-A6.3: the same schema the CUDA backend emits (sim_engine.cu:147), so one
    // collector reads both backends. The Newton/line-search fields are the shell's
    // bookkeeping until the Phase 8 solver drives them.
    Json stats                  = Json::object();
    stats["schema_version"]     = 1;
    stats["backend"]            = "luisa";
    stats["frame"]              = m_current_frame;
    stats["pipeline"]           = m_pipeline_type == PipelineType::Basic ? "ipc" : "al-ipc";
    stats["completed"]          = m_frame_completed;
    stats["converged"]          = false;  // the Newton loop is Phase 8 (T8.4)
    stats["newton_iterations"]  = m_newton_iter;
    stats["line_search_trials"] = m_line_search_iter;
    stats["linear_solver_iterations"] = 0;
    stats["hit_newton_limit"]         = false;
    stats["hit_line_search_limit"]    = false;
    stats["last_line_search_alpha"]   = 1.0;
    stats["last_ccd_toi"]             = 1.0;
    stats["last_cfl_alpha"]           = 1.0;
    // OP-A1/A6 attribution: what the frame cost in submissions (see LStreamCounters).
    stats["frame_batching"] = m_frame_batching;
    if(!m_frame_counters.empty())
    {
        stats["counters"] = m_frame_counters;
    }
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
