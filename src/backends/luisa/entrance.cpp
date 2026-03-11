#include <sim_engine.h>
#include <backends/common/module.h>

namespace uipc::backend::luisa
{
// SimEngine Implementation

SimEngine::SimEngine(EngineCreateInfo* info)
    : backend::SimEngine(info)
{
    // Initialize LuisaCompute context
    // In a real implementation, argv[0] or the executable path would be passed
    m_luisa_context = std::make_unique<luisa::compute::Context>(
        luisa::string{"uipc_luisa_backend"});
    
    // Create device - default to CUDA backend, but could be "dx", "metal", or "cpu"
    // The backend can be configured through the EngineCreateInfo
    m_luisa_device = m_luisa_context->create_device(m_backend_name);
    
    // Create compute stream for command submission
    m_compute_stream = m_luisa_device.create_stream(luisa::compute::StreamTag::COMPUTE);
}

SimEngine::~SimEngine()
{
    // Cleanup is handled by RAII destructors of LuisaCompute objects
    m_compute_stream = nullptr;
    m_luisa_device = nullptr;
    m_luisa_context.reset();
}

void SimEngine::do_init(InitInfo& info)
{
    // Initialize simulation systems
    build();
    init_scene();
}

void SimEngine::do_advance()
{
    // Main simulation step
    // This would implement the Newton-based IPC solver
    // adapted for LuisaCompute's kernel dispatch model
    m_state = SimEngineState::RebuildScene;
    event_rebuild_scene();
    
    // TODO: Implement full simulation pipeline:
    // 1. Predict motion
    // 2. Build collision pairs
    // 3. Newton iteration loop
    // 4. Line search
    // 5. Update velocity
    
    m_current_frame++;
}

void SimEngine::do_sync()
{
    // Synchronize host and device data
    m_compute_stream << luisa::compute::synchronize();
}

void SimEngine::do_retrieve()
{
    // Retrieve simulation results from GPU
    event_write_scene();
}

SizeT SimEngine::get_frame() const
{
    return m_current_frame;
}

bool SimEngine::do_dump(DumpInfo& info)
{
    // Implement checkpoint saving
    return true;
}

bool SimEngine::do_try_recover(RecoverInfo& info)
{
    // Implement recovery attempt
    return true;
}

void SimEngine::do_apply_recover(RecoverInfo& info)
{
    // Apply recovered state
}

void SimEngine::do_clear_recover(RecoverInfo& info)
{
    // Clear recovery data
}

void SimEngine::build()
{
    // Build all simulation systems
    m_state = SimEngineState::BuildSystems;
    build_systems();
}

void SimEngine::init_scene()
{
    m_state = SimEngineState::InitScene;
    event_init_scene();
}

void SimEngine::dump_global_surface()
{
    // Dump surface mesh for debugging
}

// Event dispatchers
void SimEngine::event_init_scene()
{
    m_on_init_scene();
}

void SimEngine::event_rebuild_scene()
{
    m_on_rebuild_scene();
}

void SimEngine::event_write_scene()
{
    m_on_write_scene();
}

// Getters
SimEngineState SimEngine::state() const noexcept
{
    return m_state;
}

SizeT SimEngine::newton_iter() const noexcept
{
    return m_newton_iter;
}

SizeT SimEngine::line_search_iter() const noexcept
{
    return m_line_search_iter;
}

luisa::compute::Context& SimEngine::luisa_context() noexcept
{
    return *m_luisa_context;
}

luisa::compute::Device& SimEngine::luisa_device() noexcept
{
    return m_luisa_device;
}

luisa::compute::Stream& SimEngine::compute_stream() noexcept
{
    return m_compute_stream;
}

}  // namespace uipc::backend::luisa

// =============================================================================
// Backend Entry Points (C API)
// =============================================================================

/**
 * @brief Create a new LuisaCompute backend engine instance
 * 
 * This is the main entry point for the UIPC backend system.
 * It creates a SimEngine that uses LuisaCompute for GPU acceleration.
 */
UIPC_BACKEND_API EngineInterface* uipc_create_engine(EngineCreateInfo* info)
{
    return new uipc::backend::luisa::SimEngine(info);
}

/**
 * @brief Destroy a LuisaCompute backend engine instance
 */
UIPC_BACKEND_API void uipc_destroy_engine(EngineInterface* engine)
{
    delete engine;
}
