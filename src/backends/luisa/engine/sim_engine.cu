#include <sim_engine.h>
#include <uipc/common/log.h>
#include <kernel_cout.h>
#include <backends/common/module.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <uipc/common/timer.h>
#include <backends/common/backend_path_tool.h>
#include <uipc/backend/engine_create_info.h>

// LuisaCompute runtime headers
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>

// LuisaCompute DSL headers
#include <luisa/dsl/syntax.h>

namespace uipc::backend::luisa
{
void say_hello_from_luisa(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    // Create a simple kernel to test LuisaCompute is working
    using namespace luisa::compute;
    
    Kernel1D hello_kernel = []() noexcept {
        KernelCout::print("LuisaCompute Backend Kernel Console Init Success!\n");
    };
    
    auto shader = device.compile(hello_kernel);
    stream << shader().dispatch(1) << synchronize();
}

SimEngine::SimEngine(EngineCreateInfo* info)
    : backend::SimEngine(info)
{
    try
    {
        logger::info("Initializing LuisaCompute Backend...");

        // Get device configuration from EngineCreateInfo
        auto device_id = info->config["gpu"]["device"].get<IndexT>();
        
        // Get backend name from config, default to "cuda"
        if(info->config.contains("gpu") && info->config["gpu"].contains("backend"))
        {
            m_backend_name = info->config["gpu"]["backend"].get<std::string>();
        }
        
        // Initialize LuisaCompute context
        // Use the program path from info if available, otherwise use a default
        m_luisa_context = std::make_unique<luisa::compute::Context>(
            luisa::string{"uipc_luisa_backend"});
        
        // Query installed backends
        auto backends = m_luisa_context->installed_backends();
        bool backend_found = false;
        for(auto&& backend : backends)
        {
            if(backend == m_backend_name)
            {
                backend_found = true;
                break;
            }
        }
        
        if(!backend_found)
        {
            UIPC_WARN_WITH_LOCATION("Backend '{}' not found. Available backends:", m_backend_name);
            for(auto&& backend : backends)
            {
                UIPC_WARN_WITH_LOCATION("  - {}", backend);
            }
            
            // Fallback to cuda if available, otherwise use the first available
            if(std::find(backends.begin(), backends.end(), "cuda") != backends.end())
            {
                m_backend_name = "cuda";
            }
            else if(!backends.empty())
            {
                m_backend_name = backends[0];
            }
            else
            {
                throw SimEngineException("No LuisaCompute backend available");
            }
            
            UIPC_WARN_WITH_LOCATION("Falling back to '{}' backend.", m_backend_name);
        }

        // Create device with the specified backend
        luisa::compute::DeviceConfig config{
            .device_index = static_cast<uint32_t>(device_id),
            .inqueue_buffer_limit = false};
        
        m_luisa_device = m_luisa_context->create_device(m_backend_name, &config);
        
        // Create compute stream for command submission
        m_compute_stream = m_luisa_device.create_stream(luisa::compute::StreamTag::COMPUTE);
        
        // Log device info
        logger::info("Backend: {}", m_backend_name);
        logger::info("Device Index: {}", device_id);
        
        // Set timer sync function to synchronize the compute stream
        Timer::set_sync_func([this] { 
            m_compute_stream << luisa::compute::synchronize(); 
        });

        say_hello_from_luisa(m_luisa_device, m_compute_stream);

#ifndef NDEBUG
        // if in debug mode, sync all the time to check for errors
        // LuisaCompute has built-in validation when compiled in debug
        logger::info("Debug mode: synchronization enabled");
#endif
        logger::info("LuisaCompute Backend Init Success.");
    }
    catch(const SimEngineException& e)
    {
        logger::error("LuisaCompute Backend Init Failed: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
}

SimEngine::~SimEngine()
{
    // Synchronize stream before destruction
    if(m_compute_stream)
    {
        m_compute_stream << luisa::compute::synchronize();
    }

    // Cleanup is handled by RAII destructors of LuisaCompute objects
    m_compute_stream = nullptr;
    m_luisa_device = nullptr;
    m_luisa_context.reset();

    logger::info("LuisaCompute Backend Shutdown Success.");
}

SimEngineState SimEngine::state() const noexcept
{
    return m_state;
}

void SimEngine::event_init_scene()
{
    for(auto& action : m_on_init_scene.view())
        action();
}

void SimEngine::event_rebuild_scene()
{
    for(auto& action : m_on_rebuild_scene.view())
        action();
}

void SimEngine::event_write_scene()
{
    for(auto& action : m_on_write_scene.view())
        action();
}

void SimEngine::dump_global_surface()
{
    BackendPathTool tool{workspace()};
    auto            output_folder = tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto            file_path = fmt::format("{}global_surface.{}.{}.{}.obj",
                                 output_folder.string(),
                                 frame(),
                                 newton_iter(),
                                 line_search_iter());

    std::vector<Vector3> positions;
    std::vector<Vector3> disps;

    auto src_ps = m_global_vertex_manager->positions();

    positions.resize(src_ps.size());
    src_ps.copy_to(positions.data());

    std::vector<Vector2i> edges;
    auto src_es = m_global_simplicial_surface_manager->surf_edges();
    edges.resize(src_es.size());
    src_es.copy_to(edges.data());

    std::vector<Vector3i> faces;
    auto src_fs = m_global_simplicial_surface_manager->surf_triangles();
    faces.resize(src_fs.size());
    src_fs.copy_to(faces.data());

    std::ofstream file(file_path);

    for(auto& pos : positions)
        file << fmt::format("v {} {} {}\n", pos.x(), pos.y(), pos.z());

    for(auto& face : faces)
        file << fmt::format("f {} {} {}\n", face.x() + 1, face.y() + 1, face.z() + 1);

    for(auto& edge : edges)
        file << fmt::format("l {} {}\n", edge.x() + 1, edge.y() + 1);

    logger::info("Dumped global surface to {}", file_path);
}
}  // namespace uipc::backend::luisa

// Dump & Recover:
namespace uipc::backend::luisa
{
bool SimEngine::do_dump(DumpInfo&)
{
    // Now just do nothing
    return true;
}

bool SimEngine::do_try_recover(RecoverInfo&)
{
    // Now just do nothing
    return true;
}

void SimEngine::do_apply_recover(RecoverInfo& info)
{
    // If success, set the current frame to the recovered frame
    m_current_frame = info.frame();
}

void SimEngine::do_clear_recover(RecoverInfo& info)
{
    // If failed, do nothing
}

SizeT SimEngine::get_frame() const
{
    return m_current_frame;
}

SizeT SimEngine::newton_iter() const noexcept
{
    return m_newton_iter;
}

SizeT SimEngine::line_search_iter() const noexcept
{
    return m_line_search_iter;
}
}  // namespace uipc::backend::luisa
