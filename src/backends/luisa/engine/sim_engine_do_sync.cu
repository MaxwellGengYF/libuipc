#include <sim_engine.h>

namespace uipc::backend::luisa
{
void SimEngine::do_sync()
{
    try
    {
        // Sync the compute stream
        m_compute_stream << synchronize();
    }
    catch(const SimEngineException& e)
    {
        logger::error("SimEngine Sync Error: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
}
}  // namespace uipc::backend::luisa
