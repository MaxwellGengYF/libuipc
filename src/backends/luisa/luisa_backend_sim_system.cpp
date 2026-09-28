#include <luisa_backend_sim_system.h>

#include <sim_engine.h>
#include <uipc/common/log.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(LuisaBackendSimSystem);

void LuisaBackendSimSystem::do_build()
{
    logger::info("[LuisaBackendSimSystem] do_build()");

    // Prove that the system graph resolves: require() must return this very
    // system (throws SimEngineException otherwise).
    auto& self        = require<LuisaBackendSimSystem>();
    m_self_resolved   = (&self == this);
    UIPC_ASSERT(m_self_resolved, "LuisaBackendSimSystem failed to resolve itself");

    // find() on a system that exists must also succeed.
    auto* found = find<LuisaBackendSimSystem>();
    UIPC_ASSERT(found == this, "LuisaBackendSimSystem find() did not return self");
}
}// namespace uipc::backend::luisa
