#include <luisa_backend_sim_system.h>

#include <sim_engine.h>
#include <uipc/common/log.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(LuisaAuxSimSystem);
REGISTER_SIM_SYSTEM(LuisaBackendSimSystem);

void LuisaAuxSimSystem::do_build()
{
    m_built = true;
}

void LuisaBackendSimSystem::do_build()
{
    logger::info("[LuisaBackendSimSystem] do_build()");

    // find() must locate an existing sibling without creating a dependency.
    auto* found = find<LuisaAuxSimSystem>();
    UIPC_ASSERT(found != nullptr, "LuisaBackendSimSystem: find<LuisaAuxSimSystem>() failed");

    // require() creates a strong dependency edge and must resolve to the same
    // object. (Requiring *self* would be reported as a dependency cycle, which
    // is exactly what this replaces.)
    auto& required = require<LuisaAuxSimSystem>();
    m_aux_resolved = (&required == found);
    UIPC_ASSERT(m_aux_resolved, "LuisaBackendSimSystem: require<LuisaAuxSimSystem>() mismatch");
}
}// namespace uipc::backend::luisa
