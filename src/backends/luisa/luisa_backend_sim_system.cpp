#include <luisa_backend_sim_system.h>
#include <sim_engine.h>
#include <uipc/common/log.h>

// T4.1: the root system hangs the global geometry managers off the system graph.
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
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

    // T4.1: the real managers. require<> creates the dependency edge (the manager
    // systems are built before this one), find<> proves pure lookup. The engine
    // drives their init() in SimEngine::init_scene() in the reference's order.
    auto& vertex_manager = require<GlobalVertexManager>();
    auto& body_manager   = require<GlobalBodyManager>();
    auto* surface_manager = find<GlobalSimplicialSurfaceManager>();

    UIPC_ASSERT(surface_manager != nullptr,
                "LuisaBackendSimSystem: find<GlobalSimplicialSurfaceManager>() failed");
    m_geometry_managers_resolved = true;

    logger::info(
        "[LuisaBackendSimSystem] geometry managers resolved: vertex={}, body={}, "
        "simplicial_surface={}",
        static_cast<const void*>(&vertex_manager),
        static_cast<const void*>(&body_manager),
        static_cast<const void*>(surface_manager));
}
}// namespace uipc::backend::luisa
