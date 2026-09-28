#pragma once
/********************************************************************
 * @file luisa_backend_sim_system.h
 * @brief The backend's own root SimSystem + a tiny auxiliary system.
 *
 * Every backend owns one root system that anchors the dependency graph.
 * `LuisaAuxSimSystem` exists so the root system can prove that the system graph
 * actually resolves: `require<LuisaAuxSimSystem>()` creates a real strong
 * dependency edge (a self-require would be reported as a cycle), and
 * `find<LuisaAuxSimSystem>()` proves lookup. Phase 4+ hangs the real managers
 * off the root system: since T4.1 the root system also requires the three global
 * geometry managers, which is what makes them part of the backend's system graph
 * (the reference engine does the equivalent `require<>()` calls in `build()`).
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
class LuisaAuxSimSystem final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    [[nodiscard]] bool built() const noexcept { return m_built; }

  private:
    void         do_build() override;
    bool         m_built = false;
};

class LuisaBackendSimSystem final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    // Resolution probe used by the engine + tests.
    [[nodiscard]] bool aux_resolved() const noexcept { return m_aux_resolved; }
    // T4.1: true once the global geometry managers resolved through the root
    // system (require<GlobalVertexManager>/require<GlobalBodyManager>/
    // find<GlobalSimplicialSurfaceManager>).
    [[nodiscard]] bool geometry_managers_resolved() const noexcept
    {
        return m_geometry_managers_resolved;
    }

  private:
    void do_build() override;

    bool m_aux_resolved             = false;
    bool m_geometry_managers_resolved = false;
};
}// namespace uipc::backend::luisa
