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
 * off the root system.
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

  private:
    void do_build() override;

    bool m_aux_resolved = false;
};
}// namespace uipc::backend::luisa
