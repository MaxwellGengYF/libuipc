#pragma once
/********************************************************************
 * @file luisa_backend_sim_system.h
 * @brief The backend's own root SimSystem.
 *
 * Every backend owns one system that anchors the dependency graph (the CUDA
 * backend's equivalent is created through the same REGISTER_SIM_SYSTEM
 * mechanism). For the M1 shell it proves registration, instantiation and
 * find/require resolution inside the stub engine; Phase 4+ hangs the real
 * managers off it.
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
class LuisaBackendSimSystem final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    // Resolution probe used by the engine + tests: requires the global vertex
    // manager equivalent once it exists. For now it only validates that the
    // system itself resolves.
    [[nodiscard]] bool self_resolved() const noexcept { return m_self_resolved; }

  private:
    void do_build() override;

    bool m_self_resolved = false;
};
}// namespace uipc::backend::luisa
