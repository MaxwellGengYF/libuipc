#pragma once
/********************************************************************
 * @file finite_element_external_force_reporter.h
 * @brief T5.3 — the FEM external-force reporter base (port of
 * src/backends/cuda/finite_element/finite_element_external_force_reporter.h).
 *
 * A pure reporter base: a concrete reporter declares a uid and scatter-adds its
 * forces into the manager's per-vertex buffer in `do_step(ExternalForceInfo&)`.
 * The only reference implementation is
 * `FiniteElementExternalVertexForce` (T5.4, with its constraint); T5.3's test
 * registers a test-local reporter with a reserved uid instead (see the cookbook
 * §7.18 uid table and luisa_fem_dynamics_test.cpp).
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/finite_element_external_force_manager.h>

namespace uipc::backend::luisa
{
class FEMExternalForceManager;

/**
 * @brief Base class for finite element external force reporters
 *
 * Reporters scatter-add external forces into the per-vertex force buffer.
 * The manager then computes per-vertex accelerations a = F/m.
 *
 * Lifecycle:
 * - do_init(): Called once during initialization
 * - do_step(): Called every frame to scatter forces into the shared buffer
 */
class FiniteElementExternalForceReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    U64 uid() const noexcept;

    using ExternalForceInfo = FEMExternalForceManager::ExternalForceInfo;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual U64  get_uid() const noexcept = 0;
    virtual void do_init() = 0;
    virtual void do_step(ExternalForceInfo& info) = 0;

  private:
    friend class FEMExternalForceManager;
    virtual void do_build() override final;

    void init();
    void step(ExternalForceInfo& info);

    SizeT m_index = ~0ull;
};
}  // namespace uipc::backend::luisa
