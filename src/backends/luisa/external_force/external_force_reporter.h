#pragma once
/********************************************************************
 * @file external_force_reporter.h
 * @brief Port of src/backends/cuda/external_force/external_force_reporter.h
 *        (T4.2). Identical shape; the only change is the namespace.
 *
 * The lifecycle (init / clear / step) is private and driven by
 * GlobalExternalForceManager; a concrete reporter only implements the hooks.
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
class GlobalExternalForceManager;

/**
 * @brief Base class for external force reporters (agnostic to specific dynamics type)
 *
 * This is a general interface for different types of external force reporters.
 * Specific implementations (e.g., AffineBodyExternalForceManager, FEMExternalForceManager)
 * should derive from this class.
 */
class ExternalForceReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    U64 uid() const noexcept;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual U64  get_uid() const noexcept  = 0;
    virtual void do_init()                 = 0;
    virtual void do_clear()                = 0;  // Clear force buffers
    virtual void do_step()                 = 0;  // Compute accelerations

  private:
    friend class GlobalExternalForceManager;
    virtual void do_build() override final;

    void init();
    void clear();
    void step();

    SizeT m_index = ~0ull;
};
}  // namespace uipc::backend::luisa
