#pragma once
/********************************************************************
 * @file affine_body_external_force_reporter.h
 * @brief T6.1a — the ABD external-force reporter base (port of
 * src/backends/cuda/affine_body/affine_body_external_force_reporter.h).
 *
 * A pure reporter base: a concrete reporter declares a uid and writes/adds its
 * forces into the manager's per-body buffer in `do_step(ExternalForceInfo&)`; the
 * manager then computes `a_ext = M^-1 F`.
 *
 * Specific implementations:
 * - AffineBodyExternalBodyForce: applies force directly to a body
 *
 * Lifecycle:
 * - do_init(): called once during initialization
 * - do_step(): called every frame AFTER constraints have updated
 *              `body_id_to_external_force`
 *********************************************************************/
#include <sim_system.h>
#include <affine_body/affine_body_external_force_manager.h>

namespace uipc::backend::luisa
{
class AffineBodyExternalForceManager;

class AffineBodyExternalForceReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    U64 uid() const noexcept;

    using ExternalForceInfo = AffineBodyExternalForceManager::ExternalForceInfo;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual U64  get_uid() const noexcept = 0;
    virtual void do_init() = 0;
    virtual void do_step(ExternalForceInfo& info) = 0;

  private:
    friend class AffineBodyExternalForceManager;
    virtual void do_build() override final;

    void init();                         // only be called by AffineBodyExternalForceManager
    void step(ExternalForceInfo& info);  // only be called by AffineBodyExternalForceManager

    SizeT m_index = ~0ull;
};
}  // namespace uipc::backend::luisa
