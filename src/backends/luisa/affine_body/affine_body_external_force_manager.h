#pragma once
/********************************************************************
 * @file affine_body_external_force_manager.h
 * @brief T6.1a — the ABD external-force manager (port of
 * src/backends/cuda/affine_body/affine_body_external_force_manager.h).
 *
 * The ABD end of the T4.2 generic external-force lifecycle: it derives from the
 * ported `ExternalForceReporter` (so `GlobalExternalForceManager` drives it) and
 * fans `init`/`step` out to its own `AffineBodyExternalForceReporter`s.
 * Per frame the engine calls:
 *   clear() -> zero `body_id_to_external_force` BEFORE the constraints write it
 *   step()  -> every reporter adds into `body_id_to_external_force`, then the
 *              manager writes `body_id_to_external_force_acc = M^-1 * F`
 * `UID = 666` is the reference's uid ("Same UID as external force constitution";
 * it collides with the frontend constitution uid on purpose, exactly like the FEM
 * manager's 671).
 *
 * Device mapping: `cuda_tool::BufferView<Vector12>` -> `lud::LBufferView<LVector12>`,
 * `cuda_tool::CBufferView<Matrix12x12>` -> `lud::LCBufferView<LMatrix12x12>`.
 *********************************************************************/
#include <external_force/external_force_reporter.h>
#include <affine_body/type_define.h>  // LVector12 (the global type layer has LVector3 only)

namespace uipc::backend::luisa
{
class AffineBodyDynamics;
class AffineBodyExternalForceReporter;

/**
 * @brief Manager for affine body external forces
 *
 * This reporter manages all affine-body-specific external force reporters.
 * It is responsible for:
 * 1. Clearing the external force buffer before constraints update
 * 2. Collecting external forces from all registered reporters
 * 3. Computing the external accelerations after all forces have been applied
 *
 * Hierarchy:
 * - AffineBodyExternalForceManager (this class, derives from ExternalForceReporter)
 *   - Manages AffineBodyExternalForceReporter instances
 *     - AffineBodyExternalBodyForce (applies force to body)
 */
class AffineBodyExternalForceManager final : public ExternalForceReporter
{
  public:
    using ExternalForceReporter::ExternalForceReporter;

    static constexpr U64 UID = 666;  // Same UID as external force constitution

    class Impl
    {
      public:
        AffineBodyDynamics* affine_body_dynamics = nullptr;
        SimSystemSlotCollection<AffineBodyExternalForceReporter> m_reporters;

        void clear();
        void step();
    };

    class ExternalForceInfo
    {
      public:
        ExternalForceInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }
        lud::LBufferView<LVector12> external_forces() noexcept;

      private:
        friend class AffineBodyExternalForceManager;
        Impl* m_impl = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual U64  get_uid() const noexcept override { return UID; }
    virtual void do_init() override;
    virtual void do_clear() override;  // Clear external force buffers
    virtual void do_step() override;   // Compute accelerations

    friend class AffineBodyExternalForceReporter;
    void register_reporter(AffineBodyExternalForceReporter* reporter);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
