#pragma once
/********************************************************************
 * @file finite_element_external_force_manager.h
 * @brief T5.3 — the FEM external-force manager (port of
 * src/backends/cuda/finite_element/finite_element_external_force_manager.h).
 *
 * The FEM end of the T4.2 generic external-force lifecycle: it derives from the
 * ported `ExternalForceReporter` (so `GlobalExternalForceManager` drives it) and
 * fans `init`/`step` out to its own `FiniteElementExternalForceReporter`s.
 *
 * Per frame the engine calls (`step_animation_and_external_forces`):
 *   clear() -> zero `vertex_external_forces` BEFORE the animation writes the
 *              force attributes
 *   step()  -> every reporter scatter-adds into `vertex_external_forces`, then the
 *              manager writes `vertex_external_force_accs = F / m`, and **0** when
 *              `m <= 0` (the reference's massless-vertex guard; see the .cpp)
 *
 * `UID = 671` is the reference's reporter uid (it collides with the frontend
 * constitution uid 671 of `uipc::constitution::FiniteElementExternalForce` on
 * purpose: the reference uses the same number for the constraint, the reporter and
 * the constitution).
 *
 * Device mapping: `cuda_tool::BufferView<Vector3>` -> `lud::LBufferView<LVector3>`,
 * `cuda_tool::CBufferView<Vector3>` -> `lud::LCBufferView<LVector3>`.
 *********************************************************************/
#include <external_force/external_force_reporter.h>

namespace uipc::backend::luisa
{
class FiniteElementMethod;
class FiniteElementExternalForceReporter;

/**
 * @brief Manager for finite element external forces
 *
 * This reporter manages all FEM-specific external force reporters.
 * It is responsible for:
 * 1. Clearing the external force buffer before constraints update
 * 2. Collecting external forces from all registered reporters
 * 3. Computing the external accelerations after all forces have been applied
 *
 * Hierarchy:
 * - FEMExternalForceManager (this class, derives from ExternalForceReporter)
 * - Manages FiniteElementExternalForceReporter instances
 * - FiniteElementExternalVertexForce (applies force to vertices, T5.4)
 */
class FEMExternalForceManager final : public ExternalForceReporter
{
  public:
    using ExternalForceReporter::ExternalForceReporter;

    static constexpr U64 UID = 671;

    class Impl
    {
      public:
        FiniteElementMethod* finite_element_method = nullptr;
        SimSystemSlotCollection<FiniteElementExternalForceReporter> m_reporters;

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
        lud::LBufferView<LVector3> external_forces() noexcept;

      private:
        friend class FEMExternalForceManager;
        Impl* m_impl = nullptr;
    };

  private:
    void do_build(BuildInfo& info) override;
    U64  get_uid() const noexcept override { return UID; }
    void do_init() override;
    void do_clear() override;
    void do_step() override;

    friend class FiniteElementExternalForceReporter;
    void register_reporter(FiniteElementExternalForceReporter* reporter);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
