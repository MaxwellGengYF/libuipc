#pragma once
/********************************************************************
 * @file fem_diag_preconditioner.h
 * @brief T5.4b — the FEM block-diagonal preconditioner (port of
 *        src/backends/cuda/finite_element/fem_diag_preconditioner.cu).
 *
 * The reference declares the class *and* its two kernels inside the `.cu`.  The
 * port keeps the reference's public surface (the three `require<>`d systems, the
 * `diag_inv` buffer, the four hooks) in this header, the kernels + the
 * `REGISTER_SIM_SYSTEM` in the `.cpp`, and the *bodies* in
 * `fem_diag_preconditioner_function.h` — exactly like T5.2's constitutions
 * (cookbook §7.17 rule 11) and T5.3's `.cu`-only classes, so the unit test can
 * register the *real* preconditioner into a standalone `GlobalLinearSystem::Impl`
 * and drive `assemble`/`apply` directly.
 *
 * WHAT IT IS
 * ----------
 * A `LocalPreconditioner` bound (through `BuildInfo::connect`) to
 * `FEMLinearSubsystem`: `do_assemble` inverts the FEM slice's *diagonal* BCOO
 * blocks into `diag_inv` (one 3x3 block per vertex), and `do_apply` computes
 * `z_i = diag_inv(i) * r_i` over the subsystem's dof window, **returning early
 * when `converged != 0`** (the reference's `if(*converged != 0) return;`).
 *
 * The MAS alternative (`FEMMASPreconditioner`) is selected by the scene config
 * `linear_system/fem_preconditioner == "mas"`; the reference *defers* to it by
 * throwing `SimSystemException`, which the port preserves verbatim (its own
 * `mas_preconditioner_engine` is M6's task, see the progress doc's M5 closure).
 *********************************************************************/
#include <linear_system/local_preconditioner.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
class FEMDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    FiniteElementMethod* finite_element_method = nullptr;
    GlobalLinearSystem*  global_linear_system  = nullptr;
    FEMLinearSubsystem*  fem_linear_subsystem  = nullptr;

    /// One inverted 3x3 diagonal block per (FEM) vertex, i.e. one per DoF block.
    lud::LDeviceBuffer<LMatrix3x3> diag_inv;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override;
    void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override;
};
}  // namespace uipc::backend::luisa
