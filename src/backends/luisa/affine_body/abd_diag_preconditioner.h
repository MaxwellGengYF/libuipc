#pragma once
/********************************************************************
 * @file affine_body/abd_diag_preconditioner.h
 * @brief T6.4a — the ABD block-diagonal preconditioner (port of
 *        src/backends/cuda/affine_body/abd_diag_preconditioner.cu, 87 lines).
 *
 * The ABD sibling of the already-ported `FEMDiagPreconditioner` (T5.4b): a
 * `LocalPreconditioner` bound (through `BuildInfo::connect`) to
 * `ABDLinearSubsystem`, whose `do_assemble` inverts the subsystem's *dense
 * per-body* 12x12 diagonal Hessian into `diag_inv` and whose `do_apply` computes
 * `z_i = diag_inv(i) * r_i` over the subsystem's dof window, **returning early
 * when `converged != 0`**.
 *
 * The reference declares the class *and* its two kernels inside the `.cu`.  The
 * port keeps the reference's public surface (the two pointers, `diag_inv`, the four
 * hooks) in this header, the kernels + `REGISTER_SIM_SYSTEM` in the `.cpp`, and the
 * *bodies* in `abd_diag_preconditioner_function.h` -- exactly like T5.4b's FEM pair,
 * so the unit test can register the *real* preconditioner into a standalone
 * `GlobalLinearSystem::Impl` and drive `assemble`/`apply` directly (cookbook
 * §7.20 rule 1's harness).
 *
 * There is no "mas" alternative to defer to here: the reference's ABD side has no
 * such selection (only the FEM one is gated on
 * `linear_system/fem_preconditioner == "mas"`), so `do_build` is the reference's
 * three statements and nothing else.
 *********************************************************************/
#include <linear_system/local_preconditioner.h>
#include <affine_body/abd_linear_subsystem.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    /// One inverted 12x12 diagonal block per ABD body, i.e. one per DoF block run.
    lud::LDeviceBuffer<LMatrix12x12> diag_inv;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override;
    void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override;
};
}  // namespace uipc::backend::luisa
