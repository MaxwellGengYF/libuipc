#pragma once
/********************************************************************
 * @file affine_body/abd_diag_preconditioner_function.h
 * @brief T6.4a — the ABD block-diagonal preconditioner's two kernel bodies,
 *        factored so the shipped shaders and the unit test run the *same* code.
 *
 * Same recipe as T5.4b's `fem_diag_preconditioner_function.h` (cookbook §7.4 /
 * §7.20): the kernel lambda builds the device views and calls one routine here,
 * which the unit test also instantiates with `storage::memory` views over host
 * arrays.  The reference
 * (`src/backends/cuda/affine_body/abd_diag_preconditioner.cu`, 87 lines) is a
 * `.cu`-only class with two `__global__` kernels:
 *
 *   do_assemble: `diag_inv(i) = cuda_tool::eigen::inverse(diag_hessian(i));`
 *                (one thread per *body*; `i >= n` returns -- the caller's dispatch)
 *   do_apply:    `if(*converged != 0) return;`
 *                `z.segment<12>(i * 12) = diag_inv(i) * r.segment<12>(i * 12);`
 *
 * Two differences from the FEM sibling that the reference's own code fixes:
 *  * `do_assemble` does **not** read the assembled BCOO matrix (`info.A()`); it
 *    reads `ABDLinearSubsystem::diag_hessian()`, the subsystem's *dense per-body*
 *    12x12 buffer that the assembly accumulated (one element per body, no `i == j`
 *    filtering, no dof-window arithmetic) -- so the row body is a plain elementwise
 *    inverse;
 *  * `do_apply`'s `i` is the **body index** (`i * 12`, not `i * 3`), and the
 *    r/z views it is handed are already the subsystem's window
 *    (`GlobalLinearSystem::Impl::apply_preconditioner` slices by
 *    `diag_dof_offsets_counts`), exactly like the FEM one.
 *
 * The `converged` early-out is *not* folded into the row routine: it is one scalar
 * read for the whole dispatch, so the kernel wraps its body in
 * `$if(converged->read(0) == 0)` and the host driver wraps its call in the same
 * test (the FEM sibling's convention).
 *
 * `lps::inverse` for a 12x12 is the new generic branch T6.4a added to the numerics
 * layer (`luisa_device/math/matrix_ops.h`): the reference's
 * `cuda_tool::eigen::inverse<Matrix12x12>` dispatches on the size and uses
 * `GaussEliminationInverse` for `N > 4` (the analytic adjugate forms only exist for
 * 2/3/4, cookbook §7.21 rule 11), so the port's `inverse` does the same.
 *********************************************************************/
#include <type_define.h>
#include <affine_body/abd_linear_subsystem_function.h>
#include <luisa_device/linear/linear.h>
#include <luisa_device/math/matrix_ops.h>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

namespace abd_diag_preconditioner_detail
{
/// `diag_inv` in the reference is `DeviceBuffer<Matrix12x12>`, so `diag_inv(i)` is
/// the view's *element* access (the same pair of flavours the FEM sibling uses).
using DiagInvDeviceView     = lla::LDenseVectorDeviceView<lcrt::Var<LMatrix12x12>>;
using DiagInvHostMemoryView = lla::LDenseVectorHostMemoryView<LMatrix12x12>;

/// Read a 12x12 block out of a view: the `storage::memory` flavour already hands
/// out the value, the device flavour's `Expr<LMatrix12x12>` has to go through
/// `load_mat12x12` (type_define.h).
template <typename View, typename I>
[[nodiscard]] inline auto load_mat12x12_block(View v, I&& i) noexcept
{
    if constexpr(View::storage_type::is_device)
    {
        return load_mat12x12(v.read(std::forward<I>(i)));
    }
    else
    {
        return v.read(std::forward<I>(i));
    }
}

/// `ABDDiagPreconditioner_do_assemble_kernel`'s row body:
/// `diag_inv(i) = cuda_tool::eigen::inverse(diag_hessian(i));`
template <typename DiagHessianView, typename I>
[[nodiscard]] inline auto assemble_row(DiagHessianView diag_hessian, const I& i) noexcept
{
    return lps::inverse(load_mat12x12_block(diag_hessian, i));
}

/// `ABDDiagPreconditioner_do_apply_kernel`'s row body:
/// `z.segment<12>(i * 12) = diag_inv(i) * r.segment<12>(i * 12);`
template <typename ZView, typename RView, typename DiagInvView, typename I>
void apply_row(ZView z, RView r, DiagInvView diag_inv, const I& i) noexcept
{
    constexpr auto kDof = abd_linear_subsystem_detail::kDofPerBody;
    auto r12 = r.template segment<kDof>(i * kDof).template to_vec<kDof>();
    auto H   = load_mat12x12_block(diag_inv, i);
    z.template segment<kDof>(i * kDof) = H * r12;
}
}  // namespace abd_diag_preconditioner_detail
}  // namespace uipc::backend::luisa
