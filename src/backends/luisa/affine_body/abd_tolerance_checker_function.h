#pragma once
/********************************************************************
 * @file affine_body/abd_tolerance_checker_function.h
 * @brief T6.4a — the ABD tolerance checker's kernel body, factored so the shipped
 *        shader and the unit test run the *same* code.
 *
 * Same recipe as T5.4b's `fem_diag_preconditioner_function.h` (cookbook §7.4 /
 * §7.20).  The reference
 * (`src/backends/cuda/affine_body/abd_tolerance_checker.cu`, 90 lines) is a
 * `.cu`-only class with one `__global__` kernel:
 *
 *   IndexT success_value = *success;      // a plain device read
 *   if(success_value == 0) return;        // "already failed stays failed"
 *   for(IndexT i = 3; i < 12; ++i)        // the *rotational* dofs only
 *       if(abs(dq[i]) > abs_tol)
 *       { cuda_tool::atomic_exch(success.data(), 0); break; }
 *
 * so `success` is a one-element device int, seeded **1** by the host before the
 * launch, and the whole dispatch is an OR over the bodies' `dq` blocks: the first
 * thread that finds a rotational component beyond `abs_tol` exchanges the flag to 0
 * and every later thread that reads the flag returns immediately.
 *
 * The decision is factored out (the *store* is not: the reference's `break` after
 * the exchange is value-free because the exchange is idempotent, and the flag read
 * is the caller's, exactly like the FEM sibling's `converged` early-out):
 *  * `exceeded_row` is the reference's nine-component test, as a predicate;
 *  * `failed_row` folds in the `success_value == 0` early return.
 * Both are generic over the scalar (host `double` / device `lcrt::Var<double>`) and
 * over the view flavour, so the unit test instantiates them with
 * `storage::memory` views.
 *
 * Mapping note: the reference's `abs()` is the port's `lps::abs` (exact for a
 * double), and its `>` is a strict comparison -- `|dq_i| == abs_tol` is
 * **converged** (the boundary the test pins).
 *********************************************************************/
#include <type_define.h>
#include <luisa_device/linear/linear.h>
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

namespace abd_tolerance_checker_detail
{
/// The reference's `cuda_tool::CBufferView<Vector12> dqs` (const device flavour)
/// and the host-memory flavour the unit test drives.
using DqDeviceView     = lla::LCDenseVectorDeviceView<lcrt::Var<LVector12>>;
using DqHostMemoryView = lla::LCDenseVectorHostMemoryView<LVector12>;

/// Read a `Vector12` out of a view: the `storage::memory` flavour already hands out
/// the value, the device flavour's `Expr<LVector12>` has to go through `load_vec12`
/// (type_define.h) -- the same helper pair the FEM sibling's block readers use.
template <typename View, typename I>
[[nodiscard]] inline auto load_vec12_at(View v, I&& i) noexcept
{
    if constexpr(View::storage_type::is_device)
    {
        return load_vec12(v.read(std::forward<I>(i)));
    }
    else
    {
        return v.read(std::forward<I>(i));
    }
}

/// The reference's inner loop, as a predicate:
/// `for(i = 3; i < 12; ++i) if(abs(dq[i]) > abs_tol) { fail; break; }`
///
/// The `break` is dropped: the folded OR is the same predicate (the exchange it
/// guards is idempotent), and the loop's nine comparisons are compile-time unrolled.
template <typename DqView, typename I, typename T>
[[nodiscard]] inline auto exceeded_row(DqView dqs, const I& i, const T& abs_tol) noexcept
{
    auto dq = load_vec12_at(dqs, std::forward<decltype(i)>(i));
    // components 0..2 are the translation, which the reference ignores
    return scalar_detail::pred_or(lps::abs(dq(3)) > abs_tol,
                                  lps::abs(dq(4)) > abs_tol,
                                  lps::abs(dq(5)) > abs_tol,
                                  lps::abs(dq(6)) > abs_tol,
                                  lps::abs(dq(7)) > abs_tol,
                                  lps::abs(dq(8)) > abs_tol,
                                  lps::abs(dq(9)) > abs_tol,
                                  lps::abs(dq(10)) > abs_tol,
                                  lps::abs(dq(11)) > abs_tol);
}

/// The reference's full per-thread decision:
/// `IndexT success_value = *success; if(success_value == 0) return;` followed by
/// `exceeded_row`.  `success_value` is read by the caller (the reference reads it
/// once per thread, before the loop).
template <typename DqView, typename I, typename T, typename S>
[[nodiscard]] inline auto failed_row(DqView       dqs,
                                     const I&     i,
                                     const T&     abs_tol,
                                     const S&     success_value) noexcept
{
    return scalar_detail::pred_and(success_value != 0,
                                   exceeded_row(dqs, i, abs_tol));
}
}  // namespace abd_tolerance_checker_detail
}  // namespace uipc::backend::luisa
