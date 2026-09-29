#pragma once
/********************************************************************
 * @file affine_body/constitutions/joint_limit_penalty.h
 * @brief T6.3b2 — the cubic joint-limit penalty (port of
 *        src/backends/cuda/affine_body/constitutions/joint_limit_penalty.h).
 *
 * A *header-only* device function set (the reference includes it from both the
 * prismatic and the revolute `.cu`): given a scalar `x` and the interval
 * `[lower, upper]`, it returns the penalty `E`, `dE_dx`, `d2E_dx2` of
 *
 *     E(x) = 0                                     if lower <= x <= upper
 *          = strength * (|x - bound| / w)^3         with `w = upper - lower`
 *
 * i.e. a *cubic* one-sided barrier outside the interval, growing as the cube of the
 * *normalised* excursion.  When `w == 0` (a zero-width range) the normalisation is
 * dropped and the same cubic is used on the raw excursion — reachable from the
 * frontend whenever a user passes `lower == upper`, which
 * `AffineBodyPrismaticJointLimit::apply_to` allows (it only requires
 * `lower <= upper`).
 *
 * The template parameters `ComputeE` / `ComputeDerivatives` are the reference's
 * compile-time switches: `eval_penalty` computes all three, `eval_penalty_energy`
 * only `E`, `eval_penalty_derivatives` only the two derivatives.  They are kept
 * because the *zeroing* of the not-requested outputs is part of the contract (the
 * kernels rely on it: the limit's energy kernel ignores the derivatives, and its
 * gradient/Hessian kernels ignore `E`).
 *
 * PORT MAPPING (the T5.x "one generic routine over the scalar", cookbook §7.16 rule 1)
 *  * `UIPC_GENERIC` -> `template <typename T>` (`double` host / `lcrt::Var<double>`
 *    device); `static_cast<T>(0/1/3/6)` -> `lps::zero_of`/`one_of`/`make_scalar`
 *    (a `static_cast` is not available in the DSL); the `T`-arithmetic `*`/`-`/`/`
 *    is spelled with `lps::detail::{mul,sub,div,neg}` in the reference's own
 *    *association order* (RULE M1; a `-three * s * d * d` is
 *    `(((-three) * s) * d) * d`), so the values are bit-identical.
 *  * the `if constexpr(RUNTIME_CHECK) UIPC_KERNEL_ASSERT(lower <= upper, ...)` keeps
 *    its placement with the `utils/codim_thickness.h` host/device split
 *    (`lcrt::device_assert` / `UIPC_ASSERT`, §7.16 rule 6).  DEVIATION: the
 *    device_assert carries a *literal* message — LC's `device_assert` takes no
 *    printf-style arguments, so the two values are not interpolated on the device
 *    path; the host path keeps them.
 *  * **DEVIATION (the T2.2 `svd.h` precedent, RULE M2): the reference's
 *    device-driven branches and its two early `return`s are rewritten as `select`s,
 *    so that ONE body is valid on the host and in the DSL.**  The reference's own
 *    structure is
 *
 *        if(inside) -> zeros and return
 *        if(w == 0) { if(x > upper) A else B ; return }
 *        if(x > upper) C else D
 *
 *    LC's `Expr<bool>` cannot drive a plain `if` in a *host-and-device* template
 *    (`$if`/`$else` would make the routine device-only, exactly the trade-off
 *    `svd.h` documents), so the port computes all four candidate values and
 *    *selects* on the same three predicates.  `select` is a real select
 *    instruction, the *not-taken* arm's value is discarded (including a NaN from
 *    `inv_w = 1/0` in the `w == 0` case), and every selected expression is the
 *    reference's own association order — so the result is bitwise the reference's
 *    for every reachable input.  The test pins that with the reference-shaped
 *    (`if`-structured, host-only) oracle of the same formulas, plus an FD check.
 *********************************************************************/
#include <type_define.h>
#include <uipc/common/config.h>
#include <luisa_device/math/math.h>
#include <utils/scalar_util.h>

namespace uipc::backend::luisa::joint_limit
{
namespace detail
{
    /// The reference's `if constexpr(RUNTIME_CHECK) { UIPC_KERNEL_ASSERT(...) }`
    /// (the `utils/codim_thickness.h` helper, re-spelled for this file).
    template <typename Cond>
    inline void assert_lower_le_upper(Cond&& cond) noexcept
    {
        if constexpr(lps::is_dsl_value_v<std::remove_cvref_t<Cond>>)
        {
            lcrt::device_assert(cond, "joint_limit::eval_penalty requires lower <= upper");
        }
        else
        {
            UIPC_ASSERT(static_cast<bool>(cond),
                        "joint_limit::eval_penalty requires lower <= upper");
        }
    }

    template <bool ComputeE, bool ComputeDerivatives, typename T>
    inline void eval_penalty_impl(
        const T& x, const T& lower, const T& upper, const T& strength, T& E, T& dE_dx, T& d2E_dx2)
    {
        using lps::detail::add;
        using lps::detail::div;
        using lps::detail::mul;
        using lps::detail::neg;
        using lps::detail::sub;

        const T zero  = lps::zero_of<T>();
        const T one   = lps::one_of<T>();
        const T three = lps::make_scalar<T>(3.0);
        const T six   = lps::make_scalar<T>(6.0);
        // the reference's `-three`: a *literal*, not `neg(three)`, so that the
        // association order of `-three * strength * ...` is reproduced exactly
        const T minus_three = lps::make_scalar<T>(-3.0);

        if constexpr(uipc::RUNTIME_CHECK)
        {
            assert_lower_le_upper(lower <= upper);
        }

        if constexpr(ComputeE)
            E = zero;
        if constexpr(ComputeDerivatives)
        {
            dE_dx   = zero;
            d2E_dx2 = zero;
        }

        // `x >= lower && x <= upper` (the reference's early out) and `x > upper`:
        // device predicates, so the *results* are selected instead of returning early
        // (`pred_and` / `pred_or` are the host-`bool`/DSL spellings, RULE M2).
        const auto over_upper = (x > upper);
        const auto inside     = scalar_detail::pred_and(x >= lower, x <= upper);

        // ---- the `w == 0` branch (the raw excursion) -------------------------
        // reference: `d = x - upper` / `d = lower - x`, then
        //   E = strength * d * d * d ; dE_dx = (+/-)three * strength * d * d ;
        //   d2E_dx2 = six * strength * d
        const T d_up = sub(x, upper);
        const T d_lo = sub(lower, x);
        const T d    = scalar_detail::select_scalar(d_lo, d_up, over_upper);
        const T E_w0 = mul(mul(mul(strength, d), d), d);
        const T g_w0 = mul(mul(mul(scalar_detail::select_scalar(minus_three, three, over_upper),
                                   strength),
                               d),
                           d);
        const T h_w0 = mul(mul(six, strength), d);

        // ---- the `w != 0` branch (the normalised excursion) ------------------
        // reference: `inv_w = 1/w`, `r = (x - upper) * inv_w` / `(lower - x) * inv_w`,
        //   E = strength * r * r * r ; dE_dx = (+/-)three * strength * r * r * inv_w ;
        //   d2E_dx2 = six * strength * r * inv_w * inv_w
        const T w     = sub(upper, lower);
        const T inv_w = div(one, w);
        const T r =
            mul(scalar_detail::select_scalar(sub(lower, x), d_up, over_upper), inv_w);
        const T E_wnz = mul(mul(mul(strength, r), r), r);
        const T g_wnz =
            mul(mul(mul(mul(scalar_detail::select_scalar(minus_three, three, over_upper),
                           strength),
                       r),
                   r),
                inv_w);
        const T h_wnz = mul(mul(mul(mul(six, strength), r), inv_w), inv_w);

        const auto w_is_zero = (w == zero);
        const T    E_out     = scalar_detail::select_scalar(E_wnz, E_w0, w_is_zero);
        const T    g_out     = scalar_detail::select_scalar(g_wnz, g_w0, w_is_zero);
        const T    h_out     = scalar_detail::select_scalar(h_wnz, h_w0, w_is_zero);

        if constexpr(ComputeE)
            E = scalar_detail::select_scalar(E_out, zero, inside);
        if constexpr(ComputeDerivatives)
        {
            // NOTE: the reference zeroes dE_dx / d2E_dx2 *before* its branches, so the
            // `inside` case is the literal zero here too.
            dE_dx   = scalar_detail::select_scalar(g_out, zero, inside);
            d2E_dx2 = scalar_detail::select_scalar(h_out, zero, inside);
        }
    }
}  // namespace detail

template <typename T>
inline void eval_penalty(
    const T& x, const T& lower, const T& upper, const T& strength, T& E, T& dE_dx, T& d2E_dx2)
{
    detail::eval_penalty_impl<true, true>(x, lower, upper, strength, E, dE_dx, d2E_dx2);
}

template <typename T>
inline T eval_penalty_energy(const T& x, const T& lower, const T& upper, const T& strength)
{
    T E       = lps::zero_of<T>();
    T dE_dx   = lps::zero_of<T>();
    T d2E_dx2 = lps::zero_of<T>();
    detail::eval_penalty_impl<true, false>(x, lower, upper, strength, E, dE_dx, d2E_dx2);
    return E;
}

template <typename T>
inline void eval_penalty_derivatives(
    const T& x, const T& lower, const T& upper, const T& strength, T& dE_dx, T& d2E_dx2)
{
    T E = lps::zero_of<T>();
    detail::eval_penalty_impl<false, true>(x, lower, upper, strength, E, dE_dx, d2E_dx2);
}
}  // namespace uipc::backend::luisa::joint_limit
