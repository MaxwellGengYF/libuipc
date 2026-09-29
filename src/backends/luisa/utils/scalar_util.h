#pragma once
/********************************************************************
 * @file scalar_util.h
 * @brief T5.2 (batch b) — host/device-generic scalar helpers for the shell
 *        constitutions.
 *
 * The shell utilities and constitutions are written as *one* implementation over
 * the scalar type `S` (host `double` / device `lcrt::Var<double>`), and they need
 * three things the numerics layer deliberately does not offer as a single call:
 *
 *  1. `p ? t : f` on possibly-DSL scalars.  `lps::select` is host-only by design
 *     (LC's own `select` has a custom-struct overload that would also match a
 *     container, cookbook §7.6 trap 3) and LC's `select` is reachable by ADL only,
 *     so a *qualified* call never finds it — the same measurement T5.2 batch a
 *     recorded for the matrix assembler's nested `ProxyRange` (cookbook §7.17
 *     rule 4, measured C2672/C3861).  The dispatcher below is that same helper,
 *     factored out instead of being written a fourth time.
 *  2. Predicate folding.  LC requires `&`/`|` for DSL scalars and rejects
 *     `&&`/`||` (cookbook §7.6 trap 2); on the host `bool & bool` promotes to
 *     `int`, so a third operand would mix `int` and `bool` (MSVC C4805).
 *  3. `isfinite(x)` for possibly-DSL scalars.  LC exposes `isnan`/`isinf` as DSL
 *     intrinsics but no `isfinite`, and the `x - x == 0` trick would rest on the
 *     fast-math assumption that the shader compiler may fold away (LC compiles its
 *     CUDA shaders with `-use_fast_math`), hence `!(isnan(x) | isinf(x))`.
 *
 * Nothing here is shell specific; batch c (rods/springs) can reuse it.
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/math.h>

#include <cmath>
#include <utility>

namespace uipc::backend::luisa::scalar_detail
{
/**
 * @brief The sweep count the *shell* constitutions pass to `lps::make_spd_inplace` for
 *        their 12x12 element Hessians.
 *
 * The reference's `utils/make_spd.h` is a *tolerance-driven* Jacobi loop
 * (`while(calc_sumDiagOff(M) > 1e-6)`), i.e. its sweep count is device data — a port
 * needs a fixed count (RULE M2), and T2.2 pinned the layer's default to 20.  For the
 * 12x12 Hessian of a bending stencil that default makes LC's nvrtc spend **~90 min** on
 * a *single* shader and (measured) crash the compiler process, so the shells pass an
 * explicit 8, which is within the range the reference's own loop needs for a 12x12 and
 * still projects to machine precision (the unit test's eigenvalue-clamp identity holds
 * to ~1e-10 relative).  Reported as a deviation in the T5.2 batch-b report and in
 * cookbook §7.17 rule 25; the 9x9 membrane Hessian keeps the layer default.
 */
inline constexpr int kShellMakeSpdSweeps = 8;

/// `p ? t : f` for a host scalar or a DSL scalar (`p` may be the device predicate).
template <typename F, typename T, typename P>
[[nodiscard]] inline auto select_scalar(F&& f, T&& t, P&& p) noexcept
{
    if constexpr(lps::is_dsl_value_v<F> || lps::is_dsl_value_v<T> || lps::is_dsl_value_v<P>)
        return lcrt::select(std::forward<F>(f), std::forward<T>(t), std::forward<P>(p));
    else
        return lps::select(std::forward<F>(f), std::forward<T>(t), std::forward<P>(p));
}

/// Logical AND of two or more predicates, host `bool` and DSL boolean alike.
template <typename A, typename B>
[[nodiscard]] inline auto pred_and(A&& a, B&& b) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<A>>
                 && std::is_arithmetic_v<std::remove_cvref_t<B>>)
        return static_cast<bool>(a) && static_cast<bool>(b);
    else
        return static_cast<A&&>(a) & static_cast<B&&>(b);
}

template <typename A, typename B, typename... Rest>
[[nodiscard]] inline auto pred_and(A&& a, B&& b, Rest&&... rest) noexcept
{
    return pred_and(pred_and(static_cast<A&&>(a), static_cast<B&&>(b)),
                    static_cast<Rest&&>(rest)...);
}

/// Logical OR of two or more predicates (see `pred_and`).
template <typename A, typename B>
[[nodiscard]] inline auto pred_or(A&& a, B&& b) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<A>>
                 && std::is_arithmetic_v<std::remove_cvref_t<B>>)
        return static_cast<bool>(a) || static_cast<bool>(b);
    else
        return static_cast<A&&>(a) | static_cast<B&&>(b);
}

template <typename A, typename B, typename... Rest>
[[nodiscard]] inline auto pred_or(A&& a, B&& b, Rest&&... rest) noexcept
{
    return pred_or(pred_or(static_cast<A&&>(a), static_cast<B&&>(b)),
                   static_cast<Rest&&>(rest)...);
}

/// `isfinite` of a host or a DSL scalar (see the file comment for why not `x - x`).
template <typename S>
[[nodiscard]] inline auto is_finite_scalar(const S& v) noexcept
{
    if constexpr(lps::is_dsl_value_v<S>)
        return !(lcrt::isnan(v) | lcrt::isinf(v));
    else
        return std::isfinite(v) != 0;
}

/// `isfinite` of a 3-vector (the reference's `is_finite_vec3`).
template <typename S>
[[nodiscard]] inline auto is_finite_vec3(const lps::LVec<S, 3>& v) noexcept
{
    return pred_and(is_finite_scalar(v(0)),
                    is_finite_scalar(v(1)),
                    is_finite_scalar(v(2)));
}

/// Element-wise `p ? t : f` of two equal-dim vectors.  `lps::select` refuses
/// containers on purpose (LC's own `select` has a custom-struct overload that would
/// happily build a `Var<LVec<...>>`, cookbook §7.6 trap 3), so a container select is
/// per element — the same formulation `utils/matrix_assembler.h` uses.
template <typename T, int N, typename P>
[[nodiscard]] inline lps::LVec<T, N> select_container(const lps::LVec<T, N>& f,
                                                     const lps::LVec<T, N>& t,
                                                     const P&               p) noexcept
{
    return lps::make_vec<T, N>([&](int i) noexcept { return select_scalar(f(i), t(i), p); });
}

template <typename T, int R, int C, typename P>
[[nodiscard]] inline lps::LMat<T, R, C> select_container(const lps::LMat<T, R, C>& f,
                                                        const lps::LMat<T, R, C>& t,
                                                        const P&                  p) noexcept
{
    return lps::make_mat<T, R, C>([&](int r, int c) noexcept
                                  { return select_scalar(f(r, c), t(r, c), p); });
}
}  // namespace uipc::backend::luisa::scalar_detail
