#pragma once
/********************************************************************
 * @file math.h
 * @brief T2.1 — scalar math for the *generic* (host/device) numerics layer.
 *
 * The LuisaCompute backend has no separate device compilation: kernel bodies are
 * ordinary host C++ that *builds IR*, so a numeric routine is written **once** as
 * a template over the scalar type:
 *
 *   T = double / float          -> evaluated on the host
 *   T = lcrt::Var<double> ...   -> evaluated on the device through the LC DSL
 *
 * Everything in this header follows the same dispatch rule: for an arithmetic
 * type forward to `std::`, otherwise forward to LuisaCompute's DSL intrinsic of
 * the same name.
 *
 * Two LC-specific facts drive the design:
 *  - `lcrt::Expr<T>` has a *deleted* copy assignment (single assignment), while
 *    `lcrt::Var<T>` is assignable (it emits `assign`).  Computed DSL values are
 *    therefore materialised into `Var<T>` — see `storage_of_t` below.
 *  - a DSL boolean (`Expr<bool>`) cannot be used as a C++ `if` condition, so
 *    comparisons must never leak into generic host-side control flow.  `min`,
 *    `max`, `clamp`, `sign`, `abs` ... all dispatch to the DSL intrinsic for
 *    non-arithmetic types precisely for this reason.
 *
 * Include only through <luisa_device/math/math.h>; the vector/matrix headers
 * build on top of it.
 *********************************************************************/
#include <luisa_device/compat.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace uipc::backend::luisa::luisa_device::math
{
/// True when `T` is a *device* scalar/expression type rather than a host value.
template <typename T>
inline constexpr bool is_device_scalar_v = !std::is_arithmetic_v<std::remove_cvref_t<T>>;

/// True when `T` is an LC DSL value (expression, variable, reference, ...).
template <typename T>
inline constexpr bool is_dsl_value_v = lcrt::is_dsl_v<std::remove_cvref_t<T>>;

// ---------------------------------------------------------------------------
//  Element storage type
// ---------------------------------------------------------------------------
namespace detail
{
template <typename T>
struct storage_of
{
    using type = T;
};

// LC's Expr is single-assignment; materialise DSL values into Var<T> locals so
// that containers of them stay mutable (exactly what `Float x = a + b;` does).
template <typename T>
    requires(!std::is_arithmetic_v<T>)
struct storage_of<T>
{
    using type = lcrt::Var<lcrt::expr_value_t<T>>;
};
}// namespace detail

/**
 * @brief Element type used to *store* the result of a computation.
 *
 * host: `storage_of_t<double>` = `double`
 * device: `storage_of_t<Expr<double>>` = `Var<double>`
 *
 * It is idempotent, so it can be applied to already-stored types.
 */
template <typename T>
using storage_of_t = typename detail::storage_of<std::remove_cvref_t<T>>::type;

// ---------------------------------------------------------------------------
//  Scalar arithmetic trampolines
// ---------------------------------------------------------------------------
// LuisaCompute declares its DSL *scalar* operators in the **global** namespace
// (`luisa/dsl/operators.h`, `LUISA_MAKE_GLOBAL_DSL_BINARY_OP`).  They are only
// reachable through unqualified lookup that actually reaches the global
// namespace, and no built-in candidate exists for a DSL type.  Inside
// `namespace math` our own vector/matrix overloads stop that lookup, so every
// arithmetic operation on a possibly-DSL *scalar* in this layer is routed through
// these trampolines: DSL values go to the explicit global operator, host values
// to the built-in operators (always candidates for operator syntax).
//
// The assumption about LC is asserted below so that an upstream move of those
// operators into a namespace fails loudly here instead of subtly at a call site.
static_assert(requires(lcrt::Var<double> a, lcrt::Var<double> b) { ::operator+(a, b); },
              "LuisaCompute's DSL scalar operators are expected at global scope "
              "(luisa/dsl/operators.h) -- update the math shim's scalar trampolines "
              "if that ever changes.");

namespace detail
{
#define UIPC_LUISA_SCALAR_TRAMPOLINE(NAME, OP)                             \
    template <typename A, typename B>                                      \
    [[nodiscard]] inline auto NAME(A&& a, B&& b) noexcept                  \
    {                                                                      \
        if constexpr(is_dsl_value_v<A> || is_dsl_value_v<B>)               \
            return ::operator OP(std::forward<A>(a), std::forward<B>(b));  \
        else                                                                   \
            return std::forward<A>(a) OP std::forward<B>(b);               \
    }

UIPC_LUISA_SCALAR_TRAMPOLINE(add, +)
UIPC_LUISA_SCALAR_TRAMPOLINE(sub, -)
UIPC_LUISA_SCALAR_TRAMPOLINE(mul, *)
UIPC_LUISA_SCALAR_TRAMPOLINE(div, /)

#undef UIPC_LUISA_SCALAR_TRAMPOLINE

template <typename A>
[[nodiscard]] inline auto neg(A&& a) noexcept
{
    if constexpr(is_dsl_value_v<A>)
        return ::operator-(std::forward<A>(a));
    else
        return -std::forward<A>(a);
}

/// Result element type of combining two scalar element types.  All four
/// arithmetic operators share it, so `add` is used for the computation.
template <typename A, typename B>
using scalar_result_t = storage_of_t<decltype(add(std::declval<A>(), std::declval<B>()))>;
}// namespace detail

// ---------------------------------------------------------------------------
//  Literals
// ---------------------------------------------------------------------------
/// Scale of a DSL/host scalar (i.e. `float` for `Var<float>`).
template <typename T>
using scalar_value_t = std::conditional_t<std::is_arithmetic_v<std::remove_cvref_t<T>>,
                                          std::remove_cvref_t<T>,
                                          lcrt::expr_value_t<std::remove_cvref_t<T>>>;

/// Zero / one of a generic scalar type (host literal or DSL literal).
template <typename T>
[[nodiscard]] inline auto zero_of() noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<T>>)
        return static_cast<std::remove_cvref_t<T>>(0);
    else
        return std::remove_cvref_t<T>{static_cast<scalar_value_t<T>>(0)};
}

template <typename T>
[[nodiscard]] inline auto one_of() noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<T>>)
        return static_cast<std::remove_cvref_t<T>>(1);
    else
        return std::remove_cvref_t<T>{static_cast<scalar_value_t<T>>(1)};
}

/// Convert a host literal to the generic scalar type `T` (identity on the host).
template <typename T, typename S>
    requires std::is_arithmetic_v<S>
[[nodiscard]] inline auto make_scalar(S s) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<T>>)
        return static_cast<std::remove_cvref_t<T>>(s);
    else
        return std::remove_cvref_t<T>{static_cast<scalar_value_t<T>>(s)};
}

// ---------------------------------------------------------------------------
//  Scalar math
// ---------------------------------------------------------------------------
// Each function is written once: `std::` for host scalars, LC's DSL intrinsic
// otherwise.  The `using std::NAME;` inside the host branch is the standard
// two-phase-lookup idiom (it hides the enclosing function, never recurses).
#define UIPC_LUISA_MATH_UNARY(NAME)                                          \
    template <typename X>                                                    \
    [[nodiscard]] inline auto NAME(X&& x) noexcept                           \
    {                                                                        \
        if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)           \
        {                                                                    \
            using std::NAME;                                                 \
            return NAME(std::forward<X>(x));                                 \
        }                                                                    \
        else                                                                 \
        {                                                                    \
            return lcrt::NAME(std::forward<X>(x));                           \
        }                                                                    \
    }

UIPC_LUISA_MATH_UNARY(abs)
UIPC_LUISA_MATH_UNARY(sqrt)
UIPC_LUISA_MATH_UNARY(exp)
UIPC_LUISA_MATH_UNARY(exp2)
UIPC_LUISA_MATH_UNARY(log)
UIPC_LUISA_MATH_UNARY(log2)
UIPC_LUISA_MATH_UNARY(log10)
UIPC_LUISA_MATH_UNARY(sin)
UIPC_LUISA_MATH_UNARY(cos)
UIPC_LUISA_MATH_UNARY(tan)
UIPC_LUISA_MATH_UNARY(asin)
UIPC_LUISA_MATH_UNARY(acos)
UIPC_LUISA_MATH_UNARY(atan)
UIPC_LUISA_MATH_UNARY(sinh)
UIPC_LUISA_MATH_UNARY(cosh)
UIPC_LUISA_MATH_UNARY(tanh)
UIPC_LUISA_MATH_UNARY(asinh)
UIPC_LUISA_MATH_UNARY(acosh)
UIPC_LUISA_MATH_UNARY(atanh)
UIPC_LUISA_MATH_UNARY(floor)
UIPC_LUISA_MATH_UNARY(ceil)
UIPC_LUISA_MATH_UNARY(round)
UIPC_LUISA_MATH_UNARY(trunc)
UIPC_LUISA_MATH_UNARY(isnan)
UIPC_LUISA_MATH_UNARY(isinf)

#undef UIPC_LUISA_MATH_UNARY

// The four below have a DSL intrinsic but **no** `std::` counterpart, so they
// cannot use the macro (its `using std::NAME;` would be looked up eagerly).

/// Fractional part (DSL intrinsic; `x - floor(x)` on the host).
template <typename X>
[[nodiscard]] inline auto fract(X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
    {
        using S = std::remove_cvref_t<X>;
        return static_cast<S>(x - std::floor(x));
    }
    else
    {
        return lcrt::fract(std::forward<X>(x));
    }
}

/// Radians -> degrees.
template <typename X>
[[nodiscard]] inline auto degrees(X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
        return static_cast<std::remove_cvref_t<X>>(x * (180.0 / 3.14159265358979323846));
    else
        return lcrt::degrees(std::forward<X>(x));
}

/// Degrees -> radians.
template <typename X>
[[nodiscard]] inline auto radians(X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
        return static_cast<std::remove_cvref_t<X>>(x * (3.14159265358979323846 / 180.0));
    else
        return lcrt::radians(std::forward<X>(x));
}

/// Sign of x: -1, 0 or +1.
template <typename X>
[[nodiscard]] inline auto sign(X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
    {
        using S = std::remove_cvref_t<X>;
        return x > S{0} ? S{1} : (x < S{0} ? S{-1} : S{0});
    }
    else
    {
        return lcrt::sign(std::forward<X>(x));
    }
}

/// Clamp to [0, 1] (DSL intrinsic; `clamp(x, 0, 1)` on the host).
template <typename X>
[[nodiscard]] inline auto saturate(X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
    {
        using S = std::remove_cvref_t<X>;
        return x < S{0} ? S{0} : (x > S{1} ? S{1} : static_cast<S>(x));
    }
    else
    {
        return lcrt::saturate(std::forward<X>(x));
    }
}

/// fabs is the C spelling of the same operation.
template <typename X>
[[nodiscard]] inline auto fabs(X&& x) noexcept
{
    return abs(std::forward<X>(x));
}

/// rsqrt (DSL intrinsic; `1/sqrt(x)` on the host).
template <typename X>
[[nodiscard]] inline auto rsqrt(X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
        return static_cast<std::remove_cvref_t<X>>(1) / std::sqrt(x);
    else
        return lcrt::rsqrt(std::forward<X>(x));
}

/// Cube root — LC has no `cbrt`, so the device path goes through `pow`.
template <typename X>
[[nodiscard]] inline auto cbrt(X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
    {
        using std::cbrt;
        return cbrt(std::forward<X>(x));
    }
    else
    {
        using S = scalar_value_t<X>;
        return lcrt::pow(std::forward<X>(x), lcrt::Expr<S>{static_cast<S>(1.0 / 3.0)});
    }
}

/// x raised to an *integer* power (unrolled; `lcrt::pow` needs a float exponent).
template <typename X>
[[nodiscard]] inline auto powi(X&& x, int n) noexcept
{
    using S = std::remove_cvref_t<X>;
    S result = one_of<S>();
    S base   = x;
    int e    = n < 0 ? -n : n;
    for(int i = 0; i < e; ++i)
        result = detail::mul(result, base);
    if(n < 0)
        return detail::div(one_of<S>(), result);
    return result;
}

// ---------------------------------------------------------------------------
//  Binary math
// ---------------------------------------------------------------------------
#define UIPC_LUISA_MATH_BINARY(NAME)                                          \
    template <typename X, typename Y>                                        \
    [[nodiscard]] inline auto NAME(X&& x, Y&& y) noexcept                     \
    {                                                                        \
        if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)            \
        {                                                                    \
            using std::NAME;                                                 \
            return NAME(std::forward<X>(x), std::forward<Y>(y));             \
        }                                                                    \
        else                                                                 \
        {                                                                    \
            return lcrt::NAME(std::forward<X>(x), std::forward<Y>(y));       \
        }                                                                    \
    }

UIPC_LUISA_MATH_BINARY(pow)
UIPC_LUISA_MATH_BINARY(atan2)
UIPC_LUISA_MATH_BINARY(fmod)
UIPC_LUISA_MATH_BINARY(copysign)

#undef UIPC_LUISA_MATH_BINARY

/// Euclidean remainder (`x - y * floor(x / y)`), matching LC's `mod`.
template <typename X, typename Y>
[[nodiscard]] inline auto mod(X&& x, Y&& y) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
        return x - y * std::floor(x / y);
    else
        return lcrt::mod(std::forward<X>(x), std::forward<Y>(y));
}

/// min / max.  Written with arithmetic dispatch because a DSL comparison yields
/// an `Expr<bool>`, which cannot drive C++ control flow.
template <typename X, typename Y>
[[nodiscard]] inline auto min(X&& x, Y&& y) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>> && std::is_arithmetic_v<std::remove_cvref_t<Y>>)
    {
        using R = std::common_type_t<std::remove_cvref_t<X>, std::remove_cvref_t<Y>>;
        return x < y ? static_cast<R>(x) : static_cast<R>(y);
    }
    else
    {
        return lcrt::min(std::forward<X>(x), std::forward<Y>(y));
    }
}

template <typename X, typename Y>
[[nodiscard]] inline auto max(X&& x, Y&& y) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>> && std::is_arithmetic_v<std::remove_cvref_t<Y>>)
    {
        using R = std::common_type_t<std::remove_cvref_t<X>, std::remove_cvref_t<Y>>;
        return x < y ? static_cast<R>(y) : static_cast<R>(x);
    }
    else
    {
        return lcrt::max(std::forward<X>(x), std::forward<Y>(y));
    }
}

template <typename V, typename L, typename U>
[[nodiscard]] inline auto clamp(V&& v, L&& lo, U&& hi) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<V>> && std::is_arithmetic_v<std::remove_cvref_t<L>>
                 && std::is_arithmetic_v<std::remove_cvref_t<U>>)
        return max(lo, min(hi, std::forward<V>(v)));
    else
        return lcrt::clamp(std::forward<V>(v), std::forward<L>(lo), std::forward<U>(hi));
}

/// Linear interpolation: `a + (b - a) * t`.
template <typename A, typename B, typename T>
[[nodiscard]] inline auto lerp(A&& a, B&& b, T&& t) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<A>>)
        return a + (b - a) * t;
    else
        return lcrt::lerp(std::forward<A>(a), std::forward<B>(b), std::forward<T>(t));
}

/// `mix` is the GLSL spelling of `lerp`.
template <typename A, typename B, typename T>
[[nodiscard]] inline auto mix(A&& a, B&& b, T&& t) noexcept
{
    return lerp(std::forward<A>(a), std::forward<B>(b), std::forward<T>(t));
}

/// fma: `x * y + z`.
template <typename X, typename Y, typename Z>
[[nodiscard]] inline auto fma(X&& x, Y&& y, Z&& z) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
    {
        using std::fma;
        return fma(std::forward<X>(x), std::forward<Y>(y), std::forward<Z>(z));
    }
    else
    {
        return lcrt::fma(std::forward<X>(x), std::forward<Y>(y), std::forward<Z>(z));
    }
}

/// step / smoothstep (GLSL argument order: edge first).
template <typename E, typename X>
[[nodiscard]] inline auto step(E&& edge, X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
        return x < edge ? std::remove_cvref_t<X>{0} : std::remove_cvref_t<X>{1};
    else
        return lcrt::step(std::forward<E>(edge), std::forward<X>(x));
}

template <typename L, typename R, typename X>
[[nodiscard]] inline auto smoothstep(L&& left, R&& right, X&& x) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<X>>)
    {
        const auto t = clamp((x - left) / (right - left), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
    else
    {
        return lcrt::smoothstep(std::forward<L>(left), std::forward<R>(right), std::forward<X>(x));
    }
}

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] inline auto pi() noexcept
{
    if constexpr(std::is_arithmetic_v<T>)
        return static_cast<T>(3.14159265358979323846);
    else
        return T{static_cast<scalar_value_t<T>>(3.14159265358979323846)};
}
}// namespace uipc::backend::luisa::luisa_device::math

namespace uipc::backend::luisa::luisa_device
{
/// Short alias used by ported kernel code (`lps::sqrt(...)`).
namespace lps = math;
}// namespace uipc::backend::luisa::luisa_device
