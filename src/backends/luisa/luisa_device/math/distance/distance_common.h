#pragma once
/********************************************************************
 * @file distance_common.h
 * @brief Shared prologue for the ported distance / CCD primitives (T2.3).
 *
 * Brings in the generic numerics layer and provides the handful of small
 * helpers the ported reference code relies on but that `math/` does not expose
 * under the same name:
 *  - `detail::array_max/array_min(a, b)`: Eigen's `.array().max()/.min()`
 *    element-wise vector max/min, returning an `LVec`;
 *  - `detail::any_of/all_of(v)`: Eigen's `.array().any()/.all()` OR/AND
 *    reductions over a vector of booleans.  Device-safe: the reductions use
 *    `|`/`&`, never the short-circuiting `||`/`&&` (cookbook §7.6 trap #2);
 *  - `detail::select_scalar(f, t, p)` == `p ? t : f` for host *and* DSL scalar
 *    predicates.  Unqualified `select` is enough: `math::select` is found by
 *    ordinary lookup and SFINAE-removes itself for DSL operands, while LC's
 *    `lcrt::select` is then found by ADL.  Both use the `(false, true, pred)`
 *    argument order (cookbook §7.6 trap #1).
 *
 * Reference-faithfulness notes for the whole `distance/` directory:
 *  - the reference is `src/backends/cuda/utils/distance/` (read-only);
 *  - `Eigen::Vector<T,3>` -> `LVec<T,3>`, `Eigen::Matrix<T,N,N>` -> `LMat<T,N,N>`,
 *    `Eigen::Matrix<T,N,1>` -> `LVec<T,N>`;
 *  - `T(0)` / `T(1)` -> `math::zero_of<T>()` / `math::one_of<T>()` (a plain
 *    `T(0)` is ill-formed for `T = lcrt::Var<double>`).
 *********************************************************************/
#include <luisa_device/math/math.h>
#include <luisa_device/math/mat.h>
#include <luisa_device/math/vec.h>

namespace uipc::backend::luisa::luisa_device::math::distance
{
using math::LMat;
using math::LVec;

namespace detail
{
/// Eigen `.array().max(other)` -> element-wise `max` of two equal-dim vectors.
template <typename A, typename B>
[[nodiscard]] inline auto array_max(const A& a, const B& b) noexcept
{
    using R       = math::detail::element_result_t<math::vec_element_t<A>, math::vec_element_t<B>>;
    constexpr int N = math::vec_dim_v<A>;
    return math::make_vec<R, N>([&](int i) noexcept { return math::max(a(i), b(i)); });
}

/// Eigen `.array().min(other)` -> element-wise `min` of two equal-dim vectors.
template <typename A, typename B>
[[nodiscard]] inline auto array_min(const A& a, const B& b) noexcept
{
    using R       = math::detail::element_result_t<math::vec_element_t<A>, math::vec_element_t<B>>;
    constexpr int N = math::vec_dim_v<A>;
    return math::make_vec<R, N>([&](int i) noexcept { return math::min(a(i), b(i)); });
}

/// OR-reduction over a vector of booleans (Eigen `.array().any()`).
template <typename V>
[[nodiscard]] inline auto any_of(const V& v) noexcept
{
    using E = std::remove_cvref_t<decltype(v(0))>;
    using B = math::storage_of_t<E>;  // bool on host, Var<bool> on the device
    constexpr int N = math::vec_dim_v<V>;
    B acc           = static_cast<B>(v(0));
    for(int i = 1; i < N; ++i)
        acc = acc | static_cast<B>(v(i));
    return acc;
}

/// AND-reduction over a vector of booleans (Eigen `.array().all()`).
template <typename V>
[[nodiscard]] inline auto all_of(const V& v) noexcept
{
    using E = std::remove_cvref_t<decltype(v(0))>;
    using B = math::storage_of_t<E>;
    constexpr int N = math::vec_dim_v<V>;
    B acc           = static_cast<B>(v(0));
    for(int i = 1; i < N; ++i)
        acc = acc & static_cast<B>(v(i));
    return acc;
}

/// `p ? t : f`, valid for host booleans and DSL boolean expressions alike.
template <typename F, typename U, typename P>
[[nodiscard]] inline auto select_scalar(F&& f, U&& t, P&& p) noexcept
{
    return select(static_cast<F&&>(f), static_cast<U&&>(t), static_cast<P&&>(p));
}

/// Element-wise `p ? t : f` of two equal-dim vectors (`select_scalar` component by
/// component).  `select` must never see a container — LC's own `select` has a
/// custom-struct overload that would happily build a `Var<LVec<...>>` (cookbook
/// §7.6 trap #3), so containers are always selected element-wise through
/// `math::make_vec`.
///
/// T2.3 addition to this header (minimal, documented): the flag-dispatched
/// primitives select whole flag vectors, which needs the vector flavour.
template <typename V, typename W, typename P>
[[nodiscard]] inline auto select_vec(const V& f, const W& t, P&& p) noexcept
{
    using R         = math::detail::element_result_t<math::vec_element_t<V>, math::vec_element_t<W>>;
    constexpr int N = math::vec_dim_v<V>;
    return math::make_vec<R, N>([&](int i) noexcept { return select_scalar(f(i), t(i), p); });
}

/// Logical AND of two or more predicates, valid for host `bool`s and DSL boolean
/// expressions alike.
///
/// `&`/`|` are the only spellings LC offers for DSL scalars (`&&`/`||` are not
/// overloaded for them — cookbook §7.6 trap #2), but on the host `bool & bool`
/// *promotes to `int`*, so a third operand would mix `int` and `bool` (MSVC C4805).
/// The host branch therefore folds with `&&`/`||`, which is equivalent here (the
/// operands are pure predicate expressions, and short-circuiting only skips
/// evaluations that have no side effects).
///
/// T2.3 addition to this header (minimal, documented).
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
}  // namespace detail

/// Result type of a *predicate* of a primitive templated on the scalar `T`:
/// `bool` when `T` is a host arithmetic type, a DSL boolean expression otherwise.
///
/// T2.3 addition to this header (minimal, documented): several reference routines
/// return a data-dependent predicate (`need_mollify`, the four `*_ccd`).  The
/// pinned wrapper signatures declare them `bool`, which is correct on the host but
/// cannot carry a device value (RULE M2 / cookbook §7.3).  Those routines are
/// therefore overloaded on `is_arithmetic_v<T>`: the arithmetic overload keeps the
/// pinned `bool` signature, the device overload returns `bool_of_t<T>`.
template <typename T>
using bool_of_t = std::conditional_t<std::is_arithmetic_v<T>, bool, lcrt::Expr<bool>>;
}  // namespace uipc::backend::luisa::luisa_device::math::distance
