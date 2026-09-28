#pragma once
/********************************************************************
 * @file utils.h
 * @brief Port of src/backends/cuda/animator/utils.h (T4.2).
 *
 * The reference's `lerp` is a UIPC_GENERIC function template instantiated for
 * both host and device scalars; its law is *linear interpolation with a clamped
 * alpha*:
 *
 *     alpha' = clamp(alpha, 0, 1)
 *     lerp(src, dst, alpha) = src * (1 - alpha') + dst * alpha'
 *
 * The clamp is load-bearing (an animator's keyframes are evaluated past the last
 * key: alpha > 1 must hold the last key, not extrapolate), and the
 * `src * (1 - a) + dst * a` spelling is kept verbatim -- it is not the same
 * floating-point expression as `src + (dst - src) * a` (lps::lerp), and the
 * reference's rounding is what the animation keyframe values are defined by.
 *
 * Port mapping (cookbook §5, §7.1):
 *   - `UIPC_GENERIC` -> an ordinary template; the scalar type is either `double`
 *     (host evaluation, unit tests) or `lcrt::Var<double>` (device evaluation).
 *   - `Eigen::Matrix<T, M, N>` covers Eigen's vectors *and* matrices; the
 *     numerics layer spells those as two distinct templates, so there is one
 *     overload for `lps::LVec<T, N>` and one for `lps::LMat<T, R, C>`; the
 *     generic scalar overload is the reference's second overload unchanged.
 *   - `std::clamp` cannot be used for a DSL scalar (RULE M2: a DSL comparison is
 *     not a host bool), so the clamp dispatches on `std::is_arithmetic_v<T>`.
 * These helpers are pure templates over the numerics layer; nothing here is a
 * buffer element type, so no new LC element type is needed for them
 * (type_define.h is unchanged by this task -- see the report).
 *********************************************************************/
#include <type_define.h>
#include <luisa_device/math/mat.h>
#include <algorithm>

namespace uipc::backend::luisa
{
namespace animator
{
namespace detail
{
/// `std::clamp(alpha, T{0}, T{1})` for a host scalar, `lcrt::clamp` for a DSL
/// scalar (a DSL comparison must not leak into host control flow, RULE M2).
template <typename T>
[[nodiscard]] inline T clamp01(T alpha) noexcept
{
    if constexpr(std::is_arithmetic_v<T>)
    {
        return std::clamp(alpha, T{0}, T{1});
    }
    else
    {
        return lcrt::clamp(alpha, T{0}, T{1});
    }
}
}  // namespace detail

/// Eigen::Matrix<T,M,N> analogue (Eigen's matrices *and* its column vectors).
template <typename T, int N>
[[nodiscard]] inline lps::LVec<T, N> lerp(const lps::LVec<T, N>& src,
                                          const lps::LVec<T, N>& dst,
                                          T alpha) noexcept
{
    alpha = detail::clamp01(alpha);
    return src * (T{1} - alpha) + dst * alpha;
}

template <typename T, int R, int C>
[[nodiscard]] inline lps::LMat<T, R, C> lerp(const lps::LMat<T, R, C>& src,
                                             const lps::LMat<T, R, C>& dst,
                                             T alpha) noexcept
{
    alpha = detail::clamp01(alpha);
    return src * (T{1} - alpha) + dst * alpha;
}

template <typename T>
[[nodiscard]] inline T lerp(const T& src, const T& dst, T alpha) noexcept
{
    alpha = detail::clamp01(alpha);
    return src * (T{1} - alpha) + dst * alpha;
}
}  // namespace animator
}  // namespace uipc::backend::luisa
