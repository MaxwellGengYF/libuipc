#pragma once
/********************************************************************
 * @file codim_thickness.h
 * @brief T5.2 (batch b) — codimension thickness conventions (port of
 *        src/backends/cuda/utils/codim_thickness.h).
 *
 * First consumer is the shell membrane constitutions of batch b: they need
 * `triangle_thickness` for the one-sided -> full thickness conversion.  The whole
 * reference file is ported (it is a handful of one-liners) rather than only the
 * one function, so T8.1's contact/collision code finds the complete surface.
 *
 * Mapping / deviations (reported in the T5.2 batch-b report):
 *  * `UIPC_GENERIC` -> `template <typename S>` (`S = double` host,
 *    `S = lcrt::Var<double>` device), like the rest of this layer;
 *  * `UIPC_KERNEL_ASSERT(cond, "...")` keeps its `if constexpr(RUNTIME_CHECK)`
 *    placement (cookbook §7.16 rule 6): `lcrt::device_assert` on the device path
 *    and `UIPC_ASSERT` (which aborts, §7.16 rule 7) on the host path, because a
 *    host `if` cannot take a device predicate (RULE M2);
 *  * `is_active_D`'s `a && b` becomes `pred_and(a, b)` — LC requires `&` for DSL
 *    scalars, while on the host `bool & bool` promotes to int (§7.6 trap 2).
 *********************************************************************/
#include <type_define.h>
#include <uipc/common/config.h>
#include <uipc/common/log.h>
#include <luisa_device/math/math.h>
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace codim_thickness_detail
{
/// The reference's `if constexpr(RUNTIME_CHECK) { UIPC_KERNEL_ASSERT(...) }`:
/// `lcrt::device_assert` when the condition is a device predicate, `UIPC_ASSERT`
/// (which aborts, cookbook §7.16 rule 7) when it is a host `bool`.
template <typename Cond>
inline void assert_same_thickness(Cond&& same, const char* message) noexcept
{
    if constexpr(lps::is_dsl_value_v<std::remove_cvref_t<Cond>>)
        lcrt::device_assert(same, message);
    else
        UIPC_ASSERT(static_cast<bool>(same), "{}", message);
}
}  // namespace codim_thickness_detail

/**
 * @brief Edge thickness
 */
template <typename S>
[[nodiscard]] inline S edge_thickness(const S& thickness_E0, const S& thickness_E1) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        codim_thickness_detail::assert_same_thickness(
            thickness_E0 == thickness_E1, "Edge thickness should be the same");
    }

    return thickness_E0;
}

/**
 * @brief Triangle thickness
 */
template <typename S>
[[nodiscard]] inline S triangle_thickness(const S& thickness_T0,
                                          const S& thickness_T1,
                                          const S& thickness_T2) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        codim_thickness_detail::assert_same_thickness(
            (thickness_T0 == thickness_T1) & (thickness_T1 == thickness_T2),
            "Triangle thickness should be the same");
    }

    return thickness_T0;
}

/**
 * @brief Point-Triangle thickness calculation
 */
template <typename S>
[[nodiscard]] inline S PT_thickness(const S& thickness_P,
                                    const S& thickness_T0,
                                    const S& thickness_T1,
                                    const S& thickness_T2) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        codim_thickness_detail::assert_same_thickness(
            (thickness_T0 == thickness_T1) & (thickness_T1 == thickness_T2),
            "Triangle thickness should be the same");
    }

    // return (thickness_P + thickness_T0) / 2.0;
    return lps::detail::add(thickness_P, thickness_T0);
}

/**
 * @brief Edge-Edge thickness calculation
 */
template <typename S>
[[nodiscard]] inline S EE_thickness(const S& thickness_Ea0,
                                    const S& thickness_Ea1,
                                    const S& thickness_Eb0,
                                    const S& thickness_Eb1) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        codim_thickness_detail::assert_same_thickness(
            (thickness_Ea0 == thickness_Ea1) & (thickness_Eb0 == thickness_Eb1),
            "Edge thickness should be the same");
    }

    // return (thickness_Ea0 + thickness_Eb0) / 2.0;
    return lps::detail::add(thickness_Ea0, thickness_Eb0);
}

/**
 * @brief Point-Edge thickness calculation
 */
template <typename S>
[[nodiscard]] inline S PE_thickness(const S& thickness_P,
                                    const S& thickness_E0,
                                    const S& thickness_E1) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        codim_thickness_detail::assert_same_thickness(thickness_E0 == thickness_E1,
                                                      "Edge thickness should be the same");
    }

    // return (thickness_P + thickness_E0) / 2.0;
    return lps::detail::add(thickness_P, thickness_E0);
}

/**
 * @brief Point-Point thickness calculation
 */
template <typename S>
[[nodiscard]] inline S PP_thickness(const S& thickness_P0, const S& thickness_P1) noexcept
{
    // return (thickness_P0 + thickness_P1) / 2.0;
    return lps::detail::add(thickness_P0, thickness_P1);
}

/**
 * @brief the range of d^2, considering the thickness
 */
template <typename S>
[[nodiscard]] inline lps::LVec<S, 2> D_range(const S& xi, const S& d_hat) noexcept
{
    auto upper = lps::detail::add(xi, d_hat);
    auto lower = xi;
    return lps::LVec<S, 2>{lps::detail::mul(lower, lower), lps::detail::mul(upper, upper)};
}

/**
 * @brief check if D is in the active range [D_range.x(), D_range.y())
 */
template <typename S>
[[nodiscard]] inline auto is_active_D(const lps::LVec<S, 2>& D_range, const S& D) noexcept
{
    return scalar_detail::pred_and(D_range(0) < D, D < D_range(1));
}
}  // namespace uipc::backend::luisa
