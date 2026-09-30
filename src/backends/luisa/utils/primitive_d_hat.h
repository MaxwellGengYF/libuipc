#pragma once
/********************************************************************
 * @file primitive_d_hat.h
 * @brief T7.2b — per-primitive d_hat ("DCD expansion") conventions (port of
 *        src/backends/cuda/utils/primitive_d_hat.h, 95 lines).
 *
 * First consumer is the BVH simplex trajectory filters' detect stage: the swept AABB
 * of an edge/triangle uses the *averaged* d_hat of its vertices
 * (`edge_dcd_expansion`, `triangle_dcd_expansion`), while the leaf predicate uses the
 * pairwise spellings (`PP_d_hat`, `PE_d_hat`, `PT_d_hat`, `EE_d_hat`).  The whole
 * reference file is ported (a handful of one-liners) rather than only the five
 * functions the filters call, so T8.1's contact code finds the complete surface.
 *
 * Mapping / deviations (the `utils/codim_thickness.h` conventions, T5.2 batch b):
 *  * `UIPC_GENERIC` -> `template <typename S>` (`S = double` host,
 *    `S = lcrt::Var<double>` device);
 *  * the reference's `if constexpr(RUNTIME_CHECK) { UIPC_KERNEL_ASSERT(...) }` keeps its
 *    placement: `lcrt::device_assert` on a device predicate, `UIPC_ASSERT` (which
 *    aborts) on a host `bool` (a host `if` cannot take a device predicate, RULE M2);
 *  * `x / 2.0` -> `lps::detail::div(x, 2.0)`, `x + y` -> `lps::detail::add(x, y)` so the
 *    same body compiles for `double` and for a DSL scalar (same spelling as the ported
 *    codim_thickness.h; the arithmetic is the reference's, in the reference's order).
 *********************************************************************/
#include <type_define.h>
#include <uipc/common/config.h>
#include <uipc/common/log.h>
#include <luisa_device/math/math.h>
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace primitive_d_hat_detail
{
/// The reference's `if constexpr(RUNTIME_CHECK) { UIPC_KERNEL_ASSERT(...) }`: the same
/// helper the ported codim_thickness.h uses (device_assert / host UIPC_ASSERT).
template <typename Cond>
inline void assert_same_d_hat(Cond&& same, const char* message) noexcept
{
    if constexpr(lps::is_dsl_value_v<std::remove_cvref_t<Cond>>)
        lcrt::device_assert(same, message);
    else
        UIPC_ASSERT(static_cast<bool>(same), "{}", message);
}
}  // namespace primitive_d_hat_detail

/// The reference's `point_dcd_expansion`: a point's expansion is its own d_hat.
template <typename S>
[[nodiscard]] inline S point_dcd_expansion(const S& d_hat_P) noexcept
{
    return d_hat_P;
}

/**
 * @brief Edge d_hat
 */
template <typename S>
[[nodiscard]] inline S edge_dcd_expansion(const S& d_hat_E0, const S& d_hat_E1) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        primitive_d_hat_detail::assert_same_d_hat(d_hat_E0 == d_hat_E1,
                                                 "Edge d_hat should be the same");
    }

    return lps::detail::div(lps::detail::add(d_hat_E0, d_hat_E1), 2.0);
}

/**
 * @brief Triangle d_hat
 */
template <typename S>
[[nodiscard]] inline S triangle_dcd_expansion(const S& d_hat_T0,
                                              const S& d_hat_T1,
                                              const S& d_hat_T2) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        primitive_d_hat_detail::assert_same_d_hat(
            (d_hat_T0 == d_hat_T1) & (d_hat_T1 == d_hat_T2),
            "Triangle d_hat should be the same");
    }

    return lps::detail::div(lps::detail::add(lps::detail::add(d_hat_T0, d_hat_T1), d_hat_T2),
                            3.0);
}

/**
 * @brief Point-Triangle d_hat calculation
 */
template <typename S>
[[nodiscard]] inline S PT_d_hat(const S& d_hat_P,
                                const S& d_hat_T0,
                                const S& d_hat_T1,
                                const S& d_hat_T2) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        primitive_d_hat_detail::assert_same_d_hat((d_hat_T0 == d_hat_T1) & (d_hat_T1 == d_hat_T2),
                                                 "Triangle d_hat should be the same");
    }

    return lps::detail::div(lps::detail::add(d_hat_P, d_hat_T0), 2.0);
}

/**
 * @brief Edge-Edge d_hat calculation
 */
template <typename S>
[[nodiscard]] inline S EE_d_hat(const S& d_hat_Ea0,
                                const S& d_hat_Ea1,
                                const S& d_hat_Eb0,
                                const S& d_hat_Eb1) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        primitive_d_hat_detail::assert_same_d_hat(
            (d_hat_Ea0 == d_hat_Ea1) & (d_hat_Eb0 == d_hat_Eb1),
            "Edge d_hat should be the same");
    }

    return lps::detail::div(lps::detail::add(d_hat_Ea0, d_hat_Eb0), 2.0);
}

/**
 * @brief Point-Edge d_hat calculation
 */
template <typename S>
[[nodiscard]] inline S PE_d_hat(const S& d_hat_P, const S& d_hat_E0, const S& d_hat_E1) noexcept
{
    if constexpr(uipc::RUNTIME_CHECK)
    {
        primitive_d_hat_detail::assert_same_d_hat(d_hat_E0 == d_hat_E1,
                                                 "Edge d_hat should be the same");
    }

    return lps::detail::div(lps::detail::add(d_hat_P, d_hat_E0), 2.0);
}

/**
 * @brief Point-Point d_hat calculation
 */
template <typename S>
[[nodiscard]] inline S PP_d_hat(const S& d_hat_P0, const S& d_hat_P1) noexcept
{
    return lps::detail::div(lps::detail::add(d_hat_P0, d_hat_P1), 2.0);
}
}  // namespace uipc::backend::luisa
