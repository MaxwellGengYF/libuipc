#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <uipc/common/config.h>

namespace uipc::backend::luisa
{
/**
 * @brief Point d_hat expansion
 */
inline float point_dcd_expansion(const float& d_hat_P)
{
    return d_hat_P;
}

/**
 * @brief Edge d_hat expansion
 */
inline float edge_dcd_expansion(const float& d_hat_E0, const float& d_hat_E1)
{
    if constexpr(RUNTIME_CHECK)
    {
        LUISA_ASSERT(d_hat_E0 == d_hat_E1, "Edge d_hat should be the same");
    }

    return (d_hat_E0 + d_hat_E1) / 2.0f;
}

/**
 * @brief Triangle d_hat expansion
 */
inline float triangle_dcd_expansion(const float& d_hat_T0,
                                    const float& d_hat_T1,
                                    const float& d_hat_T2)
{
    if constexpr(RUNTIME_CHECK)
    {
        LUISA_ASSERT(d_hat_T0 == d_hat_T1 && d_hat_T1 == d_hat_T2,
                     "Triangle d_hat should be the same");
    }

    return (d_hat_T0 + d_hat_T1 + d_hat_T2) / 3.0f;
}

/**
 * @brief Point-Triangle d_hat calculation
 */
inline float PT_d_hat(const float& d_hat_P,
                      const float& d_hat_T0,
                      const float& d_hat_T1,
                      const float& d_hat_T2)
{
    if constexpr(RUNTIME_CHECK)
    {
        LUISA_ASSERT(d_hat_T0 == d_hat_T1 && d_hat_T1 == d_hat_T2,
                     "Triangle d_hat should be the same");
    }

    return (d_hat_P + d_hat_T0) / 2.0f;
}

/**
 * @brief Edge-Edge d_hat calculation
 */
inline float EE_d_hat(const float& d_hat_Ea0,
                      const float& d_hat_Ea1,
                      const float& d_hat_Eb0,
                      const float& d_hat_Eb1)
{
    if constexpr(RUNTIME_CHECK)
    {
        LUISA_ASSERT(d_hat_Ea0 == d_hat_Ea1 && d_hat_Eb0 == d_hat_Eb1,
                     "Edge d_hat should be the same");
    }

    return (d_hat_Ea0 + d_hat_Eb0) / 2.0f;
}

/**
 * @brief Point-Edge d_hat calculation
 */
inline float PE_d_hat(const float& d_hat_P, const float& d_hat_E0, const float& d_hat_E1)
{
    if constexpr(RUNTIME_CHECK)
    {
        LUISA_ASSERT(d_hat_E0 == d_hat_E1, "Edge d_hat should be the same");
    }

    return (d_hat_P + d_hat_E0) / 2.0f;
}

/**
 * @brief Point-Point d_hat calculation
 */
inline float PP_d_hat(const float& d_hat_P0, const float& d_hat_P1)
{
    return (d_hat_P0 + d_hat_P1) / 2.0f;
}
}  // namespace uipc::backend::luisa
