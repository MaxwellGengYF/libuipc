#pragma once
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
namespace detail::bit_operation
{
    using namespace luisa::compute::dsl;

    /// Warp ballot - returns bit mask of threads where predicate is true
    /// Note: In luisa-compute, warp_active_bit_mask returns uint4 (128-bit mask)
    /// For 32-thread warps, only the .x component is used
    [[nodiscard]] inline auto WARP_BALLOT(Expr<bool> predicate) noexcept
    {
        // warp_active_bit_mask returns uint4 containing 128-bit mask
        // For standard 32-lane warps, the mask is in the .x component
        return warp_active_bit_mask(predicate).x;
    }
}// namespace detail::bit_operation
}// namespace uipc::backend::luisa
