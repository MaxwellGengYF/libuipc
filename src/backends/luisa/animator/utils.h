#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/mathematics.h>

namespace uipc::backend::luisa
{
template<typename T, int N>
LUISA_GENERIC luisa::Vector<T, N> lerp(const luisa::Vector<T, N>& src,
                                       const luisa::Vector<T, N>& dst,
                                       T                               alpha)
{
    alpha = luisa::clamp(alpha, T{0}, T{1});
    return src * (T{1} - alpha) + dst * alpha;
}

template<typename T>
LUISA_GENERIC T lerp(const T& src, const T& dst, T alpha)
{
    alpha = luisa::clamp(alpha, T{0}, T{1});
    return src * (T{1} - alpha) + dst * alpha;
}
}  // namespace uipc::backend::luisa
