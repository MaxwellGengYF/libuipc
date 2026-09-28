#pragma once
/********************************************************************
 * @file kernel.h
 * @brief Compile-once / dispatch-per-frame kernel helpers (plan decision D3).
 *
 * LC compiles a kernel DSL lambda into a Shader that owns JIT'd device code.
 * Compiling inside a frame loop would re-JIT every frame (risk R4), so every
 * call site must compile once. The idiomatic pattern is a function-local
 * `static`, which this helper makes explicit:
 *
 *   // host
 *   static auto k = lc::compile1<lcrt::Buffer<double>, uint>(
 *       [](lcrt::BufferVar<double> v, lcrt::Var<uint> n) noexcept {
 *           auto i = lcrt::dispatch_x();
 *           $if (i < n) { v->write(i, 1.0); };
 *       });
 *   stream << k(view, count).dispatch(count);
 *
 * `compile1<Prototype...>` takes the *argument prototype* types (the same ones
 * you would write in a `Kernel1D<...>` declaration): `Buffer<T>` for a device
 * buffer (becomes `BufferVar<T>` inside the lambda), and the plain scalar /
 * vector types for value arguments (become `Var<...>`).
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

#include <string>
#include <string_view>

#include <luisa_device/context.h>
#include <luisa_device/stream.h>

namespace uipc::backend::luisa::luisa_device
{
// Compile a 1D kernel shader from an argument prototype list.
template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<1u, Args...> compile1(Def&& def)
{
    return device().compile<1>(lcrt::Kernel1D<Args...>{std::forward<Def>(def)});
}

template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<2u, Args...> compile2(Def&& def)
{
    return device().compile<2>(lcrt::Kernel2D<Args...>{std::forward<Def>(def)});
}

template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<3u, Args...> compile3(Def&& def)
{
    return device().compile<3>(lcrt::Kernel3D<Args...>{std::forward<Def>(def)});
}

// ---- launch helpers (cuda_tool::best_block_dim / best_grid_dim parity) -----
namespace kernel
{
// LC lets a kernel pick its own block size with set_block_size(); this mirrors
// cuda_tool::default_block_dim for call sites that want the same number.
inline constexpr uint32_t kDefaultBlockDim = 256u;

[[nodiscard]] constexpr uint32_t block_dim(uint32_t /*shared_mem*/ = 0u) noexcept
{
    return kDefaultBlockDim;
}

[[nodiscard]] constexpr uint32_t grid_dim(uint64_t n, uint32_t block = kDefaultBlockDim) noexcept
{
    return static_cast<uint32_t>((n + block - 1u) / block);
}
}// namespace kernel

// A record of a dispatch: keeps the intended element count so call sites can
// read it back (cuda_tool code often passes `n` around separately).
struct Launch1D
{
    uint64_t count = 0u;
};
}// namespace uipc::backend::luisa::luisa_device
