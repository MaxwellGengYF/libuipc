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

#include <luisa_device/tuning.h>
#include <luisa_device/context.h>
#include <luisa_device/stream.h>

// The NVTX switch (backends/common/nvtx_range.h owns the full rationale). Defined here
// as well so this header stays usable on its own; the value is the build's and is never
// overridden.
#ifndef UIPC_NVTX
#define UIPC_NVTX 0
#endif

namespace uipc::backend::luisa::luisa_device
{
// Compile a 1D kernel shader from an argument prototype list.
template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<1u, Args...> compile1(Def&& def)
{
    compile_stats::note_compile();
    return device().compile<1>(lcrt::Kernel1D<Args...>{std::forward<Def>(def)});
}

template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<2u, Args...> compile2(Def&& def)
{
    compile_stats::note_compile();
    return device().compile<2>(lcrt::Kernel2D<Args...>{std::forward<Def>(def)});
}

template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<3u, Args...> compile3(Def&& def)
{
    compile_stats::note_compile();
    return device().compile<3>(lcrt::Kernel3D<Args...>{std::forward<Def>(def)});
}

// ---- shader naming (BN1 I1, plan 13.2 N1) -----------------------------------
// The same three helpers, plus a name the profiler shows for every later dispatch of
// the compiled shader: the name reaches LC's `Resource::set_name`, and the CUDA backend
// then wraps each dispatch in `nvtxRangePushA(name)` / `nvtxRangePop()`
// (cuda_shader.cpp:71-74). One `set_name` per kernel is enough -- the label lives on the
// shader object, not on the dispatch command.
//
// The name is set *after* compile on purpose. The cache key is built from `ShaderOption`
// before `device().compile()` runs, and `set_name` is a runtime label that is not part of
// that hash (plan 13.2, N1), so labelling here cannot invalidate `<exe>/.cache`. Using
// `ShaderOption::name` instead would be hashed and force a full re-JIT (risk B4). Because
// the call site stores the result in a function-local `static`, the label is attached
// exactly once, on the run that already pays the compile.
//
// `set_name` is `const noexcept` (install-msvc/include/luisa/runtime/rhi/resource.h:306)
// and cannot throw, hence no try/catch. With `UIPC_NVTX=0` the call is compiled out and
// the named overload behaves exactly like the unnamed one.
template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<1u, Args...> compile1(std::string_view name,
                                                        Def&& def)
{
    compile_stats::note_compile();
    auto shader = device().compile<1>(lcrt::Kernel1D<Args...>{std::forward<Def>(def)});
#if UIPC_NVTX
    shader.set_name(name);
#else
    // The label only exists for the profiler; the parameter stays so that a call site
    // reads the same in both configurations.
    (void)name;
#endif
    return shader;
}

template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<2u, Args...> compile2(std::string_view name,
                                                        Def&& def)
{
    compile_stats::note_compile();
    auto shader = device().compile<2>(lcrt::Kernel2D<Args...>{std::forward<Def>(def)});
#if UIPC_NVTX
    shader.set_name(name);
#else
    (void)name;
#endif
    return shader;
}

template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<3u, Args...> compile3(std::string_view name,
                                                        Def&& def)
{
    compile_stats::note_compile();
    auto shader = device().compile<3>(lcrt::Kernel3D<Args...>{std::forward<Def>(def)});
#if UIPC_NVTX
    shader.set_name(name);
#else
    (void)name;
#endif
    return shader;
}

// ---- launch helpers (cuda_tool::best_block_dim / best_grid_dim parity) -----
namespace kernel
{
// LC lets a kernel pick its own block size with set_block_size(); this mirrors
// cuda_tool::default_block_dim for call sites that want the same number.
inline constexpr uint32_t kDefaultBlockDim = luisa_device::kDefaultBlockDim;

/// OP-C1: the *retunable* block size.  `luisa_device/tuning.h` owns the knob, the
/// config keys and the sweep protocol; a kernel body spells
/// `set_block_size(luisa_device::block_dim_for("name"))` and this is the unnamed
/// spelling, kept so a `cuda_tool`-shaped call site reads the same.
[[nodiscard]] inline uint32_t block_dim(uint32_t /*shared_mem*/ = 0u) noexcept
{
    return luisa_device::block_dim_for();
}

[[nodiscard]] inline uint32_t grid_dim(uint64_t n, uint32_t block = 0u) noexcept
{
    if(block == 0u)
        block = luisa_device::block_dim_for();
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
