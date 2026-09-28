#pragma once
/********************************************************************
 * @file log.h
 * @brief Kernel-side logging facade (replacement for cuda_tool::KernelCout /
 *        LoggerViewer).
 *
 * The reference backend prints from kernels with a *streaming* helper:
 *     cout << "x = " << x << "\n";
 * LC has no device-side stream object: its device print is a single
 * `device_log(fmt, args...)` call whose format string is parsed by the runtime.
 * A streaming shim is therefore impossible; call sites must be rewritten to
 *     LC_LOG("x = {}", x);
 * This header supplies LC_LOG plus a no-op `KernelCout::viewer()` so the
 * surrounding plumbing (passing a `cout_view` argument) ports without change.
 *
 * Kernel printing is not free, so it is compiled out unless
 * UIPC_LUISA_KERNEL_PRINT is defined (mirrors cuda_tool::KernelCout's
 * debug-only contract, which was already gated by the CUDA backend).
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

#include <luisa_device/context.h>

#if !defined(UIPC_LUISA_KERNEL_PRINT)
#define UIPC_LUISA_KERNEL_PRINT 0
#endif

#if UIPC_LUISA_KERNEL_PRINT
// Device-side print. `fmt` uses the LC {} placeholder syntax.
#define LC_LOG(...) ::lcrt::device_log(__VA_ARGS__)
#define LC_LOG_ENABLED 1
#else
#define LC_LOG(...) ((void)0)
#define LC_LOG_ENABLED 0
#endif

namespace uipc::backend::luisa::luisa_device
{
// A device-capturable no-op handle. cuda_tool code threads a `LoggerViewer`
// through kernels; LC uses device_log() instead, so this type exists purely so
// the plumbing keeps compiling.
class LoggerViewer
{
  public:
    LoggerViewer() = default;
    // Streaming into the viewer is a no-op: use LC_LOG instead.
    template <typename T>
    const LoggerViewer& operator<<(const T&) const noexcept
    {
        return *this;
    }
};

class KernelCout
{
  public:
    KernelCout() = default;
    [[nodiscard]] static LoggerViewer viewer() noexcept { return {}; }
};
}// namespace uipc::backend::luisa::luisa_device
