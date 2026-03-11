#include <kernel_cout.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <uipc/common/log.h>

namespace uipc::backend::luisa
{
// Note: LuisaCompute uses device_log() DSL function for kernel-side logging.
// The log output is automatically handled by the LuisaCompute runtime.
// This implementation file provides compatibility with the UIPC backend architecture
// and can be extended for custom log handling if needed in the future.

// The KernelCout class is header-only as device_log is a DSL intrinsic.
// No additional implementation is required for basic functionality.

// Future extension: If custom log buffer management is needed,
// we could implement a ring buffer using LuisaCompute Buffer<char>
// and read it back on the host side after kernel execution.

}  // namespace uipc::backend::luisa
