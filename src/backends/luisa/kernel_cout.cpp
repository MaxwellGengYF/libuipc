#include <kernel_cout.h>

#include <luisa_device/luisa_device.h>

namespace uipc::backend::luisa
{
// Currently header-only: the LC device print path lives in luisa_device::KernelCout.
// This translation unit exists so the header is compiled by the backend target and
// so future stateful additions (per-stream print buffers) have a home.
}
