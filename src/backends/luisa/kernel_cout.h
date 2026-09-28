#pragma once
/********************************************************************
 * @file kernel_cout.h
 * @brief Backend-level device console facade.
 *
 * Thin wrapper over luisa_device::KernelCout, kept so that ported call sites
 * (`KernelCout::viewer()`) resolve unchanged. In LC, device printing is
 * device_log(); use LC_LOG(...) inside kernels (see luisa_device/log.h).
 *********************************************************************/
#include <type_define.h>

namespace uipc::backend::luisa
{
class KernelCout
{
  public:
    // Kept for call-site compatibility; LC prints with device_log/LC_LOG.
    [[nodiscard]] static luisa_device::LoggerViewer viewer() noexcept
    {
        return luisa_device::KernelCout::viewer();
    }
};
}// namespace uipc::backend::luisa
