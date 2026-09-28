#pragma once
/********************************************************************
 * @file debug.h
 * @brief Debug / error helpers + sync callbacks (cuda_tool::Debug parity).
 *
 * The reference backend exposes
 *     cuda_tool::Debug::set_sync_callback(cb)
 *     cuda_tool::Debug::debug_sync_all()
 * so that a debug build can inspect device state at every synchronization
 * point, plus CUDA_TOOL_CHECK for error checking.
 *
 * In LC, errors surface as exceptions from the runtime and the equivalent of
 * `wait_device()` is `stream.synchronize()`. This header provides the same
 * callback hook on top of LStream::synchronize().
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <luisa_device/context.h>
#include <luisa_device/stream.h>

namespace uipc::backend::luisa::luisa_device
{
// Throwing check (LC relies on exceptions rather than error codes).
inline void check(bool ok, const char* expr, const char* file, int line)
{
    if(!ok)
    {
        throw std::runtime_error{std::string{"[luisa_device] check failed: "} + expr +
                                 " at " + file + ":" + std::to_string(line)};
    }
}

#define LUISA_DEVICE_CHECK(EXPR) \
    ::uipc::backend::luisa::luisa_device::check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)

namespace debug
{
using SyncCallback = std::function<void()>;

// Registered callbacks run right before every explicit device synchronization.
[[nodiscard]] inline std::vector<SyncCallback>& sync_callbacks()
{
    static std::vector<SyncCallback> s_callbacks;
    return s_callbacks;
}

inline void set_sync_callback(SyncCallback cb)
{
    sync_callbacks().clear();
    sync_callbacks().push_back(std::move(cb));
}

inline void clear_sync_callback()
{
    sync_callbacks().clear();
}

// cuda_tool::Debug::debug_sync_all()
inline void debug_sync_all()
{
    for(auto& cb : sync_callbacks())
    {
        cb();
    }
    main_stream().synchronize();
}

// Synchronize the main stream, running registered callbacks first.
inline void sync()
{
    debug_sync_all();
}
}// namespace debug
}// namespace uipc::backend::luisa::luisa_device
