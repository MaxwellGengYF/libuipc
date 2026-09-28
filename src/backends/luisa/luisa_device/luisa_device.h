#pragma once
#include <luisa_device/compat.h>
/********************************************************************
 * @file luisa_device.h
 * @brief Umbrella header for the luisa_device foundation layer.
 *
 * luisa_device is the LuisaCompute analogue of the reference backend's
 * `cuda_tool` layer: buffers, views, streams, kernel compilation, atomics,
 * warp builtins, logging and debug helpers. Ported backend code should include
 * this header (via <luisa/type_define.h>) instead of <cuda_tool/cuda_tool.h>.
 *********************************************************************/
#include <luisa_device/context.h>
#include <luisa_device/stream.h>
#include <luisa_device/view.h>
#include <luisa_device/buffer.h>
#include <luisa_device/atomic.h>
#include <luisa_device/builtins.h>
#include <luisa_device/kernel.h>
#include <luisa_device/log.h>
#include <luisa_device/debug.h>

namespace uipc::backend::luisa
{
// Short alias used throughout the ported backend sources.
namespace lc = luisa_device;
}// namespace uipc::backend::luisa
