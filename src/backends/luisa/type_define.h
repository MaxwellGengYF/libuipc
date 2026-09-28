#pragma once
/********************************************************************
 * @file type_define.h
 * @brief Core aliases for the LuisaCompute backend.
 *
 * Mirrors src/backends/cuda/type_define.h but for the DSL paradigm: there are no
 * launch qualifier macros (UIPC_HOST/DEVICE/GLOBAL) because kernels are built
 * with device.compile<1>, and the device utility layer is luisa_device
 * (the LuisaCompute analogue of cuda_tool).
 *
 * Pulled into host (.cpp) translation units, so it must stay lightweight.
 *********************************************************************/
#include <uipc/common/type_define.h>

#include <luisa_device/luisa_device.h>

namespace uipc::backend::luisa
{
using Float    = uipc::Float;
using IndexT   = uipc::IndexT;
using SizeT    = uipc::SizeT;
using Vector2  = uipc::Vector2;
using Vector3  = uipc::Vector3;
using Vector4  = uipc::Vector4;
using Vector6  = uipc::Vector6;
using Vector9  = uipc::Vector9;
using Vector12 = uipc::Vector12;
using Matrix2x2 = uipc::Matrix2x2;
using Matrix3x3 = uipc::Matrix3x3;
using Matrix4x4 = uipc::Matrix4x4;
using Matrix6x6 = uipc::Matrix6x6;
using Matrix9x9 = uipc::Matrix9x9;
using Matrix12x12 = uipc::Matrix12x12;

// Device utility layer (short name used by ported sources).
namespace lud = luisa_device;
namespace lc  = luisa_device;
}// namespace uipc::backend::luisa
