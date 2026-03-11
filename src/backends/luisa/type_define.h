#pragma once
/********************************************************************
 * @file   type_define.h
 * @brief  Type definitions for LuisaCompute backend
 * 
 * Based on uipc/common/type_define.h but adapted for LuisaCompute
 *********************************************************************/
#include <luisa/core/basic_types.h>
#include <luisa/core/mathematics.h>
#include <uipc/common/type_define.h>

namespace uipc::backend::luisa
{
// Import common UIPC types
using uipc::IndexT;
using uipc::SizeT;
using uipc::Float;
using uipc::U64;
using uipc::S;
using uipc::U;
using uipc::SP;

// LuisaCompute vector types for physics simulation
using Vector2 = luisa::float2;
using Vector3 = luisa::float3;
using Vector4 = luisa::float4;
using Vector2i = luisa::int2;
using Vector3i = luisa::int3;
using Vector4i = luisa::int4;

// Matrix types
using Matrix2x2 = luisa::float2x2;
using Matrix3x3 = luisa::float3x3;
using Matrix4x4 = luisa::float4x4;

// Higher dimensional vectors for ABD (Affine Body Dynamics)
using Vector6 = luisa::float4;  // Store 6 floats with padding
using Vector12 = luisa::float4; // Store 12 floats with padding (3 x float4)

// Matrix types for ABD
using Matrix6x6 = luisa::float3x3;   // Store 6x6 using larger blocks
using Matrix12x12 = luisa::float4x4; // Store 12x12 using 4x4 blocks

// Device/host qualifiers - simplified for LuisaCompute
#define UIPC_GENERIC 
#define UIPC_DEVICE 
#define UIPC_HOST 

// For compatibility with existing code
using namespace luisa;
}  // namespace uipc::backend::luisa
