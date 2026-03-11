#pragma once
/********************************************************************
 * @file   type_define.h
 * @brief  Type definitions for luisa-compute backend distance utilities
 * 
 * This file provides the type mappings from CUDA/muda backend to
 * luisa-compute backend for distance computation utilities.
 *********************************************************************/

#include <luisa/core/basic_types.h>
#include <array>

// UIPC common types
#include <uipc/common/type_define.h>

namespace uipc::backend::luisa {

using IndexT = uipc::IndexT;
using Float  = uipc::Float;

// Vector types from luisa
using Vector2 = luisa::float2;
using Vector3 = luisa::float3;
using Vector4 = luisa::float4;
using Vector2i = luisa::int2;
using Vector3i = luisa::int3;
using Vector4i = luisa::int4;

// Large vectors as std::array (luisa native limited to 4)
using Vector9 = std::array<float, 9>;
using Vector12 = std::array<float, 12>;

// Matrix types from luisa
using Matrix3x3 = luisa::float3x3;
using Matrix4x4 = luisa::float4x4;

// Generic callable macros for luisa-compute
#define LC_GPU_CALLABLE luisa::compute::callable

}  // namespace uipc::backend::luisa

// Helper functions for luisa vector operations
namespace luisa {

// max/min for vectors
template<typename T>
LC_GPU_CALLABLE Vector<T, 3> max(const Vector<T, 3>& a, const Vector<T, 3>& b) {
    return Vector<T, 3>{
        a.x > b.x ? a.x : b.x,
        a.y > b.y ? a.y : b.y,
        a.z > b.z ? a.z : b.z
    };
}

template<typename T>
LC_GPU_CALLABLE Vector<T, 3> min(const Vector<T, 3>& a, const Vector<T, 3>& b) {
    return Vector<T, 3>{
        a.x < b.x ? a.x : b.x,
        a.y < b.y ? a.y : b.y,
        a.z < b.z ? a.z : b.z
    };
}

template<typename T>
LC_GPU_CALLABLE Vector<T, 2> max(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>{
        a.x > b.x ? a.x : b.x,
        a.y > b.y ? a.y : b.y
    };
}

template<typename T>
LC_GPU_CALLABLE Vector<T, 2> min(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>{
        a.x < b.x ? a.x : b.x,
        a.y < b.y ? a.y : b.y
    };
}

template<typename T>
LC_GPU_CALLABLE T max(T a, T b) {
    return a > b ? a : b;
}

template<typename T>
LC_GPU_CALLABLE T min(T a, T b) {
    return a < b ? a : b;
}

// Vector element access helper for std::array-like access
namespace compute {

// Helper to get element from vector by index
template<typename T, size_t N>
LC_GPU_CALLABLE T vector_get(const Vector<T, N>& v, size_t i) {
    return v[i];
}

template<typename T, size_t N>
LC_GPU_CALLABLE void vector_set(Vector<T, N>& v, size_t i, T val) {
    v[i] = val;
}

} // namespace compute

}  // namespace luisa
