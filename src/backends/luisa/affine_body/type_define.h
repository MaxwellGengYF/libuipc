#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/matrix.h>

namespace uipc::backend::luisa
{
using Float = double;

// Use luisa float types
using float3 = luisa::float3;
using float3x3 = luisa::float3x3;

// 12-element vector for ABD (Affine Body Dynamics)
struct Vector12
{
    float data[12];
    
    __attribute__((always_inline)) float& operator[](size_t i) { return data[i]; }
    __attribute__((always_inline)) const float& operator[](size_t i) const { return data[i]; }
    
    __attribute__((always_inline)) float* begin() { return data; }
    __attribute__((always_inline)) const float* begin() const { return data; }
    __attribute__((always_inline)) float* end() { return data + 12; }
    __attribute__((always_inline)) const float* end() const { return data + 12; }
    
    // segment access: returns a float3 view at offset
    __attribute__((always_inline)) float3 segment3(size_t offset) const
    {
        return float3{data[offset], data[offset + 1], data[offset + 2]};
    }
    
    __attribute__((always_inline)) void set_segment3(size_t offset, const float3& v)
    {
        data[offset] = v.x;
        data[offset + 1] = v.y;
        data[offset + 2] = v.z;
    }
    
    static __attribute__((always_inline)) Vector12 zeros()
    {
        Vector12 v;
        for(int i = 0; i < 12; ++i) v.data[i] = 0.0f;
        return v;
    }
};

// Generic vector type
template<typename T, size_t N>
struct Vector
{
    T data[N];
    
    __attribute__((always_inline)) T& operator[](size_t i) { return data[i]; }
    __attribute__((always_inline)) const T& operator[](size_t i) const { return data[i]; }
    
    __attribute__((always_inline)) T* begin() { return data; }
    __attribute__((always_inline)) const T* begin() const { return data; }
    __attribute__((always_inline)) T* end() { return data + N; }
    __attribute__((always_inline)) const T* end() const { return data + N; }
    
    static __attribute__((always_inline)) Vector zeros()
    {
        Vector v;
        for(size_t i = 0; i < N; ++i) v.data[i] = T(0);
        return v;
    }
};

using Vector9 = Vector<float, 9>;      // 9-element vector for ABD
using Vector3 = luisa::float3;         // 3-element vector
using Vector4 = luisa::float4;         // 4-element vector

using Matrix3x3 = luisa::float3x3;     // 3x3 matrix
using Matrix4x4 = luisa::float4x4;     // 4x4 matrix
using Matrix9x9 = Matrix<float, 9, 9>; // 9x9 matrix for Hessian transformation

using Matrix3x12 = Vector<float, 36>;  // 3x12 = 36 elements, row-major or col-major layout
using Matrix12x3 = Vector<float, 36>;  // 12x3 = 36 elements

// 12x12 matrix for Hessian accumulation
struct Matrix12x12
{
    float data[144];  // 12x12 = 144 elements, stored in column-major order (matching Eigen)
    
    __attribute__((always_inline)) float& operator()(size_t row, size_t col)
    {
        return data[col * 12 + row];  // column-major: col * stride + row
    }
    
    __attribute__((always_inline)) const float& operator()(size_t row, size_t col) const
    {
        return data[col * 12 + row];
    }
    
    __attribute__((always_inline)) float* begin() { return data; }
    __attribute__((always_inline)) const float* begin() const { return data; }
    
    static __attribute__((always_inline)) Matrix12x12 zeros()
    {
        Matrix12x12 m;
        for(int i = 0; i < 144; ++i) m.data[i] = 0.0f;
        return m;
    }
    
    // Block access: 3x3 block at (row, col)
    __attribute__((always_inline)) float3x3 block3x3(size_t row, size_t col) const
    {
        float3x3 result;
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                result[c][r] = (*this)(row + r, col + c);
        return result;
    }
    
    __attribute__((always_inline)) void set_block3x3(size_t row, size_t col, const float3x3& m)
    {
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                (*this)(row + r, col + c) = m[c][r];
    }
    
    // Add to 3x3 block
    __attribute__((always_inline)) void add_block3x3(size_t row, size_t col, const float3x3& m)
    {
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                (*this)(row + r, col + c) += m[c][r];
    }
    
    // 3-element row block access (1x3 at row, col)
    __attribute__((always_inline)) float3 row_block3(size_t row, size_t col) const
    {
        return float3{(*this)(row, col), (*this)(row, col + 1), (*this)(row, col + 2)};
    }
    
    // 3-element col block access (3x1 at row, col)
    __attribute__((always_inline)) float3 col_block3(size_t row, size_t col) const
    {
        return float3{(*this)(row, col), (*this)(row + 1, col), (*this)(row + 2, col)};
    }
    
    __attribute__((always_inline)) void set_row_block3(size_t row, size_t col, const float3& v)
    {
        (*this)(row, col) = v.x;
        (*this)(row, col + 1) = v.y;
        (*this)(row, col + 2) = v.z;
    }
    
    __attribute__((always_inline)) void set_col_block3(size_t row, size_t col, const float3& v)
    {
        (*this)(row, col) = v.x;
        (*this)(row + 1, col) = v.y;
        (*this)(row + 2, col) = v.z;
    }
    
    __attribute__((always_inline)) void add_row_block3(size_t row, size_t col, const float3& v)
    {
        (*this)(row, col) += v.x;
        (*this)(row, col + 1) += v.y;
        (*this)(row, col + 2) += v.z;
    }
    
    __attribute__((always_inline)) void add_col_block3(size_t row, size_t col, const float3& v)
    {
        (*this)(row, col) += v.x;
        (*this)(row + 1, col) += v.y;
        (*this)(row + 2, col) += v.z;
    }
};

// Helper matrix type
template<typename T, size_t M, size_t N>
struct Matrix
{
    T data[M * N];
    
    __attribute__((always_inline)) T& operator()(size_t row, size_t col)
    {
        return data[col * M + row];  // column-major
    }
    
    __attribute__((always_inline)) const T& operator()(size_t row, size_t col) const
    {
        return data[col * M + row];
    }
};

}  // namespace uipc::backend::luisa
