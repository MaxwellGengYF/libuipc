#pragma once
#include <luisa/luisa-compute.h>
#include <core/basic_types.h>
#include <array>
#include <cmath>

namespace uipc::backend::luisa
{
namespace sym::strainlimiting_baraff_witkin_shell_2d
{
    // Custom vector types for larger dimensions (not natively supported in LC)
    using Float6 = std::array<float, 6>;
    using Float9 = std::array<float, 9>;
    
    // Custom matrix types (column-major storage)
    struct Float3x2 {
        luisa::Float3 cols[2];
        
        Float3x2() = default;
        Float3x2(const luisa::Float3& c0, const luisa::Float3& c1) : cols{c0, c1} {}
        
        [[nodiscard]] constexpr luisa::Float3& operator[](size_t i) noexcept { return cols[i]; }
        [[nodiscard]] constexpr const luisa::Float3& operator[](size_t i) const noexcept { return cols[i]; }
        
        static Float3x2 zero() noexcept {
            return Float3x2{luisa::Float3{0.0f}, luisa::Float3{0.0f}};
        }
    };
    
    struct Float2x2 {
        luisa::Float2 cols[2];
        
        Float2x2() = default;
        Float2x2(const luisa::Float2& c0, const luisa::Float2& c1) : cols{c0, c1} {}
        
        [[nodiscard]] constexpr luisa::Float2& operator[](size_t i) noexcept { return cols[i]; }
        [[nodiscard]] constexpr const luisa::Float2& operator[](size_t i) const noexcept { return cols[i]; }
        
        static Float2x2 identity() noexcept {
            return Float2x2{
                luisa::Float2{1.0f, 0.0f},
                luisa::Float2{0.0f, 1.0f}
            };
        }
    };
    
    struct Float3x3 {
        luisa::Float3 cols[3];
        
        Float3x3() = default;
        Float3x3(const luisa::Float3& c0, const luisa::Float3& c1, const luisa::Float3& c2) 
            : cols{c0, c1, c2} {}
        
        [[nodiscard]] constexpr luisa::Float3& operator[](size_t i) noexcept { return cols[i]; }
        [[nodiscard]] constexpr const luisa::Float3& operator[](size_t i) const noexcept { return cols[i]; }
        
        static Float3x3 identity() noexcept {
            return Float3x3{
                luisa::Float3{1.0f, 0.0f, 0.0f},
                luisa::Float3{0.0f, 1.0f, 0.0f},
                luisa::Float3{0.0f, 0.0f, 1.0f}
            };
        }
        
        static Float3x3 zero() noexcept {
            return Float3x3{
                luisa::Float3{0.0f},
                luisa::Float3{0.0f},
                luisa::Float3{0.0f}
            };
        }
    };
    
    struct Float6x6 {
        Float6 cols[6];
        
        Float6x6() = default;
        
        [[nodiscard]] constexpr Float6& operator[](size_t i) noexcept { return cols[i]; }
        [[nodiscard]] constexpr const Float6& operator[](size_t i) const noexcept { return cols[i]; }
        
        static Float6x6 identity() noexcept {
            Float6x6 m;
            for(int i = 0; i < 6; ++i) {
                m[i].fill(0.0f);
                m[i][i] = 1.0f;
            }
            return m;
        }
        
        static Float6x6 zero() noexcept {
            Float6x6 m;
            for(int i = 0; i < 6; ++i) {
                m[i].fill(0.0f);
            }
            return m;
        }
    };
    
    struct Float6x9 {
        Float9 cols[6];
        
        Float6x9() = default;
        
        [[nodiscard]] constexpr Float9& operator[](size_t i) noexcept { return cols[i]; }
        [[nodiscard]] constexpr const Float9& operator[](size_t i) const noexcept { return cols[i]; }
        
        static Float6x9 zero() noexcept {
            Float6x9 m;
            for(int i = 0; i < 6; ++i) {
                m[i].fill(0.0f);
            }
            return m;
        }
    };
    
    // Helper operators for custom types
    inline Float3x2 operator*(const Float3x2& a, const Float2x2& b) {
        Float3x2 result;
        // Column-major: result.col[i] = sum_j(a.col[j] * b[j][i])
        for(int i = 0; i < 2; ++i) {
            result[i] = a[0] * b[0][i] + a[1] * b[1][i];
        }
        return result;
    }
    
    inline Float3x2 operator*(const Float3x2& m, float s) {
        return Float3x2{m[0] * s, m[1] * s};
    }
    
    inline Float3x2 operator+(const Float3x2& a, const Float3x2& b) {
        return Float3x2{a[0] + b[0], a[1] + b[1]};
    }
    
    inline Float3x3 operator+(const Float3x3& a, const Float3x3& b) {
        return Float3x3{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    }
    
    inline Float3x3 operator*(const Float3x3& a, const Float3x3& b) {
        Float3x3 result = Float3x3::zero();
        for(int col = 0; col < 3; ++col) {
            for(int row = 0; row < 3; ++row) {
                float sum = 0.0f;
                for(int k = 0; k < 3; ++k) {
                    sum += a[k][row] * b[col][k];
                }
                result[col][row] = sum;
            }
        }
        return result;
    }
    
    inline Float3x3 operator*(const Float3x3& m, float s) {
        return Float3x3{m[0] * s, m[1] * s, m[2] * s};
    }
    
    inline Float3x3 operator/(const Float3x3& m, float s) {
        return Float3x3{m[0] / s, m[1] / s, m[2] / s};
    }
    
    inline luisa::Float3 operator*(const Float3x3& m, const luisa::Float3& v) {
        return m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    }
    
    inline Float6x6 operator+(const Float6x6& a, const Float6x6& b) {
        Float6x6 result;
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 6; ++j) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }
    
    inline Float6x6 operator*(const Float6x6& m, float s) {
        Float6x6 result;
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 6; ++j) {
                result[i][j] = m[i][j] * s;
            }
        }
        return result;
    }
    
    inline Float6 operator*(const Float6x6& m, const Float6& v) {
        Float6 result;
        for(int i = 0; i < 6; ++i) {
            result[i] = 0.0f;
            for(int j = 0; j < 6; ++j) {
                result[i] += m[j][i] * v[j];  // column-major
            }
        }
        return result;
    }
    
    inline Float6x6 operator*(const Float6x6& a, const Float6x6& b) {
        Float6x6 result = Float6x6::zero();
        for(int col = 0; col < 6; ++col) {
            for(int row = 0; row < 6; ++row) {
                float sum = 0.0f;
                for(int k = 0; k < 6; ++k) {
                    sum += a[k][row] * b[col][k];
                }
                result[col][row] = sum;
            }
        }
        return result;
    }
    
    inline Float6 operator*(float s, const Float6& v) {
        Float6 result;
        for(int i = 0; i < 6; ++i) result[i] = s * v[i];
        return result;
    }
    
    inline Float6 operator+(const Float6& a, const Float6& b) {
        Float6 result;
        for(int i = 0; i < 6; ++i) result[i] = a[i] + b[i];
        return result;
    }
    
    inline float dot(const Float6& a, const Float6& b) {
        float sum = 0.0f;
        for(int i = 0; i < 6; ++i) sum += a[i] * b[i];
        return sum;
    }
    
    inline float length_squared(const Float6& v) {
        return dot(v, v);
    }
    
    inline float length(const Float6& v) {
        return std::sqrt(length_squared(v));
    }
    
    inline Float6 normalize(const Float6& v) {
        float len = length(v);
        Float6 result;
        for(int i = 0; i < 6; ++i) result[i] = v[i] / len;
        return result;
    }
    
    inline Float6x6 outer_product(const Float6& a, const Float6& b) {
        Float6x6 result;
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 6; ++j) {
                result[i][j] = a[j] * b[i];  // column-major: result[col][row]
            }
        }
        return result;
    }
    
    // Transpose operations
    inline Float2x2 transpose(const Float3x2& m) {
        // 3x2 -> 2x3 (represented as 2x2 with Float2 cols)
        Float2x2 result;
        result[0] = luisa::Float2(m[0][0], m[1][0]);
        result[1] = luisa::Float2(m[0][1], m[1][1]);
        return result;
    }
    
    inline luisa::Float3 operator*(const Float3x2& m, const luisa::Float2& v) {
        return m[0] * v[0] + m[1] * v[1];
    }
    
    inline luisa::Float2 operator*(const Float2x2& m, const luisa::Float2& v) {
        return m[0] * v[0] + m[1] * v[1];
    }
    
    inline Float2x2 operator*(const Float2x2& a, const Float2x2& b) {
        Float2x2 result;
        for(int col = 0; col < 2; ++col) {
            for(int row = 0; row < 2; ++row) {
                float sum = 0.0f;
                for(int k = 0; k < 2; ++k) {
                    sum += a[k][row] * b[col][k];
                }
                result.cols[col][row] = sum;
            }
        }
        return result;
    }
    
    inline float length_squared(const Float3x2& m) {
        return luisa::length_squared(m[0]) + luisa::length_squared(m[1]);
    }

    // Flatten: 3x2 matrix -> 6-vector (column-major)
    inline Float6 flatten(const Float3x2& F)
    {
        Float6 R;
        R[0] = F[0][0];
        R[1] = F[0][1];
        R[2] = F[0][2];
        R[3] = F[1][0];
        R[4] = F[1][1];
        R[5] = F[1][2];
        return R;
    }

    inline Float3x2 F3x2(const Float3x2& Ds3x2,
                         const Float2x2& Dms2x2_inv)
    {
        return Ds3x2 * Dms2x2_inv;
    }

    inline Float3x2 Ds3x2(const luisa::Float3& x0,
                          const luisa::Float3& x1,
                          const luisa::Float3& x2)
    {
        Float3x2 M;
        M[0] = x1 - x0;
        M[1] = x2 - x0;
        return M;
    }


    inline Float2x2 Dm2x2(const luisa::Float3& x0, const luisa::Float3& x1, const luisa::Float3& x2)
    {
        luisa::Float3 v01 = x1 - x0;
        luisa::Float3 v02 = x2 - x0;
        // compute uv coordinates by rotating each triangle normal to (0, 1, 0)
        luisa::Float3 normal = luisa::normalize(luisa::cross(v01, v02));
        luisa::Float3 target = luisa::Float3(0.0f, 0.0f, 1.0f);

        luisa::Float3   vec      = luisa::cross(normal, target);
        float           cos_val  = luisa::dot(normal, target);
        Float3x3 rotation = Float3x3::identity();

        if(cos_val + 1.0f == 0.0f)
        {
            rotation[0][0] = -1.0f;
            rotation[1][1] = -1.0f;
        }
        else
        {
            Float3x3 cross_vec = Float3x3::zero();

            cross_vec[1][0] = vec.z;
            cross_vec[2][0] = -vec.y;
            cross_vec[0][1] = -vec.z;
            cross_vec[2][1] = vec.x;
            cross_vec[0][2] = vec.y;
            cross_vec[1][2] = -vec.x;

            rotation = rotation + cross_vec + cross_vec * cross_vec / (1.0f + cos_val);
        }

        luisa::Float3 rotate_uv0 = rotation * x0;
        luisa::Float3 rotate_uv1 = rotation * x1;
        luisa::Float3 rotate_uv2 = rotation * x2;

        auto uv0 = luisa::Float2(rotate_uv0.x, rotate_uv0.y);
        auto uv1 = luisa::Float2(rotate_uv1.x, rotate_uv1.y);
        auto uv2 = luisa::Float2(rotate_uv2.x, rotate_uv2.y);


        Float2x2 M;
        M[0] = uv1 - uv0;
        M[1] = uv2 - uv0;

        return M;
    }

    inline Float6x9 dFdX(const Float2x2& DmI)
    {
        Float6x9 dfdx = Float6x9::zero();
        float d0 = DmI[0][0];
        float d1 = DmI[1][0];
        float d2 = DmI[0][1];
        float d3 = DmI[1][1];
        float s0 = d0 + d1;
        float s1 = d2 + d3;

        for(int i = 0; i < 3; i++)
        {
            dfdx[i][i]     = -s0;
            dfdx[i + 3][i] = -s1;
        }

        for(int i = 0; i < 3; i++)
        {
            dfdx[i][i + 3]     = d0;
            dfdx[i + 3][i + 3] = d2;
        }

        for(int i = 0; i < 3; i++)
        {
            dfdx[i][i + 6]     = d1;
            dfdx[i + 3][i + 6] = d3;
        }

        return dfdx;
    }


    inline float E(const Float3x2& F,
                   const luisa::Float2&   anisotropic_a,
                   const luisa::Float2&   anisotropic_b,
                   float                  stretchS,
                   float                  shearS,
                   float                  strainRate)
    {
        // Compute F^T * F as 2x2 matrix
        float ft_f_00 = luisa::dot(F[0], F[0]);
        float ft_f_01 = luisa::dot(F[0], F[1]);
        float ft_f_10 = ft_f_01;
        float ft_f_11 = luisa::dot(F[1], F[1]);
        
        // I6 = a^T * F^T * F * b
        float I6 = anisotropic_a.x * (ft_f_00 * anisotropic_b.x + ft_f_01 * anisotropic_b.y)
                 + anisotropic_a.y * (ft_f_10 * anisotropic_b.x + ft_f_11 * anisotropic_b.y);
        float shear_energy = I6 * I6;

        float I5u = luisa::length(F * anisotropic_a);
        float I5v = luisa::length(F * anisotropic_b);

        float ucoeff = 1.0f;
        float vcoeff = 1.0f;

        if(I5u <= 1.0f)
        {
            ucoeff = 0.0f;
        }
        if(I5v <= 1.0f)
        {
            vcoeff = 0.0f;
        }


        float stretch_energy =
            std::pow(I5u - 1.0f, 2) + ucoeff * strainRate * std::pow(I5u - 1.0f, 3)
            + std::pow(I5v - 1.0f, 2) + vcoeff * strainRate * std::pow(I5v - 1.0f, 3);

        return (stretchS * stretch_energy + shearS * shear_energy);
    }

    inline void dEdF(Float3x2&       R,
                     const Float3x2& F,
                     const luisa::Float2&   anisotropic_a,
                     const luisa::Float2&   anisotropic_b,
                     float                  stretchS,
                     float                  shearS,
                     float                  strainRate)
    {
        // Compute F^T * F
        float ft_f_00 = luisa::dot(F[0], F[0]);
        float ft_f_01 = luisa::dot(F[0], F[1]);
        float ft_f_10 = ft_f_01;
        float ft_f_11 = luisa::dot(F[1], F[1]);
        
        float I6 = anisotropic_a.x * (ft_f_00 * anisotropic_b.x + ft_f_01 * anisotropic_b.y)
                 + anisotropic_a.y * (ft_f_10 * anisotropic_b.x + ft_f_11 * anisotropic_b.y);
        
        Float3x2 stretch_pk1 = Float3x2::zero();
        Float3x2 shear_pk1;

        // shear_pk1 = 2 * (I6 - a^T * b) * (F * a * b^T + F * b * a^T)
        float at_b = anisotropic_a.x * anisotropic_b.x + anisotropic_a.y * anisotropic_b.y;
        luisa::Float3 Fa = F[0] * anisotropic_a.x + F[1] * anisotropic_a.y;
        luisa::Float3 Fb = F[0] * anisotropic_b.x + F[1] * anisotropic_b.y;
        
        float coeff = 2.0f * (I6 - at_b);
        shear_pk1[0] = coeff * (Fa * anisotropic_b.x + Fb * anisotropic_a.x);
        shear_pk1[1] = coeff * (Fa * anisotropic_b.y + Fb * anisotropic_a.y);
        
        float I5u    = luisa::dot(Fa, Fa);
        float I5v    = luisa::dot(Fb, Fb);
        float ucoeff = 1.0f - 1.0f / std::sqrt(I5u);
        float vcoeff = 1.0f - 1.0f / std::sqrt(I5v);

        if(I5u > 1.0f)
        {
            ucoeff += 1.5f * strainRate * (std::sqrt(I5u) + 1.0f / std::sqrt(I5u) - 2.0f);
        }
        if(I5v > 1.0f)
        {
            vcoeff += 1.5f * strainRate * (std::sqrt(I5v) + 1.0f / std::sqrt(I5v) - 2.0f);
        }

        // stretch_pk1 = ucoeff * 2 * F * a * a^T + vcoeff * 2 * F * b * b^T
        stretch_pk1[0] = ucoeff * 2.0f * (Fa * anisotropic_a.x) + vcoeff * 2.0f * (Fb * anisotropic_b.x);
        stretch_pk1[1] = ucoeff * 2.0f * (Fa * anisotropic_a.y) + vcoeff * 2.0f * (Fb * anisotropic_b.y);

        R[0] = stretch_pk1[0] * stretchS + shear_pk1[0] * shearS;
        R[1] = stretch_pk1[1] * stretchS + shear_pk1[1] * shearS;
    }

    inline void ddEddF(Float6x6&       R,
                       const Float3x2& F,
                       const luisa::Float2&   anisotropic_a,
                       const luisa::Float2&   anisotropic_b,
                       float                  stretchS,
                       float                  shearS,
                       float                  strainRate)
    {
        Float6x6 H_stretc = Float6x6::zero();
        {
            luisa::Float3 Fa = F[0] * anisotropic_a.x + F[1] * anisotropic_a.y;
            luisa::Float3 Fb = F[0] * anisotropic_b.x + F[1] * anisotropic_b.y;
            
            float I5u = luisa::dot(Fa, Fa);
            float I5v = luisa::dot(Fb, Fb);
            float invSqrtI5u = 1.0f / std::sqrt(I5u);
            float invSqrtI5v = 1.0f / std::sqrt(I5v);

            float sqrtI5u = std::sqrt(I5u);
            float sqrtI5v = std::sqrt(I5v);

            if(sqrtI5u > 1.0f)
            {
                float val = 2.0f * (((sqrtI5u - 1.0f) * (3.0f * sqrtI5u * strainRate - 3.0f * strainRate + 2.0f))
                       / (2.0f * sqrtI5u));
                H_stretc[0][0] = val;
                H_stretc[1][1] = val;
                H_stretc[2][2] = val;
            }
            if(sqrtI5v > 1.0f)
            {
                float val = 2.0f * (((sqrtI5v - 1.0f) * (3.0f * sqrtI5v * strainRate - 3.0f * strainRate + 2.0f))
                       / (2.0f * sqrtI5v));
                H_stretc[3][3] = val;
                H_stretc[4][4] = val;
                H_stretc[5][5] = val;
            }

            auto fu = luisa::normalize(F[0]);
            auto fv = luisa::normalize(F[1]);

            float uCoeff = (sqrtI5u > 1.0f) ?
                               (3.0f * I5u * strainRate - 3.0f * strainRate + 2.0f) / (std::sqrt(I5u)) :
                               2.0f;

            float vCoeff = (sqrtI5v > 1.0f) ?
                               (3.0f * I5v * strainRate - 3.0f * strainRate + 2.0f) / (std::sqrt(I5v)) :
                               2.0f;

            // H_stretc.block<3, 3>(0, 0)
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    H_stretc[i][j] += uCoeff * fu[i] * fu[j];
            
            // H_stretc.block<3, 3>(3, 3)
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    H_stretc[i + 3][j + 3] += vCoeff * fv[i] * fv[j];
        }

        Float6x6 H_shear = Float6x6::zero();
        {
            Float6x6 H = Float6x6::zero();
            H[0][3] = H[1][4] = H[2][5] = H[3][0] = H[4][1] = H[5][2] = 1.0f;
            
            // Compute F^T * F
            float ft_f_00 = luisa::dot(F[0], F[0]);
            float ft_f_01 = luisa::dot(F[0], F[1]);
            float ft_f_10 = ft_f_01;
            float ft_f_11 = luisa::dot(F[1], F[1]);
            
            float I6 = anisotropic_a.x * (ft_f_00 * anisotropic_b.x + ft_f_01 * anisotropic_b.y)
                     + anisotropic_a.y * (ft_f_10 * anisotropic_b.x + ft_f_11 * anisotropic_b.y);
                     
            float signI6 = (I6 >= 0.0f) ? 1.0f : -1.0f;
            
            // g = F * (a * b^T + b * a^T)
            luisa::Float3 g0 = F[0] * (anisotropic_a.x * anisotropic_b.x + anisotropic_b.x * anisotropic_a.x)
                             + F[1] * (anisotropic_a.y * anisotropic_b.x + anisotropic_b.y * anisotropic_a.x);
            luisa::Float3 g1 = F[0] * (anisotropic_a.x * anisotropic_b.y + anisotropic_b.x * anisotropic_a.y)
                             + F[1] * (anisotropic_a.y * anisotropic_b.y + anisotropic_b.y * anisotropic_a.y);
            
            Float6 vec_g;
            vec_g[0] = g0.x;
            vec_g[1] = g0.y;
            vec_g[2] = g0.z;
            vec_g[3] = g1.x;
            vec_g[4] = g1.y;
            vec_g[5] = g1.z;

            float I2 = length_squared(F);
            float lambda0 = 0.5f * (I2 + std::sqrt(I2 * I2 + 12.0f * I6 * I6));
            
            // Compute q0 = normalized(I6 * H * vec_g + lambda0 * vec_g)
            Float6 H_vec_g = H * vec_g;
            Float6 temp;
            for(int i = 0; i < 6; ++i) temp[i] = I6 * H_vec_g[i] + lambda0 * vec_g[i];
            Float6 q0 = normalize(temp);
            
            Float6x6 T = Float6x6::identity();
            T = Float6x6::identity() * 0.5f + H * 0.5f * signI6;
            Float6  Tq     = T * q0;
            float normTq = length_squared(Tq);
            
            // H_shear = |I6| * (T - (Tq * Tq^T) / normTq) + lambda0 * (q0 * q0^T)
            H_shear = outer_product(Tq, Tq);
            for(int i = 0; i < 6; ++i)
                for(int j = 0; j < 6; ++j)
                    H_shear[i][j] = std::abs(I6) * (T[i][j] - H_shear[i][j] / normTq) + lambda0 * q0[i] * q0[j];
            
            H_shear = H_shear * 2.0f;
        }

        R = H_stretc * stretchS + H_shear * shearS;
    }
}  // namespace sym::strainlimiting_baraff_witkin_shell_2d
}  // namespace uipc::backend::luisa
