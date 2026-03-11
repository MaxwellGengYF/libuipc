#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/mathematics.h>
#include <array>

namespace uipc::backend::luisa
{
namespace sym::abd_arap
{
    constexpr double sqrt2 =
        1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276;

    // Small fixed-size vector types for use in kernels
    template<typename T, size_t N>
    using Array = std::array<T, N>;

    using Vector3 = luisa::Vector<float, 3>;
    using Matrix3x3 = luisa::Matrix<float, 3>;

    // SVD implementation for 3x3 matrices using LuisaCompute types
    // Based on the analytical SVD for 3x3 matrices
    inline void svd(const Matrix3x3& F, Matrix3x3& U, Vector3& Sigma, Matrix3x3& V)
    {
        // Compute F^T * F for eigenvalue decomposition
        Matrix3x3 FtF = luisa::transpose(F) * F;
        
        // Compute eigenvalues of FtF (squares of singular values)
        // Using the characteristic polynomial method for symmetric 3x3
        float m00 = FtF[0][0], m01 = FtF[0][1], m02 = FtF[0][2];
        float m11 = FtF[1][1], m12 = FtF[1][2];
        float m22 = FtF[2][2];
        
        // Coefficients of characteristic polynomial: det(FtF - lambda*I) = 0
        float c0 = m00 * m11 * m22 + 2.0f * m01 * m12 * m02 
                   - m00 * m12 * m12 - m11 * m02 * m02 - m22 * m01 * m01;
        float c1 = m00 * m11 + m00 * m22 + m11 * m22 - m01 * m01 - m02 * m02 - m12 * m12;
        float c2 = -(m00 + m11 + m22);
        
        // Solve cubic equation for eigenvalues
        float p = c1 - c2 * c2 / 3.0f;
        float q = 2.0f * c2 * c2 * c2 / 27.0f - c2 * c1 / 3.0f + c0;
        float discriminant = q * q / 4.0f + p * p * p / 27.0f;
        
        float s0, s1, s2;
        
        if (discriminant >= 0.0f) {
            float sqrt_disc = std::sqrt(discriminant);
            float u = std::cbrt(-q / 2.0f + sqrt_disc);
            float v = std::cbrt(-q / 2.0f - sqrt_disc);
            s2 = u + v - c2 / 3.0f;
            
            // Other roots from quadratic factor
            float a = c2 + s2;
            float b = c1 + a * s2;
            float disc2 = a * a - 4.0f * b;
            if (disc2 >= 0.0f) {
                float sqrt_disc2 = std::sqrt(disc2);
                s0 = (-a - sqrt_disc2) / 2.0f;
                s1 = (-a + sqrt_disc2) / 2.0f;
            } else {
                s0 = s1 = -a / 2.0f;
            }
        } else {
            // Three distinct real roots
            float sqrt_neg_p = std::sqrt(-p / 3.0f);
            float theta = std::acos(-q / (2.0f * sqrt_neg_p * sqrt_neg_p * sqrt_neg_p)) / 3.0f;
            float twosqrt = 2.0f * sqrt_neg_p;
            s0 = twosqrt * std::cos(theta) - c2 / 3.0f;
            s1 = twosqrt * std::cos(theta + 2.0f * 3.14159265358979323846f / 3.0f) - c2 / 3.0f;
            s2 = twosqrt * std::cos(theta + 4.0f * 3.14159265358979323846f / 3.0f) - c2 / 3.0f;
        }
        
        // Sort singular values in descending order
        if (s0 < s1) std::swap(s0, s1);
        if (s1 < s2) std::swap(s1, s2);
        if (s0 < s1) std::swap(s0, s1);
        
        Sigma = luisa::Vector<float, 3>{std::sqrt(std::max(0.0f, s0)),
                                         std::sqrt(std::max(0.0f, s1)),
                                         std::sqrt(std::max(0.0f, s2))};
        
        // Compute V matrix (eigenvectors of FtF)
        // Using cross products for robust eigenvector computation
        Vector3 v0, v1, v2;
        
        // Eigenvector for largest eigenvalue s0
        Matrix3x3 A0 = FtF;
        A0[0][0] -= s0; A0[1][1] -= s0; A0[2][2] -= s0;
        v0 = luisa::cross(luisa::Vector<float, 3>{A0[0][0], A0[0][1], A0[0][2]},
                          luisa::Vector<float, 3>{A0[1][0], A0[1][1], A0[1][2]});
        if (luisa::dot(v0, v0) < 1e-10f) {
            v0 = luisa::cross(luisa::Vector<float, 3>{A0[0][0], A0[0][1], A0[0][2]},
                              luisa::Vector<float, 3>{A0[2][0], A0[2][1], A0[2][2]});
        }
        if (luisa::dot(v0, v0) < 1e-10f) {
            v0 = luisa::cross(luisa::Vector<float, 3>{A0[1][0], A0[1][1], A0[1][2]},
                              luisa::Vector<float, 3>{A0[2][0], A0[2][1], A0[2][2]});
        }
        v0 = luisa::normalize(v0);
        
        // Eigenvector for middle eigenvalue s1
        Matrix3x3 A1 = FtF;
        A1[0][0] -= s1; A1[1][1] -= s1; A1[2][2] -= s1;
        v1 = luisa::cross(luisa::Vector<float, 3>{A1[0][0], A1[0][1], A1[0][2]},
                          luisa::Vector<float, 3>{A1[1][0], A1[1][1], A1[1][2]});
        if (luisa::dot(v1, v1) < 1e-10f) {
            v1 = luisa::cross(luisa::Vector<float, 3>{A1[0][0], A1[0][1], A1[0][2]},
                              luisa::Vector<float, 3>{A1[2][0], A1[2][1], A1[2][2]});
        }
        if (luisa::dot(v1, v1) < 1e-10f) {
            v1 = luisa::cross(luisa::Vector<float, 3>{A1[1][0], A1[1][1], A1[1][2]},
                              luisa::Vector<float, 3>{A1[2][0], A1[2][1], A1[2][2]});
        }
        v1 = luisa::normalize(v1);
        
        // Third eigenvector is cross product of first two
        v2 = luisa::cross(v0, v1);
        
        V = Matrix3x3{v0, v1, v2};
        
        // Compute U from F = U * Sigma * V^T
        // U = F * V * inv(Sigma)
        Matrix3x3 FV = F * V;
        Vector3 inv_sigma = luisa::Vector<float, 3>{
            Sigma.x > 1e-10f ? 1.0f / Sigma.x : 0.0f,
            Sigma.y > 1e-10f ? 1.0f / Sigma.y : 0.0f,
            Sigma.z > 1e-10f ? 1.0f / Sigma.z : 0.0f
        };
        
        U[0] = FV[0] * inv_sigma.x;
        U[1] = FV[1] * inv_sigma.y;
        U[2] = FV[2] * inv_sigma.z;
        
        // Orthogonalize U if needed (for numerical stability)
        U[0] = luisa::normalize(U[0]);
        U[1] = luisa::normalize(U[1] - U[0] * luisa::dot(U[0], U[1]));
        U[2] = luisa::cross(U[0], U[1]);
    }

    // Function to extract F matrix from q vector (12-element array: translation + 3x3 matrix)
    inline void extractF(Matrix3x3& F, const Array<float, 12>& q)
    {
        // q[0-2] is translation, q[3-11] is the 3x3 matrix (column-major or row-major based on usage)
        // Original Eigen code: F.row(0) = q.segment<3>(3); suggests rows are stored sequentially
        F[0] = luisa::Vector<float, 3>{q[3], q[4], q[5]};   // First row
        F[1] = luisa::Vector<float, 3>{q[6], q[7], q[8]};   // Second row
        F[2] = luisa::Vector<float, 3>{q[9], q[10], q[11]}; // Third row
        F = luisa::transpose(F); // Convert to column-major for LuisaCompute
    }

    // Function to compute the ARAP energy
    inline void E(float& energy, const float& kappa, const Array<float, 12>& q)
    {
        Matrix3x3 F;
        extractF(F, q);
        Matrix3x3 R;
        Matrix3x3 U, V;
        Vector3 Sigma;
        svd(F, U, Sigma, V);
        R = U * luisa::transpose(V);
        
        Matrix3x3 diff = F - R;
        float squared_norm = 0.0f;
        for (int i = 0; i < 3; ++i) {
            squared_norm += luisa::dot(diff[i], diff[i]);
        }
        energy = kappa * squared_norm;
    }

    // Function to compute the gradient of the ARAP energy
    inline void dEdq(Array<float, 9>& gradients, const float& kappa, const Array<float, 12>& q)
    {
        Matrix3x3 F;
        extractF(F, q);
        Matrix3x3 R;
        Matrix3x3 U, V;
        Vector3 Sigma;
        svd(F, U, Sigma, V);
        R = U * luisa::transpose(V);

        Matrix3x3 dPsi_dF = (F - R) * 2.0f;

        // Flatten gradient in column-major order to match original vec() function
        // Original: vector << mat(0, 0), mat(1, 0), mat(2, 0), mat(0, 1), mat(1, 1), mat(2, 1), mat(0, 2), mat(1, 2), mat(2, 2);
        // This stores column by column
        gradients[0] = kappa * dPsi_dF[0][0];
        gradients[1] = kappa * dPsi_dF[0][1];
        gradients[2] = kappa * dPsi_dF[0][2];
        gradients[3] = kappa * dPsi_dF[1][0];
        gradients[4] = kappa * dPsi_dF[1][1];
        gradients[5] = kappa * dPsi_dF[1][2];
        gradients[6] = kappa * dPsi_dF[2][0];
        gradients[7] = kappa * dPsi_dF[2][1];
        gradients[8] = kappa * dPsi_dF[2][2];
    }

    // Function to flatten a matrix into a vector (column-major)
    inline Array<float, 9> vec(const Matrix3x3& mat)
    {
        Array<float, 9> vector;
        // Column-major flattening
        vector[0] = mat[0][0];
        vector[1] = mat[0][1];
        vector[2] = mat[0][2];
        vector[3] = mat[1][0];
        vector[4] = mat[1][1];
        vector[5] = mat[1][2];
        vector[6] = mat[2][0];
        vector[7] = mat[2][1];
        vector[8] = mat[2][2];
        return vector;
    }

    // Outer product for 9-element vectors to create 9x9 matrix
    struct Matrix9x9 {
        std::array<std::array<float, 9>, 9> data;
        
        std::array<float, 9>& operator[](size_t i) { return data[i]; }
        const std::array<float, 9>& operator[](size_t i) const { return data[i]; }
        
        Matrix9x9 operator*(float s) const {
            Matrix9x9 result;
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j)
                    result[i][j] = data[i][j] * s;
            return result;
        }
        
        Matrix9x9 operator-(const Matrix9x9& other) const {
            Matrix9x9 result;
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j)
                    result[i][j] = data[i][j] - other[i][j];
            return result;
        }
    };
    
    inline Matrix9x9 outer_product(const Array<float, 9>& a, const Array<float, 9>& b) {
        Matrix9x9 result;
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j)
                result[i][j] = a[i] * b[j];
        return result;
    }

    // Function to compute the ARAP Hessian
    inline void ARAP_Hessian(Matrix9x9& H, const Matrix3x3& F)
    {
        Matrix3x3 U, V;
        Vector3 Sigma;
        svd(F, U, Sigma, V);
        
        // Define the twist modes (skew-symmetric matrices)
        Matrix3x3 T0, T1, T2;
        // T0 = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
        T0[0] = luisa::Vector<float, 3>{0.0f, 1.0f, 0.0f};
        T0[1] = luisa::Vector<float, 3>{-1.0f, 0.0f, 0.0f};
        T0[2] = luisa::Vector<float, 3>{0.0f, 0.0f, 0.0f};
        
        // T1 = [[0, 0, 0], [0, 0, 1], [0, -1, 0]]
        T1[0] = luisa::Vector<float, 3>{0.0f, 0.0f, 0.0f};
        T1[1] = luisa::Vector<float, 3>{0.0f, 0.0f, -1.0f};
        T1[2] = luisa::Vector<float, 3>{0.0f, 1.0f, 0.0f};
        
        // T2 = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
        T2[0] = luisa::Vector<float, 3>{0.0f, 0.0f, -1.0f};
        T2[1] = luisa::Vector<float, 3>{0.0f, 0.0f, 0.0f};
        T2[2] = luisa::Vector<float, 3>{1.0f, 0.0f, 0.0f};

        float inv_sqrt2 = 1.0f / static_cast<float>(sqrt2);
        T0 = U * T0 * luisa::transpose(V) * inv_sqrt2;
        T1 = U * T1 * luisa::transpose(V) * inv_sqrt2;
        T2 = U * T2 * luisa::transpose(V) * inv_sqrt2;

        // Flatten the twist modes
        Array<float, 9> t0 = vec(T0);
        Array<float, 9> t1 = vec(T1);
        Array<float, 9> t2 = vec(T2);

        // Get the singular values
        float s0 = Sigma.x;
        float s1 = Sigma.y;
        float s2 = Sigma.z;

        // Compute the Hessian: H = 2*I - sum of projection terms
        // Initialize as identity * 2
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                H[i][j] = (i == j) ? 2.0f : 0.0f;
            }
        }
        
        // Subtract the twist mode projections
        float denom0 = s0 + s1;
        float denom1 = s1 + s2;
        float denom2 = s0 + s2;
        
        if (denom0 > 1e-10f) {
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j)
                    H[i][j] -= (4.0f / denom0) * t0[i] * t0[j];
        }
        if (denom1 > 1e-10f) {
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j)
                    H[i][j] -= (4.0f / denom1) * t1[i] * t1[j];
        }
        if (denom2 > 1e-10f) {
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j)
                    H[i][j] -= (4.0f / denom2) * t2[i] * t2[j];
        }
    }

    // Function to compute the Hessian of the ARAP energy
    inline void ddEddq(Matrix9x9& hessians, const float& kappa, const Array<float, 12>& q)
    {
        Matrix3x3 F;
        extractF(F, q);
        ARAP_Hessian(hessians, F);
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j)
                hessians[i][j] *= kappa;
    }

}  // namespace sym::abd_arap
}  // namespace uipc::backend::luisa
