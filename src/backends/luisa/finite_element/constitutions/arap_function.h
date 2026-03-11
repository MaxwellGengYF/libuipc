#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/mathematics.h>

namespace uipc::backend::luisa
{
namespace sym::arap_3d
{
    constexpr double sqrt2 =
        1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276;

    // Helper function to flatten a 3x3 matrix to a 9-element vector (column-major)
    template <typename T>
    inline auto vec(const luisa::Matrix<T, 3>& mat) -> luisa::Vector<T, 9>
    {
        // Flatten column-major: each column becomes 3 consecutive elements
        return luisa::Vector<T, 9>{
            mat[0][0], mat[0][1], mat[0][2],  // Column 0
            mat[1][0], mat[1][1], mat[1][2],  // Column 1
            mat[2][0], mat[2][1], mat[2][2]   // Column 2
        };
    }

    // Helper function to compute squared Frobenius norm of (F - R)
    template <typename T>
    inline T squared_norm_diff(const luisa::Matrix<T, 3>& F,
                               const luisa::Matrix<T, 3>& R)
    {
        T sum = T(0);
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                T diff = F[i][j] - R[i][j];
                sum += diff * diff;
            }
        }
        return sum;
    }

    // Function to compute the ARAP energy
    // F: deformation gradient (3x3 matrix)
    // kappa: stiffness parameter
    // v: volume
    // energy: output energy value
    template <typename T>
    inline void E(T&                            energy,
                  const T&                      kappa,
                  const T&                      v,
                  const luisa::Matrix<T, 3>& F,
                  const luisa::Matrix<T, 3>& U,
                  const luisa::Vector<T, 3>& Sigma,
                  const luisa::Matrix<T, 3>& V)
    {
        // Compute rotation matrix R = U * V^T
        // V^T is the transpose of V
        auto VT = luisa::transpose(V);
        auto R = U * VT;
        
        // energy = kappa * v * ||F - R||^2
        energy = kappa * v * squared_norm_diff(F, R);
    }

    // Function to compute the gradient of the ARAP energy
    // gradients: output 9-element vector (flattened dPsi/dF)
    template <typename T>
    inline void dEdF(luisa::Vector<T, 9>&          gradients,
                     const T&                      kappa,
                     const T&                      v,
                     const luisa::Matrix<T, 3>& F,
                     const luisa::Matrix<T, 3>& U,
                     const luisa::Vector<T, 3>& Sigma,
                     const luisa::Matrix<T, 3>& V)
    {
        // Compute rotation matrix R = U * V^T
        auto VT = luisa::transpose(V);
        auto R = U * VT;

        // dPsi/dF = 2 * (F - R)
        luisa::Matrix<T, 3> dPsi_dF;
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                dPsi_dF[i][j] = T(2) * (F[i][j] - R[i][j]);
            }
        }

        T kv = kappa * v;

        // Flatten and scale: gradients = kv * vec(dPsi_dF)
        gradients[0] = kv * dPsi_dF[0][0];
        gradients[1] = kv * dPsi_dF[0][1];
        gradients[2] = kv * dPsi_dF[0][2];
        gradients[3] = kv * dPsi_dF[1][0];
        gradients[4] = kv * dPsi_dF[1][1];
        gradients[5] = kv * dPsi_dF[1][2];
        gradients[6] = kv * dPsi_dF[2][0];
        gradients[7] = kv * dPsi_dF[2][1];
        gradients[8] = kv * dPsi_dF[2][2];
    }

    // Function to compute the ARAP Hessian
    // H: output 9x9 Hessian matrix (stored as array of 9 column vectors)
    // F: deformation gradient
    // U, Sigma, V: SVD of F (F = U * diag(Sigma) * V^T)
    template <typename T>
    inline void ARAP_Hessian(luisa::Vector<T, 9>         H[9],  // H[col][row] = H[row, col]
                             const luisa::Matrix<T, 3>& F,
                             const luisa::Matrix<T, 3>& U,
                             const luisa::Vector<T, 3>& Sigma,
                             const luisa::Matrix<T, 3>& V)
    {
        // Define the twist modes (anti-symmetric matrices)
        // T0: rotation around z-axis (affects x-y plane)
        luisa::Matrix<T, 3> T0;
        T0[0] = luisa::Vector<T, 3>{T(0), T(-1), T(0)};
        T0[1] = luisa::Vector<T, 3>{T(1), T(0), T(0)};
        T0[2] = luisa::Vector<T, 3>{T(0), T(0), T(0)};
        
        // T1: rotation around x-axis (affects y-z plane)
        luisa::Matrix<T, 3> T1;
        T1[0] = luisa::Vector<T, 3>{T(0), T(0), T(0)};
        T1[1] = luisa::Vector<T, 3>{T(0), T(0), T(1)};
        T1[2] = luisa::Vector<T, 3>{T(0), T(-1), T(0)};
        
        // T2: rotation around y-axis (affects x-z plane)
        luisa::Matrix<T, 3> T2;
        T2[0] = luisa::Vector<T, 3>{T(0), T(0), T(1)};
        T2[1] = luisa::Vector<T, 3>{T(0), T(0), T(0)};
        T2[2] = luisa::Vector<T, 3>{T(-1), T(0), T(0)};

        // Transform twist modes: Ti = (1/sqrt(2)) * U * Ti * V^T
        auto VT = luisa::transpose(V);
        T0 = (T(1) / T(sqrt2)) * (U * (T0 * VT));
        T1 = (T(1) / T(sqrt2)) * (U * (T1 * VT));
        T2 = (T(1) / T(sqrt2)) * (U * (T2 * VT));

        // Flatten the twist modes
        auto t0 = vec(T0);
        auto t1 = vec(T1);
        auto t2 = vec(T2);

        // Get the singular values
        T s0 = Sigma[0];
        T s1 = Sigma[1];
        T s2 = Sigma[2];

        // Compute the Hessian: H = 2*I - sum_i (4/(si+sj)) * ti * ti^T
        // Initialize H as 2*I (identity scaled by 2)
        for(int i = 0; i < 9; ++i)
        {
            for(int j = 0; j < 9; ++j)
            {
                H[i][j] = (i == j) ? T(2) : T(0);
            }
        }

        // Subtract rank-1 updates for each twist mode
        // H -= (4 / (s0 + s1)) * t0 * t0^T
        T scale0 = T(4) / (s0 + s1);
        for(int i = 0; i < 9; ++i)
        {
            for(int j = 0; j < 9; ++j)
            {
                H[i][j] -= scale0 * t0[i] * t0[j];
            }
        }

        // H -= (4 / (s1 + s2)) * t1 * t1^T
        T scale1 = T(4) / (s1 + s2);
        for(int i = 0; i < 9; ++i)
        {
            for(int j = 0; j < 9; ++j)
            {
                H[i][j] -= scale1 * t1[i] * t1[j];
            }
        }

        // H -= (4 / (s0 + s2)) * t2 * t2^T
        T scale2 = T(4) / (s0 + s2);
        for(int i = 0; i < 9; ++i)
        {
            for(int j = 0; j < 9; ++j)
            {
                H[i][j] -= scale2 * t2[i] * t2[j];
            }
        }
    }

    // Function to compute the Hessian of the ARAP energy
    // hessians: output 9x9 Hessian matrix (stored as array of 9 column vectors)
    template <typename T>
    inline void ddEddF(luisa::Vector<T, 9>         hessians[9],
                       const T&                      kappa,
                       const T&                      v,
                       const luisa::Matrix<T, 3>& F,
                       const luisa::Matrix<T, 3>& U,
                       const luisa::Vector<T, 3>& Sigma,
                       const luisa::Matrix<T, 3>& V)
    {
        ARAP_Hessian(hessians, F, U, Sigma, V);
        
        // Scale by kappa * v
        T scale = kappa * v;
        for(int i = 0; i < 9; ++i)
        {
            for(int j = 0; j < 9; ++j)
            {
                hessians[i][j] *= scale;
            }
        }
    }

}  // namespace sym::arap_3d
}  // namespace uipc::backend::luisa
