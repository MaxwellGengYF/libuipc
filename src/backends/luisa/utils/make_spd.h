#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/mathematics.h>
#include <backends/luisa/type_define.h>

namespace uipc::backend::luisa
{
namespace detail
{
    // Helper to create a diagonal matrix from a vector
    template<int N>
    LUISA_GENERIC auto make_diagonal(const luisa::Vector<Float, N>& diag) noexcept
    {
        if constexpr(N == 2)
        {
            return luisa::Matrix<Float, 2>{
                luisa::Vector<Float, 2>{diag[0], 0.0f},
                luisa::Vector<Float, 2>{0.0f, diag[1]}};
        }
        else if constexpr(N == 3)
        {
            return luisa::Matrix<Float, 3>{
                luisa::Vector<Float, 3>{diag[0], 0.0f, 0.0f},
                luisa::Vector<Float, 3>{0.0f, diag[1], 0.0f},
                luisa::Vector<Float, 3>{0.0f, 0.0f, diag[2]}};
        }
        else if constexpr(N == 4)
        {
            return luisa::Matrix<Float, 4>{
                luisa::Vector<Float, 4>{diag[0], 0.0f, 0.0f, 0.0f},
                luisa::Vector<Float, 4>{0.0f, diag[1], 0.0f, 0.0f},
                luisa::Vector<Float, 4>{0.0f, 0.0f, diag[2], 0.0f},
                luisa::Vector<Float, 4>{0.0f, 0.0f, 0.0f, diag[3]}};
        }
    }

    // Jacobi eigenvalue decomposition for small symmetric matrices
    template<int N>
    LUISA_GENERIC void jacobi_evd(luisa::Matrix<Float, N>& H,
                                  luisa::Vector<Float, N>& eigen_values,
                                  luisa::Matrix<Float, N>& eigen_vectors) noexcept
    {
        // Initialize eigenvectors to identity
        eigen_vectors = luisa::Matrix<Float, N>{};
        for(int i = 0; i < N; ++i)
            eigen_vectors[i][i] = 1.0f;

        // Copy H to a working matrix A
        luisa::Matrix<Float, N> A = H;

        // Jacobi iterations
        constexpr int max_iterations = 50;
        constexpr Float epsilon = 1e-10f;

        for(int iter = 0; iter < max_iterations; ++iter)
        {
            // Find the largest off-diagonal element
            Float max_val = 0.0f;
            int p = 0, q = 0;
            for(int i = 0; i < N; ++i)
            {
                for(int j = i + 1; j < N; ++j)
                {
                    Float val = luisa::abs(A[i][j]);
                    if(val > max_val)
                    {
                        max_val = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if(max_val < epsilon)
                break;

            // Compute rotation
            Float app = A[p][p];
            Float aqq = A[q][q];
            Float apq = A[p][q];

            Float phi = 0.5f * luisa::atan2(2.0f * apq, aqq - app);
            Float c = luisa::cos(phi);
            Float s = luisa::sin(phi);

            // Update A
            A[p][p] = c * c * app - 2.0f * c * s * apq + s * s * aqq;
            A[q][q] = s * s * app + 2.0f * c * s * apq + c * c * aqq;
            A[p][q] = A[q][p] = 0.0f;

            for(int i = 0; i < N; ++i)
            {
                if(i != p && i != q)
                {
                    Float aip = A[i][p];
                    Float aiq = A[i][q];
                    A[i][p] = A[p][i] = c * aip - s * aiq;
                    A[i][q] = A[q][i] = s * aip + c * aiq;
                }
            }

            // Update eigenvectors
            for(int i = 0; i < N; ++i)
            {
                Float vip = eigen_vectors[i][p];
                Float viq = eigen_vectors[i][q];
                eigen_vectors[i][p] = c * vip - s * viq;
                eigen_vectors[i][q] = s * vip + c * viq;
            }
        }

        // Extract eigenvalues
        for(int i = 0; i < N; ++i)
            eigen_values[i] = A[i][i];
    }
}

template<int N>
UIPC_GENERIC void make_spd(luisa::Matrix<Float, N>& H)
{
    luisa::Vector<Float, N>    eigen_values;
    luisa::Matrix<Float, N> eigen_vectors;
    
    detail::jacobi_evd<N>(H, eigen_values, eigen_vectors);
    
    // Clamp negative eigenvalues to zero
    for(int i = 0; i < N; ++i)
    {
        eigen_values[i] = luisa::max(eigen_values[i], 0.0f);
    }
    
    // Reconstruct H = V * D * V^T
    auto D = detail::make_diagonal(eigen_values);
    H = eigen_vectors * D * luisa::transpose(eigen_vectors);
}
}  // namespace uipc::backend::luisa
