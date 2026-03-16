#include <finite_element/matrix_utils.h>
#include <algorithm/qr_svd.hpp>

namespace uipc::backend::luisa
{
UIPC_GENERIC Vector9 flatten(const Matrix3x3& A) noexcept
{
    Vector9 column;
    // Column-major order (consistent with Eigen)
    column[0] = A[0][0];
    column[1] = A[0][1];
    column[2] = A[0][2];
    column[3] = A[1][0];
    column[4] = A[1][1];
    column[5] = A[1][2];
    column[6] = A[2][0];
    column[7] = A[2][1];
    column[8] = A[2][2];
    return column;
}

UIPC_GENERIC Matrix3x3 unflatten(const Vector9& v) noexcept
{
    Matrix3x3 A;
    // Column-major order
    A[0][0] = v[0];
    A[0][1] = v[1];
    A[0][2] = v[2];
    A[1][0] = v[3];
    A[1][1] = v[4];
    A[1][2] = v[5];
    A[2][0] = v[6];
    A[2][1] = v[7];
    A[2][2] = v[8];
    return A;
}

UIPC_GENERIC Float ddot(const Matrix3x3& A, const Matrix3x3& B)
{
    Float result = 0.0f;
    for(int j = 0; j < 3; j++)
        for(int i = 0; i < 3; i++)
            result += A[j][i] * B[j][i];
    return result;
}

UIPC_GENERIC void svd(const Matrix3x3& F, Matrix3x3& U, Vector3& Sigma, Matrix3x3& V) noexcept
{
    math::qr_svd(F, Sigma, U, V);
}

UIPC_GENERIC void polar_decomposition(const Matrix3x3& F, Matrix3x3& R, Matrix3x3& S) noexcept
{
    // Use SVD-based polar decomposition
    Matrix3x3 U, V;
    Vector3 Sigma;
    svd(F, U, Sigma, V);
    
    // R = U * V^T
    R = U * luisa::transpose(V);
    
    // S = V * Sigma * V^T
    Matrix3x3 Sigma_diag = luisa::make_float3x3(
        Sigma.x, 0.0f, 0.0f,
        0.0f, Sigma.y, 0.0f,
        0.0f, 0.0f, Sigma.z
    );
    S = V * Sigma_diag * luisa::transpose(V);
}

// Helper function for Jacobi eigenvalue decomposition
namespace detail
{
    // Perform a Jacobi rotation to eliminate off-diagonal element
    template<typename MatrixType, typename VectorType, int N>
    LUISA_DEVICE inline void jacobi_rotate(MatrixType& A, MatrixType& Q, int p, int q, VectorType& eigenvalues)
    {
        if(luisa::abs(A[p][q]) < 1e-10f) return;
        
        Float tau = (A[q][q] - A[p][p]) / (2.0f * A[p][q]);
        Float t;
        if(tau >= 0.0f)
            t = 1.0f / (tau + luisa::sqrt(1.0f + tau * tau));
        else
            t = 1.0f / (tau - luisa::sqrt(1.0f + tau * tau));
        
        Float c = 1.0f / luisa::sqrt(1.0f + t * t);
        Float s = t * c;
        
        // Update eigenvalues
        Float app = A[p][p];
        Float aqq = A[q][q];
        A[p][p] = c * c * app - 2.0f * s * c * A[p][q] + s * s * aqq;
        A[q][q] = s * s * app + 2.0f * s * c * A[p][q] + c * c * aqq;
        A[p][q] = 0.0f;
        A[q][p] = 0.0f;
        
        // Update remaining elements
        for(int i = 0; i < N; i++)
        {
            if(i != p && i != q)
            {
                Float aip = A[i][p];
                Float aiq = A[i][q];
                A[i][p] = c * aip - s * aiq;
                A[p][i] = A[i][p];
                A[i][q] = s * aip + c * aiq;
                A[q][i] = A[i][q];
            }
        }
        
        // Update eigenvector matrix
        for(int i = 0; i < N; i++)
        {
            Float qip = Q[i][p];
            Float qiq = Q[i][q];
            Q[i][p] = c * qip - s * qiq;
            Q[i][q] = s * qip + c * qiq;
        }
    }
}

UIPC_GENERIC void evd(const Matrix3x3& A, Vector3& eigen_values, Matrix3x3& eigen_vectors) noexcept
{
    // Use analytical method for 3x3 symmetric matrices
    // First, symmetrize the input
    Matrix3x3 S;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            S[j][i] = 0.5f * (A[j][i] + A[i][j]);
    
    // Initialize eigenvectors as identity
    eigen_vectors = luisa::make_float3x3(1.0f);
    
    // Jacobi iterations for 3x3 matrix
    for(int iter = 0; iter < 10; iter++)
    {
        // Find largest off-diagonal element
        Float max_val = 0.0f;
        int p = 0, q = 1;
        for(int i = 0; i < 3; i++)
        {
            for(int j = i + 1; j < 3; j++)
            {
                if(luisa::abs(S[j][i]) > max_val)
                {
                    max_val = luisa::abs(S[j][i]);
                    p = i;
                    q = j;
                }
            }
        }
        
        if(max_val < 1e-10f) break;
        
        detail::jacobi_rotate<Matrix3x3, Vector3, 3>(S, eigen_vectors, p, q, eigen_values);
    }
    
    // Extract eigenvalues
    eigen_values.x = S[0][0];
    eigen_values.y = S[1][1];
    eigen_values.z = S[2][2];
}

UIPC_GENERIC void evd(const Matrix9x9& A, Vector9& eigen_values, Matrix9x9& eigen_vectors) noexcept
{
    // Jacobi eigenvalue decomposition for 9x9 symmetric matrix
    // Initialize working matrix (symmetrized)
    Matrix9x9 S = A;
    for(int i = 0; i < 9; i++)
        for(int j = i + 1; j < 9; j++)
            S[j][i] = S[i][j] = 0.5f * (A[i][j] + A[j][i]);
    
    // Initialize eigenvectors as identity
    for(int i = 0; i < 9; i++)
        for(int j = 0; j < 9; j++)
            eigen_vectors[i][j] = (i == j) ? 1.0f : 0.0f;
    
    // Jacobi iterations
    for(int iter = 0; iter < 50; iter++)
    {
        // Find largest off-diagonal element
        Float max_val = 0.0f;
        int p = 0, q = 1;
        for(int i = 0; i < 9; i++)
        {
            for(int j = i + 1; j < 9; j++)
            {
                if(luisa::abs(S[i][j]) > max_val)
                {
                    max_val = luisa::abs(S[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        
        if(max_val < 1e-10f) break;
        
        // Compute rotation
        Float tau = (S[q][q] - S[p][p]) / (2.0f * S[p][q]);
        Float t;
        if(tau >= 0.0f)
            t = 1.0f / (tau + luisa::sqrt(1.0f + tau * tau));
        else
            t = 1.0f / (tau - luisa::sqrt(1.0f + tau * tau));
        
        Float c = 1.0f / luisa::sqrt(1.0f + t * t);
        Float s = t * c;
        
        // Apply rotation
        Float app = S[p][p];
        Float aqq = S[q][q];
        S[p][p] = c * c * app - 2.0f * s * c * S[p][q] + s * s * aqq;
        S[q][q] = s * s * app + 2.0f * s * c * S[p][q] + c * c * aqq;
        S[p][q] = S[q][p] = 0.0f;
        
        for(int i = 0; i < 9; i++)
        {
            if(i != p && i != q)
            {
                Float aip = S[i][p];
                Float aiq = S[i][q];
                S[i][p] = S[p][i] = c * aip - s * aiq;
                S[i][q] = S[q][i] = s * aip + c * aiq;
            }
        }
        
        // Update eigenvectors
        for(int i = 0; i < 9; i++)
        {
            Float qip = eigen_vectors[i][p];
            Float qiq = eigen_vectors[i][q];
            eigen_vectors[i][p] = c * qip - s * qiq;
            eigen_vectors[i][q] = s * qip + c * qiq;
        }
    }
    
    // Extract eigenvalues
    for(int i = 0; i < 9; i++)
        eigen_values[i] = S[i][i];
}

UIPC_GENERIC void evd(const Matrix12x12& A, Vector12& eigen_values, Matrix12x12& eigen_vectors) noexcept
{
    // Jacobi eigenvalue decomposition for 12x12 symmetric matrix
    // Initialize working matrix (symmetrized)
    Matrix12x12 S = A;
    for(int i = 0; i < 12; i++)
        for(int j = i + 1; j < 12; j++)
            S[j][i] = S[i][j] = 0.5f * (A[i][j] + A[j][i]);
    
    // Initialize eigenvectors as identity
    for(int i = 0; i < 12; i++)
        for(int j = 0; j < 12; j++)
            eigen_vectors[i][j] = (i == j) ? 1.0f : 0.0f;
    
    // Jacobi iterations
    for(int iter = 0; iter < 60; iter++)
    {
        // Find largest off-diagonal element
        Float max_val = 0.0f;
        int p = 0, q = 1;
        for(int i = 0; i < 12; i++)
        {
            for(int j = i + 1; j < 12; j++)
            {
                if(luisa::abs(S[i][j]) > max_val)
                {
                    max_val = luisa::abs(S[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        
        if(max_val < 1e-10f) break;
        
        // Compute rotation
        Float tau = (S[q][q] - S[p][p]) / (2.0f * S[p][q]);
        Float t;
        if(tau >= 0.0f)
            t = 1.0f / (tau + luisa::sqrt(1.0f + tau * tau));
        else
            t = 1.0f / (tau - luisa::sqrt(1.0f + tau * tau));
        
        Float c = 1.0f / luisa::sqrt(1.0f + t * t);
        Float s = t * c;
        
        // Apply rotation
        Float app = S[p][p];
        Float aqq = S[q][q];
        S[p][p] = c * c * app - 2.0f * s * c * S[p][q] + s * s * aqq;
        S[q][q] = s * s * app + 2.0f * s * c * S[p][q] + c * c * aqq;
        S[p][q] = S[q][p] = 0.0f;
        
        for(int i = 0; i < 12; i++)
        {
            if(i != p && i != q)
            {
                Float aip = S[i][p];
                Float aiq = S[i][q];
                S[i][p] = S[p][i] = c * aip - s * aiq;
                S[i][q] = S[q][i] = s * aip + c * aiq;
            }
        }
        
        // Update eigenvectors
        for(int i = 0; i < 12; i++)
        {
            Float qip = eigen_vectors[i][p];
            Float qiq = eigen_vectors[i][q];
            eigen_vectors[i][p] = c * qip - s * qiq;
            eigen_vectors[i][q] = s * qip + c * qiq;
        }
    }
    
    // Extract eigenvalues
    for(int i = 0; i < 12; i++)
        eigen_values[i] = S[i][i];
}

UIPC_GENERIC Matrix9x9 clamp_to_spd(const Matrix9x9& A) noexcept
{
    Matrix9x9 Q;
    Vector9 values;
    evd(A, values, Q);
    
    // Clamp eigenvalues to be non-negative
    for(int x = 0; x < 9; x++)
        values[x] = (values[x] > 0.0f) ? values[x] : 0.0f;
    
    // Reconstruct: A = Q * diag(values) * Q^T
    Matrix9x9 B{};
    for(int i = 0; i < 9; i++)
        for(int j = 0; j < 9; j++)
            for(int k = 0; k < 9; k++)
                B[i][j] += Q[i][k] * values[k] * Q[j][k];
    
    return B;
}

UIPC_GENERIC Matrix12x12 clamp_to_spd(const Matrix12x12& A) noexcept
{
    Matrix12x12 Q;
    Vector12 values;
    evd(A, values, Q);
    
    // Clamp eigenvalues to be non-negative
    for(int x = 0; x < 12; x++)
        values[x] = (values[x] > 0.0f) ? values[x] : 0.0f;
    
    // Reconstruct: A = Q * diag(values) * Q^T
    Matrix12x12 B{};
    for(int i = 0; i < 12; i++)
        for(int j = 0; j < 12; j++)
            for(int k = 0; k < 12; k++)
                B[i][j] += Q[i][k] * values[k] * Q[j][k];
    
    return B;
}

}  // namespace uipc::backend::luisa
