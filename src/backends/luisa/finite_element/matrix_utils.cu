#include <finite_element/matrix_utils.h>
#include <luisa/dsl/dsl.h>

namespace uipc::backend::luisa
{
LUISA_DEVICE Vector9 flatten(const Matrix3x3& A) noexcept
{
    // Column-major flattening for compatibility with Eigen/Fortran conventions
    Vector9 v;
    v[0] = A[0][0];
    v[1] = A[1][0];
    v[2] = A[2][0];
    v[3] = A[0][1];
    v[4] = A[1][1];
    v[5] = A[2][1];
    v[6] = A[0][2];
    v[7] = A[1][2];
    v[8] = A[2][2];
    return v;
}

LUISA_DEVICE Matrix3x3 unflatten(const Vector9& v) noexcept
{
    // Column-major unflattening
    Matrix3x3 A;
    A[0][0] = v[0];
    A[1][0] = v[1];
    A[2][0] = v[2];
    A[0][1] = v[3];
    A[1][1] = v[4];
    A[2][1] = v[5];
    A[0][2] = v[6];
    A[1][2] = v[7];
    A[2][2] = v[8];
    return A;
}

LUISA_DEVICE Float ddot(const Matrix3x3& A, const Matrix3x3& B)
{
    Float result = 0.0f;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            result += A[i][j] * B[i][j];
    return result;
}

// Helper: Create identity matrix
LUISA_DEVICE inline Matrix3x3 make_identity3() noexcept
{
    Matrix3x3 I = {};
    I[0][0] = 1.0f;
    I[1][1] = 1.0f;
    I[2][2] = 1.0f;
    return I;
}

// Helper: Matrix multiplication
LUISA_DEVICE inline Matrix3x3 mul(const Matrix3x3& A, const Matrix3x3& B) noexcept
{
    Matrix3x3 C = {};
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            for(int k = 0; k < 3; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Helper: Matrix transpose
LUISA_DEVICE inline Matrix3x3 transpose(const Matrix3x3& A) noexcept
{
    Matrix3x3 T;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            T[i][j] = A[j][i];
    return T;
}

// Helper: 3x3 determinant
LUISA_DEVICE inline Float determinant(const Matrix3x3& A) noexcept
{
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

// Helper: Matrix-vector multiplication
LUISA_DEVICE inline Vector3 mul(const Matrix3x3& A, const Vector3& v) noexcept
{
    Vector3 r;
    r[0] = A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2];
    r[1] = A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2];
    r[2] = A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2];
    return r;
}

// Helper: Scalar multiplication
LUISA_DEVICE inline Matrix3x3 scale(const Matrix3x3& A, Float s) noexcept
{
    Matrix3x3 R;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            R[i][j] = A[i][j] * s;
    return R;
}

// Helper: Matrix addition
LUISA_DEVICE inline Matrix3x3 add(const Matrix3x3& A, const Matrix3x3& B) noexcept
{
    Matrix3x3 R;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            R[i][j] = A[i][j] + B[i][j];
    return R;
}

// Helper: Givens rotation
LUISA_DEVICE inline void givens_rotation(Float a, Float b, Float& c, Float& s) noexcept
{
    if(b == 0.0f)
    {
        c = 1.0f;
        s = 0.0f;
    }
    else if(luisa::abs(b) > luisa::abs(a))
    {
        Float tau = -a / b;
        s = 1.0f / luisa::sqrt(1.0f + tau * tau);
        c = s * tau;
    }
    else
    {
        Float tau = -b / a;
        c = 1.0f / luisa::sqrt(1.0f + tau * tau);
        s = c * tau;
    }
}

// Analytical 3x3 SVD based on the approach by McAdams et al. 2011
// "Computing the Singular Value Decomposition of 3x3 matrices with minimal branching"
LUISA_DEVICE void svd(const Matrix3x3& F, Matrix3x3& U, Vector3& Sigma, Matrix3x3& V) noexcept
{
    // Compute A = F^T * F
    Matrix3x3 A = {};
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            for(int k = 0; k < 3; k++)
                A[i][j] += F[k][i] * F[k][j];

    // Compute eigenvalues of A (these are sigma^2)
    // Characteristic polynomial: det(A - lambda*I) = 0
    Float a = -(A[0][0] + A[1][1] + A[2][2]);
    Float b = A[0][0]*A[1][1] + A[1][1]*A[2][2] + A[2][2]*A[0][0] 
            - A[0][1]*A[1][0] - A[1][2]*A[2][1] - A[2][0]*A[0][2];
    Float c = -(A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
              - A[0][2]*A[1][1]*A[2][0] - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2]);

    // Solve cubic equation: lambda^3 + a*lambda^2 + b*lambda + c = 0
    // Using Cardano's formula
    Float p = b - a*a / 3.0f;
    Float q = 2.0f*a*a*a / 27.0f - a*b / 3.0f + c;
    Float discriminant = q*q / 4.0f + p*p*p / 27.0f;

    Float lambda[3];
    if(discriminant > 0.0f)
    {
        // One real root
        Float sqrt_disc = luisa::sqrt(discriminant);
        Float u = -q / 2.0f + sqrt_disc;
        Float v = -q / 2.0f - sqrt_disc;
        u = luisa::sign(u) * luisa::pow(luisa::abs(u), 1.0f / 3.0f);
        v = luisa::sign(v) * luisa::pow(luisa::abs(v), 1.0f / 3.0f);
        lambda[0] = u + v - a / 3.0f;
        lambda[1] = lambda[0];
        lambda[2] = lambda[0];
    }
    else if(discriminant < 0.0f)
    {
        // Three distinct real roots
        Float sqrt_neg_p = luisa::sqrt(-p);
        Float cos_theta = -q / (2.0f * sqrt_neg_p * sqrt_neg_p * sqrt_neg_p);
        cos_theta = luisa::clamp(cos_theta, -1.0f, 1.0f);
        Float theta = luisa::acos(cos_theta) / 3.0f;
        Float two_sqrt_neg_p = 2.0f * sqrt_neg_p;
        lambda[0] = two_sqrt_neg_p * luisa::cos(theta) - a / 3.0f;
        lambda[1] = two_sqrt_neg_p * luisa::cos(theta + 2.0f * 3.14159265358979f / 3.0f) - a / 3.0f;
        lambda[2] = two_sqrt_neg_p * luisa::cos(theta + 4.0f * 3.14159265358979f / 3.0f) - a / 3.0f;
    }
    else
    {
        // Multiple roots
        Float u = -q / 2.0f;
        u = luisa::sign(u) * luisa::pow(luisa::abs(u), 1.0f / 3.0f);
        lambda[0] = 2.0f * u - a / 3.0f;
        lambda[1] = -u - a / 3.0f;
        lambda[2] = lambda[1];
    }

    // Sort eigenvalues in descending order and compute singular values
    // Simple bubble sort for 3 elements
    for(int i = 0; i < 2; i++)
        for(int j = i + 1; j < 3; j++)
            if(lambda[i] < lambda[j])
            {
                Float tmp = lambda[i];
                lambda[i] = lambda[j];
                lambda[j] = tmp;
            }

    Sigma[0] = luisa::sqrt(lambda[0]);
    Sigma[1] = luisa::sqrt(lambda[1]);
    Sigma[2] = luisa::sqrt(lambda[2]);

    // Compute eigenvectors (columns of V)
    // For each eigenvalue, solve (A - lambda*I)v = 0
    Matrix3x3 V_tmp = make_identity3();
    
    for(int k = 0; k < 3; k++)
    {
        // Build A - lambda*I
        Matrix3x3 M = A;
        M[0][0] -= lambda[k];
        M[1][1] -= lambda[k];
        M[2][2] -= lambda[k];

        // Cross product of two rows gives eigenvector direction
        Vector3 v;
        v[0] = M[1][0] * M[2][1] - M[1][1] * M[2][0];
        v[1] = M[2][0] * M[0][1] - M[2][1] * M[0][0];
        v[2] = M[0][0] * M[1][1] - M[0][1] * M[1][0];

        Float norm = luisa::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if(norm > 1e-10f)
        {
            v[0] /= norm;
            v[1] /= norm;
            v[2] /= norm;
        }
        else
        {
            // Degenerate case, use standard basis
            v[0] = (k == 0) ? 1.0f : 0.0f;
            v[1] = (k == 1) ? 1.0f : 0.0f;
            v[2] = (k == 2) ? 1.0f : 0.0f;
        }

        V_tmp[0][k] = v[0];
        V_tmp[1][k] = v[1];
        V_tmp[2][k] = v[2];
    }

    // Ensure V is a rotation (det(V) = 1)
    Float detV = determinant(V_tmp);
    if(detV < 0.0f)
    {
        V_tmp[0][2] = -V_tmp[0][2];
        V_tmp[1][2] = -V_tmp[1][2];
        V_tmp[2][2] = -V_tmp[2][2];
        Sigma[2] = -Sigma[2]; // Track reflection in sigma
    }

    V = V_tmp;

    // Compute U = F * V * Sigma^{-1}
    Matrix3x3 FV = mul(F, V);
    Matrix3x3 U_tmp;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            U_tmp[i][j] = (Sigma[j] > 1e-10f) ? FV[i][j] / Sigma[j] : V[i][j];

    // Ensure U is a rotation
    Float detU = determinant(U_tmp);
    if(detU < 0.0f)
    {
        U_tmp[0][2] = -U_tmp[0][2];
        U_tmp[1][2] = -U_tmp[1][2];
        U_tmp[2][2] = -U_tmp[2][2];
    }

    U = U_tmp;
}

LUISA_DEVICE void polar_decomposition(const Matrix3x3& F, Matrix3x3& R, Matrix3x3& S) noexcept
{
    Matrix3x3 U, V;
    Vector3 Sigma;
    svd(F, U, Sigma, V);

    // R = U * V^T
    R = mul(U, transpose(V));

    // S = V * Sigma * V^T
    Matrix3x3 VSigma;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            VSigma[i][j] = V[i][j] * Sigma[j];
    S = mul(VSigma, transpose(V));
}

// EVD for 3x3 using analytical method (similar to SVD eigenvalue computation)
LUISA_DEVICE void evd(const Matrix3x3& A, Vector3& eigen_values, Matrix3x3& eigen_vectors) noexcept
{
    // Characteristic polynomial coefficients
    Float a = -(A[0][0] + A[1][1] + A[2][2]);
    Float b = A[0][0]*A[1][1] + A[1][1]*A[2][2] + A[2][2]*A[0][0] 
            - A[0][1]*A[1][0] - A[1][2]*A[2][1] - A[2][0]*A[0][2];
    Float c = -(A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
              - A[0][2]*A[1][1]*A[2][0] - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2]);

    // Solve cubic equation
    Float p = b - a*a / 3.0f;
    Float q = 2.0f*a*a*a / 27.0f - a*b / 3.0f + c;
    Float discriminant = q*q / 4.0f + p*p*p / 27.0f;

    Float lambda[3];
    if(discriminant > 0.0f)
    {
        Float sqrt_disc = luisa::sqrt(discriminant);
        Float u = -q / 2.0f + sqrt_disc;
        Float v = -q / 2.0f - sqrt_disc;
        u = luisa::sign(u) * luisa::pow(luisa::abs(u), 1.0f / 3.0f);
        v = luisa::sign(v) * luisa::pow(luisa::abs(v), 1.0f / 3.0f);
        lambda[0] = u + v - a / 3.0f;
        lambda[1] = lambda[0];
        lambda[2] = lambda[0];
    }
    else if(discriminant < 0.0f)
    {
        Float sqrt_neg_p = luisa::sqrt(-p);
        Float cos_theta = -q / (2.0f * sqrt_neg_p * sqrt_neg_p * sqrt_neg_p);
        cos_theta = luisa::clamp(cos_theta, -1.0f, 1.0f);
        Float theta = luisa::acos(cos_theta) / 3.0f;
        Float two_sqrt_neg_p = 2.0f * sqrt_neg_p;
        lambda[0] = two_sqrt_neg_p * luisa::cos(theta) - a / 3.0f;
        lambda[1] = two_sqrt_neg_p * luisa::cos(theta + 2.0f * 3.14159265358979f / 3.0f) - a / 3.0f;
        lambda[2] = two_sqrt_neg_p * luisa::cos(theta + 4.0f * 3.14159265358979f / 3.0f) - a / 3.0f;
    }
    else
    {
        Float u = -q / 2.0f;
        u = luisa::sign(u) * luisa::pow(luisa::abs(u), 1.0f / 3.0f);
        lambda[0] = 2.0f * u - a / 3.0f;
        lambda[1] = -u - a / 3.0f;
        lambda[2] = lambda[1];
    }

    // Sort eigenvalues
    for(int i = 0; i < 2; i++)
        for(int j = i + 1; j < 3; j++)
            if(lambda[i] < lambda[j])
            {
                Float tmp = lambda[i];
                lambda[i] = lambda[j];
                lambda[j] = tmp;
            }

    eigen_values[0] = lambda[0];
    eigen_values[1] = lambda[1];
    eigen_values[2] = lambda[2];

    // Compute eigenvectors
    eigen_vectors = make_identity3();
    
    for(int k = 0; k < 3; k++)
    {
        Matrix3x3 M = A;
        M[0][0] -= lambda[k];
        M[1][1] -= lambda[k];
        M[2][2] -= lambda[k];

        Vector3 v;
        v[0] = M[1][0] * M[2][1] - M[1][1] * M[2][0];
        v[1] = M[2][0] * M[0][1] - M[2][1] * M[0][0];
        v[2] = M[0][0] * M[1][1] - M[0][1] * M[1][0];

        Float norm = luisa::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if(norm > 1e-10f)
        {
            v[0] /= norm;
            v[1] /= norm;
            v[2] /= norm;
        }
        else
        {
            v[0] = (k == 0) ? 1.0f : 0.0f;
            v[1] = (k == 1) ? 1.0f : 0.0f;
            v[2] = (k == 2) ? 1.0f : 0.0f;
        }

        eigen_vectors[0][k] = v[0];
        eigen_vectors[1][k] = v[1];
        eigen_vectors[2][k] = v[2];
    }
}

// Helper: 9x9 matrix multiplication
LUISA_DEVICE inline Matrix9x9 mul9(const Matrix9x9& A, const Matrix9x9& B) noexcept
{
    Matrix9x9 C = {};
    for(int i = 0; i < 9; i++)
        for(int j = 0; j < 9; j++)
            for(int k = 0; k < 9; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Helper: 12x12 matrix multiplication
LUISA_DEVICE inline Matrix12x12 mul12(const Matrix12x12& A, const Matrix12x12& B) noexcept
{
    Matrix12x12 C = {};
    for(int i = 0; i < 12; i++)
        for(int j = 0; j < 12; j++)
            for(int k = 0; k < 12; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Helper: Transpose for 9x9
LUISA_DEVICE inline Matrix9x9 transpose9(const Matrix9x9& A) noexcept
{
    Matrix9x9 T;
    for(int i = 0; i < 9; i++)
        for(int j = 0; j < 9; j++)
            T[i][j] = A[j][i];
    return T;
}

// Helper: Transpose for 12x12
LUISA_DEVICE inline Matrix12x12 transpose12(const Matrix12x12& A) noexcept
{
    Matrix12x12 T;
    for(int i = 0; i < 12; i++)
        for(int j = 0; j < 12; j++)
            T[i][j] = A[j][i];
    return T;
}

// Jacobi eigenvalue decomposition for symmetric matrices
// Uses cyclic Jacobi method with threshold
LUISA_DEVICE void evd(const Matrix9x9& A, Vector9& eigen_values, Matrix9x9& eigen_vectors) noexcept
{
    // Initialize eigenvectors to identity
    for(int i = 0; i < 9; i++)
        for(int j = 0; j < 9; j++)
            eigen_vectors[i][j] = (i == j) ? 1.0f : 0.0f;

    // Copy A to working matrix
    Matrix9x9 M = A;

    const int max_sweeps = 50;
    const Float tolerance = 1e-10f;

    for(int sweep = 0; sweep < max_sweeps; sweep++)
    {
        Float max_off_diag = 0.0f;
        
        for(int p = 0; p < 9; p++)
        {
            for(int q = p + 1; q < 9; q++)
            {
                Float off_diag = luisa::abs(M[p][q]);
                if(off_diag > max_off_diag)
                    max_off_diag = off_diag;

                if(off_diag > tolerance)
                {
                    // Compute Jacobi rotation to zero out M[p][q]
                    Float tau = (M[q][q] - M[p][p]) / (2.0f * M[p][q]);
                    Float t = luisa::sign(tau) / (luisa::abs(tau) + luisa::sqrt(1.0f + tau * tau));
                    Float c = 1.0f / luisa::sqrt(1.0f + t * t);
                    Float s = t * c;

                    // Update M
                    Float m_pp = M[p][p];
                    Float m_qq = M[q][q];
                    M[p][p] = c * c * m_pp - 2.0f * c * s * M[p][q] + s * s * m_qq;
                    M[q][q] = s * s * m_pp + 2.0f * c * s * M[p][q] + c * c * m_qq;
                    M[p][q] = M[q][p] = 0.0f;

                    for(int k = 0; k < 9; k++)
                    {
                        if(k != p && k != q)
                        {
                            Float m_pk = M[p][k];
                            Float m_qk = M[q][k];
                            M[p][k] = M[k][p] = c * m_pk - s * m_qk;
                            M[q][k] = M[k][q] = s * m_pk + c * m_qk;
                        }
                    }

                    // Update eigenvectors
                    for(int k = 0; k < 9; k++)
                    {
                        Float e_kp = eigen_vectors[k][p];
                        Float e_kq = eigen_vectors[k][q];
                        eigen_vectors[k][p] = c * e_kp - s * e_kq;
                        eigen_vectors[k][q] = s * e_kp + c * e_kq;
                    }
                }
            }
        }

        if(max_off_diag < tolerance)
            break;
    }

    // Extract eigenvalues from diagonal
    for(int i = 0; i < 9; i++)
        eigen_values[i] = M[i][i];
}

LUISA_DEVICE void evd(const Matrix12x12& A, Vector12& eigen_values, Matrix12x12& eigen_vectors) noexcept
{
    // Initialize eigenvectors to identity
    for(int i = 0; i < 12; i++)
        for(int j = 0; j < 12; j++)
            eigen_vectors[i][j] = (i == j) ? 1.0f : 0.0f;

    // Copy A to working matrix
    Matrix12x12 M = A;

    const int max_sweeps = 60;
    const Float tolerance = 1e-10f;

    for(int sweep = 0; sweep < max_sweeps; sweep++)
    {
        Float max_off_diag = 0.0f;
        
        for(int p = 0; p < 12; p++)
        {
            for(int q = p + 1; q < 12; q++)
            {
                Float off_diag = luisa::abs(M[p][q]);
                if(off_diag > max_off_diag)
                    max_off_diag = off_diag;

                if(off_diag > tolerance)
                {
                    // Compute Jacobi rotation
                    Float tau = (M[q][q] - M[p][p]) / (2.0f * M[p][q]);
                    Float t = luisa::sign(tau) / (luisa::abs(tau) + luisa::sqrt(1.0f + tau * tau));
                    Float c = 1.0f / luisa::sqrt(1.0f + t * t);
                    Float s = t * c;

                    // Update M
                    Float m_pp = M[p][p];
                    Float m_qq = M[q][q];
                    M[p][p] = c * c * m_pp - 2.0f * c * s * M[p][q] + s * s * m_qq;
                    M[q][q] = s * s * m_pp + 2.0f * c * s * M[p][q] + c * c * m_qq;
                    M[p][q] = M[q][p] = 0.0f;

                    for(int k = 0; k < 12; k++)
                    {
                        if(k != p && k != q)
                        {
                            Float m_pk = M[p][k];
                            Float m_qk = M[q][k];
                            M[p][k] = M[k][p] = c * m_pk - s * m_qk;
                            M[q][k] = M[k][q] = s * m_pk + c * m_qk;
                        }
                    }

                    // Update eigenvectors
                    for(int k = 0; k < 12; k++)
                    {
                        Float e_kp = eigen_vectors[k][p];
                        Float e_kq = eigen_vectors[k][q];
                        eigen_vectors[k][p] = c * e_kp - s * e_kq;
                        eigen_vectors[k][q] = s * e_kp + c * e_kq;
                    }
                }
            }
        }

        if(max_off_diag < tolerance)
            break;
    }

    // Extract eigenvalues from diagonal
    for(int i = 0; i < 12; i++)
        eigen_values[i] = M[i][i];
}

LUISA_DEVICE Matrix9x9 clamp_to_spd(const Matrix9x9& A) noexcept
{
    Matrix9x9 Q;
    Vector9 values;
    evd(A, values, Q);

    // Clamp negative eigenvalues to zero
    for(int i = 0; i < 9; i++)
        if(values[i] < 0.0f)
            values[i] = 0.0f;

    // Reconstruct: A = Q * diag(values) * Q^T
    Matrix9x9 result = {};
    for(int i = 0; i < 9; i++)
        for(int j = 0; j < 9; j++)
            for(int k = 0; k < 9; k++)
                result[i][j] += Q[i][k] * values[k] * Q[j][k];

    return result;
}

LUISA_DEVICE Matrix12x12 clamp_to_spd(const Matrix12x12& A) noexcept
{
    Matrix12x12 Q;
    Vector12 values;
    evd(A, values, Q);

    // Clamp negative eigenvalues to zero
    for(int i = 0; i < 12; i++)
        if(values[i] < 0.0f)
            values[i] = 0.0f;

    // Reconstruct: A = Q * diag(values) * Q^T
    Matrix12x12 result = {};
    for(int i = 0; i < 12; i++)
        for(int j = 0; j < 12; j++)
            for(int k = 0; k < 12; k++)
                result[i][j] += Q[i][k] * values[k] * Q[j][k];

    return result;
}
}  // namespace uipc::backend::luisa
