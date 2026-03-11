#pragma once
#include <type_define.h>

// ref: https://github.com/theodorekim/HOBAKv1/blob/main/src/util/MATRIX_UTIL.h

namespace uipc::backend::luisa
{
// Large vector and matrix types using std::array for fixed-size storage
// (LuisaCompute only supports up to 4x4 matrices/vectors natively)
using Vector9 = std::array<float, 9>;
using Vector12 = std::array<float, 12>;

// 9x9 and 12x12 matrices using std::array since Luisa doesn't support these sizes
using Matrix9x9 = std::array<std::array<float, 9>, 9>;
using Matrix12x12 = std::array<std::array<float, 12>, 12>;
using Matrix9x12 = std::array<std::array<float, 12>, 9>;

// flatten a matrix3x3 to a vector9 in a consistent way (column-major)
LUISA_DEVICE Vector9 flatten(const Matrix3x3& A) noexcept;

// unflatten a vector9 to a matrix3x3 in a consistent way (column-major)
LUISA_DEVICE Matrix3x3 unflatten(const Vector9& v) noexcept;

// double dot product: A:B = sum(A(i,j) * B(i,j))
LUISA_DEVICE Float ddot(const Matrix3x3& A, const Matrix3x3& B);

// compute the singular value decomposition of a matrix3x3
// the U,V are already tested and modified to be rotation matrices
// Uses analytical 3x3 SVD algorithm based on McAdams et al. 2011
LUISA_DEVICE void svd(const Matrix3x3& F, Matrix3x3& U, Vector3& Sigma, Matrix3x3& V) noexcept;

// compute the polar decomposition of a matrix3x3
// F = R * S where R is rotation and S is symmetric
LUISA_DEVICE void polar_decomposition(const Matrix3x3& F, Matrix3x3& R, Matrix3x3& S) noexcept;

// Eigenvalue decomposition for 3x3 matrix (analytical method)
LUISA_DEVICE void evd(const Matrix3x3& A, Vector3& eigen_values, Matrix3x3& eigen_vectors) noexcept;

// Eigenvalue decomposition for 9x9 matrix (Jacobi method)
LUISA_DEVICE void evd(const Matrix9x9& A, Vector9& eigen_values, Matrix9x9& eigen_vectors) noexcept;

// Eigenvalue decomposition for 12x12 matrix (Jacobi method)
LUISA_DEVICE void evd(const Matrix12x12& A, Vector12& eigen_values, Matrix12x12& eigen_vectors) noexcept;

// clamp the eigenvalues of a matrix9x9 to be semi-positive-definite
LUISA_DEVICE Matrix9x9 clamp_to_spd(const Matrix9x9& A) noexcept;

// clamp the eigenvalues of a matrix12x12 to be semi-positive-definite
LUISA_DEVICE Matrix12x12 clamp_to_spd(const Matrix12x12& A) noexcept;
}  // namespace uipc::backend::luisa
