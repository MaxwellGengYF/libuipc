#pragma once
/********************************************************************
 * @file matrix_utils.h
 * @brief T5.1 — FEM matrix helpers (port of
 * src/backends/cuda/finite_element/matrix_utils.h + .cu).
 *
 * ref: https://github.com/theodorekim/HOBAKv1/blob/main/src/util/MATRIX_UTIL.h
 *
 * The reference is `UIPC_GENERIC` (nvcc device+host). The LuisaCompute port has
 * no separate device compilation -- a kernel body is ordinary host C++ that
 * builds IR -- so every routine here is written **once**, as a template over the
 * element type, exactly like the T2.x numerics layer (cookbook §7):
 *
 *   T = double            -> host evaluation (the reference's Float)
 *   T = lcrt::Var<double> -> device evaluation (a DSL local variable)
 *
 * Call sites keep the reference's spelling (`ddot(A, B)`, `svd(F, U, Sigma, V)`,
 * `flatten(F)`), because the template parameter is deduced from the arguments.
 * The definitions therefore live in this header: a DSL call needs the body at the
 * call site to emit IR, so the reference's .cu part merges into the header (see
 * the T5.1 report).
 *
 * `svd`, `polar_decomposition` and `evd` are *using-declarations* of the T2.2
 * routines in `luisa_device::math` rather than wrappers. That is deliberate: a
 * wrapper template named `svd` in `uipc::backend::luisa` would be found by
 * ordinary lookup from `uipc::backend::luisa::fem` *and* the T2.2 `math::svd`
 * would be added by ADL for LMat arguments -- two equally good candidates, i.e.
 * an ambiguity (measured). A using-declaration introduces the very same entity,
 * so the candidate set has one element and the reference's call-site spelling
 * keeps working unchanged.
 *********************************************************************/
#include <type_define.h>
#include <luisa_device/math/matrix_ops.h>
#include <luisa_device/math/svd.h>
#include <luisa_device/math/evd.h>

namespace uipc::backend::luisa
{
/// The reference's host-facing alias (used by T5.2's constitutions).
using Matrix9x12 = lps::LMat<Float, 9, 12>;

// ---------------------------------------------------------------------------
//  The reference's decomposition surface, mapped onto the T2.2 numerics layer
//  (a using-declaration, see the file comment).
// ---------------------------------------------------------------------------
using lps::evd;
using lps::polar_decomposition;
using lps::qr_svd;
using lps::svd;

namespace matrix_utils_detail
{
using lps::detail::add;
using lps::detail::mul;
}  // namespace matrix_utils_detail

/// Flatten a matrix3x3 to a vector9 in a consistent way (column-major, i.e.
/// Eigen's own coefficient order).
template <typename T>
[[nodiscard]] inline lps::LVec<T, 9> flatten(const lps::LMat<T, 3, 3>& A) noexcept
{
    lps::LVec<T, 9> column;
    unsigned int    index = 0;
    for(unsigned int j = 0; j < 3; j++)
        for(unsigned int i = 0; i < 3; i++, index++)
            column[index] = A(i, j);

    return column;
}

/// Unflatten a vector9 to a matrix3x3 in a consistent way.
template <typename T>
[[nodiscard]] inline lps::LMat<T, 3, 3> unflatten(const lps::LVec<T, 9>& v) noexcept
{
    lps::LMat<T, 3, 3> A;
    unsigned int       index = 0;
    for(unsigned int j = 0; j < 3; j++)
        for(unsigned int i = 0; i < 3; i++, index++)
            A(i, j) = v[index];

    return A;
}

/// Frobenius inner product `A : B` (the reference's summation order: the outer
/// loop over columns x, the inner over rows y).
template <typename T>
[[nodiscard]] inline auto ddot(const lps::LMat<T, 3, 3>& A,
                               const lps::LMat<T, 3, 3>& B) noexcept
{
    T result = lps::zero_of<T>();
    for(int y = 0; y < 3; y++)
        for(int x = 0; x < 3; x++)
            result = matrix_utils_detail::add(
                result, matrix_utils_detail::mul(A(x, y), B(x, y)));

    return result;
}

/// Clamp the eigenvalues of a matrix9x9 to be semi-positive-definite.
template <typename T>
[[nodiscard]] inline lps::LMat<T, 9, 9> clamp_to_spd(const lps::LMat<T, 9, 9>& A) noexcept
{
    lps::LMat<T, 9, 9> Q;
    lps::LVec<T, 9>    values;
    lps::evd(A, values, Q);
    for(int x = 0; x < 9; x++)
        // (values[x] > 0) ? values[x] : 0  -- a select, never a branch (RULE M2)
        values[x] = lps::max(values[x], lps::zero_of<T>());
    return Q * values.asDiagonal() * Q.transpose();
}

/// Clamp the eigenvalues of a matrix12x12 to be semi-positive-definite.
template <typename T>
[[nodiscard]] inline lps::LMat<T, 12, 12> clamp_to_spd(
    const lps::LMat<T, 12, 12>& A) noexcept
{
    lps::LMat<T, 12, 12> Q;
    lps::LVec<T, 12>     values;
    lps::evd(A, values, Q);
    for(int x = 0; x < 12; x++)
        values[x] = lps::max(values[x], lps::zero_of<T>());
    return Q * values.asDiagonal() * Q.transpose();
}
}  // namespace uipc::backend::luisa
