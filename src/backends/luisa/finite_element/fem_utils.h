#pragma once
/********************************************************************
 * @file fem_utils.h
 * @brief T5.1 — FEM material-space helpers (port of
 * src/backends/cuda/finite_element/fem_utils.h + .cu).
 *
 * Same generic-over-T convention as matrix_utils.h (the reference's
 * `UIPC_GENERIC`): one implementation serves the host (`T = double`, the
 * reference's Float) and the device (`T = lcrt::Var<double>`), and the call
 * sites keep the reference's spelling (`fem::Dm_inv(x0, x1, x2, x3)`), because
 * T is deduced.
 *
 * Scalar arithmetic goes through the `math::detail` trampolines (cookbook RULE
 * M1); LVec/LMat arithmetic uses the numerics layer's operators, which ADL finds
 * for those argument types.
 *********************************************************************/
#include <finite_element/matrix_utils.h>

namespace uipc::backend::luisa::fem
{
namespace detail
{
using lps::detail::add;
using lps::detail::div;
using lps::detail::mul;
using lps::detail::neg;
using lps::detail::sub;
using lps::stored;
}  // namespace detail

/// `||F||_F^2` (the reference's `invariant2`).
template <typename T>
[[nodiscard]] inline auto invariant2(const lps::LMat<T, 3, 3>& F) noexcept
{
    return ddot(F, F);
}

/// `sigma_0^2 + sigma_1^2 + sigma_2^2`.
template <typename T>
[[nodiscard]] inline auto invariant2(const lps::LVec<T, 3>& Sigma) noexcept
{
    return detail::add(detail::add(detail::mul(Sigma[0], Sigma[0]),
                                   detail::mul(Sigma[1], Sigma[1])),
                       detail::mul(Sigma[2], Sigma[2]));
}

/// `det(F)` (the reference's `invariant3`).
template <typename T>
[[nodiscard]] inline auto invariant3(const lps::LMat<T, 3, 3>& F) noexcept
{
    return lps::determinant(F);
}

/// `sigma_0 * sigma_1 * sigma_2`.
template <typename T>
[[nodiscard]] inline auto invariant3(const lps::LVec<T, 3>& Sigma) noexcept
{
    return detail::mul(detail::mul(Sigma[0], Sigma[1]), Sigma[2]);
}

/// `a^T S a` with `S = V diag(Sigma) V^T` (the reference's `invariant4`).
/// `a` is generic vec-like (a value or a column view), like in the reference.
template <typename T, typename V>
    requires(lps::is_vec_like_v<V> && lps::vec_dim_v<V> == 3)
[[nodiscard]] inline auto invariant4(const lps::LMat<T, 3, 3>& F, const V& a) noexcept
{
    lps::LMat<T, 3, 3> U, Vm;
    lps::LVec<T, 3>    Sigma;
    svd(F, U, Sigma, Vm);
    const auto S = Vm * Sigma.asDiagonal() * Vm.transpose();
    return detail::stored(lps::dot(S * a, a));
}

/// `||F a||^2` (the reference's `invariant5`).
template <typename T, typename V>
    requires(lps::is_vec_like_v<V> && lps::vec_dim_v<V> == 3)
[[nodiscard]] inline auto invariant5(const lps::LMat<T, 3, 3>& F, const V& a) noexcept
{
    const auto Fa = F * a;
    return detail::stored(lps::dot(Fa, Fa));
}

/**
 * @brief `d det(F) / dF`, i.e. the matrix of cofactors:
 * `[f1 x f2 | f2 x f0 | f0 x f1]` (`fi` = column i).
 */
template <typename T>
[[nodiscard]] inline lps::LMat<T, 3, 3> dJdF(const lps::LMat<T, 3, 3>& F) noexcept
{
    lps::LMat<T, 3, 3> result;
    // The reference uses Eigen's member `.cross()`; the numerics layer spells it
    // as the free function `lps::cross(a, b)` (cookbook §7.5).
    result.col(0) = lps::cross(F.col(1), F.col(2));
    result.col(1) = lps::cross(F.col(2), F.col(0));
    result.col(2) = lps::cross(F.col(0), F.col(1));
    return result;
}

/// Rest-shape matrix `Ds = [x1 - x0 | x2 - x0 | x3 - x0]` (a matrix is returned
/// by value: `lps::inverse` and `determinant` take it by value anyway).
template <typename T>
[[nodiscard]] inline lps::LMat<T, 3, 3> Ds(const lps::LVec<T, 3>& x0,
                                           const lps::LVec<T, 3>& x1,
                                           const lps::LVec<T, 3>& x2,
                                           const lps::LVec<T, 3>& x3) noexcept
{
    lps::LMat<T, 3, 3> result;
    result.col(0) = x1 - x0;
    result.col(1) = x2 - x0;
    result.col(2) = x3 - x0;
    return result;
}

/// Inverse material coordinates `Dm^-1`.
template <typename T>
[[nodiscard]] inline lps::LMat<T, 3, 3> Dm_inv(const lps::LVec<T, 3>& X0,
                                               const lps::LVec<T, 3>& X1,
                                               const lps::LVec<T, 3>& X2,
                                               const lps::LVec<T, 3>& X3) noexcept
{
    return lps::inverse(Ds(X0, X1, X2, X3));
}

/// `dF / dx` (9x12), the reference's generated table (transcribed entry for
/// entry, see the .cu's `//tex:` documentation block).
template <typename T>
[[nodiscard]] inline lps::LMat<T, 9, 12> dFdx(const lps::LMat<T, 3, 3>& DmInv) noexcept
{
    const T m = DmInv(0, 0);
    const T n = DmInv(0, 1);
    const T o = DmInv(0, 2);
    const T p = DmInv(1, 0);
    const T q = DmInv(1, 1);
    const T r = DmInv(1, 2);
    const T s = DmInv(2, 0);
    const T t = DmInv(2, 1);
    const T u = DmInv(2, 2);

    const T t1 = detail::sub(detail::sub(detail::neg(m), p), s);
    const T t2 = detail::sub(detail::sub(detail::neg(n), q), t);
    const T t3 = detail::sub(detail::sub(detail::neg(o), r), u);

    lps::LMat<T, 9, 12> PFPu;  // default construction is zero
    PFPu(0, 0) = t1;
    PFPu(0, 3) = m;
    PFPu(0, 6) = p;
    PFPu(0, 9) = s;
    PFPu(1, 1) = t1;
    PFPu(1, 4) = m;
    PFPu(1, 7) = p;
    PFPu(1, 10) = s;
    PFPu(2, 2) = t1;
    PFPu(2, 5) = m;
    PFPu(2, 8) = p;
    PFPu(2, 11) = s;
    PFPu(3, 0) = t2;
    PFPu(3, 3) = n;
    PFPu(3, 6) = q;
    PFPu(3, 9) = t;
    PFPu(4, 1) = t2;
    PFPu(4, 4) = n;
    PFPu(4, 7) = q;
    PFPu(4, 10) = t;
    PFPu(5, 2) = t2;
    PFPu(5, 5) = n;
    PFPu(5, 8) = q;
    PFPu(5, 11) = t;
    PFPu(6, 0) = t3;
    PFPu(6, 3) = o;
    PFPu(6, 6) = r;
    PFPu(6, 9) = u;
    PFPu(7, 1) = t3;
    PFPu(7, 4) = o;
    PFPu(7, 7) = r;
    PFPu(7, 10) = u;
    PFPu(8, 2) = t3;
    PFPu(8, 5) = o;
    PFPu(8, 8) = r;
    PFPu(8, 11) = u;

    return PFPu;
}

/// The four tetrahedron shape-function gradients as a 3x4 matrix (column i is
/// the gradient of vertex i).
template <typename T>
[[nodiscard]] inline lps::LMat<T, 3, 4> tetrahedron_shape_gradients(
    const lps::LMat<T, 3, 3>& DmInv) noexcept
{
    lps::LMat<T, 3, 4> gradients;
    for(int i = 0; i < 3; ++i)
    {
        gradients(i, 1) = DmInv(0, i);
        gradients(i, 2) = DmInv(1, i);
        gradients(i, 3) = DmInv(2, i);
        gradients(i, 0) = detail::sub(
            detail::sub(detail::neg(gradients(i, 1)), gradients(i, 2)),
            gradients(i, 3));
    }
    return gradients;
}

/// `dE/dx_i = sum_j shape_gradient(j) * dEdF(3j + i)`.
///
/// `shape_gradient` is taken as a generic vec-like argument (the numerics layer's
/// `is_vec_like_v`), so a caller may pass either an `LVec` value or a column view
/// (`shape_gradients.col(i)`), exactly like the reference does.
template <typename T, typename SG>
    requires(lps::is_vec_like_v<SG> && lps::vec_dim_v<SG> == 3)
[[nodiscard]] inline lps::LVec<T, 3> project_F_gradient(const SG&              shape_gradient,
                                                        const lps::LVec<T, 9>& dEdF) noexcept
{
    lps::LVec<T, 3> result;  // zero
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result(i) = detail::add(
                result(i), detail::mul(shape_gradient(j), dEdF(3 * j + i)));
    return result;
}

/// The (i, j) 3x3 block of `d^2E/dx_i dx_j` for the same `F(x)` map.
template <typename T, typename SL, typename SR>
    requires(lps::is_vec_like_v<SL> && lps::vec_dim_v<SL> == 3
             && lps::is_vec_like_v<SR> && lps::vec_dim_v<SR> == 3)
[[nodiscard]] inline lps::LMat<T, 3, 3> project_F_hessian_block(
    const SL&                 left_shape_gradient,
    const lps::LMat<T, 9, 9>& ddEddF,
    const SR&                 right_shape_gradient) noexcept
{
    lps::LMat<T, 3, 3> result;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
        {
            T value = lps::zero_of<T>();
            for(int k = 0; k < 3; ++k)
                for(int l = 0; l < 3; ++l)
                    value = detail::add(
                        value,
                        detail::mul(detail::mul(left_shape_gradient(k),
                                                 ddEddF(3 * k + i, 3 * l + j)),
                                    right_shape_gradient(l)));
            result(i, j) = value;
        }
    return result;
}

/// Deformation gradient `F = Ds(x) * Dm^-1`.
template <typename T>
[[nodiscard]] inline lps::LMat<T, 3, 3> F(const lps::LVec<T, 3>& x0,
                                          const lps::LVec<T, 3>& x1,
                                          const lps::LVec<T, 3>& x2,
                                          const lps::LVec<T, 3>& x3,
                                          const lps::LMat<T, 3, 3>& DmInv) noexcept
{
    auto ds = Ds(x0, x1, x2, x3);
    return ds * DmInv;
}
}  // namespace uipc::backend::luisa::fem
