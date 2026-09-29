#pragma once
/********************************************************************
 * @file affine_body/utils.h
 * @brief T6.1a — ABD dof/affine-matrix conversions (port of
 * src/backends/cuda/affine_body/utils.{h,cu}).
 *
 * The reference's utils are `UIPC_GENERIC` and Eigen-typed.  The port follows the
 * T5.1 convention (cookbook §7.16 rule 1): the *numeric* routines are header-only
 * templates with a deduced scalar (`S` = `double` on the host, `lcrt::Var<double>`
 * in a kernel) whose operands are the numerics layer's containers, so the same
 * body serves the host oracle, the device kernel and the unit test.  The
 * deviation from the reference's `.h`+`.cu` split is reported in the T6.1a row.
 *
 * What stays Eigen-typed (and is therefore defined in utils.cpp, the reference's
 * ".cu host half"): the two *host-only* transform conversions the dynamics'
 * geometry build-up and the state accessor use --
 *
 *   reference                              port
 *   -------------------------------------  --------------------------------------
 *   q_to_transform(Vector12) -> Matrix4x4   q_to_transform(const Vector12&)      [host]
 *   transform_to_q(Matrix4x4) -> Vector12   transform_to_q(const Matrix4x4&)     [host]
 *   q_v_to_transform_v / transform_v_to_q_v the same pair, last row zeroed       [host]
 *
 * They are `q_to_transform`/`transform_to_q` overloaded on the argument type
 * (Eigen vs `lps::LVec<S,12>`/`lps::LMat<S,4,4>`), which is unambiguous: the two
 * families are unrelated types.
 *
 * `torque_to_F`'s device-side guard: the reference warns + returns zero for a
 * near-singular affine matrix.  A device predicate cannot drive control flow in
 * the DSL (RULE M2), so the port computes the result unconditionally and *selects*
 * zero when the guard fails, with the reference's warning routed through the
 * kernel-logging facade (LC_LOG, compiled out unless UIPC_LUISA_KERNEL_PRINT --
 * cookbook §7.15).  On the host the guard is a host `bool` and the message goes to
 * the logger, exactly like the reference's UIPC_KERNEL_WARN.
 *********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/matrix_ops.h>  // lps::inverse (T5.4b trap: not in mat.h)
#include <utils/scalar_util.h>
#include <uipc/common/config.h>
#include <uipc/common/log.h>

namespace uipc::backend::luisa
{
// ---------------------------------------------------------------------------
//  DoF / affine-matrix conversions (reference: utils.cu, UIPC_GENERIC)
// ---------------------------------------------------------------------------
///tex: $$ A = \begin{bmatrix} \mathbf{a}_1^T\\ \mathbf{a}_2^T\\ \mathbf{a}_3^T\end{bmatrix} $$
/// (reference: `q_to_A`; rows 0/1/2 are the q segments 3/6/9)
template <class S>
[[nodiscard]] inline lps::LMat<S, 3, 3> q_to_A(const lps::LVec<S, 12>& q) noexcept
{
    lps::LMat<S, 3, 3> A;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            A(i, j) = q(3 + 3 * i + j);
    return A;
}

/// (reference: `A_to_q`; the translational block stays zero)
template <class S>
[[nodiscard]] inline lps::LVec<S, 9> A_to_q(const lps::LMat<S, 3, 3>& A) noexcept
{
    lps::LVec<S, 9> q = lps::LVec<S, 9>::Zero();
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            q(3 * i + j) = A(i, j);
    return q;
}

/// (reference: `F_to_A`) — the row-major <-> column-major flattening permutation
/// (0,3,6,1,4,7,2,5,8); the reference spells it as nine explicit assignments.
template <class S>
[[nodiscard]] inline lps::LVec<S, 9> F_to_A(const lps::LVec<S, 9>& F) noexcept
{
    constexpr int perm[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
    lps::LVec<S, 9> A;
    for(int i = 0; i < 9; ++i)
        A(i) = F(perm[i]);
    return A;
}

/// (reference: `HF_to_HA`) — the same permutation applied to *both* indices, i.e.
/// `HA = P HF P^T` with `P = (e0, e3, e6, e1, e4, e7, e2, e5, e8)`; the reference
/// spells it as 81 explicit assignments.  The test asserts the identity against
/// the permutation matrix, which is an independent check of the mapping.
template <class S>
[[nodiscard]] inline lps::LMat<S, 9, 9> HF_to_HA(const lps::LMat<S, 9, 9>& HF) noexcept
{
    constexpr int perm[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
    lps::LMat<S, 9, 9> HA;
    for(int i = 0; i < 9; ++i)
        for(int j = 0; j < 9; ++j)
            HA(i, j) = HF(perm[i], perm[j]);
    return HA;
}

/// Skew-symmetric matrix `[v]_x` such that `[v]_x w = v x w`.
template <class S>
[[nodiscard]] inline lps::LMat<S, 3, 3> skew(const lps::LVec<S, 3>& v) noexcept
{
    // reference: `S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;`
    // NOTE: the local is *not* called `S` (the reference's spelling): a local `S`
    // would shadow the scalar template parameter, so `lps::zero_of<S>()` would
    // receive a *variable* (measured C2672 "invalid template argument for 'T',
    // expected a type" at the first device instantiation).
    lps::LMat<S, 3, 3> ret;
    ret(0, 0) = lps::zero_of<S>();
    ret(0, 1) = -v(2);
    ret(0, 2) = v(1);
    ret(1, 0) = v(2);
    ret(1, 1) = lps::zero_of<S>();
    ret(1, 2) = -v(0);
    ret(2, 0) = -v(1);
    ret(2, 1) = v(0);
    ret(2, 2) = lps::zero_of<S>();
    return ret;
}

/// Orthonormal basis around `t` (`t` is normalised in place, like the reference).
template <class S>
inline void orthonormal_basis(lps::LVec<S, 3>& t, lps::LVec<S, 3>& n, lps::LVec<S, 3>& b) noexcept
{
    t = t.normalized();
    // reference: if(|t . ex| > 0.9) test_vector = ey; else ex;  -> a select
    const auto ex = lps::LVec<S, 3>{
        lps::one_of<S>(), lps::zero_of<S>(), lps::zero_of<S>()};
    const auto ey = lps::LVec<S, 3>{
        lps::zero_of<S>(), lps::one_of<S>(), lps::zero_of<S>()};
    const auto test_vector = scalar_detail::select_container(
        ex, ey, lps::abs(lps::dot(t, ex)) > lps::make_scalar<S>(0.9));
    n = lps::cross(t, test_vector).normalized();
    b = lps::cross(t, n).normalized();
}

/// Scalar torque `tau` about the world-space unit axis `e` -> ABD generalized
/// force (reference: `torque_to_F`).
///tex: $$ \mathbf{F}^A = \frac{\tau}{2} [\mathbf{e}]_\times \mathbf{A}^{-T} $$
template <class S>
[[nodiscard]] inline lps::LVec<S, 12> torque_to_F(const S&                 tau,
                                                  const lps::LVec<S, 3>&   e,
                                                  const lps::LVec<S, 12>&  q) noexcept
{
    auto A = q_to_A(q);
    // The reference guards with the scale-invariant measure
    // |det| / (|a0||a1||a2|) in [0,1] (also NaN-catching) and drops the torque
    // when the matrix collapses.  A device predicate cannot drive control flow
    // (RULE M2), so the whole computation runs and the *zero* branch is selected.
    // NOTE: `norm` is a *free* function in this layer for views (a MatMap row has
    // no member norm(), cookbook §7.17 rule 15's cross/dot note).
    auto row_norm_prod = lps::norm(A.row(0)) * lps::norm(A.row(1)) * lps::norm(A.row(2));
    auto det           = A.determinant();
    auto ok = scalar_detail::pred_and(row_norm_prod > lps::zero_of<S>(),
                                      lps::abs(det) >= lps::make_scalar<S>(1e-6) * row_norm_prod);

    if constexpr(lps::is_dsl_value_v<S>)
    {
        LC_LOG("torque_to_F: affine matrix is near-singular; dropping joint torque");
    }
    else
    {
        if(!static_cast<bool>(ok))
        {
            logger::warn(
                "torque_to_F: affine matrix is near-singular (det={}, row_norm_prod={}); "
                "dropping joint torque",
                det,
                row_norm_prod);
        }
    }

    lps::LMat<S, 3, 3> A_inv   = A.inverse();
    lps::LMat<S, 3, 3> A_inv_T = A_inv.transpose();
    lps::LMat<S, 3, 3> FA      = (lps::make_scalar<S>(0.5) * tau) * skew(e) * A_inv_T;
    lps::LVec<S, 12>   F       = lps::LVec<S, 12>::Zero();
    F.segment<9>(3)            = A_to_q(FA);
    return scalar_detail::select_container(lps::LVec<S, 12>::Zero(), F, ok);
}

// ---------------------------------------------------------------------------
//  Host-only (Eigen) transform conversions -- defined in utils.cpp
// ---------------------------------------------------------------------------
///tex: $$ \begin{bmatrix} A & \mathbf{t}\\ 0 & 1 \end{bmatrix} $$
[[nodiscard]] Matrix4x4 q_to_transform(const Vector12& q) noexcept;

/// The inverse of `q_to_transform` (the reference's own spelling).
[[nodiscard]] Vector12 transform_to_q(const Matrix4x4& transform) noexcept;

///tex: the same, with a zero last row (a *velocity* transform).
[[nodiscard]] Matrix4x4 q_v_to_transform_v(const Vector12& q) noexcept;

/// The reference's `transform_v_to_q_v` is literally `transform_to_q`.
[[nodiscard]] Vector12 transform_v_to_q_v(const Matrix4x4& transform_v) noexcept;
}  // namespace uipc::backend::luisa
