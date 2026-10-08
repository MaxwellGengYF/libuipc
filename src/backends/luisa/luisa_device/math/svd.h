#pragma once
/********************************************************************
 * @file svd.h
 * @brief T2.2 — 3x3 SVD and polar decomposition for the generic layer.
 *
 * ## Why this is a re-implementation rather than a transcription
 *
 * The reference implementation (`algorithm/qr_svd.hpp` + `algorithm/givens.hpp`,
 * used by `cuda_tool::eigen::svd`/`pd`) is a Givens bidiagonalisation followed by
 * an implicit-shift QR iteration. It is driven by *device* data in every control
 * decision:
 *
 *     while (|a0| > tol && |a1| > tol && ...) { wilkinson shift; chase; }
 *     if (|b1| <= tol) ... else if (|b0| <= tol) ... else if ... (5 terminations)
 *     if (y2 == 0) ... if (tau > 0) ... if (S0 < S1) ...
 *
 * On the host those conditions are plain booleans; inside an LC kernel they are
 * `Expr<bool>`, which cannot drive host control flow (cookbook RULE M2), so a
 * transcription would need `$if`/`$while` and would be device-only — losing the
 * host/device equivalence that the whole numerics layer is built on.
 *
 * Instead this uses **one-sided Jacobi (Hestenes)**, which for 3x3 is:
 *   - fixed-sweep (no data-dependent loop or termination),
 *   - branchless (all pivots/guards become `select`),
 *   - more accurate than QR on small singular values (relative accuracy on σ_min),
 *   - and therefore *one* implementation that is valid on the host and the device.
 *
 * The public surface (`svd`, `qr_svd`, `pd`, `polard`) and the *signed-SVD*
 * convention (proper rotations U, V with a possible reflection moved into σ2) are
 * those of the reference, so ported call sites keep working unchanged. For
 * det(F) > 0 the polar decomposition is unique, so `pd` returns exactly the same
 * (R, S) as the reference/Eigen up to numerical error.
 *********************************************************************/
#include <luisa_device/math/evd.h>
#include <luisa_device/math/matrix_ops.h>

namespace uipc::backend::luisa::luisa_device::math
{
/// One-sided sweeps; each sweep applies the 3 column-pair rotations once.
inline constexpr int kSvdSweeps = 16;

namespace detail
{
/// Swap entries i and j of `v` when `swap` holds (device-safe).
template <typename T, int N>
inline void swap_if(LVec<T, N>& v, int i, int j, const auto& swap) noexcept
{
    auto a  = v(i);
    auto b  = v(j);
    v(i)    = select(a, b, swap);
    v(j)    = select(b, a, swap);
}

/// Swap columns i and j of `m` when `swap` holds (device-safe).
template <typename T, int R, int C>
inline void swap_cols_if(LMat<T, R, C>& m, int i, int j, const auto& swap) noexcept
{
    for(int r = 0; r < R; ++r)
    {
        auto a   = m(r, i);
        auto b   = m(r, j);
        m(r, i)  = select(a, b, swap);
        m(r, j)  = select(b, a, swap);
    }
}

/// Odd-even transposition sort of `sigma` (descending) keeping `B` and `V` aligned.
template <typename T>
inline void sort_descending(LVec<T, 3>& sigma, LMat<T, 3, 3>& B, LMat<T, 3, 3>& V) noexcept
{
    for(int pass = 0; pass < 3; ++pass)
    {
        for(int i = 0; i + 1 < 3; ++i)
        {
            if((i + pass) % 2 != 0)
                continue;
            auto swap = sigma(i) < sigma(i + 1);  // ascending order violated
            swap_if(sigma, i, i + 1, swap);
            swap_cols_if(B, i, i + 1, swap);
            swap_cols_if(V, i, i + 1, swap);
        }
    }
}
}// namespace detail

/// Normalise `v`; when |v| is below `eps`, return `fallback` instead.
template <typename T>
[[nodiscard]] inline auto normalized_completion(const LVec<T, 3>& v,
                                                const T&         eps,
                                                const LVec<T, 3>& fallback) noexcept
{
    auto n = sqrt(squaredNorm(v));
    // select(f, t, p) == (p ? t : f): well-conditioned vectors are normalised,
    // degenerate ones fall back.
    return make_vec<storage_of_t<T>, 3>(
        [&](int i)
        { return select(fallback(i), detail::div(v(i), n), n > eps); });
}

/// Normalise `v`; when |v| is below `eps`, fall back to \f$e_0\f$.
template <typename T>
[[nodiscard]] inline auto normalized_completion(const LVec<T, 3>& v, const T& eps) noexcept
{
    return normalized_completion(v, eps, LVec<T, 3>::template Unit<0>());
}

/// A unit vector orthogonal to `u` (used to complete a rank-deficient basis).
template <typename T>
[[nodiscard]] inline auto axis_orthogonal_to(const LVec<T, 3>& u) noexcept
{
    using R = storage_of_t<T>;
    auto ax = abs(u(0));
    auto ay = abs(u(1));
    auto az = abs(u(2));

    // Pick the coordinate axis least aligned with `u`, then cross with it.
    // NOTE: LC has no short-circuit `&&`/`||` for *scalar* boolean expressions
    // (only for boolean vectors); scalars use the bitwise spellings `&`/`|`.
    auto use0 = (ax <= ay) & (ax <= az);
    auto use1 = (!use0) & (ay <= az);
    auto use2 = !(use0 | use1);

    // Built element-wise: there is deliberately no container `select` in this
    // layer -- LC's own `select` has a custom-struct overload that would match a
    // container too, making such a call ambiguous (and, worse, build a
    // `Var<LVec<...>>`).
    auto u0 = LVec<T, 3>::template Unit<0>();
    auto u1 = LVec<T, 3>::template Unit<1>();
    auto u2 = LVec<T, 3>::template Unit<2>();
    auto axis = make_vec<storage_of_t<T>, 3>(
        [&](int i)
        {
            auto c0 = select(zero_of<T>(), u0(i), use0);
            auto c1 = select(c0, u1(i), use1);
            return select(c1, u2(i), use2);
        });

    return normalized_completion(cross(u, axis), make_scalar<T>(1e-30),
                                 LVec<T, 3>::template Unit<0>());
}

/**
 * @brief One-sided Jacobi SVD of a 3x3 matrix: `A == U * diag(sigma) * V^T`.
 *
 * `sigma` is sorted descending and non-negative; `U` and `V` are orthogonal but
 * not necessarily proper rotations (see `svd` for the signed convention).
 */
template <typename T>
inline void one_sided_jacobi_svd(const LMat<T, 3, 3>& A,
                                 LMat<T, 3, 3>&       U,
                                 LVec<T, 3>&          sigma,
                                 LMat<T, 3, 3>&       V,
                                 int                  sweeps = kSvdSweeps) noexcept
{
    auto B = A;
    V      = LMat<T, 3, 3>::Identity();

    // --- orthogonalise the columns of B by cyclic rotations ------------------
    for(int sweep = 0; sweep < sweeps; ++sweep)
    {
        for(int p = 0; p < 2; ++p)
        {
            for(int q = p + 1; q < 3; ++q)
            {
                auto alpha = squaredNorm(B.col(p));
                auto beta  = squaredNorm(B.col(q));
                auto gamma = dot(B.col(p), B.col(q));

                auto two_gamma = detail::mul(make_scalar<T>(2), gamma);
                auto zeta      = detail::div(detail::sub(beta, alpha), two_gamma);
                // Classical rotation t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2)) with
                // **sign(0) := +1**, *not* `lmath::sign` (whose sign(0) is 0).  The
                // convention matters for the same reason it does in
                // `evd.h::jacobi_rotate`: `zeta == 0` (equal squared column norms) with
                // `gamma != 0` is the 45 degree rotation (t = 1), not the identity.
                // `sign(0) == 0` leaves the off-diagonal un-annihilated, so the sweep
                // returns non-orthogonal columns (and a wrong sigma/U/V) for any matrix
                // whose columns have equal norm but are not orthogonal.
                auto sgn       = select(one_of<T>(), make_scalar<T>(-1), zeta < zero_of<T>());
                auto t_raw     = detail::div(sgn,
                                             detail::add(abs(zeta),
                                                         sqrt(detail::add(one_of<T>(),
                                                                          detail::mul(zeta, zeta)))));
                // gamma == 0 => the columns are already orthogonal => t = 0.
                // (Also discards the NaN/Inf of zeta == 0/0.)
                auto t = select(zero_of<T>(), t_raw, gamma != zero_of<T>());
                auto c = detail::div(one_of<T>(),
                                     sqrt(detail::add(one_of<T>(), detail::mul(t, t))));
                auto s = detail::mul(c, t);

                // B <- B * J, J = [[c, s], [-s, c]] acting on columns p, q.
                // The sign convention matters: it is the one for which
                // (J^T M J)_pq vanishes for the classical
                // t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2))
                // with zeta = (|b_q|^2 - |b_p|^2) / (2 b_p . b_q)
                // (Golub & Van Loan, Alg. 8.6.2), sign(0) := +1.
                for(int r = 0; r < 3; ++r)
                {
                    auto bp = B(r, p);
                    auto bq = B(r, q);
                    B(r, p) = detail::sub(detail::mul(c, bp), detail::mul(s, bq));
                    B(r, q) = detail::add(detail::mul(s, bp), detail::mul(c, bq));
                    auto vp = V(r, p);
                    auto vq = V(r, q);
                    V(r, p) = detail::sub(detail::mul(c, vp), detail::mul(s, vq));
                    V(r, q) = detail::add(detail::mul(s, vp), detail::mul(c, vq));
                }
            }
        }
    }

    // --- singular values are the column norms; order them descending ---------
    for(int i = 0; i < 3; ++i)
        sigma(i) = sqrt(squaredNorm(B.col(i)));
    detail::sort_descending(sigma, B, V);

    // The sign of the *third* column of B relative to cross(b0, b1) is carried by
    // sigma(2): that is what makes U (built below as a right-handed frame)
    // proper, and it is the "reflection in the smallest singular value" of the
    // reference's signed-SVD convention.
    auto one     = one_of<T>();
    auto neg_one = make_scalar<T>(-1);
    sigma(2)     = detail::mul(sigma(2),
                               select(one, neg_one, determinant(B) < zero_of<T>()));

    // --- U = B * diag(1/sigma), with an orthonormal completion for rank loss --
    const auto eps  = static_cast<scalar_value_t<T>>(1e-30);
    auto       epsv = make_scalar<T>(eps);

    auto u0 = B.col(0).eval();
    u0      = normalized_completion(u0, epsv);

    auto w1 = u0 * dot(u0, B.col(1));
    auto u1 = B.col(1).eval() - w1;
    // Fallback when column 1 vanished: any unit vector orthogonal to u0.
    auto a1 = axis_orthogonal_to(u0);
    u1      = normalized_completion(u1, epsv, a1);

    // cross(u0, u1) is right-handed, so det(U) == +1 by construction.
    auto u2 = cross(u0, u1);

    U = LMat<T, 3, 3>::Zero();
    for(int r = 0; r < 3; ++r)
    {
        U(r, 0) = u0(r);
        U(r, 1) = u1(r);
        U(r, 2) = u2(r);
    }
}

/**
 * @brief Reference-compatible SVD with the signed convention.
 *
 * Proper rotations U, V (det = +1) with a possible reflection absorbed into the
 * smallest singular value (which may become negative when det(A) < 0) — exactly
 * the `signed_svd_fixup` of the reference backend.
 */
template <typename T>
inline void svd(const LMat<T, 3, 3>& A,
                LMat<T, 3, 3>&       U,
                LVec<T, 3>&          Sigma,
                LMat<T, 3, 3>&       V,
                int                  sweeps = kSvdSweeps) noexcept
{
    one_sided_jacobi_svd(A, U, Sigma, V, sweeps);

    // U came out proper (det(U) == +1).  The reflection of the reference's
    // convention now lives in Sigma(2); V only needs the same treatment as a
    // whole, which is a sign flip of its third column *together with* Sigma(2)
    // (so `U diag(sigma) V^T` is unchanged).
    auto detV = determinant(V);
    auto flip = select(one_of<T>(), make_scalar<T>(-1), detV < zero_of<T>());
    for(int r = 0; r < 3; ++r)
        V(r, 2) = detail::mul(V(r, 2), flip);
    Sigma(2) = detail::mul(Sigma(2), flip);
}

/// The reference's spelling (`cuda_tool::eigen::svd` calls `math::qr_svd`).
/// Kept so that ported call sites do not have to change; the implementation is
/// one-sided Jacobi, see the file comment.
template <typename T>
inline void qr_svd(const LMat<T, 3, 3>& A,
                   LVec<T, 3>&          S,
                   LMat<T, 3, 3>&       U,
                   LMat<T, 3, 3>&       V,
                   int                  sweeps = kSvdSweeps) noexcept
{
    svd(A, U, S, V, sweeps);
}

/**
 * @brief Polar decomposition `A == R * S` with R orthogonal and S symmetric.
 *
 * Reference-compatible (`cuda_tool::eigen::pd`):
 * `R = U V^T`, `S = V diag(sigma) V^T`.
 */
template <typename T>
inline void pd(const LMat<T, 3, 3>& A, LMat<T, 3, 3>& R, LMat<T, 3, 3>& S,
               int sweeps = kSvdSweeps) noexcept
{
    LMat<T, 3, 3> U, V;
    LVec<T, 3>    sigma;
    svd(A, U, sigma, V, sweeps);
    R = U * V.transpose();
    S = V * sigma.asDiagonal() * V.transpose();
}

/// Alias with the descriptive name used in the plan.
template <typename T>
inline void polar_decomposition(const LMat<T, 3, 3>& A, LMat<T, 3, 3>& R, LMat<T, 3, 3>& S,
                                int sweeps = kSvdSweeps) noexcept
{
    pd(A, R, S, sweeps);
}
}// namespace uipc::backend::luisa::luisa_device::math
