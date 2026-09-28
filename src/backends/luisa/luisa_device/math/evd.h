#pragma once
/********************************************************************
 * @file evd.h
 * @brief T2.2 — symmetric eigen decomposition (Jacobi) for the generic layer.
 *
 * Ported from the reference backend's `cuda_tool::eigen::evd_jacobi`
 * (`src/backends/cuda/cuda_tool/eigen/evd.h`), with one structural change that the
 * LC model forces (and the cookbook's RULE M2 states): the reference iterates
 *
 *     while (calc_sumDiagOff(M) > 1e-6) { find_max_offdiag(M, &p, &q); rotate(p,q); }
 *
 * Both the termination test and the pivot search are *device* values there, i.e.
 * they would have to drive host control flow. Here the sweep count and the pivot
 * order are fixed, so the routine is a single branchless implementation that is
 * valid on the host and on the device and produces identical results for both.
 *
 *  - pivots cycle over the strictly-upper triangle (cyclic Jacobi), and
 *  - the number of sweeps is a compile-time constant (cyclic Jacobi converges
 *    quadratically; 20 sweeps reach machine precision for N <= 12).
 *
 * The rotation guard `|Mpq| < 1e-12 -> return` becomes a `select`: the classical
 * shift-free branch is evaluated for both cases and the harmless value is picked.
 * That avoids the 0/0 NaN of `tau = (Mqq - Mpp) / (2 * Mpq)` when Mpq == 0.
 *********************************************************************/
#include <luisa_device/math/mat.h>

#include <cstddef>

namespace uipc::backend::luisa::luisa_device::math
{
/// Default cyclic-sweep count; plenty for the small systems this layer handles.
inline constexpr int kJacobiSweeps = 20;

namespace detail
{
/// One Jacobi rotation annihilating M[p][q] (p != q), updating the vectors too.
template <typename T, int N>
inline void jacobi_rotate(LMat<T, N, N>& M, LMat<T, N, N>& E, int p, int q) noexcept
{
    // Guarded classical formula, unified over both branches of the reference:
    //     t = 1 / (tau + sqrt(1 + tau^2))               (tau >= 0)
    //     t = -1 / (-tau + sqrt(1 + tau^2))             (tau <  0)
    // <=> t = sign(tau) / (|tau| + sqrt(1 + tau^2))
    auto Mpq = M(p, q);
    auto tau = detail::div(detail::sub(M(q, q), M(p, p)), detail::mul(make_scalar<T>(2), Mpq));

    auto abs_tau = abs(tau);
    auto t_raw   = detail::div(sign(tau),
                               detail::add(abs_tau, sqrt(detail::add(make_scalar<T>(1),
                                                                     detail::mul(tau, tau)))));
    // Mpq == 0  =>  the rotation must be the identity (t = 0); this also discards
    // the NaN that 0/0 would produce.
    auto t = select(zero_of<T>(), t_raw, /*predicate=*/Mpq != zero_of<T>());

    auto c = detail::div(one_of<T>(),
                         sqrt(detail::add(one_of<T>(), detail::mul(t, t))));
    auto s = detail::mul(t, c);

    for(int i = 0; i < N; ++i)
    {
        if(i != p && i != q)
        {
            auto Mip = M(i, p);
            auto Miq = M(i, q);
            M(p, i) = M(i, p) = detail::sub(detail::mul(c, Mip), detail::mul(s, Miq));
            M(q, i) = M(i, q) = detail::add(detail::mul(s, Mip), detail::mul(c, Miq));
        }

        auto Eip = E(i, p);
        auto Eiq = E(i, q);
        E(i, p)  = detail::sub(detail::mul(c, Eip), detail::mul(s, Eiq));
        E(i, q)  = detail::add(detail::mul(s, Eip), detail::mul(c, Eiq));
    }

    auto Mpp = M(p, p);
    auto Mqq = M(q, q);

    // M(p,p) = c*c*Mpp + s*s*Mqq - 2*c*s*Mpq
    // M(q,q) = s*s*Mpp + c*c*Mqq + 2*c*s*Mpq
    auto two_cs_Mpq = detail::mul(detail::mul(make_scalar<T>(2), detail::mul(c, s)), Mpq);

    M(p, p) = detail::sub(detail::add(detail::mul(detail::mul(c, c), Mpp),
                                      detail::mul(detail::mul(s, s), Mqq)),
                          two_cs_Mpq);
    M(q, q) = detail::add(detail::add(detail::mul(detail::mul(s, s), Mpp),
                                      detail::mul(detail::mul(c, c), Mqq)),
                          two_cs_Mpq);
    M(p, q) = M(q, p) = zero_of<T>();
}

/// Odd-even transposition sort of (values, columns) — device-safe (no host
/// branches on device data): the comparison *pattern* is host-fixed and each
/// comparison result drives `select`s only.
template <typename T, int N>
inline void sort_eigensystem(LVec<T, N>& values, LMat<T, N, N>& vectors, bool ascending) noexcept
{
    // "should the two entries be swapped?"  The predicate may be a DSL value, so
    // the two orders must not be mixed in a host ternary.
    auto before = [&](int i, int j) noexcept
    {
        if(ascending)
            return values(i) > values(j);
        else
            return values(i) < values(j);
    };

    for(int pass = 0; pass < N; ++pass)
    {
        for(int i = 0; i + 1 < N; ++i)
        {
            if((i + pass) % 2 != 0)
                continue;
            auto swap = before(i, i + 1);
            for(int r = 0; r < N; ++r)
            {
                auto a            = vectors(r, i);
                auto b            = vectors(r, i + 1);
                vectors(r, i)     = select(a, b, swap);
                vectors(r, i + 1) = select(b, a, swap);
            }
            auto va       = values(i);
            auto vb       = values(i + 1);
            values(i)     = select(va, vb, swap);
            values(i + 1) = select(vb, va, swap);
        }
    }
}
}// namespace detail

/**
 * @brief Eigen decomposition of a symmetric NxN matrix (cyclic Jacobi).
 *
 * @param M        symmetric matrix (only the symmetric part is used implicitly)
 * @param values   eigenvalues
 * @param vectors  orthonormal eigenvectors, one per *column*
 * @param sort     sort eigenvalues ascending (the reference's evd_jacobi does)
 * @param sweeps   number of cyclic sweeps
 */
template <typename T, int N>
inline void jacobi_evd(const LMat<T, N, N>& M,
                       LVec<T, N>&          values,
                       LMat<T, N, N>&       vectors,
                       bool                 sort   = true,
                       int                  sweeps = kJacobiSweeps) noexcept
{
    auto A = M;
    vectors.setIdentity();

    for(int sweep = 0; sweep < sweeps; ++sweep)
        for(int p = 0; p < N - 1; ++p)
            for(int q = p + 1; q < N; ++q)
                detail::jacobi_rotate(A, vectors, p, q);

    for(int i = 0; i < N; ++i)
        values(i) = A(i, i);

    if(sort)
        detail::sort_eigensystem(values, vectors, /*ascending=*/true);
}

/// Reference-compatible spelling (`cuda_tool::eigen::evd`).
template <typename T, int N>
inline void evd(const LMat<T, N, N>& M,
                LVec<T, N>&          values,
                LMat<T, N, N>&       vectors,
                bool                 sort   = true,
                int                  sweeps = kJacobiSweeps) noexcept
{
    jacobi_evd(M, values, vectors, sort, sweeps);
}
}// namespace uipc::backend::luisa::luisa_device::math
