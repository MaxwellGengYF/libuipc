#pragma once
/********************************************************************
 * @file matrix_ops.h
 * @brief T2.2 — determinant / inverse / make_spd for the generic layer.
 *
 * Ported from the reference backend:
 *  - `cuda_tool::eigen::AnalyticalInverse` (`cuda_tool/eigen/analytic_inverse.h`)
 *    — closed-form adjugate inverses, verbatim formulas, operating on the same
 *    column-major element order (`data[c * R + r]`, Eigen's layout).
 *  - `utils/make_spd.h` — eigen-clamp-reconstruct, on top of evd.h's Jacobi.
 *
 * The formulas are arithmetic on possibly-DSL scalars, so every operation goes
 * through the `detail::{add,sub,mul,div}` trampolines (cookbook RULE M1); the
 * local lambda aliases keep the ported expressions readable.
 *********************************************************************/
#include <luisa_device/math/evd.h>

namespace uipc::backend::luisa::luisa_device::math
{
namespace detail
{
/// Infix-style spellings of the scalar trampolines, for readable formulas.
/// (RULE M1: a bare `a * b` on possibly-DSL scalars would not compile here,
/// because our own operators shadow LC's global ones.)
#define UIPC_LUISA_ARITH_ALIASES()                         \
    auto add = [](auto&& a, auto&& b) noexcept             \
    { return detail::add(a, b); };                         \
    auto sub = [](auto&& a, auto&& b) noexcept             \
    { return detail::sub(a, b); };                         \
    auto mul = [](auto&& a, auto&& b) noexcept             \
    { return detail::mul(a, b); };                         \
    auto div = [](auto&& a, auto&& b) noexcept             \
    { return detail::div(a, b); };                         \
    auto neg = [](auto&& a) noexcept                       \
    { return detail::neg(a); };                            \
    auto mul3 = [&](auto&& a, auto&& b, auto&& c) noexcept \
    { return mul(mul(a, b), c); }

/// Adjugate of a 2x2 matrix (column-major `src`), matching the reference.
template <typename T>
inline void adjugate2x2(const T* src, T* dst) noexcept
{
    UIPC_LUISA_ARITH_ALIASES();
    dst[0] = src[3];
    dst[1] = neg(src[1]);
    dst[2] = neg(src[2]);
    dst[3] = src[0];
    (void)add;
    (void)sub;
    (void)mul;
    (void)div;
    (void)mul3;
}

/// Adjugate of a 3x3 matrix (column-major `src`), matching the reference.
template <typename T>
inline void adjugate3x3(const T* src, T* dst) noexcept
{
    UIPC_LUISA_ARITH_ALIASES();
    dst[0] = sub(mul(src[4], src[8]), mul(src[5], src[7]));
    dst[1] = sub(mul(src[2], src[7]), mul(src[1], src[8]));
    dst[2] = sub(mul(src[1], src[5]), mul(src[2], src[4]));
    dst[3] = sub(mul(src[5], src[6]), mul(src[3], src[8]));
    dst[4] = sub(mul(src[0], src[8]), mul(src[2], src[6]));
    dst[5] = sub(mul(src[2], src[3]), mul(src[0], src[5]));
    dst[6] = sub(mul(src[3], src[7]), mul(src[4], src[6]));
    dst[7] = sub(mul(src[1], src[6]), mul(src[0], src[7]));
    dst[8] = sub(mul(src[0], src[4]), mul(src[1], src[3]));
    (void)add;
    (void)div;
    (void)neg;
    (void)mul3;
}

/// Adjugate of a 4x4 matrix (column-major `src`), matching the reference
/// term-for-term (each entry is a 6-term signed sum of triple products).  The
/// expressions are evaluated left-to-right in the reference's own order so the
/// floating-point result is bit-comparable with the CUDA backend.
template <typename T>
inline void adjugate4x4(const T* src, T* dst) noexcept
{
    UIPC_LUISA_ARITH_ALIASES();
    auto term = [&](int sign, int i, int j, int k) noexcept
    {
        auto v = mul3(src[i], src[j], src[k]);
        if(sign > 0)
            return stored(v);
        return stored(neg(v));
    };

    dst[0] = term(1, 5, 10, 15);
    dst[0] = sub(dst[0], term(1, 5, 11, 14));
    dst[0] = sub(dst[0], term(1, 9, 6, 15));
    dst[0] = add(dst[0], term(1, 9, 7, 14));
    dst[0] = add(dst[0], term(1, 13, 6, 11));
    dst[0] = sub(dst[0], term(1, 13, 7, 10));
    dst[1] = term(-1, 1, 10, 15);
    dst[1] = add(dst[1], term(1, 1, 11, 14));
    dst[1] = add(dst[1], term(1, 9, 2, 15));
    dst[1] = sub(dst[1], term(1, 9, 3, 14));
    dst[1] = sub(dst[1], term(1, 13, 2, 11));
    dst[1] = add(dst[1], term(1, 13, 3, 10));
    dst[2] = term(1, 1, 6, 15);
    dst[2] = sub(dst[2], term(1, 1, 7, 14));
    dst[2] = sub(dst[2], term(1, 5, 2, 15));
    dst[2] = add(dst[2], term(1, 5, 3, 14));
    dst[2] = add(dst[2], term(1, 13, 2, 7));
    dst[2] = sub(dst[2], term(1, 13, 3, 6));
    dst[3] = term(-1, 1, 6, 11);
    dst[3] = add(dst[3], term(1, 1, 7, 10));
    dst[3] = add(dst[3], term(1, 5, 2, 11));
    dst[3] = sub(dst[3], term(1, 5, 3, 10));
    dst[3] = sub(dst[3], term(1, 9, 2, 7));
    dst[3] = add(dst[3], term(1, 9, 3, 6));
    dst[4] = term(-1, 4, 10, 15);
    dst[4] = add(dst[4], term(1, 4, 11, 14));
    dst[4] = add(dst[4], term(1, 8, 6, 15));
    dst[4] = sub(dst[4], term(1, 8, 7, 14));
    dst[4] = sub(dst[4], term(1, 12, 6, 11));
    dst[4] = add(dst[4], term(1, 12, 7, 10));
    dst[5] = term(1, 0, 10, 15);
    dst[5] = sub(dst[5], term(1, 0, 11, 14));
    dst[5] = sub(dst[5], term(1, 8, 2, 15));
    dst[5] = add(dst[5], term(1, 8, 3, 14));
    dst[5] = add(dst[5], term(1, 12, 2, 11));
    dst[5] = sub(dst[5], term(1, 12, 3, 10));
    dst[6] = term(-1, 0, 6, 15);
    dst[6] = add(dst[6], term(1, 0, 7, 14));
    dst[6] = add(dst[6], term(1, 4, 2, 15));
    dst[6] = sub(dst[6], term(1, 4, 3, 14));
    dst[6] = sub(dst[6], term(1, 12, 2, 7));
    dst[6] = add(dst[6], term(1, 12, 3, 6));
    dst[7] = term(1, 0, 6, 11);
    dst[7] = sub(dst[7], term(1, 0, 7, 10));
    dst[7] = sub(dst[7], term(1, 4, 2, 11));
    dst[7] = add(dst[7], term(1, 4, 3, 10));
    dst[7] = add(dst[7], term(1, 8, 2, 7));
    dst[7] = sub(dst[7], term(1, 8, 3, 6));
    dst[8] = term(1, 4, 9, 15);
    dst[8] = sub(dst[8], term(1, 4, 11, 13));
    dst[8] = sub(dst[8], term(1, 8, 5, 15));
    dst[8] = add(dst[8], term(1, 8, 7, 13));
    dst[8] = add(dst[8], term(1, 12, 5, 11));
    dst[8] = sub(dst[8], term(1, 12, 7, 9));
    dst[9] = term(-1, 0, 9, 15);
    dst[9] = add(dst[9], term(1, 0, 11, 13));
    dst[9] = add(dst[9], term(1, 8, 1, 15));
    dst[9] = sub(dst[9], term(1, 8, 3, 13));
    dst[9] = sub(dst[9], term(1, 12, 1, 11));
    dst[9] = add(dst[9], term(1, 12, 3, 9));
    dst[10] = term(1, 0, 5, 15);
    dst[10] = sub(dst[10], term(1, 0, 7, 13));
    dst[10] = sub(dst[10], term(1, 4, 1, 15));
    dst[10] = add(dst[10], term(1, 4, 3, 13));
    dst[10] = add(dst[10], term(1, 12, 1, 7));
    dst[10] = sub(dst[10], term(1, 12, 3, 5));
    dst[11] = term(-1, 0, 5, 11);
    dst[11] = add(dst[11], term(1, 0, 7, 9));
    dst[11] = add(dst[11], term(1, 4, 1, 11));
    dst[11] = sub(dst[11], term(1, 4, 3, 9));
    dst[11] = sub(dst[11], term(1, 8, 1, 7));
    dst[11] = add(dst[11], term(1, 8, 3, 5));
    dst[12] = term(-1, 4, 9, 14);
    dst[12] = add(dst[12], term(1, 4, 10, 13));
    dst[12] = add(dst[12], term(1, 8, 5, 14));
    dst[12] = sub(dst[12], term(1, 8, 6, 13));
    dst[12] = sub(dst[12], term(1, 12, 5, 10));
    dst[12] = add(dst[12], term(1, 12, 6, 9));
    dst[13] = term(1, 0, 9, 14);
    dst[13] = sub(dst[13], term(1, 0, 10, 13));
    dst[13] = sub(dst[13], term(1, 8, 1, 14));
    dst[13] = add(dst[13], term(1, 8, 2, 13));
    dst[13] = add(dst[13], term(1, 12, 1, 10));
    dst[13] = sub(dst[13], term(1, 12, 2, 9));
    dst[14] = term(-1, 0, 5, 14);
    dst[14] = add(dst[14], term(1, 0, 6, 13));
    dst[14] = add(dst[14], term(1, 4, 1, 14));
    dst[14] = sub(dst[14], term(1, 4, 2, 13));
    dst[14] = sub(dst[14], term(1, 12, 1, 6));
    dst[14] = add(dst[14], term(1, 12, 2, 5));
    dst[15] = term(1, 0, 5, 10);
    dst[15] = sub(dst[15], term(1, 0, 6, 9));
    dst[15] = sub(dst[15], term(1, 4, 1, 10));
    dst[15] = add(dst[15], term(1, 4, 2, 9));
    dst[15] = add(dst[15], term(1, 8, 1, 6));
    dst[15] = sub(dst[15], term(1, 8, 2, 5));
}

#undef UIPC_LUISA_ARITH_ALIASES
}// namespace detail

// ---------------------------------------------------------------------------
//  Determinant
// ---------------------------------------------------------------------------
template <typename T, int N>
[[nodiscard]] inline T determinant(const LMat<T, N, N>& m) noexcept
{
    static_assert(N == 2 || N == 3 || N == 4,
                  "determinant is only implemented for 2x2, 3x3 and 4x4 here");
    if constexpr(N == 2)
    {
        return detail::sub(detail::mul(m.data[0], m.data[3]),
                           detail::mul(m.data[2], m.data[1]));
    }
    else if constexpr(N == 3)
    {
        T adj[9];
        detail::adjugate3x3(m.data.data(), adj);
        return detail::add(detail::add(detail::mul(m.data[0], adj[0]),
                                       detail::mul(m.data[1], adj[3])),
                           detail::mul(m.data[2], adj[6]));
    }
    else
    {
        T adj[16];
        detail::adjugate4x4(m.data.data(), adj);
        return detail::add(
            detail::add(detail::add(detail::mul(m.data[0], adj[0]),
                                    detail::mul(m.data[1], adj[4])),
                        detail::mul(m.data[2], adj[8])),
            detail::mul(m.data[3], adj[12]));
    }
}

// ---------------------------------------------------------------------------
//  Inverse (closed form, reference-equivalent)
// ---------------------------------------------------------------------------
template <typename T, int N>
[[nodiscard]] inline LMat<T, N, N> inverse(const LMat<T, N, N>& m) noexcept
{
    static_assert(N == 2 || N == 3 || N == 4,
                  "inverse is only implemented for 2x2, 3x3 and 4x4 here");
    LMat<T, N, N> out;
    T             det;
    if constexpr(N == 2)
    {
        detail::adjugate2x2(m.data.data(), out.data.data());
        det = detail::add(detail::mul(m.data[0], out.data[0]),
                          detail::mul(m.data[1], out.data[2]));
    }
    else if constexpr(N == 3)
    {
        detail::adjugate3x3(m.data.data(), out.data.data());
        det = detail::add(detail::add(detail::mul(m.data[0], out.data[0]),
                                      detail::mul(m.data[1], out.data[3])),
                          detail::mul(m.data[2], out.data[6]));
    }
    else
    {
        detail::adjugate4x4(m.data.data(), out.data.data());
        det = detail::add(
            detail::add(detail::add(detail::mul(m.data[0], out.data[0]),
                                    detail::mul(m.data[1], out.data[4])),
                        detail::mul(m.data[2], out.data[8])),
            detail::mul(m.data[3], out.data[12]));
    }

    auto inv_det = detail::div(one_of<T>(), det);
    for(std::size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = detail::mul(out.data[i], inv_det);
    return out;
}

/// Member spellings (Eigen's `m.determinant()` / `m.inverse()`), declared in mat.h.
template <typename T, int R, int C>
[[nodiscard]] inline auto LMat<T, R, C>::determinant() const noexcept
{
    static_assert(R == C, "determinant needs a square matrix");
    return math::determinant(*this);
}

template <typename T, int R, int C>
[[nodiscard]] inline LMat<T, R, C> LMat<T, R, C>::inverse() const noexcept
{
    static_assert(R == C, "inverse needs a square matrix");
    return math::inverse(*this);
}

// ---------------------------------------------------------------------------
//  make_spd
// ---------------------------------------------------------------------------
/**
 * @brief Project a symmetric matrix onto the PSD cone.
 *
 * Reference (`utils/make_spd.h`): eigen-decompose, clamp negative eigenvalues to
 * zero, reconstruct `V diag(max(lambda,0)) V^T`.
 */
template <typename T, int N>
[[nodiscard]] inline LMat<T, N, N> make_spd(const LMat<T, N, N>& H,
                                            int                  sweeps = kJacobiSweeps) noexcept
{
    LVec<T, N>    eigenvalues;
    LMat<T, N, N> eigenvectors;
    jacobi_evd(H, eigenvalues, eigenvectors, /*sort=*/false, sweeps);

    for(int i = 0; i < N; ++i)
        // max(lambda, 0) -- `select` (not `lcrt::select`) so the host path, where
        // the predicate is a plain bool, works too.
        eigenvalues(i) = select(eigenvalues(i), zero_of<T>(),
                                /*predicate=*/eigenvalues(i) < zero_of<T>());

    auto D = eigenvalues.asDiagonal();
    return eigenvectors * D * eigenvectors.transpose();
}

/// In-place flavour, matching the reference's `void make_spd(Matrix&)`.
template <typename T, int N>
inline void make_spd_inplace(LMat<T, N, N>& H, int sweeps = kJacobiSweeps) noexcept
{
    H = make_spd(H, sweeps);
}
}// namespace uipc::backend::luisa::luisa_device::math
