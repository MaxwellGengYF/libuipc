#pragma once
/********************************************************************
 * @file flagged.h
 * @brief T2.3 — flag-dispatched ("codim") distance / gradient / Hessian.
 *
 * Ported from `src/backends/cuda/utils/distance/distance_flagged.h` (the one
 * reference file that is self-contained: header + implementation in one).
 *
 * ## The reference
 * A flagged simplex carries, per vertex, an `IndexT` flag of 0/1.  `dim` — the
 * number of set flags — selects which sub-simplex the actual distance is taken
 * against:
 *
 *     dim 2 -> point-point     (`pp_from_pt` / `pp_from_ee` / `pp_from_pe`)
 *     dim 3 -> point-edge      (`pe_from_pt` / `pe_from_ee`)
 *     dim 4 -> the full simplex
 *
 * The reference switches on `dim` and on individual flags with nested host-style
 * `if`s.  Inside an LC kernel those are *device* values (a flag vector is computed
 * from device positions in the collision filters), so they cannot drive host
 * control flow (RULE M2, cookbook §7.3).  The port therefore
 *
 *  - keeps the reference's exact per-dim layouts as **host constant tables**
 *    (`kPeFromPt` / `kPeFromEe` are the reference's `pe_from_pt`/`pe_from_ee`
 *    bodies, transcribed), and
 *  - evaluates the (at most) 6 + 4 + 1 canonical sub-cases and *selects* the one
 *    the flags designate, rather than branching to it.
 *
 * Equivalence: the reference computes the canonical primitive for exactly one
 * subset/permutation and leaves the other output slots zero; the port computes
 * each candidate — with the same host-permuted arguments, so the canonical call is
 * literally the same function with the same inputs, evaluated in the same place of
 * the reference's own arithmetic — into the *same* slots the reference would
 * write, and chains `select`s with mutually exclusive predicates.  A select passes
 * its chosen operand through unchanged, so every output entry equals the
 * reference's bit for bit; the unselected candidates are only extra work (and any
 * NaN/Inf they produce is discarded by the select exactly like the reference's
 * not-taken branch would have been).
 *
 * ## Flags as values
 * A flag vector of a device call is itself a device value, so the port's flag
 * element type is `flag_index_t<T>`: `int` for a host `T`, a DSL `Var<int>` for a
 * device `T` (`Var`, not `Expr`: the flags are built by `select`s, so they must be
 * assignable — cookbook §7.1).  Helpers that only touch flags are templated on
 * that element type itself, because `FlagVec<T,N>` is an alias for
 * `LVec<flag_index_t<T>,N>` — a dependent alias is a non-deduced context.
 *
 * ## Invalid flags
 * The reference calls `UIPC_KERNEL_ERROR_WITH_LOCATION` when `active_count` is not
 * 2/3/4.  There is no generic host/device error path here, so an invalid flag
 * leaves the output at zero (documented in the T2.3 report).
 *********************************************************************/
#include <luisa_device/math/distance/distance_common.h>
#include <luisa_device/math/distance/edge_edge.h>
#include <luisa_device/math/distance/point_edge.h>
#include <luisa_device/math/distance/point_point.h>
#include <luisa_device/math/distance/point_triangle.h>
#include <luisa_device/math/matrix_ops.h>

namespace uipc::backend::luisa::luisa_device::math::distance
{
// RULE M1 spellings: the ported expressions below are arithmetic on possibly-DSL
// scalars, and this namespace declares no operators of its own.
using math::detail::add;
using math::detail::div;
using math::detail::mul;
using math::detail::neg;
using math::detail::sub;

/// Flag element type: `int` on the host, a DSL `Var<int>` on the device.
template <typename T>
using flag_index_t = std::conditional_t<std::is_arithmetic_v<T>, int, lcrt::Var<int>>;

/// Flag vector (`Vector<IndexT, N>` of the reference).
template <typename T, int N>
using FlagVec = LVec<flag_index_t<T>, N>;

namespace detail
{
/// The reference's `active_count`: how many flags are set.
template <typename F, int N>
[[nodiscard]] inline math::storage_of_t<F> active_count(const LVec<F, N>& flag) noexcept
{
    math::storage_of_t<F> count = zero_of<F>();
    for(int i = 0; i < N; ++i)
        count = add(count, flag(i));
    return count;
}

/// The reference's `pp_from_*` / `pe_from_*` layout: the *ascending* list of active
/// indices, gathered into `offsets`.
///
/// `offsets(s)` is the (s+1)-th active index, or the out-of-range sentinel `N` when
/// there is no (s+1)-th active index.  (The reference leaves those slots untouched,
/// i.e. undefined, and never reads them; the sentinel makes the port deterministic
/// *and* is what keeps a predicated scatter from writing into unused slots.)
///
/// A device index cannot index a host array (RULE M2), so the gather is a
/// host-fixed O(N^2) predicated select chain: `rank(i)` — the number of active
/// flags before `i` — is a device value, and `rank(i) == s` picks the slot.
template <typename F, int N>
inline void active_offsets(const LVec<F, N>& flag, LVec<F, N>& offsets) noexcept
{
    std::array<math::storage_of_t<F>, static_cast<std::size_t>(N)> rank{};
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < i; ++j)
            rank[static_cast<std::size_t>(i)] = add(rank[static_cast<std::size_t>(i)], flag(j));

    for(int s = 0; s < N; ++s)
    {
        math::storage_of_t<F> value = make_scalar<F>(N);  // "none"
        for(int i = 0; i < N; ++i)
            value = select_scalar(value,
                                  make_scalar<F>(i),
                                  /*predicate=*/(rank[static_cast<std::size_t>(i)] == s)
                                      & (flag(i) != 0));
        offsets(s) = value;
    }
}

/// The reference's `pe_from_pt`, indexed by the *inactive* index: the (point,
/// edge-end, edge-end) triple of the active vertices.
inline constexpr int kPeFromPt[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};

/// The reference's `pe_from_ee`, indexed by the *inactive* index.  Unlike
/// `pe_from_pt` this is **not** the ascending triple: the point of the resulting
/// point-edge pair is the free endpoint of the degenerate edge (the active vertex
/// whose partner is inactive), e.g. active {0,1,3} -> (3,0,1).
inline constexpr int kPeFromEe[4][3] = {{1, 2, 3}, {0, 2, 3}, {3, 0, 1}, {2, 0, 1}};

/// A flag vector built from host bits — the reference's `Vector3i{...}` literals.
template <typename F>
[[nodiscard]] inline LVec<F, 3> flag3(int a, int b, int c) noexcept
{
    return math::make_vec<F, 3>([&](int i) noexcept
                                { return make_scalar<F>(i == 0 ? a : (i == 1 ? b : c)); });
}

/// A flag vector built from host bits — the reference's `Vector4i{...}` literals.
template <typename F>
[[nodiscard]] inline LVec<F, 4> flag4(int a, int b, int c, int d) noexcept
{
    return math::make_vec<F, 4>([&](int i) noexcept
                                { return make_scalar<F>(i == 0 ? a : (i == 1 ? b : (i == 2 ? c : d))); });
}
}  // namespace detail

// ---------------------------------------------------------------------------
//  degenerate_* — the reference's "which sub-simplex is active" helpers
// ---------------------------------------------------------------------------
/// `dim` of a point-edge flag plus the active-index layout (`pp_from_pe` for
/// `dim == 2`, the identity for `dim == 3`).
template <typename F>
[[nodiscard]] inline math::storage_of_t<F> degenerate_point_edge(const LVec<F, 3>& flag,
                                                                LVec<F, 3>&       offsets) noexcept
{
    detail::active_offsets(flag, offsets);
    return detail::active_count(flag);
}

/// `dim` of a point-triangle flag plus the active-index layout (`pp_from_pt` for
/// `dim == 2`, `pe_from_pt` for `dim == 3`, the identity for `dim == 4`).
template <typename F>
[[nodiscard]] inline math::storage_of_t<F> degenerate_point_triangle(const LVec<F, 4>& flag,
                                                                     LVec<F, 4>&       offsets) noexcept
{
    detail::active_offsets(flag, offsets);
    return detail::active_count(flag);
}

/// `dim` of an edge-edge flag plus the active-index layout (`pp_from_ee` for
/// `dim == 2`, `pe_from_ee` for `dim == 3`, the identity for `dim == 4`).
template <typename F>
[[nodiscard]] inline math::storage_of_t<F> degenerate_edge_edge(const LVec<F, 4>& flag,
                                                                LVec<F, 4>&       offsets) noexcept
{
    detail::active_offsets(flag, offsets);

    auto dim = detail::active_count(flag);
    // `pe_from_ee` is **not** the ascending triple (unlike `pe_from_pt`): its entries
    // are already direct indices into the four vertices `{ea0, ea1, eb0, eb1}`, which
    // is how the reference consumes them (`Vector3i offsets = pe_from_ee(flag); P =
    // {ea0,ea1,eb0,eb1}; P[offsets[0..2]]`).  Re-gathering them through the ascending
    // layout is therefore wrong: active `{0,1,3}` (inactive 2, table row `{3,0,1}`)
    // became `{2,0,1}`, i.e. the point of the point-edge pair was `eb0` instead of the
    // free endpoint `eb1`.  Rows: inactive 0 -> {1,2,3}, 1 -> {0,2,3}, 2 -> {3,0,1},
    // 3 -> {2,0,1}.
    for(int k = 0; k < 4; ++k)
    {
        // `pe_from_ee(flag)` applies exactly when `dim == 3` and `k` is inactive
        auto pred = (dim == 3) & (flag(k) == 0);
        for(int s = 0; s < 3; ++s)
            offsets(s) = detail::select_scalar(offsets(s),
                                               make_scalar<F>(detail::kPeFromEe[k][s]),
                                               pred);
    }
    return dim;
}

// ---------------------------------------------------------------------------
//  *_distance_flag — the reference's flag constructors
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] inline FlagVec<T, 2> point_point_distance_flag(const LVec<T, 3>& p0,
                                                             const LVec<T, 3>& p1) noexcept
{
    (void)p0;
    (void)p1;
    FlagVec<T, 2> F;
    F(0) = make_scalar<flag_index_t<T>>(1);
    F(1) = make_scalar<flag_index_t<T>>(1);
    return F;
}

template <typename T>
[[nodiscard]] inline FlagVec<T, 3> point_edge_distance_flag(const LVec<T, 3>& p,
                                                            const LVec<T, 3>& e0,
                                                            const LVec<T, 3>& e1) noexcept
{
    FlagVec<T, 3> F;
    F(0) = make_scalar<flag_index_t<T>>(1);

    // F = {1, ratio<1, ratio>0}; a zero-length edge degenerates to point-point
    // against the nearer endpoint.  The reference *branches* on `e2 <= 0`; here
    // both alternatives are computed and selected (the discarded one may be a
    // 0/0 NaN, which the select drops exactly like the untaken branch would).
    auto e          = e1 - e0;
    T    e2         = math::squaredNorm(e);
    auto degenerate = e2 <= zero_of<T>();

    T    d0       = math::squaredNorm(p - e0);
    T    d1       = math::squaredNorm(p - e1);
    auto d0_first = d0 <= d1;

    auto ratio = div(math::dot(e, p - e0), e2);

    const auto zero_i = make_scalar<flag_index_t<T>>(0);
    const auto one_i  = make_scalar<flag_index_t<T>>(1);

    F(1) = detail::select_scalar(detail::select_scalar(zero_i, one_i, d0_first),
                                 detail::select_scalar(zero_i, one_i, ratio < 1.0),
                                 !degenerate);
    F(2) = detail::select_scalar(detail::select_scalar(one_i, zero_i, d0_first),
                                 detail::select_scalar(zero_i, one_i, ratio > 0.0),
                                 !degenerate);
    return F;
}

template <typename T>
[[nodiscard]] inline FlagVec<T, 4> point_triangle_distance_flag(const LVec<T, 3>& p,
                                                                const LVec<T, 3>& t0,
                                                                const LVec<T, 3>& t1,
                                                                const LVec<T, 3>& t2) noexcept
{
    // The reference's nested if/else has seven terminal cases.  They are mutually
    // exclusive and each yields a *complete* flag vector, so the whole chain is a
    // select chain over those seven vectors (the reference's F starts as {1,0,0,0}
    // and no case ever clears a bit an enclosing case set).
    auto e01  = t1 - t0;
    auto e02  = t2 - t0;
    auto nVec = math::cross(e01, e02);

    // `param.col(k) = (B B^T)^-1 (B (p - a_k))` with rows `b - a` and
    // `(b - a) x nVec` — the reference's three candidate edges.
    auto edge_param = [&](const LVec<T, 3>& a, const LVec<T, 3>& b, T& u, T& v) noexcept
    {
        auto r0  = b - a;
        auto r1  = math::cross(r0, nVec);
        auto d00 = math::dot(r0, r0);
        auto d01 = math::dot(r0, r1);
        auto d11 = math::dot(r1, r1);
        LMat<T, 2, 2> BBt(d00, d01, d01, d11);  // basis * basis^T (symmetric)
        auto          inv = math::inverse(BBt);
        auto          pa  = p - a;
        auto          rhs = math::make_vec<T, 2>([&](int i) noexcept
                                       { return i == 0 ? math::dot(r0, pa) : math::dot(r1, pa); });
        auto          par = inv * rhs;
        u                 = par(0);
        v                 = par(1);
    };

    T u0, v0, u1, v1, u2, v2;
    edge_param(t0, t1, u0, v0);
    edge_param(t1, t2, u1, v1);
    edge_param(t2, t0, u2, v2);

    const auto zero_t = zero_of<T>();
    const auto one_t  = one_of<T>();

    auto in_edge = [&](const T& u, const T& v) noexcept
    { return detail::pred_and(u > zero_t, u < one_t, v >= zero_t); };

    auto p0 = in_edge(u0, v0);  // PE t0t1
    auto p1 = in_edge(u1, v1);  // PE t1t2
    auto p2 = in_edge(u2, v2);  // PE t2t0

    auto q0 = (u0 <= zero_t) & (u2 >= one_t);  // PP t0
    auto q1 = (u1 <= zero_t) & (u0 >= one_t);  // PP t1
    auto q2 = (u2 <= zero_t) & (u1 >= one_t);  // PP t2

    auto not_pe = detail::pred_and(!p0, !p1, !p2);

    FlagVec<T, 4> F;
    F(0) = make_scalar<flag_index_t<T>>(1);
    F(1) = make_scalar<flag_index_t<T>>(0);
    F(2) = make_scalar<flag_index_t<T>>(0);
    F(3) = make_scalar<flag_index_t<T>>(0);

    F = detail::select_vec(F, detail::flag4<flag_index_t<T>>(1, 1, 1, 0), p0);
    F = detail::select_vec(F, detail::flag4<flag_index_t<T>>(1, 0, 1, 1), !p0 & p1);
    F = detail::select_vec(F, detail::flag4<flag_index_t<T>>(1, 1, 0, 1), detail::pred_and(!p0, !p1, p2));
    F = detail::select_vec(F, detail::flag4<flag_index_t<T>>(1, 1, 0, 0), detail::pred_and(not_pe, q0));
    F = detail::select_vec(F, detail::flag4<flag_index_t<T>>(1, 0, 1, 0), detail::pred_and(not_pe, !q0, q1));
    F = detail::select_vec(F, detail::flag4<flag_index_t<T>>(1, 0, 0, 1), detail::pred_and(not_pe, !q0, !q1, q2));
    F = detail::select_vec(F, detail::flag4<flag_index_t<T>>(1, 1, 1, 1), detail::pred_and(not_pe, !q0, !q1, !q2));
    return F;
}

namespace detail
{
/// The reference's `update_ee_near_parallel_candidate`: try one typed point-edge
/// candidate and keep it when it is the closest so far (the first candidate is
/// always taken).
///
/// `is_initial` is a *host* constant (each of the reference's four calls passes a
/// literal), so `if(is_initial)` is host control flow on host data — allowed by
/// RULE M2.  The `d < minD` update itself is a device predicate and is predicated.
template <typename T>
inline void update_ee_near_parallel_candidate(const LVec<T, 3>&      p,
                                              const LVec<T, 3>&      s0,
                                              const LVec<T, 3>&      s1,
                                              math::storage_of_t<T>& minD,
                                              LVec<flag_index_t<T>, 4>&     F,
                                              const LVec<flag_index_t<T>, 4>& F_pe,
                                              const LVec<flag_index_t<T>, 4>& F_pp0,
                                              const LVec<flag_index_t<T>, 4>& F_pp1,
                                              bool                   is_initial) noexcept
{
    auto pe_flag = point_edge_distance_flag<T>(p, s0, s1);

    // if(pe[1] && pe[2]) d = PE(...) else if(pe[1]) d = PP(p, s0) else d = PP(p, s1)
    T d_pe, d_pp0, d_pp1;
    point_edge_distance2(p, s0, s1, d_pe);
    point_point_distance2(p, s0, d_pp0);
    point_point_distance2(p, s1, d_pp1);

    auto is_pe = (pe_flag(1) != 0) & (pe_flag(2) != 0);
    auto d     = select_scalar(select_scalar(d_pp1, d_pp0, pe_flag(1) != 0), d_pe, is_pe);
    auto F_d   = select_vec(select_vec(F_pp1, F_pp0, pe_flag(1) != 0), F_pe, is_pe);

    if(is_initial)
    {
        minD = d;
        F    = F_d;
    }
    else
    {
        auto take = d < minD;
        minD      = select_scalar(minD, d, take);
        F         = select_vec(F, F_d, take);
    }
}
}  // namespace detail

template <typename T>
[[nodiscard]] inline FlagVec<T, 4> edge_edge_distance_flag(const LVec<T, 3>& ea0,
                                                           const LVec<T, 3>& ea1,
                                                           const LVec<T, 3>& eb0,
                                                           const LVec<T, 3>& eb1) noexcept
{
    FlagVec<T, 4> F;  // default EE: {1,1,1,1}
    for(int i = 0; i < 4; ++i)
        F(i) = make_scalar<flag_index_t<T>>(1);

    // `const`, not `constexpr`: a DSL scalar is not a literal type (the reference
    // spells this `constexpr T` because it is only ever instantiated with `Float`).
    const T kEeParallelRelTol = static_cast<T>(1e-12);

    auto u = ea1 - ea0;
    auto v = eb1 - eb0;
    auto w = ea0 - eb0;

    T a = math::squaredNorm(u);
    T b = math::dot(u, v);
    T c = math::squaredNorm(v);
    T d = math::dot(u, w);
    T e = math::dot(v, w);
    T D = sub(mul(a, c), mul(b, b));

    auto uxv2          = math::squaredNorm(math::cross(u, v));
    auto near_parallel = uxv2 <= mul(mul(kEeParallelRelTol, a), c);

    // -----------------------------------------------------------------------
    //  near-parallel / collinear: typed PP/PE candidates, closest one wins
    //  (the reference *returns* here; the port evaluates both paths and selects)
    // -----------------------------------------------------------------------
    math::storage_of_t<T> minD = zero_of<T>();
    FlagVec<T, 4>         F_np = F;
    detail::update_ee_near_parallel_candidate<T>(ea0,
                                                 eb0,
                                                 eb1,
                                                 minD,
                                                 F_np,
                                                 detail::flag4<flag_index_t<T>>(1, 0, 1, 1),
                                                 detail::flag4<flag_index_t<T>>(1, 0, 1, 0),
                                                 detail::flag4<flag_index_t<T>>(1, 0, 0, 1),
                                                 /*is_initial=*/true);
    detail::update_ee_near_parallel_candidate<T>(ea1,
                                                 eb0,
                                                 eb1,
                                                 minD,
                                                 F_np,
                                                 detail::flag4<flag_index_t<T>>(0, 1, 1, 1),
                                                 detail::flag4<flag_index_t<T>>(0, 1, 1, 0),
                                                 detail::flag4<flag_index_t<T>>(0, 1, 0, 1),
                                                 /*is_initial=*/false);
    detail::update_ee_near_parallel_candidate<T>(eb0,
                                                 ea0,
                                                 ea1,
                                                 minD,
                                                 F_np,
                                                 detail::flag4<flag_index_t<T>>(1, 1, 1, 0),
                                                 detail::flag4<flag_index_t<T>>(1, 0, 1, 0),
                                                 detail::flag4<flag_index_t<T>>(0, 1, 1, 0),
                                                 /*is_initial=*/false);
    detail::update_ee_near_parallel_candidate<T>(eb1,
                                                 ea0,
                                                 ea1,
                                                 minD,
                                                 F_np,
                                                 detail::flag4<flag_index_t<T>>(1, 1, 0, 1),
                                                 detail::flag4<flag_index_t<T>>(1, 0, 0, 1),
                                                 detail::flag4<flag_index_t<T>>(0, 1, 0, 1),
                                                 /*is_initial=*/false);

    // -----------------------------------------------------------------------
    //  general case: the reference's closest-point line parameters
    // -----------------------------------------------------------------------
    FlagVec<T, 4> F_line = F;

    auto sN     = sub(mul(b, e), mul(c, d));
    auto pA     = sN <= zero_of<T>();  // sc < 0: the s = 0 edge is visible
    auto pB     = !pA & (sN >= D);     // sc > 1: the s = 1 edge is visible
    auto pC     = !pA & !pB;
    auto tN_c   = sub(mul(a, e), mul(b, d));  // the `else` branch's tN
    auto coplanar = math::dot(math::cross(u, v), w) == zero_of<T>();

    auto pC1  = detail::pred_and(pC, tN_c > zero_of<T>(), tN_c < D, coplanar);
    auto pC1a = detail::pred_and(pC1, sN < div(D, 2.0));
    auto pC1b = detail::pred_and(pC1, !pC1a);

    auto e_plus_b = add(e, b);

    // tN / tD: materialised, since both are device values
    math::storage_of_t<T> tN = tN_c;  // the `else` branch keeps tN = a*e - b*d
    tN                       = detail::select_scalar(tN, e, pC1a);
    tN                       = detail::select_scalar(tN, e_plus_b, pC1b);
    tN                       = detail::select_scalar(tN, e, pA);
    tN                       = detail::select_scalar(tN, e_plus_b, pB);

    // every branch that assigns `tD = c`
    auto tD_reset = detail::pred_or(pA, pB, pC1a, pC1b);

    // `detail::select_scalar` picks (false, true, predicate); tD keeps `D` when the
    // line branch neither resets it nor is the `else` case
    auto tD = detail::select_scalar(D, c, tD_reset);

    F_line = detail::select_vec(F_line, detail::flag4<flag_index_t<T>>(1, 0, 1, 1), detail::pred_or(pA, pC1a));
    F_line = detail::select_vec(F_line, detail::flag4<flag_index_t<T>>(0, 1, 1, 1), detail::pred_or(pB, pC1b));

    // tc < 0 => the t = 0 edge is visible; tc > 1 => the t = 1 edge
    auto neg_d        = neg(d);
    auto neg_d_plus_b = add(neg_d, b);

    auto pLow  = tN <= zero_of<T>();
    auto pHigh = !pLow & (tN >= tD);

    auto low0  = pLow & (neg_d <= zero_of<T>());
    auto low1  = detail::pred_and(pLow, !low0, neg_d >= a);
    auto low2  = detail::pred_and(pLow, !low0, !low1);
    auto high0 = pHigh & (neg_d_plus_b <= zero_of<T>());
    auto high1 = detail::pred_and(pHigh, !high0, neg_d_plus_b >= a);
    auto high2 = detail::pred_and(pHigh, !high0, !high1);

    F_line = detail::select_vec(F_line, detail::flag4<flag_index_t<T>>(1, 0, 1, 0), low0);
    F_line = detail::select_vec(F_line, detail::flag4<flag_index_t<T>>(0, 1, 1, 0), low1);
    F_line = detail::select_vec(F_line, detail::flag4<flag_index_t<T>>(1, 1, 1, 0), low2);
    F_line = detail::select_vec(F_line, detail::flag4<flag_index_t<T>>(1, 0, 0, 1), high0);
    F_line = detail::select_vec(F_line, detail::flag4<flag_index_t<T>>(0, 1, 0, 1), high1);
    F_line = detail::select_vec(F_line, detail::flag4<flag_index_t<T>>(1, 1, 0, 1), high2);

    // near_parallel ? F_np : F_line  (the reference *returns* in the near-parallel
    // branch; `select_vec(f, t, p)` is `p ? t : f` — cookbook §7.6 trap #1).
    return detail::select_vec(F_line, F_np, near_parallel);
}

// ---------------------------------------------------------------------------
//  flag-dispatched squared distance
// ---------------------------------------------------------------------------
template <typename T>
void point_point_distance2(const FlagVec<T, 2>& flag,
                           const LVec<T, 3>&    a,
                           const LVec<T, 3>&    b,
                           T&                   D)
{
    (void)flag;  // a point pair is always fully active
    point_point_distance2(a, b, D);
}

template <typename T>
void point_edge_distance2(const FlagVec<T, 3>& flag,
                          const LVec<T, 3>&    p,
                          const LVec<T, 3>&    e0,
                          const LVec<T, 3>&    e1,
                          T&                   D)
{
    const LVec<T, 3>* P[3] = {&p, &e0, &e1};

    auto                  dim = detail::active_count(flag);
    math::storage_of_t<T> out = zero_of<T>();

    // dim == 2: point-point on the ascending active pair (`pp_from_pe`)
    for(int i = 0; i < 3; ++i)
    {
        for(int j = i + 1; j < 3; ++j)
        {
            T d;
            point_point_distance2(*P[i], *P[j], d);
            out = detail::select_scalar(out, d, detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0));
        }
    }

    // dim == 3: the reference's `{0,1,2}` (all three active)
    {
        T d;
        point_edge_distance2(p, e0, e1, d);
        out = detail::select_scalar(out, d, dim == 3);
    }

    D = out;
}

template <typename T>
void point_triangle_distance2(const FlagVec<T, 4>& flag,
                              const LVec<T, 3>&    p,
                              const LVec<T, 3>&    t0,
                              const LVec<T, 3>&    t1,
                              const LVec<T, 3>&    t2,
                              T&                   D)
{
    const LVec<T, 3>* P[4] = {&p, &t0, &t1, &t2};

    auto                  dim = detail::active_count(flag);
    math::storage_of_t<T> out = zero_of<T>();

    // dim == 2: point-point on the ascending active pair (`pp_from_pt`)
    for(int i = 0; i < 4; ++i)
    {
        for(int j = i + 1; j < 4; ++j)
        {
            T d;
            point_point_distance2(*P[i], *P[j], d);
            out = detail::select_scalar(out, d, detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0));
        }
    }

    // dim == 3: point-edge on the `pe_from_pt` permutation
    for(int k = 0; k < 4; ++k)
    {
        const int* o = detail::kPeFromPt[k];
        T          d;
        point_edge_distance2(*P[o[0]], *P[o[1]], *P[o[2]], d);
        out = detail::select_scalar(out, d, (dim == 3) & (flag(k) == 0));
    }

    // dim == 4: the full simplex
    {
        T d;
        point_triangle_distance2(p, t0, t1, t2, d);
        out = detail::select_scalar(out, d, dim == 4);
    }

    D = out;
}

template <typename T>
void edge_edge_distance2(const FlagVec<T, 4>& flag,
                         const LVec<T, 3>&    ea0,
                         const LVec<T, 3>&    ea1,
                         const LVec<T, 3>&    eb0,
                         const LVec<T, 3>&    eb1,
                         T&                   D)
{
    const LVec<T, 3>* P[4] = {&ea0, &ea1, &eb0, &eb1};

    auto                  dim = detail::active_count(flag);
    math::storage_of_t<T> out = zero_of<T>();

    // dim == 2: point-point on the ascending active pair (`pp_from_ee`)
    for(int i = 0; i < 4; ++i)
    {
        for(int j = i + 1; j < 4; ++j)
        {
            T d;
            point_point_distance2(*P[i], *P[j], d);
            out = detail::select_scalar(out, d, detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0));
        }
    }

    // dim == 3: point-edge on the `pe_from_ee` permutation
    for(int k = 0; k < 4; ++k)
    {
        const int* o = detail::kPeFromEe[k];
        T          d;
        point_edge_distance2(*P[o[0]], *P[o[1]], *P[o[2]], d);
        out = detail::select_scalar(out, d, (dim == 3) & (flag(k) == 0));
    }

    // dim == 4: the full simplex
    {
        T d;
        edge_edge_distance2(ea0, ea1, eb0, eb1, d);
        out = detail::select_scalar(out, d, dim == 4);
    }

    D = out;
}

// ---------------------------------------------------------------------------
//  flag-dispatched gradient
// ---------------------------------------------------------------------------
template <typename T>
void point_point_distance2_gradient(const FlagVec<T, 2>& flag,
                                    const LVec<T, 3>&    a,
                                    const LVec<T, 3>&    b,
                                    LVec<T, 6>&          G)
{
    (void)flag;  // a point pair is always fully active
    point_point_distance2_gradient(a, b, G);
}

template <typename T>
void point_edge_distance2_gradient(const FlagVec<T, 3>& flag,
                                   const LVec<T, 3>&    p,
                                   const LVec<T, 3>&    e0,
                                   const LVec<T, 3>&    e1,
                                   LVec<T, 9>&          G)
{
    const LVec<T, 3>* P[3] = {&p, &e0, &e1};

    auto       dim = detail::active_count(flag);
    LVec<T, 9> out;
    out.setZero();

    // dim == 2: PP gradient of the active pair, scattered to its two blocks
    for(int i = 0; i < 3; ++i)
    {
        for(int j = i + 1; j < 3; ++j)
        {
            auto       pred = detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0);
            LVec<T, 6> g6;
            point_point_distance2_gradient(*P[i], *P[j], g6);
            for(int r = 0; r < 3; ++r)
            {
                out(3 * i + r) = detail::select_scalar(out(3 * i + r), g6(r), pred);
                out(3 * j + r) = detail::select_scalar(out(3 * j + r), g6(3 + r), pred);
            }
        }
    }

    // dim == 3: identity layout
    {
        auto       pred = dim == 3;
        LVec<T, 9> g9;
        point_edge_distance2_gradient(p, e0, e1, g9);
        for(int r = 0; r < 9; ++r)
            out(r) = detail::select_scalar(out(r), g9(r), pred);
    }

    G = out;
}

template <typename T>
void point_triangle_distance2_gradient(const FlagVec<T, 4>& flag,
                                       const LVec<T, 3>&    p,
                                       const LVec<T, 3>&    t0,
                                       const LVec<T, 3>&    t1,
                                       const LVec<T, 3>&    t2,
                                       LVec<T, 12>&         G)
{
    const LVec<T, 3>* P[4] = {&p, &t0, &t1, &t2};

    auto        dim = detail::active_count(flag);
    LVec<T, 12> out;
    out.setZero();

    // dim == 2
    for(int i = 0; i < 4; ++i)
    {
        for(int j = i + 1; j < 4; ++j)
        {
            auto       pred = detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0);
            LVec<T, 6> g6;
            point_point_distance2_gradient(*P[i], *P[j], g6);
            for(int r = 0; r < 3; ++r)
            {
                out(3 * i + r) = detail::select_scalar(out(3 * i + r), g6(r), pred);
                out(3 * j + r) = detail::select_scalar(out(3 * j + r), g6(3 + r), pred);
            }
        }
    }

    // dim == 3: the `pe_from_pt` permutation
    for(int k = 0; k < 4; ++k)
    {
        auto       pred = (dim == 3) & (flag(k) == 0);
        const int* o    = detail::kPeFromPt[k];
        LVec<T, 9> g9;
        point_edge_distance2_gradient(*P[o[0]], *P[o[1]], *P[o[2]], g9);
        for(int s = 0; s < 3; ++s)
            for(int r = 0; r < 3; ++r)
                out(3 * o[s] + r) = detail::select_scalar(out(3 * o[s] + r), g9(3 * s + r), pred);
    }

    // dim == 4
    {
        auto        pred = dim == 4;
        LVec<T, 12> g12;
        point_triangle_distance2_gradient(p, t0, t1, t2, g12);
        for(int r = 0; r < 12; ++r)
            out(r) = detail::select_scalar(out(r), g12(r), pred);
    }

    G = out;
}

template <typename T>
void edge_edge_distance2_gradient(const FlagVec<T, 4>& flag,
                                  const LVec<T, 3>&    ea0,
                                  const LVec<T, 3>&    ea1,
                                  const LVec<T, 3>&    eb0,
                                  const LVec<T, 3>&    eb1,
                                  LVec<T, 12>&         G)
{
    const LVec<T, 3>* P[4] = {&ea0, &ea1, &eb0, &eb1};

    auto        dim = detail::active_count(flag);
    LVec<T, 12> out;
    out.setZero();

    // dim == 2
    for(int i = 0; i < 4; ++i)
    {
        for(int j = i + 1; j < 4; ++j)
        {
            auto       pred = detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0);
            LVec<T, 6> g6;
            point_point_distance2_gradient(*P[i], *P[j], g6);
            for(int r = 0; r < 3; ++r)
            {
                out(3 * i + r) = detail::select_scalar(out(3 * i + r), g6(r), pred);
                out(3 * j + r) = detail::select_scalar(out(3 * j + r), g6(3 + r), pred);
            }
        }
    }

    // dim == 3: the `pe_from_ee` permutation
    for(int k = 0; k < 4; ++k)
    {
        auto       pred = (dim == 3) & (flag(k) == 0);
        const int* o    = detail::kPeFromEe[k];
        LVec<T, 9> g9;
        point_edge_distance2_gradient(*P[o[0]], *P[o[1]], *P[o[2]], g9);
        for(int s = 0; s < 3; ++s)
            for(int r = 0; r < 3; ++r)
                out(3 * o[s] + r) = detail::select_scalar(out(3 * o[s] + r), g9(3 * s + r), pred);
    }

    // dim == 4
    {
        auto        pred = dim == 4;
        LVec<T, 12> g12;
        edge_edge_distance2_gradient(ea0, ea1, eb0, eb1, g12);
        for(int r = 0; r < 12; ++r)
            out(r) = detail::select_scalar(out(r), g12(r), pred);
    }

    G = out;
}

// ---------------------------------------------------------------------------
//  flag-dispatched Hessian
// ---------------------------------------------------------------------------
template <typename T>
void point_point_distance2_hessian(const FlagVec<T, 2>& flag,
                                   const LVec<T, 3>&    a,
                                   const LVec<T, 3>&    b,
                                   LMat<T, 6, 6>&       H)
{
    (void)flag;  // a point pair is always fully active
    point_point_distance2_hessian(a, b, H);
}

template <typename T>
void point_edge_distance2_hessian(const FlagVec<T, 3>& flag,
                                  const LVec<T, 3>&    p,
                                  const LVec<T, 3>&    e0,
                                  const LVec<T, 3>&    e1,
                                  LMat<T, 9, 9>&       H)
{
    const LVec<T, 3>* P[3] = {&p, &e0, &e1};

    auto          dim = detail::active_count(flag);
    LMat<T, 9, 9> out;
    out.setZero();

    // dim == 2: PP Hessian of the active pair, its four 3x3 blocks scattered
    for(int i = 0; i < 3; ++i)
    {
        for(int j = i + 1; j < 3; ++j)
        {
            auto          pred = detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0);
            LMat<T, 6, 6> h6;
            point_point_distance2_hessian(*P[i], *P[j], h6);
            const int     idx[2] = {i, j};
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int r = 0; r < 3; ++r)
                        for(int c = 0; c < 3; ++c)
                            out(3 * idx[a] + r, 3 * idx[b] + c) =
                                detail::select_scalar(out(3 * idx[a] + r, 3 * idx[b] + c),
                                                      h6(3 * a + r, 3 * b + c),
                                                      pred);
        }
    }

    // dim == 3: identity layout
    {
        auto          pred = dim == 3;
        LMat<T, 9, 9> h9;
        point_edge_distance2_hessian(p, e0, e1, h9);
        for(int r = 0; r < 9; ++r)
            for(int c = 0; c < 9; ++c)
                out(r, c) = detail::select_scalar(out(r, c), h9(r, c), pred);
    }

    H = out;
}

template <typename T>
void point_triangle_distance2_hessian(const FlagVec<T, 4>& flag,
                                      const LVec<T, 3>&    p,
                                      const LVec<T, 3>&    t0,
                                      const LVec<T, 3>&    t1,
                                      const LVec<T, 3>&    t2,
                                      LMat<T, 12, 12>&     H)
{
    const LVec<T, 3>* P[4] = {&p, &t0, &t1, &t2};

    auto            dim = detail::active_count(flag);
    LMat<T, 12, 12> out;
    out.setZero();

    // dim == 2
    for(int i = 0; i < 4; ++i)
    {
        for(int j = i + 1; j < 4; ++j)
        {
            auto          pred = detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0);
            LMat<T, 6, 6> h6;
            point_point_distance2_hessian(*P[i], *P[j], h6);
            const int     idx[2] = {i, j};
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int r = 0; r < 3; ++r)
                        for(int c = 0; c < 3; ++c)
                            out(3 * idx[a] + r, 3 * idx[b] + c) =
                                detail::select_scalar(out(3 * idx[a] + r, 3 * idx[b] + c),
                                                      h6(3 * a + r, 3 * b + c),
                                                      pred);
        }
    }

    // dim == 3: the `pe_from_pt` permutation
    for(int k = 0; k < 4; ++k)
    {
        auto          pred = (dim == 3) & (flag(k) == 0);
        const int*    o    = detail::kPeFromPt[k];
        LMat<T, 9, 9> h9;
        point_edge_distance2_hessian(*P[o[0]], *P[o[1]], *P[o[2]], h9);
        for(int s = 0; s < 3; ++s)
            for(int t = 0; t < 3; ++t)
                for(int r = 0; r < 3; ++r)
                    for(int c = 0; c < 3; ++c)
                        out(3 * o[s] + r, 3 * o[t] + c) =
                            detail::select_scalar(out(3 * o[s] + r, 3 * o[t] + c),
                                                  h9(3 * s + r, 3 * t + c),
                                                  pred);
    }

    // dim == 4
    {
        auto            pred = dim == 4;
        LMat<T, 12, 12> h12;
        point_triangle_distance2_hessian(p, t0, t1, t2, h12);
        for(int r = 0; r < 12; ++r)
            for(int c = 0; c < 12; ++c)
                out(r, c) = detail::select_scalar(out(r, c), h12(r, c), pred);
    }

    H = out;
}

template <typename T>
void edge_edge_distance2_hessian(const FlagVec<T, 4>& flag,
                                 const LVec<T, 3>&    ea0,
                                 const LVec<T, 3>&    ea1,
                                 const LVec<T, 3>&    eb0,
                                 const LVec<T, 3>&    eb1,
                                 LMat<T, 12, 12>&     H)
{
    const LVec<T, 3>* P[4] = {&ea0, &ea1, &eb0, &eb1};

    auto            dim = detail::active_count(flag);
    LMat<T, 12, 12> out;
    out.setZero();

    // dim == 2
    for(int i = 0; i < 4; ++i)
    {
        for(int j = i + 1; j < 4; ++j)
        {
            auto          pred = detail::pred_and(dim == 2, flag(i) != 0, flag(j) != 0);
            LMat<T, 6, 6> h6;
            point_point_distance2_hessian(*P[i], *P[j], h6);
            const int     idx[2] = {i, j};
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                    for(int r = 0; r < 3; ++r)
                        for(int c = 0; c < 3; ++c)
                            out(3 * idx[a] + r, 3 * idx[b] + c) =
                                detail::select_scalar(out(3 * idx[a] + r, 3 * idx[b] + c),
                                                      h6(3 * a + r, 3 * b + c),
                                                      pred);
        }
    }

    // dim == 3: the `pe_from_ee` permutation
    for(int k = 0; k < 4; ++k)
    {
        auto          pred = (dim == 3) & (flag(k) == 0);
        const int*    o    = detail::kPeFromEe[k];
        LMat<T, 9, 9> h9;
        point_edge_distance2_hessian(*P[o[0]], *P[o[1]], *P[o[2]], h9);
        for(int s = 0; s < 3; ++s)
            for(int t = 0; t < 3; ++t)
                for(int r = 0; r < 3; ++r)
                    for(int c = 0; c < 3; ++c)
                        out(3 * o[s] + r, 3 * o[t] + c) =
                            detail::select_scalar(out(3 * o[s] + r, 3 * o[t] + c),
                                                  h9(3 * s + r, 3 * t + c),
                                                  pred);
    }

    // dim == 4
    {
        auto            pred = dim == 4;
        LMat<T, 12, 12> h12;
        edge_edge_distance2_hessian(ea0, ea1, eb0, eb1, h12);
        for(int r = 0; r < 12; ++r)
            for(int c = 0; c < 12; ++c)
                out(r, c) = detail::select_scalar(out(r, c), h12(r, c), pred);
    }

    H = out;
}
}  // namespace uipc::backend::luisa::luisa_device::math::distance
