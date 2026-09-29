#pragma once
/********************************************************************
 * @file affine_body/constraints/external_articulation_constraint_function.h
 * @brief T6.4b — the trapezoid of arithmetic the *external articulation*
 * constraint (constitution uid 23 / constraint uid 24) shares between its three
 * kernels, the time integrator and the unit test (port of
 * src/backends/cuda/affine_body/constraints/external_articulation_constraint_function.h).
 *
 * The reference header is a 15-line include shell around the two SymEigen.py-generated
 * bodies; the port keeps that shape and adds the *factored* per-joint routines the
 * kernels and the unit test share (the T6.3a `abd_arap_detail` / T6.3b1
 * `abd_fixed_joint_detail` precedent): one implementation over the scalar type `S`
 * (`double` on the host, `lcrt::Var<double>` in the DSL), never a re-spelling of the
 * generated statements.
 *
 * THE TWO GENERATED BODIES AND THE ONE THING THAT DIFFERS.  The reference generates
 * `F`/`DeltaTheta`/`dDeltaTheta_dQ`/`JT_G`/`JT_H_J`/`DeltaTheta#2`/`dDeltaTheta_dF`/
 * `ddDeltaTheta_ddF` twice — once per joint uid the articulation may reference — and
 * the port transcribes both files verbatim (see their own banners for the laws and the
 * independent-oracle residuals).  The two laws are NOT the same:
 *   revolute (uid 18): F is *direction-only* ([A n ; A b] pairs) and `DeltaTheta` is
 *                      the **wrapped** `atan2` angle difference;
 *   prismatic (uid 20): F carries the two attachment *points* (`+ qk(0..2)`) and
 *                      `DeltaTheta` is the **linear** signed projected half-distance.
 *
 * THE UID DISPATCH IS A SELECT, NOT A BRANCH (RULE M2).  The reference's kernels all
 * spell the dispatch
 *
 *     if(joint_uid == RevoluteUID)      ERJ::...;
 *     else if(joint_uid == PrismaticUID) EPJ::...;
 *     else { UIPC_KERNEL_ASSERT(false, "..."); <zero>; }
 *
 * and `joint_uid` is *device* data (`joint_id_to_uid`, read per joint).  The port
 * therefore evaluates **both** candidates and selects, and maps the else-branch's
 * `UIPC_KERNEL_ASSERT(false, ...)` + zero onto (a) the `LC_LOG` kernel-logging facade
 * (a no-op unless `UIPC_LUISA_KERNEL_PRINT`, cookbook §7.15 — the `torque_to_F`
 * precedent in `affine_body/utils.h`) and (b) a select of zero when neither uid
 * matches.  `select_if(p, a, b)` is `p ? a : b` (this file's own helper, because the
 * underlying `scalar_detail::select_container(f, t, p)` means `p ? t : f` and the
 * §7.26 rule-13 polarity bug is a whole-block, silently-plausible defect).
 *
 * THE `make_spd` CLAMP IS REAL HERE.  The reference's Hessian kernel clamps
 * `G_theta_i * ddDeltaTheta_ddF` (both bodies' element tensors are non-diagonal — the
 * oracle's shape flags: revolute off-diagonal fraction 1.000e+00, prismatic diagonal
 * fraction 4.4e-08) with the tolerance-driven `make_spd`.  The port uses the fixed
 * sweep count `kMakeSpdSweeps = 8` (§7.17 rule 31's measured plateau for N = 12 in this
 * layer; RULE M2), and the test perturbs it because the clamp is observable.
 *
 * THE SEGMENTED REDUCE.  `do_compute_gradient_hessian` starts with the reference's
 * `FastSegmentalReduce().reduce(n, joint_id_to_G_theta, get_key, get_value)` whose key
 * is `joint_joint_id_to_joint_ij(I)(0)` and whose value is
 * `m_ij * (delta_theta_j - delta_theta_tilde_j)` (the `j`-side joint's current
 * `DeltaTheta`), i.e.
 *     G_theta[i] = sum_j m_ij (DeltaTheta_j - delta_theta_tilde_j) .
 * The port uses `lla::segmented_reduce` (T3.2's transcription of the same primitive),
 * which takes the keys from a *buffer*: `joint_joint_id_to_key` is materialised
 * host-side in `do_init` (its entries are static — the joint layout never changes) and
 * uploaded once.  The key sequence is ascending by construction: within one articulation
 * the entries are `i`-slow/`j`-fast with `j` shifted by the articulation's joint offset,
 * and the articulations' offsets are a prefix scan of their counts.
 ********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/math.h>
#include <luisa_device/linear/linear.h>  // lla::L*DeviceView (the two write helpers below)
// `lps::make_spd_inplace` lives in matrix_ops.h; math.h alone does not declare it (the
// T6.3a C2039 finding, re-applied).
#include <luisa_device/math/matrix_ops.h>
// `scalar_detail::select_scalar` / `select_container` — the device-predicate selects
// the reference's uid dispatch and its `if(ij[0] == ij[1])` extra term need.
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
// The reference's own namespace names (`sym::external_revolute_joint_constraint` /
// `sym::external_prismatic_joint_constraint`), so the .cu's call sites read the same.
namespace sym::external_revolute_joint_constraint
{
#include <affine_body/constraints/sym/external_articulation_revolute_joint_constraint.inl>
}  // namespace sym::external_revolute_joint_constraint

namespace sym::external_prismatic_joint_constraint
{
#include <affine_body/constraints/sym/external_articulation_prismatic_joint_constraint.inl>
}  // namespace sym::external_prismatic_joint_constraint

namespace abd_external_articulation_detail
{
/// The reference's `make_spd` is a tolerance-driven Jacobi loop (a *device* loop count,
/// RULE M2), so the port picks a fixed one: 8 is §7.17 rule 31's measured plateau for
/// N = 12 in this layer.  Measured *observable* here (the element tensors are
/// non-diagonal — see the banner), which is why the test pins it.
inline constexpr int kMakeSpdSweeps = 8;

/// `p ? a : b` for a device predicate, over *both* scalars and containers.
///
/// Spelled once, with this polarity, because `scalar_detail::select_container(f, t, p)`
/// means `p ? t : f` (the §7.26 rule-13 trap) and because the layer has no single
/// `select` that accepts a scalar and a container alike:
/// `scalar_detail::select_container` is declared for `LVec`/`LMat` only, and
/// `select_scalar` for scalars (it resolves host vs DSL itself).
template <typename P, typename T>
[[nodiscard]] inline T select_if(const P& flag, const T& if_true, const T& if_false) noexcept
{
    if constexpr(lps::is_dsl_value_v<T>)
    {
        return scalar_detail::select_scalar(if_false, if_true, flag);
    }
    else
    {
        return scalar_detail::select_container(if_false, if_true, flag);
    }
}

/// `ERJ::F(basis_k, qk, basis_l, ql)` as a value (revolute: two world directions).
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> revolute_f(const lps::LVec<S, 6>& basis_k,
                                                 const lps::LVec<S, 12>& qk,
                                                 const lps::LVec<S, 6>& basis_l,
                                                 const lps::LVec<S, 12>& ql) noexcept
{
    lps::LVec<S, 12> out;
    sym::external_revolute_joint_constraint::F(out, basis_k, qk, basis_l, ql);
    return out;
}

/// `EPJ::F(basis_k, qk, basis_l, ql)` as a value (prismatic: points + directions).
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> prismatic_f(const lps::LVec<S, 6>& basis_k,
                                                  const lps::LVec<S, 12>& qk,
                                                  const lps::LVec<S, 6>& basis_l,
                                                  const lps::LVec<S, 12>& ql) noexcept
{
    lps::LVec<S, 12> out;
    sym::external_prismatic_joint_constraint::F(out, basis_k, qk, basis_l, ql);
    return out;
}

/// `ERJ::DeltaTheta<Float>(delta_theta, basis_k, qk, qk_t, basis_l, ql, ql_t)`.
template <typename S>
[[nodiscard]] inline S revolute_delta_theta(const lps::LVec<S, 6>& basis_k,
                                            const lps::LVec<S, 12>& qk,
                                            const lps::LVec<S, 12>& qk_t,
                                            const lps::LVec<S, 6>& basis_l,
                                            const lps::LVec<S, 12>& ql,
                                            const lps::LVec<S, 12>& ql_t) noexcept
{
    S out{};
    sym::external_revolute_joint_constraint::DeltaTheta(out, basis_k, qk, qk_t, basis_l, ql, ql_t);
    return out;
}

/// `EPJ::DeltaTheta<Float>(delta_theta, basis_k, qk, qk_t, basis_l, ql, ql_t)`.
template <typename S>
[[nodiscard]] inline S prismatic_delta_theta(const lps::LVec<S, 6>& basis_k,
                                             const lps::LVec<S, 12>& qk,
                                             const lps::LVec<S, 12>& qk_t,
                                             const lps::LVec<S, 6>& basis_l,
                                             const lps::LVec<S, 12>& ql,
                                             const lps::LVec<S, 12>& ql_t) noexcept
{
    S out{};
    sym::external_prismatic_joint_constraint::DeltaTheta(out, basis_k, qk, qk_t, basis_l, ql, ql_t);
    return out;
}

/// `ERJ::dDeltaTheta_dQ` (24).
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> revolute_d_delta_theta_dq(const lps::LVec<S, 6>& basis_k,
                                                                const lps::LVec<S, 12>& qk,
                                                                const lps::LVec<S, 12>& qk_t,
                                                                const lps::LVec<S, 6>& basis_l,
                                                                const lps::LVec<S, 12>& ql,
                                                                const lps::LVec<S, 12>& ql_t) noexcept
{
    lps::LVec<S, 24> out;
    sym::external_revolute_joint_constraint::dDeltaTheta_dQ(out, basis_k, qk, qk_t, basis_l, ql, ql_t);
    return out;
}

/// `EPJ::dDeltaTheta_dQ` (24).
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> prismatic_d_delta_theta_dq(const lps::LVec<S, 6>& basis_k,
                                                                 const lps::LVec<S, 12>& qk,
                                                                 const lps::LVec<S, 12>& qk_t,
                                                                 const lps::LVec<S, 6>& basis_l,
                                                                 const lps::LVec<S, 12>& ql,
                                                                 const lps::LVec<S, 12>& ql_t) noexcept
{
    lps::LVec<S, 24> out;
    sym::external_prismatic_joint_constraint::dDeltaTheta_dQ(out, basis_k, qk, qk_t, basis_l, ql, ql_t);
    return out;
}

/// `ERJ::ddDeltaTheta_ddF` (12x12).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> revolute_dd_delta_theta_ddf(
    const lps::LVec<S, 12>& F, const lps::LVec<S, 12>& F_t) noexcept
{
    lps::LMat<S, 12, 12> out;
    sym::external_revolute_joint_constraint::ddDeltaTheta_ddF(out, F, F_t);
    return out;
}

/// `EPJ::ddDeltaTheta_ddF` (12x12).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> prismatic_dd_delta_theta_ddf(
    const lps::LVec<S, 12>& F, const lps::LVec<S, 12>& F_t) noexcept
{
    lps::LMat<S, 12, 12> out;
    sym::external_prismatic_joint_constraint::ddDeltaTheta_ddF(out, F, F_t);
    return out;
}

/// `ERJ::JT_H_J(H_F, basis_k, basis_l, basis_m, basis_n)` (24x24).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> revolute_jt_h_j(const lps::LMat<S, 12, 12>& H_F,
                                                          const lps::LVec<S, 6>& basis_k,
                                                          const lps::LVec<S, 6>& basis_l,
                                                          const lps::LVec<S, 6>& basis_m,
                                                          const lps::LVec<S, 6>& basis_n) noexcept
{
    lps::LMat<S, 24, 24> out;
    sym::external_revolute_joint_constraint::JT_H_J(out, H_F, basis_k, basis_l, basis_m, basis_n);
    return out;
}

/// `EPJ::JT_H_J(H_F, basis_k, basis_l, basis_m, basis_n)` (24x24).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> prismatic_jt_h_j(const lps::LMat<S, 12, 12>& H_F,
                                                           const lps::LVec<S, 6>& basis_k,
                                                           const lps::LVec<S, 6>& basis_l,
                                                           const lps::LVec<S, 6>& basis_m,
                                                           const lps::LVec<S, 6>& basis_n) noexcept
{
    lps::LMat<S, 24, 24> out;
    sym::external_prismatic_joint_constraint::JT_H_J(out, H_F, basis_k, basis_l, basis_m, basis_n);
    return out;
}

/// A zero `LVec<S,N>` (the reference's `G24.setZero()` / `dDeltaTheta_dQ.setZero()`).
template <typename S, int N>
[[nodiscard]] inline lps::LVec<S, N> zero_vec() noexcept
{
    return lps::make_vec<S, N>([](int) noexcept { return lps::zero_of<S>(); });
}

/// A zero `LMat<S,R,C>`.
template <typename S, int R, int C>
[[nodiscard]] inline lps::LMat<S, R, C> zero_mat() noexcept
{
    return lps::make_mat<S, R, C>([](int, int) noexcept { return lps::zero_of<S>(); });
}

// ---------------------------------------------------------------------------
//  the two assembly write patterns (the reference's
//  `DoubletVectorAssembler::segment<2>` / `TripletMatrixAssembler::half_block<2>`)
//
// Re-stated here rather than imported from the joints' headers so that this file's
// include closure stays the reference's own (the T6.3b2 `affine_body_prismatic_joint_
// function.h` precedent).  The reference's own call is
//     DVA.segment<2>(jointI * 2).write(body_ids, G24);
//     TMA.half_block<2>(joint_joint_I * HalfHessianSize).write(body_ids_i, body_ids_j, H24x24);
// ---------------------------------------------------------------------------
/// The reference's `DoubletVectorAssembler::segment<2>(base).write(bids, G24)`: two
/// consecutive doublets carrying the two 12-vectors at the two body ids.
template <typename DoubletView, typename S, typename Base, typename Idx>
inline void write_pair_gradient(const DoubletView&      doublets,
                                const lps::LVec<S, 24>& G,
                                const Base&             base,
                                const lps::LVec<Idx, 2>& indices) noexcept
{
    doublets(base).write(indices(0), lps::LVec<S, 12>(G.template segment<12>(0)));
    doublets(base + 1).write(indices(1), lps::LVec<S, 12>(G.template segment<12>(12)));
}

/// The reference's `TripletMatrixAssembler::half_block<2>(base).write(i, j, H)`: the
/// upper triangle of the 2x2 *block* matrix in the reference's `upper_LR` order —
/// `(0,0)` and `(1,1)` are fixed and the off-diagonal block is written at
/// `(indices(L), indices(R))` with `(L, R) = (0, 1)` iff `indices(0) < indices(1)`,
/// with the *same* rule applied to the block's content (`H.block(L, R)`).  Both are
/// device predicates, so they become selects (RULE M2).
template <typename TripletView, typename S, typename Base, typename Idx>
inline void write_pair_half_hessian(const TripletView&         triplets,
                                    const lps::LMat<S, 24, 24>& H,
                                    const Base&                base,
                                    const lps::LVec<Idx, 2>&   indices) noexcept
{
    const auto block = [&](int L, int R)
    { return lps::LMat<S, 12, 12>(H.template block<12, 12>(L * 12, R * 12)); };

    const auto ascending = indices(0) < indices(1);
    const auto off_row   = scalar_detail::select_scalar(indices(1), indices(0), ascending);
    const auto off_col   = scalar_detail::select_scalar(indices(0), indices(1), ascending);
    const auto off_block = scalar_detail::select_container(block(1, 0), block(0, 1), ascending);

    triplets(base).write(indices(0), indices(0), block(0, 0));
    triplets(base + 1).write(off_row, off_col, off_block);
    triplets(base + 2).write(indices(1), indices(1), block(1, 1));
}
}  // namespace abd_external_articulation_detail
}  // namespace uipc::backend::luisa
