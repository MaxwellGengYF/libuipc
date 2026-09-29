#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_prismatic_joint_function.h
 * @brief T6.3b2 — the trapezoid of arithmetic the prismatic joint, its driving
 *        constraint and its joint-limit constitution share (port of
 *        src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint_function.h).
 *
 * The reference header is a thin include shell around the two SymEigen.py-generated
 * bodies plus the three *hand-written* `compute_absolute_distance*` helpers; the port
 * keeps that shape and adds the *factored* per-joint routines the kernels and the unit
 * test share (the T6.3a `abd_arap_detail` / T6.3b1 `abd_fixed_joint_detail` precedent):
 * one implementation over the scalar type `S` (`double` on the host,
 * `lcrt::Var<double>` in the DSL), never a re-spelling of the generated statements.
 *
 * THE THREE LAWS (verbatim from the reference's `.cu`; the generated bodies' own
 * readings are in the two `sym/*.inl` banners):
 *
 *   joint (uid 20)        kappa = strength_ratio(I) * (m(bids0) + m(bids1))
 *     F01   = [ -Ft ; A_i t_i ; A_j t_j ]
 *     E     = E01 + E23
 *     G     = J01T_G01(dE01dF01(F01)) + dE23dQ
 *     H     = J01T_H01_J01(make_spd(ddE01ddF01(F01))) + ddE23ddQ
 *
 *   driving (uid 21), with `d_tidle = aim_distance - init_distance`
 *     F01_q = [ -Ft ; A_i T_i ; -(A_j T_j) ]
 *     E     = E01(kappa, F01_q, d_tidle)
 *     G     = J01T_G01(dE01dF01(kappa, F01_q, d_tidle))
 *     H     = J01T_H01_J01(make_spd(ddE01ddF01(kappa, F01_q, d_tidle)))
 *
 *   joint limit (uid 669), with `x = compute_absolute_distance` and the bounds
 *   shifted by the joint's `init_distance`
 *     E     = eval_penalty_energy(x - ..., ...)
 *     G     = dE_dx * compute_absolute_distance_derivative
 *     H     = d2E_dx2 * (dx_dq dx_dq^T) [+ compute_absolute_distance_hessian if
 *                                          dE_dx != 0]
 *
 * THE REFERENCE CLAMPS (three times, all *before* the fan-out they belong to):
 *   * the joint's `ddE01ddF01` (9x9) — **not diagonal** (it mixes the `-Ft` and the
 *     two axis blocks), so unlike T6.3b1's joints this clamp is a real computation;
 *   * the driving's `ddE01ddF01` (9x9) — also not diagonal;
 *   * the limit's `dE_dx * ddxdF01` (9x9) — its elemental `ddxdF01` is the *constant*
 *     non-diagonal `[[0,I/2,-I/2],[I/2,0,0],[-I/2,0,0]]`, so this one is not diagonal
 *     either.
 * `utils/make_spd.h` is a tolerance-driven device loop, so the port picks a fixed
 * sweep count (`kMakeSpdSweeps`, RULE M2; §7.17 rule 31's plateau).
 *
 * THE ASSEMBLY WRITE PATTERNS are T6.3b1's: the gradient as the 2-doublet run
 * `DVA.segment<2>(2 * I)` at `{bids(0), bids(1)}`, the Hessian as the upper-triangular
 * 2x2 12x12 block run `TMA.half_block<2>(HalfHessianSize * I)` with the reference's
 * `upper_LR` placement (a *device* predicate -> `select`, RULE M2).  The two helpers
 * are re-stated here rather than imported from the fixed joint's header so that this
 * file's include closure is the reference's own.
 ********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/math.h>
// `abd_jacobi_vec_x` (the external-force reporter's `J.vec_x(q)`), cookbook §7.21
#include <affine_body/abd_jacobi_matrix.h>
// `lps::make_spd_inplace` lives in matrix_ops.h; math.h alone does not declare it
// (measured C2039 in T6.3a).
#include <luisa_device/math/matrix_ops.h>
// `scalar_detail::{select_scalar, select_container}` — the device-predicate selects
// the reference's `upper_LR` needs (the FEM shells' shared helper).
#include <utils/scalar_util.h>
// the joint-limit penalty (the reference includes it from this `.cu`)
#include <affine_body/constitutions/joint_limit_penalty.h>

namespace uipc::backend::luisa
{
namespace sym::affine_body_prismatic_joint
{
#include "sym/affine_body_prismatic_joint.inl"
}  // namespace sym::affine_body_prismatic_joint

namespace sym::affine_body_driving_prismatic_joint
{
#include "sym/affine_body_driving_prismatic_joint.inl"
}  // namespace sym::affine_body_driving_prismatic_joint

// ===========================================================================
//  the prismatic joint (uid 20)
// ===========================================================================
namespace abd_prismatic_joint_detail
{
/// The reference's `make_spd` is a tolerance-driven Jacobi loop (a *device* loop
/// count, RULE M2), so the port picks a fixed one: 8 is §7.17 rule 31's measured
/// plateau for N = 6/9/12 in this layer, and the element Hessian here is 9x9.  The
/// test re-measures it against the layer default (20) **and** perturbs it — this
/// Hessian is *not* diagonal, so the sweep count is observable (unlike T6.3b1's).
inline constexpr int kMakeSpdSweeps = 8;

/// One joint's rest data, unpacked from the reference's four `Vector6` buffers.
template <typename S>
struct PrismaticJointRestT
{
    // clang-format off
    lps::LVec<S, 6> rest_c;  // [ci_bar | cj_bar]  (the two attachment points, body-local)
    lps::LVec<S, 6> rest_t;  // [ti_bar | tj_bar]  (the two axis directions, body-local)
    lps::LVec<S, 6> rest_n;  // [ni_bar | nj_bar]  (the frames' y directions)
    lps::LVec<S, 6> rest_b;  // [bi_bar | bj_bar]  (the frames' z directions)
    // clang-format on
};

/// `F01` (9-vector) — the reference's own argument order.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 9> f01(const PrismaticJointRestT<S>& r,
                                         const lps::LVec<S, 12>&       qi,
                                         const lps::LVec<S, 12>&       qj) noexcept
{
    lps::LVec<S, 9> out;
    sym::affine_body_prismatic_joint::F01<S>(out,
                                             r.rest_c.template segment<3>(0),
                                             r.rest_t.template segment<3>(0),
                                             qi,
                                             r.rest_c.template segment<3>(3),
                                             r.rest_t.template segment<3>(3),
                                             qj);
    return out;
}

/// The reference's `Es(I) = E01 + E23;`.
template <typename S>
[[nodiscard]] inline S body_energy(const S&                     kappa,
                                   const PrismaticJointRestT<S>& r,
                                   const lps::LVec<S, 12>&       qi,
                                   const lps::LVec<S, 12>&       qj) noexcept
{
    const auto F01_val = f01<S>(r, qi, qj);

    S E01_val;
    sym::affine_body_prismatic_joint::E01<S>(E01_val, kappa, F01_val);

    S E23_val;
    sym::affine_body_prismatic_joint::E23<S>(E23_val,
                                             kappa,
                                             r.rest_n.template segment<3>(0),
                                             r.rest_b.template segment<3>(0),
                                             qi,
                                             r.rest_n.template segment<3>(3),
                                             r.rest_b.template segment<3>(3),
                                             qj);

    return lps::detail::add(E01_val, E23_val);
}

/// The reference's `Vector24 G = J01T_G01 + dE23dQ;`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> body_gradient(const S&                     kappa,
                                                    const PrismaticJointRestT<S>& r,
                                                    const lps::LVec<S, 12>&       qi,
                                                    const lps::LVec<S, 12>&       qj) noexcept
{
    const auto F01_val = f01<S>(r, qi, qj);

    lps::LVec<S, 9> dE01dF01_val;
    sym::affine_body_prismatic_joint::dE01dF01<S>(dE01dF01_val, kappa, F01_val);

    lps::LVec<S, 24> J01T_G01_val;
    sym::affine_body_prismatic_joint::J01T_G01<S>(J01T_G01_val,
                                                  dE01dF01_val,
                                                  r.rest_c.template segment<3>(0),
                                                  r.rest_t.template segment<3>(0),
                                                  r.rest_c.template segment<3>(3),
                                                  r.rest_t.template segment<3>(3));

    lps::LVec<S, 24> dE23dQ_val;
    sym::affine_body_prismatic_joint::dE23dQ<S>(dE23dQ_val,
                                                kappa,
                                                r.rest_n.template segment<3>(0),
                                                r.rest_b.template segment<3>(0),
                                                qi,
                                                r.rest_n.template segment<3>(3),
                                                r.rest_b.template segment<3>(3),
                                                qj);

    return J01T_G01_val + dE23dQ_val;
}

/// The reference's `Matrix24x24 H = J01T_H01_J01 + ddE23ddQ;` with `make_spd`
/// applied to the 9x9 element Hessian *before* its fan-out.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> body_hessian(const S&                     kappa,
                                                       const PrismaticJointRestT<S>& r,
                                                       const lps::LVec<S, 12>&       qi,
                                                       const lps::LVec<S, 12>&       qj) noexcept
{
    const auto F01_val = f01<S>(r, qi, qj);

    lps::LMat<S, 9, 9> ddE01ddF01_val;
    sym::affine_body_prismatic_joint::ddE01ddF01<S>(ddE01ddF01_val, kappa, F01_val);
    lps::make_spd_inplace(ddE01ddF01_val, kMakeSpdSweeps);

    lps::LMat<S, 24, 24> J01T_H01_J01_val;
    sym::affine_body_prismatic_joint::J01T_H01_J01<S>(J01T_H01_J01_val,
                                                      ddE01ddF01_val,
                                                      r.rest_c.template segment<3>(0),
                                                      r.rest_t.template segment<3>(0),
                                                      r.rest_c.template segment<3>(3),
                                                      r.rest_t.template segment<3>(3));

    lps::LMat<S, 24, 24> ddE23ddQ_val;
    sym::affine_body_prismatic_joint::ddE23ddQ<S>(ddE23ddQ_val,
                                                  kappa,
                                                  r.rest_n.template segment<3>(0),
                                                  r.rest_b.template segment<3>(0),
                                                  qi,
                                                  r.rest_n.template segment<3>(3),
                                                  r.rest_b.template segment<3>(3),
                                                  qj);

    return J01T_H01_J01_val + ddE23ddQ_val;
}
}  // namespace abd_prismatic_joint_detail

// ===========================================================================
//  the driving prismatic joint (uid 21) + the joint-limit chain
// ===========================================================================
namespace abd_driving_prismatic_joint_detail
{
/// See `abd_prismatic_joint_detail::kMakeSpdSweeps`; the driving joint's element
/// Hessian is another non-diagonal 9x9.
inline constexpr int kMakeSpdSweeps = 8;

/// One driving pair's rest data (`rest_c` = the two attachment points, `rest_t` =
/// the two axis directions), the same `[first | second]` layout as the joint's.
template <typename S>
struct DrivingPrismaticJointRestT
{
    // clang-format off
    lps::LVec<S, 6> rest_c;
    lps::LVec<S, 6> rest_t;
    // clang-format on
};

/// `F01_q` (9-vector) — the reference's own argument order.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 9> f01_q(const DrivingPrismaticJointRestT<S>& r,
                                           const lps::LVec<S, 12>&              qi,
                                           const lps::LVec<S, 12>&              qj) noexcept
{
    lps::LVec<S, 9> out;
    sym::affine_body_driving_prismatic_joint::F01_q<S>(out,
                                                       r.rest_c.template segment<3>(0),
                                                       r.rest_t.template segment<3>(0),
                                                       qi,
                                                       r.rest_c.template segment<3>(3),
                                                       r.rest_t.template segment<3>(3),
                                                       qj);
    return out;
}

/// `compute_absolute_distance(Float& out_distance, const Vector6& C_bar,
/// const Vector6& t_bar, const Vector12& q_i, const Vector12& q_j)` — the
/// reference's first hand-written helper (it takes the *full* `Vector6`s and slices
/// them inside, exactly like the ported spelling below).
template <typename S>
[[nodiscard]] inline S compute_absolute_distance(const lps::LVec<S, 6>& C_bar,
                                                 const lps::LVec<S, 6>& t_bar,
                                                 const lps::LVec<S, 12>& q_i,
                                                 const lps::LVec<S, 12>& q_j) noexcept
{
    lps::LVec<S, 9> F01_q;
    sym::affine_body_driving_prismatic_joint::F01_q<S>(F01_q,
                                                       C_bar.template segment<3>(0),
                                                       t_bar.template segment<3>(0),
                                                       q_i,
                                                       C_bar.template segment<3>(3),
                                                       t_bar.template segment<3>(3),
                                                       q_j);
    S out_distance;
    sym::affine_body_driving_prismatic_joint::Distance<S>(out_distance, F01_q);
    return out_distance;
}

/// `compute_absolute_distance_derivative` — `J01T_G01(dxdF01(F01_q))`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> compute_absolute_distance_derivative(
    const lps::LVec<S, 6>&  C_bar,
    const lps::LVec<S, 6>&  t_bar,
    const lps::LVec<S, 12>& q_i,
    const lps::LVec<S, 12>& q_j) noexcept
{
    lps::LVec<S, 9> F01_q;
    sym::affine_body_driving_prismatic_joint::F01_q<S>(F01_q,
                                                       C_bar.template segment<3>(0),
                                                       t_bar.template segment<3>(0),
                                                       q_i,
                                                       C_bar.template segment<3>(3),
                                                       t_bar.template segment<3>(3),
                                                       q_j);

    lps::LVec<S, 9> dxdF01;
    sym::affine_body_driving_prismatic_joint::dxdF01<S>(dxdF01, F01_q);

    lps::LVec<S, 24> out_dx_dq;
    sym::affine_body_driving_prismatic_joint::J01T_G01<S>(out_dx_dq,
                                                          dxdF01,
                                                          C_bar.template segment<3>(0),
                                                          t_bar.template segment<3>(0),
                                                          C_bar.template segment<3>(3),
                                                          t_bar.template segment<3>(3));
    return out_dx_dq;
}

/// `compute_absolute_distance_hessian` — `out_H += J01T_H01_J01(
/// make_spd(dE_dx * ddxdF01))`; an *accumulate*, exactly as the reference.
template <typename S>
inline void compute_absolute_distance_hessian(lps::LMat<S, 24, 24>&  out_H,
                                              const S&                dE_dx,
                                              const lps::LVec<S, 6>&  C_bar,
                                              const lps::LVec<S, 6>&  t_bar,
                                              const lps::LVec<S, 12>& q_i,
                                              const lps::LVec<S, 12>& q_j) noexcept
{
    lps::LVec<S, 9> F01_q;
    sym::affine_body_driving_prismatic_joint::F01_q<S>(F01_q,
                                                       C_bar.template segment<3>(0),
                                                       t_bar.template segment<3>(0),
                                                       q_i,
                                                       C_bar.template segment<3>(3),
                                                       t_bar.template segment<3>(3),
                                                       q_j);

    lps::LMat<S, 9, 9> ddx_ddF;
    sym::affine_body_driving_prismatic_joint::ddxdF01<S>(ddx_ddF, F01_q);
    lps::LMat<S, 9, 9> H_D = dE_dx * ddx_ddF;
    lps::make_spd_inplace(H_D, kMakeSpdSweeps);

    lps::LMat<S, 24, 24> JT_H_J;
    sym::affine_body_driving_prismatic_joint::J01T_H01_J01<S>(JT_H_J,
                                                              H_D,
                                                              C_bar.template segment<3>(0),
                                                              t_bar.template segment<3>(0),
                                                              C_bar.template segment<3>(3),
                                                              t_bar.template segment<3>(3));

    out_H += JT_H_J;
}

/// The driving energy kernel's own law, `E01(kappa, F01_q, d_tidle)`.
template <typename S>
[[nodiscard]] inline S body_energy(const S&                            kappa,
                                   const DrivingPrismaticJointRestT<S>& r,
                                   const lps::LVec<S, 12>&              qi,
                                   const lps::LVec<S, 12>&              qj,
                                   const S&                             d_tidle) noexcept
{
    S E01;
    sym::affine_body_driving_prismatic_joint::E01<S>(E01, kappa, f01_q<S>(r, qi, qj), d_tidle);
    return E01;
}

/// The gradient's own law, `J01T_G01(dE01dF01(kappa, F01_q, d_tidle))`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> body_gradient(const S&                            kappa,
                                                    const DrivingPrismaticJointRestT<S>& r,
                                                    const lps::LVec<S, 12>&              qi,
                                                    const lps::LVec<S, 12>&              qj,
                                                    const S& d_tidle) noexcept
{
    const auto F01_q = f01_q<S>(r, qi, qj);

    lps::LVec<S, 9> G01;
    sym::affine_body_driving_prismatic_joint::dE01dF01<S>(G01, kappa, F01_q, d_tidle);

    lps::LVec<S, 24> J01T_G01;
    sym::affine_body_driving_prismatic_joint::J01T_G01<S>(J01T_G01,
                                                          G01,
                                                          r.rest_c.template segment<3>(0),
                                                          r.rest_t.template segment<3>(0),
                                                          r.rest_c.template segment<3>(3),
                                                          r.rest_t.template segment<3>(3));
    return J01T_G01;
}

/// The Hessian's own law, `J01T_H01_J01(make_spd(ddE01ddF01(kappa, F01_q, d_tidle)))`.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> body_hessian(const S&                            kappa,
                                                       const DrivingPrismaticJointRestT<S>& r,
                                                       const lps::LVec<S, 12>&              qi,
                                                       const lps::LVec<S, 12>&              qj,
                                                       const S& d_tidle) noexcept
{
    const auto F01_q = f01_q<S>(r, qi, qj);

    lps::LMat<S, 9, 9> H01;
    sym::affine_body_driving_prismatic_joint::ddE01ddF01<S>(H01, kappa, F01_q, d_tidle);
    lps::make_spd_inplace(H01, kMakeSpdSweeps);

    lps::LMat<S, 24, 24> J01T_H01_J01;
    sym::affine_body_driving_prismatic_joint::J01T_H01_J01<S>(J01T_H01_J01,
                                                              H01,
                                                              r.rest_c.template segment<3>(0),
                                                              r.rest_t.template segment<3>(0),
                                                              r.rest_c.template segment<3>(3),
                                                              r.rest_t.template segment<3>(3));
    return J01T_H01_J01;
}
}  // namespace abd_driving_prismatic_joint_detail

// ===========================================================================
//  the joint-limit constitution's own law (uid 669) — the third clamp site
// ===========================================================================
namespace abd_prismatic_joint_limit_detail
{
/// `G = dE_dx * dx_dq` (the limit k1 kernel's whole arithmetic).
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> gradient(const S&              dE_dx,
                                               const lps::LVec<S, 24>& dx_dq) noexcept
{
    return dE_dx * dx_dq;
}

/// The limit k2 kernel's `H`, i.e. the reference's
/// `H = d2E_dx2 * (dx_dq * dx_dq^T); if(dE_dx != 0) compute_absolute_distance_hessian(H, ...)`.
/// `dE_dx != 0` is a *device* predicate, so the second term is computed
/// unconditionally and its *contribution* selected (a no-op when it fails: with
/// `dE_dx == 0` every entry of `H_D` is zero, `make_spd(0) == 0` and the fan-out is
/// exactly zero, so the two spellings are bitwise equal — the test asserts it).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> hessian(const S&                d2E_dx2,
                                                  const S&                dE_dx,
                                                  const lps::LVec<S, 24>& dx_dq,
                                                  const lps::LVec<S, 6>&  C_bar,
                                                  const lps::LVec<S, 6>&  t_bar,
                                                  const lps::LVec<S, 12>& q_i,
                                                  const lps::LVec<S, 12>& q_j) noexcept
{
    lps::LMat<S, 24, 24> H = d2E_dx2 * (dx_dq * dx_dq.transpose());
    abd_driving_prismatic_joint_detail::compute_absolute_distance_hessian<S>(
        H, dE_dx, C_bar, t_bar, q_i, q_j);
    return H;
}
}  // namespace abd_prismatic_joint_limit_detail

// ===========================================================================
//  the external-force reporter's own arithmetic (uid 667)
// ===========================================================================
namespace abd_prismatic_joint_external_force_detail
{
/// The reference's two "symmetrize to avoid numerical issues when the joint is near
/// singularity" lines, with the *same* statement order:
///   `t_i = J_i.vec_x(q_i); t_j = J_j.vec_x(q_j);`
///   `t_j = 0.5 * (t_i + t_j); t_i = -t_j;`
/// (`J.vec_x(q) = A x_bar` — the *rotated* axis direction, no translation.)
template <typename S>
inline void symmetrized_axes(lps::LVec<S, 3>&       t_i,
                             lps::LVec<S, 3>&       t_j,
                             const ABDJacobiT<S>&   J_i,
                             const ABDJacobiT<S>&   J_j,
                             const lps::LVec<S, 12>& q_i,
                             const lps::LVec<S, 12>& q_j) noexcept
{
    t_i = abd_jacobi_vec_x<S>(J_i, q_i);
    t_j = abd_jacobi_vec_x<S>(J_j, q_j);
    t_j = lps::make_scalar<S>(0.5) * (t_i + t_j);
    t_i = -t_j;
}

/// The two 12-vectors the reporter scatters (`F_i[0:3] = f * t_i`,
/// `F_j[0:3] = f * t_j`, everything else zero).
template <typename S>
inline void force_pair(const S&               f,
                       const lps::LVec<S, 3>& t_i,
                       const lps::LVec<S, 3>& t_j,
                       lps::LVec<S, 12>&      F_i,
                       lps::LVec<S, 12>&      F_j) noexcept
{
    F_i = lps::LVec<S, 12>::Zero();
    F_j = lps::LVec<S, 12>::Zero();
    F_i.template segment<3>(0) = f * t_i;
    F_j.template segment<3>(0) = f * t_j;
}
}  // namespace abd_prismatic_joint_external_force_detail

// ===========================================================================
//  the two assembly write patterns (the reference's DVA / TMA spellings)
// ===========================================================================
namespace abd_prismatic_joint_detail
{
/**
 * @brief The reference's `DVA.segment<2>(2 * I).write(indices, G)`.
 *
 * Two consecutive doublets: slot `base` gets body `idx0`'s 12-vector `G[0:12]`,
 * slot `base + 1` gets body `idx1`'s `G[12:24]` (the reference's
 * `ProxyRange<2>::write` loop).  The same helper serves the driving joint's
 * `segment<StencilSize>(StencilSize * I)` (StencilSize == 2) and the joint-limit's
 * `segment<2>(2 * I)`.
 */
template <typename DoubletView, typename S, typename Base, typename Idx>
inline void write_pair_gradient(const DoubletView&       doublets,
                                const lps::LVec<S, 24>&  G,
                                const Base&              base,
                                const lps::LVec<Idx, 2>& indices) noexcept
{
    doublets(base).write(indices(0), lps::LVec<S, 12>(G.template segment<12>(0)));
    doublets(base + 1).write(indices(1), lps::LVec<S, 12>(G.template segment<12>(12)));
}

/**
 * @brief The reference's `TMA.half_block<2>(HalfHessianSize * I).write(indices, H)`.
 *
 * The upper triangle of the 2x2 *block* matrix in the reference's own `upper_LR`
 * order: `(0,0)` and `(1,1)` are fixed, and the off-diagonal block is written at
 * `(indices(L), indices(R))` with `(L, R) = (0, 1)` iff `indices(0) < indices(1)`,
 * otherwise `(1, 0)` — with the *same* rule applied to the block's content
 * (`H.block(L, R)`).  Both are device predicates, so they become selects (RULE M2).
 */
template <typename TripletView, typename S, typename Base, typename Idx>
inline void write_pair_half_hessian(const TripletView&          triplets,
                                    const lps::LMat<S, 24, 24>& H,
                                    const Base&                 base,
                                    const lps::LVec<Idx, 2>&    indices) noexcept
{
    /// The 12x12 off-diagonal block at *block* position `(L, R)`; `H` is const so the
    /// numerics layer's `block<>` returns a copy.
    const auto block = [&](int L, int R)
    { return lps::LMat<S, 12, 12>(H.template block<12, 12>(L * 12, R * 12)); };

    const auto ascending = indices(0) < indices(1);  // the reference's upper_LR test
    const auto off_row   = scalar_detail::select_scalar(indices(1), indices(0), ascending);
    const auto off_col   = scalar_detail::select_scalar(indices(0), indices(1), ascending);
    const auto off_block = scalar_detail::select_container(block(1, 0), block(0, 1), ascending);

    triplets(base).write(indices(0), indices(0), block(0, 0));
    triplets(base + 1).write(off_row, off_col, off_block);
    triplets(base + 2).write(indices(1), indices(1), block(1, 1));
}
}  // namespace abd_prismatic_joint_detail
}  // namespace uipc::backend::luisa
