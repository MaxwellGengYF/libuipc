#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_revolute_joint_function.h
 * @brief T6.3b3 — the arithmetic the revolute joint (uid 18), its driving constraint
 *        (uid 19) and its joint-limit constitution (uid 670) share (port of
 *        src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint_function.h).
 *
 * The reference header is a thin include shell around the three SymEigen.py-generated
 * bodies plus the two hand-written helpers `unwrap_angle` and
 * `compute_relative_angle` / `driving_theta_tilde`; the port keeps that shape and adds
 * the *factored* per-joint routines the kernels and the unit test share (the T6.3a
 * `abd_arap_detail` / T6.3b1 `abd_fixed_joint_detail` / T6.3b2
 * `abd_prismatic_joint_detail` precedent): one implementation over the scalar type `S`
 * (`double` on the host, `lcrt::Var<double>` in the DSL), never a re-spelling of the
 * generated statements.
 *
 * THE REFERENCE INCLUDES **ALL THREE** `.inl` HERE (not in the `.cu`) — the limit body
 * included through `<affine_body/constitutions/sym/...>`; the port uses the quoted
 * relative form for the three.  Consequence for the reader: this header is the only
 * place the three generated bodies are pulled in, so the joint's own `.cpp` parses all
 * three (the 1.1 MB reference driving body is only ~120 KB once the SymEigen comments
 * are stripped, i.e. 1192 statements).
 *
 * THE THREE LAWS (verbatim from the generated bodies; the readings are the ones the
 * unit test's independent oracles re-derive):
 *
 *   joint (uid 18), `kappa = strength_ratio(I) * (m(bids0) + m(bids1))`, with
 *   `rest_positions = [l_pos0 | l_pos1 | r_pos0 | r_pos1]` (the four body-local points
 *   the frontend writes):
 *     Faxis = [ A_i l_pos0 + ti - (A_j r_pos0 + tj) ;         (two *coincident point
 *               A_i l_pos1 + ti - (A_j r_pos1 + tj) ]           pairs* = the revolute
 *     E     = Eaxis = 1/2 kappa ||Faxis||^2                    joint's 5 dof)
 *     G     = JaxisT_Gaxis(dEaxisdFaxis(Faxis)),  Jaxis = dFaxis/dq constant (6x24)
 *     H     = JaxisT_Haxis_Jaxis(make_spd(ddEaxisddFaxis(Faxis)))
 *   The element Hessian is `kappa I6` (DIAGONAL), so unlike the prismatic joint's 9x9
 *   the reference's `make_spd` clamp on it is value-invariant — the opposite of the
 *   T6.3b2 result, and the reason the sweep-count perturbation for this batch is aimed
 *   at the *driving* / *limit* element Hessians instead (§7.26 rule 10 vs §7.25 rule
 *   19; re-measured in the test).
 *
 *   driving (uid 19), `l_basis = [n_i_bar ; b_i_bar]`, `r_basis = [n_j_bar ; b_j_bar]`:
 *     F01_q = [ A_i n_i_bar ; A_i b_i_bar ; A_j n_j_bar ; A_j b_j_bar ]      (12)
 *     theta = atan2( (v.p - u.q)/2, (u.p + v.q)/2 )      (u,v = the i half, p,q the j half)
 *     E     = 1/2 kappa (theta - theta_tilde)^2
 *     G     = J01T_G01(dEdF01(kappa, F01_q, theta_tilde))
 *     H     = J01T_H01_J01(make_spd(ddEddF01(kappa, F01_q, theta_tilde)))
 *   `theta_tilde` is a *parameter*: the reference's `driving_theta_tilde` folds the
 *   frame-start unwrap and the `aim_angle` into it (and the reference kernel's comment
 *   notes that the gradient is unaffected by that fold — the fold shifts `theta` by a
 *   constant within one frame, so `dEdF01` sees the same difference).
 *   The 12x12 element Hessian is NOT diagonal (the atan2 chain rule mixes all four
 *   blocks}, so this clamp IS a real projection.
 *
 *   joint limit (uid 670), with `wrap(x) = x` reduced to (-pi, pi]:
 *     F(basis, q)     = [ A n_bar ; A b_bar ]
 *     DeltaTheta(F,Ft)= wrap(theta(F) - theta(Ft))       (the 12-vector form)
 *     DeltaTheta(bk,qk,qkt,bl,ql,qlt) = theta(F(bk,qk),F(bl,ql))
 *                                     - theta(F(bk,qkt),F(bl,qlt))  wrapped
 *     dDeltaTheta_dQ  = dDeltaTheta/d[qk; ql]
 *     E = eval_penalty_energy(theta_prev + DeltaTheta, lower - init, upper - init, strength)
 *     G = dE_dx * dDeltaTheta_dQ
 *     H = d2E_dx2 (dx_dq dx_dq^T) [+ JT_H_J(make_spd(dE_dx * ddDeltaTheta_ddF(F, F_prev)))]
 *   (`JT_G` is generated but not called by the reference's own `.cu` — kept because the
 *   generator's region contract is the whole generated body.)
 *
 * DEVICE PREDICATES.  The two device-only branches the reference spells with `if`
 * (the `dE_dx != 0` guard around the limit Hessian's second term, and the driving's
 * `constrained == 0` early-outs) are `$if` / `select` respectively — see the two `.cpp`
 * files; the *host* facing helper here computes the limit's second term
 * unconditionally (bitwise-equal when `dE_dx == 0`: every entry of `dE_dx * ddF` is
 * zero, so `make_spd` of it is zero and the fan-out is zero — measured in the test).
 *
 * `utils/make_spd.h` is a tolerance-driven device loop (a device loop count, RULE M2),
 * so the port picks a fixed sweep count (`kMakeSpdSweeps`, §7.17 rule 31's plateau; 8
 * is the measured plateau for N = 6/9/12 in this layer).
 *
 * THE ASSEMBLY WRITE PATTERNS are T6.3b1/T6.3b2's: the gradient as the 2-doublet run
 * `DVA.segment<2>(2 * I)` at `{bids(0), bids(1)}`, the Hessian as the upper-triangular
 * 2x2 12x12 block run `TMA.half_block<2>(HalfHessianSize * I)` with the reference's
 * `upper_LR` placement (a *device* predicate -> `select`, RULE M2).  They are
 * re-stated here rather than imported from the prismatic header so that this file's
 * include closure is the reference's own.
 ********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/math.h>
// `abd_jacobi_vec_x` (the external-force reporter's `J.vec_x(q)`, cookbook §7.21)
#include <affine_body/abd_jacobi_matrix.h>
// `lps::make_spd_inplace` lives in matrix_ops.h; math.h alone does not declare it
// (measured C2039 in T6.3a).
#include <luisa_device/math/matrix_ops.h>
// `scalar_detail::{select_scalar, select_container}` — the device-predicate selects
// the reference's `upper_LR` needs (the FEM shells' shared helper).
#include <utils/scalar_util.h>
// the joint-limit penalty (the reference's `.cu` includes it; kept here so this
// header's closure covers the whole family, exactly as the prismatic one does)
#include <affine_body/constitutions/joint_limit_penalty.h>
// the reference's `unwrap_angle` uses `std::numbers::pi`
#include <numbers>

namespace uipc::backend::luisa
{
namespace sym::affine_body_revolute_joint
{
#include "sym/affine_body_revolute_joint.inl"
}  // namespace sym::affine_body_revolute_joint

namespace sym::affine_body_driving_revolute_joint
{
#include "sym/affine_body_driving_revolute_joint.inl"
}  // namespace sym::affine_body_driving_revolute_joint

namespace sym::affine_body_revolute_joint_limit
{
#include "sym/affine_body_revolute_joint_limit.inl"
}  // namespace sym::affine_body_revolute_joint_limit

// ===========================================================================
//  the two hand-written helpers of the reference header
// ===========================================================================
/**
 * @brief Unwrap `wrapped_angle` to the 2*pi representative nearest `ref`
 * (energy and gradient must pass identical args — the reference's own comment).
 *
 * The reference spells this `ref + ::remainder(wrapped_angle - ref, 2*pi)`.
 * `lps` has no `remainder` (only `mod`, which is *not* the same function), so the
 * port spells the IEEE identity `remainder(x, y) == x - y * round(x / y)` with the
 * layer's `lps::round`.  The two agree everywhere except an exact half-quotient tie
 * (where `std::remainder` rounds half-to-even and `lps::round` half-away); the test
 * measures the difference on every sampled state and asserts it is 0.
 */
template <typename S>
[[nodiscard]] inline S unwrap_angle(const S& wrapped_angle, const S& ref) noexcept
{
    const S two_pi = lps::make_scalar<S>(2.0 * std::numbers::pi);
    const S x      = lps::detail::sub(wrapped_angle, ref);
    return lps::detail::add(ref,
                            lps::detail::sub(x,
                                             lps::detail::mul(
                                                 two_pi, lps::round(lps::detail::div(x, two_pi)))));
}

/// The reference's `compute_relative_angle(Float& out_theta, const Vector6& l_basis,
/// const Vector12& q_i, const Vector6& r_basis, const Vector12& q_j)`: the relative
/// rotation angle about the axle, `theta(F01_q(l_basis, q_i, r_basis, q_j))`.
template <typename S>
[[nodiscard]] inline S compute_relative_angle(const lps::LVec<S, 6>& l_basis,
                                              const lps::LVec<S, 12>& q_i,
                                              const lps::LVec<S, 6>&  r_basis,
                                              const lps::LVec<S, 12>& q_j) noexcept
{
    lps::LVec<S, 12> F01_q;
    sym::affine_body_driving_revolute_joint::F01_q<S>(F01_q,
                                                      l_basis.template segment<3>(0),
                                                      l_basis.template segment<3>(3),
                                                      q_i,
                                                      r_basis.template segment<3>(0),
                                                      r_basis.template segment<3>(3),
                                                      q_j);
    S out_theta;
    sym::affine_body_driving_revolute_joint::theta<S>(out_theta, F01_q);
    return out_theta;
}

/// The reference's `driving_theta_tilde(F01_q, aim_angle, init_angle, ref_angle)`:
/// fold `aim_angle` into a theta_tilde matching the wrapped measurement (energy and
/// gradient must pass identical args).
template <typename S>
[[nodiscard]] inline S driving_theta_tilde(const lps::LVec<S, 12>& F01_q,
                                           const S&                aim_angle,
                                           const S&                init_angle,
                                           const S&                ref_angle) noexcept
{
    S theta_a;
    sym::affine_body_driving_revolute_joint::theta<S>(theta_a, F01_q);
    const S theta_eff = unwrap_angle<S>(theta_a, lps::detail::sub(ref_angle, init_angle));
    return lps::detail::sub(lps::detail::sub(aim_angle, init_angle),
                            lps::detail::sub(theta_eff, theta_a));
}

// ===========================================================================
//  the revolute joint (uid 18)
// ===========================================================================
namespace abd_revolute_joint_detail
{
/// See the file banner: the reference's `make_spd` is a tolerance-driven Jacobi loop,
/// so the port picks a fixed sweep count.  NOTE: for *this* joint the 6x6 element
/// Hessian is exactly `kappa I6`, so the clamp is provably value-invariant here
/// (`make_spd(diag(k), s) == diag(max(k, 0))` for every `s`) — the T6.3b1 situation,
/// not the T6.3b2 one; the test measures it and pins the sweep pin on the *driving*
/// and *limit* element Hessians, which are not diagonal.
inline constexpr int kMakeSpdSweeps = 8;

/// One joint's rest data: the reference's single `Vector12` per joint,
/// `[l_pos0 | l_pos1 | r_pos0 | r_pos1]` (its argument names are
/// `qi0_bar, qi1_bar, qj0_bar, qj1_bar`).
template <typename S>
struct RevoluteJointRestT
{
    lps::LVec<S, 12> rest_positions;
};

/// `Faxis` (6-vector) — the reference's own argument order.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 6> faxis(const RevoluteJointRestT<S>& r,
                                           const lps::LVec<S, 12>&      qi,
                                           const lps::LVec<S, 12>&      qj) noexcept
{
    lps::LVec<S, 6> out;
    sym::affine_body_revolute_joint::Faxis<S>(out,
                                              r.rest_positions.template segment<3>(0),
                                              r.rest_positions.template segment<3>(3),
                                              qi,
                                              r.rest_positions.template segment<3>(6),
                                              r.rest_positions.template segment<3>(9),
                                              qj);
    return out;
}

/// The reference's `Es(I) = Eaxis;` (E = 1/2 kappa ||Faxis||^2).
template <typename S>
[[nodiscard]] inline S body_energy(const S&                     kappa,
                                   const RevoluteJointRestT<S>& r,
                                   const lps::LVec<S, 12>&      qi,
                                   const lps::LVec<S, 12>&      qj) noexcept
{
    S E;
    sym::affine_body_revolute_joint::Eaxis<S>(E, kappa, faxis<S>(r, qi, qj));
    return E;
}

/// The reference's `G24 = JaxisT_Gaxis(dEaxisdFaxis(Faxis))`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> body_gradient(const S&                     kappa,
                                                    const RevoluteJointRestT<S>& r,
                                                    const lps::LVec<S, 12>&       qi,
                                                    const lps::LVec<S, 12>&       qj) noexcept
{
    const auto Faxis_val = faxis<S>(r, qi, qj);

    lps::LVec<S, 6> dEdF;
    sym::affine_body_revolute_joint::dEaxisdFaxis<S>(dEdF, kappa, Faxis_val);

    lps::LVec<S, 24> out;
    sym::affine_body_revolute_joint::JaxisT_Gaxis<S>(out,
                                                     dEdF,
                                                     r.rest_positions.template segment<3>(0),
                                                     r.rest_positions.template segment<3>(3),
                                                     r.rest_positions.template segment<3>(6),
                                                     r.rest_positions.template segment<3>(9));
    return out;
}

/// The reference's `H24 = JaxisT_Haxis_Jaxis(make_spd(ddEaxisddFaxis(Faxis)))` — the
/// clamp is applied to the 6x6 *element* Hessian, before the fan-out.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> body_hessian(const S&                     kappa,
                                                       const RevoluteJointRestT<S>& r,
                                                       const lps::LVec<S, 12>&       qi,
                                                       const lps::LVec<S, 12>&       qj) noexcept
{
    const auto Faxis_val = faxis<S>(r, qi, qj);

    lps::LMat<S, 6, 6> ddEddF;
    sym::affine_body_revolute_joint::ddEaxisddFaxis<S>(ddEddF, kappa, Faxis_val);
    lps::make_spd_inplace(ddEddF, kMakeSpdSweeps);

    lps::LMat<S, 24, 24> out;
    sym::affine_body_revolute_joint::JaxisT_Haxis_Jaxis<S>(
        out,
        ddEddF,
        r.rest_positions.template segment<3>(0),
        r.rest_positions.template segment<3>(3),
        r.rest_positions.template segment<3>(6),
        r.rest_positions.template segment<3>(9));
    return out;
}
}  // namespace abd_revolute_joint_detail

// ===========================================================================
//  the driving revolute joint (uid 19)
// ===========================================================================
namespace abd_driving_revolute_joint_detail
{
/// See `abd_revolute_joint_detail::kMakeSpdSweeps`; the driving's element Hessian is a
/// non-diagonal 12x12 (the atan2 chain rule), so this clamp IS observable.
inline constexpr int kMakeSpdSweeps = 8;

/// One driving pair's rest data: the two bodies' body-local frame rows,
/// `l_basis = [n_i_bar | b_i_bar]`, `r_basis = [n_j_bar | b_j_bar]`.
template <typename S>
struct DrivingRevoluteJointRestT
{
    // clang-format off
    lps::LVec<S, 6> l_basis;
    lps::LVec<S, 6> r_basis;
    // clang-format on
};

/// `F01_q` (12-vector) — the reference's own argument order.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> f01_q(const DrivingRevoluteJointRestT<S>& r,
                                            const lps::LVec<S, 12>&             qi,
                                            const lps::LVec<S, 12>&             qj) noexcept
{
    lps::LVec<S, 12> out;
    sym::affine_body_driving_revolute_joint::F01_q<S>(out,
                                                      r.l_basis.template segment<3>(0),
                                                      r.l_basis.template segment<3>(3),
                                                      qi,
                                                      r.r_basis.template segment<3>(0),
                                                      r.r_basis.template segment<3>(3),
                                                      qj);
    return out;
}

/// `theta(F01_q)` — the relative rotation angle about the axle.
template <typename S>
[[nodiscard]] inline S theta_of(const lps::LVec<S, 12>& F01_q) noexcept
{
    S out;
    sym::affine_body_driving_revolute_joint::theta<S>(out, F01_q);
    return out;
}

/// The driving energy kernel's own law, `E(kappa, F01_q, theta_tilde)`.
template <typename S>
[[nodiscard]] inline S body_energy(const S&                            kappa,
                                   const DrivingRevoluteJointRestT<S>& r,
                                   const lps::LVec<S, 12>&             qi,
                                   const lps::LVec<S, 12>&             qj,
                                   const S&                            theta_tilde) noexcept
{
    S E;
    sym::affine_body_driving_revolute_joint::E<S>(E, kappa, f01_q<S>(r, qi, qj), theta_tilde);
    return E;
}

/// The gradient's own law, `J01T_G01(dEdF01(kappa, F01_q, theta_tilde))`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> body_gradient(const S&                            kappa,
                                                    const DrivingRevoluteJointRestT<S>& r,
                                                    const lps::LVec<S, 12>&             qi,
                                                    const lps::LVec<S, 12>&             qj,
                                                    const S& theta_tilde) noexcept
{
    const auto F01_q = f01_q<S>(r, qi, qj);

    lps::LVec<S, 12> G01;
    sym::affine_body_driving_revolute_joint::dEdF01<S>(G01, kappa, F01_q, theta_tilde);

    lps::LVec<S, 24> out;
    sym::affine_body_driving_revolute_joint::J01T_G01<S>(out,
                                                         G01,
                                                         r.l_basis.template segment<3>(0),
                                                         r.l_basis.template segment<3>(3),
                                                         r.r_basis.template segment<3>(0),
                                                         r.r_basis.template segment<3>(3));
    return out;
}

/// The Hessian's own law, `J01T_H01_J01(make_spd(ddEddF01(kappa, F01_q, theta_tilde)))`.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> body_hessian(const S&                            kappa,
                                                       const DrivingRevoluteJointRestT<S>& r,
                                                       const lps::LVec<S, 12>&             qi,
                                                       const lps::LVec<S, 12>&             qj,
                                                       const S& theta_tilde) noexcept
{
    const auto F01_q = f01_q<S>(r, qi, qj);

    lps::LMat<S, 12, 12> H01;
    sym::affine_body_driving_revolute_joint::ddEddF01<S>(H01, kappa, F01_q, theta_tilde);
    lps::make_spd_inplace(H01, kMakeSpdSweeps);

    lps::LMat<S, 24, 24> out;
    sym::affine_body_driving_revolute_joint::J01T_H01_J01<S>(out,
                                                             H01,
                                                             r.l_basis.template segment<3>(0),
                                                             r.l_basis.template segment<3>(3),
                                                             r.r_basis.template segment<3>(0),
                                                             r.r_basis.template segment<3>(3));
    return out;
}
}  // namespace abd_driving_revolute_joint_detail

// ===========================================================================
//  the joint-limit constitution's own law (uid 670)
// ===========================================================================
namespace abd_revolute_joint_limit_detail
{
/// The reference's `F(basis_k, qk, basis_l, ql)` (12-vector): the four body-local frame
/// rows of one body pair, the same layout the driving joint's `F01_q` has.  It is the
/// operand the 12-vector `DeltaTheta` / `ddDeltaTheta_ddF` overloads consume.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> pair_f(const lps::LVec<S, 6>&  l_basis,
                                             const lps::LVec<S, 12>& qk,
                                             const lps::LVec<S, 6>&  r_basis,
                                             const lps::LVec<S, 12>& ql) noexcept
{
    lps::LVec<S, 12> out;
    sym::affine_body_revolute_joint_limit::F<S>(out, l_basis, qk, r_basis, ql);
    return out;
}

/// The 6-vector `DeltaTheta(basis_k, qk, qk_t, basis_l, ql, ql_t)`: the wrapped
/// difference of the pair's relative angle between the current and the previous state.
template <typename S>
[[nodiscard]] inline S delta_theta(const lps::LVec<S, 6>&  l_basis,
                                   const lps::LVec<S, 12>& qk,
                                   const lps::LVec<S, 12>& qk_t,
                                   const lps::LVec<S, 6>&  r_basis,
                                   const lps::LVec<S, 12>& ql,
                                   const lps::LVec<S, 12>& ql_t) noexcept
{
    S out;
    sym::affine_body_revolute_joint_limit::DeltaTheta<S>(
        out, l_basis, qk, qk_t, r_basis, ql, ql_t);
    return out;
}

/// The 12-vector `DeltaTheta(F, F_t)` overload (the same value, in `F` space).
template <typename S>
[[nodiscard]] inline S delta_theta_f(const lps::LVec<S, 12>& F,
                                     const lps::LVec<S, 12>& F_t) noexcept
{
    S out;
    sym::affine_body_revolute_joint_limit::DeltaTheta<S>(out, F, F_t);
    return out;
}

/// `dDeltaTheta / d[qk ; ql]` (24) — the chain the limit's k1 kernel consumes.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> d_delta_theta_dq(const lps::LVec<S, 6>&  l_basis,
                                                       const lps::LVec<S, 12>& qk,
                                                       const lps::LVec<S, 12>& qk_t,
                                                       const lps::LVec<S, 6>&  r_basis,
                                                       const lps::LVec<S, 12>& ql,
                                                       const lps::LVec<S, 12>& ql_t) noexcept
{
    lps::LVec<S, 24> out;
    sym::affine_body_revolute_joint_limit::dDeltaTheta_dQ<S>(
        out, l_basis, qk, qk_t, r_basis, ql, ql_t);
    return out;
}

/// `d^2 DeltaTheta / dF^2` (12x12) — NOT diagonal (the atan2 chain rule).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> dd_delta_theta_ddF(const lps::LVec<S, 12>& F,
                                                             const lps::LVec<S, 12>& F_t) noexcept
{
    lps::LMat<S, 12, 12> out;
    sym::affine_body_revolute_joint_limit::ddDeltaTheta_ddF<S>(out, F, F_t);
    return out;
}

/// The reference's `J_r^T H_F J_r` fan-out, called with `(lb, rb, lb, rb)` (the
/// general four-basis signature is the generated function's own).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> jt_h_j(const lps::LMat<S, 12, 12>& H_F,
                                                 const lps::LVec<S, 6>&      l_basis,
                                                 const lps::LVec<S, 6>&      r_basis) noexcept
{
    lps::LMat<S, 24, 24> out;
    sym::affine_body_revolute_joint_limit::JT_H_J<S>(out, H_F, l_basis, r_basis, l_basis, r_basis);
    return out;
}

/// `G = dE_dx * dx_dq` (the limit k1 kernel's whole arithmetic).
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> gradient(const S&               dE_dx,
                                               const lps::LVec<S, 24>& dx_dq) noexcept
{
    return dE_dx * dx_dq;
}

/**
 * @brief The limit k2 kernel's `H`, i.e. the reference's
 * `H = d2E_dx2 * (dx_dq * dx_dq^T)` plus — *when* `dE_dx != 0` — the
 * `J^T make_spd(dE_dx * ddDeltaTheta_ddF) J` term.
 *
 * `dE_dx != 0` is a *device* predicate, so the reference's guard cannot drive a
 * host-and-device `if` (RULE M2); this helper computes the second term
 * unconditionally, which is bitwise the reference for `dE_dx == 0` (every entry of
 * `dE_dx * ddF` is zero, `make_spd(0) == 0` and the fan-out of the zero matrix is
 * zero) — the same deviation `abd_prismatic_joint_limit_detail::hessian` documents,
 * and the test measures it.  The *kernel* keeps the reference's `$if` (its own
 * control flow, cheaper); this helper is what the host oracle and the device round
 * call.
 */
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> hessian(const S&                d2E_dx2,
                                                  const S&                dE_dx,
                                                  const lps::LVec<S, 24>& dx_dq,
                                                  const lps::LVec<S, 6>&  l_basis,
                                                  const lps::LVec<S, 12>& qk,
                                                  const lps::LVec<S, 12>& qk_t,
                                                  const lps::LVec<S, 6>&  r_basis,
                                                  const lps::LVec<S, 12>& ql,
                                                  const lps::LVec<S, 12>& ql_t) noexcept
{
    lps::LMat<S, 24, 24> H = d2E_dx2 * (dx_dq * dx_dq.transpose());

    lps::LMat<S, 12, 12> H_F =
        dE_dx * dd_delta_theta_ddF<S>(pair_f<S>(l_basis, qk, r_basis, ql),
                                      pair_f<S>(l_basis, qk_t, r_basis, ql_t));
    lps::make_spd_inplace(H_F, abd_driving_revolute_joint_detail::kMakeSpdSweeps);
    H += jt_h_j<S>(H_F, l_basis, r_basis);
    return H;
}
}  // namespace abd_revolute_joint_limit_detail

// ===========================================================================
//  the external-force reporter's own arithmetic (uid 668)
// ===========================================================================
namespace abd_revolute_joint_external_force_detail
{
/// The reference's `affine_body_revolute_joint_external_force_step_kernel` axis step:
/// the joint's axle directions in the two bodies' frames
/// (`e_bar_i = rest_positions[3:6] - rest_positions[0:3]`, i.e. `l_pos1 - l_pos0`, and
/// `e_bar_j = rest_positions[9:12] - rest_positions[6:9]`), rotated by each body and
/// normalised, then symmetrised (`e_j = (e_j + e_i)/2`, `e_i = -e_j`) so the pair is
/// exactly anti-parallel (`torque_to_F` then gets a unit axis on both sides).
template <typename S>
inline void symmetrized_axes(lps::LVec<S, 3>&        e_i,
                             lps::LVec<S, 3>&        e_j,
                             const lps::LVec<S, 3>&  x_bar_i,
                             const lps::LVec<S, 3>&  x_bar_j,
                             const lps::LVec<S, 12>& q_i,
                             const lps::LVec<S, 12>& q_j) noexcept
{
    e_i = abd_jacobi_vec_x<S>(ABDJacobiT<S>{.x_bar = x_bar_i}, q_i).normalized();
    e_j = abd_jacobi_vec_x<S>(ABDJacobiT<S>{.x_bar = x_bar_j}, q_j).normalized();
    e_j = lps::make_scalar<S>(0.5) * (e_j + e_i);
    e_i = -e_j;
}
}  // namespace abd_revolute_joint_external_force_detail

// ===========================================================================
//  the two assembly write patterns (the reference's DVA / TMA spellings)
// ===========================================================================
namespace abd_revolute_joint_detail
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
inline void write_pair_gradient(const DoubletView&      doublets,
                                const lps::LVec<S, 24>& G,
                                const Base&             base,
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
}  // namespace abd_revolute_joint_detail
}  // namespace uipc::backend::luisa
