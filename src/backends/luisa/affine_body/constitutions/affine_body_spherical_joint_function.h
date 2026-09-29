#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_spherical_joint_function.h
 * @brief T6.3b1 — the ABD spherical joint's arithmetic (port of
 *        src/backends/cuda/affine_body/constitutions/affine_body_spherical_joint_function.h).
 *
 * The reference header is a thin include shell around the SymEigen.py-generated
 * `sym/affine_body_spherical_joint.inl`; the port keeps that shape and adds the
 * *factored* per-joint routines (the T6.3a `abd_arap_detail` precedent): one
 * implementation over the scalar type `S`, never a re-spelling of the generated
 * statements.
 *
 * THE JOINT.  A spherical (ball) joint constrains only the two attachment points:
 * there is no frame/rotation term, so the reference's `.cu` computes only the
 * translation half of the fixed joint:
 *   Ft = A_i ci_bar + ti - (A_j cj_bar + tj)   (generated `C0` constraint)
 *   E  = Et = 1/2 kappa ||Ft||^2
 *   G  = JtT_Gt = (dFt/dq)^T dEdFt             (the 24-vector `[body_i ; body_j]`)
 *   H  = JtT_Ht_Jt = (dFt/dq)^T ddEdFt (dFt/dq)
 * and the reference clamps `ddEdFt` with `make_spd` *before* the fan-out (a no-op
 * for `kappa > 0`, load-bearing for a negative `strength_ratio` — the port keeps the
 * statement; see `affine_body_fixed_joint_function.h` for the `kMakeSpdSweeps`
 * argument).
 *
 * The two write helpers (the reference's `DoubletVectorAssembler` /
 * `TripletMatrixAssembler` patterns) are the fixed joint's, duplicated on purpose:
 * the T6.3a batch's two `_function.h` files duplicate their `body_*` routines the
 * same way (one generic implementation *per constitution*, never a shared header);
 * T6.3b2 will factor them if the six joints keep the same two patterns.
 ********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/math.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace
#include <utils/scalar_util.h>             // scalar_detail::select_scalar/container

namespace uipc::backend::luisa
{
namespace sym::affine_body_spherical_joint
{
#include "sym/affine_body_spherical_joint.inl"
}  // namespace sym::affine_body_spherical_joint

namespace abd_spherical_joint_detail
{
/// See `abd_fixed_joint_detail::kMakeSpdSweeps` (the same measured plateau; the
/// spherical joint's only element Hessian is the 3x3 `ddEdFt`).
inline constexpr int kMakeSpdSweeps = 8;

/// The joint's rest data: the reference's single `Vector6` row
/// `rest_c = [ci_bar | cj_bar]` (the two attachment points, body-local).
template <typename S>
struct SphericalJointRestT
{
    lps::LVec<S, 6> rest_c;  // [ci_bar | cj_bar]
};

/// The reference's `Es(I) = Et_val;` (`Ft` then `Et`, in that order).
template <typename S>
[[nodiscard]] inline S body_energy(const S&                      kappa,
                                   const SphericalJointRestT<S>& r,
                                   const lps::LVec<S, 12>&       qi,
                                   const lps::LVec<S, 12>&       qj)
{
    lps::LVec<S, 3> Ft_val;
    sym::affine_body_spherical_joint::Ft<S>(Ft_val,
                                            r.rest_c.template segment<3>(0),
                                            qi,
                                            r.rest_c.template segment<3>(3),
                                            qj);
    S Et_val;
    sym::affine_body_spherical_joint::Et<S>(Et_val, kappa, Ft_val);
    return Et_val;
}

/// The reference's `SJ::JtT_Gt<Float>(G, dEdFt_val, ci_bar, cj_bar);`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> body_gradient(const S&                      kappa,
                                                    const SphericalJointRestT<S>& r,
                                                    const lps::LVec<S, 12>&       qi,
                                                    const lps::LVec<S, 12>&       qj)
{
    lps::LVec<S, 3> Ft_val;
    sym::affine_body_spherical_joint::Ft<S>(Ft_val,
                                            r.rest_c.template segment<3>(0),
                                            qi,
                                            r.rest_c.template segment<3>(3),
                                            qj);

    lps::LVec<S, 3> dEdFt_val;
    sym::affine_body_spherical_joint::dEdFt<S>(dEdFt_val, kappa, Ft_val);

    lps::LVec<S, 24> JtT_Gt_val;
    sym::affine_body_spherical_joint::JtT_Gt<S>(JtT_Gt_val,
                                                dEdFt_val,
                                                r.rest_c.template segment<3>(0),
                                                r.rest_c.template segment<3>(3));
    return JtT_Gt_val;
}

/// The reference's `ddEdFt(ddEdFt_val, kappa, Ft_val); make_spd(ddEdFt_val);
/// SJ::JtT_Ht_Jt<Float>(H, ddEdFt_val, ci_bar, cj_bar);` (the clamp is *before*
/// the fan-out, as always).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> body_hessian(const S&                      kappa,
                                                       const SphericalJointRestT<S>& r,
                                                       const lps::LVec<S, 12>&       qi,
                                                       const lps::LVec<S, 12>&       qj)
{
    lps::LVec<S, 3> Ft_val;
    sym::affine_body_spherical_joint::Ft<S>(Ft_val,
                                            r.rest_c.template segment<3>(0),
                                            qi,
                                            r.rest_c.template segment<3>(3),
                                            qj);

    lps::LMat<S, 3, 3> ddEdFt_val;
    sym::affine_body_spherical_joint::ddEdFt<S>(ddEdFt_val, kappa, Ft_val);
    lps::make_spd_inplace(ddEdFt_val, kMakeSpdSweeps);

    lps::LMat<S, 24, 24> JtT_Ht_Jt_val;
    sym::affine_body_spherical_joint::JtT_Ht_Jt<S>(JtT_Ht_Jt_val,
                                                    ddEdFt_val,
                                                    r.rest_c.template segment<3>(0),
                                                    r.rest_c.template segment<3>(3));
    return JtT_Ht_Jt_val;
}

/// The reference's `DVA.segment<2>(2 * I).write(indices, G)` (see
/// `abd_fixed_joint_detail::write_pair_gradient`; this is the same statement).
template <typename DoubletView, typename S, typename Base, typename Idx>
inline void write_pair_gradient(const DoubletView&       doublets,
                                const lps::LVec<S, 24>&  G,
                                const Base&              base,
                                const lps::LVec<Idx, 2>& indices) noexcept
{
    doublets(base).write(indices(0), lps::LVec<S, 12>(G.template segment<12>(0)));
    doublets(base + 1).write(indices(1), lps::LVec<S, 12>(G.template segment<12>(12)));
}

/// The reference's `TMA.half_block<2>(HalfHessianSize * I).write(indices, H)` (see
/// `abd_fixed_joint_detail::write_pair_half_hessian`; this is the same statement,
/// `upper_LR` included).
template <typename TripletView, typename S, typename Base, typename Idx>
inline void write_pair_half_hessian(const TripletView&          triplets,
                                    const lps::LMat<S, 24, 24>& H,
                                    const Base&                 base,
                                    const lps::LVec<Idx, 2>&    indices) noexcept
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
}  // namespace abd_spherical_joint_detail
}  // namespace uipc::backend::luisa
