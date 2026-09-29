#pragma once
/********************************************************************
 * @file dihedral_angle.h
 * @brief T5.2 (batch b) — the dihedral angle between two triangles and its
 *        gradient/Hessian (port of src/backends/cuda/utils/dihedral_angle.h).
 *
 * REFERENCE: `uipc::backend::cuda::dihedral_angle` /
 * `cuda::detail::{dihedral_angle_gradient, dihedral_angle_hessian, compute_m_hat}`
 * (Codim-IPC's `Library/Math/DIHEDRAL_ANGLE.h`).  First ported here because the
 * shell bending constitutions are its only consumers (batch b); batch c
 * (kirchhoff_rod_bending) does not use it.
 *
 * Mapping / deviations (all reported in the T5.2 batch-b report):
 *  * `template <typename S>` with `S = double` (host) resp. `S = lcrt::Var<double>`
 *    (device) replaces the reference's `template <typename T> UIPC_GENERIC`: one
 *    implementation, instantiated twice (cookbook §7.4).
 *  * `Eigen::Matrix<T,3,1>` -> `lps::LVec<S,3>`, `Eigen::Matrix<T,12,1>` ->
 *    `lps::LVec<S,12>`, `Eigen::Matrix<T,12,12>` -> `lps::LMat<S,12,12>`;
 *    `std::{acos,max,min,sqrt}` -> `lps::{acos,max,min,sqrt}` (the numerics layer's
 *    dispatch; identical on the host, the DSL intrinsic on the device).
 *  * scalar arithmetic goes through the `lps::detail::{add,sub,mul,div,neg}`
 *    trampolines (RULE M1, cookbook §7.2) — the reference's expression order and
 *    association are preserved verbatim.
 *  * the reference's two device branches have no host counterpart on the device
 *    path, so they become predicates:
 * - `if(n2.cross(n1).dot(v1 - v2) < 0) DA = -DA;` (reference spelling)
 *        -> `select_scalar(DA, -DA, pred)` (`p ? t : f`);
 *      - the mapped-argument wrapper is kept (`detail::dihedral_angle_gradient`
 *        takes (v2, v0, v1, v3) exactly like the reference, and the public
 *        `dihedral_angle_gradient` forwards with the caller's order).
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/math.h>
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
/**
 * @brief Compute the dihedral angle between two planes defined by four points.
 *
 * Mid-Edge: V1-V2
 * Opposite Points: V0, V3
 */
template <typename S>
inline void dihedral_angle(const lps::LVec<S, 3>& v0,
                           const lps::LVec<S, 3>& v1,
                           const lps::LVec<S, 3>& v2,
                           const lps::LVec<S, 3>& v3,
                           S&                     DA)
{
    namespace d = lps::detail;

    const lps::LVec<S, 3> n1 = lps::cross(v1 - v0, v2 - v0);
    const lps::LVec<S, 3> n2 = lps::cross(v2 - v3, v1 - v3);

    // reference: DA = acos(max(T(-1), min(T(1), n1.dot(n2) / sqrt(|n1|^2 |n2|^2))))
    const auto ratio =
        d::div(n1.dot(n2), lps::sqrt(d::mul(n1.squaredNorm(), n2.squaredNorm())));
    DA = lps::acos(lps::max(lps::make_scalar<S>(-1.0),
                            lps::min(lps::make_scalar<S>(1.0), ratio)));

    // reference: `if(n2.cross(n1).dot(v1 - v2) < 0) DA = -DA;`
    const auto flip = lps::cross(n2, n1).dot(v1 - v2) < lps::make_scalar<S>(0.0);
    DA = scalar_detail::select_scalar(DA, d::neg(DA), flip);
}

namespace detail
{
/// Here we map our v order to rusmas' in this function for implementation
/// convenience (verbatim from the reference).
template <typename S>
inline void dihedral_angle_gradient(const lps::LVec<S, 3>& v2,  // input: V0
                                    const lps::LVec<S, 3>& v0,  // input: V1
                                    const lps::LVec<S, 3>& v1,  // input: V2
                                    const lps::LVec<S, 3>& v3,  // input: V3
                                    lps::LVec<S, 12>&      grad)
{
    namespace d = lps::detail;

    lps::LVec<S, 3> e0 = v1 - v0;
    lps::LVec<S, 3> e1 = v2 - v0;
    lps::LVec<S, 3> e2 = v3 - v0;
    lps::LVec<S, 3> e3 = v2 - v1;
    lps::LVec<S, 3> e4 = v3 - v1;
    lps::LVec<S, 3> n1 = lps::cross(e0, e1);
    lps::LVec<S, 3> n2 = lps::cross(e2, e0);

    S n1SqNorm = n1.squaredNorm();
    S n2SqNorm = n2.squaredNorm();
    S e0norm   = e0.norm();

    // fill in gradient in order with g2, g0, g1, g3 in rusmas' doc
    grad.template segment<3>(0) = d::div(d::neg(e0norm), n1SqNorm) * n1;
    grad.template segment<3>(3) =
        d::div(d::neg(e0.dot(e3)), d::mul(e0norm, n1SqNorm)) * n1
        - d::div(e0.dot(e4), d::mul(e0norm, n2SqNorm)) * n2;
    grad.template segment<3>(6) =
        d::div(e0.dot(e1), d::mul(e0norm, n1SqNorm)) * n1
        + d::div(e0.dot(e2), d::mul(e0norm, n2SqNorm)) * n2;
    grad.template segment<3>(9) = d::div(d::neg(e0norm), n2SqNorm) * n2;
}

template <typename S>
inline void compute_m_hat(const lps::LVec<S, 3>& xp,
                          const lps::LVec<S, 3>& xe0,
                          const lps::LVec<S, 3>& xe1,
                          lps::LVec<S, 3>&       mHat)
{
    namespace d     = lps::detail;
    lps::LVec<S, 3> e = xe1 - xe0;
    mHat = (xe0 + d::div((xp - xe0).dot(e), e.squaredNorm()) * e - xp).normalized();
}

/// Here we map our v order to rusmas' in this function for implementation
/// convenience (verbatim from the reference).
template <typename S>
inline void dihedral_angle_hessian(const lps::LVec<S, 3>& v2,  // input: V0
                                   const lps::LVec<S, 3>& v0,  // input: V1
                                   const lps::LVec<S, 3>& v1,  // input: V2
                                   const lps::LVec<S, 3>& v3,  // input: V3
                                   lps::LMat<S, 12, 12>&  Hess)
{
    namespace d = lps::detail;

    lps::LVec<S, 3> e[5] = {v1 - v0, v2 - v0, v3 - v0, v2 - v1, v3 - v1};
    S norm_e[5] = {
        e[0].norm(),
        e[1].norm(),
        e[2].norm(),
        e[3].norm(),
        e[4].norm(),
    };

    lps::LVec<S, 3> n1 = lps::cross(e[0], e[1]);
    lps::LVec<S, 3> n2 = lps::cross(e[2], e[0]);
    S               n1norm = n1.norm();
    S               n2norm = n2.norm();

    lps::LVec<S, 3> mHat1, mHat2, mHat3, mHat4, mHat01, mHat02;
    compute_m_hat(v1, v0, v2, mHat1);
    compute_m_hat(v1, v0, v3, mHat2);
    compute_m_hat(v0, v1, v2, mHat3);
    compute_m_hat(v0, v1, v3, mHat4);
    compute_m_hat(v2, v0, v1, mHat01);
    compute_m_hat(v3, v0, v1, mHat02);

    S cosalpha1, cosalpha2, cosalpha3, cosalpha4;
    cosalpha1 = d::div(e[0].dot(e[1]), d::mul(norm_e[0], norm_e[1]));
    cosalpha2 = d::div(e[0].dot(e[2]), d::mul(norm_e[0], norm_e[2]));
    cosalpha3 = d::div(d::neg(e[0].dot(e[3])), d::mul(norm_e[0], norm_e[3]));
    cosalpha4 = d::div(d::neg(e[0].dot(e[4])), d::mul(norm_e[0], norm_e[4]));

    S h1, h2, h3, h4, h01, h02;
    h1  = d::div(n1norm, norm_e[1]);
    h2  = d::div(n2norm, norm_e[2]);
    h3  = d::div(n1norm, norm_e[3]);
    h4  = d::div(n2norm, norm_e[4]);
    h01 = d::div(n1norm, norm_e[0]);
    h02 = d::div(n2norm, norm_e[0]);

    lps::LMat<S, 3, 3> N1_01 =
        n1 * (mHat01.transpose() / d::mul(d::mul(h01, h01), n1norm));
    lps::LMat<S, 3, 3> N1_3 = n1 * (mHat3.transpose() / d::mul(d::mul(h01, h3), n1norm));
    lps::LMat<S, 3, 3> N1_1 = n1 * (mHat1.transpose() / d::mul(d::mul(h01, h1), n1norm));
    lps::LMat<S, 3, 3> N2_4 = n2 * (mHat4.transpose() / d::mul(d::mul(h02, h4), n2norm));
    lps::LMat<S, 3, 3> N2_2 = n2 * (mHat2.transpose() / d::mul(d::mul(h02, h2), n2norm));
    lps::LMat<S, 3, 3> N2_02 =
        n2 * (mHat02.transpose() / d::mul(d::mul(h02, h02), n2norm));
    lps::LMat<S, 3, 3> M3_01_1 =
        (d::div(cosalpha3, d::mul(d::mul(h3, h01), n1norm)) * mHat01) * n1.transpose();
    lps::LMat<S, 3, 3> M1_01_1 =
        (d::div(cosalpha1, d::mul(d::mul(h1, h01), n1norm)) * mHat01) * n1.transpose();
    lps::LMat<S, 3, 3> M1_1_1 =
        (d::div(cosalpha1, d::mul(d::mul(h1, h1), n1norm)) * mHat1) * n1.transpose();
    lps::LMat<S, 3, 3> M3_3_1 =
        (d::div(cosalpha3, d::mul(d::mul(h3, h3), n1norm)) * mHat3) * n1.transpose();
    lps::LMat<S, 3, 3> M3_1_1 =
        (d::div(cosalpha3, d::mul(d::mul(h3, h1), n1norm)) * mHat1) * n1.transpose();
    lps::LMat<S, 3, 3> M1_3_1 =
        (d::div(cosalpha1, d::mul(d::mul(h1, h3), n1norm)) * mHat3) * n1.transpose();
    lps::LMat<S, 3, 3> M4_02_2 =
        (d::div(cosalpha4, d::mul(d::mul(h4, h02), n2norm)) * mHat02) * n2.transpose();
    lps::LMat<S, 3, 3> M2_02_2 =
        (d::div(cosalpha2, d::mul(d::mul(h2, h02), n2norm)) * mHat02) * n2.transpose();
    lps::LMat<S, 3, 3> M4_4_2 =
        (d::div(cosalpha4, d::mul(d::mul(h4, h4), n2norm)) * mHat4) * n2.transpose();
    lps::LMat<S, 3, 3> M2_4_2 =
        (d::div(cosalpha2, d::mul(d::mul(h2, h4), n2norm)) * mHat4) * n2.transpose();
    lps::LMat<S, 3, 3> M4_2_2 =
        (d::div(cosalpha4, d::mul(d::mul(h4, h2), n2norm)) * mHat2) * n2.transpose();
    lps::LMat<S, 3, 3> M2_2_2 =
        (d::div(cosalpha2, d::mul(d::mul(h2, h2), n2norm)) * mHat2) * n2.transpose();
    lps::LMat<S, 3, 3> B1 =
        n1 * (mHat01.transpose() / d::mul(d::mul(norm_e[0], norm_e[0]), n1norm));
    lps::LMat<S, 3, 3> B2 =
        n2 * (mHat02.transpose() / d::mul(d::mul(norm_e[0], norm_e[0]), n2norm));

    // fill in Hessian in order with g2, g0, g1, g3 in rusmus' doc
    // (the four `block(i,j) = block(j,i).transpose()` pairs are written from one
    //  materialised 3x3 value, which is the same value the reference reads back)
    Hess.template block<3, 3>(0, 0) = -(N1_01 + N1_01.transpose());

    const lps::LMat<S, 3, 3> H_3_0 = M3_01_1 - N1_3;
    Hess.template block<3, 3>(3, 0) = H_3_0;
    Hess.template block<3, 3>(0, 3) = H_3_0.transpose();

    const lps::LMat<S, 3, 3> H_6_0 = M1_01_1 - N1_1;
    Hess.template block<3, 3>(6, 0) = H_6_0;
    Hess.template block<3, 3>(0, 6) = H_6_0.transpose();

    Hess.template block<3, 3>(0, 9).setZero();
    Hess.template block<3, 3>(9, 0).setZero();

    Hess.template block<3, 3>(3, 3) = M3_3_1 + M3_3_1.transpose() - B1 + M4_4_2
                                      + M4_4_2.transpose() - B2;

    const lps::LMat<S, 3, 3> H_3_6 =
        M3_1_1 + M1_3_1.transpose() + B1 + M4_2_2 + M2_4_2.transpose() + B2;
    Hess.template block<3, 3>(3, 6) = H_3_6;
    Hess.template block<3, 3>(6, 3) = H_3_6.transpose();

    const lps::LMat<S, 3, 3> H_3_9 = M4_02_2 - N2_4;
    Hess.template block<3, 3>(3, 9) = H_3_9;
    Hess.template block<3, 3>(9, 3) = H_3_9.transpose();

    Hess.template block<3, 3>(6, 6) = M1_1_1 + M1_1_1.transpose() - B1 + M2_2_2
                                      + M2_2_2.transpose() - B2;

    const lps::LMat<S, 3, 3> H_6_9 = M2_02_2 - N2_2;
    Hess.template block<3, 3>(6, 9) = H_6_9;
    Hess.template block<3, 3>(9, 6) = H_6_9.transpose();

    Hess.template block<3, 3>(9, 9) = -(N2_02 + N2_02.transpose());
}
}  // namespace detail

/**
 * @brief Compute the gradient of the dihedral angle between two planes defined by
 * four points (`dtheta/dx`, 12-vector, stencil order x0, x1, x2, x3).
 *
 * Mid-Edge: V1-V2
 * Opposite Points: V0, V3
 */
template <typename S>
inline void dihedral_angle_gradient(const lps::LVec<S, 3>& v0,
                                    const lps::LVec<S, 3>& v1,
                                    const lps::LVec<S, 3>& v2,
                                    const lps::LVec<S, 3>& v3,
                                    lps::LVec<S, 12>&      grad)
{
    detail::dihedral_angle_gradient(v0, v1, v2, v3, grad);
}

template <typename S>
inline void dihedral_angle_hessian(const lps::LVec<S, 3>& v0,
                                   const lps::LVec<S, 3>& v1,
                                   const lps::LVec<S, 3>& v2,
                                   const lps::LVec<S, 3>& v3,
                                   lps::LMat<S, 12, 12>&  Hess)
{
    detail::dihedral_angle_hessian(v0, v1, v2, v3, Hess);
}
}  // namespace uipc::backend::luisa
