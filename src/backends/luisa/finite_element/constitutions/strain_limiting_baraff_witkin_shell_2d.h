#pragma once
/********************************************************************
 * @file strain_limiting_baraff_witkin_shell_2d.h
 * @brief T5.2 (batch b) — the strain-limiting Baraff-Witkin membrane model (port of
 *        src/backends/cuda/finite_element/constitutions/
 *        strain_limiting_baraff_witkin_shell_2d.h).
 *
 * The reference header holds the hand-written 3x2 kinematics (`Ds3x2`, `Dm2x2`,
 * `F3x2`, `flatten`, `dFdX`) and the energy/gradient/Hessian of the anisotropic
 * strain-limiting model.  It is not machine generated, so it is transcribed by hand —
 * with the same two rules the generated files follow: *scalar* arithmetic goes
 * through the `lps::detail::{add,sub,mul,div,neg}` trampolines (RULE M1) while
 * *container* arithmetic keeps the numerics layer's own operators (found by ADL; the
 * trampolines are scalar-only), and the reference's association order is preserved.
 *
 * Mapping / deviations (all reported in the T5.2 batch-b report):
 *  * `template <typename S>` (`S = double` host / `S = lcrt::Var<double>` device)
 *    replaces the reference's fixed `Float`: one implementation, instantiated twice;
 *  * `Eigen::Matrix<Float,3,2>` -> `lps::LMat<S,3,2>`, `Vector6` -> `lps::LVec<S,6>`,
 *    `Matrix<Float,6,6>` -> `lps::LMat<S,6,6>`, `Matrix2x2` -> `lps::LMat<S,2,2>`;
 *  * every `if`/`?:` on a scalar becomes a `select` (RULE M2: a device predicate
 *    cannot drive host control flow), with *both* operand values evaluated — the
 *    discarded one is either finite or a deliberately discarded 0/0 (the `cos + 1 ==
 *    0` antiparallel branch), never a trap;
 *  * a 1x1 Eigen expression (`a^T F^T F b`) is materialised as `lps::LVec<S,1>` and
 *    read with `(0)` — the layer's `matrix * vector` returns a one-element vector, so
 *    the reference's full product chain (and its rounding) is kept instead of being
 *    rewritten into a dot product;
 *  * Eigen comma initialisers (`cross_vec << ...`, the anti-diagonal of the shear `H`)
 *    become one element assignment per entry in Eigen's own row-major traversal
 *    order, and the chained `H(0,0) = H(1,1) = H(2,2) = v` becomes three assignments
 *    of the one value (a DSL `Var` gives no chained-assignment guarantee).
 *
 * `bw_shell_detail` factors the reference's kernels into the generic per-element
 * routines the unit test instantiates too (cookbook §7.4).  NOTE (reference
 * fidelity): unlike the other shell constitutions this one applies **no** `make_spd`
 * — `H = dFdx^T ddEddF dFdx` is used as is, and the reference scales `ddEddF` by
 * `Vdt2` *before* that congruence transform, which the port keeps.
 ********************************************************************/
#include <type_define.h>
#include <finite_element/codim_2d_constitution.h>
#include <luisa_device/math/matrix_ops.h>
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace sym::strainlimiting_baraff_witkin_shell_2d
{
    template <typename S>
    [[nodiscard]] inline lps::LVec<S, 6> flatten(const lps::LMat<S, 3, 2>& F)
    {
        lps::LVec<S, 6> R;
        R.template segment<3>(0) = F.col(0);
        R.template segment<3>(3) = F.col(1);
        return R;
    }

    template <typename S>
    [[nodiscard]] inline lps::LMat<S, 3, 2> F3x2(const lps::LMat<S, 3, 2>& Ds3x2,
                                                 const lps::LMat<S, 2, 2>& Dms2x2_inv)
    {
        return Ds3x2 * Dms2x2_inv;
    }

    template <typename S>
    [[nodiscard]] inline lps::LMat<S, 3, 2> Ds3x2(const lps::LVec<S, 3>& x0,
                                                  const lps::LVec<S, 3>& x1,
                                                  const lps::LVec<S, 3>& x2)
    {
        lps::LMat<S, 3, 2> M;
        M.col(0) = x1 - x0;
        M.col(1) = x2 - x0;
        return M;
    }

    template <typename S>
    [[nodiscard]] inline lps::LMat<S, 2, 2> Dm2x2(const lps::LVec<S, 3>& x0,
                                                  const lps::LVec<S, 3>& x1,
                                                  const lps::LVec<S, 3>& x2)
    {
        namespace d = lps::detail;

        const S zero = lps::make_scalar<S>(0.0);
        const S one  = lps::make_scalar<S>(1.0);

        const lps::LVec<S, 3> v01 = x1 - x0;
        const lps::LVec<S, 3> v02 = x2 - x0;
        // compute uv coordinates by rotating each triangle normal to (0, 0, 1)
        const lps::LVec<S, 3> normal = lps::cross(v01, v02).normalized();
        const lps::LVec<S, 3> target{zero, zero, one};

        const lps::LVec<S, 3> vec = lps::cross(normal, target);
        const S               cos = normal.dot(target);

        // reference (verbatim, one statement per entry in Eigen's row-major
        // comma-initialiser order):
        //     cross_vec << 0, -vec.z(), vec.y(),
        //                  vec.z(), 0, -vec.x(),
        //                 -vec.y(), vec.x(), 0;
        lps::LMat<S, 3, 3> cross_vec;
        cross_vec(0, 0) = zero;
        cross_vec(0, 1) = d::neg(vec(2));
        cross_vec(0, 2) = vec(1);
        cross_vec(1, 0) = vec(2);
        cross_vec(1, 1) = zero;
        cross_vec(1, 2) = d::neg(vec(0));
        cross_vec(2, 0) = d::neg(vec(1));
        cross_vec(2, 1) = vec(0);
        cross_vec(2, 2) = zero;

        // reference: `rotation += cross_vec + cross_vec * cross_vec / (1 + cos);`
        // (evaluated for the general case; the antiparallel branch below selects the
        //  reference's diag(-1, -1, 1) instead -- its 0/0 is discarded, never used.
        //  The sum's association and the *division* are kept verbatim.)
        const auto cc_term = cross_vec + cross_vec * cross_vec / d::add(one, cos);
        const auto rotation_general = lps::LMat<S, 3, 3>::Identity() + cc_term;

        // reference: `if(cos + 1 == 0) { rotation(0, 0) = -1; rotation(1, 1) = -1; }`
        auto rotation_singular  = lps::LMat<S, 3, 3>::Identity();
        rotation_singular(0, 0) = lps::make_scalar<S>(-1.0);
        rotation_singular(1, 1) = lps::make_scalar<S>(-1.0);

        // scselect argument order: `select(f, t, p)` is `p ? t : f` (cookbook §7.6
        // trap 1), so the *general* value is the false operand.
        const auto singular = d::add(cos, one) == zero;
        const auto rotation = scalar_detail::select_container(
            rotation_general, rotation_singular, singular);

        const lps::LVec<S, 3> rotate_uv0 = rotation * x0;
        const lps::LVec<S, 3> rotate_uv1 = rotation * x1;
        const lps::LVec<S, 3> rotate_uv2 = rotation * x2;

        const lps::LVec<S, 2> uv0{rotate_uv0(0), rotate_uv0(1)};
        const lps::LVec<S, 2> uv1{rotate_uv1(0), rotate_uv1(1)};
        const lps::LVec<S, 2> uv2{rotate_uv2(0), rotate_uv2(1)};


        lps::LMat<S, 2, 2> M;
        M.col(0) = uv1 - uv0;
        M.col(1) = uv2 - uv0;

        return M;
    }

    template <typename S>
    [[nodiscard]] inline lps::LMat<S, 6, 9> dFdX(const lps::LMat<S, 2, 2>& DmI)
    {
        namespace d = lps::detail;

        lps::LMat<S, 6, 9> dfdx;
        dfdx.setZero();
        const S d0 = DmI(0, 0);
        const S d1 = DmI(1, 0);
        const S d2 = DmI(0, 1);
        const S d3 = DmI(1, 1);
        const S s0 = d::add(d0, d1);
        const S s1 = d::add(d2, d3);

        for(int i = 0; i < 3; i++)
        {
            dfdx(i, i)     = d::neg(s0);
            dfdx(i + 3, i) = d::neg(s1);
        }

        for(int i = 0; i < 3; i++)
        {
            dfdx(i, i + 3)     = d0;
            dfdx(i + 3, i + 3) = d2;
        }

        for(int i = 0; i < 3; i++)
        {
            dfdx(i, i + 6)     = d1;
            dfdx(i + 3, i + 6) = d3;
        }

        return dfdx;
    }

    template <typename S>
    [[nodiscard]] inline S E(const lps::LMat<S, 3, 2>& F,
                             const lps::LVec<S, 2>&    anisotropic_a,
                             const lps::LVec<S, 2>&    anisotropic_b,
                             const S&                  stretchS,
                             const S&                  shearS,
                             const S&                  strainRate)
    {
        namespace d = lps::detail;

        const S one  = lps::make_scalar<S>(1.0);
        const S zero = lps::make_scalar<S>(0.0);

        // `anisotropic_a.transpose() * F.transpose() * F * anisotropic_b` (1x1)
        const S I6 = (anisotropic_a.transpose() * F.transpose() * F * anisotropic_b)(0);
        const S shear_energy = d::mul(I6, I6);

        const S I5u = (F * anisotropic_a).norm();
        const S I5v = (F * anisotropic_b).norm();

        // reference: `Float ucoeff = 1; ... if(I5u <= 1) { ucoeff = 0; }`
        // (`select(f, t, p)` == `p ? t : f`, so 1 is the false operand)
        const S ucoeff = scalar_detail::select_scalar(one, zero, I5u <= one);
        const S vcoeff = scalar_detail::select_scalar(one, zero, I5v <= one);

        const S stretch_energy =
            d::add(d::add(d::add(lps::pow(d::sub(I5u, one), 2.0),
                                 d::mul(d::mul(ucoeff, strainRate),
                                        lps::pow(d::sub(I5u, one), 3.0))),
                          lps::pow(d::sub(I5v, one), 2.0)),
                   d::mul(d::mul(vcoeff, strainRate), lps::pow(d::sub(I5v, one), 3.0)));

        return d::add(d::mul(stretchS, stretch_energy), d::mul(shearS, shear_energy));
    }

    template <typename S>
    inline void dEdF(lps::LMat<S, 3, 2>&       R,
                     const lps::LMat<S, 3, 2>& F,
                     const lps::LVec<S, 2>&    anisotropic_a,
                     const lps::LVec<S, 2>&    anisotropic_b,
                     const S&                  stretchS,
                     const S&                  shearS,
                     const S&                  strainRate)
    {
        namespace d = lps::detail;

        const S one  = lps::make_scalar<S>(1.0);
        const S zero = lps::make_scalar<S>(0.0);
        const S two  = lps::make_scalar<S>(2.0);

        const S I6 = (anisotropic_a.transpose() * F.transpose() * F * anisotropic_b)(0);

        // reference: `shear_pk1 = 2 * (I6 - anisotropic_a^T anisotropic_b)
        //             * (F * a * b^T + F * b * a^T);`
        const lps::LMat<S, 3, 2> shear_pk1 =
            d::mul(two, d::sub(I6, (anisotropic_a.transpose() * anisotropic_b)(0)))
            * ((F * anisotropic_a) * anisotropic_b.transpose()
               + (F * anisotropic_b) * anisotropic_a.transpose());

        // guard the 1/sqrt(I5) singularity: a fully collapsed triangle
        // (stretch -> 0) must not produce NaN/Inf (same hole exists in
        // Stiff-GIPC; finite huge response is strictly better)
        const S I5u = lps::max((F * anisotropic_a).squaredNorm(),
                               lps::make_scalar<S>(1e-12));
        const S I5v = lps::max((F * anisotropic_b).squaredNorm(),
                               lps::make_scalar<S>(1e-12));

        const S sqrtI5u = lps::sqrt(I5u);
        const S sqrtI5v = lps::sqrt(I5v);

        const S ucoeff_base = d::sub(one, d::div(one, sqrtI5u));
        const S vcoeff_base = d::sub(one, d::div(one, sqrtI5v));

        // reference: `if(I5u > 1) { ucoeff += 1.5 * strainRate
        //                         * (sqrt(I5u) + 1 / sqrt(I5u) - 2); }`
        const S ucoeff_extra = lps::make_scalar<S>(1.5) * strainRate
                               * d::sub(d::add(sqrtI5u, d::div(one, sqrtI5u)), two);
        const S vcoeff_extra = lps::make_scalar<S>(1.5) * strainRate
                               * d::sub(d::add(sqrtI5v, d::div(one, sqrtI5v)), two);

        const S ucoeff = d::add(ucoeff_base,
                               scalar_detail::select_scalar(zero, ucoeff_extra, I5u > one));
        const S vcoeff = d::add(vcoeff_base,
                               scalar_detail::select_scalar(zero, vcoeff_extra, I5v > one));

        // reference: `stretch_pk1 = ucoeff * Float{2} * F * a * a^T
        //                         + vcoeff * Float{2} * F * b * b^T;`
        const lps::LMat<S, 3, 2> stretch_pk1 =
            ((d::mul(ucoeff, two) * F) * anisotropic_a) * anisotropic_a.transpose()
            + ((d::mul(vcoeff, two) * F) * anisotropic_b) * anisotropic_b.transpose();

        R = stretchS * stretch_pk1 + shearS * shear_pk1;
    }

    template <typename S>
    inline void ddEddF(lps::LMat<S, 6, 6>&       R,
                       const lps::LMat<S, 3, 2>&  F,
                       const lps::LVec<S, 2>&     anisotropic_a,
                       const lps::LVec<S, 2>&     anisotropic_b,
                       const S&                   stretchS,
                       const S&                   shearS,
                       const S&                   strainRate)
    {
        namespace d = lps::detail;

        const S one   = lps::make_scalar<S>(1.0);
        const S zero  = lps::make_scalar<S>(0.0);
        const S two   = lps::make_scalar<S>(2.0);
        const S three = lps::make_scalar<S>(3.0);

        lps::LMat<S, 6, 6> H_stretc = lps::LMat<S, 6, 6>::Zero();
        {
            // same 1/sqrt(I5) singularity guard as dEdF
            const S I5u     = lps::max((F * anisotropic_a).squaredNorm(),
                                       lps::make_scalar<S>(1e-12));
            const S I5v     = lps::max((F * anisotropic_b).squaredNorm(),
                                       lps::make_scalar<S>(1e-12));
            const S sqrtI5u = lps::sqrt(I5u);
            const S sqrtI5v = lps::sqrt(I5v);

            // reference:
            //   if(sqrtI5u > 1) { H_stretc(0,0) = H_stretc(1,1) = H_stretc(2,2) =
            //       2 * (((sqrtI5u - 1) * (3 * sqrtI5u * strainRate - 3 * strainRate + 2))
            //            / (2 * sqrtI5u)); }
            const S stiff_u =
                two
                * d::div(d::mul(d::sub(sqrtI5u, one),
                                d::add(d::sub(d::mul(d::mul(three, sqrtI5u), strainRate),
                                              d::mul(three, strainRate)),
                                       two)),
                         d::mul(two, sqrtI5u));
            const S stiff_v =
                two
                * d::div(d::mul(d::sub(sqrtI5v, one),
                                d::add(d::sub(d::mul(d::mul(three, sqrtI5v), strainRate),
                                              d::mul(three, strainRate)),
                                       two)),
                         d::mul(two, sqrtI5v));

            const S diag_u = scalar_detail::select_scalar(zero, stiff_u, sqrtI5u > one);
            const S diag_v = scalar_detail::select_scalar(zero, stiff_v, sqrtI5v > one);

            H_stretc(0, 0) = diag_u;
            H_stretc(1, 1) = diag_u;
            H_stretc(2, 2) = diag_u;
            H_stretc(3, 3) = diag_v;
            H_stretc(4, 4) = diag_v;
            H_stretc(5, 5) = diag_v;

            const lps::LVec<S, 3> fu = F.col(0).normalized();
            const lps::LVec<S, 3> fv = F.col(1).normalized();

            // reference:
            //   Float uCoeff = (sqrtI5u > Float{1.0}) ?
            //       (3 * I5u * strainRate - 3 * strainRate + 2) / (sqrt(I5u)) : 2.0;
            const S uCoeff =
                scalar_detail::select_scalar(
                    two,
                    d::div(d::add(d::sub(d::mul(d::mul(three, I5u), strainRate),
                                         d::mul(three, strainRate)),
                                  two),
                           sqrtI5u),
                    sqrtI5u > one);
            const S vCoeff =
                scalar_detail::select_scalar(
                    two,
                    d::div(d::add(d::sub(d::mul(d::mul(three, I5v), strainRate),
                                         d::mul(three, strainRate)),
                                  two),
                           sqrtI5v),
                    sqrtI5v > one);


            H_stretc.block<3, 3>(0, 0) =
                H_stretc.block<3, 3>(0, 0) + uCoeff * (fu * fu.transpose());
            H_stretc.block<3, 3>(3, 3) =
                H_stretc.block<3, 3>(3, 3) + vCoeff * (fv * fv.transpose());
        }

        lps::LMat<S, 6, 6> H_shear = lps::LMat<S, 6, 6>::Zero();
        {
            // reference: `H(3,0) = H(4,1) = H(5,2) = H(0,3) = H(1,4) = H(2,5) = 1.0;`
            lps::LMat<S, 6, 6> H = lps::LMat<S, 6, 6>::Zero();
            H(3, 0)              = one;
            H(4, 1)              = one;
            H(5, 2)              = one;
            H(0, 3)              = one;
            H(1, 4)              = one;
            H(2, 5)              = one;

            const S I6 = (anisotropic_a.transpose() * F.transpose() * F * anisotropic_b)(0);
            // reference: `Float signI6 = (I6 >= 0) ? 1.0 : -1.0;`
            const S signI6 =
                scalar_detail::select_scalar(lps::make_scalar<S>(-1.0), one, I6 >= zero);

            const lps::LMat<S, 3, 2> g =
                F
                * (anisotropic_a * anisotropic_b.transpose()
                   + anisotropic_b * anisotropic_a.transpose());

            lps::LVec<S, 6> vec_g = lps::LVec<S, 6>::Zero();
            vec_g.template segment<3>(0) = g.col(0);
            vec_g.template segment<3>(3) = g.col(1);

            const S I2      = F.squaredNorm();
            const S lambda0 = d::mul(lps::make_scalar<S>(0.5),
                                     d::add(I2,
                                            lps::sqrt(d::add(d::mul(I2, I2),
                                                             d::mul(d::mul(lps::make_scalar<S>(12.0), I6),
                                                                    I6)))));
            // reference: `(I6 * H * vec_g + lambda0 * vec_g).normalized()` — the
            // scalar/container mixes keep the layer's own operators (the trampolines
            // are scalar-only) and the association `(I6 * H) * vec_g`.
            const lps::LVec<S, 6> q0 =
                ((I6 * H) * vec_g + lambda0 * vec_g).normalized();

            lps::LMat<S, 6, 6> T = lps::LMat<S, 6, 6>::Identity();
            T = lps::make_scalar<S>(0.5) * (T + signI6 * H);

            const auto Tq     = T * q0;
            const S    normTq = Tq.squaredNorm();
            H_shear = lps::abs(I6) * (T - (Tq * Tq.transpose()) / normTq)
                      + lambda0 * (q0 * q0.transpose());
            H_shear = H_shear * two;
        }

        R = stretchS * H_stretc + shearS * H_shear;
    }
}  // namespace sym::strainlimiting_baraff_witkin_shell_2d

namespace bw_shell_detail
{
/**
 * @brief The reference's `..._do_init_kernel` body: the inverse of the triangle's
 *        `Dm2x2` rest shape matrix.
 */
template <typename S>
[[nodiscard]] inline lps::LMat<S, 2, 2> rest_metric_inv(const lps::LVec<S, 3>& X0,
                                                        const lps::LVec<S, 3>& X1,
                                                        const lps::LVec<S, 3>& X2)
{
    namespace BWS = sym::strainlimiting_baraff_witkin_shell_2d;
    auto Dm       = BWS::Dm2x2(X0, X1, X2);
    // reference: `inv_Bs(I) = cuda_tool::eigen::inverse(Dm);`
    return lps::inverse(Dm);
}

/**
 * @brief The reference's `..._do_compute_energy_kernel` body.
 *
 * `V = rest_area` (the membrane element weight is the triangle *area*: the stiffness
 * attributes carry their own thickness conventions), and the anisotropic directions
 * are the reference's fixed `(1, 0)` / `(0, 1)`.
 */
template <typename S>
[[nodiscard]] inline S element_energy(const S&                  lambda,
                                      const S&                  mu,
                                      const S&                  strain_rate,
                                      const S&                  rest_area,
                                      const lps::LVec<S, 3>&    x0,
                                      const lps::LVec<S, 3>&    x1,
                                      const lps::LVec<S, 3>&    x2,
                                      const lps::LMat<S, 2, 2>& IB,
                                      const S&                  dt)
{
    namespace BWS = sym::strainlimiting_baraff_witkin_shell_2d;
    namespace d   = lps::detail;

    auto Ds = BWS::Ds3x2(x0, x1, x2);
    auto F  = Ds * IB;

    const lps::LVec<S, 2> anisotropic_a{lps::make_scalar<S>(1.0), lps::make_scalar<S>(0.0)};
    const lps::LVec<S, 2> anisotropic_b{lps::make_scalar<S>(0.0), lps::make_scalar<S>(1.0)};

    const S V = rest_area;

    auto E = BWS::E(F, anisotropic_a, anisotropic_b, lambda, mu, strain_rate);
    // reference: `energies(I) = E * V * dt * dt;`
    return d::mul(d::mul(d::mul(E, V), dt), dt);
}

/// `Float Vdt2 = V * dt * dt;` (the reference's association order).
template <typename S>
[[nodiscard]] inline S element_vdt2(const S& V, const S& dt)
{
    namespace d = lps::detail;
    return d::mul(d::mul(V, dt), dt);
}

/// `..._do_compute_gradient_hessian_kernel`, gradient part:
/// `dFdx^T * flatten(dEdF); G *= Vdt2;`
template <typename S>
[[nodiscard]] inline lps::LVec<S, 9> element_gradient(const S&                  lambda,
                                                      const S&                  mu,
                                                      const S&                  strain_rate,
                                                      const S&                  rest_area,
                                                      const lps::LVec<S, 3>&    x0,
                                                      const lps::LVec<S, 3>&    x1,
                                                      const lps::LVec<S, 3>&    x2,
                                                      const lps::LMat<S, 2, 2>& IB,
                                                      const S&                  dt)
{
    namespace BWS = sym::strainlimiting_baraff_witkin_shell_2d;

    auto Ds = BWS::Ds3x2(x0, x1, x2);
    auto F  = Ds * IB;

    const lps::LVec<S, 2> anisotropic_a{lps::make_scalar<S>(1.0), lps::make_scalar<S>(0.0)};
    const lps::LVec<S, 2> anisotropic_b{lps::make_scalar<S>(0.0), lps::make_scalar<S>(1.0)};

    auto dFdx = BWS::dFdX(IB);

    const S V = rest_area;

    lps::LMat<S, 3, 2> dEdF;
    BWS::dEdF(dEdF, F, anisotropic_a, anisotropic_b, lambda, mu, strain_rate);

    auto VecdEdF = BWS::flatten(dEdF);

    lps::LVec<S, 9> G = dFdx.transpose() * VecdEdF;

    return G * element_vdt2(V, dt);
}

/**
 * @brief The Hessian part: `ddEddF *= Vdt2; H = dFdx^T ddEddF dFdx;` with **no**
 *        `make_spd` (the reference applies none here) and the `Vdt2` scale applied
 *        *before* the congruence transform.
 */
template <typename S>
[[nodiscard]] inline lps::LMat<S, 9, 9> element_hessian(const S&                  lambda,
                                                        const S&                  mu,
                                                        const S&                  strain_rate,
                                                        const S&                  rest_area,
                                                        const lps::LVec<S, 3>&    x0,
                                                        const lps::LVec<S, 3>&    x1,
                                                        const lps::LVec<S, 3>&    x2,
                                                        const lps::LMat<S, 2, 2>& IB,
                                                        const S&                  dt)
{
    namespace BWS = sym::strainlimiting_baraff_witkin_shell_2d;

    auto Ds = BWS::Ds3x2(x0, x1, x2);
    auto F  = Ds * IB;

    const lps::LVec<S, 2> anisotropic_a{lps::make_scalar<S>(1.0), lps::make_scalar<S>(0.0)};
    const lps::LVec<S, 2> anisotropic_b{lps::make_scalar<S>(0.0), lps::make_scalar<S>(1.0)};

    auto dFdx = BWS::dFdX(IB);

    const S V = rest_area;

    lps::LMat<S, 6, 6> ddEddF;
    BWS::ddEddF(ddEddF, F, anisotropic_a, anisotropic_b, lambda, mu, strain_rate);

    ddEddF = ddEddF * element_vdt2(V, dt);

    return dFdx.transpose() * ddEddF * dFdx;
}
}  // namespace bw_shell_detail

/**
 * @brief The constitution itself (reference: the class at the bottom of
 *        strain_limiting_baraff_witkin_shell_2d.cu).
 *
 * DEVIATION (reported — the same one T5.2 batch a introduced, cookbook §7.17 rule 11):
 * the reference declares this class in its .cu; the port declares it here and keeps the
 * kernels + out-of-line members + the registration in the .cpp, so the unit test can
 * reach the reference's *public* surface (the parameter buffers and the five hooks) and
 * drive the real constitution end to end.
 */
class StrainLimitingBaraffWitkinShell2D final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 819;
    static constexpr SizeT StencilSize     = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using Codim2DConstitution::Codim2DConstitution;

    vector<Float> h_mus;
    vector<Float> h_lambdas;
    vector<Float> h_strain_rates;

    lud::LDeviceBuffer<Float>      mus;
    lud::LDeviceBuffer<Float>      lambdas;
    lud::LDeviceBuffer<Float>      strain_rates;
    lud::LDeviceBuffer<LMatrix2x2> inv_B_matrices;

    SimSystemSlot<FiniteElementMethod> fem;


    virtual U64 get_uid() const noexcept override;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override;

    virtual void do_report_extent(ReportExtentInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
