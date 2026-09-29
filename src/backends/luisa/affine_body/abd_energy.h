#pragma once
/********************************************************************
 * @file affine_body/abd_energy.h
 * @brief T6.1a — the ABD shape ("perpendicular volume") energy
 * (port of src/backends/cuda/affine_body/abd_energy.{h,cu}).
 *
 *tex:
 * $$ S = \frac{V_{\perp}}{\kappa v} $$
 * (the caller supplies kappa and v; the reference does not include them)
 *tex:
 * $$ \frac{V_{\perp}}{\kappa v}
 *    = \sum_i (a_i \cdot a_i - 1)^2 + \sum_{i \neq j} (a_i \cdot a_j)^2 $$
 *
 * DEVIATION (reported): the reference's abd_energy.cu is `UIPC_GENERIC` and holds
 * the bodies; this backend's DSL needs the body at the call site, so -- exactly
 * like T5.1's fem_utils.h/matrix_utils.h (cookbook §7.16 rule 1) -- the bodies are
 * inline in this header, templated on the scalar (`S` = `double` on the host,
 * `lcrt::Var<double>` in a kernel).  abd_energy.cpp then holds the reference's
 * host-callable Eigen-typed signatures (`Float shape_energy(const Vector12&)`, ...)
 * as thin forwards to the same template with `S = Float`, so a host call site
 * reads exactly like the reference and host/device share one arithmetic body.
 *
 * Detail deviations, both forced by the layer (cookbook §7.16 rules 3/10):
 *  * `ddV_ddai`/`ddV_daidaj` take `lps::LVec<S,3>` instead of Eigen's
 *    `Eigen::VectorBlock<const Vector12,3>` (a block has no portable spelling);
 *    this *loosens* the signature -- `q.segment<3>(i)` on a const LVec already
 *    returns an `LVec<S,3>` by value;
 *  * scalar-scalar accumulations go through `lps::detail::{add,sub,mul}`, and
 *    every literal is a double: a DSL scalar has no `operator+=`, and LC
 *    deprecates fp/integral implicit conversions.
 *********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/math.h>

namespace uipc::backend::luisa
{
namespace abd_energy_detail
{
/// reference: `constexpr auto square = [](auto x) { return x * x; };`
template <class X>
[[nodiscard]] inline auto square(const X& x) noexcept
{
    return lps::detail::mul(x, x);
}
}  // namespace abd_energy_detail

// ---------------------------------------------------------------------------
//  Hessian blocks (reference: `ddV_ddai` / `ddV_daidaj`)
// ---------------------------------------------------------------------------
///tex: $$ \frac{\partial^2 V}{\partial a_i^2} $$
template <class S>
inline void ddV_ddai(lps::LMat<S, 3, 3>&    ddV_ddai,
                     const lps::LVec<S, 3>& ai,
                     const lps::LVec<S, 3>& aj,
                     const lps::LVec<S, 3>& ak) noexcept
{
    // reference: 8.0 * ai * ai.transpose() + 4.0 * (ai.squaredNorm() - 1) * I
    //          + 4.0 * aj * aj.transpose() + 4.0 * ak * ak.transpose()
    ddV_ddai = (lps::make_scalar<S>(8.0) * (ai * ai.transpose()))
               + ((lps::make_scalar<S>(4.0)
                   * lps::detail::sub(lps::squaredNorm(ai), lps::make_scalar<S>(1.0)))
                  * lps::LMat<S, 3, 3>::Identity())
               + (lps::make_scalar<S>(4.0) * (aj * aj.transpose()))
               + (lps::make_scalar<S>(4.0) * (ak * ak.transpose()));
}

///tex: $$ \frac{\partial^2 V}{\partial a_i \partial a_j} $$
template <class S>
inline void ddV_daidaj(lps::LMat<S, 3, 3>&  ddV_daidaj,
                       const lps::LVec<S, 3>& ai,
                       const lps::LVec<S, 3>& aj,
                       const lps::LVec<S, 3>& ak) noexcept
{
    (void)ak;  // the reference's `ak` parameter is unused
    // reference: 4.0 * aj * ai.transpose() + 4.0 * ai.dot(aj) * I
    ddV_daidaj = (lps::make_scalar<S>(4.0) * (aj * ai.transpose()))
                 + ((lps::make_scalar<S>(4.0) * lps::dot(ai, aj))
                    * lps::LMat<S, 3, 3>::Identity());
}

// ---------------------------------------------------------------------------
//  Energy
// ---------------------------------------------------------------------------
///tex: $$ \frac{V_{\perp}}{\kappa v} = \sum_i (a_i\cdot a_i - 1)^2
///      + \sum_{i\neq j} (a_i \cdot a_j)^2 $$
template <class S>
[[nodiscard]] inline S shape_energy(const lps::LVec<S, 12>& q) noexcept
{
    using lps::detail::add;
    using lps::detail::mul;

    S ret = lps::zero_of<S>();

    const auto a1 = q.segment<3>(3);
    const auto a2 = q.segment<3>(6);
    const auto a3 = q.segment<3>(9);

    ret = add(ret,
              abd_energy_detail::square(
                  lps::detail::sub(lps::squaredNorm(a1), lps::make_scalar<S>(1.0))));
    ret = add(ret,
              abd_energy_detail::square(
                  lps::detail::sub(lps::squaredNorm(a2), lps::make_scalar<S>(1.0))));
    ret = add(ret,
              abd_energy_detail::square(
                  lps::detail::sub(lps::squaredNorm(a3), lps::make_scalar<S>(1.0))));

    ret = add(ret,
              mul(abd_energy_detail::square(lps::dot(a1, a2)), lps::make_scalar<S>(2.0)));
    ret = add(ret,
              mul(abd_energy_detail::square(lps::dot(a2, a3)), lps::make_scalar<S>(2.0)));
    ret = add(ret,
              mul(abd_energy_detail::square(lps::dot(a3, a1)), lps::make_scalar<S>(2.0)));

    return ret;
}

// ---------------------------------------------------------------------------
//  Gradient
// ---------------------------------------------------------------------------
///tex: $$ \frac{1}{\kappa v}\frac{\partial V_{\perp}}{\partial a_i}
///      = 2 \left(2(a_i\cdot a_i - 1) a_i + \sum_j a_j (a_j \cdot a_i)\right) $$
template <class S>
[[nodiscard]] inline lps::LVec<S, 9> shape_energy_gradient(const lps::LVec<S, 12>& q) noexcept
{
    lps::LVec<S, 9> ret;

    const auto a1 = q.segment<3>(3);
    const auto a2 = q.segment<3>(6);
    const auto a3 = q.segment<3>(9);

    auto dEda1 = ret.segment<3>(0);
    auto dEda2 = ret.segment<3>(3);
    auto dEda3 = ret.segment<3>(6);

    dEda1 = (lps::make_scalar<S>(4.0)
             * lps::detail::sub(lps::squaredNorm(a1), lps::make_scalar<S>(1.0)))
                * a1
            + (lps::make_scalar<S>(4.0) * lps::dot(a2, a1)) * a2
            + (lps::make_scalar<S>(4.0) * lps::dot(a3, a1)) * a3;

    dEda2 = (lps::make_scalar<S>(4.0)
             * lps::detail::sub(lps::squaredNorm(a2), lps::make_scalar<S>(1.0)))
                * a2
            + (lps::make_scalar<S>(4.0) * lps::dot(a3, a2)) * a3
            + (lps::make_scalar<S>(4.0) * lps::dot(a1, a2)) * a1;

    dEda3 = (lps::make_scalar<S>(4.0)
             * lps::detail::sub(lps::squaredNorm(a3), lps::make_scalar<S>(1.0)))
                * a3
            + (lps::make_scalar<S>(4.0) * lps::dot(a1, a3)) * a1
            + (lps::make_scalar<S>(4.0) * lps::dot(a2, a3)) * a2;

    return ret;
}

// ---------------------------------------------------------------------------
//  Hessian (the reference's 6-out-parameter form first, then the assembled 9x9)
// ---------------------------------------------------------------------------
template <class S>
inline void shape_energy_hessian(const lps::LVec<S, 12>& q,
                                 lps::LMat<S, 3, 3>&     ddVdda1,
                                 lps::LMat<S, 3, 3>&     ddVdda2,
                                 lps::LMat<S, 3, 3>&     ddVdda3,
                                 lps::LMat<S, 3, 3>&     ddVda1da2,
                                 lps::LMat<S, 3, 3>&     ddVda1da3,
                                 lps::LMat<S, 3, 3>&     ddVda2da3) noexcept
{
    const auto a1 = q.segment<3>(3);
    const auto a2 = q.segment<3>(6);
    const auto a3 = q.segment<3>(9);

    ddV_ddai(ddVdda1, a1, a2, a3);
    ddV_ddai(ddVdda2, a2, a3, a1);
    ddV_ddai(ddVdda3, a3, a1, a2);

    ddV_daidaj(ddVda1da2, a1, a2, a3);
    ddV_daidaj(ddVda1da3, a1, a3, a2);
    ddV_daidaj(ddVda2da3, a2, a3, a1);
}

///tex:
/// $$ \begin{bmatrix}
/// \partial^2 V/\partial a_1^2 & \partial^2 V/\partial a_1 \partial a_2 & \partial^2 V/\partial a_1 \partial a_3\\
/// \partial^2 V/\partial a_2 \partial a_1 & \partial^2 V/\partial a_2^2 & \partial^2 V/\partial a_2 \partial a_3\\
/// \partial^2 V/\partial a_3 \partial a_1 & \partial^2 V/\partial a_3 \partial a_2 & \partial^2 V/\partial a_3^2
/// \end{bmatrix} $$
template <class S>
[[nodiscard]] inline lps::LMat<S, 9, 9> shape_energy_hessian(const lps::LVec<S, 12>& q) noexcept
{
    lps::LMat<S, 9, 9> H = lps::LMat<S, 9, 9>::Zero();

    lps::LMat<S, 3, 3> ddVdda1, ddVdda2, ddVdda3, ddVda1da2, ddVda1da3, ddVda2da3;
    shape_energy_hessian(q, ddVdda1, ddVdda2, ddVdda3, ddVda1da2, ddVda1da3, ddVda2da3);

    H.block<3, 3>(0, 0) = ddVdda1;
    H.block<3, 3>(0, 3) = ddVda1da2;
    H.block<3, 3>(0, 6) = ddVda1da3;

    H.block<3, 3>(3, 0) = ddVda1da2.transpose();
    H.block<3, 3>(3, 3) = ddVdda2;
    H.block<3, 3>(3, 6) = ddVda2da3;

    H.block<3, 3>(6, 0) = ddVda1da3.transpose();
    H.block<3, 3>(6, 3) = ddVda2da3.transpose();
    H.block<3, 3>(6, 6) = ddVdda3;

    return H;
}

// ---------------------------------------------------------------------------
//  Host-facing (Eigen) overloads with the reference's exact signatures.
//  Defined in abd_energy.cpp; each one forwards to the template with S = Float,
//  so there is exactly one arithmetic body per quantity.
// ---------------------------------------------------------------------------
[[nodiscard]] Float shape_energy(const Vector12& q) noexcept;
[[nodiscard]] Vector9 shape_energy_gradient(const Vector12& q) noexcept;
void shape_energy_hessian(const Vector12& q,
                          Matrix3x3&      ddVdda1,
                          Matrix3x3&      ddVdda2,
                          Matrix3x3&      ddVdda3,
                          Matrix3x3&      ddVda1da2,
                          Matrix3x3&      ddVda1da3,
                          Matrix3x3&      ddVda2da3) noexcept;
[[nodiscard]] Matrix9x9 shape_energy_hessian(const Vector12& q) noexcept;
}  // namespace uipc::backend::luisa
