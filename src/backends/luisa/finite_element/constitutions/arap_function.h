#pragma once
/********************************************************************
 * @file arap_function.h
 * @brief T5.2 (batch a) — the ARAP energy/gradient/Hessian (port of
 *        src/backends/cuda/finite_element/constitutions/arap_function.h).
 *
 * Generic over the scalar type exactly like the rest of the layer:
 *   T = double            -> host evaluation (the reference's Float)
 *   T = lcrt::Var<double> -> device evaluation (a DSL local)
 *   T = lcrt::Expr<double>-> either (the numerics layer's trampolines)
 *
 * Mapping notes (the reference is hand written Eigen, not generated output):
 *   `svd(F, U, Sigma, V)`        -> `lps::svd` (the signed SVD of T2.2, same
 *                                   convention: U/V proper rotations, the
 *                                   reflection of `det F < 0` absorbed into
 *                                   Sigma(2); see cookbook §7.5)
 *   `.squaredNorm()`, `.col(i)`,
 *   `.transpose()`, `.setIdentity()` -> the numerics layer's own members
 *   scalar arithmetic            -> the `lps::detail::{add,sub,mul,div,neg}`
 *                                   trampolines (RULE M1)
 *   `1 / sqrt2`                  -> the reference's `constexpr double sqrt2`
 *                                   (a host constant multiplied into a
 *                                   container, exactly as in the reference)
 *
 * `ARAP3D`'s kernels are factored into the generic per-element routines of
 * `arap_3d_detail` (the reference's kernel bodies minus the buffer plumbing), so
 * the kernel and the unit test instantiate *the same* code (cookbook §7.4).
 ********************************************************************/
#include <type_define.h>
#include <finite_element/matrix_utils.h>
#include <finite_element/fem_utils.h>

namespace uipc::backend::luisa
{
namespace sym::arap_3d
{
inline constexpr double sqrt2 =
    1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276;

/// Function to compute the ARAP energy
template <typename T>
inline void E(T&                            energy,
              const T&                      kappa,
              const T&                      v,
              const lps::LMat<T, 3, 3>& F)
{
    lps::LMat<T, 3, 3> R;
    lps::LMat<T, 3, 3> U, V;
    lps::LVec<T, 3>    Sigma;
    lps::svd(F, U, Sigma, V);
    R      = U * V.transpose();
    energy = lps::detail::mul(lps::detail::mul(kappa, v), (F - R).squaredNorm());
}

/// Function to compute the gradient of the ARAP energy
template <typename T>
inline void dEdF(lps::LVec<T, 9>&       gradients,
                 const T&                kappa,
                 const T&                v,
                 const lps::LMat<T, 3, 3>& F)
{
    lps::LMat<T, 3, 3> R;
    lps::LMat<T, 3, 3> U, V;
    lps::LVec<T, 3>    Sigma;
    lps::svd(F, U, Sigma, V);
    R = U * V.transpose();

    // dPsi_dF = 2 * (F - R)   (the container product; the reference writes `2 * (F - R)`)
    lps::LMat<T, 3, 3> dPsi_dF = (F - R) * 2.0;

    T kv = lps::detail::mul(kappa, v);

    gradients.segment<3>(0) = dPsi_dF.col(0) * kv;
    gradients.segment<3>(3) = dPsi_dF.col(1) * kv;
    gradients.segment<3>(6) = dPsi_dF.col(2) * kv;
}

/// Function to flatten a matrix into a vector (the reference's `vector << ...`
/// comma initialiser, expanded into one element assignment per entry)
template <typename T>
[[nodiscard]] inline lps::LVec<T, 9> vec(const lps::LMat<T, 3, 3>& mat)
{
    lps::LVec<T, 9> vector;
    vector(0) = mat(0, 0);
    vector(1) = mat(1, 0);
    vector(2) = mat(2, 0);
    vector(3) = mat(0, 1);
    vector(4) = mat(1, 1);
    vector(5) = mat(2, 1);
    vector(6) = mat(0, 2);
    vector(7) = mat(1, 2);
    vector(8) = mat(2, 2);
    return vector;
}

/// Function to compute the ARAP Hessian
template <typename T>
inline void ARAP_Hessian(lps::LMat<T, 9, 9>&       H,
                         const lps::LMat<T, 3, 3>& F)
{
    lps::LMat<T, 3, 3> U, V;
    lps::LVec<T, 3>    Sigma;
    lps::svd(F, U, Sigma, V);
    // Define the twist modes
    lps::LMat<T, 3, 3> T0, T1, T2;
    T0.row(0) = lps::LVec<T, 3>{0.0, -1.0, 0.0};
    T0.row(1) = lps::LVec<T, 3>{1.0, 0.0, 0.0};
    T0.row(2) = lps::LVec<T, 3>{0.0, 0.0, 0.0};
    T1.row(0) = lps::LVec<T, 3>{0.0, 0.0, 0.0};
    T1.row(1) = lps::LVec<T, 3>{0.0, 0.0, 1.0};
    T1.row(2) = lps::LVec<T, 3>{0.0, -1.0, 0.0};
    T2.row(0) = lps::LVec<T, 3>{0.0, 0.0, 1.0};
    T2.row(1) = lps::LVec<T, 3>{0.0, 0.0, 0.0};
    T2.row(2) = lps::LVec<T, 3>{-1.0, 0.0, 0.0};

    const auto inv_sqrt2 = 1.0 / sqrt2;
    T0                   = inv_sqrt2 * U * T0 * V.transpose();
    T1                   = inv_sqrt2 * U * T1 * V.transpose();
    T2                   = inv_sqrt2 * U * T2 * V.transpose();

    // Flatten the twist modes
    lps::LVec<T, 9> t0 = vec(T0);
    lps::LVec<T, 9> t1 = vec(T1);
    lps::LVec<T, 9> t2 = vec(T2);

    // Get the singular values
    T s0 = Sigma(0);
    T s1 = Sigma(1);
    T s2 = Sigma(2);

    // Compute the Hessian
    H = lps::LMat<T, 9, 9>::Identity() * 2.0;
    H -= lps::detail::div(4.0, lps::detail::add(s0, s1)) * (t0 * t0.transpose());
    H -= lps::detail::div(4.0, lps::detail::add(s1, s2)) * (t1 * t1.transpose());
    H -= lps::detail::div(4.0, lps::detail::add(s0, s2)) * (t2 * t2.transpose());
}

/// Function to compute the Hessian of the ARAP energy
template <typename T>
inline void ddEddF(lps::LMat<T, 9, 9>&       hessians,
                   const T&                   kappa,
                   const T&                   v,
                   const lps::LMat<T, 3, 3>& F)
{
    ARAP_Hessian(hessians, F);
    hessians *= lps::detail::mul(kappa, v);
}
}  // namespace sym::arap_3d

namespace arap_3d_detail
{
/**
 * @brief The reference's `ARAP3D_do_compute_energy_kernel` body, factored out.
 *
 * `ARAP3D`'s kernel reads the tet, the positions and `Dm^-1` and then does exactly
 * this; the test instantiates the same routine for `S = double` (host oracle) and
 * `S = lcrt::Var<Float>` (device).
 */
template <typename S>
[[nodiscard]] inline S tet_energy(const S&                  kappa,
                                  const S&                  dt,
                                  const S&                  volume,
                                  const lps::LVec<S, 3>&    x0,
                                  const lps::LVec<S, 3>&    x1,
                                  const lps::LVec<S, 3>&    x2,
                                  const lps::LVec<S, 3>&    x3,
                                  const lps::LMat<S, 3, 3>& Dm_inv)
{
    auto F = fem::F(x0, x1, x2, x3, Dm_inv);

    S E;
    // reference: `ARAP::E(E, kappas(I) * dt * dt, volumes(I), F);`
    sym::arap_3d::E(E, lps::detail::mul(lps::detail::mul(kappa, dt), dt), volume, F);
    return E;
}

/// The reference's `dFdx.transpose() * dEdF` step of the gradient/Hessian kernel:
/// the local (4-stencil x 3) gradient, flattened as `G12 = dFdx^T dEdF`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> tet_gradient(const S&                  kappa,
                                                   const S&                  dt,
                                                   const S&                  volume,
                                                   const lps::LVec<S, 3>&    x0,
                                                   const lps::LVec<S, 3>&    x1,
                                                   const lps::LVec<S, 3>&    x2,
                                                   const lps::LVec<S, 3>&    x3,
                                                   const lps::LMat<S, 3, 3>& Dm_inv)
{
    auto F = fem::F(x0, x1, x2, x3, Dm_inv);

    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);
    auto v   = volume;

    lps::LVec<S, 9> dEdF;
    sym::arap_3d::dEdF(dEdF, kt2, v, F);

    lps::LMat<S, 9, 12> dFdx = fem::dFdx(Dm_inv);

    return dFdx.transpose() * dEdF;
}

/// The reference's `dFdx^T (make_spd(ddEddF)) dFdx` step: the local 12x12 Hessian.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> tet_hessian(const S&                  kappa,
                                                      const S&                  dt,
                                                      const S&                  volume,
                                                      const lps::LVec<S, 3>&    x0,
                                                      const lps::LVec<S, 3>&    x1,
                                                      const lps::LVec<S, 3>&    x2,
                                                      const lps::LVec<S, 3>&    x3,
                                                      const lps::LMat<S, 3, 3>& Dm_inv)
{
    auto F = fem::F(x0, x1, x2, x3, Dm_inv);

    auto kt2 = lps::detail::mul(lps::detail::mul(kappa, dt), dt);
    auto v   = volume;

    lps::LMat<S, 9, 9> ddEddF;
    sym::arap_3d::ddEddF(ddEddF, kt2, v, F);
    // the reference's `make_spd(ddEddF)` (utils/make_spd.h, in place)
    lps::make_spd_inplace(ddEddF);

    lps::LMat<S, 9, 12> dFdx = fem::dFdx(Dm_inv);

    return dFdx.transpose() * ddEddF * dFdx;
}
}  // namespace arap_3d_detail
}  // namespace uipc::backend::luisa
