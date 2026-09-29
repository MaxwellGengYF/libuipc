#pragma once
/********************************************************************
 * @file affine_body/constitutions/arap_function.h
 * @brief T6.3a — the ABD ARAP energy/gradient/Hessian (port of
 *        src/backends/cuda/affine_body/constitutions/arap_function.h).
 *
 * This file is **hand written in the reference** (no generator): it is a literal
 * transcription, changing only the spelling of the operators (cookbook §7.2 RULE
 * M1 and §7.17 rule 34: a transcription is never re-spelled by hand *and* never
 * re-derived):
 *   `svd(F, U, Sigma, V)`        -> `lps::svd` (the signed SVD of T2.2, same
 *                                   convention: U/V proper rotations with the
 *                                   reflection of `det F < 0` absorbed into
 *                                   `Sigma(2)`; see cookbook §7.5)
 *   `Eigen::Matrix<T,3,3>`       -> `lps::LMat<T,3,3>`
 *   `Eigen::Vector<T,N>`         -> `lps::LVec<T,N>`
 *   `a * b` (scalars)            -> the `lps::detail::{mul,div,add,sub}`
 *                                   trampolines; `x * container` keeps the
 *                                   numerics layer's own operator
 *   `constexpr double sqrt2`     -> the reference's constant (a host value
 *                                   multiplied into a container)
 *
 * The ABD local variable is the **12-dof affine state** `q = [t; a1; a2; a3]`
 * (§7.21 rule 6), whose *linear* part is the 3x3 map
 * `F = extractF(q)` with `F.row(i) = q.segment<3>(3 + 3i)`.  The energy is the
 * **dimensionless** ARAP density `kappa * ||F - R||_F^2` (`R = U V^T`), i.e. the
 * ABD sibling of the FEM family's `kappa * v * ||F - R||^2` (§7.17 rule 8): the
 * volume and the time-step factor `v dt^2` are *not* inside the local law here,
 * they are applied by the kernel (`E * Vdt2`, `G9 * Vdt2`, `H9 * Vdt2`) — exactly
 * as the reference splits them.  Do not "unify" the two families.
 *
 * The **Hessian is not clamped** (measured in the reference: `arap.cu` includes
 * `utils/make_spd.h` but never calls it — the include is vestigial; only
 * `ortho_potential.cu` calls `make_spd`).  `ARAP_Hessian` therefore returns the
 * exact (possibly indefinite) `d^2E/dF^2`; the test asserts FD agreement in
 * indefinite states *and* the absence of the clamp.
 *
 * DEVIATION (reported): the reference's `dEdq` fills the 9-vector with a nested
 * `for(int i)...for(int j)` loop over *constant* bounds.  A host loop over host
 * constants is legal in a kernel body (it unrolls while the IR is built), so the
 * port keeps it verbatim rather than expanding it by hand.
 ********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/math.h>
#include <luisa_device/math/svd.h>

namespace uipc::backend::luisa
{
namespace sym::abd_arap
{
inline constexpr double sqrt2 =
    1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276;

/// Function to extract the F matrix from the q vector
template <typename T>
inline void extractF(lps::LMat<T, 3, 3>& F, const lps::LVec<T, 12>& q)
{
    F.row(0) = q.segment<3>(3);
    F.row(1) = q.segment<3>(6);
    F.row(2) = q.segment<3>(9);
}

/// Function to compute the ARAP energy
template <typename T>
inline void E(T& energy, const T& kappa, const lps::LVec<T, 12>& q)
{
    lps::LMat<T, 3, 3> F;
    extractF(F, q);
    lps::LMat<T, 3, 3> R;
    lps::LMat<T, 3, 3> U, V;
    lps::LVec<T, 3>    Sigma;
    lps::svd(F, U, Sigma, V);
    R      = U * V.transpose();
    energy = lps::detail::mul(kappa, (F - R).squaredNorm());
}

/// Function to compute the gradient of the ARAP energy
template <typename T>
inline void dEdq(lps::LVec<T, 9>&         gradients,
                 const T&                 kappa,
                 const lps::LVec<T, 12>& q)
{
    lps::LMat<T, 3, 3> F;
    extractF(F, q);
    lps::LMat<T, 3, 3> R;
    lps::LMat<T, 3, 3> U, V;
    lps::LVec<T, 3>    Sigma;
    lps::svd(F, U, Sigma, V);
    R = U * V.transpose();

    lps::LMat<T, 3, 3> dPsi_dF = (F - R) * 2.0;

    // the reference's loop, kept verbatim (host bounds)
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            gradients(i * 3 + j) = lps::detail::mul(kappa, dPsi_dF(i, j));
        }
    }
}

/// Function to flatten a matrix into a vector (the reference's `vector << ...`
/// comma initialiser, expanded into one element assignment per entry in Eigen's
/// row-major traversal order)
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
inline void ARAP_Hessian(lps::LMat<T, 9, 9>& H, const lps::LMat<T, 3, 3>& F)
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

    // Compute the Hessian  (`H.setIdentity(); H *= 2;` -- the reference's order)
    H = lps::LMat<T, 9, 9>::Identity() * 2.0;
    H -= lps::detail::div(4.0, lps::detail::add(s0, s1)) * (t0 * t0.transpose());
    H -= lps::detail::div(4.0, lps::detail::add(s1, s2)) * (t1 * t1.transpose());
    H -= lps::detail::div(4.0, lps::detail::add(s0, s2)) * (t2 * t2.transpose());
}

/// Function to compute the Hessian of the ARAP energy
template <typename T>
inline void ddEddq(lps::LMat<T, 9, 9>&     hessians,
                   const T&                 kappa,
                   const lps::LVec<T, 12>&  q)
{
    lps::LMat<T, 3, 3> F;
    extractF(F, q);
    ARAP_Hessian(hessians, F);
    hessians *= kappa;
}
}  // namespace sym::abd_arap

namespace abd_arap_detail
{
/**
 * @brief The reference's `arap_compute_energy_kernel` body, factored out.
 *
 * The kernel reads the body's `q`, its volume and the global `dt` and does exactly
 * this; the test instantiates the same routine for `S = double` (host oracle) and
 * `S = lcrt::Var<double>` (device) -- cookbook §7.4.
 */
template <typename S>
[[nodiscard]] inline S body_energy(const S&                  kappa,
                                   const S&                  dt,
                                   const S&                  volume,
                                   const lps::LVec<S, 12>&   q)
{
    // reference: `Float Vdt2 = volume * dt * dt; ... shape_energies(i) = E * Vdt2;`
    auto Vdt2 = lps::detail::mul(lps::detail::mul(volume, dt), dt);

    S E;
    sym::abd_arap::E(E, kappa, q);
    return lps::detail::mul(E, Vdt2);
}

/// The reference's `G.segment<9>(3) = G9 * Vdt2;` (the translation block stays 0).
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> body_gradient(const S&                kappa,
                                                    const S&                dt,
                                                    const S&                volume,
                                                    const lps::LVec<S, 12>& q)
{
    auto Vdt2 = lps::detail::mul(lps::detail::mul(volume, dt), dt);

    lps::LVec<S, 9> G9;
    sym::abd_arap::dEdq(G9, kappa, q);

    lps::LVec<S, 12> G = lps::LVec<S, 12>::Zero();
    G.segment<9>(3)    = G9 * Vdt2;
    return G;
}

/// The reference's `H.block<9,9>(3,3) = H9x9 * Vdt2;` (no `make_spd` here -- see
/// the file comment).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> body_hessian(const S&                kappa,
                                                       const S&                dt,
                                                       const S&                volume,
                                                       const lps::LVec<S, 12>& q)
{
    auto Vdt2 = lps::detail::mul(lps::detail::mul(volume, dt), dt);

    lps::LMat<S, 9, 9> H9x9;
    sym::abd_arap::ddEddq(H9x9, kappa, q);

    lps::LMat<S, 12, 12> H = lps::LMat<S, 12, 12>::Zero();
    H.block<9, 9>(3, 3)    = H9x9 * Vdt2;
    return H;
}
}  // namespace abd_arap_detail
}  // namespace uipc::backend::luisa
