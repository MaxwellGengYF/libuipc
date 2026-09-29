#pragma once
/********************************************************************
 * @file affine_body/constitutions/ortho_potential_function.h
 * @brief T6.3a (ABD) — the orthogonal-potential energy/gradient/Hessian (port of
 *        src/backends/cuda/affine_body/constitutions/ortho_potential_function.h).
 *
 * The reference header is a thin include shell around the SymEigen.py-generated
 * `sym/ortho_potential.inl` (a "no-includes-allowed" .inl, so the generated bodies
 * can use the layer's unqualified names).  Reference include map:
 *   <cuda_tool/cuda_tool.h>   -> <affine_body/type_define.h> + <luisa_device/math/math.h>
 *   the generated `std::pow`  -> `lps::pow` (spelled by the transcriber, not here)
 *   <utils/make_spd.h>        -> `lps::make_spd_inplace` (matrix_ops.h, via math.h)
 *
 * `abd_ortho_potential_detail` factors the reference's
 * `ortho_potential_compute_{energy,gradient_hessian}_kernel` bodies into generic
 * per-body routines that the unit test instantiates too (cookbook §7.4).  The
 * Hessian is exposed twice on purpose, because the reference's kernel *clamps* it:
 *   * `body_hessian_unclamped` — `ddEddq(H9x9, kappa, q)` (the exact second
 *     derivative of the generated energy: it is what the finite-difference oracle
 *     must be compared against, and it is what the generated file itself defines);
 *   * `body_hessian` — the *shipped* path, `make_spd(H9x9)` before the `Vdt2`
 *     scaling and the 12x12 embedding, in the reference's own order.
 * The two differ only by that one line; the test asserts the eigenvalue-clamp
 * identity between them (and that the clamp is not a no-op).
 *
 * `kMakeSpdSweeps` is spelled *locally* (like `krb_detail::kMakeSpdSweeps` /
 * `hs1d_detail::kMakeSpdSweeps`, cookbook §7.17 rule 31): the reference's
 * `make_spd` is a tolerance-driven Jacobi loop, so a device port must pick a fixed
 * count and *measure* it.  §7.17 rule 31 measured 8 sweeps as the plateau for
 * N = 6/9/12 in this layer; the T6.3a test re-measures it on the *actual* 9x9
 * element Hessians (and on the 20-sweep default) rather than inheriting the number.
 ********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/math.h>
// `lps::make_spd_inplace` (the reference's `utils/make_spd.h`) lives in matrix_ops.h;
// math.h alone does not declare it (measured C2039).
#include <luisa_device/math/matrix_ops.h>

namespace uipc::backend::luisa
{
namespace sym::abd_ortho_potential
{
#include "sym/ortho_potential.inl"
}  // namespace sym::abd_ortho_potential

namespace abd_ortho_potential_detail
{
inline constexpr int kMakeSpdSweeps = 8;

/**
 * @brief The reference's `ortho_potential_compute_energy_kernel` body, factored out.
 *
 * AOP::E is the *dimensionless* density `kappa ||AᵀA - I||_F^2`; the kernel's
 * `Vdt2 = volume * dt^2` scaling is applied here, exactly as the reference splits
 * it (`shape_energies(i) = E * Vdt2;`).
 */
template <typename S>
[[nodiscard]] inline S body_energy(const S&                kappa,
                                   const S&                dt,
                                   const S&                volume,
                                   const lps::LVec<S, 12>& q)
{
    auto Vdt2 = lps::detail::mul(lps::detail::mul(volume, dt), dt);

    S E;
    sym::abd_ortho_potential::E(E, kappa, q);
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
    sym::abd_ortho_potential::dEdq(G9, kappa, q);

    lps::LVec<S, 12> G = lps::LVec<S, 12>::Zero();
    G.segment<9>(3)    = G9 * Vdt2;
    return G;
}

/// The reference's 9x9 stage without the clamp: `AOP::ddEddq(H9x9, kappa, q);`.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 9, 9> hessian_9x9(const S&                kappa,
                                                    const lps::LVec<S, 12>& q)
{
    lps::LMat<S, 9, 9> H9x9;
    sym::abd_ortho_potential::ddEddq(H9x9, kappa, q);
    return H9x9;
}

/// The reference's 9x9 stage *with* the clamp:
/// `AOP::ddEddq(H9x9, kappa, q); make_spd(H9x9);`.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 9, 9> hessian_9x9_spd(const S&                kappa,
                                                        const lps::LVec<S, 12>& q)
{
    auto H9x9 = hessian_9x9(kappa, q);
    // the reference's `make_spd(H9x9)` (utils/make_spd.h: evd, clamp negative
    // eigenvalues to 0, reconstruct) -- the port's fixed sweep count
    lps::make_spd_inplace(H9x9, kMakeSpdSweeps);
    return H9x9;
}

/// `H.block<9, 9>(3, 3) = H9x9 * Vdt2;` on a zero 12x12 (the reference's embedding).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> embed_hessian(const lps::LMat<S, 9, 9>& H9x9,
                                                        const S& dt,
                                                        const S& volume)
{
    auto Vdt2 = lps::detail::mul(lps::detail::mul(volume, dt), dt);

    lps::LMat<S, 12, 12> H = lps::LMat<S, 12, 12>::Zero();
    H.block<9, 9>(3, 3)    = H9x9 * Vdt2;
    return H;
}

/// The shipped Hessian (`make_spd` applied in the 9x9 F-block, *before* `Vdt2`).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> body_hessian(const S&                kappa,
                                                       const S&                dt,
                                                       const S&                volume,
                                                       const lps::LVec<S, 12>& q)
{
    return embed_hessian(hessian_9x9_spd(kappa, q), dt, volume);
}

/// The same embedding without the clamp -- the finite-difference oracle's target
/// (and the exact derivative of the generated energy).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> body_hessian_unclamped(const S&                kappa,
                                                                const S&                dt,
                                                                const S&                volume,
                                                                const lps::LVec<S, 12>& q)
{
    return embed_hessian(hessian_9x9(kappa, q), dt, volume);
}
}  // namespace abd_ortho_potential_detail
}  // namespace uipc::backend::luisa
