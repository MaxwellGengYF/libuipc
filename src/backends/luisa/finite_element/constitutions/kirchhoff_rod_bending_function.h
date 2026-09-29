#pragma once
/********************************************************************
 * @file kirchhoff_rod_bending_function.h
 * @brief T5.2 (batch c) — the Kirchhoff-rod bending energy/gradient/Hessian (port of
 *        src/backends/cuda/finite_element/constitutions/
 *        kirchhoff_rod_bending_function.h).
 *
 * Generic over the scalar type exactly like the rest of the layer:
 *   T = double             -> host evaluation (the reference's Float)
 *   T = lcrt::Var<double>  -> device evaluation (a DSL local)
 *
 * The `sym/kirchhoff_rod_bending.inl` body is machine generated (SymEigen.py) and is
 * transcribed by `scratch/gen_fem_cons_c.py` — never by hand (cookbook §7.17 rule 7 /
 * the T5.2 brief's §2).  `krb_detail` then factors the reference's two kernels into
 * the per-element routines the unit test instantiates too (cookbook §7.4), so the
 * shipped kernel and the oracle run *the same* code.
 *
 * Independent derivation of the transcribed energy (asserted by the unit test, not
 * assumed here).  With `e0 = X1 - X0`, `e1 = X2 - X1` and `theta` the turn angle,
 * the generated `E` simplifies to
 *      E = k pi r^4 (1 - cos theta) / L0 = 2 k pi r^4 sin^2(theta / 2) / L0 ,
 * so a straight (collinear) rod has E = 0 and dE/dX = 0, and the gradient/Hessian of
 * the reference's `E` are the exact derivative of that law.
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace (T2.2)

namespace uipc::backend::luisa
{
namespace sym::kirchhoff_rod_bending
{
#include "sym/kirchhoff_rod_bending.inl"
}  // namespace sym::kirchhoff_rod_bending

namespace krb_detail
{
/**
 * @brief The sweep count the Kirchhoff-rod 9x9 element Hessian passes to
 *        `lps::make_spd_inplace`.
 *
 * The reference's `utils/make_spd.h` is a *tolerance-driven* Jacobi loop
 * (`while(calc_sumDiagOff(M) > 1e-6)`), i.e. its sweep count is device data — the port
 * needs a host-fixed count (RULE M2).  The layer's default (20, T2.2's choice) makes
 * LC's nvrtc spend minutes on a single inlined 9x9 eigensystem (batch a measured
 * ~962 s for ARAP's 9x9), so the rod passes an explicit count, exactly like the batch-b
 * shells do for their 12x12 (`scalar_detail::kShellMakeSpdSweeps = 8`).
 *
 * Measured plateau of the cyclic sweep (the layer's own `jacobi_evd`, replicated in
 * double precision against an exact `eigen-decompose + clamp + reconstruct`, 30 random
 * symmetric matrices per N, worst relative error over the matrices' own scale):
 *
 *   N = 6 : 5 sweeps 1.9e-12   8 sweeps 4.3e-15   12 sweeps 4.3e-15   20 sweeps 3.1e-15
 *   N = 9 : 5 sweeps 1.1e-06   8 sweeps 6.8e-15   12 sweeps 4.2e-15   20 sweeps 5.9e-15
 *   N = 12: 5 sweeps 2.2e-04   8 sweeps 7.4e-15   12 sweeps 5.8e-15   20 sweeps 7.5e-15
 *
 * i.e. 8 sweeps is already at the floating-point plateau for every N this batch uses
 * (5 is *not* enough) — the port's projection is the reference's eigenvalue clamp to
 * machine precision, at ~1/3 of the JIT cost.  The unit test re-measures this on the
 * rod's *actual* element Hessians against an exact Eigen clamp.
 */
inline constexpr int kMakeSpdSweeps = 8;

/// The reference's kernel step `X.segment<3>(0) = xs(hinge[0]); X.segment<3>(3) = ...`
/// — the 9-vector of a hinge's three positions.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 9> hinge_positions(const lps::LVec<S, 3>& x0,
                                                     const lps::LVec<S, 3>& x1,
                                                     const lps::LVec<S, 3>& x2) noexcept
{
    lps::LVec<S, 9> X;
    X.segment<3>(0) = x0;
    X.segment<3>(3) = x1;
    X.segment<3>(6) = x2;
    return X;
}

/// The reference's `Float L0 = (x1_bar - x0_bar).norm() + (x2_bar - x1_bar).norm();`
/// — the rest length of the two edges of a hinge.
template <typename S>
[[nodiscard]] inline S rest_length(const lps::LVec<S, 3>& x0_bar,
                                   const lps::LVec<S, 3>& x1_bar,
                                   const lps::LVec<S, 3>& x2_bar) noexcept
{
    namespace d = lps::detail;
    return d::add((x1_bar - x0_bar).norm(), (x2_bar - x1_bar).norm());
}

/**
 * @brief The reference's `KirchhoffRodBending_do_compute_energy_kernel` body.
 *
 * The reference stores `energies(I) = E` with `E` evaluated at the *dt-scaled*
 * stiffness (`Float k = bending_stiffnesses(I) * dt * dt;`), which is why the
 * energy path and the gradient path (`G *= dt2`) agree.
 */
template <typename S>
[[nodiscard]] inline S stencil_energy(const S&               kappa,
                                      const S&               dt,
                                      const lps::LVec<S, 3>& x0,
                                      const lps::LVec<S, 3>& x1,
                                      const lps::LVec<S, 3>& x2,
                                      const S&               L0,
                                      const S&               r,
                                      const S&               pi)
{
    namespace d = lps::detail;
    // reference: `Float k = bending_stiffnesses(I) * dt * dt;`
    auto k = d::mul(d::mul(kappa, dt), dt);
    S    E;
    sym::kirchhoff_rod_bending::E(E, k, hinge_positions<S>(x0, x1, x2), L0, r, pi);
    return E;
}

/// The reference's `KRB::dEdX(G, k, X, L0, r, Pi); G *= dt2;`
template <typename S>
[[nodiscard]] inline lps::LVec<S, 9> stencil_gradient(const S&               kappa,
                                                      const S&               dt,
                                                      const lps::LVec<S, 3>& x0,
                                                      const lps::LVec<S, 3>& x1,
                                                      const lps::LVec<S, 3>& x2,
                                                      const S&               L0,
                                                      const S&               r,
                                                      const S&               pi)
{
    namespace d = lps::detail;
    lps::LVec<S, 9> G;
    sym::kirchhoff_rod_bending::dEdX(G, kappa, hinge_positions<S>(x0, x1, x2), L0, r, pi);
    return G * d::mul(dt, dt);
}

/// The reference's `KRB::ddEddX(H, k, X, L0, r, Pi); H *= dt2; make_spd(H);` — the
/// scale is applied *before* the projection, exactly as the reference orders it.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 9, 9> stencil_hessian(const S&               kappa,
                                                        const S&               dt,
                                                        const lps::LVec<S, 3>& x0,
                                                        const lps::LVec<S, 3>& x1,
                                                        const lps::LVec<S, 3>& x2,
                                                        const S&               L0,
                                                        const S&               r,
                                                        const S&               pi)
{
    namespace d = lps::detail;
    lps::LMat<S, 9, 9> H;
    sym::kirchhoff_rod_bending::ddEddX(H, kappa, hinge_positions<S>(x0, x1, x2), L0, r, pi);
    H = H * d::mul(dt, dt);
    // reference: `make_spd(H);` (utils/make_spd.h, in place)
    lps::make_spd_inplace(H, kMakeSpdSweeps);
    return H;
}
}  // namespace krb_detail
}  // namespace uipc::backend::luisa
