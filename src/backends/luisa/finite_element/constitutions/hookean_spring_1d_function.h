#pragma once
/********************************************************************
 * @file hookean_spring_1d_function.h
 * @brief T5.2 (batch c) — the Hookean-spring energy/gradient/Hessian (port of
 *        src/backends/cuda/finite_element/constitutions/
 *        hookean_spring_1d_function.h).
 *
 * Generic over the scalar type exactly like the rest of the layer:
 *   T = double             -> host evaluation (the reference's Float)
 *   T = lcrt::Var<double>  -> device evaluation (a DSL local)
 *
 * The `sym/hookean_spring_1d.inl` body is machine generated (SymEigen.py) and is
 * transcribed by `scratch/gen_fem_cons_c.py` — never by hand (cookbook §7.17 rule 7 /
 * the T5.2 brief's §2).  `hs1d_detail` then factors the reference's two kernels into
 * the per-element routines the unit test instantiates too (cookbook §7.4), so the
 * shipped kernel and the oracle run *the same* code.
 *
 * Independent derivation of the transcribed energy (asserted by the unit test): with
 * `L = |X1 - X0|` and `e = X0 - X1`,
 *      E      = k (L - L0)^2 / (2 L0^2)
 *      dE/dX  = k (L - L0) / (L0^2 L) * (e, -e)      (the generated `dEdX`)
 *      pGpk   = (L - L0)^2 / (2 L0^2) = E / k        (the parameter gradient dE/dk)
 * and the reference's kernel scales them by
 *      Vdt2   = L0 r^2 pi dt^2 ,   r = edge_thickness(t0, t1)
 * (a 1d Hookean spring's material volume: rest length x cross-section area, times
 * dt^2 for the implicit time integrator).
 *
 * `pGpk` is *not* used by the constitution itself (it is the diff-sim parameter
 * gradient surface, exactly like batch a's unused SMT sibling); it is transcribed for
 * completeness and is exercised by the unit test directly.
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace (T2.2)
#include <utils/codim_thickness.h>

namespace uipc::backend::luisa
{
namespace sym::hookean_spring_1d
{
#include "sym/hookean_spring_1d.inl"
}  // namespace sym::hookean_spring_1d

namespace hs1d_detail
{
/// The sweep count the spring's 6x6 element Hessian passes to `lps::make_spd_inplace`.
///
/// The reference's `utils/make_spd.h` is a tolerance-driven Jacobi loop, so the port
/// needs a host-fixed count (RULE M2).  This is the *same measured choice* as the
/// rod's `krb_detail::kMakeSpdSweeps` (see `kirchhoff_rod_bending_function.h` for the
/// sweep-plateau table: 8 sweeps already reaches the exact eigenvalue clamp to ~1e-14
/// relative for N = 6, 9 and 12, while 5 does not).  It is spelled locally instead of
/// shared so that neither constitution's header depends on the other's.
inline constexpr int kMakeSpdSweeps = 8;

/// The reference's kernel step `X.segment<3>(3 * i) = xs(idx(i));` — the 6-vector of a
/// spring's two positions.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 6> spring_positions(const lps::LVec<S, 3>& x0,
                                                      const lps::LVec<S, 3>& x1) noexcept
{
    lps::LVec<S, 6> X;
    X.segment<3>(0) = x0;
    X.segment<3>(3) = x1;
    return X;
}

/// The reference's `Float Vdt2 = L0 * r * r * Pi * dt * dt;` (same association order).
template <typename S>
[[nodiscard]] inline S edge_vdt2(const S& L0, const S& r, const S& pi, const S& dt) noexcept
{
    namespace d = lps::detail;
    return d::mul(d::mul(d::mul(d::mul(d::mul(L0, r), r), pi), dt), dt);
}

/// The reference's `NS::E(E, kappa, X, L0); energies(I) = E * Vdt2;`
template <typename S>
[[nodiscard]] inline S stencil_energy(const S&               kappa,
                                      const S&               L0,
                                      const S&               r,
                                      const S&               pi,
                                      const S&               dt,
                                      const lps::LVec<S, 3>& x0,
                                      const lps::LVec<S, 3>& x1)
{
    namespace d = lps::detail;
    S E;
    sym::hookean_spring_1d::E(E, kappa, spring_positions<S>(x0, x1), L0);
    return d::mul(E, edge_vdt2<S>(L0, r, pi, dt));
}

/// The reference's `NS::dEdX(G, kappa, X, L0); G *= Vdt2;`
template <typename S>
[[nodiscard]] inline lps::LVec<S, 6> stencil_gradient(const S&               kappa,
                                                      const S&               L0,
                                                      const S&               r,
                                                      const S&               pi,
                                                      const S&               dt,
                                                      const lps::LVec<S, 3>& x0,
                                                      const lps::LVec<S, 3>& x1)
{
    lps::LVec<S, 6> G;
    sym::hookean_spring_1d::dEdX(G, kappa, spring_positions<S>(x0, x1), L0);
    return G * edge_vdt2<S>(L0, r, pi, dt);
}

/// The reference's `NS::ddEddX(H, kappa, X, L0); H *= Vdt2; make_spd(H);` — the scale
/// is applied *before* the projection, exactly as the reference orders it.
template <typename S>
[[nodiscard]] inline lps::LMat<S, 6, 6> stencil_hessian(const S&               kappa,
                                                        const S&               L0,
                                                        const S&               r,
                                                        const S&               pi,
                                                        const S&               dt,
                                                        const lps::LVec<S, 3>& x0,
                                                        const lps::LVec<S, 3>& x1)
{
    lps::LMat<S, 6, 6> H;
    sym::hookean_spring_1d::ddEddX(H, kappa, spring_positions<S>(x0, x1), L0);
    H = H * edge_vdt2<S>(L0, r, pi, dt);
    // reference: `make_spd(H);`
    lps::make_spd_inplace(H, kMakeSpdSweeps);
    return H;
}
}  // namespace hs1d_detail
}  // namespace uipc::backend::luisa
