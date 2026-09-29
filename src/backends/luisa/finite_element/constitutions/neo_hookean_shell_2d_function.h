#pragma once
/********************************************************************
 * @file neo_hookean_shell_2d_function.h
 * @brief T5.2 (batch b) — the 2D Neo-Hookean *shell* (membrane) energy, gradient
 *        and Hessian (port of src/backends/cuda/finite_element/constitutions/
 *        neo_hookean_shell_2d_function.h).
 *
 * The reference header is a thin include shell around
 * `sym/neo_hookean_shell_2d.inl` (SymEigen.py output: `A`, `E`, `dEdX`, `ddEddX`),
 * which is transcribed by `scratch/gen_fem_cons_b.py` — never by hand.
 *
 * `nh_shell_detail` below factors the reference's three kernels into the generic
 * per-element routines that the unit test instantiates too (cookbook §7.4), so the
 * shipped kernel and the oracle run *the same* code.  The order of the two
 * operations the reference applies to the local Hessian is preserved exactly and is
 * *not* the same as the bending constitutions':
 *     `NH::ddEddX(H, ...); make_spd(H); H *= Vdt2;`   (project first, then scale)
 * whereas `DeviceBuffer`-side bending does `H *= Vdt2; make_spd(H);`.  Both are
 * reproduced verbatim (a reordered pair is a silently different result).
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/matrix_ops.h>  // lps::inverse / lps::make_spd_inplace
#include <utils/codim_thickness.h>

namespace uipc::backend::luisa
{
namespace sym::neo_hookean_shell_2d
{
#include "sym/neo_hookean_shell_2d.inl"
}  // namespace sym::neo_hookean_shell_2d

namespace nh_shell_detail
{
/**
 * @brief The reference's `NeoHookeanShell2D_do_init_kernel` body: the inverse of the
 *        triangle's rest shape matrix `A(X_bar)`.
 *
 * `X_bar` is the 9-vector of the three rest positions in the reference's own order
 * (`X_bar.segment<3>(3 * i) = x_bars(idx(i))`).
 */
template <typename S>
[[nodiscard]] inline lps::LMat<S, 2, 2> rest_metric_inv(const lps::LVec<S, 9>& X_bar)
{
    lps::LMat<S, 2, 2> B;
    sym::neo_hookean_shell_2d::A(B, X_bar);
    // reference: `inv_B_mats(I) = cuda_tool::eigen::inverse(B);`
    return lps::inverse(B);
}

/// The reference's `for(int i = 0; i < 3; ++i) X.segment<3>(3 * i) = xs(idx(i));`
template <typename S>
[[nodiscard]] inline lps::LVec<S, 9> gather_positions(const lps::LVec<S, 3>& x0,
                                                      const lps::LVec<S, 3>& x1,
                                                      const lps::LVec<S, 3>& x2)
{
    lps::LVec<S, 9> X;
    X.template segment<3>(0) = x0;
    X.template segment<3>(3) = x1;
    X.template segment<3>(6) = x2;
    return X;
}

/// The reference's `Vdt2 = rest_area * 2 * thickness * dt * dt;` (same association).
template <typename S>
[[nodiscard]] inline S element_vdt2(const S& rest_area, const S& thickness, const S& dt)
{
    namespace d = lps::detail;
    return d::mul(d::mul(d::mul(d::mul(rest_area, 2.0), thickness), dt), dt);
}

/// `NeoHookeanShell2D_do_compute_energy_kernel`: `E = NH::E(...) * Vdt2;`
template <typename S>
[[nodiscard]] inline S element_energy(const S&                  lambda,
                                      const S&                  mu,
                                      const lps::LVec<S, 3>&    x0,
                                      const lps::LVec<S, 3>&    x1,
                                      const lps::LVec<S, 3>&    x2,
                                      const lps::LMat<S, 2, 2>& IB,
                                      const S&                  rest_area,
                                      const S&                  thickness,
                                      const S&                  dt)
{
    namespace d = lps::detail;
    auto X = gather_positions(x0, x1, x2);

    S E;
    sym::neo_hookean_shell_2d::E(E, lambda, mu, X, IB);

    // thickness is one-sided so we multiply by 2
    return d::mul(E, element_vdt2(rest_area, thickness, dt));
}

/// `NeoHookeanShell2D_do_compute_gradient_hessian_kernel`: `NH::dEdX(G, ...); G *= Vdt2;`
template <typename S>
[[nodiscard]] inline lps::LVec<S, 9> element_gradient(const S&                  lambda,
                                                      const S&                  mu,
                                                      const lps::LVec<S, 3>&    x0,
                                                      const lps::LVec<S, 3>&    x1,
                                                      const lps::LVec<S, 3>&    x2,
                                                      const lps::LMat<S, 2, 2>& IB,
                                                      const S&                  rest_area,
                                                      const S&                  thickness,
                                                      const S&                  dt)
{
    auto X = gather_positions(x0, x1, x2);

    lps::LVec<S, 9> G;
    sym::neo_hookean_shell_2d::dEdX(G, lambda, mu, X, IB);
    return G * element_vdt2(rest_area, thickness, dt);
}

/**
 * The reference's Hessian stage, including its operation order:
 * `NH::ddEddX(H, ...); make_spd(H); H *= Vdt2;`.
 */
template <typename S>
[[nodiscard]] inline lps::LMat<S, 9, 9> element_hessian(const S&                  lambda,
                                                        const S&                  mu,
                                                        const lps::LVec<S, 3>&    x0,
                                                        const lps::LVec<S, 3>&    x1,
                                                        const lps::LVec<S, 3>&    x2,
                                                        const lps::LMat<S, 2, 2>& IB,
                                                        const S&                  rest_area,
                                                        const S&                  thickness,
                                                        const S&                  dt)
{
    auto X = gather_positions(x0, x1, x2);

    lps::LMat<S, 9, 9> H;
    sym::neo_hookean_shell_2d::ddEddX(H, lambda, mu, X, IB);
    // reference: `make_spd(H); H *= Vdt2;` (projection *before* the scale)
    lps::make_spd_inplace(H);
    return H * element_vdt2(rest_area, thickness, dt);
}
}  // namespace nh_shell_detail
}  // namespace uipc::backend::luisa
