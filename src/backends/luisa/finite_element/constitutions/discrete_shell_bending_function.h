#pragma once
/********************************************************************
 * @file discrete_shell_bending_function.h
 * @brief T5.2 (batch b) — the discrete-shell-bending energy/gradient/Hessian (port
 *        of src/backends/cuda/finite_element/constitutions/
 *        discrete_shell_bending_function.h).
 *        ref: https://www.cs.columbia.edu/cg/pdfs/10_ds.pdf
 *
 * Reference include map:
 *   <type_define.h>                            -> <type_define.h> (same)
 *   <finite_element/constitutions/discrete_shell_bending_reference.h>
 *                                              -> same relative path
 *   <utils/dihedral_angle.h>                   -> <utils/dihedral_angle.h>
 *
 * Mapping (the reference's bodies are hand-written Eigen, not generated output):
 *  * `template <typename S>` with `S = double` (host) / `S = lcrt::Var<double>`
 *    (device); `Vector12` -> `lps::LVec<S,12>`, `Matrix12x12` -> `lps::LMat<S,12,12>`;
 *  * scalar arithmetic goes through the `lps::detail::{add,sub,mul,div}` trampolines
 *    (RULE M1, cookbook §7.2), container products keep the numerics layer's own
 *    operators, and the reference's association order is preserved verbatim
 *    (`dthetadx * ddEddtheta * dthetadx.transpose()` stays left-associative);
 *  * the `sym/discrete_shell_bending.inl` include is machine generated and is
 *    transcribed by `scratch/gen_fem_cons_b.py` (never by hand).
 *
 * `dsb_detail` factors the reference's two kernels into the per-element routines the
 * unit test instantiates too (cookbook §7.4), so the shipped kernel and the oracle
 * run *the same* code.
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace (T2.2)
#include <finite_element/constitutions/discrete_shell_bending_reference.h>
#include <utils/dihedral_angle.h>

namespace uipc::backend::luisa
{
namespace sym::discrete_shell_bending
{
#include "sym/discrete_shell_bending.inl"

    template <typename S>
    inline void compute_constants(S&                     L0,
                                  S&                     h_bar,
                                  S&                     theta_bar,
                                  S&                     V_bar,
                                  const lps::LVec<S, 3>& x0_bar,
                                  const lps::LVec<S, 3>& x1_bar,
                                  const lps::LVec<S, 3>& x2_bar,
                                  const lps::LVec<S, 3>& x3_bar,
                                  const S&               thickness0,
                                  const S&               thickness1,
                                  const S&               thickness2,
                                  const S&               thickness3)
    {
        (void)thickness0;
        (void)thickness1;
        (void)thickness2;
        (void)thickness3;
        uipc::backend::luisa::detail::compute_discrete_shell_bending_reference(
            L0, h_bar, theta_bar, V_bar, x0_bar, x1_bar, x2_bar, x3_bar);
    }

    template <typename S>
    [[nodiscard]] inline S E(const lps::LVec<S, 3>& x0,
                             const lps::LVec<S, 3>& x1,
                             const lps::LVec<S, 3>& x2,
                             const lps::LVec<S, 3>& x3,
                             const S&               L0,
                             const S&               h_bar,
                             const S&               theta_bar,
                             const S&               kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        S theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        S R;
        DSB::E(R, kappa, theta, theta_bar, L0, h_bar);

        return R;
    }

    template <typename S>
    inline void dEdx(lps::LVec<S, 12>&      G,
                     const lps::LVec<S, 3>& x0,
                     const lps::LVec<S, 3>& x1,
                     const lps::LVec<S, 3>& x2,
                     const lps::LVec<S, 3>& x3,
                     const S&               L0,
                     const S&               h_bar,
                     const S&               theta_bar,
                     const S&               kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        S theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        S dEdtheta;
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);

        lps::LVec<S, 12> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        G = dEdtheta * dthetadx;
    }

    template <typename S>
    inline void ddEddx(lps::LMat<S, 12, 12>& H,
                       const lps::LVec<S, 3>&  x0,
                       const lps::LVec<S, 3>&  x1,
                       const lps::LVec<S, 3>&  x2,
                       const lps::LVec<S, 3>&  x3,
                       const S&                L0,
                       const S&                h_bar,
                       const S&                theta_bar,
                       const S&                kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        S theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        S dEdtheta;
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);

        S ddEddtheta;
        DSB::ddEddtheta(ddEddtheta, kappa, theta, theta_bar, L0, h_bar);

        lps::LVec<S, 12> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        lps::LMat<S, 12, 12> ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        H = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;
    }
}  // namespace sym::discrete_shell_bending

namespace dsb_detail
{
/**
 * @brief The reference's `DiscreteShellBending_do_compute_energy_kernel` body.
 *
 * The energy is stored *with* the `V_bar * dt * dt` scale the reference applies
 * (`energies(I) = E * V_bar * dt * dt`), with its exact association order.
 */
template <typename S>
[[nodiscard]] inline S stencil_energy(const lps::LVec<S, 3>& x0,
                                      const lps::LVec<S, 3>& x1,
                                      const lps::LVec<S, 3>& x2,
                                      const lps::LVec<S, 3>& x3,
                                      const S&               L0,
                                      const S&               h_bar,
                                      const S&               theta_bar,
                                      const S&               kappa,
                                      const S&               V_bar,
                                      const S&               dt)
{
    namespace d = lps::detail;
    auto E = sym::discrete_shell_bending::E(x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
    // reference: `energies(I) = E * V_bar * dt * dt;`
    return d::mul(d::mul(d::mul(E, V_bar), dt), dt);
}

/// The reference's `Float Vdt2 = V_bar * dt * dt;` (same association order).
template <typename S>
[[nodiscard]] inline S stencil_vdt2(const S& V_bar, const S& dt)
{
    namespace d = lps::detail;
    return d::mul(d::mul(V_bar, dt), dt);
}

/// The reference's gradient update: `DSB::dEdx(G12, ...); G12 *= Vdt2;`
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> stencil_gradient(const lps::LVec<S, 3>& x0,
                                                       const lps::LVec<S, 3>& x1,
                                                       const lps::LVec<S, 3>& x2,
                                                       const lps::LVec<S, 3>& x3,
                                                       const S&               L0,
                                                       const S&               h_bar,
                                                       const S&               theta_bar,
                                                       const S&               kappa,
                                                       const S&               V_bar,
                                                       const S&               dt)
{
    lps::LVec<S, 12> G12;
    sym::discrete_shell_bending::dEdx(G12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
    // reference: `G12 *= Vdt2;`
    return G12 * stencil_vdt2(V_bar, dt);
}

/**
 * The reference's Hessian update: `DSB::ddEddx(H12x12, ...); H12x12 *= Vdt2;
 * make_spd(H12x12);` — the scale is applied *before* the projection, exactly as
 * the reference orders it (the NH/SNK1 constitutions apply theirs the other way
 * round, which is why the order is spelled out per constitution here).
 */
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> stencil_hessian(const lps::LVec<S, 3>& x0,
                                                          const lps::LVec<S, 3>& x1,
                                                          const lps::LVec<S, 3>& x2,
                                                          const lps::LVec<S, 3>& x3,
                                                          const S&               L0,
                                                          const S&               h_bar,
                                                          const S&               theta_bar,
                                                          const S&               kappa,
                                                          const S&               V_bar,
                                                          const S&               dt)
{
    lps::LMat<S, 12, 12> H12x12;
    sym::discrete_shell_bending::ddEddx(
        H12x12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
    H12x12 = H12x12 * stencil_vdt2(V_bar, dt);
    // reference: `make_spd(H12x12);`
    lps::make_spd_inplace(H12x12, scalar_detail::kShellMakeSpdSweeps);
    return H12x12;
}
}  // namespace dsb_detail
}  // namespace uipc::backend::luisa
