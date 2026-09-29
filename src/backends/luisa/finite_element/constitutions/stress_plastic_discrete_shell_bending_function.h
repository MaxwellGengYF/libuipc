#pragma once
/********************************************************************
 * @file stress_plastic_discrete_shell_bending_function.h
 * @brief T5.2 (batch b) — stress-plastic discrete shells: energy, gradient, Hessian
 *        and the per-element history update (port of
 *        src/backends/cuda/finite_element/constitutions/
 *        stress_plastic_discrete_shell_bending_function.h).
 *
 * Structurally the strain-plastic sibling (see that file's comment for the
 * `select`-based restructuring of the reference's `bool`-returning guards), with two
 * extra pieces the reference adds here:
 *   * `try_trial_state` — the trial shear/stress state (`tau_trial`, `theta_y`,
 *     `delta_gamma`) whose four out-parameters feed both the response and the update;
 *   * `augmented_response_from_angle_delta` — the elastic-vs-plastic *branch* on
 *     `|delta| <= theta_y`, where the port evaluates both branches and selects
 *     (`p ? t : f`), so the kept branch is the reference's own arithmetic.
 * The comparison chain `|delta| <= theta_y` is the *only* thing that decides which
 * response is used, and `theta_y` is recomputed from `(yield_stress, kappa, L0,
 * h_bar)` exactly like the reference (its `elastic_scale`/`elastic_slope` split and
 * association order are preserved).
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace (T2.2)
#include <finite_element/constitutions/discrete_shell_bending_reference.h>
#include <utils/dihedral_angle.h>
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace sym::stress_plastic_discrete_shell_bending
{
    template <typename S>
    [[nodiscard]] inline S pi()
    {
        return lps::make_scalar<S>(3.14159265358979323846264338327950288);
    }

    template <typename S>
    [[nodiscard]] inline S plasticity_write_threshold()
    {
        return lps::make_scalar<S>(1e-6);
    }

    template <typename S>
    [[nodiscard]] inline S dihedral_guard_eps()
    {
        return lps::make_scalar<S>(1e-12);
    }

    template <typename S>
    [[nodiscard]] inline auto is_finite_scalar(const S& v)
    {
        return scalar_detail::is_finite_scalar(v);
    }

    /// reference: `v < T(0) ? -v : v`
    template <typename S>
    [[nodiscard]] inline S abs_value(const S& v)
    {
        namespace d = lps::detail;
        return scalar_detail::select_scalar(v, d::neg(v), v < lps::make_scalar<S>(0.0));
    }

    /// reference: `a < b ? b : a`
    template <typename S>
    [[nodiscard]] inline S max_value(const S& a, const S& b)
    {
        return scalar_detail::select_scalar(a, b, a < b);
    }

    /// reference: `v >= T(0) ? T(1) : T(-1)`
    template <typename S>
    [[nodiscard]] inline S sign_value(const S& v)
    {
        return scalar_detail::select_scalar(lps::make_scalar<S>(-1.0),
                                            lps::make_scalar<S>(1.0),
                                            v >= lps::make_scalar<S>(0.0));
    }

    template <typename S>
    [[nodiscard]] inline auto is_finite_vec3(const lps::LVec<S, 3>& v)
    {
        return scalar_detail::is_finite_vec3(v);
    }

    /// See `sym::strain_plastic_discrete_shell_bending::wrap_angle` — same 4-step
    /// predicated form of the reference's two unbounded `while` loops.
    template <typename S>
    [[nodiscard]] inline S wrap_angle(const S& angle)
    {
        namespace d   = lps::detail;
        const S Pi    = pi<S>();
        const S TwoPi = d::mul(lps::make_scalar<S>(2.0), Pi);

        auto a = angle;
        for(int k = 0; k < 5; ++k)
            a = scalar_detail::select_scalar(a, d::sub(a, TwoPi), a > Pi);
        for(int k = 0; k < 5; ++k)
            a = scalar_detail::select_scalar(a, d::add(a, TwoPi), a < d::neg(Pi));
        return a;
    }

    template <typename S>
    [[nodiscard]] inline S angle_delta(const S& theta, const S& theta_bar)
    {
        return wrap_angle(lps::detail::sub(theta, theta_bar));
    }

    /// See the strain-plastic sibling: no early returns, the guard chain is folded
    /// into the returned predicate and `theta` is always written.
    template <typename S>
    [[nodiscard]] inline auto safe_dihedral_angle(const lps::LVec<S, 3>& v0,
                                                  const lps::LVec<S, 3>& v1,
                                                  const lps::LVec<S, 3>& v2,
                                                  const lps::LVec<S, 3>& v3,
                                                  S&                     theta)
    {
        namespace d = lps::detail;

        const lps::LVec<S, 3> n1 = lps::cross(v1 - v0, v2 - v0);
        const lps::LVec<S, 3> n2 = lps::cross(v2 - v3, v1 - v3);

        const S n1_sq = n1.squaredNorm();
        const S n2_sq = n2.squaredNorm();
        const S eps   = dihedral_guard_eps<S>();

        const S denom = lps::sqrt(d::mul(n1_sq, n2_sq));

        S cos_theta = d::div(n1.dot(n2), denom);
        cos_theta = lps::max(lps::make_scalar<S>(-1.0),
                             lps::min(lps::make_scalar<S>(1.0), cos_theta));
        theta = lps::acos(cos_theta);

        theta = scalar_detail::select_scalar(
            theta,
            d::neg(theta),
            lps::cross(n2, n1).dot(v1 - v2) < lps::make_scalar<S>(0.0));

        auto ok = scalar_detail::pred_and(is_finite_vec3(v0),
                                         is_finite_vec3(v1),
                                         is_finite_vec3(v2),
                                         is_finite_vec3(v3));
        ok = scalar_detail::pred_and(
            ok, is_finite_scalar(n1_sq), is_finite_scalar(n2_sq), n1_sq > eps, n2_sq > eps);
        ok = scalar_detail::pred_and(ok, is_finite_scalar(denom), denom > eps);
        ok = scalar_detail::pred_and(ok, is_finite_scalar(cos_theta));
        ok = scalar_detail::pred_and(ok, is_finite_scalar(theta));
        return ok;
    }

    /**
     * @brief The reference's `try_trial_state`, without early returns.
     *
     * All four out-parameters are always written (the caller selects on the returned
     * predicate), and the guard chain is folded in the reference's order.
     */
    template <typename S>
    [[nodiscard]] inline auto try_trial_state(const S& theta,
                                              const S& theta_bar,
                                              const S& kappa,
                                              const S& L0,
                                              const S& h_bar,
                                              const S& yield_stress,
                                              S&       delta,
                                              S&       tau_trial,
                                              S&       theta_y,
                                              S&       delta_gamma)
    {
        namespace d = lps::detail;

        auto ok = scalar_detail::pred_and(is_finite_scalar(theta),
                                         is_finite_scalar(theta_bar),
                                         is_finite_scalar(kappa),
                                         is_finite_scalar(L0),
                                         is_finite_scalar(h_bar),
                                         is_finite_scalar(yield_stress));
        ok = scalar_detail::pred_and(ok,
                                     yield_stress >= lps::make_scalar<S>(0.0),
                                     L0 > lps::make_scalar<S>(0.0),
                                     h_bar > lps::make_scalar<S>(0.0));

        // reference: `const T elastic_slope = static_cast<T>(2.0) * kappa * L0 / h_bar;`
        const S elastic_slope = d::div(
            d::mul(d::mul(lps::make_scalar<S>(2.0), kappa), L0), h_bar);
        ok = scalar_detail::pred_and(
            ok, is_finite_scalar(elastic_slope), elastic_slope > lps::make_scalar<S>(0.0));

        delta     = angle_delta(theta, theta_bar);
        tau_trial = d::mul(elastic_slope, delta);
        theta_y   = d::div(yield_stress, elastic_slope);
        delta_gamma =
            max_value(d::sub(abs_value(delta), theta_y), lps::make_scalar<S>(0.0));

        ok = scalar_detail::pred_and(ok,
                                     is_finite_scalar(delta),
                                     is_finite_scalar(tau_trial),
                                     is_finite_scalar(theta_y),
                                     theta_y >= lps::make_scalar<S>(0.0),
                                     is_finite_scalar(delta_gamma),
                                     delta_gamma >= lps::make_scalar<S>(0.0));
        return ok;
    }

    /**
     * @brief The reference's `augmented_response_from_angle_delta`, without the
     *        elastic/plastic `if` — both branches are evaluated and selected.
     */
    template <typename S>
    [[nodiscard]] inline auto augmented_response_from_angle_delta(const S& delta,
                                                                  const S& kappa,
                                                                  const S& L0,
                                                                  const S& h_bar,
                                                                  const S& yield_stress,
                                                                  S&       energy,
                                                                  S&       dEdtheta,
                                                                  S&       ddEddtheta)
    {
        namespace d = lps::detail;

        const S zero = lps::make_scalar<S>(0.0);

        auto ok = scalar_detail::pred_and(is_finite_scalar(delta),
                                         is_finite_scalar(kappa),
                                         is_finite_scalar(L0),
                                         is_finite_scalar(h_bar),
                                         is_finite_scalar(yield_stress));
        ok = scalar_detail::pred_and(ok,
                                     yield_stress >= zero,
                                     L0 > zero,
                                     h_bar > zero);

        const S elastic_scale = d::div(d::mul(kappa, L0), h_bar);
        const S elastic_slope = d::mul(lps::make_scalar<S>(2.0), elastic_scale);
        ok                    = scalar_detail::pred_and(ok,
                                     is_finite_scalar(elastic_scale),
                                     is_finite_scalar(elastic_slope),
                                     elastic_slope > zero);

        const S theta_y = d::div(yield_stress, elastic_slope);
        ok = scalar_detail::pred_and(
            ok, is_finite_scalar(theta_y), theta_y >= zero);

        // elastic branch
        const S elastic_energy     = d::mul(d::mul(elastic_scale, delta), delta);
        const S elastic_dEdtheta   = d::mul(elastic_slope, delta);
        const S elastic_ddEddtheta = elastic_slope;

        // plastic branch
        const S delta_gamma = d::sub(abs_value(delta), theta_y);
        const S plastic_energy =
            d::add(d::mul(d::mul(elastic_scale, theta_y), theta_y),
                   d::mul(yield_stress, delta_gamma));
        const S plastic_dEdtheta   = d::mul(sign_value(delta), yield_stress);
        const S plastic_ddEddtheta = zero;

        // reference: `if(abs_value(delta) <= theta_y) { ... } else { ... }`
        const auto elastic = abs_value(delta) <= theta_y;

        energy     = scalar_detail::select_scalar(plastic_energy, elastic_energy, elastic);
        dEdtheta   = scalar_detail::select_scalar(plastic_dEdtheta, elastic_dEdtheta, elastic);
        ddEddtheta = scalar_detail::select_scalar(plastic_ddEddtheta,
                                                  elastic_ddEddtheta,
                                                  elastic);

        ok = scalar_detail::pred_and(ok,
                                     is_finite_scalar(energy),
                                     is_finite_scalar(dEdtheta),
                                     is_finite_scalar(ddEddtheta));
        return ok;
    }

    /**
     * @brief The stress-plastic history update (the reference's verbatim sequence).
     *
     * `theta_bar`/`yield_stress` keep their input value whenever the folded predicate
     * is false, so the caller may write unconditionally and an unmet yield condition
     * leaves the history unchanged.
     */
    template <typename S>
    [[nodiscard]] inline auto update_plastic_state(const S& theta,
                                                   S&       theta_bar,
                                                   S&       yield_stress,
                                                   const S& hardening_modulus,
                                                   const S& kappa,
                                                   const S& L0,
                                                   const S& h_bar)
    {
        namespace d = lps::detail;

        auto ok = scalar_detail::pred_and(is_finite_scalar(theta),
                                         is_finite_scalar(theta_bar),
                                         is_finite_scalar(yield_stress),
                                         is_finite_scalar(hardening_modulus),
                                         hardening_modulus >= lps::make_scalar<S>(0.0));

        S delta       = lps::make_scalar<S>(0.0);
        S tau_trial   = lps::make_scalar<S>(0.0);
        S theta_y     = lps::make_scalar<S>(0.0);
        S delta_gamma = lps::make_scalar<S>(0.0);
        ok            = scalar_detail::pred_and(
            ok,
            try_trial_state(
                theta, theta_bar, kappa, L0, h_bar, yield_stress, delta, tau_trial, theta_y, delta_gamma));

        ok = scalar_detail::pred_and(
            ok, delta_gamma > plasticity_write_threshold<S>());

        const S new_theta_bar =
            wrap_angle(d::add(theta_bar, d::mul(sign_value(delta), delta_gamma)));
        const S new_yield_stress =
            d::add(yield_stress, d::mul(hardening_modulus, delta_gamma));

        ok = scalar_detail::pred_and(ok,
                                     is_finite_scalar(new_theta_bar),
                                     is_finite_scalar(new_yield_stress),
                                     new_yield_stress >= lps::make_scalar<S>(0.0));

        theta_bar    = scalar_detail::select_scalar(theta_bar, new_theta_bar, ok);
        yield_stress = scalar_detail::select_scalar(yield_stress, new_yield_stress, ok);
        return ok;
    }

    /// The reference's `try_angle_delta` (`false` == "the caller must not use
    /// `theta`/`delta`").
    template <typename S>
    [[nodiscard]] inline auto try_angle_delta(const lps::LVec<S, 3>& x0,
                                              const lps::LVec<S, 3>& x1,
                                              const lps::LVec<S, 3>& x2,
                                              const lps::LVec<S, 3>& x3,
                                              const S&               theta_bar,
                                              S&                     theta,
                                              S&                     delta)
    {
        auto ok = scalar_detail::pred_and(safe_dihedral_angle(x0, x1, x2, x3, theta),
                                         is_finite_scalar(theta_bar));

        delta = angle_delta(theta, theta_bar);
        return scalar_detail::pred_and(ok, is_finite_scalar(delta));
    }

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
                             const S&               kappa,
                             const S&               yield_stress)
    {
        namespace SPDSB = sym::stress_plastic_discrete_shell_bending;

        S theta = lps::make_scalar<S>(0.0);
        S delta = lps::make_scalar<S>(0.0);
        auto ok = SPDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta);

        S energy       = lps::make_scalar<S>(0.0);
        S dEdtheta     = lps::make_scalar<S>(0.0);
        S ddEddtheta   = lps::make_scalar<S>(0.0);
        auto ok_response = SPDSB::augmented_response_from_angle_delta(
            delta, kappa, L0, h_bar, yield_stress, energy, dEdtheta, ddEddtheta);

        // reference: `if(!try_angle_delta(...)) return 0.0;` and
        //            `if(!augmented_response_from_angle_delta(...)) return 0.0;`
        return scalar_detail::select_scalar(
            lps::make_scalar<S>(0.0),
            energy,
            scalar_detail::pred_and(ok, ok_response));
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
                     const S&               kappa,
                     const S&               yield_stress)
    {
        namespace SPDSB = sym::stress_plastic_discrete_shell_bending;

        S theta = lps::make_scalar<S>(0.0);
        S delta = lps::make_scalar<S>(0.0);
        auto ok = SPDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta);

        S energy     = lps::make_scalar<S>(0.0);
        S dEdtheta   = lps::make_scalar<S>(0.0);
        S ddEddtheta = lps::make_scalar<S>(0.0);
        auto ok_response = SPDSB::augmented_response_from_angle_delta(
            delta, kappa, L0, h_bar, yield_stress, energy, dEdtheta, ddEddtheta);

        lps::LVec<S, 12> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        auto G_full = dEdtheta * dthetadx;

        // reference: both guard failures `G.setZero(); return;`
        G = scalar_detail::select_container(lps::LVec<S, 12>{},
                             G_full,
                             scalar_detail::pred_and(ok, ok_response));
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
                       const S&                kappa,
                       const S&                yield_stress)
    {
        namespace SPDSB = sym::stress_plastic_discrete_shell_bending;

        S theta = lps::make_scalar<S>(0.0);
        S delta = lps::make_scalar<S>(0.0);
        auto ok = SPDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta);

        S energy     = lps::make_scalar<S>(0.0);
        S dEdtheta   = lps::make_scalar<S>(0.0);
        S ddEddtheta = lps::make_scalar<S>(0.0);
        auto ok_response = SPDSB::augmented_response_from_angle_delta(
            delta, kappa, L0, h_bar, yield_stress, energy, dEdtheta, ddEddtheta);

        lps::LVec<S, 12> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        lps::LMat<S, 12, 12> ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        auto H_full = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;

        // reference: both guard failures `H.setZero(); return;`
        H = scalar_detail::select_container(lps::LMat<S, 12, 12>{},
                             H_full,
                             scalar_detail::pred_and(ok, ok_response));
    }
}  // namespace sym::stress_plastic_discrete_shell_bending

namespace spdsb_detail
{
/// See `pdsb_detail::stencil_energy` (same scale, same association order).
template <typename S>
[[nodiscard]] inline S stencil_energy(const lps::LVec<S, 3>& x0,
                                      const lps::LVec<S, 3>& x1,
                                      const lps::LVec<S, 3>& x2,
                                      const lps::LVec<S, 3>& x3,
                                      const S&               L0,
                                      const S&               h_bar,
                                      const S&               theta_bar,
                                      const S&               kappa,
                                      const S&               yield_stress,
                                      const S&               V_bar,
                                      const S&               dt)
{
    namespace d = lps::detail;
    auto E      = sym::stress_plastic_discrete_shell_bending::E(
        x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, yield_stress);
    // reference: `energies(I) = E * V_bar * dt * dt;`
    return d::mul(d::mul(d::mul(E, V_bar), dt), dt);
}

/// `Float Vdt2 = V_bar * dt * dt;` (the reference's association order).
template <typename S>
[[nodiscard]] inline S stencil_vdt2(const S& V_bar, const S& dt)
{
    namespace d = lps::detail;
    return d::mul(d::mul(V_bar, dt), dt);
}

/// `SPDSB::dEdx(G12, ...); G12 *= Vdt2;`
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> stencil_gradient(const lps::LVec<S, 3>& x0,
                                                       const lps::LVec<S, 3>& x1,
                                                       const lps::LVec<S, 3>& x2,
                                                       const lps::LVec<S, 3>& x3,
                                                       const S&               L0,
                                                       const S&               h_bar,
                                                       const S&               theta_bar,
                                                       const S&               kappa,
                                                       const S&               yield_stress,
                                                       const S&               V_bar,
                                                       const S&               dt)
{
    lps::LVec<S, 12> G12;
    sym::stress_plastic_discrete_shell_bending::dEdx(
        G12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, yield_stress);
    return G12 * stencil_vdt2(V_bar, dt);
}

/// `SPDSB::ddEddx(H12x12, ...); H12x12 *= Vdt2; make_spd(H12x12);`
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> stencil_hessian(const lps::LVec<S, 3>& x0,
                                                          const lps::LVec<S, 3>& x1,
                                                          const lps::LVec<S, 3>& x2,
                                                          const lps::LVec<S, 3>& x3,
                                                          const S&               L0,
                                                          const S&               h_bar,
                                                          const S&               theta_bar,
                                                          const S&               kappa,
                                                          const S&               yield_stress,
                                                          const S&               V_bar,
                                                          const S&               dt)
{
    lps::LMat<S, 12, 12> H12x12;
    sym::stress_plastic_discrete_shell_bending::ddEddx(
        H12x12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, yield_stress);
    H12x12 = H12x12 * stencil_vdt2(V_bar, dt);
    lps::make_spd_inplace(H12x12, scalar_detail::kShellMakeSpdSweeps);
    return H12x12;
}

/**
 * @brief The reference's `..._do_update_state_kernel` body (see
 *        `pdsb_detail::update_state` for the guard folding).
 */
template <typename S>
[[nodiscard]] inline auto update_state(const lps::LVec<S, 3>& x0,
                                       const lps::LVec<S, 3>& x1,
                                       const lps::LVec<S, 3>& x2,
                                       const lps::LVec<S, 3>& x3,
                                       const S&               hardening_modulus,
                                       const S&               kappa,
                                       const S&               L0,
                                       const S&               h_bar,
                                       S&                     theta_bar,
                                       S&                     yield_stress)
{
    namespace SPDSB = sym::stress_plastic_discrete_shell_bending;

    S    theta    = lps::make_scalar<S>(0.0);
    auto ok_angle = SPDSB::safe_dihedral_angle(x0, x1, x2, x3, theta);

    const S theta_bar_in    = theta_bar;
    const S yield_stress_in = yield_stress;

    auto ok_update = SPDSB::update_plastic_state(
        theta, theta_bar, yield_stress, hardening_modulus, kappa, L0, h_bar);

    auto ok_write = scalar_detail::pred_and(ok_angle, ok_update);

    theta_bar    = scalar_detail::select_scalar(theta_bar_in, theta_bar, ok_write);
    yield_stress = scalar_detail::select_scalar(yield_stress_in, yield_stress, ok_write);
    return ok_write;
}
}  // namespace spdsb_detail
}  // namespace uipc::backend::luisa
