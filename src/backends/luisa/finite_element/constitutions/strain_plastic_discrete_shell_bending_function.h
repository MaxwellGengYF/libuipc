#pragma once
/********************************************************************
 * @file strain_plastic_discrete_shell_bending_function.h
 * @brief T5.2 (batch b) — strain-plastic discrete shells: energy, gradient, Hessian
 *        and the per-element history update (port of
 *        src/backends/cuda/finite_element/constitutions/
 *        strain_plastic_discrete_shell_bending_function.h).
 *
 * The reference is *hand written* and guard-heavy: `safe_dihedral_angle`,
 * `try_angle_delta` and `update_plastic_state` return a `bool` that the callers use
 * to *return early* (energy 0, zero gradient/Hessian, "do not touch the history").
 * Those `bool`s are device predicates on the device path, so the port cannot use
 * host control flow (RULE M2) and cannot use the reference's early returns.  The
 * restructuring, applied uniformly to every guarded routine here:
 *
 *   * evaluate the whole computation unconditionally — every quantity on the
 *     guarded path is NaN-safe by construction (the cosine is clamped before
 *     `acos`), so a discarded branch never traps and never changes a kept one;
 *   * fold the guard chain into a single predicate in the *reference's order*
 *     (`scalar_detail::pred_and`) and select on it at the point where the reference
 *     returns:
 *       - `return 0.0;`          -> `select_scalar(0.0, value, ok)`
 *       - `G.setZero(); return;` -> `select` per element (the gradient/Hessian
 *         helpers below return zero when `ok` is false)
 *       - `return false;` from a routine with out-parameters
 *                                -> the out-parameters are `select`ed back to their
 *                                   input value, and the caller's write becomes an
 *                                   identity when `ok` is false (so the history is
 *                                   *not* modified, exactly like the reference).
 *   The equivalence argument is that a `select` passes its chosen operand through
 *   untouched: the kept value is bit-identical to the value the reference computes
 *   on the branch it takes.
 *
 * The one place where the reference's *unbounded* `while` loop cannot be ported is
 * `wrap_angle` — see its comment and the unit test (`wrap_angle` step form vs the
 * reference's `while` loops over |a| <= 10 pi, bit-identical).
 *
 * `pdsb_detail` factors the reference's kernels into generic per-element routines
 * (cookbook §7.4); the constitution's kernels and the unit test instantiate *the
 * same* code.
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace (T2.2)
#include <finite_element/constitutions/discrete_shell_bending_reference.h>
#include <utils/dihedral_angle.h>
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace sym::strain_plastic_discrete_shell_bending
{
#include "sym/discrete_shell_bending.inl"

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

    template <typename S>
    [[nodiscard]] inline auto is_finite_vec3(const lps::LVec<S, 3>& v)
    {
        return scalar_detail::is_finite_vec3(v);
    }

    /**
     * @brief Wrap an angle into [-pi, pi] (the reference's two `while` loops).
     *
     * RESTRUCTURE (reported): `while(angle > Pi) angle -= TwoPi;` is a *device*
     * loop whose trip count is device data, so it becomes a fixed 5-step predicated
     * form.  5 steps cover |angle| <= 11 pi, which is orders of magnitude above every
     * call site in this file: `angle_delta` is fed `theta - theta_bar` with both angles
     * in [-pi, pi] (|a| <= 2 pi), and the plastic update feeds
     * `theta_bar + direction * plastic_excess` with `|plastic_excess| <= 2 pi`
     * (|a| <= 3 pi).  For such inputs the step form is *bit-identical* to the
     * reference's loops (each step is exactly the reference's `angle -= TwoPi`), and
     * the unit test asserts that over |a| <= 10 pi.
     */
    template <typename S>
    [[nodiscard]] inline S wrap_angle(const S& angle)
    {
        namespace d  = lps::detail;
        const S Pi   = pi<S>();
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

    /**
     * @brief `safe_dihedral_angle(v0, v1, v2, v3, theta)`, without early returns.
     *
     * The reference's `return false` sites become the folded predicate returned
     * here; `theta` is always written (with the clamped/`acos`/signed value), and the
     * caller selects on the returned predicate — a discarded `theta` is never used.
     */
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

        // reference: `if(n2.cross(n1).dot(v1 - v2) < 0) theta = -theta;`
        theta = scalar_detail::select_scalar(
            theta,
            d::neg(theta),
            lps::cross(n2, n1).dot(v1 - v2) < lps::make_scalar<S>(0.0));

        // the reference's guard chain, folded in its own order
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
     * @brief The strain-plastic history update (the reference's verbatim sequence).
     *
     * The reference mutates `theta_bar`/`yield_threshold` and only when its guards
     * pass; the port computes the updated values and selects them back to the input
     * when the folded predicate is false, so the caller can write unconditionally
     * and the history is unmodified when the yield condition is not met.
     */
    template <typename S>
    [[nodiscard]] inline auto update_plastic_state(const S& theta,
                                                   S&       theta_bar,
                                                   S&       yield_threshold,
                                                   const S& hardening_modulus)
    {
        namespace d = lps::detail;

        auto ok = scalar_detail::pred_and(is_finite_scalar(theta),
                                         is_finite_scalar(theta_bar),
                                         is_finite_scalar(yield_threshold),
                                         is_finite_scalar(hardening_modulus),
                                         yield_threshold >= lps::make_scalar<S>(0.0),
                                         hardening_modulus >= lps::make_scalar<S>(0.0));

        const S delta          = angle_delta(theta, theta_bar);
        const S plastic_excess = d::sub(abs_value(delta), yield_threshold);

        ok = scalar_detail::pred_and(
            ok,
            is_finite_scalar(plastic_excess),
            plastic_excess > plasticity_write_threshold<S>());

        const S direction = scalar_detail::select_scalar(lps::make_scalar<S>(-1.0),
                                                        lps::make_scalar<S>(1.0),
                                                        delta >= lps::make_scalar<S>(0.0));

        const S new_theta_bar =
            wrap_angle(d::add(theta_bar, d::mul(direction, plastic_excess)));
        const S new_yield_threshold =
            d::add(yield_threshold, d::mul(hardening_modulus, plastic_excess));

        ok = scalar_detail::pred_and(ok,
                                     is_finite_scalar(new_theta_bar),
                                     is_finite_scalar(new_yield_threshold),
                                     new_yield_threshold >= lps::make_scalar<S>(0.0));

        theta_bar       = scalar_detail::select_scalar(theta_bar, new_theta_bar, ok);
        yield_threshold = scalar_detail::select_scalar(
            yield_threshold, new_yield_threshold, ok);
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
                             const S&               kappa)
    {
        namespace PDSB = sym::strain_plastic_discrete_shell_bending;
        namespace d    = lps::detail;

        S theta = lps::make_scalar<S>(0.0);
        S delta = lps::make_scalar<S>(0.0);
        auto ok = PDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta);

        S R;
        PDSB::E(R, kappa, d::add(theta_bar, delta), theta_bar, L0, h_bar);

        // reference: `if(!try_angle_delta(...)) return 0.0;`
        return scalar_detail::select_scalar(lps::make_scalar<S>(0.0), R, ok);
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
        namespace PDSB = sym::strain_plastic_discrete_shell_bending;
        namespace d    = lps::detail;

        S theta = lps::make_scalar<S>(0.0);
        S delta = lps::make_scalar<S>(0.0);
        auto ok = PDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta);

        S dEdtheta;
        PDSB::dEdtheta(dEdtheta, kappa, d::add(theta_bar, delta), theta_bar, L0, h_bar);

        lps::LVec<S, 12> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        auto G_full = dEdtheta * dthetadx;

        // reference: `if(!try_angle_delta(...)) { G.setZero(); return; }`
        G = scalar_detail::select_container(lps::LVec<S, 12>{}, G_full, ok);
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
        namespace PDSB = sym::strain_plastic_discrete_shell_bending;
        namespace d    = lps::detail;

        S theta = lps::make_scalar<S>(0.0);
        S delta = lps::make_scalar<S>(0.0);
        auto ok = PDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta);

        S dEdtheta;
        PDSB::dEdtheta(dEdtheta, kappa, d::add(theta_bar, delta), theta_bar, L0, h_bar);

        S ddEddtheta;
        PDSB::ddEddtheta(ddEddtheta, kappa, d::add(theta_bar, delta), theta_bar, L0, h_bar);

        lps::LVec<S, 12> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        lps::LMat<S, 12, 12> ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        auto H_full = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;

        // reference: `if(!try_angle_delta(...)) { H.setZero(); return; }`
        H = scalar_detail::select_container(lps::LMat<S, 12, 12>{}, H_full, ok);
    }
}  // namespace sym::strain_plastic_discrete_shell_bending

namespace pdsb_detail
{
/**
 * @brief The reference's `..._do_compute_energy_kernel` body (energy * V_bar * dt^2,
 *        same association order as the reference).
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
    auto E = sym::strain_plastic_discrete_shell_bending::E(
        x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
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

/// `PDSB::dEdx(G12, ...); G12 *= Vdt2;`
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
    sym::strain_plastic_discrete_shell_bending::dEdx(
        G12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
    return G12 * stencil_vdt2(V_bar, dt);
}

/// `PDSB::ddEddx(H12x12, ...); H12x12 *= Vdt2; make_spd(H12x12);` — scale *before*
/// the projection, exactly like the elastic sibling (and unlike the NH shell).
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
    sym::strain_plastic_discrete_shell_bending::ddEddx(
        H12x12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
    H12x12 = H12x12 * stencil_vdt2(V_bar, dt);
    lps::make_spd_inplace(H12x12, scalar_detail::kShellMakeSpdSweeps);
    return H12x12;
}

/**
 * @brief The reference's `..._do_update_state_kernel` body — the *history update*,
 *        one call per element and per frame.
 *
 * The reference reads the element's `theta_bar`/`yield_threshold`, computes the
 * dihedral angle from the *current* positions, returns early when the angle is not
 * usable, and writes both attributes back only when `update_plastic_state` reports a
 * (non-degenerate) plastic step.  The port computes both stages with the folded
 * predicates (`ok_angle & ok_update`) and selects: `theta_bar`/`yield_threshold` keep
 * their input value unless the reference would have written them, so a caller may
 * write unconditionally — an unmoved history is *not* a change.
 *
 * @return the "the reference wrote the history" predicate.
 */
template <typename S>
[[nodiscard]] inline auto update_state(const lps::LVec<S, 3>& x0,
                                       const lps::LVec<S, 3>& x1,
                                       const lps::LVec<S, 3>& x2,
                                       const lps::LVec<S, 3>& x3,
                                       const S&               hardening_modulus,
                                       S&                     theta_bar,
                                       S&                     yield_threshold)
{
    namespace PDSB = sym::strain_plastic_discrete_shell_bending;

    S    theta    = lps::make_scalar<S>(0.0);
    auto ok_angle = PDSB::safe_dihedral_angle(x0, x1, x2, x3, theta);

    const S theta_bar_in       = theta_bar;
    const S yield_threshold_in = yield_threshold;

    auto ok_update = PDSB::update_plastic_state(
        theta, theta_bar, yield_threshold, hardening_modulus);

    // reference: `if(!safe_dihedral_angle(...)) return;` (stage 1) and
    //            `if(update_plastic_state(...)) { ... }` (stage 2)
    auto ok_write = scalar_detail::pred_and(ok_angle, ok_update);

    theta_bar = scalar_detail::select_scalar(theta_bar_in, theta_bar, ok_write);
    yield_threshold =
        scalar_detail::select_scalar(yield_threshold_in, yield_threshold, ok_write);
    return ok_write;
}
}  // namespace pdsb_detail
}  // namespace uipc::backend::luisa
