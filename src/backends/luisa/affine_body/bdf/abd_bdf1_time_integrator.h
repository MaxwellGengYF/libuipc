#pragma once
/********************************************************************
 * @file affine_body/bdf/abd_bdf1_time_integrator.h
 * @brief T6.2 — the BDF1 ABD time integrator (port of
 * src/backends/cuda/affine_body/bdf/abd_bdf1_time_integrator.cu).
 *
 * DEVIATION (reported): the reference keeps the class *and* its kernels inside the
 * `.cu`, so nothing about it is reachable from a test.  This port follows the T5.2
 * rule (cookbook §7.17 rule 11), exactly like the FEM twin
 * `fem_bdf1_time_integrator.h`: the class is declared in this header, the kernels +
 * their cached shaders + `REGISTER_SIM_SYSTEM` stay in the `.cpp`.  The reason is
 * the T6.2 brief's §1 rule — *one* generic implementation per routine over the
 * scalar type — which the reference satisfies with a `__global__` kernel that CUDA
 * can also run on the host; the DSL cannot, so the per-body bodies below are
 * ordinary templates that the shipped kernel and the unit test both instantiate
 * (cookbook §7.4).
 *
 * BDF1 (the default: `integrator/type == "bdf1"`) per-body law, transcribed from the
 * reference's kernel (`fx = is_fixed(i) != 0`, `g = affine_gravity(i)`):
 *   q_prev  = qs
 *   q_tilde = q_prev
 *   if (!fx)                  q_tilde += (g + f_ext_acc) * dt * dt
 *   if (!fx && is_dynamic)    q_tilde += q_v * dt
 *   q_v     = (q - q_prev) / dt
 * `q` is the 12-dof affine state `[t; a1; a2; a3]` (cookbook §7.21 rule 6), so
 * `dt * dt * g` is the *generalized acceleration* the dynamics pre-computes as
 * `M^-1 G` (affine_body_dynamics.cpp step 6) — see §7.23.  The device form replaces
 * the two branches by a `select` of the *already computed* candidates, in the
 * reference's own order and association, so the chosen value is passed through
 * untouched (cookbook §7.8/§7.17 rule 19).
 *********************************************************************/
#include <affine_body/abd_time_integrator.h>
#include <time_integrator/bdf1_flag.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer; `lla::detail::cached_kernel` is the port's
// `compile1`-with-a-function-local-static idiom for the kernels below
// (cookbook §7.10 rule 1 / §7.15 rule 6).
namespace lla = luisa_device::linalg;

/// The reference's `abd_bdf1_integrator_predict_dof_kernel` /
/// `abd_bdf1_integrator_update_state_kernel` bodies, per affine body, generic over
/// the scalar type (`double` on the host, `lcrt::Var<double>` in a kernel) and over
/// the two index types.
namespace bdf1_detail
{
/// Returns `q_tilde` for body `i`.  `q_prevs(i) = qs(i)` is the caller's write (it
/// is unconditional in the reference and does not depend on the flags).
template <typename S, typename I0, typename I1>
[[nodiscard]] inline lps::LVec<S, 12> predict_dof(const lps::LVec<S, 12>& qs,
                                                 const lps::LVec<S, 12>& q_v,
                                                 const lps::LVec<S, 12>& affine_gravity,
                                                 const lps::LVec<S, 12>& external_force_acc,
                                                 const S&                dt,
                                                 const I0&               is_fixed,
                                                 const I1&               is_dynamic) noexcept
{
    const auto not_fixed = (is_fixed == 0);
    const auto is_dyn    = (is_dynamic != 0);

    // 0) fixed: q_tilde = q_prev
    auto q_tilde = qs;

    // 1) static problem: q_tilde = q_prev + (g + f_ext_acc) * dt * dt
    auto q_tilde_static = q_tilde + (affine_gravity + external_force_acc) * dt * dt;
    q_tilde = scalar_detail::select_container(q_tilde, q_tilde_static, not_fixed);

    // 2) dynamic problem q_tilde = q_prev + q_v * dt + (g + f_ext_acc) * dt * dt
    auto q_tilde_dynamic = q_tilde + q_v * dt;
    q_tilde              = scalar_detail::select_container(
        q_tilde, q_tilde_dynamic, scalar_detail::pred_and(not_fixed, is_dyn));

    return q_tilde;
}

/// Returns the new 12-dof velocity of body `i`: `(q - q_prev) * (1 / dt)`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> update_state(const lps::LVec<S, 12>& q,
                                                   const lps::LVec<S, 12>& q_prev,
                                                   const S&                dt) noexcept
{
    return (q - q_prev) * lps::detail::div(lps::one_of<S>(), dt);
}
}  // namespace bdf1_detail

class ABDBDF1Integrator final : public ABDTimeIntegrator
{
  public:
    using ABDTimeIntegrator::ABDTimeIntegrator;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_predict_dof(PredictDofInfo& info) override;
    void do_update_state(UpdateVelocityInfo& info) override;
};
}  // namespace uipc::backend::luisa
