#pragma once
/********************************************************************
 * @file fem_bdf1_time_integrator.h
 * @brief T5.3 — the BDF1 FEM time integrator (port of
 * src/backends/cuda/finite_element/bdf/fem_bdf1_time_integrator.cu).
 *
 * DEVIATION (reported): the reference keeps the class *and* its kernels inside
 * the `.cu`, so nothing about it is reachable from a test.  This port follows the
 * T5.2 rule (cookbook §7.17 rule 11): the class is declared in this header, the
 * kernels + their cached shaders + `REGISTER_SIM_SYSTEM` stay in the `.cpp`.
 * The reason is the T5.3 brief's §2.1 rule -- *one* generic implementation per
 * routine over the scalar type -- which the reference satisfies with a
 * `__global__` kernel that CUDA can also run on the host; the DSL cannot, so the
 * per-vertex bodies below are ordinary templates that the shipped kernel and the
 * unit test both instantiate (cookbook §7.4).
 *
 * BDF1 (the default: `integrator/type == "bdf1"`) per-vertex law, transcribed
 * from the reference's kernel:
 *   x_prev  = xs
 *   x_tilde = x_prev
 *   if (!is_fixed)            x_tilde += (g + f_ext_acc) * dt * dt
 *   if (!is_fixed && is_dyn)  x_tilde += v * dt
 *   v       = (x - x_prev) / dt
 * The device form replaces the two branches by a `select` of the *already
 * computed* candidates, in the reference's own order and association, so the
 * chosen value is passed through untouched (cookbook §7.8/§7.17 rule 19).
 *********************************************************************/
#include <finite_element/fem_time_integrator.h>
#include <time_integrator/bdf1_flag.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer; `lla::detail::cached_kernel` is the port's
// `compile1`-with-a-function-local-static idiom for the kernels below
// (cookbook §7.10 rule 1 / §7.15 rule 6).
namespace lla = luisa_device::linalg;

/// The reference's `FEMBDF1Integrator_do_predict_dof_kernel` /
/// `FEMBDF1Integrator_do_update_state_kernel` bodies, per vertex, generic over
/// the scalar type (`double` on the host, `lcrt::Var<double>` in a kernel) and
/// over the two index types.
namespace bdf1_detail
{
/// Returns `x_tilde` for vertex `i`.  `x_prevs(i) = xs(i)` is the caller's write
/// (it is unconditional in the reference and does not depend on the flags).
template <typename S, typename I0, typename I1>
[[nodiscard]] inline lps::LVec<S, 3> predict_dof(const lps::LVec<S, 3>& xs,
                                                 const lps::LVec<S, 3>& vs,
                                                 const lps::LVec<S, 3>& gravity,
                                                 const lps::LVec<S, 3>& external_force_acc,
                                                 const S&               dt,
                                                 const I0&              is_fixed,
                                                 const I1&              is_dynamic) noexcept
{
    const auto not_fixed = (is_fixed == 0);
    const auto is_dyn    = (is_dynamic != 0);

    // 0) fixed: x_tilde = x_prev
    auto x_tilde = xs;

    // 1) static problem: x_tilde = x_prev + (g + f_ext_acc) * dt * dt
    auto x_tilde_static = x_tilde + (gravity + external_force_acc) * dt * dt;
    x_tilde             = scalar_detail::select_container(x_tilde, x_tilde_static, not_fixed);

    // 2) dynamic problem: x_tilde = x_prev + v * dt + (g + f_ext_acc) * dt * dt
    auto x_tilde_dynamic = x_tilde + vs * dt;
    x_tilde              = scalar_detail::select_container(
        x_tilde, x_tilde_dynamic, scalar_detail::pred_and(not_fixed, is_dyn));

    return x_tilde;
}

/// Returns the new velocity of vertex `i`: `(x - x_prev) * (1 / dt)`.
template <typename S>
[[nodiscard]] inline lps::LVec<S, 3> update_state(const lps::LVec<S, 3>& x,
                                                  const lps::LVec<S, 3>& x_prev,
                                                  const S&               dt) noexcept
{
    return (x - x_prev) * lps::detail::div(lps::one_of<S>(), dt);
}
}  // namespace bdf1_detail

class FEMBDF1Integrator final : public FEMTimeIntegrator
{
  public:
    using FEMTimeIntegrator::FEMTimeIntegrator;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_predict_dof(PredictDofInfo& info) override;
    void do_update_state(UpdateVelocityInfo& info) override;
};
}  // namespace uipc::backend::luisa
