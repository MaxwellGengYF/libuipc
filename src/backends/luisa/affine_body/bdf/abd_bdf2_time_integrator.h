#pragma once
/********************************************************************
 * @file affine_body/bdf/abd_bdf2_time_integrator.h
 * @brief T6.2 — the BDF2 ABD time integrator (port of
 * src/backends/cuda/affine_body/bdf/abd_bdf2_time_integrator.cu).
 *
 * DEVIATION (reported): the reference keeps the class *and* its kernels inside the
 * `.cu`; this port declares the class here and keeps the kernels, their cached
 * shaders and `REGISTER_SIM_SYSTEM` in the `.cpp` (the T5.2 rule, cookbook §7.17
 * rule 11), so that the per-body bodies are the *one* generic implementation (the
 * T6.2 brief's §1) that the shipped kernel and the unit test both instantiate
 * (cookbook §7.4).
 *
 * BDF2 per-body law, transcribed from the reference's kernel (q_n := qs is recorded
 * first, `q_prevs` *is* the q_n buffer):
 *   q_n     = qs
 *   q_n_1   = q_n_1s(i); q_v_n = q_v_ns(i); q_v_n_1 = q_v_n_1s(i)
 *   g       = gravity
 *   if (is_fixed)          q_n_1 = q_n; q_v_n = 0; q_v_n_1 = 0; g = 0
 *   else if (!is_dynamic)  q_n_1 = q_n; q_v_n = 0; q_v_n_1 = 0
 *   q_tilde = 4/3 q_n - 1/3 q_n_1 + 2/9 h (4 q_v_n - q_v_n_1) + 4/9 h^2 g
 * and, in update_state:
 *   q_v = 1/(2h) (3 q - 4 q_n + q_n_1);  q_v_n_1s = q_v_ns;  q_v_ns = q_v;
 *   q_n_1s = q_n
 * (the *last* three writes use the *old* values on their right-hand sides, so their
 * order is load-bearing).
 *
 * DEVIATION-STYLE ASYMMETRY (kept on purpose, reported): the reference's ABD BDF2
 * predict kernel has **no** `external_force_accs` argument — only `gravities()` — while
 * its BDF1 sibling and the *FEM* BDF2 twin both add `gravity + f_ext_acc`.  The port
 * keeps that asymmetry verbatim (dropping the parameter would be a "fix"), so
 * `bdf2_detail::predict_dof` below takes `gravity` alone.
 *
 * The two `if` chains become `select`s of the already computed candidates, in the
 * reference's own order, so the chosen value passes through untouched
 * (cookbook §7.8/§7.17 rule 19).
 *
 * The bootstrap (the reference's `if(state->q_n_1s().size() != info.qs().size())`
 * branch: resize + `q_n_1s = q_prevs`, `q_v_n_1s = q_vs`) lives in the `.cpp`, in
 * `do_predict_dof`, exactly as the reference has it.
 *********************************************************************/
#include <affine_body/abd_time_integrator.h>
#include <time_integrator/bdf2_flag.h>
#include <affine_body/bdf/abd_bdf2_state.h>
#include <affine_body/bdf/bdf2_function.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer; `lla::detail::cached_kernel` is the port's
// `compile1`-with-a-function-local-static idiom for the kernels below.
namespace lla = luisa_device::linalg;

namespace bdf2_detail
{
/// The reference's `abd_bdf2_integrator_predict_dof_kernel` body, minus the
/// `q_prevs(i) = qs(i)` bookkeeping copy (which the caller performs).
template <typename S, typename I0, typename I1>
[[nodiscard]] inline lps::LVec<S, 12> predict_dof(const lps::LVec<S, 12>& q_n,
                                                 const lps::LVec<S, 12>& q_n_1,
                                                 const lps::LVec<S, 12>& q_v_n,
                                                 const lps::LVec<S, 12>& q_v_n_1,
                                                 const lps::LVec<S, 12>& gravity,
                                                 const S&                dt,
                                                 const I0&               is_fixed,
                                                 const I1&               is_dynamic) noexcept
{
    const auto not_fixed = (is_fixed == 0);
    const auto is_dyn    = (is_dynamic != 0);

    const auto zero = lps::make_vec<S, 12>([](int) noexcept { return lps::zero_of<S>(); });

    // The reference's two branches, as mutually exclusive predicates.
    const auto branch_fixed  = !not_fixed;
    const auto branch_static = scalar_detail::pred_and(not_fixed, !is_dyn);
    const auto branch        = scalar_detail::pred_or(branch_fixed, branch_static);

    const auto g = gravity;

    const auto q_n_1_sel   = scalar_detail::select_container(q_n_1, q_n, branch);
    const auto q_v_n_sel   = scalar_detail::select_container(q_v_n, zero, branch);
    const auto q_v_n_1_sel = scalar_detail::select_container(q_v_n_1, zero, branch);
    const auto g_sel       = scalar_detail::select_container(g, zero, branch_fixed);

    lps::LVec<S, 12> q_tilde;
    sym::abd_bdf2::compute_q_tilde(q_tilde, q_n, q_n_1_sel, q_v_n_sel, q_v_n_1_sel, g_sel, dt);
    return q_tilde;
}
}  // namespace bdf2_detail

class ABDBDF2Integrator final : public ABDTimeIntegrator
{
  public:
    using ABDTimeIntegrator::ABDTimeIntegrator;

    SimSystemSlot<ABDBDF2State> state;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_predict_dof(PredictDofInfo& info) override;
    void do_update_state(UpdateVelocityInfo& info) override;
};
}  // namespace uipc::backend::luisa
