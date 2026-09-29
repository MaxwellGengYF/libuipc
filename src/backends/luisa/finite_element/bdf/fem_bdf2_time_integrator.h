#pragma once
/********************************************************************
 * @file fem_bdf2_time_integrator.h
 * @brief T5.3 — the BDF2 FEM time integrator (port of
 * src/backends/cuda/finite_element/bdf/fem_bdf2_time_integrator.cu).
 *
 * DEVIATION (reported): the reference keeps the class *and* its kernels inside
 * the `.cu`, so nothing about it is reachable from a test.  This port follows the
 * T5.2 rule (cookbook §7.17 rule 11) -- the class is declared here, the kernels +
 * their cached shaders + `REGISTER_SIM_SYSTEM` stay in the `.cpp` -- so that the
 * per-vertex bodies can be the *one* generic implementation (the T5.3 brief's
 * §2.1) that the shipped kernel and the unit test both instantiate (cookbook
 * §7.4).
 *
 * BDF2 per-vertex law, transcribed from the reference's kernel (x_n := xs is
 * recorded first, `x_prevs` *is* the x_n buffer):
 *   x_n     = xs
 *   x_n_1   = x_n_1s(i); v_n = vs(i); v_n_1 = v_n_1s(i)
 *   g       = gravity + external_force_acc
 *   if (is_fixed)          x_n_1 = x_n; v_n = 0; v_n_1 = 0; g = 0
 *   else if (!is_dynamic)  x_n_1 = x_n; v_n = 0; v_n_1 = 0
 *   x_tilde = 4/3 x_n - 1/3 x_n_1 + 2/9 h (4 v_n - v_n_1) + 4/9 h^2 g
 * and, in update_state:
 *   v = 1/(2h) (3 x - 4 x_n + x_n_1);  v_n_1s = vs;  vs = v;  x_n_1s = x_n
 * (the *last* three writes use the *old* values on their right-hand sides, so
 * their order is load-bearing).
 *
 * The two `if` chains become `select`s of the already computed candidates, in
 * the reference's own order, so the chosen value passes through untouched
 * (cookbook §7.8/§7.17 rule 19).
 *
 * The bootstrap (the reference's `if(state->x_n_1s().size() != info.xs().size())`
 * branch: resize + `x_n_1s = x_prevs`, `v_n_1s = vs`) lives in the `.cpp`, in
 * `do_predict_dof`, exactly as the reference has it.
 *********************************************************************/
#include <finite_element/fem_time_integrator.h>
#include <time_integrator/bdf2_flag.h>
#include <finite_element/bdf/fem_bdf2_state.h>
#include <finite_element/bdf/bdf2_function.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer; `lla::detail::cached_kernel` is the port's
// `compile1`-with-a-function-local-static idiom for the kernels below.
namespace lla = luisa_device::linalg;

namespace bdf2_detail
{
/// The reference's `FEMBDF2Integrator_do_predict_dof_kernel` body, minus the
/// `x_prevs(i) = xs(i)` bookkeeping copy (which the caller performs).
template <typename S, typename I0, typename I1>
[[nodiscard]] inline lps::LVec<S, 3> predict_dof(const lps::LVec<S, 3>& x_n,
                                                 const lps::LVec<S, 3>& x_n_1,
                                                 const lps::LVec<S, 3>& v_n,
                                                 const lps::LVec<S, 3>& v_n_1,
                                                 const lps::LVec<S, 3>& gravity,
                                                 const lps::LVec<S, 3>& external_force_acc,
                                                 const S&               dt,
                                                 const I0&              is_fixed,
                                                 const I1&              is_dynamic) noexcept
{
    const auto not_fixed = (is_fixed == 0);
    const auto is_dyn    = (is_dynamic != 0);

    const auto zero = lps::make_vec<S, 3>([](int) noexcept { return lps::zero_of<S>(); });

    // The reference's two branches, as mutually exclusive predicates.
    const auto branch_fixed  = !not_fixed;
    const auto branch_static = scalar_detail::pred_and(not_fixed, !is_dyn);
    const auto branch        = scalar_detail::pred_or(branch_fixed, branch_static);

    const auto g = gravity + external_force_acc;

    const auto x_n_1_sel = scalar_detail::select_container(x_n_1, x_n, branch);
    const auto v_n_sel   = scalar_detail::select_container(v_n, zero, branch);
    const auto v_n_1_sel = scalar_detail::select_container(v_n_1, zero, branch);
    const auto g_sel     = scalar_detail::select_container(g, zero, branch_fixed);

    lps::LVec<S, 3> x_tilde;
    sym::fem_bdf2::compute_x_tilde(x_tilde, x_n, x_n_1_sel, v_n_sel, v_n_1_sel, g_sel, dt);
    return x_tilde;
}
}  // namespace bdf2_detail

class FEMBDF2Integrator final : public FEMTimeIntegrator
{
  public:
    using FEMTimeIntegrator::FEMTimeIntegrator;

    SimSystemSlot<FEMBDF2State> state;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_predict_dof(PredictDofInfo& info) override;
    void do_update_state(UpdateVelocityInfo& info) override;
};
}  // namespace uipc::backend::luisa
