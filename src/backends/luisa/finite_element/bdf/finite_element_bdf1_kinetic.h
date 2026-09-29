#pragma once
/********************************************************************
 * @file finite_element_bdf1_kinetic.h
 * @brief T5.3 — the BDF1 kinetic system (port of
 * src/backends/cuda/finite_element/bdf/finite_element_bdf1_kinetic.cu).
 *
 * DEVIATION (reported): the reference keeps the class *and* its kernels inside
 * the `.cu`, so nothing about it is reachable from a test.  This port follows the
 * T5.2 rule (cookbook §7.17 rule 11) -- the class is declared here, the kernels +
 * their cached shaders + `REGISTER_SIM_SYSTEM` stay in the `.cpp` -- so that the
 * per-vertex bodies can be the *one* generic implementation (the T5.3 brief's
 * §2.1) that the shipped kernel and the unit test both instantiate (cookbook
 * §7.4).
 *
 * The BDF1 kinetic laws, verbatim from the reference's kernels (dx = x - x_tilde):
 *   K = is_fixed ? 0 : 0.5 * m * dot(dx, dx)
 *   G = is_fixed ? 0 : m * dx                     (the diagonal doublet block i)
 *   H = m * I                                     (the diagonal triplet block [i,i])
 * NOTE the Hessian has **no** `is_fixed` guard in the reference: a fixed vertex
 * still receives `m I` on the diagonal (the mass regularisation of the implicit
 * system).  The unit test asserts that behaviour (`fem_dynamics_kinetic_host_oracle`)
 * because it is easy to "fix" by accident.
 *********************************************************************/
#include <finite_element/finite_element_kinetic.h>
#include <time_integrator/bdf1_flag.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer; `lla::detail::cached_kernel` is the port's
// `compile1`-with-a-function-local-static idiom for the kernels below.
namespace lla = luisa_device::linalg;

/// The reference's two `FiniteElementBDF1Kinetic_*` kernel bodies, per vertex,
/// generic over the scalar type and over the index type of `is_fixed`.
namespace bdf1_kinetic_detail
{
/// `K = is_fixed ? 0 : 0.5 * m * dot(dx, dx)`
template <typename S, typename I>
[[nodiscard]] inline S energy(const lps::LVec<S, 3>& x,
                              const lps::LVec<S, 3>& x_tilde,
                              const S&               m,
                              const I&               is_fixed) noexcept
{
    const auto dx = x - x_tilde;
    const auto K  = lps::detail::mul(lps::detail::mul(0.5, m), lps::dot(dx, dx));
    return scalar_detail::select_scalar(K, lps::zero_of<S>(), is_fixed != 0);
}

/// `G = is_fixed ? 0 : m * dx`
template <typename S, typename I>
[[nodiscard]] inline lps::LVec<S, 3> gradient(const lps::LVec<S, 3>& x,
                                              const lps::LVec<S, 3>& x_tilde,
                                              const S&               m,
                                              const I&               is_fixed) noexcept
{
    const auto dx = x - x_tilde;
    const auto G  = m * dx;
    return scalar_detail::select_container(
        G,
        lps::make_vec<S, 3>([](int) noexcept { return lps::zero_of<S>(); }),
        is_fixed != 0);
}

/// `H = m * I` (unconditional -- see the file comment)
template <typename S>
[[nodiscard]] inline lps::LMat<S, 3, 3> hessian(const S& m) noexcept
{
    return m * lps::LMat<S, 3, 3>::Identity();
}
}  // namespace bdf1_kinetic_detail

class FiniteElementBDF1Kinetic final : public FiniteElementKinetic
{
  public:
    using FiniteElementKinetic::FiniteElementKinetic;

    void do_build(BuildInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
