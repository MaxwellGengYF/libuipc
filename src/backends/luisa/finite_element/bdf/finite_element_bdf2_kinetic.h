#pragma once
/********************************************************************
 * @file finite_element_bdf2_kinetic.h
 * @brief T5.3 — the BDF2 kinetic system (port of
 * src/backends/cuda/finite_element/bdf/finite_element_bdf2_kinetic.cu).
 *
 * DEVIATION (reported): the reference keeps the class *and* its kernels inside
 * the `.cu`.  This port declares the class here and keeps the kernels, their
 * cached shaders and `REGISTER_SIM_SYSTEM` in the `.cpp` (the T5.2 rule, cookbook
 * §7.17 rule 11), so that the per-vertex bodies are the *one* generic
 * implementation (the T5.3 brief's §2.1) that the shipped kernel and the unit test
 * both instantiate (cookbook §7.4).
 *
 * The BDF2 kinetic laws, verbatim from the reference's kernels (dx = x - x_tilde,
 * `inv_beta` is the *device/host argument* the kernel takes):
 *   K = is_fixed ? 0 : inv_beta / 2 * dot(dx, m * dx)
 *   G = is_fixed ? 0 : inv_beta * m * dx
 *   H = (inv_beta * m) * I                       (no `is_fixed` guard, like BDF1)
 * with `beta = 4/9` and `inv_beta = 1 / beta = 9/4` (`Float` is `double` here, so
 * both are exact).  The unit test asserts `beta == 4.0/9.0` and
 * `inv_beta == 9.0/4.0` *as doubles*, which catches a float-literal transcription.
 *********************************************************************/
#include <finite_element/finite_element_kinetic.h>
#include <time_integrator/bdf2_flag.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer; `lla::detail::cached_kernel` is the port's
// `compile1`-with-a-function-local-static idiom for the kernels below.
namespace lla = luisa_device::linalg;

/// The reference's two `FiniteElementBDF2Kinetic_*` kernel bodies, per vertex.
namespace bdf2_kinetic_detail
{
/// `K = is_fixed ? 0 : inv_beta / 2 * dot(dx, m * dx)`
template <typename S, typename I>
[[nodiscard]] inline S energy(const lps::LVec<S, 3>& x,
                              const lps::LVec<S, 3>& x_tilde,
                              const S&               m,
                              const S&               inv_beta,
                              const I&               is_fixed) noexcept
{
    const auto dx = x - x_tilde;
    const auto K =
        lps::detail::mul(lps::detail::div(inv_beta, 2.0), lps::dot(dx, m * dx));
    return scalar_detail::select_scalar(K, lps::zero_of<S>(), is_fixed != 0);
}

/// `G = is_fixed ? 0 : inv_beta * m * dx`
template <typename S, typename I>
[[nodiscard]] inline lps::LVec<S, 3> gradient(const lps::LVec<S, 3>& x,
                                              const lps::LVec<S, 3>& x_tilde,
                                              const S&               m,
                                              const S&               inv_beta,
                                              const I&               is_fixed) noexcept
{
    const auto dx = x - x_tilde;
    const auto G  = lps::detail::mul(inv_beta, m) * dx;
    return scalar_detail::select_container(
        G,
        lps::make_vec<S, 3>([](int) noexcept { return lps::zero_of<S>(); }),
        is_fixed != 0);
}

/// `H = (inv_beta * m) * I` (unconditional -- see the file comment)
template <typename S>
[[nodiscard]] inline lps::LMat<S, 3, 3> hessian(const S& m, const S& inv_beta) noexcept
{
    return lps::detail::mul(inv_beta, m) * lps::LMat<S, 3, 3>::Identity();
}
}  // namespace bdf2_kinetic_detail

class FiniteElementBDF2Kinetic final : public FiniteElementKinetic
{
  public:
    static constexpr Float beta     = 4.0 / 9.0;  // BDF2 beta coefficient
    static constexpr Float inv_beta = Float{1} / beta;

    using FiniteElementKinetic::FiniteElementKinetic;

    void do_build(BuildInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
