#pragma once
/********************************************************************
 * @file affine_body/bdf/affine_body_bdf2_kinetic.h
 * @brief T6.2 — the BDF2 ABD kinetic system (port of
 * src/backends/cuda/affine_body/bdf/affine_body_bdf2_kinetic.cu).
 *
 * DEVIATION (reported): the reference keeps the class *and* its kernels inside the
 * `.cu`.  This port declares the class here and keeps the kernels, their cached
 * shaders and `REGISTER_SIM_SYSTEM` in the `.cpp` (the T5.2 rule, cookbook §7.17
 * rule 11), so that the per-body bodies are the *one* generic implementation (the
 * T6.2 brief's §1) that the shipped kernel and the unit test both instantiate
 * (cookbook §7.4).
 *
 * The BDF2 kinetic laws, verbatim from the reference's kernels (dq = q - q_tilde,
 * `M` the per-body dyadic mass, `inv_beta` the *device/host argument* the kernel
 * takes):
 *   K = (is_fixed || ext_kinetic) ? 0 : (0.5 * inv_beta) * dq.dot(M * dq)
 *   G = is_fixed ? 0 : inv_beta * (M * dq)
 *   H = inv_beta * M.to_mat()                    (no `is_fixed` guard, like BDF1)
 * with `beta = 4/9` and `inv_beta = 1 / beta = 9/4` (`Float` is `double` here, so
 * both are exact).  The unit test asserts `beta == 4.0/9.0` and
 * `inv_beta == 9.0/4.0` *as doubles*, which catches a float-literal transcription.
 *********************************************************************/
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <time_integrator/bdf2_flag.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer; `lla::detail::cached_kernel` is the port's
// `compile1`-with-a-function-local-static idiom for the kernels below.
namespace lla = luisa_device::linalg;

/// The reference's two `AffineBodyBDF2Kinetic_*` kernel bodies, per body.
namespace bdf2_kinetic_detail
{
/// `K = (is_fixed || ext_kinetic) ? 0 : (0.5 * inv_beta) * dq.dot(M * dq)`
template <typename S, typename I0, typename I1>
[[nodiscard]] inline S energy(const lps::LVec<S, 12>&        q,
                              const lps::LVec<S, 12>&        q_tilde,
                              const ABDJacobiDyadicMassT<S>& m,
                              const S&                       inv_beta,
                              const I0&                      is_fixed,
                              const I1&                      ext_kinetic) noexcept
{
    const auto dq = q - q_tilde;
    const auto K  = lps::detail::mul(lps::detail::mul(0.5, inv_beta),
                                    lps::dot(dq, abd_dyadic_mass_mul(m, dq)));
    return scalar_detail::select_scalar(
        K,
        lps::zero_of<S>(),
        scalar_detail::pred_or(is_fixed != 0, ext_kinetic != 0));
}

/// `G = is_fixed ? 0 : inv_beta * (M * dq)`
template <typename S, typename I0>
[[nodiscard]] inline lps::LVec<S, 12> gradient(const lps::LVec<S, 12>&        q,
                                              const lps::LVec<S, 12>&        q_tilde,
                                              const ABDJacobiDyadicMassT<S>& m,
                                              const S&                       inv_beta,
                                              const I0&                      is_fixed) noexcept
{
    const auto dq = q - q_tilde;
    const auto G  = inv_beta * abd_dyadic_mass_mul(m, dq);
    return scalar_detail::select_container(
        G,
        lps::make_vec<S, 12>([](int) noexcept { return lps::zero_of<S>(); }),
        is_fixed != 0);
}

/// `H = inv_beta * M.to_mat()` (unconditional -- see the file comment)
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> hessian(const ABDJacobiDyadicMassT<S>& m,
                                                  const S&                       inv_beta) noexcept
{
    return inv_beta * abd_dyadic_mass_to_mat(m);
}
}  // namespace bdf2_kinetic_detail

class AffineBodyBDF2Kinetic final : public AffineBodyKinetic
{
  public:
    static constexpr Float beta     = 4.0 / 9.0;  // BDF2 beta coefficient
    static constexpr Float inv_beta = Float{1} / beta;

    using AffineBodyKinetic::AffineBodyKinetic;

    void do_build(BuildInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
