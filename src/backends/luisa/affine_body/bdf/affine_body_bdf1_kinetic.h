#pragma once
/********************************************************************
 * @file affine_body/bdf/affine_body_bdf1_kinetic.h
 * @brief T6.2 — the BDF1 ABD kinetic system (port of
 * src/backends/cuda/affine_body/bdf/affine_body_bdf1_kinetic.cu).
 *
 * DEVIATION (reported): the reference keeps the class *and* its kernels inside the
 * `.cu`, so nothing about it is reachable from a test.  This port follows the T5.2
 * rule (cookbook §7.17 rule 11) exactly like the FEM twin
 * `finite_element_bdf1_kinetic.h` -- the class is declared here, the kernels + their
 * cached shaders + `REGISTER_SIM_SYSTEM` stay in the `.cpp` -- so that the per-body
 * bodies can be the *one* generic implementation (the T6.2 brief's §1) that the
 * shipped kernel and the unit test both instantiate (cookbook §7.4).
 *
 * THE ABD MASS OPERATOR (why this is not the FEM twin's scalar mass): the ABD kinetic
 * energy is `½ dq · (M dq)` where `M` is the per-body **dyadic mass**
 * `Jᵀ Mᵢ J` (`ABDJacobiDyadicMass`, an aggregate of `m`, `m x̄`, `m x̄⊗x̄`), and
 * `M dq` is the factored `abd_dyadic_mass_mul` (cookbook §7.21); `M.to_mat()` is the
 * dense 12x12 `abd_dyadic_mass_to_mat`, which is exactly what the gradient/Hessian
 * stage writes into the *dense per-body* Hessian buffer (T6.1b's corrected
 * `ComputeGradientHessianInfo` shape, cookbook §7.22 rule 6).  Where the FEM kernel
 * computes `m * dx` and `m * I`, this one computes `M * dq` and `M.to_mat()`.
 *
 * The BDF1 kinetic laws, verbatim from the reference's kernels (dq = q - q_tilde):
 *   K = (is_fixed || ext_kinetic) ? 0 : 0.5 * dq.dot(M * dq)
 *   G = is_fixed ? 0 : M * dq                 (the per-body dense 12-vector)
 *   H = M.to_mat()                            (the per-body dense 12x12)
 * NOTE the Hessian has **no** `is_fixed` guard in the reference (only K and G are
 * branched), the same asymmetry the FEM batch measured: a fixed body still receives
 * `M` on its diagonal block (the mass regularisation of the implicit system).  The
 * unit test asserts that behaviour (`abd_dynamics_kinetic`) because it is easy to
 * "fix" by accident.
 *********************************************************************/
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <time_integrator/bdf1_flag.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer; `lla::detail::cached_kernel` is the port's
// `compile1`-with-a-function-local-static idiom for the kernels below.
namespace lla = luisa_device::linalg;

/// The reference's two `AffineBodyBDF1Kinetic_*` kernel bodies, per body, generic
/// over the scalar type and over the two index types.
namespace bdf1_kinetic_detail
{
/// `K = (is_fixed || ext_kinetic) ? 0 : 0.5 * dq.dot(M * dq)`
template <typename S, typename I0, typename I1>
[[nodiscard]] inline S energy(const lps::LVec<S, 12>&        q,
                              const lps::LVec<S, 12>&        q_tilde,
                              const ABDJacobiDyadicMassT<S>& m,
                              const I0&                      is_fixed,
                              const I1&                      ext_kinetic) noexcept
{
    const auto dq = q - q_tilde;
    const auto K  = lps::detail::mul(0.5, lps::dot(dq, abd_dyadic_mass_mul(m, dq)));
    return scalar_detail::select_scalar(
        K,
        lps::zero_of<S>(),
        scalar_detail::pred_or(is_fixed != 0, ext_kinetic != 0));
}

/// `G = is_fixed ? 0 : M * (q - q_tilde)`
template <typename S, typename I0>
[[nodiscard]] inline lps::LVec<S, 12> gradient(const lps::LVec<S, 12>&        q,
                                              const lps::LVec<S, 12>&        q_tilde,
                                              const ABDJacobiDyadicMassT<S>& m,
                                              const I0&                      is_fixed) noexcept
{
    const auto dq = q - q_tilde;
    const auto G  = abd_dyadic_mass_mul(m, dq);
    return scalar_detail::select_container(
        G,
        lps::make_vec<S, 12>([](int) noexcept { return lps::zero_of<S>(); }),
        is_fixed != 0);
}

/// `H = M.to_mat()` (unconditional -- see the file comment)
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> hessian(const ABDJacobiDyadicMassT<S>& m) noexcept
{
    return abd_dyadic_mass_to_mat(m);
}
}  // namespace bdf1_kinetic_detail

class AffineBodyBDF1Kinetic final : public AffineBodyKinetic
{
  public:
    using AffineBodyKinetic::AffineBodyKinetic;

    void do_build(BuildInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
