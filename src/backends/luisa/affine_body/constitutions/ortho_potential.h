#pragma once
/********************************************************************
 * @file affine_body/constitutions/ortho_potential.h
 * @brief T6.3a — the ABD orthogonal-potential constitution (uid 1).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/ortho_potential.cu (class
 * + kernels + `REGISTER_SIM_SYSTEM` in one translation unit).
 *
 * DEVIATION (the T5.2 §7.17 rule 11 precedent, reported in the T6.3a report): the
 * class declaration lives in this header and the kernels + the out-of-line member
 * definitions + the registration stay in `ortho_potential.cpp`, so the unit test
 * can reach the reference's *public* surface (the reference declares the four hooks
 * and the parameter buffers in the class's `public:` section).
 *
 * This is the uid **1** constitution — the frontend
 * `uipc::constitution::AffineBodyConstitution`'s default (`{"name":
 * "OrthoPotential"}`) — i.e. the very uid T6.1a's test-local mock constitution
 * claimed while no backend constitution existed.  The collision is resolved in the
 * T6.3a batch (see the cookbook §7.24 rule on uid ownership): the mock moved to the
 * grep-proven-unused uid **8**, so uid 1 belongs to this class alone and both the
 * filtered and the full suite can assert the group's real content.
 *
 * The energy law is the generated one (see `ortho_potential_function.h`):
 * `E = kappa ||AᵀA - I||_F^2` with `A` the 3x3 linear part of the 12-dof state, and
 * the kernel energy is `E * volume * dt^2`.  Unlike the ABD ARAP (which ships the
 * *unclamped* exact Hessian) this one **clamps**: the reference calls
 * `make_spd(H9x9)` before scaling, because `ddEddq` is indefinite for `AᵀA < I`
 * (e.g. the collapsed state `A = 0` gives `-4 kappa I`).
 ********************************************************************/
#include <affine_body/affine_body_constitution.h>
#include <affine_body/constitutions/ortho_potential_function.h>

namespace uipc::backend::luisa
{
class OrthoPotential final : public AffineBodyConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 1ull;

    using AffineBodyConstitution::AffineBodyConstitution;

    vector<Float> h_kappas;

    lud::LDeviceBuffer<Float> kappas;

    virtual U64 get_uid() const noexcept override;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(AffineBodyDynamics::FilteredInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
