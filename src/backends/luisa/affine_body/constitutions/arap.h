#pragma once
/********************************************************************
 * @file affine_body/constitutions/arap.h
 * @brief T6.3a — the ABD ARAP constitution (uid 2).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/arap.cu (class + kernels
 * + `REGISTER_SIM_SYSTEM` in one translation unit).
 *
 * DEVIATION (the T5.2 §7.17 rule 11 precedent, reported in the T6.3a report): the
 * *class declaration* lives in this header and the kernels + the out-of-line
 * member definitions + the registration stay in `arap.cpp`.  The split is a pure
 * file split (same class name, same members, same access, same overrides, same
 * registration macro) and exists so that the unit test can reach the reference's
 * *public* surface: the reference declares the four hooks (`do_build` / `do_init` /
 * `do_compute_energy` / `do_compute_gradient_hessian`) in the class's `public:`
 * section together with the parameter buffers (`kappas`, `h_kappas`), so a test can
 * drive the real constitution's kernels on the real device buffers.
 *
 * SCOPE NOTE (T6.3a): `AffineBodyConstitution` has **no** `report_extent`/
 * `assemble` hooks (unlike the FEM family, §7.17): an ABD constitution only
 * *computes* energy/gradient/Hessian into the dense per-body windows the
 * line-search/linear-subsystem infos hand over (§7.22 rule 6); the extents belong
 * to the subsystem.  The `AffineBodyConstitution::Impl` the base carries is
 * therefore enough to reach the dynamics' `q`/`volume` windows.
 ********************************************************************/
#include <affine_body/affine_body_constitution.h>
#include <affine_body/constitutions/arap_function.h>

namespace uipc::backend::luisa
{
class ARAP final : public AffineBodyConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 2ull;

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
