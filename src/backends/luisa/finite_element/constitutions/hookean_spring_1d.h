#pragma once
/********************************************************************
 * @file hookean_spring_1d.h
 * @brief T5.2 (batch c) — HookeanSpring1D, the 1d Hookean-spring constitution
 *        (uid 12).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/hookean_spring_1d.cu
 * (class + kernels + `REGISTER_SIM_SYSTEM` in one translation unit; base
 * `Codim1DConstitution`).
 *
 * DEVIATIONS (reported in the T5.2 batch-c report):
 *  1. the *class declaration* lives here and the kernels + out-of-line member
 *     definitions + the registration stay in `hookean_spring_1d.cpp` — the same split
 *     T5.2 batch a introduced (cookbook §7.17 rule 11), so the unit test can reach the
 *     reference's *public* surface (its `kappas` buffer and the five hooks) and drive
 *     the real constitution end to end;
 *  2. the reference includes
 *     `<finite_element/codim_1d_constitution_diff_parm_reporter.h>`, which belongs to
 *     the diff-sim surface (T5.5) and is not ported — the include is dropped, nothing
 *     in the reference's class uses it.
 ********************************************************************/
#include <finite_element/codim_1d_constitution.h>
#include <finite_element/constitutions/hookean_spring_1d_function.h>

namespace uipc::backend::luisa
{
class HookeanSpring1D final : public Codim1DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 12ull;
    static constexpr SizeT StencilSize     = 2;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using Codim1DConstitution::Codim1DConstitution;

    vector<Float>             h_kappas;
    lud::LDeviceBuffer<Float> kappas;

    virtual U64 get_uid() const noexcept override;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_report_extent(ReportExtentInfo& info) override;

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
