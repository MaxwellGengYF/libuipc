#pragma once
/********************************************************************
 * @file neo_hookean_shell_2d.h
 * @brief T5.2 (batch b) — NeoHookeanShell2D, the 2D Neo-Hookean shell (membrane)
 *        constitution (uid 11).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/neo_hookean_shell_2d.cu
 * (class + kernels + REGISTER_SIM_SYSTEM in one translation unit;
 * `Codim2DConstitution` base).
 *
 * DEVIATION (reported, the same one T5.2 batch a introduced — cookbook §7.17 rule 11):
 * the *class declaration* lives here, the kernels + out-of-line members + the
 * registration stay in `neo_hookean_shell_2d.cpp`, so the unit test can reach the
 * reference's *public* surface (the parameter buffers and the five hooks) and drive
 * the real constitution end to end.
 ********************************************************************/
#include <finite_element/codim_2d_constitution.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>

namespace uipc::backend::luisa
{
class NeoHookeanShell2D final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 11;
    static constexpr SizeT StencilSize     = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using Codim2DConstitution::Codim2DConstitution;

    vector<Float> h_lambdas;
    vector<Float> h_mus;

    lud::LDeviceBuffer<Float>     lambdas;
    lud::LDeviceBuffer<Float>     mus;
    lud::LDeviceBuffer<LMatrix2x2> inv_B_matrices;

    SimSystemSlot<FiniteElementMethod> fem;

    virtual U64 get_uid() const noexcept override;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override;

    virtual void do_report_extent(ReportExtentInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
