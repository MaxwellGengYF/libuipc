#pragma once
/********************************************************************
 * @file arap_3d.h
 * @brief T5.2 (batch a) — ARAP3D, the 3D volumetric ARAP constitution.
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/arap_3d.cu (class +
 * kernels + `REGISTER_SIM_SYSTEM` in one translation unit).
 *
 * DEVIATION (reported in the T5.2 report): the *class declaration* lives in this
 * header and the kernels + the out-of-line member definitions + the registration
 * stay in `arap_3d.cpp`.  The split is a pure file split (same class name, same
 * members, same access, same overrides, same registration macro) and exists so
 * that the unit test can reach the reference's *public* surface -- the reference
 * declares `do_build`/`do_report_extent`/`do_init`/`do_compute_energy`/
 * `do_compute_gradient_hessian` in the class's `public:` section together with the
 * parameter buffers (`kappas`, `h_kappas`), so a test can drive the real
 * constitution end to end (build/init/report_extent + the energy/gradient/Hessian
 * kernels on the real method buffers).  With the class hidden in a .cpp the test
 * could only reach the base class, and the batch's validation requirement 6
 * ("a real scene ... so the constitution's ... assemble/energy path is exercised
 * end to end") would be unverifiable.
 ********************************************************************/
#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/arap_function.h>

namespace uipc::backend::luisa
{
class ARAP3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 9;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_kappas;

    lud::LDeviceBuffer<Float> kappas;

    virtual U64 get_uid() const noexcept override;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_report_extent(ReportExtentInfo& info) override;

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
