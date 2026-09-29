#pragma once
/********************************************************************
 * @file kirchhoff_rod_bending.h
 * @brief T5.2 (batch c) — KirchhoffRodBending, the Kirchhoff-rod bending
 *        constitution (uid 15).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/kirchhoff_rod_bending.cu
 * (class + kernels + `REGISTER_SIM_SYSTEM` in one translation unit; base
 * `FiniteElementExtraConstitution`).
 *
 * DEVIATIONS (reported in the T5.2 batch-c report):
 *  1. the *class declaration* lives here and the kernels + out-of-line member
 *     definitions + the registration stay in `kirchhoff_rod_bending.cpp` — the same
 *     split T5.2 batch a introduced (cookbook §7.17 rule 11), so the unit test can
 *     reach the reference's *public* surface (its buffers and the five hooks) and
 *     drive the real constitution end to end;
 *  2. the reference spells the uid constant `KirchhoffRodBendingUID`; the port uses
 *     the batch-a/b spelling `ConstitutionUID` (same value, 15) for consistency
 *     across the three batches' files;
 *  3. the reference's `hinges` device buffer is `DeviceBuffer<Vector3i>` (Eigen);
 *     the port stores `LVector3i` (the registered LC element type, byte-identical
 *     to `Vector3i` — see type_define.h's static_assert).
 ********************************************************************/
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/constitutions/kirchhoff_rod_bending_function.h>

namespace uipc::backend::luisa
{
class KirchhoffRodBending final : public FiniteElementExtraConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 15;
    static constexpr SizeT StencilSize     = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;
    using Base                             = FiniteElementExtraConstitution;

    using Base::Base;

    virtual U64 get_uid() const noexcept override;

    /// X0, X1, X2; X1 is the center vertex of the hinge
    vector<Vector3i> h_hinges;
    vector<Float>    h_bending_stiffness;

    lud::LDeviceBuffer<LVector3i> hinges;
    lud::LDeviceBuffer<Float>     bending_stiffnesses;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(FilteredInfo& info) override;

    virtual void do_report_extent(ReportExtentInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
