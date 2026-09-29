#pragma once
/********************************************************************
 * @file strain_plastic_discrete_shell_bending.h
 * @brief T5.2 (batch b) — StrainPlasticDiscreteShellBending (uid 31) and its
 *        per-element history TimeIntegrator.
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/
 * strain_plastic_discrete_shell_bending.cu — one translation unit containing *both*
 * systems (`StrainPlasticDiscreteShellBending` : FiniteElementExtraConstitution and
 * `StrainPlasticDiscreteShellBendingTimeIntegrator` : TimeIntegrator), each with its
 * own `REGISTER_SIM_SYSTEM`.  The port keeps both classes (and their registrations) in
 * one header/.cpp pair, exactly like the reference, so the T5.3 dynamics sees the
 * same system set.
 *
 * The per-element history lives in the constitution's device buffers
 * (`theta_bars`, `yield_thresholds`); the TimeIntegrator's `do_update_state` kernel is
 * the only writer, and it calls the *same* generic routine
 * (`sym::strain_plastic_discrete_shell_bending::update_plastic_state`) the unit test
 * drives directly (cookbook §7.4).  The reference's update order — dihedral angle ->
 * try/fail guard -> plastic excess -> `theta_bar` -> `yield_threshold` -> write both
 * only when the yield condition was met — is preserved verbatim; see the function
 * header for the select-based formulation.
 *
 * DEVIATION (reported, as in batch a — cookbook §7.17 rule 11): the class
 * *declarations* are in this header (kernels + out-of-line members + registration in
 * the .cpp) so the unit test can reach the reference's public surface (the buffers and
 * every hook) and drive the real constitutions end to end.
 ********************************************************************/
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/finite_element_method.h>
#include <time_integrator/time_integrator.h>
#include <finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class StrainPlasticDiscreteShellBending final : public FiniteElementExtraConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 31;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;
    using Base                             = FiniteElementExtraConstitution;

    using Base::Base;

    virtual U64 get_uid() const noexcept override;

    class InitInfo
    {
      public:
        bool        valid_bending() const { return oppo_verts.size() == 2; }
        IndexT      edge_index = -1;
        set<IndexT> oppo_verts;
    };

    vector<Vector4i> h_stencils;
    vector<Float>    h_bending_stiffness;
    vector<Float>    h_rest_lengths;
    vector<Float>    h_h_bars;
    vector<Float>    h_theta_bars;
    vector<Float>    h_yield_thresholds;
    vector<Float>    h_hardening_moduli;
    vector<Float>    h_V_bars;

    lud::LDeviceBuffer<LVector4i> stencils;
    lud::LDeviceBuffer<Float>     bending_stiffnesses;
    lud::LDeviceBuffer<Float>     rest_lengths;
    lud::LDeviceBuffer<Float>     h_bars;
    lud::LDeviceBuffer<Float>     theta_bars;
    lud::LDeviceBuffer<Float>     yield_thresholds;
    lud::LDeviceBuffer<Float>     hardening_moduli;
    lud::LDeviceBuffer<Float>     V_bars;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(FilteredInfo& info) override;

    virtual void do_report_extent(ReportExtentInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};

class StrainPlasticDiscreteShellBendingTimeIntegrator final : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<StrainPlasticDiscreteShellBending> pdsb;
    SimSystemSlot<FiniteElementMethod>               fem;
    BufferDump                                       dump_theta_bars;
    BufferDump                                       dump_yield_thresholds;

    void do_build(TimeIntegrator::BuildInfo& info) override;

    void do_init(TimeIntegrator::InitInfo& info) override;

    void do_predict_dof(TimeIntegrator::PredictDofInfo& info) override;

    void do_update_state(TimeIntegrator::UpdateVelocityInfo& info) override;

    bool do_dump(DumpInfo& info) override;

    bool do_try_recover(RecoverInfo& info) override;

    void do_apply_recover(RecoverInfo& info) override;

    void do_clear_recover(RecoverInfo& info) override;
};
}  // namespace uipc::backend::luisa
