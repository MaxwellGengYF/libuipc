#pragma once
/********************************************************************
 * @file stress_plastic_discrete_shell_bending.h
 * @brief T5.2 (batch b) — StressPlasticDiscreteShellBending (uid 32) and its
 *        per-element history TimeIntegrator.
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/
 * stress_plastic_discrete_shell_bending.cu — one translation unit containing both
 * systems (the constitution and its history integrator), each with its own
 * `REGISTER_SIM_SYSTEM`.  The port keeps that shape (see the strain-plastic sibling
 * for the shared argument: the reference declares the integrator here, and the unit
 * test drives the *same* generic update routine it calls).
 *
 * The strain-plastic vs stress-plastic difference is the state variable and the
 * yield law (`bending_yield_threshold` / hardening on the threshold vs
 * `bending_yield_stress` / hardening on the stress, with the trial state
 * `tau_trial = 2 kappa L0 / h_bar * delta` and `theta_y = yield_stress /
 * elastic_slope`); both are transcribed verbatim in the function header.
 ********************************************************************/
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/finite_element_method.h>
#include <time_integrator/time_integrator.h>
#include <finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class StressPlasticDiscreteShellBending final : public FiniteElementExtraConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 32;
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
    vector<Float>    h_yield_stresses;
    vector<Float>    h_hardening_moduli;
    vector<Float>    h_V_bars;

    lud::LDeviceBuffer<LVector4i> stencils;
    lud::LDeviceBuffer<Float>     bending_stiffnesses;
    lud::LDeviceBuffer<Float>     rest_lengths;
    lud::LDeviceBuffer<Float>     h_bars;
    lud::LDeviceBuffer<Float>     theta_bars;
    lud::LDeviceBuffer<Float>     yield_stresses;
    lud::LDeviceBuffer<Float>     hardening_moduli;
    lud::LDeviceBuffer<Float>     V_bars;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(FilteredInfo& info) override;

    virtual void do_report_extent(ReportExtentInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};

class StressPlasticDiscreteShellBendingTimeIntegrator final : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<StressPlasticDiscreteShellBending> spdsb;
    SimSystemSlot<FiniteElementMethod>               fem;
    BufferDump                                       dump_theta_bars;
    BufferDump                                       dump_yield_stresses;

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
