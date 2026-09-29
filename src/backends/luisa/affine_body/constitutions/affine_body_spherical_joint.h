#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_spherical_joint.h
 * @brief T6.3b1 — the ABD spherical joint (`ConstitutionUID = 26`).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_spherical_joint.cu
 * (class + kernels + `REGISTER_SIM_SYSTEM` in one translation unit).
 *
 * DEVIATION (the T6.3a file split, reported in the fixed joint's header): the class
 * declaration lives here and the kernels + the out-of-line members + the
 * registration stay in `affine_body_spherical_joint.cpp`, so a unit test can drive
 * the reference's *public* hooks on the real device buffers.
 *
 * A plain `InterAffineBodyConstitution` (no dof reporter / time integrator / driving
 * / external force / joint limit): one *per-joint* record (the two body ids, the
 * body-local anchor row `rest_cs = [ci_bar | cj_bar]` and the frontend's
 * `strength_ratio`), one energy / two gradient doublets / three 12x12 Hessian
 * triplets per joint, and the *generated* ball-joint laws (see
 * `affine_body_spherical_joint_function.h`).  The frontend's
 * `uipc::constitution::AffineBodySphericalJoint::create_geometry` owns the
 * `l_position` / `r_position` convention (see the fixed joint's header).
 ********************************************************************/
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constitutions/affine_body_spherical_joint_function.h>

namespace uipc::backend::luisa
{
class AffineBodySphericalJoint final : public InterAffineBodyConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 26ull;
    /// The reference's `2 * (2 + 1) / 2` (see the fixed joint's header).
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    lud::LDeviceBuffer<LVector2i> body_ids;
    lud::LDeviceBuffer<LVector6>  rest_cs;  // anchor in each body's local frame [ci_bar | cj_bar]
    lud::LDeviceBuffer<Float>     strength_ratios;

    vector<Vector2i> h_body_ids;
    vector<Vector6>  h_rest_cs;
    vector<Float>    h_strength_ratios;

    U64 get_uid() const noexcept override;

    void do_build(BuildInfo& info) override;
    void do_init(FilteredInfo& info) override;

    void do_report_energy_extent(EnergyExtentInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
