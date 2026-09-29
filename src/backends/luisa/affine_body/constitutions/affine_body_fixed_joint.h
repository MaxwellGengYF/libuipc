#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_fixed_joint.h
 * @brief T6.3b1 — the ABD fixed joint (`ConstitutionUID = 25`).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_fixed_joint.cu
 * (class + kernels + `REGISTER_SIM_SYSTEM` in one translation unit).
 *
 * DEVIATION (the T5.2 §7.17 rule 11 / T6.3a precedent, reported): the *class
 * declaration* lives in this header and the kernels + the out-of-line member
 * definitions + the registration stay in `affine_body_fixed_joint.cpp`.  It is a pure
 * file split (same class name, same members, same access, same overrides, same
 * registration macro) whose only motivation is that a unit test can reach the
 * reference's *public* surface: the reference declares the six hooks and the
 * parameter buffers in the class's `public:` section, and the test drives the real
 * kernels on the real device buffers through them.
 *
 * WHAT THIS SYSTEM IS.  A plain `InterAffineBodyConstitution` (no dof reporter, no
 * time integrator, no driving, no external force, no joint limit — those are T6.3b2's
 * and T6.4's): it owns one *per-joint* record (the two body ids, the four body-local
 * `Vector6` rest-frame rows and the frontend's `strength_ratio`), reports one energy /
 * two gradient doublets / three 12x12 Hessian triplets per joint, and computes them
 * with the *generated* joint laws (see `affine_body_fixed_joint_function.h`).
 *
 * The per-joint rest frames are the frontend's own convention
 * (`uipc::constitution::AffineBodyFixedJoint::create_geometry`): `rest_cs` holds the
 * two attachment points in each body's *local* frame (`[ci_bar | cj_bar]`), and
 * `rest_ts` / `rest_ns` / `rest_bs` hold the body-local directions that map to the
 * world x / y / z axes at rest (`[t_bar | t_bar_j]`, ...) — i.e. the two frames the
 * joint's rotation term compares.  If the geometry carries the `l_position` /
 * `r_position` vertex attributes the frontend wrote them; otherwise the two local
 * points are recovered from the *scene* vertex (`LT.inverse() * Ps[i]`), exactly as
 * the reference does.
 ********************************************************************/
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constitutions/affine_body_fixed_joint_function.h>

namespace uipc::backend::luisa
{
class AffineBodyFixedJoint final : public InterAffineBodyConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 25ull;
    /// The reference's `2 * (2 + 1) / 2`: the 12x12-triplet run length of a two-body
    /// `TMA.half_block<2>` (the upper triangle of a 2x2 block of 12x12 blocks).
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    lud::LDeviceBuffer<LVector2i> body_ids;
    lud::LDeviceBuffer<LVector6>  rest_cs;  // midpoint in body-local frame [ci_bar | cj_bar]
    lud::LDeviceBuffer<LVector6>  rest_ts;  // t_bar in body space
    lud::LDeviceBuffer<LVector6>  rest_ns;  // n_bar in body space
    lud::LDeviceBuffer<LVector6>  rest_bs;  // b_bar in body space
    lud::LDeviceBuffer<Float>     strength_ratios;

    vector<Vector2i> h_body_ids;
    vector<Vector6>  h_rest_cs;
    vector<Vector6>  h_rest_ts;
    vector<Vector6>  h_rest_ns;
    vector<Vector6>  h_rest_bs;
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
