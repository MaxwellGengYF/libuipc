#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_revolute_joint_limit.h
 * @brief T6.3b3 — the revolute joint-limit constitution
 *        (`ConstitutionUID = 670`, `JointUID = AffineBodyRevoluteJoint::ConstitutionUID`).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu
 * (class `AffineBodyRevoluteJointLimit`).
 *
 * WHAT THIS SYSTEM IS.  A second `InterAffineBodyConstitution` that *annotates* the
 * revolute joint's own geometry: `uipc::constitution::AffineBodyRevoluteJointLimit::
 * apply_to` registers 670 as an `extra_constitution_uid` on a geometry whose *base*
 * uid stays 18, and writes the three per-edge limit attributes
 * (`limit/lower`, `limit/upper`, `limit/strength`).  The limit's `do_init` therefore
 * asserts that the underlying geometry's `constitution_uid` is still `JointUID` (18) —
 * the cross-reference that *finds* the joint geometry — and then reuses that geometry's
 * joints by index: `body_ids`/`l_basis`/`r_basis`/`init_angles`/`current_angles` come
 * from `AffineBodyRevoluteJoint`, and the same index-alignment assert as the driving
 * constraint's applies.
 *
 * It is filtered by the *extra* uid, not by `constraint_uids`: the manager's
 * `constitution_inter_geo_infos` (base **plus** extra hits) is what `FilteredInfo`
 * slices for a constitution, so a limit instance only ever sees geometries that
 * carry 670.  (The frontend's `apply_to` calls `Base::apply_to(sc)` — the
 * `InterAffineBodyExtraConstitution` frontend base, which appends 670 to
 * `extra_constitution_uids` — *after* the `constitution_uid == 18` check.)
 *
 * THE LAW, AND WHAT IS NEW VS THE PRISMATIC LIMIT (uid 669).  The energy is the cubic
 * penalty of `joint_limit_penalty.h` on the *unwrapped relative angle*
 *
 *     x = (current_angle - init_angle) + DeltaTheta(basis_k, qk, qk_t, basis_l, ql, ql_t)
 *     E = eval_penalty_energy(x, limit/lower - init_angle, limit/upper - init_angle, strength)
 *
 * i.e. the same *shape* as the prismatic limit's distance chain, but the measured
 * quantity is the angle *difference* between the current and the previous state
 * (`DeltaTheta`, the limit body's own generated function) added to the joint's
 * *stored* multi-turn angle — which is why this limit needs the manager's `q_prevs`
 * window (`info.q_prevs()`) and the joint's `current_angles` state, while the prismatic
 * one needed neither.  The gradient/Hessian ride on the generated chains
 * `dDeltaTheta_dQ` (24) and `ddDeltaTheta_ddF` (12x12, **not** diagonal, so the
 * reference's `make_spd` clamp on `dE_dx * ddDeltaTheta_ddF` is a real projection and
 * the sweep count IS observable here — the third clamp site of this batch):
 *
 *     G = dE_dx * dx_dq
 *     H = d2E_dx2 (dx_dq dx_dq^T) [+ J^T make_spd(dE_dx * ddDeltaTheta_ddF(F, F_prev)) J
 *                                  if dE_dx != 0]
 *
 * `dE_dx != 0` is a *device* predicate, so the kernel keeps the reference's `$if`
 * (a statement guard, exactly like the prismatic limit's); the *host-callable* helper
 * in the function header computes the second term unconditionally and is bitwise the
 * same function (measured in the test).
 *
 * DEVIATION (the T6.3b1/T6.3b2 file split): class here, kernels + members +
 * `REGISTER_SIM_SYSTEM` in the `.cpp`.
 *********************************************************************/
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constitutions/affine_body_revolute_joint.h>
#include <affine_body/constitutions/affine_body_revolute_joint_function.h>

namespace uipc::backend::luisa
{
class AffineBodyRevoluteJointLimit final : public InterAffineBodyConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 670;
    static constexpr U64   JointUID        = AffineBodyRevoluteJoint::ConstitutionUID;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    SimSystemSlot<AffineBodyRevoluteJoint> revolute_joint;

    vector<Float> h_lowers;
    vector<Float> h_uppers;
    vector<Float> h_strengths;

    lud::LDeviceBuffer<Float> lowers;
    lud::LDeviceBuffer<Float> uppers;
    lud::LDeviceBuffer<Float> strengths;

    void do_build(BuildInfo& info) override;
    void do_init(FilteredInfo& info) override;

    void do_report_energy_extent(EnergyExtentInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;

    U64 get_uid() const noexcept override;
};
}  // namespace uipc::backend::luisa
