#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_prismatic_joint_limit.h
 * @brief T6.3b2 — the prismatic joint-limit constitution
 *        (`ConstitutionUID = 669`, `JointUID = AffineBodyPrismaticJoint::ConstitutionUID`).
 *
 * REFERENCE: src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu
 * (class `AffineBodyPrismaticJointLimit`).
 *
 * WHAT THIS SYSTEM IS.  A second `InterAffineBodyConstitution` that *annotates* the
 * prismatic joint's own geometry: `uipc::constitution::AffineBodyPrismaticJointLimit::
 * apply_to` registers 669 as an `extra_constitution_uid` on a geometry whose *base*
 * uid stays 20, and writes the three per-edge limit attributes
 * (`limit/lower`, `limit/upper`, `limit/strength`).  The limit's `do_init`
 * therefore asserts that the underlying geometry's `constitution_uid` is still
 * `JointUID` (20) — the cross-reference that *finds* the joint geometry — and then
 * reuses that geometry's joints by index: `body_ids`/`rest_cs`/`rest_ts`/
 * `init_distances` come from `AffineBodyPrismaticJoint`, and the same
 * index-alignment assert as the driving constraint's applies.
 *
 * It is filtered by the *extra* uid, not by `constraint_uids`: the manager's
 * `constitution_inter_geo_infos` (base **plus** extra hits) is what `FilteredInfo`
 * slices for a constitution, so a limit instance only ever sees geometries that
 * carry 669.  (The frontend's `apply_to` calls `Base::apply_to(sc)` — the
 * `InterAffineBodyConstitution` frontend base, which appends 669 to
 * `extra_constitution_uids` — *after* the `constitution_uid == 20` check.)
 *
 * The energy is the cubic penalty of `joint_limit_penalty.h` on the *absolute
 * distance* between the two attachment points, with both bounds shifted by the
 * joint's `init_distance`: `x = compute_absolute_distance`, `lower = limit/lower -
 * init`, `upper = limit/upper - init`.  The gradient/Hessian ride on the driving
 * joint's own `Distance` chain (`compute_absolute_distance_derivative` /
 * `..._hessian`), which is why the limit includes the same function header.
 *
 * `ref_qs` is the reference's per-entry *reference state* (`[q_l ; q_r]`, a
 * `DeviceBuffer<Vector24>`): the port needs the `LVector24` device element type for
 * it (`affine_body/type_define.h`, the T6.3b1 `LVector6` precedent).  NOTE: the
 * reference never reads it back — the three limit kernels do not take it — so it is
 * a *bookkeeping* buffer whose only use is its size (`ref_qs.size()` drives all three
 * extents).  The port keeps it, with the reference's data, so the extents and the
 * recorded state are the reference's.
 *
 * DEVIATION (the T6.3b1 file split): class here, kernels + members +
 * `REGISTER_SIM_SYSTEM` in the `.cpp`.
 *********************************************************************/
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constitutions/affine_body_prismatic_joint.h>
#include <affine_body/constitutions/affine_body_prismatic_joint_function.h>

namespace uipc::backend::luisa
{
class AffineBodyPrismaticJointLimit final : public InterAffineBodyConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 669;
    static constexpr U64   JointUID        = AffineBodyPrismaticJoint::ConstitutionUID;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    using Vector24    = uipc::Matrix<Float, 24, 1>;
    using Matrix24x24 = uipc::Matrix<Float, 24, 24>;

    SimSystemSlot<AffineBodyPrismaticJoint> prismatic_joint;

    vector<Vector24> h_ref_qs;
    vector<Float>    h_lowers;
    vector<Float>    h_uppers;
    vector<Float>    h_strengths;

    lud::LDeviceBuffer<LVector24> ref_qs;
    lud::LDeviceBuffer<Float>     lowers;
    lud::LDeviceBuffer<Float>     uppers;
    lud::LDeviceBuffer<Float>     strengths;

    void do_build(BuildInfo& info) override;
    void do_init(FilteredInfo& info) override;

    void do_report_energy_extent(EnergyExtentInfo& info) override;
    void do_compute_energy(ComputeEnergyInfo& info) override;

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override;
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;

    U64 get_uid() const noexcept override;
};
}  // namespace uipc::backend::luisa
