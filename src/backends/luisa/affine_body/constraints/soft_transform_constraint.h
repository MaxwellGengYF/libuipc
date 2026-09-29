#pragma once
/********************************************************************
 * @file affine_body/constraints/soft_transform_constraint.h
 * @brief T6.4b — the ABD **soft transform** constraint (uid 16), the ABD sibling of
 * the ported FEM `SoftPositionConstraint` (uid 14).
 *
 * REFERENCE: src/backends/cuda/affine_body/constraints/soft_transform_constraint.cu.
 *
 * WHAT THIS SYSTEM IS.  An `AffineBodyConstraint` (so `AffineBodyAnimator` drives it)
 * that, for every *instance* marked with the frontend's
 * `SoftTransformConstraint::apply_to` protocol
 *   `instances/is_constrained`  IndexT    (0 -> the instance contributes nothing)
 *   `instances/aim_transform`   Matrix4x4 (the target transform, shared with
 *                                          `instances/transforms` by the frontend)
 *   `instances/strength_ratio`  Vector2   (s_t, s_r — translation/rotation weights)
 * penalises the *12-dof* distance between the body's current state `q` and a
 * substep-lerped aim state:
 *
 *     q_aim = lerp(q_prev, transform_to_q(aim_transform), substep_ratio)
 *     dq    = q - q_aim
 *     M_c   = compute_constraint_mass(body_mass, s_t, s_r)
 *     E     = is_fixed ? 0 : 0.5 dq^T M_c dq
 *     G     = is_fixed ? 0 : M_c dq
 *     H     = is_fixed ? 0 : M_c
 *
 * THE CONSTRAINT MASS — the one non-obvious law (verbatim from the reference's
 * `compute_constraint_mass`).  `ABDJacobiDyadicMass` is the *ABD mass of a point mass
 * concentrated at the body's center of mass* (`M_cm = m J(c)^T J(c)`, built from the
 * dyadic triple `(m, m*c, m c c^T)`); the body's full `M` splits by the parallel-axis
 * theorem into `M_cm` + `M_rot`, so the strength pair re-weights the two *physical*
 * parts and not the matrix entries:
 *
 *     M_constraint = s_t M_cm + s_r (M - M_cm) = s_r M + (s_t - s_r) M_cm
 *
 * `s_t == s_r == s` therefore reproduces `s * M` exactly, and `s_t = 1, s_r = 0`
 * freezes the translation only (the frontend's `RotatingMotor` uses `{0, s}` and
 * `LinearMotor` `{s, 0}`).
 *
 * HOW IT DIFFERS FROM THE FEM SIBLING (`soft_position_constraint`, uid 14).  Read that
 * file first; the *contract* is the same, three things differ:
 *  * the dof is a *12-vector* (translation + the 3x3 linear part), not a vertex
 *    position, and the per-entry mass is the body's `ABDJacobiDyadicMass` — so the
 *    Hessian is the 12x12 `M_constraint` (constant in `q`) rather than `s m I`;
 *  * the aim state is `transform_to_q(aim_transform)` (the frontend's `aim_transform`
 *    *shares* `instances/transforms`, so it is a Matrix4x4 attribute, not a Vector3);
 *  * the entries are *instances* of the geometry, not vertices, and the index used in
 *    the kernels is the **global body id** (`local_index + backend_abd_body_offset`),
 *    exactly the ABD convention the joints use.
 * The base class, the `is_fixed`-guarded gradient/Hessian, the "energy count ==
 * gradient count == hessian count == the number of constrained entries" extent
 * contract and the `strength_ratio >= 0` assert are all the FEM sibling's.
 *
 * DEVIATION (reported, the T6.3b1 file-split/T5.3 precedent): the reference declares
 * the class *and* its three kernels inside the `.cu`; the port keeps the class and the
 * `compute_constraint_mass` law (as a free template in
 * `soft_transform_constraint_detail`, so the unit test drives the *shipped* law) in
 * this header and the kernels + the member definitions in the `.cpp`.
 *********************************************************************/
#include <affine_body/affine_body_constraint.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace soft_transform_constraint_detail
{
/**
 * @brief The reference's `compute_constraint_mass(mass, translation_strength,
 *        rotation_strength)`.
 *
 * One implementation over the scalar type `S` (`double` on the host,
 * `lcrt::Var<double>` in the DSL) — the T6.1a/T5.1 convention.
 *
 * The reference's `UIPC_KERNEL_ASSERT(m > 0.0, ...)` has no DSL spelling (a device
 * assert), so the port routes it through the `LC_LOG` kernel-logging facade, which is
 * a no-op unless `UIPC_LUISA_KERNEL_PRINT` is defined — the `torque_to_F` precedent
 * (cookbook §7.21 rule 5, `affine_body/utils.h`).
 */
template <class S>
[[nodiscard]] inline lps::LMat<S, 12, 12> compute_constraint_mass(
    const ABDJacobiDyadicMassT<S>& mass,
    const S&                       translation_strength,
    const S&                       rotation_strength) noexcept
{
    const S s_t = translation_strength;
    const S s_r = rotation_strength;
    const S m   = mass.m_mass;

    LC_LOG("ABDJacobiDyadicMass has non-positive mass, cannot build constraint mass matrix.");

    lps::LMat<S, 12, 12> M = abd_dyadic_mass_to_mat<S>(mass);

    // Build M_cm = m*J(c)^T*J(c): the ABD mass matrix of a point mass m
    // concentrated at the center of mass c (where mc = m*c is the first moment).
    // M = M_cm + M_rot  (parallel-axis decomposition)
    // M_cm captures CM translation kinetics; M_rot captures rotation/deformation about CM.
    const lps::LVec<S, 3> mc   = mass.m_mass_times_x_bar;  // m*c
    const lps::LMat<S, 3, 3> mccT = mc * mc.transpose() / m;  // m*c*c^T

    const ABDJacobiDyadicMassT<S> cm_mass{.m_mass = m,
                                          .m_mass_times_x_bar = mc,
                                          .m_mass_times_dyadic_x_bar = mccT};
    const lps::LMat<S, 12, 12> M_cm = abd_dyadic_mass_to_mat<S>(cm_mass);

    // M_constraint = s_t*M_cm + s_r*(M - M_cm)
    //              = s_r*M + (s_t - s_r)*M_cm
    return s_r * M + (s_t - s_r) * M_cm;
}
}  // namespace soft_transform_constraint_detail

class SoftTransformConstraint final : public AffineBodyConstraint
{
    static constexpr U64 SoftTransformConstraintUID = 16ull;

  public:
    using AffineBodyConstraint::AffineBodyConstraint;

    // The reference's public host staging vectors and device buffers (the test drives
    // them; the class surface is the reference's).
    vector<IndexT>   h_constrained_bodies;
    vector<Vector12> h_aim_transforms;
    vector<Vector2>  h_strength_ratios;

    lud::LDeviceBuffer<IndexT>    constrained_bodies;
    lud::LDeviceBuffer<LVector12> aim_transforms;
    lud::LDeviceBuffer<LVector2>  strength_ratios;

    void do_build(BuildInfo& info) override;
    U64  get_uid() const noexcept override;
    void do_init(AffineBodyAnimator::FilteredInfo& info) override;
    void do_step(AffineBodyAnimator::FilteredInfo& info) override;
    void do_report_extent(AffineBodyAnimator::ReportExtentInfo& info) override;
    void do_compute_energy(AffineBodyAnimator::ComputeEnergyInfo& info) override;
    void do_compute_gradient_hessian(
        AffineBodyAnimator::ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa
