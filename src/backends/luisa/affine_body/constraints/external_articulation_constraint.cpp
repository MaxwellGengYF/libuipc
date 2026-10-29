/********************************************************************
 * @file affine_body/constraints/external_articulation_constraint.cpp
 * @brief T6.4b — the ABD external articulation constraint: its five kernels, the
 *        out-of-line members and the three system registrations (reference:
 *        src/backends/cuda/affine_body/constraints/external_articulation_constraint.cu).
 *
 * The three classes are declared in `external_articulation_constraint.h` (the
 * T6.3b1 file-split deviation, reported there).
 *
 * THE DEVICE BOUNDARIES THIS FILE CROSSES, AND HOW
 *  * `joint_id_to_uid` is *device* data, so the reference's
 *    `if(uid == RevoluteUID) ERJ::...; else if(uid == PrismaticUID) EPJ::...; else
 *    { UIPC_KERNEL_ASSERT(false, ...); <zero>; }` becomes "evaluate both candidates and
 *    select" (`abd_external_articulation_detail::select_if`), with the else-branch's
 *    assert routed through the `LC_LOG` kernel-logging facade and its zero selected
 *    when neither uid matches.  See `external_articulation_constraint_function.h`.
 *  * `ref_q_prev_ids[s] >= 0 ? ref_q_prevs(id) : q_prevs(body)` is *also* a device
 *    predicate, so both candidates are read.  The read of a `-1` sentinel is
 *    therefore clamped to 0 **and** `ref_q_prevs` is padded to one element in
 *    `do_init` (PORT ADDITION: `lla`/LC cannot pass a zero-length buffer as a kernel
 *    argument, and the reference's ternary simply never performs the read).
 *  * `FastSegmentalReduce().reduce(n, out, get_key, get_value)` has no LC equivalent
 *    that computes its value in-kernel: `lla::segmented_reduce` reads *both* the sorted
 *    keys and the values from buffers.  Keys are static (built once in `do_init`),
 *    values are one kernel launch (`..._compute_reduce_value_kernel`).  The reduce
 *    itself is T3.2's transcription of the same primitive and keeps its contract
 *    (zero-fill the output by identity, warp head-segmented fold, atomic combine only
 *    when a segment crosses a warp).
 *  * the two assembly write patterns come from
 *    `abd_external_articulation_detail::{write_pair_gradient, write_pair_half_hessian}`
 *    — the port's spelling of the reference's
 *    `DoubletVectorAssembler::segment<2>` / `TripletMatrixAssembler::half_block<2>`
 *    (including the reference's `upper_LR` block placement).
 ********************************************************************/
#include <affine_body/constraints/external_articulation_constraint.h>
#include <affine_body/utils.h>  // orthonormal_basis (the reference includes it too)
#include <frame_stream.h>
#include <kernel_cout.h>  // LC_LOG
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>
#include <luisa_device/linear/linear.h>

// `uipc::Transform` = `Eigen::Transform<Float,3,Eigen::Affine>`: the reference's `.cu`
// gets it through `cuda_tool`; the port needs the definition for `LT.rotation()` /
// `LT.inverse()` (the T6.3b1 C2079 finding, re-applied).
#include <Eigen/Geometry>
#include <algorithm>
#include <iterator>
#include <tuple>

namespace uipc::backend::luisa
{
namespace
{
// NOTE: a namespace alias has to be spelled with the `namespace` keyword here --
// MSVC parses `using Detail = <a-namespace>;` as a *type* alias and rejects it
// (measured C2061 "syntax error: identifier 'abd_external_articulation_detail'").
namespace Detail = abd_external_articulation_detail;

// ---------------------------------------------------------------------------
//  the per-joint state loader + the two branchless dispatches
// ---------------------------------------------------------------------------
/// One joint's dofs, loaded from the four (joint, dof) buffers.
struct JointDofs
{
    lps::LVec<lcrt::Var<double>, 6>  basis_k;
    lps::LVec<lcrt::Var<double>, 6>  basis_l;
    lps::LVec<lcrt::Var<double>, 12> qk;
    lps::LVec<lcrt::Var<double>, 12> ql;
    lps::LVec<lcrt::Var<double>, 12> q_prevk;
    lps::LVec<lcrt::Var<double>, 12> q_prevl;
};

[[nodiscard]] inline JointDofs load_joint_dofs(
    const lcrt::Var<IndexT>&          joint_id,
    const lcrt::BufferVar<LVector2i>& body_ids,
    const lcrt::BufferVar<LVector6>&  l_basis,
    const lcrt::BufferVar<LVector6>&  r_basis,
    const lcrt::BufferVar<LVector12>& qs,
    const lcrt::BufferVar<LVector12>& q_prevs,
    const lcrt::BufferVar<LVector2i>& ref_q_prev_ids,
    const lcrt::BufferVar<LVector12>& ref_q_prevs) noexcept
{
    const auto bid = load_vec2i(body_ids.read(joint_id));
    const auto rid = load_vec2i(ref_q_prev_ids.read(joint_id));

    // reference: `ref_q_prev_ids[0] >= 0 ? ref_q_prevs(ref_q_prev_ids[0])
    //                                    : q_prevs(body_ids[0])`
    // A *device* predicate -> both candidates are read, so the negative sentinel is
    // clamped to 0 first (that is why `ref_q_prevs` is never empty; see the header).
    const auto lid = scalar_detail::select_scalar(lcrt::cast<IndexT>(0), rid(0), rid(0) >= 0);
    const auto ridx = scalar_detail::select_scalar(lcrt::cast<IndexT>(0), rid(1), rid(1) >= 0);

    JointDofs d;
    d.basis_k = load_vec6(l_basis.read(joint_id));
    d.basis_l = load_vec6(r_basis.read(joint_id));
    d.qk      = load_vec12(qs.read(bid(0)));
    d.ql      = load_vec12(qs.read(bid(1)));
    d.q_prevk = Detail::select_if(rid(0) >= 0,
                                 load_vec12(ref_q_prevs.read(lid)),
                                 load_vec12(q_prevs.read(bid(0))));
    d.q_prevl = Detail::select_if(rid(1) >= 0,
                                 load_vec12(ref_q_prevs.read(ridx)),
                                 load_vec12(q_prevs.read(bid(1))));
    return d;
}

/// The reference's `compute_delta_theta(joint_id)` (the energy kernel's lambda, the
/// gradient/Hessian kernels' `compute_dDeltaTheta_dQ`'s scalar half and the time
/// integrator's body): the uid dispatch as a select, with the unknown-uid branch's
/// `UIPC_KERNEL_ASSERT(false, ...)` mapped onto the kernel-logging facade (a no-op
/// unless `UIPC_LUISA_KERNEL_PRINT`) and its zero selected.
[[nodiscard]] inline lcrt::Var<double> joint_delta_theta(
    const JointDofs&        d,
    const lcrt::Var<U64>&   uid,
    const lcrt::Var<U64>&   revolute_uid,
    const lcrt::Var<U64>&   prismatic_uid) noexcept
{
    const auto is_rev = uid == revolute_uid;
    const auto is_pri = uid == prismatic_uid;
    LC_LOG("ExternalArticulationConstraint: Unknown joint UID {}", uid);
    const auto known = Detail::select_if(is_rev,
                                         Detail::revolute_delta_theta<lcrt::Var<double>>(
                                             d.basis_k, d.qk, d.q_prevk, d.basis_l, d.ql, d.q_prevl),
                                         Detail::prismatic_delta_theta<lcrt::Var<double>>(
                                             d.basis_k, d.qk, d.q_prevk, d.basis_l, d.ql, d.q_prevl));
    return Detail::select_if(is_rev | is_pri, known, lcrt::Var<double>{0.0});
}

/// The reference's `compute_dDeltaTheta_dQ(joint_id)` (24, branchless).
[[nodiscard]] inline lps::LVec<lcrt::Var<double>, 24> joint_d_delta_theta_dq(
    const JointDofs&      d,
    const lcrt::Var<U64>& uid,
    const lcrt::Var<U64>& revolute_uid,
    const lcrt::Var<U64>& prismatic_uid) noexcept
{
    const auto is_rev = uid == revolute_uid;
    const auto is_pri = uid == prismatic_uid;
    const auto known  = Detail::select_if(
        is_rev,
        Detail::revolute_d_delta_theta_dq<lcrt::Var<double>>(
            d.basis_k, d.qk, d.q_prevk, d.basis_l, d.ql, d.q_prevl),
        Detail::prismatic_d_delta_theta_dq<lcrt::Var<double>>(
            d.basis_k, d.qk, d.q_prevk, d.basis_l, d.ql, d.q_prevl));
    return Detail::select_if(is_rev | is_pri, known, Detail::zero_vec<lcrt::Var<double>, 24>());
}

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `external_articulation_constraint_compute_energy_kernel`
auto& external_articulation_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,     // energies
                                                lcrt::Buffer<double>,     // joint_joint_id_to_mass
                                                lcrt::Buffer<LVector2i>,  // joint_joint_id_to_joint_ij
                                                lcrt::Buffer<double>,     // joint_id_to_delta_theta_tilde
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_body_ids
                                                lcrt::Buffer<LVector6>,   // joint_id_to_L_basis
                                                lcrt::Buffer<LVector6>,   // joint_id_to_R_basis
                                                lcrt::Buffer<U64>,        // joint_id_to_uid
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_ref_q_prev_ids
                                                lcrt::Buffer<LVector12>,  // ref_q_prevs
                                                U64,                      // PrismaticUID
                                                U64,                      // RevoluteUID
                                                uint32_t>(                // n
        [](lcrt::BufferVar<double>     energies,
           lcrt::BufferVar<double>     joint_joint_id_to_mass,
           lcrt::BufferVar<LVector2i>  joint_joint_id_to_joint_ij,
           lcrt::BufferVar<double>     joint_id_to_delta_theta_tilde,
           lcrt::BufferVar<LVector2i>  joint_id_to_body_ids,
           lcrt::BufferVar<LVector6>   joint_id_to_L_basis,
           lcrt::BufferVar<LVector6>   joint_id_to_R_basis,
           lcrt::BufferVar<U64>        joint_id_to_uid,
           lcrt::BufferVar<LVector12>  qs,
           lcrt::BufferVar<LVector12>  q_prevs,
           lcrt::BufferVar<LVector2i>  joint_id_to_ref_q_prev_ids,
           lcrt::BufferVar<LVector12>  ref_q_prevs,
           lcrt::Var<U64>              PrismaticUID,
           lcrt::Var<U64>              RevoluteUID,
           lcrt::Var<uint32_t>         n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto m_ij = joint_joint_id_to_mass.read(I);
                auto ij   = load_vec2i(joint_joint_id_to_joint_ij.read(I));

                auto delta_theta_i = joint_delta_theta(
                    load_joint_dofs(ij(0), joint_id_to_body_ids, joint_id_to_L_basis,
                                    joint_id_to_R_basis, qs, q_prevs,
                                    joint_id_to_ref_q_prev_ids, ref_q_prevs),
                    joint_id_to_uid.read(ij(0)), RevoluteUID, PrismaticUID);
                auto delta_theta_j = joint_delta_theta(
                    load_joint_dofs(ij(1), joint_id_to_body_ids, joint_id_to_L_basis,
                                    joint_id_to_R_basis, qs, q_prevs,
                                    joint_id_to_ref_q_prev_ids, ref_q_prevs),
                    joint_id_to_uid.read(ij(1)), RevoluteUID, PrismaticUID);

                auto delta_theta_tilde_i = joint_id_to_delta_theta_tilde.read(ij(0));
                auto delta_theta_tilde_j = joint_id_to_delta_theta_tilde.read(ij(1));

                // reference: `energies(I) = 0.5 * (delta_theta_i -
                // delta_theta_tilde_i) * m_ij * (delta_theta_j -
                // delta_theta_tilde_j);` (same association order)
                energies.write(I,
                               0.5 * (delta_theta_i - delta_theta_tilde_i) * m_ij
                                   * (delta_theta_j - delta_theta_tilde_j));
            };
        });
    return k;
}

/// The segmented reduce's *value* kernel (PORT ADDITION; see the file banner):
/// `m_ij * (delta_theta_j - delta_theta_tilde_j)` for every joint-joint entry.
auto& external_articulation_reduce_value_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,     // values
                                                lcrt::Buffer<double>,     // joint_joint_id_to_mass
                                                lcrt::Buffer<LVector2i>,  // joint_joint_id_to_joint_ij
                                                lcrt::Buffer<double>,     // joint_id_to_delta_theta_tilde
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_body_ids
                                                lcrt::Buffer<LVector6>,   // joint_id_to_L_basis
                                                lcrt::Buffer<LVector6>,   // joint_id_to_R_basis
                                                lcrt::Buffer<U64>,        // joint_id_to_uid
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_ref_q_prev_ids
                                                lcrt::Buffer<LVector12>,  // ref_q_prevs
                                                U64,                      // RevoluteUID
                                                U64,                      // PrismaticUID
                                                uint32_t>(                // n
        [](lcrt::BufferVar<double>     values,
           lcrt::BufferVar<double>     joint_joint_id_to_mass,
           lcrt::BufferVar<LVector2i>  joint_joint_id_to_joint_ij,
           lcrt::BufferVar<double>     joint_id_to_delta_theta_tilde,
           lcrt::BufferVar<LVector2i>  joint_id_to_body_ids,
           lcrt::BufferVar<LVector6>   joint_id_to_L_basis,
           lcrt::BufferVar<LVector6>   joint_id_to_R_basis,
           lcrt::BufferVar<U64>        joint_id_to_uid,
           lcrt::BufferVar<LVector12>  qs,
           lcrt::BufferVar<LVector12>  q_prevs,
           lcrt::BufferVar<LVector2i>  joint_id_to_ref_q_prev_ids,
           lcrt::BufferVar<LVector12>  ref_q_prevs,
           lcrt::Var<U64>              RevoluteUID,
           lcrt::Var<U64>              PrismaticUID,
           lcrt::Var<uint32_t>         n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto m_ij = joint_joint_id_to_mass.read(I);
                auto ij   = load_vec2i(joint_joint_id_to_joint_ij.read(I));

                // the reference's `ExternalArticulationConstraintGetValue`: the *j*-side
                // joint's current DeltaTheta against its own prediction.
                auto delta_theta_j = joint_delta_theta(
                    load_joint_dofs(ij(1), joint_id_to_body_ids, joint_id_to_L_basis,
                                    joint_id_to_R_basis, qs, q_prevs,
                                    joint_id_to_ref_q_prev_ids, ref_q_prevs),
                    joint_id_to_uid.read(ij(1)), RevoluteUID, PrismaticUID);

                values.write(I,
                             m_ij
                                 * (delta_theta_j
                                    - joint_id_to_delta_theta_tilde.read(ij(1))));
            };
        });
    return k;
}

/// `external_articulation_constraint_compute_gradient_hessian_k1_kernel`
auto& external_articulation_gradient_hessian_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,     // joint_id_to_G_theta
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_body_ids
                                                lcrt::Buffer<LVector6>,   // joint_id_to_L_basis
                                                lcrt::Buffer<LVector6>,   // joint_id_to_R_basis
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_ref_q_prev_ids
                                                lcrt::Buffer<LVector12>,  // ref_q_prevs
                                                lcrt::Buffer<U64>,        // joint_id_to_uid
                                                U64,                      // RevoluteUID
                                                U64,                      // PrismaticUID
                                                lcrt::Buffer<int32_t>,    // G indices
                                                lcrt::Buffer<double>,     // G values
                                                uint32_t,  // G total segments
                                                int32_t,   // G doublet index offset
                                                int32_t,   // G doublet count
                                                int32_t,   // G total doublet count
                                                uint32_t>(  // n
        [](lcrt::BufferVar<double>     joint_id_to_G_theta,
           lcrt::BufferVar<LVector2i>  joint_id_to_body_ids,
           lcrt::BufferVar<LVector6>   joint_id_to_L_basis,
           lcrt::BufferVar<LVector6>   joint_id_to_R_basis,
           lcrt::BufferVar<LVector12>  qs,
           lcrt::BufferVar<LVector12>  q_prevs,
           lcrt::BufferVar<LVector2i>  joint_id_to_ref_q_prev_ids,
           lcrt::BufferVar<LVector12>  ref_q_prevs,
           lcrt::BufferVar<U64>        joint_id_to_uid,
           lcrt::Var<U64>              RevoluteUID,
           lcrt::Var<U64>              PrismaticUID,
           lcrt::BufferVar<int32_t>    G_indices,
           lcrt::BufferVar<double>     G_values,
           lcrt::Var<uint32_t>         G_total_segments,
           lcrt::Var<int32_t>          G_doublet_index_offset,
           lcrt::Var<int32_t>          G_doublet_count,
           lcrt::Var<int32_t>          G_total_doublet_count,
           lcrt::Var<uint32_t>         n) noexcept
        {
            auto jointI = lcrt::dispatch_x();
            $if(jointI < n)
            {
                const auto G_theta = joint_id_to_G_theta.read(jointI);
                const auto bid     = load_vec2i(joint_id_to_body_ids.read(jointI));

                const auto d = load_joint_dofs(jointI, joint_id_to_body_ids,
                                               joint_id_to_L_basis, joint_id_to_R_basis, qs,
                                               q_prevs, joint_id_to_ref_q_prev_ids,
                                               ref_q_prevs);

                // reference: `G24 *= G_theta;`
                auto G24 = joint_d_delta_theta_dq(d, joint_id_to_uid.read(jointI),
                                                  RevoluteUID, PrismaticUID);
                G24      = G24 * G_theta;

                auto ii = lcrt::cast<std::int32_t>(jointI);

                auto D = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 12>::from_handles_full(
                    G_indices,
                    G_values,
                    G_total_segments,
                    G_doublet_index_offset,
                    G_doublet_count,
                    G_total_doublet_count,
                    std::int32_t{0},
                    G_total_segments);
                abd_external_articulation_detail::write_pair_gradient(D, G24, ii * 2, bid);
            };
        });
    return k;
}

/// `external_articulation_constraint_compute_gradient_hessian_k2_kernel`
auto& external_articulation_gradient_hessian_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,     // joint_id_to_G_theta
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_body_ids
                                                lcrt::Buffer<double>,     // joint_joint_id_to_mass
                                                lcrt::Buffer<LVector2i>,  // joint_joint_id_to_joint_ij
                                                lcrt::Buffer<LVector6>,   // joint_id_to_L_basis
                                                lcrt::Buffer<LVector6>,   // joint_id_to_R_basis
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_ref_q_prev_ids
                                                lcrt::Buffer<LVector12>,  // ref_q_prevs
                                                lcrt::Buffer<U64>,        // joint_id_to_uid
                                                U64,                      // RevoluteUID
                                                U64,                      // PrismaticUID
                                                uint32_t,                 // HalfHessianSize
                                                lcrt::Buffer<int32_t>,    // H row indices
                                                lcrt::Buffer<int32_t>,    // H col indices
                                                lcrt::Buffer<double>,     // H values
                                                int32_t,                  // H total block rows
                                                int32_t,                  // H total block cols
                                                int32_t,  // H triplet index offset
                                                int32_t,  // H triplet count
                                                int32_t,  // H total triplet count
                                                int32_t,  // H submatrix offset x
                                                int32_t,  // H submatrix offset y
                                                int32_t,  // H submatrix extent x
                                                int32_t,  // H submatrix extent y
                                                uint32_t>(  // n
        [](lcrt::BufferVar<double>     joint_id_to_G_theta,
           lcrt::BufferVar<LVector2i>  joint_id_to_body_ids,
           lcrt::BufferVar<double>     joint_joint_id_to_mass,
           lcrt::BufferVar<LVector2i>  joint_joint_id_to_joint_ij,
           lcrt::BufferVar<LVector6>   joint_id_to_L_basis,
           lcrt::BufferVar<LVector6>   joint_id_to_R_basis,
           lcrt::BufferVar<LVector12>  qs,
           lcrt::BufferVar<LVector12>  q_prevs,
           lcrt::BufferVar<LVector2i>  joint_id_to_ref_q_prev_ids,
           lcrt::BufferVar<LVector12>  ref_q_prevs,
           lcrt::BufferVar<U64>        joint_id_to_uid,
           lcrt::Var<U64>              RevoluteUID,
           lcrt::Var<U64>              PrismaticUID,
           lcrt::Var<uint32_t>         HalfHessianSize,
           lcrt::BufferVar<int32_t>    H_row_indices,
           lcrt::BufferVar<int32_t>    H_col_indices,
           lcrt::BufferVar<double>     H_values,
           lcrt::Var<int32_t>          H_total_rows,
           lcrt::Var<int32_t>          H_total_cols,
           lcrt::Var<int32_t>          H_triplet_index_offset,
           lcrt::Var<int32_t>          H_triplet_count,
           lcrt::Var<int32_t>          H_total_triplet_count,
           lcrt::Var<int32_t>          H_sm_offset_x,
           lcrt::Var<int32_t>          H_sm_offset_y,
           lcrt::Var<int32_t>          H_sm_extent_x,
           lcrt::Var<int32_t>          H_sm_extent_y,
           lcrt::Var<uint32_t>         n) noexcept
        {
            auto joint_joint_I = lcrt::dispatch_x();
            $if(joint_joint_I < n)
            {
                const auto m_ij = joint_joint_id_to_mass.read(joint_joint_I);
                const auto ij   = load_vec2i(joint_joint_id_to_joint_ij.read(joint_joint_I));

                const auto di = load_joint_dofs(ij(0), joint_id_to_body_ids,
                                                joint_id_to_L_basis, joint_id_to_R_basis, qs,
                                                q_prevs, joint_id_to_ref_q_prev_ids,
                                                ref_q_prevs);
                const auto dj = load_joint_dofs(ij(1), joint_id_to_body_ids,
                                                joint_id_to_L_basis, joint_id_to_R_basis, qs,
                                                q_prevs, joint_id_to_ref_q_prev_ids,
                                                ref_q_prevs);

                auto dDeltaTheta_dQ_i = joint_d_delta_theta_dq(
                    di, joint_id_to_uid.read(ij(0)), RevoluteUID, PrismaticUID);
                auto dDeltaTheta_dQ_j = joint_d_delta_theta_dq(
                    dj, joint_id_to_uid.read(ij(1)), RevoluteUID, PrismaticUID);

                // reference: `H24x24 = dDeltaTheta_dQ_i * m_ij * dDeltaTheta_dQ_j.transpose();`
                auto H24x24 = dDeltaTheta_dQ_i * m_ij * dDeltaTheta_dQ_j.transpose();

                // reference: `if(ij[0] == ij[1]) { ... H24x24 += JT_H_J; }` — a device
                // predicate, so the extra term is always computed and *selected in*.
                {
                    const auto is_diagonal = ij(0) == ij(1);
                    const auto uid_i       = joint_id_to_uid.read(ij(0));
                    const auto is_rev      = uid_i == RevoluteUID;
                    const auto is_pri      = uid_i == PrismaticUID;

                    // `ERJ::F<Float>(F, basis_k, qk, basis_l, ql)` and its *_t twin;
                    // the prismatic law differs (its F carries the attachment points).
                    const auto dd_rev = Detail::revolute_dd_delta_theta_ddf<
                        lcrt::Var<double>>(
                        Detail::revolute_f<lcrt::Var<double>>(
                            di.basis_k, di.qk, di.basis_l, di.ql),
                        Detail::revolute_f<lcrt::Var<double>>(
                            di.basis_k, di.q_prevk, di.basis_l, di.q_prevl));
                    const auto dd_pri = Detail::prismatic_dd_delta_theta_ddf<
                        lcrt::Var<double>>(
                        Detail::prismatic_f<lcrt::Var<double>>(
                            di.basis_k, di.qk, di.basis_l, di.ql),
                        Detail::prismatic_f<lcrt::Var<double>>(
                            di.basis_k, di.q_prevk, di.basis_l, di.q_prevl));

                    const auto dd = Detail::select_if(is_rev | is_pri,
                                                      Detail::select_if(is_rev, dd_rev, dd_pri),
                                                      Detail::zero_mat<lcrt::Var<double>, 12, 12>());

                    // reference: `Matrix12x12 HF = G_theta_i * ddDeltaTheta_ddF;
                    //             make_spd(HF);`
                    auto HF = joint_id_to_G_theta.read(ij(0)) * dd;
                    lps::make_spd_inplace(HF, abd_external_articulation_detail::kMakeSpdSweeps);

                    // both generated JT_H_J's take the same (H_F, basis_k, basis_l,
                    // basis_m, basis_n) and return the same shape; the *Jacobian*
                    // inside differs, so the two candidates are selected.
                    const auto jt_rev = Detail::revolute_jt_h_j<lcrt::Var<double>>(
                        HF, di.basis_k, di.basis_l, di.basis_k, di.basis_l);
                    const auto jt_pri = Detail::prismatic_jt_h_j<lcrt::Var<double>>(
                        HF, di.basis_k, di.basis_l, di.basis_k, di.basis_l);
                    const auto jt = Detail::select_if(is_rev, jt_rev, jt_pri);

                    H24x24 = H24x24
                             + Detail::select_if(is_diagonal, jt,
                                                 Detail::zero_mat<lcrt::Var<double>, 24, 24>());
                }

                const auto body_ids_i = load_vec2i(joint_id_to_body_ids.read(ij(0)));
                const auto body_ids_j = load_vec2i(joint_id_to_body_ids.read(ij(1)));

                auto ii = lcrt::cast<std::int32_t>(joint_joint_I);

                auto T = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 12, 12>::from_handles_full(
                    H_row_indices,
                    H_col_indices,
                    H_values,
                    H_total_rows,
                    H_total_cols,
                    H_triplet_index_offset,
                    H_triplet_count,
                    H_total_triplet_count,
                    H_sm_offset_x,
                    H_sm_offset_y,
                    H_sm_extent_x,
                    H_sm_extent_y);
                abd_external_articulation_detail::write_pair_half_hessian(
                    T, H24x24, ii * lcrt::cast<std::int32_t>(HalfHessianSize),
                    // NOTE: the reference passes the two *body* id pairs, in the
                    // (ij[0], ij[1]) order (NOT sorted) -- `half_block<2>`'s own
                    // upper_LR handles the descending case.
                    lps::LVec<lcrt::Var<IndexT>, 2>{body_ids_i(0), body_ids_j(1)});
            };
        });
    return k;
}

/// `external_articulation_constraint_time_integrator_update_state_kernel`
auto& external_articulation_update_state_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,     // joint_id_to_delta_theta
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_body_ids
                                                lcrt::Buffer<LVector6>,   // joint_id_to_L_basis
                                                lcrt::Buffer<LVector6>,   // joint_id_to_R_basis
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector2i>,  // joint_id_to_ref_q_prev_ids
                                                lcrt::Buffer<LVector12>,  // ref_q_prevs
                                                lcrt::Buffer<U64>,        // joint_id_to_uid
                                                U64,                      // RevoluteUID
                                                U64,                      // PrismaticUID
                                                uint32_t>(                // n
        [](lcrt::BufferVar<double>     joint_id_to_delta_theta,
           lcrt::BufferVar<LVector2i>  joint_id_to_body_ids,
           lcrt::BufferVar<LVector6>   joint_id_to_L_basis,
           lcrt::BufferVar<LVector6>   joint_id_to_R_basis,
           lcrt::BufferVar<LVector12>  qs,
           lcrt::BufferVar<LVector12>  q_prevs,
           lcrt::BufferVar<LVector2i>  joint_id_to_ref_q_prev_ids,
           lcrt::BufferVar<LVector12>  ref_q_prevs,
           lcrt::BufferVar<U64>        joint_id_to_uid,
           lcrt::Var<U64>              RevoluteUID,
           lcrt::Var<U64>              PrismaticUID,
           lcrt::Var<uint32_t>         n) noexcept
        {
            auto jointI = lcrt::dispatch_x();
            $if(jointI < n)
            {
                const auto d = load_joint_dofs(jointI, joint_id_to_body_ids,
                                               joint_id_to_L_basis, joint_id_to_R_basis, qs,
                                               q_prevs, joint_id_to_ref_q_prev_ids,
                                               ref_q_prevs);
                joint_id_to_delta_theta.write(
                    jointI,
                    joint_delta_theta(d, joint_id_to_uid.read(jointI), RevoluteUID,
                                      PrismaticUID));
            };
        });
    return k;
}
}  // namespace

// ===========================================================================
//  ExternalArticulationConstituion (uid 23) -- the (deliberately empty) constitution
// ===========================================================================
REGISTER_SIM_SYSTEM(ExternalArticulationConstituion);

U64 ExternalArticulationConstituion::get_uid() const noexcept
{
    return ConstitutionUID;
}

void ExternalArticulationConstituion::do_build(BuildInfo& info)
{
    (void)info;
}

void ExternalArticulationConstituion::do_init(FilteredInfo& info)
{
    (void)info;
}

void ExternalArticulationConstituion::do_report_energy_extent(EnergyExtentInfo& info)
{
    info.energy_count(0);
}

void ExternalArticulationConstituion::do_compute_energy(ComputeEnergyInfo& info)
{
    (void)info;
}

void ExternalArticulationConstituion::do_report_gradient_hessian_extent(
    GradientHessianExtentInfo& info)
{
    info.gradient_count(0);
    info.hessian_count(0);
}

void ExternalArticulationConstituion::do_compute_gradient_hessian(
    ComputeGradientHessianInfo& info)
{
    (void)info;
}

// ===========================================================================
//  ExternalArticulationConstraint (uid 24)
// ===========================================================================
REGISTER_SIM_SYSTEM(ExternalArticulationConstraint);

U64 ExternalArticulationConstraint::get_uid() const noexcept
{
    return ConstraintUID;
}

void ExternalArticulationConstraint::do_build(BuildInfo& info)
{
    (void)info;
    // The reference writes the per-joint `delta_theta` back to the scene through this
    // hook (the T6.3b2 `AffineBodyPrismaticJoint` precedent).
    on_write_scene([this]() { write_scene(); });
}

auto ExternalArticulationConstraint::get_revolute_basis(const geometry::SimplicialComplex* L,
                                                        IndexT L_inst_id,
                                                        const geometry::SimplicialComplex* R,
                                                        IndexT R_inst_id,
                                                        const geometry::SimplicialComplex* joint_mesh,
                                                        IndexT joint_index)
{
    auto topo_view = joint_mesh->edges().topo().view();
    auto pos_view  = joint_mesh->positions().view();

    Vector2i e = topo_view[joint_index];
    Vector3  t = pos_view[e[1]] - pos_view[e[0]];
    Vector3  n;
    Vector3  b;
    // reference: `orthonormal_basis(t, n, b);` (Eigen `Vector3` in the reference,
    // because its `affine_body/utils.h` is Eigen-typed).  The port's routine is the
    // *generic* one over the numerics layer's `LVec<S,3>`, so the host-side call
    // bridges through `to_luisa`/`to_eigen` (the T6.3b1 idiom).  `t` is normalised in
    // place by the routine, exactly as in the reference.
    {
        auto t_lv = to_luisa(t);
        auto n_lv = LVector3{};
        auto b_lv = LVector3{};
        orthonormal_basis(t_lv, n_lv, b_lv);
        t = to_eigen(t_lv);
        n = to_eigen(n_lv);
        b = to_eigen(b_lv);
    }

    auto compute_bn_bar = [&](const geometry::SimplicialComplex* geo, IndexT inst_id) -> Vector6
    {
        UIPC_ASSERT(geo, "ExternalArticulationConstraint: Geometry is null when computing bn_bar");

        const Matrix4x4& trans = geo->transforms().view()[inst_id];
        Transform        T{trans};
        Matrix3x3        InvRot = T.rotation().inverse();
        Vector6          bn_bar;
        bn_bar.segment<3>(0) = InvRot * n;
        bn_bar.segment<3>(3) = InvRot * b;
        return bn_bar;
    };

    Vector6 L_bn_bar = compute_bn_bar(L, L_inst_id);
    Vector6 R_bn_bar = compute_bn_bar(R, R_inst_id);

    return std::tuple{L_bn_bar, R_bn_bar};
}

auto ExternalArticulationConstraint::get_prismatic_basis(const geometry::SimplicialComplex* L,
                                                         IndexT L_inst_id,
                                                         const geometry::SimplicialComplex* R,
                                                         IndexT R_inst_id,
                                                         const geometry::SimplicialComplex* joint_mesh,
                                                         IndexT joint_index)
{
    auto topo_view = joint_mesh->edges().topo().view();
    auto pos_view  = joint_mesh->positions().view();

    Vector2i e = topo_view[joint_index];
    Vector3  t = pos_view[e[1]] - pos_view[e[0]];
    t          = t.normalized();
    Vector3 c  = pos_view[e[0]];

    auto compute_ct_bar = [&](const geometry::SimplicialComplex* geo, IndexT inst_id) -> Vector6
    {
        UIPC_ASSERT(geo, "ExternalArticulationConstraint: Geometry is null when computing bn_bar");

        const Matrix4x4& trans = geo->transforms().view()[inst_id];
        Transform        T{trans};

        Matrix3x3 InvRot = T.rotation().inverse();
        Vector6   bn_bar;
        bn_bar.segment<3>(0) = T.inverse() * c;
        bn_bar.segment<3>(3) = InvRot * t;
        return bn_bar;
    };

    Vector6 L_ct_bar = compute_ct_bar(L, L_inst_id);
    Vector6 R_ct_bar = compute_ct_bar(R, R_inst_id);

    return std::tuple{L_ct_bar, R_ct_bar};
}

IndexT ExternalArticulationConstraint::get_ref_q_prev(const geometry::SimplicialComplex* geo,
                                                      IndexT            inst_id,
                                                      IndexT            body_id)
{
    IndexT ref_q_prev_id = -1;
    auto   ref_dof_prev  = geo->instances().find<Vector12>("ref_dof_prev");
    if(ref_dof_prev)
    {
        auto iter = h_body_id_to_ref_q_prev_id.find(body_id);
        if(iter != h_body_id_to_ref_q_prev_id.end())
        {
            ref_q_prev_id = iter->second;
        }
        else
        {
            auto ref_dof_prev_view = ref_dof_prev->view();
            ref_q_prev_id = static_cast<IndexT>(h_ref_q_prevs.size());
            // record it to use it in `do_step`
            h_attr_ref_q_prevs.push_back({ref_dof_prev, inst_id});
            h_ref_q_prevs.push_back(ref_dof_prev_view[inst_id]);
        }
    }

    return ref_q_prev_id;
}

void ExternalArticulationConstraint::build_joint_info(
    span<S<geometry::GeometrySlot>>        geo_slots,
    InterAffineBodyAnimator::FilteredInfo& info,
    const geometry::Geometry&              mesh,
    IndexT                                 index)
{
    auto joint_mesh = mesh.as<geometry::SimplicialComplex>();
    UIPC_ASSERT(joint_mesh,
                "ExternalArticulationConstraint: Joint geometry ({}) is not a SimplicialComplex",
                joint_mesh->to_json().dump(4));

    UIPC_ASSERT(index < static_cast<IndexT>(joint_mesh->edges().size()) && index >= 0,
                "ExternalArticulationConstraint: Joint index ({}) out of range in Joint Geometry ({}), [0, {}) expected",
                index,
                joint_mesh->to_json().dump(4),
                joint_mesh->edges().size());

    auto uid_attr = joint_mesh->meta().find<U64>(builtin::constitution_uid);
    UIPC_ASSERT(uid_attr,
                "ExternalArticulationConstraint: 'constitution_uid' meta attribute not found in joint geometry");

    U64 uid_value = uid_attr->view()[0];
    UIPC_ASSERT(uid_value == RevoluteJointConstitutionUID
                    || uid_value == PrismaticJointConstitutionUID,
                "ExternalArticulationConstraint: Now only support RevoluteJointConstraint (UID={}) and PrismaticJointConstraint (UID={}), but found joint with UID {}",
                RevoluteJointConstitutionUID,
                PrismaticJointConstitutionUID,
                uid_value);

    h_joint_id_to_uid.push_back(uid_value);

    // 1. Get Joint's two Links
    auto joint_l_geo_id = joint_mesh->edges().find<IndexT>("l_geo_id");
    UIPC_ASSERT(joint_l_geo_id,
                "ExternalArticulationConstraint: Joint geometry must have 'l_geo_id' attribute on `edges`");
    auto joint_r_geo_id = joint_mesh->edges().find<IndexT>("r_geo_id");
    UIPC_ASSERT(joint_r_geo_id,
                "ExternalArticulationConstraint: Joint geometry must have 'r_geo_id' attribute on `edges`");
    auto joint_l_inst_id = joint_mesh->edges().find<IndexT>("l_inst_id");
    UIPC_ASSERT(joint_l_inst_id,
                "ExternalArticulationConstraint: Joint geometry must have 'l_inst_id' attribute on `edges`");
    auto joint_r_inst_id = joint_mesh->edges().find<IndexT>("r_inst_id");
    UIPC_ASSERT(joint_r_inst_id,
                "ExternalArticulationConstraint: Joint geometry must have 'r_inst_id' attribute on `edges`");
    auto joint_l_geo_id_view  = joint_l_geo_id->view();
    auto joint_r_geo_id_view  = joint_r_geo_id->view();
    auto joint_l_inst_id_view = joint_l_inst_id->view();
    auto joint_r_inst_id_view = joint_r_inst_id->view();

    const IndexT l_gid = joint_l_geo_id_view[index];
    const IndexT r_gid = joint_r_geo_id_view[index];
    const IndexT l_iid = joint_l_inst_id_view[index];
    const IndexT r_iid = joint_r_inst_id_view[index];

    Vector2i body_id = {
        info.body_id(l_gid, l_iid),
        info.body_id(r_gid, r_iid),
    };
    h_joint_id_to_body_ids.push_back(body_id);

    auto L_link_geo = info.body_geo(geo_slots, l_gid);
    auto R_link_geo = info.body_geo(geo_slots, r_gid);

    auto l_pos0_attr = joint_mesh->edges().find<Vector3>("l_position0");
    auto l_pos1_attr = joint_mesh->edges().find<Vector3>("l_position1");
    auto r_pos0_attr = joint_mesh->edges().find<Vector3>("r_position0");
    auto r_pos1_attr = joint_mesh->edges().find<Vector3>("r_position1");
    bool use_local   = l_pos0_attr && l_pos1_attr && r_pos0_attr && r_pos1_attr;

    Vector6 L_basis, R_basis;

    if(use_local)
    {
        Transform LT{L_link_geo->transforms().view()[l_iid]};
        Transform RT{R_link_geo->transforms().view()[r_iid]};

        if(uid_value == RevoluteJointConstitutionUID)
        {
            Vector3 t = LT.rotation()
                        * (l_pos1_attr->view()[index] - l_pos0_attr->view()[index]);
            Vector3 n, b_vec;
            // the port's generic routine (see `get_revolute_basis`)
            {
                auto t_lv = to_luisa(t);
                auto n_lv = LVector3{};
                auto b_lv = LVector3{};
                orthonormal_basis(t_lv, n_lv, b_lv);
                t     = to_eigen(t_lv);
                n     = to_eigen(n_lv);
                b_vec = to_eigen(b_lv);
            }
            Matrix3x3 L_inv_rot   = LT.rotation().inverse();
            Matrix3x3 R_inv_rot   = RT.rotation().inverse();
            L_basis.segment<3>(0) = L_inv_rot * n;
            L_basis.segment<3>(3) = L_inv_rot * b_vec;
            R_basis.segment<3>(0) = R_inv_rot * n;
            R_basis.segment<3>(3) = R_inv_rot * b_vec;
        }
        else if(uid_value == PrismaticJointConstitutionUID)
        {
            Vector3 l_t = (LT.rotation()
                           * (l_pos1_attr->view()[index] - l_pos0_attr->view()[index]))
                              .normalized();
            Vector3 l_c = LT * l_pos0_attr->view()[index];
            Vector3 r_t = (RT.rotation()
                           * (r_pos1_attr->view()[index] - r_pos0_attr->view()[index]))
                              .normalized();
            Vector3 r_c           = RT * r_pos0_attr->view()[index];
            L_basis.segment<3>(0) = LT.inverse() * l_c;
            L_basis.segment<3>(3) = LT.rotation().inverse() * l_t;
            R_basis.segment<3>(0) = RT.inverse() * r_c;
            R_basis.segment<3>(3) = RT.rotation().inverse() * r_t;
        }
    }
    else
    {
        if(uid_value == RevoluteJointConstitutionUID)
        {
            std::tie(L_basis, R_basis) = get_revolute_basis(
                L_link_geo, l_iid, R_link_geo, r_iid, joint_mesh, index);
        }
        else if(uid_value == PrismaticJointConstitutionUID)
        {
            std::tie(L_basis, R_basis) = get_prismatic_basis(
                L_link_geo, l_iid, R_link_geo, r_iid, joint_mesh, index);
        }
    }

    h_joint_id_to_L_basis.push_back(L_basis);
    h_joint_id_to_R_basis.push_back(R_basis);

    // 2. Get Joint's 2 ref_q_prev (if any)
    Vector2i ref_q_prev_ids = {
        get_ref_q_prev(L_link_geo, l_iid, body_id[0]),
        get_ref_q_prev(R_link_geo, r_iid, body_id[1]),
    };

    h_joint_id_to_ref_q_prev_ids.push_back(ref_q_prev_ids);
}

void ExternalArticulationConstraint::do_init(InterAffineBodyAnimator::FilteredInfo& info)
{
    auto scene     = world().scene();
    auto geo_slots = scene.geometries();

    auto art_count = info.anim_inter_geo_infos().size();

    h_art_id_to_joint_offsets_counts.resize(art_count);
    h_art_id_to_joint_joint_offsets_counts.resize(art_count);

    // 1. Collect Data
    info.for_each(
        geo_slots,
        [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
        {
            auto joint_collection = geo["joint"];
            UIPC_ASSERT(joint_collection,
                        "ExternalArticulationConstraint: 'joint' attribute collection not found in geometry");

            h_art_id_to_joint_offsets_counts.counts()[I.index()] =
                static_cast<IndexT>(joint_collection->size());

            auto geo_ids = joint_collection->find<IndexT>("geo_id");
            UIPC_ASSERT(geo_ids,
                        "ExternalArticulationConstraint: 'geo_id' attribute not found in joint collection");
            auto geo_id_view = geo_ids->view();

            auto indices = joint_collection->find<IndexT>("index");
            UIPC_ASSERT(indices,
                        "ExternalArticulationConstraint: 'index' attribute not found in joint collection");
            auto index_view = indices->view();

            auto delta_theta_tilde = joint_collection->find<Float>("delta_theta_tilde");
            UIPC_ASSERT(delta_theta_tilde,
                        "ExternalArticulationConstraint: 'delta_theta_tilde' attribute not found in joint collection");
            auto delta_theta_tilde_view = delta_theta_tilde->view();

            auto joint_joint_collection = geo["joint_joint"];
            UIPC_ASSERT(joint_joint_collection,
                        "ExternalArticulationConstraint: 'joint_joint' attribute collection not found in geometry");
            UIPC_ASSERT(joint_joint_collection->size()
                            == joint_collection->size() * joint_collection->size(),
                        "ExternalArticulationConstraint: 'joint_joint' collection size mismatch, expected {} * {} = {}, got {}",
                        joint_collection->size(),
                        joint_collection->size(),
                        joint_collection->size() * joint_collection->size(),
                        joint_joint_collection->size());

            auto mass = joint_joint_collection->find<Float>("mass");
            UIPC_ASSERT(mass,
                        "ExternalArticulationConstraint: 'mass' attribute not found in joint_joint collection");

            auto mass_view = mass->view();

            h_art_id_to_joint_joint_offsets_counts.counts()[I.index()] =
                static_cast<IndexT>(mass->size());

            for(auto&& [geo_id, index, dtt] :
                zip(geo_id_view, index_view, delta_theta_tilde_view))
            {
                // find the joint geometry in the scene
                auto geo_slot = scene.find_geometry(geo_id);

                h_joint_id_to_art_id.push_back(static_cast<IndexT>(I.index()));
                h_joint_id_to_delta_theta_tilde.push_back(dtt);

                build_joint_info(geo_slots, info, geo_slot->geometry(), index);
            }

            for(auto&& m : mass_view)
            {
                h_joint_joint_id_to_mass.push_back(m);
                h_joint_joint_id_to_art_id.push_back(static_cast<IndexT>(I.index()));
            }

            auto n_joint = static_cast<IndexT>(joint_collection->size());
            for(IndexT i = 0; i < n_joint; ++i)
            {
                for(IndexT j = 0; j < n_joint; ++j)
                {
                    // i must go slower than j
                    // so that the `fast_segmental_reduce` can work correctly in gradient/hessian computation
                    h_joint_joint_id_to_joint_ij.push_back(Vector2i{i, j});
                }
            }

            // theta_tilde and mass will change in each step, so we just prepare the slots here
            // later we will fill them in do_step()
            h_art_joint_delta_theta_tilde.push_back(delta_theta_tilde);
            h_art_joint_joint_mass.push_back(mass);
        });

    // 2. Compute Offsets
    h_art_id_to_joint_offsets_counts.scan();
    // shift the joint_ij by joint offsets
    for(auto&& [I, ij] : enumerate(h_joint_joint_id_to_joint_ij))
    {
        auto art_id   = h_joint_joint_id_to_art_id[I];
        auto j_offset = h_art_id_to_joint_offsets_counts.offsets()[art_id];
        ij += Vector2i{j_offset, j_offset};
    }
    h_art_id_to_joint_joint_offsets_counts.scan();

    // PORT ADDITION: the segmented reduce's keys (static; see the header note).
    h_joint_joint_id_to_key.clear();
    h_joint_joint_id_to_key.reserve(h_joint_joint_id_to_joint_ij.size());
    for(auto&& ij : h_joint_joint_id_to_joint_ij)
    {
        h_joint_joint_id_to_key.push_back(static_cast<std::int32_t>(ij(0)));
    }

    // 3. Upload to Device

    // Joint Info
    auto n_joint = static_cast<SizeT>(h_joint_id_to_art_id.size());

    {
        auto h_art_joint_offsets = h_art_id_to_joint_offsets_counts.offsets();
        art_id_to_joint_offsets.resize(h_art_joint_offsets.size());
        if(!h_art_joint_offsets.empty())
        {
            art_id_to_joint_offsets.view().copy_from(h_art_joint_offsets);
        }

        auto h_art_joint_counts = h_art_id_to_joint_offsets_counts.counts();
        art_id_to_joint_counts.resize(h_art_joint_counts.size());
        if(!h_art_joint_counts.empty())
        {
            art_id_to_joint_counts.view().copy_from(h_art_joint_counts);
        }

        joint_id_to_art_id.resize(n_joint);
        if(!h_joint_id_to_art_id.empty())
        {
            joint_id_to_art_id.view().copy_from(span<const IndexT>{h_joint_id_to_art_id});
        }

        joint_id_to_uid.resize(h_joint_id_to_uid.size());
        if(!h_joint_id_to_uid.empty())
        {
            joint_id_to_uid.view().copy_from(span<const U64>{h_joint_id_to_uid});
        }

        joint_id_to_body_ids.resize(h_joint_id_to_body_ids.size());
        {
            // the host `Vector2i` (Eigen) -> device `LVector2i` crossing: the global
            // `to_luisa(Vector2i)` is an exact match, so `copy_host_to_device`-style
            // element-wise staging is the port's idiom (the T6.3b2 rule-15 note).
            std::vector<LVector2i> staging;
            staging.reserve(h_joint_id_to_body_ids.size());
            for(auto& v : h_joint_id_to_body_ids)
                staging.push_back(to_luisa(v));
            if(!staging.empty())
            {
                joint_id_to_body_ids.view().copy_from(span<const LVector2i>{staging});
            }
        }
    }

    // init delta_theta to zero
    joint_id_to_delta_theta.resize(n_joint);
    joint_id_to_delta_theta.fill(0.0);
    h_joint_id_to_delta_theta.assign(n_joint, 0.0);

    // init delta_theta_tilde
    joint_id_to_delta_theta_tilde.resize(h_joint_id_to_delta_theta_tilde.size());
    if(!h_joint_id_to_delta_theta_tilde.empty())
    {
        joint_id_to_delta_theta_tilde.view().copy_from(
            span<const Float>{h_joint_id_to_delta_theta_tilde});
    }

    joint_id_to_L_basis.resize(h_joint_id_to_L_basis.size());
    if(!h_joint_id_to_L_basis.empty())
    {
        std::vector<LVector6> staging;
        staging.reserve(h_joint_id_to_L_basis.size());
        for(auto& v : h_joint_id_to_L_basis)
            staging.push_back(to_luisa(v));
        joint_id_to_L_basis.view().copy_from(span<const LVector6>{staging});
    }

    joint_id_to_R_basis.resize(h_joint_id_to_R_basis.size());
    if(!h_joint_id_to_R_basis.empty())
    {
        std::vector<LVector6> staging;
        staging.reserve(h_joint_id_to_R_basis.size());
        for(auto& v : h_joint_id_to_R_basis)
            staging.push_back(to_luisa(v));
        joint_id_to_R_basis.view().copy_from(span<const LVector6>{staging});
    }

    joint_id_to_G_theta.resize(n_joint);

    // Joint-Joint Info
    {
        auto h_art_mass_offsets = h_art_id_to_joint_joint_offsets_counts.offsets();
        art_id_to_joint_joint_offsets.resize(h_art_mass_offsets.size());
        if(!h_art_mass_offsets.empty())
        {
            art_id_to_joint_joint_offsets.view().copy_from(h_art_mass_offsets);
        }

        auto h_art_mass_counts = h_art_id_to_joint_joint_offsets_counts.counts();
        art_id_to_joint_joint_counts.resize(h_art_mass_counts.size());
        if(!h_art_mass_counts.empty())
        {
            art_id_to_joint_joint_counts.view().copy_from(h_art_mass_counts);
        }

        joint_joint_id_to_mass.resize(h_joint_joint_id_to_mass.size());
        if(!h_joint_joint_id_to_mass.empty())
        {
            joint_joint_id_to_mass.view().copy_from(
                span<const Float>{h_joint_joint_id_to_mass});
        }

        joint_joint_id_to_value.resize(h_joint_joint_id_to_mass.size());

        joint_joint_id_to_joint_ij.resize(h_joint_joint_id_to_joint_ij.size());
        if(!h_joint_joint_id_to_joint_ij.empty())
        {
            std::vector<LVector2i> staging;
            staging.reserve(h_joint_joint_id_to_joint_ij.size());
            for(auto& v : h_joint_joint_id_to_joint_ij)
                staging.push_back(to_luisa(v));
            joint_joint_id_to_joint_ij.view().copy_from(span<const LVector2i>{staging});
        }

        joint_joint_id_to_key.resize(h_joint_joint_id_to_key.size());
        if(!h_joint_joint_id_to_key.empty())
        {
            joint_joint_id_to_key.view().copy_from(
                span<const std::int32_t>{h_joint_joint_id_to_key});
        }
    }

    // Ref q_prev
    joint_id_to_ref_q_prev_ids.resize(h_joint_id_to_ref_q_prev_ids.size());
    if(!h_joint_id_to_ref_q_prev_ids.empty())
    {
        std::vector<LVector2i> staging;
        staging.reserve(h_joint_id_to_ref_q_prev_ids.size());
        for(auto& v : h_joint_id_to_ref_q_prev_ids)
            staging.push_back(to_luisa(v));
        joint_id_to_ref_q_prev_ids.view().copy_from(span<const LVector2i>{staging});
    }

    // PORT ADDITION: padded to at least one element (the kernels read the *clamped*
    // index of the discarded select candidate even when no joint has a
    // `ref_dof_prev`; a zero-length buffer cannot be a kernel argument).
    {
        std::vector<LVector12> staging;
        staging.reserve(h_ref_q_prevs.size());
        for(auto& v : h_ref_q_prevs)
            staging.push_back(to_luisa(v));
        ref_q_prevs.resize(std::max<SizeT>(1u, staging.size()));
        if(!staging.empty())
        {
            ref_q_prevs.view().subview(0, staging.size()).copy_from(
                span<const LVector12>{staging});
        }
    }
}

void ExternalArticulationConstraint::do_step(InterAffineBodyAnimator::FilteredInfo& info)
{
    // only collect mass and delta_theta_tilde
    (void)info;

    // Update Joint Mass
    h_joint_joint_id_to_mass.clear();
    for(auto&& mass : h_art_joint_joint_mass)
    {
        auto mass_view = mass->view();
        std::ranges::copy(mass_view, std::back_inserter(h_joint_joint_id_to_mass));
    }

    // Update Predicted Delta Theta
    h_joint_id_to_delta_theta_tilde.clear();
    for(auto&& delta_theta_tilde : h_art_joint_delta_theta_tilde)
    {
        auto delta_theta_tilde_view = delta_theta_tilde->view();
        std::ranges::copy(delta_theta_tilde_view,
                          std::back_inserter(h_joint_id_to_delta_theta_tilde));
    }

    // Update Ref Dof Prev
    std::ranges::transform(h_attr_ref_q_prevs,
                           h_ref_q_prevs.begin(),
                           [&](const std::tuple<AttrRefQPrev, IndexT>& tuple) -> Vector12
                           {
                               auto&& [attr, inst_id] = tuple;
                               auto view              = attr->view();
                               return view[inst_id];
                           });

    // upload to device (the reference's `async_copy`; the LC `copy_from` of a
    // zero-size span is not a no-op -- the T6.3b1 port-necessity note).
    if(!h_joint_joint_id_to_mass.empty())
    {
        joint_joint_id_to_mass.view().copy_from(span<const Float>{h_joint_joint_id_to_mass});
    }
    if(!h_joint_id_to_delta_theta_tilde.empty())
    {
        joint_id_to_delta_theta_tilde.view().copy_from(
            span<const Float>{h_joint_id_to_delta_theta_tilde});
    }
    if(!h_ref_q_prevs.empty())
    {
        std::vector<LVector12> staging;
        staging.reserve(h_ref_q_prevs.size());
        for(auto& v : h_ref_q_prevs)
            staging.push_back(to_luisa(v));
        ref_q_prevs.view().subview(0, staging.size()).copy_from(
            span<const LVector12>{staging});
    }
}

void ExternalArticulationConstraint::do_report_extent(
    InterAffineBodyAnimator::ReportExtentInfo& info)
{
    auto e_count = static_cast<SizeT>(joint_joint_id_to_mass.size());
    info.energy_count(e_count);

    auto g_count = static_cast<SizeT>(joint_id_to_art_id.size()) * 2;
    info.gradient_count(g_count);

    if(info.gradient_only())
        return;

    auto h_count = static_cast<SizeT>(joint_joint_id_to_mass.size()) * HalfHessianSize;
    info.hessian_count(h_count);
}

void ExternalArticulationConstraint::do_compute_energy(
    InterAffineBodyAnimator::ComputeEnergyInfo& info)
{
    int n = static_cast<int>(joint_joint_id_to_mass.size());
    if(n > 0)
    {
        auto& k = external_articulation_energy_kernel();
        frame_stream().record(
            k(info.energies(),
              joint_joint_id_to_mass.cview(),
              joint_joint_id_to_joint_ij.cview(),
              joint_id_to_delta_theta_tilde.cview(),
              joint_id_to_body_ids.cview(),
              joint_id_to_L_basis.cview(),
              joint_id_to_R_basis.cview(),
              joint_id_to_uid.cview(),
              info.qs(),
              info.q_prevs(),
              joint_id_to_ref_q_prev_ids.cview(),
              ref_q_prevs.cview(),
              PrismaticJointConstitutionUID,
              RevoluteJointConstitutionUID,
              static_cast<uint32_t>(n))
                .dispatch(n));
    }
}

void ExternalArticulationConstraint::do_compute_gradient_hessian(
    InterAffineBodyAnimator::GradientHessianInfo& info)
{
    // Compute G^theta
    const auto n_gradient = static_cast<uint32_t>(joint_id_to_G_theta.size());
    const auto n_hessian  = static_cast<uint32_t>(joint_joint_id_to_mass.size());
    if(n_gradient > 0 && n_hessian > 0)
    {
        // PORT ADDITION: the reference's `FastSegmentalReduce` computes its value with
        // an in-kernel functor; `lla::segmented_reduce` needs it in a buffer.
        {
            auto& k = external_articulation_reduce_value_kernel();
            frame_stream().record(
                k(joint_joint_id_to_value.view(),
                  joint_joint_id_to_mass.cview(),
                  joint_joint_id_to_joint_ij.cview(),
                  joint_id_to_delta_theta_tilde.cview(),
                  joint_id_to_body_ids.cview(),
                  joint_id_to_L_basis.cview(),
                  joint_id_to_R_basis.cview(),
                  joint_id_to_uid.cview(),
                  info.qs(),
                  info.q_prevs(),
                  joint_id_to_ref_q_prev_ids.cview(),
                  ref_q_prevs.cview(),
                  RevoluteJointConstitutionUID,
                  PrismaticJointConstitutionUID,
                  static_cast<uint32_t>(n_hessian))
                    .dispatch(n_hessian));
        }
        lla::segmented_reduce(frame_stream(),
                              joint_joint_id_to_key.cview(),
                              joint_joint_id_to_value.cview(),
                              joint_id_to_G_theta.view(),
                              static_cast<size_t>(n_gradient),
                              static_cast<size_t>(n_hessian));
    }

    // Compute Gradient
    if(n_gradient > 0)
    {
        auto gradients = info.gradients();
        auto& k        = external_articulation_gradient_hessian_k1_kernel();
        frame_stream().record(
            k(joint_id_to_G_theta.cview(),
              joint_id_to_body_ids.cview(),
              joint_id_to_L_basis.cview(),
              joint_id_to_R_basis.cview(),
              info.qs(),
              info.q_prevs(),
              joint_id_to_ref_q_prev_ids.cview(),
              ref_q_prevs.cview(),
              joint_id_to_uid.cview(),
              RevoluteJointConstitutionUID,
              PrismaticJointConstitutionUID,
              gradients.indices(),
              gradients.values(),
              static_cast<uint32_t>(gradients.total_extent()),
              static_cast<std::int32_t>(gradients.doublet_index_offset()),
              static_cast<std::int32_t>(gradients.doublet_count()),
              static_cast<std::int32_t>(gradients.total_doublet_count()),
              static_cast<uint32_t>(n_gradient))
                .dispatch(n_gradient));
    }

    if(info.gradient_only())
        return;

    // Compute Hessian
    if(n_hessian > 0)
    {
        auto hessians = info.hessians();
        auto& k       = external_articulation_gradient_hessian_k2_kernel();
        frame_stream().record(
            k(joint_id_to_G_theta.cview(),
              joint_id_to_body_ids.cview(),
              joint_joint_id_to_mass.cview(),
              joint_joint_id_to_joint_ij.cview(),
              joint_id_to_L_basis.cview(),
              joint_id_to_R_basis.cview(),
              info.qs(),
              info.q_prevs(),
              joint_id_to_ref_q_prev_ids.cview(),
              ref_q_prevs.cview(),
              joint_id_to_uid.cview(),
              RevoluteJointConstitutionUID,
              PrismaticJointConstitutionUID,
              static_cast<uint32_t>(HalfHessianSize),
              hessians.row_indices(),
              hessians.col_indices(),
              hessians.values(),
              static_cast<std::int32_t>(hessians.total_rows()),
              static_cast<std::int32_t>(hessians.total_cols()),
              static_cast<std::int32_t>(hessians.triplet_index_offset()),
              static_cast<std::int32_t>(hessians.triplet_count()),
              static_cast<std::int32_t>(hessians.total_triplet_count()),
              static_cast<std::int32_t>(hessians.submatrix_offset().x),
              static_cast<std::int32_t>(hessians.submatrix_offset().y),
              static_cast<std::int32_t>(hessians.extent().x),
              static_cast<std::int32_t>(hessians.extent().y),
              static_cast<uint32_t>(n_hessian))
                .dispatch(n_hessian));
    }
}

void ExternalArticulationConstraint::write_scene()
{
    auto geo_slots = world().scene().geometries();

    // `copy_to` (the reference's blocking device->host copy); the frame stream must be
    // flushed first (the T6.3b2 `AffineBodyPrismaticJoint::write_scene` precedent,
    // §7.9 rule 7).
    frame_stream().synchronize();
    if(!h_joint_id_to_delta_theta.empty())
    {
        joint_id_to_delta_theta.cview().copy_to(
            span<Float>{h_joint_id_to_delta_theta});
    }
    lud::LStream::current().synchronize();

    this->for_each(
        geo_slots,
        [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
        {
            auto joint_collection = geo["joint"];
            auto delta_theta      = joint_collection->find<Float>("delta_theta");

            if(delta_theta)
            {
                auto delta_theta_view = view(*delta_theta);
                auto [offset, count]  = h_art_id_to_joint_offsets_counts[I.index()];
                std::ranges::copy(
                    span{h_joint_id_to_delta_theta}.subspan(offset, count),
                    delta_theta_view.begin());
            }
        });
}

bool ExternalArticulationConstraint::do_dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_delta_theta.dump(fmt::format("{}delta_theta.{}", path, frame),
                                 joint_id_to_delta_theta);
}

bool ExternalArticulationConstraint::do_try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_delta_theta.load(fmt::format("{}delta_theta.{}", path, frame));
}

void ExternalArticulationConstraint::do_apply_recover(RecoverInfo& info)
{
    (void)info;
    dump_delta_theta.apply_to(joint_id_to_delta_theta);
}

void ExternalArticulationConstraint::do_clear_recover(RecoverInfo& info)
{
    (void)info;
    dump_delta_theta.clean_up();
}

// ===========================================================================
//  ExternalArticulationConstraintTimeIntegrator
// ===========================================================================
REGISTER_SIM_SYSTEM(ExternalArticulationConstraintTimeIntegrator);

void ExternalArticulationConstraintTimeIntegrator::do_build(BuildInfo& info)
{
    (void)info;
    constraint           = require<ExternalArticulationConstraint>();
    affine_body_dynamics = require<AffineBodyDynamics>();
}

void ExternalArticulationConstraintTimeIntegrator::do_init(InitInfo& info)
{
    (void)info;
}

void ExternalArticulationConstraintTimeIntegrator::do_predict_dof(PredictDofInfo& info)
{
    (void)info;
}

void ExternalArticulationConstraintTimeIntegrator::do_update_state(UpdateVelocityInfo& info)
{
    (void)info;
    auto& c   = constraint;
    auto& abd = affine_body_dynamics;

    int n = static_cast<int>(c->joint_id_to_delta_theta.size());
    if(n > 0)
    {
        auto& k = external_articulation_update_state_kernel();
        frame_stream().record(
            k(c->joint_id_to_delta_theta.view(),
              c->joint_id_to_body_ids.cview(),
              c->joint_id_to_L_basis.cview(),
              c->joint_id_to_R_basis.cview(),
              abd->qs(),
              abd->q_prevs(),
              c->joint_id_to_ref_q_prev_ids.cview(),
              c->ref_q_prevs.cview(),
              c->joint_id_to_uid.cview(),
              ExternalArticulationConstraint::RevoluteJointConstitutionUID,
              ExternalArticulationConstraint::PrismaticJointConstitutionUID,
              static_cast<uint32_t>(n))
                .dispatch(n));
    }
}
}  // namespace uipc::backend::luisa
