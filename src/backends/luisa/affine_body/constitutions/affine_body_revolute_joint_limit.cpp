/********************************************************************
 * @file affine_body/constitutions/affine_body_revolute_joint_limit.cpp
 * @brief T6.3b3 — the revolute joint-limit constitution: its three kernels, the
 *        out-of-line members and the system registration (reference:
 *        src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu,
 *        class `AffineBodyRevoluteJointLimit`).
 *
 * The class is declared in `affine_body_revolute_joint_limit.h` (the file-split
 * deviation, reported there).
 *
 * THE ONE DEVICE PREDICATE, and why it is a `$if` and not a select: the k2 kernel's
 * `if(dE_dx != 0.0f) { ... H += JT_H_J; }` guards a *statement* (an accumulating
 * `H += ...`), not a value.  `$if` is the DSL's device branch and the reference's
 * guard *is* a device branch, so the port transcribes it as `$if(dE_dx != 0.0)` — the
 * statement is emitted only when the predicate holds, which is exactly the reference's
 * control flow (and cheaper than the branchless spelling, whose second term would be
 * all-zero anyway).  The *host-callable* helper
 * `abd_revolute_joint_limit_detail::hessian` is the branchless twin (that is what the
 * unit test and the device round drive).
 *
 * THE ANGLE IS *UNWRAPPED ACROSS FRAMES*: the kernels read `current_angles` as the
 * frame-start angle and add the generated `DeltaTheta` between the current and the
 * previous state, so `x` is the joint's multi-turn angle and the limit keeps working
 * past +-pi and across turns (the reference's own comment on the class).
 ********************************************************************/
#include <affine_body/constitutions/affine_body_revolute_joint_limit.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace
{
// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `affine_body_revolute_joint_limit_compute_energy_kernel`
auto& revolute_joint_limit_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // l_basis
                                                lcrt::Buffer<LVector6>,   // r_basis
                                                lcrt::Buffer<double>,     // init_angles
                                                lcrt::Buffer<double>,     // current_angles
                                                lcrt::Buffer<double>,     // lowers
                                                lcrt::Buffer<double>,     // uppers
                                                lcrt::Buffer<double>,     // strengths
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<double>,     // Es
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector2i> body_ids,
           lcrt::BufferVar<LVector6>  l_basis,
           lcrt::BufferVar<LVector6>  r_basis,
           lcrt::BufferVar<double>    init_angles,
           lcrt::BufferVar<double>    current_angles,
           lcrt::BufferVar<double>    lowers,
           lcrt::BufferVar<double>    uppers,
           lcrt::BufferVar<double>    strengths,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::BufferVar<double>    Es,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bid = load_vec2i(body_ids->read(I));

                auto lb = load_vec6(l_basis->read(I));
                auto rb = load_vec6(r_basis->read(I));

                auto qk      = load_vec12(qs->read(bid(0)));
                auto ql      = load_vec12(qs->read(bid(1)));
                auto q_prevk = load_vec12(q_prevs->read(bid(0)));
                auto q_prevl = load_vec12(q_prevs->read(bid(1)));

                auto init_a = init_angles->read(I);

                // Unwrapped relative angle at the last committed state; valid past
                // +-pi and across turns.
                auto theta_prev = lps::detail::sub(current_angles->read(I), init_a);

                auto delta = abd_revolute_joint_limit_detail::delta_theta<lcrt::Var<double>>(
                    lb, qk, q_prevk, rb, ql, q_prevl);

                auto x        = lps::detail::add(theta_prev, delta);
                auto lower    = lps::detail::sub(lowers->read(I), init_a);
                auto upper    = lps::detail::sub(uppers->read(I), init_a);
                auto strength = strengths->read(I);

                auto E = joint_limit::eval_penalty_energy<lcrt::Var<double>>(x, lower, upper,
                                                                             strength);

                Es->write(I, E);
            };
        });
    return k;
}

/// `affine_body_revolute_joint_limit_compute_gradient_hessian_k1_kernel`
auto& revolute_joint_limit_gradient_hessian_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // l_basis
                                                lcrt::Buffer<LVector6>,   // r_basis
                                                lcrt::Buffer<double>,     // init_angles
                                                lcrt::Buffer<double>,     // current_angles
                                                lcrt::Buffer<double>,     // lowers
                                                lcrt::Buffer<double>,     // uppers
                                                lcrt::Buffer<double>,     // strengths
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<int32_t>,    // G indices
                                                lcrt::Buffer<double>,     // G values
                                                uint32_t,  // G total segments
                                                int32_t,   // G doublet index offset
                                                int32_t,   // G doublet count
                                                int32_t,   // G total doublet count
                                                uint32_t>(  // n
        [](lcrt::BufferVar<LVector2i> body_ids,
           lcrt::BufferVar<LVector6>  l_basis,
           lcrt::BufferVar<LVector6>  r_basis,
           lcrt::BufferVar<double>    init_angles,
           lcrt::BufferVar<double>    current_angles,
           lcrt::BufferVar<double>    lowers,
           lcrt::BufferVar<double>    uppers,
           lcrt::BufferVar<double>    strengths,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::BufferVar<int32_t>   G_indices,
           lcrt::BufferVar<double>    G_values,
           lcrt::Var<uint32_t>        G_total_segments,
           lcrt::Var<int32_t>         G_doublet_index_offset,
           lcrt::Var<int32_t>         G_doublet_count,
           lcrt::Var<int32_t>         G_total_doublet_count,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bid = load_vec2i(body_ids->read(I));

                auto lb = load_vec6(l_basis->read(I));
                auto rb = load_vec6(r_basis->read(I));

                auto qk      = load_vec12(qs->read(bid(0)));
                auto ql      = load_vec12(qs->read(bid(1)));
                auto q_prevk = load_vec12(q_prevs->read(bid(0)));
                auto q_prevl = load_vec12(q_prevs->read(bid(1)));

                auto init_a = init_angles->read(I);

                // Same unwrapped frame-start angle as do_compute_energy.
                auto theta_prev = lps::detail::sub(current_angles->read(I), init_a);

                auto delta = abd_revolute_joint_limit_detail::delta_theta<lcrt::Var<double>>(
                    lb, qk, q_prevk, rb, ql, q_prevl);

                auto x        = lps::detail::add(theta_prev, delta);
                auto lower    = lps::detail::sub(lowers->read(I), init_a);
                auto upper    = lps::detail::sub(uppers->read(I), init_a);
                auto strength = strengths->read(I);

                // Float dE_dx = 0.0f; Float d2E_dx2 = 0.0f;
                // joint_limit::eval_penalty_derivatives<Float>(x, lower, upper, strength,
                //                                             dE_dx, d2E_dx2);
                lcrt::Var<double> dE_dx{0.0};
                lcrt::Var<double> d2E_dx2{0.0};
                joint_limit::eval_penalty_derivatives<lcrt::Var<double>>(
                    x, lower, upper, strength, dE_dx, d2E_dx2);

                auto dx_dq = abd_revolute_joint_limit_detail::d_delta_theta_dq<lcrt::Var<double>>(
                    lb, qk, q_prevk, rb, ql, q_prevl);

                auto G = abd_revolute_joint_limit_detail::gradient<lcrt::Var<double>>(dE_dx,
                                                                                     dx_dq);

                auto ii = lcrt::cast<std::int32_t>(I);

                auto D = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 12>::from_handles_full(
                    G_indices,
                    G_values,
                    G_total_segments,
                    G_doublet_index_offset,
                    G_doublet_count,
                    G_total_doublet_count,
                    std::int32_t{0},
                    G_total_segments);
                abd_revolute_joint_detail::write_pair_gradient(D, G, ii * 2, bid);
            };
        });
    return k;
}

/// `affine_body_revolute_joint_limit_compute_gradient_hessian_k2_kernel`
auto& revolute_joint_limit_gradient_hessian_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // l_basis
                                                lcrt::Buffer<LVector6>,   // r_basis
                                                lcrt::Buffer<double>,     // init_angles
                                                lcrt::Buffer<double>,     // current_angles
                                                lcrt::Buffer<double>,     // lowers
                                                lcrt::Buffer<double>,     // uppers
                                                lcrt::Buffer<double>,     // strengths
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<int32_t>,  // H row indices
                                                lcrt::Buffer<int32_t>,  // H col indices
                                                lcrt::Buffer<double>,   // H values
                                                int32_t,                // H total block rows
                                                int32_t,                // H total block cols
                                                int32_t,  // H triplet index offset
                                                int32_t,  // H triplet count
                                                int32_t,  // H total triplet count
                                                int32_t,  // H submatrix offset x
                                                int32_t,  // H submatrix offset y
                                                int32_t,  // H submatrix extent x
                                                int32_t,  // H submatrix extent y
                                                uint32_t,  // HalfHessianSize
                                                uint32_t>(  // n
        [](lcrt::BufferVar<LVector2i> body_ids,
           lcrt::BufferVar<LVector6>  l_basis,
           lcrt::BufferVar<LVector6>  r_basis,
           lcrt::BufferVar<double>    init_angles,
           lcrt::BufferVar<double>    current_angles,
           lcrt::BufferVar<double>    lowers,
           lcrt::BufferVar<double>    uppers,
           lcrt::BufferVar<double>    strengths,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::BufferVar<int32_t>   H_row_indices,
           lcrt::BufferVar<int32_t>   H_col_indices,
           lcrt::BufferVar<double>    H_values,
           lcrt::Var<int32_t>         H_total_rows,
           lcrt::Var<int32_t>         H_total_cols,
           lcrt::Var<int32_t>         H_triplet_index_offset,
           lcrt::Var<int32_t>         H_triplet_count,
           lcrt::Var<int32_t>         H_total_triplet_count,
           lcrt::Var<int32_t>         H_sm_offset_x,
           lcrt::Var<int32_t>         H_sm_offset_y,
           lcrt::Var<int32_t>         H_sm_extent_x,
           lcrt::Var<int32_t>         H_sm_extent_y,
           lcrt::Var<uint32_t>        HalfHessianSize,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bid = load_vec2i(body_ids->read(I));

                auto lb = load_vec6(l_basis->read(I));
                auto rb = load_vec6(r_basis->read(I));

                auto qk      = load_vec12(qs->read(bid(0)));
                auto ql      = load_vec12(qs->read(bid(1)));
                auto q_prevk = load_vec12(q_prevs->read(bid(0)));
                auto q_prevl = load_vec12(q_prevs->read(bid(1)));

                auto init_a = init_angles->read(I);

                // Same unwrapped frame-start angle as do_compute_energy.
                auto theta_prev = lps::detail::sub(current_angles->read(I), init_a);

                auto delta = abd_revolute_joint_limit_detail::delta_theta<lcrt::Var<double>>(
                    lb, qk, q_prevk, rb, ql, q_prevl);

                auto x        = lps::detail::add(theta_prev, delta);
                auto lower    = lps::detail::sub(lowers->read(I), init_a);
                auto upper    = lps::detail::sub(uppers->read(I), init_a);
                auto strength = strengths->read(I);

                lcrt::Var<double> dE_dx{0.0};
                lcrt::Var<double> d2E_dx2{0.0};
                joint_limit::eval_penalty_derivatives<lcrt::Var<double>>(
                    x, lower, upper, strength, dE_dx, d2E_dx2);

                auto dx_dq = abd_revolute_joint_limit_detail::d_delta_theta_dq<lcrt::Var<double>>(
                    lb, qk, q_prevk, rb, ql, q_prevl);

                // Matrix24x24 H = d2E_dx2 * (dx_dq * dx_dq.transpose());
                // if(dE_dx != 0.0f) { ... H += JT_H_J; }
                lps::LMat<lcrt::Var<double>, 24, 24> H = d2E_dx2 * (dx_dq * dx_dq.transpose());
                $if(dE_dx != 0.0)
                {
                    auto F = abd_revolute_joint_limit_detail::pair_f<lcrt::Var<double>>(
                        lb, qk, rb, ql);
                    auto F_prev = abd_revolute_joint_limit_detail::pair_f<lcrt::Var<double>>(
                        lb, q_prevk, rb, q_prevl);

                    auto ddx_ddF =
                        abd_revolute_joint_limit_detail::dd_delta_theta_ddF<lcrt::Var<double>>(
                            F, F_prev);

                    lps::LMat<lcrt::Var<double>, 12, 12> H_F = dE_dx * ddx_ddF;
                    lps::make_spd_inplace(
                        H_F, abd_driving_revolute_joint_detail::kMakeSpdSweeps);

                    H += abd_revolute_joint_limit_detail::jt_h_j<lcrt::Var<double>>(H_F, lb, rb);
                };

                auto ii = lcrt::cast<std::int32_t>(I);

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
                abd_revolute_joint_detail::write_pair_half_hessian(
                    T, H, ii * lcrt::cast<std::int32_t>(HalfHessianSize), bid);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyRevoluteJointLimit);

U64 AffineBodyRevoluteJointLimit::get_uid() const noexcept
{
    return ConstitutionUID;
}

void AffineBodyRevoluteJointLimit::do_build(BuildInfo& info)
{
    (void)info;
    revolute_joint = require<AffineBodyRevoluteJoint>();
}

void AffineBodyRevoluteJointLimit::do_init(FilteredInfo& info)
{
    auto geo_slots = world().scene().geometries();

    h_lowers.clear();
    h_uppers.clear();
    h_strengths.clear();

    info.for_each(
        geo_slots,
        [&](geometry::Geometry& geo)
        {
            auto uid = geo.meta().find<U64>(builtin::constitution_uid);
            UIPC_ASSERT(uid && uid->view()[0] == JointUID,
                        "AffineBodyRevoluteJointLimit must be attached on base revolute joint geometry (UID={})",
                        JointUID);

            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc, "AffineBodyRevoluteJointLimit geometry must be SimplicialComplex");

            auto lower_attr = sc->edges().find<Float>("limit/lower");
            UIPC_ASSERT(lower_attr,
                        "AffineBodyRevoluteJointLimit requires `limit/lower` attribute on edges");
            auto lower_view = lower_attr->view();

            auto upper_attr = sc->edges().find<Float>("limit/upper");
            UIPC_ASSERT(upper_attr,
                        "AffineBodyRevoluteJointLimit requires `limit/upper` attribute on edges");
            auto upper_view = upper_attr->view();

            auto strength_attr = sc->edges().find<Float>("limit/strength");
            UIPC_ASSERT(strength_attr,
                        "AffineBodyRevoluteJointLimit requires `limit/strength` attribute on edges");
            auto strength_view = strength_attr->view();

            for(SizeT i = 0; i < sc->edges().size(); ++i)
            {
                UIPC_ASSERT(lower_view[i] <= upper_view[i],
                            "AffineBodyRevoluteJointLimit: requires `limit/lower <= limit/upper` on edge {}, but got lower={} upper={}",
                            i,
                            lower_view[i],
                            upper_view[i]);
                h_lowers.push_back(lower_view[i]);
                h_uppers.push_back(upper_view[i]);
                h_strengths.push_back(strength_view[i]);
            }
        });

    // Sanity check: limit constitution reuses body_ids/l_basis/r_basis/init_angles/
    // current_angles from AffineBodyRevoluteJoint via the shared index `I`. Both must
    // therefore enumerate exactly the same (geo, edge) sequence.
    UIPC_ASSERT(h_lowers.size() == revolute_joint->h_body_ids.size(),
                "AffineBodyRevoluteJointLimit: joint count {} must equal "
                "AffineBodyRevoluteJoint joint count {} (index alignment required)",
                h_lowers.size(),
                revolute_joint->h_body_ids.size());

    lowers.resize(h_lowers.size());
    if(!h_lowers.empty())
        lowers.view().copy_from(span<const Float>{h_lowers});

    uppers.resize(h_uppers.size());
    if(!h_uppers.empty())
        uppers.view().copy_from(span<const Float>{h_uppers});

    strengths.resize(h_strengths.size());
    if(!h_strengths.empty())
        strengths.view().copy_from(span<const Float>{h_strengths});
}

void AffineBodyRevoluteJointLimit::do_report_energy_extent(EnergyExtentInfo& info)
{
    info.energy_count(lowers.size());
}

void AffineBodyRevoluteJointLimit::do_compute_energy(ComputeEnergyInfo& info)
{
    int n = (int)lowers.size();
    if(n > 0)
    {
        auto& k = revolute_joint_limit_energy_kernel();
        frame_stream().record(k(revolute_joint->body_ids.cview(),
                                revolute_joint->l_basis.cview(),
                                revolute_joint->r_basis.cview(),
                                revolute_joint->init_angles.cview(),
                                revolute_joint->current_angles.cview(),
                                lowers.cview(),
                                uppers.cview(),
                                strengths.cview(),
                                info.qs(),
                                info.q_prevs(),
                                info.energies(),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}

void AffineBodyRevoluteJointLimit::do_report_gradient_hessian_extent(
    GradientHessianExtentInfo& info)
{
    info.gradient_count(2 * lowers.size());
    if(info.gradient_only())
        return;

    info.hessian_count(HalfHessianSize * lowers.size());
}

void AffineBodyRevoluteJointLimit::do_compute_gradient_hessian(
    ComputeGradientHessianInfo& info)
{
    int n = (int)lowers.size();
    if(n == 0)
        return;

    auto gradients = info.gradients();
    auto hessians  = info.hessians();

    {
        auto& k = revolute_joint_limit_gradient_hessian_k1_kernel();
        frame_stream().record(k(revolute_joint->body_ids.cview(),
                                revolute_joint->l_basis.cview(),
                                revolute_joint->r_basis.cview(),
                                revolute_joint->init_angles.cview(),
                                revolute_joint->current_angles.cview(),
                                lowers.cview(),
                                uppers.cview(),
                                strengths.cview(),
                                info.qs(),
                                info.q_prevs(),
                                gradients.indices(),
                                gradients.values(),
                                static_cast<uint32_t>(gradients.total_extent()),
                                static_cast<std::int32_t>(gradients.doublet_index_offset()),
                                static_cast<std::int32_t>(gradients.doublet_count()),
                                static_cast<std::int32_t>(gradients.total_doublet_count()),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }

    if(info.gradient_only())
        return;

    {
        auto& k = revolute_joint_limit_gradient_hessian_k2_kernel();
        frame_stream().record(k(revolute_joint->body_ids.cview(),
                                revolute_joint->l_basis.cview(),
                                revolute_joint->r_basis.cview(),
                                revolute_joint->init_angles.cview(),
                                revolute_joint->current_angles.cview(),
                                lowers.cview(),
                                uppers.cview(),
                                strengths.cview(),
                                info.qs(),
                                info.q_prevs(),
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
                                static_cast<uint32_t>(HalfHessianSize),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}
}  // namespace uipc::backend::luisa
