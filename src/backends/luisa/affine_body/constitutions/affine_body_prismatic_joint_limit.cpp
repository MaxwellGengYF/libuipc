/********************************************************************
 * @file affine_body/constitutions/affine_body_prismatic_joint_limit.cpp
 * @brief T6.3b2 — the prismatic joint-limit constitution: its three kernels, the
 *        out-of-line members and the system registration (reference:
 *        src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu,
 *        class `AffineBodyPrismaticJointLimit`).
 *
 * The class is declared in `affine_body_prismatic_joint_limit.h` (the file-split
 * deviation, reported there).
 *
 * THE ONE DEVICE PREDICATE, and why it is NOT a select this time: the k2 kernel's
 * `if(dE_dx != 0.0f) compute_absolute_distance_hessian(H, ...)` guards a *statement*
 * (an accumulating `out_H += ...`), not a value.  `$if` is the DSL's device branch
 * and the reference's guard *is* a device branch, so the port transcribes it as
 * `$if(dE_dx != 0.0)` — the statement is emitted only when the predicate holds, which
 * is exactly the reference's control flow (and cheaper than the branchless spelling,
 * whose second term would be all-zero anyway).
 ********************************************************************/
#include <affine_body/constitutions/affine_body_prismatic_joint_limit.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <affine_body/utils.h>  // transform_to_q (the reference's `affine_body/utils.h`)
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace
{
// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `affine_body_prismatic_joint_limit_compute_energy_kernel`
auto& prismatic_joint_limit_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<LVector6>,   // rest_ts
                                                lcrt::Buffer<double>,     // init_distances
                                                lcrt::Buffer<double>,     // lowers
                                                lcrt::Buffer<double>,     // uppers
                                                lcrt::Buffer<double>,     // strengths
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<double>,     // Es
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector2i> body_ids,
           lcrt::BufferVar<LVector6>  rest_cs,
           lcrt::BufferVar<LVector6>  rest_ts,
           lcrt::BufferVar<double>    init_distances,
           lcrt::BufferVar<double>    lowers,
           lcrt::BufferVar<double>    uppers,
           lcrt::BufferVar<double>    strengths,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<double>    Es,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bid = load_vec2i(body_ids->read(I));

                auto C_bar = load_vec6(rest_cs->read(I));
                auto t_bar = load_vec6(rest_ts->read(I));
                auto qk    = load_vec12(qs->read(bid(0)));
                auto ql    = load_vec12(qs->read(bid(1)));

                auto x = abd_driving_prismatic_joint_detail::compute_absolute_distance<
                    lcrt::Var<double>>(C_bar, t_bar, qk, ql);

                auto init_d   = init_distances->read(I);
                auto lower    = lps::detail::sub(lowers->read(I), init_d);
                auto upper    = lps::detail::sub(uppers->read(I), init_d);
                auto strength = strengths->read(I);

                auto E = joint_limit::eval_penalty_energy<lcrt::Var<double>>(
                    x, lower, upper, strength);

                Es->write(I, E);
            };
        });
    return k;
}

/// `affine_body_prismatic_joint_limit_compute_gradient_hessian_k1_kernel`
auto& prismatic_joint_limit_gradient_hessian_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<LVector6>,   // rest_ts
                                                lcrt::Buffer<double>,     // init_distances
                                                lcrt::Buffer<double>,     // lowers
                                                lcrt::Buffer<double>,     // uppers
                                                lcrt::Buffer<double>,     // strengths
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<int32_t>,    // G indices
                                                lcrt::Buffer<double>,     // G values
                                                uint32_t,  // G total segments
                                                int32_t,   // G doublet index offset
                                                int32_t,   // G doublet count
                                                int32_t,   // G total doublet count
                                                uint32_t>(  // n
        [](lcrt::BufferVar<LVector2i> body_ids,
           lcrt::BufferVar<LVector6>  rest_cs,
           lcrt::BufferVar<LVector6>  rest_ts,
           lcrt::BufferVar<double>    init_distances,
           lcrt::BufferVar<double>    lowers,
           lcrt::BufferVar<double>    uppers,
           lcrt::BufferVar<double>    strengths,
           lcrt::BufferVar<LVector12> qs,
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

                auto C_bar = load_vec6(rest_cs->read(I));
                auto t_bar = load_vec6(rest_ts->read(I));

                auto qk = load_vec12(qs->read(bid(0)));
                auto ql = load_vec12(qs->read(bid(1)));

                auto x = abd_driving_prismatic_joint_detail::compute_absolute_distance<
                    lcrt::Var<double>>(C_bar, t_bar, qk, ql);

                auto init_d   = init_distances->read(I);
                auto lower    = lps::detail::sub(lowers->read(I), init_d);
                auto upper    = lps::detail::sub(uppers->read(I), init_d);
                auto strength = strengths->read(I);

                // Float dE_dx = 0.0f; Float d2E_dx2 = 0.0f;
                // joint_limit::eval_penalty_derivatives<Float>(x, lower, upper, strength,
                //                                             dE_dx, d2E_dx2);
                lcrt::Var<double> dE_dx{0.0};
                lcrt::Var<double> d2E_dx2{0.0};
                joint_limit::eval_penalty_derivatives<lcrt::Var<double>>(
                    x, lower, upper, strength, dE_dx, d2E_dx2);

                auto dx_dq = abd_driving_prismatic_joint_detail::
                    compute_absolute_distance_derivative<lcrt::Var<double>>(
                        C_bar, t_bar, qk, ql);

                auto G = abd_prismatic_joint_limit_detail::gradient<lcrt::Var<double>>(
                    dE_dx, dx_dq);

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
                abd_prismatic_joint_detail::write_pair_gradient(D, G, ii * 2, bid);
            };
        });
    return k;
}

/// `affine_body_prismatic_joint_limit_compute_gradient_hessian_k2_kernel`
auto& prismatic_joint_limit_gradient_hessian_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<LVector6>,   // rest_ts
                                                lcrt::Buffer<double>,     // init_distances
                                                lcrt::Buffer<double>,     // lowers
                                                lcrt::Buffer<double>,     // uppers
                                                lcrt::Buffer<double>,     // strengths
                                                lcrt::Buffer<LVector12>,  // qs
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
           lcrt::BufferVar<LVector6>  rest_cs,
           lcrt::BufferVar<LVector6>  rest_ts,
           lcrt::BufferVar<double>    init_distances,
           lcrt::BufferVar<double>    lowers,
           lcrt::BufferVar<double>    uppers,
           lcrt::BufferVar<double>    strengths,
           lcrt::BufferVar<LVector12> qs,
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

                auto C_bar = load_vec6(rest_cs->read(I));
                auto t_bar = load_vec6(rest_ts->read(I));

                auto qk = load_vec12(qs->read(bid(0)));
                auto ql = load_vec12(qs->read(bid(1)));

                auto x = abd_driving_prismatic_joint_detail::compute_absolute_distance<
                    lcrt::Var<double>>(C_bar, t_bar, qk, ql);

                auto init_d   = init_distances->read(I);
                auto lower    = lps::detail::sub(lowers->read(I), init_d);
                auto upper    = lps::detail::sub(uppers->read(I), init_d);
                auto strength = strengths->read(I);

                lcrt::Var<double> dE_dx{0.0};
                lcrt::Var<double> d2E_dx2{0.0};
                joint_limit::eval_penalty_derivatives<lcrt::Var<double>>(
                    x, lower, upper, strength, dE_dx, d2E_dx2);

                auto dx_dq = abd_driving_prismatic_joint_detail::
                    compute_absolute_distance_derivative<lcrt::Var<double>>(
                        C_bar, t_bar, qk, ql);

                // Matrix24x24 H = d2E_dx2 * (dx_dq * dx_dq.transpose());
                // if(dE_dx != 0.0f) { compute_absolute_distance_hessian(H, dE_dx, ...); }
                lps::LMat<lcrt::Var<double>, 24, 24> H =
                    d2E_dx2 * (dx_dq * dx_dq.transpose());
                $if(dE_dx != 0.0)
                {
                    abd_driving_prismatic_joint_detail::compute_absolute_distance_hessian<
                        lcrt::Var<double>>(H, dE_dx, C_bar, t_bar, qk, ql);
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
                abd_prismatic_joint_detail::write_pair_half_hessian(
                    T, H, ii * lcrt::cast<std::int32_t>(HalfHessianSize), bid);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyPrismaticJointLimit);

U64 AffineBodyPrismaticJointLimit::get_uid() const noexcept
{
    return ConstitutionUID;
}

void AffineBodyPrismaticJointLimit::do_build(BuildInfo& info)
{
    (void)info;
    prismatic_joint = require<AffineBodyPrismaticJoint>();
}

void AffineBodyPrismaticJointLimit::do_init(FilteredInfo& info)
{
    auto geo_slots = world().scene().geometries();

    h_ref_qs.clear();
    h_lowers.clear();
    h_uppers.clear();
    h_strengths.clear();

    info.for_each(
        geo_slots,
        [&](geometry::Geometry& geo)
        {
            auto uid = geo.meta().find<U64>(builtin::constitution_uid);
            UIPC_ASSERT(uid && uid->view()[0] == JointUID,
                        "AffineBodyPrismaticJointLimit must be attached on base prismatic joint geometry (UID={})",
                        JointUID);

            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc, "AffineBodyPrismaticJointLimit geometry must be SimplicialComplex");

            auto l_geo_id = sc->edges().find<IndexT>("l_geo_id");
            UIPC_ASSERT(l_geo_id,
                        "AffineBodyPrismaticJointLimit requires `l_geo_id` attribute on edges");
            auto l_geo_id_view = l_geo_id->view();
            auto r_geo_id      = sc->edges().find<IndexT>("r_geo_id");
            UIPC_ASSERT(r_geo_id,
                        "AffineBodyPrismaticJointLimit requires `r_geo_id` attribute on edges");
            auto r_geo_id_view = r_geo_id->view();
            auto l_inst_id     = sc->edges().find<IndexT>("l_inst_id");
            UIPC_ASSERT(l_inst_id,
                        "AffineBodyPrismaticJointLimit requires `l_inst_id` attribute on edges");
            auto l_inst_id_view = l_inst_id->view();
            auto r_inst_id      = sc->edges().find<IndexT>("r_inst_id");
            UIPC_ASSERT(r_inst_id,
                        "AffineBodyPrismaticJointLimit requires `r_inst_id` attribute on edges");
            auto r_inst_id_view = r_inst_id->view();

            auto lower_attr = sc->edges().find<Float>("limit/lower");
            UIPC_ASSERT(lower_attr,
                        "AffineBodyPrismaticJointLimit requires `limit/lower` attribute on edges");
            auto lower_view = lower_attr->view();

            auto upper_attr = sc->edges().find<Float>("limit/upper");
            UIPC_ASSERT(upper_attr,
                        "AffineBodyPrismaticJointLimit requires `limit/upper` attribute on edges");
            auto upper_view = upper_attr->view();

            auto strength_attr = sc->edges().find<Float>("limit/strength");
            UIPC_ASSERT(strength_attr,
                        "AffineBodyPrismaticJointLimit requires `limit/strength` attribute on edges");
            auto strength_view = strength_attr->view();

            auto edges = sc->edges().topo().view();

            for(auto&& [i, e] : enumerate(edges))
            {
                (void)e;
                IndexT l_gid = l_geo_id_view[i];
                IndexT r_gid = r_geo_id_view[i];
                IndexT l_iid = l_inst_id_view[i];
                IndexT r_iid = r_inst_id_view[i];

                auto* left_sc  = info.body_geo(geo_slots, l_gid);
                auto* right_sc = info.body_geo(geo_slots, r_gid);

                UIPC_ASSERT(l_iid >= 0
                                && l_iid
                                       < static_cast<IndexT>(left_sc->instances().size()),
                            "AffineBodyPrismaticJointLimit: left instance ID {} out of range [0, {})",
                            l_iid,
                            left_sc->instances().size());
                UIPC_ASSERT(r_iid >= 0
                                && r_iid
                                       < static_cast<IndexT>(right_sc->instances().size()),
                            "AffineBodyPrismaticJointLimit: right instance ID {} out of range [0, {})",
                            r_iid,
                            right_sc->instances().size());

                Vector24 ref;
                ref.segment<12>(0) = transform_to_q(left_sc->transforms().view()[l_iid]);
                ref.segment<12>(12) = transform_to_q(right_sc->transforms().view()[r_iid]);
                h_ref_qs.push_back(ref);

                UIPC_ASSERT(lower_view[i] <= upper_view[i],
                            "AffineBodyPrismaticJointLimit: requires `limit/lower <= limit/upper` on edge {}, but got lower={} upper={}",
                            i,
                            lower_view[i],
                            upper_view[i]);
                h_lowers.push_back(lower_view[i]);
                h_uppers.push_back(upper_view[i]);
                h_strengths.push_back(strength_view[i]);
            }
        });

    // Sanity check: limit constitution reuses body_ids/rest_cs/rest_ts/init_distances
    // from AffineBodyPrismaticJoint via the shared index `I`. Both must
    // therefore enumerate exactly the same (geo, edge) sequence.
    UIPC_ASSERT(h_ref_qs.size() == prismatic_joint->h_body_ids.size(),
                "AffineBodyPrismaticJointLimit: joint count {} must equal "
                "AffineBodyPrismaticJoint joint count {} (index alignment required)",
                h_ref_qs.size(),
                prismatic_joint->h_body_ids.size());

    // NOTE (the trap documented on `to_luisa(const Vector24&)` in
    // affine_body/type_define.h): `copy_host_to_device` cannot see that overload
    // (its `to_luisa(v)` is a dependent call resolved at its own definition point,
    // where every *earlier* overload is viable through Eigen's converting
    // constructor -> C2668), so the pair state is converted explicitly here, where
    // the exact-match overload is visible.
    ref_qs.resize(h_ref_qs.size());
    if(!h_ref_qs.empty())
    {
        std::vector<LVector24> ref_qs_staging;
        ref_qs_staging.reserve(h_ref_qs.size());
        for(const auto& v : h_ref_qs)
            ref_qs_staging.push_back(to_luisa(v));
        ref_qs.view().copy_from(span<const LVector24>{ref_qs_staging});
    }

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

void AffineBodyPrismaticJointLimit::do_report_energy_extent(EnergyExtentInfo& info)
{
    info.energy_count(ref_qs.size());
}

void AffineBodyPrismaticJointLimit::do_compute_energy(ComputeEnergyInfo& info)
{
    int n = (int)ref_qs.size();
    if(n > 0)
    {
        auto& k = prismatic_joint_limit_energy_kernel();
        frame_stream().record(k(prismatic_joint->body_ids.cview(),
                                prismatic_joint->rest_cs.cview(),
                                prismatic_joint->rest_ts.cview(),
                                prismatic_joint->init_distances.cview(),
                                lowers.cview(),
                                uppers.cview(),
                                strengths.cview(),
                                info.qs(),
                                info.energies(),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}

void AffineBodyPrismaticJointLimit::do_report_gradient_hessian_extent(
    GradientHessianExtentInfo& info)
{
    info.gradient_count(2 * ref_qs.size());
    if(info.gradient_only())
        return;

    info.hessian_count(HalfHessianSize * ref_qs.size());
}

void AffineBodyPrismaticJointLimit::do_compute_gradient_hessian(
    ComputeGradientHessianInfo& info)
{
    int n = (int)ref_qs.size();
    if(n == 0)
        return;

    auto gradients = info.gradients();
    auto hessians  = info.hessians();

    {
        auto& k = prismatic_joint_limit_gradient_hessian_k1_kernel();
        frame_stream().record(k(prismatic_joint->body_ids.cview(),
                                prismatic_joint->rest_cs.cview(),
                                prismatic_joint->rest_ts.cview(),
                                prismatic_joint->init_distances.cview(),
                                lowers.cview(),
                                uppers.cview(),
                                strengths.cview(),
                                info.qs(),
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
        auto& k = prismatic_joint_limit_gradient_hessian_k2_kernel();
        frame_stream().record(k(prismatic_joint->body_ids.cview(),
                                prismatic_joint->rest_cs.cview(),
                                prismatic_joint->rest_ts.cview(),
                                prismatic_joint->init_distances.cview(),
                                lowers.cview(),
                                uppers.cview(),
                                strengths.cview(),
                                info.qs(),
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
