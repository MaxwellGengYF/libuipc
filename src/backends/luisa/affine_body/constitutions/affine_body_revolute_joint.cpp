/********************************************************************
 * @file affine_body/constitutions/affine_body_revolute_joint.cpp
 * @brief T6.3b3 — the ABD revolute joint: its four kernels, their cached shaders,
 *        the out-of-line members and the system registration (reference:
 *        src/backends/cuda/affine_body/constitutions/affine_body_revolute_joint.cu,
 *        classes `AffineBodyRevoluteJoint`, `...DofReporter`, `...TimeIntegrator`).
 *
 * The classes are declared in `affine_body_revolute_joint.h` (the file-split
 * deviation, reported there).
 *
 * KERNEL MAPPING (cookbook §7.9/§7.10/§7.13, the T6.3b1/T6.3b2 notes)
 *   * `cuda_tool::CBufferView<T>` -> `lud::LCBufferView<L*>`; the joint's own buffers
 *     are `lud::LDeviceBuffer<...>`, read through `.cview()`.
 *   * the reference's `DoubletVectorAssembler` / `TripletMatrixAssembler` are not
 *     ported: the two write patterns live in
 *     `abd_revolute_joint_detail::{write_pair_gradient, write_pair_half_hessian}`
 *     and this file builds the *device* views from the handles the assembly hands
 *     over (the T6.1b idiom).
 *   * the angle kernel is the one that reads `affine_body_dynamics->qs()` *directly*
 *     (exactly like the reference's `compute_current_angles` and the prismatic
 *     sibling): the other three kernels take the manager's `info.qs()` window.
 *   * the reference's `UIPC_KERNEL_WARN_WITH_LOCATION` (a device *warning*) is routed
 *     through the device-log facade `LC_LOG` (§7.15: it is compiled out unless
 *     `UIPC_LUISA_KERNEL_PRINT` is defined for this TU), inside the reference's own
 *     `$if` predicate — a device predicate cannot drive a host `if` (RULE M2).
 *   * the reference's `UIPC_KERNEL_ASSERT(::isfinite(current_angles(I)), ...)` becomes
 *     `lcrt::device_assert` (LC has a real device assert, §7.16 rule 6); the message
 *     is the literal part (a device formatted string is not available).
 ********************************************************************/
#include <affine_body/constitutions/affine_body_revolute_joint.h>
#include <affine_body/utils.h>  // orthonormal_basis (the reference's `affine_body/utils.h`)
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <luisa_device/linear/linear.h>
// `uipc::Transform` = `Eigen::Transform<Float,3,Eigen::Affine>`: the reference's `.cu`
// gets it through `cuda_tool`; the port needs the definition for `LT.rotation()` /
// `LT.inverse()` (measured C2079 in T6.3b1).
#include <Eigen/Geometry>
#include <numbers>
#include <fmt/format.h>

namespace uipc::backend::luisa
{
namespace
{
// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `affine_body_revolute_joint_compute_current_angles_kernel`
auto& revolute_joint_current_angles_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // l_basis
                                                lcrt::Buffer<LVector6>,   // r_basis
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<double>,     // current_angles
                                                lcrt::Buffer<double>,     // init_angles
                                                double,                   // PI
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector2i> body_ids,
           lcrt::BufferVar<LVector6>  l_basis,
           lcrt::BufferVar<LVector6>  r_basis,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<double>    current_angles,
           lcrt::BufferVar<double>    init_angles,
           lcrt::Var<double>          PI,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bids = load_vec2i(body_ids->read(I));

                auto q_i = load_vec12(qs->read(bids(0)));
                auto q_j = load_vec12(qs->read(bids(1)));

                auto lb = load_vec6(l_basis->read(I));
                auto rb = load_vec6(r_basis->read(I));

                // Float theta; compute_relative_angle(theta, lb, q_i, rb, q_j);
                auto theta = compute_relative_angle<lcrt::Var<double>>(lb, q_i, rb, q_j);

                // Unwrap the (-pi, pi] atan2 angle against the stored multi-turn angle;
                // exact while the per-frame rotation is < pi.
                auto prev_rel = lps::detail::sub(current_angles->read(I), init_angles->read(I));
                auto rel      = unwrap_angle<lcrt::Var<double>>(theta, prev_rel);

                // the reference: `if(::fabs(rel - prev_rel) > 0.9 * PI) { warn }`
                $if(lps::abs(lps::detail::sub(rel, prev_rel))
                    > lps::detail::mul(lps::make_scalar<lcrt::Var<double>>(0.9), PI))
                {
                    LC_LOG(
                        "revolute joint {}: per-frame rotation is close to the pi unwrap "
                        "limit; current_angle may alias onto a wrong turn",
                        I);
                };

                auto angle = lps::detail::add(rel, init_angles->read(I));
                current_angles->write(I, angle);

                // reference: UIPC_KERNEL_ASSERT(::isfinite(current_angles(I)), ...)
                // (`lps` has no `isfinite`; the layer's host/device predicate is
                // `scalar_detail::is_finite_scalar`, cookbook §7.21)
                lcrt::device_assert(scalar_detail::is_finite_scalar(angle),
                                    "revolute joint: current_angle is not finite");
            };
        });
    return k;
}

/// `affine_body_revolute_joint_compute_energy_kernel`
auto& revolute_joint_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector12>,  // rest_positions
                                                lcrt::Buffer<double>,     // strength_ratio
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<double>,     // Es
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector2i>           body_ids,
           lcrt::BufferVar<LVector12>           rest_positions,
           lcrt::BufferVar<double>              strength_ratio,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
           lcrt::BufferVar<LVector12>           qs,
           lcrt::BufferVar<double>              Es,
           lcrt::Var<uint32_t>                  n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bids = load_vec2i(body_ids->read(I));

                auto kappa = lps::detail::mul(
                    strength_ratio->read(I),
                    lps::detail::add(body_masses->read(bids(0)).m_mass,
                                     body_masses->read(bids(1)).m_mass));

                auto qi = load_vec12(qs->read(bids(0)));
                auto qj = load_vec12(qs->read(bids(1)));

                abd_revolute_joint_detail::RevoluteJointRestT<lcrt::Var<double>> rest{
                    .rest_positions = load_vec12(rest_positions->read(I))};

                auto E = abd_revolute_joint_detail::body_energy<lcrt::Var<double>>(kappa, rest,
                                                                                  qi, qj);
                Es->write(I, E);
            };
        });
    return k;
}

/// `affine_body_revolute_joint_compute_gradient_hessian_k1_kernel`
auto& revolute_joint_gradient_hessian_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector12>,  // rest_positions
                                                lcrt::Buffer<double>,     // strength_ratio
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<int32_t>,    // G indices
                                                lcrt::Buffer<double>,     // G values
                                                uint32_t,  // G total segments
                                                int32_t,   // G doublet index offset
                                                int32_t,   // G doublet count
                                                int32_t,   // G total doublet count
                                                uint32_t>(  // n
        [](lcrt::BufferVar<LVector2i>           body_ids,
           lcrt::BufferVar<LVector12>           rest_positions,
           lcrt::BufferVar<double>              strength_ratio,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
           lcrt::BufferVar<LVector12>           qs,
           lcrt::BufferVar<int32_t>             G_indices,
           lcrt::BufferVar<double>              G_values,
           lcrt::Var<uint32_t>                  G_total_segments,
           lcrt::Var<int32_t>                   G_doublet_index_offset,
           lcrt::Var<int32_t>                   G_doublet_count,
           lcrt::Var<int32_t>                   G_total_doublet_count,
           lcrt::Var<uint32_t>                  n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bids = load_vec2i(body_ids->read(I));

                auto kappa = lps::detail::mul(
                    strength_ratio->read(I),
                    lps::detail::add(body_masses->read(bids(0)).m_mass,
                                     body_masses->read(bids(1)).m_mass));

                auto qi = load_vec12(qs->read(bids(0)));
                auto qj = load_vec12(qs->read(bids(1)));

                abd_revolute_joint_detail::RevoluteJointRestT<lcrt::Var<double>> rest{
                    .rest_positions = load_vec12(rest_positions->read(I))};

                auto ii = lcrt::cast<std::int32_t>(I);

                // reference: DoubletVectorAssembler DVA{G12s};
                //            DVA.segment<StencilSize>(StencilSize * I).write(bids, G24);
                auto G = abd_revolute_joint_detail::body_gradient<lcrt::Var<double>>(kappa, rest,
                                                                                     qi, qj);
                auto D = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 12>::from_handles_full(
                    G_indices,
                    G_values,
                    G_total_segments,
                    G_doublet_index_offset,
                    G_doublet_count,
                    G_total_doublet_count,
                    std::int32_t{0},
                    G_total_segments);
                abd_revolute_joint_detail::write_pair_gradient(D, G, ii * 2, bids);
            };
        });
    return k;
}

/// `affine_body_revolute_joint_compute_gradient_hessian_k2_kernel`
auto& revolute_joint_gradient_hessian_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector12>,  // rest_positions
                                                lcrt::Buffer<double>,     // strength_ratio
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<int32_t>,  // H row indices
                                                lcrt::Buffer<int32_t>,  // H col indices
                                                lcrt::Buffer<double>,   // H values
                                                int32_t,  // H total block rows
                                                int32_t,  // H total block cols
                                                int32_t,  // H triplet index offset
                                                int32_t,  // H triplet count
                                                int32_t,  // H total triplet count
                                                int32_t,  // H submatrix offset x
                                                int32_t,  // H submatrix offset y
                                                int32_t,  // H submatrix extent x
                                                int32_t,  // H submatrix extent y
                                                uint32_t,  // HalfHessianSize
                                                uint32_t>(  // n
        [](lcrt::BufferVar<LVector2i>           body_ids,
           lcrt::BufferVar<LVector12>           rest_positions,
           lcrt::BufferVar<double>              strength_ratio,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
           lcrt::BufferVar<LVector12>           qs,
           lcrt::BufferVar<int32_t>             H_row_indices,
           lcrt::BufferVar<int32_t>             H_col_indices,
           lcrt::BufferVar<double>              H_values,
           lcrt::Var<int32_t>                   H_total_rows,
           lcrt::Var<int32_t>                   H_total_cols,
           lcrt::Var<int32_t>                   H_triplet_index_offset,
           lcrt::Var<int32_t>                   H_triplet_count,
           lcrt::Var<int32_t>                   H_total_triplet_count,
           lcrt::Var<int32_t>                   H_sm_offset_x,
           lcrt::Var<int32_t>                   H_sm_offset_y,
           lcrt::Var<int32_t>                   H_sm_extent_x,
           lcrt::Var<int32_t>                   H_sm_extent_y,
           lcrt::Var<uint32_t>                  HalfHessianSize,
           lcrt::Var<uint32_t>                  n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bids = load_vec2i(body_ids->read(I));

                auto kappa = lps::detail::mul(
                    strength_ratio->read(I),
                    lps::detail::add(body_masses->read(bids(0)).m_mass,
                                     body_masses->read(bids(1)).m_mass));

                auto qi = load_vec12(qs->read(bids(0)));
                auto qj = load_vec12(qs->read(bids(1)));

                abd_revolute_joint_detail::RevoluteJointRestT<lcrt::Var<double>> rest{
                    .rest_positions = load_vec12(rest_positions->read(I))};

                auto ii = lcrt::cast<std::int32_t>(I);

                // reference: TripletMatrixAssembler TMA{H12x12s};
                //            TMA.half_block<StencilSize>(HalfHessianSize * I).write(bids, H24);
                auto H = abd_revolute_joint_detail::body_hessian<lcrt::Var<double>>(kappa, rest,
                                                                                    qi, qj);
                auto T =
                    lla::LTripletMatrixDeviceView<lcrt::Var<double>, 12, 12>::from_handles_full(
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
                    T, H, ii * lcrt::cast<std::int32_t>(HalfHessianSize), bids);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyRevoluteJoint);
REGISTER_SIM_SYSTEM(AffineBodyRevoluteJointDofReporter);
REGISTER_SIM_SYSTEM(AffineBodyRevoluteJointTimeIntegrator);

U64 AffineBodyRevoluteJoint::get_uid() const noexcept
{
    return ConstitutionUID;
}

void AffineBodyRevoluteJoint::do_build(BuildInfo& info)
{
    (void)info;
    // the reference's body (`InterAffineBodyConstitution::do_build()` -- the base's
    // own virtual -- does the uid check, the manager resolution and the registration)
    affine_body_dynamics = require<AffineBodyDynamics>();
    on_write_scene([this]() { write_scene(); });
}

void AffineBodyRevoluteJoint::do_init(FilteredInfo& info)
{
    auto geo_slots = world().scene().geometries();

    h_geo_joint_offsets_counts.resize(info.inter_geo_infos().size());

    list<Vector2i> body_ids_list;
    list<Vector12> rest_positions_list;
    list<Float>    strength_ratio_list;
    list<Vector6>  l_basis_list;
    list<Vector6>  r_basis_list;
    list<Float>    init_angles_list;

    IndexT geo_index = 0;
    info.for_each(
        geo_slots,
        [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
        {
            auto uid       = geo.meta().find<U64>(builtin::constitution_uid);
            U64  uid_value = uid->view()[0];
            UIPC_ASSERT(uid_value == ConstitutionUID,
                        "AffineBodyRevoluteJoint: Geometry constitution UID mismatch");

            auto joint_geo_id = I.geo_info().geo_id;

            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc, "AffineBodyRevoluteJoint: Geometry must be a simplicial complex");

            h_geo_joint_offsets_counts.counts()[geo_index] = sc->edges().size();

            auto l_geo_id = sc->edges().find<IndexT>("l_geo_id");
            UIPC_ASSERT(l_geo_id,
                        "AffineBodyRevoluteJoint: Geometry must have 'l_geo_id' attribute on `edges`");
            auto l_geo_id_view = l_geo_id->view();

            auto r_geo_id = sc->edges().find<IndexT>("r_geo_id");
            UIPC_ASSERT(r_geo_id,
                        "AffineBodyRevoluteJoint: Geometry must have 'r_geo_id' attribute on `edges`");
            auto r_geo_id_view = r_geo_id->view();

            auto l_inst_id = sc->edges().find<IndexT>("l_inst_id");
            UIPC_ASSERT(l_inst_id,
                        "AffineBodyRevoluteJoint: Geometry must have 'l_inst_id' attribute on `edges`");
            auto l_inst_id_view = l_inst_id->view();

            auto r_inst_id = sc->edges().find<IndexT>("r_inst_id");
            UIPC_ASSERT(r_inst_id,
                        "AffineBodyRevoluteJoint: Geometry must have 'r_inst_id' attribute on `edges`");
            auto r_inst_id_view = r_inst_id->view();

            auto strength_ratio = sc->edges().find<Float>("strength_ratio");
            UIPC_ASSERT(strength_ratio,
                        "AffineBodyRevoluteJoint: Geometry must have 'strength_ratio' attribute on `edges`");
            auto strength_ratio_view = strength_ratio->view();

            auto init_angle_attr = sc->edges().find<Float>("init_angle");
            UIPC_ASSERT(init_angle_attr,
                        "AffineBodyRevoluteJoint: Geometry must have 'init_angle' attribute on `edges`");
            auto init_angle_view = init_angle_attr->view();

            auto Es = sc->edges().topo().view();
            auto Ps = sc->positions().view();

            auto l_pos0_attr = sc->edges().find<Vector3>("l_position0");
            auto l_pos1_attr = sc->edges().find<Vector3>("l_position1");
            auto r_pos0_attr = sc->edges().find<Vector3>("r_position0");
            auto r_pos1_attr = sc->edges().find<Vector3>("r_position1");
            bool use_local = l_pos0_attr && l_pos1_attr && r_pos0_attr && r_pos1_attr;

            for(auto&& [i, e] : enumerate(Es))
            {
                IndexT l_gid = l_geo_id_view[i];
                IndexT r_gid = r_geo_id_view[i];
                IndexT l_iid = l_inst_id_view[i];
                IndexT r_iid = r_inst_id_view[i];

                Vector2i body_ids = {info.body_id(l_gid, l_iid), info.body_id(r_gid, r_iid)};
                body_ids_list.push_back(body_ids);

                auto left_sc  = info.body_geo(geo_slots, l_gid);
                auto right_sc = info.body_geo(geo_slots, r_gid);

                UIPC_ASSERT(l_iid >= 0
                                && l_iid
                                       < static_cast<IndexT>(left_sc->instances().size()),
                            "AffineBodyRevoluteJoint: Left instance ID {} is out of range [0, {})",
                            l_iid,
                            left_sc->instances().size());
                UIPC_ASSERT(r_iid >= 0
                                && r_iid
                                       < static_cast<IndexT>(right_sc->instances().size()),
                            "AffineBodyRevoluteJoint: Right instance ID {} is out of range [0, {})",
                            r_iid,
                            right_sc->instances().size());

                Transform LT{left_sc->transforms().view()[l_iid]};
                Transform RT{right_sc->transforms().view()[r_iid]};

                Matrix3x3 L_inv_rot = LT.rotation().inverse();
                Matrix3x3 R_inv_rot = RT.rotation().inverse();

                Vector12 rest_pos;
                Vector3  t;

                if(use_local)
                {
                    rest_pos.segment<3>(0) = l_pos0_attr->view()[i];
                    rest_pos.segment<3>(3) = l_pos1_attr->view()[i];
                    rest_pos.segment<3>(6) = r_pos0_attr->view()[i];
                    rest_pos.segment<3>(9) = r_pos1_attr->view()[i];

                    t = LT.rotation()
                        * (l_pos1_attr->view()[i] - l_pos0_attr->view()[i]);
                }
                else
                {
                    Vector3 P0 = Ps[e[0]];
                    Vector3 P1 = Ps[e[1]];
                    t          = P1 - P0;

                    rest_pos.segment<3>(0) = LT.inverse() * P0;
                    rest_pos.segment<3>(3) = LT.inverse() * P1;
                    rest_pos.segment<3>(6) = RT.inverse() * P0;
                    rest_pos.segment<3>(9) = RT.inverse() * P1;
                }
                UIPC_ASSERT(t.squaredNorm() > 1e-24,
                            "AffineBodyRevoluteJoint: Edge with zero length detected, "
                            "Joint GeometryID = {}, LinkGeoIDs = ({}, {}), LinkInstIDs = ({}, {}), "
                            "Edge = ({}, {})",
                            joint_geo_id,
                            l_gid,
                            r_gid,
                            l_iid,
                            r_iid,
                            e(0),
                            e(1));

                rest_positions_list.push_back(rest_pos);

                // reference: `orthonormal_basis(t, n, b);` (Eigen `Vector3` in the
                // reference, because its `affine_body/utils.h` is Eigen-typed).
                // The port's routine is a *generic* one over the numerics layer's
                // `LVec<S,3>` (the T6.1a convention: one implementation for the host
                // and the DSL, cookbook §7.21 rule 5), so the host-side call bridges
                // through `to_luisa`/`to_eigen` rather than re-spelling the algorithm
                // for Eigen.  `t` is normalised in place by the routine, exactly as in
                // the reference (which passes its un-normalised `t`).
                Vector3 n, b;
                {
                    auto t_l = to_luisa(t);
                    auto n_l = lps::LVec<Float, 3>{};
                    auto b_l = lps::LVec<Float, 3>{};
                    orthonormal_basis<Float>(t_l, n_l, b_l);
                    n = to_eigen(n_l);
                    b = to_eigen(b_l);
                }

                // Storage layout: [n, b] per body
                Vector6 lb;
                lb.segment<3>(0) = L_inv_rot * n;
                lb.segment<3>(3) = L_inv_rot * b;
                l_basis_list.push_back(lb);

                Vector6 rb;
                rb.segment<3>(0) = R_inv_rot * n;
                rb.segment<3>(3) = R_inv_rot * b;
                r_basis_list.push_back(rb);

                init_angles_list.push_back(init_angle_view[i]);
            }

            std::ranges::copy(strength_ratio_view, std::back_inserter(strength_ratio_list));
            ++geo_index;
        });

    h_geo_joint_offsets_counts.scan();

    auto move_to = [](auto& dst, auto& src)
    {
        dst.resize(src.size());
        std::ranges::move(src, dst.begin());
    };

    move_to(h_body_ids, body_ids_list);
    move_to(h_rest_positions, rest_positions_list);
    move_to(h_strength_ratio, strength_ratio_list);
    move_to(h_l_basis, l_basis_list);
    move_to(h_r_basis, r_basis_list);

    h_init_angles.resize(init_angles_list.size());
    std::ranges::copy(init_angles_list, h_init_angles.begin());

    // Seed the persistent unwrapped angle with init_angles (zero relative rotation).
    h_current_angles = h_init_angles;

    copy_host_to_device(body_ids, h_body_ids);
    copy_host_to_device(l_basis, h_l_basis);
    copy_host_to_device(r_basis, h_r_basis);

    // NOTE (the trap documented on `to_luisa(const Vector24&)` in
    // affine_body/type_define.h, now hit for `Vector12`): the `rest_positions`
    // element type's `to_luisa(const Vector12&)` overload is declared *after*
    // `copy_host_to_device` in the include order, so that helper's dependent
    // unqualified call cannot see it and every earlier `to_luisa` becomes viable
    // through Eigen's converting constructor -> C2668.  The pair state is therefore
    // converted explicitly here, where the exact-match overload is visible.
    rest_positions.resize(h_rest_positions.size());
    if(!h_rest_positions.empty())
    {
        std::vector<LVector12> rest_positions_staging;
        rest_positions_staging.reserve(h_rest_positions.size());
        for(const auto& v : h_rest_positions)
            rest_positions_staging.push_back(to_luisa(v));
        rest_positions.view().copy_from(span<const LVector12>{rest_positions_staging});
    }

    strength_ratio.resize(h_strength_ratio.size());
    if(!h_strength_ratio.empty())
        strength_ratio.view().copy_from(span<const Float>{h_strength_ratio});

    init_angles.resize(h_init_angles.size());
    if(!h_init_angles.empty())
        init_angles.view().copy_from(span<const Float>{h_init_angles});

    current_angles.resize(h_current_angles.size());
    if(!h_current_angles.empty())
        current_angles.view().copy_from(span<const Float>{h_current_angles});

    compute_current_angles();
    write_scene();
}

void AffineBodyRevoluteJoint::compute_current_angles()
{
    int n = (int)body_ids.size();
    if(n > 0)
    {
        auto& k = revolute_joint_current_angles_kernel();
        frame_stream().record(k(body_ids.cview(),
                                l_basis.cview(),
                                r_basis.cview(),
                                affine_body_dynamics->qs(),
                                current_angles.view(),
                                init_angles.cview(),
                                std::numbers::pi,
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}

void AffineBodyRevoluteJoint::write_scene()
{
    auto geo_slots = world().scene().geometries();

    // `copy_to` (the reference's blocking device->host copy); the ported BufferDump /
    // buffer view pair needs the frame stream flushed first, exactly as in
    // `BufferDump::dump` (§7.9 rule 7).
    frame_stream().synchronize();
    if(!h_current_angles.empty())
        current_angles.cview().copy_to(span<Float>{h_current_angles});
    lud::main_stream().synchronize();

    IndexT geo_joint_index = 0;

    this->for_each(
        geo_slots,
        [&](geometry::Geometry& geo)
        {
            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc, "AffineBodyRevoluteJoint: Geometry must be a simplicial complex");

            auto angle = sc->edges().find<Float>("angle");

            if(angle)
            {
                auto angle_view      = view(*angle);
                auto [offset, count] = h_geo_joint_offsets_counts[geo_joint_index];
                UIPC_ASSERT(angle_view.size() == count,
                            "AffineBodyRevoluteJoint: angle attribute size {} mismatch with joint count {}",
                            angle_view.size(),
                            count);

                auto src = span{h_current_angles}.subspan(offset, count);
                std::ranges::copy(src, angle_view.begin());
            }

            ++geo_joint_index;
        });
}

void AffineBodyRevoluteJoint::adopt_scene_angles()
{
    auto geo_slots = world().scene().geometries();

    // A mismatch vs h_current_angles (last published) means a user-authored angle
    // (reset/teleport); overlay it, the following resync snaps it to the pose.
    h_adopted_angles.resize(h_current_angles.size());
    frame_stream().synchronize();
    if(!h_adopted_angles.empty())
        current_angles.cview().copy_to(span<Float>{h_adopted_angles});
    lud::main_stream().synchronize();

    IndexT geo_joint_index = 0;
    bool   adopted         = false;

    this->for_each(
        geo_slots,
        [&](geometry::Geometry& geo)
        {
            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc, "AffineBodyRevoluteJoint: Geometry must be a simplicial complex");

            auto angle = sc->edges().find<Float>("angle");
            if(angle)
            {
                auto angle_view      = angle->view();
                auto [offset, count] = h_geo_joint_offsets_counts[geo_joint_index];
                UIPC_ASSERT(angle_view.size() == count,
                            "AffineBodyRevoluteJoint: angle attribute size {} mismatch with joint count {}",
                            angle_view.size(),
                            count);

                for(IndexT i = 0; i < count; ++i)
                {
                    Float user_angle = angle_view[i];
                    if(user_angle != h_current_angles[offset + i])
                    {
                        h_adopted_angles[offset + i] = user_angle;
                        adopted                      = true;
                    }
                }
            }

            ++geo_joint_index;
        });

    if(adopted)
    {
        current_angles.resize(h_adopted_angles.size());
        if(!h_adopted_angles.empty())
        {
            current_angles.view().copy_from(span<const Float>{h_adopted_angles});
            lud::main_stream().synchronize();
        }
    }
}

void AffineBodyRevoluteJoint::do_report_energy_extent(EnergyExtentInfo& info)
{
    info.energy_count(body_ids.size());  // one energy per joint
}

void AffineBodyRevoluteJoint::do_compute_energy(ComputeEnergyInfo& info)
{
    int n = (int)body_ids.size();
    if(n > 0)
    {
        auto& k = revolute_joint_energy_kernel();
        frame_stream().record(k(body_ids.cview(),
                                rest_positions.cview(),
                                strength_ratio.cview(),
                                info.body_masses(),
                                info.qs(),
                                info.energies(),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}

void AffineBodyRevoluteJoint::do_report_gradient_hessian_extent(
    GradientHessianExtentInfo& info)
{
    info.gradient_count(2 * body_ids.size());  // each joint has 2 * Vector12 gradients

    if(info.gradient_only())
        return;

    info.hessian_count(HalfHessianSize
                       * body_ids.size());  // each joint has HalfHessianSize * Matrix12x12
}

void AffineBodyRevoluteJoint::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    int n = (int)body_ids.size();
    if(n == 0)
        return;

    auto gradients = info.gradients();
    auto hessians  = info.hessians();

    {
        auto& k = revolute_joint_gradient_hessian_k1_kernel();
        frame_stream().record(k(body_ids.cview(),
                                rest_positions.cview(),
                                strength_ratio.cview(),
                                info.body_masses(),
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
        auto& k = revolute_joint_gradient_hessian_k2_kernel();
        frame_stream().record(k(body_ids.cview(),
                                rest_positions.cview(),
                                strength_ratio.cview(),
                                info.body_masses(),
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

bool AffineBodyRevoluteJoint::do_dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return curr_angles_dump.dump(fmt::format("{}rj_current_angle.{}", path, frame),
                                 current_angles);
}

bool AffineBodyRevoluteJoint::do_try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return curr_angles_dump.load(fmt::format("{}rj_current_angle.{}", path, frame));
}

void AffineBodyRevoluteJoint::do_apply_recover(RecoverInfo& info)
{
    (void)info;
    curr_angles_dump.apply_to(current_angles);
}

void AffineBodyRevoluteJoint::do_clear_recover(RecoverInfo& info)
{
    (void)info;
    curr_angles_dump.clean_up();
}

// ---------------------------------------------------------------------------
//  AffineBodyRevoluteJointDofReporter
// ---------------------------------------------------------------------------
void AffineBodyRevoluteJointDofReporter::do_build(BuildInfo& info)
{
    (void)info;
    revolute_joint = require<AffineBodyRevoluteJoint>();
}

void AffineBodyRevoluteJointDofReporter::do_update_dof_attributes(
    UpdateDofAttributesInfo& info)
{
    (void)info;
    // Adopt user-authored angles (reset/teleport winding a > pi jump can't unwrap),
    // resync current_angles to the new pose, then republish.
    revolute_joint->adopt_scene_angles();
    revolute_joint->compute_current_angles();
    revolute_joint->write_scene();
}

// ---------------------------------------------------------------------------
//  AffineBodyRevoluteJointTimeIntegrator
// ---------------------------------------------------------------------------
void AffineBodyRevoluteJointTimeIntegrator::do_init(InitInfo& info)
{
    (void)info;
}

void AffineBodyRevoluteJointTimeIntegrator::do_build(BuildInfo& info)
{
    (void)info;
    revolute_joint        = require<AffineBodyRevoluteJoint>();
    affine_body_dynamics  = require<AffineBodyDynamics>();
}

void AffineBodyRevoluteJointTimeIntegrator::do_predict_dof(PredictDofInfo& info)
{
    (void)info;
}

void AffineBodyRevoluteJointTimeIntegrator::do_update_state(UpdateVelocityInfo& info)
{
    (void)info;
    revolute_joint->compute_current_angles();
}
}  // namespace uipc::backend::luisa
