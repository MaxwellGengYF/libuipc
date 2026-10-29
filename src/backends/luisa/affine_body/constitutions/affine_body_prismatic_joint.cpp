/********************************************************************
 * @file affine_body/constitutions/affine_body_prismatic_joint.cpp
 * @brief T6.3b2 — the ABD prismatic joint: its four kernels, their cached shaders,
 *        the out-of-line members and the system registration (reference:
 *        src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu,
 *        classes `AffineBodyPrismaticJoint`, `...DofReporter`, `...TimeIntegrator`).
 *
 * The classes are declared in `affine_body_prismatic_joint.h` (the file-split
 * deviation, reported there).
 *
 * KERNEL MAPPING (cookbook §7.9/§7.10/§7.13, the T6.3b1 notes)
 *   * `cuda_tool::CBufferView<T>` -> `lud::LCBufferView<L*>`; the joint's own buffers
 *     are `lud::LDeviceBuffer<...>`, read through `.cview()`.
 *   * the reference's `DoubletVectorAssembler` / `TripletMatrixAssembler` are not
 *     ported: the two write patterns live in
 *     `abd_prismatic_joint_detail::{write_pair_gradient, write_pair_half_hessian}`
 *     and this file builds the *device* views from the handles the assembly hands
 *     over (the T6.1b idiom).
 *   * one statement reads `affine_body_dynamics->qs()` *directly* (the distance
 *     kernel, exactly like the reference): the other three kernels take the
 *     manager's `info.qs()` window.  Both name the same device buffer.
 *   * the reference's `if(!gradient_only)` becomes `$if(gradient_only == 0u)` (RULE
 *     M2: a device predicate cannot drive host control flow) — here the *host* side
 *     already splits gradient-only off (the reference does too, with an early
 *     `return`), so this kernel has no such branch.
 ********************************************************************/
#include <affine_body/constitutions/affine_body_prismatic_joint.h>
#include <affine_body/utils.h>  // orthonormal_basis (the reference's `affine_body/utils.h`)
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <luisa_device/linear/linear.h>
// `uipc::Transform` = `Eigen::Transform<Float,3,Eigen::Affine>`: the reference's `.cu`
// gets it through `cuda_tool`; the port needs the definition for `LT.rotation()` /
// `LT.inverse()` (measured C2079 in T6.3b1).
#include <Eigen/Geometry>
#include <fmt/format.h>

namespace uipc::backend::luisa
{
namespace
{
// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `affine_body_prismatic_joint_compute_current_distances_kernel`
auto& prismatic_joint_current_distances_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<LVector6>,   // rest_ts
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<double>,     // current_distances
                                                lcrt::Buffer<double>,     // init_distances
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector2i> body_ids,
           lcrt::BufferVar<LVector6>  rest_cs,
           lcrt::BufferVar<LVector6>  rest_ts,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<double>    current_distances,
           lcrt::BufferVar<double>    init_distances,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bids = load_vec2i(body_ids->read(I));

                auto q_i = load_vec12(qs->read(bids(0)));
                auto q_j = load_vec12(qs->read(bids(1)));

                auto C_bar = load_vec6(rest_cs->read(I));
                auto t_bar = load_vec6(rest_ts->read(I));

                // Float distance; compute_absolute_distance(distance, C_bar, t_bar, q_i, q_j);
                auto distance =
                    abd_driving_prismatic_joint_detail::compute_absolute_distance<
                        lcrt::Var<double>>(C_bar, t_bar, q_i, q_j);

                current_distances->write(I,
                                         distance + init_distances->read(I));
            };
        });
    return k;
}

/// `affine_body_prismatic_joint_compute_energy_kernel`
auto& prismatic_joint_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<LVector6>,   // rest_ts
                                                lcrt::Buffer<LVector6>,   // rest_ns
                                                lcrt::Buffer<LVector6>,   // rest_bs
                                                lcrt::Buffer<double>,     // strength_ratio
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<double>,     // Es
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector2i>           body_ids,
           lcrt::BufferVar<LVector6>            rest_cs,
           lcrt::BufferVar<LVector6>            rest_ts,
           lcrt::BufferVar<LVector6>            rest_ns,
           lcrt::BufferVar<LVector6>            rest_bs,
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

                abd_prismatic_joint_detail::PrismaticJointRestT<lcrt::Var<double>> rest{
                    .rest_c = load_vec6(rest_cs->read(I)),
                    .rest_t = load_vec6(rest_ts->read(I)),
                    .rest_n = load_vec6(rest_ns->read(I)),
                    .rest_b = load_vec6(rest_bs->read(I))};

                auto E = abd_prismatic_joint_detail::body_energy<lcrt::Var<double>>(
                    kappa, rest, qi, qj);

                Es->write(I, E);
            };
        });
    return k;
}

/// `affine_body_prismatic_joint_compute_gradient_hessian_k1_kernel`
auto& prismatic_joint_gradient_hessian_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<LVector6>,   // rest_ts
                                                lcrt::Buffer<LVector6>,   // rest_ns
                                                lcrt::Buffer<LVector6>,   // rest_bs
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
           lcrt::BufferVar<LVector6>            rest_cs,
           lcrt::BufferVar<LVector6>            rest_ts,
           lcrt::BufferVar<LVector6>            rest_ns,
           lcrt::BufferVar<LVector6>            rest_bs,
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

                abd_prismatic_joint_detail::PrismaticJointRestT<lcrt::Var<double>> rest{
                    .rest_c = load_vec6(rest_cs->read(I)),
                    .rest_t = load_vec6(rest_ts->read(I)),
                    .rest_n = load_vec6(rest_ns->read(I)),
                    .rest_b = load_vec6(rest_bs->read(I))};

                auto ii = lcrt::cast<std::int32_t>(I);

                // reference: DoubletVectorAssembler DVA{G12s};
                //            DVA.segment<2>(2 * I).write(indices, G);
                auto G = abd_prismatic_joint_detail::body_gradient<lcrt::Var<double>>(
                    kappa, rest, qi, qj);
                auto D = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 12>::from_handles_full(
                    G_indices,
                    G_values,
                    G_total_segments,
                    G_doublet_index_offset,
                    G_doublet_count,
                    G_total_doublet_count,
                    std::int32_t{0},
                    G_total_segments);
                abd_prismatic_joint_detail::write_pair_gradient(D, G, ii * 2, bids);
            };
        });
    return k;
}

/// `affine_body_prismatic_joint_compute_gradient_hessian_k2_kernel`
auto& prismatic_joint_gradient_hessian_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<LVector6>,   // rest_ts
                                                lcrt::Buffer<LVector6>,   // rest_ns
                                                lcrt::Buffer<LVector6>,   // rest_bs
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
           lcrt::BufferVar<LVector6>            rest_cs,
           lcrt::BufferVar<LVector6>            rest_ts,
           lcrt::BufferVar<LVector6>            rest_ns,
           lcrt::BufferVar<LVector6>            rest_bs,
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

                abd_prismatic_joint_detail::PrismaticJointRestT<lcrt::Var<double>> rest{
                    .rest_c = load_vec6(rest_cs->read(I)),
                    .rest_t = load_vec6(rest_ts->read(I)),
                    .rest_n = load_vec6(rest_ns->read(I)),
                    .rest_b = load_vec6(rest_bs->read(I))};

                auto ii = lcrt::cast<std::int32_t>(I);

                // reference: TripletMatrixAssembler TMA{H12x12s};
                //            TMA.half_block<2>(HalfHessianSize * I).write(indices, H);
                const auto H = abd_prismatic_joint_detail::body_hessian<lcrt::Var<double>>(
                    kappa, rest, qi, qj);
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
                    T, H, ii * lcrt::cast<std::int32_t>(HalfHessianSize), bids);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyPrismaticJoint);
REGISTER_SIM_SYSTEM(AffineBodyPrismaticJointDofReporter);
REGISTER_SIM_SYSTEM(AffineBodyPrismaticJointTimeIntegrator);

U64 AffineBodyPrismaticJoint::get_uid() const noexcept
{
    return ConstitutionUID;
}

void AffineBodyPrismaticJoint::do_build(BuildInfo& info)
{
    (void)info;
    // the reference's body (`InterAffineBodyConstitution::do_build()` -- the base's
    // own virtual -- does the uid check, the manager resolution and the registration)
    affine_body_dynamics = require<AffineBodyDynamics>();
    on_write_scene([this]() { write_scene(); });
}

void AffineBodyPrismaticJoint::do_init(FilteredInfo& info)
{
    auto geo_slots = world().scene().geometries();

    h_geo_joint_offsets_counts.resize(info.inter_geo_infos().size());

    list<Vector2i> body_ids_list;
    list<Vector6>  rest_c_list;
    list<Vector6>  rest_t_list;
    list<Vector6>  rest_n_list;
    list<Vector6>  rest_b_list;
    list<Float>    strength_ratio_list;
    list<Float>    init_distances_list;

    IndexT geo_index = 0;
    info.for_each(
        geo_slots,
        [&](geometry::Geometry& geo)
        {
            auto uid       = geo.meta().find<U64>(builtin::constitution_uid);
            U64  uid_value = uid->view()[0];
            UIPC_ASSERT(uid_value == ConstitutionUID,
                        "AffineBodyPrismaticJoint: Geometry constitution UID mismatch");

            auto sc = geo.as<geometry::SimplicialComplex>();

            h_geo_joint_offsets_counts.counts()[geo_index] = sc->edges().size();

            auto l_geo_id = sc->edges().find<IndexT>("l_geo_id");
            UIPC_ASSERT(l_geo_id,
                        "AffineBodyPrismaticJoint: Geometry must have 'l_geo_id' attribute on `edges`");
            auto l_geo_id_view = l_geo_id->view();

            auto r_geo_id = sc->edges().find<IndexT>("r_geo_id");
            UIPC_ASSERT(r_geo_id,
                        "AffineBodyPrismaticJoint: Geometry must have 'r_geo_id' attribute on `edges`");
            auto r_geo_id_view = r_geo_id->view();

            auto l_inst_id = sc->edges().find<IndexT>("l_inst_id");
            UIPC_ASSERT(l_inst_id,
                        "AffineBodyPrismaticJoint: Geometry must have 'l_inst_id' attribute on `edges`");
            auto l_inst_id_view = l_inst_id->view();

            auto r_inst_id = sc->edges().find<IndexT>("r_inst_id");
            UIPC_ASSERT(r_inst_id,
                        "AffineBodyPrismaticJoint: Geometry must have 'r_inst_id' attribute on `edges`");
            auto r_inst_id_view = r_inst_id->view();

            auto strength_ratio = sc->edges().find<Float>("strength_ratio");
            UIPC_ASSERT(strength_ratio,
                        "AffineBodyPrismaticJoint: Geometry must have 'strength_ratio' attribute on `edges`");
            auto strength_ratio_view = strength_ratio->view();

            auto init_distance_attr = sc->edges().find<Float>("init_distance");
            UIPC_ASSERT(init_distance_attr,
                        "AffineBodyPrismaticJoint: Geometry must have 'init_distance' attribute on `edges`");
            auto init_distance_view = init_distance_attr->view();

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
                            "AffineBodyPrismaticJoint: Left instance ID {} is out of range [0, {})",
                            l_iid,
                            left_sc->instances().size());
                UIPC_ASSERT(r_iid >= 0
                                && r_iid
                                       < static_cast<IndexT>(right_sc->instances().size()),
                            "AffineBodyPrismaticJoint: Right instance ID {} is out of range [0, {})",
                            r_iid,
                            right_sc->instances().size());

                Transform LT{left_sc->transforms().view()[l_iid]};
                Transform RT{right_sc->transforms().view()[r_iid]};

                Vector3 t;
                Vector6 rest_position_c;
                if(use_local)
                {
                    Vector3 lp0 = LT * l_pos0_attr->view()[i];
                    Vector3 lp1 = LT * l_pos1_attr->view()[i];
                    t           = (lp1 - lp0).normalized();
                    rest_position_c.segment<3>(0) = l_pos0_attr->view()[i];
                    rest_position_c.segment<3>(3) = r_pos0_attr->view()[i];
                }
                else
                {
                    Vector3 P0 = Ps[e[0]];
                    Vector3 P1 = Ps[e[1]];
                    UIPC_ASSERT((P0 - P1).squaredNorm() > 0,
                                "AffineBodyPrismaticJoint: Edge positions must not be too close");
                    t                             = (P1 - P0).normalized();
                    rest_position_c.segment<3>(0) = LT.inverse() * P0;
                    rest_position_c.segment<3>(3) = RT.inverse() * P0;
                }

                // reference: `orthonormal_basis(t, n, b);` (Eigen `Vector3` in the
                // reference, because its `affine_body/utils.h` is Eigen-typed).
                // The port's routine is a *generic* one over the numerics layer's
                // `LVec<S,3>` (the T6.1a convention: one implementation for the host
                // and the DSL, cookbook §7.21 rule 5), so the host-side call bridges
                // through `to_luisa`/`to_eigen` rather than re-spelling the algorithm
                // for Eigen.  `t` is already normalised, and the routine normalises in
                // place, so the value is unchanged.
                Vector3 n, b;
                {
                    auto t_l = to_luisa(t);
                    auto n_l = lps::LVec<Float, 3>{};
                    auto b_l = lps::LVec<Float, 3>{};
                    orthonormal_basis<Float>(t_l, n_l, b_l);
                    n = to_eigen(n_l);
                    b = to_eigen(b_l);
                }

                rest_c_list.push_back(rest_position_c);

                Matrix3x3 LT_rotation = LT.rotation();
                Matrix3x3 RT_rotation = RT.rotation();

                Vector6 rest_vec_t;
                rest_vec_t.segment<3>(0) = LT_rotation.inverse() * t;
                rest_vec_t.segment<3>(3) = RT_rotation.inverse() * t;
                rest_t_list.push_back(rest_vec_t);

                Vector6 rest_vec_n;
                rest_vec_n.segment<3>(0) = LT_rotation.inverse() * n;
                rest_vec_n.segment<3>(3) = RT_rotation.inverse() * n;
                rest_n_list.push_back(rest_vec_n);

                Vector6 rest_vec_b;
                rest_vec_b.segment<3>(0) = LT_rotation.inverse() * b;
                rest_vec_b.segment<3>(3) = RT_rotation.inverse() * b;
                rest_b_list.push_back(rest_vec_b);

                init_distances_list.push_back(init_distance_view[i]);
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
    move_to(h_rest_cs, rest_c_list);
    move_to(h_rest_ts, rest_t_list);
    move_to(h_rest_ns, rest_n_list);
    move_to(h_rest_bs, rest_b_list);
    move_to(h_strength_ratios, strength_ratio_list);

    h_init_distances.resize(init_distances_list.size());
    std::ranges::copy(init_distances_list, h_init_distances.begin());

    h_current_distances.resize(h_init_distances.size());

    copy_host_to_device(body_ids, h_body_ids);
    copy_host_to_device(rest_cs, h_rest_cs);
    copy_host_to_device(rest_ts, h_rest_ts);
    copy_host_to_device(rest_ns, h_rest_ns);
    copy_host_to_device(rest_bs, h_rest_bs);

    strength_ratios.resize(h_strength_ratios.size());
    if(!h_strength_ratios.empty())
        strength_ratios.view().copy_from(span<const Float>{h_strength_ratios});

    init_distances.resize(h_init_distances.size());
    if(!h_init_distances.empty())
        init_distances.view().copy_from(span<const Float>{h_init_distances});

    current_distances.resize(h_current_distances.size());
    if(!h_current_distances.empty())
        current_distances.view().copy_from(span<const Float>{h_current_distances});

    compute_current_distances();
    write_scene();
}

void AffineBodyPrismaticJoint::compute_current_distances()
{
    int n = (int)body_ids.size();
    if(n > 0)
    {
        auto& k = prismatic_joint_current_distances_kernel();
        frame_stream().record(k(body_ids.cview(),
                                rest_cs.cview(),
                                rest_ts.cview(),
                                affine_body_dynamics->qs(),
                                current_distances.view(),
                                init_distances.cview(),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}

void AffineBodyPrismaticJoint::write_scene()
{
    auto geo_slots = world().scene().geometries();

    // `copy_to` (the reference's blocking device->host copy); the ported BufferDump /
    // buffer view pair needs the frame stream flushed first, exactly as in
    // `BufferDump::dump` (§7.9 rule 7).
    frame_stream().synchronize();
    if(!h_current_distances.empty())
        current_distances.cview().copy_to(span<Float>{h_current_distances});
    lud::LStream::current().synchronize();

    IndexT geo_joint_index = 0;

    this->for_each(
        geo_slots,
        [&](geometry::Geometry& geo)
        {
            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc, "AffineBodyPrismaticJoint: Geometry must be a simplicial complex");

            auto distance = sc->edges().find<Float>("distance");

            if(distance)
            {
                auto distance_view   = view(*distance);
                auto [offset, count] = h_geo_joint_offsets_counts[geo_joint_index];
                UIPC_ASSERT(distance_view.size() == count,
                            "AffineBodyPrismaticJoint: distance attribute size {} mismatch with joint count {}",
                            distance_view.size(),
                            count);

                auto src = span{h_current_distances}.subspan(offset, count);
                std::ranges::copy(src, distance_view.begin());
            }

            ++geo_joint_index;
        });
}

void AffineBodyPrismaticJoint::do_report_energy_extent(EnergyExtentInfo& info)
{
    info.energy_count(body_ids.size());  // one energy per joint
}

void AffineBodyPrismaticJoint::do_compute_energy(ComputeEnergyInfo& info)
{
    int n = (int)body_ids.size();
    if(n > 0)
    {
        auto& k = prismatic_joint_energy_kernel();
        frame_stream().record(k(body_ids.cview(),
                                rest_cs.cview(),
                                rest_ts.cview(),
                                rest_ns.cview(),
                                rest_bs.cview(),
                                strength_ratios.cview(),
                                info.body_masses(),
                                info.qs(),
                                info.energies(),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}

void AffineBodyPrismaticJoint::do_report_gradient_hessian_extent(
    GradientHessianExtentInfo& info)
{
    info.gradient_count(2 * body_ids.size());  // each joint has 2 * Vector12 gradients

    if(info.gradient_only())
        return;

    info.hessian_count(HalfHessianSize
                       * body_ids.size());  // each joint has HalfHessianSize * Matrix12x12
}

void AffineBodyPrismaticJoint::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    int n = (int)body_ids.size();
    if(n == 0)
        return;

    auto gradients = info.gradients();
    auto hessians  = info.hessians();

    {
        auto& k = prismatic_joint_gradient_hessian_k1_kernel();
        frame_stream().record(k(body_ids.cview(),
                                rest_cs.cview(),
                                rest_ts.cview(),
                                rest_ns.cview(),
                                rest_bs.cview(),
                                strength_ratios.cview(),
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
        auto& k = prismatic_joint_gradient_hessian_k2_kernel();
        frame_stream().record(k(body_ids.cview(),
                                rest_cs.cview(),
                                rest_ts.cview(),
                                rest_ns.cview(),
                                rest_bs.cview(),
                                strength_ratios.cview(),
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

bool AffineBodyPrismaticJoint::do_dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return curr_distances_dump.dump(
        fmt::format("{}pj_current_distances.{}", path, frame), current_distances);
}

bool AffineBodyPrismaticJoint::do_try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return curr_distances_dump.load(fmt::format("{}pj_current_distances.{}", path, frame));
}

void AffineBodyPrismaticJoint::do_apply_recover(RecoverInfo& info)
{
    (void)info;
    curr_distances_dump.apply_to(current_distances);
}

void AffineBodyPrismaticJoint::do_clear_recover(RecoverInfo& info)
{
    (void)info;
    curr_distances_dump.clean_up();
}

// ---------------------------------------------------------------------------
//  AffineBodyPrismaticJointDofReporter
// ---------------------------------------------------------------------------
void AffineBodyPrismaticJointDofReporter::do_build(BuildInfo& info)
{
    (void)info;
    prismatic_joint = require<AffineBodyPrismaticJoint>();
}

void AffineBodyPrismaticJointDofReporter::do_update_dof_attributes(
    UpdateDofAttributesInfo& info)
{
    (void)info;
    // Re-sync persistent per-joint distances (current_distances) with the
    // new vertex/body attribute layout before the next frame uses them.
    prismatic_joint->compute_current_distances();
}

// ---------------------------------------------------------------------------
//  AffineBodyPrismaticJointTimeIntegrator
// ---------------------------------------------------------------------------
void AffineBodyPrismaticJointTimeIntegrator::do_init(InitInfo& info)
{
    (void)info;
}

void AffineBodyPrismaticJointTimeIntegrator::do_build(BuildInfo& info)
{
    (void)info;
    prismatic_joint       = require<AffineBodyPrismaticJoint>();
    affine_body_dynamics  = require<AffineBodyDynamics>();
}

void AffineBodyPrismaticJointTimeIntegrator::do_predict_dof(PredictDofInfo& info)
{
    (void)info;
}

void AffineBodyPrismaticJointTimeIntegrator::do_update_state(UpdateVelocityInfo& info)
{
    (void)info;
    prismatic_joint->compute_current_distances();
}
}  // namespace uipc::backend::luisa
