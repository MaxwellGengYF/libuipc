/********************************************************************
 * @file affine_body/constitutions/affine_body_driving_prismatic_joint.cpp
 * @brief T6.3b2 — the ABD driving prismatic joint: its three kernels, the
 *        out-of-line members and the system registration (reference:
 *        src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu,
 *        class `AffineBodyDrivingPrismaticJoint`).
 *
 * The class is declared in `affine_body_driving_prismatic_joint.h` (the file-split
 * deviation, reported there).
 *
 * THE TWO DEVICE PREDICATES THIS FILE INLINES, AND WHY THEY ARE `select`s:
 *   * `if(constrained == 0) { write zeros; return; }` — the reference's early out.
 *     A *device* predicate cannot return from the DSL function, and the values it
 *     skips are exactly zero, so the port computes the general branch and *selects*
 *     the reference's own zero vector / zero matrix when the flag is clear.  The
 *     resulting doublets/triplets are the reference's zeros bit for bit (the write
 *     still happens: `TMA.half_block`/`DVA.segment` emit the entries either way).
 *   * `if(passive == 1) aim_distance = curr_distances(I);` — a device select on the
 *     two *scalars* that feed `d_tidle` (RULE M2).
 * The energy kernel's `if(constrained == 0) { Es(I) = 0.0; return; }` is the same
 * statement with a scalar output.
 ********************************************************************/
#include <affine_body/constitutions/affine_body_driving_prismatic_joint.h>
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

/// `affine_body_driving_prismatic_joint_compute_energy_kernel`
auto& driving_prismatic_joint_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<int32_t>,    // is_constrained
                                                lcrt::Buffer<int32_t>,    // is_passive
                                                lcrt::Buffer<double>,     // strength_ratios
                                                lcrt::Buffer<LVector6>,   // rest_positions
                                                lcrt::Buffer<LVector6>,   // rest_tangents
                                                lcrt::Buffer<double>,     // init_distances
                                                lcrt::Buffer<double>,     // curr_distances
                                                lcrt::Buffer<double>,     // aim_distances
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<double>,               // Es
                                                uint32_t>(                          // n
        [](lcrt::BufferVar<LVector2i>           body_ids,
           lcrt::BufferVar<int32_t>             is_constrained,
           lcrt::BufferVar<int32_t>             is_passive,
           lcrt::BufferVar<double>              strength_ratios,
           lcrt::BufferVar<LVector6>            rest_positions,
           lcrt::BufferVar<LVector6>            rest_tangents,
           lcrt::BufferVar<double>              init_distances,
           lcrt::BufferVar<double>              curr_distances,
           lcrt::BufferVar<double>              aim_distances,
           lcrt::BufferVar<LVector12>           qs,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
           lcrt::BufferVar<double>              Es,
           lcrt::Var<uint32_t>                  n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bids        = load_vec2i(body_ids->read(I));
                auto constrained = is_constrained->read(I);

                auto kappa = lps::detail::mul(
                    strength_ratios->read(I),
                    lps::detail::add(body_masses->read(bids(0)).m_mass,
                                     body_masses->read(bids(1)).m_mass));

                auto passive      = is_passive->read(I);
                auto aim_distance = scalar_detail::select_scalar(
                    aim_distances->read(I), curr_distances->read(I), passive == 1);

                auto d_tidle = lps::detail::sub(aim_distance, init_distances->read(I));

                auto q_i = load_vec12(qs->read(bids(0)));
                auto q_j = load_vec12(qs->read(bids(1)));

                abd_driving_prismatic_joint_detail::DrivingPrismaticJointRestT<
                    lcrt::Var<double>>
                    rest{.rest_c = load_vec6(rest_positions->read(I)),
                         .rest_t = load_vec6(rest_tangents->read(I))};

                auto E01 = abd_driving_prismatic_joint_detail::body_energy<lcrt::Var<double>>(
                    kappa, rest, q_i, q_j, d_tidle);

                // reference: `if(constrained == 0) { Es(I) = 0.0; return; }`
                Es->write(I,
                          scalar_detail::select_scalar(
                              E01, lcrt::Var<double>{0.0}, constrained == 0));
            };
        });
    return k;
}

/// `affine_body_driving_prismatic_joint_compute_gradient_hessian_k1_kernel`
auto& driving_prismatic_joint_gradient_hessian_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<int32_t>,    // is_constrained
                                                lcrt::Buffer<int32_t>,    // is_passive
                                                lcrt::Buffer<double>,     // strength_ratios
                                                lcrt::Buffer<LVector6>,   // rest_positions
                                                lcrt::Buffer<LVector6>,   // rest_tangents
                                                lcrt::Buffer<double>,     // init_distances
                                                lcrt::Buffer<double>,     // curr_distances
                                                lcrt::Buffer<double>,     // aim_distances
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<int32_t>,  // G indices
                                                lcrt::Buffer<double>,   // G values
                                                uint32_t,               // G total segments
                                                int32_t,                // G doublet index offset
                                                int32_t,                // G doublet count
                                                int32_t,  // G total doublet count
                                                uint32_t>(  // n
        [](lcrt::BufferVar<LVector2i>           body_ids,
           lcrt::BufferVar<int32_t>             is_constrained,
           lcrt::BufferVar<int32_t>             is_passive,
           lcrt::BufferVar<double>              strength_ratios,
           lcrt::BufferVar<LVector6>            rest_positions,
           lcrt::BufferVar<LVector6>            rest_tangents,
           lcrt::BufferVar<double>              init_distances,
           lcrt::BufferVar<double>              curr_distances,
           lcrt::BufferVar<double>              aim_distances,
           lcrt::BufferVar<LVector12>           qs,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
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
                auto bids        = load_vec2i(body_ids->read(I));
                auto constrained = is_constrained->read(I);

                auto kappa = lps::detail::mul(
                    strength_ratios->read(I),
                    lps::detail::add(body_masses->read(bids(0)).m_mass,
                                     body_masses->read(bids(1)).m_mass));

                auto passive      = is_passive->read(I);
                auto aim_distance = scalar_detail::select_scalar(
                    aim_distances->read(I), curr_distances->read(I), passive == 1);

                auto d_tidle = lps::detail::sub(aim_distance, init_distances->read(I));

                auto q_i = load_vec12(qs->read(bids(0)));
                auto q_j = load_vec12(qs->read(bids(1)));

                abd_driving_prismatic_joint_detail::DrivingPrismaticJointRestT<
                    lcrt::Var<double>>
                    rest{.rest_c = load_vec6(rest_positions->read(I)),
                         .rest_t = load_vec6(rest_tangents->read(I))};

                auto G = abd_driving_prismatic_joint_detail::body_gradient<lcrt::Var<double>>(
                    kappa, rest, q_i, q_j, d_tidle);

                // reference: `if(constrained == 0) { DVA...write(bids,
                // Vector24::Zero()); return; }`
                // NOTE: `select_container(f, t, p)` is `p ? t : f`, so the general
                // value comes *first* — the two arms reversed here silently zeroed
                // every constrained pair (measured: the assembly's driving blocks
                // all-zero, `dev` exactly 1.0).
                auto const zero_G = lps::make_vec<lcrt::Var<double>, 24>(
                    [](int) noexcept { return lps::zero_of<lcrt::Var<double>>(); });
                G = scalar_detail::select_container(G, zero_G, constrained == 0);

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
                abd_prismatic_joint_detail::write_pair_gradient(D, G, ii * 2, bids);
            };
        });
    return k;
}

/// `affine_body_driving_prismatic_joint_compute_gradient_hessian_k2_kernel`
auto& driving_prismatic_joint_gradient_hessian_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<int32_t>,    // is_constrained
                                                lcrt::Buffer<int32_t>,    // is_passive
                                                lcrt::Buffer<double>,     // strength_ratios
                                                lcrt::Buffer<LVector6>,   // rest_positions
                                                lcrt::Buffer<LVector6>,   // rest_tangents
                                                lcrt::Buffer<double>,     // init_distances
                                                lcrt::Buffer<double>,     // curr_distances
                                                lcrt::Buffer<double>,     // aim_distances
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
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
        [](lcrt::BufferVar<LVector2i>           body_ids,
           lcrt::BufferVar<int32_t>             is_constrained,
           lcrt::BufferVar<int32_t>             is_passive,
           lcrt::BufferVar<double>              strength_ratios,
           lcrt::BufferVar<LVector6>            rest_positions,
           lcrt::BufferVar<LVector6>            rest_tangents,
           lcrt::BufferVar<double>              init_distances,
           lcrt::BufferVar<double>              curr_distances,
           lcrt::BufferVar<double>              aim_distances,
           lcrt::BufferVar<LVector12>           qs,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
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
                auto bids        = load_vec2i(body_ids->read(I));
                auto constrained = is_constrained->read(I);

                auto kappa = lps::detail::mul(
                    strength_ratios->read(I),
                    lps::detail::add(body_masses->read(bids(0)).m_mass,
                                     body_masses->read(bids(1)).m_mass));

                auto passive      = is_passive->read(I);
                auto aim_distance = scalar_detail::select_scalar(
                    aim_distances->read(I), curr_distances->read(I), passive == 1);

                auto d_tidle = lps::detail::sub(aim_distance, init_distances->read(I));

                auto q_i = load_vec12(qs->read(bids(0)));
                auto q_j = load_vec12(qs->read(bids(1)));

                abd_driving_prismatic_joint_detail::DrivingPrismaticJointRestT<
                    lcrt::Var<double>>
                    rest{.rest_c = load_vec6(rest_positions->read(I)),
                         .rest_t = load_vec6(rest_tangents->read(I))};

                auto H = abd_driving_prismatic_joint_detail::body_hessian<lcrt::Var<double>>(
                    kappa, rest, q_i, q_j, d_tidle);

                // reference: `if(constrained == 0) { TMA...write(bids,
                // Matrix24x24::Zero()); return; }`
                auto const zero_H = lps::make_mat<lcrt::Var<double>, 24, 24>(
                    [](int, int) noexcept { return lps::zero_of<lcrt::Var<double>>(); });
                H = scalar_detail::select_container(H, zero_H, constrained == 0);

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
                    T, H, ii * lcrt::cast<std::int32_t>(HalfHessianSize), bids);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyDrivingPrismaticJoint);

U64 AffineBodyDrivingPrismaticJoint::get_uid() const noexcept
{
    return ConstraintUID;
}

void AffineBodyDrivingPrismaticJoint::do_build(BuildInfo& info)
{
    (void)info;
    prismatic_joint = require<AffineBodyPrismaticJoint>();
}

void AffineBodyDrivingPrismaticJoint::do_init(InterAffineBodyAnimator::FilteredInfo& info)
{
    auto geo_slots = world().scene().geometries();
    h_geo_joint_offsets_counts.resize(info.anim_inter_geo_infos().size());

    list<IndexT> is_constrained_list;
    list<Float>  strength_ratios_list;
    list<IndexT> is_passive_list;
    list<Float>  aim_distances_list;

    IndexT joint_offset = 0;
    info.for_each(
        geo_slots,
        [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
        {
            (void)I;
            // check uid
            {
                auto constraint_uids = geo.meta().find<VectorXu64>(builtin::constraint_uids);
                UIPC_ASSERT(constraint_uids,
                            "AffineBodyDrivingPrismaticJoint: Geometry must have 'constraint_uids' attribute");
                bool has_this_constraint = false;
                for(auto&& uid_value : constraint_uids->view().front())
                {
                    if(uid_value == ConstraintUID)
                    {
                        has_this_constraint = true;
                        break;
                    }
                }
                UIPC_ASSERT(has_this_constraint,
                            "AffineBodyDrivingPrismaticJoint: Geometry must have constraint UID {}",
                            ConstraintUID);
            }
            // check constitution uid
            {
                auto constitution_uid = geo.meta().find<U64>(builtin::constitution_uid);
                UIPC_ASSERT(constitution_uid,
                            "AffineBodyDrivingPrismaticJoint: Geometry must have 'constitution_uid' attribute");
                U64 uid_value = constitution_uid->view().front();
                UIPC_ASSERT(uid_value == AffineBodyPrismaticJoint::ConstitutionUID,
                            "AffineBodyDrivingPrismaticJoint: Geometry constitution UID mismatch");
            }
            // get simplicial complex
            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc,
                        "AffineBodyDrivingPrismaticJoint geometry must be SimplicialComplex");

            h_geo_joint_offsets_counts.counts()[joint_offset] = sc->edges().size();
            UIPC_ASSERT(h_geo_joint_offsets_counts.counts()[joint_offset] > 0,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have at least one edge");

            // Driving-only attributes; shared attributes (body_ids, rest_cs,
            // rest_ts, init_distances) are supplied by AffineBodyPrismaticJoint.
            auto is_constrained = sc->edges().find<IndexT>("driving/is_constrained");
            UIPC_ASSERT(is_constrained,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have 'driving/is_constrained' attribute on `edges`");
            std::ranges::copy(is_constrained->view(),
                              std::back_inserter(is_constrained_list));

            auto strength_ratios = sc->edges().find<Float>("driving/strength_ratio");
            UIPC_ASSERT(strength_ratios,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have 'driving/strength_ratio' attribute on `edges`")
            std::ranges::copy(strength_ratios->view(),
                              std::back_inserter(strength_ratios_list));

            auto is_passive = sc->edges().find<IndexT>("is_passive");
            UIPC_ASSERT(is_passive,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have 'is_passive' attribute on `edges`")
            std::ranges::copy(is_passive->view(), std::back_inserter(is_passive_list));

            auto aim_distances = sc->edges().find<Float>("aim_distance");
            UIPC_ASSERT(aim_distances,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have 'aim_distance' attribute on `edges`")
            std::ranges::copy(aim_distances->view(),
                              std::back_inserter(aim_distances_list));

            joint_offset++;
        });
    // scan to compute offsets
    h_geo_joint_offsets_counts.scan();

    h_is_constrained.resize(is_constrained_list.size());
    std::ranges::copy(is_constrained_list, h_is_constrained.begin());

    h_strength_ratios.resize(strength_ratios_list.size());
    std::ranges::copy(strength_ratios_list, h_strength_ratios.begin());

    h_is_passive.resize(is_passive_list.size());
    std::ranges::copy(is_passive_list, h_is_passive.begin());

    h_aim_distances.resize(aim_distances_list.size());
    std::ranges::copy(aim_distances_list, h_aim_distances.begin());

    is_constrained.resize(h_is_constrained.size());
    if(!h_is_constrained.empty())
        is_constrained.view().copy_from(span<const IndexT>{h_is_constrained});

    strength_ratios.resize(h_strength_ratios.size());
    if(!h_strength_ratios.empty())
        strength_ratios.view().copy_from(span<const Float>{h_strength_ratios});

    is_passive.resize(h_is_passive.size());
    if(!h_is_passive.empty())
        is_passive.view().copy_from(span<const IndexT>{h_is_passive});

    aim_distances.resize(h_aim_distances.size());
    if(!h_aim_distances.empty())
        aim_distances.view().copy_from(span<const Float>{h_aim_distances});

    // Sanity check: driving joint reuses body_ids/rest_cs/rest_ts/init_distances
    // from AffineBodyPrismaticJoint via the shared index `I`. Both must
    // therefore enumerate exactly the same (geo, edge) sequence.
    UIPC_ASSERT(h_is_constrained.size() == prismatic_joint->h_body_ids.size(),
                "AffineBodyDrivingPrismaticJoint: joint count {} must equal "
                "AffineBodyPrismaticJoint joint count {} (index alignment required)",
                h_is_constrained.size(),
                prismatic_joint->h_body_ids.size());
}

void AffineBodyDrivingPrismaticJoint::do_step(InterAffineBodyAnimator::FilteredInfo& info)
{
    auto  geo_slots       = world().scene().geometries();
    SizeT geo_joint_index = 0;

    // Mirror all per-edge driving attributes host-side, then upload each
    // buffer once at the end. No change-detection: attributes are expected
    // to change every step (they are the animation signal), so guarding
    // the uploads is pointless.
    info.for_each(
        geo_slots,
        [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
        {
            (void)I;
            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc,
                        "AffineBodyDrivingPrismaticJoint: Geometry must be a simplicial complex");

            auto [offset, count] = h_geo_joint_offsets_counts[geo_joint_index];

            UIPC_ASSERT(static_cast<IndexT>(sc->edges().size()) == count,
                        "AffineBodyDrivingPrismaticJoint: Geometry edges size {} mismatch with joint count {}",
                        sc->edges().size(),
                        count);

            auto is_constrained = sc->edges().find<IndexT>("driving/is_constrained");
            UIPC_ASSERT(is_constrained,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have 'driving/is_constrained' attribute on `edges`");
            std::ranges::copy(is_constrained->view(),
                              span{h_is_constrained}.subspan(offset, count).begin());

            auto is_passive = sc->edges().find<IndexT>("is_passive");
            UIPC_ASSERT(is_passive,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have 'is_passive' attribute on `edges`")
            std::ranges::copy(is_passive->view(),
                              span{h_is_passive}.subspan(offset, count).begin());

            auto strength_ratios = sc->edges().find<Float>("driving/strength_ratio");
            UIPC_ASSERT(strength_ratios,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have 'driving/strength_ratio' attribute on `edges`")
            std::ranges::copy(strength_ratios->view(),
                              span{h_strength_ratios}.subspan(offset, count).begin());

            auto aim_distances = sc->edges().find<Float>("aim_distance");
            UIPC_ASSERT(aim_distances,
                        "AffineBodyDrivingPrismaticJoint: Geometry must have 'aim_distance' attribute on `edges`");
            std::ranges::copy(aim_distances->view(),
                              span{h_aim_distances}.subspan(offset, count).begin());

            ++geo_joint_index;
        });

    // NOTE (port necessity, the T6.3b1 idiom): the reference's `copy_from` is a plain
    // cudaMemcpy, which is a no-op for an empty source; an LC `copy_from` of a
    // zero-size span is not, so each upload is guarded.  The values are identical.
    if(!h_aim_distances.empty())
        aim_distances.view().copy_from(span<const Float>{h_aim_distances});
    if(!h_is_constrained.empty())
        is_constrained.view().copy_from(span<const IndexT>{h_is_constrained});
    if(!h_strength_ratios.empty())
        strength_ratios.view().copy_from(span<const Float>{h_strength_ratios});
    if(!h_is_passive.empty())
        is_passive.view().copy_from(span<const IndexT>{h_is_passive});
}

void AffineBodyDrivingPrismaticJoint::do_report_extent(
    InterAffineBodyAnimator::ReportExtentInfo& info)
{
    info.energy_count(is_constrained.size());
    info.gradient_count(2 * is_constrained.size());
    if(info.gradient_only())
        return;

    info.hessian_count(HalfHessianSize * is_constrained.size());
}

void AffineBodyDrivingPrismaticJoint::do_compute_energy(
    InterAffineBodyAnimator::ComputeEnergyInfo& info)
{
    int n = (int)is_constrained.size();
    if(n > 0)
    {
        auto& k = driving_prismatic_joint_energy_kernel();
        frame_stream().record(k(prismatic_joint->body_ids.cview(),
                                is_constrained.cview(),
                                is_passive.cview(),
                                strength_ratios.cview(),
                                prismatic_joint->rest_cs.cview(),
                                prismatic_joint->rest_ts.cview(),
                                prismatic_joint->init_distances.cview(),
                                prismatic_joint->current_distances.cview(),
                                aim_distances.cview(),
                                info.qs(),
                                info.body_masses(),
                                info.energies(),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}

void AffineBodyDrivingPrismaticJoint::do_compute_gradient_hessian(
    InterAffineBodyAnimator::GradientHessianInfo& info)
{
    int n = (int)is_constrained.size();
    if(n == 0)
        return;

    auto gradients = info.gradients();
    auto hessians  = info.hessians();

    {
        auto& k = driving_prismatic_joint_gradient_hessian_k1_kernel();
        frame_stream().record(k(prismatic_joint->body_ids.cview(),
                                is_constrained.cview(),
                                is_passive.cview(),
                                strength_ratios.cview(),
                                prismatic_joint->rest_cs.cview(),
                                prismatic_joint->rest_ts.cview(),
                                prismatic_joint->init_distances.cview(),
                                prismatic_joint->current_distances.cview(),
                                aim_distances.cview(),
                                info.qs(),
                                info.body_masses(),
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
        auto& k = driving_prismatic_joint_gradient_hessian_k2_kernel();
        frame_stream().record(k(prismatic_joint->body_ids.cview(),
                                is_constrained.cview(),
                                is_passive.cview(),
                                strength_ratios.cview(),
                                prismatic_joint->rest_cs.cview(),
                                prismatic_joint->rest_ts.cview(),
                                prismatic_joint->init_distances.cview(),
                                prismatic_joint->current_distances.cview(),
                                aim_distances.cview(),
                                info.qs(),
                                info.body_masses(),
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
