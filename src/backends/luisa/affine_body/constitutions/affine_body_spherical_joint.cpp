/********************************************************************
 * @file affine_body/constitutions/affine_body_spherical_joint.cpp
 * @brief T6.3b1 — the ABD spherical joint: its two kernels, their cached shaders,
 *        the out-of-line members and the system registration (reference:
 *        src/backends/cuda/affine_body/constitutions/affine_body_spherical_joint.cu).
 *
 * The class itself is declared in `affine_body_spherical_joint.h`; the kernel
 * mapping (windowed `LCBufferView` arguments, the per-joint host bookkeeping, the
 * device views built from the assembly's handles, `$if(gradient_only == 0u)`) is the
 * one documented in `affine_body_fixed_joint.cpp` — the only difference from that
 * file is the reference's own: the spherical joint has no frame rows, so `rest_cs`
 * is its only rest buffer and the gradient/Hessian kernels compute the translation
 * half alone.
 ********************************************************************/
#include <affine_body/constitutions/affine_body_spherical_joint.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <luisa_device/linear/linear.h>
// `uipc::Transform` = `Eigen::Transform<Float,3,Eigen::Affine>` (see the fixed
// joint's .cpp: measured C2079 without the definition).
#include <Eigen/Geometry>

namespace uipc::backend::luisa
{
namespace
{
// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `affine_body_spherical_joint_compute_energy_kernel`
auto& spherical_joint_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<double>,     // strength_ratios
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<double>,     // Es
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector2i>            body_ids,
           lcrt::BufferVar<LVector6>             rest_cs,
           lcrt::BufferVar<double>               strength_ratios,
           lcrt::BufferVar<ABDJacobiDyadicMass>  body_masses,
           lcrt::BufferVar<LVector12>            qs,
           lcrt::BufferVar<double>               Es,
           lcrt::Var<uint32_t>                   n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bids = load_vec2i(body_ids->read(I));
                auto kappa = lps::detail::mul(
                    strength_ratios->read(I),
                    lps::detail::add(body_masses->read(bids(0)).m_mass,
                                     body_masses->read(bids(1)).m_mass));

                abd_spherical_joint_detail::SphericalJointRestT<lcrt::Var<double>> rest{
                    .rest_c = load_vec6(rest_cs->read(I))};

                auto E = abd_spherical_joint_detail::body_energy<lcrt::Var<double>>(
                    kappa,
                    rest,
                    load_vec12(qs->read(bids(0))),
                    load_vec12(qs->read(bids(1))));

                Es->write(I, E);
            };
        });
    return k;
}

/// `affine_body_spherical_joint_compute_gradient_hessian_kernel`
auto& spherical_joint_gradient_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<LVector6>,   // rest_cs
                                                lcrt::Buffer<double>,     // strength_ratios
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<int32_t>,    // G indices
                                                lcrt::Buffer<double>,     // G values
                                                uint32_t,  // G total segments
                                                int32_t,   // G doublet index offset
                                                int32_t,   // G doublet count
                                                int32_t,   // G total doublet count
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
                                                uint32_t,  // gradient_only
                                                uint32_t,  // HalfHessianSize
                                                uint32_t>(  // n
        [](lcrt::BufferVar<LVector2i>            body_ids,
           lcrt::BufferVar<LVector6>             rest_cs,
           lcrt::BufferVar<double>               strength_ratios,
           lcrt::BufferVar<ABDJacobiDyadicMass>  body_masses,
           lcrt::BufferVar<LVector12>            qs,
           lcrt::BufferVar<int32_t>              G_indices,
           lcrt::BufferVar<double>               G_values,
           lcrt::Var<uint32_t>                   G_total_segments,
           lcrt::Var<int32_t>                    G_doublet_index_offset,
           lcrt::Var<int32_t>                    G_doublet_count,
           lcrt::Var<int32_t>                    G_total_doublet_count,
           lcrt::BufferVar<int32_t>              H_row_indices,
           lcrt::BufferVar<int32_t>              H_col_indices,
           lcrt::BufferVar<double>               H_values,
           lcrt::Var<int32_t>                    H_total_rows,
           lcrt::Var<int32_t>                    H_total_cols,
           lcrt::Var<int32_t>                    H_triplet_index_offset,
           lcrt::Var<int32_t>                    H_triplet_count,
           lcrt::Var<int32_t>                    H_total_triplet_count,
           lcrt::Var<int32_t>                    H_sm_offset_x,
           lcrt::Var<int32_t>                    H_sm_offset_y,
           lcrt::Var<int32_t>                    H_sm_extent_x,
           lcrt::Var<int32_t>                    H_sm_extent_y,
           lcrt::Var<uint32_t>                   gradient_only,
           lcrt::Var<uint32_t>                   HalfHessianSize,
           lcrt::Var<uint32_t>                   n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto bids = load_vec2i(body_ids->read(I));
                auto kappa = lps::detail::mul(
                    strength_ratios->read(I),
                    lps::detail::add(body_masses->read(bids(0)).m_mass,
                                     body_masses->read(bids(1)).m_mass));

                abd_spherical_joint_detail::SphericalJointRestT<lcrt::Var<double>> rest{
                    .rest_c = load_vec6(rest_cs->read(I))};

                auto qi = load_vec12(qs->read(bids(0)));
                auto qj = load_vec12(qs->read(bids(1)));

                auto ii = lcrt::cast<std::int32_t>(I);

                // reference: DoubletVectorAssembler DVA{G12s};
                //            DVA.segment<2>(2 * I).write(indices, G);
                auto G = abd_spherical_joint_detail::body_gradient<lcrt::Var<double>>(
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
                abd_spherical_joint_detail::write_pair_gradient(D, G, ii * 2, bids);

                // reference: `if(!gradient_only) { ... }`
                $if(gradient_only == 0u)
                {
                    const auto H =
                        abd_spherical_joint_detail::body_hessian<lcrt::Var<double>>(
                            kappa, rest, qi, qj);
                    auto T = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 12, 12>::
                        from_handles_full(H_row_indices,
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
                    // reference: TripletMatrixAssembler TMA{H12x12s};
                    //            TMA.half_block<2>(HalfHessianSize * I).write(indices, H);
                    abd_spherical_joint_detail::write_pair_half_hessian(
                        T, H, ii * lcrt::cast<std::int32_t>(HalfHessianSize), bids);
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodySphericalJoint);

U64 AffineBodySphericalJoint::get_uid() const noexcept
{
    return ConstitutionUID;
}

void AffineBodySphericalJoint::do_build(BuildInfo& info)
{
    (void)info;
    affine_body_dynamics = require<AffineBodyDynamics>();
}

void AffineBodySphericalJoint::do_init(FilteredInfo& info)
{
    auto geo_slots = world().scene().geometries();

    list<Vector2i> body_ids_list;
    list<Vector6>  rest_c_list;
    list<Float>    strength_ratio_list;

    info.for_each(
        geo_slots,
        [&](geometry::Geometry& geo)
        {
            auto sc = geo.as<geometry::SimplicialComplex>();

            auto l_geo_id       = sc->vertices().find<IndexT>("l_geo_id");
            auto l_geo_id_view  = l_geo_id->view();
            auto r_geo_id       = sc->vertices().find<IndexT>("r_geo_id");
            auto r_geo_id_view  = r_geo_id->view();
            auto l_inst_id      = sc->vertices().find<IndexT>("l_inst_id");
            auto l_inst_id_view = l_inst_id->view();
            auto r_inst_id      = sc->vertices().find<IndexT>("r_inst_id");
            auto r_inst_id_view = r_inst_id->view();
            auto strength_ratio_view =
                sc->vertices().find<Float>("strength_ratio")->view();

            auto pos0_attr = sc->vertices().find<Vector3>("l_position");
            auto pos1_attr = sc->vertices().find<Vector3>("r_position");
            const bool use_local_positions = pos0_attr && pos1_attr;

            auto        Ps       = sc->positions().view();
            const SizeT n_joints = sc->vertices().size();
            for(SizeT i = 0; i < n_joints; ++i)
            {
                IndexT l_gid = l_geo_id_view[i];
                IndexT r_gid = r_geo_id_view[i];
                IndexT l_iid = l_inst_id_view[i];
                IndexT r_iid = r_inst_id_view[i];

                body_ids_list.push_back({info.body_id(l_gid, l_iid),
                                         info.body_id(r_gid, r_iid)});

                Transform LT{info.body_geo(geo_slots, l_gid)->transforms().view()[l_iid]};
                Transform RT{info.body_geo(geo_slots, r_gid)->transforms().view()[r_iid]};

                Vector6 rest_c;
                if(use_local_positions)
                {
                    rest_c.segment<3>(0) = pos0_attr->view()[i];
                    rest_c.segment<3>(3) = pos1_attr->view()[i];
                }
                else
                {
                    Vector3 anchor       = Ps[i];
                    rest_c.segment<3>(0) = LT.inverse() * anchor;
                    rest_c.segment<3>(3) = RT.inverse() * anchor;
                }
                rest_c_list.push_back(rest_c);
            }

            std::ranges::copy(strength_ratio_view,
                              std::back_inserter(strength_ratio_list));
        });

    auto move_to = [](auto& dst, auto& src)
    {
        dst.resize(src.size());
        std::ranges::move(src, dst.begin());
    };

    move_to(h_body_ids, body_ids_list);
    move_to(h_rest_cs, rest_c_list);
    move_to(h_strength_ratios, strength_ratio_list);

    copy_host_to_device(body_ids, h_body_ids);
    copy_host_to_device(rest_cs, h_rest_cs);

    strength_ratios.resize(h_strength_ratios.size());
    if(!h_strength_ratios.empty())
        strength_ratios.view().copy_from(span<const Float>{h_strength_ratios});
}

void AffineBodySphericalJoint::do_report_energy_extent(EnergyExtentInfo& info)
{
    info.energy_count(body_ids.size());
}

void AffineBodySphericalJoint::do_compute_energy(ComputeEnergyInfo& info)
{
    int n = (int)body_ids.size();
    if(n > 0)
    {
        auto& k = spherical_joint_energy_kernel();
        frame_stream().record(k(body_ids.cview(),
                                rest_cs.cview(),
                                strength_ratios.cview(),
                                info.body_masses(),
                                info.qs(),
                                info.energies(),
                                static_cast<uint32_t>(n))
                                .dispatch(n));
    }
}

void AffineBodySphericalJoint::do_report_gradient_hessian_extent(
    GradientHessianExtentInfo& info)
{
    info.gradient_count(2 * body_ids.size());
    if(info.gradient_only())
        return;
    info.hessian_count(HalfHessianSize * body_ids.size());
}

void AffineBodySphericalJoint::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto gradient_only = info.gradient_only();
    int  n             = (int)body_ids.size();
    if(n == 0)
        return;

    auto gradients = info.gradients();
    auto hessians  = info.hessians();

    auto& k = spherical_joint_gradient_hessian_kernel();
    frame_stream().record(
        k(body_ids.cview(),
          rest_cs.cview(),
          strength_ratios.cview(),
          info.body_masses(),
          info.qs(),
          gradients.indices(),
          gradients.values(),
          static_cast<uint32_t>(gradients.total_extent()),
          static_cast<std::int32_t>(gradients.doublet_index_offset()),
          static_cast<std::int32_t>(gradients.doublet_count()),
          static_cast<std::int32_t>(gradients.total_doublet_count()),
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
          gradient_only ? 1u : 0u,
          static_cast<uint32_t>(HalfHessianSize),
          static_cast<uint32_t>(n))
            .dispatch(n));
}
}  // namespace uipc::backend::luisa
