#include <collision_detection/filters/info_stackless_bvh_v0_simplex_trajectory_filter.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>
#include <array>
/*************************************************************************************************
 * T7.2b — InfoStacklessBVHV0SimplexTrajectoryFilter (port of
 * src/backends/cuda/collision_detection/filters/info_stackless_bvh_v0_simplex_trajectory_filter.cu,
 * 1731 lines).
 *
 * The v0 flavour, on T7.1b's `InfoStacklessBVHV0`.  Read the v1 flavour's file note
 * (info_stackless_bvh_simplex_trajectory_filter.cpp) first: it carries the
 * predicate/device-data decision and the four-way table.  v0's two differences:
 *
 *  1. no `alpha == 0` DCD branch in its four leaf predicates (the shared pass kernels
 *     are instantiated with `Dcd == false`, exactly like the LBVH/stackless flavours)
 *     -- so v0's candidate sets must equal the stackless flavour's on the same scene;
 *  2. the reference calls the *plain* `InfoStacklessBVHV0::query`/`detect` (each doing
 *     its own retry read-back, hence one host read per query) and reads its four
 *     filter-active counts from four separate `DeviceVar`s, where v1 splits the query
 *     to batch the four counts into one transfer.
 *
 * MAPPING NOTES (as v1, minus the batching machinery)
 * ---------------------------------------------------
 *  * the reference's `node_pred`/leaf predicates -> `sftd::KeepAllNodes{}` /
 *    `sftd::KeepAllLeaves{}` + the selection passes (the cull is implied by the leaf
 *    predicate; see the v1 header note);
 *  * the reference's `if(Es.size() > 0)` / `if(Fs.size() > 0)` guards around the
 *    AllE-AllE and AllP-AllT searches are transcribed (v1 has no such guards);
 *  * `DeviceSelect().If(..., selected_PP_count.data(), ...)` ->
 *    `lla::select_valid_pairs` into the four single-slot `selected_XX_count` buffers.
 *************************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(InfoStacklessBVHV0SimplexTrajectoryFilter);

void InfoStacklessBVHV0SimplexTrajectoryFilter::do_build(BuildInfo&)
{
    auto& config = world().scene().config();
    auto  method = config.find<std::string>("collision_detection/method");
    if(method->view()[0] != "info_stackless_bvh_v0")
    {
        throw SimSystemException("Info stackless BVH V0 unused");
    }

}

void InfoStacklessBVHV0SimplexTrajectoryFilter::do_detect(DetectInfo& info)
{
    m_impl.detect(info);
}

void InfoStacklessBVHV0SimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info);
}

void InfoStacklessBVHV0SimplexTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}

lud::LCBufferView<LVector2i> InfoStacklessBVHV0SimplexTrajectoryFilter::candidate_PTs() const noexcept
{
    return m_impl.candidate_AllP_AllT_pairs.view();
}

lud::LCBufferView<LVector2i> InfoStacklessBVHV0SimplexTrajectoryFilter::candidate_EEs() const noexcept
{
    return m_impl.candidate_AllE_AllE_pairs.view();
}

lud::LCBufferView<Float> InfoStacklessBVHV0SimplexTrajectoryFilter::toi_PTs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    return m_impl.tois.cview(pp_size + pe_size, pt_size);
}

lud::LCBufferView<Float> InfoStacklessBVHV0SimplexTrajectoryFilter::toi_EEs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    auto ee_size = m_impl.candidate_AllE_AllE_pairs.size();
    return m_impl.tois.cview(pp_size + pe_size + pt_size, ee_size);
}

void InfoStacklessBVHV0SimplexTrajectoryFilter::Impl::detect(DetectInfo& info)
{
    auto& stream = frame_stream();

    auto alpha   = info.alpha();
    auto Ps      = info.positions();
    auto dxs     = info.displacements();
    auto codimVs = info.codim_vertices();
    auto Vs      = info.surf_vertices();
    auto Es      = info.surf_edges();
    auto Fs      = info.surf_triangles();
    auto v2bs    = info.v2b();
    auto body_self_collisions = info.body_self_collision();
    auto contact_element_ids  = info.contact_element_ids();
    auto cmts                 = info.contact_mask_tabular();

    point_aabbs.resize(Vs.size());
    triangle_aabbs.resize(Fs.size());
    edge_aabbs.resize(Es.size());
    point_bids.resize(Vs.size());
    triangle_bids.resize(Fs.size());
    edge_bids.resize(Es.size());
    point_cids.resize(Vs.size());
    triangle_cids.resize(Fs.size());
    edge_cids.resize(Es.size());

    // build AABBs for codim vertices
    if(codimVs.size() > 0)
    {
        codim_point_aabbs.resize(codimVs.size());
        codim_point_bids.resize(codimVs.size());
        codim_point_cids.resize(codimVs.size());

        auto       n = static_cast<std::uint32_t>(codimVs.size());
        auto&      k = sftd::detect_info_k1_kernel();
        stream.record(k(codimVs, Ps, dxs, codim_point_aabbs.view(), v2bs,
                        contact_element_ids, codim_point_bids.view(), codim_point_cids.view(),
                        info.thicknesses(), info.d_hats(), alpha, n)
                          .dispatch(sftd::round_up_block(n)));
    }

    // build AABBs for surf vertices (including codim vertices)
    {
        auto n = static_cast<std::uint32_t>(Vs.size());
        if(n > 0)
        {
            auto& k = sftd::detect_info_k2_kernel();
            stream.record(k(Vs, dxs, Ps, point_aabbs.view(), v2bs, contact_element_ids,
                            point_bids.view(), point_cids.view(), info.thicknesses(),
                            info.d_hats(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    // build AABBs for edges
    {
        auto n = static_cast<std::uint32_t>(Es.size());
        if(n > 0)
        {
            auto& k = sftd::detect_info_k3_kernel();
            stream.record(k(Es, Ps, edge_aabbs.view(), v2bs, contact_element_ids,
                            edge_bids.view(), edge_cids.view(), dxs, info.thicknesses(),
                            info.d_hats(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    // build AABBs for triangles
    {
        auto n = static_cast<std::uint32_t>(Fs.size());
        if(n > 0)
        {
            auto& k = sftd::detect_info_k4_kernel();
            stream.record(k(Fs, Ps, triangle_aabbs.view(), v2bs, contact_element_ids,
                            triangle_bids.view(), triangle_cids.view(), dxs,
                            info.thicknesses(), info.d_hats(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    lbvh_E.build(edge_aabbs.view(), edge_bids.cview(), edge_cids.cview());
    lbvh_T.build(triangle_aabbs.view(), triangle_bids.cview(), triangle_cids.cview());

    // Use AllP to query CodimP
    if(codimVs.size() > 0)
    {
        lbvh_CodimP.build(codim_point_aabbs.view(), codim_point_bids.cview(),
                          codim_point_cids.cview());

        lbvh_CodimP.query(point_aabbs.view(), point_bids.cview(), point_cids.cview(), cmts,
                          sftd::KeepAllNodes{}, sftd::KeepAllLeaves{},
                          raw_AllP_CodimP_pairs);
    }

    // Use CodimP to query AllE
    if(codimVs.size() > 0)
    {
        lbvh_E.query(codim_point_aabbs.view(), codim_point_bids.cview(),
                     codim_point_cids.cview(), cmts,
                     sftd::KeepAllNodes{}, sftd::KeepAllLeaves{},
                     raw_CodimP_AllE_pairs);
    }

    // Use AllE to query AllE
    if(Es.size() > 0)
    {
        lbvh_E.detect(cmts, sftd::KeepAllNodes{}, sftd::KeepAllLeaves{},
                      raw_AllE_AllE_pairs);
    }

    // Use AllP to query AllT
    if(Fs.size() > 0)
    {
        lbvh_T.query(point_aabbs.view(), point_bids.cview(), point_cids.cview(), cmts,
                     sftd::KeepAllNodes{}, sftd::KeepAllLeaves{},
                     raw_AllP_AllT_pairs);
    }

    // -----------------------------------------------------------------------
    // the reference's four leaf predicates, as selection passes (`Dcd == false`:
    // v0 has no `alpha == 0` exact-distance branch -- unlike the v1 flavour)
    // -----------------------------------------------------------------------
    candidate_counts.resize(4u);

    // AllP and CodimP
    selection.record(
        stream,
        raw_AllP_CodimP_pairs.size(),
        [&](const lud::LBufferView<std::int32_t>& flags)
        {
            auto  n = static_cast<std::uint32_t>(raw_AllP_CodimP_pairs.size());
            auto& k = sftd::detect_AllP_CodimP_pass_kernel<false>();
            stream.record(k(raw_AllP_CodimP_pairs.view(), Vs, codimVs, Ps, dxs,
                            info.thicknesses(), info.dimensions(),
                            info.contact_element_ids(), info.contact_mask_tabular(),
                            info.subscene_element_ids(), info.subscene_mask_tabular(),
                            info.v2b(), info.body_self_collision(), info.d_hats(), flags,
                            info.contact_mask_size(), info.subscene_mask_size(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        },
        raw_AllP_CodimP_pairs.view(), candidate_AllP_CodimP_pairs,
        candidate_counts.view(0u, 1u));

    // CodimP and AllE
    selection.record(
        stream,
        raw_CodimP_AllE_pairs.size(),
        [&](const lud::LBufferView<std::int32_t>& flags)
        {
            auto  n = static_cast<std::uint32_t>(raw_CodimP_AllE_pairs.size());
            auto& k = sftd::detect_CodimP_AllE_pass_kernel<false>();
            stream.record(k(raw_CodimP_AllE_pairs.view(), codimVs, Es, Ps, dxs,
                            info.thicknesses(), info.contact_element_ids(),
                            info.contact_mask_tabular(), info.subscene_element_ids(),
                            info.subscene_mask_tabular(), info.v2b(),
                            info.body_self_collision(), info.d_hats(), flags,
                            info.contact_mask_size(), info.subscene_mask_size(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        },
        raw_CodimP_AllE_pairs.view(), candidate_CodimP_AllE_pairs,
        candidate_counts.view(1u, 1u));

    // AllE and AllE
    selection.record(
        stream,
        raw_AllE_AllE_pairs.size(),
        [&](const lud::LBufferView<std::int32_t>& flags)
        {
            auto  n = static_cast<std::uint32_t>(raw_AllE_AllE_pairs.size());
            auto& k = sftd::detect_AllE_AllE_pass_kernel<false>();
            stream.record(k(raw_AllE_AllE_pairs.view(), Es, Ps, dxs, info.thicknesses(),
                            info.contact_element_ids(), info.contact_mask_tabular(),
                            info.subscene_element_ids(), info.subscene_mask_tabular(),
                            info.v2b(), info.body_self_collision(), info.d_hats(), flags,
                            info.contact_mask_size(), info.subscene_mask_size(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        },
        raw_AllE_AllE_pairs.view(), candidate_AllE_AllE_pairs,
        candidate_counts.view(2u, 1u));

    // AllP and AllT
    selection.record(
        stream,
        raw_AllP_AllT_pairs.size(),
        [&](const lud::LBufferView<std::int32_t>& flags)
        {
            auto  n = static_cast<std::uint32_t>(raw_AllP_AllT_pairs.size());
            auto& k = sftd::detect_AllP_AllT_pass_kernel<false>();
            stream.record(k(raw_AllP_AllT_pairs.view(), Vs, Fs, Ps, dxs,
                            info.thicknesses(), info.contact_element_ids(),
                            info.contact_mask_tabular(), info.subscene_element_ids(),
                            info.subscene_mask_tabular(), info.v2b(),
                            info.body_self_collision(), info.d_hats(), flags,
                            info.contact_mask_size(), info.subscene_mask_size(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        },
        raw_AllP_AllT_pairs.view(), candidate_AllP_AllT_pairs,
        candidate_counts.view(3u, 1u));

    const auto candidate_count = sftd::read_candidate_counts(candidate_counts);
    candidate_AllP_CodimP_pairs.publish(candidate_count[0]);
    candidate_CodimP_AllE_pairs.publish(candidate_count[1]);
    candidate_AllE_AllE_pairs.publish(candidate_count[2]);
    candidate_AllP_AllT_pairs.publish(candidate_count[3]);
}

void InfoStacklessBVHV0SimplexTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info)
{
    auto& stream = frame_stream();

    auto positions = info.positions();

    SizeT N_PCoimP  = candidate_AllP_CodimP_pairs.size();
    SizeT N_CodimPE = candidate_CodimP_AllE_pairs.size();
    SizeT N_PTs     = candidate_AllP_AllT_pairs.size();
    SizeT N_EEs     = candidate_AllE_AllE_pairs.size();

    temp_PPs.resize(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    temp_PEs.resize(N_CodimPE + N_PTs + N_EEs);

    temp_PTs.resize(N_PTs);
    temp_EEs.resize(N_EEs);

    SizeT temp_PP_offset = 0;
    SizeT temp_PE_offset = 0;

    // AllP and CodimP
    if(N_PCoimP > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PCoimP);

        auto  n = static_cast<std::uint32_t>(candidate_AllP_CodimP_pairs.size());
        auto& k = sftd::filter_active_k1_kernel();
        stream.record(k(positions, candidate_AllP_CodimP_pairs.view(), info.surf_vertices(),
                        info.codim_vertices(), info.thicknesses(), PP_view, info.d_hats(),
                        n)
                          .dispatch(sftd::round_up_block(n)));

        temp_PP_offset += N_PCoimP;
    }
    // CodimP and AllE
    if(N_CodimPE > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_CodimPE);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_CodimPE);

        auto  n = static_cast<std::uint32_t>(candidate_CodimP_AllE_pairs.size());
        auto& k = sftd::filter_active_k2_kernel();
        stream.record(k(positions, candidate_CodimP_AllE_pairs.view(),
                        info.codim_vertices(), info.surf_edges(), info.thicknesses(),
                        PP_view, PE_view, info.d_hats(), n)
                          .dispatch(sftd::round_up_block(n)));

        temp_PP_offset += N_CodimPE;
        temp_PE_offset += N_CodimPE;
    }

    // AllP and AllT
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PTs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_PTs);

        auto n = static_cast<std::uint32_t>(candidate_AllP_AllT_pairs.size());
        if(n > 0)
        {
            auto& k = sftd::filter_active_k3_kernel();
            stream.record(k(positions, candidate_AllP_AllT_pairs.view(),
                            info.surf_vertices(), info.surf_triangles(),
                            info.thicknesses(), PP_view, PE_view, temp_PTs.view(),
                            info.d_hats(), n)
                              .dispatch(sftd::round_up_block(n)));
        }

        temp_PP_offset += N_PTs;
        temp_PE_offset += N_PTs;
    }
    // AllE and AllE
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_EEs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_EEs);

        auto n = static_cast<std::uint32_t>(candidate_AllE_AllE_pairs.size());
        if(n > 0)
        {
            auto& k = sftd::filter_active_k4_kernel();
            stream.record(k(positions, info.rest_positions(),
                            candidate_AllE_AllE_pairs.view(), info.surf_edges(),
                            info.thicknesses(), PP_view, PE_view, temp_EEs.view(),
                            info.d_hats(), n)
                              .dispatch(sftd::round_up_block(n)));
        }

        temp_PP_offset += N_EEs;
        temp_PE_offset += N_EEs;
    }

    UIPC_ASSERT(temp_PP_offset == temp_PPs.size(), "size mismatch");
    UIPC_ASSERT(temp_PE_offset == temp_PEs.size(), "size mismatch");

    {  // select the valid ones
        PPs.resize(temp_PPs.size());
        PEs.resize(temp_PEs.size());
        PTs.resize(temp_PTs.size());
        EEs.resize(temp_EEs.size());

        lla::select_valid_pairs<LVector2i>(stream, temp_PPs.cview(), PPs.view(),
                                          selected_PP_count.view(), temp_PPs.size());

        lla::select_valid_pairs<LVector3i>(stream, temp_PEs.cview(), PEs.view(),
                                          selected_PE_count.view(), temp_PEs.size());

        lla::select_valid_pairs<LVector4i>(stream, temp_PTs.cview(), PTs.view(),
                                          selected_PT_count.view(), temp_PTs.size());

        lla::select_valid_pairs<LVector4i>(stream, temp_EEs.cview(), EEs.view(),
                                          selected_EE_count.view(), temp_EEs.size());

        IndexT PP_count =
            static_cast<IndexT>(lud::read_host_sync(stream, selected_PP_count.cview()));
        IndexT PE_count =
            static_cast<IndexT>(lud::read_host_sync(stream, selected_PE_count.cview()));
        IndexT PT_count =
            static_cast<IndexT>(lud::read_host_sync(stream, selected_PT_count.cview()));
        IndexT EE_count =
            static_cast<IndexT>(lud::read_host_sync(stream, selected_EE_count.cview()));

        PPs.resize(static_cast<SizeT>(PP_count));
        PEs.resize(static_cast<SizeT>(PE_count));
        PTs.resize(static_cast<SizeT>(PT_count));
        EEs.resize(static_cast<SizeT>(EE_count));
    }

    info.PPs(PPs.cview());
    info.PEs(PEs.cview());
    info.PTs(PTs.cview());
    info.EEs(EEs.cview());
}

void InfoStacklessBVHV0SimplexTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
{
    auto& stream = frame_stream();

    auto toi_size = candidate_AllP_CodimP_pairs.size() + candidate_CodimP_AllE_pairs.size()
                    + candidate_AllP_AllT_pairs.size() + candidate_AllE_AllE_pairs.size();

    tois.resize(toi_size);

    SizeT offset  = 0;
    auto  PP_tois = tois.view(offset, candidate_AllP_CodimP_pairs.size());
    offset += candidate_AllP_CodimP_pairs.size();
    auto PE_tois = tois.view(offset, candidate_CodimP_AllE_pairs.size());
    offset += candidate_CodimP_AllE_pairs.size();
    auto PT_tois = tois.view(offset, candidate_AllP_AllT_pairs.size());
    offset += candidate_AllP_AllT_pairs.size();
    auto EE_tois = tois.view(offset, candidate_AllE_AllE_pairs.size());
    offset += candidate_AllE_AllE_pairs.size();

    UIPC_ASSERT(offset == toi_size, "size mismatch");

    // AllP and CodimP
    {
        auto n = static_cast<std::uint32_t>(candidate_AllP_CodimP_pairs.size());
        if(n > 0)
        {
            auto& k = sftd::filter_toi_k1_kernel();
            stream.record(k(PP_tois, candidate_AllP_CodimP_pairs.view(),
                            info.codim_vertices(), info.surf_vertices(),
                            info.thicknesses(), info.positions(), info.displacements(),
                            info.d_hats(), info.toi_safety_margin(), info.alpha(), n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    // CodimP and AllE
    {
        auto n = static_cast<std::uint32_t>(candidate_CodimP_AllE_pairs.size());
        if(n > 0)
        {
            auto& k = sftd::filter_toi_k2_kernel();
            stream.record(k(PE_tois, candidate_CodimP_AllE_pairs.view(),
                            info.codim_vertices(), info.thicknesses(), info.surf_edges(),
                            info.positions(), info.displacements(), info.d_hats(),
                            info.toi_safety_margin(), info.alpha(), n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    // AllP and AllT
    {
        auto n = static_cast<std::uint32_t>(candidate_AllP_AllT_pairs.size());
        if(n > 0)
        {
            auto& k = sftd::filter_toi_k3_kernel();
            stream.record(k(PT_tois, candidate_AllP_AllT_pairs.view(), info.surf_vertices(),
                            info.surf_triangles(), info.thicknesses(), info.positions(),
                            info.displacements(), info.d_hats(), info.toi_safety_margin(),
                            info.alpha(), n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    // AllE and AllE
    {
        auto n = static_cast<std::uint32_t>(candidate_AllE_AllE_pairs.size());
        if(n > 0)
        {
            auto& k = sftd::filter_toi_k4_kernel();
            stream.record(k(EE_tois, candidate_AllE_AllE_pairs.view(), info.surf_edges(),
                            info.thicknesses(), info.positions(), info.displacements(),
                            info.d_hats(), info.toi_safety_margin(), info.alpha(), n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    if(tois.size())
    {
        // get min toi
        lla::device_reduce_min(stream, tois.cview(), info.toi(), tois.size());
    }
    else
    {
        info.toi().fill(sftd::kLargeEnoughToi, &stream);
    }
}
}// namespace uipc::backend::luisa
