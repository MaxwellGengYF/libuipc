#include <collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>
#include <array>
/*************************************************************************************************
 * T7.2b — InfoStacklessBVHSimplexTrajectoryFilter (port of
 * src/backends/cuda/collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.cu,
 * 1809 lines).
 *
 * The reference's structure, on T7.1b's `InfoStacklessBVH`:
 *
 *   detect          4 info swept-AABB builders (AABB + bid + cid) + 3 info builds +
 *                   the four keep-all searches through the reference's *split* API
 *                   (launch_query / launch_detect + prepare_query_result, so the four
 *                   `cpNum`s are downloaded in one transfer) + the four detect
 *                   predicate selection passes with the info flavour's `Dcd` branch
 *   filter_active   the four classifiers + the four valid-pair compactions, whose
 *                   counts are downloaded in one transfer (`selected_counts`)
 *   filter_toi      the four TOI kernels + the min-reduce
 *
 * MAPPING NOTES
 * -------------
 *  * the reference's `node_pred` (a node-level *cull* on `body_self_collision` and the
 *    contact mask) and its four leaf predicates are device functors; the port passes
 *    `sftd::KeepAllNodes{}`/`KeepAllLeaves{}` here (the ported predicates cannot read
 *    device arrays) and applies the leaf predicate as a selection pass -- see the
 *    header note for the proof that the cull is implied by the leaf predicate, so the
 *    surviving set is exact.
 *  * `collect_query_counts_kernel` is transcribed (a one-thread kernel); `query_counts`
 *    is `IndexT` and the batched read is `copy_to` after a `flush` (§7.10).
 *  * `DeviceSelect().If(..., selected_counts.data() + k, ...)` -> the port's
 *    `lla::select_valid_pairs` into `selected_counts.view(k, 1)` (the +k pointer
 *    arithmetic of the reference's `DeviceVar<IndexT>` is a subview here).
 *  * `cluster`/`m_cpNum` accesses go through the ported `QueryBuffer`'s public members
 *    (T7.1b).
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

/// `InfoStacklessBVHSimplexTrajectoryFilter_collect_query_counts_kernel`
auto& collect_query_counts_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
                                                lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
                                                lcrt::Buffer<IndexT>>(
        [](lcrt::BufferVar<IndexT> allp_codimp_count, lcrt::BufferVar<IndexT> codimp_alle_count,
           lcrt::BufferVar<IndexT> alle_alle_count, lcrt::BufferVar<IndexT> allp_allt_count,
           lcrt::BufferVar<IndexT> counts) noexcept
        {
            $if(lcrt::dispatch_x() == 0u)
            {
                counts->write(0u, allp_codimp_count->read(0u));
                counts->write(1u, codimp_alle_count->read(0u));
                counts->write(2u, alle_alle_count->read(0u));
                counts->write(3u, allp_allt_count->read(0u));
            };
        });
    return k;
}
}// namespace

REGISTER_SIM_SYSTEM(InfoStacklessBVHSimplexTrajectoryFilter);

void InfoStacklessBVHSimplexTrajectoryFilter::do_build(BuildInfo&)
{
    auto& config = world().scene().config();
    auto  method = config.find<std::string>("collision_detection/method");
    if(method->view()[0] != "info_stackless_bvh")
    {
        throw SimSystemException("Info stackless BVH unused");
    }

    m_impl.query_counts.resize(4);
    m_impl.selected_counts.resize(4);
}

void InfoStacklessBVHSimplexTrajectoryFilter::do_detect(DetectInfo& info)
{
    m_impl.detect(info);
}

void InfoStacklessBVHSimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info);
}

void InfoStacklessBVHSimplexTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}

lud::LCBufferView<LVector2i> InfoStacklessBVHSimplexTrajectoryFilter::candidate_PTs() const noexcept
{
    return m_impl.candidate_AllP_AllT_pairs.view();
}

lud::LCBufferView<LVector2i> InfoStacklessBVHSimplexTrajectoryFilter::candidate_EEs() const noexcept
{
    return m_impl.candidate_AllE_AllE_pairs.view();
}

lud::LCBufferView<Float> InfoStacklessBVHSimplexTrajectoryFilter::toi_PTs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    return m_impl.tois.cview(pp_size + pe_size, pt_size);
}

lud::LCBufferView<Float> InfoStacklessBVHSimplexTrajectoryFilter::toi_EEs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    auto ee_size = m_impl.candidate_AllE_AllE_pairs.size();
    return m_impl.tois.cview(pp_size + pe_size + pt_size, ee_size);
}

void InfoStacklessBVHSimplexTrajectoryFilter::Impl::detect(DetectInfo& info)
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

    auto launch_allp_codimp = [&](bool rebuild_query)
    {
        lbvh_CodimP.launch_query(point_aabbs.view(), point_bids.cview(), point_cids.cview(), cmts,
                                 sftd::KeepAllNodes{}, sftd::KeepAllLeaves{},
                                 raw_AllP_CodimP_pairs, rebuild_query);
    };
    auto launch_codimp_alle = [&](bool rebuild_query)
    {
        lbvh_E.launch_query(codim_point_aabbs.view(), codim_point_bids.cview(),
                            codim_point_cids.cview(), cmts,
                            sftd::KeepAllNodes{}, sftd::KeepAllLeaves{},
                            raw_CodimP_AllE_pairs, rebuild_query);
    };
    auto launch_alle_alle = [&]
    {
        lbvh_E.launch_detect(cmts, sftd::KeepAllNodes{}, sftd::KeepAllLeaves{},
                             raw_AllE_AllE_pairs);
    };
    auto launch_allp_allt = [&](bool rebuild_query)
    {
        lbvh_T.launch_query(point_aabbs.view(), point_bids.cview(), point_cids.cview(), cmts,
                            sftd::KeepAllNodes{}, sftd::KeepAllLeaves{},
                            raw_AllP_AllT_pairs, rebuild_query);
    };

    if(codimVs.size() > 0)
    {
        lbvh_CodimP.build(codim_point_aabbs.view(), codim_point_bids.cview(),
                          codim_point_cids.cview());
        launch_allp_codimp(true);
    }
    else
    {
        raw_AllP_CodimP_pairs.m_cpNum.view().fill(IndexT{0});
    }
    launch_codimp_alle(true);
    launch_alle_alle();
    launch_allp_allt(true);

    stream.record(collect_query_counts_kernel()(raw_AllP_CodimP_pairs.m_cpNum.cview(),
                                                raw_CodimP_AllE_pairs.m_cpNum.cview(),
                                                raw_AllE_AllE_pairs.m_cpNum.cview(),
                                                raw_AllP_AllT_pairs.m_cpNum.cview(),
                                                query_counts.view())
                      .dispatch(1u));

    std::array<IndexT, 4> host_counts{};
    stream.flush();
    query_counts.cview().copy_to(span<IndexT>{host_counts});
    lud::main_stream().synchronize();

    if(lbvh_CodimP.prepare_query_result(raw_AllP_CodimP_pairs, host_counts[0]))
        launch_allp_codimp(false);
    if(lbvh_E.prepare_query_result(raw_CodimP_AllE_pairs, host_counts[1]))
        launch_codimp_alle(false);
    if(lbvh_E.prepare_query_result(raw_AllE_AllE_pairs, host_counts[2]))
        launch_alle_alle();
    if(lbvh_T.prepare_query_result(raw_AllP_AllT_pairs, host_counts[3]))
        launch_allp_allt(false);

    // -----------------------------------------------------------------------
    // the reference's four leaf predicates, as selection passes (`Dcd == true`:
    // this variant is the one with the `alpha == 0` exact-distance branch)
    // -----------------------------------------------------------------------
    candidate_counts.resize(4u);

    // AllP and CodimP
    selection.record(
        stream,
        raw_AllP_CodimP_pairs.size(),
        [&](const lud::LBufferView<std::int32_t>& flags)
        {
            auto  n = static_cast<std::uint32_t>(raw_AllP_CodimP_pairs.size());
            auto& k = sftd::detect_AllP_CodimP_pass_kernel<true>();
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
            auto& k = sftd::detect_CodimP_AllE_pass_kernel<true>();
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
            auto& k = sftd::detect_AllE_AllE_pass_kernel<true>();
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
            auto& k = sftd::detect_AllP_AllT_pass_kernel<true>();
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

void InfoStacklessBVHSimplexTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info)
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
                                          selected_counts.view(0u, 1u), temp_PPs.size());

        lla::select_valid_pairs<LVector3i>(stream, temp_PEs.cview(), PEs.view(),
                                          selected_counts.view(1u, 1u), temp_PEs.size());

        lla::select_valid_pairs<LVector4i>(stream, temp_PTs.cview(), PTs.view(),
                                          selected_counts.view(2u, 1u), temp_PTs.size());

        lla::select_valid_pairs<LVector4i>(stream, temp_EEs.cview(), EEs.view(),
                                          selected_counts.view(3u, 1u), temp_EEs.size());

        std::array<std::uint32_t, 4> host_counts{};
        stream.flush();
        selected_counts.cview().copy_to(span<std::uint32_t>{host_counts});
        lud::main_stream().synchronize();

        IndexT PP_count = static_cast<IndexT>(host_counts[0]);
        IndexT PE_count = static_cast<IndexT>(host_counts[1]);
        IndexT PT_count = static_cast<IndexT>(host_counts[2]);
        IndexT EE_count = static_cast<IndexT>(host_counts[3]);

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

void InfoStacklessBVHSimplexTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
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
