#include <collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>
/*************************************************************************************************
 * T7.2b — StacklessBVHSimplexTrajectoryFilter (port of
 * src/backends/cuda/collision_detection/filters/lbvh_simplex_trajectory_filter.cu, 1544
 * lines).
 *
 * The kernels are the shared ones (details/simplex_trajectory_filter_kernels.h); this
 * file is the reference's structure around them:
 *
 *   detect            4 swept-AABB builders + 3 BVH builds + 4 keep-all traversals
 *                     (lbvh_CodimP.query, lbvh_E.query CodimP-AllE, lbvh_E.detect
 *                     AllE-AllE, lbvh_T.query AllP-AllT) + the 4 predicate *selection*
 *                     passes (see the header/kernels notes) + one batched count read
 *   filter_active     the 4 classifiers + the 4 `DeviceSelect::If` valid-pair
 *                     compactions (`lla::select_valid_pairs`)
 *   filter_toi        the 4 TOI kernels + the manager's min-reduce
 *
 * MAPPING NOTES
 * -------------
 *  * `k<<<best_grid_dim(n,k), best_block_dim(k)>>>(...)` -> `stream.record(kernel(...)
 *    .dispatch(round_up_block(n)))` with the `n > 0` guards the reference has.
 *  * `DeviceSelect().If(temp.data(), out.data(), count.data(), n, pred{pair(0) != -1})`
 *    -> `lla::select_valid_pairs<T>(stream, temp.cview(), out.view(), count.view(), n)`,
 *    which is the *same* stable compaction for the port's pair element types (the
 *    layer's `select_if` is instantiated for arithmetic scalars only).
 *  * `IndexT PP_count = selected_PP_count;` (an implicit device read) ->
 *    `lud::read_host_sync(stream, selected_PP_count.cview())`; the four reference
 *    reads are kept as four reads (the other three variants differ in how they batch
 *    them, and that difference is part of the four-way table).
 *  * `DeviceReduce().Min(tois.data(), info.toi().data(), tois.size())` with the
 *    reference's `if(tois.size())` guard -> `lla::device_reduce_min(...)`, and the
 *    reference's `else info.toi().fill(large_enough_toi)` stays the n == 0 branch
 *    (the ported reduce would otherwise *write* the identity for n == 0).
 *  * the `PrintDebugInfo` block is `constexpr false` in the reference; not ported.
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;
}// namespace

REGISTER_SIM_SYSTEM(StacklessBVHSimplexTrajectoryFilter);

void StacklessBVHSimplexTrajectoryFilter::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();
    auto  method = config.find<std::string>("collision_detection/method");
    if(method->view()[0] != "stackless_bvh")
    {
        throw SimSystemException("Stackless BVH unused");
    }
}

void StacklessBVHSimplexTrajectoryFilter::do_detect(DetectInfo& info)
{
    m_impl.detect(info);
}

void StacklessBVHSimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info);
}

void StacklessBVHSimplexTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}

lud::LCBufferView<LVector2i> StacklessBVHSimplexTrajectoryFilter::candidate_PTs() const noexcept
{
    return m_impl.candidate_AllP_AllT_pairs.view();
}

lud::LCBufferView<LVector2i> StacklessBVHSimplexTrajectoryFilter::candidate_EEs() const noexcept
{
    return m_impl.candidate_AllE_AllE_pairs.view();
}

lud::LCBufferView<Float> StacklessBVHSimplexTrajectoryFilter::toi_PTs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    return m_impl.tois.cview(pp_size + pe_size, pt_size);
}

lud::LCBufferView<Float> StacklessBVHSimplexTrajectoryFilter::toi_EEs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    auto ee_size = m_impl.candidate_AllE_AllE_pairs.size();
    return m_impl.tois.cview(pp_size + pe_size + pt_size, ee_size);
}

void StacklessBVHSimplexTrajectoryFilter::Impl::detect(DetectInfo& info)
{
    auto& stream = frame_stream();

    auto alpha   = info.alpha();
    auto Ps      = info.positions();
    auto dxs     = info.displacements();
    auto codimVs = info.codim_vertices();
    auto Vs      = info.surf_vertices();
    auto Es      = info.surf_edges();
    auto Fs      = info.surf_triangles();

    point_aabbs.resize(Vs.size());
    triangle_aabbs.resize(Fs.size());
    edge_aabbs.resize(Es.size());

    // build AABBs for codim vertices
    if(codimVs.size() > 0)
    {
        codim_point_aabbs.resize(codimVs.size());

        auto       n = static_cast<std::uint32_t>(codimVs.size());
        auto& k = sftd::detect_k1_kernel();
        stream.record(k(codimVs, Ps, dxs, codim_point_aabbs.view(), info.thicknesses(),
                        info.d_hats(), alpha, n)
                          .dispatch(sftd::round_up_block(n)));
    }

    // build AABBs for surf vertices (including codim vertices)
    {
        auto n = static_cast<std::uint32_t>(Vs.size());
        if(n > 0)
        {
            auto& k = sftd::detect_k2_kernel();
            stream.record(k(Vs, dxs, Ps, point_aabbs.view(), info.thicknesses(),
                            info.d_hats(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    // build AABBs for edges
    {
        auto n = static_cast<std::uint32_t>(Es.size());
        if(n > 0)
        {
            auto& k = sftd::detect_k3_kernel();
            stream.record(k(Es, Ps, edge_aabbs.view(), dxs, info.thicknesses(),
                            info.d_hats(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    // build AABBs for triangles
    {
        auto n = static_cast<std::uint32_t>(Fs.size());
        if(n > 0)
        {
            auto& k = sftd::detect_k4_kernel();
            stream.record(k(Fs, Ps, triangle_aabbs.view(), dxs, info.thicknesses(),
                            info.d_hats(), alpha, n)
                              .dispatch(sftd::round_up_block(n)));
        }
    }

    lbvh_E.build(edge_aabbs.view());
    lbvh_T.build(triangle_aabbs.view());

    if(codimVs.size() > 0)
    {
        // Use AllP to query CodimP
        {
            lbvh_CodimP.build(codim_point_aabbs.view());

            lbvh_CodimP.query(point_aabbs.view(),  // AllP
                              sftd::KeepAll{},
                              raw_AllP_CodimP_pairs);
        }

        // Use CodimP to query AllE
        if(codimVs.size())
        {
            lbvh_E.query(codim_point_aabbs.view(),
                         sftd::KeepAll{},
                         raw_CodimP_AllE_pairs);
        }
    }

    // Use AllE to query AllE
    {
        lbvh_E.detect(sftd::KeepAll{},
                      raw_AllE_AllE_pairs);
    }

    // Use AllP to query AllT
    {
        lbvh_T.query(point_aabbs.view(),
                     sftd::KeepAll{},
                     raw_AllP_AllT_pairs);
    }

    // -----------------------------------------------------------------------
    // The reference's detect-time per-pair predicates, as selection passes over
    // the keep-all traversal's output (see the header note).
    //
    // The four reported counts live in one four-slot buffer (the reference's
    // info_stackless_bvh variant uses exactly this layout); the port needs it in all
    // four variants because the counts come from the selection passes, not the BVH.
    // -----------------------------------------------------------------------
    candidate_counts.resize(4u);

    // AllP and CodimP
    selection.record(
        stream,
        raw_AllP_CodimP_pairs.size(),
        [&](const lud::LBufferView<std::int32_t>& flags)
        {
            auto       n = static_cast<std::uint32_t>(raw_AllP_CodimP_pairs.size());
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
            auto       n = static_cast<std::uint32_t>(raw_CodimP_AllE_pairs.size());
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
            auto       n = static_cast<std::uint32_t>(raw_AllE_AllE_pairs.size());
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
            auto       n = static_cast<std::uint32_t>(raw_AllP_AllT_pairs.size());
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

    // the reference reads its BVH's cpNum inside each query; the port's counts come
    // from the selection passes, so they are read back in one batch (an implicit
    // synchronous read in the reference, `§7.10` in the port)
    const auto candidate_count = sftd::read_candidate_counts(candidate_counts);
    candidate_AllP_CodimP_pairs.publish(candidate_count[0]);
    candidate_CodimP_AllE_pairs.publish(candidate_count[1]);
    candidate_AllE_AllE_pairs.publish(candidate_count[2]);
    candidate_AllP_AllT_pairs.publish(candidate_count[3]);
}

void StacklessBVHSimplexTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info)
{
    auto& stream = frame_stream();

    // we will filter-out the active pairs
    auto positions = info.positions();

    SizeT N_PCoimP  = candidate_AllP_CodimP_pairs.size();
    SizeT N_CodimPE = candidate_CodimP_AllE_pairs.size();
    SizeT N_PTs     = candidate_AllP_AllT_pairs.size();
    SizeT N_EEs     = candidate_AllE_AllE_pairs.size();

    // PT, EE, PT, PP can degenerate to PP
    temp_PPs.resize(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    // PT, EE, PT can degenerate to PE
    temp_PEs.resize(N_CodimPE + N_PTs + N_EEs);

    temp_PTs.resize(N_PTs);
    temp_EEs.resize(N_EEs);

    SizeT temp_PP_offset = 0;
    SizeT temp_PE_offset = 0;

    // AllP and CodimP
    if(N_PCoimP > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PCoimP);

        auto       n = static_cast<std::uint32_t>(candidate_AllP_CodimP_pairs.size());
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

        auto       n = static_cast<std::uint32_t>(candidate_CodimP_AllE_pairs.size());
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

void StacklessBVHSimplexTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
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
