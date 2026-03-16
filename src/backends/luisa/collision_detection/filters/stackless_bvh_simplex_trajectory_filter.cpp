#include <collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h>
#include <sim_engine.h>
#include <utils/distance/distance_flagged.h>
#include <utils/distance.h>
#include <utils/codim_thickness.h>
#include <utils/simplex_contact_mask_utils.h>
#include <utils/primitive_d_hat.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
constexpr bool PrintDebugInfo = false;

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
    m_impl.detect(info, world());
}

void StacklessBVHSimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info, world());
}

void StacklessBVHSimplexTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info, world());
}

void StacklessBVHSimplexTrajectoryFilter::Impl::detect(DetectInfo& info, WorldVisitor& world)
{
    auto& engine = static_cast<SimEngine&>(world.sim_engine());
    auto& device = engine.device();
    auto  stream = engine.compute_stream();

    auto alpha   = info.alpha();
    auto Ps      = info.positions();
    auto dxs     = info.displacements();
    auto codimVs = info.codim_vertices();
    auto Vs      = info.surf_vertices();
    auto Es      = info.surf_edges();
    auto Fs      = info.surf_triangles();

    // Resize AABB buffers
    if(point_aabbs.size() < Vs.size())
        point_aabbs = device.create_buffer<AABB>(Vs.size());
    if(triangle_aabbs.size() < Fs.size())
        triangle_aabbs = device.create_buffer<AABB>(Fs.size());
    if(edge_aabbs.size() < Es.size())
        edge_aabbs = device.create_buffer<AABB>(Es.size());

    // build AABBs for codim vertices
    if(codimVs.size() > 0)
    {
        if(codim_point_aabbs.size() < codimVs.size())
            codim_point_aabbs = device.create_buffer<AABB>(codimVs.size());

        auto codimVs_view = codimVs;
        auto Ps_view = Ps;
        auto dxs_view = dxs;
        auto thicknesses_view = info.thicknesses();
        auto d_hats_view = info.d_hats();
        SizeT codim_count = codimVs.size();

        Kernel1D codim_aabb_kernel = [&](BufferVar<IndexT> codim_vertices,
                                         BufferVar<Vector3> positions,
                                         BufferVar<Vector3> displacements,
                                         BufferVar<AABB> aabbs_out,
                                         BufferVar<Float> thicknesses,
                                         BufferVar<Float> d_hats,
                                         Float alpha_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < codim_count)
            {
                IndexT vI = codim_vertices.read(i);

                Float thickness = thicknesses.read(vI);
                Float d_hat_expansion = point_dcd_expansion(d_hats.read(vI));

                Vector3 pos = positions.read(vI);
                Vector3 pos_t = pos + displacements.read(vI) * alpha_val;

                AABB aabb;
                aabb.extend(pos.cast<float>()).extend(pos_t.cast<float>());

                float expand = d_hat_expansion + thickness;

                aabb.min().array() -= expand;
                aabb.max().array() += expand;
                aabbs_out.write(i, aabb);
            };
        };

        auto shader = device.compile(codim_aabb_kernel);
        stream << shader(codimVs_view,
                         Ps_view,
                         dxs_view,
                         codim_point_aabbs.view(),
                         thicknesses_view,
                         d_hats_view,
                         alpha)
                      .dispatch(codim_count);
    }

    // build AABBs for surf vertices (including codim vertices)
    {
        auto Vs_view = Vs;
        auto dxs_view = dxs;
        auto Ps_view = Ps;
        auto thicknesses_view = info.thicknesses();
        auto d_hats_view = info.d_hats();
        SizeT V_count = Vs.size();

        Kernel1D point_aabb_kernel = [&](BufferVar<IndexT> surf_vertices,
                                         BufferVar<Vector3> displacements,
                                         BufferVar<Vector3> positions,
                                         BufferVar<AABB> aabbs_out,
                                         BufferVar<Float> thicknesses,
                                         BufferVar<Float> d_hats,
                                         Float alpha_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < V_count)
            {
                IndexT vI = surf_vertices.read(i);

                Float thickness = thicknesses.read(vI);
                Float d_hat_expansion = point_dcd_expansion(d_hats.read(vI));

                Vector3 pos = positions.read(vI);
                Vector3 pos_t = pos + displacements.read(vI) * alpha_val;

                AABB aabb;
                aabb.extend(pos.cast<float>()).extend(pos_t.cast<float>());

                float expand = d_hat_expansion + thickness;

                aabb.min().array() -= expand;
                aabb.max().array() += expand;
                aabbs_out.write(i, aabb);
            };
        };

        auto shader = device.compile(point_aabb_kernel);
        stream << shader(Vs_view,
                         dxs_view,
                         Ps_view,
                         point_aabbs.view(),
                         thicknesses_view,
                         d_hats_view,
                         alpha)
                      .dispatch(V_count);
    }

    // build AABBs for edges
    {
        auto Es_view = Es;
        auto Ps_view = Ps;
        auto dxs_view = dxs;
        auto thicknesses_view = info.thicknesses();
        auto d_hats_view = info.d_hats();
        SizeT E_count = Es.size();

        Kernel1D edge_aabb_kernel = [&](BufferVar<Vector2i> surf_edges,
                                        BufferVar<Vector3> positions,
                                        BufferVar<AABB> aabbs_out,
                                        BufferVar<Vector3> displacements,
                                        BufferVar<Float> thicknesses,
                                        BufferVar<Float> d_hats,
                                        Float alpha_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < E_count)
            {
                Vector2i eI = surf_edges.read(i);

                Float thickness = edge_thickness(thicknesses.read(eI[0]), thicknesses.read(eI[1]));
                Float d_hat_expansion = edge_dcd_expansion(d_hats.read(eI[0]), d_hats.read(eI[1]));

                Vector3 pos0 = positions.read(eI[0]);
                Vector3 pos1 = positions.read(eI[1]);
                Vector3 pos0_t = pos0 + displacements.read(eI[0]) * alpha_val;
                Vector3 pos1_t = pos1 + displacements.read(eI[1]) * alpha_val;

                AABB aabb;

                aabb.extend(pos0.cast<float>())
                    .extend(pos1.cast<float>())
                    .extend(pos0_t.cast<float>())
                    .extend(pos1_t.cast<float>());

                float expand = d_hat_expansion + thickness;

                aabb.min().array() -= expand;
                aabb.max().array() += expand;
                aabbs_out.write(i, aabb);
            };
        };

        auto shader = device.compile(edge_aabb_kernel);
        stream << shader(Es_view,
                         Ps_view,
                         edge_aabbs.view(),
                         dxs_view,
                         thicknesses_view,
                         d_hats_view,
                         alpha)
                      .dispatch(E_count);
    }

    // build AABBs for triangles
    {
        auto Fs_view = Fs;
        auto Ps_view = Ps;
        auto dxs_view = dxs;
        auto thicknesses_view = info.thicknesses();
        auto d_hats_view = info.d_hats();
        SizeT F_count = Fs.size();

        Kernel1D triangle_aabb_kernel = [&](BufferVar<Vector3i> surf_triangles,
                                            BufferVar<Vector3> positions,
                                            BufferVar<AABB> aabbs_out,
                                            BufferVar<Vector3> displacements,
                                            BufferVar<Float> thicknesses,
                                            BufferVar<Float> d_hats,
                                            Float alpha_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < F_count)
            {
                Vector3i fI = surf_triangles.read(i);

                Float thickness = triangle_thickness(thicknesses.read(fI[0]),
                                                     thicknesses.read(fI[1]),
                                                     thicknesses.read(fI[2]));
                Float d_hat_expansion = triangle_dcd_expansion(
                    d_hats.read(fI[0]), d_hats.read(fI[1]), d_hats.read(fI[2]));

                Vector3 pos0 = positions.read(fI[0]);
                Vector3 pos1 = positions.read(fI[1]);
                Vector3 pos2 = positions.read(fI[2]);
                Vector3 pos0_t = pos0 + displacements.read(fI[0]) * alpha_val;
                Vector3 pos1_t = pos1 + displacements.read(fI[1]) * alpha_val;
                Vector3 pos2_t = pos2 + displacements.read(fI[2]) * alpha_val;

                AABB aabb;

                aabb.extend(pos0.cast<float>())
                    .extend(pos1.cast<float>())
                    .extend(pos2.cast<float>())
                    .extend(pos0_t.cast<float>())
                    .extend(pos1_t.cast<float>())
                    .extend(pos2_t.cast<float>());

                float expand = d_hat_expansion + thickness;

                aabb.min().array() -= expand;
                aabb.max().array() += expand;
                aabbs_out.write(i, aabb);
            };
        };

        auto shader = device.compile(triangle_aabb_kernel);
        stream << shader(Fs_view,
                         Ps_view,
                         triangle_aabbs.view(),
                         dxs_view,
                         thicknesses_view,
                         d_hats_view,
                         alpha)
                      .dispatch(F_count);
    }

    // Build Stackless BVH structures
    if(!lbvh_E)
        lbvh_E = std::make_unique<ThisBVH>(device);
    if(!lbvh_T)
        lbvh_T = std::make_unique<ThisBVH>(device);

    lbvh_E->build(edge_aabbs.view());
    lbvh_T->build(triangle_aabbs.view());

    if(codimVs.size() > 0)
    {
        // Use AllP to query CodimP
        {
            if(!lbvh_CodimP)
                lbvh_CodimP = std::make_unique<ThisBVH>(device);
            lbvh_CodimP->build(codim_point_aabbs.view());

            lbvh_CodimP->query(
                point_aabbs.view(),
                [&](IndexT i, IndexT j) -> bool { return true; },
                candidate_AllP_CodimP_pairs);
        }

        // Use CodimP to query AllE
        if(codimVs.size())
        {
            lbvh_E->query(
                codim_point_aabbs.view(),
                [&](IndexT i, IndexT j) -> bool { return true; },
                candidate_CodimP_AllE_pairs);
        }
    }

    // Use AllE to query AllE
    {
        lbvh_E->detect(
            [&](IndexT i, IndexT j) -> bool { return true; },
            candidate_AllE_AllE_pairs);
    }

    // Use AllP to query AllT
    {
        lbvh_T->query(
            point_aabbs.view(),
            [&](IndexT i, IndexT j) -> bool { return true; },
            candidate_AllP_AllT_pairs);
    }
}

void StacklessBVHSimplexTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info, WorldVisitor& world)
{
    auto& engine = static_cast<SimEngine&>(world.sim_engine());
    auto& device = engine.device();
    auto  stream = engine.compute_stream();

    // we will filter-out the active pairs
    auto positions = info.positions();

    SizeT N_PCoimP  = candidate_AllP_CodimP_pairs.size();
    SizeT N_CodimPE = candidate_CodimP_AllE_pairs.size();
    SizeT N_PTs     = candidate_AllP_AllT_pairs.size();
    SizeT N_EEs     = candidate_AllE_AllE_pairs.size();

    // PT, EE, PT, PP can degenerate to PP
    if(temp_PPs.size() < N_PCoimP + N_CodimPE + N_PTs + N_EEs)
        temp_PPs = device.create_buffer<Vector2i>(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    // PT, EE, PT can degenerate to PE
    if(temp_PEs.size() < N_CodimPE + N_PTs + N_EEs)
        temp_PEs = device.create_buffer<Vector3i>(N_CodimPE + N_PTs + N_EEs);

    if(temp_PTs.size() < N_PTs)
        temp_PTs = device.create_buffer<Vector4i>(N_PTs);
    if(temp_EEs.size() < N_EEs)
        temp_EEs = device.create_buffer<Vector4i>(N_EEs);

    SizeT temp_PP_offset = 0;
    SizeT temp_PE_offset = 0;

    // AllP and CodimP
    if(N_PCoimP > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PCoimP);

        auto positions_view = positions;
        auto PCodimP_pairs_view = candidate_AllP_CodimP_pairs.view();
        auto surf_vertices_view = info.surf_vertices();
        auto codim_vertices_view = info.codim_vertices();
        auto thicknesses_view = info.thicknesses();
        auto d_hats_view = info.d_hats();

        Kernel1D filter_pp_kernel = [&](BufferVar<Vector3> positions_buf,
                                        BufferVar<Vector2i> PCodimP_pairs,
                                        BufferVar<IndexT> surf_vertices,
                                        BufferVar<IndexT> codim_vertices,
                                        BufferVar<Float> thicknesses,
                                        BufferVar<Vector2i> temp_PPs_out,
                                        BufferVar<Float> d_hats) noexcept
        {
            auto i = dispatch_x();
            $if(i < N_PCoimP)
            {
                // default invalid
                temp_PPs_out.write(i, Vector2i{-1, -1});

                Vector2i indices = PCodimP_pairs.read(i);

                IndexT P0 = surf_vertices.read(indices[0]);
                IndexT P1 = codim_vertices.read(indices[1]);

                Vector3 V0 = positions_buf.read(P0);
                Vector3 V1 = positions_buf.read(P1);

                Float thickness = PP_thickness(thicknesses.read(P0), thicknesses.read(P1));
                Float d_hat = PP_d_hat(d_hats.read(P0), d_hats.read(P1));

                Vector2 range = D_range(thickness, d_hat);

                Float D;
                distance::point_point_distance2(V0, V1, D);

                $if(is_active_D(range, D))
                {
                    temp_PPs_out.write(i, Vector2i{cast<int>(P0), cast<int>(P1)});
                };
            };
        };

        auto shader = device.compile(filter_pp_kernel);
        stream << shader(positions_view,
                         PCodimP_pairs_view,
                         surf_vertices_view,
                         codim_vertices_view,
                         thicknesses_view,
                         PP_view,
                         d_hats_view)
                      .dispatch(N_PCoimP);

        temp_PP_offset += N_PCoimP;
    }

    // CodimP and AllE
    if(N_CodimPE > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_CodimPE);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_CodimPE);

        auto positions_view = positions;
        auto CodimP_AllE_pairs_view = candidate_CodimP_AllE_pairs.view();
        auto codim_vertices_view = info.codim_vertices();
        auto surf_edges_view = info.surf_edges();
        auto thicknesses_view = info.thicknesses();
        auto d_hats_view = info.d_hats();

        Kernel1D filter_pe_kernel = [&](BufferVar<Vector3> positions_buf,
                                        BufferVar<Vector2i> CodimP_AllE_pairs,
                                        BufferVar<IndexT> codim_veritces,
                                        BufferVar<Vector2i> surf_edges,
                                        BufferVar<Float> thicknesses,
                                        BufferVar<Vector2i> temp_PPs_out,
                                        BufferVar<Vector3i> temp_PEs_out,
                                        BufferVar<Float> d_hats) noexcept
        {
            auto i = dispatch_x();
            $if(i < N_CodimPE)
            {
                temp_PPs_out.write(i, Vector2i{-1, -1});
                temp_PEs_out.write(i, Vector3i{-1, -1, -1});

                Vector2i indices = CodimP_AllE_pairs.read(i);
                IndexT   V       = codim_veritces.read(indices[0]);
                Vector2i E       = surf_edges.read(indices[1]);

                Vector3i vIs = Vector3i{cast<int>(V), E[0], E[1]};
                Vector3 Ps[] = {positions_buf.read(vIs[0]), positions_buf.read(vIs[1]), positions_buf.read(vIs[2])};

                Float thickness = PE_thickness(
                    thicknesses.read(V), thicknesses.read(E[0]), thicknesses.read(E[1]));

                Float d_hat = PE_d_hat(d_hats.read(V), d_hats.read(E[0]), d_hats.read(E[1]));

                Vector3i flag =
                    distance::point_edge_distance_flag(Ps[0], Ps[1], Ps[2]);

                Vector2 range = D_range(thickness, d_hat);

                Float D;
                distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);

                $if(!is_active_D(range, D))
                {
                    $return;
                };

                Vector3i offsets;
                auto dim = distance::degenerate_point_edge(flag, offsets);

                $switch(dim)
                {
                    $case (2)
                    {
                        IndexT V0 = vIs[offsets[0]];
                        IndexT V1 = vIs[offsets[1]];
                        temp_PPs_out.write(i, Vector2i{cast<int>(V0), cast<int>(V1)});
                    };
                    $case (3)
                    {
                        temp_PEs_out.write(i, vIs);
                    };
                    $default
                    {
                        // unexpected degenerate case
                    };
                };
            };
        };

        auto shader = device.compile(filter_pe_kernel);
        stream << shader(positions_view,
                         CodimP_AllE_pairs_view,
                         codim_vertices_view,
                         surf_edges_view,
                         thicknesses_view,
                         PP_view,
                         PE_view,
                         d_hats_view)
                      .dispatch(N_CodimPE);

        temp_PP_offset += N_CodimPE;
        temp_PE_offset += N_CodimPE;
    }

    // AllP and AllT
    if(N_PTs > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PTs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_PTs);

        auto positions_view = positions;
        auto PT_pairs_view = candidate_AllP_AllT_pairs.view();
        auto surf_vertices_view = info.surf_vertices();
        auto surf_triangles_view = info.surf_triangles();
        auto thicknesses_view = info.thicknesses();
        auto d_hats_view = info.d_hats();
        auto temp_PTs_view = temp_PTs.view();

        Kernel1D filter_pt_kernel = [&](BufferVar<Vector3> positions_buf,
                                        BufferVar<Vector2i> PT_pairs,
                                        BufferVar<IndexT> surf_vertices,
                                        BufferVar<Vector3i> surf_triangles,
                                        BufferVar<Float> thicknesses,
                                        BufferVar<Vector2i> temp_PPs_out,
                                        BufferVar<Vector3i> temp_PEs_out,
                                        BufferVar<Vector4i> temp_PTs_out,
                                        BufferVar<Float> d_hats) noexcept
        {
            auto i = dispatch_x();
            $if(i < N_PTs)
            {
                temp_PPs_out.write(i, Vector2i{-1, -1});
                temp_PEs_out.write(i, Vector3i{-1, -1, -1});
                temp_PTs_out.write(i, Vector4i{-1, -1, -1, -1});

                Vector2i indices = PT_pairs.read(i);
                IndexT   V       = surf_vertices.read(indices[0]);
                Vector3i F       = surf_triangles.read(indices[1]);

                Vector4i vIs  = Vector4i{cast<int>(V), F[0], F[1], F[2]};
                Vector3  Ps[] = {positions_buf.read(vIs[0]),
                                 positions_buf.read(vIs[1]),
                                 positions_buf.read(vIs[2]),
                                 positions_buf.read(vIs[3])};

                Float thickness = PT_thickness(thicknesses.read(V),
                                               thicknesses.read(F[0]),
                                               thicknesses.read(F[1]),
                                               thicknesses.read(F[2]));

                Float d_hat =
                    PT_d_hat(d_hats.read(V), d_hats.read(F[0]), d_hats.read(F[1]), d_hats.read(F[2]));

                Vector4i flag =
                    distance::point_triangle_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

                Vector2 range = D_range(thickness, d_hat);

                Float D;
                distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

                $if(!is_active_D(range, D))
                {
                    $return;
                };

                Vector4i offsets;
                auto dim = distance::degenerate_point_triangle(flag, offsets);

                $switch(dim)
                {
                    $case (2)
                    {
                        IndexT V0 = vIs[offsets[0]];
                        IndexT V1 = vIs[offsets[1]];
                        temp_PPs_out.write(i, Vector2i{cast<int>(V0), cast<int>(V1)});
                    };
                    $case (3)
                    {
                        IndexT V0 = vIs[offsets[0]];
                        IndexT V1 = vIs[offsets[1]];
                        IndexT V2 = vIs[offsets[2]];
                        temp_PEs_out.write(i, Vector3i{cast<int>(V0), cast<int>(V1), cast<int>(V2)});
                    };
                    $case (4)
                    {
                        temp_PTs_out.write(i, vIs);
                    };
                    $default
                    {
                        // unexpected degenerate case
                    };
                };
            };
        };

        auto shader = device.compile(filter_pt_kernel);
        stream << shader(positions_view,
                         PT_pairs_view,
                         surf_vertices_view,
                         surf_triangles_view,
                         thicknesses_view,
                         PP_view,
                         PE_view,
                         temp_PTs_view,
                         d_hats_view)
                      .dispatch(N_PTs);

        temp_PP_offset += N_PTs;
        temp_PE_offset += N_PTs;
    }

    // AllE and AllE
    if(N_EEs > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_EEs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_EEs);

        auto positions_view = positions;
        auto rest_positions_view = info.rest_positions();
        auto EE_pairs_view = candidate_AllE_AllE_pairs.view();
        auto surf_edges_view = info.surf_edges();
        auto thicknesses_view = info.thicknesses();
        auto d_hats_view = info.d_hats();
        auto temp_EEs_view = temp_EEs.view();

        Kernel1D filter_ee_kernel = [&](BufferVar<Vector3> positions_buf,
                                        BufferVar<Vector3> rest_positions_buf,
                                        BufferVar<Vector2i> EE_pairs,
                                        BufferVar<Vector2i> surf_edges,
                                        BufferVar<Float> thicknesses,
                                        BufferVar<Vector2i> temp_PPs_out,
                                        BufferVar<Vector3i> temp_PEs_out,
                                        BufferVar<Vector4i> temp_EEs_out,
                                        BufferVar<Float> d_hats) noexcept
        {
            auto i = dispatch_x();
            $if(i < N_EEs)
            {
                temp_PPs_out.write(i, Vector2i{-1, -1});
                temp_PEs_out.write(i, Vector3i{-1, -1, -1});
                temp_EEs_out.write(i, Vector4i{-1, -1, -1, -1});

                Vector2i indices = EE_pairs.read(i);
                Vector2i E0      = surf_edges.read(indices[0]);
                Vector2i E1      = surf_edges.read(indices[1]);

                Vector4i vIs  = Vector4i{E0[0], E0[1], E1[0], E1[1]};
                Vector3  Ps[] = {positions_buf.read(vIs[0]),
                                 positions_buf.read(vIs[1]),
                                 positions_buf.read(vIs[2]),
                                 positions_buf.read(vIs[3])};

                Float thickness = EE_thickness(thicknesses.read(E0[0]),
                                               thicknesses.read(E0[1]),
                                               thicknesses.read(E1[0]),
                                               thicknesses.read(E1[1]));

                Float d_hat = EE_d_hat(
                    d_hats.read(E0[0]), d_hats.read(E0[1]), d_hats.read(E1[0]), d_hats.read(E1[1]));

                Vector2 range = D_range(thickness, d_hat);

                Vector4i flag =
                    distance::edge_edge_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

                Float D;
                distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

                // Corner case: exact/near-zero EE distance may appear for degenerate or
                // intersecting edge-edge candidates. Treat it as an active EE pair instead
                // of hard-aborting in the trajectory filter stage.
                $if(D <= range.x())
                {
                    temp_EEs_out.write(i, vIs);
                    $return;
                };

                $if(!is_active_D(range, D))
                {
                    $return;
                };

                Float eps_x;
                distance::edge_edge_mollifier_threshold(rest_positions_buf.read(vIs[0]),
                                                        rest_positions_buf.read(vIs[1]),
                                                        rest_positions_buf.read(vIs[2]),
                                                        rest_positions_buf.read(vIs[3]),
                                                        1e-3f,
                                                        eps_x);

                $if(distance::need_mollify(Ps[0], Ps[1], Ps[2], Ps[3], eps_x))
                {
                    temp_EEs_out.write(i, vIs);
                }
                $else
                {
                    Vector4i offsets;
                    auto dim = distance::degenerate_edge_edge(flag, offsets);

                    $switch(dim)
                    {
                        $case (2)
                        {
                            IndexT V0 = vIs[offsets[0]];
                            IndexT V1 = vIs[offsets[1]];
                            temp_PPs_out.write(i, Vector2i{cast<int>(V0), cast<int>(V1)});
                        };
                        $case (3)
                        {
                            IndexT V0 = vIs[offsets[0]];
                            IndexT V1 = vIs[offsets[1]];
                            IndexT V2 = vIs[offsets[2]];
                            temp_PEs_out.write(i, Vector3i{cast<int>(V0), cast<int>(V1), cast<int>(V2)});
                        };
                        $case (4)
                        {
                            temp_EEs_out.write(i, vIs);
                        };
                        $default
                        {
                            // unexpected degenerate case
                        };
                    };
                };
            };
        };

        auto shader = device.compile(filter_ee_kernel);
        stream << shader(positions_view,
                         rest_positions_view,
                         EE_pairs_view,
                         surf_edges_view,
                         thicknesses_view,
                         PP_view,
                         PE_view,
                         temp_EEs_view,
                         d_hats_view)
                      .dispatch(N_EEs);

        temp_PP_offset += N_EEs;
        temp_PE_offset += N_EEs;
    }

    UIPC_ASSERT(temp_PP_offset == temp_PPs.size(), "size mismatch");
    UIPC_ASSERT(temp_PE_offset == temp_PEs.size(), "size mismatch");

    // select the valid ones using host-side processing
    // LuisaCompute doesn't have CUB DeviceSelect equivalent
    {
        if(PPs.size() < temp_PPs.size())
            PPs = device.create_buffer<Vector2i>(temp_PPs.size());
        if(PEs.size() < temp_PEs.size())
            PEs = device.create_buffer<Vector3i>(temp_PEs.size());
        if(PTs.size() < temp_PTs.size())
            PTs = device.create_buffer<Vector4i>(temp_PTs.size());
        if(EEs.size() < temp_EEs.size())
            EEs = device.create_buffer<Vector4i>(temp_EEs.size());

        // Download to host, filter, upload back
        std::vector<Vector2i> h_temp_PPs(temp_PPs.size());
        std::vector<Vector3i> h_temp_PEs(temp_PEs.size());
        std::vector<Vector4i> h_temp_PTs(temp_PTs.size());
        std::vector<Vector4i> h_temp_EEs(temp_EEs.size());

        stream << temp_PPs.view().copy_to(h_temp_PPs.data())
               << temp_PEs.view().copy_to(h_temp_PEs.data())
               << temp_PTs.view().copy_to(h_temp_PTs.data())
               << temp_EEs.view().copy_to(h_temp_EEs.data())
               << synchronize();

        std::vector<Vector2i> h_PPs;
        std::vector<Vector3i> h_PEs;
        std::vector<Vector4i> h_PTs;
        std::vector<Vector4i> h_EEs;

        for(auto& PP : h_temp_PPs)
            if(PP[0] != -1)
                h_PPs.push_back(PP);
        for(auto& PE : h_temp_PEs)
            if(PE[0] != -1)
                h_PEs.push_back(PE);
        for(auto& PT : h_temp_PTs)
            if(PT[0] != -1)
                h_PTs.push_back(PT);
        for(auto& EE : h_temp_EEs)
            if(EE[0] != -1)
                h_EEs.push_back(EE);

        if(h_PPs.size() > PPs.size())
            PPs = device.create_buffer<Vector2i>(h_PPs.size());
        if(h_PEs.size() > PEs.size())
            PEs = device.create_buffer<Vector3i>(h_PEs.size());
        if(h_PTs.size() > PTs.size())
            PTs = device.create_buffer<Vector4i>(h_PTs.size());
        if(h_EEs.size() > EEs.size())
            EEs = device.create_buffer<Vector4i>(h_EEs.size());

        if(!h_PPs.empty())
            stream << PPs.view(0, h_PPs.size()).copy_from(h_PPs.data());
        if(!h_PEs.empty())
            stream << PEs.view(0, h_PEs.size()).copy_from(h_PEs.data());
        if(!h_PTs.empty())
            stream << PTs.view(0, h_PTs.size()).copy_from(h_PTs.data());
        if(!h_EEs.empty())
            stream << EEs.view(0, h_EEs.size()).copy_from(h_EEs.data());
        stream << synchronize();

        info.PPs(PPs.view(0, h_PPs.size()));
        info.PEs(PEs.view(0, h_PEs.size()));
        info.PTs(PTs.view(0, h_PTs.size()));
        info.EEs(EEs.view(0, h_EEs.size()));
    }

    if constexpr(PrintDebugInfo)
    {
        std::vector<Vector2i> PPs_host;
        std::vector<Vector3i> PEs_host;
        std::vector<Vector4i> PTs_host;
        std::vector<Vector4i> EEs_host;

        auto PPs_view = info.PPs();
        auto PEs_view = info.PEs();
        auto PTs_view = info.PTs();
        auto EEs_view = info.EEs();

        PPs_host.resize(PPs_view.size());
        PEs_host.resize(PEs_view.size());
        PTs_host.resize(PTs_view.size());
        EEs_host.resize(EEs_view.size());

        stream << PPs_view.copy_to(PPs_host.data())
               << PEs_view.copy_to(PEs_host.data())
               << PTs_view.copy_to(PTs_host.data())
               << EEs_view.copy_to(EEs_host.data())
               << synchronize();

        std::cout << "filter result:" << std::endl;

        for(auto&& PP : PPs_host)
        {
            std::cout << "PP: " << PP.transpose() << "\n";
        }

        for(auto&& PE : PEs_host)
        {
            std::cout << "PE: " << PE.transpose() << "\n";
        }

        for(auto&& PT : PTs_host)
        {
            std::cout << "PT: " << PT.transpose() << "\n";
        }

        for(auto&& EE : EEs_host)
        {
            std::cout << "EE: " << EE.transpose() << "\n";
        }

        std::cout << std::flush;
    }
}

void StacklessBVHSimplexTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info, WorldVisitor& world)
{
    auto& engine = static_cast<SimEngine&>(world.sim_engine());
    auto& device = engine.device();
    auto  stream = engine.compute_stream();

    auto toi_size =
        candidate_AllP_CodimP_pairs.size() + candidate_CodimP_AllE_pairs.size()
        + candidate_AllP_AllT_pairs.size() + candidate_AllE_AllE_pairs.size();

    if(tois.size() < toi_size)
        tois = device.create_buffer<Float>(toi_size);

    auto offset  = 0;
    auto PP_tois = tois.view(offset, candidate_AllP_CodimP_pairs.size());
    offset += candidate_AllP_CodimP_pairs.size();
    auto PE_tois = tois.view(offset, candidate_CodimP_AllE_pairs.size());
    offset += candidate_CodimP_AllE_pairs.size();
    auto PT_tois = tois.view(offset, candidate_AllP_AllT_pairs.size());
    offset += candidate_AllP_AllT_pairs.size();
    auto EE_tois = tois.view(offset, candidate_AllE_AllE_pairs.size());
    offset += candidate_AllE_AllE_pairs.size();

    UIPC_ASSERT(offset == toi_size, "size mismatch");

    // TODO: Now hard code the minimum separation coefficient
    constexpr Float eta = 0.1f;

    // TODO: Now hard code the maximum iteration
    constexpr SizeT max_iter = 1000;

    // large enough toi (>1)
    constexpr Float large_enough_toi = 1.1f;

    // AllP and CodimP
    if(candidate_AllP_CodimP_pairs.size() > 0)
    {
        auto PCodimP_pairs_view = candidate_AllP_CodimP_pairs.view();
        auto codim_vertices_view = info.codim_vertices();
        auto surf_vertices_view = info.surf_vertices();
        auto thicknesses_view = info.thicknesses();
        auto positions_view = info.positions();
        auto dxs_view = info.displacements();
        auto d_hats_view = info.d_hats();
        SizeT count = candidate_AllP_CodimP_pairs.size();

        Kernel1D toi_pp_kernel = [&](BufferVar<Float> PP_tois_out,
                                     BufferVar<Vector2i> PCodimP_pairs,
                                     BufferVar<IndexT> codim_vertices,
                                     BufferVar<IndexT> surf_vertices,
                                     BufferVar<Float> thicknesses,
                                     BufferVar<Vector3> positions,
                                     BufferVar<Vector3> dxs,
                                     BufferVar<Float> d_hats,
                                     Float alpha_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < count)
            {
                Vector2i indices = PCodimP_pairs.read(i);
                IndexT V0      = surf_vertices.read(indices[0]);
                IndexT V1      = codim_vertices.read(indices[1]);

                Float thickness = PP_thickness(thicknesses.read(V0), thicknesses.read(V1));
                Float d_hat = PP_d_hat(d_hats.read(V0), d_hats.read(V1));

                Vector3 VP0  = positions.read(V0);
                Vector3 VP1  = positions.read(V1);
                Vector3 dVP0 = alpha_val * dxs.read(V0);
                Vector3 dVP1 = alpha_val * dxs.read(V1);

                Float toi = large_enough_toi;

                bool faraway = !distance::point_point_ccd_broadphase(
                    VP0, VP1, dVP0, dVP1, d_hat + thickness);

                $if(faraway)
                {
                    PP_tois_out.write(i, toi);
                }
                $else
                {
                    bool hit = distance::point_point_ccd(
                        VP0, VP1, dVP0, dVP1, eta, thickness, max_iter, toi);

                    $if(!hit)
                    {
                        toi = large_enough_toi;
                    };

                    PP_tois_out.write(i, toi);
                };
            };
        };

        auto shader = device.compile(toi_pp_kernel);
        stream << shader(PP_tois,
                         PCodimP_pairs_view,
                         codim_vertices_view,
                         surf_vertices_view,
                         thicknesses_view,
                         positions_view,
                         dxs_view,
                         d_hats_view,
                         info.alpha())
                      .dispatch(count);
    }

    // CodimP and AllE
    if(candidate_CodimP_AllE_pairs.size() > 0)
    {
        auto CodimP_AllE_pairs_view = candidate_CodimP_AllE_pairs.view();
        auto codim_vertices_view = info.codim_vertices();
        auto thicknesses_view = info.thicknesses();
        auto surf_edges_view = info.surf_edges();
        auto Ps_view = info.positions();
        auto dxs_view = info.displacements();
        auto d_hats_view = info.d_hats();
        SizeT count = candidate_CodimP_AllE_pairs.size();

        Kernel1D toi_pe_kernel = [&](BufferVar<Float> PE_tois_out,
                                     BufferVar<Vector2i> CodimP_AllE_pairs,
                                     BufferVar<IndexT> codim_vertices,
                                     BufferVar<Float> thicknesses,
                                     BufferVar<Vector2i> surf_edges,
                                     BufferVar<Vector3> Ps,
                                     BufferVar<Vector3> dxs,
                                     BufferVar<Float> d_hats,
                                     Float alpha_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < count)
            {
                Vector2i indices = CodimP_AllE_pairs.read(i);
                IndexT   V       = codim_vertices.read(indices[0]);
                Vector2i E       = surf_edges.read(indices[1]);

                Float thickness = PE_thickness(
                    thicknesses.read(V), thicknesses.read(E[0]), thicknesses.read(E[1]));
                Float d_hat = PE_d_hat(d_hats.read(V), d_hats.read(E[0]), d_hats.read(E[1]));

                Vector3 VP  = Ps.read(V);
                Vector3 dVP = alpha_val * dxs.read(V);

                Vector3 EP0  = Ps.read(E[0]);
                Vector3 EP1  = Ps.read(E[1]);
                Vector3 dEP0 = alpha_val * dxs.read(E[0]);
                Vector3 dEP1 = alpha_val * dxs.read(E[1]);

                Float toi = large_enough_toi;

                bool faraway = !distance::point_edge_ccd_broadphase(
                    VP, EP0, EP1, dVP, dEP0, dEP1, d_hat + thickness);

                $if(faraway)
                {
                    PE_tois_out.write(i, toi);
                }
                $else
                {
                    bool hit = distance::point_edge_ccd(
                        VP, EP0, EP1, dVP, dEP0, dEP1, eta, thickness, max_iter, toi);

                    $if(!hit)
                    {
                        toi = large_enough_toi;
                    };

                    PE_tois_out.write(i, toi);
                };
            };
        };

        auto shader = device.compile(toi_pe_kernel);
        stream << shader(PE_tois,
                         CodimP_AllE_pairs_view,
                         codim_vertices_view,
                         thicknesses_view,
                         surf_edges_view,
                         Ps_view,
                         dxs_view,
                         d_hats_view,
                         info.alpha())
                      .dispatch(count);
    }

    // AllP and AllT
    if(candidate_AllP_AllT_pairs.size() > 0)
    {
        auto PT_pairs_view = candidate_AllP_AllT_pairs.view();
        auto surf_vertices_view = info.surf_vertices();
        auto surf_triangles_view = info.surf_triangles();
        auto thicknesses_view = info.thicknesses();
        auto Ps_view = info.positions();
        auto dxs_view = info.displacements();
        auto d_hats_view = info.d_hats();
        SizeT count = candidate_AllP_AllT_pairs.size();

        Kernel1D toi_pt_kernel = [&](BufferVar<Float> PT_tois_out,
                                     BufferVar<Vector2i> PT_pairs,
                                     BufferVar<IndexT> surf_vertices,
                                     BufferVar<Vector3i> surf_triangles,
                                     BufferVar<Float> thicknesses,
                                     BufferVar<Vector3> Ps,
                                     BufferVar<Vector3> dxs,
                                     BufferVar<Float> d_hats,
                                     Float alpha_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < count)
            {
                Vector2i indices = PT_pairs.read(i);
                IndexT   V       = surf_vertices.read(indices[0]);
                Vector3i F       = surf_triangles.read(indices[1]);

                Float thickness = PT_thickness(thicknesses.read(V),
                                               thicknesses.read(F[0]),
                                               thicknesses.read(F[1]),
                                               thicknesses.read(F[2]));
                Float d_hat =
                    PT_d_hat(d_hats.read(V), d_hats.read(F[0]), d_hats.read(F[1]), d_hats.read(F[2]));

                Vector3 VP  = Ps.read(V);
                Vector3 dVP = alpha_val * dxs.read(V);

                Vector3 FP0 = Ps.read(F[0]);
                Vector3 FP1 = Ps.read(F[1]);
                Vector3 FP2 = Ps.read(F[2]);

                Vector3 dFP0 = alpha_val * dxs.read(F[0]);
                Vector3 dFP1 = alpha_val * dxs.read(F[1]);
                Vector3 dFP2 = alpha_val * dxs.read(F[2]);

                Float toi = large_enough_toi;

                bool faraway = !distance::point_triangle_ccd_broadphase(
                    VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, d_hat + thickness);

                $if(faraway)
                {
                    PT_tois_out.write(i, toi);
                }
                $else
                {
                    bool hit = distance::point_triangle_ccd(
                        VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, eta, thickness, max_iter, toi);

                    $if(!hit)
                    {
                        toi = large_enough_toi;
                    };

                    PT_tois_out.write(i, toi);
                };
            };
        };

        auto shader = device.compile(toi_pt_kernel);
        stream << shader(PT_tois,
                         PT_pairs_view,
                         surf_vertices_view,
                         surf_triangles_view,
                         thicknesses_view,
                         Ps_view,
                         dxs_view,
                         d_hats_view,
                         info.alpha())
                      .dispatch(count);
    }

    // AllE and AllE
    if(candidate_AllE_AllE_pairs.size() > 0)
    {
        auto EE_pairs_view = candidate_AllE_AllE_pairs.view();
        auto surf_edges_view = info.surf_edges();
        auto thicknesses_view = info.thicknesses();
        auto Ps_view = info.positions();
        auto dxs_view = info.displacements();
        auto d_hats_view = info.d_hats();
        SizeT count = candidate_AllE_AllE_pairs.size();

        Kernel1D toi_ee_kernel = [&](BufferVar<Float> EE_tois_out,
                                     BufferVar<Vector2i> EE_pairs,
                                     BufferVar<Vector2i> surf_edges,
                                     BufferVar<Float> thicknesses,
                                     BufferVar<Vector3> Ps,
                                     BufferVar<Vector3> dxs,
                                     BufferVar<Float> d_hats,
                                     Float alpha_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < count)
            {
                Vector2i indices = EE_pairs.read(i);
                Vector2i E0      = surf_edges.read(indices[0]);
                Vector2i E1      = surf_edges.read(indices[1]);

                Float thickness = EE_thickness(thicknesses.read(E0[0]),
                                               thicknesses.read(E0[1]),
                                               thicknesses.read(E1[0]),
                                               thicknesses.read(E1[1]));

                Float d_hat = EE_d_hat(
                    d_hats.read(E0[0]), d_hats.read(E0[1]), d_hats.read(E1[0]), d_hats.read(E1[1]));

                Vector3 EP0  = Ps.read(E0[0]);
                Vector3 EP1  = Ps.read(E0[1]);
                Vector3 dEP0 = alpha_val * dxs.read(E0[0]);
                Vector3 dEP1 = alpha_val * dxs.read(E0[1]);

                Vector3 EP2  = Ps.read(E1[0]);
                Vector3 EP3  = Ps.read(E1[1]);
                Vector3 dEP2 = alpha_val * dxs.read(E1[0]);
                Vector3 dEP3 = alpha_val * dxs.read(E1[1]);

                Float toi = large_enough_toi;

                bool faraway = !distance::edge_edge_ccd_broadphase(
                    EP0, EP1, EP2, EP3, dEP0, dEP1, dEP2, dEP3, d_hat + thickness);

                $if(faraway)
                {
                    EE_tois_out.write(i, toi);
                }
                $else
                {
                    bool hit = distance::edge_edge_ccd(
                        EP0, EP1, EP2, EP3, dEP0, dEP1, dEP2, dEP3, eta, thickness, max_iter, toi);

                    $if(!hit)
                    {
                        toi = large_enough_toi;
                    };

                    EE_tois_out.write(i, toi);
                };
            };
        };

        auto shader = device.compile(toi_ee_kernel);
        stream << shader(EE_tois,
                         EE_pairs_view,
                         surf_edges_view,
                         thicknesses_view,
                         Ps_view,
                         dxs_view,
                         d_hats_view,
                         info.alpha())
                      .dispatch(count);
    }

    if(tois.size() > 0)
    {
        // get min toi using host reduction
        std::vector<Float> h_tois(tois.size());
        stream << tois.view().copy_to(h_tois.data())
               << synchronize();
        
        Float min_toi = large_enough_toi;
        for(auto t : h_tois)
            min_toi = std::min(min_toi, t);
        
        stream << info.toi().fill(min_toi);
    }
    else
    {
        info.toi().fill(large_enough_toi);
    }
}

}  // namespace uipc::backend::luisa
