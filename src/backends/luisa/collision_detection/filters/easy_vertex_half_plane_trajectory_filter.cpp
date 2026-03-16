#include <collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h>
#include <utils/codim_thickness.h>
#include <sim_engine.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(EasyVertexHalfPlaneTrajectoryFilter);

constexpr bool PrintDebugInfo = false;

void EasyVertexHalfPlaneTrajectoryFilter::do_detect(DetectInfo& info)
{
    // do nothing
}

void EasyVertexHalfPlaneTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info, world());
}

void EasyVertexHalfPlaneTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info, world());
}

void EasyVertexHalfPlaneTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info, WorldVisitor& world)
{
    auto& engine = static_cast<SimEngine&>(world.sim_engine());
    auto& device = engine.device();
    auto  stream = engine.compute_stream();

    auto query = [&]
    {
        // Reset collision counter to 0
        IndexT zero = 0;
        stream << num_collisions.copy_from(&zero, 1);

        auto plane_vertex_offset = info.half_plane_vertex_offset();
        auto surf_vertices_view = info.surf_vertices();
        auto positions_view = info.positions();
        auto thicknesses_view = info.thicknesses();
        auto contact_element_ids_view = info.contact_element_ids();
        auto subscene_element_ids_view = info.subscene_element_ids();
        auto contact_mask_tabular_view = info.contact_mask_tabular();
        auto subscene_mask_tabular_view = info.subscene_mask_tabular();
        auto plane_positions_view = info.plane_positions();
        auto plane_normals_view = info.plane_normals();
        auto d_hats_view = info.d_hats();
        auto PHs_view = PHs.view();
        SizeT max_count = PHs.size();
        SizeT surf_vert_count = surf_vertices_view.size();
        SizeT plane_count = plane_positions_view.size();
        
        // Compute table dimensions (tables are square N x N)
        SizeT contact_table_size = contact_mask_tabular_view.size();
        SizeT contact_table_dim = static_cast<SizeT>(std::sqrt(static_cast<double>(contact_table_size)));
        SizeT subscene_table_size = subscene_mask_tabular_view.size();
        SizeT subscene_table_dim = static_cast<SizeT>(std::sqrt(static_cast<double>(subscene_table_size)));

        Kernel1D detect_kernel = [&](BufferVar<IndexT> surf_vertices,
                                     BufferVar<Vector3> positions,
                                     BufferVar<Float> thicknesses,
                                     BufferVar<IndexT> contact_element_ids,
                                     BufferVar<IndexT> subscene_element_ids,
                                     BufferVar<IndexT> contact_mask_tabular,
                                     BufferVar<IndexT> subscene_mask_tabular,
                                     BufferVar<Vector3> half_plane_positions,
                                     BufferVar<Vector3> half_plane_normals,
                                     BufferVar<Float> d_hats,
                                     BufferVar<Vector2i> PHs_out,
                                     BufferVar<IndexT> num_collisions_buf,
                                     IndexT plane_vertex_offset,
                                     SizeT max_count,
                                     SizeT plane_count,
                                     SizeT contact_dim,
                                     SizeT subscene_dim) noexcept
        {
            auto i = dispatch_x();
            $if(i < surf_vert_count)
            {
                IndexT vI = surf_vertices.read(i);
                
                $for(j, plane_count)
                {
                    IndexT vJ = plane_vertex_offset + cast<IndexT>(j);

                    Float d_hat = d_hats.read(vI);

                    IndexT L = contact_element_ids.read(vI);
                    IndexT R = contact_element_ids.read(vJ);

                    IndexT sL = subscene_element_ids.read(vI);
                    IndexT sR = subscene_element_ids.read(vJ);

                    // Check subscene mask (flattened 2D table: row * dim + col)
                    $if(subscene_mask_tabular.read(sL * subscene_dim + sR) == 0)
                    {
                        $continue;
                    };

                    // Check contact mask (flattened 2D table: row * dim + col)
                    $if(contact_mask_tabular.read(L * contact_dim + R) == 0)
                    {
                        $continue;
                    };

                    Vector3 pos = positions.read(vI);

                    Vector3 plane_pos = half_plane_positions.read(j);
                    Vector3 plane_normal = half_plane_normals.read(j);

                    Vector3 diff = pos - plane_pos;

                    Float dst = dot(diff, plane_normal);

                    Float thickness = thicknesses.read(vI);

                    Float D = dst * dst;

                    Vector2 range = D_range(thickness, d_hat);

                    // Note: MUDA_ASSERT equivalent not available in LuisaCompute device code
                    
                    $if(is_active_D(range, D))
                    {
                        auto last = num_collisions_buf.atomic(0).fetch_add(1);

                        $if(last < max_count)
                        {
                            PHs_out.write(last, Vector2i{cast<int>(vI), cast<int>(j)});
                        };
                    };
                };
            };
        };

        auto shader = device.compile(detect_kernel);
        stream << shader(surf_vertices_view,
                         positions_view,
                         thicknesses_view,
                         contact_element_ids_view,
                         subscene_element_ids_view,
                         contact_mask_tabular_view,
                         subscene_mask_tabular_view,
                         plane_positions_view,
                         plane_normals_view,
                         d_hats_view,
                         PHs_view,
                         num_collisions.view(),
                         plane_vertex_offset,
                         max_count,
                         plane_count,
                         contact_table_dim,
                         subscene_table_dim)
                      .dispatch(surf_vert_count);
    };

    query();

    // Read back the collision count
    stream << num_collisions.copy_to(&h_num_collisions, 1)
           << synchronize();

    if(h_num_collisions > PHs.size())
    {
        PHs = device.create_buffer<Vector2i>(static_cast<size_t>(h_num_collisions * reserve_ratio));
        query();
        
        // Read back again after resize
        stream << num_collisions.copy_to(&h_num_collisions, 1)
               << synchronize();
    }

    info.PHs(PHs.view(0, h_num_collisions));

    if constexpr(PrintDebugInfo)
    {
        std::vector<Vector2i> phs(h_num_collisions);
        stream << PHs.view(0, h_num_collisions).copy_to(phs.data())
               << synchronize();
        for(auto& ph : phs)
        {
            std::cout << "vI: " << ph[0] << ", pI: " << ph[1] << std::endl;
        }
    }
}

void EasyVertexHalfPlaneTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info, WorldVisitor& world)
{
    auto& engine = static_cast<SimEngine&>(world.sim_engine());
    auto& device = engine.device();
    auto  stream = engine.compute_stream();

    // Fill toi with 1.1f
    Float initial_toi = 1.1f;
    stream << info.toi().fill(initial_toi);
    
    if(tois.size() < info.surf_vertices().size())
    {
        tois = device.create_buffer<Float>(info.surf_vertices().size());
    }

    // TODO: just hard code the slackness for now
    constexpr Float eta = 0.1f;

    auto surf_vertices_view = info.surf_vertices();
    auto plane_vertex_offset = info.half_plane_vertex_offset();
    auto positions_view = info.positions();
    auto thicknesses_view = info.thicknesses();
    auto contact_element_ids_view = info.contact_element_ids();
    auto subscene_element_ids_view = info.subscene_element_ids();
    auto subscene_mask_tabular_view = info.subscene_mask_tabular();
    auto contact_mask_tabular_view = info.contact_mask_tabular();
    auto displacements_view = info.displacements();
    auto plane_positions_view = info.plane_positions();
    auto plane_normals_view = info.plane_normals();
    auto tois_view = tois.view();
    Float alpha = info.alpha();
    SizeT surf_vert_count = surf_vertices_view.size();
    SizeT plane_count = plane_positions_view.size();
    
    // Compute table dimensions (tables are square N x N)
    SizeT contact_table_size = contact_mask_tabular_view.size();
    SizeT contact_table_dim = static_cast<SizeT>(std::sqrt(static_cast<double>(contact_table_size)));
    SizeT subscene_table_size = subscene_mask_tabular_view.size();
    SizeT subscene_table_dim = static_cast<SizeT>(std::sqrt(static_cast<double>(subscene_table_size)));

    Kernel1D toi_kernel = [&](BufferVar<IndexT> surf_vertices,
                              BufferVar<Vector3> positions,
                              BufferVar<Float> thicknesses,
                              BufferVar<IndexT> contact_element_ids,
                              BufferVar<IndexT> subscene_element_ids,
                              BufferVar<IndexT> subscene_mask_tabular,
                              BufferVar<IndexT> contact_mask_tabular,
                              BufferVar<Vector3> displacements,
                              BufferVar<Vector3> half_plane_positions,
                              BufferVar<Vector3> half_plane_normals,
                              BufferVar<Float> tois_out,
                              Float alpha_val,
                              Float eta_val,
                              IndexT plane_vertex_offset,
                              SizeT plane_count,
                              SizeT contact_dim,
                              SizeT subscene_dim) noexcept
    {
        auto i = dispatch_x();
        $if(i < surf_vert_count)
        {
            Float min_toi = 1.1f;  // large enough

            IndexT vI = surf_vertices.read(i);

            $for(j, plane_count)
            {
                IndexT vJ = plane_vertex_offset + cast<IndexT>(j);

                IndexT L = contact_element_ids.read(vI);
                IndexT R = contact_element_ids.read(vJ);

                IndexT sL = subscene_element_ids.read(vI);
                IndexT sR = subscene_element_ids.read(vJ);

                // Check masks (flattened 2D tables)
                $if(subscene_mask_tabular.read(sL * subscene_dim + sR) == 0)
                {
                    $continue;
                };

                $if(contact_mask_tabular.read(L * contact_dim + R) == 0)
                {
                    $continue;
                };

                Vector3 x = positions.read(vI);
                Vector3 dx = displacements.read(vI) * alpha_val;
                Vector3 x_t = x + dx;

                Vector3 P = half_plane_positions.read(j);
                Vector3 N = half_plane_normals.read(j);

                Float thickness = thicknesses.read(vI);

                Float t = -dot(N, dx);
                $if(t <= 0.0f)  // moving away from the plane, no collision
                {
                    $continue;
                };

                // t > 0, moving towards the plane
                Vector3 diff = P - x;
                Float t0 = -dot(N, diff) - thickness;

                Float this_toi = t0 / t * (1.0f - eta_val);

                min_toi = min(min_toi, this_toi);
            };

            tois_out.write(i, min_toi);
        };
    };

    auto shader = device.compile(toi_kernel);
    stream << shader(surf_vertices_view,
                     positions_view,
                     thicknesses_view,
                     contact_element_ids_view,
                     subscene_element_ids_view,
                     subscene_mask_tabular_view,
                     contact_mask_tabular_view,
                     displacements_view,
                     plane_positions_view,
                     plane_normals_view,
                     tois_view,
                     alpha,
                     eta,
                     plane_vertex_offset,
                     plane_count,
                     contact_table_dim,
                     subscene_table_dim)
                  .dispatch(surf_vert_count);

    // Compute min using host reduction
    // LuisaCompute doesn't have built-in device reduction like CUB
    SizeT count = surf_vert_count;
    
    if(count > 0)
    {
        std::vector<Float> h_tois(count);
        stream << tois_view.copy_to(h_tois.data())
               << synchronize();
        
        Float min_toi = 1.1f;
        for(SizeT i = 0; i < count; ++i)
        {
            min_toi = std::min(min_toi, h_tois[i]);
        }
        
        stream << info.toi().fill(min_toi);
    }
}

}  // namespace uipc::backend::luisa
