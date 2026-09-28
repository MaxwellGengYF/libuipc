#include <finite_element/finite_element_surface_reporter.h>
#include <uipc/builtin/attribute_name.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(FiniteElementSurfaceReporter);

void FiniteElementSurfaceReporter::do_build(BuildInfo& info)
{
    m_impl.finite_element_method         = &require<FiniteElementMethod>();
    m_impl.finite_element_vertex_reporter = &require<FiniteElementVertexReporter>();
}

void FiniteElementSurfaceReporter::Impl::init(backend::WorldVisitor& world)
{
    auto& geo_infos = fem().geo_infos;

    auto geo_slots = world.scene().geometries();
    auto N         = geo_infos.size() + 1;  // +1 to store the total count

    geo_surf_vertex_counts.resize(N);
    geo_surf_vertex_offsets.resize(N);
    geo_surf_edge_counts.resize(N);
    geo_surf_edge_offsets.resize(N);
    geo_surf_triangle_counts.resize(N);
    geo_surf_triangle_offsets.resize(N);

    std::ranges::fill(geo_surf_vertex_counts, 0);
    std::ranges::fill(geo_surf_edge_counts, 0);
    std::ranges::fill(geo_surf_triangle_counts, 0);

    // for every geometry count the surf vertices, edges and triangles
    for(auto&& [i, geo_info] : enumerate(geo_infos))
    {
        auto geo_slot = geo_slots[geo_info.geo_slot_index];

        auto sc = geo_slot->geometry().as<geometry::SimplicialComplex>();

        UIPC_ASSERT(sc != nullptr, "Geometry is not a simplicial complex, why?");

        auto vert_is_surf = sc->vertices().find<IndexT>(builtin::is_surf);
        if(vert_is_surf)
        {
            auto view = vert_is_surf->view();
            auto count =
                std::ranges::count_if(view,
                                      [](IndexT is_surf) { return is_surf; });

            geo_surf_vertex_counts[i] = count;
        }

        auto edge_is_surf = sc->edges().find<IndexT>(builtin::is_surf);
        if(edge_is_surf)
        {
            auto view = edge_is_surf->view();
            auto count =
                std::ranges::count_if(view,
                                      [](IndexT is_surf) { return is_surf; });

            geo_surf_edge_counts[i] = count;
        }

        auto tri_is_surf = sc->triangles().find<IndexT>(builtin::is_surf);
        if(tri_is_surf)
        {
            auto view = tri_is_surf->view();
            auto count =
                std::ranges::count_if(view,
                                      [](IndexT is_surf) { return is_surf; });

            geo_surf_triangle_counts[i] = count;
        }
    }

    // compute the offsets
    // `SizeT{0}` (the reference passes an int literal): MSVC's exclusive_scan
    // instantiations with a `SizeT` value type and an `int` init warn C4244 inside
    // <numeric>; the value is unchanged.
    std::exclusive_scan(geo_surf_vertex_counts.begin(),
                        geo_surf_vertex_counts.end(),
                        geo_surf_vertex_offsets.begin(),
                        SizeT{0});

    std::exclusive_scan(geo_surf_edge_counts.begin(),
                        geo_surf_edge_counts.end(),
                        geo_surf_edge_offsets.begin(),
                        SizeT{0});

    std::exclusive_scan(geo_surf_triangle_counts.begin(),
                        geo_surf_triangle_counts.end(),
                        geo_surf_triangle_offsets.begin(),
                        SizeT{0});

    surf_vertices.resize(geo_surf_vertex_offsets.back());
    surf_edges.resize(geo_surf_edge_offsets.back());
    surf_triangles.resize(geo_surf_triangle_offsets.back());

    auto global_vertex_offset = finite_element_vertex_reporter->vertex_offset();

    auto push_surf_element =
        []<typename E, typename F>(span<const IndexT> is_surf, span<E> dst_e, F&& f)
    {
        SizeT I = 0;
        for(auto&& [i, is_surf] : enumerate(is_surf))
        {
            if(is_surf)
            {
                f(i, dst_e[I++]);
            }
        }
    };

    for(auto&& [i, geo_info] : enumerate(geo_infos))
    {
        auto geo_slot = geo_slots[geo_info.geo_slot_index];

        auto sc = geo_slot->geometry().as<geometry::SimplicialComplex>();


        auto vert_is_surf = sc->vertices().find<IndexT>(builtin::is_surf);
        if(vert_is_surf)
        {
            auto is_surf  = vert_is_surf->view();
            auto subspan  = span{surf_vertices}.subspan(geo_surf_vertex_offsets[i],
                                                       geo_surf_vertex_counts[i]);

            push_surf_element(is_surf,
                              subspan,
                              [&](IndexT i, IndexT& dst_e)
                              { dst_e = i + global_vertex_offset; });
        }

        auto edge_is_surf = sc->edges().find<IndexT>(builtin::is_surf);
        if(edge_is_surf)
        {
            auto is_surf = edge_is_surf->view();

            auto subspan = span{surf_edges}.subspan(geo_surf_edge_offsets[i],
                                                    geo_surf_edge_counts[i]);

            auto edges = sc->edges().topo().view();

            push_surf_element(is_surf,
                              subspan,
                              [&](IndexT i, LVector2i& dst_e)
                              {
                                  dst_e = LVector2i{edges[i][0] + global_vertex_offset,
                                                    edges[i][1] + global_vertex_offset};
                              });
        };

        auto tri_is_surf = sc->triangles().find<IndexT>(builtin::is_surf);
        if(tri_is_surf)
        {
            auto is_surf = tri_is_surf->view();

            auto subspan = span{surf_triangles}.subspan(geo_surf_triangle_offsets[i],
                                                        geo_surf_triangle_counts[i]);

            auto triangles = sc->triangles().topo().view();

            push_surf_element(is_surf,
                              subspan,
                              [&](IndexT i, LVector3i& dst_e)
                              {
                                  dst_e = LVector3i{triangles[i][0] + global_vertex_offset,
                                                    triangles[i][1] + global_vertex_offset,
                                                    triangles[i][2] + global_vertex_offset};
                              });
        }

        // update the global offset
        global_vertex_offset += sc->vertices().size();
    }
}

void FiniteElementSurfaceReporter::Impl::report_count(backend::WorldVisitor& world,
                                                      SurfaceCountInfo& info)
{
    info.surf_vertex_count(surf_vertices.size());
    info.surf_edge_count(surf_edges.size());
    info.surf_triangle_count(surf_triangles.size());
}

void FiniteElementSurfaceReporter::Impl::report_attributes(backend::WorldVisitor& world,
                                                           SurfaceAttributeInfo& info)
{
    // the reference's `cuda_tool::BufferLaunch().copy<T>(dst, src.data())` is a
    // host->device byte copy; the port copies the element-wise converted arrays
    // (the element types are byte-identical, cookbook §7.13 rule 1)
    info.surf_vertices().copy_from(span<const IndexT>{surf_vertices});

    if(!surf_edges.empty())
    {
        info.surf_edges().copy_from(span<const LVector2i>{surf_edges});
    }

    if(!surf_triangles.empty())
    {
        info.surf_triangles().copy_from(span<const LVector3i>{surf_triangles});
    }
}

void FiniteElementSurfaceReporter::do_init(SurfaceInitInfo& info)
{
    m_impl.init(world());
}

void FiniteElementSurfaceReporter::do_report_count(SurfaceCountInfo& info)
{
    m_impl.report_count(world(), info);
}

void FiniteElementSurfaceReporter::do_report_attributes(SurfaceAttributeInfo& info)
{
    m_impl.report_attributes(world(), info);
}
}  // namespace uipc::backend::luisa
