#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/simplicial_surface_reporter.h>
#include <uipc/common/zip.h>
#include <uipc/common/enumerate.h>
#include <utils/offset_count_collection.h>
#include <sim_engine.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(GlobalSimplicialSurfaceManager);

void GlobalSimplicialSurfaceManager::add_reporter(SimplicialSurfaceReporter* reporter) noexcept
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    m_impl.reporters.register_sim_system(*reporter);
}

luisa::compute::BufferView<IndexT> GlobalSimplicialSurfaceManager::codim_vertices() const noexcept
{
    return m_impl.codim_vertices.view();
}

luisa::compute::BufferView<IndexT> GlobalSimplicialSurfaceManager::surf_vertices() const noexcept
{
    return m_impl.surf_vertices.view();
}

luisa::compute::BufferView<Vector2i> GlobalSimplicialSurfaceManager::surf_edges() const noexcept
{
    return m_impl.surf_edges.view();
}

luisa::compute::BufferView<Vector3i> GlobalSimplicialSurfaceManager::surf_triangles() const noexcept
{
    return m_impl.surf_triangles.view();
}

void GlobalSimplicialSurfaceManager::do_build()
{
    m_impl.global_vertex_manager = find<GlobalVertexManager>();
    m_impl.sim_engine = &engine();
}

void GlobalSimplicialSurfaceManager::Impl::init()
{
    auto reporter_view = reporters.view();

    // 1) initialize the reporters
    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = i;

    for(auto&& R : reporter_view)
    {
        SurfaceInitInfo info;
        R->init(info);
    }

    auto reporter_count = reporter_view.size();

    // 2) compute the counts and offsets
    reporter_infos.resize(reporter_view.size());
    OffsetCountCollection<IndexT> vertex_offsets_counts;
    vertex_offsets_counts.resize(sim_engine->luisa_device(), reporter_count);
    OffsetCountCollection<IndexT> edge_offsets_counts;
    edge_offsets_counts.resize(sim_engine->luisa_device(), reporter_count);
    OffsetCountCollection<IndexT> triangle_offsets_counts;
    triangle_offsets_counts.resize(sim_engine->luisa_device(), reporter_count);

    span<IndexT> vertex_counts   = vertex_offsets_counts.counts();
    span<IndexT> edge_counts     = edge_offsets_counts.counts();
    span<IndexT> triangle_counts = triangle_offsets_counts.counts();

    for(auto&& [R, Rinfo] : zip(reporter_view, reporter_infos))
    {
        SurfaceCountInfo info;
        R->report_count(info);
        auto V = info.m_surf_vertex_count;
        auto E = info.m_surf_edge_count;
        auto F = info.m_surf_triangle_count;

        Rinfo.surf_vertex_count   = V;
        Rinfo.surf_edge_count     = E;
        Rinfo.surf_triangle_count = F;

        vertex_counts[R->m_index]   = V;
        edge_counts[R->m_index]     = E;
        triangle_counts[R->m_index] = F;
    }

    vertex_offsets_counts.scan(sim_engine->compute_stream());
    edge_offsets_counts.scan(sim_engine->compute_stream());
    triangle_offsets_counts.scan(sim_engine->compute_stream());

    span<const IndexT> vertex_offsets   = vertex_offsets_counts.offsets();
    span<const IndexT> edge_offsets     = edge_offsets_counts.offsets();
    span<const IndexT> triangle_offsets = triangle_offsets_counts.offsets();

    for(auto&& [i, Rinfo] : enumerate(reporter_infos))
    {
        Rinfo.surf_vertex_offset   = vertex_offsets[i];
        Rinfo.surf_edge_offset     = edge_offsets[i];
        Rinfo.surf_triangle_offset = triangle_offsets[i];
    }

    SizeT total_surf_vertex_count   = vertex_offsets_counts.total_count();
    SizeT total_surf_edge_count     = edge_offsets_counts.total_count();
    SizeT total_surf_triangle_count = triangle_offsets_counts.total_count();

    // 3) resize the device buffer
    codim_vertices = sim_engine->luisa_device().create_buffer<IndexT>(total_surf_vertex_count);
    surf_vertices = sim_engine->luisa_device().create_buffer<IndexT>(total_surf_vertex_count);
    surf_edges = sim_engine->luisa_device().create_buffer<Vector2i>(total_surf_edge_count);
    surf_triangles = sim_engine->luisa_device().create_buffer<Vector3i>(total_surf_triangle_count);
    codim_vertex_flags = sim_engine->luisa_device().create_buffer<IndexT>(total_surf_vertex_count);
    selected_codim_0d_count = sim_engine->luisa_device().create_buffer<int>(1);

    // 4) collect surface attributes
    for(auto&& [R, Rinfo] : zip(reporter_view, reporter_infos))
    {
        SurfaceAttributeInfo info{this, R->m_index};
        R->report_attributes(info);
    }

    // 5) collect Codim0D vertices
    _collect_codim_vertices();
}

void GlobalSimplicialSurfaceManager::Impl::_collect_codim_vertices()
{
    auto dim = global_vertex_manager->dimensions();
    auto vertex_count = surf_vertices.size();

    if(vertex_count == 0)
    {
        // No vertices to process
        codim_vertices = sim_engine->luisa_device().create_buffer<IndexT>(0);
        return;
    }

    // Step 1: Flag vertices with dimension <= 1 (codim 0D vert and vert from codim 1D edge)
    Kernel1D flag_kernel = [&](BufferVar<IndexT> dim_buffer,
                               BufferVar<IndexT> surf_vert_buffer,
                               BufferVar<IndexT> flags_buffer) noexcept
    {
        auto I = dispatch_idx().x;
        if(I >= vertex_count)
            return;
        auto vI = surf_vert_buffer.read(I);
        auto d = dim_buffer.read(vI);
        flags_buffer.write(I, d <= 1 ? 1 : 0);
    };

    auto flag_shader = sim_engine->luisa_device().compile(flag_kernel);
    sim_engine->compute_stream() << flag_shader(dim, surf_vertices, codim_vertex_flags).dispatch(vertex_count);

    // Step 2: Stream compaction using exclusive scan + scatter
    // Read flags to host for scan (similar to OffsetCountCollection approach)
    luisa::vector<IndexT> host_flags(vertex_count);
    sim_engine->compute_stream() << codim_vertex_flags.view().copy_to(host_flags.data())
                                 << luisa::compute::synchronize();

    // Compute exclusive scan for scatter offsets
    luisa::vector<IndexT> host_offsets(vertex_count + 1);
    IndexT running_sum = 0;
    for(SizeT i = 0; i < vertex_count; ++i)
    {
        host_offsets[i] = running_sum;
        running_sum += host_flags[i];
    }
    host_offsets[vertex_count] = running_sum;
    IndexT selected_count = running_sum;

    // Upload offsets to GPU
    auto offsets_buffer = sim_engine->luisa_device().create_buffer<IndexT>(vertex_count);
    sim_engine->compute_stream() << offsets_buffer.view().copy_from(host_offsets.data())
                                 << luisa::compute::synchronize();

    // Resize output buffer
    codim_vertices = sim_engine->luisa_device().create_buffer<IndexT>(selected_count);

    // Step 3: Scatter selected vertices
    if(selected_count > 0)
    {
        Kernel1D scatter_kernel = [&](BufferVar<IndexT> surf_vert_buffer,
                                      BufferVar<IndexT> flags_buffer,
                                      BufferVar<IndexT> offsets_buffer,
                                      BufferVar<IndexT> output_buffer) noexcept
        {
            auto I = dispatch_idx().x;
            if(I >= vertex_count)
                return;
            auto flag = flags_buffer.read(I);
            if(flag == 1)
            {
                auto offset = offsets_buffer.read(I);
                auto vI = surf_vert_buffer.read(I);
                output_buffer.write(offset, vI);
            }
        };

        auto scatter_shader = sim_engine->luisa_device().compile(scatter_kernel);
        sim_engine->compute_stream() << scatter_shader(surf_vertices, codim_vertex_flags, offsets_buffer, codim_vertices).dispatch(vertex_count);
    }

    // Update count
    int count = static_cast<int>(selected_count);
    sim_engine->compute_stream() << selected_codim_0d_count.view().copy_from(&count)
                                 << luisa::compute::synchronize();
}

void GlobalSimplicialSurfaceManager::init()
{
    m_impl.init();
}

void GlobalSimplicialSurfaceManager::rebuild()
{
    UIPC_ASSERT(false, "Not implemented yet");
}

luisa::compute::BufferView<IndexT> GlobalSimplicialSurfaceManager::SurfaceAttributeInfo::surf_vertices() noexcept
{
    const auto& info = reporter_info();
    return m_impl->surf_vertices.view(info.surf_vertex_offset, info.surf_vertex_count);
}

luisa::compute::BufferView<Vector2i> GlobalSimplicialSurfaceManager::SurfaceAttributeInfo::surf_edges() noexcept
{
    const auto& info = reporter_info();
    return m_impl->surf_edges.view(info.surf_edge_offset, info.surf_edge_count);
}

luisa::compute::BufferView<Vector3i> GlobalSimplicialSurfaceManager::SurfaceAttributeInfo::surf_triangles() noexcept
{
    const auto& info = reporter_info();
    return m_impl->surf_triangles.view(info.surf_triangle_offset, info.surf_triangle_count);
}

const GlobalSimplicialSurfaceManager::ReporterInfo& GlobalSimplicialSurfaceManager::SurfaceAttributeInfo::reporter_info() const noexcept
{
    return m_impl->reporter_infos[m_index];
}

void GlobalSimplicialSurfaceManager::SurfaceCountInfo::surf_vertex_count(SizeT count) noexcept
{
    m_surf_vertex_count = count;
}

void GlobalSimplicialSurfaceManager::SurfaceCountInfo::surf_edge_count(SizeT count) noexcept
{
    m_surf_edge_count = count;
}

void GlobalSimplicialSurfaceManager::SurfaceCountInfo::surf_triangle_count(SizeT count) noexcept
{
    m_surf_triangle_count = count;
}

void GlobalSimplicialSurfaceManager::SurfaceCountInfo::changable(bool value) noexcept
{
    m_changable = value;
}
}  // namespace uipc::backend::luisa
