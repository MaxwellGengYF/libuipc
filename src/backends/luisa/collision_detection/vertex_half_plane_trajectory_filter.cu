#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <sim_engine.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

namespace uipc::backend::luisa
{
void VertexHalfPlaneTrajectoryFilter::do_build()
{
    m_impl.global_vertex_manager = &require<GlobalVertexManager>();
    m_impl.global_simplicial_surface_manager =
        &require<GlobalSimplicialSurfaceManager>();
    m_impl.global_contact_manager     = &require<GlobalContactManager>();
    m_impl.half_plane                 = &require<HalfPlane>();
    m_impl.half_plane_vertex_reporter = &require<HalfPlaneVertexReporter>();
    auto global_trajectory_filter     = &require<GlobalTrajectoryFilter>();

    BuildInfo info;
    do_build(info);

    global_trajectory_filter->add_filter(this);
}

void VertexHalfPlaneTrajectoryFilter::do_detect(GlobalTrajectoryFilter::DetectInfo& info)
{
    DetectInfo this_info{&m_impl};
    this_info.m_alpha = info.alpha();

    do_detect(this_info);  // call the derived class implementation
}

void VertexHalfPlaneTrajectoryFilter::Impl::label_active_vertices(
    GlobalTrajectoryFilter::LabelActiveVerticesInfo& info)
{
    auto& engine = this->global_vertex_manager->engine();
    auto& stream = engine.compute_stream();
    auto& device = engine.luisa_device();

    if(PHs.size() > 0)
    {
        Kernel1D label_ph_kernel = [&](BufferVar<Vector2i> PHs_buf, BufferVar<IndexT> is_active) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < PHs_buf.size())
            {
                auto PH = PHs_buf.read(i);
                auto P  = PH[0];
                $if(is_active.read(P) == 0)
                {
                    is_active.atomic(P).exchange(1);
                };
            };
        };
        auto shader = device.compile(label_ph_kernel);
        stream << shader(PHs, info.vert_is_active()).dispatch(PHs.size());
    }
}

void VertexHalfPlaneTrajectoryFilter::do_filter_active(GlobalTrajectoryFilter::FilterActiveInfo& info)
{
    FilterActiveInfo this_info{&m_impl};
    do_filter_active(this_info);

    logger::info("VertexHalfPlaneTrajectoryFilter PHs: {}.", m_impl.PHs.size());
}

void VertexHalfPlaneTrajectoryFilter::do_filter_toi(GlobalTrajectoryFilter::FilterTOIInfo& info)
{
    FilterTOIInfo this_info{&m_impl};
    this_info.m_alpha = info.alpha();
    this_info.m_toi   = info.toi();
    do_filter_toi(this_info);
}

void VertexHalfPlaneTrajectoryFilter::Impl::record_friction_candidates(
    GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info)
{
    auto& engine = this->global_vertex_manager->engine();
    auto& stream = engine.compute_stream();
    auto& device = engine.luisa_device();

    loose_resize(friction_PHs, PHs.size(), device);
    if(PHs.size() > 0)
    {
        stream << friction_PHs.copy_from(PHs);
    }
}

void VertexHalfPlaneTrajectoryFilter::do_record_friction_candidates(
    GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info)
{
    m_impl.record_friction_candidates(info);
}

void VertexHalfPlaneTrajectoryFilter::do_label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info)
{
    m_impl.label_active_vertices(info);
}

BufferView<Vector2i> VertexHalfPlaneTrajectoryFilter::PHs() noexcept
{
    return m_impl.PHs;
}

BufferView<Vector2i> VertexHalfPlaneTrajectoryFilter::friction_PHs() noexcept
{
    return m_impl.friction_PHs;
}

Float VertexHalfPlaneTrajectoryFilter::BaseInfo::d_hat() const noexcept
{
    return m_impl->global_contact_manager->d_hat();
}

BufferView<Float> VertexHalfPlaneTrajectoryFilter::BaseInfo::d_hats() const noexcept
{
    return m_impl->global_vertex_manager->d_hats();
}

Float VertexHalfPlaneTrajectoryFilter::DetectInfo::alpha() const noexcept
{
    return m_alpha;
}

IndexT VertexHalfPlaneTrajectoryFilter::BaseInfo::half_plane_vertex_offset() const noexcept
{
    return m_impl->half_plane_vertex_reporter->vertex_offset();
}

BufferView<Vector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::plane_normals() const noexcept
{
    return m_impl->half_plane->normals();
}

BufferView<Vector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::plane_positions() const noexcept
{
    return m_impl->half_plane->positions();
}

BufferView<Vector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::positions() const noexcept
{
    return m_impl->global_vertex_manager->positions();
}

BufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::contact_element_ids() const noexcept
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

BufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::subscene_element_ids() const noexcept
{
    return m_impl->global_vertex_manager->subscene_element_ids();
}

BufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::contact_mask_tabular() const noexcept
{
    return m_impl->global_contact_manager->contact_mask_tabular();
}

BufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::subscene_mask_tabular() const noexcept
{
    return m_impl->global_contact_manager->subscene_mask_tabular();
}

BufferView<Float> VertexHalfPlaneTrajectoryFilter::BaseInfo::thicknesses() const noexcept
{
    return m_impl->global_vertex_manager->thicknesses();
}

BufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::surf_vertices() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_vertices();
}

BufferView<Vector3> VertexHalfPlaneTrajectoryFilter::DetectInfo::displacements() const noexcept
{
    return m_impl->global_vertex_manager->displacements();
}

void VertexHalfPlaneTrajectoryFilter::FilterActiveInfo::PHs(BufferView<Vector2i> PHs) noexcept
{
    m_impl->PHs = PHs;
}

BufferView<Float> VertexHalfPlaneTrajectoryFilter::FilterTOIInfo::toi() noexcept
{
    return m_toi;
}

void VertexHalfPlaneTrajectoryFilter::do_clear_friction_candidates()
{
    // In LuisaCompute, buffers are immutable in size
    // We reset the buffer to an empty state
    m_impl.friction_PHs = Buffer<Vector2i>{};
}

bool VertexHalfPlaneTrajectoryFilter::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_PHs.dump(fmt::format("{}PHs.{}", path, frame), PHs);
}

bool VertexHalfPlaneTrajectoryFilter::Impl::try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_PHs.load(fmt::format("{}PHs.{}", path, frame));
}

void VertexHalfPlaneTrajectoryFilter::Impl::apply_recover(RecoverInfo& info)
{
    dump_PHs.apply_to(recovered_PHs);
    // temporary switch to the recovered PHs, which will be used 
    // in the record_friction_candidates() function to recover the friction candidates.
    PHs = recovered_PHs.view();
}

void VertexHalfPlaneTrajectoryFilter::Impl::clear_recover(RecoverInfo& info)
{
    dump_PHs.clean_up();
}

bool VertexHalfPlaneTrajectoryFilter::do_dump(DumpInfo& info)
{
    return m_impl.dump(info);
}

bool VertexHalfPlaneTrajectoryFilter::do_try_recover(RecoverInfo& info)
{
    return m_impl.try_recover(info);
}

void VertexHalfPlaneTrajectoryFilter::do_apply_recover(RecoverInfo& info)
{
    m_impl.apply_recover(info);
}

void VertexHalfPlaneTrajectoryFilter::do_clear_recover(RecoverInfo& info)
{
    m_impl.clear_recover(info);
}
}  // namespace uipc::backend::luisa
