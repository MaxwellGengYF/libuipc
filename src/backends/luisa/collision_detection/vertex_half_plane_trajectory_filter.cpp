#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <frame_stream.h>
#include <luisa_device/atomic.h>
#include <luisa_device/linear/algorithms.h>
#include <fmt/format.h>
/*************************************************************************************************
 * T7.2a — VertexHalfPlaneTrajectoryFilter (port of
 * src/backends/cuda/collision_detection/vertex_half_plane_trajectory_filter.cu, 235 lines).
 *
 * KERNEL MAPPING: the one label-active-vertices kernel is a verbatim transcription with
 * `cuda_tool::atomic_exch(&is_active(P), 1)` -> `lud::atomic_exch(is_active.atomic(P), 1)` (the
 * discard of its [[nodiscard]] return becomes `(void)`),
 * and `best_grid_dim/best_block_dim` -> `dispatch(n)`.
 *
 * `do_build()` transcribes the reference's five `require<>`s in order
 * (GlobalVertexManager, GlobalSimplicialSurfaceManager, GlobalContactManager, HalfPlane,
 * HalfPlaneVertexReporter, GlobalTrajectoryFilter) and then calls the derived `do_build`,
 * so a filter in a scene without a half plane is invalidated by the engine's
 * SimSystemException handler -- the reference's own scene-gating mechanism.
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

/// `VertexHalfPlaneTrajectoryFilter_label_active_vertices_kernel`
auto& vertex_half_plane_trajectory_filter_label_active_vertices_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector2i>, lcrt::Buffer<std::int32_t>, std::uint32_t>(
            [](lcrt::BufferVar<LVector2i> PHs,
               lcrt::BufferVar<std::int32_t> is_active,
               lcrt::Var<std::uint32_t> n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto PH = PHs->read(i);
                    auto P  = PH.data[0];
                    $if(is_active->read(P) == 0)
                    {
                        (void)lud::atomic_exch(is_active.atomic(P), std::int32_t{1});
                    };
                };
            });
    return k;
}
}// namespace

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
    int n = static_cast<int>(PHs.size());
    if(n > 0)
    {
        auto& stream = frame_stream();
        stream.record(vertex_half_plane_trajectory_filter_label_active_vertices_kernel()(
                          PHs, info.vert_is_active(), static_cast<std::uint32_t>(n))
                          .dispatch(static_cast<std::uint32_t>(n)));
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
    loose_resize(friction_PHs, PHs.size());
    friction_PHs.view().copy_from(PHs);
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

lud::LCBufferView<LVector2i> VertexHalfPlaneTrajectoryFilter::PHs() noexcept
{
    return m_impl.PHs;
}

lud::LCBufferView<LVector2i> VertexHalfPlaneTrajectoryFilter::friction_PHs() noexcept
{
    return m_impl.friction_PHs.cview();
}

Float VertexHalfPlaneTrajectoryFilter::BaseInfo::d_hat() const noexcept
{
    return m_impl->global_contact_manager->d_hat();
}

lud::LCBufferView<Float> VertexHalfPlaneTrajectoryFilter::BaseInfo::d_hats() const noexcept
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

lud::LCBufferView<LVector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::plane_normals() const noexcept
{
    return m_impl->half_plane->normals();
}

lud::LCBufferView<LVector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::plane_positions() const noexcept
{
    return m_impl->half_plane->positions();
}

lud::LCBufferView<LVector3> VertexHalfPlaneTrajectoryFilter::BaseInfo::positions() const noexcept
{
    return m_impl->global_vertex_manager->positions();
}

lud::LCBufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::contact_element_ids() const noexcept
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

lud::LCBufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::subscene_element_ids() const noexcept
{
    return m_impl->global_vertex_manager->subscene_element_ids();
}

lud::LCBufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::contact_mask_tabular() const noexcept
{
    return m_impl->global_contact_manager->contact_mask_tabular();
}

lud::LCBufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::subscene_mask_tabular() const noexcept
{
    return m_impl->global_contact_manager->subscene_mask_tabular();
}

IndexT VertexHalfPlaneTrajectoryFilter::BaseInfo::contact_mask_size() const noexcept
{
    return m_impl->global_contact_manager->contact_mask_size();
}

IndexT VertexHalfPlaneTrajectoryFilter::BaseInfo::subscene_mask_size() const noexcept
{
    return m_impl->global_contact_manager->subscene_mask_size();
}

lud::LCBufferView<Float> VertexHalfPlaneTrajectoryFilter::BaseInfo::thicknesses() const noexcept
{
    return m_impl->global_vertex_manager->thicknesses();
}

lud::LCBufferView<IndexT> VertexHalfPlaneTrajectoryFilter::BaseInfo::surf_vertices() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_vertices();
}

lud::LCBufferView<LVector3> VertexHalfPlaneTrajectoryFilter::DetectInfo::displacements() const noexcept
{
    return m_impl->global_vertex_manager->displacements();
}

void VertexHalfPlaneTrajectoryFilter::FilterActiveInfo::PHs(lud::LCBufferView<LVector2i> PHs) noexcept
{
    m_impl->PHs = PHs;
}

lud::LBufferView<Float> VertexHalfPlaneTrajectoryFilter::FilterTOIInfo::toi() noexcept
{
    return m_toi;
}

void VertexHalfPlaneTrajectoryFilter::do_clear_friction_candidates()
{
    m_impl.friction_PHs.resize(0);
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
    PHs = recovered_PHs.cview();
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
}// namespace uipc::backend::luisa
