#include <collision_detection/simplex_trajectory_filter.h>
#include <frame_stream.h>
#include <luisa_device/atomic.h>
#include <luisa_device/linear/algorithms.h>
#include <fmt/format.h>
/*************************************************************************************************
 * T7.2a — SimplexTrajectoryFilter (port of
 * src/backends/cuda/collision_detection/simplex_trajectory_filter.cu, 409 lines).
 *
 * KERNEL MAPPING
 * --------------
 * The reference's four `SimplexTrajectoryFilter_label_active_vertices_k{1..4}_kernel`s are
 * the same kernel over four element types (Vector4i/Vector4i/Vector3i/Vector2i); the port
 * keeps one *template* so each element type still gets its own cached shader (the cache
 * key is the instantiation, cookbook §7.32 rule 3), while the four dispatch sites stay
 * separate exactly like the reference's (`n > 0` guard per array).
 *
 *   cuda_tool::atomic_exch(&is_active(P), 1)   ->  lud::atomic_exch(is_active.atomic(P), 1)
 *                                                  (the discarded [[nodiscard]] return is
 *                                                  an explicit `(void)` cast)
 *   best_grid_dim / best_block_dim             ->  dispatch(n) + set_block_size
 *   `PT[j]` on a Vector4i                      ->  `pair.data[j]` on the LVec element
 *
 * Everything else is a verbatim transcription: the DTYPE/size contracts, the
 * friction-candidate copies (loose_resize + copy_from), the recovery dump/apply cycle
 * (`apply_recover` temporarily points the candidate views at the recovered buffers, so
 * `record_friction_candidates` recovers the friction set -- the reference's own comment),
 * and the `logger::info` lines.
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

/// `SimplexTrajectoryFilter_label_active_vertices_k{1..4}_kernel` (one instantiation per
/// candidate-pair element type).  `E` is `LVector4i` / `LVector3i` / `LVector2i`.
template <typename E>
auto& simplex_trajectory_filter_label_active_vertices_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<E>, lcrt::Buffer<std::int32_t>, std::uint32_t>(
            [](lcrt::BufferVar<E> pairs,
               lcrt::BufferVar<std::int32_t> is_active,
               lcrt::Var<std::uint32_t> n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto pair = pairs->read(i);
                    for(int j = 0; j < lps::vec_dim_v<E>; ++j)
                    {
                        auto P = pair.data[j];
                        $if(is_active->read(P) == 0)
                        {
                            (void)lud::atomic_exch(is_active.atomic(P), std::int32_t{1});
                        };
                    }
                };
            });
    return k;
}
}// namespace

void SimplexTrajectoryFilter::do_build()
{
    m_impl.global_vertex_manager = require<GlobalVertexManager>();
    m_impl.global_simplicial_surface_manager = require<GlobalSimplicialSurfaceManager>();
    m_impl.global_contact_manager = require<GlobalContactManager>();
    m_impl.global_body_manager    = require<GlobalBodyManager>();
    const auto constitution =
        world().scene().config().find<std::string>("contact/constitution")->view()[0];
    m_impl.toi_safety_margin       = constitution == "al-ipc" ? 0.001 : 0.1;
    auto& global_trajectory_filter = require<GlobalTrajectoryFilter>();

    BuildInfo info;
    do_build(info);

    global_trajectory_filter.add_filter(this);
}

void SimplexTrajectoryFilter::do_detect(GlobalTrajectoryFilter::DetectInfo& info)
{
    DetectInfo this_info{&m_impl};
    this_info.m_alpha = info.alpha();
    do_detect(this_info);
}

void SimplexTrajectoryFilter::Impl::label_active_vertices(
    GlobalTrajectoryFilter::LabelActiveVerticesInfo& info)
{
    auto& stream = frame_stream();

    if(PTs.size() > 0)
    {
        auto n = static_cast<std::uint32_t>(PTs.size());
        stream.record(simplex_trajectory_filter_label_active_vertices_kernel<LVector4i>()(
                          PTs, info.vert_is_active(), n)
                          .dispatch(n));
    }

    if(EEs.size() > 0)
    {
        auto n = static_cast<std::uint32_t>(EEs.size());
        stream.record(simplex_trajectory_filter_label_active_vertices_kernel<LVector4i>()(
                          EEs, info.vert_is_active(), n)
                          .dispatch(n));
    }

    if(PEs.size() > 0)
    {
        auto n = static_cast<std::uint32_t>(PEs.size());
        stream.record(simplex_trajectory_filter_label_active_vertices_kernel<LVector3i>()(
                          PEs, info.vert_is_active(), n)
                          .dispatch(n));
    }

    if(PPs.size() > 0)
    {
        auto n = static_cast<std::uint32_t>(PPs.size());
        stream.record(simplex_trajectory_filter_label_active_vertices_kernel<LVector2i>()(
                          PPs, info.vert_is_active(), n)
                          .dispatch(n));
    }
}

void SimplexTrajectoryFilter::do_filter_active(GlobalTrajectoryFilter::FilterActiveInfo& info)
{
    FilterActiveInfo this_info{&m_impl};
    do_filter_active(this_info);

    logger::info("SimplexTrajectoryFilter PTs: {}, EEs: {}, PEs: {}, PPs: {}",
                 m_impl.PTs.size(),
                 m_impl.EEs.size(),
                 m_impl.PEs.size(),
                 m_impl.PPs.size());
}

void SimplexTrajectoryFilter::do_filter_toi(GlobalTrajectoryFilter::FilterTOIInfo& info)
{
    FilterTOIInfo this_info{&m_impl};
    this_info.m_alpha = info.alpha();
    this_info.m_toi   = info.toi();
    do_filter_toi(this_info);
}

void SimplexTrajectoryFilter::Impl::record_friction_candidates(
    GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info)
{
    // PT
    loose_resize(friction_PT, PTs.size());
    friction_PT.view().copy_from(PTs);

    // EE
    loose_resize(friction_EE, EEs.size());
    friction_EE.view().copy_from(EEs);

    // PE
    loose_resize(friction_PE, PEs.size());
    friction_PE.view().copy_from(PEs);

    // PP
    loose_resize(friction_PP, PPs.size());
    friction_PP.view().copy_from(PPs);

    logger::info("SimplexTrajectoryFilter Friction PT: {}, EE: {}, PE: {}, PP: {}",
                 friction_PT.size(),
                 friction_EE.size(),
                 friction_PE.size(),
                 friction_PP.size());
}


void SimplexTrajectoryFilter::do_record_friction_candidates(GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info)
{
    m_impl.record_friction_candidates(info);
}

void SimplexTrajectoryFilter::do_label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info)
{
    m_impl.label_active_vertices(info);
}

Float SimplexTrajectoryFilter::BaseInfo::d_hat() const noexcept
{
    return m_impl->global_contact_manager->d_hat();
}

Float SimplexTrajectoryFilter::BaseInfo::toi_safety_margin() const noexcept
{
    return m_impl->toi_safety_margin;
}

lud::LCBufferView<Float> SimplexTrajectoryFilter::BaseInfo::d_hats() const noexcept
{
    return m_impl->global_vertex_manager->d_hats();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::v2b() const noexcept
{
    return m_impl->global_vertex_manager->body_ids();
}

lud::LCBufferView<LVector3> SimplexTrajectoryFilter::BaseInfo::positions() const noexcept
{
    return m_impl->global_vertex_manager->positions();
}

lud::LCBufferView<LVector3> SimplexTrajectoryFilter::BaseInfo::rest_positions() const noexcept
{
    return m_impl->global_vertex_manager->rest_positions();
}

lud::LCBufferView<Float> SimplexTrajectoryFilter::BaseInfo::thicknesses() const noexcept
{
    return m_impl->global_vertex_manager->thicknesses();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::dimensions() const noexcept
{
    return m_impl->global_vertex_manager->dimensions();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::body_self_collision() const noexcept
{
    return m_impl->global_body_manager->self_collision();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::codim_vertices() const noexcept
{
    return m_impl->global_simplicial_surface_manager->codim_vertices();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::surf_vertices() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_vertices();
}

lud::LCBufferView<LVector2i> SimplexTrajectoryFilter::BaseInfo::surf_edges() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_edges();
}

lud::LCBufferView<LVector3i> SimplexTrajectoryFilter::BaseInfo::surf_triangles() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_triangles();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::contact_element_ids() const noexcept
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::subscene_element_ids() const noexcept
{
    return m_impl->global_vertex_manager->subscene_element_ids();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::contact_mask_tabular() const noexcept
{
    return m_impl->global_contact_manager->contact_mask_tabular();
}

lud::LCBufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::subscene_mask_tabular() const noexcept
{
    return m_impl->global_contact_manager->subscene_mask_tabular();
}

IndexT SimplexTrajectoryFilter::BaseInfo::contact_mask_size() const noexcept
{
    return m_impl->global_contact_manager->contact_mask_size();
}

IndexT SimplexTrajectoryFilter::BaseInfo::subscene_mask_size() const noexcept
{
    return m_impl->global_contact_manager->subscene_mask_size();
}

lud::LCBufferView<LVector4i> SimplexTrajectoryFilter::PTs() const noexcept
{
    return m_impl.PTs;
}

lud::LCBufferView<LVector4i> SimplexTrajectoryFilter::EEs() const noexcept
{
    return m_impl.EEs;
}

lud::LCBufferView<LVector3i> SimplexTrajectoryFilter::PEs() const noexcept
{
    return m_impl.PEs;
}

lud::LCBufferView<LVector2i> SimplexTrajectoryFilter::PPs() const noexcept
{
    return m_impl.PPs;
}

lud::LCBufferView<LVector4i> SimplexTrajectoryFilter::friction_PTs() const noexcept
{
    return m_impl.friction_PT.cview();
}

lud::LCBufferView<LVector4i> SimplexTrajectoryFilter::friction_EEs() const noexcept
{
    return m_impl.friction_EE.cview();
}

lud::LCBufferView<LVector3i> SimplexTrajectoryFilter::friction_PEs() const noexcept
{
    return m_impl.friction_PE.cview();
}


lud::LCBufferView<LVector2i> SimplexTrajectoryFilter::friction_PPs() const noexcept
{
    return m_impl.friction_PP.cview();
}

lud::LCBufferView<LVector3> SimplexTrajectoryFilter::DetectInfo::displacements() const noexcept
{
    return m_impl->global_vertex_manager->displacements();
}

void SimplexTrajectoryFilter::FilterActiveInfo::PTs(lud::LCBufferView<LVector4i> PTs) noexcept
{
    m_impl->PTs = PTs;
}

void SimplexTrajectoryFilter::FilterActiveInfo::EEs(lud::LCBufferView<LVector4i> EEs) noexcept
{
    m_impl->EEs = EEs;
}

void SimplexTrajectoryFilter::FilterActiveInfo::PEs(lud::LCBufferView<LVector3i> PEs) noexcept
{
    m_impl->PEs = PEs;
}

void SimplexTrajectoryFilter::FilterActiveInfo::PPs(lud::LCBufferView<LVector2i> PPs) noexcept
{
    m_impl->PPs = PPs;
}

lud::LBufferView<Float> SimplexTrajectoryFilter::FilterTOIInfo::toi() noexcept
{
    return m_toi;
}

void SimplexTrajectoryFilter::do_clear_friction_candidates()
{
    m_impl.friction_PT.resize(0);
    m_impl.friction_EE.resize(0);
    m_impl.friction_PE.resize(0);
    m_impl.friction_PP.resize(0);
}

bool SimplexTrajectoryFilter::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_PTs.dump(fmt::format("{}PTs.{}", path, frame), PTs)      //
           && dump_EEs.dump(fmt::format("{}EEs.{}", path, frame), EEs)   //
           && dump_PEs.dump(fmt::format("{}PEs.{}", path, frame), PEs)   //
           && dump_PPs.dump(fmt::format("{}PPs.{}", path, frame), PPs);  //
}

bool SimplexTrajectoryFilter::Impl::try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_PTs.load(fmt::format("{}PTs.{}", path, frame))      //
           && dump_EEs.load(fmt::format("{}EEs.{}", path, frame))   //
           && dump_PEs.load(fmt::format("{}PEs.{}", path, frame))   //
           && dump_PPs.load(fmt::format("{}PPs.{}", path, frame));  //
}

void SimplexTrajectoryFilter::Impl::apply_recover(RecoverInfo& info)
{
    dump_PTs.apply_to(recovered_PT);
    dump_EEs.apply_to(recovered_EE);
    dump_PEs.apply_to(recovered_PE);
    dump_PPs.apply_to(recovered_PP);

    // temporary switch to the recovered PHs, which will be used
    // in the record_friction_candidates() function to recover the friction candidates.
    PTs = recovered_PT.cview();
    EEs = recovered_EE.cview();
    PEs = recovered_PE.cview();
    PPs = recovered_PP.cview();
}

void SimplexTrajectoryFilter::Impl::clear_recover(RecoverInfo& info)
{
    dump_PTs.clean_up();
    dump_EEs.clean_up();
    dump_PEs.clean_up();
    dump_PPs.clean_up();
}

bool SimplexTrajectoryFilter::do_dump(DumpInfo& info)
{
    return m_impl.dump(info);
}

bool SimplexTrajectoryFilter::do_try_recover(RecoverInfo& info)
{
    return m_impl.try_recover(info);
}

void SimplexTrajectoryFilter::do_apply_recover(RecoverInfo& info)
{
    m_impl.apply_recover(info);
}

void SimplexTrajectoryFilter::do_clear_recover(RecoverInfo& info)
{
    m_impl.clear_recover(info);
}
}// namespace uipc::backend::luisa
