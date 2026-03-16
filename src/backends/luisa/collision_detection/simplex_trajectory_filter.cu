#include <collision_detection/simplex_trajectory_filter.h>
#include <sim_engine.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

namespace uipc::backend::luisa
{
void SimplexTrajectoryFilter::do_build()
{
    m_impl.global_vertex_manager = require<GlobalVertexManager>();
    m_impl.global_simplicial_surface_manager = require<GlobalSimplicialSurfaceManager>();
    m_impl.global_contact_manager  = require<GlobalContactManager>();
    m_impl.global_body_manager     = require<GlobalBodyManager>();
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

void SimplexTrajectoryFilter::Impl::label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info)
{
    auto& engine = this->global_vertex_manager->engine();
    auto& stream = engine.compute_stream();
    auto& device = engine.luisa_device();

    // Kernel for PTs
    if(PTs.size() > 0)
    {
        Kernel1D label_pt_kernel = [&](BufferVar<Vector4i> PTs_buf, BufferVar<IndexT> is_active) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < PTs_buf.size())
            {
                auto PT = PTs_buf.read(i);
                for(int j = 0; j < 4; ++j)
                {
                    auto P = PT[j];
                    $if(is_active.read(P) == 0)
                    {
                        is_active.atomic(P).exchange(1);
                    };
                }
            };
        };
        auto shader = device.compile(label_pt_kernel);
        stream << shader(PTs, info.vert_is_active()).dispatch(PTs.size());
    }

    // Kernel for EEs
    if(EEs.size() > 0)
    {
        Kernel1D label_ee_kernel = [&](BufferVar<Vector4i> EEs_buf, BufferVar<IndexT> is_active) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < EEs_buf.size())
            {
                auto EE = EEs_buf.read(i);
                for(int j = 0; j < 4; ++j)
                {
                    auto P = EE[j];
                    $if(is_active.read(P) == 0)
                    {
                        is_active.atomic(P).exchange(1);
                    };
                }
            };
        };
        auto shader = device.compile(label_ee_kernel);
        stream << shader(EEs, info.vert_is_active()).dispatch(EEs.size());
    }

    // Kernel for PEs
    if(PEs.size() > 0)
    {
        Kernel1D label_pe_kernel = [&](BufferVar<Vector3i> PEs_buf, BufferVar<IndexT> is_active) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < PEs_buf.size())
            {
                auto PE = PEs_buf.read(i);
                for(int j = 0; j < 3; ++j)
                {
                    auto P = PE[j];
                    $if(is_active.read(P) == 0)
                    {
                        is_active.atomic(P).exchange(1);
                    };
                }
            };
        };
        auto shader = device.compile(label_pe_kernel);
        stream << shader(PEs, info.vert_is_active()).dispatch(PEs.size());
    }

    // Kernel for PPs
    if(PPs.size() > 0)
    {
        Kernel1D label_pp_kernel = [&](BufferVar<Vector2i> PPs_buf, BufferVar<IndexT> is_active) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < PPs_buf.size())
            {
                auto PP = PPs_buf.read(i);
                for(int j = 0; j < 2; ++j)
                {
                    auto P = PP[j];
                    $if(is_active.read(P) == 0)
                    {
                        is_active.atomic(P).exchange(1);
                    };
                }
            };
        };
        auto shader = device.compile(label_pp_kernel);
        stream << shader(PPs, info.vert_is_active()).dispatch(PPs.size());
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
    auto& engine = this->global_vertex_manager->engine();
    auto& stream = engine.compute_stream();
    auto& device = engine.luisa_device();

    // PT
    loose_resize(friction_PT, PTs.size(), device);
    if(PTs.size() > 0)
    {
        stream << friction_PT.copy_from(PTs);
    }

    // EE
    loose_resize(friction_EE, EEs.size(), device);
    if(EEs.size() > 0)
    {
        stream << friction_EE.copy_from(EEs);
    }

    // PE
    loose_resize(friction_PE, PEs.size(), device);
    if(PEs.size() > 0)
    {
        stream << friction_PE.copy_from(PEs);
    }

    // PP
    loose_resize(friction_PP, PPs.size(), device);
    if(PPs.size() > 0)
    {
        stream << friction_PP.copy_from(PPs);
    }

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

BufferView<Float> SimplexTrajectoryFilter::BaseInfo::d_hats() const noexcept
{
    return m_impl->global_vertex_manager->d_hats();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::v2b() const noexcept
{
    return m_impl->global_vertex_manager->body_ids();
}

BufferView<Vector3> SimplexTrajectoryFilter::BaseInfo::positions() const noexcept
{
    return m_impl->global_vertex_manager->positions();
}

BufferView<Vector3> SimplexTrajectoryFilter::BaseInfo::rest_positions() const noexcept
{
    return m_impl->global_vertex_manager->rest_positions();
}

BufferView<Float> SimplexTrajectoryFilter::BaseInfo::thicknesses() const noexcept
{
    return m_impl->global_vertex_manager->thicknesses();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::dimensions() const noexcept
{
    return m_impl->global_vertex_manager->dimensions();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::body_self_collision() const noexcept
{
    return m_impl->global_body_manager->self_collision();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::codim_vertices() const noexcept
{
    return m_impl->global_simplicial_surface_manager->codim_vertices();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::surf_vertices() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_vertices();
}

BufferView<Vector2i> SimplexTrajectoryFilter::BaseInfo::surf_edges() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_edges();
}

BufferView<Vector3i> SimplexTrajectoryFilter::BaseInfo::surf_triangles() const noexcept
{
    return m_impl->global_simplicial_surface_manager->surf_triangles();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::contact_element_ids() const noexcept
{
    return m_impl->global_vertex_manager->contact_element_ids();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::subscene_element_ids() const noexcept
{
    return m_impl->global_vertex_manager->subscene_element_ids();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::contact_mask_tabular() const noexcept
{
    return m_impl->global_contact_manager->contact_mask_tabular();
}

BufferView<IndexT> SimplexTrajectoryFilter::BaseInfo::subscene_mask_tabular() const noexcept
{
    return m_impl->global_contact_manager->subscene_mask_tabular();
}

BufferView<Vector4i> SimplexTrajectoryFilter::PTs() const noexcept
{
    return m_impl.PTs;
}

BufferView<Vector4i> SimplexTrajectoryFilter::EEs() const noexcept
{
    return m_impl.EEs;
}

BufferView<Vector3i> SimplexTrajectoryFilter::PEs() const noexcept
{
    return m_impl.PEs;
}

BufferView<Vector2i> SimplexTrajectoryFilter::PPs() const noexcept
{
    return m_impl.PPs;
}

BufferView<Vector4i> SimplexTrajectoryFilter::friction_PTs() const noexcept
{
    return m_impl.friction_PT;
}

BufferView<Vector4i> SimplexTrajectoryFilter::friction_EEs() const noexcept
{
    return m_impl.friction_EE;
}

BufferView<Vector3i> SimplexTrajectoryFilter::friction_PEs() const noexcept
{
    return m_impl.friction_PE;
}


BufferView<Vector2i> SimplexTrajectoryFilter::friction_PPs() const noexcept
{
    return m_impl.friction_PP;
}

BufferView<Vector3> SimplexTrajectoryFilter::DetectInfo::displacements() const noexcept
{
    return m_impl->global_vertex_manager->displacements();
}

void SimplexTrajectoryFilter::FilterActiveInfo::PTs(BufferView<Vector4i> PTs) noexcept
{
    m_impl->PTs = PTs;
}

void SimplexTrajectoryFilter::FilterActiveInfo::EEs(BufferView<Vector4i> EEs) noexcept
{
    m_impl->EEs = EEs;
}

void SimplexTrajectoryFilter::FilterActiveInfo::PEs(BufferView<Vector3i> PEs) noexcept
{
    m_impl->PEs = PEs;
}

void SimplexTrajectoryFilter::FilterActiveInfo::PPs(BufferView<Vector2i> PPs) noexcept
{
    m_impl->PPs = PPs;
}

BufferView<Float> SimplexTrajectoryFilter::FilterTOIInfo::toi() noexcept
{
    return m_toi;
}

void SimplexTrajectoryFilter::do_clear_friction_candidates()
{
    // In LuisaCompute, buffers are immutable in size
    // We reset the views to empty views to indicate no friction candidates
    // The actual buffers will be resized when needed in record_friction_candidates
    m_impl.friction_PT = Buffer<Vector4i>{};
    m_impl.friction_EE = Buffer<Vector4i>{};
    m_impl.friction_PE = Buffer<Vector3i>{};
    m_impl.friction_PP = Buffer<Vector2i>{};
}

bool SimplexTrajectoryFilter::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_PTs.dump(fmt::format("{}PTs.{}", path, frame), PTs)
           && dump_EEs.dump(fmt::format("{}EEs.{}", path, frame), EEs)
           && dump_PEs.dump(fmt::format("{}PEs.{}", path, frame), PEs)
           && dump_PPs.dump(fmt::format("{}PPs.{}", path, frame), PPs);
}

bool SimplexTrajectoryFilter::Impl::try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_PTs.load(fmt::format("{}PTs.{}", path, frame))
           && dump_EEs.load(fmt::format("{}EEs.{}", path, frame))
           && dump_PEs.load(fmt::format("{}PEs.{}", path, frame))
           && dump_PPs.load(fmt::format("{}PPs.{}", path, frame));
}

void SimplexTrajectoryFilter::Impl::apply_recover(RecoverInfo& info)
{
    dump_PTs.apply_to(recovered_PT);
    dump_EEs.apply_to(recovered_EE);
    dump_PEs.apply_to(recovered_PE);
    dump_PPs.apply_to(recovered_PP);

    // temporary switch to the recovered PHs, which will be used
    // in the record_friction_candidates() function to recover the friction candidates.
    PTs = recovered_PT.view();
    EEs = recovered_EE.view();
    PEs = recovered_PE.view();
    PPs = recovered_PP.view();
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
}  // namespace uipc::backend::luisa
