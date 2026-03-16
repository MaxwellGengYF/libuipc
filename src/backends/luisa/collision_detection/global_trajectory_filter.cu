#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class SimSystemCreator<luisa::GlobalTrajectoryFilter>
{
  public:
    static U<luisa::GlobalTrajectoryFilter> create(luisa::SimEngine& engine)
    {
        auto contact_enable = engine.world().scene().config().find<IndexT>("contact/enable");
        if(contact_enable->view()[0])
            return make_unique<luisa::GlobalTrajectoryFilter>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend


namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(GlobalTrajectoryFilter);

void GlobalTrajectoryFilter::do_build()
{
    auto& config               = world().scene().config();
    auto  friction_enable_attr = config.find<IndexT>("contact/friction/enable");

    m_impl.friction_enabled = friction_enable_attr->view()[0];

    m_impl.global_contact_manager = require<GlobalContactManager>();

    on_init_scene([&] { m_impl.init(engine()); });
}

void GlobalTrajectoryFilter::add_filter(TrajectoryFilter* filter)
{
    check_state(SimEngineState::BuildSystems, "add_filter()");
    UIPC_ASSERT(filter != nullptr, "Input TrajectoryFilter is nullptr.");
    m_impl.filters.register_sim_system(*filter);
}

void GlobalTrajectoryFilter::Impl::init(SimEngine& engine)
{
    auto filter_view = filters.view();
    auto& device = engine.luisa_device();
    tois = device.create_buffer<Float>(filter_view.size());
    h_tois.resize(filter_view.size());
}

void GlobalTrajectoryFilter::detect(Float alpha)
{
    for(auto filter : m_impl.filters.view())
    {
        DetectInfo info;
        info.m_alpha = alpha;
        filter->detect(info);
    }
}

void GlobalTrajectoryFilter::filter_active()
{
    if(m_impl.global_contact_manager->cfl_enabled())
    {
        auto is_active =
            m_impl.global_contact_manager->m_impl.vert_is_active_contact.view();
        // clear the active flag using fill command on stream
        auto& stream = engine().compute_stream();
        stream << is_active.fill(0);
    }

    for(auto filter : m_impl.filters.view())
    {
        FilterActiveInfo info(&m_impl);
        filter->filter_active(info);
    }
}

Float GlobalTrajectoryFilter::Impl::filter_toi(Float alpha, luisa::compute::Stream& stream)
{
    auto filter_view = filters.view();
    for(auto&& [i, filter] : enumerate(filter_view))
    {
        FilterTOIInfo info;
        info.m_toi   = tois.view().subview(i, 1);
        info.m_alpha = alpha;
        filter->filter_toi(info);
    }

    // Copy from device to host
    stream << tois.copy_to(h_tois.data())
           << luisa::compute::synchronize();

    if constexpr(uipc::RUNTIME_CHECK)
    {
        for(auto&& [i, toi] : enumerate(h_tois))
        {
            UIPC_ASSERT(toi > 0.0f, "Invalid toi[{}] value: {}", filter_view[i]->name(), toi);
        }
    }

    auto min_toi = *std::min_element(h_tois.begin(), h_tois.end());

    return min_toi < 1.0 ? min_toi : 1.0;
}

Float GlobalTrajectoryFilter::filter_toi(Float alpha)
{
    auto& stream = engine().compute_stream();
    return m_impl.filter_toi(alpha, stream);
}

void GlobalTrajectoryFilter::record_friction_candidates()
{
    // Check if friction candidates should be discarded before recording new ones
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    if(m_impl.should_discard_friction_candidates)
    {
        clear_friction_candidates();
        m_impl.should_discard_friction_candidates = false;
        // No need to record friction candidates if they are discarded
        // Just early return
        return;
    }

    for(auto filter : m_impl.filters.view())
    {
        RecordFrictionCandidatesInfo info;
        filter->record_friction_candidates(info);
    }
}

void GlobalTrajectoryFilter::label_active_vertices()
{
    for(auto filter : m_impl.filters.view())
    {
        LabelActiveVerticesInfo info(&m_impl);
        filter->label_active_vertices(info);
    }
}

void GlobalTrajectoryFilter::clear_friction_candidates()
{
    for(auto filter : m_impl.filters.view())
    {
        filter->clear_friction_candidates();
    }
}

void GlobalTrajectoryFilter::require_discard_friction()
{
    m_impl.should_discard_friction_candidates = true;
}

BufferView<IndexT> GlobalTrajectoryFilter::LabelActiveVerticesInfo::vert_is_active() const noexcept
{
    return m_impl->global_contact_manager->m_impl.vert_is_active_contact.view();
}

void GlobalTrajectoryFilter::do_apply_recover(RecoverInfo& info)
{
    // Friction candidates are already recovered, no need to discard them.
    m_impl.should_discard_friction_candidates = false;
}
}  // namespace uipc::backend::luisa
