#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
// The test target's include path puts src/backends/none first, so the engine shell
// header must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>
#include <uipc/common/enumerate.h>
#include <algorithm>
/*************************************************************************************************
 * T7.2a — GlobalTrajectoryFilter (port of
 * src/backends/cuda/collision_detection/global_trajectory_filter.cu, 161 lines).
 *
 * Transcribed verbatim except for the documented type mappings and the test seam:
 *  * `cuda_tool::VarView<Float>` -> `lud::LBufferView<Float>` (the port's spelling of a
 *    one-element device view, cookbook §7.19/§7.22);
 *  * `info.m_toi = cuda_tool::VarView<Float>{tois.data() + i}` -> `tois.view(i, 1)`;
 *  * `tois.view().copy_to(h_tois.data())` -> `tois.cview().copy_to(span<Float>{h_tois})`;
 *  * the `RUNTIME_CHECK` block's `UIPC_ASSERT(toi > 0.0f, ...)` is kept verbatim (it
 *    fires on a filter that reported a non-positive toi -- the reference's invariant);
 *  * the reference's `if(m_impl.global_contact_manager->cfl_enabled())` is *not* guarded
 *    by a null test there either: the manager is a hard dependency of do_build
 *    (`require<GlobalContactManager>()`), so the slot is always resolved.
 *************************************************************************************************/
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
            return uipc::make_unique<luisa::GlobalTrajectoryFilter>(engine);
        return nullptr;
    }
};
}// namespace uipc::backend


namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(GlobalTrajectoryFilter);

void GlobalTrajectoryFilter::do_build()
{
    auto& config               = world().scene().config();
    auto  friction_enable_attr = config.find<IndexT>("contact/friction/enable");

    m_impl.friction_enabled = friction_enable_attr->view()[0];

    m_impl.global_contact_manager = require<GlobalContactManager>();

    on_init_scene([&] { m_impl.init(); });
}

void GlobalTrajectoryFilter::add_filter(TrajectoryFilter* filter)
{
    check_state(SimEngineState::BuildSystems, "add_filter()");
    UIPC_ASSERT(filter != nullptr, "Input TrajectoryFilter is nullptr.");
    m_impl.filters.register_sim_system(*filter);
}

void GlobalTrajectoryFilter::Impl::init()
{
    auto filter_view = filters.view();
    tois.resize(filter_view.size());
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
        auto is_acitive =
            m_impl.global_contact_manager->m_impl.vert_is_active_contact.view();
        is_acitive.fill(0);  // clear the active flag
    }

    for(auto filter : m_impl.filters.view())
    {
        FilterActiveInfo info(&m_impl);
        filter->filter_active(info);
    }
}

Float GlobalTrajectoryFilter::Impl::filter_toi(Float alpha)
{
    auto filter_view = filters.view();
    for(auto&& [i, filter] : enumerate(filter_view))
    {
        FilterTOIInfo info;
        info.m_toi   = tois.view(static_cast<size_t>(i), 1u);
        info.m_alpha = alpha;
        filter->filter_toi(info);
    }
    tois.cview().copy_to(span<Float>{h_tois});

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
    return m_impl.filter_toi(alpha);
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

lud::LBufferView<IndexT> GlobalTrajectoryFilter::LabelActiveVerticesInfo::vert_is_active() const noexcept
{
    return m_impl->global_contact_manager->m_impl.vert_is_active_contact.view();
}

void GlobalTrajectoryFilter::do_apply_recover(RecoverInfo& info)
{
    // Friction candidates are already recovered, no need to discard them.
    m_impl.should_discard_friction_candidates = false;
}

Float GlobalTrajectoryFilter::run_stage(Stage stage, Float alpha)
{
    switch(stage)
    {
        case Stage::Detect:
            detect(alpha);
            return 1.0;
        case Stage::FilterActive:
            filter_active();
            return 1.0;
        case Stage::FilterTOI:
            return filter_toi(alpha);
        case Stage::RecordFrictionCandidates:
            record_friction_candidates();
            return 1.0;
        case Stage::LabelActiveVertices:
            label_active_vertices();
            return 1.0;
        case Stage::ClearFrictionCandidates:
            clear_friction_candidates();
            return 1.0;
    }
    return 1.0;
}
}// namespace uipc::backend::luisa
