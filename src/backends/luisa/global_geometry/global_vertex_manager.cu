#include <global_geometry/global_vertex_manager.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/range.h>
#include <global_geometry/vertex_reporter.h>
#include <sim_engine.h>
#include <collision_detection/global_trajectory_filter.h>

/*************************************************************************************************
* Core Implementation
*************************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(GlobalVertexManager);

void GlobalVertexManager::do_build()
{
    auto d_hat = world().scene().config().find<Float>("contact/d_hat");
    m_impl.default_d_hat            = d_hat->view()[0];
    m_impl.global_trajectory_filter = find<GlobalTrajectoryFilter>();
    m_impl.sim_engine = &engine();
}

void GlobalVertexManager::Impl::init()
{
    auto vertex_reporter_view = vertex_reporters.view();

    // 1) Setup index for each vertex reporter

    // ref: https://github.com/spiriMirror/libuipc/issues/271
    // Sort by uid to ensure the order is consistent
    std::ranges::sort(vertex_reporter_view,
                      [](const VertexReporter* l, const VertexReporter* r)
                      { return l->uid() < r->uid(); });
    for(auto&& [i, R] : enumerate(vertex_reporter_view))
        R->m_index = i;

    // 2) Count the number of vertices reported by each reporter
    auto N = vertex_reporter_view.size();
    reporter_vertex_offsets_counts.resize(sim_engine->luisa_device(), N);

    span<IndexT> reporter_vertex_counts = reporter_vertex_offsets_counts.counts();

    for(auto&& [i, R] : enumerate(vertex_reporter_view))
    {
        VertexCountInfo info;
        R->report_count(info);
        // get count back
        reporter_vertex_counts[i] = info.m_count;
    }
    reporter_vertex_offsets_counts.scan(sim_engine->compute_stream());
    SizeT total_count = reporter_vertex_offsets_counts.total_count();

    // 3) Initialize buffers for vertex attributes
    auto& device = sim_engine->luisa_device();
    coindices = device.create_buffer<IndexT>(total_count);
    positions = device.create_buffer<Vector3>(total_count);
    rest_positions = device.create_buffer<Vector3>(total_count);
    safe_positions = device.create_buffer<Vector3>(total_count);
    contact_element_ids = device.create_buffer<IndexT>(total_count);
    subscene_element_ids = device.create_buffer<IndexT>(total_count);
    thicknesses = device.create_buffer<Float>(total_count);
    dimensions = device.create_buffer<IndexT>(total_count);
    displacements = device.create_buffer<Vector3>(total_count);
    displacement_norms = device.create_buffer<Float>(total_count);
    body_ids = device.create_buffer<IndexT>(total_count);
    d_hats = device.create_buffer<Float>(total_count);
    
    // Single-element buffers for reductions
    axis_max_disp = device.create_buffer<Float>(1);
    max_disp_norm = device.create_buffer<Float>(1);
    min_pos = device.create_buffer<Vector3>(1);
    max_pos = device.create_buffer<Vector3>(1);

    // Initialize with default values
    sim_engine->compute_stream() << contact_element_ids.view().fill(0)
                                 << subscene_element_ids.view().fill(0)
                                 << thicknesses.view().fill(0.0f)
                                 << dimensions.view().fill(3)  // default 3D
                                 << displacements.view().fill(Vector3::Zero())
                                 << displacement_norms.view().fill(0.0f)
                                 << body_ids.view().fill(-1)  // -1 means no care about body id
                                 << d_hats.view().fill(default_d_hat)
                                 << luisa::compute::synchronize();

    // 4) Create the subviews for each attribute_reporter,
    //    so that each reporter can write to its own subview
    for(auto&& [i, R] : enumerate(vertex_reporter_view))
    {
        VertexAttributeInfo attributes{
            this,
            i,
            0  // frame = 0 for initialization
        };
        R->report_attributes(attributes);
    }

    // 5) Initialize previous positions and safe positions
    sim_engine->compute_stream() << prev_positions.view().copy_from(positions.view())
                                 << safe_positions.view().copy_from(positions.view())
                                 << luisa::compute::synchronize();

    // 6) Other initializations
    // axis_max_disp is already a buffer, no direct assignment
}

void GlobalVertexManager::Impl::update_attributes(SizeT frame)
{
    auto vertex_reporter_view = vertex_reporters.view();

    for(auto&& [i, R] : enumerate(vertex_reporter_view))
    {
        VertexAttributeInfo attributes{this, i, frame};
        R->report_attributes(attributes);
    }
}

void GlobalVertexManager::Impl::rebuild()
{
    UIPC_ASSERT(false, "Not implemented yet");
}

void GlobalVertexManager::add_reporter(VertexReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_impl.vertex_reporters.register_sim_system(*reporter);
}

void GlobalVertexManager::Impl::step_forward(Float alpha)
{
    auto vertex_count = positions.size();
    if(vertex_count == 0)
        return;

    Kernel1D step_forward_kernel = [&](BufferVar<Vector3> pos_buffer,
                                       BufferVar<Vector3> safe_pos_buffer,
                                       BufferVar<Vector3> disp_buffer,
                                       Float alpha_val) noexcept
    {
        auto i = dispatch_idx().x;
        if(i >= vertex_count)
            return;
        auto safe_pos = safe_pos_buffer.read(i);
        auto disp = disp_buffer.read(i);
        pos_buffer.write(i, safe_pos + alpha_val * disp);
    };

    auto shader = sim_engine->luisa_device().compile(step_forward_kernel);
    sim_engine->compute_stream() << shader(positions, safe_positions, displacements, alpha).dispatch(vertex_count);
}

void GlobalVertexManager::Impl::collect_vertex_displacements()
{
    for(auto&& [i, R] : enumerate(vertex_reporters.view()))
    {
        VertexDisplacementInfo vd{this, i};
        R->report_displacements(vd);
    }
}

void GlobalVertexManager::Impl::record_prev_positions()
{
    sim_engine->compute_stream() << prev_positions.view().copy_from(positions.view());
}

void GlobalVertexManager::Impl::record_start_point()
{
    sim_engine->compute_stream() << safe_positions.view().copy_from(positions.view());
}

Float GlobalVertexManager::Impl::compute_axis_max_displacement()
{
    // Host-side computation for reduction
    // Download displacements and compute max absolute value on CPU
    auto vertex_count = displacements.size();
    if(vertex_count == 0)
        return 0.0f;

    luisa::vector<Vector3> host_displacements(vertex_count);
    sim_engine->compute_stream() << displacements.view().copy_to(host_displacements.data())
                                 << luisa::compute::synchronize();

    Float max_disp = 0.0f;
    for(const auto& disp : host_displacements)
    {
        for(int j = 0; j < 3; ++j)
        {
            max_disp = std::max(max_disp, std::abs(disp[j]));
        }
    }

    // Store result in buffer
    sim_engine->compute_stream() << axis_max_disp.view().copy_from(&max_disp)
                                 << luisa::compute::synchronize();

    return max_disp;
}

AABB GlobalVertexManager::Impl::compute_vertex_bounding_box()
{
    Float max_float = std::numeric_limits<Float>::max();
    
    auto vertex_count = positions.size();
    if(vertex_count == 0)
    {
        Vector3 min_pos_host{0, 0, 0};
        Vector3 max_pos_host{0, 0, 0};
        vertex_bounding_box = AABB{min_pos_host.cast<float>(), max_pos_host.cast<float>()};
        return vertex_bounding_box;
    }

    // Download positions and compute bounding box on CPU
    luisa::vector<Vector3> host_positions(vertex_count);
    sim_engine->compute_stream() << positions.view().copy_to(host_positions.data())
                                 << luisa::compute::synchronize();

    Vector3 min_pos_host{max_float, max_float, max_float};
    Vector3 max_pos_host{-max_float, -max_float, -max_float};

    for(const auto& pos : host_positions)
    {
        min_pos_host = min_pos_host.cwiseMin(pos);
        max_pos_host = max_pos_host.cwiseMax(pos);
    }

    // Store results in buffers
    sim_engine->compute_stream() << min_pos.view().copy_from(min_pos_host.data())
                                 << max_pos.view().copy_from(max_pos_host.data())
                                 << luisa::compute::synchronize();

    vertex_bounding_box = AABB{min_pos_host.cast<float>(), max_pos_host.cast<float>()};
    return vertex_bounding_box;
}
}  // namespace uipc::backend::luisa

// Dump & Recover:
namespace uipc::backend::luisa
{
bool GlobalVertexManager::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_positions.dump(fmt::format("{}positions.{}", path, frame), positions, sim_engine->compute_stream())  //
           && dump_prev_positions.dump(fmt::format("{}prev_positions.{}", path, frame),
                                       prev_positions, sim_engine->compute_stream());
}

bool GlobalVertexManager::Impl::try_recover(RecoverInfo& info)
{
    auto path = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    return dump_positions.load(fmt::format("{}positions.{}", path, info.frame()))  //
           && dump_prev_positions.load(
               fmt::format("{}prev_positions.{}", path, info.frame()));
}

void GlobalVertexManager::Impl::apply_recover(RecoverInfo& info)
{
    dump_positions.apply_to(positions, sim_engine->compute_stream());
    dump_prev_positions.apply_to(prev_positions, sim_engine->compute_stream());
}

void GlobalVertexManager::Impl::clear_recover(RecoverInfo& info)
{
    dump_positions.clean_up();
    dump_prev_positions.clean_up();
}
}  // namespace uipc::backend::luisa


/*************************************************************************************************
* API Implementation
*************************************************************************************************/
namespace uipc::backend::luisa
{
void GlobalVertexManager::VertexCountInfo::count(SizeT count) noexcept
{
    m_count = count;
}

void GlobalVertexManager::VertexCountInfo::changeable(bool is_changable) noexcept
{
    m_changable = is_changable;
}

GlobalVertexManager::VertexAttributeInfo::VertexAttributeInfo(Impl* impl, SizeT index, SizeT frame) noexcept
    : m_impl(impl)
    , m_index(index)
    , m_frame(frame)
{
}

luisa::compute::BufferView<Vector3> GlobalVertexManager::VertexAttributeInfo::rest_positions() const noexcept
{
    return m_impl->subview(m_impl->rest_positions, m_index);
}

luisa::compute::BufferView<Float> GlobalVertexManager::VertexAttributeInfo::thicknesses() const noexcept
{
    return m_impl->subview(m_impl->thicknesses, m_index);
}

luisa::compute::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::coindices() const noexcept
{
    return m_impl->subview(m_impl->coindices, m_index);
}

luisa::compute::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::dimensions() const noexcept
{
    return m_impl->subview(m_impl->dimensions, m_index);
}

luisa::compute::BufferView<Vector3> GlobalVertexManager::VertexAttributeInfo::positions() const noexcept
{
    return m_impl->subview(m_impl->positions, m_index);
}

luisa::compute::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::contact_element_ids() const noexcept
{
    return m_impl->subview(m_impl->contact_element_ids, m_index);
}

luisa::compute::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::subscene_element_ids() const noexcept
{
    return m_impl->subview(m_impl->subscene_element_ids, m_index);
}

luisa::compute::BufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::body_ids() const noexcept
{
    return m_impl->subview(m_impl->body_ids, m_index);
}

luisa::compute::BufferView<Float> GlobalVertexManager::VertexAttributeInfo::d_hats() const noexcept
{
    return m_impl->subview(m_impl->d_hats, m_index);
}

void GlobalVertexManager::VertexAttributeInfo::require_discard_friction() const noexcept
{
    // If the vertex attributes are updated in a way that will ruin the friction computation
    // we need to discard the friction information in the global trajectory filter.
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    if(m_impl->global_trajectory_filter)
        m_impl->global_trajectory_filter->require_discard_friction();
}

SizeT GlobalVertexManager::VertexAttributeInfo::frame() const noexcept
{
    return m_frame;
}

GlobalVertexManager::VertexDisplacementInfo::VertexDisplacementInfo(Impl* impl, SizeT index) noexcept
    : m_impl(impl)
    , m_index(index)
{
}

luisa::compute::BufferView<Vector3> GlobalVertexManager::VertexDisplacementInfo::displacements() const noexcept
{
    return m_impl->subview(m_impl->displacements, m_index);
}

luisa::compute::BufferView<const IndexT> GlobalVertexManager::VertexDisplacementInfo::coindices() const noexcept
{
    return m_impl->subview(m_impl->coindices, m_index);
}

bool GlobalVertexManager::do_dump(DumpInfo& info)
{
    return m_impl.dump(info);
}

bool GlobalVertexManager::do_try_recover(RecoverInfo& info)
{
    return m_impl.try_recover(info);
}

void GlobalVertexManager::do_apply_recover(RecoverInfo& info)
{
    m_impl.apply_recover(info);
}

void GlobalVertexManager::do_clear_recover(RecoverInfo& info)
{
    m_impl.clear_recover(info);
}

void GlobalVertexManager::init()
{
    m_impl.init();
}

void GlobalVertexManager::update_attributes()
{
    m_impl.update_attributes(engine().frame());
}

void GlobalVertexManager::rebuild()
{
    m_impl.rebuild();
}

void GlobalVertexManager::record_prev_positions()
{
    m_impl.record_prev_positions();
}

void GlobalVertexManager::collect_vertex_displacements()
{
    m_impl.collect_vertex_displacements();
}

luisa::compute::BufferView<const IndexT> GlobalVertexManager::coindices() const noexcept
{
    return m_impl.coindices.view();
}

luisa::compute::BufferView<const IndexT> GlobalVertexManager::body_ids() const noexcept
{
    return m_impl.body_ids.view();
}

luisa::compute::BufferView<const Float> GlobalVertexManager::d_hats() const noexcept
{
    return m_impl.d_hats.view();
}

luisa::compute::BufferView<const Vector3> GlobalVertexManager::positions() const noexcept
{
    return m_impl.positions.view();
}

luisa::compute::BufferView<const Vector3> GlobalVertexManager::prev_positions() const noexcept
{
    return m_impl.prev_positions.view();
}

luisa::compute::BufferView<const Vector3> GlobalVertexManager::rest_positions() const noexcept
{
    return m_impl.rest_positions.view();
}

luisa::compute::BufferView<const Vector3> GlobalVertexManager::safe_positions() const noexcept
{
    return m_impl.safe_positions.view();
}

luisa::compute::BufferView<const IndexT> GlobalVertexManager::contact_element_ids() const noexcept
{
    return m_impl.contact_element_ids.view();
}

luisa::compute::BufferView<const IndexT> GlobalVertexManager::subscene_element_ids() const noexcept
{
    return m_impl.subscene_element_ids.view();
}

luisa::compute::BufferView<const Vector3> GlobalVertexManager::displacements() const noexcept
{
    return m_impl.displacements.view();
}

luisa::compute::BufferView<const Float> GlobalVertexManager::thicknesses() const noexcept
{
    return m_impl.thicknesses.view();
}

Float GlobalVertexManager::compute_axis_max_displacement()
{
    return m_impl.compute_axis_max_displacement();
}

AABB GlobalVertexManager::compute_vertex_bounding_box()
{
    return m_impl.compute_vertex_bounding_box();
}

void GlobalVertexManager::step_forward(Float alpha)
{
    m_impl.step_forward(alpha);
}

void GlobalVertexManager::record_start_point()
{
    m_impl.record_start_point();
}

luisa::compute::BufferView<const IndexT> GlobalVertexManager::dimensions() const noexcept
{
    return m_impl.dimensions.view();
}

AABB GlobalVertexManager::vertex_bounding_box() const noexcept
{
    return m_impl.vertex_bounding_box;
}
}  // namespace uipc::backend::luisa
