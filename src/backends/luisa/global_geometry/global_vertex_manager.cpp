#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/vertex_reporter.h>
#include <uipc/common/enumerate.h>
#include <sim_engine.h>
#include <uipc/common/range.h>
#include <frame_stream.h>

// NVTX range for the device->host read-back (BN1 I2, bench_plan.md 13.3).
#include <backends/common/nvtx_range.h>
// T4.2: the reference's GlobalVertexManager::do_build() also resolves the
// GlobalJointDofManager slot (now that the system is ported).
#include <joint_dof_system/global_joint_dof_manager.h>
// T7.2a: ... and the GlobalTrajectoryFilter slot (the trajectory-filter manager is
// ported; `find<GlobalTrajectoryFilter>()` needs the complete type).
#include <collision_detection/global_trajectory_filter.h>

#include <luisa_device/linear/algorithms.h>

#include <array>
#include <limits>
#include <memory>
#include <vector>

/*************************************************************************************************
* Core Implementation
*
* Port notes (cookbook §3, §7.9-§7.13)
* ------------------------------------
*  * Every `__global__ void k(...) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; ... }`
*    becomes a `device.compile<1>` kernel dispatched over the frame LStream; the
*    early-out becomes `$if (i < n) { ... };` and `n` stays an explicit argument
*    (the reference passes it too, and a dispatch may be rounded up).
*  * Kernels are compiled once and deliberately leaked through
*    `linalg::detail::cached_kernel` (cookbook §7.10 trap 1).
*  * Vector arithmetic on a buffer element happens on the numerics layer:
*    `load_vec3(element)` -> `lps::LVec<Var<double>,3>` -> `store_vec3(...)`.
*  * The reference's `CUB DeviceReduce` with `cwiseMin/cwiseMax` over Vector3 is
*    replaced by a pack kernel (per-component, transposed into one contiguous
*    array) + `device_reduce_min/max<double>` per component: min/max are exact and
*    order-independent, so the result is identical to the reference's.
*  * A host read-back commits the frame CommandList first (cuda_tool's `copy_to`
*    synchronises; a bare LC copy on the main stream would not be ordered after
*    commands still recorded in an open frame), see `read_back`.
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------
auto& step_forward_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<LVector3>,
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<LVector3> pos,
           lcrt::BufferVar<LVector3> safe_pos,
           lcrt::BufferVar<LVector3> disp,
           lcrt::Var<double>         alpha,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto s = load_vec3(safe_pos->read(i));
                auto d = load_vec3(disp->read(i));
                lcrt::Var<LVector3> out;
                store_vec3(out, s + alpha * d);
                pos->write(i, out);
            };
        });
    return k;
}

auto& setup_ccd_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<LVector3>,
                                                uint32_t>(
        [](lcrt::BufferVar<LVector3> pos,
           lcrt::BufferVar<LVector3> tmp_pos,
           lcrt::BufferVar<LVector3> disp,
           lcrt::BufferVar<LVector3> base_pos,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto p = load_vec3(pos->read(i));
                auto b = load_vec3(base_pos->read(i));
                lcrt::Var<LVector3> d;
                store_vec3(d, p - b);
                disp->write(i, d);
                lcrt::Var<LVector3> t;
                store_vec3(t, p);
                tmp_pos->write(i, t);
                // NOTE: a buffer element is written as a Var<LVector3>; a bare
                // LVec<Var<double>,3> is not an Expr<LVector3>.
                lcrt::Var<LVector3> src;
                store_vec3(src, b);
                pos->write(i, src);
            };
        });
    return k;
}

auto& propagate_relative_d_hat_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>, double, double, uint32_t>(
            [](lcrt::BufferVar<double> d_hats,
               lcrt::Var<double>       old_default,
               lcrt::Var<double>       new_d_hat,
               lcrt::Var<uint32_t>     n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    // only rewrite entries still holding the absolute contact/d_hat
                    // default; per-geometry meta d_hat values are left untouched
                    $if(d_hats->read(i) == old_default) { d_hats->write(i, new_d_hat); };
                };
            });
    return k;
}

/// comps[c * n + i] = positions[i][c]  (per-component, so the scalar reductions
/// of linalg can produce a component-wise min/max)
auto& pack_components_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,
                                                lcrt::Buffer<LVector3>,
                                                uint32_t>(
        [](lcrt::BufferVar<double>   comps,
           lcrt::BufferVar<LVector3> positions,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto p = positions->read(i);
                comps->write(i, p.data[0]);
                comps->write(n + i, p.data[1]);
                comps->write(n * 2u + i, p.data[2]);
            };
        });
    return k;
}

/// out[i] = max(|disp[i].x|, |disp[i].y|, |disp[i].z|); the reference folds the
/// flat 3n array with `MaxAbsOp`, which is exactly the max of these per-element
/// maxima (min/max are exact and associative).
auto& axis_max_displacement_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<double>,
                                                uint32_t>(
        [](lcrt::BufferVar<LVector3> disp,
           lcrt::BufferVar<double>   out,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto d = disp->read(i);
                auto a = lcrt::max(lcrt::max(lcrt::abs(d.data[0]), lcrt::abs(d.data[1])),
                                   lcrt::abs(d.data[2]));
                out->write(i, a);
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  Scratch: per-stream temporaries (they must outlive the caller's open frame
//  CommandList, cookbook §7.10 trap 2, and growth must retire the old allocation)
// ---------------------------------------------------------------------------
struct VertexManagerScratch
{
    lud::LDeviceBuffer<double>         scalars;
    std::vector<std::shared_ptr<void>> retired;
};

VertexManagerScratch& scratch_of(lud::LStream& stream)
{
    return lla::detail::default_scratch<VertexManagerScratch>(stream);
}

/// cuda_tool::DeviceBuffer::resize(n, init) parity: grow the buffer, then fill the
/// newly exposed tail with `init` (LDeviceBuffer::resize only value-initialises it).
/// The fill is a device-side dispatch on the frame stream, like cuda_tool's
/// `fill(subview(old, n - old), init, s)`.
template <typename T>
void resize_init(lud::LDeviceBuffer<T>& buffer, size_t n, const T& init)
{
    const size_t old = buffer.size();
    buffer.resize(n);
    if(n > old)
    {
        buffer.view(old, n - old).fill(init, &frame_stream());
    }
}

/// cuda_tool parity: reading a device buffer to the host synchronises. The frame
/// CommandList is committed first so the read observes the recorded work, and
/// re-opened afterwards so the caller keeps batching.
template <typename T>
void read_back(lud::LStream& stream, const lud::LBufferView<T>& src, T* dst, size_t count)
{
    if(count == 0)
    {
        return;
    }
    // I2: a read-back is a full device->host stall (two synchronizes), so it has to be
    // its own band on the timeline instead of an unexplained gap between two `LC.sync`
    // ranges. Scoped, so an early return cannot leave the range open.
    uipc::nvtx::range guard{"LC.readback"};

    const bool was_open = stream.frame_open();
    stream.synchronize();  // commits the recorded work, then waits
    stream.record(src.native().copy_to(lcrt::span{dst, count}));
    stream.synchronize();
    if(was_open)
    {
        stream.begin();
    }
}
}  // namespace

REGISTER_SIM_SYSTEM(GlobalVertexManager);

void GlobalVertexManager::do_build()
{
    auto d_hat = world().scene().config().find<Float>("contact/d_hat");
    m_impl.default_d_hat = d_hat->view()[0];

    auto d_hat_relative = world().scene().config().find<Float>("contact/d_hat_relative");
    m_impl.d_hat_relative = d_hat_relative ? d_hat_relative->view()[0] : 0.0;

    // The reference also resolves three optional cross-system slots here:
    // m_impl.global_trajectory_filter  = find<GlobalTrajectoryFilter>();
    // m_impl.global_active_set_manager = find<GlobalActiveSetManager>();
    // m_impl.global_joint_dof_manager  = find<GlobalJointDofManager>();
    // T7.2a resolved the first one (the manager now exists and its
    // `require_discard_friction()` half is live again -- the reference's `if(auto& ...)`
    // guard keeps it a no-op in a scene without contact). GlobalActiveSetManager is
    // still un-ported (T8.x) and `find<T>()` needs a complete, ISimSystem-derived T, so
    // that slot stays empty; GlobalJointDofManager was resolved by T4.2.
    m_impl.global_trajectory_filter  = find<GlobalTrajectoryFilter>();
    m_impl.global_joint_dof_manager = find<GlobalJointDofManager>();
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
    reporter_vertex_offsets_counts.resize(N);

    span<IndexT> reporter_vertex_counts = reporter_vertex_offsets_counts.counts();

    for(auto&& [i, R] : enumerate(vertex_reporter_view))
    {
        VertexCountInfo info;
        R->report_count(info);
        // get count back
        reporter_vertex_counts[i] = info.m_count;
    }
    reporter_vertex_offsets_counts.scan();
    SizeT total_count = reporter_vertex_offsets_counts.total_count();

    // 3) Initialize buffers for vertex attributes
    coindices.resize(total_count);
    positions.resize(total_count);
    rest_positions.resize(total_count);
    safe_positions.resize(total_count);
    resize_init(contact_element_ids, total_count, 0);
    resize_init(subscene_element_ids, total_count, 0);
    resize_init(thicknesses, total_count, 0.0);
    resize_init(dimensions, total_count, 3);  // default 3D
    resize_init(displacements, total_count, LVector3{});
    resize_init(displacement_norms, total_count, 0.0);
    resize_init(body_ids, total_count, -1);            // -1 means no care about body id
    resize_init(d_hats, total_count, default_d_hat);   // use default d_hat if not specified
    min_max_components.resize(6);                      // packed [min, max] components

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
    //    (the reference's `prev_positions = positions;` device->device assignment)
    prev_positions.resize(total_count);
    prev_positions.view().copy_from(positions.cview(), &frame_stream());
    safe_positions.resize(total_count);
    safe_positions.view().copy_from(positions.cview(), &frame_stream());

    // 6) Other initializations
    axis_max_disp.copy_from(0.0);

    // 7) Scene diagonal of the rest configuration, for scene-relative
    //    parameter adaptation (contact/d_hat_relative etc.)
    {
        auto box       = compute_vertex_bounding_box();
        scene_diagonal = (box.max() - box.min()).norm();
        logger::info("Scene diagonal: {}", scene_diagonal);
    }

    // 8) Propagate the scene-relative d_hat (contact/d_hat_relative, Stiff-GIPC
    //    convention) into the per-vertex d_hats buffer: entries still holding
    //    the absolute contact/d_hat default are replaced, per-geometry meta
    //    d_hat values are untouched. Without this, the relative override only
    //    affected GlobalContactManager's scalar (CFL/logging) while the contact
    //    kernels kept using the absolute default.
    if(d_hat_relative > 0.0)
    {
        Float rel_d_hat = d_hat_relative * scene_diagonal;
        logger::info("Per-vertex d_hat (relative): {} = {} x scene_diagonal({})",
                     rel_d_hat,
                     d_hat_relative,
                     scene_diagonal);
        auto k = &propagate_relative_d_hat_kernel();
        auto n = static_cast<uint32_t>(d_hats.size());
        if(n > 0)
        {
            frame_stream().record(
                (*k)(d_hats.view(), default_d_hat, rel_d_hat, n).dispatch(n));
        }
        default_d_hat = rel_d_hat;
    }
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
    // Not implemented in the reference either; kept verbatim so the two backends
    // stay equivalent (the OffsetCountCollection::scan() path it would use is
    // covered by init() and by the unit test).
    UIPC_ASSERT(false, "Not implemented yet");
}

void GlobalVertexManager::add_reporter(VertexReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_impl.vertex_reporters.register_sim_system(*reporter);
}

void GlobalVertexManager::Impl::step_forward(Float alpha)
{
    auto& stream = frame_stream();
    auto  k      = &step_forward_kernel();
    auto  n      = static_cast<uint32_t>(positions.size());
    if(n > 0)
    {
        stream.record((*k)(positions.view(), safe_positions.view(), displacements.view(), alpha, n)
                          .dispatch(n));
    }
}

void GlobalVertexManager::Impl::collect_vertex_displacements()
{
    for(auto&& [i, R] : enumerate(vertex_reporters.view()))
    {
        VertexDisplacementInfo vd{this, i};
        R->report_displacements(vd);
    }
}

void GlobalVertexManager::Impl::setup_ccd(lud::LCBufferView<LVector3> base_positions)
{
    auto& tmp_pos = safe_positions;
    UIPC_ASSERT(base_positions.size() == positions.size(),
                "Base positions size not equal to vertex count");
    auto& stream = frame_stream();
    auto  k      = &setup_ccd_kernel();
    auto  n      = static_cast<uint32_t>(positions.size());
    if(n > 0)
    {
        stream.record(
            (*k)(positions.view(), tmp_pos.view(), displacements.view(), base_positions, n)
                .dispatch(n));
    }
}

void GlobalVertexManager::Impl::restore_ccd()
{
    positions.view().copy_from(safe_positions.cview(), &frame_stream());
}

void GlobalVertexManager::Impl::overwrite_positions(lud::LCBufferView<LVector3> src)
{
    UIPC_ASSERT(src.size() == positions.size(), "Source size not equal to vertex count");
    positions.view().copy_from(src, &frame_stream());
}

void GlobalVertexManager::VertexAttributeInfo::require_discard_friction() const noexcept
{
    // If the vertex attributes are updated in a way that will ruin the friction computation
    // (e.g. geometry reset, topology change), all friction systems must discard their
    // stale candidates before recording new ones at the start of the next frame.
    // Without discarding, incorrect frictional forces would be applied.
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    //
    // T7.2a restored the GlobalTrajectoryFilter half (the system is ported now);
    // GlobalActiveSetManager is still un-ported (T8.x), so its slot is always empty and
    // the reference's second `if(auto& ...)` body is a no-op. Restore it together with
    // that system.
    if(auto& gtf = m_impl->global_trajectory_filter)
        gtf->require_discard_friction();
}

void GlobalVertexManager::Impl::record_prev_positions()
{
    prev_positions.view().copy_from(positions.cview(), &frame_stream());
}

void GlobalVertexManager::Impl::record_start_point()
{
    safe_positions.view().copy_from(positions.cview(), &frame_stream());
}

Float GlobalVertexManager::Impl::compute_axis_max_displacement()
{
    auto& stream = frame_stream();
    auto  n      = displacements.size();

    // CUB's reduce identity (`0.0`) also covers the n == 0 case.
    axis_max_disp.copy_from(0.0, &stream);

    if(n > 0)
    {
        auto  count   = static_cast<uint32_t>(n);
        auto& scratch = scratch_of(stream);
        auto  maxes   = lla::detail::carve(scratch.scalars, scratch.retired, n);
        auto  k       = &axis_max_displacement_kernel();
        stream.record((*k)(displacements.cview(), maxes, count).dispatch(count));
        lla::device_reduce_max<double>(stream, maxes.subview(0, n), axis_max_disp.view(), n);
    }

    const bool was_open = stream.frame_open();
    stream.synchronize();
    Float result = axis_max_disp.copy_to(&stream);
    if(was_open)
    {
        stream.begin();
    }
    return result;
}

AABB GlobalVertexManager::Impl::compute_vertex_bounding_box()
{
    auto&       stream    = frame_stream();
    const auto  n         = positions.size();
    const Float max_float = std::numeric_limits<Float>::max();

    // CUB's reduce identities, also the n == 0 result.
    std::array<Float, 6> identity{max_float, max_float, max_float,
                                  -max_float, -max_float, -max_float};
    auto                 minmax = min_max_components.view(0, 6);
    minmax.copy_from(span<const Float>{identity.data(), identity.size()}, &stream);

    if(n > 0)
    {
        auto  count   = static_cast<uint32_t>(n);
        auto& scratch = scratch_of(stream);
        auto  comps   = lla::detail::carve(scratch.scalars, scratch.retired, n * 3);

        auto k = &pack_components_kernel();
        stream.record((*k)(comps, positions.cview(), count).dispatch(count));

        for(size_t c = 0; c < 3; ++c)
        {
            lla::device_reduce_min<double>(
                stream, comps.subview(c * n, n), minmax.subview(c, 1), n);
            lla::device_reduce_max<double>(
                stream, comps.subview(c * n, n), minmax.subview(3 + c, 1), n);
        }
    }

    std::array<Float, 6> host{};
    read_back(stream, minmax, host.data(), host.size());

    Vector3 min_pos_host{host[0], host[1], host[2]};
    Vector3 max_pos_host{host[3], host[4], host[5]};

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

    return dump_positions.dump(fmt::format("{}positions.{}", path, frame), positions)  //
           && dump_prev_positions.dump(fmt::format("{}prev_positions.{}", path, frame),
                                       prev_positions);
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
    dump_positions.apply_to(positions);
    dump_prev_positions.apply_to(prev_positions);
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

lud::LBufferView<LVector3> GlobalVertexManager::VertexAttributeInfo::rest_positions() const noexcept
{
    return m_impl->subview(m_impl->rest_positions, m_index);
}

lud::LBufferView<Float> GlobalVertexManager::VertexAttributeInfo::thicknesses() const noexcept
{
    return m_impl->subview(m_impl->thicknesses, m_index);
}

lud::LBufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::coindices() const noexcept
{
    return m_impl->subview(m_impl->coindices, m_index);
}

lud::LBufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::dimensions() const noexcept
{
    return m_impl->subview(m_impl->dimensions, m_index);
}

lud::LBufferView<LVector3> GlobalVertexManager::VertexAttributeInfo::positions() const noexcept
{
    return m_impl->subview(m_impl->positions, m_index);
}

lud::LBufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::contact_element_ids() const noexcept
{
    return m_impl->subview(m_impl->contact_element_ids, m_index);
}

lud::LBufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::subscene_element_ids() const noexcept
{
    return m_impl->subview(m_impl->subscene_element_ids, m_index);
}

lud::LBufferView<IndexT> GlobalVertexManager::VertexAttributeInfo::body_ids() const noexcept
{
    return m_impl->subview(m_impl->body_ids, m_index);
}

lud::LBufferView<Float> GlobalVertexManager::VertexAttributeInfo::d_hats() const noexcept
{
    return m_impl->subview(m_impl->d_hats, m_index);  // Assuming d_hats are stored in thicknesses
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

lud::LBufferView<LVector3> GlobalVertexManager::VertexDisplacementInfo::displacements() const noexcept
{
    return m_impl->subview(m_impl->displacements, m_index);
}

lud::LCBufferView<IndexT> GlobalVertexManager::VertexDisplacementInfo::coindices() const noexcept
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

lud::LCBufferView<IndexT> GlobalVertexManager::coindices() const noexcept
{
    return m_impl.coindices.cview();
}

lud::LCBufferView<IndexT> GlobalVertexManager::body_ids() const noexcept
{
    return m_impl.body_ids.cview();
}

lud::LCBufferView<Float> GlobalVertexManager::d_hats() const noexcept
{
    return m_impl.d_hats.cview();
}

lud::LCBufferView<LVector3> GlobalVertexManager::positions() const noexcept
{
    return m_impl.positions.cview();
}

lud::LCBufferView<LVector3> GlobalVertexManager::prev_positions() const noexcept
{
    return m_impl.prev_positions.cview();
}

lud::LCBufferView<LVector3> GlobalVertexManager::rest_positions() const noexcept
{
    return m_impl.rest_positions.cview();
}

lud::LCBufferView<LVector3> GlobalVertexManager::safe_positions() const noexcept
{
    return m_impl.safe_positions.cview();
}

lud::LCBufferView<IndexT> GlobalVertexManager::contact_element_ids() const noexcept
{
    return m_impl.contact_element_ids.cview();
}

lud::LCBufferView<IndexT> GlobalVertexManager::subscene_element_ids() const noexcept
{
    return m_impl.subscene_element_ids.cview();
}

lud::LCBufferView<LVector3> GlobalVertexManager::displacements() const noexcept
{
    return m_impl.displacements.cview();
}

lud::LCBufferView<Float> GlobalVertexManager::thicknesses() const noexcept
{
    return m_impl.thicknesses.cview();
}

Float GlobalVertexManager::compute_axis_max_displacement()
{
    return m_impl.compute_axis_max_displacement();
}

AABB GlobalVertexManager::compute_vertex_bounding_box()
{
    return m_impl.compute_vertex_bounding_box();
}

Float GlobalVertexManager::scene_diagonal() const noexcept
{
    return m_impl.scene_diagonal;
}

void GlobalVertexManager::step_forward(Float alpha)
{
    m_impl.step_forward(alpha);
}

void GlobalVertexManager::record_start_point()
{
    m_impl.record_start_point();
}

void GlobalVertexManager::setup_ccd(lud::LCBufferView<LVector3> base_positions)
{
    m_impl.setup_ccd(base_positions);
}

void GlobalVertexManager::restore_ccd()
{
    m_impl.restore_ccd();
}

void GlobalVertexManager::overwrite_positions(lud::LCBufferView<LVector3> src)
{
    m_impl.overwrite_positions(src);
}

lud::LCBufferView<IndexT> GlobalVertexManager::dimensions() const noexcept
{
    return m_impl.dimensions.cview();
}

AABB GlobalVertexManager::vertex_bounding_box() const noexcept
{
    return m_impl.vertex_bounding_box;
}
}  // namespace uipc::backend::luisa
