/********************************************************************
 * @file global_dytopo_effect_manager.cpp
 * @brief Global dynamic-topology effect manager (port of
 *        src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.cu).
 *
 * Port notes (cookbook §7.9–§7.13)
 * ------------------------------
 *  * The four `__global__` helpers become LC DSL kernels compiled once and cached
 *    through `linalg::detail::cached_kernel` (deliberately leaked, §7.10 trap 1).
 *    Their doublet/triplet *views* are rebuilt inside the kernel from the raw row /
 *    col / value buffers with `from_handles` — a view is never captured into a
 *    kernel (§7.9 rule 2), it is passed as launch arguments.
 *  * k1's `in_range` lambda is a host-side helper in the reference; on the device
 *    path the comparison chain is written out with `&` (a device predicate is not a
 *    host bool, §7.6 trap 2) and the two "prev/next is out of range" flags are
 *    reconstructed with clamped neighbour indices plus the end-point disjunction,
 *    which is exactly the reference's `if(I > 0)` / `if(I < n - 1)` guards.
 *  * k3's `selected_hessian[N] = 0` write (the reference's
 *    `VarView<IndexT>{selected_hessian.data() + N}`) is done by thread 0, which is
 *    also the reference's behaviour; the dispatch is `max(N, 1)` threads so the
 *    N == 0 case still clears that slot (see the header).
 *  * `std::as_const(sorted_dytopo_effect_gradient).viewer()` is spelled
 *    `cview()`; the two const views k1/k3 read are passed as raw buffers.
 *  * `DeviceScan().ExclusiveSum` is T3.2's `exclusive_scan`; the total is the
 *    last slot of the exclusive-scan output (`selected_hessian_offsets[N]`), which
 *    is exactly the reference's `+ N` read-back.
 *  * Host read-backs (`Vector2i h_range = gradient_range;`, the scan total) are
 *    mid-frame synchronisations: commit, wait, copy, re-open (§7.13 rule 3).
 *  * The manager only exists when the scene enables contact or uses an
 *    InterPrimitive constitution — the SimSystemCreator specialisation below.
 *********************************************************************/
// NOTE: the test target's include path puts src/backends/none first, so the
// bare <sim_engine.h> would resolve to the none backend's shell (cookbook
// §7.14 rule 7). The qualified form is identical in the module build.
#include <backends/luisa/sim_engine.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <uipc/common/enumerate.h>
#include <kernel_cout.h>
#include <uipc/common/unit.h>
#include <uipc/common/timer.h>
#include <uipc/common/zip.h>
#include <energy_component_flags.h>

#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

#include <span>

namespace uipc::backend
{
template <>
class SimSystemCreator<luisa::GlobalDyTopoEffectManager>
{
  public:
    static U<luisa::GlobalDyTopoEffectManager> create(luisa::SimEngine& engine)
    {
        auto dytopo_effect_enable_attr =
            engine.world().scene().config().find<IndexT>("contact/enable");
        bool dytopo_effect_enable = dytopo_effect_enable_attr->view()[0] != 0;

        auto& types = engine.world().scene().constitution_tabular().types();
        bool  has_inter_primitive_constitution =
            types.find(std::string{builtin::InterPrimitive}) != types.end();

        if(dytopo_effect_enable || has_inter_primitive_constitution)
            return uipc::make_unique<luisa::GlobalDyTopoEffectManager>(engine);
        return nullptr;
    }
};
}// namespace uipc::backend

namespace uipc::backend::luisa
{
namespace
{
    namespace lla = luisa_device::linalg;

    using TripletDeviceView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
    using DoubletDeviceView = lla::LCDoubletVectorDeviceView<lcrt::Var<double>, 3>;

    /// k1 — find the contiguous partition of the sorted gradient whose segment
    /// index lies in `[range_x, range_y)`; write {start, end} to `gradient_range`.
    ///
    /// `gradient_range` is the flat {x, y} pair (see the manager header): index 0
    /// is written by the lane that starts the partition, index 1 by the lane that
    /// ends it — the reference's `gradient_range->x() = I` / `->y() = I + 1`.
    auto& distribute_k1_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<double>,
                                                    std::int32_t,
                                                    std::int32_t,
                                                    std::uint32_t>(
            [](lcrt::BufferVar<std::int32_t> gradient_range,
               lcrt::BufferVar<std::int32_t> indices,
               lcrt::BufferVar<double>       values,
               lcrt::Var<std::int32_t>       range_x,
               lcrt::Var<std::int32_t>       range_y,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    auto view = DoubletDeviceView::from_handles(indices, values, n, n);

                    auto idx_prev = lcrt::max(I, 1u) - 1u;  // I == 0 -> 0, value ignored
                    auto idx_next = lcrt::min(I + 1u, n - 1u);

                    auto i      = indices->read(I);
                    auto i_prev = indices->read(idx_prev);
                    auto i_next = indices->read(idx_next);

                    auto this_in = (i >= range_x) & (i < range_y);
                    auto prev_in = (i_prev >= range_x) & (i_prev < range_y);
                    auto next_in = (i_next >= range_x) & (i_next < range_y);

                    // reference: `if(!this_in_range) return;`
                    $if(this_in)
                    {
                        // reference: `prev_in_range = I > 0 ? in_range(prev) : false;`
                        //            `if(!prev_in_range) gradient_range->x() = I;`
                        auto write_x = (I == 0u) | !prev_in;
                        // reference: `next_in_range = I < n-1 ? in_range(next) : false;`
                        //            `if(!next_in_range) gradient_range->y() = I + 1;`
                        auto write_y = (I == n - 1u) | !next_in;

                        $if(write_x) { gradient_range->write(0u, lcrt::cast<std::int32_t>(I)); };
                        $if(write_y)
                        {
                            gradient_range->write(1u, lcrt::cast<std::int32_t>(I + 1u));
                        };
                    };
                };
            });
        return k;
    }

    /// k2 — copy the partition into the receiver's classified gradient.
    auto& distribute_k2_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<double>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<double>,
                                                    std::int32_t,
                                                    std::uint32_t>(
            [](lcrt::BufferVar<std::int32_t> src_indices,
               lcrt::BufferVar<double>       src_values,
               lcrt::BufferVar<std::int32_t> dst_indices,
               lcrt::BufferVar<double>       dst_values,
               lcrt::Var<std::int32_t>       range_x,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    auto src = DoubletDeviceView::from_handles(src_indices, src_values, n, n);
                    auto dst = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>::from_handles(
                        dst_indices, dst_values, n, n);

                    auto f = src(range_x + lcrt::cast<std::int32_t>(I));
                    dst(I).write(f.index, f.value);
                };
            });
        return k;
    }

    /// k3 — mark every hessian entry whose (i, j) lies inside the receiver's
    /// ranges, and clear the exclusive-scan's "last" slot.
    auto& distribute_k3_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<double>,
                                                    std::int32_t,
                                                    std::int32_t,
                                                    std::int32_t,
                                                    std::int32_t,
                                                    std::uint32_t,
                                                    std::uint32_t>(
            [](lcrt::BufferVar<std::int32_t> selected_hessian,
               lcrt::BufferVar<std::int32_t> row_indices,
               lcrt::BufferVar<std::int32_t> col_indices,
               lcrt::BufferVar<double>       values,
               lcrt::Var<std::int32_t>       i_begin,
               lcrt::Var<std::int32_t>       i_end,
               lcrt::Var<std::int32_t>       j_begin,
               lcrt::Var<std::int32_t>       j_end,
               lcrt::Var<std::uint32_t>      n_triplets,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    $if(I < n_triplets)
                    {
                        auto view = TripletDeviceView::from_handles(
                            row_indices, col_indices, values, n_triplets, n_triplets, n_triplets);

                        auto t     = view(I);
                        auto in_i  = (t.row_index >= i_begin) & (t.row_index < i_end);
                        auto in_j  = (t.col_index >= j_begin) & (t.col_index < j_end);
                        auto keep  = in_i & in_j;
                        // select(f, t, p) is p ? t : f (cookbook §7.6 trap 1)
                        selected_hessian->write(I, lcrt::select(0, 1, keep));
                    };

                    // reference: `if(I == 0) last = 0;` with
                    // `last = VarView<IndexT>{selected_hessian.data() + N}`
                    $if(I == 0u) { selected_hessian->write(n_triplets, std::int32_t{0}); };
                };
            });
        return k;
    }

    /// k4 — scatter the selected hessians into the receiver's classified matrix.
    auto& distribute_k4_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<double>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<std::int32_t>,
                                                    lcrt::Buffer<double>,
                                                    std::uint32_t>(
            [](lcrt::BufferVar<std::int32_t> selected_hessian,
               lcrt::BufferVar<std::int32_t> selected_hessian_offsets,
               lcrt::BufferVar<std::int32_t> src_row_indices,
               lcrt::BufferVar<std::int32_t> src_col_indices,
               lcrt::BufferVar<double>       src_values,
               lcrt::BufferVar<std::int32_t> dst_row_indices,
               lcrt::BufferVar<std::int32_t> dst_col_indices,
               lcrt::BufferVar<double>       dst_values,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    $if(selected_hessian->read(I) != 0)
                    {
                        auto src = TripletDeviceView::from_handles(
                            src_row_indices, src_col_indices, src_values, n, n, n);
                        auto dst = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>::from_handles(
                            dst_row_indices, dst_col_indices, dst_values, n, n, n);

                        auto t      = src(I);
                        auto offset = selected_hessian_offsets->read(I);
                        dst(offset).write(t.row_index, t.col_index, t.value);
                    };
                };
            });
        return k;
    }

    /// cuda_tool parity: a device read-back synchronises, and the recorded frame
    /// work must be committed and completed first (cookbook §7.13 rule 3).
    template <typename T>
    void read_back_into(lud::LStream& stream, const lud::LBufferView<T>& one, std::span<T> dst)
    {
        const bool was_open = stream.frame_open();
        stream.synchronize();
        if(!dst.empty())
        {
            stream.record(one.native().copy_to(dst));
            stream.synchronize();
        }
        if(was_open)
        {
            stream.begin();
        }
    }

    template <typename T>
    T read_back(lud::LStream& stream, const lud::LBufferView<T>& one)
    {
        T host{};
        read_back_into(stream, one, std::span<T>{&host, 1u});
        return host;
    }
}// namespace

REGISTER_SIM_SYSTEM(GlobalDyTopoEffectManager);

lla::CBCOOVectorView<Float, 3> GlobalDyTopoEffectManager::gradients() const noexcept
{
    return m_impl.sorted_dytopo_effect_gradient.view();
}

lla::CBCOOMatrixView<Float, 3, 3> GlobalDyTopoEffectManager::hessians() const noexcept
{
    return m_impl.sorted_dytopo_effect_hessian.view();
}

void GlobalDyTopoEffectManager::do_build()
{
    const auto& config = world().scene().config();

    m_impl.global_vertex_manager = require<GlobalVertexManager>();
}

void GlobalDyTopoEffectManager::Impl::init(WorldVisitor& world)
{
    // 3) reporters
    auto dytopo_effect_reporter_view = dytopo_effect_reporters.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_reporter_view))
        R->m_index = i;

    reporter_energy_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_gradient_offsets_counts.resize(dytopo_effect_reporter_view.size());
    reporter_hessian_offsets_counts.resize(dytopo_effect_reporter_view.size());

    // 4) receivers
    auto dytopo_effect_receiver_view = dytopo_effect_receivers.view();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->init();
    for(auto&& [i, R] : enumerate(dytopo_effect_receiver_view))
        R->m_index = i;

    classified_dytopo_effect_gradients.resize(dytopo_effect_receiver_view.size());
    classified_dytopo_effect_hessians.resize(dytopo_effect_receiver_view.size());
}

void GlobalDyTopoEffectManager::Impl::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    _assemble(info);
    auto receivers = dytopo_effect_receivers.view();
    if(receivers.size() == 1)
    {
        DyTopoClassifyInfo classify_info;
        auto*              receiver = receivers[0];
        receiver->report(classify_info);

        auto dynamic_range = classify_info.gradient_i_range();
        auto vertex_count =
            static_cast<IndexT>(global_vertex_manager->positions().size());

        // A single diagonal receiver starting at zero owns the complete dynamic
        // vertex prefix. Any trailing global vertices are non-DOF geometry (for
        // example, half planes), whose contact kernels only differentiate the
        // dynamic vertex. The final GlobalLinearSystem conversion already sorts
        // and reduces these raw entries, so doing that here as well is redundant.
        if(classify_info.is_diag() && dynamic_range.x() == 0 && dynamic_range.y() != 0
           && dynamic_range.y() <= vertex_count && classify_info.hessian_i_range() == dynamic_range
           && classify_info.hessian_j_range() == dynamic_range)
        {
            Timer timer{"Distribute Dytopo Effect"};

            ClassifiedDyTopoEffectInfo classified_info;
            classified_info.m_gradients = collected_dytopo_effect_gradient.cview();
            if(!info.m_gradient_only)
                classified_info.m_hessians = collected_dytopo_effect_hessian.cview();
            receiver->receive(classified_info);
            return;
        }
    }

    _convert_matrix();
    _distribute(info);
}

void GlobalDyTopoEffectManager::Impl::_assemble(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Assemble Dytopo Effect"};

    auto vertex_count = global_vertex_manager->positions().size();

    auto reporter_gradient_counts = reporter_gradient_offsets_counts.counts();
    auto reporter_hessian_counts  = reporter_hessian_offsets_counts.counts();
    bool gradient_only            = info.m_gradient_only;

    logger::info("DyTopo Effect Assembly: GradientOnly={}, ComponentFlags={}",
                 info.m_gradient_only,
                 enum_flags_name(info.m_component_flags));

    {
        Timer timer{"Report Extent"};
        for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
        {
            reporter_gradient_counts[i] = 0;
            reporter_hessian_counts[i]  = 0;

            if(!has_flags(info.m_component_flags, reporter->component_flags()))
                continue;

            GradientHessianExtentInfo extent_info;
            extent_info.m_gradient_only = gradient_only;
            reporter->report_gradient_hessian_extent(extent_info);

            reporter_gradient_counts[i] = extent_info.m_gradient_count;
            reporter_hessian_counts[i]  = gradient_only ? 0 : extent_info.m_hessian_count;
            logger::info("<{}> DyTopo Grad3 count: {}, DyTopo Hess3x3 count: {}",
                         reporter->name(),
                         extent_info.m_gradient_count,
                         extent_info.m_hessian_count);
        }
    }

    {
        Timer timer{"Scan and Allocate"};
        // scan
        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        auto total_gradient_count = reporter_gradient_offsets_counts.total_count();
        auto total_hessian_count  = reporter_hessian_offsets_counts.total_count();

        // allocate
        loose_resize_entries(collected_dytopo_effect_gradient, total_gradient_count);
        loose_resize_entries(sorted_dytopo_effect_gradient, total_gradient_count);
        loose_resize_entries(collected_dytopo_effect_hessian, total_hessian_count);
        loose_resize_entries(sorted_dytopo_effect_hessian, total_hessian_count);
        collected_dytopo_effect_gradient.reshape(vertex_count);
        collected_dytopo_effect_hessian.reshape(vertex_count, vertex_count);
    }

    // collect
    for(auto&& [i, reporter] : enumerate(dytopo_effect_reporters.view()))
    {
        if(!has_flags(info.m_component_flags, reporter->component_flags()))
            continue;

        auto [g_offset, g_count] = reporter_gradient_offsets_counts[i];
        auto [h_offset, h_count] = reporter_hessian_offsets_counts[i];

        GradientHessianInfo info;
        info.m_gradient_only = gradient_only;

        info.m_gradients =
            collected_dytopo_effect_gradient.view().subview(g_offset, g_count);
        info.m_hessians = collected_dytopo_effect_hessian.view().subview(h_offset, h_count);

        reporter->assemble(info);
    }
}

void GlobalDyTopoEffectManager::Impl::_convert_matrix()
{
    Timer timer{"Convert Dytopo Matrix"};

    // The reference converts on the default stream; the port uses the shared frame
    // stream so the conversion is batched into the caller's frame (T3.3).
    auto& stream = frame_stream();
    matrix_converter.convert(collected_dytopo_effect_hessian, sorted_dytopo_effect_hessian, stream);
    matrix_converter.convert(collected_dytopo_effect_gradient, sorted_dytopo_effect_gradient, stream);
}

void GlobalDyTopoEffectManager::Impl::_distribute(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Distribute Dytopo Effect"};

    auto& stream = frame_stream();

    auto vertex_count = global_vertex_manager->positions().size();

    for(auto&& [i, receiver] : enumerate(dytopo_effect_receivers.view()))
    {
        DyTopoClassifyInfo classify_info;
        receiver->report(classify_info);


        ClassifiedDyTopoEffectInfo classified_info;
        auto& classified_gradients = classified_dytopo_effect_gradients[i];
        classified_gradients.reshape(vertex_count);
        auto& classified_hessians = classified_dytopo_effect_hessians[i];
        classified_hessians.reshape(vertex_count, vertex_count);

        // 1) report gradient
        if(classify_info.is_diag())
        {
            const auto N = sorted_dytopo_effect_gradient.doublet_count();

            // clear the range in device
            loose_resize(gradient_range, 2u);
            // `gradient_range = Vector2i{0, 0}`: the reference assigns the whole
            // device scalar; the port keeps the pair in one flat buffer and clears it
            // with a device-side fill of the two elements, which is ordered before the
            // k1 dispatch below and captures no host pointer.
            gradient_range.view().fill(std::int32_t{0}, &stream);

            // partition
            {
                auto k = &distribute_k1_kernel();
                auto n = static_cast<std::uint32_t>(N);
                if(n > 0)
                {
                    auto range = classify_info.gradient_i_range();
                    stream.record((*k)(gradient_range.view(),
                                       std::as_const(sorted_dytopo_effect_gradient).indices(),
                                       std::as_const(sorted_dytopo_effect_gradient).values(),
                                       static_cast<std::int32_t>(range.x()),
                                       static_cast<std::int32_t>(range.y()),
                                       n)
                                      .dispatch(n));
                }
            }

            // Vector2i h_range = gradient_range;  // copy back
            std::int32_t h_range[2] = {0, 0};
            read_back_into(stream, gradient_range.cview(), std::span<std::int32_t>{h_range, 2u});

            auto count = static_cast<std::size_t>(h_range[1] - h_range[0]);

            loose_resize_entries(classified_gradients, count);

            // fill
            if(count > 0)
            {
                auto k = &distribute_k2_kernel();
                auto n = static_cast<std::uint32_t>(count);
                stream.record((*k)(std::as_const(sorted_dytopo_effect_gradient).indices(),
                                   std::as_const(sorted_dytopo_effect_gradient).values(),
                                   classified_gradients.indices(),
                                   classified_gradients.values(),
                                   static_cast<std::int32_t>(h_range[0]),
                                   n)
                                  .dispatch(n));
            }

            classified_info.m_gradients = classified_gradients.view();
        }

        // 2) report hessian
        if(!info.m_gradient_only && !classify_info.is_empty())
        {
            const auto N = sorted_dytopo_effect_hessian.triplet_count();

            // +1 for calculate the total count
            loose_resize(selected_hessian, N + 1);
            loose_resize(selected_hessian_offsets, N + 1);

            // select
            {
                auto k = &distribute_k3_kernel();
                auto n = static_cast<std::uint32_t>(std::max<std::size_t>(N, 1u));
                auto i_range = classify_info.hessian_i_range();
                auto j_range = classify_info.hessian_j_range();
                stream.record((*k)(selected_hessian.view(),
                                   std::as_const(sorted_dytopo_effect_hessian).row_indices(),
                                   std::as_const(sorted_dytopo_effect_hessian).col_indices(),
                                   std::as_const(sorted_dytopo_effect_hessian).values(),
                                   static_cast<std::int32_t>(i_range.x()),
                                   static_cast<std::int32_t>(i_range.y()),
                                   static_cast<std::int32_t>(j_range.x()),
                                   static_cast<std::int32_t>(j_range.y()),
                                   static_cast<std::uint32_t>(N),
                                   n)
                                  .dispatch(n));
            }

            // scan  (DeviceScan().ExclusiveSum over N + 1 elements)
            lla::exclusive_scan<std::int32_t>(stream,
                                              selected_hessian.cview(),
                                              selected_hessian_offsets.view(),
                                              N + 1);

            // VarView<IndexT>{selected_hessian_offsets.data() + N}.copy_to(&h_total_count)
            auto h_total_count = static_cast<std::size_t>(
                read_back(stream, selected_hessian_offsets.cview(N, 1u)));

            loose_resize_entries(classified_hessians, h_total_count);

            // fill
            if(h_total_count > 0)
            {
                auto k = &distribute_k4_kernel();
                auto n = static_cast<std::uint32_t>(N);
                if(n > 0)
                {
                    stream.record(
                        (*k)(selected_hessian.cview(),
                             selected_hessian_offsets.cview(),
                             std::as_const(sorted_dytopo_effect_hessian).row_indices(),
                             std::as_const(sorted_dytopo_effect_hessian).col_indices(),
                             std::as_const(sorted_dytopo_effect_hessian).values(),
                             classified_hessians.row_indices(),
                             classified_hessians.col_indices(),
                             classified_hessians.values(),
                             n)
                            .dispatch(n));
                }
            }

            classified_info.m_hessians = classified_hessians.view();
        }

        receiver->receive(classified_info);
    }
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    lla::LTripletMatrix<Float, 3>& m, SizeT size)
{
    if(size > m.triplet_capacity())
    {
        m.reserve_triplets_discard(size * reserve_ratio);
    }
    m.resize_triplets_discard(size);
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    lla::LDoubletVector<Float, 3>& v, SizeT size)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets_discard(size * reserve_ratio);
    }
    v.resize_doublets_discard(size);
}

void GlobalDyTopoEffectManager::init()
{
    m_impl.init(world());
}

void GlobalDyTopoEffectManager::compute_dytopo_effect(ComputeDyTopoEffectInfo& info)
{
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::compute_dytopo_effect()
{
    ComputeDyTopoEffectInfo info;
    m_impl.compute_dytopo_effect(info);
}

void GlobalDyTopoEffectManager::add_reporter(DyTopoEffectReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");
    auto flag = reporter->component_flags();
    UIPC_ASSERT(is_valid_flag(flag),
                "reporter component_flags() is not valid single flag, it's {}",
                enum_flags_name(flag));
    m_impl.dytopo_effect_reporters.register_sim_system(*reporter);

    // classify into contact / non-contact
    if(reporter->component_flags() == EnergyComponentFlags::Contact)
    {
        m_impl.contact_reporters.register_sim_system(*reporter);
    }
    else
    {
        m_impl.non_contact_reporters.register_sim_system(*reporter);
    }
}

void GlobalDyTopoEffectManager::add_receiver(DyTopoEffectReceiver* receiver)
{
    check_state(SimEngineState::BuildSystems, "add_receiver()");
    UIPC_ASSERT(receiver != nullptr, "receiver is nullptr");
    m_impl.dytopo_effect_receivers.register_sim_system(*receiver);
}
}// namespace uipc::backend::luisa
