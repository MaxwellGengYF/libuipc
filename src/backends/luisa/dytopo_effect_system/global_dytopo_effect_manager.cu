#include <sim_engine.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <uipc/common/enumerate.h>
#include <kernel_cout.h>
#include <uipc/common/unit.h>
#include <uipc/common/zip.h>
#include <energy_component_flags.h>

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
            return make_unique<luisa::GlobalDyTopoEffectManager>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(GlobalDyTopoEffectManager);

CBCOOVector3 GlobalDyTopoEffectManager::gradients() const noexcept
{
    return CBCOOVector3{
        m_impl.sorted_dytopo_effect_gradient.indices.view(),
        m_impl.sorted_dytopo_effect_gradient.values.view(),
        m_impl.sorted_dytopo_effect_gradient.size()};
}

CBCOOMatrix3 GlobalDyTopoEffectManager::hessians() const noexcept
{
    return CBCOOMatrix3{
        m_impl.sorted_dytopo_effect_hessian.row_indices.view(),
        m_impl.sorted_dytopo_effect_hessian.col_indices.view(),
        m_impl.sorted_dytopo_effect_hessian.values.view(),
        m_impl.sorted_dytopo_effect_hessian.size()};
}

void GlobalDyTopoEffectManager::do_build()
{
    const auto& config = world().scene().config();

    m_impl.global_vertex_manager = require<GlobalVertexManager>();
    m_impl.matrix_converter = MatrixConverter<Float, 3>(engine().device());
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
            reporter_hessian_counts[i] = gradient_only ? 0 : extent_info.m_hessian_count;
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

        GradientHessianInfo grad_hess_info;
        grad_hess_info.m_gradient_only = gradient_only;

        auto g_view = collected_dytopo_effect_gradient.view().subview(g_offset, g_count);
        auto h_view = collected_dytopo_effect_hessian.view().subview(h_offset, h_count);
        
        grad_hess_info.m_gradients = MutableDoubletVector3{g_view.indices, g_view.values, g_view.count, g_view.count};
        grad_hess_info.m_hessians = MutableTripletMatrix3{h_view.row_indices, h_view.col_indices, h_view.values, h_view.count, h_view.count};

        reporter->assemble(grad_hess_info);
    }
}

void GlobalDyTopoEffectManager::Impl::_convert_matrix()
{
    Timer timer{"Convert Dytopo Matrix"};

    matrix_converter.convert(collected_dytopo_effect_hessian, sorted_dytopo_effect_hessian);
    matrix_converter.convert(collected_dytopo_effect_gradient, sorted_dytopo_effect_gradient);
}

void GlobalDyTopoEffectManager::Impl::_distribute(ComputeDyTopoEffectInfo& info)
{
    Timer timer{"Distribute Dytopo Effect"};

    auto& device = engine().device();
    auto& stream = engine().stream();

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
            Vector2i h_range_init{0, 0};
            stream << gradient_range.view().copy_from(&h_range_init) << synchronize();

            // partition kernel - find gradient range
            if(N > 0)
            {
                Kernel1D partition_kernel = [](BufferVar<Vector2i> gradient_range,
                                               BufferVar<int> dytopo_indices,
                                               BufferVar<Eigen::Matrix<Float, 3, 1>> dytopo_values,
                                               UInt N,
                                               Vector2i range) {
                    UInt I = dispatch_id().x;
                    $if(I < N) {
                        auto in_range = [](Int i, Vector2i r) -> Bool {
                            return i >= r.x && i < r.y;
                        };

                        Int idx = dytopo_indices.read(I);
                        Bool this_in_range = in_range(idx, range);

                        $if(this_in_range) {
                            Bool prev_in_range = false;
                            $if(I > 0u) {
                                Int prev_idx = dytopo_indices.read(I - 1u);
                                prev_in_range = in_range(prev_idx, range);
                            };

                            Bool next_in_range = false;
                            $if(I < N - 1u) {
                                Int next_idx = dytopo_indices.read(I + 1u);
                                next_in_range = in_range(next_idx, range);
                            };

                            // if the prev is not in range, then this is the start of the partition
                            $if(!prev_in_range) {
                                Vector2i gr = gradient_range.read(0u);
                                gr.x = static_cast<Int>(I);
                                gradient_range.write(0u, gr);
                            };

                            // if the next is not in range, then this is the end of the partition
                            $if(!next_in_range) {
                                Vector2i gr = gradient_range.read(0u);
                                gr.y = static_cast<Int>(I) + 1;
                                gradient_range.write(0u, gr);
                            };
                        };
                    };
                };

                auto partition_shader = device.compile(partition_kernel);
                stream << partition_shader(gradient_range.view(),
                                           sorted_dytopo_effect_gradient.indices.view(),
                                           sorted_dytopo_effect_gradient.values.view(),
                                           static_cast<uint>(N),
                                           classify_info.gradient_i_range())
                              .dispatch(N);
            }

            // Copy back the range
            luisa::vector<Vector2i> h_range_vec(1);
            stream << gradient_range.view().copy_to(h_range_vec.data()) << synchronize();
            Vector2i h_range = h_range_vec[0];

            auto count = h_range.y() - h_range.x();

            loose_resize_entries(classified_gradients, count);

            // fill
            if(count > 0)
            {
                Kernel1D fill_kernel = [](BufferVar<int> src_indices,
                                          BufferVar<Eigen::Matrix<Float, 3, 1>> src_values,
                                          BufferVar<int> dst_indices,
                                          BufferVar<Eigen::Matrix<Float, 3, 1>> dst_values,
                                          Int range_x,
                                          UInt count) {
                    UInt I = dispatch_id().x;
                    $if(I < count) {
                        UInt src_idx = static_cast<UInt>(range_x) + I;
                        int idx = src_indices.read(src_idx);
                        Eigen::Matrix<Float, 3, 1> val = src_values.read(src_idx);
                        dst_indices.write(I, idx);
                        dst_values.write(I, val);
                    };
                };

                auto fill_shader = device.compile(fill_kernel);
                stream << fill_shader(sorted_dytopo_effect_gradient.indices.view(),
                                      sorted_dytopo_effect_gradient.values.view(),
                                      classified_gradients.indices.view(),
                                      classified_gradients.values.view(),
                                      h_range.x(),
                                      static_cast<uint>(count))
                              .dispatch(count);
            }

            auto cg_view = classified_gradients.view();
            classified_info.m_gradients = DoubletVector3{cg_view.indices, cg_view.values, cg_view.block_count, cg_view.block_count};
        }

        // 2) report hessian
        if(!info.m_gradient_only && !classify_info.is_empty())
        {
            const auto N = sorted_dytopo_effect_hessian.triplet_count();

            // +1 for calculate the total count
            loose_resize(selected_hessian, N + 1);
            loose_resize(selected_hessian_offsets, N + 1);

            // select kernel
            if(N > 0)
            {
                Kernel1D select_kernel = [](BufferVar<IndexT> selected_hessian,
                                            BufferVar<IndexT> last,
                                            BufferVar<int> dytopo_row_indices,
                                            BufferVar<int> dytopo_col_indices,
                                            BufferVar<Eigen::Matrix<Float, 3, 3>> dytopo_values,
                                            UInt N,
                                            Vector2i i_range,
                                            Vector2i j_range) {
                    UInt I = dispatch_id().x;
                    $if(I < N) {
                        auto in_range = [](Int i, Vector2i r) -> Bool {
                            return i >= r.x && i < r.y;
                        };

                        Int row_idx = dytopo_row_indices.read(I);
                        Int col_idx = dytopo_col_indices.read(I);

                        Bool in_i = in_range(row_idx, i_range);
                        Bool in_j = in_range(col_idx, j_range);

                        selected_hessian.write(I, in_i && in_j ? 1 : 0);
                    };

                    // fill the last one as 0, so that we can calculate the total count
                    // during the exclusive scan
                    $if(I == 0u) {
                        last.write(0u, 0);
                    };
                };

                auto select_shader = device.compile(select_kernel);
                stream << select_shader(selected_hessian.view(0, N),
                                        selected_hessian.view(N, 1),
                                        sorted_dytopo_effect_hessian.row_indices.view(),
                                        sorted_dytopo_effect_hessian.col_indices.view(),
                                        sorted_dytopo_effect_hessian.values.view(),
                                        static_cast<uint>(N),
                                        classify_info.hessian_i_range(),
                                        classify_info.hessian_j_range())
                              .dispatch(N);
            }
            else
            {
                // Still need to set the last element to 0
                IndexT zero = 0;
                stream << selected_hessian.view(N, 1).copy_from(&zero) << synchronize();
            }

            // scan
            // Use DeviceScan from LuisaCompute
            luisa::compute::DeviceScan::exclusive_sum(
                selected_hessian.view(0, N + 1),
                selected_hessian_offsets.view(0, N + 1));

            // Get total count
            luisa::vector<IndexT> h_total_count_vec(1);
            stream << selected_hessian_offsets.view(N, 1).copy_to(h_total_count_vec.data())
                   << synchronize();
            IndexT h_total_count = h_total_count_vec[0];

            loose_resize_entries(classified_hessians, h_total_count);

            // fill
            if(h_total_count > 0 && N > 0)
            {
                Kernel1D fill_hessian_kernel = [](BufferVar<IndexT> selected_hessian,
                                                   BufferVar<IndexT> selected_hessian_offsets,
                                                   BufferVar<int> dytopo_row_indices,
                                                   BufferVar<int> dytopo_col_indices,
                                                   BufferVar<Eigen::Matrix<Float, 3, 3>> dytopo_values,
                                                   BufferVar<int> classified_row_indices,
                                                   BufferVar<int> classified_col_indices,
                                                   BufferVar<Eigen::Matrix<Float, 3, 3>> classified_values,
                                                   UInt N) {
                    UInt I = dispatch_id().x;
                    $if(I < N) {
                        IndexT is_selected = selected_hessian.read(I);
                        $if(is_selected != 0) {
                            IndexT offset = selected_hessian_offsets.read(I);
                            int row_idx = dytopo_row_indices.read(I);
                            int col_idx = dytopo_col_indices.read(I);
                            Eigen::Matrix<Float, 3, 3> val = dytopo_values.read(I);
                            classified_row_indices.write(static_cast<UInt>(offset), row_idx);
                            classified_col_indices.write(static_cast<UInt>(offset), col_idx);
                            classified_values.write(static_cast<UInt>(offset), val);
                        };
                    };
                };

                auto fill_hessian_shader = device.compile(fill_hessian_kernel);
                stream << fill_hessian_shader(selected_hessian.view(0, N),
                                              selected_hessian_offsets.view(0, N),
                                              sorted_dytopo_effect_hessian.row_indices.view(),
                                              sorted_dytopo_effect_hessian.col_indices.view(),
                                              sorted_dytopo_effect_hessian.values.view(),
                                              classified_hessians.row_indices.view(),
                                              classified_hessians.col_indices.view(),
                                              classified_hessians.values.view(),
                                              static_cast<uint>(N))
                              .dispatch(N);
            }

            auto ch_view = classified_hessians.view();
            classified_info.m_hessians = TripletMatrix3{ch_view.row_indices, ch_view.col_indices, ch_view.values, ch_view.block_count, ch_view.block_count};
        }

        receiver->receive(classified_info);
    }
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    DeviceTripletMatrix<Float, 3>& m, SizeT size)
{
    auto& device = engine().device();
    if(size > m.values.size())
    {
        auto new_capacity = static_cast<size_t>(size * reserve_ratio);
        m.row_indices = device.create_buffer<int>(new_capacity);
        m.col_indices = device.create_buffer<int>(new_capacity);
        m.values = device.create_buffer<Eigen::Matrix<Float, 3, 3>>(new_capacity);
    }
    // Note: We don't actually resize the buffers, we just track the logical size
    // The matrix_converter will use the size() method to know how many elements are valid
}

void GlobalDyTopoEffectManager::Impl::loose_resize_entries(
    DeviceDoubletVector<Float, 3>& v, SizeT size)
{
    auto& device = engine().device();
    if(size > v.values.size())
    {
        auto new_capacity = static_cast<size_t>(size * reserve_ratio);
        v.indices = device.create_buffer<int>(new_capacity);
        v.values = device.create_buffer<Eigen::Matrix<Float, 3, 1>>(new_capacity);
    }
}
}  // namespace uipc::backend::luisa


namespace uipc::backend::luisa
{
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
}  // namespace uipc::backend::luisa
