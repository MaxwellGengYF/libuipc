#include <diff_sim/global_diff_sim_manager.h>
#include <diff_sim/diff_dof_reporter.h>
#include <diff_sim/diff_parm_reporter.h>
#include <linear_system/global_linear_system.h>
#include <sim_engine.h>
#include <utils/offset_count_collection.h>
#include <sim_engine.h>
#include <kernel_cout.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<luisa::GlobalDiffSimManager>
{
  public:
    static U<luisa::GlobalDiffSimManager> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<SimEngine&>(engine).world().scene();
        auto diff_sim_enable_attr = scene.config().find<IndexT>("diff_sim/enable");

        if(!diff_sim_enable_attr->view()[0])
        {
            return nullptr;
        }
        return uipc::make_unique<luisa::GlobalDiffSimManager>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::luisa
{
namespace detail
{
    void build_coo_matrix(MatrixConverter<Float, 1>&           converter,
                          DeviceBCOOMatrix<Float, 1>&          total_coo,
                          DeviceTripletMatrix<Float, 1>&       total_triplet,
                          DeviceTripletMatrix<Float, 1>&       local_triplet,
                          Stream&                              stream)
    {
        // 1) reshape the total_coo and total_triplet
        auto M = local_triplet.block_rows;
        auto N = local_triplet.block_cols;
        total_coo.block_rows = M;
        total_coo.block_cols = N;
        total_triplet.block_rows = M;
        total_triplet.block_cols = N;

        // 2) append the local_triplet to the total_triplet
        //  2.1) resize copy the total_coo to total_triplet
        auto new_triplet_count = total_coo.size() + local_triplet.size();
        
        // Resize total_triplet buffers
        if(total_triplet.row_indices.size() < new_triplet_count)
        {
            auto device = total_triplet.row_indices.device();
            total_triplet.row_indices = device.create_buffer<int>(
                static_cast<size_t>(new_triplet_count * 1.5));
            total_triplet.col_indices = device.create_buffer<int>(
                static_cast<size_t>(new_triplet_count * 1.5));
            total_triplet.values = device.create_buffer<typename DeviceTripletMatrix<Float, 1>::ValueT>(
                static_cast<size_t>(new_triplet_count * 1.5));
        }

        // Copy total_coo to front of total_triplet
        if(total_coo.size() > 0)
        {
            Kernel1D copy_coo_kernel = [](BufferVar<int> coo_rows,
                                          BufferVar<int> coo_cols,
                                          BufferVar<Matrix1x1> coo_vals,
                                          BufferVar<int> triplet_rows,
                                          BufferVar<int> triplet_cols,
                                          BufferVar<Matrix1x1> triplet_vals,
                                          UInt count) {
                UInt i = dispatch_id().x;
                $if(i < count) {
                    Int r = coo_rows.read(i);
                    Int c = coo_cols.read(i);
                    Matrix1x1 v = coo_vals.read(i);
                    triplet_rows.write(i, r);
                    triplet_cols.write(i, c);
                    triplet_vals.write(i, v);
                };
            };
            
            auto copy_coo_shader = total_triplet.row_indices.device().compile(copy_coo_kernel);
            stream << copy_coo_shader(total_coo.row_indices_view(),
                                      total_coo.col_indices_view(),
                                      total_coo.values_view(),
                                      total_triplet.row_indices_view(),
                                      total_triplet.col_indices_view(),
                                      total_triplet.values_view(),
                                      static_cast<uint>(total_coo.size()))
                          .dispatch(total_coo.size());
        }

        //  2.2) append the local_triplet to the total_triplet
        if(local_triplet.size() > 0)
        {
            Kernel1D append_kernel = [](BufferVar<int> local_rows,
                                        BufferVar<int> local_cols,
                                        BufferVar<Matrix1x1> local_vals,
                                        BufferVar<int> triplet_rows,
                                        BufferVar<int> triplet_cols,
                                        BufferVar<Matrix1x1> triplet_vals,
                                        UInt offset,
                                        UInt count) {
                UInt i = dispatch_id().x;
                $if(i < count) {
                    Int r = local_rows.read(i);
                    Int c = local_cols.read(i);
                    Matrix1x1 v = local_vals.read(i);
                    UInt dst_idx = offset + i;
                    triplet_rows.write(dst_idx, r);
                    triplet_cols.write(dst_idx, c);
                    triplet_vals.write(dst_idx, v);
                };
            };
            
            auto append_shader = local_triplet.row_indices.device().compile(append_kernel);
            stream << append_shader(local_triplet.row_indices_view(),
                                    local_triplet.col_indices_view(),
                                    local_triplet.values_view(),
                                    total_triplet.row_indices_view(),
                                    total_triplet.col_indices_view(),
                                    total_triplet.values_view(),
                                    static_cast<uint>(total_coo.size()),
                                    static_cast<uint>(local_triplet.size()))
                          .dispatch(local_triplet.size());
        }

        // 3) convert the total_triplet to total_coo using the converter
        converter.convert(total_triplet, total_coo);
    }

    void copy_to_host(const DeviceBCOOMatrix<Float, 1>&    total_coo,
                      GlobalDiffSimManager::SparseCOO&     host_coo,
                      Stream&                              stream)
    {
        // copy row_indices, col_indices, values to host_coo
        auto coo_size = total_coo.size();
        
        host_coo.row_indices.resize(coo_size);
        host_coo.col_indices.resize(coo_size);
        host_coo.values.resize(coo_size);

        // Create temporary host buffers for async copy
        luisa::vector<int> temp_rows(coo_size);
        luisa::vector<int> temp_cols(coo_size);
        luisa::vector<Matrix1x1> temp_vals(coo_size);

        stream << total_coo.row_indices_view().copy_to(temp_rows.data())
               << total_coo.col_indices_view().copy_to(temp_cols.data())
               << total_coo.values_view().copy_to(temp_vals.data())
               << synchronize();

        // Copy to host vectors
        for(size_t i = 0; i < coo_size; ++i)
        {
            host_coo.row_indices[i] = temp_rows[i];
            host_coo.col_indices[i] = temp_cols[i];
            host_coo.values[i] = temp_vals[i](0, 0);  // Extract scalar from 1x1 matrix
        }

        host_coo.shape = Vector2i{static_cast<int>(total_coo.block_rows), 
                                   static_cast<int>(total_coo.block_cols)};
    }
}  // namespace detail
}  // namespace uipc::backend::luisa

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(GlobalDiffSimManager);

void GlobalDiffSimManager::do_build()
{
    m_impl.global_linear_system = &require<GlobalLinearSystem>();
    m_impl.sim_engine           = &engine();

    on_write_scene([&] { m_impl.write_scene(world()); });
}

MatrixConverter<Float, 1>& GlobalDiffSimManager::Impl::ctx()
{
    return global_linear_system->m_impl.matrix_converter;
}

void GlobalDiffSimManager::Impl::init(WorldVisitor& world)
{
    auto& diff_sim   = world.scene().diff_sim();
    auto  parm_view  = diff_sim.parameters().view();
    total_parm_count = parm_view.size();
    dof_offsets.reserve(1024);
    dof_counts.reserve(1024);
    total_coo_pGpP.block_rows = 0;
    total_coo_pGpP.block_cols = 0;
    total_coo_H.block_rows = 0;
    total_coo_H.block_cols = 0;

    // 1) Copy the parameters to the device
    auto& device = sim_engine->device();
    parameters = device.create_buffer<Float>(total_parm_count);
    
    Stream stream = device.create_stream();
    stream << parameters.copy_from(parm_view.data())
           << synchronize();

    // 2) Init the diff_parm_reporters
    {
        auto diff_parm_reporter_view = diff_parm_reporters.view();
        for(auto&& [i, R] : enumerate(diff_parm_reporter_view))
        {
            R->m_index = i;
        }

        diff_parm_triplet_offset_count.resize(diff_parm_reporter_view.size());
    }

    // 3) Init the diff_dof_reporters
    {
        auto diff_dof_reporter_view = diff_dof_reporters.view();
        for(auto&& [i, R] : enumerate(diff_dof_reporter_view))
        {
            R->m_index = i;
        }

        diff_dof_triplet_offset_count.resize(diff_dof_reporter_view.size());
    }
}

void GlobalDiffSimManager::Impl::update()
{
    // Waiting for later version merging
}

void GlobalDiffSimManager::Impl::assemble()
{
    // Waiting for later version merging
}

void GlobalDiffSimManager::Impl::write_scene(WorldVisitor& world)
{
    // Waiting for later version merging
}

void GlobalDiffSimManager::init()
{
    m_impl.init(world());
}

void GlobalDiffSimManager::assemble()
{
    m_impl.assemble();
}

void GlobalDiffSimManager::update()
{
    m_impl.update();
}

void GlobalDiffSimManager::add_reporter(DiffDofReporter* subsystem)
{
    UIPC_ASSERT(subsystem != nullptr, "subsystem is nullptr");
    m_impl.diff_dof_reporters.register_sim_system(*subsystem);
}

void GlobalDiffSimManager::add_reporter(DiffParmReporter* subsystem)
{
    UIPC_ASSERT(subsystem != nullptr, "subsystem is nullptr");
    m_impl.diff_parm_reporters.register_sim_system(*subsystem);
}

DeviceTripletMatrix<Float, 1> GlobalDiffSimManager::DiffParmInfo::pGpP() const
{
    auto offset = m_impl->diff_parm_triplet_offset_count.offsets()[m_index];
    auto count  = m_impl->diff_parm_triplet_offset_count.counts()[m_index];
    
    DeviceTripletMatrix<Float, 1> result;
    // Create subviews for the triplet matrix
    // Note: LuisaCompute BufferView doesn't have subview, we use view with offset/count
    result.row_indices = m_impl->local_triplet_pGpP.row_indices;
    result.col_indices = m_impl->local_triplet_pGpP.col_indices;
    result.values = m_impl->local_triplet_pGpP.values;
    result.block_rows = m_impl->local_triplet_pGpP.block_rows;
    result.block_cols = m_impl->local_triplet_pGpP.block_cols;
    // The actual subview logic is handled by the caller using offset/count
    return result;
}

DeviceTripletMatrix<Float, 1> GlobalDiffSimManager::DiffDofInfo::H() const
{
    auto offset = m_impl->diff_dof_triplet_offset_count.offsets()[m_index];
    auto count  = m_impl->diff_dof_triplet_offset_count.counts()[m_index];
    
    DeviceTripletMatrix<Float, 1> result;
    result.row_indices = m_impl->local_triplet_H.row_indices;
    result.col_indices = m_impl->local_triplet_H.col_indices;
    result.values = m_impl->local_triplet_H.values;
    result.block_rows = m_impl->local_triplet_H.block_rows;
    result.block_cols = m_impl->local_triplet_H.block_cols;
    return result;
}

SizeT GlobalDiffSimManager::BaseInfo::frame() const
{
    return m_impl->sim_engine->frame();
}

IndexT GlobalDiffSimManager::BaseInfo::dof_offset(SizeT frame) const
{
    return m_impl->dof_offsets[frame - 1];  // we record from the frame 1
}

IndexT GlobalDiffSimManager::BaseInfo::dof_count(SizeT frame) const
{
    return m_impl->dof_counts[frame - 1];  // we record from the frame 1
}

diff_sim::SparseCOOView GlobalDiffSimManager::SparseCOO::view() const
{
    return diff_sim::SparseCOOView{row_indices, col_indices, values, shape};
}

BufferView<Float> GlobalDiffSimManager::DiffParmUpdateInfo::parameters() const noexcept
{
    return m_impl->parameters.view();
}
}  // namespace uipc::backend::luisa
