#include <dytopo_effect_system/dytopo_effect_line_search_reporter.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(DyTopoEffectLineSearchReporter);

void DyTopoEffectLineSearchReporter::do_build(LineSearchReporter::BuildInfo& info)
{
    m_impl.global_dytopo_effect_manager = require<GlobalDyTopoEffectManager>();
}

void DyTopoEffectLineSearchReporter::do_init(LineSearchReporter::InitInfo& info)
{
    m_impl.init();
}

void DyTopoEffectLineSearchReporter::Impl::init() 
{
    // Initialize the single-element energy buffer
    auto& device = global_dytopo_effect_manager->engine().device();
    energy = device.create_buffer<Float>(1);
}

void DyTopoEffectLineSearchReporter::Impl::compute_energy(bool is_init)
{
    auto& manager   = global_dytopo_effect_manager->m_impl;
    auto  reporters = manager.dytopo_effect_reporters.view();

    auto energy_counts = manager.reporter_energy_offsets_counts.counts();
    for(auto&& [i, reporter] : enumerate(reporters))
    {
        GlobalDyTopoEffectManager::EnergyExtentInfo extent_info;
        reporter->report_energy_extent(extent_info);
        energy_counts[i] = extent_info.m_energy_count;
    }

    manager.reporter_energy_offsets_counts.scan();
    
    // Resize energies buffer if needed
    auto total_count = manager.reporter_energy_offsets_counts.total_count();
    if(energies.size() < total_count)
    {
        auto& device = global_dytopo_effect_manager->engine().device();
        energies = device.create_buffer<Float>(static_cast<size_t>(total_count * reserve_ratio));
    }

    for(auto&& [i, reporter] : enumerate(reporters))
    {
        GlobalDyTopoEffectManager::EnergyInfo this_info;
        auto [offset, count]   = manager.reporter_energy_offsets_counts[i];
        this_info.m_energies   = energies.view(offset, count);
        this_info.m_is_initial = is_init;
        reporter->compute_energy(this_info);
    }
}

void DyTopoEffectLineSearchReporter::do_record_start_point(LineSearcher::RecordInfo& info)
{
    // Do nothing, because GlobalVertexManager will do the record start point for all the vertices we need
}

void DyTopoEffectLineSearchReporter::do_step_forward(LineSearcher::StepInfo& info)
{
    // Do nothing, because GlobalVertexManager will do the step forward for all the vertices we need
}

void DyTopoEffectLineSearchReporter::do_compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    using namespace luisa::compute;

    m_impl.compute_energy(info.is_initial());

    // Perform reduction sum using LuisaCompute
    // We need to sum all elements in m_impl.energies and store in m_impl.energy
    auto& device = m_impl.global_dytopo_effect_manager->engine().device();
    Stream stream = device.create_stream();
    
    auto energies_view = m_impl.energies.view();
    auto energy_view = m_impl.energy.view();
    auto n = m_impl.energies.size();
    
    if(n == 0)
    {
        // No energies to sum, set to zero
        Float zero = 0.0f;
        stream << energy_view.copy_from(&zero) << synchronize();
    }
    else if(n == 1)
    {
        // Single element, just copy
        stream << energy_view.copy_from(energies_view) << synchronize();
    }
    else
    {
        // Multi-stage reduction
        // First stage: reduce in blocks, store partial sums
        // Second stage: reduce partial sums to single value
        
        const uint block_size = 256;
        uint num_blocks = (static_cast<uint>(n) + block_size - 1) / block_size;
        
        // Buffer for partial sums
        Buffer<Float> partial_sums = device.create_buffer<Float>(num_blocks);
        
        // Kernel 1: Block-level reduction
        Kernel1D block_reduce_kernel = [](BufferVar<Float> input, 
                                          BufferVar<Float> output,
                                          UInt n,
                                          UInt block_size) {
            UInt block_idx = block_id().x;
            UInt tid = thread_id().x;
            UInt idx = block_idx * block_size + tid;
            
            // Use warp-level reduction
            Float sum = 0.0f;
            $if(idx < n) {
                sum = input.read(idx);
            };
            
            // Warp reduction using shuffle
            $for(warp_offset, 16u, 0u, -1u) {
                sum = sum + warp_shuffle_down(sum, warp_offset);
            };
            
            // First thread in warp writes to shared/output
            $if(tid % 32u == 0u) {
                UInt warp_id = tid / 32u;
                // Use atomic add to accumulate warp sums
                // Since we only have one output per block, use a simple approach
                output.atomic(block_idx).fetch_add(sum);
            };
        };
        
        // Clear partial sums first
        Kernel1D clear_kernel = [](BufferVar<Float> buf, UInt size) {
            UInt i = dispatch_id().x;
            $if(i < size) {
                buf.write(i, 0.0f);
            };
        };
        
        auto clear_shader = device.compile(clear_kernel);
        stream << clear_shader(partial_sums.view(), num_blocks).dispatch(num_blocks)
               << synchronize();
        
        auto block_reduce_shader = device.compile(block_reduce_kernel);
        stream << block_reduce_shader(energies_view, partial_sums.view(), 
                                      static_cast<uint>(n), block_size).dispatch(num_blocks, block_size)
               << synchronize();
        
        // Kernel 2: Final reduction of partial sums
        if(num_blocks == 1)
        {
            // Just copy the single partial sum
            stream << energy_view.copy_from(partial_sums.view(0, 1)) << synchronize();
        }
        else
        {
            // Reduce partial sums
            Kernel1D final_reduce_kernel = [](BufferVar<Float> input,
                                              BufferVar<Float> output,
                                              UInt n) {
                Float sum = 0.0f;
                $for(i, n) {
                    sum = sum + input.read(i);
                };
                output.write(0, sum);
            };
            
            auto final_reduce_shader = device.compile(final_reduce_kernel);
            stream << final_reduce_shader(partial_sums.view(), energy_view, num_blocks).dispatch(1)
                   << synchronize();
        }
    }

    info.energy(m_impl.energy);
}
}  // namespace uipc::backend::luisa
