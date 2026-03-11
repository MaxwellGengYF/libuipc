#pragma once
#include <line_search/line_search_reporter.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>

namespace uipc::backend::luisa
{
/**
 * @brief Line search reporter for dynamic topology effects
 * 
 * Reports energy values during line search for dynamic topology effects.
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 * 
 * Replaces muda::DeviceBuffer with luisa::compute::Buffer and muda::DeviceVar
 * with single-element buffers.
 */
class DyTopoEffectLineSearchReporter final : public LineSearchReporter
{
  public:
    using LineSearchReporter::LineSearchReporter;

    class Impl;

    class Impl
    {
      public:
        void init();
        void compute_energy(bool is_init);

        SimSystemSlot<GlobalDyTopoEffectManager> global_dytopo_effect_manager;

        // Single-element buffer for scalar energy (replaces muda::DeviceVar<Float>)
        Buffer<Float> energy;
        // Energy buffer for multiple components (replaces muda::DeviceBuffer<Float>)
        Buffer<Float> energies;
        
        Float reserve_ratio = 1.5;

        /**
         * @brief Resize buffer with reserve ratio for amortized growth
         * 
         * Unlike muda::DeviceBuffer, luisa::compute::Buffer doesn't support
         * reserve/capacity. We simply recreate the buffer when size exceeds
         * current capacity.
         */
        template <typename T>
        void loose_resize(Buffer<T>& buffer, SizeT size)
        {
            if(size > buffer.size())
            {
                // Buffer needs to be resized - will be recreated by caller
                // with appropriate size through device.create_buffer<T>(size)
                buffer = Buffer<T>{};
            }
        }
    };

    virtual void do_record_start_point(LineSearcher::RecordInfo& info) override;
    virtual void do_step_forward(LineSearcher::StepInfo& info) override;

  private:
    virtual void do_init(LineSearchReporter::InitInfo& info) override;
    virtual void do_build(LineSearchReporter::BuildInfo& info) override;
    virtual void do_compute_energy(LineSearcher::ComputeEnergyInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
