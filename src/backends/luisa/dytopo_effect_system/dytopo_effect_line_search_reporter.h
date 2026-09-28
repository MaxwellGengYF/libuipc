#pragma once
/********************************************************************
 * @file dytopo_effect_line_search_reporter.h
 * @brief The line-search side of the dytopo effects (port of
 *        src/backends/cuda/dytopo_effect_system/dytopo_effect_line_search_reporter.h).
 *
 * It forwards the dytopo reporters' energies into the LineSearcher's energy
 * reduction. `do_record_start_point`/`do_step_forward` are deliberately empty in
 * the reference (GlobalVertexManager owns the vertex state); kept verbatim.
 *********************************************************************/
#include <line_search/line_search_reporter.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>

namespace uipc::backend::luisa
{
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

        lud::LDeviceBuffer<Float> energies;
        Float                     reserve_ratio = 1.5;

        template <typename T>
        void loose_resize(lud::LDeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
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
}// namespace uipc::backend::luisa
