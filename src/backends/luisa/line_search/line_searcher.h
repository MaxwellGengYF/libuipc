#pragma once
/********************************************************************
 * @file line_searcher.h
 * @brief Global line-search driver (port of
 *        src/backends/cuda/line_search/line_searcher.h).
 *
 * Pure host/manager code: it fans out to the registered LineSearchReporters and
 * reduces their energies. Two port notes:
 *
 *  * `cuda_tool::VarView<Float> energy` (a raw device-pointer view of one scalar)
 *    becomes `lud::LBufferView<Float>` over a one-element subview of
 *    `m_device_energy_values`. A reporter sets it with `energy().fill(v)` or a
 *    one-element copy; the manager reads the whole array back at once.
 *  * `cuda_tool::DeviceReduce().Sum` becomes `linalg::device_reduce_sum` on the
 *    shared frame stream (T3.2). The host read-back of the energy array is a
 *    mid-frame synchronisation (cookbook §7.13 rule 3): commit, wait, copy, and
 *    re-open the frame if the caller had one.
 *
 * `init()/record_start_point()/step_forward()/compute_energy()` are private with
 * `friend class SimEngine`, exactly like the reference: they are driven by the
 * engine's Newton/line-search loop, which is Phase 8. T4.3 wires `init()` into
 * SimEngine::init_scene() (reference order); the other three are only reachable
 * from the Phase-8 loop, so this task does not exercise them.
 *********************************************************************/
#include <sim_system.h>
#include <luisa_device/buffer.h>

#include <sstream>
#include <string>

namespace uipc::backend::luisa
{
class LineSearchReporter;

class LineSearcher : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class RecordInfo
    {
      public:
    };

    class StepInfo
    {
      public:
        Float alpha;
    };

    class ComputeEnergyInfo
    {
      public:
        ComputeEnergyInfo(LineSearcher* impl, lud::LBufferView<Float> energy) noexcept;
        Float                    dt() noexcept;
        lud::LBufferView<Float>  energy() noexcept;
        bool                     is_initial() noexcept;

      private:
        friend class LineSearcher;
        LineSearcher*           m_impl = nullptr;
        lud::LBufferView<Float> m_energy;
        bool                    m_energy_set = false;
        bool                    m_is_initial = false;
    };

    void add_reporter(LineSearchReporter* reporter);

    SizeT max_iter() const noexcept;

  protected:
    void do_build() override;

  private:
    friend class SimEngine;
    void  init();                        // only be called by SimEngine
    void  record_start_point();          // only be called by SimEngine
    void  step_forward(Float alpha);     // only be called by SimEngine
    Float compute_energy(bool is_initial);  // only be called by SimEngine

    SimSystemSlotCollection<LineSearchReporter> m_reporters;

    lud::LDeviceBuffer<Float> m_device_energy_values;
    vector<Float>             m_energy_values;
    bool                      m_report_energy = false;
    std::stringstream         m_report_stream;
    S<const geometry::AttributeSlot<Float>> m_dt_attr;
    IndexT                                  m_max_iter = 64;
};
}// namespace uipc::backend::luisa
