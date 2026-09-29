#pragma once
/********************************************************************
 * @file fem_line_search_reporter.h
 * @brief T5.3 (pulled forward from T5.5) — the FEM line-search reporter (port of
 * src/backends/cuda/finite_element/fem_line_search_reporter.h).
 *
 * Why it is here: `FiniteElementKinetic` (T5.3's centrepiece) declares
 * `compute_energy(FEMLineSearchReporter::ComputeEnergyInfo&)`, and
 * `FiniteElementAnimator` / `FiniteElementElastics` call through this class, so
 * its *interface* cannot be deferred (the T5.3 brief's §1.2 dependency-cycle
 * policy).  The whole class is small (84 + 156 lines in the reference) and has no
 * T5.4/T5.5 dependency, so it is ported in full -- T5.5 still owns the
 * *diff-reporters* and the active-set/stiffness pieces, nothing here.
 *
 * Port mapping:
 *   cuda_tool::BufferView<Float>  -> lud::LBufferView<Float>
 *   cuda_tool::DeviceBuffer<T>    -> lud::LDeviceBuffer<T>
 *   cuda_tool::DeviceVar<T>       -> lud::LDeviceVar<T>
 *   cuda_tool::DeviceReduce().Sum -> lla::device_reduce_sum on the shared frame
 *                                    stream (T3.2, cookbook §7.15 rule 6)
 *   `fem().x_temps = fem().xs;`   -> an explicit deep device copy (`copy_device`)
 *                                    because an LDeviceBuffer assignment shares
 *                                    the allocation (cookbook §7.16 rule 4b)
 *   `resize_discard(n)`           -> `loose_resize(buffer, n)` with the reference's
 *                                    growth policy, as T4.3's
 *                                    dytopo_effect_line_search_reporter does
 *
 * `Impl::compute_energy` and its two helpers are the only device work here, and
 * the engine only reaches them from the Phase-8 Newton/line-search loop, so this
 * task exercises `init()` only (through `LineSearcher::init()`).
 *********************************************************************/
#include <sim_system.h>
#include <line_search/line_search_reporter.h>
#include <finite_element/finite_element_method.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
class FiniteElementKinetic;
class FEMLineSearchSubreporter;
class FEMLineSearchReporter final : public LineSearchReporter
{
  public:
    using LineSearchReporter::LineSearchReporter;

    class ReportExtentInfo
    {
      public:
        void energy_count(SizeT count) { m_energy_count = count; }

      private:
        friend class FEMLineSearchReporter;
        SizeT m_energy_count = 0;
    };

    class ComputeEnergyInfo
    {
      public:
        ComputeEnergyInfo(lud::LBufferView<Float> energies, Float dt)
            : m_energies(energies)
            , m_dt(dt)
        {
        }

        lud::LBufferView<Float> energies() const { return m_energies; }
        Float                   dt() const noexcept { return m_dt; }

      private:
        friend class FEMLineSearchReporter;
        lud::LBufferView<Float> m_energies;
        Float                   m_dt = 0.0;
    };

    class Impl
    {
      public:
        void init(LineSearchReporter::InitInfo& info);
        void record_start_point(LineSearcher::RecordInfo& info);
        void step_forward(LineSearcher::StepInfo& info);
        void compute_energy(LineSearcher::ComputeEnergyInfo& info);

        SimSystemSlot<FiniteElementMethod> finite_element_method;

        SimSystemSlot<FiniteElementKinetic> finite_element_kinetic;
        lud::LDeviceBuffer<Float>           kinetic_energies;
        lud::LDeviceVar<Float>              total_kinetic_energy;

        SimSystemSlotCollection<FEMLineSearchSubreporter> reporters;
        OffsetCountCollection<IndexT>                     reporter_energy_offsets_counts;
        lud::LDeviceBuffer<Float>                         reporter_energies;
        lud::LDeviceVar<Float>                            total_reporter_energy;

        FiniteElementMethod::Impl& fem()
        {
            return finite_element_method->m_impl;
        }

        /// `cuda_tool::DeviceVector::resize_discard(n)` with the reference's growth
        /// policy (the T4.3 dytopo reporter's helper, same spelling).
        template <typename T>
        void loose_resize(lud::LDeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
        }

        Float reserve_ratio = 1.5;
    };

  protected:
    void do_init(InitInfo& info) override;
    void do_build(LineSearchReporter::BuildInfo& info) override;
    void do_record_start_point(LineSearcher::RecordInfo& info) override;
    void do_step_forward(LineSearcher::StepInfo& info) override;
    void do_compute_energy(LineSearcher::ComputeEnergyInfo& info) override;

  private:
    Impl m_impl;

    friend class FEMLineSearchSubreporter;
    void add_reporter(FEMLineSearchSubreporter* reporter);

    friend class FiniteElementKinetic;
    void add_kinetic(FiniteElementKinetic* kinetic);
};
}  // namespace uipc::backend::luisa
