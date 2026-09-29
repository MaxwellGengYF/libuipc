#pragma once
/********************************************************************
 * @file affine_body/abd_line_search_reporter.h
 * @brief SCOPE NOTE (T6.4) — the ABD line-search reporter's *declaration surface*
 * (port of src/backends/cuda/affine_body/abd_line_search_reporter.h).
 *
 * WHY THIS FILE EXISTS IN T6.1a
 * -----------------------------
 * `AffineBodyConstitution` and `AffineBodyKinetic` (both in scope here) declare
 * `compute_energy(ABDLineSearchReporter::ComputeEnergyInfo&)`, so the nested type
 * has to exist (T5.3's §7.18 rule 3 policy; the FEM analogue is
 * fem_line_search_reporter.h, which T5.3 ported in full because it had no later
 * dependency — this one *does*, see below).
 *
 * WHAT IS PORTED HERE: the class declaration (a `LineSearchReporter`), both nested
 * info classes with the reference's access surface, `Impl`'s member list, the
 * `friend` set and the private hook declarations.
 *
 * WHAT IS **NOT** DEFINED HERE (all T6.4's, which owns the ABD line search, the
 * tolerance checker and the AL pipeline):
 *  * `Impl::{init, record_start_point, step_forward, compute_energy}` (the
 *    reference's kernels + `kinetic->compute_energy` / `cst->compute_energy`
 *    fan-outs — they need the T6.2 kinetic and the T6.3 constitutions),
 *  * the five `do_*` hooks and `REGISTER_SIM_SYSTEM`.
 * Nothing is stubbed: the hooks are declared, so an unreachable definition is a
 * link error (cookbook §7.16 rule 13).
 *
 * PULLED FORWARD BY T6.1b: `add_reporter` — its only caller is
 * `ABDLineSearchSubreporter::do_build()`, which T6.1b had to define (the animator's
 * line-search subreporter needs the base's vtable).  It lives in the new
 * abd_line_search_reporter.cpp with a SCOPE NOTE naming the rest as T6.4's.
 *********************************************************************/
#include <sim_system.h>
#include <line_search/line_search_reporter.h>
#include <affine_body/affine_body_dynamics.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
class ABDLineSearchSubreporter;  // T6.4
class ABDLineSearchReporter final : public LineSearchReporter
{
  public:
    using LineSearchReporter::LineSearchReporter;

    class ReportExtentInfo
    {
      public:
        void energy_count(SizeT count) { m_energy_count = count; }

      private:
        friend class ABDLineSearchReporter;
        SizeT m_energy_count = 0;
    };

    class ComputeEnergyInfo
    {
      public:
        lud::LBufferView<Float> energies() const { return m_energies; }
        Float                   dt() const noexcept { return m_dt; }

      private:
        friend class ABDLineSearchReporter;
        lud::LBufferView<Float> m_energies;
        Float                   m_dt = 0.0;
    };

    /// SCOPE NOTE (T6.4): the member list is the reference's; every method body and
    /// the registration are T6.4's.
    class Impl
    {
      public:
        void init(LineSearchReporter::InitInfo& info);
        void record_start_point(LineSearcher::RecordInfo& info);
        void step_forward(LineSearcher::StepInfo& info);
        void compute_energy(LineSearcher::ComputeEnergyInfo& info);

        SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

        // intermediate energy buffers
        lud::LDeviceBuffer<Float> body_id_to_kinetic_energy;
        lud::LDeviceVar<Float>    abd_kinetic_energy;
        lud::LDeviceBuffer<Float> body_id_to_shape_energy;
        lud::LDeviceVar<Float>    abd_shape_energy;

        SimSystemSlotCollection<ABDLineSearchSubreporter> reporters;
        OffsetCountCollection<IndexT>                     reporter_energy_offsets_counts;
        lud::LDeviceBuffer<Float>                         reporter_energies;
        lud::LDeviceVar<Float>                            total_reporter_energy;

        AffineBodyDynamics::Impl& abd() const
        {
            return affine_body_dynamics->m_impl;
        }
    };

  private:
    virtual void do_build(LineSearchReporter::BuildInfo& info) override;
    virtual void do_init(LineSearchReporter::InitInfo& info) override;
    virtual void do_record_start_point(LineSearcher::RecordInfo& info) override;
    virtual void do_step_forward(LineSearcher::StepInfo& info) override;
    virtual void do_compute_energy(LineSearcher::ComputeEnergyInfo& info) override;

    friend class ABDLineSearchSubreporter;
    void add_reporter(ABDLineSearchSubreporter* reporter);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
