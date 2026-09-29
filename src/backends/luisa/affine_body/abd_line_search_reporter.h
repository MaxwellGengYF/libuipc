#pragma once
/********************************************************************
 * @file affine_body/abd_line_search_reporter.h
 * @brief T6.4a (SCOPE NOTE CLOSED) — the ABD line-search reporter
 *        (port of src/backends/cuda/affine_body/abd_line_search_reporter.h).
 *
 * HISTORY (why this file existed before it was complete)
 * ------------------------------------------------------
 * T6.1a ported the *declaration surface* only, because `AffineBodyConstitution`
 * and `AffineBodyKinetic` declare `compute_energy(ABDLineSearchReporter::ComputeEnergyInfo&)`
 * and a nested type needs its enclosing class complete (T5.3's §7.18 rule 3
 * policy).  T6.1b then pulled two bodies forward: `add_reporter` (its only caller,
 * `ABDLineSearchSubreporter::do_build()`, is compiled) and the subreporter's four
 * forwarding hooks (a *vtable* needs them for the animator's and the inter-body
 * manager's subreporters).  T6.4a defines the rest -- `Impl::{init,
 * record_start_point, step_forward, compute_energy}` (the reference's kernels + the
 * kinetic/constitution/reporter energy fan-out in abd_line_search_reporter.cpp),
 * the five `do_*` hooks and `REGISTER_SIM_SYSTEM` -- and the SCOPE NOTEs are gone
 * with it.
 *
 * WHAT IT IS
 * ----------
 * A `LineSearchReporter` the `LineSearcher` drives: `record_start_point` snapshots
 * every body's `q` (`q_temp`), `step_forward(alpha)` advances `q = q_temp + alpha *
 * dq` for every *non-fixed* body, and `compute_energy` combines three sums --
 * kinetic (the special-cased `AffineBodyKinetic`, with fixed / external-kinetic
 * bodies zeroed), shape (each `AffineBodyConstitution`'s slice) and the
 * subreporters' (each `ABDLineSearchSubreporter`'s slice) -- into the line
 * searcher's one-element energy buffer.
 *
 * This class is the `friend` of `AffineBodyDynamics`, `AffineBodyKinetic`,
 * `AffineBodyConstitution` and `ABDLineSearchSubreporter`, i.e. the fan-out above
 * is exactly the access the reference relies on.
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

    /// T6.4a: the member list is the reference's; every method body is in
    /// abd_line_search_reporter.cpp.
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

    void do_build(LineSearchReporter::BuildInfo& info) override;
    void do_init(LineSearchReporter::InitInfo& info) override;
    void do_record_start_point(LineSearcher::RecordInfo& info) override;
    void do_step_forward(LineSearcher::StepInfo& info) override;
    void do_compute_energy(LineSearcher::ComputeEnergyInfo& info) override;

    /// TEST SEAM (T6.4a, reported): the reference declares the five hooks in a
    /// `private:` section (its own `.cu` is the only reader); the port makes them
    /// equivalent to the FEM sibling's `protected:` hooks so the unit test can
    /// drive the *same* calls `LineSearcher` makes (`record_start_point`,
    /// `step_forward` and `compute_energy` are private on the `LineSearchReporter`
    /// base and are only reachable through `LineSearcher`, which no test can drive
    /// before Phase 8).  The class is `final`, so nothing else can override them.
    friend class ABDLineSearchSubreporter;
    void add_reporter(ABDLineSearchSubreporter* reporter);

  private:
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
