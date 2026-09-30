#pragma once
/********************************************************************
 * @file global_trajectory_filter.h
 * @brief The global trajectory-filter manager (port of
 *        src/backends/cuda/collision_detection/global_trajectory_filter.h, 115 lines).
 *
 * WHAT IT IS
 * ----------
 * The registry and driver of every `TrajectoryFilter` in the scene.  It owns the
 * per-filter `toi` device slots and the frame-level pipeline stages:
 *
 *   detect(alpha) -> filter_active() -> filter_toi(alpha) -> (min-reduce to a step
 *   fraction) and, when friction is enabled, record_friction_candidates();
 *   GlobalContactManager additionally drives label_active_vertices() (the CFL active
 *   set) and clear_friction_candidates() (the "discard, the set is stale" path,
 *   reference issue #303).
 *
 * All six entry points are private in the reference with an explicit friend list
 * (`SimEngine`, `ContactExporterManager`, `GlobalContactManager`) because the Newton
 * loop is their only caller.  The port keeps that shape *and* its friend list (the two
 * un-ported friends are declared, never defined -- they are T8.x).
 *
 * THE TEST SEAM (`run_stage`)
 * --------------------------
 * The port's engine shell has no Newton loop yet (the reference's `advance_ipc.cu` /
 * `advance_al.cu` are Phase 8 / T8.4; the port's `SimEngine::advance()` is the frame
 * prologue/epilogue), and the other two friends do not exist before T8.1.  Without a
 * way in, none of the manager's stage loops could be driven at all -- the [filter]
 * test would be limited to the filters' own `Impl`s.  `run_stage(Stage, alpha)`
 * therefore forwards to the six reference bodies unchanged; it is additive (it adds no
 * behaviour and no state) and is removed together with the seam when T8.4 wires the
 * real pipeline.  Every other member here is the reference's.
 *
 * WIRING (where the reference calls these)
 * ---------------------------------------
 *   SimEngine::build()                -> `find<GlobalTrajectoryFilter>()`
 *   SimEngine::advance_ipc/al         -> detect / filter_active / filter_toi /
 *                                        record_friction_candidates (T8.4)
 * ContactExporterManager -> detect / filter_active.  T8.1a ported the manager and
 * found that **its own body never calls them** (the reference's friend declaration is
 * for the *concrete* exporters T8.1b ports, which run detect/filter_active to build
 * their per-primitive-type energy/gradient/Hessian readbacks).
 *   GlobalContactManager              -> label_active_vertices /
 *                                        clear_friction_candidates (T8.1)
 *   the filters' `do_build`           -> add_filter(this)
 *********************************************************************/
#include <sim_system.h>
#include <type_define.h>

namespace uipc::backend::luisa
{
class TrajectoryFilter;
class GlobalContactManager;
class ContactExporterManager;

class GlobalTrajectoryFilter final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class FilterTOIInfo
    {
      public:
        Float                     alpha() const noexcept { return m_alpha; }
        lud::LBufferView<Float>   toi() const noexcept { return m_toi; }

      private:
        friend class GlobalTrajectoryFilter;
        Float                   m_alpha = 0.0;
        lud::LBufferView<Float> m_toi;
    };

    class DetectInfo
    {
      public:
        Float alpha() const noexcept { return m_alpha; }

      private:
        friend class GlobalTrajectoryFilter;
        Float m_alpha = 0.0;
    };

    class FilterActiveInfo
    {
      public:
        FilterActiveInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }


      private:
        friend class GlobalTrajectoryFilter;
        Impl* m_impl;
    };

    class LabelActiveVerticesInfo
    {
      public:
        LabelActiveVerticesInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }
        lud::LBufferView<IndexT> vert_is_active() const noexcept;

      private:
        friend class GlobalTrajectoryFilter;
        Impl* m_impl;
    };

    class RecordFrictionCandidatesInfo
    {
      public:
    };

    class Impl
    {
      public:
        void  init();
        Float filter_toi(Float alpha);

        SimSystemSlotCollection<TrajectoryFilter> filters;
        SimSystemSlot<GlobalContactManager>       global_contact_manager;
        bool                                      friction_enabled = false;
        bool should_discard_friction_candidates                    = false;


        lud::LDeviceBuffer<Float> tois;
        vector<Float>             h_tois;
    };

    template <std::derived_from<SimSystem> T>
    SimSystemSlot<T> find()
    {
        return m_impl.filters.find<T>();
    }

    void add_filter(TrajectoryFilter* filter);
    void require_discard_friction();

    // -----------------------------------------------------------------------
    // T7.2a test seam -- see the header note.  `run_stage` is the only addition
    // to the reference's surface; each branch is one of the private bodies below.
    // -----------------------------------------------------------------------
    enum class Stage
    {
        Detect,
        FilterActive,
        FilterTOI,
        RecordFrictionCandidates,
        LabelActiveVertices,
        ClearFrictionCandidates,
    };

    /// Drive one pipeline stage over the registered filters, exactly as the Newton
    /// loop / the contact exporter / the contact manager do.  Returns `filter_toi`'s
    /// reduced step fraction for `Stage::FilterTOI` and 1.0 otherwise.
    Float run_stage(Stage stage, Float alpha = 0.0);

  private:
    virtual void do_build() override final;
    virtual void do_apply_recover(RecoverInfo& info) override final;

    friend class SimEngine;
    friend class ContactExporterManager;
    void detect(Float alpha);  // called by SimEngine and ContactExporterManager
    void filter_active();      // called by SimEngine and ContactExporterManager

    Float filter_toi(Float alpha);       // only called by SimEngine
    void  record_friction_candidates();  // only called by SimEngine
    friend class GlobalContactManager;
    void label_active_vertices();      // only called by GlobalContactManager
    void clear_friction_candidates();  // called by GlobalContactManager

    Impl m_impl;
};
}// namespace uipc::backend::luisa
