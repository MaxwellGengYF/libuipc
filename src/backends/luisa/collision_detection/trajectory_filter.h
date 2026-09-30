#pragma once
/********************************************************************
 * @file trajectory_filter.h
 * @brief The trajectory-filter interface (port of
 *        src/backends/cuda/collision_detection/trajectory_filter.h, 30 lines).
 *
 * A `TrajectoryFilter` is a `SimSystem` that GlobalTrajectoryFilter owns and drives
 * through the reference's six pipeline stages.  The public API is empty: every stage
 * is a protected `do_*` the concrete filters implement, reached only by
 * GlobalTrajectoryFilter (hence `friend class GlobalTrajectoryFilter`), which is the
 * reference's own access shape.
 *
 * `do_record_friction_candidates` / `do_label_active_vertices` /
 * `do_clear_friction_candidates` are the friction/CFL hooks; the vertex-half-plane
 * filters implement all six (three of them as the reference's no-ops).
 *********************************************************************/
#include <sim_system.h>
#include <collision_detection/global_trajectory_filter.h>

namespace uipc::backend::luisa
{
class TrajectoryFilter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  protected:
    virtual void do_detect(GlobalTrajectoryFilter::DetectInfo& info)              = 0;
    virtual void do_filter_active(GlobalTrajectoryFilter::FilterActiveInfo& info) = 0;
    virtual void do_filter_toi(GlobalTrajectoryFilter::FilterTOIInfo& info)       = 0;
    virtual void do_record_friction_candidates(
        GlobalTrajectoryFilter::RecordFrictionCandidatesInfo&) = 0;
    virtual void do_label_active_vertices(
        GlobalTrajectoryFilter::LabelActiveVerticesInfo& info) = 0;
    virtual void do_clear_friction_candidates() = 0;

  private:
    friend class GlobalTrajectoryFilter;

    void detect(GlobalTrajectoryFilter::DetectInfo& info);
    void filter_active(GlobalTrajectoryFilter::FilterActiveInfo& info);
    void filter_toi(GlobalTrajectoryFilter::FilterTOIInfo& info);
    void record_friction_candidates(GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info);
    void label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info);
    void clear_friction_candidates();
};
}// namespace uipc::backend::luisa
