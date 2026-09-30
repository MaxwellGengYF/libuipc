#pragma once
/********************************************************************
 * @file al_vertex_half_plane_trajectory_filter.h
 * @brief The AL-IPC vertex-vs-half-plane trajectory filter (port of
 *        src/backends/cuda/collision_detection/filters/al_vertex_half_plane_trajectory_filter.h,
 *        40 lines).
 *
 * The augmented-Lagrangian twin of EasyVertexHalfPlaneTrajectoryFilter.  It is created
 * in *every* scene like its sibling (no SimSystemCreator specialisation in the reference
 * either); `do_build` requires ALIPCPipelineFlag, so in an IPC scene the engine's
 * SimSystemException handler invalidates it and only the Easy filter survives -- and
 * vice versa.
 *
 * THE THREE MEASURABLE DIFFERENCES FROM THE EASY FILTER (the [filter] test pins all
 * three; see the .cpp for the transcribed bodies):
 *  1. `do_filter_active` is a no-op: the AL candidate set is *not* the mask/D-range
 *     filtered one -- it is the full `surf_vertices x plane_instances` grid that
 *     `filter_toi` writes (`PHs(i * n_planes + j) = {vI, j}` for EVERY pair, before its
 *     mask checks), so `candidate_PHs().size() == n_surf * n_planes`.
 *  2. the toi slackness is `eta = 0.001` (the Easy filter uses 0.1);
 *  3. `filter_toi` writes a per-*pair* toi grid (1.1 for a pair that moves away or is
 *     mask-disabled) and min-reduces the whole grid, instead of writing one per-vertex
 *     minimum.  `toi_PHs()` therefore exposes `n_surf * n_planes` values.
 * The AL filter also ignores `d_hat` completely (no `is_active_D` band), while the Easy
 * filter's candidate production is exactly that band.
 *********************************************************************/
#include <collision_detection/vertex_half_plane_trajectory_filter.h>

namespace uipc::backend::luisa
{
class ALVertexHalfPlaneTrajectoryFilter final : public VertexHalfPlaneTrajectoryFilter
{
  public:
    using VertexHalfPlaneTrajectoryFilter::VertexHalfPlaneTrajectoryFilter;

    class Impl
    {
      public:
        void filter_active(FilterActiveInfo& info);
        void filter_toi(FilterTOIInfo& info);

        lud::LDeviceVar<IndexT> num_collisions;
        IndexT                  h_num_collisions = 0;

        /**
         * @brief [Vertex-HalfPlane] pairs
         */
        lud::LDeviceBuffer<LVector2i> PHs;

        Float reserve_ratio = 1.1f;

        lud::LDeviceBuffer<Float> tois;
    };

    virtual lud::LCBufferView<LVector2i> candidate_PHs() const noexcept override;
    virtual lud::LCBufferView<Float>     toi_PHs() const noexcept override;

  private:
    Impl         m_impl;
    virtual void do_build(BuildInfo& info) override;
    virtual void do_detect(DetectInfo& info) override;
    virtual void do_filter_active(FilterActiveInfo& info) override;
    virtual void do_filter_toi(FilterTOIInfo& info) override;
};
}// namespace uipc::backend::luisa
