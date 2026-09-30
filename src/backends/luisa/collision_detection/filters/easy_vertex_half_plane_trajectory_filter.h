#pragma once
/********************************************************************
 * @file easy_vertex_half_plane_trajectory_filter.h
 * @brief The IPC vertex-vs-half-plane trajectory filter (port of
 *        src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h,
 *        43 lines).
 *
 * The IPC flavour of the pair: `do_filter_active` produces the candidate
 * `PHs = (vertex, half_plane_index)` set (the vertices whose plane gap is inside
 * `(thickness, thickness + d_hat)` and whose contact/subscene masks allow the pair), and
 * `do_filter_toi` computes, per *surface* vertex, the smallest swept-plane TOI with
 * `eta = 0.1` slack, then reduces the minimum into the manager's `toi` slot.
 *
 * `do_detect` is the reference's explicit no-op (the half-plane filters do all their
 * work in the active/toi stages).
 *
 * `candidate_PHs()` / `toi_PHs()` return the filter's own buffers (the reference's
 * `GlobalActiveSetManager` reads exactly these), so the test can compare the device's
 * candidate set and TOI grid against its host oracle.
 *********************************************************************/
#include <collision_detection/vertex_half_plane_trajectory_filter.h>

namespace uipc::backend::luisa
{
class EasyVertexHalfPlaneTrajectoryFilter final : public VertexHalfPlaneTrajectoryFilter
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

    /// The number of `[Vertex-HalfPlane]` pairs `filter_active` last recorded.
    ///
    /// A **port-only test seam** (the reference has no accessor): `candidate_PHs()` cannot
    /// express it, because the port pads the pair buffer to one element so that it stays a
    /// valid LC kernel argument -- so a zero-pair and a one-pair run both report
    /// `candidate_PHs().size() == 1`.  §7.35's plane-less-scene rule is the only user.
    [[nodiscard]] IndexT num_pairs() const noexcept { return m_impl.h_num_collisions; }

  private:
    Impl m_impl;

    // Inherited via VertexHalfPlaneTrajectoryFilter

    virtual void do_build(BuildInfo& info) override;
    virtual void do_detect(DetectInfo& info) override;
    virtual void do_filter_active(FilterActiveInfo& info) override;
    virtual void do_filter_toi(FilterTOIInfo& info) override;
};
}// namespace uipc::backend::luisa
