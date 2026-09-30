#pragma once
/********************************************************************
 * @file info_stackless_bvh_v0_simplex_trajectory_filter.h
 * @brief InfoStacklessBVHV0SimplexTrajectoryFilter -- the `info_stackless_bvh_v0`
 *        flavour of the simplex trajectory filter (port of
 *        src/backends/cuda/collision_detection/filters/info_stackless_bvh_v0_simplex_trajectory_filter.h
 *        (91 lines) + .cu (1731 lines)).
 *
 * WHAT IT IS
 * ----------
 * The v0 "info" flavour: the same structure as the v1 flavour
 * (`info_stackless_bvh_simplex_trajectory_filter.h`) over T7.1b's
 * `InfoStacklessBVHV0`.  Read the v1 header's note first -- it carries the
 * predicate/device-data decision and its proof.  The **two differences** between v0
 * and v1 (measured with a normalising diff of the two reference files) are:
 *
 *  1. **no `alpha == 0` DCD branch.**  v0's four leaf predicates are the *same bodies*
 *     as the LBVH/Stackless variants' (they read `v2b` instead of the resolved
 *     `LeafPredInfo::bid_i/bid_j`, which carry the same values -- `bids(i) =
 *     v2bs(first vertex)` -- so the two spellings are equivalent); the info v1 flavour
 *     is the only one with the exact-distance branch.  The port's shared kernels encode
 *     this as `Dcd == false`, so **v0's candidate sets must equal the stackless
 *     flavour's on the same scene at every alpha** -- the [bvh_filter] four-way table
 *     pins exactly that.
 *  2. **the per-count downloads.**  v0 calls the plain `InfoStacklessBVHV0::query` /
 *     `detect` (each doing its own retry read-back) and reads its four filter-active
 *     counts from four separate `DeviceVar`s, where v1 splits the query
 *     (`launch_query` + `prepare_query_result`) to batch the four counts into one
 *     transfer and keeps its four selection counts in one four-slot buffer.  The pair
 *     sets are unaffected; the *number* of host read-backs is not.
 *
 * The info structures' node cull is dropped in the port for the same (proved) reason as
 * v1: every cull condition is implied by the leaf predicate, so keeping all nodes is
 * exact.  The bid/cid arrays (and therefore the node metadata the cull consumed) are
 * still built and pinned (`point_bids[i] == v2b(Vs[i])`, `point_cids[i] ==
 * contact_element_ids(Vs[i])`, and the edge/triangle arrays' first-vertex convention).
 *********************************************************************/
#include <sim_system.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/details/simplex_trajectory_filter_kernels.h>
#include <collision_detection/info_stackless_bvh_v0.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::luisa
{
class InfoStacklessBVHV0SimplexTrajectoryFilter final : public SimplexTrajectoryFilter
{
  public:
    using SimplexTrajectoryFilter::SimplexTrajectoryFilter;

    class Impl
    {
      public:
        void detect(DetectInfo& info);
        void filter_active(FilterActiveInfo& info);
        void filter_toi(FilterTOIInfo& info);

        /****************************************************
        *                   Broad Phase
        ****************************************************/

        lud::LDeviceBuffer<LAABB>   codim_point_aabbs;
        lud::LDeviceBuffer<LAABB>   point_aabbs;
        lud::LDeviceBuffer<LAABB>   edge_aabbs;
        lud::LDeviceBuffer<LAABB>   triangle_aabbs;
        lud::LDeviceBuffer<IndexT>  codim_point_bids;
        lud::LDeviceBuffer<IndexT>  point_bids;
        lud::LDeviceBuffer<IndexT>  edge_bids;
        lud::LDeviceBuffer<IndexT>  triangle_bids;
        lud::LDeviceBuffer<IndexT>  codim_point_cids;
        lud::LDeviceBuffer<IndexT>  point_cids;
        lud::LDeviceBuffer<IndexT>  edge_cids;
        lud::LDeviceBuffer<IndexT>  triangle_cids;

        using ThisBVH = InfoStacklessBVHV0;

        // CodimP count always less or equal to AllP count.
        ThisBVH              lbvh_CodimP;
        ThisBVH::QueryBuffer raw_AllP_CodimP_pairs;

        // Used to detect CodimP-AllE, and AllE-AllE pairs.
        ThisBVH              lbvh_E;
        ThisBVH::QueryBuffer raw_CodimP_AllE_pairs;
        ThisBVH::QueryBuffer raw_AllE_AllE_pairs;

        // Used to detect AllP-AllT pairs.
        ThisBVH              lbvh_T;
        ThisBVH::QueryBuffer raw_AllP_AllT_pairs;

        // The reported (predicate-selected) candidate pairs.
        sftd::SelectedPairs candidate_AllP_CodimP_pairs;
        sftd::SelectedPairs candidate_CodimP_AllE_pairs;
        sftd::SelectedPairs candidate_AllE_AllE_pairs;
        sftd::SelectedPairs candidate_AllP_AllT_pairs;

        lud::LDeviceVar<std::uint32_t> selected_PT_count;
        lud::LDeviceVar<std::uint32_t> selected_EE_count;
        lud::LDeviceVar<std::uint32_t> selected_PE_count;
        lud::LDeviceVar<std::uint32_t> selected_PP_count;

        lud::LDeviceBuffer<LVector4i> temp_PTs;
        lud::LDeviceBuffer<LVector4i> temp_EEs;
        lud::LDeviceBuffer<LVector3i> temp_PEs;
        lud::LDeviceBuffer<LVector2i> temp_PPs;

        lud::LDeviceBuffer<LVector4i> PTs;
        lud::LDeviceBuffer<LVector4i> EEs;
        lud::LDeviceBuffer<LVector3i> PEs;
        lud::LDeviceBuffer<LVector2i> PPs;


        /****************************************************
        *                   CCD TOI
        ****************************************************/

        lud::LDeviceBuffer<Float> tois;  // PP, PE, PT, EE

        /// T7.2b: the detect-time selection machinery + the four reported counts
        /// `[PP, PE, PT, EE]`.
        sftd::SelectionWorkspace          selection;
        lud::LDeviceBuffer<std::uint32_t> candidate_counts;
    };

    virtual lud::LCBufferView<LVector2i> candidate_PTs() const noexcept override;
    virtual lud::LCBufferView<LVector2i> candidate_EEs() const noexcept override;
    virtual lud::LCBufferView<Float>     toi_PTs() const noexcept override;
    virtual lud::LCBufferView<Float>     toi_EEs() const noexcept override;

  private:
    Impl m_impl;

    virtual void do_build(BuildInfo& info) override final;
    virtual void do_detect(DetectInfo& info) override final;
    virtual void do_filter_active(FilterActiveInfo& info) override final;
    virtual void do_filter_toi(FilterTOIInfo& info) override final;
};
}// namespace uipc::backend::luisa
