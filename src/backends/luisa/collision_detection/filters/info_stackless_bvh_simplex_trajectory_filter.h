#pragma once
/********************************************************************
 * @file info_stackless_bvh_simplex_trajectory_filter.h
 * @brief InfoStacklessBVHSimplexTrajectoryFilter -- the `info_stackless_bvh` (v1)
 *        flavour of the simplex trajectory filter (port of
 *        src/backends/cuda/collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.h
 *        (91 lines) + .cu (1809 lines)).
 *
 * WHAT IT IS
 * ----------
 * The v1 "info" flavour of the four M7 broad-phase trajectory filters.  It drives
 * T7.1b's `InfoStacklessBVH` (the stackless tree plus per-leaf/per-node body/contact
 * metadata and the two-predicate query contract) and carries the reference's own
 * per-primitive bid/cid arrays, built by the detect kernels next to the swept AABBs.
 *
 * TWO DIFFERENCES FROM THE LBVH/STACKLESS/V0 FLAVOURS (the four-way table)
 * -----------------------------------------------------------------------
 *  1. **the `alpha == 0` DCD branch.**  Each of the four detect predicates has an
 *     `if(alpha == 0.0)` branch that replaces the conservative `*_ccd_broadphase` box
 *     test with the *exact* squared-distance test (`D2 < expand*expand`), because at
 *     alpha == 0 the ccd box test degenerates to a per-axis test and passes ~50x more
 *     candidates.  This is the *only* behavioural difference between the four variants
 *     (the port's shared kernels encode it as the `Dcd` template flag); v0 and the
 *     plain flavours have no such branch.
 *  2. **the batched count download.**  Its `detect` splits each query into
 *     `launch_query` + `prepare_query_result` so the four `cpNum` values can be read
 *     back in *one* transfer (`collect_query_counts_kernel` + `query_counts.copy_to`),
 *     and `filter_active` downloads its four selection counts the same way
 *     (`selected_counts` is one four-slot buffer).  The other three variants read each
 *     count where the BVH's own retry needs it.  The pair *sets* are unaffected.
 *
 * THE PREDICATE / DEVICE-DATA DECISION (re-verified for this variant)
 * ------------------------------------------------------------------
 * `InfoStacklessBVH`'s `NodePred` **is** a node-level cull that reads device data: the
 * reference's predicate is
 *
 *     bool bid_cull = node_bid != -1 && qbid != -1 && qbid == node_bid
 *                     && !body_self_collision(qbid);
 *     bool cid_cull = node_cid != -1 && qcid != -1 && !cmts(qcid, node_cid);
 *     return !(bid_cull || cid_cull);
 *
 * so it prunes subtrees using `body_self_collision` and the *contact* mask (the ported
 * `NodePred` cannot read either: a captured view becomes a shader-bound buffer,
 * `§7.32 rule 3`).  The port passes `KeepAllNodes` instead, and that is *exact* here
 * because every cull condition is implied by the leaf predicate:
 *
 *  * a node's bid/cid are `-1` unless *every* leaf below shares them (the ported
 *    `propagateInformativeMetadata` contract, `§7.32`), so for a culled internal node
 *    with `node_bid == qbid` every leaf pair inside it has `bid_i == bid_j == qbid` and
 *    `!body_self_collision(qbid)` -- which is exactly one of the leaf predicate's
 *    rejections (`info.bid_i == info.bid_j && bid_i != -1 && !body_self_collision`);
 *  * `cid_cull`'s `!cmts(qcid, node_cid)` is the *first conjunct* of the leaf
 *    predicate's `allow_PT/PE/PP/EE_contact` group (each is a conjunction of
 *    `table[cids[0] * extent + cids[k]] != 0` terms, and the node's cid is the leaf's
 *    cid for a uniform node), so a cid-culled node can only contain pairs the leaf
 *    predicate rejects.
 *
 * No "device data reaches the predicate" hook is therefore needed (the same conclusion
 * T7.2a reached for the family, re-derived here from this variant's own predicate).
 * The test pins the equivalence by comparing the device's candidate sets with a
 * brute-force host oracle, and pins the bid/cid arrays themselves
 * (`point_bids[i] == v2b(Vs[i])` etc.).
 *
 * TYPE MAPPINGS: as in `lbvh_simplex_trajectory_filter.h`, plus the info variant's
 * `build(aabbs, BIDs, CIDs)` and `launch_query/launch_detect + prepare_query_result`
 * (T7.1b's API); the port's `query_counts` / `selected_counts` are `IndexT` / `uint32_t`
 * four-slot buffers exactly as the reference spells them.
 *********************************************************************/
#include <sim_system.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/details/simplex_trajectory_filter_kernels.h>
#include <collision_detection/info_stackless_bvh.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::luisa
{
class InfoStacklessBVHSimplexTrajectoryFilter final : public SimplexTrajectoryFilter
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

        using ThisBVH = InfoStacklessBVH;

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

        // The reported (predicate-selected) candidate pairs (the reference's
        // `candidate_XX_pairs` members; see the LBVH flavour's header note).
        sftd::SelectedPairs candidate_AllP_CodimP_pairs;
        sftd::SelectedPairs candidate_CodimP_AllE_pairs;
        sftd::SelectedPairs candidate_AllE_AllE_pairs;
        sftd::SelectedPairs candidate_AllP_AllT_pairs;

        // Four query counts and four compacted PP/PE/PT/EE counts are each
        // downloaded in one contiguous transfer.
        lud::LDeviceBuffer<IndexT>       query_counts;
        lud::LDeviceBuffer<std::uint32_t> selected_counts;

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
