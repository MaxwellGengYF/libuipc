#pragma once
/********************************************************************
 * @file stackless_bvh_simplex_trajectory_filter.h
 * @brief StacklessBVHSimplexTrajectoryFilter -- the `stackless_bvh` flavour of the
 *        simplex trajectory filter (port of
 *        src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.h
 *        (83 lines) + .cu (1698 lines)).
 *
 * WHAT IT IS
 * ----------
 * One of the four M7 broad-phase trajectory filters.  Structurally identical to
 * `LBVHSimplexTrajectoryFilter` (read that file's note first: the keep-all traversal +
 * device selection-pass design decision, and the pair-buffer layering), with exactly
 * two differences:
 *
 *  1. the traversal structure: `ThisBVH = StacklessBVH` (T7.1a's culbvh build + the
 *     stackless `escape`-link walk) instead of `AtomicCountingLBVH`.  The reference's
 *     `StacklessBVH::QueryBuffer` reserves 50*1024 pairs up front and grows by
 *     `reserve_ratio = 1.2`, while `AtomicCountingLBVH::QueryBuffer` starts empty and
 *     grows by 1.1 -- the [bvh_filter] test pins both initial capacities.
 *  2. the reference's *own* stackless copy passes `max_iter` and `large_enough_toi`
 *     into the four TOI kernels as extra arguments (its `constexpr SizeT max_iter =
 *     1000;` / `constexpr Float large_enough_toi = 1.1;` live below the kernels and are
 *     forwarded at each dispatch), while the other three copies use the file-scope
 *     constants directly.  The values are identical, so the port's shared TOI kernels
 *     bake them in (the difference is recorded, not pinned by a value).
 *
 * Everything else -- the keep-all traversal calls, the four detect predicate
 * selection passes, the four classifiers, the four `DeviceSelect::If` compactions and
 * the TOI min-reduce -- is the same code as the LBVH flavour's (see the shared kernels
 * header).
 *
 * THE ONE PORT DESIGN DECISION (see details/simplex_trajectory_filter_kernels.h)
 * -----------------------------------------------------------------------------
 * The reference passes a *device* predicate into `AtomicCountingLBVH::detect/query`;
 * the port's predicates are record-time LC callables that cannot read per-primitive
 * device arrays (`§7.32 rule 3`, `§7.33 rules 1-3`).  The traversal therefore runs
 * with the keep-all predicate and the reference's predicate body is applied as a
 * device *selection pass* over the reported pairs -- the surviving set is exactly the
 * reference's (the predicate is called inside the traversal callback, i.e. after the
 * fixed `node.intersects(Q)` test, so it cannot change which nodes are visited).  What
 * differs is only the reported pair *count* (hence the pair-buffer retry), which
 * `§7.33 rule 1` already records as host-visible but not semantic.  Consequently the
 * Impl holds the reference's pair buffers in two layers:
 *
 *   `raw_XX_pairs`        the keep-all traversal's output (`ThisBVH::QueryBuffer`)
 *   `candidate_XX_pairs`  the reported (predicate-selected) set (`SelectedPairs`)
 *
 * `candidate_PTs()/candidate_EEs()/toi_PTs()/toi_EEs()` -- the four exports the
 * reference's `GlobalActiveSetManager`/`GlobalContactManager` read -- keep their names
 * and contents.
 *
 * TYPE MAPPINGS (`§3`, `§7.31`-`§7.33`)
 * -------------------------------------
 *  * `cuda_tool::DeviceBuffer/DeviceVar/BufferView` -> `lud::LDeviceBuffer/LDeviceVar/
 *    LBufferView`; `Vector2i/Vector3i/Vector4i` -> `LVector2i/LVector3i/LVector4i`;
 *    `AABB` -> `LAABB`.
 *  * `CDense2D<IndexT>` masks -> flat views + an explicit extent
 *    (`BaseInfo::contact_mask_size()/subscene_mask_size()`).
 *  * `resize_discard(n)` -> `resize(n)`; the reference's `selected_PP_count` etc. are
 *    `DeviceVar<IndexT>`, the port's are `DeviceVar<std::uint32_t>` because the ported
 *    compaction primitive writes a `uint32_t` count (`select_if`'s convention); the
 *    value is the same.
 *  * the reference's implicit device-scalar reads (`IndexT x = device_var;`) become
 *    `lud::read_host_sync(stream, view)` (`§7.10`).
 *  * `REGISTER_SIM_SYSTEM` + the `collision_detection/method == "linear_bvh"` gate in
 *    `do_build` are the reference's (a scene selects one of the four filters; the
 *    engine's `SimSystemException` handler invalidates the other three).
 *********************************************************************/
#include <sim_system.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/atomic_counting_lbvh.h>
#include <collision_detection/stackless_bvh.h>
#include <collision_detection/details/simplex_trajectory_filter_kernels.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::luisa
{
class StacklessBVHSimplexTrajectoryFilter final : public SimplexTrajectoryFilter
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

        lud::LDeviceBuffer<LAABB> codim_point_aabbs;
        lud::LDeviceBuffer<LAABB> point_aabbs;
        lud::LDeviceBuffer<LAABB> edge_aabbs;
        lud::LDeviceBuffer<LAABB> triangle_aabbs;

        using ThisBVH = StacklessBVH;

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

        // The reported (predicate-selected) candidate pairs -- the reference's
        // `candidate_XX_pairs` members (see the header note).
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
        /// `[PP, PE, PT, EE]` (see the header note).
        sftd::SelectionWorkspace       selection;
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
