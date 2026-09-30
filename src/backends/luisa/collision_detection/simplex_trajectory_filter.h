#pragma once
/********************************************************************
 * @file simplex_trajectory_filter.h
 * @brief The simplex-pair trajectory-filter base (port of
 *        src/backends/cuda/collision_detection/simplex_trajectory_filter.h, 207 lines).
 *
 * WHAT IT IS
 * ----------
 * The shared base of the four BVH simplex filters (T7.2b).  It owns the four candidate
 * pair arrays the reference's filters produce (PT/EE/PE/PP), the per-array
 * friction-candidate copies, the recovery dumps, the vertex/element attribute surface
 * (`BaseInfo`) and the two device kernels that label the CFL-active vertices.
 *
 * CONTRACTS (pinned by the [filter] test)
 * --------------------------------------
 *  * `FilterActiveInfo::PTs/EEs/PEs/PPs(...)` *store the view* the filter produced into
 *    the base's `Impl`; the base's own `PTs()/EEs()/PEs()/PPs()` accessors then hand the
 *    same views to the consumers (`GlobalActiveSetManager`, `GlobalContactManager`'s
 *    feasible-step pre-cap).  Nothing is copied and no size is validated here.
 *  * `Impl::loose_resize` is the reference's `resize_discard` policy: grow to
 *    `size * reserve_ratio` (1.1) when the size exceeds the capacity, then resize.  The
 *    friction arrays therefore keep their capacity across frames (the retry/growth
 *    contract the test pins).
 *  * `candidate_PTs()/candidate_EEs()/toi_PTs()/toi_EEs()` are the *derived* filter's
 *    exported result (the reference's `GlobalActiveSetManager` and
 *    `GlobalContactManager::compute_feasible_step` read exactly these).
 *  * the vertex-half-plane filters are *not* SimplexTrajectoryFilters: the reference has
 *    two independent bases that both derive from TrajectoryFilter.
 *
 * TYPE MAPPINGS
 * -------------
 *  * `CBufferView<T>` -> `lud::LCBufferView<T>`; `DeviceBuffer<T>` -> `lud::LDeviceBuffer<T>`;
 *    `VarView<Float>` -> `lud::LBufferView<Float>`; `Vector4i/Vector3i/Vector2i` ->
 *    `LVector4i/LVector3i/LVector2i` (the registered device element types).
 *  * **the flat mask layout**: `CBuffer2DView<IndexT> contact_mask_tabular()` becomes a
 *    flat `LCBufferView<IndexT>` (`index = row * extent + col`) plus the extent, which
 *    is why `BaseInfo` also exposes `contact_mask_size()` / `subscene_mask_size()`.
 *    The reference's 2D view carried the extent itself; the port's device layer has no
 *    2D buffer (T3.1/T3.2 decision -- see contact_system/global_contact_manager.h).
 *  * `resize_discard(n)` -> `loose_resize(buffer, n)` (`reserve` + `resize`), the T4.3
 *    dytopo reporter's established spelling of `cuda_tool::DeviceVector::resize_discard`.
 *
 * `friend class GlobalDCDFilter` is the reference's own (an un-ported consumer, T8.1);
 * the friendship is declared, never used, exactly like the reference.
 *********************************************************************/
#include <collision_detection/trajectory_filter.h>
#include <collision_detection/global_trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_body_manager.h>
#include <contact_system/global_contact_manager.h>
#include <type_define.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class SimplexTrajectoryFilter : public TrajectoryFilter
{
  public:
    using TrajectoryFilter::TrajectoryFilter;

    class Impl;

    class BuildInfo
    {
      public:
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        Float d_hat() const noexcept;
        Float toi_safety_margin() const noexcept;

        // Vertex Attributes

        /**
         * @brief Vertex Id to Body Id mapping.
         */
        lud::LCBufferView<Float>    d_hats() const noexcept;
        lud::LCBufferView<IndexT>   v2b() const noexcept;
        lud::LCBufferView<LVector3> positions() const noexcept;
        lud::LCBufferView<LVector3> rest_positions() const noexcept;
        lud::LCBufferView<Float>    thicknesses() const noexcept;
        lud::LCBufferView<IndexT>   dimensions() const noexcept;
        lud::LCBufferView<IndexT>   contact_element_ids() const noexcept;
        lud::LCBufferView<IndexT>   subscene_element_ids() const noexcept;
        lud::LCBufferView<IndexT>   contact_mask_tabular() const noexcept;
        lud::LCBufferView<IndexT>   subscene_mask_tabular() const noexcept;
        /// the extents of the flat mask layouts (see the header note)
        IndexT contact_mask_size() const noexcept;
        IndexT subscene_mask_size() const noexcept;
        // Body Attributes

        /**
         * @brief Tell if the body needs self-collision
         */
        lud::LCBufferView<IndexT> body_self_collision() const noexcept;

        // Topologies

        lud::LCBufferView<IndexT>    codim_vertices() const noexcept;
        lud::LCBufferView<IndexT>    surf_vertices() const noexcept;
        lud::LCBufferView<LVector2i> surf_edges() const noexcept;
        lud::LCBufferView<LVector3i> surf_triangles() const noexcept;

      protected:
        friend class SimplexTrajectoryFilter;
        Impl* m_impl = nullptr;
    };

    class DetectInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        Float alpha() const noexcept { return m_alpha; }

        lud::LCBufferView<LVector3> displacements() const noexcept;

      private:
        friend class SimplexTrajectoryFilter;
        Float m_alpha = 0.0;
    };

    class FilterActiveInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        /**
         * @brief Candidate point-triangle pairs.
         */
        void PTs(lud::LCBufferView<LVector4i> PTs) noexcept;
        /**
         * @brief Candidate edge-edge pairs.
         */
        void EEs(lud::LCBufferView<LVector4i> EEs) noexcept;
        /**
         * @brief Candidate point-edge pairs.
         */
        void PEs(lud::LCBufferView<LVector3i> PEs) noexcept;
        /**
         * @brief Candidate point-point pairs.
         */
        void PPs(lud::LCBufferView<LVector2i> PPs) noexcept;
    };

    class FilterTOIInfo : public DetectInfo
    {
      public:
        using DetectInfo::DetectInfo;

        lud::LBufferView<Float> toi() noexcept;

      private:
        friend class SimplexTrajectoryFilter;
        lud::LBufferView<Float> m_toi;
    };

    class Impl
    {
      public:
        void record_friction_candidates(GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info);
        void label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info);
        bool dump(DumpInfo& info);
        bool try_recover(RecoverInfo& info);
        void apply_recover(RecoverInfo& info);
        void clear_recover(RecoverInfo& info);

        SimSystemSlot<GlobalVertexManager>            global_vertex_manager;
        SimSystemSlot<GlobalSimplicialSurfaceManager> global_simplicial_surface_manager;
        SimSystemSlot<GlobalContactManager>           global_contact_manager;
        SimSystemSlot<GlobalBodyManager>              global_body_manager;
        Float                                         toi_safety_margin = 0.1;

        lud::LCBufferView<LVector4i> PTs;
        lud::LCBufferView<LVector4i> EEs;
        lud::LCBufferView<LVector3i> PEs;
        lud::LCBufferView<LVector2i> PPs;

        lud::LDeviceBuffer<LVector4i> friction_PT;
        lud::LDeviceBuffer<LVector4i> friction_EE;
        lud::LDeviceBuffer<LVector3i> friction_PE;
        lud::LDeviceBuffer<LVector2i> friction_PP;

        lud::LDeviceBuffer<LVector4i> recovered_PT;
        lud::LDeviceBuffer<LVector4i> recovered_EE;
        lud::LDeviceBuffer<LVector3i> recovered_PE;
        lud::LDeviceBuffer<LVector2i> recovered_PP;

        Float reserve_ratio = 1.1;

        BufferDump dump_PTs;
        BufferDump dump_EEs;
        BufferDump dump_PEs;
        BufferDump dump_PPs;

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
    };

    lud::LCBufferView<LVector4i> PTs() const noexcept;
    lud::LCBufferView<LVector4i> EEs() const noexcept;
    lud::LCBufferView<LVector3i> PEs() const noexcept;
    lud::LCBufferView<LVector2i> PPs() const noexcept;

    lud::LCBufferView<LVector4i> friction_PTs() const noexcept;
    lud::LCBufferView<LVector4i> friction_EEs() const noexcept;
    lud::LCBufferView<LVector3i> friction_PEs() const noexcept;
    lud::LCBufferView<LVector2i> friction_PPs() const noexcept;

    virtual lud::LCBufferView<LVector2i> candidate_PTs() const noexcept = 0;
    virtual lud::LCBufferView<LVector2i> candidate_EEs() const noexcept = 0;
    virtual lud::LCBufferView<Float>     toi_PTs() const noexcept       = 0;
    virtual lud::LCBufferView<Float>     toi_EEs() const noexcept       = 0;

  protected:
    virtual void do_build(BuildInfo& info)                = 0;
    virtual void do_detect(DetectInfo& info)              = 0;
    virtual void do_filter_active(FilterActiveInfo& info) = 0;
    virtual void do_filter_toi(FilterTOIInfo& info)       = 0;
    virtual bool do_dump(DumpInfo& info) override;
    virtual bool do_try_recover(RecoverInfo& info) override;
    virtual void do_apply_recover(RecoverInfo& info) override;
    virtual void do_clear_recover(RecoverInfo& info) override;

  private:
    friend class GlobalDCDFilter;  // SCOPE NOTE: an un-ported consumer (T8.1)
    Impl m_impl;

    virtual void do_build() override final;

    virtual void do_detect(GlobalTrajectoryFilter::DetectInfo& info) override final;
    virtual void do_filter_active(GlobalTrajectoryFilter::FilterActiveInfo& info) override final;
    virtual void do_filter_toi(GlobalTrajectoryFilter::FilterTOIInfo& info) override final;
    virtual void do_record_friction_candidates(
        GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info) override final;
    virtual void do_label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info) final override;
    virtual void do_clear_friction_candidates() override final;
};
}// namespace uipc::backend::luisa
