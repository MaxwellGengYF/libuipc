#pragma once
/********************************************************************
 * @file vertex_half_plane_trajectory_filter.h
 * @brief The vertex-vs-half-plane trajectory-filter base (port of
 *        src/backends/cuda/collision_detection/vertex_half_plane_trajectory_filter.h,
 *        151 lines).
 *
 * WHAT IT IS
 * ----------
 * The second of the reference's two filter bases (the other is
 * `simplex_trajectory_filter.h`); both derive from `TrajectoryFilter`, and its two
 * concrete filters are `filters/easy_vertex_half_plane_trajectory_filter.*` (IPC) and
 * `filters/al_vertex_half_plane_trajectory_filter.*` (AL-IPC).
 *
 * It owns the candidate `PHs` array (`Vector2i{vertex, half_plane_index}`), its
 * friction/recovery copies, and the `BaseInfo` surface the derived filters read: the
 * per-vertex arrays from GlobalVertexManager, the two contact masks, the half plane's
 * per-instance normals/positions and the half-plane vertex slice's offset.
 *
 * CONTRACTS (pinned by the [filter] test)
 * --------------------------------------
 *  * `PHs()` returns the *stored view* (the last `FilterActiveInfo::PHs(...)` or the
 *    `filter_toi` grid the AL filter leaves there) -- no copy, no size validation.
 *  * `half_plane_vertex_offset()` is `VertexReporter::vertex_offset()` of the
 *    HalfPlaneVertexReporter (its slice of the global vertex arrays), NOT the half
 *    plane's geometry id; the derived filters use it to index the plane's vertex
 *    attributes (`contact_element_ids(vJ)` etc.).
 *  * `plane_positions()/plane_normals()` are per half-plane-*instance* (j indexes
 *    `0 .. plane_positions().size())`), which is why every filter's inner loop is
 *    `for(j = 0; j < half_plane_positions.total_size(); ++j)`.
 *  * `Impl` holds **raw pointers** here (the reference's own choice, unlike the
 *    Simplex base's `SimSystemSlot`s); `do_build()` resolves all five with `require<>`,
 *    so they are non-null for the lifetime of a valid filter.
 *
 * TYPE MAPPINGS
 * -------------
 * `CBufferView<T>` -> `lud::LCBufferView<T>`; `Vector2i/Vector3` ->
 * `LVector2i/LVector3`; `VarView<Float>` -> `lud::LBufferView<Float>`; the two
 * `CBuffer2DView<IndexT>` masks become flat views + an extent (see
 * simplex_trajectory_filter.h's header note), hence `contact_mask_size()` /
 * `subscene_mask_size()` on `BaseInfo`; `resize_discard` -> `loose_resize`.
 *********************************************************************/
#include <collision_detection/trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <type_define.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class VertexHalfPlaneTrajectoryFilter : public TrajectoryFilter
{
  public:
    using TrajectoryFilter::TrajectoryFilter;

    class Impl;

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl)
            : m_impl(impl)
        {
        }

        Float                       d_hat() const noexcept;
        lud::LCBufferView<Float>    d_hats() const noexcept;

        IndexT                           half_plane_vertex_offset() const noexcept;
        lud::LCBufferView<LVector3>      plane_normals() const noexcept;
        lud::LCBufferView<LVector3>      plane_positions() const noexcept;

        lud::LCBufferView<LVector3>  positions() const noexcept;
        lud::LCBufferView<Float>     thicknesses() const noexcept;
        lud::LCBufferView<IndexT>    contact_element_ids() const noexcept;
        lud::LCBufferView<IndexT>    subscene_element_ids() const noexcept;
        lud::LCBufferView<IndexT>    contact_mask_tabular() const noexcept;
        lud::LCBufferView<IndexT>    subscene_mask_tabular() const noexcept;
        /// the extents of the flat mask layouts (see the header note)
        IndexT                       contact_mask_size() const noexcept;
        IndexT                       subscene_mask_size() const noexcept;
        lud::LCBufferView<IndexT>    surf_vertices() const noexcept;

      private:
        friend class VertexHalfPlaneTrajectoryFilter;
        Impl* m_impl;
    };

    class DetectInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        Float                       alpha() const noexcept;
        lud::LCBufferView<LVector3> displacements() const noexcept;

      private:
        friend class VertexHalfPlaneTrajectoryFilter;
        Float m_alpha;
    };

    class FilterActiveInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        /**
         * @brief Candidate vertex-half-plane pairs.
         */
        void PHs(lud::LCBufferView<LVector2i> Ps) noexcept;
    };

    class FilterTOIInfo : public DetectInfo
    {
      public:
        using DetectInfo::DetectInfo;

        lud::LBufferView<Float> toi() noexcept;

      private:
        friend class VertexHalfPlaneTrajectoryFilter;
        lud::LBufferView<Float> m_toi;
    };

    class BuildInfo
    {
      public:
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

        GlobalVertexManager*            global_vertex_manager            = nullptr;
        GlobalSimplicialSurfaceManager* global_simplicial_surface_manager = nullptr;
        GlobalContactManager*           global_contact_manager           = nullptr;
        HalfPlane*                      half_plane                       = nullptr;
        HalfPlaneVertexReporter*        half_plane_vertex_reporter       = nullptr;

        lud::LCBufferView<LVector2i>  PHs;
        lud::LDeviceBuffer<LVector2i> friction_PHs;
        lud::LDeviceBuffer<LVector2i> recovered_PHs;

        Float reserve_ratio = 1.1;

        BufferDump dump_PHs;

        /// `cuda_tool::DeviceVector::resize_discard(n)` with the reference's growth policy.
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

    lud::LCBufferView<LVector2i>         PHs() noexcept;
    lud::LCBufferView<LVector2i>         friction_PHs() noexcept;
    virtual lud::LCBufferView<LVector2i> candidate_PHs() const noexcept = 0;
    virtual lud::LCBufferView<Float>     toi_PHs() const noexcept       = 0;

  protected:
    virtual void do_detect(DetectInfo& info)              = 0;
    virtual void do_filter_active(FilterActiveInfo& info) = 0;
    virtual void do_filter_toi(FilterTOIInfo& info)       = 0;

    virtual void do_build(BuildInfo& info) {};
    virtual bool do_dump(DumpInfo& info) override;
    virtual bool do_try_recover(RecoverInfo& info) override;
    virtual void do_apply_recover(RecoverInfo& info) override;
    virtual void do_clear_recover(RecoverInfo& info) override;

  private:
    Impl         m_impl;
    virtual void do_build() override final;

    virtual void do_detect(GlobalTrajectoryFilter::DetectInfo& info) override final;
    virtual void do_filter_active(GlobalTrajectoryFilter::FilterActiveInfo& info) override final;
    virtual void do_filter_toi(GlobalTrajectoryFilter::FilterTOIInfo& info) override final;
    virtual void do_record_friction_candidates(
        GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info) override final;
    virtual void do_label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info) override final;
    virtual void do_clear_friction_candidates() override final;
};
}// namespace uipc::backend::luisa
