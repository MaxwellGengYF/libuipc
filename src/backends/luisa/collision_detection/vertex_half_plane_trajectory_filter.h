#pragma once
#include <collision_detection/trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <luisa/runtime/buffer.h>
#include <implicit_geometry/half_plane.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class HalfPlaneVertexReporter;
class VertexHalfPlaneTrajectoryFilter : public TrajectoryFilter
{
  public:
    using TrajectoryFilter::TrajectoryFilter;

    class Impl;

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        Float                    d_hat() const noexcept;
        BufferView<Float> d_hats() const noexcept;

        IndexT                     half_plane_vertex_offset() const noexcept;
        BufferView<Vector3> plane_normals() const noexcept;
        BufferView<Vector3> plane_positions() const noexcept;

        BufferView<Vector3>  positions() const noexcept;
        BufferView<Float>    thicknesses() const noexcept;
        BufferView<IndexT>   contact_element_ids() const noexcept;
        BufferView<IndexT>   subscene_element_ids() const noexcept;
        BufferView<IndexT>   contact_mask_tabular() const noexcept;
        BufferView<IndexT>   subscene_mask_tabular() const noexcept;
        BufferView<IndexT>   surf_vertices() const noexcept;

      private:
        friend class VertexHalfPlaneTrajectoryFilter;
        Impl* m_impl;
    };

    class DetectInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        Float                      alpha() const noexcept { return m_alpha; }
        BufferView<Vector3> displacements() const noexcept;

      private:
        friend class VertexHalfPlaneTrajectoryFilter;
        Float m_alpha = 0.0;
    };

    class FilterActiveInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        /**
         * @brief Candidate vertex-half-plane pairs.
         */
        void PHs(BufferView<Vector2i> Ps) noexcept;
    };

    class FilterTOIInfo : public DetectInfo
    {
      public:
        using DetectInfo::DetectInfo;

        BufferView<Float> toi() noexcept;

      private:
        friend class VertexHalfPlaneTrajectoryFilter;
        BufferView<Float> m_toi;
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

        SimSystemSlot<GlobalVertexManager> global_vertex_manager;
        SimSystemSlot<GlobalSimplicialSurfaceManager> global_simplicial_surface_manager;
        SimSystemSlot<GlobalContactManager> global_contact_manager;
        SimSystemSlot<HalfPlane> half_plane;
        SimSystemSlot<HalfPlaneVertexReporter> half_plane_vertex_reporter;

        BufferView<Vector2i>  PHs;
        Buffer<Vector2i> friction_PHs;
        Buffer<Vector2i> recovered_PHs;

        Float reserve_ratio = 1.1;

        BufferDump dump_PHs;

        template <typename T>
        void loose_resize(Buffer<T>& buffer, SizeT size)
        {
            if(size > buffer.size())
            {
                // In LuisaCompute, we need to create a new buffer with larger capacity
                // The actual resize/reallocation is handled by the device
                // This is a simplified version - actual implementation may need
                // device reference for buffer recreation
            }
        }
    };

    BufferView<Vector2i> PHs() const noexcept;
    BufferView<Vector2i> friction_PHs() const noexcept;

  protected:
    virtual void do_detect(DetectInfo& info)              = 0;
    virtual void do_filter_active(FilterActiveInfo& info) = 0;
    virtual void do_filter_toi(FilterTOIInfo& info)       = 0;

    virtual void do_build(BuildInfo& info){};
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
}  // namespace uipc::backend::luisa
