#pragma once
#include <collision_detection/trajectory_filter.h>
#include <collision_detection/global_trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_body_manager.h>
#include <contact_system/global_contact_manager.h>
#include <luisa/runtime/buffer.h>
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

        // Vertex Attributes

        /**
         * @brief Vertex Id to Body Id mapping.
         */
        BufferView<Float>    d_hats() const noexcept;
        BufferView<IndexT>   v2b() const noexcept;
        BufferView<Vector3>  positions() const noexcept;
        BufferView<Vector3>  rest_positions() const noexcept;
        BufferView<Float>    thicknesses() const noexcept;
        BufferView<IndexT>   dimensions() const noexcept;
        BufferView<IndexT>   contact_element_ids() const noexcept;
        BufferView<IndexT>   subscene_element_ids() const noexcept;
        BufferView<IndexT>   contact_mask_tabular() const noexcept;
        BufferView<IndexT>   subscene_mask_tabular() const noexcept;

        // Body Attributes

        /**
         * @brief Tell if the body needs self-collision
         */
        BufferView<IndexT> body_self_collision() const noexcept;

        // Topologies

        BufferView<IndexT>   codim_vertices() const noexcept;
        BufferView<IndexT>   surf_vertices() const noexcept;
        BufferView<Vector2i> surf_edges() const noexcept;
        BufferView<Vector3i> surf_triangles() const noexcept;

      protected:
        friend class SimplexTrajectoryFilter;
        Impl* m_impl = nullptr;
    };

    class DetectInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        Float alpha() const noexcept { return m_alpha; }

        BufferView<Vector3> displacements() const noexcept;

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
        void PTs(BufferView<Vector4i> PTs) noexcept;
        /**
         * @brief Candidate edge-edge pairs.
         */
        void EEs(BufferView<Vector4i> EEs) noexcept;
        /**
         * @brief Candidate point-edge pairs.
         */
        void PEs(BufferView<Vector3i> PEs) noexcept;
        /**
         * @brief Candidate point-point pairs.
         */
        void PPs(BufferView<Vector2i> PPs) noexcept;
    };

    class FilterTOIInfo : public DetectInfo
    {
      public:
        using DetectInfo::DetectInfo;

        BufferView<Float> toi() noexcept;

      private:
        friend class SimplexTrajectoryFilter;
        BufferView<Float> m_toi;
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
        SimSystemSlot<GlobalBodyManager>    global_body_manager;

        BufferView<Vector4i> PTs;
        BufferView<Vector4i> EEs;
        BufferView<Vector3i> PEs;
        BufferView<Vector2i> PPs;

        Buffer<Vector4i> friction_PT;
        Buffer<Vector4i> friction_EE;
        Buffer<Vector3i> friction_PE;
        Buffer<Vector2i> friction_PP;

        Buffer<Vector4i> recovered_PT;
        Buffer<Vector4i> recovered_EE;
        Buffer<Vector3i> recovered_PE;
        Buffer<Vector2i> recovered_PP;

        Float reserve_ratio = 1.1;

        BufferDump dump_PTs;
        BufferDump dump_EEs;
        BufferDump dump_PEs;
        BufferDump dump_PPs;

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

    BufferView<Vector4i> PTs() const noexcept;
    BufferView<Vector4i> EEs() const noexcept;
    BufferView<Vector3i> PEs() const noexcept;
    BufferView<Vector2i> PPs() const noexcept;

    BufferView<Vector4i> friction_PTs() const noexcept;
    BufferView<Vector4i> friction_EEs() const noexcept;
    BufferView<Vector3i> friction_PEs() const noexcept;
    BufferView<Vector2i> friction_PPs() const noexcept;

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
    friend class GlobalDCDFilter;
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
}  // namespace uipc::backend::luisa
