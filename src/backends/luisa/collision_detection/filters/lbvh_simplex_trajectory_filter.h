#pragma once
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/atomic_counting_lbvh.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::luisa
{
class LBVHSimplexTrajectoryFilter final : public SimplexTrajectoryFilter
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

        Buffer<AABB> codim_point_aabbs;
        Buffer<AABB> point_aabbs;
        Buffer<AABB> edge_aabbs;
        Buffer<AABB> triangle_aabbs;

        using ThisBVH = AtomicCountingLBVH;

        // CodimP count always less or equal to AllP count.
        ThisBVH              lbvh_CodimP;
        ThisBVH::QueryBuffer candidate_AllP_CodimP_pairs;

        // Used to detect CodimP-AllE, and AllE-AllE pairs.
        ThisBVH              lbvh_E;
        ThisBVH::QueryBuffer candidate_CodimP_AllE_pairs;
        ThisBVH::QueryBuffer candidate_AllE_AllE_pairs;

        // Used to detect AllP-AllT pairs.
        ThisBVH              lbvh_T;
        ThisBVH::QueryBuffer candidate_AllP_AllT_pairs;

        Buffer<IndexT> selected_PT_count;
        Buffer<IndexT> selected_EE_count;
        Buffer<IndexT> selected_PE_count;
        Buffer<IndexT> selected_PP_count;

        Buffer<Vector4i> temp_PTs;
        Buffer<Vector4i> temp_EEs;
        Buffer<Vector3i> temp_PEs;
        Buffer<Vector2i> temp_PPs;

        Buffer<Vector4i> PTs;
        Buffer<Vector4i> EEs;
        Buffer<Vector3i> PEs;
        Buffer<Vector2i> PPs;


        /****************************************************
        *                   CCD TOI
        ****************************************************/

        Buffer<Float> tois;  // PP, PE, PT, EE
    };

  private:
    Impl m_impl;

    virtual void do_build(BuildInfo& info) override final;
    virtual void do_detect(DetectInfo& info) override final;
    virtual void do_filter_active(FilterActiveInfo& info) override final;
    virtual void do_filter_toi(FilterTOIInfo& info) override final;
};
}  // namespace uipc::backend::luisa
