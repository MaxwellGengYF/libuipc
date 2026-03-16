#pragma once
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/stackless_bvh.h>
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
        void detect(DetectInfo& info, WorldVisitor& world);
        void filter_active(FilterActiveInfo& info, WorldVisitor& world);
        void filter_toi(FilterTOIInfo& info, WorldVisitor& world);

        /****************************************************
        *                   Broad Phase
        ****************************************************/

        Buffer<AABB> codim_point_aabbs;
        Buffer<AABB> point_aabbs;
        Buffer<AABB> edge_aabbs;
        Buffer<AABB> triangle_aabbs;

        using ThisBVH = StacklessBVH;

        // Use unique_ptr for lazy initialization (StacklessBVH requires Device& in constructor)
        std::unique_ptr<ThisBVH> lbvh_CodimP;
        std::unique_ptr<ThisBVH> lbvh_E;
        std::unique_ptr<ThisBVH> lbvh_T;

        // CodimP count always less or equal to AllP count.
        ThisBVH::QueryBuffer candidate_AllP_CodimP_pairs{nullptr};

        // Used to detect CodimP-AllE, and AllE-AllE pairs.
        ThisBVH::QueryBuffer candidate_CodimP_AllE_pairs{nullptr};
        ThisBVH::QueryBuffer candidate_AllE_AllE_pairs{nullptr};

        // Used to detect AllP-AllT pairs.
        ThisBVH::QueryBuffer candidate_AllP_AllT_pairs{nullptr};

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
