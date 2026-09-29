#pragma once
/********************************************************************
 * @file affine_body_surface_reporter.h
 * @brief T6.1a — ABD simplicial-surface reporter (port of
 * src/backends/cuda/affine_body/affine_body_surface_reporter.h).
 *
 * Collects the *surface* vertices/edges/triangles of every ABD geometry, once per
 * instance, in global vertex numbering, and reports them to the
 * GlobalSimplicialSurfaceManager.  Pure host bookkeeping, so the port maps only
 * the element types (Vector2i -> LVector2i, Vector3i -> LVector3i; byte-identical).
 *
 * The class name keeps the reference's odd casing (`AffinebodySurfaceReporter`)
 * because it is part of the "identical public surface" contract.
 *********************************************************************/
#include <sim_system.h>
#include <global_geometry/simplicial_surface_reporter.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_vertex_reporter.h>

namespace uipc::backend::luisa
{
class AffinebodySurfaceReporter : public SimplicialSurfaceReporter
{
  public:
    using SimplicialSurfaceReporter::SimplicialSurfaceReporter;

    class Impl
    {
      public:
        void init(backend::WorldVisitor& world);

        void report_count(backend::WorldVisitor& world, SurfaceCountInfo& info);
        void report_attributes(backend::WorldVisitor& world, SurfaceAttributeInfo& info);

        AffineBodyDynamics*       affine_body_dynamics = nullptr;
        AffineBodyDynamics::Impl& abd() { return affine_body_dynamics->m_impl; }
        AffineBodyVertexReporter* affine_body_vertex_reporter = nullptr;

        vector<IndexT>   surf_vertices;
        vector<LVector2i> surf_edges;
        vector<LVector3i> surf_triangles;
    };

  protected:
    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(SurfaceInitInfo& info) override;
    virtual void do_report_count(SurfaceCountInfo& info) override;
    virtual void do_report_attributes(SurfaceAttributeInfo& info) override;

  private:
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
