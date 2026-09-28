#pragma once
/********************************************************************
 * @file half_plane_vertex_reporter.h
 * @brief Port of src/backends/cuda/implicit_geometry/half_plane_vertex_reporter.h
 *        (T4.2). Identical shape; the only change is the namespace.
 *********************************************************************/
#include <global_geometry/vertex_reporter.h>

namespace uipc::backend::luisa
{
class HalfPlane;
class HalfPlaneBodyReporter;
class HalfPlaneVertexReporter : public VertexReporter
{
  public:
    using VertexReporter::VertexReporter;

    class Impl;

    class Impl
    {
      public:
        HalfPlane*             half_plane    = nullptr;
        HalfPlaneBodyReporter* body_reporter = nullptr;


        void report_count(GlobalVertexManager::VertexCountInfo& info);
        void report_attributes(GlobalVertexManager::VertexAttributeInfo& info);
        void report_displacements(GlobalVertexManager::VertexDisplacementInfo& info);
    };

  protected:
    virtual void do_build(BuildInfo& info) override;
    void do_report_count(GlobalVertexManager::VertexCountInfo& vertex_count_info) override;
    void do_report_attributes(GlobalVertexManager::VertexAttributeInfo& vertex_attribute_info) override;
    void do_report_displacements(GlobalVertexManager::VertexDisplacementInfo& vertex_displacement_info) override;
    virtual U64 get_uid() const noexcept override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
