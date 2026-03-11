#pragma once
#include <global_geometry/vertex_reporter.h>

namespace uipc::backend::luisa
{
class HalfPlane;
class HalfPlaneBodyReporter;

/**
 * @brief Vertex reporter for HalfPlane implicit geometry
 * 
 * Reports vertex counts, attributes, and displacements for half-plane implicit geometries
 * to the GlobalVertexManager.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 * Uses luisa::compute::Buffer for device memory management.
 */
class HalfPlaneVertexReporter final : public VertexReporter
{
  public:
    using VertexReporter::VertexReporter;

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
    virtual luisa::ulong get_uid() const noexcept override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
