#pragma once
#include <global_geometry/body_reporter.h>

namespace uipc::backend::luisa
{
// Forward declaration for HalfPlane
// This will be defined in half_plane.h when fully implemented
class HalfPlane;

/**
 * @brief Body reporter for HalfPlane implicit geometry
 * 
 * Reports body counts and attributes for half-plane implicit geometries to the GlobalBodyManager.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 * Uses luisa::compute::Buffer for device memory management.
 */
class HalfPlaneBodyReporter final : public BodyReporter
{
  public:
    using BodyReporter::BodyReporter;

    class Impl
    {
      public:
        void report_count(BodyCountInfo& info);
        void report_attributes(BodyAttributeInfo& info);

        HalfPlane* half_plane = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(InitInfo& info) override;
    void         do_report_count(BodyCountInfo& info) override;
    void         do_report_attributes(BodyAttributeInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
