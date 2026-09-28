#pragma once
/********************************************************************
 * @file finite_element_vertex_reporter.h
 * @brief T5.1 — FEM vertex reporter (port of
 * src/backends/cuda/finite_element/finite_element_vertex_reporter.h).
 *
 * Reports the FEM vertices (positions, rest positions, body ids, dimensions,
 * thicknesses, contact ids, d_hats) to the GlobalVertexManager, and keeps the
 * per-geometry `global_vertex_offset` meta attribute in sync. It is the reporter
 * the FiniteElementStateAccessorFeatureOverrider drives when a user writes the
 * FEM state back.
 *********************************************************************/
#include <global_geometry/vertex_reporter.h>
#include <finite_element/finite_element_method.h>

namespace uipc::backend::luisa
{
class FiniteElementBodyReporter;
class FiniteElementVertexReporter final : public VertexReporter
{
  public:
    using VertexReporter::VertexReporter;

    class Impl
    {
      public:
        void report_count(VertexCountInfo& info);
        void init_attributes(VertexAttributeInfo& info);
        void update_attributes(VertexAttributeInfo& info);
        void report_displacements(VertexDisplacementInfo& info);

        FiniteElementMethod* finite_element_method;
        FiniteElementMethod::Impl& fem()
        {
            return finite_element_method->m_impl;
        }
        FiniteElementBodyReporter* body_reporter = nullptr;

        bool require_update_attributes = false;
    };

    // Request to update vertex attributes before next simulation step
    void request_attribute_update() noexcept;

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_report_count(VertexCountInfo& info) override;
    virtual void do_report_attributes(VertexAttributeInfo& info) override;
    virtual void do_report_displacements(VertexDisplacementInfo& info) override;
    virtual U64  get_uid() const noexcept override;

  private:
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
