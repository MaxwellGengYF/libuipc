#pragma once
/********************************************************************
 * @file affine_body_vertex_reporter.h
 * @brief T6.1a — ABD vertex reporter (port of
 * src/backends/cuda/affine_body/affine_body_vertex_reporter.h).
 *
 * Reports the ABD vertices (positions, rest positions, body ids, contact ids,
 * d_hats) to the GlobalVertexManager and keeps the per-geometry
 * `global_vertex_offset` meta attribute in sync.  It is the reporter the
 * AffineBodyStateAccessorFeatureOverrider drives when a user writes the ABD state
 * back (T6.1a).
 *
 * Device mapping: `cuda_tool::CBufferView<Vector3>` -> `lud::LCBufferView<LVector3>`,
 * `cuda_tool::BufferView<Vector12>` -> `lud::LBufferView<LVector12>`.  The
 * `ABDJacobi`/`ABDJacobiDyadicMass` buffers keep their own aggregate element types
 * (affine_body/type_define.h).
 *********************************************************************/
#include <global_geometry/vertex_reporter.h>
#include <affine_body/affine_body_dynamics.h>

namespace uipc::backend::luisa
{
class AffineBodyBodyReporter;

class AffineBodyVertexReporter final : public VertexReporter
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

        AffineBodyDynamics*     affine_body_dynamics;
        AffineBodyBodyReporter* body_reporter = nullptr;
        AffineBodyDynamics::Impl& abd() { return affine_body_dynamics->m_impl; }

        bool require_update_attributes = false;
    };

    // Request to update vertex attributes before next simulation step
    void request_attribute_update() noexcept;

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_report_count(VertexCountInfo& info) override;
    virtual void do_report_attributes(VertexAttributeInfo& info) override;
    virtual void do_report_displacements(VertexDisplacementInfo& info) override;
    virtual U64 get_uid() const noexcept override;

  private:
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
