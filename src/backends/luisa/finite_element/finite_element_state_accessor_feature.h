#pragma once
/********************************************************************
 * @file finite_element_state_accessor_feature.h
 * @brief T5.1 — the FEM state-accessor feature overrider (port of
 * src/backends/cuda/finite_element/finite_element_state_accessor_feature.h).
 *
 * Bridges `core::FiniteElementStateAccessorFeature` (the public
 * `world.features().find<FiniteElementStateAccessorFeature>()` API) to the FEM
 * method's device state: `xs`/`vs` for copy_from/copy_to, plus the two
 * externally-owned-buffer exports.
 *********************************************************************/
#include <type_define.h>
#include <uipc/core/finite_element_state_accessor_feature.h>

namespace uipc::backend::luisa
{
class FiniteElementMethod;
class FiniteElementVertexReporter;

class FiniteElementStateAccessorFeatureOverrider final : public core::FiniteElementStateAccessorFeatureOverrider
{
  public:
    FiniteElementStateAccessorFeatureOverrider(FiniteElementMethod& fem,
                                               FiniteElementVertexReporter& vertex_reporter);

    SizeT get_vertex_count() override;
    void  do_copy_from(const geometry::SimplicialComplex& state_geo) override;
    void  do_copy_to(geometry::SimplicialComplex& state_geo) override;

    void do_copy_position_to(backend::BufferView buffer_view,
                             IndexT vertex_offset,
                             SizeT vertex_count) override;
    void do_copy_velocity_to(backend::BufferView buffer_view,
                             IndexT vertex_offset,
                             SizeT vertex_count) override;

  private:
    FiniteElementMethod& m_fem;
    FiniteElementVertexReporter& m_vertex_reporter;
};
}  // namespace uipc::backend::luisa
