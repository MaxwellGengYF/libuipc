#pragma once
/********************************************************************
 * @file affine_body_state_accessor_feature.h
 * @brief T6.1a — the ABD state-accessor feature overrider (port of
 * src/backends/cuda/affine_body/affine_body_state_accessor_feature.h).
 *
 * Bridges `core::AffineBodyStateAccessorFeature` (the public
 * `world.features().find<AffineBodyStateAccessorFeature>()` API) to the ABD
 * system's device state: `body_id_to_q`/`body_id_to_q_v` for copy_from/copy_to,
 * plus the two externally-owned-buffer exports.
 *********************************************************************/
#include <type_define.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

namespace uipc::backend::luisa
{
class AffineBodyDynamics;
class AffineBodyVertexReporter;
class GlobalJointDofManager;

class AffineBodyStateAccessorFeatureOverrider final : public core::AffineBodyStateAccessorFeatureOverrider
{
  public:
    AffineBodyStateAccessorFeatureOverrider(AffineBodyDynamics& abd,
                                            AffineBodyVertexReporter& vertex_reporter,
                                            GlobalJointDofManager& joint_dof_manager);

    SizeT get_body_count() override;
    void  do_copy_from(const geometry::SimplicialComplex& state_geo) override;
    void  do_copy_to(geometry::SimplicialComplex& state_geo) override;

    void do_copy_transform_to(backend::BufferView buffer_view,
                              IndexT              body_offset,
                              SizeT               body_count) override;

    void do_copy_velocity_to(backend::BufferView buffer_view,
                             IndexT              body_offset,
                             SizeT               body_count) override;

  private:
    AffineBodyDynamics&      m_abd;
    AffineBodyVertexReporter& m_vertex_reporter;
    GlobalJointDofManager&   m_joint_dof_manager;

    mutable vector<Vector12> m_buffer;
};
}  // namespace uipc::backend::luisa
