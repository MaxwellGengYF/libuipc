#pragma once
#include <affine_body/type_define.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

namespace uipc::backend::luisa
{
class AffineBodyDynamics;
class AffineBodyVertexReporter;

/**
 * @brief Affine body state accessor feature overrider for LuisaCompute backend
 * 
 * Provides access to affine body state data (q, q_v) for reading and writing.
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class AffineBodyStateAccessorFeatureOverrider final : public core::AffineBodyStateAccessorFeatureOverrider
{
  public:
    AffineBodyStateAccessorFeatureOverrider(AffineBodyDynamics& abd,
                                            AffineBodyVertexReporter& vertex_reporter);

    SizeT get_body_count() override;
    void  do_copy_from(const geometry::SimplicialComplex& state_geo) override;
    void  do_copy_to(geometry::SimplicialComplex& state_geo) override;

  private:
    AffineBodyDynamics&       m_abd;
    AffineBodyVertexReporter& m_vertex_reporter;
    mutable vector<Vector12>  m_buffer;
};
}  // namespace uipc::backend::luisa
