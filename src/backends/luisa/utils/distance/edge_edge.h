#pragma once
#include "type_define.h"

namespace uipc::backend::luisa::distance
{
LC_GPU_CALLABLE void edge_edge_distance2(const Vector3& ea0,
                                         const Vector3& ea1,
                                         const Vector3& eb0,
                                         const Vector3& eb1,
                                         Float& dist2);

LC_GPU_CALLABLE void edge_edge_distance2_gradient(const Vector3& ea0,
                                                  const Vector3& ea1,
                                                  const Vector3& eb0,
                                                  const Vector3& eb1,
                                                  std::array<Float, 12>& grad);

LC_GPU_CALLABLE void edge_edge_distance2_hessian(const Vector3& ea0,
                                                 const Vector3& ea1,
                                                 const Vector3& eb0,
                                                 const Vector3& eb1,
                                                 std::array<Float, 144>& Hessian);
}  // namespace uipc::backend::luisa::distance

#include "details/edge_edge.inl"
