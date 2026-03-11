#pragma once
#include "type_define.h"

namespace uipc::backend::luisa::distance
{
LC_GPU_CALLABLE void point_edge_distance2(const Vector3& p,
                                          const Vector3& e0,
                                          const Vector3& e1,
                                          Float& dist2);

LC_GPU_CALLABLE void point_edge_distance2_gradient(const Vector3& p,
                                                   const Vector3& e0,
                                                   const Vector3& e1,
                                                   std::array<Float, 9>& grad);

LC_GPU_CALLABLE void point_edge_distance2_hessian(const Vector3& p,
                                                  const Vector3& e0,
                                                  const Vector3& e1,
                                                  std::array<Float, 81>& Hessian);
}  // namespace uipc::backend::luisa::distance

#include "details/point_edge.inl"
