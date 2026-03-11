#pragma once
#include "type_define.h"

namespace uipc::backend::luisa::distance
{
LC_GPU_CALLABLE void point_triangle_distance2(const Vector3& p,
                                              const Vector3& t0,
                                              const Vector3& t1,
                                              const Vector3& t2,
                                              Float& dist2);

LC_GPU_CALLABLE void point_triangle_distance2_gradient(const Vector3& p,
                                                       const Vector3& t0,
                                                       const Vector3& t1,
                                                       const Vector3& t2,
                                                       std::array<Float, 12>& grad);

LC_GPU_CALLABLE void point_triangle_distance2_hessian(const Vector3& p,
                                                      const Vector3& t0,
                                                      const Vector3& t1,
                                                      const Vector3& t2,
                                                      std::array<Float, 144>& Hessian);
}  // namespace uipc::backend::luisa::distance

#include "details/point_triangle.inl"
