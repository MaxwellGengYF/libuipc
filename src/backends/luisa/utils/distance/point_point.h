#pragma once
#include "type_define.h"

namespace uipc::backend::luisa::distance
{
/**
 * @brief Compute squared distance between two 3D points
 * @param a First point
 * @param b Second point  
 * @param dist2 Output squared distance
 */
LC_GPU_CALLABLE void point_point_distance2(const Vector3& a,
                                           const Vector3& b,
                                           Float&         dist2);

/**
 * @brief Compute gradient of squared point-point distance
 * @param a First point
 * @param b Second point
 * @param grad Output gradient (6 elements: [d/da, d/db])
 */
LC_GPU_CALLABLE void point_point_distance2_gradient(const Vector3& a,
                                                    const Vector3& b,
                                                    float          grad[6]);

/**
 * @brief Compute Hessian of squared point-point distance
 * @param a First point
 * @param b Second point
 * @param Hessian Output Hessian (6x6 elements, column-major)
 */
LC_GPU_CALLABLE void point_point_distance2_hessian(const Vector3& a,
                                                   const Vector3& b,
                                                   float          Hessian[36]);
}  // namespace uipc::backend::luisa::distance

#include "details/point_point.inl"
