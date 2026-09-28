#pragma once
/********************************************************************
 * @file point_triangle.h
 * @brief T2.3 — point-triangle squared distance, gradient and Hessian.
 * Ported from `src/backends/cuda/utils/distance/point_triangle.h` +
 * `details/point_triangle.inl` (closed-form; gradient/Hessian are the generated
 * scalar expressions, transcribed as-is so the arithmetic order — and therefore
 * the result — matches the reference).
 *********************************************************************/
#include <luisa_device/math/distance/distance_common.h>

namespace uipc::backend::luisa::luisa_device::math::distance
{
/// Squared distance from `p` to the triangle `(t0, t1, t2)`.
template <typename T>
void point_triangle_distance2(const LVec<T, 3>& p,
                              const LVec<T, 3>& t0,
                              const LVec<T, 3>& t1,
                              const LVec<T, 3>& t2,
                              T&                dist2);

/// Gradient w.r.t. `(p, t0, t1, t2)`, 12 components.
template <typename T>
void point_triangle_distance2_gradient(const LVec<T, 3>& p,
                                       const LVec<T, 3>& t0,
                                       const LVec<T, 3>& t1,
                                       const LVec<T, 3>& t2,
                                       LVec<T, 12>&      grad);

/// Hessian w.r.t. `(p, t0, t1, t2)`, 12x12.
template <typename T>
void point_triangle_distance2_hessian(const LVec<T, 3>& p,
                                      const LVec<T, 3>& t0,
                                      const LVec<T, 3>& t1,
                                      const LVec<T, 3>& t2,
                                      LMat<T, 12, 12>&  Hessian);
}  // namespace uipc::backend::luisa::luisa_device::math::distance

#include "details/point_triangle.inl"
