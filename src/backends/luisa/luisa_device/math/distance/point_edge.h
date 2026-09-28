#pragma once
/********************************************************************
 * @file point_edge.h
 * @brief T2.3 — point-edge squared distance, gradient and Hessian.
 * Ported from `src/backends/cuda/utils/distance/point_edge.h` +
 * `details/point_edge.inl` (closed-form; the gradient/Hessian are the generated
 * (Symbolic Math Toolbox) scalar expressions `g_PE3D`/`h_PE3D`, transcribed as-is
 * so the arithmetic order — and therefore the result — matches the reference).
 *********************************************************************/
#include <luisa_device/math/distance/distance_common.h>

namespace uipc::backend::luisa::luisa_device::math::distance
{
/// `|(e0 - p) x (e1 - p)|^2 / |e1 - e0|^2`.
template <typename T>
void point_edge_distance2(const LVec<T, 3>& p,
                          const LVec<T, 3>& e0,
                          const LVec<T, 3>& e1,
                          T&                dist2);

/// Gradient w.r.t. `(p, e0, e1)`, 9 components.
template <typename T>
void point_edge_distance2_gradient(const LVec<T, 3>& p,
                                   const LVec<T, 3>& e0,
                                   const LVec<T, 3>& e1,
                                   LVec<T, 9>&       grad);

/// Hessian w.r.t. `(p, e0, e1)`, 9x9.
template <typename T>
void point_edge_distance2_hessian(const LVec<T, 3>& p,
                                  const LVec<T, 3>& e0,
                                  const LVec<T, 3>& e1,
                                  LMat<T, 9, 9>&    Hessian);
}  // namespace uipc::backend::luisa::luisa_device::math::distance

#include "details/point_edge.inl"
