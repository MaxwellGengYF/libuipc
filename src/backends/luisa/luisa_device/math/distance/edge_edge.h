#pragma once
/********************************************************************
 * @file edge_edge.h
 * @brief T2.3 — edge-edge squared distance, gradient and Hessian.
 * Ported from `src/backends/cuda/utils/distance/edge_edge.h` +
 * `details/edge_edge.inl` (closed-form; gradient/Hessian are the generated
 * scalar expressions, transcribed as-is so the arithmetic order — and therefore
 * the result — matches the reference).
 *********************************************************************/
#include <luisa_device/math/distance/distance_common.h>

namespace uipc::backend::luisa::luisa_device::math::distance
{
/// Squared distance between the segments `(ea0, ea1)` and `(eb0, eb1)`.
template <typename T>
void edge_edge_distance2(const LVec<T, 3>& ea0,
                         const LVec<T, 3>& ea1,
                         const LVec<T, 3>& eb0,
                         const LVec<T, 3>& eb1,
                         T&                dist2);

/// Gradient w.r.t. `(ea0, ea1, eb0, eb1)`, 12 components.
template <typename T>
void edge_edge_distance2_gradient(const LVec<T, 3>& ea0,
                                  const LVec<T, 3>& ea1,
                                  const LVec<T, 3>& eb0,
                                  const LVec<T, 3>& eb1,
                                  LVec<T, 12>&      grad);

/// Hessian w.r.t. `(ea0, ea1, eb0, eb1)`, 12x12.
template <typename T>
void edge_edge_distance2_hessian(const LVec<T, 3>& ea0,
                                 const LVec<T, 3>& ea1,
                                 const LVec<T, 3>& eb0,
                                 const LVec<T, 3>& eb1,
                                 LMat<T, 12, 12>&  Hessian);
}  // namespace uipc::backend::luisa::luisa_device::math::distance

#include "details/edge_edge.inl"
