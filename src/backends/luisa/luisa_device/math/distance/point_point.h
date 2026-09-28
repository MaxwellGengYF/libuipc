#pragma once
/********************************************************************
 * @file point_point.h
 * @brief T2.3 — point-point squared distance, gradient and Hessian.
 *
 * Ported from `src/backends/cuda/utils/distance/point_point.h` +
 * `details/point_point.inl` onto the generic numerics layer
 * (`math/vec.h`, `math/mat.h`): `Eigen::Vector<T,3>` -> `LVec<T,3>`,
 * `Eigen::Matrix<T,N,N>` -> `LMat<T,N,N>`.
 *
 * As everywhere in `luisa_device/math`, ONE implementation is instantiated with
 * `T = double` (host) or `T = lcrt::Var<double>` (device); see cookbook §7.
 *********************************************************************/
#include <luisa_device/math/distance/distance_common.h>

namespace uipc::backend::luisa::luisa_device::math::distance
{
/// `|a - b|^2`.
template <typename T>
void point_point_distance2(const LVec<T, 3>& a, const LVec<T, 3>& b, T& dist2);

/// Gradient of `point_point_distance2` w.r.t. `(a, b)`, 6 components.
template <typename T>
void point_point_distance2_gradient(const LVec<T, 3>& a,
                                    const LVec<T, 3>& b,
                                    LVec<T, 6>&       grad);

/// Hessian of `point_point_distance2` w.r.t. `(a, b)`, 6x6.
template <typename T>
void point_point_distance2_hessian(const LVec<T, 3>& a,
                                   const LVec<T, 3>& b,
                                   LMat<T, 6, 6>&    Hessian);
}  // namespace uipc::backend::luisa::luisa_device::math::distance

#include "details/point_point.inl"
