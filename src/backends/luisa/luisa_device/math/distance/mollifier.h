#pragma once
/********************************************************************
 * @file mollifier.h
 * @brief T2.3 — edge-edge mollifier for codim-IPC.
 * Ported from `src/backends/cuda/utils/distance/edge_edge_mollifier.h` +
 * `details/edge_edge_mollifier.inl`.
 *
 * The reference header is named `edge_edge_mollifier.h`; this is its
 * `math/distance/` counterpart (`mollifier.h` keeps the name short, matching the
 * file set in the plan).
 *********************************************************************/
#include <luisa_device/math/distance/distance_common.h>

namespace uipc::backend::luisa::luisa_device::math::distance
{
/// Threshold below which the mollifier is considered disabled (the reference's
/// `EEMollifierDisabledThreshold`).
inline constexpr double EEMollifierDisabledThreshold = -1.0;

/// `eps_x` = the mollifier threshold derived from the rest configuration.
template <typename T>
void edge_edge_mollifier_threshold(const LVec<T, 3>& ea0_rest,
                                   const LVec<T, 3>& ea1_rest,
                                   const LVec<T, 3>& eb0_rest,
                                   const LVec<T, 3>& eb1_rest,
                                   double            coeff,
                                   T&                eps_x);

/// Whether the current configuration is inside the mollified (near-parallel)
/// region.
///
/// Host flavour of the pinned signature; the device flavour (returning
/// `bool_of_t<T>` instead of a host `bool` — a device predicate cannot be a
/// `bool`, RULE M2) is declared with its definition in `details/mollifier.inl`.
template <typename T>
    requires std::is_arithmetic_v<T>
bool need_mollify(const LVec<T, 3>& ea0,
                  const LVec<T, 3>& ea1,
                  const LVec<T, 3>& eb0,
                  const LVec<T, 3>& eb1,
                  T                 eps_x);

/// `|(ea1 - ea0) x (eb1 - eb0)|^2`.
template <typename T>
void edge_edge_cross_norm2(const LVec<T, 3>& ea0,
                           const LVec<T, 3>& ea1,
                           const LVec<T, 3>& eb0,
                           const LVec<T, 3>& eb1,
                           T&                result);

template <typename T>
void edge_edge_cross_norm2_gradient(const LVec<T, 3>& ea0,
                                    const LVec<T, 3>& ea1,
                                    const LVec<T, 3>& eb0,
                                    const LVec<T, 3>& eb1,
                                    LVec<T, 12>&      grad);

template <typename T>
void edge_edge_cross_norm2_hessian(const LVec<T, 3>& ea0,
                                   const LVec<T, 3>& ea1,
                                   const LVec<T, 3>& eb0,
                                   const LVec<T, 3>& eb1,
                                   LMat<T, 12, 12>&  Hessian);

/// The mollifier value `e` and its derivatives.
template <typename T>
void edge_edge_mollifier(const LVec<T, 3>& ea0,
                         const LVec<T, 3>& ea1,
                         const LVec<T, 3>& eb0,
                         const LVec<T, 3>& eb1,
                         T                 eps_x,
                         T&                e);

template <typename T>
void edge_edge_mollifier_gradient(const LVec<T, 3>& ea0,
                                  const LVec<T, 3>& ea1,
                                  const LVec<T, 3>& eb0,
                                  const LVec<T, 3>& eb1,
                                  T                 eps_x,
                                  LVec<T, 12>&      g);

template <typename T>
void edge_edge_mollifier_hessian(const LVec<T, 3>& ea0,
                                 const LVec<T, 3>& ea1,
                                 const LVec<T, 3>& eb0,
                                 const LVec<T, 3>& eb1,
                                 T                 eps_x,
                                 LMat<T, 12, 12>&  H);
}  // namespace uipc::backend::luisa::luisa_device::math::distance

#include "details/mollifier.inl"
