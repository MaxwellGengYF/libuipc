#pragma once
/********************************************************************
 * @file ccd.h
 * @brief T2.3 — continuous collision detection (conservative-advancement /
 * additive CCD) and its broad-phase filters.
 * Ported from `src/backends/cuda/utils/distance/ccd.h` + `details/ccd.inl`.
 *
 * Ref: https://github.com/ipc-sim/Codim-IPC/tree/main/Library/Math/Distance
 *
 * ## Control flow
 * The reference drives CCD with
 *     while (true) { if (--max_iter < 0) return true;
 *                    ...Newton step...
 *                    if (toc > toc_prev) return false; }
 * i.e. every termination test is *device* data, which cannot drive a host loop
 * inside an LC kernel (cookbook RULE M2).  The port therefore runs a
 * **fixed-iteration predicated loop**: `max_iter` (a *host* integer) bounds the
 * loop, and once a termination test fires the state is frozen so that further
 * iterations are no-ops (every termination test in the reference leaves a fixed
 * point behind, so the result is bit-comparable to the early-return version).
 * See `details/ccd.inl` for the exact equivalence argument.
 *
 * ## Result type
 * The reference's four `*_ccd` routines *return* a predicate ("a collision was
 * found"), which is device data on the device path.  The pinned `bool` signature is
 * kept for the host instantiation (`T` arithmetic); the device instantiation is a
 * second, constrained overload that returns `bool_of_t<T>` (a DSL `Expr<bool>`).
 * Parameter lists are identical.
 *********************************************************************/
#include <luisa_device/math/distance/distance_common.h>

/// Host flavour of the pinned `bool` signature; the device flavour (returning
/// `bool_of_t<T>` instead of a host `bool` — a device predicate cannot be a
/// `bool`, RULE M2) is declared with its definition in `details/ccd.inl`.

namespace uipc::backend::luisa::luisa_device::math::distance
{
/// AABB/Tolerance broad phase of a point against a segment.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_edge_cd_broadphase(const LVec<T, 3>& x0,
                              const LVec<T, 3>& x1,
                              const LVec<T, 3>& x2,
                              T                 dist);

/// 2D swept-AABB broad phase of a point against a segment.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_edge_ccd_broadphase(const LVec<T, 2>& p,
                               const LVec<T, 2>& e0,
                               const LVec<T, 2>& e1,
                               const LVec<T, 2>& dp,
                               const LVec<T, 2>& de0,
                               const LVec<T, 2>& de1,
                               T                 dist);

/// AABB/Tolerance broad phase of a point against a triangle.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_triangle_cd_broadphase(const LVec<T, 3>& p,
                                  const LVec<T, 3>& t0,
                                  const LVec<T, 3>& t1,
                                  const LVec<T, 3>& t2,
                                  T                 dist);

/// AABB/Tolerance broad phase of a segment against a segment.
template <typename T>
    requires std::is_arithmetic_v<T>
bool edge_edge_cd_broadphase(const LVec<T, 3>& ea0,
                             const LVec<T, 3>& ea1,
                             const LVec<T, 3>& eb0,
                             const LVec<T, 3>& eb1,
                             T                 dist);

/// 3D swept-AABB broad phase of a point against a triangle.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_triangle_ccd_broadphase(const LVec<T, 3>& p,
                                   const LVec<T, 3>& t0,
                                   const LVec<T, 3>& t1,
                                   const LVec<T, 3>& t2,
                                   const LVec<T, 3>& dp,
                                   const LVec<T, 3>& dt0,
                                   const LVec<T, 3>& dt1,
                                   const LVec<T, 3>& dt2,
                                   T                 dist);

/// 3D swept-AABB broad phase of a segment against a segment.
template <typename T>
    requires std::is_arithmetic_v<T>
bool edge_edge_ccd_broadphase(const LVec<T, 3>& ea0,
                              const LVec<T, 3>& ea1,
                              const LVec<T, 3>& eb0,
                              const LVec<T, 3>& eb1,
                              const LVec<T, 3>& dea0,
                              const LVec<T, 3>& dea1,
                              const LVec<T, 3>& deb0,
                              const LVec<T, 3>& deb1,
                              T                 dist);

/// 3D swept-AABB broad phase of a point against a segment.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_edge_ccd_broadphase(const LVec<T, 3>& p,
                               const LVec<T, 3>& e0,
                               const LVec<T, 3>& e1,
                               const LVec<T, 3>& dp,
                               const LVec<T, 3>& de0,
                               const LVec<T, 3>& de1,
                               T                 dist);

/// 3D swept-AABB broad phase of a point against a point.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_point_ccd_broadphase(const LVec<T, 3>& p0,
                                const LVec<T, 3>& p1,
                                const LVec<T, 3>& dp0,
                                const LVec<T, 3>& dp1,
                                T                 dist);

/// Additive CCD of a point against a triangle.  Returns true when a collision is
/// found within `max_iter` Newton iterations; `toc` is the time of contact.
///
/// The `requires` clauses split the pinned host signature from the device flavour
/// (whose result is a device predicate); the parameter list is the same in both.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_triangle_ccd(LVec<T, 3> p,
                        LVec<T, 3> t0,
                        LVec<T, 3> t1,
                        LVec<T, 3> t2,
                        LVec<T, 3> dp,
                        LVec<T, 3> dt0,
                        LVec<T, 3> dt1,
                        LVec<T, 3> dt2,
                        T          eta,
                        T          thickness,
                        int        max_iter,
                        T&         toc);

/// Additive CCD of a segment against a segment.
template <typename T>
    requires std::is_arithmetic_v<T>
bool edge_edge_ccd(LVec<T, 3> ea0,
                   LVec<T, 3> ea1,
                   LVec<T, 3> eb0,
                   LVec<T, 3> eb1,
                   LVec<T, 3> dea0,
                   LVec<T, 3> dea1,
                   LVec<T, 3> deb0,
                   LVec<T, 3> deb1,
                   T          eta,
                   T          thickness,
                   int        max_iter,
                   T&         toc);

/// Additive CCD of a point against a segment.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_edge_ccd(LVec<T, 3> p,
                    LVec<T, 3> e0,
                    LVec<T, 3> e1,
                    LVec<T, 3> dp,
                    LVec<T, 3> de0,
                    LVec<T, 3> de1,
                    T          eta,
                    T          thickness,
                    int        max_iter,
                    T&         toc);

/// Additive CCD of a point against a point.
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_point_ccd(LVec<T, 3> p0,
                     LVec<T, 3> p1,
                     LVec<T, 3> dp0,
                     LVec<T, 3> dp1,
                     T          eta,
                     T          thickness,
                     int        max_iter,
                     T&         toc);
}  // namespace uipc::backend::luisa::luisa_device::math::distance

#include "details/ccd.inl"
