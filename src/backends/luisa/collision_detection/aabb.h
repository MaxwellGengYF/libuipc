#pragma once
/********************************************************************
 * @file aabb.h
 * @brief Axis-aligned bounding box: the reference's host type **plus** the device
 *        element type the ported BVHs need (T7.1a).
 *
 * Reference: src/backends/cuda/collision_detection/aabb.h (9 lines, host-only:
 * `using AABB = Eigen::AlignedBox<float, 3>;`).
 *
 * WHY THIS FILE GREW
 * ------------------
 * In the CUDA backend an `Eigen::AlignedBox<float,3>` is *also* a device type: nvcc
 * compiles the same header for both sides, so `cuda_tool::DeviceBuffer<AABB>` works.
 * An LC buffer element must be a *reflected aggregate*
 * (`detail::TypeDesc<T>` / `LUISA_STRUCT`), and Eigen's fixed-size types are not
 * aggregates (measured in T4.1: `is_aggregate_v<Eigen::Vector3d> == false`, cookbook
 * §7.16), so the port keeps the host `AABB` (unchanged, still `Eigen::AlignedBox`)
 * and adds a device element type with the **same bytes**:
 *
 * ---------------------------------------------------------------------------
 *   host (unchanged)                 device (new)
 * ---------------------------------------------------------------------------
 *   Eigen::AlignedBox<float, 3>      LAABB
 *     min_/max_ : Vector3f             min_/max_ : LVec<float,3> (std::array<float,3>)
 *     24 bytes, align 4                24 bytes, align 4
 * ---------------------------------------------------------------------------
 *
 * `LAABB` is deliberately *not* LC's own `float3`: `luisa::float3` is
 * `Vector<float,3>`, which is `alignas(16)` and therefore **16 bytes** (LC pads a
 * 3-vector to its alignment, `vector_alignment_v<T,3> == min(sizeof(T)*4,16)`), so a
 * `{float3 min, float3 max}` element would be 32 bytes instead of 24 and every
 * `LinearBVHNode` would grow by 8 bytes of padding. `lps::LVec<float,3>` is a plain
 * `std::array<float,3>` (the numerics layer's whole point: Eigen-identical bytes), so
 * `sizeof(LAABB) == sizeof(AABB)` and `LinearBVHNode` is 16 bytes, exactly like the
 * reference's (`{uint32 parent, left, right, object_idx}`), while
 * `StacklessBVH::Node {int32 lc, int32 escape, AABB bound}` is 4+4+24 = 32 bytes.
 *
 * DEVICE-VIEW CONVENTIONS
 * -----------------------
 *  * a kernel reads an element as `Expr<LAABB>` and materialises it with
 *    `load_aabb(e) -> LBox<Var<float>>`, exactly like `load_vec3`/`store_vec3`
 *    (type_define.h);
 *  * the generic box routines below are written **once**, over `LBox<T>` with
 *    `T = float`/`double` (host: the test oracles, the host-side bookkeeping) or
 *    `T = lcrt::Var<float>` (device: the kernels).  They are expression-only (no
 *    host control flow on device values, RULE M2) so both instantiations record the
 *    same arithmetic.  Boolean conjunctions use the **bitwise** `&`: LC's `&&` on
 *    scalars is rejected by a static_assert ("logical operator requires boolean
 *    vector operands; use the bitwise operators"), the same rule the ported kernels
 *    follow elsewhere;
 *  * the empty box is Eigen's `setEmpty()`: `min = +FLT_MAX`, `max = -FLT_MAX`
 *    (`ScalarTraits<float>::highest()/lowest()`), which is what makes
 *    `merge(default_aabb, a) == a` in the reference's `DeviceReduce` and what the
 *    scene-box kernels start from.
 *
 * ORDER MATTERS: the `LUISA_STRUCT` registrations must appear before the first use of
 * `Expr<LAABB>`/`Var<LAABB>` (an unregistered struct's `Expr` has no members — measured
 * C2039 in scratch/t71a_spike.cpp).
 *********************************************************************/
#include <type_define.h>

#include <Eigen/Geometry>

#include <luisa_device/math/vec.h>

#include <limits>

namespace uipc::backend::luisa
{
// float based AABB (the reference's host type, unchanged).
using AABB = Eigen::AlignedBox<float, 3>;

// ---------------------------------------------------------------------------
//  Device element types (see the header note)
// ---------------------------------------------------------------------------
/// Device float3: 12 bytes / align 4, the bytes of `Eigen::Vector3f`.
using LVector3f = lps::LVec<float, 3>;

/// The device AABB: the reference's `LinearBVHAABB`, as an LC buffer element.
struct LAABB
{
    LVector3f min;
    LVector3f max;
};

static_assert(sizeof(LVector3f) == 3u * sizeof(float),
              "the device float3 must be a packed std::array<float,3> (LC's own "
              "float3 is 16 B because of its alignas(16))");
static_assert(sizeof(LAABB) == sizeof(AABB),
              "LAABB must keep the host AABB's 24-byte layout "
              "(Eigen::AlignedBox<float,3> = {Vector3f min, Vector3f max})");
}// namespace uipc::backend::luisa

// ---------------------------------------------------------------------------
//  LuisaCompute struct registration (global scope, before first use)
// ---------------------------------------------------------------------------
LUISA_STRUCT(uipc::backend::luisa::LVector3f, data){};
LUISA_STRUCT(uipc::backend::luisa::LAABB, min, max){};

namespace uipc::backend::luisa
{
// ---------------------------------------------------------------------------
//  Generic box (host float/double, device Var<float>/Var<double>)
// ---------------------------------------------------------------------------
/// A 3D box of scalars -- the generic spelling of `Eigen::AlignedBox<T,3>`.
template <typename T>
struct LBox
{
    lps::LVec<T, 3> min;
    lps::LVec<T, 3> max;
};

/// Eigen's `AlignedBox::setEmpty()`: `min = highest()`, `max = lowest()`.
template <typename T>
[[nodiscard]] inline LBox<T> box_empty() noexcept
{
    constexpr float hi = std::numeric_limits<float>::max();
    constexpr float lo = -hi;
    LBox<T>         r;
    for(int k = 0; k < 3; ++k)
    {
        r.min(k) = static_cast<T>(hi);
        r.max(k) = static_cast<T>(lo);
    }
    return r;
}

/// `AlignedBox::extend(const AlignedBox&)`: component-wise min of the minima and
/// max of the maxima (Eigen does *not* special-case the empty box).
template <typename T>
[[nodiscard]] inline LBox<T> box_merged(const LBox<T>& a, const LBox<T>& b) noexcept
{
    LBox<T> r;
    for(int k = 0; k < 3; ++k)
    {
        r.min(k) = lps::min(a.min(k), b.min(k));
        r.max(k) = lps::max(a.max(k), b.max(k));
    }
    return r;
}

/// `AlignedBox::extend(const Vector3&)`: grow the box so that it contains `p`.
template <typename T>
[[nodiscard]] inline LBox<T> box_extend_point(const LBox<T>& a,
                                              const lps::LVec<T, 3>& p) noexcept
{
    LBox<T> r = a;
    for(int k = 0; k < 3; ++k)
    {
        r.min(k) = lps::min(a.min(k), p(k));
        r.max(k) = lps::max(a.max(k), p(k));
    }
    return r;
}

/// `AlignedBox::intersects`: `(min <= b.max).all() && (b.min <= max).all()`.
template <typename T>
[[nodiscard]] inline auto box_intersects(const LBox<T>& a, const LBox<T>& b) noexcept
{
    return ((a.min(0) <= b.max(0)) & (a.min(1) <= b.max(1)) & (a.min(2) <= b.max(2))
            & (b.min(0) <= a.max(0)) & (b.min(1) <= a.max(1)) & (b.min(2) <= a.max(2)));
}

/// `AlignedBox::contains(const Vector3&)`.
template <typename T>
[[nodiscard]] inline auto box_contains(const LBox<T>& a, const lps::LVec<T, 3>& p) noexcept
{
    return ((p(0) >= a.min(0)) & (p(1) >= a.min(1)) & (p(2) >= a.min(2)) & (p(0) <= a.max(0))
            & (p(1) <= a.max(1)) & (p(2) <= a.max(2)));
}

/// `AlignedBox::isEmpty()`: `(min > max).any()`.
template <typename T>
[[nodiscard]] inline auto box_is_empty(const LBox<T>& a) noexcept
{
    return ((a.min(0) > a.max(0)) | (a.min(1) > a.max(1)) | (a.min(2) > a.max(2)));
}

/// `AlignedBox::center()`: `(min + max) / 2` (in `T`'s precision, like Eigen).
template <typename T>
[[nodiscard]] inline lps::LVec<T, 3> box_center(const LBox<T>& a) noexcept
{
    lps::LVec<T, 3> r;
    for(int k = 0; k < 3; ++k)
        r(k) = (a.min(k) + a.max(k)) / static_cast<T>(2);
    return r;
}

/// `AlignedBox::sizes()`: `max - min`.
template <typename T>
[[nodiscard]] inline lps::LVec<T, 3> box_sizes(const LBox<T>& a) noexcept
{
    lps::LVec<T, 3> r;
    for(int k = 0; k < 3; ++k)
        r(k) = a.max(k) - a.min(k);
    return r;
}

/// `AlignedBox::volume()`: the product of the sizes (the reference uses it only in
/// its `volume() >= 0` kernel assertion, so the fold order is irrelevant).
template <typename T>
[[nodiscard]] inline auto box_volume(const LBox<T>& a) noexcept
{
    auto s = box_sizes(a);
    return ((s(0) * s(1)) * s(2));
}

// ---------------------------------------------------------------------------
//  Host <-> device conversion (identical bytes, unrelated types)
// ---------------------------------------------------------------------------
[[nodiscard]] inline LAABB to_luisa(const AABB& b) noexcept
{
    LAABB r;
    for(int k = 0; k < 3; ++k)
    {
        r.min(k) = b.min()[k];
        r.max(k) = b.max()[k];
    }
    return r;
}

[[nodiscard]] inline AABB to_eigen(const LAABB& b) noexcept
{
    AABB r;
    for(int k = 0; k < 3; ++k)
    {
        r.min()[k] = b.min(k);
        r.max()[k] = b.max(k);
    }
    return r;
}

/// The empty box as a device element -- Eigen's `AlignedBox::setEmpty()`
/// (`min = +FLT_MAX`, `max = -FLT_MAX`), used as the identity of the box union and as
/// the reference's `LinearBVHAABB default_aabb;` / `LinearBVHNode default_node` fill.
[[nodiscard]] inline LAABB empty_laabb() noexcept
{
    LAABB r;
    for(int k = 0; k < 3; ++k)
    {
        r.min(k) = std::numeric_limits<float>::max();
        r.max(k) = -std::numeric_limits<float>::max();
    }
    return r;
}

/// Host-side view of a device element as the generic box (oracle arithmetic).
[[nodiscard]] inline LBox<float> to_box(const LAABB& b) noexcept
{
    return LBox<float>{b.min, b.max};
}

// ---------------------------------------------------------------------------
//  Kernel-side element access (the load_vec3/store_vec3 idiom)
// ---------------------------------------------------------------------------
/// Materialise a device AABB element as a mutable box of DSL scalars.
[[nodiscard]] inline LBox<lcrt::Var<float>> load_aabb(const lcrt::Expr<LAABB>& e) noexcept
{
    LBox<lcrt::Var<float>> r;
    for(int k = 0; k < 3; ++k)
    {
        r.min(k) = e.min.data[k];
        r.max(k) = e.max.data[k];
    }
    return r;
}

/// Store a box of DSL scalars into a device AABB element (a kernel-local Var).
inline void store_aabb(lcrt::Var<LAABB>& dst, const LBox<lcrt::Var<float>>& box) noexcept
{
    for(int k = 0; k < 3; ++k)
    {
        dst.min.data[k] = box.min(k);
        dst.max.data[k] = box.max(k);
    }
}
}// namespace uipc::backend::luisa
