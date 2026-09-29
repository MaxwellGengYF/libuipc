#pragma once
/********************************************************************
 * @file type_define.h
 * @brief Core aliases for the LuisaCompute backend.
 *
 * Mirrors src/backends/cuda/type_define.h but for the DSL paradigm: there are no
 * launch qualifier macros (UIPC_HOST/DEVICE/GLOBAL) because kernels are built
 * with device.compile<1>, and the device utility layer is luisa_device
 * (the LuisaCompute analogue of cuda_tool).
 *
 * Pulled into host (.cpp) translation units, so it must stay lightweight.
 *
 * ---------------------------------------------------------------------------
 * Device element types for vector-valued attributes (T4.1)
 * ---------------------------------------------------------------------------
 * The reference backend stores a `Vector3` (Eigen::Vector3d) attribute as a
 * `cuda_tool::DeviceBuffer<Vector3>`; the mechanical mapping "T unchanged" does
 * NOT survive the port, because an LC buffer element type must be *reflected* by
 * LuisaCompute and Eigen's fixed-size types are not aggregates: LC's
 * `detail::TypeDesc<T>` primary template builds the struct description from
 * structured bindings and therefore `static_assert(std::is_aggregate_v<T>)`
 * (luisa/ast/member_reflect.inl.h). `device.create_buffer<Eigen::Vector3d>()`
 * accordingly does not compile (verified with scratch/t41_probe1.cpp:
 * is_aggregate_v<Vector3> == false).
 *
 * The ported backend therefore stores the vector-valued attributes as the
 * numerics layer's `lps::LVec<T,N>` -- a plain `std::array<T,N>` with the *same*
 * bytes as the Eigen alias (LVec<double,3> == 24 B == Eigen::Vector3d) -- and
 * registers the four instantiations the geometry managers need as LC struct
 * types (below). Order matters in this header:
 *   1. the aliases (the registration macro takes the type as its first argument,
 *      and an unparenthesised comma inside `LVec<T, N>` would split it),
 *   2. the `LUISA_STRUCT` registrations (a specialisation must be declared
 *      before its first use, and the macro must be invoked at *global* scope:
 *      inside `uipc::backend::luisa` the name `luisa` would resolve to the
 *      enclosing backend namespace, see luisa_device/compat.h),
 *   3. the kernel-side access helpers that use `Expr<LVector3>`.
 *
 * Consequences for ported code:
 *  - `LBufferView<LVector3> positions()` replaces
 *    `cuda_tool::CBufferView<Vector3>`; both sides of a copy must use the same
 *    element type, so a reporter converts the host Eigen data with
 *    `to_luisa(Vector3)` / `to_eigen(LVector3)` (identical bytes, unrelated
 *    types: the explicit conversion is well defined, a raw cast would not be).
 *  - Inside a kernel an element is read as `Expr<LVector3>`/written as
 *    `Var<LVector3>`; its components are `element.data[i]`, and
 *    `load_vec3()/store_vec3()` materialise it as an `lps::LVec<Var<Float>,3>`
 *    so the numerics layer can be used on it.
 *********************************************************************/
#include <uipc/common/type_define.h>
#include <vector>

#include <luisa_device/luisa_device.h>
#include <luisa_device/math/vec.h>
// T5.1: this header now names LMat<Float,3,3> (LMatrix3x3), so the matrix header
// must be complete here -- vec.h only forward-declares LMat.
#include <luisa_device/math/mat.h>

namespace uipc::backend::luisa
{
using Float    = uipc::Float;
using IndexT   = uipc::IndexT;
using SizeT    = uipc::SizeT;
using Vector2  = uipc::Vector2;
using Vector3  = uipc::Vector3;
using Vector4  = uipc::Vector4;
using Vector6  = uipc::Vector6;
using Vector9  = uipc::Vector9;
using Vector12 = uipc::Vector12;
using Matrix2x2 = uipc::Matrix2x2;
using Matrix3x3 = uipc::Matrix3x3;
using Matrix4x4 = uipc::Matrix4x4;
using Matrix6x6 = uipc::Matrix6x6;
using Matrix9x9 = uipc::Matrix9x9;
using Matrix12x12 = uipc::Matrix12x12;

// Device utility layer (short name used by ported sources).
namespace lud = luisa_device;
namespace lc  = luisa_device;

// Numerics layer (short name used by ported device code).
namespace lps = luisa_device::math;

// LuisaCompute's engine namespace. `::luisa` is the *global* one: inside
// `uipc::backend::luisa` an unqualified `luisa` resolves to the enclosing backend
// namespace instead (luisa_device/compat.h). NOTE: never write
// `LVec<::luisa::...>` -- the `<:` digraph would be tokenised as `[`.
namespace lcrt = ::luisa::compute;

// ---------------------------------------------------------------------------
// Device element types (the ported backend's device ABI for Eigen vectors)
// ---------------------------------------------------------------------------
using LVector2  = lps::LVec<Float, 2>;
using LVector3  = lps::LVec<Float, 3>;
using LVector2i = lps::LVec<IndexT, 2>;
using LVector3i = lps::LVec<IndexT, 3>;
// T5.1 additions (the FEM method's element buffers): `tets` is a Vector4i
// attribute and `Dm3x3_invs` a Matrix3x3 attribute.
using LVector4i  = lps::LVec<IndexT, 4>;
using LMatrix3x3 = lps::LMat<Float, 3, 3>;
// T5.2 (batch b): the shell membrane constitutions store a per-triangle 2x2 rest
// metric inverse (`inv_B_matrices`), exactly like the reference's
// `DeviceBuffer<Matrix2x2>`.
using LMatrix2x2 = lps::LMat<Float, 2, 2>;
// T6.3b1: the ABD joints store their per-joint rest frames as one 6-vector per
// quantity (`rest_cs` / `rest_ts` / `rest_ns` / `rest_bs` -- the two bodies'
// body-local attachment points / x- / y- / z-frame directions), i.e. the
// reference's `DeviceBuffer<Vector6>`.
using LVector6 = lps::LVec<Float, 6>;

static_assert(sizeof(LVector2) == sizeof(Vector2));
static_assert(sizeof(LVector3) == sizeof(Vector3));
static_assert(sizeof(LVector2i) == sizeof(Vector2i));
static_assert(sizeof(LVector3i) == sizeof(Vector3i));
static_assert(alignof(LVector3) == alignof(Vector3));
static_assert(sizeof(LVector4i) == sizeof(Vector4i));
static_assert(sizeof(LVector6) == sizeof(Vector6));
static_assert(alignof(LVector6) == alignof(Float));
// LMat's storage is a plain column-major std::array<T,R*C>, i.e. the same bytes
// as Eigen's Matrix<double,3,3> (72 B) -- which is what makes the FEM dump files
// portable between the two backends.
static_assert(sizeof(LMatrix3x3) == sizeof(Matrix3x3));
static_assert(alignof(LMatrix3x3) == alignof(Matrix3x3));
static_assert(sizeof(LMatrix2x2) == sizeof(Matrix2x2));
// Eigen over-aligns fixed-size types whose byte size is a multiple of 16
// (EIGEN_MAX_ALIGN_BYTES), so `Matrix2x2` (32 B) is alignas(16) while LMatrix2x2 is
// naturally aligned to its four doubles.  The host<->device conversion is
// element-wise, so only the size and the storage order have to agree -- unlike
// LMatrix3x3, whose 72 B is *not* a multiple of 16 and therefore matches Eigen's
// alignment exactly.
static_assert(alignof(LMatrix2x2) == alignof(Float));

// Host <-> device element conversion (identical layout, unrelated types).
[[nodiscard]] inline LVector2 to_luisa(const Vector2& v) noexcept
{
    return LVector2{v.x(), v.y()};
}
[[nodiscard]] inline LVector3 to_luisa(const Vector3& v) noexcept
{
    return LVector3{v.x(), v.y(), v.z()};
}
[[nodiscard]] inline LVector2i to_luisa(const Vector2i& v) noexcept
{
    return LVector2i{v.x(), v.y()};
}
[[nodiscard]] inline LVector3i to_luisa(const Vector3i& v) noexcept
{
    return LVector3i{v.x(), v.y(), v.z()};
}
[[nodiscard]] inline Vector2 to_eigen(const LVector2& v) noexcept
{
    return Vector2{v(0), v(1)};
}
[[nodiscard]] inline Vector3 to_eigen(const LVector3& v) noexcept
{
    return Vector3{v(0), v(1), v(2)};
}
[[nodiscard]] inline Vector2i to_eigen(const LVector2i& v) noexcept
{
    return Vector2i{v(0), v(1)};
}
[[nodiscard]] inline Vector3i to_eigen(const LVector3i& v) noexcept
{
    return Vector3i{v(0), v(1), v(2)};
}
[[nodiscard]] inline LVector4i to_luisa(const Vector4i& v) noexcept
{
    return LVector4i{v(0), v(1), v(2), v(3)};
}
[[nodiscard]] inline Vector4i to_eigen(const LVector4i& v) noexcept
{
    return Vector4i{v(0), v(1), v(2), v(3)};
}
// T6.3b1: the ABD joints' per-joint 6-vectors (two 3-vectors side by side:
// [first | second], the reference's `rest_*s.segment<3>(0)` / `.segment<3>(3)`).
[[nodiscard]] inline LVector6 to_luisa(const Vector6& v) noexcept
{
    return LVector6{v(0), v(1), v(2), v(3), v(4), v(5)};
}
[[nodiscard]] inline Vector6 to_eigen(const LVector6& v) noexcept
{
    return Vector6{v(0), v(1), v(2), v(3), v(4), v(5)};
}
[[nodiscard]] inline LMatrix3x3 to_luisa(const Matrix3x3& m) noexcept
{
    LMatrix3x3 r;
    for(int c = 0; c < 3; ++c)
        for(int i = 0; i < 3; ++i)
            r(i, c) = m(i, c);
    return r;
}
[[nodiscard]] inline Matrix3x3 to_eigen(const LMatrix3x3& m) noexcept
{
    Matrix3x3 r;
    for(int c = 0; c < 3; ++c)
        for(int i = 0; i < 3; ++i)
            r(i, c) = m(i, c);
    return r;
}
[[nodiscard]] inline LMatrix2x2 to_luisa(const Matrix2x2& m) noexcept
{
    LMatrix2x2 r;
    for(int c = 0; c < 2; ++c)
        for(int i = 0; i < 2; ++i)
            r(i, c) = m(i, c);
    return r;
}
[[nodiscard]] inline Matrix2x2 to_eigen(const LMatrix2x2& m) noexcept
{
    Matrix2x2 r;
    for(int c = 0; c < 2; ++c)
        for(int i = 0; i < 2; ++i)
            r(i, c) = m(i, c);
    return r;
}

/// Convert a host `vector<EigenType>` to the matching device element buffer
/// (`to_luisa(v)` element-wise; used by the constitutions' `do_init`, which reads
/// Eigen-typed attributes and uploads the corresponding `L*` element type).
template <typename Dst, typename SrcContainer>
void copy_host_to_device(lud::LDeviceBuffer<Dst>& dst, const SrcContainer& src)
{
    std::vector<Dst> host;
    host.reserve(src.size());
    for(const auto& v : src)
        host.push_back(to_luisa(v));
    dst.resize(host.size());
    if(!host.empty())
        // NOTE: the *parenthesised* form is required here -- `span` is a
        // using-declared class template (`using std::span;` in uipc/common/span.h),
        // and MSVC does not parse `span<const Dst>{host}` as a type-id in this
        // dependent context (measured C2760/C3878: "unexpected '{'; expected ')'").
        dst.view().copy_from(span<const Dst>(host));
}
}// namespace uipc::backend::luisa

// ---------------------------------------------------------------------------
// LuisaCompute struct registration of the element types
//
// `LUISA_STRUCT` has to be invoked at global scope and ends with
// `: luisa_compute_extension<S>`, so the braces+semicolon are part of the
// invocation (LC's own tests spell it `LUISA_STRUCT(S, a, b) {};`).
// ---------------------------------------------------------------------------
LUISA_STRUCT(uipc::backend::luisa::LVector2, data){};
LUISA_STRUCT(uipc::backend::luisa::LVector3, data){};
LUISA_STRUCT(uipc::backend::luisa::LVector2i, data){};
LUISA_STRUCT(uipc::backend::luisa::LVector3i, data){};
// T5.1: the FEM method's `tets` and `Dm3x3_invs` element types.
LUISA_STRUCT(uipc::backend::luisa::LVector4i, data){};
// T6.3b1: the ABD joints' per-joint rest-frame 6-vectors.
LUISA_STRUCT(uipc::backend::luisa::LVector6, data){};
LUISA_STRUCT(uipc::backend::luisa::LMatrix3x3, data){};
// T5.2 (batch b): the shell membrane constitutions' 2x2 per-triangle inverse of
// the rest shape matrix.
LUISA_STRUCT(uipc::backend::luisa::LMatrix2x2, data){};

namespace uipc::backend::luisa
{
// ---------------------------------------------------------------------------
// Kernel-side element access helpers
// ---------------------------------------------------------------------------
/// Materialise a device vector element as a mutable DSL vector (numerics layer).
[[nodiscard]] inline lps::LVec<lcrt::Var<Float>, 3> load_vec3(
    const lcrt::Expr<LVector3>& e) noexcept
{
    lps::LVec<lcrt::Var<Float>, 3> v;
    v(0) = e.data[0];
    v(1) = e.data[1];
    v(2) = e.data[2];
    return v;
}

[[nodiscard]] inline lps::LVec<lcrt::Var<Float>, 2> load_vec2(
    const lcrt::Expr<LVector2>& e) noexcept
{
    lps::LVec<lcrt::Var<Float>, 2> v;
    v(0) = e.data[0];
    v(1) = e.data[1];
    return v;
}

[[nodiscard]] inline lps::LVec<lcrt::Var<IndexT>, 2> load_vec2i(
    const lcrt::Expr<LVector2i>& e) noexcept
{
    lps::LVec<lcrt::Var<IndexT>, 2> v;
    v(0) = e.data[0];
    v(1) = e.data[1];
    return v;
}

[[nodiscard]] inline lps::LVec<lcrt::Var<IndexT>, 3> load_vec3i(
    const lcrt::Expr<LVector3i>& e) noexcept
{
    lps::LVec<lcrt::Var<IndexT>, 3> v;
    v(0) = e.data[0];
    v(1) = e.data[1];
    v(2) = e.data[2];
    return v;
}

/// T6.3b1: the ABD joints' per-joint rest-frame 6-vector (two 3-vectors).
[[nodiscard]] inline lps::LVec<lcrt::Var<Float>, 6> load_vec6(
    const lcrt::Expr<LVector6>& e) noexcept
{
    lps::LVec<lcrt::Var<Float>, 6> v;
    for(int i = 0; i < 6; ++i)
        v(i) = e.data[i];
    return v;
}

/// Store a DSL vector into a device vector element (a kernel-local Var).
inline void store_vec3(lcrt::Var<LVector3>& dst,
                       const lps::LVec<lcrt::Var<Float>, 3>& v) noexcept
{
    dst.data[0] = v(0);
    dst.data[1] = v(1);
    dst.data[2] = v(2);
}

inline void store_vec2i(lcrt::Var<LVector2i>& dst,
                        const lps::LVec<lcrt::Var<IndexT>, 2>& v) noexcept
{
    dst.data[0] = v(0);
    dst.data[1] = v(1);
}

inline void store_vec3i(lcrt::Var<LVector3i>& dst,
                        const lps::LVec<lcrt::Var<IndexT>, 3>& v) noexcept
{
    dst.data[0] = v(0);
    dst.data[1] = v(1);
    dst.data[2] = v(2);
}
[[nodiscard]] inline lps::LVec<lcrt::Var<IndexT>, 4> load_vec4i(
    const lcrt::Expr<LVector4i>& e) noexcept
{
    lps::LVec<lcrt::Var<IndexT>, 4> v;
    v(0) = e.data[0];
    v(1) = e.data[1];
    v(2) = e.data[2];
    v(3) = e.data[3];
    return v;
}
/// Materialise a device matrix element as a mutable DSL matrix (numerics layer).
[[nodiscard]] inline lps::LMat<lcrt::Var<Float>, 3, 3> load_mat3x3(
    const lcrt::Expr<LMatrix3x3>& e) noexcept
{
    lps::LMat<lcrt::Var<Float>, 3, 3> m;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            m(r, c) = e.data[c * 3 + r];  // column-major, LMat's own layout
    return m;
}
inline void store_mat3x3(lcrt::Var<LMatrix3x3>& dst,
                         const lps::LMat<lcrt::Var<Float>, 3, 3>& m) noexcept
{
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            dst.data[c * 3 + r] = m(r, c);
}

// T5.2 (batch b): the 2x2 flavour used by the shell membrane constitutions.
[[nodiscard]] inline lps::LMat<lcrt::Var<Float>, 2, 2> load_mat2x2(
    const lcrt::Expr<LMatrix2x2>& e) noexcept
{
    lps::LMat<lcrt::Var<Float>, 2, 2> m;
    for(int c = 0; c < 2; ++c)
        for(int r = 0; r < 2; ++r)
            m(r, c) = e.data[c * 2 + r];  // column-major, LMat's own layout
    return m;
}
inline void store_mat2x2(lcrt::Var<LMatrix2x2>& dst,
                         const lps::LMat<lcrt::Var<Float>, 2, 2>& m) noexcept
{
    for(int c = 0; c < 2; ++c)
        for(int r = 0; r < 2; ++r)
            dst.data[c * 2 + r] = m(r, c);
}
}// namespace uipc::backend::luisa
