#pragma once
/********************************************************************
 * @file affine_body/type_define.h
 * @brief T6.1a — the ABD type layer (port of
 * src/backends/cuda/affine_body/type_define.h + the buffer element types the
 * reference declares in abd_jacobi_matrix.h).
 *
 * WHAT THIS HEADER OWNS
 * ---------------------
 * Affine Body Dynamics stores 12-dof aggregates (Vector12/Matrix12x12) and
 * per-vertex jacobians in device buffers.  An LC buffer element type must be
 * *reflected* by LuisaCompute (detail::TypeDesc<T> requires
 * std::is_aggregate_v<T>, cookbook §7.13 rule 1), and Eigen's fixed-size types
 * are not aggregates -- so every ABD buffer element is an aggregate over the
 * numerics layer's LVec/LMat (a plain std::array) with the *same bytes* as the
 * corresponding Eigen type.  The layout argument (all static_asserted below):
 *
 *   LVector12    = lps::LVec<double,12>    = std::array<double,12>   =  96 B
 *                  Eigen::Matrix<double,12,1>                         =  96 B
 *                  (Eigen over-aligns it to 16 B because 96 % 16 == 0; the port's
 *                   LVec is aligned to 8 B.  Only the *size* and the storage order
 *                   matter for a buffer: every host<->device crossing is an
 *                   element-wise copy through to_luisa()/to_eigen(), never a
 *                   reinterpret_cast.)
 *   LMatrix12x12 = lps::LMat<double,12,12> = std::array<double,144> = 1152 B
 *                  Eigen::Matrix<double,12,12>                        = 1152 B
 *                  (LMat is column-major, i.e. Eigen's default order.)
 *   LMatrix3x12  = 288 B == Eigen Matrix<double,3,12>; LMatrix12x3 the same.
 *
 *   ABDJacobi              = { LVector3 x_bar; }                       =  24 B
 *       the reference's ABDJacobi has exactly one (private) Vector3 member
 *       m_x_bar and no base class/virtuals, so its object layout is one Vector3
 *       at offset 0 -- the port's aggregate is byte-identical, and it is what the
 *       `vertex_id_to_J` device buffer holds.
 *   ABDJacobiDyadicMass    = { double m_mass; LVector3 m_mass_times_x_bar;
 *                              LMatrix3x3 m_mass_times_dyadic_x_bar; }  = 104 B
 *       member-for-member the reference's class (double at 0, Vector3 at 8,
 *       Matrix3x3 at 32; every member is 8-byte aligned, so there is no padding
 *       on either side).  Asserted in the T6.1a test against a mimicked
 *       Eigen-layout struct (sizeof == 104, offsets 0/8/32).
 *
 * ORDER MATTERS (measured C2908/C2766): LC's `LUISA_STRUCT(S, ...)` *derives* the
 * DSL access (Expr<S>/Ref<S>/Var<S> member accessors).  Any function that reads
 * `e.member` of an `Expr<S>` or writes `dst.member` of a `Var<S>` must therefore
 * come *after* the registration of `S` -- the same ordering the backend's global
 * type_define.h uses (aliases -> registrations -> kernel access helpers).
 *
 * The arithmetic the reference attaches to its two classes lives in
 * affine_body/abd_jacobi_matrix.{h,cpp} as *factored* routines templated on the
 * scalar, following the T5.1/T5.2 convention ("one generic implementation per
 * routine over the scalar type", cookbook §7.16 rule 1).
 *********************************************************************/
#include <type_define.h>  // the backend's global type layer (LVector3, LMatrix3x3, ...)
#include <uipc/common/type_define.h>

namespace uipc::backend::luisa
{
// ---------------------------------------------------------------------------
//  ABD's own Eigen aliases (the reference's affine_body/type_define.h)
// ---------------------------------------------------------------------------
using Matrix3x12  = uipc::Matrix<Float, 3, 12>;
using Matrix12x3  = uipc::Matrix<Float, 12, 3>;
using Matrix24x24 = uipc::Matrix<Float, 24, 24>;  // joint/constraint Hessians (T6.3/T6.4)

// ---------------------------------------------------------------------------
//  Device element types
// ---------------------------------------------------------------------------
using LVector12    = lps::LVec<Float, 12>;
using LMatrix12x12 = lps::LMat<Float, 12, 12>;
using LMatrix3x12  = lps::LMat<Float, 3, 12>;
using LMatrix12x3  = lps::LMat<Float, 12, 3>;

static_assert(sizeof(LVector12) == sizeof(Vector12));
static_assert(sizeof(LMatrix12x12) == sizeof(Matrix12x12));
static_assert(sizeof(LMatrix3x12) == sizeof(Matrix3x12));
static_assert(sizeof(LMatrix12x3) == sizeof(Matrix12x3));
// Eigen aligns a 96 B / 1152 B fixed-size type to 16 B, LMat/LVec to 8 B; only the
// byte size and the (column-major) storage order take part in the buffer ABI.
static_assert(alignof(LVector12) == alignof(Float));
static_assert(alignof(LMatrix12x12) == alignof(Float));

// ---------------------------------------------------------------------------
//  Generic (host `double` or DSL `Var<double>`) views of the two ABD aggregates
//
//  These are the operand types of the factored routines in
//  abd_jacobi_matrix.{h,cpp}; the buffer element types below are their
//  `S == Float` layout-identical siblings.
// ---------------------------------------------------------------------------
template <class S>
struct ABDJacobiT
{
    lps::LVec<S, 3> x_bar;
};

template <class S>
struct ABDJacobiDyadicMassT
{
    // clang-format off
    S                  m_mass;
    lps::LVec<S, 3>    m_mass_times_x_bar;
    lps::LMat<S, 3, 3> m_mass_times_dyadic_x_bar;
    // clang-format on
};

// ---------------------------------------------------------------------------
//  Buffer element types
// ---------------------------------------------------------------------------
///tex: $$ \mathbf{J}(\bar{\mathbf{x}}) $$ — the per-vertex affine jacobian.
struct ABDJacobi
{
    LVector3 x_bar{};
};

///tex: $$ \mathbf{J}^T\mathbf{M}_i\mathbf{J} $$ — the per-body (dyadic) mass.
struct ABDJacobiDyadicMass
{
    // clang-format off
    Float      m_mass = 0.0;                          //tex: $$ m $$
    LVector3   m_mass_times_x_bar{};                  //tex: $$ m\bar{\mathbf{x}} $$
    LMatrix3x3 m_mass_times_dyadic_x_bar{};           //tex: $$ m\bar{\mathbf{x}} \otimes \bar{\mathbf{x}} $$
    // clang-format on
};

static_assert(sizeof(ABDJacobi) == sizeof(Vector3));
static_assert(sizeof(ABDJacobiDyadicMass) == 8 + 24 + sizeof(Matrix3x3));
static_assert(offsetof(ABDJacobiDyadicMass, m_mass) == 0);
static_assert(offsetof(ABDJacobiDyadicMass, m_mass_times_x_bar) == 8);
static_assert(offsetof(ABDJacobiDyadicMass, m_mass_times_dyadic_x_bar) == 32);
}  // namespace uipc::backend::luisa

// ---------------------------------------------------------------------------
//  LuisaCompute struct registration (global scope, `{};` suffix -- §7.13)
// ---------------------------------------------------------------------------
LUISA_STRUCT(uipc::backend::luisa::LVector12, data){};
LUISA_STRUCT(uipc::backend::luisa::LMatrix12x12, data){};
LUISA_STRUCT(uipc::backend::luisa::LMatrix3x12, data){};
LUISA_STRUCT(uipc::backend::luisa::LMatrix12x3, data){};
LUISA_STRUCT(uipc::backend::luisa::ABDJacobi, x_bar){};
LUISA_STRUCT(uipc::backend::luisa::ABDJacobiDyadicMass,
             m_mass,
             m_mass_times_x_bar,
             m_mass_times_dyadic_x_bar){};

namespace uipc::backend::luisa
{
// ---------------------------------------------------------------------------
//  Host <-> device crossings (byte-identical layouts, unrelated types)
// ---------------------------------------------------------------------------
[[nodiscard]] inline LVector12 to_luisa(const Vector12& v) noexcept
{
    LVector12 r;
    for(int i = 0; i < 12; ++i)
        r(i) = v(i);
    return r;
}
[[nodiscard]] inline Vector12 to_eigen(const LVector12& v) noexcept
{
    Vector12 r;
    for(int i = 0; i < 12; ++i)
        r(i) = v(i);
    return r;
}
[[nodiscard]] inline LMatrix12x12 to_luisa(const Matrix12x12& m) noexcept
{
    LMatrix12x12 r;
    for(int c = 0; c < 12; ++c)
        for(int i = 0; i < 12; ++i)
            r(i, c) = m(i, c);
    return r;
}
[[nodiscard]] inline Matrix12x12 to_eigen(const LMatrix12x12& m) noexcept
{
    Matrix12x12 r;
    for(int c = 0; c < 12; ++c)
        for(int i = 0; i < 12; ++i)
            r(i, c) = m(i, c);
    return r;
}

/// Host rest position -> per-vertex jacobian element.  (NOTE: the name is not
/// `to_luisa(Vector3)` -- that overload already exists in the global type_define.h
/// with the LVector3 return type, and two functions may not differ by return type
/// alone, measured C2556.)
[[nodiscard]] inline ABDJacobi to_abd_jacobi(const Vector3& x_bar) noexcept
{
    return ABDJacobi{.x_bar = LVector3{x_bar.x(), x_bar.y(), x_bar.z()}};
}
[[nodiscard]] inline Vector3 to_eigen(const ABDJacobi& j) noexcept
{
    return Vector3{j.x_bar(0), j.x_bar(1), j.x_bar(2)};
}

/// Host triad -> per-body dyadic mass element (the reference's
/// `ABDJacobiDyadicMass::from_dyadic_mass`).
[[nodiscard]] inline ABDJacobiDyadicMass to_abd_dyadic_mass(Float            m,
                                                            const Vector3&   m_x_bar,
                                                            const Matrix3x3& m_x_bar_x_bar) noexcept
{
    ABDJacobiDyadicMass r;
    r.m_mass                    = m;
    r.m_mass_times_x_bar        = to_luisa(m_x_bar);
    r.m_mass_times_dyadic_x_bar = to_luisa(m_x_bar_x_bar);
    return r;
}

// Generic views (host or DSL) of the two element types.
[[nodiscard]] inline ABDJacobiT<Float> to_generic(const ABDJacobi& j) noexcept
{
    return ABDJacobiT<Float>{.x_bar = j.x_bar};
}
[[nodiscard]] inline ABDJacobiDyadicMassT<Float> to_generic(
    const ABDJacobiDyadicMass& m) noexcept
{
    return ABDJacobiDyadicMassT<Float>{.m_mass = m.m_mass,
                                       .m_mass_times_x_bar = m.m_mass_times_x_bar,
                                       .m_mass_times_dyadic_x_bar =
                                           m.m_mass_times_dyadic_x_bar};
}

// ---------------------------------------------------------------------------
//  Kernel-side element access (the Expr<T> of a buffer read / the Var<T> write)
// ---------------------------------------------------------------------------
[[nodiscard]] inline ABDJacobiT<lcrt::Var<Float>> load_jacobi(
    const lcrt::Expr<ABDJacobi>& e) noexcept
{
    return ABDJacobiT<lcrt::Var<Float>>{.x_bar = load_vec3(e.x_bar)};
}

inline void store_jacobi(lcrt::Var<ABDJacobi>& dst,
                         const ABDJacobiT<lcrt::Var<Float>>& j) noexcept
{
    store_vec3(dst.x_bar, j.x_bar);
}

[[nodiscard]] inline ABDJacobiDyadicMassT<lcrt::Var<Float>> load_dyadic_mass(
    const lcrt::Expr<ABDJacobiDyadicMass>& e) noexcept
{
    return ABDJacobiDyadicMassT<lcrt::Var<Float>>{
        .m_mass                    = e.m_mass,
        .m_mass_times_x_bar        = load_vec3(e.m_mass_times_x_bar),
        .m_mass_times_dyadic_x_bar = load_mat3x3(e.m_mass_times_dyadic_x_bar)};
}

inline void store_dyadic_mass(lcrt::Var<ABDJacobiDyadicMass>&                dst,
                              const ABDJacobiDyadicMassT<lcrt::Var<Float>>& m) noexcept
{
    dst.m_mass = m.m_mass;
    store_vec3(dst.m_mass_times_x_bar, m.m_mass_times_x_bar);
    store_mat3x3(dst.m_mass_times_dyadic_x_bar, m.m_mass_times_dyadic_x_bar);
}

[[nodiscard]] inline lps::LVec<lcrt::Var<Float>, 12> load_vec12(
    const lcrt::Expr<LVector12>& e) noexcept
{
    lps::LVec<lcrt::Var<Float>, 12> v;
    for(int i = 0; i < 12; ++i)
        v(i) = e.data[i];
    return v;
}

inline void store_vec12(lcrt::Var<LVector12>&                    dst,
                        const lps::LVec<lcrt::Var<Float>, 12>& v) noexcept
{
    for(int i = 0; i < 12; ++i)
        dst.data[i] = v(i);
}

[[nodiscard]] inline lps::LMat<lcrt::Var<Float>, 12, 12> load_mat12x12(
    const lcrt::Expr<LMatrix12x12>& e) noexcept
{
    lps::LMat<lcrt::Var<Float>, 12, 12> m;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            m(r, c) = e.data[c * 12 + r];  // column-major, LMat's own layout
    return m;
}

inline void store_mat12x12(lcrt::Var<LMatrix12x12>&                 dst,
                           const lps::LMat<lcrt::Var<Float>, 12, 12>& m) noexcept
{
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            dst.data[c * 12 + r] = m(r, c);
}
}  // namespace uipc::backend::luisa
