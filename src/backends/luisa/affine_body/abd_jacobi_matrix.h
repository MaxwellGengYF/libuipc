#pragma once
/********************************************************************
 * @file affine_body/abd_jacobi_matrix.h
 * @brief T6.1a — the ABD per-vertex jacobian and per-body dyadic mass
 * (port of src/backends/cuda/affine_body/abd_jacobi_matrix.h + its
 * details/abd_jacobi_matrix.inl).
 *
 * THE REFERENCE'S SHAPE vs THE PORT'S
 * ----------------------------------
 * The reference uses two `UIPC_GENERIC` classes whose methods are the arithmetic:
 *
 *   ABDJacobi           { Vector3 m_x_bar; }      // one member: the rest position
 *     operator*(Vector12)             -> Vector3      (point_from_affine / point_x)
 *     vec_x(Vector12)                 -> Vector3      (rotation/scaling only)
 *     to_mat()                        -> Matrix3x12
 *     ABDJacobiT (the .T() adapter):  operator*(Vector3) -> Vector12
 *     static JT_H_J(ABDJacobiT, Matrix3x3, ABDJacobi) -> Matrix12x12
 *
 *   ABDJacobiDyadicMass { double m_mass; Vector3 m_mass_times_x_bar;
 *                         Matrix3x3 m_mass_times_dyadic_x_bar; }
 *     operator*(Vector12) -> Vector12,  operator+=,  add_to(Matrix12x12&),
 *     to_mat(), mass(), mass_times_x_bar(), center_of_mass(),
 *     inertia_tensor_cm(), static atomic_add(dst, src)
 *
 * The port splits that surface in two, because a DSL call needs the *body* at the
 * call site while the buffer element type must stay a plain aggregate
 * (cookbook §7.16 rule 1):
 *
 *  1. the *element types* (affine_body/type_define.h) are aggregates with the
 *     reference's bytes; their host-facing convenience members are defined in
 *     abd_jacobi_matrix.cpp (they are Eigen-typed and are what the ported host
 *     code and the unit test call), and
 *  2. one *factored* routine per arithmetic operation, templated on the scalar
 *     (`S` = `double` on the host, `lcrt::Var<double>` in a kernel), defined in
 *     details/abd_jacobi_matrix.inl so a DSL call site can see the body.  The
 *     generic operand views are ABDJacobiT<S> / ABDJacobiDyadicMassT<S>
 *     (type_define.h), which is the port's replacement for "an Eigen type usable
 *     on both sides".
 *
 * Every routine below is a line-by-line re-spelling of the reference's body: the
 * same expressions in the same order, `Eigen::VectorBlock` -> LVec::segment<>,
 * `MatrixXd::block<r,c>` -> LMat::block<r,c> (same (rows, cols) order), and a
 * device-side `if` guard (none of these bodies has one) would become a select.
 *
 * DEVIATIONS (all reported in the T6.1a row / cookbook §7.21):
 *  - `ABDJacobiStack<N>`/`2`/`3`/`4` are ported as class templates over the
 *    generic views (`operator*`/`to_mat` are templates on the scalar).  They are
 *    *dead code in the reference* (grep: only their own declaration/inl mention
 *    them), and the port keeps them for surface parity; the unit test
 *    instantiates them so they cannot rot silently.
 *  - `#pragma unroll` (a CUDA pragma) is dropped: MSVC reports it as C4068, and
 *    the build must be warning-free.
 *  - `ABDJacobiDyadicMass::atomic_add` (a device atomic add on a struct in
 *    *global memory*) has no direct LC spelling: an AtomicRef of a LUISA_STRUCT
 *    exposes AtomicRef members, not a whole-struct fetch_add (cookbook §7.20
 *    rule 1).  The port therefore *declares* it for surface parity but does not
 *    define it; a batch that needs it (none does today -- grep-proven unused in
 *    the reference) must use the scalar-alias idiom of §7.20 rule 1:
 *    `view.native().as<double>()` + 13 scalar fetch_adds (1 mass + 3 x_bar + 9
 *    dyadic), exactly like FiniteElementExternalVertexForce.
 *********************************************************************/
#include <affine_body/type_define.h>

namespace uipc::backend::luisa
{
// ---------------------------------------------------------------------------
//  ABDJacobi arithmetic (the reference's ABDJacobi/ABDJacobiT members)
// ---------------------------------------------------------------------------
/// tex: $$ \mathbf{J}\mathbf{q} = \bar{\mathbf{x}} \cdot \mathbf{a} + \mathbf{t} $$
/// (reference: `ABDJacobi::operator*(const Vector12&)` / `point_x`)
template <class S>
[[nodiscard]] lps::LVec<S, 3> abd_jacobi_point(const ABDJacobiT<S>& j,
                                               const lps::LVec<S, 12>& q) noexcept;

/// Rotation/scaling only (reference: `ABDJacobi::vec_x`).
template <class S>
[[nodiscard]] lps::LVec<S, 3> abd_jacobi_vec_x(const ABDJacobiT<S>& j,
                                               const lps::LVec<S, 12>& q) noexcept;

/// tex: the reference's `ABDJacobi::to_mat()`.
template <class S>
[[nodiscard]] lps::LMat<S, 3, 12> abd_jacobi_to_mat(const ABDJacobiT<S>& j) noexcept;

/// tex: $$ \mathbf{J}^T\mathbf{g} $$ (reference: `ABDJacobiT::operator*(const Vector3&)`)
template <class S>
[[nodiscard]] lps::LVec<S, 12> abd_jacobi_T_mul(const ABDJacobiT<S>& j,
                                                const lps::LVec<S, 3>& g) noexcept;

/// tex: $$ \mathbf{J}^T\mathbf{H}\mathbf{J} $$ (reference:
/// `ABDJacobi::JT_H_J(const ABDJacobiT& lhs_J_T, const Matrix3x3& Hessian,
/// const ABDJacobi& rhs_J)`).  Both jacobis are passed as generic views: in the
/// reference the left one arrives through the `.T()` adapter, which is only a tag.
template <class S>
[[nodiscard]] lps::LMat<S, 12, 12> abd_jacobi_JT_H_J(const ABDJacobiT<S>& lhs_J_T,
                                                     const lps::LMat<S, 3, 3>& Hessian,
                                                     const ABDJacobiT<S>& rhs_J) noexcept;

// ---------------------------------------------------------------------------
//  ABDJacobiDyadicMass arithmetic
// ---------------------------------------------------------------------------
/// tex: $$ \mathbf{J}^T\mathbf{M}_i\mathbf{J}\,\mathbf{p} $$ (reference:
/// `ABDJacobiDyadicMass::operator*(const Vector12& p)`)
template <class S>
[[nodiscard]] lps::LVec<S, 12> abd_dyadic_mass_mul(
    const ABDJacobiDyadicMassT<S>& mJTJ, const lps::LVec<S, 12>& p) noexcept;

/// In-place accumulate (reference: `operator+=`).
template <class S>
void abd_dyadic_mass_add(ABDJacobiDyadicMassT<S>&       dst,
                         const ABDJacobiDyadicMassT<S>& rhs) noexcept;

/// tex: accumulate $$ \mathbf{J}^T\mathbf{M}_i\mathbf{J} $$ into `h`
/// (reference: `ABDJacobiDyadicMass::add_to(Matrix12x12&)`).
template <class S>
void abd_dyadic_mass_add_to(const ABDJacobiDyadicMassT<S>& mJTJ,
                            lps::LMat<S, 12, 12>&          h) noexcept;

/// tex: $$ \mathbf{J}^T\mathbf{M}_i\mathbf{J} $$ (reference: `to_mat()`).
template <class S>
[[nodiscard]] lps::LMat<S, 12, 12> abd_dyadic_mass_to_mat(
    const ABDJacobiDyadicMassT<S>& mJTJ) noexcept;

/// Center of mass in the reference frame (reference: `center_of_mass()`; the
/// reference guards it with `UIPC_KERNEL_ASSERT(m > 0)`).
template <class S>
[[nodiscard]] lps::LVec<S, 3> abd_dyadic_mass_center_of_mass(
    const ABDJacobiDyadicMassT<S>& mJTJ) noexcept;

/// Inertia tensor about the center of mass (reference: `inertia_tensor_cm()`).
/// Returns zero when the mass is non-positive (the reference's own guard).
template <class S>
[[nodiscard]] lps::LMat<S, 3, 3> abd_dyadic_mass_inertia_tensor_cm(
    const ABDJacobiDyadicMassT<S>& mJTJ) noexcept;

// ---------------------------------------------------------------------------
//  ABDJacobiStack<N> (port of the reference's class templates)
//
//  A stack of `N` per-vertex jacobians of one stencil: `operator*` maps a body
//  dof `q` to the N point positions, `to_mat()` builds the 3N x 12 matrix and the
//  `.T()` adapter maps back.  Dead code in the reference (kept for parity).
// ---------------------------------------------------------------------------
template <size_t N>
class ABDJacobiStack
{
  protected:
    const ABDJacobi* m_jacobis[N];

  public:
    class ABDJacobiStackT
    {
        const ABDJacobiStack& m_origin;

      public:
        explicit ABDJacobiStackT(const ABDJacobiStack& j)
            : m_origin(j)
        {
        }

        template <class S>
        /// The reference's `ABDJacobiStackT::operator*` returns **Vector12** (the
        /// transpose map `sum_i J_i^T g_i`); `ABDJacobiStack::operator*` below is
        /// the stack map `q -> (J_1 q, ..., J_N q)`.
        [[nodiscard]] lps::LVec<S, 12> operator*(
            const lps::LVec<S, static_cast<int>(3 * N)>& g) const noexcept;
    };

    template <class S>
    [[nodiscard]] lps::LVec<S, static_cast<int>(3 * N)> operator*(
        const lps::LVec<S, 12>& q) const noexcept;

    template <class S>
    [[nodiscard]] lps::LMat<S, static_cast<int>(3 * N), 12> to_mat() const noexcept;

    [[nodiscard]] ABDJacobiStackT T() const { return ABDJacobiStackT(*this); }
};

class ABDJacobiStack2 : public ABDJacobiStack<2>
{
  public:
    ABDJacobiStack2(const ABDJacobi& j1, const ABDJacobi& j2)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
    }
};

class ABDJacobiStack3 : public ABDJacobiStack<3>
{
  public:
    ABDJacobiStack3(const ABDJacobi& j1, const ABDJacobi& j2, const ABDJacobi& j3)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
        m_jacobis[2] = &j3;
    }
};

class ABDJacobiStack4 : public ABDJacobiStack<4>
{
  public:
    ABDJacobiStack4(const ABDJacobi& j1,
                    const ABDJacobi& j2,
                    const ABDJacobi& j3,
                    const ABDJacobi& j4)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
        m_jacobis[2] = &j3;
        m_jacobis[3] = &j4;
    }
};

// ---------------------------------------------------------------------------
//  Host-facing (Eigen) conveniences on the element types.
//
//  These are the reference's host-callable spellings; they forward to the
//  factored routines with S = Float, so host and device cannot drift apart.
// ---------------------------------------------------------------------------
/// Host convenience: the element types' `from_dyadic_mass` (reference:
/// `ABDJacobiDyadicMass::from_dyadic_mass`), used by the geometry build-up.
[[nodiscard]] ABDJacobiDyadicMass make_abd_dyadic_mass(Float          m,
                                                       const Vector3& m_x_bar,
                                                       const Matrix3x3&
                                                           m_x_bar_x_bar) noexcept;

/// Host convenience: the reference's `ABDJacobi::point_x(q)` in Eigen types.
[[nodiscard]] Vector3 abd_jacobi_point_x_eigen(const ABDJacobi& j,
                                               const Vector12&  q) noexcept;

/// Host convenience: the reference's `ABDJacobi::to_mat()` in Eigen types.
[[nodiscard]] Matrix3x12 abd_jacobi_to_mat_eigen(const ABDJacobi& j) noexcept;

/// Host convenience: the reference's `ABDJacobiDyadicMass::to_mat()`.
[[nodiscard]] Matrix12x12 abd_dyadic_mass_to_mat_eigen(
    const ABDJacobiDyadicMass& m) noexcept;

/// Host convenience: the reference's `ABDJacobiDyadicMass::mass()`.
[[nodiscard]] Float abd_dyadic_mass_mass(const ABDJacobiDyadicMass& m) noexcept;

/// Host convenience: the reference's `ABDJacobiDyadicMass::center_of_mass()`.
[[nodiscard]] Vector3 abd_dyadic_mass_center_of_mass_eigen(
    const ABDJacobiDyadicMass& m) noexcept;

/// Host convenience: the reference's `ABDJacobiDyadicMass::inertia_tensor_cm()`.
[[nodiscard]] Matrix3x3 abd_dyadic_mass_inertia_tensor_cm_eigen(
    const ABDJacobiDyadicMass& m) noexcept;
}  // namespace uipc::backend::luisa

#include "details/abd_jacobi_matrix.inl"
