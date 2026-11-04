#pragma once
/********************************************************************
 * @file friction_utils.h
 * @brief T8.1b — the friction tangential-basis / relative-displacement / Jacobian
 * helpers (port of src/backends/cuda/utils/friction_utils.h, 305 reference lines).
 *
 * WHY THIS FILE IS IN THIS BATCH.  The reference's frictional contact models
 * (`contact_models/codim_ipc_simplex_frictional_contact_function.h` and
 * `contact_models/ipc_vertex_half_plane_contact_function.h`) call
 * `friction::point_triangle_tangent_basis` / `*_closest_point` / `*_tan_rel_dx` /
 * `*_jacobi` (`*` = point_triangle / edge_edge / point_edge / point_point) and
 * `friction::friction_energy / friction_gradient / friction_hessian` (the latter three
 * live in `contact_models/codim_ipc_contact_function.h`).  None of them existed in the
 * port, so this header is the (unlisted but required) dependency the T8.1b batch adds —
 * the T7.2b "two additions to luisa_device/linear/algorithms.h" precedent.
 *
 * TYPE MAPPING (cookbook §3, §5): `Eigen::Vector<T,3>` -> `lps::LVec<T,3>`,
 * `Eigen::Matrix<T,R,C>` -> `lps::LMat<T,R,C>`, `Eigen::Matrix<T,N,1>` -> `lps::LVec<T,N>`
 * (a column vector), `Eigen::Matrix<T,1,N>` -> the row vector's `transpose()` spelling
 * (`LMat<T,1,N>`), `.col(i)`/`.row(i)`/`.block<>()`/`.segment<>()` are the layer's
 * strided views (mat.h/vec.h), `cuda_tool::eigen::inverse` -> `lps::inverse`,
 * `a.cross(b)` -> `lps::cross(a, b)`, `a.normalized()`/`a.squaredNorm()`/`a.dot(b)`
 * unchanged.  The functions stay `template <typename T>`: `T = double` evaluates on the
 * host (the [contact_model] test's oracle), `T = lcrt::Var<double>` records device IR.
 *
 * THE ONE DEVIATION, AND WHY IT IS EXACT.  The reference's
 * `point_point_tangent_basis` branches on a *device* predicate
 * (`if(xCross.squaredNorm() > yCross.squaredNorm())`), which cannot drive host control
 * flow inside an LC kernel (RULE M2, cookbook §7.3).  The port builds both candidate
 * bases and selects element-wise with `distance::detail::select_vec(f, t, pred)`
 * (`pred ? t : f`): a select passes its chosen operand through *unchanged*, so the
 * written `basis` is bit-identical to the reference's chosen branch, and the discarded
 * candidate is only extra work (the flagged-distance layer uses the same argument, see
 * luisa_device/math/distance/flagged.h's header).  Nothing else branches on device data
 * (the other `*_tangent_basis` bodies are straight-line arithmetic).
 *********************************************************************/
#include <type_define.h>
#include <luisa_device/math/distance/distance_common.h>
#include <luisa_device/math/matrix_ops.h>

namespace uipc::backend::luisa::friction
{
namespace ldist = luisa_device::math::distance;
using ldist::detail::select_scalar;
using ldist::detail::select_vec;
using lps::LMat;
using lps::LVec;

/// The reference's `tangent_rel_dx`: project `rel_dx` onto the two-column basis.
template <typename T>
inline void tangent_rel_dx(const LVec<T, 3>&    rel_dx,
                           const LMat<T, 3, 2>& basis,
                           LVec<T, 2>&          tan_rel_dx)
{
    tan_rel_dx = basis.transpose() * rel_dx;
}

// Point - Triangle

template <typename T>
inline void point_triangle_tangent_basis(const LVec<T, 3>& P,
                                         const LVec<T, 3>& T0,
                                         const LVec<T, 3>& T1,
                                         const LVec<T, 3>& T2,
                                         LMat<T, 3, 2>&    basis)
{
    LVec<T, 3> v12 = T1 - T0;
    basis.col(0)   = v12.normalized();
    basis.col(1)   = lps::cross(lps::cross(v12, T2 - T0), v12).normalized();
}

template <typename T>
inline void point_triangle_closest_point(const LVec<T, 3>& P,
                                         const LVec<T, 3>& T0,
                                         const LVec<T, 3>& T1,
                                         const LVec<T, 3>& T2,
                                         LVec<T, 2>&       beta)
{
    LMat<T, 2, 3> basis;
    basis.row(0)      = T1 - T0;
    basis.row(1)      = T2 - T0;
    LMat<T, 2, 2> BBT = basis * basis.transpose();
    LVec<T, 2>    rhs = basis * (P - T0);

    // Note: don't use Eigen solve, it doesn't work for CUDA
    beta = lps::inverse(BBT) * rhs;
}

template <typename T>
inline void point_triangle_rel_dx(const LVec<T, 3>& dP,
                                  const LVec<T, 3>& dT0,
                                  const LVec<T, 3>& dT1,
                                  const LVec<T, 3>& dT2,
                                  const LVec<T, 2>& beta,
                                  LVec<T, 3>&       rel_dx)
{
    rel_dx = dP - (dT0 + beta(0) * (dT1 - dT0) + beta(1) * (dT2 - dT0));
}

template <typename T>
inline void point_triangle_tan_rel_dx(const LVec<T, 3>&    dP,
                                      const LVec<T, 3>&    dT0,
                                      const LVec<T, 3>&    dT1,
                                      const LVec<T, 3>&    dT2,
                                      const LMat<T, 3, 2>& basis,
                                      const LVec<T, 2>&    beta,
                                      LVec<T, 2>&          tan_rel_dx)
{
    LVec<T, 3> rel_dx;
    point_triangle_rel_dx(dP, dT0, dT1, dT2, beta, rel_dx);
    tangent_rel_dx(rel_dx, basis, tan_rel_dx);
}

template <typename T>
inline void apply_point_triangle_jacobi(const LVec<T, 2>&    G2,
                                        const LMat<T, 3, 2>& basis,
                                        const LVec<T, 2>&    beta,
                                        LVec<T, 12>&         G12)
{
    G12.template segment<3>(0) = basis * G2;
    G12.template segment<3>(3) = (-1.0 + beta(0) + beta(1)) * G12.template segment<3>(0);
    G12.template segment<3>(6) = -beta(0) * G12.template segment<3>(0);
    G12.template segment<3>(9) = -beta(1) * G12.template segment<3>(0);
}

template <typename T>
inline void point_triangle_jacobi(const LMat<T, 3, 2>& basis,
                                  const LVec<T, 2>&    beta,
                                  LMat<T, 2, 12>&      J)
{
    J.template block<2, 3>(0, 0) = basis.transpose();
    J.template block<2, 3>(0, 3) = (-1.0 + beta(0) + beta(1)) * basis.transpose();
    J.template block<2, 3>(0, 6) = -beta(0) * basis.transpose();
    J.template block<2, 3>(0, 9) = -beta(1) * basis.transpose();
}

// Edge - Edge

template <typename T>
inline void edge_edge_tangent_basis(const LVec<T, 3>& Ea0,
                                    const LVec<T, 3>& Ea1,
                                    const LVec<T, 3>& Eb0,
                                    const LVec<T, 3>& Eb1,
                                    LMat<T, 3, 2>&    basis)
{
    LVec<T, 3> v01 = Ea1 - Ea0;
    basis.col(0)   = v01.normalized();
    basis.col(1)   = lps::cross(lps::cross(v01, Eb1 - Eb0), v01).normalized();
}

template <typename T>
inline void edge_edge_closest_point(const LVec<T, 3>& Ea0,
                                    const LVec<T, 3>& Ea1,
                                    const LVec<T, 3>& Eb0,
                                    const LVec<T, 3>& Eb1,
                                    LVec<T, 2>&       gamma)
{
    LVec<T, 3> e20 = Ea0 - Eb0;
    LVec<T, 3> e01 = Ea1 - Ea0;
    LVec<T, 3> e23 = Eb1 - Eb0;

    LMat<T, 2, 2> coefMtr;
    coefMtr(0, 0) = e01.squaredNorm();
    coefMtr(0, 1) = coefMtr(1, 0) = -lps::dot(e23, e01);
    coefMtr(1, 1)                 = e23.squaredNorm();

    LVec<T, 2> rhs;
    rhs(0) = -lps::dot(e20, e01);
    rhs(1) = lps::dot(e20, e23);

    // Note: don't use Eigen solve, it doesn't work for CUDA
    gamma = lps::inverse(coefMtr) * rhs;
}

template <typename T>
inline void edge_edge_rel_dx(const LVec<T, 3>& dEa0,
                             const LVec<T, 3>& dEa1,
                             const LVec<T, 3>& dEb0,
                             const LVec<T, 3>& dEb1,
                             const LVec<T, 2>& gamma,
                             LVec<T, 3>&       rel_dx)
{
    rel_dx = dEa0 + gamma(0) * (dEa1 - dEa0) - (dEb0 + gamma(1) * (dEb1 - dEb0));
}

template <typename T>
inline void edge_edge_tan_rel_dx(const LVec<T, 3>&    dEa0,
                                 const LVec<T, 3>&    dEa1,
                                 const LVec<T, 3>&    dEb0,
                                 const LVec<T, 3>&    dEb1,
                                 const LMat<T, 3, 2>& basis,
                                 const LVec<T, 2>&    gamma,
                                 LVec<T, 2>&          tan_rel_dx)
{
    LVec<T, 3> rel_dx;
    edge_edge_rel_dx(dEa0, dEa1, dEb0, dEb1, gamma, rel_dx);
    tangent_rel_dx(rel_dx, basis, tan_rel_dx);
}

template <typename T>
inline void apply_edge_edge_jacobi(const LVec<T, 2>&    G2,
                                   const LMat<T, 3, 2>& basis,
                                   const LVec<T, 2>&    gamma,
                                   LVec<T, 12>&         G12)
{
    LVec<T, 3> relDXTan3D = basis * G2;
    G12.template segment<3>(0) = (1.0 - gamma(0)) * relDXTan3D;
    G12.template segment<3>(3) = gamma(0) * relDXTan3D;
    G12.template segment<3>(6) = (gamma(1) - 1.0) * relDXTan3D;
    G12.template segment<3>(9) = -gamma(1) * relDXTan3D;
}

template <typename T>
inline void edge_edge_jacobi(const LMat<T, 3, 2>& basis,
                             const LVec<T, 2>&    gamma,
                             LMat<T, 2, 12>&      J)
{
    J.template block<2, 3>(0, 0) = (1.0 - gamma(0)) * basis.transpose();
    J.template block<2, 3>(0, 3) = gamma(0) * basis.transpose();
    J.template block<2, 3>(0, 6) = (gamma(1) - 1.0) * basis.transpose();
    J.template block<2, 3>(0, 9) = -gamma(1) * basis.transpose();
}

// Point - Edge

template <typename T>
inline void point_edge_tangent_basis(const LVec<T, 3>& P,
                                     const LVec<T, 3>& E0,
                                     const LVec<T, 3>& E1,
                                     LMat<T, 3, 2>&    basis)
{
    LVec<T, 3> v12 = E1 - E0;
    basis.col(0)   = v12.normalized();
    basis.col(1)   = lps::cross(v12, P - E0).normalized();
}

template <typename T>
inline void point_edge_closest_point(const LVec<T, 3>& P,
                                     const LVec<T, 3>& E0,
                                     const LVec<T, 3>& E1,
                                     T&                eta)
{
    LVec<T, 3> e12 = E1 - E0;
    eta            = lps::dot(P - E0, e12) / e12.squaredNorm();
}

template <typename T>
inline void point_edge_rel_dx(const LVec<T, 3>& dP,
                              const LVec<T, 3>& dE0,
                              const LVec<T, 3>& dE1,
                              const T&          eta,
                              LVec<T, 3>&       rel_dx)
{
    rel_dx = dP - (dE0 + eta * (dE1 - dE0));
}

template <typename T>
inline void point_edge_tan_rel_dx(const LVec<T, 3>&    dP,
                                  const LVec<T, 3>&    dE0,
                                  const LVec<T, 3>&    dE1,
                                  const LMat<T, 3, 2>& basis,
                                  const T&             eta,
                                  LVec<T, 2>&          tan_rel_dx)
{
    LVec<T, 3> rel_dx;
    point_edge_rel_dx(dP, dE0, dE1, eta, rel_dx);
    tangent_rel_dx(rel_dx, basis, tan_rel_dx);
}

template <typename T>
inline void apply_point_edge_jacobi(const LVec<T, 2>&    G2,
                                    const LMat<T, 3, 2>& basis,
                                    const T&             eta,
                                    LVec<T, 9>&          G9)
{
    G9.template segment<3>(0) = basis * G2;
    G9.template segment<3>(3) = (eta - 1.0) * G9.template segment<3>(0);
    G9.template segment<3>(6) = -eta * G9.template segment<3>(0);
}

template <typename T>
inline void point_edge_jacobi(const LMat<T, 3, 2>& basis, const T& eta, LMat<T, 2, 9>& J)
{
    J.template block<2, 3>(0, 0) = basis.transpose();
    J.template block<2, 3>(0, 3) = (eta - 1.0) * basis.transpose();
    J.template block<2, 3>(0, 6) = -eta * basis.transpose();
}

// Point - Point

template <typename T>
inline void point_point_tangent_basis(const LVec<T, 3>& P0,
                                      const LVec<T, 3>& P1,
                                      LMat<T, 3, 2>&    basis)
{
    LVec<T, 3> v01    = P1 - P0;
    LVec<T, 3> xCross = lps::cross(LVec<T, 3>::template Unit<0>(), v01);
    LVec<T, 3> yCross = lps::cross(LVec<T, 3>::template Unit<1>(), v01);
    // reference: `if(xCross.squaredNorm() > yCross.squaredNorm()) { ... } else { ... }`
    // -- a *device* predicate, so the port evaluates both candidates and selects
    // (see the file header: a select passes its chosen operand through unchanged).
    const auto use_x    = xCross.squaredNorm() > yCross.squaredNorm();
    LVec<T, 3> x_col0   = xCross.normalized();
    LVec<T, 3> x_col1   = lps::cross(v01, xCross).normalized();
    LVec<T, 3> y_col0   = yCross.normalized();
    LVec<T, 3> y_col1   = lps::cross(v01, yCross).normalized();
    basis.col(0)        = select_vec(y_col0, x_col0, use_x);
    basis.col(1)        = select_vec(y_col1, x_col1, use_x);
}

template <typename T>
inline void point_point_rel_dx(const LVec<T, 3>& dP0, const LVec<T, 3>& dP1, LVec<T, 3>& rel_dx)
{
    rel_dx = dP0 - dP1;
}

template <typename T>
inline void point_point_tan_rel_dx(const LVec<T, 3>&    dP0,
                                   const LVec<T, 3>&    dP1,
                                   const LMat<T, 3, 2>& basis,
                                   LVec<T, 2>&          tan_rel_dx)
{
    LVec<T, 3> rel_dx;
    point_point_rel_dx(dP0, dP1, rel_dx);
    tangent_rel_dx(rel_dx, basis, tan_rel_dx);
}

template <typename T>
inline void apply_point_point_jacobi(const LVec<T, 2>&    G2,
                                     const LMat<T, 3, 2>& basis,
                                     LVec<T, 6>&          G6)
{
    G6.template segment<3>(0) = basis * G2;
    G6.template segment<3>(3) = -G6.template segment<3>(0);
}

template <typename T>
inline void point_point_jacobi(const LMat<T, 3, 2>& basis, LMat<T, 2, 6>& J)
{
    J.template block<2, 3>(0, 0) = basis.transpose();
    J.template block<2, 3>(0, 3) = -basis.transpose();
}

}  // namespace uipc::backend::luisa::friction
