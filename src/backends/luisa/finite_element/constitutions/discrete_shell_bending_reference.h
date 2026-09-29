#pragma once
/********************************************************************
 * @file discrete_shell_bending_reference.h
 * @brief T5.2 (batch b) — the discrete-shell-bending reference metric (port of
 *        src/backends/cuda/finite_element/constitutions/
 *        discrete_shell_bending_reference.h).
 *
 * REFERENCE: `detail::compute_discrete_shell_bending_reference`, which fills the
 * rest length `L0`, the average height `h_bar`, the rest dihedral angle
 * `theta_bar` and the outer weight from the four rest positions of a bending
 * stencil.  The reference's own comment is kept verbatim below: `L0 / h_bar`
 * already carries the discrete-shells area normalization (3 L0^2 / A), so the
 * generic outer multiplier stays neutral.
 *
 * Mapping: `template <typename S>` (`S = double` host / `S = lcrt::Var<double>`
 * device), `Eigen::Vector3` -> `lps::LVec<S,3>`, scalar arithmetic through the
 * `lps::detail::{add,sub,mul,div}` trampolines (RULE M1) with the reference's
 * association order preserved verbatim (`A / 3.0 / L0` stays two divisions), and
 * the cross product re-spelled as the layer's free `lps::cross(a, b)` (Eigen's
 * `.cross(b)` member has no counterpart on `LVec`).
 ********************************************************************/
#include <type_define.h>
#include <luisa_device/math/math.h>
#include <utils/dihedral_angle.h>

namespace uipc::backend::luisa::detail
{
template <typename S>
inline void compute_discrete_shell_bending_reference(S&                     L0,
                                                     S&                     h_bar,
                                                     S&                     theta_bar,
                                                     S&                     outer_weight,
                                                     const lps::LVec<S, 3>& x0_bar,
                                                     const lps::LVec<S, 3>& x1_bar,
                                                     const lps::LVec<S, 3>& x2_bar,
                                                     const lps::LVec<S, 3>& x3_bar)
{
    namespace d = lps::detail;

    L0 = (x2_bar - x1_bar).norm();
    lps::LVec<S, 3> n1 = lps::cross(x1_bar - x0_bar, x2_bar - x0_bar);
    lps::LVec<S, 3> n2 = lps::cross(x2_bar - x3_bar, x1_bar - x3_bar);
    // reference: `Float A = (n1.norm() + n2.norm()) / 2.0;`
    S A = d::div(d::add(n1.norm(), n2.norm()), 2.0);
    // reference: `h_bar = A / 3.0 / L0;`
    h_bar = d::div(d::div(A, 3.0), L0);
    dihedral_angle(x0_bar, x1_bar, x2_bar, x3_bar, theta_bar);

    // L0 / h_bar already contains the complete Discrete Shells reference
    // metric (3 * L0^2 / A). Keep the generic outer multiplier neutral;
    // multiplying by A again would cancel the required area normalization.
    outer_weight = lps::make_scalar<S>(1.0);
}
}  // namespace uipc::backend::luisa::detail
