/********************************************************************
 * @file luisa_abd_constraint_test.cpp
 * @brief T6.4b — the ABD **constraints** of the LuisaCompute backend:
 *  * the **external articulation** constraint (constitution uid 23 + constraint uid 24,
 *    with its two SymEigen.py-generated bodies — the revolute and the prismatic half),
 *    plus its time integrator;
 *  * the **soft transform** constraint (uid 16), the ABD sibling of the FEM
 *    `SoftPositionConstraint` (uid 14);
 *  * the **external body force** pair (uid 666: the constraint *and* its reporter).
 *
 * Every oracle here is derived independently *in this file*: the reference backend is
 * CUDA-only and cannot be host-included, so nothing is copied from it (the Python
 * expression-level oracle `scratch/t64b_oracle.py` is the other, earlier independent
 * pass — see the generated bodies' own banners for its residuals).
 *
 * WHAT IS ORACLED, AND HOW
 *  (a) **closed forms** for the two generated articulation bodies, written out at the
 *      assertion.  With the pair dof vector `q = [qk ; ql]` (12 dofs each,
 *      `A(i,j) = q(3 + 3i + j)`), a 6-vector `basis = [first ; second]` and
 *      `u = F[0:3] v = F[3:6] p = F[6:9] qv = F[9:12]`:
 *        revolute  (uid 18): F = [A_k n_k ; A_k b_k ; A_l n_l ; A_l b_l]  (directions)
 *                            theta = atan2((v.p - u.qv)/2, (u.p + v.qv)/2)
 *                            DeltaTheta = wrap(theta(F) - theta(F_t))
 *        prismatic (uid 20): F = [A_k c_k + t_k ; A_k t_k ; A_l c_l + t_l ; A_l t_l]
 *                            DeltaTheta = ((p - u)/2).(v + qv)
 *                                         - ((p_t - u_t)/2).(v_t + q_t)
 *                            (a **linear** — quadratic in F — law, NOT an angle)
 *      and the derivative chains (`dDeltaTheta_dQ` 24, `dDeltaTheta_dF` 12,
 *      `ddDeltaTheta_ddF` 12x12) plus the two fan-outs
 *      `JT_G = J^T G_F`, `JT_H_J = J(bk,bl)^T H_F J(bm,bn)` with the 12x24 `J = dF/dq`.
 *      The *structure* flags the batch turns on: `ddDeltaTheta_ddF` is **not**
 *      diagonal (measured off-diagonal fraction 1.000e+00 for both bodies), so the
 *      `make_spd` clamp on `G_theta * ddDeltaTheta_ddF` is a real projection and the
 *      sweep count `abd_external_articulation_detail::kMakeSpdSweeps == 8` is
 *      observable (0-vs-8 vs 8-vs-20, measured here).
 *  (b) **central finite differences with an h-sweep** for every shipped gradient and
 *      Hessian of the two generated bodies (and the fan-out's two-operand question:
 *      `JT_H_J(H, bk, bl, bm, bn)` with `(bm, bn) != (bk, bl)` is pinned against
 *      `J1^T H J2`, so a swapped operand cannot pass).
 *  (c) the **soft transform** contract: its `compute_constraint_mass` law
 *      (`M_c = s_r M + (s_t - s_r) M_cm`, the parallel-axis split — `s_t == s_r == s`
 *      reproduces `s M` exactly, `s_t = 1, s_r = 0` gives `M_cm`), its three kernels
 *      (energy/gradient/Hessian, each `is_fixed`-guarded) driven directly through the
 *      public `AffineBodyAnimator::{ComputeEnergyInfo, ComputeGradientHessianInfo}`
 *      constructors, and the attribute contract (`is_constrained`, `aim_transform`,
 *      `strength_ratio`) through a real scene.
 *  (d) the **external body force** pair (uid 666): the constraint's extent (0/0/0) and
 *      its (body_id, force) table, the reporter's scatter-add (with *duplicate* body
 *      ids, which is what makes the vector atomic observable), and the shared uid.
 *  (e) a **real scene with one engine**: the host bookkeeping of all three systems, the
 *      assembly's COO differenced against the same scene with the articulation's
 *      `mass = 0` (a single-engine differencing trick: the sole difference between the
 *      two scenes is the articulation's mass, so the difference of the two assemblies
 *      is exactly the articulation's contribution), and the BDF1 prediction by hand.
 *  (f) **device == host** for the shipped kernels; the generated bodies are *not*
 *      bitwise (the layer's `lps::atan2`/`pow` and `make_spd`'s Jacobi sweeps), so a
 *      per-routine bound is asserted and the measurement recorded.
 *  (g) **uid ownership** for 16/23/24/666, printed and REQUIREd, including the 666
 *      constraint/reporter sharing and the 24 -> 18/20 cross-reference the articulation
 *      performs (its `build_joint_info` reads the *joint geometry's* `constitution_uid`
 *      and asserts it is 18 or 20), plus the *resolved* uid-16 collision with
 *      `luisa_abd_linear_test.cpp` (whose synthetic claim moved to uid 27).
 *
 * Coverage map:
 *  1) closed forms of the two generated bodies + the clamps ............... abd_constraint_articulation_closed_forms
 *  2) central FD (gradient/Hessian/symmetry/h-sweep/two-operand) .......... abd_constraint_articulation_finite_difference
 *  3) the assembly fan-out (COO difference, gradient) + the energy kernel . abd_constraint_articulation_assembly
 *  4) the soft transform contract + its three kernels .................... abd_constraint_soft_transform
 *  5) the external body force pair (666) + its atomic scatter ............ abd_constraint_external_body_force
 *  6) uid ownership of 16/23/24/666 ...................................... abd_constraint_uid_ownership
 *  7) a real scene, one engine, the BDF1 prediction by hand .............. abd_constraint_scene
 *  8) device == host ..................................................... abd_constraint_device_vs_host
 *  9) per-case assertion counts ......................................... abd_constraint_summary
 *
 * HARNESS HAZARDS RESPECTED (cookbook §7.27 rules 2/13-15, §7.28 rules 3/8/10)
 *  * no test-local kernel is written anywhere in this file: every device statement is
 *    driven through a *shipped* kernel (the two big generated 12x12 `make_spd` chains
 *    deadlock LC's compiler inside a test-local kernel — T6.3b3's headline finding), so
 *    the closed-forms and FD cases drive only the `S = double` *host* instantiations;
 *  * one engine per case (the assembly case uses the two-engine, no-advance idiom the
 *    [abd_joint_c] assembly case proved);
 *  * every device read-back synchronizes the frame stream first (a recorded kernel is
 *    invisible to a bare `copy_to`, §7.28 rule 10);
 *  * a failing `UIPC_ASSERT` in do_build/do_init aborts the process rather than
 *    reporting a Catch2 failure — so the scene cases' assertions are all *after*
 *    `world.init()`, and a mis-built scene shows up as a loud rc, not a silent pass.
 ********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/soft_transform_constraint.h>
#include <uipc/constitution/affine_body_external_force.h>
#include <uipc/constitution/affine_body_revolute_joint.h>
#include <uipc/constitution/affine_body_prismatic_joint.h>
#include <uipc/constitution/external_articulation_constraint.h>

#include <affine_body/type_define.h>
#include <affine_body/utils.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <animator/global_animator.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_animator.h>
#include <affine_body/affine_body_constraint.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/inter_affine_body_animator.h>
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/affine_body_external_force_manager.h>
#include <affine_body/affine_body_external_force_reporter.h>
#include <affine_body/affine_body_external_body_force.h>
#include <affine_body/constraints/external_articulation_constraint.h>
#include <affine_body/constraints/external_articulation_constraint_function.h>
#include <affine_body/constraints/soft_transform_constraint.h>
#include <affine_body/constraints/affine_body_external_body_force_constraint.h>
#include <linear_system/global_linear_system.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the shell header must
// be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;
// NOTE: spelled with the `namespace` keyword -- MSVC parses
// `using X = <a-namespace>;` as a *type* alias and rejects it (measured C2187/C2061).
namespace Detail = gg::abd_external_articulation_detail;

namespace
{
// ---------------------------------------------------------------------------
//  small helpers (the house style of the sibling ABD test files)
// ---------------------------------------------------------------------------
using Dense3   = Eigen::Matrix<double, 3, 3>;
using Dense3v  = Eigen::Matrix<double, 3, 1>;
using Dense6   = Eigen::Matrix<double, 6, 6>;
using Dense6v  = Eigen::Matrix<double, 6, 1>;
using Dense12  = Eigen::Matrix<double, 12, 12>;
using Dense12v = Eigen::Matrix<double, 12, 1>;
using Dense12x24 = Eigen::Matrix<double, 12, 24>;
using Dense24  = Eigen::Matrix<double, 24, 24>;
using Dense24v = Eigen::Matrix<double, 24, 1>;

double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

template <typename Derived>
double max_abs(const Eigen::MatrixBase<Derived>& m)
{
    return m.cwiseAbs().maxCoeff();
}

template <int N>
double min_eigenvalue(const Eigen::Matrix<double, N, N>& m)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> es(m);
    return es.eigenvalues()(0);
}

/// `max |A - B| / max(1, |A|_max, |B|_max)` -- the scale-free deviation the FEM/ABD
/// rows calibrate against (§7.17 rule 14's dev()).
template <typename A, typename B>
double dev(const A& a, const B& b)
{
    const double scale = std::max({1.0, max_abs(a), max_abs(b)});
    return max_abs(a - b) / scale;
}

/// `max |A - A^T|` over a square matrix (`0` == symmetric to the bit).
template <typename Derived>
double asymmetry(const Eigen::MatrixBase<Derived>& m)
{
    return max_abs(Eigen::MatrixXd(m - m.transpose()));
}

// ---- Eigen <-> numerics layer crossings (the sibling files' `lv_of`/`eig_of`) ----
template <int N>
[[nodiscard]] lps::LVec<double, N> lv_of(const Eigen::Matrix<double, N, 1>& v)
{
    lps::LVec<double, N> out;
    for(int i = 0; i < N; ++i)
        out(i) = v(i);
    return out;
}
template <int N>
[[nodiscard]] Eigen::Matrix<double, N, 1> eig_of(const lps::LVec<double, N>& v)
{
    Eigen::Matrix<double, N, 1> out;
    for(int i = 0; i < N; ++i)
        out(i) = v(i);
    return out;
}
template <int R, int C>
[[nodiscard]] lps::LMat<double, R, C> lm_of(const Eigen::Matrix<double, R, C>& m)
{
    lps::LMat<double, R, C> out;
    for(int c = 0; c < C; ++c)
        for(int r = 0; r < R; ++r)
            out(r, c) = m(r, c);
    return out;
}
template <int R, int C>
[[nodiscard]] Eigen::Matrix<double, R, C> eig_of(const lps::LMat<double, R, C>& m)
{
    Eigen::Matrix<double, R, C> out;
    for(int c = 0; c < C; ++c)
        for(int r = 0; r < R; ++r)
            out(r, c) = m(r, c);
    return out;
}

// ---- the 12-dof state conventions ----------------------------------------
Vector12 q_of(const Matrix3x3& A, const Vector3& t = Vector3::Zero())
{
    Vector12 q = Vector12::Zero();
    q.segment<3>(0) = t;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            q(3 + 3 * i + j) = A(i, j);
    return q;
}
Matrix3x3 A_of(const Vector12& q)
{
    Matrix3x3 A;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            A(i, j) = q(3 + 3 * i + j);
    return A;
}
Vector3 t_of(const Vector12& q)
{
    return q.segment<3>(0);
}
Matrix4x4 make_transform(const Matrix3x3& A, const Vector3& t)
{
    Matrix4x4 T = Matrix4x4::Identity();
    T.block<3, 3>(0, 0) = A;
    T.block<3, 1>(0, 3) = t;
    return T;
}
std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// Read an LC buffer back to the host (frame-stream first: a command may still be
/// recorded in an open frame — §7.28 rule 10).
template <typename T>
std::vector<T> read_device(const lud::LCBufferView<T>& view)
{
    std::vector<T> host(view.size());
    gg::frame_stream().synchronize();
    if(!host.empty())
    {
        view.copy_to(span<T>{host});
        lud::main_stream().synchronize();
    }
    return host;
}
std::vector<Vector12> read_vec12(const lud::LCBufferView<gg::LVector12>& view)
{
    auto raw = read_device(view);
    std::vector<Vector12> host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

// ---------------------------------------------------------------------------
//  the hand laws of the two generated articulation bodies (independent derivations)
// ---------------------------------------------------------------------------
/// `F(basis_k, qk, basis_l, ql)` of *both* bodies, in the test's own spelling.
/// `revolute == true`: two body-local directions.  `false`: the prismatic body's two
/// attachment *points* + directions (its `F` carries the translations).
[[nodiscard]] Dense12v hand_F(bool      revolute,
                              const Dense6v&  basis_k,
                              const Dense12v& qk,
                              const Dense6v&  basis_l,
                              const Dense12v& ql)
{
    const Matrix3x3 Ak = A_of(qk);
    const Matrix3x3 Al = A_of(ql);
    const Vector3   tk = t_of(qk);
    const Vector3   tl = t_of(ql);

    Dense12v F;
    if(revolute)
    {
        F.segment<3>(0) = Ak * basis_k.segment<3>(0);
        F.segment<3>(3) = Ak * basis_k.segment<3>(3);
        F.segment<3>(6) = Al * basis_l.segment<3>(0);
        F.segment<3>(9) = Al * basis_l.segment<3>(3);
    }
    else
    {
        F.segment<3>(0) = Ak * basis_k.segment<3>(0) + tk;
        F.segment<3>(3) = Ak * basis_k.segment<3>(3);
        F.segment<3>(6) = Al * basis_l.segment<3>(0) + tl;
        F.segment<3>(9) = Al * basis_l.segment<3>(3);
    }
    return F;
}

/// `theta(F) = atan2((v.p - u.qv)/2, (u.p + v.qv)/2)` — the *revolute* law's scalar.
[[nodiscard]] double hand_theta(const Dense12v& F)
{
    const Dense3v u = F.segment<3>(0), v = F.segment<3>(3);
    const Dense3v p = F.segment<3>(6), qv = F.segment<3>(9);
    return std::atan2((v.dot(p) - u.dot(qv)) / 2.0, (u.dot(p) + v.dot(qv)) / 2.0);
}

/// `DeltaTheta(F, F_t)` of both bodies.
[[nodiscard]] double hand_delta_theta_F(bool revolute, const Dense12v& F, const Dense12v& F_t)
{
    if(revolute)
    {
        // the *wrapped* difference, spelled with std::remainder (the port's
        // `unwrap_angle` identity, cookbook §7.27 rule 3)
        const double d = hand_theta(F) - hand_theta(F_t);
        return std::remainder(d, 2.0 * std::numbers::pi);
    }
    const Dense3v u = F.segment<3>(0), v = F.segment<3>(3);
    const Dense3v p = F.segment<3>(6), qv = F.segment<3>(9);
    const Dense3v ut = F_t.segment<3>(0), vt = F_t.segment<3>(3);
    const Dense3v pt = F_t.segment<3>(6), qt = F_t.segment<3>(9);
    return ((p - u) / 2.0).dot(v + qv) - ((pt - ut) / 2.0).dot(vt + qt);
}

/// The 6-vector overload's law.
[[nodiscard]] double hand_delta_theta_q(bool            revolute,
                                        const Dense6v&  basis_k,
                                        const Dense12v& qk,
                                        const Dense12v& qk_t,
                                        const Dense6v&  basis_l,
                                        const Dense12v& ql,
                                        const Dense12v& ql_t)
{
    return hand_delta_theta_F(revolute,
                              hand_F(revolute, basis_k, qk, basis_l, ql),
                              hand_F(revolute, basis_k, qk_t, basis_l, ql_t));
}

// ---- central differences with an h-sweep ---------------------------------
template <typename ScalarLaw, int N>
[[nodiscard]] Eigen::Matrix<double, N, 1> fd_grad(const ScalarLaw& law,
                                                  const Eigen::Matrix<double, N, 1>& x,
                                                  double                             h)
{
    Eigen::Matrix<double, N, 1> g;
    for(int k = 0; k < N; ++k)
    {
        auto xp = x;
        auto xm = x;
        xp(k) += h;
        xm(k) -= h;
        g(k) = (law(xp) - law(xm)) / (2.0 * h);
    }
    return g;
}

template <typename ScalarLaw, int N>
[[nodiscard]] Eigen::Matrix<double, N, N> fd_hess(const ScalarLaw& law,
                                                  const Eigen::Matrix<double, N, 1>& x,
                                                  double                             h)
{
    Eigen::Matrix<double, N, N> H;
    for(int k = 0; k < N; ++k)
    {
        auto xp = x;
        auto xm = x;
        xp(k) += h;
        xm(k) -= h;
        H.col(k) = (fd_grad(law, xp, h) - fd_grad(law, xm, h)) / (2.0 * h);
    }
    return 0.5 * (H + H.transpose());
}

/// The 12x24 `J = dF/d[qk ; ql]` of a body's `F`, by central differences.
[[nodiscard]] Dense12x24 fd_F_jacobian(bool            revolute,
                                       const Dense6v&  basis_k,
                                       const Dense6v&  basis_l,
                                       const Dense24v& q24,
                                       double          h = 1e-7)
{
    Dense12x24 J;
    for(int k = 0; k < 24; ++k)
    {
        auto xp = q24;
        auto xm = q24;
        xp(k) += h;
        xm(k) -= h;
        J.col(k) = (hand_F(revolute, basis_k, xp.segment<12>(0), basis_l, xp.segment<12>(12))
                    - hand_F(revolute, basis_k, xm.segment<12>(0), basis_l, xm.segment<12>(12)))
                   / (2.0 * h);
    }
    return J;
}

// ---- the shipped generated bodies, called with S = double (host) ----------
[[nodiscard]] Dense12v shipped_F(bool            revolute,
                                 const Dense6v&  basis_k,
                                 const Dense12v& qk,
                                 const Dense6v&  basis_l,
                                 const Dense12v& ql)
{
    if(revolute)
        return eig_of(Detail::revolute_f<double>(
            lv_of<6>(basis_k), lv_of<12>(qk), lv_of<6>(basis_l), lv_of<12>(ql)));
    return eig_of(Detail::prismatic_f<double>(
        lv_of<6>(basis_k), lv_of<12>(qk), lv_of<6>(basis_l), lv_of<12>(ql)));
}

[[nodiscard]] double shipped_delta_theta_F(bool revolute, const Dense12v& F, const Dense12v& F_t)
{
    double out = 0.0;
    if(revolute)
        gg::sym::external_revolute_joint_constraint::DeltaTheta<double>(
            out, lv_of<12>(F), lv_of<12>(F_t));
    else
        gg::sym::external_prismatic_joint_constraint::DeltaTheta<double>(
            out, lv_of<12>(F), lv_of<12>(F_t));
    return out;
}

[[nodiscard]] double shipped_delta_theta_q(bool            revolute,
                                           const Dense6v&  basis_k,
                                           const Dense12v& qk,
                                           const Dense12v& qk_t,
                                           const Dense6v&  basis_l,
                                           const Dense12v& ql,
                                           const Dense12v& ql_t)
{
    if(revolute)
        return Detail::revolute_delta_theta<double>(lv_of<6>(basis_k), lv_of<12>(qk),
                                                   lv_of<12>(qk_t), lv_of<6>(basis_l),
                                                   lv_of<12>(ql), lv_of<12>(ql_t));
    return Detail::prismatic_delta_theta<double>(lv_of<6>(basis_k), lv_of<12>(qk),
                                                lv_of<12>(qk_t), lv_of<6>(basis_l),
                                                lv_of<12>(ql), lv_of<12>(ql_t));
}

[[nodiscard]] Dense24v shipped_d_delta_theta_dq(bool            revolute,
                                                const Dense6v&  basis_k,
                                                const Dense12v& qk,
                                                const Dense12v& qk_t,
                                                const Dense6v&  basis_l,
                                                const Dense12v& ql,
                                                const Dense12v& ql_t)
{
    if(revolute)
        return eig_of(Detail::revolute_d_delta_theta_dq<double>(
            lv_of<6>(basis_k), lv_of<12>(qk), lv_of<12>(qk_t), lv_of<6>(basis_l),
            lv_of<12>(ql), lv_of<12>(ql_t)));
    return eig_of(Detail::prismatic_d_delta_theta_dq<double>(
        lv_of<6>(basis_k), lv_of<12>(qk), lv_of<12>(qk_t), lv_of<6>(basis_l),
        lv_of<12>(ql), lv_of<12>(ql_t)));
}

[[nodiscard]] Dense12v shipped_d_delta_theta_dF(bool revolute, const Dense12v& F, const Dense12v& F_t)
{
    if(revolute)
    {
        lps::LVec<double, 12> out;
        gg::sym::external_revolute_joint_constraint::dDeltaTheta_dF<double>(
            out, lv_of<12>(F), lv_of<12>(F_t));
        return eig_of(out);
    }
    lps::LVec<double, 12> out;
    gg::sym::external_prismatic_joint_constraint::dDeltaTheta_dF<double>(
        out, lv_of<12>(F), lv_of<12>(F_t));
    return eig_of(out);
}

[[nodiscard]] Dense12 shipped_dd_delta_theta_ddF(bool revolute, const Dense12v& F, const Dense12v& F_t)
{
    if(revolute)
        return eig_of(Detail::revolute_dd_delta_theta_ddf<double>(lv_of<12>(F), lv_of<12>(F_t)));
    return eig_of(Detail::prismatic_dd_delta_theta_ddf<double>(lv_of<12>(F), lv_of<12>(F_t)));
}

[[nodiscard]] Dense24v shipped_jt_g(bool            revolute,
                                    const Dense12v& G_F,
                                    const Dense6v&  basis_k,
                                    const Dense6v&  basis_l)
{
    lps::LVec<double, 24> out;
    if(revolute)
        gg::sym::external_revolute_joint_constraint::JT_G<double>(
            out, lv_of<12>(G_F), lv_of<6>(basis_k), lv_of<6>(basis_l));
    else
        gg::sym::external_prismatic_joint_constraint::JT_G<double>(
            out, lv_of<12>(G_F), lv_of<6>(basis_k), lv_of<6>(basis_l));
    return eig_of(out);
}

[[nodiscard]] Dense24 shipped_jt_h_j(bool            revolute,
                                     const Dense12&  H_F,
                                     const Dense6v&  basis_k,
                                     const Dense6v&  basis_l,
                                     const Dense6v&  basis_m,
                                     const Dense6v&  basis_n)
{
    lps::LMat<double, 24, 24> out;
    if(revolute)
        gg::sym::external_revolute_joint_constraint::JT_H_J<double>(
            out, lm_of<12, 12>(H_F), lv_of<6>(basis_k), lv_of<6>(basis_l), lv_of<6>(basis_m),
            lv_of<6>(basis_n));
    else
        gg::sym::external_prismatic_joint_constraint::JT_H_J<double>(
            out, lm_of<12, 12>(H_F), lv_of<6>(basis_k), lv_of<6>(basis_l), lv_of<6>(basis_m),
            lv_of<6>(basis_n));
    return eig_of(out);
}

// ---------------------------------------------------------------------------
//  the soft transform constraint's hand law
// ---------------------------------------------------------------------------
/// The `ABDJacobiDyadicMass::to_mat()` 12x12, re-derived from its documented
/// triple layout: `M(i,i) += m`, the i-th translation row/column picks up `m c`
/// against the (3 + 3i)-th 3-block and `M(3+3i, 3+3i) += m c c^T`.
[[nodiscard]] Dense12 hand_dyadic_to_mat(double m, const Dense3v& mc, const Dense3& S_dy)
{
    Dense12 M = Dense12::Zero();
    for(int i = 0; i < 3; ++i)
    {
        M(i, i) += m;
        M.block<1, 3>(i, 3 + 3 * i) += mc.transpose();
        M.block<3, 1>(3 + 3 * i, i) += mc;
        M.block<3, 3>(3 + 3 * i, 3 + 3 * i) += S_dy;
    }
    return M;
}

/// `M_c = s_r M + (s_t - s_r) M_cm` with `M_cm` the ABD mass of a point mass at the
/// body's center of mass (`m J(c)^T J(c)`, i.e. the same layout built from
/// `(m, m c, m c c^T)`).
[[nodiscard]] Dense12 hand_constraint_mass(double          m,
                                           const Dense3v&  mc,
                                           const Dense3&   S_dy,
                                           double          s_t,
                                           double          s_r)
{
    const Dense12 M    = hand_dyadic_to_mat(m, mc, S_dy);
    const Dense3  mccT = mc * mc.transpose() / m;
    const Dense12 M_cm = hand_dyadic_to_mat(m, mc, mccT);
    return s_r * M + (s_t - s_r) * M_cm;
}

/// The *shipped* `compute_constraint_mass` (S = double) on the same input.
[[nodiscard]] Dense12 shipped_constraint_mass(double          m,
                                              const Dense3v&  mc,
                                              const Dense3&   S_dy,
                                              double          s_t,
                                              double          s_r)
{
    const gg::ABDJacobiDyadicMassT<double> mass{.m_mass = m,
                                                .m_mass_times_x_bar = lv_of<3>(mc),
                                                .m_mass_times_dyadic_x_bar = lm_of<3, 3>(S_dy)};
    return eig_of(
        gg::soft_transform_constraint_detail::compute_constraint_mass<double>(mass, s_t, s_r));
}

/// The clamped lerp of `cuda/animator/utils.h` (the port's `animator::lerp`), spelled
/// here for the host oracle: `src (1 - a') + dst a'`, `a' = clamp(a, 0, 1)`.
[[nodiscard]] Vector12 hand_clamped_lerp(const Vector12& src, const Vector12& dst, double alpha)
{
    const double a = std::clamp(alpha, 0.0, 1.0);
    return src * (1.0 - a) + dst * a;
}

// ---------------------------------------------------------------------------
//  scene helpers
// ---------------------------------------------------------------------------
constexpr Float kL     = 0.5;
constexpr Float kRho   = 1000.0;
constexpr Float kKappa = 1.0e8;
/// uid 2 = the frontend's "ARAP".
constexpr U64 kArapUID = 2ull;

std::array<Vector3, 8> cube_vertices(Float L)
{
    // vertex index = x + 2y + 4z -- the SAME order the six-tet fan below assumes
    // (swapping 5 and 6 inverts two of the six tets and the whole mesh's volume
    // collapses to 0, which the ABD constitution's own assert catches)
    return {Vector3{0, 0, 0} * L, Vector3{1, 0, 0} * L, Vector3{0, 1, 0} * L,
            Vector3{1, 1, 0} * L, Vector3{0, 0, 1} * L, Vector3{1, 0, 1} * L,
            Vector3{0, 1, 1} * L, Vector3{1, 1, 1} * L};
}
std::array<Vector4i, 6> cube_tets()
{
    return {Vector4i{0, 1, 3, 7}, Vector4i{0, 3, 2, 7}, Vector4i{0, 2, 6, 7},
            Vector4i{0, 6, 4, 7}, Vector4i{0, 4, 5, 7}, Vector4i{0, 5, 1, 7}};
}

/// One ABD body: an `L`-cube of density `rho`, constitution uid `uid` and the
/// *initial* transform `(A, t)` -- which is also the ABD state `q`.
SimplicialComplex make_body(Float            kappa,
                            const Matrix3x3& A,
                            const Vector3&   t,
                            U64              uid   = kArapUID,
                            const Matrix3x3& V     = Matrix3x3::Zero(),
                            const Vector3&   v     = Vector3::Zero())
{
    const auto vs = cube_vertices(kL);
    const auto ts = cube_tets();
    auto sc = tetmesh(span<const Vector3>{vs.data(), vs.size()},
                      span<const Vector4i>{ts.data(), ts.size()});
    label_surface(sc);
    sc.instances().resize(1);
    uipc::constitution::AffineBodyConstitution abd;
    abd.apply_to(sc, kappa, kRho);
    auto cuid = sc.meta().find<U64>(builtin::constitution_uid);
    REQUIRE(cuid != nullptr);
    view(*cuid).front() = uid;
    view(sc.transforms())[0] = make_transform(A, t);
    view(*sc.instances().find<Matrix4x4>(builtin::velocity))[0] = make_transform(V, v);
    return sc;
}

Json constraint_scene_config()
{
    auto config                    = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = false;
    config["integrator"]["type"]     = std::string{"bdf1"};
    // zero gravity: the BDF1 prediction is then `q + q_v dt + a_ext dt^2`
    config["gravity"] = Vector3{0.0, 0.0, 0.0};
    return config;
}

// ---------------------------------------------------------------------------
//  the probe: the built ABD/constraint systems of one scene
// ---------------------------------------------------------------------------
class T64BProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::AffineBodyDynamics*    abd            = nullptr;
    inline static gg::ABDLinearSubsystem*    abd_linear     = nullptr;
    inline static gg::AffineBodyAnimator*    animator       = nullptr;
    inline static gg::GlobalAnimator*        global_animator = nullptr;
    inline static gg::SoftTransformConstraint* soft_transform = nullptr;
    inline static gg::AffineBodyExternalBodyForceConstraint* ext_body_force_constraint = nullptr;
    inline static gg::AffineBodyExternalBodyForce* ext_body_force_reporter = nullptr;
    inline static gg::AffineBodyExternalForceManager* ext_force_manager = nullptr;
    inline static gg::ExternalArticulationConstraint* articulation = nullptr;
    inline static gg::ExternalArticulationConstituion* articulation_constitution = nullptr;
    inline static gg::ExternalArticulationConstraintTimeIntegrator* articulation_integrator = nullptr;
    inline static gg::InterAffineBodyAnimator* inter_animator = nullptr;
    inline static gg::InterAffineBodyConstitutionManager* inter_manager = nullptr;
    inline static T64BProbe* self = nullptr;

    /// The scene's `dt` -- the *same* attribute the kernels read.
    [[nodiscard]] Float dt()
    {
        auto attr = world().scene().config().find<Float>("dt");
        UIPC_ASSERT(attr, "T64BProbe: the scene config has no `dt`");
        return attr->view()[0];
    }
    [[nodiscard]] S<const geometry::AttributeSlot<Float>> dt_attr()
    {
        return world().scene().config().find<Float>("dt");
    }

  private:
    void do_build() override
    {
        self = this;
        abd                    = &require<gg::AffineBodyDynamics>();
        abd_linear             = find<gg::ABDLinearSubsystem>();
        animator               = find<gg::AffineBodyAnimator>();
        global_animator        = find<gg::GlobalAnimator>();
        soft_transform         = find<gg::SoftTransformConstraint>();
        ext_body_force_constraint = find<gg::AffineBodyExternalBodyForceConstraint>();
        ext_body_force_reporter   = find<gg::AffineBodyExternalBodyForce>();
        ext_force_manager      = find<gg::AffineBodyExternalForceManager>();
        articulation           = find<gg::ExternalArticulationConstraint>();
        articulation_constitution = find<gg::ExternalArticulationConstituion>();
        articulation_integrator   = find<gg::ExternalArticulationConstraintTimeIntegrator>();
        inter_animator         = find<gg::InterAffineBodyAnimator>();
        inter_manager          = find<gg::InterAffineBodyConstitutionManager>();
    }
};
REGISTER_SIM_SYSTEM(T64BProbe);
}  // namespace

namespace
{
// ---------------------------------------------------------------------------
//  case accounting (the sibling files' pattern)
// ---------------------------------------------------------------------------
std::vector<std::pair<std::string, int>> g_case_counts;
struct Counter
{
    int checks = 0;
};
struct CaseScope
{
    std::string name;
    Counter     counter;
    explicit CaseScope(std::string n)
        : name(std::move(n))
    {
    }
    ~CaseScope() { g_case_counts.emplace_back(name, counter.checks); }
};
#define T64B_CHECK(counter, cond) \
    do \
    { \
        ++(counter).checks; \
        CHECK(cond); \
    } while(false)

// ---------------------------------------------------------------------------
//  deterministic pseudo-random inputs (a fixed seed -> reproducible oracles)
// ---------------------------------------------------------------------------
std::mt19937& rng()
{
    static std::mt19937 g{0x64B0u};
    return g;
}
double rand_uniform(double lo, double hi)
{
    return std::uniform_real_distribution<double>(lo, hi)(rng());
}
std::vector<double> rand_vec(int n, double lo, double hi)
{
    std::vector<double> v(static_cast<std::size_t>(n));
    for(auto& x : v)
        x = rand_uniform(lo, hi);
    return v;
}
[[nodiscard]] Dense12v v12v(const std::vector<double>& v, int off = 0)
{
    Dense12v out;
    for(int i = 0; i < 12; ++i)
        out(i) = v[static_cast<std::size_t>(off + i)];
    return out;
}
[[nodiscard]] Dense6v v6v(const std::vector<double>& v, int off = 0)
{
    Dense6v out;
    for(int i = 0; i < 6; ++i)
        out(i) = v[static_cast<std::size_t>(off + i)];
    return out;
}

/// A *properly scaled* random state (`A` near a rotation) so the transcendental
/// `atan2` chains stay well conditioned.
[[nodiscard]] Dense12v random_q()
{
    Dense12v q = Dense12v::Zero();
    for(int i = 0; i < 12; ++i)
        q(i) = rand_uniform(-0.35, 0.35);
    q.segment<9>(3) += Dense12v::Ones().segment<9>(3) * 1.0;  // start near I
    return q;
}

// ---------------------------------------------------------------------------
//  the reference's `orthonormal_basis(t, n, b)` and Eigen's polar rotation
// ---------------------------------------------------------------------------
/// `t` normalised in place, `test = ex` unless `|t . ex| > 0.9` (then `ey`),
/// `n = (t x test).normalized()`, `b = (t x n).normalized()` — the port's
/// `orthonormal_basis` re-derived (T6.3b1's hand spelling, re-used here).
void hand_orthonormal_basis(Vector3& t, Vector3& n, Vector3& b)
{
    t                  = t.normalized();
    const Vector3 test = std::abs(t.dot(Vector3(1, 0, 0))) > 0.9 ? Vector3(0, 1, 0)
                                                                : Vector3(1, 0, 0);
    n = t.cross(test).normalized();
    b = t.cross(n).normalized();
}
/// The *rotation* part of a linear map (`U V^T` of its SVD) — the independent spelling
/// of what Eigen's `Transform<...,Eigen::Affine>::rotation()` computes.
Matrix3x3 eigen_polar_rotation(const Matrix3x3& A)
{
    Eigen::JacobiSVD<Matrix3x3> sv(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return sv.matrixU() * sv.matrixV().transpose();
}

// ---------------------------------------------------------------------------
//  the assembly: a test-local GlobalLinearSystem::Impl (the T6.1b/T6.3b idiom)
// ---------------------------------------------------------------------------
using BlockMap = std::map<std::pair<int, int>, Matrix3x3>;

template <typename TripletT>
BlockMap read_blocks(const TripletT& A)
{
    const auto n = static_cast<std::size_t>(A.triplet_count());
    std::vector<std::int32_t> rows(n), cols(n);
    std::vector<double>       vals(n * 9);
    if(n > 0)
    {
        A.copy_to_row_indices(span<std::int32_t>{rows});
        A.copy_to_col_indices(span<std::int32_t>{cols});
        A.copy_to_values(span<double>{vals});
    }
    BlockMap out;
    for(std::size_t t = 0; t < n; ++t)
    {
        Matrix3x3 blk;
        for(int cc = 0; cc < 3; ++cc)
            for(int rr = 0; rr < 3; ++rr)
                blk(rr, cc) = vals[t * 9 + static_cast<std::size_t>(cc * 3 + rr)];
        auto key = std::make_pair(static_cast<int>(rows[t]), static_cast<int>(cols[t]));
        auto it  = out.find(key);
        if(it == out.end())
            out.emplace(key, blk);
        else
            it->second += blk;
    }
    return out;
}

/// `A - B` over the union of the two maps' keys (a missing key is a zero block).
BlockMap sub_blocks(const BlockMap& a, const BlockMap& b)
{
    BlockMap out = a;
    for(const auto& [key, blk] : b)
    {
        auto it = out.find(key);
        if(it == out.end())
            out.emplace(key, -blk);
        else
            it->second -= blk;
    }
    return out;
}

/// Writes the per-geometry dof offsets the way the engine does (`§7.22 rule 12`).
void prepare_dof_layout(gg::ABDLinearSubsystem& abd_linear)
{
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(abd_linear);
    impl.init();
}

struct AssemblyRun
{
    BlockMap          blocks;
    Eigen::VectorXd   gradient;
    std::size_t       triplet_count   = 0;
    std::size_t       expected_extent = 0;
    std::size_t       dof_count       = 0;
};

AssemblyRun run_assembly(gg::ABDLinearSubsystem& abd_linear)
{
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(abd_linear);
    impl.init();

    AssemblyRun out;
    out.dof_count = static_cast<std::size_t>(impl.diag_dof_offsets_counts.total_count());
    REQUIRE(impl._update_subsystem_extent());
    out.expected_extent =
        static_cast<std::size_t>(impl.subsystem_triplet_offsets_counts.counts()[0]);
    impl._assemble_linear_system();
    // the assembly is *recorded* on the frame stream: flush it before reading
    gg::frame_stream().synchronize();

    out.triplet_count = static_cast<std::size_t>(impl.triplet_A.triplet_count());
    out.blocks        = read_blocks(impl.triplet_A);
    out.gradient.resize(static_cast<Eigen::Index>(out.dof_count));
    impl.b.copy_to(span<double>{out.gradient.data(), out.dof_count});
    return out;
}

// ---------------------------------------------------------------------------
//  driving a constraint's hooks directly (the [abd_joint_c] harness)
// ---------------------------------------------------------------------------
void setup_animator_energy_impl(gg::AffineBodyAnimator::Impl& impl,
                                gg::AffineBodyDynamics&        abd,
                                S<const geometry::AttributeSlot<Float>> dt_attr,
                                SizeT                          n)
{
    impl.affine_body_dynamics = &abd;
    impl.dt_attr              = dt_attr;
    impl.constraint_energy_offsets_counts.resize(1);
    impl.constraint_energy_offsets_counts.counts()[0] = static_cast<IndexT>(n);
    impl.constraint_energy_offsets_counts.scan();
}

template <typename Constraint>
std::vector<double> run_constraint_energy(Constraint&                     c,
                                          gg::AffineBodyDynamics&         abd,
                                          S<const geometry::AttributeSlot<Float>> dt_attr,
                                          Float                           dt,
                                          SizeT                           n)
{
    gg::AffineBodyAnimator::Impl impl;
    setup_animator_energy_impl(impl, abd, dt_attr, n);
    // `BaseInfo::substep_ratio()` dereferences `m_impl->global_animator` (the
    // reference does too), so the test-built Impl has to carry the scene's real one --
    // a null here is a SIGSEGV, not a Catch2 failure (measured)
    REQUIRE(T64BProbe::global_animator != nullptr);
    impl.global_animator = T64BProbe::global_animator;
    lud::LDeviceBuffer<Float> Es(n);
    gg::AffineBodyAnimator::ComputeEnergyInfo info{&impl, 0, dt, Es.view()};
    c.do_compute_energy(info);
    gg::frame_stream().synchronize();
    return read_device(Es.cview());
}

/// The articulation's `ComputeEnergyInfo` (the *inter*-body animator's).
template <typename Constraint>
std::vector<double> run_inter_constraint_energy(Constraint&                     c,
                                                gg::AffineBodyDynamics&         abd,
                                                S<const geometry::AttributeSlot<Float>> dt_attr,
                                                Float                           dt,
                                                SizeT                           n)
{
    gg::InterAffineBodyAnimator::Impl impl;
    impl.affine_body_dynamics = &abd;
    impl.dt_attr              = dt_attr;
    impl.constraint_energy_offsets_counts.resize(1);
    impl.constraint_energy_offsets_counts.counts()[0] = static_cast<IndexT>(n);
    impl.constraint_energy_offsets_counts.scan();
    lud::LDeviceBuffer<Float> Es(n);
    gg::InterAffineBodyAnimator::ComputeEnergyInfo info{&impl, 0, dt, Es.view()};
    c.do_compute_energy(info);
    gg::frame_stream().synchronize();
    return read_device(Es.cview());
}

// ---------------------------------------------------------------------------
//  the articulation scene: 2 ABD bodies + one joint (uid 18 or 20) + an articulation
// ---------------------------------------------------------------------------
struct ArtRest
{
    Vector3 l0 = Vector3{0.10, -0.05, 0.20};
    Vector3 l1 = Vector3{0.45, 0.10, 0.06};
    Vector3 r0 = Vector3{-0.20, 0.15, 0.10};
    Vector3 r1 = Vector3{-0.02, 0.30, -0.12};
};

struct ArtScene
{
    Matrix3x3 A0 = Eigen::AngleAxisd(0.6, Vector3{0.3, -1.0, 0.5}.normalized()).toRotationMatrix()
                   * Vector3{1.25, 0.9, 1.1}.asDiagonal();
    Matrix3x3 A1 = Eigen::AngleAxisd(-1.1, Vector3{1.0, 0.4, -0.7}.normalized()).toRotationMatrix()
                   * Vector3{0.85, 1.3, 1.0}.asDiagonal();
    Vector3 t0 = Vector3{0.2, -0.1, 0.3};
    Vector3 t1 = Vector3{-0.4, 0.25, 0.15};
    ArtRest rest;
    /// The previous state the articulation reads through the `ref_dof_prev` protocol
    /// (a *different* state, so `DeltaTheta != 0`; the bodies' own `q_prevs` equal their
    /// `q` before any Newton solve — the port's solve path is Phase 8's).
    Matrix3x3 A0_prev = Eigen::AngleAxisd(0.25, Vector3{0.2, 0.9, -0.4}.normalized()).toRotationMatrix()
                        * Vector3{1.1, 1.0, 0.95}.asDiagonal();
    Matrix3x3 A1_prev = Eigen::AngleAxisd(-0.35, Vector3{-0.6, 0.3, 0.8}.normalized()).toRotationMatrix();
    Vector3 t0_prev = Vector3{0.15, -0.05, 0.25};
    Vector3 t1_prev = Vector3{-0.35, 0.2, 0.1};
    Float   joint_ratio        = 100.0;
    Float   articulation_mass  = 64.0;
    Float   delta_theta_tilde  = 0.17;
    /// `strength_ratio` of the frontend `SoftTransformConstraint` (one instance).
    Vector2 stc_strength{100.0, 1.0e6};
    IndexT  stc_is_constrained = 1;
    /// The frontend `AffineBodyExternalBodyForce`'s per-instance generalized force.
    Vector12 ext_force = Vector12::Zero();

    ArtScene()
    {
        ext_force.segment<3>(0) = Vector3{5.0, -2.0, 3.0};
        ext_force(5)            = 0.75;
    }

    /// `prismatic == false` builds the uid-18 (revolute) joint, `true` the uid-20 one.
    /// `zero_mass` sets the articulation's `joint_joint/mass` to 0 — the assembly
    /// differencing trick: both scenes have the *same* COO layout (the extents depend on
    /// the counts, not the values), so `assembled(mass) - assembled(0)` is exactly the
    /// articulation's contribution.
    [[nodiscard]] Scene build(const std::string& name,
                              bool               prismatic,
                              bool               with_articulation,
                              bool               zero_mass          = false,
                              bool               with_soft_transform = false,
                              bool               with_ext_force      = false) const
    {
        Scene scene{constraint_scene_config()};
        auto  object = scene.objects().create(name);

        auto body0 = make_body(kKappa, A0, t0);
        auto body1 = make_body(kKappa, A1, t1);

        // the articulation's `ref_dof_prev` protocol: a *different* previous state
        for(auto* sc : {&body0, &body1})
        {
            auto attr = sc->instances().create<Vector12>("ref_dof_prev", Vector12::Zero());
            view(*attr)[0] = (sc == &body0) ? q_of(A0_prev, t0_prev) : q_of(A1_prev, t1_prev);
        }

        if(with_soft_transform)
        {
            uipc::constitution::SoftTransformConstraint stc;
            for(auto* sc : {&body0, &body1})
            {
                stc.apply_to(*sc, stc_strength);
                view(*sc->instances().find<IndexT>(builtin::is_constrained))[0] = stc_is_constrained;
            }
        }
        if(with_ext_force)
        {
            uipc::constitution::AffineBodyExternalBodyForce ext;
            for(auto* sc : {&body0, &body1})
            {
                ext.apply_to(*sc, ext_force);
                view(*sc->instances().find<IndexT>(builtin::is_constrained))[0] = 1;
            }
        }

        auto [body0_slot, _0] = object->geometries().create(body0);
        auto [body1_slot, _1] = object->geometries().create(body1);
        (void)_0;
        (void)_1;

        if(with_articulation)
        {
            std::vector<Vector3> l_pos0{rest.l0};
            std::vector<Vector3> l_pos1{rest.l1};
            std::vector<Vector3> r_pos0{rest.r0};
            std::vector<Vector3> r_pos1{rest.r1};
            std::vector<S<geometry::SimplicialComplexSlot>> l_slots{body0_slot};
            std::vector<S<geometry::SimplicialComplexSlot>> r_slots{body1_slot};
            std::vector<IndexT> l_ids{0};
            std::vector<IndexT> r_ids{0};
            std::vector<Float>  ratios{joint_ratio};

      // NOTE: a lambda-constructed value -- `SimplicialComplex`'s move-assignment is
      // deleted, so the two branches cannot assign into a declared local
      SimplicialComplex joint_geo = [&]() -> SimplicialComplex
      {
          if(prismatic)
          {
              uipc::constitution::AffineBodyPrismaticJoint joint;
              return joint.create_geometry(span<const Vector3>(l_pos0),
                                           span<const Vector3>(l_pos1),
                                           span<const Vector3>(r_pos0),
                                           span<const Vector3>(r_pos1),
                                           span<S<geometry::SimplicialComplexSlot>>(l_slots),
                                           span<IndexT>(l_ids),
                                           span<S<geometry::SimplicialComplexSlot>>(r_slots),
                                           span<IndexT>(r_ids),
                                           span<Float>(ratios));
          }
          uipc::constitution::AffineBodyRevoluteJoint joint;
          return joint.create_geometry(span<const Vector3>(l_pos0),
                                       span<const Vector3>(l_pos1),
                                       span<const Vector3>(r_pos0),
                                       span<const Vector3>(r_pos1),
                                       span<S<geometry::SimplicialComplexSlot>>(l_slots),
                                       span<IndexT>(l_ids),
                                       span<S<geometry::SimplicialComplexSlot>>(r_slots),
                                       span<IndexT>(r_ids),
                                       span<Float>(ratios));
      }();
      label_surface(joint_geo);
            auto [joint_slot, _j] = object->geometries().create(joint_geo);
            (void)_j;

            uipc::constitution::ExternalArticulationConstraint eac;
            std::vector<S<const GeometrySlot>> joint_geos{joint_slot};
            std::vector<IndexT>                indices{0};
            auto articulation = eac.create_geometry(span<S<const GeometrySlot>>(joint_geos),
                                                    span<IndexT>(indices));

            view(*articulation["joint_joint"]->find<Float>("mass"))[0] =
                zero_mass ? 0.0 : static_cast<Float>(articulation_mass);
            view(*articulation["joint"]->find<Float>("delta_theta_tilde"))[0] =
                delta_theta_tilde;

            object->geometries().create(articulation);
        }
        return scene;
    }

    /// The joint-link's *initial* transform (`LT` in the reference's `build_joint_info`).
    [[nodiscard]] Matrix3x3 link_A(bool left) const { return left ? A0 : A1; }
};

/// The articulation's per-joint basis *pair*, re-derived from the reference's
/// `build_joint_info` (both of its branches).
///
/// NOTE the revolute branch's asymmetry (the reason this helper takes both links at
/// once): the axle comes from the **left** edge's two body-local points, and both
/// sides' `[n ; b]` are that world axle's orthonormal frame mapped into *each* body's
/// own frame — the right edge's `r_position*` are NOT used by the revolute path (they
/// are by the prismatic one).
[[nodiscard]] std::pair<Dense6v, Dense6v> hand_basis_pair(bool            prismatic,
                                                         const Matrix3x3& Al,
                                                         const Vector3&   tl,
                                                         const Vector3&   l0,
                                                         const Vector3&   l1,
                                                         const Matrix3x3& Ar,
                                                         const Vector3&   tr,
                                                         const Vector3&   r0,
                                                         const Vector3&   r1)
{
    const Transform LTl{make_transform(Al, tl)};
    const Transform LTr{make_transform(Ar, tr)};
    const Matrix3x3 Rl = eigen_polar_rotation(Al);
    const Matrix3x3 Rr = eigen_polar_rotation(Ar);
    Dense6v         L, R;
    if(!prismatic)
    {
        Vector3 tw = Rl * (l1 - l0);
        Vector3 n, b;
        hand_orthonormal_basis(tw, n, b);
        L.segment<3>(0) = Rl.inverse() * n;
        L.segment<3>(3) = Rl.inverse() * b;
        R.segment<3>(0) = Rr.inverse() * n;
        R.segment<3>(3) = Rr.inverse() * b;
    }
    else
    {
        const Vector3 lt = (Rl * (l1 - l0)).normalized();
        const Vector3 lc = LTl * l0;
        const Vector3 rt = (Rr * (r1 - r0)).normalized();
        const Vector3 rc = LTr * r0;
        L.segment<3>(0) = LTl.inverse() * lc;
        L.segment<3>(3) = Rl.inverse() * lt;
        R.segment<3>(0) = LTr.inverse() * rc;
        R.segment<3>(3) = Rr.inverse() * rt;
    }
    return {L, R};
}
}  // namespace

namespace
{
// ===========================================================================
//  1) closed forms of the two generated articulation bodies
// ===========================================================================
TEST_CASE("abd_constraint_articulation_closed_forms", "[luisa][abd_constraint]")
{
    test_runtime();
    CaseScope scope{"abd_constraint_articulation_closed_forms"};
    auto&     c = scope.counter;

    // the shipped clamp's sweep count is a *value* on this path (both element
    // tensors are non-diagonal) -- pin it and measure its observability
    T64B_CHECK(c, Detail::kMakeSpdSweeps == 8);

    for(int kind = 0; kind < 2; ++kind)
    {
        const bool revolute = (kind == 0);

        for(int sample = 0; sample < 3; ++sample)
        {
            const Dense12v qk   = random_q();
            const Dense12v ql   = random_q();
            const Dense12v qk_t = qk + v12v(rand_vec(12, -0.15, 0.15));
            const Dense12v ql_t = ql + v12v(rand_vec(12, -0.15, 0.15));
            const Dense6v  bk   = v6v(rand_vec(6, -0.6, 0.6));
            const Dense6v  bl   = v6v(rand_vec(6, -0.6, 0.6));
            const Dense6v  bm   = v6v(rand_vec(6, -0.6, 0.6));
            const Dense6v  bn   = v6v(rand_vec(6, -0.6, 0.6));

            // ---- F ---------------------------------------------------------
            const Dense12v F_hand   = hand_F(revolute, bk, qk, bl, ql);
            const Dense12v F_shipped = shipped_F(revolute, bk, qk, bl, ql);
            T64B_CHECK(c, dev(F_shipped, F_hand) <= 1e-14);

            const Dense12v F_t = hand_F(revolute, bk, qk_t, bl, ql_t);

            // ---- DeltaTheta, both overloads --------------------------------
            const double dth_hand = hand_delta_theta_F(revolute, F_hand, F_t);
            T64B_CHECK(c, rel_err(shipped_delta_theta_F(revolute, F_hand, F_t), dth_hand) <= 1e-13);
            T64B_CHECK(c,
                       rel_err(shipped_delta_theta_q(revolute, bk, qk, qk_t, bl, ql, ql_t),
                               hand_delta_theta_q(revolute, bk, qk, qk_t, bl, ql, ql_t))
                           <= 1e-13);
            T64B_CHECK(c, std::abs(dth_hand) < 1.0);

            // ---- dDeltaTheta_dF vs a central difference ---------------------
            auto law_F = [&](const Dense12v& x)
            { return hand_delta_theta_F(revolute, x, F_t); };
            const Dense12v gF_fd = fd_grad(law_F, F_hand, 1e-6);
            T64B_CHECK(c, dev(shipped_d_delta_theta_dF(revolute, F_hand, F_t), gF_fd) <= 1e-8);

            // ---- ddDeltaTheta_ddF vs a central difference (h-sweep) ---------
            const Dense12 dd = shipped_dd_delta_theta_ddF(revolute, F_hand, F_t);
            for(double h : {1e-4, 1e-5})
            {
                T64B_CHECK(c, dev(dd, fd_hess(law_F, F_hand, h)) <= 1e-5);
            }
            // the element tensor is NOT diagonal -> `make_spd` on it projects
            {
                const double off = max_abs(Dense12(dd - Dense12(dd.diagonal().asDiagonal())));
                const double dia = max_abs(Dense12(dd.diagonal().asDiagonal()));
                T64B_CHECK(c, off > 0.5 * std::max(1e-300, max_abs(dd)));
                (void)dia;
            }
            // a Hessian of a scalar law is symmetric (the shipped body's own answer)
            T64B_CHECK(c, asymmetry(dd) <= 1e-9 * std::max(1.0, max_abs(dd)));

            // ---- dDeltaTheta_dQ vs a central difference ---------------------
            const Dense24v q24 = (Dense24v() << qk, ql).finished();
            auto           law_q = [&](const Dense24v& x)
            {
                return hand_delta_theta_q(revolute, bk, x.segment<12>(0), qk_t, bl,
                                          x.segment<12>(12), ql_t);
            };
            const Dense24v dQ_fd = fd_grad(law_q, q24, 1e-6);
            T64B_CHECK(c,
                       dev(shipped_d_delta_theta_dq(revolute, bk, qk, qk_t, bl, ql, ql_t), dQ_fd)
                           <= 1e-8);

            // ---- JT_G / JT_H_J against J^T G / J1^T H J2 --------------------
            const Dense12x24 J1 = fd_F_jacobian(revolute, bk, bl, q24);
            const Dense12x24 J2 = fd_F_jacobian(revolute, bm, bn, q24);

            const Dense12v G_F = v12v(rand_vec(12, -1.0, 1.0));
            T64B_CHECK(c, dev(shipped_jt_g(revolute, G_F, bk, bl), Dense24v(J1.transpose() * G_F))
                               <= 1e-8);

            Dense12 H_F = Dense12::Zero();
            for(int r = 0; r < 12; ++r)
                for(int cc = 0; cc < 12; ++cc)
                    H_F(r, cc) = rand_uniform(-1.0, 1.0);
            H_F = 0.5 * (H_F + H_F.transpose());

            // the two-operand question: (bm, bn) != (bk, bl), so a swapped operand
            // cannot pass
            T64B_CHECK(c,
                       dev(shipped_jt_h_j(revolute, H_F, bk, bl, bm, bn),
                           Dense24(J1.transpose() * H_F * J2))
                           <= 1e-8);
            T64B_CHECK(c,
                       dev(shipped_jt_h_j(revolute, H_F, bk, bl, bk, bl),
                           Dense24(J1.transpose() * H_F * J1))
                           <= 1e-8);
        }
    }

    // ---- degenerate inputs and the two pinned laws -------------------------
    {
        const Dense6v bk   = (Dense6v() << 0.3, -0.2, 0.1, 0.5, 0.1, -0.4).finished();
        const Dense6v bl   = (Dense6v() << -0.1, 0.4, 0.2, 0.2, -0.3, 0.6).finished();
        const Dense12v qk  = random_q();
        const Dense12v ql  = random_q();

        // D1: F == F_t -> DeltaTheta vanishes to the *generated* form's rounding (its
        // atan2 spells the (s, c) difference as a pair of products, so the exact-zero
        // case comes out at a few ulp of the operands, not at a bitwise 0)
        for(int kind = 0; kind < 2; ++kind)
        {
            const bool     revolute = (kind == 0);
            const Dense12v F        = hand_F(revolute, bk, qk, bl, ql);
            T64B_CHECK(c, std::abs(shipped_delta_theta_F(revolute, F, F)) <= 1e-15);
            T64B_CHECK(c, std::abs(shipped_delta_theta_q(revolute, bk, qk, qk, bl, ql, ql))
                               <= 1e-15);
            T64B_CHECK(c, hand_delta_theta_F(revolute, F, F) == 0.0);
        }

        // D2 (revolute only): a *common* rotation of both frames **at both states**
        // leaves theta (and hence DeltaTheta) invariant -- the revolute law's defining
        // property (its inputs are differences of the two frames' rotations)
        {
            const Dense3 R = Eigen::AngleAxisd(1.2, Dense3v(0.3, -0.7, 0.5).normalized())
                                 .toRotationMatrix();
            const Dense12v qk_t2 = qk + v12v(rand_vec(12, -0.15, 0.15));
            const Dense12v ql_t2 = ql + v12v(rand_vec(12, -0.15, 0.15));
            const Dense12v F     = hand_F(true, bk, qk, bl, ql);
            const Dense12v F2    = hand_F(true, bk, qk_t2, bl, ql_t2);
            Dense12v       F_rot, F2_rot;
            for(int blk = 0; blk < 4; ++blk)
            {
                F_rot.segment<3>(3 * blk)  = R * F.segment<3>(3 * blk);
                F2_rot.segment<3>(3 * blk) = R * F2.segment<3>(3 * blk);
            }
            T64B_CHECK(c, rel_err(hand_theta(F_rot), hand_theta(F)) <= 1e-13);
            T64B_CHECK(c,
                       rel_err(hand_delta_theta_F(true, F_rot, F2_rot),
                               hand_delta_theta_F(true, F, F2))
                           <= 1e-12);
            T64B_CHECK(c,
                       rel_err(shipped_delta_theta_F(true, F_rot, F2_rot),
                               shipped_delta_theta_F(true, F, F2))
                           <= 1e-12);
        }

        // D3: the *prismatic* law is linear in F's blocks -- a pure translation of
        // both attachment points by `d` moves DeltaTheta by `d . (v + q ... )`; the
        // oracle is the hand law, the check is that the shipped body agrees on a
        // state built from *arbitrary* (non-orthonormal) bases (the reference's laws
        // make no orthonormality assumption)
        {
            const Dense12v qk_t = qk + v12v(rand_vec(12, -0.2, 0.2));
            const Dense12v ql_t = ql + v12v(rand_vec(12, -0.2, 0.2));
            const Dense12v F    = hand_F(false, bk, qk, bl, ql);
            const Dense12v F_t  = hand_F(false, bk, qk_t, bl, ql_t);
            T64B_CHECK(c, rel_err(shipped_delta_theta_F(false, F, F_t),
                                  hand_delta_theta_F(false, F, F_t))
                               <= 1e-13);
        }
    }

    // ---- the make_spd clamp's observability (the sweep-count pin) ----------
    {
        const Dense12v F    = hand_F(true, (Dense6v() << 0.2, 0.3, -0.1, 0.4, -0.2, 0.5).finished(),
                                     random_q(),
                                     (Dense6v() << -0.3, 0.1, 0.2, 0.3, 0.4, -0.2).finished(),
                                     random_q());
        const Dense12v F_t  = F + v12v(rand_vec(12, -0.2, 0.2));
        const double   gth  = -3.25;  // a *negative* G_theta -> an indefinite HF
        const Dense12  dd   = shipped_dd_delta_theta_ddF(true, F, F_t);
        const Dense12  HF   = gth * dd;

        auto spd = [&](int sweeps)
        {
            lps::LMat<double, 12, 12> h = lm_of<12, 12>(HF);
            lps::make_spd_inplace(h, sweeps);
            return eig_of(h);
        };
        const Dense12 h0  = spd(0);
        const Dense12 h8  = spd(Detail::kMakeSpdSweeps);
        const Dense12 h20 = spd(20);
        const double  d08 = dev(h0, h8);
        const double  d820 = dev(h8, h20);
        std::printf("[t64b] make_spd sweeps: 0-vs-8 %.3e, 8-vs-20 %.3e, |HF|max %.6e\n",
                    d08, d820, max_abs(HF));
        // the *projection* is observable, and 8 is the layer's plateau
        T64B_CHECK(c, d08 > 1e-3);
        T64B_CHECK(c, d820 <= 1e-12);
        T64B_CHECK(c, min_eigenvalue(h8) >= -1e-9);
        T64B_CHECK(c, min_eigenvalue(HF) < -1e-3);  // raw HF really is indefinite
    }

    std::printf("[t64b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  2) central finite differences with an h-sweep
// ===========================================================================
TEST_CASE("abd_constraint_articulation_finite_difference", "[luisa][abd_constraint]")
{
    test_runtime();
    CaseScope scope{"abd_constraint_articulation_finite_difference"};
    auto&     c = scope.counter;

    for(int kind = 0; kind < 2; ++kind)
    {
        const bool revolute = (kind == 0);

        const Dense12v qk   = random_q();
        const Dense12v ql   = random_q();
        const Dense12v qk_t = qk + v12v(rand_vec(12, -0.12, 0.12));
        const Dense12v ql_t = ql + v12v(rand_vec(12, -0.12, 0.12));
        const Dense6v  bk   = v6v(rand_vec(6, -0.5, 0.5));
        const Dense6v  bl   = v6v(rand_vec(6, -0.5, 0.5));
        const Dense6v  bm   = v6v(rand_vec(6, -0.5, 0.5));
        const Dense6v  bn   = v6v(rand_vec(6, -0.5, 0.5));

        const Dense12v F   = hand_F(revolute, bk, qk, bl, ql);
        const Dense12v F_t = hand_F(revolute, bk, qk_t, bl, ql_t);
        const Dense24v q24 = (Dense24v() << qk, ql).finished();

        // ---- the h-sweep: the *gradient* is h-insensitive at 1e-8 ----------
        auto law_F = [&](const Dense12v& x)
        { return hand_delta_theta_F(revolute, x, F_t); };

        const Dense12v gF_shipped = shipped_d_delta_theta_dF(revolute, F, F_t);
        for(double h : {1e-4, 1e-5, 1e-6, 1e-7})
        {
            const double d = dev(gF_shipped, fd_grad(law_F, F, h));
            T64B_CHECK(c, d <= (h >= 1e-6 ? 1e-8 : 5e-7));
        }

        // ---- the Hessian's h-sweep: the FD's own roundoff floor -------------
        const Dense12 dd_shipped = shipped_dd_delta_theta_ddF(revolute, F, F_t);
        double       worst_h     = 0.0;
        for(double h : {1e-3, 1e-4, 1e-5})
        {
            const double d = dev(dd_shipped, fd_hess(law_F, F, h));
            worst_h        = std::max(worst_h, d);
            T64B_CHECK(c, d <= 1e-4);
        }
        std::printf("[t64b] %s ddDeltaTheta_ddF worst FD dev %.3e\n",
                    revolute ? "revolute" : "prismatic", worst_h);
        // the h = 1e-7 *hessian* is dominated by roundoff (eps/h^2 ~ 1e-2), and the
        // measurement is recorded, not asserted as "good": the point of the sweep is
        // that the *only* h-dependence is the FD's
        T64B_CHECK(c, dev(dd_shipped, fd_hess(law_F, F, 1e-7)) > worst_h);

        // ---- dDeltaTheta_dQ's h-sweep --------------------------------------
        auto law_q = [&](const Dense24v& x)
        {
            return hand_delta_theta_q(revolute, bk, x.segment<12>(0), qk_t, bl,
                                      x.segment<12>(12), ql_t);
        };
        const Dense24v dQ_shipped = shipped_d_delta_theta_dq(revolute, bk, qk, qk_t, bl, ql, ql_t);
        for(double h : {1e-5, 1e-6, 1e-7})
        {
            T64B_CHECK(c, dev(dQ_shipped, fd_grad(law_q, q24, h)) <= 1e-8);
        }

        // ---- the rule-34 basis question: the fan-outs take their operands in the
        // same order as their dE/dF siblings, so the *plain* comparison is the
        // agreement -- and the *swapped* one is not -------------------------
        const Dense12x24 J1 = fd_F_jacobian(revolute, bk, bl, q24);
        const Dense12x24 J2 = fd_F_jacobian(revolute, bm, bn, q24);
        Dense12 H_F         = Dense12::Zero();
        for(int r = 0; r < 12; ++r)
            for(int cc = 0; cc < 12; ++cc)
                H_F(r, cc) = rand_uniform(-1.0, 1.0);
        H_F = 0.5 * (H_F + H_F.transpose());
        const Dense24 plain  = shipped_jt_h_j(revolute, H_F, bk, bl, bm, bn);
        const Dense24 swap   = shipped_jt_h_j(revolute, H_F, bm, bn, bk, bl);
        const Dense24 want   = Dense24(J1.transpose() * H_F * J2);
        T64B_CHECK(c, dev(plain, want) <= 1e-8);
        // `J2 != J1` here, so the swapped call cannot coincide -- a non-vacuity pin
        T64B_CHECK(c, dev(plain, swap) > 1e-6);

        // ---- symmetry of the element Hessian, and the rank-1 structure of the
        // kernel's first term (the two flags the sweep pin rests on) ----------
        T64B_CHECK(c, asymmetry(dd_shipped) <= 1e-9 * std::max(1.0, max_abs(dd_shipped)));
        const double   gtheta = 2.5;
        const Dense12  hf     = gtheta * dd_shipped;
        const Dense24v dQv    = dQ_shipped;
        const Dense24  rank1  = (dQv * 1.75) * dQv.transpose();
        T64B_CHECK(c, min_eigenvalue(rank1) >= -1e-9);
        T64B_CHECK(c, std::abs(Eigen::JacobiSVD<Dense24>(rank1).singularValues()(1)) <= 1e-9);
        T64B_CHECK(c, (min_eigenvalue(hf) < 0.0 || max_abs(hf) == 0.0));
    }

    std::printf("[t64b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}
}  // namespace

namespace
{
/// `out[row, col] += blk` through find/emplace — `std::map::operator[]` would
/// default-construct an *uninitialised* Eigen 3x3 (§7.25 rule 17).
void add_block(BlockMap& out, int row, int col, const Matrix3x3& blk)
{
    auto key = std::make_pair(row, col);
    auto it  = out.find(key);
    if(it == out.end())
        out.emplace(key, blk);
    else
        it->second += blk;
}

/// One 12x12 diagonal block, `zero_out_lower`-ed and unpacked into 16 3x3 triplets
/// (the ABD assembly's convention, §7.24 rule 11).
void add_body_blocks(BlockMap& out, int body, const Dense12& diag12)
{
    Dense12 h = diag12;
    for(int a = 0; a < 4; ++a)
        for(int b = a + 1; b < 4; ++b)
            h.block<3, 3>(b * 3, a * 3).setZero();
    for(int a = 0; a < 4; ++a)
        for(int b = 0; b < 4; ++b)
            add_block(out, body * 4 + a, body * 4 + b, h.block<3, 3>(a * 3, b * 3));
}

/// The two bodies' dof offsets, read from the geometry metas the engine writes.
std::vector<IndexT> body_dof_offsets(World& world)
{
    std::vector<IndexT> out;
    auto visitor = uipc::backend::WorldVisitor{world};
    for(auto& slot : visitor.scene().geometries())
    {
        auto sc = slot->geometry().as<SimplicialComplex>();
        if(!sc)
            continue;
        auto cuid = sc->meta().find<U64>(builtin::constitution_uid);
        if(!cuid)
            continue;
        if(view(*cuid)[0] != kArapUID)
            continue;
        auto doff = sc->meta().find<IndexT>(builtin::dof_offset);
        REQUIRE(doff != nullptr);
        out.push_back(view(*doff)[0]);
    }
    return out;
}

// ---------------------------------------------------------------------------
//  a one-off probe kernel: the U64 argument order and `select`'s polarity
//
// The articulation's uid dispatch (`external_articulation_constraint_function.h`)
// rests on two ABI assumptions the *shipped* kernels cannot show directly: that a
// `U64` kernel argument arrives in the order the signature declares it, and that
// `scalar_detail::select_scalar(f, t, p)` is `p ? t : f` on the *device* (the host
// `lps::select` and LC's `lcrt::select` both document that polarity).  This tiny
// kernel (no `make_spd` chain -- the T6.3b3 deadlock does not apply) pins both.
// ---------------------------------------------------------------------------
auto& t64b_uid_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<U64>,  // out
                                                lcrt::Buffer<U64>,  // in
                                                U64,                // a
                                                U64,                // b
                                                uint32_t>(          // n
        [](lcrt::BufferVar<U64> out,
           lcrt::BufferVar<U64> in,
           lcrt::Var<U64>       a,
           lcrt::Var<U64>       b,
           lcrt::Var<uint32_t>  n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                out.write(0u, a);
                out.write(1u, b);
                // `p ? t : f`
                out.write(2u, lcrt::select(U64{10}, U64{20}, a == b));
                out.write(3u, lcrt::select(U64{10}, U64{20}, a != b));
                out.write(4u, lcrt::cast<U64>(a == b));
                // the *shipped* kernels' shape: a buffer *read* against a Var
                const auto v = in.read(i);
                out.write(5u, lcrt::select(U64{10}, U64{20}, v == a));
                out.write(6u, lcrt::cast<U64>(v == a));
                out.write(7u, lcrt::cast<U64>(v == b));
                out.write(8u, v);
            };
        });
    return k;
}

// ===========================================================================
//  3) the articulation's gradient/Hessian plumbing and its assembly fan-out
// ===========================================================================
TEST_CASE("abd_constraint_articulation_assembly", "[luisa][abd_constraint]")
{
    test_runtime();
    CaseScope scope{"abd_constraint_articulation_assembly"};
    auto&     c = scope.counter;

    for(int kind = 0; kind < 2; ++kind)
    {
        const bool revolute  = (kind == 0);
        const bool prismatic = !revolute;
        ArtScene   scene_data;
        const std::string tag = revolute ? "rev" : "pri";

        // ONE engine (a second Engine's teardown is where T6.3b1 recorded the
        // multi-engine hazard, and this case must not risk the rest of the filter):
        // the "base" assembly is the same world with the constraint's
        // `joint_joint_id_to_mass` device buffer zeroed between the two assemblies --
        // the extents depend on the *counts*, so the COO layout is identical and the
        // mass is the constraint's only value-level input.
        Engine engine{"luisa", workspace_path("uipc_luisa_abd_constraint_asm_" + tag)};
        World  world{engine};
        Scene  scene = scene_data.build("abd_constraint_asm", prismatic, true, false);
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T64BProbe::abd_linear != nullptr);
        REQUIRE(T64BProbe::articulation != nullptr);
        REQUIRE(T64BProbe::articulation->is_valid());

        auto* articulation = T64BProbe::articulation;
        REQUIRE(articulation->joint_joint_id_to_mass.size() == 1u);

        // The dof layout is written by `GlobalLinearSystem::init()` -- i.e. only after a
        // global-linear-system impl has been driven (§7.22 rule 12) -- so it must be
        // prepared *before* the per-geometry `builtin::dof_offset` metas are read: they
        // keep their `-1` default otherwise, and an oracle built from a `-1` offset
        // writes out of bounds (measured here: a heap corruption that killed the whole
        // filter).
        prepare_dof_layout(*T64BProbe::abd_linear);
        const auto offsets = body_dof_offsets(world);
        REQUIRE(offsets.size() == 2u);
        T64B_CHECK(c, offsets[0] == 0);
        T64B_CHECK(c, offsets[1] == 12);

        const auto mass_now = read_device(articulation->joint_joint_id_to_mass.cview());
        REQUIRE(mass_now.size() == 1u);
        T64B_CHECK(c, rel_err(mass_now[0], scene_data.articulation_mass) <= 1e-15);

        // one `advance()`: the *inter-body animator* has to step before a constraint's
        // slices reach the linear subsystem's assembly (§7.26 rule 12 / §7.27 rule 11 --
        // in the pre-step world a constraint's assembled contribution is invisible),
        // and the time integrator's update-state is exercised by the same frame.
        world.advance();
        REQUIRE(world.is_valid());

        // ---- (C) the hand oracle of the articulation's *own* contribution -----
        // NOTE: for the *revolute* body the R basis shares the L basis' frame rows (the
        // reference's own `use_local` branch), which is why the pair helper takes *both*
        // links at once.
        const auto [bk_pair, bl_pair] = hand_basis_pair(prismatic,
                                                        scene_data.A0, scene_data.t0,
                                                        scene_data.rest.l0, scene_data.rest.l1,
                                                        scene_data.A1, scene_data.t1,
                                                        scene_data.rest.r0, scene_data.rest.r1);
        const Dense6v  bk_use = bk_pair;
        const Dense6v  bl_use = bl_pair;
        const Dense12v qk     = q_of(scene_data.A0, scene_data.t0);
        const Dense12v ql     = q_of(scene_data.A1, scene_data.t1);
        const Dense12v qk_t   = q_of(scene_data.A0_prev, scene_data.t0_prev);
        const Dense12v ql_t   = q_of(scene_data.A1_prev, scene_data.t1_prev);

        const double dth = hand_delta_theta_q(revolute, bk_use, qk, qk_t, bl_use, ql, ql_t);
        const double x   = dth - static_cast<double>(scene_data.delta_theta_tilde);
        const double m   = static_cast<double>(scene_data.articulation_mass);
        const double G_theta = m * x;

        // the *shipped* dQ/dd/JT_H_J (pinned against FD in cases 1/2) so that this
        // case isolates the *plumbing*: the segmented reduce, the uid dispatch, the two
        // write patterns, the offset slicing and the assembly
        const Dense24v dQ = shipped_d_delta_theta_dq(revolute, bk_use, qk, qk_t, bl_use, ql, ql_t);
        const Dense12v F  = hand_F(revolute, bk_use, qk, bl_use, ql);
        const Dense12v F_t = hand_F(revolute, bk_use, qk_t, bl_use, ql_t);
        Dense12          HF = G_theta * shipped_dd_delta_theta_ddF(revolute, F, F_t);
        {
            lps::LMat<double, 12, 12> h = lm_of<12, 12>(HF);
            lps::make_spd_inplace(h, Detail::kMakeSpdSweeps);
            HF = eig_of(h);
        }
        const Dense24  jt  = shipped_jt_h_j(revolute, HF, bk_use, bl_use, bk_use, bl_use);
        const Dense24  H24 = (dQ * m) * dQ.transpose() + jt;
        const Dense24v G24 = dQ * G_theta;

        // ---- (C2) the *same* law from the device's own basis and states: this
        // separates a hand-basis mismatch from a hand-state mismatch -----------
        {
            const auto qs_d   = read_vec12(T64BProbe::abd->qs());
            const auto qp_d   = read_vec12(T64BProbe::abd->q_prevs());
            const auto lb     = read_device(articulation->joint_id_to_L_basis.cview());
            const auto rb     = read_device(articulation->joint_id_to_R_basis.cview());
            const auto bids   = read_device(articulation->joint_id_to_body_ids.cview());
            const auto rids   = read_device(articulation->joint_id_to_ref_q_prev_ids.cview());
            const auto rqp    = read_vec12(articulation->ref_q_prevs.cview());
            REQUIRE(lb.size() == 1u);
            REQUIRE(rqp.size() >= 2u);
            const Dense6v  lbv    = eig_of(lb[0]);
            const Dense6v  rbv    = eig_of(rb[0]);
            const Dense12v qk_d   = Dense12v(qs_d[static_cast<std::size_t>(bids[0](0))]);
            const Dense12v ql_d   = Dense12v(qs_d[static_cast<std::size_t>(bids[0](1))]);
            const Dense12v qk_t_d = Dense12v(rqp[static_cast<std::size_t>(rids[0](0))]);
            const Dense12v ql_t_d = Dense12v(rqp[static_cast<std::size_t>(rids[0](1))]);
            const double dth_dev_basis =
                hand_delta_theta_q(revolute, lbv, qk_d, qk_t_d, rbv, ql_d, ql_t_d);
            // the *other* body's law on the same inputs: if the device's answer matches
            // the wrong branch, the uid dispatch is broken
            const double dth_other_branch =
                hand_delta_theta_q(!revolute, lbv, qk_d, qk_t_d, rbv, ql_d, ql_t_d);
            const auto uid_d = read_device(articulation->joint_id_to_uid.cview());
            const auto dth_written = read_device(articulation->joint_id_to_delta_theta.cview());
            std::printf("[t64b] %s basis check: |hb-basis dev| L %.3e R %.3e, dth(dev basis) "
                        "%.9e vs dth(hand) %.9e, other branch %.9e, uid %llu, written %.9e\n",
                        tag.c_str(), dev(lbv, bk_use), dev(rbv, bl_use), dth_dev_basis, dth,
                        dth_other_branch,
                        static_cast<unsigned long long>(uid_d[0]), dth_written[0]);
            T64B_CHECK(c, dev(lbv, bk_use) <= 1e-12);
            T64B_CHECK(c, dev(rbv, bl_use) <= 1e-12);
            T64B_CHECK(c, rel_err(dth_dev_basis, dth) <= 1e-12);
            T64B_CHECK(c, std::abs(dth_other_branch - dth) > 1e-6);  // the pin's non-vacuity
        }

        // ---- (C3) the dispatch's two ABI assumptions, on the device ----------
        {
            lud::LDeviceBuffer<U64> probe(9);
            lud::LDeviceBuffer<U64> probe_in(4);
            {
                std::vector<U64> ids{18, 20, 18, 18};
                probe_in.view().copy_from(span<const U64>{ids});
                lud::main_stream().synchronize();
            }
            auto& k = t64b_uid_probe_kernel();
            gg::frame_stream().record(
                k(probe.view(), probe_in.cview(), U64{18}, U64{20}, 1u).dispatch(1u));
            const auto got = read_device(probe.cview());
            REQUIRE(got.size() == 9u);
            std::printf("[t64b] uid probe: a %llu b %llu eq-> %llu ne-> %llu flag %llu "
                        "(read==a)-> %llu eq %llu ne %llu v %llu\n",
                        static_cast<unsigned long long>(got[0]),
                        static_cast<unsigned long long>(got[1]),
                        static_cast<unsigned long long>(got[2]),
                        static_cast<unsigned long long>(got[3]),
                        static_cast<unsigned long long>(got[4]),
                        static_cast<unsigned long long>(got[5]),
                        static_cast<unsigned long long>(got[6]),
                        static_cast<unsigned long long>(got[7]),
                        static_cast<unsigned long long>(got[8]));
            // the arguments arrive in the declared order
            T64B_CHECK(c, got[0] == 18ull);
            T64B_CHECK(c, got[1] == 20ull);
            // `p ? t : f`: for a == b (false here) the *false* arm (10) is selected;
            // for a != b (true) the *true* arm (20) is
            T64B_CHECK(c, got[2] == 10ull);
            T64B_CHECK(c, got[3] == 20ull);
            T64B_CHECK(c, got[4] == 0ull);
            // ... and a *buffer read* compared against a Var behaves identically
            // (the shipped kernels' shape)
            T64B_CHECK(c, got[5] == 20ull);
            T64B_CHECK(c, got[6] == 1ull);
            T64B_CHECK(c, got[7] == 0ull);
            T64B_CHECK(c, got[8] == 18ull);
        }

        // ---- (F) the energy kernel, driven directly (the mass is still m) -----
        {
            const auto Es = run_inter_constraint_energy(*articulation,
                                                        *T64BProbe::abd,
                                                        T64BProbe::self->dt_attr(),
                                                        0.01,
                                                        1);
            REQUIRE(Es.size() == 1u);
            const double want_E = 0.5 * x * m * x;
            std::printf("[t64b] %s articulation energy: E %.9e (hand %.9e), dth %.9e x %.9e\n",
                        tag.c_str(), Es[0], want_E, dth, x);
            // the energy implies |x| exactly, so the *magnitude* is readable even if
            // the wrap's sign is not
            const double x_dev = std::sqrt(2.0 * Es[0] / m);
            std::printf("[t64b] %s |x| from E: %.9e (hand |x| %.9e)\n",
                        tag.c_str(), x_dev, std::abs(x));
            T64B_CHECK(c, rel_err(x_dev, std::abs(x)) <= 1e-13);
            T64B_CHECK(c, rel_err(Es[0], want_E) <= 1e-13);
            T64B_CHECK(c, std::abs(Es[0]) > 1e-6);
        }

        // ---- (G) the shipped k1/k2 kernels, driven directly (the [abd_joint_c]
        // section-G / §7.26 rule-12 harness) --------------------------------
        {
            const int  n_joint = 1;
            const int  n_jj    = 1;
            lud::LDeviceBuffer<int32_t> g_idx(2 * n_joint);
            lud::LDeviceBuffer<double>  g_val(12 * 2 * n_joint);
            lud::LDeviceBuffer<int32_t> h_row(3 * n_jj), h_col(3 * n_jj);
            lud::LDeviceBuffer<double>  h_val(144 * 3 * n_jj);
            {
                std::vector<int32_t> minus(3 * n_jj, -1);
                h_row.view().copy_from(span<const int32_t>{minus});
                h_col.view().copy_from(span<const int32_t>{minus});
                std::vector<double> zeros(144 * 3 * n_jj, 0.0);
                h_val.view().copy_from(span<const double>{zeros});
                lud::main_stream().synchronize();
            }
            auto Gv = lla::LDoubletVectorView<Float, 12>::from_handles_full(
                g_idx.view(), g_val.view(), std::uint32_t{2}, std::int32_t{0},
                std::int32_t{2}, std::int32_t{2}, std::int32_t{0}, std::uint32_t{2});
            auto Hv = lla::LTripletMatrixView<Float, 12, 12>::from_handles_full(
                h_row.view(), h_col.view(), h_val.view(), std::int32_t{8}, std::int32_t{8},
                std::int32_t{0}, std::int32_t{3}, std::int32_t{3}, std::int32_t{0},
                std::int32_t{0}, std::int32_t{8}, std::int32_t{8});

            gg::InterAffineBodyAnimator::Impl aimpl;
            aimpl.affine_body_dynamics = T64BProbe::abd;
            aimpl.dt_attr              = T64BProbe::self->dt_attr();
            aimpl.constraint_gradient_offsets_counts.resize(1);
            aimpl.constraint_gradient_offsets_counts.counts()[0] = static_cast<IndexT>(2);
            aimpl.constraint_gradient_offsets_counts.scan();
            aimpl.constraint_hessian_offsets_counts.resize(1);
            aimpl.constraint_hessian_offsets_counts.counts()[0] = static_cast<IndexT>(3);
            aimpl.constraint_hessian_offsets_counts.scan();
            gg::InterAffineBodyAnimator::GradientHessianInfo ghi{&aimpl, 0, 0.01, Gv, Hv, false};
            articulation->do_compute_gradient_hessian(ghi);
            gg::frame_stream().synchronize();

            // the segmented reduce's output, read straight off the constraint
            const auto gtheta = read_device(articulation->joint_id_to_G_theta.cview());
            REQUIRE(gtheta.size() == 1u);
            std::printf("[t64b] %s segmented reduce: G_theta %.9e (hand %.9e)\n",
                        tag.c_str(), gtheta[0], G_theta);
            T64B_CHECK(c, rel_err(gtheta[0], G_theta) <= 1e-12);

            const auto idx  = read_device(g_idx.cview());
            const auto vals = read_device(g_val.cview());
            const auto rows = read_device(h_row.cview());
            const auto cols = read_device(h_col.cview());
            const auto hvals = read_device(h_val.cview());

            T64B_CHECK(c, idx.size() == 2u);
            T64B_CHECK(c, idx[0] == 0);
            T64B_CHECK(c, idx[1] == 1);
            for(int s = 0; s < 2; ++s)
            {
                const Dense12v got = v12v(vals, s * 12);
                const Dense12v want = Dense12v(G24.segment<12>(s * 12));
                T64B_CHECK(c, dev(got, want) <= 1e-11);
                T64B_CHECK(c, max_abs(want) > 1e-6);
            }

            // the three 12x12 blocks: the two diagonal ones at the two body ids, the
            // off-diagonal one at the reference's `upper_LR` placement (bids(0) == 0 <
            // bids(1) == 1, so it is H24.block(0, 12))
            T64B_CHECK(c, rows.size() == 3u);
            auto triplet_block = [&](std::size_t t)
            {
                Dense12 blk;
                for(int cc = 0; cc < 12; ++cc)
                    for(int rr = 0; rr < 12; ++rr)
                        blk(rr, cc) = hvals[t * 144 + static_cast<std::size_t>(cc * 12 + rr)];
                return blk;
            };
            T64B_CHECK(c, (rows[0] == 0 && cols[0] == 0));
            T64B_CHECK(c, (rows[2] == 1 && cols[2] == 1));
            T64B_CHECK(c, (rows[1] == 0 && cols[1] == 1));
            T64B_CHECK(c, dev(triplet_block(0), Dense12(H24.block<12, 12>(0, 0))) <= 1e-10);
            T64B_CHECK(c, dev(triplet_block(1), Dense12(H24.block<12, 12>(0, 12))) <= 1e-10);
            T64B_CHECK(c, dev(triplet_block(2), Dense12(H24.block<12, 12>(12, 12))) <= 1e-10);
        }

        // ---- (A) the assembly: the full COO, then the same with mass = 0 ------
        const auto full = run_assembly(*T64BProbe::abd_linear);
        articulation->joint_joint_id_to_mass.fill(0.0);
        lud::main_stream().synchronize();
        const auto base = run_assembly(*T64BProbe::abd_linear);
        // restore (the case's later statements and the next iteration read it)
        articulation->joint_joint_id_to_mass.view().copy_from(span<const Float>{mass_now});
        lud::main_stream().synchronize();

        T64B_CHECK(c, base.triplet_count == full.triplet_count);
        T64B_CHECK(c, base.dof_count == full.dof_count);
        T64B_CHECK(c, base.dof_count == 24u);
        std::printf("[t64b] %s assembly: %zu triplets, %zu/%zu keys, |b_base| %.6e "
                    "|b_full| %.6e\n",
                    tag.c_str(), full.triplet_count, base.blocks.size(), full.blocks.size(),
                    base.gradient.norm(), full.gradient.norm());

        const auto      d_blocks = sub_blocks(full.blocks, base.blocks);
        Eigen::VectorXd d_grad   = full.gradient - base.gradient;

        // the COO residual
        BlockMap  want;
        const int b0 = 0, b1 = 1;
        add_body_blocks(want, b0, Dense12(H24.block<12, 12>(0, 0)));
        add_body_blocks(want, b1, Dense12(H24.block<12, 12>(12, 12)));
        {
            const Dense12 off = Dense12(H24.block<12, 12>(0, 12));
            for(int a = 0; a < 4; ++a)
                for(int b = 0; b < 4; ++b)
                    add_block(want, b0 * 4 + a, b1 * 4 + b, off.block<3, 3>(a * 3, b * 3));
        }
        // a missing key is a zero block (the COO stores a triplet per *written* 3x3
        // sub-block; a key the articulation leaves at exactly zero may or may not be
        // materialised -- both readings are accepted)
        double worst_key   = 0.0;
        double worst_extra = 0.0;
        for(const auto& [key, blk] : want)
        {
            auto it  = d_blocks.find(key);
            auto got = (it == d_blocks.end()) ? Matrix3x3::Zero() : it->second;
            worst_key = std::max(worst_key, dev(got, blk));
        }
        for(const auto& [key, blk] : d_blocks)
        {
            if(want.count(key))
                continue;
            worst_extra = std::max(worst_extra, max_abs(blk));
        }
        std::printf("[t64b] %s articulation COO: %zu keys (want %zu), worst |block| dev %.3e, "
                    "unexpected %.3e\n",
                    tag.c_str(), d_blocks.size(), want.size(), worst_key, worst_extra);
        T64B_CHECK(c, want.size() == 48u);
        T64B_CHECK(c, worst_key <= 1e-10);
        T64B_CHECK(c, worst_extra <= 1e-12);

        // the gradient residual (the same differencing, on the assembled `b`)
        Dense24v want_grad = Dense24v::Zero();
        want_grad.segment<12>(0)  = G24.segment<12>(0);
        want_grad.segment<12>(12) = G24.segment<12>(12);
        Eigen::VectorXd want_full = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(full.dof_count));
        for(int i = 0; i < 24; ++i)
            want_full(static_cast<Eigen::Index>(offsets[i / 12] + (i % 12))) = want_grad(i);
        const double grad_res = dev(d_grad, want_full);
        std::printf("[t64b] %s articulation gradient rel %.3e (|dG| %.6e, |want| %.6e)\n",
                    tag.c_str(), grad_res, d_grad.norm(), want_full.norm());
        std::printf("[t64b] %s dG  =", tag.c_str());
        for(int i = 0; i < 24; ++i)
            std::printf(" %.4e", d_grad(i));
        std::printf("\n");
        std::printf("[t64b] %s want=", tag.c_str());
        for(int i = 0; i < 24; ++i)
            std::printf(" %.4e", want_full(i));
        std::printf("\n");
        T64B_CHECK(c, want_full.norm() > 1.0);
        T64B_CHECK(c, d_grad.norm() > 1.0);
        T64B_CHECK(c, grad_res <= 1e-11);
    }

    std::printf("[t64b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  4) the soft transform constraint (uid 16): its law and its three kernels
// ===========================================================================
TEST_CASE("abd_constraint_soft_transform", "[luisa][abd_constraint]")
{
    test_runtime();
    CaseScope scope{"abd_constraint_soft_transform"};
    auto&     c = scope.counter;

    // ---- (A) the constraint-mass law, on the host -------------------------
    for(int s = 0; s < 4; ++s)
    {
        const double   m   = rand_uniform(0.5, 5.0);
        const Dense3v  mc  = Dense3v(rand_uniform(-1, 1), rand_uniform(-1, 1), rand_uniform(-1, 1));
        Dense3         Sdy = Dense3::Zero();
        for(int r = 0; r < 3; ++r)
            for(int cc = 0; cc < 3; ++cc)
                Sdy(r, cc) = rand_uniform(-1.0, 1.0);

        // the hand-built 12x12 from the documented (m, m c, m c c^T) layout is the
        // *shipped* T6.1a routine's answer (an independent re-derivation of `to_mat`)
        const Dense12 M_hand = hand_dyadic_to_mat(m, mc, Sdy);
        const gg::ABDJacobiDyadicMassT<double> mass{.m_mass = m,
                                                    .m_mass_times_x_bar = lv_of<3>(mc),
                                                    .m_mass_times_dyadic_x_bar = lm_of<3, 3>(Sdy)};
        T64B_CHECK(c, dev(M_hand, eig_of(gg::abd_dyadic_mass_to_mat<double>(mass))) <= 1e-15);

        const double s_t = rand_uniform(0.0, 3.0);
        const double s_r = rand_uniform(0.0, 3.0);
        const Dense12 Mc_hand    = hand_constraint_mass(m, mc, Sdy, s_t, s_r);
        const Dense12 Mc_shipped = shipped_constraint_mass(m, mc, Sdy, s_t, s_r);
        T64B_CHECK(c, dev(Mc_shipped, Mc_hand) <= 1e-14);

        // P1: s_t == s_r == s  =>  s M  (the parallel-axis split is exact)
        const Dense12 Ms = shipped_constraint_mass(m, mc, Sdy, s_t, s_t);
        T64B_CHECK(c, dev(Ms, Dense12(s_t * M_hand)) <= 1e-14);
        // P2: s_t = 1, s_r = 0  =>  M_cm (the point mass at the center of mass)
        const Dense12 Mcm = shipped_constraint_mass(m, mc, Sdy, 1.0, 0.0);
        T64B_CHECK(c,
                   dev(Mcm, hand_dyadic_to_mat(m, mc, mc * mc.transpose() / m)) <= 1e-14);
        // P3: s_t = 0, s_r = 1  =>  M - M_cm (the translation part is dropped, which is
        // what the frontend's RotatingMotor ({0, s}) asks for)
        T64B_CHECK(c,
                   dev(shipped_constraint_mass(m, mc, Sdy, 0.0, 1.0),
                       Dense12(M_hand
                               - hand_dyadic_to_mat(m, mc, mc * mc.transpose() / m)))
                       <= 1e-14);
        // P3b: s_t = 1, s_r = 1 => M
        T64B_CHECK(c, dev(shipped_constraint_mass(m, mc, Sdy, 1.0, 1.0), M_hand) <= 1e-14);
        // P4: the interpolation identity M_c(a,b) + M_c(b,a) == (a+b) M
        T64B_CHECK(c,
                   dev(Dense12(shipped_constraint_mass(m, mc, Sdy, s_t, s_r)
                               + shipped_constraint_mass(m, mc, Sdy, s_r, s_t)),
                       Dense12((s_t + s_r) * M_hand))
                       <= 1e-14);
    }

    // ---- (A2) a *physically consistent* dyadic triple is PSD ---------------
    {
        for(int s = 0; s < 3; ++s)
        {
            const double  m  = rand_uniform(0.5, 5.0);
            const Dense3v mc = Dense3v(rand_uniform(-1, 1), rand_uniform(-1, 1), rand_uniform(-1, 1));
            Dense3        B;
            for(int r = 0; r < 3; ++r)
                for(int cc = 0; cc < 3; ++cc)
                    B(r, cc) = rand_uniform(-1.0, 1.0);
            // the Schur-complement condition: S_dy - m c c^T ⪰ 0
            const Dense3 Sdy = mc * mc.transpose() / m + B * B.transpose();

            // the shipped 12x12 and the CM-only 12x12 are both PSD...
            T64B_CHECK(c, min_eigenvalue(shipped_constraint_mass(m, mc, Sdy, 1.0, 0.0)) >= -1e-9);
            T64B_CHECK(c, min_eigenvalue(shipped_constraint_mass(m, mc, Sdy, 0.0, 1.0)) >= -1e-9);
            // ... so `s_r M + (s_t - s_r) M_cm` is PSD whenever s_t >= s_r >= 0
            T64B_CHECK(c, min_eigenvalue(shipped_constraint_mass(m, mc, Sdy, 2.0, 0.5)) >= -1e-9);
            // P6: zero strengths -> the zero matrix
            T64B_CHECK(c, max_abs(shipped_constraint_mass(m, mc, Sdy, 0.0, 0.0)) == 0.0);
        }
    }

    // ---- (B) a real scene: the attribute contract + the three kernels ------
    {
        ArtScene scene_data;
        const std::string tag = "stc";
        Engine engine{"luisa", workspace_path("uipc_luisa_abd_constraint_stc")};
        World  world{engine};
        Scene  scene = scene_data.build("abd_constraint_stc", false, false, false, true);
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T64BProbe::soft_transform != nullptr);
        REQUIRE(T64BProbe::soft_transform->is_valid());
        REQUIRE(T64BProbe::animator != nullptr);

        auto* stc = T64BProbe::soft_transform;

        // the attribute contract, read through the *host* staging vectors the kernel
        // consumed
        T64B_CHECK(c, stc->h_constrained_bodies.size() == 2u);
        T64B_CHECK(c, stc->h_constrained_bodies[0] == 0);
        T64B_CHECK(c, stc->h_constrained_bodies[1] == 1);
        T64B_CHECK(c, stc->h_strength_ratios.size() == 2u);
        T64B_CHECK(c, rel_err(stc->h_strength_ratios[0](0), scene_data.stc_strength(0)) <= 1e-15);
        T64B_CHECK(c, rel_err(stc->h_strength_ratios[0](1), scene_data.stc_strength(1)) <= 1e-15);
        T64B_CHECK(c, stc->h_aim_transforms.size() == 2u);

        // the extent contract: one energy/gradient/Hessian entry per constrained body
        {
            gg::AffineBodyAnimator::Impl        impl;
            gg::AffineBodyAnimator::ReportExtentInfo info;
            stc->do_report_extent(info);
            T64B_CHECK(c, info.energy_count() == 2u);
            T64B_CHECK(c, info.gradient_count() == 2u);
            T64B_CHECK(c, info.hessian_count() == 2u);
            (void)impl;
        }

        // move the aim: `aim_transform` *shares* the instances' `transforms` storage
        // (the frontend's `share`), so writing it also moves the geometry's transform --
        // the dynamics' `q` is a copy taken at `init()`, so the constraint's `dq` becomes
        // non-zero.  The constraint re-reads the aim in `do_step`.
        const Matrix3x3 A_aim = Eigen::AngleAxisd(0.4, Vector3{0.2, -0.5, 0.8}.normalized())
                                    .toRotationMatrix()
                                * Vector3{1.1, 0.9, 1.05}.asDiagonal();
        const Vector3 t_aim{0.35, 0.1, -0.2};
        {
            auto visitor = uipc::backend::WorldVisitor{world};
            for(auto& slot : visitor.scene().geometries())
            {
                auto sc = slot->geometry().as<SimplicialComplex>();
                if(!sc)
                    continue;
                auto cuid = sc->meta().find<U64>(builtin::constitution_uid);
                if(!cuid || view(*cuid)[0] != kArapUID)
                    continue;
                auto aim = sc->instances().find<Matrix4x4>(builtin::aim_transform);
                REQUIRE(aim != nullptr);
                view(*aim)[0] = make_transform(A_aim, t_aim);
            }
            lud::main_stream().synchronize();
        }
        // one `advance()`: the ABD animator's step fan-out is what re-reads the aim
        // through the constraint's `do_step` (the engine's own `FilteredInfo`; the
        // animator's `m_impl` is private, so the test cannot build one itself)
        world.advance();
        REQUIRE(world.is_valid());
        // the state the kernels see
        const auto qs      = read_vec12(T64BProbe::abd->qs());
        const auto q_prevs = read_vec12(T64BProbe::abd->q_prevs());
        const auto masses  = read_device(T64BProbe::abd->body_masses());
        const auto fixed   = read_device(T64BProbe::abd->body_is_fixed());
        const auto aims    = read_vec12(stc->aim_transforms.cview());
        REQUIRE(qs.size() == 2u);
        REQUIRE(q_prevs.size() == 2u);
        REQUIRE(aims.size() == 2u);
        REQUIRE(fixed.size() == 2u);

        const double ratio = static_cast<double>(T64BProbe::global_animator == nullptr
                                                     ? 1.0
                                                     : T64BProbe::global_animator->substep_ratio());

        Dense12  hand_M[2];
        Dense12v hand_dq[2];
        for(int i = 0; i < 2; ++i)
        {
            const auto&   dm  = masses[static_cast<std::size_t>(i)];
            const Dense3v mc{gg::to_eigen(dm.m_mass_times_x_bar)};
            const Dense3  Sdy = gg::to_eigen(dm.m_mass_times_dyadic_x_bar);
            hand_M[i]         = hand_constraint_mass(dm.m_mass, mc, Sdy,
                                                     scene_data.stc_strength(0),
                                                     scene_data.stc_strength(1));
            const Vector12 q_aim = gg::transform_to_q(make_transform(A_aim, t_aim));
            hand_dq[i] = Dense12v(
                qs[static_cast<std::size_t>(i)]
                - hand_clamped_lerp(q_prevs[static_cast<std::size_t>(i)], q_aim, ratio));
        }
        // non-vacuity: the aim really moved, so dq is not zero
        T64B_CHECK(c, max_abs(hand_dq[0]) > 1e-3);

        // (B1) the energy kernel, driven directly
        {
            const auto Es = run_constraint_energy(*stc, *T64BProbe::abd,
                                                 T64BProbe::self->dt_attr(), 0.01, 2);
            REQUIRE(Es.size() == 2u);
            for(int i = 0; i < 2; ++i)
            {
                const double want = fixed[static_cast<std::size_t>(i)]
                                        ? 0.0
                                        : 0.5 * hand_dq[i].dot(hand_M[i] * hand_dq[i]);
                std::printf("[t64b] stc energy[%d] device %.9e hand %.9e\n", i, Es[i], want);
                T64B_CHECK(c, rel_err(Es[i], want) <= 1e-13);
            }
            T64B_CHECK(c, std::abs(Es[0]) > 1e-6);
        }

        // (B2) the gradient/Hessian kernels, driven directly over test-owned COO
        {
            const int  n       = 2;
            const int  n_trip  = 2;
            lud::LDeviceBuffer<int32_t> g_idx(n);
            lud::LDeviceBuffer<double>  g_val(12 * n);
            lud::LDeviceBuffer<int32_t> h_row(n_trip), h_col(n_trip);
            lud::LDeviceBuffer<double>  h_val(144 * n_trip);
            {
                std::vector<int32_t> minus(n_trip, -1);
                h_row.view().copy_from(span<const int32_t>{minus});
                h_col.view().copy_from(span<const int32_t>{minus});
                std::vector<double> zeros(144 * n_trip, 0.0);
                h_val.view().copy_from(span<const double>{zeros});
                lud::main_stream().synchronize();
            }
            auto Gv = lla::LDoubletVectorView<Float, 12>::from_handles_full(
                g_idx.view(), g_val.view(), std::uint32_t{static_cast<uint32_t>(n)},
                std::int32_t{0}, std::int32_t{n}, std::int32_t{n}, std::int32_t{0},
                std::uint32_t{static_cast<uint32_t>(n)});
            auto Hv = lla::LTripletMatrixView<Float, 12, 12>::from_handles_full(
                h_row.view(), h_col.view(), h_val.view(), std::int32_t{8}, std::int32_t{8},
                std::int32_t{0}, std::int32_t{n_trip}, std::int32_t{n_trip}, std::int32_t{0},
                std::int32_t{0}, std::int32_t{8}, std::int32_t{8});

            gg::AffineBodyAnimator::Impl impl;
            setup_animator_energy_impl(impl, *T64BProbe::abd, T64BProbe::self->dt_attr(), n);
            impl.global_animator = T64BProbe::global_animator;
            impl.constraint_gradient_offsets_counts.resize(1);
            impl.constraint_gradient_offsets_counts.counts()[0] = static_cast<IndexT>(n);
            impl.constraint_gradient_offsets_counts.scan();
            impl.constraint_hessian_offsets_counts.resize(1);
            impl.constraint_hessian_offsets_counts.counts()[0] = static_cast<IndexT>(n_trip);
            impl.constraint_hessian_offsets_counts.scan();
            gg::AffineBodyAnimator::ComputeGradientHessianInfo ghi{&impl, 0, 0.01, Gv, Hv, false};
            stc->do_compute_gradient_hessian(ghi);
            gg::frame_stream().synchronize();

            const auto idx = read_device(g_idx.cview());
            const auto val = read_device(g_val.cview());
            const auto row = read_device(h_row.cview());
            const auto col = read_device(h_col.cview());
            const auto hv  = read_device(h_val.cview());

            T64B_CHECK(c, idx.size() == 2u);
            T64B_CHECK(c, idx[0] == 0);
            T64B_CHECK(c, idx[1] == 1);
            for(int i = 0; i < 2; ++i)
            {
                const Dense12v want_G = fixed[static_cast<std::size_t>(i)]
                                            ? Dense12v::Zero()
                                            : Dense12v(hand_M[i] * hand_dq[i]);
                const Dense12v got_G  = v12v(val, i * 12);
                T64B_CHECK(c, dev(got_G, want_G) <= 1e-13);
            }
            for(int t = 0; t < 2; ++t)
            {
                T64B_CHECK(c, row[static_cast<std::size_t>(t)] == t);
                T64B_CHECK(c, col[static_cast<std::size_t>(t)] == t);
                Dense12 got;
                for(int cc = 0; cc < 12; ++cc)
                    for(int rr = 0; rr < 12; ++rr)
                        got(rr, cc) = hv[static_cast<std::size_t>(t) * 144
                                         + static_cast<std::size_t>(cc * 12 + rr)];
                const Dense12 want_H = fixed[static_cast<std::size_t>(t)]
                                           ? Dense12::Zero()
                                           : hand_M[t];
                T64B_CHECK(c, dev(got, want_H) <= 1e-13);
            }
        }
    }

    std::printf("[t64b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  5) the external body force pair (uid 666): extent, apply, shared uid
// ===========================================================================
TEST_CASE("abd_constraint_external_body_force", "[luisa][abd_constraint]")
{
    test_runtime();
    CaseScope scope{"abd_constraint_external_body_force"};
    auto&     c = scope.counter;

    ArtScene scene_data;
    Engine   engine{"luisa", workspace_path("uipc_luisa_abd_constraint_extforce")};
    World    world{engine};
    Scene    scene = scene_data.build("abd_constraint_extforce", false, false, false, false, true);
    world.init(scene);
    REQUIRE(world.is_valid());

    auto* constraint = T64BProbe::ext_body_force_constraint;
    auto* reporter   = T64BProbe::ext_body_force_reporter;
    REQUIRE(constraint != nullptr);
    REQUIRE(constraint->is_valid());
    REQUIRE(reporter != nullptr);
    REQUIRE(reporter->is_valid());

    // ---- (A) the constraint's table and its extent contract ---------------
    T64B_CHECK(c, constraint->m_impl.h_forces.size() == 2u);
    T64B_CHECK(c, constraint->m_impl.h_body_ids.size() == 2u);
    T64B_CHECK(c, constraint->m_impl.h_body_ids[0] == 0);
    T64B_CHECK(c, constraint->m_impl.h_body_ids[1] == 1);
    for(int i = 0; i < 2; ++i)
        T64B_CHECK(c, dev(constraint->m_impl.h_forces[static_cast<std::size_t>(i)],
                          scene_data.ext_force)
                           <= 1e-15);
    T64B_CHECK(c, constraint->forces().size() == 2u);
    T64B_CHECK(c, constraint->body_ids().size() == 2u);

    {
        gg::AffineBodyAnimator::ReportExtentInfo info;
        constraint->do_report_extent(info);
        T64B_CHECK(c, info.energy_count() == 0u);
        T64B_CHECK(c, info.gradient_count() == 0u);
        T64B_CHECK(c, info.hessian_count() == 0u);
        // the energy hook is the reference's empty one (the forces are applied through
        // the reporter, not through the linear system)
        gg::AffineBodyAnimator::Impl eimpl;
        setup_animator_energy_impl(eimpl, *T64BProbe::abd, T64BProbe::self->dt_attr(), 0);
        lud::LDeviceBuffer<Float>    Es(0);
        gg::AffineBodyAnimator::ComputeEnergyInfo einfo{&eimpl, 0, 0.01, Es.view()};
        constraint->do_compute_energy(einfo);
        T64B_CHECK(c, true);  // reached: the energy hook is a no-op
        // ... and `do_step` is the reference's re-read of the per-instance attributes
        T64B_CHECK(c, constraint->m_impl.affine_body_dynamics == T64BProbe::abd);
    }

    // ---- (B) the reporter's apply, *differenced* (the buffer may hold a stale
    // value, and the reference's kernel is an *atomic add*) -------------------
    gg::AffineBodyExternalForceManager::Impl impl;
    impl.affine_body_dynamics = T64BProbe::abd;
    {
        gg::AffineBodyExternalForceManager::ExternalForceInfo info{&impl};
        const auto before = read_vec12(T64BProbe::abd->body_external_forces());
        REQUIRE(before.size() == 2u);

        reporter->do_step(info);
        gg::frame_stream().synchronize();
        const auto after = read_vec12(T64BProbe::abd->body_external_forces());
        REQUIRE(after.size() == 2u);
        for(int i = 0; i < 2; ++i)
        {
            const Dense12v d = Dense12v(after[static_cast<std::size_t>(i)]
                                        - before[static_cast<std::size_t>(i)]);
            T64B_CHECK(c, dev(d, Dense12v(scene_data.ext_force)) <= 1e-13);
        }
    }

    // ---- (C) the *atomic* add: a synthetic table with duplicate body ids ----
    {
        // the constraint's device buffers are a public test seam (the T5.4b 671
        // precedent); two entries target body 0, one targets body 1
        const std::vector<gg::LVector12> forces{gg::to_luisa(scene_data.ext_force),
                                                gg::to_luisa(Vector12(2.0 * scene_data.ext_force)),
                                                gg::to_luisa(Vector12(-1.0 * scene_data.ext_force))};
        const std::vector<IndexT> ids{0, 0, 1};
        constraint->m_impl.forces.resize(forces.size());
        constraint->m_impl.forces.view().copy_from(span<const gg::LVector12>{forces});
        constraint->m_impl.body_ids.resize(ids.size());
        constraint->m_impl.body_ids.view().copy_from(span<const IndexT>{ids});
        lud::main_stream().synchronize();

        gg::AffineBodyExternalForceManager::ExternalForceInfo info{&impl};
        const auto before = read_vec12(T64BProbe::abd->body_external_forces());
        reporter->do_step(info);
        gg::frame_stream().synchronize();
        const auto after = read_vec12(T64BProbe::abd->body_external_forces());

        // body 0 received the *sum* of its two entries (3x), body 1 got -1x
        const Dense12v d0 = Dense12v(after[0] - before[0]);
        const Dense12v d1 = Dense12v(after[1] - before[1]);
        T64B_CHECK(c, dev(d0, Dense12v(3.0 * scene_data.ext_force)) <= 1e-13);
        T64B_CHECK(c, dev(d1, Dense12v(-1.0 * scene_data.ext_force)) <= 1e-13);
        std::printf("[t64b] ext-force atomic: |d0| %.6e |d1| %.6e (duplicate body 0)\n",
                    d0.norm(), d1.norm());
    }

    // ---- (D) the manager's `a = M^-1 F` and the uid triple ----------------
    {
        // restore the real table and let the *manager* run its own clear+step through
        // the engine path: one `advance()` (the scene's BDF1 prediction case checks the
        // consequence; here only the acceleration identity is pinned)
        world.advance();
        REQUIRE(world.is_valid());
        const auto F   = read_vec12(T64BProbe::abd->body_external_forces());
        const auto Acc = read_vec12(T64BProbe::abd->body_external_force_accs());
        const auto Minv = read_device(T64BProbe::abd->body_mass_invs());
        REQUIRE(F.size() == 2u);
        REQUIRE(Acc.size() == 2u);
        REQUIRE(Minv.size() == 2u);
        for(int i = 0; i < 2; ++i)
        {
            const Dense12v Fi    = Dense12v(F[static_cast<std::size_t>(i)]);
            const Dense12  Mi    = gg::to_eigen(Minv[static_cast<std::size_t>(i)]);
            const Dense12v want  = Dense12v(Mi * Fi);
            T64B_CHECK(c, dev(Dense12v(Acc[static_cast<std::size_t>(i)]), want) <= 1e-12);
            // ... and the force really is the frontend's value
            T64B_CHECK(c, dev(Fi, Dense12v(scene_data.ext_force)) <= 1e-13);
        }
    }

    std::printf("[t64b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}
}  // namespace

namespace
{
// ===========================================================================
//  6) uid ownership of 16 / 23 / 24 / 666, printed and REQUIREd
// ===========================================================================
TEST_CASE("abd_constraint_uid_ownership", "[luisa][abd_constraint]")
{
    test_runtime();
    CaseScope scope{"abd_constraint_uid_ownership"};
    auto&     c = scope.counter;

    ArtScene scene_data;
    Engine   engine{"luisa", workspace_path("uipc_luisa_abd_constraint_uid")};
    World    world{engine};
    Scene    scene = scene_data.build("abd_constraint_uid", false, true, false, true, true);
    world.init(scene);
    REQUIRE(world.is_valid());

    // ---- the four systems' own uids ---------------------------------------
    REQUIRE(T64BProbe::soft_transform != nullptr);
    REQUIRE(T64BProbe::articulation != nullptr);
    REQUIRE(T64BProbe::articulation_constitution != nullptr);
    REQUIRE(T64BProbe::articulation_integrator != nullptr);
    REQUIRE(T64BProbe::ext_body_force_constraint != nullptr);
    REQUIRE(T64BProbe::ext_body_force_reporter != nullptr);
    REQUIRE(T64BProbe::ext_force_manager != nullptr);

    std::printf("[t64b] uid map: soft_transform=%llu articulation=%llu constitution=%llu\n"
                "             ext_reporters: constraint=%llu reporter=%llu manager=%llu\n",
                static_cast<unsigned long long>(T64BProbe::soft_transform->uid()),
                static_cast<unsigned long long>(T64BProbe::articulation->uid()),
                static_cast<unsigned long long>(T64BProbe::articulation_constitution->uid()),
                static_cast<unsigned long long>(T64BProbe::ext_body_force_constraint->uid()),
                static_cast<unsigned long long>(T64BProbe::ext_body_force_reporter->uid()),
                static_cast<unsigned long long>(T64BProbe::ext_force_manager->uid()));

    REQUIRE(T64BProbe::soft_transform->uid() == 16ull);
    REQUIRE(T64BProbe::articulation->uid() == 24ull);
    REQUIRE(T64BProbe::articulation_constitution->uid() == 23ull);
    // the 666 *sharing*: the constraint's registry, the reporter's registry and the
    // manager's registry are three different ones (the T5.4b 671 / T6.3b2 667 /
    // T6.3b3 668 precedent) -- measured on all three, not assumed
    REQUIRE(T64BProbe::ext_body_force_constraint->uid() == 666ull);
    REQUIRE(T64BProbe::ext_body_force_reporter->uid() == 666ull);
    REQUIRE(T64BProbe::ext_force_manager->uid() == 666ull);

    // ---- the geometry-side ownership --------------------------------------
    auto visitor = uipc::backend::WorldVisitor{world};
    int  joint_geos = 0, articulation_geos = 0;
    U64  joint_uid = 0;
    for(auto& slot : visitor.scene().geometries())
    {
        auto& geo  = slot->geometry();
        auto  cuid = geo.meta().find<U64>(builtin::constitution_uid);
        if(!cuid)
            continue;
        const U64 uid = view(*cuid)[0];
        if(uid == 18ull || uid == 20ull)
        {
            ++joint_geos;
            joint_uid = uid;
            // the frontend's own shape, measured: a *plain* joint geometry (no
            // driving/limit instance applied) carries **no** `constraint_uids` at all --
            // the joint is an InterAffineBody *constitution*, and only the constraints
            // applied to it (19/21 drivers, 667/668 torques, 669/670 limits) add
            // entries.  The articulation's 24 is the *articulation geometry's* entry and
            // must never appear here.
            auto cuids = geo.meta().find<VectorXu64>(builtin::constraint_uids);
            std::printf("[t64b] joint geometry (uid %llu): constraint_uids %s\n",
                        static_cast<unsigned long long>(uid),
                        cuids ? "present" : "absent");
            if(cuids)
            {
                const auto& v = view(*cuids).front();
                for(auto id : v)
                    T64B_CHECK(c, id != 24ull);
            }
        }
        if(uid == 23ull)
        {
            ++articulation_geos;
            auto cuids = geo.meta().find<VectorXu64>(builtin::constraint_uids);
            REQUIRE(cuids != nullptr);
            const auto& v = view(*cuids).front();
            T64B_CHECK(c, v.size() == 1u);
            if(v.size() == 1u)
                T64B_CHECK(c, v(0) == 24ull);
        }
    }
    T64B_CHECK(c, joint_geos == 1);
    T64B_CHECK(c, articulation_geos == 1);

    // the articulation's own cross-reference, read off its host table: the joint it
    // looks up carries uid 18 (the scene builds the revolute joint)
    REQUIRE(T64BProbe::articulation->h_joint_id_to_uid.size() == 1u);
    std::printf("[t64b] articulation -> joint constitution uid %llu\n",
                static_cast<unsigned long long>(T64BProbe::articulation->h_joint_id_to_uid[0]));
    T64B_CHECK(c, T64BProbe::articulation->h_joint_id_to_uid[0] == 18ull);
    T64B_CHECK(c, joint_uid == 18ull);

    // ---- the uid-16 collision: the *real* constraint owns the group -------
    // luisa_abd_linear_test.cpp's synthetic claim (uid 27 since T6.4b) does not exist
    // in *this* scene, so the real `SoftTransformConstraint` must own exactly the two
    // marked bodies -- and the (resolved) collision there is documented at that file's
    // `kSoftTransformUID`.
    REQUIRE(T64BProbe::soft_transform->h_constrained_bodies.size() == 2u);
    T64B_CHECK(c, T64BProbe::soft_transform->h_constrained_bodies[0] == 0);
    T64B_CHECK(c, T64BProbe::soft_transform->h_constrained_bodies[1] == 1);
    // the external-force constraint owns the same two bodies through its *own* table
    T64B_CHECK(c, T64BProbe::ext_body_force_constraint->m_impl.h_body_ids.size() == 2u);

    // the *constitution* of the articulation is deliberately empty: its two extent
    // hooks report zero (called here for reachability; the counters themselves are a
    // private part of the manager's `*ExtentInfo` in this port, so the *effect* is
    // pinned where it is observable -- the assembly case's differenced COO is exactly
    // the constraint's, i.e. the constitution contributed nothing)
    {
        gg::InterAffineBodyConstitutionManager::EnergyExtentInfo einfo;
        T64BProbe::articulation_constitution->do_report_energy_extent(einfo);
        gg::InterAffineBodyConstitutionManager::GradientHessianExtentInfo ginfo;
        T64BProbe::articulation_constitution->do_report_gradient_hessian_extent(ginfo);
        auto impl = gg::InterAffineBodyConstitutionManager::Impl{};
        lud::LDeviceBuffer<Float> Es0(0);
        gg::InterAffineBodyConstitutionManager::EnergyInfo einfo0{
            &impl, 0, 0.01, Es0.view()};
        T64BProbe::articulation_constitution->do_compute_energy(einfo0);
        T64B_CHECK(c, true);  // reached: the constitution's hooks are the empty ones
    }

    std::printf("[t64b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  7) a real scene: the host bookkeeping and the BDF1 prediction by hand
// ===========================================================================
TEST_CASE("abd_constraint_scene", "[luisa][abd_constraint]")
{
    test_runtime();
    CaseScope scope{"abd_constraint_scene"};
    auto&     c = scope.counter;

    ArtScene scene_data;
    Engine   engine{"luisa", workspace_path("uipc_luisa_abd_constraint_scene")};
    World    world{engine};
    Scene    scene = scene_data.build("abd_constraint_scene", false, true, false, true, true);
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T64BProbe::articulation != nullptr);
    REQUIRE(T64BProbe::soft_transform != nullptr);
    REQUIRE(T64BProbe::ext_body_force_constraint != nullptr);

    auto* articulation = T64BProbe::articulation;

    // ---- the articulation's host bookkeeping ------------------------------
    T64B_CHECK(c, articulation->h_joint_id_to_art_id.size() == 1u);
    T64B_CHECK(c, articulation->h_joint_id_to_art_id[0] == 0);
    T64B_CHECK(c, articulation->h_joint_id_to_body_ids.size() == 1u);
    T64B_CHECK(c, articulation->h_joint_id_to_body_ids[0](0) == 0);
    T64B_CHECK(c, articulation->h_joint_id_to_body_ids[0](1) == 1);
    T64B_CHECK(c, articulation->h_joint_joint_id_to_mass.size() == 1u);
    T64B_CHECK(c, articulation->h_joint_joint_id_to_mass[0]
                     == static_cast<Float>(scene_data.articulation_mass));
    T64B_CHECK(c, articulation->h_joint_joint_id_to_joint_ij.size() == 1u);
    T64B_CHECK(c, articulation->h_joint_joint_id_to_joint_ij[0](0) == 0);
    T64B_CHECK(c, articulation->h_joint_joint_id_to_joint_ij[0](1) == 0);
    T64B_CHECK(c, articulation->h_joint_id_to_delta_theta_tilde.size() == 1u);
    T64B_CHECK(c, rel_err(articulation->h_joint_id_to_delta_theta_tilde[0],
                          scene_data.delta_theta_tilde)
                         <= 1e-15);

    // the two `ref_dof_prev` ids (the port's padded-buffer path): both links carried
    // the attribute, so both ids are real (>= 0) and the padded buffer holds the two
    // previous states
    T64B_CHECK(c, articulation->h_ref_q_prevs.size() == 2u);
    REQUIRE(articulation->h_joint_id_to_ref_q_prev_ids.size() == 1u);
    T64B_CHECK(c, articulation->h_joint_id_to_ref_q_prev_ids[0](0) >= 0);
    T64B_CHECK(c, articulation->h_joint_id_to_ref_q_prev_ids[0](1) >= 0);
    T64B_CHECK(c, articulation->ref_q_prevs.size() >= 2u);

    // the basis the constraint derived from the joint geometry equals the hand
    // re-derivation of the reference's `use_local` path
    {
        REQUIRE(articulation->h_joint_id_to_L_basis.size() == 1u);
        const auto [want_L, want_R] = hand_basis_pair(false,
                                                      scene_data.A0, scene_data.t0,
                                                      scene_data.rest.l0, scene_data.rest.l1,
                                                      scene_data.A1, scene_data.t1,
                                                      scene_data.rest.r0, scene_data.rest.r1);
          const Dense6v got_L  = Dense6v(articulation->h_joint_id_to_L_basis[0]);
          const Dense6v got_R  = Dense6v(articulation->h_joint_id_to_R_basis[0]);
        std::printf("[t64b] articulation basis: |L| %.6e |R| %.6e\n", got_L.norm(), got_R.norm());
        T64B_CHECK(c, dev(got_L, want_L) <= 1e-13);
        T64B_CHECK(c, dev(got_R, want_R) <= 1e-13);
    }

    // the extent contract (1 energy, 2 gradient doublets, 3 hessian blocks)
    {
        gg::InterAffineBodyAnimator::ReportExtentInfo info;
        articulation->do_report_extent(info);
        T64B_CHECK(c, info.energy_count() == 1u);
        T64B_CHECK(c, info.gradient_count() == 2u);
        T64B_CHECK(c, info.hessian_count() == 3u);
    }

    // the soft transform's and the external force's bookkeeping
    T64B_CHECK(c, T64BProbe::soft_transform->h_constrained_bodies.size() == 2u);
    T64B_CHECK(c, T64BProbe::ext_body_force_constraint->m_impl.h_body_ids.size() == 2u);
    T64B_CHECK(c, T64BProbe::ext_body_force_constraint->m_impl.h_body_ids[1] == 1);

    // ---- one advance: the BDF1 prediction and the force chain by hand -----
    const double dt = static_cast<double>(T64BProbe::self->dt());
    T64B_CHECK(c, dt > 0.0);
    const auto q_prev = read_vec12(T64BProbe::abd->q_prevs());
    const auto qv     = read_vec12(T64BProbe::abd->q_vs());
    const auto minv   = read_device(T64BProbe::abd->body_mass_invs());
    REQUIRE(q_prev.size() == 2u);
    REQUIRE(minv.size() == 2u);

    world.advance();
    REQUIRE(world.is_valid());
    world.retrieve();
    REQUIRE(world.is_valid());

    const auto q_tilde = read_vec12(T64BProbe::abd->q_tildes());
    const auto acc     = read_vec12(T64BProbe::abd->body_external_force_accs());
    REQUIRE(q_tilde.size() == 2u);
    REQUIRE(acc.size() == 2u);

    double worst_pred = 0.0;
    for(int i = 0; i < 2; ++i)
    {
        // `q_tilde = q_prev + (g + f_ext_acc) dt^2 + q_v dt` with g = 0 (the scene's
        // config) -- the reference's BDF1 prediction, re-derived here
        const Dense12v want = Dense12v(q_prev[static_cast<std::size_t>(i)])
                              + Dense12v(acc[static_cast<std::size_t>(i)]) * (dt * dt)
                              + Dense12v(qv[static_cast<std::size_t>(i)]) * dt;
        worst_pred = std::max(worst_pred,
                              dev(Dense12v(q_tilde[static_cast<std::size_t>(i)]), want));
        // ... and the acceleration really is M^-1 F (the 666 chain's last link)
        const Dense12  Mi     = gg::to_eigen(minv[static_cast<std::size_t>(i)]);
        const Dense12v F_hand = Dense12v(scene_data.ext_force);
        T64B_CHECK(c, dev(Dense12v(acc[static_cast<std::size_t>(i)]),
                          Dense12v(Mi * F_hand))
                         <= 1e-12);
    }
    std::printf("[t64b] BDF1 prediction worst dev %.3e (dt %.6e)\n", worst_pred, dt);
    T64B_CHECK(c, worst_pred <= 1e-12);
    // non-vacuity: the external force really moved the prediction
    T64B_CHECK(c, dev(Dense12v(q_tilde[0]), Dense12v(q_prev[0])) > 1e-9);

        // ---- the time integrator's own state, and the `write_scene` write-back
        {
            const Dense6v  bk = Dense6v(articulation->h_joint_id_to_L_basis[0]);
            const Dense6v  bl = Dense6v(articulation->h_joint_id_to_R_basis[0]);
            const Dense12v qk = q_of(scene_data.A0, scene_data.t0);
            const Dense12v ql = q_of(scene_data.A1, scene_data.t1);
            const Dense12v qk_t = q_of(scene_data.A0_prev, scene_data.t0_prev);
            const Dense12v ql_t = q_of(scene_data.A1_prev, scene_data.t1_prev);
            const double dth  = hand_delta_theta_q(true, bk, qk, qk_t, bl, ql, ql_t);

            // the update-state kernel's device output (the *same* buffer `write_scene`
            // publishes) -- the numerical statement
            const auto dth_dev = read_device(articulation->joint_id_to_delta_theta.cview());
            REQUIRE(dth_dev.size() == 1u);
            std::printf("[t64b] articulation update-state: device %.9e hand %.9e\n",
                        dth_dev[0], dth);
            T64B_CHECK(c, rel_err(dth_dev[0], dth) <= 1e-12);
            T64B_CHECK(c, std::abs(dth) > 1e-6);

            // the frontend `delta_theta` attribute is written by the constraint's
            // `write_scene`, which this port drives *only* from the engine's dump path
            // (`SimEngine::do_dump` -> `event_write_scene` -> the `on_write_scene`
            // actions -- `World::retrieve()` is a no-op until Phase 8, measured in
            // sim_engine.cpp), so the case drives it explicitly.
            bool dumped = false;
            try
            {
                dumped = world.dump();
            }
            catch(const std::exception& e)
            {
                std::printf("[t64b] world.dump() threw: %s\n", e.what());
            }
            T64B_CHECK(c, dumped);

            bool found = false;
            auto visitor2 = uipc::backend::WorldVisitor{world};
            for(auto& slot : visitor2.scene().geometries())
            {
                auto& geo  = slot->geometry();
                auto  cuid = geo.meta().find<U64>(builtin::constitution_uid);
                if(!cuid || view(*cuid)[0] != 23ull)
                    continue;
                auto dth_attr = geo["joint"]->find<Float>("delta_theta");
                REQUIRE(dth_attr != nullptr);
                const double exposed = view(*dth_attr)[0];
                std::printf("[t64b] articulation delta_theta written back: %.9e (hand %.9e)\n",
                            exposed, dth);
                T64B_CHECK(c, rel_err(exposed, dth) <= 1e-12);
                found = true;
            }
            T64B_CHECK(c, found);
        }

    std::printf("[t64b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  8) device vs host for every shipped routine of this batch
// ===========================================================================
TEST_CASE("abd_constraint_device_vs_host", "[luisa][abd_constraint]")
{
    test_runtime();
    CaseScope scope{"abd_constraint_device_vs_host"};
    auto&     c = scope.counter;

    ArtScene scene_data;
    Engine   engine{"luisa", workspace_path("uipc_luisa_abd_constraint_dvh")};
    World    world{engine};
    // all three systems in one scene, so every device-vs-host round below has its
    // system built (the external-force *reporter* requires the 666 constraint, which
    // only exists when a geometry asks for it -- measured: a scene without it makes the
    // reporter's `require<>` fail loudly)
    Scene scene = scene_data.build("abd_constraint_dvh", true, true, false, true, true);
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T64BProbe::articulation != nullptr);
    REQUIRE(T64BProbe::soft_transform != nullptr);

    auto* articulation = T64BProbe::articulation;
    auto* stc          = T64BProbe::soft_transform;

    // ---- the articulation's energy: the shipped kernel vs the host law -----
    {
        const auto qs      = read_vec12(T64BProbe::abd->qs());
        const auto q_prevs = read_vec12(T64BProbe::abd->q_prevs());
        const auto body_ids = read_device(articulation->joint_id_to_body_ids.cview());
        const auto l_basis  = read_device(articulation->joint_id_to_L_basis.cview());
        const auto r_basis  = read_device(articulation->joint_id_to_R_basis.cview());
        const auto uid      = read_device(articulation->joint_id_to_uid.cview());
        const auto dtt      = read_device(articulation->joint_id_to_delta_theta_tilde.cview());
        const auto mass     = read_device(articulation->joint_joint_id_to_mass.cview());
        const auto ij       = read_device(articulation->joint_joint_id_to_joint_ij.cview());
        const auto refids   = read_device(articulation->joint_id_to_ref_q_prev_ids.cview());
        const auto refq     = read_vec12(articulation->ref_q_prevs.cview());

        REQUIRE(body_ids.size() == 1u);
        REQUIRE(uid.size() == 1u);
        REQUIRE(dtt.size() == 1u);
        REQUIRE(mass.size() == 1u);
        REQUIRE(ij.size() == 1u);

        const auto Es = run_inter_constraint_energy(*articulation, *T64BProbe::abd,
                                                    T64BProbe::self->dt_attr(), 0.01, 1);
        REQUIRE(Es.size() == 1u);

        // the host law, from the *device* data the kernel itself reads
        auto prev_of = [&](int side) -> Vector12
        {
            const int rid = refids[0](side);
            return rid >= 0 ? refq[static_cast<std::size_t>(rid)]
                            : q_prevs[static_cast<std::size_t>(body_ids[0](side))];
        };
        const int      i0      = ij[0](0);
        const int      j0      = ij[0](1);
        const bool     prism   = (uid[static_cast<std::size_t>(i0)] == 20ull);
        const Dense6v  bk      = eig_of(l_basis[static_cast<std::size_t>(i0)]);
        const Dense6v  bl      = eig_of(r_basis[static_cast<std::size_t>(i0)]);
        const Dense12v qk      = Dense12v(qs[static_cast<std::size_t>(body_ids[static_cast<std::size_t>(i0)](0))]);
        const Dense12v ql      = Dense12v(qs[static_cast<std::size_t>(body_ids[static_cast<std::size_t>(i0)](1))]);
        const Dense12v qk_t    = Dense12v(prev_of(0));
        const Dense12v ql_t    = Dense12v(prev_of(1));
        const double   dth     = hand_delta_theta_q(!prism, bk, qk, qk_t, bl, ql, ql_t);
        const double   x       = dth - dtt[static_cast<std::size_t>(i0)];
        const double   m       = mass[0];
        const double   want    = 0.5 * x * m * (dth - dtt[static_cast<std::size_t>(j0)]);

        std::printf("[t64b] dvh articulation energy: device %.12e host %.12e\n", Es[0], want);
        T64B_CHECK(c, rel_err(Es[0], want) <= 1e-12);
    }

    // ---- the soft transform's energy: the shipped kernel vs the host law ---
    {
        const auto qs      = read_vec12(T64BProbe::abd->qs());
        const auto q_prevs = read_vec12(T64BProbe::abd->q_prevs());
        const auto masses  = read_device(T64BProbe::abd->body_masses());
        const auto fixed   = read_device(T64BProbe::abd->body_is_fixed());
        const auto idx     = read_device(stc->constrained_bodies.cview());
        const auto aims    = read_vec12(stc->aim_transforms.cview());
        const auto ratios  = read_device(stc->strength_ratios.cview());
        REQUIRE(idx.size() == 2u);
        REQUIRE(aims.size() == 2u);

        const auto Es = run_constraint_energy(*stc, *T64BProbe::abd,
                                             T64BProbe::self->dt_attr(), 0.01, 2);
        REQUIRE(Es.size() == 2u);
        const double ratio = static_cast<double>(T64BProbe::global_animator->substep_ratio());
        double       worst = 0.0;
        for(int k = 0; k < 2; ++k)
        {
            const int  b  = static_cast<int>(idx[static_cast<std::size_t>(k)]);
            const auto dm = masses[static_cast<std::size_t>(b)];
            const Dense3v mc{gg::to_eigen(dm.m_mass_times_x_bar)};
            const Dense3  Sdy = gg::to_eigen(dm.m_mass_times_dyadic_x_bar);
            const Dense12 M   = hand_constraint_mass(dm.m_mass, mc, Sdy,
                                                    ratios[static_cast<std::size_t>(k)](0),
                                                    ratios[static_cast<std::size_t>(k)](1));
            const Vector12 q_aim = aims[static_cast<std::size_t>(k)];
            const Dense12v dq    = Dense12v(qs[static_cast<std::size_t>(b)]
                                        - hand_clamped_lerp(q_prevs[static_cast<std::size_t>(b)],
                                                            q_aim, ratio));
            const double   want  = fixed[static_cast<std::size_t>(b)]
                                       ? 0.0
                                       : 0.5 * dq.dot(M * dq);
            worst = std::max(worst, rel_err(Es[static_cast<std::size_t>(k)], want));
        }
        std::printf("[t64b] dvh soft-transform energy worst rel %.3e\n", worst);
        // the mass law's *generic* routine uses `lps::pow`/divisions, so the round is
        // per-routine-bounded, not bitwise (§7.26 rule 17)
        T64B_CHECK(c, worst <= 1e-12);
    }

    // ---- the external-force scatter: the shipped kernel vs a host loop -----
    {
        gg::AffineBodyExternalForceManager::Impl impl;
        impl.affine_body_dynamics = T64BProbe::abd;
        const auto before = read_vec12(T64BProbe::abd->body_external_forces());
        gg::AffineBodyExternalForceManager::ExternalForceInfo info{&impl};
        T64BProbe::ext_body_force_reporter->do_step(info);
        gg::frame_stream().synchronize();
        const auto after = read_vec12(T64BProbe::abd->body_external_forces());

        const auto ids    = read_device(T64BProbe::ext_body_force_constraint->body_ids());
        const auto forces = read_vec12(T64BProbe::ext_body_force_constraint->forces());
        REQUIRE(ids.size() == forces.size());
        std::vector<Vector12> host = before;
        for(std::size_t k = 0; k < ids.size(); ++k)
            host[static_cast<std::size_t>(ids[k])] += forces[k];

        double worst = 0.0;
        for(std::size_t i = 0; i < host.size(); ++i)
            worst = std::max(worst, dev(Dense12v(after[i]), Dense12v(host[i])));
        std::printf("[t64b] dvh ext-force scatter worst dev %.3e\n", worst);
        T64B_CHECK(c, worst <= 1e-14);
    }

    std::printf("[t64b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  9) per-case assertion counts
// ===========================================================================
TEST_CASE("abd_constraint_summary", "[luisa][abd_constraint]")
{
    int total = 0;
    for(const auto& [name, checks] : g_case_counts)
    {
        std::printf("[t64b] case %-46s %5d checks\n", name.c_str(), checks);
        total += checks;
        CHECK(checks > 0);
    }
    std::printf("[t64b] summary: %d cases, %d checks\n",
                static_cast<int>(g_case_counts.size()), total);
}
}  // namespace
