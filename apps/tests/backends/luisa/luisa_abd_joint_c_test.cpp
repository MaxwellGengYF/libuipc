/********************************************************************
 * @file luisa_abd_joint_c_test.cpp
 * @brief T6.3b3 — the ABD joints of the LuisaCompute backend, batch 3 (final): the
 *        **revolute joint** (uid 18) and its three companions — the driving
 *        constraint (uid 19), the external-force pair (constraint + reporter, uid
 *        668) and the joint-limit constitution (uid 670) — plus the joint's dof
 *        reporter and time integrator.
 *
 * Every oracle here is derived independently *in this file*: the reference backend is
 * CUDA-only and cannot be host-included, so nothing is copied from it.  The oracles are
 *
 *  (a) **closed forms**, written out at the assertion.  With the pair dof vector
 *      `q = [qk ; ql]` (12 dofs each, `A(i,j) = q(3+3i+j)`), the joint-limit body's
 *      rest points `rest_positions = [l0 | l1 | r0 | r1]` and the four body-local
 *      frame rows `basis_k = [n_k ; b_k]`, `basis_l = [n_l ; b_l]`:
 *
 *        Faxis = [ A_k l0 + tk - (A_l r0 + tl) ;        (6 = two *coincident point
 *                  A_k l1 + tk - (A_l r1 + tl) ]          pairs* — the revolute joint
 *        Eaxis = 1/2 kappa |Faxis|^2                      leaves exactly the twist free
 *        dEaxisdFaxis   = kappa Faxis
 *        ddEaxisddFaxis = kappa I6                      (DIAGONAL — unlike T6.3b2)
 *        JaxisT_Gaxis / JaxisT_Haxis_Jaxis = J^T G / J^T H J with the constant
 *                        6x24 `Jaxis = dFaxis/dq`
 *
 *      The **driving** joint's element law is the `atan2` relative angle:
 *        u = A_k n_k, v = A_k b_k, p = A_l n_l, q = A_l b_l
 *        F01_q = [u ; v ; p ; q]                        (12)
 *        theta = atan2( (v.p - u.q)/2, (u.p + v.q)/2 )
 *        E     = 1/2 kappa (theta - theta_tilde)^2
 *        dEdF01 / ddEddF01 = the *analytic* atan2 chain rule, and the 12x12 is NOT
 *                        diagonal (the `make_spd` clamp on it IS a real projection —
 *                        the §7.26 rule-10 axis, re-pinned here)
 *      and the **joint limit** (uid 670) is the cubic penalty of
 *      `joint_limit_penalty.h` on `x = (current_angle - init_angle) + DeltaTheta`,
 *      where the generated `DeltaTheta` is the *wrapped* angle difference between the
 *      current and the previous state:
 *        F(bk, qk, bl, ql) = [A_k n_k ; A_k b_k ; A_l n_l ; A_l b_l]      (12)
 *        DeltaTheta(F, Ft) = wrap(theta(F) - theta(Ft))
 *        dDeltaTheta_dQ / dDeltaTheta_dF / ddDeltaTheta_ddF = the chain rule
 *        E = penalty(x), dE_dx, d2E_dx2 -> G = dE_dx * dx_dq,
 *        H = d2E_dx2 (dx_dq dx_dq^T) [+ J^T make_spd(dE_dx * ddDeltaTheta_ddF) J]
 *      The case re-implements the reference's *branchy* penalty as a host oracle
 *      (inside / above / below / zero-width), because the port's version is the
 *      branchless `select` spelling required by RULE M2 (see
 *      `joint_limit_penalty.h`), and asserts the two agree in every branch.
 *  (b) central finite differences over the 24 pair dofs with an h-sweep, the symmetry,
 *      the rule-34 basis question (the fan-outs take their element Hessians in the same
 *      order as their `dE/dF` siblings, so the *plain* comparison is the agreement) and
 *      the *diagonal* (joint) vs *non-diagonal* (driving, limit) element Hessians.
 *  (c) degenerates pinned from the source: a coincident rest pair (`E == 0` and
 *      `|G| == 0` *bitwise*), `kappa = 0`, a reflection, a singular `A`, a zero-width
 *      limit range, a limit `x` inside the range, a negative limit strength, the
 *      **free twist** (the revolute joint's defining property: rotating one body about
 *      the axle keeps `E_axis == 0` while `theta` moves by the twist angle) and the
 *      **winding wrap** of `DeltaTheta` past +-pi.
 *  (d) the fan-out into `ABDLinearSubsystem`'s assembly, isolated by *differencing*
 *      two real assemblies (the same 2-body scene with and without the joint
 *      geometries), the 16-way 3x3 unpacking, the `zero_out_lower` of the diagonal
 *      blocks, the reference's `upper_LR` on a *descending* pair, and the gradient
 *      scatter — plus the joint's/driving's/limit's *energy kernels* driven directly
 *      through the public `EnergyInfo` constructors.
 *  (e) a real scene: two ABD bodies, one revolute joint carrying a driving instance,
 *      an external-force (torque) instance and a limit instance, one `advance()` — the
 *      frontend `angle` attribute against the hand-derived relative angle, the
 *      `current_angles` state, the BDF1 prediction *including the external-force
 *      acceleration* the reporter chain produced, the dof layout, the joint's per-edge
 *      host bookkeeping against the hand-computed polar frames, the dof-reporter
 *      *adoption* of a user-authored winding, and the re-sync of a poisoned state.
 *  (f) device == host for every generic routine this batch ships (the joint's, the
 *      driving's, the limit's, the penalty's, `unwrap_angle`, `compute_relative_angle`
 *      and the external-force pair incl. `torque_to_F`) — *not* bitwise (the generated
 *      laws use `lps::pow`/`lps::atan2`, and `make_spd`'s Jacobi sweeps differ in
 *      associativity), so per-routine bounds are asserted and recorded.
 *  (g) uid ownership for 18/19/668/670, *proved*: the owners are printed and
 *      REQUIREd, the 668 constraint/reporter sharing is measured on both registries,
 *      and the limit's `JointUID` cross-reference (18) is pinned against the geometry.
 *
 * Coverage map:
 *  1) closed forms, both element laws, the derivatives, the clamps ......... abd_joint_c_closed_forms
 *  2) central FD (g/H/symmetry/h-sweep) + the basis question ............... abd_joint_c_finite_difference
 *  3) rest/zero-kappa/reflection/singular + the twist + the penalty ....... abd_joint_c_degenerate_inputs
 *  4) the assembly fan-out (COO, upper_LR, gradient) + the energy kernels .. abd_joint_c_assembly
 *  5) a real scene, one advance(), BDF1 + the ext-torque acc, the
 *     dof reporter / time integrator ...................................... abd_joint_c_scene
 *  6) device == host for the ported generic routines ...................... abd_joint_c_device_vs_host
 *  7) uid ownership of 18/19/668/670 + the 668 sharing .................... abd_joint_c_uid_ownership
 *  8) per-case assertion counts .......................................... abd_joint_c_summary
 *
 * UID OWNERSHIP.  Both constitutions (`18`, `670`) are sorted by uid in
 * `InterAffineBodyConstitutionManager::Impl::init` (a `std::ranges::sort`) and mapped
 * `uid -> index` in a loop, so with two equal-uid entries the **last** one owns the
 * geometry group; the registration order is `SimEngine::build_systems()`'s
 * demangled-type-name order (§7.23 rule 7).  `668` is shared by the *constraint*
 * registry and the *reporter* registry on purpose (the reference's own spelling, the
 * T5.4b `671` / T6.3b2 `667` precedent).  This file does not assume any of that: the
 * `uid_ownership`, `assembly` and `scene` cases print the owner they measured and
 * REQUIRE it, so a stolen group fails with the diagnostic above the assertion.
 ********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_revolute_joint.h>
#include <uipc/constitution/affine_body_driving_revolute_joint.h>
#include <uipc/constitution/affine_body_revolute_joint_external_force.h>
#include <uipc/constitution/affine_body_revolute_joint_limit.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

#include <affine_body/type_define.h>
#include <affine_body/utils.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/inter_affine_body_animator.h>
#include <affine_body/constitutions/affine_body_revolute_joint.h>
#include <affine_body/constitutions/affine_body_driving_revolute_joint.h>
#include <affine_body/constitutions/affine_body_revolute_joint_external_force.h>
#include <affine_body/constitutions/affine_body_revolute_joint_limit.h>
#include <affine_body/constitutions/affine_body_revolute_joint_function.h>
#include <affine_body/constitutions/joint_limit_penalty.h>
#include <linear_system/global_linear_system.h>
#include <joint_dof_system/global_joint_dof_manager.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <numbers>
#include <random>
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

template <typename Derived>
bool all_finite(const Eigen::MatrixBase<Derived>& m)
{
    return m.allFinite();
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

/// The 12-dof state of an affine body: `q = [t; a1; a2; a3]` with `a_i` row `i` of
/// `A` (`q(3 + 3i + j) == A(i,j)`), so `p = A x_bar + t`.
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
/// recorded in an open frame).
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

/// Upload a `std::vector<T>` to a fresh device buffer.
template <typename T>
lud::LDeviceBuffer<T> upload(const std::vector<T>& host)
{
    lud::LDeviceBuffer<T> buffer(host.size());
    if(!host.empty())
    {
        buffer.view().copy_from(span<const T>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}

template <typename L, typename H, typename F>
lud::LDeviceBuffer<L> upload_as(const std::vector<H>& host, F&& to_ldev)
{
    std::vector<L> staging;
    staging.reserve(host.size());
    for(auto& v : host)
        staging.push_back(to_ldev(v));
    return upload(staging);
}

// ---------------------------------------------------------------------------
//  Eigen <-> numerics-layer conversions for the generic routines
// ---------------------------------------------------------------------------
lps::LVec<double, 6> lv6(const Vector3& a, const Vector3& b)
{
    lps::LVec<double, 6> out;
    for(int i = 0; i < 3; ++i)
    {
        out(i)     = a(i);
        out(3 + i) = b(i);
    }
    return out;
}

lps::LVec<double, 3> lv3(const Vector3& a)
{
    lps::LVec<double, 3> out;
    for(int i = 0; i < 3; ++i)
        out(i) = a(i);
    return out;
}

/// Generic bridges between a fixed-size Eigen vector/matrix and the numerics layer's
/// `LVec`/`LMat` (the ported *generic* routines are `lps::`-typed, the oracles here are
/// Eigen-typed, so every crossing is explicit — never a reinterpret_cast).
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

template <int N>
[[nodiscard]] lps::LMat<double, N, N> lm_of(const Eigen::Matrix<double, N, N>& m)
{
    lps::LMat<double, N, N> out;
    for(int cidx = 0; cidx < N; ++cidx)
        for(int ridx = 0; ridx < N; ++ridx)
            out(ridx, cidx) = m(ridx, cidx);
    return out;
}

template <int N>
[[nodiscard]] Eigen::Matrix<double, N, N> eig_of(const lps::LMat<double, N, N>& m)
{
    Eigen::Matrix<double, N, N> out;
    for(int cidx = 0; cidx < N; ++cidx)
        for(int ridx = 0; ridx < N; ++ridx)
            out(ridx, cidx) = m(ridx, cidx);
    return out;
}

Dense3v to_eigen3(const lps::LVec<double, 3>& v)
{
    Dense3v out;
    for(int i = 0; i < 3; ++i)
        out(i) = v(i);
    return out;
}

Dense6v to_eigen6(const lps::LVec<double, 6>& v)
{
    Dense6v out;
    for(int i = 0; i < 6; ++i)
        out(i) = v(i);
    return out;
}

Dense6 to_eigen6(const lps::LMat<double, 6, 6>& m)
{
    Dense6 out;
    for(int c = 0; c < 6; ++c)
        for(int r = 0; r < 6; ++r)
            out(r, c) = m(r, c);
    return out;
}

Dense12v to_eigen12(const lps::LVec<double, 12>& v)
{
    Dense12v out;
    for(int i = 0; i < 12; ++i)
        out(i) = v(i);
    return out;
}

Dense12 to_eigen12(const lps::LMat<double, 12, 12>& m)
{
    Dense12 out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m(r, c);
    return out;
}

Dense24v to_eigen24(const lps::LVec<double, 24>& v)
{
    Dense24v out;
    for(int i = 0; i < 24; ++i)
        out(i) = v(i);
    return out;
}

Dense24 to_eigen24(const lps::LMat<double, 24, 24>& m)
{
    Dense24 out;
    for(int c = 0; c < 24; ++c)
        for(int r = 0; r < 24; ++r)
            out(r, c) = m(r, c);
    return out;
}

// ---------------------------------------------------------------------------
//  deterministic admissible random states
// ---------------------------------------------------------------------------
std::mt19937& rng()
{
    static std::mt19937 gen{20240918u};
    return gen;
}

double rand_uniform(double lo, double hi)
{
    return std::uniform_real_distribution<double>(lo, hi)(rng());
}

Eigen::JacobiSVD<Matrix3x3> svd_of(const Matrix3x3& A)
{
    return Eigen::JacobiSVD<Matrix3x3>(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
}

Matrix3x3 random_admissible_A()
{
    const Vector3 axis = Vector3{rand_uniform(-1.0, 1.0),
                                 rand_uniform(-1.0, 1.0),
                                 rand_uniform(-1.0, 1.0)}
                             .normalized();
    const Matrix3x3 R = Eigen::AngleAxisd(rand_uniform(-std::numbers::pi, std::numbers::pi),
                                          axis)
                            .toRotationMatrix();
    Matrix3x3 S;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            S(i, j) = rand_uniform(-0.25, 0.25);
    S = 0.5 * (S + S.transpose());
    Matrix3x3 A = R * (Matrix3x3::Identity() + S);
    auto  sv = svd_of(A);
    double smin = sv.singularValues()(2);
    if(smin < 0.35)
        A = A * (0.35 / smin);
    sv = svd_of(A);
    double smax = sv.singularValues()(0);
    if(smax > 2.5)
        A = A * (2.5 / smax);
    return A;
}

Vector3 rand_vec(double lo, double hi)
{
    return Vector3{rand_uniform(lo, hi), rand_uniform(lo, hi), rand_uniform(lo, hi)};
}

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

#define T63B3_CHECK(counter, cond) \
    do \
    { \
        ++(counter).checks; \
        CHECK(cond); \
    } while(false)

/// The *rotation* part of a linear map (`R = U V^T` of its SVD) — the independent
/// spelling of what Eigen's `Transform<...,Eigen::Affine>::rotation()` computes
/// (`computeRotationScaling`, a polar decomposition).  The scene's bodies have
/// `det A > 0`, so the SVD's `U V^T` is the proper rotation Eigen returns.
Matrix3x3 polar_rotation(const Matrix3x3& A)
{
    const auto sv = svd_of(A);
    return sv.matrixU() * sv.matrixV().transpose();
}

/// The reference's `orthonormal_basis(t, n, b)` (`affine_body/utils.h`), re-derived
/// here: `t` normalised in place, `test = ex` unless `|t . ex| > 0.9` (then `ey`),
/// `n = (t x test).normalized()`, `b = (t x n).normalized()`.  The case also asserts
/// that the *ported* routine reproduces this hand spelling bit for bit.
void hand_orthonormal_basis(Vector3& t, Vector3& n, Vector3& b)
{
    t = t.normalized();
    const Vector3 test = std::abs(t.dot(Vector3(1, 0, 0))) > 0.9 ? Vector3(0, 1, 0)
                                                                : Vector3(1, 0, 0);
    n = t.cross(test).normalized();
    b = t.cross(n).normalized();
}

// ===========================================================================
//  the joint's rest data in the reference's `rest_positions` layout
// ===========================================================================
/// `rest_positions = [l0 | l1 | r0 | r1]` — the four *body-local* points the frontend
/// writes (`l_position0/1`, `r_position0/1`), i.e. the two endpoints of the shared
/// axle in each body's own frame.
struct RevRest
{
    Vector3 l0 = Vector3::Zero();
    Vector3 l1 = Vector3::Zero();
    Vector3 r0 = Vector3::Zero();
    Vector3 r1 = Vector3::Zero();

    [[nodiscard]] Vector12 pack() const
    {
        Vector12 out;
        out.segment<3>(0) = l0;
        out.segment<3>(3) = l1;
        out.segment<3>(6) = r0;
        out.segment<3>(9) = r1;
        return out;
    }

    /// A *consistent* (zero-energy) pair through the world axle `P0 -> P1`, built from
    /// the two bodies' transformations: both bodies' points map to the same world
    /// points, so `Faxis == 0`.  `bump` offsets the right body's endpoints (a
    /// deliberate constraint violation, so the assembled checks are non-vacuous).
    void set_consistent(const Matrix3x3& Ai,
                        const Vector3&   ti,
                        const Matrix3x3& Aj,
                        const Vector3&   tj,
                        const Vector3&   P0,
                        const Vector3&   P1,
                        double           bump = 0.0)
    {
        // the reference's `LT.inverse() * P` (an affine inverse applied to a point):
        // `A^{-1} (P - t)` — spelled directly so the oracle does not need a 4x4
        // homogeneous product (and so it stays obviously the same formula)
        l0 = Ai.inverse() * (P0 - ti);
        l1 = Ai.inverse() * (P1 - ti);
        r0 = Aj.inverse() * (P0 + Vector3{bump, -0.5 * bump, 0.25 * bump} - tj);
        r1 = Aj.inverse() * (P1 + Vector3{-0.5 * bump, bump, 0.5 * bump} - tj);
    }

    /// The world axle direction the joint's `do_init` derives:
    /// `t = LT.rotation() * (l1 - l0)` (the *polar* rotation, un-normalised — the
    /// ported `orthonormal_basis` normalises in place, exactly like the reference).
    [[nodiscard]] Vector3 axle_world(const Matrix3x3& Ai) const
    {
        return polar_rotation(Ai) * (l1 - l0);
    }
};

/// The two bodies' body-local frame rows `[n_bar | b_bar]`, exactly as the joint's
/// `do_init` computes them (`L^{-1} n`, `L^{-1} b` with `L = polar(LT)` and `n`, `b`
/// the world orthonormal basis of the axle).
struct RevBasis
{
    Vector6 l_basis = Vector6::Zero();
    Vector6 r_basis = Vector6::Zero();

    void set(const Matrix3x3& Ai, const Matrix3x3& Aj, const Vector3& axle_world)
    {
        Vector3 t_w = axle_world;
        Vector3 n_w, b_w;
        hand_orthonormal_basis(t_w, n_w, b_w);  // normalises t_w in place
        const Matrix3x3 Li = polar_rotation(Ai).inverse();
        const Matrix3x3 Lj = polar_rotation(Aj).inverse();
        l_basis.segment<3>(0) = Li * n_w;
        l_basis.segment<3>(3) = Li * b_w;
        r_basis.segment<3>(0) = Lj * n_w;
        r_basis.segment<3>(3) = Lj * b_w;
    }
};

// ===========================================================================
//  the analytic oracle — the revolute joint (uid 18)
// ===========================================================================
/// `Faxis = [A_k l0 + tk - (A_l r0 + tl) ; A_k l1 + tk - (A_l r1 + tl)]`.
Dense6v Faxis_of(const RevRest& r, const Vector12& qk, const Vector12& ql)
{
    Dense6v out;
    const Matrix3x3 Ak = A_of(qk);
    const Matrix3x3 Al = A_of(ql);
    out.segment<3>(0) = Ak * r.l0 + t_of(qk) - Al * r.r0 - t_of(ql);
    out.segment<3>(3) = Ak * r.l1 + t_of(qk) - Al * r.r1 - t_of(ql);
    return out;
}

/// The *constant* pair Jacobian `dFaxis/dq` (6x24).  `k` is the row, `(a, b)` its
/// `(component, point)` split (`a = k % 3`, `b = k / 3`): the coefficient of `dti(a)`
/// is `+1`, of `dtj(a)` is `-1`, of `dA_i(a,m)` is the `b`-th rest point's `m`-th
/// component, and of `dA_j(a,m)` its negative.  NOTE the *pair* layout: `tj` occupies
/// 12..14, so `A_j` starts at 15 (§7.25 rule 9's trap).
Eigen::Matrix<double, 6, 24> Jaxis_of(const RevRest& r)
{
    Eigen::Matrix<double, 6, 24> J = Eigen::Matrix<double, 6, 24>::Zero();
    for(int k = 0; k < 6; ++k)
    {
        const int     a = k % 3;
        const int     b = k / 3;
        const Vector3 xb = (b == 0) ? r.l0 : r.l1;
        const Vector3 yb = (b == 0) ? r.r0 : r.r1;
        J(k, a)      = +1.0;
        J(k, 12 + a) = -1.0;
        for(int m = 0; m < 3; ++m)
        {
            J(k, 3 + 3 * a + m)  = +xb(m);
            J(k, 15 + 3 * a + m) = -yb(m);
        }
    }
    return J;
}

double Eaxis_of(const Dense6v& F, double kappa)
{
    return 0.5 * kappa * F.dot(F);
}

/// The whole joint law at a state (raw + clamped element Hessian, the chain, and the
/// shipped `H` built from the reference's `make_spd`-ed 6x6).
struct RevLocal
{
    Dense6v  F;
    double   E = 0.0;
    Dense6v  dEdF;
    Dense6   d2EdF2;
    Dense6   d2EdF2_spd;
    Dense24v G;
    Dense24  H_raw, H;
};

template <typename Derived>
typename Derived::PlainObject exact_spd(const Eigen::MatrixBase<Derived>& m)
{
    using Mat = typename Derived::PlainObject;
    const Mat A = m;
    Eigen::SelfAdjointEigenSolver<Mat> es(A);
    Eigen::Matrix<double, Eigen::Dynamic, 1> w = es.eigenvalues();
    for(Eigen::Index i = 0; i < w.size(); ++i)
        w(i) = std::max(w(i), 0.0);
    return es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
}

RevLocal rev_local(double kappa, const RevRest& r, const Vector12& qk, const Vector12& ql)
{
    RevLocal out;
    out.F      = Faxis_of(r, qk, ql);
    out.E      = Eaxis_of(out.F, kappa);
    out.dEdF   = kappa * out.F;
    out.d2EdF2 = kappa * Dense6::Identity();
    out.d2EdF2_spd = exact_spd(out.d2EdF2);
    const auto J = Jaxis_of(r);
    out.G        = J.transpose() * out.dEdF;
    out.H_raw    = J.transpose() * out.d2EdF2 * J;
    out.H        = J.transpose() * out.d2EdF2_spd * J;
    return out;
}

// ===========================================================================
//  the analytic oracle — the driving joint (uid 19) and the limit (uid 670)
// ===========================================================================
Dense12v F01q_of(const Vector6& l_basis, const Vector12& qk, const Vector6& r_basis, const Vector12& ql)
{
    const Matrix3x3 Ak = A_of(qk);
    const Matrix3x3 Al = A_of(ql);
    Dense12v        out;
    out.segment<3>(0) = Ak * l_basis.segment<3>(0);
    out.segment<3>(3) = Ak * l_basis.segment<3>(3);
    out.segment<3>(6) = Al * r_basis.segment<3>(0);
    out.segment<3>(9) = Al * r_basis.segment<3>(3);
    return out;
}

/// `theta = atan2(s, c)` with `s = (v.p - u.q)/2`, `c = (u.p + v.q)/2`.
double theta_of_F(const Dense12v& F)
{
    const Dense3v u = F.segment<3>(0);
    const Dense3v v = F.segment<3>(3);
    const Dense3v p = F.segment<3>(6);
    const Dense3v q = F.segment<3>(9);
    return std::atan2((v.dot(p) - u.dot(q)) / 2.0, (u.dot(p) + v.dot(q)) / 2.0);
}

/// The **analytic** gradient/Hessian of `theta` in `F` space.  `s` and `c` are
/// *quadratic* in `F` (they multiply two different blocks), so the chain rule needs
/// their second-derivative tensors as well:
///   H_ij = ( (c_j s_i - s_j c_i + c S2_ij - s C2_ij) D - N_i (2 s s_j + 2 c c_j) ) / D^2
/// (the transposed first cross term — a *measured* trap of this batch's oracle
/// development: it is off by O(1)).
void theta_grad_hess_F(const Dense12v& F, Dense12v& g, Dense12& H)
{
    const Dense3v u = F.segment<3>(0);
    const Dense3v v = F.segment<3>(3);
    const Dense3v p = F.segment<3>(6);
    const Dense3v q = F.segment<3>(9);
    const double  s = (v.dot(p) - u.dot(q)) / 2.0;
    const double  c = (u.dot(p) + v.dot(q)) / 2.0;
    const double  D = s * s + c * c;

    Dense12v ds, dc;
    ds.segment<3>(0) = -q / 2.0;
    ds.segment<3>(3) = p / 2.0;
    ds.segment<3>(6) = v / 2.0;
    ds.segment<3>(9) = -u / 2.0;
    dc.segment<3>(0) = p / 2.0;
    dc.segment<3>(3) = q / 2.0;
    dc.segment<3>(6) = u / 2.0;
    dc.segment<3>(9) = v / 2.0;

    Dense12 S2 = Dense12::Zero(), C2 = Dense12::Zero();
    for(int k = 0; k < 3; ++k)
    {
        S2(3 + k, 6 + k) = S2(6 + k, 3 + k) = 0.5;
        S2(0 + k, 9 + k) = S2(9 + k, 0 + k) = -0.5;
        C2(0 + k, 6 + k) = C2(6 + k, 0 + k) = 0.5;
        C2(3 + k, 9 + k) = C2(9 + k, 3 + k) = 0.5;
    }

    const Dense12v N = c * ds - s * dc;
    g               = N / D;
    H = (((ds * dc.transpose() - dc * ds.transpose()) + c * S2 - s * C2) * D
         - N * (2.0 * s * ds + 2.0 * c * dc).transpose())
        / (D * D);
}

double E_driving_of(const Dense12v& F, double kappa, double theta_tilde)
{
    return 0.5 * kappa * std::pow(theta_of_F(F) - theta_tilde, 2);
}

Dense12v dE_driving_of(const Dense12v& F, double kappa, double theta_tilde)
{
    Dense12v g;
    Dense12 H;
    theta_grad_hess_F(F, g, H);
    return kappa * (theta_of_F(F) - theta_tilde) * g;
}

Dense12 ddE_driving_of(const Dense12v& F, double kappa, double theta_tilde)
{
    Dense12v g;
    Dense12  H;
    theta_grad_hess_F(F, g, H);
    return kappa * (g * g.transpose() + (theta_of_F(F) - theta_tilde) * H);
}

/// The *constant* `dF01_q/dq` (12x24): `F01_q = [A_k n_k ; A_k b_k ; A_l n_l ; A_l b_l]`
/// has no translation dependence at all.
Eigen::Matrix<double, 12, 24> J01q_of(const Vector6& l_basis, const Vector6& r_basis)
{
    Eigen::Matrix<double, 12, 24> J = Eigen::Matrix<double, 12, 24>::Zero();
    for(int a = 0; a < 3; ++a)
        for(int m = 0; m < 3; ++m)
        {
            J(a, 3 + 3 * a + m)       = l_basis(0 + m);
            J(3 + a, 3 + 3 * a + m)   = l_basis(3 + m);
            J(6 + a, 15 + 3 * a + m)  = r_basis(0 + m);
            J(9 + a, 15 + 3 * a + m)  = r_basis(3 + m);
        }
    return J;
}

/// The wrapped difference in (-pi, pi]: `wrap(theta(F) - theta(Ft))`.
double wrap_angle(double a)
{
    return std::atan2(std::sin(a), std::cos(a));
}

double delta_theta_F(const Dense12v& F, const Dense12v& F_t)
{
    return wrap_angle(theta_of_F(F) - theta_of_F(F_t));
}

double delta_theta_q(const Vector6& l_basis,
                     const Vector12& qk,
                     const Vector12& qk_t,
                     const Vector6& r_basis,
                     const Vector12& ql,
                     const Vector12& ql_t)
{
    return delta_theta_F(F01q_of(l_basis, qk, r_basis, ql),
                         F01q_of(l_basis, qk_t, r_basis, ql_t));
}

/// The reference's branchy penalty (the *un-ported* form; the port is branchless).
struct Penalty
{
    double E = 0.0, dE = 0.0, d2E = 0.0;
};

Penalty penalty_ref(double x, double lower, double upper, double strength)
{
    Penalty out;
    if(x >= lower && x <= upper)
        return out;  // zeros
    const double w = upper - lower;
    if(w == 0.0)
    {
        if(x > upper)
        {
            const double d = x - upper;
            out.E = strength * d * d * d;
            out.dE = 3.0 * strength * d * d;
            out.d2E = 6.0 * strength * d;
        }
        else
        {
            const double d = lower - x;
            out.E = strength * d * d * d;
            out.dE = -3.0 * strength * d * d;
            out.d2E = 6.0 * strength * d;
        }
        return out;
    }
    const double inv_w = 1.0 / w;
    if(x > upper)
    {
        const double r = (x - upper) * inv_w;
        out.E = strength * r * r * r;
        out.dE = 3.0 * strength * r * r * inv_w;
        out.d2E = 6.0 * strength * r * inv_w * inv_w;
    }
    else
    {
        const double r = (lower - x) * inv_w;
        out.E = strength * r * r * r;
        out.dE = -3.0 * strength * r * r * inv_w;
        out.d2E = 6.0 * strength * r * inv_w * inv_w;
    }
    return out;
}

/// The whole limit law at a state (the reference's three kernels' arithmetic).
struct LimitLocal
{
    double   x = 0.0, E = 0.0, dE_dx = 0.0, d2E_dx2 = 0.0;
    Dense12v F, F_prev;
    Dense24v dx_dq;
    Dense24v G;
    Dense24  H;
};

LimitLocal limit_local(double           theta_prev,
                       double           lower,
                       double           upper,
                       double           strength,
                       const Vector6&   l_basis,
                       const Vector12&  qk,
                       const Vector12&  qk_t,
                       const Vector6&   r_basis,
                       const Vector12&  ql,
                       const Vector12&  ql_t)
{
    LimitLocal out;
    out.F      = F01q_of(l_basis, qk, r_basis, ql);
    out.F_prev = F01q_of(l_basis, qk_t, r_basis, ql_t);
    out.x      = theta_prev + delta_theta_F(out.F, out.F_prev);

    const auto p = penalty_ref(out.x, lower, upper, strength);
    out.E        = p.E;
    out.dE_dx    = p.dE;
    out.d2E_dx2  = p.d2E;

    Dense12v g_th;
    Dense12  H_th;
    theta_grad_hess_F(out.F, g_th, H_th);
    const auto J = J01q_of(l_basis, r_basis);
    out.dx_dq    = J.transpose() * g_th;
    out.G        = out.dE_dx * out.dx_dq;

    out.H = out.d2E_dx2 * (out.dx_dq * out.dx_dq.transpose());
    if(out.dE_dx != 0.0)
    {
        const Dense12     H_F = exact_spd(out.dE_dx * H_th);
        const Eigen::Matrix<double, 12, 24> Jm = J;
        out.H += Jm.transpose() * H_F * Jm;
    }
    return out;
}

/// The external-force reporter's arithmetic (the reference's axis step + `torque_to_F`
/// is called by the *kernel*; here the two axes and the pair are re-derived).
struct ExtForceLocal
{
    Dense3v  e_i, e_j;
    Dense12v F_i, F_j;
};

/// `e_i = (A_i e_bar_i).normalized()`, `e_j = (A_j e_bar_j).normalized()`, then
/// `e_j = (e_i + e_j)/2`, `e_i = -e_j` (the reference's symmetrisation).
ExtForceLocal ext_force_local(const Vector3& e_bar_i,
                              const Vector3& e_bar_j,
                              const Vector12& qk,
                              const Vector12& ql)
{
    ExtForceLocal out;
    out.e_i = (A_of(qk) * e_bar_i).normalized();
    out.e_j = (A_of(ql) * e_bar_j).normalized();
    out.e_j = 0.5 * (out.e_i + out.e_j);
    out.e_i = -out.e_j;
    return out;
}

/// Center-difference helpers over the 24 pair dofs.
constexpr double kFdH = 1e-6;

template <typename Eval>
Dense24v fd_gradient(const Eval& eval, const Dense24v& q, double h)
{
    Dense24v g;
    for(int k = 0; k < 24; ++k)
    {
        Dense24v qp = q, qm = q;
        qp(k) += h;
        qm(k) -= h;
        g(k) = (eval(qp) - eval(qm)) / (2 * h);
    }
    return g;
}

template <typename Grad>
Dense24 fd_hessian(const Grad& grad, const Dense24v& q, double h)
{
    Dense24 H;
    for(int k = 0; k < 24; ++k)
    {
        Dense24v qp = q, qm = q;
        qp(k) += h;
        qm(k) -= h;
        H.col(k) = (grad(qp) - grad(qm)) / (2 * h);
    }
    return 0.5 * (H + H.transpose());
}

Dense24v pack_pair(const Vector12& qk, const Vector12& ql)
{
    Dense24v q;
    q.segment<12>(0)  = qk;
    q.segment<12>(12) = ql;
    return q;
}
}  // namespace


// ===========================================================================
//  the probe: the built ABD/joint systems of one scene
// ===========================================================================
namespace
{
class T63B3Probe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::AffineBodyDynamics*                         abd          = nullptr;
    inline static gg::ABDLinearSubsystem*                         abd_linear   = nullptr;
    inline static gg::AffineBodyRevoluteJoint*                    joint        = nullptr;
    inline static gg::AffineBodyDrivingRevoluteJoint*             driving      = nullptr;
    inline static gg::AffineBodyRevoluteJointExternalForceConstraint* ext_constraint = nullptr;
    inline static gg::AffineBodyRevoluteJointExternalForce*       ext_reporter = nullptr;
    inline static gg::AffineBodyRevoluteJointLimit*               limit        = nullptr;
    inline static gg::AffineBodyRevoluteJointDofReporter*         dof_reporter = nullptr;
    inline static gg::AffineBodyRevoluteJointTimeIntegrator*      integrator   = nullptr;
    inline static gg::GlobalJointDofManager*                      joint_dof    = nullptr;
    inline static T63B3Probe*                                     self         = nullptr;

    /// The scene's `dt` -- the *same* attribute the joint kernels read.
    [[nodiscard]] Float dt()
    {
        auto attr = world().scene().config().find<Float>("dt");
        UIPC_ASSERT(attr, "T63B3Probe: the scene config has no `dt`");
        return attr->view()[0];
    }

    [[nodiscard]] S<const geometry::AttributeSlot<Float>> dt_attr()
    {
        return world().scene().config().find<Float>("dt");
    }

  private:
    void do_build() override
    {
        self           = this;
        abd            = &require<gg::AffineBodyDynamics>();
        abd_linear     = find<gg::ABDLinearSubsystem>();
        joint          = find<gg::AffineBodyRevoluteJoint>();
        driving        = find<gg::AffineBodyDrivingRevoluteJoint>();
        ext_constraint = find<gg::AffineBodyRevoluteJointExternalForceConstraint>();
        ext_reporter   = find<gg::AffineBodyRevoluteJointExternalForce>();
        limit          = find<gg::AffineBodyRevoluteJointLimit>();
        dof_reporter   = find<gg::AffineBodyRevoluteJointDofReporter>();
        integrator     = find<gg::AffineBodyRevoluteJointTimeIntegrator>();
        joint_dof      = find<gg::GlobalJointDofManager>();
    }
};
REGISTER_SIM_SYSTEM(T63B3Probe);
}  // namespace

namespace
{
// ---------------------------------------------------------------------------
//  scene helpers
// ---------------------------------------------------------------------------
constexpr Float kL     = 0.5;
constexpr Float kRho   = 1000.0;
constexpr Float kKappa = 1.0e8;

/// uid 1 = the frontend's "OrthoPotential", uid 2 = "ARAP".
constexpr U64 kArapUID = 2ull;

std::array<Vector3, 8> cube_vertices(Float L)
{
    return {Vector3{0, 0, 0} * L, Vector3{1, 0, 0} * L, Vector3{0, 1, 0} * L, Vector3{1, 1, 0} * L,
            Vector3{0, 0, 1} * L, Vector3{1, 0, 1} * L, Vector3{0, 1, 1} * L, Vector3{1, 1, 1} * L};
}

/// The six-tet decomposition of `[0,L]^3` (vertex index = `x + 2y + 4z`, the fan from
/// vertex 0 to vertex 7) -- the *positively oriented* fan (§7.24 rule 13).
std::array<Vector4i, 6> cube_tets()
{
    return {Vector4i{0, 1, 3, 7}, Vector4i{0, 3, 2, 7}, Vector4i{0, 2, 6, 7},
            Vector4i{0, 6, 4, 7}, Vector4i{0, 4, 5, 7}, Vector4i{0, 5, 1, 7}};
}

/// One ABD body: an `L`-cube of density `rho`, constitution uid `uid` and the
/// *initial* transform `(A, t)` -- which is also the ABD state `q`.
SimplicialComplex make_body(Float             kappa,
                            const Matrix3x3& A,
                            const Vector3&   t,
                            U64              uid,
                            const Matrix3x3& V = Matrix3x3::Zero(),
                            const Vector3&   v = Vector3::Zero())
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

Json joint_scene_config()
{
    auto config                      = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = false;
    config["integrator"]["type"]     = std::string{"bdf1"};
    // zero gravity: the BDF1 prediction is then `q + q_v dt + a_ext dt^2`
    config["gravity"] = Vector3{0.0, 0.0, 0.0};
    return config;
}

/// The 2-body scene every world case shares.
///
/// Both bodies sit at *rotated and stretched* initial transforms (so the joint's
/// `l_basis`/`r_basis` really are polar-rotation inverses of a world frame, T6.3b1
/// rule 3), and each joint's four body-local axle endpoints are arbitrary body-local
/// points.  `with_joint == false` builds the identical scene *without* the joint
/// geometry: the assembly case differences the two, isolating the joint's contribution
/// without needing an oracle for the ARAP body term.
struct RevScene
{
    Matrix3x3 A0 = Eigen::AngleAxisd(0.6, Vector3{0.3, -1.0, 0.5}.normalized()).toRotationMatrix()
                   * Vector3{1.25, 0.9, 1.1}.asDiagonal();
    Matrix3x3 A1 = Eigen::AngleAxisd(-1.1, Vector3{1.0, 0.4, -0.7}.normalized()).toRotationMatrix()
                   * Vector3{0.85, 1.3, 1.0}.asDiagonal();
    Vector3 t0 = Vector3{0.2, -0.1, 0.3};
    Vector3 t1 = Vector3{-0.4, 0.25, 0.15};
    Matrix3x3 V0 = 0.5 * Matrix3x3::Identity();
    Matrix3x3 V1 = 0.15 * Matrix3x3::Identity();
    Vector3   v0  = Vector3::Zero();
    Vector3   v1  = Vector3{0.1, -0.15, 0.05};

    /// The two pairs' *body-local* axle endpoints.  Pair 0 is the ascending pair
    /// (`k = body 0`, `l = body 1`), pair 1 the *descending* one (`k = body 1`,
    /// `l = body 0`) with its own data, so the `upper_LR` path is exercised.
    RevRest rest0, rest1;
    Float   bump0 = 0.05;  // the deliberate violations -> a non-zero joint energy
    Float   bump1 = -0.04;

    Float  ratio       = 100.0;  // the joint's strength ratio
    Float  dr_ratio    = 30.0;   // the driving instance's strength ratio
    Float  aim_angle   = 0.21;   // the driving target angle
    IndexT passive     = 0;
    IndexT constrained = 1;      // driving/is_constrained
    Float  ext_torque  = 12.5;   // external_torque
    IndexT ext_constrained = 1;  // external_torque/is_constrained
    Float  low         = -0.015;  // limit/lower (the scene's relative angle is ~0.019
                                  // rad, so both edges' limits *bind*: measured below)
    Float  high        = 0.015;   // limit/upper
    Float  lim_str     = 40.0;    // limit/strength

    RevScene()
    {
        // the world axle of pair 0 (a *stretched* body makes the polar/linear
        // distinction observable) and pair 1's own axle
        rest0.set_consistent(A0, t0, A1, t1, Vector3{0.12, -0.05, 0.20},
                             Vector3{0.45, 0.10, 0.06}, bump0);
        rest1.set_consistent(A1, t1, A0, t0, Vector3{-0.20, 0.15, 0.10},
                             Vector3{-0.02, 0.30, -0.12}, bump1);
    }

    [[nodiscard]] Scene build(const std::string& name,
                              bool               with_joint,
                              bool               with_driving = true) const
    {
        Json  config = joint_scene_config();
        Scene scene{config};
        auto  object = scene.objects().create(name);

        auto body0 = object->geometries().create(make_body(kKappa, A0, t0, kArapUID, V0, v0));
        auto body1 = object->geometries().create(make_body(kKappa, A1, t1, kArapUID, V1, v1));

        if(with_joint)
        {
            // pair 0 ascending (l = body 0), pair 1 descending (l = body 1)
            std::vector<S<geometry::SimplicialComplexSlot>> l_slots{body0.geometry, body1.geometry};
            std::vector<S<geometry::SimplicialComplexSlot>> r_slots{body1.geometry, body0.geometry};
            std::vector<IndexT>          l_ids{0, 0};
            std::vector<IndexT>          r_ids{0, 0};
            std::vector<Vector3>         l_pos0{rest0.l0, rest1.l0};
            std::vector<Vector3>         l_pos1{rest0.l1, rest1.l1};
            std::vector<Vector3>         r_pos0{rest0.r0, rest1.r0};
            std::vector<Vector3>         r_pos1{rest0.r1, rest1.r1};
            std::vector<Float>           ratios{ratio, ratio};

            uipc::constitution::AffineBodyRevoluteJoint joint;
            auto joint_geo = joint.create_geometry(span<const Vector3>(l_pos0),
                                                   span<const Vector3>(l_pos1),
                                                   span<const Vector3>(r_pos0),
                                                   span<const Vector3>(r_pos1),
                                                   span<S<geometry::SimplicialComplexSlot>>(l_slots),
                                                   span<IndexT>(l_ids),
                                                   span<S<geometry::SimplicialComplexSlot>>(r_slots),
                                                   span<IndexT>(r_ids),
                                                   span<Float>(ratios));

            // the driving instance (uid 19): the frontend's apply_to zeroes
            // `driving/is_constrained` and `aim_angle`, so the scene's values are
            // written afterwards (through the very attribute views the kernel reads).
            // NOTE: *every* edge is written -- `.front()` alone leaves the second
            // joint unconstrained, which silently zeroes its energy (the T6.3b2
            // lesson, re-applied).
            if(with_driving)
            {
                uipc::constitution::AffineBodyDrivingRevoluteJoint driving;
                driving.apply_to(joint_geo, dr_ratio);
                std::ranges::fill(
                    view(*joint_geo.edges().find<IndexT>("driving/is_constrained")),
                    constrained);
                std::ranges::fill(view(*joint_geo.edges().find<IndexT>("is_passive")),
                                  passive);
                std::ranges::fill(view(*joint_geo.edges().find<Float>("aim_angle")), aim_angle);
            }

            // the external-force instance (uid 668, constraint + reporter)
            uipc::constitution::AffineBodyRevoluteJointExternalForce ext_force_c;
            ext_force_c.apply_to(joint_geo, ext_torque);
            std::ranges::fill(
                view(*joint_geo.edges().find<IndexT>("external_torque/is_constrained")),
                ext_constrained);

            // the joint-limit instance (uid 670, an `extra_constitution_uid` on the
            // same geometry)
            uipc::constitution::AffineBodyRevoluteJointLimit limit;
            limit.apply_to(joint_geo, low, high, lim_str);

            object->geometries().create(joint_geo);
        }

        return scene;
    }
};

/// The `k`-side body index of an edge (`edge 0` is the ascending pair, `edge 1` the
/// descending one).
int edge_k_body(int edge)
{
    return edge == 0 ? 0 : 1;
}
int edge_l_body(int edge)
{
    return edge == 0 ? 1 : 0;
}

/// Both sides' body-local rest data + frames of one scene edge, exactly as the
/// joint's `do_init` derives them from the bodies' initial transforms and the local
/// endpoints: `t = polar(A_k) (l1 - l0)` (the reference's `LT.rotation() * (l1 - l0)`),
/// then `orthonormal_basis(t, n, b)` and `basis = polar(A)^{-1} [n ; b]`.
struct SceneFrames
{
    RevRest  rest;
    RevBasis basis;
};

SceneFrames scene_frames(const RevScene& s, int edge)
{
    const bool      k_is_body0 = (edge == 0);
    const Matrix3x3 Ak         = k_is_body0 ? s.A0 : s.A1;
    const Matrix3x3 Al         = k_is_body0 ? s.A1 : s.A0;
    SceneFrames     out;
    out.rest = k_is_body0 ? s.rest0 : s.rest1;
    out.basis.set(Ak, Al, out.rest.axle_world(Ak));
    return out;
}

/// One ABD body of a test scene: the dof offset, the uid, the device `q` and the
/// dynamics' dyadic-mass 12x12 map.
struct BodyInfo
{
    SizeT     dof_offset = 0;
    SizeT     body       = 0;  // = dof_offset / 12
    U64       uid        = 0;
    Vector12  q;
    Matrix3x3 A;
    Vector3   t;
    Dense12   mass = Dense12::Zero();
};

std::vector<BodyInfo> scene_bodies(World& world, gg::AffineBodyDynamics& abd)
{
    const auto qs     = read_vec12(abd.qs());
    const auto masses = read_device(abd.body_masses());

    std::vector<BodyInfo> out;
    auto visitor   = uipc::backend::WorldVisitor{world};
    auto geo_slots = visitor.scene().geometries();
    for(auto& slot : geo_slots)
    {
        auto sc = slot->geometry().as<SimplicialComplex>();
        if(!sc)
            continue;
        auto cuid = sc->meta().find<U64>(builtin::constitution_uid);
        if(!cuid)
            continue;
        const U64 uid = view(*cuid)[0];
        if(uid != kArapUID && uid != 1ull)
            continue;

        auto doff = sc->meta().find<IndexT>(builtin::dof_offset);
        REQUIRE(doff != nullptr);

        BodyInfo info;
        info.dof_offset = static_cast<SizeT>(view(*doff)[0]);
        info.body       = info.dof_offset / 12u;
        info.uid        = uid;
        REQUIRE(info.body < qs.size());
        REQUIRE(info.body < masses.size());
        info.q    = qs[info.body];
        info.A    = A_of(info.q);
        info.t    = t_of(info.q);
        info.mass = gg::abd_dyadic_mass_to_mat_eigen(masses[info.body]);
        out.push_back(info);
    }
    return out;
}

// ---------------------------------------------------------------------------
//  the assembly: a test-local GlobalLinearSystem::Impl (the T6.1b idiom)
// ---------------------------------------------------------------------------
using BlockMap = std::map<std::pair<int, int>, Matrix3x3>;

template <typename TripletT>
BlockMap read_blocks(const TripletT& A)
{
    const auto                n = static_cast<std::size_t>(A.triplet_count());
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

/// Writes the per-geometry dof offsets the way the engine does: the ABD subsystem's
/// `receive_init_dof_info` runs from `GlobalLinearSystem::init()`, so a test that wants
/// the dof layout must drive a (throwaway) test-local `Impl` first (§7.22 rule 12).
void prepare_dof_layout(gg::ABDLinearSubsystem& abd_linear)
{
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(abd_linear);
    impl.init();
}

struct AssemblyRun
{
    BlockMap        blocks;
    Eigen::VectorXd gradient;
    std::size_t     triplet_count   = 0;
    std::size_t     expected_extent = 0;
    std::size_t     dof_count       = 0;
    std::size_t     block_count     = 0;
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
    out.block_count = out.expected_extent / 16u;
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
//  the constitutions' / constraint's *energy* kernels, driven directly
// ---------------------------------------------------------------------------
void setup_manager_energy_impl(gg::InterAffineBodyConstitutionManager::Impl& impl,
                               gg::AffineBodyDynamics&                       abd,
                               S<const geometry::AttributeSlot<Float>>       dt_attr,
                               SizeT                                         n)
{
    impl.affine_body_dynamics = &abd;
    impl.dt_attr              = dt_attr;
    impl.constitution_energy_offsets_counts.resize(1);
    impl.constitution_energy_offsets_counts.counts()[0] = static_cast<IndexT>(n);
    impl.constitution_energy_offsets_counts.scan();
}

template <typename Constitution>
std::vector<double> run_constitution_energy(Constitution&                           c,
                                            gg::AffineBodyDynamics&                 abd,
                                            S<const geometry::AttributeSlot<Float>> dt_attr,
                                            Float                                   dt,
                                            SizeT                                   n)
{
    gg::InterAffineBodyConstitutionManager::Impl impl;
    setup_manager_energy_impl(impl, abd, dt_attr, n);

    lud::LDeviceBuffer<Float> Es(n);
    gg::InterAffineBodyConstitutionManager::EnergyInfo info{&impl, 0, dt, Es.view()};
    c.do_compute_energy(info);
    gg::frame_stream().synchronize();
    return read_device(Es.cview());
}

void setup_animator_energy_impl(gg::InterAffineBodyAnimator::Impl&      impl,
                                gg::AffineBodyDynamics&                 abd,
                                S<const geometry::AttributeSlot<Float>> dt_attr,
                                SizeT                                   n)
{
    impl.affine_body_dynamics = &abd;
    impl.dt_attr              = dt_attr;
    impl.constraint_energy_offsets_counts.resize(1);
    impl.constraint_energy_offsets_counts.counts()[0] = static_cast<IndexT>(n);
    impl.constraint_energy_offsets_counts.scan();
}

template <typename Constraint>
std::vector<double> run_constraint_energy(Constraint&                             c,
                                          gg::AffineBodyDynamics&                 abd,
                                          S<const geometry::AttributeSlot<Float>> dt_attr,
                                          Float                                   dt,
                                          SizeT                                   n)
{
    gg::InterAffineBodyAnimator::Impl impl;
    setup_animator_energy_impl(impl, abd, dt_attr, n);

    lud::LDeviceBuffer<Float> Es(n);
    gg::InterAffineBodyAnimator::ComputeEnergyInfo info{&impl, 0, dt, Es.view()};
    c.do_compute_energy(info);
    gg::frame_stream().synchronize();
    return read_device(Es.cview());
}

/// The joint's expected contribution to the global 3x3-block COO: the two diagonal
/// 12x12 blocks `zero_out_lower`-ed, the off-diagonal one placed by the reference's
/// `upper_LR` (i.e. at `(min(bi,bj), max(bi,bj))`, with the block content the
/// corresponding 12x12 sub-block of the *local* H), each unpacked into 16 3x3 triplets.
void add_pair_blocks(BlockMap& out, int bk, int bl, const Dense24& H_local)
{
    // NOTE: `out[key] += blk` would be undefined here -- `std::map::operator[]`
    // default-constructs the mapped `Matrix3x3` and Eigen does *not* zero a
    // default-constructed fixed-size matrix (measured: a ~1e257 entry).  The
    // find/emplace idiom is mandatory (§7.25 rule 17).
    auto add = [&](int r, int c, const Matrix3x3& blk)
    {
        auto key = std::make_pair(r, c);
        auto it  = out.find(key);
        if(it == out.end())
            out.emplace(key, blk);
        else
            it->second += blk;
    };

    const bool ascending = bk < bl;
    auto       add_body  = [&](int body, const Dense12& diag12)
    {
        Dense12 h = diag12;
        for(int a = 0; a < 4; ++a)
            for(int b = a + 1; b < 4; ++b)
                h.block<3, 3>(b * 3, a * 3).setZero();
        for(int a = 0; a < 4; ++a)
            for(int b = 0; b < 4; ++b)
                add(body * 4 + a, body * 4 + b, h.block<3, 3>(a * 3, b * 3));
    };
    add_body(bk, Dense12(H_local.block<12, 12>(0, 0)));
    add_body(bl, Dense12(H_local.block<12, 12>(12, 12)));

    const Dense12 off = ascending ? Dense12(H_local.block<12, 12>(0, 12))
                                  : Dense12(H_local.block<12, 12>(12, 0));
    const int     lo  = std::min(bk, bl);
    const int     hi  = std::max(bk, bl);
    for(int a = 0; a < 4; ++a)
        for(int b = 0; b < 4; ++b)
            add(lo * 4 + a, hi * 4 + b, off.block<3, 3>(a * 3, b * 3));
}

/// The scene's expected assembled contribution (the joint + the limit, and — only
/// after the animator has stepped — the driving constraint).  `driving_only`
/// restricts it to the driving term (how the scene case isolates it by differencing
/// two worlds).
///
/// NOTE (the T6.3b3 hazard, measured): the driving's `theta_tilde` needs the *joint's*
/// per-edge `init_angle` / `current_angle`, and the scene case initialises a second
/// world (the no-driving one) whose destruction invalidates `T63B3Probe`'s pointers —
/// so the angles are passed in explicitly rather than read through the probe.
BlockMap scene_expected_blocks(const RevScene&  s,
                               const Vector12    q_body[2],
                               const int         body_of[2],
                               double            k_joint,
                               double            k_drv,
                               const double      init_a[2],
                               const double      cur_a[2],
                               bool              with_driving,
                               bool              driving_only = false)
{
    BlockMap out;
    for(int edge = 0; edge < 2; ++edge)
    {
        const auto fr = scene_frames(s, edge);
        const auto kb = edge_k_body(edge);
        const auto lb = edge_l_body(edge);
        const int  bk = body_of[kb];
        const int  bl = body_of[lb];

        if(!driving_only)
        {
            const auto loc = rev_local(k_joint, fr.rest, q_body[kb], q_body[lb]);
            add_pair_blocks(out, bk, bl, loc.H);

            // the limit of this edge (the scene's bounds, with the *same* frames)
            const double th_prev = cur_a[edge] - init_a[edge];
            const auto   lim     = limit_local(th_prev,
                                               s.low - init_a[edge],
                                               s.high - init_a[edge],
                                               s.lim_str,
                                               fr.basis.l_basis,
                                               q_body[kb],
                                               q_body[kb],  // q_prev == q (no step yet)
                                               fr.basis.r_basis,
                                               q_body[lb],
                                               q_body[lb]);
            add_pair_blocks(out, bk, bl, lim.H);
        }

        if(with_driving)
        {
            const auto   basis = fr.basis;
            const double th    = theta_of_F(F01q_of(basis.l_basis, q_body[kb],
                                                    basis.r_basis, q_body[lb]));
            // theta_tilde = aim - init - (unwrap(theta, current - init) - theta)
            const double th_eff = cur_a[edge] - init_a[edge]
                                  + std::remainder(th - (cur_a[edge] - init_a[edge]),
                                                   2.0 * std::numbers::pi);
            const double tt = s.aim_angle - init_a[edge] - (th_eff - th);
            const auto   H  = to_eigen24(
                gg::abd_driving_revolute_joint_detail::body_hessian<double>(
                    k_drv,
                    gg::abd_driving_revolute_joint_detail::DrivingRevoluteJointRestT<double>{
                        .l_basis    = lv_of<6>(basis.l_basis),
                        .r_basis    = lv_of<6>(basis.r_basis)},
                    gg::to_luisa(q_body[kb]),
                    gg::to_luisa(q_body[lb]),
                    tt));
            add_pair_blocks(out, bk, bl, H);
        }
    }
    return out;
}
}  // namespace


// ===========================================================================
//  thin wrappers over the ported generic routines (host, S = double)
// ===========================================================================
namespace
{
gg::abd_revolute_joint_detail::RevoluteJointRestT<double> rj_rest(const RevRest& r)
{
    return gg::abd_revolute_joint_detail::RevoluteJointRestT<double>{
        .rest_positions = lv_of<12>(r.pack())};
}

double rj_E(double kappa, const RevRest& r, const Vector12& qk, const Vector12& ql)
{
    return gg::abd_revolute_joint_detail::body_energy<double>(
        kappa, rj_rest(r), gg::to_luisa(qk), gg::to_luisa(ql));
}

Dense24v rj_G(double kappa, const RevRest& r, const Vector12& qk, const Vector12& ql)
{
    return to_eigen24(gg::abd_revolute_joint_detail::body_gradient<double>(
        kappa, rj_rest(r), gg::to_luisa(qk), gg::to_luisa(ql)));
}

Dense24 rj_H(double kappa, const RevRest& r, const Vector12& qk, const Vector12& ql)
{
    return to_eigen24(gg::abd_revolute_joint_detail::body_hessian<double>(
        kappa, rj_rest(r), gg::to_luisa(qk), gg::to_luisa(ql)));
}

Dense6v rj_Faxis(const RevRest& r, const Vector12& qk, const Vector12& ql)
{
    return to_eigen6(gg::abd_revolute_joint_detail::faxis<double>(
        rj_rest(r), gg::to_luisa(qk), gg::to_luisa(ql)));
}

gg::abd_driving_revolute_joint_detail::DrivingRevoluteJointRestT<double> drj_rest(
    const RevBasis& b)
{
    return gg::abd_driving_revolute_joint_detail::DrivingRevoluteJointRestT<double>{
        .l_basis = lv_of<6>(b.l_basis), .r_basis = lv_of<6>(b.r_basis)};
}

Dense12v drj_F01q(const RevBasis& b, const Vector12& qk, const Vector12& ql)
{
    return to_eigen12(gg::abd_driving_revolute_joint_detail::f01_q<double>(
        drj_rest(b), gg::to_luisa(qk), gg::to_luisa(ql)));
}

double drj_theta(const RevBasis& b, const Vector12& qk, const Vector12& ql)
{
    return gg::abd_driving_revolute_joint_detail::theta_of<double>(
        lv_of<12>(drj_F01q(b, qk, ql)));
}

double drj_E(double kappa, const RevBasis& b, const Vector12& qk, const Vector12& ql, double tt)
{
    return gg::abd_driving_revolute_joint_detail::body_energy<double>(
        kappa, drj_rest(b), gg::to_luisa(qk), gg::to_luisa(ql), tt);
}

Dense24v drj_G(double kappa, const RevBasis& b, const Vector12& qk, const Vector12& ql, double tt)
{
    return to_eigen24(gg::abd_driving_revolute_joint_detail::body_gradient<double>(
        kappa, drj_rest(b), gg::to_luisa(qk), gg::to_luisa(ql), tt));
}

Dense24 drj_H(double kappa, const RevBasis& b, const Vector12& qk, const Vector12& ql, double tt)
{
    return to_eigen24(gg::abd_driving_revolute_joint_detail::body_hessian<double>(
        kappa, drj_rest(b), gg::to_luisa(qk), gg::to_luisa(ql), tt));
}

double erj_delta_theta(const RevBasis& b,
                       const Vector12& qk,
                       const Vector12& qk_t,
                       const Vector12& ql,
                       const Vector12& ql_t)
{
    return gg::abd_revolute_joint_limit_detail::delta_theta<double>(
        lv_of<6>(b.l_basis), gg::to_luisa(qk), gg::to_luisa(qk_t),
        lv_of<6>(b.r_basis), gg::to_luisa(ql),
        gg::to_luisa(ql_t));
}

Dense24v erj_ddelta_dq(const RevBasis& b,
                       const Vector12& qk,
                       const Vector12& qk_t,
                       const Vector12& ql,
                       const Vector12& ql_t)
{
    return to_eigen24(gg::abd_revolute_joint_limit_detail::d_delta_theta_dq<double>(
        lv_of<6>(b.l_basis), gg::to_luisa(qk), gg::to_luisa(qk_t),
        lv_of<6>(b.r_basis), gg::to_luisa(ql),
        gg::to_luisa(ql_t)));
}

/// The ported penalty (host `double`).
Penalty penalty_port(double x, double lower, double upper, double strength)
{
    Penalty out;
    double  dE = 0.0, d2E = 0.0;
    gg::joint_limit::eval_penalty<double>(x, lower, upper, strength, out.E, dE, d2E);
    out.dE  = dE;
    out.d2E = d2E;
    return out;
}

double penalty_port_energy(double x, double lower, double upper, double strength)
{
    return gg::joint_limit::eval_penalty_energy<double>(x, lower, upper, strength);
}

std::pair<double, double> penalty_port_derivs(double x,
                                              double lower,
                                              double upper,
                                              double strength)
{
    double dE = 1.0, d2E = 1.0;  // deliberately non-zero: the routine must zero them
    gg::joint_limit::eval_penalty_derivatives<double>(x, lower, upper, strength, dE, d2E);
    return {dE, d2E};
}

constexpr double kSceneDt = 0.01;
}  // namespace


// ===========================================================================
//  1) closed forms: the two element laws and their derivatives, the (diagonal /
//     non-diagonal) element Hessians, the clamps, `unwrap_angle`, the angle helpers,
//     and every branch of the limit's penalty
// ===========================================================================
TEST_CASE("abd_joint_c_closed_forms", "[luisa][abd_joint_c]")
{
    test_runtime();
    CaseScope scope{"abd_joint_c_closed_forms"};
    auto&     c = scope.counter;

    const double kappa = 2.5;

    // a strained, *stretched* pair (the polar frames are load-bearing, rule 3)
    RevRest r;
    r.l0 = Vector3{0.11, -0.07, 0.05};
    r.l1 = Vector3{0.42, 0.16, -0.08};
    r.r0 = Vector3{-0.04, 0.09, 0.13};
    r.r1 = Vector3{0.23, 0.31, 0.02};
    const Matrix3x3 A0 = Eigen::AngleAxisd(0.4, Vector3{1, 0, 0}.normalized()).toRotationMatrix()
                         * Vector3{1.15, 0.9, 1.05}.asDiagonal();
    const Matrix3x3 A1 = Eigen::AngleAxisd(-0.7, Vector3{0, 0, 1}.normalized()).toRotationMatrix()
                         * Vector3{0.95, 1.2, 1.0}.asDiagonal();
    RevBasis basis;
    basis.set(A0, A1, r.axle_world(A0));
    const Vector12 qk = q_of(1.1 * Matrix3x3::Identity(), Vector3{0.3, -0.2, 0.4});
    const Vector12 ql = q_of(0.95 * Matrix3x3::Identity(), Vector3{-0.1, 0.5, -0.3});

    // ---- (A) the joint's element law ----------------------------------------
    {
        const auto loc = rev_local(kappa, r, qk, ql);

        const Dense6v F_port = rj_Faxis(r, qk, ql);
        T63B3_CHECK(c, dev(F_port, Faxis_of(r, qk, ql)) <= 1e-15);
        T63B3_CHECK(c, rel_err(rj_E(kappa, r, qk, ql), loc.E) <= 1e-15);
        T63B3_CHECK(c, dev(rj_G(kappa, r, qk, ql), loc.G) <= 1e-13);
        T63B3_CHECK(c, dev(rj_H(kappa, r, qk, ql), loc.H) <= 1e-12);
        std::printf("[t63b3] closed: strained pair E %.10e, |G| %.6e, |H| %.6e, |Faxis| "
                    "%.6e\n",
                    loc.E, max_abs(loc.G), max_abs(loc.H), max_abs(loc.F));

        // the *constant* pair Jacobian against its own FD Jacobian (the
        // A_j-starts-at-15 trap, §7.25 rule 9)
        {
            const Dense24v q24 = pack_pair(qk, ql);
            Eigen::Matrix<double, 6, 24> Jfd;
            for(int k = 0; k < 24; ++k)
            {
                Dense24v qp = q24, qm = q24;
                qp(k) += kFdH;
                qm(k) -= kFdH;
                Jfd.col(k) = (Faxis_of(r, Vector12(qp.segment<12>(0)), Vector12(qp.segment<12>(12)))
                              - Faxis_of(r, Vector12(qm.segment<12>(0)), Vector12(qm.segment<12>(12))))
                             / (2 * kFdH);
            }
            T63B3_CHECK(c, dev(Jaxis_of(r), Jfd) <= 1e-8);
            // the fan-out of the *generated* `JaxisT_Gaxis` against `J^T dE/dF`
            const auto JG = rj_G(kappa, r, qk, ql);
            T63B3_CHECK(c, dev(JG, Jaxis_of(r).transpose() * loc.dEdF) <= 1e-13);
            // ... and the element Hessian is `kappa I6`: DIAGONAL, hence the reference's
            // `make_spd` clamp on it is value-invariant (the T6.3b1 situation, the
            // *opposite* of T6.3b2's joints -- the sweep pin is aimed at the driving
            // and the limit instead)
            T63B3_CHECK(c, dev(loc.d2EdF2, kappa * Dense6::Identity()) <= 1e-15);
            T63B3_CHECK(c, max_abs(loc.d2EdF2 - Dense6(loc.d2EdF2.diagonal().asDiagonal())) == 0.0);
            // the generated 6x6 (host call) is the same matrix
            lps::LMat<double, 6, 6> H6;
            gg::sym::affine_body_revolute_joint::ddEaxisddFaxis<double>(H6, kappa, lv_of<6>(loc.F));
            T63B3_CHECK(c, dev(to_eigen6(H6), loc.d2EdF2) <= 1e-15);
            {
                auto a = H6, b = H6;
                lps::make_spd_inplace(a, gg::abd_revolute_joint_detail::kMakeSpdSweeps);
                lps::make_spd_inplace(b, 0);
                double w = 0.0;
                for(int cc = 0; cc < 6; ++cc)
                    for(int rr = 0; rr < 6; ++rr)
                        w = std::max(w, std::abs(a(rr, cc) - b(rr, cc)));
                std::printf("[t63b3] closed: the joint's 6x6 element Hessian is diagonal -> "
                            "0-vs-%d sweeps differ by %.3e (the clamp is a no-op here)\n",
                            gg::abd_revolute_joint_detail::kMakeSpdSweeps, w);
                T63B3_CHECK(c, w == 0.0);
            }
        }
    }

    // ---- (A2) the *sweep-count* pin on the non-diagonal element Hessians (the §7.26
    //      rule-10 axis, aimed at the two systems that need it) -----------------
    {
        const Dense12v F01q = F01q_of(basis.l_basis, qk, basis.r_basis, ql);
        const double   tt   = 0.17;
        lps::LMat<double, 12, 12> H12_d, H12_l;
        gg::sym::affine_body_driving_revolute_joint::ddEddF01<double>(H12_d, kappa,
                                                                     lv_of<12>(F01q), tt);
        Dense12v g_th;
        Dense12  H_th;
        theta_grad_hess_F(F01q, g_th, H_th);
        for(int cc = 0; cc < 12; ++cc)
            for(int rr = 0; rr < 12; ++rr)
                H12_l(rr, cc) = (2.7 * H_th)(rr, cc);  // the limit's clamped 12x12

        auto sweep_diff = [](const lps::LMat<double, 12, 12>& H, int a, int b)
        {
            auto x = H, y = H;
            lps::make_spd_inplace(x, a);
            lps::make_spd_inplace(y, b);
            double w = 0.0;
            for(int cc = 0; cc < 12; ++cc)
                for(int rr = 0; rr < 12; ++rr)
                    w = std::max(w, std::abs(x(rr, cc) - y(rr, cc)));
            return w;
        };
        const double drv_0_8 = sweep_diff(H12_d, 0, gg::abd_driving_revolute_joint_detail::kMakeSpdSweeps);
        const double drv_8_20 = sweep_diff(H12_d, gg::abd_driving_revolute_joint_detail::kMakeSpdSweeps, 20);
        const double lim_0_8 = sweep_diff(H12_l, 0, gg::abd_driving_revolute_joint_detail::kMakeSpdSweeps);
        std::printf("[t63b3] closed: sweep pin -- 0-vs-%d: driving %.3e, limit %.3e; "
                    "%d-vs-20: driving %.3e\n",
                    gg::abd_driving_revolute_joint_detail::kMakeSpdSweeps, drv_0_8, lim_0_8,
                    gg::abd_driving_revolute_joint_detail::kMakeSpdSweeps, drv_8_20);
        T63B3_CHECK(c, drv_0_8 > 1e-6);
        T63B3_CHECK(c, lim_0_8 > 1e-6);
        T63B3_CHECK(c, drv_8_20 == 0.0);
    }

    // ---- (B) over random admissible states: the shipped vs the exact clamp ----
    {
        constexpr int kStates = 40;
        double        worst_E = 0.0, worst_G = 0.0, worst_H = 0.0, worst_dE = 0.0;
        double        worst_ddE = 0.0, worst_theta = 0.0, worst_delta = 0.0;
        double        worst_ddq = 0.0;
        int           nondiag = 0;
        for(int s = 0; s < kStates; ++s)
        {
            RevRest rr;
            const Matrix3x3 Ak = random_admissible_A();
            const Matrix3x3 Al = random_admissible_A();
            const Vector3   tk = rand_vec(-0.4, 0.4);
            const Vector3   tl = rand_vec(-0.4, 0.4);
            rr.set_consistent(Ak, tk, Al, tl, rand_vec(-0.5, 0.5), rand_vec(-0.5, 0.5),
                              rand_uniform(-0.2, 0.2));
            RevBasis bb;
            bb.set(Ak, Al, rr.axle_world(Ak));
            const Vector12 q_k = q_of(Ak, tk);
            const Vector12 q_l = q_of(Al, tl);
            const Vector12 q_k_t = q_of(Ak, tk + rand_vec(-0.1, 0.1));
            const Vector12 q_l_t = q_of(Al, tl + rand_vec(-0.1, 0.1));
            const double   kp    = rand_uniform(0.5, 2.0);
            const double   tt    = rand_uniform(-0.5, 0.5);

            const auto loc = rev_local(kp, rr, q_k, q_l);
            worst_E  = std::max(worst_E, rel_err(rj_E(kp, rr, q_k, q_l), loc.E));
            worst_G  = std::max(worst_G, dev(rj_G(kp, rr, q_k, q_l), loc.G));
            worst_H  = std::max(worst_H, dev(rj_H(kp, rr, q_k, q_l), loc.H));

            // the driving's three laws against the *analytic* atan2 chain rule
            const Dense12v F = F01q_of(bb.l_basis, q_k, bb.r_basis, q_l);
            worst_theta = std::max(worst_theta, std::abs(drj_theta(bb, q_k, q_l) - theta_of_F(F)));
            worst_E     = std::max(worst_E, rel_err(drj_E(kp, bb, q_k, q_l, tt),
                                                    E_driving_of(F, kp, tt)));
            worst_dE    = std::max(worst_dE, dev(drj_G(kp, bb, q_k, q_l, tt),
                                                 J01q_of(bb.l_basis, bb.r_basis).transpose()
                                                     * dE_driving_of(F, kp, tt)));
            {
                lps::LMat<double, 12, 12> HH;
                gg::sym::affine_body_driving_revolute_joint::ddEddF01<double>(HH, kp, lv_of<12>(F), tt);
                const Dense12 H_law = ddE_driving_of(F, kp, tt);
                worst_ddE = std::max(worst_ddE, dev(to_eigen12(HH), H_law));
                if(max_abs(H_law - Dense12(H_law.diagonal().asDiagonal())) > 1e-12 * max_abs(H_law))
                    ++nondiag;
                // the shipped fan-out equals `J^T make_spd(H) J` with the layer's clamp
                auto H_spd = HH;
                lps::make_spd_inplace(H_spd, gg::abd_driving_revolute_joint_detail::kMakeSpdSweeps);
                const Eigen::Matrix<double, 12, 24> J = J01q_of(bb.l_basis, bb.r_basis);
                worst_H = std::max(worst_H, dev(drj_H(kp, bb, q_k, q_l, tt),
                                                J.transpose() * to_eigen12(H_spd) * J));
            }

            // the limit's `DeltaTheta` (both overloads) and its `q`-space chain
            const double d_gen = erj_delta_theta(bb, q_k, q_k_t, q_l, q_l_t);
            worst_delta = std::max(worst_delta,
                                   std::abs(d_gen - delta_theta_q(bb.l_basis, q_k, q_k_t,
                                                                  bb.r_basis, q_l, q_l_t)));
            {
                const double d12 = gg::abd_revolute_joint_limit_detail::delta_theta_f<double>(
                    lv_of<12>(F), lv_of<12>(F01q_of(bb.l_basis, q_k_t, bb.r_basis, q_l_t)));
                worst_delta = std::max(worst_delta, std::abs(d_gen - d12));
            }
            {
                Dense12v g_th;
                Dense12 H_th;
                theta_grad_hess_F(F, g_th, H_th);
                worst_ddq = std::max(worst_ddq,
                                     dev(erj_ddelta_dq(bb, q_k, q_k_t, q_l, q_l_t),
                                         J01q_of(bb.l_basis, bb.r_basis).transpose() * g_th));
            }
        }
        std::printf("[t63b3] closed (%d states): joint E/G/H %.3e/%.3e/%.3e, driving "
                    "theta %.3e dE %.3e ddE %.3e (non-diag %d), DeltaTheta %.3e, "
                    "dx_dq %.3e\n",
                    kStates, worst_E, worst_G, worst_H, worst_theta, worst_dE,
                    worst_ddE, nondiag, worst_delta, worst_ddq);
        T63B3_CHECK(c, worst_E <= 1e-13);
        T63B3_CHECK(c, worst_G <= 1e-13);
        // NOTE: the bound is the *layer's* 8-sweep Jacobi `make_spd` vs the exact clamp
        // (~1e-10 for a 12x12), not a transcription residual: the joint's own H is exact
        // (~1e-16) and the driving's shipped H is compared against the *layer's* clamp
        T63B3_CHECK(c, worst_H <= 1e-9);
        T63B3_CHECK(c, worst_theta <= 1e-15);
        T63B3_CHECK(c, worst_dE <= 1e-12);
        T63B3_CHECK(c, worst_ddE <= 1e-11);
        T63B3_CHECK(c, nondiag > 0);
        T63B3_CHECK(c, worst_delta <= 1e-13);
        T63B3_CHECK(c, worst_ddq <= 1e-12);
    }

    // ---- (C) `unwrap_angle` and the two angle helpers ------------------------
    {
        // the ported `unwrap_angle`'s spelling is `ref + (x - 2pi round(x/2pi))`, while
        // the reference calls `::remainder` (half-to-even ties): measured on 200 states
        double worst_unwrap = 0.0, worst_rel = 0.0, worst_tt = 0.0;
        for(int s = 0; s < 200; ++s)
        {
            const double wrapped = rand_uniform(-3.0 * std::numbers::pi, 3.0 * std::numbers::pi);
            const double ref     = rand_uniform(-4.0 * std::numbers::pi, 4.0 * std::numbers::pi);
            const double unwrapped = gg::unwrap_angle<double>(wrapped, ref);
            const double r = ref
                             + std::remainder(wrapped - ref, 2.0 * std::numbers::pi);
            worst_unwrap = std::max(worst_unwrap, std::abs(unwrapped - r));

            // the relative-angle helper vs the hand law, and `driving_theta_tilde`'s
            // fold identity: `theta - theta_tilde == aim - init - (theta_eff - theta)`
            const RevBasis bb = basis;
            const double   th = drj_theta(bb, qk, ql);
            const double   ch = gg::compute_relative_angle<double>(
                                                                   lv_of<6>(bb.l_basis), gg::to_luisa(qk),
                                                                   lv_of<6>(bb.r_basis), gg::to_luisa(ql));
            worst_rel = std::max(worst_rel, std::abs(ch - theta_of_F(F01q_of(bb.l_basis, qk,
                                                                            bb.r_basis, ql))));
            const double init_a = rand_uniform(-0.7, 0.7);  // the joint's `init_angle`
            const double aim    = rand_uniform(-1.0, 1.0);
            const double tt     = gg::driving_theta_tilde<double>(
                lv_of<12>(drj_F01q(bb, qk, ql)), aim, init_a, ch);
            const double theta_eff = gg::unwrap_angle<double>(th, ch - init_a);
            // the fold identity: `tt == aim - init - (theta_eff - theta)`, i.e. the
            // *graded* quantity `theta - tt` is the unwrapped excursion
            worst_tt = std::max(worst_tt, std::abs(tt - (aim - init_a - (theta_eff - th))));
        }
        std::printf("[t63b3] closed: unwrap_angle vs ::remainder %.3e, compute_relative_angle "
                    "%.3e, driving_theta_tilde fold %.3e\n",
                    worst_unwrap, worst_rel, worst_tt);
        T63B3_CHECK(c, worst_unwrap <= 1e-15);
        T63B3_CHECK(c, worst_rel <= 1e-15);
        T63B3_CHECK(c, worst_tt <= 1e-14);
    }

    // ---- (D) the joint-limit penalty: every branch of the reference's body ----
    {
        struct Case
        {
            double x, lower, upper, strength;
            const char* name;
        };
        const Case cases[] = {
            {0.10, -0.05, 0.20, 3.0, "inside"},
            {-0.05, -0.05, 0.20, 3.0, "at lower"},
            {0.20, -0.05, 0.20, 3.0, "at upper"},
            {0.35, -0.05, 0.20, 3.0, "above"},
            {-0.30, -0.05, 0.20, 3.0, "below"},
            {0.35, -0.05, 0.20, -3.0, "above, negative strength"},
            {-0.30, -0.05, 0.20, -3.0, "below, negative strength"},
            {0.30, 0.10, 0.10, 5.0, "zero width, above"},
            {-0.20, 0.10, 0.10, 5.0, "zero width, below"},
            {0.10, 0.10, 0.10, 5.0, "zero width, exactly at the bound"},
            {1e6, -1.0, 1.0, 1e-3, "far outside"},
        };
        double worst_E = 0.0, worst_dE = 0.0, worst_d2E = 0.0;
        for(const auto& cs : cases)
        {
            const auto ref  = penalty_ref(cs.x, cs.lower, cs.upper, cs.strength);
            const auto port = penalty_port(cs.x, cs.lower, cs.upper, cs.strength);
            worst_E   = std::max(worst_E, std::abs(port.E - ref.E));
            worst_dE  = std::max(worst_dE, std::abs(port.dE - ref.dE));
            worst_d2E = std::max(worst_d2E, std::abs(port.d2E - ref.d2E));
            const auto derivs = penalty_port_derivs(cs.x, cs.lower, cs.upper, cs.strength);
            T63B3_CHECK(c, std::abs(penalty_port_energy(cs.x, cs.lower, cs.upper, cs.strength)
                                    - ref.E)
                                   <= 1e-15);
            T63B3_CHECK(c, std::abs(derivs.first - ref.dE) <= 1e-15);
            T63B3_CHECK(c, std::abs(derivs.second - ref.d2E) <= 1e-15);
            if(std::string_view{cs.name} == "inside")
            {
                T63B3_CHECK(c, port.E == 0.0);
                T63B3_CHECK(c, port.dE == 0.0);
                T63B3_CHECK(c, port.d2E == 0.0);
            }
            if(std::string_view{cs.name} == "at upper" || std::string_view{cs.name} == "at lower")
            {
                // the *inclusive* interval: at the bound the penalty is exactly zero
                T63B3_CHECK(c, (port.E == 0.0 && port.dE == 0.0 && port.d2E == 0.0));
            }
        }
        std::printf("[t63b3] closed: penalty branches (%zu cases), worst |dE| %.3e |dE_dx| "
                    "%.3e |d2E_dx2| %.3e vs the branchy oracle\n",
                    sizeof(cases) / sizeof(cases[0]), worst_E, worst_dE, worst_d2E);
        T63B3_CHECK(c, worst_E <= 1e-15);
        T63B3_CHECK(c, worst_dE <= 1e-15);
        T63B3_CHECK(c, worst_d2E <= 1e-15);
    }

    // ---- (E) the limit's own law at a state and the external-force pair -------
    {
        // the bounds put the state's angle *outside* them (a limit whose interval
        // contains x contributes nothing)
        const double th = drj_theta(basis, qk, ql);
        const double lo = th + 0.05;
        const double hi = th + 0.60;
        const auto   lim = limit_local(th, lo, hi, 40.0, basis.l_basis, qk, qk, basis.r_basis,
                                       ql, ql);
        std::printf("[t63b3] closed: limit x %.10e E %.6e dE_dx %.6e d2E_dx2 %.6e |G| %.3e "
                    "|H| %.3e\n",
                    lim.x, lim.E, lim.dE_dx, lim.d2E_dx2, max_abs(lim.G), max_abs(lim.H));
        T63B3_CHECK(c, std::abs(lim.x - th) <= 1e-15);
        T63B3_CHECK(c, lim.E > 0.0);
        T63B3_CHECK(c, std::abs(lim.dE_dx) > 1e-9);
        T63B3_CHECK(c, dev(erj_ddelta_dq(basis, qk, qk, ql, ql), lim.dx_dq) <= 1e-12);
        T63B3_CHECK(c, dev(lim.G, lim.dE_dx * lim.dx_dq) <= 1e-15);
        T63B3_CHECK(c, dev(lim.G,
                           to_eigen24(gg::abd_revolute_joint_limit_detail::gradient<double>(
                               lim.dE_dx, lv_of<24>(lim.dx_dq))))
                           <= 1e-15);
        // the ported branchless `hessian` helper is the reference's three kernels'
        // statement (the `dE_dx != 0` term included, unconditionally)
        {
            const Dense24 H2 = to_eigen24(gg::abd_revolute_joint_limit_detail::hessian<double>(
                lim.d2E_dx2, lim.dE_dx, lv_of<24>(lim.dx_dq), lv_of<6>(basis.l_basis), gg::to_luisa(qk),
                gg::to_luisa(qk), lv_of<6>(basis.r_basis), gg::to_luisa(ql), gg::to_luisa(ql)));
            T63B3_CHECK(c, dev(H2, lim.H) <= 1e-9);
            // ... and for `dE_dx == 0` the second term is *exactly* zero, so the
            // reference's `$if` guard and the branchless spelling are bitwise equal
            const Dense24 H_zero = to_eigen24(gg::abd_revolute_joint_limit_detail::hessian<double>(
                lim.d2E_dx2, 0.0, lv_of<24>(lim.dx_dq), lv_of<6>(basis.l_basis), gg::to_luisa(qk),
                gg::to_luisa(qk), lv_of<6>(basis.r_basis), gg::to_luisa(ql), gg::to_luisa(ql)));
            const Dense24 H_only =
                lim.d2E_dx2 * (lim.dx_dq * lim.dx_dq.transpose());
            // the reference's `$if` *skips* the second statement; the port's branchless
            // helper evaluates it and adds `J^T make_spd(0) J`, whose value is the
            // Jacobi EVD's round-off of a *zero* 12x12 (measured ~1e-16, i.e. the
            // deviation is the layer's eigensolver on a zero matrix, not a law change)
            T63B3_CHECK(c, dev(H_zero, H_only) <= 1e-12);
            std::printf("[t63b3] closed: the reference's `if(dE_dx != 0)` guard is "
                        "value-free: |H(0) - d2E (dx dx^T)| %.3e (the zero matrix's EVD "
                        "round-off)\n",
                        max_abs(H_zero - H_only));
        }

        // the external-force pair: the axle direction comes from the *edge's own*
        // endpoints (`l1 - l0`), is normalised per body and symmetrised
        const Dense3v  e_bar_i = r.l1 - r.l0;
        const Dense3v  e_bar_j = r.r1 - r.r0;
        const auto     ef      = ext_force_local(e_bar_i, e_bar_j, qk, ql);
        lps::LVec<double, 3> e_i, e_j;
        gg::abd_revolute_joint_external_force_detail::symmetrized_axes<double>(
            e_i, e_j, lv3(e_bar_i), lv3(e_bar_j), gg::to_luisa(qk), gg::to_luisa(ql));
        T63B3_CHECK(c, dev(to_eigen3(e_i), ef.e_i) <= 1e-15);
        T63B3_CHECK(c, dev(to_eigen3(e_j), ef.e_j) <= 1e-15);
        T63B3_CHECK(c, (to_eigen3(e_i) + to_eigen3(e_j)).norm() == 0.0);
        // `torque_to_F` on the two half-axes: the *translational* dofs stay zero and the
        // rotational block carries `A^{-T}`'s image (`F[3:12] = A_to_q(tau/2 [e]_x
        // A^{-T})`), so the check is on the first three dofs
        {
            auto F_i = gg::torque_to_F<double>(12.5, e_i, gg::to_luisa(qk));
            double trans = 0.0;
            for(int k = 0; k < 3; ++k)
                trans = std::max(trans, std::abs(F_i(k)));
            T63B3_CHECK(c, trans == 0.0);
            T63B3_CHECK(c, max_abs(eig_of<12>(F_i)) > 0.0);
            auto F_j = gg::torque_to_F<double>(12.5, e_j, gg::to_luisa(ql));
            T63B3_CHECK(c, max_abs(eig_of<12>(F_j)) > 0.0);
            // the two axes are anti-parallel, so the pair is `F_j(-e)`: with the *same*
            // state the two forces are negatives of each other through the axis sign
            T63B3_CHECK(c, (to_eigen3(e_i) + to_eigen3(e_j)).norm() == 0.0);
        }
    }

    std::printf("[t63b3] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}


// ===========================================================================
//  2) central finite differences over the 24 pair dofs: the gradient is the
//     derivative of E, the Hessian the derivative of the gradient, the h-sweep, the
//     symmetry, the rule-34 basis question, and what the clamp does to `(g, H)`
// ===========================================================================
TEST_CASE("abd_joint_c_finite_difference", "[luisa][abd_joint_c]")
{
    test_runtime();
    CaseScope scope{"abd_joint_c_finite_difference"};
    auto&     c = scope.counter;

    constexpr int kStates = 40;

    double worst_g = 0.0, worst_h6 = 0.0, worst_h4 = 0.0, worst_h3 = 0.0, worst_h2 = 0.0;
    double worst_sym = 0.0, worst_raw = 0.0, worst_shipped = 0.0, worst_d_g = 0.0;
    double worst_driving_g = 0.0, worst_driving_h = 0.0, worst_driving_raw = 0.0;
    double worst_limit_g = 0.0, worst_limit_h = 0.0;
    double worst_J = 0.0;
    int    nondiag_drv = 0;

    for(int s = 0; s < kStates; ++s)
    {
        const Matrix3x3 Ak = random_admissible_A();
        const Matrix3x3 Al = random_admissible_A();
        const Vector3   tk = rand_vec(-0.4, 0.4);
        const Vector3   tl = rand_vec(-0.4, 0.4);
        RevRest         r;
        r.set_consistent(Ak, tk, Al, tl, rand_vec(-0.5, 0.5), rand_vec(-0.5, 0.5),
                         rand_uniform(-0.2, 0.2));
        RevBasis basis;
        basis.set(Ak, Al, r.axle_world(Ak));
        const Vector12 qk   = q_of(Ak, tk);
        const Vector12 ql   = q_of(Al, tl);
        const Vector12 qk_t = q_of(Ak, tk + rand_vec(-0.1, 0.1));
        const Vector12 ql_t = q_of(Al, tl + rand_vec(-0.1, 0.1));
        const double   kappa = rand_uniform(0.5, 2.0);
        const double   tt    = rand_uniform(-0.5, 0.5);
        const Dense24v q24   = pack_pair(qk, ql);

        // ---- the joint --------------------------------------------------------
        {
            const Dense24v G = rj_G(kappa, r, qk, ql);
            const Dense24v Gfd =
                fd_gradient([&](const Dense24v& qq)
                            { return rj_E(kappa, r, Vector12(qq.segment<12>(0)),
                                          Vector12(qq.segment<12>(12))); },
                            q24, kFdH);
            worst_g = std::max(worst_g, dev(G, Gfd));

            auto grad = [&](const Dense24v& qq)
            { return rj_G(kappa, r, Vector12(qq.segment<12>(0)), Vector12(qq.segment<12>(12))); };

            const Dense24 H    = rj_H(kappa, r, qk, ql);
            const Dense24 Hfd6 = fd_hessian(grad, q24, 1e-6);
            const Dense24 Hfd4 = fd_hessian(grad, q24, 1e-4);
            const Dense24 Hfd3 = fd_hessian(grad, q24, 1e-3);
            const Dense24 Hfd2 = fd_hessian(grad, q24, 1e-2);
            worst_h6 = std::max(worst_h6, dev(H, Hfd6));
            worst_h4 = std::max(worst_h4, dev(H, Hfd4));
            worst_h3 = std::max(worst_h3, dev(H, Hfd3));
            worst_h2 = std::max(worst_h2, dev(H, Hfd2));
            worst_sym =
                std::max(worst_sym, max_abs(H - Dense24(H.transpose())) / max_abs(H));

            // the joint's element Hessian is `kappa I6` (DIAGONAL), so its clamp is a
            // *no-op* and the shipped `H` IS `dG/dq`: `H_raw == H` here (the T6.3b1
            // situation; the clamp story of this batch lives in the driving/limit)
            const auto loc = rev_local(kappa, r, qk, ql);
            worst_raw      = std::max(worst_raw, dev(loc.H_raw, Hfd6));
            worst_shipped  = std::max(worst_shipped, dev(H, Hfd6));
            worst_d_g      = std::max(worst_d_g, dev(loc.H_raw, H));
        }

        // ---- the driving joint ------------------------------------------------
        {
            const Dense24v G = drj_G(kappa, basis, qk, ql, tt);
            const Dense24v Gfd =
                fd_gradient([&](const Dense24v& qq)
                            { return drj_E(kappa, basis, Vector12(qq.segment<12>(0)),
                                           Vector12(qq.segment<12>(12)), tt); },
                            q24, kFdH);
            worst_driving_g = std::max(worst_driving_g, dev(G, Gfd));

            auto grad = [&](const Dense24v& qq)
            {
                return drj_G(kappa, basis, Vector12(qq.segment<12>(0)),
                             Vector12(qq.segment<12>(12)), tt);
            };
            const Dense24 H    = drj_H(kappa, basis, qk, ql, tt);
            const Dense24 Hfd6 = fd_hessian(grad, q24, 1e-6);
            worst_driving_h    = std::max(worst_driving_h, dev(H, Hfd6));

            // the *analytic* unclamped chain of the shipped gradient (the FD of the
            // shipped gradient is the unclamped chain: the reference clamps only H)
            const Dense12v F    = F01q_of(basis.l_basis, qk, basis.r_basis, ql);
            const auto     J    = J01q_of(basis.l_basis, basis.r_basis);
            const Dense24  raw  = J.transpose() * ddE_driving_of(F, kappa, tt) * J;
            worst_driving_raw   = std::max(worst_driving_raw, dev(raw, Hfd6));
            Dense12v g_th;
            Dense12  H_th;
            theta_grad_hess_F(F, g_th, H_th);
            if(max_abs(H_th - Dense12(H_th.diagonal().asDiagonal())) > 1e-12 * max_abs(H_th))
                ++nondiag_drv;
        }

        // ---- the joint limit --------------------------------------------------
        {
            // pick the bounds so that the *initial* angle is outside the interval (so
            // `dE_dx != 0` and the Hessian has both terms)
            const double th = theta_of_F(F01q_of(basis.l_basis, qk, basis.r_basis, ql));
            const double lo = th + rand_uniform(0.05, 0.3);
            const double hi = lo + rand_uniform(0.05, 0.5);
            const double str = rand_uniform(1.0, 50.0);
            const auto   lim = limit_local(th, lo, hi, str, basis.l_basis, qk, qk_t,
                                           basis.r_basis, ql, ql_t);

            const Dense24v Gfd =
                fd_gradient([&](const Dense24v& qq)
                            { return limit_local(th, lo, hi, str, basis.l_basis,
                                                 Vector12(qq.segment<12>(0)), qk_t,
                                                 basis.r_basis, Vector12(qq.segment<12>(12)),
                                                 ql_t)
                                         .E; },
                            q24, kFdH);
            worst_limit_g = std::max(worst_limit_g, dev(lim.G, Gfd));

            if(std::abs(lim.dE_dx) > 1e-9)
            {
                auto grad = [&](const Dense24v& qq)
                {
                    return limit_local(th, lo, hi, str, basis.l_basis,
                                       Vector12(qq.segment<12>(0)), qk_t, basis.r_basis,
                                       Vector12(qq.segment<12>(12)), ql_t)
                        .G;
                };
                worst_limit_h = std::max(worst_limit_h, dev(lim.H, fd_hessian(grad, q24, 1e-5)));
            }
        }

        // ---- the *constant* pair Jacobians, vs their own FD Jacobians ----------
        {
            const auto J = Jaxis_of(r);
            Eigen::Matrix<double, 6, 24> Jfd;
            for(int k = 0; k < 24; ++k)
            {
                Dense24v qp = q24, qm = q24;
                qp(k) += kFdH;
                qm(k) -= kFdH;
                Jfd.col(k) = (Faxis_of(r, Vector12(qp.segment<12>(0)), Vector12(qp.segment<12>(12)))
                              - Faxis_of(r, Vector12(qm.segment<12>(0)), Vector12(qm.segment<12>(12))))
                             / (2 * kFdH);
            }
            worst_J = std::max(worst_J, dev(J, Jfd));

            const auto Jd = J01q_of(basis.l_basis, basis.r_basis);
            Eigen::Matrix<double, 12, 24> Jdfd;
            for(int k = 0; k < 24; ++k)
            {
                Dense24v qp = q24, qm = q24;
                qp(k) += kFdH;
                qm(k) -= kFdH;
                Jdfd.col(k) =
                    (F01q_of(basis.l_basis, Vector12(qp.segment<12>(0)), basis.r_basis,
                             Vector12(qp.segment<12>(12)))
                     - F01q_of(basis.l_basis, Vector12(qm.segment<12>(0)), basis.r_basis,
                               Vector12(qm.segment<12>(12))))
                    / (2 * kFdH);
            }
            worst_J = std::max(worst_J, dev(Jd, Jdfd));
        }
    }

    std::printf("[t63b3] FD: joint grad %3.1e  H(1e-6) %3.1e (1e-4) %3.1e (1e-3) %3.1e "
                "(1e-2) %3.1e  asym %3.1e  non-diagonal driving states %d/%d\n",
                worst_g, worst_h6, worst_h4, worst_h3, worst_h2, worst_sym, nondiag_drv,
                kStates);
    std::printf("[t63b3] FD: the joint's clamp is a no-op: dev(H_raw, FD) %3.1e = "
                "dev(H_shipped, FD) %3.1e, ||H_raw - H_shipped|| %3.1e\n",
                worst_raw, worst_shipped, worst_d_g);
    std::printf("[t63b3] FD: driving grad %3.1e  H %3.1e (raw chain vs FD %3.1e); "
                "limit grad %3.1e  H %3.1e; constant Jaxis/J01q vs FD %3.1e\n",
                worst_driving_g, worst_driving_h, worst_driving_raw, worst_limit_g,
                worst_limit_h, worst_J);

    T63B3_CHECK(c, worst_g <= 1e-9);
    T63B3_CHECK(c, worst_driving_g <= 1e-9);
    T63B3_CHECK(c, worst_limit_g <= 1e-8);
    // the rule-34 basis question, answered *negative* again: the generated fan-outs
    // take their element Hessians in the same order as their `dEdF` siblings, so the
    // *plain* comparison IS the agreement (`worst_raw` for the joint, the analytic
    // chain for the driving)
    T63B3_CHECK(c, worst_raw <= 1e-5);
    T63B3_CHECK(c, worst_driving_raw <= 1e-5);
    // the joint's element Hessian is diagonal -> its clamp is exactly value-invariant
    T63B3_CHECK(c, worst_d_g <= 1e-12);
    T63B3_CHECK(c, nondiag_drv > 0);
    // the driving's and the limit's shipped H are `make_spd`-ed while the FD
    // differentiates the *unclamped* gradient: their deviation is O(clamp), not
    // round-off, and is pinned as a lower bound
    T63B3_CHECK(c, worst_driving_h > 1e-4);
    T63B3_CHECK(c, worst_limit_h > 1e-6);
    T63B3_CHECK(c, worst_sym <= 1e-12);
    T63B3_CHECK(c, worst_J <= 1e-8);

    std::printf("[t63b3] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  3) degenerates: the coincident rest pair, the *free twist*, zero/negative kappa,
//     reflections and singular states, the limit's branches and the winding wrap
// ===========================================================================
TEST_CASE("abd_joint_c_degenerate_inputs", "[luisa][abd_joint_c]")
{
    test_runtime();
    CaseScope scope{"abd_joint_c_degenerate_inputs"};
    auto&     c = scope.counter;

    // ---- (D1) a coincident rest pair: E == 0 and |G| == 0 *bitwise* -------------
    {
        RevRest r;
        r.l0 = Vector3{0.05, 0.1, -0.02};
        r.l1 = Vector3{0.35, 0.15, 0.05};
        r.r0 = r.l0;
        r.r1 = r.l1;
        const Vector12 qk = q_of(Matrix3x3::Identity(), Vector3{1.0, -0.5, 2.0});
        const Vector12 ql = qk;  // the same pose -> Faxis == 0

        T63B3_CHECK(c, rj_E(2.5, r, qk, ql) == 0.0);
        T63B3_CHECK(c, max_abs(rj_G(2.5, r, qk, ql)) == 0.0);
        const Dense24 H = rj_H(2.5, r, qk, ql);
        T63B3_CHECK(c, all_finite(H));
        T63B3_CHECK(c, max_abs(H) > 0.0);  // the *curvature* is not zero
        T63B3_CHECK(c, min_eigenvalue(H) >= -1e-10 * max_abs(H));
        std::printf("[t63b3] degen: coincident pair E %.1e |G| %.1e |H| %.6e (min eig %.3e)\n",
                    rj_E(2.5, r, qk, ql), max_abs(rj_G(2.5, r, qk, ql)), max_abs(H),
                    min_eigenvalue(H));
    }

    // ---- (D2) the *free twist*: the revolute joint's defining property ----------
    //  A rigid rotation of one body about the shared axle keeps both point pairs
    //  coincident (`E_axis == 0`) while the relative angle moves by exactly the twist
    //  angle.  `delta_theta_q` then measures the wedge in angles, and pairs that are
    //  *forward* equal in `q` still give `E == 0`.
    {
        const Matrix3x3 Ak = Eigen::AngleAxisd(0.3, Vector3{1, 2, 0.5}.normalized())
                                 .toRotationMatrix();
        const Matrix3x3 Al = Eigen::AngleAxisd(-0.8, Vector3{0.2, 1, -1}.normalized())
                                 .toRotationMatrix();
        const Vector3 tk = Vector3{0.1, -0.2, 0.3};
        const Vector3 tl = Vector3{-0.3, 0.15, 0.2};
        const Vector3 P0 = Vector3{0.2, -0.1, 0.25};
        const Vector3 P1 = Vector3{0.5, 0.05, 0.10};
        RevRest       r;
        r.set_consistent(Ak, tk, Al, tl, P0, P1, 0.0);
        RevBasis basis;
        basis.set(Ak, Al, r.axle_world(Ak));
        const Vector12 qk = q_of(Ak, tk);
        const Vector12 ql = q_of(Al, tl);

        T63B3_CHECK(c, max_abs(rj_Faxis(r, qk, ql)) <= 1e-15);

        const Vector3   axis = (P1 - P0).normalized();
        const double    phi  = 0.3;
        const Matrix3x3 R    = Eigen::AngleAxisd(phi, axis).toRotationMatrix();
        const Vector12  ql_twist = q_of(R * Al, P0 + R * (tl - P0));
        const auto      Faxis_twist = rj_Faxis(r, qk, ql_twist);
        const double    E_twist     = rj_E(2.5, r, qk, ql_twist);
        const double    d_theta     = gg::compute_relative_angle<double>(
                                    lv_of<6>(basis.l_basis), gg::to_luisa(qk),
                                    lv_of<6>(basis.r_basis), gg::to_luisa(ql_twist))
                                - gg::compute_relative_angle<double>(
                                    lv_of<6>(basis.l_basis), gg::to_luisa(qk),
                                    lv_of<6>(basis.r_basis), gg::to_luisa(ql));
        std::printf("[t63b3] degen: free twist by %.3f rad -> |Faxis| %.3e, E %.3e, "
                    "|d theta| %.10f (the twist is the joint's free dof)\n",
                    phi, max_abs(Faxis_twist), E_twist, std::abs(d_theta));
        T63B3_CHECK(c, max_abs(Faxis_twist) <= 1e-14);
        T63B3_CHECK(c, E_twist <= 1e-14);
        T63B3_CHECK(c, std::abs(std::abs(d_theta) - phi) <= 1e-12);
        // ... and the *sign* flips with the twist direction
        const Vector12 ql_twist_m = q_of(R.inverse() * Al, P0 + R.inverse() * (tl - P0));
        const double   d_theta_m  = gg::compute_relative_angle<double>(
                                        lv_of<6>(basis.l_basis), gg::to_luisa(qk),
                                        lv_of<6>(basis.r_basis), gg::to_luisa(ql_twist_m))
                                    - gg::compute_relative_angle<double>(
                                        lv_of<6>(basis.l_basis), gg::to_luisa(qk),
                                        lv_of<6>(basis.r_basis), gg::to_luisa(ql));
        T63B3_CHECK(c, std::abs(d_theta + d_theta_m) <= 1e-12);
    }

    // ---- (D3) the winding wrap of `DeltaTheta` past +-pi ------------------------
    {
        const Matrix3x3 Ak = Eigen::AngleAxisd(0.2, Vector3{0, 1, 0}).toRotationMatrix();
        const Matrix3x3 Al = Eigen::AngleAxisd(-0.4, Vector3{1, 0, 0}).toRotationMatrix();
        const Vector3   tk = Vector3{0.05, -0.1, 0.2};
        const Vector3   tl = Vector3{-0.2, 0.1, 0.15};
        const Vector3   P0 = Vector3{0.1, 0.0, 0.2};
        const Vector3   P1 = Vector3{0.4, 0.1, 0.05};
        RevRest         r;
        r.set_consistent(Ak, tk, Al, tl, P0, P1, 0.0);
        RevBasis       basis;
        basis.set(Ak, Al, r.axle_world(Ak));
        const Vector12 qk = q_of(Ak, tk);
        const Vector12 ql = q_of(Al, tl);

        // a twist of pi + 0.3: the naive difference is > pi, the generated wrap is not
        const Vector3   axis = (P1 - P0).normalized();
        const double    phi  = std::numbers::pi + 0.3;
        const Matrix3x3 R    = Eigen::AngleAxisd(phi, axis).toRotationMatrix();
        const Vector12  ql_w = q_of(R * Al, P0 + R * (tl - P0));
        const double    d    = erj_delta_theta(basis, qk, qk, ql_w, ql);
        const double    d_law = delta_theta_q(basis.l_basis, qk, qk, basis.r_basis, ql_w, ql);
        std::printf("[t63b3] degen: winding: a %.6f rad twist wraps to DeltaTheta %.10f "
                    "(in (-pi, pi])\n",
                    phi, d);
        T63B3_CHECK(c, std::abs(d - d_law) <= 1e-13);
        T63B3_CHECK(c, std::abs(d) <= std::numbers::pi + 1e-12);
        T63B3_CHECK(c, std::abs(std::abs(d) - (2.0 * std::numbers::pi - phi)) <= 1e-12);
        // a zero twist is exactly zero
        T63B3_CHECK(c, erj_delta_theta(basis, qk, qk, ql, ql) == 0.0);
    }

    // ---- (D4) zero and negative kappa, reflections and singular states ----------
    {
        RevRest r;
        r.l0 = Vector3{0.3, -0.2, 0.1};
        r.l1 = Vector3{0.6, 0.1, -0.15};
        r.r0 = Vector3{0.1, 0.25, -0.15};
        r.r1 = Vector3{0.4, 0.45, -0.4};
        const Vector12 qk = q_of(1.3 * Matrix3x3::Identity(), Vector3{0.1, 0.2, 0.3});
        const Vector12 ql = q_of(0.8 * Matrix3x3::Identity(), Vector3{-0.3, 0.1, 0.2});

        T63B3_CHECK(c, rj_E(0.0, r, qk, ql) == 0.0);
        T63B3_CHECK(c, max_abs(rj_G(0.0, r, qk, ql)) == 0.0);
        T63B3_CHECK(c, max_abs(rj_H(0.0, r, qk, ql)) == 0.0);

        const double kneg = -3.0;
        const auto   lneg = rev_local(kneg, r, qk, ql);
        const Dense24 Hneg = rj_H(kneg, r, qk, ql);
        T63B3_CHECK(c, lneg.E < 0.0);
        T63B3_CHECK(c, dev(rj_G(kneg, r, qk, ql), lneg.G) <= 1e-13);
        T63B3_CHECK(c, rel_err(rj_E(kneg, r, qk, ql), lneg.E) <= 1e-14);
        // the element Hessian is `kappa I6`: for kappa < 0 the clamp maps it to ZERO
        // (the T6.3b1 result, *unlike* T6.3b2's joints whose non-diagonal element
        // Hessians survive the clamp with a flipped spectrum)
        T63B3_CHECK(c, all_finite(Hneg));
        T63B3_CHECK(c, max_abs(Hneg) == 0.0);
        std::printf("[t63b3] degen: kappa = -3 -> E %.6e, |G| %.6e, |H| %.6e (the diagonal "
                    "element Hessian is clamped to zero)\n",
                    lneg.E, max_abs(lneg.G), max_abs(Hneg));

        // a reflection and a rank-deficient `A`: the laws are polynomials, so both stay
        // finite.  A *collapsed* (zero) body is different and is pinned as such: its
        // axle direction `orthonormal_basis(0)` divides by zero and the driving's
        // `atan2` chain rule has 1/(s^2+c^2) = 1/0 there — both properties are the
        // reference's own (it has no guard either), so the *energy* stays finite while
        // the driving's gradient/Hessian do not.
        const Matrix3x3 reflect  = Vector3{-1.0, 1.0, 1.0}.asDiagonal();
        const Matrix3x3 singular = Vector3{1.0, 1.0, 0.0}.asDiagonal();
        for(const auto& A : {reflect, singular})
        {
            const Vector12 qa = q_of(A, Vector3{0.1, -0.2, 0.3});
            const Vector12 qb = q_of(A * 0.5, Vector3{-0.1, 0.2, 0.1});
            T63B3_CHECK(c, all_finite(Dense24v(rj_G(1.5, r, qa, qb))));
            T63B3_CHECK(c, all_finite(rj_H(1.5, r, qa, qb)));
            T63B3_CHECK(c, std::isfinite(rj_E(1.5, r, qa, qb)));
            RevBasis bb;
            bb.set(A, A * 0.5, r.axle_world(A));
            T63B3_CHECK(c, all_finite(Dense24v(drj_G(1.5, bb, qa, qb, 0.2))));
            T63B3_CHECK(c, all_finite(drj_H(1.5, bb, qa, qb, 0.2)));
            T63B3_CHECK(c, std::isfinite(drj_E(1.5, bb, qa, qb, 0.2)));
            T63B3_CHECK(c, std::isfinite(drj_theta(bb, qa, qb)));
        }
        {
            // the collapsed body: finite *energy*, non-finite driving derivatives
            const Matrix3x3 zero(Matrix3x3::Zero());
            const Vector12  qa = q_of(zero, Vector3{0.1, -0.2, 0.3});
            const Vector12  qb = q_of(zero, Vector3{-0.1, 0.2, 0.1});
            // (an arbitrary *finite* basis: with `A = 0` the driving's `F01_q` is the
            // zero vector whatever the basis is)
            RevBasis bb0;
            bb0.set(Matrix3x3::Identity(), Matrix3x3::Identity(), Vector3{1, 0, 0});
            T63B3_CHECK(c, std::isfinite(rj_E(1.5, r, qa, qb)));
            T63B3_CHECK(c, std::isfinite(drj_E(1.5, bb0, qa, qb, 0.2)));
            // (the driving's theta is `atan2(0, 0) = 0` — finite — but its gradient's
            // 1/(s^2+c^2) is not: pinned, not "fixed")
            const auto G_zero = drj_G(1.5, bb0, qa, qb, 0.2);
            std::printf("[t63b3] degen: a *collapsed* body: joint E finite %.3e, driving E "
                        "finite %.3e, driving |G| finite? %d (the reference has no guard "
                        "either)\n",
                        rj_E(1.5, r, qa, qb), drj_E(1.5, bb0, qa, qb, 0.2),
                        static_cast<int>(all_finite(G_zero)));
            T63B3_CHECK(c, !all_finite(G_zero));
        }
        std::printf("[t63b3] degen: reflection / singular / zero A -> E,G,H finite "
                    "(reflection E %.6e)\n",
                    rj_E(1.5, r, q_of(reflect), q_of(reflect * 0.5)));
    }

    // ---- (D5) the limit's degenerate branches ----------------------------------
    {
        const Matrix3x3 Ak = Eigen::AngleAxisd(0.5, Vector3{0, 1, 0}).toRotationMatrix();
        const Matrix3x3 Al = Eigen::AngleAxisd(-0.2, Vector3{1, 0, 1}.normalized())
                                 .toRotationMatrix();
        const Vector3 tk = Vector3{0.1, 0.2, 0.3};
        const Vector3 tl = Vector3{-0.2, 0.1, 0.25};
        RevRest       r;
        r.set_consistent(Ak, tk, Al, tl, Vector3{0.15, -0.05, 0.05}, Vector3{0.35, 0.1, -0.1},
                         0.03);
        RevBasis       basis;
        basis.set(Ak, Al, r.axle_world(Ak));
        const Vector12 qk   = q_of(Ak, tk);
        const Vector12 ql   = q_of(Al, tl);
        const Vector12 qk_t = q_of(Ak, tk + Vector3{0.01, -0.02, 0.005});
        const Vector12 ql_t = q_of(Al, tl + Vector3{-0.005, 0.01, 0.002});
        const double   th   = theta_of_F(F01q_of(basis.l_basis, qk, basis.r_basis, ql));

        // zero-width range: the *raw* excursion branch
        {
            const auto lim = limit_local(th, 0.0, 0.0, 3.0, basis.l_basis, qk, qk_t,
                                         basis.r_basis, ql, ql_t);
            const auto ref = penalty_ref(lim.x, 0.0, 0.0, 3.0);
            T63B3_CHECK(c, std::abs(lim.E - ref.E) <= 1e-14);
            T63B3_CHECK(c, std::abs(lim.dE_dx - ref.dE) <= 1e-14);
            T63B3_CHECK(c, std::abs(lim.d2E_dx2 - ref.d2E) <= 1e-14);
            T63B3_CHECK(c, std::abs(lim.dE_dx) > 0.0);
            std::printf("[t63b3] degen: zero-width limit x %.6e -> E %.6e dE_dx %.6e "
                        "d2E_dx2 %.6e\n",
                        lim.x, lim.E, lim.dE_dx, lim.d2E_dx2);
        }

        // the limit *inside* the interval: E = G = H = 0 exactly
        {
            const auto lim = limit_local(th, th - 0.5, th + 0.5, 10.0, basis.l_basis, qk,
                                         qk_t, basis.r_basis, ql, ql_t);
            T63B3_CHECK(c, lim.E == 0.0);
            T63B3_CHECK(c, max_abs(lim.G) == 0.0);
            T63B3_CHECK(c, max_abs(lim.H) == 0.0);
        }

        // a negative strength: the second Hessian term is clamped to *zero* while the
        // first keeps its sign
        {
            const auto out = limit_local(th, th + 0.05, th + 0.6, -10.0, basis.l_basis, qk,
                                         qk_t, basis.r_basis, ql, ql_t);
            T63B3_CHECK(c, out.E < 0.0);
            T63B3_CHECK(c, out.d2E_dx2 < 0.0);
            T63B3_CHECK(c, all_finite(out.H));
            // the ported branchless `hessian` helper reproduces the oracle (whose second
            // term is `J^T make_spd(dE_dx ddDeltaTheta_ddF) J` with the *negative*
            // `dE_dx`, i.e. the clamp of an indefinite 12x12 -- a real projection)
            const Dense24 H_port = to_eigen24(
                gg::abd_revolute_joint_limit_detail::hessian<double>(
                    out.d2E_dx2, out.dE_dx, lv_of<24>(out.dx_dq), lv_of<6>(basis.l_basis),
                    gg::to_luisa(qk), gg::to_luisa(qk_t), lv_of<6>(basis.r_basis),
                    gg::to_luisa(ql), gg::to_luisa(ql_t)));
            T63B3_CHECK(c, dev(H_port, out.H) <= 1e-9);
            const Dense24 only_first = out.d2E_dx2 * (out.dx_dq * out.dx_dq.transpose());
            const double   extra      = max_abs(out.H - only_first);
            std::printf("[t63b3] degen: negative limit strength -> E %.6e dE_dx %.6e "
                        "d2E_dx2 %.6e, |H| %.6e (the second term's contribution %.3e)\n",
                        out.E, out.dE_dx, out.d2E_dx2, max_abs(out.H), extra);
            T63B3_CHECK(c, std::isfinite(extra));
        }
    }

    // ---- (D6) a large coefficient with tiny rest frames -------------------------
    {
        RevRest r;
        r.l0 = Vector3{1e-6, -2e-6, 3e-6};
        r.l1 = Vector3{4e-6, 1e-6, -1e-6};
        r.r0 = r.l0;  // a *coincident* pair (the same pose -> Faxis == 0 exactly)
        r.r1 = r.l1;
        const Vector12 qk = q_of(Matrix3x3::Identity(), Vector3{1e-6, -1e-6, 2e-6});
        const Vector12 ql = qk;
        const double   k  = 1e12;
        T63B3_CHECK(c, rj_E(k, r, qk, ql) == 0.0);
        T63B3_CHECK(c, all_finite(Dense24v(rj_G(k, r, qk, ql))));
        T63B3_CHECK(c, all_finite(rj_H(k, r, qk, ql)));
        // (the *same pose* test above is exact; a *nearby* pose stays finite)
        const Vector12 ql2 = q_of(1.0000001 * Matrix3x3::Identity(), Vector3{1e-6, -1e-6, 2e-6});
        T63B3_CHECK(c, std::isfinite(rj_E(k, r, qk, ql2)));
        T63B3_CHECK(c, all_finite(Dense24v(rj_G(k, r, qk, ql2))));
        std::printf("[t63b3] degen: kappa 1e12 with 1e-6 rest frames -> E %.3e |G| %.3e |H| "
                    "%.3e (no conditioning cliff)\n",
                    rj_E(k, r, qk, ql2), max_abs(rj_G(k, r, qk, ql2)),
                    max_abs(rj_H(k, r, qk, ql2)));
    }

    std::printf("[t63b3] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}


// ===========================================================================
//  4) the assembly fan-out (differencing two real assemblies) + the three
//     energy kernels + the driving's directly-driven kernels
// ===========================================================================
TEST_CASE("abd_joint_c_assembly", "[luisa][abd_joint_c]")
{
    test_runtime();
    CaseScope scope{"abd_joint_c_assembly"};

    RevScene scene_data;

    // ---- (A) the same 2-body scene twice, with and without the joints -----------
    Engine engine_a{"luisa", workspace_path("uipc_luisa_abd_joint_c_base")};
    World  world_a{engine_a};
    Scene  base_scene = scene_data.build("abd_joint_c_base", false);
    world_a.init(base_scene);
    REQUIRE(world_a.is_valid());
    REQUIRE(T63B3Probe::abd != nullptr);
    REQUIRE(T63B3Probe::abd_linear != nullptr);
    REQUIRE(T63B3Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B3Probe::abd_linear);


    auto*      linear_a = T63B3Probe::abd_linear;
    const auto base     = run_assembly(*linear_a);

    Engine engine_b{"luisa", workspace_path("uipc_luisa_abd_joint_c_joint")};
    World  world_b{engine_b};
    Scene  joint_scene = scene_data.build("abd_joint_c_joint", true);
    world_b.init(joint_scene);
    REQUIRE(world_b.is_valid());
    REQUIRE(T63B3Probe::abd != nullptr);
    REQUIRE(T63B3Probe::abd_linear != nullptr);
    REQUIRE(T63B3Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B3Probe::abd_linear);
    auto*      abd_b    = T63B3Probe::abd;
    auto*      linear_b = T63B3Probe::abd_linear;
    const auto joint_run = run_assembly(*linear_b);

    auto& c = scope.counter;

    // ---- (B) uid ownership (proved, see the file comment) ----------------------
    const auto* joint  = T63B3Probe::joint;
    const auto* drv    = T63B3Probe::driving;
    const auto* lim    = T63B3Probe::limit;
    const auto* econst = T63B3Probe::ext_constraint;

    const bool joint_owns = joint != nullptr && joint->is_valid() && joint->h_body_ids.size() == 2u;
    const bool drv_owns   = drv != nullptr && drv->is_valid() && drv->h_is_constrained.size() == 2u;
    const bool lim_owns   = lim != nullptr && lim->is_valid() && lim->h_lowers.size() == 2u;
    const bool econst_owns = econst != nullptr && econst->is_valid() && econst->h_torques.size() == 2u;
    std::printf("[t63b3] assembly uid ownership: uid18 = %s (%zu joints), uid19 = %s (%zu), "
                "uid670 = %s (%zu), uid668(constraint) = %s (%zu)\n",
                joint_owns ? "uipc::backend::luisa::AffineBodyRevoluteJoint" : "ANOTHER SYSTEM",
                joint ? joint->h_body_ids.size() : 0u,
                drv_owns ? "uipc::backend::luisa::AffineBodyDrivingRevoluteJoint"
                         : "ANOTHER SYSTEM",
                drv ? drv->h_is_constrained.size() : 0u,
                lim_owns ? "uipc::backend::luisa::AffineBodyRevoluteJointLimit"
                         : "ANOTHER SYSTEM",
                lim ? lim->h_lowers.size() : 0u,
                econst_owns
                    ? "uipc::backend::luisa::AffineBodyRevoluteJointExternalForceConstraint"
                    : "ANOTHER SYSTEM",
                econst ? econst->h_torques.size() : 0u);
    REQUIRE(joint_owns);
    REQUIRE(drv_owns);
    REQUIRE(lim_owns);
    REQUIRE(econst_owns);

    // ---- (C) the extent budget difference --------------------------------------
    T63B3_CHECK(c, base.dof_count == 24u);
    T63B3_CHECK(c, base.block_count == 2u);
    T63B3_CHECK(c, joint_run.dof_count == 24u);
    // hand-derived: the two joints add their upper-triangular 2x2 12x12-block run (3
    // blocks each) AND so does the joint-limit constitution (its hook is driven by the
    // linear subsystem), i.e. 2 * (3 + 3) = 12 more blocks -> 12 * 16 more triplets.
    // The *driving* constraint's blocks are absent here because its extents are
    // reported by `InterAffineBodyAnimator::step`, which only runs in `advance()`
    // (the `scene` case measures the larger budget after one advance).
    T63B3_CHECK(c, joint_run.block_count == 2u + 12u);
    T63B3_CHECK(c, joint_run.triplet_count == base.triplet_count + 12u * 16u);
    std::printf("[t63b3] assembly extent: base %zu triplets / %zu blocks, joint %zu / %zu\n",
                base.triplet_count, base.block_count, joint_run.triplet_count,
                joint_run.block_count);

    const auto bodies = scene_bodies(world_b, *abd_b);
    REQUIRE(bodies.size() == 2u);
    const Vector12 q_body[2] = {bodies[0].q, bodies[1].q};

    // ---- (D) the three energy kernels, on the real dynamics windows -------------
    {
        const auto dt      = T63B3Probe::self->dt();
        const auto dt_attr = T63B3Probe::self->dt_attr();

        const auto e_joint = run_constitution_energy(*T63B3Probe::joint, *abd_b, dt_attr, dt, 2);
        const auto e_drv   = run_constraint_energy(*T63B3Probe::driving, *abd_b, dt_attr, dt, 2);
        const auto e_lim   = run_constitution_energy(*T63B3Probe::limit, *abd_b, dt_attr, dt, 2);
        REQUIRE(e_joint.size() == 2u);
        REQUIRE(e_drv.size() == 2u);
        REQUIRE(e_lim.size() == 2u);

        // the *same* `kappa = strength_ratio * (m_k + m_l)` every kernel uses
        const auto masses   = read_device(abd_b->body_masses());
        const auto kappa_of = [&](double ratio)
        { return ratio * (masses[0].m_mass + masses[1].m_mass); };

        double worst = 0.0, worst_drv = 0.0, worst_lim = 0.0;
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto fr = scene_frames(scene_data, edge);
            const auto kb = edge_k_body(edge);
            const auto lb = edge_l_body(edge);

            worst = std::max(
                worst, rel_err(e_joint[edge], rj_E(kappa_of(scene_data.ratio), fr.rest,
                                                   q_body[kb], q_body[lb])));

            // the driving's `theta_tilde` exactly as the kernel folds it
            const double th     = theta_of_F(F01q_of(fr.basis.l_basis, q_body[kb],
                                                     fr.basis.r_basis, q_body[lb]));
            const double init_a = joint->h_init_angles[edge];
            const double ref_a  = joint->h_current_angles[edge];
            const double th_eff = ref_a - init_a
                                  + std::remainder(th - (ref_a - init_a),
                                                   2.0 * std::numbers::pi);
            const double tt = scene_data.aim_angle - init_a - (th_eff - th);
            worst_drv       = std::max(worst_drv,
                                       rel_err(e_drv[edge],
                                               drj_E(kappa_of(scene_data.dr_ratio), fr.basis,
                                                     q_body[kb], q_body[lb], tt)));

            const double th_prev = ref_a - init_a;
            worst_lim            = std::max(
                worst_lim,
                rel_err(e_lim[edge],
                        penalty_ref(th_prev + delta_theta_q(fr.basis.l_basis, q_body[kb],
                                                            q_body[kb], fr.basis.r_basis,
                                                            q_body[lb], q_body[lb]),
                                    scene_data.low - init_a, scene_data.high - init_a,
                                    scene_data.lim_str)
                            .E));
        }
        std::printf("[t63b3] energy kernels: joint [%.8e %.8e], driving [%.8e %.8e], limit "
                    "[%.8e %.8e]; worst rel %.3e / %.3e / %.3e\n",
                    e_joint[0], e_joint[1], e_drv[0], e_drv[1], e_lim[0], e_lim[1], worst,
                    worst_drv, worst_lim);
        T63B3_CHECK(c, worst <= 1e-13);
        T63B3_CHECK(c, worst_drv <= 1e-13);
        T63B3_CHECK(c, worst_lim <= 1e-13);
        T63B3_CHECK(c, std::abs(e_joint[0]) > 1e-12);
        T63B3_CHECK(c, std::abs(e_drv[0]) > 1e-12);
        T63B3_CHECK(c, std::abs(e_lim[0]) > 1e-12);
    }

    // ---- (E) the assembled COO and gradient *difference* vs the hand-derived
    //      joint+limit contribution ------------------------------------------------
    const int body_of_uid0 = static_cast<int>(bodies[0].body);
    const int body_of_uid1 = static_cast<int>(bodies[1].body);

    const auto   masses2      = read_device(abd_b->body_masses());
    const double kappa_joint  = scene_data.ratio * (masses2[0].m_mass + masses2[1].m_mass);
    const double kappa_drv    = scene_data.dr_ratio * (masses2[0].m_mass + masses2[1].m_mass);

    {
        const int body_of[2] = {body_of_uid0, body_of_uid1};
        // the joint's + the joint-limit's blocks (no driving: the animator has not
        // stepped yet, so its extents are still zero)
        const BlockMap expected =
            scene_expected_blocks(scene_data, q_body, body_of, kappa_joint, kappa_drv,
                                  joint->h_init_angles.data(), joint->h_current_angles.data(),
                                  false);
        Eigen::VectorXd expected_g = Eigen::VectorXd::Zero(24);
        auto            scatter_into = [](Eigen::VectorXd& target, const Dense24v& G, int bk,
                                          int bl)
        {
            target.segment<12>(static_cast<Eigen::Index>(bk) * 12) += G.segment<12>(0);
            target.segment<12>(static_cast<Eigen::Index>(bl) * 12) += G.segment<12>(12);
        };

        for(int edge = 0; edge < 2; ++edge)
        {
            const auto fr = scene_frames(scene_data, edge);
            const auto kb = edge_k_body(edge);
            const auto lb = edge_l_body(edge);
            const int  bk = body_of[kb];
            const int  bl = body_of[lb];

            scatter_into(expected_g, rj_G(kappa_joint, fr.rest, q_body[kb], q_body[lb]), bk, bl);
            const double init_a  = joint->h_init_angles[edge];
            const double th_prev = joint->h_current_angles[edge] - init_a;
            const auto   lim     = limit_local(th_prev, scene_data.low - init_a,
                                               scene_data.high - init_a, scene_data.lim_str,
                                               fr.basis.l_basis, q_body[kb], q_body[kb],
                                               fr.basis.r_basis, q_body[lb], q_body[lb]);
            scatter_into(expected_g, lim.G, bk, bl);
        }

        const BlockMap produced = sub_blocks(joint_run.blocks, base.blocks);

        double worst_entry = 0.0, worst_block = 0.0;
        int    compared = 0, missing = 0, extra = 0;
        for(const auto& [key, blk] : expected)
        {
            auto it = produced.find(key);
            if(it == produced.end())
            {
                std::printf("[t63b3] assembly: missing key (%d, %d)\n", key.first, key.second);
                ++missing;
                continue;
            }
            worst_block = std::max(worst_block, dev(it->second, blk));
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                    worst_entry = std::max(worst_entry, std::abs(it->second(rr, cc) - blk(rr, cc)));
            ++compared;
        }
        for(const auto& [key, blk] : produced)
            if(expected.find(key) == expected.end() && max_abs(blk) > 0.0)
                ++extra;
        std::printf("[t63b3] assembly COO (joint+limit-only, by differencing): %d keys "
                    "compared, worst |entry| %.3e, worst |block| %.3e, missing %d, "
                    "non-zero extra %d\n",
                    compared, worst_entry, worst_block, missing, extra);
        T63B3_CHECK(c, missing == 0);
        T63B3_CHECK(c, extra == 0);
        // the *descending* pair (edge 1) shares its (min, max) key set with the
        // ascending one, so the two joints' off-diagonal blocks land on the same keys
        T63B3_CHECK(c, produced.size() == expected.size());
        T63B3_CHECK(c, compared == static_cast<int>(expected.size()));
        T63B3_CHECK(c, worst_entry <= 1e-6);
        T63B3_CHECK(c, worst_block > 0.0);
        T63B3_CHECK(c, worst_block <= 1e-8);  // the assembled blocks are *relative*

        Eigen::VectorXd produced_g = joint_run.gradient - base.gradient;
        double          worst_g    = 0.0;
        for(Eigen::Index i = 0; i < expected_g.size(); ++i)
            worst_g = std::max(worst_g, rel_err(produced_g(i), expected_g(i)));
        std::printf("[t63b3] assembly gradient (joint+limit-only): worst rel %.3e, "
                    "|b_joint| %.6e\n",
                    worst_g, expected_g.norm());
        T63B3_CHECK(c, expected_g.norm() > 1.0);
        T63B3_CHECK(c, worst_g <= 1e-11);

        // ---- (F) the absolute gradient (base + joints), which pins that the joint
        // really lands at the bodies' dof offsets
        Eigen::VectorXd expected_abs = base.gradient + expected_g;
        double          worst_abs    = 0.0;
        for(Eigen::Index i = 0; i < expected_abs.size(); ++i)
            worst_abs = std::max(worst_abs, rel_err(joint_run.gradient(i), expected_abs(i)));
        std::printf("[t63b3] assembly gradient (absolute): worst rel %.3e (|b| %.6e)\n",
                    worst_abs, expected_abs.norm());
        T63B3_CHECK(c, expected_abs.norm() > 1.0);
        T63B3_CHECK(c, worst_abs <= 1e-11);
    }

    // ---- (G) the driving constraint's k1/k2 kernels, driven directly -------------
    //  The `scene` case measures that the driving's *assembled* contribution needs the
    //  animator's step; this section isolates the kernel question by handing
    //  `do_compute_gradient_hessian` a `GradientHessianInfo` over test-owned COO
    //  buffers -- exactly the situation the animator's linear-subsystem reporter
    //  creates.
    {
        const auto dt      = T63B3Probe::self->dt();
        const auto dt_attr = T63B3Probe::self->dt_attr();
        const int  n_pairs = 2;

        lud::LDeviceBuffer<int32_t> g_idx(2 * n_pairs);
        lud::LDeviceBuffer<double>  g_val(12 * 2 * n_pairs);
        const int                   n_triplets = 3 * n_pairs;
        lud::LDeviceBuffer<int32_t> h_row(n_triplets), h_col(n_triplets);
        lud::LDeviceBuffer<double>  h_val(144 * n_triplets);
        {
            std::vector<int32_t> minus(n_triplets, -1);
            h_row.view().copy_from(span<const int32_t>{minus});
            h_col.view().copy_from(span<const int32_t>{minus});
            std::vector<double> zeros(144 * n_triplets, 0.0);
            h_val.view().copy_from(span<const double>{zeros});
            lud::main_stream().synchronize();
        }

        auto Gv = lla::LDoubletVectorView<Float, 12>::from_handles_full(
            g_idx.view(), g_val.view(), std::uint32_t{2 * n_pairs}, std::int32_t{0},
            std::int32_t{2 * n_pairs}, std::int32_t{2 * n_pairs}, std::int32_t{0},
            std::uint32_t{2 * n_pairs});
        auto Hv = lla::LTripletMatrixView<Float, 12, 12>::from_handles_full(
            h_row.view(), h_col.view(), h_val.view(), std::int32_t{8}, std::int32_t{8},
            std::int32_t{0}, std::int32_t{n_triplets}, std::int32_t{n_triplets},
            std::int32_t{0}, std::int32_t{0}, std::int32_t{8}, std::int32_t{8});

        gg::InterAffineBodyAnimator::Impl aimpl;
        aimpl.affine_body_dynamics = abd_b;
        aimpl.dt_attr              = dt_attr;
        aimpl.constraint_gradient_offsets_counts.resize(1);
        aimpl.constraint_gradient_offsets_counts.counts()[0] = static_cast<IndexT>(2 * n_pairs);
        aimpl.constraint_gradient_offsets_counts.scan();
        aimpl.constraint_hessian_offsets_counts.resize(1);
        aimpl.constraint_hessian_offsets_counts.counts()[0] = static_cast<IndexT>(n_triplets);
        aimpl.constraint_hessian_offsets_counts.scan();
        gg::InterAffineBodyAnimator::GradientHessianInfo ghi{&aimpl, 0, dt, Gv, Hv, false};
        T63B3Probe::driving->do_compute_gradient_hessian(ghi);
        gg::frame_stream().synchronize();

        const auto idx   = read_device(g_idx.cview());
        const auto vals  = read_device(g_val.cview());
        const auto rows  = read_device(h_row.cview());
        const auto cols  = read_device(h_col.cview());
        const auto hvals = read_device(h_val.cview());

        // the produced blocks: here a triplet *is* a 12x12 block (144 doubles) and its
        // row/col are the *raw body ids* of the pair
        std::map<std::pair<int, int>, Dense12> produced_drv;
        for(std::size_t t = 0; t < rows.size(); ++t)
        {
            Dense12 blk;
            for(int cc = 0; cc < 12; ++cc)
                for(int rr = 0; rr < 12; ++rr)
                    blk(rr, cc) = hvals[t * 144 + static_cast<std::size_t>(cc * 12 + rr)];
            auto key = std::make_pair(static_cast<int>(rows[t]), static_cast<int>(cols[t]));
            auto it  = produced_drv.find(key);
            if(it == produced_drv.end())
                produced_drv.emplace(key, blk);
            else
                it->second += blk;
        }

        const int       body_of2[2] = {body_of_uid0, body_of_uid1};
        Eigen::VectorXd expected_g2 = Eigen::VectorXd::Zero(24);
        std::map<std::pair<int, int>, Dense12> expected_h2;
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto fr = scene_frames(scene_data, edge);
            const auto kb = edge_k_body(edge);
            const auto lb = edge_l_body(edge);
            const int  bk = body_of2[kb];
            const int  bl = body_of2[lb];

            const double th     = theta_of_F(F01q_of(fr.basis.l_basis, q_body[kb],
                                                     fr.basis.r_basis, q_body[lb]));
            const double init_a = joint->h_init_angles[edge];
            const double ref_a  = joint->h_current_angles[edge];
            const double th_eff = ref_a - init_a
                                  + std::remainder(th - (ref_a - init_a),
                                                   2.0 * std::numbers::pi);
            const double tt = scene_data.aim_angle - init_a - (th_eff - th);

            const auto G = drj_G(kappa_drv, fr.basis, q_body[kb], q_body[lb], tt);
            expected_g2.segment<12>(static_cast<Eigen::Index>(bk) * 12) += G.segment<12>(0);
            expected_g2.segment<12>(static_cast<Eigen::Index>(bl) * 12) += G.segment<12>(12);

            const Dense24 H = drj_H(kappa_drv, fr.basis, q_body[kb], q_body[lb], tt);
            auto          add12 = [&](int rr, int cc, const Dense12& blk)
            {
                auto key = std::make_pair(rr, cc);
                auto it  = expected_h2.find(key);
                if(it == expected_h2.end())
                    expected_h2.emplace(key, blk);
                else
                    it->second += blk;
            };
            // the reference's `upper_LR` order: the off-diagonal block lands at
            // `(min, max)` and, when the pair descends, is the *transposed* member
            const bool ascending = bk < bl;
            add12(bk, bk, Dense12(H.block<12, 12>(0, 0)));
            add12(bl, bl, Dense12(H.block<12, 12>(12, 12)));
            add12(std::min(bk, bl), std::max(bk, bl),
                  ascending ? Dense12(H.block<12, 12>(0, 12)) : Dense12(H.block<12, 12>(12, 0)));
        }

        double worst_g2 = 0.0, worst_h2 = 0.0;
        int    compared2 = 0;
        for(const auto& [key, blk] : expected_h2)
        {
            auto it = produced_drv.find(key);
            if(it == produced_drv.end())
                continue;
            ++compared2;
            worst_h2 = std::max(worst_h2, dev(it->second, blk));
        }
        Eigen::VectorXd produced_g2 = Eigen::VectorXd::Zero(24);
        for(std::size_t t = 0; t < idx.size(); ++t)
            for(int k = 0; k < 12; ++k)
                produced_g2(static_cast<Eigen::Index>(idx[t]) * 12 + k) +=
                    vals[t * 12 + static_cast<std::size_t>(k)];
        for(Eigen::Index k = 0; k < expected_g2.size(); ++k)
            worst_g2 = std::max(worst_g2, rel_err(produced_g2(k), expected_g2(k)));

        std::printf("[t63b3] assembly: the driving's k1/k2 kernels driven directly: "
                    "gradient worst rel %.3e (|b| %.6e, produced %.6e), Hessian worst dev "
                    "%.3e (%d of %zu keys, %zu produced)\n",
                    worst_g2, expected_g2.norm(), produced_g2.norm(), worst_h2, compared2,
                    expected_h2.size(), produced_drv.size());
        T63B3_CHECK(c, expected_g2.norm() > 1.0);
        T63B3_CHECK(c, worst_g2 <= 1e-10);
        T63B3_CHECK(c, !expected_h2.empty());
        T63B3_CHECK(c, compared2 == static_cast<int>(expected_h2.size()));
        T63B3_CHECK(c, produced_drv.size() == expected_h2.size());
        T63B3_CHECK(c, worst_h2 <= 1e-8);
    }

    std::printf("[t63b3] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}


// ===========================================================================
//  5) a real scene: two ABD bodies, one revolute joint carrying a driving
//     instance, an external-torque instance and a limit instance, one `advance()`
// ===========================================================================
TEST_CASE("abd_joint_c_scene", "[luisa][abd_joint_c]")
{
    test_runtime();
    CaseScope scope{"abd_joint_c_scene"};

    RevScene scene_data;
    Scene    scene = scene_data.build("abd_joint_c_scene", true);

    Engine engine{"luisa", workspace_path("uipc_luisa_abd_joint_c_scene")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    auto& c = scope.counter;
    REQUIRE(T63B3Probe::abd != nullptr);
    REQUIRE(T63B3Probe::abd_linear != nullptr);
    REQUIRE(T63B3Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B3Probe::abd_linear);

    // The T6.3b3 hazard (measured: a SIGSEGV): this case initialises a *second* world
    // later on (the no-driving one), whose `init` re-points `T63B3Probe` at its own
    // systems -- so the *driving* world's systems and the joint's per-edge angles are
    // captured here and used from here on.
    auto* const j_ptr      = T63B3Probe::joint;
    auto* const drv_ptr    = T63B3Probe::driving;
    auto* const lim_ptr    = T63B3Probe::limit;
    auto* const abd_ptr    = T63B3Probe::abd;
    auto* const linear_ptr = T63B3Probe::abd_linear;
    auto* const dof_ptr    = T63B3Probe::joint_dof;
    REQUIRE(j_ptr != nullptr);
    REQUIRE(drv_ptr != nullptr);
    REQUIRE(lim_ptr != nullptr);
    REQUIRE(abd_ptr != nullptr);
    REQUIRE(linear_ptr != nullptr);
    REQUIRE(dof_ptr != nullptr);
    const double scene_init_a[2] = {j_ptr->h_init_angles[0], j_ptr->h_init_angles[1]};
    const double scene_cur_a[2]  = {j_ptr->h_current_angles[0], j_ptr->h_current_angles[1]};

    // ---- (A) the per-edge *host* bookkeeping, against the scene's own definitions --
    {
        REQUIRE(T63B3Probe::joint != nullptr);
        REQUIRE(T63B3Probe::joint->is_valid());
        const auto& j = *T63B3Probe::joint;
        REQUIRE(j.h_body_ids.size() == 2u);
        REQUIRE(j.h_rest_positions.size() == 2u);
        REQUIRE(j.h_l_basis.size() == 2u);
        REQUIRE(j.h_r_basis.size() == 2u);
        REQUIRE(j.h_strength_ratio.size() == 2u);
        REQUIRE(j.h_init_angles.size() == 2u);
        REQUIRE(j.h_current_angles.size() == 2u);

        auto cmp = [&](const auto& got, int n)  // a Vector6/Vector12 difference norm
        {
            double w = 0.0;
            for(int k = 0; k < n; ++k)
                w = std::max(w, std::abs(got(k)));
            return w;
        };
        double worst_rest = 0.0, worst_basis = 0.0, worst_angle = 0.0;
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto fr = scene_frames(scene_data, edge);
            worst_rest    = std::max(worst_rest, cmp(j.h_rest_positions[edge] - fr.rest.pack(), 12));
            worst_basis =
                std::max(worst_basis, cmp(j.h_l_basis[edge] - fr.basis.l_basis, 6));
            worst_basis =
                std::max(worst_basis, cmp(j.h_r_basis[edge] - fr.basis.r_basis, 6));
            // the measured angle == the hand relative angle of the scene state
            const auto   bodies0 = scene_bodies(world, *T63B3Probe::abd);
            const Vector12 qk    = bodies0[edge_k_body(edge)].q;
            const Vector12 ql    = bodies0[edge_l_body(edge)].q;
            const double th = theta_of_F(F01q_of(fr.basis.l_basis, qk, fr.basis.r_basis, ql));
            worst_angle = std::max(worst_angle,
                                   std::abs(j.h_current_angles[edge] - j.h_init_angles[edge] - th));
        }
        std::printf("[t63b3] scene host bookkeeping: %zu joints, worst |rest - hand| %.3e, "
                    "|basis - hand| %.3e, |angle - hand| %.3e, ratios %.3f/%.3f, init "
                    "%.1e/%.1e\n",
                    j.h_body_ids.size(), worst_rest, worst_basis, worst_angle,
                    j.h_strength_ratio[0], j.h_strength_ratio[1], j.h_init_angles[0],
                    j.h_init_angles[1]);
        T63B3_CHECK(c, worst_rest <= 1e-14);
        T63B3_CHECK(c, worst_basis <= 1e-14);
        T63B3_CHECK(c, worst_angle <= 1e-13);
        T63B3_CHECK(c, j.h_strength_ratio[0] == scene_data.ratio);
        T63B3_CHECK(c, j.h_strength_ratio[1] == scene_data.ratio);
        // the frontend's `init_angle` attribute is created zero and never written
        T63B3_CHECK(c, j.h_init_angles[0] == 0.0);
        T63B3_CHECK(c, j.h_init_angles[1] == 0.0);
        T63B3_CHECK(c, j.h_geo_joint_offsets_counts.total_count() == 2);
        // the ascending / descending pair split (the `upper_LR` path is exercised by
        // the *scene*, not only by the unit cases)
        T63B3_CHECK(c, j.h_body_ids[0](0) < j.h_body_ids[0](1));
        T63B3_CHECK(c, j.h_body_ids[1](0) > j.h_body_ids[1](1));

        // the *polar* rotation finding re-pinned (T6.3b1 §7.25 rule 3): the axle
        // direction is a `LT.rotation() * (l1 - l0)` with the *polar* rotation, so a
        // stretched body's linear part gives a different direction
        {
            const auto   fr   = scene_frames(scene_data, 0);
            const auto   l_w  = fr.rest.axle_world(scene_data.A0).normalized();
            const auto   a_l  = (scene_data.A0 * (fr.rest.l1 - fr.rest.l0)).normalized();
            const double dev_pol =
                (polar_rotation(scene_data.A0) * (fr.rest.l1 - fr.rest.l0).normalized() - l_w)
                    .norm();
            std::printf("[t63b3] scene frame: the axle direction is the *polar* one "
                        "(|R_polar t - R_linear t| %.6e > 0 -> the bodies are stretched)\n",
                        (l_w - a_l).norm());
            T63B3_CHECK(c, dev_pol <= 1e-14);
            T63B3_CHECK(c, (l_w - a_l).norm() > 1e-3);
        }

        // the device buffers mirror the host staging
        {
            const auto h_bids = read_device(j.body_ids.cview());
            REQUIRE(h_bids.size() == 2u);
            const auto h_rest = read_vec12(j.rest_positions.cview());
            REQUIRE(h_rest.size() == 2u);
            double worst = 0.0;
            for(std::size_t i = 0; i < h_bids.size(); ++i)
            {
                const auto e = gg::to_eigen(h_bids[i]);
                for(int k = 0; k < 2; ++k)
                    worst = std::max(worst,
                                     std::abs(static_cast<double>(e(k))
                                              - static_cast<double>(j.h_body_ids[i](k))));
                for(int k = 0; k < 12; ++k)
                    worst = std::max(worst,
                                     std::abs(h_rest[i](k) - j.h_rest_positions[i](k)));
            }
            T63B3_CHECK(c, worst == 0.0);
        }
    }

    // ---- (B) the three companion systems' host records --------------------------
    {
        REQUIRE(T63B3Probe::driving != nullptr);
        REQUIRE(T63B3Probe::driving->is_valid());
        REQUIRE(T63B3Probe::ext_constraint != nullptr);
        REQUIRE(T63B3Probe::ext_constraint->is_valid());
        REQUIRE(T63B3Probe::limit != nullptr);
        REQUIRE(T63B3Probe::limit->is_valid());

        const auto& drv    = *T63B3Probe::driving;
        const auto& econst = *T63B3Probe::ext_constraint;
        const auto& lim    = *T63B3Probe::limit;

        REQUIRE(drv.h_is_constrained.size() == 2u);
        T63B3_CHECK(c, drv.h_is_constrained[0] == scene_data.constrained);
        T63B3_CHECK(c, drv.h_strength_ratios[0] == scene_data.dr_ratio);
        T63B3_CHECK(c, drv.h_is_passive[0] == scene_data.passive);
        T63B3_CHECK(c, drv.h_aim_angles[0] == scene_data.aim_angle);
        T63B3_CHECK(c, drv.h_geo_joint_offsets_counts.total_count() == 2);

        REQUIRE(econst.h_torques.size() == 2u);
        T63B3_CHECK(c, econst.h_torques[0] == scene_data.ext_torque);
        T63B3_CHECK(c, econst.h_is_constrained[0] == scene_data.ext_constrained);

        REQUIRE(lim.h_lowers.size() == 2u);
        T63B3_CHECK(c, lim.h_lowers[0] == scene_data.low);
        T63B3_CHECK(c, lim.h_uppers[0] == scene_data.high);
        T63B3_CHECK(c, lim.h_strengths[0] == scene_data.lim_str);
    }

    // ---- (C) the dof layout and the frontend `angle` attribute -------------------
    const auto bodies = scene_bodies(world, *T63B3Probe::abd);
    REQUIRE(bodies.size() == 2u);
    const Vector12 q_body[2] = {bodies[0].q, bodies[1].q};

    {
        double worst = 0.0;
        int    counted = 0;
        auto   visitor   = uipc::backend::WorldVisitor{world};
        auto   geo_slots = visitor.scene().geometries();
        for(auto& slot : geo_slots)
        {
            auto sc = slot->geometry().as<SimplicialComplex>();
            if(!sc)
                continue;
            auto angle = sc->edges().find<Float>("angle");
            if(!angle)
                continue;
            const auto aview = view(*angle);
            if(aview.size() != 2)
                continue;
            for(int edge = 0; edge < 2; ++edge)
            {
                const auto   fr = scene_frames(scene_data, edge);
                const double th = theta_of_F(F01q_of(fr.basis.l_basis, q_body[edge_k_body(edge)],
                                                     fr.basis.r_basis, q_body[edge_l_body(edge)]));
                worst = std::max(worst, std::abs(aview[edge] - th));
                ++counted;
            }
        }
        std::printf("[t63b3] scene: the frontend `angle` attribute vs the hand relative "
                    "angle, worst |dev| %.3e (%d entries)\n",
                    worst, counted);
        T63B3_CHECK(c, counted == 2);
        T63B3_CHECK(c, worst <= 1e-12);
    }
    {
        gg::GlobalLinearSystem::Impl impl;
        impl.diag_subsystems.register_sim_system(*T63B3Probe::abd_linear);
        impl.init();
        T63B3_CHECK(c, impl.diag_dof_offsets_counts.total_count() == 24);
        T63B3_CHECK(c, bodies[0].dof_offset == 0u);
        T63B3_CHECK(c, bodies[1].dof_offset == 12u);
        REQUIRE(T63B3Probe::integrator != nullptr);
        T63B3_CHECK(c, T63B3Probe::integrator->is_valid());
        REQUIRE(T63B3Probe::dof_reporter != nullptr);
        T63B3_CHECK(c, T63B3Probe::dof_reporter->is_valid());
        std::printf("[t63b3] scene: dof count %d (the joint registers no dof of its own: "
                    "its reporter/integrator re-sync the per-joint angle state)\n",
                    static_cast<int>(impl.diag_dof_offsets_counts.total_count()));
    }

    // ---- (D) one advance(): the state, the prediction, the torque chain ----------
    auto feature = world.features().find<AffineBodyStateAccessorFeature>();
    REQUIRE(feature != nullptr);
    T63B3_CHECK(c, feature->body_count() == 2);

    const auto dt = T63B3Probe::self->dt();

    world.advance();
    REQUIRE(world.is_valid());
    world.sync();
    REQUIRE(world.is_valid());

    {
        const auto bodies_after = scene_bodies(world, *T63B3Probe::abd);
        REQUIRE(bodies_after.size() == 2u);

        // the ABD *state* is still the scene's initial state (the solve is T8.3's)
        double worst_state = 0.0;
        for(int i = 0; i < 2; ++i)
            for(int k = 0; k < 12; ++k)
                worst_state = std::max(worst_state, std::abs(bodies_after[i].q(k) - q_body[i](k)));

        // the BDF1 prediction, *including* the external-torque acceleration the
        // reporter chain produced
        const auto q_tilde = read_vec12(T63B3Probe::abd->q_tildes());
        const auto accs    = read_vec12(T63B3Probe::abd->body_external_force_accs());
        const auto forces  = read_vec12(T63B3Probe::abd->body_external_forces());
        REQUIRE(q_tilde.size() == 2u);
        REQUIRE(accs.size() == 2u);
        REQUIRE(forces.size() == 2u);

        const Matrix3x3 V[2] = {scene_data.V0, scene_data.V1};
        const Vector3   v[2] = {scene_data.v0, scene_data.v1};

        // the hand-derived force pair per edge: the axle from the edge's own rest
        // endpoints (`l1 - l0`, `r1 - r0`), symmetrised, then `torque_to_F`
        Dense12v F_body[2] = {Dense12v::Zero(), Dense12v::Zero()};
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto fr   = scene_frames(scene_data, edge);
            const auto kb   = edge_k_body(edge);
            const auto lb   = edge_l_body(edge);
            const auto ebar_i = fr.rest.l1 - fr.rest.l0;
            const auto ebar_j = fr.rest.r1 - fr.rest.r0;
            const auto ef     = ext_force_local(ebar_i, ebar_j, q_body[kb], q_body[lb]);
            auto       F_i    = gg::torque_to_F<double>(scene_data.ext_torque, lv3(ef.e_i),
                                                        gg::to_luisa(q_body[kb]));
            auto       F_j    = gg::torque_to_F<double>(scene_data.ext_torque, lv3(ef.e_j),
                                                        gg::to_luisa(q_body[lb]));
            F_body[kb] += eig_of<12>(F_i);
            F_body[lb] += eig_of<12>(F_j);
        }

        double worst_pred = 0.0, worst_force = 0.0, worst_acc = 0.0, worst_state2 = 0.0;
        for(int i = 0; i < 2; ++i)
        {
            const Vector12 qv    = gg::transform_to_q(make_transform(V[i], v[i]));
            const Dense12  M     = bodies[i].mass;
            const Dense12v a_ext = M.inverse() * F_body[i];
            const Vector12 expected =
                q_body[i] + static_cast<double>(dt) * qv + std::pow(dt, 2) * a_ext;

            for(int k = 0; k < 12; ++k)
            {
                worst_pred =
                    std::max(worst_pred, std::abs(q_tilde[bodies[i].body](k) - expected(k)));
                worst_force = std::max(worst_force,
                                       rel_err(forces[bodies[i].body](k), F_body[i](k)));
                worst_acc   = std::max(worst_acc, rel_err(accs[bodies[i].body](k), a_ext(k)));
                worst_state2 =
                    std::max(worst_state2, std::abs(bodies_after[i].q(k) - expected(k)));
            }
        }
        std::printf("[t63b3] scene: advance -> |q_state - q_initial| %.3e, "
                    "|q_tilde - (q + q_v dt + a_ext dt^2)| %.3e (dt %.6f), external "
                    "torque force rel %.3e, a_ext rel %.3e\n",
                    worst_state, worst_pred, static_cast<double>(dt), worst_force, worst_acc);
        T63B3_CHECK(c, worst_state == 0.0);
        T63B3_CHECK(c, worst_pred <= 1e-12);
        T63B3_CHECK(c, worst_force <= 1e-13);
        T63B3_CHECK(c, worst_acc <= 1e-12);
        T63B3_CHECK(c, Dense12v(F_body[0]).norm() > 1.0);
        T63B3_CHECK(c, worst_state2 > 1e-9);  // the reporter really moved the prediction
    }

    // ---- (E) the energy kernels + the assembled blocks at the advanced state -----
    {
        const auto dt_attr = T63B3Probe::self->dt_attr();
        const auto e_joint = run_constitution_energy(*j_ptr, *abd_ptr, dt_attr, dt, 2);
        const auto e_drv   = run_constraint_energy(*drv_ptr, *abd_ptr, dt_attr, dt, 2);
        const auto e_lim   = run_constitution_energy(*lim_ptr, *abd_ptr, dt_attr, dt, 2);

        const auto   masses  = read_device(T63B3Probe::abd->body_masses());
        const double k_joint = scene_data.ratio * (masses[0].m_mass + masses[1].m_mass);
        const double k_drv   = scene_data.dr_ratio * (masses[0].m_mass + masses[1].m_mass);

        double     worst = 0.0, worst_drv = 0.0, worst_lim = 0.0;
        const int body_of[2] = {static_cast<int>(bodies[0].body),
                                static_cast<int>(bodies[1].body)};
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto fr = scene_frames(scene_data, edge);
            const auto kb = edge_k_body(edge);
            const auto lb = edge_l_body(edge);

            worst = std::max(worst, rel_err(e_joint[edge], rj_E(k_joint, fr.rest, q_body[kb],
                                                                q_body[lb])));
            const double th     = theta_of_F(F01q_of(fr.basis.l_basis, q_body[kb],
                                                     fr.basis.r_basis, q_body[lb]));
            const double init_a = j_ptr->h_init_angles[edge];
            const double ref_a  = j_ptr->h_current_angles[edge];
            const double th_eff = ref_a - init_a
                                  + std::remainder(th - (ref_a - init_a),
                                                   2.0 * std::numbers::pi);
            const double tt = scene_data.aim_angle - init_a - (th_eff - th);
            worst_drv       = std::max(worst_drv,
                                       rel_err(e_drv[edge], drj_E(k_drv, fr.basis, q_body[kb],
                                                                  q_body[lb], tt)));
            const double th_prev = ref_a - init_a;
            worst_lim            = std::max(
                worst_lim,
                rel_err(e_lim[edge],
                        penalty_ref(th_prev + delta_theta_q(fr.basis.l_basis, q_body[kb],
                                                            q_body[kb], fr.basis.r_basis,
                                                            q_body[lb], q_body[lb]),
                                    scene_data.low - init_a, scene_data.high - init_a,
                                    scene_data.lim_str)
                            .E));
        }
        std::printf("[t63b3] scene energy kernels: joint %.8e/%.8e, driving %.8e/%.8e, "
                    "limit %.8e/%.8e; worst rel %.3e / %.3e / %.3e\n",
                    e_joint[0], e_joint[1], e_drv[0], e_drv[1], e_lim[0], e_lim[1], worst,
                    worst_drv, worst_lim);
        T63B3_CHECK(c, worst <= 1e-13);
        T63B3_CHECK(c, worst_drv <= 1e-13);
        T63B3_CHECK(c, worst_lim <= 1e-13);

        // the assembled *cross-body* keys are an absolute statement about the joints
        // (the same-body ones carry the ARAP body term, §7.25 rule 16)
        const auto run = run_assembly(*linear_ptr);
        T63B3_CHECK(c, run.dof_count == 24u);
        T63B3_CHECK(c, run.block_count == 2u + 18u);


        double worst_h  = 0.0;
        int    compared = 0;
        for(const auto& [key, blk] : scene_expected_blocks(scene_data, q_body, body_of,
                                                          k_joint, k_drv, scene_init_a,
                                                          scene_cur_a, true))
        {
            if(key.first / 4 == key.second / 4)
                continue;
            ++compared;
            auto it = run.blocks.find(key);
            REQUIRE(it != run.blocks.end());
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                    worst_h = std::max(worst_h, std::abs(it->second(rr, cc) - blk(rr, cc)));
        }
        std::printf("[t63b3] scene assembled joint blocks (cross-body): worst "
                    "|dev-hand-derived| %.3e (%d keys)\n",
                    worst_h, compared);
        T63B3_CHECK(c, compared == 16);
        T63B3_CHECK(c, worst_h <= 1e-8);
    }

    // ---- (F) the dof reporter's *winding adoption* and the integrator's re-sync ----
    {
        // A user-authored angle (a reset/teleport) is *adopted*: write `theta + 2pi`
        // into the frontend's `angle` attribute, let the reporter run, and the joint's
        // published angle + device state must both carry the new winding.
        auto visitor   = uipc::backend::WorldVisitor{world};
        auto geo_slots = visitor.scene().geometries();
        int  patched   = 0;
        double theta0  = 0.0;
        for(auto& slot : geo_slots)
        {
            auto sc = slot->geometry().as<SimplicialComplex>();
            if(!sc)
                continue;
            auto angle = sc->edges().find<Float>("angle");
            if(!angle)
                continue;
            auto aview = view(*angle);
            if(aview.size() != 2)
                continue;
            const auto fr = scene_frames(scene_data, 0);
            theta0 = theta_of_F(F01q_of(fr.basis.l_basis, q_body[0], fr.basis.r_basis, q_body[1]));
            for(int edge = 0; edge < 2; ++edge)
                aview[edge] = static_cast<Float>(aview[edge] + 2.0 * std::numbers::pi);
            ++patched;
        }
        REQUIRE(patched == 1);
        dof_ptr->update_dof_attributes();
        gg::frame_stream().synchronize();

        const auto angles_dev = read_device(j_ptr->current_angles.cview());
        REQUIRE(angles_dev.size() == 2u);
        double worst_adopt = 0.0;
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto   fr = scene_frames(scene_data, edge);
            const double th = theta_of_F(F01q_of(fr.basis.l_basis, q_body[edge_k_body(edge)],
                                                 fr.basis.r_basis, q_body[edge_l_body(edge)]));
            worst_adopt = std::max(worst_adopt,
                                   std::abs(static_cast<double>(angles_dev[edge])
                                            - (th + 2.0 * std::numbers::pi)));
        }
        std::printf("[t63b3] scene: the dof reporter adopted the user winding: worst "
                    "|current - (theta + 2pi)| %.3e (theta0 %.6f)\n",
                    worst_adopt, theta0);
        T63B3_CHECK(c, worst_adopt <= 1e-12);

        // ... and the *integrator's* hook (`do_update_state` -> `compute_current_angles`)
        // restores the device state from a poisoned buffer: poison, call the joint's
        // public re-derivation, and the value must come back (the same statements the
        // integrator runs).
        {
            std::vector<Float> wrong(j_ptr->h_current_angles.size(), -12345.0);
            j_ptr->current_angles.view().copy_from(span<const Float>{wrong});
            lud::main_stream().synchronize();
            j_ptr->compute_current_angles();
            gg::frame_stream().synchronize();
            const auto restored = read_device(j_ptr->current_angles.cview());
            REQUIRE(restored.size() == 2u);
            // the reference's contract: re-derive from the *pose*, using the stored
            // value only as the unwrap reference.  With a poisoned (-12345) reference
            // the result must still be a *representative* of the measured angle, i.e.
            // `restored - theta` is an exact multiple of 2*pi.
            double worst = 0.0;
            for(int edge = 0; edge < 2; ++edge)
            {
                const auto   fr = scene_frames(scene_data, edge);
                const double th = theta_of_F(F01q_of(fr.basis.l_basis,
                                                     q_body[edge_k_body(edge)],
                                                     fr.basis.r_basis,
                                                     q_body[edge_l_body(edge)]));
                worst = std::max(worst,
                                 std::abs(std::remainder(static_cast<double>(restored[edge]) - th,
                                                         2.0 * std::numbers::pi)));
            }
            std::printf("[t63b3] scene: the integrator's re-derivation from a poisoned "
                        "angle state: worst |remainder(current - theta, 2pi)| %.3e "
                        "(finite + a 2pi-representative)\n",
                        worst);
            T63B3_CHECK(c, worst <= 1e-9);
            T63B3_CHECK(c, std::isfinite(restored[0]));
        }
    }

    std::printf("[t63b3] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}


// ===========================================================================
//  6) device == host for every generic routine this batch ships
//
//     The kernels are file-local (their cached shaders live in anonymous
//     namespaces), so the device round dispatches a *test* kernel that calls the
//     *same* generic routines with `S = lcrt::Var<double>` and compares against the
//     host's `S = double`.  The 24-vectors / 24x24 matrices are stored as flat
//     doubles (the round's stride must be consumed exactly — the T6.3b2 rule-16
//     lesson).
// ===========================================================================
namespace
{
constexpr int kRoundStride = (1 + 24 + 576)          // joint E, G, H (24x24)
                             + (1 + 24 + 576)        // driving E, G, H
                             + 1                     // theta (compute_relative_angle)
                             + (1 + 24 + 144)        // DeltaTheta, dDeltaTheta_dQ, ddDeltaTheta_ddF
                             + (1 + 24 + 576)        // limit E, G, H
                             + 3                     // the three penalty scalars
                             + 1                     // unwrap_angle
                             + 3 + 3 + 12 + 12;      // e_i, e_j, F_i, F_j

// The round's per-state *field offsets* (one flat record per state, in the order the
// host reader consumes them).
constexpr int kOffJoint      = 0;                     // 601
constexpr int kOffDriving    = kOffJoint + 601;       // 601   (driving E, G, H)
// NOTE: kOffDriving + 25 .. kOffDriving + 600 is the driving's 24x24 element
// Hessian and kOffLimit + 25 .. kOffLimit + 600 the limit's: both stay *reserved*
// (written by nobody) because their device kernels deadlock LC's compiler -- see
// the note above the round kernels.  The reader still consumes them, so the
// stride stays exact and a layout slip is still caught by `kk == kRoundStride`.
constexpr int kOffTheta = kOffDriving + 601; // 1202
constexpr int kOffLimitChain = kOffTheta + 1;         // 1203
constexpr int kOffLimit      = kOffLimitChain + 169;  // 1372  (limit E, G, H)
constexpr int kOffPenalty    = kOffLimit + 601;       // 1973
constexpr int kOffUnwrap     = kOffPenalty + 3;       // 1976
constexpr int kOffExt        = kOffUnwrap + 1;        // 1977
static_assert(kOffExt + 3 + 3 + 12 + 12 == kRoundStride,
              "the round's field offsets must tile kRoundStride exactly");

// WHY THE ROUND IS SEVERAL KERNELS AND NOT ONE (T6.3b3 finding, measured).
// `luisa::compute::Device::compile` **deadlocks** (the process parks, CPU time and GPU
// utilisation freeze at 0, only a `taskkill` ends it) on a test-local kernel as soon as it
// carries one of the two big 12x12 `make_spd` chains -- the driving's element Hessian
// (`body_hessian`) or the limit's `hessian`.  Measured with `#if 0` bisection and
// per-ctor markers, all reproducible in isolation:
//   * one kernel holding joint + driving + limit + external force   -> deadlock
//   * disabling any *one* of {driving block, limit Hessian, external force} -> still deadlocks
//   * disabling *two* of them                                      -> compiles and runs
//   * a 5-kernel split: joint / driving E+G / limit chain / limit E+G / external
//     force all compile, and the two `make_spd` kernels above deadlock on their own.
// The same templates compile and run fine inside the *shipped* subsystem kernels (the
// assembly case drives the driving's real k1/k2 shaders and measures Hessian dev
// 1.709e-15), so this is an LC compile-path limitation triggered by these particular
// test kernels, not a defect in the transcription.  The device-vs-host round therefore
// covers every chain except those two Hessians, which `abd_joint_c_assembly` covers
// through the real subsystem path.
auto& t63b3_round_joint_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,         // out (flat)
                                                lcrt::Buffer<gg::LVector12>,  // rest_positions
                                                lcrt::Buffer<gg::LVector12>,  // qk
                                                lcrt::Buffer<gg::LVector12>,  // ql
                                                lcrt::Buffer<double>,         // kappas
                                                uint32_t>(                    // n
        [](lcrt::BufferVar<double>       out,
           lcrt::BufferVar<gg::LVector12> rest_positions,
           lcrt::BufferVar<gg::LVector12> qk_buf,
           lcrt::BufferVar<gg::LVector12> ql_buf,
           lcrt::BufferVar<double>        kappas,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto kappa = kappas->read(i);
                auto qk    = gg::load_vec12(qk_buf->read(i));
                auto ql    = gg::load_vec12(ql_buf->read(i));
                gg::abd_revolute_joint_detail::RevoluteJointRestT<lcrt::Var<double>> rj{
                    .rest_positions = gg::load_vec12(rest_positions->read(i))};

                int  k   = kOffJoint;
                auto put = [&](auto&& v) { out->write(i * kRoundStride + k++, v); };

                put(gg::abd_revolute_joint_detail::body_energy<lcrt::Var<double>>(kappa, rj, qk,
                                                                                 ql));
                auto G = gg::abd_revolute_joint_detail::body_gradient<lcrt::Var<double>>(
                    kappa, rj, qk, ql);
                for(int q = 0; q < 24; ++q)
                    put(G(q));
                auto H = gg::abd_revolute_joint_detail::body_hessian<lcrt::Var<double>>(
                    kappa, rj, qk, ql);
                for(int cidx = 0; cidx < 24; ++cidx)
                    for(int ridx = 0; ridx < 24; ++ridx)
                        put(H(ridx, cidx));
            };
        });
    return k;
}

auto& t63b3_round_driving_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,
                                                lcrt::Buffer<gg::LVector6>,   // l_basis
                                                lcrt::Buffer<gg::LVector6>,   // r_basis
                                                lcrt::Buffer<gg::LVector12>,  // qk
                                                lcrt::Buffer<gg::LVector12>,  // ql
                                                lcrt::Buffer<double>,         // kappas
                                                lcrt::Buffer<double>,         // theta_tildes
                                                uint32_t>(
        [](lcrt::BufferVar<double>       out,
           lcrt::BufferVar<gg::LVector6>  l_basis,
           lcrt::BufferVar<gg::LVector6>  r_basis,
           lcrt::BufferVar<gg::LVector12> qk_buf,
           lcrt::BufferVar<gg::LVector12> ql_buf,
           lcrt::BufferVar<double>        kappas,
           lcrt::BufferVar<double>        theta_tildes,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto kappa = kappas->read(i);
                auto tt    = theta_tildes->read(i);
                auto qk    = gg::load_vec12(qk_buf->read(i));
                auto ql    = gg::load_vec12(ql_buf->read(i));
                auto lb    = gg::load_vec6(l_basis->read(i));
                auto rb    = gg::load_vec6(r_basis->read(i));

                gg::abd_driving_revolute_joint_detail::DrivingRevoluteJointRestT<
                    lcrt::Var<double>>
                    dr{.l_basis = lb, .r_basis = rb};

                int  k   = kOffDriving;
                auto put = [&](auto&& v) { out->write(i * kRoundStride + k++, v); };

                put(gg::abd_driving_revolute_joint_detail::body_energy<lcrt::Var<double>>(
                    kappa, dr, qk, ql, tt));
                auto Gd = gg::abd_driving_revolute_joint_detail::body_gradient<
                    lcrt::Var<double>>(kappa, dr, qk, ql, tt);
                for(int q = 0; q < 24; ++q)
                    put(Gd(q));

                // ---- the angle helper (one slot, right after the driving block) -----
                out->write(i * kRoundStride + kOffTheta,
                           gg::compute_relative_angle<lcrt::Var<double>>(lb, qk, rb, ql));
            };
        });
    return k;
}


auto& t63b3_round_limit_chain_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,
                                                lcrt::Buffer<gg::LVector6>,   // l_basis
                                                lcrt::Buffer<gg::LVector6>,   // r_basis
                                                lcrt::Buffer<gg::LVector12>,  // qk
                                                lcrt::Buffer<gg::LVector12>,  // ql
                                                lcrt::Buffer<gg::LVector12>,  // qk_t
                                                lcrt::Buffer<gg::LVector12>,  // ql_t
                                                lcrt::Buffer<double>,         // x_in
                                                lcrt::Buffer<double>,         // lows
                                                lcrt::Buffer<double>,         // highs
                                                lcrt::Buffer<double>,         // strengths
                                                lcrt::Buffer<double>,         // y_in (unwrap ref)
                                                uint32_t>(
        [](lcrt::BufferVar<double>       out,
           lcrt::BufferVar<gg::LVector6>  l_basis,
           lcrt::BufferVar<gg::LVector6>  r_basis,
           lcrt::BufferVar<gg::LVector12> qk_buf,
           lcrt::BufferVar<gg::LVector12> ql_buf,
           lcrt::BufferVar<gg::LVector12> qk_t_buf,
           lcrt::BufferVar<gg::LVector12> ql_t_buf,
           lcrt::BufferVar<double>        x_ins,
           lcrt::BufferVar<double>        lows,
           lcrt::BufferVar<double>        highs,
           lcrt::BufferVar<double>        strengths,
           lcrt::BufferVar<double>        y_ins,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto x_in = x_ins->read(i);
                auto y_in = y_ins->read(i);
                auto lo   = lows->read(i);
                auto hi   = highs->read(i);
                auto str  = strengths->read(i);
                auto qk   = gg::load_vec12(qk_buf->read(i));
                auto ql   = gg::load_vec12(ql_buf->read(i));
                auto qk_t = gg::load_vec12(qk_t_buf->read(i));
                auto ql_t = gg::load_vec12(ql_t_buf->read(i));
                auto lb   = gg::load_vec6(l_basis->read(i));
                auto rb   = gg::load_vec6(r_basis->read(i));

                int  k   = kOffLimitChain;
                auto put = [&](auto&& v) { out->write(i * kRoundStride + k++, v); };

                put(gg::abd_revolute_joint_limit_detail::delta_theta<lcrt::Var<double>>(
                    lb, qk, qk_t, rb, ql, ql_t));
                auto dx_dq = gg::abd_revolute_joint_limit_detail::d_delta_theta_dq<
                    lcrt::Var<double>>(lb, qk, qk_t, rb, ql, ql_t);
                for(int q = 0; q < 24; ++q)
                    put(dx_dq(q));
                auto ddx_ddF =
                    gg::abd_revolute_joint_limit_detail::dd_delta_theta_ddF<lcrt::Var<double>>(
                        gg::abd_revolute_joint_limit_detail::pair_f<lcrt::Var<double>>(
                            lb, qk, rb, ql),
                        gg::abd_revolute_joint_limit_detail::pair_f<lcrt::Var<double>>(
                            lb, qk_t, rb, ql_t));
                for(int cidx = 0; cidx < 12; ++cidx)
                    for(int ridx = 0; ridx < 12; ++ridx)
                        put(ddx_ddF(ridx, cidx));

                // jump over the *reserved* limit E/G/H block (filled by
                // t63b3_round_limit_kernel; its Hessian half is the one that deadlocks)
                k = kOffPenalty;

                // ---- the three penalty scalars, then `unwrap_angle` ----------------
                put(gg::joint_limit::eval_penalty_energy<lcrt::Var<double>>(x_in, lo, hi, str));
                lcrt::Var<double> dE_only{0.0}, d2E_only{0.0};
                gg::joint_limit::eval_penalty_derivatives<lcrt::Var<double>>(
                    x_in, lo, hi, str, dE_only, d2E_only);
                put(dE_only);
                put(d2E_only);
                put(gg::unwrap_angle<lcrt::Var<double>>(x_in, y_in));
            };
        });
    return k;
}


auto& t63b3_round_limit_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,
                                                lcrt::Buffer<gg::LVector6>,   // l_basis
                                                lcrt::Buffer<gg::LVector6>,   // r_basis
                                                lcrt::Buffer<gg::LVector12>,  // qk
                                                lcrt::Buffer<gg::LVector12>,  // ql
                                                lcrt::Buffer<gg::LVector12>,  // qk_t
                                                lcrt::Buffer<gg::LVector12>,  // ql_t
                                                lcrt::Buffer<double>,         // x_in
                                                lcrt::Buffer<double>,         // lows
                                                lcrt::Buffer<double>,         // highs
                                                lcrt::Buffer<double>,         // strengths
                                                uint32_t>(
        [](lcrt::BufferVar<double>       out,
           lcrt::BufferVar<gg::LVector6>  l_basis,
           lcrt::BufferVar<gg::LVector6>  r_basis,
           lcrt::BufferVar<gg::LVector12> qk_buf,
           lcrt::BufferVar<gg::LVector12> ql_buf,
           lcrt::BufferVar<gg::LVector12> qk_t_buf,
           lcrt::BufferVar<gg::LVector12> ql_t_buf,
           lcrt::BufferVar<double>        x_ins,
           lcrt::BufferVar<double>        lows,
           lcrt::BufferVar<double>        highs,
           lcrt::BufferVar<double>        strengths,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto x_in = x_ins->read(i);
                auto lo   = lows->read(i);
                auto hi   = highs->read(i);
                auto str  = strengths->read(i);
                auto qk   = gg::load_vec12(qk_buf->read(i));
                auto ql   = gg::load_vec12(ql_buf->read(i));
                auto qk_t = gg::load_vec12(qk_t_buf->read(i));
                auto ql_t = gg::load_vec12(ql_t_buf->read(i));
                auto lb   = gg::load_vec6(l_basis->read(i));
                auto rb   = gg::load_vec6(r_basis->read(i));

                lcrt::Var<double> limE{0.0}, limdE{0.0}, limd2E{0.0};
                gg::joint_limit::eval_penalty<lcrt::Var<double>>(x_in, lo, hi, str, limE, limdE,
                                                                 limd2E);
                auto dx_dq = gg::abd_revolute_joint_limit_detail::d_delta_theta_dq<
                    lcrt::Var<double>>(lb, qk, qk_t, rb, ql, ql_t);

                int  k   = kOffLimit;
                auto put = [&](auto&& v) { out->write(i * kRoundStride + k++, v); };

                put(limE);
                auto GL = gg::abd_revolute_joint_limit_detail::gradient<lcrt::Var<double>>(
                    limdE, dx_dq);
                for(int q = 0; q < 24; ++q)
                    put(GL(q));
                // the 24x24 `hessian` is *not* round-tripped here -- see the note above
            };
        });
    return k;
}

auto& t63b3_round_ext_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,
                                                lcrt::Buffer<gg::LVector12>,  // rest_positions
                                                lcrt::Buffer<gg::LVector12>,  // qk
                                                lcrt::Buffer<gg::LVector12>,  // ql
                                                lcrt::Buffer<double>,         // torques
                                                uint32_t>(
        [](lcrt::BufferVar<double>       out,
           lcrt::BufferVar<gg::LVector12> rest_positions,
           lcrt::BufferVar<gg::LVector12> qk_buf,
           lcrt::BufferVar<gg::LVector12> ql_buf,
           lcrt::BufferVar<double>        torques,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto tau   = torques->read(i);
                auto qk    = gg::load_vec12(qk_buf->read(i));
                auto ql    = gg::load_vec12(ql_buf->read(i));
                auto X_bar = gg::load_vec12(rest_positions->read(i));

                int  k   = kOffExt;
                auto put = [&](auto&& v) { out->write(i * kRoundStride + k++, v); };

                lps::LVec<lcrt::Var<double>, 3> e_i, e_j;
                gg::abd_revolute_joint_external_force_detail::symmetrized_axes<
                    lcrt::Var<double>>(e_i, e_j,
                                       X_bar.template segment<3>(3)
                                           - X_bar.template segment<3>(0),
                                       X_bar.template segment<3>(9)
                                           - X_bar.template segment<3>(6),
                                       qk, ql);
                for(int q = 0; q < 3; ++q)
                    put(e_i(q));
                for(int q = 0; q < 3; ++q)
                    put(e_j(q));
                auto F_i = gg::torque_to_F<lcrt::Var<double>>(tau, e_i, qk);
                auto F_j = gg::torque_to_F<lcrt::Var<double>>(tau, e_j, ql);
                for(int q = 0; q < 12; ++q)
                    put(F_i(q));
                for(int q = 0; q < 12; ++q)
                    put(F_j(q));
            };
        });
    return k;
}
}  // namespace

TEST_CASE("abd_joint_c_device_vs_host", "[luisa][abd_joint_c]")
{
    test_runtime();
    CaseScope scope{"abd_joint_c_device_vs_host"};
    auto&     c = scope.counter;

    constexpr int N = 16;

    std::vector<Vector12> qks, qls, qks_t, qls_t;
    std::vector<Vector12> rests;
    std::vector<Vector6>  lbs, rbs;
    std::vector<double>   kappas, tts, xs, ls, hs, ss, taus;
    std::vector<RevRest>  rj_rests;
    std::vector<RevBasis> bases;
    for(int i = 0; i < N; ++i)
    {
        const Matrix3x3 Ak = random_admissible_A();
        const Matrix3x3 Al = random_admissible_A();
        const Vector3   tk = rand_vec(-0.4, 0.4);
        const Vector3   tl = rand_vec(-0.4, 0.4);
        RevRest         r;
        r.set_consistent(Ak, tk, Al, tl, rand_vec(-0.5, 0.5), rand_vec(-0.5, 0.5),
                         rand_uniform(-0.2, 0.2));
        RevBasis basis;
        basis.set(Ak, Al, r.axle_world(Ak));
        const Vector12 qk   = q_of(Ak, tk);
        const Vector12 ql   = q_of(Al, tl);
        const Vector12 qk_t = q_of(Ak, tk + rand_vec(-0.1, 0.1));
        const Vector12 ql_t = q_of(Al, tl + rand_vec(-0.1, 0.1));
        qks.push_back(qk);
        qls.push_back(ql);
        qks_t.push_back(qk_t);
        qls_t.push_back(ql_t);
        rests.push_back(r.pack());
        lbs.push_back(basis.l_basis);
        rbs.push_back(basis.r_basis);
        rj_rests.push_back(r);
        bases.push_back(basis);
        kappas.push_back(rand_uniform(0.5, 2.0));
        tts.push_back(rand_uniform(-0.5, 0.5));
        // the penalty input: half inside / half outside the interval, plus zero-width
        // cases so both the `w == 0` and the normalised branch are rounded
        const double lo = rand_uniform(-0.3, 0.1);
        const double hi = (i % 5 == 0) ? lo : lo + rand_uniform(0.05, 0.8);
        ls.push_back(lo);
        hs.push_back(hi);
        ss.push_back(rand_uniform(1.0, 50.0));
        xs.push_back(rand_uniform(lo - 0.5, hi + 0.5));
        taus.push_back(rand_uniform(-20.0, 20.0));
    }

    auto rests_d = upload_as<gg::LVector12>(rests, [](const Vector12& q) { return gg::to_luisa(q); });
    auto lbs_d   = upload_as<gg::LVector6>(lbs, [](const Vector6& v) { return gg::to_luisa(v); });
    auto rbs_d   = upload_as<gg::LVector6>(rbs, [](const Vector6& v) { return gg::to_luisa(v); });
    auto qks_d   = upload_as<gg::LVector12>(qks, [](const Vector12& q) { return gg::to_luisa(q); });
    auto qls_d   = upload_as<gg::LVector12>(qls, [](const Vector12& q) { return gg::to_luisa(q); });
    auto qks_t_d = upload_as<gg::LVector12>(qks_t, [](const Vector12& q) { return gg::to_luisa(q); });
    auto qls_t_d = upload_as<gg::LVector12>(qls_t, [](const Vector12& q) { return gg::to_luisa(q); });
    auto kd      = upload(kappas);
    auto ttd     = upload(tts);
    auto xd      = upload(xs);
    auto lod     = upload(ls);
    auto hid     = upload(hs);
    auto sd      = upload(ss);
    auto taud    = upload(taus);

    lud::LDeviceBuffer<double> out(static_cast<SizeT>(N) * kRoundStride);

    const auto n = static_cast<uint32_t>(N);
    auto& k_joint  = t63b3_round_joint_kernel();
    gg::frame_stream().record(
        k_joint(out.view(), rests_d.view(), qks_d.view(), qls_d.view(), kd.view(), n)
            .dispatch(N));

    auto& k_drv = t63b3_round_driving_kernel();
    gg::frame_stream().record(
        k_drv(out.view(), lbs_d.view(), rbs_d.view(), qks_d.view(), qls_d.view(), kd.view(),
              ttd.view(), n)
            .dispatch(N));

    auto& k_lc = t63b3_round_limit_chain_kernel();
    gg::frame_stream().record(
        k_lc(out.view(), lbs_d.view(), rbs_d.view(), qks_d.view(), qls_d.view(), qks_t_d.view(),
             qls_t_d.view(), xd.view(), lod.view(), hid.view(), sd.view(), hid.view(), n)
            .dispatch(N));

    auto& k_lim = t63b3_round_limit_kernel();
    gg::frame_stream().record(
        k_lim(out.view(), lbs_d.view(), rbs_d.view(), qks_d.view(), qls_d.view(), qks_t_d.view(),
              qls_t_d.view(), xd.view(), lod.view(), hid.view(), sd.view(), n)
            .dispatch(N));

    auto& k_ext = t63b3_round_ext_kernel();
    gg::frame_stream().record(
        k_ext(out.view(), rests_d.view(), qks_d.view(), qls_d.view(), taud.view(), n)
            .dispatch(N));

    gg::frame_stream().synchronize();

    const auto flat = read_device(out.cview());
    REQUIRE(flat.size() == static_cast<std::size_t>(N) * kRoundStride);

    double worst_rj_e = 0.0, worst_rj_g = 0.0, worst_rj_h = 0.0;
    double worst_drv_e = 0.0, worst_drv_g = 0.0;
    double worst_theta = 0.0, worst_delta = 0.0, worst_ddq = 0.0, worst_ddF = 0.0;
    double worst_lim_e = 0.0, worst_lim_g = 0.0;
    double worst_pen = 0.0, worst_pen2 = 0.0, worst_unwrap = 0.0;
    double worst_ext = 0.0;
    for(int i = 0; i < N; ++i)
    {
        const double* base = flat.data() + static_cast<std::size_t>(i) * kRoundStride;
        int           kk    = 0;
        auto          get   = [&]() { return base[kk++]; };
        auto          getv12 = [&]()
        {
            Dense12v v;
            for(int q = 0; q < 12; ++q)
                v(q) = base[kk++];
            return v;
        };
        auto getv24 = [&]()
        {
            Dense24v v;
            for(int q = 0; q < 24; ++q)
                v(q) = base[kk++];
            return v;
        };
        auto getv6 = [&]()
        {
            Dense6v v;
            for(int q = 0; q < 6; ++q)
                v(q) = base[kk++];
            return v;
        };
        auto getv3 = [&]()
        {
            Dense3v v;
            for(int q = 0; q < 3; ++q)
                v(q) = base[kk++];
            return v;
        };
        auto getH24 = [&]()
        {
            Dense24 H;
            for(int cidx = 0; cidx < 24; ++cidx)
                for(int ridx = 0; ridx < 24; ++ridx)
                    H(ridx, cidx) = base[kk++];
            return H;
        };
        auto getH12 = [&]()
        {
            Dense12 H;
            for(int cidx = 0; cidx < 12; ++cidx)
                for(int ridx = 0; ridx < 12; ++ridx)
                    H(ridx, cidx) = base[kk++];
            return H;
        };

        const auto&    rj  = rj_rests[i];
        const auto&    bs  = bases[i];
        const Vector12 qk  = qks[i];
        const Vector12 ql  = qls[i];
        const Vector12 qkt = qks_t[i];
        const Vector12 qlt = qls_t[i];
        const double   kp  = kappas[i];

        // ---- the joint --------------------------------------------------------
        {
            const auto loc = rev_local(kp, rj, qk, ql);
            worst_rj_e = std::max(worst_rj_e, rel_err(get(), loc.E));
            worst_rj_g = std::max(worst_rj_g, dev(getv24(), loc.G));
            worst_rj_h = std::max(worst_rj_h, dev(getH24(), loc.H));
        }
        // ---- the driving ------------------------------------------------------
        {
            const Dense12v F = F01q_of(bs.l_basis, qk, bs.r_basis, ql);
            const auto     J = J01q_of(bs.l_basis, bs.r_basis);
            const Dense24  Hraw = drj_H(kp, bs, qk, ql, tts[i]);
            worst_drv_e = std::max(worst_drv_e, rel_err(get(), E_driving_of(F, kp, tts[i])));
            worst_drv_g = std::max(worst_drv_g,
                                   dev(getv24(),
                                       J.transpose() * dE_driving_of(F, kp, tts[i])));
            // NOT device-round-tripped: the driving element Hessian is a 12x12
            // `make_spd` chain, and a test-local kernel that contains it deadlocks
            // LC's `Device::compile` (see the note above the round kernels).  The
            // entries are still consumed so the round's stride stays exact.
            (void)getH24();
            (void)Hraw;
        }
        // ---- the angle helpers + the limit chain ------------------------------
        {
            const double theta_port = get();
            const double theta_host = theta_of_F(F01q_of(bs.l_basis, qk, bs.r_basis, ql));
            worst_theta             = std::max(worst_theta, std::abs(theta_port - theta_host));

            const double d_port = get();
            const double d_law  = delta_theta_q(bs.l_basis, qk, qkt, bs.r_basis, ql, qlt);
            worst_delta         = std::max(worst_delta, std::abs(d_port - d_law));

            const auto ddq_port = getv24();
            const Dense12v F       = F01q_of(bs.l_basis, qk, bs.r_basis, ql);
            Dense12v       g_th;
            Dense12       H_th;
            theta_grad_hess_F(F, g_th, H_th);
            worst_ddq = std::max(worst_ddq,
                                 dev(ddq_port, J01q_of(bs.l_basis, bs.r_basis).transpose() * g_th));
            // NOTE: the 144 entries must be *consumed* here -- skipping them shifts every
            // later field and the deviations then read O(1) (the T6.3b2 rule-16 trap)
            const auto ddF_port = getH12();
            worst_ddF           = std::max(worst_ddF, dev(ddF_port, H_th));

            // the limit's E/G/H: the device's penalty input is `xs[i]`, so the oracle's
            // `theta_prev` is back-solved (`x = theta_prev + delta`)
            const double delta = delta_theta_q(bs.l_basis, qk, qkt, bs.r_basis, ql, qlt);
            const auto   ref   = penalty_ref(xs[i], ls[i], hs[i], ss[i]);
            const auto   lim   = limit_local(xs[i] - delta, ls[i], hs[i], ss[i], bs.l_basis,
                                             qk, qkt, bs.r_basis, ql, qlt);
            worst_lim_e = std::max(worst_lim_e, std::abs(get() - ref.E));
            worst_lim_g = std::max(worst_lim_g, dev(getv24(), lim.G));
            // NOT device-round-tripped (same `make_spd` deadlock as the driving
            // Hessian); consumed to keep the stride exact.
            (void)getH24();
            (void)lim.H;

            // the three penalty entry points
            worst_pen  = std::max(worst_pen, std::abs(get() - ref.E));
            worst_pen2 = std::max(worst_pen2, std::abs(get() - ref.dE));
            worst_pen2 = std::max(worst_pen2, std::abs(get() - ref.d2E));

            // `unwrap_angle(x, y)` with the device's two inputs, spelled the way the
            // reference does: `ref + remainder(wrapped - ref, 2*pi)` -- the
            // representative of `wrapped` nearest `ref` (the closed-forms case pins the
            // same identity against `std::remainder` directly)
            const double xv = xs[i];
            const double yv = hs[i];
            worst_unwrap    = std::max(
                worst_unwrap,
                std::abs(get() - (yv + std::remainder(xv - yv, 2.0 * std::numbers::pi))));
        }
        // ---- the external-force pair -----------------------------------------
        {
            const Dense12v X_bar = rests[i];
            const auto     ef    = ext_force_local(X_bar.segment<3>(3) - X_bar.segment<3>(0),
                                                  X_bar.segment<3>(9) - X_bar.segment<3>(6),
                                                  qk, ql);
            const auto     e_i   = getv3();
            const auto     e_j   = getv3();
            const Dense12v F_i   = getv12();
            const Dense12v F_j   = getv12();
            auto           F_i_law = gg::torque_to_F<double>(taus[i], lv3(ef.e_i),
                                                             gg::to_luisa(qk));
            auto           F_j_law = gg::torque_to_F<double>(taus[i], lv3(ef.e_j),
                                                             gg::to_luisa(ql));
            double w = 0.0;
            w = std::max(w, dev(e_i, ef.e_i));
            w = std::max(w, dev(e_j, ef.e_j));
            w = std::max(w, dev(Dense12v(eig_of<12>(F_i_law)), F_i));
            w = std::max(w, dev(Dense12v(eig_of<12>(F_j_law)), F_j));
            worst_ext = std::max(worst_ext, w);

            // the whole bag was consumed: a silent stride mismatch is the classic
            // layout trap (its last field deviates by exactly 1.0)
            if(i + 1 == N)
                T63B3_CHECK(c, kk == kRoundStride);
        }
    }

    std::printf("[t63b3] device vs host (%d states): joint E/G/H %.3e/%.3e/%.3e, driving "
                "E/G %.3e/%.3e (H not round-tripped: LC compile deadlock), theta %.3e, delta "
                "%.3e, dq %.3e, ddF %.3e, limit E/G %.3e/%.3e (H likewise), penalty "
                "%.3e/%.3e, unwrap %.3e, ext %.3e\n",
                N, worst_rj_e, worst_rj_g, worst_rj_h, worst_drv_e, worst_drv_g, worst_theta,
                worst_delta, worst_ddq, worst_ddF, worst_lim_e, worst_lim_g, worst_pen,
                worst_pen2, worst_unwrap, worst_ext);

    // the generic routines are the same statements, but the generated laws use
    // `lps::pow`/`lps::atan2` (lcrt::* on the DSL) and `make_spd`'s sweeps differ in
    // associativity: assert a per-routine bound and *record* the measurement
    T63B3_CHECK(c, (worst_rj_e <= 1e-12 && worst_rj_g <= 1e-12 && worst_rj_h <= 1e-12));
    T63B3_CHECK(c, (worst_drv_e <= 1e-12 && worst_drv_g <= 1e-12));
    T63B3_CHECK(c, worst_theta <= 1e-12);
    T63B3_CHECK(c, worst_delta <= 1e-12);
    T63B3_CHECK(c, worst_ddq <= 1e-12);
    T63B3_CHECK(c, worst_ddF <= 1e-12);
    T63B3_CHECK(c, (worst_lim_e <= 1e-12 && worst_lim_g <= 1e-12));
    T63B3_CHECK(c, (worst_pen <= 1e-12 && worst_pen2 <= 1e-12));
    T63B3_CHECK(c, worst_unwrap <= 1e-12);
    T63B3_CHECK(c, worst_ext <= 1e-12);

    std::printf("[t63b3] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  7) uid ownership of 18/19/668/670, *proved* (plus the 668 sharing)
// ===========================================================================
TEST_CASE("abd_joint_c_uid_ownership", "[luisa][abd_joint_c]")
{
    test_runtime();
    CaseScope scope{"abd_joint_c_uid_ownership"};
    auto&     c = scope.counter;

    RevScene scene_data;
    Scene    scene = scene_data.build("abd_joint_c_uid", true);
    Engine   engine{"luisa", workspace_path("uipc_luisa_abd_joint_c_uid")};
    World    world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    const bool joint_ok = T63B3Probe::joint != nullptr && T63B3Probe::joint->is_valid()
                          && T63B3Probe::joint->h_body_ids.size() == 2u;
    const bool drv_ok = T63B3Probe::driving != nullptr && T63B3Probe::driving->is_valid()
                        && T63B3Probe::driving->h_is_constrained.size() == 2u;
    const bool lim_ok = T63B3Probe::limit != nullptr && T63B3Probe::limit->is_valid()
                        && T63B3Probe::limit->h_lowers.size() == 2u;
    const bool econst_ok = T63B3Probe::ext_constraint != nullptr
                           && T63B3Probe::ext_constraint->is_valid()
                           && T63B3Probe::ext_constraint->h_torques.size() == 2u;
    const bool erep_ok = T63B3Probe::ext_reporter != nullptr
                         && T63B3Probe::ext_reporter->is_valid();

    std::printf("[t63b3] uid ownership: uid18 = %s (%zu joints), uid19 = %s (%zu), "
                "uid670 = %s (%zu), uid668 constraint = %s (%zu), uid668 reporter = %s\n",
                joint_ok ? "uipc::backend::luisa::AffineBodyRevoluteJoint" : "ANOTHER SYSTEM",
                T63B3Probe::joint ? T63B3Probe::joint->h_body_ids.size() : 0u,
                drv_ok ? "uipc::backend::luisa::AffineBodyDrivingRevoluteJoint"
                       : "ANOTHER SYSTEM",
                T63B3Probe::driving ? T63B3Probe::driving->h_is_constrained.size() : 0u,
                lim_ok ? "uipc::backend::luisa::AffineBodyRevoluteJointLimit"
                       : "ANOTHER SYSTEM",
                T63B3Probe::limit ? T63B3Probe::limit->h_lowers.size() : 0u,
                econst_ok
                    ? "uipc::backend::luisa::AffineBodyRevoluteJointExternalForceConstraint"
                    : "ANOTHER SYSTEM",
                T63B3Probe::ext_constraint ? T63B3Probe::ext_constraint->h_torques.size() : 0u,
                erep_ok ? "uipc::backend::luisa::AffineBodyRevoluteJointExternalForce"
                        : "ANOTHER SYSTEM");
    REQUIRE(joint_ok);
    REQUIRE(drv_ok);
    REQUIRE(lim_ok);
    REQUIRE(econst_ok);
    REQUIRE(erep_ok);

    // the declared uids, as the *registry* sees them (the 668 sharing is between two
    // different registries — the constraint's `constraint_uids` and the reporter's
    // `ExternalForceReporter` uid — exactly as the reference spells it)
    T63B3_CHECK(c, T63B3Probe::joint->get_uid() == 18ull);
    T63B3_CHECK(c, T63B3Probe::driving->get_uid() == 19ull);
    T63B3_CHECK(c, T63B3Probe::ext_constraint->get_uid() == 668ull);
    T63B3_CHECK(c, T63B3Probe::ext_reporter->get_uid() == 668ull);
    T63B3_CHECK(c, T63B3Probe::limit->get_uid() == 670ull);
    // ... and the base-class `uid()` accessor agrees (that one resolves through the
    // *table*, so it is the ownership statement)
    T63B3_CHECK(c, T63B3Probe::joint->uid() == 18ull);
    T63B3_CHECK(c, T63B3Probe::driving->uid() == 19ull);
    T63B3_CHECK(c, T63B3Probe::ext_constraint->uid() == 668ull);
    T63B3_CHECK(c, T63B3Probe::ext_reporter->uid() == 668ull);
    T63B3_CHECK(c, T63B3Probe::limit->uid() == 670ull);

    // the *cross-reference* the limit uses to find the joint geometry is the joint's
    // own uid, and the frontend keeps the base uid at 18 (670 is an *extra* uid)
    T63B3_CHECK(c, gg::AffineBodyRevoluteJointLimit::JointUID == 18ull);
    {
        auto visitor   = uipc::backend::WorldVisitor{world};
        auto geo_slots = visitor.scene().geometries();
        int  found     = 0;
        for(auto& slot : geo_slots)
        {
            auto sc = slot->geometry().as<SimplicialComplex>();
            if(!sc)
                continue;
            if(sc->edges().find<Float>("limit/lower") == nullptr)
                continue;
            ++found;
            auto base_uid = sc->meta().find<U64>(builtin::constitution_uid);
            REQUIRE(base_uid != nullptr);
            T63B3_CHECK(c, view(*base_uid)[0] == 18ull);
            auto extras = sc->meta().find<VectorXu64>(builtin::extra_constitution_uids);
            REQUIRE(extras != nullptr);
            bool has_670 = false;
            for(auto uid : view(*extras)[0])
                if(uid == 670ull)
                    has_670 = true;
            T63B3_CHECK(c, has_670);
            auto cuids = sc->meta().find<VectorXu64>(builtin::constraint_uids);
            REQUIRE(cuids != nullptr);
            bool has_drv = false, has_ext = false;
            for(auto uid : view(*cuids)[0])
            {
                if(uid == 19ull)
                    has_drv = true;
                if(uid == 668ull)
                    has_ext = true;
            }
            T63B3_CHECK(c, has_drv);
            T63B3_CHECK(c, has_ext);
        }
        std::printf("[t63b3] uid ownership: %d limit-bearing geometry (base uid 18, extra 670, "
                    "constraints {19, 668})\n",
                    found);
        T63B3_CHECK(c, found == 1);
    }

    std::printf("[t63b3] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  8) per-case assertion counts
// ===========================================================================
TEST_CASE("abd_joint_c_summary", "[luisa][abd_joint_c]")
{
    int total = 0;
    for(const auto& [name, checks] : g_case_counts)
    {
        std::printf("[t63b3] case %-42s %5d checks\n", name.c_str(), checks);
        total += checks;
        CHECK(checks > 0);
    }
    std::printf("[t63b3] summary: %d cases, %d checks\n",
                static_cast<int>(g_case_counts.size()), total);
}
