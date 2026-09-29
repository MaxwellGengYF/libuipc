/********************************************************************
 * @file luisa_abd_joint_b_test.cpp
 * @brief T6.3b2 — the ABD joints of the LuisaCompute backend, batch 2: the
 *        **prismatic joint** (uid 20) and its three companions — the driving
 *        constraint (uid 21), the external-force pair (constraint + reporter, uid
 *        667) and the joint-limit constitution (uid 669) — plus the joint's dof
 *        reporter and time integrator.
 *
 * Every oracle here is derived independently *in this file*: the reference backend is
 * CUDA-only and cannot be host-included, so nothing is copied from it.  The oracles are
 *
 *  (a) **closed forms**, written out at the assertion.  With the pair dof vector
 *      `q = [qi ; qj]` (12 dofs each, `A(i,j) = q(3+3i+j)`, §7.21 rule 6), the joint's
 *      four body-local `Vector6` rest rows (`[ci_bar | cj_bar]`, the two *axis*
 *      directions `[ti_bar | tj_bar]` and their orthonormal companions) and the
 *      abbreviation
 *
 *        a = -Ft = A_j cj_bar + tj - A_i ci_bar - ti        (3, the *negative* mismatch)
 *        b = A_i ti_bar ,  c = A_j tj_bar                   (3 each)
 *
 *      the generated laws are
 *
 *        F01 = [a ; b ; c]                                  (9)
 *        E01 = 1/2 kappa ( |a x b|^2 + |a x c|^2 )
 *        E23 = 1/2 kappa ( |A_i ni - A_j nj|^2 + |A_i bi - A_j bj|^2 )
 *        E   = E01 + E23
 *
 *      `E01` is a *quartic* polynomial in the state (`a` and `b` are both linear), so
 *      the element Hessian is **not** constant — and `ddE01ddF01` is *not* `kappa I`
 *      (unlike T6.3b1's joints), which makes the reference's `make_spd` clamp a real
 *      projection, not a no-op.  The file derives the three derivatives analytically:
 *
 *        G01 = dE01/dF01 = kappa [ a(B2+C2) - s b - t c ; A2 b - s a ; A2 c - t a ]
 *        H01 = d2E01/dF01^2 = kappa * [ (B2+C2)I - b b^T - c c^T ,  a b^T - s I ,  a c^T - t I ;
 *                                       b a^T - s I            ,  A2 I - a a^T,      0      ;
 *                                       c a^T - t I            ,      0      ,  A2 I - a a^T ]
 *        (A2 = |a|^2, B2 = |b|^2, C2 = |c|^2, s = a.b, t = a.c — the `sym/*.inl`
 *         statements are exactly this matrix, verified in the `closed_forms` case)
 *
 *      and the two fan-outs ride on the *constant* pair Jacobian
 *
 *        J01 = dF01/dq  (9x24):  row k : -I at ti(k), +I at tj(k),
 *                                -ci(m) at A_i(k,m), +cj(m) at A_j(k,m)
 *                                row 3+k : +ti_bar(m) at A_i(k,m)
 *                                row 6+k : +tj_bar(m) at A_j(k,m)
 *
 *      (`F01` is affine in `q`, so `G = J01^T G01 + dE23dQ` and
 *      `H = J01^T H01 J01 + ddE23ddQ` are the chain rule with a constant `J01` — the
 *      *opposite* split from T6.3b1, where the Jacobian was constant-by-law and the
 *      Hessian constant; `scratch/t63b2_jac.py` checks `J01` against a
 *      finite-difference Jacobian and shows the "body-local `12 + 3k + m`" trap
 *      deviating by 1.0).  `E23` contributes directly in `q` space:
 *        dE23dQ: kappa [P(k) ni(m) + Q(k) bi(m)] / -kappa [P(k) nj(m) + Q(k) bj(m)]
 *        ddE23ddQ: the two 12x12 A-blocks and their cross block, `delta_kk'`-blocked.
 *
 *      The **driving** joint (uid 21) uses the same `[a ; b ; c]` shape with
 *      `a = -Ft`, `b = A_i T_i`, `c = -(A_j T_j)`:
 *        Distance = a . (b - c)/2      (the signed distance along the mean axis)
 *        dxdF01   = [ (b-c)/2 ; a/2 ; -a/2 ]
 *        ddxdF01  = the *constant* [[0, I/2, -I/2]; [I/2, 0, 0]; [-I/2, 0, 0]]
 *        E01      = 1/2 kappa ( (b.a - d)^2 + (-c.a - d)^2 )
 *        dE01dF01 = kappa [ (u-d) b - (v-d) c ; (u-d) a ; -(v-d) a ]
 *        ddE01ddF01 = kappa * [ b b^T + c c^T , a b^T + (u-d)I , a c^T - (v-d)I ;
 *                               b a^T + (u-d)I ,      a a^T    ,       0        ;
 *                               c a^T - (v-d)I ,        0      ,      a a^T      ]
 *        (u = b.a, v = -c.a)
 *      and the **joint limit** (uid 669) is the cubic penalty of `joint_limit_penalty.h`
 *      on `x = Distance` with both bounds shifted by the joint's `init_distance`:
 *        E = penalty(x), dE_dx, d2E_dx2  ->  G = dE_dx * dx_dq,
 *        H = d2E_dx2 (dx_dq dx_dq^T) [+ J01^T make_spd(dE_dx * ddxdF01) J01 if
 *                                     dE_dx != 0]
 *      The case re-implements the reference's *branchy* penalty as a host oracle
 *      (inside / above / below / zero-width), because the port's version is the
 *      branchless `select` spelling required by RULE M2 (see
 *      `joint_limit_penalty.h`), and asserts the two agree in every branch,
 *      including the `w == 0` raw-cubic one.
 *  (b) central finite differences over the 24 pair dofs with an h-sweep, the symmetry,
 *      the rule-34 basis question (the fan-outs take their `H9x9` in the same `F01`
 *      order as their `dEdF` siblings, so the *plain* comparison is the agreement) and
 *      the state-independence of `J01` vs the state-*dependence* of `H01`.
 *  (c) degenerates pinned from the source: a coincident rest pair (`E == 0` and
 *      `|G| == 0` *bitwise*), `kappa = 0`, a reflection, a singular `A` (no division
 *      anywhere in either law), a zero-width limit range, a limit `x` inside the range,
 *      and a negative limit strength (whose `make_spd` clamps the second term to zero).
 *  (d) the fan-out into `ABDLinearSubsystem`'s assembly, isolated by *differencing*
 *      two real assemblies (the same 2-body scene with and without the joint
 *      geometries), the 16-way 3x3 unpacking, the `zero_out_lower` of the diagonal
 *      blocks, the reference's `upper_LR` on a *descending* pair, and the gradient
 *      scatter — plus the joint's/driving's/limit's *energy kernels* driven directly
 *      through the public `EnergyInfo` constructors.
 *  (e) a real scene: two ABD bodies, one prismatic joint carrying a driving instance,
 *      an external-force instance and a limit instance, one `advance()` — the frontend
 *      `distance` attribute against the hand-derived absolute distance, the BDF1
 *      prediction *including the external-force acceleration* the reporter chain
 *      produced, the dof layout, the joint's per-edge host bookkeeping against the
 *      hand-computed body-local frames, and the dof-reporter/time-integrator re-sync.
 *  (f) device == host for every generic routine this batch ships (the joint's, the
 *      driving's, the limit's, the penalty's and the external-force pair's) — *not*
 *      bitwise (the generated laws use `lps::pow`, and `make_spd`'s Jacobi sweeps
 *      differ in associativity), so per-routine bounds are asserted and recorded.
 *  (g) uid ownership for 20/21/667/669, *proved*: the owners are printed and
 *      REQUIREd, the 667 constraint/reporter sharing is measured on both registries.
 *
 * Coverage map:
 *  1) closed forms, both element laws, the derivatives, the clamp .... abd_joint_b_closed_forms
 *  2) central FD (g/H/symmetry/h-sweep) + the basis question ....... abd_joint_b_finite_difference
 *  3) rest/zero-kappa/reflection/singular + the penalty branches ... abd_joint_b_degenerate_inputs
 *  4) the assembly fan-out (COO, upper_LR, gradient) + the three
 *     energy kernels .............................................. abd_joint_b_assembly
 *  5) a real scene, one advance(), BDF1 + the ext-force acc,
 *     the dof reporter / time integrator .......................... abd_joint_b_scene
 *  6) device == host for the ported generic routines .............. abd_joint_b_device_vs_host
 *  7) uid ownership of 20/21/667/669 + the 667 sharing ............ abd_joint_b_uid_ownership
 *  8) per-case assertion counts ................................... abd_joint_b_summary
 *
 * UID OWNERSHIP.  Both constitutions (`20`, `669`) are sorted by uid in
 * `InterAffineBodyConstitutionManager::Impl::init` (a `std::ranges::sort`) and mapped
 * `uid -> index` in a loop, so with two equal-uid entries the **last** one owns the
 * geometry group; the registration order is `SimEngine::build_systems()`'s
 * demangled-type-name order (§7.23 rule 7).  `667` is shared by the *constraint*
 * registry and the *reporter* registry on purpose (the reference's own spelling, the
 * T5.4b `671` precedent).  This file does not assume any of that: the
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
#include <uipc/constitution/affine_body_prismatic_joint.h>
#include <uipc/constitution/affine_body_driving_prismatic_joint.h>
#include <uipc/constitution/affine_body_prismatic_joint_external_force.h>
#include <uipc/constitution/affine_body_prismatic_joint_limit.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

#include <affine_body/type_define.h>
#include <affine_body/utils.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/inter_affine_body_animator.h>
#include <affine_body/constitutions/affine_body_prismatic_joint.h>
#include <affine_body/constitutions/affine_body_driving_prismatic_joint.h>
#include <affine_body/constitutions/affine_body_prismatic_joint_external_force.h>
#include <affine_body/constitutions/affine_body_prismatic_joint_limit.h>
#include <affine_body/constitutions/affine_body_prismatic_joint_function.h>
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
using Dense3    = Eigen::Matrix<double, 3, 3>;
using Dense3v   = Eigen::Matrix<double, 3, 1>;
using Dense6v   = Eigen::Matrix<double, 6, 1>;
using Dense9    = Eigen::Matrix<double, 9, 9>;
using Dense9v   = Eigen::Matrix<double, 9, 1>;
using Dense24   = Eigen::Matrix<double, 24, 24>;
using Dense24v  = Eigen::Matrix<double, 24, 1>;
using Dense12   = Eigen::Matrix<double, 12, 12>;
using Dense12v  = Eigen::Matrix<double, 12, 1>;

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

Dense9v to_eigen9(const lps::LVec<double, 9>& v)
{
    Dense9v out;
    for(int i = 0; i < 9; ++i)
        out(i) = v(i);
    return out;
}

Dense9 to_eigen9(const lps::LMat<double, 9, 9>& m)
{
    Dense9 out;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
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
    static std::mt19937 gen{20240917u};
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

#define T63B2_CHECK(counter, cond) \
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

// ---------------------------------------------------------------------------
//  the joint's / driving joint's rest data in the reference's `[first | second]`
//  layout
// ---------------------------------------------------------------------------
struct PrismRest
{
    Vector3 ci = Vector3::Zero();  // rest_cs[0..2]
    Vector3 cj = Vector3::Zero();  // rest_cs[3..5]
    Vector3 ti = Vector3::Zero();  // rest_ts[0..2]   (the axis direction, body-local)
    Vector3 tj = Vector3::Zero();  // rest_ts[3..5]
    Vector3 ni = Vector3::Zero();  // rest_ns[0..2]
    Vector3 nj = Vector3::Zero();  // rest_ns[3..5]
    Vector3 bi = Vector3::Zero();  // rest_bs[0..2]
    Vector3 bj = Vector3::Zero();  // rest_bs[3..5]

    /// The unit-case spelling: the frames are free inputs, so the axis direction is
    /// the *input* `t` and the frames are `L^{-1} t`, `L^{-1} n`, `L^{-1} b` with
    /// `L = A` (the reference's `LT.rotation()` when the transform is a pure rotation).
    void set_unit(const Matrix3x3& A_i, const Matrix3x3& A_j, const Vector3& t = Vector3(1, 0, 0))
    {
        Vector3 ti_w = t.normalized();
        Vector3 ni_w, bi_w;
        hand_orthonormal_basis(ti_w, ni_w, bi_w);
        ti = A_i.inverse() * ti_w;
        ni = A_i.inverse() * ni_w;
        bi = A_i.inverse() * bi_w;
        tj = A_j.inverse() * ti_w;
        nj = A_j.inverse() * ni_w;
        bj = A_j.inverse() * bi_w;
    }

    /// The **scene** spelling (the T6.3b1 rule-3 finding re-pinned): the edge
    /// direction `t` is built from the bodies' *full* transforms
    /// (`t = (LT * lp1 - LT * lp0).normalized()`), while the rest rows divide by the
    /// *polar* rotation of those transforms (`LT.rotation() == computeRotationScaling`),
    /// so a stretched initial state does **not** give `A_i t_i == t`.
    void set_scene(const Matrix3x3& A_i, const Matrix3x3& A_j, const Vector3& t = Vector3(1, 0, 0))
    {
        Vector3 ti_w = t.normalized();
        Vector3 ni_w, bi_w;
        hand_orthonormal_basis(ti_w, ni_w, bi_w);
        const Matrix3x3 Li = polar_rotation(A_i).inverse();
        const Matrix3x3 Lj = polar_rotation(A_j).inverse();
        ti = Li * ti_w;
        ni = Li * ni_w;
        bi = Li * bi_w;
        tj = Lj * ti_w;
        nj = Lj * ni_w;
        bj = Lj * bi_w;
    }

    [[nodiscard]] lps::LVec<double, 6> rest_c() const { return lv6(ci, cj); }
    [[nodiscard]] lps::LVec<double, 6> rest_t() const { return lv6(ti, tj); }
    [[nodiscard]] lps::LVec<double, 6> rest_n() const { return lv6(ni, nj); }
    [[nodiscard]] lps::LVec<double, 6> rest_b() const { return lv6(bi, bj); }
};

// ===========================================================================
//  the analytic oracle
// ===========================================================================
/// `Ft = A_i ci + ti - (A_j cj + tj)` (the *physical* mismatch, without the
/// generated file's leading minus).
Dense3v Ft_of(const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    return A_of(qi) * r.ci + t_of(qi) - A_of(qj) * r.cj - t_of(qj);
}

/// `F01 = [-Ft ; A_i ti_bar ; A_j tj_bar]` (the measured sign convention: the first
/// three rows are the *negative* mismatch — a `+Ft` spelling is off by exactly 2x on
/// `J01T_G01`, which is what `scratch/t63b2_oracle.py` found).
Dense9v F01_of(const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    Dense9v out;
    out.segment<3>(0) = -Ft_of(r, qi, qj);
    out.segment<3>(3) = A_of(qi) * r.ti;
    out.segment<3>(6) = A_of(qj) * r.tj;
    return out;
}

/// The *constant* pair Jacobian `dF01/dq` (9x24) — the derivation is in the file
/// comment; `scratch/t63b2_jac.py` checks it against an FD Jacobian of the law above
/// and shows the "body-local offset" (`12 + 3k + m`) trap deviating by 1.0.
Eigen::Matrix<double, 9, 24> J01_of(const PrismRest& r)
{
    Eigen::Matrix<double, 9, 24> J = Eigen::Matrix<double, 9, 24>::Zero();
    for(int k = 0; k < 3; ++k)
    {
        J(k, k)      = -1.0;
        J(k, 12 + k) = +1.0;
        for(int m = 0; m < 3; ++m)
        {
            J(k, 3 + 3 * k + m)  = -r.ci(m);
            J(k, 15 + 3 * k + m) = +r.cj(m);
            // NOTE: the *pair* layout — A_j starts at 15 (12..14 is tj).
            J(3 + k, 3 + 3 * k + m)  = +r.ti(m);
            J(6 + k, 15 + 3 * k + m) = +r.tj(m);
        }
    }
    return J;
}

/// The driving joint's own pair Jacobian: the same shape as `J01_of`, but its third
/// block is `-(A_j T_j)`, so rows 6..8 carry `-tj_bar(m)` (measured with an FD
/// Jacobian of `F01q_of`: `scratch/t63b2_jac.py`, and re-measured in the
/// `finite_difference` case).
Eigen::Matrix<double, 9, 24> J01q_of(const PrismRest& r)
{
    auto J = J01_of(r);
    for(int k = 0; k < 3; ++k)
        for(int m = 0; m < 3; ++m)
            J(6 + k, 15 + 3 * k + m) = -r.tj(m);
    return J;
}

/// `E01 = 1/2 kappa (|a x b|^2 + |a x c|^2)` with `[a ; b ; c] = F01`.
double E01_of(const Dense9v& F01, double kappa)
{
    const Dense3v a = F01.segment<3>(0);
    const Dense3v b = F01.segment<3>(3);
    const Dense3v c = F01.segment<3>(6);
    return 0.5 * kappa * ((a.cross(b)).squaredNorm() + (a.cross(c)).squaredNorm());
}

/// `dE01/dF01` (analytic; see the derivation in the file comment).
Dense9v dE01dF01_of(const Dense9v& F01, double kappa)
{
    const Dense3v a = F01.segment<3>(0);
    const Dense3v b = F01.segment<3>(3);
    const Dense3v c = F01.segment<3>(6);
    const double  A2 = a.squaredNorm();
    const double  B2 = b.squaredNorm();
    const double  C2 = c.squaredNorm();
    const double  s = a.dot(b);
    const double  t = a.dot(c);
    Dense9v       out;
    out.segment<3>(0) = kappa * (a * (B2 + C2) - b * s - c * t);
    out.segment<3>(3) = kappa * (b * A2 - a * s);
    out.segment<3>(6) = kappa * (c * A2 - a * t);
    return out;
}

/// `d2E01/dF01^2` (analytic).
Dense9 d2E01dF01_of(const Dense9v& F01, double kappa)
{
    const Dense3v a  = F01.segment<3>(0);
    const Dense3v b  = F01.segment<3>(3);
    const Dense3v c  = F01.segment<3>(6);
    const double  A2 = a.squaredNorm();
    const double  B2 = b.squaredNorm();
    const double  C2 = c.squaredNorm();
    const double  s  = a.dot(b);
    const double  t  = a.dot(c);

    Dense9 H = Dense9::Zero();
    H.block<3, 3>(0, 0) =
        (B2 + C2) * Dense3::Identity() - b * b.transpose() - c * c.transpose();
    // NOTE (measured, sympy-verified): the cross blocks are NOT `a b^T - s I` —
    // `d2E01/dF01^2 [a_i, b_j] = kappa (2 a_i b_j - b_i a_j - s delta_ij)`, because
    // `|a x b|^2`'s second derivative in (a, b) has both orderings.  A transposed
    // spelling passes the *gradient* check and fails the Hessian one (measured:
    // dev 1.02 against the FD of the same law).
    H.block<3, 3>(0, 3) = 2 * a * b.transpose() - b * a.transpose() - s * Dense3::Identity();
    H.block<3, 3>(0, 6) = 2 * a * c.transpose() - c * a.transpose() - t * Dense3::Identity();
    H.block<3, 3>(3, 3) = A2 * Dense3::Identity() - a * a.transpose();
    H.block<3, 3>(6, 6) = A2 * Dense3::Identity() - a * a.transpose();
    H.block<3, 3>(3, 0) = H.block<3, 3>(0, 3).transpose();
    H.block<3, 3>(6, 0) = H.block<3, 3>(0, 6).transpose();
    return kappa * H;
}

/// `E23` and its two analytic `q`-space derivatives, packed as one struct.
struct E23Local
{
    double                E = 0.0;
    Dense24v              G = Dense24v::Zero();
    Dense24               H = Dense24::Zero();
    Dense3v               P, Q;  // the two mismatches, for the assertions
};

E23Local E23_local(const PrismRest& r, double kappa, const Vector12& qi, const Vector12& qj)
{
    E23Local out;
    const Matrix3x3 Ai = A_of(qi);
    const Matrix3x3 Aj = A_of(qj);
    out.P = Ai * r.ni - Aj * r.nj;
    out.Q = Ai * r.bi - Aj * r.bj;
    out.E = 0.5 * kappa * (out.P.squaredNorm() + out.Q.squaredNorm());

    // gradient: dE/dA_i(k,m) = kappa (P(k) ni(m) + Q(k) bi(m)),
    //           dE/dA_j(k,m) = -kappa (P(k) nj(m) + Q(k) bj(m))
    for(int k = 0; k < 3; ++k)
    {
        for(int m = 0; m < 3; ++m)
        {
            out.G(3 + 3 * k + m)  = kappa * (out.P(k) * r.ni(m) + out.Q(k) * r.bi(m));
            out.G(15 + 3 * k + m) = -kappa * (out.P(k) * r.nj(m) + out.Q(k) * r.bj(m));
        }
    }

    // Hessian: the two 3x3-blocked diagonal A-blocks (one 3x3 per row `k`) and the
    // single cross block pair.  `delta_kk'` blocks the *row* index only, so the
    // `(m, m')` matrix inside each block is fully populated, and the cross block's
    // transpose counterpart takes the transposed matrix (`D` / `D^T`).
    for(int k = 0; k < 3; ++k)
    {
        Dense3 D = Dense3::Zero();
        for(int m = 0; m < 3; ++m)
            for(int m2 = 0; m2 < 3; ++m2)
            {
                D(m, m2) = -kappa * (r.ni(m) * r.nj(m2) + r.bi(m) * r.bj(m2));
                out.H(3 + 3 * k + m, 3 + 3 * k + m2) +=
                    kappa * (r.ni(m) * r.ni(m2) + r.bi(m) * r.bi(m2));
                out.H(15 + 3 * k + m, 15 + 3 * k + m2) +=
                    kappa * (r.nj(m) * r.nj(m2) + r.bj(m) * r.bj(m2));
            }
        out.H.block<3, 3>(3 + 3 * k, 15 + 3 * k) += D;
        out.H.block<3, 3>(15 + 3 * k, 3 + 3 * k) += D.transpose();
    }
    return out;
}

/// The reference's `make_spd(M) = V max(w,0) V^T` (an *exact* clamp; the shipped one
/// is the layer's fixed-sweep Jacobi approximation of it).
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

/// The whole prismatic joint law at a state: E/G/H, the element pieces, and the
/// *shipped* H (the element Hessian clamped by the reference's `make_spd` before its
/// fan-out).  `H_raw` is the unclamped chain — the two differ whenever `H01` is
/// indefinite, which is the rule-19 statement for this batch.
struct PrismLocal
{
    double  E01 = 0.0, E23 = 0.0, E = 0.0;
    Dense9v F01;
    Dense9v G01;  // dE01/dF01
    Dense9   H01;  // d2E01/dF01^2 (raw)
    Dense9   H01_spd;  // exact_spd(H01)
    Dense24v G, G23;
    Dense24  H_raw, H;  // H = J01^T H01_spd J01 + ddE23ddQ
    Dense24  H23;
};

PrismLocal prism_local(double kappa, const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    PrismLocal out;
    out.F01  = F01_of(r, qi, qj);
    out.E01  = E01_of(out.F01, kappa);
    out.G01  = dE01dF01_of(out.F01, kappa);
    out.H01  = d2E01dF01_of(out.F01, kappa);
    out.H01_spd = exact_spd(out.H01);

    const auto e23 = E23_local(r, kappa, qi, qj);
    out.E23 = e23.E;
    out.G23 = e23.G;
    out.H23 = e23.H;
    out.E   = out.E01 + out.E23;

    const auto J01 = J01_of(r);
    out.G          = J01.transpose() * out.G01 + out.G23;
    out.H_raw      = J01.transpose() * out.H01 * J01 + out.H23;
    out.H          = J01.transpose() * out.H01_spd * J01 + out.H23;
    return out;
}

// ---------------------------------------------------------------------------
//  the driving joint + the joint limit, analytic
// ---------------------------------------------------------------------------
Dense9v F01q_of(const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    Dense9v out;
    out.segment<3>(0) = -Ft_of(r, qi, qj);
    out.segment<3>(3) = A_of(qi) * r.ti;
    out.segment<3>(6) = -(A_of(qj) * r.tj);
    return out;
}

/// `Distance = a . (b - c)/2`.
double distance_of(const Dense9v& F01q)
{
    const Dense3v a = F01q.segment<3>(0);
    const Dense3v b = F01q.segment<3>(3);
    const Dense3v c = F01q.segment<3>(6);
    return a.dot((b - c) / 2.0);
}

Dense9v dxdF01_of(const Dense9v& F01q)
{
    const Dense3v a = F01q.segment<3>(0);
    const Dense3v b = F01q.segment<3>(3);
    const Dense3v c = F01q.segment<3>(6);
    Dense9v       out;
    out.segment<3>(0) = (b - c) / 2.0;
    out.segment<3>(3) = a / 2.0;
    out.segment<3>(6) = -a / 2.0;
    return out;
}

/// The *constant* `[[0, I/2, -I/2]; [I/2, 0, 0]; [-I/2, 0, 0]]`.
Dense9 ddxdF01_of()
{
    Dense9 H = Dense9::Zero();
    H.block<3, 3>(0, 3) = 0.5 * Dense3::Identity();
    H.block<3, 3>(0, 6) = -0.5 * Dense3::Identity();
    H.block<3, 3>(3, 0) = 0.5 * Dense3::Identity();
    H.block<3, 3>(6, 0) = -0.5 * Dense3::Identity();
    return H;
}

/// `E01 = 1/2 kappa ((b.a - d)^2 + (-c.a - d)^2)`.
double driving_E01_of(const Dense9v& F01q, double kappa, double d)
{
    const Dense3v a = F01q.segment<3>(0);
    const Dense3v b = F01q.segment<3>(3);
    const Dense3v c = F01q.segment<3>(6);
    return 0.5 * kappa * (std::pow(b.dot(a) - d, 2) + std::pow(-c.dot(a) - d, 2));
}

Dense9v driving_dE01dF01_of(const Dense9v& F01q, double kappa, double d)
{
    const Dense3v a  = F01q.segment<3>(0);
    const Dense3v b  = F01q.segment<3>(3);
    const Dense3v c  = F01q.segment<3>(6);
    const double  u  = b.dot(a);
    const double  v  = -c.dot(a);
    Dense9v       out;
    out.segment<3>(0) = kappa * ((u - d) * b - (v - d) * c);
    out.segment<3>(3) = kappa * (u - d) * a;
    out.segment<3>(6) = -kappa * (v - d) * a;
    return out;
}

Dense9 driving_d2E01dF01_of(const Dense9v& F01q, double kappa, double d)
{
    const Dense3v a  = F01q.segment<3>(0);
    const Dense3v b  = F01q.segment<3>(3);
    const Dense3v c  = F01q.segment<3>(6);
    const double  u  = b.dot(a);
    const double  v  = -c.dot(a);
    Dense9        H  = Dense9::Zero();
    H.block<3, 3>(0, 0) = b * b.transpose() + c * c.transpose();
    // NOTE (measured, sympy-verified): `d2E01/dF01^2 [a_i, b_j] = kappa (b_i a_j +
    // (u-d) delta_ij)` and `[a_i, c_j] = kappa (c_i a_j - (v-d) delta_ij)` — the
    // *other* ordering than the prismatic joint's (whose `E01` is quartic), so the two
    // laws need their own spelling and neither may be transcribed from the other.
    H.block<3, 3>(0, 3) = b * a.transpose() + (u - d) * Dense3::Identity();
    H.block<3, 3>(0, 6) = c * a.transpose() - (v - d) * Dense3::Identity();
    H.block<3, 3>(3, 0) = H.block<3, 3>(0, 3).transpose();
    H.block<3, 3>(6, 0) = H.block<3, 3>(0, 6).transpose();
    H.block<3, 3>(3, 3) = a * a.transpose();
    H.block<3, 3>(6, 6) = a * a.transpose();
    return kappa * H;
}

/// The *host* oracle of `joint_limit_penalty.h`: the reference's branch structure,
/// written out independently (the port is branchless, RULE M2).
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
    Dense9v  F01q;
    Dense24v dx_dq;
    Dense24v G;
    Dense24  H;
};

LimitLocal limit_local(double x_offset,
                       double lower,
                       double upper,
                       double strength,
                       const PrismRest& r,
                       const Vector12&  qi,
                       const Vector12&  qj)
{
    LimitLocal out;
    out.F01q = F01q_of(r, qi, qj);
    out.x    = distance_of(out.F01q) + x_offset;

    const auto p = penalty_ref(out.x, lower, upper, strength);
    out.E        = p.E;
    out.dE_dx    = p.dE;
    out.d2E_dx2  = p.d2E;

    const auto J01q = J01q_of(r);  // the driving's own (negative third block)
    out.dx_dq       = J01q.transpose() * dxdF01_of(out.F01q);
    out.G           = out.dE_dx * out.dx_dq;

    const Dense9 ddxd = ddxdF01_of();
    out.H             = out.d2E_dx2 * (out.dx_dq * out.dx_dq.transpose());
    if(out.dE_dx != 0.0)
    {
        const Dense9 H_D = exact_spd(out.dE_dx * ddxd);
        out.H += J01q.transpose() * H_D * J01q;
    }
    return out;
}

/// The external-force reporter's arithmetic (the reference's two lines + the pair).
struct ExtForceLocal
{
    Dense3v t_i, t_j;
    Dense12v F_i, F_j;
};

ExtForceLocal ext_force_local(double f,
                              const PrismRest& r,
                              const Vector12&  qi,
                              const Vector12&  qj)
{
    ExtForceLocal out;
    // J.vec_x(q) = A x_bar (rotation/scaling only)
    const Dense3v t_i_raw = A_of(qi) * r.ti;
    const Dense3v t_j_raw = A_of(qj) * r.tj;
    out.t_j               = 0.5 * (t_i_raw + t_j_raw);
    out.t_i               = -out.t_j;
    out.F_i               = Dense12v::Zero();
    out.F_j               = Dense12v::Zero();
    out.F_i.segment<3>(0) = f * out.t_i;
    out.F_j.segment<3>(0) = f * out.t_j;
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

Dense24v pack_pair(const Vector12& qi, const Vector12& qj)
{
    Dense24v q;
    q.segment<12>(0)  = qi;
    q.segment<12>(12) = qj;
    return q;
}
}  // namespace


// ===========================================================================
//  the probe: the built ABD/joint systems of one scene
// ===========================================================================
namespace
{
class T63B2Probe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::AffineBodyDynamics*                         abd          = nullptr;
    inline static gg::ABDLinearSubsystem*                         abd_linear   = nullptr;
    inline static gg::AffineBodyPrismaticJoint*                   joint        = nullptr;
    inline static gg::AffineBodyDrivingPrismaticJoint*            driving      = nullptr;
    inline static gg::AffineBodyPrismaticJointExternalForceConstraint* ext_constraint = nullptr;
    inline static gg::AffineBodyPrismaticJointExternalForce*      ext_reporter = nullptr;
    inline static gg::AffineBodyPrismaticJointLimit*              limit        = nullptr;
    inline static gg::AffineBodyPrismaticJointDofReporter*        dof_reporter = nullptr;
    inline static gg::AffineBodyPrismaticJointTimeIntegrator*     integrator   = nullptr;
    inline static gg::GlobalJointDofManager*                      joint_dof    = nullptr;
    inline static T63B2Probe*                                     self         = nullptr;

    /// The scene's `dt` -- the *same* attribute the joint kernels read.
    [[nodiscard]] Float dt()
    {
        auto attr = world().scene().config().find<Float>("dt");
        UIPC_ASSERT(attr, "T63B2Probe: the scene config has no `dt`");
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
        joint          = find<gg::AffineBodyPrismaticJoint>();
        driving        = find<gg::AffineBodyDrivingPrismaticJoint>();
        ext_constraint = find<gg::AffineBodyPrismaticJointExternalForceConstraint>();
        ext_reporter   = find<gg::AffineBodyPrismaticJointExternalForce>();
        limit          = find<gg::AffineBodyPrismaticJointLimit>();
        dof_reporter   = find<gg::AffineBodyPrismaticJointDofReporter>();
        integrator     = find<gg::AffineBodyPrismaticJointTimeIntegrator>();
        joint_dof      = find<gg::GlobalJointDofManager>();
    }
};
REGISTER_SIM_SYSTEM(T63B2Probe);
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
/// `rest_t/n/b = L_rot^{-1} t/n/b` really is a polar-rotation inverse, T6.3b1 rule 3),
/// and the joints' local anchors/axis endpoints are arbitrary body-local points.
/// `with_joint == false` builds the identical scene *without* the joint geometry: the
/// assembly case differences the two, isolating the joint's contribution without
/// needing an oracle for the ARAP body term.
struct PrismScene
{
    Matrix3x3 A0 = Eigen::AngleAxisd(0.6, Vector3{0.3, -1.0, 0.5}.normalized()).toRotationMatrix()
                   * Vector3{1.25, 0.9, 1.1}.asDiagonal();
    Matrix3x3 A1 = Eigen::AngleAxisd(-1.1, Vector3{1.0, 0.4, -0.7}.normalized()).toRotationMatrix()
                   * Vector3{0.85, 1.3, 1.0}.asDiagonal();
    Vector3 t0 = Vector3{0.2, -0.1, 0.3};
    Vector3 t1 = Vector3{-0.4, 0.25, 0.15};
    Matrix3x3 V0 = 0.5 * Matrix3x3::Identity();
    Matrix3x3 V1 = 0.15 * Matrix3x3::Identity();
    Vector3   v0 = Vector3::Zero();
    Vector3   v1 = Vector3{0.1, -0.15, 0.05};

    /// Pair 0 (ascending: `l = body 0`, `r = body 1`): the two endpoints of the axis
    /// in body 0's frame, and the attachment point in body 1's frame.  Pair 1 is the
    /// *descending* pair (`l = body 1`, `r = body 0`) with its own body-local data, so
    /// the two joints are not coincidentally symmetric.
    Vector3 l_pos0_0  = Vector3{0.10, 0.05, -0.02};
    Vector3 l_pos1_0  = Vector3{0.40, -0.05, 0.18};
    Vector3 r_pos0_0  = Vector3{-0.02, 0.08, 0.03};
    Vector3 r_pos1_0  = Vector3{0.23, 0.13, -0.17};
    Vector3 l_pos0_1  = Vector3{0.35, -0.05, 0.10};
    Vector3 l_pos1_1  = Vector3{0.55, 0.25, -0.05};
    Vector3 r_pos0_1  = Vector3{0.05, -0.02, 0.20};
    Vector3 r_pos1_1  = Vector3{0.15, -0.27, 0.50};
    Float   ratio     = 100.0;  // the joint's strength ratio (both pairs)
    Float   dr_ratio  = 30.0;   // the driving instance's strength ratio
    Float   aim       = 0.21;   // the driving aim distance
    IndexT  passive   = 0;
    IndexT  constrained = 1;    // driving/is_constrained
    Float   ext_force   = 12.5;  // the external-force scalar
    IndexT  ext_constrained = 1;  // external_force/is_constrained
    Float   low        = -0.05;  // limit/lower
    Float   high       = 0.62;   // limit/upper
    Float   lim_str    = 40.0;   // limit/strength

    /// `with_driving == false` builds the same scene *without* the driving instance
    /// (the driving's `is_constrained` then keeps the frontend's default 0), which
    /// is how the scene case isolates the driving's own assembled contribution.
    [[nodiscard]] Scene build(const std::string& name,
                              bool              with_joint,
                              bool              with_driving = true) const
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
            std::vector<IndexT> l_ids{0, 0};
            std::vector<IndexT> r_ids{0, 0};
            std::vector<Vector3> l_pos0{l_pos0_0, l_pos0_1};
            std::vector<Vector3> l_pos1{l_pos1_0, l_pos1_1};
            std::vector<Vector3> r_pos0{r_pos0_0, r_pos0_1};
            std::vector<Vector3> r_pos1{r_pos1_0, r_pos1_1};
            std::vector<Float>   ratios{ratio, ratio};

            uipc::constitution::AffineBodyPrismaticJoint joint;
            auto joint_geo = joint.create_geometry(span<const Vector3>(l_pos0),
                                                   span<const Vector3>(l_pos1),
                                                   span<const Vector3>(r_pos0),
                                                   span<const Vector3>(r_pos1),
                                                   span<S<geometry::SimplicialComplexSlot>>(l_slots),
                                                   span<IndexT>(l_ids),
                                                   span<S<geometry::SimplicialComplexSlot>>(r_slots),
                                                   span<IndexT>(r_ids),
                                                   span<Float>(ratios));

            // the driving instance (uid 21): the frontend's apply_to zeroes
            // `driving/is_constrained` and `aim_distance`, so the scene's values are
            // written afterwards (through the very attribute views the kernel reads).
            // NOTE: *every* edge is written — `.front()` alone leaves the second
            // joint unconstrained, which silently zeroes its energy (measured).
            if(with_driving)
            {
                uipc::constitution::AffineBodyDrivingPrismaticJoint driving;
                driving.apply_to(joint_geo, dr_ratio);
                std::ranges::fill(
                    view(*joint_geo.edges().find<IndexT>("driving/is_constrained")),
                    constrained);
                std::ranges::fill(view(*joint_geo.edges().find<IndexT>("is_passive")),
                                  passive);
                std::ranges::fill(
                    view(*joint_geo.edges().find<Float>("aim_distance")), aim);
            }

            // the external-force instance (uid 667, constraint + reporter)
            uipc::constitution::AffineBodyPrismaticJointExternalForce ext_force_c;
            ext_force_c.apply_to(joint_geo, ext_force);
            std::ranges::fill(
                view(*joint_geo.edges().find<IndexT>("external_force/is_constrained")),
                ext_constrained);

            // the joint-limit instance (uid 669, an `extra_constitution_uid` on the
            // same geometry)
            uipc::constitution::AffineBodyPrismaticJointLimit limit;
            limit.apply_to(joint_geo, low, high, lim_str);

            object->geometries().create(joint_geo);
        }

        return scene;
    }
};

/// The `l`-side body index of an edge (`edge 0` is the ascending pair, `edge 1` the
/// descending one).
int edge_l_body(int edge)
{
    return edge == 0 ? 0 : 1;
}
int edge_r_body(int edge)
{
    return edge == 0 ? 1 : 0;
}

/// The joint's per-edge rest data as the *scene* defines it (the `do_init` formulas,
/// re-derived here from the bodies' initial transforms and the local endpoints).
PrismRest scene_rest(const PrismScene& s, int edge)
{
    const bool      l_is_body0 = (edge == 0);
    const Matrix3x3 Ai         = l_is_body0 ? s.A0 : s.A1;
    const Matrix3x3 Aj         = l_is_body0 ? s.A1 : s.A0;
    const Vector3   lp0        = l_is_body0 ? s.l_pos0_0 : s.l_pos0_1;
    const Vector3   lp1        = l_is_body0 ? s.l_pos1_0 : s.l_pos1_1;
    const Vector3   rp0        = l_is_body0 ? s.r_pos0_0 : s.r_pos0_1;

    // the reference: `t = (LT * lp1 - LT * lp0).normalized()` -- the *full* transform,
    // whose linear part acts on the difference.
    Vector3 t_world = (Ai * (lp1 - lp0)).normalized();
    Vector3 n_world, b_world;
    hand_orthonormal_basis(t_world, n_world, b_world);

    const Matrix3x3 Li = polar_rotation(Ai).inverse();
    const Matrix3x3 Lj = polar_rotation(Aj).inverse();
    PrismRest       r;
    r.ci = lp0;
    r.cj = rp0;
    r.ti = Li * t_world;
    r.ni = Li * n_world;
    r.bi = Li * b_world;
    r.tj = Lj * t_world;
    r.nj = Lj * n_world;
    r.bj = Lj * b_world;
    return r;
}

Vector6 vec6(const Vector3& a, const Vector3& b)
{
    Vector6 v;
    v.segment<3>(0) = a;
    v.segment<3>(3) = b;
    return v;
}

/// One pair's expected *host* bookkeeping.
struct ExpectedPair
{
    Vector6 c, t, n, b;
    Float   init = 0.0;
};

ExpectedPair expected_pair(const PrismScene& s, int edge)
{
    const auto r = scene_rest(s, edge);
    return ExpectedPair{vec6(r.ci, r.cj), vec6(r.ti, r.tj), vec6(r.ni, r.nj), vec6(r.bi, r.bj)};
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
//
//  A joint's `ComputeEnergyInfo` is an `InterAffineBodyConstitutionManager::EnergyInfo`
//  (public constructor + public Impl), and the driving constraint's is an
//  `InterAffineBodyAnimator::ComputeEnergyInfo` (same story), so both can be driven on
//  the real device windows.
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

// forward declarations of the helpers that live *later* in this anonymous
// namespace (`pj_H`/`dpj_H` come from the ported-routine wrappers below,
// `add_joint_blocks` from the assembly section)
Dense24 pj_H(double kappa, const PrismRest& r, const Vector12& qi, const Vector12& qj);
Dense24 dpj_H(double kappa,
              const PrismRest& r,
              const Vector12& qi,
              const Vector12& qj,
              double          d);
void    add_joint_blocks(BlockMap& out, int bi, int bj, const Dense24& H_local);

/// The scene's expected *assembled* contribution from the three joint-side systems.
/// The joint-limit's blocks are present as soon as the scene is initialised (it is a
/// *constitution*: the linear subsystem drives its `do_compute_gradient_hessian`
/// directly), while the *driving constraint*'s blocks only exist once the animator has
/// stepped — its extents are reported by `InterAffineBodyAnimator::step`, so before
/// the first `advance()` an assembly silently omits them (measured: 14 vs 20
/// twelve-by-twelve blocks in the two `*_assembly` cases).  `with_driving`
/// distinguishes the two situations.
BlockMap scene_expected_blocks(const PrismScene& s,
                               const Vector12    q_body[2],
                               const int         body_of[2],
                               double            k_joint,
                               double            k_drv,
                               bool              with_driving,
                               bool              driving_only = false)
{
    BlockMap out;
    for(int edge = 0; edge < 2; ++edge)
    {
        const auto r  = scene_rest(s, edge);
        const auto lb = edge_l_body(edge);
        const auto rb = edge_r_body(edge);
        const int  bi = body_of[lb];
        const int  bj = body_of[rb];

        if(!driving_only)
            add_joint_blocks(out, bi, bj, pj_H(k_joint, r, q_body[lb], q_body[rb]));

        if(!driving_only)
        {
            const auto lim =
                limit_local(0.0, s.low, s.high, s.lim_str, r, q_body[lb], q_body[rb]);
            add_joint_blocks(out, bi, bj, lim.H);
        }

        if(with_driving)
            add_joint_blocks(out, bi, bj, dpj_H(k_drv, r, q_body[lb], q_body[rb], s.aim));
    }
    return out;
}

/// The joint's expected contribution to the global 3x3-block COO: the two diagonal
/// 12x12 blocks `zero_out_lower`-ed, the off-diagonal one placed by the reference's
/// `upper_LR` (i.e. at `(min(bi,bj), max(bi,bj))`, with the block content the
/// corresponding 12x12 sub-block of the *local* H), each unpacked into 16 3x3 triplets.
void add_joint_blocks(BlockMap& out, int bi, int bj, const Dense24& H_local)
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

    const bool ascending = bi < bj;
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
    add_body(bi, Dense12(H_local.block<12, 12>(0, 0)));
    add_body(bj, Dense12(H_local.block<12, 12>(12, 12)));

    const Dense12 off = ascending ? Dense12(H_local.block<12, 12>(0, 12))
                                  : Dense12(H_local.block<12, 12>(12, 0));
    const int     lo  = std::min(bi, bj);
    const int     hi  = std::max(bi, bj);
    for(int a = 0; a < 4; ++a)
        for(int b = 0; b < 4; ++b)
            add(lo * 4 + a, hi * 4 + b, off.block<3, 3>(a * 3, b * 3));
}
}  // namespace


// ===========================================================================
//  thin wrappers over the ported generic routines (host, S = double)
// ===========================================================================
namespace
{
gg::abd_prismatic_joint_detail::PrismaticJointRestT<double> pj_rest(const PrismRest& r)
{
    return gg::abd_prismatic_joint_detail::PrismaticJointRestT<double>{
        .rest_c = r.rest_c(), .rest_t = r.rest_t(), .rest_n = r.rest_n(), .rest_b = r.rest_b()};
}

gg::abd_driving_prismatic_joint_detail::DrivingPrismaticJointRestT<double> dpj_rest(
    const PrismRest& r)
{
    return gg::abd_driving_prismatic_joint_detail::DrivingPrismaticJointRestT<double>{
        .rest_c = r.rest_c(), .rest_t = r.rest_t()};
}

double pj_E(double kappa, const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    return gg::abd_prismatic_joint_detail::body_energy<double>(
        kappa, pj_rest(r), gg::to_luisa(qi), gg::to_luisa(qj));
}

Dense24v pj_G(double kappa, const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen24(gg::abd_prismatic_joint_detail::body_gradient<double>(
        kappa, pj_rest(r), gg::to_luisa(qi), gg::to_luisa(qj)));
}

Dense24 pj_H(double kappa, const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen24(gg::abd_prismatic_joint_detail::body_hessian<double>(
        kappa, pj_rest(r), gg::to_luisa(qi), gg::to_luisa(qj)));
}

/// `F01` alone (the ported routine, to pin the measured sign convention).
Dense9v pj_F01(const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen9(gg::abd_prismatic_joint_detail::f01<double>(
        pj_rest(r), gg::to_luisa(qi), gg::to_luisa(qj)));
}

double dpj_E(double kappa, const PrismRest& r, const Vector12& qi, const Vector12& qj, double d)
{
    return gg::abd_driving_prismatic_joint_detail::body_energy<double>(
        kappa, dpj_rest(r), gg::to_luisa(qi), gg::to_luisa(qj), d);
}

Dense24v dpj_G(double kappa, const PrismRest& r, const Vector12& qi, const Vector12& qj, double d)
{
    return to_eigen24(gg::abd_driving_prismatic_joint_detail::body_gradient<double>(
        kappa, dpj_rest(r), gg::to_luisa(qi), gg::to_luisa(qj), d));
}

Dense24 dpj_H(double kappa, const PrismRest& r, const Vector12& qi, const Vector12& qj, double d)
{
    return to_eigen24(gg::abd_driving_prismatic_joint_detail::body_hessian<double>(
        kappa, dpj_rest(r), gg::to_luisa(qi), gg::to_luisa(qj), d));
}

Dense9v dpj_F01q(const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen9(gg::abd_driving_prismatic_joint_detail::f01_q<double>(
        dpj_rest(r), gg::to_luisa(qi), gg::to_luisa(qj)));
}

/// `compute_absolute_distance` (the reference's first hand-written helper).
double abs_distance(const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    return gg::abd_driving_prismatic_joint_detail::compute_absolute_distance<double>(
        r.rest_c(), r.rest_t(), gg::to_luisa(qi), gg::to_luisa(qj));
}

Dense24v abs_distance_deriv(const PrismRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen24(
        gg::abd_driving_prismatic_joint_detail::compute_absolute_distance_derivative<double>(
            r.rest_c(), r.rest_t(), gg::to_luisa(qi), gg::to_luisa(qj)));
}

/// `compute_absolute_distance_hessian` (an *accumulate*: `out_H += ...`).
Dense24 abs_distance_hess_add(const Dense24&  seed,
                              double          dE_dx,
                              const PrismRest& r,
                              const Vector12& qi,
                              const Vector12& qj)
{
    lps::LMat<double, 24, 24> H;
    for(int c = 0; c < 24; ++c)
        for(int i = 0; i < 24; ++i)
            H(i, c) = seed(i, c);
    gg::abd_driving_prismatic_joint_detail::compute_absolute_distance_hessian<double>(
        H, dE_dx, r.rest_c(), r.rest_t(), gg::to_luisa(qi), gg::to_luisa(qj));
    return to_eigen24(H);
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

/// The ported `eval_penalty_energy` / `eval_penalty_derivatives` (the two
/// partial-computation entry points, which zero the outputs they do not own).
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

/// A `Vector6` pairing of the joint's `C_bar`/`t_bar` for the three helpers above.
constexpr double kSceneDt = 0.01;
}  // namespace

// ===========================================================================
//  1) closed forms: the two element laws and their three derivatives, the
//     (non-diagonal) element Hessian and the reference's clamp
// ===========================================================================
TEST_CASE("abd_joint_b_closed_forms", "[luisa][abd_joint_b]")
{
    test_runtime();
    CaseScope scope{"abd_joint_b_closed_forms"};
    auto&     c = scope.counter;

    const double kappa = 2.5;

    // ---- (A) the element laws at a strained state ---------------------------
    {
        PrismRest r;
        r.ci = Vector3{0.11, -0.07, 0.05};
        r.cj = Vector3{-0.04, 0.09, 0.13};
        r.set_unit(Eigen::AngleAxisd(0.4, Vector3{1, 0, 0}.normalized()).toRotationMatrix(),
                   Eigen::AngleAxisd(-0.7, Vector3{0, 0, 1}.normalized()).toRotationMatrix());
        const Vector12 qi = q_of(1.1 * Matrix3x3::Identity(), Vector3{0.3, -0.2, 0.4});
        const Vector12 qj = q_of(0.95 * Matrix3x3::Identity(), Vector3{-0.1, 0.5, -0.3});

        const auto loc = prism_local(kappa, r, qi, qj);

        // the *sign convention* of `F01` is measured, not assumed: it is `[-Ft ; ...]`
        const Dense9v F01_port = pj_F01(r, qi, qj);
        T63B2_CHECK(c, dev(F01_port, F01_of(r, qi, qj)) <= 1e-14);
        T63B2_CHECK(c, dev(F01_port, loc.F01) <= 1e-15);
        {
            // ... and a `+Ft` spelling is off by exactly 2x on the first three rows
            Dense9v plus = loc.F01;
            plus.segment<3>(0) = -plus.segment<3>(0);
            T63B2_CHECK(c, max_abs(plus - loc.F01) > 1.0);
        }

        T63B2_CHECK(c, rel_err(pj_E(kappa, r, qi, qj), loc.E) <= 1e-14);
        T63B2_CHECK(c, dev(pj_G(kappa, r, qi, qj), loc.G) <= 1e-13);
        T63B2_CHECK(c, dev(pj_H(kappa, r, qi, qj), loc.H) <= 1e-11);
        std::printf("[t63b2] closed: strained pair E %.10e (E01 %.6e E23 %.6e), |G| %.6e, "
                    "|H| %.6e, |F01| %.6e\n",
                    loc.E, loc.E01, loc.E23, max_abs(loc.G), max_abs(loc.H), max_abs(loc.F01));

        // the two element pieces against the generated bodies' own functions
        T63B2_CHECK(c, std::abs(loc.E01 - E01_of(loc.F01, kappa)) == 0.0);
        T63B2_CHECK(c, loc.E > 0.0);
        T63B2_CHECK(c, std::abs(loc.E - (loc.E01 + loc.E23)) == 0.0);

        // the *analytic* element derivatives vs the FD of the analytic `E01` in F01
        // space (an independent derivative of an independent law)
        {
            const auto g_fd = [&](const Dense9v& F)
            {
                Dense9v g;
                for(int k = 0; k < 9; ++k)
                {
                    Dense9v fp = F, fm = F;
                    fp(k) += 1e-6;
                    fm(k) -= 1e-6;
                    g(k) = (E01_of(fp, kappa) - E01_of(fm, kappa)) / 2e-6;
                }
                return g;
            };
              // the four-point second difference (the form validated against sympy
              // for this element law): the gradient-difference variant is dominated by
              // the nested-difference round-off for a quartic law
              const auto H_fd = [&](const Dense9v& F)
              {
                  const double h = 1e-3;
                  Dense9        H;
                  for(int i = 0; i < 9; ++i)
                  {
                      for(int j = 0; j < 9; ++j)
                      {
                          Dense9v pp = F, pm = F, mp = F, mm = F;
                          pp(i) += h; pp(j) += h;
                          pm(i) += h; pm(j) -= h;
                          mp(i) -= h; mp(j) += h;
                          mm(i) -= h; mm(j) -= h;
                          H(i, j) = (E01_of(pp, kappa) - E01_of(pm, kappa)
                                     - E01_of(mp, kappa) + E01_of(mm, kappa))
                                    / (4 * h * h);
                      }
                  }
                  return 0.5 * (H + H.transpose());
              };
              T63B2_CHECK(c, dev(dE01dF01_of(loc.F01, kappa), g_fd(loc.F01)) <= 1e-8);
              {
                  const Dense9 H_law = d2E01dF01_of(loc.F01, kappa);
                  const Dense9 H_num = H_fd(loc.F01);
                  std::printf("[t63b2] closed: |H_law| %.6e |H_fd| %.6e (H_law(0,0) %.6e, "
                              "H_fd(0,0) %.6e, H_fd(3,3) %.6e)\n",
                              max_abs(H_law), max_abs(H_num), H_law(0, 0), H_num(0, 0),
                              H_num(3, 3));
                  T63B2_CHECK(c, dev(H_law, H_num) <= 1e-5);
              }
        }

        // E23's three pieces against FD
        {
            const Dense24v q = pack_pair(qi, qj);
            const Dense24v g_fd =
                fd_gradient([&](const Dense24v& qq)
                            { return E23_local(r, kappa, Vector12(qq.segment<12>(0)),
                                               Vector12(qq.segment<12>(12))).E; },
                            q, kFdH);
            T63B2_CHECK(c, dev(loc.G23, g_fd) <= 1e-8);
            const Dense24 h_fd =
                fd_hessian([&](const Dense24v& qq)
                           { return E23_local(r, kappa, Vector12(qq.segment<12>(0)),
                                              Vector12(qq.segment<12>(12)))
                                        .G; },
                           q, 1e-4);
            T63B2_CHECK(c, dev(loc.H23, h_fd) <= 1e-5);
        }

        // the *element* Hessian's spectrum: for this law it is indefinite, which is
        // the T6.3b2 statement the T6.3b1 clamp pin was missing (rule 19)
        std::printf("[t63b2] closed: min eig(H01) %.6e (|H01| %.6e), min eig(H_shipped) "
                    "%.6e, min eig(H_raw) %.6e\n",
                    min_eigenvalue(loc.H01), max_abs(loc.H01), min_eigenvalue(loc.H),
                    min_eigenvalue(loc.H_raw));
        // the exact clamp is *PSD* (that is what `make_spd` is for) and it moved the
        // matrix (this state's element Hessian is indefinite)
        T63B2_CHECK(c, min_eigenvalue(loc.H01_spd) >= -1e-12 * max_abs(loc.H01_spd));
        T63B2_CHECK(c, max_abs(loc.H01_spd - loc.H01) > 0.0);
        // ... and the clamp is NOT a no-op for kappa > 0: the unclamped chain differs
        const double clamped_dev = dev(loc.H_raw, loc.H);
        std::printf("[t63b2] closed: ||H_raw - H_shipped|| %.3e (kappa > 0: the element "
                    "Hessian is indefinite, so the clamp is a real projection)\n",
                    clamped_dev);
        T63B2_CHECK(c, clamped_dev > 1e-6);
        T63B2_CHECK(c, min_eigenvalue(loc.H) >= -1e-10 * max_abs(loc.H));
    }

    // ---- (A2) the *sweep-count* pin (rule 19): the element Hessians this batch
    //      clamps are non-diagonal, so a 0-sweep `make_spd` is a different matrix —
    //      the axis T6.3b1's joints could not pin --------------------------------
    {
        PrismRest r;
        r.ci = Vector3{0.11, -0.07, 0.05};
        r.cj = Vector3{-0.04, 0.09, 0.13};
        r.set_unit(Eigen::AngleAxisd(0.4, Vector3{1, 0, 0}.normalized()).toRotationMatrix(),
                   Eigen::AngleAxisd(-0.7, Vector3{0, 0, 1}.normalized()).toRotationMatrix());
        const Vector12 qi = q_of(1.1 * Matrix3x3::Identity(), Vector3{0.3, -0.2, 0.4});
        const Vector12 qj = q_of(0.95 * Matrix3x3::Identity(), Vector3{-0.1, 0.5, -0.3});

        // the joint's element 9x9, the driving's, and the limit's `dE_dx * ddxdF01`
        // (all three are what the reference hands to `make_spd`)
        lps::LMat<double, 9, 9> H9_j;
        gg::sym::affine_body_prismatic_joint::ddE01ddF01<double>(H9_j, kappa,
                                                                lv_of<9>(F01_of(r, qi, qj)));
        const double                  d = 0.17;
        lps::LMat<double, 9, 9>       H9_d;
        gg::sym::affine_body_driving_prismatic_joint::ddE01ddF01<double>(
            H9_d, kappa, lv_of<9>(F01q_of(r, qi, qj)), d);
        const lps::LMat<double, 9, 9> H9_lim = lm_of<9>(2.7 * ddxdF01_of());

        auto sweep_diff = [](const lps::LMat<double, 9, 9>& H, int a, int b)
        {
            auto x = H, y = H;
            lps::make_spd_inplace(x, a);
            lps::make_spd_inplace(y, b);
            double w = 0.0;
            for(int cc = 0; cc < 9; ++cc)
                for(int rr = 0; rr < 9; ++rr)
                    w = std::max(w, std::abs(x(rr, cc) - y(rr, cc)));
            return w;
        };
        const double joint_0_8 =
            sweep_diff(H9_j, 0, gg::abd_prismatic_joint_detail::kMakeSpdSweeps);
        const double joint_8_20 =
            sweep_diff(H9_j, gg::abd_prismatic_joint_detail::kMakeSpdSweeps, 20);
        const double drv_0_8 =
            sweep_diff(H9_d, 0, gg::abd_driving_prismatic_joint_detail::kMakeSpdSweeps);
        const double drv_8_20 =
            sweep_diff(H9_d, gg::abd_driving_prismatic_joint_detail::kMakeSpdSweeps, 20);
        const double lim_0_8 =
            sweep_diff(H9_lim, 0, gg::abd_driving_prismatic_joint_detail::kMakeSpdSweeps);

        std::printf("[t63b2] closed: sweep pin (rule 19) -- 0-vs-%d sweeps: joint %.3e, "
                    "driving %.3e, limit %.3e; %d-vs-20: joint %.3e, driving %.3e\n",
                    gg::abd_prismatic_joint_detail::kMakeSpdSweeps, joint_0_8, drv_0_8,
                    lim_0_8, gg::abd_prismatic_joint_detail::kMakeSpdSweeps, joint_8_20,
                    drv_8_20);
        T63B2_CHECK(c, joint_0_8 > 1e-6);  // the joint's clamp *is* sweep-sensitive
        T63B2_CHECK(c, drv_0_8 > 1e-6);
        T63B2_CHECK(c, lim_0_8 > 1e-6);
        T63B2_CHECK(c, joint_8_20 == 0.0);  // 8 sweeps already on the plateau
        T63B2_CHECK(c, drv_8_20 == 0.0);
    }

    // ---- (B) over random admissible states: the shipped vs the exact clamp, and
    //      how often the element Hessian is indefinite --------------------------
    {
        constexpr int kStates = 40;
        double        worst_clamp = 0.0, worst_E = 0.0, worst_G = 0.0;
        double        worst_H01 = 0.0, worst_h33 = 0.0;
        int           indefinite = 0;
        for(int s = 0; s < kStates; ++s)
        {
            PrismRest r;
            r.ci = rand_vec(-0.3, 0.3);
            r.cj = rand_vec(-0.3, 0.3);
            const Matrix3x3 Ai = random_admissible_A();
            const Matrix3x3 Aj = random_admissible_A();
            r.set_unit(Ai, Aj, rand_vec(-1.0, 1.0));
            const Vector12 qi    = q_of(Ai, rand_vec(-0.5, 0.5));
            const Vector12 qj    = q_of(Aj, rand_vec(-0.5, 0.5));
            const double   kp    = rand_uniform(0.5, 2.0);

            const auto loc = prism_local(kp, r, qi, qj);
            worst_E        = std::max(worst_E, rel_err(pj_E(kp, r, qi, qj), loc.E));
            worst_G        = std::max(worst_G, dev(pj_G(kp, r, qi, qj), loc.G));
            worst_clamp    = std::max(worst_clamp, dev(pj_H(kp, r, qi, qj), loc.H));
            // (the `indefinite` count is taken from the *ported* element Hessian
            // further down, not from this analytic chain)

            // the ported *element* Hessian (unclamped) vs the analytic one: this is
            // what pins the generated `ddE01ddF01` matrix entry by entry
            {
                lps::LMat<double, 9, 9> H01_port;
                gg::sym::affine_body_prismatic_joint::ddE01ddF01<double>(
                    H01_port, kp, lv_of<9>(loc.F01));
                worst_H01 = std::max(worst_H01, dev(to_eigen9(H01_port), loc.H01));
                // the *shipped* element Hessian is indefinite in most states, which is
                // exactly why the reference's clamp is a real projection here (rule 19)
                const Dense9 H01_shipped = to_eigen9(H01_port);
                if(min_eigenvalue(H01_shipped) < -1e-12 * max_abs(H01_shipped))
                    ++indefinite;

                // ... and the *clamped* 9x9 the layer's 8 sweeps produce vs the layer's
                // 20 sweeps (the sweep-count plateau, measured on a NON-diagonal 9x9)
                auto a = H01_port, b = H01_port;
                lps::make_spd_inplace(a, gg::abd_prismatic_joint_detail::kMakeSpdSweeps);
                lps::make_spd_inplace(b, 20);
                double w = 0.0;
                for(int cc = 0; cc < 9; ++cc)
                    for(int rr = 0; rr < 9; ++rr)
                        w = std::max(w, std::abs(a(rr, cc) - b(rr, cc)));
                worst_h33 = std::max(worst_h33, w);
            }
        }
        std::printf("[t63b2] closed (40 states): E %.3e G %.3e H %.3e (vs the exact clamp), "
                    "ddE01ddF01 %.3e, 8-vs-20 sweeps %.3e, indefinite %d/%d\n",
                    worst_E, worst_G, worst_clamp, worst_H01, worst_h33, indefinite, kStates);
        T63B2_CHECK(c, worst_E <= 1e-13);
        T63B2_CHECK(c, worst_G <= 1e-12);
        T63B2_CHECK(c, worst_clamp <= 1e-8);  // the 8-sweep Jacobi vs the exact clamp
        T63B2_CHECK(c, worst_H01 <= 1e-12);
        // this is the rule-19 statement for T6.3b2: the element Hessian is NOT diagonal
        T63B2_CHECK(c, indefinite > 0);
        T63B2_CHECK(c, worst_h33 <= 1e-10);
    }

    // ---- (C) scaling laws and translation invariance -------------------------
    {
        PrismRest r;
        r.ci = Vector3{0.2, 0.1, -0.1};
        r.cj = Vector3{-0.1, 0.3, 0.2};
        r.set_unit(Matrix3x3::Identity(), 1.2 * Matrix3x3::Identity(), Vector3{0.3, 0.9, -0.2});
        const Vector12 qi = q_of(1.1 * Matrix3x3::Identity(), Vector3{0.3, 0.2, -0.1});
        const Vector12 qj = q_of(0.9 * Matrix3x3::Identity(), Vector3{-0.2, 0.4, 0.5});

        const double E1 = pj_E(kappa, r, qi, qj);
        T63B2_CHECK(c, rel_err(pj_E(2.0 * kappa, r, qi, qj), 2.0 * E1) <= 1e-15);
        T63B2_CHECK(c, rel_err(pj_E(-kappa, r, qi, qj), -E1) <= 1e-15);
        // translation invariance: the laws read `A` and the *difference* of the
        // translations only
        Vector12 qis = qi, qjs = qj;
        qis.segment<3>(0) += Vector3{1.5, -2.0, 0.75};
        qjs.segment<3>(0) += Vector3{1.5, -2.0, 0.75};
        T63B2_CHECK(c, pj_E(kappa, r, qis, qjs) == E1);
        T63B2_CHECK(c, dev(pj_G(kappa, r, qis, qjs), pj_G(kappa, r, qi, qj)) <= 1e-15);
        T63B2_CHECK(c, dev(pj_H(kappa, r, qis, qjs), pj_H(kappa, r, qi, qj)) <= 1e-13);
        // a common *rotation* of both bodies rotates `a`, `b` and `c` together, so E is
        // invariant and H keeps its spectrum
        const Matrix3x3 R =
            Eigen::AngleAxisd(0.9, Vector3{1, 2, -1}.normalized()).toRotationMatrix();
        const Vector12 qir = q_of(R * A_of(qi), R * t_of(qi));
        const Vector12 qjr = q_of(R * A_of(qj), R * t_of(qj));
        T63B2_CHECK(c, rel_err(pj_E(kappa, r, qir, qjr), E1) <= 1e-12);
        T63B2_CHECK(c, std::abs(E01_of(F01_of(r, qi, qj), kappa)
                                + E23_local(r, kappa, qi, qj).E
                                - pj_E(kappa, r, qi, qj))
                           <= 1e-14);
    }

    // ---- (D) the driving joint's laws ---------------------------------------
    {
        PrismRest r;
        r.ci = Vector3{0.11, -0.07, 0.05};
        r.cj = Vector3{-0.04, 0.09, 0.13};
        r.set_unit(Eigen::AngleAxisd(0.4, Vector3{1, 0, 0}.normalized()).toRotationMatrix(),
                   Eigen::AngleAxisd(-0.7, Vector3{0, 0, 1}.normalized()).toRotationMatrix());
        const Vector12 qi = q_of(1.1 * Matrix3x3::Identity(), Vector3{0.3, -0.2, 0.4});
        const Vector12 qj = q_of(0.95 * Matrix3x3::Identity(), Vector3{-0.1, 0.5, -0.3});
        const double   d  = 0.17;

        const Dense9v F01q = dpj_F01q(r, qi, qj);
        T63B2_CHECK(c, dev(F01q, F01q_of(r, qi, qj)) <= 1e-15);
        T63B2_CHECK(c, std::abs(abs_distance(r, qi, qj) - distance_of(F01q)) <= 1e-15);
        // `compute_absolute_distance_derivative` is `J01^T dxdF01` (a *24*-vector), so it
        // is compared against the analytic chain, not against `dxdF01` itself
        T63B2_CHECK(c, dev(abs_distance_deriv(r, qi, qj),
                            J01q_of(r).transpose() * dxdF01_of(F01q))
                           <= 1e-13);

        T63B2_CHECK(c, rel_err(dpj_E(kappa, r, qi, qj, d), driving_E01_of(F01q, kappa, d)) <= 1e-14);
        T63B2_CHECK(c, dev(dpj_G(kappa, r, qi, qj, d),
                           J01q_of(r).transpose() * driving_dE01dF01_of(F01q, kappa, d))
                           <= 1e-13);

        // the *element* 9x9 (raw + clamped) and its 3x3 follow-ups, against the
        // analytic forms; the clamp there is observable too
        {
            lps::LMat<double, 9, 9> H9_port;
            gg::sym::affine_body_driving_prismatic_joint::ddE01ddF01<double>(
                H9_port, kappa, lv_of<9>(F01q), d);
            const Dense9 H9_law = driving_d2E01dF01_of(F01q, kappa, d);
            T63B2_CHECK(c, dev(to_eigen9(H9_port), H9_law) <= 1e-12);

            lps::LMat<double, 9, 9> ddxd;
            gg::sym::affine_body_driving_prismatic_joint::ddxdF01<double>(ddxd,
                                                                         lv_of<9>(F01q));
            T63B2_CHECK(c, dev(to_eigen9(ddxd), ddxdF01_of()) <= 1e-15);

            // the limit's k2 chain: `d2E_dx2 (dx dx^T) [+ J^T make_spd(dE_dx ddx) J]`
            const double dE_dx   = 2.7;
            const double d2E_dx2 = -1.3;
            const Dense24 H = abs_distance_hess_add(
                d2E_dx2 * (abs_distance_deriv(r, qi, qj)
                           * abs_distance_deriv(r, qi, qj).transpose()),
                dE_dx,
                r,
                qi,
                qj);
            const auto  J01 = J01q_of(r);
            const Dense24 expected =
                d2E_dx2
                    * (J01.transpose() * dxdF01_of(F01q)
                       * (J01.transpose() * dxdF01_of(F01q)).transpose())
                + J01.transpose() * exact_spd(dE_dx * ddxdF01_of()) * J01;
            T63B2_CHECK(c, dev(H, expected) <= 1e-9);

            // the ported `hessian` helper is the same statement
            {
                const Dense24 H2 = to_eigen24(gg::abd_prismatic_joint_limit_detail::hessian<double>(
                    d2E_dx2,
                    dE_dx,
                    lv_of<24>(J01.transpose() * dxdF01_of(F01q)),
                    r.rest_c(),
                    r.rest_t(),
                    gg::to_luisa(qi),
                    gg::to_luisa(qj)));
                T63B2_CHECK(c, dev(H2, expected) <= 1e-9);
            }

            // ... and the `dE_dx == 0` case: the reference skips the second statement,
            // the port computes it and adds it — which must be *exactly* zero
            const Dense24 H_zero = abs_distance_hess_add(Dense24::Zero(), 0.0, r, qi, qj);
            T63B2_CHECK(c, max_abs(H_zero) == 0.0);
            std::printf("[t63b2] closed: the reference's `if(dE_dx != 0)` guard is "
                        "value-free: |H(0)| %.3e\n",
                        max_abs(H_zero));
        }
    }

    // ---- (E) the joint-limit penalty: every branch of the reference's body ----
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

            T63B2_CHECK(c, std::abs(penalty_port_energy(cs.x, cs.lower, cs.upper, cs.strength)
                                     - ref.E)
                                   <= 1e-15);
            const auto derivs = penalty_port_derivs(cs.x, cs.lower, cs.upper, cs.strength);
            T63B2_CHECK(c, std::abs(derivs.first - ref.dE) <= 1e-15);
            T63B2_CHECK(c, std::abs(derivs.second - ref.d2E) <= 1e-15);
            if(std::string_view{cs.name} == "inside")
            {
                T63B2_CHECK(c, port.E == 0.0);
                T63B2_CHECK(c, port.dE == 0.0);
                T63B2_CHECK(c, port.d2E == 0.0);
            }
            if(std::string_view{cs.name} == "at upper" || std::string_view{cs.name} == "at lower")
            {
                // the *inclusive* interval: at the bound the penalty is exactly zero
                T63B2_CHECK(c, (port.E == 0.0 && port.dE == 0.0 && port.d2E == 0.0));
            }
        }
        std::printf("[t63b2] closed: penalty branches (%zu cases), worst |dE| %.3e "
                    "|dE_dx| %.3e |d2E_dx2| %.3e vs the branchy oracle\n",
                    sizeof(cases) / sizeof(cases[0]), worst_E, worst_dE, worst_d2E);
        T63B2_CHECK(c, worst_E <= 1e-15);
        T63B2_CHECK(c, worst_dE <= 1e-15);
        T63B2_CHECK(c, worst_d2E <= 1e-15);

        // the two *partial* entry points write only their own outputs (the contract
        // the limit kernels rely on), and they agree with the full one
        {
            const auto refd = penalty_ref(0.35, -0.05, 0.20, 3.0);
            double     dE = 7.0, d2E = 7.0;
            gg::joint_limit::eval_penalty_derivatives<double>(0.35, -0.05, 0.20, 3.0, dE, d2E);
            T63B2_CHECK(c, (dE != 7.0 && d2E != 7.0));
            T63B2_CHECK(c, std::abs(dE - refd.dE) <= 1e-15);
            T63B2_CHECK(c, std::abs(d2E - refd.d2E) <= 1e-15);
            T63B2_CHECK(c,
                        std::abs(penalty_port_energy(0.35, -0.05, 0.20, 3.0) - refd.E) <= 1e-15);
        }
    }

    // ---- (F) the limit's own law at a state and the external-force pair -------
    {
        PrismRest r;
        r.ci = Vector3{0.05, 0.02, -0.01};
        r.cj = Vector3{-0.03, 0.04, 0.06};
        r.set_unit(Eigen::AngleAxisd(0.2, Vector3{0, 1, 0}.normalized()).toRotationMatrix(),
                   Eigen::AngleAxisd(0.9, Vector3{1, 0, 0}.normalized()).toRotationMatrix(),
                   Vector3{0.2, -0.7, 0.4});
        const Vector12 qi = q_of(1.05 * Matrix3x3::Identity(), Vector3{0.1, 0.2, -0.3});
        const Vector12 qj = q_of(0.98 * Matrix3x3::Identity(), Vector3{-0.2, 0.1, 0.4});

        // the bounds are chosen so that the state's distance is *outside* them (a
        // limit whose interval contains `x` contributes nothing at all)
        // the bounds are chosen so that the state's distance is *outside* them (a limit
        const auto lim = limit_local(0.0, 0.05, 0.40, 40.0, r, qi, qj);
        std::printf("[t63b2] closed: limit x %.10e E %.6e dE_dx %.6e d2E_dx2 %.6e "
                    "|G| %.3e |H| %.3e\n",
                    lim.x, lim.E, lim.dE_dx, lim.d2E_dx2, max_abs(lim.G), max_abs(lim.H));
        T63B2_CHECK(c, std::abs(lim.x - abs_distance(r, qi, qj)) <= 1e-15);
        T63B2_CHECK(c, lim.E > 0.0);
        T63B2_CHECK(c, std::abs(lim.dE_dx) > 1e-9);
        T63B2_CHECK(c, dev(lim.dx_dq, abs_distance_deriv(r, qi, qj)) <= 1e-13);
        T63B2_CHECK(c, dev(lim.G, lim.dE_dx * lim.dx_dq) <= 1e-15);
        T63B2_CHECK(c, dev(lim.G,
                           to_eigen24(gg::abd_prismatic_joint_limit_detail::gradient<double>(
                               lim.dE_dx, lv_of<24>(lim.dx_dq))))
                           <= 1e-15);

        const auto ef = ext_force_local(12.5, r, qi, qj);
        {
            // the ported arithmetic, with the *same* `ABDJacobiT` inputs
            lps::LVec<double, 3> t_i, t_j;
            gg::abd_prismatic_joint_external_force_detail::symmetrized_axes<double>(
                t_i,
                t_j,
                gg::ABDJacobiT<double>{.x_bar = lv3(r.ti)},
                gg::ABDJacobiT<double>{.x_bar = lv3(r.tj)},
                gg::to_luisa(qi),
                gg::to_luisa(qj));
            T63B2_CHECK(c, dev(to_eigen3(t_i), ef.t_i) <= 1e-15);
            T63B2_CHECK(c, dev(to_eigen3(t_j), ef.t_j) <= 1e-15);
            // the symmetrisation: `t_i == -t_j` and `t_i + t_j == 0`
            T63B2_CHECK(c, dev(to_eigen3(t_i) + to_eigen3(t_j), Dense3v(Dense3v::Zero())) == 0.0);

            lps::LVec<double, 12> F_i, F_j;
            gg::abd_prismatic_joint_external_force_detail::force_pair<double>(
                12.5, t_i, t_j, F_i, F_j);
            T63B2_CHECK(c, dev(eig_of<12>(F_i), ef.F_i) <= 1e-15);
            T63B2_CHECK(c, dev(eig_of<12>(F_j), ef.F_j) <= 1e-15);
            // the force is *pure translation* in the 12-dof sense: only the first three
            // dofs are non-zero
            double rest_i = 0.0;
            for(int k = 3; k < 12; ++k)
                rest_i = std::max(rest_i, std::abs(ef.F_i(k)));
            T63B2_CHECK(c, rest_i == 0.0);
            T63B2_CHECK(c, max_abs(ef.F_i) > 0.0);
        }
    }

    std::printf("[t63b2] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}


// ===========================================================================
//  2) central finite differences over the 24 pair dofs: the gradient is the
//     derivative of E, the Hessian the derivative of the gradient, the h-sweep, the
//     symmetry, the rule-34 basis question, and what the clamp does to `(g, H)`
// ===========================================================================
TEST_CASE("abd_joint_b_finite_difference", "[luisa][abd_joint_b]")
{
    test_runtime();
    CaseScope scope{"abd_joint_b_finite_difference"};
    auto&     c = scope.counter;

    constexpr int kStates = 40;

    double worst_g = 0.0, worst_h6 = 0.0, worst_h4 = 0.0, worst_h3 = 0.0, worst_h2 = 0.0;
    double worst_sym = 0.0, worst_raw = 0.0, worst_shipped = 0.0, worst_d_g = 0.0;
    double worst_driving_g = 0.0, worst_driving_h = 0.0, worst_driving_raw = 0.0;
    double worst_limit_g = 0.0, worst_limit_h = 0.0;
    double worst_J = 0.0;
    int    indefinite = 0;

    for(int s = 0; s < kStates; ++s)
    {
        PrismRest r;
        r.ci = rand_vec(-0.3, 0.3);
        r.cj = rand_vec(-0.3, 0.3);
        const Matrix3x3 Ai = random_admissible_A();
        const Matrix3x3 Aj = random_admissible_A();
        r.set_unit(Ai, Aj, rand_vec(-1.0, 1.0));
        const Vector12 qi     = q_of(Ai, rand_vec(-0.5, 0.5));
        const Vector12 qj     = q_of(Aj, rand_vec(-0.5, 0.5));
        const double   kappa  = rand_uniform(0.5, 2.0);
        const double   d_tidl = rand_uniform(-0.4, 0.4);
        const Dense24v q      = pack_pair(qi, qj);

        // ---- the joint --------------------------------------------------------
        {
            const Dense24v G = pj_G(kappa, r, qi, qj);
            const Dense24v Gfd =
                fd_gradient([&](const Dense24v& qq)
                            { return pj_E(kappa,
                                          r,
                                          Vector12(qq.segment<12>(0)),
                                          Vector12(qq.segment<12>(12))); },
                            q,
                            kFdH);
            worst_g = std::max(worst_g, dev(G, Gfd));

            auto grad = [&](const Dense24v& qq)
            { return pj_G(kappa, r, Vector12(qq.segment<12>(0)), Vector12(qq.segment<12>(12))); };

            const Dense24 H    = pj_H(kappa, r, qi, qj);
            const Dense24 Hfd6 = fd_hessian(grad, q, 1e-6);
            const Dense24 Hfd4 = fd_hessian(grad, q, 1e-4);
            const Dense24 Hfd3 = fd_hessian(grad, q, 1e-3);
            const Dense24 Hfd2 = fd_hessian(grad, q, 1e-2);
            worst_h6 = std::max(worst_h6, dev(H, Hfd6));
            worst_h4 = std::max(worst_h4, dev(H, Hfd4));
            worst_h3 = std::max(worst_h3, dev(H, Hfd3));
            worst_h2 = std::max(worst_h2, dev(H, Hfd2));
            worst_sym = std::max(worst_sym, max_abs(H - Dense24(H.transpose())) / max_abs(H));
            // NOTE: the shipped `H` is the *clamped* element Hessian's fan-out while
            // the FD differentiates the *unclamped* gradient, so `dev(H, Hfd)` is
            // O(clamp) and does not fall with h (the clamp's eigenvalue selection is not
            // smooth).  The pins below are therefore (a) the *unclamped* analytic chain
            // against the FD (`worst_raw`) and (b) that the shipped one is measurably
            // different (`worst_d_g`).

            // what the clamp does to the *pair* `(g, H)`: the finite difference of the
            // *shipped* gradient is the *unclamped* chain (the reference clamps only
            // the Hessian), so the shipped H is NOT d(g)/dq — and the deviation is
            // recorded, not assumed to be round-off (§7.25 rule 6, re-pinned here for
            // the opposite reason: the element Hessian is indefinite even for kappa > 0)
            const auto loc = prism_local(kappa, r, qi, qj);
            worst_raw     = std::max(worst_raw, dev(loc.H_raw, Hfd6));
            worst_shipped = std::max(worst_shipped, dev(H, Hfd6));
            worst_d_g     = std::max(worst_d_g, dev(loc.H_raw, H));
            if(min_eigenvalue(loc.H01) < -1e-12 * max_abs(loc.H01))
                ++indefinite;
        }

        // ---- the driving joint ------------------------------------------------
        {
            const Dense24v G = dpj_G(kappa, r, qi, qj, d_tidl);
            const Dense24v Gfd =
                fd_gradient([&](const Dense24v& qq)
                            { return dpj_E(kappa,
                                           r,
                                           Vector12(qq.segment<12>(0)),
                                           Vector12(qq.segment<12>(12)),
                                           d_tidl); },
                            q,
                            kFdH);
            worst_driving_g = std::max(worst_driving_g, dev(G, Gfd));

            auto grad = [&](const Dense24v& qq)
            {
                return dpj_G(kappa, r, Vector12(qq.segment<12>(0)),
                             Vector12(qq.segment<12>(12)), d_tidl);
            };
            const Dense24 H    = dpj_H(kappa, r, qi, qj, d_tidl);
            const Dense24 Hfd6 = fd_hessian(grad, q, 1e-6);
            worst_driving_h    = std::max(worst_driving_h, dev(H, Hfd6));

            // the driving `(g, H)` pair has the same clamp story: the *analytic* raw
            // chain (no clamp) is what the FD of the shipped gradient gives
            const Dense9v F01q = dpj_F01q(r, qi, qj);
            const auto    J01  = J01q_of(r);  // the driving's own Jacobian
            const Dense24 raw =
                J01.transpose() * driving_d2E01dF01_of(F01q, kappa, d_tidl) * J01;
            worst_driving_raw = std::max(worst_driving_raw, dev(raw, Hfd6));
        }

        // ---- the joint limit --------------------------------------------------
        {
            const double lo  = rand_uniform(-0.3, 0.0);
            const double hi  = lo + rand_uniform(0.05, 0.8);
            const double str = rand_uniform(1.0, 50.0);
            // pick the bounds so that the *initial* distance is outside the interval
            // (so `dE_dx != 0` and the Hessian has both terms)
            const double x0 = distance_of(F01q_of(r, qi, qj));
            const double off = std::min(lo - x0, hi - x0) - 0.11;

            const auto lim = limit_local(off, lo, hi, str, r, qi, qj);

            const Dense24v Gfd =
                fd_gradient([&](const Dense24v& qq)
                            { return limit_local(off, lo, hi, str, r,
                                                 Vector12(qq.segment<12>(0)),
                                                 Vector12(qq.segment<12>(12)))
                                         .E; },
                            q,
                            kFdH);
            worst_limit_g = std::max(worst_limit_g, dev(lim.G, Gfd));

            if(std::abs(lim.dE_dx) > 1e-9)
            {
                auto grad = [&](const Dense24v& qq)
                {
                    return limit_local(off, lo, hi, str, r, Vector12(qq.segment<12>(0)),
                                       Vector12(qq.segment<12>(12)))
                        .G;
                };
                worst_limit_h = std::max(worst_limit_h, dev(lim.H, fd_hessian(grad, q, 1e-5)));
            }
        }

        // ---- the *constant* pair Jacobian, vs its own FD Jacobian -------------
        {
            const auto J = J01_of(r);
            Eigen::Matrix<double, 9, 24> Jfd;
            for(int k = 0; k < 24; ++k)
            {
                Dense24v qp = q, qm = q;
                qp(k) += kFdH;
                qm(k) -= kFdH;
                Jfd.col(k) = (F01_of(r, Vector12(qp.segment<12>(0)), Vector12(qp.segment<12>(12)))
                              - F01_of(r, Vector12(qm.segment<12>(0)), Vector12(qm.segment<12>(12))))
                             / (2 * kFdH);
            }
            worst_J = std::max(worst_J, dev(J, Jfd));
        }
    }

    std::printf("[t63b2] FD: prismatic grad %3.1e  H(1e-6) %3.1e (1e-4) %3.1e (1e-3) %3.1e "
                "(1e-2) %3.1e  asym %3.1e  indefinite %d/%d\n",
                worst_g, worst_h6, worst_h4, worst_h3, worst_h2, worst_sym, indefinite, kStates);
    std::printf("[t63b2] FD: the clamp and the FD: dev(H_raw, FD) %3.1e vs dev(H_shipped, FD) "
                "%3.1e; ||H_raw - H_shipped|| %3.1e\n",
                worst_raw, worst_shipped, worst_d_g);
    std::printf("[t63b2] FD: driving grad %3.1e  H %3.1e  (raw chain vs FD %3.1e); "
                "limit grad %3.1e  H %3.1e; constant J01 vs FD %3.1e\n",
                worst_driving_g, worst_driving_h, worst_driving_raw, worst_limit_g,
                worst_limit_h, worst_J);

    // the gradient is a true derivative of the shipped energy
    T63B2_CHECK(c, worst_g <= 1e-9);
    T63B2_CHECK(c, worst_driving_g <= 1e-9);
    T63B2_CHECK(c, worst_limit_g <= 1e-8);
    // the *unclamped* analytic chain is the derivative of the shipped gradient
    // (`worst_raw` below); the shipped (clamped) H deliberately is not
    // the driving's and the limit's *shipped* H are clamped (`make_spd`), while the
    // FD differentiates the *unclamped* gradient: their deviation is O(clamp), not
    // round-off, and is pinned as a lower bound
    T63B2_CHECK(c, worst_driving_h > 1e-4);
    T63B2_CHECK(c, worst_limit_h > 1e-6);
    // symmetry: the q-space Hessian is a sum of symmetric fan-outs (the clamped
    // element Hessian's reconstruction is symmetric to round-off, §7.24 rule 6)
    T63B2_CHECK(c, worst_sym <= 1e-12);
    // the rule-34 basis question, answered *negative* for this file too: the
    // generated `J01T_H01_J01` takes its `H9x9` in the same `F01` order as its
    // `dE01dF01` sibling, so the plain comparison above IS the agreement — asserted
    // by the two numbers being an order of magnitude apart from the *unclamped*
    // chain's (the clamp, not a permutation, is what moves the shipped H)
    T63B2_CHECK(c, worst_raw <= 1e-5);
    T63B2_CHECK(c, worst_shipped > 1e-3);  // the clamp makes `(g, H)` inconsistent
    T63B2_CHECK(c, worst_d_g > 1e-9);
    T63B2_CHECK(c, worst_driving_raw <= 1e-5);
    // the element Hessian is non-diagonal, hence indefinite somewhere
    T63B2_CHECK(c, indefinite > 0);
    // the constant pair Jacobian against its FD Jacobian (the A_j-starts-at-15 trap)
    T63B2_CHECK(c, worst_J <= 1e-8);

    std::printf("[t63b2] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  3) degenerates: the coincident rest pair, zero/negative kappa, reflections and
//     singular states, and every branch of the limit's penalty
// ===========================================================================
TEST_CASE("abd_joint_b_degenerate_inputs", "[luisa][abd_joint_b]")
{
    test_runtime();
    CaseScope scope{"abd_joint_b_degenerate_inputs"};
    auto&     c = scope.counter;

    // ---- (D1) the coincident rest pair: E == 0 and |G| == 0 *bitwise* ---------
    {
        PrismRest r;
        r.ci = Vector3{0.05, 0.1, -0.02};
        r.cj = r.ci;
        r.ti = r.tj = Vector3(1, 0, 0);
        r.ni = r.nj = Vector3(0, 1, 0);
        r.bi = r.bj = Vector3(0, 0, 1);
        const Vector12 qi = q_of(Matrix3x3::Identity(), Vector3{1.0, -0.5, 2.0});
        const Vector12 qj = qi;  // the same pose -> a == 0 and E23's mismatches == 0

        T63B2_CHECK(c, pj_E(2.5, r, qi, qj) == 0.0);
        T63B2_CHECK(c, max_abs(pj_G(2.5, r, qi, qj)) == 0.0);
        const Dense24 H = pj_H(2.5, r, qi, qj);
        T63B2_CHECK(c, all_finite(H));
        T63B2_CHECK(c, max_abs(H) > 0.0);  // the *curvature* is not zero
        T63B2_CHECK(c, min_eigenvalue(H) >= -1e-10 * max_abs(H));
        std::printf("[t63b2] degen: coincident pair E %.1e |G| %.1e |H| %.6e (min eig %.3e)\n",
                    pj_E(2.5, r, qi, qj), max_abs(pj_G(2.5, r, qi, qj)), max_abs(H),
                    min_eigenvalue(H));
    }

    // ---- (D2) zero and negative kappa ----------------------------------------
    {
        PrismRest r;
        r.ci = Vector3{0.3, -0.2, 0.1};
        r.cj = Vector3{0.1, 0.25, -0.15};
        r.set_unit(Matrix3x3::Identity(), Matrix3x3::Identity(), Vector3{0.2, 0.3, 0.9});
        const Vector12 qi = q_of(1.3 * Matrix3x3::Identity(), Vector3{0.1, 0.2, 0.3});
        const Vector12 qj = q_of(0.8 * Matrix3x3::Identity(), Vector3{-0.3, 0.1, 0.2});

        T63B2_CHECK(c, pj_E(0.0, r, qi, qj) == 0.0);
        T63B2_CHECK(c, max_abs(pj_G(0.0, r, qi, qj)) == 0.0);
        T63B2_CHECK(c, max_abs(pj_H(0.0, r, qi, qj)) == 0.0);

        const double kneg = -3.0;
        const auto   lneg = prism_local(kneg, r, qi, qj);
        const Dense24 Hneg = pj_H(kneg, r, qi, qj);
        T63B2_CHECK(c, lneg.E < 0.0);
        T63B2_CHECK(c, dev(pj_G(kneg, r, qi, qj), lneg.G) <= 1e-13);
        T63B2_CHECK(c, rel_err(pj_E(kneg, r, qi, qj), lneg.E) <= 1e-14);
        // the *element* Hessian is `kappa * (...)` (not `kappa I`): for a negative
        // kappa the clamp makes it `-kappa * PSD_of(-H01)` — a *different* matrix from
        // zero, which is the T6.3b2 correction to the T6.3b1 clamp story
        T63B2_CHECK(c, all_finite(Hneg));
        T63B2_CHECK(c, max_abs(Hneg) > 0.0);
        T63B2_CHECK(c, dev(Hneg, lneg.H) <= 1e-8);
        std::printf("[t63b2] degen: kappa = -3 -> E %.6e, |G| %.6e, |H| %.6e (element H01: "
                    "min eig %.3e)\n",
                    lneg.E, max_abs(lneg.G), max_abs(Hneg), min_eigenvalue(lneg.H01));
    }

    // ---- (D3) a reflection and a singular state: finite everywhere ----------
    {
        // both laws are polynomials (no division, no SVD), so a reflection and a
        // rank-deficient `A` are perfectly finite
        PrismRest r;
        r.ci = Vector3{0.1, 0.2, 0.3};
        r.cj = Vector3{-0.1, 0.05, 0.2};
        r.set_unit(Matrix3x3::Identity(), Matrix3x3::Identity(), Vector3{0.4, -0.2, 0.7});

        const Matrix3x3 reflect = Vector3{-1.0, 1.0, 1.0}.asDiagonal();
        const Matrix3x3 singular = Vector3{1.0, 1.0, 0.0}.asDiagonal();
        for(const auto& A : {reflect, singular, Matrix3x3(Matrix3x3::Zero())})
        {
            const Vector12 qi = q_of(A, Vector3{0.1, -0.2, 0.3});
            const Vector12 qj = q_of(A * 0.5, Vector3{-0.1, 0.2, 0.1});
            T63B2_CHECK(c, all_finite(Dense24v(pj_G(1.5, r, qi, qj))));
            T63B2_CHECK(c, all_finite(pj_H(1.5, r, qi, qj)));
            T63B2_CHECK(c, std::isfinite(pj_E(1.5, r, qi, qj)));
            T63B2_CHECK(c, all_finite(Dense24v(dpj_G(1.5, r, qi, qj, 0.2))));
            T63B2_CHECK(c, all_finite(dpj_H(1.5, r, qi, qj, 0.2)));
            T63B2_CHECK(c, std::isfinite(dpj_E(1.5, r, qi, qj, 0.2)));
            T63B2_CHECK(c, std::isfinite(abs_distance(r, qi, qj)));
            T63B2_CHECK(c, all_finite(Dense24v(abs_distance_deriv(r, qi, qj))));
        }
        std::printf("[t63b2] degen: reflection / singular / zero A -> E,G,H finite "
                    "(reflection E %.6e, singular |G| %.3e)\n",
                    pj_E(1.5, r, q_of(reflect), q_of(reflect * 0.5)),
                    max_abs(pj_G(1.5, r, q_of(singular), q_of(singular * 0.5))));
    }

    // ---- (D4) the limit's degenerate branches -------------------------------
    {
        PrismRest r;
        r.ci = Vector3{0.02, -0.03, 0.04};
        r.cj = Vector3{-0.01, 0.05, 0.02};
        r.set_unit(Matrix3x3::Identity(), 1.1 * Matrix3x3::Identity(), Vector3{0.5, 0.1, -0.3});
        const Vector12 qi = q_of(Matrix3x3::Identity(), Vector3{0.1, 0.2, 0.3});
        const Vector12 qj = q_of(1.05 * Matrix3x3::Identity(), Vector3{-0.2, 0.1, 0.25});

        // zero-width range: the *raw* excursion branch
        {
            const auto lim = limit_local(0.0, 0.0, 0.0, 3.0, r, qi, qj);
            const auto ref = penalty_ref(lim.x, 0.0, 0.0, 3.0);
            T63B2_CHECK(c, std::abs(lim.E - ref.E) <= 1e-14);
            T63B2_CHECK(c, std::abs(lim.dE_dx - ref.dE) <= 1e-14);
            T63B2_CHECK(c, std::abs(lim.d2E_dx2 - ref.d2E) <= 1e-14);
            T63B2_CHECK(c, std::abs(lim.dE_dx) > 0.0);
            std::printf("[t63b2] degen: zero-width limit x %.6e -> E %.6e dE_dx %.6e "
                        "d2E_dx2 %.6e\n",
                        lim.x, lim.E, lim.dE_dx, lim.d2E_dx2);
        }

        // the limit *inside* the interval: E = G = H = 0 exactly
        {
            const double x0 = abs_distance(r, qi, qj);
            const auto   lim = limit_local(0.0, x0 - 0.5, x0 + 0.5, 10.0, r, qi, qj);
            T63B2_CHECK(c, lim.E == 0.0);
            T63B2_CHECK(c, max_abs(lim.G) == 0.0);
            T63B2_CHECK(c, max_abs(lim.H) == 0.0);
        }

        // a negative strength: the second Hessian term is clamped to *zero* (the
        // reference's `make_spd(dE_dx * ddxdF01)` with both factors negative), while
        // the first term keeps its sign
        {
            const double x0  = abs_distance(r, qi, qj);
            const auto   lim = limit_local(0.0, x0 - 0.5, x0 + 0.5, -10.0, r, qi, qj);
            // outside the interval: x = x0 is *inside*, so shift the bounds away
            const auto out = limit_local(0.0, x0 + 0.05, x0 + 0.6, -10.0, r, qi, qj);
            (void)lim;
            T63B2_CHECK(c, out.E < 0.0);
            T63B2_CHECK(c, out.d2E_dx2 < 0.0);
            T63B2_CHECK(c, all_finite(out.H));
            const Dense9 dxd = ddxdF01_of();
            const auto  J01  = J01q_of(r);
            const Dense9 H_D  = exact_spd(out.dE_dx * dxd);
            const Dense24 exp = out.d2E_dx2 * (out.dx_dq * out.dx_dq.transpose())
                                + J01.transpose() * H_D * J01;
            T63B2_CHECK(c, dev(out.H, exp) <= 1e-9);
            std::printf("[t63b2] degen: negative limit strength -> E %.6e dE_dx %.6e "
                        "d2E_dx2 %.6e, |H| %.6e (second term clamped: |H_D| %.3e)\n",
                        out.E, out.dE_dx, out.d2E_dx2, max_abs(out.H), max_abs(H_D));
        }
    }

    // ---- (D5) the joint at a *large* coefficient and tiny rest frames --------
    {
        PrismRest r;
        r.ci = Vector3{1e-6, -2e-6, 3e-6};
        r.cj = Vector3{2e-6, 1e-6, -1e-6};
        r.set_unit(Matrix3x3::Identity(), Matrix3x3::Identity(), Vector3{1e-6, 2e-6, -3e-6});
        const Vector12 qi = q_of(Matrix3x3::Identity(), Vector3{1e-6, -1e-6, 2e-6});
        const Vector12 qj = qi;
        const double   k  = 1e12;
        T63B2_CHECK(c, std::isfinite(pj_E(k, r, qi, qj)));
        T63B2_CHECK(c, all_finite(Dense24v(pj_G(k, r, qi, qj))));
        T63B2_CHECK(c, all_finite(pj_H(k, r, qi, qj)));
        std::printf("[t63b2] degen: kappa 1e12 with 1e-6 rest frames -> E %.3e |G| %.3e "
                    "|H| %.3e (no conditioning cliff)\n",
                    pj_E(k, r, qi, qj), max_abs(pj_G(k, r, qi, qj)), max_abs(pj_H(k, r, qi, qj)));
    }

    std::printf("[t63b2] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}


// ===========================================================================
//  4) the assembly fan-out (differencing two real assemblies) + the three
//     energy kernels
// ===========================================================================
TEST_CASE("abd_joint_b_assembly", "[luisa][abd_joint_b]")
{
    test_runtime();
    CaseScope scope{"abd_joint_b_assembly"};

    PrismScene scene_data;

    // ---- (A) the same 2-body scene twice, with and without the joints -----------
    Engine engine_a{"luisa", workspace_path("uipc_luisa_abd_joint_b_base")};
    World  world_a{engine_a};
    Scene  base_scene = scene_data.build("abd_joint_b_base", false);
    world_a.init(base_scene);
    REQUIRE(world_a.is_valid());
    REQUIRE(T63B2Probe::abd != nullptr);
    REQUIRE(T63B2Probe::abd_linear != nullptr);
    REQUIRE(T63B2Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B2Probe::abd_linear);
    auto*      linear_a = T63B2Probe::abd_linear;
    const auto base     = run_assembly(*linear_a);

    Engine engine_b{"luisa", workspace_path("uipc_luisa_abd_joint_b_joint")};
    World  world_b{engine_b};
    Scene  joint_scene = scene_data.build("abd_joint_b_joint", true);
    world_b.init(joint_scene);
    REQUIRE(world_b.is_valid());
    REQUIRE(T63B2Probe::abd != nullptr);
    REQUIRE(T63B2Probe::abd_linear != nullptr);
    REQUIRE(T63B2Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B2Probe::abd_linear);
    auto*      abd_b     = T63B2Probe::abd;
    auto*      linear_b  = T63B2Probe::abd_linear;
    const auto joint_run = run_assembly(*linear_b);

    auto& c = scope.counter;

    // ---- (B) uid ownership (proved, see the file comment) ----------------------
    const auto* joint = T63B2Probe::joint;
    const auto* drv   = T63B2Probe::driving;
    const auto* lim   = T63B2Probe::limit;
    const auto* econst = T63B2Probe::ext_constraint;

    const bool joint_owns = joint != nullptr && joint->is_valid() && joint->h_body_ids.size() == 2u;
    const bool drv_owns =
        drv != nullptr && drv->is_valid() && drv->h_is_constrained.size() == 2u;
    const bool lim_owns = lim != nullptr && lim->is_valid() && lim->h_ref_qs.size() == 2u;
    const bool econst_owns =
        econst != nullptr && econst->is_valid() && econst->h_forces.size() == 2u;
    std::printf("[t63b2] assembly uid ownership: uid20 = %s (%zu joints), uid21 = %s (%zu), "
                "uid669 = %s (%zu), uid667(constraint) = %s (%zu)\n",
                joint_owns ? "uipc::backend::luisa::AffineBodyPrismaticJoint" : "ANOTHER SYSTEM",
                joint ? joint->h_body_ids.size() : 0u,
                drv_owns ? "uipc::backend::luisa::AffineBodyDrivingPrismaticJoint"
                         : "ANOTHER SYSTEM",
                drv ? drv->h_is_constrained.size() : 0u,
                lim_owns ? "uipc::backend::luisa::AffineBodyPrismaticJointLimit"
                         : "ANOTHER SYSTEM",
                lim ? lim->h_ref_qs.size() : 0u,
                econst_owns
                    ? "uipc::backend::luisa::AffineBodyPrismaticJointExternalForceConstraint"
                    : "ANOTHER SYSTEM",
                econst ? econst->h_forces.size() : 0u);
    REQUIRE(joint_owns);
    REQUIRE(drv_owns);
    REQUIRE(lim_owns);
    REQUIRE(econst_owns);

    // ---- (C) the extent budget difference --------------------------------------
    T63B2_CHECK(c, base.dof_count == 24u);
    T63B2_CHECK(c, base.block_count == 2u);
    T63B2_CHECK(c, joint_run.dof_count == 24u);
    // hand-derived: the two joints add their upper-triangular 2x2 12x12-block run (3
    // blocks each) AND so does the joint-limit constitution (its hook is driven by the
    // linear subsystem), i.e. 2 * (3 + 3) = 12 more blocks -> 12 * 16 more triplets.
    // The *driving* constraint's blocks are absent here because its extents are
    // reported by `InterAffineBodyAnimator::step`, which only runs in `advance()`
    // (the `scene` case measures 2 + 18 after one advance).
    T63B2_CHECK(c, joint_run.block_count == 2u + 12u);
    T63B2_CHECK(c, joint_run.triplet_count == base.triplet_count + 12u * 16u);
    std::printf("[t63b2] assembly extent: base %zu triplets / %zu blocks, joint %zu / %zu\n",
                base.triplet_count, base.block_count, joint_run.triplet_count,
                joint_run.block_count);

    const auto bodies = scene_bodies(world_b, *abd_b);
    REQUIRE(bodies.size() == 2u);
    const Vector12 q_body[2] = {bodies[0].q, bodies[1].q};

    // ---- (D) the three energy kernels, on the real dynamics windows -------------
    {
        const auto dt      = T63B2Probe::self->dt();
        const auto dt_attr = T63B2Probe::self->dt_attr();

        const auto e_joint = run_constitution_energy(*T63B2Probe::joint, *abd_b, dt_attr, dt, 2);
        const auto e_drv   = run_constraint_energy(*T63B2Probe::driving, *abd_b, dt_attr, dt, 2);
        const auto e_lim   = run_constitution_energy(*T63B2Probe::limit, *abd_b, dt_attr, dt, 2);
        REQUIRE(e_joint.size() == 2u);
        REQUIRE(e_drv.size() == 2u);
        REQUIRE(e_lim.size() == 2u);

        // the *same* `kappa = strength_ratio * (m_i + m_j)` every kernel uses
        const auto masses   = read_device(abd_b->body_masses());
        const auto kappa_of = [&](double ratio)
        { return ratio * (masses[0].m_mass + masses[1].m_mass); };

        double worst = 0.0;
        double worst_drv = 0.0;
        double worst_lim = 0.0;
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto r  = scene_rest(scene_data, edge);
            const auto lb = edge_l_body(edge);
            const auto rb = edge_r_body(edge);

            const double k  = kappa_of(scene_data.ratio);
            worst           = std::max(
                worst, rel_err(e_joint[edge], pj_E(k, r, q_body[lb], q_body[rb])));

            const double kd = kappa_of(scene_data.dr_ratio);
            worst_drv       = std::max(
                worst_drv,
                rel_err(e_drv[edge], dpj_E(kd, r, q_body[lb], q_body[rb], scene_data.aim)));

            // the limit's `x` is the reference's absolute distance and its bounds are
            // shifted by the joint's `init_distance` (zero here)
            const double x = abs_distance(r, q_body[lb], q_body[rb]);
            worst_lim      = std::max(worst_lim,
                                      rel_err(e_lim[edge],
                                              penalty_ref(x, scene_data.low, scene_data.high,
                                                          scene_data.lim_str)
                                                  .E));
        }
        std::printf("[t63b2] energy kernels: joint [%.8e %.8e], driving [%.8e %.8e], limit "
                    "[%.8e %.8e]; worst rel %.3e / %.3e / %.3e\n",
                    e_joint[0], e_joint[1], e_drv[0], e_drv[1], e_lim[0], e_lim[1], worst,
                    worst_drv, worst_lim);
        T63B2_CHECK(c, worst <= 1e-13);
        T63B2_CHECK(c, worst_drv <= 1e-13);
        T63B2_CHECK(c, worst_lim <= 1e-13);
        T63B2_CHECK(c, std::abs(e_joint[0]) > 1e-12);
        T63B2_CHECK(c, std::abs(e_drv[0]) > 1e-12);
        T63B2_CHECK(c, std::abs(e_lim[0]) > 1e-12);
    }

    // ---- (E) the assembled COO and gradient *difference* vs the hand-derived joint
    //      contribution ------------------------------------------------------
    const int body_of_uid0 = static_cast<int>(bodies[0].body);
    const int body_of_uid1 = static_cast<int>(bodies[1].body);

    const auto   masks       = read_device(abd_b->body_masses());
    const double kappa_joint = scene_data.ratio * (masks[0].m_mass + masks[1].m_mass);
    const double kappa_drv   = scene_data.dr_ratio * (masks[0].m_mass + masks[1].m_mass);

    {
        const int body_of[2] = {body_of_uid0, body_of_uid1};
        // the joint's + the joint-limit's blocks and gradient scatter (no driving: the
        // animator has not stepped yet, so its extents are still zero)
        const BlockMap  expected =
            scene_expected_blocks(scene_data, q_body, body_of, kappa_joint, kappa_drv, false);
        Eigen::VectorXd expected_g = Eigen::VectorXd::Zero(24);
        auto scatter_into = [](Eigen::VectorXd& target, const Dense24v& G, int bi, int bj)
        {
            target.segment<12>(static_cast<Eigen::Index>(bi) * 12) += G.segment<12>(0);
            target.segment<12>(static_cast<Eigen::Index>(bj) * 12) += G.segment<12>(12);
        };

        for(int edge = 0; edge < 2; ++edge)
        {
            const auto r  = scene_rest(scene_data, edge);
            const auto lb = edge_l_body(edge);
            const auto rb = edge_r_body(edge);
            const int  bi = body_of[lb];
            const int  bj = body_of[rb];

            scatter_into(expected_g, pj_G(kappa_joint, r, q_body[lb], q_body[rb]), bi, bj);
            scatter_into(expected_g,
                         limit_local(0.0, scene_data.low, scene_data.high,
                                     scene_data.lim_str, r, q_body[lb], q_body[rb])
                             .G,
                         bi,
                         bj);
        }

        const BlockMap produced = sub_blocks(joint_run.blocks, base.blocks);

        double worst_entry = 0.0, worst_block = 0.0;
        int    compared = 0, missing = 0, extra = 0;
        for(const auto& [key, blk] : expected)
        {
            auto it = produced.find(key);
            if(it == produced.end())
            {
                std::printf("[t63b2] assembly: missing key (%d, %d)\n", key.first, key.second);
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
        std::printf("[t63b2] assembly COO (joint-only, by differencing): %d keys compared, "
                    "worst |entry| %.3e, worst |block| %.3e, missing %d, non-zero extra %d\n",
                    compared, worst_entry, worst_block, missing, extra);
        T63B2_CHECK(c, missing == 0);
        T63B2_CHECK(c, extra == 0);
        // the *descending* pair (edge 1) shares its (min, max) key set with the
        // ascending one, so the two joints' off-diagonal blocks land on the same keys
        T63B2_CHECK(c, produced.size() == expected.size());
        T63B2_CHECK(c, compared == static_cast<int>(expected.size()));
        T63B2_CHECK(c, worst_entry <= 1e-6);
        T63B2_CHECK(c, worst_block > 0.0);
        T63B2_CHECK(c, worst_block <= 1e-8);  // the assembled blocks are *relative*

        Eigen::VectorXd produced_g = joint_run.gradient - base.gradient;
        double          worst_g    = 0.0;
        for(Eigen::Index i = 0; i < expected_g.size(); ++i)
            worst_g = std::max(worst_g, rel_err(produced_g(i), expected_g(i)));
        std::printf("[t63b2] assembly gradient (joint-only): worst rel %.3e, |b_joint| %.6e\n",
                    worst_g, expected_g.norm());
        T63B2_CHECK(c, expected_g.norm() > 1.0);
        T63B2_CHECK(c, worst_g <= 1e-11);

        // ---- (F) the absolute gradient (base + joints), which pins that the joint
        //      really lands at the bodies' dof offsets
        Eigen::VectorXd expected_abs = base.gradient + expected_g;
        double          worst_abs    = 0.0;
        for(Eigen::Index i = 0; i < expected_abs.size(); ++i)
            worst_abs = std::max(worst_abs, rel_err(joint_run.gradient(i), expected_abs(i)));
        std::printf("[t63b2] assembly gradient (absolute): worst rel %.3e (|b| %.6e)\n",
                    worst_abs, expected_abs.norm());
        T63B2_CHECK(c, expected_abs.norm() > 1.0);
        T63B2_CHECK(c, worst_abs <= 1e-11);
    }

    // ---- (G) the driving constraint's k1/k2 kernels, driven directly -----------
    //  The `scene` case measures that the driving's *assembled* contribution is
    //  absent while its extents are reserved (a FINDING of this batch, reported);
    //  this section isolates the kernel question by handing
    //  `do_compute_gradient_hessian` a `GradientHessianInfo` over test-owned COO
    //  buffers -- exactly the situation the animator's linear-subsystem reporter
    //  creates.  If the values below match, the kernels are right and the gap is in
    //  the animator/assembly path; if they are zero, the kernel is at fault.
    {
        const auto dt      = T63B2Probe::self->dt();
        const auto dt_attr = T63B2Probe::self->dt_attr();
        const int  n_pairs = 2;

        lud::LDeviceBuffer<int32_t> g_idx(2 * n_pairs);
        lud::LDeviceBuffer<double>  g_val(12 * 2 * n_pairs);
        // here the constraint is handed a `LTripletMatrixView<Float,12,12>`, so a
        // triplet *is* one 12x12 block (144 doubles, column-major per block) and the
        // reporter's 3 blocks per pair are 3 triplets
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
            g_idx.view(),
            g_val.view(),
            std::uint32_t{2 * n_pairs},
            std::int32_t{0},
            std::int32_t{2 * n_pairs},
            std::int32_t{2 * n_pairs},
            std::int32_t{0},
            std::uint32_t{2 * n_pairs});
        auto Hv = lla::LTripletMatrixView<Float, 12, 12>::from_handles_full(
            h_row.view(),
            h_col.view(),
            h_val.view(),
            std::int32_t{8},
            std::int32_t{8},
            std::int32_t{0},
            std::int32_t{n_triplets},
            std::int32_t{n_triplets},
            std::int32_t{0},
            std::int32_t{0},
            std::int32_t{8},
            std::int32_t{8});

        gg::InterAffineBodyAnimator::Impl aimpl;
        aimpl.affine_body_dynamics = abd_b;
        aimpl.dt_attr              = dt_attr;
        // the animator's `GradientHessianInfo` *slices* the view it is handed by the
        // constraint's own offset table (`subview(offset, count)`), so a bare Impl
        // needs that table filled (the same table the reporter's extent uses)
        aimpl.constraint_gradient_offsets_counts.resize(1);
        aimpl.constraint_gradient_offsets_counts.counts()[0] =
            static_cast<IndexT>(2 * n_pairs);
        aimpl.constraint_gradient_offsets_counts.scan();
        aimpl.constraint_hessian_offsets_counts.resize(1);
        aimpl.constraint_hessian_offsets_counts.counts()[0] =
            static_cast<IndexT>(3 * n_pairs);
        aimpl.constraint_hessian_offsets_counts.scan();
        gg::InterAffineBodyAnimator::GradientHessianInfo ghi{&aimpl, 0, dt, Gv, Hv, false};
        T63B2Probe::driving->do_compute_gradient_hessian(ghi);
        gg::frame_stream().synchronize();

        const auto idx    = read_device(g_idx.cview());
        const auto vals   = read_device(g_val.cview());
        const auto rows   = read_device(h_row.cview());
        const auto cols   = read_device(h_col.cview());
        const auto hvals  = read_device(h_val.cview());

        // the produced blocks: here a triplet *is* a 12x12 block (144 doubles) and its
        // row/col are the *raw body ids* of the pair (measured: the constraint writes
        // `bids` into this view), not the animator's `body * 4 + a` sub-blocks that the
        // assembled COO carries
        std::map<std::pair<int, int>, Dense12> produced_drv;
        for(std::size_t t = 0; t < rows.size(); ++t)
        {
            Dense12 blk;
            for(int cc = 0; cc < 12; ++cc)
                for(int rr = 0; rr < 12; ++rr)
                    blk(rr, cc) =
                        hvals[t * 144 + static_cast<std::size_t>(cc * 12 + rr)];
            auto key = std::make_pair(static_cast<int>(rows[t]), static_cast<int>(cols[t]));
            auto it  = produced_drv.find(key);
            if(it == produced_drv.end())
                produced_drv.emplace(key, blk);
            else
                it->second += blk;
        }

        const int body_of2[2] = {body_of_uid0, body_of_uid1};
        Eigen::VectorXd expected_g2 = Eigen::VectorXd::Zero(24);
        std::map<std::pair<int, int>, Dense12> expected_h2;
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto  r  = scene_rest(scene_data, edge);
            const auto  lb = edge_l_body(edge);
            const auto  rb = edge_r_body(edge);
            const int bi = body_of2[lb];
            const int bj = body_of2[rb];
            const auto  G  = dpj_G(kappa_drv, r, q_body[lb], q_body[rb], scene_data.aim);
            expected_g2.segment<12>(static_cast<Eigen::Index>(bi) * 12) += G.segment<12>(0);
            expected_g2.segment<12>(static_cast<Eigen::Index>(bj) * 12) += G.segment<12>(12);

            const Dense24 H = dpj_H(kappa_drv, r, q_body[lb], q_body[rb], scene_data.aim);
            auto add12 = [&](int rr, int cc, const Dense12& blk)
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
            // `H.block(1,0)` -- exactly what `write_pair_half_hessian` reproduces
            const bool ascending = bi < bj;
            add12(bi, bi, Dense12(H.block<12, 12>(0, 0)));
            add12(bj, bj, Dense12(H.block<12, 12>(12, 12)));
            add12(std::min(bi, bj),
                  std::max(bi, bj),
                  ascending ? Dense12(H.block<12, 12>(0, 12))
                            : Dense12(H.block<12, 12>(12, 0)));
        }

        double worst_g2 = 0.0, worst_h2 = 0.0;
        int compared2 = 0;
        for(const auto& [key, blk] : expected_h2)
        {
            auto it = produced_drv.find(key);
            if(it == produced_drv.end())
                continue;
            ++compared2;
            worst_h2 = std::max(worst_h2, dev(it->second, blk));
        }
        // the gradient doublets: `(index, value)` pairs of 12-vectors, accumulated
        Eigen::VectorXd produced_g2 = Eigen::VectorXd::Zero(24);
        for(std::size_t t = 0; t < idx.size(); ++t)
            for(int k = 0; k < 12; ++k)
                produced_g2(static_cast<Eigen::Index>(idx[t]) * 12 + k) +=
                    vals[t * 12 + static_cast<std::size_t>(k)];
        for(Eigen::Index k = 0; k < expected_g2.size(); ++k)
            worst_g2 = std::max(worst_g2, rel_err(produced_g2(k), expected_g2(k)));

        std::printf("[t63b2] assembly: the driving's k1/k2 kernels driven directly: "
                    "gradient worst rel %.3e (|b| %.6e, produced %.6e), Hessian worst dev "
                    "%.3e (%d of %zu keys, %zu produced)/\n",
                    worst_g2,
                    expected_g2.norm(),
                    produced_g2.norm(),
                    worst_h2,
                    compared2,
                    expected_h2.size(),
                    produced_drv.size());
        T63B2_CHECK(c, expected_g2.norm() > 1.0);
        T63B2_CHECK(c, worst_g2 <= 1e-10);
        T63B2_CHECK(c, !expected_h2.empty());
        T63B2_CHECK(c, compared2 == static_cast<int>(expected_h2.size()));
        T63B2_CHECK(c, produced_drv.size() == expected_h2.size());
        T63B2_CHECK(c, worst_h2 <= 1e-8);
        T63B2_CHECK(c, worst_h2 <= 1e-8);
    }

    std::printf("[t63b2] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  5) a real scene: two ABD bodies, one prismatic joint carrying a driving
//     instance, an external-force instance and a limit instance, one `advance()`
// ===========================================================================
// ===========================================================================
TEST_CASE("abd_joint_b_scene", "[luisa][abd_joint_b]")
{
    test_runtime();
    CaseScope scope{"abd_joint_b_scene"};

    PrismScene scene_data;
    Scene      scene = scene_data.build("abd_joint_b_scene", true);

    Engine engine{"luisa", workspace_path("uipc_luisa_abd_joint_b_scene")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    auto& c = scope.counter;
    REQUIRE(T63B2Probe::abd != nullptr);
    REQUIRE(T63B2Probe::abd_linear != nullptr);
    REQUIRE(T63B2Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B2Probe::abd_linear);

    // ---- (A) the per-edge *host* bookkeeping, against the scene's own definitions --
    {
        REQUIRE(T63B2Probe::joint != nullptr);
        REQUIRE(T63B2Probe::joint->is_valid());
        const auto& j = *T63B2Probe::joint;
        REQUIRE(j.h_body_ids.size() == 2u);
        REQUIRE(j.h_rest_cs.size() == 2u);
        REQUIRE(j.h_rest_ts.size() == 2u);
        REQUIRE(j.h_rest_ns.size() == 2u);
        REQUIRE(j.h_rest_bs.size() == 2u);
        REQUIRE(j.h_strength_ratios.size() == 2u);
        REQUIRE(j.h_init_distances.size() == 2u);

        auto cmp6 = [&](const Vector6& got, const Vector6& want)
        {
            double w = 0.0;
            for(int k = 0; k < 6; ++k)
                w = std::max(w, std::abs(got(k) - want(k)));
            return w;
        };
        double worst_rest = 0.0;
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto want = expected_pair(scene_data, edge);
            worst_rest      = std::max(worst_rest, cmp6(j.h_rest_cs[edge], want.c));
            worst_rest      = std::max(worst_rest, cmp6(j.h_rest_ts[edge], want.t));
            worst_rest      = std::max(worst_rest, cmp6(j.h_rest_ns[edge], want.n));
            worst_rest      = std::max(worst_rest, cmp6(j.h_rest_bs[edge], want.b));
        }
        std::printf("[t63b2] scene host bookkeeping: %zu joints, worst |rest - hand| %.3e, "
                    "ratios %.3f/%.3f, init %.1e/%.1e\n",
                    j.h_body_ids.size(), worst_rest, j.h_strength_ratios[0],
                    j.h_strength_ratios[1], j.h_init_distances[0], j.h_init_distances[1]);
        T63B2_CHECK(c, worst_rest <= 1e-14);
        T63B2_CHECK(c, j.h_strength_ratios[0] == scene_data.ratio);
        T63B2_CHECK(c, j.h_strength_ratios[1] == scene_data.ratio);
        // the frontend's `init_distance` attribute is created zero and never written
        T63B2_CHECK(c, j.h_init_distances[0] == 0.0);
        T63B2_CHECK(c, j.h_geo_joint_offsets_counts.total_count() == 2);
        // the ascending / descending pair split (the `upper_LR` path is exercised by
        // the *scene*, not only by the unit cases)
        T63B2_CHECK(c, j.h_body_ids[0](0) < j.h_body_ids[0](1));
        T63B2_CHECK(c, j.h_body_ids[1](0) > j.h_body_ids[1](1));

        // the device buffers mirror the host staging
        {
            const auto h_bids = read_device(j.body_ids.cview());
            REQUIRE(h_bids.size() == 2u);
            double worst = 0.0;
            for(std::size_t i = 0; i < h_bids.size(); ++i)
            {
                const auto e = gg::to_eigen(h_bids[i]);
                for(int k = 0; k < 2; ++k)
                    worst = std::max(worst,
                                     std::abs(static_cast<double>(e(k))
                                              - static_cast<double>(j.h_body_ids[i](k))));
            }
            // NOTE: the reference reads this state out of the *instance transform*
            // attribute (`transform_to_q(transforms()[i])`) rather than out of `q`, and
            // the ABD dynamics has rewritten that attribute from `q` in the meantime
            // (`q_to_transform`), so the two agree to round-off (measured 1.1e-17), not
            // bitwise — a comparison with `== 0.0` would be a scale trap (§7.24 rule 14).
            T63B2_CHECK(c, worst <= 1e-15);
        }

        // the *polar* rotation finding re-pinned (T6.3b1 §7.25 rule 3): the edge
        // direction is built from the bodies' *full* transforms while the rest rows
        // divide by `LT.rotation()`, so a stretched state does not give `A_l t_l == t`
        {
            const auto  r      = scene_rest(scene_data, 0);
            const auto  Ai     = scene_data.A0;
            const auto  lp0    = scene_data.l_pos0_0;
            const auto  lp1    = scene_data.l_pos1_0;
            const auto  t_w    = (Ai * (lp1 - lp0)).normalized();
            const auto  a_t    = Ai * r.ti;
            const auto  a_pol  = polar_rotation(Ai) * r.ti;
            const double dev_lin = (a_t - t_w).norm();
            const double dev_pol = (a_pol - t_w).norm();
            std::printf("[t63b2] scene frame: |A t_bar - t| %.6e (linear part) vs "
                        "|R_polar t_bar - t| %.6e (the reference's `rotation()`) -> the "
                        "polar spelling is the exact one\n",
                        dev_lin, dev_pol);
            T63B2_CHECK(c, dev_pol <= 1e-14);
            T63B2_CHECK(c, dev_lin > 1e-3);  // the body is stretched, so the two differ
        }

        // the ported `orthonormal_basis` vs the hand spelling of the same routine
        {
            const auto r = scene_rest(scene_data, 0);
            Vector3    t_w = (A_of(q_of(scene_data.A0)) * r.ti).eval();
            Vector3    n_w, b_w;
            hand_orthonormal_basis(t_w, n_w, b_w);
            auto t_l = gg::to_luisa(t_w);
            auto n_l = lps::LVec<Float, 3>{};
            auto b_l = lps::LVec<Float, 3>{};
            gg::orthonormal_basis<Float>(t_l, n_l, b_l);
            double worst = 0.0;
            for(int k = 0; k < 3; ++k)
            {
                worst = std::max(worst, std::abs(static_cast<double>(n_l(k)) - n_w(k)));
                worst = std::max(worst, std::abs(static_cast<double>(b_l(k)) - b_w(k)));
            }
            T63B2_CHECK(c, worst <= 1e-15);
        }
    }

    // ---- (B) the three companion systems' host records --------------------------
    {
        REQUIRE(T63B2Probe::driving != nullptr);
        REQUIRE(T63B2Probe::driving->is_valid());
        REQUIRE(T63B2Probe::ext_constraint != nullptr);
        REQUIRE(T63B2Probe::ext_constraint->is_valid());
        REQUIRE(T63B2Probe::limit != nullptr);
        REQUIRE(T63B2Probe::limit->is_valid());

        const auto& drv    = *T63B2Probe::driving;
        const auto& econst = *T63B2Probe::ext_constraint;
        const auto& lim    = *T63B2Probe::limit;

        REQUIRE(drv.h_is_constrained.size() == 2u);
        T63B2_CHECK(c, drv.h_is_constrained[0] == scene_data.constrained);
        T63B2_CHECK(c, drv.h_strength_ratios[0] == scene_data.dr_ratio);
        T63B2_CHECK(c, drv.h_is_passive[0] == scene_data.passive);
        T63B2_CHECK(c, drv.h_aim_distances[0] == scene_data.aim);
        T63B2_CHECK(c, drv.h_geo_joint_offsets_counts.total_count() == 2);

        REQUIRE(econst.h_forces.size() == 2u);
        T63B2_CHECK(c, econst.h_forces[0] == scene_data.ext_force);
        T63B2_CHECK(c, econst.h_is_constrained[0] == scene_data.ext_constrained);

        REQUIRE(lim.h_ref_qs.size() == 2u);
        T63B2_CHECK(c, lim.h_lowers[0] == scene_data.low);
        T63B2_CHECK(c, lim.h_uppers[0] == scene_data.high);
        T63B2_CHECK(c, lim.h_strengths[0] == scene_data.lim_str);
        // `ref_qs` is the reference's per-entry `[q_l ; q_r]` *recorded state*
        {
            const Vector12 q0 = q_of(scene_data.A0, scene_data.t0);
            const Vector12 q1 = q_of(scene_data.A1, scene_data.t1);
            double         worst = 0.0;
            for(int k = 0; k < 12; ++k)
            {
                worst = std::max(worst, std::abs(lim.h_ref_qs[0](k) - q0(k)));
                worst = std::max(worst, std::abs(lim.h_ref_qs[0](12 + k) - q1(k)));
                worst = std::max(worst, std::abs(lim.h_ref_qs[1](k) - q1(k)));
                worst = std::max(worst, std::abs(lim.h_ref_qs[1](12 + k) - q0(k)));
            }
            T63B2_CHECK(c, worst == 0.0);
            // (the reference never reads `ref_qs` back in a kernel: its only use is the
            // extent `ref_qs.size()`; the buffer is kept so the *recorded* state is the
            // reference's)
        }
    }

    // ---- (C) the dof layout and the frontend `distance` attribute ----------------
    const auto bodies = scene_bodies(world, *T63B2Probe::abd);
    REQUIRE(bodies.size() == 2u);
    const Vector12 q_body[2] = {bodies[0].q, bodies[1].q};

    {
        double worst = 0.0;
        auto   visitor   = uipc::backend::WorldVisitor{world};
        auto   geo_slots = visitor.scene().geometries();
        for(auto& slot : geo_slots)
        {
            auto sc = slot->geometry().as<SimplicialComplex>();
            if(!sc)
                continue;
            auto distance = sc->edges().find<Float>("distance");
            if(!distance)
                continue;
            const auto dview = view(*distance);
            for(int edge = 0; edge < 2; ++edge)
            {
                const auto   r = scene_rest(scene_data, edge);
                const double expected = abs_distance(r, q_body[edge_l_body(edge)],
                                                     q_body[edge_r_body(edge)]);
                worst = std::max(worst, std::abs(dview[edge] - expected));
            }
        }
        std::printf("[t63b2] scene: the frontend `distance` attribute vs the hand-derived "
                    "absolute distance, worst |dev| %.3e\n",
                    worst);
        T63B2_CHECK(c, worst <= 1e-12);
    }
    {
        // the joint's dof count/offset contribution: none of its own -- the ABD dofs
        // are the bodies' 12 each, and the joint's persistent state is the scalar
        // `current_distances`
        gg::GlobalLinearSystem::Impl impl;
        impl.diag_subsystems.register_sim_system(*T63B2Probe::abd_linear);
        impl.init();
        T63B2_CHECK(c, impl.diag_dof_offsets_counts.total_count() == 24);
        T63B2_CHECK(c, bodies[0].dof_offset == 0u);
        T63B2_CHECK(c, bodies[1].dof_offset == 12u);
        REQUIRE(T63B2Probe::integrator != nullptr);
        T63B2_CHECK(c, T63B2Probe::integrator->is_valid());
        REQUIRE(T63B2Probe::dof_reporter != nullptr);
        T63B2_CHECK(c, T63B2Probe::dof_reporter->is_valid());
        std::printf("[t63b2] scene: dof count %d (the joint registers no dof of its own: "
                    "its reporter/integrator re-sync the per-joint scalar state)\n",
                    static_cast<int>(impl.diag_dof_offsets_counts.total_count()));
    }

    // ---- (D) one advance(): the state, the prediction, the force chain -----------
    auto feature = world.features().find<AffineBodyStateAccessorFeature>();
    REQUIRE(feature != nullptr);
    T63B2_CHECK(c, feature->body_count() == 2);

    const auto dt = T63B2Probe::self->dt();

    world.advance();
    REQUIRE(world.is_valid());
    world.sync();
    REQUIRE(world.is_valid());

    {
        const auto bodies_after = scene_bodies(world, *T63B2Probe::abd);
        REQUIRE(bodies_after.size() == 2u);

        // the ABD *state* is still the scene's initial state (the solve is T8.3's)
        double worst_state = 0.0;
        for(int i = 0; i < 2; ++i)
            for(int k = 0; k < 12; ++k)
                worst_state = std::max(worst_state, std::abs(bodies_after[i].q(k) - q_body[i](k)));

        // the BDF1 prediction, *including* the external-force acceleration the reporter
        // chain produced (the frame order is clear -> animator step -> force manager
        // step -> predict, so the acceleration is available)
        const auto q_tilde = read_vec12(T63B2Probe::abd->q_tildes());
        const auto accs    = read_vec12(T63B2Probe::abd->body_external_force_accs());
        const auto forces  = read_vec12(T63B2Probe::abd->body_external_forces());
        REQUIRE(q_tilde.size() == 2u);
        REQUIRE(accs.size() == 2u);
        REQUIRE(forces.size() == 2u);

        const Matrix3x3 V[2] = {scene_data.V0, scene_data.V1};
        const Vector3   v[2] = {scene_data.v0, scene_data.v1};

        // the hand-derived force pair per edge
        Dense12v F_body[2] = {Dense12v::Zero(), Dense12v::Zero()};
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto  r  = scene_rest(scene_data, edge);
            const auto  lb = edge_l_body(edge);
            const auto  rb = edge_r_body(edge);
            const auto  ef =
                ext_force_local(scene_data.ext_force, r, q_body[lb], q_body[rb]);
            F_body[lb] += ef.F_i;
            F_body[rb] += ef.F_j;
        }

        double worst_pred = 0.0, worst_force = 0.0, worst_acc = 0.0, worst_state2 = 0.0;
        for(int i = 0; i < 2; ++i)
        {
            const Vector12 qv = gg::transform_to_q(make_transform(V[i], v[i]));
            const Dense12  M  = bodies[i].mass;
            const Dense12v a_ext = M.inverse() * F_body[i];
            const Vector12 expected =
                q_body[i] + static_cast<double>(dt) * qv + std::pow(dt, 2) * a_ext;

            for(int k = 0; k < 12; ++k)
            {
                worst_pred = std::max(worst_pred,
                                      std::abs(q_tilde[bodies[i].body](k) - expected(k)));
                worst_force =
                    std::max(worst_force,
                             rel_err(forces[bodies[i].body](k), F_body[i](k)));
                worst_acc = std::max(worst_acc,
                                     rel_err(accs[bodies[i].body](k), a_ext(k)));
                worst_state2 = std::max(worst_state2, std::abs(bodies_after[i].q(k) - expected(k)));
            }
        }
        std::printf("[t63b2] scene: advance -> |q_state - q_initial| %.3e, "
                    "|q_tilde - (q + q_v dt + a_ext dt^2)| %.3e (dt %.6f), external force rel "
                    "%.3e, a_ext rel %.3e\n",
                    worst_state, worst_pred, static_cast<double>(dt), worst_force, worst_acc);
        T63B2_CHECK(c, worst_state == 0.0);
        T63B2_CHECK(c, worst_pred <= 1e-12);
        T63B2_CHECK(c, worst_force <= 1e-13);
        T63B2_CHECK(c, worst_acc <= 1e-12);
        T63B2_CHECK(c, Dense12v(F_body[0]).norm() > 1.0);
        // ... and the prediction is NOT the force-free one (the reporter really
        // contributed to the prediction)
        T63B2_CHECK(c, worst_state2 > 1e-9);
    }

    // ---- (E) the dof reporter / time integrator re-sync --------------------------
    {
        // Make the persistent state *wrong* on the device, then let the reporter's
        // `do_update_dof_attributes` restore it: that is exactly the hook's contract
        // (the reference re-syncs `current_distances` whenever the layout is rebuilt).
        auto& j = *T63B2Probe::joint;
        {
            std::vector<Float> wrong(j.h_current_distances.size(), -12345.0);
            if(!wrong.empty())
            {
                j.current_distances.view().copy_from(span<const Float>{wrong});
                lud::main_stream().synchronize();
            }
        }
        REQUIRE(T63B2Probe::joint_dof != nullptr);
        T63B2Probe::joint_dof->update_dof_attributes();
        gg::frame_stream().synchronize();
        const auto restored = read_device(j.current_distances.cview());
        REQUIRE(restored.size() == 2u);
        double worst = 0.0;
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto   r = scene_rest(scene_data, edge);
            const double expected =
                abs_distance(r, q_body[edge_l_body(edge)], q_body[edge_r_body(edge)]);
            worst = std::max(worst, std::abs(static_cast<double>(restored[edge]) - expected));
        }
        std::printf("[t63b2] scene: the dof reporter restored the per-joint state, worst "
                    "|current - hand| %.3e (from a poisoned buffer)\n",
                    worst);
        T63B2_CHECK(c, worst <= 1e-12);
    }

    // ---- (F) the state's energy through the real kernels + the cross-body blocks ---
    {
        const auto dt_attr = T63B2Probe::self->dt_attr();
        const auto e_joint = run_constitution_energy(*T63B2Probe::joint, *T63B2Probe::abd,
                                                     dt_attr, dt, 2);
        const auto e_drv   = run_constraint_energy(*T63B2Probe::driving, *T63B2Probe::abd,
                                                   dt_attr, dt, 2);
        const auto e_lim   = run_constitution_energy(*T63B2Probe::limit, *T63B2Probe::abd,
                                                     dt_attr, dt, 2);

        const auto   masses  = read_device(T63B2Probe::abd->body_masses());
        const double k_joint = scene_data.ratio * (masses[0].m_mass + masses[1].m_mass);
        const double k_drv   = scene_data.dr_ratio * (masses[0].m_mass + masses[1].m_mass);

        double worst = 0.0, worst_drv = 0.0, worst_lim = 0.0;
        const int body_of[2] = {static_cast<int>(bodies[0].body),
                                static_cast<int>(bodies[1].body)};
        const BlockMap expected =
            scene_expected_blocks(scene_data, q_body, body_of, k_joint, k_drv, true);
        for(int edge = 0; edge < 2; ++edge)
        {
            const auto  r  = scene_rest(scene_data, edge);
            const auto  lb = edge_l_body(edge);
            const auto  rb = edge_r_body(edge);

            worst = std::max(worst, rel_err(e_joint[edge], pj_E(k_joint, r, q_body[lb], q_body[rb])));
            worst_drv = std::max(
                worst_drv,
                rel_err(e_drv[edge], dpj_E(k_drv, r, q_body[lb], q_body[rb], scene_data.aim)));
            const double x = abs_distance(r, q_body[lb], q_body[rb]);
            worst_lim      = std::max(
                worst_lim,
                rel_err(e_lim[edge],
                        penalty_ref(x, scene_data.low, scene_data.high, scene_data.lim_str).E));
        }
        std::printf("[t63b2] scene energy kernels: joint %.8e/%.8e, driving %.8e/%.8e, "
                    "limit %.8e/%.8e; worst rel %.3e / %.3e / %.3e\n",
                    e_joint[0], e_joint[1], e_drv[0], e_drv[1], e_lim[0], e_lim[1], worst,
                    worst_drv, worst_lim);
        T63B2_CHECK(c, worst <= 1e-13);
        T63B2_CHECK(c, worst_drv <= 1e-13);
        T63B2_CHECK(c, worst_lim <= 1e-13);

        // the assembled *cross-body* keys are an absolute statement about the joints
        // (the same-body ones carry the ARAP body term, §7.25 rule 16)
        const auto run = run_assembly(*T63B2Probe::abd_linear);
        T63B2_CHECK(c, run.dof_count == 24u);
        T63B2_CHECK(c, run.block_count == 2u + 18u);

        // the same scene *without* the driving instance: its `driving/is_constrained`
        // keeps the frontend's 0, so it reports the same extents but contributes
        // exactly zero -- differencing the two assemblies isolates the driving
        {
            Engine engine_c{"luisa", workspace_path("uipc_luisa_abd_joint_b_nodriving")};
            World  world_c{engine_c};
            Scene  scene_c = scene_data.build("abd_joint_b_nodriving", true, false);
            world_c.init(scene_c);
            REQUIRE(world_c.is_valid());
            REQUIRE(T63B2Probe::abd_linear != nullptr);
            REQUIRE(T63B2Probe::abd_linear->is_valid());
            prepare_dof_layout(*T63B2Probe::abd_linear);
            world_c.advance();
            REQUIRE(world_c.is_valid());
            world_c.sync();
            const auto run_c          = run_assembly(*T63B2Probe::abd_linear);
            const auto driving_blocks = sub_blocks(run.blocks, run_c.blocks);
            const BlockMap expected_driving = scene_expected_blocks(
                scene_data, q_body, body_of, 0.0, k_drv, true, true);
            double worst_d    = 0.0;
            int    compared_d = 0;
            std::printf("[t63b2] scene: expected_driving %zu keys, produced %zu keys\n",
                        expected_driving.size(), driving_blocks.size());
            for(const auto& [key, blk] : expected_driving)
            {
                auto it = driving_blocks.find(key);
                std::printf("[t63b2]   key (%3d,%3d) exp |%.6e| got |%.6e|\n",
                            key.first,
                            key.second,
                            max_abs(blk),
                            it == driving_blocks.end() ? -1.0 : max_abs(it->second));
                if(it == driving_blocks.end())
                    continue;
                ++compared_d;
                worst_d = std::max(worst_d, dev(it->second, blk));
            }
            std::printf("[t63b2] scene: the driving's *differenced* assembled blocks: "
                        "worst dev %.3e (%d keys)\n",
                        worst_d, compared_d);

            // the same isolation for the *gradient*: does the driving's doublet run
            // reach the assembled `b` at all?
            Eigen::VectorXd expected_dg = Eigen::VectorXd::Zero(24);
            for(int edge = 0; edge < 2; ++edge)
            {
                const auto r  = scene_rest(scene_data, edge);
                const auto lb = edge_l_body(edge);
                const auto rb = edge_r_body(edge);
                const Dense24v G =
                    dpj_G(k_drv, r, q_body[lb], q_body[rb], scene_data.aim);
                const int bi = body_of[lb];
                const int bj = body_of[rb];
                expected_dg.segment<12>(static_cast<Eigen::Index>(bi) * 12) +=
                    G.segment<12>(0);
                expected_dg.segment<12>(static_cast<Eigen::Index>(bj) * 12) +=
                    G.segment<12>(12);
            }
            const Eigen::VectorXd produced_dg = run.gradient - run_c.gradient;
            double                worst_dg    = 0.0;
            for(Eigen::Index k = 0; k < expected_dg.size(); ++k)
                worst_dg = std::max(worst_dg, rel_err(produced_dg(k), expected_dg(k)));
            std::printf("[t63b2] scene: the driving's *differenced* gradient: worst rel "
                        "%.3e (|b_driving| %.6e, |produced| %.6e)\n",
                        worst_dg, expected_dg.norm(), produced_dg.norm());
            T63B2_CHECK(c, expected_dg.norm() > 1.0);
            T63B2_CHECK(c, worst_dg <= 1e-10);
            T63B2_CHECK(c, compared_d > 0);
            T63B2_CHECK(c, worst_d <= 1e-8);
        }
        double worst_h  = 0.0;
        int    compared = 0;
        for(const auto& [key, blk] : expected)
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
        std::printf("[t63b2] scene assembled joint blocks (cross-body): worst "
                    "|dev-hand-derived| %.3e (%d keys)\n",
                    worst_h, compared);
        T63B2_CHECK(c, compared == 16);
        T63B2_CHECK(c, worst_h <= 1e-8);
    }

    std::printf("[t63b2] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}


// ===========================================================================
//  6) device == host for every generic routine this batch ships
//
//     The kernels are file-local (their cached shaders live in anonymous
//     namespaces), so the device round dispatches a *test* kernel that calls the
//     *same* generic routines with `S = lcrt::Var<double>` and compares against the
//     host's `S = double`.  The 24-vectors / 24x24 matrices are stored as 12-vectors /
//     12x12 blocks (there is no registered 24x24 element type in the port).
// ===========================================================================
namespace
{
/// One thread's whole output bag, packed sequentially into a flat `double` buffer.
/// NOTE: a 24x24 Hessian is **576** doubles (the test writes and reads the full
/// matrix, column by column), and the stride must match the writer *exactly* — a
/// 144-entry slot silently shifts every later field (measured: a `dev` of exactly
/// 1.0 on the last field, the classic layout-mismatch signature).
constexpr int kRoundStride = 1 + 24 + 576          // prismatic E, G, H (24x24)
                             + 1 + 24 + 576        // driving E, G, H
                             + 1 + 1 + 24          // distance, absolute distance, dx_dq
                             + 1 + 24 + 576        // limit E, G, H
                             + 3                   // the three penalty scalars
                             + 3 + 3 + 12 + 12;    // t_i, t_j, F_i, F_j

auto& t63b2_round_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,  // out (flat)
                                                lcrt::Buffer<gg::LVector6>,  // rest_c
                                                lcrt::Buffer<gg::LVector6>,  // rest_t
                                                lcrt::Buffer<gg::LVector6>,  // rest_n
                                                lcrt::Buffer<gg::LVector6>,  // rest_b
                                                lcrt::Buffer<gg::LVector12>,  // qi
                                                lcrt::Buffer<gg::LVector12>,  // qj
                                                lcrt::Buffer<double>,         // kappas
                                                lcrt::Buffer<double>,         // d_tidles
                                                lcrt::Buffer<double>,         // x (abs dist)
                                                lcrt::Buffer<double>,         // lower
                                                lcrt::Buffer<double>,         // upper
                                                lcrt::Buffer<double>,         // strength
                                                lcrt::Buffer<double>,         // f (ext force)
                                                uint32_t>(                     // n
        [](lcrt::BufferVar<double>      out,
           lcrt::BufferVar<gg::LVector6> rest_c,
           lcrt::BufferVar<gg::LVector6> rest_t,
           lcrt::BufferVar<gg::LVector6> rest_n,
           lcrt::BufferVar<gg::LVector6> rest_b,
           lcrt::BufferVar<gg::LVector12> qi_buf,
           lcrt::BufferVar<gg::LVector12> qj_buf,
           lcrt::BufferVar<double>       kappas,
           lcrt::BufferVar<double>       d_tidles,
           lcrt::BufferVar<double>       xs,
           lcrt::BufferVar<double>       lowers,
           lcrt::BufferVar<double>       uppers,
           lcrt::BufferVar<double>       strengths,
           lcrt::BufferVar<double>       fs,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto kappa    = kappas->read(i);
                auto d_tidle  = d_tidles->read(i);
                auto x_val    = xs->read(i);
                auto low      = lowers->read(i);
                auto high     = uppers->read(i);
                auto strength = strengths->read(i);
                auto f        = fs->read(i);

                auto qi = gg::load_vec12(qi_buf->read(i));
                auto qj = gg::load_vec12(qj_buf->read(i));

                gg::abd_prismatic_joint_detail::PrismaticJointRestT<lcrt::Var<double>> pr{
                    .rest_c = gg::load_vec6(rest_c->read(i)),
                    .rest_t = gg::load_vec6(rest_t->read(i)),
                    .rest_n = gg::load_vec6(rest_n->read(i)),
                    .rest_b = gg::load_vec6(rest_b->read(i))};
                gg::abd_driving_prismatic_joint_detail::DrivingPrismaticJointRestT<
                    lcrt::Var<double>>
                    dr{.rest_c = gg::load_vec6(rest_c->read(i)),
                       .rest_t = gg::load_vec6(rest_t->read(i))};

                int k = 0;
                auto put = [&](auto&& v) { out->write(i * kRoundStride + k++, v); };
                auto putv = [&](const auto& v)
                {
                    for(int q = 0; q < v.dim; ++q)
                        out->write(i * kRoundStride + k++, v(q));
                };

                // ---- the prismatic joint -------------------------------------
                put(gg::abd_prismatic_joint_detail::body_energy<lcrt::Var<double>>(kappa, pr,
                                                                                   qi, qj));
                auto G = gg::abd_prismatic_joint_detail::body_gradient<lcrt::Var<double>>(
                    kappa, pr, qi, qj);
                putv(G);
                auto H = gg::abd_prismatic_joint_detail::body_hessian<lcrt::Var<double>>(
                    kappa, pr, qi, qj);
                for(int cidx = 0; cidx < 24; ++cidx)
                    for(int ridx = 0; ridx < 24; ++ridx)
                        out->write(i * kRoundStride + k++, H(ridx, cidx));

                // ---- the driving joint ---------------------------------------
                put(gg::abd_driving_prismatic_joint_detail::body_energy<lcrt::Var<double>>(
                    kappa, dr, qi, qj, d_tidle));
                auto Gd = gg::abd_driving_prismatic_joint_detail::body_gradient<
                    lcrt::Var<double>>(kappa, dr, qi, qj, d_tidle);
                putv(Gd);
                auto Hd = gg::abd_driving_prismatic_joint_detail::body_hessian<
                    lcrt::Var<double>>(kappa, dr, qi, qj, d_tidle);
                for(int cidx = 0; cidx < 24; ++cidx)
                    for(int ridx = 0; ridx < 24; ++ridx)
                        out->write(i * kRoundStride + k++, Hd(ridx, cidx));

                // ---- the distance chain + the limit --------------------------
                auto C_bar = gg::load_vec6(rest_c->read(i));
                auto T_bar = gg::load_vec6(rest_t->read(i));

                put(gg::abd_driving_prismatic_joint_detail::compute_absolute_distance<
                    lcrt::Var<double>>(C_bar, T_bar, qi, qj));
                put(x_val);  // (the caller's x, checked against the same oracle)
                auto dx_dq = gg::abd_driving_prismatic_joint_detail::
                    compute_absolute_distance_derivative<lcrt::Var<double>>(
                        C_bar, T_bar, qi, qj);
                putv(dx_dq);

                // the penalty writes into *variables*, so it is evaluated into
                // temporaries and then emitted (the same statements, in the same order
                // as `joint_limit::eval_penalty`'s body)
                lcrt::Var<double> limE{0.0}, limdE{0.0}, limd2E{0.0};
                gg::joint_limit::eval_penalty<lcrt::Var<double>>(x_val, low, high, strength, limE,
                                                           limdE, limd2E);
                put(limE);
                auto GL = gg::abd_prismatic_joint_limit_detail::gradient<lcrt::Var<double>>(limdE,
                                                                                     dx_dq);
                putv(GL);
                auto HL = gg::abd_prismatic_joint_limit_detail::hessian<lcrt::Var<double>>(
                    limd2E, limdE, dx_dq, C_bar, T_bar, qi, qj);
                for(int cidx = 0; cidx < 24; ++cidx)
                    for(int ridx = 0; ridx < 24; ++ridx)
                        out->write(i * kRoundStride + k++, HL(ridx, cidx));

                // ---- the three penalty scalars ------------------------------
                put(gg::joint_limit::eval_penalty_energy<lcrt::Var<double>>(x_val, low, high,
                                                                     strength));
                lcrt::Var<double> dE_only{0.0}, d2E_only{0.0};
                gg::joint_limit::eval_penalty_derivatives<lcrt::Var<double>>(
                    x_val, low, high, strength, dE_only, d2E_only);
                put(dE_only);
                put(d2E_only);

                // ---- the external-force pair --------------------------------
                lps::LVec<lcrt::Var<double>, 3> t_i, t_j;
                gg::abd_prismatic_joint_external_force_detail::symmetrized_axes<lcrt::Var<double>>(
                    t_i, t_j,
                    gg::ABDJacobiT<lcrt::Var<double>>{.x_bar = T_bar.template segment<3>(0)},
                    gg::ABDJacobiT<lcrt::Var<double>>{.x_bar = T_bar.template segment<3>(3)},
                    qi, qj);
                putv(t_i);
                putv(t_j);
                lps::LVec<lcrt::Var<double>, 12> F_i, F_j;
                gg::abd_prismatic_joint_external_force_detail::force_pair<lcrt::Var<double>>(
                    f, t_i, t_j, F_i, F_j);
                putv(F_i);
                putv(F_j);
            };
        });
    return k;
}
}  // namespace

TEST_CASE("abd_joint_b_device_vs_host", "[luisa][abd_joint_b]")
{
    test_runtime();
    CaseScope scope{"abd_joint_b_device_vs_host"};
    auto&     c = scope.counter;

    constexpr int N = 16;

    std::vector<Vector12>  qis, qjs;
    std::vector<Vector6>   rc, rt, rn, rb;
    std::vector<double>    kappas, d_tidles, xs, lowers, uppers, strengths, fs;
    std::vector<PrismRest> rests;
    for(int i = 0; i < N; ++i)
    {
        PrismRest r;
        r.ci = rand_vec(-0.3, 0.3);
        r.cj = rand_vec(-0.3, 0.3);
        const Matrix3x3 Ai = random_admissible_A();
        const Matrix3x3 Aj = random_admissible_A();
        r.set_unit(Ai, Aj, rand_vec(-1.0, 1.0));
        const Vector12 qi = q_of(Ai, rand_vec(-0.5, 0.5));
        const Vector12 qj = q_of(Aj, rand_vec(-0.5, 0.5));
        qis.push_back(qi);
        qjs.push_back(qj);
        kappas.push_back(rand_uniform(0.5, 2.0));
        d_tidles.push_back(rand_uniform(-0.4, 0.4));
        // the penalty input: half inside / half outside the interval, and two
        // zero-width cases so both the `w == 0` and the normalised branch are rounded
        const double lo = rand_uniform(-0.3, 0.1);
        const double hi = (i % 5 == 0) ? lo : lo + rand_uniform(0.05, 0.8);
        lowers.push_back(lo);
        uppers.push_back(hi);
        strengths.push_back(rand_uniform(1.0, 50.0));
        xs.push_back(rand_uniform(lo - 0.5, hi + 0.5));
        fs.push_back(rand_uniform(-20.0, 20.0));
        rests.push_back(r);
        rc.push_back(vec6(r.ci, r.cj));
        rt.push_back(vec6(r.ti, r.tj));
        rn.push_back(vec6(r.ni, r.nj));
        rb.push_back(vec6(r.bi, r.bj));
    }

    auto qd  = upload_as<gg::LVector12>(qis, [](const Vector12& q) { return gg::to_luisa(q); });
    auto qjd = upload_as<gg::LVector12>(qjs, [](const Vector12& q) { return gg::to_luisa(q); });
    auto rcd = upload_as<gg::LVector6>(rc, [](const Vector6& v) { return gg::to_luisa(v); });
    auto rtd = upload_as<gg::LVector6>(rt, [](const Vector6& v) { return gg::to_luisa(v); });
    auto rnd = upload_as<gg::LVector6>(rn, [](const Vector6& v) { return gg::to_luisa(v); });
    auto rbd = upload_as<gg::LVector6>(rb, [](const Vector6& v) { return gg::to_luisa(v); });
    auto kd  = upload(kappas);
    auto dd  = upload(d_tidles);
    auto xd  = upload(xs);
    auto lod = upload(lowers);
    auto hid = upload(uppers);
    auto sd  = upload(strengths);
    auto fd  = upload(fs);

    lud::LDeviceBuffer<double> out(static_cast<SizeT>(N) * kRoundStride);

    auto& k = t63b2_round_kernel();
    gg::frame_stream().record(k(out.view(), rcd.view(), rtd.view(), rnd.view(), rbd.view(),
                                qd.view(), qjd.view(), kd.view(), dd.view(), xd.view(),
                                lod.view(), hid.view(), sd.view(), fd.view(),
                                static_cast<uint32_t>(N))
                                .dispatch(N));
    gg::frame_stream().synchronize();

    const auto flat = read_device(out.cview());
    REQUIRE(flat.size() == static_cast<std::size_t>(N) * kRoundStride);

    double worst_prism_e = 0.0, worst_prism_g = 0.0, worst_prism_h = 0.0;
    double worst_drv_e = 0.0, worst_drv_g = 0.0, worst_drv_h = 0.0;
    double worst_dist = 0.0, worst_dxdq = 0.0;
    double worst_lim_e = 0.0, worst_lim_g = 0.0, worst_lim_h = 0.0;
    double worst_pen = 0.0, worst_pen2 = 0.0;
    double worst_ext = 0.0;
    for(int i = 0; i < N; ++i)
    {
        const double* base = flat.data() + static_cast<std::size_t>(i) * kRoundStride;
        int          kk    = 0;
        auto         get   = [&]() { return base[kk++]; };
        auto         getv  = [&](int n)
        {
            Dense24v v = Dense24v::Zero();
            for(int q = 0; q < n; ++q)
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
        auto getH = [&]()
        {
            Dense24 H;
            for(int cidx = 0; cidx < 24; ++cidx)
                for(int ridx = 0; ridx < 24; ++ridx)
                    H(ridx, cidx) = base[kk++];
            return H;
        };

        const auto& r  = rests[i];
        const auto& qi = qis[i];
        const auto& qj = qjs[i];
        const double kappa = kappas[i];

        // ---- the prismatic joint ---------------------------------------------
        {
            const auto loc = prism_local(kappa, r, qi, qj);
            worst_prism_e = std::max(worst_prism_e, rel_err(get(), loc.E));
            worst_prism_g = std::max(worst_prism_g, dev(getv(24), loc.G));
            worst_prism_h = std::max(worst_prism_h, dev(getH(), loc.H));
        }
        // ---- the driving joint ----------------------------------------------
        {
            const Dense9v F01q = F01q_of(r, qi, qj);
            const auto    J01  = J01q_of(r);
                // the driving's *shipped* H (the reference clamps `ddE01ddF01`
                // before its fan-out): the ported host routine the kernel mirrors
                const Dense24 Hraw = dpj_H(kappa, r, qi, qj, d_tidles[i]);
            worst_drv_e = std::max(worst_drv_e, rel_err(get(), driving_E01_of(F01q, kappa,
                                                                              d_tidles[i])));
            worst_drv_g = std::max(worst_drv_g,
                                   dev(getv(24),
                                       J01.transpose()
                                           * driving_dE01dF01_of(F01q, kappa, d_tidles[i])));
            worst_drv_h = std::max(worst_drv_h, dev(getH(), Hraw));
        }
        // ---- the distance chain + the limit ---------------------------------
        {
            const Dense9v F01q = F01q_of(r, qi, qj);
            const double  x_port = get();
            const double  x_in   = get();
            worst_dist = std::max(worst_dist, std::abs(x_port - distance_of(F01q)));
            worst_dist = std::max(worst_dist, std::abs(x_in - xs[i]));
            const auto J01   = J01q_of(r);
                // the device's `compute_absolute_distance_derivative` against the
                // *same* routine on the host (a device-vs-host round)
                const auto dxdq = abs_distance_deriv(r, qi, qj);
                // NOTE: the 24 entries of this block must be *consumed* here -- skipping
                // them shifts every later field by 24 and the limit/penalty/force
                // deviations then read O(1) (measured before the fix: limit E 5.8e3,
                // penalty 1.1e4, force 1.5)
                const auto dxdq_dev = getv(24);
                worst_dxdq = std::max(worst_dxdq, dev(dxdq_dev, dxdq));
                T63B2_CHECK(c, dxdq.norm() > 1.0);
            const auto ref = penalty_ref(xs[i], lowers[i], uppers[i], strengths[i]);
            // `limit_local`'s first argument is the *offset* added to the law's own
            // distance, so that its `x` is exactly the value the kernel was fed
            const auto lim = limit_local(xs[i] - distance_of(F01q), lowers[i], uppers[i],
                                         strengths[i], r, qi, qj);
            worst_lim_e      = std::max(worst_lim_e, std::abs(get() - ref.E));
            worst_lim_g      = std::max(worst_lim_g, dev(getv(24), lim.G));
            worst_lim_h      = std::max(worst_lim_h, dev(getH(), lim.H));
            // the three penalty entry points
            worst_pen  = std::max(worst_pen, std::abs(get() - ref.E));
            worst_pen2 = std::max(worst_pen2, std::abs(get() - ref.dE));
            worst_pen2 = std::max(worst_pen2, std::abs(get() - ref.d2E));
        }
        // ---- the external-force pair ----------------------------------------
        {
            const auto ef = ext_force_local(fs[i], r, qi, qj);
            const auto ti = getv3();
            const auto tj = getv3();
            Dense24v   Fi = getv(12);
            Dense24v   Fj = getv(12);
            double     w  = 0.0;
            w             = std::max(w, dev(ti, ef.t_i));
            w             = std::max(w, dev(tj, ef.t_j));
            w             = std::max(w, dev(Dense12v(Fi.segment<12>(0)), ef.F_i));
            w             = std::max(w, dev(Dense12v(Fj.segment<12>(0)), ef.F_j));
            worst_ext     = std::max(worst_ext, w);
        }
    // the whole bag was consumed: a silent stride mismatch is the classic
    // layout trap of this file (its last field deviates by exactly 1.0)
    if(i + 1 == N)
        T63B2_CHECK(c, kk == kRoundStride);
    }

    std::printf("[t63b2] device vs host (16 states): prismatic E/G/H %.3e/%.3e/%.3e, driving "
                "E/G/H %.3e/%.3e/%.3e, distance/dx_dq %.3e/%.3e, limit E/G/H %.3e/%.3e/%.3e, "
                "penalty %.3e/%.3e, ext force %.3e\n",
                worst_prism_e, worst_prism_g, worst_prism_h, worst_drv_e, worst_drv_g, worst_drv_h,
                worst_dist, worst_dxdq, worst_lim_e, worst_lim_g, worst_lim_h, worst_pen,
                worst_pen2, worst_ext);

    // the generic routines are the same statements, but the generated laws use
    // `lps::pow` (lcrt::pow on the DSL) and `make_spd`'s sweeps differ in
    // associativity: assert a per-routine bound and *record* the measurement
    // (§7.17 rule 34 / §7.24 rule 15) -- this round is NOT bitwise.
    T63B2_CHECK(c, (worst_prism_e <= 1e-12 && worst_prism_g <= 1e-12 && worst_prism_h <= 1e-12));
    T63B2_CHECK(c, (worst_drv_e <= 1e-12 && worst_drv_g <= 1e-12 && worst_drv_h <= 1e-12));
    T63B2_CHECK(c, (worst_dist <= 1e-12 && worst_dxdq <= 1e-12));
    T63B2_CHECK(c, (worst_lim_e <= 1e-12 && worst_lim_g <= 1e-12 && worst_lim_h <= 1e-12));
    T63B2_CHECK(c, (worst_pen <= 1e-12 && worst_pen2 <= 1e-12));
    T63B2_CHECK(c, worst_ext <= 1e-12);

    std::printf("[t63b2] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  7) uid ownership of 20/21/667/669, *proved* (plus the 667 sharing)
// ===========================================================================
TEST_CASE("abd_joint_b_uid_ownership", "[luisa][abd_joint_b]")
{
    test_runtime();
    CaseScope scope{"abd_joint_b_uid_ownership"};
    auto&     c = scope.counter;

    PrismScene scene_data;
    Scene      scene = scene_data.build("abd_joint_b_uid", true);
    Engine     engine{"luisa", workspace_path("uipc_luisa_abd_joint_b_uid")};
    World      world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    const bool joint_ok = T63B2Probe::joint != nullptr && T63B2Probe::joint->is_valid()
                          && T63B2Probe::joint->h_body_ids.size() == 2u;
    const bool drv_ok = T63B2Probe::driving != nullptr && T63B2Probe::driving->is_valid()
                        && T63B2Probe::driving->h_is_constrained.size() == 2u;
    const bool lim_ok = T63B2Probe::limit != nullptr && T63B2Probe::limit->is_valid()
                        && T63B2Probe::limit->h_ref_qs.size() == 2u;
    const bool econst_ok = T63B2Probe::ext_constraint != nullptr
                           && T63B2Probe::ext_constraint->is_valid()
                           && T63B2Probe::ext_constraint->h_forces.size() == 2u;
    const bool erep_ok = T63B2Probe::ext_reporter != nullptr
                         && T63B2Probe::ext_reporter->is_valid();

    std::printf("[t63b2] uid ownership: uid20 = %s (%zu joints), uid21 = %s (%zu), "
                "uid669 = %s (%zu), uid667 constraint = %s (%zu), uid667 reporter = %s\n",
                joint_ok ? "uipc::backend::luisa::AffineBodyPrismaticJoint" : "ANOTHER SYSTEM",
                T63B2Probe::joint ? T63B2Probe::joint->h_body_ids.size() : 0u,
                drv_ok ? "uipc::backend::luisa::AffineBodyDrivingPrismaticJoint"
                       : "ANOTHER SYSTEM",
                T63B2Probe::driving ? T63B2Probe::driving->h_is_constrained.size() : 0u,
                lim_ok ? "uipc::backend::luisa::AffineBodyPrismaticJointLimit"
                       : "ANOTHER SYSTEM",
                T63B2Probe::limit ? T63B2Probe::limit->h_ref_qs.size() : 0u,
                econst_ok
                    ? "uipc::backend::luisa::AffineBodyPrismaticJointExternalForceConstraint"
                    : "ANOTHER SYSTEM",
                T63B2Probe::ext_constraint ? T63B2Probe::ext_constraint->h_forces.size() : 0u,
                erep_ok ? "uipc::backend::luisa::AffineBodyPrismaticJointExternalForce"
                        : "ANOTHER SYSTEM");
    REQUIRE(joint_ok);
    REQUIRE(drv_ok);
    REQUIRE(lim_ok);
    REQUIRE(econst_ok);
    REQUIRE(erep_ok);

    // the declared uids, as the *registry* sees them (the 667 sharing is between two
    // different registries — the constraint's `constraint_uids` and the reporter's
    // `ExternalForceReporter` uid — exactly as the reference spells it)
    T63B2_CHECK(c, T63B2Probe::joint->get_uid() == 20ull);
    T63B2_CHECK(c, T63B2Probe::driving->get_uid() == 21ull);
    T63B2_CHECK(c, T63B2Probe::ext_constraint->get_uid() == 667ull);
    T63B2_CHECK(c, T63B2Probe::ext_reporter->get_uid() == 667ull);
    T63B2_CHECK(c, T63B2Probe::limit->get_uid() == 669ull);
    // ... and the base-class `uid()` accessor agrees (that one resolves through the
    // *table*, so it is the ownership statement)
    T63B2_CHECK(c, T63B2Probe::joint->uid() == 20ull);
    T63B2_CHECK(c, T63B2Probe::driving->uid() == 21ull);
    T63B2_CHECK(c, T63B2Probe::ext_constraint->uid() == 667ull);
    T63B2_CHECK(c, T63B2Probe::ext_reporter->uid() == 667ull);
    T63B2_CHECK(c, T63B2Probe::limit->uid() == 669ull);

    // the *cross-reference* the limit uses to find the joint geometry is the joint's
    // own uid, and the frontend keeps the base uid at 20 (669 is an *extra* uid)
    T63B2_CHECK(c, gg::AffineBodyPrismaticJointLimit::JointUID == 20ull);
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
            T63B2_CHECK(c, view(*base_uid)[0] == 20ull);
            auto extras = sc->meta().find<VectorXu64>(builtin::extra_constitution_uids);
            REQUIRE(extras != nullptr);
            bool has_669 = false;
            for(auto uid : view(*extras)[0])
                if(uid == 669ull)
                    has_669 = true;
            T63B2_CHECK(c, has_669);
            auto cuids = sc->meta().find<VectorXu64>(builtin::constraint_uids);
            REQUIRE(cuids != nullptr);
            bool has_drv = false, has_ext = false;
            for(auto uid : view(*cuids)[0])
            {
                if(uid == 21ull)
                    has_drv = true;
                if(uid == 667ull)
                    has_ext = true;
            }
            T63B2_CHECK(c, has_drv);
            T63B2_CHECK(c, has_ext);
        }
        std::printf("[t63b2] uid ownership: %d limit-bearing geometry (base uid 20, extra 669, "
                    "constraints {21, 667})\n",
                    found);
        T63B2_CHECK(c, found == 1);
    }

    std::printf("[t63b2] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  8) per-case assertion counts
// ===========================================================================
TEST_CASE("abd_joint_b_summary", "[luisa][abd_joint_b]")
{
    int total = 0;
    for(const auto& [name, checks] : g_case_counts)
    {
        std::printf("[t63b2] case %-42s %5d checks\n", name.c_str(), checks);
        total += checks;
        CHECK(checks > 0);
    }
    std::printf("[t63b2] summary: %d cases, %d checks\n",
                static_cast<int>(g_case_counts.size()), total);
}
