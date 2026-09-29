/********************************************************************
 * @file luisa_abd_joint_a_test.cpp
 * @brief T6.3b1 — the ABD joints of the LuisaCompute backend, batch 1: the
 *        **fixed joint** (uid 25, `uipc::backend::luisa::AffineBodyFixedJoint`) and
 *        the **spherical joint** (uid 26, `uipc::backend::luisa::AffineBodySphericalJoint`).
 *
 * Every oracle here is derived independently *in this file*: the reference backend is
 * CUDA-only and cannot be host-included, so nothing is copied from it.  The oracles are
 *  (a) hand-derived closed forms of the two joint laws, written out at the assertion.
 *      With the pair dof vector `q = [qi ; qj]` (12 dofs each, `A(i,j) = q(3+3i+j)`,
 *      `§7.21 rule 6`), the two attachment points in each body's local frame
 *      `ci_bar`/`cj_bar` and the two body-local frame triads `(t,n,b)`/`(t_j,n_j,b_j)`:
 *
 *        Ft = A_i ci_bar + ti - (A_j cj_bar + tj)          (the ball constraint)
 *        Fr = [A_i t - A_j t_j ; A_i n - A_j n_j ; A_i b - A_j b_j]
 *        E  = 1/2 kappa (||Ft||^2 + ||Fr||^2), and the spherical joint is `Ft` alone.
 *
 *      Both laws are *quadratic* in the state with constant Jacobians.  Writing them as
 *      `Ft = J_t q + const`, `Fr = J_r q + const` gives the whole gradient/Hessian in
 *      closed form:
 *
 *        J_t(k, k) = 1, J_t(k, 12+k) = -1
 *        J_t(k, 3+3k+m) = ci_bar(m), J_t(k, 12+3k+m) = -cj_bar(m)     (3x24)
 *        J_r(3a+k, 3+3k+m) = (d_a)_m, J_r(3a+k, 12+3k+m) = -(e_a)_m   (9x24)
 *
 *      (`d_a` = t/n/b, `e_a` = t_j/n_j/b_j; the row index `3a+k` is the group/component
 *      pair and the *column* is the `A`-row `k` — that is what makes the generated
 *      `JrT_Gr` row `R(3) = G(0) t(0) + G(3) n(0) + G(6) b(0)` come out right)
 *        G = kappa (J_t^T Ft + J_r^T Fr)
 *        H = kappa (J_t^T J_t + J_r^T J_r)   -- *exactly*, the law is a polynomial
 *      and the reference's `make_spd` clamp on the two *element* Hessians
 *      (`ddEdFt = kappa I_3`, `ddEdFr = kappa I_9`) makes the shipped one
 *      `max(kappa, 0) * (J_t^T J_t + J_r^T J_r)`.
 *      The two `J`s are built here from `ci_bar`/`cj_bar`/the triads alone — no
 *      generated statement is transcribed into the oracle.  They were also checked
 *      against a python re-parse of the reference `.inl` (the §7.17 rule 30c technique,
 *      `scratch/t63b1_oracle.py`) before this file existed: E/G/H 2e-16/0e0/0e0 and the
 *      two fan-outs 1.5e-10/2.5e-10 over 20 random states.
 *  (b) central finite differences over the **24 pair dofs** (h = 1e-6 plus a measured
 *      sweep) of the *shipped* generic routines: gradient vs E, Hessian vs gradient,
 *      the Hessian's symmetry, and the rule-34 basis question — unlike the ABD ARAP
 *      (cookbook §7.24 rule 5) the joints' generated `J^T H J` fan-outs are *in the same
 *      dof order* as their `dEdF` siblings, so the **plain** FD comparison is the
 *      agreement and no transposition is needed (the case asserts that explicitly).
 *  (c) degenerate/rigid inputs pinned from the source: the joint laws contain no
 *      division and no SVD at all, so every one of E/G/H is finite in *every* state
 *      (including a reflection), `E == 0.0` *exactly* at a coincident rest pair (unlike
 *      the SVD-derived zeros of T6.3a §7.24 rule 14), a zero `kappa` zeroes everything,
 *      and a *negative* `kappa` is what makes the reference's `make_spd` observable
 *      (the shipped H collapses to 0 while G stays `kappa ...`).
 *  (d) the fan-out into `ABDLinearSubsystem`'s assembly, isolated by *differencing* two
 *      real assemblies (the same 2-body scene with and without the joint geometries):
 *      the 12x12 `(body_i, body_j)` blocks, the 16-way 3x3 unpacking, the
 *      `zero_out_lower` of the two diagonal blocks, the reference's `upper_LR`
 *      off-diagonal placement (a *descending* pair is included on purpose), and the
 *      gradient scatter at the bodies' dof offsets.  The joints' *energy kernels* are
 *      driven directly through a test-built `InterAffineBodyConstitutionManager::Impl`
 *      (the T6.3a batch could not reach its energy hook at all; here it can).
 *  (e) a real scene (one fixed joint and one spherical joint between two ABD bodies),
 *      one `advance()`, the frontend state accessor, the BDF1 prediction by hand, and
 *      the per-joint *host* bookkeeping (`h_body_ids` / `h_rest_*` / `h_strength_ratios`)
 *      against the hand-computed body-local rest frames.
 *  (f) device == host for the six ported generic routines (S = double vs
 *      S = lcrt::Var<double>) — *not* bitwise (the generated laws use `lps::pow`, which
 *      lowers to `lcrt::pow` on the DSL), so a per-routine bound is asserted and the
 *      measurement recorded (§7.24 rule 15).
 *  (g) uid ownership for 25 and 26, *proved* (not assumed): the real constitutions must
 *      own the geometries their uids name, so a mock that stole the group would fail the
 *      REQUIRE with the owner line printed above it.
 *
 * Coverage map:
 *  1) closed forms (both joints, the scaling laws, the clamp) .. abd_joint_a_closed_forms
 *  2) central FD (g/H/symmetry) + the basis question ......... abd_joint_a_finite_difference
 *  3) rest/reflection/zero- and negative-kappa ................ abd_joint_a_degenerate_inputs
 *  4) the assembly fan-out (COO, upper_LR, gradient) + the
 *     energy kernels ........................................ abd_joint_a_assembly
 *  5) a real scene, one advance(), the BDF1 prediction ....... abd_joint_a_scene
 *  6) device == host for the generic routines ................ abd_joint_a_device_vs_host
 *  7) uid ownership of 25 and 26 ............................. abd_joint_a_uid_ownership
 *  8) per-case assertion counts .............................. abd_joint_a_summary
 *
 * UID OWNERSHIP.  Both joints are plain `InterAffineBodyConstitution`s, and
 * `InterAffineBodyConstitutionManager::Impl::init` sorts the registered constitutions by
 * uid (`std::ranges::sort`, not stable) and maps `uid -> index` in a loop, so with two
 * equal-uid entries the **last** one owns the geometry group; the registration order is
 * `SimEngine::build_systems()`'s demangled-type-name order (§7.23 rule 7).  T6.1b's
 * test-local `t61b::T61bInterConstitution` used to claim uid 25 ('t' < 'u', so the real
 * `uipc::backend::luisa::AffineBodyFixedJoint` wins) — it was **moved to uid 28** in this
 * batch so that the *meaning* of that file's scene (a joint geometry handed to the mock,
 * whose hooks write nothing, so the assembly's constitution term stays exactly zero)
 * survives; see the note in `luisa_abd_linear_test.cpp`.  This file does not assume the
 * outcome: the `uid_ownership`, `assembly` and `scene` cases print the owner they
 * measured and REQUIRE it.
 ********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_fixed_joint.h>
#include <uipc/constitution/affine_body_spherical_joint.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

#include <affine_body/type_define.h>
#include <affine_body/utils.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/constitutions/affine_body_fixed_joint.h>
#include <affine_body/constitutions/affine_body_fixed_joint_function.h>
#include <affine_body/constitutions/affine_body_spherical_joint.h>
#include <affine_body/constitutions/affine_body_spherical_joint_function.h>
#include <linear_system/global_linear_system.h>
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
using Dense6v  = Eigen::Matrix<double, 6, 1>;
using Dense9   = Eigen::Matrix<double, 9, 9>;
using Dense9v  = Eigen::Matrix<double, 9, 1>;
using Dense24  = Eigen::Matrix<double, 24, 24>;
using Dense24v = Eigen::Matrix<double, 24, 1>;
using Dense12  = Eigen::Matrix<double, 12, 12>;
using Dense3x24 = Eigen::Matrix<double, 3, 24>;
using Dense9x24 = Eigen::Matrix<double, 9, 24>;

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
/// `A` (`q(3 + 3i + j) == A(i,j)`, §7.21 rule 6), so `p = A x_bar + t`.
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

Dense9v to_eigen9(const lps::LVec<double, 9>& v)
{
    Dense9v out;
    for(int i = 0; i < 9; ++i)
        out(i) = v(i);
    return out;
}

Dense3v to_eigen3(const lps::LVec<double, 3>& v)
{
    Dense3v out;
    for(int i = 0; i < 3; ++i)
        out(i) = v(i);
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
//
//  `A = R (I + S)` with `S` symmetric and `|S| <= 0.25`, then rescaled so that
//  `sigma_min >= 0.35` and `sigma_max <= 2.5`: well conditioned and far enough from
//  the rest state that the gradient is O(kappa) rather than O(round-off).
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

#define T63B1_CHECK(counter, cond) \
    do \
    { \
        ++(counter).checks; \
        CHECK(cond); \
    } while(false)
}  // namespace

// ===========================================================================
//  the oracle
// ===========================================================================
namespace
{
/// The *rotation* part of a linear map (`R = U V^T` of its SVD) — the independent
/// spelling of what Eigen's `Transform<...,Eigen::Affine>::rotation()` computes
/// (`computeRotationScaling`, a polar decomposition).  The scene's bodies have
/// `det A > 0`, so the SVD's `U V^T` is the proper rotation Eigen returns.
Matrix3x3 polar_rotation(const Matrix3x3& A)
{
    const auto sv = svd_of(A);
    return sv.matrixU() * sv.matrixV().transpose();
}

/// One joint's rest data in the reference's own `[first | second]` layout.
struct JointRest
{
    Vector3 ci_bar = Vector3::Zero();  // rest_cs[0..2]
    Vector3 cj_bar = Vector3::Zero();  // rest_cs[3..5]
    Vector3 ti     = Vector3::Zero();  // rest_ts[0..2]
    Vector3 tj     = Vector3::Zero();  // rest_ts[3..5]
    Vector3 ni     = Vector3::Zero();  // rest_ns[0..2]
    Vector3 nj     = Vector3::Zero();  // rest_ns[3..5]
    Vector3 bi     = Vector3::Zero();  // rest_bs[0..2]
    Vector3 bj     = Vector3::Zero();  // rest_bs[3..5]

    /// The body-local frame triads, straight from the reference's
    /// `rest_x = LR.inverse() * e_x` (`do_init`) with the two *frame sources* `Li`/`Lj`
    /// (i.e. the matrices whose inverse maps the world axes into body space).
    void set_frames_inv(const Matrix3x3& Li, const Matrix3x3& Lj)
    {
        ti = Li * Vector3(1, 0, 0);
        ni = Li * Vector3(0, 1, 0);
        bi = Li * Vector3(0, 0, 1);
        tj = Lj * Vector3(1, 0, 0);
        nj = Lj * Vector3(0, 1, 0);
        bj = Lj * Vector3(0, 0, 1);
    }

    /// The free-input spelling used by the unit cases (the frames are the *inputs*
    /// there, so any invertible `A` is a legitimate choice): `L = A`.
    void set_frames(const Matrix3x3& A_i, const Matrix3x3& A_j)
    {
        set_frames_inv(A_i.inverse(), A_j.inverse());
    }

    /// The **scene** spelling (FINDING): the reference's `do_init` calls
    /// `LT.rotation()` on an `Eigen::Transform<Float,3,Eigen::Affine>`, and Eigen
    /// implements that as `computeRotationScaling()` — a *polar decomposition* — so the
    /// frame source is the *rotation* part of the body's initial transform, not its
    /// linear part.  A *stretched* initial state therefore does not give a zero
    /// rotation term at the scene's own state.
    void set_frames_polar(const Matrix3x3& A_i, const Matrix3x3& A_j)
    {
        set_frames_inv(polar_rotation(A_i).inverse(), polar_rotation(A_j).inverse());
    }

    [[nodiscard]] lps::LVec<double, 6> rest_c() const { return lv6(ci_bar, cj_bar); }
    [[nodiscard]] lps::LVec<double, 6> rest_t() const { return lv6(ti, tj); }
    [[nodiscard]] lps::LVec<double, 6> rest_n() const { return lv6(ni, nj); }
    [[nodiscard]] lps::LVec<double, 6> rest_b() const { return lv6(bi, bj); }
};

/// `Ft = A_i ci_bar + ti - (A_j cj_bar + tj)`.
Dense3v Ft_of(const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    return A_of(qi) * r.ci_bar + t_of(qi) - A_of(qj) * r.cj_bar - t_of(qj);
}

/// `Ft` (spelled separately so the degenerate-input pins can call it by hand).
Dense3v ft_of_hand(const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    return Ft_of(r, qi, qj);
}

/// `Fr = [A_i t - A_j t_j ; A_i n - A_j n_j ; A_i b - A_j b_j]`.
Dense9v Fr_of(const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    const Matrix3x3 Ai = A_of(qi);
    const Matrix3x3 Aj = A_of(qj);
    Dense9v         out;
    out.segment<3>(0) = Ai * r.ti - Aj * r.tj;
    out.segment<3>(3) = Ai * r.ni - Aj * r.nj;
    out.segment<3>(6) = Ai * r.bi - Aj * r.bj;
    return out;
}

/// `J_t` (3x24), the constant Jacobian of `Ft` (derivation in the file comment).
Dense3x24 jacobian_t(const JointRest& r)
{
    Dense3x24 J = Dense3x24::Zero();
    for(int k = 0; k < 3; ++k)
    {
        // dFt(k)/dti(k) = 1 ; dFt(k)/dtj(k) = -1
        J(k, k)      = 1.0;
        J(k, 12 + k) = -1.0;
        // dFt(k)/dA_i(k, m) = ci_bar(m) ; dFt(k)/dA_j(k, m) = -cj_bar(m).  NOTE the
        // pair layout: `q = [ti(0..2); A_i(3..11); tj(12..14); A_j(15..23)]`, so the
        // *second* body's A-block starts at 15 (12 is `tj`), not at 12.
        for(int m = 0; m < 3; ++m)
        {
            J(k, 3 + 3 * k + m)  = r.ci_bar(m);
            J(k, 15 + 3 * k + m) = -r.cj_bar(m);
        }
    }
    return J;
}

/// `J_r` (9x24), the constant Jacobian of `Fr` (derivation in the file comment).
Dense9x24 jacobian_r(const JointRest& r)
{
    const Vector3 d[3] = {r.ti, r.ni, r.bi};
    const Vector3 e[3] = {r.tj, r.nj, r.bj};
    Dense9x24     J    = Dense9x24::Zero();
    for(int a = 0; a < 3; ++a)  // the group: t/n/b
    {
        for(int k = 0; k < 3; ++k)  // the component AND the `A`-row
        {
            for(int m = 0; m < 3; ++m)
            {
                J(3 * a + k, 3 + 3 * k + m)  = d[a](m);
                J(3 * a + k, 15 + 3 * k + m) = -e[a](m);  // A_j starts at 15 (12 is t_j)
            }
        }
    }
    return J;
}

/// The closed-form joint quantities of the *whole* pair law (both terms).
struct JointLocal
{
    double   E          = 0.0;
    Dense24v G          = Dense24v::Zero();
    Dense24  H          = Dense24::Zero();  // the unclamped exact Hessian
    Dense24  H_shipped  = Dense24::Zero();  // after the reference's make_spd clamps
    Dense3v  Ft;
    Dense9v  Fr;
};

/// The reference's `make_spd(M) = V max(w,0) V^T` (an *exact* clamp; the shipped one
/// is the layer's fixed-sweep Jacobi approximation of it).
template <int N>
Eigen::Matrix<double, N, N> exact_spd(const Eigen::Matrix<double, N, N>& M)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> es(M);
    Eigen::Matrix<double, N, 1> w = es.eigenvalues();
    for(int i = 0; i < N; ++i)
        w(i) = std::max(w(i), 0.0);
    return es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
}

JointLocal joint_local(double kappa, const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    JointLocal out;
    out.Ft = Ft_of(r, qi, qj);
    out.Fr = Fr_of(r, qi, qj);
    const Dense3x24 Jt = jacobian_t(r);
    const Dense9x24 Jr = jacobian_r(r);

    out.E = 0.5 * kappa * (out.Ft.squaredNorm() + out.Fr.squaredNorm());
    out.G = kappa * (Jt.transpose() * out.Ft + Jr.transpose() * out.Fr);
    out.H = kappa * (Jt.transpose() * Jt + Jr.transpose() * Jr);
    // the reference clamps the *element* Hessians (kappa I_3 and kappa I_9) before
    // the fan-outs, which is exactly the >= 0 part of kappa
    out.H_shipped = std::max(kappa, 0.0) * (Jt.transpose() * Jt + Jr.transpose() * Jr);
    return out;
}

/// The spherical joint's law is the fixed joint's translation half.
JointLocal spherical_local(double kappa, const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    JointLocal out;
    out.Ft = Ft_of(r, qi, qj);
    const Dense3x24 Jt = jacobian_t(r);
    out.E            = 0.5 * kappa * out.Ft.squaredNorm();
    out.G            = kappa * (Jt.transpose() * out.Ft);
    out.H            = kappa * (Jt.transpose() * Jt);
    out.H_shipped    = std::max(kappa, 0.0) * (Jt.transpose() * Jt);
    return out;
}

// ---------------------------------------------------------------------------
//  the ported generic routines, called exactly like the two kernels do (host)
// ---------------------------------------------------------------------------
gg::abd_fixed_joint_detail::FixedJointRestT<double> fixed_rest(const JointRest& r)
{
    return gg::abd_fixed_joint_detail::FixedJointRestT<double>{
        .rest_c = r.rest_c(), .rest_t = r.rest_t(), .rest_n = r.rest_n(), .rest_b = r.rest_b()};
}

gg::abd_spherical_joint_detail::SphericalJointRestT<double> spherical_rest(const JointRest& r)
{
    return gg::abd_spherical_joint_detail::SphericalJointRestT<double>{.rest_c = r.rest_c()};
}

double fixed_E_host(double kappa, const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    return gg::abd_fixed_joint_detail::body_energy<double>(
        kappa, fixed_rest(r), gg::to_luisa(qi), gg::to_luisa(qj));
}

Dense24v fixed_G_host(double kappa, const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen24(gg::abd_fixed_joint_detail::body_gradient<double>(
        kappa, fixed_rest(r), gg::to_luisa(qi), gg::to_luisa(qj)));
}

Dense24 fixed_H_host(double kappa, const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen24(gg::abd_fixed_joint_detail::body_hessian<double>(
        kappa, fixed_rest(r), gg::to_luisa(qi), gg::to_luisa(qj)));
}

double spherical_E_host(double kappa, const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    return gg::abd_spherical_joint_detail::body_energy<double>(
        kappa, spherical_rest(r), gg::to_luisa(qi), gg::to_luisa(qj));
}

Dense24v spherical_G_host(double kappa, const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen24(gg::abd_spherical_joint_detail::body_gradient<double>(
        kappa, spherical_rest(r), gg::to_luisa(qi), gg::to_luisa(qj)));
}

Dense24 spherical_H_host(double kappa, const JointRest& r, const Vector12& qi, const Vector12& qj)
{
    return to_eigen24(gg::abd_spherical_joint_detail::body_hessian<double>(
        kappa, spherical_rest(r), gg::to_luisa(qi), gg::to_luisa(qj)));
}

// ---------------------------------------------------------------------------
//  central finite differences over the 24 *pair* dofs
// ---------------------------------------------------------------------------
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
//  the probe: the built ABD systems of one scene
// ===========================================================================
class T63B1Probe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::AffineBodyDynamics*       abd       = nullptr;
    inline static gg::ABDLinearSubsystem*       abd_linear = nullptr;
    inline static gg::GlobalLinearSystem*       gls       = nullptr;
    inline static gg::AffineBodyFixedJoint*     fixed     = nullptr;
    inline static gg::AffineBodySphericalJoint* spherical = nullptr;
    inline static T63B1Probe*                   self      = nullptr;

    /// The scene's `dt` -- the *same* attribute the joint kernels read
    /// (`world().scene().config().find<Float>("dt")`).
    [[nodiscard]] Float dt()
    {
        auto attr = world().scene().config().find<Float>("dt");
        UIPC_ASSERT(attr, "T63B1Probe: the scene config has no `dt`");
        return attr->view()[0];
    }

    [[nodiscard]] S<const geometry::AttributeSlot<Float>> dt_attr()
    {
        return world().scene().config().find<Float>("dt");
    }

  private:
    void do_build() override
    {
        self       = this;
        abd        = &require<gg::AffineBodyDynamics>();
        gls        = &require<gg::GlobalLinearSystem>();
        abd_linear = find<gg::ABDLinearSubsystem>();
        fixed      = find<gg::AffineBodyFixedJoint>();
        spherical  = find<gg::AffineBodySphericalJoint>();
    }
};
REGISTER_SIM_SYSTEM(T63B1Probe);

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
/// vertex 0 to vertex 7) -- the *positively oriented* one the other ABD test files use
/// (a wrong orientation makes `compute_mesh_volume` negative and `apply_to` throws,
/// §7.24 rule 13).
std::array<Vector4i, 6> cube_tets()
{
    return {Vector4i{0, 1, 3, 7}, Vector4i{0, 3, 2, 7}, Vector4i{0, 2, 6, 7},
            Vector4i{0, 6, 4, 7}, Vector4i{0, 4, 5, 7}, Vector4i{0, 5, 1, 7}};
}

/// One ABD body: an `L`-cube of density `rho`, constitution uid `uid` (the frontend
/// writes it) and the *initial* transform `(A, t)` — which is also the ABD state `q`
/// (the dynamics reads `builtin::transform`), so a hand-derived state needs no
/// conversion.
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
    // zero gravity: the BDF1 prediction is then a pure `q + q_v dt` (case 5)
    config["gravity"] = Vector3{0.0, 0.0, 0.0};
    return config;
}

/// The 2-body scene every world case shares.
///
/// Both bodies sit at *rotated and stretched* initial transforms (so the joint's
/// `rest_t/n/b = A^{-1} e_x/y/z` really is an inverse rotation and the hand-derived
/// frames are non-trivial), and the joint's local anchors are arbitrary body-local
/// points.  `with_fixed = with_spherical = false` builds the identical scene *without*
/// the joint geometries: the assembly case differences the two, which isolates the
/// joint's contribution without needing an oracle for the ARAP body term.
struct JointScene
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

    /// body-local anchors: pair 0 uses `(ci_bar, cj_bar)` with `l = body 0` and
    /// `r = body 1`; pair 1 is the *descending* pair `l = body 1`, `r = body 0` (its
    /// anchor pair is swapped so the two joints are not coincidentally symmetric).
    Vector3 ci_bar = Vector3{0.05, 0.10, -0.02};
    Vector3 cj_bar = Vector3{-0.03, 0.07, 0.11};
    Float   ratio  = 100.0;

    [[nodiscard]] Scene build(const std::string& name, bool with_fixed, bool with_spherical) const
    {
        Json  config = joint_scene_config();
        Scene scene{config};
        auto  object = scene.objects().create(name);

        auto body0 = object->geometries().create(make_body(kKappa, A0, t0, kArapUID, V0, v0));
        auto body1 = object->geometries().create(make_body(kKappa, A1, t1, kArapUID, V1, v1));

        if(with_fixed)
        {
            // pair 0: ascending (l = body 0), pair 1: descending (l = body 1)
            std::vector<S<geometry::SimplicialComplexSlot>> l_slots{body0.geometry, body1.geometry};
            std::vector<S<geometry::SimplicialComplexSlot>> r_slots{body1.geometry, body0.geometry};
            std::vector<IndexT>                             l_ids{0, 0};
            std::vector<IndexT>                             r_ids{0, 0};
            std::vector<Vector3>                            l_pos{ci_bar, cj_bar};
            std::vector<Vector3>                            r_pos{cj_bar, ci_bar};
            std::vector<Float>                              ratios{ratio, ratio};
            uipc::constitution::AffineBodyFixedJoint        joint;
            auto                                            joint_geo = joint.create_geometry(
                span<const Vector3>(l_pos),
                span<const Vector3>(r_pos),
                span<S<geometry::SimplicialComplexSlot>>(l_slots),
                span<IndexT>(l_ids),
                span<S<geometry::SimplicialComplexSlot>>(r_slots),
                span<IndexT>(r_ids),
                span<Float>(ratios));
            object->geometries().create(joint_geo);
        }

        if(with_spherical)
        {
            // one ascending pair, anchors 0 (the origin of each body's local frame)
            std::vector<S<geometry::SimplicialComplexSlot>> sl{body0.geometry};
            std::vector<S<geometry::SimplicialComplexSlot>> sr{body1.geometry};
            std::vector<IndexT>                             si{0};
            std::vector<Float>                              s_ratio{ratio};
            std::vector<Vector3>                            s_pos{Vector3::Zero()};
            uipc::constitution::AffineBodySphericalJoint    joint;
            auto                                            joint_geo = joint.create_geometry(
                span<const Vector3>(s_pos),
                span<const Vector3>(s_pos),
                span<S<geometry::SimplicialComplexSlot>>(sl),
                span<IndexT>(si),
                span<S<geometry::SimplicialComplexSlot>>(sr),
                span<IndexT>(si),
                span<Float>(s_ratio));
            object->geometries().create(joint_geo);
        }

        return scene;
    }
};

/// The joint's per-pair rest data as the *scene* defines it (the `do_init` formulas,
/// re-derived here from the bodies' initial transforms and the anchors).
JointRest scene_joint_rest(const JointScene& s, bool l_is_body0)
{
    JointRest r;
    if(l_is_body0)
    {
        r.ci_bar = s.ci_bar;
        r.cj_bar = s.cj_bar;
        r.set_frames_polar(s.A0, s.A1);
    }
    else
    {
        r.ci_bar = s.cj_bar;
        r.cj_bar = s.ci_bar;
        r.set_frames_polar(s.A1, s.A0);
    }
    return r;
}

Vector6 vec6(const Vector3& a, const Vector3& b)
{
    Vector6 v;
    v.segment<3>(0) = a;
    v.segment<3>(3) = b;
    return v;
}

/// One joint *vertex*'s expected host bookkeeping, straight from the scene: the
/// reference's `do_init` writes `rest_c = [l_position | r_position]` (the frontend's own
/// body-local anchors) and `rest_x = [A_l^{-1} e_x | A_r^{-1} e_x]` (the bodies' initial
/// transforms' linear parts), so the *descending* joint vertex simply has its two halves
/// swapped relative to the ascending one.
struct ExpectedJointVertex
{
    Vector6 c, t, n, b;
};

ExpectedJointVertex expected_fixed_vertex(const JointScene& s, bool l_is_body0)
{
    const Matrix3x3 Ai = l_is_body0 ? s.A0 : s.A1;
    const Matrix3x3 Aj = l_is_body0 ? s.A1 : s.A0;
    const Vector3   ci = l_is_body0 ? s.ci_bar : s.cj_bar;
    const Vector3   cj = l_is_body0 ? s.cj_bar : s.ci_bar;
    const Matrix3x3 Li = polar_rotation(Ai).inverse();
    const Matrix3x3 Lj = polar_rotation(Aj).inverse();
    return ExpectedJointVertex{vec6(ci, cj),
                               vec6(Li * Vector3(1, 0, 0), Lj * Vector3(1, 0, 0)),
                               vec6(Li * Vector3(0, 1, 0), Lj * Vector3(0, 1, 0)),
                               vec6(Li * Vector3(0, 0, 1), Lj * Vector3(0, 0, 1))};
}

/// One ABD body of a test scene: the dof offset, the uid, the device `q` and the
/// dynamics' dyadic mass's scalar `m`.
struct BodyInfo
{
    SizeT    dof_offset = 0;
    SizeT    body       = 0;  // = dof_offset / 12
    U64      uid        = 0;
    Vector12 q;
    Float    mass = 0.0;
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
        info.mass = masses[info.body].m_mass;
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

/// Writes the per-geometry dof offsets (`builtin::dof_offset`) the way the engine
/// does: the ABD subsystem's `receive_init_dof_info` runs from `GlobalLinearSystem::init()`,
/// so a test that wants the dof layout must drive a (throwaway) test-local `Impl`
/// first (§7.22 rule 12).
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
//  the joints' *energy* kernels, driven directly
//
//  The T6.3a batch could not reach its energy hook at all (its
//  `ABDLineSearchReporter::ComputeEnergyInfo` is T6.4's and has no creator).  Here the
//  joint's `ComputeEnergyInfo` is an `InterAffineBodyConstitutionManager::EnergyInfo`,
//  whose constructor and the manager's `Impl` are both public, so a test can hand it the
//  dynamics' windows and the scene's `dt` and call the *real* kernel.
// ---------------------------------------------------------------------------
struct EnergyRun
{
    std::vector<double> Es;
};

/// The manager `Impl` a bare energy hook needs: the dynamics (for `qs`/`body_masses`)
/// and the scene's `dt`, plus a one-constitution energy offset table.
void setup_energy_impl(gg::InterAffineBodyConstitutionManager::Impl& impl,
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

EnergyRun run_fixed_energy(gg::AffineBodyFixedJoint&               joint,
                           gg::AffineBodyDynamics&                 abd,
                           S<const geometry::AttributeSlot<Float>> dt_attr,
                           Float                                   dt,
                           SizeT                                   n)
{
    gg::InterAffineBodyConstitutionManager::Impl impl;
    setup_energy_impl(impl, abd, dt_attr, n);

    lud::LDeviceBuffer<Float> Es(n);
    gg::InterAffineBodyConstitutionManager::EnergyInfo info{&impl, 0, dt, Es.view()};
    joint.do_compute_energy(info);
    gg::frame_stream().synchronize();

    EnergyRun out;
    out.Es = read_device(Es.cview());
    return out;
}

EnergyRun run_spherical_energy(gg::AffineBodySphericalJoint&           joint,
                               gg::AffineBodyDynamics&                 abd,
                               S<const geometry::AttributeSlot<Float>> dt_attr,
                               Float                                   dt,
                               SizeT                                   n)
{
    gg::InterAffineBodyConstitutionManager::Impl impl;
    setup_energy_impl(impl, abd, dt_attr, n);

    lud::LDeviceBuffer<Float> Es(n);
    gg::InterAffineBodyConstitutionManager::EnergyInfo info{&impl, 0, dt, Es.view()};
    joint.do_compute_energy(info);
    gg::frame_stream().synchronize();

    EnergyRun out;
    out.Es = read_device(Es.cview());
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
    // default-constructed fixed-size matrix (measured: a ~1e257 entry).  The model
    // test's find/emplace idiom is mandatory.
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
        // the reference's zero_out_lower on a same-body 12x12
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
//  1) closed forms: the two joint laws and their derivatives against the
//     hand-derived `J_t` / `J_r` construction, the scaling laws and the clamp
// ===========================================================================
TEST_CASE("abd_joint_a_closed_forms", "[luisa][abd_joint_a]")
{
    test_runtime();
    CaseScope scope{"abd_joint_a_closed_forms"};
    auto&     c = scope.counter;

    const double kappa = 2.5;

    // ---- (A) the rest/coincident state: `Ft == 0` and `Fr == 0` *exactly* (the laws
    //      are polynomials with no division, so unlike the SVD-derived zeros of the
    //      constitutions these are exact), hence `E == 0.0` and `G == 0.0`, while the
    //      Hessian is the *curvature* of both terms and is not zero.
    {
        JointRest r;
        r.ci_bar = Vector3{0.05, 0.1, -0.02};
        r.cj_bar = Vector3{0.05, 0.1, -0.02};
        r.ti = r.tj = Vector3(1, 0, 0);
        r.ni = r.nj = Vector3(0, 1, 0);
        r.bi = r.bj = Vector3(0, 0, 1);
        const Vector12 qi = q_of(Matrix3x3::Identity(), Vector3{1.0, -0.5, 2.0});
        const Vector12 qj = qi;  // the same pose -> Ft = 0 and Fr = 0

        const auto loc = joint_local(kappa, r, qi, qj);
        T63B1_CHECK(c, fixed_E_host(kappa, r, qi, qj) == 0.0);
        T63B1_CHECK(c, max_abs(fixed_G_host(kappa, r, qi, qj)) == 0.0);
        T63B1_CHECK(c, fixed_E_host(kappa, r, qi, qj) == loc.E);
        const Dense24 H = fixed_H_host(kappa, r, qi, qj);
        T63B1_CHECK(c, dev(H, loc.H) <= 1e-13);
        T63B1_CHECK(c, min_eigenvalue(H) >= -1e-12 * max_abs(H));
        T63B1_CHECK(c, max_abs(H) > 0.0);
        std::printf("[t63b1] closed: rest pair E %.3e G %.3e |H| %.6e (min eig %.3e)\n",
                    fixed_E_host(kappa, r, qi, qj),
                    max_abs(fixed_G_host(kappa, r, qi, qj)),
                    max_abs(H),
                    min_eigenvalue(H));
        // the spherical joint's law is the same statement on Ft alone
        T63B1_CHECK(c, spherical_E_host(kappa, r, qi, qj) == 0.0);
        T63B1_CHECK(c, max_abs(spherical_G_host(kappa, r, qi, qj)) == 0.0);
    }

    // ---- (B) a strained pair, both joints: E/G/H against the closed form, and (with
    //      `Fr` forced to zero) the fixed joint's translation half against the spherical
    //      joint's whole law -- the generated `JtT_*` bodies are the same statements, so
    //      the two sides must agree *bitwise*.
    {
        JointRest r;
        r.ci_bar = Vector3{0.11, -0.07, 0.05};
        r.cj_bar = Vector3{-0.04, 0.09, 0.13};
        r.set_frames(Eigen::AngleAxisd(0.4, Vector3{1, 0, 0}.normalized()).toRotationMatrix(),
                     Eigen::AngleAxisd(-0.7, Vector3{0, 0, 1}.normalized()).toRotationMatrix());
        const Vector12 qi = q_of(Matrix3x3::Identity() * 1.1, Vector3{0.3, -0.2, 0.4});
        const Vector12 qj = q_of(Matrix3x3::Identity() * 0.95, Vector3{-0.1, 0.5, -0.3});

        const auto loc = joint_local(kappa, r, qi, qj);
        T63B1_CHECK(c, rel_err(fixed_E_host(kappa, r, qi, qj), loc.E) <= 1e-14);
        T63B1_CHECK(c, dev(fixed_G_host(kappa, r, qi, qj), loc.G) <= 1e-14);
        T63B1_CHECK(c, dev(fixed_H_host(kappa, r, qi, qj), loc.H_shipped) <= 1e-13);
        // the unclamped H equals the clamped one whenever kappa > 0 (the clamp is a
        // no-op there): asserted, not assumed
        T63B1_CHECK(c, dev(loc.H, loc.H_shipped) == 0.0);

        const auto sloc = spherical_local(kappa, r, qi, qj);
        T63B1_CHECK(c, rel_err(spherical_E_host(kappa, r, qi, qj), sloc.E) <= 1e-14);
        T63B1_CHECK(c, dev(spherical_G_host(kappa, r, qi, qj), sloc.G) <= 1e-14);
        T63B1_CHECK(c, dev(spherical_H_host(kappa, r, qi, qj), sloc.H_shipped) <= 1e-13);
        std::printf("[t63b1] closed: strained pair fixed E %.6e sph E %.6e "
                    "(|Ft| %.3e |Fr| %.3e)\n",
                    fixed_E_host(kappa, r, qi, qj), spherical_E_host(kappa, r, qi, qj),
                    loc.Ft.norm(), loc.Fr.norm());
        {
            // the same pose and the same frames and anchors -> Ft = Fr = 0, i.e. the two
            // joints see the same *zero* constraint.  Splitting the rotation term out:
            // with equal frames the fixed joint's E is `Er + Et == Et + Er`.
            JointRest r0 = r;
            r0.ti = r0.tj;
            r0.ni = r0.nj;
            r0.bi = r0.bj;
            r0.cj_bar = r0.ci_bar;
            T63B1_CHECK(c, fixed_E_host(kappa, r0, qi, qi) == 0.0);
            T63B1_CHECK(c, max_abs(fixed_G_host(kappa, r0, qi, qi)) == 0.0);
            // the joint cannot have less energy than its own translation constraint
            T63B1_CHECK(c,
                        fixed_E_host(kappa, r, qi, qj) - spherical_E_host(kappa, r, qi, qj)
                            >= 0.0);
        }
    }

    // ---- (C) the scaling laws: the local routines are linear in kappa (the kernel's
    //      `kappa = strength_ratio * (m_i + m_j)` is the caller's job), and translation
    //      invariant (only `A` and the *difference* of the translations enter).
    {
        JointRest r;
        r.ci_bar = Vector3{0.2, 0.1, -0.1};
        r.cj_bar = Vector3{-0.1, 0.3, 0.2};
        r.set_frames(Matrix3x3::Identity(), 1.2 * Matrix3x3::Identity());
        const Vector12 qi = q_of(1.1 * Matrix3x3::Identity(), Vector3{0.3, 0.2, -0.1});
        const Vector12 qj = q_of(0.9 * Matrix3x3::Identity(), Vector3{-0.2, 0.4, 0.5});

        const double E1 = fixed_E_host(kappa, r, qi, qj);
        T63B1_CHECK(c, rel_err(fixed_E_host(2.0 * kappa, r, qi, qj), 2.0 * E1) <= 1e-15);
        T63B1_CHECK(c, rel_err(fixed_E_host(-kappa, r, qi, qj), -E1) <= 1e-15);
        // translation invariance: a common shift of both bodies leaves everything alone
        Vector12 qis = qi, qjs = qj;
        qis.segment<3>(0) += Vector3{1.5, -2.0, 0.75};
        qjs.segment<3>(0) += Vector3{1.5, -2.0, 0.75};
        T63B1_CHECK(c, fixed_E_host(kappa, r, qis, qjs) == E1);
        T63B1_CHECK(c,
                    dev(fixed_G_host(kappa, r, qis, qjs), fixed_G_host(kappa, r, qi, qj)) <= 1e-15);
        T63B1_CHECK(c,
                    dev(fixed_H_host(kappa, r, qis, qjs), fixed_H_host(kappa, r, qi, qj)) == 0.0);
        // a common *rotation* of both bodies (`A -> R A`, `t -> R t`) rotates Ft/Fr, so E
        // is invariant and H keeps its spectrum
        const Matrix3x3 R = Eigen::AngleAxisd(0.9, Vector3{1, 2, -1}.normalized()).toRotationMatrix();
        const Vector12  qir = q_of(R * A_of(qi), R * t_of(qi));
        const Vector12  qjr = q_of(R * A_of(qj), R * t_of(qj));
        T63B1_CHECK(c, rel_err(fixed_E_host(kappa, r, qir, qjr), E1) <= 1e-13);
        const Dense24                          Hr = fixed_H_host(kappa, r, qir, qjr);
        Eigen::SelfAdjointEigenSolver<Dense24> e1(fixed_H_host(kappa, r, qi, qj));
        Eigen::SelfAdjointEigenSolver<Dense24> e2(Hr);
        T63B1_CHECK(c, dev(e1.eigenvalues(), e2.eigenvalues()) <= 1e-12);
    }

    // ---- (D) the reference's `make_spd` on the two *element* Hessians: a no-op for
    //      kappa > 0 (every element Hessian is `kappa I`), and the whole H for a
    //      *negative* kappa (the clamp zeroes both fan-outs) while E and G keep their
    //      unclamped values.  Also the local 8-sweep clamp == the layer's 20-sweep one.
    {
        JointRest r;
        r.ci_bar = Vector3{0.3, -0.2, 0.1};
        r.cj_bar = Vector3{0.1, 0.25, -0.15};
        r.set_frames(Matrix3x3::Identity(), Matrix3x3::Identity());
        const Vector12 qi = q_of(1.3 * Matrix3x3::Identity(), Vector3{0.1, 0.2, 0.3});
        const Vector12 qj = q_of(0.8 * Matrix3x3::Identity(), Vector3{-0.3, 0.1, 0.2});

        const double   kneg = -3.0;
        const auto     lneg = joint_local(kneg, r, qi, qj);
        const Dense24  Hneg = fixed_H_host(kneg, r, qi, qj);
        T63B1_CHECK(c, max_abs(Hneg) == 0.0);  // fully clamped
        T63B1_CHECK(c, dev(fixed_G_host(kneg, r, qi, qj), lneg.G) <= 1e-14);
        T63B1_CHECK(c, rel_err(fixed_E_host(kneg, r, qi, qj), lneg.E) <= 1e-14);
        T63B1_CHECK(c, lneg.E < 0.0);  // a negative coefficient is a negative energy
        std::printf("[t63b1] closed clamp: kappa < 0 -> max|H| %.3e while |G| %.6e, E %.6e\n",
                    max_abs(Hneg), max_abs(fixed_G_host(kneg, r, qi, qj)),
                    fixed_E_host(kneg, r, qi, qj));

        // the shipped 9x9/3x3 clamps vs the layer's 20 sweeps (kappa > 0: both are the
        // identical matrix; the element Hessians are diagonal, so 1 sweep would do)
        const lps::LMat<double, 9, 9> k9 = kappa * lps::LMat<double, 9, 9>::Identity();
        const lps::LMat<double, 3, 3> k3 = kappa * lps::LMat<double, 3, 3>::Identity();
        auto                          a9 = k9, b9 = k9;
        auto                          a3 = k3, b3 = k3;
        lps::make_spd_inplace(a9, gg::abd_fixed_joint_detail::kMakeSpdSweeps);
        lps::make_spd_inplace(b9, 20);
        lps::make_spd_inplace(a3, gg::abd_fixed_joint_detail::kMakeSpdSweeps);
        lps::make_spd_inplace(b3, 20);
        double worst_spd = 0.0;
        for(int cc = 0; cc < 9; ++cc)
            for(int rr = 0; rr < 9; ++rr)
                worst_spd = std::max(worst_spd, std::abs(a9(rr, cc) - b9(rr, cc)));
        for(int cc = 0; cc < 3; ++cc)
            for(int rr = 0; rr < 3; ++rr)
                worst_spd = std::max(worst_spd, std::abs(a3(rr, cc) - b3(rr, cc)));
        T63B1_CHECK(c, worst_spd == 0.0);
        bool kappa_identity = true;
        for(int cc = 0; cc < 9; ++cc)
            for(int rr = 0; rr < 9; ++rr)
                if(a9(rr, cc) != (rr == cc ? kappa : 0.0))
                    kappa_identity = false;
        T63B1_CHECK(c, kappa_identity);  // kappa > 0: the clamp is a no-op
    }

    std::printf("[t63b1] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  2) central finite differences over the 24 pair dofs: the gradient is the
//     derivative of E, the Hessian the derivative of the gradient, the h-sweep,
//     the symmetry, and the rule-34 basis question (the joints are self-consistent)
// ===========================================================================
TEST_CASE("abd_joint_a_finite_difference", "[luisa][abd_joint_a]")
{
    test_runtime();
    CaseScope scope{"abd_joint_a_finite_difference"};
    auto&     c = scope.counter;

    constexpr int kStates = 40;

    double worst_fx_g = 0.0, worst_fx_h6 = 0.0, worst_fx_h4 = 0.0;
    double worst_fx_h3 = 0.0, worst_fx_h2 = 0.0, worst_fx_sym = 0.0;
    double worst_sp_g = 0.0, worst_sp_h6 = 0.0, worst_sp_h4 = 0.0;
    double worst_sp_sym = 0.0, worst_sp_proj = 0.0;
    int    fx_indefinite = 0;

    for(int s = 0; s < kStates; ++s)
    {
        JointRest r;
        r.ci_bar = rand_vec(-0.3, 0.3);
        r.cj_bar = rand_vec(-0.3, 0.3);
        const Matrix3x3 Ai = random_admissible_A();
        const Matrix3x3 Aj = random_admissible_A();
        r.set_frames(Ai, Aj);
        const Vector12 qi    = q_of(Ai, rand_vec(-0.5, 0.5));
        const Vector12 qj    = q_of(Aj, rand_vec(-0.5, 0.5));
        const double   kappa = rand_uniform(0.5, 2.0);
        const Dense24v q     = pack_pair(qi, qj);

        // ---- the fixed joint --------------------------------------------------
        {
            const Dense24v G = fixed_G_host(kappa, r, qi, qj);
            const Dense24v Gfd =
                fd_gradient([&](const Dense24v& qq)
                            { return fixed_E_host(kappa,
                                                  r,
                                                  Vector12(qq.segment<12>(0)),
                                                  Vector12(qq.segment<12>(12))); },
                            q,
                            kFdH);
            worst_fx_g = std::max(worst_fx_g, dev(G, Gfd));

            const Dense24 H = fixed_H_host(kappa, r, qi, qj);
            auto          grad = [&](const Dense24v& qq)
            { return fixed_G_host(kappa, r, Vector12(qq.segment<12>(0)), Vector12(qq.segment<12>(12))); };
            const Dense24 Hfd6 = fd_hessian(grad, q, 1e-6);
            const Dense24 Hfd4 = fd_hessian(grad, q, 1e-4);
            const Dense24 Hfd3 = fd_hessian(grad, q, 1e-3);
            const Dense24 Hfd2 = fd_hessian(grad, q, 1e-2);
            worst_fx_h6        = std::max(worst_fx_h6, dev(H, Hfd6));
            worst_fx_h4        = std::max(worst_fx_h4, dev(H, Hfd4));
            worst_fx_h3        = std::max(worst_fx_h3, dev(H, Hfd3));
            worst_fx_h2        = std::max(worst_fx_h2, dev(H, Hfd2));
            worst_fx_sym = std::max(worst_fx_sym, max_abs(H - Dense24(H.transpose())) / max_abs(H));
            if(min_eigenvalue(H) < -1e-12 * max_abs(H))
                ++fx_indefinite;
        }

        // ---- the spherical joint ---------------------------------------------
        {
            const Dense24v G = spherical_G_host(kappa, r, qi, qj);
            const Dense24v Gfd =
                fd_gradient([&](const Dense24v& qq)
                            { return spherical_E_host(kappa,
                                                      r,
                                                      Vector12(qq.segment<12>(0)),
                                                      Vector12(qq.segment<12>(12))); },
                            q,
                            kFdH);
            worst_sp_g = std::max(worst_sp_g, dev(G, Gfd));

            const Dense24 H = spherical_H_host(kappa, r, qi, qj);
            auto          grad = [&](const Dense24v& qq)
            { return spherical_G_host(kappa, r, Vector12(qq.segment<12>(0)), Vector12(qq.segment<12>(12))); };
            const Dense24 Hfd6 = fd_hessian(grad, q, 1e-6);
            const Dense24 Hfd4 = fd_hessian(grad, q, 1e-4);
            worst_sp_h6        = std::max(worst_sp_h6, dev(H, Hfd6));
            worst_sp_h4        = std::max(worst_sp_h4, dev(H, Hfd4));
            worst_sp_sym = std::max(worst_sp_sym, max_abs(H - Dense24(H.transpose())) / max_abs(H));

            // the closed-form `kappa J_t^T J_t` (built from the hand-derived Jacobian
            // alone) against the ported routine
            worst_sp_proj =
                std::max(worst_sp_proj,
                         dev(spherical_local(kappa, r, qi, qj).H_shipped,
                             spherical_H_host(kappa, r, qi, qj)));
        }
    }

    std::printf("[t63b1] FD fixed : grad %3.1e  H(h=1e-6) %3.1e  (1e-4) %3.1e  (1e-3) %3.1e  "
                "(1e-2) %3.1e  asym %3.1e  indefinite %d/%d\n",
                worst_fx_g, worst_fx_h6, worst_fx_h4, worst_fx_h3, worst_fx_h2, worst_fx_sym,
                fx_indefinite, kStates);
    std::printf("[t63b1] FD sph   : grad %3.1e  H(h=1e-6) %3.1e  (1e-4) %3.1e  asym %3.1e; "
                "closed-form kappa Jt^T Jt vs ported %3.1e\n",
                worst_sp_g, worst_sp_h6, worst_sp_h4, worst_sp_sym, worst_sp_proj);

    // the gradient is a true derivative of the shipped energy
    T63B1_CHECK(c, worst_fx_g <= 1e-9);
    T63B1_CHECK(c, worst_sp_g <= 1e-9);
    // ... and the Hessian the derivative of the shipped gradient, at the brief's h = 1e-6
    // and at the FEM rows' measured optimum h = 1e-4
    T63B1_CHECK(c, worst_fx_h6 <= 1e-8);
    T63B1_CHECK(c, worst_fx_h4 <= 1e-6);
    T63B1_CHECK(c, worst_sp_h6 <= 1e-8);
    T63B1_CHECK(c, worst_sp_h4 <= 1e-6);
    // the joint laws are *quadratic* in the state, so the Hessian is CONSTANT: the
    // central-difference Hessian has no truncation error at all and the h-sweep is pure
    // round-off (which *falls* with h, unlike the constitutive families' sweeps).
    // Asserted twice: (i) the ported Hessian does not depend on the state, (ii) the
    // measured sweep is round-off-only at every h.
    {
        JointRest r;
        r.ci_bar = rand_vec(-0.3, 0.3);
        r.cj_bar = rand_vec(-0.3, 0.3);
        r.set_frames(random_admissible_A(), random_admissible_A());
        const double kappa = 1.7;
        const Vector12 a0 = q_of(random_admissible_A(), rand_vec(-1, 1));
        const Vector12 a1 = q_of(random_admissible_A(), rand_vec(-1, 1));
        const Vector12 b0 = q_of(random_admissible_A(), rand_vec(-1, 1));
        const Vector12 b1 = q_of(random_admissible_A(), rand_vec(-1, 1));
        T63B1_CHECK(c, max_abs(fixed_H_host(kappa, r, a0, a1) - fixed_H_host(kappa, r, b0, b1))
                           == 0.0);
        T63B1_CHECK(c,
                    max_abs(spherical_H_host(kappa, r, a0, a1)
                            - spherical_H_host(kappa, r, b0, b1))
                        == 0.0);
    }
    T63B1_CHECK(c, worst_fx_h2 <= worst_fx_h6);  // round-off ~ eps/h: smaller h is worse
    T63B1_CHECK(c, worst_fx_h6 <= 1e-8);
    // symmetry: the generated fan-outs are entry-by-entry symmetric (the reference uses
    // the *same* temporary for R(i,j) and R(j,i)), so this is round-off only
    T63B1_CHECK(c, worst_fx_sym <= 1e-15);
    T63B1_CHECK(c, worst_sp_sym <= 1e-15);
    // the hand-derived closed form IS the shipped Hessian
    T63B1_CHECK(c, worst_sp_proj <= 1e-13);

    std::printf("[t63b1] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  3) degenerate/rigid inputs, pinned from the source: the joint laws are
//     polynomials (no division, no SVD), so E/G/H are finite everywhere --
//     including at a reflection -- and zero *exactly* where the constraint is
// ===========================================================================
TEST_CASE("abd_joint_a_degenerate_inputs", "[luisa][abd_joint_a]")
{
    test_runtime();
    CaseScope scope{"abd_joint_a_degenerate_inputs"};
    auto&     c = scope.counter;

    const double kappa = 3.0;

    JointRest r;
    r.ci_bar = Vector3{0.15, -0.05, 0.2};
    r.cj_bar = Vector3{-0.2, 0.1, 0.05};
    r.set_frames(Matrix3x3::Identity(), Matrix3x3::Identity());

    // ---- (D1) `A_i == A_j == 0` (a degenerate deformation): the law is still a
    //      polynomial, so E/G/H are finite.  `Ft = ti - tj` (the anchors drop out) and
    //      `Fr = 0`, so E = 1/2 kappa |ti - tj|^2 and G has only the two translation
    //      3-blocks.  No input "repairing".
    {
        const Vector12 qi = q_of(Matrix3x3::Zero(), Vector3{1.0, 2.0, 3.0});
        const Vector12 qj = q_of(Matrix3x3::Zero(), Vector3{0.0, 1.0, 1.0});
        const double   E  = fixed_E_host(kappa, r, qi, qj);
        const Dense24v G  = fixed_G_host(kappa, r, qi, qj);
        const Dense24  H  = fixed_H_host(kappa, r, qi, qj);
        const double   E_expected = 0.5 * kappa * (Vector3{1.0, 1.0, 2.0}).squaredNorm();
        T63B1_CHECK(c, all_finite(G));
        T63B1_CHECK(c, all_finite(H));
        T63B1_CHECK(c, rel_err(E, E_expected) <= 1e-14);
        T63B1_CHECK(c, dev(G.segment<3>(0), kappa * Vector3{1.0, 1.0, 2.0}) <= 1e-14);
        T63B1_CHECK(c, dev(G.segment<3>(12), -kappa * Vector3{1.0, 1.0, 2.0}) <= 1e-14);
        // the A-blocks of the gradient do *not* vanish here: `dFt(k)/dA_i(k,m)` is
        // `ci_bar(m)`, so `(J_t^T Ft)(3+3k+m) = ci_bar(m) Ft(k)` -- a `ci_bar (x) Ft`
        // outer product (and its negative `cj_bar (x) Ft` on the right body).  Deriving
        // that by hand is what makes this assertion a pin rather than an assumption.
        {
            const Dense3v Ft_here = ft_of_hand(r, qi, qj);
            double       worst    = 0.0;
            for(int k = 0; k < 3; ++k)
                for(int m = 0; m < 3; ++m)
                {
                    worst = std::max(worst,
                                     std::abs(G(3 + 3 * k + m)
                                              - kappa * r.ci_bar(m) * Ft_here(k)));
                    worst = std::max(worst,
                                     std::abs(G(15 + 3 * k + m)
                                              + kappa * r.cj_bar(m) * Ft_here(k)));
                }
            std::printf("[t63b1] degen: the A-blocks of G are `kappa ci (x) Ft` / "
                        "`-kappa cj (x) Ft`, worst dev %.3e (|G_Ab| %.3e)\n",
                        worst, max_abs(G.segment<9>(3)));
            T63B1_CHECK(c, worst <= 1e-14);
        }
        std::printf("[t63b1] degen: A=0 -> E %.6e (1/2 kappa |dt|^2 %.6e), G finite %d, "
                    "H finite %d\n",
                    E, E_expected, static_cast<int>(all_finite(G)),
                    static_cast<int>(all_finite(H)));
    }

    // ---- (D2) a *reflection* (`A_i = diag(-1,1,1)`): the joint has no SVD and no
    //      determinant branch, so the whole law goes through unchanged and is finite;
    //      the closed form is the same `joint_local` (its Jacobians are state
    //      independent).
    {
        const Matrix3x3 Ai = Vector3{-1.0, 1.0, 1.0}.asDiagonal();
        const Matrix3x3 Aj = Vector3{1.0, -1.0, 1.0}.asDiagonal();
        r.set_frames(Ai, Aj);
        const Vector12 qi  = q_of(Ai, Vector3{0.1, -0.2, 0.3});
        const Vector12 qj  = q_of(Aj, Vector3{-0.1, 0.2, -0.3});
        const auto     loc = joint_local(kappa, r, qi, qj);
        const double   E   = fixed_E_host(kappa, r, qi, qj);
        const Dense24v G   = fixed_G_host(kappa, r, qi, qj);
        const Dense24  H   = fixed_H_host(kappa, r, qi, qj);
        T63B1_CHECK(c, all_finite(G));
        T63B1_CHECK(c, all_finite(H));
        T63B1_CHECK(c, rel_err(E, loc.E) <= 1e-14);
        T63B1_CHECK(c, dev(G, loc.G) <= 1e-14);
        T63B1_CHECK(c, dev(H, loc.H_shipped) <= 1e-13);
        std::printf("[t63b1] degen: reflection det(A) %.3f / %.3f -> E %.6e |G| %.6e finite %d\n",
                    Ai.determinant(), Aj.determinant(), E, max_abs(G),
                    static_cast<int>(all_finite(H)));
        r.set_frames(Matrix3x3::Identity(), Matrix3x3::Identity());
    }

    // ---- (D3) `kappa == 0` (a user's `strength_ratio = 0`): the whole joint is inert --
    //      E == 0, G == 0 and the clamped H == 0 *exactly* (the element Hessians are the
    //      zero matrix).
    {
        const Vector12 qi = q_of(1.2 * Matrix3x3::Identity(), Vector3{0.3, 0.4, -0.2});
        const Vector12 qj = q_of(0.7 * Matrix3x3::Identity(), Vector3{-0.3, -0.4, 0.2});
        T63B1_CHECK(c, fixed_E_host(0.0, r, qi, qj) == 0.0);
        T63B1_CHECK(c, max_abs(fixed_G_host(0.0, r, qi, qj)) == 0.0);
        T63B1_CHECK(c, max_abs(fixed_H_host(0.0, r, qi, qj)) == 0.0);
        T63B1_CHECK(c, spherical_E_host(0.0, r, qi, qj) == 0.0);
        T63B1_CHECK(c, max_abs(spherical_G_host(0.0, r, qi, qj)) == 0.0);
        T63B1_CHECK(c, max_abs(spherical_H_host(0.0, r, qi, qj)) == 0.0);
    }

    // ---- (D4) the *negative kappa* pin: the shipped H is NOT `dG/dq` once `kappa < 0`
    //      (G is unclamped there, H is not), and this file asserts the *measured* size of
    //      that difference so a future reader cannot mistake the clamp for round-off.
    {
        const double   kneg = -2.0;
        const Vector12 qi   = q_of(1.1 * Matrix3x3::Identity(), Vector3{0.2, 0.1, -0.3});
        const Vector12 qj   = q_of(0.9 * Matrix3x3::Identity(), Vector3{-0.2, 0.3, 0.1});
        const Dense24  H    = spherical_H_host(kneg, r, qi, qj);
        T63B1_CHECK(c, max_abs(H) == 0.0);
        const Dense24v G = spherical_G_host(kneg, r, qi, qj);
        T63B1_CHECK(c, max_abs(G) > 0.0);
        const Dense24 Hfd =
            fd_hessian([&](const Dense24v& qq)
                       { return spherical_G_host(kneg,
                                                 r,
                                                 Vector12(qq.segment<12>(0)),
                                                 Vector12(qq.segment<12>(12))); },
                       pack_pair(qi, qj),
                       1e-6);
        const double measured = dev(H, Hfd);
        T63B1_CHECK(c, measured > 1e-4);  // the clamp really is observable
        std::printf("[t63b1] degen: kappa < 0 shipped-H vs FD(dG/dq) deviation %.3e "
                    "(|G| %.3e) -- the clamp pinned, not a rounding artefact\n",
                    measured, max_abs(G));
    }

    // ---- (D5) an extreme scale (a 1e12 coefficient and 1e-6 rest frames) stays finite:
    //      the law has no division, so there is no conditioning cliff.
    {
        JointRest tiny;
        tiny.ci_bar = tiny.cj_bar = Vector3{1e-6, -2e-6, 3e-6};
        tiny.set_frames(Matrix3x3::Identity(), Matrix3x3::Identity());
        const Vector12 qi = q_of(1e-3 * Matrix3x3::Identity(), Vector3{1e-6, 0.0, -1e-6});
        const Vector12 qj = q_of(2e-3 * Matrix3x3::Identity(), Vector3{0.0, 1e-6, 1e-6});
        T63B1_CHECK(c, std::isfinite(fixed_E_host(1e12, tiny, qi, qj)));
        T63B1_CHECK(c, all_finite(fixed_G_host(1e12, tiny, qi, qj)));
        T63B1_CHECK(c, all_finite(fixed_H_host(1e12, tiny, qi, qj)));
    }

    std::printf("[t63b1] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  4) the fan-out into ABDLinearSubsystem's assembly, isolated by differencing a
//     scene *with* and *without* the joint geometries; plus the joints' energy
//     kernels through a test-built manager Impl
// ===========================================================================
TEST_CASE("abd_joint_a_assembly", "[luisa][abd_joint_a]")
{
    test_runtime();
    CaseScope scope{"abd_joint_a_assembly"};

    JointScene scene_data;

    // ---- (A) the same 2-body scene twice, with and without the joints -----------
    Engine engine_a{"luisa", workspace_path("uipc_luisa_abd_joint_a_base")};
    World  world_a{engine_a};
    Scene  base_scene = scene_data.build("abd_joint_a_base", false, false);
    world_a.init(base_scene);
    REQUIRE(world_a.is_valid());
    REQUIRE(T63B1Probe::abd != nullptr);
    REQUIRE(T63B1Probe::abd_linear != nullptr);
    REQUIRE(T63B1Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B1Probe::abd_linear);
    auto*      linear_a = T63B1Probe::abd_linear;
    const auto base     = run_assembly(*linear_a);

    Engine engine_b{"luisa", workspace_path("uipc_luisa_abd_joint_a_joint")};
    World  world_b{engine_b};
    Scene  joint_scene = scene_data.build("abd_joint_a_joint", true, true);
    world_b.init(joint_scene);
    REQUIRE(world_b.is_valid());
    REQUIRE(T63B1Probe::abd != nullptr);
    REQUIRE(T63B1Probe::abd_linear != nullptr);
    REQUIRE(T63B1Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B1Probe::abd_linear);
    auto*      abd_b     = T63B1Probe::abd;
    auto*      linear_b  = T63B1Probe::abd_linear;
    const auto joint_run = run_assembly(*linear_b);

    auto& c = scope.counter;

    // ---- (B) uid ownership (proved, see the file comment) ----------------------
    const bool fixed_owns = T63B1Probe::fixed != nullptr && T63B1Probe::fixed->is_valid()
                            && T63B1Probe::fixed->h_body_ids.size() == 2u;
    const bool spherical_owns = T63B1Probe::spherical != nullptr
                                && T63B1Probe::spherical->is_valid()
                                && T63B1Probe::spherical->h_body_ids.size() == 1u;
    std::printf("[t63b1] assembly uid ownership: uid25 = %s (%zu joints), uid26 = %s (%zu)\n",
                fixed_owns ? "uipc::backend::luisa::AffineBodyFixedJoint" : "ANOTHER SYSTEM",
                T63B1Probe::fixed ? T63B1Probe::fixed->h_body_ids.size() : 0u,
                spherical_owns ? "uipc::backend::luisa::AffineBodySphericalJoint"
                               : "ANOTHER SYSTEM",
                T63B1Probe::spherical ? T63B1Probe::spherical->h_body_ids.size() : 0u);
    REQUIRE(fixed_owns);
    REQUIRE(spherical_owns);

    // ---- (C) the extent budget difference --------------------------------------
    T63B1_CHECK(c, base.dof_count == 24u);
    T63B1_CHECK(c, base.block_count == 2u);
    T63B1_CHECK(c, joint_run.dof_count == 24u);
    // hand-derived: the three joints add 3 upper-triangular 12x12 blocks each -> 9 more
    // 12x12 blocks -> 9 * 16 more 3x3 triplets
    T63B1_CHECK(c, joint_run.block_count == 2u + 9u);
    T63B1_CHECK(c, joint_run.triplet_count == base.triplet_count + 9u * 16u);
    std::printf("[t63b1] assembly extent: base %zu triplets / %zu blocks, joint %zu / %zu\n",
                base.triplet_count, base.block_count, joint_run.triplet_count,
                joint_run.block_count);

    // ---- (D) the joints' *energy kernels*, on the real dynamics windows ---------
    {
        const auto dt      = T63B1Probe::self->dt();
        const auto dt_attr = T63B1Probe::self->dt_attr();
        const auto bodies  = scene_bodies(world_b, *abd_b);
        REQUIRE(bodies.size() == 2u);

        const auto e_fixed = run_fixed_energy(*T63B1Probe::fixed, *abd_b, dt_attr, dt, 2);
        const auto e_sph   = run_spherical_energy(*T63B1Probe::spherical, *abd_b, dt_attr, dt, 1);
        REQUIRE(e_fixed.Es.size() == 2u);
        REQUIRE(e_sph.Es.size() == 1u);

        // the same two joints, hand-derived: pair 0 (l = body 0, r = body 1) and pair 1
        // (l = body 1, r = body 0) for the fixed joint, one ascending pair with the
        // origin anchors for the spherical one
        const auto r_fixed0 = scene_joint_rest(scene_data, true);
        const auto r_fixed1 = scene_joint_rest(scene_data, false);
        JointRest  r_sph    = scene_joint_rest(scene_data, true);
        r_sph.ci_bar        = Vector3::Zero();
        r_sph.cj_bar        = Vector3::Zero();

        const Vector12 q0 = bodies[0].q;
        const Vector12 q1 = bodies[1].q;
        const double   k0 = scene_data.ratio * (bodies[0].mass + bodies[1].mass);

        const double expect_fixed0 = fixed_E_host(k0, r_fixed0, q0, q1);
        const double expect_fixed1 = fixed_E_host(k0, r_fixed1, q1, q0);
        const double expect_sph    = spherical_E_host(k0, r_sph, q0, q1);
        const double worst_e = std::max({rel_err(e_fixed.Es[0], expect_fixed0),
                                         rel_err(e_fixed.Es[1], expect_fixed1),
                                         rel_err(e_sph.Es[0], expect_sph)});
        std::printf("[t63b1] energy kernel: fixed [%.8e %.8e] (hand %.8e %.8e), sph %.8e "
                    "(hand %.8e), worst rel %.3e; masses %.6e/%.6e kappa %.6e\n",
                    e_fixed.Es[0], e_fixed.Es[1], expect_fixed0, expect_fixed1, e_sph.Es[0],
                    expect_sph, worst_e, static_cast<double>(bodies[0].mass),
                    static_cast<double>(bodies[1].mass), k0);
        T63B1_CHECK(c, std::abs(expect_fixed0) > 1e-12);
        T63B1_CHECK(c, std::abs(expect_sph) > 1e-12);
        T63B1_CHECK(c, worst_e <= 1e-13);
    }

    // ---- (E) the assembled COO and gradient *difference* vs the hand-derived joint
    //      contribution ---------------------------------------------------------
    const auto bodies = scene_bodies(world_b, *abd_b);
    REQUIRE(bodies.size() == 2u);
    const int  body_of_uid0 = static_cast<int>(bodies[0].body);
    const int  body_of_uid1 = static_cast<int>(bodies[1].body);
    const auto r_fixed0     = scene_joint_rest(scene_data, true);
    const auto r_fixed1     = scene_joint_rest(scene_data, false);
    JointRest  r_sph        = scene_joint_rest(scene_data, true);
    r_sph.ci_bar            = Vector3::Zero();
    r_sph.cj_bar            = Vector3::Zero();
    const Vector12 q0 = bodies[0].q;
    const Vector12 q1 = bodies[1].q;
    const double   k0 = scene_data.ratio * (bodies[0].mass + bodies[1].mass);

    const auto l0 = joint_local(k0, r_fixed0, q0, q1);
    const auto l1 = joint_local(k0, r_fixed1, q1, q0);
    const auto ls = spherical_local(k0, r_sph, q0, q1);
    {
        const Dense24 H0 = fixed_H_host(k0, r_fixed0, q0, q1);  // == l0.H_shipped (kappa > 0)
        const Dense24 H1 = fixed_H_host(k0, r_fixed1, q1, q0);
        const Dense24 Hs = spherical_H_host(k0, r_sph, q0, q1);

        BlockMap expected;
        add_joint_blocks(expected, body_of_uid0, body_of_uid1, H0);
        // pair 1: (l = body 1, r = body 0) *descending* -- the upper_LR path
        add_joint_blocks(expected, body_of_uid1, body_of_uid0, H1);
        add_joint_blocks(expected, body_of_uid0, body_of_uid1, Hs);

        const BlockMap produced = sub_blocks(joint_run.blocks, base.blocks);

        double worst_entry = 0.0, worst_block = 0.0;
        int    compared = 0, missing = 0, extra = 0;
        for(const auto& [key, blk] : expected)
        {
            auto it = produced.find(key);
            if(it == produced.end())
            {
                std::printf("[t63b1] assembly: missing key (%d, %d)\n", key.first, key.second);
                ++missing;
                continue;
            }
            worst_block = std::max(worst_block, (it->second - blk).norm());
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                    worst_entry = std::max(worst_entry, std::abs(it->second(rr, cc) - blk(rr, cc)));
            ++compared;
        }
        for(const auto& [key, blk] : produced)
            if(expected.find(key) == expected.end() && max_abs(blk) > 0.0)
                ++extra;
        std::printf("[t63b1] assembly COO (joint-only, by differencing): %d keys compared, "
                    "worst |entry| %.3e, worst |block| %.3e, missing %d, non-zero extra %d\n",
                    compared, worst_entry, worst_block, missing, extra);
        T63B1_CHECK(c, missing == 0);
        T63B1_CHECK(c, extra == 0);
        T63B1_CHECK(c, worst_entry <= 1e-9);
        T63B1_CHECK(c, worst_block > 0.0);

        // the gradient scatter: G at the bodies' dof offsets
        auto scatter_into = [](Eigen::VectorXd& target, const Dense24v& G, int bi, int bj)
        {
            target.segment<12>(static_cast<Eigen::Index>(bi) * 12) += G.segment<12>(0);
            target.segment<12>(static_cast<Eigen::Index>(bj) * 12) += G.segment<12>(12);
        };
        Eigen::VectorXd expected_g = Eigen::VectorXd::Zero(24);
        scatter_into(expected_g, l0.G, body_of_uid0, body_of_uid1);
        scatter_into(expected_g, l1.G, body_of_uid1, body_of_uid0);
        scatter_into(expected_g, ls.G, body_of_uid0, body_of_uid1);
        Eigen::VectorXd produced_g = joint_run.gradient - base.gradient;
        double          worst_g    = 0.0;
        for(Eigen::Index i = 0; i < expected_g.size(); ++i)
            worst_g = std::max(worst_g, rel_err(produced_g(i), expected_g(i)));
        std::printf("[t63b1] assembly gradient (joint-only): worst rel %.3e, |b_joint| %.6e\n",
                    worst_g, expected_g.norm());
        T63B1_CHECK(c, expected_g.norm() > 1.0);
        T63B1_CHECK(c, worst_g <= 1e-12);

        // ---- (F) the whole assembled gradient (base + joints) against the base scene's
        //      own gradient plus the joint scatter: an absolute (not differenced) check
        //      that the joint really lands at the bodies' dof offsets
        Eigen::VectorXd expected_abs = base.gradient;
        scatter_into(expected_abs, l0.G, body_of_uid0, body_of_uid1);
        scatter_into(expected_abs, l1.G, body_of_uid1, body_of_uid0);
        scatter_into(expected_abs, ls.G, body_of_uid0, body_of_uid1);
        double worst_abs = 0.0;
        for(Eigen::Index i = 0; i < expected_abs.size(); ++i)
            worst_abs = std::max(worst_abs, rel_err(joint_run.gradient(i), expected_abs(i)));
        std::printf("[t63b1] assembly gradient (absolute): worst rel %.3e (|b| %.6e)\n",
                    worst_abs, expected_abs.norm());
        T63B1_CHECK(c, expected_abs.norm() > 1.0);
        T63B1_CHECK(c, worst_abs <= 1e-12);
    }

    std::printf("[t63b1] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  5) a real scene: two ABD bodies, one fixed joint and one spherical joint, one
//     `advance()`, the frontend state accessor, the BDF1 prediction and the
//     per-joint host bookkeeping
// ===========================================================================
TEST_CASE("abd_joint_a_scene", "[luisa][abd_joint_a]")
{
    test_runtime();
    CaseScope scope{"abd_joint_a_scene"};

    JointScene scene_data;
    Scene      scene = scene_data.build("abd_joint_a_scene", true, true);

    Engine engine{"luisa", workspace_path("uipc_luisa_abd_joint_a_scene")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    auto& c = scope.counter;
    REQUIRE(T63B1Probe::abd != nullptr);
    REQUIRE(T63B1Probe::abd_linear != nullptr);
    REQUIRE(T63B1Probe::abd_linear->is_valid());
    prepare_dof_layout(*T63B1Probe::abd_linear);

    // ---- (A) the per-joint *host* bookkeeping, against the scene's own definitions --
    {
        REQUIRE(T63B1Probe::fixed != nullptr);
        REQUIRE(T63B1Probe::fixed->is_valid());
        REQUIRE(T63B1Probe::spherical != nullptr);
        REQUIRE(T63B1Probe::spherical->is_valid());
        const auto& f = *T63B1Probe::fixed;
        REQUIRE(f.h_body_ids.size() == 2u);
        REQUIRE(f.h_strength_ratios.size() == 2u);

        const auto v0 = expected_fixed_vertex(scene_data, true);
        const auto v1 = expected_fixed_vertex(scene_data, false);
        auto       cmp6 = [&](const Vector6& got, const Vector6& want)
        {
            double w = 0.0;
            for(int k = 0; k < 6; ++k)
                w = std::max(w, std::abs(got(k) - want(k)));
            return w;
        };
        double worst_rest = 0.0;
        worst_rest        = std::max(worst_rest, cmp6(f.h_rest_cs[0], v0.c));
        worst_rest        = std::max(worst_rest, cmp6(f.h_rest_ts[0], v0.t));
        worst_rest        = std::max(worst_rest, cmp6(f.h_rest_ns[0], v0.n));
        worst_rest        = std::max(worst_rest, cmp6(f.h_rest_bs[0], v0.b));
        worst_rest        = std::max(worst_rest, cmp6(f.h_rest_cs[1], v1.c));
        worst_rest        = std::max(worst_rest, cmp6(f.h_rest_ts[1], v1.t));
        worst_rest        = std::max(worst_rest, cmp6(f.h_rest_ns[1], v1.n));
        worst_rest        = std::max(worst_rest, cmp6(f.h_rest_bs[1], v1.b));
        std::printf("[t63b1] scene host bookkeeping: fixed %zu joints, worst |rest - hand| "
                    "%.3e, ratios %.3f/%.3f\n",
                    f.h_body_ids.size(), worst_rest, f.h_strength_ratios[0],
                    f.h_strength_ratios[1]);
        T63B1_CHECK(c, worst_rest <= 1e-14);
        T63B1_CHECK(c, f.h_strength_ratios[0] == scene_data.ratio);
        T63B1_CHECK(c, f.h_strength_ratios[1] == scene_data.ratio);

        // the spherical joint's single pair, anchors 0
        const auto& s = *T63B1Probe::spherical;
        REQUIRE(s.h_body_ids.size() == 1u);
        double worst_s = 0.0;
        for(int k = 0; k < 6; ++k)
            worst_s = std::max(worst_s, std::abs(s.h_rest_cs[0](k)));
        T63B1_CHECK(c, worst_s == 0.0);
        T63B1_CHECK(c, s.h_strength_ratios[0] == scene_data.ratio);
    }

    // ---- (B) the frontend state accessor + one advance() -------------------------
    auto feature = world.features().find<AffineBodyStateAccessorFeature>();
    REQUIRE(feature != nullptr);
    T63B1_CHECK(c, feature->body_count() == 2);
    auto state_geo = feature->create_geometry();
    T63B1_CHECK(c, state_geo.instances().size() == 2);
    auto trans_attr = state_geo.instances().find<Matrix4x4>(builtin::transform);
    if(!trans_attr)
        trans_attr = state_geo.instances().create<Matrix4x4>(builtin::transform,
                                                             Matrix4x4::Identity());

    auto check_accessor = [&](const char* when)
    {
        feature->copy_to(state_geo);
        auto       trans  = view(*trans_attr);
        const auto bodies = scene_bodies(world, *T63B1Probe::abd);
        double     worst  = 0.0;
        for(const auto& b : bodies)
        {
            REQUIRE(b.body < 2u);
            const Matrix4x4 expected = gg::q_to_transform(b.q);
            for(int r = 0; r < 4; ++r)
                for(int cc = 0; cc < 4; ++cc)
                    worst = std::max(worst, std::abs(trans[b.body](r, cc) - expected(r, cc)));
        }
        std::printf("[t63b1] scene %s: accessor vs device q, worst |dT| = %.3e\n", when, worst);
        return worst;
    };
    T63B1_CHECK(c, check_accessor("before advance") <= 1e-12);

    world.advance();
    REQUIRE(world.is_valid());
    world.sync();
    REQUIRE(world.is_valid());
    T63B1_CHECK(c, check_accessor("after advance") <= 1e-12);

    // ---- (C) the BDF1 prediction by hand, and the *state* staying put -------------
    // `q_tilde = q + q_v dt` (gravity 0); the state update path is T8.3's + T6.4's, so
    // `q` is still the scene's initial state after one advance() (§7.24 rule 12).
    const auto dt     = T63B1Probe::self->dt();
    const auto bodies = scene_bodies(world, *T63B1Probe::abd);
    REQUIRE(bodies.size() == 2u);
    {
        const auto q_tilde = read_vec12(T63B1Probe::abd->q_tildes());
        REQUIRE(q_tilde.size() == 2u);
        const Matrix3x3 V[2] = {scene_data.V0, scene_data.V1};
        const Vector3   v[2] = {scene_data.v0, scene_data.v1};
        double          worst = 0.0;
        for(int i = 0; i < 2; ++i)
        {
            const Vector12 qv = gg::transform_to_q(make_transform(V[i], v[i]));
            const Vector12 q0 = (i == 0 ? q_of(scene_data.A0, scene_data.t0)
                                        : q_of(scene_data.A1, scene_data.t1));
            const Vector12 expected = q0 + static_cast<double>(dt) * qv;
            for(int k = 0; k < 12; ++k)
            {
                worst = std::max(worst, std::abs(q_tilde[bodies[i].body](k) - expected(k)));
                // the state is the scene's initial state
                worst = std::max(worst, std::abs(bodies[i].q(k) - q0(k)));
            }
        }
        std::printf("[t63b1] scene BDF1: |q_tilde - (q + q_v dt)| and |q_state - q_initial| "
                    "worst %.3e (dt %.6f)\n",
                    worst, static_cast<double>(dt));
        T63B1_CHECK(c, worst <= 1e-12);
    }

    // ---- (D) the joints at the *read-back* state, against the hand-derived laws ----
    {
        const auto   r0 = scene_joint_rest(scene_data, true);
        const auto   r1 = scene_joint_rest(scene_data, false);
        JointRest    rs = r0;
        rs.ci_bar       = Vector3::Zero();
        rs.cj_bar       = Vector3::Zero();
        const double k0 = scene_data.ratio * (bodies[0].mass + bodies[1].mass);

        const auto   l0 = joint_local(k0, r0, bodies[0].q, bodies[1].q);
        const auto   l1 = joint_local(k0, r1, bodies[1].q, bodies[0].q);
        const auto   ls = spherical_local(k0, rs, bodies[0].q, bodies[1].q);
        const double ef = fixed_E_host(k0, r0, bodies[0].q, bodies[1].q)
                          + fixed_E_host(k0, r1, bodies[1].q, bodies[0].q);
        const double es = spherical_E_host(k0, rs, bodies[0].q, bodies[1].q);
        T63B1_CHECK(c, rel_err(ef, l0.E + l1.E) <= 1e-13);
        T63B1_CHECK(c, rel_err(es, ls.E) <= 1e-13);
        T63B1_CHECK(c, ef + es > 0.0);

        // the *state's* energy through the real kernels
        const auto e_fixed = run_fixed_energy(
            *T63B1Probe::fixed, *T63B1Probe::abd, T63B1Probe::self->dt_attr(), dt, 2);
        const auto e_sph = run_spherical_energy(
            *T63B1Probe::spherical, *T63B1Probe::abd, T63B1Probe::self->dt_attr(), dt, 1);
        double worst = 0.0;
        worst        = std::max(worst, rel_err(e_fixed.Es[0], l0.E));
        worst        = std::max(worst, rel_err(e_fixed.Es[1], l1.E));
        worst        = std::max(worst, rel_err(e_sph.Es[0], ls.E));
        std::printf("[t63b1] scene energy kernels: fixed %.8e/%.8e (hand %.8e/%.8e), sph %.8e "
                    "(hand %.8e), worst rel %.3e\n",
                    e_fixed.Es[0], e_fixed.Es[1], l0.E, l1.E, e_sph.Es[0], ls.E, worst);
        T63B1_CHECK(c, worst <= 1e-13);

        // the assembled joint blocks at the read-back state
        const auto run = run_assembly(*T63B1Probe::abd_linear);
        T63B1_CHECK(c, run.dof_count == 24u);
        T63B1_CHECK(c, run.block_count == 2u + 9u);
        BlockMap expected;
        add_joint_blocks(expected,
                         static_cast<int>(bodies[0].body),
                         static_cast<int>(bodies[1].body),
                         fixed_H_host(k0, r0, bodies[0].q, bodies[1].q));
        add_joint_blocks(expected,
                         static_cast<int>(bodies[1].body),
                         static_cast<int>(bodies[0].body),
                         fixed_H_host(k0, r1, bodies[1].q, bodies[0].q));
        add_joint_blocks(expected,
                         static_cast<int>(bodies[0].body),
                         static_cast<int>(bodies[1].body),
                         spherical_H_host(k0, rs, bodies[0].q, bodies[1].q));
        // the *absolute* triplet_A also carries the ABD body term (kappa = 1e8), which
        // the base scene's differencing removes in the assembly case; here the
        // *cross-body* keys are the ones the body term cannot touch, so they are the
        // absolute statement (the diagonal ones are covered by the differenced check).
        double worst_h  = 0.0;
        int    compared = 0;
        for(const auto& [key, blk] : expected)
        {
            if(key.first / 4 == key.second / 4)
                continue;  // a same-body key: the body term lives there
            ++compared;
            auto it = run.blocks.find(key);
            REQUIRE(it != run.blocks.end());
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                {
                    const double w = std::abs(it->second(rr, cc) - blk(rr, cc));
                    if(w > 1.0)
                        std::printf("[t63b1] scene key (%d,%d) block(%d,%d) dev %.6e "
                                    "(produced %.6e, hand-derived %.6e)\n",
                                    key.first, key.second, rr, cc, w, it->second(rr, cc),
                                    blk(rr, cc));
                    worst_h = std::max(worst_h, w);
                }
        }
        std::printf("[t63b1] scene assembled joint blocks (cross-body): worst "
                    "|dev-hand-derived| %.3e (%d keys, %zu in the map)\n",
                    worst_h, compared, expected.size());
        T63B1_CHECK(c, compared == 16);
        T63B1_CHECK(c, worst_h <= 1e-9);
    }

    std::printf("[t63b1] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  6) device == host for the six ported generic routines
//
//     The joints' kernels are file-local (their cached shaders live in an anonymous
//     namespace in the .cpp), so the device round dispatches a *test* kernel that calls
//     the *same* generic routines with `S = lcrt::Var<double>` and compares against the
//     host's `S = double`.  The 24-vector/24x24 results are stored as two 12-vectors and
//     four 12x12 blocks (there is no registered 24-dim element type in the port, and none
//     is needed).
// ===========================================================================
namespace
{
auto& t63b1_round_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector6>,   // rest_c
                                                lcrt::Buffer<gg::LVector6>,   // rest_t
                                                lcrt::Buffer<gg::LVector6>,   // rest_n
                                                lcrt::Buffer<gg::LVector6>,   // rest_b
                                                lcrt::Buffer<gg::LVector12>,  // qi
                                                lcrt::Buffer<gg::LVector12>,  // qj
                                                lcrt::Buffer<double>,         // kappas
                                                lcrt::Buffer<double>,         // fixed E
                                                lcrt::Buffer<gg::LVector12>,  // fixed G0
                                                lcrt::Buffer<gg::LVector12>,  // fixed G1
                                                lcrt::Buffer<gg::LMatrix12x12>,  // fixed H00
                                                lcrt::Buffer<gg::LMatrix12x12>,  // fixed H01
                                                lcrt::Buffer<gg::LMatrix12x12>,  // fixed H10
                                                lcrt::Buffer<gg::LMatrix12x12>,  // fixed H11
                                                lcrt::Buffer<double>,         // sph E
                                                lcrt::Buffer<gg::LVector12>,  // sph G0
                                                lcrt::Buffer<gg::LVector12>,  // sph G1
                                                lcrt::Buffer<gg::LMatrix12x12>,  // sph H00
                                                lcrt::Buffer<gg::LMatrix12x12>,  // sph H01
                                                lcrt::Buffer<gg::LMatrix12x12>,  // sph H10
                                                lcrt::Buffer<gg::LMatrix12x12>,  // sph H11
                                                uint32_t>(                       // n
        [](lcrt::BufferVar<gg::LVector6>       rest_c,
           lcrt::BufferVar<gg::LVector6>       rest_t,
           lcrt::BufferVar<gg::LVector6>       rest_n,
           lcrt::BufferVar<gg::LVector6>       rest_b,
           lcrt::BufferVar<gg::LVector12>      qi_buf,
           lcrt::BufferVar<gg::LVector12>      qj_buf,
           lcrt::BufferVar<double>             kappas,
           lcrt::BufferVar<double>             fE,
           lcrt::BufferVar<gg::LVector12>      fG0,
           lcrt::BufferVar<gg::LVector12>      fG1,
           lcrt::BufferVar<gg::LMatrix12x12>   fH00,
           lcrt::BufferVar<gg::LMatrix12x12>   fH01,
           lcrt::BufferVar<gg::LMatrix12x12>   fH10,
           lcrt::BufferVar<gg::LMatrix12x12>   fH11,
           lcrt::BufferVar<double>             sE,
           lcrt::BufferVar<gg::LVector12>      sG0,
           lcrt::BufferVar<gg::LVector12>      sG1,
           lcrt::BufferVar<gg::LMatrix12x12>   sH00,
           lcrt::BufferVar<gg::LMatrix12x12>   sH01,
           lcrt::BufferVar<gg::LMatrix12x12>   sH10,
           lcrt::BufferVar<gg::LMatrix12x12>   sH11,
           lcrt::Var<uint32_t>                 n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto kappa = kappas->read(i);
                auto qi    = gg::load_vec12(qi_buf->read(i));
                auto qj    = gg::load_vec12(qj_buf->read(i));

                gg::abd_fixed_joint_detail::FixedJointRestT<lcrt::Var<double>> fr{
                    .rest_c = gg::load_vec6(rest_c->read(i)),
                    .rest_t = gg::load_vec6(rest_t->read(i)),
                    .rest_n = gg::load_vec6(rest_n->read(i)),
                    .rest_b = gg::load_vec6(rest_b->read(i))};
                fE->write(
                    i, gg::abd_fixed_joint_detail::body_energy<lcrt::Var<double>>(kappa, fr, qi, qj));
                auto G = gg::abd_fixed_joint_detail::body_gradient<lcrt::Var<double>>(kappa, fr, qi, qj);
                lcrt::Var<gg::LVector12> g0, g1;
                gg::store_vec12(g0, lps::LVec<lcrt::Var<Float>, 12>(G.template segment<12>(0)));
                gg::store_vec12(g1, lps::LVec<lcrt::Var<Float>, 12>(G.template segment<12>(12)));
                fG0->write(i, g0);
                fG1->write(i, g1);
                const auto H =
                    gg::abd_fixed_joint_detail::body_hessian<lcrt::Var<double>>(kappa, fr, qi, qj);
                auto store_block = [&](lcrt::BufferVar<gg::LMatrix12x12>& dst, int br, int bc)
                {
                    lcrt::Var<gg::LMatrix12x12> out;
                    gg::store_mat12x12(
                        out,
                        lps::LMat<lcrt::Var<Float>, 12, 12>(H.template block<12, 12>(br, bc)));
                    dst->write(i, out);
                };
                store_block(fH00, 0, 0);
                store_block(fH01, 0, 12);
                store_block(fH10, 12, 0);
                store_block(fH11, 12, 12);

                gg::abd_spherical_joint_detail::SphericalJointRestT<lcrt::Var<double>> sr{
                    .rest_c = gg::load_vec6(rest_c->read(i))};
                sE->write(
                    i, gg::abd_spherical_joint_detail::body_energy<lcrt::Var<double>>(kappa, sr, qi, qj));
                auto S = gg::abd_spherical_joint_detail::body_gradient<lcrt::Var<double>>(kappa, sr, qi, qj);
                lcrt::Var<gg::LVector12> sg0, sg1;
                gg::store_vec12(sg0, lps::LVec<lcrt::Var<Float>, 12>(S.template segment<12>(0)));
                gg::store_vec12(sg1, lps::LVec<lcrt::Var<Float>, 12>(S.template segment<12>(12)));
                sG0->write(i, sg0);
                sG1->write(i, sg1);
                const auto SH =
                    gg::abd_spherical_joint_detail::body_hessian<lcrt::Var<double>>(kappa, sr, qi, qj);
                auto store_sblock = [&](lcrt::BufferVar<gg::LMatrix12x12>& dst, int br, int bc)
                {
                    lcrt::Var<gg::LMatrix12x12> out;
                    gg::store_mat12x12(
                        out,
                        lps::LMat<lcrt::Var<Float>, 12, 12>(SH.template block<12, 12>(br, bc)));
                    dst->write(i, out);
                };
                store_sblock(sH00, 0, 0);
                store_sblock(sH01, 0, 12);
                store_sblock(sH10, 12, 0);
                store_sblock(sH11, 12, 12);
            };
        });
    return k;
}
}  // namespace

TEST_CASE("abd_joint_a_device_vs_host", "[luisa][abd_joint_a]")
{
    test_runtime();
    CaseScope scope{"abd_joint_a_device_vs_host"};
    auto&     c = scope.counter;

    constexpr int N = 16;

    std::vector<Vector12>  qis, qjs;
    std::vector<Vector6>   rc, rt, rn, rb;
    std::vector<double>    kappas;
    std::vector<JointRest> rests;
    for(int i = 0; i < N; ++i)
    {
        JointRest r;
        r.ci_bar = rand_vec(-0.3, 0.3);
        r.cj_bar = rand_vec(-0.3, 0.3);
        const Matrix3x3 Ai = random_admissible_A();
        const Matrix3x3 Aj = random_admissible_A();
        r.set_frames(Ai, Aj);
        qis.push_back(q_of(Ai, rand_vec(-0.5, 0.5)));
        qjs.push_back(q_of(Aj, rand_vec(-0.5, 0.5)));
        kappas.push_back(rand_uniform(0.5, 2.0));
        rests.push_back(r);
        rc.push_back(vec6(r.ci_bar, r.cj_bar));
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

    lud::LDeviceBuffer<double>           fE(N), sE(N);
    lud::LDeviceBuffer<gg::LVector12>    fG0(N), fG1(N), sG0(N), sG1(N);
    lud::LDeviceBuffer<gg::LMatrix12x12> fH00(N), fH01(N), fH10(N), fH11(N);
    lud::LDeviceBuffer<gg::LMatrix12x12> sH00(N), sH01(N), sH10(N), sH11(N);

    auto& k = t63b1_round_kernel();
    gg::frame_stream().record(k(rcd.view(),
                                rtd.view(),
                                rnd.view(),
                                rbd.view(),
                                qd.view(),
                                qjd.view(),
                                kd.view(),
                                fE.view(),
                                fG0.view(),
                                fG1.view(),
                                fH00.view(),
                                fH01.view(),
                                fH10.view(),
                                fH11.view(),
                                sE.view(),
                                sG0.view(),
                                sG1.view(),
                                sH00.view(),
                                sH01.view(),
                                sH10.view(),
                                sH11.view(),
                                static_cast<uint32_t>(N))
                                .dispatch(N));
    gg::frame_stream().synchronize();

    auto flatten12 = [](const std::vector<gg::LVector12>& raw)
    {
        std::vector<double> out;
        out.reserve(raw.size() * 12u);
        for(const auto& v : raw)
        {
            const Vector12 e = gg::to_eigen(v);
            for(int k2 = 0; k2 < 12; ++k2)
                out.push_back(e(k2));
        }
        return out;
    };
    auto flatten_block = [](const std::vector<gg::LMatrix12x12>& raw)
    {
        std::vector<double> out;
        out.reserve(raw.size() * 144u);
        for(const auto& v : raw)
        {
            const Matrix12x12 e = gg::to_eigen(v);
            for(int rr = 0; rr < 12; ++rr)
                for(int cc = 0; cc < 12; ++cc)
                    out.push_back(e(rr, cc));
        }
        return out;
    };

    const auto                               dfE = read_device(fE.cview());
    const auto                               dsE = read_device(sE.cview());
    const auto                               dfG0 = flatten12(read_device(fG0.cview()));
    const auto                               dfG1 = flatten12(read_device(fG1.cview()));
    const auto                               dsG0 = flatten12(read_device(sG0.cview()));
    const auto                               dsG1 = flatten12(read_device(sG1.cview()));
    const std::array<std::vector<double>, 4> dfH{flatten_block(read_device(fH00.cview())),
                                                 flatten_block(read_device(fH01.cview())),
                                                 flatten_block(read_device(fH10.cview())),
                                                 flatten_block(read_device(fH11.cview()))};
    const std::array<std::vector<double>, 4> dsH{flatten_block(read_device(sH00.cview())),
                                                 flatten_block(read_device(sH01.cview())),
                                                 flatten_block(read_device(sH10.cview())),
                                                 flatten_block(read_device(sH11.cview()))};

    std::vector<double> hfE(N), hsE(N);
    std::vector<double> hfG(static_cast<std::size_t>(N) * 24u), hsG(static_cast<std::size_t>(N) * 24u);
    std::vector<double> hfH(static_cast<std::size_t>(N) * 576u), hsH(static_cast<std::size_t>(N) * 576u);
    for(int i = 0; i < N; ++i)
    {
        hfE[i]            = fixed_E_host(kappas[i], rests[i], qis[i], qjs[i]);
        hsE[i]            = spherical_E_host(kappas[i], rests[i], qis[i], qjs[i]);
        const Dense24v Gf = fixed_G_host(kappas[i], rests[i], qis[i], qjs[i]);
        const Dense24v Gs = spherical_G_host(kappas[i], rests[i], qis[i], qjs[i]);
        const Dense24  Hf = fixed_H_host(kappas[i], rests[i], qis[i], qjs[i]);
        const Dense24  Hs = spherical_H_host(kappas[i], rests[i], qis[i], qjs[i]);
        for(int k2 = 0; k2 < 24; ++k2)
        {
            hfG[static_cast<std::size_t>(i) * 24u + k2] = Gf(k2);
            hsG[static_cast<std::size_t>(i) * 24u + k2] = Gs(k2);
        }
        for(int rr = 0; rr < 24; ++rr)
            for(int cc = 0; cc < 24; ++cc)
            {
                hfH[(static_cast<std::size_t>(i) * 24u + rr) * 24u + cc] = Hf(rr, cc);
                hsH[(static_cast<std::size_t>(i) * 24u + rr) * 24u + cc] = Hs(rr, cc);
            }
    }
    // the device layout: 4 blocks per sample
    auto device_H = [&](const std::vector<double>& h00,
                        const std::vector<double>& h01,
                        const std::vector<double>& h10,
                        const std::vector<double>& h11)
    {
        std::vector<double> out(static_cast<std::size_t>(N) * 576u);
        for(int i = 0; i < N; ++i)
            for(int blk = 0; blk < 4; ++blk)
            {
                const auto& src = (blk == 0) ? h00 : (blk == 1) ? h01 : (blk == 2) ? h10 : h11;
                const int   br  = (blk == 0 || blk == 1) ? 0 : 12;
                const int   bc  = (blk == 0 || blk == 2) ? 0 : 12;
                for(int rr = 0; rr < 12; ++rr)
                    for(int cc = 0; cc < 12; ++cc)
                        out[(static_cast<std::size_t>(i) * 24u + br + rr) * 24u + bc + cc] =
                            src[static_cast<std::size_t>(i) * 144u + rr * 12 + cc];
            }
        return out;
    };
    const auto dfH_flat = device_H(dfH[0], dfH[1], dfH[2], dfH[3]);
    const auto dsH_flat = device_H(dsH[0], dsH[1], dsH[2], dsH[3]);

    auto worst = [](const std::vector<double>& a, const std::vector<double>& b)
    {
        double w = 0.0;
        REQUIRE(a.size() == b.size());
        for(std::size_t i = 0; i < a.size(); ++i)
            w = std::max(w, std::abs(a[i] - b[i]));
        return w;
    };
    auto scale_of = [](const std::vector<double>& v)
    {
        double s = 0.0;
        for(double x : v)
            s = std::max(s, std::abs(x));
        return s;
    };
    auto flatten_half = [&](const std::vector<double>& g, int off)
    {
        std::vector<double> out(static_cast<std::size_t>(N) * 12u);
        for(int i = 0; i < N; ++i)
            for(int k2 = 0; k2 < 12; ++k2)
                out[static_cast<std::size_t>(i) * 12u + k2] =
                    g[static_cast<std::size_t>(i) * 24u + off + k2];
        return out;
    };
    const double hfG0 = worst(flatten_half(hfG, 0), dfG0) / scale_of(dfG0);
    const double hfG1 = worst(flatten_half(hfG, 12), dfG1) / scale_of(dfG1);
    const double hsG0 = worst(flatten_half(hsG, 0), dsG0) / scale_of(dsG0);
    const double hsG1 = worst(flatten_half(hsG, 12), dsG1) / scale_of(dsG1);

    const double r_fe = worst(dfE, hfE) / scale_of(hfE);
    const double r_fg = std::max(hfG0, hfG1);
    const double r_fh = worst(dfH_flat, hfH) / scale_of(hfH);
    const double r_se = worst(dsE, hsE) / scale_of(hsE);
    const double r_sg = std::max(hsG0, hsG1);
    const double r_sh = worst(dsH_flat, hsH) / scale_of(hsH);

    std::printf("[t63b1] device vs host (%d states): fixed E %.3e G %.3e H %.3e | "
                "sph E %.3e G %.3e H %.3e  (relative %.2e/%.2e/%.2e, %.2e/%.2e/%.2e; halves "
                "%.2e %.2e %.2e %.2e)\n",
                N, worst(dfE, hfE), worst(flatten_half(hfG, 0), dfG0), worst(dfH_flat, hfH),
                worst(dsE, hsE), worst(flatten_half(hsG, 0), dsG0), worst(dsH_flat, hsH), r_fe, r_fg,
                r_fh, r_se, r_sg, r_sh, hfG0, hfG1, hsG0, hsG1);
    // the generic routines are the same statements, but the generated laws call
    // `lps::pow`, which lowers to `lcrt::pow` on the DSL: assert a per-routine bound and
    // *record* the measurement (§7.24 rule 15) -- this round is NOT bitwise.
    T63B1_CHECK(c, (r_fe <= 1e-12 && r_fg <= 1e-12 && r_fh <= 1e-12));
    T63B1_CHECK(c, (r_se <= 1e-12 && r_sg <= 1e-12 && r_sh <= 1e-12));

    std::printf("[t63b1] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  7) uid ownership of 25 and 26, *proved* (the real constitutions own the
//     geometries their uids name; a stolen group would zero their host records)
// ===========================================================================
TEST_CASE("abd_joint_a_uid_ownership", "[luisa][abd_joint_a]")
{
    test_runtime();
    CaseScope scope{"abd_joint_a_uid_ownership"};
    auto&     c = scope.counter;

    JointScene scene_data;
    Scene      scene = scene_data.build("abd_joint_a_uid", true, true);
    Engine     engine{"luisa", workspace_path("uipc_luisa_abd_joint_a_uid")};
    World      world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    // the probe's `find<>` (registered in every scene; an invalidated system is still
    // returned, cf. §7.22 rule 13 -- so `is_valid()` is part of the question)
    const bool fixed_ok = T63B1Probe::fixed != nullptr && T63B1Probe::fixed->is_valid()
                          && T63B1Probe::fixed->h_body_ids.size() == 2u;
    const bool sph_ok = T63B1Probe::spherical != nullptr && T63B1Probe::spherical->is_valid()
                        && T63B1Probe::spherical->h_body_ids.size() == 1u;
    std::printf("[t63b1] uid ownership: uid25 = %s (valid %d, joints %zu), "
                "uid26 = %s (valid %d, joints %zu)\n",
                fixed_ok ? "uipc::backend::luisa::AffineBodyFixedJoint" : "ANOTHER SYSTEM",
                static_cast<int>(fixed_ok),
                T63B1Probe::fixed ? T63B1Probe::fixed->h_body_ids.size() : 0u,
                sph_ok ? "uipc::backend::luisa::AffineBodySphericalJoint" : "ANOTHER SYSTEM",
                static_cast<int>(sph_ok),
                T63B1Probe::spherical ? T63B1Probe::spherical->h_body_ids.size() : 0u);
    REQUIRE(fixed_ok);
    REQUIRE(sph_ok);

    // the device records the same body ids as the host staging
    const auto h_bids = read_device(T63B1Probe::fixed->body_ids.cview());
    REQUIRE(h_bids.size() == 2u);
    double worst = 0.0;
    for(std::size_t i = 0; i < h_bids.size(); ++i)
    {
        const auto e = gg::to_eigen(h_bids[i]);
        worst        = std::max(worst,
                                std::abs(static_cast<double>(e(0))
                                         - static_cast<double>(T63B1Probe::fixed->h_body_ids[i](0))));
        worst        = std::max(worst,
                                std::abs(static_cast<double>(e(1))
                                         - static_cast<double>(T63B1Probe::fixed->h_body_ids[i](1))));
    }
    T63B1_CHECK(c, worst == 0.0);
    // the two pairs really are the ascending and the descending one (the `upper_LR` path
    // is exercised by the *scene*, not only by the unit cases)
    T63B1_CHECK(c, T63B1Probe::fixed->h_body_ids[0](0) < T63B1Probe::fixed->h_body_ids[0](1));
    T63B1_CHECK(c, T63B1Probe::fixed->h_body_ids[1](0) > T63B1Probe::fixed->h_body_ids[1](1));
    // ... and the geometry groups are disjoint (each uid names exactly its own joints)
    T63B1_CHECK(c,
                T63B1Probe::fixed->h_body_ids.size() + T63B1Probe::spherical->h_body_ids.size()
                    == 3u);

    std::printf("[t63b1] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  8) per-case assertion counts
// ===========================================================================
TEST_CASE("abd_joint_a_summary", "[luisa][abd_joint_a]")
{
    int total = 0;
    for(const auto& [name, checks] : g_case_counts)
    {
        std::printf("[t63b1] case %-42s %5d checks\n", name.c_str(), checks);
        total += checks;
        CHECK(checks > 0);
    }
    std::printf("[t63b1] summary: %d cases, %d checks\n",
                static_cast<int>(g_case_counts.size()), total);
}
