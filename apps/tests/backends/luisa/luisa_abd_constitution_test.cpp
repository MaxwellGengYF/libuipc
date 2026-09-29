/********************************************************************
 * @file luisa_abd_constitution_test.cpp
 * @brief T6.3a — the ABD constitutions of the LuisaCompute backend: the ABD ARAP
 *        (uid 2, `uipc::backend::luisa::ARAP`) and the orthogonal potential
 *        (uid 1, `uipc::backend::luisa::OrthoPotential`).
 *
 * Every oracle in this file is derived independently *here*: the reference backend
 * is CUDA-only and cannot be host-included, so nothing is copied from it.  The
 * oracles are
 *  (a) hand-derived closed forms with the derivation written out at the assertion:
 *      * ARAP: `E = kappa ||F - R||_F^2` with `R = U V^T` from the signed SVD.  At a
 *        rigid map `R = F` so `E == 0` exactly; at a stretch `F = s I` (`s > 0`) the
 *        polar factor is `I`, so `E = 3 kappa (s - 1)^2` and `dE/dF = 2 kappa (s-1) I`;
 *        at `F = diag(a,b,c)` (`a,b,c > 0`) `R = I`, so `E = kappa sum_i (a_i - 1)^2`.
 *        The Hessian at an isotropic stretch is derived from the *projector* form
 *        `H = 2 kappa (I_9 - (1/s) P_a)` with `P_a = (I_9 - P)/2` the projector onto
 *        the antisymmetric (rotation) directions and `P` the transpose permutation --
 *        no reference formula is transcribed for this one; it follows from `E` being
 *        flat along rotations (eigenvalue 0) and quadratic across them (2 kappa).
 *      * ortho potential: the law is read out of the *generated* file's own symbol
 *        map (`sym/ortho_potential.inl`): with `C = A^T A - I`,
 *        `E = kappa ||C||_F^2` (the three `(|a_i|^2 - 1)^2` terms are the diagonal of
 *        `C`, the three `2 (...)^2` terms its two off-diagonal entries), `dE/dA = 4 kappa A C`,
 *        and its exact Hessian, derived by differentiating `dE/dA` along one
 *        elementary direction `A -> A + e E_rs`,
 *        `H[(i,j),(r,s)] = 4 kappa [ delta_ir C(s,j) + A(i,s) A(r,j) + (A A^T)(i,r) delta_js ]`
 *        (the derivation is repeated at the assertion).
 *  (b) the *independent analytic sibling* T6.1a already ports: `abd_energy.h`'s
 *      `shape_energy` / `shape_energy_gradient` / `shape_energy_hessian` (the
 *      transcription of the reference's own `abd_energy.cu`, i.e. the operator form
 *      `V_perp`) must equal the SymEigen-generated law divided by `kappa` -- two
 *      independent derivations of one function, one generated and one analytic.
 *  (c) central finite differences over the **12 element dofs** (h = 1e-6 plus a
 *      measured sweep), the Hessian's symmetry, and the eigenvalue-clamp identity
 *      where the reference clamps (`make_spd`, ortho only -- the ARAP ships the
 *      *unclamped* exact Hessian, which this file pins).
 *      FINDING (cookbook §7.24 rule 5): the reference's **ABD ARAP Hessian is spelled
 *      in the column-major `vec` basis while its `dEdq` writes the dof (row-major)
 *      order**, so the shipped Hessian is the `transpose_permutation()`-conjugate of
 *      the true one and the plain FD comparison is off by ~1e-2 (the *FEM* sibling's
 *      `dEdF` is column-major, so the FEM ARAP is self-consistent and does not have
 *      this).  The port keeps both statements verbatim (§7.17 rule 34) and this case
 *      pins *both* -- the conjugated comparison and the measurable plain deviation.
 *      The generated ortho potential is self-consistent (its `dEdq`'s index order is
 *      the one its `ddEddq` uses: `m = 3i + j`), and the same case proves it.
 *  (d) degenerate inputs pinned from the source (neither constitution has an input
 *      guard) -- which of E/G/H stay finite, with the exact values where the branch
 *      cannot matter.
 *  (e) the fan-out into `ABDLinearSubsystem`'s assembly (a test-local
 *      `GlobalLinearSystem::Impl`, the T6.1b idiom): the extent it reports, the
 *      `(body, vertex)` block indices + the upper-triangular COO convention, the
 *      assembled gradient at the body's dof offset, and the assembled 12x12 block
 *      against a dense oracle.
 *  (f) a real scene (one uid-2 body **and** one uid-1 body, so both constitutions are
 *      in play in one dof space), one `advance()`, read back through the frontend
 *      state accessor.
 *  (g) device == host for every ported generic routine (S = double vs
 *      S = lcrt::Var<double>), bitwise.
 *
 * Coverage map:
 *  1) closed forms + kappa/volume/dt scaling + no-clamp .... abd_constitution_closed_forms
 *  2) central FD (g/H/symmetry) + the SPD-clamp identity .... abd_constitution_finite_difference
 *  3) zero-volume/singular/reflection, pinned from the source . abd_constitution_degenerate_inputs
 *  4) the assembly fan-out (extents, indices, dense block) .. abd_constitution_assembly
 *  5) a real scene, both constitutions, one advance() ....... abd_constitution_scene
 *  6) device == host for the generic routines ............... abd_constitution_device_vs_host
 *  7) per-case assertion counts ............................ abd_constitution_summary
 *
 * UID OWNERSHIP (the T5.2 §7.17 rule 12 / §7.24 rule 9 resolution).
 * `uipc::constitution::AffineBodyConstitution` writes uid 1 ("OrthoPotential") or
 * uid 2 ("ARAP") into a geometry's `constitution_uid` meta attribute, and
 * `AffineBodyDynamics::Impl::_build_constitutions` sorts the registered backend
 * constitutions by uid and maps each uid to the **last** of the equal-uid entries,
 * in a build order that `SimEngine::build_systems()` fixes by demangled type name.
 * T6.1a's test-local mock used to claim uid 1; with the real `OrthoPotential` in the
 * same binary that claim loses (`uipc::...` sorts after `T61A...`), so T6.3a **moved
 * the mock to uid 8** (grep-proven unused, the last frontend-registered
 * `AffineBody`-type uid) and labelled that file's scene geometries with it (see
 * `luisa_abd_base_test.cpp`).  The same move was applied to
 * `luisa_abd_linear_test.cpp`'s scenes, whose oracle is the *assembly* and therefore
 * needs the constitution term to stay exactly zero.  This file does not assume the
 * resolution: the `assembly` and `scene` cases **prove** ownership (the real
 * constitution's `do_init` ran with exactly this scene's bodies -- its parameter
 * buffer holds one entry per body -- and the assembled block equals this file's
 * hand-derived contribution) and print a named owner line; if the group ever moves
 * the case fails with that line above it instead of silently weakening an assertion.
 *
 * Reserved test uids after this batch: **8** (T6.1a's mock, moved here from 1),
 * 16 / 25 (T6.1b's constraint and inter-body stubs), 666 (the ABD external-force
 * manager), 671 (the FEM external-force manager).
 ********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/geometry/utils/compute_mesh_volume.h>
#include <uipc/geometry/utils/affine_body/compute_dyadic_mass.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

#include <affine_body/type_define.h>
#include <affine_body/utils.h>
#include <affine_body/abd_energy.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_constitution.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/constitutions/arap.h>
#include <affine_body/constitutions/arap_function.h>
#include <affine_body/constitutions/ortho_potential.h>
#include <affine_body/constitutions/ortho_potential_function.h>
#include <global_geometry/global_body_manager.h>
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
using Dense9  = Eigen::Matrix<double, 9, 9>;
using Dense9v = Eigen::Matrix<double, 9, 1>;
using Dense12 = Eigen::Matrix<double, 12, 12>;
using Dense12v = Eigen::Matrix<double, 12, 1>;

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

template <int N, typename Derived>
double max_rel_dev(const Eigen::Matrix<double, N, N>& a,
                   const Eigen::MatrixBase<Derived>& b)
{
    double worst = 0.0;
    for(int c = 0; c < N; ++c)
        for(int r = 0; r < N; ++r)
            worst = std::max(worst, rel_err(a(r, c), b(r, c)));
    return worst;
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
//  Eigen <-> lps conversions for the generic routines (the ported layer's own
//  `to_luisa`/`to_eigen` cover 3/12; 9 is spelled here because no LC buffer uses it)
// ---------------------------------------------------------------------------
Dense9v to_eigen(lps::LVec<double, 9> v)
{
    Dense9v out;
    for(int i = 0; i < 9; ++i)
        out(i) = v(i);
    return out;
}

Vector9 to_eigen9(lps::LVec<double, 9> v)
{
    Vector9 out;
    for(int i = 0; i < 9; ++i)
        out(i) = v(i);
    return out;
}

Dense9 to_eigen9(lps::LMat<double, 9, 9> m)
{
    Dense9 out;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
            out(r, c) = m(r, c);
    return out;
}

lps::LVec<double, 9> lv9(const Vector9& v)
{
    lps::LVec<double, 9> out;
    for(int i = 0; i < 9; ++i)
        out(i) = v(i);
    return out;
}

// ---------------------------------------------------------------------------
//  deterministic admissible random states
//
//  `A = R (I + S)` with `S` symmetric and `|S| <= 0.25`, then rescaled so that
//  `sigma_min >= 0.35` and `sigma_max <= 2.5`: well conditioned, `det A > 0` (so the
//  signed SVD's branch cannot flip), and far enough from the rest state that the
//  gradient is O(kappa) rather than O(round-off).
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

/// A modest strain matrix (symmetric part only -- the rotation part is drawn
/// separately) so that `A` stays non-singular.
Eigen::JacobiSVD<Matrix3x3> svd_of(const Matrix3x3& A)
{
    return Eigen::JacobiSVD<Matrix3x3>(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
}

/// A random admissible `A` (see the comment above).
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
    // keep the singular values in a comfortable window
    auto  sv = svd_of(A);
    double smin = sv.singularValues()(2);
    double smax = sv.singularValues()(0);
    if(smin < 0.35)
        A = A * (0.35 / smin);
    sv = svd_of(A);
    smax = sv.singularValues()(0);
    if(smax > 2.5)
        A = A * (2.5 / smax);
    return A;
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

#define T63A_CHECK(counter, cond) \
    do                            \
    {                             \
        ++(counter).checks;       \
        CHECK(cond);              \
    } while(false)
}  // namespace

// ===========================================================================
//  the oracles
// ===========================================================================
namespace
{
// ---------------------------------------------------------------------------
//  ARAP: `E = kappa ||A - R||_F^2`, `R = U V^T` (the derivation is in the file
//  comment).  `R` is unique whenever `sigma_i` are distinct and `det A > 0`, which
//  every state this file feeds it satisfies -- the two SVDs (Eigen's JacobiSVD and
//  lc's one-sided Jacobi, `lps::svd`) then agree on `R` and differ only inside the
//  degenerate frames, where `R` itself is ambiguous.
// ---------------------------------------------------------------------------
struct ArapLocal
{
    double    E = 0.0;
    Vector9   G;  // row-major: G(3i + j) = dE/dA(i, j) (the reference's dEdq order)
    Matrix3x3 R = Matrix3x3::Identity();
};

Matrix3x3 polar_R(const Matrix3x3& A)
{
    auto sv = svd_of(A);
    return sv.matrixU() * sv.matrixV().transpose();
}

ArapLocal arap_local(double kappa, const Matrix3x3& A)
{
    ArapLocal out;
    out.R = polar_R(A);
    out.E = kappa * (A - out.R).squaredNorm();
    const Matrix3x3 dEdA = 2.0 * kappa * (A - out.R);
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            out.G(3 * i + j) = dEdA(i, j);
    return out;
}

/// The transpose permutation `P` on the 9-dim *column-major* `vec` basis
/// (`vec(M)(3c + r) = M(r,c)`): `P vec(M) = vec(M^T)`.
Dense9 transpose_permutation()
{
    Dense9 P = Dense9::Zero();
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            P(3 * c + r, 3 * r + c) = 1.0;
    return P;
}

/// The orthogonal projector onto the *antisymmetric* (rotation) directions,
/// `P_a = (I - P)/2` -- an independent spelling of the reference's three twist modes:
/// `sum_i vec(T_i) vec(T_i)^T / 2 == P_a`, whose eigenvalues are `1` (x3, rotations)
/// and `0` (x6).
Dense9 antisym_projector()
{
    return 0.5 * (Dense9::Identity() - transpose_permutation());
}

/// The ARAP Hessian at an isotropic stretch `A = s I` (`s > 0`),
/// `H = 2 kappa (I - (1/s) P_a)` -- derived here from the *physics* of the model
/// (`E` is flat along the `R`-twisted rotation directions and quadratic across them)
/// and verified against an independent Eigen transcription + FD of the reference's own
/// gradient before this file was written.  It needs `R = I` (the flat directions are
/// `Omega R`, so they are the plain antisymmetric ones only when `R = I`), which is
/// exactly what an isotropic stretch gives.
Dense9 arap_hessian_iso(double kappa, double s)
{
    return 2.0 * kappa * (Dense9::Identity() - (1.0 / s) * antisym_projector());
}

/// The kernel-level (`* volume * dt^2`, 12x12) ARAP quantities for a `q` whose linear
/// part is `s I`.
struct KernelLocal
{
    double      E = 0.0;
    Dense12v    G = Dense12v::Zero();
    Dense12     H = Dense12::Zero();
};

/// Embed a 9-dim local quantity into the 12-dof state (`[t; A]`, the translation block
/// stays zero) and apply the kernel's `volume * dt^2` factor -- the reference's
/// `G.segment<9>(3) = G9 * Vdt2; H.block<9,9>(3,3) = H9x9 * Vdt2;`.
KernelLocal embed_local(double E_local, const Vector9& G9, const Dense9& H9,
                        double volume, double dt)
{
    const double Vdt2 = volume * dt * dt;
    KernelLocal out;
    out.E = E_local * Vdt2;
    for(int k = 0; k < 9; ++k)
        out.G(3 + k) = G9(k) * Vdt2;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
            out.H(3 + r, 3 + c) = H9(r, c) * Vdt2;
    return out;
}

// ---------------------------------------------------------------------------
//  ortho potential: `E = kappa ||C||_F^2`, `C = A^T A - I`, `dE/dA = 4 kappa A C`, and
//  the exact Hessian `H[(i,j),(r,s)] = 4 kappa [delta_ir C(s,j) + A(i,s)A(r,j)
//  + (A A^T)(i,r) delta_js]` in the same row-major index order the generated file uses
//  (`m = 3i + j`).  Derivation: differentiate `dE/dA = 4 kappa A C` along the
//  elementary direction `A -> A + e E_rs`; `dC = E_sr A + A^T E_rs`.  Both the law and
//  the Hessian were checked against the *generated expression* (a python re-parse of
//  `sym/ortho_potential.inl`, the §7.17 rule 30c technique) before this file existed:
//  E 2.5e-16, G 3.3e-16, H 4.3e-16 relative over 20 random states.
// ---------------------------------------------------------------------------
struct OrthoLocal
{
    double  E = 0.0;
    Vector9 G;
    Dense9  H;
};

OrthoLocal ortho_local(double kappa, const Matrix3x3& A)
{
    const Matrix3x3 C   = A.transpose() * A - Matrix3x3::Identity();
    const Matrix3x3 dEdA = 4.0 * kappa * A * C;
    const Matrix3x3 AAt = A * A.transpose();

    OrthoLocal out;
    out.E = kappa * C.squaredNorm();
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            out.G(3 * i + j) = dEdA(i, j);
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            for(int r = 0; r < 3; ++r)
                for(int s = 0; s < 3; ++s)
                    out.H(3 * i + j, 3 * r + s) =
                        4.0 * kappa
                        * ((i == r ? C(s, j) : 0.0) + A(i, s) * A(r, j)
                           + (j == s ? AAt(i, r) : 0.0));
    return out;
}

// ---------------------------------------------------------------------------
//  the ported generic routines, called exactly like the two kernels do (host, S = double)
// ---------------------------------------------------------------------------
double arap_E_host(double kappa, double dt, double volume, const Vector12& q)
{
    return gg::abd_arap_detail::body_energy<double>(kappa, dt, volume, gg::to_luisa(q));
}

Vector12 arap_G_host(double kappa, double dt, double volume, const Vector12& q)
{
    return gg::to_eigen(gg::abd_arap_detail::body_gradient<double>(kappa, dt, volume, gg::to_luisa(q)));
}

Matrix12x12 arap_H_host(double kappa, double dt, double volume, const Vector12& q)
{
    return gg::to_eigen(gg::abd_arap_detail::body_hessian<double>(kappa, dt, volume, gg::to_luisa(q)));
}

double ortho_E_host(double kappa, double dt, double volume, const Vector12& q)
{
    return gg::abd_ortho_potential_detail::body_energy<double>(kappa, dt, volume, gg::to_luisa(q));
}

Vector12 ortho_G_host(double kappa, double dt, double volume, const Vector12& q)
{
    return gg::to_eigen(
        gg::abd_ortho_potential_detail::body_gradient<double>(kappa, dt, volume, gg::to_luisa(q)));
}

Matrix12x12 ortho_H_host(double kappa, double dt, double volume, const Vector12& q)
{
    return gg::to_eigen(
        gg::abd_ortho_potential_detail::body_hessian<double>(kappa, dt, volume, gg::to_luisa(q)));
}

Matrix12x12 ortho_H_unclamped_host(double kappa, double dt, double volume, const Vector12& q)
{
    return gg::to_eigen(gg::abd_ortho_potential_detail::body_hessian_unclamped<double>(
        kappa, dt, volume, gg::to_luisa(q)));
}

// ---------------------------------------------------------------------------
//  central finite differences over the 12 element dofs
// ---------------------------------------------------------------------------
/// The 12x12 embedding of the transpose permutation `P` (`transpose_permutation()`
/// acts on the nine `A`-dofs; the three translation dofs are untouched).  It is needed
/// because of the FINDING pinned at its first call site: the reference's ABD ARAP
/// Hessian is spelled in the *column-major* `vec` basis while its `dEdq` writes the
/// dof (row-major) order, so the shipped Hessian is the `P`-conjugate of the true one.
[[nodiscard]] Matrix12x12 permute_dofs(const Matrix12x12& H)
{
    const Dense9 P   = transpose_permutation();
    Matrix12x12  out = H;
    out.block<9, 9>(3, 3) = P * Dense9(H.block<9, 9>(3, 3)) * P;
    return out;
}

constexpr double kFdH = 1e-6;

template <typename Eval>
Vector12 fd_gradient(const Eval& eval, const Vector12& q, double h)
{
    Vector12 g;
    for(int k = 0; k < 12; ++k)
    {
        Vector12 qp = q, qm = q;
        qp(k) += h;
        qm(k) -= h;
        g(k) = (eval(qp) - eval(qm)) / (2 * h);
    }
    return g;
}

template <typename Grad>
Matrix12x12 fd_hessian(const Grad& grad, const Vector12& q, double h)
{
    Matrix12x12 H;
    for(int k = 0; k < 12; ++k)
    {
        Vector12 qp = q, qm = q;
        qp(k) += h;
        qm(k) -= h;
        H.col(k) = (grad(qp) - grad(qm)) / (2 * h);
    }
    return 0.5 * (H + H.transpose());
}

/// `max |A - B| / max(1, |A|_max, |B|_max)` -- the scale-free deviation the FEM rows
/// calibrate against.
template <typename A, typename B>
double dev(const A& a, const B& b)
{
    const double scale = std::max({1.0, max_abs(a), max_abs(b)});
    return max_abs(a - b) / scale;
}

/// The exact eigenvalue clamp (`max(lambda, 0)` on a symmetric 9x9): the reference's
/// `make_spd` contract, computed independently with Eigen.
Dense9 exact_spd(const Dense9& H)
{
    Eigen::SelfAdjointEigenSolver<Dense9> es(H);
    Vector9                               w = es.eigenvalues();
    for(int i = 0; i < 9; ++i)
        w(i) = std::max(w(i), 0.0);
    return es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
}

/// The sorted eigenvalue vector of a symmetric 9x9.
Vector9 sorted_eigs(const Dense9& H)
{
    Eigen::SelfAdjointEigenSolver<Dense9> es(H);
    return es.eigenvalues();
}
}  // namespace

// ===========================================================================
//  the probe: the built ABD systems of one scene
// ===========================================================================
class T63AProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::AffineBodyDynamics* abd        = nullptr;
    inline static gg::ABDLinearSubsystem* abd_linear = nullptr;
    inline static gg::GlobalLinearSystem* gls        = nullptr;
    inline static gg::ARAP*               arap       = nullptr;
    inline static gg::OrthoPotential*     ortho      = nullptr;
    inline static T63AProbe*              self       = nullptr;

    /// The scene's `dt` -- the *same* attribute `ABDLinearSubsystem::Impl::_assemble_kinetic_shape`
    /// reads (`world().scene().config().find<Float>("dt")`).
    [[nodiscard]] Float dt()
    {
        auto attr = world().scene().config().find<Float>("dt");
        UIPC_ASSERT(attr, "T63AProbe: the scene config has no `dt`");
        // the slot's own const `view()` (the same spelling the ABD subsystem uses:
        // `dt_attr->view()[0]`)
        return attr->view()[0];
    }

  private:
    void do_build() override
    {
        self       = this;
        abd        = &require<gg::AffineBodyDynamics>();
        gls        = &require<gg::GlobalLinearSystem>();
        abd_linear = find<gg::ABDLinearSubsystem>();
        arap       = find<gg::ARAP>();
        ortho      = find<gg::OrthoPotential>();
    }
};
REGISTER_SIM_SYSTEM(T63AProbe);

namespace
{
// ---------------------------------------------------------------------------
//  scene helpers
// ---------------------------------------------------------------------------
constexpr Float kL     = 0.5;
constexpr Float kRho   = 1000.0;
constexpr Float kKappa = 1.0e8;

/// uid 1 = the frontend's "OrthoPotential", uid 2 = "ARAP" (both written by
/// `uipc::constitution::AffineBodyConstitution`; the test overwrites the meta to pick
/// which one a body carries).
constexpr U64 kOrthoUID = 1ull;
constexpr U64 kArapUID  = 2ull;

std::array<Vector3, 8> cube_vertices(Float L)
{
    return {Vector3{0, 0, 0} * L, Vector3{1, 0, 0} * L, Vector3{0, 1, 0} * L, Vector3{1, 1, 0} * L,
            Vector3{0, 0, 1} * L, Vector3{1, 0, 1} * L, Vector3{0, 1, 1} * L, Vector3{1, 1, 1} * L};
}

/// The six-tet decomposition of `[0,L]^3` (vertex index = `x + 2y + 4z`, the fan from
/// vertex 0 to vertex 7) -- the *positively oriented* one the other ABD test files use;
/// a wrong orientation makes the frontend's `compute_mesh_volume` negative and
/// `apply_to` throws (measured while writing this file).
std::array<Vector4i, 6> cube_tets()
{
    return {Vector4i{0, 1, 3, 7}, Vector4i{0, 3, 2, 7}, Vector4i{0, 2, 6, 7},
            Vector4i{0, 6, 4, 7}, Vector4i{0, 4, 5, 7}, Vector4i{0, 5, 1, 7}};
}

/// One ABD body: an `L`-cube of density `rho`, `kappa`, the affine state `A`/`t`, and
/// the *uid* that routes it into a backend constitution's group.
SimplicialComplex make_body(Float             kappa,
                            const Matrix3x3&  A,
                            const Vector3&    t,
                            U64               uid,
                            const Matrix3x3&  rest_map = Matrix3x3::Identity(),
                            const Matrix3x3&  V        = Matrix3x3::Zero(),
                            const Vector3&    v        = Vector3::Zero())
{
    auto vs = cube_vertices(kL);
    if(rest_map != Matrix3x3::Identity())
    {
        for(auto& v : vs)
            v = rest_map * v;
    }
    auto ts = cube_tets();
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
    // the initial velocity (`builtin::velocity`, the frontend's own attribute): with a
    // *zero* gravity the BDF1 prediction is `q_tilde = q + q_v dt`, so an isotropic
    // velocity keeps an isotropic stretch an isotropic stretch (case 5 relies on that
    // to keep a closed-form Hessian).
    view(*sc.instances().find<Matrix4x4>(builtin::velocity))[0] = make_transform(V, v);
    return sc;
}

Json abd_scene_config()
{
    auto config                          = Scene::default_config();
    config["sanity_check"]["enable"]     = false;
    config["contact"]["enable"]          = false;
    config["integrator"]["type"]         = std::string{"bdf1"};
    // zero gravity: the BDF1 prediction is then a pure `q + q_v dt`, which keeps the
    // scene's closed forms exact (see the scene case)
    config["gravity"]                    = Vector3{0.0, 0.0, 0.0};
    return config;
}
}  // namespace

// ===========================================================================
//  1) closed forms (E/G/H against hand-derived values, the scaling laws, and the
//     "ARAP ships the unclamped Hessian" pin)
// ===========================================================================
TEST_CASE("abd_constitution_closed_forms", "[luisa][abd_constitution]")
{
    test_runtime();
    CaseScope scope{"abd_constitution_closed_forms"};
    auto&     c = scope.counter;

    const double kappa = 2.5;
    const double dt    = 0.02;
    const double vol   = 0.125;
    const double Vdt2  = vol * dt * dt;

    // ---- (A) ARAP ----------------------------------------------------------
    // (A1) rigid map: `R = A`, so `E == 0` and `dE/dA == 0` exactly; the Hessian's
    //      flat directions are the *R-twisted* rotations, so its spectrum is
    //      {0,0,0} u {2 kappa v dt^2 x 6} (a property of the model, independent of the
    //      SVD's branch).
    {
        const Matrix3x3 A =
            Eigen::AngleAxisd(0.7, Vector3{1.0, -2.0, 3.0}.normalized()).toRotationMatrix();
        const Vector12 q = q_of(A, Vector3{1.0, -0.5, 2.0});
        // `R = U V^T` is computed by an iterative SVD, so `R == A` only to round-off:
        // the *scale-relative* bound is the honest statement of "`E` is zero here"
        const double E = arap_E_host(kappa, dt, vol, q);
        T63A_CHECK(c, std::abs(E) <= 1e-14 * 3.0 * kappa * Vdt2);
        const Vector12 G = arap_G_host(kappa, dt, vol, q);
        double         worst_g = 0.0;
        for(int k = 0; k < 12; ++k)
            worst_g = std::max(worst_g, std::abs(G(k)));
        T63A_CHECK(c, worst_g <= 2e-13 * (2.0 * kappa * Vdt2));
        std::printf("[t63a] ARAP rigid: |E| %.3e (scale %.3e), |G|max %.3e (scale %.3e)//n",
                    std::abs(E), 3.0 * kappa * Vdt2, worst_g, 2.0 * kappa * Vdt2);
        const Matrix12x12 H = arap_H_host(kappa, dt, vol, q);
        const Dense9      H9(H.block<9, 9>(3, 3));
        T63A_CHECK(c, dev(H9, Dense9(H9.transpose())) == 0.0);  // exactly symmetric
        const Vector9 w = sorted_eigs(H9);
        for(int i = 0; i < 3; ++i)
            T63A_CHECK(c, std::abs(w(i)) <= 1e-12 * (2.0 * kappa * Vdt2));
        for(int i = 3; i < 9; ++i)
            T63A_CHECK(c, rel_err(w(i), 2.0 * kappa * Vdt2) <= 1e-12);
        // the translation block is identically zero (the local law does not see t)
        for(int r = 0; r < 12; ++r)
        {
            if(r >= 3 && r < 12)
                continue;
            for(int cc = 0; cc < 12; ++cc)
                T63A_CHECK(c, H(r, cc) == 0.0);
        }
    }

    // (A2) isotropic stretch `A = s I` (`s > 0`): `R = I`, so
    //      `E = 3 kappa (s-1)^2`, `dE/dA = 2 kappa (s-1) I`, and the Hessian is the
    //      *projector* form `H = 2 kappa (I_9 - (1/s) P_a)` (derivation at
    //      `arap_hessian_iso`); the kernel adds `v dt^2`.
    for(double s : {0.5, 0.8, 1.3, 2.2})
    {
        const Matrix3x3 A = s * Matrix3x3::Identity();
        const Vector12  q = q_of(A, Vector3{0.3, 0.0, -1.0});
        T63A_CHECK(c, rel_err(arap_E_host(kappa, dt, vol, q), 3.0 * kappa * (s - 1.0) * (s - 1.0) * Vdt2)
                           <= 1e-14);
        const Vector12 G = arap_G_host(kappa, dt, vol, q);
        for(int k = 0; k < 12; ++k)
        {
            const double expected = (k >= 3 && k < 12 && (k - 3) % 4 == 0)
                                        ? 2.0 * kappa * (s - 1.0) * Vdt2
                                        : 0.0;
            T63A_CHECK(c, rel_err(G(k), expected) <= 1e-14);
        }
        const Matrix12x12 H = arap_H_host(kappa, dt, vol, q);
        const Dense9      H9(H.block<9, 9>(3, 3));
        T63A_CHECK(c, dev(H9, arap_hessian_iso(kappa, s) * Vdt2) <= 1e-14);
        // the *unclamped* pin: at s = 0.5 the flat directions carry a negative
        // eigenvalue (-2 kappa v dt^2) and the shipped Hessian keeps it -- the
        // reference calls `make_spd` for the ortho potential only.
        if(s < 1.0)
        {
            const double expected_min = 2.0 * kappa * (1.0 - 1.0 / s) * Vdt2;
            T63A_CHECK(c, rel_err(min_eigenvalue(H9), expected_min) <= 1e-12);
            T63A_CHECK(c, min_eigenvalue(H9) < 0.0);
        }
    }

    // (A3) a diagonal stretch `A = diag(a,b,c)`, `a,b,c > 0`: `R = I`, so
    //      `E = kappa sum (a_i - 1)^2` and `dE/dA = 2 kappa diag(a_i - 1)`.
    {
        const Matrix3x3 A = Vector3{1.4, 0.7, 1.15}.asDiagonal();
        const Vector12  q = q_of(A, Vector3{2.0, -1.0, 0.5});
        const double    E_expected =
            kappa
            * ((1.4 - 1.0) * (1.4 - 1.0) + (0.7 - 1.0) * (0.7 - 1.0) + (1.15 - 1.0) * (1.15 - 1.0))
            * Vdt2;
        T63A_CHECK(c, rel_err(arap_E_host(kappa, dt, vol, q), E_expected) <= 1e-14);
        const Vector12 G = arap_G_host(kappa, dt, vol, q);
        const double   d[3] = {1.4, 0.7, 1.15};
        for(int k = 0; k < 12; ++k)
        {
            double expected = 0.0;
            if(k >= 3 && k < 12 && (k - 3) % 4 == 0)
                expected = 2.0 * kappa * (d[(k - 3) / 4] - 1.0) * Vdt2;
            T63A_CHECK(c, rel_err(G(k), expected) <= 1e-14);
        }
        // the same through the oracle's own polar factor (an independent Eigen SVD
        // spelling of `R`): the two agree on a positive diagonal map.
        const auto l = arap_local(kappa, A);
        T63A_CHECK(c, dev(l.R, Matrix3x3::Identity()) <= 1e-15);
        T63A_CHECK(c, rel_err(l.E * Vdt2, E_expected) <= 1e-15);
    }

    // (A4) the kernel's scaling laws: linear in `volume`, quadratic in `dt`; and the
    //      whole 12-dof law is translation invariant (only `A` enters).
    {
        const Matrix3x3 A = Vector3{1.2, 0.9, 1.05}.asDiagonal();
        const Vector12  q0 = q_of(A, Vector3::Zero());
        const Vector12  q1 = q_of(A, Vector3{3.0, -2.0, 0.75});
        const Vector12  G0 = arap_G_host(kappa, dt, vol, q0);
        const Vector12  G1 = arap_G_host(kappa, dt, vol, q1);
        T63A_CHECK(c, dev(G0, G1) == 0.0);
        T63A_CHECK(c, dev(Matrix12x12(arap_H_host(kappa, dt, vol, q0)),
                          Matrix12x12(arap_H_host(kappa, dt, vol, q1)))
                           == 0.0);
        const double E0 = arap_E_host(kappa, dt, vol, q0);
        T63A_CHECK(c, rel_err(arap_E_host(kappa, dt, 2.0 * vol, q0), 2.0 * E0) <= 1e-15);
        T63A_CHECK(c, rel_err(arap_E_host(kappa, 2.0 * dt, vol, q0), 4.0 * E0) <= 1e-15);
        T63A_CHECK(c, rel_err(arap_E_host(2.0 * kappa, dt, vol, q0), 2.0 * E0) <= 1e-15);
    }

    // ---- (B) ortho potential ------------------------------------------------
    // (B1) the law read out of the generated symbol map: with `C = A^T A - I`,
    //      `E = kappa ||C||_F^2`, `dE/dA = 4 kappa A C`, and the exact Hessian of
    //      `ortho_local`.  A hand-chosen strained state, then the *independent*
    //      analytic sibling T6.1a already ports (`abd_energy.h`, the transcription of
    //      the reference's own `abd_energy.cu`, i.e. the operator form) divided by
    //      kappa: two derivations of one function.
    {
        const Matrix3x3 A =
            Eigen::AngleAxisd(-0.4, Vector3{0.3, 1.0, -0.7}.normalized()).toRotationMatrix()
            * Vector3{1.35, 0.85, 1.1}.asDiagonal();
        const Vector12 q   = q_of(A, Vector3{-0.5, 1.5, 0.25});
        const auto     loc = ortho_local(kappa, A);

        T63A_CHECK(c, rel_err(ortho_E_host(kappa, dt, vol, q), loc.E * Vdt2) <= 1e-14);
        const Vector12 G = ortho_G_host(kappa, dt, vol, q);
        for(int k = 0; k < 3; ++k)
            T63A_CHECK(c, G(k) == 0.0);  // the translation rows stay zero
        for(int k = 0; k < 9; ++k)
            T63A_CHECK(c, rel_err(G(3 + k), loc.G(k) * Vdt2) <= 1e-14);
        const Matrix12x12 H    = ortho_H_host(kappa, dt, vol, q);
        const Matrix12x12 Hraw = ortho_H_unclamped_host(kappa, dt, vol, q);
        const Dense9      H9(H.block<9, 9>(3, 3));
        const Dense9      Hraw9(Hraw.block<9, 9>(3, 3));
        // the *unclamped* 12x12 is the hand-derived law (the generated `ddEddq` scaled);
        // the shipped one is the reference's `make_spd` clamp of that block -- which is
        // *not* a no-op here, so the two are asserted separately
        T63A_CHECK(c, dev(Hraw9, loc.H * Vdt2) <= 1e-13);
        T63A_CHECK(c, dev(H9, exact_spd(Hraw9)) <= 1e-13);
        T63A_CHECK(c, min_eigenvalue(H9) >= 0.0);
        std::printf("[t63a] ortho strained: hand-derived vs generated %.3e, shipped vs exact "
                    "clamp %.3e (min eig(raw) %.3e)//n",
                    dev(Hraw9, loc.H * Vdt2), dev(H9, exact_spd(Hraw9)),
                    min_eigenvalue(Hraw9));

        // the T6.1a analytic sibling, on the same q (Eigen-typed host signatures)
        T63A_CHECK(c, rel_err(gg::shape_energy(q) * kappa, loc.E) <= 1e-14);
        const Vector9 g_sib = gg::shape_energy_gradient(q);
        for(int k = 0; k < 9; ++k)
            T63A_CHECK(c, rel_err(g_sib(k) * kappa, loc.G(k)) <= 1e-13);
        const Dense9 h_sib(gg::shape_energy_hessian(q));
        T63A_CHECK(c, dev(h_sib * kappa, loc.H) <= 1e-13);
    }

    // (B2) `A = I` (the rest state): `C = 0`, so `E == 0`, `G == 0` and the Hessian is
    //      `4 kappa (I + P)`, i.e. eigenvalues {0,0,0} u {8 kappa v dt^2 x 6}
    //      (flat along the rotations, which the potential does not penalise).
    {
        const Vector12 q = q_of(Matrix3x3::Identity(), Vector3{0.4, -0.2, 1.0});
        T63A_CHECK(c, ortho_E_host(kappa, dt, vol, q) == 0.0);
        const Vector12 G = ortho_G_host(kappa, dt, vol, q);
        for(int k = 0; k < 12; ++k)
            T63A_CHECK(c, G(k) == 0.0);
        const Matrix12x12 H = ortho_H_host(kappa, dt, vol, q);
        const Dense9      H9(H.block<9, 9>(3, 3));
        const Vector9     w = sorted_eigs(H9);
        for(int i = 0; i < 3; ++i)
            T63A_CHECK(c, std::abs(w(i)) <= 1e-12 * (8.0 * kappa * Vdt2));
        for(int i = 3; i < 9; ++i)
            T63A_CHECK(c, rel_err(w(i), 8.0 * kappa * Vdt2) <= 1e-12);
        T63A_CHECK(c, dev(H9, ortho_local(kappa, Matrix3x3::Identity()).H * Vdt2) <= 1e-14);
    }

    // (B3) an *orthogonal* map (`A = R`, any orientation, including a reflection):
    //      `A^T A = I`, so `E == 0` and `G == 0` **exactly** -- the potential measures
    //      the orthogonality defect only, it does not see the orientation.  The
    //      Hessian does see it (`H = 4 kappa (A (x) A + I)`), and stays finite.
    {
        const Matrix3x3 Rrot =
            Eigen::AngleAxisd(1.1, Vector3{0.0, 0.0, 1.0}.normalized()).toRotationMatrix();
        const Matrix3x3 refl = Vector3{-1.0, 1.0, 1.0}.asDiagonal();
        for(const Matrix3x3& A : {Rrot, refl})
        {
            const Vector12 q = q_of(A, Vector3{0.7, 0.7, -0.3});
            T63A_CHECK(c, ortho_E_host(kappa, dt, vol, q) == 0.0);
            const Vector12 G = ortho_G_host(kappa, dt, vol, q);
            for(int k = 0; k < 12; ++k)
                T63A_CHECK(c, G(k) == 0.0);
            const Matrix12x12 H = ortho_H_host(kappa, dt, vol, q);
            const Dense9 H9(H.block<9, 9>(3, 3));
            T63A_CHECK(c, all_finite(H9));
            T63A_CHECK(c, dev(H9, ortho_local(kappa, A).H * Vdt2) <= 1e-13);
            T63A_CHECK(c, min_eigenvalue(H9) >= -1e-12 * max_abs(H9));
        }
    }

    // (B4) scaling laws + translation invariance (the local law reads `A` only, and
    //      the kernel factor is `volume * dt^2`).
    {
        const Matrix3x3 A  = Vector3{1.3, 0.75, 1.0}.asDiagonal();
        const Vector12  q0 = q_of(A, Vector3::Zero());
        const Vector12  q1 = q_of(A, Vector3{-1.0, 2.0, 0.5});
        T63A_CHECK(c, ortho_E_host(kappa, dt, vol, q0) == ortho_E_host(kappa, dt, vol, q1));
        T63A_CHECK(c, dev(ortho_G_host(kappa, dt, vol, q0),
                          ortho_G_host(kappa, dt, vol, q1))
                           == 0.0);
        const double E0 = ortho_E_host(kappa, dt, vol, q0);
        T63A_CHECK(c, rel_err(ortho_E_host(kappa, dt, 2.0 * vol, q0), 2.0 * E0) <= 1e-15);
        T63A_CHECK(c, rel_err(ortho_E_host(kappa, 2.0 * dt, vol, q0), 4.0 * E0) <= 1e-15);
        T63A_CHECK(c, rel_err(ortho_E_host(3.0 * kappa, dt, vol, q0), 3.0 * E0) <= 1e-15);
        // a zero *volume* zeroes the kernel energy while the local law stays `3 kappa`
        // (the same "the kernel's own factor, not the density" finding as the FEM
        // springs, §7.17 rule 33)
        T63A_CHECK(c, ortho_E_host(kappa, dt, 0.0, q0) == 0.0);
        T63A_CHECK(c, ortho_local(kappa, Matrix3x3::Zero()).E == 3.0 * kappa);
    }

    std::printf("[t63a] %s: %d checks/n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  2) central finite differences over the 12 element dofs, the Hessian's symmetry,
//     the eigenvalue-clamp identity (ortho) and the absence of a clamp (ARAP)
// ===========================================================================
namespace
{
lps::LMat<double, 9, 9> lm9(const Dense9& m)
{
    lps::LMat<double, 9, 9> out;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
            out(r, c) = m(r, c);
    return out;
}
}  // namespace

TEST_CASE("abd_constitution_finite_difference", "[luisa][abd_constitution]")
{
    test_runtime();
    CaseScope scope{"abd_constitution_finite_difference"};
    auto&     c = scope.counter;

    constexpr int kStates = 40;

    double worst_arap_g = 0.0, worst_arap_h6 = 0.0, worst_arap_h4 = 0.0;
    double worst_arap_h3 = 0.0, worst_arap_h2 = 0.0, worst_arap_sym = 0.0;
    double worst_arap_h6_plain = 0.0;
    double worst_ortho_g = 0.0, worst_ortho_h6 = 0.0, worst_ortho_h4 = 0.0;
    double worst_ortho_h3 = 0.0, worst_ortho_h2 = 0.0, worst_ortho_sym = 0.0;
    double worst_ortho_clamp = 0.0, worst_ortho_clamp20 = 0.0, worst_ortho_proj = 0.0;
    int    arap_indefinite = 0, ortho_indefinite = 0, ortho_clamped_states = 0;

    for(int s = 0; s < kStates; ++s)
    {
        const Matrix3x3 A     = random_admissible_A();
        const Vector12  q     = q_of(A, Vector3{rand_uniform(-1.0, 1.0),
                                                rand_uniform(-1.0, 1.0),
                                                rand_uniform(-1.0, 1.0)});
        const double    kappa = rand_uniform(0.5, 2.0);
        const double    vol   = rand_uniform(0.3, 1.5);
        const double    dt    = rand_uniform(0.005, 0.05);

        // ---- ARAP: the shipped Hessian IS the exact (unclamped) one ----------
        {
            const Vector12 G = arap_G_host(kappa, dt, vol, q);
            const Vector12 Gfd =
                fd_gradient([&](const Vector12& qq) { return arap_E_host(kappa, dt, vol, qq); },
                            q,
                            kFdH);
            worst_arap_g = std::max(worst_arap_g, dev(G, Gfd));

            const Matrix12x12 H = arap_H_host(kappa, dt, vol, q);
            auto grad = [&](const Vector12& qq) { return arap_G_host(kappa, dt, vol, qq); };
            const Matrix12x12 Hfd6 = permute_dofs(fd_hessian(grad, q, 1e-6));
            const Matrix12x12 Hfd4 = permute_dofs(fd_hessian(grad, q, 1e-4));
            const Matrix12x12 Hfd3 = permute_dofs(fd_hessian(grad, q, 1e-3));
            const Matrix12x12 Hfd2 = permute_dofs(fd_hessian(grad, q, 1e-2));
            // the same finite difference WITHOUT the conjugation: measurably wrong, which is
            // what makes the check above non-vacuous
            worst_arap_h6_plain = std::max(worst_arap_h6_plain, dev(H, fd_hessian(grad, q, 1e-6)));
            worst_arap_h6 = std::max(worst_arap_h6, dev(H, Hfd6));
            worst_arap_h4 = std::max(worst_arap_h4, dev(H, Hfd4));
            worst_arap_h3 = std::max(worst_arap_h3, dev(H, Hfd3));
            worst_arap_h2 = std::max(worst_arap_h2, dev(H, Hfd2));
            worst_arap_sym =
                std::max(worst_arap_sym, max_abs(H - Matrix12x12(H.transpose())) / max_abs(H));
            if(min_eigenvalue(Dense9(H.block<9, 9>(3, 3))) < -1e-12 * max_abs(H))
                ++arap_indefinite;
        }

        // ---- ortho: FD against the *unclamped* analytic Hessian, plus the clamp --
        {
            const Vector12 G = ortho_G_host(kappa, dt, vol, q);
            const Vector12 Gfd =
                fd_gradient([&](const Vector12& qq) { return ortho_E_host(kappa, dt, vol, qq); },
                            q,
                            kFdH);
            worst_ortho_g = std::max(worst_ortho_g, dev(G, Gfd));

            const Matrix12x12 Hraw = ortho_H_unclamped_host(kappa, dt, vol, q);
            const Matrix12x12 H    = ortho_H_host(kappa, dt, vol, q);
            auto grad = [&](const Vector12& qq) { return ortho_G_host(kappa, dt, vol, qq); };
            const Matrix12x12 Hfd6 = fd_hessian(grad, q, 1e-6);
            const Matrix12x12 Hfd4 = fd_hessian(grad, q, 1e-4);
            const Matrix12x12 Hfd3 = fd_hessian(grad, q, 1e-3);
            const Matrix12x12 Hfd2 = fd_hessian(grad, q, 1e-2);
            worst_ortho_h6 = std::max(worst_ortho_h6, dev(Hraw, Hfd6));
            worst_ortho_h4 = std::max(worst_ortho_h4, dev(Hraw, Hfd4));
            worst_ortho_h3 = std::max(worst_ortho_h3, dev(Hraw, Hfd3));
            worst_ortho_h2 = std::max(worst_ortho_h2, dev(Hraw, Hfd2));
            worst_ortho_sym =
                std::max(worst_ortho_sym, max_abs(H - Matrix12x12(H.transpose())) / max_abs(H));

            // the hand-derived 9x9 formula vs the *generated* ddEddq (the scaled,
            // embedded 9x9 block is `kappa ||C||^2`'s exact Hessian times `v dt^2`)
            const double    Vdt2 = vol * dt * dt;
            const Dense9    Hraw9(Hraw.block<9, 9>(3, 3));
            worst_ortho_proj =
                std::max(worst_ortho_proj,
                         dev(Hraw9, ortho_local(kappa, A).H * Vdt2));

            // the reference's `make_spd(H9x9)` clamp, in the 9x9 F-block *before* the
            // `v dt^2` scaling and the embedding: the shipped 12x12's 9x9 block must
            // equal the exact eigenvalue clamp of the unclamped one (up to the Jacobi
            // sweep count -- `kMakeSpdSweeps` is 8, measured below)
            const Dense9 shipped(H.block<9, 9>(3, 3) / Vdt2);
            const Dense9 exact = exact_spd(Dense9(Hraw9 / Vdt2));
            worst_ortho_clamp = std::max(worst_ortho_clamp, dev(shipped, exact));

            // the sweep count: 8 (the local `kMakeSpdSweeps`) vs the layer default (20)
            // vs the exact clamp
            const auto raw9 = lm9(Hraw9);
            auto       h8   = raw9;
            auto       h20  = raw9;
            lps::make_spd_inplace(h8, gg::abd_ortho_potential_detail::kMakeSpdSweeps);
            lps::make_spd_inplace(h20, 20);
            worst_ortho_clamp20 =
                std::max(worst_ortho_clamp20, dev(to_eigen9(h8), to_eigen9(h20)));

            const double eig_min_raw = min_eigenvalue(Hraw9);
            if(eig_min_raw < -1e-12 * max_abs(Hraw9))
                ++ortho_indefinite;
            if(eig_min_raw < -1e-12 * max_abs(Hraw9)
               && max_abs(Dense9(H.block<9, 9>(3, 3))) > 1e-12 * max_abs(Hraw9))
                ++ortho_clamped_states;
        }
    }

    std::printf("[t63a] FD ARAP : grad %3.1e  H(h=1e-6) %3.1e  (1e-4) %3.1e  (1e-3) %3.1e  "
                "(1e-2) %3.1e  asym %3.1e  indefinite %d/%d\n",
                worst_arap_g, worst_arap_h6, worst_arap_h4, worst_arap_h3, worst_arap_h2,
                worst_arap_sym, arap_indefinite, kStates);
    std::printf("[t63a] FD ortho: grad %3.1e  H(h=1e-6) %3.1e  (1e-4) %3.1e  (1e-3) %3.1e  "
                "(1e-2) %3.1e  asym %3.1e  indefinite %d/%d  clamp 8v20 %3.1e\n",
                worst_ortho_g, worst_ortho_h6, worst_ortho_h4, worst_ortho_h3, worst_ortho_h2,
                worst_ortho_sym, ortho_indefinite, kStates, worst_ortho_clamp20);
    std::printf("[t63a] ARAP unpermuted FD Hessian deviation (the reference-basis finding): %3.1e\n",
                worst_arap_h6_plain);
    std::printf("[t63a] ortho hand-derived 9x9 vs generated (scaled) %3.1e; shipped vs exact "
                "eigen clamp %3.1e; clamped-and-not-a-no-op %d/%d\n",
                worst_ortho_proj, worst_ortho_clamp, ortho_clamped_states, kStates);

    // the gradient is a true derivative of the *shipped* energy
    T63A_CHECK(c, worst_arap_g <= 1e-9);
    T63A_CHECK(c, worst_ortho_g <= 1e-9);
    // the Hessian is the true derivative of the shipped gradient: h = 1e-6 (the brief's
    // step) and h = 1e-4 (the measured round-off optimum of the FEM rows)
    T63A_CHECK(c, worst_arap_h6 <= 1e-8);
    T63A_CHECK(c, worst_arap_h4 <= 1e-6);
    // the permutation is real: the unpermuted FD comparison is off by ~1e-2
    T63A_CHECK(c, worst_arap_h6_plain > 1e-4);
    T63A_CHECK(c, worst_ortho_h6 <= 1e-4);
    T63A_CHECK(c, worst_ortho_h4 <= 1e-6);
    // ... and the sweep is not flat: the truncation error grows with h
    T63A_CHECK(c, worst_arap_h2 > worst_arap_h3);
    T63A_CHECK(c, worst_ortho_h2 > worst_ortho_h3);
    // symmetry (the ARAP Hessian is a sum of outer products, the ortho one is generated
    // symmetric statement by statement: both are symmetric to round-off)
    T63A_CHECK(c, worst_arap_sym <= 1e-15);
    T63A_CHECK(c, worst_ortho_sym <= 1e-15);
    // the hand-derived ortho formula IS the generated Hessian
    T63A_CHECK(c, worst_ortho_proj <= 1e-13);
    // the reference's clamp is the exact eigenvalue clamp, and it is not a no-op
    T63A_CHECK(c, worst_ortho_clamp <= 1e-13);
    T63A_CHECK(c, ortho_clamped_states > 0);
    T63A_CHECK(c, worst_ortho_clamp20 <= 1e-15);
    // the ARAP Hessian is *indefinite* in a good fraction of the states and is shipped
    // unclamped (the reference only clamps the ortho potential)
    T63A_CHECK(c, arap_indefinite > 0);

    std::printf("[t63a] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  3) degenerate inputs, pinned from the source (neither constitution guards its
//     input; the only guard in either is the SVD's own completion)
// ===========================================================================
TEST_CASE("abd_constitution_degenerate_inputs", "[luisa][abd_constitution]")
{
    test_runtime();
    CaseScope scope{"abd_constitution_degenerate_inputs"};
    auto&     c = scope.counter;

    const double kappa = 3.0;
    const double dt    = 0.01;
    const double vol   = 0.1;
    const double Vdt2  = vol * dt * dt;

    // ---- ARAP --------------------------------------------------------------
    // (D1) zero deformation `A = 0`: `R = U V^T` is a rotation (the SVD completes the
    //      rank-deficient frame), so `E = kappa ||R||_F^2 = 3 kappa` and `G = -2 kappa R`
    //      are finite, while `ARAP_Hessian` divides by `sigma_0 + sigma_1 = 0` and the
    //      Hessian is not finite.  The port does not "repair" it.
    {
        const Vector12 q = q_of(Matrix3x3::Zero(), Vector3{1.0, 2.0, 3.0});
        const double   E = arap_E_host(kappa, dt, vol, q);
        const Vector12 G = arap_G_host(kappa, dt, vol, q);
        const Matrix12x12 H = arap_H_host(kappa, dt, vol, q);
        std::printf("[t63a] ARAP A=0: E %.6e (3 kappa v dt^2 = %.6e)  G finite %d  H finite %d\n",
                    E, 3.0 * kappa * Vdt2, static_cast<int>(all_finite(G)),
                    static_cast<int>(all_finite(H)));
        T63A_CHECK(c, std::isfinite(E));
        T63A_CHECK(c, rel_err(E, 3.0 * kappa * Vdt2) <= 1e-14);
        T63A_CHECK(c, all_finite(G));
        T63A_CHECK(c, !all_finite(H));
    }

    // (D2) singular but signed, `A = diag(1,1,0)`: the SVD's completion gives
    //      `U = V = I` (the source: every `gamma` is 0 so no rotation happens, and the
    //      third column vanishes), so `R = I` exactly and
    //      `E = kappa ||diag(0,0,-1)||^2 = kappa`, `dE/dA = -2 kappa e_2 e_2^T`.
    //      The Hessian's denominators are `sigma_0+sigma_1 = 2`, `sigma_1+sigma_2 = 1`,
    //      `sigma_0+sigma_2 = 1`: finite, but *indefinite* (the flat rotation direction
    //      of `sigma_2 = 0` has eigenvalue `-2 kappa`), and unclamped.
    {
        const Matrix3x3 A = Vector3{1.0, 1.0, 0.0}.asDiagonal();
        const Vector12  q = q_of(A, Vector3{-1.0, 0.5, 0.0});
        const double    E = arap_E_host(kappa, dt, vol, q);
        const Vector12  G = arap_G_host(kappa, dt, vol, q);
        const Matrix12x12 H = arap_H_host(kappa, dt, vol, q);
        std::printf("[t63a] ARAP diag(1,1,0): E %.6e (kappa v dt^2 = %.6e)  G(8) %.6e "
                    "(-2 kappa v dt^2 = %.6e)  min eig(H9) %.6e\n",
                    E, kappa * Vdt2, G(11), -2.0 * kappa * Vdt2,
                    min_eigenvalue(Dense9(H.block<9, 9>(3, 3))));
        T63A_CHECK(c, rel_err(E, kappa * Vdt2) <= 1e-14);
        T63A_CHECK(c, all_finite(G));
        T63A_CHECK(c, rel_err(G(11), -2.0 * kappa * Vdt2) <= 1e-14);
        T63A_CHECK(c, all_finite(H));
        T63A_CHECK(c, min_eigenvalue(Dense9(H.block<9, 9>(3, 3))) < 0.0);
    }

    // (D3) a *reflection* `A = diag(-1,1,1)` (`det A < 0`): the signed-SVD convention
    //      absorbs the sign into `sigma_2`, so `U = diag(-1,1,-1)`, `V = I` (the source's
    //      own branch: `u_0 = B.col(0)/|B.col(0)|`, `u_1 = B.col(1)`, `u_2 = u_0 x u_1`)
    //      and `R = diag(-1,1,-1)`: `E = kappa ||diag(0,0,2)||^2 = 4 kappa` and
    //      `G = 2 kappa diag(0,0,2)` are finite.  `sigma_1 + sigma_2 = 1 + (-1) = 0`
    //      makes the Hessian non-finite.
    {
        const Matrix3x3 A = Vector3{-1.0, 1.0, 1.0}.asDiagonal();
        const Vector12  q = q_of(A, Vector3{0.0, 0.0, 0.0});
        const double    E = arap_E_host(kappa, dt, vol, q);
        const Vector12  G = arap_G_host(kappa, dt, vol, q);
        const Matrix12x12 H = arap_H_host(kappa, dt, vol, q);
        std::printf("[t63a] ARAP reflection: E %.6e (4 kappa v dt^2 = %.6e)  G(11) %.6e "
                    "(4 kappa v dt^2 = %.6e)  H finite %d\n",
                    E, 4.0 * kappa * Vdt2, G(11), 4.0 * kappa * Vdt2,
                    static_cast<int>(all_finite(H)));
        T63A_CHECK(c, std::isfinite(E));
        T63A_CHECK(c, rel_err(E, 4.0 * kappa * Vdt2) <= 1e-13);
        T63A_CHECK(c, all_finite(G));
        T63A_CHECK(c, rel_err(G(11), 4.0 * kappa * Vdt2) <= 1e-13);
        T63A_CHECK(c, !all_finite(H));
    }

    // ---- ortho potential ---------------------------------------------------
    // (D4) the potential is a *polynomial* in the 9 dofs: E/G/H are finite in every
    //      state (there is no division anywhere in the generated file), and the exact
    //      values at the same degenerate states are hand-derivable:
    //        `A = 0`            : `C = -I`, `E = 3 kappa`, `G = 0`, `H = -4 kappa I`
    //                             (negative definite -- the reason the reference clamps)
    //        `A = diag(1,1,0)`  : `C = diag(0,0,-1)`, `E = kappa`, `G = 4 kappa A C = 0`
    //        `A = a reflection` : `A^T A = I`, `E = 0`, `G = 0` exactly
    {
        const Matrix3x3 A0  = Matrix3x3::Zero();
        const Vector12  q0  = q_of(A0, Vector3{1.0, -1.0, 0.5});
        const double    E0  = ortho_E_host(kappa, dt, vol, q0);
        const Vector12  G0  = ortho_G_host(kappa, dt, vol, q0);
        const Matrix12x12 H0 = ortho_H_host(kappa, dt, vol, q0);
        const Matrix12x12 H0raw = ortho_H_unclamped_host(kappa, dt, vol, q0);
        T63A_CHECK(c, rel_err(E0, 3.0 * kappa * Vdt2) <= 1e-14);
        for(int k = 0; k < 12; ++k)
            T63A_CHECK(c, G0(k) == 0.0);
        T63A_CHECK(c, dev(Dense9(H0raw.block<9, 9>(3, 3)), -4.0 * kappa * Vdt2 * Dense9::Identity()) <= 1e-14);
        T63A_CHECK(c, min_eigenvalue(Dense9(H0raw.block<9, 9>(3, 3))) < 0.0);
        // the shipped one is the clamp of a negative-definite Hessian: *zero*
        T63A_CHECK(c, all_finite(H0));
        T63A_CHECK(c, max_abs(H0) <= 1e-13 * max_abs(H0raw));
        std::printf("[t63a] ortho A=0: E %.6e (3 kappa v dt^2 = %.6e)  |H_raw| %.6e  "
                    "|H_shipped| %.6e  min eig(raw) %.6e\n",
                    E0, 3.0 * kappa * Vdt2, max_abs(H0raw), max_abs(H0),
                    min_eigenvalue(Dense9(H0raw.block<9, 9>(3, 3))));

        const Matrix3x3 As = Vector3{1.0, 1.0, 0.0}.asDiagonal();
        const Vector12  qs = q_of(As, Vector3::Zero());
        T63A_CHECK(c, rel_err(ortho_E_host(kappa, dt, vol, qs), kappa * Vdt2) <= 1e-14);
        const Vector12 Gs = ortho_G_host(kappa, dt, vol, qs);
        for(int k = 0; k < 12; ++k)
            T63A_CHECK(c, Gs(k) == 0.0);
        T63A_CHECK(c, all_finite(ortho_H_host(kappa, dt, vol, qs)));

        const Matrix3x3 Ar = Vector3{-1.0, 1.0, 1.0}.asDiagonal();
        const Vector12  qr = q_of(Ar, Vector3{2.0, 0.0, -1.0});
        T63A_CHECK(c, ortho_E_host(kappa, dt, vol, qr) == 0.0);
        const Vector12 Gr = ortho_G_host(kappa, dt, vol, qr);
        for(int k = 0; k < 12; ++k)
            T63A_CHECK(c, Gr(k) == 0.0);
        const Matrix12x12 Hr = ortho_H_host(kappa, dt, vol, qr);
        T63A_CHECK(c, all_finite(Hr));
        T63A_CHECK(c, dev(Dense9(Hr.block<9, 9>(3, 3)),
                          ortho_local(kappa, Ar).H * Vdt2)
                           <= 1e-13);
    }

    // (D5) a zero *volume*: the local law is untouched (it does not know the volume),
    //      the kernel's own `volume * dt^2` factor makes all three kernel quantities
    //      exactly zero -- "the kernel value, not the density, is what a test must
    //      assert" (§7.17 rule 33).
    {
        const Matrix3x3 A = Vector3{1.3, 0.8, 1.0}.asDiagonal();
        const Vector12  q = q_of(A, Vector3{0.1, 0.2, 0.3});
        T63A_CHECK(c, ortho_E_host(kappa, dt, 0.0, q) == 0.0);
        const Vector12 G = ortho_G_host(kappa, dt, 0.0, q);
        for(int k = 0; k < 12; ++k)
            T63A_CHECK(c, G(k) == 0.0);
        T63A_CHECK(c, dev(Matrix12x12(ortho_H_host(kappa, dt, 0.0, q)), Matrix12x12::Zero()) == 0.0);
        T63A_CHECK(c, ortho_local(kappa, A).E > 0.0);
        T63A_CHECK(c, arap_E_host(kappa, dt, 0.0, q) == 0.0);
    }

    std::printf("[t63a] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  4) the fan-out into ABDLinearSubsystem's assembly: the extent it reports, the
//     (body, 3x3-block) indices it writes, the assembled gradient at the body's dof
//     offset, and the assembled 12x12 block against a hand-derived dense block
//     (the `abd_linear_chain` oracle style).
//
//     The oracle here is the *plumbing*: the per-body values are the hand-derived
//     closed forms of case 1 (an ortho body in a strained state, an ARAP body at an
//     isotropic stretch -- the two states whose Hessian this file can write down in
//     closed form), placed at the hand-derived global indices.  Cases 1-3 prove that
//     the shipped routines equal those closed forms, so this case pins the *fan-out*.
// ===========================================================================
namespace
{
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

/// One ABD body of a test scene, with everything the oracle needs -- read back from
/// the scene's own attributes (`kappa`, `builtin::volume`, `builtin::dof_offset`) and
/// the dynamics' device `q`.
struct BodyInfo
{
    SizeT     dof_offset = 0;
    SizeT     body       = 0;  // = dof_offset / 12
    U64       uid        = 0;
    double    kappa      = 0.0;
    double    volume     = 0.0;
    Vector12  q;
};

std::vector<BodyInfo> scene_bodies(World& world, gg::AffineBodyDynamics& abd)
{
    const auto qs = read_vec12(abd.qs());

    std::vector<BodyInfo> out;
    auto                  visitor  = uipc::backend::WorldVisitor{world};
    auto                  geo_slots = visitor.scene().geometries();
    for(auto& slot : geo_slots)
    {
        auto sc = slot->geometry().as<SimplicialComplex>();
        if(!sc)
            continue;
        auto cuid = sc->meta().find<U64>(builtin::constitution_uid);
        if(!cuid)
            continue;
        const U64 uid = view(*cuid)[0];
        if(uid != kOrthoUID && uid != kArapUID)
            continue;

        auto doff = sc->meta().find<IndexT>(builtin::dof_offset);
        auto kap  = sc->instances().find<Float>("kappa");
        auto vol  = sc->meta().find<Float>(builtin::volume);
        REQUIRE(doff != nullptr);
        REQUIRE(kap != nullptr);
        REQUIRE(vol != nullptr);

        BodyInfo info;
        info.dof_offset = static_cast<SizeT>(view(*doff)[0]);
        info.body       = info.dof_offset / 12u;
        info.uid        = uid;
        info.kappa      = view(*kap)[0];
        info.volume     = view(*vol)[0];
        REQUIRE(info.body < qs.size());
        info.q = qs[info.body];
        out.push_back(info);
    }
    return out;
}

/// The assembled 12x12 block map: the body term only, upper-triangular in 3x3 blocks
/// (`zero_out_lower`), keyed by the *blocked* global indices `body * 4 + a`.
BlockMap expected_blocks(const std::vector<BodyInfo>& bodies, double dt)
{
    BlockMap out;
    for(const auto& b : bodies)
    {
        const auto   A    = A_of(b.q);
        const double Vdt2 = b.volume * dt * dt;

        Matrix12x12 H12 = Matrix12x12::Zero();
        if(b.uid == kOrthoUID)
        {
            // the hand-derived law *plus the reference's own eigenvalue clamp*
            // (`make_spd` on the 9x9 F-block, before the `Vdt2` scaling and the
            // embedding -- the reference's order).  Without the clamp this oracle is
            // off by the clamp's own size, which for a κ = 1e8 body is ~1.6e2 here
            // (measured): the shipped raster really is the clamped one, and the FD case
            // proves `shipped == exact_spd(unclamped)`.
            const auto loc = ortho_local(b.kappa, A);
            const Dense9 H9 = exact_spd(loc.H * Vdt2);
            for(int cc = 0; cc < 9; ++cc)
                for(int rr = 0; rr < 9; ++rr)
                    H12(3 + rr, 3 + cc) = H9(rr, cc);
        }
        else
        {
            // the ARAP body sits at an isotropic stretch (see the scene): the closed
            // form `2 kappa (I - (1/s) P_a)` is exact there
            const double s = (A(0, 0) + A(1, 1) + A(2, 2)) / 3.0;
            const Dense9 H9 = arap_hessian_iso(b.kappa, s);
            for(int cc = 0; cc < 9; ++cc)
                for(int rr = 0; rr < 9; ++rr)
                    H12(3 + rr, 3 + cc) = H9(rr, cc) * Vdt2;
        }

        // the reference's `zero_out_lower` on a diagonal block
        for(int a = 0; a < 4; ++a)
            for(int b2 = a + 1; b2 < 4; ++b2)
                H12.block<3, 3>(b2 * 3, a * 3).setZero();

        for(int a = 0; a < 4; ++a)
        {
            for(int b2 = 0; b2 < 4; ++b2)
            {
                const auto key = std::make_pair(static_cast<int>(b.body * 4 + a),
                                                static_cast<int>(b.body * 4 + b2));
                const Matrix3x3 blk = H12.block<3, 3>(a * 3, b2 * 3);
                auto            it  = out.find(key);
                if(it == out.end())
                    out.emplace(key, blk);
                else
                    it->second += blk;
            }
        }
    }
    return out;
}

/// The assembled gradient: per body the hand-derived `G12` at `body * 12 + k`.
Eigen::VectorXd expected_gradient(const std::vector<BodyInfo>& bodies, double dt)
{
    Eigen::VectorXd b = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(bodies.size()) * 12);
    for(const auto& body : bodies)
    {
        const auto   A    = A_of(body.q);
        const double Vdt2 = body.volume * dt * dt;
        Eigen::VectorXd g = Eigen::VectorXd::Zero(12);
        if(body.uid == kOrthoUID)
        {
            const auto loc = ortho_local(body.kappa, A);
            for(int k = 0; k < 9; ++k)
                g(3 + k) = loc.G(k) * Vdt2;
        }
        else
        {
            const double s = (A(0, 0) + A(1, 1) + A(2, 2)) / 3.0;
            for(int k = 0; k < 3; ++k)
                g(3 + 4 * k) = 2.0 * body.kappa * (s - 1.0) * Vdt2;
        }
        for(int k = 0; k < 12; ++k)
            b(static_cast<Eigen::Index>(body.body * 12u) + k) = g(k);
    }
    return b;
}
}  // namespace

namespace
{
/// The 2-body scene every remaining case shares: an **ARAP** body (uid 2) at an
/// isotropic stretch `A = s I` (a state whose Hessian/E/G are closed-form here) and an
/// **ortho-potential** body (uid 1) in a general strained state.  Both are labelled by
/// the uid the *frontend* writes (`kArapUID` / `kOrthoUID`), so each lands in the
/// group of the real T6.3a constitution.
struct TwoBodyScene
{
    Matrix3x3 A_arap   = 1.4 * Matrix3x3::Identity();
    Vector3   t_arap   = Vector3{0.3, -0.2, 0.5};
    Matrix3x3 A_ortho  = Matrix3x3::Identity();
    Vector3   t_ortho  = Vector3{-0.4, 0.6, 0.1};
    double    s_arap   = 1.4;

    TwoBodyScene()
    {
        A_ortho = Eigen::AngleAxisd(-0.35, Vector3{0.5, 1.0, -0.25}.normalized()).toRotationMatrix()
                  * Vector3{1.3, 0.85, 1.05}.asDiagonal();
    }

    [[nodiscard]] Scene build(const std::string& name) const
    {
        Json config                  = abd_scene_config();
        Scene scene{config};
        auto  object = scene.objects().create(name);
        // The initial velocities (`builtin::velocity`): the ARAP body's is *isotropic*
        // (`V = 0.5 I`), so with the zero gravity of `abd_scene_config()` the BDF1
        // prediction `q_tilde = q + q_v dt` keeps its state an isotropic stretch and the
        // closed-form Hessian of case 1 stays exact in the scene case; the ortho body
        // gets an anisotropic velocity (its law reads only `A`).
        object->geometries().create(
            make_body(kKappa, A_arap, t_arap, kArapUID, Matrix3x3::Identity(),
                      0.5 * Matrix3x3::Identity(), Vector3::Zero()));
        object->geometries().create(
            make_body(kKappa, A_ortho, t_ortho, kOrthoUID, Matrix3x3::Identity(),
                      -0.2 * Matrix3x3::Identity(), Vector3{0.1, -0.15, 0.05}));
        return scene;
    }
};

/// Drives the real `ABDLinearSubsystem` through a *test-local* `GlobalLinearSystem::Impl`
/// (the T6.1b idiom: `GlobalLinearSystem::solve()` is T8.3's, so the assembly is driven
/// from the test) and reports what it produced.
struct AssemblyRun
{
    BlockMap            blocks;
    Eigen::VectorXd     gradient;
    std::size_t         triplet_count       = 0;
    std::size_t         expected_extent     = 0;
    std::size_t         dof_count           = 0;
    std::size_t         block_count         = 0;
};

/// Writes the per-geometry dof offsets (`builtin::dof_offset`) the way the engine does:
/// the ABD subsystem's `receive_init_dof_info` runs from `GlobalLinearSystem::init()`,
/// and nothing calls that until a linear system is initialised -- so a test that wants
/// the dof layout must drive a (throwaway) test-local `Impl` first (§7.22 rule 12).
void prepare_dof_layout(gg::ABDLinearSubsystem& abd_linear)
{
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(abd_linear);
    impl.init();
}

AssemblyRun run_assembly(gg::ABDLinearSubsystem& abd_linear)
{
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(abd_linear);
    impl.init();

    AssemblyRun out;
    out.dof_count = static_cast<std::size_t>(impl.diag_dof_offsets_counts.total_count());
    REQUIRE(impl._update_subsystem_extent());
    out.expected_extent = static_cast<std::size_t>(impl.subsystem_triplet_offsets_counts.counts()[0]);
    out.block_count     = out.expected_extent / 16u;
    impl._assemble_linear_system();
    // the assembly is *recorded* on the frame stream: flush it before reading
    gg::frame_stream().synchronize();

    out.triplet_count = static_cast<std::size_t>(impl.triplet_A.triplet_count());
    out.blocks        = read_blocks(impl.triplet_A);
    out.gradient.resize(static_cast<Eigen::Index>(out.dof_count));
    impl.b.copy_to(span<double>{out.gradient.data(), out.dof_count});
    return out;
}
}  // namespace

TEST_CASE("abd_constitution_assembly", "[luisa][abd_constitution]")
{
    test_runtime();
    CaseScope scope{"abd_constitution_assembly"};

    const TwoBodyScene two;
    Scene              scene = two.build("abd_constitution_assembly");

    Engine engine{"luisa", workspace_path("uipc_luisa_abd_constitution_assembly")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    auto& c = scope.counter;
    REQUIRE(T63AProbe::abd != nullptr);
    REQUIRE(T63AProbe::abd_linear != nullptr);
    REQUIRE(T63AProbe::abd_linear->is_valid());
    // the dof layout (`builtin::dof_offset`) is written by the *linear subsystem's*
    // init, so it has to be driven before any body is addressed by its dof offset
    prepare_dof_layout(*T63AProbe::abd_linear);

    // ---- who owns which uid? (proved, not assumed -- see the file comment) ----
    const bool ortho_owns = T63AProbe::ortho != nullptr && T63AProbe::ortho->is_valid()
                            && T63AProbe::ortho->kappas.size() == 1u;
    const bool arap_owns = T63AProbe::arap != nullptr && T63AProbe::arap->is_valid()
                           && T63AProbe::arap->kappas.size() == 1u;
    std::printf("[t63a] assembly uid ownership: uid1 = %s (kappas %zu), uid2 = %s (kappas %zu)\n",
                ortho_owns ? "uipc::backend::luisa::OrthoPotential" : "ANOTHER SYSTEM",
                T63AProbe::ortho ? T63AProbe::ortho->kappas.size() : 0u,
                arap_owns ? "uipc::backend::luisa::ARAP" : "ANOTHER SYSTEM",
                T63AProbe::arap ? T63AProbe::arap->kappas.size() : 0u);
    if(!(ortho_owns && arap_owns))
        std::printf("[t63a] SKIP: uid group not owned by the real T6.3a constitution; the "
                    "values below would be another system's/n");
    // A uid group is owned by the *last* equal-uid entry of the uid-sorted constitution
    // list, in `SimEngine::build_systems()`'s demangled-name order: for uid 1 and 2 in
    // this binary that is the real constitution (T6.1a's mock moved to uid 8).  If this
    // ever fails, the line above names the case -- no assertion is weakened.
    REQUIRE(ortho_owns);
    REQUIRE(arap_owns);

    const double dt = T63AProbe::self->dt();
    const auto   bodies = scene_bodies(world, *T63AProbe::abd);
    REQUIRE(bodies.size() == 2u);
    std::printf("[t63a] assembly bodies: %zu (dof offsets %zu / %zu, uids %llu / %llu), dt %.6f/n",
                bodies.size(), bodies[0].dof_offset, bodies[1].dof_offset,
                static_cast<unsigned long long>(bodies[0].uid),
                static_cast<unsigned long long>(bodies[1].uid), static_cast<double>(dt));

    const auto run = run_assembly(*T63AProbe::abd_linear);

    // ---- (A) the extent budget ---------------------------------------------
    // hand-derived: one 12x12 block per body and nothing else -- no reporter of this
    // binary claims this scene's geometries (they gate on their own markers) and the
    // *kinetic* term of the body block is zero here (T6.1b's `zt61b::T61bKineticOwner`
    // owns the dynamics' single kinetic slot in this binary and is disabled in a scene
    // without its marker, so `body_id_to_kinetic_*` keeps the subsystem's zero-fill).
    T63A_CHECK(c, run.dof_count == 24u);
    T63A_CHECK(c, run.block_count == 2u);
    T63A_CHECK(c, run.expected_extent == 2u * 16u);
    T63A_CHECK(c, run.triplet_count == 32u);
    T63A_CHECK(c, run.blocks.size() == 32u);

    // ---- (B) the (body, 3x3-block) indices it writes ------------------------
    // the upper-triangular COO convention: every 12x12 body block is written as all 16
    // 3x3 sub-blocks `(body*4 + a, body*4 + b)`, and the strictly-lower ones
    // (`a > b`) are *zero* (`zero_out_lower`) -- the symmetric reconstruction
    // regenerates them.
    {
        int lower_nonzero = 0;
        for(const auto& [key, blk] : run.blocks)
        {
            const int body_i = key.first / 4;
            const int a      = key.first % 4;
            const int body_j = key.second / 4;
            const int b      = key.second % 4;
            T63A_CHECK(c, body_i == body_j);
            T63A_CHECK(c, (body_i >= 0 && body_i < 2));
            if(a > b)
            {
                if(max_abs(blk) > 0.0)
                    ++lower_nonzero;
            }
        }
        T63A_CHECK(c, lower_nonzero == 0);
        // the translation 3x3 block of each body's block is zero (the local laws do not
        // see `t`), and the non-translation part is not
        double off_block_sum = 0.0;
        for(const auto& [key, blk] : run.blocks)
        {
            const int a = key.first % 4;
            const int b = key.second % 4;
            if(a == 0 || b == 0)
                T63A_CHECK(c, max_abs(blk) == 0.0);
            else
                off_block_sum += max_abs(blk);
        }
        T63A_CHECK(c, off_block_sum > 0.0);
    }

    // ---- (C) the assembled content against the hand-derived dense block ------
    {
        const auto expected = expected_blocks(bodies, dt);
        REQUIRE(expected.size() == run.blocks.size());
        double worst      = 0.0;
        double worst_blk  = 0.0;
        int    compared   = 0;
        int    missing    = 0;
        for(const auto& [key, blk] : expected)
        {
            auto it = run.blocks.find(key);
            if(it == run.blocks.end())
            {
                std::printf("[t63a] assembly COO: missing block key (%d, %d)/n", key.first, key.second);
                ++missing;
                continue;
            }
            const double d = (it->second - blk).norm();
            worst_blk      = std::max(worst_blk, d);
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                    worst = std::max(worst, std::abs(it->second(rr, cc) - blk(rr, cc)));
            ++compared;
        }
        (void)compared;
        std::printf("[t63a] assembly COO: %zu triplets, %zu keys, %d keys compared, "
                    "worst |entry| %.3e, worst |block| %.3e, |H| %.3e/n",
                    run.triplet_count, run.blocks.size(), static_cast<int>(expected.size()),
                    worst, worst_blk, max_abs(expected.begin()->second));
        T63A_CHECK(c, missing == 0);
        T63A_CHECK(c, worst <= 1e-10);
        T63A_CHECK(c, worst_blk > 0.0);
    }

    // ---- (D) the assembled gradient at each body's dof offset ---------------
    {
        const Eigen::VectorXd expected = expected_gradient(bodies, dt);
        REQUIRE(expected.size() == static_cast<Eigen::Index>(run.dof_count));
        double worst = 0.0;
        for(Eigen::Index i = 0; i < expected.size(); ++i)
            worst = std::max(worst, rel_err(run.gradient(i), expected(i)));
        std::printf("[t63a] assembly gradient: worst rel %.3e, |b| %.6e/n",
                    worst, expected.norm());
        T63A_CHECK(c, expected.norm() > 1.0);  // not a vacuous all-zero comparison
        T63A_CHECK(c, worst <= 1e-12);
    }

    std::printf("[t63a] %s: %d checks/n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  5) a real scene: one uid-2 (ARAP) and one uid-1 (ortho) body, one `advance()`,
//     everything read back through the *frontend* state accessor
// ===========================================================================
TEST_CASE("abd_constitution_scene", "[luisa][abd_constitution]")
{
    test_runtime();
    CaseScope scope{"abd_constitution_scene"};

    TwoBodyScene two;
    Scene        scene = two.build("abd_constitution_scene");
    Engine engine{"luisa", workspace_path("uipc_luisa_abd_constitution_scene")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    auto& c = scope.counter;
    REQUIRE(T63AProbe::abd != nullptr);
    REQUIRE(T63AProbe::abd_linear != nullptr);
    REQUIRE(T63AProbe::abd_linear->is_valid());
    // the dof layout (`builtin::dof_offset`) is written by the *linear subsystem's*
    // init, so it has to be driven before any body is addressed by its dof offset
    prepare_dof_layout(*T63AProbe::abd_linear);

    const Float dt = T63AProbe::self->dt();
    T63A_CHECK(c, dt > 0.0f);

    const bool ortho_owns = T63AProbe::ortho != nullptr && T63AProbe::ortho->is_valid()
                            && T63AProbe::ortho->kappas.size() == 1u;
    const bool arap_owns = T63AProbe::arap != nullptr && T63AProbe::arap->is_valid()
                           && T63AProbe::arap->kappas.size() == 1u;
    std::printf("[t63a] scene uid ownership: uid1 = %s, uid2 = %s; dt = %.8f\n",
                ortho_owns ? "uipc::backend::luisa::OrthoPotential" : "ANOTHER SYSTEM",
                arap_owns ? "uipc::backend::luisa::ARAP" : "ANOTHER SYSTEM",
                static_cast<double>(dt));
    if(!(ortho_owns && arap_owns))
        std::printf("[t63a] SKIP: uid group not owned by the real T6.3a constitution\n");
    REQUIRE(ortho_owns);
    REQUIRE(arap_owns);

    // ---- (A) the frontend state accessor --------------------------------
    auto feature = world.features().find<AffineBodyStateAccessorFeature>();
    REQUIRE(feature != nullptr);
    T63A_CHECK(c, feature->body_count() == 2);
    auto state_geo = feature->create_geometry();
    T63A_CHECK(c, state_geo.instances().size() == 2);
    auto trans_attr = state_geo.instances().find<Matrix4x4>(builtin::transform);
    if(!trans_attr)
        trans_attr = state_geo.instances().create<Matrix4x4>(builtin::transform,
                                                             Matrix4x4::Identity());

    auto check_accessor = [&](const char* when)
    {
        feature->copy_to(state_geo);
        auto              trans = view(*trans_attr);
        const auto        bodies = scene_bodies(world, *T63AProbe::abd);
        double            worst  = 0.0;
        for(const auto& b : bodies)
        {
            REQUIRE(b.body < 2u);
            const Matrix4x4 expected = gg::q_to_transform(b.q);
            for(int r = 0; r < 4; ++r)
                for(int cc = 0; cc < 4; ++cc)
                    worst = std::max(worst,
                                     std::abs(trans[b.body](r, cc) - expected(r, cc)));
        }
        std::printf("[t63a] scene %s: accessor vs device q, worst |dT| = %.3e\n", when, worst);
        T63A_CHECK(c, worst <= 1e-12);
    };
    check_accessor("before advance");

    // ---- (B) one advance() ----------------------------------------------
    world.advance();
    REQUIRE(world.is_valid());
    world.sync();
    REQUIRE(world.is_valid());
    check_accessor("after advance");

    // the BDF1 *prediction*'s own law on the ARAP body: with an isotropic velocity
    // `V = 0.5 I` (and the zero gravity of this config) `q_tilde = q + q_v dt` keeps it
    // an isotropic stretch, `A -> A + 0.5 I dt`.  The *state* `q` is deliberately not
    // advanced by one `advance()` in this batch: the update path is T8.3's `solve()` +
    // T6.4's line search, so the integrator's output lands in `q_tilde` (this is pinned
    // below, not assumed).
    const auto bodies = scene_bodies(world, *T63AProbe::abd);
    REQUIRE(bodies.size() == 2u);
    {
        const auto q_tilde = read_vec12(T63AProbe::abd->q_tildes());
        REQUIRE(q_tilde.size() == 2u);
        const auto& b_arap = (bodies[0].uid == kArapUID) ? bodies[0] : bodies[1];
        const auto& b_state = b_arap.q;
        const auto  A_t     = A_of(q_tilde[b_arap.body]);
        const double s_t    = (A_t(0, 0) + A_t(1, 1) + A_t(2, 2)) / 3.0;
        const double s_pred = two.s_arap + 0.5 * static_cast<double>(dt);
        T63A_CHECK(c, dev(A_t, s_t * Matrix3x3::Identity()) <= 1e-12);
        T63A_CHECK(c, rel_err(s_t, s_pred) <= 1e-12);
        // the state is the scene's initial state (the update path is not in this batch)
        const auto A_s = A_of(b_state);
        T63A_CHECK(c, dev(A_s, two.A_arap) <= 1e-14);
        std::printf("[t63a] scene BDF1 prediction: ARAP stretch %.12f (predicted %.12f) "
                    "isotropic to %.1e; state stretch %.12f\n",
                    s_t, s_pred, dev(A_t, s_t * Matrix3x3::Identity()),
                    (A_s(0, 0) + A_s(1, 1) + A_s(2, 2)) / 3.0);
    }

    // ---- (C) the assembly at the read-back q -----------------------------
    const auto run = run_assembly(*T63AProbe::abd_linear);
    T63A_CHECK(c, run.dof_count == 24u);
    T63A_CHECK(c, run.block_count == 2u);
    T63A_CHECK(c, run.triplet_count == 32u);
    {
        const auto expected = expected_blocks(bodies, dt);
        double     worst    = 0.0;
        for(const auto& [key, blk] : expected)
        {
            auto it = run.blocks.find(key);
            REQUIRE(it != run.blocks.end());
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                    worst = std::max(worst, std::abs(it->second(rr, cc) - blk(rr, cc)));
        }
        std::printf("[t63a] scene assembled block: worst |dev-hand-derived| %.3e\n", worst);
        T63A_CHECK(c, worst <= 1e-10);
    }
    {
        const Eigen::VectorXd expected = expected_gradient(bodies, dt);
        double                worst    = 0.0;
        for(Eigen::Index i = 0; i < expected.size(); ++i)
            worst = std::max(worst, rel_err(run.gradient(i), expected(i)));
        std::printf("[t63a] scene assembled gradient: worst rel %.3e, |b| %.6e\n",
                    worst, expected.norm());
        T63A_CHECK(c, expected.norm() > 1.0);
        T63A_CHECK(c, worst <= 1e-12);
    }

    // ---- (D) the energy law at the read-back q ---------------------------
    // `ARAP::do_compute_energy` is *unreachable* from a test in T6.3a: its
    // `ComputeEnergyInfo` wraps `ABDLineSearchReporter::ComputeEnergyInfo`, whose
    // `energies` view is writable only by `ABDLineSearchReporter` (a `friend`), and
    // that system is T6.4's (it has no creator here).  The energy is therefore checked
    // through the *same* generic routine the kernel body calls, against this file's
    // closed forms (case 1 proves the routine equals them).
    {
        double worst_arap = 0.0, worst_ortho = 0.0;
        for(const auto& b : bodies)
        {
            const auto A    = A_of(b.q);
            const auto Vdt2 = b.volume * static_cast<double>(dt) * static_cast<double>(dt);
            if(b.uid == kOrthoUID)
            {
                const auto loc = ortho_local(b.kappa, A);
                worst_ortho    = std::max(worst_ortho,
                                          rel_err(ortho_E_host(b.kappa, dt, b.volume, b.q),
                                                  loc.E * Vdt2));
            }
            else
            {
                const double s   = (A(0, 0) + A(1, 1) + A(2, 2)) / 3.0;
                const double E   = 3.0 * b.kappa * (s - 1.0) * (s - 1.0) * Vdt2;
                worst_arap       = std::max(worst_arap,
                                            rel_err(arap_E_host(b.kappa, dt, b.volume, b.q), E));
            }
        }
        std::printf("[t63a] scene energy law: ARAP worst rel %.3e, ortho worst rel %.3e\n",
                    worst_arap, worst_ortho);
        T63A_CHECK(c, worst_arap <= 1e-13);
        T63A_CHECK(c, worst_ortho <= 1e-13);
    }

    std::printf("[t63a] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  6) device == host for every ported generic routine
//
//     The two constitutions' kernels are file-local (their cached shaders live in an
//     anonymous namespace in the .cpp, exactly like every other ported system), so the
//     device round dispatches a *test* kernel that calls the *same* generic routines
//     with `S = lcrt::Var<double>` and compares against the host's `S = double`: one
//     arithmetic body, two instantiations -- bitwise where the arithmetic is identical
//     (the T6.1a/T6.2 convention).
// ===========================================================================
namespace
{
auto& t63a_round_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<gg::LVector12>,      // qs
                                   lcrt::Buffer<double>,             // kappas
                                   lcrt::Buffer<double>,             // volumes
                                   double,                           // dt
                                   lcrt::Buffer<double>,             // arap E
                                   lcrt::Buffer<gg::LVector12>,      // arap G
                                   lcrt::Buffer<gg::LMatrix12x12>,   // arap H
                                   lcrt::Buffer<double>,             // ortho E
                                   lcrt::Buffer<gg::LVector12>,      // ortho G
                                   lcrt::Buffer<gg::LMatrix12x12>,   // ortho H
                                   uint32_t>(                        // n
            [](lcrt::BufferVar<gg::LVector12>    qs,
               lcrt::BufferVar<double>           kappas,
               lcrt::BufferVar<double>           volumes,
               lcrt::Var<double>                 dt,
               lcrt::BufferVar<double>           arap_E,
               lcrt::BufferVar<gg::LVector12>    arap_G,
               lcrt::BufferVar<gg::LMatrix12x12> arap_H,
               lcrt::BufferVar<double>           ortho_E,
               lcrt::BufferVar<gg::LVector12>    ortho_G,
               lcrt::BufferVar<gg::LMatrix12x12> ortho_H,
               lcrt::Var<uint32_t>               n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto kappa = kappas->read(i);
                    auto vol   = volumes->read(i);
                    auto q     = gg::load_vec12(qs->read(i));

                    auto E1 = gg::abd_arap_detail::body_energy<lcrt::Var<double>>(kappa, dt, vol, q);
                    arap_E->write(i, E1);
                    auto G1 = gg::abd_arap_detail::body_gradient<lcrt::Var<double>>(kappa, dt, vol, q);
                    lcrt::Var<gg::LVector12> g1;
                    gg::store_vec12(g1, G1);
                    arap_G->write(i, g1);
                    auto H1 = gg::abd_arap_detail::body_hessian<lcrt::Var<double>>(kappa, dt, vol, q);
                    lcrt::Var<gg::LMatrix12x12> h1;
                    gg::store_mat12x12(h1, H1);
                    arap_H->write(i, h1);

                    auto E2 = gg::abd_ortho_potential_detail::body_energy<lcrt::Var<double>>(
                        kappa, dt, vol, q);
                    ortho_E->write(i, E2);
                    auto G2 = gg::abd_ortho_potential_detail::body_gradient<lcrt::Var<double>>(
                        kappa, dt, vol, q);
                    lcrt::Var<gg::LVector12> g2;
                    gg::store_vec12(g2, G2);
                    ortho_G->write(i, g2);
                    auto H2 = gg::abd_ortho_potential_detail::body_hessian<lcrt::Var<double>>(
                        kappa, dt, vol, q);
                    lcrt::Var<gg::LMatrix12x12> h2;
                    gg::store_mat12x12(h2, H2);
                    ortho_H->write(i, h2);
                };
            });
    return k;
}
}  // namespace

TEST_CASE("abd_constitution_device_vs_host", "[luisa][abd_constitution]")
{
    test_runtime();
    CaseScope scope{"abd_constitution_device_vs_host"};
    auto&     c = scope.counter;

    constexpr int N     = 16;
    const double  dt    = 0.013;

    std::vector<Vector12> qs;
    std::vector<double>   kappas, volumes;
    for(int i = 0; i < N; ++i)
    {
        qs.push_back(q_of(random_admissible_A(),
                          Vector3{rand_uniform(-2.0, 2.0),
                                  rand_uniform(-2.0, 2.0),
                                  rand_uniform(-2.0, 2.0)}));
        kappas.push_back(rand_uniform(0.5, 2.0));
        volumes.push_back(rand_uniform(0.2, 1.5));
    }

    auto q_dev = upload_as<gg::LVector12>(qs, [](const Vector12& q) { return gg::to_luisa(q); });
    auto k_dev = upload(kappas);
    auto v_dev = upload(volumes);

    lud::LDeviceBuffer<double>              aE(N), oE(N);
    lud::LDeviceBuffer<gg::LVector12>       aG(N), oG(N);
    lud::LDeviceBuffer<gg::LMatrix12x12>    aH(N), oH(N);

    auto& k = t63a_round_kernel();
    gg::frame_stream().record(k(q_dev.view(),
                                k_dev.view(),
                                v_dev.view(),
                                dt,
                                aE.view(),
                                aG.view(),
                                aH.view(),
                                oE.view(),
                                oG.view(),
                                oH.view(),
                                static_cast<uint32_t>(N))
                                .dispatch(N));
    gg::frame_stream().synchronize();

    // the device element types are the registered ABD structs, so each one is
    // converted through the ported `to_eigen` (never a `reinterpret_cast`) and then
    // flattened exactly like the host side below
    auto flatten_G = [](const std::vector<gg::LVector12>& raw)
    {
        std::vector<double> out;
        out.reserve(raw.size() * 12u);
        for(const auto& v : raw)
        {
            const Vector12 e = gg::to_eigen(v);
            for(int k = 0; k < 12; ++k)
                out.push_back(e(k));
        }
        return out;
    };
    auto flatten_H = [](const std::vector<gg::LMatrix12x12>& raw)
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

    const auto dE_arap  = read_device(aE.cview());
    const auto dG_arap  = flatten_G(read_device(aG.cview()));
    const auto dH_arap  = flatten_H(read_device(aH.cview()));
    const auto dE_ortho = read_device(oE.cview());
    const auto dG_ortho = flatten_G(read_device(oG.cview()));
    const auto dH_ortho = flatten_H(read_device(oH.cview()));

    std::vector<double> hE_arap(N), hE_ortho(N);
    std::vector<double> hG_arap(static_cast<std::size_t>(N) * 12u), hG_ortho(static_cast<std::size_t>(N) * 12u);
    std::vector<double> hH_arap(static_cast<std::size_t>(N) * 144u), hH_ortho(static_cast<std::size_t>(N) * 144u);
    for(int i = 0; i < N; ++i)
    {
        hE_arap[i]  = arap_E_host(kappas[i], dt, volumes[i], qs[i]);
        hE_ortho[i] = ortho_E_host(kappas[i], dt, volumes[i], qs[i]);
        const Vector12 Ga = arap_G_host(kappas[i], dt, volumes[i], qs[i]);
        const Vector12 Go = ortho_G_host(kappas[i], dt, volumes[i], qs[i]);
        const Matrix12x12 Ha = arap_H_host(kappas[i], dt, volumes[i], qs[i]);
        const Matrix12x12 Ho = ortho_H_host(kappas[i], dt, volumes[i], qs[i]);
        for(int kk = 0; kk < 12; ++kk)
        {
            hG_arap[static_cast<std::size_t>(i) * 12u + kk] = Ga(kk);
            hG_ortho[static_cast<std::size_t>(i) * 12u + kk] = Go(kk);
        }
        for(int rr = 0; rr < 12; ++rr)
        {
            for(int cc = 0; cc < 12; ++cc)
            {
                hH_arap[(static_cast<std::size_t>(i) * 12u + rr) * 12u + cc] = Ha(rr, cc);
                hH_ortho[(static_cast<std::size_t>(i) * 12u + rr) * 12u + cc] = Ho(rr, cc);
            }
        }
    }

    auto worst = [](const std::vector<double>& a, const std::vector<double>& b)
    {
        double w = 0.0;
        REQUIRE(a.size() == b.size());
        for(std::size_t i = 0; i < a.size(); ++i)
            w = std::max(w, std::abs(a[i] - b[i]));
        return w;
    };
    const double d_ea = worst(dE_arap, hE_arap);
    const double d_ga = worst(dG_arap, hG_arap);
    const double d_ha = worst(dH_arap, hH_arap);
    const double d_eo = worst(dE_ortho, hE_ortho);
    const double d_go = worst(dG_ortho, hG_ortho);
    const double d_ho = worst(dH_ortho, hH_ortho);

    // the deviations are measured *relative to each quantity's own scale* (the values
    // here are O(1e-4)): the generic routines are the same statements, but the DSL
    // lowers the `sqrt` inside `lps::svd` and the `pow` inside the generated ortho
    // expression to `lcrt::*`, so the last bits differ.  §7.17 rule 34: assert a bound
    // per routine and *record* the measurement.
    auto scale_of = [](const std::vector<double>& v)
    {
        double s = 0.0;
        for(double x : v)
            s = std::max(s, std::abs(x));
        return s;
    };
    const double r_ea = d_ea / scale_of(hE_arap);
    const double r_ga = d_ga / scale_of(hG_arap);
    const double r_ha = d_ha / scale_of(hH_arap);
    const double r_eo = d_eo / scale_of(hE_ortho);
    const double r_go = d_go / scale_of(hG_ortho);
    const double r_ho = d_ho / scale_of(hH_ortho);

    std::printf("[t63a] device vs host (16 states): ARAP E %.3e G %.3e H %.3e | "
                "ortho E %.3e G %.3e H %.3e  (relative: %.2e/%.2e/%.2e, %.2e/%.2e/%.2e)\n",
                d_ea, d_ga, d_ha, d_eo, d_go, d_ho, r_ea, r_ga, r_ha, r_eo, r_go, r_ho);
    T63A_CHECK(c, (r_ea <= 1e-12 && r_ga <= 1e-12 && r_ha <= 1e-12));
    T63A_CHECK(c, (r_eo <= 1e-12 && r_go <= 1e-12 && r_ho <= 1e-12));

    std::printf("[t63a] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  7) per-case assertion counts
// ===========================================================================
TEST_CASE("abd_constitution_summary", "[luisa][abd_constitution]")
{
    int total = 0;
    for(const auto& [name, checks] : g_case_counts)
    {
        std::printf("[t63a] case %-42s %5d checks\n", name.c_str(), checks);
        total += checks;
        CHECK(checks > 0);
    }
    std::printf("[t63a] summary: %d cases, %d checks\n",
                static_cast<int>(g_case_counts.size()), total);
}
