/********************************************************************
 * @file luisa_fem_constitution_c_test.cpp
 * @brief T5.2 (batch c) — the rod / spring / particle FEM constitutions of the
 *        LuisaCompute backend: `kirchhoff_rod_bending` (uid 15, an extra
 *        constitution), `hookean_spring_1d` (uid 12), `particle_0d` (uid 13) and the
 *        four uid-0 placeholders `empty_0d/1d/2d/3d`.
 *
 * Every oracle in this file is derived independently *here*: the reference backend is
 * CUDA-only and cannot be host-included, so nothing is copied mechanically.  The
 * oracles are
 *   (a) hand-derived closed forms.  The generated `sym/kirchhoff_rod_bending.inl`
 *       contains `x7 = (e0.e1 + |e0||e1|)^-2` and a bracket that is
 *       `4 |e0 x e1|^2`, so the whole energy collapses to
 *           E = k pi r^4 (1 - cos theta) / ((1 + cos theta) L0) = k pi r^4 tan^2(theta/2) / L0
 *       with `e0 = x1 - x0`, `e1 = x2 - x1` and `theta` the turn angle; its gradient
 *       is derived analytically (below) from `c = cos theta = e0.e1 / (|e0||e1|)`.
 *       The generated `sym/hookean_spring_1d.inl` is the Hookean law
 *           E = k (L - L0)^2 / (2 L0^2),   L = |x1 - x0|
 *       scaled by the reference kernel's `Vdt2 = L0 r^2 pi dt^2`
 *       (`r = edge_thickness(t0, t1)`, the 1d "material volume" x dt^2).
 *   (b) central finite differences (h = 1e-6) *of those independently derived closed
 *       forms* with respect to the element dofs (9 for a rod hinge, 6 for a spring) —
 *       compared against the *ported* analytic gradient and the *ported unclamped*
 *       analytic Hessian (so the comparison holds in every state, PSD or not).
 *   (c) the `make_spd` eigenvalue-clamp identity: the shipped, projected Hessian is
 *       compared against an exact Eigen eigen-clamp of the unclamped analytic Hessian
 *       (the reference's `make_spd` is exactly that clamp; the port's fixed 8-sweep
 *       Jacobi is measured here on the *actual* element Hessians).
 *   (d) structural invariants: Hessian symmetry, positive semidefiniteness at the rest
 *       state, and the exact zero energy/gradient of a straight rod / relaxed spring.
 *   (e) the degenerate states the reference has *no* guard for (zero-length edge, zero
 *       rest length, collapsed rod) — pinned against the closed form, which predicts the
 *       exact classification (finite / +/-inf / NaN) that the test then checks.
 *   (f) device-vs-host equivalence of the ported per-element routines
 *       (S = double vs S = lcrt::Var<double> inside a Kernel1D).
 *   (g) a real scene through the public Engine/World API: the *real*
 *       `do_build`/`do_init`/`do_report_extent`/`do_compute_energy`/
 *       `do_compute_gradient_hessian` run on the real FEM device buffers and the
 *       assembled doublets/triplets are compared against the same ported per-element
 *       routines; the four uid-0 placeholders are exercised by proving that a geometry
 *       with **no** constitution is still picked up by the FEM method (which is only
 *       possible because `Empty0D/1D/2D/3D` register uid 0).
 *
 * There is no plastic history in this batch (the brief's validation item 5 does not
 * apply): the batch is rods/springs/particles, all of which are history-free in the
 * reference.
 *
 * SHARED-BINARY NOTE (cookbook §7.14 rule 6 + the brief's §4bis):
 * `luisa_fem_base_test.cpp` registers *mock* constitutions that claim the frontend
 * uids — including HookeanSpring's 12 (dim 1) and Particle's 13 (dim 0).  The method
 * hands a (dim, uid) geometry group to the *first* system registered with that uid, so
 * in a full-suite run those two groups can belong to the mocks: the
 * `fem_constitution_c_scene` case *detects* that (the constitution's own `do_init`
 * output is compared against the scene's element count), says so loudly with a `WARN`
 * and skips only the affected block.  uid 15 (KirchhoffRodBending) and uid 0 (the four
 * placeholders) are claimed by no mock, so those always run — run
 * `scripts/luisa_build.py run "[constitution_c]"` to exercise all of them.
 *********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/kirchhoff_rod_bending.h>
#include <uipc/constitution/hookean_spring.h>
#include <uipc/constitution/particle.h>
#include <uipc/constitution/empty.h>

#include <type_define.h>
#include <frame_stream.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_elastics.h>
#include <finite_element/codim_1d_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/constitutions/kirchhoff_rod_bending.h>
#include <finite_element/constitutions/kirchhoff_rod_bending_function.h>
#include <finite_element/constitutions/hookean_spring_1d.h>
#include <finite_element/constitutions/hookean_spring_1d_function.h>
#include <utils/matrix_assembler.h>
#include <utils/codim_thickness.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the engine header must
// be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;

// The ported symbols under test.
namespace sym_krb  = uipc::backend::luisa::sym::kirchhoff_rod_bending;
namespace sym_hs1d = uipc::backend::luisa::sym::hookean_spring_1d;
namespace krb      = uipc::backend::luisa::krb_detail;
namespace hs1d     = uipc::backend::luisa::hs1d_detail;

namespace
{
constexpr double kPi  = 3.14159265358979323846264338327950288;
constexpr double kFdH = 1e-6;

double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

/// `max |entry|` of any Eigen matrix/vector *expression*.
template <typename Derived>
double max_abs(const Eigen::MatrixBase<Derived>& m)
{
    return m.cwiseAbs().maxCoeff();
}

/// Smallest / largest eigenvalue of a symmetric matrix.
template <int N>
double min_eigenvalue(const Matrix<Float, N, N>& m)
{
    Eigen::SelfAdjointEigenSolver<Matrix<Float, N, N>> es(m);
    return es.eigenvalues()(0);
}

template <int N>
double max_eigenvalue(const Matrix<Float, N, N>& m)
{
    Eigen::SelfAdjointEigenSolver<Matrix<Float, N, N>> es(m);
    return es.eigenvalues()(N - 1);
}

/// The reference's `make_spd`: eigen-decompose, clamp the negative eigenvalues to zero,
/// reconstruct.  This is the *definition* the port's fixed-sweep Jacobi approximates.
template <int N>
Matrix<Float, N, N> eigen_clamp(const Matrix<Float, N, N>& m)
{
    Eigen::SelfAdjointEigenSolver<Matrix<Float, N, N>> es(m);
    Vector<Float, N> w = es.eigenvalues();
    for(int i = 0; i < N; ++i)
        w(i) = w(i) < 0.0 ? 0.0 : w(i);
    return es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
}

/// A file-local, deterministically seeded RNG (cookbook §7.17 rule 29: the stream is
/// process-wide, so a case's random states depend on which cases ran before it).
std::mt19937& rng()
{
    static std::mt19937 r(20240918u);
    return r;
}

double rand_uniform(double a, double b)
{
    return std::uniform_real_distribution<double>(a, b)(rng());
}

Vector3 rand_vec3(double lo, double hi)
{
    return Vector3{rand_uniform(lo, hi), rand_uniform(lo, hi), rand_uniform(lo, hi)};
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

template <typename T, typename U, typename Conv>
lud::LDeviceBuffer<T> upload(const std::vector<U>& src, Conv&& conv)
{
    std::vector<T> host;
    host.reserve(src.size());
    for(auto& v : src)
        host.push_back(conv(v));
    lud::LDeviceBuffer<T> buffer;
    buffer.copy_from(span<const T>{host});
    return buffer;
}

template <typename T>
lud::LDeviceBuffer<T> upload(const std::vector<T>& src)
{
    lud::LDeviceBuffer<T> buffer;
    buffer.copy_from(span<const T>{src});
    return buffer;
}

// ---------------------------------------------------------------------------
//  Eigen <-> numerics-layer shims (the T5.1 convention)
// ---------------------------------------------------------------------------
template <int N>
[[nodiscard]] inline lps::LVec<double, N> lv(const Vector<Float, N>& v) noexcept
{
    lps::LVec<double, N> out;
    for(int i = 0; i < N; ++i)
        out(i) = v(i);
    return out;
}

template <int N>
[[nodiscard]] inline Vector<Float, N> to_eig(const lps::LVec<double, N>& v) noexcept
{
    Vector<Float, N> out;
    for(int i = 0; i < N; ++i)
        out(i) = v(i);
    return out;
}

template <int R, int C>
[[nodiscard]] inline Matrix<Float, R, C> to_eig(const lps::LMat<double, R, C>& m) noexcept
{
    Matrix<Float, R, C> out;
    for(int c = 0; c < C; ++c)
        for(int r = 0; r < R; ++r)
            out(r, c) = m(r, c);
    return out;
}

/// Pack the three hinge positions in the *device* layout
/// (`X.segment<3>(0) = x0; X.segment<3>(3) = x1; X.segment<3>(6) = x2;`).
Vector9 pack9(const Vector3& a, const Vector3& b, const Vector3& c)
{
    Vector9 v;
    v.segment<3>(0) = a;
    v.segment<3>(3) = b;
    v.segment<3>(6) = c;
    return v;
}

Vector6 pack6(const Vector3& a, const Vector3& b)
{
    Vector6 v;
    v.segment<3>(0) = a;
    v.segment<3>(3) = b;
    return v;
}

void unpack9(const Vector9& v, Vector3& a, Vector3& b, Vector3& c)
{
    a = v.segment<3>(0);
    b = v.segment<3>(3);
    c = v.segment<3>(6);
}

void unpack6(const Vector6& v, Vector3& a, Vector3& b)
{
    a = v.segment<3>(0);
    b = v.segment<3>(3);
}

// ---------------------------------------------------------------------------
//  Independently derived oracles (written from the mathematics, *not* from the
//  ported code)
// ---------------------------------------------------------------------------

/// `theta = angle(e0, e1)`, `e0 = x1 - x0`, `e1 = x2 - x1`; returns `cos theta` clamped
/// to [-1, 1] (a clamp is what `acos` needs, not a repair of the energy).
double rod_cos_theta(const Vector3& x0, const Vector3& x1, const Vector3& x2)
{
    const Vector3 u = x1 - x0;
    const Vector3 v = x2 - x1;
    const double  a = u.norm();
    const double  b = v.norm();
    if(a <= 0.0 || b <= 0.0)
        return 0.0;  // "undefined"; only the degenerate states reach this
    return std::clamp(u.dot(v) / (a * b), -1.0, 1.0);
}

/**
 * @brief `E = k pi r^4 (1 - cos theta) / ((1 + cos theta) L0) = K tan^2(theta/2)`.
 *
 * The derived content of `sym/kirchhoff_rod_bending.inl`'s `E`.  The generated file
 * computes `x7 = (e0.e1 + |e0||e1|)^-2` and a bracket that equals `4 |e0 x e1|^2`, so
 *
 *   E = (1/4) k pi r^4 x7 bracket / L0
 *     = k pi r^4 |e0|^2|e1|^2 sin^2(theta) / (L0 |e0|^2|e1|^2 (1 + cos theta)^2)
 *     = (k pi r^4 / L0) (1 - cos^2 theta) / (1 + cos theta)^2
 *     = (k pi r^4 / L0) (1 - cos theta) / (1 + cos theta) = K tan^2(theta/2).
 *
 * MEASURED TRAP (recorded in cookbook §7.17 rule 30): the first version of this oracle
 * cancelled `(1 - cos^2)/(1 + cos)^2` to `(1 - cos)` — an algebra slip, not a port bug —
 * and a *right-angle* hand-computed configuration cannot see it, because `cos = 0`
 * makes `(1-c)` and `(1-c)/(1+c)` equal.  The FD comparison caught it (rel 0.5, i.e. the
 * model ratio `1/(1 + cos)`) and the hand-computed configuration below is now a 60
 * degree hinge, where the two differ by a factor 3/2.
 *
 * Note: *no* dt^2 here; the reference's energy kernel passes `k * dt * dt`, and the
 * gradient kernel scales its result by `dt^2`.
 */
double rod_energy_closed(double k, double r, double L0, const Vector3& x0, const Vector3& x1, const Vector3& x2)
{
    const double c = rod_cos_theta(x0, x1, x2);
    return k * kPi * std::pow(r, 4.0) * (1.0 - c) / ((1.0 + c) * L0);
}

/**
 * @brief The analytic gradient of `rod_energy_closed` w.r.t. `(x0, x1, x2)`.
 *
 * With `u = e0`, `v = e1`, `a = |u|`, `b = |v|`, `c = u.v / (a b)` and
 * `K = k pi r^4 / L0`:
 *      dE/dc  = -2 K / (1 + c)^2
 *      dc/du  = v/(ab) - c u/a^2 ,  dc/dv = u/(ab) - c v/b^2
 *      du/dx0 = -I, du/dx1 = +I, dv/dx1 = -I, dv/dx2 = +I
 *      dc/dx0 = -dc/du ,  dc/dx1 = dc/du - dc/dv ,  dc/dx2 = +dc/dv
 *      dE/dx0 = -dE/dc du ,  dE/dx1 = +dE/dc (du - dv) ,  dE/dx2 = +dE/dc dv
 * (measured trap, cookbook §7.17 rule 30: the first version of this function had the
 *  *signs* of the `x1`/`x2` blocks flipped -- the FD of the generated energy caught it
 *  immediately, `scratch/t52c_eval.py` localised it.)
 */
Vector9 rod_gradient_closed(double k, double r, double L0, const Vector3& x0, const Vector3& x1, const Vector3& x2)
{
    const Vector3 u  = x1 - x0;
    const Vector3 v  = x2 - x1;
    const double  a  = u.norm();
    const double  b  = v.norm();
    const double  ab = a * b;
    const double  c  = rod_cos_theta(x0, x1, x2);

    const double K    = k * kPi * std::pow(r, 4.0) / L0;
    const double dEdc = -2.0 * K / ((1.0 + c) * (1.0 + c));

    const Vector3 du = (v / ab) - (c * u) / (a * a);
    const Vector3 dv = (u / ab) - (c * v) / (b * b);

    Vector9 g;
    g.segment<3>(0) = -dEdc * du;
    g.segment<3>(3) = dEdc * (du - dv);
    g.segment<3>(6) = dEdc * dv;
    return g;
}

/// `E = k (L - L0)^2 / (2 L0^2)` — the derived content of
/// `sym/hookean_spring_1d.inl`'s `E`.
double spring_energy_closed(double k, double L0, const Vector3& x0, const Vector3& x1)
{
    const double L = (x1 - x0).norm();
    return 0.5 * k * (L - L0) * (L - L0) / (L0 * L0);
}

/// The analytic gradient of `spring_energy_closed`: `dE/dX = k (L - L0)/(L0^2 L) e`
/// for `x0` and its negative for `x1`, with `e = x0 - x1` (exactly the generated
/// `dEdX`'s `R(0) = x0 * x7`).
Vector6 spring_gradient_closed(double k, double L0, const Vector3& x0, const Vector3& x1)
{
    const Vector3 e = x0 - x1;
    const double  L = e.norm();
    const Vector3 f = e * (k * (L - L0) / (L0 * L0 * L));
    Vector6       g;
    g.segment<3>(0) = f;
    g.segment<3>(3) = -f;
    return g;
}

/// The reference kernel's `Vdt2 = L0 * r * r * Pi * dt * dt` (association order kept).
double spring_vdt2(double L0, double r, double pi, double dt)
{
    return ((((L0 * r) * r) * pi) * dt) * dt;
}

// ---------------------------------------------------------------------------
//  Central finite differences of an independently derived energy
// ---------------------------------------------------------------------------
template <int N, typename Fn>
Vector<Float, N> fd_gradient(Fn&& f, const Vector<Float, N>& X, double h)
{
    Vector<Float, N> g;
    for(int i = 0; i < N; ++i)
    {
        auto Xp = X;
        auto Xm = X;
        Xp(i) += h;
        Xm(i) -= h;
        g(i) = (f(Xp) - f(Xm)) / (2.0 * h);
    }
    return g;
}

template <int N, typename Fn>
Matrix<Float, N, N> fd_hessian(Fn&& f, const Vector<Float, N>& X, double h)
{
    Matrix<Float, N, N> H;
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
        {
            auto Xpp = X;
            auto Xpm = X;
            auto Xmp = X;
            auto Xmm = X;
            Xpp(i) += h;
            Xpp(j) += h;
            Xpm(i) += h;
            Xpm(j) -= h;
            Xmp(i) -= h;
            Xmp(j) += h;
            Xmm(i) -= h;
            Xmm(j) -= h;
            H(i, j) = (f(Xpp) - f(Xpm) - f(Xmp) + f(Xmm)) / (4.0 * h * h);
        }
    return H;
}

/// A rod hinge state: the three *current* positions (the rest positions only enter
/// through `L0`).
struct RodState
{
    Vector3 x0, x1, x2;
};

/// A *non-degenerate* random rod hinge (the two edges are neither degenerate nor
/// collinear to within 1e-3 rad).
RodState random_rod_state()
{
    for(int attempt = 0; attempt < 100; ++attempt)
    {
        const Vector3 x0 = rand_vec3(-1.0, 1.0);
        const Vector3 x1 = x0 + rand_vec3(0.5, 1.5);
        const Vector3 x2 = x1 + rand_vec3(0.5, 1.5);
        const Vector3 u  = x1 - x0;
        const Vector3 v  = x2 - x1;
        const double  s  = u.cross(v).norm() / (u.norm() * v.norm());
        if(s > 1e-3)
            return {x0, x1, x2};
    }
    // deterministic fallback (a right angle)
    return {Vector3{0.0, 0.0, 0.0}, Vector3{1.0, 0.0, 0.0}, Vector3{1.0, 1.0, 0.0}};
}

struct SpringState
{
    Vector3 x0, x1;
};

SpringState random_spring_state()
{
    const Vector3 x0 = rand_vec3(-1.0, 1.0);
    const Vector3 d  = rand_vec3(-1.5, 1.5);
    return {x0, x0 + d};
}

/// `E_ported(X)` through the shipped generic routine, for the FD/self-consistency
/// comparisons (`X` is the 9-dof pack of the three positions).
double ported_rod_energy(double k, double dt, double r, double L0, const Vector9& X)
{
    Vector3 a, b, c;
    unpack9(X, a, b, c);
    return krb::stencil_energy<double>(k, dt, lv<3>(a), lv<3>(b), lv<3>(c), L0, r, kPi);
}
}  // namespace
namespace
{
/// The shipped, *unclamped* analytic Hessian (`ddEddX * dt^2`, before `make_spd`).
Matrix<Float, 9, 9> rod_hessian_unclamped(double k, double dt, double r, double L0, const Vector9& X)
{
    lps::LMat<double, 9, 9> H;
    sym_krb::ddEddX(H, k, lv<9>(X), L0, r, kPi);
    return to_eig(H) * (dt * dt);
}

/// The shipped, projected Hessian (`stencil_hessian`).
Matrix<Float, 9, 9> rod_hessian(double k, double dt, double r, double L0, const Vector9& X)
{
    Vector3 a, b, c;
    unpack9(X, a, b, c);
    return to_eig(
        krb::stencil_hessian<double>(k, dt, lv<3>(a), lv<3>(b), lv<3>(c), L0, r, kPi));
}

Vector9 rod_gradient(double k, double dt, double r, double L0, const Vector9& X)
{
    Vector3 a, b, c;
    unpack9(X, a, b, c);
    return to_eig(
        krb::stencil_gradient<double>(k, dt, lv<3>(a), lv<3>(b), lv<3>(c), L0, r, kPi));
}

/// `E_ported` for a spring, through the shipped generic routine.
double ported_spring_energy(double k, double L0, double r, double pi, double dt, const Vector6& X)
{
    Vector3 a, b;
    unpack6(X, a, b);
    return hs1d::stencil_energy<double>(k, L0, r, pi, dt, lv<3>(a), lv<3>(b));
}

Vector6 spring_gradient_ported(double k, double L0, double r, double pi, double dt, const Vector6& X)
{
    Vector3 a, b;
    unpack6(X, a, b);
    return to_eig(hs1d::stencil_gradient<double>(k, L0, r, pi, dt, lv<3>(a), lv<3>(b)));
}

Matrix<Float, 6, 6> spring_hessian_unclamped(double k, double L0, double r, double pi, double dt, const Vector6& X)
{
    lps::LMat<double, 6, 6> H;
    sym_hs1d::ddEddX(H, k, lv<6>(X), L0);
    return to_eig(H) * spring_vdt2(L0, r, pi, dt);
}

Matrix<Float, 6, 6> spring_hessian(double k, double L0, double r, double pi, double dt, const Vector6& X)
{
    Vector3 a, b;
    unpack6(X, a, b);
    return to_eig(hs1d::stencil_hessian<double>(k, L0, r, pi, dt, lv<3>(a), lv<3>(b)));
}
}  // namespace

// ---------------------------------------------------------------------------
//  1) hand-derived closed forms
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_c_rod_closed_forms", "[luisa][fem][constitution_c]")
{
    test_runtime();

    const double dt = 1.0;

    // =======================================================================
    //  A) the closed form vs the ported energy, on randomized states
    // =======================================================================
    {
        double worst = 0.0, scale = 1.0;
        for(int e = 0; e < 200; ++e)
        {
            const auto   s   = random_rod_state();
            const double k   = rand_uniform(0.5, 5.0);
            const double r   = rand_uniform(1e-3, 5e-2);
            const double dtv = rand_uniform(0.5, 2.0);
            // the rest length is an *independent* quantity (the rest edge lengths)
            const double L0 = rand_uniform(0.5, 3.0);

            const double host = rod_energy_closed(k, r, L0, s.x0, s.x1, s.x2) * dtv * dtv;
            const double dev  = krb::stencil_energy<double>(
                k, dtv, lv<3>(s.x0), lv<3>(s.x1), lv<3>(s.x2), L0, r, kPi);

            worst = std::max(worst, std::abs(dev - host));
            scale = std::max(scale, std::abs(host));
        }
        std::printf("[t52c] rod energy vs the closed form k pi r^4 (1-cos)/L0: worst abs "
                    "%.3e rel %.3e\n",
                    worst,
                    worst / scale);
        CHECK(worst / scale <= 1e-13);
    }

    // =======================================================================
    //  B) a hand-computed configuration: a 60 degree hinge
    //     e0 = (1,0,0), e1 = (1/2, sqrt(3)/2, 0)  =>  cos theta = 1/2
    //     =>  E = k pi r^4 (1 - 1/2)/((1 + 1/2) L0) = k pi r^4 / (3 L0)
    //     (a *right angle* would not distinguish (1-c) from (1-c)/(1+c) -- see the
    //      oracle's doc comment)
    // =======================================================================
    {
        const double k = 2.0, r = 0.5, L0 = 1.0;
        const Vector3 x0{0.0, 0.0, 0.0};
        const Vector3 x1{1.0, 0.0, 0.0};
        const Vector3 x2{1.0 + 0.5, std::sqrt(3.0) / 2.0, 0.0};
        // cos(theta) == 1/2 exactly (up to rounding) => the hand value
        const double want = k * kPi * std::pow(r, 4.0) / (3.0 * L0);  // = 0.130899...
        const double got = krb::stencil_energy<double>(
            k, dt, lv<3>(x0), lv<3>(x1), lv<3>(x2), L0, r, kPi);
        std::printf("[t52c] rod 60-degree hinge: E %.15g (hand-derived %.15g)\n", got, want);
        CHECK(rel_err(got, want) <= 1e-15);
    }

    // =======================================================================
    //  C) a *straight* rod has exactly zero energy and gradient
    //     (the axis-aligned case is exact by construction -- `e0 x e1` is
    //      componentwise zero there; the random-direction case carries the usual
    //      round-off of `(u.y (u.z f) - u.z (u.y f))`)
    // =======================================================================
    {
        const double k = 1.7, r = 0.03, dtv = 1.1;
        for(int e = 0; e < 50; ++e)
        {
            const Vector3 x0{rand_uniform(-2.0, 2.0), 0.0, 0.0};
            const Vector3 x1{x0.x() + rand_uniform(0.3, 2.0), 0.0, 0.0};
            const Vector3 x2{x1.x() + rand_uniform(0.3, 2.0), 0.0, 0.0};
            const double  L0 = (x1 - x0).norm() + (x2 - x1).norm();

            const double E = krb::stencil_energy<double>(
                k, dtv, lv<3>(x0), lv<3>(x1), lv<3>(x2), L0, r, kPi);
            const Vector9 G = rod_gradient(k, dtv, r, L0, pack9(x0, x1, x2));

            CHECK(E == 0.0);
            CHECK(max_abs(G) == 0.0);
        }

        double worst_e = 0.0, worst_g = 0.0, scale_g = 1.0;
        for(int e = 0; e < 50; ++e)
        {
            const Vector3 x0 = rand_vec3(-2.0, 2.0);
            const Vector3 dir = Vector3{rand_uniform(-1.0, 1.0), rand_uniform(-1.0, 1.0), rand_uniform(-1.0, 1.0)};
            const Vector3 u   = dir * rand_uniform(0.3, 2.0);
            const Vector3 x1  = x0 + u;
            const Vector3 x2  = x1 + u * rand_uniform(0.3, 3.0);

            const double kk = rand_uniform(0.5, 3.0), rr = rand_uniform(1e-3, 5e-2);
            const double L0 = (x1 - x0).norm() + (x2 - x1).norm();

            const double E = krb::stencil_energy<double>(
                kk, dt, lv<3>(x0), lv<3>(x1), lv<3>(x2), L0, rr, kPi);
            const Vector9 G = rod_gradient(kk, dt, rr, L0, pack9(x0, x1, x2));

            worst_e = std::max(worst_e, std::abs(E));
            worst_g = std::max(worst_g, max_abs(G));
            scale_g = std::max(scale_g, kk * kPi * std::pow(rr, 4.0) / L0);
        }
        std::printf("[t52c] straight rod: axis-aligned E == 0 and G == 0 exactly; random "
                    "directions max|E| %.3e, max|G| %.3e (scale %.3e)\n",
                    worst_e,
                    worst_g,
                    scale_g);
        CHECK(worst_e <= 1e-14 * scale_g);
        CHECK(worst_g <= 1e-12 * scale_g);
    }

    // =======================================================================
    //  D) the derived gradient: the ported gradient must equal it exactly
    // =======================================================================
    {
        double worst = 0.0;
        double worst_rel = 0.0, worst_rel_state_c = 0.0, worst_rel_state_scale = 0.0;
        for(int e = 0; e < 200; ++e)
        {
            const auto   s   = random_rod_state();
            const double k   = rand_uniform(0.5, 5.0);
            const double r   = rand_uniform(1e-3, 5e-2);
            const double dtv = rand_uniform(0.5, 2.0);
            const double L0  = rand_uniform(0.5, 3.0);

            const Vector9 host = rod_gradient_closed(k, r, L0, s.x0, s.x1, s.x2) * (dtv * dtv);
            const Vector9 dev  = rod_gradient(k, dtv, r, L0, pack9(s.x0, s.x1, s.x2));

            const double scale = std::max(1e-300, max_abs(host));
            const double abs_e = max_abs(dev - host);
            worst = std::max(worst, abs_e);
            if(abs_e / scale > worst_rel)
            {
                worst_rel             = abs_e / scale;
                worst_rel_state_c     = rod_cos_theta(s.x0, s.x1, s.x2);
                worst_rel_state_scale = scale;
            }
        }
        std::printf("[t52c] rod gradient vs the derived analytic gradient: worst abs %.3e"
                    " (worst per-state rel %.3e at |G| %.3e, cos theta %.4f)\n",
                    worst,
                    worst_rel,
                    worst_rel_state_scale,
                    worst_rel_state_c);
        CHECK(worst_rel <= 1e-9);
    }

    // =======================================================================
    //  E) the scaling laws: E is homogeneous of degree 1 in k, 4 in r, 2 in dt and
    //     -1 in L0 -- checked as exact ratios on a fixed state
    // =======================================================================
    {
        const auto s = random_rod_state();
        const double k = 1.3, r = 0.02, L0 = 1.7, dtv = 0.8;
        const auto   E = [&](double kk, double rr, double ll, double dd)
        {
            return krb::stencil_energy<double>(
                kk, dd, lv<3>(s.x0), lv<3>(s.x1), lv<3>(s.x2), ll, rr, kPi);
        };
        const double e0 = E(k, r, L0, dtv);
        CHECK(rel_err(E(2.0 * k, r, L0, dtv), 2.0 * e0) <= 1e-14);
        CHECK(rel_err(E(k, 2.0 * r, L0, dtv), 16.0 * e0) <= 1e-12);
        CHECK(rel_err(E(k, r, 2.0 * L0, dtv), 0.5 * e0) <= 1e-14);
        CHECK(rel_err(E(k, r, L0, 2.0 * dtv), 4.0 * e0) <= 1e-14);
    }
}

// ---------------------------------------------------------------------------
//  the Hookean spring's closed forms
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_c_spring_closed_forms", "[luisa][fem][constitution_c]")
{
    test_runtime();

    const double dt = 1.0;

    // =======================================================================
    //  A) the closed form vs the ported energy, on randomized states
    // =======================================================================
    {
        double worst = 0.0, scale = 1.0;
        for(int e = 0; e < 200; ++e)
        {
            const auto   s   = random_spring_state();
            const double k   = rand_uniform(0.5, 5.0);
            const double L0  = rand_uniform(0.5, 3.0);
            const double r   = rand_uniform(1e-3, 5e-2);
            const double dtv = rand_uniform(0.5, 2.0);

            const double host = spring_energy_closed(k, L0, s.x0, s.x1)
                                * spring_vdt2(L0, r, kPi, dtv);
            const double dev = ported_spring_energy(k, L0, r, kPi, dtv, pack6(s.x0, s.x1));

            worst = std::max(worst, std::abs(dev - host));
            scale = std::max(scale, std::abs(host));
        }
        std::printf("[t52c] spring energy vs the closed form k(L-L0)^2/(2 L0^2) * L0 r^2 pi "
                    "dt^2: worst abs %.3e rel %.3e\n",
                    worst,
                    worst / scale);
        CHECK(worst / scale <= 1e-14);
    }

    // =======================================================================
    //  B) a hand-computed configuration: k = 40, L0 = 2, r = 0.01, dt = 1, L = 2.1
    //     E = 40 * 0.01 / 8 * 2 * 1e-4 * pi = 3.14159265358979e-05
    // =======================================================================
    {
        const double k = 40.0, L0 = 2.0, r = 0.01, dtv = 1.0;
        const Vector3 x0{0.0, 0.0, 0.0}, x1{2.1, 0.0, 0.0};
        const double  want = 40.0 * (0.1 * 0.1) / (2.0 * 4.0) * 2.0 * 1e-4 * kPi * dtv * dtv;
        const double  got  = ported_spring_energy(k, L0, r, kPi, dtv, pack6(x0, x1));
        std::printf("[t52c] spring stretched by 0.1: E %.15g (hand-derived %.15g)\n", got, want);
        CHECK(rel_err(got, want) <= 1e-15);

        // the gradient of the same state, also hand-derived
        const Vector6 Gh = spring_gradient_closed(k, L0, x0, x1) * spring_vdt2(L0, r, kPi, dtv);
        const Vector6 Gd = spring_gradient_ported(k, L0, r, kPi, dtv, pack6(x0, x1));
        // dE/dX0 = k (L-L0)/(L0^2 L) e = 40*0.1/(4*2.1) * (-2.1,0,0) * 6.2832e-4
        std::printf("[t52c] spring gradient: G0.x %.15g (hand-derived %.15g)\n", Gd(0), Gh(0));
        CHECK(rel_err(Gd(0), Gh(0)) <= 1e-15);
        CHECK(max_abs(Gd - Gh) <= 1e-18);
    }

    // =======================================================================
    //  C) a relaxed spring (L == L0) has exactly zero energy and gradient, and its
    //     Hessian is rank-1 PSD with the single eigenvalue 2 k Vdt2 / L0^2
    // =======================================================================
    {
        // ---- exact: an axis-aligned unit-length spring at its own rest length ----
        {
            const double k = 3.0, L0 = 1.0, r = 0.02, dtv = 1.2;
            const Vector3 x0{0.0, 0.0, 0.0}, x1{1.0, 0.0, 0.0};  // L == L0 exactly
            const auto X = pack6(x0, x1);
            CHECK(ported_spring_energy(k, L0, r, kPi, dtv, X) == 0.0);
            CHECK(max_abs(spring_gradient_ported(k, L0, r, kPi, dtv, X)) == 0.0);

            // the derived rank-1 PSD Hessian: its single eigenvalue is 2 k Vdt2 / L0^2
            const auto   H     = spring_hessian(k, L0, r, kPi, dtv, X);
            const double vdt2  = spring_vdt2(L0, r, kPi, dtv);
            const double want  = 2.0 * k * vdt2 / (L0 * L0);
            const double got   = max_eigenvalue<6>(H);
            const double got_m = min_eigenvalue<6>(H);
            std::printf("[t52c] relaxed spring (exact): E == 0 and G == 0 exactly; Hessian "
                        "max eig %.15g (hand-derived 2 k Vdt2 / L0^2 = %.15g), min eig %.3e\n",
                        got,
                        want,
                        got_m);
            CHECK(rel_err(got, want) <= 1e-12);
            CHECK(got_m >= -1e-8 * got);
        }

        // ---- randomized: `L == L0` up to the construction's own rounding ---------
        const double k = 3.0, r = 0.02, dtv = 1.2;
        double worst_e = 0.0, worst_g = 0.0;
        double worst_max_rel = 0.0, worst_min_ratio = 0.0, worst_clamp_rel = 0.0;
        for(int e = 0; e < 50; ++e)
        {
            const Vector3 x0 = rand_vec3(-2.0, 2.0);
            const Vector3 d  = rand_vec3(-1.0, 1.0);
            const double  L0 = d.norm() * rand_uniform(0.5, 2.0);
            const Vector3 x1 = x0 + d.normalized() * L0;  // L == L0 to ~1e-16 relative

            const auto   X   = pack6(x0, x1);
            const double vdt2 = spring_vdt2(L0, r, kPi, dtv);

            const double E = ported_spring_energy(k, L0, r, kPi, dtv, X);
            const Vector6 G = spring_gradient_ported(k, L0, r, kPi, dtv, X);
            // `E = k Vdt2 (dL/L0)^2 / 2` with `dL/L0 <= 1e-14` after the construction's
            // rounding, and `|dE/dX| <= k Vdt2 / L0 * (dL/L0)`
            worst_e = std::max(worst_e, std::abs(E) / (k * vdt2));
            worst_g = std::max(worst_g, max_abs(G) / (k * vdt2 / L0));

            const auto   H       = spring_hessian(k, L0, r, kPi, dtv, X);
            const auto   H_raw   = spring_hessian_unclamped(k, L0, r, kPi, dtv, X);
            const double want    = 2.0 * k * vdt2 / (L0 * L0);
            const double got_max = max_eigenvalue<6>(H);
            const double got_min = min_eigenvalue<6>(H);
            worst_max_rel = std::max(worst_max_rel, std::abs(got_max - want) / want);
            worst_min_ratio = std::min(worst_min_ratio, got_min / got_max);
            worst_clamp_rel = std::max(
                worst_clamp_rel,
                max_abs(H - eigen_clamp(H_raw)) / std::max(1.0, max_abs(H)));
        }
        std::printf("[t52c] relaxed spring (random, L == L0 to 1e-16): max |E|/(k Vdt2) %.3e,"
                    " max |G|/(k Vdt2/L0) %.3e, max eig rel %.3e, min/max %.3e, clamp rel "
                    "%.3e\n",
                    worst_e,
                    worst_g,
                    worst_max_rel,
                    worst_min_ratio,
                    worst_clamp_rel);
        CHECK(worst_e <= 1e-24);
        CHECK(worst_g <= 1e-12);
        CHECK(worst_max_rel <= 1e-9);
        CHECK(worst_min_ratio >= -1e-8);
        CHECK(worst_clamp_rel <= 1e-10);
    }

    // =======================================================================
    //  C2) `pGpk`, the reference's *parameter* gradient: the generated 6-vector is
    //      exactly `dEdX / k` (its `x7 = (L - L0)/(L0^2 L)` is `dEdX`'s without the
    //      stiffness), i.e. `d^2E/dXdk` -- checked against the derived gradient at
    //      `k = 1` on random states.  The constitution itself does not call it (it is
    //      the diff-sim surface, T5.5), so this is the only consumer.
    // =======================================================================
    {
        double worst = 0.0, scale = 1.0;
        for(int e = 0; e < 3; ++e)
        {
            const auto s  = random_spring_state();
            const double L0 = rand_uniform(0.5, 3.0);
            lps::LVec<double, 6> X;
            X.segment<3>(0) = lv<3>(s.x0);
            X.segment<3>(3) = lv<3>(s.x1);

            lps::LVec<double, 6> pGpk;
            sym_hs1d::pGpk(pGpk, 1.0, X, L0);
            const Vector6 got  = to_eig(pGpk);
            const Vector6 want = spring_gradient_closed(1.0, L0, s.x0, s.x1);

            worst = std::max(worst, max_abs(got - want));
            scale = std::max(scale, max_abs(want));
            CHECK(max_abs(got - want) <= 1e-15 * std::max(1.0, max_abs(want)));
        }
        std::printf("[t52c] spring pGpk == dEdX/k (derived): worst abs %.3e (scale %.3e)//n",
                    worst,
                    scale);
    }

    // =======================================================================
    //  D) the derived gradient vs the ported one
    // =======================================================================
    {
        double worst = 0.0, scale = 1.0;
        for(int e = 0; e < 200; ++e)
        {
            const auto   s   = random_spring_state();
            const double k   = rand_uniform(0.5, 5.0);
            const double L0  = rand_uniform(0.5, 3.0);
            const double r   = rand_uniform(1e-3, 5e-2);
            const double dtv = rand_uniform(0.5, 2.0);

            const Vector6 host = spring_gradient_closed(k, L0, s.x0, s.x1)
                                 * spring_vdt2(L0, r, kPi, dtv);
            const Vector6 dev = spring_gradient_ported(k, L0, r, kPi, dtv, pack6(s.x0, s.x1));
            worst = std::max(worst, max_abs(dev - host));
            scale = std::max(scale, max_abs(host));
        }
        std::printf("[t52c] spring gradient vs the derived analytic gradient: worst abs %.3e"
                    " rel %.3e\n",
                    worst,
                    worst / scale);
        CHECK(worst / scale <= 1e-13);
    }
}

// ---------------------------------------------------------------------------
//  2) central finite differences of the *independently derived* closed forms
//
//  The FD probes the closed form (not the ported code), so a mismatch means the
//  ported analytic derivative is wrong -- not merely that the port disagrees with
//  itself.  Two steps are measured: the brief's h = 1e-6 and h = 1e-4.  The Hessian
//  of a central difference is round-off limited as `eps * |E| / h^2` against an
//  analytic Hessian of scale `|E| / L^2`, i.e. ~`4 eps L^2 / h^2` -- which is why a
//  *stationary* state needs the larger step (batch b measured the same effect for the
//  discrete-shells rest state and settled on 1e-4).  Both numbers are printed so the
//  report quotes what was actually measured.
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_c_finite_difference", "[luisa][fem][constitution_c]")
{
    test_runtime();

    const double dt         = 0.9;
    const int    n_states   = 40;
    const double h_fine     = 1e-6;  // the brief's step
    const double h_coarse   = 1e-4;  // the round-off optimum (see the comment above)

    // =======================================================================
    //  A) the Kirchhoff rod (9 element dofs)
    // =======================================================================
    {
        double worst_g_rel = 0.0;
        double worst_h_rel_fine = 0.0, worst_h_rel_coarse = 0.0;
        double worst_h_abs_fine = 0.0, worst_h_abs_coarse = 0.0;
        double worst_asym = 0.0;
        double worst_clamp_rel = 0.0, worst_clamp_abs = 0.0;
        int    indefinite = 0;

        for(int e = 0; e < n_states; ++e)
        {
            const auto s  = random_rod_state();
            const double k  = rand_uniform(0.5, 5.0);
            const double r  = rand_uniform(0.1, 0.8);
            const double L0 = rand_uniform(0.5, 3.0);

            const Vector9 X = pack9(s.x0, s.x1, s.x2);

            const auto host_E = [&](const Vector9& x) -> double
            {
                Vector3 a, b, c;
                unpack9(x, a, b, c);
                return rod_energy_closed(k, r, L0, a, b, c) * dt * dt;
            };

            const Vector9 fdG            = fd_gradient<9>(host_E, X, h_fine);
            const Vector9 G              = rod_gradient(k, dt, r, L0, X);
            const Matrix9x9 H_unclamped  = rod_hessian_unclamped(k, dt, r, L0, X);
            const Matrix9x9 H            = rod_hessian(k, dt, r, L0, X);

            const Matrix9x9 fdH_fine   = fd_hessian<9>(host_E, X, h_fine);
            const Matrix9x9 fdH_coarse = fd_hessian<9>(host_E, X, h_coarse);

            const double g_scale = std::max(1.0, max_abs(fdG));
            const double h_scale_fine   = std::max(1.0, max_abs(fdH_fine));
            const double h_scale_coarse = std::max(1.0, max_abs(fdH_coarse));

            worst_g_rel = std::max(worst_g_rel, max_abs(G - fdG) / g_scale);
            worst_h_abs_fine   = std::max(worst_h_abs_fine, max_abs(H_unclamped - fdH_fine));
            worst_h_abs_coarse = std::max(worst_h_abs_coarse, max_abs(H_unclamped - fdH_coarse));
            worst_h_rel_fine = std::max(worst_h_rel_fine,
                                        max_abs(H_unclamped - fdH_fine) / h_scale_fine);
            worst_h_rel_coarse = std::max(worst_h_rel_coarse,
                                          max_abs(H_unclamped - fdH_coarse) / h_scale_coarse);

            worst_asym = std::max(worst_asym,
                                  max_abs(H_unclamped - H_unclamped.transpose())
                                      / std::max(1.0, max_abs(H_unclamped)));

            // the reference's `make_spd` (eigenvalue clamp) as the port projects it
            const Matrix9x9 clamped = eigen_clamp(H_unclamped);
            worst_clamp_abs = std::max(worst_clamp_abs, max_abs(H - clamped));
            worst_clamp_rel = std::max(worst_clamp_rel,
                                       max_abs(H - clamped) / std::max(1.0, max_abs(clamped)));

            if(min_eigenvalue<9>(H_unclamped) < -1e-10 * max_eigenvalue<9>(H_unclamped))
                ++indefinite;
        }

        std::printf("[t52c] rod FD (h=1e-6):  gradient rel %.3e | Hessian rel %.3e (abs %.3e)\n",
                    worst_g_rel,
                    worst_h_rel_fine,
                    worst_h_abs_fine);
        std::printf("[t52c] rod FD (h=1e-4):  Hessian rel %.3e (abs %.3e)\n",
                    worst_h_rel_coarse,
                    worst_h_abs_coarse);
        std::printf("[t52c] rod Hessian asymmetry %.3e | make_spd clamp identity abs %.3e rel "
                    "%.3e | indefinite local Hessians %d/%d\n",
                    worst_asym,
                    worst_clamp_abs,
                    worst_clamp_rel,
                    indefinite,
                    n_states);

        // the brief's bounds are 1e-4 / 1e-3 at h = 1e-6; the measured values are 5-6
        // orders tighter, and the coarse (round-off-optimal) step reaches ~1e-8, so the
        // assertions are: the brief's gradient bound tightened to 1e-6, the brief's
        // Hessian bound at h = 1e-6 kept as-is (2e-4 measured, only ~5x margin) and the
        // *measured-optimal* step asserted at 1e-6.
        CHECK(worst_g_rel <= 1e-6);
        CHECK(worst_h_rel_fine <= 1e-3);
        CHECK(worst_h_rel_coarse <= 1e-6);
        CHECK(worst_asym <= 1e-12);
        CHECK(worst_clamp_rel <= 1e-10);
    }

    // =======================================================================
    //  A2) the PSD state: a rod that is *straight at rest* (a stationary state)
    //      -- the analytic Hessian must be PSD there (the reference's clamp is a
    //      no-op) and must match the FD Hessian at the measured optimum step
    // =======================================================================
    {
        double worst_h_rel = 0.0;
        double worst_min_eig = 0.0;
        for(int e = 0; e < 40; ++e)
        {
            const Vector3 x0  = rand_vec3(-1.0, 1.0);
            const Vector3 dir = Vector3{rand_uniform(-1.0, 1.0), rand_uniform(-1.0, 1.0), rand_uniform(-1.0, 1.0)};
            const Vector3 u   = dir * rand_uniform(0.5, 1.5);
            const Vector3 x1  = x0 + u;
            const Vector3 x2  = x1 + u * rand_uniform(0.5, 2.0);  // straight

            const double k  = rand_uniform(0.5, 5.0);
            const double r  = rand_uniform(0.1, 0.8);
            const double L0 = (x1 - x0).norm() + (x2 - x1).norm();

            const Vector9 X = pack9(x0, x1, x2);

            const auto host_E = [&](const Vector9& x) -> double
            {
                Vector3 a, b, c;
                unpack9(x, a, b, c);
                return rod_energy_closed(k, r, L0, a, b, c) * dt * dt;
            };

            const auto H = rod_hessian_unclamped(k, dt, r, L0, X);
            const auto fdH = fd_hessian<9>(host_E, X, h_coarse);

            worst_h_rel = std::max(worst_h_rel,
                                   max_abs(H - fdH) / std::max(1.0, max_abs(fdH)));
            // the *analytic* Hessian's PSD-ness (not the FD one's: at a stationary state
            // an h = 1e-4 central difference is still round-off dominated, so only the
            // analytic matrix is a meaningful PSD claim)
            worst_min_eig = std::min(worst_min_eig,
                                     min_eigenvalue<9>(H) / max_eigenvalue<9>(H));
        }
        std::printf("[t52c] rod straight-rest FD (h=1e-4): Hessian rel %.3e | min eig / max eig "
                    "%.3e (PSD to round-off)\n",
                    worst_h_rel,
                    worst_min_eig);
        CHECK(worst_h_rel <= 1e-4);
        CHECK(worst_min_eig >= -1e-8);
    }

    // =======================================================================
    //  B) the Hookean spring (6 element dofs)
    // =======================================================================
    {
        const double r = 0.5;  // a friendly (well-scaled) cross-section

        double worst_g_rel = 0.0;
        double worst_h_rel_fine = 0.0, worst_h_rel_coarse = 0.0;
        double worst_h_abs_fine = 0.0, worst_h_abs_coarse = 0.0;
        double worst_asym = 0.0;
        double worst_clamp_rel = 0.0;
        int    indefinite = 0;

        for(int e = 0; e < n_states; ++e)
        {
            const auto   s  = random_spring_state();
            const double k  = rand_uniform(0.5, 5.0);
            const double L0 = rand_uniform(0.5, 3.0);
            const double dtv = rand_uniform(0.5, 2.0);

            const Vector6 X = pack6(s.x0, s.x1);

            const auto host_E = [&](const Vector6& x) -> double
            {
                Vector3 a, b;
                unpack6(x, a, b);
                return spring_energy_closed(k, L0, a, b) * spring_vdt2(L0, r, kPi, dtv);
            };

            const Vector6   fdG           = fd_gradient<6>(host_E, X, h_fine);
            const Vector6   G             = spring_gradient_ported(k, L0, r, kPi, dtv, X);
            const Matrix6x6 H_unclamped   = spring_hessian_unclamped(k, L0, r, kPi, dtv, X);
            const Matrix6x6 H             = spring_hessian(k, L0, r, kPi, dtv, X);
            const Matrix6x6 fdH_fine      = fd_hessian<6>(host_E, X, h_fine);
            const Matrix6x6 fdH_coarse    = fd_hessian<6>(host_E, X, h_coarse);

            worst_g_rel = std::max(worst_g_rel, max_abs(G - fdG) / std::max(1.0, max_abs(fdG)));
            worst_h_abs_fine = std::max(worst_h_abs_fine, max_abs(H_unclamped - fdH_fine));
            worst_h_abs_coarse = std::max(worst_h_abs_coarse, max_abs(H_unclamped - fdH_coarse));
            worst_h_rel_fine = std::max(worst_h_rel_fine,
                                        max_abs(H_unclamped - fdH_fine)
                                            / std::max(1.0, max_abs(fdH_fine)));
            worst_h_rel_coarse = std::max(worst_h_rel_coarse,
                                          max_abs(H_unclamped - fdH_coarse)
                                              / std::max(1.0, max_abs(fdH_coarse)));

            worst_asym = std::max(worst_asym,
                                  max_abs(H_unclamped - H_unclamped.transpose())
                                      / std::max(1.0, max_abs(H_unclamped)));

            const Matrix6x6 clamped = eigen_clamp(H_unclamped);
            worst_clamp_rel = std::max(worst_clamp_rel,
                                       max_abs(H - clamped) / std::max(1.0, max_abs(clamped)));

            if(min_eigenvalue<6>(H_unclamped) < -1e-10 * max_eigenvalue<6>(H_unclamped))
                ++indefinite;
        }

        std::printf("[t52c] spring FD (h=1e-6): gradient rel %.3e | Hessian rel %.3e (abs %.3e)\n",
                    worst_g_rel,
                    worst_h_rel_fine,
                    worst_h_abs_fine);
        std::printf("[t52c] spring FD (h=1e-4): Hessian rel %.3e (abs %.3e)\n",
                    worst_h_rel_coarse,
                    worst_h_abs_coarse);
        std::printf("[t52c] spring Hessian asymmetry %.3e | make_spd clamp identity rel %.3e | "
                    "indefinite local Hessians %d/%d\n",
                    worst_asym,
                    worst_clamp_rel,
                    indefinite,
                    n_states);

        // the brief's bounds are 1e-4 / 1e-3 at h = 1e-6; the measured values are 5-6
        // orders tighter, and the coarse (round-off-optimal) step reaches ~1e-8, so the
        // assertions are: the brief's gradient bound tightened to 1e-6, the brief's
        // Hessian bound at h = 1e-6 kept as-is (2e-4 measured, only ~5x margin) and the
        // *measured-optimal* step asserted at 1e-6.
        CHECK(worst_g_rel <= 1e-6);
        CHECK(worst_h_rel_fine <= 1e-3);
        CHECK(worst_h_rel_coarse <= 1e-6);
        CHECK(worst_asym <= 1e-12);
        CHECK(worst_clamp_rel <= 1e-10);
    }
}

namespace
{
// ---------------------------------------------------------------------------
//  Shared helpers for the device and scene cases
// ---------------------------------------------------------------------------
template <int N>
struct ElementResult
{
    double                       energy = 0.0;
    Vector<Float, 3 * N>         gradient;
    Matrix<Float, 3 * N, 3 * N>  hessian;
};

template <typename Derived>
bool all_finite(const Eigen::MatrixBase<Derived>& m)
{
    for(Eigen::Index i = 0; i < m.size(); ++i)
        if(!std::isfinite(m.derived()(i)))
            return false;
    return true;
}

const char* classify(double v)
{
    if(std::isnan(v))
        return "NaN";
    if(std::isinf(v))
        return v > 0.0 ? "+inf" : "-inf";
    return "finite";
}

/// The doublet/triplet scratch of one constitution's slice (the layout the elastics
/// hands to a constitution: its own slice at offset 0 and *global* vertex indices as
/// block rows/cols -- the reference's convention).
template <int N>
struct SceneSlice
{
    static constexpr int Half = N * (N + 1) / 2;

    SizeT n = 0;

    lud::LDeviceBuffer<double>  energies;
    lud::LDeviceBuffer<int32_t> d_indices;
    lud::LDeviceBuffer<double>  d_values;
    lud::LDeviceBuffer<int32_t> t_rows;
    lud::LDeviceBuffer<int32_t> t_cols;
    lud::LDeviceBuffer<double>  t_values;

    void init(SizeT count)
    {
        n = count;
        energies.resize(n);
        d_indices.resize(N * n);
        d_values.resize(3 * N * n);
        t_rows.resize(Half * n);
        t_cols.resize(Half * n);
        t_values.resize(9 * Half * n);

        // sentinels: a `gradient_only` call must not touch the triplet buffers
        const std::vector<int32_t> rows(Half * n, -7);
        const std::vector<int32_t> cols(Half * n, -7);
        const std::vector<double>  vals(9 * Half * n, -7.0);
        t_rows.copy_from(span<const int32_t>(rows));
        t_cols.copy_from(span<const int32_t>(cols));
        t_values.copy_from(span<const double>(vals));
    }

    lla::LDoubletVectorView<Float, 3> doublet_view()
    {
        const auto m = static_cast<int32_t>(N * n);
        return lla::LDoubletVectorView<Float, 3>{
            m, 0, m, m, 0, m, d_indices.view(), d_values.view()};
    }

    lla::LTripletMatrixView<Float, 3, 3> triplet_view(int32_t vertex_count)
    {
        const auto m = static_cast<int32_t>(Half * n);
        return lla::LTripletMatrixView<Float, 3, 3>{
            vertex_count,
            vertex_count,
            0,
            m,
            m,
            lla::make_index_pair(0, 0),
            lla::make_index_pair(vertex_count, vertex_count),
            t_rows.view(),
            t_cols.view(),
            t_values.view()};
    }
};

/// Check a constitution's assembled energies/doublets/triplets against a per-element
/// oracle: element `e` owns the doublets `N*e .. N*e+N-1` and the triplets
/// `Half*e .. Half*e+Half-1` of *its own slice*, with the reference's `upper_LR` block
/// ordering inside the triplets.
template <int N, typename Fn>
void check_assembly(SizeT                         n,
                    const std::vector<std::array<int, N>>& elems,
                    const std::vector<double>&    dev_e,
                    const std::vector<int32_t>&   dev_di,
                    const std::vector<double>&    dev_dv,
                    const std::vector<int32_t>&   dev_tr,
                    const std::vector<int32_t>&   dev_tc,
                    const std::vector<double>&    dev_tv,
                    bool                          has_hessian,
                    Fn&&                          fn,
                    double                        tol,
                    const char*                   what)
{
    constexpr int Half = N * (N + 1) / 2;

    // Two normalisations are reported: the *global* one (relative to the largest
    // element's own magnitude) and a per-element one.  The per-element one is the
    // meaningful bound here: a near-relaxed spring element (`|L - L0| ~ eps`) amplifies
    // its own device-vs-host difference through the generated expression's cancellation,
    // so a global scale would either hide it or demand a meaningless bound.
    double worst_e = 0.0, worst_g = 0.0, worst_h = 0.0;
    double scale_e = 1.0, scale_g = 1.0, scale_h = 1.0;
    double rel_e_elem = 0.0, rel_g_elem = 0.0, rel_h_elem = 0.0;
    for(SizeT e = 0; e < n; ++e)
    {
        const auto r  = fn(e);
        const double ae = std::abs(dev_e[e] - r.energy);
        worst_e = std::max(worst_e, ae);
        scale_e = std::max(scale_e, std::abs(r.energy));
        rel_e_elem = std::max(rel_e_elem, ae / std::max(1e-300, std::abs(r.energy)));

        double ag = 0.0, sg = 0.0;
        for(int i = 0; i < N; ++i)
        {
            CHECK(dev_di[N * e + i] == elems[e][i]);
            for(int c = 0; c < 3; ++c)
            {
                const double d = std::abs(dev_dv[(N * e + i) * 3 + c] - r.gradient(3 * i + c));
                ag = std::max(ag, d);
                sg = std::max(sg, std::abs(r.gradient(3 * i + c)));
            }
        }
        worst_g = std::max(worst_g, ag);
        scale_g = std::max(scale_g, sg);
        rel_g_elem = std::max(rel_g_elem, ag / std::max(1e-300, sg));

        if(!has_hessian)
            continue;

        double ah = 0.0, sh = 0.0;
        int    k  = 0;
        for(int i = 0; i < N; ++i)
            for(int j = i; j < N; ++j, ++k)
            {
                // the reference's `upper_LR`: the pair is swapped when the left index is
                // not smaller than the right one
                const bool keep = elems[e][i] < elems[e][j];
                const int  l    = keep ? i : j;
                const int  rr   = keep ? j : i;
                CHECK(dev_tr[Half * e + k] == elems[e][l]);
                CHECK(dev_tc[Half * e + k] == elems[e][rr]);
                for(int c = 0; c < 3; ++c)
                    for(int q = 0; q < 3; ++q)
                    {
                        const double d =
                            std::abs(dev_tv[(Half * e + k) * 9 + c * 3 + q]
                                     - r.hessian(3 * l + q, 3 * rr + c));
                        ah = std::max(ah, d);
                        sh = std::max(sh, std::abs(r.hessian(3 * l + q, 3 * rr + c)));
                    }
            }
        worst_h = std::max(worst_h, ah);
        scale_h = std::max(scale_h, sh);
        rel_h_elem = std::max(rel_h_elem, ah / std::max(1e-300, sh));
    }

    const double rel_e = worst_e / scale_e;
    const double rel_g = worst_g / scale_g;
    const double rel_h = worst_h / scale_h;
    std::printf("[t52c] %s: assembled vs the element oracle  E %.3e (global rel %.3e, "
                "per-element rel %.3e)  G %.3e (%.3e, %.3e)  H %.3e (%.3e, %.3e)\n",
                what,
                worst_e,
                rel_e,
                rel_e_elem,
                worst_g,
                rel_g,
                rel_g_elem,
                worst_h,
                rel_h,
                rel_h_elem);
    CHECK(rel_e_elem <= tol);
    CHECK(rel_g_elem <= tol);
    if(has_hessian)
        CHECK(rel_h_elem <= tol);
}

// ---------------------------------------------------------------------------
//  The shared-binary probe (see the file comment): it only *reads* the built systems.
// ---------------------------------------------------------------------------
class FEMConstitutionCProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::FiniteElementMethod* fem    = nullptr;
    inline static gg::KirchhoffRodBending* krb    = nullptr;
    inline static gg::HookeanSpring1D*     spring = nullptr;

  private:
    void do_build() override
    {
        fem    = &require<gg::FiniteElementMethod>();
        krb    = find<gg::KirchhoffRodBending>({.exact = false});
        spring = find<gg::HookeanSpring1D>({.exact = false});
    }
};
REGISTER_SIM_SYSTEM(FEMConstitutionCProbe);

// ---------------------------------------------------------------------------
//  The device kernels of the device-vs-host case.
//
//  They call the *shipped* per-element routines for the energy and the gradient, and
//  the *generated* `ddEddX` directly for the Hessian.  The Hessian is deliberately the
//  unclamped one: `krb_detail::stencil_hessian` inlines the 8-sweep 9x9 Jacobi, and a
//  second copy of that in a test shader costs minutes of nvrtc (cookbook §7.17 rule 13),
//  while the *projected* device path is already covered end to end by the scene case
//  (the shipped kernel's assembled triplets are compared against the host oracle there).
// ---------------------------------------------------------------------------
auto& rod_element_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector3i>,  // hinges
                                                lcrt::Buffer<gg::LVector3>,   // xs
                                                lcrt::Buffer<double>,         // kappas
                                                lcrt::Buffer<double>,         // L0s
                                                lcrt::Buffer<double>,         // radii
                                                lcrt::Buffer<double>,         // out_e
                                                lcrt::Buffer<double>,         // out_g (9)
                                                lcrt::Buffer<double>,         // out_h (81)
                                                double,                       // dt
                                                double,                       // pi
                                                uint32_t>(                    // n
        [](lcrt::BufferVar<gg::LVector3i> hinges,
           lcrt::BufferVar<gg::LVector3>  xs,
           lcrt::BufferVar<double>        kappas,
           lcrt::BufferVar<double>        L0s,
           lcrt::BufferVar<double>        radii,
           lcrt::BufferVar<double>        out_e,
           lcrt::BufferVar<double>        out_g,
           lcrt::BufferVar<double>        out_h,
           lcrt::Var<double>              dt,
           lcrt::Var<double>              pi,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto hinge = gg::load_vec3i(hinges->read(i));
                auto x0    = gg::load_vec3(xs->read(hinge(0)));
                auto x1    = gg::load_vec3(xs->read(hinge(1)));
                auto x2    = gg::load_vec3(xs->read(hinge(2)));

                auto kappa = kappas->read(i);
                auto L0    = L0s->read(i);
                auto r     = radii->read(i);

                auto E = krb::stencil_energy<lcrt::Var<double>>(kappa, dt, x0, x1, x2, L0, r, pi);
                out_e->write(i, E);

                auto G = krb::stencil_gradient<lcrt::Var<double>>(kappa, dt, x0, x1, x2, L0, r, pi);
                for(int c = 0; c < 9; ++c)
                    out_g->write(i * 9u + c, G(c));

                auto X = krb::hinge_positions<lcrt::Var<double>>(x0, x1, x2);
                lps::LMat<lcrt::Var<double>, 9, 9> H;
                sym_krb::ddEddX(H, kappa, X, L0, r, pi);
                auto Hdt2 = H * (dt * dt);
                for(int r9 = 0; r9 < 9; ++r9)
                    for(int c9 = 0; c9 < 9; ++c9)
                        out_h->write(i * 81u + r9 * 9u + c9, Hdt2(r9, c9));
            };
        });
    return k;
}

auto& spring_element_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector2i>,  // indices
                                                lcrt::Buffer<gg::LVector3>,   // xs
                                                lcrt::Buffer<double>,         // kappas
                                                lcrt::Buffer<double>,         // L0s
                                                lcrt::Buffer<double>,         // radii
                                                lcrt::Buffer<double>,         // out_e
                                                lcrt::Buffer<double>,         // out_g (6)
                                                lcrt::Buffer<double>,         // out_h (36)
                                                double,                       // dt
                                                double,                       // pi
                                                uint32_t>(                    // n
        [](lcrt::BufferVar<gg::LVector2i> indices,
           lcrt::BufferVar<gg::LVector3>  xs,
           lcrt::BufferVar<double>        kappas,
           lcrt::BufferVar<double>        L0s,
           lcrt::BufferVar<double>        radii,
           lcrt::BufferVar<double>        out_e,
           lcrt::BufferVar<double>        out_g,
           lcrt::BufferVar<double>        out_h,
           lcrt::Var<double>              dt,
           lcrt::Var<double>              pi,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto idx = gg::load_vec2i(indices->read(i));
                auto x0  = gg::load_vec3(xs->read(idx(0)));
                auto x1  = gg::load_vec3(xs->read(idx(1)));

                auto kappa = kappas->read(i);
                auto L0    = L0s->read(i);
                auto r     = radii->read(i);

                auto E = hs1d::stencil_energy<lcrt::Var<double>>(kappa, L0, r, pi, dt, x0, x1);
                out_e->write(i, E);

                auto G = hs1d::stencil_gradient<lcrt::Var<double>>(kappa, L0, r, pi, dt, x0, x1);
                for(int c = 0; c < 6; ++c)
                    out_g->write(i * 6u + c, G(c));

                auto X = hs1d::spring_positions<lcrt::Var<double>>(x0, x1);
                lps::LMat<lcrt::Var<double>, 6, 6> H;
                sym_hs1d::ddEddX(H, kappa, X, L0);
                auto Hvdt2 = H * hs1d::edge_vdt2<lcrt::Var<double>>(L0, r, pi, dt);
                for(int r6 = 0; r6 < 6; ++r6)
                    for(int c6 = 0; c6 < 6; ++c6)
                        out_h->write(i * 36u + r6 * 6u + c6, Hvdt2(r6, c6));
            };
        });
    return k;
}

/// One hinge's rest length, exactly as the shipped kernel computes it.
double hinge_rest_length(const Vector3& x0_bar, const Vector3& x1_bar, const Vector3& x2_bar)
{
    return (x1_bar - x0_bar).norm() + (x2_bar - x1_bar).norm();
}
}  // namespace

// ---------------------------------------------------------------------------
//  3) device vs host: the *same* generic routines, S = double vs S = lcrt::Var<double>
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_c_device_vs_host", "[luisa][fem][constitution_c]")
{
    test_runtime();

    const int    count = 24;
    const double dt    = 0.7;

    // ---- the Kirchhoff rod ---------------------------------------------------
    {
        std::vector<Vector3i>    hinges;
        std::vector<Vector3>     xs;
        std::vector<double>      kappas, L0s, radii;
        std::vector<RodState>    states;

        for(int e = 0; e < count; ++e)
        {
            const auto s = random_rod_state();
            states.push_back(s);
            hinges.push_back(Vector3i{3 * e, 3 * e + 1, 3 * e + 2});
            xs.push_back(s.x0);
            xs.push_back(s.x1);
            xs.push_back(s.x2);
            kappas.push_back(rand_uniform(0.5, 5.0));
            radii.push_back(rand_uniform(1e-3, 5e-2));
            // an independent rest state (a straight rod of a random length)
            L0s.push_back(rand_uniform(0.5, 3.0));
        }

        auto hinge_dev = upload<gg::LVector3i>(hinges, [](const Vector3i& v) { return gg::to_luisa(v); });
        auto xs_dev    = upload<gg::LVector3>(xs, [](const Vector3& v) { return gg::to_luisa(v); });
        auto kappa_dev = upload(kappas);
        auto L0_dev    = upload(L0s);
        auto r_dev     = upload(radii);

        lud::LDeviceBuffer<double> out_e(count);
        lud::LDeviceBuffer<double> out_g(count * 9);
        lud::LDeviceBuffer<double> out_h(count * 81);

        const auto n32 = static_cast<uint32_t>(count);
        auto&      k   = rod_element_kernel();
        gg::frame_stream().record(k(hinge_dev.view(),
                                    xs_dev.view(),
                                    kappa_dev.view(),
                                    L0_dev.view(),
                                    r_dev.view(),
                                    out_e.view(),
                                    out_g.view(),
                                    out_h.view(),
                                    dt,
                                    kPi,
                                    n32)
                                    .dispatch(n32));
        gg::frame_stream().synchronize();

        const auto dev_e = read_device(out_e.cview());
        const auto dev_g = read_device(out_g.cview());
        const auto dev_h = read_device(out_h.cview());

        double de = 0.0, dg = 0.0, dh = 0.0;
        double se = 1.0, sg = 1.0, sh = 1.0;
        double pe_e = 0.0, pe_g = 0.0, pe_h = 0.0;  // per element, own scale
        for(int e = 0; e < count; ++e)
        {
            const auto& s = states[e];
            const auto  X = pack9(s.x0, s.x1, s.x2);
            const auto  E = krb::stencil_energy<double>(
                kappas[e], dt, lv<3>(s.x0), lv<3>(s.x1), lv<3>(s.x2), L0s[e], radii[e], kPi);
            const auto G = rod_gradient(kappas[e], dt, radii[e], L0s[e], X);
            const auto H = rod_hessian_unclamped(kappas[e], dt, radii[e], L0s[e], X);

            const double ae = std::abs(dev_e[e] - E);
            de = std::max(de, ae);
            se = std::max(se, std::abs(E));
            pe_e = std::max(pe_e, ae / std::max(1e-300, std::abs(E)));

            double ag = 0.0, sg_e = 0.0;
            for(int i = 0; i < 9; ++i)
            {
                const double d = std::abs(dev_g[e * 9 + i] - G(i));
                ag = std::max(ag, d);
                sg_e = std::max(sg_e, std::abs(G(i)));
            }
            dg = std::max(dg, ag);
            sg = std::max(sg, sg_e);
            pe_g = std::max(pe_g, ag / std::max(1e-300, sg_e));

            double ah = 0.0, sh_e = 0.0;
            for(int r9 = 0; r9 < 9; ++r9)
                for(int c9 = 0; c9 < 9; ++c9)
                {
                    const double d = std::abs(dev_h[(e * 9 + r9) * 9 + c9] - H(r9, c9));
                    ah = std::max(ah, d);
                    sh_e = std::max(sh_e, std::abs(H(r9, c9)));
                }
            dh = std::max(dh, ah);
            sh = std::max(sh, sh_e);
            pe_h = std::max(pe_h, ah / std::max(1e-300, sh_e));
        }
        std::printf("[t52c] rod device vs host: E global rel %.3e / per-element %.3e, G %.3e / "
                    "%.3e, H %.3e / %.3e\n",
                    de / se,
                    pe_e,
                    dg / sg,
                    pe_g,
                    dh / sh,
                    pe_h);
        // The bound is set by LC's *device* transcendental, not by rounding: the
        // generated body needs `|e| = std::pow(x, 1.0/2.0)`, and LC's CUDA backend
        // routes `lc_pow(lc_double, lc_double)` through `powf` for a *non-integral*
        // exponent (cuda_builtin/cuda_device_math.h), i.e. the device evaluates it in
        // single precision (~1e-7 relative) where nvcc's `pow(double, double)` in the
        // CUDA reference is accurate to 1e-15.  Measured per-element: 1.0e-7 (reported
        // above), while the *host* instantiation of the same routine matches the
        // independent oracle to 6.9e-15..2.0e-13 (the closed-form and FD cases).
        // See cookbook §7.17 rule 34.
        CHECK(pe_e <= 1e-5);
        CHECK(pe_g <= 1e-5);
        CHECK(pe_h <= 1e-5);
        CHECK(de / se <= 1e-8);
        CHECK(dg / sg <= 1e-8);
    }

    // ---- the Hookean spring --------------------------------------------------
    {
        std::vector<Vector2i> indices;
        std::vector<Vector3>  xs;
        std::vector<double>   kappas, L0s, radii;
        std::vector<SpringState> states;

        for(int e = 0; e < count; ++e)
        {
            const auto s = random_spring_state();
            states.push_back(s);
            indices.push_back(Vector2i{2 * e, 2 * e + 1});
            xs.push_back(s.x0);
            xs.push_back(s.x1);
            kappas.push_back(rand_uniform(0.5, 5.0));
            L0s.push_back(rand_uniform(0.5, 3.0));
            radii.push_back(rand_uniform(1e-3, 5e-2));
        }

        auto idx_dev   = upload<gg::LVector2i>(indices, [](const Vector2i& v) { return gg::to_luisa(v); });
        auto xs_dev    = upload<gg::LVector3>(xs, [](const Vector3& v) { return gg::to_luisa(v); });
        auto kappa_dev = upload(kappas);
        auto L0_dev    = upload(L0s);
        auto r_dev     = upload(radii);

        lud::LDeviceBuffer<double> out_e(count);
        lud::LDeviceBuffer<double> out_g(count * 6);
        lud::LDeviceBuffer<double> out_h(count * 36);

        const auto n32 = static_cast<uint32_t>(count);
        auto&      k   = spring_element_kernel();
        gg::frame_stream().record(k(idx_dev.view(),
                                    xs_dev.view(),
                                    kappa_dev.view(),
                                    L0_dev.view(),
                                    r_dev.view(),
                                    out_e.view(),
                                    out_g.view(),
                                    out_h.view(),
                                    dt,
                                    kPi,
                                    n32)
                                    .dispatch(n32));
        gg::frame_stream().synchronize();

        const auto dev_e = read_device(out_e.cview());
        const auto dev_g = read_device(out_g.cview());
        const auto dev_h = read_device(out_h.cview());

        double de = 0.0, dg = 0.0, dh = 0.0;
        double se = 1.0, sg = 1.0, sh = 1.0;
        double pe_e = 0.0, pe_g = 0.0, pe_h = 0.0;  // per element, own scale
        for(int e = 0; e < count; ++e)
        {
            const auto& s = states[e];
            const auto  X = pack6(s.x0, s.x1);
            const auto  E = ported_spring_energy(kappas[e], L0s[e], radii[e], kPi, dt, X);
            const auto  G = spring_gradient_ported(kappas[e], L0s[e], radii[e], kPi, dt, X);
            const auto  H = spring_hessian_unclamped(kappas[e], L0s[e], radii[e], kPi, dt, X);

            const double ae = std::abs(dev_e[e] - E);
            de = std::max(de, ae);
            se = std::max(se, std::abs(E));
            pe_e = std::max(pe_e, ae / std::max(1e-300, std::abs(E)));

            double ag = 0.0, sg_e = 0.0;
            for(int i = 0; i < 6; ++i)
            {
                const double d = std::abs(dev_g[e * 6 + i] - G(i));
                ag = std::max(ag, d);
                sg_e = std::max(sg_e, std::abs(G(i)));
            }
            dg = std::max(dg, ag);
            sg = std::max(sg, sg_e);
            pe_g = std::max(pe_g, ag / std::max(1e-300, sg_e));

            double ah = 0.0, sh_e = 0.0;
            for(int r6 = 0; r6 < 6; ++r6)
                for(int c6 = 0; c6 < 6; ++c6)
                {
                    const double d = std::abs(dev_h[(e * 6 + r6) * 6 + c6] - H(r6, c6));
                    ah = std::max(ah, d);
                    sh_e = std::max(sh_e, std::abs(H(r6, c6)));
                }
            dh = std::max(dh, ah);
            sh = std::max(sh, sh_e);
            pe_h = std::max(pe_h, ah / std::max(1e-300, sh_e));
        }
        std::printf("[t52c] spring device vs host: E global rel %.3e / per-element %.3e, G %.3e "
                    "/ %.3e, H %.3e / %.3e\n",
                    de / se,
                    pe_e,
                    dg / sg,
                    pe_g,
                    dh / sh,
                    pe_h);
        // same bound (and the same cause) as the rod above: the length's
        // `std::pow(x, 1.0/2.0)` is `powf` on the device
        CHECK(pe_e <= 1e-5);
        CHECK(pe_g <= 1e-5);
        CHECK(pe_h <= 1e-5);
        CHECK(de / se <= 1e-8);
        CHECK(dg / sg <= 1e-8);
    }
}

// ---------------------------------------------------------------------------
//  4) degenerate states, pinned from the *source* plus the derived closed forms
//
//  Neither constitution has any input guard in the reference (no branch, no assert,
//  no clamping of the length): whatever the generated expression evaluates to on a
//  degenerate state is what the port must return.  Where a closed form predicts the
//  exact classification (finite / +/-inf / NaN) the test asserts it; where the two
//  disagree (the rod's *folded* state, where the generated `0 * inf` differs from the
//  closed form's limit) the classification is printed and only non-finiteness is
//  asserted.  The one guard that *does* exist is `utils/codim_thickness.h`'s
//  `edge_thickness(t0, t1)` RUNTIME_CHECK (an abort, not an exception), so it is not
//  exercised here -- exactly like T5.1's `check_report_extent` note.
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_c_degenerate_inputs", "[luisa][fem][constitution_c]")
{
    test_runtime();

    const double dt = 1.0;
    const double k  = 2.0;
    const double r  = 0.3;

    // =======================================================================
    //  A) the Kirchhoff rod
    // =======================================================================
    {
        const Vector3 a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{1.0, 1.0, 0.0};

        // (a) a collapsed current configuration (all three positions equal) with a
        //     healthy rest length: `x7 = (...)^-2 = +inf`, the bracket is exactly 0
        //     => `inf * 0` = NaN
        {
            const double L0 = 2.0;
            const double E  = krb::stencil_energy<double>(k, dt, lv<3>(a), lv<3>(a), lv<3>(a), L0, r, kPi);
            const auto   G  = rod_gradient(k, dt, r, L0, pack9(a, a, a));
            const auto   H  = rod_hessian_unclamped(k, dt, r, L0, pack9(a, a, a));
            std::printf("[t52c] rod collapsed (x0 == x1 == x2): E = %s, gradient finite? %d, "
                        "Hessian finite? %d\n",
                        classify(E),
                        static_cast<int>(all_finite(G)),
                        static_cast<int>(all_finite(H)));
            CHECK(std::isnan(E));
            CHECK(!all_finite(G));
            CHECK(!all_finite(H));
        }

        // (b) a zero *rest* length (both rest edges degenerate) with a bent current
        //     state => the energy's `... / L0` is a division by zero of a positive
        //     numerator => +inf
        {
            const double E = krb::stencil_energy<double>(k, dt, lv<3>(a), lv<3>(b), lv<3>(c), 0.0, r, kPi);
            std::printf("[t52c] rod zero rest length, bent current: E = %s\n", classify(E));
            CHECK(std::isinf(E));
            CHECK(E > 0.0);
        }

        // (c) a zero rest length *and* a straight current state => `0 / 0` = NaN
        {
            const Vector3 d{2.0, 0.0, 0.0};
            const double  E = krb::stencil_energy<double>(k, dt, lv<3>(a), lv<3>(b), lv<3>(d), 0.0, r, kPi);
            std::printf("[t52c] rod zero rest length, straight current: E = %s\n", classify(E));
            CHECK(std::isnan(E));
        }

        // (d) folded back on itself (e1 antiparallel to e0): the closed form's limit is
        //     +inf, but the generated expression computes `0 * inf` => NaN (or a huge
        //     finite value, depending on the last bit of `u.v + |u||v|`).  Only
        //     non-finiteness is asserted -- this is the expression's own behaviour.
        {
            const Vector3 u  = b - a;
            const Vector3 d  = a + u * 3.0 - u * 4.0;  // e1 = -(b - a) => antiparallel
            const double  E  = krb::stencil_energy<double>(k, dt, lv<3>(a), lv<3>(b), lv<3>(d), 2.0, r, kPi);
            const auto    G  = rod_gradient(k, dt, r, 2.0, pack9(a, b, d));
            const auto    H  = rod_hessian_unclamped(k, dt, r, 2.0, pack9(a, b, d));
            std::printf("[t52c] rod folded (antiparallel edges): E = %s, gradient finite? %d, "
                        "Hessian finite? %d\n",
                        classify(E),
                        static_cast<int>(all_finite(G)),
                        static_cast<int>(all_finite(H)));
            CHECK((!std::isfinite(E) || std::abs(E) > 1e6));
            CHECK(!all_finite(G));
            CHECK(!all_finite(H));
        }

        // (e) a *reflection* is not a rod concept, but a rod whose two edges are
        //     collinear-and-opposite has already been covered; the remaining legal
        //     degenerate case is a zero-length *current* edge with a nonzero rest
        //     length, which is (a) with a different L0 -- covered above.
    }

    // =======================================================================
    //  B) the Hookean spring
    // =======================================================================
    {
        const Vector3 a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0};

        // (a) a zero-length *current* edge with a healthy rest length: the generated
        //     `E = k (L - L0)^2 / (2 L0^2)` is finite -- exactly `k / 2` -- while the
        //     gradient's `x7 = k(-L0 + L)/(L0^2 L)` is a division by zero multiplied by
        //     `x0 = 0` => NaN
        {
            const double L0 = 2.0;
            const double E  = hs1d::stencil_energy<double>(k, L0, r, kPi, dt, lv<3>(a), lv<3>(a));
            const double want = 0.5 * k * hs1d::edge_vdt2<double>(L0, r, kPi, dt);
            const auto   G = spring_gradient_ported(k, L0, r, kPi, dt, pack6(a, a));
            const auto   H = spring_hessian_unclamped(k, L0, r, kPi, dt, pack6(a, a));
            std::printf("[t52c] spring zero current length: E %.15g (hand-derived k Vdt2 / 2 = "
                        "%.15g), gradient finite? %d, Hessian finite? %d\n",
                        E,
                        want,
                        static_cast<int>(all_finite(G)),
                        static_cast<int>(all_finite(H)));
            CHECK(rel_err(E, want) <= 1e-15);
            CHECK(!all_finite(G));
            CHECK(!all_finite(H));
        }

        // (b) a zero *rest* length with a stretched current edge: the generated density's
        //     `... / L0^2` is +inf, but the kernel's own scale `Vdt2 = L0 r^2 pi dt^2` is
        //     proportional to L0 as well -- so `E * Vdt2` is `inf * 0` = NaN (measured;
        //     the pure density alone is +inf)
        {
            const double E = hs1d::stencil_energy<double>(k, 0.0, r, kPi, dt, lv<3>(a), lv<3>(b));
            const double density = [&]
            {
                lps::LVec<double, 6> X;
                X.segment<3>(0) = lv<3>(a);
                X.segment<3>(3) = lv<3>(b);
                double r_out = 0.0;
                sym_hs1d::E(r_out, k, X, 0.0);
                return r_out;
            }();
            std::printf("[t52c] spring zero rest length, stretched: density = %s, kernel "
                        "energy (density * Vdt2) = %s\n",
                        classify(density),
                        classify(E));
            CHECK(std::isinf(density));
            CHECK(std::isnan(E));
        }

        // (c) zero rest length and zero current length => 0 / 0 = NaN
        {
            const double E = hs1d::stencil_energy<double>(k, 0.0, r, kPi, dt, lv<3>(a), lv<3>(a));
            std::printf("[t52c] spring zero rest length, zero current length: E = %s\n",
                        classify(E));
            CHECK(std::isnan(E));
        }

        // (d) a compressed state (L < L0) has an *indefinite* unclamped Hessian -- the
        //     whole reason the reference calls `make_spd` on it -- and the shipped
        //     routine's projection makes it PSD
        {
            const double L0  = 2.0;
            const Vector3 x1{0.3, 0.1, 0.0};  // L = 0.316... < L0
            const auto X = pack6(a, x1);
            const auto H_raw = spring_hessian_unclamped(k, L0, r, kPi, dt, X);
            const auto H     = spring_hessian(k, L0, r, kPi, dt, X);
            const double min_raw = min_eigenvalue<6>(H_raw);
            const double max_raw = max_eigenvalue<6>(H_raw);
            const double min_clamped = min_eigenvalue<6>(H);
            const double max_clamped = max_eigenvalue<6>(H);
            std::printf("[t52c] spring compressed (L < L0): unclamped min eig %.3e (max %.3e) "
                        "=> clamped min eig %.3e (max %.3e)\n",
                        min_raw,
                        max_raw,
                        min_clamped,
                        max_clamped);
            CHECK(min_raw < 0.0);
            CHECK(min_clamped >= -1e-8 * max_clamped);
            // the projection is the identity on the *positive* spectrum
            CHECK(rel_err(max_clamped, max_raw) <= 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
//  5) real scenes through the public Engine/World API
//
//  SHARED-BINARY NOTE (brief §4bis): `luisa_fem_base_test.cpp` registers mock
//  constitutions claiming the frontend uids 9/11/12/13/17 -- including
//  HookeanSpring's 12 and Particle's 13.  In a full-suite run the method can hand
//  those (dim, uid) groups to the mocks, so the spring's block below *detects* that
//  (its own `do_init` output is compared against the scene's edge count), prints a
//  loud WARN and skips its end-to-end half.  uid 15 (KirchhoffRodBending) and uid 0
//  (the four `Empty*` placeholders) are claimed by no mock and always run; run
//  `scripts/luisa_build.py run "[constitution_c]"` to exercise the spring too.
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_c_scene", "[luisa][fem][constitution_c]")
{
    test_runtime();

    const double dt = 1.0;

    // =======================================================================
    //  A) the Kirchhoff rod: an *extra* constitution (uid 15) on a linemesh
    // =======================================================================
    {
        FEMConstitutionCProbe::fem    = nullptr;
        FEMConstitutionCProbe::krb    = nullptr;
        FEMConstitutionCProbe::spring = nullptr;

        const std::vector<Vector3>  rest_points{{0.0, 0.0, 0.0},
                                                {1.0, 0.0, 0.0},
                                                {1.0, 1.0, 0.0},
                                                {2.0, 1.0, 0.0}};
        const std::vector<Vector2i> edges{{0, 1}, {1, 2}, {2, 3}};
        // the deformed configuration: a bent polyline (the *rest* one is straight, so
        // the two hinges carry a non-zero turn angle)
        const std::vector<Vector3>  bent{{0.0, 0.0, 0.0},
                                         {1.0, 0.0, 0.0},
                                         {1.04, 0.98, 0.25},
                                         {2.1, 1.12, 0.30}};
        const double                stiffness = 5.0;
        const double                thickness = 0.02;

        auto config = Scene::default_config();
        config["sanity_check"]["enable"] = false;
        config["contact"]["enable"]      = false;
        Scene scene{config};
        auto  object   = scene.objects().create("t52c_rod");
        auto  rod_mesh = linemesh(span<const Vector3>{rest_points}, span<const Vector2i>{edges});
        label_surface(rod_mesh);
        // a 1d *base* constitution (uid 0 -> Empty1D) supplies the thickness/mass, the
        // extra constitution the bending stiffness -- the same layering the shells use
        Empty empty;
        empty.apply_to(rod_mesh, 1.0e3, thickness);
        KirchhoffRodBending krb_cons;
        krb_cons.apply_to(rod_mesh, stiffness);

        const auto [slot, rest_slot] = object->geometries().create(rod_mesh);
        (void)rest_slot;
        // the deformed state goes into the *live* slot (the rest slot is the snapshot
        // `create()` took, and the method reads `x_bars` from it)
        {
            auto sc  = slot->geometry().as<SimplicialComplex>();
            auto pos = view(sc->positions());
            REQUIRE(pos.size() == bent.size());
            for(size_t i = 0; i < bent.size(); ++i)
                pos[i] = bent[i];
        }

        Engine engine{"luisa", workspace_path("uipc_luisa_t52c_rod")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(FEMConstitutionCProbe::fem != nullptr);
        REQUIRE(FEMConstitutionCProbe::krb != nullptr);

        auto& c = *FEMConstitutionCProbe::krb;
        CHECK(c.uid() == 15);

        // ---- the expected hinges, derived from the mesh's own edge topology ------
        std::map<IndexT, std::set<IndexT>> hinge_map;
        for(const auto& e : edges)
        {
            hinge_map[e(0)].insert(e(1));
            hinge_map[e(1)].insert(e(0));
        }
        std::vector<std::array<int, 3>> expected;
        for(auto&& [v, conn] : hinge_map)
            if(conn.size() >= 2)
                for(auto v1 : conn)
                    for(auto v2 : conn)
                        if(v1 < v2)
                            expected.push_back({static_cast<int>(v1),
                                                static_cast<int>(v),
                                                static_cast<int>(v2)});

        // the geometry's global vertex offset (the method writes it during init)
        const auto vertex_offset =
            slot->geometry().as<SimplicialComplex>()->meta().find<IndexT>(
                builtin::backend_fem_vertex_offset);
        REQUIRE(vertex_offset != nullptr);
        const auto offset = vertex_offset->view().front();
        for(auto& h : expected)
            for(auto& v : h)
                v += static_cast<int>(offset);

        const SizeT n = c.h_hinges.size();
        std::printf("[t52c] KirchhoffRodBending: %llu hinges (expected %llu), vertex "
                    "offset %d\n",
                    static_cast<unsigned long long>(n),
                    static_cast<unsigned long long>(expected.size()),
                    static_cast<int>(offset));
        REQUIRE(n == expected.size());
        REQUIRE(n > 0);

        {
            auto got = c.h_hinges;
            std::vector<std::array<int, 3>> got_arr;
            for(const auto& h : got)
                got_arr.push_back({h(0), h(1), h(2)});
            std::sort(got_arr.begin(), got_arr.end());
            std::sort(expected.begin(), expected.end());
            bool same = got_arr.size() == expected.size();
            for(SizeT i = 0; same && i < got_arr.size(); ++i)
                same = (got_arr[i] == expected[i]);
            CHECK(same);
            for(const auto& k : c.h_bending_stiffness)
                CHECK(rel_err(k, stiffness) <= 1e-15);
        }

        // ---- the extent contract ------------------------------------------------
        {
            gg::FiniteElementElastics::ReportExtentInfo info;
            c.do_report_extent(info);
            // `check()` aborts (UIPC_ASSERT, not an exception) unless the constitution
            // queried `gradient_only()` and reported a zero Hessian count when it was
            // set -- so reaching the next line *is* the assertion.  A `gradient_only`
            // ReportExtentInfo cannot be built from a test (its flag is private), which
            // is the same limitation T5.1 recorded for the two invalid combinations.
            info.check("KirchhoffRodBending");
            std::printf("[t52c] KirchhoffRodBending report_extent + check() passed\n");
        }

        // ---- drive the real kernels --------------------------------------------
        const auto&     h_hinges  = c.h_hinges;
        const auto&     h_stiff   = c.h_bending_stiffness;
        auto            hinges_dev = read_device(c.hinges.cview());
        const auto      stiff_dev = read_device(c.bending_stiffnesses.cview());
        REQUIRE(hinges_dev.size() == n);
        REQUIRE(stiff_dev.size() == n);
        for(SizeT e = 0; e < n; ++e)
            for(int k = 0; k < 3; ++k)
                CHECK(hinges_dev[e](k) == h_hinges[e](k));

        gg::FiniteElementExtraConstitution::Impl extra_impl;
        extra_impl.finite_element_method = FEMConstitutionCProbe::fem;

        gg::FiniteElementExtraConstitution::BaseInfo base_info{&extra_impl, dt};
        const auto xs_dev    = read_device(base_info.xs());
        const auto xbars_dev = read_device(base_info.x_bars());
        const auto thick_dev = read_device(base_info.thicknesses());

        std::vector<Vector3> host_xs, host_xbars;
        std::vector<double>  host_thick;
        for(auto& x : xs_dev)
            host_xs.push_back(gg::to_eigen(x));
        for(auto& x : xbars_dev)
            host_xbars.push_back(gg::to_eigen(x));
        for(auto& t : thick_dev)
            host_thick.push_back(t);

        std::vector<std::array<int, 3>> elems(n);
        for(SizeT e = 0; e < n; ++e)
            for(int k = 0; k < 3; ++k)
                elems[e][k] = h_hinges[e](k);

        const auto elem_fn = [&](SizeT e) -> ElementResult<3>
        {
            const auto h  = h_hinges[e];
            const auto x0 = host_xs[h(0)], x1 = host_xs[h(1)], x2 = host_xs[h(2)];
            const auto b0 = host_xbars[h(0)], b1 = host_xbars[h(1)], b2 = host_xbars[h(2)];
            const double L0 = hinge_rest_length(b0, b1, b2);
            const double rv = host_thick[h(1)];
            const double kk = h_stiff[e];
            return {krb::stencil_energy<double>(
                        kk, dt, lv<3>(x0), lv<3>(x1), lv<3>(x2), L0, rv, kPi),
                    to_eig(krb::stencil_gradient<double>(
                        kk, dt, lv<3>(x0), lv<3>(x1), lv<3>(x2), L0, rv, kPi)),
                    to_eig(krb::stencil_hessian<double>(
                        kk, dt, lv<3>(x0), lv<3>(x1), lv<3>(x2), L0, rv, kPi))};
        };

        gg::FiniteElementElastics::Impl elastics_impl;
        elastics_impl.constitution_energy_offsets_counts.resize(1);
        elastics_impl.constitution_energy_offsets_counts.counts()[0] = static_cast<IndexT>(n);
        elastics_impl.constitution_energy_offsets_counts.scan();
        elastics_impl.constitution_gradient_offsets_counts.resize(1);
        elastics_impl.constitution_gradient_offsets_counts.counts()[0] = static_cast<IndexT>(3 * n);
        elastics_impl.constitution_gradient_offsets_counts.scan();
        elastics_impl.constitution_hessian_offsets_counts.resize(1);
        elastics_impl.constitution_hessian_offsets_counts.counts()[0] = static_cast<IndexT>(6 * n);
        elastics_impl.constitution_hessian_offsets_counts.scan();

        auto          slice        = SceneSlice<3>{};
        const int32_t vertex_count = static_cast<int32_t>(base_info.xs().size());
        slice.init(n);

        {
            gg::FiniteElementElastics::ComputeEnergyInfo base_e{
                &elastics_impl, 0, dt, slice.energies.view()};
            gg::FiniteElementExtraConstitution::ComputeEnergyInfo e_info{&extra_impl, &base_e, dt};
            REQUIRE(e_info.energies().size() == n);
            c.do_compute_energy(e_info);
        }
        const auto dev_e = read_device(slice.energies.cview());

        {
            auto g_view = slice.doublet_view();
            auto h_view = slice.triplet_view(vertex_count);
            gg::FiniteElementElastics::ComputeGradientHessianInfo base_gh{
                &elastics_impl, 0, false, dt, g_view, h_view};
            gg::FiniteElementExtraConstitution::ComputeGradientHessianInfo gh_info{
                &extra_impl, &base_gh, dt};
            REQUIRE(gh_info.gradients().doublet_count() == static_cast<int32_t>(3 * n));
            c.do_compute_gradient_hessian(gh_info);
        }
        const auto dev_di = read_device(slice.d_indices.cview());
        const auto dev_dv = read_device(slice.d_values.cview());
        const auto dev_tr = read_device(slice.t_rows.cview());
        const auto dev_tc = read_device(slice.t_cols.cview());
        const auto dev_tv = read_device(slice.t_values.cview());

        check_assembly<3>(n,
                          elems,
                          dev_e,
                          dev_di,
                          dev_dv,
                          dev_tr,
                          dev_tc,
                          dev_tv,
                          /*has_hessian=*/true,
                          elem_fn,
                          // the assembled values differ from the host oracle only through
                          // the device's float-precision `pow(x, 0.5)` (see the
                          // device_vs_host case); measured per-element 2.3e-7
                          1e-5,
                          "KirchhoffRodBending");

        // ---- the `gradient_only` pass, twice --------------------------------
        //
        //  (1) with a real (sentinel-filled) Hessian scratch: the reference's
        //      `if(gradient_only) return;` must leave the triplet buffers untouched;
        //  (2) with a **default-constructed, null-handle** Hessian view -- the state the
        //      reference's elastics actually hands over on a gradient-only pass.  This
        //      is the trap the brief calls out: `LBufferView::subview()` on a null handle
        //      crashes, so the constitution has to build the Hessian slice *conditionally*
        //      (cooked §7.17 rule 6 / the cd5f8bc9 guard).  The gradients must still be
        //      bit-identical to the full pass.
        {
            auto slice2 = SceneSlice<3>{};
            slice2.init(n);
            auto g_only = slice2.doublet_view();
            auto h_only = slice2.triplet_view(vertex_count);
            gg::FiniteElementElastics::ComputeGradientHessianInfo base_gh2{
                &elastics_impl, 0, /*gradient_only=*/true, dt, g_only, h_only};
            gg::FiniteElementExtraConstitution::ComputeGradientHessianInfo gh2{
                &extra_impl, &base_gh2, dt};
            CHECK(gh2.gradient_only() == true);
            c.do_compute_gradient_hessian(gh2);

            double untouched = 1.0;
            for(const auto row : read_device(slice2.t_rows.cview()))
                untouched = std::min(untouched, row == -7 ? 1.0 : 0.0);
            for(const auto v : read_device(slice2.t_values.cview()))
                untouched = std::min(untouched, v == -7.0 ? 1.0 : 0.0);
            for(const auto row : read_device(slice2.t_rows.cview()))
                CHECK(row == -7);
            for(const auto v : read_device(slice2.t_values.cview()))
                CHECK(v == -7.0);
            const auto g2 = read_device(slice2.d_values.cview());
            REQUIRE(g2.size() == dev_dv.size());
            double dg = 0.0;
            for(SizeT i = 0; i < g2.size(); ++i)
                dg = std::max(dg, std::abs(g2[i] - dev_dv[i]));
            std::printf("[t52c] KirchhoffRodBending gradient_only (real scratch): gradients "
                        "bit-identical %d, triplets untouched %d\n",
                        static_cast<int>(dg == 0.0),
                        static_cast<int>(untouched > 0.0));
            CHECK(dg == 0.0);
            CHECK(untouched > 0.0);
        }
        {
            auto slice3 = SceneSlice<3>{};
            slice3.init(n);
            auto g_only = slice3.doublet_view();
            // the *empty* Hessian view the elastics passes on a gradient-only pass
            lla::LTripletMatrixView<Float, 3, 3> h_empty{};
            gg::FiniteElementElastics::ComputeGradientHessianInfo base_gh3{
                &elastics_impl, 0, /*gradient_only=*/true, dt, g_only, h_empty};
            gg::FiniteElementExtraConstitution::ComputeGradientHessianInfo gh3{
                &extra_impl, &base_gh3, dt};
            CHECK(gh3.hessians().triplet_count() == 0);
            c.do_compute_gradient_hessian(gh3);

            const auto g3 = read_device(slice3.d_values.cview());
            REQUIRE(g3.size() == dev_dv.size());
            double dg = 0.0;
            for(SizeT i = 0; i < g3.size(); ++i)
                dg = std::max(dg, std::abs(g3[i] - dev_dv[i]));
            std::printf("[t52c] KirchhoffRodBending gradient_only (null-handle Hessian "
                        "view): no crash, gradients bit-identical %d\n",
                        static_cast<int>(dg == 0.0));
            CHECK(dg == 0.0);
        }

        // ---- the frame path ----------------------------------------------------
        world.advance();
        world.sync();
        world.advance();
        world.sync();
        CHECK(world.is_valid());
    }

    // =======================================================================
    //  B) the Hookean spring (uid 12, dim 1 -- the mock may own the group)
    // =======================================================================
    {
        FEMConstitutionCProbe::fem    = nullptr;
        FEMConstitutionCProbe::spring = nullptr;

        const std::vector<Vector3>  rest_points{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
        const std::vector<Vector2i> edges{{0, 1}, {1, 2}};
        // two *different* stiffnesses (3 kPa and 7 kPa) so the per-edge attribute path
        // is exercised, not just a constant
        const std::vector<double>   kappas{3.0, 7.0};
        const double                thickness = 0.01;

        auto config = Scene::default_config();
        config["sanity_check"]["enable"] = false;
        config["contact"]["enable"]      = false;
        Scene scene{config};
        auto  object   = scene.objects().create("t52c_spring");
        auto  rod_mesh = linemesh(span<const Vector3>{rest_points}, span<const Vector2i>{edges});
        label_surface(rod_mesh);
        HookeanSpring spring_cons;
        spring_cons.apply_to(rod_mesh, kappas[0], 1.0e3, thickness);
        // per-edge stiffnesses (the frontend's `apply_to` fills a constant)
        {
            auto attr = rod_mesh.edges().find<Float>("kappa");
            REQUIRE(attr != nullptr);
            auto kview = view(*attr);
            for(size_t i = 0; i < kappas.size(); ++i)
                kview[i] = kappas[i];
        }

        const auto [slot, rest_slot] = object->geometries().create(rod_mesh);
        (void)rest_slot;
        // a *stretched* configuration in the live slot (the rest stays the unit edges)
        {
            auto sc  = slot->geometry().as<SimplicialComplex>();
            auto pos = view(sc->positions());
            pos[0] = Vector3{0.0, 0.0, 0.0};
            pos[1] = Vector3{1.3, 0.15, 0.0};
            pos[2] = Vector3{2.1, 0.4, 0.2};
        }

        Engine engine{"luisa", workspace_path("uipc_luisa_t52c_spring")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(FEMConstitutionCProbe::spring != nullptr);

        auto& c = *FEMConstitutionCProbe::spring;
        CHECK(c.uid() == 12);
        CHECK(c.dim() == 1);

        const SizeT n        = c.h_kappas.size();
        const SizeT expected = edges.size();
        if(n != expected)
        {
            WARN(fmt::format("HookeanSpring (uid 12) did NOT own its (dim, uid) geometry "
                             "group in this binary: luisa_fem_base_test.cpp's mock "
                             "constitution claims uid 12 first.  Run the filtered case to "
                             "exercise it end to end."));
            std::printf("[t52c] SKIPPED HookeanSpring end-to-end checks (group owned "
                        "elsewhere: %llu of %llu elements)\n",
                        static_cast<unsigned long long>(n),
                        static_cast<unsigned long long>(expected));
        }
        else
        {
            // ---- `do_init`: the per-edge kappas --------------------------------
            {
                const auto dev_k = read_device(c.kappas.cview());
                REQUIRE(dev_k.size() == n);
                double worst = 0.0;
                for(SizeT e = 0; e < n; ++e)
                    worst = std::max(worst, rel_err(dev_k[e], kappas[e]));
                std::printf("[t52c] HookeanSpring do_init kappas vs the scene attributes: "
                            "worst rel %.3e\n",
                            worst);
                CHECK(worst <= 1e-15);
            }

            // ---- the extent contract -------------------------------------------
            {
                gg::FiniteElementElastics::ReportExtentInfo info;
                c.do_report_extent(info);
                info.check("HookeanSpring1D");
                std::printf("[t52c] HookeanSpring1D report_extent + check() passed\n");
            }

            // ---- drive the real kernels ----------------------------------------
            gg::Codim1DConstitution::BaseInfo probe{&c, 0, dt};
            const auto indices_dev = read_device(probe.indices());
            const auto xs_dev      = read_device(probe.xs());
            const auto xbars_dev   = read_device(probe.x_bars());
            const auto restlen_dev = read_device(probe.rest_lengths());
            const auto thick_dev   = read_device(probe.thicknesses());
            REQUIRE(indices_dev.size() == n);
            REQUIRE(restlen_dev.size() == n);

            std::vector<Vector3> host_xs, host_xbars;
            std::vector<double>  host_thick;
            for(auto& x : xs_dev)
                host_xs.push_back(gg::to_eigen(x));
            for(auto& x : xbars_dev)
                host_xbars.push_back(gg::to_eigen(x));
            for(auto& t : thick_dev)
                host_thick.push_back(t);

            std::vector<std::array<int, 2>> elems(n);
            for(SizeT e = 0; e < n; ++e)
            {
                const auto idx = gg::to_eigen(indices_dev[e]);
                elems[e][0]    = idx(0);
                elems[e][1]    = idx(1);
            }

            const auto elem_fn = [&](SizeT e) -> ElementResult<2>
            {
                const auto x0 = host_xs[elems[e][0]], x1 = host_xs[elems[e][1]];
                const auto b0 = host_xbars[elems[e][0]], b1 = host_xbars[elems[e][1]];
                const double L0 = (b1 - b0).norm();  // the method's own rest length
                const double rv = gg::edge_thickness(host_thick[elems[e][0]],
                                                   host_thick[elems[e][1]]);
                const double kk = c.h_kappas[e];
                return {hs1d::stencil_energy<double>(kk, L0, rv, kPi, dt, lv<3>(x0), lv<3>(x1)),
                        to_eig(hs1d::stencil_gradient<double>(
                            kk, L0, rv, kPi, dt, lv<3>(x0), lv<3>(x1))),
                        to_eig(hs1d::stencil_hessian<double>(
                            kk, L0, rv, kPi, dt, lv<3>(x0), lv<3>(x1)))};
            };

            // the method's own rest lengths must equal the hand-computed ones
            {
                double worst = 0.0;
                for(SizeT e = 0; e < n; ++e)
                {
                    const double L = (host_xbars[elems[e][1]] - host_xbars[elems[e][0]]).norm();
                    worst = std::max(worst, rel_err(restlen_dev[e], L));
                }
                std::printf("[t52c] HookeanSpring method rest lengths vs the hand model: "
                            "worst rel %.3e\n",
                            worst);
                CHECK(worst <= 1e-15);
            }

            gg::FiniteElementElastics::Impl elastics_impl;
            elastics_impl.constitution_energy_offsets_counts.resize(1);
            elastics_impl.constitution_energy_offsets_counts.counts()[0] = static_cast<IndexT>(n);
            elastics_impl.constitution_energy_offsets_counts.scan();
            elastics_impl.constitution_gradient_offsets_counts.resize(1);
            elastics_impl.constitution_gradient_offsets_counts.counts()[0] =
                static_cast<IndexT>(2 * n);
            elastics_impl.constitution_gradient_offsets_counts.scan();
            elastics_impl.constitution_hessian_offsets_counts.resize(1);
            elastics_impl.constitution_hessian_offsets_counts.counts()[0] =
                static_cast<IndexT>(3 * n);
            elastics_impl.constitution_hessian_offsets_counts.scan();

            auto          slice        = SceneSlice<2>{};
            const int32_t vertex_count = static_cast<int32_t>(probe.xs().size());
            slice.init(n);

            {
                auto g_view = slice.doublet_view();
                auto h_view = slice.triplet_view(vertex_count);
                gg::Codim1DConstitution::ComputeGradientHessianInfo gh_info{
                    &c, 0, false, dt, g_view, h_view};
                REQUIRE(gh_info.gradients().doublet_count() == static_cast<int32_t>(2 * n));
                c.do_compute_gradient_hessian(gh_info);
            }
            const auto dev_dv = read_device(slice.d_values.cview());
            const auto dev_tr = read_device(slice.t_rows.cview());
            const auto dev_tc = read_device(slice.t_cols.cview());
            const auto dev_tv = read_device(slice.t_values.cview());

            // the energy needs its own slice (the sentinels of the Hessian scratch are
            // still in place from the call above)
            {
                gg::Codim1DConstitution::ComputeEnergyInfo e_info{
                    &c, 0, dt, slice.energies.view()};
                REQUIRE(e_info.energies().size() == n);
                c.do_compute_energy(e_info);

                // the indirections of the base info must be the constitution's own slice
                CHECK(e_info.indices().size() == n);
                CHECK(e_info.dt() == dt);
            }
            const auto dev_e = read_device(slice.energies.cview());

            check_assembly<2>(n,
                              elems,
                              dev_e,
                              read_device(slice.d_indices.cview()),
                              dev_dv,
                              dev_tr,
                              dev_tc,
                              dev_tv,
                              /*has_hessian=*/true,
                              elem_fn,
                              // the same bound as the rod's assembly (measured 2.2e-7
                              // per element)
                              1e-5,
                              "HookeanSpring1D");

            // ---- the `gradient_only` pass with a null-handle Hessian view --------
            {
                auto slice2 = SceneSlice<2>{};
                slice2.init(n);
                auto g_only = slice2.doublet_view();
                lla::LTripletMatrixView<Float, 3, 3> h_empty{};
                gg::Codim1DConstitution::ComputeGradientHessianInfo gh2{
                    &c, 0, /*gradient_only=*/true, dt, g_only, h_empty};
                CHECK(gh2.gradient_only() == true);
                c.do_compute_gradient_hessian(gh2);
                const auto g2 = read_device(slice2.d_values.cview());
                REQUIRE(g2.size() == dev_dv.size());
                double dg = 0.0;
                for(SizeT i = 0; i < g2.size(); ++i)
                    dg = std::max(dg, std::abs(g2[i] - dev_dv[i]));
                std::printf("[t52c] HookeanSpring1D gradient_only (null-handle Hessian "
                            "view): no crash, gradients bit-identical %d\n",
                            static_cast<int>(dg == 0.0));
                CHECK(dg == 0.0);
            }

            world.advance();
            world.sync();
            CHECK(world.is_valid());
        }
    }

    // =======================================================================
    //  C) the uid-0 placeholders (`Empty0D/1D/2D/3D`) and `Particle0D`
    //
    //  A geometry is only picked up by the FEM method when it carries a
    //  `constitution_uid` that some *registered* constitution claims, so the uid-0
    //  placeholders are load-bearing: `Empty::apply_to` writes uid 0 and the four
    //  `Empty<d>` systems are what keeps such a geometry in the method (with zero
    //  energy).  The observables are the method's own buffers -- they only contain all
    //  five geometries if every `Empty<d>` claimed its (dim, 0) group and reported
    //  zero extents.  `Particle0D`'s `do_build`/`do_init` are no-ops in the reference,
    //  so *which* system owns the (0, 13) group is not observable; what is observable
    //  is that the group is included at all.
    // =======================================================================
    {
        FEMConstitutionCProbe::fem = nullptr;

        const std::vector<Vector3> cloud_pts{{0.0, 0.0, 0.0},
                                             {0.3, 0.1, 0.0},
                                             {0.1, 0.4, 0.2},
                                             {0.2, 0.2, 0.5}};
        const std::vector<Vector3> line_pts{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
        const std::vector<Vector2i> line_edges{{0, 1}, {1, 2}};
        const std::vector<Vector3> tri_pts{{0.0, 0.0, 0.0},
                                           {1.0, 0.0, 0.0},
                                           {0.0, 1.0, 0.0},
                                           {1.0, 1.0, 0.0}};
        const std::vector<Vector3i> tris{{0, 1, 2}, {1, 3, 2}};
        const std::vector<Vector3>  tet_pts{{0.0, 0.0, 0.0},
                                            {1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}};
        const std::vector<Vector4i> tets{{0, 1, 2, 3}};
        const std::vector<Vector3>  particle_pts{{5.0, 0.0, 0.0}, {6.0, 0.0, 0.0}};

        auto config = Scene::default_config();
        config["sanity_check"]["enable"] = false;
        config["contact"]["enable"]      = false;
        Scene scene{config};
        auto  object = scene.objects().create("t52c_placeholders");

        Empty empty;
        auto  cloud = pointcloud(span<const Vector3>{cloud_pts});
        empty.apply_to(cloud, 1.0e3, 0.01);
        object->geometries().create(cloud);

        auto line = linemesh(span<const Vector3>{line_pts}, span<const Vector2i>{line_edges});
        label_surface(line);
        empty.apply_to(line, 1.0e3, 0.01);
        object->geometries().create(line);

        auto tri = trimesh(span<const Vector3>{tri_pts}, span<const Vector3i>{tris});
        label_surface(tri);
        empty.apply_to(tri, 1.0e3, 0.01);
        object->geometries().create(tri);

        auto tet = tetmesh(span<const Vector3>{tet_pts}, span<const Vector4i>{tets});
        label_surface(tet);
        empty.apply_to(tet, 1.0e3, 0.01);
        object->geometries().create(tet);

        Particle particle;
        auto     pcloud = pointcloud(span<const Vector3>{particle_pts});
        particle.apply_to(pcloud, 1.0e3, 0.01);
        object->geometries().create(pcloud);

        Engine engine{"luisa", workspace_path("uipc_luisa_t52c_placeholders")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(FEMConstitutionCProbe::fem != nullptr);

        auto& fem = *FEMConstitutionCProbe::fem;

        const SizeT want_vertices = cloud_pts.size() + line_pts.size() + tri_pts.size()
                                    + tet_pts.size() + particle_pts.size();
        const SizeT want_0ds = cloud_pts.size() + particle_pts.size();
        // NOTE: the method's per-dimension primitive buffers hold one *element* each
        // (`LDeviceBuffer<LVector2i>`'s `size()` is the element count, not the scalar
        // count), so `codim_1ds` has one entry per edge, not two.
        const SizeT want_1ds   = line_edges.size();
        const SizeT want_2ds   = tris.size();
        const SizeT want_tets  = tets.size();

        const auto got_vertices = fem.xs().size();
        const auto got_0ds      = fem.codim_0ds().size();
        const auto got_1ds      = fem.codim_1ds().size();
        const auto got_2ds      = fem.codim_2ds().size();
        const auto got_tets     = fem.tets().size();

        std::printf("[t52c] placeholders: the method holds %llu vertices (want %llu), "
                    "codim_0ds %llu/%llu, codim_1ds %llu/%llu, codim_2ds %llu/%llu, tets "
                    "%llu/%llu\n",
                    static_cast<unsigned long long>(got_vertices),
                    static_cast<unsigned long long>(want_vertices),
                    static_cast<unsigned long long>(got_0ds),
                    static_cast<unsigned long long>(want_0ds),
                    static_cast<unsigned long long>(got_1ds),
                    static_cast<unsigned long long>(want_1ds),
                    static_cast<unsigned long long>(got_2ds),
                    static_cast<unsigned long long>(want_2ds),
                    static_cast<unsigned long long>(got_tets),
                    static_cast<unsigned long long>(want_tets));

        CHECK(got_vertices == want_vertices);
        CHECK(got_0ds == want_0ds);
        CHECK(got_1ds == want_1ds);
        CHECK(got_2ds == want_2ds);
        CHECK(got_tets == want_tets);

        // the per-dimension material-space buffers exist for every dim
        CHECK(fem.rest_lengths().size() == line_edges.size());
        CHECK(fem.rest_areas().size() == tris.size());
        CHECK(fem.rest_volumes().size() == tets.size());

        world.advance();
        world.sync();
        CHECK(world.is_valid());
    }
}
