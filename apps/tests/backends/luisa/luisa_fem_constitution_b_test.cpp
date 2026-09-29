/********************************************************************
 * @file luisa_fem_constitution_b_test.cpp
 * @brief T5.2 (batch b) — the shell FEM constitutions of the LuisaCompute backend:
 *        discrete_shell_bending (+ its two plastic variants) and the two membrane
 *        constitutions (neo_hookean_shell_2d, strain_limiting_baraff_witkin_shell_2d).
 *
 * Every oracle in this file is derived independently *here*: the reference backend is
 * CUDA-only and cannot be host-included, so nothing is copied mechanically.  The
 * oracles are
 *   (a) hand-derived closed forms — the discrete-shells bending law
 *       `D = L0 kappa (theta - theta_bar)^2 / h_bar`, the 2D Neo-Hookean shell energy
 *       `0.5 lambda (ln J)^2 + 0.5 mu (Ic - 2 - 2 ln J)` (derived from the generated
 *       expression's own symbol map, see below), the strain-limiting stretch/shear
 *       laws of Baraff-Witkin, and the elastic/plastic branches of the stress-plastic
 *       response;
 *   (b) central finite differences (h = 1e-6) *with respect to the element dofs* (12
 *       for a bending stencil, 9 for a triangle) through the ported per-element
 *       routines, on randomized admissible states;
 *   (c) an independent Eigen implementation of the reference's *definitions* (the
 *       dihedral angle formula and the `L0 / h_bar / theta_bar` reference metric) to
 *       check the transcribed `utils/dihedral_angle.h` and the constitutions'
 *       `do_init` constants;
 *   (d) structural invariants (Hessian symmetry, positive semidefiniteness at the
 *       rest/flat state, rigid-motion invariance);
 *   (e) the *history update* of the two plastic constitutions: the reference's update
 *       sequence driven step by step (including a step where the yield condition is
 *       **not** met, where the history must not change), on the host *and* on the
 *       device through the constitution's own `theta_bars` / `yield_thresholds`
 *       buffers;
 *   (f) device-vs-host equivalence of the ported per-element routines
 *       (S = double vs S = lcrt::Var<double> inside a Kernel1D), and
 *   (g) a real scene through the public Engine/World API for every constitution, where
 *       the *real* `do_init`/`do_compute_energy`/`do_compute_gradient_hessian` run on
 *       the real FEM device buffers and the assembled doublets/triplets are compared
 *       against the same hand-written host model.
 *
 * Shared-binary rule (cookbook §7.14 rule 6 + the T5.2 brief's §4bis): every system
 * registered by every test file is built for every scene, and `luisa_fem_base_test.cpp`
 * registers *mock* constitutions that claim the frontend uids — dim 2 with
 * `NeoHookeanShell`'s uid (11) and the extra bucket with `DiscreteShellBending`'s uid
 * (17).  The method hands a (dim, uid) geometry group to the *first* system registered
 * with that uid, so in a full-suite run those two groups can belong to the mocks.  The
 * scene case below *detects* that (it compares the constitution's own `do_init` output
 * against the scene's element count), says so loudly with a `WARN`, and skips only the
 * affected half **after** printing the diagnosis; the uniquely-numbered constitutions
 * (31, 32, 819) always run.  Run with `scripts/luisa_build.py run "[constitution_b]"`
 * to exercise all five (the mocks are not built then).  The probe system registered
 * here only *reads* the built systems, so it cannot change another test file's
 * expectations.
 *********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/discrete_shell_bending.h>
#include <uipc/constitution/strain_plastic_discrete_shell_bending.h>
#include <uipc/constitution/stress_plastic_discrete_shell_bending.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <uipc/constitution/strain_limiting_baraff_witkin.h>
#include <uipc/constitution/elastic_moduli.h>

#include <type_define.h>
#include <frame_stream.h>
#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_elastics.h>
#include <finite_element/codim_2d_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/constitutions/discrete_shell_bending.h>
#include <finite_element/constitutions/strain_plastic_discrete_shell_bending.h>
#include <finite_element/constitutions/stress_plastic_discrete_shell_bending.h>
#include <finite_element/constitutions/neo_hookean_shell_2d.h>
#include <finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.h>
#include <utils/matrix_assembler.h>
#include <utils/dihedral_angle.h>
#include <utils/codim_thickness.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the shell header must
// be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
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
namespace fem  = uipc::backend::luisa::fem;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;

// The ported symbols under test.
namespace sym_dsb  = uipc::backend::luisa::sym::discrete_shell_bending;
namespace sym_pdsb = uipc::backend::luisa::sym::strain_plastic_discrete_shell_bending;
namespace sym_spdsb = uipc::backend::luisa::sym::stress_plastic_discrete_shell_bending;
namespace sym_nh   = uipc::backend::luisa::sym::neo_hookean_shell_2d;
namespace sym_bws  = uipc::backend::luisa::sym::strainlimiting_baraff_witkin_shell_2d;

namespace
{
constexpr double kPi   = 3.14159265358979323846264338327950288;
constexpr double kFdH = 1e-6;
// The *stationary* (rest) states need their own central-difference step: the analytic
// gradient there is `2 L0 kappa (theta - theta_bar) / h_bar`, i.e. a cancelling
// difference of two O(1) angles, so the FD Hessian of such a state is dominated by
// round-off.  Measured over 40 random rest stencils (kFdRestH sweep, |H| ~ 1.9e1):
//   h = 1e-6 -> |dH| 7.3e-2   h = 1e-4 -> 8.6e-6   h = 1e-3 -> 4.4e-5   h = 1e-2 -> 4.4e-3
// so the FD noise falls as `1/h^2` and the truncation rises as `h^2`: 1e-4 is the
// measured optimum (worst rel 4.5e-7; the *filtered* run's RNG stream gives 1.1e-5).
constexpr double kFdRestH = 1e-4;

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

/// Smallest eigenvalue of a symmetric matrix.
template <int N>
double min_eigenvalue(const Matrix<Float, N, N>& m)
{
    Eigen::SelfAdjointEigenSolver<Matrix<Float, N, N>> es(m);
    return es.eigenvalues()(0);
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

template <int R, int C>
[[nodiscard]] inline lps::LMat<double, R, C> lm(const Matrix<Float, R, C>& m) noexcept
{
    lps::LMat<double, R, C> out;
    for(int c = 0; c < C; ++c)
        for(int r = 0; r < R; ++r)
            out(r, c) = m(r, c);
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

// ---------------------------------------------------------------------------
//  Independently typed Eigen oracles for the reference's *definitions*.
//
//  These are written from the mathematical definitions (and from the reference
//  source's own comment / symbol map), not from the ported code:
//
//   * `eig_dihedral` is the Codim-IPC dihedral angle:
//         n1 = (v1-v0) x (v2-v0), n2 = (v2-v3) x (v1-v3),
//         DA = acos(clamp(n1.n2 / sqrt(|n1|^2 |n2|^2), -1, 1)),
//         then flipped when n2 x n1 . (v1 - v2) < 0.
//   * `eig_dsb_constants` is the discrete-shells reference metric:
//         L0 = |X2 - X1| (the mid-edge), A = (|n1| + |n2|) / 2,
//         h_bar = A / 3 / L0, theta_bar = the rest dihedral angle,
//         and the (neutral) outer weight 1.
//   * `eig_dsb_energy` is the discrete-shells bending energy the generated
//     `sym/discrete_shell_bending.inl` simplifies to:
//         E = L0 kappa (theta - theta_bar)^2 / h_bar.
//   * `eig_nh_shell_energy` is the 2D Neo-Hookean shell energy the generated
//     `sym/neo_hookean_shell_2d.inl` simplifies to (its own symbol map names
//     `x29 = log(det(IB) * det(Ds^T Ds)) = 2 ln J`, `Ic = tr(F^T F)`):
//         E = 0.5 lambda (ln J)^2 + 0.5 mu (Ic - 2 - 2 ln J),  F = Ds * IB.
//   * `eig_bws_*` are Baraff-Witkin's strain-limiting laws with the fixed
//     anisotropic directions a = (1, 0), b = (0, 1):
//         I6 = a^T F^T F b, I5u = |F a|, I5v = |F b|,
//         stretch_energy = (I5u-1)^2 + [I5u>1] sr (I5u-1)^3 + (same for v),
//         E = stretchS * stretch_energy + shearS * I6^2.
//   * `eig_plastic_*` are the strain-/stress-plastic yield laws (see the history
//     test for the derivation of the update sequence).
// ---------------------------------------------------------------------------
struct Stencil
{
    Vector3 X0, X1, X2, X3;  // rest configuration
    Vector3 x0, x1, x2, x3;  // deformed configuration
};

Stencil unit_square_bending_stencil(double fold_angle = 0.0);

double eig_dihedral(const Vector3& v0,
                    const Vector3& v1,
                    const Vector3& v2,
                    const Vector3& v3)
{
    const Vector3 n1 = (v1 - v0).cross(v2 - v0);
    const Vector3 n2 = (v2 - v3).cross(v1 - v3);
    double        DA = std::acos(std::max(
        -1.0, std::min(1.0, n1.dot(n2) / std::sqrt(n1.squaredNorm() * n2.squaredNorm()))));
    if(n2.cross(n1).dot(v1 - v2) < 0.0)
        DA = -DA;
    return DA;
}

void eig_dsb_constants(const Vector3& X0,
                       const Vector3& X1,
                       const Vector3& X2,
                       const Vector3& X3,
                       double&        L0,
                       double&        h_bar,
                       double&        theta_bar,
                       double&        outer_weight)
{
    L0                 = (X2 - X1).norm();
    const Vector3 n1   = (X1 - X0).cross(X2 - X0);
    const Vector3 n2   = (X2 - X3).cross(X1 - X3);
    const double  A    = (n1.norm() + n2.norm()) / 2.0;
    h_bar              = A / 3.0 / L0;
    theta_bar          = eig_dihedral(X0, X1, X2, X3);
    outer_weight       = 1.0;
}

double eig_dsb_energy(const Stencil& s, double kappa, double L0, double h_bar, double theta_bar)
{
    const double theta = eig_dihedral(s.x0, s.x1, s.x2, s.x3);
    return L0 * kappa * (theta - theta_bar) * (theta - theta_bar) / h_bar;
}

/// `F = Ds * IB` for a triangle (the 3x2 deformation gradient of the membrane model).
Matrix<double, 3, 2> eig_F(const Stencil& s, const Matrix2x2& IB)
{
    Matrix<double, 3, 2> Ds;
    Ds.col(0) = s.x1 - s.x0;
    Ds.col(1) = s.x2 - s.x0;
    return Ds * IB;
}

double eig_nh_shell_energy(const Matrix<double, 3, 2>& F, double lambda, double mu)
{
    const double Ic = F.squaredNorm();
    const double J  = (F.transpose() * F).determinant();
    const double lnJ = 0.5 * std::log(std::max(J, std::numeric_limits<double>::min()));
    return 0.5 * lambda * lnJ * lnJ + 0.5 * mu * (Ic - 2.0 - 2.0 * lnJ);
}

double eig_bws_energy(const Matrix<double, 3, 2>& F,
                      double                      stretchS,
                      double                      shearS,
                      double                      strain_rate)
{
    const Vector2 a{1.0, 0.0};
    const Vector2 b{0.0, 1.0};

    const double I6 = a.transpose() * F.transpose() * F * b;
    const double I5u = (F * a).norm();
    const double I5v = (F * b).norm();

    const double ucoeff = I5u <= 1.0 ? 0.0 : 1.0;
    const double vcoeff = I5v <= 1.0 ? 0.0 : 1.0;

    const double stretch_energy = std::pow(I5u - 1.0, 2.0)
                                  + ucoeff * strain_rate * std::pow(I5u - 1.0, 3.0)
                                  + std::pow(I5v - 1.0, 2.0)
                                  + vcoeff * strain_rate * std::pow(I5v - 1.0, 3.0);

    return stretchS * stretch_energy + shearS * I6 * I6;
}

/// The reference's `ddEddF` — the F-space Hessian of the strain-limiting model —
/// re-transcribed here in Eigen notation (the T5.1 oracle convention: the arithmetic
/// follows the *reference source*, the notation follows Eigen, so the ported
/// `lps::LMat` transcription can be checked branch by branch).
///
/// NOTE (measured, see the report/cookbook): this reference formula is a *stabilised*
/// Hessian, not the exact derivative of the reference's own energy — it coincides with
/// the central-difference Hessian only in the regimes the model was derived for
/// (`I6 == 0` with `I5u, I5v >= 1`), and deviates by O(0.07-0.2) relative in sheared
/// states and by O(0.67) in the shrunk regime, where the true Hessian is indefinite.
Matrix6x6 eig_bws_hessian(const Matrix<double, 3, 2>& F,
                          double                     stretchS,
                          double                     shearS,
                          double                     sr)
{
    const Vector2 a{1.0, 0.0};
    const Vector2 b{0.0, 1.0};

    const double I6 = a.transpose() * F.transpose() * F * b;

    Matrix6x6 H_stretc = Matrix6x6::Zero();
    {
        const double I5u = std::max((F * a).squaredNorm(), 1e-12);
        const double I5v = std::max((F * b).squaredNorm(), 1e-12);
        const double sqrtI5u = std::sqrt(I5u);
        const double sqrtI5v = std::sqrt(I5v);

        if(sqrtI5u > 1.0)
        {
            const double v = 2.0
                             * (((sqrtI5u - 1.0) * (3.0 * sqrtI5u * sr - 3.0 * sr + 2.0))
                                / (2.0 * sqrtI5u));
            H_stretc(0, 0) = H_stretc(1, 1) = H_stretc(2, 2) = v;
        }
        if(sqrtI5v > 1.0)
        {
            const double v = 2.0
                             * (((sqrtI5v - 1.0) * (3.0 * sqrtI5v * sr - 3.0 * sr + 2.0))
                                / (2.0 * sqrtI5v));
            H_stretc(3, 3) = H_stretc(4, 4) = H_stretc(5, 5) = v;
        }

        const Vector3 fu = F.col(0).normalized();
        const Vector3 fv = F.col(1).normalized();

        const double uCoeff = (sqrtI5u > 1.0)
                                  ? (3.0 * I5u * sr - 3.0 * sr + 2.0) / sqrtI5u
                                  : 2.0;
        const double vCoeff = (sqrtI5v > 1.0)
                                  ? (3.0 * I5v * sr - 3.0 * sr + 2.0) / sqrtI5v
                                  : 2.0;

        H_stretc.block<3, 3>(0, 0) += uCoeff * (fu * fu.transpose());
        H_stretc.block<3, 3>(3, 3) += vCoeff * (fv * fv.transpose());
    }

    Matrix6x6 H_shear = Matrix6x6::Zero();
    {
        Matrix6x6 H = Matrix6x6::Zero();
        H(3, 0) = H(4, 1) = H(5, 2) = H(0, 3) = H(1, 4) = H(2, 5) = 1.0;

        const double signI6 = (I6 >= 0.0) ? 1.0 : -1.0;

        const Matrix<double, 3, 2> g = F * (a * b.transpose() + b * a.transpose());

        Vector6 vec_g = Vector6::Zero();
        vec_g.segment<3>(0) = g.col(0);
        vec_g.segment<3>(3) = g.col(1);

        const double I2      = F.squaredNorm();
        const double lambda0 = 0.5 * (I2 + std::sqrt(I2 * I2 + 12.0 * I6 * I6));
        const Vector6 q0 = (I6 * H * vec_g + lambda0 * vec_g).normalized();

        Matrix6x6 T = Matrix6x6::Identity();
        T           = 0.5 * (T + signI6 * H);

        const Vector6 Tq     = T * q0;
        const double  normTq = Tq.squaredNorm();
        H_shear = std::fabs(I6) * (T - (Tq * Tq.transpose()) / normTq)
                  + lambda0 * (q0 * q0.transpose());
        H_shear *= 2.0;
    }

    return stretchS * H_stretc + shearS * H_shear;
}

/// The reference's `wrap_angle` (two `while` loops) — the oracle the ported 4-step
/// predicated form is compared against.
double eig_wrap_angle(double angle)
{
    const double Pi    = 3.14159265358979323846264338327950288;
    const double TwoPi = 2.0 * Pi;
    while(angle > Pi)
        angle -= TwoPi;
    while(angle < -Pi)
        angle += TwoPi;
    return angle;
}

double eig_angle_delta(double theta, double theta_bar)
{
    return eig_wrap_angle(theta - theta_bar);
}

// --- strain-plastic yield law (the reference's update_plastic_state) ---------
struct PlasticState
{
    double theta_bar       = 0.0;
    double yield_threshold = 0.0;
};

/// @return true when the reference would write the history.
bool eig_strain_plastic_update(double        theta,
                               PlasticState& state,
                               double        hardening_modulus,
                               bool          verbose = false)
{
    const double delta          = eig_angle_delta(theta, state.theta_bar);
    const double plastic_excess = std::abs(delta) - state.yield_threshold;
    if(verbose)
    {
        std::printf("[t52b]   theta %.6f  theta_bar %.6f  delta %.6f  excess %.6f\n",
                    theta,
                    state.theta_bar,
                    delta,
                    plastic_excess);
    }
    if(plastic_excess <= 1e-6)
        return false;  // yield condition not met -> history unchanged

    const double direction = delta >= 0.0 ? 1.0 : -1.0;
    state.theta_bar =
        eig_wrap_angle(state.theta_bar + direction * plastic_excess);
    state.yield_threshold = state.yield_threshold + hardening_modulus * plastic_excess;
    return true;
}

// --- stress-plastic yield law (the reference's try_trial_state + response) ---
struct StressPlasticState
{
    double theta_bar    = 0.0;
    double yield_stress = 0.0;
};

double eig_stress_plastic_elastic_slope(double kappa, double L0, double h_bar)
{
    return 2.0 * kappa * L0 / h_bar;
}

static_assert(true, "");  // keep clang-format from collapsing the banner above

struct StressPlasticStep
{
    bool   written      = false;
    double delta        = 0.0;
    double tau_trial    = 0.0;
    double theta_y      = 0.0;
    double delta_gamma  = 0.0;
};

StressPlasticStep eig_stress_plastic_step(double               theta,
                                          double               theta_bar,
                                          double               kappa,
                                          double               L0,
                                          double               h_bar,
                                          double               yield_stress)
{
    StressPlasticStep out;
    const double      elastic_slope = eig_stress_plastic_elastic_slope(kappa, L0, h_bar);
    out.delta                       = eig_angle_delta(theta, theta_bar);
    out.tau_trial                   = elastic_slope * out.delta;
    out.theta_y                     = yield_stress / elastic_slope;
    out.delta_gamma = std::max(std::abs(out.delta) - out.theta_y, 0.0);
    return out;
}

bool eig_stress_plastic_update(double              theta,
                               StressPlasticState& state,
                               double              hardening_modulus,
                               double              kappa,
                               double              L0,
                               double              h_bar)
{
    const auto step = eig_stress_plastic_step(
        theta, state.theta_bar, kappa, L0, h_bar, state.yield_stress);
    if(step.delta_gamma <= 1e-6)
        return false;  // yield condition not met -> history unchanged

    const double direction = step.delta >= 0.0 ? 1.0 : -1.0;
    state.theta_bar        = eig_wrap_angle(state.theta_bar + direction * step.delta_gamma);
    state.yield_stress     = state.yield_stress + hardening_modulus * step.delta_gamma;
    return true;
}

/// The stress-plastic response (the reference's `augmented_response_from_angle_delta`):
/// @return (energy, dEdtheta, ddEddtheta).
struct Response
{
    double energy     = 0.0;
    double dEdtheta   = 0.0;
    double ddEddtheta = 0.0;
    double theta_y    = 0.0;
    double delta_gamma = 0.0;
};

Response eig_stress_plastic_response(double delta,
                                     double kappa,
                                     double L0,
                                     double h_bar,
                                     double yield_stress)
{
    const double elastic_scale = kappa * L0 / h_bar;
    const double elastic_slope = 2.0 * elastic_scale;
    const double theta_y       = yield_stress / elastic_slope;

    Response r;
    r.theta_y = theta_y;
    if(std::abs(delta) <= theta_y)
    {
        r.energy     = elastic_scale * delta * delta;
        r.dEdtheta   = elastic_slope * delta;
        r.ddEddtheta = elastic_slope;
    }
    else
    {
        const double delta_gamma = std::abs(delta) - theta_y;
        r.delta_gamma            = delta_gamma;
        r.energy     = elastic_scale * theta_y * theta_y + yield_stress * delta_gamma;
        r.dEdtheta   = (delta >= 0.0 ? 1.0 : -1.0) * yield_stress;
        r.ddEddtheta = 0.0;
    }
    return r;
}

// ---------------------------------------------------------------------------
//  the ported per-element routines, called exactly like the kernels do
// ---------------------------------------------------------------------------
double ported_dsb_energy(const Stencil& s,
                         double         kappa,
                         double         L0,
                         double         h_bar,
                         double         theta_bar,
                         double         V_bar,
                         double         dt)
{
    return gg::dsb_detail::stencil_energy<double>(lv<3>(s.x0),
                                                  lv<3>(s.x1),
                                                  lv<3>(s.x2),
                                                  lv<3>(s.x3),
                                                  L0,
                                                  h_bar,
                                                  theta_bar,
                                                  kappa,
                                                  V_bar,
                                                  dt);
}

Vector12 ported_dsb_gradient(const Stencil& s,
                             double         kappa,
                             double         L0,
                             double         h_bar,
                             double         theta_bar,
                             double         V_bar,
                             double         dt)
{
    return to_eig(gg::dsb_detail::stencil_gradient<double>(lv<3>(s.x0),
                                                           lv<3>(s.x1),
                                                           lv<3>(s.x2),
                                                           lv<3>(s.x3),
                                                           L0,
                                                           h_bar,
                                                           theta_bar,
                                                           kappa,
                                                           V_bar,
                                                           dt));
}

Matrix12x12 ported_dsb_hessian(const Stencil& s,
                               double         kappa,
                               double         L0,
                               double         h_bar,
                               double         theta_bar,
                               double         V_bar,
                               double         dt)
{
    return to_eig(gg::dsb_detail::stencil_hessian<double>(lv<3>(s.x0),
                                                          lv<3>(s.x1),
                                                          lv<3>(s.x2),
                                                          lv<3>(s.x3),
                                                          L0,
                                                          h_bar,
                                                          theta_bar,
                                                          kappa,
                                                          V_bar,
                                                          dt));
}

double ported_pdsb_energy(const Stencil& s,
                          double         kappa,
                          double         L0,
                          double         h_bar,
                          double         theta_bar,
                          double         V_bar,
                          double         dt)
{
    return gg::pdsb_detail::stencil_energy<double>(lv<3>(s.x0),
                                                   lv<3>(s.x1),
                                                   lv<3>(s.x2),
                                                   lv<3>(s.x3),
                                                   L0,
                                                   h_bar,
                                                   theta_bar,
                                                   kappa,
                                                   V_bar,
                                                   dt);
}

Vector12 ported_pdsb_gradient(const Stencil& s,
                              double         kappa,
                              double         L0,
                              double         h_bar,
                              double         theta_bar,
                              double         V_bar,
                              double         dt)
{
    return to_eig(gg::pdsb_detail::stencil_gradient<double>(lv<3>(s.x0),
                                                            lv<3>(s.x1),
                                                            lv<3>(s.x2),
                                                            lv<3>(s.x3),
                                                            L0,
                                                            h_bar,
                                                            theta_bar,
                                                            kappa,
                                                            V_bar,
                                                            dt));
}

Matrix12x12 ported_pdsb_hessian(const Stencil& s,
                                double         kappa,
                                double         L0,
                                double         h_bar,
                                double         theta_bar,
                                double         V_bar,
                                double         dt)
{
    return to_eig(gg::pdsb_detail::stencil_hessian<double>(lv<3>(s.x0),
                                                           lv<3>(s.x1),
                                                           lv<3>(s.x2),
                                                           lv<3>(s.x3),
                                                           L0,
                                                           h_bar,
                                                           theta_bar,
                                                           kappa,
                                                           V_bar,
                                                           dt));
}

double ported_spdsb_energy(const Stencil& s,
                           double         kappa,
                           double         L0,
                           double         h_bar,
                           double         theta_bar,
                           double         yield_stress,
                           double         V_bar,
                           double         dt)
{
    return gg::spdsb_detail::stencil_energy<double>(lv<3>(s.x0),
                                                    lv<3>(s.x1),
                                                    lv<3>(s.x2),
                                                    lv<3>(s.x3),
                                                    L0,
                                                    h_bar,
                                                    theta_bar,
                                                    kappa,
                                                    yield_stress,
                                                    V_bar,
                                                    dt);
}

Vector12 ported_spdsb_gradient(const Stencil& s,
                               double         kappa,
                               double         L0,
                               double         h_bar,
                               double         theta_bar,
                               double         yield_stress,
                               double         V_bar,
                               double         dt)
{
    return to_eig(gg::spdsb_detail::stencil_gradient<double>(lv<3>(s.x0),
                                                             lv<3>(s.x1),
                                                             lv<3>(s.x2),
                                                             lv<3>(s.x3),
                                                             L0,
                                                             h_bar,
                                                             theta_bar,
                                                             kappa,
                                                             yield_stress,
                                                             V_bar,
                                                             dt));
}

Matrix12x12 ported_spdsb_hessian(const Stencil& s,
                                 double         kappa,
                                 double         L0,
                                 double         h_bar,
                                 double         theta_bar,
                                 double         yield_stress,
                                 double         V_bar,
                                 double         dt)
{
    return to_eig(gg::spdsb_detail::stencil_hessian<double>(lv<3>(s.x0),
                                                            lv<3>(s.x1),
                                                            lv<3>(s.x2),
                                                            lv<3>(s.x3),
                                                            L0,
                                                            h_bar,
                                                            theta_bar,
                                                            kappa,
                                                            yield_stress,
                                                            V_bar,
                                                            dt));
}

double ported_nh_energy(const Stencil&   s,
                        double           lambda,
                        double           mu,
                        const Matrix2x2& IB,
                        double           rest_area,
                        double           thickness,
                        double           dt)
{
    return gg::nh_shell_detail::element_energy<double>(lambda,
                                                       mu,
                                                       lv<3>(s.x0),
                                                       lv<3>(s.x1),
                                                       lv<3>(s.x2),
                                                       lm(IB),
                                                       rest_area,
                                                       thickness,
                                                       dt);
}

Vector9 ported_nh_gradient(const Stencil&   s,
                           double           lambda,
                           double           mu,
                           const Matrix2x2& IB,
                           double           rest_area,
                           double           thickness,
                           double           dt)
{
    return to_eig(gg::nh_shell_detail::element_gradient<double>(lambda,
                                                                mu,
                                                                lv<3>(s.x0),
                                                                lv<3>(s.x1),
                                                                lv<3>(s.x2),
                                                                lm(IB),
                                                                rest_area,
                                                                thickness,
                                                                dt));
}

Matrix9x9 ported_nh_hessian(const Stencil&   s,
                            double           lambda,
                            double           mu,
                            const Matrix2x2& IB,
                            double           rest_area,
                            double           thickness,
                            double           dt)
{
    return to_eig(gg::nh_shell_detail::element_hessian<double>(lambda,
                                                               mu,
                                                               lv<3>(s.x0),
                                                               lv<3>(s.x1),
                                                               lv<3>(s.x2),
                                                               lm(IB),
                                                               rest_area,
                                                               thickness,
                                                               dt));
}

double ported_bws_energy(const Stencil&   s,
                         double           lambda,
                         double           mu,
                         double           strain_rate,
                         const Matrix2x2& IB,
                         double           rest_area,
                         double           dt)
{
    return gg::bw_shell_detail::element_energy<double>(lambda,
                                                       mu,
                                                       strain_rate,
                                                       rest_area,
                                                       lv<3>(s.x0),
                                                       lv<3>(s.x1),
                                                       lv<3>(s.x2),
                                                       lm(IB),
                                                       dt);
}

Vector9 ported_bws_gradient(const Stencil&   s,
                            double           lambda,
                            double           mu,
                            double           strain_rate,
                            const Matrix2x2& IB,
                            double           rest_area,
                            double           dt)
{
    return to_eig(gg::bw_shell_detail::element_gradient<double>(lambda,
                                                                mu,
                                                                strain_rate,
                                                                rest_area,
                                                                lv<3>(s.x0),
                                                                lv<3>(s.x1),
                                                                lv<3>(s.x2),
                                                                lm(IB),
                                                                dt));
}

Matrix9x9 ported_bws_hessian(const Stencil&   s,
                             double           lambda,
                             double           mu,
                             double           strain_rate,
                             const Matrix2x2& IB,
                             double           rest_area,
                             double           dt)
{
    return to_eig(gg::bw_shell_detail::element_hessian<double>(lambda,
                                                               mu,
                                                               strain_rate,
                                                               rest_area,
                                                               lv<3>(s.x0),
                                                               lv<3>(s.x1),
                                                               lv<3>(s.x2),
                                                               lm(IB),
                                                               dt));
}
}  // namespace

namespace
{
// ---------------------------------------------------------------------------
//  canonical configurations
// ---------------------------------------------------------------------------
/**
 * @brief The canonical bending stencil (`X0, X1, X2, X3`; `(X1, X2)` is the
 *        mid-edge), rest angle exactly 0.
 *
 *     X0 = (-1, 0, 0),  X1 = (0, 0, -1),  X2 = (0, 0, 1),  X3 = (1, 0, 0)
 *
 * The two plane normals are both `(0, -2, 0)`, so `theta_bar == 0` *exactly*
 * (`acos(1) == 0`), `L0 = |X2 - X1| = 2` and `A = (|n1| + |n2|)/2 = 2`, hence
 * `h_bar = A / 3 / L0 = 1/3`.  Folding the `X3` half by `phi` about the mid-edge
 * (the z axis) gives `theta = +phi` exactly at `phi = pi/2` (`acos(0)`), so the
 * discrete-shells energy of the moved stencil is the hand-derived
 * `6 kappa (phi - theta_bar)^2` (times `V_bar * dt^2`, `V_bar = 1`).
 */
Stencil unit_square_bending_stencil(double fold_angle)
{
    Stencil s;
    s.X0 = Vector3{-1.0, 0.0, 0.0};
    s.X1 = Vector3{0.0, 0.0, -1.0};
    s.X2 = Vector3{0.0, 0.0, 1.0};
    s.X3 = Vector3{1.0, 0.0, 0.0};

    s.x0 = s.X0;
    s.x1 = s.X1;
    s.x2 = s.X2;
    // rotate X3 about the mid-edge (the z axis) by `fold_angle`
    s.x3 = Vector3{std::cos(fold_angle), std::sin(fold_angle), 0.0};
    return s;
}

/// The canonical bending stencil with the *rest* configuration folded by `fold_angle`
/// as well (so `theta_bar == fold_angle` and the deformed state's fold is independent).
Stencil folded_rest_bending_stencil(double rest_fold)
{
    const auto flat = unit_square_bending_stencil(rest_fold);
    Stencil    s    = flat;
    // the "rest" configuration is the folded one, the deformed configuration follows
    s.X0 = flat.x0;
    s.X1 = flat.x1;
    s.X2 = flat.x2;
    s.X3 = flat.x3;
    return s;
}

/// The canonical membrane triangle: `X0 = 0`, `X1 = (1,0,0)`, `X2 = (0,1,0)`, whose
/// rest shape matrix is the identity (so `IB == I` and `F == Ds`).
Stencil unit_triangle()
{
    Stencil s;
    s.X0 = s.x0 = Vector3{0.0, 0.0, 0.0};
    s.X1 = s.x1 = Vector3{1.0, 0.0, 0.0};
    s.X2 = s.x2 = Vector3{0.0, 1.0, 0.0};
    s.X3 = s.x3 = Vector3::Zero();
    return s;
}

Matrix3x3 rotation_matrix(const Vector3& axis, double angle)
{
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

/// `x -> Q x + t` on a triangle.
Stencil rotate_triangle(const Stencil& s, const Matrix3x3& Q, const Vector3& t)
{
    Stencil out = s;
    out.x0      = Q * s.X0 + t;
    out.x1      = Q * s.X1 + t;
    out.x2      = Q * s.X2 + t;
    return out;
}

/// The ported per-triangle rest metric inverse (both membrane constitutions compute
/// the same 2x2 inverse of the rest shape matrix).
Matrix2x2 nh_IB(const Stencil& s)
{
    // the reference builds `X_bar` from the three rest positions and calls `A(X_bar)`
    const auto X_bar = gg::nh_shell_detail::gather_positions<double>(
        lv<3>(s.X0), lv<3>(s.X1), lv<3>(s.X2));
    return to_eig(gg::nh_shell_detail::rest_metric_inv<double>(X_bar));
}

Matrix2x2 bws_IB(const Stencil& s)
{
    return to_eig(gg::bw_shell_detail::rest_metric_inv<double>(
        lv<3>(s.X0), lv<3>(s.X1), lv<3>(s.X2)));
}

double triangle_rest_area(const Stencil& s)
{
    return 0.5 * ((s.X1 - s.X0).cross(s.X2 - s.X0)).norm();
}

// ---------------------------------------------------------------------------
//  finite differences on the element dofs
// ---------------------------------------------------------------------------
Stencil perturb_stencil(const Stencil& s, int dof, double d)
{
    Stencil out = s;
    auto    x = std::array<Vector3, 4>{out.x0, out.x1, out.x2, out.x3};
    x[dof / 3](dof % 3) += d;
    out.x0 = x[0];
    out.x1 = x[1];
    out.x2 = x[2];
    out.x3 = x[3];
    return out;
}

Stencil perturb_triangle(const Stencil& s, int dof, double d)
{
    Stencil out = s;
    auto    x = std::array<Vector3, 3>{out.x0, out.x1, out.x2};
    x[dof / 3](dof % 3) += d;
    out.x0 = x[0];
    out.x1 = x[1];
    out.x2 = x[2];
    return out;
}

/// Central-difference gradient of an element energy w.r.t. the 12 bending dofs.
template <typename Eval>
Vector12 fd_gradient_12(const Eval& eval, const Stencil& s)
{
    Vector12 g;
    for(int k = 0; k < 12; ++k)
    {
        const double ep = eval(perturb_stencil(s, k, kFdH));
        const double em = eval(perturb_stencil(s, k, -kFdH));
        g(k)            = (ep - em) / (2 * kFdH);
    }
    return g;
}

template <typename Grad>
Matrix12x12 fd_hessian_12(const Grad& grad, const Stencil& s, double h = kFdH)
{
    Matrix12x12 H;
    for(int k = 0; k < 12; ++k)
    {
        const Vector12 gp = grad(perturb_stencil(s, k, h));
        const Vector12 gm = grad(perturb_stencil(s, k, -h));
        H.col(k)          = (gp - gm) / (2 * h);
    }
    return 0.5 * (H + H.transpose());
}

/// Central-difference gradient of an element energy w.r.t. the 9 triangle dofs.
template <typename Eval>
Vector9 fd_gradient_9(const Eval& eval, const Stencil& s)
{
    Vector9 g;
    for(int k = 0; k < 9; ++k)
    {
        const double ep = eval(perturb_triangle(s, k, kFdH));
        const double em = eval(perturb_triangle(s, k, -kFdH));
        g(k)            = (ep - em) / (2 * kFdH);
    }
    return g;
}

template <typename Grad>
Matrix9x9 fd_hessian_9(const Grad& grad, const Stencil& s, double h = kFdH)
{
    Matrix9x9 H;
    for(int k = 0; k < 9; ++k)
    {
        const Vector9 gp = grad(perturb_triangle(s, k, h));
        const Vector9 gm = grad(perturb_triangle(s, k, -h));
        H.col(k)         = (gp - gm) / (2 * h);
    }
    return 0.5 * (H + H.transpose());
}

// ---------------------------------------------------------------------------
//  deterministic random admissible states
//
//  "Admissible" = a well-conditioned rest configuration (finite Dm^-1, no
//  degenerate plane) and a deformed state that is a rotation of the rest shape plus a
//  moderate affine strain, so the local Hessians stay inside their PSD branch and no
//  field-difference step crosses one of the reference's branch boundaries.
// ---------------------------------------------------------------------------
std::mt19937& rng()
{
    static std::mt19937 gen{20240917u};
    return gen;
}

double rand_uniform(double lo, double hi)
{
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng());
}

Stencil random_bending_stencil()
{
    Stencil s;
    // rest: the canonical stencil, randomly rotated/scaled and translated
    const auto base = unit_square_bending_stencil(0.0);
    const auto Q    = rotation_matrix(Vector3{rand_uniform(-1, 1),
                                             rand_uniform(-1, 1),
                                             rand_uniform(-1, 1)},
                                      rand_uniform(-3.14, 3.14));
    const double scale = rand_uniform(0.7, 1.4);
    const Vector3 t{rand_uniform(-1, 1), rand_uniform(-1, 1), rand_uniform(-1, 1)};

    s.X0 = Q * base.X0 * scale + t;
    s.X1 = Q * base.X1 * scale + t;
    s.X2 = Q * base.X2 * scale + t;
    s.X3 = Q * base.X3 * scale + t;

    // deform: fold by a moderate angle plus a small random affine strain
    Matrix3x3 strain = Matrix3x3::Identity();
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            strain(r, c) = (r == c ? 1.0 : 0.0) + rand_uniform(-0.15, 0.15);

    const double fold = rand_uniform(-1.0, 1.0);
    auto         folded = unit_square_bending_stencil(fold);
    auto         map_x = [&](const Vector3& X) -> Vector3 { return strain * Q * X * scale + t; };

    s.x0 = map_x(folded.x0);
    s.x1 = map_x(folded.x1);
    s.x2 = map_x(folded.x2);
    s.x3 = map_x(folded.x3);
    return s;
}

Stencil random_triangle()
{
    Stencil s;
    s.X0 = Vector3{0.0, 0.0, 0.0};
    s.X1 = Vector3{1.0, 0.15, -0.1} * rand_uniform(0.7, 1.4);
    s.X2 = Vector3{-0.1, 1.0, 0.2} * rand_uniform(0.7, 1.4);

    const auto Q = rotation_matrix(Vector3{rand_uniform(-1, 1),
                                          rand_uniform(-1, 1),
                                          rand_uniform(-1, 1)},
                                   rand_uniform(-3.14, 3.14));
    Matrix3x3 strain = Matrix3x3::Identity();
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            strain(r, c) = (r == c ? 1.0 : 0.0) + rand_uniform(-0.2, 0.2);

    const Vector3 t{rand_uniform(-1, 1), rand_uniform(-1, 1), rand_uniform(-1, 1)};
    auto          map_x = [&](const Vector3& X) -> Vector3 { return Q * strain * X + t; };

    // rest and deformed configurations (the deformed one keeps the same shape family,
    // so Dm^-1 stays well conditioned)
    s.X0 = s.X0 + t;
    s.X1 = s.X1 + t;
    s.X2 = s.X2 + t;
    s.x0 = map_x(s.X0 - t);
    s.x1 = map_x(s.X1 - t);
    s.x2 = map_x(s.X2 - t);
    return s;
}
}  // namespace

// ---------------------------------------------------------------------------
//  the shared-binary probe (see the file comment)
// ---------------------------------------------------------------------------
class FEMConstitutionBProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::DiscreteShellBending*                    dsb     = nullptr;
    inline static gg::StrainPlasticDiscreteShellBending*       pdsb    = nullptr;
    inline static gg::StressPlasticDiscreteShellBending*       spdsb   = nullptr;
    inline static gg::NeoHookeanShell2D*                       nh      = nullptr;
    inline static gg::StrainLimitingBaraffWitkinShell2D*       bws     = nullptr;
    inline static gg::FiniteElementMethod*                     fem     = nullptr;
    inline static gg::FiniteElementElastics*                   elastics = nullptr;

  private:
    void do_build() override
    {
        fem      = &require<gg::FiniteElementMethod>();
        elastics = &require<gg::FiniteElementElastics>();

        // `find<T>({.exact = false})` returns the registered system of type T when the
        // scene's constitution uids made the engine build it (else nullptr).
        dsb   = find<gg::DiscreteShellBending>({.exact = false});
        pdsb  = find<gg::StrainPlasticDiscreteShellBending>({.exact = false});
        spdsb = find<gg::StressPlasticDiscreteShellBending>({.exact = false});
        nh    = find<gg::NeoHookeanShell2D>({.exact = false});
        bws   = find<gg::StrainLimitingBaraffWitkinShell2D>({.exact = false});
    }
};
REGISTER_SIM_SYSTEM(FEMConstitutionBProbe);

// ---------------------------------------------------------------------------
//  1) hand-derived closed forms
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_b_closed_forms", "[luisa][fem][constitution_b]")
{
    test_runtime();

    const double kappa = 1.3;
    const double dt    = 1.0;

    // =======================================================================
    //  A) the dihedral-angle utility against the reference's *definition*
    // =======================================================================
    {
        double worst_angle = 0.0, worst_grad = 0.0, worst_hess = 0.0;
        for(int trial = 0; trial < 200; ++trial)
        {
            const auto s = random_bending_stencil();

            // the ported routine
            double theta = 0.0;
            uipc::backend::luisa::dihedral_angle(
                lv<3>(s.x0), lv<3>(s.x1), lv<3>(s.x2), lv<3>(s.x3), theta);
            worst_angle = std::max(worst_angle,
                                   rel_err(theta, eig_dihedral(s.x0, s.x1, s.x2, s.x3)));

            lps::LVec<double, 12> g_ported;
            uipc::backend::luisa::dihedral_angle_gradient(
                lv<3>(s.x0), lv<3>(s.x1), lv<3>(s.x2), lv<3>(s.x3), g_ported);

            // the oracle: central differences of the *definition*
            auto theta_of = [&](const Stencil& t)
            { return eig_dihedral(t.x0, t.x1, t.x2, t.x3); };
            const Vector12 g_fd = fd_gradient_12(theta_of, s);
            worst_grad          = std::max(worst_grad, max_abs(to_eig(g_ported) - g_fd));

            lps::LMat<double, 12, 12> h_ported;
            uipc::backend::luisa::dihedral_angle_hessian(
                lv<3>(s.x0), lv<3>(s.x1), lv<3>(s.x2), lv<3>(s.x3), h_ported);

            auto grad_of = [&](const Stencil& t)
            {
                lps::LVec<double, 12> g;
                uipc::backend::luisa::dihedral_angle_gradient(
                    lv<3>(t.x0), lv<3>(t.x1), lv<3>(t.x2), lv<3>(t.x3), g);
                return to_eig(g);
            };
            const Matrix12x12 h_fd = fd_hessian_12(grad_of, s);
            worst_hess = std::max(worst_hess, max_abs(to_eig(h_ported) - h_fd));
        }
        std::printf("[t52b] dihedral angle vs the Eigen definition: %.3e;"
                    " gradient (analytic - FD) %.3e; Hessian (analytic - FD) %.3e\n",
                    worst_angle,
                    worst_grad,
                    worst_hess);
        CHECK(worst_angle <= 1e-12);
        CHECK(worst_grad <= 1e-6);
        CHECK(worst_hess <= 1e-5);
    }

    // =======================================================================
    //  B) `wrap_angle` — the ported 4-step predicated form vs the reference's
    //     two `while` loops (`|a| <= 10 pi`, far beyond every call site)
    // =======================================================================
    {
        int failures = 0;
        for(int i = 0; i < 4000; ++i)
        {
            const double a = rand_uniform(-11.0 * kPi, 11.0 * kPi);
            // the reference's `while` loops, applied to the same double
            const double expect = eig_wrap_angle(a);
            if(sym_pdsb::wrap_angle<double>(a) != expect)
                ++failures;
            if(sym_spdsb::wrap_angle<double>(a) != expect)
                ++failures;
        }
        CHECK(failures == 0);

        // the two derivations of the plastic laws agree on their angle deltas
        for(int i = 0; i < 200; ++i)
        {
            const double theta     = rand_uniform(-3.14159265358979, 3.14159265358979);
            const double theta_bar = rand_uniform(-3.14159265358979, 3.14159265358979);
            CHECK(std::abs(sym_pdsb::angle_delta<double>(theta, theta_bar)
                           - eig_angle_delta(theta, theta_bar))
                  <= 1e-15);
            CHECK(std::abs(sym_spdsb::angle_delta<double>(theta, theta_bar)
                           - eig_angle_delta(theta, theta_bar))
                  <= 1e-15);
        }
    }

    // =======================================================================
    //  C) discrete shells (`discrete_shell_bending`)
    //
    //  The canonical stencil has L0 = 2, h_bar = 1/3, theta_bar = 0, V_bar = 1, so
    //  E_element = V_bar dt^2 L0 kappa (theta - theta_bar)^2 / h_bar
    //            = 6 kappa (theta - theta_bar)^2 for dt = 1.
    // =======================================================================
    {
        const auto flat = unit_square_bending_stencil(0.0);

        double L0 = 0.0, h_bar = 0.0, theta_bar = 0.0, V_bar = 0.0;
        sym_dsb::compute_constants<double>(L0,
                                           h_bar,
                                           theta_bar,
                                           V_bar,
                                           lv<3>(flat.X0),
                                           lv<3>(flat.X1),
                                           lv<3>(flat.X2),
                                           lv<3>(flat.X3),
                                           0.001,
                                           0.001,
                                           0.001,
                                           0.001);

        // the hand-derived constants of the canonical stencil
        CHECK(rel_err(L0, 2.0) <= 1e-15);
        CHECK(rel_err(h_bar, 1.0 / 3.0) <= 1e-15);
        CHECK(std::abs(theta_bar) == 0.0);
        CHECK(rel_err(V_bar, 1.0) <= 1e-15);

        // an independent Eigen implementation of the same reference metric
        double L0_o = 0.0, h_bar_o = 0.0, theta_bar_o = 0.0, outer_o = 0.0;
        eig_dsb_constants(flat.X0,
                          flat.X1,
                          flat.X2,
                          flat.X3,
                          L0_o,
                          h_bar_o,
                          theta_bar_o,
                          outer_o);
        CHECK(rel_err(L0, L0_o) <= 1e-15);
        CHECK(rel_err(h_bar, h_bar_o) <= 1e-15);
        CHECK(rel_err(theta_bar, theta_bar_o) <= 1e-15);
        CHECK(rel_err(V_bar, outer_o) <= 1e-15);

        // ---- a flat rest configuration has zero bending energy and zero gradient --
        CHECK(std::abs(ported_dsb_energy(flat, kappa, L0, h_bar, theta_bar, V_bar, dt))
              == 0.0);
        CHECK(max_abs(ported_dsb_gradient(flat, kappa, L0, h_bar, theta_bar, V_bar, dt))
              <= 1e-15);
        {
            const Matrix12x12 H = ported_dsb_hessian(flat, kappa, L0, h_bar, theta_bar, V_bar, dt);
            CHECK(max_abs(H - H.transpose()) <= 1e-12);
            CHECK(min_eigenvalue(H) >= -1e-8);
            // the energy is invariant under rigid motions, so the flat state's Hessian
            // annihilates them: the 12-vector of a pure translation is in its kernel
            Vector12 translation;
            for(int i = 0; i < 4; ++i)
                for(int r = 0; r < 3; ++r)
                    translation(3 * i + r) = 1.0;
            CHECK(max_abs(H * translation) <= 1e-8);
        }

        // ---- folded by exactly pi/2: E = 6 kappa (pi/2)^2 ------------------------
        {
            const double    phi    = kPi / 2.0;
            const Stencil   folded = unit_square_bending_stencil(phi);
            const double    E      = ported_dsb_energy(folded, kappa, L0, h_bar, theta_bar, V_bar, dt);
            const double    expect = 6.0 * kappa * phi * phi;
            CHECK(rel_err(E, expect) <= 1e-13);
            // against the independent Eigen oracle as well
            CHECK(rel_err(E, eig_dsb_energy(folded, kappa, L0, h_bar, theta_bar)) <= 1e-13);

            // the gradient: dE/dx = (dEdtheta) * dtheta/dx, so it must equal the FD of
            // the energy (the state is smooth and well conditioned)
            auto energy_of = [&](const Stencil& s)
            { return ported_dsb_energy(s, kappa, L0, h_bar, theta_bar, V_bar, dt); };
            const Vector12 g_analytic =
                ported_dsb_gradient(folded, kappa, L0, h_bar, theta_bar, V_bar, dt);
            const Vector12 g_fd = fd_gradient_12(energy_of, folded);
            std::printf("[t52b] DSB folded closed form: E %.12f (expect %.12f)"
                        "  |G_analytic - G_FD| %.3e\n",
                        E,
                        expect,
                        max_abs(g_analytic - g_fd));
            CHECK(max_abs(g_analytic - g_fd) <= 1e-6);
        }

        // ---- a folded rest state: E = 6 kappa (theta - theta_bar)^2 --------------
        {
            const double rest_fold = 0.3;
            const double cur_fold  = 0.9;
            const auto   rest      = folded_rest_bending_stencil(rest_fold);
            const auto   cur       = unit_square_bending_stencil(cur_fold);

            double L0_r = 0.0, h_bar_r = 0.0, theta_bar_r = 0.0, V_bar_r = 0.0;
            sym_dsb::compute_constants<double>(L0_r,
                                               h_bar_r,
                                               theta_bar_r,
                                               V_bar_r,
                                               lv<3>(rest.X0),
                                               lv<3>(rest.X1),
                                               lv<3>(rest.X2),
                                               lv<3>(rest.X3),
                                               0.001,
                                               0.001,
                                               0.001,
                                               0.001);
            CHECK(rel_err(theta_bar_r, rest_fold) <= 1e-15);

            const double E = ported_dsb_energy(cur, kappa, L0_r, h_bar_r, theta_bar_r, V_bar_r, dt);
            CHECK(rel_err(E, 6.0 * kappa * (cur_fold - rest_fold) * (cur_fold - rest_fold))
                  <= 1e-13);

            // scaling laws: linear in kappa and V_bar, quadratic in dt
            const double E1 = E;
            CHECK(rel_err(ported_dsb_energy(cur, 2.0 * kappa, L0_r, h_bar_r, theta_bar_r, V_bar_r, dt),
                          2.0 * E1)
                  <= 1e-14);
            CHECK(rel_err(ported_dsb_energy(cur, kappa, L0_r, h_bar_r, theta_bar_r, V_bar_r, 2.0),
                          4.0 * E1)
                  <= 1e-14);
            CHECK(rel_err(ported_dsb_energy(cur, kappa, L0_r, h_bar_r, theta_bar_r, 3.0, dt),
                          3.0 * E1)
                  <= 1e-14);
        }
    }

    // =======================================================================
    //  D) strain-plastic discrete shells: E = V_bar dt^2 L0 kappa delta^2 / h_bar
    //     with delta = wrap(theta - theta_bar)
    // =======================================================================
    {
        const auto flat = unit_square_bending_stencil(0.0);
        double     L0 = 0.0, h_bar = 0.0, theta_bar = 0.0, V_bar = 0.0;
        sym_pdsb::compute_constants<double>(L0,
                                            h_bar,
                                            theta_bar,
                                            V_bar,
                                            lv<3>(flat.X0),
                                            lv<3>(flat.X1),
                                            lv<3>(flat.X2),
                                            lv<3>(flat.X3),
                                            0.001,
                                            0.001,
                                            0.001,
                                            0.001);
        CHECK(std::abs(theta_bar) == 0.0);

        // flat rest -> 0
        CHECK(std::abs(ported_pdsb_energy(flat, kappa, L0, h_bar, theta_bar, V_bar, dt)) == 0.0);
        CHECK(max_abs(ported_pdsb_gradient(flat, kappa, L0, h_bar, theta_bar, V_bar, dt))
              <= 1e-15);

        // folded by pi/2 -> 6 kappa (pi/2)^2
        {
            const double  phi    = kPi / 2.0;
            const Stencil folded = unit_square_bending_stencil(phi);
            const double  E = ported_pdsb_energy(folded, kappa, L0, h_bar, theta_bar, V_bar, dt);
            CHECK(rel_err(E, 6.0 * kappa * phi * phi) <= 1e-13);
        }

        // a *folded rest* state: delta = wrap(theta - theta_bar) is the plastic
        // driving quantity, and the plastic energy only sees delta (never the raw
        // theta), which is the whole point of the history variable
        {
            const double rest_fold = 0.4;
            const double cur_fold  = -1.2;  // delta = -1.6
            const auto   rest      = folded_rest_bending_stencil(rest_fold);
            const auto   cur       = unit_square_bending_stencil(cur_fold);
            double L0_r = 0.0, h_bar_r = 0.0, theta_bar_r = 0.0, V_bar_r = 0.0;
            sym_pdsb::compute_constants<double>(L0_r,
                                                h_bar_r,
                                                theta_bar_r,
                                                V_bar_r,
                                                lv<3>(rest.X0),
                                                lv<3>(rest.X1),
                                                lv<3>(rest.X2),
                                                lv<3>(rest.X3),
                                                0.001,
                                                0.001,
                                                0.001,
                                                0.001);
            const double theta_cur = eig_dihedral(cur.x0, cur.x1, cur.x2, cur.x3);
            const double delta     = eig_angle_delta(theta_cur, theta_bar_r);
            const double E      = ported_pdsb_energy(cur, kappa, L0_r, h_bar_r, theta_bar_r, V_bar_r, dt);
            const double expect = 6.0 * kappa * delta * delta;
            CHECK(rel_err(E, expect) <= 1e-12);
        }
    }

    // =======================================================================
    //  E) stress-plastic discrete shells: the elastic and plastic response branches
    //     elastic_scale = kappa L0 / h_bar, elastic_slope = 2 elastic_scale,
    //     theta_y = yield_stress / elastic_slope
    // =======================================================================
    {
        const auto flat = unit_square_bending_stencil(0.0);
        double     L0 = 0.0, h_bar = 0.0, theta_bar = 0.0, V_bar = 0.0;
        sym_spdsb::compute_constants<double>(L0,
                                             h_bar,
                                             theta_bar,
                                             V_bar,
                                             lv<3>(flat.X0),
                                             lv<3>(flat.X1),
                                             lv<3>(flat.X2),
                                             lv<3>(flat.X3),
                                             0.001,
                                             0.001,
                                             0.001,
                                             0.001);

        const double elastic_scale = kappa * L0 / h_bar;  // = 6 kappa
        CHECK(rel_err(elastic_scale, 6.0 * kappa) <= 1e-15);

        // ---- elastic branch: a large yield stress keeps |delta| <= theta_y --------
        {
            const double yield_stress = 100.0;  // theta_y = 100 / (2 * 6 kappa) = 6.41 rad
            const double phi          = 0.5;    // |delta| = 0.5 < theta_y
            const auto   folded       = unit_square_bending_stencil(phi);

            const double E = ported_spdsb_energy(
                folded, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt);

            const auto   resp   = eig_stress_plastic_response(phi, kappa, L0, h_bar, yield_stress);
            const double expect = resp.energy * V_bar * dt * dt;

            CHECK(rel_err(E, expect) <= 1e-14);
            CHECK(rel_err(E, elastic_scale * phi * phi) <= 1e-14);
            CHECK(rel_err(resp.dEdtheta, 2.0 * elastic_scale * phi) <= 1e-15);
            CHECK(rel_err(resp.ddEddtheta, 2.0 * elastic_scale) <= 1e-15);
            CHECK(max_abs(ported_spdsb_gradient(folded, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt)
                          - fd_gradient_12(
                              [&](const Stencil& s)
                              {
                                  return ported_spdsb_energy(
                                      s, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt);
                              },
                              folded))
                  <= 1e-6);
        }

        // ---- plastic branch: a small yield stress saturates --------------------
        {
            const double yield_stress = 0.24;   // theta_y = 0.24 / (12 kappa) = 0.0154
            const double phi          = 0.8;    // |delta| >> theta_y -> plastic
            const auto   folded       = unit_square_bending_stencil(phi);

            const auto   resp   = eig_stress_plastic_response(phi, kappa, L0, h_bar, yield_stress);
            const double E      = ported_spdsb_energy(
                folded, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt);
            const double expect = resp.energy * V_bar * dt * dt;

            CHECK(rel_err(E, expect) <= 1e-14);
            // saturated slope: dEdtheta == yield_stress (constant), ddEddtheta == 0
            CHECK(rel_err(resp.dEdtheta, yield_stress) <= 1e-15);
            CHECK(resp.ddEddtheta == 0.0);
            CHECK(rel_err(resp.energy,
                          elastic_scale * resp.theta_y * resp.theta_y
                              + yield_stress * resp.delta_gamma)
                  <= 1e-15);

            // the plastic branch is *linear* in delta, so the element energy is linear
            // in (theta - theta_bar) too: doubling the excess doubles the excess part
            const auto   folded2  = unit_square_bending_stencil(1.6);
            const double E2       = ported_spdsb_energy(
                folded2, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt);
            const double d1       = phi - resp.theta_y;
            const double d2       = 1.6 - resp.theta_y;
            const double e_elastic = elastic_scale * resp.theta_y * resp.theta_y;
            CHECK(rel_err(E2 / (V_bar * dt * dt), e_elastic + yield_stress * d2) <= 1e-13);
            CHECK(rel_err(E / (V_bar * dt * dt), e_elastic + yield_stress * d1) <= 1e-13);
            CHECK(max_abs(ported_spdsb_gradient(folded, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt)
                          - fd_gradient_12(
                              [&](const Stencil& s)
                              {
                                  return ported_spdsb_energy(
                                      s, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt);
                              },
                              folded))
                  <= 1e-6);
        }
    }

    // =======================================================================
    //  F) Neo-Hookean shell (2D): E = 0.5 lambda (ln J)^2 + 0.5 mu (Ic - 2 - 2 ln J)
    // =======================================================================
    {
        const double lambda = 3.3;
        const double mu     = 1.7;
        const auto   rest   = unit_triangle();
        const auto   IB     = nh_IB(rest);
        CHECK(max_abs(IB - Matrix2x2::Identity()) == 0.0);
        const double rest_area = triangle_rest_area(rest);
        CHECK(rel_err(rest_area, 0.5) <= 1e-15);

        // ---- the generated `sym::neo_hookean_shell_2d::E` against the hand-derived
        //      closed form, for arbitrary F (the canonical triangle has F == Ds) -----
        double worst = 0.0;
        for(int trial = 0; trial < 50; ++trial)
        {
            Matrix<double, 3, 2> F;
            for(int c = 0; c < 2; ++c)
                for(int r = 0; r < 3; ++r)
                    F(r, c) = rand_uniform(-1.5, 1.5);
            if((F.transpose() * F).determinant() < 1e-3)
                continue;

            Stencil s = rest;
            s.x1      = F.col(0);
            s.x2      = F.col(1);

            lps::LVec<double, 9> X;
            X.segment<3>(0) = lv<3>(s.x0);
            X.segment<3>(3) = lv<3>(s.x1);
            X.segment<3>(6) = lv<3>(s.x2);

            double E = 0.0;
            sym_nh::E<double>(E, lambda, mu, X, lm(IB));
            worst = std::max(worst, rel_err(E, eig_nh_shell_energy(F, lambda, mu)));
        }
        std::printf("[t52b] NH shell E vs the hand-derived closed form: worst rel %.3e\n",
                    worst);
        CHECK(worst <= 1e-12);

        // ---- F = s I: E = 2 lambda (ln s)^2 + mu (s^2 - 1 - 2 ln s) --------------
        {
            const double s = 1.7;
            Stencil      t = unit_triangle();
            t.x1           = Vector3{s, 0.0, 0.0};
            t.x2           = Vector3{0.0, s, 0.0};

            const double expect = 2.0 * lambda * std::log(s) * std::log(s)
                                  + mu * (s * s - 1.0 - 2.0 * std::log(s));
            const double E      = ported_nh_energy(t, lambda, mu, IB, rest_area, 0.001, dt)
                             / (rest_area * 2.0 * 0.001);

            CHECK(rel_err(E, expect) <= 1e-13);

            // the hand-derived F-space gradient, dE/dF = mu F + (lambda ln J - mu) F^+^T
            // (F^+^T = F (F^T F)^-1), restricted to the canonical triangle's two dof
            // blocks: dE/dx1 = column 0, dE/dx2 = column 1
            const Matrix<double, 3, 2> F = eig_F(t, IB);
            const double               J = (F.transpose() * F).determinant();
            const double               lnJ = 0.5 * std::log(J);
            const Matrix<double, 3, 2> FplusT = F * (F.transpose() * F).inverse();
            const Matrix<double, 3, 2> dEdF = mu * F + (lambda * lnJ - mu) * FplusT;

            const Vector9  G = ported_nh_gradient(t, lambda, mu, IB, rest_area, 0.001, dt)
                              / (rest_area * 2.0 * 0.001);
            for(int r = 0; r < 3; ++r)
            {
                CHECK(rel_err(G(3 + r), dEdF(r, 0)) <= 1e-13);
                CHECK(rel_err(G(6 + r), dEdF(r, 1)) <= 1e-13);
            }
        }

        // ---- rest / rigid rotation: zero energy, zero gradient -------------------
        {
            const double E0 = ported_nh_energy(rest, lambda, mu, IB, rest_area, 0.001, dt);
            CHECK(std::abs(E0) <= 1e-30);
            CHECK(max_abs(ported_nh_gradient(rest, lambda, mu, IB, rest_area, 0.001, dt))
                  <= 1e-14);

            const auto Q    = rotation_matrix(Vector3{1.0, -2.0, 0.5}, 1.1);
            const auto rot  = rotate_triangle(rest, Q, Vector3{0.3, -0.2, 0.7});
            const double Er = ported_nh_energy(rot, lambda, mu, IB, rest_area, 0.001, dt);
            // `Q` carries O(eps) rounding, and this 2D Neo-Hookean law is *linear* in
            // (Ic - 2) and in `ln J`, whose round-off only partially cancels: the
            // rotated state's energy is O(eps) = 1e-16, not O(eps^2).
            std::printf("[t52b] NH shell energy at a rigidly rotated rest state: %.3e\n",
                        Er);
            CHECK(std::abs(Er) <= 1e-13);
            CHECK(max_abs(ported_nh_gradient(rot, lambda, mu, IB, rest_area, 0.001, dt))
                  <= 1e-13);

            const auto HR = ported_nh_hessian(rot, lambda, mu, IB, rest_area, 0.001, dt);
            CHECK(max_abs(HR - HR.transpose()) <= 1e-12);
            CHECK(min_eigenvalue(HR) >= -1e-8);
        }

        // ---- an arbitrary rest triangle: E(rest) == 0 and G(rest) == 0 -----------
        {
            double e_worst = 0.0, g_worst = 0.0;
            for(int trial = 0; trial < 20; ++trial)
            {
                auto r = random_triangle();
                auto IB_r = nh_IB(r);
                Stencil at_rest = r;
                at_rest.x0 = r.X0;
                at_rest.x1 = r.X1;
                at_rest.x2 = r.X2;

                const double area = triangle_rest_area(r);
                e_worst = std::max(e_worst,
                                   rel_err(ported_nh_energy(at_rest, lambda, mu, IB_r, area, 0.001, dt),
                                           0.0));
                g_worst = std::max(g_worst,
                                   max_abs(ported_nh_gradient(at_rest, lambda, mu, IB_r, area, 0.001, dt)));
            }
            std::printf("[t52b] NH shell at arbitrary rest states: max |E| %.3e  max |G| %.3e\n",
                        e_worst,
                        g_worst);
            CHECK(e_worst <= 1e-14);
            CHECK(g_worst <= 1e-12);
        }
    }

    // =======================================================================
    //  G) strain-limiting Baraff-Witkin: rest/rotation invariance and the
    //     stretch/shear closed forms
    // =======================================================================
    {
        const double stretchS = 2.5;
        const double shearS   = 0.75;
        const double sr       = 100.0;  // the frontend's legacy strain_rate default

        const auto   rest      = unit_triangle();
        const auto   IB        = bws_IB(rest);
        CHECK(max_abs(IB - Matrix2x2::Identity()) == 0.0);
        const double rest_area = triangle_rest_area(rest);

        // ---- the canonical triangle at rest: E = 0, G = 0, and F^T F = I ---------
        {
            const auto F = eig_F(rest, IB);
            CHECK(max_abs(F.transpose() * F - Matrix2x2::Identity()) <= 1e-15);
            const double E = ported_bws_energy(rest, stretchS, shearS, sr, IB, rest_area, dt);
            CHECK(std::abs(E) == 0.0);
            CHECK(max_abs(ported_bws_gradient(rest, stretchS, shearS, sr, IB, rest_area, dt))
                  <= 1e-14);
            const Matrix9x9 H = ported_bws_hessian(rest, stretchS, shearS, sr, IB, rest_area, dt);
            CHECK(max_abs(H - H.transpose()) <= 1e-12);
            CHECK(min_eigenvalue(H) >= -1e-8);
        }

        // ---- arbitrary rest triangles: E = 0, G = 0 (F(rest)^T F(rest) == I) -----
        {
            double e_worst = 0.0, g_worst = 0.0, metric_worst = 0.0;
            for(int trial = 0; trial < 20; ++trial)
            {
                auto    r      = random_triangle();
                auto    IB_r   = bws_IB(r);
                Stencil at_rest = r;
                at_rest.x0      = r.X0;
                at_rest.x1      = r.X1;
                at_rest.x2      = r.X2;

                const auto   F    = eig_F(at_rest, IB_r);
                const double area = triangle_rest_area(r);
                metric_worst      = std::max(
                    metric_worst, max_abs(F.transpose() * F - Matrix2x2::Identity()));
                e_worst = std::max(
                    e_worst,
                    rel_err(ported_bws_energy(at_rest, stretchS, shearS, sr, IB_r, area, dt), 0.0));
                g_worst = std::max(
                    g_worst,
                    max_abs(ported_bws_gradient(at_rest, stretchS, shearS, sr, IB_r, area, dt)));
            }
            std::printf("[t52b] BW shell at arbitrary rest states: |F^T F - I| %.3e"
                        "  max |E| %.3e  max |G| %.3e\n",
                        metric_worst,
                        e_worst,
                        g_worst);
            CHECK(metric_worst <= 1e-14);
            CHECK(e_worst <= 1e-13);
            CHECK(g_worst <= 1e-11);
        }

        // ---- rigid motion of a deformed triangle leaves E and G at rest values ----
        {
            const auto Q   = rotation_matrix(Vector3{-1.0, 0.5, 2.0}, -0.9);
            const auto rot = rotate_triangle(rest, Q, Vector3{1.0, 2.0, -3.0});
            CHECK(std::abs(ported_bws_energy(rot, stretchS, shearS, sr, IB, rest_area, dt))
                  <= 1e-25);
            CHECK(max_abs(ported_bws_gradient(rot, stretchS, shearS, sr, IB, rest_area, dt))
                  <= 1e-14);
        }

        // ---- uniform in-plane stretch along the first anisotropic direction ------
        {
            const double s = 1.6;  // > 1, so the cubic strain-limiting term is active
            Stencil      t = unit_triangle();
            t.x1           = Vector3{s, 0.0, 0.0};
            t.x2           = Vector3{0.0, 1.0, 0.0};

            const auto   F      = eig_F(t, IB);
            const double expect = stretchS
                                  * (std::pow(s - 1.0, 2.0) + sr * std::pow(s - 1.0, 3.0));
            const double E = ported_bws_energy(t, stretchS, shearS, sr, IB, rest_area, dt)
                             / (rest_area * dt * dt);

            CHECK(rel_err(E, expect) <= 1e-13);
            CHECK(rel_err(E, eig_bws_energy(F, stretchS, shearS, sr)) <= 1e-13);

            // a stretch *below* 1 de-activates the cubic term (the ucoeff select)
            const double s_lo   = 0.7;
            Stencil      t_lo   = unit_triangle();
            t_lo.x1             = Vector3{s_lo, 0.0, 0.0};
            t_lo.x2             = Vector3{0.0, 1.0, 0.0};
            const double E_lo   = ported_bws_energy(t_lo, stretchS, shearS, sr, IB, rest_area, dt)
                                / (rest_area * dt * dt);
            CHECK(rel_err(E_lo, stretchS * std::pow(s_lo - 1.0, 2.0)) <= 1e-13);
        }

        // ---- pure shear: I5u == 1 (inactive), I5v > 1, shear_energy == I6^2 ------
        {
            const double k = 0.45;
            Stencil      t = unit_triangle();
            t.x1           = Vector3{1.0, 0.0, 0.0};
            t.x2           = Vector3{k, 1.0, 0.0};

            const auto   F    = eig_F(t, IB);
            const double I5v  = (F * Vector2{0.0, 1.0}).norm();
            const double I6   = Vector2{1.0, 0.0}.transpose() * F.transpose() * F
                              * Vector2{0.0, 1.0};
            const double expect = stretchS
                                      * (std::pow(I5v - 1.0, 2.0)
                                         + sr * std::pow(I5v - 1.0, 3.0))
                                  + shearS * I6 * I6;
            const double E = ported_bws_energy(t, stretchS, shearS, sr, IB, rest_area, dt)
                             / (rest_area * dt * dt);

            CHECK(rel_err(I6, k) <= 1e-15);
            CHECK(rel_err(E, expect) <= 1e-13);
            CHECK(rel_err(E, eig_bws_energy(F, stretchS, shearS, sr)) <= 1e-13);
        }

        // ---- the 6x9 `dFdX` is exactly the derivative of `F(x)` -----------------
        {
            auto F_of = [&](const Stencil& s) -> Matrix<double, 3, 2>
            { return eig_F(s, IB); };

            // dF/dx from the ported routine
            const auto dFdx_ported = sym_bws::dFdX<double>(lm(IB));

            double worst = 0.0;
            for(int k = 0; k < 9; ++k)
            {
                const auto Fp = F_of(perturb_triangle(rest, k, kFdH));
                const auto Fm = F_of(perturb_triangle(rest, k, -kFdH));
                const auto dF = (Fp - Fm) / (2 * kFdH);
                for(int r = 0; r < 3; ++r)
                    for(int c = 0; c < 2; ++c)
                        worst = std::max(worst,
                                         std::abs(to_eig(dFdx_ported)(c * 3 + r, k) - dF(r, c)));
            }
            CHECK(worst <= 1e-9);
        }
    }
}

namespace
{
/**
 * @brief Compare an analytic Hessian against the FD one, honouring the reference's
 *        eigenvalue-clamp projection (`make_spd`).
 *
 * The constitutions replace an indefinite local Hessian by its eigenvalue-clamped
 * version, so the FD comparison is only valid where the true (FD) Hessian is already
 * PSD; everywhere else the *identity* `H_analytic == clamp(H_true)` is asserted
 * instead (cookbook §7.17 rule 8).  States within the FD's own sign uncertainty of the
 * boundary are counted as indeterminate.
 */
template <int N>
bool check_hessian_clamped(const Matrix<Float, N, N>& H_analytic,
                           const Matrix<Float, N, N>& H_true,
                           double              psd_eps   = 1e-6,
                           double              rel_bound = 1e-5)
{
    const double min_eig = min_eigenvalue(H_true);
    if(min_eig > psd_eps)
    {
        const double scale = std::max(1.0, max_abs(H_true));
        CHECK(rel_err(max_abs(H_analytic - H_true), 0.0) <= rel_bound);
        (void)scale;
        return true;
    }
    if(min_eig < -psd_eps)
    {
        Eigen::SelfAdjointEigenSolver<Matrix<Float, N, N>> es_true(H_true);
        Eigen::SelfAdjointEigenSolver<Matrix<Float, N, N>> es_an(H_analytic);
        const double clamped =
            (es_an.eigenvalues() - es_true.eigenvalues().cwiseMax(0.0)).cwiseAbs().maxCoeff();
        CHECK(clamped <= 1e-4 * std::max(1.0, es_true.eigenvalues().cwiseAbs().maxCoeff()));
        return true;
    }
    return false;  // indeterminate: the state sits on the projection boundary
}

/// The FD states of the Baraff-Witkin energy must sit away from the model's
/// `I5u > 1` / `I5v > 1` branch boundary (the reference's `select`s), otherwise the
/// central difference straddles two branches.
Stencil bws_fd_state(double& i5u, double& i5v, double& i6, bool both_large, bool pure = false)
{
    const double a = both_large ? rand_uniform(1.25, 1.6) : rand_uniform(0.45, 0.85);
    const double b = both_large ? rand_uniform(1.25, 1.6) : rand_uniform(0.45, 0.85);
    // `pure` removes the shear, which is the regime the reference's stabilised
    // Hessian was derived for (`I6 == 0` with `I5u, I5v >= 1`)
    const double k = pure ? 0.0 : rand_uniform(-0.2, 0.2);

    Stencil s = unit_triangle();
    s.x1      = Vector3{a, 0.0, 0.0};
    s.x2      = Vector3{a * k, b, 0.0};

    // a rigid motion of the whole triangle leaves the invariants unchanged
    const auto Q = rotation_matrix(Vector3{rand_uniform(-1, 1),
                                          rand_uniform(-1, 1),
                                          rand_uniform(-1, 1)},
                                   rand_uniform(-3.14, 3.14));
    const auto t = Vector3{rand_uniform(-1, 1), rand_uniform(-1, 1), rand_uniform(-1, 1)};
    auto       rotate = [&](const Vector3& x) -> Vector3 { return Q * x + t; };
    s.x0              = rotate(s.x0);
    s.x1              = rotate(s.x1);
    s.x2              = rotate(s.x2);

    const auto F  = eig_F(s, Matrix2x2::Identity());
    i5u           = (F * Vector2{1.0, 0.0}).norm();
    i5v           = (F * Vector2{0.0, 1.0}).norm();
    i6            = Vector2{1.0, 0.0}.transpose() * F.transpose() * F * Vector2{0.0, 1.0};
    return s;
}
}  // namespace

// ---------------------------------------------------------------------------
//  2) finite differences of the ported per-element routines
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_b_finite_difference", "[luisa][fem][constitution_b]")
{
    test_runtime();

    const double kappa = 0.9;
    const double dt    = 1.0;

    // =======================================================================
    //  A) discrete shells / strain-plastic shells: FD w.r.t. the 12 dofs
    // =======================================================================
    {
        double g_worst = 0.0, sym_worst = 0.0, rest_h_worst = 0.0;
        int    psd_states = 0, clamped_states = 0, indeterminate = 0;

        for(int trial = 0; trial < 40; ++trial)
        {
            const auto s = random_bending_stencil();

            // ---- the *rest* configuration is the PSD state of a bending stencil ----
            {
                Stencil at_rest = s;
                at_rest.x0      = s.X0;
                at_rest.x1      = s.X1;
                at_rest.x2      = s.X2;
                at_rest.x3      = s.X3;

                double L0r = 0.0, h_barr = 0.0, theta_barr = 0.0, V_barr = 0.0;
                sym_dsb::compute_constants<double>(L0r,
                                                   h_barr,
                                                   theta_barr,
                                                   V_barr,
                                                   lv<3>(s.X0),
                                                   lv<3>(s.X1),
                                                   lv<3>(s.X2),
                                                   lv<3>(s.X3),
                                                   0.001,
                                                   0.001,
                                                   0.001,
                                                   0.001);

                auto grad_of = [&](const Stencil& t)
                { return ported_dsb_gradient(t, kappa, L0r, h_barr, theta_barr, V_barr, dt); };
                const Matrix12x12 H_fd =
                    fd_hessian_12(grad_of, at_rest, kFdRestH);
                const Matrix12x12 H_port =
                    ported_dsb_hessian(at_rest, kappa, L0r, h_barr, theta_barr, V_barr, dt);

              // At `theta == theta_bar` the energy is stationary and its analytic Hessian
              // is `dthetadx * ddEddtheta * dthetadx^T` (exactly PSD); the FD one carries
              // the step-size noise of a stationary state, and that noise grows with the
              // stencil (the dihedral's `acos` conditioning does not cancel there).  The
              // comparison is therefore *relative to the Hessian's own scale*; the tight
              // rest-state statement is the eigenvalue-clamp identity of the membrane
              // cases (1e-11), this one pins the scale.
              CHECK(min_eigenvalue(H_port) >= -1e-8 * max_abs(H_port));
              // relative to the Hessian's own scale (its entries are O(1e1) here, but
              // the FD noise is an *absolute* property of the step, see `kFdRestH`)
              rest_h_worst = std::max(
                  rest_h_worst, max_abs(H_port - H_fd) / max_abs(H_port));
            }

            double L0 = 0.0, h_bar = 0.0, theta_bar = 0.0, V_bar = 0.0;
            sym_dsb::compute_constants<double>(L0,
                                               h_bar,
                                               theta_bar,
                                               V_bar,
                                               lv<3>(s.X0),
                                               lv<3>(s.X1),
                                               lv<3>(s.X2),
                                               lv<3>(s.X3),
                                               0.001,
                                               0.001,
                                               0.001,
                                               0.001);

            // ---- discrete shells (elastic) ----------------------------------
            {
                auto energy_of = [&](const Stencil& t)
                { return ported_dsb_energy(t, kappa, L0, h_bar, theta_bar, V_bar, dt); };
                auto grad_of = [&](const Stencil& t)
                { return ported_dsb_gradient(t, kappa, L0, h_bar, theta_bar, V_bar, dt); };

                const Vector12    G      = grad_of(s);
                const Vector12    G_fd   = fd_gradient_12(energy_of, s);
                const Matrix12x12 H_fd   = fd_hessian_12(grad_of, s);
                const Matrix12x12 H_port = ported_dsb_hessian(s, kappa, L0, h_bar, theta_bar, V_bar, dt);

                g_worst   = std::max(g_worst, rel_err(max_abs(G - G_fd), 0.0));
                sym_worst = std::max(sym_worst, max_abs(H_port - H_port.transpose()));
                if(check_hessian_clamped(H_port, H_fd))
                {
                    if(min_eigenvalue(H_fd) > 1e-6)
                        ++psd_states;
                    else
                        ++clamped_states;
                }
                else
                {
                    ++indeterminate;
                }
            }

            // ---- strain-plastic shells --------------------------------------
            {
                auto energy_of = [&](const Stencil& t)
                { return ported_pdsb_energy(t, kappa, L0, h_bar, theta_bar, V_bar, dt); };
                auto grad_of = [&](const Stencil& t)
                { return ported_pdsb_gradient(t, kappa, L0, h_bar, theta_bar, V_bar, dt); };

                const Vector12    G      = grad_of(s);
                const Vector12    G_fd   = fd_gradient_12(energy_of, s);
                const Matrix12x12 H_fd   = fd_hessian_12(grad_of, s);
                const Matrix12x12 H_port =
                    ported_pdsb_hessian(s, kappa, L0, h_bar, theta_bar, V_bar, dt);

                g_worst   = std::max(g_worst, rel_err(max_abs(G - G_fd), 0.0));
                sym_worst = std::max(sym_worst, max_abs(H_port - H_port.transpose()));
                if(check_hessian_clamped(H_port, H_fd))
                {
                    if(min_eigenvalue(H_fd) > 1e-6)
                        ++psd_states;
                    else
                        ++clamped_states;
                }
                else
                {
                    ++indeterminate;
                }
            }

            // ---- stress-plastic shells (large yield stress -> elastic branch) --
            {
                const double yield_stress = 50.0;  // far above the sampled |delta|
                auto         energy_of    = [&](const Stencil& t)
                {
                    return ported_spdsb_energy(
                        t, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt);
                };
                auto grad_of = [&](const Stencil& t)
                {
                    return ported_spdsb_gradient(
                        t, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt);
                };

                const Vector12    G    = grad_of(s);
                const Vector12    G_fd = fd_gradient_12(energy_of, s);
                const Matrix12x12 H_fd = fd_hessian_12(grad_of, s);
                const Matrix12x12 H_port = ported_spdsb_hessian(
                    s, kappa, L0, h_bar, theta_bar, yield_stress, V_bar, dt);

                g_worst   = std::max(g_worst, rel_err(max_abs(G - G_fd), 0.0));
                sym_worst = std::max(sym_worst, max_abs(H_port - H_port.transpose()));
                if(check_hessian_clamped(H_port, H_fd))
                {
                    if(min_eigenvalue(H_fd) > 1e-6)
                        ++psd_states;
                    else
                        ++clamped_states;
                }
                else
                {
                    ++indeterminate;
                }
            }
        }

        std::printf("[t52b] bending (DSB/PDSB/SPDSB) FD: gradient rel %.3e, Hessian"
                    " asymmetry %.3e; PSD states %d, clamped states %d, indeterminate %d,"
                    " rest-state Hessian rel %.3e\n",
                    g_worst,
                    sym_worst,
                    psd_states,
                    clamped_states,
                    indeterminate,
                    rest_h_worst);
          CHECK(g_worst <= 1e-4);
          CHECK(sym_worst <= 1e-12);
          // Every *random folded* state of a bending stencil has an indefinite local
          // Hessian (the energy only constrains the dihedral angle and the theta-Hessian
          // term changes sign), so those states exercise the eigenvalue-clamp identity;
          // the *rest* configuration above is the PSD state where the raw FD comparison
          // applies -- relative to the Hessian's own scale, with the documented
          // stationary-state noise floor of the `kFdRestH` step (measured worst
          // 4.5e-7 in a single-case run, 1.1e-5 in the filtered/full-suite RNG stream).
          CHECK(clamped_states > 0);
          CHECK(rest_h_worst <= 1e-4);
    }

    // =======================================================================
    //  B) Neo-Hookean shell: FD w.r.t. the 9 dofs
    // =======================================================================
    {
        const double lambda = 2.7;
        const double mu     = 1.4;

        double g_worst = 0.0, h_worst = 0.0, sym_worst = 0.0, rest_h_worst = 0.0;
        int    psd_states = 0, clamped_states = 0, indeterminate = 0;

        for(int trial = 0; trial < 40; ++trial)
        {
            const auto s    = random_triangle();
            const auto IB   = nh_IB(s);
            const double area = triangle_rest_area(s);

            // ---- the rest state (the membrane's PSD state) --------------------
            {
                Stencil at_rest = s;
                at_rest.x0      = s.X0;
                at_rest.x1      = s.X1;
                at_rest.x2      = s.X2;

                auto grad_of = [&](const Stencil& t)
                { return ported_nh_gradient(t, lambda, mu, IB, area, 0.001, dt); };
                // NOTE: unlike a bending stencil, the Neo-Hookean gradient does not
                // vanish through a cancelling difference of O(1) angles at rest, so the
                // plain h = 1e-6 step applies here (measured 1.9e-10).
                const Matrix9x9 H_fd = fd_hessian_9(grad_of, at_rest);
                const Matrix9x9 H_port =
                    ported_nh_hessian(at_rest, lambda, mu, IB, area, 0.001, dt);

                // The 2D Neo-Hookean Hessian at rest is *indefinite* in the deviatoric
                // directions (the `- mu ln J` branch is concave there), so even here the
                // reference's `make_spd` projection is active and the identity
                // `H_analytic == clamp(H_true)` is the property to assert (cookbook
                // §7.17 rule 8) — not a raw FD comparison.
                CHECK(min_eigenvalue(H_port) >= -1e-8 * max_abs(H_port));
                Eigen::SelfAdjointEigenSolver<Matrix9x9> es_true(H_fd), es_an(H_port);
                const double clamp_err =
                    (es_an.eigenvalues() - es_true.eigenvalues().cwiseMax(0.0))
                        .cwiseAbs()
                        .maxCoeff()
                    / std::max(1.0, es_true.eigenvalues().cwiseAbs().maxCoeff());
                CHECK(clamp_err <= 1e-4);
                if(es_true.eigenvalues()(0) > 1e-6)
                    ++psd_states;
                rest_h_worst = std::max(rest_h_worst, clamp_err);
            }

            auto energy_of = [&](const Stencil& t)
            { return ported_nh_energy(t, lambda, mu, IB, area, 0.001, dt); };
            auto grad_of = [&](const Stencil& t)
            { return ported_nh_gradient(t, lambda, mu, IB, area, 0.001, dt); };

            const Vector9   G      = grad_of(s);
            const Vector9   G_fd   = fd_gradient_9(energy_of, s);
            const Matrix9x9 H_fd   = fd_hessian_9(grad_of, s);
            const Matrix9x9 H_port = ported_nh_hessian(s, lambda, mu, IB, area, 0.001, dt);

            g_worst   = std::max(g_worst, rel_err(max_abs(G - G_fd), 0.0));
            sym_worst = std::max(sym_worst, max_abs(H_port - H_port.transpose()));
            if(check_hessian_clamped(H_port, H_fd))
            {
                if(min_eigenvalue(H_fd) > 1e-6)
                    ++psd_states;
                else
                    ++clamped_states;
            }
            else
            {
                ++indeterminate;
            }
        }

        std::printf("[t52b] NH shell FD: gradient rel %.3e, Hessian asymmetry %.3e;"
                    " PSD states %d, clamped states %d, indeterminate %d,"
                    " rest-state clamp identity %.3e\n",
                    g_worst,
                    sym_worst,
                    psd_states,
                    clamped_states,
                    indeterminate,
                    rest_h_worst);
        CHECK(g_worst <= 1e-4);
        CHECK(sym_worst <= 1e-12);
        CHECK(rest_h_worst <= 1e-4);
        (void)h_worst;
        (void)psd_states;
    }

    // =======================================================================
    //  C) Baraff-Witkin: FD w.r.t. the 9 dofs (both branch families)
    // =======================================================================
    {
        const double stretchS = 2.5;
        const double shearS   = 0.75;
        const double sr       = 100.0;

        double g_worst = 0.0, sym_worst = 0.0;
        // the ported Hessian vs the *independent Eigen transcription* of the
        // reference's own `ddEddF` (this is the transcription check)
        double transcription_worst = 0.0;
        // the FD deviation of that (reference) Hessian: exact in the model's own
        // regime, a measured property of the reference formula everywhere else
        double h_worst_exact = 0.0, h_worst_general = 0.0;
        double min_margin   = std::numeric_limits<double>::max();

        for(int trial = 0; trial < 60; ++trial)
        {
            const bool exact_family = (trial % 3 == 0);
            const bool both_large   = exact_family || (trial % 2 == 0);

            double i5u = 0.0, i5v = 0.0, i6 = 0.0;
            const auto s =
                bws_fd_state(i5u, i5v, i6, both_large, /*pure=*/exact_family);
            CHECK(std::abs(i5u - 1.0) > 0.1);
            CHECK(std::abs(i5v - 1.0) > 0.1);
            if(exact_family)
                CHECK(std::abs(i6) <= 1e-15);
            min_margin = std::min({min_margin, std::abs(i5u - 1.0), std::abs(i5v - 1.0)});

            const auto   IB   = bws_IB(s);
            const double area = triangle_rest_area(s);

            auto energy_of = [&](const Stencil& t)
            { return ported_bws_energy(t, stretchS, shearS, sr, IB, area, dt); };
            auto grad_of = [&](const Stencil& t)
            { return ported_bws_gradient(t, stretchS, shearS, sr, IB, area, dt); };

            const Vector9   G      = grad_of(s);
            const Vector9   G_fd   = fd_gradient_9(energy_of, s);
            const Matrix9x9 H_fd   = fd_hessian_9(grad_of, s);
            const Matrix9x9 H_port = ported_bws_hessian(s, stretchS, shearS, sr, IB, area, dt);

            g_worst   = std::max(g_worst, rel_err(max_abs(G - G_fd), 0.0));
            sym_worst = std::max(sym_worst, max_abs(H_port - H_port.transpose()));

            // the reference's F-space Hessian, re-transcribed independently in Eigen
            const Matrix<double, 3, 2> F = eig_F(s, IB);
            const Matrix6x6            H6_ref = eig_bws_hessian(F, stretchS, shearS, sr);
            lps::LMat<double, 6, 6>    H6_port;
            sym_bws::ddEddF<double>(H6_port,
                                    lm(F),
                                    lv<2>(Vector2{1.0, 0.0}),
                                    lv<2>(Vector2{0.0, 1.0}),
                                    stretchS,
                                    shearS,
                                    sr);
            transcription_worst = std::max(
                transcription_worst, rel_err(max_abs(to_eig(H6_port) - H6_ref), 0.0));

            // relative to the Hessian's own scale (the `rel_err(x, 0)` metric saturates
            // at 1 as soon as the absolute deviation exceeds 1)
            const double fd_dev =
                max_abs(H_port - H_fd) / std::max(1.0, max_abs(H_fd));
            if(exact_family)
                h_worst_exact = std::max(h_worst_exact, fd_dev);
            else
                h_worst_general = std::max(h_worst_general, fd_dev);
        }

        std::printf("[t52b] BW shell FD: gradient rel %.3e, Hessian asymmetry %.3e;"
                    " ddEddF vs the independent Eigen transcription %.3e;"
                    " Hessian-vs-FD in the model's own regime (I6 == 0, I5 >= 1) %.3e,"
                    " outside it %.3e (min branch margin %.3f)\n",
                    g_worst,
                    sym_worst,
                    transcription_worst,
                    h_worst_exact,
                    h_worst_general,
                    min_margin);
        CHECK(g_worst <= 1e-4);
        CHECK(sym_worst <= 1e-12);
        // the port is a faithful transcription of the reference formula ...
        CHECK(transcription_worst <= 1e-12);
        // ... which *is* the exact Hessian of the reference's energy in the regime the
        // model was derived for (I6 == 0 and both anisotropic invariants >= 1) ...
        CHECK(h_worst_exact <= 1e-3);
        // ... and is a *stabilised* (inexact) Hessian outside it: the true Hessian is
        // indefinite for a shrunk/sheared triangle, so the reference keeps a PSD
        // surrogate.  The bound below only pins the *measured magnitude* of that
        // reference property (0.07-1.5 x the Hessian's own scale over this sweep; the
        // transcription itself is pinned at 1e-12 above); see the batch-b report and
        // cookbook §7.17.
        CHECK(h_worst_general <= 2.0);
    }
}

namespace
{
// ---------------------------------------------------------------------------
//  the device probes: the *same* generic per-element routines instantiated for
//  S = lcrt::Var<double> inside a Kernel1D (cookbook §7.4)
// ---------------------------------------------------------------------------
using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

constexpr int DsbStencilSize     = 4;
constexpr int DsbHalfHessianSize = DsbStencilSize * (DsbStencilSize + 1) / 2;
constexpr int TriStencilSize     = 3;
constexpr int TriHalfHessianSize = TriStencilSize * (TriStencilSize + 1) / 2;

/// Baraff-Witkin element routines on the device.
auto& bws_element_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<gg::LVector3>,    // xs (3 per element)
                                   lcrt::Buffer<gg::LMatrix2x2>,  // IB
                                   lcrt::Buffer<double>,      // rest_area
                                   lcrt::Buffer<double>,      // lambda
                                   lcrt::Buffer<double>,      // mu
                                   lcrt::Buffer<double>,      // strain_rate
                                   lcrt::Buffer<double>,      // out_energies
                                   lcrt::Buffer<double>,      // out_gradients (9)
                                   lcrt::Buffer<double>,      // out_hessians (81)
                                   double,                    // dt
                                   uint32_t>(                 // n
            [](lcrt::BufferVar<gg::LVector3>   xs,
               lcrt::BufferVar<gg::LMatrix2x2> IBs,
               lcrt::BufferVar<double>     areas,
               lcrt::BufferVar<double>     lambdas,
               lcrt::BufferVar<double>     mus,
               lcrt::BufferVar<double>     strain_rates,
               lcrt::BufferVar<double>     out_e,
               lcrt::BufferVar<double>     out_g,
               lcrt::BufferVar<double>     out_h,
               lcrt::Var<double>           dt,
               lcrt::Var<uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto x0 = gg::load_vec3(xs->read(i * 3u + 0u));
                    auto x1 = gg::load_vec3(xs->read(i * 3u + 1u));
                    auto x2 = gg::load_vec3(xs->read(i * 3u + 2u));
                    auto IB = gg::load_mat2x2(IBs->read(i));

                    auto lambda      = lambdas->read(i);
                    auto mu          = mus->read(i);
                    auto strain_rate = strain_rates->read(i);
                    auto rest_area   = areas->read(i);

                    auto E = gg::bw_shell_detail::element_energy<lcrt::Var<double>>(
                        lambda, mu, strain_rate, rest_area, x0, x1, x2, IB, dt);
                    out_e->write(i, E);

                    auto G = gg::bw_shell_detail::element_gradient<lcrt::Var<double>>(
                        lambda, mu, strain_rate, rest_area, x0, x1, x2, IB, dt);
                    for(int c = 0; c < 9; ++c)
                        out_g->write(i * 9u + c, G(c));

                    auto H = gg::bw_shell_detail::element_hessian<lcrt::Var<double>>(
                        lambda, mu, strain_rate, rest_area, x0, x1, x2, IB, dt);
                    for(int r = 0; r < 9; ++r)
                        for(int c = 0; c < 9; ++c)
                            out_h->write(i * 81u + r * 9u + c, H(r, c));
                };
            });
    return k;
}

/// The NH shell's *whole* per-element energy on the device (`element_energy`, i.e. the
/// generated density times `Vdt2`) -- the direct device-vs-host check of the generated
/// `sym::neo_hookean_shell_2d` code, independent of the elastics' plumbing (the scene
/// case drives the shipped kernel instead).  One third of the states are *translated*
/// far from the origin: the generated density expands `det(A)` in the raw coordinates,
/// so a translated stencil exercises that expansion's cancellation.
auto& nh_element_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<gg::LVector3>,  // xs (3 per element)
                                   lcrt::Buffer<gg::LMatrix2x2>, // IBs
                                   lcrt::Buffer<double>,         // lambdas
                                   lcrt::Buffer<double>,         // mus
                                   lcrt::Buffer<double>,         // rest areas
                                   lcrt::Buffer<double>,         // thicknesses
                                   lcrt::Buffer<double>,         // out energies
                                   double,                       // dt
                                   uint32_t>(                    // n
            [](lcrt::BufferVar<gg::LVector3>  xs,
               lcrt::BufferVar<gg::LMatrix2x2> IBs,
               lcrt::BufferVar<double>        lambdas,
               lcrt::BufferVar<double>        mus,
               lcrt::BufferVar<double>        areas,
               lcrt::BufferVar<double>        thicknesses,
               lcrt::BufferVar<double>        out_e,
               lcrt::Var<double>              dt,
               lcrt::Var<uint32_t>            n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    out_e->write(i,
                                 gg::nh_shell_detail::element_energy<lcrt::Var<double>>(
                                     lambdas->read(i),
                                     mus->read(i),
                                     gg::load_vec3(xs->read(3u * i + 0u)),
                                     gg::load_vec3(xs->read(3u * i + 1u)),
                                     gg::load_vec3(xs->read(3u * i + 2u)),
                                     gg::load_mat2x2(IBs->read(i)),
                                     areas->read(i),
                                     thicknesses->read(i),
                                     dt));
                };
            });
    return k;
}

/// The discrete-shells *energy* on the device (the cheapest of the five routines;
/// the gradient/Hessian kernel of the constitution itself is exercised by the scene
/// case, where its `make_spd(12x12)` inlining is paid once anyway).

auto& dsb_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<gg::LVector4i>,  // stencils
                                   lcrt::Buffer<gg::LVector3>,    // xs
                                   lcrt::Buffer<double>,      // L0
                                   lcrt::Buffer<double>,      // h_bar
                                   lcrt::Buffer<double>,      // theta_bar
                                   lcrt::Buffer<double>,      // kappa
                                   lcrt::Buffer<double>,      // V_bar
                                   lcrt::Buffer<double>,      // out_energies
                                   double,                    // dt
                                   uint32_t>(                 // n
            [](lcrt::BufferVar<gg::LVector4i> stencils,
               lcrt::BufferVar<gg::LVector3>  xs,
               lcrt::BufferVar<double>    L0s,
               lcrt::BufferVar<double>    h_bars,
               lcrt::BufferVar<double>    theta_bars,
               lcrt::BufferVar<double>    kappas,
               lcrt::BufferVar<double>    V_bars,
               lcrt::BufferVar<double>    out_e,
               lcrt::Var<double>          dt,
               lcrt::Var<uint32_t>        n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto stencil = gg::load_vec4i(stencils->read(i));
                    auto x0      = gg::load_vec3(xs->read(stencil(0)));
                    auto x1      = gg::load_vec3(xs->read(stencil(1)));
                    auto x2      = gg::load_vec3(xs->read(stencil(2)));
                    auto x3      = gg::load_vec3(xs->read(stencil(3)));

                    auto E = gg::dsb_detail::stencil_energy<lcrt::Var<double>>(x0,
                                                                              x1,
                                                                              x2,
                                                                              x3,
                                                                              L0s->read(i),
                                                                              h_bars->read(i),
                                                                              theta_bars->read(i),
                                                                              kappas->read(i),
                                                                              V_bars->read(i),
                                                                              dt);
                    out_e->write(i, E);
                };
            });
    return k;
}

/**
 * @brief One frame of the strain-plastic *history update* on the device.
 *
 * This kernel body is the reference's
 * `StrainPlasticDiscreteShellBendingTimeIntegrator_do_update_state_kernel` minus the
 * buffer plumbing, and it calls the *same* generic routine
 * (`gg::pdsb_detail::update_state`) the ported TimeIntegrator kernel calls — so the
 * test drives the shipped history update.  It reads `theta_bar` / `yield_threshold`
 * and writes them back unconditionally, which is the port's formulation of the
 * reference's conditional writes (an unmoved history is not a change).
 */
auto& pdsb_update_state_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<gg::LVector4i>,  // stencils
                                   lcrt::Buffer<gg::LVector3>,    // xs
                                   lcrt::Buffer<double>,      // hardening_moduli
                                   lcrt::Buffer<double>,      // theta_bars (rw)
                                   lcrt::Buffer<double>,      // yield_thresholds (rw)
                                   uint32_t>(                 // n
            [](lcrt::BufferVar<gg::LVector4i> stencils,
               lcrt::BufferVar<gg::LVector3>  xs,
               lcrt::BufferVar<double>    hardening_moduli,
               lcrt::BufferVar<double>    theta_bars,
               lcrt::BufferVar<double>    yield_thresholds,
               lcrt::Var<uint32_t>        n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto stencil = gg::load_vec4i(stencils->read(i));
                    auto x0      = gg::load_vec3(xs->read(stencil(0)));
                    auto x1      = gg::load_vec3(xs->read(stencil(1)));
                    auto x2      = gg::load_vec3(xs->read(stencil(2)));
                    auto x3      = gg::load_vec3(xs->read(stencil(3)));

                    auto theta_bar       = theta_bars->read(i);
                    auto yield_threshold = yield_thresholds->read(i);

                    (void)gg::pdsb_detail::update_state<lcrt::Var<double>>(x0,
                                                                          x1,
                                                                          x2,
                                                                          x3,
                                                                          hardening_moduli->read(i),
                                                                          theta_bar,
                                                                          yield_threshold);

                    theta_bars->write(i, theta_bar);
                    yield_thresholds->write(i, yield_threshold);
                };
            });
    return k;
}

/// The stress-plastic history update (same shape; the reference's extra
/// `(kappa, L0, h_bar)` arguments come from the constitution's own buffers).
auto& spdsb_update_state_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<gg::LVector4i>,  // stencils
                                   lcrt::Buffer<gg::LVector3>,    // xs
                                   lcrt::Buffer<double>,      // hardening_moduli
                                   lcrt::Buffer<double>,      // bending_stiffnesses
                                   lcrt::Buffer<double>,      // rest_lengths
                                   lcrt::Buffer<double>,      // h_bars
                                   lcrt::Buffer<double>,      // theta_bars (rw)
                                   lcrt::Buffer<double>,      // yield_stresses (rw)
                                   uint32_t>(                 // n
            [](lcrt::BufferVar<gg::LVector4i> stencils,
               lcrt::BufferVar<gg::LVector3>  xs,
               lcrt::BufferVar<double>    hardening_moduli,
               lcrt::BufferVar<double>    bending_stiffnesses,
               lcrt::BufferVar<double>    rest_lengths,
               lcrt::BufferVar<double>    h_bars,
               lcrt::BufferVar<double>    theta_bars,
               lcrt::BufferVar<double>    yield_stresses,
               lcrt::Var<uint32_t>        n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto stencil = gg::load_vec4i(stencils->read(i));
                    auto x0      = gg::load_vec3(xs->read(stencil(0)));
                    auto x1      = gg::load_vec3(xs->read(stencil(1)));
                    auto x2      = gg::load_vec3(xs->read(stencil(2)));
                    auto x3      = gg::load_vec3(xs->read(stencil(3)));

                    auto theta_bar    = theta_bars->read(i);
                    auto yield_stress = yield_stresses->read(i);

                    (void)gg::spdsb_detail::update_state<lcrt::Var<double>>(
                        x0,
                        x1,
                        x2,
                        x3,
                        hardening_moduli->read(i),
                        bending_stiffnesses->read(i),
                        rest_lengths->read(i),
                        h_bars->read(i),
                        theta_bar,
                        yield_stress);

                    theta_bars->write(i, theta_bar);
                    yield_stresses->write(i, yield_stress);
                };
            });
    return k;
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
}  // namespace

// ---------------------------------------------------------------------------
//  3) device vs host: the *same* generic routines, S = double vs
//     S = lcrt::Var<double>
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_b_device_vs_host", "[luisa][fem][constitution_b]")
{
    test_runtime();

    const int count = 24;

    // ---- Baraff-Witkin: energy, gradient and Hessian -------------------------
    {
        std::vector<Vector3>   xs;
        std::vector<Matrix2x2> IBs;
        std::vector<double>    areas, lambdas, mus, rates;
        std::vector<Stencil>   states;

        for(int e = 0; e < count; ++e)
        {
            const auto s  = random_triangle();
            const auto IB = bws_IB(s);
            states.push_back(s);
            xs.push_back(s.x0);
            xs.push_back(s.x1);
            xs.push_back(s.x2);
            IBs.push_back(IB);
            areas.push_back(triangle_rest_area(s));
            lambdas.push_back(rand_uniform(0.5, 3.0));
            mus.push_back(rand_uniform(0.5, 3.0));
            rates.push_back(rand_uniform(0.0, 200.0));
        }

        auto xs_dev    = upload<gg::LVector3>(xs, [](const Vector3& v) { return gg::to_luisa(v); });
        auto ib_dev    = upload<gg::LMatrix2x2>(IBs, [](const Matrix2x2& m) { return gg::to_luisa(m); });
        auto area_dev  = upload(areas);
        auto lam_dev   = upload(lambdas);
        auto mu_dev    = upload(mus);
        auto rate_dev  = upload(rates);

        lud::LDeviceBuffer<double> out_e(count);
        lud::LDeviceBuffer<double> out_g(count * 9);
        lud::LDeviceBuffer<double> out_h(count * 81);

        const auto n32 = static_cast<uint32_t>(count);
        auto&      k   = bws_element_kernel();
        gg::frame_stream().record(k(xs_dev.view(),
                                    ib_dev.view(),
                                    area_dev.view(),
                                    lam_dev.view(),
                                    mu_dev.view(),
                                    rate_dev.view(),
                                    out_e.view(),
                                    out_g.view(),
                                    out_h.view(),
                                    1.0,
                                    n32)
                                    .dispatch(n32));
        gg::frame_stream().synchronize();

        const auto dev_e = read_device(out_e.cview());
        const auto dev_g = read_device(out_g.cview());
        const auto dev_h = read_device(out_h.cview());

        std::vector<double> host_e(count), host_g(count * 9), host_h(count * 81);
        for(int e = 0; e < count; ++e)
        {
            const auto& s = states[e];
            host_e[e] = ported_bws_energy(s, lambdas[e], mus[e], rates[e], IBs[e], areas[e], 1.0);
            const Vector9 G = ported_bws_gradient(s, lambdas[e], mus[e], rates[e], IBs[e], areas[e], 1.0);
            const Matrix9x9 H = ported_bws_hessian(s, lambdas[e], mus[e], rates[e], IBs[e], areas[e], 1.0);
            for(int i = 0; i < 9; ++i)
                host_g[e * 9 + i] = G(i);
            for(int r = 0; r < 9; ++r)
                for(int c = 0; c < 9; ++c)
                    host_h[(e * 9 + r) * 9 + c] = H(r, c);
        }

        double de = 0.0, dg = 0.0, dh = 0.0;
        for(size_t i = 0; i < dev_e.size(); ++i)
            de = std::max(de, std::abs(dev_e[i] - host_e[i]));
        for(size_t i = 0; i < dev_g.size(); ++i)
            dg = std::max(dg, std::abs(dev_g[i] - host_g[i]));
        for(size_t i = 0; i < dev_h.size(); ++i)
            dh = std::max(dh, std::abs(dev_h[i] - host_h[i]));

        std::printf("[t52b] BW shell device vs host: E %.3e  G %.3e  H %.3e\n", de, dg, dh);
        CHECK(de <= 1e-12);
        CHECK(dg <= 1e-12);
        CHECK(dh <= 1e-12);
    }

    // ---- discrete shells: the energy --------------------------------------
    {
        std::vector<Vector4i> stencils;
        std::vector<Vector3>  xs;
        std::vector<double>   L0s, h_bars, theta_bars, kappas, V_bars;
        std::vector<Stencil>  states;

        for(int e = 0; e < count; ++e)
        {
            const auto s = random_bending_stencil();
            states.push_back(s);
            stencils.push_back(Vector4i{4 * e, 4 * e + 1, 4 * e + 2, 4 * e + 3});
            xs.push_back(s.x0);
            xs.push_back(s.x1);
            xs.push_back(s.x2);
            xs.push_back(s.x3);

            double L0 = 0.0, h_bar = 0.0, theta_bar = 0.0, V_bar = 0.0;
            sym_dsb::compute_constants<double>(L0,
                                               h_bar,
                                               theta_bar,
                                               V_bar,
                                               lv<3>(s.X0),
                                               lv<3>(s.X1),
                                               lv<3>(s.X2),
                                               lv<3>(s.X3),
                                               0.001,
                                               0.001,
                                               0.001,
                                               0.001);
            L0s.push_back(L0);
            h_bars.push_back(h_bar);
            theta_bars.push_back(theta_bar);
            kappas.push_back(rand_uniform(0.5, 2.0));
            V_bars.push_back(V_bar);
        }

        auto stencil_dev = upload<gg::LVector4i>(stencils, [](const Vector4i& v) { return gg::to_luisa(v); });
        auto xs_dev      = upload<gg::LVector3>(xs, [](const Vector3& v) { return gg::to_luisa(v); });
        auto L0_dev      = upload(L0s);
        auto hbar_dev    = upload(h_bars);
        auto thetabar_dev = upload(theta_bars);
        auto kappa_dev   = upload(kappas);
        auto Vbar_dev    = upload(V_bars);

        lud::LDeviceBuffer<double> out_e(count);

        const auto n32 = static_cast<uint32_t>(count);
        auto&      k   = dsb_energy_kernel();
        gg::frame_stream().record(k(stencil_dev.view(),
                                    xs_dev.view(),
                                    L0_dev.view(),
                                    hbar_dev.view(),
                                    thetabar_dev.view(),
                                    kappa_dev.view(),
                                    Vbar_dev.view(),
                                    out_e.view(),
                                    1.0,
                                    n32)
                                    .dispatch(n32));
        gg::frame_stream().synchronize();

        const auto dev_e = read_device(out_e.cview());

        double de = 0.0;
        for(int e = 0; e < count; ++e)
        {
            const double host = ported_dsb_energy(
                states[e], kappas[e], L0s[e], h_bars[e], theta_bars[e], V_bars[e], 1.0);
            de = std::max(de, std::abs(dev_e[e] - host));
        }
        std::printf("[t52b] DSB device vs host energy: %.3e\n", de);
        // LC compiles its shaders with `-use_fast_math`, so the DSL's `acos` (the
        // dihedral angle) is approximate: the device energy differs from the host
    // O(1e-7) here (the angle error ~1e-8 enters the energy linearly through the
    // `(theta - theta_bar)` scale).  The *host* instantiation agrees with the
    // independent oracle bitwise (the history case).  Measured 1.01e-6 in the
    // filtered run, 7.66e-7 in this case alone: `1e-5` is the fast-math bound.
    CHECK(de <= 1e-5);
    }
    // ---- Neo-Hookean shell: `element_energy` (generated density * `Vdt2`) -------
    {
        std::vector<Vector3>   xs;
        std::vector<Matrix2x2> IBs;
        std::vector<double>    areas, lambdas, mus, thick;
        for(int e = 0; e < count; ++e)
        {
            auto s = random_triangle();
            if(e % 3 == 2)
            {
                // the same triangle, moved far from the origin (the rest state too)
                const Vector3 t{8.0 * static_cast<double>(e) + 4.0, -5.0, 3.0};
                s.x0 += t;
                s.x1 += t;
                s.x2 += t;
                s.X0 += t;
                s.X1 += t;
                s.X2 += t;
            }
            xs.push_back(s.x0);
            xs.push_back(s.x1);
            xs.push_back(s.x2);
            IBs.push_back(nh_IB(s));
            areas.push_back(triangle_rest_area(s));
            lambdas.push_back(rand_uniform(0.5, 3.0));
            mus.push_back(rand_uniform(0.5, 3.0));
            thick.push_back(rand_uniform(1e-4, 1e-2));
        }

        auto xs_dev    = upload<gg::LVector3>(xs, [](const Vector3& v) { return gg::to_luisa(v); });
        auto ib_dev    = upload<gg::LMatrix2x2>(IBs, [](const Matrix2x2& m) { return gg::to_luisa(m); });
        auto area_dev  = upload(areas);
        auto lam_dev   = upload(lambdas);
        auto mu_dev    = upload(mus);
        auto thick_dev = upload(thick);

        lud::LDeviceBuffer<double> out_e(count);
        const auto                 n32 = static_cast<uint32_t>(count);
        auto&                      k   = nh_element_energy_kernel();
        gg::frame_stream()
            .record(k(xs_dev.view(),
                      ib_dev.view(),
                      lam_dev.view(),
                      mu_dev.view(),
                      area_dev.view(),
                      thick_dev.view(),
                      out_e.view(),
                      1.0,
                      n32)
                        .dispatch(n32));
        gg::frame_stream().synchronize();
        const auto dev_e = read_device(out_e.cview());

        double worst = 0.0, scale = 1.0;
        for(int e = 0; e < count; ++e)
        {
            const double host_e = gg::nh_shell_detail::element_energy<double>(
                lambdas[e],
                mus[e],
                lv<3>(xs[3 * e + 0]),
                lv<3>(xs[3 * e + 1]),
                lv<3>(xs[3 * e + 2]),
                lm(IBs[e]),
                areas[e],
                thick[e],
                1.0);
            worst = std::max(worst, std::abs(dev_e[e] - host_e));
            scale = std::max(scale, std::abs(host_e));
        }
        std::printf("[t52b] NeoHookeanShell2D element energy: device vs host %.3e (rel %.3e)\n",
                    worst,
                    worst / scale);
        // the same bound as the discrete-shells energy in this case: LC's fast-math
        // `log` (the generated density's `log(det IB * det A)`) is the floor here
        CHECK(worst / scale <= 1e-5);
    }
}

namespace
{
// ---------------------------------------------------------------------------
//  a "strip patch": `strips` disjoint unit quads in the y = 0 plane, far apart.
//
//  Each quad contributes exactly one bending stencil (its diagonal has two opposite
//  vertices); the two triangles of a quad are wound consistently, so the rest dihedral
//  angle of its stencil is 0 (checked against the Eigen oracle below).  Folding the
//  `X3` half about the diagonal by `phi` makes `theta == phi`, which is what drives the
//  plastic history.
// ---------------------------------------------------------------------------
struct StripPatch
{
    std::vector<Vector3>             rest_positions;
    std::vector<Vector3i>            triangles;
    std::vector<std::array<SizeT, 4>> strip_vertices;  // (v_opp0, v_mid0, v_mid1, v_opp1)
    SizeT                            strip_count = 0;
};

StripPatch make_strip_patch(SizeT strips, double spacing = 8.0)
{
    StripPatch p;
    for(SizeT i = 0; i < strips; ++i)
    {
        const double x0   = spacing * static_cast<double>(i);
        const SizeT  base = p.rest_positions.size();
        p.rest_positions.push_back(Vector3{x0, 0.0, 0.0});          // v0 (opposite)
        p.rest_positions.push_back(Vector3{x0 + 1.0, 0.0, 0.0});    // v1 (mid-edge)
        p.rest_positions.push_back(Vector3{x0 + 1.0, 0.0, 1.0});    // v2 (opposite)
        p.rest_positions.push_back(Vector3{x0, 0.0, 1.0});          // v3 (mid-edge)
        p.triangles.push_back(Vector3i{static_cast<int>(base),
                                       static_cast<int>(base + 1),
                                       static_cast<int>(base + 3)});
        p.triangles.push_back(Vector3i{static_cast<int>(base + 1),
                                       static_cast<int>(base + 2),
                                       static_cast<int>(base + 3)});
        // the reference's stencil order is (opposite0, mid-edge0, mid-edge1, opposite1)
        p.strip_vertices.push_back({base, base + 1, base + 3, base + 2});
    }
    p.strip_count = strips;
    return p;
}

/// Fold the `X3` vertex of a stencil about the axis through `X1` in the direction
/// `X2 - X1` by `phi` (the same construction the canonical stencil builder uses).
Vector3 fold_x3(const Vector3& X1, const Vector3& X2, const Vector3& X3, double phi)
{
    const Vector3 axis = (X2 - X1).normalized();
    return X1 + Eigen::AngleAxisd(phi, axis).toRotationMatrix() * (X3 - X1);
}

/// Build the folded deformed configuration of a strip from its rest stencil, using the
/// same `(opp0, mid0, mid1, opp1)` stencil order.
std::array<Vector3, 4> folded_strip(const std::array<Vector3, 4>& rest, double phi)
{
    std::array<Vector3, 4> out = rest;
    out[3]                     = fold_x3(rest[1], rest[2], rest[3], phi);
    return out;
}

/// The scene of the strip patch with a membrane (for the thickness attribute) plus the
/// given extra bending constitution.  `mesh_out` (optional) receives the geometry slot's
/// complex so the caller can read the per-edge attributes back.
Scene make_plastic_strip_scene(const StripPatch&           patch,
                               std::string_view            name,
                               bool                        stress_plastic,
                               const std::vector<double>&  yield_values,
                               const std::vector<double>&  hardening_values,
                               S<geometry::GeometrySlot>*  slot_out = nullptr)
{
    auto config = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = false;

    Scene scene{config};
    auto  object = scene.objects().create(name);

    auto mesh = trimesh(span<const Vector3>{patch.rest_positions},
                        span<const Vector3i>{patch.triangles});
    label_surface(mesh);

    // a membrane first: it writes the vertex `thickness` attribute the FEM method needs
    NeoHookeanShell nhs;
    nhs.apply_to(mesh, ElasticModuli2D::youngs_poisson(1.0e6, 0.3), 500.0, 0.001);

    const Float kappa = 5.0_kPa;
    if(stress_plastic)
    {
        StressPlasticDiscreteShellBending spdsb;
        spdsb.apply_to(mesh, kappa, 0.1_kPa, 0.0);
    }
    else
    {
        StrainPlasticDiscreteShellBending pdsb;
        pdsb.apply_to(mesh, kappa, 0.1, 0.0);
    }

    // per-strip attributes: find each strip's diagonal edge and write its values
    {
        auto edges = mesh.edges().topo().view();
        auto bs    = mesh.edges().find<Float>("bending_stiffness");
        auto hm    = mesh.edges().find<Float>("bending_hardening_modulus");
        auto yt    = mesh.edges().find<Float>("bending_yield_threshold");
        auto ys    = mesh.edges().find<Float>("bending_yield_stress");
        REQUIRE(bs != nullptr);
        REQUIRE(hm != nullptr);

        for(SizeT s = 0; s < patch.strip_count; ++s)
        {
            const auto v = patch.strip_vertices[s];
            const int  a = static_cast<int>(std::min(v[1], v[2]));
            const int  b = static_cast<int>(std::max(v[1], v[2]));
            bool       found = false;
            for(SizeT e = 0; e < edges.size(); ++e)
            {
                const int e0 = std::min(edges[e](0), edges[e](1));
                const int e1 = std::max(edges[e](0), edges[e](1));
                if(e0 == a && e1 == b)
                {
                    view(*bs)[e] = kappa;
                    view(*hm)[e] = hardening_values[s];
                    if(yt)
                        view(*yt)[e] = yield_values[s];
                    if(ys)
                        view(*ys)[e] = yield_values[s];
                    found        = true;
                    break;
                }
            }
            CHECK(found);
        }
    }

    // `trimesh` returns the complex by value; the scene takes the slot, which is what
    // the caller reads the (final) attributes back from.
    const auto [slot, rest_slot] = object->geometries().create(mesh);
    (void)rest_slot;
    if(slot_out)
        *slot_out = slot;
    return scene;
}
}  // namespace

// ---------------------------------------------------------------------------
//  4) the plastic history update (the reference's sequence, step by step)
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_b_plastic_history", "[luisa][fem][constitution_b]")
{
    test_runtime();

    // =======================================================================
    //  A) the *scalar* sequence with hand-computed values (the reference's
    //     `update_plastic_state`, strain-plastic)
    //
    //     theta_bar0 = 0.00, yield_threshold0 = 0.05
    //     the reference's step is `theta_bar += sign(delta) * (|delta| - yield)`:
    //     step 1  theta =  0.30 -> delta  0.30, excess 0.25 -> theta_bar  0.25
    //     step 2  theta =  0.28 -> |delta| 0.03 < 0.05      -> NO CHANGE
    //     step 3  theta = -0.50 -> delta -0.75, excess 0.70 -> theta_bar -0.45
    //     step 4  theta =  0.00, hardening 0.10 -> delta 0.45, excess 0.40
    //                             -> theta_bar -0.05, yield 0.05 + 0.1*0.40 = 0.09
    //     (all |delta| < pi, so no wrap in steps 1-4)
    // =======================================================================
    {
        PlasticState state{0.0, 0.05};

        // ---- the oracle (independently typed `while`-loop formulation) ----------
        CHECK(eig_strain_plastic_update(0.30, state, 0.0, true) == true);
        CHECK(rel_err(state.theta_bar, 0.25) <= 1e-14);
        CHECK(state.yield_threshold == 0.05);

        const auto before = state;
        CHECK(eig_strain_plastic_update(0.28, state, 0.0, true) == false);
        CHECK(state.theta_bar == before.theta_bar);           // must not change
        CHECK(state.yield_threshold == before.yield_threshold);

        CHECK(eig_strain_plastic_update(-0.50, state, 0.0, true) == true);
        CHECK(rel_err(state.theta_bar, -0.45) <= 1e-14);
        CHECK(state.yield_threshold == 0.05);

        CHECK(eig_strain_plastic_update(0.0, state, 0.1, true) == true);
        CHECK(rel_err(state.theta_bar, -0.05) <= 1e-14);
        CHECK(rel_err(state.yield_threshold, 0.09) <= 1e-14);

        // ---- the ported routine, same sequence, same values --------------------
        {
            double theta_bar       = 0.0;
            double yield_threshold = 0.05;
            CHECK(sym_pdsb::update_plastic_state<double>(0.30, theta_bar, yield_threshold, 0.0));
            CHECK(rel_err(theta_bar, 0.25) <= 1e-14);
            CHECK(yield_threshold == 0.05);

            const double tb = theta_bar, yt = yield_threshold;
            CHECK(!sym_pdsb::update_plastic_state<double>(0.28, theta_bar, yield_threshold, 0.0));
            CHECK(theta_bar == tb);  // the history is untouched when the yield is not met
            CHECK(yield_threshold == yt);

            CHECK(sym_pdsb::update_plastic_state<double>(-0.50, theta_bar, yield_threshold, 0.0));
            CHECK(rel_err(theta_bar, -0.45) <= 1e-14);
            CHECK(yield_threshold == 0.05);

            CHECK(sym_pdsb::update_plastic_state<double>(0.0, theta_bar, yield_threshold, 0.1));
            CHECK(rel_err(theta_bar, -0.05) <= 1e-14);
            CHECK(rel_err(yield_threshold, 0.09) <= 1e-14);
        }

        // ---- the `plastic_write_threshold` (1e-6) branch ------------------------
        {
            double theta_bar       = 0.0;
            double yield_threshold = 0.10;
            // excess = 5e-7 <= 1e-6 -> no write
            CHECK(!sym_pdsb::update_plastic_state<double>(
                0.10 + 5e-7, theta_bar, yield_threshold, 0.0));
            CHECK(theta_bar == 0.0);
            CHECK(yield_threshold == 0.10);
            // excess = 5e-6 > 1e-6 -> write (`theta_bar += excess`)
            CHECK(sym_pdsb::update_plastic_state<double>(
                0.10 + 5e-6, theta_bar, yield_threshold, 0.0));
            CHECK(rel_err(theta_bar, 5e-6) <= 1e-12);
        }

        // ---- the wrap path (|delta| > pi) --------------------------------------
        {
            double       theta_bar       = 3.0;
            double       yield_threshold = 0.05;
            const double delta           = eig_angle_delta(-3.0, 3.0);  // = -6 + 2 pi
            const double excess          = std::abs(delta) - 0.05;
            // `delta` wraps to +0.2831853071795862, so the direction is +1 and
            // theta_bar = wrap(3.0 + 0.2331853071795862) = -3.05
            CHECK(delta > 0.0);
            CHECK(sym_pdsb::update_plastic_state<double>(-3.0, theta_bar, yield_threshold, 0.0));
            CHECK(std::abs(theta_bar - (-3.05)) <= 1e-12);
            CHECK(rel_err(theta_bar, eig_wrap_angle(3.0 + excess)) <= 1e-15);
        }
    }

    // =======================================================================
    //  B) the *stress-plastic* sequence with hand-computed values
    //
    //     kappa = 1, L0 = 1, h_bar = 1 -> elastic_slope = 2,
    //     yield_stress = 0.4 -> theta_y = 0.2
    //     |delta| <= 0.2 -> elastic response, |delta| > 0.2 -> plastic response
    // =======================================================================
    {
        const double kappa = 1.0, L0 = 1.0, h_bar = 1.0, yield_stress = 0.4;
        CHECK(rel_err(eig_stress_plastic_elastic_slope(kappa, L0, h_bar), 2.0) <= 1e-15);

        // the response branches (kappa L0 / h_bar = 1 -> elastic_scale = 1)
        {
            const auto elastic = eig_stress_plastic_response(0.15, kappa, L0, h_bar, yield_stress);
            CHECK(rel_err(elastic.theta_y, 0.2) <= 1e-15);
            CHECK(rel_err(elastic.energy, 0.15 * 0.15) <= 1e-15);
            CHECK(rel_err(elastic.dEdtheta, 2.0 * 0.15) <= 1e-15);
            CHECK(rel_err(elastic.ddEddtheta, 2.0) <= 1e-15);

            const auto plastic = eig_stress_plastic_response(0.5, kappa, L0, h_bar, yield_stress);
            CHECK(rel_err(plastic.delta_gamma, 0.3) <= 1e-15);
            CHECK(rel_err(plastic.energy, 0.04 + 0.4 * 0.3) <= 1e-15);
            CHECK(rel_err(plastic.dEdtheta, 0.4) <= 1e-15);
            CHECK(plastic.ddEddtheta == 0.0);
        }

        // the ported response must reproduce both branches
        {
            double energy = 0.0, dEdtheta = 0.0, ddEddtheta = 0.0;
            CHECK(sym_spdsb::augmented_response_from_angle_delta<double>(
                0.15, kappa, L0, h_bar, yield_stress, energy, dEdtheta, ddEddtheta));
            CHECK(rel_err(energy, 0.0225) <= 1e-15);
            CHECK(rel_err(dEdtheta, 0.3) <= 1e-15);
            CHECK(rel_err(ddEddtheta, 2.0) <= 1e-15);

            CHECK(sym_spdsb::augmented_response_from_angle_delta<double>(
                -0.5, kappa, L0, h_bar, yield_stress, energy, dEdtheta, ddEddtheta));
            CHECK(rel_err(energy, 0.04 + 0.4 * 0.3) <= 1e-15);
            CHECK(rel_err(dEdtheta, -0.4) <= 1e-15);
            CHECK(ddEddtheta == 0.0);
        }

        // the update sequence: theta_bar0 = 0, yield_stress0 = 0.4, hardening = 0
        // (theta_y = 0.2, so `|delta| <= 0.2` is the no-change condition)
        {
            StressPlasticState oracle{0.0, 0.4};
            double             theta_bar      = 0.0;
            double             yield_stress_v = 0.4;

            // step 1: theta = 0.5 -> delta 0.5, delta_gamma = 0.3 -> theta_bar 0.3
            {
                const auto step = eig_stress_plastic_step(
                    0.5, oracle.theta_bar, kappa, L0, h_bar, oracle.yield_stress);
                CHECK(rel_err(step.tau_trial, 1.0) <= 1e-15);
                CHECK(rel_err(step.theta_y, 0.2) <= 1e-15);
                CHECK(rel_err(step.delta_gamma, 0.3) <= 1e-15);
                CHECK(eig_stress_plastic_update(0.5, oracle, 0.0, kappa, L0, h_bar));
                CHECK(rel_err(oracle.theta_bar, 0.3) <= 1e-14);
                CHECK(rel_err(oracle.yield_stress, 0.4) <= 1e-15);

                CHECK(sym_spdsb::update_plastic_state<double>(
                    0.5, theta_bar, yield_stress_v, 0.0, kappa, L0, h_bar));
                CHECK(rel_err(theta_bar, 0.3) <= 1e-14);
                CHECK(rel_err(yield_stress_v, 0.4) <= 1e-15);
            }

            // step 2: theta = 0.45 -> |delta| 0.15 <= theta_y 0.2 -> NO CHANGE
            {
                const auto before = oracle;
                CHECK(!eig_stress_plastic_update(0.45, oracle, 0.0, kappa, L0, h_bar));
                CHECK(oracle.theta_bar == before.theta_bar);
                CHECK(oracle.yield_stress == before.yield_stress);

                const double tb = theta_bar, ys = yield_stress_v;
                CHECK(!sym_spdsb::update_plastic_state<double>(
                    0.45, theta_bar, yield_stress_v, 0.0, kappa, L0, h_bar));
                CHECK(theta_bar == tb);
                CHECK(yield_stress_v == ys);
            }

            // step 3: theta = -0.3 -> delta -0.6, delta_gamma 0.4, hardening 0.1
            //         -> theta_bar -0.1, yield_stress 0.4 + 0.1*0.4 = 0.44
            {
                CHECK(eig_stress_plastic_update(-0.3, oracle, 0.1, kappa, L0, h_bar));
                CHECK(rel_err(oracle.theta_bar, -0.1) <= 1e-14);
                CHECK(rel_err(oracle.yield_stress, 0.44) <= 1e-14);

                CHECK(sym_spdsb::update_plastic_state<double>(
                    -0.3, theta_bar, yield_stress_v, 0.1, kappa, L0, h_bar));
                CHECK(rel_err(theta_bar, -0.1) <= 1e-14);
                CHECK(rel_err(yield_stress_v, 0.44) <= 1e-14);
            }
        }
    }

    // =======================================================================
    //  C) the *device* path: the constitution's own history buffers, driven through
    //     the same generic routine its TimeIntegrator kernel calls
    // =======================================================================
    const std::vector<std::vector<double>> yields = {
        {0.05, 0.60, 0.10, 0.20},   // per strip
    };
    const std::vector<std::vector<double>> hardening = {
        {0.10, 0.05, 0.00, 0.00},
    };
    // the fold angle of each step (uniform over the strips; the per-strip yields make
    // the histories diverge).  Step 1 is `0.0999995 = 0.30 - 0.20 - 5e-7`, i.e. strip
    // 3 (yield 0.20, theta_bar 0.30) has an excess of exactly 5e-7 <= 1e-6 and must
    // *not* be written.
    const std::vector<double> folds = {0.30, 0.0999995, -0.40, 0.10};

    const auto patch = make_strip_patch(4);

    // The per-strip yield targets are expressed as *angles* (the strain-plastic
    // `yield_threshold` is an angle directly).  The stress-plastic state variable is a
    // stress, so the same targets are converted with the reference's own law
    // `theta_y = yield_stress / (2 kappa L0 / h_bar)`, where `L0`/`h_bar` come from the
    // independently typed `eig_dsb_constants` metric of each strip's rest stencil.
    const double kappa_attr = 5.0_kPa;
    std::vector<double> spdsb_yields;
    for(SizeT s = 0; s < patch.strip_count; ++s)
    {
        const auto& v = patch.strip_vertices[s];  // (opp0, mid0, mid1, opp1)
        double      L0 = 0.0, h_bar = 0.0, theta_bar = 0.0, outer = 0.0;
        eig_dsb_constants(patch.rest_positions[v[0]],
                          patch.rest_positions[v[1]],
                          patch.rest_positions[v[2]],
                          patch.rest_positions[v[3]],
                          L0,
                          h_bar,
                          theta_bar,
                          outer);
        CHECK(std::abs(theta_bar) <= 1e-15);  // a flat rest patch
        const double elastic_slope = 2.0 * kappa_attr * L0 / h_bar;
      spdsb_yields.push_back(yields[0][s] * elastic_slope);
  }


    auto run_plastic_history = [&](bool stress_plastic)
    {
        auto scene = make_plastic_strip_scene(patch,
                                              stress_plastic ? "t52b_spdsb_hist"
                                                             : "t52b_pdsb_hist",
                                              stress_plastic,
                                              stress_plastic ? spdsb_yields : yields[0],
                                              hardening[0]);
        const auto workspace =
            workspace_path(stress_plastic ? "uipc_luisa_t52b_spdsb_hist" : "uipc_luisa_t52b_pdsb_hist");
        Engine engine{"luisa", workspace};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());

        // the history buffers of the real constitution
        lud::LDeviceBuffer<Float>* theta_bars = nullptr;
        lud::LDeviceBuffer<Float>* yields_dev = nullptr;
        lud::LDeviceBuffer<Float>* hardening_dev = nullptr;
        lud::LDeviceBuffer<Float>* kappas = nullptr;
        lud::LDeviceBuffer<Float>* L0s = nullptr;
        lud::LDeviceBuffer<Float>* h_bars = nullptr;
        lud::LDeviceBuffer<gg::LVector4i>* stencils = nullptr;

        if(stress_plastic)
        {
            REQUIRE(FEMConstitutionBProbe::spdsb != nullptr);
            auto& c        = *FEMConstitutionBProbe::spdsb;
            CHECK(c.uid() == 32);
            theta_bars     = &c.theta_bars;
            yields_dev     = &c.yield_stresses;
            hardening_dev  = &c.hardening_moduli;
            kappas         = &c.bending_stiffnesses;
            L0s            = &c.rest_lengths;
            h_bars         = &c.h_bars;
            stencils       = &c.stencils;
        }
        else
        {
            REQUIRE(FEMConstitutionBProbe::pdsb != nullptr);
            auto& c        = *FEMConstitutionBProbe::pdsb;
            CHECK(c.uid() == 31);
            theta_bars     = &c.theta_bars;
            yields_dev     = &c.yield_thresholds;
            hardening_dev  = &c.hardening_moduli;
            kappas         = &c.bending_stiffnesses;
            L0s            = &c.rest_lengths;
            h_bars         = &c.h_bars;
            stencils       = &c.stencils;
        }

        REQUIRE(stencils->size() == patch.strip_count);

        // ---- `do_init` wrote the rest dihedral angle (0 here) and the per-edge data --
        auto host_theta_bar =
            read_device(theta_bars->cview());
        auto host_yield  = read_device(yields_dev->cview());
        auto host_harden = read_device(hardening_dev->cview());
        auto host_kappa  = read_device(kappas->cview());
        auto host_L0     = read_device(L0s->cview());
        auto host_h      = read_device(h_bars->cview());
        auto host_sten   = read_device(stencils->cview());

        REQUIRE(host_theta_bar.size() == patch.strip_count);
        REQUIRE(host_yield.size() == patch.strip_count);

        // map every stencil to its strip (the strips are disjoint)
        auto strip_of = [&](const gg::LVector4i& s)
        {
            const Vector4i v = gg::to_eigen(s);
            for(SizeT i = 0; i < patch.strip_count; ++i)
            {
                std::array<int, 4> want;
                for(int k = 0; k < 4; ++k)
                    want[k] = static_cast<int>(patch.strip_vertices[i][k]);
                std::sort(want.begin(), want.end());
                std::array<int, 4> got{v(0), v(1), v(2), v(3)};
                std::sort(got.begin(), got.end());
                if(want == got)
                    return i;
            }
            return static_cast<SizeT>(~0ull);
        };

        std::vector<SizeT> strip_ids(patch.strip_count);
        for(SizeT e = 0; e < patch.strip_count; ++e)
        {
            strip_ids[e] = strip_of(host_sten[e]);
            REQUIRE(strip_ids[e] != static_cast<SizeT>(~0ull));
            // the initial history: theta_bar == the rest dihedral angle == 0, and the
            // yield state variable is the frontend/attribute value verbatim
            const auto& yield_in = stress_plastic ? spdsb_yields : yields[0];
            CHECK(std::abs(host_theta_bar[e]) <= 1e-14);
            CHECK(rel_err(host_yield[e], yield_in[strip_ids[e]]) <= 1e-12);
            CHECK(rel_err(host_harden[e], hardening[0][strip_ids[e]]) <= 1e-12);
            if(stress_plastic)
            {
                // `theta_y = yield_stress / elastic_slope` must be the target angle
                const double elastic_slope =
                    2.0 * host_kappa[e] * host_L0[e] / host_h[e];
                CHECK(rel_err(host_yield[e] / elastic_slope, yields[0][strip_ids[e]])
                      <= 1e-12);
                (void)yield_in;
            }
        }

        // ---- the rest positions of the method (the folded states are built from them)
        REQUIRE(FEMConstitutionBProbe::fem != nullptr);
        auto rest_positions = read_device(FEMConstitutionBProbe::fem->x_bars());
        std::vector<Vector3> host_rest;
        host_rest.reserve(rest_positions.size());
        for(auto& p : rest_positions)
            host_rest.push_back(gg::to_eigen(p));
        REQUIRE(host_rest.size() >= patch.rest_positions.size());

        // the reference's per-strip stencil in the *method's* index space
        std::vector<std::array<SizeT, 4>> stencil_indices(patch.strip_count);
        for(SizeT e = 0; e < patch.strip_count; ++e)
        {
            const Vector4i v = gg::to_eigen(host_sten[e]);
            for(int k = 0; k < 4; ++k)
                stencil_indices[e][k] = static_cast<SizeT>(v(k));
        }

        // ---- the oracle state, per element ------------------------------------
        std::vector<double> oracle_theta_bar(patch.strip_count, 0.0);
        std::vector<double> oracle_yield(patch.strip_count);
        for(SizeT e = 0; e < patch.strip_count; ++e)
            oracle_yield[e] = host_yield[e];

        lud::LDeviceBuffer<gg::LVector3> xs_dev(host_rest.size());

        bool saw_a_write = false, saw_a_hold = false;

        for(const double fold : folds)
        {
            // 1) fold every strip by `fold` about its diagonal
            auto host_xs = host_rest;
            for(SizeT e = 0; e < patch.strip_count; ++e)
            {
                const auto& idx = stencil_indices[e];
                const auto  rest =
                    std::array<Vector3, 4>{host_rest[idx[0]], host_rest[idx[1]], host_rest[idx[2]], host_rest[idx[3]]};
                const auto posed = folded_strip(rest, fold);
                for(int k2 = 0; k2 < 4; ++k2)
                    host_xs[idx[k2]] = posed[k2];
            }

            std::vector<gg::LVector3> host_xs_l;
            host_xs_l.reserve(host_xs.size());
            for(auto& x : host_xs)
                host_xs_l.push_back(gg::to_luisa(x));
            xs_dev.copy_from(span<const gg::LVector3>(host_xs_l));

            // 2) one update step on the device, on the constitution's own buffers
            const auto n32 = static_cast<uint32_t>(patch.strip_count);
            if(stress_plastic)
            {
                auto& k = spdsb_update_state_kernel();
                gg::frame_stream().record(k(stencils->view(),
                                            xs_dev.view(),
                                            hardening_dev->view(),
                                            kappas->view(),
                                            L0s->view(),
                                            h_bars->view(),
                                            theta_bars->view(),
                                            yields_dev->view(),
                                            n32)
                                            .dispatch(n32));
            }
            else
            {
                auto& k = pdsb_update_state_kernel();
                gg::frame_stream().record(k(stencils->view(),
                                            xs_dev.view(),
                                            hardening_dev->view(),
                                            theta_bars->view(),
                                            yields_dev->view(),
                                            n32)
                                            .dispatch(n32));
            }
            gg::frame_stream().synchronize();

            auto dev_theta_bar = read_device(theta_bars->cview());
            auto dev_yield     = read_device(yields_dev->cview());

            // 3) the hand-written sequence for every element, from the *reference's*
            //    definition applied to the same positions
            double worst_tb      = 0.0, worst_y = 0.0;
            double worst_host_tb = 0.0;
            for(SizeT e = 0; e < patch.strip_count; ++e)
            {
                const auto& idx = stencil_indices[e];
                const auto  rest =
                    std::array<Vector3, 4>{host_rest[idx[0]], host_rest[idx[1]], host_rest[idx[2]], host_rest[idx[3]]};
                const auto  posed = folded_strip(rest, fold);
                const double theta = eig_dihedral(posed[0], posed[1], posed[2], posed[3]);

                // the *host* instantiation of the same generic routine, driven with the
                // same positions: it must reproduce the oracle to round-off, which
                // isolates the device's own arithmetic (see below)

                if(stress_plastic)
                {
                    StressPlasticState st{oracle_theta_bar[e], oracle_yield[e]};
                    const bool written = eig_stress_plastic_update(theta,
                                                                   st,
                                                                   host_harden[e],
                                                                   host_kappa[e],
                                                                   host_L0[e],
                                                                   host_h[e]);
                    if(written)
                        saw_a_write = true;
                    else
                        saw_a_hold = true;
                    worst_tb = std::max(worst_tb, std::abs(dev_theta_bar[e] - st.theta_bar));
                    worst_y  = std::max(worst_y, std::abs(dev_yield[e] - st.yield_stress));

                    // the *host* instantiation of the same generic routine, driven from
                    // the same pre-step state with the same positions, must reproduce the
                    // oracle to round-off (this isolates the device's own arithmetic)
                    double tb_host = oracle_theta_bar[e];
                    double y_host  = oracle_yield[e];
                    (void)gg::spdsb_detail::update_state<double>(lv<3>(posed[0]),
                                                                lv<3>(posed[1]),
                                                                lv<3>(posed[2]),
                                                                lv<3>(posed[3]),
                                                                host_harden[e],
                                                                host_kappa[e],
                                                                host_L0[e],
                                                                host_h[e],
                                                                tb_host,
                                                                y_host);
                    worst_host_tb =
                        std::max(worst_host_tb, std::abs(tb_host - st.theta_bar));

                    oracle_theta_bar[e] = st.theta_bar;
                    oracle_yield[e]     = st.yield_stress;
                }
                else
                {
                    PlasticState st{oracle_theta_bar[e], oracle_yield[e]};
                    const bool   written =
                        eig_strain_plastic_update(theta, st, host_harden[e], false);
                    if(written)
                        saw_a_write = true;
                    else
                        saw_a_hold = true;
                    worst_tb = std::max(worst_tb, std::abs(dev_theta_bar[e] - st.theta_bar));
                    worst_y  = std::max(worst_y, std::abs(dev_yield[e] - st.yield_threshold));

                    double tb_host = oracle_theta_bar[e];
                    double y_host  = oracle_yield[e];
                    (void)gg::pdsb_detail::update_state<double>(lv<3>(posed[0]),
                                                               lv<3>(posed[1]),
                                                               lv<3>(posed[2]),
                                                               lv<3>(posed[3]),
                                                               host_harden[e],
                                                               tb_host,
                                                               y_host);
                    worst_host_tb =
                        std::max(worst_host_tb, std::abs(tb_host - st.theta_bar));

                    oracle_theta_bar[e] = st.theta_bar;
                    oracle_yield[e]     = st.yield_threshold;
                }
            }

            std::printf("[t52b] %s history step (fold %+.7f):"
                        " max |theta_bar_dev - oracle| %.3e  max |state_dev - oracle| %.3e"
                        "  max |host-generic - oracle| %.3e\n",
                        stress_plastic ? "SPDSB" : "PDSB",
                        fold,
                        worst_tb,
                        worst_y,
                        worst_host_tb);
            // The host instantiation of the shipped generic routine reproduces the
            // oracle to round-off ...
            CHECK(worst_host_tb <= 1e-14);
            // ... while the *device* history differs by ~1e-7: LC compiles its shaders
            // with `-use_fast_math`, so the DSL's `acos`/`sqrt` are approximate and the
            // dihedral angle carries ~1e-8, which the update passes on unchanged (the
            // step is a *difference* of two O(1) angles).  The state variable itself
            // (`yield_threshold` / `yield_stress`) is a plain sum, so it stays tighter.
            CHECK(worst_tb <= 1e-6);
            CHECK(worst_y <= 1e-7);
        }

        // the sequence must contain both kinds of step
        CHECK(saw_a_write);
        CHECK(saw_a_hold);

        // and the final state must still carry the hand-computable structure: strip 1
        // (yield target 0.60 rad, i.e. a yield stress of `0.60 * elastic_slope` in the
        // stress-plastic case) never yielded, so its history is exactly the initial one
        {
            const auto& yield_in = stress_plastic ? spdsb_yields : yields[0];
            for(SizeT e = 0; e < patch.strip_count; ++e)
            {
                if(strip_ids[e] == 1)
                {
                    CHECK(std::abs(oracle_theta_bar[e]) <= 1e-14);
                    CHECK(rel_err(oracle_yield[e], yield_in[1]) <= 1e-12);
                }
            }
        }
    };

    run_plastic_history(false);  // strain-plastic  (uid 31)
    run_plastic_history(true);   // stress-plastic  (uid 32)
}

// ---------------------------------------------------------------------------
//  5) degenerate / singular states, *pinned from the reference source* (the port
//     must reproduce the reference's behaviour, including "no repair")
//
//  What the source says (read off the reference implementations):
//   * `discrete_shell_bending_function.h::E/dEdx/ddEddx` call the *plain*
//     `dihedral_angle`, whose `acos(n1.n2 / sqrt(|n1|^2 |n2|^2))` is 0/0 for a
//     degenerate plane (three collinear points, or a zero-length mid-edge with a
//     vanishing area) -> NaN, and there is **no** input guard anywhere.  E, G and H
//     are therefore non-finite and the port does not repair them.
//   * the two *plastic* variants guard first (`safe_dihedral_angle` /
//     `try_angle_delta`) and return `0` resp. `setZero()` on failure, and the
//     history update returns `false` (no write).
//   * `neo_hookean_shell_2d.inl`'s `x29 = log(...)`: the logarithm's argument is
//     `det(IB) * (a squared-determinant expression)`, so a *reflection* is invisible
//     (the sign is squared away) and a collapsed triangle (det -> 0) gives `-inf`
//     resp. `+inf`.
//   * the strain-limiting model guards its `1/sqrt(I5)` with `max(., 1e-12)`, so a
//     collapsed triangle still returns a finite energy, but its `ddEddF` builds
//     `q0 = (...).normalized()` from a *zero* vector there -> NaN.
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_b_degenerate_inputs", "[luisa][fem][constitution_b]")
{
    test_runtime();

    const double kappa = 1.3;
    const double dt    = 1.0;

    // ---- a degenerate bending stencil: three collinear points -----------------
    auto degenerate_stencil = []()
    {
        Stencil s;
        // X0, X1, X2 on one line and X3 on the same line: both triangle normals
        // vanish identically
        s.X0 = s.x0 = Vector3{0.0, 0.0, 0.0};
        s.X1 = s.x1 = Vector3{1.0, 0.0, 0.0};
        s.X2 = s.x2 = Vector3{2.0, 0.0, 0.0};
        s.X3 = s.x3 = Vector3{3.0, 0.0, 0.0};
        return s;
    };

    // =======================================================================
    //  A) discrete shells (elastic): no guard -> non-finite energy/gradient
    // =======================================================================
    {
        const auto s = degenerate_stencil();

        double L0 = 0.0, h_bar = 0.0, theta_bar = 0.0, V_bar = 0.0;
        sym_dsb::compute_constants<double>(L0,
                                           h_bar,
                                           theta_bar,
                                           V_bar,
                                           lv<3>(s.X0),
                                           lv<3>(s.X1),
                                           lv<3>(s.X2),
                                           lv<3>(s.X3),
                                           0.001,
                                           0.001,
                                           0.001,
                                           0.001);

        // the collapsed planes make the rest dihedral angle a *degenerate* value; the
        // reference has no check for it.  NOTE (pinned, reported): the reference's
        // `std::max(-1, std::min(1, 0/0))` returns `1` (std::min keeps its *first*
        // argument when the comparison is false) and therefore `theta_bar == 0`,
        // whereas the numerics layer's `lps::min(x, y) = x < y ? x : y` keeps the NaN
        // and `acos(-1) == pi`.  Both are finite, and the energy below is non-finite
        // either way (`h_bar = A / 3 / L0` is 0/0 = NaN), so the *behaviour* is the
        // same; only a NaN-input comparison inside the layer differs.
        CHECK(std::isfinite(theta_bar));
        CHECK(std::abs(std::abs(theta_bar) - kPi) <= 1e-15);

        const double E = ported_dsb_energy(s, kappa, L0, h_bar, theta_bar, V_bar, dt);
        CHECK(!std::isfinite(E));

        const Vector12 G = ported_dsb_gradient(s, kappa, L0, h_bar, theta_bar, V_bar, dt);
        CHECK(!std::isfinite(max_abs(G)));

        // and the Hessian: `make_spd` cannot repair a NaN (its comparisons are all
        // false, so the select keeps the NaN through the reconstruction)
        const Matrix12x12 H =
            ported_dsb_hessian(s, kappa, L0, h_bar, theta_bar, V_bar, dt);
        CHECK(!std::isfinite(max_abs(H)));
    }

    // =======================================================================
    //  B) a zero-length mid-edge (`L0 == 0`, so `h_bar == +inf`) of an otherwise
    //     well-shaped stencil
    // =======================================================================
    {
        Stencil s;
        s.X0 = Vector3{-1.0, 0.0, 0.0};
        s.X1 = Vector3{0.0, 0.0, -1.0};
        s.X2 = Vector3{0.0, 0.0, -1.0};  // X2 == X1: the mid-edge has zero length
        s.X3 = Vector3{1.0, 0.0, 0.0};
        s.x0 = s.X0;
        s.x1 = s.X1;
        s.x2 = s.X2;
        s.x3 = s.X3;

        double L0 = 0.0, h_bar = 0.0, theta_bar = 0.0, V_bar = 0.0;
        sym_dsb::compute_constants<double>(L0,
                                           h_bar,
                                           theta_bar,
                                           V_bar,
                                           lv<3>(s.X0),
                                           lv<3>(s.X1),
                                           lv<3>(s.X2),
                                           lv<3>(s.X3),
                                           0.001,
                                           0.001,
                                           0.001,
                                           0.001);

        CHECK(L0 == 0.0);
        // `A / 3 / L0` is `0 / 3 / 0`, i.e. NaN (a zero-length mid-edge collapses both
        // triangles, so the area `A` vanishes as well -- there is no "well-shaped"
        // stencil with `L0 == 0`)
        CHECK(std::isnan(h_bar));

        // elastic: `L0 kappa (theta - theta_bar)^2 / h_bar` with `h_bar = NaN`
        const double E = ported_dsb_energy(s, kappa, L0, h_bar, theta_bar, V_bar, dt);
        std::printf("[t52b] DSB with a zero-length mid-edge: L0 %.3e  h_bar %.3e"
                    "  theta_bar %.3e  E %.3e\n",
                    L0,
                    h_bar,
                    theta_bar,
                    E);
        CHECK(std::isnan(E));

        // the *plastic* variants guard `L0`/`h_bar` (the stress-plastic one explicitly:
        // `L0 <= 0 || h_bar <= 0`), so they return exactly 0
        CHECK(ported_pdsb_energy(s, kappa, L0, h_bar, theta_bar, V_bar, dt) == 0.0);
        CHECK(ported_spdsb_energy(s, kappa, L0, h_bar, theta_bar, 0.1, V_bar, dt) == 0.0);
    }

    // =======================================================================
    //  C) the plastic constitutions: every guard of `update_plastic_state` must
    //     leave the history untouched and return false
    // =======================================================================
    {
        // ---- a degenerate configuration (collinear points) -----------------------
        const auto s = degenerate_stencil();

        // the strain-plastic energy/gradient/Hessian are exactly 0 (`return 0.0` /
        // `setZero()`), because `try_angle_delta` fails
        CHECK(ported_pdsb_energy(s, kappa, 2.0, 1.0 / 3.0, 0.0, 1.0, dt) == 0.0);
        CHECK(max_abs(ported_pdsb_gradient(s, kappa, 2.0, 1.0 / 3.0, 0.0, 1.0, dt)) == 0.0);
        CHECK(max_abs(ported_pdsb_hessian(s, kappa, 2.0, 1.0 / 3.0, 0.0, 1.0, dt)) == 0.0);
        CHECK(ported_spdsb_energy(s, kappa, 2.0, 1.0 / 3.0, 0.0, 0.1, 1.0, dt) == 0.0);
        CHECK(max_abs(ported_spdsb_gradient(s, kappa, 2.0, 1.0 / 3.0, 0.0, 0.1, 1.0, dt)) == 0.0);
        CHECK(max_abs(ported_spdsb_hessian(s, kappa, 2.0, 1.0 / 3.0, 0.0, 0.1, 1.0, dt)) == 0.0);

        // ---- the history update on a degenerate configuration -------------------
        {
            double theta_bar = 0.25;
            double yield     = 0.05;
            const double tb0 = theta_bar, y0 = yield;
            CHECK(!gg::pdsb_detail::update_state<double>(
                lv<3>(s.x0), lv<3>(s.x1), lv<3>(s.x2), lv<3>(s.x3), 0.1, theta_bar, yield));
            CHECK(theta_bar == tb0);  // the history must not change
            CHECK(yield == y0);

            double yield_stress = 36000.0;
            const double ys0    = yield_stress;
            CHECK(!gg::spdsb_detail::update_state<double>(lv<3>(s.x0),
                                                          lv<3>(s.x1),
                                                          lv<3>(s.x2),
                                                          lv<3>(s.x3),
                                                          0.1,
                                                          kappa,
                                                          2.0,
                                                          1.0 / 3.0,
                                                          theta_bar,
                                                          yield_stress));
            CHECK(theta_bar == tb0);
            CHECK(yield_stress == ys0);
        }

        // ---- the scalar guards of `update_plastic_state` ------------------------
        {
            // a negative yield threshold / hardening modulus is rejected outright
            double theta_bar = 0.0;
            double yield     = -0.1;
            CHECK(!sym_pdsb::update_plastic_state<double>(0.5, theta_bar, yield, 0.0));
            CHECK(theta_bar == 0.0);
            CHECK(yield == -0.1);

            double yield_ok = 0.05;
            CHECK(!sym_pdsb::update_plastic_state<double>(0.5, theta_bar, yield_ok, -0.1));
            CHECK(theta_bar == 0.0);
            CHECK(yield_ok == 0.05);

            // a non-finite angle never writes
            double tb_nan = std::numeric_limits<double>::quiet_NaN();
            double y_ok   = 0.05;
            CHECK(!sym_pdsb::update_plastic_state<double>(0.5, tb_nan, y_ok, 0.0));
            CHECK(std::isnan(tb_nan));
            CHECK(y_ok == 0.05);

            // the stress-plastic variant additionally rejects `L0 <= 0`, `h_bar <= 0`
            // and a negative yield stress (`try_trial_state`)
            double tb2 = 0.0, ys2 = 0.4;
            CHECK(!sym_spdsb::update_plastic_state<double>(0.5, tb2, ys2, 0.0, 1.0, 0.0, 1.0));
            CHECK(tb2 == 0.0);
            CHECK(ys2 == 0.4);

            double tb3 = 0.0, ys3 = -0.4;
            CHECK(!sym_spdsb::update_plastic_state<double>(0.5, tb3, ys3, 0.0, 1.0, 1.0, 1.0));
            CHECK(tb3 == 0.0);
            CHECK(ys3 == -0.4);
        }
    }

    // =======================================================================
    //  D) Neo-Hookean shell: `log(det(IB) * (...))`
    // =======================================================================
    {
        const double lambda = 3.3, mu = 1.7;

        // ---- a degenerate *rest* triangle: `A(X_bar)` is singular --------------
        {
            Stencil s;
            s.X0 = s.x0 = Vector3{0.0, 0.0, 0.0};
            s.X1 = s.x1 = Vector3{1.0, 0.0, 0.0};
            s.X2 = s.x2 = Vector3{2.0, 0.0, 0.0};  // collinear -> zero area
            const auto   IB   = nh_IB(s);
            const double area = triangle_rest_area(s);
            CHECK(area == 0.0);
            CHECK(!std::isfinite(max_abs(IB)));

            const double E = ported_nh_energy(s, lambda, mu, IB, area, 0.001, dt);
            CHECK(!std::isfinite(E));
        }

        // ---- a collapsed *deformed* triangle with a valid rest metric ----------
        {
            auto   s = unit_triangle();
            const auto IB   = nh_IB(s);
            const double area = triangle_rest_area(s);

            Stencil collapsed = s;
            collapsed.x1      = Vector3{0.0, 0.0, 0.0};
            collapsed.x2      = Vector3{0.0, 0.0, 0.0};
            const double E = ported_nh_energy(collapsed, lambda, mu, IB, area, 0.001, dt);
            // `x29 = log(0) = -inf` -> `0.5 mu (Ic - 2 - 2 ln J)` = +inf
            CHECK(!std::isfinite(E));
            CHECK(!std::isfinite(
                max_abs(ported_nh_gradient(collapsed, lambda, mu, IB, area, 0.001, dt))));
        }

        // ---- a *reflection* is invisible: `J = det(F^T F)^(1/2) >= 0` ----------
        {
            const auto   s    = unit_triangle();
            const auto   IB   = nh_IB(s);
            const double area = triangle_rest_area(s);

            // F = diag(s, s) * (a reflection of the first axis): |det F| is the same
            Stencil stretched = s;
            stretched.x1      = Vector3{1.7, 0.0, 0.0};
            stretched.x2      = Vector3{0.0, 1.7, 0.0};

            Stencil reflected = s;
            reflected.x1      = Vector3{-1.7, 0.0, 0.0};
            reflected.x2      = Vector3{0.0, 1.7, 0.0};

            const double E_str = ported_nh_energy(stretched, lambda, mu, IB, area, 0.001, dt);
            const double E_ref = ported_nh_energy(reflected, lambda, mu, IB, area, 0.001, dt);
            std::printf("[t52b] NH shell: E(stretch) %.12f  E(reflection) %.12f\n", E_str, E_ref);
            CHECK(rel_err(E_str, E_ref) <= 1e-14);
            CHECK(std::isfinite(E_str));
        }
    }

    // =======================================================================
    //  E) Baraff-Witkin: the `max(., 1e-12)` guard and the `q0.normalized()` hole
    // =======================================================================
    {
        const double stretchS = 2.5, shearS = 0.75, sr = 100.0;

        // ---- a degenerate rest triangle: `normal.normalized()` is 0/0 ----------
        {
            Stencil s;
            s.X0 = s.x0 = Vector3{0.0, 0.0, 0.0};
            s.X1 = s.x1 = Vector3{1.0, 0.0, 0.0};
            s.X2 = s.x2 = Vector3{2.0, 0.0, 0.0};
            const auto   IB   = bws_IB(s);
            const double area = triangle_rest_area(s);
            CHECK(!std::isfinite(max_abs(IB)));

            const double E = ported_bws_energy(s, stretchS, shearS, sr, IB, area, dt);
            CHECK(!std::isfinite(E));
        }

        // ---- a collapsed deformed triangle with a valid rest metric ------------
        {
            auto   s    = unit_triangle();
            const auto IB   = nh_IB(s);
            const double area = triangle_rest_area(s);

            Stencil collapsed = s;
            collapsed.x1      = Vector3{0.0, 0.0, 0.0};
            collapsed.x2      = Vector3{0.0, 0.0, 0.0};

            // `I5u = I5v = 0` -> the `max(., 1e-12)` guard keeps the *energy* finite
            // (the ucoeff select de-activates the cubic term: `(0 - 1)^2` twice)
            const double E = ported_bws_energy(collapsed, stretchS, shearS, sr, IB, area, dt)
                             / (area * dt * dt);
            const double G_norm = max_abs(
                ported_bws_gradient(collapsed, stretchS, shearS, sr, IB, area, dt));
            const double H_norm = max_abs(
                ported_bws_hessian(collapsed, stretchS, shearS, sr, IB, area, dt));
            std::printf("[t52b] BW shell collapsed triangle: E %.6f (expect %.6f),"
                        " |G| %.3e, |H| %.3e\n",
                        E,
                        2.0 * stretchS,
                        G_norm,
                        H_norm);
            CHECK(rel_err(E, 2.0 * stretchS) <= 1e-12);
            CHECK(std::isfinite(G_norm));
            // ... while `ddEddF` builds `q0 = (I6*H*vec_g + lambda0*vec_g).normalized()`
            // from `vec_g == 0` (g = F * (...) = 0), i.e. 0/0 -> NaN
            CHECK(!std::isfinite(H_norm));
        }
    }
}

namespace
{
// ---------------------------------------------------------------------------
//  scene builders for the end-to-end case
// ---------------------------------------------------------------------------
enum class BendingKind
{
    Elastic,
    StrainPlastic,
    StressPlastic
};

/// The strip patch plus a membrane (for the `thickness` attribute the FEM method
/// needs) and one bending constitution on its edges.  `yield_value` / `hardening` are
/// per-strip values written to the edges the constitution reads.
Scene make_bending_scene(const StripPatch&          patch,
                         BendingKind                kind,
                         std::string_view           name,
                         double                     kappa,
                         const std::vector<double>& yield_value,
                         const std::vector<double>& hardening,
                         S<geometry::GeometrySlot>* slot_out = nullptr)
{
    auto config = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = false;

    Scene scene{config};
    auto  object = scene.objects().create(name);

    auto mesh = trimesh(span<const Vector3>{patch.rest_positions},
                        span<const Vector3i>{patch.triangles});
    label_surface(mesh);

    NeoHookeanShell nhs;
    nhs.apply_to(mesh, ElasticModuli2D::youngs_poisson(1.0e6, 0.3), 500.0, 0.001);

    switch(kind)
    {
        case BendingKind::Elastic:
        {
            DiscreteShellBending dsb;
            dsb.apply_to(mesh, kappa);
            break;
        }
        case BendingKind::StrainPlastic:
        {
            StrainPlasticDiscreteShellBending pdsb;
            pdsb.apply_to(mesh, kappa, 0.1, 0.0);
            break;
        }
        case BendingKind::StressPlastic:
        {
            StressPlasticDiscreteShellBending spdsb;
            spdsb.apply_to(mesh, kappa, 0.1_kPa, 0.0);
            break;
        }
    }

    // per-strip attributes on the bending stencil's middle edge (the quad diagonal)
    if(kind != BendingKind::Elastic)
    {
        auto edges = mesh.edges().topo().view();
        auto bs    = mesh.edges().find<Float>("bending_stiffness");
        auto hm    = mesh.edges().find<Float>("bending_hardening_modulus");
        auto yt    = mesh.edges().find<Float>("bending_yield_threshold");
        auto ys    = mesh.edges().find<Float>("bending_yield_stress");
        REQUIRE(bs != nullptr);
        REQUIRE(hm != nullptr);

        for(SizeT s = 0; s < patch.strip_count; ++s)
        {
            const auto v = patch.strip_vertices[s];
            const int  a = static_cast<int>(std::min(v[1], v[2]));
            const int  b = static_cast<int>(std::max(v[1], v[2]));
            bool       found = false;
            for(SizeT e = 0; e < edges.size(); ++e)
            {
                const int e0 = std::min(edges[e](0), edges[e](1));
                const int e1 = std::max(edges[e](0), edges[e](1));
                if(e0 == a && e1 == b)
                {
                    view(*bs)[e] = kappa;
                    view(*hm)[e] = hardening[s];
                    if(yt)
                        view(*yt)[e] = yield_value[s];
                    if(ys)
                        view(*ys)[e] = yield_value[s];
                    found        = true;
                    break;
                }
            }
            CHECK(found);
        }
    }

    const auto [slot, rest_slot] = object->geometries().create(mesh);
    (void)rest_slot;

    // The deformed configuration has to be written *through the live slot*: the rest
    // slot is the snapshot `create()` takes, and the FEM method reads `x_bars` from it
    // (writing into `mesh` before `create()` would make the *rest* state the deformed
    // one, which is exactly the trap this test hit first).
    {
        auto sc  = slot->geometry().as<SimplicialComplex>();
        auto pos = view(sc->positions());
        for(SizeT s = 0; s < patch.strip_count; ++s)
        {
            const auto&  v   = patch.strip_vertices[s];
            const double phi = 0.35 + 0.05 * static_cast<double>(s);
            // the stencil order is (opp0, mid0, mid1, opp1); fold the `opp1` vertex
            // about the mid-edge
            const Vector3 axis =
                (patch.rest_positions[v[2]] - patch.rest_positions[v[1]]).normalized();
            const Vector3 x1 = patch.rest_positions[v[1]];
            const Vector3 x3 = patch.rest_positions[v[3]];
            const Vector3 moved =
                x1 + Eigen::AngleAxisd(phi, axis).toRotationMatrix() * (x3 - x1);
            pos[v[3]] = moved;
        }
    }

    if(slot_out)
        *slot_out = slot;
    return scene;
}

/// A membrane triangle patch with per-triangle moduli (the Neo-Hookean and
/// strain-limiting shells).
Scene make_membrane_scene(const StripPatch&           patch,
                          bool                        baraff_witkin,
                          std::string_view            name,
                          const std::vector<double>&  moduli_a,
                          const std::vector<double>&  moduli_b,
                          const std::vector<double>&  strain_rates,
                          S<geometry::GeometrySlot>*  slot_out = nullptr)
{
    auto config = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = false;

    Scene scene{config};
    auto  object = scene.objects().create(name);

    auto mesh = trimesh(span<const Vector3>{patch.rest_positions},
                        span<const Vector3i>{patch.triangles});
    label_surface(mesh);

    if(baraff_witkin)
    {
        StrainLimitingBaraffWitkinShell sw;
        sw.apply_to(mesh,
                    ElasticModuli2D::youngs_poisson(1.0e6, 0.3),
                    ElasticModuli2D::youngs_poisson(5.0e5, 0.3),
                    500.0,
                    0.001,
                    strain_rates[0]);
    }
    else
    {
        NeoHookeanShell nhs;
        nhs.apply_to(mesh, ElasticModuli2D::youngs_poisson(1.0e6, 0.3), 500.0, 0.001);
    }

    // distinct per-triangle parameters (the constitutions must read them per element)
    {
        auto lambda = mesh.triangles().find<Float>("lambda");
        auto mu     = mesh.triangles().find<Float>("mu");
        REQUIRE(lambda != nullptr);
        REQUIRE(mu != nullptr);
        auto sr = mesh.triangles().find<Float>("strain_rate");

        for(SizeT i = 0; i < view(*lambda).size(); ++i)
        {
            view(*lambda)[i] = moduli_a[i % moduli_a.size()];
            view(*mu)[i]     = moduli_b[i % moduli_b.size()];
            if(sr)
                view(*sr)[i] = strain_rates[i % strain_rates.size()];
        }
    }

    const auto [slot, rest_slot] = object->geometries().create(mesh);
    (void)rest_slot;

    // the deformed state goes into the *live* slot (see `make_bending_scene`)
    {
        auto sc  = slot->geometry().as<SimplicialComplex>();
        auto pos = view(sc->positions());
        for(SizeT s = 0; s < patch.strip_count; ++s)
        {
            const auto&  v = patch.strip_vertices[s];
            const double k = 0.15 * static_cast<double>(s + 1);
            // shear the quad: v2 and v3 move along x
            pos[v[2]] = patch.rest_positions[v[2]] + Vector3{k * 0.5, 0.0, 0.0};
            pos[v[3]] = patch.rest_positions[v[3]] + Vector3{k, 0.2 * k, 0.0};
        }
    }

    if(slot_out)
        *slot_out = slot;
    return scene;
}
}  // namespace

namespace
{
/// The element-local result of one element (the oracle of the assembly checks).
template <int N>
struct ElementResultN
{
    double                              energy = 0.0;
    Vector<Float, 3 * N>                gradient;
    Matrix<Float, 3 * N, 3 * N>         hessian;
};

/// Check the constitution's assembled energies/doublets/triplets against the
/// per-element oracle: the element `e` owns the doublets `N*e .. N*e+N-1` and the
/// triplets `Half*e .. Half*e+Half-1` of *its own slice*, with the reference's
/// `upper_LR` block ordering inside the triplets.
template <int N, typename Fn>
void check_assembly_oracle(SizeT                                  n,
                           const std::vector<std::array<int, N>>& elems,
                           const std::vector<double>&             dev_e,
                           const std::vector<int32_t>&            dev_di,
                           const std::vector<double>&             dev_dv,
                           const std::vector<int32_t>&            dev_tr,
                           const std::vector<int32_t>&            dev_tc,
                           const std::vector<double>&             dev_tv,
                           bool                                   has_hessian,
                           Fn&&                                   fn,
                           double                                 tol,
                           const char*                            what)
{
    constexpr int Half = N * (N + 1) / 2;

    // an energy-only call (`dev_di` empty) has no doublet/triplet data yet -- the
    // energy kernel does not touch those buffers
    const bool check_vectors = !dev_di.empty();

    // The device carries LC's fast-math transcendentals (the bending routines go
    // through `acos`), so the comparison is *relative to the element's own scale* --
    // an absolute bound would be meaningless for kappa = 5 kPa (~1e2 gradients).
    double worst_e = 0.0, worst_g = 0.0, worst_h = 0.0;
    double scale_e = 1.0, scale_g = 1.0, scale_h = 1.0;
    for(SizeT e = 0; e < n; ++e)
    {
        const auto r = fn(e);
        worst_e      = std::max(worst_e, std::abs(dev_e[e] - r.energy));
        scale_e      = std::max(scale_e, std::abs(r.energy));

        if(!check_vectors)
            continue;

        for(int i = 0; i < N; ++i)
        {
            CHECK(dev_di[N * e + i] == elems[e][i]);
            for(int c = 0; c < 3; ++c)
            {
                worst_g = std::max(
                    worst_g, std::abs(dev_dv[(N * e + i) * 3 + c] - r.gradient(3 * i + c)));
                scale_g = std::max(scale_g, std::abs(r.gradient(3 * i + c)));
            }
        }

        if(!has_hessian)
            continue;

        int k = 0;
        for(int i = 0; i < N; ++i)
            for(int j = i; j < N; ++j, ++k)
            {
                const bool keep = elems[e][i] < elems[e][j];
                const int  l    = keep ? i : j;
                const int  rr   = keep ? j : i;
                CHECK(dev_tr[Half * e + k] == elems[e][l]);
                CHECK(dev_tc[Half * e + k] == elems[e][rr]);
                for(int c = 0; c < 3; ++c)
                    for(int q = 0; q < 3; ++q)
                    {
                        worst_h = std::max(
                            worst_h,
                            std::abs(dev_tv[(Half * e + k) * 9 + c * 3 + q]
                                     - r.hessian(3 * l + q, 3 * rr + c)));
                        scale_h = std::max(scale_h,
                                           std::abs(r.hessian(3 * l + q, 3 * rr + c)));
                    }
            }
    }

    const double rel_e = worst_e / scale_e;
    const double rel_g = worst_g / scale_g;
    const double rel_h = worst_h / scale_h;
    std::printf("[t52b] %s: assembled vs the element oracle  E %.3e (rel %.3e)  G %.3e"
                " (rel %.3e)  H %.3e (rel %.3e)\n",
                what,
                worst_e,
                rel_e,
                worst_g,
                rel_g,
                worst_h,
                rel_h);
    CHECK(rel_e <= tol);
    CHECK(rel_g <= tol);
    if(has_hessian)
        CHECK(rel_h <= tol);
}

/// The doublet/triplet scratch of one constitution's slice (the layout the elastics
/// hands to a constitution: its own slice at offset 0 and global vertex indices as
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
}  // namespace

// ---------------------------------------------------------------------------
//  6) real scenes through the public Engine/World API
//
//  Every constitution runs its own scene: the engine builds and initializes the real
//  system (`do_build` / `do_init`, i.e. the stencil / attribute / constant setup), and
//  the test then drives the *real* energy and gradient/Hessian kernels with scratch
//  buffers laid out exactly like the elastics' (this constitution's own slice at
//  offset 0, global vertex indices as block rows/cols) and compares the assembled
//  energies / doublets / triplets against a host model built from the shipped
//  per-element routines.
//
//  SHARED-BINARY NOTE (the brief's §4bis): `luisa_fem_base_test.cpp` registers mock
//  constitutions that claim the *frontend* uids -- including DiscreteShellBending's 17
//  and NeoHookeanShell's 11.  In a full-suite run such a mock can own the (dim, uid)
//  group, so those two cases *detect* it (the constitution's own `do_init` output is
//  compared against the scene's element count), say so loudly with a WARN and skip
//  only the affected block.  Run the filtered case (`"<exe> [constitution_b]"`) to
//  exercise all five end to end.
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_b_scene", "[luisa][fem][constitution_b]")
{
    test_runtime();

    const double dt = 1.0;

    // =======================================================================
    //  A) the bending constitutions (extra constitutions, uids 17 / 31 / 32)
    // =======================================================================
    const auto run_bending = [&](BendingKind                kind,
                                 U64                        uid,
                                 const char*                name,
                                 double                     kappa,
                                 const std::vector<double>& yield_value,
                                 const std::vector<double>& hardening)
    {
        const auto patch = make_strip_patch(4);
        S<geometry::GeometrySlot> slot;
        auto                      scene = make_bending_scene(patch,
                                        kind,
                                        fmt::format("t52b_scene_{}", name),
                                        kappa,
                                        yield_value,
                                        hardening,
                                        &slot);

        const auto workspace = workspace_path(fmt::format("uipc_luisa_t52b_{}", name));
        Engine     engine{"luisa", workspace};
        World      world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());

        // ---- which system owns this uid's group? -------------------------------
        gg::FiniteElementExtraConstitution* sys = nullptr;
        switch(kind)
        {
            case BendingKind::Elastic:
                sys = FEMConstitutionBProbe::dsb;
                break;
            case BendingKind::StrainPlastic:
                sys = FEMConstitutionBProbe::pdsb;
                break;
            default:
                sys = FEMConstitutionBProbe::spdsb;
                break;
        }

        if(sys == nullptr)
        {
            WARN(fmt::format("the {} constitution was not built for this scene", name));
            return;
        }
        CHECK(sys->uid() == uid);
        // the demangled dynamic type name of any of the three is a
        // `...DiscreteShellBending...` class
        const bool named_ok =
            std::string_view(sys->name()).find("DiscreteShellBending") != std::string_view::npos;
        CHECK(named_ok);

        // the constitution's own `do_init` output decides whether it owned the group
        const auto& stencils_host = (kind == BendingKind::Elastic)
                                        ? FEMConstitutionBProbe::dsb->h_stencils
                                    : (kind == BendingKind::StrainPlastic)
                                        ? FEMConstitutionBProbe::pdsb->h_stencils
                                        : FEMConstitutionBProbe::spdsb->h_stencils;
        const SizeT n = stencils_host.size();

        if(n != patch.strip_count)
        {
            WARN(fmt::format("{} (uid {}) did NOT own its (dim, uid) geometry group in this "
                             "binary: luisa_fem_base_test.cpp's mock constitution claims "
                             "that uid first.  Run the filtered case to exercise it end "
                             "to end.",
                             name,
                             uid));
            std::printf("[t52b] SKIPPED %s end-to-end checks (group owned elsewhere:"
                        " %llu of %llu elements)\n",
                        name,
                        static_cast<unsigned long long>(n),
                        static_cast<unsigned long long>(patch.strip_count));
            return;
        }

        // ---- 1) `do_init`: the per-element constants ---------------------------
        const auto read_const = [&](auto&& member) { return read_device(member.cview()); };

        std::vector<double>  L0s, h_bars, thetas, V_bars, kappas, exp_yield;
        std::vector<Vector4i> stencils;
        if(kind == BendingKind::Elastic)
        {
            auto& c = *FEMConstitutionBProbe::dsb;
            L0s     = read_const(c.rest_lengths);
            h_bars  = read_const(c.h_bars);
            thetas  = read_const(c.theta_bars);
            V_bars  = read_const(c.V_bars);
            kappas  = read_const(c.bending_stiffnesses);
            exp_yield.assign(n, 0.0);
        }
        else if(kind == BendingKind::StrainPlastic)
        {
            auto& c = *FEMConstitutionBProbe::pdsb;
            L0s     = read_const(c.rest_lengths);
            h_bars  = read_const(c.h_bars);
            thetas  = read_const(c.theta_bars);
            V_bars  = read_const(c.V_bars);
            kappas  = read_const(c.bending_stiffnesses);
            exp_yield = read_const(c.yield_thresholds);
        }
        else
        {
            auto& c = *FEMConstitutionBProbe::spdsb;
            L0s     = read_const(c.rest_lengths);
            h_bars  = read_const(c.h_bars);
            thetas  = read_const(c.theta_bars);
            V_bars  = read_const(c.V_bars);
            kappas  = read_const(c.bending_stiffnesses);
            exp_yield = read_const(c.yield_stresses);
        }

        for(SizeT e = 0; e < n; ++e)
            stencils.push_back(stencils_host[e]);

        double worst_constant = 0.0;
        for(SizeT e = 0; e < n; ++e)
        {
            const Vector4i v = stencils[e];
            // the stencil must be one of the strip quads
            SizeT strip = ~0ull;
            for(SizeT s = 0; s < patch.strip_count && strip == ~0ull; ++s)
            {
                std::array<int, 4> want, got{v(0), v(1), v(2), v(3)};
                for(int k = 0; k < 4; ++k)
                    want[k] = static_cast<int>(patch.strip_vertices[s][k]);
                std::sort(want.begin(), want.end());
                std::sort(got.begin(), got.end());
                if(want == got)
                    strip = s;
            }
            REQUIRE(strip != ~0ull);
            // the stencil's own vertex order is (opp0, mid0, mid1, opp1)
            const auto& rp = patch.rest_positions;

            // the reference metric, recomputed independently in Eigen
            double L0_o = 0.0, h_bar_o = 0.0, theta_bar_o = 0.0, outer = 0.0;
            eig_dsb_constants(rp[v(0)],
                              rp[v(1)],
                              rp[v(2)],
                              rp[v(3)],
                              L0_o,
                              h_bar_o,
                              theta_bar_o,
                              outer);

            worst_constant = std::max(worst_constant, rel_err(L0s[e], L0_o));
            worst_constant = std::max(worst_constant, rel_err(h_bars[e], h_bar_o));
            worst_constant = std::max(worst_constant, rel_err(thetas[e], theta_bar_o));
            worst_constant = std::max(worst_constant, rel_err(V_bars[e], outer));
            CHECK(rel_err(kappas[e], kappa) <= 1e-15);
            if(kind != BendingKind::Elastic)
                CHECK(rel_err(exp_yield[e], yield_value[strip]) <= 1e-12);
        }
        std::printf("[t52b] %s do_init constants vs the independent Eigen metric:"
                    " worst rel %.3e\n",
                    name,
                    worst_constant);
        CHECK(worst_constant <= 1e-12);

        // ---- 2) the element oracle --------------------------------------------
        std::vector<std::array<int, 4>> elems(n);
        for(SizeT e = 0; e < n; ++e)
            for(int k = 0; k < 4; ++k)
                elems[e][k] = stencils[e](k);

        gg::FiniteElementExtraConstitution::Impl extra_impl;
        extra_impl.finite_element_method    = FEMConstitutionBProbe::fem;
        gg::FiniteElementExtraConstitution::BaseInfo base_info{&extra_impl, dt};

        const auto           xs_dev = read_device(base_info.xs());
        std::vector<Vector3> host_xs;
        host_xs.reserve(xs_dev.size());
        for(auto& x : xs_dev)
            host_xs.push_back(gg::to_eigen(x));

        std::vector<double> hardenings(n, 0.0);
        if(kind == BendingKind::StrainPlastic)
            hardenings = read_const(FEMConstitutionBProbe::pdsb->hardening_moduli);
        else if(kind == BendingKind::StressPlastic)
            hardenings = read_const(FEMConstitutionBProbe::spdsb->hardening_moduli);

        const double vdt2_scale = 1.0;  // (the constitutions apply it internally)

        auto elem_fn = [&](SizeT e)
        {
            Stencil s;
            const Vector4i v = stencils[e];
            s.x0             = host_xs[v(0)];
            s.x1             = host_xs[v(1)];
            s.x2             = host_xs[v(2)];
            s.x3             = host_xs[v(3)];

            if(kind == BendingKind::Elastic)
            {
                return ElementResultN<4>{
                    ported_dsb_energy(s, kappas[e], L0s[e], h_bars[e], thetas[e], V_bars[e], dt),
                    ported_dsb_gradient(s, kappas[e], L0s[e], h_bars[e], thetas[e], V_bars[e], dt),
                    ported_dsb_hessian(s, kappas[e], L0s[e], h_bars[e], thetas[e], V_bars[e], dt)};
            }
            if(kind == BendingKind::StrainPlastic)
            {
                return ElementResultN<4>{
                    ported_pdsb_energy(s, kappas[e], L0s[e], h_bars[e], thetas[e], V_bars[e], dt),
                    ported_pdsb_gradient(s, kappas[e], L0s[e], h_bars[e], thetas[e], V_bars[e], dt),
                    ported_pdsb_hessian(s, kappas[e], L0s[e], h_bars[e], thetas[e], V_bars[e], dt)};
            }
            return ElementResultN<4>{
                ported_spdsb_energy(s, kappas[e], L0s[e], h_bars[e], thetas[e], exp_yield[e], V_bars[e], dt),
                ported_spdsb_gradient(s, kappas[e], L0s[e], h_bars[e], thetas[e], exp_yield[e], V_bars[e], dt),
                ported_spdsb_hessian(s, kappas[e], L0s[e], h_bars[e], thetas[e], exp_yield[e], V_bars[e], dt)};
        };

        // ---- 3) drive the real kernels ----------------------------------------
        gg::FiniteElementElastics::Impl elastics_impl;
        elastics_impl.constitution_energy_offsets_counts.resize(1);
        elastics_impl.constitution_energy_offsets_counts.counts()[0] = static_cast<IndexT>(n);
        elastics_impl.constitution_energy_offsets_counts.scan();
        elastics_impl.constitution_gradient_offsets_counts.resize(1);
        elastics_impl.constitution_gradient_offsets_counts.counts()[0] =
            static_cast<IndexT>(4 * n);
        elastics_impl.constitution_gradient_offsets_counts.scan();
        elastics_impl.constitution_hessian_offsets_counts.resize(1);
        elastics_impl.constitution_hessian_offsets_counts.counts()[0] =
            static_cast<IndexT>(10 * n);
        elastics_impl.constitution_hessian_offsets_counts.scan();

        auto          slice = SceneSlice<4>{};
        const int32_t vertex_count = static_cast<int32_t>(base_info.xs().size());
        slice.init(n);

        {
            gg::FiniteElementElastics::ComputeEnergyInfo base_e{
                &elastics_impl, 0, dt, slice.energies.view()};
            gg::FiniteElementExtraConstitution::ComputeEnergyInfo e_info{
                &extra_impl, &base_e, dt};
            REQUIRE(e_info.energies().size() == n);
            CHECK(e_info.dt() == dt);
            switch(kind)
            {
                case BendingKind::Elastic:
                    FEMConstitutionBProbe::dsb->do_compute_energy(e_info);
                    break;
                case BendingKind::StrainPlastic:
                    FEMConstitutionBProbe::pdsb->do_compute_energy(e_info);
                    break;
                default:
                    FEMConstitutionBProbe::spdsb->do_compute_energy(e_info);
                    break;
            }
        }

        const auto dev_e = read_device(slice.energies.cview());
        // energy only: the doublet/triplet buffers are still untouched (empty vectors
        // tell the checker to skip them)
        check_assembly_oracle<4>(n, elems, dev_e, {}, {}, {}, {}, {}, false, elem_fn, 1e-3, name);

        {
            auto g_view = slice.doublet_view();
            auto h_view = slice.triplet_view(vertex_count);
            gg::FiniteElementElastics::ComputeGradientHessianInfo base_gh{
                &elastics_impl, 0, false, dt, g_view, h_view};
            gg::FiniteElementExtraConstitution::ComputeGradientHessianInfo gh_info{
                &extra_impl, &base_gh, dt};
            REQUIRE(gh_info.gradients().doublet_count() == static_cast<int32_t>(4 * n));
            CHECK(gh_info.gradient_only() == false);
            switch(kind)
            {
                case BendingKind::Elastic:
                    FEMConstitutionBProbe::dsb->do_compute_gradient_hessian(gh_info);
                    break;
                case BendingKind::StrainPlastic:
                    FEMConstitutionBProbe::pdsb->do_compute_gradient_hessian(gh_info);
                    break;
                default:
                    FEMConstitutionBProbe::spdsb->do_compute_gradient_hessian(gh_info);
                    break;
            }
        }

        const auto dev_dv = read_device(slice.d_values.cview());
        const auto dev_tv = read_device(slice.t_values.cview());
        check_assembly_oracle<4>(n,
                                 elems,
                                 dev_e,
                                 read_device(slice.d_indices.cview()),
                                 dev_dv,
                                 read_device(slice.t_rows.cview()),
                                 read_device(slice.t_cols.cview()),
                                 dev_tv,
                                 /*has_hessian=*/true,
                                 elem_fn,
                                 1e-3,
                                 name);

        // ---- 4) the `gradient_only` pass --------------------------------------
        //
        //  NOTE (pinned/reported): the *elastics* turns the Hessian view into an empty
        //  one for a gradient-only pass (`ComputeGradientHessianInfo::hessians()`), and
        //  the port then hands the doublet buffers over as placeholders -- so the
        //  sentinel assertion below is only meaningful for the dim-2 constitutions
        //  (whose info carries the caller's views).  For the extras the observable is
        //  that the *gradients* come back bit-identical to the full pass (a stray
        //  Hessian write would land in the placeholder = doublet buffers and show up
        //  here).
        {
            auto slice2 = SceneSlice<4>{};
            slice2.init(n);
            auto g_only = slice2.doublet_view();
            auto h_only = slice2.triplet_view(vertex_count);
            gg::FiniteElementElastics::ComputeGradientHessianInfo base_gh2{
                &elastics_impl, 0, /*gradient_only=*/true, dt, g_only, h_only};
            gg::FiniteElementExtraConstitution::ComputeGradientHessianInfo gh2{
                &extra_impl, &base_gh2, dt};
            CHECK(gh2.gradient_only() == true);
            switch(kind)
            {
                case BendingKind::Elastic:
                    FEMConstitutionBProbe::dsb->do_compute_gradient_hessian(gh2);
                    break;
                case BendingKind::StrainPlastic:
                    FEMConstitutionBProbe::pdsb->do_compute_gradient_hessian(gh2);
                    break;
                default:
                    FEMConstitutionBProbe::spdsb->do_compute_gradient_hessian(gh2);
                    break;
            }
            const auto g2 = read_device(slice2.d_values.cview());
            std::fflush(stdout);
            REQUIRE(g2.size() == dev_dv.size());
            double worst = 0.0;
            for(SizeT i = 0; i < g2.size(); ++i)
                worst = std::max(worst, std::abs(g2[i] - dev_dv[i]));
            CHECK(worst == 0.0);
            for(auto row : read_device(slice2.t_rows.cview()))
                CHECK(row == -7);
            for(auto v : read_device(slice2.t_values.cview()))
                CHECK(v == -7.0);
        }
    };

    run_bending(BendingKind::Elastic, 17, "DiscreteShellBending", 5.0_kPa, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0});
    run_bending(BendingKind::StrainPlastic,
                31,
                "StrainPlasticDSB",
                5.0_kPa,
                {0.05, 0.60, 0.10, 0.20},
                {0.10, 0.05, 0.0, 0.0});
    run_bending(BendingKind::StressPlastic,
                32,
                "StressPlasticDSB",
                5.0_kPa,
                {0.05, 0.60, 0.10, 0.20},
                {0.10, 0.05, 0.0, 0.0});

    // =======================================================================
    //  B) the membrane constitutions (dim-2, uids 11 and 819)
    // =======================================================================
    const auto run_membrane = [&](bool baraff_witkin, U64 uid, const char* name)
    {
        const auto patch = make_strip_patch(3);
        S<geometry::GeometrySlot> slot;

        // distinct per-triangle moduli and strain rates
        const std::vector<double> moduli_a = {2.7, 1.4, 3.1};
        const std::vector<double> moduli_b = {1.4, 2.6, 0.9};
        const std::vector<double> rates    = {100.0, 50.0, 200.0};

        auto scene = make_membrane_scene(patch,
                                         baraff_witkin,
                                         fmt::format("t52b_scene_{}", name),
                                         moduli_a,
                                         moduli_b,
                                         rates,
                                         &slot);

        const auto workspace = workspace_path(fmt::format("uipc_luisa_t52b_{}", name));
        Engine     engine{"luisa", workspace};
        World      world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());

        gg::Codim2DConstitution* sys = baraff_witkin
                                           ? static_cast<gg::Codim2DConstitution*>(
                                                 FEMConstitutionBProbe::bws)
                                           : static_cast<gg::Codim2DConstitution*>(
                                                 FEMConstitutionBProbe::nh);
        if(sys == nullptr)
        {
            WARN(fmt::format("the {} constitution was not built for this scene", name));
            return;
        }
        CHECK(sys->uid() == uid);
        CHECK(sys->dim() == 2);

        const SizeT n = baraff_witkin ? FEMConstitutionBProbe::bws->h_mus.size()
                                      : FEMConstitutionBProbe::nh->h_mus.size();
        const SizeT expected = 2 * patch.strip_count;  // two triangles per quad
        if(n != expected)
        {
            WARN(fmt::format("{} (uid {}) did NOT own its (dim, uid) geometry group in this "
                             "binary: luisa_fem_base_test.cpp's mock constitution claims "
                             "uid 11 first.  Run the filtered case to exercise it end to "
                             "end.",
                             name,
                             uid));
            std::printf("[t52b] SKIPPED %s end-to-end checks (group owned elsewhere:"
                        " %llu of %llu elements)\n",
                        name,
                        static_cast<unsigned long long>(n),
                        static_cast<unsigned long long>(expected));
            return;
        }

        // ---- the method's element data -----------------------------------------
        gg::Codim2DConstitution::BaseInfo probe{sys, 0, dt};
        const auto elems = read_device(probe.indices());
        const auto xs    = read_device(probe.xs());
        const auto xbars = read_device(probe.x_bars());
        const auto areas = read_device(probe.rest_areas());
        const auto thick = read_device(probe.thicknesses());
        REQUIRE(elems.size() == n);

        const auto IBs = read_device(baraff_witkin
                                         ? FEMConstitutionBProbe::bws->inv_B_matrices.cview()
                                         : FEMConstitutionBProbe::nh->inv_B_matrices.cview());
        REQUIRE(IBs.size() == n);

        std::vector<Vector3> host_xs, host_xbars;
        for(auto& x : xs)
            host_xs.push_back(gg::to_eigen(x));
        for(auto& x : xbars)
            host_xbars.push_back(gg::to_eigen(x));

        std::vector<std::array<int, 3>> elem_idx(n);
        for(SizeT e = 0; e < n; ++e)
        {
            const Vector3i t = gg::to_eigen(elems[e]);
            for(int k = 0; k < 3; ++k)
                elem_idx[e][k] = t(k);
        }

        // ---- `do_init`: the per-triangle parameters and the rest metric --------
        const auto h_lambdas = baraff_witkin ? FEMConstitutionBProbe::bws->h_lambdas
                                             : FEMConstitutionBProbe::nh->h_lambdas;
        const auto h_mus = baraff_witkin ? FEMConstitutionBProbe::bws->h_mus
                                         : FEMConstitutionBProbe::nh->h_mus;
        std::vector<double> h_rates;
        if(baraff_witkin)
            h_rates.assign(FEMConstitutionBProbe::bws->h_strain_rates.begin(),
                           FEMConstitutionBProbe::bws->h_strain_rates.end());

        double worst_ib = 0.0, worst_param = 0.0;
        for(SizeT e = 0; e < n; ++e)
        {
            Stencil s;
            s.X0 = s.x0 = host_xbars[elem_idx[e][0]];
            s.X1 = s.x1 = host_xbars[elem_idx[e][1]];
            s.X2 = s.x2 = host_xbars[elem_idx[e][2]];

            // the rest metric, recomputed independently (the Neo-Hookean `A(X)` /
            // Baraff-Witkin `Dm2x2`)
            const auto IB_oracle = baraff_witkin ? bws_IB(s) : nh_IB(s);
            worst_ib = std::max(worst_ib, rel_err(max_abs(to_eig(IBs[e]) - IB_oracle), 0.0));

            // the per-triangle moduli the frontend wrote (cycled from the vectors above)
            worst_param = std::max(worst_param,
                                   rel_err(h_lambdas[e], moduli_a[e % moduli_a.size()]));
            worst_param = std::max(worst_param,
                                   rel_err(h_mus[e], moduli_b[e % moduli_b.size()]));
            if(baraff_witkin)
                worst_param = std::max(worst_param,
                                       rel_err(h_rates[e], rates[e % rates.size()]));
        }
        std::printf("[t52b] %s do_init: rest metric vs the Eigen oracle %.3e, parameters"
                    " vs the scene attributes %.3e\n",
                    name,
                    worst_ib,
                    worst_param);
        CHECK(worst_ib <= 1e-13);
        CHECK(worst_param <= 1e-14);

        // ---- drive the real kernels --------------------------------------------
        auto slice = SceneSlice<3>{};
        slice.init(n);
        const int32_t vertex_count = static_cast<int32_t>(probe.xs().size());

        auto elem_fn = [&](SizeT e)
        {
            Stencil s;
            s.x0 = host_xs[elem_idx[e][0]];
            s.x1 = host_xs[elem_idx[e][1]];
            s.x2 = host_xs[elem_idx[e][2]];

            // The reference's `triangle_thickness(t0, t1, t2)` returns the *first*
            // vertex's thickness (its kernel asserts that the three agree; the
            // frontend writes the attribute uniformly).  It is *not* an average, and
            // every membrane energy is proportional to the thickness: an oracle that
            // averages the three vertices is off by `avg / t0`.
            const double t0 = thick[elem_idx[e][0]];
            CHECK(rel_err(thick[elem_idx[e][1]], t0) <= 1e-15);
            CHECK(rel_err(thick[elem_idx[e][2]], t0) <= 1e-15);
            const double thickness = t0;

            if(baraff_witkin)
            {
                return ElementResultN<3>{
                    ported_bws_energy(
                        s, h_lambdas[e], h_mus[e], h_rates[e], to_eig(IBs[e]), areas[e], dt),
                    ported_bws_gradient(
                        s, h_lambdas[e], h_mus[e], h_rates[e], to_eig(IBs[e]), areas[e], dt),
                    ported_bws_hessian(
                        s, h_lambdas[e], h_mus[e], h_rates[e], to_eig(IBs[e]), areas[e], dt)};
            }
            return ElementResultN<3>{
                ported_nh_energy(
                    s, h_lambdas[e], h_mus[e], to_eig(IBs[e]), areas[e], thickness, dt),
                ported_nh_gradient(
                    s, h_lambdas[e], h_mus[e], to_eig(IBs[e]), areas[e], thickness, dt),
                ported_nh_hessian(
                    s, h_lambdas[e], h_mus[e], to_eig(IBs[e]), areas[e], thickness, dt)};
        };

        {
            gg::Codim2DConstitution::ComputeEnergyInfo e_info{sys, 0, dt, slice.energies.view()};
            REQUIRE(e_info.indices().size() == n);
            CHECK(e_info.dt() == dt);
            if(baraff_witkin)
                FEMConstitutionBProbe::bws->do_compute_energy(e_info);
            else
                FEMConstitutionBProbe::nh->do_compute_energy(e_info);
        }

        const auto dev_e = read_device(slice.energies.cview());
    check_assembly_oracle<3>(
            n, elem_idx, dev_e, {}, {}, {}, {}, {}, false, elem_fn, 1e-6, name);

        {
            auto g_view = slice.doublet_view();
            auto h_view = slice.triplet_view(vertex_count);
            gg::Codim2DConstitution::ComputeGradientHessianInfo gh_info{
                sys, 0, false, dt, g_view, h_view};
            REQUIRE(gh_info.indices().size() == n);
            if(baraff_witkin)
                FEMConstitutionBProbe::bws->do_compute_gradient_hessian(gh_info);
            else
                FEMConstitutionBProbe::nh->do_compute_gradient_hessian(gh_info);
        }

        const auto dev_dv = read_device(slice.d_values.cview());
        const auto dev_tv = read_device(slice.t_values.cview());
        check_assembly_oracle<3>(n,
                                 elem_idx,
                                 dev_e,
                                 read_device(slice.d_indices.cview()),
                                 dev_dv,
                                 read_device(slice.t_rows.cview()),
                                 read_device(slice.t_cols.cview()),
                                 dev_tv,
                                 /*has_hessian=*/true,
                                 elem_fn,
                                 1e-3,
                                 name);

        // ---- `gradient_only` must not touch the Hessian buffers ----------------
        {
            auto slice2 = SceneSlice<3>{};
            slice2.init(n);
            auto g_only = slice2.doublet_view();
            auto h_only = slice2.triplet_view(vertex_count);
            gg::Codim2DConstitution::ComputeGradientHessianInfo gh2{
                sys, 0, true, dt, g_only, h_only};
            if(baraff_witkin)
                FEMConstitutionBProbe::bws->do_compute_gradient_hessian(gh2);
            else
                FEMConstitutionBProbe::nh->do_compute_gradient_hessian(gh2);

            const auto g2 = read_device(slice2.d_values.cview());
            REQUIRE(g2.size() == dev_dv.size());
            double worst = 0.0;
            for(SizeT i = 0; i < g2.size(); ++i)
                worst = std::max(worst, std::abs(g2[i] - dev_dv[i]));
            CHECK(worst == 0.0);
            for(auto row : read_device(slice2.t_rows.cview()))
                CHECK(row == -7);
            for(auto v : read_device(slice2.t_values.cview()))
                CHECK(v == -7.0);
        }
    };

    run_membrane(/*baraff_witkin=*/false, 11, "NeoHookeanShell2D");
    run_membrane(/*baraff_witkin=*/true, 819, "StrainLimitingBaraffWitkin");
}
