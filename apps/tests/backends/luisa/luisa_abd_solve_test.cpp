/********************************************************************
 * @file luisa_abd_solve_test.cpp
 * @brief T6.4a — the ABD *solve plumbing* of the LuisaCompute backend: the
 *        Newton-tolerance subsystem (manager + checker base + the
 *        max-translation checker), the ABD block-diagonal preconditioner, the ABD
 *        tolerance checker and the completed ABD line-search reporter.
 *
 * Every oracle here is derived independently *in this file* (the reference backend
 * is CUDA-only and cannot be host-included, so nothing is copied from it).  The
 * oracles are
 *  (a) **hand-derived matrix identities** for the 12x12 inverse that T6.4a added to
 *      the numerics layer (`lps::inverse` for `N > 4` = the reference's
 *      `cuda_tool::eigen::GaussEliminationInverse`, cookbook §7.21 rule 11):
 *      a scaled identity plus a rank-1 update has a closed-form inverse
 *      (Sherman-Morrison, written out below), a diagonal block inverts entrywise,
 *      `I + c E01` inverts to `I - c E01` **exactly** (the element is nilpotent), a
 *      row-swap permutation inverts to itself **bitwise** and is the *only* input
 *      that exercises the reference's "add the first row below into row i" pivot
 *      step, and the singular/under-`eps` inputs pin what the reference *does*
 *      (its two `return result;` statements return an **uninitialized** Eigen
 *      matrix; the port's `LMat` is zero-initialised, so the port's answer is the
 *      deterministic zero matrix — asserted here);
 *  (b) **Eigen** (`Eigen::Matrix<double,12,12>::inverse()`, i.e. an LU with partial
 *      pivoting — a different algorithm from the ported Gauss-Jordan) for the
 *      random SPD blocks, the assembled `diag_hessian` read back from the device
 *      and the preconditioner's `z = D^-1 r`;
 *  (c) the **reference's own laws** for the tolerance checker: the nine rotational
 *      components with a strict `>`, the three translational ones ignored, and the
 *      `success` flag's "already failed stays failed" fold — each written out as a
 *      plain host loop over random `dq` batches;
 *  (d) the **manager/checker contract**: `pre_newton(frame)` -> every checker,
 *      `check()` -> every checker *until the first failure* (the short-circuit is
 *      observable: a checker after the failing one keeps its previous state), the
 *      `CheckResultInfo`'s iteration count, the `ResultInfo::converged()` OR, and
 *      the real `MaxTranslationChecker`'s `res <= abs_tol` comparison with both the
 *      absolute and the scene-relative (`velocity_tol_relative * scene_diagonal *
 *      dt`, the Stiff-GIPC convention) tolerance — the latter driven directly
 *      (the class's hooks are public, see `max_translation_checker.h`);
 *  (e) a **real scene** for the line-search reporter: its `record_start_point` /
 *      `step_forward` / `compute_energy` bodies driven through their public hooks
 *      (which is what the `LineSearcher` does), with the `dq` the *shipped*
 *      `ABDLinearSubsystem::retrieve_solution` shader produces from a hand-written
 *      global solution vector (`GlobalLinearSystem::Impl::distribute_solution`) —
 *      so `q = q_temp + alpha * dq` is asserted against the same hand-written
 *      numbers, the fixed-body guard included — plus a **test-local subreporter**
 *      that makes the reporter's energy fan-out a *differenced* statement
 *      (`total(with the mock) - total(without it) == the mock's own values`);
 *  (f) a **device round** for every new generic routine: the tolerance row (a
 *      test-local kernel + the shipped guards, compared against the host
 *      instantiation of the *same* factored routine over random `dq`, including the
 *      `success` flag's atomic exchange) and the 12x12 inverse
 *      (`lps::inverse` — the device instantiation is the *shipped* preconditioner
 *      shader, whose `diag_inv` output is compared against the host instantiation
 *      on the same input; no test-local 12x12 kernel is written on purpose:
 *      cookbook §7.27 rule 2's `Device::compile` deadlock is a property of big
 *      unrolled 12x12 chains in *test* kernels).
 *
 * Coverage map (case -> oracle):
 *  1) the 12x12 generic inverse, host ........................ abd_solve_inverse12
 *  2) the assembled ABD preconditioner through a standalone
 * `GlobalLinearSystem::Impl` (assemble -> Eigen identity -> apply ->
 * the `converged != 0` early-out -> the copy fallback) ... abd_solve_diag_apply
 *  3) the tolerance checker's boundary/atomic laws, host ...... abd_solve_tolerance_boundary
 *  4) the tolerance row device round .......................... abd_solve_tolerance_device
 *  5) the manager/checker contract + max_translation + the ABD
 * checker's convergence flip on a real scene ................ abd_solve_manager
 *  6) the line-search reporter's bodies + registration + the
 * subreporter fan-out on a real scene ....................... abd_solve_line_search
 *  7) every new generic routine on the device vs the host ..... abd_solve_device_vs_host
 *  8) per-case assertion accounting ........................... abd_solve_summary
 *
 * Shared-binary rules (cookbook §7.14 rule 6, §7.18 rule 6, §7.25 rule 18): the
 * systems the port registers are built for *every* scene of the whole test binary,
 * so the probe below only reads them and the test-local checkers/subreporter are
 * **inert unless this file's case activates them** (a static knob, cleared by the
 * case) — nothing here can change another file's numbers.
 *
 * WHAT THIS FILE CANNOT REACH (stated so nothing is implied)
 * ----------------------------------------------------------
 * The Newton loop itself (`SimEngine::pre_newton`/`check` call sites) and the line
 * search's *iteration* (the `LineSearcher::do_line_search` loop, the energy
 * comparison and the rollback) are Phase 8: this batch wires the systems into
 * `SimEngine` (the tolerance manager's `init()`) but the engine's `advance()` still
 * runs the prologue/epilogue only.  `ResultInfo::frame()`/`newton_iter()` are
 * *inputs* the engine fills, so they are set by the test.  The preconditioner is
 * never reached from `solve()` (T8.3) either: it is driven through the standalone
 * `GlobalLinearSystem::Impl`, exactly like T5.4b's FEM sibling.
 ********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_fixed_joint.h>

#include <affine_body/type_define.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/abd_diag_preconditioner.h>
#include <affine_body/abd_diag_preconditioner_function.h>
#include <affine_body/abd_tolerance_checker.h>
#include <affine_body/abd_tolerance_checker_function.h>
#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <newton_tolerance/newton_tolerance_manager.h>
#include <newton_tolerance/newton_tolerance_checker.h>
#include <newton_tolerance/max_translation_checker.h>
#include <linear_system/global_linear_system.h>
#include <line_search/line_searcher.h>
#include <line_search/line_search_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>
#include <luisa_device/math/matrix_ops.h>
#include <luisa_device/atomic.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
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

namespace t64a
{
// ---------------------------------------------------------------------------
//  small helpers (the house style of the sibling ABD test files)
// ---------------------------------------------------------------------------
using Dense3   = Eigen::Matrix<double, 3, 3>;
using Dense4   = Eigen::Matrix<double, 4, 4>;
using Dense12  = Eigen::Matrix<double, 12, 12>;
using Dense12v = Eigen::Matrix<double, 12, 1>;
using Dense24  = Eigen::Matrix<double, 24, 24>;
using Dense24v = Eigen::Matrix<double, 24, 1>;

[[nodiscard]] inline double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

template <typename Derived>
[[nodiscard]] inline double max_abs(const Eigen::MatrixBase<Derived>& m)
{
    return m.cwiseAbs().maxCoeff();
}

/// `max |A - B| / max(1, |A|_max, |B|_max)` -- the scale-free deviation the
/// FEM/ABD rows calibrate against (§7.17 rule 14's `dev()`).
template <typename A, typename B>
[[nodiscard]] inline double dev(const A& a, const B& b)
{
    const double scale = std::max({1.0, max_abs(a), max_abs(b)});
    return max_abs(a - b) / scale;
}

/// Eigen -> the port's container (the seam every oracle crosses).
template <int N>
[[nodiscard]] inline lps::LMat<double, N, N> to_lmat_n(const Eigen::Matrix<double, N, N>& m)
{
    lps::LMat<double, N, N> out;
    for(int c = 0; c < N; ++c)
        for(int r = 0; r < N; ++r)
            out(r, c) = m(r, c);
    return out;
}

template <int N>
[[nodiscard]] inline Eigen::Matrix<double, N, N> to_eigen_n(const lps::LMat<double, N, N>& m)
{
    Eigen::Matrix<double, N, N> out;
    for(int c = 0; c < N; ++c)
        for(int r = 0; r < N; ++r)
            out(r, c) = m(r, c);
    return out;
}

/// A fixed, well-conditioned symmetric 3x3 and 4x4 (the `N <= 4` analytic path).
[[nodiscard]] inline Dense3 spd3() noexcept
{
    Dense3 A;
    A << 2.0, 0.5, -0.25, 0.5, 3.0, 0.75, -0.25, 0.75, 1.5;
    return A;
}

[[nodiscard]] inline Dense4 spd4() noexcept
{
    Dense4 A = Dense4::Zero();
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            A(i, j) = 0.25 * static_cast<double>((i + 1) * (j + 2)) - 0.5;
    Dense4 S = A + A.transpose();
    for(int i = 0; i < 4; ++i)
        S(i, i) += 12.0;
    return S;
}

/// The 12x12 container the *buffers* hold (`gg::LMatrix12x12` == `lps::LMat<double,12,12>`).
[[nodiscard]] inline gg::LMatrix12x12 to_lm12(const Dense12& m)
{
    gg::LMatrix12x12 out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] inline lps::LMat<double, 12, 12> to_lmat(const Dense12& m)
{
    lps::LMat<double, 12, 12> out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] inline Dense12 to_eigen(const lps::LMat<double, 12, 12>& m)
{
    Dense12 out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m(r, c);
    return out;
}

// ---------------------------------------------------------------------------
//  the hand-derived 12x12 blocks of case 1
// ---------------------------------------------------------------------------
/// `M = scale * I + off * u u^T` with `u = (1, ..., 12)`.
///
/// **Hand-derived inverse** (Sherman-Morrison, written out here so the oracle
/// shares nothing with the ported Gauss-Jordan):
///     M^-1 = I / scale - (off / scale^2) * u u^T / (1 + (off / scale) * u.u)
[[nodiscard]] inline Dense12 rank1_block(double scale, double off)
{
    // NOTE: the vector is `Dense12v` (12x1), NOT a `Dense12` with one filled
    // column -- an Eigen matrix's default constructor does not zero, so the wider
    // spelling silently carries garbage into `u * u^T` (cookbook §7.25 rule 17a:
    // this file hit it and the oracle returned NaN).
    Dense12v u;
    for(int i = 0; i < 12; ++i)
        u(i) = static_cast<double>(i + 1);
    Dense12 M = scale * Dense12::Identity() + off * (u * u.transpose());
    return M;
}

[[nodiscard]] inline Dense12 rank1_block_inverse(double scale, double off)
{
    Dense12v u;
    for(int i = 0; i < 12; ++i)
        u(i) = static_cast<double>(i + 1);
    const double uu = u.dot(u);
    Dense12     out = Dense12::Identity() / scale
                  - (off / (scale * scale)) * (u * u.transpose()) / (1.0 + (off / scale) * uu);
    return out;
}

/// `A = I + c * E01` (`E01` = the matrix with a single 1 at (0, 1)): `E01^2 == 0`,
/// so `A^-1 = I - c * E01` **exactly** (every operation in the reference's sweep is
/// an add/subtract/multiply by 1).
[[nodiscard]] inline Dense12 nilpotent_block(double c)
{
    Dense12 A        = Dense12::Identity();
    A(0, 1)          = c;
    return A;
}

/// The 12x12 row-swap permutation (rows 0 and 1 exchanged).  `A^-1 == A` exactly,
/// and `A(0,0) == 0` is the *only* shape in this file that makes the reference's
/// "if |mat[i][i]| < eps: add the first row below with |mat[j][i]| > eps" step fire.
[[nodiscard]] inline Dense12 swap_block()
{
    Dense12 A = Dense12::Identity();
    A(0, 0) = 0.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 0.0;
    return A;
}

/// A genuinely singular block: row 11 is a copy of row 0.
[[nodiscard]] inline Dense12 singular_block()
{
    Dense12 A = Dense12::Identity();
    for(int c = 0; c < 12; ++c)
        A(11, c) = A(0, c);
    return A;
}

/// A diagonal block whose (0,0) entry is below `epsilon<double>` -- the reference's
/// pivot test declares it singular even though the matrix is invertible.
[[nodiscard]] inline Dense12 tiny_pivot_block(double tiny)
{
    Dense12 A  = Dense12::Identity();
    A(0, 0)    = tiny;
    return A;
}

/// A *near*-singular but well-pivoted diagonal block (all entries above `eps`).
[[nodiscard]] inline Dense12 ill_conditioned_block(double tiny)
{
    Dense12 A = Dense12::Identity();
    A(0, 0) = tiny;
    A(1, 1) = 3.7;
    A(2, 2) = -1.25;
    return A;
}

/// A random SPD block (deterministic: the eigensystem is fixed by the seed).
/// `Q diag(1 .. cond) Q^T` with `Q` the Q of a Householder QR of a random matrix
/// (materialised into an `Eigen::Matrix`, so no lazy expression leaks out).
[[nodiscard]] inline Dense12 random_spd(unsigned seed, double cond = 4.0)
{
    std::mt19937                           rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    Eigen::Matrix<double, 12, 12>          R;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            R(r, c) = d(rng);
    Eigen::HouseholderQR<Eigen::Matrix<double, 12, 12>> qr(R);
    const Dense12                                       Q = qr.householderQ() * Dense12::Identity();
    Dense12                                             D = Dense12::Zero();
    for(int i = 0; i < 12; ++i)
        D(i, i) = 1.0 + (cond - 1.0) * static_cast<double>(i) / 11.0;
    return Q * D * Q.transpose();
}

// ---------------------------------------------------------------------------
//  case accounting (the sibling files' T63Bx_CHECK/CaseScope idiom)
// ---------------------------------------------------------------------------
std::vector<std::pair<std::string, int>> g_case_counts;

struct CaseScope
{
    std::string name;
    int         checks = 0;

    explicit CaseScope(std::string n)
        : name(std::move(n))
    {
    }
    ~CaseScope() { g_case_counts.emplace_back(name, checks); }
};

#define T64A_CHECK(counter, cond) \
    do \
    { \
        ++(counter).checks; \
        CHECK(cond); \
    } while(false)

// ---------------------------------------------------------------------------
//  the scene: two ABD cubes (the T6.1b geometry), optionally with one fixed body
//  and one fixed joint
// ---------------------------------------------------------------------------
constexpr Float kL    = 0.5;
constexpr Float kRho  = 1000.0;
constexpr Float kKappa = 1.0e8;

/// The six-tet decomposition of `[0, L]^3` (the T6.1b/§7.22 rule 12 geometry:
/// every rest determinant is positive, which `AffineBodyConstitution::apply_to`'s
/// density overload requires).
[[nodiscard]] inline std::array<Vector4i, 6> cube_tets() noexcept
{
    return {{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}}};
}

[[nodiscard]] inline std::array<Vector3, 8> cube_vertices(float L) noexcept
{
    std::array<Vector3, 8> v;
    for(int i = 0; i < 8; ++i)
    {
        const double x = (i & 1) ? L : 0.0;
        const double y = (i & 2) ? L : 0.0;
        const double z = (i & 4) ? L : 0.0;
        v[i]           = Vector3{x, y, z};
    }
    return v;
}

struct SceneSpec
{
    Float kappa         = kKappa;
    Float kappa_body1   = -1.0;  // < 0: same as `kappa`
    bool  body0_fixed   = false;
    bool  with_joint    = false;
    Float velocity_tol          = -1.0;  // < 0: leave the default config value
    Float velocity_tol_relative = 0.0;
    Float transrate_tol         = -1.0;
    Float dt                    = 0.01;
};

/// Two ABD cubes (each 8 vertices / 1 instance, identity transform) plus, when
/// `with_joint`, one frontend `AffineBodyFixedJoint` between them -- which is what
/// makes the *InterAffineBody* constitution type appear in the scene and therefore
/// the inter-body manager (and its line-search subreporter) exist.
[[nodiscard]] inline Scene build_scene(const SceneSpec& spec)
{
    Json config = Scene::default_config();
    config["contact"]["enable"]  = false;
    config["sanity_check"]["enable"] = false;
    config["dt"]                         = spec.dt;
    if(spec.velocity_tol >= 0.0)
        config["newton"]["velocity_tol"] = spec.velocity_tol;
    config["newton"]["velocity_tol_relative"] = spec.velocity_tol_relative;
    if(spec.transrate_tol >= 0.0)
        config["newton"]["transrate_tol"] = spec.transrate_tol;

    Scene scene{config};
    auto  object = scene.objects().create("t64a_solve");

    std::vector<S<geometry::SimplicialComplexSlot>> slots;
    for(SizeT b = 0; b < 2; ++b)
    {
        const auto vs = cube_vertices(kL);
        const auto ts = cube_tets();
        auto       sc = tetmesh(span<const Vector3>{vs.data(), vs.size()},
                                span<const Vector4i>{ts.data(), ts.size()});
        label_surface(sc);
        sc.instances().resize(1);
        AffineBodyConstitution abd;
        const Float kappa_b = (spec.kappa_body1 > 0.0 && b == 1) ? spec.kappa_body1
                                                                 : spec.kappa;
        abd.apply_to(sc, kappa_b, kRho);
        view(sc.transforms())[0] = Matrix4x4::Identity();
        {
            auto is_fixed = sc.instances().find<IndexT>(builtin::is_fixed);
            UIPC_ASSERT(is_fixed, "t64a: apply_to must create instances/is_fixed");
            view(*is_fixed)[0] = (b == 0 && spec.body0_fixed) ? 1 : 0;
        }
        auto [slot, rest] = object->geometries().create(sc);
        slots.push_back(slot);
        (void)rest;
    }

    if(spec.with_joint)
    {
        AffineBodyFixedJoint joint;
        std::vector<IndexT>   l_ids{0};
        std::vector<IndexT>   r_ids{0};
        std::vector<Float>    ratios{100.0};
        std::vector<S<geometry::SimplicialComplexSlot>> l_slots{slots[0]};
        std::vector<S<geometry::SimplicialComplexSlot>> r_slots{slots[1]};
        auto joint_geo = joint.create_geometry(span<S<geometry::SimplicialComplexSlot>>(l_slots),
                                               span<IndexT>(l_ids),
                                               span<S<geometry::SimplicialComplexSlot>>(r_slots),
                                               span<IndexT>(r_ids),
                                               span<Float>(ratios));
        object->geometries().create(joint_geo);
    }
    return scene;
}

void step_frame(World& world)
{
    world.advance();
    world.sync();
    gg::frame_stream().synchronize();
}

[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// Read an LC buffer back to the host (frame stream first: a command may still be
/// recorded in an open frame).
template <typename T>
[[nodiscard]] std::vector<T> read_device(const lud::LCBufferView<T>& view)
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
//  the probe: grabs the built systems once they exist (the T4.1/T6.1b idiom)
// ---------------------------------------------------------------------------
class T64aProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::AffineBodyDynamics*        abd = nullptr;
    inline static gg::ABDLinearSubsystem*        abd_linear = nullptr;
    inline static gg::ABDDiagPreconditioner*     precond = nullptr;
    inline static gg::ABDToleranceChecker*       abd_tol = nullptr;
    inline static gg::MaxTranslationChecker*     max_tol = nullptr;
    inline static gg::NewtonToleranceManager*    tol_manager = nullptr;
    inline static gg::ABDLineSearchReporter*     ls_reporter = nullptr;
    inline static gg::GlobalLinearSystem*        gls = nullptr;
    inline static gg::GlobalVertexManager*       gvm = nullptr;
    inline static gg::LineSearcher*              line_searcher = nullptr;

    static void reset()
    {
        abd           = nullptr;
        abd_linear    = nullptr;
        precond       = nullptr;
        abd_tol       = nullptr;
        max_tol       = nullptr;
        tol_manager   = nullptr;
        ls_reporter   = nullptr;
        gls           = nullptr;
        gvm           = nullptr;
        line_searcher = nullptr;
    }

  private:
    void do_build() override
    {
        abd           = &require<gg::AffineBodyDynamics>();
        gvm           = &require<gg::GlobalVertexManager>();
        gls           = &require<gg::GlobalLinearSystem>();
        line_searcher = &require<gg::LineSearcher>();
        abd_linear    = find<gg::ABDLinearSubsystem>();
        precond       = find<gg::ABDDiagPreconditioner>();
        abd_tol       = find<gg::ABDToleranceChecker>();
        max_tol       = find<gg::MaxTranslationChecker>();
        tol_manager   = find<gg::NewtonToleranceManager>();
        ls_reporter   = find<gg::ABDLineSearchReporter>();
    }
};

// ---------------------------------------------------------------------------
//  the test-local Newton tolerance checkers: they exist only to *observe* the
//  manager's fan-out (the order, the frame, the short-circuit).  Both are inert
//  for every other test file: the manager only calls them when a test drives
//  `check()`, and no other file does (the tolerance manager has no engine call
//  site yet -- the Newton loop is Phase 8).
// ---------------------------------------------------------------------------
struct MockOrder
{
    inline static int serial = 0;
};

class T64aMockCheckerA final : public gg::NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    inline static int   pre_newton_calls = 0;
    inline static SizeT last_frame       = ~0ull;
    inline static int   check_calls      = 0;
    inline static int   order            = -1;
    inline static bool  fail             = false;

    static void reset()
    {
        pre_newton_calls = 0;
        last_frame       = ~0ull;
        check_calls      = 0;
        order            = -1;
        fail             = false;
    }

    void do_build(BuildInfo& info) override { (void)info; }
    void do_init(InitInfo& info) override { (void)info; }

    void do_pre_newton(PreNewtonInfo& info) override
    {
        ++pre_newton_calls;
        last_frame = info.frame();
    }

    void do_check(CheckResultInfo& info) override
    {
        ++check_calls;
        order = MockOrder::serial++;
        info.converged(!fail);
    }

    std::string do_report() override
    {
        // NOTE (T6.4a, cookbook §7.28): calling the base's own default body makes
        // the reference to `NewtonToleranceChecker::do_report` a *used* import.
        // A derived vtable slot alone is resolved from the module's import library
        // WITHOUT creating an import-table entry, which makes
        // `scripts/luisa_module_def.py` (it derives the export list from the
        // consumers' imports) drop the symbol and the next link fail again.
        return std::string{"mockA|"} + NewtonToleranceChecker::do_report();
    }
};

class T64aMockCheckerB final : public gg::NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    inline static int   pre_newton_calls = 0;
    inline static SizeT last_frame       = ~0ull;
    inline static int   check_calls      = 0;
    inline static int   order            = -1;
    inline static bool  fail             = false;

    static void reset()
    {
        pre_newton_calls = 0;
        last_frame       = ~0ull;
        check_calls      = 0;
        order            = -1;
        fail             = false;
    }

    void do_build(BuildInfo& info) override { (void)info; }
    void do_init(InitInfo& info) override { (void)info; }

    void do_pre_newton(PreNewtonInfo& info) override
    {
        ++pre_newton_calls;
        last_frame = info.frame();
    }

    void do_check(CheckResultInfo& info) override
    {
        ++check_calls;
        order = MockOrder::serial++;
        info.converged(!fail);
    }

    std::string do_report() override
    {
        return std::string{"mockB|"} + NewtonToleranceChecker::do_report();
    }
};

// ---------------------------------------------------------------------------
//  the test-local line-search subreporter: it exists so the reporter's energy
///  fan-out becomes a *differenced* statement, and so `Impl::init`'s
///  `R->m_index`/`R->init()` pass (and `report_extent`'s slicing) is observable.
// ---------------------------------------------------------------------------
class T64aMockSubreporter final : public gg::ABDLineSearchSubreporter
{
  public:
    using ABDLineSearchSubreporter::ABDLineSearchSubreporter;

    inline static bool  active        = false;
    inline static int   init_calls    = 0;
    inline static int   extent_calls  = 0;
    inline static int   energy_calls  = 0;
    inline static SizeT last_count    = 0;
    inline static double last_sum     = 0.0;

    static void reset()
    {
        active       = false;
        init_calls   = 0;
        extent_calls = 0;
        energy_calls = 0;
        last_count   = 0;
        last_sum     = 0.0;
    }

    void do_build(BuildInfo& info) override { (void)info; }
    void do_init(InitInfo& info) override
    {
        (void)info;
        ++init_calls;
    }

    void do_report_extent(ReportExtentInfo& info) override
    {
        ++extent_calls;
        info.energy_count(active ? kMockEnergyCount : 0u);
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        ++energy_calls;
        auto energies = info.energies();
        last_count    = energies.size();
        last_sum      = 0.0;
        if(active && energies.size() >= kMockEnergyCount)
        {
            std::array<Float, kMockEnergyCount> values{kMockEnergy0, kMockEnergy1};
            energies.copy_from(span<const Float>{values}, &gg::frame_stream());
            last_sum = kMockEnergy0 + kMockEnergy1;
        }
    }

    inline static constexpr SizeT kMockEnergyCount = 2u;
    inline static constexpr Float kMockEnergy0     = 1000.0;
    inline static constexpr Float kMockEnergy1     = 2000.0;
};

// ---------------------------------------------------------------------------
//  a device-side helper of case 4/7: the ABD tolerance kernel the *shipped*
//  checker uses, transcribed for a test-local batch (one small Vec12 check per
//  thread -- deliberately NOT a 12x12 chain, cookbook §7.27 rule 2).
// ---------------------------------------------------------------------------
auto& t64a_tolerance_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector12>,  // dqs
                                                lcrt::Buffer<std::int32_t>,  // success
                                                lcrt::Buffer<std::int32_t>,  // per-thread flags
                                                double,                      // abs_tol
                                                uint32_t>(                   // n
        [](lcrt::BufferVar<gg::LVector12> dqs,
           lcrt::BufferVar<std::int32_t> success,
           lcrt::BufferVar<std::int32_t> flags,
           lcrt::Var<double>             abs_tol,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto D = gg::abd_tolerance_checker_detail::DqDeviceView::from_handles(dqs, n);

                // the reference's per-thread decision (the guard + the nine-component
                // test), through the *shared* factored routine
                auto success_value = success->read(0);
                auto failed = gg::abd_tolerance_checker_detail::failed_row(
                    D, lcrt::cast<std::int32_t>(I), abs_tol, success_value);

                // record the decision, then the reference's atomic exchange
                // (LC's `select` needs DSL operands on both sides -- the port's
                // `scalar_detail::select_scalar` literals are Exprs here)
                auto flag = lcrt::select(lcrt::Var<std::int32_t>{0},
                                         lcrt::Var<std::int32_t>{1},
                                         failed);
                flags->write(I, flag);
                $if(failed)
                {
                    (void)lud::atomic_exch(success->atomic(0), std::int32_t{0});
                };
            };
        });
    return k;
}
}  // namespace t64a

// The four test-local systems.  They are created in *every* scene of the shared
// test binary (the same rule the port's own systems follow), so each one is inert
// unless `t64a`'s own case turns it on:
//  * the two mock checkers only run when a case drives `NewtonToleranceManager::check`
//    (nothing else does -- the Newton loop is Phase 8),
//  * the mock subreporter reports a zero extent and writes nothing while
//    `T64aMockSubreporter::active` is false.
REGISTER_SIM_SYSTEM(t64a::T64aProbe);
REGISTER_SIM_SYSTEM(t64a::T64aMockCheckerA);
REGISTER_SIM_SYSTEM(t64a::T64aMockCheckerB);
REGISTER_SIM_SYSTEM(t64a::T64aMockSubreporter);

namespace t64a
{
// ---------------------------------------------------------------------------
//  CASE 1 — the 12x12 generic inverse (host)
// ---------------------------------------------------------------------------
//
// The five hand-derived identities of the file header, plus the two inputs that pin
// the reference's *pivot* step and its *singular* behaviour.
TEST_CASE("abd_solve_inverse12", "[luisa][abd_solve]")
{
    CaseScope c{"inverse12"};

    // ---- (a) the analytic N <= 4 path is unchanged by the N > 4 branch ---------
    {
        const Dense3 A   = spd3();
        const auto   inv = to_eigen_n(lps::inverse(to_lmat_n(A)));
        T64A_CHECK(c, dev(inv, A.inverse()) <= 1e-15);
        T64A_CHECK(c, dev(A * inv, Dense3::Identity()) <= 1e-15);
        // the analytic 4x4 path is the *other* explicit branch of the same routine
        const Dense4 B   = spd4();
        const auto   binv = to_eigen_n(lps::inverse(to_lmat_n(B)));
        T64A_CHECK(c, dev(binv, B.inverse()) <= 1e-15);
    }

    // ---- (b) the reference's own 2/3/4 dispatch is untouched for N == 4 --------
    {
        const Dense4 A   = spd4();
        const auto   inv = to_eigen_n(lps::inverse(to_lmat_n(A)));
        T64A_CHECK(c, dev(inv, A.inverse()) <= 1e-15);
        T64A_CHECK(c, dev(A * inv, Dense4::Identity()) <= 1e-14);
    }

    // ---- (c) the hand-derived SPD block (Sherman-Morrison) ---------------------
    {
        const double scale = 3.0, off = 1.7;
        const auto   A     = rank1_block(scale, off);
        const auto   exp   = rank1_block_inverse(scale, off);
        const auto   got   = to_eigen(lps::inverse(to_lmat(A)));
        std::printf("[t64a] inverse12: rank-1 SPD dev %.3e (Eigen %.3e, |A|max %.3e)\n",
                    dev(got, exp),
                    dev(got, A.inverse()),
                    max_abs(A));
        T64A_CHECK(c, dev(got, exp) <= 1e-14);
        T64A_CHECK(c, dev(got, A.inverse()) <= 1e-14);
        T64A_CHECK(c, dev(A * got, Dense12::Identity()) <= 1e-13);
        // non-vacuous: the inverse is far from `I` and the off-diagonal entries are
        // what carries the rank-1 term
        T64A_CHECK(c, max_abs(got - Dense12::Identity()) > 1.0e-2);
        T64A_CHECK(c, std::abs(got(0, 1)) > 1.0e-3);
    }

    // ---- (d) the exact nilpotent block: A^-1 = I - c E01, bitwise --------------
    {
        const double cc  = 0.375;  // a dyadic rational, so every statement is exact
        const auto   A   = nilpotent_block(cc);
        const auto   got = to_eigen(lps::inverse(to_lmat(A)));
        Dense12      exp = Dense12::Identity();
        exp(0, 1)        = -cc;
        T64A_CHECK(c, got(0, 1) == -cc);           // bitwise
        T64A_CHECK(c, max_abs(got - exp) == 0.0);  // bitwise
    }

    // ---- (e) the row-swap block: the *only* input that fires the pivot step ----
    {
        const auto A   = swap_block();
        const auto got = to_eigen(lps::inverse(to_lmat(A)));
        T64A_CHECK(c, A(0, 0) == 0.0);            // the pivot test's precondition
        T64A_CHECK(c, max_abs(got - A) == 0.0);   // A^-1 == A, bitwise
        T64A_CHECK(c, dev(got, A.inverse()) <= 1e-15);
        // the pivot step is load-bearing here: the *no-pivot* answer would be
        // nonsense (and finite), so a broken pivot cannot hide behind a NaN
        T64A_CHECK(c, got.allFinite());
    }

    // ---- (f) a random SPD block vs Eigen (the second, different algorithm) -----
    {
        double worst   = 0.0;
        double worstI  = 0.0;
        for(unsigned seed = 1; seed <= 8; ++seed)
        {
            const auto A     = random_spd(seed, 1.0 + 2.0 * static_cast<double>(seed));
            const auto got   = to_eigen(lps::inverse(to_lmat(A)));
            worst            = std::max(worst, dev(got, A.inverse()));
            worstI           = std::max(worstI, dev(A * got, Dense12::Identity()));
        }
        std::printf("[t64a] inverse12: 8 random SPD blocks, worst dev vs Eigen %.3e, "
                    "worst |A A^-1 - I| %.3e\n",
                    worst,
                    worstI);
        T64A_CHECK(c, worst <= 1e-13);
        T64A_CHECK(c, worstI <= 1e-13);
    }

    // ---- (g) the diagonal identity (exact entrywise) --------------------------
    {
        const auto A = ill_conditioned_block(1.0e-10);
        const auto got = to_eigen(lps::inverse(to_lmat(A)));
        Dense12    exp = Dense12::Zero();
        for(int i = 0; i < 12; ++i)
            exp(i, i) = 1.0 / A(i, i);
        std::printf("[t64a] inverse12: diag(1e-10,3.7,-1.25,1..) dev vs entrywise %.3e\n",
                    dev(got, exp));
        T64A_CHECK(c, dev(got, exp) <= 1e-15);
        T64A_CHECK(c, dev(got, A.inverse()) <= 1e-14);
    }

    // ---- (h) the singular input: the port's deterministic zero matrix ----------
    //
    // The reference's `GaussEliminationInverse` *returns* on a failed pivot search,
    // handing back an **uninitialized** `Eigen::Matrix` (undefined values); the
    // port's `LMat` is zero-initialised (mat.h) and `inverse` masks the whole result
    // when a pivot was missing, so the port's answer is exactly `Zero()`.  This is
    // the "pin what the reference does" case of the brief.
    {
        const auto A   = singular_block();
        const auto got = to_eigen(lps::inverse(to_lmat(A)));
        T64A_CHECK(c, max_abs(got) == 0.0);
        T64A_CHECK(c, got.allFinite());
        // ... and the `eps` threshold is the reference's: a diagonal entry below
        // `epsilon<double>()` is *declared* singular even though the block is
        // invertible.
        const auto tiny = to_eigen(lps::inverse(to_lmat(tiny_pivot_block(1.0e-20))));
        T64A_CHECK(c, max_abs(tiny) == 0.0);
        // one ulp above the threshold is *not* singular (and inverts to 1/tiny)
        const double above_eps = std::numeric_limits<double>::epsilon() * 4.0;
        const auto   big       = to_eigen(lps::inverse(to_lmat(tiny_pivot_block(above_eps))));
        T64A_CHECK(c, big(0, 0) == 1.0 / above_eps);
    }
}

// ---------------------------------------------------------------------------
//  CASE 2 — the assembled ABD preconditioner through a standalone
//           GlobalLinearSystem::Impl (assemble -> apply -> early-out -> fallback)
// ---------------------------------------------------------------------------
TEST_CASE("abd_solve_diag_apply", "[luisa][abd_solve]")
{
    CaseScope c{"diag_apply"};
    test_runtime();

    // this case's own knobs (the mock subreporter must not add energy here)
    T64aMockSubreporter::reset();
    T64aProbe::reset();

    SceneSpec spec;
    // kappa = 1: the *assembled* 12x12 block must be well conditioned for the
    // Eigen-identity oracle to mean anything.  Measured (cookbook §7.28): with the
    // frontend's default kappa = 1e8 the ortho-potential shape Hessian is
    // *numerically singular* at the identity affine state (its rotation modes have
    // no curvature and the stiffness span is ~1e12), so `D D^-1 - I` is O(1) for
    // Eigen and for the ported routine alike; the port still reproduces the
    // reference (it inverts whatever the assembly produced).
    spec.kappa       = 1.0;
    spec.kappa_body1 = 0.5;  // a different block per body: a wrong block is visible
    auto      scene = build_scene(spec);
    Engine    engine{"luisa", workspace_path("uipc_luisa_t64a_diag")};
    World     world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T64aProbe::abd != nullptr);
    REQUIRE(T64aProbe::abd_linear != nullptr);
    REQUIRE(T64aProbe::abd_linear->is_valid());
    REQUIRE(T64aProbe::precond != nullptr);
    // the *real* preconditioner is built in every ABD scene
    CHECK(T64aProbe::precond->is_valid());
    REQUIRE(T64aProbe::gls != nullptr);

    // frame > 0 is required by the assembly's set_dof_info path (the T5.4b note)
    step_frame(world);

    // ---- the standalone Impl: the real subsystem + the real preconditioner -----
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T64aProbe::abd_linear);
    impl.local_preconditioners.register_sim_system(*T64aProbe::precond);
    impl.init();
    T64A_CHECK(c, impl.diag_dof_offsets_counts.total_count() == 24u);
    T64A_CHECK(c, impl.subsystem_infos.size() == 1u);
    T64A_CHECK(c, impl.subsystem_infos[0].is_diag);
    T64A_CHECK(c, impl.no_precond_diag_subsystem_indices.empty());

    // extent -> assemble -> convert (ge2sym + triplet->BCOO) -> _assemble_preconditioner
    impl.build_linear_system();
    T64A_CHECK(c, impl.empty_system == false);
    T64A_CHECK(c, impl.triplet_A.triplet_count() > 0);
    T64A_CHECK(c, impl.bcoo_A.triplet_count() > 0);

    constexpr SizeT kBodyCount = 2u;
    REQUIRE(static_cast<SizeT>(T64aProbe::precond->diag_inv.size()) == kBodyCount);

    // ---- the assembly path: the device's diag_inv vs the host `lps::inverse` ----
    //
    // The *assembled* `diag_hessian` and the *shipped* shader's `diag_inv` are
    // compared against both the host instantiation of the same factored routine
    // (the device-vs-host round of the new 12x12 `lps::inverse` branch) and Eigen's
    // LU inverse (a different algorithm).
    //
    // MEASURED CAVEAT (cookbook §7.28, reported): in this shared test binary
    // `abd().kinetic` is another test file's *slot-owning, marker-gated* kinetic
    // (T6.1b's `zt61b::T61bKineticOwner`, whose demangled name sorts after every
    // `uipc::backend::luisa::` system -- cookbook §7.23 rule 7), so in a scene that
    // does not carry that file's marker **no kinetic term reaches the assembly**:
    // the measured `body_id_to_kinetic_hessian` is identically zero while the dyadic
    // mass is 125.  The assembled `diag_hessian` is therefore the *shape* Hessian
    // alone, which is structurally singular (its translation and rotation directions
    // have no curvature, and the kinetic term is what makes the block SPD).  This
    // case asserts what that leaves well-defined -- the device/host/Eigen agreement
    // *on that input* -- and builds its apply oracle from a hand-written `diag_inv`
    // below, so the preconditioner under test is fully exercised either way.
    const auto diag_hessian = read_device(T64aProbe::abd_linear->diag_hessian());
    const auto diag_inv     = read_device(T64aProbe::precond->diag_inv.cview());
    REQUIRE(diag_hessian.size() == kBodyCount);
    REQUIRE(diag_inv.size() == kBodyCount);

    double worst_host = 0.0, worst_eigen = 0.0, hmax = 0.0, cond_max = 0.0;
    std::vector<Dense12> D(kBodyCount);
    for(SizeT b = 0; b < kBodyCount; ++b)
    {
        D[b]   = to_eigen(diag_hessian[b]);
        hmax   = std::max(hmax, max_abs(D[b]));
        const Dense12 got = to_eigen(diag_inv[b]);
        worst_host        = std::max(worst_host, dev(got, to_eigen(lps::inverse(to_lmat(D[b])))));
        worst_eigen       = std::max(worst_eigen, dev(got, D[b].inverse()));
        Eigen::SelfAdjointEigenSolver<Dense12> es(D[b]);
        cond_max = std::max(cond_max,
                            es.eigenvalues()(11) / std::max(1e-300, es.eigenvalues()(0)));
        // the *device* diag_hessian is symmetric (the assembly's
        // `eigen::atomic_add(diag_hessian(body), H12x12)` of an upper-packed block)
        T64A_CHECK(c, dev(D[b], D[b].transpose()) <= 1e-12);
        // the (t,t) 3x3 sub-block: the shape Hessian has no translation curvature and
        // (in this scene) no kinetic term is added -- the pinned measurement above
        T64A_CHECK(c, max_abs(D[b].block<3, 3>(0, 0)) <= 1e-15);
        // the two bodies are deliberately different, so their blocks differ
        const bool blocks_differ = (b == 0u) || (dev(D[b], D[0]) > 1e-6);
        T64A_CHECK(c, blocks_differ);
    }
    std::printf("[t64a] diag_apply: device diag_inv vs the host lps::inverse %.3e, vs Eigen "
                "%.3e, |diag_hessian|max %.3e, worst cond %.3e, (t,t) block 0\n",
                worst_host,
                worst_eigen,
                hmax,
                cond_max);
    T64A_CHECK(c, hmax > 1e-6);         // the shape Hessian is non-vacuous
    T64A_CHECK(c, worst_host <= 1e-15);  // the shipped shader == the host instantiation
    T64A_CHECK(c, worst_eigen <= 1e-12);  // ... and Eigen agrees with both

    // ---- the apply oracle on a *hand-written* diag_inv -------------------------
    //
    // `diag_inv` is the reference's public `DeviceBuffer<Matrix12x12>`, so the test
    // writes the *hand-derived* inverses of case 1's rank-1 SPD blocks into it and
    // drives the shipped apply shader: `z = diag_inv * r` with a closed-form
    // expectation (no Eigen, no assembled matrix -- the preconditioner's own
    // statement).
    const double scale0 = 3.0, off0 = 1.7;
    const double scale1 = 5.0, off1 = -1.1;  // a *negative* rank-1 term: still SPD
    std::vector<Dense12> Dh(kBodyCount);
    Dh[0] = rank1_block(scale0, off0);
    Dh[1] = rank1_block(scale1, off1);
    std::vector<Dense12> Dh_inv(kBodyCount);
    Dh_inv[0] = rank1_block_inverse(scale0, off0);
    Dh_inv[1] = rank1_block_inverse(scale1, off1);
    {
        std::vector<gg::LMatrix12x12> hand(kBodyCount);
        for(SizeT b = 0; b < kBodyCount; ++b)
            hand[b] = to_lm12(Dh_inv[b]);
        T64aProbe::precond->diag_inv.copy_from(span<const gg::LMatrix12x12>{hand});
        gg::frame_stream().synchronize();
    }

    std::vector<double> r(24, 0.0);
    for(SizeT b = 0; b < kBodyCount; ++b)
        for(int i = 0; i < 12; ++i)
            r[b * 12u + static_cast<SizeT>(i)] =
                0.5 * static_cast<double>(b + 1) + 0.25 * static_cast<double>(i) - 1.0;

    std::vector<double> z_expected(24, 0.0);
    for(SizeT b = 0; b < kBodyCount; ++b)
    {
        Dense12v rv;
        for(int i = 0; i < 12; ++i)
            rv(i) = r[b * 12u + static_cast<SizeT>(i)];
        auto zv = Dh_inv[b] * rv;                            // the hand-derived inverse
        for(int i = 0; i < 12; ++i)
            z_expected[b * 12u + static_cast<SizeT>(i)] = zv(i);
        // the hand inverse really inverts the hand block (a self-check of the oracle;
        // the *negative* rank-1 block has cond ~1.4e2, so the closed form loses ~2
        // digits and the bound is 1e-12 rather than 1e-14)
        T64A_CHECK(c, dev(Dh_inv[b] * Dh[b], Dense12::Identity()) <= 1e-12);
    }

    lla::LDenseVector<double> r_dev, z_dev;
    r_dev.resize(24);
    z_dev.resize(24);
    r_dev.copy_from(span<const double>{r});
    std::vector<double> sentinel(24, -12345.0);
    z_dev.copy_from(span<const double>{sentinel});
    lud::LDeviceVar<IndexT> converged;
    converged.copy_from(IndexT{0});
    gg::frame_stream().synchronize();

    impl.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    // the apply kernel is *recorded* on the frame stream, so the read-back must be
    // ordered after it (the T6.1b/§7.22 rule 7 discipline) -- a plain `copy_to`
    // goes through the main stream and can see the sentinel
    gg::frame_stream().synchronize();
    std::vector<double> z(24, 0.0);
    z_dev.copy_to(span<double>{z});

    double worst_apply = 0.0, zmax = 0.0, identity_gap = 0.0;
    for(SizeT i = 0; i < 24u; ++i)
    {
        worst_apply = std::max(worst_apply, rel_err(z[i], z_expected[i]));
        zmax        = std::max(zmax, std::abs(z_expected[i]));
        identity_gap = std::max(identity_gap, std::abs(r[i] - z_expected[i]));
    }
    std::printf("[t64a] diag_apply: z vs the hand-derived diag_inv * r worst rel %.3e "
                "(max |z| %.3e, |r - diag_inv r|max %.3e)\n",
                worst_apply,
                zmax,
                identity_gap);
    T64A_CHECK(c, worst_apply <= 1e-12);
    T64A_CHECK(c, zmax > 1.0e-2);
    // the preconditioner is *not* the identity -> the comparison above is non-vacuous
    T64A_CHECK(c, identity_gap > 1.0e-2);

    // a wrong *block* cannot hide: the neighbouring body's inverse gives a different z
    {
        double worst_wrong = 0.0;
        for(SizeT b = 0; b < kBodyCount; ++b)
        {
            Dense12v rv;
            for(int i = 0; i < 12; ++i)
                rv(i) = r[b * 12u + static_cast<SizeT>(i)];
            const Dense12v zw = Dh_inv[(b + 1u) % kBodyCount] * rv;
            for(int i = 0; i < 12; ++i)
                worst_wrong = std::max(worst_wrong,
                                       std::abs(zw(i) - z_expected[b * 12u + static_cast<SizeT>(i)]));
        }
        std::printf("[t64a] diag_apply: neighbour-block divergence %.3e\n", worst_wrong);
        T64A_CHECK(c, worst_wrong > 1.0e-3);
    }

    // ---- the `converged != 0` early-out ---------------------------------------
    z_dev.copy_from(span<const double>{sentinel});
    converged.copy_from(IndexT{1});
    impl.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    gg::frame_stream().synchronize();  // the early-out's kernel is on the frame stream
    z_dev.copy_to(span<double>{z});
    int n_early_out = 0;
    for(SizeT i = 0; i < 24u; ++i)
        if(z[i] == sentinel[i])
            ++n_early_out;
    std::printf("[t64a] diag_apply: converged != 0 kept %d/24 entries unchanged\n", n_early_out);
    T64A_CHECK(c, n_early_out == 24);

    // ---- the no-local-preconditioner fallback (z <- r) -------------------------
    {
        gg::GlobalLinearSystem::Impl impl_fallback;
        impl_fallback.diag_subsystems.register_sim_system(*T64aProbe::abd_linear);
        impl_fallback.init();
        REQUIRE(impl_fallback.no_precond_diag_subsystem_indices.size() == 1u);
        CHECK(impl_fallback.no_precond_diag_subsystem_indices[0] == 0);
        z_dev.copy_from(span<const double>{sentinel});
        converged.copy_from(IndexT{0});
        impl_fallback.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
        z_dev.copy_to(span<double>{z});
        int n_copied = 0;
        for(SizeT i = 0; i < 24u; ++i)
            if(z[i] == r[i])
                ++n_copied;
        std::printf("[t64a] diag_apply: fallback copied r to z for %d/24 entries\n", n_copied);
        T64A_CHECK(c, n_copied == 24);
    }
}

// ---------------------------------------------------------------------------
//  CASE 3 — the tolerance checker's boundary/atomic laws on the host
// ---------------------------------------------------------------------------
//
// The reference's kernel is three statements: read the flag, `return` if it is 0,
// then `for(c = 3; c < 12; ++c) if (abs(dq[c]) > abs_tol) { exch(0); break; }`.
// `exceeded_row`/`failed_row` (abd_tolerance_checker_function.h) are that decision,
// and this case pins every edge of it against a plain host loop.
TEST_CASE("abd_solve_tolerance_boundary", "[luisa][abd_solve]")
{
    CaseScope c{"tolerance_boundary"};

    using DqHost = gg::abd_tolerance_checker_detail::DqHostMemoryView;

    constexpr double kAbsTol = 0.25;
    const double     above   = std::nextafter(kAbsTol, 1.0);   // one ulp above
    const double     below   = std::nextafter(kAbsTol, 0.0);   // one ulp below

    // the host oracle: exactly the reference's loop, written out here
    auto oracle = [](const gg::LVector12& dq, double abs_tol) noexcept
    {
        for(int i = 3; i < 12; ++i)
        {
            if(std::abs(dq(i)) > abs_tol)
                return true;
        }
        return false;
    };

    auto make_view = [](std::vector<gg::LVector12>& dqs)
    { return DqHost::from_handles(dqs.data(), static_cast<std::int32_t>(dqs.size())); };

    auto at = [](const gg::LVector12& dq, int comp) noexcept { return dq(comp); };

    // ---- the exact boundary: `|dq| == abs_tol` is *converged* (a strict `>`) ----
    {
        std::vector<gg::LVector12> dqs(1);
        dqs[0] = gg::LVector12{};
        dqs[0](3) = kAbsTol;
        auto view = make_view(dqs);
        T64A_CHECK(c, at(dqs[0], 3) == kAbsTol);
        T64A_CHECK(c, !gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
        dqs[0](3) = above;
        T64A_CHECK(c, gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
        dqs[0](3) = below;
        T64A_CHECK(c, !gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
        dqs[0](3) = -above;  // the sign is irrelevant (`abs`)
        T64A_CHECK(c, gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
        dqs[0](3) = -kAbsTol;
        T64A_CHECK(c, !gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
    }

    // ---- every one of the nine rotational components, one at a time ------------
    {
        std::vector<gg::LVector12> dqs(1);
        for(int comp = 3; comp < 12; ++comp)
        {
            dqs[0] = gg::LVector12{};
            dqs[0](comp) = 2.0 * kAbsTol;
            auto view    = make_view(dqs);
            T64A_CHECK(c, gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
            T64A_CHECK(c, oracle(dqs[0], kAbsTol));
            // the same component *below* the tolerance is converged
            dqs[0](comp) = 0.5 * kAbsTol;
            T64A_CHECK(c, !gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
            T64A_CHECK(c, !oracle(dqs[0], kAbsTol));
        }
    }

    // ---- the three *translational* components are ignored on purpose ----------
    {
        std::vector<gg::LVector12> dqs(1);
        dqs[0] = gg::LVector12{};
        for(int comp = 0; comp < 3; ++comp)
            dqs[0](comp) = 1.0e6;
        auto view = make_view(dqs);
        T64A_CHECK(c, !gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
        T64A_CHECK(c, !oracle(dqs[0], kAbsTol));
        // ... and adding one rotational component does trip it
        dqs[0](11) = above;
        T64A_CHECK(c, gg::abd_tolerance_checker_detail::exceeded_row(view, 0, kAbsTol));
        T64A_CHECK(c, oracle(dqs[0], kAbsTol));
    }

    // ---- the `success == 0` early return (the "already failed stays failed" law) --
    {
        std::vector<gg::LVector12> dqs(1);
        dqs[0] = gg::LVector12{};
        for(int comp = 3; comp < 12; ++comp)
            dqs[0](comp) = 1.0e6;
        auto view = make_view(dqs);
        T64A_CHECK(c, gg::abd_tolerance_checker_detail::failed_row(view, 0, kAbsTol, 1));
        T64A_CHECK(c, !gg::abd_tolerance_checker_detail::failed_row(view, 0, kAbsTol, 0));
        // a *passing* row is false for both flag values
        dqs[0] = gg::LVector12{};
        T64A_CHECK(c, !gg::abd_tolerance_checker_detail::failed_row(view, 0, kAbsTol, 1));
        T64A_CHECK(c, !gg::abd_tolerance_checker_detail::failed_row(view, 0, kAbsTol, 0));
    }

    // ---- agreement with the host oracle over random `dq` batches ---------------
    {
        std::mt19937                           rng(20260924u);
        std::uniform_real_distribution<double> val(-0.4, 0.4);
        std::uniform_real_distribution<double> tol(0.01, 0.3);
        constexpr int                          kBatches = 200;
        constexpr int                          kRows    = 5;
        int                                    mismatches = 0;
        int                                    n_failed   = 0;
        int                                    consumed   = 0;
        std::vector<gg::LVector12>             dqs(kRows);
        std::vector<double>                    tols(kBatches);
        for(int b = 0; b < kBatches; ++b)
        {
            for(int r = 0; r < kRows; ++r)
            {
                dqs[r] = gg::LVector12{};
                for(int i = 0; i < 12; ++i)
                    dqs[r](i) = val(rng);
            }
            const double abs_tol = tol(rng);
            tols[b]              = abs_tol;
            auto         view    = make_view(dqs);
            for(int r = 0; r < kRows; ++r)
            {
                const bool got = gg::abd_tolerance_checker_detail::exceeded_row(view, r, abs_tol);
                const bool exp = oracle(dqs[r], abs_tol);
                if(got != exp)
                    ++mismatches;
                if(exp)
                    ++n_failed;
                ++consumed;
            }
        }
        std::printf("[t64a] tolerance_boundary: %d/%d random (dq, tol) decisions match the "
                    "host oracle (%d fail, %d distinct tolerances)\n",
                    consumed - mismatches,
                    consumed,
                    n_failed,
                    kBatches);
        T64A_CHECK(c, mismatches == 0);
        T64A_CHECK(c, consumed == kBatches * kRows);
        // the batch is non-vacuous: both verdicts occur
        T64A_CHECK(c, n_failed > 0);
        T64A_CHECK(c, n_failed < consumed);
    }

    // ---- the whole batch's `success` flag, emulated on the host ----------------
    //
    // The shipped kernel's guard makes the flag a pure OR over the rows that see it
    // still set: seeding it 0 makes *every* row decide false (the early return), and
    // seeding it 1 makes it the OR of the rows' nine-component tests.
    {
        std::mt19937                           rng(7u);
        std::uniform_real_distribution<double> val(-1.0, 1.0);
        std::vector<gg::LVector12>             dqs(9);
        for(auto& dq : dqs)
        {
            dq = gg::LVector12{};
            for(int i = 3; i < 12; ++i)
                dq(i) = val(rng);
        }
        auto view = make_view(dqs);

        auto run_batch = [&](std::int32_t seed, double abs_tol)
        {
            std::int32_t flag = seed;
            int          n_threads_that_decided = 0;
            for(int r = 0; r < static_cast<int>(dqs.size()); ++r)
            {
                if(gg::abd_tolerance_checker_detail::failed_row(view, r, abs_tol, flag))
                {
                    flag = 0;
                    ++n_threads_that_decided;
                }
            }
            return std::pair{flag, n_threads_that_decided};
        };

        // a tolerance that makes the first row fail, and one that makes none fail
        const double tight = 1.0e-3;
        const double loose = 10.0;
        const auto [flag_tight, n_tight] = run_batch(1, tight);
        const auto [flag_loose, n_loose] = run_batch(1, loose);
        const auto [flag_zero, n_zero]   = run_batch(0, tight);
        std::printf("[t64a] tolerance_boundary: success seed 1/tight -> %d (decided %d), "
                    "seed 1/loose -> %d (decided %d), seed 0 -> %d (decided %d)\n",
                    flag_tight,
                    n_tight,
                    flag_loose,
                    n_loose,
                    flag_zero,
                    n_zero);
        T64A_CHECK(c, flag_tight == 0);
        T64A_CHECK(c, n_tight >= 1);
        T64A_CHECK(c, flag_loose == 1);  // nothing exceeded -> the flag survives
        T64A_CHECK(c, n_loose == 0);
        T64A_CHECK(c, flag_zero == 0);  // "already failed stays failed"
        T64A_CHECK(c, n_zero == 0);     // every row returned early
    }
}

// ---------------------------------------------------------------------------
//  CASE 4 — the tolerance row on the device (the shipped guards + the atomic)
// ---------------------------------------------------------------------------
TEST_CASE("abd_solve_tolerance_device", "[luisa][abd_solve]")
{
    CaseScope c{"tolerance_device"};
    test_runtime();

    using DqHost = gg::abd_tolerance_checker_detail::DqHostMemoryView;

    constexpr uint32_t kRows = 64u;
    std::mt19937       rng(4242u);
    // the rotational components are sampled *around* the tolerance below, so the
    // batch contains both verdicts (nine components make "all pass" rare otherwise)
    std::uniform_real_distribution<double> val(-0.10, 0.10);

    std::vector<gg::LVector12> dqs(kRows);
    for(auto& dq : dqs)
    {
        dq = gg::LVector12{};
        for(int i = 0; i < 12; ++i)
            dq(i) = val(rng);
    }
    constexpr double kAbsTol = 0.09;

    // the host instantiation of the *same* factored routine (a memory view over the
    // very same host data the device gets)
    auto host_view = DqHost::from_handles(dqs.data(), static_cast<std::int32_t>(kRows));
    std::vector<std::int32_t> host_failed(kRows, 0);
    int                       host_n_failed = 0;
    for(uint32_t r = 0; r < kRows; ++r)
    {
        host_failed[r] = gg::abd_tolerance_checker_detail::failed_row(
                             host_view, static_cast<std::int32_t>(r), kAbsTol, 1)
                             ? 1
                             : 0;
        host_n_failed += host_failed[r];
    }
    // non-vacuous: both verdicts occur in the batch
    REQUIRE(host_n_failed > 0);
    REQUIRE(host_n_failed < static_cast<int>(kRows));

    // ---- the device side -------------------------------------------------------
    lud::LDeviceBuffer<gg::LVector12> dqs_dev;
    dqs_dev.copy_from(span<const gg::LVector12>{dqs});
    lud::LDeviceBuffer<std::int32_t> flags_dev;
    std::vector<std::int32_t>        flags_zero(kRows, -1);
    flags_dev.copy_from(span<const std::int32_t>{flags_zero});
    lud::LDeviceVar<std::int32_t> success;
    gg::frame_stream().synchronize();

    auto run_device = [&](std::int32_t seed)
    {
        success.copy_from(seed, &gg::frame_stream());
        flags_dev.copy_from(span<const std::int32_t>{flags_zero});
        gg::frame_stream().synchronize();
        auto& k = t64a_tolerance_kernel();
        gg::frame_stream().record(
            k(dqs_dev.cview(), success.view(), flags_dev.view(), kAbsTol, kRows).dispatch(kRows));
        return success.copy_to(&gg::frame_stream());
    };

    // (1) the flag starts set: the device's per-row verdicts must equal the host's
    const auto flag = run_device(1);
    const auto flags = read_device<std::int32_t>(flags_dev.cview());
    REQUIRE(flags.size() == kRows);
    int mismatches  = 0;
    int dev_n_failed = 0;
    for(uint32_t r = 0; r < kRows; ++r)
    {
        if(flags[r] != host_failed[r])
            ++mismatches;
        dev_n_failed += flags[r];
    }
    std::printf("[t64a] tolerance_device: %u/%u per-row verdicts equal the host "
                "(%d fail), success %d -> %d\n",
                kRows - static_cast<uint32_t>(mismatches),
                kRows,
                dev_n_failed,
                static_cast<int>(1),
                static_cast<int>(flag));
    T64A_CHECK(c, mismatches == 0);
    T64A_CHECK(c, dev_n_failed == host_n_failed);
    T64A_CHECK(c, flag == 0);  // the batch contains failures -> exchanged to 0
    T64A_CHECK(c, dev_n_failed > 0);
    T64A_CHECK(c, dev_n_failed < static_cast<int>(kRows));

    // (2) "already failed stays failed": seeded 0, *every* thread returns early, so
    //     no flag is writable, every verdict is false and the flag is still 0
    const auto flag0 = run_device(0);
    const auto flags_after0 = read_device<std::int32_t>(flags_dev.cview());
    int        n_decided   = 0;
    for(uint32_t r = 0; r < kRows; ++r)
        n_decided += (flags_after0[r] != 0) ? 1 : 0;
    std::printf("[t64a] tolerance_device: seeded 0 -> success %d, %d/%u rows decided "
                "(the early return suppresses all of them)\n",
                static_cast<int>(flag0),
                n_decided,
                kRows);
    T64A_CHECK(c, flag0 == 0);
    T64A_CHECK(c, n_decided == 0);

    // (3) an all-passing batch leaves the flag at 1 (the OR identity)
    {
        std::vector<gg::LVector12> soft(kRows);
        for(auto& dq : soft)
        {
            dq = gg::LVector12{};
            for(int i = 3; i < 12; ++i)
                dq(i) = 0.25 * kAbsTol;  // below the tolerance
        }
        dqs_dev.copy_from(span<const gg::LVector12>{soft});
        gg::frame_stream().synchronize();
        const auto soft_flag = run_device(1);
        const auto soft_flags = read_device<std::int32_t>(flags_dev.cview());
        int        n_soft_failed = 0;
        for(uint32_t r = 0; r < kRows; ++r)
            n_soft_failed += soft_flags[r];
        std::printf("[t64a] tolerance_device: all-passing batch -> success %d, %d failures\n",
                    static_cast<int>(soft_flag),
                    n_soft_failed);
        T64A_CHECK(c, soft_flag == 1);
        T64A_CHECK(c, n_soft_failed == 0);
    }
}

// ---------------------------------------------------------------------------
//  CASE 5 — the Newton tolerance manager/checker contract, `max_translation`'s
//           comparison and the ABD checker's convergence flip
// ---------------------------------------------------------------------------
TEST_CASE("abd_solve_manager", "[luisa][abd_solve]")
{
    CaseScope c{"manager"};
    test_runtime();

    T64aProbe::reset();
    T64aMockCheckerA::reset();
    T64aMockCheckerB::reset();
    MockOrder::serial = 0;

    SceneSpec spec;
    spec.velocity_tol  = 0.5;
    spec.transrate_tol = 0.01;
    spec.dt            = 0.01;

    auto   scene = build_scene(spec);
    Engine engine{"luisa", workspace_path("uipc_luisa_t64a_manager")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    REQUIRE(T64aProbe::tol_manager != nullptr);
    REQUIRE(T64aProbe::max_tol != nullptr);
    REQUIRE(T64aProbe::abd_tol != nullptr);
    REQUIRE(T64aProbe::abd != nullptr);
    REQUIRE(T64aProbe::abd_linear != nullptr);
    REQUIRE(T64aProbe::gvm != nullptr);
    // both real checkers are in play in an ABD scene (the base `require<>`s the
    // manager, so they are only built when it exists)
    CHECK(T64aProbe::max_tol->is_valid());
    CHECK(T64aProbe::abd_tol->is_valid());

    // ---- `do_build`'s config reads (the reference's own assertions) ------------
    T64A_CHECK(c, T64aProbe::max_tol->newton_velocity_tol == spec.velocity_tol);
    T64A_CHECK(c, T64aProbe::max_tol->velocity_tol_relative == 0.0);
    T64A_CHECK(c, T64aProbe::max_tol->dt_attr != nullptr);
    T64A_CHECK(c, T64aProbe::abd_tol->transrate_tol == spec.transrate_tol);
    T64A_CHECK(c, T64aProbe::abd_tol->dt_attr != nullptr);
    // the checkers are registered *with the manager* (the manager's collection is
    // private; this is its observable consequence -- `check()` reaches them below)
    T64A_CHECK(c, T64aMockCheckerA::check_calls == 0);
    T64A_CHECK(c, T64aMockCheckerB::check_calls == 0);

    step_frame(world);

    // ---- `pre_newton(frame)` reaches every checker ----------------------------
    T64aProbe::tol_manager->pre_newton(7);
    T64A_CHECK(c, T64aMockCheckerA::pre_newton_calls == 1);
    T64A_CHECK(c, T64aMockCheckerB::pre_newton_calls == 1);
    // PINNED REFERENCE QUIRK (cookbook §7.28): `NewtonToleranceChecker::pre_newton`
    // ignores the manager's `PreNewtonInfo` and constructs a **fresh default** one
    // for its subclass, so every checker sees frame 0 -- the frame reaches only the
    // log line.  Transcribed verbatim, and pinned here.
      std::printf("[t64a] manager: pre_newton(7) -> mock frames %llu/%llu, the real "
                  "checker's frame %llu (the reference forwards a DEFAULT info)\n",
                  static_cast<unsigned long long>(T64aMockCheckerA::last_frame),
                static_cast<unsigned long long>(T64aMockCheckerB::last_frame),
                static_cast<unsigned long long>(T64aProbe::max_tol->frame));
    T64A_CHECK(c, T64aMockCheckerA::last_frame == 0u);
    T64A_CHECK(c, T64aMockCheckerB::last_frame == 0u);
    T64A_CHECK(c, T64aProbe::max_tol->frame == 0u);
    // `do_pre_newton` still resets `res0` (that part *is* reached)
    T64A_CHECK(c, T64aProbe::max_tol->res0 == 0.0);

    // ---- `check()` with everything passing ------------------------------------
    gg::NewtonToleranceManager::ResultInfo ri;
    ri.frame(7);
    ri.newton_iter(0);
    T64aProbe::tol_manager->check(ri);
    T64A_CHECK(c, ri.converged());
    T64A_CHECK(c, T64aMockCheckerA::check_calls == 1);
    T64A_CHECK(c, T64aMockCheckerB::check_calls == 1);
    // the manager's fan-out follows the registration order -- measured, not assumed
    std::printf("[t64a] manager: mock checker order A=%d B=%d\n",
                T64aMockCheckerA::order,
                T64aMockCheckerB::order);
    T64A_CHECK(c, T64aMockCheckerA::order == 0);
    T64A_CHECK(c, T64aMockCheckerB::order == 1);
    // the two *real* checkers ran too (their `do_check` is the only writer of
    // `abs_tol`): the max-translation one used the absolute tolerance, the ABD one
    // `transrate_tol * dt`
    const double abs_tol_max    = spec.velocity_tol * spec.dt;
    const double abs_tol_abd    = spec.transrate_tol * spec.dt;
    const double max_before     = T64aProbe::max_tol->abs_tol;
    const double abd_before     = T64aProbe::abd_tol->abs_tol;
    std::printf("[t64a] manager: max_translation abs_tol %.6e (hand-derived %.6e), "
                "res %.6e, res0 %.6e\n",
                max_before,
                abs_tol_max,
                T64aProbe::max_tol->res,
                T64aProbe::max_tol->res0);
    T64A_CHECK(c, max_before == abs_tol_max);
    T64A_CHECK(c, T64aProbe::abd_tol->abs_tol == abs_tol_abd);
    // a static scene: no displacement, so `res == 0 <= abs_tol` and it converges
    T64A_CHECK(c, T64aProbe::max_tol->res == 0.0);
    T64A_CHECK(c, T64aProbe::max_tol->res <= T64aProbe::max_tol->abs_tol);

    // ---- the short-circuit: a failing mock stops the fan-out ------------------
    {
        T64aMockCheckerB::fail = true;
        MockOrder::serial      = 0;
        const auto calls_a_before = T64aMockCheckerA::check_calls;
        // make a *run* of the two real checkers observable: change their inputs so
        // that their `do_check` would write a *different* `abs_tol` (their value is
        // otherwise identical across runs, so a missing short-circuit would be
        // invisible -- the pin below is only non-vacuous because of this)
        T64aProbe::max_tol->newton_velocity_tol = 0.75;  // would become 7.5e-3
        T64aProbe::abd_tol->transrate_tol       = 0.25;  // would become 2.5e-3
        ri.frame(7);
        ri.newton_iter(3);
        T64aProbe::tol_manager->check(ri);
        T64A_CHECK(c, !ri.converged());
        T64A_CHECK(c, T64aMockCheckerA::check_calls == calls_a_before + 1);
        T64A_CHECK(c, T64aMockCheckerB::check_calls == 2);
        T64A_CHECK(c, T64aMockCheckerA::order == 0);
        T64A_CHECK(c, T64aMockCheckerB::order == 1);
        // the checkers *after* the failing one were not called at all: neither
        // `abs_tol` moved to its new value (both mocks sort before every
        // `uipc::backend::luisa::` checker, which the printout above measured)
        const double max_after = T64aProbe::max_tol->abs_tol;
        const double abd_after = T64aProbe::abd_tol->abs_tol;
        std::printf("[t64a] manager: after the failing check, max abs_tol %.6e -> %.6e "
                    "(7.5e-3 if it had run), abd abs_tol %.6e -> %.6e (2.5e-3 if it had "
                    "run)\n",
                    max_before,
                    max_after,
                    abd_before,
                    abd_after);
        T64A_CHECK(c, max_after == max_before);
        T64A_CHECK(c, abd_after == abd_before);
        T64A_CHECK(c, max_after != 0.75 * spec.dt);
        T64A_CHECK(c, abd_after != 0.25 * spec.dt);
        T64aProbe::max_tol->newton_velocity_tol = spec.velocity_tol;
        T64aProbe::abd_tol->transrate_tol       = spec.transrate_tol;
        T64aMockCheckerB::fail = false;
    }

    // ---- `MaxTranslationChecker`'s comparison, driven directly ----------------
    {
        // the *relative* mode (Stiff-GIPC): `abs_tol = rel * scene_diagonal * dt`
        const auto check_result = [&](Float rel, Float vtol)
        {
            T64aProbe::max_tol->velocity_tol_relative = rel;
            T64aProbe::max_tol->newton_velocity_tol   = vtol;
            gg::NewtonToleranceManager::CheckResultInfo info;
            T64aProbe::max_tol->do_check(info);
            // `CheckResultInfo` has no verdict *getter* in the reference (only the
            // setter), so the verdict is read the way the checker computed it
            return T64aProbe::max_tol->res <= T64aProbe::max_tol->abs_tol;
        };
        const Float diag   = T64aProbe::gvm->scene_diagonal();
        const Float rel    = 0.25;
        T64A_CHECK(c, diag > 0.0);
        CHECK(check_result(rel, 0.5));
        const double expected_rel = static_cast<double>(rel * diag * static_cast<Float>(spec.dt));
        std::printf("[t64a] manager: relative mode abs_tol %.6e (hand-derived "
                    "0.25 * scene_diagonal %.6e * dt %.6e = %.6e), res %.6e\n",
                    T64aProbe::max_tol->abs_tol,
                    diag,
                    spec.dt,
                    expected_rel,
                    T64aProbe::max_tol->res);
        T64A_CHECK(c, T64aProbe::max_tol->abs_tol == expected_rel);
        T64A_CHECK(c, T64aProbe::max_tol->res <= T64aProbe::max_tol->abs_tol);
        // back to the absolute mode ...
        CHECK(check_result(0.0, 0.5));
        T64A_CHECK(c, T64aProbe::max_tol->abs_tol == abs_tol_max);
        // ... and the comparison itself flips when `abs_tol` drops below `res`
        // (a *negative* tolerance is the only way to flip a zero residual)
        const auto converged_negative = check_result(0.0, -1.0);
        std::printf("[t64a] manager: abs_tol %.6e with res %.6e -> converged %d\n",
                    T64aProbe::max_tol->abs_tol,
                    T64aProbe::max_tol->res,
                    static_cast<int>(converged_negative));
        T64A_CHECK(c, !converged_negative);
        T64A_CHECK(c, T64aProbe::max_tol->res > T64aProbe::max_tol->abs_tol);
        // restore the checker's own values
        T64aProbe::max_tol->newton_velocity_tol   = spec.velocity_tol;
        T64aProbe::max_tol->velocity_tol_relative = 0.0;
        CHECK(check_result(0.0, spec.velocity_tol));
    }

    // ---- the ABD checker's convergence flip, on the *shipped* shader ----------
    {
        gg::GlobalLinearSystem::Impl impl;
        impl.diag_subsystems.register_sim_system(*T64aProbe::abd_linear);
        impl.init();
        impl.build_linear_system();
        REQUIRE(impl.diag_dof_offsets_counts.total_count() == 24u);

        const auto solve_and_check = [&](const std::vector<double>& x)
        {
            impl.x.copy_from(span<const double>{x});
            gg::frame_stream().synchronize();
            impl.distribute_solution();
            gg::NewtonToleranceManager::CheckResultInfo info;
            T64aProbe::abd_tol->do_check(info);
            // the checker's verdict is exactly its one-element `success` flag
            return T64aProbe::abd_tol->success.copy_to(&gg::frame_stream()) != 0;
        };

        // (1) `retrieve_solution` writes `dq = -x` (the reference's negation) --
        //     a hand-derived pin of the shipped shader
        std::vector<double> x(24, 0.0);
        for(int i = 0; i < 12; ++i)
            x[12 + static_cast<SizeT>(i)] = 0.01 * static_cast<double>(i + 1);
        CHECK(solve_and_check(x) == false);
        const auto dq = read_device(T64aProbe::abd->dqs());
        REQUIRE(dq.size() == 2u);
        double worst_neg = 0.0;
        for(int b = 0; b < 2; ++b)
            for(int i = 0; i < 12; ++i)
                worst_neg = std::max(worst_neg,
                                     std::abs(dq[static_cast<SizeT>(b)](i)
                                              + x[static_cast<SizeT>(b * 12 + i)]));
        std::printf("[t64a] manager: shipped retrieve_solution dq = -x, worst dev %.3e "
                    "(|x|max %.3e)\n",
                    worst_neg,
                    std::abs(x[23]));
        T64A_CHECK(c, worst_neg <= 0.0);  // a negation is exact
        T64A_CHECK(c, std::abs(x[23]) > 1.0e-2);
        // the failing check is reported as failed ...
        T64A_CHECK(c, T64aProbe::abd_tol->abs_tol == abs_tol_abd);
        // ... while `do_report()` still claims `Tol: < abs_tol`: the reference's
        // local `bool h_success` shadows its member, so the member stays 1 forever
        // (a pinned reference inconsistency, cookbook §7.28)
        const auto report = T64aProbe::abd_tol->do_report();
        std::printf("[t64a] manager: do_report() after a FAILED check -> \"%s\"\n",
                    report.c_str());
        T64A_CHECK(c, report.find("< ") != std::string::npos);
        T64A_CHECK(c, T64aProbe::abd_tol->h_success == 1);
        // the flag itself did flip (the local, not the member)
        T64A_CHECK(c, T64aProbe::abd_tol->success.copy_to(&gg::frame_stream()) == 0);

        // (2) the boundary: `|dq_i| == abs_tol` exactly is converged
        std::vector<double> boundary(24, 0.0);
        boundary[15] = -abs_tol_abd;  // dq(15) = +abs_tol
        CHECK(solve_and_check(boundary) == true);
        // one ulp above is not
        std::vector<double> just_above(24, 0.0);
        just_above[15] = -std::nextafter(abs_tol_abd, 1.0);
        const auto converged_above = solve_and_check(just_above);
        std::printf("[t64a] manager: |dq_15| == abs_tol -> converged true; "
                    "one ulp above -> converged %d\n",
                    static_cast<int>(converged_above));
        T64A_CHECK(c, !converged_above);

        // (3) the three translational components are ignored on purpose
        std::vector<double> translation(24, 0.0);
        for(int i = 0; i < 3; ++i)
            translation[12 + static_cast<SizeT>(i)] = -1.0e3;
        CHECK(solve_and_check(translation) == true);
        // a *rotational* component of the same magnitude is not
        std::vector<double> rotational(24, 0.0);
        rotational[12 + 3] = -1.0e3;
        CHECK(solve_and_check(rotational) == false);

        // (4) `abs_tol` is recomputed on every check (the reference's first line)
        T64aProbe::abd_tol->transrate_tol = 2.0;
        CHECK(solve_and_check(translation) == true);
        T64A_CHECK(c, T64aProbe::abd_tol->abs_tol == 2.0 * spec.dt);
        T64aProbe::abd_tol->transrate_tol = spec.transrate_tol;

        // (5) all-zero `dq` (the solver did not move) converges
        CHECK(solve_and_check(std::vector<double>(24, 0.0)) == true);
    }
}

// ---------------------------------------------------------------------------
//  CASE 6 — the line-search reporter's bodies, its registration and the state it
//           hands to a subreporter (a real scene)
// ---------------------------------------------------------------------------
TEST_CASE("abd_solve_line_search", "[luisa][abd_solve]")
{
    CaseScope c{"line_search"};
    test_runtime();

    T64aProbe::reset();
    T64aMockSubreporter::reset();

    SceneSpec spec;
    spec.with_joint    = true;
    spec.body0_fixed   = true;
    spec.dt            = 0.01;

    auto   scene = build_scene(spec);
    Engine engine{"luisa", workspace_path("uipc_luisa_t64a_line_search")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    REQUIRE(T64aProbe::ls_reporter != nullptr);
    // THE REGISTRATION (what the SCOPE NOTE said was missing): the reporter is
    // built and valid, so its subreporters resolve instead of being invalidated
    CHECK(T64aProbe::ls_reporter->is_valid());
    REQUIRE(T64aProbe::abd != nullptr);
    REQUIRE(T64aProbe::abd_linear != nullptr);
    REQUIRE(T64aProbe::line_searcher != nullptr);
    // the reporter's `Impl::init` fan-out reached the test-local subreporter
    // (and the real inter-body one, which the joint put in the scene)
    T64A_CHECK(c, T64aMockSubreporter::init_calls == 1);
    T64A_CHECK(c, T64aMockSubreporter::energy_calls == 0);

    step_frame(world);

    // ---- the shipped `retrieve_solution` gives the step body a known `dq` ------
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T64aProbe::abd_linear);
    impl.init();
    impl.build_linear_system();
    REQUIRE(impl.diag_dof_offsets_counts.total_count() == 24u);

    std::vector<double> x(24, 0.0);
    for(int i = 0; i < 12; ++i)
        x[12 + static_cast<SizeT>(i)] = 0.02 * static_cast<double>(i + 1);
    impl.x.copy_from(span<const double>{x});
    gg::frame_stream().synchronize();
    impl.distribute_solution();

    const auto q_before = read_device(T64aProbe::abd->qs());
    const auto dq       = read_device(T64aProbe::abd->dqs());
    REQUIRE(q_before.size() == 2u);
    REQUIRE(dq.size() == 2u);

    // ---- `do_record_start_point` + `do_step_forward` ---------------------------
    //
    // `q_temp` has no public accessor, so the start point is observed the way the
    // line searcher observes it: `step_forward(alpha)` must produce
    // `q = q_temp + alpha * dq`, and a *second* identical call must reproduce the
    // same `q` (the reference always evaluates from `q_temp`, never from the
    // already-stepped `q`) -- which is only true if the snapshot really happened.
    gg::LineSearcher::RecordInfo record_info;
    T64aProbe::ls_reporter->do_record_start_point(record_info);
    lud::main_stream().synchronize();
    gg::frame_stream().synchronize();

    gg::LineSearcher::StepInfo step_info;
    step_info.alpha = 1.0;
    T64aProbe::ls_reporter->do_step_forward(step_info);
    gg::frame_stream().synchronize();
    const auto q_step1 = read_device(T64aProbe::abd->qs());

    T64aProbe::ls_reporter->do_step_forward(step_info);
    gg::frame_stream().synchronize();
    const auto q_step2 = read_device(T64aProbe::abd->qs());

    double worst_moved = 0.0, worst_idempotent = 0.0, worst_fixed = 0.0;
    for(SizeT b = 0; b < 2u; ++b)
    {
        for(int i = 0; i < 12; ++i)
        {
            const double expected = b == 0u ? q_before[b](i)
                                            : q_before[b](i) + 1.0 * dq[b](i);
            worst_moved = std::max(worst_moved, std::abs(q_step1[b](i) - expected));
            worst_idempotent =
                std::max(worst_idempotent, std::abs(q_step2[b](i) - q_step1[b](i)));
            if(b == 0u)
                worst_fixed = std::max(worst_fixed, std::abs(q_step1[b](i) - q_before[b](i)));
        }
    }
    std::printf("[t64a] line_search: step_forward(1.0) vs hand-derived q_temp + dq %.3e, "
                "second call %.3e, fixed body %.3e\n",
                worst_moved,
                worst_idempotent,
                worst_fixed);
    T64A_CHECK(c, worst_moved <= 0.0);       // exactly `q_temp + 1.0 * dq`
    T64A_CHECK(c, worst_idempotent <= 0.0);  // bitwise: it re-reads `q_temp`
    T64A_CHECK(c, worst_fixed <= 0.0);       // the fixed body is untouched
    // non-vacuous: the moving body really moved
    T64A_CHECK(c, std::abs(dq[1](11)) > 1.0e-3);
    T64A_CHECK(c, std::abs(q_step1[1](11) - q_before[1](11)) > 1.0e-3);

    // ---- `do_compute_energy`: the fan-out is a *differenced* statement ---------
    //
    // The kinetic and shape terms are pinned by their own batches (T6.2/T6.3a); what
    // this case owns is the *fan-out*: the total must grow by exactly the mock
    // subreporter's own values when the mock is switched on, with everything else
    // (the same state, the same kinetic, the same shape and the real inter-body
    // subreporter) unchanged.
    lud::LDeviceVar<Float> energy;
    const auto             energy_of = [&]()
    {
        gg::LineSearcher::ComputeEnergyInfo info{T64aProbe::line_searcher, energy.view()};
        T64aProbe::ls_reporter->do_compute_energy(info);
        return energy.copy_to(&gg::frame_stream());
    };

    T64aMockSubreporter::active = false;
    const auto total_off        = energy_of();
    T64aMockSubreporter::active = true;
    const auto total_on         = energy_of();
    const auto mock_sum = T64aMockSubreporter::kMockEnergy0 + T64aMockSubreporter::kMockEnergy1;

    std::printf("[t64a] line_search: total energy without the mock %.6e, with it %.6e "
                "(difference %.6e, the mock's own sum %.6e), extent calls %d, "
                "energy calls %d, the mock's slice size %u\n",
                total_off,
                total_on,
                total_on - total_off,
                mock_sum,
                T64aMockSubreporter::extent_calls,
                T64aMockSubreporter::energy_calls,
                static_cast<unsigned>(T64aMockSubreporter::last_count));
    T64A_CHECK(c, std::isfinite(total_off));
    T64A_CHECK(c, total_off >= 0.0);  // kinetic + shape + joint energies are non-negative
    T64A_CHECK(c, T64aMockSubreporter::last_count == T64aMockSubreporter::kMockEnergyCount);
    T64A_CHECK(c, T64aMockSubreporter::energy_calls == 2);
    T64A_CHECK(c, T64aMockSubreporter::extent_calls >= 2);
    T64A_CHECK(c, T64aMockSubreporter::last_sum == mock_sum);
    T64A_CHECK(c, std::abs((total_on - total_off) - mock_sum) <= 1e-9);
    // non-vacuous: the two runs really differ
    T64A_CHECK(c, total_on != total_off);

    // ... and a third run reproduces the second one exactly (the state was not
    // consumed by the energy fan-out)
    const auto total_on_again = energy_of();
    T64A_CHECK(c, total_on_again == total_on);
    T64aMockSubreporter::active = false;
}

// ---------------------------------------------------------------------------
//  CASE 7 — every new generic routine on the device vs the host
// ---------------------------------------------------------------------------
//
// The 12x12 inverse's device instantiation is the *shipped* preconditioner shader:
// case 2 reads its `diag_inv` back and compares it against the host
// `lps::inverse`/Eigen on the same input (the §7.27 rule 2 idiom: no test-local
// kernel carries a big unrolled 12x12 chain here -- that is where
// `luisa::compute::Device::compile` deadlocks).  This case rounds the two *small*
// factored bodies, `abd_tolerance_checker_detail::failed_row` and
// `abd_diag_preconditioner_detail::apply_row`, and checks the host instantiation of
// the preconditioner's `assemble_row`.
namespace
{
/// `abd_diag_preconditioner_detail::apply_row` on the device (the shipped kernel's
/// body: `z.segment<12>(i * 12) = diag_inv(i) * r.segment<12>(i * 12)`).
auto& t64a_apply_row_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,             // r
                                                lcrt::Buffer<double>,             // z
                                                lcrt::Buffer<gg::LMatrix12x12>,   // diag_inv
                                                uint32_t>(                        // block count
        [](lcrt::BufferVar<double>            r,
           lcrt::BufferVar<double>            z,
           lcrt::BufferVar<gg::LMatrix12x12>  diag_inv,
           lcrt::Var<uint32_t>                n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto R = lla::LCDenseVectorDeviceView<lcrt::Var<double>>::from_handles(r, n * 12u);
                auto Z = lla::LDenseVectorDeviceView<lcrt::Var<double>>::from_handles(z, n * 12u);
                auto D = gg::abd_diag_preconditioner_detail::DiagInvDeviceView::from_handles(
                    diag_inv, n);
                gg::abd_diag_preconditioner_detail::apply_row(
                    Z, R, D, lcrt::cast<std::int32_t>(i));
            };
        });
    return k;
}
}  // namespace

TEST_CASE("abd_solve_device_vs_host", "[luisa][abd_solve]")
{
    CaseScope c{"device_vs_host"};
    test_runtime();

    // ---- (a) the tolerance row -------------------------------------------------
    {
        constexpr uint32_t kRows = 32u;
        std::mt19937       rng(99u);
        std::uniform_real_distribution<double> val(-0.10, 0.10);
        std::vector<gg::LVector12>             dqs(kRows);
        for(auto& dq : dqs)
        {
            dq = gg::LVector12{};
            for(int i = 0; i < 12; ++i)
                dq(i) = val(rng);
        }
        constexpr double kAbsTol = 0.09;

        // the host side: the *same* factored routine over memory views
        auto host_view = gg::abd_tolerance_checker_detail::DqHostMemoryView::from_handles(
            dqs.data(), static_cast<std::int32_t>(kRows));
        std::vector<std::int32_t> host_flags(kRows, 0);
        std::vector<double>       host_scratch(kRows, 0.0);  // the "consumed slot" log
        int                       host_failed = 0;
        for(uint32_t r = 0; r < kRows; ++r)
        {
            const bool failed = gg::abd_tolerance_checker_detail::failed_row(
                host_view, static_cast<std::int32_t>(r), kAbsTol, 1);
            host_flags[r]   = failed ? 1 : 0;
            host_scratch[r] = failed ? 1.0 : 0.0;
            host_failed += host_flags[r];
        }
        REQUIRE(host_failed > 0);
        REQUIRE(host_failed < static_cast<int>(kRows));

        lud::LDeviceBuffer<gg::LVector12> dqs_dev;
        dqs_dev.copy_from(span<const gg::LVector12>{dqs});
        lud::LDeviceBuffer<std::int32_t> flags_dev;
        std::vector<std::int32_t>        init_flags(kRows, -1);
        flags_dev.copy_from(span<const std::int32_t>{init_flags});
        lud::LDeviceVar<std::int32_t> success;
        success.copy_from(1, &gg::frame_stream());
        gg::frame_stream().synchronize();
        {
            auto& k = t64a_tolerance_kernel();
            gg::frame_stream().record(
                k(dqs_dev.cview(), success.view(), flags_dev.view(), kAbsTol, kRows)
                    .dispatch(kRows));
        }
        const auto flag  = success.copy_to(&gg::frame_stream());
        const auto flags = read_device<std::int32_t>(flags_dev.cview());
        REQUIRE(flags.size() == kRows);

        int    mismatches = 0;
        int    consumed   = 0;
        double dev_scratch_sum = 0.0;
        for(uint32_t r = 0; r < kRows; ++r)
        {
            ++consumed;  // every slot of the round is consumed (§7.26 rule 16)
            if(flags[r] != host_flags[r])
                ++mismatches;
            dev_scratch_sum += static_cast<double>(flags[r]);
        }
        std::printf("[t64a] device_vs_host: tolerance row %u/%u slots equal (host failed %d, "
                    "device failed %.0f), success %d, consumed %d/%u\n",
                    kRows - static_cast<uint32_t>(mismatches),
                    kRows,
                    host_failed,
                    dev_scratch_sum,
                    static_cast<int>(flag),
                    consumed,
                    kRows);
        T64A_CHECK(c, mismatches == 0);
        T64A_CHECK(c, consumed == static_cast<int>(kRows));
        T64A_CHECK(c, dev_scratch_sum == static_cast<double>(host_failed));
        T64A_CHECK(c, flag == 0);
    }

    // ---- (b) `apply_row`: the ported preconditioner's apply body ---------------
    {
        constexpr uint32_t kN = 2u;
        auto               A0 = random_spd(21u);
        auto               A1 = random_spd(22u, 2.5);

        std::vector<gg::LMatrix12x12> inv(kN);
        inv[0] = to_lm12(A0.inverse());
        inv[1] = to_lm12(A1.inverse());

        std::vector<double> r(kN * 12u, 0.0);
        for(uint32_t b = 0; b < kN; ++b)
            for(int i = 0; i < 12; ++i)
                r[b * 12u + static_cast<SizeT>(i)] =
                    0.5 * static_cast<double>(b + 1) - 0.25 * static_cast<double>(i) + 0.125;

        // the host round
        std::vector<double> z_host(kN * 12u, 0.0);
        {
            auto R = lla::LCDenseVectorHostMemoryView<double>::from_handles(
                r.data(), static_cast<std::int32_t>(r.size()));
            auto Z = lla::LDenseVectorHostMemoryView<double>::from_handles(
                z_host.data(), static_cast<std::int32_t>(z_host.size()));
            auto D = gg::abd_diag_preconditioner_detail::DiagInvHostMemoryView::from_handles(
                inv.data(), static_cast<std::int32_t>(inv.size()));
            for(uint32_t b = 0; b < kN; ++b)
                gg::abd_diag_preconditioner_detail::apply_row(
                    Z, R, D, static_cast<std::int32_t>(b));
        }

        // the device round
        lud::LDeviceBuffer<double>             r_dev, z_dev;
        lud::LDeviceBuffer<gg::LMatrix12x12>   inv_dev;
        r_dev.copy_from(span<const double>{r});
        std::vector<double> z_zero(kN * 12u, 0.0);
        z_dev.copy_from(span<const double>{z_zero});
        inv_dev.copy_from(span<const gg::LMatrix12x12>{inv});
        gg::frame_stream().synchronize();
        {
            auto& k = t64a_apply_row_kernel();
            gg::frame_stream().record(k(r_dev.cview(), z_dev.view(), inv_dev.cview(), kN)
                                          .dispatch(kN));
        }
        const auto z_dev_host = read_device(z_dev.cview());

        double worst_abs = 0.0, worst_rel = 0.0, worst_eigen = 0.0;
        int    n_compared = 0;
        for(uint32_t b = 0; b < kN; ++b)
        {
            const Dense12 D = b == 0u ? A0 : A1;
            for(int i = 0; i < 12; ++i)
            {
                ++n_compared;
                const double a = z_host[b * 12u + static_cast<SizeT>(i)];
                const double d = z_dev_host[b * 12u + static_cast<SizeT>(i)];
                worst_abs      = std::max(worst_abs, std::abs(a - d));
                worst_rel      = std::max(worst_rel, rel_err(a, d));
                // an independent Eigen oracle for the same statement
                Dense12v       rv;
                for(int k = 0; k < 12; ++k)
                    rv(k) = r[b * 12u + static_cast<SizeT>(k)];
                const Dense12v zv = D.inverse() * rv;
                worst_eigen       = std::max(worst_eigen, std::abs(zv(i) - d));
            }
        }
        std::printf("[t64a] device_vs_host: apply_row %d/%d entries, host-vs-device "
                    "worst abs %.3e (rel %.3e), device vs Eigen %.3e\n",
                    n_compared,
                    n_compared,
                    worst_abs,
                    worst_rel,
                    worst_eigen);
        T64A_CHECK(c, n_compared == static_cast<int>(kN * 12u));
        T64A_CHECK(c, worst_rel <= 1e-15);
        T64A_CHECK(c, worst_eigen <= 1e-12);
        // non-vacuous: the product is far from `r` (these blocks are not the identity)
        T64A_CHECK(c, std::abs(z_host[0] - r[0]) > 1e-3);
    }

    // ---- (c) the 12x12 inverse's *factored* wrapper, on the host ---------------
    //
    // `assemble_row` must be exactly `lps::inverse` of the block; the device
    // instantiation of the same routine is the shipped shader of case 2 (its
    // `diag_inv` read-back vs the host `lps::inverse` of the same `diag_hessian`).
    {
        std::vector<gg::LMatrix12x12> blocks(2);
        blocks[0] = to_lm12(random_spd(31u));
        blocks[1] = to_lm12(singular_block());
        const auto A = random_spd(31u);

        auto view = gg::abd_diag_preconditioner_detail::DiagInvHostMemoryView::from_handles(
            blocks.data(), static_cast<std::int32_t>(blocks.size()));

        const auto got0 = to_eigen(gg::abd_diag_preconditioner_detail::assemble_row(view, 0));
        const auto again =
            to_eigen(gg::abd_diag_preconditioner_detail::assemble_row(view, 0));
        const auto exp0 = to_eigen(lps::inverse(to_lmat(A)));
        std::printf("[t64a] device_vs_host: assemble_row is lps::inverse (worst %.3e vs Eigen, "
                    "reproducible %.3e), singular block %.3e\n",
                    dev(got0, A.inverse()),
                    max_abs(got0 - again),
                    max_abs(to_eigen(gg::abd_diag_preconditioner_detail::assemble_row(view, 1))));
        T64A_CHECK(c, dev(got0, A.inverse()) <= 1e-13);
        T64A_CHECK(c, max_abs(got0 - exp0) == 0.0);   // the wrapper adds nothing
        T64A_CHECK(c, max_abs(got0 - again) == 0.0);  // deterministic
        // the singular block keeps the port's documented all-zero answer
        const auto got1 = to_eigen(gg::abd_diag_preconditioner_detail::assemble_row(view, 1));
        T64A_CHECK(c, max_abs(got1) == 0.0);
        T64A_CHECK(c, max_abs(to_eigen(blocks[0])) > 1.0);  // the input is non-trivial
    }
}

// ---------------------------------------------------------------------------
//  CASE 8 — per-case assertion accounting (the sibling files' summary case)
// ---------------------------------------------------------------------------
TEST_CASE("abd_solve_summary", "[luisa][abd_solve]")
{
    int total = 0;
    for(const auto& [name, checks] : g_case_counts)
    {
        std::printf("[t64a] case %-32s %5d checks\n", name.c_str(), checks);
        total += checks;
        CHECK(checks > 0);
    }
    std::printf("[t64a] summary: %d cases, %d checks\n",
                static_cast<int>(g_case_counts.size()),
                total);
}
}  // namespace t64a
