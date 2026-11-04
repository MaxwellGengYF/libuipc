/********************************************************************
 * @file luisa_pcg_test.cpp
 * @brief T8.3a-2 — `IterativeSolver` + `LinearPCG`: the shipped solve chain
 *        (GlobalLinearSystem::Impl::build_linear_system -> solve_linear_system ->
 *        IterativeSolver::solve -> LinearPCG::do_solve -> back into
 *        Impl::spmv / Impl::apply_preconditioner) driven end to end, plus the
 *        config mapping `linear_system/solver in {linear_pcg, fused_pcg}`.
 *
 * WHAT IS DRIVEN (the reference contract, src/backends/cuda/linear_system/)
 * --------------------------------------------------------------------------
 *  * `IterativeSolver::do_build()` — `require<GlobalLinearSystem>()` ->
 *    `do_build(BuildInfo&)` -> `add_solver(this)`, i.e. the engine's
 *    `Import::iterative_solver` slot really holds the solver (asserted by identity);
 *  * `Impl::solve_linear_system()` — the reference's `if(iterative_solver)` block,
 *    including `last_solve_iterations` living on the *system* (not the solver);
 *  * `LinearPCG::pcg()` — the loop of linear_pcg.cu:234-320, its `max_iter = 2n`
 *    cap, its `|rz| <= tol_rate*|rz_0| && accuracy` stopping rule and its two
 *    NaN/Inf guards;
 *  * the four `Impl` bodies T8.3a-2 had to land with the solver (`spmv`,
 *    `spmv_dot`, `matrix_data_ptrs`, the `triplet_count_dev` upload);
 *  * the preconditioner hook (`Impl::apply_preconditioner`'s local fan-out) and its
 *    `z <- r` fallback when no `LocalPreconditioner` is registered.
 *
 * THE SYSTEM
 * ----------
 * A 1-D 3-dof-per-node chain (8 nodes / 24 dofs): `A_ii = c_i * I` with
 * `c_0 = c_7 = 1` (Dirichlet ends) and `c_i = 2` otherwise, `A_{i,i+1} = -I` —
 * i.e. the block form of the second-difference operator.  It is SPD (asserted
 * numerically), its condition number is ~32, and every entry is a small *dyadic*
 * rational, so `A*b`, `b^T A b`, `x_1` etc. are exactly representable and the
 * Class-A assertions below can be *exact* rather than tolerance-banded.
 *
 * The right-hand side alternates in sign: a smooth (let alone constant) b is nearly
 * orthogonal to every row sum of a chain operator, which makes `b^T A b` tiny and the
 * first `alpha` explode -- and a *path Laplacian* (c = 1 at both ends) is outright
 * singular.  Both traps are pinned by the `ZeroLastBlock` mode below.
 *
 * Only the upper triangle (`row <= col`) is written: `ge2sym` keeps `row <= col`
 * (cookbook §7.39 rule 5) and `LSpMv::rbk_sym_spmv` mirrors the strict upper part,
 * so the matrix the solver applies is the symmetric closure of this table — which
 * is exactly what `dense_A()` builds on the host.
 *
 * ORACLES (all derived in this file; the CUDA backend is not host-includable)
 * -------------------------------------------------------------------------
 *  1) a host *transcription* of linear_pcg.cu:234-320 over the Eigen dense matrix.
 *     Deliberately not `Eigen::ConjugateGradient`: that stops on `||r||/tol` with a
 *     different extrapolation, so its iteration count is a different quantity
 *     (documented deviation from plan §5 phase 2 — report it, never assert on it);
 *  2) exact arithmetic where the arithmetic is identical (dyadic table, no
 *     atomics): `Impl::spmv` and the first PCG step are compared with Δ == 0;
 *  3) the SPD property itself (Eigen `SelfAdjointEigenSolver`, λ_min > 0).
 *
 * NEGATIVE CONTROLS (each one proves a gate can fail)
 * --------------------------------------------------
 *  * `pcg_singular_does_not_converge` — the same chain with the last node's block
 *    zeroed *and* isolated is singular, and the default `b` has a component along
 *    its null space (`b_last != 0`): `b` is not in `range(A)`, so no x exists.  PCG
 *    must burn the whole `max_iter = 2n = 48` budget, keep `|rz|/|rz_0| > tol_rate`
 *    and emit the reference's `reached max_iter` warning.  The same case then
 *    measures *why* the research note's controls #2 ("wrong b") and #3 ("non-SPD
 *    rejected") cannot be realised as written: an SPD A is nonsingular, so every b
 *    converges, and negating a whole symmetric table leaves PCG convergent (A -> -A
 *    and M -> -M only flip alpha's sign; measured: identical count, identical
 *    curve, x negated).
 *  * `pcg_converges_vs_host`'s second half — the same system re-assembled with the
 *    coupling blocks dropped (a broken operator = a broken SpMV): the iteration
 *    still "converges", but the residual against the *true* A is orders of
 *    magnitude above `tol_rate`, so the convergence gate cannot pass it.
 *  * `pcg_diag_preconditioner` — the preconditioner's *content* matters:
 *    block-Jacobi takes strictly fewer iterations than no preconditioner, and a
 *    deliberately wrong block table (each block replaced by its neighbour's) lands
 *    on a different count.
 *
 * WHAT THIS FILE CANNOT REACH (explicit SKIPs, never silent passes)
 * ----------------------------------------------------------------
 *  * `pcg_from_mtx` — no CUDA backend (`--backend_cuda=y` cannot configure), no
 *    `_dump_A_b`/`matrix_market` in this backend (plan 11);
 *  * `pcg_frame_pcg_two_block` — BLOCKED-BY-S0.6 (multi-frame `world.advance()`
 *    deadlocks after frame 1) and by plan 10 (the engine never calls `solve()`;
 *    plan 11 wires it), so there is no per-frame counter to compare.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"
#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <type_define.h>
#include <frame_stream.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/local_preconditioner.h>
#include <linear_system/iterative_solver.h>
#include <linear_system/linear_pcg.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;

namespace t82
{
// ===========================================================================
//  1) the block table — ONE source of truth for the device writes and the oracles
// ===========================================================================
constexpr int kBlockDim = 3;
constexpr int kNodes    = 8;
constexpr int kDofs     = kNodes * kBlockDim;  // 24

/// The shared-binary marker (cookbook §7.14 rule 6): every system registered in
/// this file is inert unless the scene's geometry carries it.
constexpr const char* kMarker = "t82_pcg_marker";
/// Reserved uid (the other test files use 0x5438_33xx / 0x5431_xxxx style ids).
constexpr U64 kUidChain = 0x54383243ull;  // "T82C"

/// The three matrix shapes the same fixture can assemble.
enum class MatrixMode
{
    /// the SPD chain (Dirichlet ends, unit coupling)
    Normal,
    /// the coupling blocks dropped: a broken SpMV (the negative control)
    BrokenCoupling,
    /// the last node's block zeroed and isolated: A is singular, b not in its range
    ZeroLastBlock,
    /// the whole table negated: symmetric but negative definite
    NegativeDefinite,
};

/// The three preconditioner shapes.
enum class PrecondMode
{
    /// no `LocalPreconditioner` registered -> the `z <- r` fallback
    None,
    /// block-Jacobi: D = blockdiag(A_ii), the reference's FEMDiagPreconditioner shape
    BlockJacobi,
    /// the same size but each block replaced by its neighbour's: a *wrong*
    /// preconditioner that is still SPD, so PCG stays well posed and only the
    /// iteration count can tell them apart
    WrongBlock,
};

inline MatrixMode  g_matrix_mode  = MatrixMode::Normal;
inline PrecondMode g_precond_mode = PrecondMode::None;

/// The diagonal coefficient of block row `i`.
///
/// The values are deliberately *heterogeneous* (2, 3, 4, 6, 8, 12, 16, 24 — all
/// exactly representable): a uniform chain would make block-Jacobi a scalar
/// multiple of the identity, and PCG is invariant under `M -> c*M`, so the
/// preconditioner could not change the iteration count (§6 of the plan wants it
/// to). They are also strictly diagonally dominant (`c_i > 2` for every coupled
/// row), which is what makes the system SPD.
///
/// A *path Laplacian* (c = 1 at both ends, off-diagonal -1) is NOT usable: its
/// determinant is 0 (the constant vector is in the null space), and the first PCG
/// step then divides by `p^T A p ~ 0`. That is the trap the `ZeroLastBlock` mode
/// below deliberately reproduces.
[[nodiscard]] inline double diag_coeff(MatrixMode mode, int i)
{
    // 2, 3, 4, 6, 8, 12, 16, 24
    static constexpr double kDiag[kNodes] = {2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0};
    const double            c            = kDiag[i];
    if(mode == MatrixMode::ZeroLastBlock && i == kNodes - 1)
        return 0.0;
    if(mode == MatrixMode::NegativeDefinite)
        return -c;
    return c;
}

[[nodiscard]] inline double coupling_coeff(MatrixMode mode)
{
    return mode == MatrixMode::NegativeDefinite ? 1.0 : -1.0;
}

/// Whether the (i, i+1) blocks are written at all.
[[nodiscard]] inline bool has_coupling(MatrixMode mode)
{
    return mode != MatrixMode::BrokenCoupling && mode != MatrixMode::ZeroLastBlock;
}

/// A 3x3 block, column-major like `linalg::LMat` (`block(r, c)`).
struct Block
{
    double v[kBlockDim][kBlockDim]{};  // v[r][c]

    [[nodiscard]] double operator()(int r, int c) const { return v[r][c]; }
    void                  set(int r, int c, double x) { v[r][c] = x; }
};

struct HostTriplet
{
    int   row = 0;
    int   col = 0;
    Block block;
};

[[nodiscard]] inline Block scaled_identity(double c)
{
    Block b;
    for(int i = 0; i < kBlockDim; ++i)
        b.set(i, i, c);
    return b;
}

/// The upper-triangular triplet table of `mode` (block rows 0..kNodes-1).
[[nodiscard]] inline std::vector<HostTriplet> chain_triplets(MatrixMode mode)
{
    std::vector<HostTriplet> ts;
    ts.reserve(kNodes + kNodes - 1);
    for(int i = 0; i < kNodes; ++i)
        ts.push_back({i, i, scaled_identity(diag_coeff(mode, i))});
    if(has_coupling(mode))
    {
        for(int i = 0; i + 1 < kNodes; ++i)
            ts.push_back({i, i + 1, scaled_identity(coupling_coeff(mode))});
    }
    return ts;
}

/// The *dense* matrix the solver applies for `mode`: the symmetric closure of the
/// table, i.e. `A(i,j) = block`, `A(j,i) = block^T` — what `rbk_sym_spmv` does with
/// the upper-packed BCOO.
[[nodiscard]] inline Eigen::MatrixXd dense_A(MatrixMode mode)
{
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(kDofs, kDofs);
    for(const auto& t : chain_triplets(mode))
    {
        for(int r = 0; r < kBlockDim; ++r)
            for(int c = 0; c < kBlockDim; ++c)
            {
                A(t.row * kBlockDim + r, t.col * kBlockDim + c) += t.block(r, c);
                if(t.row != t.col)
                    A(t.col * kBlockDim + c, t.row * kBlockDim + r) += t.block(r, c);
            }
    }
    return A;
}

/// `M^-1` for the fixture's block-Jacobi (a dense 24x24, so the host
/// transcription can apply it exactly like the device kernel does).
[[nodiscard]] inline Eigen::MatrixXd dense_preconditioner_inverse(PrecondMode mode)
{
    Eigen::MatrixXd Minv = Eigen::MatrixXd::Zero(kDofs, kDofs);
    if(mode == PrecondMode::None)
        return Eigen::MatrixXd::Identity(kDofs, kDofs);
    for(int i = 0; i < kNodes; ++i)
    {
        // the wrong mode takes the neighbour's block (still SPD: c > 0 either way)
        const int       src = mode == PrecondMode::WrongBlock ? (i + 1) % kNodes : i;
        Eigen::Matrix3d D   = Eigen::Matrix3d::Zero();
        D.diagonal().setConstant(diag_coeff(MatrixMode::Normal, src));
        Minv.block<3, 3>(i * kBlockDim, i * kBlockDim) = D.inverse();
    }
    return Minv;
}

/// The default right-hand side: dyadic, and deliberately *not* constant.
///
/// A constant `b` is exactly orthogonal to every row sum of the (pinned) chain, so
/// `A*b == 0` and the first `p^T A p == 0` would divide by zero — the same trap the
/// reference would hit on a scene whose Dirichlet rows were forgotten.
[[nodiscard]] inline std::vector<double> default_rhs()
{
    // Alternating sign with a dyadic magnitude: a *smooth* right-hand side (let
    // alone a constant one) is nearly orthogonal to every row sum of a chain
    // operator, so `b^T A b` is tiny and the first `alpha` explodes. Alternating
    // puts b near the largest eigenvector instead -- large `b^T A b`, and the first
    // PCG step reduces `|rz|` by a comfortable factor.
    std::vector<double> b(kDofs);
    for(int i = 0; i < kDofs; ++i)
    {
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        b[static_cast<std::size_t>(i)] =
            sign * (1.0 + 0.125 * static_cast<double>((i * 5 + 3) % 7));
    }
    return b;
}

// ---------------------------------------------------------------------------
//  small helpers (the T54/T83 test files' idioms)
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// `std::printf` + `fflush`: the backend aborts the process on a failed UIPC_ASSERT,
/// and a redirected stdout is fully buffered -- a diagnostic without the flush is lost
/// exactly when it is needed.
#define T82_LOG(...)                                                           \
    do                                                                         \
    {                                                                          \
        std::printf(__VA_ARGS__);                                              \
        std::fflush(stdout);                                                   \
    } while(false)

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
    ~CaseScope()
    {
        T82_LOG("[t82] %s: %d checks\n", name.c_str(), counter.checks);
    }
};

#define T82_CHECK(counter, cond)                                               \
    do                                                                         \
    {                                                                          \
        ++(counter).checks;                                                    \
        CHECK(cond);                                                           \
    } while(false)

#define T82_CHECK_THROWS(counter, expr)                                        \
    do                                                                         \
    {                                                                          \
        ++(counter).checks;                                                    \
        CHECK_THROWS(expr);                                                    \
    } while(false)

#define T82_CHECK_NOTHROW(counter, expr)                                       \
    do                                                                         \
    {                                                                          \
        ++(counter).checks;                                                    \
        CHECK_NOTHROW(expr);                                                   \
    } while(false)

/// `max |a - b|` over two std::vectors (the Class-A "Δ == 0" probe).
[[nodiscard]] inline double max_abs_diff(const std::vector<double>& a,
                                         const std::vector<double>& b)
{
    if(a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double d = 0.0;
    for(std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

/// `max |a - b| / max(1, max |b|)` (the tolerance bands of plans/03 §5).
[[nodiscard]] inline double rel_diff(const std::vector<double>& a,
                                    const std::vector<double>& b)
{
    double scale = 1.0, d = 0.0;
    if(a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    for(std::size_t i = 0; i < b.size(); ++i)
    {
        scale = std::max(scale, std::abs(b[i]));
        d     = std::max(d, std::abs(a[i] - b[i]));
    }
    return d / scale;
}

[[nodiscard]] inline double norm2(const std::vector<double>& v)
{
    double s = 0.0;
    for(double x : v)
        s += x * x;
    return std::sqrt(s);
}

/// `||b - A x|| / ||b||` — the convergence gate the parity harness uses.
[[nodiscard]] inline double relative_residual(const Eigen::MatrixXd& A,
                                              const std::vector<double>& b,
                                              const std::vector<double>& x)
{
    Eigen::Map<const Eigen::VectorXd> bx(b.data(), static_cast<Eigen::Index>(b.size()));
    Eigen::Map<const Eigen::VectorXd> xx(x.data(), static_cast<Eigen::Index>(x.size()));
    Eigen::VectorXd                   r = bx - A * xx;
    return r.norm() / bx.norm();
}

// ===========================================================================
//  2) the host oracle — a transcription of linear_pcg.cu:234-320
// ===========================================================================
struct HostPcg
{
    SizeT               iters = 0;
    std::vector<double> x;
    std::vector<double> trace;  // |rz_k| / |rz_0|, k = 0..iters
    double              rel_residual = 0.0;
};

/// The reference's loop, statement by statement, over Eigen dense vectors.
///
/// The one thing it cannot model is `accuracy_statisfied(r)`: the fixture's
/// `do_accuracy_check` is a no-op, i.e. the device always reports "satisfied", and
/// the transcription therefore uses `true` for that conjunct.
[[nodiscard]] inline HostPcg host_pcg(const Eigen::MatrixXd&     A,
                                      const Eigen::MatrixXd&     Minv,
                                      const std::vector<double>& b,
                                      double                     tol_rate)
{
    const auto n = static_cast<SizeT>(b.size());

    Eigen::Map<const Eigen::VectorXd> bv(b.data(), static_cast<Eigen::Index>(n));
    Eigen::VectorXd                   x = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));

    SizeT k = 0;
    // r = b - A * x, x == 0 -> r = b
    Eigen::VectorXd r = bv;
    Eigen::VectorXd z = Minv * r;
    Eigen::VectorXd p = z;  // p = z

    double rz     = r.dot(z);
    double abs_rz0 = std::abs(rz);

    HostPcg out;
    out.trace.push_back(abs_rz0 == 0.0 ? 0.0 : 1.0);

    const auto max_iter = std::max(static_cast<SizeT>(2.0 * static_cast<double>(n)), SizeT{1});

    if(abs_rz0 == 0.0)
    {
        out.x            = std::vector<double>(x.data(), x.data() + x.size());
        out.rel_residual = 0.0;
        return out;
    }

    for(k = 1; k < max_iter; ++k)
    {
        Eigen::VectorXd Ap    = A * p;
        double          alpha = rz / p.dot(Ap);
        x += alpha * p;
        r -= alpha * Ap;
        z = Minv * r;

        const double rz_new = r.dot(z);
        out.trace.push_back(std::abs(rz_new) / abs_rz0);

        if(std::abs(rz_new) <= tol_rate * abs_rz0)
            break;

        const double beta = rz_new / rz;
        p                 = z + beta * p;
        rz                = rz_new;
    }

    out.iters        = k;
    out.x            = std::vector<double>(x.data(), x.data() + x.size());
    out.rel_residual = relative_residual(A, b, out.x);
    return out;
}

// ===========================================================================
//  3) the scene + the marker gate
// ===========================================================================
[[nodiscard]] inline bool has_marker(uipc::backend::WorldVisitor& world)
{
    for(auto& slot : world.scene().geometries())
    {
        if(auto* sc = slot->geometry().as<SimplicialComplex>())
        {
            if(sc->vertices().find<IndexT>(kMarker))
                return true;
        }
    }
    return false;
}

[[nodiscard]] inline Json base_config()
{
    Json config                      = Scene::default_config();
    config["contact"]["enable"]      = IndexT{0};
    config["sanity_check"]["enable"] = IndexT{0};
    config["integrator"]["type"]     = std::string{"bdf1"};
    config["dt"]                     = 0.01;
    return config;
}

[[nodiscard]] inline Scene build_scene(const Json& config)
{
    Scene scene{config};
    auto  object = scene.objects().create("t82");

    const std::vector<Vector3> positions = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    const std::vector<Vector4i> tets = {{0, 1, 2, 3}};

    auto mesh = tetmesh(span<const Vector3>{positions}, span<const Vector4i>{tets});
    label_surface(mesh);
    // no dof-producing constitution claims this mesh: the whole linear system of
    // this file's scenes is the fixture below
    mesh.vertices().create<IndexT>(kMarker, 1);
    object->geometries().create(mesh);
    return scene;
}

// ===========================================================================
//  4) the fixtures
// ===========================================================================
class T82DiagChain;
class T82PcgPreconditioner;

/// The single diag subsystem: it owns all 24 dofs and writes the chain's upper
/// triangle.  Values are uploaded from the host table (`LBufferView::copy_from` on
/// the window subview) instead of a trace-time kernel: the entries are written
/// *verbatim* either way, and an upload cannot hide an arithmetic difference.
class T82DiagChain final : public gg::DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    static inline bool enabled = false;

    static void reset() { enabled = false; }

  protected:
    [[nodiscard]] U64 get_uid() const noexcept override { return kUidChain; }

    void do_init(InitInfo& info) override
    {
        (void)info;
        enabled = has_marker(world());
    }

    void do_report_init_extent(gg::GlobalLinearSystem::InitDofExtentInfo& info) override
    {
        info.extent(enabled ? kDofs : SizeT{0});
    }

    void do_receive_init_dof_info(gg::GlobalLinearSystem::InitDofInfo& info) override
    {
        (void)info;
    }

    void do_report_extent(gg::GlobalLinearSystem::DiagExtentInfo& info) override
    {
        auto n = enabled ? chain_triplets(g_matrix_mode).size() : SizeT{0};
        info.extent(n, enabled ? kDofs : SizeT{0});
    }

    void do_assemble(gg::GlobalLinearSystem::DiagInfo& info) override
    {
        if(!enabled)
            return;

        auto hessians = info.hessians();
        const auto count =
            static_cast<std::size_t>(hessians.triplet_count());
        const auto ts = chain_triplets(g_matrix_mode);
        if(count != ts.size())
            return;  // the caller's extent/assembly assertions then fail loudly

        // the window's submatrix offset: with one diag subsystem it is (0, 0), and
        // the stored indices are *global* block indices (`TripletMatrixView::write()`
        // adds the offset, `at()` subtracts it)
        const auto offset = hessians.submatrix_offset();
        const auto begin  = static_cast<std::size_t>(hessians.triplet_index_offset());

        std::vector<std::int32_t> rows(count);
        std::vector<std::int32_t> cols(count);
        std::vector<double>       vals(count * 9u, 0.0);
        for(std::size_t i = 0; i < count; ++i)
        {
            rows[i] = static_cast<std::int32_t>(offset.x + ts[i].row);
            cols[i] = static_cast<std::int32_t>(offset.y + ts[i].col);
            for(int c = 0; c < kBlockDim; ++c)
                for(int r = 0; r < kBlockDim; ++r)
                    // flat storage is column-major inside a block, like LMat
                    vals[i * 9u + static_cast<std::size_t>(c * kBlockDim + r)] =
                        ts[i].block(r, c);
        }

        hessians.row_indices()
            .subview(begin, count)
            .copy_from(span<const std::int32_t>{rows});
        hessians.col_indices()
            .subview(begin, count)
            .copy_from(span<const std::int32_t>{cols});
        hessians.values()
            .subview(begin * 9u, count * 9u)
            .copy_from(span<const double>{vals});
    }

    void do_accuracy_check(gg::GlobalLinearSystem::AccuracyInfo& info) override
    {
        // the reference's `AccuracyInfo::satisfied(bool)` defaults to true; this
        // fixture has no Newton-level tolerance to model, so it leaves it alone.
        // Consequence: the device's stopping rule is `|rz_k| <= tol*|rz_0|` alone,
        // which is exactly what the host transcription implements.
        (void)info;
    }

    void do_retrieve_solution(gg::GlobalLinearSystem::SolutionInfo& info) override
    {
        (void)info;
    }

    Float do_diag_norm(gg::GlobalLinearSystem::DiagNormInfo& info) override
    {
        (void)info;
        return 0.0;
    }

    Float do_mass_norm(gg::GlobalLinearSystem::DiagNormInfo& info) override
    {
        (void)info;
        return 0.0;
    }
};

namespace detail
{
/// `z = D^-1 r` over block-diagonal 3x3 blocks (the reference's
/// `FEMDiagPreconditioner_do_apply_kernel` shape, general in the block).
auto& block_jacobi_apply_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,  // r
                                   lcrt::Buffer<double>,  // z
                                   lcrt::Buffer<double>,  // block inverses (column-major 3x3)
                                   lcrt::Buffer<std::int32_t>,  // converged
                                   uint32_t>(  // block count
            [](lcrt::BufferVar<double> r,
               lcrt::BufferVar<double> z,
               lcrt::BufferVar<double> dinv,
               lcrt::BufferVar<std::int32_t> converged,
               lcrt::Var<uint32_t> n) noexcept
            {
                auto b = lcrt::dispatch_x();
                $if(b < n)
                {
                    // the reference's `if(*converged != 0) return;`
                    $if(converged->read(0) == 0)
                    {
                        for(int c = 0; c < kBlockDim; ++c)
                        {
                            lcrt::Var<double> acc =
                                dinv->read(b * 9u + static_cast<uint32_t>(c * kBlockDim))
                                * r->read(b * 3u);
                            for(int rr = 1; rr < kBlockDim; ++rr)
                                acc = acc
                                      + dinv->read(b * 9u
                                                   + static_cast<uint32_t>(c * kBlockDim + rr))
                                            * r->read(b * 3u + static_cast<uint32_t>(rr));
                            z->write(b * 3u + static_cast<uint32_t>(c), acc);
                        }
                    };
                };
            });
    return k;
}
}  // namespace detail

/// A `LocalPreconditioner` bound to `T82DiagChain`.  `PrecondMode::None` (and a scene
/// without the marker) makes it throw `SimSystemException`, i.e. it is *invalid* and
/// never registered — that is how the `z <- r` fallback path is exercised.
class T82PcgPreconditioner final : public gg::LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    static inline bool enabled = false;

    static void reset() { enabled = false; }

    /// The inverse blocks, column-major 3x3 per node (uploaded by `do_assemble`).
    lud::LDeviceBuffer<double> diag_inv;

  protected:
    void do_build(BuildInfo& info) override
    {
        if(!has_marker(world()))
            throw uipc::backend::SimSystemException("T82PcgPreconditioner unused: no marker");
        if(g_precond_mode == PrecondMode::None)
            throw uipc::backend::SimSystemException("T82PcgPreconditioner unused: PrecondMode::None");

        auto& chain = require<T82DiagChain>();
        info.connect(&chain);
        enabled = true;
    }

    void do_init(InitInfo& info) override { (void)info; }

    void do_assemble(gg::GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        // D = blockdiag(A_ii) — the reference's `i == j` diagonal-block selection, but
        // taken from the block table instead of from `info.A()`: the table is the
        // oracle's own input, so a wrong block cannot hide behind a round trip.
        (void)info;
        const auto n = static_cast<std::size_t>(info.dof_count()) / kBlockDim;
        if(n == 0u)
            return;

        std::vector<double> inv(n * 9u, 0.0);
        for(std::size_t i = 0; i < n; ++i)
        {
            const auto src =
                g_precond_mode == PrecondMode::WrongBlock ? (i + 1u) % n : i;
            const double c = diag_coeff(MatrixMode::Normal, static_cast<int>(src));
            for(int d = 0; d < kBlockDim; ++d)
                inv[i * 9u + static_cast<std::size_t>(d * kBlockDim + d)] = 1.0 / c;
        }
        diag_inv.resize(inv.size());
        diag_inv.copy_from(span<const double>{inv});
    }

    void do_apply(gg::GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        auto n = static_cast<uint32_t>(diag_inv.size() / 9u);
        if(n == 0u)
            return;
        auto& k = detail::block_jacobi_apply_kernel();
        gg::frame_stream().record(k(info.r().buffer_view(),
                                info.z().buffer_view(),
                                diag_inv.cview(),
                                info.converged(),
                                n)
                                  .dispatch(n));
    }
};

/// Captures the engine's systems once they are built.
class T82PcgProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::GlobalLinearSystem* gls   = nullptr;
    inline static gg::LinearPCG*          pcg   = nullptr;
    inline static T82DiagChain*           chain = nullptr;
    inline static T82PcgPreconditioner*   precond = nullptr;

    static void reset()
    {
        gls     = nullptr;
        pcg     = nullptr;
        chain   = nullptr;
        precond = nullptr;
    }

  private:
    void do_build() override
    {
        gls   = &require<gg::GlobalLinearSystem>();
        pcg   = &require<gg::LinearPCG>();
        chain = &require<T82DiagChain>();
        // optional: the preconditioner throws `SimSystemException` unless its mode is
        // live (and `require<>` would throw for an invalid system)
        precond = find<T82PcgPreconditioner>();
    }
};
}  // namespace t82

REGISTER_SIM_SYSTEM(t82::T82DiagChain);
REGISTER_SIM_SYSTEM(t82::T82PcgPreconditioner);
REGISTER_SIM_SYSTEM(t82::T82PcgProbe);

namespace t82
{
// ===========================================================================
//  5) the driver: one world, the shipped chain, one solve
// ===========================================================================
struct RunOptions
{
    MatrixMode                 matrix     = MatrixMode::Normal;
    PrecondMode                precond    = PrecondMode::None;
    double                     tol_rate   = 1e-3;
    std::optional<std::string> solver;          // linear_system/solver
    std::optional<IndexT>      use_cuda_graph;  // linear_system/use_cuda_graph
    std::optional<IndexT>      check_interval;  // linear_system/check_interval
    std::optional<IndexT>      dump_linear_pcg; // extras/debug/dump_linear_pcg
    std::optional<std::string> constitution;    // contact/constitution
    /// replaces `default_rhs()`
    const std::vector<double>* b_override = nullptr;
};

struct PcgRun
{
    SizeT               iters = 0;
    std::size_t         dofs  = 0;
    std::size_t         nnz   = 0;
    std::vector<double> b;
    std::vector<double> x;
    std::vector<Float>  trace;
};

/// Build a fresh world for `options` and hand the *engine's* `GlobalLinearSystem`
/// (layout done, systems built, solver registered) to `fn`, with everything alive.
///
/// `fn` gets the same `Impl` that `LinearPCG`'s utility forwards will call back
/// into, which is what makes the shipped chain — and not a parallel copy of it —
/// the thing under test (`GlobalLinearSystem::impl()`, the file's T8.3a-2 deviation).
template <typename Fn>
void with_world(const RunOptions& options, Fn&& fn)
{
    g_matrix_mode  = options.matrix;
    g_precond_mode = options.precond;

    T82PcgProbe::reset();
    T82DiagChain::reset();
    T82PcgPreconditioner::reset();

    Json config = base_config();
    config["linear_system"]["tol_rate"] = options.tol_rate;
    if(options.solver)
        config["linear_system"]["solver"] = *options.solver;
    if(options.use_cuda_graph)
        config["linear_system"]["use_cuda_graph"] = *options.use_cuda_graph;
    if(options.check_interval)
        config["linear_system"]["check_interval"] = *options.check_interval;
    if(options.dump_linear_pcg)
        config["extras"]["debug"]["dump_linear_pcg"] = *options.dump_linear_pcg;
    if(options.constitution)
        config["contact"]["constitution"] = *options.constitution;

    auto scene = build_scene(config);

    Engine engine{"luisa", workspace_path("uipc_luisa_t82_pcg")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T82PcgProbe::gls != nullptr);
    REQUIRE(T82PcgProbe::pcg != nullptr);
    REQUIRE(T82PcgProbe::chain != nullptr);

    auto* gls = T82PcgProbe::gls;
    gls->init();

    fn(*gls, gls->impl());
}

/// `build_linear_system()` -> upload b -> `solve_linear_system()` -> read x back.
[[nodiscard]] inline PcgRun run_pcg(const RunOptions& options)
{
    PcgRun run;
    with_world(options,
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   impl.build_linear_system();
                   run.dofs = impl.b.size();
                   run.nnz  = impl.bcoo_A.triplet_count();

                   run.b = options.b_override ? *options.b_override : default_rhs();
                   REQUIRE(run.b.size() == run.dofs);
                   impl.b.copy_from(span<const double>{run.b});

                   impl.solve_linear_system();
                   run.iters = gls.last_solve_iterations();
                   run.trace = T82PcgProbe::pcg->residual_trace();

                   run.x.resize(run.dofs);
                   impl.x.copy_to(span<double>{run.x});
               });
    return run;
}

/// The worst relative deviation of two residual curves (either element type: the
/// device curve is `Float`, the oracle's is `double`).
template <typename A, typename B>
[[nodiscard]] inline double trace_rel_diff(const std::vector<A>& a, const std::vector<B>& b)
{
    if(a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for(std::size_t i = 0; i < b.size(); ++i)
    {
        const double denom = std::max(std::abs(static_cast<double>(b[i])), 1e-300);
        worst              = std::max(worst,
                         std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]))
                             / denom);
    }
    return worst;
}

[[nodiscard]] inline double trace_max_rel_monotone_violation(const std::vector<Float>& t)
{
    double worst = 0.0;
    for(std::size_t i = 1; i < t.size(); ++i)
        worst = std::max(worst,
                         static_cast<double>(t[i]) / std::max(double{t[i - 1]}, 1e-300));
    return worst;
}
}  // namespace t82

namespace t82
{
// ===========================================================================
//  CASE 1 — the headline: the device loop vs the host transcription
// ===========================================================================
TEST_CASE("pcg_converges_vs_host", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_converges_vs_host"};

    const auto  A      = dense_A(MatrixMode::Normal);
    const auto  Minv   = dense_preconditioner_inverse(PrecondMode::None);
    const auto  b      = default_rhs();
    const double tol    = 1e-3;  // the schema default
    const auto  oracle = host_pcg(A, Minv, b, tol);

    // ---- the fixture really is SPD -----------------------------------------
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    REQUIRE(es.info() == Eigen::Success);
    const double lmin = es.eigenvalues().minCoeff();
    const double lmax = es.eigenvalues().maxCoeff();
    T82_LOG("[t82] system: %d dofs, lambda_min %.6e, lambda_max %.6e, cond %.3e\n",
                kDofs,
                lmin,
                lmax,
                lmax / lmin);

    const auto run = run_pcg({.matrix = MatrixMode::Normal, .precond = PrecondMode::None});

    auto& c = scope.counter;
    T82_CHECK(c, run.dofs == static_cast<std::size_t>(kDofs));
    // 8 diagonal blocks + 7 coupling blocks, all upper (row <= col)
    T82_CHECK(c, run.nnz == 15u);
    T82_CHECK(c, lmin > 0.0);  // SPD: Cholesky-equivalent via the spectrum
    T82_CHECK(c, std::abs(A(A.rows() - 1, 0)) == 0.0);  // not accidentally dense

    // ---- the iteration count and the residual curve -------------------------
    const double worst_trace = trace_rel_diff(run.trace, oracle.trace);
    const double dr = relative_residual(A, b, run.x);
    T82_LOG(
        "[t82] pcg: device iters %zu, host iters %zu, |dr| %.3e, trace worst rel "
        "%.3e, ||b-Ax||/||b|| device %.3e host %.3e\n",
        static_cast<std::size_t>(run.iters),
        static_cast<std::size_t>(oracle.iters),
        rel_diff(run.x, oracle.x),
        worst_trace,
        dr,
        oracle.rel_residual);

    T82_CHECK(c, run.iters == oracle.iters);
    T82_CHECK(c, run.iters > 1);          // non-trivial: not the 1-step exact case
    T82_CHECK(c, run.iters < 2 * kDofs);  // converged before the cap
    T82_CHECK(c, run.trace.size() == run.iters + 1);
    T82_CHECK(c, worst_trace <= 1e-12);
    T82_CHECK(c, rel_diff(run.x, oracle.x) <= 1e-10);
    // The reference's criterion is `|rz| <= tol_rate*|rz_0|`, and with no
    // preconditioner `rz = r^T r = ||r||^2`, so the *derived* bound on
    // `||r||/||b||` is sqrt(tol_rate) -- not tol_rate (the research note's table
    // states the latter; measured below).
    T82_CHECK(c, dr <= std::sqrt(tol));
    T82_CHECK(c, oracle.rel_residual <= std::sqrt(tol));
    // `|rz_k|` is non-increasing for SPD CG (a bump is a rounding-level event)
    T82_CHECK(c, trace_max_rel_monotone_violation(run.trace) <= 1.0 + 1e-12);
    T82_CHECK(c, std::abs(run.trace.front() - 1.0) == 0.0);

    // ---- negative control: a broken operator cannot pass this gate -----------
    // The same system re-assembled with the coupling blocks dropped: PCG still
    // "converges" on the diagonal matrix it was given, but against the *true* A the
    // residual is orders of magnitude above tol_rate, and the solution is wrong.
    const auto broken =
        run_pcg({.matrix = MatrixMode::BrokenCoupling, .precond = PrecondMode::None});
    const double broken_residual = relative_residual(A, b, broken.x);
    T82_LOG(
        "[t82] broken-SpMV control: iters %zu, ||b-A x||/||b|| %.3e (gate %.1e), "
        "|dx|/|x| %.3e\n",
        static_cast<std::size_t>(broken.iters),
        broken_residual,
        tol,
        rel_diff(broken.x, oracle.x));

    T82_CHECK(c, broken_residual > 100.0 * tol);  // the gate itself is what fails
    T82_CHECK(c, rel_diff(broken.x, oracle.x) > 1e-2);
    T82_CHECK(c, broken.iters != run.iters);
}

// ===========================================================================
//  CASE 2 — the first iteration is exact (Class A)
// ===========================================================================
TEST_CASE("pcg_first_iteration_bitexact", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_first_iteration_bitexact"};

    const auto A    = dense_A(MatrixMode::Normal);
    const auto Minv = dense_preconditioner_inverse(PrecondMode::None);
    const auto b    = default_rhs();

    // tol_rate close to 1 makes the loop stop at k = 1: then the solution *is* x_1
    // and the curve is [1, |rz_1|/|rz_0|], i.e. every Class-A quantity is observable
    // through the public surface without touching the solver's internals.
    const double tol    = 0.99;
    const auto   oracle = host_pcg(A, Minv, b, tol);
    REQUIRE(oracle.iters == 1u);

    const auto run = run_pcg({.matrix = MatrixMode::Normal,
                              .precond = PrecondMode::None,
                              .tol_rate = tol});

    auto& c = scope.counter;
    const double dx1  = max_abs_diff(run.x, oracle.x);
    const double dt1  = run.trace.size() == oracle.trace.size() ?
                            std::abs(static_cast<double>(run.trace[1]) - oracle.trace[1]) :
                            std::numeric_limits<double>::infinity();
    T82_LOG(
        "[t82] first iteration: iters %zu, |dx_1| %.3e, |d(rz_1/rz_0)| %.3e "
        "(rz_1/rz_0 %.6e)\n",
        static_cast<std::size_t>(run.iters),
        dx1,
        dt1,
         static_cast<double>(run.trace[1]));

    T82_CHECK(c, run.iters == 1u);
    // x_1 = alpha * b: exact for this dyadic table (no FMA-sensitive accumulation)
    T82_CHECK(c, dx1 == 0.0);
    T82_CHECK(c, dt1 <= 1e-15);
    T82_CHECK(c, run.trace[1] < tol);  // that is why it stopped here

    // The same first step computed from the plan's Class-A ingredients only:
    // z = r (no preconditioner), Ap = A*b, rz = r^T z, alpha = rz / (b^T Ap).
    Eigen::Map<const Eigen::VectorXd> bv(b.data(), static_cast<Eigen::Index>(b.size()));
    const Eigen::VectorXd             Ab   = A * bv;
    const double                      rz0  = bv.dot(bv);
    const double                      alpha = rz0 / bv.dot(Ab);
    std::vector<double>               x1(b.size(), 0.0);
    for(std::size_t i = 0; i < b.size(); ++i)
        x1[i] = alpha * b[i];

    T82_LOG("[t82] first iteration: |x_1(task-loop) - x_1(alpha*b)| %.3e, alpha %.6e\n",
                max_abs_diff(run.x, x1),
                alpha);
    T82_CHECK(c, max_abs_diff(run.x, x1) <= 1e-15);
}

// ===========================================================================
//  CASE 3 — spmv_dot: Class A on the first iteration, Class B afterwards
// ===========================================================================
TEST_CASE("pcg_spmv_dot_consistency", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_spmv_dot_consistency"};

    const auto A = dense_A(MatrixMode::Normal);
    const auto b = default_rhs();

    with_world({.matrix = MatrixMode::Normal, .precond = PrecondMode::None},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   auto& c = scope.counter;
                   impl.build_linear_system();
                   impl.b.copy_from(span<const double>{b});

                   // ---- Class A: x = b (the first PCG iterate's direction) -----
                   lla::LDenseVector<double> y;
                   y.resize(kDofs);
                   lud::LDeviceVar<double> d_dot;

                   impl.spmv_dot(impl.b.cview(), y.view(), d_dot.view());
                   gg::frame_stream().synchronize();
                   const double dot_b = d_dot.copy_to();

                   std::vector<double> y_host(kDofs, 0.0);
                   y.copy_to(span<double>{y_host});

                   Eigen::Map<const Eigen::VectorXd> bv(b.data(),
                                                        static_cast<Eigen::Index>(b.size()));
                   const Eigen::VectorXd             Ab = A * bv;
                   const double                      dot_b_host = bv.dot(Ab);

                   std::vector<double> ap_host(Ab.data(), Ab.data() + Ab.size());
                   const double        d_ap = max_abs_diff(y_host, ap_host);

                   T82_LOG(
                       "[t82] spmv_dot (x = b): |A*b - host| %.3e, |d_dot - b^T A b| %.3e "
                       "(b^T A b %.6e)\n",
                       d_ap,
                       std::abs(dot_b - dot_b_host),
                       dot_b_host);
                   // A*b and b^T(A b) are exact for this dyadic table
                   T82_CHECK(c, d_ap == 0.0);
                   T82_CHECK(c, std::abs(dot_b - dot_b_host) <= 1e-12);

                   // ---- Class B afterwards: a generic p -----------------------
                   std::vector<double> p(kDofs);
                   for(int i = 0; i < kDofs; ++i)
                       p[static_cast<std::size_t>(i)] =
                           0.5 * static_cast<double>(i + 1) - 3.0 + 0.25 * std::sin(i);

                   lla::LDenseVector<double> p_dev, ap_dev, ap_ref;
                   p_dev.resize(kDofs);
                   ap_dev.resize(kDofs);
                   ap_ref.resize(kDofs);
                   p_dev.copy_from(span<const double>{p});

                   lud::LDeviceVar<double> d_dot_p;
                   impl.spmv_dot(p_dev.cview(), ap_dev.view(), d_dot_p.view());
                   gg::frame_stream().synchronize();
                   const double dot_p = d_dot_p.copy_to();

                   // the reference's *other* pairing (LinearPCG's): spmv + ctx.dot
                   impl.spmv(1.0, p_dev.cview(), 0.0, ap_ref.view());
                   const double dot_p_ref =
                       gls.ctx().dot(p_dev.cview(), ap_ref.cview());

                   std::vector<double> y1(kDofs, 0.0), y2(kDofs, 0.0);
                   ap_dev.copy_to(span<double>{y1});
                   ap_ref.copy_to(span<double>{y2});

                   T82_LOG(
                       "[t82] spmv_dot (generic p): |d_dot - dot(spmv)| %.3e rel, "
                       "|A*p fused - A*p plain| %.3e rel\n",
                       std::abs(dot_p - dot_p_ref) / std::max(std::abs(dot_p_ref), 1e-300),
                       rel_diff(y1, y2));

                   // never bitwise: the fused kernel sums in a different tree
                   T82_CHECK(c,
                             std::abs(dot_p - dot_p_ref)
                                     / std::max(std::abs(dot_p_ref), 1e-300)
                                 <= 1e-12);
                   T82_CHECK(c, rel_diff(y1, y2) <= 1e-12);
               });
}

// ===========================================================================
//  CASE 4 — no preconditioner: the `z <- r` fallback
// ===========================================================================
TEST_CASE("pcg_no_preconditioner", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_no_preconditioner"};

    const auto A    = dense_A(MatrixMode::Normal);
    const auto Minv = dense_preconditioner_inverse(PrecondMode::None);
    const auto b    = default_rhs();
    const double tol = 1e-3;

    const auto oracle = host_pcg(A, Minv, b, tol);

    const auto run = run_pcg({.matrix = MatrixMode::Normal, .precond = PrecondMode::None});

    auto&        c = scope.counter;
    const double worst_trace = trace_rel_diff(run.trace, oracle.trace);
    T82_LOG(
        "[t82] no-precond: device iters %zu, host iters %zu, trace worst rel %.3e, "
        "x rel %.3e\n",
        static_cast<std::size_t>(run.iters),
        static_cast<std::size_t>(oracle.iters),
        worst_trace,
        rel_diff(run.x, oracle.x));

    // the fixture really took the fallback: the mode is `None`, so the
    // `LocalPreconditioner` threw `SimSystemException` and never registered
    T82_CHECK(c, T82PcgProbe::precond == nullptr);
    T82_CHECK(c, run.iters == oracle.iters);
    T82_CHECK(c, worst_trace <= 1e-12);
    T82_CHECK(c, rel_diff(run.x, oracle.x) <= 1e-10);
    T82_CHECK(c, relative_residual(A, b, run.x) <= std::sqrt(tol));
}

// ===========================================================================
//  CASE 5 — the preconditioner hook: block-Jacobi, its content matters
// ===========================================================================
TEST_CASE("pcg_diag_preconditioner", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_diag_preconditioner"};

    const auto A    = dense_A(MatrixMode::Normal);
    const auto b    = default_rhs();
    const double tol = 1e-3;

    const auto none = run_pcg({.matrix = MatrixMode::Normal, .precond = PrecondMode::None});
    const auto diag =
        run_pcg({.matrix = MatrixMode::Normal, .precond = PrecondMode::BlockJacobi});
    const auto wrong =
        run_pcg({.matrix = MatrixMode::Normal, .precond = PrecondMode::WrongBlock});

    const auto oracle_none = host_pcg(A, dense_preconditioner_inverse(PrecondMode::None), b, tol);
    const auto oracle_diag =
        host_pcg(A, dense_preconditioner_inverse(PrecondMode::BlockJacobi), b, tol);

    auto&        c           = scope.counter;
    const double worst_trace = trace_rel_diff(diag.trace, oracle_diag.trace);
    T82_LOG(
        "[t82] preconditioner: iters none %zu, block-Jacobi %zu, wrong %zu; "
        "host none %zu, host block-Jacobi %zu; trace worst rel %.3e, x rel %.3e\n",
        static_cast<std::size_t>(none.iters),
        static_cast<std::size_t>(diag.iters),
        static_cast<std::size_t>(wrong.iters),
        static_cast<std::size_t>(oracle_none.iters),
        static_cast<std::size_t>(oracle_diag.iters),
        worst_trace,
        rel_diff(diag.x, oracle_diag.x));

    // the hook is live (the fixture registered) and it is not the identity
    T82_CHECK(c, diag.iters == oracle_diag.iters);
    T82_CHECK(c, diag.iters < none.iters);  // block-Jacobi helps: discriminating
    T82_CHECK(c, worst_trace <= 1e-12);
    T82_CHECK(c, rel_diff(diag.x, oracle_diag.x) <= 1e-10);
    T82_CHECK(c, relative_residual(A, b, diag.x) <= std::sqrt(tol));
    // a *wrong* preconditioner is not a rounding-level event: the count moves
    T82_CHECK(c, wrong.iters != diag.iters);
    T82_CHECK(c, none.iters == oracle_none.iters);
}

// ===========================================================================
//  CASE 6 — the config selection: all five cases of plan phase 3
// ===========================================================================
TEST_CASE("pcg_config_selection", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_config_selection"};
    auto&     c = scope.counter;

    // (1) no key at all: the schema's default is "fused_pcg"
    const auto default_run = run_pcg({.matrix = MatrixMode::Normal});
    // (2) the reference's exact-count spelling
    const auto linear_run =
        run_pcg({.matrix = MatrixMode::Normal, .solver = std::string{"linear_pcg"}});
    // (3) the fused spelling -> the same class, the same numbers
    const auto fused_run =
        run_pcg({.matrix = MatrixMode::Normal, .solver = std::string{"fused_pcg"}});
    // (5) use_cuda_graph + the AL-IPC constitution: validated, and inert
    const auto graph_run = run_pcg({.matrix = MatrixMode::Normal,
                                    .solver = std::string{"linear_pcg"},
                                    .use_cuda_graph = IndexT{2},
                                    .check_interval = IndexT{7},
                                    .constitution = std::string{"al-ipc"}});

    T82_LOG(
        "[t82] config: iters default %zu, \"linear_pcg\" %zu, \"fused_pcg\" %zu, "
        "graph=2/al-ipc %zu\n",
        static_cast<std::size_t>(default_run.iters),
        static_cast<std::size_t>(linear_run.iters),
        static_cast<std::size_t>(fused_run.iters),
        static_cast<std::size_t>(graph_run.iters));

    // the observable contract of the fused mapping: one class, one count, and
    // use_cuda_graph/check_interval change no number
    T82_CHECK(c, default_run.iters == linear_run.iters);
    T82_CHECK(c, fused_run.iters == linear_run.iters);
    T82_CHECK(c, graph_run.iters == linear_run.iters);
    T82_CHECK(c, max_abs_diff(default_run.x, linear_run.x) == 0.0);
    T82_CHECK(c, max_abs_diff(fused_run.x, linear_run.x) == 0.0);
    T82_CHECK(c, max_abs_diff(graph_run.x, linear_run.x) == 0.0);
    T82_CHECK(c, trace_rel_diff(fused_run.trace, linear_run.trace) == 0.0);
    T82_CHECK(c, trace_rel_diff(graph_run.trace, linear_run.trace) == 0.0);

    // (4) a value outside the schema's enum is rejected at the frontend, and the
    // plan's spelling "pcg" is *not* an accepted value (the schema says
    // {"fused_pcg", "linear_pcg"} — research §1(a))
    {
        auto good                       = base_config();
        good["linear_system"]["solver"] = std::string{"linear_pcg"};
        T82_CHECK_NOTHROW(c, build_scene(good).validate_config());

        auto bad                        = base_config();
        bad["linear_system"]["solver"]  = std::string{"pcg"};
        T82_CHECK_THROWS(c, build_scene(bad).validate_config());

        auto bad2                       = base_config();
        bad2["linear_system"]["solver"] = std::string{"not_a_solver"};
        T82_CHECK_THROWS(c, build_scene(bad2).validate_config());

        // use_cuda_graph is an enum {0,1,2} at the frontend too
        auto bad3                              = base_config();
        bad3["linear_system"]["use_cuda_graph"] = IndexT{3};
        T82_CHECK_THROWS(c, build_scene(bad3).validate_config());
    }
}

// ===========================================================================
//  CASE 7 — the batching counters (P5): one solve, one open frame list
// ===========================================================================
TEST_CASE("pcg_batching_counters", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_batching_counters"};

    const auto b = default_rhs();

    with_world({.matrix = MatrixMode::Normal, .precond = PrecondMode::BlockJacobi},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   auto& c = scope.counter;
                   impl.build_linear_system();
                   impl.b.copy_from(span<const double>{b});

                   // solve #1 is the JIT warm-up: the LSpmv/blas/update_* shaders and
                   // the fixture's preconditioner kernel are compiled here
                   impl.solve_linear_system();
                   const auto iters       = gls.last_solve_iterations();
                   const auto compiles_0  = lud::compile_stats::total();

                   auto& s  = gg::frame_stream();
                   s.reset_counters();
                   const auto before = s.counters();

                   s.begin();
                   impl.solve_linear_system();
                   s.commit();

                   const auto delta    = s.counters() - before;
                   const auto compiles = lud::compile_stats::total() - compiles_0;

                   T82_LOG(
                       "[t82] batching: iters %zu, commands %llu, commits %llu, "
                       "direct_submits %llu, syncs %llu, uploads %llu, compiles %llu\n",
                       static_cast<std::size_t>(iters),
                       static_cast<unsigned long long>(delta.commands),
                       static_cast<unsigned long long>(delta.commits),
                       static_cast<unsigned long long>(delta.direct_submits),
                       static_cast<unsigned long long>(delta.syncs),
                       static_cast<unsigned long long>(delta.uploads),
                       static_cast<unsigned long long>(compiles));

                   T82_CHECK(c, iters > 1);
                   // a host decision needs a host value: commands > commits, but
                   // commits are O(iters) -- never "one submit per solve"
                   T82_CHECK(c, delta.commands > delta.commits);
                   T82_CHECK(c, delta.commits <= 4ull * static_cast<unsigned long long>(iters) + 4ull);
                   // two host read-backs per iteration (alpha, rz_new)
                   T82_CHECK(c, delta.syncs >= static_cast<unsigned long long>(iters));
                   // the `d_converged = 0` upload is the reference's one per solve
                   T82_CHECK(c, delta.uploads >= 1ull);
                   // OP-A4: nothing is recompiled after the first solve
                   T82_CHECK(c, compiles == 0ull);
               });
}

// ===========================================================================
//  CASE 8 — the solver surface: registration, spmv, matrix_data_ptrs
// ===========================================================================
TEST_CASE("pcg_impl_surface", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_impl_surface"};

    const auto A = dense_A(MatrixMode::Normal);
    const auto b = default_rhs();

    with_world({.matrix = MatrixMode::Normal, .precond = PrecondMode::None},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   auto& c = scope.counter;

                   // ---- registration: `add_solver()` really stored the solver ------
                   auto* slot = impl.iterative_solver.view();
                   T82_CHECK(c, slot != nullptr);
                   T82_CHECK(c, slot == static_cast<gg::IterativeSolver*>(T82PcgProbe::pcg));
                   T82_CHECK(c, T82PcgProbe::pcg->is_valid());

                   // ---- the layout -------------------------------------------------
                   impl.build_linear_system();
                   T82_CHECK(c, gls.dof_count() == static_cast<SizeT>(kDofs));
                   T82_CHECK(c, impl.empty_system == false);
                   T82_CHECK(c, impl.bcoo_A.triplet_count() == 15u);
                   // `triplet_count_dev` is uploaded by `build_linear_system()`
                   T82_CHECK(c, impl.triplet_count_dev.copy_to() == IndexT{15});

                   // ---- Impl::spmv == the symmetric operator it claims to be ------
                   impl.b.copy_from(span<const double>{b});
                   lla::LDenseVector<double> y;
                   y.resize(kDofs);
                   impl.spmv(1.0, impl.b.cview(), 0.0, y.view());
                   gg::frame_stream().synchronize();

                   std::vector<double> y_host(kDofs, 0.0);
                   y.copy_to(span<double>{y_host});

                   Eigen::Map<const Eigen::VectorXd> bv(b.data(),
                                                        static_cast<Eigen::Index>(b.size()));
                   const Eigen::VectorXd Ab = A * bv;
                   const std::vector<double> ab_host(Ab.data(), Ab.data() + Ab.size());
                   const double              d = max_abs_diff(y_host, ab_host);
                   T82_LOG("[t82] surface: |Impl::spmv(b) - A*b| %.3e (exact table)\n",
                               d);
                   T82_CHECK(c, d == 0.0);

                   // ---- matrix_data_ptrs() is an identity token, and stable -------
                   const auto p1 = impl.matrix_data_ptrs();
                   const auto p2 = impl.matrix_data_ptrs();
                   T82_CHECK(c, p1[0] == p2[0]);
                   T82_CHECK(c, p1[1] == p2[1]);
                   T82_CHECK(c, p1[0] != nullptr);
                   T82_CHECK(c, p1[1] != nullptr);
                   T82_CHECK(c, p1[2] == nullptr);
                   T82_CHECK(c, p1[0] == static_cast<const void*>(&impl.bcoo_A));

                   // ---- last_solve_iterations lives on the *system* ---------------
                   T82_CHECK(c, gls.last_solve_iterations() == 0u);
                   impl.solve_linear_system();
                   const auto iters = gls.last_solve_iterations();
                   T82_LOG("[t82] surface: iters %zu, last_solve_iterations %zu\n",
                               static_cast<std::size_t>(iters),
                               static_cast<std::size_t>(gls.last_solve_iterations()));
                   T82_CHECK(c, iters > 1);
                   T82_CHECK(c, iters < 2 * kDofs);
                   T82_CHECK(c, iters == impl.last_solve_iterations);
                   T82_CHECK(c, T82PcgProbe::pcg->residual_trace().size() == iters + 1);
               });
}

// ===========================================================================
//  CASE 9 — the NaN/Inf guards, driven through their factored predicate
// ===========================================================================
TEST_CASE("pcg_nan_guard", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_nan_guard"};
    auto&     c = scope.counter;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    // the guard's trigger
    T82_CHECK(c, gg::linear_pcg_detail::rz_nan_inf(nan));
    T82_CHECK(c, gg::linear_pcg_detail::rz_nan_inf(inf));
    T82_CHECK(c, gg::linear_pcg_detail::rz_nan_inf(-inf));
    T82_CHECK(c, !gg::linear_pcg_detail::rz_nan_inf(1.0));
    T82_CHECK(c, !gg::linear_pcg_detail::rz_nan_inf(0.0));

    // the reference's message shape (the guard aborts, so the *text* is factored out
    // and pinned here — cookbook §7.22)
    const auto init_gradient =
        gg::linear_pcg_detail::init_message(3, 2, nan, nan, 1.0);
    const auto init_precond =
        gg::linear_pcg_detail::init_message(3, 2, nan, 1.0, nan);
    const auto iter_precond = gg::linear_pcg_detail::iter_message(3, 2, 7, nan, 1.0, nan);
    const auto iter_diverge = gg::linear_pcg_detail::iter_message(3, 2, 7, nan, nan, 1.0);

    T82_LOG("[t82] guard: %s\n", init_gradient.c_str());

    T82_CHECK(c, init_gradient.starts_with("Frame 3, Newton 2, PCG Init: r^T*z = nan,"));
    T82_CHECK(c,
              init_gradient.find("gradient assembling produced NaN values, likely due to "
                                 "error in formula implementation")
                  != std::string::npos);
    T82_CHECK(c,
              init_precond.find("preconditioner failed, likely due to inverse matrix "
                                "calculation failure")
                  != std::string::npos);
    T82_CHECK(c, iter_precond.starts_with("Frame 3, Newton 2, PCG Iter 7: r^T*z = nan,"));
    T82_CHECK(c,
              iter_precond.find("preconditioner failed, likely due to inverse matrix "
                                "calculation failure")
                  != std::string::npos);
    T82_CHECK(c, iter_diverge.find("PCG iteration diverged") != std::string::npos);
}

// ===========================================================================
//  CASE 10 — the one negative control that really cannot converge
// ===========================================================================
TEST_CASE("pcg_singular_does_not_converge", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    CaseScope scope{"pcg_singular_does_not_converge"};

    // `ZeroLastBlock` zeroes the last node's diagonal block *and* drops the coupling
    // to it, so A = blockdiag(SPD, 0): singular, with the last three dofs in its null
    // space.  `b` has a non-zero component there, i.e. b is not in the range of A and
    // **no** solution exists.  PCG cannot satisfy `|rz| <= tol*|rz_0|` (that component
    // of the residual never changes), so it burns exactly `max_iter = 2n` iterations
    // and the reference's `if(iter >= max_iter) logger::warn(...)` fires.
    //
    // DEVIATION (reported): the research's control #2 was "wrong b does not
    // converge", and #3 was "non-SPD rejected".  Neither is realisable as stated,
    // and the measurements below say why:
    //   * an SPD A is nonsingular, so *every* b converges -- a "wrong b" can only be
    //     discriminating against a singular operator (this case);
    //   * negating a whole symmetric table leaves PCG *convergent*: A -> -A and
    //     M -> -M only flip the sign of alpha, so the count and the residual curve
    //     are unchanged (measured, second half of this case).  "Non-SPD" is
    //     therefore not a failure mode of this algorithm, and the two research
    //     controls collapse into this one.
    const double tol      = 1e-3;
    const auto   A        = dense_A(MatrixMode::ZeroLastBlock);
    const auto   b        = default_rhs();
    const auto   max_iter = static_cast<SizeT>(2 * kDofs);

    const auto run1 = run_pcg({.matrix = MatrixMode::ZeroLastBlock, .tol_rate = tol});
    const auto run2 = run_pcg({.matrix = MatrixMode::ZeroLastBlock, .tol_rate = tol});

    auto&        c        = scope.counter;
    const double residual = relative_residual(A, b, run1.x);

    T82_LOG(
        "[t82] singular: iters %zu/%zu (max_iter %zu), ||b-Ax||/||b|| %.3e, "
        "last trace %.3e, trace identical %d\n",
        static_cast<std::size_t>(run1.iters),
        static_cast<std::size_t>(run2.iters),
        static_cast<std::size_t>(max_iter),
        residual,
        static_cast<double>(run1.trace.back()),
        static_cast<int>(run1.trace == run2.trace));

    T82_CHECK(c, run1.iters == max_iter);  // the `reached max_iter` warning path
    // one curve entry per *completed* iteration plus the initial 1.0: a loop that
    // breaks at k has k+1 entries, an exhausted loop (k = max_iter-1 is the last one
    // executed) has max_iter
    T82_CHECK(c, run1.trace.size() == max_iter);
    T82_CHECK(c, run1.trace.back() > tol);  // the gate really fails
    T82_CHECK(c, residual > tol);
    T82_CHECK(c, std::isfinite(residual));
    // deterministic: the same scene, twice, gives the same count and the same curve
    T82_CHECK(c, run1.iters == run2.iters);
    T82_CHECK(c, run1.trace == run2.trace);

    // ---- why "non-SPD" is not a control: PCG is sign-invariant ----------------
    // measured against case 1's run: identical count, identical curve, x negated
    const auto spd  = run_pcg({.matrix = MatrixMode::Normal, .tol_rate = tol});
    const auto negd = run_pcg({.matrix = MatrixMode::NegativeDefinite, .tol_rate = tol});
    const auto neg_A = dense_A(MatrixMode::NegativeDefinite);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(neg_A);
    REQUIRE(es.info() == Eigen::Success);
    std::vector<double> minus_x(spd.x.size(), 0.0);
    for(std::size_t i = 0; i < spd.x.size(); ++i)
        minus_x[i] = -spd.x[i];
    const double d_sign = max_abs_diff(negd.x, minus_x);
    T82_LOG(
        "[t82] sign invariance: SPD iters %zu, negative-definite iters %zu, "
        "lambda_max(neg) %.3e, trace identical %d, |x(-A) - (-x(A))| %.3e\n",
        static_cast<std::size_t>(spd.iters),
        static_cast<std::size_t>(negd.iters),
        es.eigenvalues().maxCoeff(),
        static_cast<int>(spd.trace == negd.trace),
        d_sign);
    T82_CHECK(c, es.eigenvalues().maxCoeff() < 0.0);  // really negative definite
    T82_CHECK(c, negd.iters == spd.iters);
    T82_CHECK(c, negd.trace == spd.trace);
    T82_CHECK(c, d_sign <= 1e-12);
    T82_CHECK(c, relative_residual(neg_A, b, negd.x) <= std::sqrt(tol));
}

// ===========================================================================
//  CASE 12 — plan §6.1's Cross-backend gate: NOT REACHABLE here
// ===========================================================================
TEST_CASE("pcg_from_mtx", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    SKIP(
        "cuda backend unavailable: --backend_cuda=y fails at configure time "
        "(src/backends/cuda/components.lua:86: unknown interface: raise()), so there is "
        "no CUDA ground-truth A.mtx/b.mtx; and this backend has no _dump_A_b / "
        "matrix_market writer yet (plan 11 phase 2/4). The Class-B residual-curve gate "
        "is covered in-process by pcg_converges_vs_host against a host transcription of "
        "linear_pcg.cu:234-320.");
}

// ===========================================================================
//  CASE 13 — plan §6.2's frame-level gate: BLOCKED
// ===========================================================================
TEST_CASE("pcg_frame_pcg_two_block", "[luisa][pcg]")
{
    using namespace t82;
    test_runtime();
    SKIP(
        "BLOCKED-BY-S0.6 and plan 10/11: multi-frame world.advance() deadlocks after "
        "frame 1 (pre-existing defect, AGENT_ENV_BRIEF), and the engine never calls "
        "GlobalLinearSystem::solve() yet, so there is no per-frame "
        "linear_solver_iterations to compare. The per-solve counters that do not need "
        "the engine loop are asserted by pcg_batching_counters.");
}
}  // namespace t82
