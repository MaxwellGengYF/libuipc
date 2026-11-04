/********************************************************************
 * @file luisa_global_preconditioner_test.cpp
 * @brief T8.3a-3 — `GlobalPreconditioner` + the global branch of
 *        `GlobalLinearSystem::Impl::{_assemble_preconditioner, apply_preconditioner}`.
 *
 * WHAT IS UNDER TEST (the reference, src/backends/cuda/linear_system/)
 * -------------------------------------------------------------------
 *  * `GlobalPreconditioner` — the class itself, with the reference's surface
 *    (`BuildInfo`, the three pure `do_*`, `do_build() final override` ->
 *    `require<GlobalLinearSystem>()` -> `do_build(info)` -> `add_preconditioner(this)`,
 *    and the two private thunks with **no** `check_state`);
 *  * `GlobalLinearSystem::add_preconditioner(GlobalPreconditioner*)` — a real
 *    `register_sim_system` into the one optional slot (it used to be a no-op);
 *  * `Impl::_assemble_preconditioner()` — the global fan-out runs **first** and the
 *    local loop is unconditional (global_linear_system.cu:422-435);
 *  * `Impl::apply_preconditioner()` — the global apply runs **first**, over the whole
 *    z/r; the local fan-out then overwrites its own dof windows; and
 *    `!global_preconditioner` gates **only** the `z <- r` fallback (.cu:466-510).
 *
 * The plan's §3.3 ("global instead of the local fan-out") is a misreading of the
 * reference; case 5 below is the pin that falsifies it (both run, in that order).
 *
 * THE SYSTEM
 * ----------
 * Two `DiagLinearSubsystem`s (so a *global* preconditioner is not the same object as
 * the union of the local ones): window 0 owns block rows 0..3, window 1 owns 4..7, so
 * the assembled system is 24 dofs / 14 upper-triangular triplets.  Each window is a
 * 3-dof-per-node chain: `A_ii = c_i * I` with `c = (2,3,4,6 | 8,12,16,24)` — all
 * dyadic, heterogeneous (a uniform chain would make block-Jacobi a scalar multiple of
 * the identity, and PCG is invariant under `M -> c*M`) — and `A_{i,i+1} = -I`.  No
 * off-diag subsystem: the two windows are not coupled, which is exactly what makes the
 * *precedence* observable (window 1 has no local preconditioner in case 5, so only the
 * global branch can have written it).
 *
 * All values are exactly representable and the per-block arithmetic is a single
 * 3x3-by-3 product, so the Class-A gates compare with `Δ == 0` rather than a
 * tolerance.  `ge2sym` keeps `row <= col` (cookbook §7.39 rule 5), so the host table
 * writes only the upper triangle; the solver applies the symmetric closure.
 *
 * ORACLES
 * -------
 *  * **the factored routines themselves** — `fem_diag_preconditioner_detail::
 *    {diag_inv_row, diag_apply_row}` over `storage::memory` views of the *assembled*
 *    `bcoo_A` arrays, read back with `LBCOOMatrix::copy_to_{row_indices,col_indices,
 *    values}`.  Not a re-implementation and not Eigen (research §8 item 7): the device
 *    preconditioner and the oracle run the same code on the same numbers, which is why
 *    `Δ == 0` is the honest gate;
 *  * `lps::inverse` for the row-sum oracle (the device kernel computes the same
 *    accumulation in the same order, then calls the same `lps::inverse`);
 *  * the shipped `LinearPCG` for every iteration count — never `Eigen::ConjugateGradient`
 *    (research §6: its stopping rule is a different quantity).
 *
 * NEGATIVE CONTROLS (each proves a gate can fail)
 * ----------------------------------------------
 *  * case 4 vs 4b — a block-diagonal M and a row-sum M (which provably consumed the
 *    off-diagonal triplets: `Σ_{j≠i}|A_ij|`) give different `z`; a preconditioner that
 *    only reads `row == col` cannot produce the second one;
 *  * case 5 — `z[window 0]` is the **local** result and `z[window 1]` the **global**
 *    one, and the two differ: moving the global apply after the local loop, or
 *    dropping the local loop, both fail here and nowhere else;
 *  * case 9 vs 10 — the same matrix and right-hand side with the mathematically
 *    identical local block-Jacobi (equal count, required *exactly*) and with a wrong
 *    block table (a different count);
 *  * case 6 — `counters.syncs == 0` and `counters.uploads == 0` across one
 *    `apply_preconditioner`, with and without a registered global preconditioner, i.e.
 *    no host read crept into the iteration.
 *
 * DEVIATIONS (reported in docs/luisa_backend_progress.md)
 * ------------------------------------------------------
 * 1. A **new TU** instead of extending luisa_fem_preconditioner_test.cpp (plan 13 §5
 *    phase 2/3 says "extend it"): that file's own gate is "104 assertions in 4 test
 *    cases, unchanged", which adding cases would make untestable.  Everything here is
 *    globbed by both build systems, so there is no build-file diff either.
 * 2. `gpc_second_registration_behaviour` pins the **measured** behaviour (silent
 *    overwrite, the slot holds the last registration, no abort) instead of the plan's
 *    "assert on a second registration": neither backend has such an assert, and
 *    src/backends/common/** is read-only (research §7 R3).
 * 3. No `check_state(SimEngineState::SolveGlobalLinearSystem, …)` on the thunks (plan
 *    §5 phase 1 asks for it): the reference has none, and the standalone-`Impl`-style
 *    test below drives `impl().apply_preconditioner(...)` in no engine state, where a
 *    `check_state` would `UIPC_ASSERT`-abort the process (research §7 R4).
 * 4. `gpc_from_cuda_A` / `gpc_frame_pcg` are explicit SKIPs — see the reason strings
 *    at the end of the file (no CUDA backend, no `_dump_A_b`, BLOCKED-BY-S0.6, and the
 *    engine never calls `solve()`).
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
#include <linear_system/global_preconditioner.h>
#include <linear_system/iterative_solver.h>
#include <linear_system/linear_pcg.h>
#include <finite_element/fem_diag_preconditioner_function.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
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
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;
namespace lfdi = uipc::backend::luisa::fem_diag_preconditioner_detail;
// The ported surface lives in `uipc::backend::luisa` (like the reference's does in
// `uipc::backend::cuda`), so a test body sees the same spellings a ported system does.
using namespace uipc::backend::luisa;

namespace t83g
{
// ===========================================================================
//  1) the layout + the one source of truth for the device writes and the oracles
// ===========================================================================
constexpr int kBlockDim    = 3;
constexpr int kSubCount    = 2;  // two diag subsystems -> two dof windows
constexpr int kNodesPerSub = 4;
constexpr int kSubDofs     = kNodesPerSub * kBlockDim;  // 12
constexpr int kDofs        = kSubCount * kSubDofs;      // 24
constexpr int kBlockRows   = kDofs / kBlockDim;         // 8

/// The shared-binary marker (cookbook §7.14 rule 6): every system registered in this
/// file is inert unless the scene's geometry carries it, because a
/// `REGISTER_SIM_SYSTEM`'d class is instantiated in *every* scene of the shared binary
/// (research §7 R11).
constexpr const char* kMarker = "t83g_gpc_marker";
/// Reserved uids: chain0 < chain1, so the uid-ascending sort in `Impl::init()` puts
/// chain0 in window 0 (the dof layout depends on it, `global_linear_system.cpp:99-104`).
constexpr U64 kUidChain0 = 0x47385030ull;  // "G8P0"
constexpr U64 kUidChain1 = 0x47385031ull;  // "G8P1"

/// Which global preconditioner the scene registers (and how it computes M).
enum class GpcMode
{
    /// no `GlobalPreconditioner` at all -> the pre-S3 path (fallback + local fan-out)
    None,
    /// M = blockdiag(A), read off the whole assembled BCOO (`row == col` triplets)
    BlockJacobi,
    /// M_ii = A_ii + Σ_{j≠i}|A_ij| — provably consumes the off-diagonal triplets
    RowSum,
    /// the same table, rotated by one block: a wrong-but-SPD M (case 10's control)
    WrongBlock,
};

/// Which local preconditioner(s) the scene registers.
enum class LocalMode
{
    /// neither window has one -> the `z <- r` fallback for both
    None,
    /// both windows: the union of the local blocks *is* blockdiag(A) (case 9)
    DiagAll,
    /// window 0 only (case 5: window 1 is left to the global branch)
    Window0,
};

inline GpcMode   g_gpc_mode      = GpcMode::None;
inline LocalMode g_local_mode    = LocalMode::None;
inline bool      g_second_global = false;
/// the rotation `WrongBlock` applies to the block-diagonal table (case 10)
inline int g_wrong_shift = 1;

class T83gChain0;
class T83gChain1;
class T83gLocalPrecond;
class T83gGlobalPrecond;

/// The diagonal coefficient of block row `i` (heterogeneous and dyadic: a uniform chain
/// would make block-Jacobi a scalar multiple of the identity, and PCG is invariant
/// under `M -> c*M`, so it could not change the iteration count).
[[nodiscard]] inline double diag_coeff(int i)
{
    // the concatenation of two 4-node chains
    static constexpr double kDiag[kBlockRows] = {2.0, 3.0, 4.0, 6.0,
                                                 8.0, 12.0, 16.0, 24.0};
    return kDiag[i];
}

struct Block
{
    double v[kBlockDim][kBlockDim]{};  // v[r][c]

    [[nodiscard]] double operator()(int r, int c) const { return v[r][c]; }
    void                  set(int r, int c, double x) { v[r][c] = x; }
};

struct HostTriplet
{
    int   row = 0;  // *global* block row/col
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

/// The upper-triangular triplet table of one window: `kNodesPerSub` diagonal blocks and
/// `kNodesPerSub - 1` coupling blocks, all with *global* block indices.
[[nodiscard]] inline std::vector<HostTriplet> window_triplets(int window)
{
    const int first = window * kNodesPerSub;
    std::vector<HostTriplet> ts;
    ts.reserve(2 * kNodesPerSub - 1);
    for(int n = first; n < first + kNodesPerSub; ++n)
        ts.push_back({n, n, scaled_identity(diag_coeff(n))});
    for(int n = first; n + 1 < first + kNodesPerSub; ++n)
        ts.push_back({n, n + 1, scaled_identity(-1.0)});
    return ts;
}

inline void sort_triplets(std::vector<HostTriplet>& ts)
{
    std::sort(ts.begin(),
              ts.end(),
              [](const HostTriplet& a, const HostTriplet& b)
              { return a.row != b.row ? a.row < b.row : a.col < b.col; });
}

// ---------------------------------------------------------------------------
//  small helpers (the T82/T54b test files' idioms)
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// `std::printf` + `fflush`: the backend aborts the process on a failed UIPC_ASSERT, and
/// a redirected stdout is fully buffered -- a diagnostic without the flush is lost
/// exactly when it is needed.
#define T83G_LOG(...)                                                          \
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
        T83G_LOG("[t83g] %s: %d checks\n", name.c_str(), counter.checks);
    }
};

#define T83G_CHECK(counter, cond)                                              \
    do                                                                         \
    {                                                                          \
        ++(counter).checks;                                                    \
        CHECK(cond);                                                           \
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

/// The full-matrix dense form of the *assembled* triplets (the symmetric closure, i.e.
/// what `rbk_sym_spmv` applies) — used only for the residual gate of the PCG cases.
struct Assembled
{
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> cols;
    std::vector<double>       vals;
    SizeT                     block_rows = 0;
};

[[nodiscard]] inline Eigen::MatrixXd dense_A(const Assembled& A)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(kDofs, kDofs);
    for(std::size_t k = 0; k < A.rows.size(); ++k)
    {
        const int r0 = A.rows[k] * kBlockDim;
        const int c0 = A.cols[k] * kBlockDim;
        for(int r = 0; r < kBlockDim; ++r)
            for(int c = 0; c < kBlockDim; ++c)
            {
                const double v = A.vals[k * 9u + static_cast<std::size_t>(c * kBlockDim + r)];
                out(r0 + r, c0 + c) += v;
                if(A.rows[k] != A.cols[k])
                    out(c0 + c, r0 + r) += v;
            }
    }
    return out;
}

[[nodiscard]] inline double relative_residual(const Eigen::MatrixXd&     A,
                                              const std::vector<double>& b,
                                              const std::vector<double>& x)
{
    Eigen::Map<const Eigen::VectorXd> bx(b.data(), static_cast<Eigen::Index>(b.size()));
    Eigen::Map<const Eigen::VectorXd> xx(x.data(), static_cast<Eigen::Index>(x.size()));
    Eigen::VectorXd                   r = bx - A * xx;
    return r.norm() / bx.norm();
}

/// The default right-hand side: dyadic and *not* constant (a constant b is exactly
/// orthogonal to every row sum of a chain operator, which makes the first `alpha`
/// explode — the trap the T82 file documents).
[[nodiscard]] inline std::vector<double> default_rhs()
{
    std::vector<double> b(kDofs);
    for(int i = 0; i < kDofs; ++i)
    {
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        b[static_cast<std::size_t>(i)] =
            sign * (1.0 + 0.125 * static_cast<double>((i * 5 + 3) % 7));
    }
    return b;
}

// ===========================================================================
//  2) the host oracles — the *factored* routines over the assembled triplets
// ===========================================================================
using HostTripletView = lla::LTripletMatrixHostMemoryView<double, 3, 3>;
using HostDense       = lla::LDenseVectorHostMemoryView<double>;
using HostDiagInv     = lfdi::DiagInvHostMemoryView;
using DeviceTripletView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using DeviceDenseView   = lla::LDenseVectorDeviceView<lcrt::Var<double>>;

/// Read the whole assembled BCOO back to the host.  `frame_stream().synchronize()`
/// first: `build_linear_system()` records the assembly into the frame list, so without
/// it the read races the assembly (cookbook §7.40 rule 3).
[[nodiscard]] inline Assembled read_A(const lla::LBCOOMatrix<double, 3, 3>& bcoo)
{
    Assembled A;
    // the container spells the block extents `rows()`/`cols()` (the *view* spells them
    // `total_rows()`/`total_cols()`)
    A.block_rows = static_cast<SizeT>(bcoo.rows());
    const auto nnz = static_cast<std::size_t>(bcoo.triplet_count());
    A.rows.resize(nnz);
    A.cols.resize(nnz);
    A.vals.resize(nnz * 9u, 0.0);
    if(nnz != 0u)
    {
        gg::frame_stream().synchronize();
        bcoo.copy_to_row_indices(span<std::int32_t>{A.rows});
        bcoo.copy_to_col_indices(span<std::int32_t>{A.cols});
        bcoo.copy_to_values(span<double>{A.vals});
        lud::main_stream().synchronize();
    }
    return A;
}

/// `M^-1` for `M = blockdiag(A)`: `diag_inv_row` over the whole assembled matrix
/// (segment offset 0, segment count = the block row count), storing only the triplets
/// its own guards accept (`in_window && is_diagonal`) — the shipped
/// `FEMDiagPreconditioner_do_assemble_kernel` without a window.
[[nodiscard]] inline std::vector<LMatrix3x3> oracle_block_jacobi_inv(const Assembled& A)
{
    std::vector<LMatrix3x3> table(static_cast<std::size_t>(A.block_rows));
    if(A.rows.empty())
        return table;

    auto rows = A.rows;
    auto cols = A.cols;
    auto vals = A.vals;
    auto trip = HostTripletView::from_handles(rows.data(),
                                             cols.data(),
                                             vals.data(),
                                             static_cast<std::int32_t>(A.block_rows),
                                             static_cast<std::int32_t>(A.block_rows),
                                             static_cast<std::int32_t>(A.rows.size()));
    for(std::size_t k = 0; k < A.rows.size(); ++k)
    {
        auto row = lfdi::diag_inv_row(trip, static_cast<std::int32_t>(k), 0, static_cast<std::int32_t>(A.block_rows));
        if(row.in_window && row.is_diagonal)
            table[static_cast<std::size_t>(row.index)] = row.value;
    }
    return table;
}

[[nodiscard]] inline LMatrix3x3 block_at(const std::vector<double>& vals, std::size_t k)
{
    LMatrix3x3 m;
    for(int c = 0; c < kBlockDim; ++c)
        for(int r = 0; r < kBlockDim; ++r)
            m(r, c) = vals[k * 9u + static_cast<std::size_t>(c * kBlockDim + r)];
    return m;
}

/// `M_ii = A_ii + Σ_{j≠i}|A_ij|`, i.e. the (1,1)-block of the row-sum splitting.  Reads
/// the strict off-diagonal triplets of the assembled matrix in **both** directions (the
/// upper-packed BCOO gives row i the block `B` and row j its transpose), which is
/// precisely what a "block-diagonal only" implementation cannot see.  The accumulation
/// order matches the device kernel's statement for statement.
[[nodiscard]] inline std::vector<LMatrix3x3> oracle_row_sum_inv(const Assembled& A)
{
    std::vector<LMatrix3x3> table(static_cast<std::size_t>(A.block_rows));
    for(std::size_t i = 0; i < static_cast<std::size_t>(A.block_rows); ++i)
    {
        LMatrix3x3 acc;  // zero-initialised (LMat's storage is `data{}`)
        for(std::size_t k = 0; k < A.rows.size(); ++k)
        {
            const auto  rg = A.rows[k];
            const auto  cg = A.cols[k];
            const auto  B  = block_at(A.vals, k);
            const auto  i32 = static_cast<std::int32_t>(i);
            for(int c = 0; c < kBlockDim; ++c)
                for(int r = 0; r < kBlockDim; ++r)
                {
                    if(rg == i32 && cg == i32)
                        acc(r, c) = B(r, c);
                    if(rg == i32 && cg != i32)
                        acc(r, c) = acc(r, c) + std::abs(B(r, c));
                    if(cg == i32 && rg != i32)
                        acc(r, c) = acc(r, c) + std::abs(B(c, r));  // the |B^T| element
                }
        }
        table[i] = lps::inverse(acc);
    }
    return table;
}

/// `z = M^-1 r` through the same factored row routine the device kernel calls.
[[nodiscard]] inline std::vector<double> oracle_apply(const std::vector<LMatrix3x3>& table,
                                                      const std::vector<double>&    r)
{
    std::vector<double> z(r.size(), 0.0);
    // the `storage::memory` views hand out mutable handles, so both inputs need a
    // mutable copy (the oracle never writes through them)
    auto                r_mut = r;
    auto                t_mut = table;
    auto                R     = HostDense::from_handles(r_mut.data(),
                                             static_cast<std::int32_t>(r_mut.size()));
    auto                Z     = HostDense::from_handles(z.data(),
                                             static_cast<std::int32_t>(z.size()));
    auto                D     = HostDiagInv::from_handles(t_mut.data(),
                                               static_cast<std::int32_t>(t_mut.size()));
    for(std::size_t k = 0; k < t_mut.size(); ++k)
        lfdi::diag_apply_row(Z, R, D, static_cast<std::int32_t>(k));
    return z;
}

/// The per-window slice of a whole-vector oracle result.
[[nodiscard]] inline std::vector<double> window_of(const std::vector<double>& v, int window)
{
    const auto begin = static_cast<std::size_t>(window * kSubDofs);
    return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(begin),
                               v.begin() + static_cast<std::ptrdiff_t>(begin + kSubDofs));
}

// ===========================================================================
//  3) the device kernels
// ===========================================================================
namespace detail
{
/// `FEMDiagPreconditioner_do_assemble_kernel`, i.e. `diag_inv_row` + the gated store,
/// over one dof window (`seg_offset`/`seg_count`) or over the whole matrix (0 / N).
auto& diag_inv_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // A rows
                                   lcrt::Buffer<std::int32_t>,  // A cols
                                   lcrt::Buffer<double>,        // A values
                                   lcrt::Buffer<LMatrix3x3>,    // diag_inv out
                                   std::int32_t,                // total block rows
                                   std::int32_t,                // total block cols
                                   std::int32_t,                // segment offset
                                   std::int32_t,                // segment count
                                   uint32_t,                    // triplet count
                                   std::int32_t>(               // store-index shift
            [](lcrt::BufferVar<std::int32_t> rows,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       vals,
               lcrt::BufferVar<LMatrix3x3>   diag_inv,
               lcrt::Var<std::int32_t>       total_block_rows,
               lcrt::Var<std::int32_t>       total_block_cols,
               lcrt::Var<std::int32_t>       seg_offset,
               lcrt::Var<std::int32_t>       seg_count,
               lcrt::Var<uint32_t>           n,
               lcrt::Var<std::int32_t>       shift) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    auto A = DeviceTripletView::from_handles(
                        rows, cols, vals, total_block_rows, total_block_cols, n);
                    auto row = lfdi::diag_inv_row(A, lcrt::cast<std::int32_t>(I), seg_offset, seg_count);
                    // the reference's two device early-outs, as the store's guards
                    $if(row.in_window)
                    {
                        $if(row.is_diagonal)
                        {
                            lcrt::Var<LMatrix3x3> out;
                            store_mat3x3(out, row.value);
                            // `shift != 0` is the *wrong-M* control: every thread writes a
                            // distinct slot, so the rotation is race-free and the result is
                            // still a symmetric-positive-definite block-diagonal table
                            diag_inv->write(lcrt::cast<std::uint32_t>((row.index + shift)
                                                                      % total_block_rows),
                                            out);
                        };
                    };
                };
            });
    return k;
}

/// `M_ii = A_ii + Σ_{j≠i}|A_ij|` computed from the *whole* assembled BCOO: one thread per
/// block row, a `$while` scan over every triplet (the port's idiom for a runtime loop —
/// `$while` with a local counter, as in aabb_reduce.cpp:185 / mas_preconditioner_engine.cpp:155).
auto& row_sum_inv_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // A rows
                                   lcrt::Buffer<std::int32_t>,  // A cols
                                   lcrt::Buffer<double>,        // A values
                                   lcrt::Buffer<LMatrix3x3>,    // diag_inv out
                                   std::int32_t,                // total block rows
                                   uint32_t>(                   // triplet count
            [](lcrt::BufferVar<std::int32_t> rows,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       vals,
               lcrt::BufferVar<LMatrix3x3>   diag_inv,
               lcrt::Var<std::int32_t>       total_block_rows,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(lcrt::cast<std::int32_t>(i) < total_block_rows)
                {
                    auto i32 = lcrt::cast<std::int32_t>(i);
                    // LMat's storage is `data{}`, so the accumulator starts at 0 on the
                    // device exactly as it does in the host oracle
                    lps::LMat<lcrt::Var<Float>, 3, 3> acc;
                    lcrt::Var<std::uint32_t>          I = 0u;
                    $while(I < n)
                    {
                        auto rg = rows->read(I);
                        auto cg = cols->read(I);
                        for(int c = 0; c < kBlockDim; ++c)
                        {
                            for(int r = 0; r < kBlockDim; ++r)
                            {
                                const auto base = I * 9u
                                                  + static_cast<std::uint32_t>(c * kBlockDim + r);
                                const auto tbase = I * 9u
                                                   + static_cast<std::uint32_t>(r * kBlockDim + c);
                                $if((rg == i32) & (cg == i32))
                                {
                                    acc(r, c) = vals->read(base);
                                };
                                $if((rg == i32) & (cg != i32))
                                {
                                    acc(r, c) = acc(r, c) + lcrt::abs(vals->read(base));
                                };
                                $if((cg == i32) & (rg != i32))
                                {
                                    acc(r, c) = acc(r, c) + lcrt::abs(vals->read(tbase));
                                };
                            }
                        }
                        I = I + 1u;
                    };
                    lcrt::Var<LMatrix3x3> out;
                    store_mat3x3(out, lps::inverse(acc));
                    diag_inv->write(i, out);
                };
            });
    return k;
}

/// `FEMDiagPreconditioner_do_apply_kernel`: `z.segment<3>(3i) = diag_inv(i) * r.segment<3>(3i)`
/// over the whole dispatch, gated on the reference's `if(*converged != 0) return;`.  One
/// kernel serves the local windows and the global whole-vector apply (same arithmetic,
/// different `n`).
auto& diag_apply_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,        // r
                                   lcrt::Buffer<double>,        // z
                                   lcrt::Buffer<std::int32_t>,  // converged
                                   lcrt::Buffer<LMatrix3x3>,    // diag_inv
                                   uint32_t>(                   // block count
            [](lcrt::BufferVar<double>       r,
               lcrt::BufferVar<double>       z,
               lcrt::BufferVar<std::int32_t> converged,
               lcrt::BufferVar<LMatrix3x3>   diag_inv,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    $if(converged->read(0) == 0)
                    {
                        auto R = DeviceDenseView::from_handles(r, n * 3u);
                        auto Z = DeviceDenseView::from_handles(z, n * 3u);
                        auto D = lfdi::DiagInvDeviceView::from_handles(diag_inv, n);
                        lfdi::diag_apply_row(Z, R, D, lcrt::cast<std::int32_t>(i));
                    };
                };
            });
    return k;
}
}  // namespace detail

// ===========================================================================
//  4) the scene + the marker gate
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

[[nodiscard]] inline Scene build_scene(const Json& config, bool with_marker)
{
    Scene scene{config};
    auto  object = scene.objects().create("t83g");

    const std::vector<Vector3> positions = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    const std::vector<Vector4i> tets = {{0, 1, 2, 3}};

    auto mesh = tetmesh(span<const Vector3>{positions}, span<const Vector4i>{tets});
    label_surface(mesh);
    // no dof-producing constitution claims this mesh: the whole linear system of this
    // file's scenes is the fixture below
    if(with_marker)
        mesh.vertices().create<IndexT>(kMarker, 1);
    object->geometries().create(mesh);
    return scene;
}

// ===========================================================================
//  5) the fixtures
// ===========================================================================
/// One dof window of the fixture's block chain.  `window()` and `get_uid()` are the only
/// things the two concrete classes add: `require<T>()` needs a concrete type, so two
/// windows need two classes (and the uid order fixes which one is window 0).
class T83gChainBase : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    static inline bool enabled[kSubCount] = {false, false};

    static void reset() { enabled[0] = enabled[1] = false; }

  protected:
    [[nodiscard]] virtual int window() const noexcept = 0;

    void do_init(InitInfo& info) override
    {
        (void)info;
        enabled[window()] = has_marker(world());
    }

    void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) override
    {
        info.extent(enabled[window()] ? SizeT{kSubDofs} : SizeT{0});
    }

    void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) override
    {
        (void)info;
    }

    void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) override
    {
        const auto n = enabled[window()] ? window_triplets(window()).size() : SizeT{0};
        info.extent(n, enabled[window()] ? SizeT{kSubDofs} : SizeT{0});
    }

    void do_assemble(GlobalLinearSystem::DiagInfo& info) override
    {
        if(!enabled[window()])
            return;

        auto       hessians = info.hessians();
        const auto count    = static_cast<std::size_t>(hessians.triplet_count());
        auto       ts       = window_triplets(window());
        if(count != ts.size())
            return;  // the caller's extent assertions then fail loudly

        // `window_triplets()` already carries *global* block indices, and
        // `row_indices()` is the whole matrix's raw index buffer (the window is selected
        // by `triplet_index_offset()`, not by rebasing the indices) — unlike
        // `TripletMatrixView::write()`, which would add `submatrix_offset()` itself.  So
        // the table is written verbatim, exactly like the T82 fixture does.
        const auto begin = static_cast<std::size_t>(hessians.triplet_index_offset());

        std::vector<std::int32_t> rows(count);
        std::vector<std::int32_t> cols(count);
        std::vector<double>       vals(count * 9u, 0.0);
        for(std::size_t i = 0; i < count; ++i)
        {
            rows[i] = static_cast<std::int32_t>(ts[i].row);
            cols[i] = static_cast<std::int32_t>(ts[i].col);
            for(int c = 0; c < kBlockDim; ++c)
                for(int r = 0; r < kBlockDim; ++r)
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

    void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) override
    {
        // the reference's `AccuracyInfo::satisfied(bool)` defaults to true; the fixture
        // has no Newton-level tolerance to model, so the device's stopping rule is
        // `|rz_k| <= tol*|rz_0|` alone
        (void)info;
    }

    void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) override
    {
        (void)info;
    }

    Float do_diag_norm(GlobalLinearSystem::DiagNormInfo& info) override
    {
        (void)info;
        return 0.0;
    }

    Float do_mass_norm(GlobalLinearSystem::DiagNormInfo& info) override
    {
        (void)info;
        return 0.0;
    }
};

class T83gChain0 final : public T83gChainBase
{
  public:
    using T83gChainBase::T83gChainBase;

  protected:
    [[nodiscard]] int window() const noexcept override { return 0; }
    [[nodiscard]] U64 get_uid() const noexcept override { return kUidChain0; }
};

class T83gChain1 final : public T83gChainBase
{
  public:
    using T83gChainBase::T83gChainBase;

  protected:
    [[nodiscard]] int window() const noexcept override { return 1; }
    [[nodiscard]] U64 get_uid() const noexcept override { return kUidChain1; }
};

/// A `LocalPreconditioner` that is the shipped `FEMDiagPreconditioner`'s shape with the
/// window taken from `info.dof_offset()/3, info.dof_count()/3`: `M` is assembled from
/// `info.A()` by `diag_inv_row` on the device, so its inverse table is *bitwise* the
/// table a whole-matrix block-Jacobi global preconditioner computes (case 9 depends on
/// exactly that).
class T83gLocalPrecond : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    lud::LDeviceBuffer<LMatrix3x3> diag_inv;

  protected:
    [[nodiscard]] virtual int window() const noexcept = 0;
    [[nodiscard]] virtual DiagLinearSubsystem* require_chain() = 0;

    void do_build(BuildInfo& info) override
    {
        const bool wanted = has_marker(world())
                            && (g_local_mode == LocalMode::DiagAll
                                || (g_local_mode == LocalMode::Window0 && window() == 0));
        if(!wanted)
            throw uipc::backend::SimSystemException(
                "T83gLocalPrecond unused: no marker, or its window is not selected by "
                "LocalMode");

        info.connect(require_chain());
    }

    void do_init(InitInfo& info) override { (void)info; }

    void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        auto block_count = static_cast<uint32_t>(info.dof_count()) / 3u;
        diag_inv.resize(block_count);
        if(block_count == 0u)
            return;

        auto  A = info.A();
        auto& k = detail::diag_inv_kernel();
        gg::frame_stream().record(k(A.row_indices(),
                                    A.col_indices(),
                                    A.values(),
                                    diag_inv.view(),
                                    static_cast<std::int32_t>(A.total_rows()),
                                    static_cast<std::int32_t>(A.total_cols()),
                                    static_cast<std::int32_t>(info.dof_offset() / 3),
                                    static_cast<std::int32_t>(info.dof_count() / 3),
                                    static_cast<uint32_t>(A.triplet_count()),
                                    std::int32_t{0})
                                      .dispatch(static_cast<uint32_t>(A.triplet_count())));
    }

    void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        auto n = static_cast<uint32_t>(diag_inv.size());
        if(n == 0u)
            return;
        auto& k = detail::diag_apply_kernel();
        gg::frame_stream().record(k(info.r().buffer_view(),
                                    info.z().buffer_view(),
                                    info.converged(),
                                    diag_inv.cview(),
                                    n)
                                      .dispatch(n));
    }
};

class T83gLocalPrecond0 final : public T83gLocalPrecond
{
  public:
    using T83gLocalPrecond::T83gLocalPrecond;

  protected:
    [[nodiscard]] int window() const noexcept override { return 0; }
    [[nodiscard]] DiagLinearSubsystem* require_chain() override
    {
        return &require<T83gChain0>();
    }
};

class T83gLocalPrecond1 final : public T83gLocalPrecond
{
  public:
    using T83gLocalPrecond::T83gLocalPrecond;

  protected:
    [[nodiscard]] int window() const noexcept override { return 1; }
    [[nodiscard]] DiagLinearSubsystem* require_chain() override
    {
        return &require<T83gChain1>();
    }
};

/// The synthetic global preconditioner — the object of this test.  It is *not* bound to a
/// dof window: `do_assemble` sees the whole assembled BCOO and `do_apply` gets the whole
/// `z`/`r`.
class T83gGlobalPrecond : public GlobalPreconditioner
{
  public:
    using GlobalPreconditioner::GlobalPreconditioner;

    lud::LDeviceBuffer<LMatrix3x3> diag_inv;

    /// what `do_assemble` observed (case 3) — the whole-matrix evidence
    inline static SizeT observed_triplets   = 0;
    inline static SizeT observed_block_rows = 0;
    inline static SizeT observed_block_cols = 0;
    inline static SizeT observed_offdiag    = 0;
    inline static int   observed_max_row    = -1;
    inline static int   observed_max_col    = -1;
    /// the registration order (case 8: the slot keeps the *last* one)
    inline static std::vector<GlobalPreconditioner*> registration_log;

    static void reset()
    {
        observed_triplets   = 0;
        observed_block_rows = 0;
        observed_block_cols = 0;
        observed_offdiag    = 0;
        observed_max_row    = -1;
        observed_max_col    = -1;
        registration_log.clear();
    }

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        if(!has_marker(world()) || g_gpc_mode == GpcMode::None)
            throw uipc::backend::SimSystemException(
                "T83gGlobalPrecond unused: no marker, or GpcMode::None");
        registration_log.push_back(this);
    }

    void do_assemble(GlobalLinearSystem::GlobalPreconditionerAssemblyInfo& info) override
    {
        auto A = info.A();

        auto block_rows = static_cast<uint32_t>(A.total_rows());
        diag_inv.resize(block_rows);
        if(block_rows == 0u)
            return;

        // ---- the observation (case 3): the *whole* matrix, off-diagonal triplets
        // included.  A host read here is fine: this runs once per solve, never inside
        // the PCG iteration (`apply` is what must stay sync-free -- case 6).
        {
            const auto n = static_cast<std::size_t>(A.triplet_count());
            observed_triplets   = static_cast<SizeT>(n);
            observed_block_rows = static_cast<SizeT>(A.total_rows());
            observed_block_cols = static_cast<SizeT>(A.total_cols());
            T83G_LOG("[t83g] global assemble: %zu triplets over %zu x %zu block rows\n",
                     n,
                     observed_block_rows,
                     observed_block_cols);

            std::vector<std::int32_t> rows(n), cols(n);
            if(n != 0u)
            {
                gg::frame_stream().synchronize();
                A.row_indices().copy_to(span<std::int32_t>{rows});
                A.col_indices().copy_to(span<std::int32_t>{cols});
                lud::main_stream().synchronize();
            }
            observed_offdiag = 0;
            observed_max_row = -1;
            observed_max_col = -1;
            for(std::size_t k = 0; k < n; ++k)
            {
                observed_offdiag += (rows[k] != cols[k]) ? 1u : 0u;
                observed_max_row = std::max(observed_max_row, static_cast<int>(rows[k]));
                observed_max_col = std::max(observed_max_col, static_cast<int>(cols[k]));
            }
        }

        const auto n = static_cast<uint32_t>(A.triplet_count());
        if(g_gpc_mode == GpcMode::RowSum)
        {
            // M_ii = A_ii + Σ_{j≠i}|A_ij|: one thread per block row, a $while scan over
            // every triplet -- the only way this M can be built is by reading the
            // off-diagonal blocks
            gg::frame_stream().record(detail::row_sum_inv_kernel()(A.row_indices(),
                                                                  A.col_indices(),
                                                                  A.values(),
                                                                  diag_inv.view(),
                                                                  static_cast<std::int32_t>(block_rows),
                                                                  n)
                                          .dispatch(block_rows));
        }
        else
        {
            // M = blockdiag(A): the shipped `diag_inv_row` over the whole matrix
            // (segment offset 0, segment count = every block row).  `shift != 0` is the
            // WrongBlock control: a race-free rotation of the same table.
            const auto shift = std::int32_t{g_gpc_mode == GpcMode::WrongBlock ? g_wrong_shift
                                                                             : 0};
            gg::frame_stream().record(detail::diag_inv_kernel()(A.row_indices(),
                                                               A.col_indices(),
                                                               A.values(),
                                                               diag_inv.view(),
                                                               static_cast<std::int32_t>(A.total_rows()),
                                                               static_cast<std::int32_t>(A.total_cols()),
                                                               std::int32_t{0},
                                                               static_cast<std::int32_t>(A.total_rows()),
                                                               n,
                                                               shift)
                                      .dispatch(n));
        }
    }

    void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        auto n = static_cast<uint32_t>(diag_inv.size());
        if(n == 0u)
            return;
        auto& k = detail::diag_apply_kernel();
        gg::frame_stream().record(k(info.r().buffer_view(),
                                    info.z().buffer_view(),
                                    info.converged(),
                                    diag_inv.cview(),
                                    n)
                                      .dispatch(n));
    }
};

/// The second global preconditioner of case 8 (see the deviation note in the file
/// header: the reference and this port both silently overwrite).
class T83gGlobalPrecondB final : public T83gGlobalPrecond
{
  public:
    using T83gGlobalPrecond::T83gGlobalPrecond;

  protected:
    void do_build(BuildInfo& info) override
    {
        if(!has_marker(world()) || g_gpc_mode == GpcMode::None || !g_second_global)
            throw uipc::backend::SimSystemException(
                "T83gGlobalPrecondB unused: no marker, GpcMode::None, or no second "
                "registration requested");
        T83gGlobalPrecond::do_build(info);
    }
};

/// Captures the engine's systems once they are built.
class T83gProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::GlobalLinearSystem* gls    = nullptr;
    inline static gg::LinearPCG*          pcg    = nullptr;
    inline static T83gChain0*             chain0 = nullptr;
    inline static T83gChain1*             chain1 = nullptr;

    static void reset()
    {
        gls    = nullptr;
        pcg    = nullptr;
        chain0 = nullptr;
        chain1 = nullptr;
    }

  private:
    void do_build() override
    {
        gls    = &require<gg::GlobalLinearSystem>();
        pcg    = &require<gg::LinearPCG>();
        chain0 = &require<T83gChain0>();
        chain1 = &require<T83gChain1>();
    }
};
}  // namespace t83g

REGISTER_SIM_SYSTEM(t83g::T83gChain0);
REGISTER_SIM_SYSTEM(t83g::T83gChain1);
REGISTER_SIM_SYSTEM(t83g::T83gLocalPrecond0);
REGISTER_SIM_SYSTEM(t83g::T83gLocalPrecond1);
REGISTER_SIM_SYSTEM(t83g::T83gGlobalPrecond);
REGISTER_SIM_SYSTEM(t83g::T83gGlobalPrecondB);
REGISTER_SIM_SYSTEM(t83g::T83gProbe);

namespace t83g
{
// ===========================================================================
//  6) the driver: one world, the shipped chain, one apply / one solve
// ===========================================================================
struct RunOptions
{
    GpcMode   gpc            = GpcMode::None;
    LocalMode local          = LocalMode::None;
    bool      second_global  = false;
    bool      with_marker    = true;
    double    tol_rate       = 1e-3;
};

/// Build a fresh world for `options` and hand the *engine's* `GlobalLinearSystem` (layout
/// done, systems built) to `fn`, with everything alive.  `fn` gets the same `Impl` the
/// solver's forwards will call back into, which is what keeps the *shipped* chain the
/// thing under test (`GlobalLinearSystem::impl()`, the T8.3a-2 deviation).
template <typename Fn>
void with_world(const RunOptions& options, Fn&& fn)
{
    g_gpc_mode      = options.gpc;
    g_local_mode    = options.local;
    g_second_global = options.second_global;

    T83gProbe::reset();
    T83gChainBase::reset();
    T83gGlobalPrecond::reset();

    Json config = base_config();
    config["linear_system"]["tol_rate"] = options.tol_rate;

    auto scene = build_scene(config, options.with_marker);

    Engine engine{"luisa", workspace_path("uipc_luisa_t83g_gpc")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T83gProbe::gls != nullptr);
    REQUIRE(T83gProbe::chain0 != nullptr);
    REQUIRE(T83gProbe::chain1 != nullptr);

    auto* gls = T83gProbe::gls;
    // T8.3b: `SimEngine::init_scene()` calls `GlobalLinearSystem::init()`, so the
    // layout is already done; calling it again would append duplicate entries to
    // `no_precond_diag_subsystem_indices` (the `z <- r` fallback set).

    fn(*gls, gls->impl());
}

/// What one `apply_preconditioner` produced.
struct ApplyRun
{
    Assembled           A;
    std::vector<double> r;
    std::vector<double> z;
    /// how many diag windows had no local preconditioner (`Impl::
    /// no_precond_diag_subsystem_indices`), i.e. how many took the `z <- r` fallback
    SizeT fallback_windows = 0;
};

/// `build_linear_system()` -> upload `r` -> one `apply_preconditioner()`.  `sentinel` is
/// pre-written into `z` so an untouched entry (a missing branch, or the `converged`
/// early-out) is distinguishable from a computed zero.  `converged` is an `IndexT` device
/// value, exactly as the solver passes it.
[[nodiscard]] inline ApplyRun run_apply(const RunOptions&     options,
                                        IndexT                converged = 0,
                                        const std::vector<double>* r_override = nullptr)
{
    ApplyRun run;
    with_world(options,
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   (void)gls;
                   impl.build_linear_system();
                   REQUIRE(impl.empty_system == false);
                   run.A = read_A(impl.bcoo_A);
                   REQUIRE(run.A.block_rows == static_cast<SizeT>(kBlockRows));
                   run.fallback_windows = impl.no_precond_diag_subsystem_indices.size();

                   run.r = r_override ? *r_override : default_rhs();
                   lla::LDenseVector<double> r_dev, z_dev;
                   r_dev.resize(run.r.size());
                   z_dev.resize(run.r.size());
                   r_dev.copy_from(span<const double>{run.r});

                   const std::vector<double> sentinel(run.r.size(), -12345.0);
                   z_dev.copy_from(span<const double>{sentinel});

                   lud::LDeviceVar<IndexT> d_converged;
                   d_converged.copy_from(converged);

                   gg::frame_stream().synchronize();
                   impl.apply_preconditioner(z_dev.view(),
                                             r_dev.cview(),
                                             d_converged.view(),
                                             nullptr);
                   gg::frame_stream().synchronize();

                   run.z.resize(run.r.size());
                   z_dev.copy_to(span<double>{run.z});
               });
    return run;
}

/// What one `solve_linear_system()` produced.
struct PcgRun
{
    Assembled           A;
    SizeT               iters = 0;
    std::vector<double> b;
    std::vector<double> x;
};

[[nodiscard]] inline PcgRun run_pcg(const RunOptions& options)
{
    PcgRun run;
    with_world(options,
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   impl.build_linear_system();
                   REQUIRE(impl.empty_system == false);
                   run.A = read_A(impl.bcoo_A);

                   run.b = default_rhs();
                   REQUIRE(run.b.size() == impl.b.size());
                   impl.b.copy_from(span<const double>{run.b});

                   impl.solve_linear_system();
                   run.iters = gls.last_solve_iterations();

                   run.x.resize(run.b.size());
                   impl.x.copy_to(span<double>{run.x});
               });
    return run;
}
}  // namespace t83g

namespace t83g
{
// ===========================================================================
//  CASE 1 — the slot: registered by `add_preconditioner(GlobalPreconditioner*)`
// ===========================================================================
TEST_CASE("gpc_slot_registered", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_slot_registered"};

    // ---- (a) the fixture scene: the slot really holds *this* object ----------
    with_world({.gpc = GpcMode::BlockJacobi, .local = LocalMode::None},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   auto& c = scope.counter;

                   // `add_preconditioner(GlobalPreconditioner*)` is no longer a no-op
                   T83G_CHECK(c, T83gGlobalPrecond::registration_log.size() == 1u);
                   REQUIRE(static_cast<bool>(impl.global_preconditioner));
                   auto* slot = impl.global_preconditioner.view();
                   T83G_CHECK(c, slot == static_cast<gg::GlobalPreconditioner*>(
                                         T83gGlobalPrecond::registration_log.front()));
                   T83G_CHECK(c, slot != nullptr);
                   T83G_CHECK(c, slot->is_valid());
                   T83G_CHECK(c, dynamic_cast<T83gGlobalPrecond*>(slot) != nullptr);
                   // one optional slot, so the *local* collection is a different thing
                   T83G_CHECK(c, impl.local_preconditioners.view().empty());
                   // both chain fixtures are in the diag collection.  Deliberately not
                   // `size() == 2`: the scene also *builds* every other registered
                   // diag subsystem (FEM/ABD/... report 0 dofs here, since no
                   // constitution claims the fixture mesh), and they are what makes the
                   // full layout 24 dofs
                   T83G_CHECK(c,
                             static_cast<bool>(impl.diag_subsystems.find<T83gChain0>()));
                   T83G_CHECK(c,
                             static_cast<bool>(impl.diag_subsystems.find<T83gChain1>()));
                   T83G_CHECK(c, gls.dof_count() == static_cast<SizeT>(kDofs));

                   impl.build_linear_system();
                   T83G_CHECK(c, impl.bcoo_A.triplet_count() == 14u);
                   T83G_CHECK(c, impl.no_precond_diag_subsystem_indices.size() >= 2u);
               });

    // ---- (b) a scene without the marker: the slot stays empty ----------------
    with_world({.gpc = GpcMode::BlockJacobi, .local = LocalMode::DiagAll, .with_marker = false},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   auto& c = scope.counter;
                   T83G_CHECK(c, T83gGlobalPrecond::registration_log.empty());
                   T83G_CHECK(c, !impl.global_preconditioner);
                   T83G_CHECK(c, impl.local_preconditioners.view().empty());
                   // the fixture's own two subsystems are built (with 0 dofs), and the
                   // scene's other diag subsystems are unaffected
                   T83G_CHECK(c,
                             static_cast<bool>(impl.diag_subsystems.find<T83gChain0>()));
                   T83G_CHECK(c,
                             static_cast<bool>(impl.diag_subsystems.find<T83gChain1>()));
                   T83G_CHECK(c, gls.dof_count() == 0u);
                   T83G_LOG("[t83g] slot (no marker): global slot empty, %zu local\n",
                            impl.local_preconditioners.view().size());
               });
}

// ===========================================================================
//  CASE 2 — no global registered: the pre-S3 path is byte-identical
// ===========================================================================
TEST_CASE("gpc_no_global_path_unchanged", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_no_global_path_unchanged"};

    // ---- (a) no local either: the `z <- r` copy fallback for *both* windows --
    const auto fallback = run_apply({.gpc = GpcMode::None, .local = LocalMode::None});
    {
        auto& c = scope.counter;
        T83G_CHECK(c, fallback.A.rows.size() == 14u);
        // the fallback is a device->device copy, so it must be bitwise
        T83G_CHECK(c, max_abs_diff(fallback.z, fallback.r) == 0.0);
        int touched = 0;
        for(std::size_t i = 0; i < fallback.z.size(); ++i)
            touched += (fallback.z[i] != -12345.0) ? 1 : 0;
        T83G_CHECK(c, touched == kDofs);  // no sentinel survived
        T83G_LOG("[t83g] no-global/fallback: %d/%d entries copied from r, %zu window(s) "
                 "took the fallback\n",
                 touched,
                 kDofs,
                 fallback.fallback_windows);
    }

    // ---- (b) two local block-Jacobi windows: z is D^-1 r, Δ == 0 -------------
    const auto local = run_apply({.gpc = GpcMode::None, .local = LocalMode::DiagAll});
    {
        auto&      c  = scope.counter;
        const auto D  = oracle_block_jacobi_inv(local.A);
        const auto zo = oracle_apply(D, local.r);
        const double d = max_abs_diff(local.z, zo);
        double       zmax = 0.0, ident = 0.0;
        for(std::size_t i = 0; i < zo.size(); ++i)
        {
            zmax  = std::max(zmax, std::abs(zo[i]));
            ident = std::max(ident, std::abs(local.r[i] - zo[i]));
        }
        T83G_LOG("[t83g] no-global/local: |z - D^-1 r| %.3e (max |z| %.3e, "
                 "|r - D^-1 r| %.3e, %zu fallback window(s))\n",
                 d,
                 zmax,
                 ident,
                 local.fallback_windows);
        // Class A: the fixture and the oracle run the same factored routines on the same
        // numbers, so the only honest gate is exact equality
        T83G_CHECK(c, d == 0.0);
        T83G_CHECK(c, ident > 1e-2);  // non-vacuous: the preconditioner is not the identity
        // the two fixture windows are exactly the ones the local fan-out now covers, so
        // the fallback set shrank by exactly 2 without touching anything else
        T83G_CHECK(c, local.fallback_windows + 2u == fallback.fallback_windows);
    }
}

// ===========================================================================
//  CASE 3 — `do_assemble` sees the whole assembled matrix, not a window
// ===========================================================================
TEST_CASE("gpc_assemble_sees_whole_A", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_assemble_sees_whole_A"};

    with_world({.gpc = GpcMode::BlockJacobi, .local = LocalMode::None},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   (void)gls;
                   impl.build_linear_system();

                   auto& c = scope.counter;
                   // called at all
                   T83G_CHECK(c, T83gGlobalPrecond::observed_triplets > 0u);
                   // called with the *whole* matrix: 8 block rows / 8 block cols, and the
                   // triplet count is the BCOO's (a windowed call would report 4 and 7)
                   T83G_CHECK(c, T83gGlobalPrecond::observed_triplets
                                     == static_cast<SizeT>(impl.bcoo_A.triplet_count()));
                   T83G_CHECK(c, T83gGlobalPrecond::observed_block_rows == 8u);
                   T83G_CHECK(c, T83gGlobalPrecond::observed_block_cols == 8u);
                   // it really saw off-diagonal blocks (6 of the 14 triplets: 3 per
                   // window's chain)
                   T83G_CHECK(c, T83gGlobalPrecond::observed_offdiag == 6u);
                   // and it spans both windows: block row/col 7 is window 1's last node
                   T83G_CHECK(c, T83gGlobalPrecond::observed_max_row == 7);
                   T83G_CHECK(c, T83gGlobalPrecond::observed_max_col == 7);
                   T83G_LOG("[t83g] assemble: %zu triplets, %zu off-diagonal, block rows "
                            "%zu x %zu, max (row,col) (%d,%d)\n",
                            T83gGlobalPrecond::observed_triplets,
                            T83gGlobalPrecond::observed_offdiag,
                            T83gGlobalPrecond::observed_block_rows,
                            T83gGlobalPrecond::observed_block_cols,
                            T83gGlobalPrecond::observed_max_row,
                            T83gGlobalPrecond::observed_max_col);
               });
}

// ===========================================================================
//  CASE 4 — the global apply, Class A: z = blockdiag(A)^-1 r
// ===========================================================================
TEST_CASE("gpc_apply_classA_blockjacobi", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_apply_classA_blockjacobi"};

    const auto run = run_apply({.gpc = GpcMode::BlockJacobi, .local = LocalMode::None});

    auto&      c  = scope.counter;
    const auto D  = oracle_block_jacobi_inv(run.A);
    const auto zo = oracle_apply(D, run.r);
    const double d = max_abs_diff(run.z, zo);

    double zmax = 0.0, nonvacuous = 0.0;
    for(std::size_t i = 0; i < zo.size(); ++i)
    {
        zmax       = std::max(zmax, std::abs(zo[i]));
        nonvacuous = std::max(nonvacuous, std::abs(run.z[i] - run.r[i]));
    }
    T83G_LOG("[t83g] apply/blockjacobi: |z - D^-1 r| %.3e (max |z| %.3e, "
             "|z - r|max %.3e)\n",
             d,
             zmax,
             nonvacuous);

    // the global branch really ran: no sentinel survived anywhere
    int kept = 0;
    for(double z : run.z)
        kept += (z == -12345.0) ? 1 : 0;
    T83G_CHECK(c, kept == 0);
    // Class A: one triplet per diagonal block, one 3x3-by-3 product per block, no
    // accumulation -> exact
    T83G_CHECK(c, d == 0.0);
    T83G_CHECK(c, zmax > 1e-2);         // non-degenerate
    T83G_CHECK(c, nonvacuous > 1e-2);   // and not the identity
}

// ===========================================================================
//  CASE 4b — the global apply really consumes the off-diagonal blocks
// ===========================================================================
TEST_CASE("gpc_apply_sees_offdiag", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_apply_sees_offdiag"};

    // the same A and r, two different M's: blockdiag(A) and A_ii + Σ_{j≠i}|A_ij|
    const auto block = run_apply({.gpc = GpcMode::BlockJacobi, .local = LocalMode::None});
    const auto rowsum = run_apply({.gpc = GpcMode::RowSum, .local = LocalMode::None});

    auto& c = scope.counter;
    T83G_CHECK(c, block.A.rows == rowsum.A.rows);   // the same assembled matrix
    T83G_CHECK(c, block.A.cols == rowsum.A.cols);
    T83G_CHECK(c, block.A.vals == rowsum.A.vals);

    const auto R   = oracle_row_sum_inv(rowsum.A);
    const auto zo  = oracle_apply(R, rowsum.r);
    const double d = max_abs_diff(rowsum.z, zo);
    const double differ = max_abs_diff(rowsum.z, block.z);
    T83G_LOG("[t83g] apply/rowsum: |z_rowsum - oracle| %.3e, "
             "max|z_rowsum - z_blockjacobi| %.3e\n",
             d,
             differ);

    T83G_CHECK(c, d == 0.0);        // the device computed the same row sums
    T83G_CHECK(c, differ > 1e-2);   // ... and they are *not* the diagonal-only answer
}

// ===========================================================================
//  CASE 5 — the precedence: global first (whole vector), local second (its windows)
// ===========================================================================
TEST_CASE("gpc_apply_precedence", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_apply_precedence"};

    // local block-Jacobi on window 0 only + a *different* M globally (RowSum).  The
    // reference applies the global one first and then overwrites window 0 with the local
    // result; the plan's "global instead of the local fan-out" reading would leave the
    // row-sum value in window 0, and a missing global branch would leave the sentinel in
    // window 1 (the `!global` gate suppresses the `z <- r` fallback).
    const auto run = run_apply({.gpc = GpcMode::RowSum, .local = LocalMode::Window0});

    auto& c = scope.counter;

    const auto D_local  = oracle_block_jacobi_inv(run.A);
    const auto z_local  = oracle_apply(D_local, run.r);
    const auto R_global = oracle_row_sum_inv(run.A);
    const auto z_global = oracle_apply(R_global, run.r);

    const auto z_l0 = window_of(z_local, 0);
    const auto z_g0 = window_of(z_global, 0);
    const auto z_g1 = window_of(z_global, 1);
    const auto z_00 = window_of(run.z, 0);
    const auto z_01 = window_of(run.z, 1);

    const double d0       = max_abs_diff(z_00, z_l0);  // window 0 == the LOCAL result
    const double d1       = max_abs_diff(z_01, z_g1);  // window 1 == the GLOBAL result
    const double differs0 = max_abs_diff(z_l0, z_g0);  // the two M's really differ

    int kept0 = 0, kept1 = 0;
    for(double z : z_00)
        kept0 += (z == -12345.0) ? 1 : 0;
    for(double z : z_01)
        kept1 += (z == -12345.0) ? 1 : 0;

    T83G_LOG("[t83g] precedence: |z[w0] - local| %.3e, |z[w1] - global| %.3e, "
             "|local[w0] - global[w0]| %.3e, sentinels kept %d/%d\n",
             d0,
             d1,
             differs0,
             kept0,
             kept1);

    T83G_CHECK(c, differs0 > 1e-2);  // the test can tell the two M's apart
    T83G_CHECK(c, kept1 == 0);       // the global branch really wrote window 1 ...
    T83G_CHECK(c, d1 == 0.0);
    T83G_CHECK(c, kept0 == 0);
    T83G_CHECK(c, d0 == 0.0);  // ... and the local fan-out then overwrote window 0
    // (a global-instead-of-local implementation fails `d0`; a local-instead-of-global one
    // fails `d1`/`kept1`; a missing local loop fails `d0`; a missing global branch leaves
    // window 1's sentinels.  z[w0] must NOT be the global answer.)
    T83G_CHECK(c, max_abs_diff(z_00, z_g0) > 1e-2);
}

// ===========================================================================
//  CASE 6 — no host read crept into the iteration
// ===========================================================================
TEST_CASE("gpc_no_sync", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_no_sync"};

    const auto r = default_rhs();

    struct Counters
    {
        lud::LStreamCounters delta;
    };

    auto measure = [&](const RunOptions& options)
    {
        Counters out;
        with_world(options,
                   [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
                   {
                       (void)gls;
                       impl.build_linear_system();

                       lla::LDenseVector<double> r_dev, z_dev;
                       r_dev.resize(r.size());
                       z_dev.resize(r.size());
                       r_dev.copy_from(span<const double>{r});
                       z_dev.copy_from(span<const double>{r});

                       lud::LDeviceVar<IndexT> d_converged;
                       d_converged.copy_from(IndexT{0});

                       // one apply inside an explicitly opened list: the preconditioner
                       // must only *record*, never synchronize (a host read of
                       // `converged` would be one sync per PCG iteration)
                       auto&      s = gg::frame_stream();
                       s.synchronize();  // finish the assembly before the measurement
                       s.reset_counters();
                       const auto before = s.counters();
                       s.begin();
                       impl.apply_preconditioner(z_dev.view(),
                                                 r_dev.cview(),
                                                 d_converged.view(),
                                                 nullptr);
                       s.commit();
                       out.delta = s.counters() - before;
                       s.synchronize();
                   });
        return out;
    };

    const auto with_global = measure({.gpc = GpcMode::BlockJacobi, .local = LocalMode::None});
    const auto with_local  = measure({.gpc = GpcMode::None, .local = LocalMode::DiagAll});

    auto& c = scope.counter;
    T83G_LOG("[t83g] counters (global): commands %llu, commits %llu, syncs %llu, "
             "uploads %llu, direct %llu\n",
             static_cast<unsigned long long>(with_global.delta.commands),
             static_cast<unsigned long long>(with_global.delta.commits),
             static_cast<unsigned long long>(with_global.delta.syncs),
             static_cast<unsigned long long>(with_global.delta.uploads),
             static_cast<unsigned long long>(with_global.delta.direct_submits));
    T83G_LOG("[t83g] counters (local):  commands %llu, commits %llu, syncs %llu, "
             "uploads %llu, direct %llu\n",
             static_cast<unsigned long long>(with_local.delta.commands),
             static_cast<unsigned long long>(with_local.delta.commits),
             static_cast<unsigned long long>(with_local.delta.syncs),
             static_cast<unsigned long long>(with_local.delta.uploads),
             static_cast<unsigned long long>(with_local.delta.direct_submits));

    T83G_CHECK(c, with_global.delta.syncs == 0ull);
    T83G_CHECK(c, with_global.delta.uploads == 0ull);
    T83G_CHECK(c, with_local.delta.syncs == 0ull);
    T83G_CHECK(c, with_local.delta.uploads == 0ull);
    // batching: the global apply went into the open list, it did not escape it
    T83G_CHECK(c, with_global.delta.commands >= 1ull);
    T83G_CHECK(c, with_global.delta.direct_submits == 0ull);
    T83G_CHECK(c, with_local.delta.direct_submits == 0ull);
}

// ===========================================================================
//  CASE 7 — the `converged` gate is a device value
// ===========================================================================
TEST_CASE("gpc_converged_gate", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_converged_gate"};

    const auto options =
        RunOptions{.gpc = GpcMode::BlockJacobi, .local = LocalMode::None};

    const auto awake = run_apply(options, IndexT{0});
    const auto done  = run_apply(options, IndexT{1});

    auto&      c = scope.counter;
    const auto D = oracle_block_jacobi_inv(awake.A);

    int kept = 0;
    for(double z : done.z)
        kept += (z == -12345.0) ? 1 : 0;

    T83G_LOG("[t83g] converged gate: converged != 0 kept %d/%d sentinels, "
             "converged == 0 recomputed (|z|max %.3e)\n",
             kept,
             kDofs,
             max_abs_diff(awake.z, std::vector<double>(awake.z.size(), 0.0)));

    // `if(*converged != 0) return;` — the same gate the shipped abd/fem preconditioners
    // use (abd_diag_preconditioner.cpp:84), read on the *device*
    T83G_CHECK(c, kept == kDofs);
    T83G_CHECK(c, max_abs_diff(done.z, std::vector<double>(done.z.size(), -12345.0)) == 0.0);
    T83G_CHECK(c, max_abs_diff(awake.z, oracle_apply(D, awake.r)) == 0.0);
}

// ===========================================================================
//  CASE 8 — two registrations: silent overwrite, the slot keeps the last
// ===========================================================================
TEST_CASE("gpc_second_registration_behaviour", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_second_registration_behaviour"};

    // DEVIATION (reported): the plan asked for "UIPC_ASSERT message shape matches the
    // reference" on a second registration.  Neither backend has such an assert --
    // `SimSystemSlot::register_sim_system()` silently overwrites
    // (src/backends/common/details/sim_system_slot.inl:7-13) -- so this case pins the
    // *measured* behaviour instead of inventing a cross-backend divergence.
    //
    // Everything happens inside ONE world: the fixtures are owned by it, so a pointer
    // captured from one `with_world` must never be compared inside another.
    with_world({.gpc            = GpcMode::BlockJacobi,
                .local          = LocalMode::None,
                .second_global  = true},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl)
               {
                   (void)gls;
                   auto& c = scope.counter;

                   REQUIRE(T83gGlobalPrecond::registration_log.size() == 2u);
                   T83G_CHECK(c, T83gGlobalPrecond::registration_log.size() == 2u);
                   auto* first  = T83gGlobalPrecond::registration_log[0];
                   auto* second = T83gGlobalPrecond::registration_log[1];
                   T83G_CHECK(c, first != second);

                   // no abort, and the slot holds the *last* registration
                   REQUIRE(static_cast<bool>(impl.global_preconditioner));
                   auto* slot = impl.global_preconditioner.view();
                   T83G_CHECK(c, slot == second);
                   T83G_CHECK(c, slot != first);
                   T83G_LOG("[t83g] two registrations: slot holds the last of the two "
                            "(%d of %zu), no abort\n",
                            slot == second ? 2 : 1,
                            T83gGlobalPrecond::registration_log.size());

                   // the surviving preconditioner is a working one
                   impl.build_linear_system();
                   auto A = read_A(impl.bcoo_A);
                   REQUIRE(A.block_rows == static_cast<SizeT>(kBlockRows));

                   const auto r = default_rhs();
                   lla::LDenseVector<double> r_dev, z_dev;
                   r_dev.resize(r.size());
                   z_dev.resize(r.size());
                   r_dev.copy_from(span<const double>{r});
                   z_dev.copy_from(span<const double>{r});

                   lud::LDeviceVar<IndexT> d_converged;
                   d_converged.copy_from(IndexT{0});

                   gg::frame_stream().synchronize();
                   impl.apply_preconditioner(z_dev.view(),
                                             r_dev.cview(),
                                             d_converged.view(),
                                             nullptr);
                   gg::frame_stream().synchronize();

                   std::vector<double> z(r.size(), 0.0);
                   z_dev.copy_to(span<double>{z});
                   T83G_CHECK(c, max_abs_diff(z, oracle_apply(oracle_block_jacobi_inv(A), r))
                                        == 0.0);
               });
}

// ===========================================================================
//  CASE 9 — PCG through the shipped LinearPCG: global == local, both < none
// ===========================================================================
TEST_CASE("gpc_pcg_iteration_count", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_pcg_iteration_count"};

    const auto none = run_pcg({.gpc = GpcMode::None, .local = LocalMode::None});
    const auto local = run_pcg({.gpc = GpcMode::None, .local = LocalMode::DiagAll});
    const auto global = run_pcg({.gpc = GpcMode::BlockJacobi, .local = LocalMode::None});

    auto& c = scope.counter;
    const auto A = dense_A(none.A);

    T83G_LOG("[t83g] pcg: iters none %zu, local block-Jacobi %zu, global block-Jacobi "
             "%zu; |x_global - x_local| %.3e, ||b-Ax||/||b|| %.3e\n",
             static_cast<std::size_t>(none.iters),
             static_cast<std::size_t>(local.iters),
             static_cast<std::size_t>(global.iters),
             max_abs_diff(global.x, local.x),
             relative_residual(A, global.b, global.x));

    // The two M's are the *same matrix*: the local table is the union of the two windows'
    // `diag_inv_row` results, the global one is `diag_inv_row` over all 8 block rows, and
    // both run the identical factored routine on the identical assembled A.  So the
    // iteration counts must coincide exactly, and x with them.
    T83G_CHECK(c, global.iters == local.iters);
    T83G_CHECK(c, local.iters < none.iters);  // block-Jacobi is discriminating
    T83G_CHECK(c, none.iters > 1u);
    T83G_CHECK(c, global.iters > 1u);
    T83G_CHECK(c, max_abs_diff(global.x, local.x) <= 1e-10);
    T83G_CHECK(c, max_abs_diff(none.x, local.x) > 1e-6);
    // both converge: ||b - Ax||/||b|| <= sqrt(tol_rate) (the reference's criterion is
    // |rz| <= tol*|rz_0|; with block-Jacobi that is not ||r||^2, so the derived bound is
    // the loose one, exactly as in luisa_pcg_test.cpp)
    T83G_CHECK(c, relative_residual(A, local.b, local.x) <= std::sqrt(1e-3));
    T83G_CHECK(c, relative_residual(A, global.b, global.x) <= std::sqrt(1e-3));
}

// ===========================================================================
//  CASE 10 — a wrong global M changes the count (the negative control)
// ===========================================================================
TEST_CASE("gpc_pcg_wrong_M_changes_count", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    CaseScope scope{"gpc_pcg_wrong_M_changes_count"};

    const auto good = run_pcg({.gpc = GpcMode::BlockJacobi, .local = LocalMode::None});
    const auto wrong = run_pcg({.gpc = GpcMode::WrongBlock, .local = LocalMode::None});

    auto&      c = scope.counter;
    const auto A = dense_A(wrong.A);

    T83G_LOG("[t83g] pcg wrong M: iters block-Jacobi %zu, wrong-block %zu; "
             "||b-Ax||/||b|| %.3e\n",
             static_cast<std::size_t>(good.iters),
             static_cast<std::size_t>(wrong.iters),
             relative_residual(A, wrong.b, wrong.x));

    // a wrong but SPD M: the solve still converges, only the count moves
    T83G_CHECK(c, wrong.iters != good.iters);
    T83G_CHECK(c, std::isfinite(relative_residual(A, wrong.b, wrong.x)));
    T83G_CHECK(c, relative_residual(A, wrong.b, wrong.x) <= std::sqrt(1e-3));
    T83G_CHECK(c, max_abs_diff(good.x, wrong.x) > 1e-9);
}

// ===========================================================================
//  CASE 11 — per-frame iteration counts: BLOCKED
// ===========================================================================
TEST_CASE("gpc_frame_pcg", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    SKIP(
        "BLOCKED-BY-S0.6 and plan 10/11: multi-frame world.advance() deadlocks after "
        "frame 1 (pre-existing defect, AGENT_ENV_BRIEF), and the engine never calls "
        "GlobalLinearSystem::solve() yet (grep -n \"solve_linear_system|"
        "build_linear_system\" src/backends/luisa/sim_engine.cpp -> no hits), so there is "
        "no per-frame linear_solver_iterations to compare. The per-solve iteration counts "
        "that do not need the engine loop are asserted by gpc_pcg_iteration_count.");
}

// ===========================================================================
//  CASE 12 — a CUDA-dumped A: NOT REACHABLE here
// ===========================================================================
TEST_CASE("gpc_from_cuda_A", "[luisa][global_precond]")
{
    using namespace t83g;
    test_runtime();
    SKIP(
        "no CUDA ground truth: --backend_cuda=y fails at configure time "
        "(src/backends/cuda/components.lua:86: unknown interface: raise()), so there is no "
        "CUDA run to dump A.mtx/b.mtx. The dump path itself exists since T8.3b "
        "(GlobalLinearSystem::_dump_A_b -> utils/matrix_market.h, exercised by "
        "[luisa][mtx] and [luisa][solve]), so only the cross-backend comparison is "
        "missing. The Class-A gate is covered in-process instead: "
        "gpc_apply_classA_blockjacobi compares z with the factored "
        "fem_diag_preconditioner_detail::{diag_inv_row, diag_apply_row} over the "
        "*assembled* BCOO, Δ == 0.");
}
}  // namespace t83g
