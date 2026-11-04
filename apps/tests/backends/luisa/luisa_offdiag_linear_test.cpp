/********************************************************************
 * @file luisa_offdiag_linear_test.cpp
 * @brief T8.3a-1 — the off-diagonal linear subsystem: registration, the block
 *        layout (extent + triplet budget) and the two `.submatrix(...)` windows
 *        the global assembler builds for it.
 *
 * WHAT IS UNDER TEST (the reference contract, src/backends/cuda/linear_system/)
 * ---------------------------------------------------------------------------
 *  * `OffDiagLinearSubsystem` (off_diag_linear_subsystem.{h,cpp}) — the
 *    `connect(l, r)` handshake, `uid() == (l.uid, r.uid)` (the deterministic
 *    block-ordering key) and the `require<GlobalLinearSystem>() ->
 *    add_subsystem(this)` registration;
 *  * `GlobalLinearSystem::add_subsystem(OffDiagLinearSubsystem*)` — a real
 *    registration (it was a no-op until this task);
 *  * `Impl::init()`'s off-diag half — the uid sort, the `diag + off_diag`
 *    `subsystem_infos` table (`index = diag_count + i`, `is_diag == false`) and
 *    `off_diag_lr_triplet_counts`;
 *  * `Impl::_update_subsystem_extent()`'s off-diag branch — per subsystem
 *    `{lr, rl}` and `subsystem_triplet_counts[idx] = lr + rl`, i.e. the *triplet
 *    budget*: `Σ diag nnz + Σ(lr + rl)`.  An under-budget allocation does not
 *    truncate on the raw-view path, it silently overwrites the *next*
 *    subsystem's slice (cuda/.../views.h compiles the device checks out), so the
 *    budget is asserted explicitly;
 *  * `Impl::_assemble_linear_system()`'s off-diag branch — the `lr` window at
 *    `(l_off, r_off)` and the `rl` window at `(r_off, l_off)`, both cut from the
 *    subsystem's own triplet slice (`rl_offset == lr_offset + lr_count`).
 *
 * THE ORACLES (all derived in this file; the CUDA backend is not host-includable)
 * ---------------------------------------------------------------------------
 *  1) a *dense* Eigen oracle built from the same block tables the kernels write
 *     (`kDiagADescs`/`kDiagBDescs`/`kOffDiagDescs` + `block_entry`), i.e. an
 *     independent re-derivation of "where every block must land";
 *  2) exactness where the arithmetic is identical: every block entry is a dyadic
 *     rational written verbatim by the kernel, there are no atomics and no
 *     reductions over duplicate keys, so the assembled values are compared
 *     **bitwise** (Δ == 0);
 *  3) the *shipped* downstream path (`ge2sym` -> `convert` -> `BCOO` ->
 *     `sym2ge`) against the same dense oracle at 1e-14 relative, plus exact
 *     symmetry of the pre-`ge2sym` triplet matrix.  Symmetry is asserted
 *     *in-process on the general (pre-`ge2sym`) matrix*, never on a
 *     Matrix-Market file: the reference's dump is upper-packed by construction,
 *     so a file-level `A == Aᵀ` compares an upper triangle with itself (§2.3 of
 *     the research note).
 *
 * THE NEGATIVE CONTROL (the test's non-vacuity proof)
 * --------------------------------------------------
 * `offdiag_assembly` runs the *whole* pipeline twice — once normally and once
 * with the off-diag blocks forced to zero — and requires
 *   * the triplet *pattern* to be identical (22 triplets, same (row, col)
 *     multiset) while
 *   * the assembled values differ by a large margin from the correct oracle and
 *     agree exactly with an oracle that has no off-diag contribution.
 * An implementation that dropped the off-diag blocks therefore cannot pass: the
 * first assembly's `Δ == 0` assertion is what fails.
 *
 * THE DOCUMENTED ONE-SIDED CASE (`offdiag_reversed_pair`)
 * ------------------------------------------------------
 * `ge2sym` keeps only `row <= col`.  The reference's only consumer
 * (abd_fem_linear_subsystem.cu: `report_extent(lr, 0)` + an `lr_hessian()`
 * write) therefore loses its contribution whenever the pair is registered as
 * `connect(fem, abd)` — silently.  This test *pins* that behaviour instead of
 * claiming symmetry, so nobody "fixes" it silently later (research §2.1/§2.2).
 *
 * WHAT THIS FILE CANNOT REACH
 * ---------------------------
 * No `world.advance()` anywhere (the multi-frame advance deadlocks after frame 1
 * — a pre-existing defect owned by S0.6), and no CUDA-side ground truth: the
 * `offdiag_needs_cuda_ground_truth` case SKIPs with the reason string instead of
 * passing silently.
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
#include <linear_system/global_linear_system_function.h>
#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/off_diag_linear_subsystem.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
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
namespace lgls = uipc::backend::luisa::global_linear_system_detail;
namespace lod  = uipc::backend::luisa::off_diag_layout_detail;

namespace t83
{
// ===========================================================================
//  1) the block tables — ONE source of truth for the device writes and the
//     host oracle
// ===========================================================================
/// `GlobalLinearSystem::DoFBlockSize`: 3 dofs per block row.
constexpr int kBlockDim = 3;
/// DiagA (6 dofs) owns block rows 0..1, DiagB (12 dofs) block rows 2..5.
constexpr int kDenseDim  = 18;
constexpr int kBlockRows = kDenseDim / kBlockDim;  // 6

/// Reserved uids (the mocks and real systems elsewhere use 0/1/2/9/10/15/16/...
/// — see the T4.x/T5.x/T6.x test files; neither value collides).
constexpr U64 kUidDiagA = 0x54383344ull;  // "T83D"
constexpr U64 kUidDiagB = 0x54383345ull;

constexpr SizeT kDiagADofs   = 6;
constexpr SizeT kDiagBDofs   = 12;
constexpr SizeT kDiagABlocks = 4;
constexpr SizeT kDiagBBlocks = 8;
constexpr SizeT kLrBlocks    = 5;
constexpr SizeT kRlBlocks    = 5;

/// The shared-binary marker (cookbook §7.14 rule 6): every system registered in
/// this file reports a zero extent unless a geometry of the *current* scene
/// carries this vertex attribute, so the fixtures are inert in every other test
/// file's scene.
constexpr const char* kMarker = "t83_offdiag_marker";

/// One written block of a fixture's pattern: where it goes *inside the view*, and
/// which entry of the value table it carries.  `transposed` stores the transpose
/// of `block_entry(k, ...)`, which is how a subsystem reports the *lower* mirror
/// of a symmetric block (a Hessian block `A(j, i)` is `A(i, j)ᵀ`).
struct BlockDesc
{
    int  row        = 0;
    int  col        = 0;
    int  k          = 0;
    bool transposed = false;
};

/// DiagA: a full 2x2 block pattern that deliberately contains one *strictly
/// lower* block (1, 0) — `ge2sym` must drop exactly that one, so the pre-/post-
/// `ge2sym` count difference is observable — and that block is the transpose of
/// its (0, 1) counterpart, i.e. the matrix DiagA reports is symmetric (a Hessian
/// block is; `ge2sym` is only sound on a symmetric input).
inline constexpr BlockDesc kDiagADescs[] = {
    {0, 0, 0, false}, {0, 1, 1, false}, {1, 0, 1, true}, {1, 1, 2, false}};

/// DiagB: 8 of the 16 blocks of a 4x4 block grid, all upper (`row <= col`).
inline constexpr BlockDesc kDiagBDescs[] = {{0, 0, 0, false},
                                            {1, 1, 1, false},
                                            {2, 2, 2, false},
                                            {3, 3, 3, false},
                                            {0, 1, 4, false},
                                            {1, 2, 5, false},
                                            {2, 3, 6, false},
                                            {0, 3, 7, false}};

/// The off-diag pattern as a 2 x 4 block grid over (left-local, right-local):
/// 5 of the 8 possible blocks, deliberately asymmetric, so a missing transpose in
/// the `rl` mirror is visible.  The reference's `OffDiagInfo` exposes both views
/// precisely so one stiffness expression can fill both.
inline constexpr BlockDesc kOffDiagDescs[] = {
    {0, 0, 0, false}, {0, 1, 1, false}, {0, 2, 2, false}, {0, 3, 3, false}, {1, 0, 4, false}};

static_assert(std::size(kDiagADescs) == kDiagABlocks);
static_assert(std::size(kDiagBDescs) == kDiagBBlocks);
static_assert(std::size(kOffDiagDescs) == kLrBlocks);

/// Entry (r, c) of block `k` of the *off-diagonal* pattern.  Dyadic rationals
/// only (an integer over 8), so the device and the host evaluate *the same*
/// doubles and the value comparison can be bitwise.  Deliberately non-symmetric
/// (`(r, c)` and `(c, r)` differ) so a missing transpose in the `rl` mirror is
/// visible.
[[nodiscard]] inline double block_entry(int k, int r, int c) noexcept
{
    return static_cast<double>((k + 1) * 16 + r * kBlockDim + c) / 8.0;
}

/// Entry (r, c) of block `k` of a *diagonal* subsystem.  A diagonal Hessian block
/// is symmetric (`A(r, c) == A(c, r)`) — `ge2sym`/`sym2ge` never symmetrise a
/// diagonal block, they copy it verbatim, so an asymmetric one would make the
/// whole assembled matrix asymmetric for real.  The transpose flag of a diag
/// descriptor is therefore a no-op (`m^T == m`), which the test pins by reporting
/// DiagA's lower block (1, 0) as the *transpose* of its (0, 1) block.
[[nodiscard]] inline double diagonal_block_entry(int k, int r, int c) noexcept
{
    const int lo = r < c ? r : c;
    const int hi = r < c ? c : r;
    return static_cast<double>((k + 1) * 16 + lo * kBlockDim + hi) / 8.0;
}

using DenseA = Eigen::Matrix<double, kDenseDim, kDenseDim>;

struct OracleOptions
{
    /// include the off-diag `lr` contribution
    bool off_diag_lr = true;
    /// include the off-diag `rl` (transposed) contribution
    bool off_diag_rl = true;
    /// the pair is registered as connect(diag_B, diag_A), i.e. left = DiagB
    bool reverse = false;
};

/// The dense (18 x 18) matrix a correct assembler must produce, derived from the
/// block tables above — *not* from anything the device wrote.  This is the matrix
/// **as reported by the subsystems**: a diagonal subsystem only reports its upper
/// (`row <= col`) blocks (the solver's packed convention), so the result is not
/// symmetric; `symmetric_closure()` below is the matrix the linear system actually
/// *represents*.
[[nodiscard]] DenseA build_oracle(const OracleOptions& opt = {})
{
    DenseA A = DenseA::Zero();

    auto put = [&A](int block_row, int block_col, int k, bool transposed, bool diagonal)
    {
        for(int c = 0; c < kBlockDim; ++c)
            for(int r = 0; r < kBlockDim; ++r)
            {
                const int rr = transposed ? c : r;
                const int cc = transposed ? r : c;
                A(block_row * kBlockDim + r, block_col * kBlockDim + c) =
                    diagonal ? diagonal_block_entry(k, rr, cc) : block_entry(k, rr, cc);
            }
    };

    // the two diagonal subsystems: DiagA at block 0, DiagB at block 2 (uid order)
    for(const auto& d : kDiagADescs)
        put(d.row, d.col, d.k, d.transposed, /*diagonal=*/true);
    for(const auto& d : kDiagBDescs)
        put(2 + d.row, 2 + d.col, d.k, d.transposed, /*diagonal=*/true);

    // the off-diag pair.  `lr` is the (left, right) block and `rl` the (right,
    // left) one: the second is always the first one's transposed mirror, and
    // which of the two *windows* is the mirrored one depends on the connection
    // (`connect(diag_a, diag_b)` has A on the left, `connect(diag_b, diag_a)` has
    // B on the left, and the pattern is defined in (A-local, B-local) terms).
    const int l_off = opt.reverse ? 2 : 0;
    const int r_off = opt.reverse ? 0 : 2;
    if(opt.off_diag_lr)
        for(const auto& d : kOffDiagDescs)
        {
            if(opt.reverse)
                put(l_off + d.col, r_off + d.row, d.k, !d.transposed, false);
            else
                put(l_off + d.row, r_off + d.col, d.k, d.transposed, false);
        }
    if(opt.off_diag_rl)
        for(const auto& d : kOffDiagDescs)
        {
            if(opt.reverse)
                put(r_off + d.row, l_off + d.col, d.k, d.transposed, false);
            else
                put(r_off + d.col, l_off + d.row, d.k, !d.transposed, false);
        }

    return A;
}

/// The symmetric matrix the linear system *represents*: every `row <= col` block
/// mirrored below as its transpose, exactly what `sym2ge`/`rbk_sym_spmv` do with
/// the packed upper form.  Any reported strictly-lower block is overwritten by the
/// mirror (which is what `ge2sym` does to it).
[[nodiscard]] DenseA symmetric_closure(const DenseA& reported)
{
    DenseA out = reported;
    for(int bi = 0; bi < kBlockRows; ++bi)
        for(int bj = bi + 1; bj < kBlockRows; ++bj)
            out.block<kBlockDim, kBlockDim>(bj * kBlockDim, bi * kBlockDim) =
                reported
                    .block<kBlockDim, kBlockDim>(bi * kBlockDim, bj * kBlockDim)
                    .transpose();
    return out;
}

/// `max |A(l,r) - A(r,l)^T|` over the two off-diag block windows: the property a
/// two-sided `OffDiagInfo` write establishes and a one-sided one violates.
[[nodiscard]] double off_diag_mirror_error(const DenseA& A, int l_off, int l_n, int r_off, int r_n)
{
    double worst = 0.0;
    for(int i = 0; i < l_n; ++i)
        for(int j = 0; j < r_n; ++j)
            for(int r = 0; r < kBlockDim; ++r)
                for(int c = 0; c < kBlockDim; ++c)
                {
                    const double a =
                        A((l_off + i) * kBlockDim + r, (r_off + j) * kBlockDim + c);
                    const double b =
                        A((r_off + j) * kBlockDim + c, (l_off + i) * kBlockDim + r);
                    worst = std::max(worst, std::abs(a - b));
                }
    return worst;
}

// ---------------------------------------------------------------------------
//  small helpers (the T54/T61b test files' idioms)
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

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
        std::printf("[t83] %s: %d checks\n", name.c_str(), counter.checks);
    }
};

#define T83_CHECK(counter, cond)                                               \
    do                                                                         \
    {                                                                          \
        ++(counter).checks;                                                    \
        CHECK(cond);                                                           \
    } while(false)

/// `max |a - b|` over two dense matrices.
[[nodiscard]] inline double max_abs_diff(const DenseA& a, const DenseA& b)
{
    return (a - b).cwiseAbs().maxCoeff();
}

/// `max |a - b|` relative to the oracle's magnitude (the T0/T5 tolerance band).
[[nodiscard]] inline double rel_diff(const DenseA& a, const DenseA& b)
{
    const double scale = std::max(1.0, b.cwiseAbs().maxCoeff());
    return max_abs_diff(a, b) / scale;
}

// ---------------------------------------------------------------------------
//  the triplet <-> dense bridge (host side)
// ---------------------------------------------------------------------------
using HostTriplet = lla::MatrixTriplet<double, kBlockDim, kBlockDim>;

/// Dense matrix of a general (not necessarily symmetric) triplet matrix.  Duplicate
/// (row, col) keys are *summed*, which is also what the BCOO conversion does — the
/// callers assert uniqueness separately so this cannot hide an overlap.
[[nodiscard]] inline DenseA dense_from_triplets(const std::vector<HostTriplet>& ts)
{
    DenseA A = DenseA::Zero();
    for(const auto& t : ts)
    {
        for(int c = 0; c < kBlockDim; ++c)
            for(int r = 0; r < kBlockDim; ++r)
                A(t.row_index * kBlockDim + r, t.col_index * kBlockDim + c) +=
                    t.value(r, c);
    }
    return A;
}

[[nodiscard]] inline std::map<std::pair<int, int>, int> key_multiset(const std::vector<HostTriplet>& ts)
{
    std::map<std::pair<int, int>, int> keys;
    for(const auto& t : ts)
        ++keys[{t.row_index, t.col_index}];
    return keys;
}

// ===========================================================================
//  2) the device write kernels
// ===========================================================================
using TripletDeviceView =
    lla::LTripletMatrixDeviceView<lcrt::Var<double>, kBlockDim, kBlockDim>;

/// A 3x3 *off-diagonal* block built element-wise from the host table (trace-time
/// constants).  `scale` is a launch argument (1.0 normally, 0.0 for the negative
/// control), so one compiled kernel serves both assemblies and the *pattern* stays
/// identical.
[[nodiscard]] inline lps::LMat<lcrt::Var<double>, kBlockDim, kBlockDim> device_block(
    int k, bool transposed, const lcrt::Var<double>& scale)
{
    return lps::make_mat<lcrt::Var<double>, kBlockDim, kBlockDim>(
        [k, transposed, &scale](int r, int c) noexcept
        {
            const int rr = transposed ? c : r;
            const int cc = transposed ? r : c;
            return lcrt::Var<double>{block_entry(k, rr, cc)} * scale;
        });
}

/// A 3x3 *diagonal* block (symmetric in the block, like a real Hessian block).
[[nodiscard]] inline lps::LMat<lcrt::Var<double>, kBlockDim, kBlockDim> device_diagonal_block(int k, bool transposed)
{
    return lps::make_mat<lcrt::Var<double>, kBlockDim, kBlockDim>(
        [k, transposed](int r, int c) noexcept
        {
            const int rr = transposed ? c : r;
            const int cc = transposed ? r : c;
            // symmetric in the block, so `m^T == m`
            return lcrt::Var<double>{diagonal_block_entry(k, rr, cc)};
        });
}

/// The 12 launch arguments of a triplet view, flattened exactly like
/// `TripletDeviceView::from_handles_full` expects them.
struct ViewLaunch
{
    lud::LBufferView<std::int32_t> rows;
    lud::LBufferView<std::int32_t> cols;
    lud::LBufferView<double>       vals;
    std::int32_t                   total_rows          = 0;
    std::int32_t                   total_cols          = 0;
    std::int32_t                   triplet_offset      = 0;
    std::int32_t                   triplet_count       = 0;
    std::int32_t                   total_triplet_count = 0;
    std::int32_t                   submatrix_offset_x  = 0;
    std::int32_t                   submatrix_offset_y  = 0;
    std::int32_t                   submatrix_extent_x  = 0;
    std::int32_t                   submatrix_extent_y  = 0;
};

[[nodiscard]] inline ViewLaunch to_launch(lla::LTripletMatrixView<double, kBlockDim, kBlockDim> v)
{
    return ViewLaunch{v.row_indices(),
                      v.col_indices(),
                      v.values(),
                      static_cast<std::int32_t>(v.total_rows()),
                      static_cast<std::int32_t>(v.total_cols()),
                      static_cast<std::int32_t>(v.triplet_index_offset()),
                      static_cast<std::int32_t>(v.triplet_count()),
                      static_cast<std::int32_t>(v.total_triplet_count()),
                      static_cast<std::int32_t>(v.submatrix_offset().x),
                      static_cast<std::int32_t>(v.submatrix_offset().y),
                      static_cast<std::int32_t>(v.extent().x),
                      static_cast<std::int32_t>(v.extent().y)};
}

/// Record one single-threaded assembly dispatch for `v` (plus the fixture's own
/// extra arguments) onto the backend's frame stream.
template <typename Shader, typename... Extra>
void launch(Shader& shader, const ViewLaunch& v, Extra&&... extra)
{
    gg::frame_stream().record(shader(v.rows,
                                     v.cols,
                                     v.vals,
                                     v.total_rows,
                                     v.total_cols,
                                     v.triplet_offset,
                                     v.triplet_count,
                                     v.total_triplet_count,
                                     v.submatrix_offset_x,
                                     v.submatrix_offset_y,
                                     v.submatrix_extent_x,
                                     v.submatrix_extent_y,
                                     std::forward<Extra>(extra)...)
                                  .dispatch(1));
}

/// The reusable kernel parameter list (the reference's `TripletMatrixView` shape).
#define T83_VIEW_KERNEL_ARGS                                                      \
    lcrt::Buffer<std::int32_t>, lcrt::Buffer<std::int32_t>, lcrt::Buffer<double>, \
        std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t,     \
        std::int32_t, std::int32_t, std::int32_t, std::int32_t

#define T83_VIEW_KERNEL_PARAMS                                                      \
    lcrt::BufferVar<std::int32_t> rows, lcrt::BufferVar<std::int32_t> cols,         \
        lcrt::BufferVar<double> vals, lcrt::Var<std::int32_t> total_rows,           \
        lcrt::Var<std::int32_t> total_cols, lcrt::Var<std::int32_t> triplet_offset, \
        lcrt::Var<std::int32_t> triplet_count,                                      \
        lcrt::Var<std::int32_t> total_triplet_count,                                \
        lcrt::Var<std::int32_t> sm_offset_x, lcrt::Var<std::int32_t> sm_offset_y,   \
        lcrt::Var<std::int32_t> sm_extent_x, lcrt::Var<std::int32_t> sm_extent_y

#define T83_MAKE_DEVICE_VIEW                                                   \
    TripletDeviceView::from_handles_full(rows,                                 \
                                         cols,                                 \
                                         vals,                                 \
                                         total_rows,                           \
                                         total_cols,                           \
                                         triplet_offset,                       \
                                         triplet_count,                        \
                                         total_triplet_count,                  \
                                         sm_offset_x,                          \
                                         sm_offset_y,                          \
                                         sm_extent_x,                          \
                                         sm_extent_y)

/// DiagA: the 4 blocks of `kDiagADescs`, in the diagonal window's local
/// coordinates (the window's submatrix offset puts them at block rows 0..1).
auto& diag_a_kernel()
{
    static auto& k = lla::detail::cached_kernel<T83_VIEW_KERNEL_ARGS>(
        [](T83_VIEW_KERNEL_PARAMS) noexcept
        {
            auto H = T83_MAKE_DEVICE_VIEW;
            for(int i = 0; i < static_cast<int>(kDiagABlocks); ++i)
            {
                const auto& d = kDiagADescs[i];
                H(i).write(d.row, d.col, device_diagonal_block(d.k, d.transposed));
            }
        });
    return k;
}

/// DiagB: the 8 blocks of `kDiagBDescs` (block rows 2..5 globally).
auto& diag_b_kernel()
{
    static auto& k = lla::detail::cached_kernel<T83_VIEW_KERNEL_ARGS>(
        [](T83_VIEW_KERNEL_PARAMS) noexcept
        {
            auto H = T83_MAKE_DEVICE_VIEW;
            for(int i = 0; i < static_cast<int>(kDiagBBlocks); ++i)
            {
                const auto& d = kDiagBDescs[i];
                H(i).write(d.row, d.col, device_diagonal_block(d.k, d.transposed));
            }
        });
    return k;
}

/// The off-diag *pattern* kernel: block `i` at local `(d.row, d.col)`.
auto& offdiag_pattern_kernel()
{
    static auto& k = lla::detail::cached_kernel<T83_VIEW_KERNEL_ARGS, double>(
        [](T83_VIEW_KERNEL_PARAMS, lcrt::Var<double> scale) noexcept
        {
            auto H = T83_MAKE_DEVICE_VIEW;
            for(int i = 0; i < static_cast<int>(kLrBlocks); ++i)
            {
                const auto& d = kOffDiagDescs[i];
                H(i).write(d.row, d.col, device_block(d.k, d.transposed, scale));
            }
        });
    return k;
}

/// The off-diag *mirror* kernel: block `i` at local `(d.col, d.row)` with the
/// block transposed — the (right, left) window of the same coupling.
auto& offdiag_mirror_kernel()
{
    static auto& k = lla::detail::cached_kernel<T83_VIEW_KERNEL_ARGS, double>(
        [](T83_VIEW_KERNEL_PARAMS, lcrt::Var<double> scale) noexcept
        {
            auto H = T83_MAKE_DEVICE_VIEW;
            for(int i = 0; i < static_cast<int>(kLrBlocks); ++i)
            {
                const auto& d = kOffDiagDescs[i];
                H(i).write(d.col, d.row, device_block(d.k, !d.transposed, scale));
            }
        });
    return k;
}

#undef T83_MAKE_DEVICE_VIEW
#undef T83_VIEW_KERNEL_PARAMS
#undef T83_VIEW_KERNEL_ARGS

// ===========================================================================
//  3) the scene + the marker gate
// ===========================================================================
/// A tetmesh with *no* constitution: the scene only has to exist (the engine
/// builds every registered system for every scene of the shared test binary) and
/// carry the marker.  No dof-producing system claims it, so the engine-level dof
/// layout below is entirely this file's fixtures.
[[nodiscard]] Scene build_scene()
{
    auto config                      = Scene::default_config();
    config["contact"]["enable"]      = IndexT{0};
    config["sanity_check"]["enable"] = IndexT{0};
    config["integrator"]["type"]     = std::string{"bdf1"};
    config["dt"]                     = 0.01;

    Scene scene{config};
    auto  object = scene.objects().create("t83");

    const std::vector<Vector3> positions = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    const std::vector<Vector4i> tets = {{0, 1, 2, 3}};

    auto mesh = tetmesh(span<const Vector3>{positions}, span<const Vector4i>{tets});
    label_surface(mesh);
    mesh.vertices().create<IndexT>(kMarker, 1);
    object->geometries().create(mesh);

    return scene;
}

/// `world()` inside a `SimSystem` is the engine's `WorldVisitor` (the T54 idiom:
/// `world().scene().geometries()`), which is this file's only scene access.
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

// ===========================================================================
//  4) the three fixtures — all REGISTER_SIM_SYSTEM types (a SimSystem cannot be
//     constructed without a SimEngine&), each gated on the marker
// ===========================================================================
class T83DiagA final : public gg::DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    static inline bool enabled = false;

    static void reset() { enabled = false; }

  protected:
    [[nodiscard]] U64 get_uid() const noexcept override { return kUidDiagA; }

    void do_init(InitInfo& info) override
    {
        (void)info;
        // set by `Impl::init()`'s step 1, i.e. *before* `report_init_extent`
        enabled = has_marker(world());
    }

    void do_report_init_extent(gg::GlobalLinearSystem::InitDofExtentInfo& info) override
    {
        info.extent(enabled ? kDiagADofs : SizeT{0});
    }

    void do_receive_init_dof_info(gg::GlobalLinearSystem::InitDofInfo& info) override
    {
        (void)info;  // `dof_offset()` / `dof_count()` read the collection directly
    }

    void do_report_extent(gg::GlobalLinearSystem::DiagExtentInfo& info) override
    {
        info.extent(enabled ? kDiagABlocks : SizeT{0}, enabled ? kDiagADofs : SizeT{0});
    }

    void do_assemble(gg::GlobalLinearSystem::DiagInfo& info) override
    {
        if(!enabled)
            return;
        auto hessians = info.hessians();
        if(static_cast<SizeT>(hessians.triplet_count()) < kDiagABlocks)
            return;
        launch(diag_a_kernel(), to_launch(hessians));
    }

    void do_accuracy_check(gg::GlobalLinearSystem::AccuracyInfo& info) override
    {
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

class T83DiagB final : public gg::DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    static inline bool enabled = false;

    static void reset() { enabled = false; }

  protected:
    [[nodiscard]] U64 get_uid() const noexcept override { return kUidDiagB; }

    void do_init(InitInfo& info) override
    {
        (void)info;
        enabled = has_marker(world());
    }

    void do_report_init_extent(gg::GlobalLinearSystem::InitDofExtentInfo& info) override
    {
        info.extent(enabled ? kDiagBDofs : SizeT{0});
    }

    void do_receive_init_dof_info(gg::GlobalLinearSystem::InitDofInfo& info) override
    {
        (void)info;
    }

    void do_report_extent(gg::GlobalLinearSystem::DiagExtentInfo& info) override
    {
        info.extent(enabled ? kDiagBBlocks : SizeT{0}, enabled ? kDiagBDofs : SizeT{0});
    }

    void do_assemble(gg::GlobalLinearSystem::DiagInfo& info) override
    {
        if(!enabled)
            return;
        auto hessians = info.hessians();
        if(static_cast<SizeT>(hessians.triplet_count()) < kDiagBBlocks)
            return;
        launch(diag_b_kernel(), to_launch(hessians));
    }

    void do_accuracy_check(gg::GlobalLinearSystem::AccuracyInfo& info) override
    {
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

/// The off-diag subsystem between them.  Its block tables are the file's
/// `kOffDiagDescs`; the three static switches drive the three scenarios:
///  * `reverse`  — `connect(diag_b, diag_a)` instead of `(diag_a, diag_b)`
///                 (`offdiag_reversed_pair`);
///  * `lr_only`  — report `{lr, 0}` and touch only `lr_hessian()`, i.e. exactly
///                 the shape of the reference's only consumer
///                 (abd_fem_linear_subsystem.cu) (`offdiag_reversed_pair`);
///  * `zero_blocks` — write the blocks with *zero* values (identical pattern),
///                 the negative control of `offdiag_assembly`.
class T83OffDiag final : public gg::OffDiagLinearSubsystem
{
  public:
    using OffDiagLinearSubsystem::OffDiagLinearSubsystem;

    static inline bool enabled     = false;
    static inline bool reverse     = false;
    static inline bool lr_only     = false;
    static inline bool zero_blocks = false;

    static void reset()
    {
        enabled     = false;
        reverse     = false;
        lr_only     = false;
        zero_blocks = false;
    }

  protected:
    void do_build(BuildInfo& info) override
    {
        auto& a = require<T83DiagA>();
        auto& b = require<T83DiagB>();
        // BuildInfo::connect() is the *only* way the global system learns the two
        // block coordinates, and it must happen here (the base asserts both ends
        // are set before it registers).
        if(reverse)
            info.connect(&b, &a);
        else
            info.connect(&a, &b);
    }

    void do_init(InitInfo& info) override
    {
        // The base's default body is the reference's own no-op; it is called
        // explicitly so the module import is a *used* one.  A symbol reachable only
        // through a vtable slot never enters the exe's import table, so
        // scripts/luisa_module_def.py cannot see it and the next link fails
        // (cookbook §7.28 rule 7 / §7.39).
        OffDiagLinearSubsystem::do_init(info);
        enabled = has_marker(world());
    }

    void report_extent(gg::GlobalLinearSystem::OffDiagExtentInfo& info) override
    {
        if(!enabled)
        {
            info.extent(SizeT{0}, SizeT{0});
            return;
        }
        info.extent(kLrBlocks, lr_only ? SizeT{0} : kRlBlocks);
    }

    void assemble(gg::GlobalLinearSystem::OffDiagInfo& info) override
    {
        if(!enabled)
            return;

        const double scale = zero_blocks ? 0.0 : 1.0;

        // The pattern is defined in (DiagA-local, DiagB-local) terms.  The `lr`
        // window is the (left, right) block of the pair, so it carries the pattern
        // *transposed* when the left end is DiagB (the two windows have different
        // shapes: 2x4 for (A, B) and 4x2 for (B, A)); `rl` is the mirror.
        auto lr = info.lr_hessian();
        if(static_cast<SizeT>(lr.triplet_count()) >= kLrBlocks)
        {
            if(reverse)
                launch(offdiag_mirror_kernel(), to_launch(lr), scale);
            else
                launch(offdiag_pattern_kernel(), to_launch(lr), scale);
        }

        auto rl = info.rl_hessian();
        if(static_cast<SizeT>(rl.triplet_count()) >= kRlBlocks)
        {
            if(reverse)
                launch(offdiag_pattern_kernel(), to_launch(rl), scale);
            else
                launch(offdiag_mirror_kernel(), to_launch(rl), scale);
        }
    }
};

// ===========================================================================
//  5) the probe: captures the fixtures once the engine has built them
// ===========================================================================
class T83Probe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::GlobalLinearSystem* gls      = nullptr;
    inline static T83DiagA*               diag_a   = nullptr;
    inline static T83DiagB*               diag_b   = nullptr;
    inline static T83OffDiag*             off_diag = nullptr;

    static void reset()
    {
        gls      = nullptr;
        diag_a   = nullptr;
        diag_b   = nullptr;
        off_diag = nullptr;
    }

  private:
    void do_build() override
    {
        gls      = &require<gg::GlobalLinearSystem>();
        diag_a   = &require<T83DiagA>();
        diag_b   = &require<T83DiagB>();
        off_diag = &require<T83OffDiag>();
    }
};
}  // namespace t83

REGISTER_SIM_SYSTEM(t83::T83DiagA);
REGISTER_SIM_SYSTEM(t83::T83DiagB);
REGISTER_SIM_SYSTEM(t83::T83OffDiag);
REGISTER_SIM_SYSTEM(t83::T83Probe);

namespace t83
{
// ===========================================================================
//  the driver: a test-local `GlobalLinearSystem::Impl`
// ===========================================================================
/// Register the three fixtures in a *test-local* `Impl` and run the shipped
/// `init()` (the off-diag half: sort + subsystem table + `{lr, rl}` sizing).
/// The engine's own `Impl` is not touched, so the layout is exactly these three
/// subsystems regardless of which other systems the scene builds.
void init_impl(gg::GlobalLinearSystem::Impl& impl)
{
    impl.diag_subsystems.register_sim_system(*T83Probe::diag_a);
    impl.diag_subsystems.register_sim_system(*T83Probe::diag_b);
    impl.off_diag_subsystems.register_sim_system(*T83Probe::off_diag);
    impl.init();
}

[[nodiscard]] std::vector<HostTriplet> read_triplets(
    const lla::LTripletMatrix<double, kBlockDim, kBlockDim>& matrix)
{
    std::vector<HostTriplet> ts(static_cast<std::size_t>(matrix.triplet_count()));
    matrix.copy_to_triplets(span<HostTriplet>{ts});
    return ts;
}

/// The whole shipped downstream path: `ge2sym` -> `convert` -> `BCOO` ->
/// `sym2ge`, and the dense matrix that comes out.  Returns the intermediate
/// BCOO key count through `bcoo_count` (used for the "which side survived
/// ge2sym" assertions).
[[nodiscard]] DenseA assemble_pipeline(gg::GlobalLinearSystem::Impl& impl, SizeT& bcoo_count)
{
    impl.converter.ge2sym(impl.triplet_A);
    impl.converter.convert(impl.triplet_A, impl.bcoo_A);

    lla::LBCOOMatrix<double, kBlockDim, kBlockDim> general;
    impl.converter.sym2ge(impl.bcoo_A, general);

    bcoo_count = static_cast<SizeT>(impl.bcoo_A.non_zeros());
    return dense_from_triplets(read_triplets(general));
}
}  // namespace t83

// ===========================================================================
//  CASE 1 — the layout: registration, the subsystem table and the triplet budget
// ===========================================================================
TEST_CASE("offdiag_layout", "[luisa][offdiag]")
{
    using namespace t83;
    test_runtime();
    T83Probe::reset();
    T83DiagA::reset();
    T83DiagB::reset();
    T83OffDiag::reset();
    CaseScope scope{"offdiag_layout"};

    // ---- (A) the duplicate/self-pair predicate (no engine needed) -----------
    {
        auto& c = scope.counter;
        // distinct pairs: ok
        const std::vector<U64> ok_l{10ull, 20ull, 30ull};
        const std::vector<U64> ok_r{20ull, 30ull, 10ull};
        auto ok = lod::check_unique_pairs(span<const U64>{ok_l}, span<const U64>{ok_r});
        T83_CHECK(c, ok.ok);
        T83_CHECK(c, !ok.self);

        // the same (l, r) twice -> duplicate, not self
        const std::vector<U64> dup_l{10ull, 20ull, 10ull};
        const std::vector<U64> dup_r{20ull, 30ull, 20ull};
        auto dup = lod::check_unique_pairs(span<const U64>{dup_l}, span<const U64>{dup_r});
        T83_CHECK(c, !dup.ok);
        T83_CHECK(c, !dup.self);
        T83_CHECK(c, dup.first == 0u);
        T83_CHECK(c, dup.second == 2u);

        // l == r for one entry -> self
        const std::vector<U64> self_l{10ull, 20ull};
        const std::vector<U64> self_r{20ull, 20ull};
        auto                   self =
            lod::check_unique_pairs(span<const U64>{self_l}, span<const U64>{self_r});
        T83_CHECK(c, !self.ok);
        T83_CHECK(c, self.self);
        T83_CHECK(c, self.first == 1u);
        T83_CHECK(c, self.second == 1u);

        // the reversed pair is *not* a duplicate (it owns the transposed block)
        const std::vector<U64> rev_l{10ull, 20ull};
        const std::vector<U64> rev_r{20ull, 10ull};
        auto rev = lod::check_unique_pairs(span<const U64>{rev_l}, span<const U64>{rev_r});
        T83_CHECK(c, rev.ok);

        // an empty collection is trivially ok
        auto empty = lod::check_unique_pairs(span<const U64>{}, span<const U64>{});
        T83_CHECK(c, empty.ok);

        std::printf("[t83] layout: predicate ok/dup/self/rev = %d/%d/%d/%d\n",
                    ok.ok ? 1 : 0,
                    dup.ok ? 1 : 0,
                    self.ok ? 1 : 0,
                    rev.ok ? 1 : 0);
    }

    // ---- (B) the real registration through the engine ----------------------
    auto   scene = build_scene();
    Engine engine{"luisa", workspace_path("uipc_luisa_t83a_layout")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T83Probe::gls != nullptr);
    REQUIRE(T83Probe::diag_a != nullptr);
    REQUIRE(T83Probe::diag_b != nullptr);
    REQUIRE(T83Probe::off_diag != nullptr);
    REQUIRE(T83Probe::diag_a->is_valid());
    REQUIRE(T83Probe::diag_b->is_valid());
    REQUIRE(T83Probe::off_diag->is_valid());

    {
        auto& c = scope.counter;
        // T8.3b: the engine *does* call `GlobalLinearSystem::init()` now
        // (`SimEngine::init_scene()`, the reference's position), so the engine-level
        // layout is already done here and the extents below are read from it.  The
        // test-local `Impl` elsewhere in this file still drives the same subsystems
        // independently, which is what proves the two families coexist in one `Impl`.
        T83_CHECK(c, T83Probe::gls->dof_count() == kDenseDim);
        // the diag offsets are the uid-ascending scan: DiagA (0x..44) first
        T83_CHECK(c, T83Probe::diag_a->dof_offset() == 0);
        T83_CHECK(c, T83Probe::diag_a->dof_count() == static_cast<IndexT>(kDiagADofs));
        T83_CHECK(c, T83Probe::diag_b->dof_offset() == static_cast<IndexT>(kDiagADofs));
        T83_CHECK(c, T83Probe::diag_b->dof_count() == static_cast<IndexT>(kDiagBDofs));
        T83_CHECK(c, T83OffDiag::enabled);
        T83_CHECK(c, T83DiagA::enabled);
        T83_CHECK(c, T83DiagB::enabled);
        std::printf("[t83] layout: engine dof_count %zu (DiagA %d+%d, DiagB %d+%d)\n",
                    static_cast<size_t>(T83Probe::gls->dof_count()),
                    static_cast<int>(T83Probe::diag_a->dof_offset()),
                    static_cast<int>(T83Probe::diag_a->dof_count()),
                    static_cast<int>(T83Probe::diag_b->dof_offset()),
                    static_cast<int>(T83Probe::diag_b->dof_count()));
    }

    // ---- (C) the test-local Impl: the off-diag half of `init()` ------------
    {
        gg::GlobalLinearSystem::Impl impl;
        init_impl(impl);
        auto& c = scope.counter;

        // diag first, off-diag after — with `index = diag_count + i`
        REQUIRE(impl.subsystem_infos.size() == 3u);
        T83_CHECK(c, impl.subsystem_infos[0].is_diag);
        T83_CHECK(c, impl.subsystem_infos[1].is_diag);
        T83_CHECK(c, !impl.subsystem_infos[2].is_diag);
        T83_CHECK(c, impl.subsystem_infos[0].index == 0u);
        T83_CHECK(c, impl.subsystem_infos[1].index == 1u);
        T83_CHECK(c, impl.subsystem_infos[2].index == 2u);
        T83_CHECK(c, impl.subsystem_infos[0].local_index == 0u);
        T83_CHECK(c, impl.subsystem_infos[1].local_index == 1u);
        T83_CHECK(c, impl.subsystem_infos[2].local_index == 0u);

        // the dof layout is Σ diag extents (the off-diag subsystems add none)
        T83_CHECK(c, impl.diag_dof_offsets_counts.total_count() == kDenseDim);
        T83_CHECK(c, impl.diag_dof_offsets_counts.counts()[0] == static_cast<IndexT>(kDiagADofs));
        T83_CHECK(c, impl.diag_dof_offsets_counts.counts()[1] == static_cast<IndexT>(kDiagBDofs));

        // the off-diag `{lr, rl}` split is only *sized* by `init()` (zero-filled:
        // the extents are reported one step later) ...
        REQUIRE(impl.off_diag_lr_triplet_counts.size() == 1u);
        T83_CHECK(c, impl.off_diag_lr_triplet_counts[0].x == 0u);
        T83_CHECK(c, impl.off_diag_lr_triplet_counts[0].y == 0u);

        // ... and `_update_subsystem_extent()` fills the budget
        REQUIRE(impl._update_subsystem_extent());
        const std::vector<IndexT> expected_counts{static_cast<IndexT>(kDiagABlocks),
                                                  static_cast<IndexT>(kDiagBBlocks),
                                                  static_cast<IndexT>(kLrBlocks + kRlBlocks)};
        const std::vector<IndexT> expected_offsets{
            0, static_cast<IndexT>(kDiagABlocks), static_cast<IndexT>(kDiagABlocks + kDiagBBlocks)};
        std::vector<IndexT> counts(
            impl.subsystem_triplet_offsets_counts.counts().begin(),
            impl.subsystem_triplet_offsets_counts.counts().end());
        std::vector<IndexT> offsets(
            impl.subsystem_triplet_offsets_counts.offsets().begin(),
            impl.subsystem_triplet_offsets_counts.offsets().end());
        T83_CHECK(c, counts == expected_counts);
        T83_CHECK(c, offsets == expected_offsets);
        T83_CHECK(c, impl.off_diag_lr_triplet_counts[0].x == kLrBlocks);
        T83_CHECK(c, impl.off_diag_lr_triplet_counts[0].y == kRlBlocks);

        // the triplet budget: Σ diag nnz + Σ(lr + rl).  Under-budget would
        // silently overwrite the *next* subsystem's slice.
        const SizeT expected_total = kDiagABlocks + kDiagBBlocks + kLrBlocks + kRlBlocks;
        T83_CHECK(c, static_cast<SizeT>(impl.triplet_A.non_zeros()) == expected_total);
        T83_CHECK(c, static_cast<SizeT>(impl.triplet_A.triplet_capacity()) >= expected_total);
        std::printf(
            "[t83] layout: counts {%d,%d,%d} offsets {%d,%d,%d} "
            "lr/rl {%zu,%zu} budget %d\n",
            static_cast<int>(counts[0]),
            static_cast<int>(counts[1]),
            static_cast<int>(counts[2]),
            static_cast<int>(offsets[0]),
            static_cast<int>(offsets[1]),
            static_cast<int>(offsets[2]),
            impl.off_diag_lr_triplet_counts[0].x,
            impl.off_diag_lr_triplet_counts[0].y,
            impl.triplet_A.non_zeros());
    }
}

// ===========================================================================
//  CASE 2 — the assembly: block coordinates, values, budget and symmetry
// ===========================================================================
TEST_CASE("offdiag_assembly", "[luisa][offdiag]")
{
    using namespace t83;
    test_runtime();
    T83Probe::reset();
    T83DiagA::reset();
    T83DiagB::reset();
    T83OffDiag::reset();
    CaseScope scope{"offdiag_assembly"};

    auto   scene = build_scene();
    Engine engine{"luisa", workspace_path("uipc_luisa_t83a_assembly")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T83Probe::off_diag != nullptr);

    gg::GlobalLinearSystem::Impl impl;
    init_impl(impl);
    auto& c = scope.counter;

    REQUIRE(impl._update_subsystem_extent());
    {
        const SizeT expected_total = kDiagABlocks + kDiagBBlocks + kLrBlocks + kRlBlocks;
        T83_CHECK(c, static_cast<SizeT>(impl.triplet_A.non_zeros()) == expected_total);
    }

    // ---- (A) the pre-ge2sym triplet: pattern, budget and values ------------
    // (the *general* matrix: both off-diag windows are present, and DiagA's
    //  strictly-lower block (1, 0) is still there)
    impl._assemble_linear_system();
    gg::frame_stream().synchronize();

    auto ts = read_triplets(impl.triplet_A);
    {
        REQUIRE(ts.size() == kDiagABlocks + kDiagBBlocks + kLrBlocks + kRlBlocks);

        // every slot was written (the assembler pre-fills row/col with -1)
        bool any_unwritten = false;
        for(const auto& t : ts)
            any_unwritten |= (t.row_index < 0 || t.col_index < 0);
        T83_CHECK(c, !any_unwritten);

        // the hand-derived (row, col) multiset: each key exactly once
        std::map<std::pair<int, int>, int> expected_keys;
        for(const auto& d : kDiagADescs)
            ++expected_keys[{d.row, d.col}];
        for(const auto& d : kDiagBDescs)
            ++expected_keys[{2 + d.row, 2 + d.col}];
        for(const auto& d : kOffDiagDescs)  // the lr window at (0, 2)
            ++expected_keys[{d.row, 2 + d.col}];
        for(const auto& d : kOffDiagDescs)  // the rl window at (2, 0)
            ++expected_keys[{2 + d.col, d.row}];
        T83_CHECK(c, key_multiset(ts) == expected_keys);

        // ... and every block value is the table's (bitwise: same arithmetic,
        // no atomics, no duplicate keys)
        const auto oracle    = build_oracle();
        const auto dense_pre = dense_from_triplets(ts);
        T83_CHECK(c, max_abs_diff(dense_pre, oracle) == 0.0);

        // the off-diag pair must be written on BOTH sides: `A(l, r) == A(r, l)^T`
        // exactly.  A one-sided assembly (the plan's §7 trap) fails right here,
        // and this is the only place where it can be caught: `ge2sym` drops the
        // strictly-lower window, and after that the loss is invisible.
        const double mirror = off_diag_mirror_error(dense_pre, 0, 2, 2, 4);
        T83_CHECK(c, mirror == 0.0);

        // NOTE what is deliberately NOT asserted: the reported matrix as a whole is
        // *not* symmetric, because a diagonal subsystem reports only its upper
        // (`row <= col`) blocks -- that is the packed convention `ge2sym` relies on
        // (the closure is `symmetric_closure()`, checked in (B)).
        std::printf(
            "[t83] assembly: %zu triplets, %zu distinct keys, "
            "|A_pre - oracle| %.3e, |A(l,r) - A(r,l)^T| %.3e\n",
            ts.size(),
            key_multiset(ts).size(),
            max_abs_diff(dense_pre, oracle),
            mirror);
    }

    // ---- (B) the shipped downstream path: ge2sym -> BCOO -> sym2ge ----------
    {
        SizeT      bcoo_count = 0;
        const auto dense_out  = assemble_pipeline(impl, bcoo_count);

        // ge2sym keeps row <= col: DiagA loses its (1, 0) block and the whole
        // rl window (it is strictly lower), so 3 + 8 + 5 of the 22 survive
        const SizeT expected_bcoo = (kDiagABlocks - 1u) + kDiagBBlocks + kLrBlocks;
        T83_CHECK(c, bcoo_count == expected_bcoo);

        // ... and the surviving keys are exactly the upper half
        auto bcoo_triplets = read_triplets(impl.bcoo_A);
        bool all_upper     = true;
        for(const auto& t : bcoo_triplets)
            all_upper &= (t.row_index <= t.col_index);
        T83_CHECK(c, all_upper);
        for(const auto& t : bcoo_triplets)
            T83_CHECK(c, (t.row_index >= 0 && t.col_index >= 0));

        // `sym2ge` reconstructs the *general* matrix from the packed upper form:
        // it must equal the symmetric closure of the reported matrix exactly (it
        // only mirrors and re-sorts, no arithmetic)
        const auto oracle = symmetric_closure(build_oracle());
        T83_CHECK(c, max_abs_diff(dense_out, oracle) == 0.0);
        // ... and it is symmetric -- asserted on a matrix that really came out of
        // the solver-side path, not on a Matrix-Market file (whose upper-packed
        // dump would be compared with itself; research §2.3)
        T83_CHECK(c, max_abs_diff(dense_out, DenseA{dense_out.transpose()}) == 0.0);
        // ... and it is the oracle, within the parity tolerance band
        const double rel = rel_diff(dense_out, oracle);
        T83_CHECK(c, rel <= 1e-14);
        std::printf(
            "[t83] assembly: BCOO %zu blocks (hand-derived %zu), "
            "|sym2ge(bcoo) - symmetric_closure(oracle)| rel %.3e\n",
            bcoo_count,
            expected_bcoo,
            rel);
    }

    // ---- (C) the NEGATIVE CONTROL: the off-diag blocks forced to zero ------
    // Same pattern, same kernel, `scale = 0`: the assembled values must change by
    // a large margin, and must then agree with an oracle that has no off-diag
    // block at all.  An implementation that dropped the off-diag contribution
    // would have produced *this* matrix in (A) — where the Δ == 0 assertion is.
    {
        T83OffDiag::zero_blocks = true;

        // `ge2sym` shrank `triplet_A` to the packed upper part in (B), so the
        // extent path has to run again before the assembler can use the full
        // 22-block budget (that is also the idempotency of the extent step).
        REQUIRE(impl._update_subsystem_extent());
        T83_CHECK(c,
                  static_cast<SizeT>(impl.triplet_A.non_zeros())
                      == kDiagABlocks + kDiagBBlocks + kLrBlocks + kRlBlocks);
        impl._assemble_linear_system();
        gg::frame_stream().synchronize();

        auto ts_zero = read_triplets(impl.triplet_A);
        REQUIRE(ts_zero.size() == ts.size());
        // the *pattern* is unchanged — only the values differ
        T83_CHECK(c, key_multiset(ts_zero) == key_multiset(ts));

        const auto dense_zero = dense_from_triplets(ts_zero);
        const auto without_offdiag =
            build_oracle(OracleOptions{.off_diag_lr = false, .off_diag_rl = false});
        const auto with_offdiag = build_oracle();

        T83_CHECK(c, max_abs_diff(dense_zero, without_offdiag) == 0.0);
        const double dropped = max_abs_diff(dense_zero, with_offdiag);
        // ... and it is *measurably* different from the correct assembly, so the
        // positive assertions in (A)/(B) are not vacuous
        T83_CHECK(c, dropped > 1e-3);
        T83_CHECK(c, max_abs_diff(dense_zero, dense_from_triplets(ts)) > 1e-3);

        // the same difference survives the whole downstream path
        SizeT      bcoo_zero      = 0;
        const auto dense_out_zero = assemble_pipeline(impl, bcoo_zero);
        T83_CHECK(c, bcoo_zero == (kDiagABlocks - 1u) + kDiagBBlocks + kLrBlocks);
        // the pipeline's output is the *symmetric closure* (what the solver sees)
        const double dropped_out =
            max_abs_diff(dense_out_zero, symmetric_closure(with_offdiag));
        T83_CHECK(c, dropped_out > 1e-3);
        T83_CHECK(c, max_abs_diff(dense_out_zero, symmetric_closure(without_offdiag)) == 0.0);

        std::printf(
            "[t83] assembly: negative control (off-diag blocks zeroed): "
            "pattern identical, |A - oracle| %.3e, |A - A_without_offdiag| %.3e\n",
            dropped,
            max_abs_diff(dense_zero, without_offdiag));

        T83OffDiag::zero_blocks = false;
    }
}

// ===========================================================================
//  CASE 3 — the documented one-sided loss (`connect(b, a)` + `report_extent(lr, 0)`)
// ===========================================================================
TEST_CASE("offdiag_reversed_pair", "[luisa][offdiag]")
{
    using namespace t83;
    test_runtime();
    T83Probe::reset();
    T83DiagA::reset();
    T83DiagB::reset();
    T83OffDiag::reset();
    CaseScope scope{"offdiag_reversed_pair"};

    // the reference's only consumer shape: the pair is registered in the opposite
    // order and only `lr_hessian()` is written (abd_fem_linear_subsystem.cu:119-148)
    T83OffDiag::reverse = true;
    T83OffDiag::lr_only = true;

    auto   scene = build_scene();
    Engine engine{"luisa", workspace_path("uipc_luisa_t83a_reversed")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    gg::GlobalLinearSystem::Impl impl;
    init_impl(impl);
    auto& c = scope.counter;

    // the uid of the off-diag subsystem follows the *connection*, not the type:
    // (diag_b, diag_a) sorts after (diag_a, diag_b), but it is the only one here
    REQUIRE(impl.off_diag_lr_triplet_counts.size() == 1u);

    REQUIRE(impl._update_subsystem_extent());
    // the budget follows the reported extent: lr only, no rl block at all
    T83_CHECK(c, impl.off_diag_lr_triplet_counts[0].x == kLrBlocks);
    T83_CHECK(c, impl.off_diag_lr_triplet_counts[0].y == 0u);
    T83_CHECK(c, static_cast<SizeT>(impl.triplet_A.non_zeros()) == kDiagABlocks + kDiagBBlocks + kLrBlocks);

    impl._assemble_linear_system();
    gg::frame_stream().synchronize();

    auto ts = read_triplets(impl.triplet_A);
    {
        REQUIRE(ts.size() == kDiagABlocks + kDiagBBlocks + kLrBlocks);

        // pre-ge2sym the matrix is *not* symmetric: the contribution sits at
        // (block 2+q, block 0+p), i.e. strictly *below* the diagonal, because the
        // left end of the pair is DiagB (block rows 2..5).  The `rl` window was
        // never written, so the pair is *one-sided* -- the property that makes the
        // contribution disappear below (a two-sided write is mirror-consistent,
        // see `offdiag_assembly` (A)).
        const auto oracle_lr_only = build_oracle(
            OracleOptions{.off_diag_lr = true, .off_diag_rl = false, .reverse = true});
        const auto dense_pre = dense_from_triplets(ts);
        T83_CHECK(c, max_abs_diff(dense_pre, oracle_lr_only) == 0.0);
        const double one_sided = off_diag_mirror_error(dense_pre, 2, 4, 0, 2);
        T83_CHECK(c, one_sided > 1e-3);

        // the off-diag blocks are the *lower* ones: every non-diag off-diag key
        // has row > col
        bool offdiag_is_lower = true;
        for(const auto& t : ts)
            if(t.row_index >= 2 && t.col_index < 2)
                offdiag_is_lower &= (t.row_index > t.col_index);
        T83_CHECK(c, offdiag_is_lower);

        std::printf(
            "[t83] reversed: %zu triplets, |A_pre - oracle_lr_only| %.3e, "
            "|A(l,r) - A(r,l)^T| %.3e (one-sided)\n",
            ts.size(),
            max_abs_diff(dense_pre, oracle_lr_only),
            one_sided);
    }

    {
        // ge2sym keeps row <= col, so *every* off-diag block of the reversed pair
        // is dropped: the contribution vanishes with no diagnostic.  That is the
        // reference's behaviour (its only consumer writes one side), pinned here
        // so nobody "fixes" it silently — a real fix is the consumer's job.
        SizeT      bcoo_count = 0;
        const auto dense_out  = assemble_pipeline(impl, bcoo_count);

        const SizeT expected_bcoo = (kDiagABlocks - 1u) + kDiagBBlocks;  // 11
        T83_CHECK(c, bcoo_count == expected_bcoo);

        // the pipeline's output is the *symmetric closure*, i.e. what the solver
        // would consume: the closure of the full pair (what a two-sided write of
        // the same coupling would have produced) ...
        const auto   with_offdiag = symmetric_closure(build_oracle(
            OracleOptions{.off_diag_lr = true, .off_diag_rl = true, .reverse = true}));
        const double lost         = max_abs_diff(dense_out, with_offdiag);
        T83_CHECK(c, lost > 1e-3);

        // ... versus what actually came out: exactly the closure of the two
        // diagonal subsystems, i.e. the off-diag contribution is gone
        const auto diagonal_only = symmetric_closure(
            build_oracle(OracleOptions{.off_diag_lr = false, .off_diag_rl = false}));
        T83_CHECK(c, max_abs_diff(dense_out, diagonal_only) == 0.0);

        std::printf(
            "[t83] reversed: BCOO %zu blocks (hand-derived %zu), the "
            "one-sided contribution is gone: |A - A_with_offdiag| %.3e\n",
            bcoo_count,
            expected_bcoo,
            lost);
    }
}

// ===========================================================================
//  CASE 4 — the CUDA ground truth: BLOCKED (SKIP, never a silent pass)
// ===========================================================================
TEST_CASE("offdiag_needs_cuda_ground_truth", "[luisa][offdiag]")
{
    test_runtime();
    SKIP(
        "cuda backend unavailable: --backend_cuda=y fails at configure, no dump in the Luisa "
        "backend (plan 11), no off-diag consumer (plan 18)");
}
