#pragma once
/********************************************************************
 * @file mas_preconditioner_engine_function.h
 * @brief T6.4c — the MAS engine's host/device-generic helpers, factored so
 *        the shipped shaders and the unit test run the *same* code.
 *
 * Same recipe as T5.4b's `fem_diag_preconditioner_function.h` and T6.4a's
 * `abd_tolerance_checker_function.h` (cookbook §7.4 / §7.20 rule 1 / §7.28
 * rule 5): every routine whose decision can be expressed as a *value* is written
 * once, templated on the scalar/container flavour, so the unit test can instantiate
 * it with `storage::memory` views (host `double`) and compare it against the device
 * instantiation that the shipped kernel calls.
 *
 * The reference is `src/backends/cuda/finite_element/mas_preconditioner_engine.cu`
 * (1726 lines) + `.h` (205).  It is *not* reconstructable from memory: the level
 * table/colouring, the per-bank shared-memory exchange and the per-cluster
 * Gauss-Jordan are transcribed from it line by line (see the mapping table in
 * `mas_preconditioner_engine.cpp`).
 *
 * WHAT IS FACTORED HERE
 * ---------------------
 *  1. the integer laws of the algorithm (`sym_index`, `bank_align`, `mat_entry`,
 *     `lanemask_lt`, `num_levels`, `level1_begin`) — host and device identical
 *     because they are `constexpr` int math;
 *  2. the device/host scalar gaps the algorithm needs and the numerics layer does
 *     not offer as one call: `popcount_scalar`, `ctz_scalar`, `min_scalar`
 *     (`lcrt::min` is host-invisible, `std::min` device-invisible);
 *  3. `transitive_closure_mask` — the reference's per-lane closure loop
 *     (`visited ^ connect_msk` -> `__ffs(todo) - 1` -> `cache_mask(next_visit)`),
 *     rewritten as a *fixed* 16-iteration loop (RULE M2: the reference's loop bound
 *     is device data; the closure grows by at least one lane per iteration, and a
 *     bank has BANKSIZE lanes, so 16 iterations is exact — the reference's own
 *     `while(connect_msk != ~0U)` early exit is subsumed because OR-ing an
 *     already-visited lane's mask is idempotent).  The mask *source* is a callback,
 *     so the device instantiation reads the per-lane `$shared` slot and the host
 *     instantiation reads a plain array — one body, two flavours.
 *  4. the flat cluster-matrix addressing and its 3x3 block readers/writers (the
 *     port stores the reference's array-of-120-3x3-matrices as one flat scalar
 *     buffer; see the deviation note in `mas_preconditioner_engine.h`).
 *********************************************************************/
#include <type_define.h>
#include <finite_element/mas_preconditioner_engine.h>
#include <luisa_device/linear/linear.h>
#include <utils/scalar_util.h>

#include <bit>
#include <cstdint>

namespace uipc::backend::luisa
{
namespace mas_preconditioner_detail
{
namespace lla = luisa_device::linalg;

inline constexpr int kBanksize        = MASPreconditionerEngine::BANKSIZE;         // 16
inline constexpr int kClusterDim      = MASPreconditionerEngine::CLUSTER_MAT_DIM;  // 48
inline constexpr int kSymBlockCount   = MASPreconditionerEngine::SYM_BLOCK_COUNT;  // 136
inline constexpr int kEntriesPerClust = MASPreconditionerEngine::CLUSTER_ENTRIES;  // 1224
inline constexpr int kVec3PerNode     = MASPreconditionerEngine::VEC3_PER_NODE;    // 3

// ---------------------------------------------------------------------------
//  1. integer laws (host and device identical)
// ---------------------------------------------------------------------------
/// The reference's `sym_index(row, col)`: the upper-triangle block index of a
/// BANKSIZE x BANKSIZE block matrix.  Generic over the index flavour: host `int`
/// indices (the table/oracle code) and DSL indices (the kernels) both work, because
/// every operation is integer arithmetic.
template <typename R, typename C>
[[nodiscard]] inline auto sym_index(const R& row, const C& col) noexcept
{
    return kBanksize * row - row * (row + 1) / 2 + col;
}

/// The reference's `bank_align(n)`.
[[nodiscard]] constexpr int bank_align(int n) noexcept
{
    return (n + kBanksize - 1) / kBanksize * kBanksize;
}

/// Column-major 3x3 entry index (Eigen's default storage order, i.e. the flat
/// layout of every `LMatrix3x3`/`Matrix3x3` buffer the port touches).  Generic over
/// the index flavour, like `sym_index`.
template <typename R, typename C>
[[nodiscard]] inline auto mat_entry(const R& row, const C& col) noexcept
{
    return col * 3 + row;
}

/// The reference's `lanemask_lt(lane_id)`: the bits below `lane_id`.  Generic over
/// the flavour so a kernel's `lane_id` (a DSL value) can be passed directly.
template <typename L>
[[nodiscard]] inline auto lanemask_lt(const L& lane_id) noexcept
{
    return (std::uint32_t{1} << lane_id) - std::uint32_t{1};
}

/// The flat index of the 3x3 block `(br, bc)` of a cluster matrix inside the
/// port's flat scalar cluster buffer (`si * 9 + mat_entry(i, j)`).
[[nodiscard]] inline int cluster_block_base(int br, int bc) noexcept
{
    return sym_index(br, bc) * 9;
}

/// The reference's `compute_num_levels`: the number of hierarchy levels of a
/// problem with `vert_num` nodes, capped at MAX_LEVELS.
///
/// `n_level = 1; level_sz = bank_align(vert_num); while(level_sz > 16) {
///     level_sz /= 16; n_level++; level_sz = bank_align(level_sz); } n_level++;`
[[nodiscard]] constexpr int num_levels(int vert_num) noexcept
{
    int n_level  = 1;
    int level_sz = bank_align(vert_num);
    while(level_sz > kBanksize)
    {
        level_sz /= kBanksize;
        n_level++;
        level_sz = bank_align(level_sz);
    }
    n_level++;
    return n_level < MASPreconditionerEngine::MAX_LEVELS ? n_level
                                                         : MASPreconditionerEngine::MAX_LEVELS;
}

/// The reference's `level1_begin`:
/// `bank_align(max(bank_align(N), m_total_nodes))`.
[[nodiscard]] constexpr int level1_begin(int mapped_nodes, int total_nodes) noexcept
{
    const int padded_N = bank_align(mapped_nodes);
    return bank_align(padded_N > total_nodes ? padded_N : total_nodes);
}

// ---------------------------------------------------------------------------
//  2. scalar gaps (host scalar / DSL scalar)
// ---------------------------------------------------------------------------
/// `__popc` / `std::popcount`.
template <typename X>
[[nodiscard]] inline auto popcount_scalar(const X& x) noexcept
{
    if constexpr(lps::is_dsl_value_v<X>)
        return lcrt::popcount(x);
    else
        return static_cast<std::uint32_t>(
            std::popcount(static_cast<std::uint32_t>(x)));
}

/// `__ffs(x) - 1` (the lowest set bit index); **32** when `x == 0`, exactly like
/// CUDA's `__ffs(0) == 0` minus one -- the callers test `j < lane_count` rather
/// than relying on the value.
template <typename X>
[[nodiscard]] inline auto ctz_scalar(const X& x) noexcept
{
    if constexpr(lps::is_dsl_value_v<X>)
        return lcrt::ctz(x);
    else
        return static_cast<std::uint32_t>(
            std::countr_zero(static_cast<std::uint32_t>(x)));
}

/// `min` for a host scalar or a DSL scalar (`lcrt::min` is not reachable from host
/// code and `std::min` is not a DSL intrinsic).
template <typename A, typename B>
[[nodiscard]] inline auto min_scalar(const A& a, const B& b) noexcept
{
    if constexpr(lps::is_dsl_value_v<A> || lps::is_dsl_value_v<B>)
        return lcrt::min(a, b);
    else
        return a < b ? a : b;
}

/// `max` for a host scalar or a DSL scalar (see `min_scalar`).
template <typename A, typename B>
[[nodiscard]] inline auto max_scalar(const A& a, const B& b) noexcept
{
    if constexpr(lps::is_dsl_value_v<A> || lps::is_dsl_value_v<B>)
        return lcrt::max(a, b);
    else
        return a > b ? a : b;
}

/// `uint32_t` index of a host or DSL value: the DSL needs an explicit `cast`, a host
/// index is a `static_cast`.  (Every index the engine computes is a non-negative
/// integer, so the two agree.)
template <typename I>
[[nodiscard]] inline auto as_u32(const I& i) noexcept
{
    if constexpr(lps::is_dsl_value_v<I>)
        return lcrt::cast<std::uint32_t>(i);
    else
        return static_cast<std::uint32_t>(i);
}

/// `int32_t` index of a host or DSL value (see `as_u32`).
template <typename I>
[[nodiscard]] inline auto as_i32(const I& i) noexcept
{
    if constexpr(lps::is_dsl_value_v<I>)
        return lcrt::cast<std::int32_t>(i);
    else
        return static_cast<std::int32_t>(i);
}

/// The reference's elected-lane test, as a value:
/// `__popc(connect_msk & lanemask_lt(lane_id)) == 0`.
template <typename M, typename L>
[[nodiscard]] inline auto elected_prefix(const M& connect_mask, const L& lane_id) noexcept
{
    return popcount_scalar(connect_mask & lanemask_lt(lane_id));
}

// ---------------------------------------------------------------------------
//  3. the transitive closure (see the file comment)
// ---------------------------------------------------------------------------
/**
 * @brief The reference's per-lane closure over a bank's masks.
 *
 * Reference (`prepare_prefix_sum_L0` / `next_level_cluster`):
 * @code
 *   unsigned int visited = (1U << lane_id);
 *   while(connect_msk != ~0U) {
 *       unsigned int todo = visited ^ connect_msk;
 *       if(!todo) break;
 *       unsigned int next_visit = __ffs(todo) - 1;
 *       visited |= (1U << next_visit);
 *       connect_msk |= cache_mask(next_visit + local_warp_id * BANKSIZE);
 *   }
 * @endcode
 * `own_lane_bit` is `1U << lane_id`, `own_mask` the lane's own connect mask and
 * `read_mask(j)` the mask of lane `j` of the same bank (a `$shared` slot on the
 * device, an array element on the host).
 *
 * The resulting mask is the *least fixpoint* of "merge the masks of the lanes you
 * are connected to", which is order independent, so the fixed 16-iteration loop
 * computes exactly what the reference's data-dependent loop computes (each
 * iteration that finds a new lane adds at least one lane, and there are only
 * `lane_count` lanes).
 */
template <typename Mask, typename ReadMask, typename N>
[[nodiscard]] inline auto transitive_closure_mask(const Mask& own_lane_bit,
                                                  const Mask& own_mask,
                                                  ReadMask&&  read_mask,
                                                  const N&    lane_count) noexcept
{
    auto visited = own_lane_bit;
    auto mask    = own_mask;
    for(int it = 0; it < kBanksize; ++it)
    {
        // the reference's `todo = visited ^ connect_msk` (its `visited` is a subset
        // of its `connect_msk` at every step, so XOR is the set difference)
        auto todo = mask ^ visited;
        // `next_visit = __ffs(todo) - 1`; `j == lane_count` <=> todo == 0 <=> break
        auto j    = ctz_scalar(todo);
        auto more = j < lane_count;
        // both operands of `select` are built before the select, so the index is
        // clamped into range for the discarded lanes (its value is irrelevant
        // there, but `1u << 32` must not be built)
        auto jc = min_scalar(j, lane_count - 1u);
        visited  = visited | scalar_detail::select_scalar(std::uint32_t{0},
                                                         std::uint32_t{1} << jc,
                                                         more);
        mask = mask | scalar_detail::select_scalar(std::uint32_t{0},
                                                  read_mask(jc),
                                                  more);
    }
    return mask;
}

// ---------------------------------------------------------------------------
//  4. flat cluster-matrix access
// ---------------------------------------------------------------------------
/// The port's cluster-matrix buffers are *flat scalar* buffers (the reference
/// stores `ClusterMatrixSymT<Scalar>` = `Eigen::Matrix<Scalar,3,3>
/// M[SYM_BLOCK_COUNT]`), so every block access goes through these two helpers.
using ClusterHessDeviceView =
    lla::LDenseVectorDeviceView<lcrt::Var<Float>>;  // the assembled cluster Hessians (double)
using ClusterInvDeviceView =
    lla::LDenseVectorDeviceView<lcrt::Var<float>>;  // the inverted clusters (float, GIPC parity)

/// Load the 3x3 block that starts at flat index `base` of a double cluster buffer.
template <typename HView, typename I>
[[nodiscard]] inline lps::LMat<lcrt::Var<double>, 3, 3> load_cluster_block(const HView& H,
                                                                          const I& base) noexcept
{
    return lps::make_mat<lcrt::Var<double>, 3, 3>(
        [&](int r, int c) noexcept { return H.read(base + (c * 3 + r)); });
}

/// Store a 3x3 block at flat index `base` of a double cluster buffer.
template <typename HView, typename I, typename M>
inline void store_cluster_block(const HView& H, const I& base, const M& M3) noexcept
{
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            H.write(base + (c * 3 + r), M3(r, c));
}

/// Load the 3x3 block that starts at flat index `base` of a float cluster buffer.
template <typename HView, typename I>
[[nodiscard]] inline lps::LMat<lcrt::Var<float>, 3, 3> load_cluster_block_f(const HView& H,
                                                                            const I& base) noexcept
{
    return lps::make_mat<lcrt::Var<float>, 3, 3>(
        [&](int r, int c) noexcept { return H.read(base + (c * 3 + r)); });
}
}  // namespace mas_preconditioner_detail
}  // namespace uipc::backend::luisa
