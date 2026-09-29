#include <finite_element/mas_preconditioner_engine.h>
#include <finite_element/mas_preconditioner_engine_function.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>
#include <uipc/common/log.h>
#include <uipc/common/json.h>
#include <fmt/format.h>
#include <cstdio>
#include <fstream>
#include <vector>

/*************************************************************************************************
 * T6.4c — MASPreconditionerEngine (port of
 * src/backends/cuda/finite_element/mas_preconditioner_engine.{h,cu}).
 *
 * Read the scope note in mas_preconditioner_engine.h first (the storage deviations,
 * the full-block dispatch rule and the test seams).
 *
 * KERNEL MAP (the reference's named __global__ functions -> this file)
 *   MASPreconditionerEngine_build_connect_mask_L0_kernel          -> build_connect_mask_L0_kernel
 *   MASPreconditionerEngine_prepare_prefix_sum_L0_kernel          -> prepare_prefix_sum_L0_kernel
 *   MASPreconditionerEngine_build_level1_kernel                   -> build_level1_kernel
 *   MASPreconditionerEngine_build_connect_mask_Lx_kernel          -> build_connect_mask_Lx_kernel
 *   MASPreconditionerEngine_next_level_cluster_kernel             -> next_level_cluster_kernel
 *   MASPreconditionerEngine_prefix_sum_Lx_kernel                  -> prefix_sum_Lx_kernel
 *   MASPreconditionerEngine_compute_next_level_kernel             -> compute_next_level_kernel
 *   MASPreconditionerEngine_aggregation_kernel_kernel             -> aggregation_kernel
 *   MASPreconditionerEngine_scatter_hessian_to_clusters_k1_kernel -> scatter_k1_kernel
 *   MASPreconditionerEngine_scatter_hessian_to_clusters_k2_kernel -> scatter_k2_kernel
 *   MASPreconditionerEngine_invert_cluster_matrices_kernel        -> invert_cluster_matrices_kernel
 *   MASPreconditionerEngine_build_multi_level_R_kernel            -> build_multi_level_R_kernel
 *   MASPreconditionerEngine_schwarz_local_solve_kernel            -> schwarz_local_solve_kernel
 *   MASPreconditionerEngine_collect_final_Z_kernel                -> collect_final_Z_kernel
 *
 * The reference's `UIPC_KERNEL_ASSERT`s are the LC_LOG facade in the port (they are
 * compiled out by default, exactly like the reference's non-debug build); where an
 * assert *guarded* an out-of-range read (the `going_next` walks), the port instead
 * stops the walk — that is the only behavioural addition, and it is inert whenever
 * the assert would have passed (cookbook §7.30).
 *
 * Three CUDA constructs have no LC counterpart and are rewritten (RULE M2 — a *device*
 * value may not drive host control flow):
 *   * `if(tid >= n) return;` -> whole-block dispatch + a clamped index + `$if`-gated
 *     stores.  A *return* would leave the warp/shared-memory collectives of these
 *     kernels half-populated (LC compiles a `$if` to a real branch and `sync_block`
 *     to `__syncthreads`, which must be reached by every live thread);
 *   * `while(<device condition>)` -> a fixed-bound `$while`/`$for` with a `live` flag
 *     (the closure loop: at most BANKSIZE iterations; the level walks: at most
 *     MAX_LEVELS);
 *   * `cub::WarpReduce<T, BANKSIZE>` -> `mas_preconditioner_detail::logical_warp16_sum`
 *     (a 16-lane butterfly) and cub's full-warp reduce -> `lcrt::warp_active_sum`.
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using mas_preconditioner_detail::as_i32;
using mas_preconditioner_detail::as_u32;
using mas_preconditioner_detail::bank_align;
using mas_preconditioner_detail::ctz_scalar;
using mas_preconditioner_detail::elected_prefix;
using mas_preconditioner_detail::lanemask_lt;
using mas_preconditioner_detail::level1_begin;
using mas_preconditioner_detail::load_cluster_block;
using mas_preconditioner_detail::load_cluster_block_f;
using mas_preconditioner_detail::max_scalar;
using mas_preconditioner_detail::min_scalar;
using mas_preconditioner_detail::num_levels;
using mas_preconditioner_detail::popcount_scalar;
using mas_preconditioner_detail::sym_index;
using mas_preconditioner_detail::transitive_closure_mask;

constexpr int     kBanksize  = MASPreconditionerEngine::BANKSIZE;
constexpr int     kMaxLevels = MASPreconditionerEngine::MAX_LEVELS;
/// `DEFAULT_WARPNUM`: the logical-warp count of a DEFAULT_BLOCKSIZE block (the
/// reference's `__shared__ ...[DEFAULT_WARPNUM]` arrays).
constexpr uint32_t kWarpNum = MASPreconditionerEngine::DEFAULT_WARPNUM;
constexpr int     kMatDim    = MASPreconditionerEngine::CLUSTER_MAT_DIM;
constexpr int     kEpc       = MASPreconditionerEngine::CLUSTER_ENTRIES;
/// Scalars per BCOO triplet (the T3.1 flat 3x3 block layout).
constexpr int     kBlockEntries = 9;
constexpr int     kVec3      = MASPreconditionerEngine::VEC3_PER_NODE;
constexpr uint32_t kBlock     = MASPreconditionerEngine::DEFAULT_BLOCKSIZE;
constexpr uint32_t kInvertBlk = MASPreconditionerEngine::INVERT_BLOCK_SIZE;
/// `mas_preconditioner_detail::mat_entry`, spelled short for the 3x3 sub-indices
/// (generic so a DSL `row`/`col` index can be passed).
template <typename R, typename C>
[[nodiscard]] inline auto me(const R& r, const C& c) noexcept
{
    return mas_preconditioner_detail::mat_entry(r, c);
}

[[nodiscard]] constexpr uint32_t round_up(uint64_t n, uint32_t block) noexcept
{
    return static_cast<uint32_t>((n + block - 1u) / block * block);
}

/// `cub::WarpReduce<T, BANKSIZE>::Sum`: a butterfly over the *16-lane logical warp*
/// the calling thread belongs to, so all 16 lanes end with the same sum.  (`lane ^ d`
/// with `d < 16` stays inside a 16-lane aligned group, so no group predicate is
/// needed; every lane of the hardware warp must be active, which the callers
/// guarantee by construction — see each call site.)
template <typename T>
[[nodiscard]] inline lcrt::Var<T> logical_warp16_sum(const lcrt::Var<T>& v) noexcept
{
    // `warp_read_lane(r, lane ^ d)` with d < 16 stays inside the caller's 16-lane
    // aligned group, so the butterfly needs no group predicate.
    lcrt::Var<T> r = v;
    for(std::uint32_t d = 8u; d > 0u; d >>= 1u)
    {
        auto other = lcrt::warp_read_lane(r, lcrt::warp_lane_id() ^ d);
        r          = r + other;
    }
    return r;
}

// ============================================================================
//  K1: build_connect_mask_L0_kernel
// ============================================================================
auto& build_connect_mask_L0_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::uint32_t>,  // neighbor_start
                                                lcrt::Buffer<std::uint32_t>,  // neighbor_num
                                                lcrt::Buffer<std::uint32_t>,  // neighbor_list
                                                lcrt::Buffer<std::uint32_t>,  // fine_connect_mask
                                                lcrt::Buffer<std::int32_t>,   // part_to_real
                                                lcrt::Buffer<std::int32_t>,   // real_to_part
                                                std::uint32_t>(  // N
        [](lcrt::BufferVar<std::uint32_t> neighbor_start,
           lcrt::BufferVar<std::uint32_t> neighbor_num,
           lcrt::BufferVar<std::uint32_t> neighbor_list,
           lcrt::BufferVar<std::uint32_t> fine_connect_mask,
           lcrt::BufferVar<std::int32_t>  part_to_real,
           lcrt::BufferVar<std::int32_t>  real_to_part,
           lcrt::Var<std::uint32_t>       n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < n;
            auto tid      = lcrt::min(gid, n - 1u);
            auto bank_id  = tid / kBanksize;
            auto lane_id  = tid % kBanksize;
            auto idx      = part_to_real->read(lcrt::cast<std::int32_t>(tid));
            $if(in_range)
            {
                $if(idx >= 0)
                {
                    auto uidx = lcrt::cast<std::uint32_t>(idx);
                    auto nbr_num = neighbor_num->read(uidx);
                    auto start   = neighbor_start->read(uidx);

                    lcrt::Var<std::uint32_t> connect_msk = 1u << lane_id;
                    lcrt::Var<std::uint32_t> nk          = 0u;
                    lcrt::Var<std::uint32_t> i           = 0u;
                    $while(i < nbr_num)
                    {
                        auto nbr_id   = neighbor_list->read(start + i);
                        auto nbr_part = real_to_part->read(lcrt::cast<std::int32_t>(nbr_id));
                        $if(nbr_part >= 0)
                        {
                            auto nbr_part_u = lcrt::cast<std::uint32_t>(nbr_part);
                            $if(nbr_part_u / kBanksize == bank_id)
                            {
                                connect_msk = connect_msk | (1u << (nbr_part_u % kBanksize));
                            }
                            $else
                            {
                                // compact the cross-bank neighbors to the front of the
                                // list (the reference prunes in place)
                                neighbor_list->write(start + nk, nbr_id);
                                nk = nk + 1u;
                            };
                        };
                        i = i + 1u;
                    };
                    neighbor_num->write(uidx, nk);
                    fine_connect_mask->write(uidx, connect_msk);
                };
            };
        });
    return k;
}

// ============================================================================
//  K2: prepare_prefix_sum_L0_kernel
// ============================================================================
auto& prepare_prefix_sum_L0_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::uint32_t>,  // fine_connect_mask
                                                lcrt::Buffer<std::int32_t>,   // prefix_original
                                                lcrt::Buffer<std::int32_t>,   // part_to_real
                                                std::uint32_t>(  // N
        [](lcrt::BufferVar<std::uint32_t> fine_connect_mask,
           lcrt::BufferVar<std::int32_t>  prefix_orig,
           lcrt::BufferVar<std::int32_t>  part_to_real,
           lcrt::Var<std::uint32_t>       N) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<std::uint32_t> cache_mask{kBlock};     // __shared__ cache_mask_s[DEFAULT_BLOCKSIZE]
            $shared<std::int32_t>  prefix_sum{kWarpNum};  // __shared__ prefix_sum_s[DEFAULT_WARPNUM]

            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < N;
            auto tid      = lcrt::min(gid, N - 1u);
            auto warp_id  = tid / kBanksize;
            auto lwarp    = lcrt::thread_x() / kBanksize;
            auto lane_id  = tid % kBanksize;
            auto idx      = part_to_real->read(lcrt::cast<std::int32_t>(tid));

            $if(in_range)
            {
                $if(idx >= 0)
                {
                    auto uidx = lcrt::cast<std::uint32_t>(idx);
                    auto cm   = fine_connect_mask->read(uidx);
                    $if(lane_id == 0u)
                    {
                        prefix_sum[lwarp] = 0;
                    };
                    cache_mask[lcrt::thread_x()] = cm;

                    // the reference's closure (`visited`/`cache_mask`), fixed bound
                    auto connect_msk = transitive_closure_mask(
                        1u << lane_id,
                        cm,
                        [&](auto j) noexcept { return cache_mask[j + lwarp * kBanksize]; },
                        kBanksize);
                    fine_connect_mask->write(uidx, connect_msk);

                    auto elected = elected_prefix(connect_msk, lane_id);
                    $if(elected == 0u)
                    {
                        (void)prefix_sum.atomic(lwarp).fetch_add(std::int32_t{1});
                    };
                    $if(lane_id == 0u)
                    {
                        prefix_orig->write(warp_id, prefix_sum[lwarp]);
                    };
                };
            };
        });
    return k;
}

// ============================================================================
//  K3: build_level1_kernel
// ============================================================================
auto& build_level1_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,   // level_sizes (2 per level)
                                                lcrt::Buffer<std::int32_t>,   // coarse_space_tables
                                                lcrt::Buffer<std::int32_t>,   // going_next (view [0, level1_begin))
                                                lcrt::Buffer<std::uint32_t>,  // fine_connect_mask
                                                lcrt::Buffer<std::int32_t>,   // prefix_sum_original
                                                lcrt::Buffer<std::int32_t>,   // prefix_original
                                                lcrt::Buffer<std::int32_t>,   // part_to_real
                                                std::uint32_t,                // N
                                                std::int32_t>(  // level1_begin
        [](lcrt::BufferVar<std::int32_t>  level_size,
           lcrt::BufferVar<std::int32_t>  coarse_table,
           lcrt::BufferVar<std::int32_t>  going_next_L0,
           lcrt::BufferVar<std::uint32_t> fine_connect_mask,
           lcrt::BufferVar<std::int32_t>  prefix_sum_orig,
           lcrt::BufferVar<std::int32_t>  prefix_orig,
           lcrt::BufferVar<std::int32_t>  part_to_real,
           lcrt::Var<std::uint32_t>       N,
           lcrt::Var<std::int32_t>        level1_begin) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<std::uint32_t> elected_mask{kWarpNum};
            $shared<std::uint32_t> lane_prefix{kBlock};

            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < N;
            auto tid      = lcrt::min(gid, N - 1u);
            auto warp_id  = tid / kBanksize;
            auto lwarp    = lcrt::thread_x() / kBanksize;
            auto lane_id  = tid % kBanksize;

            $if(lane_id == 0u)
            {
                elected_mask[lwarp] = 0u;
            };
            $if(tid == N - 1u)
            {
                level_size->write(2, prefix_sum_orig->read(warp_id) + prefix_orig->read(warp_id));
                level_size->write(3, level1_begin);
            };

            auto idx = part_to_real->read(lcrt::cast<std::int32_t>(tid));
            $if(in_range)
            {
                $if(idx >= 0)
                {
                    auto uidx = lcrt::cast<std::uint32_t>(idx);
                    auto cm   = fine_connect_mask->read(uidx);
                    auto elected = elected_prefix(cm, lane_id);
                    $if(elected == 0u)
                    {
                        (void)elected_mask.atomic(lwarp).fetch_or(1u << lane_id);
                    };
                    auto lp = popcount_scalar(elected_mask[lwarp] & lanemask_lt(lane_id))
                              + lcrt::cast<std::uint32_t>(prefix_sum_orig->read(warp_id));
                    lane_prefix[lcrt::thread_x()] = lp;
                    // conn_msk always holds the own bit, so __ffs(conn_msk) - 1 is safe
                    auto elected_lane = ctz_scalar(cm);
                    auto the_lane_prefix =
                        lane_prefix[elected_lane + lwarp * kBanksize];
                    coarse_table->write(uidx, lcrt::cast<std::int32_t>(the_lane_prefix));
                    going_next_L0->write(
                        uidx,
                        lcrt::cast<std::int32_t>(the_lane_prefix)
                            + level1_begin);
                };
            };
        });
    return k;
}

// ============================================================================
//  K4: build_connect_mask_Lx_kernel
// ============================================================================
auto& build_connect_mask_Lx_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::uint32_t>,  // neighbor_start
                                                lcrt::Buffer<std::uint32_t>,  // neighbor_num
                                                lcrt::Buffer<std::uint32_t>,  // neighbor_list
                                                lcrt::Buffer<std::int32_t>,   // coarse_space_tables
                                                lcrt::Buffer<std::uint32_t>,  // next_connect_mask
                                                lcrt::Buffer<std::uint32_t>,  // fine_connect_mask
                                                lcrt::Buffer<std::int32_t>,   // part_to_real
                                                std::int32_t,                 // level
                                                std::int32_t,                 // vert_num
                                                std::int32_t>(  // N
        [](lcrt::BufferVar<std::uint32_t> neighbor_start,
           lcrt::BufferVar<std::uint32_t> neighbor_num,
           lcrt::BufferVar<std::uint32_t> neighbor_list,
           lcrt::BufferVar<std::int32_t>  coarse_table,
           lcrt::BufferVar<std::uint32_t> next_connect_mask,
           lcrt::BufferVar<std::uint32_t> fine_connect_mask,
           lcrt::BufferVar<std::int32_t>  part_to_real,
           lcrt::Var<std::int32_t>        level,
           lcrt::Var<std::int32_t>        vert_num,
           lcrt::Var<std::int32_t>        N) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<std::int32_t> cache_msk{kBlock};

            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < N;
            auto tid      = lcrt::min(gid, N - 1u);
            auto lwarp    = lcrt::thread_x() / kBanksize;
            auto lane_id  = tid % kBanksize;

            auto idx = part_to_real->read(lcrt::cast<std::int32_t>(tid));
            $if(in_range)
            {
                $if(idx >= 0)
                {
                    auto uidx       = lcrt::cast<std::uint32_t>(idx);
                    auto prefix_msk = fine_connect_mask->read(uidx);
                    auto coarse_idx = coarse_table->read((level - std::int32_t{1}) * vert_num
                                                         + idx);
                    $if(mas_preconditioner_detail::min_scalar(coarse_idx, std::int32_t{0})
                       == std::int32_t{0})
                    {
                        // the reference's `if(coarse_idx < 0 || coarse_idx >= N) return;`
                        // (its own two early returns, as one guard so both hold)
                    };
                    $if(coarse_idx >= 0)
                    {
                        $if(coarse_idx < lcrt::cast<std::int32_t>(N))
                        {
                            auto kn          = neighbor_num->read(uidx);
                            auto start       = neighbor_start->read(uidx);
                            lcrt::Var<std::uint32_t> conn_msk = 0u;
                            lcrt::Var<std::uint32_t> nk       = 0u;
                            lcrt::Var<std::uint32_t> i        = 0u;
                            $while(i < kn)
                            {
                                auto connect = neighbor_list->read(start + i);
                                auto coarse_connect = coarse_table->read(
                                    (level - std::int32_t{1}) * vert_num
                                    + lcrt::cast<std::int32_t>(connect));
                                $if((coarse_connect < 0)
                                    | (coarse_connect >= lcrt::cast<std::int32_t>(N)))
                                {
                                    neighbor_list->write(start + nk, connect);
                                    nk = nk + 1u;
                                }
                                $else
                                {
                                    $if(coarse_idx / kBanksize == coarse_connect / kBanksize)
                                    {
                                        conn_msk = conn_msk
                                                   | (1u << (lcrt::cast<std::uint32_t>(
                                                              coarse_connect)
                                                             % kBanksize));
                                    }
                                    $else
                                    {
                                        neighbor_list->write(start + nk, connect);
                                        nk = nk + 1u;
                                    };
                                };
                                i = i + 1u;
                            };
                            neighbor_num->write(uidx, nk);
                            cache_msk[lcrt::thread_x()] = 0;

                            $if(popcount_scalar(prefix_msk) == std::uint32_t{kBanksize})
                            {
                                (void)cache_msk.atomic(lwarp * kBanksize)
                                    .fetch_or(lcrt::cast<std::int32_t>(conn_msk));
                                conn_msk = lcrt::cast<std::uint32_t>(
                                    cache_msk[lwarp * kBanksize]);
                            }
                            $else
                            {
                                auto elected_lane = ctz_scalar(prefix_msk);
                                $if(conn_msk != 0u)
                                {
                                    (void)cache_msk.atomic(lwarp * kBanksize + elected_lane)
                                        .fetch_or(lcrt::cast<std::int32_t>(conn_msk));
                                };
                                conn_msk = lcrt::cast<std::uint32_t>(
                                    cache_msk[lwarp * kBanksize + elected_lane]);
                            };
                            auto elected = elected_prefix(prefix_msk, lane_id);
                            $if((conn_msk != 0u) & (elected == 0u))
                            {
                                (void)next_connect_mask
                                    ->atomic(lcrt::cast<std::uint32_t>(coarse_idx))
                                    .fetch_or(conn_msk);
                            };
                        };
                    };
                };
            };
        });
    return k;
}

// ============================================================================
//  K5: next_level_cluster_kernel
// ============================================================================
auto& next_level_cluster_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::uint32_t>,  // next_connect_mask
                                                lcrt::Buffer<std::uint32_t>,  // next_prefixes
                                                std::uint32_t>(  // N
        [](lcrt::BufferVar<std::uint32_t> next_connect_mask,
           lcrt::BufferVar<std::uint32_t> next_prefix,
           lcrt::Var<std::uint32_t>       N) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<std::int32_t>  prefix_sum{kWarpNum};
            $shared<std::uint32_t> cached_msk{kBlock};

            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < N;
            auto idx      = lcrt::min(gid, N - 1u);
            auto lwarp    = lcrt::thread_x() / kBanksize;
            auto lane_id  = idx % kBanksize;

            $if(lane_id == 0u)
            {
                prefix_sum[lwarp] = 0;
            };

            auto cm             = (1u << lane_id) | next_connect_mask->read(idx);
            cached_msk[lcrt::thread_x()] = cm;
            auto conn_msk       = transitive_closure_mask(
                1u << lane_id,
                cm,
                [&](auto j) noexcept { return cached_msk[j + lwarp * kBanksize]; },
                kBanksize);

            $if(in_range)
            {
                next_connect_mask->write(idx, conn_msk);
            };

            auto elected = elected_prefix(conn_msk, lane_id);
            // the reference's `if(idx >= N) return;` keeps the surplus threads of a
            // partial block out of the per-warp elected-lane count; without this gate
            // a clamped lane whose mask is only its own bit would add a phantom
            // elected lane and inflate the next level's node count
            $if((elected == 0u) & in_range)
            {
                (void)prefix_sum.atomic(lwarp).fetch_add(std::int32_t{1});
            };
            $if(lane_id == 0u)
            {
                $if(in_range)
                {
                    next_prefix->write(idx / kBanksize,
                                       lcrt::cast<std::uint32_t>(prefix_sum[lwarp]));
                };
            };
        });
    return k;
}

// ============================================================================
//  K6: prefix_sum_Lx_kernel
// ============================================================================
auto& prefix_sum_Lx_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,   // level_sizes
                                                lcrt::Buffer<std::uint32_t>,  // next_prefixes
                                                lcrt::Buffer<std::uint32_t>,  // next_prefix_sums
                                                lcrt::Buffer<std::uint32_t>,  // next_connect_mask
                                                lcrt::Buffer<std::int32_t>,   // going_next (level window)
                                                std::int32_t,                 // level
                                                std::int32_t,                 // next_level_begin
                                                std::uint32_t>(  // N
        [](lcrt::BufferVar<std::int32_t>  level_size,
           lcrt::BufferVar<std::uint32_t> next_prefix,
           lcrt::BufferVar<std::uint32_t> next_prefix_sum,
           lcrt::BufferVar<std::uint32_t> next_connect_mask,
           lcrt::BufferVar<std::int32_t>  going_next_level,
           lcrt::Var<std::int32_t>        level,
           lcrt::Var<std::int32_t>        next_level_begin,
           lcrt::Var<std::uint32_t>       N) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<std::uint32_t> elected_mask{kWarpNum};
            $shared<std::uint32_t> lane_prefix{kBlock};

            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < N;
            auto idx      = lcrt::min(gid, N - 1u);
            auto warp_id  = idx / kBanksize;
            auto lwarp    = lcrt::thread_x() / kBanksize;
            auto lane_id  = idx % kBanksize;

            $if(lane_id == 0u)
            {
                elected_mask[lwarp] = 0u;
            };
            $if((idx == N - 1u) & in_range)
            {
                level_size->write(2 * (level + std::int32_t{1}),
                                  lcrt::cast<std::int32_t>(next_prefix_sum->read(warp_id)
                                                           + next_prefix->read(warp_id)));
                level_size->write(2 * (level + std::int32_t{1}) + 1, next_level_begin);
            };

            auto cm      = next_connect_mask->read(idx);
            auto elected = elected_prefix(cm, lane_id);
            // see next_level_cluster: the reference's early return keeps the surplus
            // threads out of the elected-lane mask
            $if((elected == 0u) & in_range)
            {
                (void)elected_mask.atomic(lwarp).fetch_or(1u << lane_id);
            };
            auto lp = popcount_scalar(elected_mask[lwarp] & lanemask_lt(lane_id))
                      + next_prefix_sum->read(warp_id);
            lane_prefix[lcrt::thread_x()] = lp;
            auto elected_lane = ctz_scalar(cm);
            auto the_lane_prefix =
                lane_prefix[elected_lane + lwarp * kBanksize];

            $if(in_range)
            {
                next_connect_mask->write(idx, the_lane_prefix);
                going_next_level->write(
                    idx,
                    lcrt::cast<std::int32_t>(the_lane_prefix) + next_level_begin);
            };
        });
    return k;
}

// ============================================================================
//  K7: compute_next_level_kernel
// ============================================================================
auto& compute_next_level_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,   // coarse_space_tables
                                                lcrt::Buffer<std::uint32_t>,  // next_connect_mask
                                                std::int32_t,                 // level
                                                std::uint32_t>(  // N
        [](lcrt::BufferVar<std::int32_t>  coarse_table,
           lcrt::BufferVar<std::uint32_t> next_connect_mask,
           lcrt::Var<std::int32_t>        level,
           lcrt::Var<std::uint32_t>       N) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < N;
            auto tid      = lcrt::cast<std::int32_t>(lcrt::min(gid, N - 1u));
            auto NN       = lcrt::cast<std::int32_t>(N);
            auto next     = coarse_table->read((level - std::int32_t{1}) * NN + tid);
            $if(in_range)
            {
                $if((next < 0) | (next >= NN))
                {
                    coarse_table->write(level * NN + tid, -1);
                }
                $else
                {
                    coarse_table->write(
                        level * NN + tid,
                        lcrt::cast<std::int32_t>(
                            next_connect_mask->read(lcrt::cast<std::uint32_t>(next))));
                };
            };
        });
    return k;
}

// ============================================================================
//  K8: aggregation_kernel (the per-vertex level table)
// ============================================================================
auto& aggregation_kernel_shader()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // coarse_tables (MAX_LEVELS/node)
                                                lcrt::Buffer<std::int32_t>,  // going_next
                                                lcrt::Buffer<std::int32_t>,  // level_sizes
                                                std::int32_t,                // level_num
                                                std::uint32_t>(  // n
        [](lcrt::BufferVar<std::int32_t> coarse_table,
           lcrt::BufferVar<std::int32_t> going_next,
           lcrt::BufferVar<std::int32_t> level_size,
           lcrt::Var<std::int32_t>       level_num,
           lcrt::Var<std::uint32_t>      n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < n;
            auto tid      = lcrt::min(gid, n - 1u);
            auto base     = tid * std::uint32_t{kMaxLevels};

            // the reference default-constructs a LevelTable (its entries past
            // level_num-2 are indeterminate); the port writes a deterministic 0
            for(int l = 0; l < kMaxLevels; ++l)
                coarse_table->write(base + l, std::int32_t{0});

            auto first = going_next->read(tid);
            $if(in_range)
            {
                $if(first >= 0)
                {
                    coarse_table->write(base, first);
                    lcrt::Var<std::int32_t> current = first;
                    lcrt::Var<std::int32_t> l       = 1;
                    lcrt::Var<std::int32_t> live    = 1;
                    $while((l < level_num - std::int32_t{1}) & (live != 0))
                    {
                        // the reference's UIPC_KERNEL_ASSERT(next >= 0): the port
                        // stops the walk instead of reading going_next(-1)
                        auto next = going_next->read(current);
                        $if(next >= 0)
                        {
                            coarse_table->write(base + lcrt::cast<std::uint32_t>(l), next);
                            current = next;
                        }
                        $else
                        {
                            live = 0;
                        };
                        l = l + 1;
                    };
                };
            };
        });
    return k;
}

// ============================================================================
//  K9: scatter_hessian_to_clusters_k1_kernel
// ============================================================================
auto& scatter_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        std::int32_t,                 // offset
        std::int32_t,                 // level_num
        lcrt::Buffer<std::int32_t>,   // going_next
        lcrt::Buffer<std::int32_t>,   // level_sizes
        lcrt::Buffer<Float>,          // cluster_hessians (flat)
        lcrt::Buffer<std::int32_t>,   // real_to_part
        lcrt::Buffer<Float>,          // triplet_values (flat 9 per triplet)
        lcrt::Buffer<std::int32_t>,   // row_ids
        lcrt::Buffer<std::int32_t>,   // col_ids
        std::int32_t,                 // total_nodes
        std::uint32_t>(               // triplet count
        [](lcrt::Var<std::int32_t>      offset,
           lcrt::Var<std::int32_t>      level_num,
           lcrt::BufferVar<std::int32_t> going_next,
           lcrt::BufferVar<std::int32_t> level_size,
           lcrt::BufferVar<Float>       cluster_hess,
           lcrt::BufferVar<std::int32_t> real_to_part,
           lcrt::BufferVar<Float>       triplet_values,
           lcrt::BufferVar<std::int32_t> row_ids,
           lcrt::BufferVar<std::int32_t> col_ids,
           lcrt::Var<std::int32_t>       total_nodes,
           lcrt::Var<std::uint32_t>      n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid      = lcrt::dispatch_x();
            auto in_range = gid < n;
            $if(in_range)
            {
                auto row_real = row_ids->read(lcrt::cast<std::int32_t>(gid)) - offset;
                auto col_real = col_ids->read(lcrt::cast<std::int32_t>(gid)) - offset;
                $if((row_real >= 0) & (row_real < total_nodes))
                {
                    $if((col_real >= 0) & (col_real < total_nodes))
                    {
                        auto vert_col = real_to_part->read(col_real);
                        auto vert_row = real_to_part->read(row_real);
                        $if((vert_col >= 0) & (vert_row >= 0))
                        {
                            // the triplet's 3x3 block (column-major, the BCOO layout)
                            auto H = lps::make_mat<lcrt::Var<Float>, 3, 3>(
                                [&](int r, int c) noexcept
                                {
                                    return triplet_values->read(
                                        std::uint32_t{9} * gid
                                        + mas_preconditioner_detail::as_u32(me(r, c)));
                                });

                            $if(vert_col / kBanksize == vert_row / kBanksize)
                            {
                                auto cid = vert_col / kBanksize;
                                $if(vert_col >= vert_row)
                                {
                                    auto si = sym_index(vert_row % kBanksize,
                                                        vert_col % kBanksize);
                                    for(int i = 0; i < 3; ++i)
                                        for(int j = 0; j < 3; ++j)
                                            (void)cluster_hess
                                                ->atomic(lcrt::cast<std::uint32_t>(cid) * kEpc
                                                         + mas_preconditioner_detail::as_u32(
                                                             si * 9 + me(i, j)))
                                                .fetch_add(H(i, j));
                                }
                                $else
                                {
                                    auto si = sym_index(vert_col % kBanksize,
                                                        vert_row % kBanksize);
                                    for(int i = 0; i < 3; ++i)
                                        for(int j = 0; j < 3; ++j)
                                            (void)cluster_hess
                                                ->atomic(lcrt::cast<std::uint32_t>(cid) * kEpc
                                                         + mas_preconditioner_detail::as_u32(
                                                             si * 9 + me(i, j)))
                                                .fetch_add(H(j, i));
                                };
                            }
                            $else
                            {
                                // walk up all levels (Galerkin H_L = R_L H R_L^T is
                                // applied independently per level)
                                auto ucol = lcrt::cast<std::uint32_t>(col_real);
                                auto urow = lcrt::cast<std::uint32_t>(row_real);
                                lcrt::Var<std::uint32_t> vc   = lcrt::cast<std::uint32_t>(vert_col);
                                lcrt::Var<std::uint32_t> vr   = lcrt::cast<std::uint32_t>(vert_row);
                                lcrt::Var<std::int32_t>  lv   = 0;
                                lcrt::Var<std::int32_t>  live = 1;
                                $while((lv < level_num - std::int32_t{1}) & (live != 0))
                                {
                                    lv = lv + 1;
                                    $if(lv == 1)
                                    {
                                        vc = lcrt::cast<std::uint32_t>(
                                            going_next->read(lcrt::cast<std::int32_t>(ucol)));
                                        vr = lcrt::cast<std::uint32_t>(
                                            going_next->read(lcrt::cast<std::int32_t>(urow)));
                                    }
                                    $else
                                    {
                                        vc = lcrt::cast<std::uint32_t>(
                                            going_next->read(lcrt::cast<std::int32_t>(vc)));
                                        vr = lcrt::cast<std::uint32_t>(
                                            going_next->read(lcrt::cast<std::int32_t>(vr)));
                                    };
                                    $if(vc / kBanksize == vr / kBanksize)
                                    {
                                        auto cid = vc / kBanksize;
                                        $if(vc >= vr)
                                        {
                                            auto si = sym_index(
                                                lcrt::cast<std::int32_t>(vr % kBanksize),
                                                lcrt::cast<std::int32_t>(vc % kBanksize));
                                            for(int i = 0; i < 3; ++i)
                                                for(int j = 0; j < 3; ++j)
                                                {
                                                    (void)cluster_hess
                                                        ->atomic(cid * kEpc
                                                                 + mas_preconditioner_detail::as_u32(
                                                                     si * 9 + me(i, j)))
                                                        .fetch_add(H(i, j));
                                                    // the reference symmetrizes the
                                                    // *diagonal* block at level 1 only
                                                    $if((lv == 1) & (vc == vr))
                                                    {
                                                        (void)cluster_hess
                                                            ->atomic(cid * kEpc
                                                                     + mas_preconditioner_detail::as_u32(
                                                                         si * 9 + me(i, j)))
                                                            .fetch_add(H(j, i));
                                                    };
                                                };
                                        }
                                        $else
                                        {
                                            auto si = sym_index(
                                                lcrt::cast<std::int32_t>(vc % kBanksize),
                                                lcrt::cast<std::int32_t>(vr % kBanksize));
                                            for(int i = 0; i < 3; ++i)
                                                for(int j = 0; j < 3; ++j)
                                                    (void)cluster_hess
                                                        ->atomic(cid * kEpc
                                                                 + mas_preconditioner_detail::as_u32(
                                                                     si * 9 + me(i, j)))
                                                        .fetch_add(H(j, i));
                                        };
                                    };
                                };
                            };
                        };
                    };
                };
            };
        });
    return k;
}

// ============================================================================
//  K10: scatter_hessian_to_clusters_k2_kernel
// ============================================================================
auto& scatter_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<std::int32_t,                 // level_num
                                                lcrt::Buffer<std::int32_t>,   // going_next
                                                lcrt::Buffer<std::int32_t>,   // level_sizes
                                                lcrt::Buffer<Float>,          // cluster_hessians
                                                lcrt::Buffer<std::int32_t>,   // part_to_real
                                                lcrt::Buffer<std::uint32_t>,  // fine_connect_masks
                                                lcrt::Buffer<std::int32_t>,   // prefix_original
                                                std::int32_t,                 // map_nodes
                                                std::uint32_t>(               // total entries
        [](lcrt::Var<std::int32_t>      level_num,
           lcrt::BufferVar<std::int32_t> going_next,
           lcrt::BufferVar<std::int32_t> level_size,
           lcrt::BufferVar<Float>       cluster_hess,
           lcrt::BufferVar<std::int32_t> part_to_real,
           lcrt::BufferVar<std::uint32_t> fine_connect,
           lcrt::BufferVar<std::int32_t> prefix_orig,
           lcrt::Var<std::int32_t>       map_nodes,
           lcrt::Var<std::uint32_t>      n) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<std::int32_t> s_prefix{1u};

            auto gid          = lcrt::dispatch_x();
            auto cluster_stride = std::uint32_t{kBanksize * kBanksize};
            auto cluster_id   = gid / cluster_stride;
            auto local_row    = (gid % cluster_stride) / kBanksize;
            auto local_col    = (gid % cluster_stride) % kBanksize;

            auto global_row = lcrt::cast<std::int32_t>(cluster_id * kBanksize + local_row);
            auto global_col = lcrt::cast<std::int32_t>(cluster_id * kBanksize + local_col);
            // total_entries = map_nodes * BANKSIZE is a whole number of blocks, so the
            // dispatch covers it exactly and the reference's two `return`s cannot fire
            $if((global_row < map_nodes) & (global_col < map_nodes))
            {
                auto rdx = part_to_real->read(global_row);
                auto cdx = part_to_real->read(global_col);

                $if(lcrt::thread_x() == 0u)
                {
                    s_prefix[0] = prefix_orig->read(lcrt::cast<std::int32_t>(cluster_id));
                };
                lcrt::sync_block();
                auto prefix = s_prefix[0];

                auto lr   = lcrt::cast<std::int32_t>(local_row);
                auto lc   = lcrt::cast<std::int32_t>(local_col);
                auto si   = sym_index(min_scalar(lr, lc), max_scalar(lr, lc));
                auto blk  = load_cluster_block(
                    cluster_hess,
                    lcrt::cast<std::uint32_t>(cluster_id) * kEpc
                        + mas_preconditioner_detail::as_u32(si * 9));
                auto invalid = (rdx < 0) | (cdx < 0);

                // `local_col >= local_row` picks the stored upper-triangle block,
                // otherwise its transpose (the reference reads the symmetric array
                // that way); the invalid lanes are zeroed so the warp reduction
                // below can keep every lane active
                lps::LMat<lcrt::Var<Float>, 3, 3> mm;
                for(int i = 0; i < 3; ++i)
                    for(int j = 0; j < 3; ++j)
                        mm(i, j) = scalar_detail::select_scalar(
                            Float{0},
                            scalar_detail::select_scalar(blk(j, i), blk(i, j), local_col >= local_row),
                            !invalid);

                $if(prefix == 1)
                {
                    // cub::WarpReduce<double> over the *full hardware warp*
                    // (`prefix` is block-uniform: a 256-thread block covers exactly
                    // one level-0/level-x cluster)
                    for(int i = 0; i < 3; ++i)
                        for(int j = 0; j < 3; ++j)
                            mm(i, j) = lcrt::warp_active_sum(mm(i, j));

                    // lane 0 of every hardware warp writes the warp's sum up the
                    // levels (the reference's `(threadIdx.x & 0x1f) == 0` test) --
                    // only if its own lane was valid
                    $if(((lcrt::thread_x() & 31u) == 0u) & !invalid)
                    {
                        lcrt::Var<std::int32_t> next_id = going_next->read(rdx);
                        lcrt::Var<std::int32_t> lv      = 0;
                        $while((next_id >= 0) & (lv < level_num - std::int32_t{1}))
                        {
                            lv       = lv + 1;
                            auto bv  = next_id % kBanksize;
                            auto si2 = sym_index(bv, bv);
                            auto cid = lcrt::cast<std::uint32_t>(next_id / kBanksize);
                            for(int i = 0; i < 3; ++i)
                                for(int j = 0; j < 3; ++j)
                                    (void)cluster_hess
                                        ->atomic(cid * kEpc
                                                 + mas_preconditioner_detail::as_u32(si2 * 9
                                                                              + me(i, j)))
                                        .fetch_add(mm(i, j));
                            next_id = going_next->read(next_id);
                        };
                    };
                }
                $else
                {
                    // prefix > 1: each lane walks up on its own, no warp collective
                    $if(!invalid)
                    {
                        lcrt::Var<std::int32_t> lv = 1;
                        $while(lv <= level_num - std::int32_t{1})
                        {
                            rdx = going_next->read(rdx);
                            cdx = going_next->read(cdx);
                            $if((rdx < 0) | (cdx < 0))
                            {
                                lv = level_num;  // the reference's `return`
                            }
                            $else
                            {
                                $if(rdx / kBanksize == cdx / kBanksize)
                                {
                                    $if(cdx >= rdx)
                                    {
                                        auto si2 = sym_index(rdx % kBanksize,
                                                             cdx % kBanksize);
                                        auto cid = lcrt::cast<std::uint32_t>(cdx / kBanksize);
                                        for(int i = 0; i < 3; ++i)
                                            for(int j = 0; j < 3; ++j)
                                                (void)cluster_hess
                                                    ->atomic(cid * kEpc
                                                             + mas_preconditioner_detail::as_u32(
                                                                 si2 * 9 + me(i, j)))
                                                    .fetch_add(mm(i, j));
                                    };
                                };
                            };
                            lv = lv + 1;
                        };
                    };
                };
                lcrt::sync_block();
            };
        });
    return k;
}

// ============================================================================
//  K11: invert_cluster_matrices_kernel
// ============================================================================
auto& invert_cluster_matrices_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<float>,  // cluster_inverses (flat)
                                                lcrt::Buffer<Float>,  // cluster_hessians (flat)
                                                std::uint32_t>(       // total (padded) threads
        [](lcrt::BufferVar<float> cluster_inv,
           lcrt::BufferVar<Float> cluster_hess,
           lcrt::Var<std::uint32_t> total_threads) noexcept
        {
            lcrt::set_block_size(kInvertBlk);
            constexpr std::uint32_t kDim  = static_cast<std::uint32_t>(kMatDim);
            constexpr std::uint32_t kPer  = 32u / static_cast<std::uint32_t>(kBanksize);  // 2
            $shared<double> s_mat{kPer * kDim * kDim};
            $shared<double> s_col{kPer * kDim};
            (void)total_threads;

            auto idx  = lcrt::dispatch_x();
            auto mat  = lcrt::cast<std::int32_t>(idx) / kMatDim;
            auto col  = lcrt::cast<std::int32_t>(idx) % kMatDim;
            auto bmat = lcrt::cast<std::int32_t>(lcrt::thread_x()) / kMatDim;  // 0 or 1
            auto mb   = lcrt::cast<std::uint32_t>(bmat) * kDim * kDim;
            auto cb   = lcrt::cast<std::uint32_t>(bmat) * kDim;

            // ---- gather the symmetric 48x48 (the reference's s_mat fill) ----
            for(int row = 0; row < kMatDim; ++row)
            {
                auto node_row = row / 3;
                auto node_col = col / 3;
                auto ubase    = lcrt::cast<std::uint32_t>(mat) * kEpc;
                // the reference's two cases: upper triangle as-is, else the
                // transposed block from the symmetric array
                auto v = lcrt::select(
                    cluster_hess->read(
                        ubase
                        + mas_preconditioner_detail::as_u32(sym_index(node_col, node_row) * 9
                                                     + me(col % 3, row % 3))),
                    cluster_hess->read(
                        ubase
                        + mas_preconditioner_detail::as_u32(sym_index(node_row, node_col) * 9
                                                     + me(row % 3, col % 3))),
                    node_col >= node_row);
                s_mat[mb + mas_preconditioner_detail::as_u32(row * kMatDim) + lcrt::cast<std::uint32_t>(col)] = v;
                $if((row == col) & (v == 0.0))
                {
                    s_mat[mb + mas_preconditioner_detail::as_u32(row * kMatDim)
                          + lcrt::cast<std::uint32_t>(col)] = 1.0;
                };
            }

            // ---- Gauss-Jordan (column-oriented, the reference's exact order) ----
            $for(pivot, 0, kMatDim)
            {
                lcrt::sync_block();
                auto pivot_val = s_mat[mb
                                       + lcrt::cast<std::uint32_t>(pivot * kMatDim + pivot)];
                s_col[cb + lcrt::cast<std::uint32_t>(col)] =
                    s_mat[mb + lcrt::cast<std::uint32_t>(col * kMatDim + pivot)];
                lcrt::sync_block();
                s_mat[mb + lcrt::cast<std::uint32_t>(col * kMatDim + pivot)] =
                    lcrt::select(0.0, 1.0, col == pivot);
                lcrt::sync_block();
                s_mat[mb + lcrt::cast<std::uint32_t>(pivot * kMatDim) + lcrt::cast<std::uint32_t>(col)] =
                    s_mat[mb + lcrt::cast<std::uint32_t>(pivot * kMatDim)
                          + lcrt::cast<std::uint32_t>(col)]
                    / pivot_val;
                lcrt::sync_block();
                $for(row, 0, kMatDim)
                {
                    $if(row != pivot)
                    {
                        auto factor = -s_col[cb + lcrt::cast<std::uint32_t>(row)];
                        s_mat[mb + lcrt::cast<std::uint32_t>(row * kMatDim)
                              + lcrt::cast<std::uint32_t>(col)] =
                            s_mat[mb + lcrt::cast<std::uint32_t>(row * kMatDim)
                                  + lcrt::cast<std::uint32_t>(col)]
                            + factor
                                  * s_mat[mb + lcrt::cast<std::uint32_t>(pivot * kMatDim)
                                          + lcrt::cast<std::uint32_t>(col)];
                    };
                };
                };
            lcrt::sync_block();

            // ---- the reference's 3x3-block symmetry fix-up ----
            $if(col % 3 < 2)
            {
                s_mat[mb + lcrt::cast<std::uint32_t>((col + 1) * kMatDim)
                      + lcrt::cast<std::uint32_t>(col)] =
                    s_mat[mb + lcrt::cast<std::uint32_t>(col * kMatDim)
                          + lcrt::cast<std::uint32_t>(col + 1)];
            }
            $else
            {
                s_mat[mb + lcrt::cast<std::uint32_t>(col * kMatDim)
                      + lcrt::cast<std::uint32_t>(col - 2)] =
                    s_mat[mb + lcrt::cast<std::uint32_t>((col - 2) * kMatDim)
                          + lcrt::cast<std::uint32_t>(col)];
            };
            lcrt::sync_block();

            for(int row = 0; row < kMatDim; ++row)
            {
                auto node_row = row / 3;
                auto node_col = col / 3;
                $if(node_col >= node_row)
                {
                    auto si = sym_index(node_row, node_col);
                    cluster_inv->write(
                        lcrt::cast<std::uint32_t>(mat) * kEpc
                            + mas_preconditioner_detail::as_u32(si * 9 + me(row % 3, col % 3)),
                        lcrt::cast<float>(
                            s_mat[mb + mas_preconditioner_detail::as_u32(row * kMatDim)
                                  + lcrt::cast<std::uint32_t>(col)]));
                };
            }
        });
    return k;
}

// ============================================================================
//  K12: build_multi_level_R_kernel
// ============================================================================
auto& build_multi_level_R_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<Float>,          // R
                                                lcrt::Buffer<float>,          // multi_level_R
                                                lcrt::Buffer<std::int32_t>,   // going_next
                                                lcrt::Buffer<std::int32_t>,   // level_sizes
                                                lcrt::Buffer<std::int32_t>,   // prefix_original
                                                lcrt::Buffer<std::uint32_t>,  // fine_connect_masks
                                                lcrt::Buffer<std::int32_t>,   // part_to_real
                                                lcrt::Buffer<std::int32_t>,   // converged
                                                std::int32_t,                 // level_num
                                                std::uint32_t>(               // N
        [](lcrt::BufferVar<Float>        R,
           lcrt::BufferVar<float>        multi_lr,
           lcrt::BufferVar<std::int32_t> going_next,
           lcrt::BufferVar<std::int32_t> level_size,
           lcrt::BufferVar<std::int32_t> prefix_orig,
           lcrt::BufferVar<std::uint32_t> fine_conn,
           lcrt::BufferVar<std::int32_t> part_to_real,
           lcrt::BufferVar<std::int32_t> converged,
           lcrt::Var<std::int32_t>       level_num,
           lcrt::Var<std::uint32_t>      N) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<float>        sum_residual{kBlock * kVec3};
            $shared<std::int32_t> prefix_sum_v{kWarpNum};

            $if(converged->read(0) == 0)
            {
                auto gid      = lcrt::dispatch_x();
                auto in_range = gid < N;
                auto tid      = lcrt::min(gid, N - 1u);
                auto idx      = part_to_real->read(lcrt::cast<std::int32_t>(tid));
                auto valid    = in_range & (idx >= 0);
                auto uidx     = lcrt::cast<std::uint32_t>(idx);

                lcrt::Var<float> r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
                $if(valid)
                {
                    r0 = lcrt::cast<float>(R->read(std::uint32_t{kVec3} * uidx));
                    r1 = lcrt::cast<float>(R->read(std::uint32_t{kVec3} * uidx + 1u));
                    r2 = lcrt::cast<float>(R->read(std::uint32_t{kVec3} * uidx + 2u));
                };

                auto lane_id  = tid % kBanksize;
                auto lwarp    = lcrt::thread_x() / kBanksize;
                auto g_warp   = tid / kBanksize;

                $if(in_range)
                {
                    multi_lr->write(std::uint32_t{kVec3} * tid, r0);
                    multi_lr->write(std::uint32_t{kVec3} * tid + 1u, r1);
                    multi_lr->write(std::uint32_t{kVec3} * tid + 2u, r2);
                };
                $if(lane_id == 0u)
                {
                    prefix_sum_v[lwarp] =
                        lcrt::select(std::int32_t{0}, prefix_orig->read(g_warp), in_range);
                };

                auto connect_msk =
                    lcrt::select(std::uint32_t{0}, fine_conn->read(uidx), valid);

                $if(prefix_sum_v[lwarp] == 1)
                {
                    // cub::WarpReduce<float, BANKSIZE> over the 16-lane logical warp;
                    // `prefix_sum_v[lwarp]` is uniform over the logical warp, so all
                    // 32 hardware lanes of the branch are active
                    r0 = logical_warp16_sum(r0);
                    r1 = logical_warp16_sum(r1);
                    r2 = logical_warp16_sum(r2);
                    $if((lane_id == 0u) & valid)
                    {
                        lcrt::Var<std::uint32_t> cur = uidx;
                        $for(l, 0, kMaxLevels)
                        {
                            // device-bounded walk: at most level_num-1 <= MAX_LEVELS-1
                            $if(l < level_num - std::int32_t{1})
                            {
                                cur = lcrt::cast<std::uint32_t>(
                                    going_next->read(lcrt::cast<std::int32_t>(cur)));
                                (void)multi_lr->atomic(std::uint32_t{kVec3} * cur)
                                    .fetch_add(r0);
                                (void)multi_lr->atomic(std::uint32_t{kVec3} * cur + 1u)
                                    .fetch_add(r1);
                                (void)multi_lr->atomic(std::uint32_t{kVec3} * cur + 2u)
                                    .fetch_add(r2);
                            };
                        };
                    };
                }
                $else
                {
                    $if(valid)
                    {
                        auto elected_lane = ctz_scalar(connect_msk);
                        sum_residual[lcrt::thread_x()] = 0.0f;
                        sum_residual[kBlock + lcrt::thread_x()] = 0.0f;
                        sum_residual[2u * kBlock + lcrt::thread_x()] = 0.0f;
                        (void)sum_residual.atomic(lwarp * kBanksize + elected_lane)
                            .fetch_add(r0);
                        (void)sum_residual.atomic(kBlock + lwarp * kBanksize + elected_lane)
                            .fetch_add(r1);
                        (void)sum_residual.atomic(2u * kBlock + lwarp * kBanksize
                                                  + elected_lane)
                            .fetch_add(r2);

                        auto elected = elected_prefix(connect_msk, lane_id);
                        $if(elected == 0u)
                        {
                            lcrt::Var<std::uint32_t> cur = uidx;
                            $for(l, 0, kMaxLevels)
                            {
                                $if(l < level_num - std::int32_t{1})
                                {
                                    cur = lcrt::cast<std::uint32_t>(
                                        going_next->read(lcrt::cast<std::int32_t>(cur)));
                                    (void)multi_lr->atomic(std::uint32_t{kVec3} * cur)
                                        .fetch_add(sum_residual[lcrt::thread_x()]);
                                    (void)multi_lr->atomic(std::uint32_t{kVec3} * cur + 1u)
                                        .fetch_add(sum_residual[kBlock + lcrt::thread_x()]);
                                    (void)multi_lr->atomic(std::uint32_t{kVec3} * cur + 2u)
                                        .fetch_add(sum_residual[2u * kBlock
                                                                + lcrt::thread_x()]);
                                };
                            };
                        };
                    };
                };
            };
        });
    return k;
}

// ============================================================================
//  K13: schwarz_local_solve_kernel
// ============================================================================
auto& schwarz_local_solve_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<float>,         // cluster_inverses
                                                lcrt::Buffer<float>,         // multi_level_R
                                                lcrt::Buffer<float>,         // multi_level_Z
                                                lcrt::Buffer<std::int32_t>,  // converged
                                                std::uint32_t>(              // N
        [](lcrt::BufferVar<float>        cluster_inv,
           lcrt::BufferVar<float>        multi_lr,
           lcrt::BufferVar<float>        multi_lz,
           lcrt::BufferVar<std::int32_t> converged,
           lcrt::Var<std::uint32_t>      N) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<float> s_R{kBanksize * kVec3};

            $if(converged->read(0) == 0)
            {
                auto gid      = lcrt::dispatch_x();
                auto in_range = gid < N;
                auto idx      = lcrt::min(gid, N - 1u);

                auto cluster_id = idx / std::uint32_t{kBanksize * kBanksize};
                auto local_row  = (idx % std::uint32_t{kBanksize * kBanksize}) / kBanksize;
                auto local_col  = (idx % std::uint32_t{kBanksize * kBanksize}) % kBanksize;
                auto vert_row   = cluster_id * kBanksize + local_row;
                auto vert_col   = cluster_id * kBanksize + local_col;

                $if(lcrt::thread_x() < std::uint32_t{kBanksize})
                {
                    s_R[std::uint32_t{kVec3} * lcrt::thread_x()] =
                        multi_lr->read(std::uint32_t{kVec3} * vert_col);
                    s_R[std::uint32_t{kVec3} * lcrt::thread_x() + 1u] =
                        multi_lr->read(std::uint32_t{kVec3} * vert_col + 1u);
                    s_R[std::uint32_t{kVec3} * lcrt::thread_x() + 2u] =
                        multi_lr->read(std::uint32_t{kVec3} * vert_col + 2u);
                };
                lcrt::sync_block();

                auto lr   = lcrt::cast<std::int32_t>(local_row);
                auto lc   = lcrt::cast<std::int32_t>(local_col);
                auto si   = sym_index(min_scalar(lr, lc), max_scalar(lr, lc));
                auto blk  = load_cluster_block_f(
                    cluster_inv,
                    lcrt::cast<std::uint32_t>(cluster_id) * kEpc
                        + mas_preconditioner_detail::as_u32(si * 9));

                // `result = cluster_inv[cluster_id].M[si] * s_R(local_col)`, or the
                // transposed block when the thread sits below the diagonal (the
                // reference's two branches)
                lcrt::Var<float> q[3];
                for(int k = 0; k < 3; ++k)
                {
                    lcrt::Var<float> acc = 0.0f;
                    for(int j = 0; j < 3; ++j)
                    {
                        auto s = s_R[std::uint32_t{kVec3} * local_col
                                     + mas_preconditioner_detail::as_u32(j)];
                        auto a = scalar_detail::select_scalar(blk(j, k), blk(k, j),
                                                              local_col >= local_row);
                        acc    = acc + a * s;
                    }
                    q[k] = acc;
                }

                // cub::WarpReduce<float, BANKSIZE> over the *16-lane logical warp*
                // (every lane of the hardware warp reaches it: the converged guard
                // above is block-uniform and nothing else diverges here)
                for(int k = 0; k < 3; ++k)
                    q[k] = logical_warp16_sum(q[k]);

                // lane 0 of the logical warp adds its warp's total to the row's
                // node (the reference's `threadIdx.x % BANKSIZE == 0`)
                $if((lcrt::thread_x() % std::uint32_t{kBanksize} == 0u) & in_range)
                {
                    (void)multi_lz->atomic(std::uint32_t{kVec3} * vert_row).fetch_add(q[0]);
                    (void)multi_lz->atomic(std::uint32_t{kVec3} * vert_row + 1u)
                        .fetch_add(q[1]);
                    (void)multi_lz->atomic(std::uint32_t{kVec3} * vert_row + 2u)
                        .fetch_add(q[2]);
                };
            };
        });
    return k;
}

// ============================================================================
//  K14: collect_final_Z_kernel
// ============================================================================
auto& collect_final_Z_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<Float>,          // Z
                                                lcrt::Buffer<float>,          // multi_level_Z
                                                lcrt::Buffer<std::int32_t>,   // coarse_tables
                                                lcrt::Buffer<std::int32_t>,   // real_to_part
                                                lcrt::Buffer<std::int32_t>,   // converged
                                                std::int32_t,                 // level_num
                                                std::uint32_t>(               // n
        [](lcrt::BufferVar<Float>        Z,
           lcrt::BufferVar<float>        multi_lz,
           lcrt::BufferVar<std::int32_t> coarse_table,
           lcrt::BufferVar<std::int32_t> real_to_part,
           lcrt::BufferVar<std::int32_t> converged,
           lcrt::Var<std::int32_t>       level_num,
           lcrt::Var<std::uint32_t>      n) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                $if(converged->read(0) == 0)
                {
                    auto rdx = real_to_part->read(lcrt::cast<std::int32_t>(gid));
                    $if(rdx >= 0)
                    {
                        auto urdx = lcrt::cast<std::uint32_t>(rdx);
                        lcrt::Var<float> cz0 = multi_lz->read(std::uint32_t{kVec3} * urdx);
                        lcrt::Var<float> cz1 = multi_lz->read(std::uint32_t{kVec3} * urdx + 1u);
                        lcrt::Var<float> cz2 = multi_lz->read(std::uint32_t{kVec3} * urdx + 2u);
                        lcrt::Var<std::int32_t> l = 1;
                        $while(l < level_num)
                        {
                            auto node = coarse_table->read(
                                gid * std::uint32_t{kMaxLevels}
                                + lcrt::cast<std::uint32_t>(l - std::int32_t{1}));
                            auto unode = lcrt::cast<std::uint32_t>(node);
                            cz0 = cz0 + multi_lz->read(std::uint32_t{kVec3} * unode);
                            cz1 = cz1 + multi_lz->read(std::uint32_t{kVec3} * unode + 1u);
                            cz2 = cz2 + multi_lz->read(std::uint32_t{kVec3} * unode + 2u);
                            l   = l + 1;
                        };
                        Z->write(std::uint32_t{kVec3} * gid, lcrt::cast<Float>(cz0));
                        Z->write(std::uint32_t{kVec3} * gid + 1u, lcrt::cast<Float>(cz1));
                        Z->write(std::uint32_t{kVec3} * gid + 2u, lcrt::cast<Float>(cz2));
                    };
                };
            };
        });
    return k;
}
}  // namespace

// ============================================================================
//  Phase 1: initialization
// ============================================================================
void MASPreconditionerEngine::compute_num_levels(int vert_num)
{
    auto n = mas_preconditioner_detail::num_levels(vert_num);
    m_level_num        = n;
    m_active_level_num = 0;  // 0 = use the full hierarchy (reset on every init_matrix)
}

void MASPreconditionerEngine::init_neighbor(int                      vert_num,
                                            int                      total_neighbor_num,
                                            int                      part_map_size,
                                            span<const unsigned int> h_neighbor_list,
                                            span<const unsigned int> h_neighbor_start,
                                            span<const unsigned int> h_neighbor_num,
                                            span<const int>          h_part_to_real,
                                            span<const int>          h_real_to_part)
{
    if(vert_num < 1)
        return;

    const int max_nodes    = part_map_size > vert_num ? part_map_size : vert_num;
    const int padded_nodes = bank_align(max_nodes);
    compute_num_levels(max_nodes);

    m_total_map_nodes = part_map_size;
    m_total_nodes     = vert_num;

    // Hierarchy buffers (the reference's sizes, `level_sizes` flattened to 2 ints
    // per level so no new LC struct type is needed)
    dense_level.resize(vert_num);
    real_to_part.resize(vert_num);
    coarse_tables.resize(static_cast<size_t>(vert_num) * kMaxLevels);
    coarse_space_tables.resize(static_cast<size_t>(vert_num) * m_level_num);
    level_sizes.resize(static_cast<size_t>(m_level_num + 1) * 2u);
    going_next.resize(static_cast<size_t>(padded_nodes) * m_level_num);
    prefix_original.resize(max_nodes);
    next_prefixes.resize(max_nodes);
    next_prefix_sums.resize(max_nodes);
    prefix_sum_original.resize(max_nodes);
    fine_connect_masks.resize(max_nodes);
    next_connect_masks.resize(max_nodes);

    // Neighbor buffers.  NOTE: an LC buffer view with a *zero* length cannot be a
    // kernel argument at all (its handle is null), so the neighbor list keeps one
    // unused element when the graph has no cross-bank edge — the T6.4b
    // "padded buffer + clamped index" idiom (§7.29 rule 8).  The kernels only reach
    // it through `neighbor_start`/`neighbor_num`, so the padding is inert.
    m_neighbor_list_size = total_neighbor_num;
    const size_t nbr_list_size =
        total_neighbor_num > 0 ? static_cast<size_t>(total_neighbor_num) : 1u;
    neighbor_lists.resize(nbr_list_size);
    neighbor_lists_init.resize(nbr_list_size);
    neighbor_starts.resize(vert_num);
    neighbor_nums.resize(vert_num);
    neighbor_nums_init.resize(vert_num);

    // Partition mappings
    part_to_real.resize(part_map_size);

    // Upload host data (the reference's `view().copy_from(host)`), on the frame
    // stream so a recorded frame's ordering is kept
    if(!h_neighbor_list.empty())
        neighbor_lists_init.view(0u, h_neighbor_list.size())
            .copy_from(span<const unsigned int>(h_neighbor_list.data(), h_neighbor_list.size()));
    neighbor_starts.view().copy_from(
        span<const unsigned int>(h_neighbor_start.data(), h_neighbor_start.size()));
    neighbor_nums_init.view().copy_from(
        span<const unsigned int>(h_neighbor_num.data(), h_neighbor_num.size()));
    part_to_real.view().copy_from(span<const int>(h_part_to_real.data(), h_part_to_real.size()));
    real_to_part.view().copy_from(span<const int>(h_real_to_part.data(), h_real_to_part.size()));
}

void MASPreconditionerEngine::init_matrix()
{
    if(m_total_nodes < 1)
        return;

    // Restore the neighbor data for the initial hierarchy build
    neighbor_lists.view().copy_from(neighbor_lists_init.cview());
    neighbor_nums.view().copy_from(neighbor_nums_init.cview());

    const int total_cluster = static_cast<int>(reorder_realtime() * 1.05);
    const int num_blocks    = total_cluster / BANKSIZE;

    // DEVIATION (documented): the port pads the cluster-matrix buffers so the
    // 96-thread Gauss-Jordan dispatch is always a whole number of blocks.  The
    // reference lets the surplus threads `return` before its `__syncthreads()`,
    // which LC cannot express (a `$if`-guarded barrier is divergent); the padding
    // lanes instead invert a *zero* cluster (whose diagonal the kernel pins to 1,
    // i.e. an exact identity inverse) and their result is never read.
    // `total_threads = m_total_num_clusters * CLUSTER_MAT_DIM / BANKSIZE` (the
    // reference's `* 3`, i.e. 48 threads per 16-node bank) rounded up to whole
    // INVERT_BLOCK_SIZE blocks
    const int total_threads = m_total_num_clusters * CLUSTER_MAT_DIM / BANKSIZE;
    const int padded_threads =
        static_cast<int>(round_up(static_cast<uint64_t>(total_threads), kInvertBlk));
    const int alloc_blocks = padded_threads / CLUSTER_MAT_DIM > num_blocks
                                 ? padded_threads / CLUSTER_MAT_DIM
                                 : num_blocks;

    cluster_hessians.resize(static_cast<size_t>(alloc_blocks) * CLUSTER_ENTRIES);
    cluster_inverses.resize(static_cast<size_t>(alloc_blocks) * CLUSTER_ENTRIES);
    multi_level_R.resize(static_cast<size_t>(total_cluster) * VEC3_PER_NODE);
    multi_level_Z.resize(static_cast<size_t>(total_cluster) * VEC3_PER_NODE);

    m_initialized = true;
}

// ============================================================================
//  Hierarchy building
// ============================================================================
MASPreconditionerEngine::Int2 MASPreconditionerEngine::level_size(int level) const
{
    std::array<std::int32_t, 2> v{};
    level_sizes.cview(static_cast<size_t>(level) * 2u, 2u).copy_to(span<std::int32_t>{v});
    lud::main_stream().synchronize();
    return Int2{static_cast<int>(v[0]), static_cast<int>(v[1])};
}

int MASPreconditionerEngine::reorder_realtime()
{
    level_sizes.view().fill(std::int32_t{0}, &frame_stream());
    coarse_space_tables.view().fill(std::int32_t{-1}, &frame_stream());
    going_next.view().fill(std::int32_t{-1}, &frame_stream());

    build_connect_mask_L0();
    prepare_prefix_sum_L0();
    build_level1();

    for(int level = 1; level < m_level_num; level++)
    {
        next_connect_masks.view().fill(std::uint32_t{0}, &frame_stream());
        build_connect_mask_Lx(level);

        m_h_level_size = level_size(level);

        next_level_cluster(level);
        prefix_sum_Lx(level);
        compute_next_level(level);
    }

    m_h_level_size = level_size(m_level_num);

    m_total_num_clusters = m_h_level_size.y;
    aggregation_kernel();

    return m_total_num_clusters;
}

// ---------------------------------------------------------------------------
// Build the connectivity mask at level 0
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::build_connect_mask_L0()
{
    const int N = m_total_map_nodes;
    if(N < 1)
        return;

    auto& k = build_connect_mask_L0_kernel();
    frame_stream().record(
        k(neighbor_starts.cview(),
          neighbor_nums.view(),
          neighbor_lists.view(),
          fine_connect_masks.view(),
          part_to_real.cview(),
          real_to_part.cview(),
          mas_preconditioner_detail::as_u32(N))
            .dispatch(round_up(N, kBlock)));
}

// ---------------------------------------------------------------------------
// Prepare the level-0 prefix sums (per-bank elected-lane counts)
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::prepare_prefix_sum_L0()
{
    const int N = m_total_map_nodes;
    if(N < 1)
        return;

    auto& k = prepare_prefix_sum_L0_kernel();
    frame_stream().record(k(fine_connect_masks.view(),
                            prefix_original.view(),
                            part_to_real.cview(),
                            mas_preconditioner_detail::as_u32(N))
                              .dispatch(round_up(N, kBlock)));
}

// ---------------------------------------------------------------------------
// Build the coarse level 1 from the prefix sums
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::build_level1()
{
    const int N = m_total_map_nodes;
    if(N < 1)
        return;

    const int warp_num = (N + BANKSIZE - 1) / BANKSIZE;

    lla::exclusive_scan(frame_stream(),
                        prefix_original.cview(0u, static_cast<size_t>(warp_num)),
                        prefix_sum_original.view(0u, static_cast<size_t>(warp_num)),
                        static_cast<size_t>(warp_num));

    const int l1_begin = level1_begin(N, m_total_nodes);

    auto& k = build_level1_kernel();
    frame_stream().record(k(level_sizes.view(),
                            coarse_space_tables.view(),
                            going_next.view(0u, static_cast<size_t>(l1_begin)),
                            fine_connect_masks.cview(),
                            prefix_sum_original.cview(),
                            prefix_original.cview(),
                            part_to_real.cview(),
                            mas_preconditioner_detail::as_u32(N),
                            l1_begin)
                              .dispatch(round_up(N, kBlock)));
}

// ---------------------------------------------------------------------------
// Build the connectivity mask at level x (coarsened)
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::build_connect_mask_Lx(int level)
{
    const int N = m_total_map_nodes;
    if(N < 1)
        return;

    auto& k = build_connect_mask_Lx_kernel();
    frame_stream().record(k(neighbor_starts.cview(),
                            neighbor_nums.view(),
                            neighbor_lists.view(),
                            coarse_space_tables.cview(),
                            next_connect_masks.view(),
                            fine_connect_masks.cview(),
                            part_to_real.cview(),
                            level,
                            m_total_nodes,
                            N)
                              .dispatch(round_up(N, kBlock)));
}

// ---------------------------------------------------------------------------
// Cluster connectivity at the next level
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::next_level_cluster(int level)
{
    (void)level;
    const int N = m_h_level_size.x;
    if(N < 1)
        return;

    auto& k = next_level_cluster_kernel();
    frame_stream().record(k(next_connect_masks.view(),
                            next_prefixes.view(),
                            mas_preconditioner_detail::as_u32(N))
                              .dispatch(round_up(N, kBlock)));
}

// ---------------------------------------------------------------------------
// Prefix sums at level x
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::prefix_sum_Lx(int level)
{
    const int N = m_h_level_size.x;
    if(N < 1)
        return;

    const int level_begin       = m_h_level_size.y;
    const int level_region_size = bank_align(N);
    const int next_level_begin  = level_begin + level_region_size;
    const int warp_num          = (N + BANKSIZE - 1) / BANKSIZE;

    lla::exclusive_scan(frame_stream(),
                        next_prefixes.cview(0u, static_cast<size_t>(warp_num)),
                        next_prefix_sums.view(0u, static_cast<size_t>(warp_num)),
                        static_cast<size_t>(warp_num));

    auto& k = prefix_sum_Lx_kernel();
    frame_stream().record(k(level_sizes.view(),
                            next_prefixes.cview(),
                            next_prefix_sums.cview(),
                            next_connect_masks.view(),
                            going_next.view(static_cast<size_t>(level_begin),
                                            static_cast<size_t>(level_region_size)),
                            level,
                            next_level_begin,
                            mas_preconditioner_detail::as_u32(N))
                              .dispatch(round_up(N, kBlock)));
}

// ---------------------------------------------------------------------------
// Compute the next coarsening level
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::compute_next_level(int level)
{
    const int N = m_total_nodes;
    if(N < 1)
        return;

    auto& k = compute_next_level_kernel();
    frame_stream().record(k(coarse_space_tables.view(),
                            next_connect_masks.cview(),
                            level,
                            mas_preconditioner_detail::as_u32(N))
                              .dispatch(round_up(N, kBlock)));
}

// ---------------------------------------------------------------------------
// Build the per-node level traversal table
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::aggregation_kernel()
{
    const int N = m_total_nodes;
    if(N < 1 || m_total_num_clusters < 1)
        return;

    auto& k = aggregation_kernel_shader();
    frame_stream().record(k(coarse_tables.view(),
                            going_next.cview(0u, static_cast<size_t>(m_total_num_clusters)),
                            level_sizes.cview(),
                            m_level_num,
                            mas_preconditioner_detail::as_u32(N))
                              .dispatch(round_up(N, kBlock)));
}

// ============================================================================
//  Phase 2: assemble the preconditioner
// ============================================================================
void MASPreconditionerEngine::set_preconditioner(lud::LCBufferView<Float>        triplet_values,
                                                lud::LCBufferView<std::int32_t> row_ids,
                                                lud::LCBufferView<std::int32_t> col_ids,
                                                int                             dof_offset)
{
    if(m_total_nodes < 1)
        return;

    // The partition adjacency and its multi-level hierarchy are invariant for the
    // lifetime of this engine: init_matrix() builds them once (the reference's note).
    const int num_cluster_blocks = m_total_num_clusters / BANKSIZE;
    if(num_cluster_blocks < 1)
        return;

    const int alloc_blocks = static_cast<int>(cluster_hessians.size() / CLUSTER_ENTRIES);
    if(alloc_blocks < num_cluster_blocks)
    {
        // grow to the padded size (see init_matrix)
        const int total_threads = m_total_num_clusters * CLUSTER_MAT_DIM / BANKSIZE;
        const int padded        = static_cast<int>(
                              round_up(static_cast<uint64_t>(total_threads), kInvertBlk))
                             / CLUSTER_MAT_DIM;
        cluster_hessians.resize(static_cast<size_t>(padded) * CLUSTER_ENTRIES);
        cluster_inverses.resize(static_cast<size_t>(padded) * CLUSTER_ENTRIES);
    }

    if(m_total_num_clusters > static_cast<int>(multi_level_R.size() / VEC3_PER_NODE))
    {
        multi_level_R.resize(static_cast<size_t>(m_total_num_clusters) * VEC3_PER_NODE);
        multi_level_Z.resize(static_cast<size_t>(m_total_num_clusters) * VEC3_PER_NODE);
    }

    // Zero *all* the cluster matrices (the reference zeroes the first
    // num_cluster_blocks; the port's padding lanes read the tail, see init_matrix)
    cluster_hessians.view().fill(Float{0.0}, &frame_stream());

    scatter_hessian_to_clusters(triplet_values, row_ids, col_ids, dof_offset);
    invert_cluster_matrices();
}

// ---------------------------------------------------------------------------
// Scatter the BCOO Hessian blocks into the cluster-level dense matrices
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::scatter_hessian_to_clusters(
    lud::LCBufferView<Float>        triplet_values,
    lud::LCBufferView<std::int32_t> row_ids,
    lud::LCBufferView<std::int32_t> col_ids,
    int                             dof_offset)
{
    // NOTE (deviation): the reference's `A.values()` is a *block-typed* view
    // (`CBufferView<Eigen::Matrix3d>`, one element per triplet), while the port's BCOO
    // container exposes a **flat scalar** view (`9` doubles per triplet, the T3.1
    // layout), so the triplet count is the element count divided by the block size.
    const int triplet_num =
        static_cast<int>(triplet_values.size() / static_cast<size_t>(kBlockEntries));

    // --- Pass 1: the finest level where both endpoints land in one cluster ---
    if(triplet_num > 0)
    {
        auto& k = scatter_k1_kernel();
        frame_stream().record(k(dof_offset,
                                m_level_num,
                                going_next.cview(0u, static_cast<size_t>(m_total_num_clusters)),
                                level_sizes.cview(),
                                cluster_hessians.view(),
                                real_to_part.cview(),
                                triplet_values,
                                row_ids,
                                col_ids,
                                m_total_nodes,
                                mas_preconditioner_detail::as_u32(triplet_num))
                                  .dispatch(round_up(triplet_num, kBlock)));
    }

    // --- Pass 2: the fine-level cluster matrices to the coarser levels ---
    const int total_entries = m_total_map_nodes * BANKSIZE;
    if(total_entries < 1)
        return;

    auto& k = scatter_k2_kernel();
    frame_stream().record(k(m_level_num,
                            going_next.cview(0u, static_cast<size_t>(m_total_num_clusters)),
                            level_sizes.cview(),
                            cluster_hessians.view(),
                            part_to_real.cview(),
                            fine_connect_masks.cview(),
                            prefix_original.cview(),
                            m_total_map_nodes,
                            mas_preconditioner_detail::as_u32(total_entries))
                              .dispatch(round_up(total_entries, kBlock)));
}

// ---------------------------------------------------------------------------
// Gauss-Jordan inversion of every 48x48 cluster matrix
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::invert_cluster_matrices()
{
    const int total_threads = m_total_num_clusters * CLUSTER_MAT_DIM / BANKSIZE;
    if(total_threads < 1)
        return;

    // whole blocks only (the padded lanes invert a never-read zero cluster)
    const int padded_threads = static_cast<int>(
        round_up(static_cast<uint64_t>(total_threads), kInvertBlk));
    const size_t padded_blocks = static_cast<size_t>(padded_threads) / CLUSTER_MAT_DIM;
    if(padded_blocks * CLUSTER_ENTRIES > cluster_inverses.size())
    {
        // the buffers are sized for exactly this dispatch in init_matrix(); growing
        // here would mean a caller built the hierarchy behind init_matrix's back
        cluster_hessians.resize(padded_blocks * CLUSTER_ENTRIES);
        cluster_inverses.resize(padded_blocks * CLUSTER_ENTRIES);
    }

    auto& k = invert_cluster_matrices_kernel();
    frame_stream().record(k(cluster_inverses.view(),
                            cluster_hessians.cview(),
                            mas_preconditioner_detail::as_u32(padded_threads))
                              .dispatch(static_cast<uint32_t>(padded_threads)));
}

// ============================================================================
//  Phase 3: apply the preconditioner  z = M^{-1} r
// ============================================================================
void MASPreconditionerEngine::build_multi_level_R(lla::LCDenseVectorView<Float> R,
                                                  lud::LCBufferView<IndexT>     converged,
                                                  lud::LStream*                 stream)
{
    (void)stream;  // the reference's CUDA stream: the port uses the frame stream
    const int N = m_total_map_nodes;
    if(N < 1)
        return;

    auto& k = build_multi_level_R_kernel();
    frame_stream().record(k(R.buffer_view(),
                            multi_level_R.view(),
                            going_next.cview(0u, static_cast<size_t>(m_total_num_clusters)),
                            level_sizes.cview(),
                            prefix_original.cview(),
                            fine_connect_masks.cview(),
                            part_to_real.cview(),
                            converged,
                            m_level_num,
                            mas_preconditioner_detail::as_u32(N))
                              .dispatch(round_up(N, kBlock)));
}

void MASPreconditionerEngine::schwarz_local_solve(lud::LCBufferView<IndexT> converged,
                                                  lud::LStream*             stream)
{
    (void)stream;
    const int N = m_total_num_clusters * BANKSIZE;  // one thread per (cluster, node pair)
    if(N < 1)
        return;

    auto& k = schwarz_local_solve_kernel();
    frame_stream().record(k(cluster_inverses.cview(),
                            multi_level_R.cview(),
                            multi_level_Z.view(),
                            converged,
                            mas_preconditioner_detail::as_u32(N))
                              .dispatch(round_up(N, kBlock)));
}

void MASPreconditionerEngine::collect_final_Z(lla::LDenseVectorView<Float> Z,
                                              lud::LCBufferView<IndexT>     converged,
                                              lud::LStream*                 stream)
{
    (void)stream;
    const int N = m_total_nodes;
    if(N < 1)
        return;

    const int level_num = (m_active_level_num > 0) ? m_active_level_num : m_level_num;
    auto&     k         = collect_final_Z_kernel();
    frame_stream().record(k(Z.buffer_view(),
                            multi_level_Z.cview(),
                            coarse_tables.cview(),
                            real_to_part.cview(),
                            converged,
                            level_num,
                            mas_preconditioner_detail::as_u32(N))
                              .dispatch(round_up(N, kBlock)));
}

void MASPreconditionerEngine::apply(lla::LCDenseVectorView<Float> r,
                                    lla::LDenseVectorView<Float>  z,
                                    lud::LCBufferView<IndexT>     converged,
                                    lud::LStream*                 stream)
{
    if(m_total_nodes < 1)
        return;

    if(m_total_num_clusters > static_cast<int>(multi_level_R.size() / VEC3_PER_NODE))
    {
        multi_level_R.resize(static_cast<size_t>(m_total_num_clusters) * VEC3_PER_NODE);
        multi_level_Z.resize(static_cast<size_t>(m_total_num_clusters) * VEC3_PER_NODE);
    }

    if(m_total_num_clusters > m_total_map_nodes)
    {
        multi_level_R
            .view(static_cast<size_t>(m_total_map_nodes),
                  static_cast<size_t>(m_total_num_clusters - m_total_map_nodes) * VEC3_PER_NODE)
            .fill(0.0f, &frame_stream());
    }

    multi_level_Z.view(0u,
                       static_cast<size_t>(m_total_num_clusters) * VEC3_PER_NODE)
        .fill(0.0f, &frame_stream());

    // 1. restrict, 2. local solve, 3. prolongate
    build_multi_level_R(r, converged, stream);
    schwarz_local_solve(converged, stream);
    collect_final_Z(z, converged, stream);
}

// ============================================================================
//  The debug dump (the reference's `dump_cluster_matrices_debug`)
// ============================================================================
void MASPreconditionerEngine::dump_cluster_matrices_debug(std::string_view output_dir,
                                                          SizeT            frame,
                                                          SizeT            newton_iter)
{
    if(!m_initialized || cluster_hessians.size() == 0)
        return;

    frame_stream().synchronize();

    const std::filesystem::path output_dir_path{output_dir};

    const size_t nb = cluster_hessians.size() / CLUSTER_ENTRIES;

    std::vector<Float> h_hess(static_cast<size_t>(nb) * CLUSTER_ENTRIES);
    std::vector<float> h_inv(static_cast<size_t>(nb) * CLUSTER_ENTRIES);
    if(!h_hess.empty())
    {
        cluster_hessians.view().copy_to(span<Float>{h_hess});
        cluster_inverses.view().copy_to(span<float>{h_inv});
        lud::main_stream().synchronize();
    }

    const int     mat_size  = static_cast<int>(nb) * CLUSTER_MAT_DIM;
    const int64_t total_nnz = static_cast<int64_t>(nb) * SYM_BLOCK_COUNT * 9;

    auto write_mtx = [&]<typename Scalar>(const std::vector<Scalar>& flat, std::string_view kind)
    {
        auto path     = output_dir_path
                    / fmt::format("mas_cluster_{}.f{}.n{}.mtx", kind, frame, newton_iter);
        auto path_str = path.string();

        auto buf = fmt::memory_buffer();
        fmt::format_to(std::back_inserter(buf),
                       "%%MatrixMarket matrix coordinate real general\n"
                       "% MAS cluster {} block-upper-triangle ({} clusters, banksize={})\n"
                       "{} {} {}\n",
                       kind,
                       nb,
                       BANKSIZE,
                       mat_size,
                       mat_size,
                       total_nnz);

        for(size_t c = 0; c < nb; c++)
        {
            const int base = static_cast<int>(c) * CLUSTER_MAT_DIM;
            for(int br = 0; br < BANKSIZE; br++)
            {
                for(int bc = br; bc < BANKSIZE; bc++)
                {
                    const int k = mas_preconditioner_detail::cluster_block_base(br, bc);
                    for(int i = 0; i < 3; i++)
                        for(int j = 0; j < 3; j++)
                            fmt::format_to(std::back_inserter(buf),
                                           "{} {} {:.17g}\n",
                                           base + br * 3 + i + 1,
                                           base + bc * 3 + j + 1,
                                           static_cast<double>(
                                               flat[static_cast<size_t>(c) * CLUSTER_ENTRIES
                                                    + static_cast<size_t>(k + me(i, j))]));
                }
            }
        }

        FILE* fp = std::fopen(path_str.c_str(), "w");
        if(!fp)
        {
            UIPC_WARN_WITH_LOCATION("MAS dump: open {} failed", path_str);
            return;
        }
        std::fwrite(buf.data(), 1, buf.size(), fp);
        std::fclose(fp);
        logger::info("MAS dump: wrote {}", path_str);
    };

    write_mtx(h_hess, "hess");
    write_mtx(h_inv, "inv");

    {
        auto path = output_dir_path
                    / fmt::format("mas_cluster_meta.f{}.n{}.json", frame, newton_iter);
        std::ofstream out(path);
        if(!out)
        {
            UIPC_WARN_WITH_LOCATION("MAS dump: open {} failed", path.string());
            return;
        }

        Json j;
        j["total_nodes"]        = m_total_nodes;
        j["total_map_nodes"]    = m_total_map_nodes;
        j["total_clusters"]     = m_total_num_clusters;
        j["num_cluster_blocks"] = nb;
        j["banksize"]           = BANKSIZE;
        j["sym_block_count"]    = SYM_BLOCK_COUNT;
        j["frame"]              = frame;
        j["newton_iter"]        = newton_iter;

        if(m_total_map_nodes > 0)
        {
            std::vector<std::int32_t> h_part_to_real(static_cast<size_t>(m_total_map_nodes));
            part_to_real.cview(0u, static_cast<size_t>(m_total_map_nodes))
                .copy_to(span<std::int32_t>{h_part_to_real});
            lud::main_stream().synchronize();
            j["part_to_real"] = h_part_to_real;
        }

        {
            Json jl = Json::array();
            for(int l = 0; l <= m_level_num; l++)
            {
                const auto sz = level_size(l);
                Json       e;
                e["level"]  = l;
                e["count"]  = sz.x;
                e["offset"] = sz.y;
                jl.push_back(e);
            }
            j["levels"]    = jl;
            j["level_num"] = m_level_num;
        }

        if(m_total_num_clusters > 0)
        {
            std::vector<std::int32_t> h_gn(static_cast<size_t>(m_total_num_clusters));
            going_next.cview(0u, static_cast<size_t>(m_total_num_clusters))
                .copy_to(span<std::int32_t>{h_gn});
            lud::main_stream().synchronize();
            j["going_next"] = h_gn;
        }

        if(m_total_nodes > 0)
        {
            std::vector<std::int32_t> h_r2p(static_cast<size_t>(m_total_nodes));
            real_to_part.cview(0u, static_cast<size_t>(m_total_nodes))
                .copy_to(span<std::int32_t>{h_r2p});
            lud::main_stream().synchronize();
            j["real_to_part"] = h_r2p;
        }

        out << j.dump(4) << '\n';
        logger::info("MAS dump: wrote {}", path.string());
    }
}
}  // namespace uipc::backend::luisa
