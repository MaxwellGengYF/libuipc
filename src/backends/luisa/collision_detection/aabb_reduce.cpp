#include <collision_detection/details/aabb_reduce.h>

#include <frame_stream.h>
#include <luisa_device/kernel.h>
#include <luisa_device/linear/algorithms.h>

#include <cstddef>
#include <cstdint>
#include <limits>

/*************************************************************************************************
 * T7.1a — the AABB union reduction (see details/aabb_reduce.h for why it is not a
 * `linalg` primitive).
 *
 * The reference's two shapes that this file unifies:
 *   - `LinearBVH::build` step 1: CUB `DeviceReduce(stream).Reduce(aabbs, out, n,
 *     merge, default_aabb)`;
 *   - `StacklessBVH::Impl::calcMaxBVFromBox`: `StacklessBVH_calcMaxBVFromBox_kernel`,
 *     a K_THREADS = 256 block that reduces its slice with `__shfl_down_sync` trees,
 *     stages the per-warp results in `__shared__ PlainAABB aabbData[K_WARPS]`,
 *     reduces those, and lets thread 0 `atomicMinf`/`atomicMaxf` the 6 scalars into
 *     the global scene box (grid-strided: the last block covers the remainder).
 *
 * Both are reproduced here as a *two-pass, atomic-free* reduction over the whole
 * buffer — the same shape as `linalg::device_reduce`:
 *   pass 1: `partials[b] = union(in[b*256 .. min(n,(b+1)*256)))`
 *   pass 2: one block folds all `ceil(n/256)` partials (a strided per-thread loop)
 * and writes `out[0]`.
 *
 * TRANSCRIBED CUDA PRIMITIVES
 *   __shfl_down_sync(0xffffffff, v, d)  -> lcrt::warp_read_lane(v, min(lane+d, W-1))
 *        (the reference's tree never uses the value a lane with lane+d >= W would
 *        read — CUDA leaves it undefined, `__shfl_down_sync` does not wrap — so the
 *        port clamps the *source* lane to the widest live lane; lane 0's final value
 *        is provably the fold of all live lanes, see the note in `warp_reduce_box`).
 *   __mm_min / __mm_max (a>b?b:a)       -> lps::min / lps::max (fminf/fmaxf on the
 *        device, exactly what the reference's helpers compile to for the ±0/NaN cases
 *        that matter here).
 *   __shared__ PlainAABB aabbData[K_WARPS] -> $shared<float> (two 3*kWarps arrays; a
 *        `Shared<LAABB>` would work too — the reference keeps the 6 scalars in a plain
 *        struct, and two flat arrays are exactly that layout).
 *   atomicMinf / atomicMaxf             -> dropped: min/max over floats is exact and
 *        order-independent, so the two-pass fold produces the identical scene box
 *        without an atomic (see aabb_reduce.h).
 *   if(idx >= size) return;             -> the port dispatches whole blocks, clamps the
 *        index and lets the surplus lanes contribute the *identity* box (the empty box
 *        is the merge identity), because the block reduction uses warp/block
 *        collectives and every live lane must reach them (cookbook §7.30 rule 16).
 *************************************************************************************************/
namespace uipc::backend::luisa::detail
{
namespace
{
namespace lla = luisa_device::linalg;

constexpr std::uint32_t kBlock = kAabbReduceBlock;
constexpr std::uint32_t kWarps = kBlock / 32u;  // 8

/// The empty box as a device literal (`AlignedBox::setEmpty()`, see aabb.h).
[[nodiscard]] inline LBox<lcrt::Var<float>> device_empty_box() noexcept
{
    return box_empty<lcrt::Var<float>>();
}

[[nodiscard]] inline LBox<lcrt::Var<float>> load_box_at(
    const lcrt::BufferVar<LAABB>& aabbs, const lcrt::Var<std::uint32_t>& i) noexcept
{
    return load_aabb(aabbs->read(i));
}

/// The reference's `__shfl_down_sync` reduction tree over one box, for a group of
/// `width` lanes whose first lane is `lane & ~(width-1)`.
///
/// `lcrt::warp_read_lane(v, src)` gathers lane `src` of the *warp*; the reference's
/// `__shfl_down_sync(mask, v, d)` reads `lane+d` and leaves it undefined for
/// `lane+d >= width`, a value its tree never uses for lane 0.  Clamping the source to
/// `lane & ~(width-1) + width-1` keeps every read inside the live group *and* leaves
/// lane 0's result the fold of the whole group: at step `d`, lane `j` has already
/// folded `j .. j+d-1` **within the group** for every `j+d <= width-1`, and the
/// clamped lanes only ever fold in lanes of the same group.
template <std::uint32_t Width>
[[nodiscard]] inline LBox<lcrt::Var<float>> group_reduce_box(const LBox<lcrt::Var<float>>& in,
                                                            std::uint32_t base) noexcept
{
    LBox<lcrt::Var<float>> r = in;
    for(std::uint32_t d = 1u; d < Width; d <<= 1u)
    {
        auto src = lcrt::min(lcrt::warp_lane_id() + d, base + (Width - 1u));
        for(int k = 0; k < 3; ++k)
        {
            auto other_min = lcrt::warp_read_lane(r.min(k), src);
            auto other_max = lcrt::warp_read_lane(r.max(k), src);
            r.min(k)       = lps::min(r.min(k), other_min);
            r.max(k)       = lps::max(r.max(k), other_max);
        }
    }
    return r;
}

/// Reduce `in` across the *block*; the value is valid in thread 0 (the reference's
/// `warpNum > 1` second stage).
[[nodiscard]] inline LBox<lcrt::Var<float>> block_reduce_box(const LBox<lcrt::Var<float>>& in) noexcept
{
    // `__shared__ PlainAABB aabbData[K_WARPS]` -> 2 x 3 floats per warp.
    $shared<float> s_min{3u * kWarps};
    $shared<float> s_max{3u * kWarps};

    auto lane = lcrt::warp_lane_id();
    auto wid  = lcrt::thread_x() / 32u;

    auto per_warp = group_reduce_box<32u>(in, 0u);
    $if(lane == 0u)
    {
        for(int k = 0; k < 3; ++k)
        {
            s_min[wid * 3u + std::uint32_t(k)] = per_warp.min(k);
            s_max[wid * 3u + std::uint32_t(k)] = per_warp.max(k);
        };
    };
    lcrt::sync_block();

    // stage 2: the first warp folds the kWarps per-warp results; the other lanes hold
    // the identity (harmless: only lane 0's value is used, and it folds lanes 0..kWarps-1)
    LBox<lcrt::Var<float>> r = device_empty_box();
    $if(lane < kWarps)
    {
        for(int k = 0; k < 3; ++k)
        {
            r.min(k) = s_min[lane * 3u + std::uint32_t(k)];
            r.max(k) = s_max[lane * 3u + std::uint32_t(k)];
        };
    };
    auto staged = group_reduce_box<kWarps>(r, 0u);
    return staged;
}

/// `partials[bid] = union(in[bid*kBlock .. min(n, (bid+1)*kBlock)))`.
auto& reduce_aabb_chunks_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LAABB>, lcrt::Buffer<LAABB>,
                                                std::uint32_t, std::uint32_t>(
        [](lcrt::BufferVar<LAABB> in, lcrt::BufferVar<LAABB> partials,
           lcrt::Var<std::uint32_t> n, lcrt::Var<std::uint32_t> block_count) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto gid = lcrt::dispatch_x();
            auto bid = gid / kBlock;

            // the surplus lanes contribute the identity box
            auto box = device_empty_box();
            $if(gid < n)
            {
                box = load_box_at(in, gid);
            };

            auto block_box = block_reduce_box(box);

            $if(lcrt::thread_x() == 0u)
            {
                $if(bid < block_count)
                {
                    lcrt::Var<LAABB> v;
                    store_aabb(v, block_box);
                    partials->write(bid, v);
                };
            };
        });
    return k;
}

/// The final pass: one block folds `partials[0 .. block_count-1]` into `out[0]`.
auto& reduce_aabb_final_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LAABB>, lcrt::Buffer<LAABB>,
                                               std::uint32_t, std::uint32_t>(
        [](lcrt::BufferVar<LAABB> partials, lcrt::BufferVar<LAABB> out,
           lcrt::Var<std::uint32_t> block_count, lcrt::Var<std::uint32_t> stride) noexcept
        {
            lcrt::set_block_size(kBlock);
            auto tid = lcrt::thread_x();

            // strided fold so that one block can swallow any number of partials
            auto box = device_empty_box();
            lcrt::Var<std::uint32_t> i = tid;
            $while(i < block_count)
            {
                box = box_merged(box, load_box_at(partials, i));
                i   = i + stride;
            };

            auto block_box = block_reduce_box(box);
            $if(tid == 0u)
            {
                lcrt::Var<LAABB> v;
                store_aabb(v, block_box);
                out->write(0u, v);
            };
        });
    return k;
}

/// `n == 0`: CUB writes the initial value (the empty box).
auto& reduce_aabb_empty_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LAABB>>(
        [](lcrt::BufferVar<LAABB> out) noexcept
        {
            lcrt::set_block_size(32u);
            $if(lcrt::dispatch_x() == 0u)
            {
                lcrt::Var<LAABB> v;
                store_aabb(v, device_empty_box());
                out->write(0u, v);
            };
        });
    return k;
}
}// namespace

void reduce_aabb_union(lud::LStream&              stream,
                       lud::LCBufferView<LAABB>   in,
                       lud::LBufferView<LAABB>    out,
                       std::size_t                n,
                       lud::LDeviceBuffer<LAABB>& partials)
{
    if(n == 0u)
    {
        stream.record(reduce_aabb_empty_kernel()(out).dispatch(32u));
        return;
    }

    const auto blocks = static_cast<std::uint32_t>(aabb_reduce_scratch_size(n));
    partials.resize(static_cast<std::size_t>(blocks));

    // whole blocks: the guard inside the kernel keeps the surplus lanes inert
    const auto threads = static_cast<std::uint32_t>(
        (n + kBlock - 1u) / kBlock * kBlock);

    stream.record(reduce_aabb_chunks_kernel()(in, partials.view(), static_cast<std::uint32_t>(n),
                                              blocks)
                      .dispatch(threads));

    // one block folds the partials (a single dispatch, whatever `blocks` is)
    stream.record(reduce_aabb_final_kernel()(partials.cview(), out, blocks, kBlock)
                      .dispatch(kBlock));
}
}// namespace uipc::backend::luisa::detail
