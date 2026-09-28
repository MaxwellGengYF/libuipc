/********************************************************************
 * @file algorithms.cpp
 * @brief T3.2 — device algorithm primitives (see algorithms.h for the contract,
 *        the reference CUB call sites and the design decisions).
 *
 * Everything here is host code that *traces* LC kernels: one function-local
 * static per (kernel, element type) so a kernel is compiled once per process
 * (R5/R4: never compile inside a frame loop), and every buffer is passed as a
 * launch argument at dispatch time (a captured view would become a bound argument
 * and keep pointing at a stale buffer — cookbook §7.9).
 *
 * Kernel inventory (all validated by scratch/t32_probe on the GPU before this file
 * was written: exclusive warp prefix scan, exclusive ballot count, the
 * per-warp shared histogram + stable intra-block rank, the transposed
 * [digit][block] radix histogram with the flat-scan destination formula,
 * device-bounded `$for` component loops):
 *
 *   scan_blocks      one 256-element tile -> exclusive prefix + block total
 *   scan_add_back    out[i] += block_sums[block(i)]
 *   block_reduce     one 256-element tile -> one partial
 *   fold_partials    one block folds the partials in a fixed order (deterministic)
 *   fill_identity    perm[i] = i
 *   flags_from_pred  (header, predicate flavour of select_if)
 *   compact_scatter  stable stream-compaction scatter
 *   rle_flags        head flags of an ascending key array
 *   rle_scatter      run key / start + the run count
 *   rle_counts       run lengths from the run starts
 *   seg_reduce       FastSegmentalReduce transcription (warp head-segmented fold,
 *                    predicated store / atomic combine)
 *   radix_hist_rank  per-warp digit histogram (transposed) + stable block-rank
 *   radix_scatter    stable scatter of (key, index) pairs
 *********************************************************************/
#include <luisa_device/linear/algorithms.h>

#include <vector>

namespace uipc::backend::luisa::luisa_device::linalg
{
namespace detail
{
using lcrt::Buffer;
using lcrt::BufferVar;
using lcrt::Expr;
using lcrt::Var;

// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------
[[nodiscard]] inline uint32_t round_up(uint64_t n, uint32_t block) noexcept
{
    return static_cast<uint32_t>((n + block - 1u) / block * block);
}

template <typename T>
[[nodiscard]] constexpr T zero_of() noexcept
{
    return T{0};
}

/// `p ? t : f`, spelled out because the argument order is the reverse of how it
/// reads (cookbook §7.6 trap #1: `select(f, t, p)`).
template <typename F, typename T, typename P>
[[nodiscard]] inline auto pick(F&& f, T&& t, P&& p) noexcept
{
    return lcrt::select(std::forward<F>(f), std::forward<T>(t), std::forward<P>(p));
}

/// Combine two scalars with the reduction kind (host and device generic).
template <ReduceKind Kind, typename V>
[[nodiscard]] inline V reduce_combine(const V& a, const V& b) noexcept
{
    if constexpr(Kind == ReduceKind::Sum)
        return lmath::detail::add(a, b);
    else if constexpr(Kind == ReduceKind::Min)
        return lcrt::min(a, b);
    else
        return lcrt::max(a, b);
}

template <ReduceKind Kind, typename V>
[[nodiscard]] inline auto warp_reduce(const V& v) noexcept
{
    if constexpr(Kind == ReduceKind::Sum)
        return lcrt::warp_active_sum(v);
    else if constexpr(Kind == ReduceKind::Min)
        return lcrt::warp_active_min(v);
    else
        return lcrt::warp_active_max(v);
}

/// Warp-wide *segment label*, constant for every lane of one segment and strictly
/// increasing across segments: the number of heads up to and including this lane.
/// (The exclusive count alone is not enough — it differs between a head lane and
/// the rest of its own segment, which would stop the fold at the segment's first
/// lane.)
[[nodiscard]] inline auto warp_segment_label(const Expr<bool>& is_head) noexcept
{
    return lcrt::warp_prefix_count_bits(is_head) + pick(0u, 1u, is_head);
}

/// Per-warp head-segmented reduction, transcribed from
/// `cub::WarpReduce::HeadSegmentedReduce` as used by
/// algorithm/details/fast_segmental_reduce.inl: a warp-wide fold whose segments
/// start at the lanes where `is_head` is set.
///
/// Implemented as a *downward* Hillis-Steele segmented scan: a lane merges with
/// lane+d iff they carry the same segment label `ord`. Because a segment is a
/// contiguous run of lanes, the label equality is exactly "no segment boundary
/// between them", so every partial stays inside its segment. After log2(32) steps
/// the entry of a head lane is the fold over the whole warp window of its segment
/// — which is precisely what HeadSegmentedReduce returns to the head lane.
/// (`lcrt::warp_prefix_sum` alone would not do: CUB's head-segmented variant resets
/// at every head, and LC has no segmented scan.)
template <ReduceKind Kind, typename V, typename O>
[[nodiscard]] inline V warp_segmented_fold(V v, const O& ord) noexcept
{
    auto lane = lcrt::warp_lane_id();
    for(std::uint32_t d = 1u; d < warp_size_v; d <<= 1u)
    {
        auto other = lcrt::warp_read_lane(v, lane + d);
        auto label = lcrt::warp_read_lane(ord, lane + d);
        auto merge = (lane + d < warp_size_v) & (label == ord);
        v          = pick(v, reduce_combine<Kind>(v, other), merge);
    }
    return v;
}

// ---------------------------------------------------------------------------
//  exclusive_scan
// ---------------------------------------------------------------------------
/// One 256-element tile per block: an exclusive prefix within the tile plus the
/// tile total. Every block is launched full (the dispatch count is rounded up), so
/// the last lane of every warp exists and the tail is predicated, not branched.
template <typename T>
[[nodiscard]] inline auto& scan_blocks_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, Buffer<T>, uint32_t>(
        [](BufferVar<T> in, BufferVar<T> out, BufferVar<T> sums, Var<uint32_t> n) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            constexpr uint32_t W = kScanBlockSize / warp_size_v;
            $shared<T> warp_total{W};
            auto tid   = lcrt::thread_x();
            auto gid   = lcrt::dispatch_x();
            auto bid   = lcrt::block_id().x;
            auto lane  = lcrt::warp_lane_id();
            auto wid   = tid / warp_size_v;
            auto valid = gid < n;
            auto idx   = lcrt::min(gid, n - 1u);
            auto v     = pick(zero_of<T>(), in->read(idx), valid);  // valid ? v : 0
            auto excl  = lcrt::warp_prefix_sum(v);                  // exclusive over lanes
            auto incl  = lmath::detail::add(excl, v);
            $if (lane == warp_size_v - 1u) { warp_total[wid] = incl; };
            lcrt::sync_block();
            Var<T> off = zero_of<T>();
            $for (w, 0u, W)
            {
                off = lmath::detail::add(off, pick(zero_of<T>(), warp_total[w], w < wid));
            };
            $if (valid) { out->write(gid, lmath::detail::add(off, excl)); };
            $if (tid == kScanBlockSize - 1u) { sums->write(bid, lmath::detail::add(off, incl)); };
        });
    return k;
}

template <typename T>
[[nodiscard]] inline auto& scan_add_back_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, uint32_t>(
        [](BufferVar<T> out, BufferVar<T> sums, Var<uint32_t> n) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            auto gid = lcrt::dispatch_x();
            $if (gid < n)
            {
                out->write(gid, lmath::detail::add(out->read(gid), sums->read(lcrt::block_id().x)));
            };
        });
    return k;
}

template <typename T>
void exclusive_scan_impl(LStream& stream, const LCBufferView<T>& in, const LBufferView<T>& out,
                         size_t n, ScanScratch& scratch)
{
    if(n == 0u)
        return;
    constexpr uint32_t B = kScanBlockSize;

    // Level sizes: level 0 is the data, level l+1 is level l's block sums.
    std::vector<size_t> lvl_n{n};
    while(lvl_n.back() > B)
        lvl_n.push_back((lvl_n.back() + B - 1u) / B);
    const size_t levels = lvl_n.size();

    // Sums layout: level l (1 <= l < levels) occupies [off(l), off(l) + lvl_n[l]).
    // Level 0 always writes lvl_n[1] block sums (levels == 1: an unused slack slot,
    // hence the +1), so every level's block-sum target is in range.
    std::vector<size_t> lvl_off(levels, 0u);
    size_t              sums_size = 0u;
    if(levels == 1u)
        sums_size = lvl_n[0];
    for(size_t l = 1; l < levels; ++l)
    {
        lvl_off[l] = sums_size;
        sums_size += lvl_n[l];
    }
    sums_size += 1u;
    auto sums = scratch.require<T>(sums_size);

    auto level_sums = [&](size_t l) {
        return sums.subview(lvl_off[l], lvl_n[l]);
    };

    // Level 0: data -> out (+ block sums).
    auto lvl0_sums = (levels == 1u) ? sums.subview(0u, 1u) : level_sums(1u);
    stream.record(scan_blocks_shader<T>()(in, out, lvl0_sums, static_cast<uint32_t>(n))
                      .dispatch(round_up(n, B)));

    // Levels 1..: scan the block sums in place, writing the next level's sums.
    for(size_t l = 1; l < levels; ++l)
    {
        auto cur  = level_sums(l);
        auto next = (l + 1u < levels) ? level_sums(l + 1u) : sums.subview(sums_size - 1u, 1u);
        stream.record(scan_blocks_shader<T>()(cur, cur, next, static_cast<uint32_t>(lvl_n[l]))
                          .dispatch(round_up(lvl_n[l], B)));
    }

    // Add every level's exclusive sums of the previous level back into its output.
    for(size_t l = levels - 1u; l-- > 0u;)
    {
        auto dest = (l == 0u) ? out : level_sums(l);
        auto add  = level_sums(l + 1u);
        stream.record(scan_add_back_shader<T>()(dest, add, static_cast<uint32_t>(lvl_n[l]))
                          .dispatch(round_up(lvl_n[l], B)));
    }
}

// ---------------------------------------------------------------------------
//  device_reduce
// ---------------------------------------------------------------------------
template <typename T, ReduceKind Kind>
[[nodiscard]] inline auto& block_reduce_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, uint32_t, T>(
        [](BufferVar<T> in, BufferVar<T> parts, Var<uint32_t> n, Var<T> identity) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            constexpr uint32_t W = kScanBlockSize / warp_size_v;
            $shared<T> warp_red{W};
            auto tid   = lcrt::thread_x();
            auto gid   = lcrt::dispatch_x();
            auto bid   = lcrt::block_id().x;
            auto lane  = lcrt::warp_lane_id();
            auto wid   = tid / warp_size_v;
            auto valid = gid < n;
            auto idx   = lcrt::min(gid, n - 1u);
            auto v     = pick(identity, in->read(idx), valid);
            auto w     = warp_reduce<Kind>(v);
            $if (lane == warp_size_v - 1u) { warp_red[wid] = w; };
            lcrt::sync_block();
            $if (tid == 0u)
            {
                Var<T> acc = warp_red[0];
                $for (k, 1u, W) { acc = reduce_combine<Kind>(acc, warp_red[k]); };
                parts->write(bid, acc);
            };
        });
    return k;
}

/// One block folds the partials with a fixed stride order (deterministic: no
/// atomics, no arrival order — the R6 requirement).
template <typename T, ReduceKind Kind>
[[nodiscard]] inline auto& fold_partials_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, uint32_t, T>(
        [](BufferVar<T> parts, BufferVar<T> out, Var<uint32_t> count, Var<T> identity) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            constexpr uint32_t W = kScanBlockSize / warp_size_v;
            $shared<T> warp_red{W};
            auto tid  = lcrt::thread_x();
            auto lane = lcrt::warp_lane_id();
            auto wid  = tid / warp_size_v;
            Var<T>     acc = identity;
            Var<uint32_t> j = tid;
            $while (j < count)
            {
                acc = reduce_combine<Kind>(acc, parts->read(j));
                j   = j + kScanBlockSize;
            };
            auto w = warp_reduce<Kind>(acc);
            $if (lane == warp_size_v - 1u) { warp_red[wid] = w; };
            lcrt::sync_block();
            $if (tid == 0u)
            {
                Var<T> r = warp_red[0];
                $for (k, 1u, W) { r = reduce_combine<Kind>(r, warp_red[k]); };
                out->write(0u, r);
            };
        });
    return k;
}

template <typename T>
[[nodiscard]] constexpr T reduce_identity(ReduceKind kind) noexcept
{
    if(kind == ReduceKind::Sum)
        return T{0};
    else if(kind == ReduceKind::Min)
        return std::numeric_limits<T>::max();
    else
        return std::numeric_limits<T>::lowest();
}

template <typename T, ReduceKind Kind>
void device_reduce_dispatch(LStream& stream, const LCBufferView<T>& in, const LBufferView<T>& out,
                            size_t n, const LBufferView<T>& partials, uint32_t parts, T identity)
{
    if(n != 0u)
    {
        stream.record(block_reduce_shader<T, Kind>()(in, partials, static_cast<uint32_t>(n),
                                                    identity)
                          .dispatch(round_up(n, kScanBlockSize)));
    }
    stream.record(
        fold_partials_shader<T, Kind>()(partials, out, parts, identity).dispatch(kScanBlockSize));
}

template <typename T>
void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<T>& in,
                        const LBufferView<T>& out, size_t n, ScanScratch& scratch)
{
    const uint32_t parts    = static_cast<uint32_t>((n + kScanBlockSize - 1u) / kScanBlockSize);
    auto           partials = scratch.require<T>(parts == 0u ? 1u : parts);
    const T        identity = reduce_identity<T>(kind);
    switch(kind)
    {
        case ReduceKind::Sum:
            device_reduce_dispatch<T, ReduceKind::Sum>(stream, in, out, n, partials, parts, identity);
            break;
        case ReduceKind::Min:
            device_reduce_dispatch<T, ReduceKind::Min>(stream, in, out, n, partials, parts, identity);
            break;
        case ReduceKind::Max:
            device_reduce_dispatch<T, ReduceKind::Max>(stream, in, out, n, partials, parts, identity);
            break;
    }
}

// ---------------------------------------------------------------------------
//  select_if (stream compaction)
// ---------------------------------------------------------------------------
/// `out_count = 0` (the n == 0 path: there is nothing to scan).
[[nodiscard]] inline auto& zero_count_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::uint32_t>>(
        [](BufferVar<std::uint32_t> out_count) noexcept
        {
            lcrt::set_block_size(32u);
            $if (lcrt::dispatch_x() == 0u) { out_count->write(0u, 0u); };
        });
    return k;
}

[[nodiscard]] inline auto& count_from_scan_shader()
{
    // out_count = scanned[n-1] + (flags[n-1] != 0), or n - that for the inverted
    // ("keep the zeros") flavour.
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>, Buffer<std::int32_t>,
                                  Buffer<std::uint32_t>, uint32_t, uint32_t>(
        [](BufferVar<std::int32_t> scanned, BufferVar<std::int32_t> flags,
           BufferVar<std::uint32_t> out_count, Var<uint32_t> n, Var<uint32_t> keep_zero) noexcept
        {
            lcrt::set_block_size(32u);
            $if (lcrt::dispatch_x() == 0u)
            {
                auto nonzero = pick(0u, 1u, flags->read(n - 1u) != 0);  // like a bool cast
                auto total   = lcrt::cast<std::uint32_t>(scanned->read(n - 1u)) + nonzero;
                out_count->write(0u, pick(total, n - total, keep_zero != 0u));
            };
        });
    return k;
}

template <typename T>
[[nodiscard]] inline auto& compact_scatter_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<std::int32_t>, Buffer<std::int32_t>, Buffer<T>,
                                  uint32_t, uint32_t>(
        [](BufferVar<T> in, BufferVar<std::int32_t> flags, BufferVar<std::int32_t> offsets,
           BufferVar<T> out, Var<uint32_t> n, Var<uint32_t> keep_zero) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            auto gid   = lcrt::dispatch_x();
            auto valid = gid < n;
            auto idx   = lcrt::min(gid, n - 1u);
            auto nz    = flags->read(idx) != 0;             // "flag set"
            auto keep  = pick(nz, !nz, keep_zero != 0u);    // keep_zero ? !nz : nz
            auto src   = lcrt::cast<std::uint32_t>(offsets->read(idx));
            auto dst   = pick(src, idx - src, keep_zero != 0u);  // zeros are packed at the front
            $if (valid & keep) { out->write(dst, in->read(idx)); };
        });
    return k;
}

template <typename T>
void select_if_impl(LStream& stream, const LCBufferView<T>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<T>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch)
{
    if(n == 0u)
    {
        stream.record(zero_count_shader()(out_count).dispatch(32u));
        return;
    }
    // One carve of 2n: the scanned flags are written next to the mirrored count
    // array, so the count kernel can still read the last input flag (the caller's
    // flags are read-only).
    auto work    = carve(scratch.i32, scratch.retired, 2u * n);
    auto scanned = work.subview(0u, n);
    exclusive_scan_impl(stream, flags, scanned, n, scratch.scan);
    // scanned[i] = #flagged before i; for the "keep zeros" flavour the destination
    // is i - scanned[i] (see compact_scatter) — no second scan is needed.
    stream.record(count_from_scan_shader()(scanned, flags, out_count,
                                          static_cast<uint32_t>(n),
                                          keep_when_zero ? 1u : 0u)
                      .dispatch(32u));
    stream.record(compact_scatter_shader<T>()(in, flags, scanned, out, static_cast<uint32_t>(n),
                                              keep_when_zero ? 1u : 0u)
                      .dispatch(round_up(n, kScanBlockSize)));
}

// ---------------------------------------------------------------------------
//  run_length_encode
// ---------------------------------------------------------------------------
template <typename Key>
[[nodiscard]] inline auto& rle_flags_shader()
{
    static auto& k = detail::cached_kernel<Buffer<Key>, Buffer<std::int32_t>, uint32_t>(
        [](BufferVar<Key> keys, BufferVar<std::int32_t> flags, Var<uint32_t> n) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            auto gid   = lcrt::dispatch_x();
            auto valid = gid < n;
            auto idx   = lcrt::min(gid, n - 1u);
            auto prev  = pick(idx, idx - 1u, idx > 0u);
            auto first = gid == 0u;
            auto changed_key = keys->read(prev) != keys->read(idx);
            auto head  = pick(changed_key, true, first);
            $if (valid) { flags->write(idx, pick(0, 1, head)); };
        });
    return k;
}

template <typename Key>
[[nodiscard]] inline auto& rle_scatter_shader()
{
    static auto& k = detail::cached_kernel<Buffer<Key>, Buffer<std::int32_t>, Buffer<std::int32_t>,
                                  Buffer<Key>, Buffer<std::int32_t>, Buffer<std::uint32_t>, uint32_t>(
        [](BufferVar<Key> keys, BufferVar<std::int32_t> flags, BufferVar<std::int32_t> offsets,
           BufferVar<Key> unique_out, BufferVar<std::int32_t> starts_out,
           BufferVar<std::uint32_t> run_count_out, Var<uint32_t> n) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            auto gid   = lcrt::dispatch_x();
            auto valid = gid < n;
            auto idx   = lcrt::min(gid, n - 1u);
            auto head  = flags->read(idx) != 0;
            $if (valid & head)
            {
                auto pos = lcrt::cast<std::uint32_t>(offsets->read(idx));
                unique_out->write(pos, keys->read(idx));
                starts_out->write(lcrt::cast<std::int32_t>(pos), lcrt::cast<std::int32_t>(idx));
            };
            $if (idx == n - 1u)
            {
                auto nonzero = pick(0u, 1u, head);
                run_count_out->write(
                    0u, lcrt::cast<std::uint32_t>(offsets->read(idx)) + nonzero);
            };
        });
    return k;
}

[[nodiscard]] inline auto& rle_counts_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>, Buffer<std::int32_t>, Buffer<std::uint32_t>,
                                  uint32_t>(
        [](BufferVar<std::int32_t> starts, BufferVar<std::int32_t> counts,
           BufferVar<std::uint32_t> run_count, Var<uint32_t> n) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            auto gid = lcrt::dispatch_x();
            auto r   = run_count->read(0u);
            $if (gid < r)
            {
                auto next_idx = lcrt::min(gid + 1u, n - 1u);
                auto last     = gid + 1u == r;
                auto end = pick(starts->read(next_idx), lcrt::cast<std::int32_t>(n), last);
                counts->write(lcrt::cast<std::int32_t>(gid), end - starts->read(gid));
            };
        });
    return k;
}

template <typename Key>
void run_length_encode_impl(LStream& stream, const LCBufferView<Key>& keys,
                            const LBufferView<Key>& unique_out, const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch)
{
    if(n == 0u)
    {
        stream.record(zero_count_shader()(run_count_out).dispatch(32u));
        return;
    }
    auto work   = carve(scratch.i32, scratch.retired, 2u * n);
    auto flags  = work.subview(0u, n);
    auto offsets = work.subview(n, n);
    stream.record(rle_flags_shader<Key>()(keys, flags, static_cast<uint32_t>(n))
                      .dispatch(round_up(n, kScanBlockSize)));
    exclusive_scan_impl(stream, flags, offsets, n, scratch.scan);
    stream.record(rle_scatter_shader<Key>()(keys, flags, offsets, unique_out, starts_out,
                                            run_count_out, static_cast<uint32_t>(n))
                      .dispatch(round_up(n, kScanBlockSize)));
    stream.record(rle_counts_shader()(starts_out, counts_out, run_count_out,
                                     static_cast<uint32_t>(n))
                      .dispatch(round_up(n, kScanBlockSize)));
}

// ---------------------------------------------------------------------------
//  segmented_reduce (FastSegmentalReduce transcription)
// ---------------------------------------------------------------------------
/// Atomic combine of a cross-warp segment partial (the reference's
/// `atomic_add(&out(i), value)` branch of fast_segmental_reduce.inl, generalised to
/// the fold kind).
template <ReduceKind Kind, typename B, typename I, typename V>
inline void atomic_combine(const B& buffer, const I& index, const V& value) noexcept
{
    if constexpr(Kind == ReduceKind::Sum)
        buffer->atomic(index).fetch_add(value);
    else if constexpr(Kind == ReduceKind::Min)
        buffer->atomic(index).fetch_min(value);
    else
        buffer->atomic(index).fetch_max(value);
}

[[nodiscard]] inline auto& fill_double_shader()
{
    static auto& k = detail::cached_kernel<Buffer<double>, uint32_t, double>(
        [](BufferVar<double> out, Var<uint32_t> n, Var<double> identity) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            auto gid = lcrt::dispatch_x();
            $if (gid < n) { out->write(gid, identity); };
        });
    return k;
}

template <typename Key, ReduceKind Kind>
[[nodiscard]] inline auto& segmented_reduce_shader()
{
    static auto& k = detail::cached_kernel<Buffer<Key>, Buffer<double>, Buffer<double>, uint32_t, uint32_t, double>(
            [](BufferVar<Key> keys, BufferVar<double> values, BufferVar<double> out,
               Var<uint32_t> n, Var<uint32_t> out_size, Var<double> identity) noexcept
            {
                lcrt::set_block_size(kScanBlockSize);
                auto gid   = lcrt::dispatch_x();
                auto lane  = lcrt::warp_lane_id();
                auto valid = gid < n;
                auto idx   = lcrt::min(gid, n - 1u);
                auto prev  = pick(idx, idx - 1u, idx > 0u);
                auto next  = lcrt::min(idx + 1u, n - 1u);
                auto has_prev = gid > 0u;
                auto has_next = gid + 1u < n;
                auto k_cur  = keys->read(idx);
                auto changed = keys->read(prev) != k_cur;
                // The reference's flag logic (fast_segmental_reduce.inl), evaluated
                // predicated instead of branched (RULE M2).
                auto head = pick(changed, true, lane == 0u);
                auto marker = lcrt::max(
                pick(0u, 1u, (lane == 0u) & has_prev & !changed),
                pick(0u, 1u,
                     (lane == warp_size_v - 1u) & has_next
                         & (keys->read(next) == k_cur)));
                auto ord = warp_segment_label(head);
                auto cross  = warp_segmented_fold<ReduceKind::Max>(marker, ord) != 0u;
                // Per-lane value; the invalid tail contributes the identity.
                auto value = pick(identity, values->read(idx), valid);
                auto seg   = warp_segmented_fold<Kind>(value, ord);
                $if (head & valid)
                {
                    auto kidx = lcrt::cast<std::uint32_t>(k_cur);
                    $if (kidx < out_size)
                    {
                        $if (cross)
                        {
                            atomic_combine<Kind>(out, kidx, seg);
                        }
                        $else
                        {
                            out->write(kidx, seg);
                        };
                    };
                };
            });
    return k;
}

/// Flat-block flavour: `components` contiguous scalars per element (the T3.1
/// block layout). The component loop is device-bounded, which the probe verified.
template <typename Key, ReduceKind Kind>
[[nodiscard]] inline auto& segmented_reduce_blocks_shader()
{
    static auto& k = detail::cached_kernel<Buffer<Key>, Buffer<double>, Buffer<double>, uint32_t, uint32_t,
                                  uint32_t, double>(
        [](BufferVar<Key> keys, BufferVar<double> values, BufferVar<double> out, Var<uint32_t> n,
           Var<uint32_t> out_size, Var<uint32_t> components, Var<double> identity) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            auto gid   = lcrt::dispatch_x();
            auto lane  = lcrt::warp_lane_id();
            auto valid = gid < n;
            auto idx   = lcrt::min(gid, n - 1u);
            auto prev  = pick(idx, idx - 1u, idx > 0u);
            auto next  = lcrt::min(idx + 1u, n - 1u);
            auto has_prev = gid > 0u;
            auto has_next = gid + 1u < n;
            auto k_cur    = keys->read(idx);
            auto changed  = keys->read(prev) != k_cur;
            auto head = pick(changed, true, lane == 0u);
            auto marker   = pick(0u, 1u,
                                 ((lane == 0u) & has_prev & !changed)
                                     | ((lane == warp_size_v - 1u) & has_next
                                        & (keys->read(next) == k_cur)));
            auto ord      = warp_segment_label(head);
            auto cross    = warp_segmented_fold<ReduceKind::Max>(marker, ord) != 0u;
            $for (c, 0u, components)
            {
                auto value = pick(identity, values->read(idx * components + c), valid);
                auto seg   = warp_segmented_fold<Kind>(value, ord);
                $if (head & valid)
                {
                    auto kidx = lcrt::cast<std::uint32_t>(k_cur);
                    $if (kidx < out_size)
                    {
                        auto slot = kidx * components + c;
                        $if (cross)
                        {
                            atomic_combine<Kind>(out, slot, seg);
                        }
                        $else
                        {
                            out->write(slot, seg);
                        };
                    };
                };
            };
        });
    return k;
}

template <typename Key>
void segmented_reduce_dispatch(LStream& stream, const LCBufferView<Key>& keys_sorted,
                               const LCBufferView<double>& values, const LBufferView<double>& out,
                               size_t out_size, size_t n, uint32_t components)
{
    // Sum is the only instantiated op (see the header: a non-sum op would need a
    // non-atomic deterministic cross-warp fold). The kernels stay generic.
    constexpr auto kKind    = ReduceKind::Sum;
    const double   identity = reduce_identity<double>(kKind);
    if(out_size != 0u)
    {
        // The identity fill must cover every *slot* of the block output, i.e.
        // `out_size * components` scalars: a segment that crosses a warp boundary is
        // accumulated with atomic_add by its head warp (the reference's is_cross_warp
        // path), so *its* slots have to be identity-filled before the combine.  Filling
        // only the first `out_size` scalars left such a segment adding onto whatever the
        // (reused) buffer still held; the reference fills the whole matrix buffer
        // (`BufferLaunch().fill<Matrix>(out, zero)`), i.e. all M*N scalars of every
        // element.  Found by T3.3 while converting with components = N*N into a reused
        // container.
        const auto filled = static_cast<uint32_t>(out_size * components);
        stream.record(fill_double_shader()(out, filled, identity)
                          .dispatch(round_up(filled, kScanBlockSize)));
    }
    if(n == 0u || out_size == 0u)
        return;
    if(components == 1u)
    {
        stream.record(segmented_reduce_shader<Key, kKind>()(keys_sorted, values, out,
                                                           static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(out_size),
                                                           identity)
                          .dispatch(round_up(n, kScanBlockSize)));
    }
    else
    {
        stream.record(segmented_reduce_blocks_shader<Key, kKind>()(
                          keys_sorted, values, out, static_cast<uint32_t>(n),
                          static_cast<uint32_t>(out_size), components, identity)
                          .dispatch(round_up(n, kScanBlockSize)));
    }
}

void segmented_reduce_impl(LStream& stream, const LCBufferView<std::int32_t>& keys_sorted,
                           const LCBufferView<double>& values, const LBufferView<double>& out,
                           size_t out_size, size_t n, uint32_t components)
{
    segmented_reduce_dispatch(stream, keys_sorted, values, out, out_size, n, components);
}

void segmented_reduce_impl(LStream& stream, const LCBufferView<std::uint64_t>& keys_sorted,
                           const LCBufferView<double>& values, const LBufferView<double>& out,
                           size_t out_size, size_t n, uint32_t components)
{
    segmented_reduce_dispatch(stream, keys_sorted, values, out, out_size, n, components);
}

// ---------------------------------------------------------------------------
//  radix sort
// ---------------------------------------------------------------------------
[[nodiscard]] inline auto& fill_identity_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>, uint32_t>(
        [](BufferVar<std::int32_t> perm, Var<uint32_t> n) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            auto gid = lcrt::dispatch_x();
            $if (gid < n) { perm->write(gid, lcrt::cast<std::int32_t>(gid)); };
        });
    return k;
}

/// Digit extraction: the key is biased (sign-bit flip for the top pass of a signed
/// key type) so that the *unsigned* digit order equals the signed key order.
template <typename K, typename B, typename S>
[[nodiscard]] inline auto radix_digit(const K& key, const B& bias, const S& shift) noexcept
{
    auto k64 = lcrt::cast<std::uint64_t>(key);
    return lcrt::cast<std::uint32_t>((k64 ^ bias) >> shift & 0xFFull);
}

/// Step 1 of a pass: the transposed [digit][block] digit histogram plus the stable
/// intra-block rank of every element (`ranks[gid]`), which the scatter reuses.
///
/// The histogram uses a per-warp shared counter array indexed by (warp, digit):
/// an atomic per lane is order independent, so the *counts* are deterministic. The
/// intra-warp rank is what has to respect the input order: LC exposes no
/// `__match_any_sync`, so each lane compares its digit against the digit of all 32
/// lanes of the warp (a 32-iteration shuffle loop, measured acceptable by the
/// probe: ~1 shuffle per element).
template <typename Key>
[[nodiscard]] inline auto& radix_hist_rank_shader()
{
    static auto& k = detail::cached_kernel<Buffer<Key>, Buffer<std::uint32_t>, Buffer<std::int32_t>, uint32_t, uint32_t,
                      uint64_t, uint32_t>(
            [](BufferVar<Key> keys, BufferVar<std::uint32_t> histo, BufferVar<std::int32_t> ranks,
               Var<uint32_t> n, Var<uint32_t> nb, Var<uint64_t> bias, Var<uint32_t> shift) noexcept
            {
                lcrt::set_block_size(kRadixBlockSize);
                constexpr uint32_t NW    = kRadixBlockSize / warp_size_v;
                constexpr uint32_t SHARD = NW * kRadixDigits;
                $shared<std::uint32_t> wcount{SHARD};
                auto tid = lcrt::thread_x();
                auto gid = lcrt::dispatch_x();
                auto bid = lcrt::block_id().x;
                $for (q, 0u, (SHARD + kRadixBlockSize - 1u) / kRadixBlockSize)
                {
                    auto s = tid + q * kRadixBlockSize;
                    $if (s < SHARD) { wcount[s] = 0u; };
                };
                lcrt::sync_block();

                auto lane  = lcrt::warp_lane_id();
                auto wid   = tid / warp_size_v;
                auto valid = gid < n;
                auto idx   = lcrt::min(gid, n - 1u);
                auto d     = radix_digit(keys->read(idx), bias, shift);
                $if (valid) { wcount.atomic(wid * kRadixDigits + d).fetch_add(1u); };
                lcrt::sync_block();

                // Per-warp exclusive prefix in place; the first kRadixDigits threads
                // also publish the block histogram.
                $if (tid < kRadixDigits)
                {
                    Var<std::uint32_t> acc = 0u;
                    $for (w, 0u, NW)
                    {
                        auto c                      = wcount[w * kRadixDigits + tid];
                        wcount[w * kRadixDigits + tid] = acc;
                        acc                         = acc + c;
                    };
                    histo->write(tid * nb + bid, acc);
                };
                lcrt::sync_block();

                auto votemask        = lcrt::warp_active_bit_mask(valid).x;
                Var<std::uint32_t> r = 0u;
                $for (j, 0u, warp_size_v)
                {
                    auto dj  = lcrt::warp_read_lane(d, j);
                    auto bit = (votemask >> j) & 1u;
                    r        = r + pick(0u, 1u, (j < lane) & (dj == d) & (bit == 1u));
                };
                $if (valid)
                {
                    ranks->write(gid, lcrt::cast<std::int32_t>(wcount[wid * kRadixDigits + d] + r));
                };
            });
    return k;
}

/// Step 3 of a pass: the stable scatter. The destination is
///   bucket_offset[digit]                      (keys with a smaller digit)
/// + (#equal digits in earlier blocks)          (the scanned histogram difference)
/// + ranks[gid]                                 (#equal digits earlier in the block)
/// which is exactly "input order inside a bucket" — the stability the converter
/// depends on, obtained without any atomic ticket (a ticket is arrival-ordered).
template <typename Key>
[[nodiscard]] inline auto& radix_scatter_shader()
{
    static auto& k = detail::cached_kernel<Buffer<Key>, Buffer<Key>, Buffer<std::int32_t>, Buffer<std::int32_t>,
                      Buffer<std::uint32_t>, Buffer<std::int32_t>, uint32_t, uint32_t, uint64_t,
                      uint32_t>([](BufferVar<Key> keys_in, BufferVar<Key> keys_out,
                                   BufferVar<std::int32_t> perm_in, BufferVar<std::int32_t> perm_out,
                                   BufferVar<std::uint32_t> scanned, BufferVar<std::int32_t> ranks,
                                   Var<uint32_t> n, Var<uint32_t> nb, Var<uint64_t> bias,
                                   Var<uint32_t> shift) noexcept
                                {
                                    lcrt::set_block_size(kRadixBlockSize);
                                    auto gid   = lcrt::dispatch_x();
                                    auto valid = gid < n;
                                    auto idx   = lcrt::min(gid, n - 1u);
                                    auto d     = radix_digit(keys_in->read(idx), bias, shift);
                                    auto bid   = lcrt::block_id().x;
                                    auto base  = scanned->read(d * nb);
                                    auto pos   = base + (scanned->read(d * nb + bid) - base)
                                               + lcrt::cast<std::uint32_t>(ranks->read(idx));
                                    $if (valid)
                                    {
                                        keys_out->write(pos, keys_in->read(idx));
                                        perm_out->write(pos, perm_in->read(idx));
                                    };
                                });
    return k;
}

template <typename Key>
[[nodiscard]] inline LDeviceBuffer<Key>& radix_key_ping_pong(SortScratch& scratch) noexcept
{
    if constexpr(std::is_same_v<Key, std::int32_t>)
        return scratch.keys_i32;
    else if constexpr(std::is_same_v<Key, std::uint32_t>)
        return scratch.keys_u32;
    else if constexpr(std::is_same_v<Key, std::int64_t>)
        return scratch.keys_i64;
    else
        return scratch.keys_u64;
}

template <typename Key>
void sort_index_pairs_impl(LStream& stream, const LCBufferView<Key>& keys_in,
                           const LBufferView<Key>& keys_out, const LBufferView<std::int32_t>& perm_out,
                           size_t n, uint32_t key_bits, SortScratch& scratch)
{
    if(n == 0u)
        return;
    const uint32_t passes = (key_bits + kRadixDigitBits - 1u) / kRadixDigitBits;
    const bool     signed_key = std::is_signed_v<Key>;
    const uint32_t nb = static_cast<uint32_t>((n + kRadixBlockSize - 1u) / kRadixBlockSize);

    auto histo = carve(scratch.histogram, scratch.retired, static_cast<size_t>(kRadixDigits) * nb);
    auto ranks = carve(scratch.ranks, scratch.retired, n);
    auto perms = carve(scratch.perm, scratch.retired, 2u * n);   // [0,n) identity, [n,2n) first ping
    auto keys  = carve(radix_key_ping_pong<Key>(scratch), scratch.retired, 2u * n);
    // NOTE: `carve` may reallocate a scratch buffer, so every per-type carve above
    // happens before the first view is used (they are different types here, except
    // histo/ranks/perms which are all carved from different buffers).
    auto perm_id = perms.subview(0u, n);
    auto perm_a  = perms.subview(n, n);
    auto perm_b  = perms.subview(0u, n);
    auto key_a   = keys.subview(0u, n);
    auto key_b   = keys.subview(n, n);

    stream.record(fill_identity_shader()(perm_id, static_cast<uint32_t>(n))
                      .dispatch(round_up(n, kScanBlockSize)));

    LBufferView<Key>         src_keys = keys_in;
    LBufferView<std::int32_t> src_perm = perm_id;
    const bool aliased = keys_in.base() == keys_out.base() && passes == 1u;
    if(aliased)
    {
        // An in-place sort with a single pass would gather from the buffer it is
        // writing (the permutation is not the identity), so stage the input.
        key_a.copy_from(keys_in, &stream);
        src_keys = key_a;
    }

    for(uint32_t pass = 0; pass < passes; ++pass)
    {
        const bool last = (pass + 1u == passes);
        // The sign-bit bias is applied to the top digit only: it is a bijective,
        // order-preserving remap of the signed key space.
        // Sign-bit bias: only meaningful (and only applied) for the *full* key
        // width, where "sorted by value" spans the whole signed range. For a
        // reduced `key_bits` the keys are taken as non-negative in [0, 2^key_bits)
        // and no bias is applied — otherwise a caller whose keys do not fill the
        // declared width would get a permuted order.
        const bool full_width = (key_bits == 8u * sizeof(Key));
        const std::uint64_t bias = (signed_key && full_width && last)
                                       ? (std::uint64_t{1} << (key_bits - 1u))
                                       : std::uint64_t{0};
        const uint32_t      shift = pass * kRadixDigitBits;
        auto dst_keys = last ? keys_out : ((pass % 2u == 0u) ? key_a : key_b);
        auto dst_perm = last ? perm_out : ((pass % 2u == 0u) ? perm_a : perm_b);

        stream.record(radix_hist_rank_shader<Key>()(src_keys, histo, ranks,
                                                   static_cast<uint32_t>(n), nb, bias, shift)
                          .dispatch(round_up(n, kRadixBlockSize)));
        // Exclusive scan of the transposed histogram (in place: the raw counts are
        // not needed afterwards). The digit-major layout makes
        //   bucket_offset[d]        = scanned[d * nb]
        //   earlier-block count[d,b] = scanned[d * nb + b] - bucket_offset[d]
        exclusive_scan_impl(stream, histo, histo, static_cast<size_t>(kRadixDigits) * nb,
                            scratch.scan);
        stream.record(radix_scatter_shader<Key>()(src_keys, dst_keys, src_perm, dst_perm, histo,
                                                  ranks, static_cast<uint32_t>(n), nb, bias, shift)
                          .dispatch(round_up(n, kRadixBlockSize)));
        src_keys = dst_keys;
        src_perm = dst_perm;
    }
}

void sort_index_pairs_impl(LStream& stream, const LCBufferView<std::int32_t>& keys_in,
                           const LBufferView<std::int32_t>& keys_out,
                           const LBufferView<std::int32_t>& perm_out, size_t n, uint32_t key_bits,
                           SortScratch& scratch)
{
    sort_index_pairs_impl<std::int32_t>(stream, keys_in, keys_out, perm_out, n, key_bits, scratch);
}

void sort_index_pairs_impl(LStream& stream, const LCBufferView<std::uint32_t>& keys_in,
                           const LBufferView<std::uint32_t>& keys_out,
                           const LBufferView<std::int32_t>& perm_out, size_t n, uint32_t key_bits,
                           SortScratch& scratch)
{
    sort_index_pairs_impl<std::uint32_t>(stream, keys_in, keys_out, perm_out, n, key_bits, scratch);
}

void sort_index_pairs_impl(LStream& stream, const LCBufferView<std::int64_t>& keys_in,
                           const LBufferView<std::int64_t>& keys_out,
                           const LBufferView<std::int32_t>& perm_out, size_t n, uint32_t key_bits,
                           SortScratch& scratch)
{
    sort_index_pairs_impl<std::int64_t>(stream, keys_in, keys_out, perm_out, n, key_bits, scratch);
}

void sort_index_pairs_impl(LStream& stream, const LCBufferView<std::uint64_t>& keys_in,
                           const LBufferView<std::uint64_t>& keys_out,
                           const LBufferView<std::int32_t>& perm_out, size_t n, uint32_t key_bits,
                           SortScratch& scratch)
{
    sort_index_pairs_impl<std::uint64_t>(stream, keys_in, keys_out, perm_out, n, key_bits, scratch);
}

void exclusive_scan_impl(LStream& stream, const LCBufferView<std::int32_t>& in,
                         const LBufferView<std::int32_t>& out, size_t n, ScanScratch& scratch)
{
    exclusive_scan_impl<std::int32_t>(stream, in, out, n, scratch);
}
void exclusive_scan_impl(LStream& stream, const LCBufferView<std::uint32_t>& in,
                         const LBufferView<std::uint32_t>& out, size_t n, ScanScratch& scratch)
{
    exclusive_scan_impl<std::uint32_t>(stream, in, out, n, scratch);
}
void exclusive_scan_impl(LStream& stream, const LCBufferView<std::int64_t>& in,
                         const LBufferView<std::int64_t>& out, size_t n, ScanScratch& scratch)
{
    exclusive_scan_impl<std::int64_t>(stream, in, out, n, scratch);
}
void exclusive_scan_impl(LStream& stream, const LCBufferView<std::uint64_t>& in,
                         const LBufferView<std::uint64_t>& out, size_t n, ScanScratch& scratch)
{
    exclusive_scan_impl<std::uint64_t>(stream, in, out, n, scratch);
}
void exclusive_scan_impl(LStream& stream, const LCBufferView<float>& in,
                         const LBufferView<float>& out, size_t n, ScanScratch& scratch)
{
    exclusive_scan_impl<float>(stream, in, out, n, scratch);
}
void exclusive_scan_impl(LStream& stream, const LCBufferView<double>& in,
                         const LBufferView<double>& out, size_t n, ScanScratch& scratch)
{
    exclusive_scan_impl<double>(stream, in, out, n, scratch);
}

void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<std::int32_t>& in,
                        const LBufferView<std::int32_t>& out, size_t n, ScanScratch& scratch)
{
    device_reduce_impl<std::int32_t>(kind, stream, in, out, n, scratch);
}
void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<std::uint32_t>& in,
                        const LBufferView<std::uint32_t>& out, size_t n, ScanScratch& scratch)
{
    device_reduce_impl<std::uint32_t>(kind, stream, in, out, n, scratch);
}
void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<std::int64_t>& in,
                        const LBufferView<std::int64_t>& out, size_t n, ScanScratch& scratch)
{
    device_reduce_impl<std::int64_t>(kind, stream, in, out, n, scratch);
}
void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<std::uint64_t>& in,
                        const LBufferView<std::uint64_t>& out, size_t n, ScanScratch& scratch)
{
    device_reduce_impl<std::uint64_t>(kind, stream, in, out, n, scratch);
}
void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<float>& in,
                        const LBufferView<float>& out, size_t n, ScanScratch& scratch)
{
    device_reduce_impl<float>(kind, stream, in, out, n, scratch);
}
void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<double>& in,
                        const LBufferView<double>& out, size_t n, ScanScratch& scratch)
{
    device_reduce_impl<double>(kind, stream, in, out, n, scratch);
}

void select_if_impl(LStream& stream, const LCBufferView<std::int32_t>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<std::int32_t>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch)
{
    select_if_impl<std::int32_t>(stream, in, flags, out, out_count, n, keep_when_zero, scratch);
}
void select_if_impl(LStream& stream, const LCBufferView<std::uint32_t>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<std::uint32_t>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch)
{
    select_if_impl<std::uint32_t>(stream, in, flags, out, out_count, n, keep_when_zero, scratch);
}
void select_if_impl(LStream& stream, const LCBufferView<std::int64_t>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<std::int64_t>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch)
{
    select_if_impl<std::int64_t>(stream, in, flags, out, out_count, n, keep_when_zero, scratch);
}
void select_if_impl(LStream& stream, const LCBufferView<std::uint64_t>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<std::uint64_t>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch)
{
    select_if_impl<std::uint64_t>(stream, in, flags, out, out_count, n, keep_when_zero, scratch);
}
void select_if_impl(LStream& stream, const LCBufferView<float>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<float>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch)
{
    select_if_impl<float>(stream, in, flags, out, out_count, n, keep_when_zero, scratch);
}
void select_if_impl(LStream& stream, const LCBufferView<double>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<double>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch)
{
    select_if_impl<double>(stream, in, flags, out, out_count, n, keep_when_zero, scratch);
}

void run_length_encode_impl(LStream& stream, const LCBufferView<std::int32_t>& keys,
                            const LBufferView<std::int32_t>& unique_out,
                            const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch)
{
    run_length_encode_impl<std::int32_t>(stream, keys, unique_out, starts_out, counts_out,
                                         run_count_out, n, scratch);
}
void run_length_encode_impl(LStream& stream, const LCBufferView<std::uint32_t>& keys,
                            const LBufferView<std::uint32_t>& unique_out,
                            const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch)
{
    run_length_encode_impl<std::uint32_t>(stream, keys, unique_out, starts_out, counts_out,
                                          run_count_out, n, scratch);
}
void run_length_encode_impl(LStream& stream, const LCBufferView<std::int64_t>& keys,
                            const LBufferView<std::int64_t>& unique_out,
                            const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch)
{
    run_length_encode_impl<std::int64_t>(stream, keys, unique_out, starts_out, counts_out,
                                         run_count_out, n, scratch);
}
void run_length_encode_impl(LStream& stream, const LCBufferView<std::uint64_t>& keys,
                            const LBufferView<std::uint64_t>& unique_out,
                            const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch)
{
    run_length_encode_impl<std::uint64_t>(stream, keys, unique_out, starts_out, counts_out,
                                          run_count_out, n, scratch);
}
}// namespace detail
}// namespace uipc::backend::luisa::luisa_device::linalg
