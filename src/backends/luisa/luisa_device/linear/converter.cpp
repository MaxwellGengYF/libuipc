/********************************************************************
 * @file converter.cpp
 * @brief T3.3 — the MatrixConverter port (see converter.h for the contract, the
 *        CUB -> T3.2 mapping and the equivalence argument of every restructure).
 *
 * Everything here is host code that *traces* LC kernels over the caller's
 * LStream.  Each kernel is a function-local static of `detail::cached_kernel`
 * (one compiled shader per (kernel, element type) for the process lifetime, R4)
 * and every buffer it touches is a launch argument (a captured view would become
 * a bound argument and keep pointing at a stale buffer, cookbook §7.9).
 *
 * Kernel inventory (all of them are index arithmetic / scatters the reference
 * also does with its own small kernels; the reusable pieces — sort, scan, RLE,
 * select, segmented reduce, gather — go through the T3.2 primitives instead):
 *   hash_ij             (uint64)(row) << 32 + col             [ref k1]
 *   unpack_ij           hash -> row / col                     [ref k2]
 *   tail_flags          1 at the last element of an equal run (restructure 2)
 *   fill_i32            a zero fill on the caller's stream     [ref BufferLaunch::fill]
 *   scatter_run_counts  col_counts_per_row[row] = run length   [ref _k1]
 *   ge2sym_counts       counts(i) = (row <= col)               [ref ge2sym k1]
 *   ge2sym_compact      in-place compaction + the total count  [ref ge2sym k2]
 *   sym2ge_flags        flags(i) = (row == col), id(i) = i      [ref sym2ge k1]
 *   sym2ge_mirror       [Diag | Upper | Lower] through the T3.1 device views
 *                                                              [ref sym2ge k2]
 * Pure copies are recorded as LC copy commands, not as kernels.
 *********************************************************************/
#include <luisa_device/linear/converter.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace uipc::backend::luisa::luisa_device::linalg
{
namespace
{
using lcrt::Buffer;
using lcrt::BufferVar;
using lcrt::Var;

/// cuda_tool::default_block_dim parity (the reference launches 256-thread blocks).
constexpr std::uint32_t kBlockSize = 256u;
/// The reference's ij-hash packing: `(uint64(row) << 32) + uint64(col)`.
constexpr std::uint64_t kHashShift = 32ull;
constexpr std::uint64_t kHashMask  = 0xFFFF'FFFFull;

[[nodiscard]] inline std::uint32_t round_up(std::uint64_t n,
                                            std::uint32_t  block = kBlockSize) noexcept
{
    return static_cast<std::uint32_t>((n + block - 1u) / block * block);
}

/// `p ? t : f` — spelled out because the argument order is the reverse of how it
/// reads (cookbook §7.6 trap 1; hit again in the T3.2 probe).
template <typename F, typename Tv, typename P>
[[nodiscard]] inline auto pick(F&& f, Tv&& t, P&& p) noexcept
{
    return lcrt::select(std::forward<F>(f), std::forward<Tv>(t), std::forward<P>(p));
}

// ---------------------------------------------------------------------------
//  local kernels (the reference's own small kernels)
// ---------------------------------------------------------------------------
/// ij_hash_input(i) = (uint64(row(i)) << 32) + uint64(col(i)).
/// Reference `matrix_converter_radix_sort_indices_and_blocks_k1_kernel`; the
/// `sort_index(i) = i` write it also did is owned by T3.2's sort now.
[[nodiscard]] inline auto& hash_ij_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::uint64_t>,
                                           std::uint32_t>(
        [](BufferVar<std::int32_t> rows, BufferVar<std::int32_t> cols,
           BufferVar<std::uint64_t> hash, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                auto r = lcrt::cast<std::uint64_t>(rows->read(i));
                auto c = lcrt::cast<std::uint64_t>(cols->read(i));
                hash->write(i, (r << kHashShift) + c);
            };
        });
    return k;
}

/// hash(i) -> (row(i), col(i)); reference `..._k2_kernel`.
[[nodiscard]] inline auto& unpack_ij_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::uint64_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           std::uint32_t>(
        [](BufferVar<std::uint64_t> hash, BufferVar<std::int32_t> rows,
           BufferVar<std::int32_t> cols, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                auto h = hash->read(i);
                rows->write(i, lcrt::cast<std::int32_t>(h >> kHashShift));
                cols->write(i, lcrt::cast<std::int32_t>(h & kHashMask));
            };
        });
    return k;
}

/// flags(i) = 1 at the last element of each run of equal keys, 0 elsewhere.
/// This is the reference's `fill(sorted_partition, 0)` plus the scatter
/// `sorted_partition[offset(i) + count(i) - 1] = 1` folded into one branchless
/// predicate (restructure 2 in converter.h).
template <typename Key>
[[nodiscard]] inline auto& tail_flags_shader()
{
    static auto& k = detail::cached_kernel<Buffer<Key>, Buffer<std::int32_t>, std::uint32_t>(
        [](BufferVar<Key> keys, BufferVar<std::int32_t> flags, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                auto next    = lcrt::min(i + 1u, n - 1u);
                auto is_last = (i + 1u == n) | (keys->read(i) != keys->read(next));
                flags->write(i, pick(0, 1, is_last));  // is_last ? 1 : 0
            };
        });
    return k;
}

/// Reference `BufferLaunch().fill<int>(buf, value)` on the *caller's* stream
/// (LC has no fill command, and `LDeviceBuffer::fill` submits on the main stream,
/// which would not be batched into the caller's frame CommandList).
[[nodiscard]] inline auto& fill_i32_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>, std::uint32_t, std::int32_t>(
        [](BufferVar<std::int32_t> buf, Var<std::uint32_t> n, Var<std::int32_t> value) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                buf->write(i, value);
            };
        });
    return k;
}

/// col_counts_per_row[unique_rows(i)] = counts(i);
/// reference `matrix_converter_calculate_block_offsets_k1_kernel`.
[[nodiscard]] inline auto& scatter_run_counts_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           std::uint32_t>(
        [](BufferVar<std::int32_t> unique_rows, BufferVar<std::int32_t> counts,
           BufferVar<std::int32_t> out, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                out->write(unique_rows->read(i), counts->read(i));
            };
        });
    return k;
}

/// counts(i) = (row(i) <= col(i)) ? 1 : 0; reference `..._ge2sym_*_k1_kernel`
/// (the reference also copies ij/blocks there; those are recorded copies here).
[[nodiscard]] inline auto& ge2sym_counts_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           std::uint32_t>(
        [](BufferVar<std::int32_t> rows, BufferVar<std::int32_t> cols,
           BufferVar<std::int32_t> counts, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                counts->write(i, pick(0, 1, rows->read(i) <= cols->read(i)));
            };
        });
    return k;
}

/// Reference `matrix_converter_ge2sym_bcoo_k2_kernel` / `..._triplet_k2_kernel`:
/// compact the flagged entries in place and record the total count.
[[nodiscard]] inline auto& ge2sym_compact_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<double>,
                                           Buffer<double>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           std::uint32_t,
                                           std::uint32_t>(
        [](BufferVar<std::int32_t> dst_rows, BufferVar<std::int32_t> dst_cols,
           BufferVar<double> dst_vals, BufferVar<double> src_vals,
           BufferVar<std::int32_t> src_rows, BufferVar<std::int32_t> src_cols,
           BufferVar<std::int32_t> counts, BufferVar<std::int32_t> offsets,
           BufferVar<std::int32_t> total, Var<std::uint32_t> n,
           Var<std::uint32_t> components) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                auto count = counts->read(i);
                $if (count != 0)
                {
                    auto dst = lcrt::cast<std::uint32_t>(offsets->read(i));
                    dst_rows->write(dst, src_rows->read(i));
                    dst_cols->write(dst, src_cols->read(i));
                    $for (c, 0u, components)
                    {
                        dst_vals->write(dst * components + c, src_vals->read(i * components + c));
                    };
                };
                $if (i + 1u == n)
                {
                    total->write(0u, offsets->read(i) + counts->read(i));
                };
            };
        });
    return k;
}

/// flags(i) = (row(i) == col(i)) ? 1 : 0 and partition_input(i) = i;
/// reference `matrix_converter_sym2ge_k1_kernel`.
[[nodiscard]] inline auto& sym2ge_flags_shader()
{
    static auto& k = detail::cached_kernel<Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           std::uint32_t>(
        [](BufferVar<std::int32_t> rows, BufferVar<std::int32_t> cols,
           BufferVar<std::int32_t> flags, BufferVar<std::int32_t> partition,
           Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                flags->write(i, pick(0, 1, rows->read(i) == cols->read(i)));
                partition->write(i, lcrt::cast<std::int32_t>(i));
            };
        });
    return k;
}

/// Reference `matrix_converter_sym2ge_k2_kernel`, written through the T3.1
/// device views (the reference writes `to(i).write(row, col, block)` and
/// `f.value.transpose()`): the outputs before `diag_count` come from the
/// diagonal part of the partition, the rest from the off-diagonal part and are
/// mirrored, transposed, into the lower half at `i + sym_size - diag_count`.
template <typename T, int N>
[[nodiscard]] inline auto& sym2ge_mirror_shader()
{
    using WView = LTripletMatrixDeviceView<lcrt::Var<T>, N, N>;
    using CView = LCTripletMatrixDeviceView<lcrt::Var<T>, N, N>;

    static auto& k = detail::cached_kernel<Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<T>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           Buffer<T>,
                                           Buffer<std::int32_t>,
                                           Buffer<std::int32_t>,
                                           std::uint32_t,
                                           std::uint32_t,
                                           std::uint32_t,
                                           std::uint32_t>(
        [](BufferVar<std::int32_t> to_rows, BufferVar<std::int32_t> to_cols, BufferVar<T> to_vals,
           BufferVar<std::int32_t> from_rows, BufferVar<std::int32_t> from_cols,
           BufferVar<T> from_vals, BufferVar<std::int32_t> diag_index,
           BufferVar<std::int32_t> offdiag_index, Var<std::uint32_t> block_rows,
           Var<std::uint32_t> sym_size, Var<std::uint32_t> general_size,
           Var<std::uint32_t> diag_count) noexcept
        {
            auto to_v =
                WView::from_handles(to_rows, to_cols, to_vals, block_rows, block_rows, general_size);
            auto from_v = CView::from_handles(from_rows, from_cols, from_vals, block_rows,
                                              block_rows, sym_size);
            auto i = lcrt::dispatch_x();
            $if (i < sym_size)
            {
                auto is_lower = i >= diag_count;
                // clamped so the discarded operand of the pick is still in range
                auto off_i = i - lcrt::min(i, diag_count);
                // is_lower ? offdiag_index[off_i] : diag_index[i]
                auto index = pick(diag_index->read(i), offdiag_index->read(off_i), is_lower);
                auto f     = from_v(index);
                to_v(i).write(f.row_index, f.col_index, f.value);
                $if (is_lower)
                {
                    auto lower_i = i + sym_size - diag_count;
                    if constexpr(N == 1)
                    {
                        to_v(lower_i).write(f.col_index, f.row_index, f.value);
                    }
                    else
                    {
                        to_v(lower_i).write(f.col_index, f.row_index, f.value.transpose());
                    }
                };
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  host-side helper: the reference's implicit host read (`int h_count = count;`)
// ---------------------------------------------------------------------------
/// Commit the caller's frame CommandList, wait, read the single element through
/// the caller's stream, then re-open the frame so the caller keeps batching.
template <typename U>
[[nodiscard]] U read_back_count(LStream& stream, const LBufferView<U>& one)
{
    const bool was_open = stream.frame_open();
    stream.synchronize();  // commits the recorded work, then waits
    U host{};
    stream.record(one.native().copy_to(lcrt::span{&host, 1u}));
    stream.synchronize();
    if(was_open)
    {
        stream.begin();
    }
    return host;
}
}// namespace

// ---------------------------------------------------------------------------
//  default stream
// ---------------------------------------------------------------------------
LStream& default_matrix_stream()
{
    // One process-wide stream, deliberately: each LStream owns its own CommandList,
    // so a second one would be a hidden batching split (its commands would jump ahead
    // of the frame's recorded work, recording nothing into the frame's list).
    return LStream::current();
}

// ---------------------------------------------------------------------------
//  loose_resize — the reference's growth policy (see the header)
// ---------------------------------------------------------------------------
template <typename T, int N>
template <typename U>
LBufferView<U> LMatrixConverter<T, N>::loose_resize(LDeviceBuffer<U>& buf, std::size_t new_size)
{
    if(new_size == 0u)
    {
        return {};
    }
    if(buf.capacity() < new_size)
    {
        // Keep the displaced allocation alive: the caller may still have recorded
        // commands referencing it in its open frame CommandList (§7.10 carve).
        if(buf.capacity() != 0u && buf.size() != 0u)
        {
            m_retired.push_back(buf.view().base());
        }
        buf.reset();
        const auto grown = static_cast<std::size_t>(static_cast<double>(new_size) * m_reserve_ratio);
        buf.reserve(std::max(new_size, grown));
        // Expose the whole capacity once: LDeviceBuffer::resize value-initialises
        // the newly exposed tail, which is then amortised over the growths.
        buf.resize(buf.capacity());
    }
    if(buf.size() < new_size)
    {
        buf.resize(buf.capacity());
    }
    return buf.view(0u, new_size);
}

// ---------------------------------------------------------------------------
//  Triplet -> BCOO
// ---------------------------------------------------------------------------
template <typename T, int N>
void LMatrixConverter<T, N>::convert(const TripletMatrix& from, BCOOMatrix& to)
{
    convert(from, to, default_matrix_stream());
}

template <typename T, int N>
void LMatrixConverter<T, N>::convert(const TripletMatrix& from, BCOOMatrix& to, LStream& stream)
{
    to.reshape(from.rows(), from.cols());
    to.resize_triplets_discard(from.triplet_count());

    if(to.triplet_count() == 0u)
    {
        return;
    }

    _radix_sort_indices_and_blocks(from, to, stream);
    _make_unique_indices(from, to, stream);
    _make_unique_block_warp_reduction(from, to, stream);
}

template <typename T, int N>
void LMatrixConverter<T, N>::_radix_sort_indices_and_blocks(const TripletMatrix& from,
                                                            BCOOMatrix&         to,
                                                            LStream&            stream)
{
    // `to` is the reference's parameter; here the sorted indices/blocks stay in
    // the scratch until _make_unique_indices decides how many entries survive.
    static_cast<void>(to);

    const auto n = from.triplet_count();
    if(n == 0u)
    {
        return;
    }
    const auto n_u32 = static_cast<std::uint32_t>(n);

    auto hash_in = loose_resize(m_ij_hash_input, n);
    auto hash    = loose_resize(m_ij_hash, n);
    auto perm    = loose_resize(m_sort_index, n);
    auto blocks  = loose_resize(m_blocks_sorted, n * block_components);

    // 1. hash ij (the reference's k1)
    stream.record(hash_ij_shader()(from.row_indices(), from.col_indices(), hash_in, n_u32)
                      .dispatch(n_u32));

    // 2. stable radix sort of the (row << 32) + col keys; `perm` is the
    //    reference's sort_index and is what the gather below consumes
    sort_index_pairs<std::uint64_t>(stream, hash_in, hash, perm, n, 0u, &m_sort_scratch);

    // 3. sort the block values (the reference's k3)
    permute_gather_strided<ScalarT>(stream, from.values(), perm, blocks, n, block_components);
}

template <typename T, int N>
void LMatrixConverter<T, N>::_make_unique_indices(const TripletMatrix& from,
                                                  BCOOMatrix&          to,
                                                  LStream&             stream)
{
    const auto n = from.triplet_count();
    if(n == 0u)
    {
        to.resize_triplets_discard(0u);
        return;
    }

    // the sorted hash array produced by _radix_sort_indices_and_blocks (the
    // reference's ij_pairs, run-length encoded): the run partition is identical
    // because the hash is injective for the indices a COO matrix carries
    auto keys    = loose_resize(m_ij_hash, n);
    auto unique  = loose_resize(m_unique_hash, n);
    auto starts  = loose_resize(m_run_starts, n);
    auto counts  = loose_resize(m_unique_counts, n);
    auto run_cnt = loose_resize(m_run_count, 1u);

    run_length_encode<std::uint64_t>(stream, keys, unique, starts, counts, run_cnt, n,
                                     &m_compact_scratch);

    const auto h_count = static_cast<std::size_t>(read_back_count<std::uint32_t>(stream, run_cnt));

    // the reference's resize_discard(h_count) — before the unpack, so the row/col
    // views below are taken from the final allocation
    to.resize_triplets_discard(h_count);

    if(h_count != 0u)
    {
        const auto h_u32 = static_cast<std::uint32_t>(h_count);
        // decode the unique hashes straight into the BCOO row/col arrays
        // (the reference's k1 over unique_ij_pairs)
        stream.record(unpack_ij_shader()(unique.subview(0u, h_count), to.row_indices(),
                                         to.col_indices(), h_u32)
                          .dispatch(h_u32));
    }
}

template <typename T, int N>
void LMatrixConverter<T, N>::_make_unique_block_warp_reduction(const TripletMatrix& from,
                                                               BCOOMatrix&          to,
                                                               LStream&             stream)
{
    const auto n       = from.triplet_count();
    const auto h_count = static_cast<std::size_t>(to.triplet_count());
    if(n == 0u || h_count == 0u)
    {
        return;
    }
    const auto n_u32 = static_cast<std::uint32_t>(n);

    auto keys   = loose_resize(m_ij_hash, n);  // sorted (see the caller order)
    auto blocks = loose_resize(m_blocks_sorted, n * block_components);
    auto flags  = loose_resize(m_tail_flags, n);
    auto segs   = loose_resize(m_seg_ids, n);

    // run tails -> exclusive scan -> the segment label of every element, i.e.
    // exactly the reference's sorted_partition_output
    stream.record(tail_flags_shader<std::uint64_t>()(keys, flags, n_u32).dispatch(n_u32));
    exclusive_scan<std::int32_t>(stream, flags, segs, n, &m_scan_scratch);

    // combine the duplicates of every unique (row, col): out[segment] += value
    // (the reference's FastSegmentalReduce over blocks_sorted)
    segmented_reduce_blocks<std::int32_t, ScalarT>(stream, segs, blocks, to.values(), h_count, n,
                                                   block_components, detail::Plus<ScalarT>{},
                                                   &m_compact_scratch);
}

// ---------------------------------------------------------------------------
//  BCOO -> BSR
// ---------------------------------------------------------------------------
template <typename T, int N>
void LMatrixConverter<T, N>::convert(const BCOOMatrix& from, BSRMatrix& to)
{
    convert(from, to, default_matrix_stream());
}

template <typename T, int N>
void LMatrixConverter<T, N>::convert(const BCOOMatrix& from, BSRMatrix& to, LStream& stream)
{
    // the row offsets first (they size the BSR), then the values/cols — BCOO and
    // BSR share both
    _calculate_block_offsets(from, to, stream);

    to.resize(from.non_zeros());

    if(from.non_zeros() != 0)
    {
        stream.record(to.values().native().copy_from(from.values().native()));
        stream.record(to.col_indices().native().copy_from(from.col_indices().native()));
    }
}

template <typename T, int N>
void LMatrixConverter<T, N>::_calculate_block_offsets(const BCOOMatrix& from,
                                                      BSRMatrix&        to,
                                                      LStream&          stream)
{
    const auto n         = static_cast<std::size_t>(from.non_zeros());
    const auto rows      = static_cast<std::size_t>(from.rows());
    const auto rows_plus = rows + 1u;

    to.reshape(from.rows(), from.cols());

    auto col_counts = loose_resize(m_col_counts_per_row, rows_plus);

    const auto rows_u32 = static_cast<std::uint32_t>(rows_plus);
    stream.record(fill_i32_shader()(col_counts, rows_u32, 0).dispatch(round_up(rows_plus)));

    if(n != 0u)
    {
        auto uniq    = loose_resize(m_unique_indices, n);
        auto counts  = loose_resize(m_unique_counts, n);
        auto starts  = loose_resize(m_run_starts, n);
        auto run_cnt = loose_resize(m_run_count, 1u);

        // the row indices of a BCOO are ascending, so the RLE gives the
        // non-empty rows and their block counts
        run_length_encode<std::int32_t>(stream, from.row_indices(), uniq, starts, counts, run_cnt,
                                        n, &m_compact_scratch);

        const auto h_count = static_cast<std::size_t>(
            read_back_count<std::uint32_t>(stream, run_cnt));

        if(h_count != 0u)
        {
            const auto h_u32 = static_cast<std::uint32_t>(h_count);
            stream.record(scatter_run_counts_shader()(uniq.subview(0u, h_count),
                                                      counts.subview(0u, h_count), col_counts, h_u32)
                              .dispatch(h_u32));
        }
    }

    // exclusive sum of the per-row block counts -> the BSR row pointers
    exclusive_scan<std::int32_t>(stream, col_counts, to.row_offsets(), rows_plus, &m_scan_scratch);
}

// ---------------------------------------------------------------------------
//  Doublet -> BCOO vector
// ---------------------------------------------------------------------------
template <typename T, int N>
void LMatrixConverter<T, N>::convert(const DoubletVector& from, BCOOVector& to)
{
    convert(from, to, default_matrix_stream());
}

template <typename T, int N>
void LMatrixConverter<T, N>::convert(const DoubletVector& from, BCOOVector& to, LStream& stream)
{
    to.reshape(from.count());
    to.resize_doublets_discard(from.doublet_count());

    if(to.doublet_count() == 0u)
    {
        return;
    }

    _radix_sort_indices_and_segments(from, to, stream);
    _make_unique_indices(from, to, stream);
    _make_unique_segment_warp_reduction(from, to, stream);
}

template <typename T, int N>
void LMatrixConverter<T, N>::_radix_sort_indices_and_segments(const DoubletVector& from,
                                                              BCOOVector&          to,
                                                              LStream&             stream)
{
    static_cast<void>(to);

    const auto n = from.doublet_count();
    if(n == 0u)
    {
        return;
    }

    auto sorted_indices  = loose_resize(m_indices_sorted, n);
    auto sorted_segments = loose_resize(m_segments_sorted, n * segment_components);

    if constexpr(segment_components == 1u)
    {
        // the reference's DeviceRadixSort().SortPairs(indices, indices_sorted,
        // segments, segments_sorted, n) — for a scalar payload T3.2's
        // sort_pairs_by_key has exactly that (key, value) shape
        sort_pairs_by_key<std::int32_t, ScalarT>(stream, from.indices(), sorted_indices,
                                                 from.values(), sorted_segments, n, 0u,
                                                 &m_sort_scratch);
    }
    else
    {
        // A segment is `segment_components` *contiguous scalars* (the T3.1 flat
        // layout), while sort_pairs_by_key's `Value` is one element and it permutes
        // the payload with permute_gather (components == 1) — so a multi-scalar
        // payload needs the strided gather, i.e. the reference's own
        // sort_index + "sort the segments" kernel shape:
        auto perm = loose_resize(m_sort_index, n);
        sort_index_pairs<std::int32_t>(stream, from.indices(), sorted_indices, perm, n, 0u,
                                       &m_sort_scratch);
        permute_gather_strided<ScalarT>(stream, from.values(), perm, sorted_segments, n,
                                        segment_components);
    }
}

template <typename T, int N>
void LMatrixConverter<T, N>::_make_unique_indices(const DoubletVector& from,
                                                  BCOOVector&          to,
                                                  LStream&             stream)
{
    const auto n = from.doublet_count();
    if(n == 0u)
    {
        to.resize_doublets_discard(0u);
        return;
    }

    auto keys    = loose_resize(m_indices_sorted, n);  // sorted by the caller
    auto uniq    = loose_resize(m_unique_indices, n);
    auto starts  = loose_resize(m_run_starts, n);
    auto counts  = loose_resize(m_unique_counts, n);
    auto run_cnt = loose_resize(m_run_count, 1u);

    run_length_encode<std::int32_t>(stream, keys, uniq, starts, counts, run_cnt, n,
                                    &m_compact_scratch);

    const auto h_count = static_cast<std::size_t>(read_back_count<std::uint32_t>(stream, run_cnt));

    to.resize_doublets_discard(h_count);

    if(h_count != 0u)
    {
        // the reference's matrix_converter_make_unique_vector_indices_k1_kernel
        // is a pure copy
        stream.record(to.indices().native().copy_from(uniq.subview(0u, h_count).native()));
    }
}

template <typename T, int N>
void LMatrixConverter<T, N>::_make_unique_segment_warp_reduction(const DoubletVector& from,
                                                                 BCOOVector&          to,
                                                                 LStream&             stream)
{
    const auto n       = from.doublet_count();
    const auto h_count = static_cast<std::size_t>(to.doublet_count());
    if(n == 0u || h_count == 0u)
    {
        return;
    }
    const auto n_u32 = static_cast<std::uint32_t>(n);

    auto keys     = loose_resize(m_indices_sorted, n);
    auto segments = loose_resize(m_segments_sorted, n * segment_components);
    auto flags    = loose_resize(m_tail_flags, n);
    auto segs     = loose_resize(m_seg_ids, n);

    stream.record(tail_flags_shader<std::int32_t>()(keys, flags, n_u32).dispatch(n_u32));
    exclusive_scan<std::int32_t>(stream, flags, segs, n, &m_scan_scratch);

    // the reference's FastSegmentalReduce<64,32> over segments_sorted
    segmented_reduce_blocks<std::int32_t, ScalarT>(stream, segs, segments, to.values(), h_count, n,
                                                   segment_components, detail::Plus<ScalarT>{},
                                                   &m_compact_scratch);
}

// ---------------------------------------------------------------------------
//  ge2sym — general -> symmetric (the upper triangular part, compacted)
// ---------------------------------------------------------------------------
template <typename T, int N>
template <typename MatrixT>
void LMatrixConverter<T, N>::ge2sym_impl(MatrixT& to, LStream& stream)
{
    const auto n = static_cast<std::size_t>(to.non_zeros());
    if(n == 0u)
    {
        // the reference reads its DeviceVar back here (a stale value for an empty
        // matrix); there is nothing to compact
        return;
    }
    const auto n_u32 = static_cast<std::uint32_t>(n);

    auto counts  = loose_resize(m_unique_counts, n);
    auto offsets = loose_resize(m_offsets, n);
    auto ij_row  = loose_resize(m_ij_row, n);
    auto ij_col  = loose_resize(m_ij_col, n);
    auto temp    = loose_resize(m_blocks_sorted, n * block_components);
    auto total   = loose_resize(m_total_count, 1u);

    // 0. the upper-triangle flag, plus the copies the in-place compaction needs
    //    (the reference's k1 writes all three)
    stream.record(ge2sym_counts_shader()(to.row_indices(), to.col_indices(), counts, n_u32)
                      .dispatch(n_u32));
    stream.record(ij_row.native().copy_from(to.row_indices().native()));
    stream.record(ij_col.native().copy_from(to.col_indices().native()));
    stream.record(temp.native().copy_from(to.values().native()));

    // 1. exclusive sum of the flags -> the destination of every kept entry
    exclusive_scan<std::int32_t>(stream, counts, offsets, n, &m_scan_scratch);

    // 2. compact in place: the destination of entry i never exceeds i, and the
    //    sources are the copies taken above, so no read/write can race
    stream.record(ge2sym_compact_shader()(to.row_indices(), to.col_indices(), to.values(), temp,
                                          ij_row, ij_col, counts, offsets, total, n_u32,
                                          block_components)
                      .dispatch(n_u32));

    const auto h_total = static_cast<std::size_t>(read_back_count<std::int32_t>(stream, total));

    to.resize_triplets_discard(h_total);
}

template <typename T, int N>
void LMatrixConverter<T, N>::ge2sym(BCOOMatrix& to)
{
    ge2sym(to, default_matrix_stream());
}

template <typename T, int N>
void LMatrixConverter<T, N>::ge2sym(BCOOMatrix& to, LStream& stream)
{
    ge2sym_impl(to, stream);
}

template <typename T, int N>
void LMatrixConverter<T, N>::ge2sym(TripletMatrix& to)
{
    ge2sym(to, default_matrix_stream());
}

template <typename T, int N>
void LMatrixConverter<T, N>::ge2sym(TripletMatrix& to, LStream& stream)
{
    ge2sym_impl(to, stream);
}

// ---------------------------------------------------------------------------
//  sym2ge — symmetric (upper packed) -> general (mirrored + transposed)
// ---------------------------------------------------------------------------
template <typename T, int N>
void LMatrixConverter<T, N>::sym2ge(const BCOOMatrix& from, BCOOMatrix& to)
{
    sym2ge(from, to, default_matrix_stream());
}

template <typename T, int N>
void LMatrixConverter<T, N>::sym2ge(const BCOOMatrix& from, BCOOMatrix& to, LStream& stream)
{
    const auto sym_size = static_cast<std::size_t>(from.non_zeros());

    if(sym_size == 0u)
    {
        // nothing to mirror; the reference's `2 * (sym_size - rows) + rows`
        // underflows here (restructure 6 in converter.h)
        to.resize_discard(from.rows(), from.cols(), 0u);
        return;
    }

    const auto sym_u32  = static_cast<std::uint32_t>(sym_size);
    const auto rows_u32 = static_cast<std::uint32_t>(from.rows());

    auto flags      = loose_resize(m_flags, sym_size);
    auto part_input = loose_resize(m_sort_index_input, sym_size);
    auto diag_index = loose_resize(m_sort_index, sym_size);
    auto off_index  = loose_resize(m_unique_indices, sym_size);
    auto sel_count  = loose_resize(m_selected_count, 1u);
    auto off_count  = loose_resize(m_offdiag_count, 1u);

    // 0. the diagonal flag + the identity permutation (the reference's k1)
    stream.record(sym2ge_flags_shader()(from.row_indices(), from.col_indices(), flags, part_input,
                                        sym_u32)
                      .dispatch(sym_u32));

    // 1. stable partition into [diag | off-diag]: the reference's
    //    DevicePartition::Flagged writes both halves of one array, T3.2's
    //    select_if writes the selected prefix — so both flavours are used and the
    //    two sequences are exactly the reference's two halves.
    select_if<std::int32_t>(stream, part_input, flags, diag_index, sel_count, sym_size,
                            &m_compact_scratch);
    select_if<std::int32_t>(stream, part_input, flags, off_index, off_count, sym_size,
                            &m_compact_scratch, /*keep_when_zero=*/true);

    const auto diag_count = static_cast<std::size_t>(
        read_back_count<std::uint32_t>(stream, sel_count));

    // 2. size the general matrix (identical to the reference's formula whenever
    //    the input is a well-formed packed-upper matrix, i.e. one diagonal entry
    //    per block row)
    const auto general_size = 2u * (sym_size - diag_count) + diag_count;

    to.resize_discard(from.rows(), from.cols(), general_size);

    // 3. copy blocks and ij as [ Diag | Upper | Lower ] (the reference's k2)
    stream.record(sym2ge_mirror_shader<T, N>()(to.row_indices(), to.col_indices(), to.values(),
                                               from.row_indices(), from.col_indices(), from.values(),
                                               diag_index, off_index, rows_u32, sym_u32,
                                               static_cast<std::uint32_t>(general_size),
                                               static_cast<std::uint32_t>(diag_count))
                      .dispatch(sym_u32));

    // 4. the general matrix is sorted by (row, col) in place (the reference's
    //    second _radix_sort_indices_and_blocks overload)
    _radix_sort_indices_and_blocks(to, stream);
}

template <typename T, int N>
void LMatrixConverter<T, N>::_radix_sort_indices_and_blocks(BCOOMatrix& to, LStream& stream)
{
    const auto n = static_cast<std::size_t>(to.non_zeros());
    if(n == 0u)
    {
        return;
    }
    const auto n_u32 = static_cast<std::uint32_t>(n);

    auto hash_in = loose_resize(m_ij_hash_input, n);
    auto hash    = loose_resize(m_ij_hash, n);
    auto perm    = loose_resize(m_sort_index, n);
    auto blocks  = loose_resize(m_blocks_sorted, n * block_components);

    stream.record(hash_ij_shader()(to.row_indices(), to.col_indices(), hash_in, n_u32)
                      .dispatch(n_u32));
    sort_index_pairs<std::uint64_t>(stream, hash_in, hash, perm, n, 0u, &m_sort_scratch);

    // the reference writes the sorted row/col in the same kernel that gathers the
    // blocks; here the row/col column is the hash decode and the blocks come from
    // permute_gather_strided into the scratch, then a recorded copy (the gather
    // must not run in place: `dst(i) = src(perm(i))` would race)
    stream.record(unpack_ij_shader()(hash, to.row_indices(), to.col_indices(), n_u32)
                      .dispatch(n_u32));
    permute_gather_strided<ScalarT>(stream, to.values(), perm, blocks, n, block_components);
    stream.record(to.values().native().copy_from(blocks.native()));
}

// ---------------------------------------------------------------------------
//  explicit instantiations — the two the reference backend uses
// ---------------------------------------------------------------------------
template class LMatrixConverter<double, 1>;
template class LMatrixConverter<double, 3>;
}// namespace uipc::backend::luisa::luisa_device::linalg
