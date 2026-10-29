#pragma once
/********************************************************************
 * @file converter.h
 * @brief T3.3 — the LuisaCompute port of
 *        src/backends/cuda/algorithm/matrix_converter.{h,details/.inl}.
 *
 * The reference class `uipc::backend::cuda::MatrixConverter<T,N>` turns the
 * triplet/doublet containers into the assembled sparse structures the solvers
 * consume:
 *
 *   convert(DeviceTripletMatrix<T,N>&, DeviceBCOOMatrix<T,N>&)   Triplet -> BCOO
 *   convert(DeviceBCOOMatrix<T,N>&,   DeviceBSRMatrix<T,N>&)     BCOO   -> BSR
 *   convert(DeviceDoubletVector<T,N>&,DeviceBCOOVector<T,N>&)    Doublet-> BCOO vector
 *   ge2sym (BCOO&) / ge2sym (Triplet&)    general -> upper (symmetric) part
 *   sym2ge (const BCOO&, BCOO&)           upper-packed -> general (mirrored)
 *   reserve_ratio (Float) / loose_resize  the scratch-buffer growth policy
 *
 * Naming follows the T3.1 precedent (§5 of the cookbook): the class is
 * `lla::LMatrixConverter<T,N>` (like `LVec`/`LMat`/`LTripletMatrix`) and a
 * `lla::MatrixConverter<T,N>` alias keeps the reference's spelling.
 *
 * The reference is CUDA-only (<<<>>> launches, cudaStream_t, CUB), so it cannot
 * be host-included and is *not* the test oracle; the equivalence tests compare
 * against an independent CPU model of the same mathematical definition
 * (apps/tests/backends/luisa/luisa_device_converter_tests.cpp).
 *
 * CUB -> T3.2 primitive mapping (the reference's call sites, in order)
 * -----------------------------------------------------------------
 *  | reference (cub::)                                    | T3.3 |
 *  |------------------------------------------------------|------|
 *  | DeviceRadixSort().SortPairs(ij_hash_in, ij_hash,      | sort_index_pairs<std::uint64_t> |
 *  |   sort_index_in, sort_index, n)                       |      |
 *  | DeviceRadixSort().SortPairs(indices, indices_sorted,  | sort_pairs_by_key<std::int32_t,double> when the |
 *  |   segments, segments_sorted, n)                       | payload is one scalar, else sort_index_pairs + |
 *  |                                                       | permute_gather_strided (restructure 8) |
 *  | _k3_kernel / _in_place_k3_kernel (gather by sort_index)| permute_gather_strided<double> |
 *  | DeviceScan().ExclusiveSum (4 sites)                   | exclusive_scan<std::int32_t> |
 *  | DeviceRunLengthEncode().Encode (3 sites)              | run_length_encode<Key> |
 *  | DevicePartition().Flagged (sym2ge)                    | select_if<std::int32_t> (both flavours) |
 *  | FastSegmentalReduce<>().reduce (block values)         | segmented_reduce_blocks<std::int32_t,double> |
 *  | cub::WarpReduce in _make_unique_*_warp_reduction      | (inside segmented_reduce_blocks) |
 *
 * Local kernels (no T3.2 primitive exists for these, and they are cheap:
 * index arithmetic / scatters over the caller's data):
 *   hash_ij, unpack_ij, tail_flags, fill_i32, scatter_run_counts,
 *   ge2sym_counts, ge2sym_compact, sym2ge_flags, sym2ge_mirror.
 * Pure buffer copies are recorded as LC copy commands (`stream.record(dst.copy_from(src))`),
 * never as kernels.
 *
 * Deliberate restructures (each with its equivalence argument)
 * -----------------------------------------------------------
 *  1. **RLE key = the sorted `uint64` hash** instead of the reference's
 *     `DeviceRunLengthEncode` over a `MatrixConverterIntPair` buffer (T3.2's
 *     `run_length_encode` is scalar-keyed).  `(row<<32)+col` is injective for
 *     the non-negative indices a COO matrix carries, and the hash array is the
 *     *sorted* one the sort already produced, so the run partition is
 *     identical to the reference's and `unique_ij_pairs` is its decode.
 *  2. **Run-tail marking = a per-element predicate**, not
 *     `fill(sorted_partition, 0)` + the scatter `sorted_partition[offset+count-1] = 1`
 *     (no fill command exists in LC, and a fill on the main stream inside an open
 *     frame would not be batched with the rest).  `flag(i) = (i+1 == n) ||
 *     (key(i) != key(i+1))` writes 1 at exactly the run tails and 0 elsewhere,
 *     i.e. the same flag array the reference's scan consumes, whose exclusive
 *     scan is the same per-element segment label.
 *  3. **No identity `sort_index_input`**: T3.2's sort owns its identity/ping-pong
 *     pair (the reference's `sort_index_input(i) = i` write is subsumed).  The
 *     buffer is kept — and used — as sym2ge's `partition_index_input`, which is
 *     the reference's own aliasing there.
 *  4. **No `DeviceBuffer<MatrixConverterIntPair>`**: the pair lives in two
 *     `int32_t` arrays, which is what the kernels read/write anyway.
 *  5. **`permute_gather_strided` for the block gather** and a recorded copy for
 *     the in-place variant's write-back: the reference's in-place `_k3` writes
 *     row/col *and* gathers the blocks in one kernel; the gather is a T3.2
 *     primitive, so it is one, and the row/col column comes from the unpack
 *     kernel (same values, same order).
 *  6. **`sym2ge` sizes the output defensively**: the reference computes
 *     `2*(sym_size - rows) + rows`, which underflows when the input is empty or
 *     has fewer entries than block rows (a malformed packed-upper input).  The
 *     port clamps it to `max(2*(sym_size-diag)+diag, sym_size)`, identical for
 *     every input the reference accepts (`sym_size >= diag_count`).
 *  7. The doublet path's `FastSegmentalReduce<64,32>` becomes
 *     `segmented_reduce_blocks` (block size 256): same fold, identical result for
 *     exactly-representable values, only the floating-point association order of
 *     a >1-warp run differs.
 *  8. **A multi-scalar payload is gathered with `permute_gather_strided`**, not
 *     with `sort_pairs_by_key`: T3.2's `Value` is one element and its internal
 *     gather uses `components == 1`, so on the flat block/segment layout it would
 *     keep only the first scalar of every element.  `sort_pairs_by_key` is used
 *     where the payload really is one scalar (`segment_components == 1`), which is
 *     the reference's own `SortPairs` shape for `T`/`N == 1`.
 *
 * Buffer/lifetime rules (cookbook §7.9, §7.10)
 * -------------------------------------------
 *  * Every kernel takes its buffers as launch arguments and is dispatched over
 *    the caller's `LStream` (recorded into the open frame CommandList); no view is
 *    ever captured into a kernel lambda, so a reallocated buffer cannot be
 *    silently kept alive by a compiled shader.
 *  * The scratch buffers are class members like the reference's
 *    (`m_ij_hash`, `m_col_counts_per_row`, ...) and grow through
 *    `loose_resize`, which keeps the reference's `reserve_ratio` policy *and*
 *    retires the previous allocation instead of freeing it: a grown scratch must
 *    not invalidate a handle an already-recorded command still references.
 *  * The reference reads a run count back to the host (`int h_count = count;`),
 *    which is a synchronisation.  The port commits the caller's frame
 *    CommandList, waits, reads, and re-opens the frame — see `read_back_count`.
 *********************************************************************/
#include <luisa_device/linear/linear.h>
#include <luisa_device/stream.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace uipc::backend::luisa::luisa_device::linalg
{
// ---------------------------------------------------------------------------
//  reference-index-pair helper (surface parity; see restructure 4 above)
// ---------------------------------------------------------------------------
/// The reference's `MatrixConverterIntPair`.  Kept because it is part of the
/// reference header's public surface; the port's kernels keep the two fields in
/// two `int32_t` arrays instead of a struct buffer.
///
/// NOTE (RULE M1, cookbook §7.2): the comparison is a *hidden friend*, not a
/// namespace-scope `operator==`.  A namespace-level `operator==` inside `linalg`
/// would hide LuisaCompute's global DSL `operator==` for every unqualified
/// comparison of DSL scalars in this namespace — `Var<uint32_t> == Var<uint32_t>`
/// then fails to compile even in an unrelated kernel (it did, one debugging round).
struct MatrixConverterIntPair
{
    std::int32_t x = 0;
    std::int32_t y = 0;

    friend constexpr bool operator==(const MatrixConverterIntPair& l,
                                     const MatrixConverterIntPair& r)
    {
        return l.x == r.x && l.y == r.y;
    }
};

/// The stream the reference's implicit "default stream" overloads use: the
/// process-wide `LStream` (`LStream::current()`, i.e. the backend's `frame_stream()`).
/// It must be the *same* stream everywhere: each `LStream` owns its own CommandList,
/// so a second one is a hidden batching split (its commands would submit immediately
/// and jump ahead of the frame's recorded work).  Consumer code inside a frame
/// passes the frame's `LStream` to the three-argument overloads instead.
[[nodiscard]] UIPC_LUISA_DEVICE_API LStream& default_matrix_stream();

// ---------------------------------------------------------------------------
//  LMatrixConverter
// ---------------------------------------------------------------------------
/**
 * @brief The reference's `MatrixConverter<T,N>` on top of T3.1/T3.2.
 *
 * `T` is the storage scalar (double/float, `uipc::Float` == double) and `N` the
 * block/segment size: the container family it works on is
 * `LTripletMatrix<T,N,N>` / `LBCOOMatrix<T,N,N>` / `LBSRMatrix<T,N>` /
 * `LDoubletVector<T,N>` / `LBCOOVector<T,N>`.
 *
 * Only `<double,1>` and `<double,3>` are instantiated (explicitly, in
 * converter.cpp) — exactly the two the reference backend uses
 * (`MatrixConverter<Float,3>` in the FEM/ABD/dytopo paths,
 * `MatrixConverter<Float,1>` in diff_sim); another combination is a link error.
 * The scratch members allocate lazily, so a converter may be constructed before
 * the LC runtime is up.
 */
template <typename T, int N>
class UIPC_LUISA_DEVICE_API LMatrixConverter
{
  public:
    using TripletMatrix = LTripletMatrix<T, N, N>;
    using BCOOMatrix    = LBCOOMatrix<T, N, N>;
    using BSRMatrix     = LBSRMatrix<T, N>;
    using DoubletVector = LDoubletVector<T, N>;
    using BCOOVector    = LBCOOVector<T, N>;

    /// Reference's `DeviceTripletMatrix<T,N>::ValueT` (`T` or `LMat<T,N,N>`).
    using BlockMatrix = typename TripletMatrix::ValueT;
    /// Reference's `DeviceDoubletVector<T,N>::ValueT` (`T` or `LVec<T,N>`).
    using SegmentVector = typename DoubletVector::ValueT;

    static_assert(std::is_same_v<T, buffer_scalar_of_t<T>>,
                  "LMatrixConverter: the element type must be a plain scalar");
    static_assert(N > 0, "LMatrixConverter: the block size must be positive");

    /// Scalars per matrix block (the T3.1 flat block layout).
    static constexpr std::uint32_t block_components = static_cast<std::uint32_t>(N)
                                                      * static_cast<std::uint32_t>(N);
    /// Scalars per segment vector entry.
    static constexpr std::uint32_t segment_components = static_cast<std::uint32_t>(N);

    using ScalarT = typename TripletMatrix::ScalarT;

    LMatrixConverter() = default;
    LMatrixConverter(const LMatrixConverter&)            = delete;
    LMatrixConverter& operator=(const LMatrixConverter&) = delete;

    // ---- growth policy (the reference's reserve_ratio) ---------------------
    void reserve_ratio(double ratio) { m_reserve_ratio = ratio; }
    [[nodiscard]] double reserve_ratio() const { return m_reserve_ratio; }

    /**
     * @brief The reference's `loose_resize`: grow `buf` to at least `new_size`
     *        with `reserve_ratio` headroom, and return the `new_size`-element view.
     *
     * Two differences from the reference, both invisible to a consumer:
     *  * the *whole capacity* is exposed once, so the value-initialisation
     *    `LDeviceBuffer::resize` performs on the newly exposed tail is amortised
     *    over the growths instead of paid on every call;
     *  * a growth retires the previous allocation (`m_retired`) instead of
     *    freeing it, because the caller's open frame CommandList may still hold a
     *    recorded handle to it (cookbook §7.10).
     */
    template <typename U>
    [[nodiscard]] LBufferView<U> loose_resize(LDeviceBuffer<U>& buf, std::size_t new_size);

    // ---- Triplet -> BCOO ---------------------------------------------------
    /// The reference's spelling (dispatches on `default_matrix_stream()`).
    void convert(const TripletMatrix& from, BCOOMatrix& to);
    void convert(const TripletMatrix& from, BCOOMatrix& to, LStream& stream);

    void _radix_sort_indices_and_blocks(const TripletMatrix& from, BCOOMatrix& to, LStream& stream);
    void _radix_sort_indices_and_blocks(BCOOMatrix& to, LStream& stream);
    void _make_unique_indices(const TripletMatrix& from, BCOOMatrix& to, LStream& stream);
    void _make_unique_block_warp_reduction(const TripletMatrix& from, BCOOMatrix& to, LStream& stream);

    // ---- BCOO -> BSR -------------------------------------------------------
    void convert(const BCOOMatrix& from, BSRMatrix& to);
    void convert(const BCOOMatrix& from, BSRMatrix& to, LStream& stream);
    void _calculate_block_offsets(const BCOOMatrix& from, BSRMatrix& to, LStream& stream);

    // ---- Doublet -> BCOO vector -------------------------------------------
    void convert(const DoubletVector& from, BCOOVector& to);
    void convert(const DoubletVector& from, BCOOVector& to, LStream& stream);

    void _radix_sort_indices_and_segments(const DoubletVector& from, BCOOVector& to, LStream& stream);
    void _make_unique_indices(const DoubletVector& from, BCOOVector& to, LStream& stream);
    void _make_unique_segment_warp_reduction(const DoubletVector& from, BCOOVector& to, LStream& stream);

    // ---- symmetric <-> general --------------------------------------------
    /// Keep the upper triangular part (`i <= j`), compacted in input order.
    void ge2sym(BCOOMatrix& to);
    void ge2sym(BCOOMatrix& to, LStream& stream);
    void ge2sym(TripletMatrix& to);
    void ge2sym(TripletMatrix& to, LStream& stream);

    /// Expand an upper-packed symmetric matrix to the general one
    /// (`[Diag | Upper | Lower]`, mirrored blocks transposed), then sort it.
    void sym2ge(const BCOOMatrix& from, BCOOMatrix& to);
    void sym2ge(const BCOOMatrix& from, BCOOMatrix& to, LStream& stream);

  private:
    /// ge2sym's body: the reference implements it twice (BCOO and triplet), the
    /// two containers share `non_zeros()/row_indices()/col_indices()/values()`.
    template <typename MatrixT>
    void ge2sym_impl(MatrixT& to, LStream& stream);

    double m_reserve_ratio = 1.5;

    // ---- the reference's scratch members ----------------------------------
    LDeviceBuffer<std::int32_t>  m_col_counts_per_row;
    LDeviceBuffer<std::int32_t>  m_unique_indices;
    LDeviceBuffer<std::int32_t>  m_unique_counts;
    LDeviceBuffer<std::uint32_t> m_run_count;  ///< RLE's run count (`DeviceVar<int>`)

    LDeviceBuffer<std::int32_t> m_sort_index_input;  ///< sym2ge's partition input (identity)
    LDeviceBuffer<std::int32_t> m_sort_index;        ///< sort permutation / partition output

    LDeviceBuffer<std::int32_t> m_offsets;

    LDeviceBuffer<std::int32_t> m_ij_row;  ///< the reference's `ij_pairs`
    LDeviceBuffer<std::int32_t> m_ij_col;

    LDeviceBuffer<std::uint64_t> m_ij_hash_input;
    LDeviceBuffer<std::uint64_t> m_ij_hash;  ///< sorted keys

    LDeviceBuffer<std::uint64_t> m_unique_hash;  ///< unique keys (RLE output)
    LDeviceBuffer<std::int32_t>  m_run_starts;   ///< RLE's starts (unused, see header note)

    LDeviceBuffer<ScalarT>      m_blocks_sorted;
    LDeviceBuffer<std::int32_t> m_seg_ids;  ///< exclusive scan of the run tails
    LDeviceBuffer<std::int32_t> m_tail_flags;

    LDeviceBuffer<std::int32_t> m_indices_sorted;
    LDeviceBuffer<ScalarT>      m_segments_sorted;

    LDeviceBuffer<std::int32_t>  m_flags;  ///< sym2ge's select flag
    LDeviceBuffer<std::uint32_t> m_selected_count;  ///< #diagonal entries (select_if)
    LDeviceBuffer<std::uint32_t> m_offdiag_count;   ///< #off-diagonal entries
    LDeviceBuffer<std::int32_t>  m_total_count;  ///< ge2sym's `Dense<int> total_count`

    // ---- T3.2 scratch ------------------------------------------------------
    SortScratch    m_sort_scratch;
    CompactScratch m_compact_scratch;
    ScanScratch    m_scan_scratch;

    /// Allocations retired by a growth; never freed while the converter lives.
    std::vector<std::shared_ptr<void>> m_retired;
};

/// The reference's spelling (`linalg::MatrixConverter<T,N>`).
template <typename T, int N>
using MatrixConverter = LMatrixConverter<T, N>;
}// namespace uipc::backend::luisa::luisa_device::linalg
