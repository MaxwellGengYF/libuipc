#pragma once
/********************************************************************
 * @file algorithms.h
 * @brief T3.2 — device algorithm primitives: the LuisaCompute replacement for the
 *        reference backend's CUB calls.
 *
 * The reference CUDA backend has no algorithm layer of its own; device code calls
 * CUB (src/backends/cuda/cuda_tool/cub.h and the call sites below). LuisaCompute
 * ships no algorithm library, so every primitive the ported backend needs is
 * reimplemented here on top of plain LC kernels:
 *
 *   | reference (CUB / cub::)            | here                            |
 *   |------------------------------------|---------------------------------|
 *   | DeviceRadixSort::SortPairs         | sort_index_pairs / sort_pairs_by_key |
 *   | DeviceScan::ExclusiveSum           | exclusive_scan                  |
 *   | DeviceRunLengthEncode::Encode      | run_length_encode               |
 *   | DevicePartition::Flagged           | select_if                       |
 *   | DeviceSelect::If                   | select_if (predicate flavour)    |
 *   | DeviceReduce::Sum/Min/Max          | device_reduce                   |
 *   | cub::WarpReduce (segmented)        | segmented_reduce                |
 *
 * Known call sites (the API is shaped to fit them, not to be generic):
 *   - algorithm/details/matrix_converter.inl
 *       * DeviceRadixSort().SortPairs(ij_hash, sorted_hash, sort_index_in,
 *         sort_index, n)             -> sort_index_pairs<uint64_t>(hash, ...)
 *       * DeviceRadixSort().SortPairs(indices, sorted_indices, segments,
 *         sorted_segments, n)        -> sort_pairs_by_key<int32_t, Value>
 *       * DeviceScan().ExclusiveSum (7x) -> exclusive_scan<int32_t>
 *       * DeviceRunLengthEncode().Encode -> run_length_encode
 *       * DevicePartition().Flagged  -> select_if
 *       * FastSegmentalReduce<64,32>().reduce(counts, values, out)
 *                                    -> segmented_reduce / segmented_reduce_blocks
 *   - algorithm/fast_segmental_reduce.{h,inl} — the algorithmic specification of
 *     `segmented_reduce` (head/`is_cross_warp` flags, warp-level head-segmented
 *     reduction, `out[key] = value` or `atomic_add(out[key], value)`).
 *
 * Contract (cookbook §7.10):
 *  * Host-side functions taking views; all dispatches are batched into the
 *    caller's LStream (which records into the frame CommandList when open),
 *    exactly like `cub::DeviceXxx(stream, ...)`. Not a class hierarchy.
 *  * **Views are launch arguments, never captured.** A captured view becomes a
 *    compile-time bound argument of the compiled shader and keeps pointing at the
 *    pre-resize buffer (cookbook §7.9, scratch/lc_lin_spike (D)) — that is the one
 *    way to write a kernel that passes while touching the wrong buffer. Every
 *    kernel below takes its buffers as parameters and every call site passes
 *    `view()`s at dispatch time.
 *  * No host control flow on device data (RULE M2): kernel loop bounds are host
 *    values or kernel scalar arguments, and data-dependent decisions are
 *    predicated (`$if` around stores, `select`/`min`/`max` around values).
 *  * Every element type actually used is instantiated explicitly in algorithms.cpp
 *    (the .cpp holds ~40 kernel instantiations); an unsupported type is a
 *    compile-time error with a readable message (see the *_type traits).
 *  * Device-bounded `$for` loops (a kernel scalar argument bound) are used for the
 *    flat-block component loops; verified on the GPU by scratch/t32_probe (D).
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/atomic.h>
#include <luisa_device/buffer.h>
#include <luisa_device/builtins.h>
#include <luisa_device/export.h>
#include <luisa_device/kernel.h>
#include <luisa_device/linear/containers.h>
#include <luisa_device/linear/views.h>
#include <luisa_device/math/math.h>
#include <luisa_device/stream.h>
#include <luisa_device/view.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace uipc::backend::luisa::luisa_device::linalg
{
namespace lmath = ::uipc::backend::luisa::luisa_device::math;
/// The device layer, reached explicitly: inside `linalg` the enclosing namespace
/// chain also contains `uipc::backend::luisa`, so a bare `luisa::...` or
/// `luisa_device::...` would be ambiguous/unfindable (see luisa_device/compat.h).
namespace lud = ::uipc::backend::luisa::luisa_device;

// ---------------------------------------------------------------------------
//  Supported element types
// ---------------------------------------------------------------------------
namespace detail
{
template <typename T>
inline constexpr bool is_arithmetic_scalar_v =
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>
    || std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>
    || std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
inline constexpr bool is_integral_scalar_v =
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>
    || std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>;

template <typename T>
inline constexpr bool is_radix_key_v = is_integral_scalar_v<T>;

// Reduction / selection ops. Host and device both instantiate them (cookbook §7.4
// host/device equivalence by construction): DSL scalars go through the numerics
// layer's trampolines (RULE M1 — `linalg` declares its own operator= / += / -=
// so unqualified scalar arithmetic inside this namespace must not rely on lookup
// reaching the global DSL operators).
template <typename T>
struct Plus
{
    static_assert(is_arithmetic_scalar_v<T>);
    static constexpr T identity() noexcept { return T{0}; }
    [[nodiscard]] T operator()(const T& a, const T& b) const noexcept
    {
        return lmath::detail::add(a, b);
    }
};

template <typename T>
struct Multiply
{
    static_assert(is_arithmetic_scalar_v<T>);
    static constexpr T identity() noexcept { return T{1}; }
    [[nodiscard]] T operator()(const T& a, const T& b) const noexcept
    {
        return lmath::detail::mul(a, b);
    }
};

template <typename T>
struct Min
{
    static_assert(is_arithmetic_scalar_v<T>);
    static constexpr T identity() noexcept { return std::numeric_limits<T>::max(); }
    [[nodiscard]] T operator()(const T& a, const T& b) const noexcept
    {
        if constexpr(lmath::is_dsl_value_v<T>)
            return lcrt::min(a, b);
        else
            return a < b ? a : b;
    }
};

template <typename T>
struct Max
{
    static_assert(is_arithmetic_scalar_v<T>);
    static constexpr T identity() noexcept { return std::numeric_limits<T>::lowest(); }
    [[nodiscard]] T operator()(const T& a, const T& b) const noexcept
    {
        if constexpr(lmath::is_dsl_value_v<T>)
            return lcrt::max(a, b);
        else
            return a > b ? a : b;
    }
};
}// namespace detail

// ---------------------------------------------------------------------------
//  Scratch buffers
// ---------------------------------------------------------------------------
/**
 * @brief Reusable device temporaries for `exclusive_scan` / `device_reduce`.
 *
 * One buffer per element type: a primitive carves a single slice per type, so a
 * later `require<T>` can never invalidate a slice another part of the same
 * operation is still reading. (If a primitive ever needs two slices of the same
 * type it must carve them in one call — see the `2n` carves in the .cpp.)
 */
struct ScanScratch
{
    LDeviceBuffer<std::int32_t>  i32;
    LDeviceBuffer<std::uint32_t> u32;
    LDeviceBuffer<std::int64_t>  i64;
    LDeviceBuffer<std::uint64_t> u64;
    LDeviceBuffer<float>         f32;
    LDeviceBuffer<double>        f64;

    /// Allocations displaced by a growth, kept alive for the commands that still
    /// reference them (see `carve`): never freed while this scratch lives.
    std::vector<std::shared_ptr<void>> retired;

    template <typename T>
    [[nodiscard]] LBufferView<T> require(size_t n);

    void clear() noexcept
    {
        i32.clear();
        u32.clear();
        i64.clear();
        u64.clear();
        f32.clear();
        f64.clear();
        retired.clear();
    }
};

/// Temporaries for `select_if` / `run_length_encode` (all int work buffers).
struct CompactScratch
{
    LDeviceBuffer<std::int32_t>  i32;         ///< flags + offsets, carved as 2n
    LDeviceBuffer<std::int32_t>  pred_flags;  ///< n, the predicate flavour's flags
    LDeviceBuffer<std::uint32_t> u32;         ///< scan sums
    ScanScratch                  scan;        ///< the flag scan's own sums

    std::vector<std::shared_ptr<void>> retired;

    void clear() noexcept
    {
        i32.clear();
        pred_flags.clear();
        u32.clear();
        scan.clear();
        retired.clear();
    }
};

/**
 * @brief Temporaries for the radix sort.
 *
 * `perm` is the sort's *internal* identity/ping-pong pair (n + n) — a caller that
 * passes its own scratch must not pass a view of it as `perm_out` (the last pass
 * would read and write the same buffer). `aux` is the extra n-element index array
 * `sort_keys` / `sort_pairs_by_key` use for the permutation they hand to the
 * gather, so it can never collide with the internal pair.
 */
struct SortScratch
{
    LDeviceBuffer<std::uint32_t> histogram;   ///< kRadixDigits * block_count
    LDeviceBuffer<std::int32_t>  ranks;       ///< n (intra-block stable ranks)
    LDeviceBuffer<std::int32_t>  perm;        ///< 2n (identity + ping-pong)
    LDeviceBuffer<std::int32_t>  aux;         ///< n (caller-side permutation)
    LDeviceBuffer<std::int32_t>  keys_i32;    ///< 2n key ping-pong, by key type
    LDeviceBuffer<std::uint32_t> keys_u32;    ///< 2n
    LDeviceBuffer<std::int64_t>  keys_i64;    ///< 2n
    LDeviceBuffer<std::uint64_t> keys_u64;    ///< 2n
    ScanScratch                  scan;        ///< the histogram scan's own sums

    std::vector<std::shared_ptr<void>> retired;

    void clear() noexcept
    {
        histogram.clear();
        ranks.clear();
        perm.clear();
        aux.clear();
        keys_i32.clear();
        keys_u32.clear();
        keys_i64.clear();
        keys_u64.clear();
        retired.clear();
        scan.clear();
    }
};

namespace detail
{
/**
 * @brief Grow-if-needed carve: keeps `size >= n` and returns the first n elements.
 *
 * Growth must not invalidate a buffer that an already-recorded command still
 * references (LC's CommandList stores raw handles and the frame may not be
 * committed yet), so instead of letting LDeviceBuffer::reserve drop the old
 * allocation, the old LC buffer is moved into `retired` — where it stays alive for
 * as long as the scratch object does — and a fresh, larger allocation takes over.
 * Every scratch struct carries its own `retired` list for this.
 */
template <typename T>
[[nodiscard]] inline LBufferView<T> carve(LDeviceBuffer<T>& buffer,
                                          std::vector<std::shared_ptr<void>>& retired, size_t n)
{
    if(n == 0u)
    {
        return {};
    }
    if(buffer.size() < n)
    {
        if(buffer.capacity() < n)
        {
            if(buffer.capacity() != 0u && buffer.size() != 0u)
                retired.push_back(buffer.view().base());  // keep the recorded handles valid
            buffer.reset();
            // 1.5x headroom: the next frame of a similar size does not grow again.
            buffer.reserve(n + n / 2u);
        }
        // Expose the whole capacity once: the value-initialisation of the newly
        // exposed tail is then amortised O(capacity) over the growths.
        buffer.resize(buffer.capacity());
    }
    return buffer.view(0u, n);
}

/**
 * @brief Compile-once kernel cache that is **deliberately never destroyed**.
 *
 * A `lcrt::Shader` held by a function-local static is destroyed during the process's
 * static-destruction phase — i.e. after the test harness's explicit
 * `Runtime::shutdown()`, whose whole purpose is that no LC resource is torn down
 * during DLL detach. Destroying a shader whose device is already gone aborts the
 * process (0xC0000409 STATUS_STACK_BUFFER_OVERRUN), which is exactly the teardown
 * trap the cookbook documents. Compiled shaders are per-process singletons with no
 * destructor work worth doing, so they are leaked on purpose: the cache is a
 * `Shader&` bound to a heap object.
 */
template <typename... Args, typename Def>
[[nodiscard]] inline lcrt::Shader<1u, Args...>& cached_kernel(Def&& def)
{
    static auto* shader =
        new lcrt::Shader<1u, Args...>(lud::compile1<Args...>(std::forward<Def>(def)));
    return *shader;
}

/// The temporary set of a primitive: the caller's, or one cached on the stream so
/// that it outlives an open frame CommandList (see LStream::cached_object).
template <typename Scratch>
[[nodiscard]] inline Scratch& default_scratch(LStream& stream)
{
    static const char tag = 0;  // one distinct address per Scratch instantiation
    return *static_cast<Scratch*>(stream.cached_object(
        &tag, []() -> void* { return new Scratch{}; }, [](void* p) { delete static_cast<Scratch*>(p); }));
}
}// namespace detail

template <typename T>
[[nodiscard]] LBufferView<T> ScanScratch::require(size_t n)
{
    static_assert(detail::is_arithmetic_scalar_v<T>,
                  "luisa_device::linalg algorithm scratch: unsupported element type "
                  "(supported: int32_t, uint32_t, int64_t, uint64_t, float, double)");
    if constexpr(std::is_same_v<T, std::int32_t>)
        return detail::carve(i32, retired, n);
    else if constexpr(std::is_same_v<T, std::uint32_t>)
        return detail::carve(u32, retired, n);
    else if constexpr(std::is_same_v<T, std::int64_t>)
        return detail::carve(i64, retired, n);
    else if constexpr(std::is_same_v<T, std::uint64_t>)
        return detail::carve(u64, retired, n);
    else if constexpr(std::is_same_v<T, float>)
        return detail::carve(f32, retired, n);
    else
        return detail::carve(f64, retired, n);
}

// ---------------------------------------------------------------------------
//  Tuning constants (host values; every kernel loop bound derives from these)
// ---------------------------------------------------------------------------
namespace detail
{
/// Elements per block in the scan/reduce kernels.
inline constexpr uint32_t kScanBlockSize = 256u;
/// Threads per block in a radix pass (16 warps -> 16*kRadixDigits shared uints).
inline constexpr uint32_t kRadixBlockSize = 512u;
/// Digits (buckets) per radix pass: 8 bits.
inline constexpr uint32_t kRadixDigitBits = 8u;
inline constexpr uint32_t kRadixDigits    = 1u << kRadixDigitBits;
constexpr uint32_t        warp_size_v     = 32u;
}// namespace detail

// ---------------------------------------------------------------------------
//  detail::*_impl — the type-dispatched, explicitly instantiated implementations
//  (defined in algorithms.cpp; one overload set per supported element type)
// ---------------------------------------------------------------------------
namespace detail
{
// ---- exclusive scan -------------------------------------------------------
UIPC_LUISA_DEVICE_API void exclusive_scan_impl(LStream& stream, const LCBufferView<std::int32_t>& in,
                         const LBufferView<std::int32_t>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void exclusive_scan_impl(LStream& stream, const LCBufferView<std::uint32_t>& in,
                         const LBufferView<std::uint32_t>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void exclusive_scan_impl(LStream& stream, const LCBufferView<std::int64_t>& in,
                         const LBufferView<std::int64_t>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void exclusive_scan_impl(LStream& stream, const LCBufferView<std::uint64_t>& in,
                         const LBufferView<std::uint64_t>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void exclusive_scan_impl(LStream& stream, const LCBufferView<float>& in,
                         const LBufferView<float>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void exclusive_scan_impl(LStream& stream, const LCBufferView<double>& in,
                         const LBufferView<double>& out, size_t n, ScanScratch& scratch);

// ---- device reduce --------------------------------------------------------
enum class ReduceKind : std::uint32_t
{
    Sum = 0u,
    Min = 1u,
    Max = 2u
};

UIPC_LUISA_DEVICE_API void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<std::int32_t>& in,
                        const LBufferView<std::int32_t>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<std::uint32_t>& in,
                        const LBufferView<std::uint32_t>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<std::int64_t>& in,
                        const LBufferView<std::int64_t>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<std::uint64_t>& in,
                        const LBufferView<std::uint64_t>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<float>& in,
                        const LBufferView<float>& out, size_t n, ScanScratch& scratch);
UIPC_LUISA_DEVICE_API void device_reduce_impl(ReduceKind kind, LStream& stream, const LCBufferView<double>& in,
                        const LBufferView<double>& out, size_t n, ScanScratch& scratch);

// ---- select_if / compaction ----------------------------------------------
UIPC_LUISA_DEVICE_API void select_if_impl(LStream& stream, const LCBufferView<std::int32_t>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<std::int32_t>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch);
UIPC_LUISA_DEVICE_API void select_if_impl(LStream& stream, const LCBufferView<std::uint32_t>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<std::uint32_t>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch);
UIPC_LUISA_DEVICE_API void select_if_impl(LStream& stream, const LCBufferView<std::int64_t>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<std::int64_t>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch);
UIPC_LUISA_DEVICE_API void select_if_impl(LStream& stream, const LCBufferView<std::uint64_t>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<std::uint64_t>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch);
UIPC_LUISA_DEVICE_API void select_if_impl(LStream& stream, const LCBufferView<float>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<float>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch);
UIPC_LUISA_DEVICE_API void select_if_impl(LStream& stream, const LCBufferView<double>& in,
                    const LCBufferView<std::int32_t>& flags, const LBufferView<double>& out,
                    const LBufferView<std::uint32_t>& out_count, size_t n, bool keep_when_zero,
                    CompactScratch& scratch);

// ---- run-length encode ---------------------------------------------------
UIPC_LUISA_DEVICE_API void run_length_encode_impl(LStream& stream, const LCBufferView<std::int32_t>& keys,
                            const LBufferView<std::int32_t>& unique_out,
                            const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch);
UIPC_LUISA_DEVICE_API void run_length_encode_impl(LStream& stream, const LCBufferView<std::uint32_t>& keys,
                            const LBufferView<std::uint32_t>& unique_out,
                            const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch);
UIPC_LUISA_DEVICE_API void run_length_encode_impl(LStream& stream, const LCBufferView<std::int64_t>& keys,
                            const LBufferView<std::int64_t>& unique_out,
                            const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch);
UIPC_LUISA_DEVICE_API void run_length_encode_impl(LStream& stream, const LCBufferView<std::uint64_t>& keys,
                            const LBufferView<std::uint64_t>& unique_out,
                            const LBufferView<std::int32_t>& starts_out,
                            const LBufferView<std::int32_t>& counts_out,
                            const LBufferView<std::uint32_t>& run_count_out, size_t n,
                            CompactScratch& scratch);

// ---- segmented reduce ----------------------------------------------------
UIPC_LUISA_DEVICE_API void segmented_reduce_impl(LStream& stream,
                           const LCBufferView<std::int32_t>& keys_sorted,
                           const LCBufferView<double>& values, const LBufferView<double>& out,
                           size_t out_size, size_t n, uint32_t components);
UIPC_LUISA_DEVICE_API void segmented_reduce_impl(LStream& stream,
                           const LCBufferView<std::uint64_t>& keys_sorted,
                           const LCBufferView<double>& values, const LBufferView<double>& out,
                           size_t out_size, size_t n, uint32_t components);

// ---- radix sort ----------------------------------------------------------
UIPC_LUISA_DEVICE_API void sort_index_pairs_impl(LStream& stream, const LCBufferView<std::int32_t>& keys_in,
                           const LBufferView<std::int32_t>& keys_out,
                           const LBufferView<std::int32_t>& perm_out, size_t n, uint32_t key_bits,
                           SortScratch& scratch);
UIPC_LUISA_DEVICE_API void sort_index_pairs_impl(LStream& stream, const LCBufferView<std::uint32_t>& keys_in,
                           const LBufferView<std::uint32_t>& keys_out,
                           const LBufferView<std::int32_t>& perm_out, size_t n, uint32_t key_bits,
                           SortScratch& scratch);
UIPC_LUISA_DEVICE_API void sort_index_pairs_impl(LStream& stream, const LCBufferView<std::int64_t>& keys_in,
                           const LBufferView<std::int64_t>& keys_out,
                           const LBufferView<std::int32_t>& perm_out, size_t n, uint32_t key_bits,
                           SortScratch& scratch);
UIPC_LUISA_DEVICE_API void sort_index_pairs_impl(LStream& stream, const LCBufferView<std::uint64_t>& keys_in,
                           const LBufferView<std::uint64_t>& keys_out,
                           const LBufferView<std::int32_t>& perm_out, size_t n, uint32_t key_bits,
                           SortScratch& scratch);
}// namespace detail

// ---------------------------------------------------------------------------
//  exclusive_scan — cub::DeviceScan::ExclusiveSum
// ---------------------------------------------------------------------------
/**
 * @brief `out[i] = sum_{j<i} in[j]`, `out[0] = 0`.
 *
 * Multi-block: a warp/block scan of every 256-element tile, a scan of the block
 * sums (recursively, until one block remains) and an add-back pass — five
 * dispatches for 1e6 elements. Exact for integer types (addition is associative
 * and exact); for float/double the association order is the fixed reduction tree
 * of the partition, so the result is deterministic but is *not* the reference
 * CUB's (which splits at a different granularity) — compare float scans with a
 * tolerance, not bit-for-bit.
 *
 * `in` and `out` may be the same buffer (every element is read before the same
 * index is written).
 *
 * @param n number of elements; 0 is a no-op.
 */
template <typename T>
void exclusive_scan(LStream& stream, const LCBufferView<T>& in, const LBufferView<T>& out,
                    size_t n, ScanScratch* scratch = nullptr)
{
    static_assert(detail::is_arithmetic_scalar_v<T>,
                  "luisa_device::linalg::exclusive_scan: unsupported element type "
                  "(supported: int32_t, uint32_t, int64_t, uint64_t, float, double)");
    detail::exclusive_scan_impl(stream, in, out, n, scratch ? *scratch : detail::default_scratch<ScanScratch>(stream));
}

// ---------------------------------------------------------------------------
//  device_reduce — cub::DeviceReduce::Sum/Min/Max
// ---------------------------------------------------------------------------
/**
 * @brief Full-buffer reduction into `out[0]`.
 *
 * Two passes: a per-block tree reduction into `block_partials`, then a single
 * block that folds the partials in a fixed order. **Deterministic** (no atomics,
 * no arrival order), which is what the reference's `DeviceReduce` wrappers are
 * compared against — but the association order differs from CUB's, so float
 * results agree to rounding rather than bit-for-bit.
 *
 * `n == 0` writes the identity (`0`, `+max`/`+inf`, `lowest`/`-inf`).
 */
template <typename T>
void device_reduce(detail::ReduceKind kind, LStream& stream, const LCBufferView<T>& in,
                   const LBufferView<T>& out, size_t n, ScanScratch* scratch = nullptr)
{
    static_assert(detail::is_arithmetic_scalar_v<T>,
                  "luisa_device::linalg::device_reduce: unsupported element type");
    detail::device_reduce_impl(kind, stream, in, out, n, scratch ? *scratch : detail::default_scratch<ScanScratch>(stream));
}

template <typename T>
void device_reduce_sum(LStream& stream, const LCBufferView<T>& in, const LBufferView<T>& out,
                       size_t n, ScanScratch* scratch = nullptr)
{
    device_reduce(detail::ReduceKind::Sum, stream, in, out, n, scratch);
}

template <typename T>
void device_reduce_min(LStream& stream, const LCBufferView<T>& in, const LBufferView<T>& out,
                       size_t n, ScanScratch* scratch = nullptr)
{
    device_reduce(detail::ReduceKind::Min, stream, in, out, n, scratch);
}

template <typename T>
void device_reduce_max(LStream& stream, const LCBufferView<T>& in, const LBufferView<T>& out,
                       size_t n, ScanScratch* scratch = nullptr)
{
    device_reduce(detail::ReduceKind::Max, stream, in, out, n, scratch);
}

// ---------------------------------------------------------------------------
//  select_if — cub::DeviceSelect::If / cub::DevicePartition::Flagged
// ---------------------------------------------------------------------------
/**
 * @brief Stable stream compaction: keep the elements whose flag is non-zero
 *        (`keep_when_zero == false`, the default) or zero, in input order.
 *
 * An exclusive scan of the flags gives every kept element its destination, so the
 * result order is exactly the input order (this is what the reference's
 * `DevicePartition::Flagged` guarantees and what `ge2sym` relies on).
 * `out_count[0]` receives the number of kept elements (a *device* value; read it
 * back with `LDeviceVar<uint32_t>::copy_to()`).
 *
 * `n == 0` writes `out_count[0] = 0` and dispatches nothing else.
 */
template <typename T>
void select_if(LStream& stream, const LCBufferView<T>& in, const LCBufferView<std::int32_t>& flags,
               const LBufferView<T>& out, const LBufferView<std::uint32_t>& out_count, size_t n,
               CompactScratch* scratch = nullptr, bool keep_when_zero = false)
{
    static_assert(detail::is_arithmetic_scalar_v<T>,
                  "luisa_device::linalg::select_if: unsupported element type");
    detail::select_if_impl(stream, in, flags, out, out_count, n, keep_when_zero,
                           scratch ? *scratch : detail::default_scratch<CompactScratch>(stream));
}

/**
 * @brief Predicate flavour of `select_if` (cub::DeviceSelect::If).
 *
 * `Pred` is a *type* with a static `apply`, not a functor object:
 * @code
 * struct is_odd {
 *     static lcrt::Expr<bool> apply(lcrt::Expr<std::int32_t> v) { return v % 2 != 0; }
 * };
 * select_if_predicated<std::int32_t, is_odd>(stream, in, out, count, n);
 * @endcode
 * The reason is the kernel cache: the flag kernel must compile *once* per
 * (element type, predicate) and a captured functor object could not live in a
 * function-local static (it would also become a bound argument — the very trap the
 * cookbook §7.9 warns about). A stateless predicate type keeps the cache honest;
 * a parameterised predicate encodes its parameters in the type.
 */
template <typename T, typename Pred>
void select_if_predicated(LStream& stream, const LCBufferView<T>& in, const LBufferView<T>& out,
                          const LBufferView<std::uint32_t>& out_count, size_t n,
                          CompactScratch* scratch = nullptr)
{
    static_assert(detail::is_arithmetic_scalar_v<T>,
                  "luisa_device::linalg::select_if_predicated: unsupported element type");
    static_assert(requires(lcrt::Expr<T> v) { Pred::apply(v); },
                  "select_if_predicated: the predicate type needs a static "
                  "apply(lcrt::Expr<T>) returning an Expr<bool>");
    auto&          sc = scratch ? *scratch : detail::default_scratch<CompactScratch>(stream);
    static auto& k = detail::cached_kernel<lcrt::Buffer<T>, lcrt::Buffer<std::int32_t>, uint32_t>(
        [](lcrt::BufferVar<T> v, lcrt::BufferVar<std::int32_t> f, lcrt::Var<uint32_t> N) noexcept {
            auto i = lcrt::dispatch_x();
            $if (i < N)
            {
                auto p = Pred::apply(v->read(i));
                f->write(i, lcrt::select(0, 1, p));  // p ? 1 : 0
            };
        });
    auto flags = detail::carve(sc.pred_flags, sc.retired, n == 0u ? 1u : n);
    if(n != 0u)
    {
        stream.record(k(in, flags, static_cast<uint32_t>(n))
                          .dispatch(static_cast<uint32_t>((n + detail::kScanBlockSize - 1u)
                                                          / detail::kScanBlockSize
                                                          * detail::kScanBlockSize)));
    }
    detail::select_if_impl(stream, in, flags.subview(0u, n), out, out_count, n, false, sc);
}

// ---------------------------------------------------------------------------
//  select_if for *struct* element types (T7.2b) — cub::DeviceSelect::If over the
//  port's pair elements (LVector2i / LVector3i / LVector4i)
// ---------------------------------------------------------------------------
/**
 * @brief Temporaries for `compact_by_flags` / `select_valid_pairs`.
 *
 * The scalar `select_if`/`select_if_predicated` spell their exclusive scan through
 * `CompactScratch`; the struct-element path needs its own offsets array (the scan of
 * the flags) because a `ScanScratch` already owns the same int32 buffer for its own
 * partial sums, so carving the offsets from it would alias the scan.
 */
struct StructSelectScratch
{
    LDeviceBuffer<std::int32_t> offsets;  ///< n (the exclusive scan of the flags)
    LDeviceBuffer<std::int32_t> flags;    ///< n (only `select_valid_pairs` needs it)
    ScanScratch                 scan;     ///< the scan's own partial sums
    std::vector<std::shared_ptr<void>> retired;

    void clear() noexcept
    {
        offsets.clear();
        flags.clear();
        scan.clear();
        retired.clear();
    }
};

namespace detail
{
/// `flags[i] = (in[i].data[0] != -1) ? 1 : 0` — the reference's four
/// `filter_active_{PP,PE,PT,EE}_pred` (`pair(0) != -1`) are literally this one test.
template <typename T>
auto& component0_valid_flag_kernel()
{
    static auto& k = cached_kernel<lcrt::Buffer<T>, lcrt::Buffer<std::int32_t>,
                                   std::uint32_t>(
        [](lcrt::BufferVar<T> in, lcrt::BufferVar<std::int32_t> flags,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto e = in->read(i);
                flags->write(i,
                             lcrt::select(std::int32_t{0},
                                          std::int32_t{1},
                                          e.data[0] != std::int32_t{-1}));
            };
        });
    return k;
}

/// The stable scatter of `compact_by_flags`: `out[offsets[i]] = in[i]` for every
/// flagged element, and the last thread publishes the total.
template <typename T>
auto& compact_scatter_kernel()
{
    static auto& k = cached_kernel<lcrt::Buffer<T>, lcrt::Buffer<std::int32_t>,
                                   lcrt::Buffer<std::int32_t>, lcrt::Buffer<T>,
                                   lcrt::Buffer<std::uint32_t>, std::uint32_t>(
        [](lcrt::BufferVar<T> in, lcrt::BufferVar<std::int32_t> flags,
           lcrt::BufferVar<std::int32_t> offsets, lcrt::BufferVar<T> out,
           lcrt::BufferVar<std::uint32_t> out_count, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kScanBlockSize);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                $if(flags->read(i) != 0)
                {
                    out->write(lcrt::cast<std::uint32_t>(offsets->read(i)), in->read(i));
                };
                // the total: the exclusive scan's last offset plus the last flag
                $if(i + 1u == n)
                {
                    out_count->write(
                        0u,
                        lcrt::cast<std::uint32_t>(offsets->read(i) + flags->read(i)));
                };
            };
        });
    return k;
}
}  // namespace detail

/**
 * @brief Write the "component 0 is not the -1 sentinel" flag of every element.
 *
 * This is the port's spelling of `cub::DeviceSelect::If`'s predicate kernel for the
 * port's *pair* element types (which `select_if_predicated` cannot serve: its flag
 * kernel and `select_if_impl` are instantiated for arithmetic scalars only).  All four
 * reference predicates this batch needs (`filter_active_PP_pred`, `..._PE_pred`,
 * `..._PT_pred`, `..._EE_pred`) are the same `pair(0) != -1` test.
 */
template <typename T>
void flag_component0_valid(LStream& stream, const LCBufferView<T>& in,
                           const LBufferView<std::int32_t>& flags, size_t n)
{
    if(n == 0u)
        return;
    stream.record(detail::component0_valid_flag_kernel<T>()(
                      in, flags, static_cast<std::uint32_t>(n))
                      .dispatch(static_cast<std::uint32_t>(n)));
}

/**
 * @brief Stable stream compaction of a *struct* element type given an int32 flags array.
 *
 * `out[i] = in[j]` for the i-th `j` (in input order) with `flags[j] != 0`, and
 * `out_count[0]` receives the number kept -- the semantics of
 * `cub::DeviceSelect::If(in, out, count, n, pred)` with `pred = flags[j] != 0`.
 * `n == 0` writes `out_count[0] = 0` and dispatches nothing.
 *
 * The exclusive scan of the flags is the same primitive `select_if` uses; the
 * scatter is a second kernel, so a caller that needs the flags for another purpose
 * can call `flag_component0_valid` (or write its own flag kernel) first.
 */
template <typename T>
void compact_by_flags(LStream& stream, const LCBufferView<T>& in,
                      const LCBufferView<std::int32_t>& flags, const LBufferView<T>& out,
                      const LBufferView<std::uint32_t>& out_count, size_t n,
                      StructSelectScratch* scratch = nullptr)
{
    auto& sc = scratch ? *scratch : detail::default_scratch<StructSelectScratch>(stream);
    auto  offsets = detail::carve(sc.offsets, sc.retired, n);
    if(n == 0u)
    {
        out_count.fill(std::uint32_t{0});
        return;
    }
    exclusive_scan(stream, flags, offsets, n, &sc.scan);
    stream.record(detail::compact_scatter_kernel<T>()(
                      in, flags, offsets, out, out_count, static_cast<std::uint32_t>(n))
                      .dispatch(static_cast<std::uint32_t>(n)));
}

/**
 * @brief `DeviceSelect::If(temp.data(), out.data(), count.data(), n, {pair(0) != -1})`.
 *
 * The reference's four filter-active selections in one call: flag the valid
 * (non-sentinel) pairs and compact them stably into `out`.
 */
template <typename T>
void select_valid_pairs(LStream& stream, const LCBufferView<T>& in, const LBufferView<T>& out,
                        const LBufferView<std::uint32_t>& out_count, size_t n,
                        StructSelectScratch* scratch = nullptr)
{
    auto& sc = scratch ? *scratch : detail::default_scratch<StructSelectScratch>(stream);
    if(n == 0u)
    {
        out_count.fill(std::uint32_t{0});
        return;
    }
    auto flags = detail::carve(sc.flags, sc.retired, n);
    flag_component0_valid<T>(stream, in, flags, n);
    compact_by_flags<T>(stream, in, flags, out, out_count, n, &sc);
}

// ---------------------------------------------------------------------------
//  run_length_encode — cub::DeviceRunLengthEncode::Encode
// ---------------------------------------------------------------------------
/**
 * @brief Run-length encode an **ascending** key array.
 *
 * Writes, for each run j in [0, run_count):
 *   `unique_out[j]` = the run's key,
 *   `starts_out[j]` = the index of the run's first element,
 *   `counts_out[j]` = the run's length.
 * `run_count_out[0]` receives the number of runs. Outputs are *not* sized by the
 * primitive: `unique_out`/`starts_out`/`counts_out` must have room for n
 * elements (a run count of n is possible: all keys distinct). The reference
 * resizes them from the host after reading the count back (matrix_converter.inl:
 * `unique_ij_pairs.resize_discard(h_count)`), which needs a synchronising
 * `run_count_out` read-back — the primitive deliberately does not do that itself.
 *
 * `n == 0` writes `run_count_out[0] = 0`.
 */
template <typename Key>
void run_length_encode(LStream& stream, const LCBufferView<Key>& keys_sorted,
                       const LBufferView<Key>& unique_out, const LBufferView<std::int32_t>& starts_out,
                       const LBufferView<std::int32_t>& counts_out,
                       const LBufferView<std::uint32_t>& run_count_out, size_t n,
                       CompactScratch* scratch = nullptr)
{
    static_assert(detail::is_radix_key_v<Key>,
                  "luisa_device::linalg::run_length_encode: unsupported key type "
                  "(supported: int32_t, uint32_t, int64_t, uint64_t)");
    detail::run_length_encode_impl(stream, keys_sorted, unique_out, starts_out, counts_out,
                                   run_count_out, n, scratch ? *scratch : detail::default_scratch<CompactScratch>(stream));
}

// ---------------------------------------------------------------------------
//  segmented_reduce — cub::WarpReduce head-segmented reduction
// (algorithm/fast_segmental_reduce.inl is the specification)
// ---------------------------------------------------------------------------
/**
 * @brief Combine adjacent equal keys: `out[key] = op(out[key], values[i])`.
 *
 * Faithful to `FastSegmentalReduce::reduce(in_size, out, get_key, get_value, op)`:
 *  * `keys_sorted` must be ascending (equal keys adjacent); the input is the same
 *    sorted COO key array the converter already produced.
 *  * `out` is indexed **by key value**, not compacted: `out_size` must be at least
 *    `max(key) + 1`, and the primitive *zero-fills* (identity-fills) it first,
 *    exactly like the reference (`BufferLaunch().fill(out, 0)`).
 *  * A segment that stays inside one warp is written with a plain store; a segment
 *    that crosses a warp/block boundary is accumulated with `atomic_add` on the
 *    *head* warp's partial, exactly like the reference's `is_cross_warp` path.
 *    Consequence (R6): the double `atomic_add` order is not reproducible, so a
 *    segment spanning more than one warp can differ in the last bits between runs
 *    and from the CPU reference. Integer / exactly-representable values are exact,
 *    and the tests compare the float case with a tolerance.
 *  * Keys outside `[0, out_size)` are dropped (the write is predicated). CUB's
 *    version would corrupt memory instead; the bound is documented, not assumed.
 *  * The op is a template parameter (the reference's ReduceOp shape) and the kernel
 *    is written as a generic fold, but only `detail::Plus` is instantiated: the
 *    cross-warp branch is an *atomic* combine and CUDA provides no
 *    atomicMin/atomicMax for double. Any other op needs a deterministic two-pass
 *    fold instead (a documented follow-up, not a silent limitation: the
 *    static_assert names it).
 *
 * @tparam Components number of contiguous scalars per element (the T3.1 flat
 *         block layout: 1 for a scalar, N for a N-vector, M*N for an MxN block).
 *         A runtime component loop is used, so any value is supported.
 */
template <typename Key, typename Value = double, typename Op = detail::Plus<Value>>
void segmented_reduce(LStream& stream, const LCBufferView<Key>& keys_sorted,
                      const LCBufferView<Value>& values, const LBufferView<Value>& out,
                      size_t out_size, size_t n, Op op = {}, CompactScratch* scratch = nullptr)
{
    static_assert(std::is_same_v<Op, detail::Plus<Value>>,
                  "luisa_device::linalg::segmented_reduce: only detail::Plus is instantiated. The "
                  "op stays a template parameter (the reference's ReduceOp shape) and the kernel is "
                  "written for a generic fold, but a non-plus op would need a *packed* atomic "
                  "combine across warps (the reference's is_cross_warp branch) and CUDA has no "
                  "atomicMin/atomicMax for double -- see the T3.2 report.");
    (void)op;
    detail::segmented_reduce_impl(stream, keys_sorted, values, out, out_size, n, 1u);
}

/**
 * @brief Flat-block flavour of `segmented_reduce` (the reference's
 *        `reduce(CBufferView<Eigen::Matrix<T,M,N>> in, BufferView<Eigen::Matrix<T,M,N>> out)`).
 *
 * T3.1 stores a block as `M*N` contiguous scalars (column-major inside the block),
 * so a "matrix" segmented reduce is this one with `components = M*N`: the reference
 * loops over the block entries and does the warp head-segmented reduction per
 * component, which is what the kernel does (with a device-bounded component loop).
 * `values`/`out` therefore hold `n * components` / `out_size * components` scalars.
 */
template <typename Key, typename Value = double, typename Op = detail::Plus<Value>>
void segmented_reduce_blocks(LStream& stream, const LCBufferView<Key>& keys_sorted,
                             const LCBufferView<Value>& values, const LBufferView<Value>& out,
                             size_t out_size, size_t n, uint32_t components, Op op = {},
                             CompactScratch* scratch = nullptr)
{
    static_assert(std::is_same_v<Key, std::int32_t> || std::is_same_v<Key, std::uint64_t>,
                  "luisa_device::linalg::segmented_reduce_blocks: only int32_t / uint64_t keys");
    static_assert(std::is_same_v<std::remove_cvref_t<Value>, double>,
                  "luisa_device::linalg::segmented_reduce_blocks: only double values");
    static_assert(std::is_same_v<Op, detail::Plus<Value>>,
                  "luisa_device::linalg::segmented_reduce_blocks: only detail::Plus is "
                  "instantiated (see segmented_reduce)");
    (void)op;
    detail::segmented_reduce_impl(stream, keys_sorted, values, out, out_size, n, components);
}

// ---------------------------------------------------------------------------
//  sort_index_pairs / sort_pairs_by_key — cub::DeviceRadixSort::SortPairs
// ---------------------------------------------------------------------------
/**
 * @brief **Stable** ascending sort of (key, index) pairs; the permutation is the
 *        output (`perm[i]` = the input index of the i-th smallest key).
 *
 * Why stability: CUB's radix sort is stable, the reference's matrix converter
 * sorts `(row << 32) | col` hashes and then walks the sorted array to build the
 * BCOO structure and to combine duplicates. Stability makes my sorted `(row,col)`
 * sequence **identical** to CUB's (including the order of equal keys), which is
 * what T3.3's "row-pointer / column-index arrays identical" validation needs. It
 * is guaranteed here by construction: the scatter destination of an element is
 * `(#keys with a smaller digit) + (#equal digits in earlier blocks) +
 * (#equal digits in earlier lanes of the same warp)` — never an atomic ticket.
 *
 * Implementation: multi-pass LSD radix sort, 8 bits (256 buckets) per pass, one
 * pass per byte of `key_bits` (8 passes for a 64-bit hash key, 4 for a 32-bit key).
 * Each pass is 3 host-side steps (cookbook §7.10 has the block-histogram detail):
 *   1. a per-warp shared-memory digit histogram + the stable intra-block rank
 *      (LC has no `__match_any_sync`, so the rank is a 32-iteration shuffle loop),
 *      written transposed as `histogram[digit * block_count + block]`;
 *   2. an `exclusive_scan` of that histogram (one flat scan, 256*block_count
 *      elements) — the digit-major layout makes
 *      `bucket_offset[digit] = P[digit*block_count]` and
 *      `#equal digits in earlier blocks = P[digit*block_count + block] - bucket_offset[digit]`;
 *   3. the scatter, which moves the key and the index in the same store.
 *
 * **Bucket-table bound.** A bucket table over the key *range* (which is exactly
 * what the brief's "per-bucket atomic counts" describes) is only possible when the
 * range is small; the converter's 64-bit `(row<<32)|col` hash has a range of
 * `rows*cols`, so a bucket table cannot exist. The multi-pass radix splits the key
 * into 8-bit digits instead: the scratch is `256 * ceil(n/512)` uints
 * (1 MB for n = 1e6) regardless of the key range, and there is no key-range bound
 * at all for `sort_index_pairs`/`sort_pairs_by_key`. For callers whose keys *are*
 * bounded (row indices, `key < bucket_count <= 256`) `bucket_sort_pairs_by_key`
 * exposes the single-pass bucket sort directly.
 *
 * Keys are sorted by value: unsigned keys naturally, signed keys through the
 * standard sign-bit flip applied to the top digit only.
 *
 * @param key_bits number of significant key bits; 0 = the full width of `Key`
 *        (passes = ceil(key_bits/8)). The caller may pass a smaller value when the
 *        keys are known to be bounded (e.g. row indices < 2^16 -> 2 passes).
 *        Contract: with `key_bits == 8*sizeof(Key)` the keys are sorted **by value**
 *        (signed types include their negatives: the top digit is sign-bit biased);
 *        with a *reduced* `key_bits` the keys must be non-negative and below
 *        `2^key_bits` (no bias is applied, so a key outside the declared range
 *        would be ordered as if it were its low `key_bits` bits).
 * @param keys_out receives the sorted keys; it may alias `keys_in` (in place).
 * @param perm_out receives the permutation (n `int32_t`).
 */
template <typename Key>
void sort_index_pairs(LStream& stream, const LCBufferView<Key>& keys_in,
                      const LBufferView<Key>& keys_out, const LBufferView<std::int32_t>& perm_out,
                      size_t n, uint32_t key_bits = 0u, SortScratch* scratch = nullptr)
{
    static_assert(detail::is_radix_key_v<Key>,
                  "luisa_device::linalg::sort_index_pairs: unsupported key type "
                  "(supported: int32_t, uint32_t, int64_t, uint64_t)");
    static_assert(sizeof(Key) == 4u || sizeof(Key) == 8u, "unsupported key width");
    if(key_bits == 0u)
        key_bits = 8u * static_cast<uint32_t>(sizeof(Key));
    else if(key_bits > 8u * sizeof(Key))
        throw std::invalid_argument{"sort_index_pairs: key_bits exceeds the key width"};
    detail::sort_index_pairs_impl(stream, keys_in, keys_out, perm_out, n, key_bits,
                                  scratch ? *scratch : detail::default_scratch<SortScratch>(stream));
}

/// Sort the keys only (a private permutation is used).
template <typename Key>
void sort_keys(LStream& stream, const LCBufferView<Key>& keys_in, const LBufferView<Key>& keys_out,
               size_t n, uint32_t key_bits = 0u, SortScratch* scratch = nullptr)
{
    auto&       sc   = scratch ? *scratch : detail::default_scratch<SortScratch>(stream);
    auto        perm = detail::carve(sc.aux, sc.retired, n);  // distinct from the internal pair
    sort_index_pairs(stream, keys_in, keys_out, perm, n, key_bits, &sc);
}

/// The CUB `SortPairs` shape: sort the keys and permute an arbitrary payload with
/// them (the payload is moved once, at the end, through the permutation).
template <typename Key, typename Value>
void sort_pairs_by_key(LStream& stream, const LCBufferView<Key>& keys_in,
                       const LBufferView<Key>& keys_out, const LCBufferView<Value>& values_in,
                       const LBufferView<Value>& values_out, size_t n, uint32_t key_bits = 0u,
                       SortScratch* scratch = nullptr);

/// Stable single-pass **bucket sort** for keys known to be in `[0, bucket_count)`.
/// `bucket_count` must be <= 256 (the radix digit width); a larger range must be
/// expressed as a multi-pass `sort_pairs_by_key`. Keys outside the range are not
/// detected (like CUB, the caller owns the precondition) but cannot write out of
/// bounds: the destination is always built from the scan of the digit histogram.
template <typename Key, typename Value>
void bucket_sort_pairs_by_key(LStream& stream, const LCBufferView<Key>& keys_in,
                              const LBufferView<Key>& keys_out, const LCBufferView<Value>& values_in,
                              const LBufferView<Value>& values_out, size_t n,
                              uint32_t bucket_count, SortScratch* scratch = nullptr);

// ---------------------------------------------------------------------------
//  permute_gather — `out[i] = in[perm[i]]` (the matrix converter's "sort the
//  block values" kernel)
// ---------------------------------------------------------------------------
/// Gather `n` elements (strided: `components` contiguous scalars per element).
template <typename T>
void permute_gather_strided(LStream& stream, const LCBufferView<T>& in,
                            const LCBufferView<std::int32_t>& perm, const LBufferView<T>& out,
                            size_t n, uint32_t components = 1u)
{
    if(n == 0u)
        return;
    // Compiled once per element type and *kept alive*: a recorded
    // ShaderDispatchCommand references the Shader resource, so a shader that dies
    // before the frame list is committed leaves a dangling handle behind (the same
    // trap the scratch buffers have — see the report).
    static auto& k = detail::cached_kernel<lcrt::Buffer<T>, lcrt::Buffer<std::int32_t>, lcrt::Buffer<T>,
                                  uint32_t, uint32_t>(
        [](lcrt::BufferVar<T> src, lcrt::BufferVar<std::int32_t> p, lcrt::BufferVar<T> dst,
           lcrt::Var<uint32_t> N, lcrt::Var<uint32_t> comp) noexcept {
        auto i = lcrt::dispatch_x();
        $if (i < N)
        {
            auto pi = lcrt::cast<uint32_t>(p->read(i));
            $for (c, 0u, comp)
            {
                dst->write(i * comp + c, src->read(pi * comp + c));
            };
        };
    });
    stream.record(k(in, perm, out, static_cast<uint32_t>(n), components)
                      .dispatch(static_cast<uint32_t>(n)));
}

template <typename T>
void permute_gather(LStream& stream, const LCBufferView<T>& in, const LCBufferView<std::int32_t>& perm,
                    const LBufferView<T>& out, size_t n)
{
    permute_gather_strided(stream, in, perm, out, n, 1u);
}

// ---------------------------------------------------------------------------
//  sort_pairs_by_key / bucket_sort_pairs_by_key definitions
// ---------------------------------------------------------------------------
template <typename Key, typename Value>
void sort_pairs_by_key(LStream& stream, const LCBufferView<Key>& keys_in,
                       const LBufferView<Key>& keys_out, const LCBufferView<Value>& values_in,
                       const LBufferView<Value>& values_out, size_t n, uint32_t key_bits,
                       SortScratch* scratch)
{
    auto&       sc = scratch ? *scratch : detail::default_scratch<SortScratch>(stream);
    if(values_in.base() == values_out.base())
    {
        throw std::invalid_argument{
            "luisa_device::linalg::sort_pairs_by_key: values_in and values_out must be distinct "
            "buffers (a permutation gather is not an in-place-safe copy)"};
    }
    // `aux` is the caller-side permutation: the sort keeps its own ping-pong pair, so
    // the permutation the gather reads cannot be overwritten by it.
    auto perm = detail::carve(sc.aux, sc.retired, n);
    sort_index_pairs(stream, keys_in, keys_out, perm, n, key_bits, &sc);
    permute_gather(stream, values_in, perm, values_out, n);
}

template <typename Key, typename Value>
void bucket_sort_pairs_by_key(LStream& stream, const LCBufferView<Key>& keys_in,
                              const LBufferView<Key>& keys_out, const LCBufferView<Value>& values_in,
                              const LBufferView<Value>& values_out, size_t n, uint32_t bucket_count,
                              SortScratch* scratch)
{
    if(bucket_count == 0u || bucket_count > detail::kRadixDigits)
    {
        throw std::invalid_argument{
            "luisa_device::linalg::bucket_sort_pairs_by_key: bucket_count must be in [1,256] "
            "(one radix digit); use sort_pairs_by_key for a wider key range"};
    }
    uint32_t bits = 0u;
    while((1u << bits) < bucket_count)
        ++bits;
    sort_pairs_by_key(stream, keys_in, keys_out, values_in, values_out, n, bits ? bits : 1u, scratch);
}
}// namespace uipc::backend::luisa::luisa_device::linalg
