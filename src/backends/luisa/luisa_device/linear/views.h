#pragma once
/********************************************************************
 * @file views.h
 * @brief T3.1 — linear-system *views*: the LuisaCompute port of
 *        src/backends/cuda/cuda_tool/linear_system/views.h.
 *
 * Scope
 * -----
 * `TripletMatrixViewT` / `DoubletVectorViewT` / `DenseVectorViewT` (plus their
 * BCOO/BCOOVector aliases and the `*Viewer` spellings the reference keeps for
 * muda parity) are re-implemented on top of the luisa_device foundation
 * (`LBufferView`/`LCBufferView`) with the accessor surface the 25+ referencing
 * files use.  Two view types the reference does not have are added because the
 * task statement asks for them: `BSRMatrixViewT` (a BSR row walk) and
 * `DenseMatrixViewT` (the reference only declares `DeviceDenseMatrix` as a
 * debug-dump buffer and ports no DenseMatrix view).
 *
 * NOT ported here (deliberately — see linear.h):
 *   * `details::LinearReductionStorage` / `LinearSystemContext` (dot/norm
 *     reduction context — belongs to the T3.4 blas/spmv work),
 *   * the cusparse-descriptor carriers (`COOMatrixViewT`, `COOVectorViewBase`),
 *   * raw element pointers (`data()`, `values().data()`, ...): LC device memory
 *     is reached through buffer *handles*, never through host pointers.  Use the
 *     exposed handles as kernel arguments, or `block()/value_at()` inside a
 *     kernel.
 *
 * Block storage (the central design decision — see cookbook §7.9)
 * ---------------------------------------------------------------
 * The reference stores a block as `Eigen::Matrix<T,M,N>` inside a muda
 * DeviceVector (raw bytes on the CUDA side).  Here the values live in a
 * **flat scalar buffer** of `nnz * M * N` elements with the block stored
 * column-major:
 *
 *     index(i, local_r, local_c) = i * M * N + local_c * M + local_r
 *
 * so the bytes of block `i` are bit-identical to `Eigen::Matrix<T,M,N>` /
 * `math::LMat<T,M,N>`.  Scalar (M == N == 1) entries stay a plain `T` path with
 * no indirection, mirroring the reference's
 * `std::conditional_t<(M==1&&N==1), T, Eigen::Matrix<T,M,N>>`.
 * scratch/lc_lin_spike measured the alternatives: `lcrt::Matrix<double,3>` is a
 * legal buffer element but its stride is 96 B (columns are alignas(16):
 * +0/+32/+64) against Eigen's 72 B, and a `LUISA_STRUCT` POD element needs one
 * registered type per (M,N) shape — neither can back the generic `<T,M,N>`
 * surface.
 *
 * Three storage flavours (the `Storage` template parameter)
 * ---------------------------------------------------------
 * The *same* accessor code is instantiated for three ways of reaching storage,
 * which is what makes host/device equivalence by construction possible:
 *
 *   * `storage::buffers` — `LBufferView<S>` handles: the flavour the ported
 *     **consumer code and the containers** use.  It is host-side: it carries no
 *     element access (a host dereference of device memory would be a hidden
 *     synchronisation) but hands out the launch arguments.
 *   * `storage::device` — `lcrt::Expr<lcrt::Buffer<S>>` handles: the flavour
 *     built **inside a kernel** from the kernel's buffer/scalar arguments.
 *   * `storage::memory` — raw host pointers: the flavour used by the unit tests
 *     and by any host-side reference evaluation of the same routine.
 *
 * RULE (views vs. kernels): LC compiles a shader *once*, and a buffer captured
 * into a kernel lambda becomes a **bound argument frozen at compile time**.
 * scratch/lc_lin_spike proves that re-dispatching such a shader after the owner
 * reallocated keeps writing to the *old* buffer.  Therefore a view is never
 * captured: pass its handles as kernel arguments
 *
 *     static auto k = lud::compile1<lcrt::Buffer<int>, lcrt::Buffer<int>,
 *                                   lcrt::Buffer<double>, uint32_t>(
 *         [](BufferVar<int> rows, BufferVar<int> cols, BufferVar<double> vals,
 *            Var<uint32_t> n) noexcept {
 *             auto A = lla::LCTripletMatrixDeviceView<double, 3>{
 *                 n_rows, n_cols, n, rows, cols, vals};   // or from_handles(...)
 *             ...
 *         });
 *     stream << k(A.row_indices(), A.col_indices(), A.values(), A.triplet_count())
 *                   .dispatch(A.triplet_count());
 *
 * Views are frame-scoped (plan risk R5): a view must not survive a resize of the
 * container that produced it.  The host flavour records the container's
 * generation cell and `use_check()` throws once the owner has resized.
 *********************************************************************/
#include <luisa_device/math/mat.h>
#include <luisa_device/view.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uipc::backend::luisa::luisa_device::linalg
{
namespace lcrt = ::luisa::compute;
namespace lmath = ::uipc::backend::luisa::luisa_device::math;

using lmath::LMat;
using lmath::LVec;

// ---------------------------------------------------------------------------
//  small helpers
// ---------------------------------------------------------------------------
/// `int2` replacement for view metadata (the reference uses CUDA's int2, whose
/// fields are plain ints; ours carries the flavour's index type).
template <typename I>
struct index_pair
{
    I x{};
    I y{};
};

template <typename I>
[[nodiscard]] constexpr index_pair<I> make_index_pair(I x, I y) noexcept
{
    return index_pair<I>{x, y};
}

/// Underlying *storage* scalar of a logical element type: `double` for `double`,
/// `float` for `float`, `double` for `lcrt::Var<double>` / `lcrt::Expr<double>`.
template <typename T>
struct buffer_scalar_of
{
    using type = std::conditional_t<std::is_arithmetic_v<std::remove_cvref_t<T>>,
                                    std::remove_cvref_t<T>,
                                    lcrt::expr_value_t<std::remove_cvref_t<T>>>;
};
template <typename T>
using buffer_scalar_of_t = typename buffer_scalar_of<T>::type;

// ---------------------------------------------------------------------------
//  Storage flavours
// ---------------------------------------------------------------------------
namespace storage
{
/// Handle flavour used by host-side code and by the owning containers.
/// Hands out `LBufferView` handles (launch arguments); no element access.
struct buffers
{
    static constexpr bool is_device = false;

    template <typename S>
    using handle = LBufferView<S>;

    using index_type = std::int32_t;

    /// Literal index of this flavour (a plain host int here, a DSL literal on the
    /// device flavour — LC's `Expr` literal constructor only accepts its own
    /// element type, so every int -> index conversion goes through this).
    [[nodiscard]] static constexpr index_type from_int(std::int32_t v) noexcept { return v; }

    /// Normalise a caller-provided index to this flavour's index type.
    template <typename I>
    [[nodiscard]] static constexpr index_type as_index(I&& i) noexcept
    {
        return static_cast<index_type>(std::forward<I>(i));
    }

    template <typename S>
    [[nodiscard]] static LBufferView<S> subview(const LBufferView<S>& b,
                                                std::size_t           offset,
                                                std::size_t           count) noexcept
    {
        return b.subview(offset, count);
    }

    // Element access through an LBufferView is a *device* operation; reaching it
    // from the host would silently synchronise (or read stale bytes).  Kept as
    // hard compile errors so the mistake is loud.
    template <typename S, typename I>
    static S load(const LBufferView<S>&, I&&) noexcept
    {
        static_assert(sizeof(S) == 0,
                      "linalg: element access through an LBufferView-backed view is a device "
                      "operation. Pass the view's handles as kernel arguments and construct a "
                      "*Device* view inside the kernel, or use storage::memory for host code.");
        return {};
    }
    template <typename S, typename I, typename V>
    static void store(const LBufferView<S>&, I&&, V&&) noexcept
    {
        static_assert(sizeof(S) == 0,
                      "linalg: element access through an LBufferView-backed view is a device "
                      "operation (see load()).");
    }
    template <typename S, typename I, typename V>
    static void atomic_add_at(const LBufferView<S>&, I&&, V&&) noexcept
    {
        static_assert(sizeof(S) == 0,
                      "linalg: atomics go through a *Device* view (see load()).");
    }
};

/// Raw host memory flavour: the host-side reference evaluation of a ported body.
struct memory
{
    static constexpr bool is_device = false;

    template <typename S>
    using handle = S*;

    using index_type = std::int32_t;

    [[nodiscard]] static constexpr index_type from_int(std::int32_t v) noexcept { return v; }

    template <typename I>
    [[nodiscard]] static constexpr index_type as_index(I&& i) noexcept
    {
        return static_cast<index_type>(std::forward<I>(i));
    }

    template <typename S>
    [[nodiscard]] static S* subview(S* b, std::size_t offset, std::size_t) noexcept
    {
        return b + offset;
    }

    template <typename S, typename I>
    [[nodiscard]] static S load(const S* p, I i) noexcept
    {
        return p[static_cast<std::size_t>(i)];
    }
    template <typename S, typename I, typename V>
    static void store(S* p, I i, V&& v) noexcept
    {
        p[static_cast<std::size_t>(i)] = static_cast<S>(std::forward<V>(v));
    }
    template <typename S, typename I, typename V>
    [[nodiscard]] static S atomic_add_at(S* p, I i, V&& v) noexcept
    {
        const auto idx = static_cast<std::size_t>(i);
        const S    old = p[idx];
        p[idx]         = static_cast<S>(old + v);
        return old;
    }
};

/// DSL flavour: built inside a kernel from the kernel's buffer arguments.
struct device
{
    static constexpr bool is_device = true;

    template <typename S>
    using handle = lcrt::Expr<lcrt::Buffer<S>>;

    /// Signed, exactly like the reference's `int` indices.  Keeping the device
    /// index type signed means the view's index arithmetic has the same
    /// signedness as the reference's and needs no cast when it is written back
    /// into the `int` index buffers.
    using index_type = lcrt::Expr<std::int32_t>;

    /// Trace-time only (constructing an `Expr` literal needs an active
    /// FunctionBuilder) — which is exactly where a device view is built.
    [[nodiscard]] static index_type from_int(std::int32_t v) noexcept
    {
        return index_type{static_cast<std::int32_t>(v)};
    }

    /// Normalise a caller-provided index to this flavour's index type: host
    /// literals become DSL literals, DSL values are cast (a kernel index is
    /// usually an unsigned `dispatch_x()`/thread id).
    template <typename I>
    [[nodiscard]] static index_type as_index(I&& i) noexcept
    {
        if constexpr(std::is_integral_v<std::remove_cvref_t<I>>)
        {
            return from_int(static_cast<std::int32_t>(i));
        }
        else
        {
            return lcrt::cast<std::int32_t>(std::forward<I>(i));
        }
    }

    template <typename S, typename I>
    [[nodiscard]] static auto subview(const lcrt::Expr<lcrt::Buffer<S>>& b, I&&, std::size_t) noexcept
    {
        // A device view covers the whole buffer argument; sub-ranges are expressed
        // through the view's own (index) offsets, exactly like the reference.
        return b;
    }

    template <typename S, typename I>
    [[nodiscard]] static auto load(const lcrt::Expr<lcrt::Buffer<S>>& b, I&& i) noexcept
    {
        return b.read(std::forward<I>(i));
    }
    template <typename S, typename I, typename V>
    static void store(const lcrt::Expr<lcrt::Buffer<S>>& b, I&& i, V&& v) noexcept
    {
        b.write(std::forward<I>(i), std::forward<V>(v));
    }
    template <typename S, typename I, typename V>
    [[nodiscard]] static auto atomic_add_at(const lcrt::Expr<lcrt::Buffer<S>>& b, I&& i, V&& v) noexcept
    {
        return b.atomic(std::forward<I>(i)).fetch_add(std::forward<V>(v));
    }
};

/// Value type `load()` yields (the type an element proxy converts to).
template <typename S, typename Storage>
using loaded_t = std::conditional_t<Storage::is_device, lcrt::Expr<S>, S>;
}// namespace storage

namespace detail
{
/// Shared generation counter (plan risk R5).  The container bumps it whenever it
/// resizes/reallocates; live views keep the value they were created with.
struct generation_cell
{
    std::uint64_t value = 0u;
};

/// Bounds checks on *host* index values (device indices cannot drive host control
/// flow — RULE M2 — so they are unchecked there, exactly as in T2.3's predicates).
inline void check_host_range(bool ok, const char* what)
{
    if(!ok)
    {
        throw std::out_of_range{what};
    }
}

/// The reference's `UIPC_KERNEL_ASSERT`: a real check on the host, a no-op on the
/// device path (there is no portable way to branch on device data — RULE M2).
inline void kernel_assert(bool ok, const char* what)
{
    check_host_range(ok, what);
}
}// namespace detail

/// Frame-lifetime guard shared by the owning container and its views.
using generation_cell = detail::generation_cell;

// ---------------------------------------------------------------------------
//  Traits of a (possibly block) matrix entry
// ---------------------------------------------------------------------------
template <typename T, int M, int N>
struct block_traits
{
    static_assert(M > 0 && N > 0, "linalg: block dimensions must be positive");

    static constexpr bool is_scalar = (M == 1 && N == 1);
    /// Logical block type: `T` for a scalar entry, `LMat<T,M,N>` otherwise.
    using value_type = std::conditional_t<is_scalar, T, LMat<T, M, N>>;
    /// Storage scalar behind `T` (double/float/int).
    using scalar_type = buffer_scalar_of_t<T>;
    static constexpr int block_size = M * N;
};

// ---------------------------------------------------------------------------
//  Triplet matrix view (COO, possibly block)
// ---------------------------------------------------------------------------
template <bool IsConst, typename T, int M, int N = M, typename Storage = storage::buffers>
class TripletMatrixViewT
{
    template <typename U>
    using auto_const_t = std::conditional_t<IsConst, const U, U>;

    template <bool OtherIsConst, typename U, int M_, int N_, typename S_>
    friend class TripletMatrixViewT;

  public:
    static_assert(!std::is_const_v<T>, "linalg: the element type must be non-const");
    static_assert(M > 0 && N > 0, "linalg: block dimensions must be positive");

    using storage_type = Storage;
    using traits       = block_traits<T, M, N>;
    using ValueT       = typename traits::value_type;
    using ScalarT      = typename traits::scalar_type;
    using index_type   = typename Storage::index_type;
    using index_pair_t = index_pair<index_type>;

    static constexpr bool IsBlockMatrix = (M > 1 || N > 1);
    static constexpr int  block_rows    = M;
    static constexpr int  block_cols    = N;
    static constexpr int  block_entries = M * N;

    using row_buffer_type   = typename Storage::template handle<std::int32_t>;
    using col_buffer_type   = row_buffer_type;
    using value_buffer_type = typename Storage::template handle<ScalarT>;

    using ConstView    = TripletMatrixViewT<true, T, M, N, Storage>;
    using NonConstView = TripletMatrixViewT<false, T, M, N, Storage>;
    using ThisView     = TripletMatrixViewT<IsConst, T, M, N, Storage>;

    /// Read-only triplet returned by `operator()` of a const view.  An aggregate,
    /// so `auto&& [i, j, block] = A(k);` decomposes exactly like the reference
    /// (the reference's `const ValueT&` member becomes a *value*: LC has no
    /// device-side lvalue reference).
    struct CTriplet
    {
        index_type row_index;
        index_type col_index;
        ValueT     value;
    };

  protected:
    index_type   m_total_rows{};
    index_type   m_total_cols{};
    index_type   m_triplet_index_offset{};
    index_type   m_triplet_count{};
    index_type   m_total_triplet_count{};
    index_pair_t m_submatrix_offset{};
    index_pair_t m_submatrix_extent{};

    row_buffer_type   m_row_indices{};
    col_buffer_type   m_col_indices{};
    value_buffer_type m_values{};

    // Frame-lifetime guard (host flavour only; null for the other flavours).
    std::shared_ptr<const detail::generation_cell> m_lifetime{};
    std::uint64_t                                  m_generation = 0u;

    /// Write proxy returned by `operator()` of a non-const view.  The reference's
    /// `Proxy` is a nested class holding `const TripletMatrixViewT&`; ours holds a
    /// copy of the (cheap, handle-only) view so no reference outlives it.
    template <typename I>
    class ProxyT
    {
        friend class TripletMatrixViewT;
        TripletMatrixViewT m_view;
        I                  m_index{};

        ProxyT(const TripletMatrixViewT& viewer, I index) noexcept
            : m_view(viewer)
            , m_index(index)
        {
        }

      public:
        [[nodiscard]] CTriplet read() && { return m_view.at(m_index); }

        template <typename R, typename C>
        void write(R&& row_index, C&& col_index, const ValueT& block) &&
        {
            const auto index = m_view.get_index(m_index);
            if constexpr(!Storage::is_device)
            {
                m_view.check_in_submatrix(row_index, col_index);
            }
            const auto global_i = m_view.m_submatrix_offset.x + row_index;
            const auto global_j = m_view.m_submatrix_offset.y + col_index;

            Storage::template store<std::int32_t>(m_view.m_row_indices, index, global_i);
            Storage::template store<std::int32_t>(m_view.m_col_indices, index, global_j);
            m_view.store_block(index, block);
        }
    };

  public:
    /// The reference's `Proxy` spelling.
    using Proxy = ProxyT<std::int32_t>;

    TripletMatrixViewT() = default;

    /// The reference's full constructor (same parameter order).  `submatrix_offset`
    /// /`submatrix_extent` replace the CUDA `int2` arguments.
    TripletMatrixViewT(index_type   total_block_rows,
                       index_type   total_block_cols,
                       index_type   triplet_index_offset,
                       index_type   triplet_count,
                       index_type   total_triplet_count,
                       index_pair_t submatrix_offset,
                       index_pair_t submatrix_extent,
                       row_buffer_type   row_indices,
                       col_buffer_type   col_indices,
                       value_buffer_type values)
        : m_total_rows(total_block_rows)
        , m_total_cols(total_block_cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_submatrix_offset(submatrix_offset)
        , m_submatrix_extent(submatrix_extent)
        , m_row_indices(row_indices)
        , m_col_indices(col_indices)
        , m_values(values)
    {
        // Host index values only (the device flavour cannot branch on them).
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(triplet_index_offset + triplet_count <= total_triplet_count,
                                  "TripletMatrixView: out of range");
            detail::kernel_assert(submatrix_offset.x >= 0 && submatrix_offset.y >= 0,
                                  "TripletMatrixView: submatrix_offset out of range");
            detail::kernel_assert(submatrix_offset.x + submatrix_extent.x <= total_block_rows,
                                  "TripletMatrixView: submatrix out of range (rows)");
            detail::kernel_assert(submatrix_offset.y + submatrix_extent.y <= total_block_cols,
                                  "TripletMatrixView: submatrix out of range (cols)");
        }
    }

    /// Reference's "whole matrix, no subview" constructor.
    TripletMatrixViewT(index_type        total_block_rows,
                       index_type        total_block_cols,
                       index_type        total_triplet_count,
                       row_buffer_type   block_row_indices,
                       col_buffer_type   block_col_indices,
                       value_buffer_type block_values)
        : TripletMatrixViewT(total_block_rows,
                             total_block_cols,
                             Storage::from_int(0),
                             total_triplet_count,
                             total_triplet_count,
                             index_pair_t{Storage::from_int(0), Storage::from_int(0)},
                             index_pair_t{total_block_rows, total_block_cols},
                             block_row_indices,
                             block_col_indices,
                             block_values)
    {
    }

    /// Device-side factory (named so a kernel body reads clearly).  Builds the
    /// "whole matrix" view from the kernel's handle arguments.
    /// Device-side factory (named so a kernel body reads clearly).  The index
    /// arguments go through `Storage::as_index`, so a kernel may pass its own
    /// scalar arguments (`lcrt::Var<uint32_t>` counts, plain ints, ...) without
    /// casting at every call site.
    template <typename R, typename C, typename N>
    [[nodiscard]] static ThisView from_handles(row_buffer_type   row_indices,
                                               col_buffer_type   col_indices,
                                               value_buffer_type values,
                                               R&&               total_rows,
                                               C&&               total_cols,
                                               N&&               count)
    {
        return ThisView{Storage::as_index(std::forward<R>(total_rows)),
                        Storage::as_index(std::forward<C>(total_cols)),
                        Storage::as_index(std::forward<N>(count)),
                        row_indices,
                        col_indices,
                        values};
    }

    /// Full device-side factory (reference constructor order; the two `int2`
    /// arguments are spelled as four scalars).
    template <typename R, typename C, typename O, typename N, typename TN,
              typename OX, typename OY, typename EX, typename EY>
    [[nodiscard]] static ThisView from_handles_full(row_buffer_type   row_indices,
                                                    col_buffer_type   col_indices,
                                                    value_buffer_type values,
                                                    R&&               total_rows,
                                                    C&&               total_cols,
                                                    O&&               triplet_index_offset,
                                                    N&&               triplet_count,
                                                    TN&&              total_triplet_count,
                                                    OX&&              submatrix_offset_x,
                                                    OY&&              submatrix_offset_y,
                                                    EX&&              submatrix_extent_x,
                                                    EY&&              submatrix_extent_y)
    {
        return ThisView{Storage::as_index(std::forward<R>(total_rows)),
                        Storage::as_index(std::forward<C>(total_cols)),
                        Storage::as_index(std::forward<O>(triplet_index_offset)),
                        Storage::as_index(std::forward<N>(triplet_count)),
                        Storage::as_index(std::forward<TN>(total_triplet_count)),
                        index_pair_t{Storage::as_index(std::forward<OX>(submatrix_offset_x)),
                                     Storage::as_index(std::forward<OY>(submatrix_offset_y))},
                        index_pair_t{Storage::as_index(std::forward<EX>(submatrix_extent_x)),
                                     Storage::as_index(std::forward<EY>(submatrix_extent_y))},
                        row_indices,
                        col_indices,
                        values};
    }

    /// non-const -> const conversion.
    template <bool OtherIsConst>
    TripletMatrixViewT(const TripletMatrixViewT<OtherIsConst, T, M, N, Storage>& other) noexcept
        requires(IsConst)
        : m_total_rows(other.m_total_rows)
        , m_total_cols(other.m_total_cols)
        , m_triplet_index_offset(other.m_triplet_index_offset)
        , m_triplet_count(other.m_triplet_count)
        , m_total_triplet_count(other.m_total_triplet_count)
        , m_submatrix_offset(other.m_submatrix_offset)
        , m_submatrix_extent(other.m_submatrix_extent)
        , m_row_indices(other.m_row_indices)
        , m_col_indices(other.m_col_indices)
        , m_values(other.m_values)
        , m_lifetime(other.m_lifetime)
        , m_generation(other.m_generation)
    {
    }

    // -- frame lifetime ------------------------------------------------------
    void set_lifetime(std::shared_ptr<const detail::generation_cell> cell,
                      std::uint64_t                                  generation) noexcept
    {
        m_lifetime   = std::move(cell);
        m_generation = generation;
    }
    [[nodiscard]] std::uint64_t generation() const noexcept { return m_generation; }
    /// Throws if the owning container reallocated since this view was created.
    void use_check() const
    {
        if(m_lifetime && m_lifetime->value != m_generation)
        {
            throw std::runtime_error{
                "linalg: view used across a resize (plan risk R5) — views are frame-scoped"};
        }
    }

    // -- metadata -----------------------------------------------------------
    [[nodiscard]] index_type   total_rows() const noexcept { return m_total_rows; }
    [[nodiscard]] index_type   total_cols() const noexcept { return m_total_cols; }
    [[nodiscard]] index_pair_t total_extent() const noexcept
    {
        return index_pair_t{m_total_rows, m_total_cols};
    }
    [[nodiscard]] index_pair_t submatrix_offset() const noexcept { return m_submatrix_offset; }
    [[nodiscard]] index_pair_t extent() const noexcept { return m_submatrix_extent; }
    [[nodiscard]] index_type   triplet_count() const noexcept { return m_triplet_count; }
    /// The reference's typo is preserved for a mechanical port; the correct
    /// spelling is provided as well.
    [[nodiscard]] index_type tripet_index_offset() const noexcept
    {
        return m_triplet_index_offset;
    }
    [[nodiscard]] index_type triplet_index_offset() const noexcept
    {
        return m_triplet_index_offset;
    }
    [[nodiscard]] index_type total_triplet_count() const noexcept
    {
        return m_total_triplet_count;
    }
    [[nodiscard]] index_type non_zeros() const noexcept { return m_triplet_count; }

    // -- handles (the launch arguments) --------------------------------------
    [[nodiscard]] auto_const_t<row_buffer_type> row_indices() const noexcept
    {
        return m_row_indices;
    }
    [[nodiscard]] auto_const_t<col_buffer_type> col_indices() const noexcept
    {
        return m_col_indices;
    }
    /// Flat scalar storage: `M*N` scalars per triplet, column-major within the
    /// block (the reference's `values()` was a block-typed view; the flat storage
    /// is what replaces the raw `values().data()` pointer arithmetic).
    [[nodiscard]] auto_const_t<value_buffer_type> values() const noexcept
    {
        return m_values;
    }

    // -- sub views -----------------------------------------------------------
    // NOTE: sub-views are *constructed* rather than copy-and-assign: LC's Expr
    // (the device flavour's index type) deletes copy assignment.
    template <typename O, typename C>
    [[nodiscard]] ThisView subview(O&& offset, C&& count) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(offset + count <= m_triplet_count,
                                  "TripletMatrixView: subview out of range");
        }
        return ThisView{m_total_rows,
                        m_total_cols,
                        m_triplet_index_offset + Storage::as_index(std::forward<O>(offset)),
                        Storage::as_index(std::forward<C>(count)),
                        m_total_triplet_count,
                        m_submatrix_offset,
                        m_submatrix_extent,
                        m_row_indices,
                        m_col_indices,
                        m_values};
    }
    template <typename O>
    [[nodiscard]] ThisView subview(O&& offset) const
    {
        return subview(std::forward<O>(offset),
                       m_triplet_count - Storage::as_index(offset));
    }
    [[nodiscard]] ThisView submatrix(index_pair_t offset, index_pair_t extent) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(offset.x >= 0 && offset.y >= 0,
                                  "TripletMatrixView: submatrix offset");
            detail::kernel_assert(offset.x + extent.x <= m_submatrix_extent.x
                                      && offset.y + extent.y <= m_submatrix_extent.y,
                                  "TripletMatrixView: submatrix extent");
        }
        return ThisView{m_total_rows,
                        m_total_cols,
                        m_triplet_index_offset,
                        m_triplet_count,
                        m_total_triplet_count,
                        index_pair_t{m_submatrix_offset.x + offset.x,
                                     m_submatrix_offset.y + offset.y},
                        extent,
                        m_row_indices,
                        m_col_indices,
                        m_values};
    }

    // -- viewers are the views themselves (muda parity) ----------------------
    [[nodiscard]] ThisView  viewer() const noexcept { return *this; }
    [[nodiscard]] ConstView cviewer() const noexcept { return as_const(); }
    [[nodiscard]] ConstView as_const() const noexcept { return ConstView{*this}; }
    [[nodiscard]] const char* kernel_file() const noexcept { return ""; }
    [[nodiscard]] int         kernel_line() const noexcept { return 0; }

    // -- element access (device / host-memory flavours) ----------------------
    /// Storage index of the element (local_r, local_c) of block `b`.
    template <typename I>
    [[nodiscard]] static auto block_element_index(I&& b, int local_r, int local_c) noexcept
    {
        return b * static_cast<std::uint32_t>(block_entries)
               + static_cast<std::uint32_t>(local_c * M + local_r);
    }

    /// The block of triplet `index` (which already includes the view's offset),
    /// read as an `LMat<T,M,N>` (or `T` for a 1x1 block).
    template <typename I>
    [[nodiscard]] auto load_block(I&& index) const noexcept
    {
        if constexpr(M == 1 && N == 1)
        {
            return Storage::template load<ScalarT>(m_values, std::forward<I>(index));
        }
        else
        {
            ValueT out{};  // zero-initialised, then assigned element-wise
            for(int c = 0; c < N; ++c)
            {
                for(int r = 0; r < M; ++r)
                {
                    out(r, c) = Storage::template load<ScalarT>(
                        m_values,
                        block_element_index(std::forward<I>(index), r, c));
                }
            }
            return out;
        }
    }

    /// Store a block (same layout).
    template <typename I>
    void store_block(I&& index, const ValueT& block) const noexcept
    {
        if constexpr(M == 1 && N == 1)
        {
            Storage::template store<ScalarT>(m_values, std::forward<I>(index), block);
        }
        else
        {
            for(int c = 0; c < N; ++c)
            {
                for(int r = 0; r < M; ++r)
                {
                    Storage::template store<ScalarT>(
                        m_values,
                        block_element_index(std::forward<I>(index), r, c),
                        block(r, c));
                }
            }
        }
    }

    /// Scalar element `(r, c)` of triplet `i` (a view-relative index).
    template <typename I>
    [[nodiscard]] auto value_at(I&& i, int r, int c) const noexcept
    {
        return Storage::template load<ScalarT>(
            m_values, block_element_index(get_index(std::forward<I>(i)), r, c));
    }

    // -- operators -----------------------------------------------------------
    // const view  -> CTriplet (read-only, decomposable)
    // non-const   -> Proxy (read/write), as in the reference
    template <typename I>
    [[nodiscard]] decltype(auto) operator()(I&& i) const
    {
        if constexpr(IsConst)
        {
            return at(std::forward<I>(i));
        }
        else
        {
            return ProxyT<std::remove_cvref_t<I>>{*this, std::forward<I>(i)};
        }
    }

  protected:
    template <typename I>
    [[nodiscard]] CTriplet at(I&& i) const
    {
        const auto index    = get_index(std::forward<I>(i));
        const auto global_i = Storage::template load<std::int32_t>(m_row_indices, index);
        const auto global_j = Storage::template load<std::int32_t>(m_col_indices, index);
        const auto sub_i    = global_i - m_submatrix_offset.x;
        const auto sub_j    = global_j - m_submatrix_offset.y;
        if constexpr(!Storage::is_device)
        {
            check_in_submatrix(sub_i, sub_j);
        }
        return CTriplet{sub_i, sub_j, load_block(index)};
    }

    template <typename I>
    [[nodiscard]] auto get_index(I&& i) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(i >= 0 && i < m_triplet_count,
                                  "TripletMatrixView: triplet index out of range");
        }
        return m_triplet_index_offset + Storage::as_index(std::forward<I>(i));
    }

    template <typename I, typename J>
    void check_in_submatrix(I&& i, J&& j) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(i >= 0 && i < m_submatrix_extent.x,
                                  "TripletMatrixView: row index out of submatrix range");
            detail::kernel_assert(j >= 0 && j < m_submatrix_extent.y,
                                  "TripletMatrixView: col index out of submatrix range");
        }
    }
};

// ---------------------------------------------------------------------------
//  Doublet (sparse-segment) vector view
// ---------------------------------------------------------------------------
template <bool IsConst, typename T, int N = 1, typename Storage = storage::buffers>
class DoubletVectorViewT
{
    template <typename U>
    using auto_const_t = std::conditional_t<IsConst, const U, U>;

    template <bool OtherIsConst, typename U, int N_, typename S_>
    friend class DoubletVectorViewT;

  public:
    static_assert(!std::is_const_v<T>, "linalg: the element type must be non-const");
    static_assert(N > 0, "linalg: segment dimension must be positive");

    using storage_type = Storage;
    /// The reference's `Eigen::Vector<T,N>`: `LVec<T,N>` for N > 1, plain T else.
    using ValueT     = std::conditional_t<(N == 1), T, LVec<T, N>>;
    using ScalarT    = buffer_scalar_of_t<T>;
    using index_type = typename Storage::index_type;

    static constexpr bool IsSegmentVector = (N > 1);
    static constexpr int  segment_dim     = N;

    using index_buffer_type = typename Storage::template handle<std::int32_t>;
    using value_buffer_type = typename Storage::template handle<ScalarT>;

    using ConstView    = DoubletVectorViewT<true, T, N, Storage>;
    using NonConstView = DoubletVectorViewT<false, T, N, Storage>;
    using ThisView     = DoubletVectorViewT<IsConst, T, N, Storage>;

    /// Read-only doublet (aggregate; see CTriplet's note).
    struct CDoublet
    {
        index_type index;
        ValueT     value;
    };

  protected:
    index_type m_total_segment_count{};
    index_type m_doublet_index_offset{};
    index_type m_doublet_count{};
    index_type m_total_doublet_count{};
    index_type m_subvector_offset{};
    index_type m_subvector_extent{};

    index_buffer_type m_indices{};
    value_buffer_type m_values{};

    std::shared_ptr<const detail::generation_cell> m_lifetime{};
    std::uint64_t                                  m_generation = 0u;

    template <typename I>
    class ProxyT
    {
        friend class DoubletVectorViewT;
        DoubletVectorViewT m_view;
        I                  m_index{};

        ProxyT(const DoubletVectorViewT& viewer, I index) noexcept
            : m_view(viewer)
            , m_index(index)
        {
        }

      public:
        [[nodiscard]] CDoublet read() && { return m_view.at(m_index); }

        template <typename SI>
        void write(SI&& segment_i, const ValueT& value) &&
        {
            const auto index = m_view.get_index(m_index);
            if constexpr(!Storage::is_device)
            {
                m_view.check_in_subvector(segment_i);
            }
            Storage::template store<std::int32_t>(
                m_view.m_indices,
                index,
                m_view.m_subvector_offset + Storage::as_index(std::forward<SI>(segment_i)));
            m_view.store_segment(index, value);
        }
    };

  public:
    using Proxy = ProxyT<std::int32_t>;

    DoubletVectorViewT() = default;

    DoubletVectorViewT(index_type        total_segment_count,
                       index_type        doublet_index_offset,
                       index_type        doublet_count,
                       index_type        total_doublet_count,
                       index_type        subvector_offset,
                       index_type        subvector_extent,
                       index_buffer_type indices,
                       value_buffer_type values)
        : m_total_segment_count(total_segment_count)
        , m_doublet_index_offset(doublet_index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_subvector_offset(subvector_offset)
        , m_subvector_extent(subvector_extent)
        , m_indices(indices)
        , m_values(values)
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(doublet_index_offset + doublet_count <= total_doublet_count,
                                  "DoubletVectorView: out of range");
            detail::kernel_assert(subvector_offset + subvector_extent <= total_segment_count,
                                  "DoubletVectorView: subvector out of range");
        }
    }

    DoubletVectorViewT(index_type        total_segment_count,
                       index_type        total_doublet_count,
                       index_buffer_type segment_indices,
                       value_buffer_type segment_values)
        : DoubletVectorViewT(total_segment_count,
                             Storage::from_int(0),
                             total_doublet_count,
                             total_doublet_count,
                             Storage::from_int(0),
                             total_segment_count,
                             segment_indices,
                             segment_values)
    {
    }

    /// Device-side factories (see TripletMatrixViewT::from_handles).
    template <typename S, typename N>
    [[nodiscard]] static ThisView from_handles(index_buffer_type indices,
                                               value_buffer_type values,
                                               S&&               total_segments,
                                               N&&               count)
    {
        return ThisView{Storage::as_index(std::forward<S>(total_segments)),
                        Storage::as_index(std::forward<N>(count)),
                        indices,
                        values};
    }

    template <typename S, typename O, typename N, typename TN, typename SO, typename SE>
    [[nodiscard]] static ThisView from_handles_full(index_buffer_type indices,
                                                    value_buffer_type values,
                                                    S&&               total_segments,
                                                    O&&               doublet_index_offset,
                                                    N&&               doublet_count,
                                                    TN&&              total_doublet_count,
                                                    SO&&              subvector_offset,
                                                    SE&&              subvector_extent)
    {
        return ThisView{Storage::as_index(std::forward<S>(total_segments)),
                        Storage::as_index(std::forward<O>(doublet_index_offset)),
                        Storage::as_index(std::forward<N>(doublet_count)),
                        Storage::as_index(std::forward<TN>(total_doublet_count)),
                        Storage::as_index(std::forward<SO>(subvector_offset)),
                        Storage::as_index(std::forward<SE>(subvector_extent)),
                        indices,
                        values};
    }

    template <bool OtherIsConst>
    DoubletVectorViewT(const DoubletVectorViewT<OtherIsConst, T, N, Storage>& other) noexcept
        requires(IsConst)
        : m_total_segment_count(other.m_total_segment_count)
        , m_doublet_index_offset(other.m_doublet_index_offset)
        , m_doublet_count(other.m_doublet_count)
        , m_total_doublet_count(other.m_total_doublet_count)
        , m_subvector_offset(other.m_subvector_offset)
        , m_subvector_extent(other.m_subvector_extent)
        , m_indices(other.m_indices)
        , m_values(other.m_values)
        , m_lifetime(other.m_lifetime)
        , m_generation(other.m_generation)
    {
    }

    // -- frame lifetime ------------------------------------------------------
    void set_lifetime(std::shared_ptr<const detail::generation_cell> cell,
                      std::uint64_t                                  generation) noexcept
    {
        m_lifetime   = std::move(cell);
        m_generation = generation;
    }
    [[nodiscard]] std::uint64_t generation() const noexcept { return m_generation; }
    void                        use_check() const
    {
        if(m_lifetime && m_lifetime->value != m_generation)
        {
            throw std::runtime_error{
                "linalg: view used across a resize (plan risk R5) — views are frame-scoped"};
        }
    }

    // -- metadata -----------------------------------------------------------
    [[nodiscard]] index_type extent() const noexcept { return m_subvector_extent; }
    [[nodiscard]] index_type total_extent() const noexcept { return m_total_segment_count; }
    [[nodiscard]] index_type subvector_offset() const noexcept { return m_subvector_offset; }
    [[nodiscard]] index_type doublet_count() const noexcept { return m_doublet_count; }
    [[nodiscard]] index_type total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }
    [[nodiscard]] index_type doublet_index_offset() const noexcept
    {
        return m_doublet_index_offset;
    }
    [[nodiscard]] index_type non_zeros() const noexcept { return m_doublet_count; }

    [[nodiscard]] auto_const_t<index_buffer_type> indices() const noexcept { return m_indices; }
    /// Flat scalar storage: `N` scalars per doublet.
    [[nodiscard]] auto_const_t<value_buffer_type> values() const noexcept { return m_values; }

    // -- sub views -----------------------------------------------------------
    template <typename O, typename C>
    [[nodiscard]] ThisView subview(O&& offset, C&& count) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(offset + count <= m_doublet_count,
                                  "DoubletVectorView: subview out of range");
        }
        return ThisView{m_total_segment_count,
                        m_doublet_index_offset + Storage::as_index(std::forward<O>(offset)),
                        Storage::as_index(std::forward<C>(count)),
                        m_total_doublet_count,
                        m_subvector_offset,
                        m_subvector_extent,
                        m_indices,
                        m_values};
    }
    template <typename O>
    [[nodiscard]] ThisView subview(O&& offset) const
    {
        return subview(std::forward<O>(offset),
                       m_doublet_count - Storage::as_index(offset));
    }
    template <typename O, typename E>
    [[nodiscard]] ThisView subvector(O&& offset, E&& extent) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(offset + extent <= m_subvector_extent,
                                  "DoubletVectorView: subvector out of range");
        }
        return ThisView{m_total_segment_count,
                        m_doublet_index_offset,
                        m_doublet_count,
                        m_total_doublet_count,
                        m_subvector_offset + Storage::as_index(std::forward<O>(offset)),
                        Storage::as_index(std::forward<E>(extent)),
                        m_indices,
                        m_values};
    }

    [[nodiscard]] ThisView  viewer() const noexcept { return *this; }
    [[nodiscard]] ConstView cviewer() const noexcept { return as_const(); }
    [[nodiscard]] ConstView as_const() const noexcept { return ConstView{*this}; }
    [[nodiscard]] const char* kernel_file() const noexcept { return ""; }
    [[nodiscard]] int         kernel_line() const noexcept { return 0; }

    // -- element access ------------------------------------------------------
    template <typename I>
    [[nodiscard]] static auto segment_element_index(I&& b, int k) noexcept
    {
        return b * static_cast<std::uint32_t>(N) + static_cast<std::uint32_t>(k);
    }

    template <typename I>
    [[nodiscard]] auto load_segment(I&& index) const noexcept
    {
        if constexpr(N == 1)
        {
            return Storage::template load<ScalarT>(m_values, std::forward<I>(index));
        }
        else
        {
            ValueT out{};
            for(int k = 0; k < N; ++k)
            {
                out(k) = Storage::template load<ScalarT>(
                    m_values, segment_element_index(std::forward<I>(index), k));
            }
            return out;
        }
    }

    template <typename I>
    void store_segment(I&& index, const ValueT& segment) const noexcept
    {
        if constexpr(N == 1)
        {
            Storage::template store<ScalarT>(m_values, std::forward<I>(index), segment);
        }
        else
        {
            for(int k = 0; k < N; ++k)
            {
                Storage::template store<ScalarT>(
                    m_values, segment_element_index(std::forward<I>(index), k), segment(k));
            }
        }
    }

    template <typename I>
    [[nodiscard]] decltype(auto) operator()(I&& i) const
    {
        if constexpr(IsConst)
        {
            return at(std::forward<I>(i));
        }
        else
        {
            return ProxyT<std::remove_cvref_t<I>>{*this, std::forward<I>(i)};
        }
    }

  protected:
    template <typename I>
    [[nodiscard]] CDoublet at(I&& i) const
    {
        const auto index    = get_index(std::forward<I>(i));
        const auto global_i = Storage::template load<std::int32_t>(m_indices, index);
        const auto sub_i    = global_i - m_subvector_offset;
        if constexpr(!Storage::is_device)
        {
            check_in_subvector(sub_i);
        }
        return CDoublet{sub_i, load_segment(index)};
    }

    template <typename I>
    [[nodiscard]] auto get_index(I&& i) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(i >= 0 && i < m_doublet_count,
                                  "DoubletVectorView: index out of range");
        }
        return m_doublet_index_offset + Storage::as_index(std::forward<I>(i));
    }

    template <typename I>
    void check_in_subvector(I&& i) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(i >= 0 && i < m_subvector_extent,
                                  "DoubletVectorView: index out of subvector range");
        }
    }
};

// ---------------------------------------------------------------------------
//  Dense vector view
// ---------------------------------------------------------------------------
/// `Size > 0` is the Eigen-`segment<SegN>` analogue: a *statically sized* view,
/// which is what makes `LVec<T,Size> v = seg.as_eigen();` (the reference's
/// `Eigen::Vector<T,SegN> v = seg.as_eigen();`) work.  `Size == 0` is dynamic.
template <bool IsConst, typename T, typename Storage = storage::buffers, int Size = 0>
class DenseVectorViewT
{
    template <typename U>
    using auto_const_t = std::conditional_t<IsConst, const U, U>;

    template <bool OtherIsConst, typename U, typename S_, int Sz_>
    friend class DenseVectorViewT;

  public:
    static_assert(!std::is_const_v<T>, "linalg: the element type must be non-const");

    using storage_type = Storage;
    using ScalarT      = buffer_scalar_of_t<T>;
    using index_type   = typename Storage::index_type;
    using value_t      = storage::loaded_t<ScalarT, Storage>;  // double | lcrt::Expr<double>
    using buffer_type  = typename Storage::template handle<ScalarT>;

    static constexpr int  static_size     = Size;
    static constexpr bool has_static_size = (Size > 0);

    using ConstView    = DenseVectorViewT<true, T, Storage, Size>;
    using NonConstView = DenseVectorViewT<false, T, Storage, Size>;
    using ThisView     = DenseVectorViewT<IsConst, T, Storage, Size>;
    /// The dynamic (whole-vector) view: what `LDenseVector::view()` returns.
    using DynamicView = DenseVectorViewT<IsConst, T, Storage, 0>;

    /// Element proxy: for a non-const view an assignable lvalue stand-in
    /// (the reference returned `T&`), for a const view a read-only value.
    class ElementRef
    {
        friend class DenseVectorViewT;
        ThisView   m_view;
        index_type m_index{};

        ElementRef(const ThisView& view, index_type index) noexcept
            : m_view(view)
            , m_index(index)
        {
        }

      public:
        [[nodiscard]] value_t read() const noexcept { return m_view.read(m_index); }
        [[nodiscard]] operator value_t() const noexcept { return read(); }

        template <typename V>
            requires(!IsConst)
        ElementRef& operator=(V&& v) noexcept
        {
            m_view.write(m_index, std::forward<V>(v));
            return *this;
        }
        template <typename V>
            requires(!IsConst)
        ElementRef& operator+=(V&& v) noexcept
        {
            m_view.write(m_index, m_view.read(m_index) + std::forward<V>(v));
            return *this;
        }
        template <typename V>
            requires(!IsConst)
        ElementRef& operator-=(V&& v) noexcept
        {
            m_view.write(m_index, m_view.read(m_index) - std::forward<V>(v));
            return *this;
        }
    };

  protected:
    buffer_type m_data{};
    index_type  m_offset{};
    index_type  m_size{};
    index_type  m_origin_size{};

    std::shared_ptr<const detail::generation_cell> m_lifetime{};
    std::uint64_t                                  m_generation = 0u;

  public:
    DenseVectorViewT() = default;

    DenseVectorViewT(buffer_type data, index_type offset, index_type size, index_type origin_size)
        : m_data(data)
        , m_offset(offset)
        , m_size(size)
        , m_origin_size(origin_size)
    {
    }

    /// Device-side factory (a whole dense vector, no window offset).
    template <typename S>
    [[nodiscard]] static ThisView from_handles(buffer_type data, S&& size)
    {
        const auto n = Storage::as_index(std::forward<S>(size));
        return ThisView{data, Storage::from_int(0), n, n};
    }

    template <bool OtherIsConst>
    DenseVectorViewT(const DenseVectorViewT<OtherIsConst, T, Storage, Size>& other) noexcept
        requires(IsConst)
        : m_data(other.m_data)
        , m_offset(other.m_offset)
        , m_size(other.m_size)
        , m_origin_size(other.m_origin_size)
        , m_lifetime(other.m_lifetime)
        , m_generation(other.m_generation)
    {
    }

    void set_lifetime(std::shared_ptr<const detail::generation_cell> cell,
                      std::uint64_t                                  generation) noexcept
    {
        m_lifetime   = std::move(cell);
        m_generation = generation;
    }
    [[nodiscard]] std::uint64_t generation() const noexcept { return m_generation; }
    void                        use_check() const
    {
        if(m_lifetime && m_lifetime->value != m_generation)
        {
            throw std::runtime_error{
                "linalg: view used across a resize (plan risk R5) — views are frame-scoped"};
        }
    }

    // -- metadata -----------------------------------------------------------
    [[nodiscard]] index_type offset() const noexcept { return m_offset; }
    [[nodiscard]] index_type size() const noexcept { return m_size; }
    [[nodiscard]] index_type origin_size() const noexcept { return m_origin_size; }
    [[nodiscard]] index_type origin_offset() const noexcept { return Storage::from_int(0); }

    /// The handle that covers this view (the reference's `buffer_view()`, which
    /// carries the window offset in its own header).
    [[nodiscard]] auto buffer_view() const noexcept
    {
        if constexpr(Storage::is_device)
        {
            // The handle already covers the whole buffer argument; the window
            // offset lives in the index arithmetic.
            return m_data;
        }
        else
        {
            return Storage::subview(m_data,
                                    static_cast<std::size_t>(m_offset),
                                    static_cast<std::size_t>(m_size));
        }
    }
    [[nodiscard]] auto_const_t<buffer_type> data_handle() const noexcept { return m_data; }

    // -- sub views -----------------------------------------------------------
    template <typename O, typename S>
    [[nodiscard]] ThisView subview(O&& offset, S&& size) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(offset >= 0 && offset + size <= m_size,
                                  "DenseVectorView: subview out of range");
        }
        return ThisView{m_data,
                        m_offset + Storage::as_index(std::forward<O>(offset)),
                        Storage::as_index(std::forward<S>(size)),
                        m_origin_size};
    }
    template <typename O, typename S>
    [[nodiscard]] ThisView segment(O&& offset, S&& size) const
    {
        return subview(std::forward<O>(offset), std::forward<S>(size));
    }

    /// Eigen's `segment<SegN>(offset)`: a statically sized subview.  The offset is
    /// templated so a kernel can segment at a device index.
    template <int SegN, typename O>
    [[nodiscard]] DenseVectorViewT<IsConst, T, Storage, SegN> segment(O&& offset) const
    {
        static_assert(SegN > 0, "linalg: segment<SegN> needs SegN > 0");
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(offset >= 0 && offset + SegN <= m_size,
                                  "DenseVectorView: segment out of range");
        }
        return DenseVectorViewT<IsConst, T, Storage, SegN>{
            m_data,
            m_offset + Storage::as_index(std::forward<O>(offset)),
            Storage::from_int(SegN),
            m_origin_size};
    }

    // -- element access ------------------------------------------------------
    template <typename I>
    [[nodiscard]] value_t read(I&& i) const noexcept
    {
        return Storage::template load<ScalarT>(m_data,
                                              m_offset + Storage::as_index(std::forward<I>(i)));
    }
    template <typename I, typename V>
        requires(!IsConst)
    void write(I&& i, V&& v) const noexcept
    {
        Storage::template store<ScalarT>(m_data,
                                         m_offset + Storage::as_index(std::forward<I>(i)),
                                         std::forward<V>(v));
    }

    /// A non-const view hands out an assignable element proxy (the reference's
    /// `T&`); a const view hands out the loaded value (the reference's `const
    /// T&`) — there is no write path on a `CView` at all.
    template <typename I>
    [[nodiscard]] decltype(auto) operator()(I&& i) const noexcept
    {
        if constexpr(IsConst)
        {
            return read(std::forward<I>(i));
        }
        else
        {
            return ElementRef{*this, Storage::as_index(std::forward<I>(i))};
        }
    }

    /// `Eigen::Vector<T,N> v = view.as_eigen();` — the view itself carries the
    /// map semantics (element access, assignment from an LVec, compound
    /// assignment), so `as_eigen()` is the identity.  Materialise with
    /// `to_vec<SegN>()`/the implicit conversion before using the math layer.
    [[nodiscard]] ThisView as_eigen() const noexcept { return *this; }

    /// Materialise into an `LVec` value (the Eigen `Vector<T,N> v = map;` step).
    template <int SegN>
    [[nodiscard]] LVec<T, SegN> to_vec() const
    {
        static_assert(SegN > 0, "linalg: to_vec<N> needs N > 0");
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(m_size == SegN, "DenseVectorView: to_vec size mismatch");
        }
        return lmath::make_vec<T, SegN>([&](int i) { return read(i); });
    }

    /// Implicit conversion for the statically sized flavour (`LVec<T,Size> v =
    /// seg.as_eigen();`).  Written as a template so that the dynamic flavour
    /// (Size == 0) never instantiates `LVec<T,0>`.
    template <int N = Size>
        requires(N > 0)
    [[nodiscard]] operator LVec<T, N>() const
    {
        return to_vec<N>();
    }

    /// Whole-vector assignment from a vector-like value (non-const views only).
    template <typename V>
        requires(!IsConst && lmath::is_vec_like_v<V>)
    ThisView& operator=(const V& v)
    {
        assign_from(v);
        return *this;
    }
    template <typename V>
        requires(!IsConst && lmath::is_vec_like_v<V>)
    ThisView& operator+=(const V& v)
    {
        compound_from(v, true);
        return *this;
    }
    template <typename V>
        requires(!IsConst && lmath::is_vec_like_v<V>)
    ThisView& operator-=(const V& v)
    {
        compound_from(v, false);
        return *this;
    }

    // -- atomics -------------------------------------------------------------
    template <typename I, typename V>
        requires(!IsConst)
    [[nodiscard]] auto atomic_add(I&& i, V&& v) const noexcept
    {
        return Storage::atomic_add_at(m_data,
                                      m_offset + Storage::as_index(std::forward<I>(i)),
                                      std::forward<V>(v));
    }

    /// `template <int SegN> atomic_add(const LVec<T,SegN>&)` (the reference's
    /// `atomic_add(const Eigen::Vector<T,SegN>&)`).
    template <int SegN>
        requires(!IsConst && SegN > 0)
    void atomic_add(const LVec<T, SegN>& value) const
    {
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(m_size == SegN, "DenseVectorView: atomic_add size mismatch");
        }
        for(int i = 0; i < SegN; ++i)
        {
            (void)atomic_add(i, value(i));
        }
    }

    /// `atomic_add(value)` for a scalar dense vector of size 1 (scalars only:
    /// the segmented form above takes an `LVec`).
    template <typename V>
        requires(!IsConst && lmath::is_scalar_like_v<V>)
    void atomic_add(V&& value) const
    {
        (void)atomic_add(0, std::forward<V>(value));
    }

  private:
    template <typename V>
    void assign_from(const V& v)
    {
        constexpr int Dim = static_cast<int>(lmath::vec_dim_v<V>);
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(m_size == static_cast<index_type>(Dim),
                                  "DenseVectorView: assignment size mismatch");
        }
        for(int i = 0; i < Dim; ++i)
        {
            write(i, v(i));
        }
    }
    template <typename V>
    void compound_from(const V& v, bool add)
    {
        constexpr int Dim = static_cast<int>(lmath::vec_dim_v<V>);
        if constexpr(!Storage::is_device)
        {
            detail::kernel_assert(m_size == static_cast<index_type>(Dim),
                                  "DenseVectorView: compound assignment size mismatch");
        }
        for(int i = 0; i < Dim; ++i)
        {
            if(add)
            {
                write(i, read(i) + v(i));
            }
            else
            {
                write(i, read(i) - v(i));
            }
        }
    }
};

// ---------------------------------------------------------------------------
//  BSR matrix view (addition — the reference exposes no BSR view)
// ---------------------------------------------------------------------------
template <bool IsConst, typename T, int N = 1, typename Storage = storage::buffers>
class BSRMatrixViewT
{
    template <typename U>
    using auto_const_t = std::conditional_t<IsConst, const U, U>;

    template <bool OtherIsConst, typename U, int N_, typename S_>
    friend class BSRMatrixViewT;

  public:
    static_assert(!std::is_const_v<T>, "linalg: the element type must be non-const");
    static_assert(N > 0);

    using storage_type = Storage;
    using traits       = block_traits<T, N, N>;
    using ValueT       = typename traits::value_type;  // T | LMat<T,N,N>
    using ScalarT      = typename traits::scalar_type;
    using index_type   = typename Storage::index_type;

    static constexpr bool IsBlockMatrix = (N > 1);
    static constexpr int  block_size    = N;

    using row_offset_buffer_type = typename Storage::template handle<std::int32_t>;
    using col_index_buffer_type  = typename Storage::template handle<std::int32_t>;
    using value_buffer_type      = typename Storage::template handle<ScalarT>;

    using ConstView    = BSRMatrixViewT<true, T, N, Storage>;
    using NonConstView = BSRMatrixViewT<false, T, N, Storage>;
    using ThisView     = BSRMatrixViewT<IsConst, T, N, Storage>;

  protected:
    index_type m_rows{};
    index_type m_cols{};
    index_type m_non_zeros{};

    row_offset_buffer_type m_row_offsets{};
    col_index_buffer_type  m_col_indices{};
    value_buffer_type      m_values{};

    std::shared_ptr<const detail::generation_cell> m_lifetime{};
    std::uint64_t                                  m_generation = 0u;

  public:
    BSRMatrixViewT() = default;

    BSRMatrixViewT(index_type             rows,
                   index_type             cols,
                   index_type             non_zero_blocks,
                   row_offset_buffer_type row_offsets,
                   col_index_buffer_type  col_indices,
                   value_buffer_type      values)
        : m_rows(rows)
        , m_cols(cols)
        , m_non_zeros(non_zero_blocks)
        , m_row_offsets(row_offsets)
        , m_col_indices(col_indices)
        , m_values(values)
    {
    }

    /// Device-side factory (mirrors the container's accessor names).
    template <typename R, typename C, typename N>
    [[nodiscard]] static ThisView from_handles(row_offset_buffer_type row_offsets,
                                               col_index_buffer_type  col_indices,
                                               value_buffer_type      values,
                                               R&&                    rows,
                                               C&&                    cols,
                                               N&&                    non_zero_blocks)
    {
        return ThisView{Storage::as_index(std::forward<R>(rows)),
                        Storage::as_index(std::forward<C>(cols)),
                        Storage::as_index(std::forward<N>(non_zero_blocks)),
                        row_offsets,
                        col_indices,
                        values};
    }

    template <bool OtherIsConst>
    BSRMatrixViewT(const BSRMatrixViewT<OtherIsConst, T, N, Storage>& other) noexcept
        requires(IsConst)
        : m_rows(other.m_rows)
        , m_cols(other.m_cols)
        , m_non_zeros(other.m_non_zeros)
        , m_row_offsets(other.m_row_offsets)
        , m_col_indices(other.m_col_indices)
        , m_values(other.m_values)
        , m_lifetime(other.m_lifetime)
        , m_generation(other.m_generation)
    {
    }

    // -- frame lifetime ------------------------------------------------------
    void set_lifetime(std::shared_ptr<const detail::generation_cell> cell,
                      std::uint64_t                                  generation) noexcept
    {
        m_lifetime   = std::move(cell);
        m_generation = generation;
    }
    [[nodiscard]] std::uint64_t generation() const noexcept { return m_generation; }
    void                        use_check() const
    {
        if(m_lifetime && m_lifetime->value != m_generation)
        {
            throw std::runtime_error{
                "linalg: view used across a resize (plan risk R5) — views are frame-scoped"};
        }
    }

    // -- metadata -----------------------------------------------------------
    [[nodiscard]] index_type rows() const noexcept { return m_rows; }
    [[nodiscard]] index_type cols() const noexcept { return m_cols; }
    [[nodiscard]] index_type non_zeros() const noexcept { return m_non_zeros; }
    [[nodiscard]] index_type block_size_value() const noexcept { return Storage::from_int(N); }

    [[nodiscard]] auto_const_t<row_offset_buffer_type> row_offsets() const noexcept
    {
        return m_row_offsets;
    }
    [[nodiscard]] auto_const_t<col_index_buffer_type> col_indices() const noexcept
    {
        return m_col_indices;
    }
    [[nodiscard]] auto_const_t<value_buffer_type> values() const noexcept { return m_values; }

    [[nodiscard]] ThisView  viewer() const noexcept { return *this; }
    [[nodiscard]] ConstView cviewer() const noexcept { return as_const(); }
    [[nodiscard]] ConstView as_const() const noexcept { return ConstView{*this}; }

    // -- row walk (device-usable) -------------------------------------------
    template <typename R>
    [[nodiscard]] auto row_begin(R&& row) const noexcept
    {
        return Storage::template load<std::int32_t>(m_row_offsets, std::forward<R>(row));
    }
    template <typename R>
    [[nodiscard]] auto row_end(R&& row) const noexcept
    {
        return Storage::template load<std::int32_t>(m_row_offsets, std::forward<R>(row) + 1u);
    }
    template <typename R>
    [[nodiscard]] auto row_non_zeros(R&& row) const noexcept
    {
        return row_end(std::forward<R>(row)) - row_begin(std::forward<R>(row));
    }
    template <typename K>
    [[nodiscard]] auto col_index(K&& k) const noexcept
    {
        return Storage::template load<std::int32_t>(m_col_indices, std::forward<K>(k));
    }

    /// Block `k` (the flat block index of the BSR arrays) as `LMat<T,N,N>`.
    template <typename K>
    [[nodiscard]] auto load_block(K&& k) const noexcept
    {
        if constexpr(N == 1)
        {
            return Storage::template load<ScalarT>(m_values, std::forward<K>(k));
        }
        else
        {
            ValueT out{};
            for(int c = 0; c < N; ++c)
            {
                for(int r = 0; r < N; ++r)
                {
                    out(r, c) = Storage::template load<ScalarT>(
                        m_values,
                        std::forward<K>(k) * static_cast<std::uint32_t>(N * N)
                            + static_cast<std::uint32_t>(c * N + r));
                }
            }
            return out;
        }
    }

    template <typename K>
    void store_block(K&& k, const ValueT& block) const noexcept
    {
        if constexpr(N == 1)
        {
            Storage::template store<ScalarT>(m_values, std::forward<K>(k), block);
        }
        else
        {
            for(int c = 0; c < N; ++c)
            {
                for(int r = 0; r < N; ++r)
                {
                    Storage::template store<ScalarT>(
                        m_values,
                        std::forward<K>(k) * static_cast<std::uint32_t>(N * N)
                            + static_cast<std::uint32_t>(c * N + r),
                        block(r, c));
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
//  Dense matrix view (addition — the reference ports no DenseMatrix view)
// ---------------------------------------------------------------------------
/// Row-major, contiguous, mirroring `DeviceDenseMatrix` (row-major per its own
/// comment), plus the `sym` flag on the container.
template <bool IsConst, typename T, typename Storage = storage::buffers>
class DenseMatrixViewT
{
    template <typename U>
    using auto_const_t = std::conditional_t<IsConst, const U, U>;

    template <bool OtherIsConst, typename U, typename S_>
    friend class DenseMatrixViewT;

  public:
    static_assert(!std::is_const_v<T>, "linalg: the element type must be non-const");

    using storage_type = Storage;
    using ScalarT      = buffer_scalar_of_t<T>;
    using index_type   = typename Storage::index_type;
    using value_t      = storage::loaded_t<ScalarT, Storage>;
    using buffer_type  = typename Storage::template handle<ScalarT>;

    using ConstView    = DenseMatrixViewT<true, T, Storage>;
    using NonConstView = DenseMatrixViewT<false, T, Storage>;
    using ThisView     = DenseMatrixViewT<IsConst, T, Storage>;

    class ElementRef
    {
        friend class DenseMatrixViewT;
        ThisView   m_view;
        index_type m_index{};

        ElementRef(const ThisView& view, index_type index) noexcept
            : m_view(view)
            , m_index(index)
        {
        }

      public:
        [[nodiscard]] value_t read() const noexcept { return m_view.read_flat(m_index); }
        [[nodiscard]] operator value_t() const noexcept { return read(); }

        template <typename V>
            requires(!IsConst)
        ElementRef& operator=(V&& v) noexcept
        {
            m_view.write_flat(m_index, std::forward<V>(v));
            return *this;
        }
    };

  protected:
    buffer_type m_data{};
    index_type  m_rows{};
    index_type  m_cols{};

    std::shared_ptr<const detail::generation_cell> m_lifetime{};
    std::uint64_t                                  m_generation = 0u;

    template <typename I>
    [[nodiscard]] value_t read_flat(I&& flat_index) const noexcept
    {
        return Storage::template load<ScalarT>(m_data, std::forward<I>(flat_index));
    }
    template <typename I, typename V>
        requires(!IsConst)
    void write_flat(I&& flat_index, V&& v) const noexcept
    {
        Storage::template store<ScalarT>(m_data, std::forward<I>(flat_index), std::forward<V>(v));
    }

  public:
    DenseMatrixViewT() = default;

    DenseMatrixViewT(buffer_type data, index_type rows, index_type cols)
        : m_data(data)
        , m_rows(rows)
        , m_cols(cols)
    {
    }

    template <typename R, typename C>
    [[nodiscard]] static ThisView from_handles(buffer_type data, R&& rows, C&& cols)
    {
        return ThisView{data,
                        Storage::as_index(std::forward<R>(rows)),
                        Storage::as_index(std::forward<C>(cols))};
    }

    template <bool OtherIsConst>
    DenseMatrixViewT(const DenseMatrixViewT<OtherIsConst, T, Storage>& other) noexcept
        requires(IsConst)
        : m_data(other.m_data)
        , m_rows(other.m_rows)
        , m_cols(other.m_cols)
        , m_lifetime(other.m_lifetime)
        , m_generation(other.m_generation)
    {
    }

    void set_lifetime(std::shared_ptr<const detail::generation_cell> cell,
                      std::uint64_t                                  generation) noexcept
    {
        m_lifetime   = std::move(cell);
        m_generation = generation;
    }
    [[nodiscard]] std::uint64_t generation() const noexcept { return m_generation; }
    void                        use_check() const
    {
        if(m_lifetime && m_lifetime->value != m_generation)
        {
            throw std::runtime_error{"linalg: view used across a resize (plan risk R5)"};
        }
    }

    [[nodiscard]] index_type rows() const noexcept { return m_rows; }
    [[nodiscard]] index_type cols() const noexcept { return m_cols; }
    [[nodiscard]] index_type size() const noexcept { return m_rows * m_cols; }
    [[nodiscard]] auto_const_t<buffer_type> buffer_view() const noexcept { return m_data; }
    [[nodiscard]] auto_const_t<buffer_type> data_handle() const noexcept { return m_data; }

    [[nodiscard]] ThisView  viewer() const noexcept { return *this; }
    [[nodiscard]] ConstView cviewer() const noexcept { return as_const(); }
    [[nodiscard]] ConstView as_const() const noexcept { return ConstView{*this}; }

    template <typename R, typename C>
    [[nodiscard]] value_t read(R&& r, C&& c) const noexcept
    {
        return read_flat(row_major_index(std::forward<R>(r), std::forward<C>(c)));
    }
    template <typename R, typename C, typename V>
        requires(!IsConst)
    void write(R&& r, C&& c, V&& v) const noexcept
    {
        write_flat(row_major_index(std::forward<R>(r), std::forward<C>(c)), std::forward<V>(v));
    }

    /// Flat (row-major) element access, the way a kernel walks a dense matrix.
    template <typename I>
    [[nodiscard]] decltype(auto) operator()(I&& flat_index) const noexcept
    {
        if constexpr(IsConst)
        {
            return read_flat(std::forward<I>(flat_index));
        }
        else
        {
            return ElementRef{*this, Storage::as_index(std::forward<I>(flat_index))};
        }
    }

    template <typename R, typename C>
    [[nodiscard]] auto row_major_index(R&& r, C&& c) const noexcept
    {
        return std::forward<R>(r) * m_cols + std::forward<C>(c);
    }
};

// ---------------------------------------------------------------------------
//  Reference-compatible aliases
// ---------------------------------------------------------------------------
// Host flavour (what consumer code and the containers use).
template <typename T, int M, int N = M>
using LTripletMatrixView = TripletMatrixViewT<false, T, M, N, storage::buffers>;
template <typename T, int M, int N = M>
using LCTripletMatrixView = TripletMatrixViewT<true, T, M, N, storage::buffers>;
template <typename T, int N = 1>
using LDoubletVectorView = DoubletVectorViewT<false, T, N, storage::buffers>;
template <typename T, int N = 1>
using LCDoubletVectorView = DoubletVectorViewT<true, T, N, storage::buffers>;
template <typename T, int N = 1>
using LBSRMatrixView = BSRMatrixViewT<false, T, N, storage::buffers>;
template <typename T, int N = 1>
using LCBSRMatrixView = BSRMatrixViewT<true, T, N, storage::buffers>;
template <typename T, int Size = 0>
using LDenseVectorView = DenseVectorViewT<false, T, storage::buffers, Size>;
template <typename T, int Size = 0>
using LCDenseVectorView = DenseVectorViewT<true, T, storage::buffers, Size>;
template <typename T>
using LDenseMatrixView = DenseMatrixViewT<false, T, storage::buffers>;
template <typename T>
using LCDenseMatrixView = DenseMatrixViewT<true, T, storage::buffers>;

// Device flavour (built inside a kernel from the kernel's arguments).
template <typename T, int M, int N = M>
using LTripletMatrixDeviceView = TripletMatrixViewT<false, T, M, N, storage::device>;
template <typename T, int M, int N = M>
using LCTripletMatrixDeviceView = TripletMatrixViewT<true, T, M, N, storage::device>;
template <typename T, int N = 1>
using LDoubletVectorDeviceView = DoubletVectorViewT<false, T, N, storage::device>;
template <typename T, int N = 1>
using LCDoubletVectorDeviceView = DoubletVectorViewT<true, T, N, storage::device>;
template <typename T, int N = 1>
using LBSRMatrixDeviceView = BSRMatrixViewT<false, T, N, storage::device>;
template <typename T, int N = 1>
using LCBSRMatrixDeviceView = BSRMatrixViewT<true, T, N, storage::device>;
template <typename T, int Size = 0>
using LDenseVectorDeviceView = DenseVectorViewT<false, T, storage::device, Size>;
template <typename T, int Size = 0>
using LCDenseVectorDeviceView = DenseVectorViewT<true, T, storage::device, Size>;
template <typename T>
using LDenseMatrixDeviceView = DenseMatrixViewT<false, T, storage::device>;
template <typename T>
using LCDenseMatrixDeviceView = DenseMatrixViewT<true, T, storage::device>;

// Host-memory flavour (unit tests / host reference evaluation).
template <typename T, int M, int N = M>
using LTripletMatrixHostMemoryView = TripletMatrixViewT<false, T, M, N, storage::memory>;
template <typename T, int M, int N = M>
using LCTripletMatrixHostMemoryView = TripletMatrixViewT<true, T, M, N, storage::memory>;
template <typename T, int N = 1>
using LDoubletVectorHostMemoryView = DoubletVectorViewT<false, T, N, storage::memory>;
template <typename T, int N = 1>
using LCDoubletVectorHostMemoryView = DoubletVectorViewT<true, T, N, storage::memory>;
template <typename T, int N = 1>
using LBSRMatrixHostMemoryView = BSRMatrixViewT<false, T, N, storage::memory>;
template <typename T, int Size = 0>
using LDenseVectorHostMemoryView = DenseVectorViewT<false, T, storage::memory, Size>;
template <typename T, int Size = 0>
using LCDenseVectorHostMemoryView = DenseVectorViewT<true, T, storage::memory, Size>;
template <typename T>
using LDenseMatrixHostMemoryView = DenseMatrixViewT<false, T, storage::memory>;

// The reference's own spellings (`cuda_tool::TripletMatrixView<Float,3>`, ...).
template <typename Ty, int M, int N = M>
using TripletMatrixView = LTripletMatrixView<Ty, M, N>;
template <typename Ty, int M, int N = M>
using CTripletMatrixView = LCTripletMatrixView<Ty, M, N>;
template <typename Ty, int M, int N = M>
using TripletMatrixViewer = LTripletMatrixView<Ty, M, N>;
template <typename Ty, int M, int N = M>
using CTripletMatrixViewer = LCTripletMatrixView<Ty, M, N>;
template <typename T, int M, int N = M>
using BCOOMatrixView = LTripletMatrixView<T, M, N>;
template <typename T, int M, int N = M>
using CBCOOMatrixView = LCTripletMatrixView<T, M, N>;

template <typename T, int N = 1>
using DoubletVectorView = LDoubletVectorView<T, N>;
template <typename T, int N = 1>
using CDoubletVectorView = LCDoubletVectorView<T, N>;
template <typename T, int N = 1>
using DoubletVectorViewer = LDoubletVectorView<T, N>;
template <typename T, int N = 1>
using CDoubletVectorViewer = LCDoubletVectorView<T, N>;
template <typename T, int N = 1>
using BCOOVectorView = LDoubletVectorView<T, N>;
template <typename T, int N = 1>
using CBCOOVectorView = LCDoubletVectorView<T, N>;

template <typename T, int Size = 0>
using DenseVectorView = LDenseVectorView<T, Size>;
template <typename T, int Size = 0>
using CDenseVectorView = LCDenseVectorView<T, Size>;
template <typename T, int Size = 0>
using DenseVectorViewer = LDenseVectorView<T, Size>;
template <typename T, int Size = 0>
using CDenseVectorViewer = LCDenseVectorView<T, Size>;

template <typename T, int N = 1>
using BSRMatrixView = LBSRMatrixView<T, N>;
template <typename T, int N = 1>
using CBSRMatrixView = LCBSRMatrixView<T, N>;
template <typename T>
using DenseMatrixView = LDenseMatrixView<T>;
template <typename T>
using CDenseMatrixView = LCDenseMatrixView<T>;
}// namespace uipc::backend::luisa::luisa_device::linalg

namespace uipc::backend::luisa::luisa_device::math
{
// The linear views are *not* registered as vec_like/mat_like: their element
// access is a buffer load (a DSL expression), and the math layer's operators need
// a statically shaped value to work with (RULE M1 also forbids a second
// arithmetic-operator namespace).  Materialise with `.to_vec<N>()` first — see
// the header of linear/views.h.  `select` must still keep its hands off them.
template <bool C, typename T, typename S, int Sz>
struct is_container_like<linalg::DenseVectorViewT<C, T, S, Sz>> : std::true_type
{
};
template <bool C, typename T, int M, int N, typename S>
struct is_container_like<linalg::TripletMatrixViewT<C, T, M, N, S>> : std::true_type
{
};
template <bool C, typename T, int N, typename S>
struct is_container_like<linalg::DoubletVectorViewT<C, T, N, S>> : std::true_type
{
};
template <bool C, typename T, int N, typename S>
struct is_container_like<linalg::BSRMatrixViewT<C, T, N, S>> : std::true_type
{
};
template <bool C, typename T, typename S>
struct is_container_like<linalg::DenseMatrixViewT<C, T, S>> : std::true_type
{
};
}// namespace uipc::backend::luisa::luisa_device::math

namespace uipc::backend::luisa::luisa_device
{
/// Short alias used by ported kernel code (`lla::LCTripletMatrixDeviceView<...>`).
namespace lla = linalg;
}// namespace uipc::backend::luisa::luisa_device
