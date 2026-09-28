#pragma once
/********************************************************************
 * @file containers.h
 * @brief T3.1 — the owning linear-system containers: the LuisaCompute port of
 *        src/backends/cuda/cuda_tool/linear_system.h.
 *
 * `LTripletMatrix` / `LBCOOMatrix` / `LBSRMatrix` / `LDoubletVector` /
 * `LBCOOVector` / `LDenseVector` / `LDenseMatrix` own the `LDeviceBuffer`s that
 * back the views in linear/views.h, with the reference's member surface:
 * reshape/resize/reserve/clear, the `*_discard` variants, `xx_count()` /
 * `xx_capacity()`, `view()/cview()/viewer()/cviewer()`, the implicit view
 * conversion, and the raw owning-buffer accessors.
 *
 * Two deliberate deviations from the reference implementation (both are
 * consequences of the flat block storage — see linear/views.h):
 *
 *  1. `values()` hands out the *flat scalar* buffer (`nnz*M*N` elements,
 *     column-major inside a block) instead of a block-typed view; the reference's
 *     raw `values().data()` pointer arithmetic is not portable to a buffer DSL.
 *  2. `resize_*_discard()` cannot skip the value initialisation (LC buffers have
 *     no "discard" resize); the newly exposed tail is zeroed like the non-discard
 *     path.  Consumers overwrite the entries they expose, so this is a superset
 *     of the reference behaviour, not a semantic change.
 *
 * Not ported (see linear.h): `details::LinearReductionStorage`,
 * `LinearSystemContext`, cusparse descriptors.
 *********************************************************************/
#include <luisa_device/buffer.h>
#include <luisa_device/linear/views.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace uipc::backend::luisa::luisa_device::linalg
{
// ---------------------------------------------------------------------------
//  Host-side descriptors (the reference's MatrixTriplet / VectorDoublet)
// ---------------------------------------------------------------------------
/// A single sparse-matrix entry.  A block (M,N > 1) stores `LMat<T,M,N>` — the
/// column-major, Eigen-layout replacement for `Eigen::Matrix<T,M,N>`.
template <typename T, int M, int N = M>
struct MatrixTriplet
{
    using ValueT = typename block_traits<T, M, N>::value_type;

    int    row_index = 0;
    int    col_index = 0;
    ValueT value{};
};

/// A single segment-vector entry (`Eigen::Vector<T,N>` -> `LVec<T,N>`).
template <typename T, int N = 1>
struct VectorDoublet
{
    using ValueT = std::conditional_t<(N == 1), T, LVec<T, N>>;

    int    index = 0;
    ValueT value{};
};

/// Logical block/segment type of the `<T,N>` container family.
template <typename T, int M, int N>
using container_value_t = typename block_traits<T, M, N>::value_type;

// ---------------------------------------------------------------------------
//  LTripletMatrix (COO, possibly block)
// ---------------------------------------------------------------------------
template <typename T, int M = 1, int N = M>
class LTripletMatrix
{
  public:
    static_assert(std::is_same_v<T, buffer_scalar_of_t<T>>,
                  "linalg: container element type must be a plain scalar "
                  "(double/float), not a DSL value");

    using traits       = block_traits<T, M, N>;
    using ValueT       = typename traits::value_type;
    using ScalarT      = typename traits::scalar_type;
    using TripletT     = MatrixTriplet<T, M, N>;
    using view_type    = LTripletMatrixView<T, M, N>;
    using const_view_type = LCTripletMatrixView<T, M, N>;

    static constexpr bool IsBlockMatrix = (M > 1 || N > 1);
    static constexpr int  block_rows    = M;
    static constexpr int  block_cols    = N;
    static constexpr int  block_entries = M * N;

    // -- shape / allocation (reference surface) ------------------------------
    void reshape(int rows, int cols)
    {
        m_rows = rows;
        m_cols = cols;
        invalidate();
    }
    void resize_triplets(std::size_t n)
    {
        // indices are kept in lock-step with values
        m_values.resize(n * static_cast<std::size_t>(block_entries));
        m_row_indices.resize(n);
        m_col_indices.resize(n);
        invalidate();
    }
    /// See the header note: the tail is zeroed, not left uninitialised.
    void resize_triplets_discard(std::size_t n) { resize_triplets(n); }
    void reserve_triplets(std::size_t n)
    {
        m_values.reserve(n * static_cast<std::size_t>(block_entries));
        m_row_indices.reserve(n);
        m_col_indices.reserve(n);
        invalidate();
    }
    void reserve_triplets_discard(std::size_t n) { reserve_triplets(n); }
    void resize(int rows, int cols, std::size_t n)
    {
        reshape(rows, cols);
        resize_triplets(n);
    }
    void resize_discard(int rows, int cols, std::size_t n)
    {
        reshape(rows, cols);
        resize_triplets_discard(n);
    }
    void clear()
    {
        m_rows = m_cols = 0;
        m_values.clear();
        m_row_indices.clear();
        m_col_indices.clear();
        invalidate();
    }
    /// Release the allocations entirely (luisa_device extension).
    void reset()
    {
        m_rows = m_cols = 0;
        m_values.reset();
        m_row_indices.reset();
        m_col_indices.reset();
        invalidate();
    }

    static constexpr int block_dim() { return N; }

    // -- metadata -----------------------------------------------------------
    int    rows() const { return m_rows; }
    int    cols() const { return m_cols; }
    std::size_t triplet_count() const { return m_row_indices.size(); }
    std::size_t triplet_capacity() const
    {
        return m_row_indices.capacity();
    }
    int non_zeros() const { return static_cast<int>(triplet_count()); }

    // -- handles ------------------------------------------------------------
    LBufferView<std::int32_t>  row_indices() { return m_row_indices.view(); }
    LCBufferView<std::int32_t> row_indices() const { return m_row_indices.cview(); }
    LBufferView<std::int32_t>  col_indices() { return m_col_indices.view(); }
    LCBufferView<std::int32_t> col_indices() const { return m_col_indices.cview(); }
    /// Flat scalar storage: `nnz * M * N` elements, column-major inside a block.
    LBufferView<ScalarT>  values() { return m_values.view(); }
    LCBufferView<ScalarT> values() const { return m_values.cview(); }
    static constexpr int  value_entries_per_triplet() { return block_entries; }

    // -- views --------------------------------------------------------------
    view_type view() { return make_view(); }
    const_view_type view() const
    {
        return const_cast<LTripletMatrix*>(this)->make_view();
    }
    const_view_type cview() const { return view(); }

    view_type       viewer() { return view(); }
    const_view_type cviewer() const { return view(); }

    operator view_type() { return view(); }
    operator const_view_type() const { return view(); }

    // -- raw owning-buffer access (cuda_tool extension, kept for parity) -----
    LDeviceBuffer<std::int32_t>& row_index_buffer() { return m_row_indices; }
    LDeviceBuffer<std::int32_t>& col_index_buffer() { return m_col_indices; }
    LDeviceBuffer<ScalarT>&      values_buffer() { return m_values; }

    // -- host <-> device helpers (luisa_device additions) --------------------
    /// Pack host triplets into the flat storage (the port's replacement for the
    /// reference's host `cudaMemcpy` of a triplet array).
    void copy_from_triplets(std::span<const TripletT> triplets)
    {
        const auto n = triplets.size();
        resize_triplets(n);
        if(n == 0u)
        {
            return;
        }
        std::vector<std::int32_t> rows(n);
        std::vector<std::int32_t> cols(n);
        std::vector<ScalarT>      flat(n * static_cast<std::size_t>(block_entries));
        for(std::size_t i = 0; i < n; ++i)
        {
            rows[i] = triplets[i].row_index;
            cols[i] = triplets[i].col_index;
            const auto& block = triplets[i].value;
            if constexpr(M == 1 && N == 1)
            {
                flat[i] = static_cast<ScalarT>(block);
            }
            else
            {
                for(int c = 0; c < N; ++c)
                    for(int r = 0; r < M; ++r)
                        flat[i * static_cast<std::size_t>(block_entries)
                             + static_cast<std::size_t>(c * M + r)] =
                            static_cast<ScalarT>(block(r, c));
            }
        }
        m_row_indices.copy_from(std::span<const std::int32_t>{rows});
        m_col_indices.copy_from(std::span<const std::int32_t>{cols});
        m_values.copy_from(std::span<const ScalarT>{flat});
        invalidate();
    }

    /// Unpack the device data into host triplets.
    void copy_to_triplets(std::span<TripletT> triplets) const
    {
        const auto n = triplet_count();
        if(triplets.size() != n)
        {
            throw std::invalid_argument{"LTripletMatrix::copy_to_triplets size mismatch"};
        }
        if(n == 0u)
        {
            return;
        }
        std::vector<std::int32_t> rows(n), cols(n);
        std::vector<ScalarT>      flat(n * static_cast<std::size_t>(block_entries));
        m_row_indices.copy_to(std::span<std::int32_t>{rows});
        m_col_indices.copy_to(std::span<std::int32_t>{cols});
        m_values.copy_to(std::span<ScalarT>{flat});
        main_stream().synchronize();

        for(std::size_t i = 0; i < n; ++i)
        {
            triplets[i].row_index = rows[i];
            triplets[i].col_index = cols[i];
            if constexpr(M == 1 && N == 1)
            {
                triplets[i].value = flat[i];
            }
            else
            {
                for(int c = 0; c < N; ++c)
                    for(int r = 0; r < M; ++r)
                        triplets[i].value(r, c) =
                            flat[i * static_cast<std::size_t>(block_entries)
                                 + static_cast<std::size_t>(c * M + r)];
            }
        }
    }

    /// Download the flat value storage (layout check in unit tests).
    void copy_to_values(std::span<ScalarT> host) const
    {
        m_values.copy_to(host);
        main_stream().synchronize();
    }
    void copy_from_values(std::span<const ScalarT> host)
    {
        m_values.copy_from(host);
        invalidate();
    }
    void copy_to_row_indices(std::span<std::int32_t> host) const
    {
        m_row_indices.copy_to(host);
        main_stream().synchronize();
    }
    void copy_to_col_indices(std::span<std::int32_t> host) const
    {
        m_col_indices.copy_to(host);
        main_stream().synchronize();
    }

    void fill(const ScalarT& v)
    {
        m_values.fill(v);
        invalidate();
    }

    // -- frame lifetime (plan risk R5) --------------------------------------
    [[nodiscard]] std::uint64_t generation() const noexcept { return m_cell->value; }
    [[nodiscard]] std::shared_ptr<const generation_cell> lifetime_cell() const noexcept
    {
        return m_cell;
    }

  protected:
    [[nodiscard]] view_type make_view()
    {
        view_type v{m_rows,
                    m_cols,
                    static_cast<std::int32_t>(m_row_indices.size()),
                    m_row_indices.view(),
                    m_col_indices.view(),
                    m_values.view()};
        v.set_lifetime(m_cell, m_cell->value);
        return v;
    }

    void invalidate() noexcept { ++m_cell->value; }

    int                              m_rows = 0;
    int                              m_cols = 0;
    LDeviceBuffer<std::int32_t>      m_row_indices;
    LDeviceBuffer<std::int32_t>      m_col_indices;
    LDeviceBuffer<ScalarT>           m_values;  // flat: nnz * M * N
    std::shared_ptr<generation_cell> m_cell = std::make_shared<generation_cell>();
};

/// BCOO is a (block) triplet matrix that additionally exposes `non_zeros()`
/// (already provided by the base; kept for muda parity).
template <typename T, int M = 1, int N = M>
class LBCOOMatrix : public LTripletMatrix<T, M, N>
{
};

// ---------------------------------------------------------------------------
//  LDoubletVector (sparse-segment vector)
// ---------------------------------------------------------------------------
template <typename T, int N = 1>
class LDoubletVector
{
  public:
    static_assert(std::is_same_v<T, buffer_scalar_of_t<T>>,
                  "linalg: container element type must be a plain scalar");

    using ValueT   = std::conditional_t<(N == 1), T, LVec<T, N>>;
    using ScalarT  = buffer_scalar_of_t<T>;
    using DoubletT = VectorDoublet<T, N>;
    using view_type       = LDoubletVectorView<T, N>;
    using const_view_type = LCDoubletVectorView<T, N>;

    static constexpr bool IsSegmentVector = (N > 1);
    static constexpr int  segment_dim     = N;

    // -- shape / allocation --------------------------------------------------
    void reshape(int num_segment)
    {
        m_count = num_segment;
        invalidate();
    }
    void resize_doublets(std::size_t nnz)
    {
        m_values.resize(nnz * static_cast<std::size_t>(N));
        m_indices.resize(nnz);
        invalidate();
    }
    void resize_doublets_discard(std::size_t nnz) { resize_doublets(nnz); }
    void reserve_doublets(std::size_t nnz)
    {
        m_values.reserve(nnz * static_cast<std::size_t>(N));
        m_indices.reserve(nnz);
        invalidate();
    }
    void reserve_doublets_discard(std::size_t nnz) { reserve_doublets(nnz); }
    void resize(int num_segment, std::size_t nnz)
    {
        reshape(num_segment);
        resize_doublets(nnz);
    }
    void resize_discard(int num_segment, std::size_t nnz)
    {
        reshape(num_segment);
        resize_doublets_discard(nnz);
    }
    void clear()
    {
        m_count = 0;
        m_values.clear();
        m_indices.clear();
        invalidate();
    }
    void reset()
    {
        m_count = 0;
        m_values.reset();
        m_indices.reset();
        invalidate();
    }

    // -- metadata -----------------------------------------------------------
    int         count() const { return m_count; }
    std::size_t doublet_count() const { return m_indices.size(); }
    std::size_t doublet_capacity() const { return m_indices.capacity(); }
    int         non_zeros() const { return static_cast<int>(doublet_count()); }

    LBufferView<std::int32_t>  indices() { return m_indices.view(); }
    LCBufferView<std::int32_t> indices() const { return m_indices.cview(); }
    LBufferView<ScalarT>       values() { return m_values.view(); }
    LCBufferView<ScalarT>      values() const { return m_values.cview(); }

    LDeviceBuffer<std::int32_t>& index_buffer() { return m_indices; }
    LDeviceBuffer<ScalarT>&      values_buffer() { return m_values; }

    // -- views --------------------------------------------------------------
    view_type view()
    {
        view_type v{m_count,
                    static_cast<std::int32_t>(m_indices.size()),
                    m_indices.view(),
                    m_values.view()};
        v.set_lifetime(m_cell, m_cell->value);
        return v;
    }
    const_view_type view() const { return const_cast<LDoubletVector*>(this)->view(); }
    const_view_type cview() const { return view(); }
    view_type       viewer() { return view(); }
    const_view_type cviewer() const { return view(); }

    operator view_type() { return view(); }
    operator const_view_type() const { return view(); }

    // -- host <-> device helpers --------------------------------------------
    void copy_from_doublets(std::span<const DoubletT> doublets)
    {
        const auto n = doublets.size();
        resize_doublets(n);
        if(n == 0u)
        {
            return;
        }
        std::vector<std::int32_t> indices(n);
        std::vector<ScalarT>      flat(n * static_cast<std::size_t>(N));
        for(std::size_t i = 0; i < n; ++i)
        {
            indices[i] = doublets[i].index;
            if constexpr(N == 1)
            {
                flat[i] = static_cast<ScalarT>(doublets[i].value);
            }
            else
            {
                for(int k = 0; k < N; ++k)
                    flat[i * static_cast<std::size_t>(N) + static_cast<std::size_t>(k)] =
                        static_cast<ScalarT>(doublets[i].value(k));
            }
        }
        m_indices.copy_from(std::span<const std::int32_t>{indices});
        m_values.copy_from(std::span<const ScalarT>{flat});
        invalidate();
    }

    void copy_to_doublets(std::span<DoubletT> doublets) const
    {
        const auto n = doublet_count();
        if(doublets.size() != n)
        {
            throw std::invalid_argument{"LDoubletVector::copy_to_doublets size mismatch"};
        }
        if(n == 0u)
        {
            return;
        }
        std::vector<std::int32_t> indices(n);
        std::vector<ScalarT>      flat(n * static_cast<std::size_t>(N));
        m_indices.copy_to(std::span<std::int32_t>{indices});
        m_values.copy_to(std::span<ScalarT>{flat});
        main_stream().synchronize();

        for(std::size_t i = 0; i < n; ++i)
        {
            doublets[i].index = indices[i];
            if constexpr(N == 1)
            {
                doublets[i].value = flat[i];
            }
            else
            {
                for(int k = 0; k < N; ++k)
                    doublets[i].value(k) = flat[i * static_cast<std::size_t>(N)
                                                 + static_cast<std::size_t>(k)];
            }
        }
    }

    void copy_to_values(std::span<ScalarT> host) const
    {
        m_values.copy_to(host);
        main_stream().synchronize();
    }
    void copy_from_values(std::span<const ScalarT> host)
    {
        m_values.copy_from(host);
        invalidate();
    }
    void copy_to_indices(std::span<std::int32_t> host) const
    {
        m_indices.copy_to(host);
        main_stream().synchronize();
    }

    void fill(const ScalarT& v)
    {
        m_values.fill(v);
        invalidate();
    }

    [[nodiscard]] std::uint64_t generation() const noexcept { return m_cell->value; }
    [[nodiscard]] std::shared_ptr<const generation_cell> lifetime_cell() const noexcept
    {
        return m_cell;
    }

  protected:
    void invalidate() noexcept { ++m_cell->value; }

    int                              m_count = 0;
    LDeviceBuffer<std::int32_t>      m_indices;
    LDeviceBuffer<ScalarT>           m_values;  // flat: nnz * N
    std::shared_ptr<generation_cell> m_cell = std::make_shared<generation_cell>();
};

/// BCOO vector is a doublet vector that additionally exposes `non_zeros()`
/// (muda parity).
template <typename T, int N = 1>
class LBCOOVector : public LDoubletVector<T, N>
{
};

// ---------------------------------------------------------------------------
//  LBSRMatrix (block sparse row)
// ---------------------------------------------------------------------------
/// Semantics follow muda/cuda_tool: `reshape(row, col)` allocates `row_offsets`
/// (`rows + 1` entries), `resize(nnz)` allocates values + col_indices.
template <typename T, int N = 1>
class LBSRMatrix
{
  public:
    static_assert(std::is_same_v<T, buffer_scalar_of_t<T>>,
                  "linalg: container element type must be a plain scalar");

    using traits    = block_traits<T, N, N>;
    using ValueT    = typename traits::value_type;  // T | LMat<T,N,N>
    using ScalarT   = typename traits::scalar_type;
    using view_type = LBSRMatrixView<T, N>;
    using const_view_type = LCBSRMatrixView<T, N>;

    static constexpr bool IsBlockMatrix = (N > 1);
    static constexpr int  block_size    = N;
    static constexpr int  block_entries = N * N;

    void reshape(int row, int col)
    {
        m_rows = row;
        m_cols = col;
        m_row_offsets.resize(static_cast<std::size_t>(row) + 1u);
        invalidate();
    }
    void resize(int non_zero_blocks)
    {
        m_values.resize(static_cast<std::size_t>(non_zero_blocks)
                        * static_cast<std::size_t>(block_entries));
        m_col_indices.resize(static_cast<std::size_t>(non_zero_blocks));
        invalidate();
    }
    void reserve(int non_zero_blocks)
    {
        m_values.reserve(static_cast<std::size_t>(non_zero_blocks)
                         * static_cast<std::size_t>(block_entries));
        m_col_indices.reserve(static_cast<std::size_t>(non_zero_blocks));
        invalidate();
    }
    void reserve_offsets(int size)
    {
        m_row_offsets.reserve(static_cast<std::size_t>(size));
        invalidate();
    }
    void clear()
    {
        m_rows = m_cols = 0;
        m_values.clear();
        m_row_offsets.clear();
        m_col_indices.clear();
        invalidate();
    }
    void reset()
    {
        m_rows = m_cols = 0;
        m_values.reset();
        m_row_offsets.reset();
        m_col_indices.reset();
        invalidate();
    }

    static constexpr int block_size_value() { return N; }

    LBufferView<std::int32_t>  row_offsets() { return m_row_offsets.view(); }
    LCBufferView<std::int32_t> row_offsets() const { return m_row_offsets.cview(); }
    LBufferView<std::int32_t>  col_indices() { return m_col_indices.view(); }
    LCBufferView<std::int32_t> col_indices() const { return m_col_indices.cview(); }
    LBufferView<ScalarT>       values() { return m_values.view(); }
    LCBufferView<ScalarT>      values() const { return m_values.cview(); }

    int         rows() const { return m_rows; }
    int         cols() const { return m_cols; }
    int         non_zeros() const { return static_cast<int>(m_col_indices.size()); }
    std::size_t non_zero_blocks() const { return m_col_indices.size(); }

    LDeviceBuffer<std::int32_t>& row_offset_buffer() { return m_row_offsets; }
    LDeviceBuffer<std::int32_t>& col_index_buffer() { return m_col_indices; }
    LDeviceBuffer<ScalarT>&      values_buffer() { return m_values; }

    // -- views --------------------------------------------------------------
    view_type view()
    {
        view_type v{m_rows,
                    m_cols,
                    non_zeros(),
                    m_row_offsets.view(),
                    m_col_indices.view(),
                    m_values.view()};
        v.set_lifetime(m_cell, m_cell->value);
        return v;
    }
    const_view_type view() const { return const_cast<LBSRMatrix*>(this)->view(); }
    const_view_type cview() const { return view(); }
    view_type       viewer() { return view(); }
    const_view_type cviewer() const { return view(); }

    operator view_type() { return view(); }
    operator const_view_type() const { return view(); }

    // -- host side construction helpers -------------------------------------
    /// Fill row_offsets / col_indices / values from a host CSR-style triple.
    void copy_from_host(std::span<const std::int32_t>          row_offsets,
                        std::span<const std::int32_t>          col_indices,
                        std::span<const ScalarT>               flat_values)
    {
        if(flat_values.size() != col_indices.size() * static_cast<std::size_t>(block_entries))
        {
            throw std::invalid_argument{"LBSRMatrix::copy_from_host value size mismatch"};
        }
        m_row_offsets.copy_from(row_offsets);
        m_col_indices.copy_from(col_indices);
        m_values.copy_from(flat_values);
        invalidate();
    }

    void copy_to_host(std::span<std::int32_t> row_offsets,
                      std::span<std::int32_t> col_indices,
                      std::span<ScalarT>      flat_values) const
    {
        m_row_offsets.copy_to(row_offsets);
        m_col_indices.copy_to(col_indices);
        m_values.copy_to(flat_values);
        main_stream().synchronize();
    }

    [[nodiscard]] std::uint64_t generation() const noexcept { return m_cell->value; }
    [[nodiscard]] std::shared_ptr<const generation_cell> lifetime_cell() const noexcept
    {
        return m_cell;
    }

  protected:
    void invalidate() noexcept { ++m_cell->value; }

    int                              m_rows = 0;
    int                              m_cols = 0;
    LDeviceBuffer<std::int32_t>      m_row_offsets;
    LDeviceBuffer<std::int32_t>      m_col_indices;
    LDeviceBuffer<ScalarT>           m_values;  // flat: nnz * N * N
    std::shared_ptr<generation_cell> m_cell = std::make_shared<generation_cell>();
};

// ---------------------------------------------------------------------------
//  LDenseVector
// ---------------------------------------------------------------------------
template <typename T>
class LDenseVector
{
  public:
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "only real numbers are supported");

    using ValueT    = T;
    using ScalarT   = T;
    using view_type       = LDenseVectorView<T>;
    using const_view_type = LCDenseVectorView<T>;

    LDenseVector()                               = default;
    LDenseVector(LDenseVector&&)                 = default;
    LDenseVector& operator=(LDenseVector&&)      = default;
    LDenseVector(const LDenseVector& o) { copy_from(o); }

    LDenseVector& operator=(const LDenseVector& o)
    {
        copy_from(o);  // deep copy (PCG p = z)
        return *this;
    }
    void copy_from(const LDenseVector& o)
    {
        resize(o.size());
        if(o.size() != 0u)
        {
            m_values.view().copy_from(o.m_values.cview());  // device -> device
        }
        invalidate();
    }

    void   resize(std::size_t n) { m_values.resize(n); invalidate(); }
    void   resize_discard(std::size_t n) { resize(n); }
    void   reserve(std::size_t n) { m_values.reserve(n); invalidate(); }
    void   reserve_discard(std::size_t n) { reserve(n); }
    std::size_t size() const { return m_values.size(); }
    std::size_t capacity() const { return m_values.capacity(); }
    void   fill(const T& value)
    {
        m_values.fill(value);
        invalidate();
    }
    void clear()
    {
        m_values.clear();
        invalidate();
    }
    void reset()
    {
        m_values.reset();
        invalidate();
    }

    view_type view()
    {
        view_type v{m_values.view(), 0, static_cast<std::int32_t>(m_values.size()),
                    static_cast<std::int32_t>(m_values.size())};
        v.set_lifetime(m_cell, m_cell->value);
        return v;
    }
    const_view_type view() const { return const_cast<LDenseVector*>(this)->view(); }
    const_view_type cview() const { return view(); }
    view_type       viewer() { return view(); }
    const_view_type viewer() const { return view(); }
    const_view_type cviewer() const { return view(); }

    /// Sub-window view (the reference reaches this through the view's subview()).
    view_type view(std::size_t offset, std::size_t count)
    {
        return view().subview(static_cast<std::int32_t>(offset),
                              static_cast<std::int32_t>(count));
    }
    const_view_type view(std::size_t offset, std::size_t count) const
    {
        return view(offset, count);
    }

    LBufferView<T>  buffer_view() { return m_values.view(); }
    LCBufferView<T> buffer_view() const { return m_values.cview(); }

    operator view_type() { return view(); }
    operator const_view_type() const { return view(); }

    void copy_to(T* host) const { copy_to(std::span<T>{host, size()}); }
    void copy_to(std::span<T> host) const
    {
        m_values.copy_to(host);
        main_stream().synchronize();
    }
    template <typename Alloc>
    void copy_to(std::vector<T, Alloc>& host) const
    {
        host.resize(m_values.size());
        copy_to(std::span<T>{host});
    }
    void copy_from(std::span<const T> host)
    {
        m_values.copy_from(host);
        invalidate();
    }

    LDeviceBuffer<T>&       values_buffer() { return m_values; }
    const LDeviceBuffer<T>& values_buffer() const { return m_values; }

    [[nodiscard]] std::uint64_t generation() const noexcept { return m_cell->value; }
    [[nodiscard]] std::shared_ptr<const generation_cell> lifetime_cell() const noexcept
    {
        return m_cell;
    }

  private:
    void invalidate() noexcept { ++m_cell->value; }

    LDeviceBuffer<T>                 m_values;
    std::shared_ptr<generation_cell> m_cell = std::make_shared<generation_cell>();
};

// ---------------------------------------------------------------------------
//  LDenseMatrix (row-major, contiguous — the reference's device debug buffer)
// ---------------------------------------------------------------------------
template <typename T>
class LDenseMatrix
{
  public:
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "only real numbers are supported");

    using value_type      = T;
    using view_type       = LDenseMatrixView<T>;
    using const_view_type = LCDenseMatrixView<T>;

    LDenseMatrix() = default;
    LDenseMatrix(std::size_t row, std::size_t col, bool sym = false)
        : m_sym(sym)
    {
        reshape(row, col);
    }

    void reshape(std::size_t row, std::size_t col)
    {
        m_row = row;
        m_col = col;
        m_data.resize(row * col);
        invalidate();
    }
    void fill(T value)
    {
        m_data.fill(value);
        invalidate();
    }
    void clear()
    {
        m_row = m_col = 0;
        m_data.clear();
        invalidate();
    }

    std::size_t row() const { return m_row; }
    std::size_t col() const { return m_col; }
    void        sym(bool sym = true) { m_sym = sym; }
    bool        sym() const { return m_sym; }

    LBufferView<T>  buffer_view() { return m_data.view(); }
    LCBufferView<T> buffer_view() const { return m_data.cview(); }

    view_type view()
    {
        view_type v{m_data.view(),
                    static_cast<std::int32_t>(m_row),
                    static_cast<std::int32_t>(m_col)};
        v.set_lifetime(m_cell, m_cell->value);
        return v;
    }
    const_view_type view() const { return const_cast<LDenseMatrix*>(this)->view(); }
    const_view_type cview() const { return view(); }

    operator view_type() { return view(); }
    operator const_view_type() const { return view(); }

    void copy_to(std::span<T> host) const
    {
        m_data.copy_to(host);
        main_stream().synchronize();
    }
    void copy_from(std::span<const T> host)
    {
        m_data.copy_from(host);
        invalidate();
    }

    [[nodiscard]] std::uint64_t generation() const noexcept { return m_cell->value; }
    [[nodiscard]] std::shared_ptr<const generation_cell> lifetime_cell() const noexcept
    {
        return m_cell;
    }

  private:
    void invalidate() noexcept { ++m_cell->value; }

    LDeviceBuffer<T>                 m_data;
    std::size_t                      m_row = 0;
    std::size_t                      m_col = 0;
    bool                             m_sym = false;
    std::shared_ptr<generation_cell> m_cell = std::make_shared<generation_cell>();
};

// ---------------------------------------------------------------------------
//  Reference-compatible container aliases
// ---------------------------------------------------------------------------
template <typename T, int M = 1, int N = M>
using DeviceTripletMatrix = LTripletMatrix<T, M, N>;
template <typename T, int M = 1, int N = M>
using DeviceBCOOMatrix = LBCOOMatrix<T, M, N>;
template <typename T>
using DeviceCOOMatrix = LBCOOMatrix<T, 1, 1>;
template <typename T, int N = 1>
using DeviceDoubletVector = LDoubletVector<T, N>;
template <typename T, int N = 1>
using DeviceBCOOVector = LBCOOVector<T, N>;
template <typename T, int N = 1>
using DeviceBSRMatrix = LBSRMatrix<T, N>;
template <typename T>
using DeviceDenseVector = LDenseVector<T>;
template <typename T>
using DeviceDenseMatrix = LDenseMatrix<T>;
}// namespace uipc::backend::luisa::luisa_device::linalg
