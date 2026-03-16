#pragma once
/********************************************************************
 * @file   matrix_converter.h
 * @brief  Sparse matrix format converter for LuisaCompute backend
 * 
 * Refactored from the CUDA backend's matrix_converter.h.
 * Uses LuisaCompute Buffer types and provides conversion between
 * sparse matrix formats (Triplet, BCOO, BSR).
 *********************************************************************/
#include <type_define.h>

namespace uipc::backend::luisa
{

/**
 * @brief Int pair structure for matrix indices
 */
struct MatrixConverterIntPair
{
    int x;
    int y;
};

constexpr bool operator==(const MatrixConverterIntPair& l, const MatrixConverterIntPair& r)
{
    return l.x == r.x && l.y == r.y;
}

/**
 * @brief Device triplet matrix for sparse matrix representation
 * 
 * A triplet matrix stores non-zero entries as (row, col, value) tuples.
 * This is a flexible format for matrix construction.
 * 
 * @tparam T Element type (Float)
 * @tparam N Block size (e.g., 3 for 3x3 blocks)
 */
template <typename T, int N>
class DeviceTripletMatrix
{
public:
    using ValueT = Eigen::Matrix<T, N, N>;  // Block matrix type
    
    Buffer<int>       row_indices;   // Row indices of non-zero blocks
    Buffer<int>       col_indices;   // Column indices of non-zero blocks
    Buffer<ValueT>    values;        // Block values
    size_t            block_rows = 0; // Number of block rows
    size_t            block_cols = 0; // Number of block cols
    
    DeviceTripletMatrix() = default;
    DeviceTripletMatrix(Device& device, size_t capacity = 0);
    
    void resize(size_t new_size);
    void clear();
    size_t size() const { return row_indices.size(); }
    
    // Views for kernel access
    auto row_indices_view() const { return row_indices.view(); }
    auto col_indices_view() const { return col_indices.view(); }
    auto values_view() const { return values.view(); }
    auto row_indices_view() { return row_indices.view(); }
    auto col_indices_view() { return col_indices.view(); }
    auto values_view() { return values.view(); }
    
    // Compatibility methods for dytopo_effect_system
    size_t triplet_count() const { return size(); }
    size_t triplet_capacity() const { return values.size(); }
    
    void reshape(size_t rows, size_t cols)
    {
        block_rows = rows;
        block_cols = cols;
    }
    
    // View struct for compatibility
    struct View
    {
        luisa::compute::BufferView<int> row_indices;
        luisa::compute::BufferView<int> col_indices;
        luisa::compute::BufferView<ValueT> values;
        size_t count = 0;
        
        View subview(size_t offset, size_t subcount) const
        {
            return View{
                row_indices.subview(offset, subcount),
                col_indices.subview(offset, subcount),
                values.subview(offset, subcount),
                subcount};
        }
    };
    
    View view() 
    { 
        return View{row_indices.view(), col_indices.view(), values.view(), size()}; 
    }
    
    struct CView
    {
        luisa::compute::BufferView<const int> row_indices;
        luisa::compute::BufferView<const int> col_indices;
        luisa::compute::BufferView<const ValueT> values;
        size_t count = 0;
    };
    
    CView view() const
    { 
        return CView{row_indices.view(), col_indices.view(), values.view(), size()}; 
    }
};

/**
 * @brief Device BCOO (Block Coordinate) matrix
 * 
 * A BCOO matrix stores non-zero entries sorted by (row, col) with unique indices.
 * This format is suitable for sparse matrix operations.
 * 
 * @tparam T Element type (Float)
 * @tparam N Block size (e.g., 3 for 3x3 blocks)
 */
template <typename T, int N>
class DeviceBCOOMatrix
{
public:
    using ValueT = Eigen::Matrix<T, N, N>;  // Block matrix type
    
    Buffer<int>       row_indices;   // Row indices of non-zero blocks (sorted, unique)
    Buffer<int>       col_indices;   // Column indices of non-zero blocks (sorted, unique)
    Buffer<ValueT>    values;        // Block values
    size_t            block_rows = 0; // Number of block rows
    size_t            block_cols = 0; // Number of block cols
    
    DeviceBCOOMatrix() = default;
    DeviceBCOOMatrix(Device& device, size_t capacity = 0);
    
    void resize(size_t new_size);
    void clear();
    size_t size() const { return row_indices.size(); }
    
    // Views for kernel access
    auto row_indices_view() const { return row_indices.view(); }
    auto col_indices_view() const { return col_indices.view(); }
    auto values_view() const { return values.view(); }
    auto row_indices_view() { return row_indices.view(); }
    auto col_indices_view() { return col_indices.view(); }
    auto values_view() { return values.view(); }
    
    // Compatibility methods for dytopo_effect_system
    size_t triplet_count() const { return size(); }
    size_t block_count() const { return size(); }
    
    void reshape(size_t rows, size_t cols)
    {
        block_rows = rows;
        block_cols = cols;
    }
    
    // View struct for compatibility
    struct View
    {
        luisa::compute::BufferView<int> row_indices;
        luisa::compute::BufferView<int> col_indices;
        luisa::compute::BufferView<ValueT> values;
        size_t block_count = 0;
    };
    
    View view() 
    { 
        return View{row_indices.view(), col_indices.view(), values.view(), size()}; 
    }
    
    struct CView
    {
        luisa::compute::BufferView<const int> row_indices;
        luisa::compute::BufferView<const int> col_indices;
        luisa::compute::BufferView<const ValueT> values;
        size_t block_count = 0;
    };
    
    CView view() const
    { 
        return CView{row_indices.view(), col_indices.view(), values.view(), size()}; 
    }
};

/**
 * @brief Device BSR (Block Sparse Row) matrix
 * 
 * A BSR matrix stores non-zero blocks in CSR format with block structure.
 * This format is optimized for sparse matrix-vector multiplication.
 * 
 * @tparam T Element type (Float)
 * @tparam N Block size (e.g., 3 for 3x3 blocks)
 */
template <typename T, int N>
class DeviceBSRMatrix
{
public:
    using ValueT = Eigen::Matrix<T, N, N>;  // Block matrix type
    
    Buffer<int>       row_offsets;   // Row pointer array (size = block_rows + 1)
    Buffer<int>       col_indices;   // Column indices of non-zero blocks
    Buffer<ValueT>    values;        // Block values
    size_t            block_rows = 0; // Number of block rows
    size_t            block_cols = 0; // Number of block cols
    
    DeviceBSRMatrix() = default;
    DeviceBSRMatrix(Device& device, size_t row_capacity = 0, size_t nnz_capacity = 0);
    
    void resize(size_t new_nnz);
    void clear();
    size_t nnz() const { return col_indices.size(); }
    
    // Views for kernel access
    auto row_offsets_view() const { return row_offsets.view(); }
    auto col_indices_view() const { return col_indices.view(); }
    auto values_view() const { return values.view(); }
    auto row_offsets_view() { return row_offsets.view(); }
    auto col_indices_view() { return col_indices.view(); }
    auto values_view() { return values.view(); }
};

/**
 * @brief Device doublet vector for sparse vector representation
 * 
 * A doublet vector stores non-zero entries as (index, value) tuples.
 * 
 * @tparam T Element type (Float)
 * @tparam N Block size (e.g., 3 for 3D vectors)
 */
template <typename T, int N>
class DeviceDoubletVector
{
public:
    using ValueT = Eigen::Matrix<T, N, 1>;  // Segment vector type
    
    Buffer<int>       indices;       // Indices of non-zero segments
    Buffer<ValueT>    values;        // Segment values
    size_t            segment_count = 0; // Total number of segments
    
    DeviceDoubletVector() = default;
    DeviceDoubletVector(Device& device, size_t capacity = 0);
    
    void resize(size_t new_size);
    void clear();
    size_t size() const { return indices.size(); }
    
    // Views for kernel access
    auto indices_view() const { return indices.view(); }
    auto values_view() const { return values.view(); }
    auto indices_view() { return indices.view(); }
    auto values_view() { return values.view(); }
    
    // Compatibility methods for dytopo_effect_system
    size_t doublet_count() const { return size(); }
    size_t doublet_capacity() const { return values.size(); }
    
    void reshape(size_t segments)
    {
        segment_count = segments;
    }
    
    // View struct for compatibility
    struct View
    {
        luisa::compute::BufferView<int> indices;
        luisa::compute::BufferView<ValueT> values;
        size_t count = 0;
        
        View subview(size_t offset, size_t subcount) const
        {
            return View{
                indices.subview(offset, subcount),
                values.subview(offset, subcount),
                subcount};
        }
    };
    
    View view() 
    { 
        return View{indices.view(), values.view(), size()}; 
    }
    
    struct CView
    {
        luisa::compute::BufferView<const int> indices;
        luisa::compute::BufferView<const ValueT> values;
        size_t count = 0;
    };
    
    CView view() const
    { 
        return CView{indices.view(), values.view(), size()}; 
    }
};

/**
 * @brief Device BCOO vector for sparse vector representation
 * 
 * A BCOO vector stores non-zero entries sorted by index with unique indices.
 * 
 * @tparam T Element type (Float)
 * @tparam N Block size (e.g., 3 for 3D vectors)
 */
template <typename T, int N>
class DeviceBCOOVector
{
public:
    using ValueT = Eigen::Matrix<T, N, 1>;  // Segment vector type
    
    Buffer<int>       indices;       // Indices of non-zero segments (sorted, unique)
    Buffer<ValueT>    values;        // Segment values
    size_t            segment_count = 0; // Total number of segments
    
    DeviceBCOOVector() = default;
    DeviceBCOOVector(Device& device, size_t capacity = 0);
    
    void resize(size_t new_size);
    void clear();
    size_t size() const { return indices.size(); }
    
    // Views for kernel access
    auto indices_view() const { return indices.view(); }
    auto values_view() const { return values.view(); }
    auto indices_view() { return indices.view(); }
    auto values_view() { return values.view(); }
    
    // Compatibility methods for dytopo_effect_system
    size_t doublet_count() const { return size(); }
    size_t block_count() const { return size(); }
    
    void reshape(size_t segments)
    {
        segment_count = segments;
    }
    
    // View struct for compatibility
    struct View
    {
        luisa::compute::BufferView<int> indices;
        luisa::compute::BufferView<ValueT> values;
        size_t block_count = 0;
    };
    
    View view() 
    { 
        return View{indices.view(), values.view(), size()}; 
    }
    
    struct CView
    {
        luisa::compute::BufferView<const int> indices;
        luisa::compute::BufferView<const ValueT> values;
        size_t block_count = 0;
    };
    
    CView view() const
    { 
        return CView{indices.view(), values.view(), size()}; 
    }
};

/**
 * @brief Matrix converter for sparse matrix format conversions
 * 
 * This class provides utilities for converting between sparse matrix formats:
 * - Triplet -> BCOO: Sort and merge duplicate entries
 * - BCOO -> BSR: Build row pointer structure
 * - Doublet -> BCOO: Sort and merge duplicate entries
 * - ge2sym: Convert general matrix to symmetric (keep only lower triangle)
 * - sym2ge: Convert symmetric matrix to general
 * 
 * @tparam T Element type (Float)
 * @tparam N Block size
 */
template <typename T, int N>
class MatrixConverter
{
    using BlockMatrix   = typename DeviceTripletMatrix<T, N>::ValueT;
    using SegmentVector = typename DeviceDoubletVector<T, N>::ValueT;
    
    Float m_reserve_ratio = 1.5f;

    // Temporary buffers for radix sort and unique operations
    Buffer<int>                    col_counts_per_row;
    Buffer<int>                    unique_indices;
    Buffer<int>                    unique_counts;
    Buffer<int>                    count;

    Buffer<int>                    sort_index_input;
    Buffer<int>                    sort_index;

    Buffer<int>                    offsets;

    Buffer<MatrixConverterIntPair> ij_pairs;
    Buffer<MatrixConverterIntPair> unique_ij_pairs;

    Buffer<uint64_t>               ij_hash_input;
    Buffer<uint64_t>               ij_hash;

    Buffer<BlockMatrix>            blocks_sorted;
    Buffer<BlockMatrix>            diag_blocks;

    Buffer<int>                    indices_sorted;
    Buffer<SegmentVector>          segments_sorted;

    Buffer<int>                    sorted_partition_input;
    Buffer<int>                    sorted_partition_output;

public:
    explicit MatrixConverter(Device& device);
    
    void  reserve_ratio(Float ratio) { m_reserve_ratio = ratio; }
    Float reserve_ratio() const { return m_reserve_ratio; }

    // Triplet -> BCOO
    void convert(const DeviceTripletMatrix<T, N>& from,
                 DeviceBCOOMatrix<T, N>&          to);

    void _radix_sort_indices_and_blocks(const DeviceTripletMatrix<T, N>& from,
                                        DeviceBCOOMatrix<T, N>& to);

    void _radix_sort_indices_and_blocks(DeviceBCOOMatrix<T, N>& to);

    void _make_unique_indices(const DeviceTripletMatrix<T, N>& from,
                              DeviceBCOOMatrix<T, N>&          to);

    void _make_unique_block_warp_reduction(const DeviceTripletMatrix<T, N>& from,
                                           DeviceBCOOMatrix<T, N>& to);

    // BCOO -> BSR
    void convert(const DeviceBCOOMatrix<T, N>& from,
                 DeviceBSRMatrix<T, N>&        to);

    void _calculate_block_offsets(const DeviceBCOOMatrix<T, N>& from,
                                  DeviceBSRMatrix<T, N>&        to);

    // Doublet -> BCOO
    void convert(const DeviceDoubletVector<T, N>& from,
                 DeviceBCOOVector<T, N>&          to);

    void _radix_sort_indices_and_segments(const DeviceDoubletVector<T, N>& from,
                                          DeviceBCOOVector<T, N>& to);

    void _make_unique_indices(const DeviceDoubletVector<T, N>& from,
                              DeviceBCOOVector<T, N>&          to);

    void _make_unique_segment_warp_reduction(const DeviceDoubletVector<T, N>& from,
                                             DeviceBCOOVector<T, N>& to);

    template <typename U>
    void loose_resize(Buffer<U>& buf, size_t new_size)
    {
        if(buf.size() < new_size)
            buf = device().create_buffer<U>(static_cast<size_t>(new_size * m_reserve_ratio));
        // Note: LuisaCompute buffers don't have resize, we track size separately
    }

    void ge2sym(DeviceBCOOMatrix<T, N>& to);
    void ge2sym(DeviceTripletMatrix<T, N>& to);
    void sym2ge(const DeviceBCOOMatrix<T, N>& from,
                DeviceBCOOMatrix<T, N>&       to);

private:
    Device& device() { return m_device; }
    Device& m_device;
};

}  // namespace uipc::backend::luisa

#include "details/matrix_converter.inl"
