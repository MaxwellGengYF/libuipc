/**
* @file matrix_market.h
* @brief Export block-sparse triplet matrices and dense vectors to Matrix Market format
* 
* Mtx format reference: https://math.nist.gov/MatrixMarket/formats.html
* 
* This is a LuisaCompute refactored version of the original CUDA backend.
* Uses luisa::compute runtime API instead of muda.
*/

#pragma once
#include <type_define.h>
#include <vector>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <fmt/printf.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/logging.h>

namespace uipc::backend::luisa
{
using namespace luisa::compute;

/**
 * @brief Triplet matrix view for LuisaCompute backend
 * 
 * Mirrors muda::CTripletMatrixView functionality for LuisaCompute Buffer views.
 */
template <typename T, int BlockDim>
class TripletMatrixView
{
public:
    using BlockMatrix = Eigen::Matrix<T, BlockDim, BlockDim>;

    TripletMatrixView(BufferView<int> row_indices,
                      BufferView<int> col_indices,
                      BufferView<BlockMatrix> values,
                      size_t triplet_count,
                      int total_rows,
                      int total_cols)
        : m_row_indices(row_indices)
        , m_col_indices(col_indices)
        , m_values(values)
        , m_triplet_count(triplet_count)
        , m_total_rows(total_rows)
        , m_total_cols(total_cols)
    {
    }

    [[nodiscard]] BufferView<int> row_indices() const noexcept { return m_row_indices; }
    [[nodiscard]] BufferView<int> col_indices() const noexcept { return m_col_indices; }
    [[nodiscard]] BufferView<BlockMatrix> values() const noexcept { return m_values; }
    [[nodiscard]] size_t triplet_count() const noexcept { return m_triplet_count; }
    [[nodiscard]] int total_rows() const noexcept { return m_total_rows; }
    [[nodiscard]] int total_cols() const noexcept { return m_total_cols; }

private:
    BufferView<int> m_row_indices;
    BufferView<int> m_col_indices;
    BufferView<BlockMatrix> m_values;
    size_t m_triplet_count;
    int m_total_rows;
    int m_total_cols;
};

/**
 * @brief Device triplet matrix for LuisaCompute backend
 * 
 * Mirrors muda::DeviceTripletMatrix functionality for LuisaCompute.
 */
template <typename T, int BlockDim>
class DeviceTripletMatrix
{
public:
    using BlockMatrix = Eigen::Matrix<T, BlockDim, BlockDim>;

    DeviceTripletMatrix(Device& device)
        : m_device(device)
        , m_total_rows(0)
        , m_total_cols(0)
        , m_triplet_count(0)
    {
    }

    void reshape(int rows, int cols)
    {
        m_total_rows = rows;
        m_total_cols = cols;
    }

    void resize_triplets(size_t count)
    {
        m_triplet_count = count;
        m_row_indices = m_device.create_buffer<int>(count);
        m_col_indices = m_device.create_buffer<int>(count);
        m_values = m_device.create_buffer<BlockMatrix>(count);
    }

    [[nodiscard]] BufferView<int> row_indices() noexcept { return m_row_indices.view(); }
    [[nodiscard]] BufferView<int> col_indices() noexcept { return m_col_indices.view(); }
    [[nodiscard]] BufferView<BlockMatrix> values() noexcept { return m_values.view(); }
    
    [[nodiscard]] BufferView<int> row_indices() const noexcept { return m_row_indices.view(); }
    [[nodiscard]] BufferView<int> col_indices() const noexcept { return m_col_indices.view(); }
    [[nodiscard]] BufferView<BlockMatrix> values() const noexcept { return m_values.view(); }
    
    [[nodiscard]] size_t triplet_count() const noexcept { return m_triplet_count; }
    [[nodiscard]] int total_rows() const noexcept { return m_total_rows; }
    [[nodiscard]] int total_cols() const noexcept { return m_total_cols; }

    [[nodiscard]] TripletMatrixView<T, BlockDim> view() const noexcept
    {
        return TripletMatrixView<T, BlockDim>(
            m_row_indices.view(),
            m_col_indices.view(),
            m_values.view(),
            m_triplet_count,
            m_total_rows,
            m_total_cols
        );
    }

private:
    Device& m_device;
    Buffer<int> m_row_indices;
    Buffer<int> m_col_indices;
    Buffer<BlockMatrix> m_values;
    int m_total_rows;
    int m_total_cols;
    size_t m_triplet_count;
};

/**
 * @brief Export a block-sparse triplet matrix to Matrix Market coordinate format
 * 
 * Converts from block-sparse format (i, j, matrixBlockNxN) to Matrix Market
 * coordinate format (i, j, v) where each scalar value in the block is written separately.
 * All entries are exported, including zeros, to preserve the complete BCOO matrix structure.
 * 
 * @tparam T Value type (typically Float/double)
 * @tparam BlockDim Block dimension (e.g., 3 for 3x3 blocks)
 * @param filename Output filename
 * @param matrix Triplet matrix view to export
 * @param stream LuisaCompute stream for data transfer
 * @param one_based If true, use 1-based indexing (Matrix Market standard), else 0-based
 * @return true if successful, false otherwise
 */
template <typename T, int BlockDim>
bool export_matrix_market(std::string_view filename,
                          const TripletMatrixView<T, BlockDim>& matrix,
                          Stream& stream,
                          bool one_based = true)
{
    using BlockMatrix = Eigen::Matrix<T, BlockDim, BlockDim>;

    // Copy data from device to host
    std::vector<int>         row_indices_host(matrix.triplet_count());
    std::vector<int>        col_indices_host(matrix.triplet_count());
    std::vector<BlockMatrix> values_host(matrix.triplet_count());

    stream << matrix.row_indices().copy_to(row_indices_host.data())
           << matrix.col_indices().copy_to(col_indices_host.data())
           << matrix.values().copy_to(values_host.data())
           << synchronize();

    // Calculate total matrix dimensions (scalar, not block)
    int total_rows = matrix.total_rows() * BlockDim;
    int total_cols = matrix.total_cols() * BlockDim;
    int exact_nnz  = 0;
    
    // Create fmt buffer
    auto out = fmt::memory_buffer();

    {
        // Count total non-zeros (each block contributes BlockDim * BlockDim entries)
        int total_nnz = static_cast<int>(matrix.triplet_count()) * BlockDim * BlockDim;
        (void)total_nnz; // Suppress unused warning

        // Write coordinate entries
        int index_offset = one_based ? 1 : 0;

        for(size_t t = 0; t < matrix.triplet_count(); ++t)
        {
            int                block_i = row_indices_host[t];
            int                block_j = col_indices_host[t];
            const BlockMatrix& block   = values_host[t];

            // Convert block indices to scalar indices
            int scalar_i_base = block_i * BlockDim;
            int scalar_j_base = block_j * BlockDim;

            // Write each scalar entry in the block (including zeros)
            for(int bi = 0; bi < BlockDim; ++bi)
            {
                for(int bj = 0; bj < BlockDim; ++bj)
                {
                    T   value    = block(bi, bj);
                    int scalar_i = scalar_i_base + bi + index_offset;
                    int scalar_j = scalar_j_base + bj + index_offset;
                    // Write all entries, including zeros, to verify BCOO matrix structure
                    ++exact_nnz;
                    // full precision, scientific notation
                    fmt::format_to(std::back_inserter(out), "{} {} {:.17g}\n", scalar_i, scalar_j, value);
                }
            }
        }
    }

    // Open output file
    FILE* fp = std::fopen(std::string(filename).c_str(), "w");
    if(!fp)
    {
        LUISA_WARNING("Failed to open file for writing: {}", filename);
        return false;
    }

    // Write Matrix Market header using fmt::printf
    fmt::fprintf(fp, "%%MatrixMarket matrix coordinate real general\n");
    fmt::fprintf(fp, "%% Exported from block-sparse triplet matrix (LuisaCompute)\n");
    fmt::fprintf(fp, "%% Block dimension: %dx%d\n", BlockDim, BlockDim);
    fmt::fprintf(fp, "%% Total rows: %d, Total cols: %d\n", total_rows, total_cols);
    fmt::fprintf(fp, "%% Total entries (including zeros): %d\n", exact_nnz);

    // Write dimensions and non-zero count
    fmt::fprintf(fp, "%d %d %d\n", total_rows, total_cols, exact_nnz);

    // Write the buffered entries to file
    std::fwrite(out.data(), 1, out.size(), fp);

    std::fclose(fp);
    return true;
}

/**
 * @brief Export a dense vector to Matrix Market array format
 * 
 * @tparam T Value type
 * @param filename Output filename
 * @param vector Buffer view to export
 * @param stream LuisaCompute stream for data transfer
 * @param one_based If true, use 1-based indexing (Matrix Market standard), else 0-based
 * @return true if successful, false otherwise
 */
template <typename T>
bool export_vector_market(std::string_view filename,
                          BufferView<T> vector,
                          Stream& stream,
                          bool one_based = true)
{
    (void)one_based; // Index not used in array format, but kept for API consistency

    // Copy data from device to host
    std::vector<T> values_host(vector.size());
    stream << vector.copy_to(values_host.data())
           << synchronize();

    // Open output file
    FILE* fp = std::fopen(std::string(filename).c_str(), "w");
    if(!fp)
    {
        LUISA_WARNING("Failed to open file for writing: {}", filename);
        return false;
    }

    // Write Matrix Market header using fmt::printf
    fmt::fprintf(fp, "%%MatrixMarket matrix array real general\n");
    fmt::fprintf(fp, "%% Exported from dense vector (LuisaCompute)\n");
    fmt::fprintf(fp, "%% Vector size: %zu\n", vector.size());

    // Write dimensions (rows, 1 column)
    fmt::fprintf(fp, "%zu 1\n", vector.size());

    // Write values
    for(size_t i = 0; i < vector.size(); ++i)
    {
        fmt::fprintf(fp, "%.17g\n", values_host[i]);
    }

    std::fclose(fp);
    return true;
}

/**
 * @brief Export a span to Matrix Market array format
 * 
 * @tparam T Value type
 * @param vector Span to export
 * @param filename Output filename
 * @param one_based If true, use 1-based indexing (Matrix Market standard), else 0-based
 * @return true if successful, false otherwise
 */
template <typename T>
bool export_vector_market(luisa::span<const T> vector, std::string_view filename, bool one_based = true)
{
    (void)one_based; // Index not used in array format

    // Open output file
    FILE* fp = std::fopen(std::string(filename).c_str(), "w");
    if(!fp)
    {
        LUISA_WARNING("Failed to open file for writing: {}", filename);
        return false;
    }
    
    // Write Matrix Market header using fmt::printf
    fmt::fprintf(fp, "%%MatrixMarket matrix array real general\n");
    fmt::fprintf(fp, "%% Exported from span (LuisaCompute)\n");
    fmt::fprintf(fp, "%% Vector size: %zu\n", vector.size());
    
    // Write dimensions (rows, 1 column)
    fmt::fprintf(fp, "%zu 1\n", vector.size());
    
    // Write values
    for(size_t i = 0; i < vector.size(); ++i)
    {
        fmt::fprintf(fp, "%.17g\n", vector[i]);
    }
    
    std::fclose(fp);
    return true;
}

/**
 * @brief Import a sparse matrix from Matrix Market coordinate format
 * 
 * Converts from Matrix Market coordinate format (i, j, v) to block-sparse format
 * (i, j, matrixBlockNxN) by grouping scalar entries into blocks.
 * 
 * @tparam T Value type (typically Float/double)
 * @tparam BlockDim Block dimension (e.g., 3 for 3x3 blocks)
 * @param matrix Device triplet matrix to populate
 * @param filename Input filename
 * @param stream LuisaCompute stream for data transfer
 * @param one_based If true, input uses 1-based indexing (Matrix Market standard), else 0-based
 * @return true if successful, false otherwise
 */
template <typename T, int BlockDim>
bool import_matrix_market(DeviceTripletMatrix<T, BlockDim>& matrix,
                          std::string_view filename,
                          Stream& stream,
                          bool one_based = true)
{
    using BlockMatrix = Eigen::Matrix<T, BlockDim, BlockDim>;

    std::ifstream file{std::string{filename}};
    if(!file.is_open())
    {
        LUISA_WARNING("Failed to open file for reading: {}", filename);
        return false;
    }

    std::string line;
    bool        header_read = false;
    int         rows = 0, cols = 0, nnz = 0;
    int         index_offset = one_based ? -1 : 0;  // Convert to 0-based

    // Read header and dimensions
    while(std::getline(file, line))
    {
        // Skip comments
        if(line.empty() || line[0] == '%')
            continue;

        if(!header_read)
        {
            std::istringstream iss(line);
            if(!(iss >> rows >> cols >> nnz))
            {
                file.close();
                return false;
            }
            header_read = true;
            break;
        }
    }

    if(!header_read || rows <= 0 || cols <= 0 || nnz <= 0)
    {
        file.close();
        return false;
    }

    // Read coordinate entries
    std::vector<std::tuple<int, int, T>> entries;
    entries.reserve(nnz);

    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '%')
            continue;

        std::istringstream iss(line);
        int                i, j;
        T                  value;
        if(iss >> i >> j >> value)
        {
            i += index_offset;  // Convert to 0-based
            j += index_offset;
            if(i >= 0 && i < rows && j >= 0 && j < cols)
            {
                entries.emplace_back(i, j, value);
            }
        }
    }
    file.close();

    // Group entries into blocks
    int block_rows = (rows + BlockDim - 1) / BlockDim;
    int block_cols = (cols + BlockDim - 1) / BlockDim;

    // Map from (block_i, block_j) to block matrix
    auto less = [](const std::pair<int, int>& a, const std::pair<int, int>& b)
    {
        if(a.first != b.first)
            return a.first < b.first;
        return a.second < b.second;
    };
    std::map<std::pair<int, int>, BlockMatrix, decltype(less)> block_map(less);

    for(const auto& [i, j, value] : entries)
    {
        int block_i = i / BlockDim;
        int block_j = j / BlockDim;
        int bi      = i % BlockDim;
        int bj      = j % BlockDim;

        auto key = std::make_pair(block_i, block_j);
        if(block_map.find(key) == block_map.end())
        {
            block_map[key] = BlockMatrix::Zero();
        }
        block_map[key](bi, bj) = value;
    }

    // Convert to triplet format
    matrix.reshape(block_rows, block_cols);
    matrix.resize_triplets(block_map.size());

    std::vector<int>         row_indices_host(block_map.size());
    std::vector<int>         col_indices_host(block_map.size());
    std::vector<BlockMatrix> values_host(block_map.size());

    size_t idx = 0;
    for(const auto& [key, block] : block_map)
    {
        row_indices_host[idx] = key.first;
        col_indices_host[idx] = key.second;
        values_host[idx]      = block;
        ++idx;
    }

    // Copy to device
    stream << matrix.row_indices().copy_from(row_indices_host.data())
           << matrix.col_indices().copy_from(col_indices_host.data())
           << matrix.values().copy_from(values_host.data())
           << synchronize();

    return true;
}

/**
 * @brief Import a dense vector from Matrix Market array format
 * 
 * @tparam T Value type
 * @param vector Device buffer to populate
 * @param filename Input filename
 * @param stream LuisaCompute stream for data transfer
 * @param one_based If true, input uses 1-based indexing (Matrix Market standard), else 0-based
 * @return true if successful, false otherwise
 */
template <typename T>
bool import_vector_market(Buffer<T>& vector,
                          std::string_view filename,
                          Stream& stream,
                          bool one_based = true)
{
    (void)one_based; // Index not used in array format

    std::ifstream file{std::string{filename}};
    if(!file.is_open())
    {
        LUISA_WARNING("Failed to open file for reading: {}", filename);
        return false;
    }

    std::string line;
    bool        header_read = false;
    int         rows = 0, cols = 0;

    // Read header and dimensions
    while(std::getline(file, line))
    {
        // Skip comments
        if(line.empty() || line[0] == '%')
            continue;

        if(!header_read)
        {
            std::istringstream iss(line);
            if(!(iss >> rows >> cols))
            {
                file.close();
                return false;
            }
            if(cols != 1)
            {
                file.close();
                return false;
            }
            header_read = true;
            break;
        }
    }

    if(!header_read || rows <= 0)
    {
        file.close();
        return false;
    }

    // Read values
    std::vector<T> values_host;
    values_host.reserve(rows);

    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '%')
            continue;

        std::istringstream iss(line);
        T                  value;
        if(iss >> value)
        {
            values_host.push_back(value);
        }
    }
    file.close();

    if(values_host.size() != static_cast<size_t>(rows))
    {
        return false;
    }

    // Resize and copy to device
    vector = vector.device().create_buffer<T>(rows);
    stream << vector.view().copy_from(values_host.data())
           << synchronize();

    return true;
}

/**
 * @brief Import a dense vector from Matrix Market array format (device buffer with device reference)
 * 
 * @tparam T Value type
 * @param device LuisaCompute device for buffer creation
 * @param filename Input filename
 * @param stream LuisaCompute stream for data transfer
 * @param one_based If true, input uses 1-based indexing (Matrix Market standard), else 0-based
 * @return Buffer<T> if successful, empty buffer otherwise
 */
template <typename T>
Buffer<T> import_vector_market(Device& device,
                               std::string_view filename,
                               Stream& stream,
                               bool one_based = true)
{
    (void)one_based; // Index not used in array format

    std::ifstream file{std::string{filename}};
    if(!file.is_open())
    {
        LUISA_WARNING("Failed to open file for reading: {}", filename);
        return Buffer<T>{};
    }

    std::string line;
    bool        header_read = false;
    int         rows = 0, cols = 0;

    // Read header and dimensions
    while(std::getline(file, line))
    {
        // Skip comments
        if(line.empty() || line[0] == '%')
            continue;

        if(!header_read)
        {
            std::istringstream iss(line);
            if(!(iss >> rows >> cols))
            {
                file.close();
                return Buffer<T>{};
            }
            if(cols != 1)
            {
                file.close();
                return Buffer<T>{};
            }
            header_read = true;
            break;
        }
    }

    if(!header_read || rows <= 0)
    {
        file.close();
        return Buffer<T>{};
    }

    // Read values
    std::vector<T> values_host;
    values_host.reserve(rows);

    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '%')
            continue;

        std::istringstream iss(line);
        T                  value;
        if(iss >> value)
        {
            values_host.push_back(value);
        }
    }
    file.close();

    if(values_host.size() != static_cast<size_t>(rows))
    {
        return Buffer<T>{};
    }

    // Create buffer and copy to device
    Buffer<T> buffer = device.create_buffer<T>(rows);
    stream << buffer.view().copy_from(values_host.data())
           << synchronize();

    return buffer;
}

}  // namespace uipc::backend::luisa
