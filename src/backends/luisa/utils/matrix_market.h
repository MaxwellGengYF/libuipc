#pragma once
/********************************************************************
 * @file matrix_market.h
 * @brief T8.3b — the host-side Matrix Market writer/reader (port of
 *        src/backends/cuda/utils/matrix_market.h).
 *
 * Ported surface, in the reference's order: `export_matrix_market`,
 * `export_vector_market`, `import_matrix_market`, `import_vector_market`.  The
 * file *format* is the reference's byte for byte — the five comment lines, the
 * `"<rows> <cols> <nnz>"` dims line with **scalar** (block dim x BlockDim)
 * dimensions, one `"i j value"` record per *scalar* entry with `%.17g`
 * (`{:.17g}` through fmt) and 1-based indices by default, **all 9 entries of a
 * 3x3 block written including the zeros**, and the vector writer's
 * `%%MatrixMarket matrix array real general` + `"<n> 1"` + one `%.17g` line per
 * element.  That is what makes a Luisa dump readable by the CUDA backend and by
 * `import_*` below, and what lets a parity diff compare the two files directly.
 *
 * THE ONE DELIBERATE DIFFERENCE: THE ARGUMENTS ARE HOST DATA, NOT DEVICE VIEWS
 * ---------------------------------------------------------------------------
 * The reference types the writer as `cuda_tool::CTripletMatrixView<T,BD>` /
 * `CDenseVectorView<T>` and downloads inside.  Neither type exists here
 * (`lla::LCTripletMatrixView` is a buffer-view bundle, and its flat `values()` is
 * column-major *within* a block, so a verbatim pointer copy would transpose every
 * block).  The writer therefore takes `std::span` over host data, and the two call
 * sites in `linear_system/global_linear_system.cpp` do the single download line
 * each (`bcoo_A.copy_to_triplets(host)` / `x.copy_to(host)`, both of which
 * synchronise the stream), i.e. exactly what the reference's download did, spelled
 * one level out.
 *
 * Consequences, all of them intended:
 *  * `MatrixTriplet<T,3,3>` carries a real block (`value(r, c)`), which is the
 *    reference's `block(bi, bj)` — no layout assumption anywhere;
 *  * nothing here includes `sim_engine.h`, `LStream` or a device header, so the
 *    format is testable without a GPU (the `[luisa][mtx]` format case), and
 *    `import_*` only *names* the container it fills through its template
 *    parameter;
 *  * `import_matrix_market` writes through `MatrixT::reshape()` +
 *    `MatrixT::copy_from_triplets()`, i.e. the owning container's own host->device
 *    path (`luisa_device/linear/containers.h`), not three hand-rolled uploads.
 *
 * NOT PORTED (plan 11 §3, last row — the plan's §4 sentence about it is a
 * misreading): `utils/matrix_unpacker.h` is **not** an MTX reader.  It is the
 * reference's device-side block-writer proxy (`ProxyRange`/`half_block`/`segment`
 * over a `TripletMatrixViewer`, Eigen-typed), and this backend already replaced it
 * with explicit per-triplet writes
 * (`affine_body/abd_linear_subsystem_function.h`).  MTX read-back lives here.
 *
 * Format reference: https://math.nist.gov/MatrixMarket/formats.html
 *
 * FINDING (measured on the written files, T8.3b): the banner is `%MatrixMarket …` with
 * a SINGLE `%`.  `fmt::fprintf(fp, "%%MatrixMarket …")` collapses `%%` to `%` (printf
 * semantics), so the reference's own dumps have the same single-`%` banner.  The port
 * reproduces it byte for byte, because cross-backend file equality is the contract; both
 * readers accept it (they skip any line whose first character is `%`).  A third-party
 * Matrix Market reader would not, since the standard requires `%%MatrixMarket`.  Do not
 * "fix" this unilaterally — it would silently stop matching CUDA's dumps.
 *********************************************************************/
#include <type_define.h>
#include <fmt/printf.h>
#include <cstdio>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace uipc::backend::luisa
{
/**
 * @brief Export block triplets (host data) to Matrix Market coordinate format.
 *
 * One `"i j value"` record per scalar entry, all `BlockDim * BlockDim` entries of
 * every block written (explicit zeros included), so the BCOO structure survives the
 * round trip.  Indices are 1-based by default (the format's convention).
 *
 * @tparam TripletT A `linalg::MatrixTriplet<T, BlockDim>`-shaped entry: it must
 *         expose `row_index`, `col_index`, `ValueT` and `value(r, c)`.
 * @param filename Output filename.
 * @param triplets The block entries to write.
 * @param block_rows Number of *block* rows (the scalar row count is
 *        `block_rows * BlockDim`).
 * @param block_cols Number of *block* columns.
 * @param one_based Use 1-based indices (the Matrix Market standard).
 * @return true on success, false if the file cannot be opened.
 *
 * The block dimension is taken from the block type itself
 * (`TripletT::ValueT::rows`, i.e. `linalg::LMat`'s static extent), not from a second
 * template argument — a `std::span<const MatrixTriplet<T, 3, 3>>` argument would
 * otherwise not deduce it.
 */
template <typename TripletT>
bool export_matrix_market(std::string_view            filename,
                          std::span<const TripletT>   triplets,
                          int                         block_rows,
                          int                         block_cols,
                          bool                        one_based = true)
{
    using BlockT = typename TripletT::ValueT;
    static constexpr int BlockDim = BlockT::rows;
    static_assert(BlockDim > 0 && BlockT::cols == BlockDim,
                  "matrix_market: the blocks must be square and non-empty");
    using ValueT = std::decay_t<decltype(std::declval<const BlockT&>()(0, 0))>;

    const int total_rows = block_rows * BlockDim;
    const int total_cols = block_cols * BlockDim;
    // every block contributes BlockDim * BlockDim scalars, zeros included
    const int exact_nnz = static_cast<int>(triplets.size()) * BlockDim * BlockDim;

    // buffer the (potentially large) body, then write header + body in one go
    auto body = fmt::memory_buffer();
    {
        const int index_offset = one_based ? 1 : 0;
        for(const auto& t : triplets)
        {
            const int scalar_i_base = t.row_index * BlockDim;
            const int scalar_j_base = t.col_index * BlockDim;

            for(int bi = 0; bi < BlockDim; ++bi)
            {
                for(int bj = 0; bj < BlockDim; ++bj)
                {
                    const ValueT value = t.value(bi, bj);
                    // full precision, scientific notation; write *all* entries,
                    // including zeros, to preserve the BCOO structure
                    fmt::format_to(std::back_inserter(body),
                                   "{} {} {:.17g}\n",
                                   scalar_i_base + bi + index_offset,
                                   scalar_j_base + bj + index_offset,
                                   value);
                }
            }
        }
    }

    FILE* fp = std::fopen(std::string(filename).c_str(), "w");
    if(!fp)
        return false;

    fmt::fprintf(fp, "%%MatrixMarket matrix coordinate real general\n");
    fmt::fprintf(fp, "%% Exported from block-sparse triplet matrix\n");
    fmt::fprintf(fp, "%% Block dimension: %dx%d\n", BlockDim, BlockDim);
    fmt::fprintf(fp, "%% Total rows: %d, Total cols: %d\n", total_rows, total_cols);
    fmt::fprintf(fp, "%% Total entries (including zeros): %d\n", exact_nnz);
    fmt::fprintf(fp, "%d %d %d\n", total_rows, total_cols, exact_nnz);

    std::fwrite(body.data(), 1, body.size(), fp);
    std::fclose(fp);
    return true;
}

/**
 * @brief Export a dense vector (host data) to Matrix Market array format.
 *
 * `%%MatrixMarket matrix array real general`, three comment lines, `"<n> 1"`, then
 * one `%.17g` line per element — the reference's text verbatim.
 *
 * @param one_based Kept for the reference's signature; an array-format vector has no
 *        index column, so it changes nothing (a vector dump has no indices at all).
 */
template <typename T>
bool export_vector_market(std::string_view     filename,
                          std::span<const T>   values,
                          bool                 one_based = true)
{
    (void)one_based;  // array format: values only, no index column

    FILE* fp = std::fopen(std::string(filename).c_str(), "w");
    if(!fp)
        return false;

    fmt::fprintf(fp, "%%MatrixMarket matrix array real general\n");
    fmt::fprintf(fp, "%% Exported from dense vector\n");
    fmt::fprintf(fp, "%% Vector size: %zu\n", values.size());
    fmt::fprintf(fp, "%zu 1\n", values.size());

    for(std::size_t i = 0; i < values.size(); ++i)
        fmt::fprintf(fp, "%.17g\n", values[i]);

    std::fclose(fp);
    return true;
}

/**
 * @brief Import a Matrix Market coordinate file into a block triplet container.
 *
 * Scalar entries are grouped into `BlockDim x BlockDim` blocks keyed by
 * `(i / BlockDim, j / BlockDim)` and handed over in ascending `(block_i, block_j)`
 * order (a `std::map` with the pair's lexicographic order — the reference's custom
 * comparator is the same order), so a block is present iff at least one of its
 * scalars appears and the round trip of `export_matrix_market` is exact.
 *
 * @tparam MatrixT An owning container (`linalg::LTripletMatrix` /
 *         `linalg::LBCOOMatrix`): it must expose `block_rows`, `TripletT`,
 *         `reshape(int, int)` and `copy_from_triplets(std::span<const TripletT>)`.
 * @param out The container to fill (reshaped + re-uploaded; its previous content is
 *        discarded).
 * @param filename Input filename.
 * @param one_based Whether the input uses 1-based indices.
 * @return true on success; false on a missing file, a malformed/empty header or a
 *         dims line that does not describe a block matrix.
 */
template <typename MatrixT>
bool import_matrix_market(MatrixT& out, std::string_view filename, bool one_based = true)
{
    static constexpr int BlockDim = MatrixT::block_rows;
    static_assert(BlockDim > 0, "matrix_market: the block dimension must be positive");
    using TripletT = typename MatrixT::TripletT;
    using BlockT   = typename TripletT::ValueT;
    using ScalarT  = std::decay_t<decltype(std::declval<BlockT&>()(0, 0))>;

    std::ifstream file{std::string{filename}};
    if(!file.is_open())
        return false;

    std::string line;
    bool        header_read = false;
    int         rows = 0, cols = 0, nnz = 0;

    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '%')
            continue;

        std::istringstream iss(line);
        if(!(iss >> rows >> cols >> nnz))
        {
            file.close();
            return false;
        }
        header_read = true;
        break;
    }

    if(!header_read || rows <= 0 || cols <= 0 || nnz <= 0)
    {
        file.close();
        return false;
    }

    const int index_offset = one_based ? -1 : 0;  // convert to 0-based

    // group the scalars into blocks; the map's order is the (i, j) order the
    // reference's custom comparator produces, and it keeps explicit zero blocks
    std::map<std::pair<int, int>, BlockT> block_map;

    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '%')
            continue;

        std::istringstream iss(line);
        int                i = 0, j = 0;
        double             value = 0.0;
        if(!(iss >> i >> j >> value))
            continue;

        i += index_offset;
        j += index_offset;
        if(i < 0 || i >= rows || j < 0 || j >= cols)
            continue;

        auto& block = block_map[std::make_pair(i / BlockDim, j / BlockDim)];
        block(i % BlockDim, j % BlockDim) = static_cast<ScalarT>(value);
    }
    file.close();

    std::vector<TripletT> triplets;
    triplets.reserve(block_map.size());
    for(const auto& [key, block] : block_map)
    {
        TripletT t;
        t.row_index = key.first;
        t.col_index = key.second;
        t.value     = block;
        triplets.push_back(t);
    }

    const int block_rows = (rows + BlockDim - 1) / BlockDim;
    const int block_cols = (cols + BlockDim - 1) / BlockDim;

    out.reshape(block_rows, block_cols);
    out.copy_from_triplets(std::span<const TripletT>{triplets});
    return true;
}

/**
 * @brief Import a Matrix Market array file into a dense vector container.
 *
 * @tparam VecT An owning dense vector (`linalg::LDenseVector`): it must expose
 *         `ScalarT`, `resize(std::size_t)` and `copy_from(std::span<const ScalarT>)`.
 * @param out The vector to fill (resized + re-uploaded).
 * @param filename Input filename.
 * @param one_based Whether the input uses 1-based indices (an array-format vector
 *        has no index column, so this changes nothing).
 * @return true on success; false on a missing file, a malformed header, a `cols != 1`
 *         dims line (that is a matrix, not a vector) or a body whose element count
 *         does not match the declared row count.
 */
template <typename VecT>
bool import_vector_market(VecT& out, std::string_view filename, bool one_based = true)
{
    (void)one_based;  // array format: values only, no index column
    using T = typename VecT::ScalarT;

    std::ifstream file{std::string{filename}};
    if(!file.is_open())
        return false;

    std::string line;
    bool        header_read = false;
    int         rows = 0, cols = 0;

    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '%')
            continue;

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

    if(!header_read || rows <= 0)
    {
        file.close();
        return false;
    }

    std::vector<T> values_host;
    values_host.reserve(static_cast<std::size_t>(rows));

    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '%')
            continue;

        std::istringstream iss(line);
        T                  value{};
        if(iss >> value)
            values_host.push_back(value);
    }
    file.close();

    if(values_host.size() != static_cast<std::size_t>(rows))
        return false;

    out.resize(values_host.size());
    out.copy_from(std::span<const T>{values_host});
    return true;
}
}  // namespace uipc::backend::luisa
