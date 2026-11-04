/********************************************************************
 * @file luisa_matrix_market_test.cpp
 * @brief T8.3b — the host-side Matrix Market writer/reader
 *        (`utils/matrix_market.h`), the file format the debug dump of
 *        `GlobalLinearSystem::solve()` produces.
 *
 * WHAT IS PINNED, AND AT WHICH PARITY CLASS (plans/11 §6.4, research §4.3)
 * ----------------------------------------------------------------------
 *  * `matrix_market_format_exact` — the whole text layout is **Class A** (byte
 *    equality with the reference's writer, `src/backends/cuda/utils/matrix_market.h`):
 *    the five comment lines, the `"<scalar rows> <scalar cols> <exact nnz>"` dims line
 *    (`block dim * BlockDim`, `triplet_count * BlockDim * BlockDim`), one `"i j value"`
 *    record per *scalar* entry, **all 9 entries of a 3x3 block written including the
 *    zeros**, row-major inside a block, 1-based by default.  The `%.17g` precision is
 *    pinned by exporting 1.0/3.0 and requiring its exact 17-digit spelling.  Device-free
 *    (no `test_runtime()`), so it cannot pay the JIT cost and cannot be skipped by a
 *    missing GPU.
 *  * `matrix_market_roundtrip` — a random-ish sparsity over a 6x6 block grid **with
 *    explicit all-zero blocks**, uploaded to a real `LTripletMatrix<double,3,3>`,
 *    downloaded, exported, re-imported into a second container and compared back:
 *    same `(i, j)` sequence and **Δ == 0** on all 9 values (**Class A**, because the
 *    only transform is the textual `%.17g` round trip of a double, which is exact).
 *  * `matrix_market_vector_roundtrip` — the same for a dense vector, `Δ == 0`, plus the
 *    two edges: `n == 1` round-trips, `n == 0` *exports* (`"0 1"` header) but does not
 *    *import* (`rows <= 0` is rejected, exactly like the reference).
 *  * `matrix_market_negative_controls` — every failure path returns `false` instead of
 *    throwing or silently succeeding: a missing file, a `cols != 1` vector header (that
 *    is a matrix, not a vector), a truncated body, an unwritable path, and the
 *    `one_based == false` variant (which must round-trip too).
 *
 * FINDING (measured, and deliberately *not* "fixed"): the writer's banner is
 * `%MatrixMarket …` with a **single** `%`, not the spec's `%%MatrixMarket`.  The
 * reference writes it through `fmt::fprintf(fp, "%%MatrixMarket …")`, and printf-style
 * formatting collapses `%%` to one `%`, so a CUDA dump has the same single-`%` banner —
 * byte equality with the reference is the contract here (plans/11 §6.4), so the port
 * reproduces it and the assertions below pin the measured text.  Consequence: the file
 * round-trips through this reader and the reference's (both skip every line whose first
 * character is `%`) but is *not* a spec-valid Matrix Market file, so a third-party
 * reader (Eigen's `loadMarket`, SciPy's `mmread`) would reject it.  Changing it is a
 * cross-backend decision, not a porting one — see the cookbook §7.41.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"
#include <utils/matrix_market.h>
#include <luisa_device/linear/containers.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace t83b_mtx
{
namespace fs  = std::filesystem;
namespace lla = uipc::backend::luisa::luisa_device::linalg;

constexpr int kBlockDim = 3;

/// `std::printf` + `fflush`: a UIPC_ASSERT aborts the process and a redirected stdout
/// is fully buffered, so a diagnostic without the flush is lost exactly when needed.
#define T83B_LOG(...) \
    do \
    { \
        std::printf(__VA_ARGS__); \
        std::fflush(stdout); \
    } while(false)

[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = fs::temp_directory_path() / name;
    fs::create_directories(p);
    return p.string();
}

/// One 3x3 block, `v[r][c]` like `linalg::LMat::operator()(r, c)`.
struct Block
{
    double v[kBlockDim][kBlockDim]{};

    [[nodiscard]] double operator()(int r, int c) const { return v[r][c]; }
    void                set(int r, int c, double x) { v[r][c] = x; }
};

[[nodiscard]] inline std::vector<std::string> read_lines(const std::string& path)
{
    std::vector<std::string> lines;
    std::ifstream            in{path};
    std::string              line;
    while(std::getline(in, line))
        lines.push_back(line);
    return lines;
}

/// Split on whitespace (the file has no other quoting).
[[nodiscard]] inline std::vector<std::string> split_ws(const std::string& line)
{
    std::vector<std::string> out;
    std::string              cur;
    for(char c : line)
    {
        if(c == ' ' || c == '\t')
        {
            if(!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
        }
        else
        {
            cur.push_back(c);
        }
    }
    if(!cur.empty())
        out.push_back(cur);
    return out;
}

/// `linalg::MatrixTriplet<double,3,3>` from a host block (the shape
/// `export_matrix_market` consumes).
[[nodiscard]] inline lla::MatrixTriplet<double, 3, 3> make_triplet(int i, int j, const Block& b)
{
    lla::MatrixTriplet<double, 3, 3> t;
    t.row_index = i;
    t.col_index = j;
    for(int r = 0; r < kBlockDim; ++r)
        for(int c = 0; c < kBlockDim; ++c)
            t.value(r, c) = b(r, c);
    return t;
}

[[nodiscard]] inline Block scaled_identity(double c)
{
    Block b;
    for(int i = 0; i < kBlockDim; ++i)
        b.set(i, i, c);
    return b;
}

[[nodiscard]] inline double max_abs_diff(const std::vector<double>& a,
                                         const std::vector<double>& b)
{
    if(a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double d = 0.0;
    for(std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

/// A deterministic "random" sparsity over `n x n` blocks: an *upper-triangular* pattern
/// (the assembled global system is `ge2sym`-packed, so only `row <= col` blocks exist)
/// that deliberately contains three shapes at once — a fully populated block, a
/// diagonal-only block, and a block that is **entirely zero** but must still be written
/// and read back (the writer's "including zeros" contract).
[[nodiscard]] inline std::vector<lla::MatrixTriplet<double, 3, 3>> sparse_table(int n)
{
    std::vector<lla::MatrixTriplet<double, 3, 3>> ts;
    for(int i = 0; i < n; ++i)
    {
        for(int j = i; j < n; ++j)
        {
            const int h = (i * 31 + j * 17) % 7;
            if(h == 0)
                continue;  // structural zero block: not part of the sparsity

            Block b;
            if(h == 1)
            {
                // explicit all-zero block: 9 zeros, kept because the block exists
            }
            else if(h == 2)
            {
                b = scaled_identity(static_cast<double>(i + 1));
            }
            else
            {
                for(int r = 0; r < kBlockDim; ++r)
                    for(int c = 0; c < kBlockDim; ++c)
                        b.set(r, c,
                              (r == c ? 4.0 : 0.0)
                                  + 0.125 * static_cast<double>((i * 7 + j * 5 + r * 3 + c) % 9)
                                  - 0.5);
            }
            ts.push_back(make_triplet(i, j, b));
        }
    }
    return ts;
}

[[nodiscard]] inline std::vector<double> block_values(
    const std::vector<lla::MatrixTriplet<double, 3, 3>>& ts)
{
    std::vector<double> out;
    out.reserve(ts.size() * 9u);
    for(const auto& t : ts)
        for(int r = 0; r < kBlockDim; ++r)
            for(int c = 0; c < kBlockDim; ++c)
                out.push_back(t.value(r, c));
    return out;
}
}  // namespace t83b_mtx

// ===========================================================================
//  CASE 1 — the exact text (Class A, device-free)
// ===========================================================================
TEST_CASE("matrix_market_format_exact", "[luisa][linear][mtx]")
{
    using namespace t83b_mtx;

    // block grid 3 x 4, so the scalar dims (9 x 12) cannot coincide with the block dims
    const int block_rows = 3;
    const int block_cols = 4;

    Block a;
    a.set(0, 0, 2.5);
    a.set(1, 1, -1.5);
    a.set(2, 2, 0.25);
    a.set(0, 2, 4.0);

    Block b;
    b.set(1, 0, -0.5);
    b.set(2, 2, 1.0 / 3.0);  // the %.17g precision probe

    std::vector<lla::MatrixTriplet<double, 3, 3>> ts{make_triplet(0, 0, a), make_triplet(1, 2, b)};

    const auto dir  = workspace_path("uipc_luisa_t83b_mtx_format");
    const auto path = dir + "/format.mtx";

    REQUIRE(uipc::backend::luisa::export_matrix_market(
        std::string_view{path}, std::span<const lla::MatrixTriplet<double, 3, 3>>{ts}, block_rows, block_cols));

    const auto lines = read_lines(path);
    T83B_LOG("[t83b] format: %zu lines, first record '%s'\n",
             lines.size(),
             lines.size() > 6 ? lines[6].c_str() : "<none>");

    REQUIRE(lines.size() == 6u + 2u * 9u);  // header + dims + 18 scalar records

    // ---- the header, byte for byte against the reference's writer -------------
    CHECK(lines[0] == "%MatrixMarket matrix coordinate real general");
    CHECK(lines[1] == "% Exported from block-sparse triplet matrix");
    CHECK(lines[2] == "% Block dimension: 3x3");
    CHECK(lines[3] == "% Total rows: 9, Total cols: 12");
    CHECK(lines[4] == "% Total entries (including zeros): 18");
    // dims line: scalar dims, and nnz == triplet_count * BlockDim * BlockDim
    CHECK(lines[5] == "9 12 18");

    // ---- block (0,0): 9 records, row-major inside the block, 1-based -----------
    const auto* values = &a;
    for(int bi = 0; bi < kBlockDim; ++bi)
    {
        for(int bj = 0; bj < kBlockDim; ++bj)
        {
            const auto fields = split_ws(lines[6u + static_cast<std::size_t>(bi * 3 + bj)]);
            REQUIRE(fields.size() == 3u);
            CHECK(std::stoi(fields[0]) == 1 + bi);  // 1-based scalar row
            CHECK(std::stoi(fields[1]) == 1 + bj);  // 1-based scalar col
            CHECK(std::stod(fields[2]) == (*values)(bi, bj));
        }
    }

    // ---- block (1,2) starts at scalar (4,7) and carries the %.17g value -------
    const auto first_of_second = split_ws(lines[6u + 9u]);
    REQUIRE(first_of_second.size() == 3u);
    CHECK(std::stoi(first_of_second[0]) == 4);  // 1 + 1*3
    CHECK(std::stoi(first_of_second[1]) == 7);  // 1 + 2*3
    // the explicit 1/3 records its full 17 significant digits (nothing is shortened)
    bool found_third = false;
    for(std::size_t i = 6u + 9u; i < lines.size(); ++i)
    {
        const auto fields = split_ws(lines[i]);
        REQUIRE(fields.size() == 3u);
        if(fields[2] == "0.33333333333333331")
            found_third = true;
    }
    CHECK(found_third);

    // ---- no 0-based index leaks into a 1-based file --------------------------
    int min_index = 1000, max_row = 0, max_col = 0;
    for(std::size_t i = 6u; i < lines.size(); ++i)
    {
        const auto fields = split_ws(lines[i]);
        REQUIRE(fields.size() == 3u);
        const int r = std::stoi(fields[0]);
        const int c = std::stoi(fields[1]);
        min_index = std::min({min_index, r, c});
        max_row   = std::max(max_row, r);
        max_col   = std::max(max_col, c);
    }
    CHECK(min_index == 1);
    // the records of blocks (0,0) and (1,2) reach scalar (6, 9); the *matrix* dims
    // (9 x 12) live in the dims line and are asserted above
    CHECK(max_row == 6);
    CHECK(max_col == 9);

    // ---- `one_based = false`: same table, 0-based indices --------------------
    const auto path0 = dir + "/format0.mtx";
    REQUIRE(uipc::backend::luisa::export_matrix_market(
        std::string_view{path0}, std::span<const lla::MatrixTriplet<double, 3, 3>>{ts}, block_rows, block_cols, false));
    const auto lines0 = read_lines(path0);
    REQUIRE(lines0.size() == lines.size());
    CHECK(split_ws(lines0[6])[0] == "0");
    CHECK(split_ws(lines0[6])[1] == "0");
    // the header is index-convention independent; only the records change
    for(std::size_t i = 0; i < 6; ++i)
        CHECK(lines0[i] == lines[i]);

    fs::remove(path);
    fs::remove(path0);
}

// ===========================================================================
//  CASE 2 — the round trip through a real container (Class A: Δ == 0)
// ===========================================================================
TEST_CASE("matrix_market_roundtrip", "[luisa][linear][mtx]")
{
    using namespace t83b_mtx;
    test_runtime();

    constexpr int n = 6;  // 6 x 6 blocks == 18 x 18 scalars
    const auto    ts = sparse_table(n);
    REQUIRE(ts.size() > 4u);

    // the source container: host triplets -> device -> back (the download the dump
    // does, so the round trip starts from exactly the data the writer sees)
    lla::LTripletMatrix<double, 3, 3> source;
    source.reshape(n, n);
    source.copy_from_triplets(std::span<const lla::MatrixTriplet<double, 3, 3>>{ts});

    std::vector<lla::MatrixTriplet<double, 3, 3>> downloaded(source.triplet_count());
    source.copy_to_triplets(downloaded);
    REQUIRE(downloaded.size() == ts.size());

    const auto dir  = workspace_path("uipc_luisa_t83b_mtx_roundtrip");
    const auto path = dir + "/roundtrip.mtx";
    REQUIRE(uipc::backend::luisa::export_matrix_market(
        std::string_view{path},
        std::span<const lla::MatrixTriplet<double, 3, 3>>{downloaded},
        source.rows(),
        source.cols()));

    // re-import into a *different* container: reshape/order come from the file alone
    lla::LTripletMatrix<double, 3, 3> restored;
    REQUIRE(uipc::backend::luisa::import_matrix_market(restored, std::string_view{path}));

    CHECK(restored.rows() == n);
    CHECK(restored.cols() == n);
    CHECK(restored.triplet_count() == downloaded.size());

    std::vector<lla::MatrixTriplet<double, 3, 3>> back(restored.triplet_count());
    restored.copy_to_triplets(back);

    // the import groups into a std::map, i.e. ascending (i, j) — and the source table
    // was built in ascending (i, j) order already, so the sequences must agree
    std::size_t index_mismatches = 0;
    for(std::size_t i = 0; i < back.size(); ++i)
    {
        if(back[i].row_index != downloaded[i].row_index
           || back[i].col_index != downloaded[i].col_index)
            ++index_mismatches;
    }
    CHECK(index_mismatches == 0);

    const double delta = max_abs_diff(block_values(back), block_values(downloaded));
    T83B_LOG("[t83b] roundtrip: %zu blocks, %d x %d blocks, index mismatches %zu, "
             "max |dA| = %.3e\n",
             back.size(),
             n,
             n,
             index_mismatches,
             delta);
    CHECK(delta == 0.0);  // Class A: only the exact double <-> text round trip

    // the explicit zero blocks really survived (they are the reason the writer writes
    // all 9 scalars): count the blocks whose 9 values are all zero
    std::size_t zero_blocks = 0;
    for(const auto& t : back)
    {
        bool all_zero = true;
        for(int r = 0; r < kBlockDim && all_zero; ++r)
            for(int c = 0; c < kBlockDim; ++c)
                if(t.value(r, c) != 0.0)
                {
                    all_zero = false;
                    break;
                }
        if(all_zero)
            ++zero_blocks;
    }
    CHECK(zero_blocks > 0);  // `sparse_table` always emits at least one
    // and the diff would notice if they had been dropped
    CHECK(restored.triplet_count() == source.triplet_count());

    fs::remove(path);
}

// ===========================================================================
//  CASE 3 — the vector writer/reader + its two edges
// ===========================================================================
TEST_CASE("matrix_market_vector_roundtrip", "[luisa][linear][mtx]")
{
    using namespace t83b_mtx;
    test_runtime();

    const std::vector<double> values{1.0, -2.5, 1.0 / 3.0, 0.0, 4e-17, -7.0};
    lla::LDenseVector<double> source;
    source.resize(values.size());
    source.copy_from(std::span<const double>{values});

    const auto dir = workspace_path("uipc_luisa_t83b_mtx_vector");

    // -- the reference's own download (`copy_to(std::vector)`, which synchronises)
    std::vector<double> downloaded;
    source.copy_to(downloaded);
    REQUIRE(downloaded.size() == values.size());

    const auto path = dir + "/v.mtx";
    REQUIRE(uipc::backend::luisa::export_vector_market(std::string_view{path},
                                                       std::span<const double>{downloaded}));

    const auto lines = read_lines(path);
    REQUIRE(lines.size() == 4u + values.size());  // 4 header lines + one per element
    CHECK(lines[0] == "%MatrixMarket matrix array real general");
    CHECK(lines[1] == "% Exported from dense vector");
    CHECK(lines[2] == "% Vector size: 6");
    CHECK(lines[3] == "6 1");  // always one column: this is what makes it a vector
    CHECK(lines[4] == "1");
    CHECK(lines[5] == "-2.5");

    lla::LDenseVector<double> restored;
    REQUIRE(uipc::backend::luisa::import_vector_market(restored, std::string_view{path}));
    CHECK(restored.size() == values.size());

    std::vector<double> back;
    restored.copy_to(back);
    const double delta = max_abs_diff(back, values);
    T83B_LOG("[t83b] vector roundtrip: n %zu, max |dv| = %.3e\n", back.size(), delta);
    CHECK(delta == 0.0);

    // -- the edges ------------------------------------------------------------
    // n == 1 round-trips
    const auto path1 = dir + "/one.mtx";
    std::vector<double> one{3.25};
    REQUIRE(uipc::backend::luisa::export_vector_market(std::string_view{path1},
                                                       std::span<const double>{one}));
    lla::LDenseVector<double> one_back;
    REQUIRE(uipc::backend::luisa::import_vector_market(one_back, std::string_view{path1}));
    CHECK(one_back.size() == 1u);
    std::vector<double> one_host;
    one_back.copy_to(one_host);
    CHECK(one_host.size() == 1u);
    CHECK(one_host[0] == 3.25);

    // n == 0 *exports* (a legal "0 1" header) but does not *import*: the reference's
    // reader rejects `rows <= 0`, and a dump of an empty system is never produced
    // anyway (`solve()` returns before the dump on an empty system).
    const auto path_zero = dir + "/zero.mtx";
    REQUIRE(uipc::backend::luisa::export_vector_market(
        std::string_view{path_zero}, std::span<const double>{}));
    const auto zero_lines = read_lines(path_zero);
    REQUIRE(zero_lines.size() == 4u);
    CHECK(zero_lines[3] == "0 1");
    lla::LDenseVector<double> zero_back;
    CHECK(!uipc::backend::luisa::import_vector_market(zero_back, std::string_view{path_zero}));

    fs::remove(path);
    fs::remove(path1);
    fs::remove(path_zero);
}

// ===========================================================================
//  CASE 4 — every failure path returns false (never throws, never "succeeds")
// ===========================================================================
TEST_CASE("matrix_market_negative_controls", "[luisa][linear][mtx]")
{
    using namespace t83b_mtx;
    test_runtime();

    namespace mm = uipc::backend::luisa;
    const auto dir = workspace_path("uipc_luisa_t83b_mtx_negative");

    // 1) a missing file
    lla::LTripletMatrix<double, 3, 3> m;
    CHECK(!mm::import_matrix_market(m, std::string_view{dir + "/does_not_exist.mtx"}));
    lla::LDenseVector<double> v;
    CHECK(!mm::import_vector_market(v, std::string_view{dir + "/does_not_exist.mtx"}));

    // 2) a vector header with cols != 1 is a matrix, not a vector
    {
        const auto path = dir + "/two_cols.mtx";
        std::ofstream out{path};
        out << "%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n";
        out.close();
        lla::LDenseVector<double> vec;
        CHECK(!mm::import_vector_market(vec, std::string_view{path}));
        fs::remove(path);
    }

    // 3) a truncated body: the dims line promises 3 values, the file has 1
    {
        const auto path = dir + "/truncated.mtx";
        std::ofstream out{path};
        out << "%%MatrixMarket matrix array real general\n3 1\n1.5\n";
        out.close();
        lla::LDenseVector<double> vec;
        CHECK(!mm::import_vector_market(vec, std::string_view{path}));
        fs::remove(path);
    }

    // 4) a matrix file with no data line at all (header only)
    {
        const auto path = dir + "/header_only.mtx";
        std::ofstream out{path};
        out << "%%MatrixMarket matrix coordinate real general\n";
        out.close();
        lla::LTripletMatrix<double, 3, 3> mat;
        CHECK(!mm::import_matrix_market(mat, std::string_view{path}));
        fs::remove(path);
    }

    // 5) an unwritable path (a directory that does not exist and is never created)
    {
        const auto bad = dir + "/no_such_dir_83b/x.mtx";
        fs::remove_all(dir + "/no_such_dir_83b");
        const std::vector<lla::MatrixTriplet<double, 3, 3>> ts{make_triplet(0, 0, Block{})};
        CHECK(!mm::export_matrix_market(std::string_view{bad},
                                        std::span<const lla::MatrixTriplet<double, 3, 3>>{ts},
                                        1,
                                        1));
        std::vector<double> values{1.0};
        CHECK(!mm::export_vector_market(std::string_view{bad},
                                       std::span<const double>{values}));
    }

    // 6) `one_based = false` round-trips as well (the writer and the reader agree)
    {
        const auto ts = sparse_table(4);
        lla::LTripletMatrix<double, 3, 3> source;
        source.reshape(4, 4);
        source.copy_from_triplets(std::span<const lla::MatrixTriplet<double, 3, 3>>{ts});
        std::vector<lla::MatrixTriplet<double, 3, 3>> host(source.triplet_count());
        source.copy_to_triplets(host);

        const auto path = dir + "/zero_based.mtx";
        REQUIRE(mm::export_matrix_market(
            std::string_view{path}, std::span<const lla::MatrixTriplet<double, 3, 3>>{host}, 4, 4, false));
        // the first record must be the first table entry with 0-based indices
        const auto lines = read_lines(path);
        REQUIRE(lines.size() > 6u);
        const auto fields = split_ws(lines[6]);
        REQUIRE(fields.size() == 3u);
        REQUIRE(!host.empty());
        CHECK(fields[0] == std::to_string(host.front().row_index));
        CHECK(fields[1] == std::to_string(host.front().col_index * kBlockDim));
        // a 0-based file may contain a 0 index, a 1-based one may not (asserted in
        // `matrix_market_format_exact`), and the *value* is the first table entry's
        CHECK(fields[2] != "");
        CHECK(std::stod(fields[2]) == host.front().value(0, 0));

        lla::LTripletMatrix<double, 3, 3> restored;
        REQUIRE(mm::import_matrix_market(restored, std::string_view{path}, false));
        std::vector<lla::MatrixTriplet<double, 3, 3>> back(restored.triplet_count());
        restored.copy_to_triplets(back);
        CHECK(back.size() == host.size());
        CHECK(max_abs_diff(block_values(back), block_values(host)) == 0.0);
        fs::remove(path);
    }
}
