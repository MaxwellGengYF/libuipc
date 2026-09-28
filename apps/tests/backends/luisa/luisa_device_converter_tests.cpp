/********************************************************************
 * @file luisa_device_converter_tests.cpp
 * @brief T3.3 — tests for the MatrixConverter port
 *        (luisa_device/linear/converter.h).
 *
 * Oracle.  The reference converter is CUDA-only (<<<>>> kernels, cudaStream_t,
 * CUB) and cannot be host-included, so the oracle here is an *independent CPU
 * model written from the mathematical definition* of each conversion — group by
 * (row, col), sum the duplicates, sort the groups; count the blocks per row and
 * prefix-sum the counts; mirror the off-diagonal blocks transposed — never a
 * transcription of the reference's kernels.  (T5.4/T8.3 will compare assembled
 * matrices against the real CUDA run; this file is the conversion-level check.)
 *
 * Coverage (brief §4):
 *  1. Triplet -> BCOO, block sizes 1 and 3, a 1 %–50 % density sweep of a 200x200
 *     block matrix (plus a larger one), with duplicates injected: row/col indices
 *     identical to the model, block values equal within 1e-6, and the
 *     order-insensitivity of the duplicate accumulation (a shuffled input gives
 *     bit-identical output when the values are exactly representable);
 *  2. BCOO -> BSR, block sizes 1 and 3: row pointers and column indices identical
 *     to the model, values within 1e-6, with empty rows, a full row, a single
 *     entry and a 0-entry matrix;
 *  3. Doublet -> BCOO vector, segment sizes 1 and 3: indices identical, values
 *     summed per index;
 *  4. ge2sym / sym2ge: the exact expansion (diagonal kept once, off-diagonals
 *     mirrored and transposed) against the model, and sym2ge(ge2sym(A)) == A for
 *     a symmetric A;
 *  5. end-to-end: a small FEM-like symmetric 3x3-block system, assembled as
 *     triplets -> ge2sym -> BCOO -> BSR, then y = A x with a device gather/scatter
 *     over the converted structure vs a dense CPU A x (within 1e-5) — both the
 *     symmetric (upper-packed) and the sym2ge-expanded general form;
 *  6. determinism: the same input converted twice, compared bitwise (R6);
 *  7. the loose_resize/reserve_ratio path: repeated conversions with growing (and
 *     shrinking) sizes on the same converter object.
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/linear/linear.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace lcrt = ::luisa::compute;

using Float = double;  // uipc::Float is double

namespace
{
// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------
struct Lcg
{
    std::uint64_t s = 0x1234'5678'9abc'def0ULL;
    std::uint64_t next_u64() noexcept
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s;
    }
    int next_int(int lo, int hi) noexcept  // [lo, hi]
    {
        return lo + static_cast<int>((next_u64() >> 33) % static_cast<std::uint64_t>(hi - lo + 1));
    }
    /// Uniform in [-1, 1).
    double next() noexcept
    {
        return static_cast<double>((next_u64() >> 11) & 0xfffffULL) / 524288.0 - 1.0;
    }
};

bool close(double a, double b, double tol)
{
    if(std::isnan(a) || std::isnan(b))
        return std::isnan(a) && std::isnan(b);
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

/// The stream the converter is driven with.  Leaked on purpose: an LStream
/// destroyed during static destruction would call into the LC runtime after
/// zzz_runtime_shutdown tore it down (cookbook §7.4 / §7.10).
lud::LStream& conv_stream()
{
    static auto* s = new lud::LStream{lud::main_stream()};
    return *s;
}

/// Deterministic Fisher-Yates (the order-insensitivity check).
template <typename T>
void shuffle_in_place(std::vector<T>& v, std::uint64_t seed)
{
    Lcg rng;
    rng.s = seed;
    for(std::size_t i = v.size(); i > 1u; --i)
        std::swap(v[i - 1u], v[static_cast<std::size_t>(rng.next_u64() % i)]);
}

template <int M, int N>
double block_value(const typename lla::MatrixTriplet<Float, M, N>::ValueT& v, int r, int c)
{
    if constexpr(M == 1 && N == 1)
        return v;
    else
        return v(r, c);
}

template <int M, int N>
void set_block(typename lla::MatrixTriplet<Float, M, N>::ValueT& v, int r, int c, double x)
{
    if constexpr(M == 1 && N == 1)
    {
        (void)r;
        (void)c;
        v = x;
    }
    else
    {
        v(r, c) = x;
    }
}

// ---------------------------------------------------------------------------
//  the independent CPU model (from the mathematical definition of each
//  conversion — not a transcription of the reference's kernels)
// ---------------------------------------------------------------------------
/// The COO/BCOO result a conversion must produce: one entry per unique (row, col)
/// pair, in ascending (row, col) order, blocks column-major (`comp` scalars).
struct HostCoo
{
    int                       rows = 0, cols = 0, comp = 1;
    std::vector<std::int32_t> row, col;
    std::vector<double>       values;

    std::size_t nnz() const { return row.size(); }
    /// Block (r, c) of entry i.
    double at(std::size_t i, int r, int c) const
    {
        const int d = static_cast<int>(std::lround(std::sqrt(static_cast<double>(comp))));
        return values[i * static_cast<std::size_t>(comp)
                      + static_cast<std::size_t>(c * d + r)];
    }
};

struct HostBsr
{
    int                       rows = 0, cols = 0, comp = 1;
    std::vector<std::int32_t> row_offsets, col;
    std::vector<double>       values;
};

struct HostCooVec
{
    int                       count = 0, comp = 1;
    std::vector<std::int32_t> index;
    std::vector<double>       values;
};

/// Triplet -> BCOO: group by (row, col), sum the block of every member of a
/// group, one entry per group in ascending (row, col) order (`std::map`'s own
/// ordering is that order).
template <int M, int N>
HostCoo cpu_triplets_to_bcoo(const std::vector<lla::MatrixTriplet<Float, M, N>>& tris,
                             int                                                 rows,
                             int                                                 cols)
{
    constexpr int comp = M * N;
    std::map<std::pair<int, int>, std::vector<double>> groups;
    for(const auto& t : tris)
    {
        auto& g = groups[{t.row_index, t.col_index}];
        if(g.empty())
            g.assign(comp, 0.0);
        for(int c = 0; c < N; ++c)
            for(int r = 0; r < M; ++r)
                g[static_cast<std::size_t>(c * M + r)] += block_value<M, N>(t.value, r, c);
    }

    HostCoo out;
    out.rows = rows;
    out.cols = cols;
    out.comp = comp;
    for(const auto& [ij, g] : groups)
    {
        out.row.push_back(ij.first);
        out.col.push_back(ij.second);
        out.values.insert(out.values.end(), g.begin(), g.end());
    }
    return out;
}

/// BCOO -> BSR by definition: row_offsets[r] is the number of blocks in the rows
/// before r (with row_offsets[rows] == nnz) and the blocks keep the BCOO order.
HostBsr cpu_bcoo_to_bsr(const HostCoo& coo)
{
    HostBsr out;
    out.rows = coo.rows;
    out.cols = coo.cols;
    out.comp = coo.comp;
    out.row_offsets.assign(static_cast<std::size_t>(coo.rows) + 1u, 0);
    for(auto r : coo.row)
        out.row_offsets[static_cast<std::size_t>(r) + 1u] += 1;
    for(std::size_t r = 0; r + 1u < out.row_offsets.size(); ++r)
        out.row_offsets[r + 1u] += out.row_offsets[r];
    out.col    = coo.col;
    out.values = coo.values;
    return out;
}

/// Doublet -> BCOO vector: sum the segments of every repeated index, one entry
/// per distinct index, ascending.
template <int N>
HostCooVec cpu_doublets_to_bcoo_vector(const std::vector<lla::VectorDoublet<Float, N>>& ds, int count)
{
    constexpr int comp = N;
    std::map<int, std::vector<double>> groups;
    for(const auto& d : ds)
    {
        auto& g = groups[d.index];
        if(g.empty())
            g.assign(comp, 0.0);
        for(int k = 0; k < N; ++k)
        {
            if constexpr(N == 1)
                g[0] += d.value;
            else
                g[static_cast<std::size_t>(k)] += d.value(k);
        }
    }
    HostCooVec out;
    out.count = count;
    out.comp  = comp;
    for(const auto& [i, g] : groups)
    {
        out.index.push_back(i);
        out.values.insert(out.values.end(), g.begin(), g.end());
    }
    return out;
}

/// ge2sym: keep the entries with row <= col, in input order (a stable filter —
/// no sorting, no transposition).
HostCoo cpu_ge2sym(const HostCoo& in)
{
    HostCoo out;
    out.rows = in.rows;
    out.cols = in.cols;
    out.comp = in.comp;
    for(std::size_t i = 0; i < in.nnz(); ++i)
    {
        if(in.row[i] <= in.col[i])
        {
            out.row.push_back(in.row[i]);
            out.col.push_back(in.col[i]);
            out.values.insert(out.values.end(),
                              in.values.begin() + static_cast<std::ptrdiff_t>(i * in.comp),
                              in.values.begin() + static_cast<std::ptrdiff_t>((i + 1u) * in.comp));
        }
    }
    return out;
}

/// sym2ge: the general matrix of an upper-packed symmetric one —
/// `G(i,j) = A(i,j)` and `G(j,i) = A(i,j)^T` — in ascending (row, col) order
/// (the converter sorts the result it builds as [Diag | Upper | Lower]).
HostCoo cpu_sym2ge(const HostCoo& in)
{
    const int d = static_cast<int>(std::lround(std::sqrt(static_cast<double>(in.comp))));
    std::map<std::pair<int, int>, std::vector<double>> g;
    for(std::size_t i = 0; i < in.nnz(); ++i)
    {
        const int r = in.row[i];
        const int c = in.col[i];
        auto&     upper = g[{r, c}];
        upper.assign(static_cast<std::size_t>(in.comp), 0.0);
        for(int k = 0; k < in.comp; ++k)
            upper[static_cast<std::size_t>(k)] = in.values[i * static_cast<std::size_t>(in.comp)
                                                            + static_cast<std::size_t>(k)];
        if(r != c)
        {
            auto& lower = g[{c, r}];
            lower.assign(static_cast<std::size_t>(in.comp), 0.0);
            // transposed block, column-major storage: T(x, y) = A(y, x)
            for(int y = 0; y < d; ++y)
                for(int x = 0; x < d; ++x)
                    lower[static_cast<std::size_t>(y * d + x)] =
                        upper[static_cast<std::size_t>(x * d + y)];
        }
    }

    HostCoo out;
    out.rows = in.rows;
    out.cols = in.cols;
    out.comp = in.comp;
    for(const auto& [ij, b] : g)
    {
        out.row.push_back(ij.first);
        out.col.push_back(ij.second);
        out.values.insert(out.values.end(), b.begin(), b.end());
    }
    return out;
}

// ---------------------------------------------------------------------------
//  host read-back of the converted containers
// ---------------------------------------------------------------------------
/// Host read-back of a (block) triplet/BCOO matrix — the two containers share
/// the accessor surface, so one helper serves both.
template <typename MatrixT>
HostCoo read_bcoo(const MatrixT& m)
{
    constexpr int M = MatrixT::block_rows;
    constexpr int N = MatrixT::block_cols;
    HostCoo       out;
    out.rows = m.rows();
    out.cols = m.cols();
    out.comp = M * N;
    const auto n = m.triplet_count();
    if(n == 0u)
        return out;
    out.row.resize(n);
    out.col.resize(n);
    out.values.resize(n * static_cast<std::size_t>(M * N));
    m.copy_to_row_indices(std::span<std::int32_t>{out.row});
    m.copy_to_col_indices(std::span<std::int32_t>{out.col});
    m.copy_to_values(std::span<double>{out.values});
    return out;
}

template <int N>
HostBsr read_bsr(const lla::LBSRMatrix<Float, N>& m)
{
    HostBsr out;
    out.rows = m.rows();
    out.cols = m.cols();
    out.comp = N * N;
    const auto n = static_cast<std::size_t>(m.non_zeros());
    out.row_offsets.resize(static_cast<std::size_t>(m.rows()) + 1u);
    out.col.resize(n);
    out.values.resize(n * static_cast<std::size_t>(N * N));
    if(n == 0u)
    {
        m.copy_to_host(std::span<std::int32_t>{out.row_offsets}, std::span<std::int32_t>{out.col},
                       std::span<double>{out.values});
        return out;
    }
    m.copy_to_host(std::span<std::int32_t>{out.row_offsets}, std::span<std::int32_t>{out.col},
                   std::span<double>{out.values});
    return out;
}

template <int N>
HostCooVec read_bcoo_vector(const lla::LBCOOVector<Float, N>& v)
{
    HostCooVec out;
    out.count = v.count();
    out.comp  = N;
    const auto n = v.doublet_count();
    if(n == 0u)
        return out;
    out.index.resize(n);
    out.values.resize(n * static_cast<std::size_t>(N));
    v.copy_to_indices(std::span<std::int32_t>{out.index});
    v.copy_to_values(std::span<double>{out.values});
    return out;
}

// ---------------------------------------------------------------------------
//  comparisons
// ---------------------------------------------------------------------------
bool same_ints(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b)
{
    return a == b;
}

bool same_doubles(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

/// Worst absolute / relative deviation between two flat scalar arrays.
double worst_diff(const std::vector<double>& got, const std::vector<double>& want)
{
    double worst = 0.0;
    for(std::size_t i = 0; i < got.size() && i < want.size(); ++i)
        worst = std::max(worst,
                         std::abs(got[i] - want[i])
                             / std::max(1.0, std::max(std::abs(got[i]), std::abs(want[i]))));
    return worst;
}

struct Stats
{
    double worst_bcoo   = 0.0;
    double worst_bsr    = 0.0;
    double worst_vec    = 0.0;
    double worst_ge2sym = 0.0;
    double worst_sym2ge = 0.0;
    double worst_spmv   = 0.0;
};
Stats g_stats{};

// ---------------------------------------------------------------------------
//  input generators
// ---------------------------------------------------------------------------
/// `nnz` random triplets on a `rows x cols` block matrix, with every (row, col)
/// pair repeated with probability `duplicate_density` (the same pair several
/// times — the duplicate-accumulation path).
template <int M, int N>
std::vector<lla::MatrixTriplet<Float, M, N>> make_triplets(std::size_t nnz,
                                                           int         rows,
                                                           int         cols,
                                                           std::uint64_t seed,
                                                           bool          exact_values)
{
    Lcg         rng;
    rng.s = seed;
    std::vector<lla::MatrixTriplet<Float, M, N>> out;
    out.reserve(nnz);
    for(std::size_t i = 0; i < nnz; ++i)
    {
        lla::MatrixTriplet<Float, M, N> t;
        if(!out.empty() && (rng.next_u64() % 4u) == 0u)
        {
            // duplicate a previous (row, col) pair with fresh values
            const auto& prev = out[static_cast<std::size_t>(rng.next_u64() % out.size())];
            t.row_index      = prev.row_index;
            t.col_index      = prev.col_index;
        }
        else
        {
            t.row_index = rng.next_int(0, rows - 1);
            t.col_index = rng.next_int(0, cols - 1);
        }
        for(int c = 0; c < N; ++c)
            for(int r = 0; r < M; ++r)
            {
                // exactly representable values (small integers) for the
                // order-insensitivity check, random doubles otherwise
                set_block<M, N>(t.value, r, c,
                                exact_values ? static_cast<double>(rng.next_int(-8, 8)) : rng.next());
            }
        out.push_back(t);
    }
    return out;
}

/// `nnz` doublets with repeated indices.
template <int N>
std::vector<lla::VectorDoublet<Float, N>> make_doublets(std::size_t   nnz,
                                                        int           segments,
                                                        std::uint64_t seed,
                                                        bool          exact_values)
{
    Lcg  rng;
    rng.s = seed;
    std::vector<lla::VectorDoublet<Float, N>> out;
    out.reserve(nnz);
    for(std::size_t i = 0; i < nnz; ++i)
    {
        lla::VectorDoublet<Float, N> d;
        if(!out.empty() && (rng.next_u64() % 4u) == 0u)
        {
            d.index = out[static_cast<std::size_t>(rng.next_u64() % out.size())].index;
        }
        else
        {
            d.index = rng.next_int(0, segments - 1);
        }
        for(int k = 0; k < N; ++k)
        {
            const double x = exact_values ? static_cast<double>(rng.next_int(-8, 8)) : rng.next();
            if constexpr(N == 1)
                d.value = x;
            else
                d.value(k) = x;
        }
        out.push_back(d);
    }
    return out;
}

// ---------------------------------------------------------------------------
//  conversion drivers (the container + converter round trip)
// ---------------------------------------------------------------------------
template <int M, int N>
HostCoo convert_triplets_to_bcoo(lla::LMatrixConverter<Float, N>&      conv,
                                 const std::vector<lla::MatrixTriplet<Float, M, N>>& tris,
                                 int   rows,
                                 int   cols,
                                 lla::LBCOOMatrix<Float, M, N>&        out)
{
    lla::LTripletMatrix<Float, M, N> tri;
    tri.reshape(rows, cols);
    tri.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, M, N>>{tris});
    conv.convert(tri, out, conv_stream());
    conv_stream().synchronize();
    return read_bcoo(out);
}
}// namespace

// ===========================================================================
//  1. Triplet -> BCOO: density sweep, duplicates, order-insensitivity
// ===========================================================================
TEST_CASE("converter_triplet_to_bcoo_sweep", "[luisa][converter]")
{
    test_runtime();

    // block size 1 and 3, densities 1 %..50 % of a 200x200 block matrix
    for(const double density : {0.01, 0.05, 0.10, 0.25, 0.50})
    {
        const int   n          = 200;
        const auto  nnz        = static_cast<std::size_t>(density * n * n);
        const auto  seed       = static_cast<std::uint64_t>(1000.0 * density);

        {
            lla::LMatrixConverter<Float, 1> conv;
            lla::LBCOOMatrix<Float, 1>      out;
            auto tris = make_triplets<1, 1>(nnz, n, n, seed, false);
            auto got  = convert_triplets_to_bcoo<1, 1>(conv, tris, n, n, out);
            auto want = cpu_triplets_to_bcoo<1, 1>(tris, n, n);
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            CHECK(got.nnz() == want.nnz());
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_bcoo  = std::max(g_stats.worst_bcoo, worst);
            CHECK(worst < 1e-6);

            // order-insensitive duplicate accumulation: shuffling the input must
            // not change the result (exactly representable values -> bitwise)
            auto       tris_exact = make_triplets<1, 1>(nnz, n, n, seed, true);
            auto       want_exact = cpu_triplets_to_bcoo<1, 1>(tris_exact, n, n);
            lla::LBCOOMatrix<Float, 1> out_a, out_b;
            auto got_a = convert_triplets_to_bcoo<1, 1>(conv, tris_exact, n, n, out_a);
            shuffle_in_place(tris_exact, seed + 7u);
            auto got_b = convert_triplets_to_bcoo<1, 1>(conv, tris_exact, n, n, out_b);
            CHECK(got_a.row == got_b.row);
            CHECK(got_a.col == got_b.col);
            CHECK(same_doubles(got_a.values, got_b.values));
            CHECK(same_doubles(got_a.values, want_exact.values));
        }

        {
            lla::LMatrixConverter<Float, 3> conv;
            lla::LBCOOMatrix<Float, 3>      out;
            auto tris = make_triplets<3, 3>(nnz, n, n, seed + 1u, false);
            auto got  = convert_triplets_to_bcoo<3, 3>(conv, tris, n, n, out);
            auto want = cpu_triplets_to_bcoo<3, 3>(tris, n, n);
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            CHECK(got.nnz() == want.nnz());
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_bcoo = std::max(g_stats.worst_bcoo, worst);
            CHECK(worst < 1e-6);

            auto       tris_exact = make_triplets<3, 3>(nnz, n, n, seed + 2u, true);
            auto       want_exact = cpu_triplets_to_bcoo<3, 3>(tris_exact, n, n);
            lla::LBCOOMatrix<Float, 3> out_a, out_b;
            auto got_a = convert_triplets_to_bcoo<3, 3>(conv, tris_exact, n, n, out_a);
            shuffle_in_place(tris_exact, seed + 9u);
            auto got_b = convert_triplets_to_bcoo<3, 3>(conv, tris_exact, n, n, out_b);
            CHECK(got_a.row == got_b.row);
            CHECK(got_a.col == got_b.col);
            CHECK(same_doubles(got_a.values, got_b.values));
            CHECK(same_doubles(got_a.values, want_exact.values));
        }
    }

    // one larger matrix (block size 3): 600x600 blocks at 4 %
    {
        const int  n    = 600;
        const auto nnz  = static_cast<std::size_t>(0.04 * n * n);
        lla::LMatrixConverter<Float, 3> conv;
        lla::LBCOOMatrix<Float, 3>      out;
        auto tris = make_triplets<3, 3>(nnz, n, n, 0xabcdu, false);
        auto got  = convert_triplets_to_bcoo<3, 3>(conv, tris, n, n, out);
        auto want = cpu_triplets_to_bcoo<3, 3>(tris, n, n);
        CHECK(got.row == want.row);
        CHECK(got.col == want.col);
        const double worst = worst_diff(got.values, want.values);
        g_stats.worst_bcoo = std::max(g_stats.worst_bcoo, worst);
        CHECK(worst < 1e-6);
        std::printf("    [triplet->bcoo] larger: %zu triplets -> %zu unique blocks, worst %.2e\n",
                    nnz, got.nnz(), worst);
    }

    std::printf("    [triplet->bcoo] density 1%%-50%% @200x200 (block 1 and 3) + 600x600 @4%%, "
                "worst relative deviation %.2e\n",
                g_stats.worst_bcoo);
}

TEST_CASE("converter_duplicate_accumulation_extremes", "[luisa][converter]")
{
    test_runtime();
    // every triplet the same (row, col): one run of `nnz` elements, i.e. a
    // segment that spans many warps and the 256-element tile boundaries (the
    // cross-warp atomic path of the segmented reduce)
    for(const std::size_t nnz : {std::size_t{1}, std::size_t{33}, std::size_t{255},
                                 std::size_t{256}, std::size_t{257}, std::size_t{1024},
                                 std::size_t{5000}})
    {
        lla::LMatrixConverter<Float, 3> conv;
        lla::LBCOOMatrix<Float, 3>      out;
        std::vector<lla::MatrixTriplet<Float, 3>> tris(nnz);
        for(std::size_t i = 0; i < nnz; ++i)
        {
            tris[i].row_index = 3;
            tris[i].col_index = 4;
            for(int c = 0; c < 3; ++c)
                for(int r = 0; r < 3; ++r)
                    tris[i].value(r, c) = static_cast<double>(1 + (i % 5));
        }
        auto got  = convert_triplets_to_bcoo<3, 3>(conv, tris, 8, 8, out);
        auto want = cpu_triplets_to_bcoo<3, 3>(tris, 8, 8);
        CHECK(got.nnz() == 1u);
        CHECK(got.row == want.row);
        CHECK(got.col == want.col);
        CHECK(same_doubles(got.values, want.values));  // exact integers
    }

    // one very long run plus short runs, mixed with singleton rows
    {
        const std::size_t nnz = 4096;
        std::vector<lla::MatrixTriplet<Float, 1>> tris;
        tris.reserve(nnz);
        Lcg rng;
        rng.s = 0x5eedu;
        for(std::size_t i = 0; i < nnz; ++i)
        {
            lla::MatrixTriplet<Float, 1> t;
            t.row_index = static_cast<int>(i % 64u);
            t.col_index = static_cast<int>((i / 64u) % 8u);
            t.value     = static_cast<double>(rng.next_int(-8, 8));
            tris.push_back(t);
        }
        lla::LMatrixConverter<Float, 1> conv;
        lla::LBCOOMatrix<Float, 1>      out, out_shuffled;
        auto got  = convert_triplets_to_bcoo<1, 1>(conv, tris, 64, 8, out);
        auto want = cpu_triplets_to_bcoo<1, 1>(tris, 64, 8);
        CHECK(got.row == want.row);
        CHECK(got.col == want.col);
        CHECK(same_doubles(got.values, want.values));
        shuffle_in_place(tris, 0x1234u);
        auto got_s = convert_triplets_to_bcoo<1, 1>(conv, tris, 64, 8, out_shuffled);
        CHECK(same_doubles(got_s.values, want.values));
    }
    std::printf("    [duplicates] runs of 1..5000 identical (i,j), exact integer sums\n");
}

// ===========================================================================
//  2. BCOO -> BSR
// ===========================================================================
namespace
{
template <int M, int N>
void bsr_case(const std::vector<lla::MatrixTriplet<Float, M, N>>& tris, int rows, int cols)
{
    lla::LMatrixConverter<Float, N> conv;
    lla::LBCOOMatrix<Float, M, N>   bcoo;
    auto got_bcoo = convert_triplets_to_bcoo<M, N>(conv, tris, rows, cols, bcoo);
    auto want_coo = cpu_triplets_to_bcoo<M, N>(tris, rows, cols);
    CHECK(got_bcoo.row == want_coo.row);
    CHECK(got_bcoo.col == want_coo.col);

    lla::LBSRMatrix<Float, N> bsr;
    conv.convert(bcoo, bsr, conv_stream());
    conv_stream().synchronize();
    auto   got  = read_bsr<N>(bsr);
    HostBsr want = cpu_bcoo_to_bsr(want_coo);

    CHECK(got.rows == rows);
    CHECK(got.cols == cols);
    CHECK(got.row_offsets == want.row_offsets);
    CHECK(got.col == want.col);
    const double worst = worst_diff(got.values, want.values);
    g_stats.worst_bsr  = std::max(g_stats.worst_bsr, worst);
    CHECK(worst < 1e-6);
    // the structure must be self-consistent: the last row pointer is the nnz
    // count and every row pointer is non-decreasing
    bool monotone = true;
    for(std::size_t r = 0; r + 1u < got.row_offsets.size(); ++r)
        monotone = monotone && got.row_offsets[r] <= got.row_offsets[r + 1u];
    CHECK(monotone);
    CHECK(static_cast<std::size_t>(got.row_offsets.back()) == want.col.size());
}
}// namespace

TEST_CASE("converter_bcoo_to_bsr", "[luisa][converter]")
{
    test_runtime();
    const int n = 200;
    for(const double density : {0.01, 0.05, 0.25, 0.50})
    {
        const auto nnz = static_cast<std::size_t>(density * n * n);
        {
            auto tris = make_triplets<1, 1>(nnz, n, n, 0x2000u + static_cast<std::uint64_t>(nnz), false);
            bsr_case<1, 1>(tris, n, n);
        }
        {
            auto tris = make_triplets<3, 3>(nnz, n, n, 0x3000u + static_cast<std::uint64_t>(nnz), false);
            bsr_case<3, 3>(tris, n, n);
        }
    }

    // empty rows: 8x8 blocks, entries only in rows 0 and 7 plus one full row
    for(const int block : {1, 3})
    {
        const int n = 8;
        if(block == 1)
        {
            std::vector<lla::MatrixTriplet<Float, 1>> tris;
            for(int c = 0; c < n; ++c)
            {
                lla::MatrixTriplet<Float, 1> t;
                t.row_index = 0;
                t.col_index = c;
                t.value     = 1.0 + c;  // a full row
                tris.push_back(t);
            }
            for(int c = 2; c < 5; ++c)
            {
                lla::MatrixTriplet<Float, 1> t;
                t.row_index = 7;
                t.col_index = c;
                t.value     = -double(c);
                tris.push_back(t);
            }
            {
                lla::MatrixTriplet<Float, 1> t;  // a single entry in a row of its own
                t.row_index = 3;
                t.col_index = 4;
                t.value     = 42.0;
                tris.push_back(t);
            }
            bsr_case<1, 1>(tris, n, n);
        }
        else
        {
            std::vector<lla::MatrixTriplet<Float, 3, 3>> tris;
            for(int c = 0; c < n; ++c)
            {
                lla::MatrixTriplet<Float, 3, 3> t;
                t.row_index = 0;
                t.col_index = c;
                for(int rr = 0; rr < 3; ++rr)
                    for(int cc = 0; cc < 3; ++cc)
                        t.value(rr, cc) = 1.0 + rr + cc;
                tris.push_back(t);
            }
            {
                lla::MatrixTriplet<Float, 3, 3> t;
                t.row_index = 5;
                t.col_index = 5;
                for(int rr = 0; rr < 3; ++rr)
                    for(int cc = 0; cc < 3; ++cc)
                        t.value(rr, cc) = (rr == cc) ? 3.0 : 0.0;
                tris.push_back(t);
            }
            bsr_case<3, 3>(tris, n, n);
        }
    }

    // a single entry and a 0-entry matrix
    {
        std::vector<lla::MatrixTriplet<Float, 1>> one(1);
        one[0].row_index = 2;
        one[0].col_index = 5;
        one[0].value     = 7.5;
        bsr_case<1, 1>(one, 6, 6);
        bsr_case<1, 1>(std::vector<lla::MatrixTriplet<Float, 1>>{}, 6, 6);
    }
    {
        std::vector<lla::MatrixTriplet<Float, 3, 3>> one(1);
        one[0].row_index = 1;
        one[0].col_index = 3;
        for(int rr = 0; rr < 3; ++rr)
            for(int cc = 0; cc < 3; ++cc)
                one[0].value(rr, cc) = 2.0 * rr - cc;
        bsr_case<3, 3>(one, 4, 4);
        bsr_case<3, 3>(std::vector<lla::MatrixTriplet<Float, 3, 3>>{}, 4, 4);
    }
    std::printf("    [bcoo->bsr] density sweep + empty rows + full row + single entry + 0 entries, "
                "worst %.2e\n",
                g_stats.worst_bsr);
}

// ===========================================================================
//  3. Doublet -> BCOO vector
// ===========================================================================
TEST_CASE("converter_doublet_to_bcoo_vector", "[luisa][converter]")
{
    test_runtime();

    for(const std::size_t nnz : {std::size_t{0}, std::size_t{1}, std::size_t{9},
                                 std::size_t{1000}, std::size_t{50'000}})
    {
        {
            const int segments = 200;
            lla::LMatrixConverter<Float, 1> conv;
            lla::LBCOOVector<Float, 1>      out;
            auto ds = make_doublets<1>(nnz, segments, 0x4444u + nnz, false);

            lla::LDoubletVector<Float, 1> d;
            d.reshape(segments);
            d.copy_from_doublets(std::span<const lla::VectorDoublet<Float, 1>>{ds});
            conv.convert(d, out, conv_stream());
            conv_stream().synchronize();

            auto      got  = read_bcoo_vector<1>(out);
            HostCooVec want = cpu_doublets_to_bcoo_vector<1>(ds, segments);
            CHECK(got.count == segments);
            CHECK(got.index == want.index);
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_vec  = std::max(g_stats.worst_vec, worst);
            CHECK(worst < 1e-6);
        }
        {
            const int segments = 150;
            lla::LMatrixConverter<Float, 3> conv;
            lla::LBCOOVector<Float, 3>      out;
            auto ds = make_doublets<3>(nnz, segments, 0x5555u + nnz, false);

            lla::LDoubletVector<Float, 3> d;
            d.reshape(segments);
            d.copy_from_doublets(std::span<const lla::VectorDoublet<Float, 3>>{ds});
            conv.convert(d, out, conv_stream());
            conv_stream().synchronize();

            auto      got  = read_bcoo_vector<3>(out);
            HostCooVec want = cpu_doublets_to_bcoo_vector<3>(ds, segments);
            CHECK(got.count == segments);
            CHECK(got.index == want.index);
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_vec  = std::max(g_stats.worst_vec, worst);
            CHECK(worst < 1e-6);
        }
    }

    // order-insensitive summation of the repeated indices (exact values)
    {
        const std::size_t nnz      = 4096;
        const int         segments = 64;
        lla::LMatrixConverter<Float, 3> conv;
        lla::LBCOOVector<Float, 3>      out_a, out_b;
        auto ds_a = make_doublets<3>(nnz, segments, 0x6666u, true);

        lla::LDoubletVector<Float, 3> d;
        d.reshape(segments);
        d.copy_from_doublets(std::span<const lla::VectorDoublet<Float, 3>>{ds_a});
        conv.convert(d, out_a, conv_stream());
        conv_stream().synchronize();
        auto got_a = read_bcoo_vector<3>(out_a);

        auto ds_b = ds_a;
        shuffle_in_place(ds_b, 0x7777u);
        lla::LDoubletVector<Float, 3> d2;
        d2.reshape(segments);
        d2.copy_from_doublets(std::span<const lla::VectorDoublet<Float, 3>>{ds_b});
        conv.convert(d2, out_b, conv_stream());
        conv_stream().synchronize();
        auto got_b = read_bcoo_vector<3>(out_b);

        auto want = cpu_doublets_to_bcoo_vector<3>(ds_a, segments);
        CHECK(got_a.index == got_b.index);
        CHECK(same_doubles(got_a.values, got_b.values));
        CHECK(same_doubles(got_a.values, want.values));
    }
    std::printf("    [doublet->bcoo] 0..50k doublets, segment size 1 and 3, worst %.2e\n",
                g_stats.worst_vec);
}

// ===========================================================================
//  4. ge2sym / sym2ge
// ===========================================================================
TEST_CASE("converter_ge2sym", "[luisa][converter]")
{
    test_runtime();
    const int n = 200;
    for(const double density : {0.01, 0.10, 0.50})
    {
        const auto nnz = static_cast<std::size_t>(density * n * n);
        {
            lla::LMatrixConverter<Float, 1> conv;
            lla::LBCOOMatrix<Float, 1>      bcoo;
            auto tris = make_triplets<1, 1>(nnz, n, n, 0x8888u + nnz, false);
            auto coo  = convert_triplets_to_bcoo<1, 1>(conv, tris, n, n, bcoo);

            conv.ge2sym(bcoo, conv_stream());
            conv_stream().synchronize();
            auto       got  = read_bcoo(bcoo);
            const auto want = cpu_ge2sym(coo);
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            CHECK(got.values == want.values);  // a pure compaction: bit-identical
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_ge2sym = std::max(g_stats.worst_ge2sym, worst);
            CHECK(worst < 1e-6);
            bool upper = true;
            for(std::size_t i = 0; i < got.nnz(); ++i)
                upper = upper && got.row[i] <= got.col[i];
            CHECK(upper);
        }
        {
            lla::LMatrixConverter<Float, 3> conv;
            lla::LBCOOMatrix<Float, 3>      bcoo;
            auto tris = make_triplets<3, 3>(nnz, n, n, 0x9999u + nnz, false);
            auto coo  = convert_triplets_to_bcoo<3, 3>(conv, tris, n, n, bcoo);

            conv.ge2sym(bcoo, conv_stream());
            conv_stream().synchronize();
            auto       got  = read_bcoo(bcoo);
            const auto want = cpu_ge2sym(coo);
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            CHECK(same_doubles(got.values, want.values));
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_ge2sym = std::max(g_stats.worst_ge2sym, worst);
            CHECK(worst < 1e-6);
        }
    }

    // ge2sym on the *triplet* container (the reference's FEM path).  A triplet
    // matrix is in assembly order, so ge2sym is a *stable filter* there: the
    // result is the subsequence with row <= col, duplicates and all — sorting and
    // duplicate accumulation only happen in the BCOO/BSR conversion.
    {
        auto tris = make_triplets<3, 3>(5000u, 64, 64, 0xaaaau, false);
        std::vector<lla::MatrixTriplet<Float, 3, 3>> want;
        for(const auto& t : tris)
            if(t.row_index <= t.col_index)
                want.push_back(t);

        lla::LMatrixConverter<Float, 3>  conv;
        lla::LTripletMatrix<Float, 3, 3> tri;
        tri.reshape(64, 64);
        tri.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 3, 3>>{tris});
        conv.ge2sym(tri, conv_stream());
        conv_stream().synchronize();
        auto got = read_bcoo(tri);

        CHECK(got.nnz() == want.size());
        bool same = got.nnz() == want.size();
        for(std::size_t i = 0; same && i < want.size(); ++i)
        {
            same = same && got.row[i] == want[i].row_index && got.col[i] == want[i].col_index;
            for(int r = 0; r < 3 && same; ++r)
                for(int c = 0; c < 3 && same; ++c)
                    same = same
                           && close(got.at(i, r, c), block_value<3, 3>(want[i].value, r, c), 1e-12);
        }
        CHECK(same);
    }
    {
        lla::LMatrixConverter<Float, 1> conv;
        lla::LBCOOMatrix<Float, 1>      empty;
        empty.reshape(4, 4);
        empty.resize_triplets_discard(0u);
        conv.ge2sym(empty, conv_stream());
        conv_stream().synchronize();
        CHECK(empty.non_zeros() == 0);

        lla::LBCOOMatrix<Float, 1> diag;
        diag.reshape(4, 4);
        lla::MatrixTriplet<Float, 1> t;
        t.row_index = 2;
        t.col_index = 2;
        t.value     = 5.0;
        diag.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 1>>{&t, 1u});
        conv.ge2sym(diag, conv_stream());
        conv_stream().synchronize();
        auto got = read_bcoo(diag);
        CHECK(got.nnz() == 1u);
        CHECK(got.row[0] == 2);
        CHECK(got.col[0] == 2);
        CHECK(got.values[0] == 5.0);
    }
    std::printf("    [ge2sym] density sweep + triplet + empty + single diagonal, worst %.2e\n",
                g_stats.worst_ge2sym);
}

TEST_CASE("converter_sym2ge_roundtrip", "[luisa][converter]")
{
    test_runtime();

    // (a) a random general matrix A: ge2sym keeps its upper part U, and
    //     sym2ge(U) must be the general matrix of the symmetric interpretation
    //     of U (diagonal kept once, off-diagonals mirrored + transposed)
    const int n = 150;
    for(const double density : {0.01, 0.10, 0.50})
    {
        const auto nnz = static_cast<std::size_t>(density * n * n);
        for(int block = 0; block < 2; ++block)
        {
            HostCoo upper, got, want;
            if(block == 0)
            {
                lla::LMatrixConverter<Float, 1> conv;
                lla::LBCOOMatrix<Float, 1>      bcoo;
                auto tris = make_triplets<1, 1>(nnz, n, n, 0xbbbbu + nnz, false);
                auto coo  = convert_triplets_to_bcoo<1, 1>(conv, tris, n, n, bcoo);
                upper     = cpu_ge2sym(coo);
                // sym2ge takes the *upper-packed* form (the reference's ge2sym
                // output); produce it on the device and check it as well
                conv.ge2sym(bcoo, conv_stream());
                conv_stream().synchronize();
                const auto upper_dev = read_bcoo(bcoo);
                CHECK(upper_dev.row == upper.row);
                CHECK(upper_dev.col == upper.col);
                CHECK(same_doubles(upper_dev.values, upper.values));
                lla::LBCOOMatrix<Float, 1> general;
                conv.sym2ge(bcoo, general, conv_stream());
                conv_stream().synchronize();
                got  = read_bcoo(general);
                want = cpu_sym2ge(upper);
            }
            else
            {
                lla::LMatrixConverter<Float, 3> conv;
                lla::LBCOOMatrix<Float, 3>      bcoo;
                auto tris = make_triplets<3, 3>(nnz, n, n, 0xccccu + nnz, false);
                auto coo  = convert_triplets_to_bcoo<3, 3>(conv, tris, n, n, bcoo);
                upper     = cpu_ge2sym(coo);
                conv.ge2sym(bcoo, conv_stream());
                conv_stream().synchronize();
                const auto upper_dev = read_bcoo(bcoo);
                CHECK(upper_dev.row == upper.row);
                CHECK(upper_dev.col == upper.col);
                CHECK(same_doubles(upper_dev.values, upper.values));
                lla::LBCOOMatrix<Float, 3> general;
                conv.sym2ge(bcoo, general, conv_stream());
                conv_stream().synchronize();
                got  = read_bcoo(general);
                want = cpu_sym2ge(upper);
            }
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_sym2ge = std::max(g_stats.worst_sym2ge, worst);
            CHECK(worst < 1e-6);

            // the exact expansion: the diagonal is present once, every
            // off-diagonal appears twice (transposed) and the result is sorted
            std::map<std::pair<int, int>, std::size_t> pos;  // (row, col) -> entry
            for(std::size_t i = 0; i < got.nnz(); ++i)
                pos[{got.row[i], got.col[i]}] = i;

            bool sorted = true, mirrored = true;
            std::size_t diag_seen = 0;
            for(std::size_t i = 0; i < got.nnz(); ++i)
            {
                if(i > 0u)
                    sorted = sorted
                             && ((got.row[i] > got.row[i - 1u])
                                 || (got.row[i] == got.row[i - 1u] && got.col[i] > got.col[i - 1u]));
                if(got.row[i] == got.col[i])
                {
                    ++diag_seen;  // exactly one diagonal entry per block row
                }
                else
                {
                    // the transposed counterpart of every off-diagonal entry
                    const auto it = pos.find({got.col[i], got.row[i]});
                    if(it == pos.end())
                    {
                        mirrored = false;
                    }
                    else
                    {
                    const std::size_t j   = it->second;
                    const int         dim = static_cast<int>(std::lround(
                        std::sqrt(static_cast<double>(got.comp))));
                    for(int r = 0; r < dim && mirrored; ++r)
                    {
                        for(int c = 0; c < dim && mirrored; ++c)
                                mirrored = mirrored
                                           && close(got.at(j, r, c), got.at(i, c, r), 1e-12);
                    }
                }
            }
            }
            CHECK(sorted);
            CHECK(mirrored);
            // the size follows from the definition: diag + 2 * off-diagonal,
            // and the diagonal of the packed input is kept exactly once
            std::size_t diag = 0;
            for(std::size_t i = 0; i < upper.nnz(); ++i)
                diag += (upper.row[i] == upper.col[i]) ? 1u : 0u;
            CHECK(got.nnz() == diag + 2u * (upper.nnz() - diag));
            CHECK(diag_seen == diag);
        }
    }

    // (b) sym2ge(ge2sym(A)) == A for a symmetric A
    for(int block = 0; block < 2; ++block)
    {
        const int  m   = 80;
        const auto nnz = static_cast<std::size_t>(0.2 * m * m);
        if(block == 0)
        {
            lla::LMatrixConverter<Float, 1> conv;
            lla::LBCOOMatrix<Float, 1>      bcoo;
            // symmetric input: every (i, j) is accompanied by (j, i) with the
            // same value
            auto tris = make_triplets<1, 1>(nnz, m, m, 0xddddu, false);
            for(auto& t : tris)
                std::swap(t.row_index, t.col_index);
            auto tris2 = make_triplets<1, 1>(nnz, m, m, 0xddddu, false);
            for(auto& t : tris2)
                t.value = -t.value;  // a *different* lower part: ge2sym must drop it
            tris.insert(tris.end(), tris2.begin(), tris2.end());

            auto coo = convert_triplets_to_bcoo<1, 1>(conv, tris, m, m, bcoo);
            conv.ge2sym(bcoo, conv_stream());
            conv_stream().synchronize();
            auto upper = read_bcoo(bcoo);
            lla::LBCOOMatrix<Float, 1> general;
            conv.sym2ge(bcoo, general, conv_stream());
            conv_stream().synchronize();
            auto got  = read_bcoo(general);
            // build the expected symmetric general matrix from the *upper* part
            auto want = cpu_sym2ge(cpu_ge2sym(coo));
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_sym2ge = std::max(g_stats.worst_sym2ge, worst);
            CHECK(worst < 1e-6);
            (void)upper;
        }
        else
        {
            lla::LMatrixConverter<Float, 3> conv;
            lla::LBCOOMatrix<Float, 3>      bcoo;
            auto                            tris = make_triplets<3, 3>(nnz, m, m, 0xeeeeu, false);
            for(auto& t : tris)
                std::swap(t.row_index, t.col_index);
            auto tris2 = make_triplets<3, 3>(nnz, m, m, 0xeeeeu, false);
            for(auto& t : tris2)
                t.value = -t.value;
            tris.insert(tris.end(), tris2.begin(), tris2.end());

            auto coo = convert_triplets_to_bcoo<3, 3>(conv, tris, m, m, bcoo);
            conv.ge2sym(bcoo, conv_stream());
            conv_stream().synchronize();
            lla::LBCOOMatrix<Float, 3> general;
            conv.sym2ge(bcoo, general, conv_stream());
            conv_stream().synchronize();
            auto got  = read_bcoo(general);
            auto want = cpu_sym2ge(cpu_ge2sym(coo));
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            const double worst = worst_diff(got.values, want.values);
            g_stats.worst_sym2ge = std::max(g_stats.worst_sym2ge, worst);
            CHECK(worst < 1e-6);
        }
    }

    // (c) degenerate: an empty upper-packed matrix and a diagonal-only one
    {
        lla::LMatrixConverter<Float, 3> conv;
        lla::LBCOOMatrix<Float, 3>      empty, general;
        empty.reshape(6, 6);
        empty.resize_triplets_discard(0u);
        conv.sym2ge(empty, general, conv_stream());
        conv_stream().synchronize();
        CHECK(general.non_zeros() == 0);
        CHECK(general.rows() == 6);
        CHECK(general.cols() == 6);
    }
    std::printf("    [sym2ge] upper-packed -> general (mirrored + transposed), worst %.2e\n",
                g_stats.worst_sym2ge);
}

// ===========================================================================
//  5. end-to-end: FEM-like assembly -> BCOO/BSR -> y = A x on the device
// ===========================================================================
namespace
{
/// Dense y = A x over a *host* general COO matrix (the CPU oracle of the device
/// SpMV).  `x` has rows * d entries, `d` = sqrt(comp).
std::vector<double> host_dense_spmv(const HostCoo& a, const std::vector<double>& x)
{
    const int   d     = static_cast<int>(std::lround(std::sqrt(static_cast<double>(a.comp))));
    const auto  dofs  = static_cast<std::size_t>(a.rows) * static_cast<std::size_t>(d);
    std::vector<double> y(dofs, 0.0);
    for(std::size_t i = 0; i < a.nnz(); ++i)
    {
        for(int r = 0; r < d; ++r)
        {
            double acc = 0.0;
            for(int c = 0; c < d; ++c)
                acc += a.at(i, r, c)
                       * x[static_cast<std::size_t>(a.col[i]) * static_cast<std::size_t>(d)
                           + static_cast<std::size_t>(c)];
            y[static_cast<std::size_t>(a.row[i]) * static_cast<std::size_t>(d)
              + static_cast<std::size_t>(r)] += acc;
        }
    }
    return y;
}

std::vector<double> make_vector(std::size_t n, std::uint64_t seed)
{
    Lcg                 rng;
    rng.s = seed;
    std::vector<double> out(n);
    for(auto& v : out)
        v = rng.next();
    return out;
}

/// A FEM-like symmetric system: a `grid^3` node lattice, one 8-node cell per
/// grid cell, and a random *symmetric* 3x3 block per node pair (a<=b) written to
/// both (a,b) and (b,a) — every interior node pair is therefore written by many
/// cells (duplicates) and the assembled matrix is symmetric.
std::vector<lla::MatrixTriplet<Float, 3, 3>> make_fem_triplets(int grid, std::uint64_t seed)
{
    Lcg  rng;
    rng.s = seed;
    std::vector<lla::MatrixTriplet<Float, 3, 3>> tris;
    for(int z = 0; z + 1 < grid; ++z)
        for(int y = 0; y + 1 < grid; ++y)
            for(int x = 0; x + 1 < grid; ++x)
            {
                int idx[8];
                for(int c = 0; c < 8; ++c)
                {
                    const int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
                    idx[c] = ((z + dz) * grid + (y + dy)) * grid + (x + dx);
                }
                for(int a = 0; a < 8; ++a)
                    for(int b = a; b < 8; ++b)
                    {
                        // one random symmetric block per unordered pair
                        double v[3][3];
                        for(int r = 0; r < 3; ++r)
                            for(int c = 0; c < 3; ++c)
                                v[r][c] = rng.next();
                        for(int r = 0; r < 3; ++r)
                            for(int c = r; c < 3; ++c)
                                v[r][c] = v[c][r] = 0.5 * (v[r][c] + v[c][r]);
                        for(int r = 0; r < 3; ++r)
                            v[r][r] += 6.0;  // diagonally dominant

                        lla::MatrixTriplet<Float, 3, 3> t;
                        t.row_index = idx[a];
                        t.col_index = idx[b];
                        for(int r = 0; r < 3; ++r)
                            for(int c = 0; c < 3; ++c)
                                t.value(r, c) = v[r][c];
                        tris.push_back(t);

                        if(a != b)
                        {
                            lla::MatrixTriplet<Float, 3, 3> u = t;
                            u.row_index = idx[b];
                            u.col_index = idx[a];
                            tris.push_back(u);
                        }
                    }
            }
    return tris;
}

/// y = A x for a *general* BSR matrix (one row walk per block row; every
/// contribution goes through an atomic so no row's accumulator can be clobbered).
template <int N>
struct BsrSpMVKernel
{
    static auto& get()
    {
        using CView  = lla::LCBSRMatrixDeviceView<lcrt::Var<double>, N>;
        using CDense = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
        using Dense  = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<std::int32_t>,
                                      lcrt::Buffer<std::int32_t>,
                                      lcrt::Buffer<double>,
                                      lcrt::Buffer<double>,
                                      lcrt::Buffer<double>,
                                      std::uint32_t>(
            [](lcrt::BufferVar<std::int32_t> offsets, lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double> vals, lcrt::BufferVar<double> x,
               lcrt::BufferVar<double> y, lcrt::Var<std::uint32_t> rows) noexcept
            {
                auto A  = CView::from_handles(offsets, cols, vals, rows, rows, 0u);
                auto xd = CDense::from_handles(x, rows * static_cast<std::uint32_t>(N));
                auto yd = Dense::from_handles(y, rows * static_cast<std::uint32_t>(N));
                auto r  = lcrt::dispatch_x();
                $if (r < rows)
                {
                    lcrt::Var<std::int32_t> k = A.row_begin(r);
                    auto                    e = A.row_end(r);
                    $while (k < e)
                    {
                        auto c = A.col_index(k);
                        auto b = A.load_block(k);
                        if constexpr(N == 1)
                        {
                            (void)yd.atomic_add(r, b * xd.read(c));  // y_r += B x_c
                        }
                        else
                        {
                            // materialise the segment into an LVec (the
                            // reference's `Vector<N> = seg.as_eigen();` step)
                            lps::LVec<lcrt::Var<double>, N> xc =
                                xd.segment<N>(c * static_cast<std::int32_t>(N)).as_eigen();
                            yd.segment<N>(r * static_cast<std::uint32_t>(N)).atomic_add(b * xc);
                        }
                        k = k + 1;
                    };
                };
            });
        return k;
    }
};

/// y = A x for an **upper-packed symmetric** BSR matrix: the block (r, c > r)
/// contributes `B x_c` to y_r and `B^T x_r` to y_c; a diagonal block contributes
/// once (this is the structure `ge2sym` + `convert` produce for the FEM system).
template <int N>
struct BsrSymSpMVKernel
{
    static auto& get()
    {
        using CView  = lla::LCBSRMatrixDeviceView<lcrt::Var<double>, N>;
        using CDense = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
        using Dense  = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<std::int32_t>,
                                      lcrt::Buffer<std::int32_t>,
                                      lcrt::Buffer<double>,
                                      lcrt::Buffer<double>,
                                      lcrt::Buffer<double>,
                                      std::uint32_t>(
            [](lcrt::BufferVar<std::int32_t> offsets, lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double> vals, lcrt::BufferVar<double> x,
               lcrt::BufferVar<double> y, lcrt::Var<std::uint32_t> rows) noexcept
            {
                auto A  = CView::from_handles(offsets, cols, vals, rows, rows, 0u);
                auto xd = CDense::from_handles(x, rows * static_cast<std::uint32_t>(N));
                auto yd = Dense::from_handles(y, rows * static_cast<std::uint32_t>(N));
                auto r  = lcrt::dispatch_x();
                $if (r < rows)
                {
                    lcrt::Var<std::int32_t> k = A.row_begin(r);
                    auto                    e = A.row_end(r);
                    $while (k < e)
                    {
                        auto c = A.col_index(k);
                        auto b = A.load_block(k);
                        if constexpr(N == 1)
                        {
                            (void)yd.atomic_add(r, b * xd.read(c));
                            $if (c != lcrt::cast<std::int32_t>(r))
                            {
                                (void)yd.atomic_add(c, b * xd.read(r));
                            };
                        }
                        else
                        {
                            lps::LVec<lcrt::Var<double>, N> xc =
                                xd.segment<N>(c * static_cast<std::int32_t>(N)).as_eigen();
                            lps::LVec<lcrt::Var<double>, N> xr =
                                xd.segment<N>(r * static_cast<std::uint32_t>(N)).as_eigen();
                            yd.segment<N>(r * static_cast<std::uint32_t>(N)).atomic_add(b * xc);
                            $if (c != lcrt::cast<std::int32_t>(r))
                            {
                                yd.segment<N>(c * static_cast<std::int32_t>(N))
                                    .atomic_add(b.transpose() * xr);
                            };
                        }
                        k = k + 1;
                    };
                };
            });
        return k;
    }
};

/// y = A x over a (general) BCOO triplet view — the scatter/gather form.
template <int N>
struct BcooSpMVKernel
{
    static auto& get()
    {
        using CView  = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, N, N>;
        using CDense = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
        using Dense  = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<std::int32_t>,
                                      lcrt::Buffer<std::int32_t>,
                                      lcrt::Buffer<double>,
                                      lcrt::Buffer<double>,
                                      lcrt::Buffer<double>,
                                      std::uint32_t,
                                      std::uint32_t>(
            [](lcrt::BufferVar<std::int32_t> rows_i, lcrt::BufferVar<std::int32_t> cols_i,
               lcrt::BufferVar<double> vals, lcrt::BufferVar<double> x,
               lcrt::BufferVar<double> y, lcrt::Var<std::uint32_t> block_rows,
               lcrt::Var<std::uint32_t> n) noexcept
            {
                auto A  = CView::from_handles(rows_i, cols_i, vals, block_rows, block_rows, n);
                auto xd = CDense::from_handles(x, block_rows * static_cast<std::uint32_t>(N));
                auto yd = Dense::from_handles(y, block_rows * static_cast<std::uint32_t>(N));
                auto i  = lcrt::dispatch_x();
                $if (i < n)
                {
                    auto t = A(i);
                    if constexpr(N == 1)
                    {
                        (void)yd.atomic_add(t.row_index, t.value * xd.read(t.col_index));
                    }
                    else
                    {
                        lps::LVec<lcrt::Var<double>, N> xc =
                            xd.segment<N>(t.col_index * static_cast<std::int32_t>(N)).as_eigen();
                        yd.segment<N>(t.row_index * static_cast<std::int32_t>(N))
                            .atomic_add(t.value * xc);
                    }
                };
            });
        return k;
    }
};
}// namespace

TEST_CASE("converter_end_to_end_spmv", "[luisa][converter]")
{
    test_runtime();
    const int  grid  = 4;  // 64 nodes, 27 cells, 64 * 8 blocks per cell
    const auto tris  = make_fem_triplets(grid, 0xf00du);
    const int  nodes = grid * grid * grid;
    const auto dofs  = static_cast<std::size_t>(nodes) * 3u;

    lla::LMatrixConverter<Float, 3> conv;
    lla::LTripletMatrix<Float, 3, 3> tri;
    tri.reshape(nodes, nodes);
    tri.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 3, 3>>{tris});
    CHECK(tri.triplet_count() == tris.size());

    auto want_coo = cpu_triplets_to_bcoo<3, 3>(tris, nodes, nodes);
    auto upper    = cpu_ge2sym(want_coo);
    auto general  = cpu_sym2ge(upper);

    // the reference's FEM path: ge2sym(triplet) -> convert -> BSR.  On the
    // triplet container ge2sym is a stable filter (assembly order, duplicates
    // kept); the conversion is what sorts and accumulates.
    std::vector<lla::MatrixTriplet<Float, 3, 3>> filtered;
    for(const auto& t : tris)
        if(t.row_index <= t.col_index)
            filtered.push_back(t);

    conv.ge2sym(tri, conv_stream());
    conv_stream().synchronize();
    auto got_upper = read_bcoo(tri);
    CHECK(got_upper.nnz() == filtered.size());
    bool filtered_ok = got_upper.nnz() == filtered.size();
    for(std::size_t i = 0; filtered_ok && i < filtered.size(); ++i)
    {
        filtered_ok = filtered_ok && got_upper.row[i] == filtered[i].row_index
                      && got_upper.col[i] == filtered[i].col_index;
        for(int r = 0; r < 3 && filtered_ok; ++r)
            for(int c = 0; c < 3 && filtered_ok; ++c)
                filtered_ok = filtered_ok
                              && close(got_upper.at(i, r, c),
                                       block_value<3, 3>(filtered[i].value, r, c), 1e-12);
    }
    CHECK(filtered_ok);

    lla::LBCOOMatrix<Float, 3, 3> bcoo;
    conv.convert(tri, bcoo, conv_stream());
    conv_stream().synchronize();
    auto got_bcoo = read_bcoo(bcoo);
    CHECK(got_bcoo.row == upper.row);
    CHECK(got_bcoo.col == upper.col);
    CHECK(same_doubles(got_bcoo.values, upper.values));

    lla::LBSRMatrix<Float, 3> bsr;
    conv.convert(bcoo, bsr, conv_stream());
    conv_stream().synchronize();

    // (a) the symmetric SpMV over the upper-packed BSR must reproduce the dense
    //     product of the symmetric matrix
    const auto x = make_vector(dofs, 0xbeefu);
    const auto y_want = host_dense_spmv(general, x);

    lud::LDeviceBuffer<double> x_dev, y_dev;
    x_dev.copy_from(std::span<const double>{x});
    y_dev.resize(dofs);
    y_dev.fill(0.0);
    conv_stream().synchronize();
    conv_stream().record(BsrSymSpMVKernel<3>::get()(bsr.row_offsets(), bsr.col_indices(),
                                                    bsr.values(), x_dev.view(), y_dev.view(),
                                                    static_cast<std::uint32_t>(nodes))
                             .dispatch(static_cast<std::uint32_t>(nodes)));
    conv_stream().synchronize();
    std::vector<double> y_got(dofs, 0.0);
    y_dev.view().copy_to(std::span<double>{y_got});
    lud::main_stream().synchronize();
    double worst_sym = 0.0;
    for(std::size_t i = 0; i < dofs; ++i)
        worst_sym = std::max(worst_sym, std::abs(y_got[i] - y_want[i]) / std::max(1.0, std::abs(y_want[i])));
    CHECK(worst_sym < 1e-5);

    // (b) the same matrix expanded by sym2ge and used through a general SpMV
    lla::LBCOOMatrix<Float, 3, 3> general_bcoo;
    conv.sym2ge(bcoo, general_bcoo, conv_stream());
    conv_stream().synchronize();
    auto got_general = read_bcoo(general_bcoo);
    CHECK(got_general.row == general.row);
    CHECK(got_general.col == general.col);
    CHECK(worst_diff(got_general.values, general.values) < 1e-6);

    lla::LBSRMatrix<Float, 3> general_bsr;
    conv.convert(general_bcoo, general_bsr, conv_stream());
    conv_stream().synchronize();
    y_dev.fill(0.0);
    conv_stream().synchronize();
    conv_stream().record(BsrSpMVKernel<3>::get()(general_bsr.row_offsets(), general_bsr.col_indices(),
                                                 general_bsr.values(), x_dev.view(), y_dev.view(),
                                                 static_cast<std::uint32_t>(nodes))
                             .dispatch(static_cast<std::uint32_t>(nodes)));
    conv_stream().synchronize();
    std::vector<double> y_got2(dofs, 0.0);
    y_dev.view().copy_to(std::span<double>{y_got2});
    lud::main_stream().synchronize();
    double worst_gen = 0.0;
    for(std::size_t i = 0; i < dofs; ++i)
        worst_gen = std::max(worst_gen, std::abs(y_got2[i] - y_want[i]) / std::max(1.0, std::abs(y_want[i])));
    CHECK(worst_gen < 1e-5);

    // (c) the BCOO scatter/gather form over the general triplets
    y_dev.fill(0.0);
    conv_stream().synchronize();
    conv_stream().record(BcooSpMVKernel<3>::get()(general_bcoo.row_indices(), general_bcoo.col_indices(),
                                                  general_bcoo.values(), x_dev.view(), y_dev.view(),
                                                  static_cast<std::uint32_t>(nodes),
                                                  static_cast<std::uint32_t>(general_bcoo.non_zeros()))
                             .dispatch(static_cast<std::uint32_t>(general_bcoo.non_zeros())));
    conv_stream().synchronize();
    std::vector<double> y_got3(dofs, 0.0);
    y_dev.view().copy_to(std::span<double>{y_got3});
    lud::main_stream().synchronize();
    double worst_bcoo = 0.0;
    for(std::size_t i = 0; i < dofs; ++i)
        worst_bcoo = std::max(worst_bcoo, std::abs(y_got3[i] - y_want[i]) / std::max(1.0, std::abs(y_want[i])));
    CHECK(worst_bcoo < 1e-5);

    g_stats.worst_spmv = std::max({worst_sym, worst_gen, worst_bcoo});
    std::printf("    [e2e] FEM-like %d nodes (%zu dofs), %zu assembled triplets -> %zu unique "
                "(upper) / %zu general blocks; SpMV rel err: symmetric %.2e, general-BSR %.2e, "
                "BCOO %.2e\n",
                nodes, dofs, tris.size(), upper.nnz(), general.nnz(), worst_sym, worst_gen, worst_bcoo);
}

TEST_CASE("converter_end_to_end_spmv_block1", "[luisa][converter]")
{
    test_runtime();
    // the same shape with block size 1 (a scalar FEM-like system)
    const int  n   = 60;
    const auto nnz = static_cast<std::size_t>(3.0 * n);
    auto       tris_a = make_triplets<1, 1>(nnz, n, n, 0x1111u, false);
    std::vector<lla::MatrixTriplet<Float, 1>> tris;
    for(const auto& t : tris_a)
    {
        tris.push_back(t);
        auto u      = t;
        u.row_index = t.col_index;
        u.col_index = t.row_index;  // a symmetric pair
        tris.push_back(u);
    }

    lla::LMatrixConverter<Float, 1> conv;
    lla::LTripletMatrix<Float, 1, 1> tri;
    tri.reshape(n, n);
    tri.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 1>>{tris});

    auto want_coo = cpu_triplets_to_bcoo<1, 1>(tris, n, n);
    auto general  = cpu_sym2ge(cpu_ge2sym(want_coo));

    conv.ge2sym(tri, conv_stream());
    lla::LBCOOMatrix<Float, 1, 1> bcoo;
    conv.convert(tri, bcoo, conv_stream());
    lla::LBSRMatrix<Float, 1> bsr;
    conv.convert(bcoo, bsr, conv_stream());
    conv_stream().synchronize();

    auto       got_bsr = read_bsr<1>(bsr);
    const auto want    = cpu_bcoo_to_bsr(cpu_ge2sym(want_coo));
    CHECK(got_bsr.row_offsets == want.row_offsets);
    CHECK(got_bsr.col == want.col);

    const auto x      = make_vector(static_cast<std::size_t>(n), 0x2222u);
    const auto y_want = host_dense_spmv(general, x);

    lud::LDeviceBuffer<double> x_dev, y_dev;
    x_dev.copy_from(std::span<const double>{x});
    y_dev.resize(static_cast<std::size_t>(n));
    y_dev.fill(0.0);
    conv_stream().synchronize();
    conv_stream().record(BsrSymSpMVKernel<1>::get()(bsr.row_offsets(), bsr.col_indices(),
                                                    bsr.values(), x_dev.view(), y_dev.view(),
                                                    static_cast<std::uint32_t>(n))
                             .dispatch(static_cast<std::uint32_t>(n)));
    conv_stream().synchronize();
    std::vector<double> y_got(static_cast<std::size_t>(n), 0.0);
    y_dev.view().copy_to(std::span<double>{y_got});
    lud::main_stream().synchronize();
    double worst = 0.0;
    for(std::size_t i = 0; i < y_got.size(); ++i)
        worst = std::max(worst, std::abs(y_got[i] - y_want[i]) / std::max(1.0, std::abs(y_want[i])));
    CHECK(worst < 1e-5);
    g_stats.worst_spmv = std::max(g_stats.worst_spmv, worst);
    std::printf("    [e2e block1] symmetric SpMV over the converted BSR, worst %.2e\n", worst);
}

// ===========================================================================
//  6. determinism (R6)
// ===========================================================================
TEST_CASE("converter_determinism", "[luisa][converter]")
{
    test_runtime();
    const int  n   = 300;
    const auto nnz = static_cast<std::size_t>(0.05 * n * n);

    // the same converter object, the same input, twice: bitwise identical —
    // including the "convert twice into the *same* container" case, where the
    // output buffer still holds the previous result (the accumulation primitives
    // must overwrite their slots, not add to stale data)
    {
        lla::LMatrixConverter<Float, 3> conv;
        lla::LTripletMatrix<Float, 3, 3> tri;
        auto tris = make_triplets<3, 3>(nnz, n, n, 0x3333u, false);
        tri.reshape(n, n);
        tri.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 3, 3>>{tris});

        lla::LBCOOMatrix<Float, 3, 3> a, b;
        conv.convert(tri, a, conv_stream());
        conv_stream().synchronize();
        conv.convert(tri, b, conv_stream());
        conv_stream().synchronize();
        auto got_a = read_bcoo(a);
        auto got_b = read_bcoo(b);
        CHECK(got_a.row == got_b.row);
        CHECK(got_a.col == got_b.col);
        CHECK(same_doubles(got_a.values, got_b.values));

        // into the same container
        conv.convert(tri, a, conv_stream());
        conv_stream().synchronize();
        auto got_a2 = read_bcoo(a);
        CHECK(got_a2.row == got_a.row);
        CHECK(got_a2.col == got_a.col);
        CHECK(same_doubles(got_a2.values, got_a.values));

        // BCOO -> BSR twice, into the same container
        lla::LBSRMatrix<Float, 3> s1, s2;
        conv.convert(a, s1, conv_stream());
        conv.convert(a, s2, conv_stream());
        conv_stream().synchronize();
        auto got_s1 = read_bsr<3>(s1);
        auto got_s2 = read_bsr<3>(s2);
        CHECK(got_s1.row_offsets == got_s2.row_offsets);
        CHECK(got_s1.col == got_s2.col);
        CHECK(same_doubles(got_s1.values, got_s2.values));
        conv.convert(a, s1, conv_stream());
        conv_stream().synchronize();
        auto got_s3 = read_bsr<3>(s1);
        CHECK(same_doubles(got_s3.values, got_s1.values));
        CHECK(got_s3.row_offsets == got_s1.row_offsets);

        // ge2sym and sym2ge twice
        lla::LBCOOMatrix<Float, 3, 3> g1, g2;
        conv.sym2ge(a, g1, conv_stream());
        conv_stream().synchronize();
        conv.sym2ge(a, g2, conv_stream());
        conv_stream().synchronize();
        auto got_g1 = read_bcoo(g1);
        auto got_g2 = read_bcoo(g2);
        CHECK(got_g1.row == got_g2.row);
        CHECK(got_g1.col == got_g2.col);
        CHECK(same_doubles(got_g1.values, got_g2.values));
        conv.sym2ge(a, g1, conv_stream());
        conv_stream().synchronize();
        auto got_g3 = read_bcoo(g1);
        CHECK(got_g3.row == got_g1.row);
        CHECK(same_doubles(got_g3.values, got_g1.values));
    }

    // the doublet path, twice into the same container
    {
        lla::LMatrixConverter<Float, 3> conv;
        auto ds = make_doublets<3>(20'000u, 500, 0x4444u, false);
        lla::LDoubletVector<Float, 3> d;
        d.reshape(500);
        d.copy_from_doublets(std::span<const lla::VectorDoublet<Float, 3>>{ds});
        lla::LBCOOVector<Float, 3> v;
        conv.convert(d, v, conv_stream());
        conv_stream().synchronize();
        auto got1 = read_bcoo_vector<3>(v);
        conv.convert(d, v, conv_stream());
        conv_stream().synchronize();
        auto got2 = read_bcoo_vector<3>(v);
        CHECK(got1.index == got2.index);
        CHECK(same_doubles(got1.values, got2.values));
    }
    std::printf("    [determinism] triplet/bsr/ge2sym/sym2ge/doublet twice: bitwise identical\n");
}

// ===========================================================================
//  7. the loose_resize / reserve_ratio growth path
// ===========================================================================
TEST_CASE("converter_repeated_growth", "[luisa][converter]")
{
    test_runtime();
    // one converter object, growing (and shrinking) inputs: every conversion must
    // still be exact, i.e. the scratch growth never corrupts or drops data
    lla::LMatrixConverter<Float, 3> conv;
    double                          worst = 0.0;
    std::size_t                     total = 0;
    for(const std::size_t nnz : {std::size_t{1}, std::size_t{3}, std::size_t{17}, std::size_t{100},
                                 std::size_t{1000}, std::size_t{5000}, std::size_t{20000},
                                 std::size_t{3}, std::size_t{700}, std::size_t{100000},
                                 std::size_t{50}})
    {
        const int   n    = 512;
        auto        tris = make_triplets<3, 3>(nnz, n, n, 0x5555u + nnz, false);
        lla::LTripletMatrix<Float, 3, 3> tri;
        tri.reshape(n, n);
        tri.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 3, 3>>{tris});

        lla::LBCOOMatrix<Float, 3, 3> bcoo;
        conv.convert(tri, bcoo, conv_stream());
        conv_stream().synchronize();
        auto got  = read_bcoo(bcoo);
        auto want = cpu_triplets_to_bcoo<3, 3>(tris, n, n);
        CHECK(got.row == want.row);
        CHECK(got.col == want.col);
        const double dev = worst_diff(got.values, want.values);
        worst            = std::max(worst, dev);
        CHECK(dev < 1e-6);

        lla::LBSRMatrix<Float, 3> bsr;
        conv.convert(bcoo, bsr, conv_stream());
        conv_stream().synchronize();
        auto got_bsr  = read_bsr<3>(bsr);
        auto want_bsr = cpu_bcoo_to_bsr(want);
        CHECK(got_bsr.row_offsets == want_bsr.row_offsets);
        CHECK(dev < 1e-6);

        // the doublet path on the same object (a third scratch set to grow)
        auto ds = make_doublets<3>(nnz / 2u + 1u, 300, 0x6666u + nnz, false);
        lla::LDoubletVector<Float, 3> d;
        d.reshape(300);
        d.copy_from_doublets(std::span<const lla::VectorDoublet<Float, 3>>{ds});
        lla::LBCOOVector<Float, 3> vec;
        conv.convert(d, vec, conv_stream());
        conv_stream().synchronize();
        auto got_vec  = read_bcoo_vector<3>(vec);
        auto want_vec = cpu_doublets_to_bcoo_vector<3>(ds, 300);
        CHECK(got_vec.index == want_vec.index);
        CHECK(worst_diff(got_vec.values, want_vec.values) < 1e-6);
        total += nnz;
    }
    g_stats.worst_bcoo = std::max(g_stats.worst_bcoo, worst);
    std::printf("    [growth] 11 conversions on one converter, 1..100000 triplets (%zu total), "
                "worst %.2e\n",
                total, worst);
}

// ===========================================================================
//  8. small / degenerate cases
// ===========================================================================
TEST_CASE("converter_small_cases", "[luisa][converter]")
{
    test_runtime();
    for(const std::size_t nnz : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}})
    {
        {
            auto tris = make_triplets<1, 1>(nnz, 4, 4, 0x7777u + nnz, false);
            lla::LMatrixConverter<Float, 1> conv;
            lla::LBCOOMatrix<Float, 1>      bcoo;
            auto got  = convert_triplets_to_bcoo<1, 1>(conv, tris, 4, 4, bcoo);
            auto want = cpu_triplets_to_bcoo<1, 1>(tris, 4, 4);
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            CHECK(same_doubles(got.values, want.values));
            lla::LBSRMatrix<Float, 1> bsr;
            conv.convert(bcoo, bsr, conv_stream());
            conv_stream().synchronize();
            auto got_bsr  = read_bsr<1>(bsr);
            auto want_bsr = cpu_bcoo_to_bsr(want);
            CHECK(got_bsr.row_offsets == want_bsr.row_offsets);
            CHECK(got_bsr.col == want_bsr.col);
            CHECK(same_doubles(got_bsr.values, want_bsr.values));
        }
        {
            auto tris = make_triplets<3, 3>(nnz, 4, 4, 0x8888u + nnz, false);
            lla::LMatrixConverter<Float, 3> conv;
            lla::LBCOOMatrix<Float, 3, 3>   bcoo;
            auto got  = convert_triplets_to_bcoo<3, 3>(conv, tris, 4, 4, bcoo);
            auto want = cpu_triplets_to_bcoo<3, 3>(tris, 4, 4);
            CHECK(got.row == want.row);
            CHECK(got.col == want.col);
            CHECK(same_doubles(got.values, want.values));
        }
    }

    // 0 rows / 0 entries
    {
        lla::LMatrixConverter<Float, 1> conv;
        lla::LBCOOMatrix<Float, 1>      bcoo;
        bcoo.reshape(0, 0);
        bcoo.resize_triplets_discard(0u);
        lla::LBSRMatrix<Float, 1> bsr;
        conv.convert(bcoo, bsr, conv_stream());
        conv_stream().synchronize();
        CHECK(bsr.rows() == 0);
        CHECK(bsr.non_zeros() == 0);
        auto got = read_bsr<1>(bsr);
        CHECK(got.row_offsets.size() == 1u);
        CHECK(got.row_offsets[0] == 0);
    }

    // every triplet on the same (i, j) and a single row with many columns
    {
        std::vector<lla::MatrixTriplet<Float, 1>> tris(50);
        for(std::size_t i = 0; i < tris.size(); ++i)
        {
            tris[i].row_index = 0;
            tris[i].col_index = 0;
            tris[i].value     = 1.0;
        }
        lla::LMatrixConverter<Float, 1> conv;
        lla::LBCOOMatrix<Float, 1>      bcoo;
        auto got = convert_triplets_to_bcoo<1, 1>(conv, tris, 2, 2, bcoo);
        CHECK(got.nnz() == 1u);
        CHECK(got.values.size() == 1u);
        CHECK(got.values[0] == 50.0);
    }
    std::printf("    [small cases] 0/1/2/3 entries, 0 rows, 50 identical entries: ok\n");
}

// ===========================================================================
//  9. the reference's own call shape (two-argument overloads) + reserve_ratio
// ===========================================================================
TEST_CASE("converter_reference_call_shape", "[luisa][converter]")
{
    test_runtime();
    // The reference's methods take (from, to) only — the stream is implicit.  The
    // port keeps that spelling (it dispatches on a process-lifetime LStream over
    // the main stream) next to the explicit-stream overloads the backend uses.
    const int  n   = 60;
    const auto nnz = static_cast<std::size_t>(0.2 * n * n);

    lla::LMatrixConverter<Float, 3> conv;
    CHECK(conv.reserve_ratio() == 1.5);  // the reference's default
    conv.reserve_ratio(1.1);             // a tighter growth policy
    CHECK(conv.reserve_ratio() == 1.1);

    auto                             tris = make_triplets<3, 3>(nnz, n, n, 0x2b2bu, false);
    lla::LTripletMatrix<Float, 3, 3> tri;
    tri.reshape(n, n);
    tri.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 3, 3>>{tris});

    auto               want = cpu_triplets_to_bcoo<3, 3>(tris, n, n);
    lla::LBCOOMatrix<Float, 3, 3> bcoo;
    conv.ge2sym(tri);          // two-argument spelling
    conv.convert(tri, bcoo);   // two-argument spelling
    lud::main_stream().synchronize();
    auto got  = read_bcoo(bcoo);
    auto upper = cpu_ge2sym(want);
    CHECK(got.row == upper.row);
    CHECK(got.col == upper.col);
    CHECK(same_doubles(got.values, upper.values));

    lla::LBSRMatrix<Float, 3> bsr;
    conv.convert(bcoo, bsr);
    lud::main_stream().synchronize();
    auto   got_bsr  = read_bsr<3>(bsr);
    HostBsr want_bsr = cpu_bcoo_to_bsr(upper);
    CHECK(got_bsr.row_offsets == want_bsr.row_offsets);
    CHECK(got_bsr.col == want_bsr.col);
    CHECK(same_doubles(got_bsr.values, want_bsr.values));

    // the doublet two-argument overload (with the scalar-segment converter, the
    // only instantiation whose payload sort goes through sort_pairs_by_key)
    lla::LMatrixConverter<Float, 1> conv1;
    auto ds = make_doublets<1>(500u, 40, 0x2b2cu, false);
    lla::LDoubletVector<Float, 1> d;
    d.reshape(40);
    d.copy_from_doublets(std::span<const lla::VectorDoublet<Float, 1>>{ds});
    lla::LBCOOVector<Float, 1> vec;
    conv1.convert(d, vec);
    lud::main_stream().synchronize();
    auto      got_vec  = read_bcoo_vector<1>(vec);
    HostCooVec want_vec = cpu_doublets_to_bcoo_vector<1>(ds, 40);
    CHECK(got_vec.index == want_vec.index);
    CHECK(same_doubles(got_vec.values, want_vec.values));
    std::printf("    [reference call shape] (from, to) overloads + reserve_ratio(1.1) growth: ok\n");
}

// ===========================================================================
//  9. T3.2 regression found by T3.3: segmented_reduce_blocks' identity fill
// ===========================================================================
namespace
{
/// A cross-warp segment: its head lane is lane 0 of a warp w > 0, so the head
/// warp accumulates with `atomic_add` (the reference's is_cross_warp path) and
/// therefore *relies on the out buffer having been identity-filled*.
std::vector<std::int32_t> cross_warp_keys(std::size_t n)
{
    std::vector<std::int32_t> keys(n, 0);
    for(std::size_t i = 0; i < n; ++i)
    {
        // runs of 32 (a run starts at every warp boundary), then a long run
        keys[i] = static_cast<std::int32_t>(i / 40u);
    }
    return keys;
}
}// namespace

TEST_CASE("converter_segmented_reduce_blocks_out_fill", "[luisa][converter]")
{
    test_runtime();
    // The converter's duplicate accumulation calls `segmented_reduce_blocks` with
    // `components = N*N` and an `out` that is reused across conversions.  T3.2's
    // primitive filled only `out_size` scalars (instead of `out_size*components`),
    // so every slot a cross-warp segment accumulates into was left holding the
    // previous call's value.  This case is the regression: a fresh `out` prefilled
    // with a sentinel must come out exactly as the CPU model says.
    constexpr std::uint32_t components = 9;
    const std::size_t       n          = 120;
    const std::size_t       out_size   = 16;
    const auto              keys       = cross_warp_keys(n);

    std::vector<double> values(n * components);
    for(std::size_t i = 0; i < n; ++i)
        for(std::uint32_t c = 0; c < components; ++c)
            values[i * components + c] = static_cast<double>((i % 5) + c + 1);

    // the CPU model: out[key * components + c] = sum of values over the key's run
    std::vector<double> want(out_size * components, 0.0);
    for(std::size_t i = 0; i < n; ++i)
    {
        const auto k = static_cast<std::size_t>(keys[i]);
        for(std::uint32_t c = 0; c < components; ++c)
            want[k * components + c] += values[i * components + c];
    }

    lud::LDeviceBuffer<std::int32_t> keys_dev;
    lud::LDeviceBuffer<double>       values_dev;
    keys_dev.copy_from(std::span<const std::int32_t>{keys});
    values_dev.copy_from(std::span<const double>{values});

    for(const double sentinel : {0.0, 1.0, -1000.0})
    {
        lud::LDeviceBuffer<double> out_dev;
        out_dev.resize(out_size * components);
        out_dev.fill(sentinel);  // a stale value in every slot
        conv_stream().synchronize();
        lla::segmented_reduce_blocks<std::int32_t, double>(conv_stream(), keys_dev.cview(),
                                                           values_dev.cview(), out_dev.view(),
                                                           out_size, n, components);
        conv_stream().synchronize();
        std::vector<double> got(out_size * components, 0.0);
        out_dev.view().copy_to(std::span<double>{got});
        lud::main_stream().synchronize();
        CHECK(same_doubles(got, want));
    }

    // ... and the same buffer reused for a second call must be overwritten, not
    // accumulated onto
    lud::LDeviceBuffer<double> out_dev;
    out_dev.resize(out_size * components);
    out_dev.fill(7.5);
    conv_stream().synchronize();
    lla::segmented_reduce_blocks<std::int32_t, double>(conv_stream(), keys_dev.cview(),
                                                       values_dev.cview(), out_dev.view(), out_size,
                                                       n, components);
    conv_stream().synchronize();
    lla::segmented_reduce_blocks<std::int32_t, double>(conv_stream(), keys_dev.cview(),
                                                       values_dev.cview(), out_dev.view(), out_size,
                                                       n, components);
    conv_stream().synchronize();
    std::vector<double> got2(out_size * components, 0.0);
    out_dev.view().copy_to(std::span<double>{got2});
    lud::main_stream().synchronize();
    CHECK(same_doubles(got2, want));
    std::printf("    [segmented_reduce_blocks fill] 3 sentinels + reuse: exact\n");
}
