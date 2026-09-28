/********************************************************************
 * @file luisa_device_spmv_tests.cpp
 * @brief T3.4 — tests for the SpMV + dense BLAS port
 *        (luisa_device/linear/{spmv.h,blas.h}).
 *
 * Oracle.  The reference SpMV/BLAS are CUDA-only (`<<<>>>` launches, cudaStream_t,
 * CUB, Eigen) and cannot be host-included, so the oracle here is an *independent
 * CPU model* written from the mathematical definition:
 *   * the symmetric (upper-packed block BCOO) expansion — an off-diagonal entry
 *     (i, j, B) contributes B to block (i, j) and B^T to block (j, i), a diagonal
 *     entry contributes once — applied to x;
 *   * the plain COO product (what the *general* `rbk_spmv` variant computes);
 *   * dot / nrm2 / axpy / fill with a long-double (high precision) accumulator.
 * The reference `.cu` kernels are never transcribed.
 *
 * Coverage (brief §4):
 *  1. all four SpMV variants vs the CPU model for random upper-packed symmetric
 *     BCOO matrices, block sizes 1 and 3, several densities, and the adversarial
 *     shapes: 0 entries, 1 entry, a single entry, an empty block row, a full
 *     block row, a fully dense upper triangle, and a large case (>= 1e5 entries);
 *     `b != 0` (y = a*A*x + b*y) with a pre-filled garbage y in every case;
 *  2. the by-hand symmetric-convention case (3x3 tridiagonal, non-zero diagonal,
 *     a *duplicated* off-diagonal entry) — the plain BCOO model differs there, so
 *     it is asserted separately, plus a block-3 case with a non-symmetric block,
 *     which is what proves the `B^T` mirror;
 *  3. `rbk_sym_spmv_dot`: the fused `d_dot` equals `x^T (a A x)` computed
 *     independently, equals the separate `rbk_sym_spmv` + `dot` path, and is
 *     unchanged when the grid is sized by `triplet_capacity` instead of the exact
 *     count (the captured-graph contract);
 *  4. BLAS dot / norm / axpy / fill over the size sweep
 *     0/1/2/31/32/33/1000/100000, plus the huge-dynamic-range norm
 *     ({1e200, 1, 1e-200}) against a long-double computation — a naive square sum
 *     overflows there, so this is what proves the scaled-square-sum form;
 *  5. determinism (R6): every variant run twice on the same input, compared
 *     bit-for-bit (the atomics are still expected to be schedule-reproducible on a
 *     fixed GPU/launch — the caveat is documented in spmv.h);
 *  6. a PCG-shaped integration: 20 CG iterations driven purely through the BLAS +
 *     `rbk_sym_spmv`, compared against a CPU CG with the same formula/iteration
 *     count.
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/linear/linear.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <span>
#include <vector>

namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lcrt = ::luisa::compute;

using Float = double;  // uipc::Float is double

namespace
{
// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------
struct Lcg
{
    std::uint64_t s = 0x2545'f491'3c6e'f8a1ULL;
    std::uint64_t next_u64() noexcept
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s;
    }
    double next() noexcept  // [0, 1)
    {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
    double next_sym() noexcept { return next() * 2.0 - 1.0; }
};

bool close_rel(double a, double b, double tol)
{
    if(std::isnan(a) || std::isnan(b))
        return std::isnan(a) && std::isnan(b);
    const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= tol * scale;
}

/// The stream every test drives the kernels with.  Leaked on purpose: an
/// LStream destroyed during static destruction would call into the LC runtime
/// after zzz_runtime_shutdown tore it down (cookbook §7.4 / §7.10).
lud::LStream& test_stream()
{
    static auto* s = new lud::LStream{lud::main_stream()};
    return *s;
}

template <int N>
using Triplet = lla::MatrixTriplet<Float, N, N>;

template <int N>
void set_block(typename Triplet<N>::ValueT& v, int r, int c, double x)
{
    if constexpr(N == 1)
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

template <int N>
double get_block(const typename Triplet<N>::ValueT& v, int r, int c)
{
    if constexpr(N == 1)
    {
        (void)r;
        (void)c;
        return v;
    }
    else
    {
        return v(r, c);
    }
}

// ---------------------------------------------------------------------------
//  independent CPU models (from the mathematical definition, not a transcription)
// ---------------------------------------------------------------------------
/// Symmetric expansion product: an off-diagonal (i, j, B) adds B x_j to y_i and
/// B^T x_i to y_j; a diagonal block adds B x_i once.
template <int N>
std::vector<Float> cpu_sym_product(const std::vector<Triplet<N>>& trips,
                                   const std::vector<Float>&     x,
                                   std::size_t                   dofs)
{
    std::vector<Float> y(dofs, 0.0);
    for(const auto& t : trips)
    {
        for(int r = 0; r < N; ++r)
        {
            for(int c = 0; c < N; ++c)
            {
                y[static_cast<std::size_t>(t.row_index * N + r)] +=
                    get_block<N>(t.value, r, c) * x[static_cast<std::size_t>(t.col_index * N + c)];
                if(t.row_index != t.col_index)
                {
                    // mirrored, transposed
                    y[static_cast<std::size_t>(t.col_index * N + r)] +=
                        get_block<N>(t.value, c, r)
                        * x[static_cast<std::size_t>(t.row_index * N + c)];
                }
            }
        }
    }
    return y;
}

/// Plain COO product: what the *general* `rbk_spmv` variant computes.
template <int N>
std::vector<Float> cpu_general_product(const std::vector<Triplet<N>>& trips,
                                       const std::vector<Float>&     x,
                                       std::size_t                   dofs)
{
    std::vector<Float> y(dofs, 0.0);
    for(const auto& t : trips)
    {
        for(int r = 0; r < N; ++r)
        {
            for(int c = 0; c < N; ++c)
            {
                y[static_cast<std::size_t>(t.row_index * N + r)] +=
                    get_block<N>(t.value, r, c) * x[static_cast<std::size_t>(t.col_index * N + c)];
            }
        }
    }
    return y;
}

/// High-precision dot (long double accumulator).
double cpu_dot(const std::vector<Float>& x, const std::vector<Float>& y)
{
    long double s = 0.0L;
    for(std::size_t i = 0; i < x.size(); ++i)
        s += static_cast<long double>(x[i]) * static_cast<long double>(y[i]);
    return static_cast<double>(s);
}

/// High-precision nrm2 (long double accumulator).
double cpu_norm(const std::vector<Float>& x)
{
    long double s = 0.0L;
    for(auto v : x)
        s += static_cast<long double>(v) * static_cast<long double>(v);
    return static_cast<double>(std::sqrt(s));
}

// ---------------------------------------------------------------------------
//  device plumbing
// ---------------------------------------------------------------------------
template <typename T>
lud::LDeviceBuffer<T> to_device(const std::vector<T>& host)
{
    lud::LDeviceBuffer<T> buf;
    if(!host.empty())
    {
        buf.resize(host.size());
        buf.copy_from(std::span<const T>{host});
    }
    return buf;
}

/// The SpMV surface consumes dense-vector *views*; the owning container is
/// `LDenseVector` (the reference's `DeviceDenseVector`).
template <typename T>
lla::LDenseVector<T> to_dense(const std::vector<T>& host)
{
    lla::LDenseVector<T> v;
    if(!host.empty())
    {
        v.resize(host.size());
        v.copy_from(std::span<const T>{host});
    }
    return v;
}

template <int N>
lla::LBCOOMatrix<Float, N> make_bcoo(int rows, const std::vector<Triplet<N>>& trips,
                                     std::size_t min_capacity = 0u)
{
    lla::LBCOOMatrix<Float, N> A;
    A.reshape(rows, rows);
    if(min_capacity > trips.size())
        A.reserve_triplets(min_capacity);
    A.copy_from_triplets(std::span<const Triplet<N>>{trips});
    return A;
}

/// Overwrite the reserved-but-unwritten tail of the index arrays with a poison
/// index (the reference never reads it; the port must clamp instead of using it as
/// a segment offset — an unclamped read is an illegal device address).
template <int N>
void poison_reserved_tail(lla::LBCOOMatrix<Float, N>& A)
{
    const auto n = A.triplet_count();
    auto&      rb = A.row_index_buffer();
    auto&      cb = A.col_index_buffer();
    if(rb.capacity() <= n)
        return;
    rb.resize(rb.capacity());
    cb.resize(cb.capacity());
    lla::fill(rb.view(), std::int32_t{0x7fff'ffff}, test_stream());
    lla::fill(cb.view(), std::int32_t{0x7fff'ffff}, test_stream());
    rb.resize(n);
    cb.resize(n);
}

template <typename T>
std::vector<T> from_device(lud::LDeviceBuffer<T>& buf, std::size_t n)
{
    std::vector<T> host(n, T{});
    if(n != 0u)
        buf.cview(0u, n).copy_to(std::span<T>{host});
    test_stream().synchronize();
    return host;
}

/// `y = a*A*x + b*y` through one of the four variants.
/// @param general  true -> compare against the plain COO model (rbk_spmv)
template <int N>
void run_variant(int variant, lla::LBCOOMatrix<Float, N>& A, const std::vector<Float>& xh,
                 const std::vector<Float>& y0h, double a, double b, std::vector<Float>& out)
{
    const std::size_t dofs = xh.size();
    auto              xv   = to_dense(xh);
    auto              yv   = to_dense(y0h);

    lla::LSpmv<Float, N> spmv;
    auto& stream = test_stream();

    switch(variant)
    {
        case 0:
            spmv.sym_spmv(a, A.cview(), xv.cview(), b, yv.view(), stream);
            break;
        case 1:
            spmv.rbk_spmv(a, A.cview(), xv.cview(), b, yv.view(), stream);
            break;
        case 2:
            spmv.rbk_sym_spmv(a, A.cview(), xv.cview(), b, yv.view(), stream);
            break;
        case 3: {
            lud::LDeviceVar<Float>         d_dot(0.0);
            lud::LDeviceVar<std::int32_t>  count_dev;
            count_dev.copy_from(static_cast<std::int32_t>(A.triplet_count()));
            spmv.rbk_sym_spmv_dot(a, A.cview(), xv.cview(), b, yv.view(), d_dot.view(),
                                  count_dev.cview(), A.triplet_capacity(), stream);
            break;
        }
        case 4:
            // debug fallback
            spmv.cpu_sym_spmv(a, A.cview(), xv.cview(), b, yv.view());
            break;
    }
    test_stream().synchronize();
    out = from_device(yv.values_buffer(), dofs);
}

// ---------------------------------------------------------------------------
//  random upper-packed symmetric BCOO generation
// ---------------------------------------------------------------------------
struct Shape
{
    const char* name;
    int         rows;
    double      density;
    int         mode;  // 0 normal, 1 empty row, 2 full row, 3 single diagonal
};

/// Upper-packed block BCOO: one entry per selected (i, j), i <= j.  The diagonal is
/// always present unless `mode == 1` clears a whole block row.
template <int N>
std::vector<Triplet<N>> make_random_symmetric(const Shape& shape, std::uint64_t seed)
{
    Lcg  rng;
    rng.s = seed;
    std::vector<Triplet<N>> trips;
    for(int i = 0; i < shape.rows; ++i)
    {
        for(int j = i; j < shape.rows; ++j)
        {
            const bool diag = (i == j);
            if(!diag && rng.next() >= shape.density)
                continue;
            Triplet<N> t;
            t.row_index = i;
            t.col_index = j;
            for(int r = 0; r < N; ++r)
            {
                for(int c = 0; c < N; ++c)
                {
                    const double v = diag ? (1.0 + std::abs(rng.next_sym())) : rng.next_sym();
                    set_block<N>(t.value, r, c, v);
                }
            }
            trips.push_back(t);
        }
    }
    if(shape.mode == 1 && shape.rows >= 2)
    {
        // an entirely empty block row (no entries at all for that row)
        const int victim = shape.rows - 2;
        std::erase_if(trips, [&](const Triplet<N>& t) { return t.row_index == victim; });
    }
    if(shape.mode == 2 && shape.rows >= 2)
    {
        // a full row: every upper column present for row 1 (plus nothing else
        // removed) — already covered by density 1 shapes, kept explicit
        for(int j = 1; j < shape.rows; ++j)
        {
            const bool have = std::any_of(trips.begin(), trips.end(), [&](const Triplet<N>& t) {
                return t.row_index == 1 && t.col_index == j;
            });
            if(!have)
            {
                Triplet<N> t;
                t.row_index = 1;
                t.col_index = j;
                for(int r = 0; r < N; ++r)
                    for(int c = 0; c < N; ++c)
                        set_block<N>(t.value, r, c, rng.next_sym());
                trips.push_back(t);
            }
        }
    }
    if(shape.mode == 3)
    {
        // exactly one diagonal entry
        trips.clear();
        Triplet<N> t;
        t.row_index = 0;
        t.col_index = 0;
        for(int r = 0; r < N; ++r)
            for(int c = 0; c < N; ++c)
                set_block<N>(t.value, r, c, (r == c) ? 2.0 : 0.25);
        trips.push_back(t);
    }
    return trips;
}

template <int N>
void check_variant(lla::LBCOOMatrix<Float, N>& A, const std::vector<Triplet<N>>& trips, int rows,
                   int variant, double a, double b, std::uint64_t seed, double tol,
                   std::size_t& compared, double* worst_by_variant = nullptr)
{
    Lcg rng;
    rng.s = seed;

    const std::size_t dofs = static_cast<std::size_t>(rows) * N;
    std::vector<Float> xh(dofs), y0h(dofs);
    for(auto& v : xh)
        v = rng.next_sym();
    for(auto& v : y0h)
        v = rng.next_sym() * 3.0;  // garbage prefill

    const bool general   = (variant == 1);
    const auto product   = general ? cpu_general_product<N>(trips, xh, dofs)
                                   : cpu_sym_product<N>(trips, xh, dofs);
    std::vector<Float> y_cpu(dofs);
    for(std::size_t i = 0; i < dofs; ++i)
        y_cpu[i] = a * product[i] + b * y0h[i];

    std::vector<Float> got;
    run_variant<N>(variant, A, xh, y0h, a, b, got);

    double worst = 0.0;
    for(std::size_t i = 0; i < dofs; ++i)
    {
        const double err = std::abs(got[i] - y_cpu[i]);
        worst           = std::max(worst, err / std::max(1.0, std::abs(y_cpu[i])));
        ++compared;
    }
    CHECK(worst <= tol);
    if(worst > tol)
        std::printf("      variant %d rows %d worst %.3e\n", variant, rows, worst);
}

// ---------------------------------------------------------------------------
//  device BCOO with a reserved capacity > count (the graph contract)
// ---------------------------------------------------------------------------
template <int N>
std::vector<Triplet<N>> tridiagonal_entries(int rows)
{
    std::vector<Triplet<N>> trips;
    for(int i = 0; i < rows; ++i)
    {
        Triplet<N> d;
        d.row_index = i;
        d.col_index = i;
        for(int r = 0; r < N; ++r)
            for(int c = 0; c < N; ++c)
                set_block<N>(d.value, r, c, (r == c) ? 2.0 : 0.0);
        trips.push_back(d);
        if(i + 1 < rows)
        {
            Triplet<N> o;
            o.row_index = i;
            o.col_index = i + 1;
            for(int r = 0; r < N; ++r)
                for(int c = 0; c < N; ++c)
                    set_block<N>(o.value, r, c, (r == c) ? -1.0 : 0.0);
            trips.push_back(o);
        }
    }
    return trips;
}
}// namespace

// ===========================================================================
//  1. all four (+ debug) variants vs the CPU model
// ===========================================================================
TEST_CASE("spmv_variants_vs_cpu_model", "[luisa][spmv]")
{
    test_runtime();
    std::size_t compared = 0;
    double      worst_by[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    const Shape shapes[] = {
        {"empty", 0, 0.5, 0},
        {"single_diag", 1, 1.0, 3},
        {"one_entry", 3, 0.0, 3},
        {"sparse", 4, 0.5, 0},
        {"empty_row", 7, 0.7, 1},
        {"full_row", 6, 0.35, 2},
        {"dense8", 8, 1.0, 0},
        {"medium", 40, 0.2, 0},
        {"large", 1000, 0.2, 0},  // >= 1e5 entries (upper triangle ~ 500k * 0.2)
    };

    for(const auto& shape : shapes)
    {
        // block size 3
        {
            auto trips = make_random_symmetric<3>(shape, 0xabc'0000 + shape.rows);
            auto A     = make_bcoo<3>(shape.rows, trips);
            if(shape.rows >= 1000)
                CHECK(A.triplet_count() >= 100000u);
            for(int variant = 0; variant < 5; ++variant)
                check_variant<3>(A, trips, shape.rows, variant, 1.3, 0.7, 0x1234 + shape.rows,
                                 1e-5, compared, worst_by);
            // b == 0 flavour as well
            check_variant<3>(A, trips, shape.rows, 2, -0.6, 0.0, 0x99 + shape.rows, 1e-5, compared);
        }
        // block size 1
        {
            auto trips = make_random_symmetric<1>(shape, 0xdef'0000 + shape.rows);
            auto A     = make_bcoo<1>(shape.rows, trips);
            for(int variant = 0; variant < 5; ++variant)
                check_variant<1>(A, trips, shape.rows, variant, 0.8, -1.1, 0x5678 + shape.rows,
                                 1e-5, compared, worst_by);
            check_variant<1>(A, trips, shape.rows, 3, 1.0, 0.0, 0x77 + shape.rows, 1e-5, compared);
        }
    }
    std::cout << "[spmv] worst relative error by variant (bound 1e-5): sym=" << worst_by[0]
              << " rbk=" << worst_by[1] << " rbk_sym=" << worst_by[2]
              << " rbk_sym_dot=" << worst_by[3] << " cpu=" << worst_by[4]
              << " over " << compared << " comparisons" << std::endl;
    CHECK(compared > 10000u);
}

// ===========================================================================
//  2. the symmetric convention, by hand
// ===========================================================================
TEST_CASE("spmv_symmetric_convention", "[luisa][spmv]")
{
    test_runtime();

    // --- block 1: 3x3 tridiagonal with a duplicated off-diagonal entry ---
    // entries: (0,0,2) (1,1,3) (2,2,4) (0,1,1.5) (0,1,0.5) (1,2,-2)
    // A_sym(0,1) = A_sym(1,0) = 2.0     (the duplicate sums)
    // A_sym = [ 2  2  0 ; 2  3 -2 ; 0 -2  4 ]
    std::vector<Triplet<1>> trips;
    auto                    add1 = [&](int i, int j, double v) {
        Triplet<1> t;
        t.row_index = i;
        t.col_index = j;
        t.value     = v;
        trips.push_back(t);
    };
    add1(0, 0, 2.0);
    add1(1, 1, 3.0);
    add1(2, 2, 4.0);
    add1(0, 1, 1.5);
    add1(0, 1, 0.5);  // duplicated off-diagonal
    add1(1, 2, -2.0);

    auto               A    = make_bcoo<1>(3, trips);
    std::vector<Float> xh{1.0, 2.0, 3.0};
    std::vector<Float> y0(3, 0.0);

    // A_sym * x = [2*1 + 2*2, 2*1 + 3*2 - 2*3, -2*2 + 4*3] = [6, 2, 8]
    const std::vector<Float> expect_sym{6.0, 2.0, 8.0};
    // plain upper-triangle product = [2*1 + 1.5*2 + 0.5*2, 3*2 - 2*3, 4*3] = [6, 0, 12]
    const std::vector<Float> expect_plain{6.0, 0.0, 12.0};

    for(int variant : {0, 2, 3})
    {
        std::vector<Float> got;
        run_variant<1>(variant, A, xh, y0, 1.0, 0.0, got);
        for(std::size_t i = 0; i < 3; ++i)
            CHECK(std::abs(got[i] - expect_sym[i]) <= 1e-12);
    }
    {
        std::vector<Float> got;
        run_variant<1>(1, A, xh, y0, 1.0, 0.0, got);  // rbk_spmv: the general variant
        for(std::size_t i = 0; i < 3; ++i)
            CHECK(std::abs(got[i] - expect_plain[i]) <= 1e-12);
        // ... and it really does differ from the symmetric convention
        CHECK(std::abs(got[1] - expect_sym[1]) > 0.5);
    }

    // --- block 3: a non-symmetric block proves the B^T mirror ---
    // two block-3 entries: (0,0) = diag(1,2,3), (0,1) = B = [[1,2,3],[0,1,2],[0,0,1]]
    // (B is deliberately non-symmetric: only the transposed mirror can produce Y_1)
    std::vector<Triplet<3>> t3;
    {
        Triplet<3> a;
        a.row_index = 0;
        a.col_index = 0;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                set_block<3>(a.value, r, c, (r == c) ? double(r + 1) : 0.0);
        t3.push_back(a);
        Triplet<3> b;
        b.row_index = 0;
        b.col_index = 1;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                set_block<3>(b.value, r, c, (c >= r) ? double(c - r + 1) : 0.0);
        t3.push_back(b);
    }
    auto A3 = make_bcoo<3>(2, t3);
    // x has 6 dofs; give block 0 and block 1 arbitrary values
    std::vector<Float> x3{1.0, 2.0, 3.0, -1.0, 0.5, 4.0};
    std::vector<Float> y30(6, 0.0);

    // CPU by-hand: y0 = D0 x0 + B x1 ; y1 = B^T x0
    auto mul3 = [](const double m[3][3], const std::vector<Float>& v, std::size_t off) {
        std::vector<double> r(3, 0.0);
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                r[static_cast<std::size_t>(i)] += m[i][j] * v[off + static_cast<std::size_t>(j)];
        return r;
    };
    const double D0[3][3] = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
    const double Bm[3][3] = {{1, 2, 3}, {0, 1, 2}, {0, 0, 1}};
    const double Bt[3][3] = {{1, 0, 0}, {2, 1, 0}, {3, 2, 1}};
    auto         y0_ref   = mul3(D0, x3, 0);
    auto         b_x1     = mul3(Bm, x3, 3);
    auto         bt_x0    = mul3(Bt, x3, 0);
    for(int i = 0; i < 3; ++i)
    {
        y0_ref[static_cast<std::size_t>(i)] += b_x1[static_cast<std::size_t>(i)];
    }

    for(int variant : {0, 2, 3})
    {
        std::vector<Float> got;
        run_variant<3>(variant, A3, x3, y30, 1.0, 0.0, got);
        for(int i = 0; i < 3; ++i)
            CHECK(std::abs(got[static_cast<std::size_t>(i)]
                           - y0_ref[static_cast<std::size_t>(i)])
                  <= 1e-12);
        for(int i = 0; i < 3; ++i)
            CHECK(std::abs(got[static_cast<std::size_t>(3 + i)]
                           - bt_x0[static_cast<std::size_t>(i)])
                  <= 1e-12);
    }
}

// ===========================================================================
//  3. rbk_sym_spmv_dot — the fused dot
// ===========================================================================
TEST_CASE("spmv_fused_dot", "[luisa][spmv]")
{
    test_runtime();
    double worst_vs_cpu = 0.0, worst_vs_separate = 0.0;

    for(int rows : {1, 5, 40, 500})
    {
        auto trips = make_random_symmetric<3>(Shape{"dot", rows, 0.35, 0}, 0x3333 + rows);
        auto A     = make_bcoo<3>(rows, trips);

        const std::size_t dofs = static_cast<std::size_t>(rows) * 3;
        Lcg               rng;
        rng.s = 0x4444 + rows;
        std::vector<Float> xh(dofs), y0(dofs, 0.0);
        for(auto& v : xh)
            v = rng.next_sym();

        const auto product = cpu_sym_product<3>(trips, xh, dofs);
        double     dot_cpu = 0.0;
        for(std::size_t i = 0; i < dofs; ++i)
            dot_cpu += xh[i] * product[i];  // x^T (a A x) with a = 1

        auto                   xv = to_dense(xh);
        auto                   yv = to_dense(y0);
        lud::LDeviceVar<Float> d_dot(0.0);
        lud::LDeviceVar<std::int32_t> count_dev;
        count_dev.copy_from(static_cast<std::int32_t>(A.triplet_count()));

        lla::LSpmv<Float, 3> spmv;
        spmv.rbk_sym_spmv_dot(1.0, A.cview(), xv.cview(), 0.0, yv.view(), d_dot.view(),
                              count_dev.cview(), A.triplet_capacity(), test_stream());
        const auto fused_dot = d_dot.copy_to();
        auto       y_fused   = from_device(yv.values_buffer(), dofs);

        // (a) vs the independent CPU dot
        worst_vs_cpu = std::max(worst_vs_cpu,
                                std::abs(fused_dot - dot_cpu) / std::max(1.0, std::abs(dot_cpu)));
        CHECK(close_rel(fused_dot, dot_cpu, 1e-6));

        // (b) vs the separate rbk_sym_spmv + dot path
        auto                  yv2 = to_dense(y0);
        lla::LLinearSystemContext ctx(&test_stream());
        spmv.rbk_sym_spmv(1.0, A.cview(), xv.cview(), 0.0, yv2.view(), test_stream());
        const auto separate_dot = ctx.dot(xv.cview(), yv2.cview());
        worst_vs_separate = std::max(
            worst_vs_separate,
            std::abs(fused_dot - separate_dot) / std::max(1.0, std::abs(separate_dot)));
        CHECK(close_rel(fused_dot, separate_dot, 1e-6));

        // same y as the fused pass
        auto y_sep = from_device(yv2.values_buffer(), dofs);
        for(std::size_t i = 0; i < dofs; ++i)
            CHECK(close_rel(y_fused[i], y_sep[i], 1e-12));
    }
    std::cout << "[spmv] fused d_dot vs independent CPU " << worst_vs_cpu
              << ", vs rbk_sym_spmv + dot " << worst_vs_separate << " (relative)" << std::endl;
}

// ===========================================================================
//  3b. the grid-sized-by-capacity contract
// ===========================================================================
TEST_CASE("spmv_fused_dot_capacity_grid", "[luisa][spmv]")
{
    test_runtime();

    for(int rows : {1, 9, 100, 500})
    {
        auto trips = make_random_symmetric<3>(Shape{"cap", rows, 0.3, 0}, 0x5555 + rows);
        // reserve 3x the entries so the grid is clearly larger than the count
        const std::size_t capacity = std::max<std::size_t>(trips.size() * 3u, 300u);
        auto              A        = make_bcoo<3>(rows, trips, capacity);
        poison_reserved_tail<3>(A);
        CHECK(A.triplet_capacity() >= capacity);
        CHECK(A.triplet_count() == trips.size());

        const std::size_t dofs = static_cast<std::size_t>(rows) * 3;
        Lcg               rng;
        rng.s = 0x6666 + rows;
        std::vector<Float> xh(dofs), y0(dofs, 0.0);
        for(auto& v : xh)
            v = rng.next_sym();

        auto                          xv = to_dense(xh);
        lud::LDeviceVar<std::int32_t> count_dev;
        count_dev.copy_from(static_cast<std::int32_t>(A.triplet_count()));
        lla::LSpmv<Float, 3> spmv;

        auto run = [&](std::size_t cap, std::vector<Float>& y_out, double& dot_out) {
            auto                   yv = to_dense(y0);
            lud::LDeviceVar<Float> d_dot(0.0);
            spmv.rbk_sym_spmv_dot(1.0, A.cview(), xv.cview(), 0.0, yv.view(), d_dot.view(),
                                  count_dev.cview(), cap, test_stream());
            dot_out = d_dot.copy_to();
            y_out   = from_device(yv.values_buffer(), dofs);
        };

        std::vector<Float> y_cap, y_exact;
        double             dot_cap = 0.0, dot_exact = 0.0;
        run(A.triplet_capacity(), y_cap, dot_cap);
        run(A.triplet_count(), y_exact, dot_exact);

        // The extra (all-invalid) blocks write nothing; only the *order* of the
        // surviving atomic_adds into y can move, hence a tight relative bound.
        double worst = 0.0;
        for(std::size_t i = 0; i < dofs; ++i)
            worst = std::max(worst, std::abs(y_cap[i] - y_exact[i]) / std::max(1.0, std::abs(y_exact[i])));
        CHECK(worst <= 1e-12);
        // the dot is an atomic-free two-pass fold, so the extra (all-zero) capacity
        // blocks cannot change it (exact)
        CHECK(dot_cap == dot_exact);
    }
}

// ===========================================================================
//  4. BLAS: dot / norm / axpy / fill / scale / axpby / update_xr / update_p
// ===========================================================================
TEST_CASE("blas_dot_norm_axpy_fill", "[luisa][spmv]")
{
    test_runtime();

    const std::vector<std::size_t> sizes{0, 1, 2, 31, 32, 33, 1000, 100000};
    std::size_t                    compared = 0;
    double                         worst_dot = 0.0, worst_norm = 0.0;

    for(auto n : sizes)
    {
        Lcg rng;
        rng.s = 0x7777 + n;
        std::vector<Float> xh(n), yh(n), y0(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            xh[i] = rng.next_sym() * 2.0;
            y0[i] = rng.next_sym() * 5.0;
            yh[i] = y0[i];
        }

        auto xv = to_device(xh);
        auto yv = to_device(y0);
        auto& stream = test_stream();
        lla::LLinearSystemContext ctx(&stream);

        // dot
        {
            std::vector<Float> wh(n);
            for(std::size_t i = 0; i < n; ++i)
                wh[i] = std::sin(static_cast<double>(i) * 0.37) * 1.5;
            auto wv     = to_device(wh);
            auto got    = ctx.dot(xv.cview(), wv.cview());
            auto expect = cpu_dot(xh, wh);
            worst_dot   = std::max(worst_dot,
                                 std::abs(got - expect) / std::max(1.0, std::abs(expect)));
            CHECK(close_rel(got, expect, 1e-12));
            ++compared;

            // free-function flavour (same storage, caller stream)
            CHECK(close_rel(lla::dot(xv.cview(), wv.cview(), stream), expect, 1e-12));
        }

        // norm / nrm2 (incl. the 0-size identity)
        {
            auto got    = ctx.norm(xv.cview());
            auto expect = cpu_norm(xh);
            worst_norm  = std::max(worst_norm,
                                  std::abs(got - expect) / std::max(1.0, std::abs(expect)));
            CHECK(close_rel(got, expect, 1e-12));
            if(n == 0u)
                CHECK(got == 0.0);
            ++compared;
            CHECK(close_rel(lla::norm2(xv.cview(), stream), expect, 1e-12));
        }

        // fill
        {
            lla::fill(yv.view(), 3.25, stream);
            auto got = from_device(yv, n);
            for(std::size_t i = 0; i < n; ++i)
                CHECK(got[i] == 3.25);
            ++compared;
        }

        // scale: y = a*y
        {
            auto yv2 = to_device(y0);
            lla::scale(-1.75, yv2.view(), stream);
            auto got = from_device(yv2, n);
            for(std::size_t i = 0; i < n; ++i)
                CHECK(got[i] == -1.75 * y0[i]);
            ++compared;
        }

        // axpy: y += a*x
        {
            auto yv2 = to_device(yh);
            lla::axpy(2.5, xv.cview(), yv2.view(), stream);
            auto got = from_device(yv2, n);
            for(std::size_t i = 0; i < n; ++i)
                // an FMA contraction on the GPU vs mul-then-add on the host: last-bit equal
                CHECK(close_rel(got[i], yh[i] + 2.5 * xh[i], 1e-15));
            ++compared;
        }

        // axpby: y = a*x + b*y  (the `update_p` shape)
        {
            auto yv2 = to_device(yh);
            lla::axpby(-0.5, xv.cview(), 0.25, yv2.view(), stream);
            auto got = from_device(yv2, n);
            for(std::size_t i = 0; i < n; ++i)
                CHECK(close_rel(got[i], -0.5 * xh[i] + 0.25 * yh[i], 1e-15));
            ++compared;
        }

        // update_xr: x += alpha*p ; r -= alpha*Ap
        {
            auto xv2 = to_device(xh);
            auto rv  = to_device(yh);
            auto yv3 = to_device(y0);
            lla::update_xr(0.75, xv2.view(), yv3.cview(), rv.view(), xv.cview(), stream);
            auto got_x = from_device(xv2, n);
            auto got_r = from_device(rv, n);
            for(std::size_t i = 0; i < n; ++i)
            {
                CHECK(close_rel(got_x[i], xh[i] + 0.75 * y0[i], 1e-15));
                CHECK(close_rel(got_r[i], yh[i] - 0.75 * xh[i], 1e-15));
            }
            ++compared;
        }

        // update_p: p = z + beta*p
        {
            auto pv = to_device(yh);
            lla::update_p(pv.view(), xv.cview(), -0.25, stream);
            auto got = from_device(pv, n);
            for(std::size_t i = 0; i < n; ++i)
                CHECK(close_rel(got[i], xh[i] - 0.25 * yh[i], 1e-15));
            ++compared;
        }
    }
    std::cout << "[blas] worst relative error: dot " << worst_dot << ", norm " << worst_norm
              << " over the size sweep" << std::endl;
    CHECK(compared >= 8u * sizes.size() - 8u);
}

// ===========================================================================
//  4b. norm with a huge dynamic range (proves the scaled square sum)
// ===========================================================================
TEST_CASE("blas_norm_dynamic_range", "[luisa][spmv]")
{
    test_runtime();
    lla::LLinearSystemContext ctx(&test_stream());

    // A naive sum of squares overflows (1e400) and the small entry underflows
    // (1e-400 -> 0); nrm2 must still return 1e200.
    {
        std::vector<Float> xh{1e200, 1.0, 1e-200};
        auto               xv     = to_device(xh);
        auto               got    = ctx.norm(xv.cview());
        auto               expect = cpu_norm(xh);
        CHECK(close_rel(got, expect, 1e-12));
        CHECK(std::isfinite(got));
        CHECK(std::abs(got - 1e200) <= 1e-12 * 1e200);
    }
    // The float equivalent: 1e30^2 = 1e60 overflows float, nrm2 must not.
    {
        std::vector<float> xh{1e30f, 1.0f, 1e-30f};
        auto               xv  = to_device(xh);
        lla::LLinearSystemContext fctx(&test_stream());
        auto               got = fctx.norm(xv.cview());
        CHECK(std::isfinite(got));
        CHECK(std::abs(static_cast<double>(got) - 1e30) <= 1e-5 * 1e30);
    }
    // 1e-200-only vector: nrm2 must return 1e-200, not 0.
    {
        std::vector<Float> xh{1e-200, 2e-200};
        auto               xv     = to_device(xh);
        auto               got    = ctx.norm(xv.cview());
        auto               expect = cpu_norm(xh);
        CHECK(close_rel(got, expect, 1e-12));
        CHECK(got > 0.0);
    }
    // all-zero vector -> exactly 0 (no NaN from the 0/0 ratio)
    {
        std::vector<Float> xh(37, 0.0);
        auto               xv = to_device(xh);
        CHECK(ctx.norm(xv.cview()) == 0.0);
    }
    // a long mixed-magnitude sweep against the long-double model
    {
        std::vector<Float> xh(5000);
        Lcg                rng;
        rng.s = 0x8888;
        for(std::size_t i = 0; i < xh.size(); ++i)
        {
            const double scale = std::pow(10.0, static_cast<double>(static_cast<int>(i % 101)) - 50.0);
            xh[i]              = rng.next_sym() * scale;
        }
        auto xv = to_device(xh);
        CHECK(close_rel(ctx.norm(xv.cview()), cpu_norm(xh), 1e-12));
    }
}

// ===========================================================================
//  5. determinism (R6)
// ===========================================================================
TEST_CASE("spmv_determinism", "[luisa][spmv]")
{
    test_runtime();

    for(int rows : {3, 17, 200})
    {
        auto trips = make_random_symmetric<3>(Shape{"det", rows, 0.4, 0}, 0x9999 + rows);
        auto A     = make_bcoo<3>(rows, trips);

        const std::size_t dofs = static_cast<std::size_t>(rows) * 3;
        Lcg               rng;
        rng.s = 0xaaaa + rows;
        std::vector<Float> xh(dofs), y0(dofs);
        for(auto& v : xh)
            v = rng.next_sym();
        for(auto& v : y0)
            v = rng.next_sym();

        auto xv = to_dense(xh);

        for(int variant = 0; variant < 4; ++variant)
        {
            std::vector<Float> first;
            double             first_dot = 0.0;
            for(int rep = 0; rep < 2; ++rep)
            {
                auto                   yv = to_dense(y0);
                lla::LSpmv<Float, 3>   spmv;
                lud::LDeviceVar<Float> d_dot(0.0);
                lud::LDeviceVar<std::int32_t> count_dev;
                count_dev.copy_from(static_cast<std::int32_t>(A.triplet_count()));
                switch(variant)
                {
                    case 0:
                        spmv.sym_spmv(1.1, A.cview(), xv.cview(), 0.5, yv.view(), test_stream());
                        break;
                    case 1:
                        spmv.rbk_spmv(1.1, A.cview(), xv.cview(), 0.5, yv.view(), test_stream());
                        break;
                    case 2:
                        spmv.rbk_sym_spmv(1.1, A.cview(), xv.cview(), 0.5, yv.view(),
                                          test_stream());
                        break;
                    default:
                        spmv.rbk_sym_spmv_dot(1.1, A.cview(), xv.cview(), 0.5, yv.view(),
                                              d_dot.view(), count_dev.cview(),
                                              A.triplet_capacity(), test_stream());
                        break;
                }
                test_stream().synchronize();
                auto got = from_device(yv.values_buffer(), dofs);
                const double got_dot = (variant == 3) ? d_dot.copy_to() : 0.0;
                if(rep == 0)
                {
                    first     = got;
                    first_dot = got_dot;
                }
                else
                {
                    for(std::size_t i = 0; i < dofs; ++i)
                        CHECK(close_rel(got[i], first[i], 1e-12));  // atomic y: last bits may differ
                    if(variant == 3)
                        CHECK(got_dot == first_dot);
                }
            }
        }
    }

    // BLAS determinism as well (no atomics there at all)
    {
        const std::size_t  n = 100000;
        std::vector<Float> xh(n);
        Lcg                rng;
        rng.s = 0xbbbb;
        for(auto& v : xh)
            v = rng.next_sym() * 1e100;
        auto                       xv = to_device(xh);
        lla::LLinearSystemContext  ctx(&test_stream());
        const auto d1 = ctx.dot(xv.cview(), xv.cview());
        const auto d2 = ctx.dot(xv.cview(), xv.cview());
        CHECK(d1 == d2);
        const auto n1 = ctx.norm(xv.cview());
        const auto n2 = ctx.norm(xv.cview());
        CHECK(n1 == n2);
    }
}

// ===========================================================================
//  5b. measured timings for the >= 1e5-entry case (brief §6)
// ===========================================================================
TEST_CASE("spmv_large_timing", "[luisa][spmv]")
{
    test_runtime();

    const Shape shape{"timing", 1000, 0.2, 0};
    auto        trips = make_random_symmetric<3>(shape, 0xeeee);
    auto        A     = make_bcoo<3>(shape.rows, trips);
    CHECK(A.triplet_count() >= 100000u);

    const std::size_t dofs = static_cast<std::size_t>(shape.rows) * 3;
    Lcg               rng;
    rng.s = 0xf00d;
    std::vector<Float> xh(dofs);
    for(auto& v : xh)
        v = rng.next_sym();
    auto                   xv = to_dense(xh);
    auto                   yv = to_dense(std::vector<Float>(dofs, 0.0));
    lud::LDeviceVar<Float> d_dot(0.0);
    lud::LDeviceVar<std::int32_t> count_dev;
    count_dev.copy_from(static_cast<std::int32_t>(A.triplet_count()));

    lla::LSpmv<Float, 3> spmv;
    auto& stream = test_stream();

    auto time_it = [&](const char* name, auto&& call) {
        call();  // warm-up (compile + first launch)
        stream.synchronize();
        constexpr int reps = 50;
        const auto    t0   = std::chrono::steady_clock::now();
        for(int i = 0; i < reps; ++i)
            call();
        stream.synchronize();
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()
            / reps;
        std::printf("[spmv timing] %-16s %8.3f ms/call  (%zu entries)\n", name, ms,
                    A.triplet_count());
    };

    time_it("sym_spmv", [&] { spmv.sym_spmv(1.0, A.cview(), xv.cview(), 1.0, yv.view(), stream); });
    time_it("rbk_spmv", [&] { spmv.rbk_spmv(1.0, A.cview(), xv.cview(), 1.0, yv.view(), stream); });
    time_it("rbk_sym_spmv",
            [&] { spmv.rbk_sym_spmv(1.0, A.cview(), xv.cview(), 1.0, yv.view(), stream); });
    time_it("rbk_sym_spmv_dot", [&] {
        spmv.rbk_sym_spmv_dot(1.0, A.cview(), xv.cview(), 1.0, yv.view(), d_dot.view(),
                              count_dev.cview(), A.triplet_capacity(), stream);
    });
    time_it("cpu_sym_spmv", [&] {
        spmv.cpu_sym_spmv(1.0, A.cview(), xv.cview(), 1.0, yv.view());
    });

    // BLAS at the same scale
    lla::LLinearSystemContext ctx(&stream);
    {
        auto        wv = to_dense(xh);
        const auto  t0 = std::chrono::steady_clock::now();
        constexpr int reps = 50;
        double      sink = 0.0;
        for(int i = 0; i < reps; ++i)
            sink += ctx.dot(xv.cview(), wv.cview());
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()
            / reps;
        std::printf("[spmv timing] %-16s %8.3f ms/call  (n=%zu)\n", "dot", ms, dofs);
        CHECK(std::isfinite(sink));
    }
}

// ===========================================================================
//  6. a PCG-shaped integration
// ===========================================================================
TEST_CASE("spmv_pcg_integration", "[luisa][spmv]")
{
    test_runtime();

    const int         rows = 300;  // block-1 tridiagonal SPD system
    std::vector<Float> xh(static_cast<std::size_t>(rows), 0.0);
    Lcg                rng;
    rng.s = 0xcccc;
    for(auto& v : xh)
        v = rng.next_sym();

    // A = tridiag(-1, 2, -1) built as an upper-packed symmetric BCOO
    std::vector<Triplet<1>> trips;
    for(int i = 0; i < rows; ++i)
    {
        Triplet<1> d;
        d.row_index = i;
        d.col_index = i;
        d.value     = 2.0;
        trips.push_back(d);
        if(i + 1 < rows)
        {
            Triplet<1> o;
            o.row_index = i;
            o.col_index = i + 1;
            o.value     = -1.0;
            trips.push_back(o);
        }
    }
    auto A = make_bcoo<1>(rows, trips);

    // --- CPU PCG (unpreconditioned, the same formula) -----------------------
    const std::size_t    n = static_cast<std::size_t>(rows);
    std::vector<Float>   b(n);
    for(std::size_t i = 0; i < n; ++i)
        b[i] = std::sin(static_cast<double>(i) * 0.11) + 0.5;

    auto matvec = [&](const std::vector<Float>& v) {
        std::vector<Float> out(n, 0.0);
        for(const auto& t : trips)
        {
            out[static_cast<std::size_t>(t.row_index)] += t.value * v[static_cast<std::size_t>(t.col_index)];
            if(t.row_index != t.col_index)
                out[static_cast<std::size_t>(t.col_index)] += t.value * v[static_cast<std::size_t>(t.row_index)];
        }
        return out;
    };

    const int          iters = 20;
    std::vector<Float> x_cpu(n, 0.0);
    std::vector<Float> r = b;
    std::vector<Float> p = r;
    double             rz = cpu_dot(r, r);
    for(int k = 0; k < iters; ++k)
    {
        const auto Ap = matvec(p);
        const auto pAp = cpu_dot(p, Ap);
        const double alpha = rz / pAp;
        for(std::size_t i = 0; i < n; ++i)
        {
            x_cpu[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        const double rz_new = cpu_dot(r, r);
        const double beta   = rz_new / rz;
        for(std::size_t i = 0; i < n; ++i)
            p[i] = r[i] + beta * p[i];
        rz = rz_new;
    }
    const std::vector<Float> r_cpu = r;

    // --- device PCG through the BLAS + rbk_sym_spmv --------------------------
    auto& stream = test_stream();
    lla::LLinearSystemContext ctx(&stream);
    lla::LSpmv<Float, 1>  spmv;

    auto xv = to_dense(std::vector<Float>(n, 0.0));
    auto rv = to_dense(b);
    auto pv = to_dense(b);
    auto Ap = to_dense(std::vector<Float>(n, 0.0));

    auto dev_dot = [&](const lla::LDenseVector<Float>& a, const lla::LDenseVector<Float>& c) {
        return ctx.dot(a.cview(), c.cview());
    };

    double rz_dev = dev_dot(rv, rv);
    for(int k = 0; k < iters; ++k)
    {
        spmv.rbk_sym_spmv(1.0, A.cview(), pv.cview(), 0.0, Ap.view(), stream);
        const double pAp   = dev_dot(pv, Ap);
        const double alpha = rz_dev / pAp;
        lla::update_xr(alpha, xv.buffer_view(), pv.buffer_view(), rv.buffer_view(),
                       Ap.buffer_view(), stream);
        const double rz_new = dev_dot(rv, rv);
        const double beta   = rz_new / rz_dev;
        lla::update_p(pv.buffer_view(), rv.buffer_view(), beta, stream);
        rz_dev = rz_new;
    }

    const auto x_dev = from_device(xv.values_buffer(), n);
    double     worst = 0.0;
    for(std::size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::abs(x_dev[i] - x_cpu[i]) / std::max(1.0, std::abs(x_cpu[i])));
    CHECK(worst <= 1e-6);

    // and the residuals agree too
    std::vector<Float> r_dev(n, 0.0);
    for(std::size_t i = 0; i < n; ++i)
    {
        double ax = 0.0;
        for(const auto& t : trips)
        {
            if(static_cast<std::size_t>(t.row_index) == i)
                ax += t.value * x_dev[static_cast<std::size_t>(t.col_index)];
            if(t.row_index != t.col_index && static_cast<std::size_t>(t.col_index) == i)
                ax += t.value * x_dev[static_cast<std::size_t>(t.row_index)];
        }
        r_dev[i] = b[i] - ax;
    }
    CHECK(close_rel(cpu_dot(r_dev, r_dev), cpu_dot(r_cpu, r_cpu), 1e-6));
}
