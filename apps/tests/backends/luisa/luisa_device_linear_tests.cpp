/********************************************************************
 * @file luisa_device_linear_tests.cpp
 * @brief T3.1 — tests for the linear-algebra containers and views
 *        (luisa_device/linear/{views,containers,linear}.h).
 *
 * Coverage (brief §6):
 *  1. round-trip host -> container -> device kernel -> container -> host, for
 *     every container type, with block/segment sizes 1 and 3, compared against a
 *     host reference model that the device never produced;
 *  2. view semantics: mutating through a view mutates the container, a CView has
 *     no write path, subview/segment offsets agree with a host computation,
 *     resize/clear leave a well-defined state and the generation guard fires;
 *  3. block layout: a block written as an LMat on the device reads back
 *     column-major and equals the host's LMat/Eigen block;
 *  4. a real device kernel per container type (SpMV scatter/gather over triplets,
 *     a BSR row walk, dense vector/matrix sweeps) — not just a copy;
 *  5. sizes 0, 1, small and >= 100k entries.
 *
 * Structure mirrors luisa_device_matrix_tests.cpp: host semantics first, then
 * device smoke tests driven by one generic host model.
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/linear/linear.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <type_traits>
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
        return lo + static_cast<int>(next_u64() >> 33) % (hi - lo + 1);
    }
    double next() noexcept
    {
        return static_cast<double>((next_u64() >> 11) & 0xfffffULL) / 524288.0 - 1.0;
    }
};

bool close(double a, double b, double tol)
{
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

/// Host triplets for a `dofs x dofs` block matrix, `nnz` entries.
template <int M, int N>
std::vector<lla::MatrixTriplet<Float, M, N>> make_triplets(std::size_t nnz, int dofs, std::uint64_t seed)
{
    Lcg rng;
    rng.s = seed;
    std::vector<lla::MatrixTriplet<Float, M, N>> out(nnz);
    const int blocks = dofs / (M > N ? M : N);
    for(std::size_t i = 0; i < nnz; ++i)
    {
        auto& t       = out[i];
        t.row_index   = rng.next_int(0, blocks - 1);
        t.col_index   = rng.next_int(0, blocks - 1);
        if constexpr(M == 1 && N == 1)
        {
            t.value = rng.next();
        }
        else
        {
            for(int c = 0; c < N; ++c)
                for(int r = 0; r < M; ++r) t.value(r, c) = rng.next();
        }
    }
    return out;
}

/// y = A x, computed on the host (the reference model for the device SpMV).
template <int M, int N>
std::vector<double> host_spmv(const std::vector<lla::MatrixTriplet<Float, M, N>>& tris,
                              const std::vector<double>&                          x,
                              int                                                 dofs)
{
    std::vector<double> y(static_cast<std::size_t>(dofs), 0.0);
    for(const auto& t : tris)
    {
        if constexpr(M == 1 && N == 1)
        {
            y[static_cast<std::size_t>(t.row_index)] += t.value * x[static_cast<std::size_t>(t.col_index)];
        }
        else
        {
            for(int r = 0; r < M; ++r)
            {
                double acc = 0.0;
                for(int c = 0; c < N; ++c)
                {
                    acc += t.value(r, c) * x[static_cast<std::size_t>(t.col_index * N + c)];
                }
                y[static_cast<std::size_t>(t.row_index * M + r)] += acc;
            }
        }
    }
    return y;
}

/// Deterministic device input vector.
std::vector<double> make_dense(int n, std::uint64_t seed)
{
    Lcg                 rng;
    rng.s = seed;
    std::vector<double> out(static_cast<std::size_t>(n));
    for(int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = rng.next();
    return out;
}

// ---------------------------------------------------------------------------
//  device kernels (one per container type / accessor group)
// ---------------------------------------------------------------------------
/// Triplet matrix: read every triplet through the view, scale the block and write
/// it back (exercises the Proxy read/write path for both flavours).
template <int M, int N>
struct TripletScaleKernel
{
    static auto& get()
    {
        using WView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, M, N>;
        static auto k = lud::compile1<lcrt::Buffer<std::int32_t>,
                                     lcrt::Buffer<std::int32_t>,
                                     lcrt::Buffer<double>,
                                     uint32_t>(
            [](lcrt::BufferVar<std::int32_t> rows,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       vals,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto A = WView::from_handles(rows, cols, vals, n, n, n);
                auto i = lcrt::dispatch_x();
                $if (i < n)
                {
                    auto t = A(i).read();
                    if constexpr(M == 1 && N == 1)
                    {
                        lcrt::Var<double> b = t.value * 2.0 - 1.0;
                        A(i).write(t.row_index, t.col_index, b);
                    }
                    else
                    {
                        lps::LMat<lcrt::Var<double>, M, N> b{};
                        for(int r = 0; r < M; ++r)
                        {
                            for(int c = 0; c < N; ++c)
                            {
                                b(r, c) = t.value(r, c) * 2.0 - 1.0;
                            }
                        }
                        A(i).write(t.row_index, t.col_index, b);
                    }
                };
            });
        return k;
    }
};

/// Triplet matrix SpMV: y = A x, a scatter/gather over the triplets with atomic
/// accumulation into y (the accessor pattern the PCG path uses).
template <int M, int N>
struct TripletSpMVKernel
{
    static auto& get()
    {
        using CView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, M, N>;
        using CDense = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
        using Dense  = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<std::int32_t>,
                                     lcrt::Buffer<std::int32_t>,
                                     lcrt::Buffer<double>,
                                     lcrt::Buffer<double>,
                                     lcrt::Buffer<double>,
                                     uint32_t,
                                     uint32_t>(
            [](lcrt::BufferVar<std::int32_t> rows,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       vals,
               lcrt::BufferVar<double>       x,
               lcrt::BufferVar<double>       y,
               lcrt::Var<uint32_t>           n,
               lcrt::Var<uint32_t>           dofs) noexcept
            {
                auto A  = CView::from_handles(rows, cols, vals, dofs, dofs, n);
                auto xd = CDense::from_handles(x, dofs);
                auto yd = Dense::from_handles(y, dofs);
                auto i  = lcrt::dispatch_x();
                $if (i < n)
                {
                    auto t = A(i);  // CTriplet: row_index, col_index, value
                    if constexpr(M == 1 && N == 1)
                    {
                        auto xv = xd.read(t.col_index);
                        (void)yd.atomic_add(t.row_index, t.value * xv);
                    }
                    else
                    {
                        // block row `r` covers dofs [r*M, r*M+M), block col `c`
                        // covers [c*N, c*N+N)
                        auto xs = xd.segment<N>(t.col_index * static_cast<uint32_t>(N));
                        auto ys = yd.segment<M>(t.row_index * static_cast<uint32_t>(M));
                        lps::LVec<lcrt::Var<double>, M> acc{};
                        for(int rr = 0; rr < M; ++rr)
                        {
                            lcrt::Var<double> sum = 0.0;
                            for(int cc = 0; cc < N; ++cc)
                            {
                                sum = sum + t.value(rr, cc) * xs.read(cc);
                            }
                            acc(rr) = sum;
                        }
                        ys.atomic_add(acc);
                    }
                };
            });
        return k;
    }
};

/// Doublet vector: read a doublet and write it back with the value scaled.
template <int N>
struct DoubletScaleKernel
{
    static auto& get()
    {
        using WView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, N>;
        static auto k = lud::compile1<lcrt::Buffer<std::int32_t>,
                                     lcrt::Buffer<double>,
                                     uint32_t>(
            [](lcrt::BufferVar<std::int32_t> idx,
               lcrt::BufferVar<double>       vals,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto A = WView::from_handles(idx, vals, n, n);
                auto i = lcrt::dispatch_x();
                $if (i < n)
                {
                    auto d = A(i).read();
                    if constexpr(N == 1)
                    {
                        lcrt::Var<double> b = d.value * 3.0 + 1.0;
                        A(i).write(d.index, b);
                    }
                    else
                    {
                        lps::LVec<lcrt::Var<double>, N> b{};
                        for(int k = 0; k < N; ++k) b(k) = d.value(k) * 3.0 + 1.0;
                        A(i).write(d.index, b);
                    }
                };
            });
        return k;
    }
};

/// Doublet vector gather: y[segment] += sum(value), the doublet analogue of a
/// scatter (atomic, one thread per doublet).
template <int N>
struct DoubletGatherKernel
{
    static auto& get()
    {
        using CView = lla::LCDoubletVectorDeviceView<lcrt::Var<double>, N>;
        using Dense = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<std::int32_t>,
                                     lcrt::Buffer<double>,
                                     lcrt::Buffer<double>,
                                     uint32_t>(
            [](lcrt::BufferVar<std::int32_t> idx,
               lcrt::BufferVar<double>       vals,
               lcrt::BufferVar<double>       y,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto A  = CView::from_handles(idx, vals, n, n);
                auto yd = Dense::from_handles(y, n);
                auto i  = lcrt::dispatch_x();
                $if (i < n)
                {
                    auto d = A(i);
                    if constexpr(N == 1)
                    {
                        (void)yd.atomic_add(d.index, d.value);
                    }
                    else
                    {
                        lcrt::Var<double> sum = 0.0;
                        for(int k = 0; k < N; ++k) sum = sum + d.value(k);
                        (void)yd.atomic_add(d.index, sum);
                    }
                };
            });
        return k;
    }
};

/// BSR row walk: row_sum[r] = sum of every element of every block in row r.
template <int N>
struct BSRRowWalkKernel
{
    static auto& get()
    {
        using CView = lla::LCBSRMatrixDeviceView<lcrt::Var<double>, N>;
        using Dense = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<std::int32_t>,
                                     lcrt::Buffer<std::int32_t>,
                                     lcrt::Buffer<double>,
                                     lcrt::Buffer<double>,
                                     uint32_t>(
            [](lcrt::BufferVar<std::int32_t> offsets,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       vals,
               lcrt::BufferVar<double>       row_sum,
               lcrt::Var<uint32_t>           rows) noexcept
            {
                auto A  = CView::from_handles(offsets, cols, vals, rows, rows, 0u);
                auto yd = Dense::from_handles(row_sum, rows);
                auto r  = lcrt::dispatch_x();
                $if (r < rows)
                {
                    lcrt::Var<double> acc = 0.0;
                    // the counter must be a mutable DSL local: an Expr is
                    // single-assignment
                    lcrt::Var<std::int32_t> k   = A.row_begin(r);
                    auto                    end = A.row_end(r);
                    $while (k < end)
                    {
                        auto block = A.load_block(k);
                        if constexpr(N == 1)
                        {
                            acc = acc + block;
                        }
                        else
                        {
                            for(int rr = 0; rr < N; ++rr)
                            {
                                for(int cc = 0; cc < N; ++cc)
                                {
                                    acc = acc + block(rr, cc);
                                }
                            }
                        }
                        k = k + 1;
                    };
                    yd.write(r, acc);
                };
            });
        return k;
    }
};

/// Dense vector: y = x * 2 + 1 through operator() read/write.
struct DenseVectorSweepKernel
{
    static auto& get()
    {
        using CDense = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
        using Dense  = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<double>, lcrt::Buffer<double>, uint32_t>(
            [](lcrt::BufferVar<double> x, lcrt::BufferVar<double> y, lcrt::Var<uint32_t> n) noexcept
            {
                auto xd = CDense::from_handles(x, n);
                auto yd = Dense::from_handles(y, n);
                auto i  = lcrt::dispatch_x();
                $if (i < n)
                {
                    yd(i) = xd(i) * 2.0 + 1.0;
                };
            });
        return k;
    }
};

/// Dense vector: fixed-size segment arithmetic (`segment<N>` + `as_eigen()` +
/// `to_vec<N>()` + `atomic_add`), the Eigen-map pattern of the reference.
template <int SegN>
struct DenseVectorSegmentKernel
{
    static auto& get()
    {
        using CDense = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
        using Dense  = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<double>, lcrt::Buffer<double>, uint32_t>(
            [](lcrt::BufferVar<double> x, lcrt::BufferVar<double> y, lcrt::Var<uint32_t> nseg) noexcept
            {
                auto xd = CDense::from_handles(x, nseg * static_cast<uint32_t>(SegN));
                auto yd = Dense::from_handles(y, nseg * static_cast<uint32_t>(SegN));
                auto i  = lcrt::dispatch_x();
                $if (i < nseg)
                {
                    auto seg_i = i * static_cast<uint32_t>(SegN);
                    // `LVec<T,SegN> v = seg.as_eigen();` — the Eigen-map read step
                    lps::LVec<lcrt::Var<double>, SegN> v = xd.segment<SegN>(seg_i).as_eigen();
                    // mutable map: += onto the destination segment
                    yd.segment<SegN>(seg_i).as_eigen() += v * 2.0;
                    // and an atomic add of the same vector
                    yd.segment<SegN>(seg_i).atomic_add(v);
                };
            });
        return k;
    }
};

/// Dense matrix: out(r, c) = in(r, c) * 3 - 1, row-major through (r, c).
struct DenseMatrixSweepKernel
{
    static auto& get()
    {
        using CDense = lla::LCDenseMatrixDeviceView<lcrt::Var<double>>;
        using Dense  = lla::LDenseMatrixDeviceView<lcrt::Var<double>>;
        static auto k = lud::compile1<lcrt::Buffer<double>, lcrt::Buffer<double>, uint32_t, uint32_t>(
            [](lcrt::BufferVar<double> in, lcrt::BufferVar<double> out, lcrt::Var<uint32_t> rows, lcrt::Var<uint32_t> cols) noexcept
            {
                auto md = CDense::from_handles(in, rows, cols);
                auto od = Dense::from_handles(out, rows, cols);
                auto i  = lcrt::dispatch_x();
                $if (i < rows * cols)
                {
                    auto r = i / cols;
                    auto c = i % cols;
                    auto v = md.read(r, c);
                    od.write(r, c, v * 3.0 - 1.0);
                };
            });
        return k;
    }
};

// ---------------------------------------------------------------------------
//  the const-view contract, checked at compile time (dependent context)
// ---------------------------------------------------------------------------
template <typename V>
struct cannot_write_triplet
{
    template <typename U>
    static constexpr bool check()
    {
        return !requires(U& A, lps::LMat<Float, 3, 3>& block) { A(0).write(0, 0, block); };
    }
    static constexpr bool value = check<V>();
};
template <typename V>
struct triplet_triplet_is_an_aggregate
{
    template <typename U>
    static constexpr bool check()
    {
        return requires(U& A) {
            A(0).row_index;
            A(0).col_index;
            A(0).value;
        };
    }
    static constexpr bool value = check<V>();
};
template <typename V>
struct cannot_write_dense
{
    template <typename U>
    static constexpr bool check()
    {
        return !requires(U& x) { x.write(0, 1.0); };
    }
    static constexpr bool value = check<V>();
};
template <typename V>
struct cannot_atomic_add_dense
{
    template <typename U>
    static constexpr bool check()
    {
        return !requires(U& x) { x.atomic_add(0, 1.0); };
    }
    static constexpr bool value = check<V>();
};
template <typename V>
struct cannot_assign_dense_element
{
    template <typename U>
    static constexpr bool check()
    {
        return !requires(U& x) { x(0) = 1.0; };
    }
    static constexpr bool value = check<V>();
};
}// namespace

// ---------------------------------------------------------------------------
//  1. view semantics on the host (metadata, generation guard, const contract)
// ---------------------------------------------------------------------------
TEST_CASE("linear_host_view_semantics", "[luisa][linear]")
{
    test_runtime();

    using Triplet = lla::LTripletMatrix<Float, 3>;
    Triplet m;
    m.reshape(8, 8);
    m.resize_triplets(5);
    CHECK(m.rows() == 8);
    CHECK(m.cols() == 8);
    CHECK(m.triplet_count() == 5u);
    CHECK(m.non_zeros() == 5);
    CHECK(m.triplet_capacity() >= 5u);
    CHECK(m.block_dim() == 3);
    CHECK(Triplet::block_dim() == 3);

    auto v = m.view();
    CHECK(v.total_rows() == 8);
    CHECK(v.total_cols() == 8);
    CHECK(v.triplet_count() == 5);
    CHECK(v.total_triplet_count() == 5);
    CHECK(v.tripet_index_offset() == 0);        // the reference's spelling
    CHECK(v.triplet_index_offset() == 0);       // the corrected one
    CHECK(v.extent().x == 8);
    CHECK(v.submatrix_offset().x == 0);
    CHECK(v.kernel_file() != nullptr);
    CHECK(v.kernel_line() == 0);

    // subview / submatrix offsets agree with a host computation
    auto s = v.subview(2, 3);
    CHECK(s.triplet_count() == 3);
    CHECK(s.triplet_index_offset() == 2);
    CHECK(s.total_triplet_count() == 5);
    auto s2 = v.subview(1);
    CHECK(s2.triplet_count() == 4);
    auto sub = v.submatrix({2, 4}, {3, 2});
    CHECK(sub.submatrix_offset().x == 2);
    CHECK(sub.submatrix_offset().y == 4);
    CHECK(sub.extent().x == 3);
    CHECK(sub.extent().y == 2);
    CHECK(sub.triplet_count() == 5);  // submatrix() does not change the triplet window

    // a const view is a different type that still carries the metadata
    auto cv = m.cview();
    CHECK(cv.triplet_count() == 5);
    CHECK(cv.as_const().triplet_count() == 5);
    CHECK(v.cviewer().triplet_count() == 5);
    CHECK(v.viewer().triplet_count() == 5);

    // out-of-range subview throws on the host
    CHECK_THROWS(v.subview(4, 3));

    // the generation guard (plan risk R5): a view must not survive a resize
    const auto gen_before = m.generation();
    v.use_check();  // same generation: fine
    m.resize_triplets(64);
    CHECK(m.generation() != gen_before);
    CHECK_THROWS(v.use_check());
    // ... and a fresh view is valid again
    auto v2 = m.view();
    v2.use_check();
    CHECK(v2.triplet_count() == 64);

    // clear() leaves a valid, well-defined state
    m.clear();
    CHECK(m.rows() == 0);
    CHECK(m.cols() == 0);
    CHECK(m.triplet_count() == 0u);
    auto vclear = m.view();
    CHECK(vclear.triplet_count() == 0);
    CHECK(vclear.total_rows() == 0);
    vclear.use_check();

    // the reference spellings resolve to the same types
    static_assert(std::is_same_v<lla::TripletMatrixView<Float, 3>, lla::LTripletMatrixView<Float, 3>>);
    static_assert(std::is_same_v<lla::BCOOMatrixView<Float, 3>, lla::LTripletMatrixView<Float, 3>>);
    static_assert(std::is_same_v<lla::CBCOOMatrixView<Float, 3>, lla::LCTripletMatrixView<Float, 3>>);
    static_assert(std::is_same_v<lla::DeviceTripletMatrix<Float, 3>, lla::LTripletMatrix<Float, 3>>);
    static_assert(std::is_same_v<lla::DeviceBCOOVector<Float, 3>, lla::LBCOOVector<Float, 3>>);

    // the CView contract (a C-view has no write path at all).  The checks run on
    // flavours with element access: a host `LBufferView` view refuses element
    // access entirely (that is also a compile-time guarantee, see views.h).
    static_assert(cannot_write_triplet<lla::LCTripletMatrixHostMemoryView<Float, 3>>::value);
    static_assert(cannot_write_triplet<lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3>>::value);
    static_assert(triplet_triplet_is_an_aggregate<lla::LCTripletMatrixHostMemoryView<Float, 3>>::value);
    static_assert(cannot_write_dense<lla::LCDenseVectorHostMemoryView<Float>>::value);
    static_assert(cannot_write_dense<lla::LCDenseVectorDeviceView<lcrt::Var<double>>>::value);
    static_assert(cannot_atomic_add_dense<lla::LCDenseVectorHostMemoryView<Float>>::value);
    static_assert(cannot_assign_dense_element<lla::LCDenseVectorHostMemoryView<Float>>::value);
    static_assert(cannot_assign_dense_element<lla::LCDenseVectorDeviceView<lcrt::Var<double>>>::value);
}

// ---------------------------------------------------------------------------
//  2. block layout: the flat storage is column-major per block
// ---------------------------------------------------------------------------
TEST_CASE("linear_block_layout", "[luisa][linear]")
{
    test_runtime();

    // host side: pack triplets, read the flat storage back, compare against the
    // host LMat (which is Eigen's column-major layout)
    {
        constexpr int              M = 3, N = 3;
        std::vector<lla::MatrixTriplet<Float, M, N>> tris(2);
        for(int i = 0; i < 2; ++i)
        {
            tris[static_cast<std::size_t>(i)].row_index = i;
            tris[static_cast<std::size_t>(i)].col_index = 1 - i;
            for(int r = 0; r < M; ++r)
            {
                for(int c = 0; c < N; ++c)
                {
                    tris[static_cast<std::size_t>(i)].value(r, c) = 10.0 * r + c + 100.0 * i;
                }
            }
        }
        lla::LTripletMatrix<Float, M, N> m;
        m.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, M, N>>{tris});

        std::vector<Float> flat(m.values_buffer().size(), 0.0);
        m.copy_to_values(std::span<Float>{flat});

        for(int i = 0; i < 2; ++i)
        {
            for(int c = 0; c < N; ++c)
            {
                for(int r = 0; r < M; ++r)
                {
                    // index(i, r, c) = i*M*N + c*M + r  (column-major per block)
                    const std::size_t idx = static_cast<std::size_t>(i) * M * N
                                            + static_cast<std::size_t>(c * M + r);
                    CHECK(flat[idx] == tris[static_cast<std::size_t>(i)].value(r, c));
                    CHECK(flat[idx]
                          == tris[static_cast<std::size_t>(i)].value.data[static_cast<std::size_t>(c * M + r)]);
                }
            }
        }

        // and the round trip back to triplets reproduces the host input
        std::vector<lla::MatrixTriplet<Float, M, N>> back(2);
        m.copy_to_triplets(std::span<lla::MatrixTriplet<Float, M, N>>{back});
        bool same = true;
        for(std::size_t i = 0; i < back.size(); ++i)
        {
            same = same && back[i].row_index == tris[i].row_index && back[i].col_index == tris[i].col_index;
            for(int r = 0; r < M; ++r)
                for(int c = 0; c < N; ++c) same = same && back[i].value(r, c) == tris[i].value(r, c);
        }
        CHECK(same);
    }

    // scalar (1x1) entries must be a plain T path with no indirection: the flat
    // storage of an nnz-entry scalar triplet matrix is exactly nnz long.
    {
        lla::LTripletMatrix<Float, 1> m;
        m.reshape(4, 4);
        m.resize_triplets(7);
        CHECK(m.values_buffer().size() == 7u);
        CHECK((lla::LTripletMatrix<Float, 1>::block_dim() == 1));
        CHECK((lla::LTripletMatrix<Float, 3>::block_dim() == 3));
    }

    // device side: write column-major blocks from a kernel and compare the raw
    // bytes with the host LMat (the layout claim, on the GPU).
    {
        constexpr int  M = 3, N = 3;
        constexpr std::size_t kBlocks = 4u;
        lla::LTripletMatrix<Float, M, N> m;
        std::vector<lla::MatrixTriplet<Float, M, N>> tris(kBlocks);
        for(std::size_t i = 0; i < kBlocks; ++i)
        {
            tris[i].row_index = static_cast<int>(i);
            tris[i].col_index = static_cast<int>(i);
            tris[i].value     = lps::LMat<Float, M, N>::Zero();
        }
        m.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, M, N>>{tris});

        // the same *host* block for the expected result
        lps::LMat<Float, M, N> expected;
        for(int r = 0; r < M; ++r)
            for(int c = 0; c < N; ++c) expected(r, c) = 100.0 * r + c;

        using WView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, M, N>;
        static auto write_block = lud::compile1<lcrt::Buffer<std::int32_t>,
                                                lcrt::Buffer<std::int32_t>,
                                                lcrt::Buffer<double>,
                                                uint32_t>(
            [](lcrt::BufferVar<std::int32_t> rows,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       vals,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto A = WView::from_handles(rows, cols, vals, n, n, n);
                auto i = lcrt::dispatch_x();
                $if (i < n)
                {
                    auto t = A(i).read();  // CTriplet (the Proxy's read step)
                    // build the block in the *device* math layer, then store it
                    lps::LMat<lcrt::Var<double>, M, N> b{};
                    for(int r = 0; r < M; ++r)
                    {
                        for(int c = 0; c < N; ++c)
                        {
                            b(r, c) = static_cast<double>(100 * r + c);
                        }
                    }
                    A(i).write(t.row_index, t.col_index, b);
                };
            });

        auto& rt     = test_runtime();
        auto&  stream = rt.main_stream();
        stream << write_block(m.row_indices(), m.col_indices(), m.values(),
                              static_cast<std::uint32_t>(kBlocks))
                      .dispatch(static_cast<std::uint32_t>(kBlocks))
               << lcrt::synchronize();

        std::vector<Float> flat(m.values_buffer().size(), 0.0);
        m.copy_to_values(std::span<Float>{flat});
        bool layout_ok = true;
        for(std::size_t i = 0; i < kBlocks; ++i)
        {
            for(int c = 0; c < N; ++c)
            {
                for(int r = 0; r < M; ++r)
                {
                    const std::size_t idx = i * M * N + static_cast<std::size_t>(c * M + r);
                    layout_ok = layout_ok && (flat[idx] == expected(r, c))
                                && (flat[idx]
                                    == expected.data[static_cast<std::size_t>(c * M + r)]);
                }
            }
        }
        CHECK(layout_ok);

        // and the host model of the device write: index(i, r, c) == i*9 + c*3 + r
        std::vector<Float> host_flat(kBlocks * M * N, 0.0);
        for(std::size_t i = 0; i < kBlocks; ++i)
            for(int c = 0; c < N; ++c)
                for(int r = 0; r < M; ++r)
                    host_flat[i * M * N + static_cast<std::size_t>(c * M + r)] = expected(r, c);
        CHECK(flat == host_flat);
    }
}

// ---------------------------------------------------------------------------
//  3. triplet/BCOO round trip + a real device SpMV, sizes 0/1/small/100k
// ---------------------------------------------------------------------------
namespace
{
template <int M, int N>
void triplet_roundtrip_impl(std::size_t nnz, int dofs, std::uint64_t seed)
{
    using Triplet = lla::LTripletMatrix<Float, M, N>;
    const auto host_tris = make_triplets<M, N>(nnz, dofs, seed);

    Triplet m;
    m.reshape(dofs, dofs);
    m.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, M, N>>{host_tris});
    CHECK(m.triplet_count() == nnz);
    CHECK(m.non_zeros() == static_cast<int>(nnz));

    auto& rt     = test_runtime();
    auto&  stream = rt.main_stream();

    // ---- (a) mutate through the view: block <- block * 2 - 1 --------------
    if(nnz > 0u)
    {
        stream << TripletScaleKernel<M, N>::get()(m.row_indices(),
                                                 m.col_indices(),
                                                 m.values(),
                                                 static_cast<std::uint32_t>(nnz))
                      .dispatch(static_cast<std::uint32_t>(nnz))
               << lcrt::synchronize();
    }

    std::vector<lla::MatrixTriplet<Float, M, N>> got(nnz);
    m.copy_to_triplets(std::span<lla::MatrixTriplet<Float, M, N>>{got});

    std::uint64_t bad   = 0u;
    std::uint64_t first = 0u;
    for(std::size_t i = 0; i < nnz; ++i)
    {
        // host model of "scale by 2 and shift by -1"
        const auto& src = host_tris[i];
        bool        ok  = got[i].row_index == src.row_index && got[i].col_index == src.col_index;
        if constexpr(M == 1 && N == 1)
        {
            ok = ok && close(got[i].value, src.value * 2.0 - 1.0, 1e-12);
        }
        else
        {
            for(int r = 0; r < M; ++r)
                for(int c = 0; c < N; ++c)
                    ok = ok && close(got[i].value(r, c), src.value(r, c) * 2.0 - 1.0, 1e-12);
        }
        if(!ok)
        {
            if(bad == 0u) first = i;
            ++bad;
        }
    }
    if(bad != 0u)
        std::printf("    [scale] nnz=%zu first mismatch at %llu\n", nnz,
                    static_cast<unsigned long long>(first));
    CHECK(bad == 0u);

    // ---- (b) a real SpMV: y = A x with atomics into y --------------------
    // The (a) step rewrote the container, so the host model must be derived from
    // the scaled host input — never from the device output.
    std::vector<lla::MatrixTriplet<Float, M, N>> scaled_tris = host_tris;
    for(auto& t : scaled_tris)
    {
        if constexpr(M == 1 && N == 1)
        {
            t.value = t.value * 2.0 - 1.0;
        }
        else
        {
            for(int r = 0; r < M; ++r)
                for(int c = 0; c < N; ++c) t.value(r, c) = t.value(r, c) * 2.0 - 1.0;
        }
    }
    std::vector<double> x        = make_dense(dofs, seed ^ 0xabcdu);
    auto                expect_y = host_spmv<M, N>(scaled_tris, x, dofs);

    lud::LDeviceBuffer<double> dev_x;
    lud::LDeviceBuffer<double> dev_y;
    dev_x.copy_from(std::span<const double>{x});
    dev_y.resize(static_cast<std::size_t>(dofs));
    dev_y.fill(0.0);

    if(nnz > 0u)
    {
        stream << TripletSpMVKernel<M, N>::get()(m.row_indices(),
                                                 m.col_indices(),
                                                 m.values(),
                                                 dev_x.view(),
                                                 dev_y.view(),
                                                 static_cast<std::uint32_t>(nnz),
                                                 static_cast<std::uint32_t>(dofs))
                      .dispatch(static_cast<std::uint32_t>(nnz))
               << lcrt::synchronize();
    }

    std::vector<double> got_y(static_cast<std::size_t>(dofs), 0.0);
    if(dofs > 0)
    {
        dev_y.view().copy_to(std::span<double>{got_y});
        stream.synchronize();
    }

    std::uint64_t ybad = 0u;
    double        worst = 0.0;
    for(int i = 0; i < dofs; ++i)
    {
        if(!close(got_y[static_cast<std::size_t>(i)], expect_y[static_cast<std::size_t>(i)], 1e-9))
        {
            ++ybad;
            worst = std::max(worst,
                             std::abs(got_y[static_cast<std::size_t>(i)]
                                      - expect_y[static_cast<std::size_t>(i)]));
        }
    }
    if(ybad != 0u)
        std::printf("    [spmv] nnz=%zu %llu/%d entries differ, worst |diff| = %.3e\n",
                    nnz,
                    static_cast<unsigned long long>(ybad),
                    dofs,
                    worst);
    CHECK(ybad == 0u);
}
}// namespace

TEST_CASE("linear_triplet_matrix_roundtrip", "[luisa][linear][device]")
{
    test_runtime();
    // sizes 0, 1, small, 100k; block sizes 1x1 and 3x3
    for(const auto nnz : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{100'000}})
    {
        triplet_roundtrip_impl<1, 1>(nnz, 1024, 0x1111u + nnz);
        triplet_roundtrip_impl<3, 3>(nnz, 900, 0x2222u + nnz);
    }
}

TEST_CASE("linear_bcoo_matrix_roundtrip", "[luisa][linear][device]")
{
    test_runtime();
    // BCOO is the triplet matrix under the muda name; exercise its own surface.
    using BCOO = lla::LBCOOMatrix<Float, 3>;
    const auto host_tris = make_triplets<3, 3>(11u, 30, 0x3333u);

    BCOO m;
    m.reshape(30, 30);
    m.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 3, 3>>{host_tris});
    CHECK(m.non_zeros() == 11);

    auto v = m.view();  // implicit conversion to BCOOMatrixView
    CHECK(v.total_rows() == 30);
    CHECK(v.triplet_count() == 11);

    lud::main_stream() << TripletScaleKernel<3, 3>::get()(m.row_indices(),
                                                          m.col_indices(),
                                                          m.values(),
                                                          11u)
                              .dispatch(11u)
                       << lcrt::synchronize();

    std::vector<lla::MatrixTriplet<Float, 3, 3>> got(11u);
    m.copy_to_triplets(std::span<lla::MatrixTriplet<Float, 3, 3>>{got});
    bool ok = true;
    for(std::size_t i = 0; i < got.size(); ++i)
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                ok = ok && close(got[i].value(r, c), host_tris[i].value(r, c) * 2.0 - 1.0, 1e-12);
    CHECK(ok);
}

// ---------------------------------------------------------------------------
//  4. doublet / BCOO vector round trip + gather
// ---------------------------------------------------------------------------
namespace
{
template <int N>
void doublet_roundtrip_impl(std::size_t nnz, int segments, std::uint64_t seed)
{
    using Doublet = lla::LDoubletVector<Float, N>;
    Lcg  rng;
    rng.s = seed;
    std::vector<lla::VectorDoublet<Float, N>> host_d(nnz);
    for(std::size_t i = 0; i < nnz; ++i)
    {
        host_d[i].index = rng.next_int(0, segments - 1);
        if constexpr(N == 1)
        {
            host_d[i].value = rng.next();
        }
        else
        {
            for(int k = 0; k < N; ++k) host_d[i].value(k) = rng.next();
        }
    }

    Doublet m;
    m.reshape(segments);
    m.copy_from_doublets(std::span<const lla::VectorDoublet<Float, N>>{host_d});
    CHECK(m.doublet_count() == nnz);
    CHECK(m.count() == segments);

    auto& rt     = test_runtime();
    auto&  stream = rt.main_stream();

    // (a) mutate through the view
    if(nnz > 0u)
    {
        stream << DoubletScaleKernel<N>::get()(m.indices(), m.values(), static_cast<std::uint32_t>(nnz))
                      .dispatch(static_cast<std::uint32_t>(nnz))
               << lcrt::synchronize();
    }

    std::vector<lla::VectorDoublet<Float, N>> got(nnz);
    m.copy_to_doublets(std::span<lla::VectorDoublet<Float, N>>{got});

    std::uint64_t bad = 0u;
    for(std::size_t i = 0; i < nnz; ++i)
    {
        bool ok = got[i].index == host_d[i].index;
        if constexpr(N == 1)
        {
            ok = ok && close(got[i].value, host_d[i].value * 3.0 + 1.0, 1e-12);
        }
        else
        {
            for(int k = 0; k < N; ++k)
                ok = ok && close(got[i].value(k), host_d[i].value(k) * 3.0 + 1.0, 1e-12);
        }
        bad += ok ? 0u : 1u;
    }
    CHECK(bad == 0u);

    // (b) gather: y[segment] += sum(value)  (atomic, per doublet)
    lud::LDeviceBuffer<double> dev_y;
    dev_y.resize(static_cast<std::size_t>(segments));
    dev_y.fill(0.0);

    if(nnz > 0u)
    {
        stream << DoubletGatherKernel<N>::get()(m.indices(),
                                               m.values(),
                                               dev_y.view(),
                                               static_cast<std::uint32_t>(nnz))
                      .dispatch(static_cast<std::uint32_t>(nnz))
               << lcrt::synchronize();
    }

    std::vector<double> got_y(static_cast<std::size_t>(segments), 0.0);
    if(segments > 0)
    {
        dev_y.view().copy_to(std::span<double>{got_y});
        stream.synchronize();
    }

    // host model over the *original* (pre-scale) values: the device read them
    // before the scale kernel wrote, so model the gathered value accordingly.
    std::vector<double> want(static_cast<std::size_t>(segments), 0.0);
    // The scale kernel rewrote the container; the host model is re-derived from
    // the host *input* (never from the device output).
    for(std::size_t i = 0; i < nnz; ++i)
    {
        double sum = 0.0;
        if constexpr(N == 1)
        {
            sum = host_d[i].value * 3.0 + 1.0;
        }
        else
        {
            for(int k = 0; k < N; ++k) sum += host_d[i].value(k) * 3.0 + 1.0;
        }
        want[static_cast<std::size_t>(host_d[i].index)] += sum;
    }

    std::uint64_t ybad = 0u;
    for(int i = 0; i < segments; ++i)
        if(!close(got_y[static_cast<std::size_t>(i)], want[static_cast<std::size_t>(i)], 1e-9))
            ++ybad;
    if(ybad != 0u)
        std::printf("    [doublet gather] nnz=%zu %llu/%d segments differ\n",
                    nnz,
                    static_cast<unsigned long long>(ybad),
                    segments);
    CHECK(ybad == 0u);
}
}// namespace

TEST_CASE("linear_doublet_vector_roundtrip", "[luisa][linear][device]")
{
    test_runtime();
    for(const auto nnz : {std::size_t{0}, std::size_t{1}, std::size_t{9}, std::size_t{100'000}})
    {
        doublet_roundtrip_impl<1>(nnz, 512, 0x4444u + nnz);
        doublet_roundtrip_impl<3>(nnz, 512, 0x5555u + nnz);
    }
}

TEST_CASE("linear_bcoo_vector_roundtrip", "[luisa][linear][device]")
{
    test_runtime();
    using BCOOVec = lla::LBCOOVector<Float, 3>;
    Lcg  rng;
    rng.s = 0x6666u;
    std::vector<lla::VectorDoublet<Float, 3>> host_d(8u);
    for(auto& d : host_d)
    {
        d.index = rng.next_int(0, 15);
        for(int k = 0; k < 3; ++k) d.value(k) = rng.next();
    }
    BCOOVec m;
    m.reshape(16);
    m.copy_from_doublets(std::span<const lla::VectorDoublet<Float, 3>>{host_d});
    CHECK(m.non_zeros() == 8);

    auto cv = m.cview();
    CHECK(cv.doublet_count() == 8);
    CHECK(cv.total_extent() == 16);
    CHECK(cv.extent() == 16);
    CHECK(cv.subvector_offset() == 0);
    auto sub = cv.subvector(4, 8);
    CHECK(sub.subvector_offset() == 4);
    CHECK(sub.extent() == 8);
    auto sv = cv.subview(2, 3);
    CHECK(sv.doublet_count() == 3);
    CHECK(sv.doublet_index_offset() == 2);
    CHECK(cv.total_doublet_count() == 8);
}

// ---------------------------------------------------------------------------
//  5. BSR: host fill -> device row walk -> host model
// ---------------------------------------------------------------------------
namespace
{
template <int N>
void bsr_roundtrip_impl(int rows, int max_nnz_per_row, std::uint64_t seed)
{
    Lcg rng;
    rng.s = seed;
    // host CSR structure
    std::vector<std::int32_t> row_offsets(static_cast<std::size_t>(rows) + 1u, 0);
    std::vector<std::int32_t> row_cols;
    std::vector<Float>        flat;  // N*N per block, column-major
    std::vector<double>       want(static_cast<std::size_t>(rows), 0.0);
    for(int r = 0; r < rows; ++r)
    {
        const int nnz_row = (max_nnz_per_row > 0) ? rng.next_int(0, max_nnz_per_row) : 0;
        for(int k = 0; k < nnz_row; ++k)
        {
            row_cols.push_back(rng.next_int(0, rows - 1));
            for(int c = 0; c < N; ++c)
            {
                for(int rr = 0; rr < N; ++rr)
                {
                    const double v = rng.next();
                    flat.push_back(v);
                    want[static_cast<std::size_t>(r)] += v;  // host model: sum of all elements
                }
            }
        }
        row_offsets[static_cast<std::size_t>(r) + 1u] = static_cast<std::int32_t>(row_cols.size());
    }

    lla::LBSRMatrix<Float, N> m;
    m.reshape(rows, rows);
    m.resize(static_cast<int>(row_cols.size()));
    m.copy_from_host(std::span<const std::int32_t>{row_offsets},
                     std::span<const std::int32_t>{row_cols},
                     std::span<const Float>{flat});
    CHECK(m.rows() == rows);
    CHECK(m.cols() == rows);
    CHECK(m.non_zeros() == static_cast<int>(row_cols.size()));
    CHECK(static_cast<std::size_t>(m.non_zero_blocks()) == row_cols.size());

    lud::LDeviceBuffer<double> dev_sum;
    dev_sum.resize(static_cast<std::size_t>(rows));
    dev_sum.fill(0.0);

    if(rows > 0 && !row_cols.empty())
    {
        test_runtime().main_stream()
            << BSRRowWalkKernel<N>::get()(m.row_offsets(),
                                          m.col_indices(),
                                          m.values(),
                                          dev_sum.view(),
                                          static_cast<std::uint32_t>(rows))
                   .dispatch(static_cast<std::uint32_t>(rows))
            << lcrt::synchronize();
    }

    std::vector<double> got(static_cast<std::size_t>(rows), 0.0);
    if(rows > 0)
    {
        dev_sum.view().copy_to(std::span<double>{got});
        test_runtime().main_stream().synchronize();
    }

    std::uint64_t bad = 0u;
    for(int r = 0; r < rows; ++r)
        if(!close(got[static_cast<std::size_t>(r)], want[static_cast<std::size_t>(r)], 1e-9)) ++bad;
    if(bad != 0u)
        std::printf("    [bsr row walk] rows=%d %llu differ\n", rows, static_cast<unsigned long long>(bad));
    CHECK(bad == 0u);
}
}// namespace

TEST_CASE("linear_bsr_row_walk", "[luisa][linear][device]")
{
    test_runtime();
    bsr_roundtrip_impl<1>(0, 0, 0x7777u);
    bsr_roundtrip_impl<1>(1, 1, 0x7778u);
    bsr_roundtrip_impl<1>(37, 5, 0x7779u);
    bsr_roundtrip_impl<3>(24, 4, 0x777au);
    bsr_roundtrip_impl<3>(20'000, 5, 0x777bu);  // 100k blocks in the worst case
}

// ---------------------------------------------------------------------------
//  6. dense vector / dense matrix
// ---------------------------------------------------------------------------
TEST_CASE("linear_dense_vector_roundtrip", "[luisa][linear][device]")
{
    test_runtime();
    for(const int n : {0, 1, 7, 30'000, 100'000})
    {
        lla::LDenseVector<Float> x, y;
        const auto               host_x = make_dense(n, 0x8888u + static_cast<std::uint64_t>(n));
        x.resize(static_cast<std::size_t>(n));
        x.copy_from(std::span<const double>{host_x});
        y.resize(static_cast<std::size_t>(n));
        y.fill(0.0);

        CHECK(x.size() == static_cast<std::size_t>(n));
        auto xv = x.view();
        CHECK(xv.size() == n);
        CHECK(xv.origin_size() == n);
        CHECK(xv.offset() == 0);
        auto sub = xv.subview(0, n);
        CHECK(sub.size() == n);

        auto& stream = test_runtime().main_stream();
        if(n > 0)
        {
            stream << DenseVectorSweepKernel::get()(x.buffer_view(), y.buffer_view(), static_cast<std::uint32_t>(n))
                          .dispatch(static_cast<std::uint32_t>(n))
                   << lcrt::synchronize();
        }

        std::vector<double> got(static_cast<std::size_t>(n), 0.0);
        y.copy_to(std::span<double>{got});

        std::uint64_t bad = 0u;
        for(int i = 0; i < n; ++i)
        {
            const double want = host_x[static_cast<std::size_t>(i)] * 2.0 + 1.0;
            bad += close(got[static_cast<std::size_t>(i)], want, 1e-12) ? 0u : 1u;
        }
        CHECK(bad == 0u);

        // container copy/assignment semantics (the PCG p = z pattern)
        lla::LDenseVector<Float> z;
        z.copy_from(x);
        CHECK(z.size() == x.size());
        std::vector<double> z_host(static_cast<std::size_t>(n), 0.0);
        z.copy_to(std::span<double>{z_host});
        bool same = z_host.size() == host_x.size();
        for(std::size_t i = 0; i < z_host.size(); ++i) same = same && z_host[i] == host_x[i];
        CHECK(same);
    }
}

TEST_CASE("linear_dense_vector_segment_kernel", "[luisa][linear][device]")
{
    test_runtime();
    constexpr int          SegN  = 3;
    constexpr int          nseg  = 1000;
    constexpr int          n     = nseg * SegN;
    std::vector<double>    host_x = make_dense(n, 0x9999u);

    lla::LDenseVector<Float> x, y;
    x.resize(static_cast<std::size_t>(n));
    x.copy_from(std::span<const double>{host_x});
    y.resize(static_cast<std::size_t>(n));
    y.fill(0.0);

    test_runtime().main_stream()
        << DenseVectorSegmentKernel<SegN>::get()(x.buffer_view(), y.buffer_view(), static_cast<std::uint32_t>(nseg))
               .dispatch(static_cast<std::uint32_t>(nseg))
        << lcrt::synchronize();

    std::vector<double> got(static_cast<std::size_t>(n), 0.0);
    y.copy_to(std::span<double>{got});

    // host model: y = x*2 (from as_eigen +=) then + x (the atomic add),
    // i.e. y = 3x, per segment (the segment starts at zero)
    std::uint64_t bad = 0u;
    for(int i = 0; i < n; ++i)
    {
        const double want = 3.0 * host_x[static_cast<std::size_t>(i)];
        bad += close(got[static_cast<std::size_t>(i)], want, 1e-12) ? 0u : 1u;
    }
    CHECK(bad == 0u);

    // the statically sized segment view is a different type with a compile-time
    // size (the Eigen `segment<SegN>` analogue)
    static_assert(std::is_same_v<decltype(x.view().segment<SegN>(0)),
                                 lla::LDenseVectorView<Float, SegN>>);
    static_assert(lla::LDenseVectorView<Float, SegN>::has_static_size);
    static_assert(!lla::LDenseVectorView<Float>::has_static_size);
}

TEST_CASE("linear_dense_matrix_roundtrip", "[luisa][linear][device]")
{
    test_runtime();
    for(const int rows : {0, 1, 17})
    {
        const int  cols     = rows > 0 ? rows + 3 : 0;
        const auto host_in  = make_dense(rows * cols, 0xaaaau + static_cast<std::uint64_t>(rows));

        lla::LDenseMatrix<Float> in, out;
        in.reshape(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        in.copy_from(std::span<const double>{host_in});
        out.reshape(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        out.fill(0.0);
        CHECK(in.row() == static_cast<std::size_t>(rows));
        CHECK(in.col() == static_cast<std::size_t>(cols));
        in.sym(true);
        // `sym()` (the getter) is the const overload
        const auto& cin = in;
        CHECK(cin.sym());

        if(rows * cols > 0)
        {
            test_runtime().main_stream()
                << DenseMatrixSweepKernel::get()(in.buffer_view(),
                                                 out.buffer_view(),
                                                 static_cast<std::uint32_t>(rows),
                                                 static_cast<std::uint32_t>(cols))
                       .dispatch(static_cast<std::uint32_t>(rows * cols))
                << lcrt::synchronize();
        }

        std::vector<double> got(static_cast<std::size_t>(rows * cols), 0.0);
        out.copy_to(std::span<double>{got});

        std::uint64_t bad = 0u;
        for(int i = 0; i < rows * cols; ++i)
        {
            const double want = host_in[static_cast<std::size_t>(i)] * 3.0 - 1.0;
            bad += close(got[static_cast<std::size_t>(i)], want, 1e-12) ? 0u : 1u;
        }
        CHECK(bad == 0u);
    }
}

// ---------------------------------------------------------------------------
//  7. a view mutation is visible in the container, and in a *subview* window
// ---------------------------------------------------------------------------
TEST_CASE("linear_view_mutates_container", "[luisa][linear][device]")
{
    test_runtime();
    using Triplet = lla::LTripletMatrix<Float, 1>;
    const auto host_tris = make_triplets<1, 1>(6u, 8, 0xbbbbu);

    Triplet m;
    m.reshape(8, 8);
    m.copy_from_triplets(std::span<const lla::MatrixTriplet<Float, 1, 1>>{host_tris});

    auto v_before = m.view();  // container -> view share the storage
    CHECK(v_before.triplet_count() == 6);
    CHECK(v_before.generation() == m.generation());

    // The kernel receives the *whole* arrays and re-creates the subview in-kernel
    // (a view is not a kernel argument); it writes a marker through the window
    // [2, 5) only.
    static auto write_window = lud::compile1<lcrt::Buffer<std::int32_t>,
                                             lcrt::Buffer<std::int32_t>,
                                             lcrt::Buffer<double>,
                                             uint32_t>(
        [](lcrt::BufferVar<std::int32_t> rows,
           lcrt::BufferVar<std::int32_t> cols,
           lcrt::BufferVar<double>       vals,
           lcrt::Var<uint32_t>           count) noexcept
        {
            auto A  = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 1>::from_handles(
                rows, cols, vals, count, count, count);
            auto  W       = A.subview(2, 3);  // view-relative window
            auto  i       = lcrt::dispatch_x();
            $if (i < 3u)
            {
                auto t = W(i).read();  // index is view-relative, globals come from the view
                W(i).write(t.row_index, t.col_index, 123.0);
            };
        });

    test_runtime().main_stream()
        << write_window(m.row_indices(), m.col_indices(), m.values(), 6u).dispatch(3u)
        << lcrt::synchronize();

    // mutating through the view mutated the container: exactly entries [2, 5)
    std::vector<double> vals(m.values_buffer().size(), 0.0);
    m.copy_to_values(std::span<double>{vals});
    std::vector<std::int32_t> rows(m.triplet_count()), cols(m.triplet_count());
    m.copy_to_row_indices(std::span<std::int32_t>{rows});
    m.copy_to_col_indices(std::span<std::int32_t>{cols});

    std::uint64_t touched_ok  = 0u;
    std::uint64_t others_ok   = 0u;
    for(std::size_t i = 0; i < 6u; ++i)
    {
        const bool in_window = (i >= 2u && i < 5u);
        if(in_window)
        {
            touched_ok += (vals[i] == 123.0) ? 1u : 0u;
            // the row/col written are view-relative + the submatrix offset (0)
            others_ok += (rows[i] == host_tris[i].row_index && cols[i] == host_tris[i].col_index) ? 1u : 0u;
        }
        else
        {
            others_ok += (vals[i] == host_tris[i].value) ? 1u : 0u;
        }
    }
    CHECK(touched_ok == 3u);
    CHECK(others_ok == 6u);
    CHECK(m.triplet_count() == 6u);
    CHECK(m.view().triplet_count() == 6);

    // the container's own metadata is untouched by the window write
    CHECK(m.view().total_triplet_count() == 6);
    CHECK(m.view().tripet_index_offset() == 0);

    // a resize bumps the generation: the pre-resize view is stale, a new one is
    // valid again (plan risk R5)
    const auto gen = m.generation();
    m.resize_triplets(6u);
    CHECK(m.generation() != gen);
    CHECK_THROWS(v_before.use_check());
    m.view().use_check();
    CHECK(m.triplet_count() == 6u);
}
