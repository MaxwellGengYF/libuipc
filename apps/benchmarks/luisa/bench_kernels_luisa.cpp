/********************************************************************
 * @file bench_kernels_luisa.cpp
 * @brief Tier 1 (bench_plan.md §4.2) for the LuisaCompute backend: the product kernels.
 *
 * This is the Luisa implementation of `common/bench_kernels.h`.  It wraps the *product*
 * code, never a re-implementation:
 *
 * | case | product code | what a frame uses it for |
 * |---|---|---|
 * | K1 | `luisa_device::linalg::LSpmv` (4 variants) | one PCG operator apply |
 * | K2 | `linalg::dot` / `norm` / `axpy` / `fill` | the PCG inner products |
 * | K0 | `LBufferView::fill` / `copy_from` | the bandwidth roofline |
 *
 * FIXTURES ARE DUPLICATED ON PURPOSE.  The matrix/vector builders below are the same
 * construction the SpMV test uses (`apps/tests/backends/luisa/luisa_device_spmv_tests.cpp`,
 * `make_random_symmetric` / `make_bcoo` / `to_dense`): a benchmark must not link the test
 * binary, and a benchmark that re-implements the kernel would measure the wrong thing.
 * Only the "random" shape (mode 0) is reproduced -- the adversarial shapes belong to the
 * correctness suite, not to a throughput measurement.
 *
 * Sizing follows §4.2: K1 is the `spmv_large_timing` shape (1000 rows, density 0.2,
 * >= 1e5 block entries, 3 dofs per row); K2 sweeps the two sizes §4.2 calls out
 * (n = 1e6 and the anomalous n = 3000); K0 is 64 MiB of float, i.e. `16M` elements.
 *********************************************************************/
#include <common/bench_kernels.h>

#include <luisa_device/luisa_device.h>
#include <luisa_device/linear/linear.h>
#include <luisa_device/tuning.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace uipc::benchmark
{
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lud  = uipc::backend::luisa::luisa_device;

namespace
{
using Float = double;  // uipc::Float is double

// ---------------------------------------------------------------------------
//  deterministic fixtures (same construction as the SpMV tests, mode 0 only)
// ---------------------------------------------------------------------------
struct Lcg
{
    std::uint64_t s = 0x2545'f491'3c6e'f8a1ULL;
    std::uint64_t next_u64() noexcept
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s;
    }
    double next() noexcept { return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0); }
    double next_sym() noexcept { return next() * 2.0 - 1.0; }
};

template <int N>
using Triplet = lla::MatrixTriplet<Float, N, N>;

/// Upper-packed symmetric block BCOO: one entry per selected (i, j), i <= j, the
/// diagonal always present.  `density` is the probability of an off-diagonal entry.
template <int N>
std::vector<Triplet<N>> make_random_symmetric(std::uint64_t rows, double density, std::uint64_t seed)
{
    Lcg rng;
    rng.s = seed;
    std::vector<Triplet<N>> trips;
    trips.reserve(static_cast<std::size_t>(rows));

    auto set_block = [](typename Triplet<N>::ValueT& v, int r, int c, double x) noexcept
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
    };

    for(std::uint64_t i = 0; i < rows; ++i)
    {
        for(std::uint64_t j = i; j < rows; ++j)
        {
            const bool diag = (i == j);
            if(!diag && rng.next() >= density)
                continue;
            Triplet<N> t;
            t.row_index = static_cast<std::int32_t>(i);
            t.col_index = static_cast<std::int32_t>(j);
            for(int r = 0; r < N; ++r)
                for(int c = 0; c < N; ++c)
                    set_block(t.value, r, c, diag ? (1.0 + std::abs(rng.next_sym())) : rng.next_sym());
            trips.push_back(t);
        }
    }
    return trips;
}

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
}  // namespace

struct BenchKernels::Impl
{
    DeviceInfo info;
    /// Leaked on purpose: an `LStream` destroyed during static destruction would call
    /// into the LC runtime after it is gone (cookbook §7.4/§7.10), same as the fixtures.
    lud::LStream* stream = nullptr;

    // K1
    lla::LSpmv<Float, 3>      spmv;
    lla::LBCOOMatrix<Float, 3> A;
    lla::LDenseVector<Float>  x, y;
    lud::LDeviceVar<Float>         d_dot{0.0};
    lud::LDeviceVar<std::int32_t>  count_dev;
    SpmvSetup                      spmv_setup;

    // K2
    lla::LDenseVector<Float> dense_x, dense_y;
    std::uint64_t            dense_n = 0;
    double                   dot_result = 0.0;

    // K0
    lud::LDeviceBuffer<float>      bw_a, bw_b;
    std::uint64_t                  bw_floats = 0;
    std::unique_ptr<lla::LLinearSystemContext> ctx;
};

BenchKernels::BenchKernels() : m_impl{std::make_unique<Impl>()} {}

BenchKernels::~BenchKernels() = default;

void BenchKernels::init(std::string_view program_path)
{
    auto& rt = lud::Runtime::instance();
    if(!rt.initialized())
        rt.init("cuda", std::string{program_path});
    m_impl->info.backend = "luisa";
    m_impl->info.device_name = std::string{lud::device().backend_name()} + " device";
    m_impl->info.runtime_version = "LuisaCompute " + std::string{lud::device().backend_name()};
    m_impl->info.extra = "block_dim=" + std::to_string(lud::block_dim_for())
                         + " ccd_max_iter=" + std::to_string(lud::ccd_max_iter());
    if(m_impl->stream == nullptr)
        m_impl->stream = new lud::LStream{lud::main_stream()};
    if(!m_impl->ctx)
        m_impl->ctx = std::make_unique<lla::LLinearSystemContext>(m_impl->stream);
}

void BenchKernels::shutdown()
{
    // Tear the module runtime down *here*, exactly like the backend test executable's
    // `zzz_runtime_shutdown.cpp` case: an LC runtime destroyed during static destruction
    // runs after the module's own statics are gone and fast-fails (0xC0000409) -- with
    // stdio unflushed, which is why the last lines of a run would silently disappear.
    // The shader cache itself is process-lifetime on purpose (luisa_device/linear/
    // algorithms.h: `cached_kernel` leaks), so nothing else needs releasing.
    auto& rt = lud::Runtime::instance();
    if(rt.initialized())
        rt.shutdown();
}

const DeviceInfo& BenchKernels::info() const noexcept
{
    return m_impl->info;
}

std::uint32_t BenchKernels::block_dim() const noexcept
{
    return lud::block_dim_for();
}

void BenchKernels::synchronize()
{
    m_impl->stream->synchronize();
}

// ---------------------------------------------------------------------------
//  K1 — SpMV
// ---------------------------------------------------------------------------
BenchKernels::SpmvSetup BenchKernels::make_spmv(std::uint64_t rows,
                                                double        entries_per_row,
                                                std::uint32_t seed)
{
    const auto density = rows == 0 ? 0.0 : std::min(1.0, entries_per_row / static_cast<double>(rows));
    auto trips = make_random_symmetric<3>(rows, density, seed);

    m_impl->A = lla::LBCOOMatrix<Float, 3>{};
    m_impl->A.reshape(static_cast<int>(rows), static_cast<int>(rows));
    m_impl->A.copy_from_triplets(std::span<const Triplet<3>>{trips});

    const auto dofs = rows * 3u;
    Lcg        rng;
    rng.s = 0xf00d'0000u + seed;
    std::vector<Float> xh(dofs);
    for(auto& v : xh)
        v = rng.next_sym();
    m_impl->x = to_dense(xh);
    m_impl->y = to_dense(std::vector<Float>(dofs, 0.0));
    m_impl->count_dev.copy_from(static_cast<std::int32_t>(m_impl->A.triplet_count()));

    auto& s      = m_impl->spmv_setup;
    s.rows       = rows;
    s.triplets   = m_impl->A.triplet_count();
    s.dofs       = dofs;
    // A diagonal block touches 9 values once; an off-diagonal block is mirrored, so it
    // contributes 18.  x and y are both streamed once.
    s.nnz        = s.triplets * 9u;  // scalar blocks stored (the mirror reads them again)
    s.bytes_per_call = s.triplets * 9u * sizeof(Float) * 2u  // the mirrored block traffic
                       + dofs * sizeof(Float) * 2u;          // x read + y read/write
    return s;
}

const BenchKernels::SpmvSetup& BenchKernels::spmv_setup() const noexcept
{
    return m_impl->spmv_setup;
}

const char* BenchKernels::name(SpmvVariant v) noexcept
{
    switch(v)
    {
        case SpmvVariant::Sym: return "sym_spmv";
        case SpmvVariant::Rbk: return "rbk_spmv";
        case SpmvVariant::RbkSym: return "rbk_sym_spmv";
        case SpmvVariant::RbkSymDot: return "rbk_sym_spmv_dot";
        default: return "?";
    }
}

void BenchKernels::spmv(SpmvVariant variant)
{
    auto& i = *m_impl;
    switch(variant)
    {
        case SpmvVariant::Sym:
            i.spmv.sym_spmv(1.0, i.A.cview(), i.x.cview(), 1.0, i.y.view(), *i.stream);
            break;
        case SpmvVariant::Rbk:
            i.spmv.rbk_spmv(1.0, i.A.cview(), i.x.cview(), 1.0, i.y.view(), *i.stream);
            break;
        case SpmvVariant::RbkSym:
            i.spmv.rbk_sym_spmv(1.0, i.A.cview(), i.x.cview(), 1.0, i.y.view(), *i.stream);
            break;
        case SpmvVariant::RbkSymDot:
            i.spmv.rbk_sym_spmv_dot(1.0,
                                    i.A.cview(),
                                    i.x.cview(),
                                    1.0,
                                    i.y.view(),
                                    i.d_dot.view(),
                                    i.count_dev.cview(),
                                    i.A.triplet_capacity(),
                                    *i.stream);
            break;
        default: break;
    }
}

double BenchKernels::spmv_checksum() const
{
    auto& i = *m_impl;
    i.stream->synchronize();
    const std::size_t probe = std::min<std::size_t>(64u, i.spmv_setup.dofs);
    std::vector<Float> host(probe, 0.0);
    if(probe != 0u)
        i.y.buffer_view().subview(0u, probe).copy_to(std::span<Float>{host});
    i.stream->synchronize();
    double sum = 0.0;
    for(auto v : host)
        sum += v;
    return sum;
}

// ---------------------------------------------------------------------------
//  K2 — dense BLAS / reduction
// ---------------------------------------------------------------------------
void BenchKernels::make_dense(std::uint64_t n, std::uint32_t seed)
{
    m_impl->dense_n = n;
    Lcg rng;
    rng.s = 0xbeef'0000u + seed;
    std::vector<Float> xh(n), yh(n);
    for(std::size_t k = 0; k < n; ++k)
    {
        xh[k] = rng.next_sym();
        yh[k] = rng.next_sym();
    }
    m_impl->dense_x = n == 0u ? lla::LDenseVector<Float>{} : to_dense(xh);
    m_impl->dense_y = n == 0u ? lla::LDenseVector<Float>{} : to_dense(yh);
}

std::uint64_t BenchKernels::dense_n() const noexcept
{
    return m_impl->dense_n;
}

std::uint64_t BenchKernels::dense_bytes_per_call(int op) const noexcept
{
    const auto n = m_impl->dense_n;
    switch(static_cast<BlasOp>(op))
    {
        case BlasOp::Dot: return 2u * n * sizeof(Float);
        case BlasOp::Nrm2: return n * sizeof(Float);
        case BlasOp::Axpy: return 3u * n * sizeof(Float);
        case BlasOp::Fill: return n * sizeof(Float);
        default: return 0u;
    }
}

const char* BenchKernels::name(BlasOp op) noexcept
{
    switch(op)
    {
        case BlasOp::Dot: return "dot";
        case BlasOp::Nrm2: return "nrm2";
        case BlasOp::Axpy: return "axpy";
        case BlasOp::Fill: return "fill";
        default: return "?";
    }
}

void BenchKernels::blas(BlasOp op)
{
    auto& i = *m_impl;
    switch(op)
    {
        case BlasOp::Dot:
            i.dot_result = lla::dot(i.dense_x.buffer_view(), i.dense_y.buffer_view(), *i.stream);
            break;
        case BlasOp::Nrm2:
            i.dot_result = lla::norm(i.dense_x.buffer_view(), *i.stream);
            break;
        case BlasOp::Axpy:
            lla::axpy(Float{0.5}, i.dense_x.buffer_view(), i.dense_y.buffer_view(), *i.stream);
            break;
        case BlasOp::Fill:
            lla::fill(i.dense_y.buffer_view(), Float{1.0}, *i.stream);
            break;
        default: break;
    }
}

double BenchKernels::blas_checksum() const
{
    auto& i = *m_impl;
    i.stream->synchronize();
    const std::size_t probe = std::min<std::size_t>(64u, i.dense_n);
    std::vector<Float> host(probe, 0.0);
    if(probe != 0u)
        i.dense_y.buffer_view().subview(0u, probe).copy_to(std::span<Float>{host});
    i.stream->synchronize();
    double sum = 0.0;
    for(auto v : host)
        sum += v;
    return sum + 1.0e-300 * i.dot_result;  // keeps the reduction result in the checksum
}

// ---------------------------------------------------------------------------
//  K0 — pure bandwidth
// ---------------------------------------------------------------------------
void BenchKernels::make_bandwidth(std::uint64_t floats, std::uint32_t seed)
{
    m_impl->bw_floats = floats;
    m_impl->bw_a.resize(floats);
    m_impl->bw_b.resize(floats);
    Lcg rng;
    rng.s = 0xfeed'0000u + seed;
    std::vector<float> host(floats);
    for(auto& v : host)
        v = static_cast<float>(rng.next_sym());
    m_impl->bw_a.view().copy_from(std::span<const float>{host});
    m_impl->stream->synchronize();
}

std::uint64_t BenchKernels::bandwidth_floats() const noexcept
{
    return m_impl->bw_floats;
}

const char* BenchKernels::name(BwOp op) noexcept
{
    switch(op)
    {
        case BwOp::Fill: return "fill";
        case BwOp::Copy: return "copy";
        default: return "?";
    }
}

std::uint64_t BenchKernels::bandwidth_bytes_per_call(BwOp op) const noexcept
{
    const auto bytes = m_impl->bw_floats * sizeof(float);
    switch(op)
    {
        case BwOp::Fill: return bytes;
        case BwOp::Copy: return 2u * bytes;
        default: return 0u;
    }
}

void BenchKernels::bandwidth(BwOp op)
{
    auto& i = *m_impl;
    switch(op)
    {
        case BwOp::Fill: i.bw_b.view().fill(1.0f, i.stream); break;
        case BwOp::Copy: i.bw_b.view().copy_from(i.bw_a.cview(), i.stream); break;
        default: break;
    }
}

double BenchKernels::bandwidth_checksum() const
{
    auto& i = *m_impl;
    i.stream->synchronize();
    const std::size_t probe = std::min<std::size_t>(64u, i.bw_floats);
    std::vector<float> host(probe, 0.0f);
    if(probe != 0u)
        i.bw_b.cview(0u, probe).copy_to(std::span<float>{host});
    i.stream->synchronize();
    double sum = 0.0;
    for(auto v : host)
        sum += v;
    return sum;
}
}  // namespace uipc::benchmark
