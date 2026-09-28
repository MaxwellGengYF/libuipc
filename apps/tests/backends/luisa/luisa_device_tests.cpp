/********************************************************************
 * @file luisa_device_tests.cpp
 * @brief T1.3 — foundation unit tests for the luisa_device layer.
 *
 * Covers: runtime lifecycle, buffer/container semantics, subviews, device
 * transfers, dispatch with runtime counts, CommandList batching, atomics
 * (incl. the double-atomic shim), warp/block reductions and device print.
 *
 * Builds in-tree (Catch2 via uipc_add_test) or standalone (see
 * luisa_test_shim.h and apps/tests/backends/luisa/standalone/CMakeLists.txt).
 *********************************************************************/
#include "luisa_test_shim.h"

#include <luisa_device/luisa_device.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lud = uipc::backend::luisa::luisa_device;

// LC derives its runtime/builtin-data directory from the program path, so the
// tests must pass a real one (argv[0]) rather than a placeholder string.
#if defined(UIPC_LUISA_TESTS_STANDALONE)
#define UIPC_TEST_PROGRAM_PATH ::uipc_luisa_test::program_path()
#else
#define UIPC_TEST_PROGRAM_PATH "luisa_backend_tests"
#endif

namespace
{
// The Runtime is process-wide; initialize it once on first use.
lud::Runtime& runtime(const char* backend = "cuda")
{
    auto& rt = lud::Runtime::instance();
    if(!rt.initialized())
    {
        rt.init(backend, UIPC_TEST_PROGRAM_PATH);
    }
    return rt;
}
}// namespace

TEST_CASE("runtime_init", "[luisa][device]")
{
    auto& rt = runtime();
    CHECK(rt.initialized());
    // idempotent
    rt.init("cuda", UIPC_TEST_PROGRAM_PATH);
    CHECK(rt.initialized());
    auto name = std::string{rt.backend_name()};
    CHECK(name == "cuda");
    auto dev_name = std::string{rt.device().backend_name()};
    CHECK(!dev_name.empty());
}

TEST_CASE("buffer_ops", "[luisa][device]")
{
    runtime();

    lud::LDeviceBuffer<double> buf;
    CHECK(buf.size() == 0u);

    buf.resize(16u);
    CHECK(buf.size() == 16u);
    CHECK(buf.capacity() >= 16u);

    // fill (device kernel)
    buf.fill(3.5);
    std::vector<double> host(16u, 0.0);
    buf.view().copy_to(luisa::span{host});
    lud::main_stream().synchronize();
    bool all_filled = true;
    for(auto v : host) all_filled = all_filled && std::abs(v - 3.5) < 1e-12;
    CHECK(all_filled);

    // upload
    std::vector<double> src(16u);
    std::iota(src.begin(), src.end(), 1.0);
    buf.copy_from(luisa::span<const double>{src});
    buf.view().copy_to(luisa::span{host});
    lud::main_stream().synchronize();
    bool up_ok = true;
    for(size_t i = 0; i < host.size(); ++i) up_ok = up_ok && std::abs(host[i] - src[i]) < 1e-12;
    CHECK(up_ok);

    // subview read-back
    std::vector<double> half(4u, 0.0);
    buf.view(4u, 4u).copy_to(luisa::span{half});
    lud::main_stream().synchronize();
    bool sub_ok = true;
    for(size_t i = 0; i < half.size(); ++i) sub_ok = sub_ok && std::abs(half[i] - src[4u + i]) < 1e-12;
    CHECK(sub_ok);

    // grow + shrink semantics
    const auto cap_before = buf.capacity();
    buf.resize(64u);
    CHECK(buf.size() == 64u);
    CHECK(buf.capacity() >= cap_before);
    buf.clear();
    CHECK(buf.size() == 0u);
    CHECK(buf.capacity() >= cap_before);  // clear keeps the allocation

    // d2d copy
    lud::LDeviceBuffer<double> dst;
    dst.resize(16u);
    std::vector<double> src2(16u);
    std::iota(src2.begin(), src2.end(), 100.0);
    dst.copy_from(luisa::span<const double>{src2});
    std::vector<double> back(16u, 0.0);
    dst.view().copy_to(luisa::span{back});
    lud::main_stream().synchronize();
    bool d2d_ok = true;
    for(size_t i = 0; i < back.size(); ++i) d2d_ok = d2d_ok && std::abs(back[i] - src2[i]) < 1e-12;
    CHECK(d2d_ok);
}

TEST_CASE("device_var_scalar", "[luisa][device]")
{
    runtime();
    lud::LDeviceVar<double> v;
    v.copy_from(42.0);
    lud::main_stream().synchronize();
    CHECK(std::abs(static_cast<double>(v) - 42.0) < 1e-12);
}

TEST_CASE("parallel_for", "[luisa][device]")
{
    runtime();

    constexpr uint32_t N = 4096u;
    lud::LDeviceBuffer<double> out;
    out.resize(N);

    // compile-once, dispatch with a *runtime* count
    static auto k = lud::compile1<luisa::compute::Buffer<double>, uint32_t>(
        [](luisa::compute::BufferVar<double> v, luisa::compute::Var<uint32_t> n) noexcept {
            auto i = luisa::compute::dispatch_x();
            $if (i < n) { v->write(i, luisa::compute::cast<double>(i) * 2.0); };
        });

    const uint32_t count = N / 2u;
    lud::main_stream() << k(out.view(), count).dispatch(N);

    // resize() value-initializes, so elements past `count` must stay 0.
    std::vector<double> host(N, -1.0);
    out.view().copy_to(luisa::span{host});
    lud::main_stream().synchronize();

    bool ok = true;
    for(uint32_t i = 0; i < N; ++i)
    {
        const double want = (i < count) ? static_cast<double>(i) * 2.0 : 0.0;
        ok = ok && std::abs(host[i] - want) < 1e-12;
    }
    CHECK(ok);
}

TEST_CASE("stream_commands", "[luisa][device]")
{
    runtime();

    constexpr uint32_t N = 1024u;
    lud::LDeviceBuffer<double> a, b;
    a.resize(N);
    b.resize(N);

    static auto set_k = lud::compile1<luisa::compute::Buffer<double>, double>(
        [](luisa::compute::BufferVar<double> v, luisa::compute::Var<double> value) noexcept {
            v->write(luisa::compute::dispatch_x(), value);
        });
    static auto copy_k = lud::compile1<luisa::compute::Buffer<double>, luisa::compute::Buffer<double>>(
        [](luisa::compute::BufferVar<double> dst, luisa::compute::BufferVar<double> src) noexcept {
            auto i = luisa::compute::dispatch_x();
            dst->write(i, src->read(i));
        });

    // Batch several dispatches into one CommandList and commit once.
    lud::LStream stream;
    stream.begin();
    stream.record(set_k(a.view(), 7.0).dispatch(N));
    stream.record(copy_k(b.view(), a.view()).dispatch(N));
    stream.commit();
    stream.synchronize();

    std::vector<double> host(N, 0.0);
    b.view().copy_to(luisa::span{host});
    lud::main_stream().synchronize();
    bool ok = true;
    for(auto v : host) ok = ok && std::abs(v - 7.0) < 1e-12;
    CHECK(ok);
}

TEST_CASE("atomic_and_reductions", "[luisa][device]")
{
    runtime();

    constexpr uint32_t N = 8192u;
    lud::LDeviceBuffer<double> values;
    values.resize(N);
    std::vector<double> init(N);
    for(uint32_t i = 0; i < N; ++i) init[i] = 1.0;  // sum == N
    values.copy_from(luisa::span<const double>{init});

    lud::LDeviceVar<double> warp_acc, block_acc;
    warp_acc.copy_from(0.0);
    block_acc.copy_from(0.0);

    // warp reduction + double atomic add (uses the AtomicRef<double> shim)
    static auto warp_k = lud::compile1<luisa::compute::Buffer<double>, luisa::compute::Buffer<double>>(
        [](luisa::compute::BufferVar<double> v, luisa::compute::BufferVar<double> out) noexcept {
            auto w = luisa::compute::warp_active_sum(v->read(luisa::compute::dispatch_x()));
            $if (luisa::compute::warp_lane_id() == 0u)
            {
                out->atomic(0u).fetch_add(w);
            };
        });

    // block reduction through Shared<T> + sync_block()
    static auto block_k = lud::compile1<luisa::compute::Buffer<double>, luisa::compute::Buffer<double>>(
        [](luisa::compute::BufferVar<double> v, luisa::compute::BufferVar<double> out) noexcept {
            luisa::compute::set_block_size(256u);
            $shared<double> scratch{256u};
            auto tid = luisa::compute::thread_x();
            scratch[tid] = v->read(luisa::compute::dispatch_x());
            luisa::compute::sync_block();
            $if (tid == 0u)
            {
                luisa::compute::Var<double> s = 0.0;
                $for (k, 0u, 256u) { s += scratch[k]; };
                out->atomic(0u).fetch_add(s);
            };
        });

    lud::main_stream() << warp_k(values.view(), warp_acc.view()).dispatch(N)
                      << block_k(values.view(), block_acc.view()).dispatch(N)
                      << luisa::compute::synchronize();

    const double want = static_cast<double>(N);
    CHECK(std::abs(warp_acc.copy_to() - want) < 1e-6);
    CHECK(std::abs(block_acc.copy_to() - want) < 1e-6);
}

TEST_CASE("kernel_print", "[luisa][device]")
{
    runtime();
    lud::LDeviceBuffer<double> buf;
    buf.resize(4u);

    // LC_LOG is compiled out unless UIPC_LUISA_KERNEL_PRINT is on; when on it
    // must not break dispatch or the following copy.
    static auto k = lud::compile1<luisa::compute::Buffer<double>>(
        [](luisa::compute::BufferVar<double> v) noexcept {
            auto i = luisa::compute::dispatch_x();
            v->write(i, 1.0);
            LC_LOG("print test i={}", i);
        });

    lud::main_stream() << k(buf.view()).dispatch(4u) << luisa::compute::synchronize();
    std::vector<double> host(4u, 0.0);
    buf.view().copy_to(luisa::span{host});
    lud::main_stream().synchronize();
    bool ok = true;
    for(auto v : host) ok = ok && std::abs(v - 1.0) < 1e-12;
    CHECK(ok);
}
