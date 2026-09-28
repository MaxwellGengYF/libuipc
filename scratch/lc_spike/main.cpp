// T0.1 — LuisaCompute device spike.
//
// Verifies, on the local NVIDIA GPU through LC's CUDA backend:
//   1. Context + Device + Stream creation.
//   2. Kernel1D compile (device.compile<1>) + dispatch.
//   3. CommandList recording and commit (explicit runtime command path).
//   4. device->host copy back and value verification.
//   5. device_log() capture (kernel printf replacement for muda::Logger).
//   6. Warp reduction (warp_active_sum) + atomic_fetch_add, i.e. the
//      primitives the linear-algebra layer (Phase 3) depends on.
//
// Exit code 0 == all checks passed.

#include <luisa/luisa-compute.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace luisa;
using namespace luisa::compute;

namespace {

int g_failures = 0;

void check(bool cond, const char *what) {
    if (cond) {
        std::printf("  [ ok ] %s\n", what);
    } else {
        std::printf("  [FAIL] %s\n", what);
        ++g_failures;
    }
    std::fflush(stdout);
}

}// namespace

int main(int argc, char *argv[]) {
    const char *backend = (argc > 1 && argv[1] != nullptr) ? argv[1] : "cuda";
    std::printf("== LC spike == backend=%s\n", backend);

    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    auto backend_name = std::string{device.backend_name()};
    std::printf("  device: %s\n", backend_name.c_str());
    Stream stream = device.create_stream();

    constexpr uint32_t N = 1024u;
    auto buf = device.create_buffer<float>(N);
    auto sum = device.create_buffer<float>(1u);

    // ---- 1. trivial Kernel1D: buf[i] = 1 + i -----------------------------
    auto write_kernel = device.compile<1>([&]() noexcept {
        auto i = dispatch_id().x;
        buf->write(i, 1.0f + cast<float>(i));
    });

    // ---- 6. warp reduction + atomic --------------------------------------
    // Sum N values: intra-warp reduce with warp_active_sum, then one atomic per
    // warp. Also exercises device_log (only lane 0 of block 0 prints).
    auto reduce_kernel = device.compile<1>([&]() noexcept {
        auto i = dispatch_id().x;
        auto v = buf->read(i);
        auto w = warp_active_sum(v);
        $if (warp_lane_id() == 0u) {
            sum->atomic(0u).fetch_add(w);
            device_log("warp[{}] partial={}", i / warp_lane_count(), w);
        };
    });

    // ---- 3. explicit CommandList recording + commit ----------------------
    CommandList cmd = CommandList::create();
    cmd << write_kernel().dispatch(N);
    stream << cmd.commit();

    float zero = 0.0f;
    stream << sum.copy_from(luisa::span{&zero, 1u});

    CommandList cmd2 = CommandList::create();
    cmd2 << reduce_kernel().dispatch(N);
    stream << cmd2.commit();

    // ---- 4. copy back + verify -------------------------------------------
    std::vector<float> host(N, 0.0f);
    float host_sum = 0.0f;
    stream << buf.copy_to(luisa::span{host.data(), N})
           << sum.copy_to(luisa::span{&host_sum, 1u})
           << synchronize();

    std::printf("-- checks --\n");
    bool write_ok = true;
    for (uint32_t i = 0; i < N; ++i) {
        if (std::abs(host[i] - (1.0f + static_cast<float>(i))) > 1e-6f) {
            write_ok = false;
            break;
        }
    }
    check(write_ok, "Kernel1D write: buf[i] == 1 + i");

    double expected = 0.0;
    for (uint32_t i = 0; i < N; ++i) { expected += 1.0 + static_cast<double>(i); }
    const double rel_err = std::abs(static_cast<double>(host_sum) - expected) / expected;
    std::printf("  warp/atomic sum = %.1f (expected %.1f, rel err %.3e)\n",
                static_cast<double>(host_sum), expected, rel_err);
    check(rel_err < 1e-5, "warp_active_sum + atomic_fetch_add reduction");

    std::printf("== spike %s ==\n", g_failures == 0 ? "PASSED" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
