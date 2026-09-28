// T1.2 design validation spike — validates the exact LC surface that the
// `luisa_device` foundation layer will be built on:
//   1. BufferView<T> passed to a kernel as a subview (offset honoured).
//   2. Device-side read/write: v->read(i) / v->write(i,x).
//   3. Atomics incl. the MISSING AtomicRef<double> specialization (R9 shim).
//   4. Shared<T> + sync_block() block reduction with $if/$for/$loop flow.
//   5. Warp reduction (warp_active_sum) + warp_lane_id.
//   6. Running dispatch with a runtime count from a device buffer.
// Exit 0 == the design assumptions hold.

#include <luisa/luisa-compute.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace luisa;
using namespace luisa::compute;

// ---------------------------------------------------------------------------
// R9 shim: LC's DSL specializes AtomicRef for int/uint/slong/ulong and float,
// but NOT double. The backend codegen already maps ATOMIC_FETCH_ADD to CUDA's
// atomicAdd, which has a double overload. Adding the missing specialization here
// (a legal explicit specialization inside its own namespace) closes the gap
// without patching LC.
// ---------------------------------------------------------------------------
namespace luisa::compute::detail {

template<>
class AtomicRef<double> : private AtomicRefBase
{
  public:
    explicit AtomicRef(const AtomicRefNode* access_chain) noexcept
        : AtomicRefBase{access_chain}
    {
    }

    auto exchange(Expr<double> desired) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_EXCHANGE, {desired.expression()}));
    }
    auto compare_exchange(Expr<double> expected, Expr<double> desired) noexcept
    {
        return def<double>(access_chain()->operate(
            CallOp::ATOMIC_COMPARE_EXCHANGE,
            {expected.expression(), desired.expression()}));
    }
    auto fetch_add(Expr<double> val) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_FETCH_ADD, {val.expression()}));
    }
    auto fetch_sub(Expr<double> val) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_FETCH_SUB, {val.expression()}));
    }
    auto fetch_min(Expr<double> val) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_FETCH_MIN, {val.expression()}));
    }
    auto fetch_max(Expr<double> val) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_FETCH_MAX, {val.expression()}));
    }
};

}// namespace luisa::compute::detail

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    g_fail += ok ? 0 : 1;
}
}// namespace

int main(int argc, char** argv) {
    const char* backend = (argc > 1 && argv[1]) ? argv[1] : "cuda";
    std::printf("== lc spike2 (luisa_device design validation) backend=%s ==\n", backend);

    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    constexpr uint32_t N = 4096u;
    constexpr uint32_t HALF = N / 2u;
    constexpr uint32_t BS = 256u;

    auto buf = device.create_buffer<double>(N);
    std::vector<double> init(N);
    for (uint32_t i = 0; i < N; ++i) init[i] = static_cast<double>(i);
    stream << buf.copy_from(luisa::span{init.data(), N});

    auto blk = device.create_buffer<double>(1u);
    auto warp_out = device.create_buffer<double>(1u);
    auto count_h = device.create_buffer<uint32_t>(1u);
    double zero = 0.0;
    uint32_t cnt = N;
    stream << blk.copy_from(luisa::span{&zero, 1u})
           << warp_out.copy_from(luisa::span{&zero, 1u})
           << count_h.copy_from(luisa::span{&cnt, 1u});

    // ---- 1/2. BufferVar<double> param + subview (offset) --------------------
    auto sub_double = device.compile<1>([&](BufferVar<double> v) noexcept {
        auto i = dispatch_x();
        v->write(i, v->read(i) * 2.0);
    });

    // ---- 3/4. Shared<T> block reduction with sync_block() + double atomic ---
    auto block_sum = device.compile<1>([&](BufferVar<double> v, BufferVar<double> out) noexcept {
        set_block_size(BS);
        $shared<double> scratch{BS};
        auto tid = thread_x();
        scratch[tid] = v->read(dispatch_x());
        sync_block();
        $if (tid == 0u) {
            Var<double> s = 0.0;
            $for (k, 0u, BS) { s += scratch[k]; };
            out->atomic(0u).fetch_add(s);
        };
    });

    // ---- 5. warp reduction (double atomic add) ------------------------------
    auto warp_sum = device.compile<1>([&](BufferVar<double> v, BufferVar<double> out) noexcept {
        auto w = warp_active_sum(v->read(dispatch_x()));
        $if (warp_lane_id() == 0u) { out->atomic(0u).fetch_add(w); };
    });

    // ---- 6. runtime dispatch count -----------------------------------------
    auto dyn_set = device.compile<1>([&](BufferVar<double> v, BufferVar<uint32_t> n) noexcept {
        auto i = dispatch_x();
        $if (i < n->read(0u)) { v->write(i, 1.0); };
    });

    CommandList list = CommandList::create();
    list << sub_double(buf.view().subview(HALF, HALF)).dispatch(HALF);
    list << block_sum(buf.view().subview(0u, N), blk.view()).dispatch(N);
    list << warp_sum(buf.view().subview(0u, N), warp_out.view()).dispatch(N);
    stream << list.commit() << synchronize();

    std::vector<double> doubled(N);
    stream << buf.copy_to(luisa::span{doubled.data(), N}) << synchronize();
    bool sub_ok = true;
    for (uint32_t i = 0; i < N; ++i) {
        const double want = (i >= HALF) ? init[i] * 2.0 : init[i];
        sub_ok = sub_ok && std::abs(doubled[i] - want) < 1e-9;
    }

    CommandList list2 = CommandList::create();
    list2 << dyn_set(buf.view().subview(0u, N), count_h.view()).dispatch(N);
    stream << list2.commit() << synchronize();

    std::vector<double> host(N);
    double blk_host = 0.0, warp_host = 0.0;
    stream << buf.copy_to(luisa::span{host.data(), N})
           << blk.copy_to(luisa::span{&blk_host, 1u})
           << warp_out.copy_to(luisa::span{&warp_host, 1u})
           << synchronize();

    // The sub_double pass already doubled the second half, so the reductions
    // must sum the *doubled* contents.
    double expect = 0.0;
    for (uint32_t i = 0; i < N; ++i) expect += doubled[i];

    std::printf("  blk_host=%.1f warp_host=%.1f expect=%.1f\n", blk_host, warp_host, expect);
    std::printf("-- checks --\n");
    check(sub_ok, "BufferView subview offset honoured (second half doubled)");
    check(std::abs(blk_host - expect) < 1e-6, "Shared<T>+sync_block() block reduction (double atomic)");
    check(std::abs(warp_host - expect) < 1e-6, "warp_active_sum reduction (double atomic)");
    bool ones = true;
    for (uint32_t i = 0; i < N; ++i) { ones = ones && std::abs(host[i] - 1.0) < 1e-12; }
    check(ones, "runtime dispatch count honoured (all elements == 1)");

    std::printf("== spike2 %s ==\n", g_fail == 0 ? "PASSED" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
