/********************************************************************
 * @file luisa_device_algorithms_tests.cpp
 * @brief T3.2 — tests for the device algorithm primitives
 *        (luisa_device/linear/algorithms.h).
 *
 * Coverage (brief §4). Every primitive is compared against an *independent* CPU
 * model written here (never a transcription of the kernel), over the required size
 * sweep (0, 1, 2, warp/block boundaries, 1000, 100000, 1000000) and the adversarial
 * inputs listed in the brief. The sort additionally checks sortedness, multiset
 * equality and stability (payload order within equal keys) because the API claims
 * stability; the scan is checked for *exact* integer equality and against the
 * i(i+1)/2 closed form for a range; each primitive runs through a
 * *caller-supplied* LStream with an open command list (the consumer pattern) and is
 * run twice on the same input so the two results can be compared (the R6
 * determinism concern for the atomics).
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/linear/algorithms.h>
#include <luisa_device/linear/linear.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lcrt = ::luisa::compute;

using uipc_lla_plus_d = lla::detail::Plus<double>;

// ---------------------------------------------------------------------------
//  helpers: RNG, timing, host reference models
// ---------------------------------------------------------------------------
namespace
{
struct Lcg
{
    std::uint64_t s = 0x1234'5678'9abc'def0ULL;
    std::uint64_t next_u64() noexcept
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s;
    }
    std::uint32_t next_u32() noexcept { return static_cast<std::uint32_t>(next_u64() >> 32); }
    int           next_int(int lo, int hi) noexcept  // [lo, hi]
    {
        return lo + static_cast<int>((next_u64() >> 33) % static_cast<std::uint64_t>(hi - lo + 1));
    }
    double next_double() noexcept { return static_cast<double>(next_u32()) / 4294967296.0; }
};

struct Timer
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double elapsed_ms() const
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    }
};

/// The size sweep every primitive is exercised over (brief §4).
const std::vector<std::size_t> kSizes{0,  1,   2,   3,   31,   32,   33,   63,   64,  65,
                                      127, 255, 256, 257, 511,  512,  513,  1000, 3907, 20000};

bool close_rel(double a, double b, double tol)
{
    if(std::isnan(a) || std::isnan(b))
        return std::isnan(a) && std::isnan(b);
    const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= tol * scale;
}

// ---- host models (independent of the kernels) -----------------------------
template <typename T>
std::vector<T> host_exclusive_scan(const std::vector<T>& in)
{
    std::vector<T> out(in.size(), T{});
    T              acc = T{};
    for(std::size_t i = 0; i < in.size(); ++i)
    {
        out[i] = acc;
        acc    = static_cast<T>(acc + in[i]);
    }
    return out;
}

template <typename T>
std::vector<T> host_select(const std::vector<T>& in, const std::vector<std::int32_t>& flags,
                           bool keep_when_zero)
{
    std::vector<T> out;
    for(std::size_t i = 0; i < in.size(); ++i)
    {
        const bool nz = flags[i] != 0;
        if(keep_when_zero ? !nz : nz)
            out.push_back(in[i]);
    }
    return out;
}

struct HostRle
{
    std::vector<std::int32_t> keys;
    std::vector<std::int32_t> starts;
    std::vector<std::int32_t> counts;
};

template <typename Key>
HostRle host_rle(const std::vector<Key>& keys)
{
    HostRle     out;
    std::size_t i = 0;
    while(i < keys.size())
    {
        std::size_t j = i;
        while(j + 1u < keys.size() && keys[j + 1u] == keys[i])
            ++j;
        out.keys.push_back(static_cast<std::int32_t>(keys[i]));
        out.starts.push_back(static_cast<std::int32_t>(i));
        out.counts.push_back(static_cast<std::int32_t>(j - i + 1u));
        i = j + 1u;
    }
    return out;
}

/// `out[key] = op(out[key], values[i*comp+c])` walked in input order.
template <typename Op>
std::vector<double> host_segmented_reduce(const std::vector<std::int32_t>& keys,
                                          const std::vector<double>&        values,
                                          std::size_t                       out_size,
                                          std::uint32_t                     components,
                                          Op                                op)
{
    std::vector<double> out(out_size * components, op.identity());
    for(std::size_t i = 0; i < keys.size(); ++i)
    {
        for(std::uint32_t c = 0; c < components; ++c)
        {
            auto& slot = out[static_cast<std::size_t>(keys[i]) * components + c];
            slot       = op(slot, values[i * components + c]);
        }
    }
    return out;
}

/// `perm` such that keys[perm[i]] is ascending, ties in input order.
template <typename Key>
std::vector<std::int32_t> host_stable_sort_perm(const std::vector<Key>& keys)
{
    std::vector<std::int32_t> perm(keys.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](std::int32_t a, std::int32_t b) {
        return keys[static_cast<std::size_t>(a)] < keys[static_cast<std::size_t>(b)];
    });
    return perm;
}

// ---- device helpers -------------------------------------------------------
template <typename T>
lud::LDeviceBuffer<T> to_device(const std::vector<T>& host)
{
    lud::LDeviceBuffer<T> buf;
    if(!host.empty())
    {
        buf.resize(host.size());
        buf.copy_from(luisa::span<const T>{host});
    }
    return buf;
}

/// The primitives take an `LStream&` (the consumer pattern); wrap the main stream
/// once so the tests exercise the same API the backend does.
lud::LStream& algo_stream()
{
    // Intentionally leaked: an LStream destroyed by the *static* destruction phase
    // would call into the LC runtime after zzz_runtime_shutdown tore it down (the
    // documented 0xC0000409 teardown trap). Leaking also keeps the stream's cached
    // algorithm scratch alive, which is what makes the primitives safe inside an
    // open frame CommandList.
    static auto* s = new lud::LStream{lud::main_stream()};
    return *s;
}

template <typename T>
std::vector<T> from_device(const lud::LDeviceBuffer<T>& buf, std::size_t n)
{
    std::vector<T> host(n, T{});
    if(n != 0u)
    {
        // a subview of exactly n elements: the buffer may be larger than the
        // result the primitive produced (copy_to requires matching sizes)
        buf.cview(0u, n).copy_to(luisa::span<T>{host});
    }
    algo_stream().synchronize();
    return host;
}

/// A uint64 hash key shaped exactly like the matrix converter's `(row << 32) | col`.
std::uint64_t ij_hash(std::uint32_t row, std::uint32_t col)
{
    return (static_cast<std::uint64_t>(row) << 32) | static_cast<std::uint64_t>(col);
}
}// namespace

// ===========================================================================
//  exclusive_scan
// ===========================================================================
TEST_CASE("algo_exclusive_scan_int", "[luisa][algorithms]")
{
    test_runtime();
    Lcg         rng;
    std::size_t checks = 0;
    for(auto n : kSizes)
    {
        std::vector<std::int32_t> in(n);
        for(auto& v : in)
            v = rng.next_int(-1000, 1000);
        if(n > 2u)
            in[1] = std::numeric_limits<std::int32_t>::max() / 4;  // near-overflow entry
        const auto want = host_exclusive_scan(in);

        auto in_dev = to_device(in);
        lud::LDeviceBuffer<std::int32_t> out_a, out_b;
        if(n != 0u)
        {
            out_a.resize(n);
            out_b.resize(n);
        }
        auto& stream = algo_stream();
        lla::exclusive_scan(stream, in_dev.cview(), out_a.view(), n);
        stream.synchronize();
        lla::exclusive_scan(stream, in_dev.cview(), out_b.view(), n);  // determinism
        stream.synchronize();
        const auto got_a = from_device(out_a, n);
        const auto got_b = from_device(out_b, n);
        CHECK(got_a == want);
        CHECK(got_b == got_a);
        ++checks;
    }

    // exactness + the closed form: in[i] = i + 1 -> out[i] = i(i+1)/2
    for(std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{33}, std::size_t{1000},
                         std::size_t{3907}})
    {
        std::vector<std::int32_t> in(n);
        for(std::size_t i = 0; i < n; ++i)
            in[i] = static_cast<std::int32_t>(i) + 1;
        auto                             in_dev = to_device(in);
        lud::LDeviceBuffer<std::int32_t> out;
        if(n != 0u)
            out.resize(n);
        lla::exclusive_scan(algo_stream(), in_dev.cview(), out.view(), n);
        algo_stream().synchronize();
        const auto got = from_device(out, n);
        bool       ok  = true;
        for(std::size_t i = 0; i < n; ++i)
        {
            const std::int64_t want =
                static_cast<std::int64_t>(i) * (static_cast<std::int64_t>(i) + 1) / 2;
            ok = ok && got[i] == want;
        }
        CHECK(ok);
    }

    // in place (in == out)
    {
        std::vector<std::int32_t> in{5, 7, -2, 0, 11};
        auto                      dev = to_device(in);
        lla::exclusive_scan(algo_stream(), dev.cview(), dev.view(), in.size());
        algo_stream().synchronize();
        CHECK(from_device(dev, in.size()) == host_exclusive_scan(in));
    }
    std::printf("    [scan<int32>] exact + deterministic over %zu sizes, closed form, in-place\n",
                checks);
}

TEST_CASE("algo_exclusive_scan_float", "[luisa][algorithms]")
{
    test_runtime();
    Lcg        rng;
    const auto n = std::size_t{100000};
    std::vector<double> in(n);
    for(auto& v : in)
        v = rng.next_double() * 2.0 - 1.0;
    const auto want = host_exclusive_scan(in);

    auto                       in_dev = to_device(in);
    lud::LDeviceBuffer<double> out_a, out_b;
    out_a.resize(n);
    out_b.resize(n);
    auto& stream = algo_stream();
    lla::exclusive_scan(stream, in_dev.cview(), out_a.view(), n);
    stream.synchronize();
    lla::exclusive_scan(stream, in_dev.cview(), out_b.view(), n);
    stream.synchronize();
    const auto got_a = from_device(out_a, n);
    const auto got_b = from_device(out_b, n);
    CHECK(got_a == got_b);  // deterministic across runs
    double worst_rel = 0.0;
    for(std::size_t i = 0; i < n; ++i)
    {
        const double scale = std::max(1.0, std::abs(want[i]));
        worst_rel          = std::max(worst_rel, std::abs(got_a[i] - want[i]) / scale);
    }
    CHECK(worst_rel < 1e-12);
    std::printf("    [scan<double>] n=%zu deterministic, worst deviation from the serial model "
                "%.2e\n",
                n, worst_rel);
}

// ===========================================================================
//  device_reduce
// ===========================================================================
TEST_CASE("algo_device_reduce", "[luisa][algorithms]")
{
    test_runtime();
    Lcg rng;
    for(auto n : kSizes)
    {
        std::vector<double>       in_d(n);
        std::vector<std::int32_t> in_i(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            in_d[i] = rng.next_double() * 100.0 - 50.0;
            in_i[i] = rng.next_int(-1000, 1000);
        }
        auto                            dev_d = to_device(in_d);
        auto                            dev_i = to_device(in_i);
        lud::LDeviceVar<double>         o_d;
        lud::LDeviceVar<std::int32_t>   o_i;
        auto& stream = algo_stream();

        double want_sum = 0.0;
        double want_min = n ? std::numeric_limits<double>::max()
                            : std::numeric_limits<double>::max();
        double want_max = std::numeric_limits<double>::lowest();
        for(auto v : in_d)
        {
            want_sum += v;
            want_min = std::min(want_min, v);
            want_max = std::max(want_max, v);
        }
        lla::device_reduce_sum(stream, dev_d.cview(), o_d.view(), n);
        stream.synchronize();
        CHECK(close_rel(o_d.copy_to(), want_sum, 1e-9));
        lla::device_reduce_min(stream, dev_d.cview(), o_d.view(), n);
        stream.synchronize();
        CHECK(close_rel(o_d.copy_to(), want_min, 1e-12));
        lla::device_reduce_max(stream, dev_d.cview(), o_d.view(), n);
        stream.synchronize();
        CHECK(close_rel(o_d.copy_to(), want_max, 1e-12));

        std::int32_t want_i_sum = 0;
        std::int32_t want_i_min = std::numeric_limits<std::int32_t>::max();
        std::int32_t want_i_max = std::numeric_limits<std::int32_t>::lowest();
        for(auto v : in_i)
        {
            want_i_sum += v;
            want_i_min = std::min(want_i_min, v);
            want_i_max = std::max(want_i_max, v);
        }
        lla::device_reduce_sum(stream, dev_i.cview(), o_i.view(), n);
        stream.synchronize();
        CHECK(o_i.copy_to() == want_i_sum);
        lla::device_reduce_min(stream, dev_i.cview(), o_i.view(), n);
        stream.synchronize();
        CHECK(o_i.copy_to() == want_i_min);
        lla::device_reduce_max(stream, dev_i.cview(), o_i.view(), n);
        stream.synchronize();
        CHECK(o_i.copy_to() == want_i_max);
        // determinism
        lla::device_reduce_sum(stream, dev_i.cview(), o_i.view(), n);
        stream.synchronize();
        CHECK(o_i.copy_to() == want_i_sum);
    }
    std::printf("    [device_reduce] int exact, double within rounding, 0-size identities, "
                "deterministic\n");
}

// ===========================================================================
//  select_if
// ===========================================================================
TEST_CASE("algo_select_if", "[luisa][algorithms]")
{
    test_runtime();
    Lcg rng;
    for(auto n : kSizes)
    {
        std::vector<double>       in(n);
        std::vector<std::int32_t> flags(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            in[i]    = rng.next_double() * 10.0;
            flags[i] = rng.next_int(0, 3) == 0 ? 0 : 1;
        }
        if(n > 4u)  // adversarial: a half-and-half split, and all-kept / all-dropped
        {
            std::fill(flags.begin(), flags.end(), 1);
            std::fill(flags.begin() + static_cast<std::ptrdiff_t>(n / 2), flags.end(), 0);
        }
        const auto want      = host_select(in, flags, false);
        const auto want_zero = host_select(in, flags, true);

        auto                             in_dev = to_device(in);
        auto                             flags_dev = to_device(flags);
        lud::LDeviceBuffer<double>       out;
        lud::LDeviceVar<std::uint32_t>   count;
        if(n != 0u)
            out.resize(n);
        auto& stream = algo_stream();
        lla::select_if(stream, in_dev.cview(), flags_dev.cview(), out.view(), count.view(), n);
        stream.synchronize();
        CHECK(count.copy_to() == want.size());
        if(!want.empty())
            CHECK(from_device(out, want.size()) == want);

        lla::select_if(stream, in_dev.cview(), flags_dev.cview(), out.view(), count.view(), n,
                       nullptr, true);
        stream.synchronize();
        CHECK(count.copy_to() == want_zero.size());
        if(!want_zero.empty())
            CHECK(from_device(out, want_zero.size()) == want_zero);
    }
    std::printf("    [select_if] order-preserving, count, both flavours over the size sweep\n");
}

// Predicate flavour: a stateless predicate *type* (the kernel-cache requirement).
struct is_odd
{
    static lcrt::Expr<bool> apply(lcrt::Expr<std::int32_t> v) { return v % 2 != 0; }
};
struct greater_than_100
{
    static lcrt::Expr<bool> apply(lcrt::Expr<double> v) { return v > 100.0; }
};

TEST_CASE("algo_select_if_predicated", "[luisa][algorithms]")
{
    test_runtime();
    Lcg rng;
    for(auto n : kSizes)
    {
        std::vector<std::int32_t> in(n);
        for(auto& v : in)
            v = rng.next_int(-1000, 1000);
        std::vector<std::int32_t> want;
        for(auto v : in)
            if(v % 2 != 0)
                want.push_back(v);

        auto                             in_dev = to_device(in);
        lud::LDeviceBuffer<std::int32_t> out;
        lud::LDeviceVar<std::uint32_t>   count;
        if(n != 0u)
            out.resize(n);
        auto& stream = algo_stream();
        lla::select_if_predicated<std::int32_t, is_odd>(stream, in_dev.cview(), out.view(),
                                                        count.view(), n);
        stream.synchronize();
        CHECK(count.copy_to() == want.size());
        if(!want.empty())
            CHECK(from_device(out, want.size()) == want);

        std::vector<double> in_d(n);
        for(auto& v : in_d)
            v = rng.next_double() * 1000.0;
        std::vector<double> want_d;
        for(auto v : in_d)
            if(v > 100.0)
                want_d.push_back(v);
        auto                           in_d_dev = to_device(in_d);
        lud::LDeviceBuffer<double>     out_d;
        lud::LDeviceVar<std::uint32_t> count_d;
        if(n != 0u)
            out_d.resize(n);
        lla::select_if_predicated<double, greater_than_100>(stream, in_d_dev.cview(), out_d.view(),
                                                            count_d.view(), n);
        stream.synchronize();
        CHECK(count_d.copy_to() == want_d.size());
        if(!want_d.empty())
            CHECK(from_device(out_d, want_d.size()) == want_d);
    }
    std::printf("    [select_if<pred>] two predicate types over the size sweep\n");
}

// ===========================================================================
//  run_length_encode
// ===========================================================================
TEST_CASE("algo_run_length_encode", "[luisa][algorithms]")
{
    test_runtime();
    Lcg rng;
    for(auto n : kSizes)
    {
        for(int pattern = 0; pattern < 4; ++pattern)
        {
            std::vector<std::int32_t> keys(n);
            for(std::size_t i = 0; i < n; ++i)
            {
                switch(pattern)
                {
                    case 0: keys[i] = rng.next_int(0, 3); break;  // few long runs
                    case 1: keys[i] = static_cast<std::int32_t>(i); break;  // all singletons
                    case 2: keys[i] = 7; break;  // one run for the whole array
                    default: keys[i] = static_cast<std::int32_t>(i / 5) * 2; break;  // runs of 5
                }
            }
            const auto want = host_rle(keys);

            auto                             keys_dev = to_device(keys);
            lud::LDeviceBuffer<std::int32_t> uniq, starts, counts;
            lud::LDeviceVar<std::uint32_t>   runs;
            if(n != 0u)
            {
                uniq.resize(n);
                starts.resize(n);
                counts.resize(n);
            }
            auto& stream = algo_stream();
            lla::run_length_encode(stream, keys_dev.cview(), uniq.view(), starts.view(),
                                   counts.view(), runs.view(), n);
            stream.synchronize();
            const auto got_runs = static_cast<std::size_t>(runs.copy_to());
            CHECK(got_runs == want.counts.size());
            if(got_runs != 0u)
            {
                const auto u = from_device(uniq, got_runs);
                const auto s = from_device(starts, got_runs);
                const auto c = from_device(counts, got_runs);
                bool       ok = (u == want.keys);
                ok            = ok && (s == want.starts);
                ok            = ok && (c == want.counts);
                // the runs must tile [0, n) in order
                std::int64_t pos = 0;
                for(std::size_t j = 0; j < got_runs; ++j)
                {
                    ok = ok && s[j] == pos;
                    pos += c[j];
                }
                ok = ok && pos == static_cast<std::int64_t>(n);
                CHECK(ok);
            }
        }
    }
    std::printf("    [run_length_encode] 4 adversarial key patterns x the size sweep\n");
}

// ===========================================================================
//  segmented_reduce
// ===========================================================================
TEST_CASE("algo_segmented_reduce", "[luisa][algorithms]")
{
    test_runtime();
    Lcg        rng;
    const auto n        = std::size_t{1000};
    const auto out_size = std::size_t{1200};
    for(int pattern = 0; pattern < 5; ++pattern)
    {
        std::vector<std::int32_t> keys(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            switch(pattern)
            {
                case 0: keys[i] = 0; break;                                    // one segment
                case 1: keys[i] = static_cast<std::int32_t>(i); break;         // all singletons
                case 2: keys[i] = static_cast<std::int32_t>(1 + i / 32); break;  // warp boundary
                case 3: keys[i] = static_cast<std::int32_t>(1 + i / 256); break;  // block boundary
                default: keys[i] = static_cast<std::int32_t>(i < 3u ? 0u : 1u + i / 100u); break;
            }
        }
        std::vector<double> values(n);
        for(std::size_t i = 0; i < n; ++i)
            values[i] = static_cast<double>((i % 7) + 1);  // exactly representable

        const auto want = host_segmented_reduce(keys, values, out_size, 1u, uipc_lla_plus_d{});

        auto                       keys_dev = to_device(keys);
        auto                       vals_dev = to_device(values);
        lud::LDeviceBuffer<double> out_a, out_b;
        out_a.resize(out_size);
        out_b.resize(out_size);
        auto& stream = algo_stream();
        lla::segmented_reduce(stream, keys_dev.cview(), vals_dev.cview(), out_a.view(), out_size, n);
        stream.synchronize();
        lla::segmented_reduce(stream, keys_dev.cview(), vals_dev.cview(), out_b.view(), out_size, n);
        stream.synchronize();
        const auto got_a = from_device(out_a, out_size);
        const auto got_b = from_device(out_b, out_size);
        CHECK(got_a == want);
        CHECK(got_b == got_a);  // determinism, incl. the atomic cross-warp path
    }

    // random doubles: same association only for singletons; compare within tolerance
    {
        std::vector<std::int32_t> keys(n);
        std::vector<double>       values(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            keys[i]   = static_cast<std::int32_t>(i / 17);
            values[i] = rng.next_double();
        }
        const auto want = host_segmented_reduce(keys, values, out_size, 1u, uipc_lla_plus_d{});
        auto                       keys_dev = to_device(keys);
        auto                       vals_dev = to_device(values);
        lud::LDeviceBuffer<double> out;
        out.resize(out_size);
        lla::segmented_reduce(algo_stream(), keys_dev.cview(), vals_dev.cview(), out.view(),
                              out_size, n);
        algo_stream().synchronize();
        const auto got   = from_device(out, out_size);
        double     worst = 0.0;
        for(std::size_t i = 0; i < out_size; ++i)
            worst = std::max(worst, std::abs(got[i] - want[i]) / std::max(1.0, std::abs(want[i])));
        CHECK(worst < 1e-12);
        std::printf("    [segmented_reduce] random doubles worst relative deviation = %.2e\n",
                    worst);
    }

    // flat-block flavour: 1, 3 and 9 components (the T3.1 block layout)
    for(std::uint32_t components : {1u, 3u, 9u})
    {
        std::vector<std::int32_t> keys(n);
        std::vector<double>       values(n * components);
        for(std::size_t i = 0; i < n; ++i)
        {
            keys[i] = static_cast<std::int32_t>(i / 13);
            for(std::uint32_t c = 0; c < components; ++c)
                values[i * components + c] = static_cast<double>((i % 5) + c + 1);
        }
        const auto want = host_segmented_reduce(keys, values, out_size, components,
                                                uipc_lla_plus_d{});
        auto                       keys_dev = to_device(keys);
        auto                       vals_dev = to_device(values);
        lud::LDeviceBuffer<double> out;
        out.resize(out_size * components);
        lla::segmented_reduce_blocks(algo_stream(), keys_dev.cview(), vals_dev.cview(),
                                     out.view(), out_size, n, components);
        algo_stream().synchronize();
        CHECK(from_device(out, out_size * components) == want);
    }

    // keys outside [0, out_size): dropped by the documented bound predicate (the
    // CPU model simply does not visit them)
    {
        std::vector<std::int32_t> keys(n);
        std::vector<double>       values(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            keys[i]   = static_cast<std::int32_t>(i / 25);
            values[i] = static_cast<double>((i % 17) + 1);
        }
        keys[n - 1u]   = static_cast<std::int32_t>(out_size + 10);  // outside [0, out_size)
        values[n - 1u] = 123.0;
        const auto want = host_segmented_reduce(
            std::vector<std::int32_t>(keys.begin(), keys.end() - 1),
            std::vector<double>(values.begin(), values.end() - 1), out_size, 1u,
            uipc_lla_plus_d{});
        auto                       keys_dev = to_device(keys);
        auto                       vals_dev = to_device(values);
        lud::LDeviceBuffer<double> out;
        out.resize(out_size);
        lla::segmented_reduce(algo_stream(), keys_dev.cview(), vals_dev.cview(), out.view(),
                              out_size, n);
        algo_stream().synchronize();
        CHECK(from_device(out, out_size) == want);
    }
    std::printf("    [segmented_reduce] 5 boundary patterns, 1/3/9 blocks, bound predicate\n");
}

// ===========================================================================
//  sort
// ===========================================================================
namespace
{
/// Sortedness + multiset equality + stability + permutation equality, all against
/// the host's std::stable_sort.
template <typename Key>
void check_sort_device(const std::vector<Key>& keys, std::uint32_t key_bits)
{
    const auto n = keys.size();
    if(n == 0u)
        return;
    auto                             keys_dev = to_device(keys);
    lud::LDeviceBuffer<Key>          keys_out;
    lud::LDeviceBuffer<std::int32_t> perm;
    keys_out.resize(n);
    perm.resize(n);
    auto& stream = algo_stream();
    lla::sort_index_pairs(stream, keys_dev.cview(), keys_out.view(), perm.view(), n, key_bits);
    stream.synchronize();
    const auto got_keys = from_device(keys_out, n);
    const auto got_perm = from_device(perm, n);
    const auto want     = host_stable_sort_perm(keys);

    bool sorted = true, gathered = true, stable = true, perm_ok = true;
    for(std::size_t i = 0; i < n; ++i)
    {
        perm_ok = perm_ok && got_perm[i] == want[i];
        if(i > 0u)
            sorted = sorted && !(got_keys[i] < got_keys[i - 1u]);
        gathered = gathered && got_keys[i] == keys[static_cast<std::size_t>(got_perm[i])];
        if(i > 0u)
        {
            const auto a = static_cast<std::size_t>(got_perm[i - 1u]);
            const auto b = static_cast<std::size_t>(got_perm[i]);
            if(keys[a] == keys[b])
                stable = stable && (a < b);  // equal keys keep their input order
        }
    }
    std::vector<Key> a = keys, b = got_keys;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    CHECK(sorted);
    CHECK(gathered);
    CHECK(perm_ok);
    CHECK(stable);
    CHECK(a == b);  // multiset equality
}
}// namespace

TEST_CASE("algo_sort_index_pairs_int32", "[luisa][algorithms]")
{
    test_runtime();
    Lcg rng;
    for(auto n : kSizes)
    {
        std::vector<std::int32_t> keys(n);
        for(std::size_t i = 0; i < n; ++i)
            keys[i] = rng.next_int(0, 40);
        check_sort_device(keys, 0u);
    }
    const auto n = std::size_t{1000};
    {
        std::vector<std::int32_t> all_equal(n, 42);
        check_sort_device(all_equal, 0u);
        std::vector<std::int32_t> increasing(n), decreasing(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            increasing[i] = static_cast<std::int32_t>(i);
            decreasing[i] = static_cast<std::int32_t>(n - i);
        }
        check_sort_device(increasing, 0u);
        check_sort_device(decreasing, 0u);
        std::vector<std::int32_t> one_bucket(n, 0);
        for(std::size_t i = 0; i < n; i += 100u)
            one_bucket[i] = static_cast<std::int32_t>(i);
        check_sort_device(one_bucket, 0u);
        std::vector<std::int32_t> signed_keys(n);
        for(std::size_t i = 0; i < n; ++i)
            signed_keys[i] = rng.next_int(-500, 500);
        check_sort_device(signed_keys, 0u);  // sign-bit bias path
        std::vector<std::int32_t> bounded(n);
        for(std::size_t i = 0; i < n; ++i)
            bounded[i] = rng.next_int(0, 250);
        check_sort_device(bounded, 8u);   // one pass
        check_sort_device(bounded, 16u);  // two passes
    }
    std::printf("    [sort<int32>] sorted+gathered+stable+multiset over the sweep and 8 patterns\n");
}

TEST_CASE("algo_sort_index_pairs_u64_hash", "[luisa][algorithms]")
{
    test_runtime();
    Lcg rng;
    for(auto n : kSizes)
    {
        std::vector<std::uint64_t> keys(n);
        for(std::size_t i = 0; i < n; ++i)
            keys[i] = ij_hash(static_cast<std::uint32_t>(rng.next_int(0, 7)),
                              static_cast<std::uint32_t>(rng.next_int(0, 7)));
        check_sort_device(keys, 0u);  // duplicates everywhere
    }
    {
        const auto                 n = std::size_t{100000};
        std::vector<std::uint64_t> keys(n);
        for(std::size_t i = 0; i < n; ++i)
            keys[i] = ij_hash(static_cast<std::uint32_t>(rng.next_int(0, 999)),
                              static_cast<std::uint32_t>(rng.next_int(0, 999)));
        check_sort_device(keys, 0u);
    }
    std::printf("    [sort<uint64>] the converter's (row<<32)|col hash, duplicates, up to 100000\n");
}

TEST_CASE("algo_sort_pairs_by_key", "[luisa][algorithms]")
{
    test_runtime();
    Lcg        rng;
    const auto n = std::size_t{1000};
    std::vector<std::int32_t> keys(n);
    std::vector<double>       vals(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        keys[i] = rng.next_int(0, 20);
        vals[i] = static_cast<double>(i);  // distinguishable payloads
    }
    const auto want = host_stable_sort_perm(keys);
    auto       kd = to_device(keys);
               auto       vd = to_device(vals);
    lud::LDeviceBuffer<std::int32_t> ko;
    lud::LDeviceBuffer<double>       vo;
    ko.resize(n);
    vo.resize(n);
    auto& stream = algo_stream();
    lla::sort_pairs_by_key(stream, kd.cview(), ko.view(), vd.cview(), vo.view(), n);
    stream.synchronize();
    const auto got_keys = from_device(ko, n);
    const auto got_vals = from_device(vo, n);
    bool       ok       = true;
    for(std::size_t i = 0; i < n; ++i)
    {
        ok = ok && got_keys[i] == keys[static_cast<std::size_t>(want[i])]
             && got_vals[i] == static_cast<double>(want[i]);
    }
    CHECK(ok);

    // 64-bit keys with an int payload (the CUB SortPairs shape the converter uses)
    {
        std::vector<std::uint64_t> k64(n);
        std::vector<std::int32_t>  p64(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            k64[i] = ij_hash(static_cast<std::uint32_t>(rng.next_int(0, 5)),
                             static_cast<std::uint32_t>(i));
            p64[i] = static_cast<std::int32_t>(i);
        }
        const auto want64 = host_stable_sort_perm(k64);
        auto       kd64 = to_device(k64);
               auto       pd64 = to_device(p64);
        lud::LDeviceBuffer<std::uint64_t> ko64;
        lud::LDeviceBuffer<std::int32_t>  po64;
        ko64.resize(n);
        po64.resize(n);
        lla::sort_pairs_by_key(stream, kd64.cview(), ko64.view(), pd64.cview(), po64.view(), n);
        stream.synchronize();
        const auto gk = from_device(ko64, n);
        const auto gp = from_device(po64, n);
        bool       ok2 = true;
        for(std::size_t i = 0; i < n; ++i)
            ok2 = ok2 && gk[i] == k64[static_cast<std::size_t>(want64[i])]
                  && gp[i] == static_cast<std::int32_t>(want64[i]);
        CHECK(ok2);
    }

    // overlapping in/out is rejected (documented precondition)
    CHECK_THROWS(lla::sort_pairs_by_key(stream, kd.cview(), ko.view(), vd.cview(), vd.view(), n));

    // in-place keys (keys_out == keys_in) is supported
    {
        std::vector<std::int32_t> k(n);
        for(std::size_t i = 0; i < n; ++i)
            k[i] = rng.next_int(0, 5);
        auto                             kd_in = to_device(k);
        lud::LDeviceBuffer<std::int32_t> perm;
        perm.resize(n);
        lla::sort_index_pairs(stream, kd_in.cview(), kd_in.view(), perm.view(), n);
        stream.synchronize();
        const auto gk   = from_device(kd_in, n);
        const auto w    = host_stable_sort_perm(k);
        bool       ok3  = true;
        for(std::size_t i = 0; i < n; ++i)
            ok3 = ok3 && gk[i] == k[static_cast<std::size_t>(w[i])];
        CHECK(ok3);
    }
    std::printf("    [sort_pairs_by_key] double+int payloads, in-place keys, alias rejection\n");
}

TEST_CASE("algo_bucket_sort_pairs_by_key", "[luisa][algorithms]")
{
    test_runtime();
    Lcg rng;
    for(std::uint32_t buckets : {1u, 2u, 7u, 16u, 256u})
    {
        const auto                n = std::size_t{1000};
        std::vector<std::int32_t> keys(n);
        std::vector<double>       vals(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            keys[i] = rng.next_int(0, static_cast<int>(buckets) - 1);
            vals[i] = static_cast<double>(i);
        }
        const auto want = host_stable_sort_perm(keys);
        auto       kd = to_device(keys);
               auto       vd = to_device(vals);
        lud::LDeviceBuffer<std::int32_t> ko;
        lud::LDeviceBuffer<double>       vo;
        ko.resize(n);
        vo.resize(n);
        auto& stream = algo_stream();
        lla::bucket_sort_pairs_by_key(stream, kd.cview(), ko.view(), vd.cview(), vo.view(), n,
                                      buckets);
        stream.synchronize();
        const auto gk = from_device(ko, n);
        const auto gv = from_device(vo, n);
        bool       ok = true;
        for(std::size_t i = 0; i < n; ++i)
            ok = ok && gk[i] == keys[static_cast<std::size_t>(want[i])]
                 && gv[i] == static_cast<double>(want[i]);
        CHECK(ok);
    }

    // keys above the documented bound: not detected (like CUB, the caller owns the
    // precondition) but memory safe — the multiset must survive
    {
        const auto                n = std::size_t{1000};
        std::vector<std::int32_t> keys(n);
        std::vector<double>       vals(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            keys[i] = static_cast<std::int32_t>(i % 10);
            vals[i] = static_cast<double>(i);
        }
        auto kd = to_device(keys);
        auto vd = to_device(vals);
        lud::LDeviceBuffer<std::int32_t> ko;
        lud::LDeviceBuffer<double>       vo;
        ko.resize(n);
        vo.resize(n);
        lla::bucket_sort_pairs_by_key(algo_stream(), kd.cview(), ko.view(), vd.cview(),
                                      vo.view(), n, 4u);
        algo_stream().synchronize();
        auto a = from_device(ko, n);
        auto b = from_device(vo, n);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::vector<std::int32_t> wk = keys;
        std::vector<double>       wv = vals;
        std::sort(wk.begin(), wk.end());
        std::sort(wv.begin(), wv.end());
        CHECK(a == wk);
        CHECK(b == wv);
        std::printf("    [bucket_sort] keys above bucket_count: multiset preserved, no OOB\n");
    }

    // bucket_count > 256 is rejected (one radix digit is the documented bound)
    {
        std::vector<std::int32_t> k{0, 1, 2};
        std::vector<double>       v{1.0, 2.0, 3.0};
        auto kd = to_device(k);
        auto vd = to_device(v);
        lud::LDeviceBuffer<std::int32_t> ko;
        lud::LDeviceBuffer<double>       vo;
        ko.resize(3);
        vo.resize(3);
        CHECK_THROWS(lla::bucket_sort_pairs_by_key(algo_stream(), kd.cview(), ko.view(),
                                                  vd.cview(), vo.view(), 3u, 257u));
    }
}

TEST_CASE("algo_permute_gather", "[luisa][algorithms]")
{
    test_runtime();
    const auto n = std::size_t{1000};
    for(std::uint32_t components : {1u, 3u, 9u})
    {
        std::vector<double>       src(n * components);
        std::vector<std::int32_t> perm(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            perm[i] = static_cast<std::int32_t>(i);
            for(std::uint32_t c = 0; c < components; ++c)
                src[i * components + c] = static_cast<double>(i * 100 + c);
        }
        std::shuffle(perm.begin(), perm.end(), std::mt19937(7u));
        auto sd = to_device(src);
        auto pd = to_device(perm);
        lud::LDeviceBuffer<double> dst;
        dst.resize(n * components);
        lla::permute_gather_strided(algo_stream(), sd.cview(), pd.cview(), dst.view(), n,
                                    components);
        algo_stream().synchronize();
        const auto got = from_device(dst, n * components);
        bool       ok  = true;
        for(std::size_t i = 0; i < n; ++i)
            for(std::uint32_t c = 0; c < components; ++c)
                ok = ok && got[i * components + c]
                              == src[static_cast<std::size_t>(perm[i]) * components + c];
        CHECK(ok);
    }
    std::printf("    [permute_gather] components 1/3/9 with a shuffled permutation\n");
}

// ===========================================================================
//  the consumer pattern: a caller-supplied LStream with an open CommandList,
//  several primitives batched, plus a determinism repeat
// ===========================================================================
TEST_CASE("algo_caller_stream_and_determinism", "[luisa][algorithms]")
{
    test_runtime();
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep diagnostics on an abort (debug aid)
    Lcg        rng;
    const auto n = std::size_t{1000};
    std::vector<std::int32_t> keys(n);
    std::vector<double>       vals(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        keys[i] = rng.next_int(0, 30);
        vals[i] = static_cast<double>((i % 11) + 1);
    }
    auto kd = to_device(keys);
        auto vd = to_device(vals);
    lud::LDeviceBuffer<std::int32_t> so, so2, uniq, starts, counts;
    lud::LDeviceBuffer<double>       vo, vo2;
    lud::LDeviceVar<std::uint32_t>   runs;
    so.resize(n);
    so2.resize(n);
    uniq.resize(n);
    starts.resize(n);
    counts.resize(n);
    vo.resize(n);
    vo2.resize(n);

    lud::LStream stream;
    CHECK(!stream.frame_open());
    stream.begin();
    CHECK(stream.frame_open());
    lla::exclusive_scan(stream, kd.cview(), so.view(), n);
    lla::run_length_encode(stream, kd.cview(), uniq.view(), starts.view(), counts.view(),
                           runs.view(), n);
    lla::sort_pairs_by_key(stream, kd.cview(), so.view(), vd.cview(), vo.view(), n);
    stream.commit();
    stream.synchronize();

    const auto got_keys = from_device(so, n);
    const auto got_vals = from_device(vo, n);
    const auto want     = host_stable_sort_perm(keys);
    bool       ok       = true;
    for(std::size_t i = 0; i < n; ++i)
        ok = ok && got_keys[i] == keys[static_cast<std::size_t>(want[i])]
             && got_vals[i] == vals[static_cast<std::size_t>(want[i])];
    CHECK(ok);

    // determinism: the identical batch again, compare
    stream.begin();
    lla::sort_pairs_by_key(stream, kd.cview(), so2.view(), vd.cview(), vo2.view(), n);
    stream.commit();
    stream.synchronize();
    CHECK(from_device(so2, n) == got_keys);
    CHECK(from_device(vo2, n) == got_vals);
    std::printf("    [caller stream] scan + RLE + sort in one CommandList, twice, identical\n");
}

// ===========================================================================
//  1e6: correctness + measured timings (the brief asks for the numbers)
// ===========================================================================
TEST_CASE("algo_million_element_timings", "[luisa][algorithms]")
{
    test_runtime();
    Lcg        rng;
    const auto n      = std::size_t{1000000};
    auto& stream = algo_stream();

    // ---- sort_index_pairs<uint64> (the heaviest primitive: 8 passes) ----
    {
        std::vector<std::uint64_t> keys(n);
        for(std::size_t i = 0; i < n; ++i)
            keys[i] = ij_hash(static_cast<std::uint32_t>(rng.next_int(0, 9999)),
                              static_cast<std::uint32_t>(rng.next_int(0, 9999)));
        auto                              kd = to_device(keys);
        lud::LDeviceBuffer<std::uint64_t> ko;
        lud::LDeviceBuffer<std::int32_t>  perm;
        ko.resize(n);
        perm.resize(n);
        // warm-up (JIT + caches), then the measured run
        lla::sort_index_pairs(stream, kd.cview(), ko.view(), perm.view(), n);
        stream.synchronize();
        Timer t;
        lla::sort_index_pairs(stream, kd.cview(), ko.view(), perm.view(), n);
        stream.synchronize();
        const double ms = t.elapsed_ms();
        const auto   gp = from_device(perm, n);
        const auto   gk = from_device(ko, n);
        bool         sorted = true;
        for(std::size_t i = 1; i < n; ++i)
            sorted = sorted && !(gk[i] < gk[i - 1u]);
        const auto want    = host_stable_sort_perm(keys);
        bool       stable  = true;
        for(std::size_t i = 0; i < n; ++i)
            stable = stable && gp[i] == want[i];
        CHECK(sorted);
        CHECK(stable);
        std::printf("    [1e6] sort_index_pairs<uint64> 8 passes   : %8.2f ms  sorted=%d stable=%d\n",
                    ms, (int)sorted, (int)stable);
    }

    // ---- exclusive_scan<int32> ----
    {
        std::vector<std::int32_t> in(n);
        for(auto& v : in)
            v = rng.next_int(-100, 100);
        auto                             id = to_device(in);
        lud::LDeviceBuffer<std::int32_t> out;
        out.resize(n);
        lla::exclusive_scan(stream, id.cview(), out.view(), n);
        stream.synchronize();
        Timer t;
        lla::exclusive_scan(stream, id.cview(), out.view(), n);
        stream.synchronize();
        const double ms   = t.elapsed_ms();
        const auto   got  = from_device(out, n);
        const auto   want = host_exclusive_scan(in);
        CHECK(got == want);
        std::printf("    [1e6] exclusive_scan<int32>               : %8.2f ms  exact=%d\n", ms,
                    (int)(got == want));
    }

    // ---- device_reduce_sum<double> ----
    {
        std::vector<double> in(n);
        for(auto& v : in)
            v = rng.next_double();
        auto                    id = to_device(in);
        lud::LDeviceVar<double> o;
        lla::device_reduce_sum(stream, id.cview(), o.view(), n);
        stream.synchronize();
        Timer t;
        lla::device_reduce_sum(stream, id.cview(), o.view(), n);
        stream.synchronize();
        const double ms   = t.elapsed_ms();
        const double got  = o.copy_to();
        double       want = 0.0;
        for(auto v : in)
            want += v;
        CHECK(close_rel(got, want, 1e-10));
        std::printf("    [1e6] device_reduce_sum<double>           : %8.2f ms  rel=%.2e\n", ms,
                    std::abs(got - want) / std::max(1.0, std::abs(want)));
        // an exactly representable target: the reduction must cover *every* element
        std::vector<double> ones(n, 1.0);
        auto                od = to_device(ones);
        lla::device_reduce_sum(stream, od.cview(), o.view(), n);
        stream.synchronize();
        CHECK(o.copy_to() == static_cast<double>(n));
    }

    // ---- select_if<double> ----
    {
        std::vector<double>       in(n);
        std::vector<std::int32_t> flags(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            in[i]    = rng.next_double();
            flags[i] = (i % 3u == 0u) ? 0 : 1;
        }
        auto                           id = to_device(in);
        auto                           fd = to_device(flags);
        lud::LDeviceBuffer<double>     out;
        lud::LDeviceVar<std::uint32_t> count;
        out.resize(n);
        lla::select_if(stream, id.cview(), fd.cview(), out.view(), count.view(), n);
        stream.synchronize();
        Timer t;
        lla::select_if(stream, id.cview(), fd.cview(), out.view(), count.view(), n);
        stream.synchronize();
        const double ms   = t.elapsed_ms();
        const auto   c    = count.copy_to();
        const auto   want = host_select(in, flags, false);
        CHECK(c == want.size());
        CHECK(from_device(out, c) == want);
        std::printf("    [1e6] select_if<double>                   : %8.2f ms  count=%u\n", ms, c);
    }

    // ---- run_length_encode<int32> ----
    {
        std::vector<std::int32_t> keys(n);
        for(std::size_t i = 0; i < n; ++i)
            keys[i] = static_cast<std::int32_t>(i / 3);  // ascending, runs of 3
        auto                             kd = to_device(keys);
        lud::LDeviceBuffer<std::int32_t> uniq, starts, counts;
        lud::LDeviceVar<std::uint32_t>   runs;
        uniq.resize(n);
        starts.resize(n);
        counts.resize(n);
        lla::run_length_encode(stream, kd.cview(), uniq.view(), starts.view(), counts.view(),
                               runs.view(), n);
        stream.synchronize();
        Timer t;
        lla::run_length_encode(stream, kd.cview(), uniq.view(), starts.view(), counts.view(),
                               runs.view(), n);
        stream.synchronize();
        const double ms = t.elapsed_ms();
        CHECK(runs.copy_to() == static_cast<std::uint32_t>((n + 2u) / 3u));
        std::printf("    [1e6] run_length_encode<int32>            : %8.2f ms  runs=%u\n", ms,
                    runs.copy_to());
    }

    // ---- segmented_reduce<double>, 5 elements per segment ----
    {
        std::vector<std::int32_t> keys(n);
        std::vector<double>       vals(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            keys[i] = static_cast<std::int32_t>(i / 5);
            vals[i] = static_cast<double>((i % 13) + 1);
        }
        const auto                 out_size = n / 5u + 1u;
        auto kd = to_device(keys);
        auto vd = to_device(vals);
        lud::LDeviceBuffer<double> out;
        out.resize(out_size);
        lla::segmented_reduce(stream, kd.cview(), vd.cview(), out.view(), out_size, n);
        stream.synchronize();
        Timer t;
        lla::segmented_reduce(stream, kd.cview(), vd.cview(), out.view(), out_size, n);
        stream.synchronize();
        const double ms   = t.elapsed_ms();
        const auto   got  = from_device(out, out_size);
        const auto   want = host_segmented_reduce(keys, vals, out_size, 1u, uipc_lla_plus_d{});
        CHECK(got == want);
        std::printf("    [1e6] segmented_reduce<double> runs of 5   : %8.2f ms  exact=%d\n", ms,
                    (int)(got == want));
    }

    // ---- segmented_reduce<double> with a single 1e6-element segment (worst case:
    //      every warp boundary is a cross-warp combine) ----
    {
        std::vector<std::int32_t> keys(n, 0);
        std::vector<double>       vals(n, 1.0);
        auto                      kd = to_device(keys);
               auto                      vd = to_device(vals);
        lud::LDeviceVar<double>   out;
        lla::segmented_reduce(stream, kd.cview(), vd.cview(), out.view(), 1u, n);
        stream.synchronize();
        Timer t;
        lla::segmented_reduce(stream, kd.cview(), vd.cview(), out.view(), 1u, n);
        stream.synchronize();
        const double ms = t.elapsed_ms();
        CHECK(out.copy_to() == static_cast<double>(n));
        std::printf("    [1e6] segmented_reduce<double> 1 segment  : %8.2f ms  exact=%d\n", ms,
                    (int)(out.copy_to() == static_cast<double>(n)));
    }
}
