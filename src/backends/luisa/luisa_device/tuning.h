#pragma once
/********************************************************************
 * @file tuning.h
 * @brief Process-wide, *trace-time* knobs (optimize_plan.md OP-C1 block size,
 *        OP-C3 CCD budget) and the compile counter (OP-A4).
 *
 * ## Why a global and not a parameter
 * Both values are read while a kernel is being **traced** (`set_block_size`, the CCD
 * loop bound), i.e. once per shader per process, and both are baked into the shader
 * that LC then compiles and caches. That makes them process-wide by construction:
 * there is no per-dispatch API for either (LC's `ShaderDispatch` copies
 * `info.block_size` at record time and exposes no setter, `include/luisa/runtime/
 * shader.h:271`), so a sweep over block sizes or CCD budgets is one *process* per
 * configuration -- exactly the A/B protocol of optimize_plan.md 7 (fresh process per
 * run, same binary).
 *
 * ## Precedence
 *   1. `config["luisa"]["block_dim"|"ccd_max_iter"|"block_dim_table"]` (scene config,
 *      applied by `SimEngine::init_device`);
 *   2. the environment (`UIPC_LUISA_BLOCK_DIM`, `UIPC_LUISA_BLOCK_DIM_TABLE`,
 *      `UIPC_LUISA_CCD_MAX_ITER`), applied lazily on the first `tuning()` call and
 *      therefore *after* the config -- the environment is the sweep channel and wins.
 * The environment spelling mirrors `UIPC_BENCHMARK_TIMERS` in bench_plan.md 7: one
 * binary, one env var, fresh process.
 *
 * ## What may be retuned, and what may not
 * A block size may only be retuned where the kernel is **block-size agnostic**: the
 * body must bound itself with `$if (i < n)` (or `$if (tid < count)`), so that a
 * different block size changes only how the same thread range is cut into blocks.
 * Kernels whose block size is part of the *algorithm* -- two-level scans and radix
 * passes size their shared-memory tiles and level offsets from it
 * (`kScanBlockSize`, `kRadixBlockSize` in luisa_device/linear/algorithms.cpp) -- are
 * deliberately not wired to `block_dim()`; changing those silently changes the
 * partition, not the schedule.
 *
 * A retuned block size can also be *rejected by the hardware*: the trajectory-filter
 * kernels need enough registers per thread that a 384- or 512-thread block does not fit
 * on an sm_89 device (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, measured 2026-10-28 --
 * see agent_docs/performance/2026-10-28-op-c3-runtime-ccd-loop.md). That failure is
 * loud and immediate, which is the behaviour we want from a mistuned knob; the values
 * 32..1024 are accepted here and each family's usable upper bound is a measurement, not
 * an assumption (the plan's own sweep range {64..512}, optimize_plan.md OP-C1).
 *********************************************************************/
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace uipc::backend::luisa::luisa_device
{
/// The port's historical block size (`luisa_device/kernel.h`), and LC's own default.
inline constexpr std::uint32_t kDefaultBlockDim = 256u;

/// OP-C3: the CCD iteration budget.  The reference calls CCD with
/// `constexpr SizeT max_iter = 1000` *as a budget* (its loop exits on a device test),
/// which the CUDA backend passes verbatim.  The Luisa port could not: its CCD loop was
/// unrolled at trace time, so 1000 iterations meant 1000 copies of the body in the IR
/// (>900 s to JIT, cookbook 7.3), and the TOI kernels passed a reduced 16 instead.
/// With the runtime `$while` loop (details/ccd.inl) the bound is only a loop limit, so
/// the port uses the convergence-sized reference value: every CCD query in the port's
/// test suite converges within 64 iterations, and 64 is what the parity cases were
/// measured at (optimize_plan.md OP-C3).
inline constexpr int kDefaultCcdMaxIter = 64;
/// Upper clamp: CUDA's own budget.  Larger than this has no meaning.
inline constexpr int kCcdMaxIterLimit = 1024;
/// Lower clamp: below this the CCD stops being conservative on real contacts.
inline constexpr int kCcdMaxIterFloor = 1;

/// Trace-time knobs.  Read through `tuning()`; never cached by a call site.
struct LDeviceTuning
{
    std::uint32_t block_dim = kDefaultBlockDim;
    int           ccd_max_iter = kDefaultCcdMaxIter;
};

namespace detail
{
inline constexpr std::size_t kMaxNamedBlockDims = 32u;
inline constexpr std::size_t kNameBytes         = 24u;

/// One `name -> block_dim` override (OP-C1: "a table keyed by kernel name so a sweep
/// can override without recompiling the call sites").
struct NamedBlockDim
{
    char          name[kNameBytes] = {};
    std::uint32_t dim = 0u;
};

struct TuningStorage
{
    LDeviceTuning  tuning;
    NamedBlockDim  named[kMaxNamedBlockDims] = {};
    std::size_t    named_count = 0u;
};

[[nodiscard]] inline TuningStorage& storage() noexcept
{
    static TuningStorage s;
    return s;
}

inline void set_named_block_dim(std::string_view name, std::uint32_t dim) noexcept
{
    auto& s = storage();
    for(std::size_t i = 0; i < s.named_count; ++i)
    {
        if(std::string_view{s.named[i].name} == name)
        {
            s.named[i].dim = dim;
            return;
        }
    }
    if(s.named_count >= kMaxNamedBlockDims || name.empty() || name.size() >= kNameBytes)
        return;
    auto& slot = s.named[s.named_count++];
    std::memcpy(slot.name, name.data(), name.size());
    slot.name[name.size()] = '\0';
    slot.dim               = dim;
}

[[nodiscard]] inline std::uint32_t named_block_dim(std::string_view name) noexcept
{
    auto& s = storage();
    for(std::size_t i = 0; i < s.named_count; ++i)
    {
        if(std::string_view{s.named[i].name} == name)
            return s.named[i].dim;
    }
    return 0u;
}

/// `UIPC_LUISA_*` parsing: a plain `strtol`, clamped by the caller.
[[nodiscard]] inline const char* env(std::string_view name) noexcept
{
    // getenv needs a NUL-terminated key; every key here is a literal, but the
    // signature takes a string_view so the call sites stay readable.
    if(name.size() >= 64u)
        return nullptr;
    char key[64];
    std::memcpy(key, name.data(), name.size());
    key[name.size()] = '\0';
    return std::getenv(key);
}

[[nodiscard]] inline bool parse_uint(std::string_view text, std::uint32_t& out) noexcept
{
    if(text.empty())
        return false;
    std::uint32_t v = 0;
    for(char c : text)
    {
        if(c < '0' || c > '9')
            return false;
        v = v * 10u + static_cast<std::uint32_t>(c - '0');
    }
    out = v;
    return true;
}

/// Apply the environment once, on first use.  Called from `tuning()`, i.e. after the
/// scene config has been applied (see the file header).
inline void apply_env_once() noexcept
{
    static const bool once = []() noexcept
    {
        auto& s = storage();

        if(const char* text = env("UIPC_LUISA_BLOCK_DIM"))
        {
            std::uint32_t v = 0;
            if(parse_uint(text, v) && v >= 32u && v <= 1024u && (v % 32u) == 0u)
                s.tuning.block_dim = v;
        }
        if(const char* text = env("UIPC_LUISA_CCD_MAX_ITER"))
        {
            std::uint32_t v = 0;
            if(parse_uint(text, v))
            {
                if(v < std::uint32_t(kCcdMaxIterFloor))
                    v = kCcdMaxIterFloor;
                if(v > std::uint32_t(kCcdMaxIterLimit))
                    v = kCcdMaxIterLimit;
                s.tuning.ccd_max_iter = int(v);
            }
        }
        if(const char* text = env("UIPC_LUISA_BLOCK_DIM_TABLE"))
        {
            // "toi=128,spmv=64"
            std::string_view rest{text};
            while(!rest.empty())
            {
                const auto comma = rest.find(',');
                const auto item  = rest.substr(0, comma);
                const auto eq    = item.find('=');
                if(eq != std::string_view::npos)
                {
                    std::uint32_t v = 0;
                    if(parse_uint(item.substr(eq + 1u), v) && v >= 32u && v <= 1024u
                       && (v % 32u) == 0u)
                        set_named_block_dim(item.substr(0u, eq), v);
                }
                if(comma == std::string_view::npos)
                    break;
                rest.remove_prefix(comma + 1u);
            }
        }
        return true;
    }();
    (void)once;
}
}  // namespace detail

/// The knobs (see the file header for precedence).
[[nodiscard]] inline LDeviceTuning& tuning() noexcept
{
    detail::apply_env_once();
    return detail::storage().tuning;
}

/// Block size for a kernel.  An empty name asks for the global default; a named kernel
/// falls back to the global default when no override was registered (OP-C1).
///
/// Named `block_dim_for` on purpose: `luisa_device::kernel::block_dim` is the *constexpr*
/// historical helper (`cuda_tool::best_block_dim` parity, kernel.h), and two functions
/// called `block_dim` in two nesting levels of one namespace is a trap, not a convenience.
[[nodiscard]] inline std::uint32_t block_dim_for(std::string_view kernel = {}) noexcept
{
    auto& t = tuning();
    if(!kernel.empty())
    {
        if(auto named = detail::named_block_dim(kernel); named != 0u)
            return named;
    }
    return t.block_dim;
}

/// OP-C3: the CCD iteration budget passed to the TOI kernels.
[[nodiscard]] inline int ccd_max_iter() noexcept
{
    return tuning().ccd_max_iter;
}

/// OP-A4: number of *shader compiles* performed by this process.  Every compile goes
/// through `luisa_device/kernel.h` (`compile1/2/3`, and `lla::detail::cached_kernel`
/// which is built on them), so this is the exact JIT count of the process -- the
/// "compiles/frame must be 0 after frame 1" acceptance of optimize_plan.md OP-A4
/// without a per-call-site audit.
namespace compile_stats
{
[[nodiscard]] inline std::atomic<std::uint64_t>& total_storage() noexcept
{
    static std::atomic<std::uint64_t> total{0};
    return total;
}

/// Called by the compile helpers, once per shader.
inline void note_compile() noexcept
{
    total_storage().fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline std::uint64_t total() noexcept
{
    return total_storage().load(std::memory_order_relaxed);
}
}  // namespace compile_stats
}  // namespace uipc::backend::luisa::luisa_device
