#pragma once
/********************************************************************
 * @file nvtx_range.h
 * @brief The single NVTX switch shared by the CUDA and the Luisa backend
 * (bench_plan.md 13.3, task BN1, instrumentation items I1-I6).
 *
 * One compile-time switch, `UIPC_NVTX`, decides whether the calls below emit NVTX
 * events or vanish:
 *
 *  - `UIPC_NVTX=1`: `<nvtx3/nvToolsExt.h>` is included. Those headers are
 *    self-contained on this machine -- header-only, they resolve the injection
 *    library at run time (no `nvToolsExt64_1.lib` is shipped for CUDA 12.9) and fall
 *    back to no-ops when no tool is attached, so an unprofiled run pays only an
 *    already-resolved stub call.
 *  - `UIPC_NVTX=0` (the default): every function below is an empty `inline` stub, so
 *    a call site compiles identically in both configurations and the compiler folds
 *    the whole thing away. There is deliberately no `#if` at any call site.
 *
 * Two invariants this header exists to protect (BN1's own acceptance criterion):
 *
 *  1. Turning it on must not change any measurement by more than 1 %. Everything here
 *     is one push/pop pair (or one mark) on the calling host thread: no allocation, no
 *     synchronisation, no device code. Tens of nanoseconds next to the >= 100 ms
 *     measured window of a Tier-1 case (bench_plan.md 4.2).
 *  2. It changes no cache key. Naming a compiled shader goes through LuisaCompute's
 *     `Resource::set_name` -- a runtime label that is *not* hashed (plan 13.2, N1) --
 *     and never through `ShaderOption::name`, which is hashed and would force a full
 *     re-JIT (B4). A `UIPC_NVTX=1` build and a `UIPC_NVTX=0` build therefore share one
 *     `<exe>/.cache`.
 *
 * Vocabulary (do not invent names): range names come from bench_plan.md 12.2 (the
 * already-instrumented stage names, reused *verbatim* so the profiler timeline joins
 * the `Timer` JSON by name -- what gate G4 needs) and from 4.2 (the Tier-1 case IDs,
 * e.g. `K1.spmv.rbk_sym`). The LuisaCompute infrastructure ranges are the fixed set
 * `LC.begin` / `LC.commit` / `LC.flush` / `LC.sync` / `LC.readback` (13.3, I2).
 *
 * Usage:
 *
 *   uipc::nvtx::range r{"Assemble Subsystems"};  // RAII: pops on scope exit
 *   uipc::nvtx::push("PCG");                     // manual; must be matched by pop()
 *   uipc::nvtx::pop();
 *   uipc::nvtx::name_thread("main");
 *   uipc::nvtx::mark("frame 42");
 *
 * The "static string" (ASCII, `...A`) entry points are used throughout instead of the
 * `...Ex` + `nvtxEventAttributes_t` ones: the attribute struct has to be filled in on
 * the stack for every event, and none of the attributes (colour, category, payload)
 * are used here. NVTX copies the message during the call, so a temporary buffer is
 * always safe.
 *
 * This header is host-only but must stay includable from `.cu` translation units
 * (`cuda_tool/launch_base.h` includes it); device code is unaffected because nothing
 * here is `__device__` or reachable from a kernel.
 *********************************************************************/
#include <string>
#include <string_view>

// Set by the build (`xmake f --nvtx=y`, see the backends' xmake.lua). A translation
// unit that misses the define gets the no-op build, which is the safe default.
#ifndef UIPC_NVTX
#define UIPC_NVTX 0
#endif

#if UIPC_NVTX

// nvToolsExt.h pulls in <windows.h>, which defines the `min`/`max` macros that break
// every `std::min`/`std::max` call. Define them away *before* the include, otherwise
// switching the instrumentation on would change how unrelated headers compile
// (NOMINMAX is already PUBLIC on uipc_core, so this is a no-op in the CMake build and
// a fix for a TU that reaches here first in the xmake build). Deliberately not
// WIN32_LEAN_AND_MEAN: this header must not make declarations other code relies on
// disappear.
#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <nvtx3/nvToolsExt.h>

#include <cstddef>
#include <cstdint>

namespace uipc::nvtx
{
/// True when the instrumentation is compiled in. The one place a call site may branch
/// on the switch; everything else stays uniform thanks to the stubs below.
inline constexpr bool enabled = true;

namespace detail
{
    /// Range names in this project are short ("Compute Feasible Step",
    /// "K1.spmv.rbk_sym", ...: a few dozen characters at most). NVTX's ASCII API wants a
    /// NUL-terminated C string, so a `std::string_view` is copied into this bounded stack
    /// buffer: no allocation, no lifetime requirement, and a longer name is truncated
    /// rather than read past its end.
    inline constexpr std::size_t kMaxNameLength = 128;

    [[nodiscard]] inline const char* ascii(std::string_view name,
                                           char (&buffer)[kMaxNameLength]) noexcept
    {
        const std::size_t n =
            name.size() < kMaxNameLength - 1 ? name.size() : kMaxNameLength - 1;
        for(std::size_t i = 0; i < n; ++i)
        {
            buffer[i] = name[i];
        }
        buffer[n] = '\0';
        return buffer;
    }
}  // namespace detail

/// Open a range the profiler shows as `name`, on the calling thread.
/// Three overloads keep the common cases allocation-free: a string literal is passed
/// straight through, a `std::string` uses its NUL-terminated buffer, and a
/// `std::string_view` goes through the bounded stack copy above.
inline void push(const char* name) noexcept
{
    ::nvtxRangePushA(name);
}

inline void push(const std::string& name) noexcept
{
    push(name.c_str());
}

inline void push(std::string_view name) noexcept
{
    char buffer[detail::kMaxNameLength];
    push(detail::ascii(name, buffer));
}

/// Close the innermost range pushed on *this* thread. NVTX keeps one stack per thread
/// per domain, so a push and its pop must happen on the same thread -- use `range`
/// below whenever the range can be left by an early return or an exception.
inline void pop() noexcept
{
    ::nvtxRangePop();
}

/// Rename the calling thread in the timeline's thread row.
/// NVTX has no portable "name the calling thread" entry point: `nvtxNameOsThreadA`
/// takes an OS thread id, which only the platform header knows. Windows and Linux
/// cover every configuration that can attach a profiler here (macOS has no CUDA).
inline void name_thread(const char* name) noexcept
{
#if defined(_WIN32)
    ::nvtxNameOsThreadA(static_cast<uint32_t>(::GetCurrentThreadId()), name);
#else
    (void)name;  // no portable thread id: silently skip on other platforms
#endif
}

inline void name_thread(std::string_view name) noexcept
{
    char buffer[detail::kMaxNameLength];
    name_thread(detail::ascii(name, buffer));
}

/// A point event (no duration) -- for "this happened once", not "this took time".
inline void mark(const char* name) noexcept
{
    ::nvtxMarkA(name);
}

inline void mark(std::string_view name) noexcept
{
    char buffer[detail::kMaxNameLength];
    mark(detail::ascii(name, buffer));
}

/// RAII guard: pushes in the constructor, pops in the destructor, so a range is closed
/// no matter how its scope is left. Non-copyable and non-movable on purpose -- moving
/// it would make "who pops" ambiguous and could unbalance the thread's stack.
class range
{
  public:
    explicit range(const char* name) noexcept { push(name); }
    explicit range(const std::string& name) noexcept { push(name); }
    explicit range(std::string_view name) noexcept { push(name); }
    ~range() noexcept { pop(); }

    range(const range&)            = delete;
    range& operator=(const range&) = delete;
    range(range&&)                 = delete;
    range& operator=(range&&)      = delete;
};
}  // namespace uipc::nvtx

#else  // UIPC_NVTX

namespace uipc::nvtx
{
/// False in the no-op build: the switch is compile-time, so `if constexpr` on this
/// (or on `UIPC_NVTX` directly) removes the branch entirely.
inline constexpr bool enabled = false;

inline void push(const char* /*name*/) noexcept {}
inline void push(const std::string& /*name*/) noexcept {}
inline void push(std::string_view /*name*/) noexcept {}
inline void pop() noexcept {}
inline void name_thread(const char* /*name*/) noexcept {}
inline void name_thread(std::string_view /*name*/) noexcept {}
inline void mark(const char* /*name*/) noexcept {}
inline void mark(std::string_view /*name*/) noexcept {}

/// Same API, same call sites, zero cost: every member is empty and inlined away.
class range
{
  public:
    explicit range(const char* /*name*/) noexcept {}
    explicit range(const std::string& /*name*/) noexcept {}
    explicit range(std::string_view /*name*/) noexcept {}
    ~range() noexcept {}

    range(const range&)            = delete;
    range& operator=(const range&) = delete;
    range(range&&)                 = delete;
    range& operator=(range&&)      = delete;
};
}  // namespace uipc::nvtx

#endif  // UIPC_NVTX
