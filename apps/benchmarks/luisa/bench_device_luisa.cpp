/********************************************************************
 * @file bench_device_luisa.cpp
 * @brief LuisaCompute implementation of `uipc::benchmark::BenchDevice`
 *        (bench_plan.md §6.1 / B1, the LC half of the two-TU contract of
 *        common/bench_device.h).
 *
 * Everything is built on the in-tree LC facade, not on raw LC:
 *   init()/shutdown()  -> luisa_device::Runtime (context.h), the process-wide
 *                         Context + Device + main Stream, torn down stream ->
 *                         device -> context (context.h:41-48);
 *   kernels            -> luisa_device::compile1<...> (kernel.h:37-41);
 *   submission         -> lcrt::Stream / lcrt::CommandList;
 *   graphs             -> lcrt::CudaGraphExt (cuda_graph_ext.h, CUDA-only).
 *
 * Four asymmetries against the CUDA TU are recorded here because they decide
 * how the resulting numbers may be read (§2.3, "the structural asymmetry the
 * benchmark must expose"):
 *
 *  1. Compile-once, not zero-JIT. compile_fill()/compile_increment() run the LC
 *     DSL through JIT (NVRTC) on first call. The compile is deliberately outside
 *     the measured window (the header's "compile-once discipline"), but it is
 *     minutes-scale on a cold .cache (cookbook §7.37 rule 8) and must never be
 *     attributed to a dispatch. That is the difference the plan wants isolated
 *     from "Luisa is slow" reports (Q4).
 *  2. One stream, and it is the facade's. `Runtime::main_stream()` is the exact
 *     lcrt::Stream the production Luisa backend submits into
 *     (src/backends/luisa/frame_stream.cpp), so H0.1/H0.4 measure the real
 *     submission path rather than a benchmark-only stream object. The CUDA TU
 *     has to create a dedicated non-blocking stream instead.
 *  3. A batching layer exists and is the point of H0.2. begin_batch() opens one
 *     lcrt::CommandList, dispatch() only *records* into it, end_batch() submits
 *     it once. commits therefore stops tracking dispatches, which is precisely
 *     the "does batching change wall time here" question (Q3).
 *  4. Graphs are a CUDA-backend extension, resolved at run time. When
 *     device().extension<CudaGraphExt>() is null (any non-CUDA LC backend) every
 *     graph entry point degrades to a counted refusal instead of aborting, so a
 *     record never silently reports 0 ms for a capture that never happened.
 *
 * Lifetime rule that shapes this file: destroying an LC resource after
 * Runtime::shutdown() calls destroy_* on a freed device and aborts the process
 * (cookbook §7.10 trap 1, quoted in view.h:180-184). Compiled shaders are
 * therefore heap-pinned and only deleted by shutdown(), which runs *before* the
 * runtime goes away; a BenchDevice whose shaders outlive it leaks them on
 * purpose -- one process per case, bench_plan.md B3.
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa/backends/ext/cuda/cuda_graph_ext.h>

#include <luisa_device/context.h>
#include <luisa_device/kernel.h>
#include <luisa_device/stream.h>

#include <common/bench_device.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The repo's spelling of the LC run time, declared at file scope so a
// `::lcrt::` qualified name also resolves (compat.h:18 declares the same alias
// inside the luisa_device namespace).
namespace lcrt = ::luisa::compute;
namespace lud  = ::uipc::backend::luisa::luisa_device;

namespace uipc::benchmark::detail
{
inline constexpr std::size_t kFloatBytes = sizeof(float);

using Clock = std::chrono::steady_clock;

[[nodiscard]] inline double ms_since(Clock::time_point start) noexcept
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/**
 * @brief The one kernel prototype of Tier 0: (buffer, thread count, value).
 *
 * LC derives the uniform layout from the *prototype* list, so fill and increment
 * are the same `lcrt::Shader` type (kernel.h:37-41: compile1<Args...> returns
 * Shader<1u, Args...>). That is why dispatch() needs no kernel-identity branch on
 * this backend while the CUDA TU dispatches on the function pointer.
 */
using Tier0Shader = lcrt::Shader<1u, lcrt::Buffer<float>, ::luisa::uint, float>;

/**
 * @brief Abort the process on a condition the benchmark cannot survive.
 *
 * Why abort and not return/throw: a silently skipped call is measured as a *fast*
 * case, and §6.2 calls that out ("a wrong kernel that runs in 0 ms must fail
 * loudly"). A thrown exception would also be caught by the case runner and turned
 * into a "skipped" row, which is just as misleading. Named `lc_*` so the two
 * backend TUs' helpers stay distinguishable.
 */
[[noreturn]] inline void lc_fatal(const char* what)
{
    std::fprintf(stderr, "FATAL bench_device_luisa: %s\n", what);
    std::fflush(stderr);
    std::abort();
}

/// Teardown and probe paths warn instead of aborting: a destructor must not kill
/// the run, and a refusal is data (H0.6), not a crash.
inline void lc_warn(const char* what) noexcept
{
    std::fprintf(stderr, "WARNING bench_device_luisa: %s\n", what);
    std::fflush(stderr);
}

/**
 * @brief Collector for LC's own log lines, used to quote refusals verbatim.
 *
 * CudaGraphExt has no error return: a refused capture is an invalid handle *plus*
 * a LUISA_WARNING_WITH_LOCATION on LC's default logger
 * (cuda_graph_ext.h:24-26 "Unsupported content must be refused loudly (warning +
 * failure return)"). Capturing that text is the only way to record *why* the
 * backend refused, which is what H0.6 asks for.
 *
 * The sink is additive (core/logging.h:49-55: default_logger_add_sink +
 * create_sink_with_callback), so libuipc's logging setup keeps working and nothing
 * has to be restored afterwards -- LC exposes no remove_sink. The state is
 * shared-ownership on purpose: the logger outlives every BenchDevice, so a raw
 * pointer into the device would dangle at process teardown.
 */
class LogCapture
{
  public:
    void begin()
    {
        std::scoped_lock lock{m_mutex};
        m_active = true;
        m_text.clear();
    }

    [[nodiscard]] std::string end()
    {
        std::scoped_lock lock{m_mutex};
        m_active = false;
        return m_text;
    }

    // Not noexcept: it appends to a std::string, and a logger callback that turns
    // an allocation failure into std::terminate would kill the run mid-probe.
    void push(const char* level, const char* message)
    {
        std::scoped_lock lock{m_mutex};
        if(!m_active)
            return;
        // Bounded: one refused capture emits one or two lines; anything longer is a
        // runaway logger and must not end up inside a CSV field.
        if(m_text.size() > 4096u)
            return;
        m_text += '[';
        m_text += level == nullptr ? "log" : level;
        m_text += "] ";
        m_text += message == nullptr ? "" : message;
        m_text += "; ";
    }

  private:
    std::mutex  m_mutex;
    bool        m_active = false;
    std::string m_text;
};

/// Add the one capture sink this process will ever install. Idempotent by
/// construction: the caller keeps the returned flag in BenchDevice::Impl.
inline void install_log_sink(const std::shared_ptr<LogCapture>& capture) noexcept
{
#if !defined(LUISA_CUSTOM_LOGGER)
    luisa::detail::default_logger_add_sink(luisa::detail::create_sink_with_callback(
        [capture](const char* level, const char* message)
        { capture->push(level, message); }));
#else
    // A LUISA_CUSTOM_LOGGER build replaces LC's logger wholesale and has no sink API
    // (core/logging.h:40-41 only exposes set_custom_logger there). The H0.6 notes then
    // carry this file's own observations and the header citations, without LC's text.
    (void)capture;
#endif
}

/// Trim LC's copied warning text into something a CSV field can hold.
[[nodiscard]] inline std::string join_notes(const std::string& first,
                                            const std::string& captured)
{
    if(captured.empty())
        return first;
    std::string text = first;
    text += " | LC log: ";
    text += captured;
    return text;
}

// ---- kernel factories -----------------------------------------------------
// Compiled once per case, outside the measured window (header contract). The
// returned object is pinned on the heap: ~Shader calls destroy_shader() on its
// device, and that device may already be gone at process teardown (see the file
// header). shutdown() deletes it while the device is still alive.

// clang-format off
// Kept as written: clang-format cannot parse the DSL's `$if(cond) { ... };`
// statement macro and re-indents the kernel bodies into something unreadable.
// Precedent: src/backends/luisa/affine_body/constitutions/*.h.
[[nodiscard]] inline Tier0Shader* make_fill_shader()
{
    // Named with the §4.2 case ID so `nsys stats --report nvtx_kern_sum` produces one row
    // per case instead of a single `kernel_main` row (§13.2 N1, gate G6(a)). The name is a
    // runtime label: it is not part of the shader hash, so it cannot invalidate
    // <exe>/.cache.
    return new Tier0Shader{
        lud::compile1<lcrt::Buffer<float>, ::luisa::uint, float>("H0.fill",
            [](lcrt::BufferVar<float> b,
               lcrt::Var<::luisa::uint> n,
               lcrt::Var<float> v) noexcept
            {
                // dispatch_x() is the flat thread index (dsl/builtin.h:297-300);
                // dispatch(n) is a *thread* count, never a block count
                // (lc_optimize §5.7).
                const auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    b->write(i, v);
                };
            })};
}

[[nodiscard]] inline Tier0Shader* make_increment_shader()
{
    // Same §4.2 case-ID label as the fill kernel (see above); a distinct name because the
    // two kernels answer two different questions in a profile.
    return new Tier0Shader{
        lud::compile1<lcrt::Buffer<float>, ::luisa::uint, float>("H0.increment",
            // Same prototype as fill so both kernels share one Shader type; `n`
            // guards the write, `v` is unused here by contract (b[i] += 1.0f).
            [](lcrt::BufferVar<float> b,
               lcrt::Var<::luisa::uint> n,
               lcrt::Var<float> /*v*/) noexcept
            {
                const auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    b->write(i, b->read(i) + 1.0f);
                };
            })};
}

/// H0.6 printing-kernel row. `device_log` is LC's only device print facility
/// (dsl/stmt.h:531-537; the repo wraps it in LC_LOG, luisa_device/log.h:32).
[[nodiscard]] inline Tier0Shader* make_printing_shader()
{
    return new Tier0Shader{
        lud::compile1<lcrt::Buffer<float>, ::luisa::uint, float>(
            [](lcrt::BufferVar<float> b,
               lcrt::Var<::luisa::uint> n,
               lcrt::Var<float>         v) noexcept
            {
                const auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    b->write(i, v);
                    lcrt::device_log("[probe] b[{}] = {}", i, v);
                };
              })};
}
// clang-format on

/// Outcome of one throw-away capture attempt (H0.6).
struct ProbeCapture
{
    bool        graph_valid = false;  ///< create_graph produced a graph
    bool        exec_valid  = false;  ///< ... and it instantiated
    std::string stages;               ///< our own verdict + LC's verbatim warnings

    [[nodiscard]] bool accepted() const noexcept { return graph_valid && exec_valid; }
};
}// namespace uipc::benchmark::detail

namespace uipc::benchmark
{
using detail::Clock;
using detail::kFloatBytes;
using detail::ms_since;

// ---- pimpl ----------------------------------------------------------------

struct BenchDevice::Impl
{
    // Borrowed, not owned: lud::Runtime holds the Context, the Device and the
    // single main Stream for the whole process (context.h:41-48).
    lcrt::Device* m_device = nullptr;
    lcrt::Stream* m_stream = nullptr;
    /// nullptr when the backend has no graph extension (any non-CUDA LC backend);
    /// every graph entry point then refuses instead of aborting (§2.4 "Enabling").
    lcrt::CudaGraphExt* m_ext = nullptr;

    DeviceInfo m_info;
    Counters   m_counters;
    bool       m_initialized = false;

    // Compiled kernels, heap-pinned (see the file header). Owned by this object but
    // deleted only from shutdown(), i.e. while the LC device is still alive.
    detail::Tier0Shader* m_fill_shader      = nullptr;
    detail::Tier0Shader* m_increment_shader = nullptr;
    detail::Tier0Shader* m_printing_shader  = nullptr;

    /// The open batch (H0.2): one CommandList, committed once by end_batch().
    std::optional<lcrt::CommandList> m_batch;

    /// Graph state (H0.3/H0.6/H0.7). Both handles are raw and unreferenced in LC,
    /// so the destroy order in graph_destroy() is mandatory (cuda_graph_ext.h:147).
    std::optional<lcrt::CudaGraphInstance>     m_graph;
    std::optional<lcrt::CudaGraphExecInstance> m_exec;

    /// LC warning collector for the H0.6 notes and for diagnosable refusals.
    std::shared_ptr<detail::LogCapture> m_log;
    bool                                m_log_sink_installed = false;

    void fill_info();
    void ensure_log();
    void release_shaders() noexcept;

    /// Lazily compiled printing kernel for the H0.6 row. The JIT cost is paid
    /// here, once, outside every measured window -- a printing shader is never
    /// part of a timed case.
    [[nodiscard]] detail::Tier0Shader& printing_shader()
    {
        if(m_printing_shader == nullptr)
            m_printing_shader = detail::make_printing_shader();
        return *m_printing_shader;
    }

    /// Record `nodes` fill dispaches into a fresh CommandList. Used by the capture
    /// and the update paths, which must produce a structurally identical list.
    [[nodiscard]] lcrt::CommandList make_node_list(const detail::Tier0Shader& shader,
                                                   lcrt::Buffer<float>&         buffer,
                                                   std::uint32_t                threads,
                                                   float                        v,
                                                   std::uint32_t                nodes);

    /// create_graph -> instantiate on a throw-away list, plus the sync+dismantle
    /// order the extension requires. Never launches (see probe_refusals()).
    detail::ProbeCapture probe_capture(lcrt::CommandList&& list);
};

void BenchDevice::Impl::fill_info()
{
    m_info.backend = "luisa";

    // The adapter name. The LC device interface exposes only query(property)
    // (device_interface.h:197), and this revision's CUDA backend answers
    // "device_name" with the *backend* name -- cuda_device.cpp:1500-1503 returns
    // "cuda". The real adapter list comes from Context::backend_device_names
    // (runtime/context.h:80), which asks the same backend module for
    // cuDeviceGetName (cuda_device.cpp:1842-1856). Prefer it; keep query() as the
    // documented fallback for a backend that answers with an empty list.
    std::string device_name;
    auto        names = lud::Runtime::instance().context().backend_device_names(
        std::string_view{m_device->backend_name()});
    if(!names.empty())
        device_name = names.front();
    if(device_name.empty())
        device_name = m_device->query("device_name");
    m_info.device_name = device_name;

    // LC 0.10.1 exposes its revision as build-time macros (version.h); there is no
    // runtime version query.
#if defined(LUISA_COMPUTE_VERSION_MAJOR) && defined(LUISA_COMPUTE_VERSION_MINOR) \
    && defined(LUISA_COMPUTE_VERSION_PATCH)
    const auto major = LUISA_COMPUTE_VERSION_MAJOR;
    const auto minor = LUISA_COMPUTE_VERSION_MINOR;
    const auto patch = LUISA_COMPUTE_VERSION_PATCH;
    m_info.runtime_version =
        std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
#else
    m_info.runtime_version = "0.10.x";
#endif

    // No driver string is obtainable from LC 0.10.1: DeviceInterface::query answers
    // only "device_name"/"total_memory"/"free_memory" (cuda_device.cpp:1500-1522)
    // and every other property returns {} with a warning. Recorded as
    // "driver=unavailable" rather than silently omitted, so a record reader is not
    // left guessing whether it was forgotten.
    const std::string total_memory = m_device->query("total_memory");
    char              buf[320];
    std::snprintf(buf,
                  sizeof(buf),
                  "lc_backend=%s lc=%s stream=main_stream graph_ext=%s "
                  "total_memory_bytes=%s driver=unavailable(no_lc_query)",
                  std::string{m_device->backend_name()}.c_str(),
                  m_info.runtime_version.c_str(),
                  m_ext != nullptr ? "CudaGraphExt" : "absent",
                  total_memory.empty() ? "?" : total_memory.c_str());
    m_info.extra = buf;
}

void BenchDevice::Impl::ensure_log()
{
    if(m_log_sink_installed)
        return;
    m_log = std::make_shared<detail::LogCapture>();
    detail::install_log_sink(m_log);
    m_log_sink_installed = true;
}

void BenchDevice::Impl::release_shaders() noexcept
{
    // Only valid while the LC device is alive: ~ShaderBase calls
    // device()->destroy_shader(handle()) (shader.h:260-262).
    delete m_fill_shader;
    delete m_increment_shader;
    delete m_printing_shader;
    m_fill_shader      = nullptr;
    m_increment_shader = nullptr;
    m_printing_shader  = nullptr;
}

lcrt::CommandList BenchDevice::Impl::make_node_list(const detail::Tier0Shader& shader,
                                                    lcrt::Buffer<float>&         buffer,
                                                    std::uint32_t                threads,
                                                    float                        v,
                                                    std::uint32_t                nodes)
{
    // reserve() so the recording loop never reallocates: this is what
    // CommandList::create(reserved_command_size, ...) exists for
    // (runtime/command_list.h:41-44) and it is the same shape LC's own graph
    // benchmark uses (benchmark_part2_verify.cpp:265-271).
    auto list = lcrt::CommandList::create(nodes, 0u);
    for(std::uint32_t i = 0; i < nodes; ++i)
    {
        // One statement per node, identical arguments: LC's DAG builder derives the
        // edges from the declared READ/WRITE usages, so N writes to one buffer
        // serialise (WAW) while the CUDA TU's hand-built nodes stay independent.
        // That asymmetry is recorded, not hidden: see the H0.3 note in the plan.
        list << shader(buffer.view(), ::luisa::uint{threads}, v).dispatch(threads);
    }
    return list;
}

detail::ProbeCapture BenchDevice::Impl::probe_capture(lcrt::CommandList&& list)
{
    detail::ProbeCapture result;
    if(m_ext == nullptr || m_stream == nullptr)
    {
        result.stages = "no CudaGraphExt on this device, nothing to attempt";
        return result;
    }

    ensure_log();
    if(m_log)
        m_log->begin();

    // Lifetime: these two RAII wrappers are the only owners of the handles, and
    // they are destroyed in reverse declaration order -- exec first, then the graph
    // template, which is the order cuda_graph_ext.h:147 requires. `exec` is an
    // optional at function scope, not a local of the `if`, so that the destructors
    // run *after* the synchronise() below instead of before it.
    auto                                       graph = m_ext->create_graph(std::move(list));
    std::optional<lcrt::CudaGraphExecInstance> exec;
    result.graph_valid = graph.handle().handle != lcrt::CudaGraphExt::invalid_handle;
    if(result.graph_valid)
    {
        exec.emplace(m_ext->instantiate(graph.handle().handle));
        result.exec_valid =
            exec->handle().handle != lcrt::CudaGraphExt::invalid_handle;
    }

    m_stream->synchronize();  // cuda_graph_ext.h:147, before any destroy

    if(m_log)
        result.stages = m_log->end();
    return result;
}

// ---- lifecycle ------------------------------------------------------------

BenchDevice::BenchDevice()
    : m_impl(std::make_unique<Impl>())
{
}

BenchDevice::~BenchDevice()
{
    if(m_impl)
        shutdown();
}

void BenchDevice::init(std::string_view program_path)
{
    Impl& d = *m_impl;
    if(d.m_initialized)
        return;

    // "cuda" is not a knob: CudaGraphExt is the only LC backend extension that can
    // express H0.3/H0.6/H0.7 (§2.4 "CUDA backend only"), and Tier 0 compares
    // against the CUDA backend.
    //
    // `program_path` is argv[0]: LC derives its runtime/data (i.e. .cache)
    // directories from it, and a backend loaded as a module never sees argv[0]
    // (context.h:69-72). Runtime::init is idempotent (context.cpp:27-30).
    lud::Runtime::instance().init("cuda", program_path);

    auto& runtime = lud::Runtime::instance();
    d.m_device    = &runtime.device();
    d.m_stream    = &runtime.main_stream();
    // static_cast inside Device::extension<Ext>() (device.h:137-140); nullptr when
    // the backend does not create the extension.
    d.m_ext = runtime.device().extension<lcrt::CudaGraphExt>();

    d.fill_info();
    d.m_initialized = true;

    std::printf("[bench_device_luisa] %s | runtime %s | %s\n",
                d.m_info.device_name.c_str(),
                d.m_info.runtime_version.c_str(),
                d.m_info.extra.c_str());
    std::fflush(stdout);
}

void BenchDevice::shutdown()
{
    if(!m_impl)
        return;
    Impl& d = *m_impl;
    if(!d.m_initialized)
        return;  // safe to call twice, and safe when init() never ran

    graph_destroy();

    // A list that was opened and never committed would trip ~CommandList's
    // "Destructing non-empty command list. Did you forget to commit?" assertion
    // (command_list.cpp:21-24), so close it first.
    if(d.m_batch.has_value())
        end_batch();

    // Before the runtime dies: see release_shaders().
    d.release_shaders();

    d.m_device = nullptr;
    d.m_stream = nullptr;
    d.m_ext    = nullptr;

    // stream -> device -> context, and safe to call twice (context.h:39-48).
    lud::Runtime::instance().shutdown();
    d.m_initialized = false;
}

const DeviceInfo& BenchDevice::info() const noexcept
{
    return m_impl->m_info;
}

// ---- memory ---------------------------------------------------------------

BufferHandle BenchDevice::alloc(std::size_t count)
{
    Impl& d = *m_impl;
    if(!d.m_initialized)
        detail::lc_fatal("alloc() before init()");
    if(count == 0u)
    {
        // Buffer's constructor refuses size 0 outright
        // (runtime/buffer.h:103-105 detail::error_buffer_size_is_zero()), so a
        // zero-float request has to stay a null handle, exactly like the CUDA row.
        return nullptr;
    }
    // Device::create_buffer<T>(size) (device.h:239-243) -> Buffer<float>, which owns
    // the device handle and destroys it in its destructor (buffer.h:117-119).
    auto* buffer = new lcrt::Buffer<float>{d.m_device->create_buffer<float>(count)};
    return buffer;
}

void BenchDevice::release(BufferHandle buffer)
{
    if(buffer == nullptr)
        return;
    // Must run before shutdown(): the destructor calls destroy_buffer().
    delete static_cast<lcrt::Buffer<float>*>(buffer);
}

void BenchDevice::upload(BufferHandle buffer, const float* host, std::size_t count)
{
    if(count == 0u)
        return;
    Impl& d = *m_impl;
    auto* b = static_cast<lcrt::Buffer<float>*>(buffer);
    if(b == nullptr)
        detail::lc_fatal("upload() on a null buffer");

    // view(0, count), not the whole buffer: BufferView::copy_from(span) sizes the
    // command from the *view* (runtime/buffer.h:292-299), so handing over the full
    // buffer while passing `count < capacity` floats would read past `host`. Same
    // idiom as the facade's LDeviceBuffer::copy_from (luisa_device/buffer.h:139).
    *d.m_stream << b->view(0u, count).copy_from(std::span<const float>{host, count});
    d.m_counters.h2d_bytes += count * kFloatBytes;

    // Counted sync, not a hidden one: LC stages an upload through its own host
    // block before the device copy (cuda_device upload path), and `host` is usually
    // a caller temporary. Mirrors the CUDA TU's reasoning.
    sync();
}

void BenchDevice::download(BufferHandle buffer, float* host, std::size_t count)
{
    if(count == 0u)
        return;
    Impl& d = *m_impl;
    auto* b = static_cast<lcrt::Buffer<float>*>(buffer);
    if(b == nullptr)
        detail::lc_fatal("download() on a null buffer");

    // BufferView::copy_to(span) (runtime/buffer.h:283-291); same count reasoning as
    // upload().
    *d.m_stream << b->view(0u, count).copy_to(std::span<float>{host, count});
    d.m_counters.d2h_bytes += count * kFloatBytes;
    sync();
}

void BenchDevice::zero(BufferHandle buffer, std::size_t count)
{
    if(count == 0u)
        return;
    Impl& d = *m_impl;
    auto* b = static_cast<lcrt::Buffer<float>*>(buffer);
    if(b == nullptr)
        detail::lc_fatal("zero() on a null buffer");

    // LC 0.10.1 has no fill/memset command for Buffer<T>: searched "fill" in
    // runtime/buffer.h, runtime/rhi/command.h and runtime/rhi/command_encoder.h --
    // none exists (the repo's own LBufferView::fill() is a *device-side kernel*,
    // luisa_device/view.h:194-202, not a runtime command). Zeroing is therefore an
    // upload, and it is counted as the H2D transfer it really is.
    //
    // The source block is a function-local static: its address is stable for the
    // process, so the deferred encoding of the copy can never outlive the data.
    static const std::array<float, 4096u> zeros{};
    for(std::size_t offset = 0u; offset < count; offset += zeros.size())
    {
        const std::size_t n = std::min(zeros.size(), count - offset);
        *d.m_stream << b->view(offset, n).copy_from(
            std::span<const float>{zeros.data(), n});
    }
    d.m_counters.h2d_bytes += count * kFloatBytes;
}

// ---- kernels --------------------------------------------------------------

KernelHandle BenchDevice::compile_fill()
{
    Impl& d = *m_impl;
    if(!d.m_initialized)
        detail::lc_fatal("compile_fill() before init()");
    if(d.m_fill_shader == nullptr)
        d.m_fill_shader = detail::make_fill_shader();
    return d.m_fill_shader;
}

KernelHandle BenchDevice::compile_increment()
{
    Impl& d = *m_impl;
    if(!d.m_initialized)
        detail::lc_fatal("compile_increment() before init()");
    if(d.m_increment_shader == nullptr)
        d.m_increment_shader = detail::make_increment_shader();
    return d.m_increment_shader;
}

// ---- submission -----------------------------------------------------------

void BenchDevice::dispatch(KernelHandle  kernel,
                           BufferHandle  buffer,
                           std::uint32_t threads,
                           float         v)
{
    Impl& d = *m_impl;
    auto* shader = static_cast<detail::Tier0Shader*>(kernel);
    auto* buf    = static_cast<lcrt::Buffer<float>*>(buffer);
    if(shader == nullptr || buf == nullptr)
        detail::lc_fatal("dispatch(): null kernel or buffer handle (not from "
                         "compile_fill()/compile_increment()/alloc())");
    if(d.m_stream == nullptr)
        detail::lc_fatal("dispatch() before init() or after shutdown()");

    // One statement = one LC dispatch. `threads` is the *total* thread count handed
    // to ShaderInvoke<1>::dispatch(uint) (runtime/shader.h:182-184), which the
    // backend turns into a grid using its own block size -- dispatch(n) is not a
    // block count (lc_optimize §5.7). threads == 0 is a submitted no-op here: LC's
    // normal path ignores empty launches (cuda_graph_ext.cpp:851-853), and it is
    // counted exactly like the CUDA row.
    ++d.m_counters.dispatches;

    if(d.m_batch.has_value())
    {
        // Batched path (H0.2): the statement is only *recorded*; end_batch() submits
        // the whole list with one commit. Nothing is counted in `commits` yet.
        *d.m_batch << (*shader)(buf->view(), ::luisa::uint{threads}, v).dispatch(threads);
        return;
    }

    // Plain path (H0.1): one command handed to the main stream. Stream::operator<<
    // returns a temporary Delegate whose destructor commits it (runtime/stream.h:85
    // plus Delegate::~Delegate), so this is one submission.
    *d.m_stream << (*shader)(buf->view(), ::luisa::uint{threads}, v).dispatch(threads);
    ++d.m_counters.commits;
}

void BenchDevice::begin_batch()
{
    Impl& d = *m_impl;
    if(!d.m_initialized)
        detail::lc_fatal("begin_batch() before init()");
    if(d.m_batch.has_value())
    {
        // Nest safety, trivial form: one open list per device. A nested list would
        // need a stack and would make the commit counter ambiguous (§6.5).
        detail::lc_fatal("begin_batch() while a batch is already open");
    }
    // LC's batching unit is one CommandList submitted once (stream.h:1-17).
    d.m_batch.emplace();
    ++d.m_counters.batches_opened;
}

void BenchDevice::end_batch()
{
    Impl& d = *m_impl;
    if(!d.m_batch.has_value())
        detail::lc_fatal("end_batch() without begin_batch()");

    // One submission for the whole list: this is the number the `commits` counter
    // exists to prove (§6.5 "commits: proves batching works").
    *d.m_stream << d.m_batch->commit();
    // The moved-from list is empty and its destructor's "did you forget to commit?"
    // assertion is satisfied (command_list.cpp:21-24,92-95).
    d.m_batch.reset();
    ++d.m_counters.commits;
}

void BenchDevice::sync()
{
    Impl& d = *m_impl;
    if(d.m_stream == nullptr)
        detail::lc_fatal("sync() before init() or after shutdown()");

    const auto t0 = Clock::now();
    // Stream::synchronize() (runtime/stream.h:95) is LC's single fence for the main
    // stream, and it is the same call the production backend uses at read-back
    // sites (80 of them, §2.3).
    d.m_stream->synchronize();
    d.m_counters.sync_ms += ms_since(t0);
    ++d.m_counters.syncs;
}

// ---- graphs ---------------------------------------------------------------

bool BenchDevice::graph_supported() const noexcept
{
    // Unlike the CUDA TU ("always"), the LC answer is a runtime question: the
    // extension is created lazily by the backends that have one
    // (cuda_device.cpp:1524-1530), and a non-CUDA LC backend returns nullptr.
    return m_impl->m_ext != nullptr;
}

bool BenchDevice::graph_capture_fill(KernelHandle  kernel,
                                     BufferHandle  buffer,
                                     std::uint32_t threads,
                                     float         v,
                                     std::uint32_t nodes)
{
    Impl& d = *m_impl;
    if(!d.m_initialized)
        detail::lc_fatal("graph_capture_fill() before init()");
    auto* shader = static_cast<detail::Tier0Shader*>(kernel);
    auto* buf    = static_cast<lcrt::Buffer<float>*>(buffer);
    if(shader == nullptr || buf == nullptr)
        detail::lc_fatal("graph_capture_fill(): null kernel or buffer handle");

    // A second capture replaces the first; destroy it in the defined order first.
    graph_destroy();

    if(d.m_ext == nullptr)
    {
        ++d.m_counters.graph_refusals;
        detail::lc_warn("graph_capture_fill: no CudaGraphExt on this device, nothing "
                        "was captured");
        return false;
    }
    if(nodes == 0u)
    {
        // Refused before create_graph() on purpose: an empty list returns
        // invalid_resource_handle *silently* (cuda_graph_ext.cpp:1063), and a
        // silent empty capture measured as 0 ms would fail the benchmark's whole
        // correctness idea (§2.4 "Failure signalling"). Counting it here keeps
        // graph_refusals the single place a refused capture shows up.
        ++d.m_counters.graph_refusals;
        detail::lc_warn("graph_capture_fill: nodes == 0 is an empty command list, "
                        "which CudaGraphExt refuses with an invalid handle and no "
                        "warning (cuda_graph_ext.cpp:1063)");
        return false;
    }

    auto list = d.make_node_list(*shader, *buf, threads, v, nodes);

    d.ensure_log();
    if(d.m_log)
        d.m_log->begin();

    // create_graph(CommandList&&) builds the DAG at capture time, never by stream
    // capture (cuda_graph_ext.h:126-129). The list must be committed first: LC's
    // own graph benchmark does exactly `std::move(list.commit()).command_list()`
    // (test_tile_cuda_graph.cpp:70).
    auto graph = d.m_ext->create_graph(std::move(list.commit()).command_list());
    const auto graph_handle = graph.handle().handle;
    std::string lc_log;
    if(d.m_log)
        lc_log = d.m_log->end();

    if(graph_handle == lcrt::CudaGraphExt::invalid_handle)
    {
        // invalid_handle == ~0ull (cuda_graph_ext.h:114). Quoting LC's warning is the
        // difference between "some capture failed" and a diagnosis. `graph` is
        // destroyed by ~CudaGraphInstance, which is a no-op for an invalid handle
        // (cuda_graph_ext.h:404-408), and no sync is needed because nothing on this
        // graph was ever launched.
        ++d.m_counters.graph_refusals;
        const std::string text = detail::join_notes(
            "graph_capture_fill: create_graph() returned invalid_resource_handle "
            "(refused content, or a driver failure)",
            lc_log);
        detail::lc_warn(text.c_str());
        return false;
    }

    // The instantiate is the one-off cost H0.3 amortises over 20 replays and the
    // cost H0.7 compares against a whole-graph update (cuda_graph_ext.h:134-145).
    const auto t0 = Clock::now();
    auto       exec = d.m_ext->instantiate(graph_handle);
    const double instantiate_ms = ms_since(t0);
    const auto   exec_handle    = exec.handle().handle;

    if(exec_handle == lcrt::CudaGraphExt::invalid_handle)
    {
        // The graph stays alive until `graph` goes out of scope, which destroys it
        // via ~CudaGraphInstance (cuda_graph_ext.h:404-408). No sync: nothing was
        // launched through this graph, so there is no in-flight work to fence.
        ++d.m_counters.graph_refusals;
        const std::string text = detail::join_notes(
            "graph_capture_fill: instantiate() returned invalid_resource_handle",
            lc_log);
        detail::lc_warn(text.c_str());
        return false;
    }

    // LC 0.10.1 has no cudaGraphGetNodes analogue -- every public member of
    // CudaGraphExt (cuda_graph_ext.h) was checked: capabilities(), update(),
    // update_*_node(), set_node_*/is_node_*, add_*, build_body_graph(),
    // get_*_handle(), launch(), upload(), destroy_*. A non-empty capture can
    // therefore not be verified by node count the way the CUDA TU does; the
    // detectable silent case is the empty list above, and the *other* silent case,
    // a zero-sized dispatch, is refused by the DAG visitor
    // (cuda_graph_ext.cpp:851-861), so it surfaces as invalid_handle too.
    d.m_graph.emplace(std::move(graph));
    d.m_exec.emplace(std::move(exec));
    ++d.m_counters.graph_captures;  // successful captures only; refusals are counted above
    ++d.m_counters.graph_instantiates;
    d.m_counters.graph_instantiate_ms += instantiate_ms;
    return true;
}

void BenchDevice::graph_launch()
{
    Impl& d = *m_impl;
    if(d.m_ext == nullptr || !d.m_exec.has_value() || d.m_stream == nullptr)
        detail::lc_fatal("graph_launch() without a captured graph");

    // HOST boundary, not a stream event: CudaGraphExt::launch(exec) returns a
    // CudaGraphLaunch that is *not* marked LUISA_MARK_STREAM_EVENT_TYPE, so
    // `stream << CudaGraphExt::launch(exec)` does not compile (Stream::operator<<
    // needs is_stream_event_v, runtime/stream.h:87-92; only event/sparse/swapchain
    // are marked, runtime/stream_event.h:14). Hence the direct call form used by
    // LC's own benchmark (benchmark_part2_verify.cpp:296-299), and hence "a graph
    // can never be a member of a batched frame CommandList" (§2.4 "Launch").
    d.m_ext->launch(d.m_exec->handle().handle, d.m_stream->handle());
    ++d.m_counters.graph_launches;
}

double BenchDevice::graph_update_fill(KernelHandle  kernel,
                                      BufferHandle  buffer,
                                      std::uint32_t threads,
                                      float         v,
                                      std::uint32_t nodes)
{
    Impl& d = *m_impl;
    if(d.m_ext == nullptr || !d.m_exec.has_value())
        detail::lc_fatal("graph_update_fill() without a captured graph");
    auto* shader = static_cast<detail::Tier0Shader*>(kernel);
    auto* buf    = static_cast<lcrt::Buffer<float>*>(buffer);
    if(shader == nullptr || buf == nullptr)
        detail::lc_fatal("graph_update_fill(): null kernel or buffer handle");

    const auto t0 = Clock::now();

    // 1. Re-capture an equivalent list. This is LC's own contract for the
    //    whole-graph path: update(exec, CommandList&&) = "Update all params from a
    //    structurally-identical CommandList" and is implemented as re-capture +
    //    cuGraphExecUpdate (cuda_graph_ext.h:167-174,
    //    cuda_graph_ext.cpp:1424-1434). Per-node update_kernel_node() is *not* the
    //    pattern: it builds one pointer per Argument while the node was created with
    //    the packed Params ABI, so it cannot reproduce the layout (§2.4, H0.7).
    auto list = d.make_node_list(*shader, *buf, threads, v, nodes);

    const double rebuild_ms = ms_since(t0);

    // 2. Whole-graph update, with the diagnostics overload so a refusal names its
    //    reason instead of just returning false (cuda_graph_ext.h:239-241).
    lcrt::CudaGraphExt::GraphUpdateError error{};
    const auto                           tu = Clock::now();
    const bool updated                     = d.m_ext->update(
        d.m_exec->handle().handle, std::move(list.commit()).command_list(), &error);
    const double update_ms = ms_since(tu);
    const double total_ms  = ms_since(t0);

    std::printf("graph_update_fill: nodes=%u rebuild=%.3f ms "
                "graph_update=%.3f ms error_type=%u error_node=%llu detail=%s\n",
                static_cast<unsigned>(nodes),
                rebuild_ms,
                update_ms,
                static_cast<unsigned>(error.type),
                static_cast<unsigned long long>(error.error_node),
                error.detail.empty() ? "-" : error.detail.c_str());
    std::fflush(stdout);

    // The plan's split is deliberate: rebuild_ms is "what a topology change costs",
    // update_ms is "what an update costs" (H0.7 vs destroy_exec + instantiate).
    ++d.m_counters.graph_updates;
    d.m_counters.graph_update_ms += total_ms;

    if(!updated)
    {
        // A failed whole-graph update is a REBUILD, not an invalid handle: §6.5
        // defines graph_refusals as "count of invalid_handle returns" and
        // graph_rebuilds as the rebuild budget ("topology-key mismatch / refusal /
        // capacity growth"), which is precisely what the caller must now do. The
        // exec keeps running the previous bake (cuGraphExecUpdate is atomic), so the
        // graph stays usable and the counters stay honest.
        ++d.m_counters.graph_rebuilds;
        detail::lc_warn("graph_update_fill: update() refused; the caller must "
                        "destroy_exec + instantiate (see error_type above)");
    }
    return total_ms;
}

void BenchDevice::graph_destroy()
{
    Impl& d = *m_impl;

    // Order is mandatory (§2.4 "Lifetime", cuda_graph_ext.h:147 "Caller must sync
    // the stream before destroying"): idle the stream, destroy the exec, then the
    // graph template. The sync is raw on purpose -- this is teardown, not a measured
    // window, so §6.5's syncs/sync_ms must not include it.
    if((d.m_exec.has_value() || d.m_graph.has_value()) && d.m_stream != nullptr)
        d.m_stream->synchronize();

    if(d.m_exec.has_value())
    {
        if(d.m_ext != nullptr)
            d.m_ext->destroy_exec(d.m_exec->handle().handle);
        // release() invalidates the wrapper so ~CudaGraphExecInstance does not
        // destroy the handle twice (cuda_graph_ext.h:416-426).
        (void)d.m_exec->release();
        d.m_exec.reset();
    }
    if(d.m_graph.has_value())
    {
        if(d.m_ext != nullptr)
            d.m_ext->destroy_graph(d.m_graph->handle().handle);
        (void)d.m_graph->release();  // cuda_graph_ext.h:404-414
        d.m_graph.reset();
    }
}

// ---- H0.6 refusal matrix --------------------------------------------------

std::vector<RefusalResult> BenchDevice::probe_refusals()
{
    Impl& d = *m_impl;
    if(!d.m_initialized)
        detail::lc_fatal("probe_refusals() before init()");

    std::vector<RefusalResult> rows;
    rows.reserve(static_cast<std::size_t>(RefusalKind::Count));

    // One tiny arena for every row: a fresh allocation per row would measure the
    // allocator, and no probe ever launches (see the note at the end).
    constexpr std::uint32_t kArenaFloats = 256u;
    auto* arena                          = static_cast<lcrt::Buffer<float>*>(alloc(kArenaFloats));
    if(arena == nullptr)
        detail::lc_fatal("probe_refusals(): alloc() returned no buffer");
    if(d.m_fill_shader == nullptr)
    {
        // Compiled here so a probe does not depend on compile_fill() having run
        // first. The one-off JIT is paid outside every measured window.
        d.m_fill_shader = detail::make_fill_shader();
    }

    if(d.m_ext == nullptr)
    {
        // No capture is possible at all, so the matrix is filled with the reason
        // instead of being empty: a missing row is indistinguishable from a row
        // nobody bothered to probe.
        const std::string why =
            "not probed: device().extension<lcrt::CudaGraphExt>() is null on this "
            "device, so no capture can be attempted (cuda_graph_ext.h:106-110 makes "
            "the extension CUDA-only; device_interface.h:198 returns nullptr for a "
            "backend that has none)";
        for(std::uint32_t k = 0; k < static_cast<std::uint32_t>(RefusalKind::Count); ++k)
        {
            RefusalResult row;
            row.kind     = static_cast<RefusalKind>(k);
            row.accepted = false;
            row.note     = why;
            rows.push_back(std::move(row));
        }
        release(arena);
        sync();
        return rows;
    }

    // ---- 1. printing kernel ------------------------------------------------
    {
        // LC refuses a printing dispatch at capture time: the DAG visitor rejects any
        // shader whose requires_printing() is set, together with non-native shaders,
        // bound arguments, indirect and multiple dispatch
        // (cuda_graph_ext.cpp:820-834; the contract is stated in
        // cuda_graph_ext.h:24-26 "printing kernels (kernel_cout)"). This is not
        // academic for libuipc: the port's debug print path is device_log
        // (luisa_device/log.h:26-37), so a debug scene cannot be captured.
        auto list = lcrt::CommandList::create(1u, 0u);
        list << d.printing_shader()(arena->view(), ::luisa::uint{1u}, 1.0f).dispatch(1u);
        auto probe = d.probe_capture(std::move(list));

        RefusalResult row;
        row.kind     = RefusalKind::PrintingKernel;
        row.accepted = probe.accepted();
        row.note     = detail::join_notes(
            "LC refuses a dispatch of a printing shader at capture (the visitor tests "
            "shader->requires_printing(), cuda_graph_ext.cpp:820-834): the kernel is "
            "the DSL's device_log (dsl/stmt.h:531-537). Verdict from this attempt: "
            + std::string{probe.graph_valid ? "graph created" : "invalid graph handle"}
            + (probe.exec_valid ? ", instantiated" : ", not instantiated"),
            probe.stages);
        rows.push_back(std::move(row));
    }

    // ---- 2. indirect dispatch ---------------------------------------------
    {
        // The DSL *can* express it: ShaderInvoke<1>::dispatch(const
        // IndirectDispatchBuffer&, offset, max_dispatch_size) exists in this
        // revision (runtime/shader.h:189-193) and the buffer comes from
        // Device::create_indirect_dispatch_buffer (runtime/device.h:150), so the
        // probe constructs the real content class rather than reporting a surface
        // gap. The header marks it "WARNING: Not supported by most backends".
        auto indirect = d.m_device->create_indirect_dispatch_buffer(1u);
        auto list     = lcrt::CommandList::create(1u, 0u);
        list << (*d.m_fill_shader)(arena->view(), ::luisa::uint{4u}, 2.0f)
                    .dispatch(indirect);
        auto probe = d.probe_capture(std::move(list));

        RefusalResult row;
        row.kind     = RefusalKind::IndirectDispatch;
        row.accepted = probe.accepted();
        row.note     = detail::join_notes(
            "Constructed with ShaderInvoke<1>::dispatch(const IndirectDispatchBuffer&) "
            "(runtime/shader.h:189-193) over a create_indirect_dispatch_buffer "
            "(runtime/device.h:150). LC's DAG visitor refuses any command with "
            "dispatch->is_indirect() (cuda_graph_ext.cpp:826-834), which matters "
            "directly for optimize_plan.md OP-A3/C5: an indirect size is a device value, "
            "a graph bakes the launch size at capture (cuda_graph_ext.h:28-43). "
            "Verdict: "
            + std::string{probe.graph_valid ? "graph created" : "invalid graph handle"}
            + (probe.exec_valid ? ", instantiated" : ", not instantiated")
            + ". The probe never launches a captured graph, so an uninitialised "
              "indirect size is never executed.",
            probe.stages);
        rows.push_back(std::move(row));
    }

    // ---- 3. multiple dispatch ---------------------------------------------
    {
        // A list with 2 dispatches. LC's multi-launch command is
        // ShaderInvoke<1>::dispatch(luisa::span<const uint3>) (runtime/shader.h:185-187);
        // it is refused at capture by dispatch->is_multiple_dispatch()
        // (cuda_graph_ext.cpp:827-834). This is the row that decides whether a
        // per-frame submission can ever become one LC graph (§2.4, H0.3).
        std::array<::luisa::uint3, 2u> sizes{::luisa::uint3{4u, 1u, 1u},
                                             ::luisa::uint3{4u, 1u, 1u}};
        auto list = lcrt::CommandList::create(1u, 0u);
        list << (*d.m_fill_shader)(arena->view(), ::luisa::uint{8u}, 3.0f)
                    .dispatch(std::span<const ::luisa::uint3>{sizes});
        auto probe = d.probe_capture(std::move(list));

        RefusalResult row;
        row.kind     = RefusalKind::MultipleDispatch;
        row.accepted = probe.accepted();
        row.note     = detail::join_notes(
            "One command that launches twice, built with "
            "ShaderInvoke<1>::dispatch(span<const uint3>) (runtime/shader.h:185-187). "
            "LC refuses it (dispatch->is_multiple_dispatch(), "
            "cuda_graph_ext.cpp:827-834), so '2 statements in one graph' is not "
            "available as a workaround for the multi-dispatch refusal. Verdict: "
            + std::string{probe.graph_valid ? "graph created" : "invalid graph handle"}
            + (probe.exec_valid ? ", instantiated" : ", not instantiated"),
            probe.stages);
        rows.push_back(std::move(row));
    }

    // ---- 4. zero-sized dispatch -------------------------------------------
    {
        // dispatch(0): a zero thread count. LC's ordinary path ignores an empty
        // launch (cuda_graph_ext.cpp:851-853), which is exactly why the graph path
        // must refuse it -- turning it into "no node" would change what the list
        // means. Reported as measured, not assumed.
        auto list = lcrt::CommandList::create(1u, 0u);
        list << (*d.m_fill_shader)(arena->view(), ::luisa::uint{0u}, 0.0f).dispatch(0u);
        auto probe = d.probe_capture(std::move(list));

        RefusalResult row;
        row.kind     = RefusalKind::ZeroSizedDispatch;
        row.accepted = probe.accepted();
        row.note     = detail::join_notes(
            "LC refuses a zero-sized dispatch at capture "
            "(cuda_graph_ext.cpp:851-861: 'empty (zero-sized) dispatch cannot be "
            "captured into a graph'), whereas the plain submission path ignores it "
            "silently. dispatch() here submits one, so the two paths disagree by "
            "design. Verdict: "
            + std::string{probe.graph_valid ? "graph created" : "invalid graph handle"}
            + (probe.exec_valid ? ", instantiated" : ", not instantiated"),
            probe.stages);
        rows.push_back(std::move(row));
    }

    // ---- 5. empty command list --------------------------------------------
    {
        // The dangerous row: create_graph on an empty list returns
        // invalid_resource_handle with no warning at all (cuda_graph_ext.cpp:1063),
        // so a harness that only checked "did it take 0 ms?" would report a win for
        // a capture that produced nothing.
        lcrt::CommandList empty_list;
        auto              probe = d.probe_capture(std::move(empty_list.commit()).command_list());

        RefusalResult row;
        row.kind     = RefusalKind::EmptyCommandList;
        row.accepted = probe.accepted();
        row.note     = detail::join_notes(
            "create_graph on a list with no commands, no callbacks and no presents "
            "(CommandList::empty(), runtime/command_list.h:63) returns "
            "invalid_resource_handle silently: cuda_graph_ext.cpp:1063 "
            "'if (cmdlist.empty()) { return {invalid_handle, nullptr}; }'. A refused "
            "capture must never be recorded as a 0 ms graph -- hence the explicit "
            "nodes == 0 pre-check in graph_capture_fill(). Verdict: "
            + std::string{probe.graph_valid ? "graph created" : "invalid graph handle"}
            + (probe.exec_valid ? ", instantiated" : ", not instantiated"),
            probe.stages);
        rows.push_back(std::move(row));
    }

    // ---- 6. callback in the list ------------------------------------------
    {
        // CommandList::add_callback(luisa::move_only_function<void()>&&) exists in
        // this revision (runtime/command_list.h:47), so the content class is
        // constructible. Note what a list carrying a callback means for a graph: the
        // callback is not a node, it is drained on the *host* during capture
        // (cuda_graph_ext.cpp:1093 steal_callbacks + :1136 `for (auto &cb :
        // user_callbacks) { cb(); }`). The row therefore measures two things: whether
        // the capture is accepted, and whether the callback ran at capture time
        // (i.e. never again on replay).
        bool              callback_ran = false;
        lcrt::CommandList list         = lcrt::CommandList::create(1u, 1u);
        list << (*d.m_fill_shader)(arena->view(), ::luisa::uint{4u}, 4.0f).dispatch(4u);
        list.add_callback([&callback_ran] { callback_ran = true; });
        auto probe = d.probe_capture(std::move(list));

        RefusalResult row;
        row.kind     = RefusalKind::CallbackInList;
        row.accepted = probe.accepted();
        row.note     = detail::join_notes(
            "Accepted as content, but with a silent semantic difference: "
            "CudaGraphExt steals the list's callbacks and runs them once on the host "
            "during create_graph (cuda_graph_ext.cpp:1093 and :1136), so a replay "
            "carries no callback at all. Measured: callback ran during capture = "
            + std::string{callback_ran ? "yes" : "no"}
            + (probe.exec_valid ? ", graph instantiated" : ", not instantiated"),
            probe.stages);
        rows.push_back(std::move(row));
    }

    release(arena);

    // Nothing above launched anything -- a captured graph is never replayed by a
    // probe -- but the device must be idle when this returns, so the caller can
    // trust the counters it reads next. One counted sync, like the CUDA row.
    sync();
    return rows;
}

// ---- counters -------------------------------------------------------------

Counters& BenchDevice::counters() noexcept
{
    return m_impl->m_counters;
}

const Counters& BenchDevice::counters() const noexcept
{
    return m_impl->m_counters;
}

void BenchDevice::reset_counters()
{
    // Graph handles and the open batch survive a reset: this clears the
    // *measurement*, not the state.
    m_impl->m_counters = Counters{};
}

// ---- enum names -----------------------------------------------------------

std::string_view to_string(RefusalKind kind) noexcept
{
    switch(kind)
    {
        case RefusalKind::PrintingKernel:
            return "printing_kernel";
        case RefusalKind::IndirectDispatch:
            return "indirect_dispatch";
        case RefusalKind::MultipleDispatch:
            return "multiple_dispatch";
        case RefusalKind::ZeroSizedDispatch:
            return "zero_sized_dispatch";
        case RefusalKind::EmptyCommandList:
            return "empty_command_list";
        case RefusalKind::CallbackInList:
            return "callback_in_list";
        case RefusalKind::Count:
            break;
    }
    return "count";
}
}// namespace uipc::benchmark
