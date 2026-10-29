#pragma once
/********************************************************************
 * @file bench_device.h
 * @brief The one benchmark device abstraction (bench_plan.md B1, §6.1).
 *
 * Every benchmark scenario is written ONCE against this header and compiled
 * into two binaries, `uipc_bench_t0_cuda` and `uipc_bench_t0_luisa`. Each
 * binary links a *different* implementation TU of the same class:
 *
 *   apps/benchmarks/cuda/bench_device_cuda.cu     (nvcc, <<<>>> launches)
 *   apps/benchmarks/luisa/bench_device_luisa.cpp  (LC DSL, CommandList, graphs)
 *
 * The class is concrete (pimpl, no vtable): the hot loop of H0.1 issues
 * `dispatch()` a hundred thousand times through an ordinary call, so a virtual
 * call would land in the measurement. Because both implementations must satisfy
 * exactly the signatures below, "same scenario, two backends" is enforced by the
 * linker instead of by discipline (§6.1).
 *
 * One process per backend per case (bench_plan.md B3: a second Engine in one
 * process breaks the LC stream), so main() runs one case and exits.
 *********************************************************************/
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace uipc::benchmark
{
/// Opaque device buffer of float32 elements.
using BufferHandle = void*;
/// Opaque compiled kernel (LC Shader / CUDA function).
using KernelHandle = void*;

struct DeviceInfo
{
    std::string backend;          ///< "cuda" | "luisa"
    std::string device_name;      ///< e.g. "NVIDIA GeForce RTX 4060"
    std::string runtime_version;  ///< CUDA runtime version, or the LC revision
    std::string extra;            ///< driver / LC backend detail
};

/// Counters of bench_plan.md §6.5 that Tier 0 is responsible for.
struct Counters
{
    std::uint64_t dispatches = 0;  ///< submit units issued
    std::uint64_t commits    = 0;  ///< CommandList commits (CUDA: per launch)
    std::uint64_t syncs      = 0;
    double        sync_ms    = 0.0;
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t d2h_bytes  = 0;
    std::uint64_t batches_opened     = 0;  ///< begin_batch()/end_batch() pairs
    std::uint64_t graph_captures     = 0;
    std::uint64_t graph_instantiates = 0;
    std::uint64_t graph_updates      = 0;
    std::uint64_t graph_launches     = 0;
    std::uint64_t graph_rebuilds     = 0;
    std::uint64_t graph_refusals     = 0;
    double        graph_instantiate_ms = 0.0;
    double        graph_update_ms      = 0.0;
};

/// The content classes of the H0.6 refusal matrix (§2.4).
enum class RefusalKind
{
    PrintingKernel,  ///< kernel_cout / printf-style kernel
    IndirectDispatch,  ///< dispatch with an indirect (device-provided) launch size
    MultipleDispatch,   ///< more than one dispatch in one captured list
    ZeroSizedDispatch,  ///< dispatch(0)
    EmptyCommandList,   ///< a list with no content at all
    CallbackInList,     ///< a host callback recorded in the list
    Count
};

[[nodiscard]] std::string_view to_string(RefusalKind kind) noexcept;

/// One row of the H0.6 refusal matrix.
struct RefusalResult
{
    RefusalKind kind = RefusalKind::Count;
    bool accepted    = false;  ///< did the backend produce a launchable graph?
    std::string note;          ///< the backend's verbatim verdict / warning
};

/**
 * @brief Per-backend device: buffers, compiled kernels, submission, sync, graphs.
 *
 * Compile-once discipline: compile_*() is called once per case (outside the
 * measured window); dispatch() only submits.
 */
class BenchDevice
{
  public:
    BenchDevice();
    ~BenchDevice();
    BenchDevice(const BenchDevice&)            = delete;
    BenchDevice& operator=(const BenchDevice&) = delete;

    // ---- lifecycle ---------------------------------------------------------
    /// Bring the runtime up. `program_path` is argv[0]: LC derives its runtime
    /// directories from it, and a backend loaded as a module never sees it.
    void                            init(std::string_view program_path);
    void                            shutdown();
    [[nodiscard]] const DeviceInfo& info() const noexcept;

    // ---- memory ------------------------------------------------------------
    [[nodiscard]] BufferHandle alloc(std::size_t count);  ///< count floats
    void                       release(BufferHandle buffer);
    void upload(BufferHandle buffer, const float* host, std::size_t count);
    void download(BufferHandle buffer, float* host, std::size_t count);
    void zero(BufferHandle buffer, std::size_t count);

    // ---- kernels (identical scenario on both backends) ---------------------
    /// b[i] = v for i < n. One statement, one buffer argument plus a scalar.
    [[nodiscard]] KernelHandle compile_fill();
    /// b[i] += 1.0f for i < n. Real per-element work, used to separate enqueue
    /// cost from execution cost (H0.5) and to provide a checksum.
    [[nodiscard]] KernelHandle compile_increment();

    // ---- submission --------------------------------------------------------
    /// One statement = one submit unit. CUDA: one <<<>>>. LC: one dispatch.
    void dispatch(KernelHandle kernel, BufferHandle buffer, std::uint32_t threads, float v);

    /// Open / close a batch. CUDA has no batching layer, so both are counters
    /// only (documented asymmetry, §2.3). LC: opens one CommandList and commits
    /// it once.
    void begin_batch();
    void end_batch();

    /// Block until every submitted command finished (counted).
    void sync();

    // ---- graphs (H0.3, H0.6, H0.7, S8) ------------------------------------
    [[nodiscard]] bool graph_supported() const noexcept;

    /// Record `nodes` fill dispatches into one list and capture it.
    /// returns false when the backend refused (counters.graph_refusals bumped).
    bool graph_capture_fill(KernelHandle  kernel,
                            BufferHandle  buffer,
                            std::uint32_t threads,
                            float         v,
                            std::uint32_t nodes);

    /// One graph replay (counted).
    void graph_launch();

    /// Whole-graph update: re-capture an equivalent list and update the exec in
    /// place. Returns the measured milliseconds (also accumulated in counters).
    double graph_update_fill(KernelHandle  kernel,
                             BufferHandle  buffer,
                             std::uint32_t threads,
                             float         v,
                             std::uint32_t nodes);

    void graph_destroy();

    /// H0.6: pin the accepted/refused set of the graph API against this
    /// revision. CUDA's explicit cudaGraphAddKernelNode path has a different
    /// refusal set than LC's, so each implementation fills its own rows.
    [[nodiscard]] std::vector<RefusalResult> probe_refusals();

    // ---- counters ----------------------------------------------------------
    [[nodiscard]] Counters&       counters() noexcept;
    [[nodiscard]] const Counters& counters() const noexcept;
    void                          reset_counters();

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
}// namespace uipc::benchmark
