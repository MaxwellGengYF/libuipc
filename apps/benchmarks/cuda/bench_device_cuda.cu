/********************************************************************
 * @file bench_device_cuda.cu
 * @brief CUDA implementation of `uipc::benchmark::BenchDevice` (bench_plan.md
 *        §6.1 / B1, the CUDA half of the two-TU contract of common/bench_device.h).
 *
 * The file is deliberately self-contained: plain CUDA runtime API plus the four
 * kernels below. It does NOT link or call anything from `uipc_backend_cuda.dll`,
 * so `uipc_bench_t0_cuda.exe` depends on cudart only and keeps working while the
 * backend is being edited (the whole point of a Tier 0 harness).
 *
 * Three asymmetries against the Luisa TU are recorded here because they decide
 * how the resulting numbers may be read (§2.3, "the structural asymmetry the
 * benchmark must expose"):
 *
 *  1. Stream object. This file uses one dedicated non-blocking stream, created
 *     with cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking). The production
 *     CUDA backend uses the *legacy default stream* cudaStream_t{nullptr}
 *     (src/backends/cuda/cuda_tool/stream.h:43, cuda_tool::default_stream()).
 *     Legacy-default has implicit synchronisation semantics with every other
 *     blocking stream; a non-blocking explicit stream has none. The *number* of
 *     driver calls per launch is identical (one), so H0.1/H0.4 submit counts are
 *     comparable, but the stream object is not. It cannot even be substituted:
 *     CUDA forbids capturing the legacy default stream, so a real handle is
 *     mandatory the moment a case wants stream capture (H0.3/H0.6/H0.7 here
 *     build the DAG explicitly instead, but the stream type must not paint us
 *     into a corner). Recorded in DeviceInfo::extra as
 *     "stream=dedicated_non_blocking" so no record can be mislabelled.
 *  2. Zero JIT. compile_fill()/compile_increment() return the address of a
 *     __global__ stub that nvcc already compiled into this object file. There is
 *     nothing to compile at run time, which is precisely the difference the plan
 *     wants isolated from "Luisa is slow" reports (Q4; cookbook §7.37 rule 8:
 *     1376 s cold vs 3 s warm).
 *  3. No batching layer. 1 <<<>>> launch = 1 driver call, 383 launch sites in
 *     the backend, no batching layer (§2.3). begin_batch()/end_batch() therefore
 *     only count; H0.2's CUDA row means "there is nothing to batch", never
 *     "batching did not help".
 *********************************************************************/
#include <cuda_runtime.h>

#include <common/bench_device.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace uipc::benchmark::detail
{
// ---- launch shape ---------------------------------------------------------

/// Mirror of kDefaultBlockDim in src/backends/luisa/luisa_device/kernel.h:60, so
/// that a "1 thread" dispatch of H0.1 costs the same *GPU* work on both backends
/// and only the submission path differs.
inline constexpr unsigned kBlockDim = 256u;

inline constexpr std::size_t kFloatBytes = sizeof(float);

/// Total thread count -> grid, the shaping rule of the existing spmv timing case
/// (dispatch(n) takes a thread count, not a block count -- lc_optimize §5.7).
[[nodiscard]] inline unsigned grid_for(std::uint32_t threads) noexcept
{
    return (threads + kBlockDim - 1u) / kBlockDim;
}

using Clock = std::chrono::steady_clock;

[[nodiscard]] inline double ms_since(Clock::time_point start) noexcept
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// ---- error handling -------------------------------------------------------

[[nodiscard]] inline std::string cuda_error_text(cudaError_t e)
{
    char buf[320];
    std::snprintf(buf,
                  sizeof(buf),
                  "%s (%d): %s",
                  cudaGetErrorName(e),
                  static_cast<int>(e),
                  cudaGetErrorString(e));
    return buf;
}

[[noreturn]] inline void fatal(const char* what)
{
    std::fprintf(stderr, "FATAL bench_device_cuda: %s\n", what);
    std::fflush(stderr);
    std::abort();
}

/**
 * @brief Abort the process on a CUDA error.
 *
 * Why abort and not return an error: a silently failed CUDA call is measured as a
 * *fast* case, and §6.2 calls that out ("a wrong kernel that runs in 0 ms must
 * fail loudly"). A thrown exception would also be caught by the case runner and
 * turned into a "skipped" row, which is just as misleading.
 */
inline void check_cuda(cudaError_t e, const char* expr, const char* file, int line)
{
    if(e == cudaSuccess)
        return;
    std::fprintf(stderr,
                 "FATAL [%s:%d] %s\n      %s\n      aborting: a silently failed CUDA call would be "
                 "measured as a fast case (bench_plan.md § 6.2)\n",
                 file,
                 line,
                 expr,
                 cuda_error_text(e).c_str());
    std::fflush(stderr);
    std::abort();
}

/// Teardown paths warn instead of aborting: a destructor must not kill the run,
/// and the sticky error is cleared so the remaining destroys still happen.
inline void warn_cuda(cudaError_t e, const char* expr) noexcept
{
    if(e == cudaSuccess)
        return;
    std::fprintf(
        stderr, "WARNING bench_device_cuda %s: %s\n", expr, cuda_error_text(e).c_str());
    std::fflush(stderr);
    (void)cudaGetLastError();
}

// ---- kernels --------------------------------------------------------------
// These are compiled by nvcc into this .obj: compile_fill()/compile_increment()
// only hand out the address, there is no JIT, no PTX cache and no first-call
// compile that a warm-up could absorb (asymmetry 2 in the file header).

/// b[i] = v for i < n. One statement, one buffer argument plus a scalar (H0.1).
__global__ void fill_kernel(float* b, unsigned n, float v)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        b[i] = v;
}

/// b[i] += 1.0f for i < n. Real per-element work: separates enqueue cost from
/// execution cost (H0.5) and gives the checksum something to verify.
__global__ void increment_kernel(float* b, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        b[i] += 1.0f;
}

/// H0.6 printing-kernel row. `printf` writes to the device print buffer, which
/// the runtime drains at the next synchronisation -- nothing about a kernel node
/// inspects it, so an explicitly constructed graph has nothing to refuse.
__global__ void printing_kernel(float* b, unsigned n, float v)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        b[i] = v;
        printf("[probe] b[%u] = %.3f\n", i, static_cast<double>(b[i]));
    }
}

/// H0.6 callback row: the host function of a cudaGraphAddHostNode node.
void CUDART_CB host_probe_callback(void* user_data)
{
    if(auto* flag = static_cast<int*>(user_data))
        *flag = 1;
}

// ---- kernel handles -------------------------------------------------------

using KernelFn = void*;

[[nodiscard]] inline KernelFn fill_kernel_fn() noexcept
{
    // The host address of a __global__ stub is exactly what
    // cudaKernelNodeParams::func and cudaLaunchKernel expect; the cast through the
    // concrete function type keeps nvcc from warning about func->object pointers.
    return reinterpret_cast<KernelFn>(
        reinterpret_cast<void (*)(float*, unsigned, float)>(fill_kernel));
}

[[nodiscard]] inline KernelFn increment_kernel_fn() noexcept
{
    return reinterpret_cast<KernelFn>(
        reinterpret_cast<void (*)(float*, unsigned)>(increment_kernel));
}

/// Only the H0.6 printing-kernel row needs this one, so it is not part of
/// is_known_kernel() and dispatch() cannot reach it by accident.
[[nodiscard]] inline KernelFn printing_kernel_fn() noexcept
{
    return reinterpret_cast<KernelFn>(
        reinterpret_cast<void (*)(float*, unsigned, float)>(printing_kernel));
}

[[nodiscard]] inline bool is_known_kernel(KernelHandle k) noexcept
{
    return k == fill_kernel_fn() || k == increment_kernel_fn();
}

// ---- graph node plumbing --------------------------------------------------

/**
 * @brief Baked arguments of one graph kernel node.
 *
 * `params` is the `void*[]` handed to cudaGraphAddKernelNode; its three entries
 * point at the fields directly above it, so the whole triple has to sit at a
 * stable address for as long as the node exists. CUDA copies the parameter
 * *values* at node-creation time, so this is belt-and-braces -- but the storage
 * below is reserved once per graph (never reallocated), freed only after
 * cudaGraphDestroy(), and it makes §2.4's "everything is baked" contract true in
 * this file instead of merely believed.
 */
struct NodeArg
{
    float*   buffer    = nullptr;
    unsigned count     = 0u;
    float    value     = 0.0f;
    void*    params[3] = {nullptr, nullptr, nullptr};

    void bind(float* b, unsigned n, float v) noexcept
    {
        buffer    = b;
        count     = n;
        value     = v;
        params[0] = &buffer;
        params[1] = &count;
        params[2] = &value;
    }
};

/// `arg` is taken by non-const reference on purpose: kernelParams is a `void*[]`
/// that the graph reads through, so handing out a const-qualified pointer would only
/// need a const_cast one line later.
[[nodiscard]] inline cudaKernelNodeParams kernel_node_params(KernelFn func,
                                                             NodeArg& arg,
                                                             unsigned grid,
                                                             unsigned block) noexcept
{
    cudaKernelNodeParams np{};
    np.func           = func;
    np.gridDim        = dim3(grid, 1u, 1u);
    np.blockDim       = dim3(block, 1u, 1u);
    np.sharedMemBytes = 0u;
    np.kernelParams   = arg.params;
    np.extra          = nullptr;
    return np;
}

[[nodiscard]] inline std::string_view update_result_name(cudaGraphExecUpdateResult r) noexcept
{
    switch(r)
    {
        case cudaGraphExecUpdateSuccess:
            return "cudaGraphExecUpdateSuccess";
        case cudaGraphExecUpdateError:
            return "cudaGraphExecUpdateError";
        case cudaGraphExecUpdateErrorTopologyChanged:
            return "cudaGraphExecUpdateErrorTopologyChanged";
        case cudaGraphExecUpdateErrorNodeTypeChanged:
            return "cudaGraphExecUpdateErrorNodeTypeChanged";
        case cudaGraphExecUpdateErrorFunctionChanged:
            return "cudaGraphExecUpdateErrorFunctionChanged";
        case cudaGraphExecUpdateErrorParametersChanged:
            return "cudaGraphExecUpdateErrorParametersChanged";
        case cudaGraphExecUpdateErrorNotSupported:
            return "cudaGraphExecUpdateErrorNotSupported";
        case cudaGraphExecUpdateErrorUnsupportedFunctionChange:
            return "cudaGraphExecUpdateErrorUnsupportedFunctionChange";
        case cudaGraphExecUpdateErrorAttributesChanged:
            return "cudaGraphExecUpdateErrorAttributesChanged";
    }
    return "cudaGraphExecUpdateResult(unknown)";
}

/// "12.9" from 12090, "595.71" from 595710. Both getters encode
/// (1000 * major + 10 * minor), so the same formatter serves both.
[[nodiscard]] inline std::string version_string(int v)
{
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

// ---- H0.6 probe plumbing --------------------------------------------------

/// One row of the refusal matrix under construction: every stage is recorded
/// verbatim so the note quotes what CUDA actually said.
struct ProbeReport
{
    bool        accepted = false;
    std::string note;

    /// Record one stage. Returns false as soon as a stage failed.
    bool stage(const char* what, cudaError_t e)
    {
        if(!note.empty())
            note += "; ";
        if(e == cudaSuccess)
        {
            note += what;
            note += " -> cudaSuccess";
        }
        else
        {
            note += what;
            note += " -> ";
            note += cuda_error_text(e);
        }
        return e == cudaSuccess;
    }
};

/**
 * @brief create -> build -> instantiate -> replay one throw-away graph.
 *
 * A CUDA error is sticky (the next call keeps returning the first one), so the
 * chain stops at the first failure instead of echoing that failure four times.
 * `accepted` demands a successful *replay*, not merely an instantiation: H0.6's
 * whole purpose is that a refused or silently empty capture must not pass at 0 ms.
 *
 * @param build  bool(cudaGraph_t, ProbeReport&) -- appends its own stages.
 */
template <typename Build>
ProbeReport probe_graph(cudaStream_t stream, Build&& build)
{
    ProbeReport     r;
    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t exec  = nullptr;

    bool ok =
        r.stage("cudaGraphCreate", cudaGraphCreate(&graph, 0u)) && build(graph, r)
        && r.stage("cudaGraphInstantiate", cudaGraphInstantiate(&exec, graph, 0ull));
    if(ok)
    {
        ok = r.stage("cudaGraphLaunch", cudaGraphLaunch(exec, stream))
             && r.stage("cudaStreamSynchronize", cudaStreamSynchronize(stream));
    }

    (void)cudaGetLastError();  // drop the sticky error before tearing down
    if(exec != nullptr)
        (void)cudaGraphExecDestroy(exec);
    if(graph != nullptr)
        (void)cudaGraphDestroy(graph);

    r.accepted = ok;
    return r;
}
}  // namespace uipc::benchmark::detail

#define UIPC_BENCH_CUDA_CHECK(EXPR)                                            \
    ::uipc::benchmark::detail::check_cuda((EXPR), #EXPR, __FILE__, __LINE__)

namespace uipc::benchmark
{
using detail::Clock;
using detail::kBlockDim;
using detail::kFloatBytes;
using detail::ms_since;

// ---- pimpl ----------------------------------------------------------------

struct BenchDevice::Impl
{
    cudaStream_t m_stream = nullptr;
    DeviceInfo   m_info;
    Counters     m_counters;
    bool         m_initialized = false;

    // Graph state (H0.3/H0.6/H0.7). Both handles are raw and unreferenced, so the
    // destroy order below is mandatory (cuda_graph_ext.h:147 has the same rule).
    cudaGraph_t     m_graph = nullptr;
    cudaGraphExec_t m_exec  = nullptr;
    /// One NodeArg per node of m_graph. Reserved once, never reallocated: the node
    /// ABI points into this storage, so a reallocation would dangle every node added
    /// before it. Cleared only after the graph is destroyed.
    std::vector<detail::NodeArg> m_graph_args;

    bool m_batch_open = false;

    void fill_info();
    void create_stream();
    void destroy_stream() noexcept;

    /// Guarded stream for every command path. A silent nullptr would mean the legacy
    /// default stream -- a different object, with different synchronisation rules, from
    /// the one this file claims to measure.
    [[nodiscard]] cudaStream_t stream() const noexcept;

    /// Add `nodes` independent fill nodes to `graph`, baking one NodeArg each.
    /// Returns the first failing cudaError_t; `args` must be reserved by this
    /// function before the first node is added (see m_graph_args).
    cudaError_t add_fill_nodes(cudaGraph_t                   graph,
                               detail::KernelFn              func,
                               float*                        buffer,
                               std::uint32_t                 threads,
                               float                         v,
                               std::uint32_t                 nodes,
                               std::vector<detail::NodeArg>& args);
};

void BenchDevice::Impl::create_stream()
{
    // NOT cudaStream_t{nullptr}: see asymmetry 1 in the file header. CUDA forbids
    // capturing the legacy default stream, and implicit legacy-stream
    // synchronisation would silently serialise a benchmark that is trying to
    // measure enqueue cost.
    UIPC_BENCH_CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
}

cudaStream_t BenchDevice::Impl::stream() const noexcept
{
    if(m_stream == nullptr)
        detail::fatal("BenchDevice command issued before init() or after shutdown()");
    return m_stream;
}

void BenchDevice::Impl::destroy_stream() noexcept
{
    if(m_stream == nullptr)
        return;
    detail::warn_cuda(cudaStreamDestroy(m_stream), "cudaStreamDestroy");
    m_stream = nullptr;
}

void BenchDevice::Impl::fill_info()
{
    int device = 0;
    UIPC_BENCH_CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    UIPC_BENCH_CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    int runtime_version = 0;
    int driver_version  = 0;
    UIPC_BENCH_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    UIPC_BENCH_CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    m_info.backend     = "cuda";
    m_info.device_name = props.name;
    // Patch level of the runtime is not encoded by cudaRuntimeGetVersion(), so the
    // build-time CUDART_VERSION is reported alongside it in `extra`.
    m_info.runtime_version = detail::version_string(runtime_version);

    char buf[320];
    std::snprintf(buf,
                  sizeof(buf),
                  "driver=%s cc=%d.%d sms=%d cudart_build=%d.%d stream=dedicated_non_blocking",
                  detail::version_string(driver_version).c_str(),
                  props.major,
                  props.minor,
                  props.multiProcessorCount,
                  CUDART_VERSION / 1000,
                  (CUDART_VERSION % 1000) / 10);
    m_info.extra = buf;
}

cudaError_t BenchDevice::Impl::add_fill_nodes(cudaGraph_t      graph,
                                              detail::KernelFn func,
                                              float*           buffer,
                                              std::uint32_t    threads,
                                              float            v,
                                              std::uint32_t    nodes,
                                              std::vector<detail::NodeArg>& args)
{
    args.clear();
    args.reserve(nodes);  // storage stability is load-bearing, not an optimisation

    const unsigned grid = detail::grid_for(threads);
    for(std::uint32_t i = 0; i < nodes; ++i)
    {
        detail::NodeArg& arg = args.emplace_back();
        arg.bind(buffer, static_cast<unsigned>(threads), v);

        const cudaKernelNodeParams np =
            detail::kernel_node_params(func, arg, grid, detail::kBlockDim);

        cudaGraphNode_t node = nullptr;
        // No dependencies: all nodes write the same value to the same buffer, so an
        // independent (concurrent) schedule is legal. LC builds its DAG from declared
        // READ/WRITE usages and lets independent nodes overlap (§2.4 "Scheduling");
        // this is the CUDA equivalent of that contract.
        const cudaError_t e = cudaGraphAddKernelNode(&node, graph, nullptr, 0u, &np);
        if(e != cudaSuccess)
            return e;
        // The handle is not kept on purpose: graph_capture_fill() re-reads the node
        // count from the graph itself, and graph_update_fill() builds a throw-away
        // graph whose handles must not end up in the instance state.
        node = nullptr;
    }
    return cudaSuccess;
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
    // `program_path` is argv[0]. LC derives its runtime/cache directories from it;
    // CUDA does not, so it is intentionally unused here (documented, not forgotten).
    (void)program_path;

    Impl& d = *m_impl;
    if(d.m_initialized)
        return;

    // Device 0: one process per backend per case (B3), so there is no device selection
    // to negotiate and CUDA_VISIBLE_DEVICES is the only knob.
    UIPC_BENCH_CUDA_CHECK(cudaSetDevice(0));

    d.create_stream();
    d.fill_info();
    d.m_initialized = true;

    std::printf("[bench_device_cuda] %s | runtime %s | %s\n",
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

    graph_destroy();
    d.destroy_stream();
    d.m_initialized = false;
}

const DeviceInfo& BenchDevice::info() const noexcept
{
    return m_impl->m_info;
}

// ---- memory ---------------------------------------------------------------

BufferHandle BenchDevice::alloc(std::size_t count)
{
    if(count == 0u)
        return nullptr;  // cudaMalloc(0) is cudaErrorInvalidValue; nothing to point at.

    void* p = nullptr;
    // cudaMalloc is implicitly synchronising (legacy behaviour): allocation belongs in
    // setup, never inside a measured window. Buffer pools are Tier 2 work (§6.5).
    UIPC_BENCH_CUDA_CHECK(cudaMalloc(&p, count * kFloatBytes));
    return p;
}

void BenchDevice::release(BufferHandle buffer)
{
    if(buffer == nullptr)
        return;
    UIPC_BENCH_CUDA_CHECK(cudaFree(buffer));
}

void BenchDevice::upload(BufferHandle buffer, const float* host, std::size_t count)
{
    if(count == 0u)
        return;
    Impl& d = *m_impl;
    UIPC_BENCH_CUDA_CHECK(cudaMemcpyAsync(
        buffer, host, count * kFloatBytes, cudaMemcpyHostToDevice, d.stream()));
    d.m_counters.h2d_bytes += count * kFloatBytes;

    // The CUDA backend uploads straight from the user pointer (§2.3 "Upload path"),
    // and that pointer is usually a host temporary, so the copy must be complete
    // before this returns -- one *counted* sync, deliberately not a hidden one.
    sync();
}

void BenchDevice::download(BufferHandle buffer, float* host, std::size_t count)
{
    if(count == 0u)
        return;
    Impl& d = *m_impl;
    // cudaMemcpyAsync(dst, src, ...): a D2H copy takes the HOST pointer as the
    // destination and the device pointer as the source. Getting this backwards is a
    // cudaErrorInvalidValue at run time, not a compile error.
    UIPC_BENCH_CUDA_CHECK(cudaMemcpyAsync(
        host, buffer, count * kFloatBytes, cudaMemcpyDeviceToHost, d.stream()));
    d.m_counters.d2h_bytes += count * kFloatBytes;
    sync();
}

void BenchDevice::zero(BufferHandle buffer, std::size_t count)
{
    if(count == 0u)
        return;
    // Async memset on the dedicated stream: zeroing is part of the submit stream, so
    // it is measured like any other command and never as a device-wide flush.
    UIPC_BENCH_CUDA_CHECK(cudaMemsetAsync(buffer, 0, count * kFloatBytes, m_impl->stream()));
}

// ---- kernels --------------------------------------------------------------

KernelHandle BenchDevice::compile_fill()
{
    // Nothing to compile: nvcc compiled it. Zero JIT is the point of this row of the
    // comparison (asymmetry 2 in the file header), not an omission.
    return detail::fill_kernel_fn();
}

KernelHandle BenchDevice::compile_increment()
{
    return detail::increment_kernel_fn();
}

// ---- submission -----------------------------------------------------------

void BenchDevice::dispatch(KernelHandle kernel, BufferHandle buffer, std::uint32_t threads, float v)
{
    Impl& d = *m_impl;

    // One statement = one submit unit = one <<<>>> = one driver call. There is no
    // batching layer to commit into, so dispatches and commits stay in lockstep on
    // this backend (§2.3; §6.5 "commits: n/a, 1 per launch"). The invariant
    // commits == dispatches IS the CUDA row of the story.
    ++d.m_counters.dispatches;
    ++d.m_counters.commits;

    if(threads == 0u)
    {
        // A zero-sized grid is rejected by CUDA, so nothing is launched. This is a
        // refusal-free no-op: the caller asked for a statement, the counters record
        // that it was submitted, and no graph_refusals is touched (that counter is
        // for captures the *backend* refused, H0.6). Note the deliberate, tiny
        // over-count: commits is +1 with no driver call behind it, kept because a
        // caller-issued statement is what §6.5 counts.
        return;
    }

    const unsigned grid = detail::grid_for(threads);
    if(kernel == detail::fill_kernel_fn())
    {
        detail::fill_kernel<<<grid, kBlockDim, 0, d.stream()>>>(
            static_cast<float*>(buffer), static_cast<unsigned>(threads), v);
    }
    else if(kernel == detail::increment_kernel_fn())
    {
        detail::increment_kernel<<<grid, kBlockDim, 0, d.stream()>>>(
            static_cast<float*>(buffer), static_cast<unsigned>(threads));
    }
    else
    {
        detail::fatal(
            "dispatch(): unknown kernel handle (not from compile_fill()/"
            "compile_increment())");
    }

    // No cudaGetLastError() here on purpose: it is one extra host call inside the
    // 100 000-iteration loop that H0.1 times. A launch that was rejected is a sticky
    // error, so the next sync() reports it and aborts the run (§6.2) -- loud, just
    // not in the measured path.
}

void BenchDevice::begin_batch()
{
    // Counted only. CUDA has no batching layer to open: "1 <<<>>> launch = 1 driver
    // call, 383 launch sites, no batching layer" (§2.3). H0.2's CUDA row therefore
    // says "nothing to batch", which must not be read as "batching did not help".
    ++m_impl->m_counters.batches_opened;
    m_impl->m_batch_open = true;
}

void BenchDevice::end_batch()
{
    if(!m_impl->m_batch_open)
        detail::fatal("end_batch() without begin_batch()");
    m_impl->m_batch_open = false;
}

void BenchDevice::sync()
{
    Impl& d = *m_impl;

    const auto t0 = Clock::now();
    // cudaStreamSynchronize also returns an asynchronous launch failure, which is what
    // lets dispatch() stay free of a per-launch error check.
    UIPC_BENCH_CUDA_CHECK(cudaStreamSynchronize(d.stream()));
    d.m_counters.sync_ms += ms_since(t0);
    ++d.m_counters.syncs;
}

// ---- graphs ---------------------------------------------------------------

bool BenchDevice::graph_supported() const noexcept
{
    // Explicit cudaGraphAddKernelNode construction needs no capture-capable stream and
    // no device-side launch support, so it is available wherever a graph can be
    // instantiated -- i.e. always on a CUDA device. (The Luisa TU has to answer
    // "is device.extension<CudaGraphExt>() non-null?" instead; §2.4.)
    return true;
}

bool BenchDevice::graph_capture_fill(KernelHandle  kernel,
                                     BufferHandle  buffer,
                                     std::uint32_t threads,
                                     float         v,
                                     std::uint32_t nodes)
{
    Impl& d = *m_impl;
    if(!d.m_initialized)
        detail::fatal("graph_capture_fill() before init()");
    if(!detail::is_known_kernel(kernel))
        detail::fatal("graph_capture_fill(): unknown kernel handle");

    // A second capture replaces the first; destroy it in the defined order first.
    graph_destroy();

    cudaGraph_t graph  = nullptr;
    auto        refuse = [&](const char* what, cudaError_t e)
    {
        ++d.m_counters.graph_refusals;
        std::fprintf(stderr,
                     "graph_capture_fill: %s failed: %s\n",
                     what,
                     detail::cuda_error_text(e).c_str());
        std::fflush(stderr);
        (void)cudaGetLastError();  // sticky: clear it so teardown still runs
        if(graph != nullptr)
            (void)cudaGraphDestroy(graph);
        d.m_graph_args.clear();
        return false;
    };

    // Explicit DAG construction, never stream capture: this mirrors LC's
    // CudaGraphExt contract ("never by stream capture, which would linearize", §2.4)
    // and is what makes the two backends' graph costs comparable at all.
    cudaError_t e = cudaGraphCreate(&graph, 0u);
    if(e != cudaSuccess)
        return refuse("cudaGraphCreate", e);

    // threads == 0 makes grid == 0 and CUDA rejects the node, so a zero-sized capture
    // is refused (graph_refusals) instead of producing a graph that "runs" in 0 ms.
    e = d.add_fill_nodes(graph, kernel, static_cast<float*>(buffer), threads, v, nodes, d.m_graph_args);
    if(e != cudaSuccess)
        return refuse("cudaGraphAddKernelNode", e);

    // H0.6's discipline applied to the measured path (§2.4: "an empty command list
    // returns invalid silently ... a silently empty capture must fail the benchmark,
    // not pass at 0 ms"). Ask the graph how many nodes it really has instead of
    // trusting the loop that just ran.
    std::size_t real_nodes = 0;
    e                      = cudaGraphGetNodes(graph, nullptr, &real_nodes);
    if(e != cudaSuccess)
        return refuse("cudaGraphGetNodes(count)", e);
    if(real_nodes != nodes)
    {
        char what[160];
        std::snprintf(what,
                      sizeof(what),
                      "built %zu nodes, requested %u",
                      real_nodes,
                      static_cast<unsigned>(nodes));
        std::fprintf(stderr, "graph_capture_fill: %s\n", what);
        return refuse("node-count verification", cudaErrorInvalidValue);
    }

    cudaGraphExec_t exec = nullptr;
    // The instantiate is the one-off cost H0.3 amortises over 20 replays, and the cost
    // H0.7 pays per topology change, so it is timed separately (§6.5).
    const auto t0               = Clock::now();
    e                           = cudaGraphInstantiate(&exec, graph, 0ull);
    const double instantiate_ms = ms_since(t0);
    if(e != cudaSuccess)
        return refuse("cudaGraphInstantiate", e);

    d.m_graph = graph;
    d.m_exec  = exec;
    ++d.m_counters.graph_captures;
    ++d.m_counters.graph_instantiates;
    d.m_counters.graph_instantiate_ms += instantiate_ms;
    return true;
}

void BenchDevice::graph_launch()
{
    Impl& d = *m_impl;
    if(d.m_exec == nullptr)
        detail::fatal("graph_launch() without a captured graph");

    // Host-side launch into the same dedicated stream, so replays serialise with any
    // preceding plain dispatch: "the graph boundary is a host boundary" (§2.4).
    UIPC_BENCH_CUDA_CHECK(cudaGraphLaunch(d.m_exec, d.stream()));
    ++d.m_counters.graph_launches;
}

double BenchDevice::graph_update_fill(KernelHandle  kernel,
                                      BufferHandle  buffer,
                                      std::uint32_t threads,
                                      float         v,
                                      std::uint32_t nodes)
{
    Impl& d = *m_impl;
    if(d.m_exec == nullptr)
        detail::fatal("graph_update_fill() without a captured graph");
    if(!detail::is_known_kernel(kernel))
        detail::fatal("graph_update_fill(): unknown kernel handle");

    const auto t0 = Clock::now();

    // 1. Re-capture an equivalent graph: same topology, same launch shape, same value
    // -- but baked into *fresh* host arg storage. That is exactly the audit question
    // of H0.7: if cudaGraphExecUpdate compared the raw pointer bytes of kernelParams,
    // this update would fail with ErrorParametersChanged even though nothing semantic
    // changed. If it succeeds, whole-graph update reproduces the baked layout and
    // per-epoch re-arguments are affordable.
    cudaGraph_t                  graph = nullptr;
    std::vector<detail::NodeArg> args;  // must outlive `graph`
    cudaError_t                  e = cudaGraphCreate(&graph, 0u);
    if(e == cudaSuccess)
        e = d.add_fill_nodes(graph, kernel, static_cast<float*>(buffer), threads, v, nodes, args);

    const double rebuild_ms = ms_since(t0);

    // 2. Update the instantiated graph in place. CUDA 12.9 takes the result as a
    // struct (cudaGraphExecUpdateResultInfo{result, errorNode, errorFromNode}); the
    // older (hGraphExec, hGraph, &errorNode, &result) form is gone from
    // cuda_runtime_api.h:13532, so the audit fields are read from the struct.
    double                        exec_update_ms = 0.0;
    cudaGraphExecUpdateResultInfo result{};
    bool                          update_attempted = false;
    bool                          ok               = (e == cudaSuccess);
    if(ok)
    {
        update_attempted = true;
        const auto tu    = Clock::now();
        e                = cudaGraphExecUpdate(d.m_exec, graph, &result);
        exec_update_ms   = ms_since(tu);
        // A failed update returns cudaErrorGraphExecUpdateFailure, but a *successful*
        // call can still carry a non-Success result code, so both are checked: only
        // the pair proves the new graph reproduced the instantiated layout (H0.7).
        ok = (e == cudaSuccess) && (result.result == cudaGraphExecUpdateSuccess);
    }

    // The exec owns its own copy of the parameters, so the temporary graph (and its arg
    // storage) can go now; m_graph/m_graph_args are untouched by an update.
    (void)cudaGetLastError();
    if(graph != nullptr)
        (void)cudaGraphDestroy(graph);
    args.clear();

    const double total_ms = ms_since(t0);
    // Printed as its own string, not inline in the call: a temporary std::string inside
    // a printf argument list is legal but one refactor away from a dangling c_str().
    const std::string result_name =
        update_attempted ? std::string(detail::update_result_name(result.result)) :
                           std::string{"not attempted (re-capture failed)"};

    // The split is printed because it answers two different questions: rebuild_ms is
    // "what a topology change costs", exec_update_ms is "what an update costs"
    // (H0.7 measures destroy_exec + instantiate against exactly this).
    std::printf(
        "graph_update_fill: nodes=%u rebuild=%.3f ms cudaGraphExecUpdate=%.3f ms "
        "result=%s errorNode=%s errorFromNode=%s\n",
        static_cast<unsigned>(nodes),
        rebuild_ms,
        exec_update_ms,
        result_name.c_str(),
        result.errorNode == nullptr ? "null" : "set",
        result.errorFromNode == nullptr ? "null" : "set");
    std::fflush(stdout);

    if(!ok)
    {
        ++d.m_counters.graph_refusals;
        std::fprintf(stderr,
                     "graph_update_fill: update refused (%s)%s%s\n",
                     detail::cuda_error_text(e).c_str(),
                     (update_attempted && result.result != cudaGraphExecUpdateSuccess) ?
                         " -- cudaGraphExecUpdateResult != Success, i.e. the new graph "
                         "does NOT reproduce the instantiated layout (H0.7 audit finding)" :
                         "",
                     result.errorNode == nullptr ? "" : ", errorNode reported");
        std::fflush(stderr);
        // The docs guarantee an update is atomic: on failure the exec still runs the
        // previous bake, so the graph stays usable and the counters stay honest.
    }

    // graph_updates/graph_update_ms carry the whole re-argument path (re-capture +
    // update), which is what LC's update(exec, CommandList&&) costs: "re-capture +
    // cuGraphExecUpdate" (§2.4). The split above is informational; this is the
    // comparable number, and the returned value is the accumulated one.
    ++d.m_counters.graph_updates;
    d.m_counters.graph_update_ms += total_ms;
    ++d.m_counters.graph_rebuilds;  // an update is a rebuild that did not re-instantiate
    return total_ms;
}

void BenchDevice::graph_destroy()
{
    Impl& d = *m_impl;

    // Order is mandatory (the same rule as cuda_graph_ext.h:147): the stream must be
    // idle before the exec is destroyed, and the arg storage only after the graph
    // that points into it is gone.
    if(d.m_stream != nullptr)
        detail::warn_cuda(cudaStreamSynchronize(d.m_stream),
                          "cudaStreamSynchronize(graph_destroy)");

    if(d.m_exec != nullptr)
    {
        detail::warn_cuda(cudaGraphExecDestroy(d.m_exec), "cudaGraphExecDestroy");
        d.m_exec = nullptr;
    }
    if(d.m_graph != nullptr)
    {
        detail::warn_cuda(cudaGraphDestroy(d.m_graph), "cudaGraphDestroy");
        d.m_graph = nullptr;
    }
    d.m_graph_args.clear();
}

// ---- H0.6 refusal matrix --------------------------------------------------

std::vector<RefusalResult> BenchDevice::probe_refusals()
{
    Impl& d = *m_impl;
    if(!d.m_initialized)
        detail::fatal("probe_refusals() before init()");

    std::vector<RefusalResult> rows;
    rows.reserve(static_cast<std::size_t>(RefusalKind::Count));

    // One tiny arena for every row; a fresh cudaMalloc per row would measure the
    // allocator, and some probes never reach a launch at all.
    constexpr std::uint32_t kArenaFloats = 256u;
    auto*                   arena = static_cast<float*>(alloc(kArenaFloats));

    // ---- 1. printing kernel ------------------------------------------------
    {
        // CUDA: nothing refuses it. A kernel node is a function pointer plus args; a
        // `printf` inside it only writes the device print buffer, which the runtime
        // drains at the next sync. (LC refuses kernel_cout at capture time because its
        // capture path inspects the kernel IR, §2.4 "What is refused".)
        detail::NodeArg     arg;
        detail::ProbeReport probe = detail::probe_graph(
            d.m_stream,
            [&](cudaGraph_t graph, detail::ProbeReport& rep)
            {
                arg.bind(arena, 1u, 1.0f);
                const cudaKernelNodeParams np =
                    detail::kernel_node_params(detail::printing_kernel_fn(), arg, 1u, 1u);
                cudaGraphNode_t node = nullptr;
                return rep.stage("cudaGraphAddKernelNode(printf kernel)",
                                 cudaGraphAddKernelNode(&node, graph, nullptr, 0u, &np));
            });
        RefusalResult row;
        row.kind     = RefusalKind::PrintingKernel;
        row.accepted = probe.accepted;
        row.note =
            "CUDA accepts it: a kernel node is only {func, gridDim, blockDim, "
            "sharedMemBytes, kernelParams, extra}, and printf is a device-side side "
            "effect no API inspects. LC refuses kernel_cout at capture because its "
            "capture path looks at the kernel IR. | "
            + probe.note;
        rows.push_back(std::move(row));
    }

    // ---- 2. indirect dispatch ---------------------------------------------
    {
        // CUDA has no way to express it, which is stronger than a refusal: there is no
        // error to quote. cudaKernelNodeParams (driver_types.h:3330-3337) is exactly
        // {func, gridDim, blockDim, sharedMemBytes, kernelParams, extra},
        // cudaGraphKernelNodeSetParams() takes the same struct, and
        // cudaGraphKernelNodeGetParams() below returns only those fields -- verified
        // on the live node, not assumed.
        detail::NodeArg     arg;
        std::string         readback;
        detail::ProbeReport probe = detail::probe_graph(
            d.m_stream,
            [&](cudaGraph_t graph, detail::ProbeReport& rep)
            {
                arg.bind(arena, 4u, 2.0f);
                const cudaKernelNodeParams np =
                    detail::kernel_node_params(detail::fill_kernel_fn(), arg, 1u, 1u);
                cudaGraphNode_t node = nullptr;
                if(!rep.stage("cudaGraphAddKernelNode",
                              cudaGraphAddKernelNode(&node, graph, nullptr, 0u, &np)))
                    return false;
                // The nearest thing to "try setting it": read the node's ABI back and
                // show that nothing indirect survived, because nothing indirect exists.
                cudaKernelNodeParams back{};
                const bool ok = rep.stage("cudaGraphKernelNodeGetParams",
                                          cudaGraphKernelNodeGetParams(node, &back));
                char       buf[256];
                std::snprintf(buf,
                              sizeof(buf),
                              "read-back: grid=(%u,%u,%u) block=(%u,%u,%u) extra=%s "
                              "kernelParams=%s -- six fields, no indirect-launch slot",
                              back.gridDim.x,
                              back.gridDim.y,
                              back.gridDim.z,
                              back.blockDim.x,
                              back.blockDim.y,
                              back.blockDim.z,
                              back.extra == nullptr ? "null" : "set",
                              back.kernelParams == nullptr ? "null" : "set");
                readback = buf;
                return ok;
            });
        RefusalResult row;
        row.kind = RefusalKind::IndirectDispatch;
        // Deliberately false even though the probe graph itself is legal: the *content
        // class* cannot be expressed on this backend, so a scenario needing it cannot
        // run. Quoting a successful instantiate as "accepted" would misreport a
        // capability gap as a feature.
        row.accepted = false;
        row.note =
            "Refused by absence, not by an error: the graph node ABI has no "
            "device-provided launch-size field. cudaKernelNodeParams "
            "(driver_types.h:3330) and cudaGraphKernelNodeSetParams() expose only "
            "{func, gridDim, blockDim, sharedMemBytes, kernelParams, extra}. LC "
            "refuses indirect dispatch at capture and returns invalid_handle; CUDA "
            "silently has no parameter for it, so the failure shows up as a compile "
            "error, not a runtime code. Consequence for optimize_plan.md OP-A3/C5: a "
            "graph can only replay a size baked at build time, so a size change is a "
            "rebuild, never an update. | "
            + readback + " | " + probe.note;
        rows.push_back(std::move(row));
    }

    // ---- 3. multiple dispatch ---------------------------------------------
    {
        // Two independent nodes in one graph: the normal case, not an exception.
        std::array<detail::NodeArg, 2> args;
        detail::ProbeReport            probe = detail::probe_graph(
            d.m_stream,
            [&](cudaGraph_t graph, detail::ProbeReport& rep)
            {
                for(std::size_t i = 0; i < args.size(); ++i)
                {
                    args[i].bind(arena, 4u, static_cast<float>(i) + 1.0f);
                    const cudaKernelNodeParams np = detail::kernel_node_params(
                        detail::fill_kernel_fn(), args[i], 1u, 1u);
                    cudaGraphNode_t node = nullptr;
                    if(!rep.stage("cudaGraphAddKernelNode",
                                  cudaGraphAddKernelNode(&node, graph, nullptr, 0u, &np)))
                        return false;
                }
                char buf[96];
                std::snprintf(buf, sizeof(buf), "added %zu independent kernel nodes", args.size());
                if(!rep.note.empty())
                    rep.note += "; ";
                rep.note += buf;
                return true;
            });
        RefusalResult row;
        row.kind     = RefusalKind::MultipleDispatch;
        row.accepted = probe.accepted;
        row.note =
            "CUDA accepts 2+ nodes: that is the whole point of a graph, and "
            "independent nodes may even run concurrently (§ 2.4 Scheduling). LC "
            "refuses more than one dispatch in one captured list -- which is exactly "
            "why libuipc's per-frame submission cannot be batched into a single LC "
            "graph today. | "
            + probe.note;
        rows.push_back(std::move(row));
    }

    // ---- 4. zero-sized dispatch -------------------------------------------
    {
        // What LC's dispatch(0) turns into: gridDimX = 0. Where CUDA rejects it is
        // reported as it happens rather than assumed.
        detail::NodeArg     arg;
        detail::ProbeReport probe = detail::probe_graph(
            d.m_stream,
            [&](cudaGraph_t graph, detail::ProbeReport& rep)
            {
                arg.bind(arena, 0u, 0.0f);
                const cudaKernelNodeParams np = detail::kernel_node_params(
                    detail::fill_kernel_fn(), arg, 0u, detail::kBlockDim);
                cudaGraphNode_t node = nullptr;
                return rep.stage("cudaGraphAddKernelNode(gridDimX=0)",
                                 cudaGraphAddKernelNode(&node, graph, nullptr, 0u, &np));
            });
        RefusalResult row;
        row.kind     = RefusalKind::ZeroSizedDispatch;
        row.accepted = probe.accepted;
        row.note =
            "CUDA rejects a zero-sized grid; the stage that reported it is quoted "
            "verbatim below (dispatch() never issues one: threads == 0 is a no-op "
            "there). LC refuses zero-sized dispatch at capture. | "
            + probe.note;
        rows.push_back(std::move(row));
    }

    // ---- 5. empty command list --------------------------------------------
    {
        // No nodes at all. Whether CUDA instantiates it is reported as measured; an
        // accepted value that does not reflect the binary would defeat H0.6.
        detail::ProbeReport probe =
            detail::probe_graph(d.m_stream,
                                [](cudaGraph_t /*graph*/, detail::ProbeReport& /*rep*/)
                                { return true; });
        RefusalResult row;
        row.kind     = RefusalKind::EmptyCommandList;
        row.accepted = probe.accepted;
        row.note =
            "CUDA's answer for a graph with no nodes is quoted verbatim below. This "
            "is the row that matters most for § 2.4: LC returns invalid_handle for an "
            "empty list *silently*, so an LC case can pass at 0 ms without anything "
            "having run -- hence the node-count verification in "
            "graph_capture_fill(). | "
            + probe.note;
        rows.push_back(std::move(row));
    }

    // ---- 6. callback in the list ------------------------------------------
    {
        // cudaGraphAddHostNode is a first-class node type. The header goes further:
        // cudaLaunchHostFunc (unlike the deprecated cudaStreamAddCallback) IS supported
        // with cudaStreamBeginCapture (cuda_runtime_api.h:2862-2864), so a callback is
        // legal in CUDA either way.
        int                 callback_ran = 0;
        detail::ProbeReport probe        = detail::probe_graph(
            d.m_stream,
            [&](cudaGraph_t graph, detail::ProbeReport& rep)
            {
                cudaHostNodeParams hp{};
                hp.fn                = &detail::host_probe_callback;
                hp.userData          = &callback_ran;
                cudaGraphNode_t node = nullptr;
                return rep.stage("cudaGraphAddHostNode",
                                 cudaGraphAddHostNode(&node, graph, nullptr, 0u, &hp));
            });
        RefusalResult row;
        row.kind     = RefusalKind::CallbackInList;
        row.accepted = probe.accepted;
        char buf[128];
        std::snprintf(buf,
                      sizeof(buf),
                      "callback executed during the replay: %s",
                      callback_ran != 0 ? "yes" : "no");
        row.note =
            "CUDA accepts it: cudaGraphAddHostNode is a legal node type "
            "(cudaHostNodeParams, driver_types.h:1415), and cudaLaunchHostFunc is even "
            "supported during stream capture (cuda_runtime_api.h:2862-2864), unlike the "
            "deprecated cudaStreamAddCallback. It has no plain <<<>>> equivalent -- the "
            "closest stream API is cudaLaunchHostFunc -- so a callback always needs an "
            "explicitly built node. LC refuses callbacks in a captured list. | ";
        row.note += buf;
        row.note += " | ";
        row.note += probe.note;
        rows.push_back(std::move(row));
    }

    release(arena);

    // One counted sync at the end: probe rows used the raw stream sync above so that
    // the §6.5 sync counters keep reporting only the measured windows, but the device
    // must be idle when this returns.
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
    // Graph handles survive a reset: this clears the *measurement*, not the state.
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
}  // namespace uipc::benchmark

#undef UIPC_BENCH_CUDA_CHECK
