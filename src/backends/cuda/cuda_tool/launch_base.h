#pragma once
// CRTP launch base providing a fluent, chainable stream-bound launcher.
// Replacement for cuda_tool::LaunchBase (the algorithm backends subclass it).
#include <cuda_tool/stream.h>
#include <functional>
#include <string>
#include <string_view>

// NVTX ranges for a *host-side* launch label (BN1 I1/I6, bench_plan.md 13.3). Host-only
// header, but it must stay includable from `.cu`; nothing here is device code.
#include <backends/common/nvtx_range.h>

namespace uipc::backend::cuda_tool
{
// Non-template core holding the stream and stream-level operations.
class LaunchCore
{
  public:
    explicit LaunchCore(cudaStream_t stream = default_stream()) noexcept
        : m_stream(stream)
    {
    }
    virtual ~LaunchCore()
    {
        // Safety net: the NVTX range stack is per thread, so a label that was pushed and
        // never popped would stay open over everything the thread does afterwards. The
        // common call pattern is a temporary chain
        // (`stream.next<T>().kernel_name("PCG") << ...`), whose destruction is the end of
        // the launch anyway; `pop_kernel_label()` stays available for the explicit form.
        pop_kernel_label();
    }

    cudaStream_t stream() const { return m_stream; }

    void wait() { CUDA_TOOL_CHECK(cudaStreamSynchronize(m_stream)); }
    void callback(const std::function<void(cudaStream_t, cudaError_t)>& cb)
    {
        CUDA_TOOL_CHECK(cudaStreamAddCallback(
            m_stream,
            [](cudaStream_t s, cudaError_t e, void* user)
            {
                (*static_cast<std::function<void(cudaStream_t, cudaError_t)>*>(user))(s, e);
            },
            const_cast<std::function<void(cudaStream_t, cudaError_t)>*>(&cb),
            0));
    }

    static void wait_stream(cudaStream_t s)
    {
        CUDA_TOOL_CHECK(cudaStreamSynchronize(s));
    }
    static void wait_device() { CUDA_TOOL_CHECK(cudaDeviceSynchronize()); }
    static void wait_event(cudaEvent_t e)
    {
        CUDA_TOOL_CHECK(cudaEventSynchronize(e));
    }

  protected:
    void init_stream(cudaStream_t s) { m_stream = s; }

    // I1/I6 (bench_plan.md 13.3): one range *around the launch*, named after the label
    // the call site gave with `.kernel_name("PCG")`. This is what `nvtx_kern_sum` needs --
    // nsys attributes a launch to the NVTX range that was open on the submitting thread,
    // so a host-side push/pop is enough to get one report row per labelled kernel. It is
    // deliberately not a device-side label: nothing here reaches the kernel, and a
    // device label would have to be compiled in (the `muda` kernel-name metadata this
    // replaced has no counterpart in the raw `<<<>>>` launch path).
    void push_kernel_label(std::string_view name)
    {
        // One label per launch: a second name while one is open is ignored rather than
        // stacked, because the matching pop could not know how many times to pop.
        if(m_kernel_label_active)
        {
            return;
        }
        uipc::nvtx::push(name);
        m_kernel_label_active = true;
    }

    // Pops only what was pushed: the flag makes this idempotent, so a stray or doubled
    // pop can never unbalance the (per-thread) NVTX stack and swallow later ranges.
    void pop_kernel_label()
    {
        if(!m_kernel_label_active)
        {
            return;
        }
        uipc::nvtx::pop();
        m_kernel_label_active = false;
    }

    cudaStream_t m_stream;
    bool         m_kernel_label_active = false;
};

template <typename T>
class LaunchBase : public LaunchCore
{
  public:
    using derived_type = T;
    explicit LaunchBase(cudaStream_t stream = default_stream()) noexcept
        : LaunchCore(stream)
    {
    }

    T& kernel_name(std::string_view name)
    {
        // Opens the launch range (see push_kernel_label); `pop_kernel_label()` or the
        // destructor closes it. With `UIPC_NVTX=0` both calls are empty inline stubs, so
        // the no-label behaviour and the labelled one are identical when the switch is
        // off. `file_line` below stays a no-op: the profiler has no field for it.
        push_kernel_label(name);
        return derived();
    }
    T& file_line(std::string_view, int) { return derived(); }
    T& wait()
    {
        LaunchCore::wait();
        return derived();
    }
    T& callback(const std::function<void(cudaStream_t, cudaError_t)>& cb)
    {
        LaunchCore::callback(cb);
        return derived();
    }

    template <typename Next>
    Next next(Next n)
    {
        n.init_stream(this->stream());
        return n;
    }
    template <typename Next, typename... Args>
    Next next(Args&&... args)
    {
        Next n(std::forward<Args>(args)...);
        n.init_stream(this->stream());
        return n;
    }

  protected:
    T& derived() noexcept { return *static_cast<T*>(this); }
};

// free helpers matching muda's wait_device()/wait_stream()
inline void wait_device()
{
    LaunchCore::wait_device();
}
inline void wait_stream(cudaStream_t s)
{
    LaunchCore::wait_stream(s);
}
}  // namespace uipc::backend::cuda_tool
