#pragma once
/********************************************************************
 * @file stream.h
 * @brief LStream — the frame command-list facade (replaces cuda_tool::Stream).
 *
 * LuisaCompute separates *recording* (CommandList) from *submission* (Stream).
 * The reference CUDA backend submits every launch/copy immediately on a
 * cudaStream_t. LStream keeps that mental model while routing the commands
 * through an LC CommandList:
 *
 *   - `record(cmd)` appends to the currently open frame CommandList.
 *   - `record_upload(cmd)` is for commands that capture a *host* pointer; it
 *     commits what is recorded and then submits the command immediately (the
 *     host buffer is consumed by the statement, exactly like cuda_tool does).
 *   - `commit()` submits the list to the main stream (stream << list.commit()).
 *   - `synchronize()` commits, waits, and re-opens the frame list if it was open.
 *
 * When no frame is open, `record()` submits immediately, so standalone code
 * (unit tests, setup/teardown) behaves exactly like `stream << cmd`.
 *
 * ## The two contracts that make batching safe (optimize_plan.md OP-A1/A2)
 *
 * 1. **A recorded command is not executed until commit.** An LC CommandList
 *    stores raw resource handles and raw host pointers; the backend encodes it
 *    when it is submitted. Therefore everything a recorded command captured -
 *    a device Buffer *and* the source pointer of an upload - must stay valid
 *    until `commit()` returns. Hence `record_upload()` never defers: a stack
 *    buffer handed to `copy_from(span)` would otherwise be read after the
 *    statement ended (garbage value, no error).
 *
 * 2. **Statement order stops implying execution order.** With a list open,
 *    `stream << cmd` on the *raw* stream is a command that jumps ahead of
 *    everything recorded so far. Code that needs ordering must go through
 *    `record()` / `record_upload()` / `synchronize()`; reaching through
 *    `stream()` is only legitimate where the lcrt::Stream *identity* is needed
 *    (the kernel_cout log callback, `handle()` map keys).
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

#include <luisa_device/context.h>
#include <luisa_device/export.h>

namespace uipc::backend::luisa::luisa_device
{
/**
 * @brief Per-stream command counters (bench_plan.md §6.5, optimize_plan.md OP-A6).
 *
 * Sampled as a delta around a frame by the engine, so the same numbers answer
 * "how many submissions did one frame cost" without a profiler. `commands` is
 * what a CommandList absorbed; `direct_submits` is what escaped it (a batching
 * miss, or an upload); `commits` is the number of actual submissions.
 */
struct LStreamCounters
{
    uint64_t commands       = 0;  // appended to an open frame list
    uint64_t direct_submits = 0;  // submitted outside a list (miss or upload)
    uint64_t commits        = 0;  // non-empty lists submitted (real submits)
    uint64_t frame_opens    = 0;  // begin() calls that actually opened a list
    uint64_t uploads        = 0;  // record_upload() calls (host-sourced commands)
    uint64_t syncs          = 0;  // synchronize() calls
    double   sync_ms        = 0.0;  // wall time spent inside synchronize()

    [[nodiscard]] LStreamCounters operator-(const LStreamCounters& o) const noexcept
    {
        LStreamCounters d;
        d.commands       = commands - o.commands;
        d.direct_submits = direct_submits - o.direct_submits;
        d.commits        = commits - o.commits;
        d.frame_opens    = frame_opens - o.frame_opens;
        d.uploads        = uploads - o.uploads;
        d.syncs          = syncs - o.syncs;
        d.sync_ms        = sync_ms - o.sync_ms;
        return d;
    }
};

class UIPC_LUISA_DEVICE_API LStream
{
  public:
    explicit LStream(lcrt::Stream& stream = main_stream()) noexcept
        : m_stream{&stream}
    {
    }
    LStream(const LStream&)            = delete;
    LStream& operator=(const LStream&) = delete;
    ~LStream();

    /**
     * @brief The process-wide frame stream (the backend's `frame_stream()`).
     *
     * Owned by the device layer so that the low-level containers (`LBufferView`,
     * `LDeviceBuffer`, `LDeviceVar`) can default to it without including the
     * backend's `frame_stream.h`. Deliberately leaked (cookbook §7.10 trap 1):
     * destroying a long-lived LStream after Runtime::shutdown() releases cached
     * device buffers whose device is already gone, which aborts the process.
     */
    [[nodiscard]] static LStream& current();

    // ---- per-stream cached objects ----------------------------------------
    /**
     * @brief Lazily created, type-erased object owned by this stream and kept alive
     *        until the stream is destroyed.
     *
     * Needed because LC's CommandList stores *raw* resource handles: a temporary
     * buffer that is destroyed while a recorded command is still pending in an open
     * frame CommandList leaves a dangling handle behind (the failure is a crash at
     * commit, not a wrong number). The linalg algorithm primitives therefore take
     * their temporaries from here when the caller does not supply scratch storage
     * (see luisa_device/linear/algorithms.h).
     *
     * @param key     identity of the object (one static address per type)
     * @param create  allocates the object on the first call
     * @param destroy releases it when the stream dies
     */
    [[nodiscard]] void* cached_object(const void* key, void* (*create)(), void (*destroy)(void*));

    // ---- frame batching ---------------------------------------------------
    // Open a frame command list (idempotent).
    void begin();
    [[nodiscard]] bool frame_open() const noexcept { return m_list.has_value(); }

    // Record a command whose arguments all live in device memory
    // (dispatch / fill / device->device copy / device->host copy).
    template <typename Cmd>
    void record(Cmd&& cmd)
    {
        if(m_list.has_value())
        {
            ++m_counters.commands;
            m_dirty = true;
            (*m_list) << std::forward<Cmd>(cmd);
        }
        else
        {
            ++m_counters.direct_submits;
            (*m_stream) << std::forward<Cmd>(cmd);
        }
    }

    /**
     * @brief Record a command that captures a *host* pointer (an upload).
     *
     * A recorded upload is a deferred `memcpy` executed at commit (LC's
     * `BufferUploadCommand` stores the source pointer). Recording one would
     * therefore read host memory after the caller's statement ended - a silent
     * wrong value, not a crash. So this commits whatever is recorded (preserving
     * order) and submits the upload right away, which consumes the host pointer at
     * the end of the statement, exactly like the reference backend's
     * `BufferLaunch(stream).copy(...)`. The frame list is re-opened afterwards so a
     * frame does not lose its batching because of one upload.
     */
    template <typename Cmd>
    void record_upload(Cmd&& cmd)
    {
        ++m_counters.uploads;
        const bool reopen = m_list.has_value();
        commit();
        ++m_counters.direct_submits;
        (*m_stream) << std::forward<Cmd>(cmd);
        if(reopen)
        {
            begin();
        }
    }

    // Submit the open list (no synchronize). A no-op when nothing is recorded.
    void commit();
    // Submit the open list and synchronize the main stream.
    void flush();

    // ---- stream operations -------------------------------------------------
    /**
     * @brief Commit, wait for the device, and re-open the frame list if it was open.
     *
     * The re-open matters for OP-A1: a read-back used to commit the list and leave
     * it closed, which silently degraded the rest of the frame to per-statement
     * submission (the BVH pair-count retry loop is the common case). Re-opening
     * keeps "a barrier, not a mode switch" true.
     */
    void synchronize();
    // cuda_tool::wait_device() / Debug::debug_sync_all() equivalent.
    void wait() { synchronize(); }

    [[nodiscard]] const LStreamCounters& counters() const noexcept { return m_counters; }
    void reset_counters() noexcept { m_counters = LStreamCounters{}; }

    // The raw stream. Legitimate only where the stream *identity* is needed
    // (log callback installation, `handle()` keys); a command or a wait must go
    // through record()/record_upload()/synchronize() above.
    [[nodiscard]] lcrt::Stream&      stream() noexcept { return *m_stream; }
    [[nodiscard]] lcrt::CommandList* command_list() noexcept { return m_list ? &*m_list : nullptr; }

  private:
    struct CachedObject
    {
        const void* key;
        void*       ptr;
        void (*destroy)(void*);
    };

    lcrt::Stream*                    m_stream;
    std::optional<lcrt::CommandList> m_list;
    // Whether the open list has absorbed at least one command. An open but empty
    // list is dropped instead of submitted, so `commits` counts real submissions
    // (a frame with no work must not look like it submitted once).
    bool m_dirty = false;
    std::vector<CachedObject>        m_cached;
    LStreamCounters                  m_counters;
};
}// namespace uipc::backend::luisa::luisa_device
