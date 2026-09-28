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
 *   - `commit()` submits the list to the main stream (stream << list.commit()).
 *   - `wait()` commits and then synchronizes the main stream.
 *
 * When no frame is open, `record()` submits immediately, so standalone code
 * (unit tests, setup/teardown) behaves exactly like `stream << cmd`.
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

#include <optional>
#include <vector>

#include <luisa_device/context.h>
#include <luisa_device/export.h>

namespace uipc::backend::luisa::luisa_device
{
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

    // Record a command (dispatch/copy/fill) into the open list, or submit it
    // directly when no frame is open.
    template <typename Cmd>
    void record(Cmd&& cmd)
    {
        if(m_list.has_value())
        {
            (*m_list) << std::forward<Cmd>(cmd);
        }
        else
        {
            (*m_stream) << std::forward<Cmd>(cmd);
        }
    }

    // Submit the open list (no synchronize).
    void commit();
    // Submit the open list and synchronize the main stream.
    void flush();

    // ---- stream operations -------------------------------------------------
    void synchronize();
    // cuda_tool::wait_device() / Debug::debug_sync_all() equivalent.
    void wait() { synchronize(); }

    [[nodiscard]] lcrt::Stream&       stream() noexcept { return *m_stream; }
    [[nodiscard]] lcrt::CommandList*  command_list() noexcept { return m_list ? &*m_list : nullptr; }

  private:
    struct CachedObject
    {
        const void* key;
        void*       ptr;
        void (*destroy)(void*);
    };

    lcrt::Stream*                     m_stream;
    std::optional<lcrt::CommandList>  m_list;
    std::vector<CachedObject>         m_cached;
};
}// namespace uipc::backend::luisa::luisa_device
