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

#include <luisa_device/context.h>

namespace uipc::backend::luisa::luisa_device
{
class LStream
{
  public:
    explicit LStream(lcrt::Stream& stream = main_stream()) noexcept
        : m_stream{&stream}
    {
    }

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
    lcrt::Stream*                     m_stream;
    std::optional<lcrt::CommandList>  m_list;
};
}// namespace uipc::backend::luisa::luisa_device
