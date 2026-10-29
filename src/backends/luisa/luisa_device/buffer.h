#pragma once
/********************************************************************
 * @file buffer.h
 * @brief Owning device containers (replacement for cuda_tool::DeviceVector /
 *        DeviceVar / DeviceBuffer).
 *
 * LC `Buffer<T>` has a *fixed* capacity: there is no resize. `LDeviceBuffer<T>`
 * restores the cuda_tool semantics on top of an LC buffer:
 *   - `resize(n)` grows the capacity (geometric over-allocation) and
 *     value-initialises the newly exposed tail, exactly like
 *     cuda_tool::DeviceVector::resize.
 *   - `clear()` keeps the allocation and sets size to 0.
 *   - `view()` / `cview()` hand out LBufferView<T> value types that share
 *     ownership of the underlying LC buffer.
 *
 * Rules (plan risk R5): a resize may reallocate, so views must be frame-scoped.
 * In debug builds `resize()` bumps a generation counter and views record the
 * generation they were created with; `LBufferView::use_check()` asserts.
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <luisa_device/context.h>
#include <luisa_device/stream.h>
#include <luisa_device/view.h>

namespace uipc::backend::luisa::luisa_device
{
template <typename T>
class LDeviceBuffer
{
  public:
    using value_type = T;

    LDeviceBuffer() = default;
    explicit LDeviceBuffer(size_t n) { resize(n); }
    explicit LDeviceBuffer(std::span<const T> host) { copy_from(host); }

    // ---- capacity / size ---------------------------------------------------
    [[nodiscard]] size_t size() const noexcept { return m_size; }
    [[nodiscard]] size_t capacity() const noexcept { return m_capacity; }
    [[nodiscard]] bool   empty() const noexcept { return m_size == 0u; }

    void reserve(size_t n)
    {
        if(n <= m_capacity)
        {
            return;
        }
        // Growth is a *batching barrier*, not an optimization opportunity: the open
        // frame CommandList holds raw handles into `m_buffer`, and LC does not
        // refcount them, so the old allocation must not be released while any
        // recorded command may still reference it (that is a use-after-free at
        // commit, i.e. a crash). flush() = commit + wait + re-open the frame list,
        // which makes both the preserving copy below and the release ordered.
        // Steady-state frames never get here: `reserve` grows geometrically and the
        // plan's rule is to pre-size the buffers an epoch touches (optimize_plan OP-G3).
        auto& stream = LStream::current();
        stream.flush();

        // geometric growth to keep steady-state frames allocation-free
        const size_t grown = std::max(n, m_capacity * 2u);
        auto         next  = std::make_shared<lcrt::Buffer<T>>(
            device().create_buffer<T>(grown));
        if(m_buffer && m_size != 0u)
        {
            // Raw on purpose: this copy reads the allocation that `m_buffer` is
            // about to release, so it must be *submitted* (encoded) here, not left
            // pending in a list. The stream is idle after the flush above, so it is
            // also ordered.
            stream.stream() << next->view(0u, m_size).copy_from(m_buffer->view(0u, m_size));
        }
        m_buffer   = std::move(next);
        m_capacity = grown;
        ++m_generation;
    }

    void resize(size_t n)
    {
        const size_t old_size = m_size;
        reserve(n);
        m_size = n;
        if(n > old_size && m_buffer)
        {
            // value-initialise the newly exposed tail (cuda_tool parity)
            if constexpr(std::is_trivially_copyable_v<T>)
            {
                // `zeros` is a local, so the upload must be consumed by this
                // statement: LBufferView::copy_from(span) routes to record_upload().
                // (Spell the view as LBufferView, not lcrt::BufferView: the LC
                // view's copy_from only *builds* a command, which would be discarded.)
                std::vector<T> zeros(n - old_size);
                LBufferView<T>{m_buffer, old_size, n - old_size}.copy_from(lcrt::span<const T>{zeros});
            }
        }
    }

    void clear() noexcept
    {
        m_size = 0u;
    }

    // Release the allocation entirely (rare; used by shutdown paths).
    void reset() noexcept
    {
        m_buffer.reset();
        m_size     = 0u;
        m_capacity = 0u;
        ++m_generation;
    }

    // ---- access ------------------------------------------------------------
    [[nodiscard]] lcrt::Buffer<T>& buffer() const
    {
        if(!m_buffer)
        {
            throw std::runtime_error{"LDeviceBuffer::buffer() on an unallocated buffer"};
        }
        return *m_buffer;
    }

    [[nodiscard]] LBufferView<T> view()
    {
        check_allocated();
        return LBufferView<T>{m_buffer, 0u, m_size};
    }
    [[nodiscard]] LBufferView<T> view(size_t offset, size_t count)
    {
        return view().subview(offset, count);
    }
    [[nodiscard]] LCBufferView<T> cview() const
    {
        check_allocated();
        return LBufferView<T>{m_buffer, 0u, m_size};
    }
    [[nodiscard]] LCBufferView<T> cview(size_t offset, size_t count) const
    {
        return cview().subview(offset, count);
    }

    // ---- transfers ---------------------------------------------------------
    void copy_from(std::span<const T> host)
    {
        resize(host.size());
        if(m_size == 0u)
        {
            // An empty copy must not dereference the (never allocated) handle:
            // resize(0) leaves m_buffer null.
            return;
        }
        // upload: submitted immediately through the frame stream (record_upload), so
        // `host` only has to outlive this statement.
        LBufferView<T>{m_buffer, 0u, m_size}.copy_from(host);
    }

    void copy_to(std::span<T> host) const
    {
        if(host.size() != m_size)
        {
            throw std::invalid_argument{"LDeviceBuffer::copy_to size mismatch"};
        }
        if(m_size != 0u)
        {
            // Recorded, i.e. ordered after the frame's pending kernels. The caller
            // must synchronize the frame stream before reading `host`.
            LBufferView<T>{m_buffer, 0u, m_size}.copy_to(host);
        }
    }

    void fill(const T& value)
    {
        if(m_size != 0u)
        {
            view().fill(value);
        }
    }

    // ---- debug lifetime check ---------------------------------------------
    [[nodiscard]] uint64_t generation() const noexcept { return m_generation; }

  private:
    void check_allocated() const
    {
        if(!m_buffer && m_size != 0u)
        {
            throw std::runtime_error{"LDeviceBuffer is unallocated"};
        }
    }

    std::shared_ptr<lcrt::Buffer<T>> m_buffer;
    size_t                                     m_size       = 0u;
    size_t                                     m_capacity   = 0u;
    uint64_t                                   m_generation = 0u;
};

// ---------------------------------------------------------------------------
//  The reference's implicitly-synchronous device scalar read, made explicit
// ---------------------------------------------------------------------------
/**
 * @brief Read one device element back to the host, in stream order.
 *
 * The reference backend reads device scalars *implicitly synchronously* --
 * `int h = device_var;`, `view.copy_to(&h);` -- which is ordered because
 * cuda_tool submits every launch on the CUDA default stream.  The port's LStream
 * batches its commands into the frame CommandList, so such a read has to commit the
 * list first (`LStream::flush`), otherwise it would observe the value from before the
 * recorded dispatches -- the ported BVHs' "read the reported pair count back and
 * resize if it did not fit" retry loop depends on this (T7.1a).
 *
 * @param stream the LStream the caller has been recording into
 * @param view   a one-element view (the reference's `Dense<T>`/`DeviceVar<T>`)
 */
template <typename T>
[[nodiscard]] inline T read_host_sync(LStream& stream, const LBufferView<T>& view)
{
    T v{};
    // Record the download (so it is ordered after the recorded dispatches) and then
    // commit + wait. `synchronize()` re-opens the frame list, so a read-back is a
    // barrier, not a switch back to per-statement submission (OP-A1/R-4).
    stream.record(view.native().copy_to(lcrt::span{&v, 1u}));
    stream.synchronize();
    return v;
}

// A single element in device memory (replacement for cuda_tool::DeviceVar).
template <typename T>
class LDeviceVar
{
  public:
    using value_type = T;

    LDeviceVar() { resize(1u); }
    explicit LDeviceVar(const T& value)
        : LDeviceVar()
    {
        copy_from(value);
    }

    [[nodiscard]] LBufferView<T>  view() { return m_buffer.view(); }
    [[nodiscard]] LCBufferView<T> cview() const { return m_buffer.cview(); }

    operator LBufferView<T>() { return view(); }
    operator LCBufferView<T>() const { return cview(); }

    void copy_from(const T& value, LStream* s = nullptr)
    {
        // `&value` is this function's parameter: an upload must be submitted now.
        auto& stream = s ? *s : LStream::current();
        stream.record_upload(m_buffer.buffer().view().copy_from(lcrt::span{&value, 1u}));
    }

    [[nodiscard]] T copy_to(LStream* s = nullptr) const
    {
        T     v{};
        auto& stream = s ? *s : LStream::current();
        stream.record(m_buffer.cview().native().copy_to(lcrt::span{&v, 1u}));
        stream.synchronize();
        return v;
    }

    // implicit host read-back (cuda_tool parity)
    operator T() const { return copy_to(); }

  private:
    void resize(size_t n) { m_buffer.resize(n); }
    LDeviceBuffer<T> m_buffer;
};
}// namespace uipc::backend::luisa::luisa_device
