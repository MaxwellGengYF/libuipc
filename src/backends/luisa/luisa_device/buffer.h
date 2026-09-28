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
        // geometric growth to keep steady-state frames allocation-free
        const size_t grown = std::max(n, m_capacity * 2u);
        auto         next  = std::make_shared<lcrt::Buffer<T>>(
            device().create_buffer<T>(grown));
        if(m_buffer && m_size != 0u)
        {
            main_stream() << next->view(0u, m_size).copy_from(m_buffer->view(0u, m_size));
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
                std::vector<T> zeros(n - old_size);
                main_stream() << m_buffer->view(old_size, n - old_size)
                                     .copy_from(lcrt::span<const T>{zeros});
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
        main_stream() << m_buffer->view(0u, m_size).copy_from(host);
    }

    void copy_to(std::span<T> host) const
    {
        if(host.size() != m_size)
        {
            throw std::invalid_argument{"LDeviceBuffer::copy_to size mismatch"};
        }
        if(m_size != 0u)
        {
            main_stream() << m_buffer->view(0u, m_size).copy_to(host);
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
        auto& stream = s ? s->stream() : main_stream();
        stream << m_buffer.buffer().view().copy_from(lcrt::span{&value, 1u});
    }

    [[nodiscard]] T copy_to(LStream* s = nullptr) const
    {
        T     v{};
        auto& stream = s ? s->stream() : main_stream();
        stream << m_buffer.cview().native().copy_to(lcrt::span{&v, 1u});
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
