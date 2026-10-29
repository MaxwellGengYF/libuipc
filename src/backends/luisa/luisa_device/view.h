#pragma once
/********************************************************************
 * @file view.h
 * @brief Host-side device views (replacement for cuda_tool::CBufferView/BufferView).
 *
 * LC's `lcrt::BufferView<T>` is a lightweight, non-owning handle made
 * of (buffer handle, element stride, byte offset, size). It is the type a kernel
 * invocation accepts for a `Buffer<T>`/`BufferVar<T>` parameter, so it is the
 * natural device view. Its only drawback is that it does not keep the buffer
 * alive.
 *
 * `LBufferView<T>` adds that ownership (a shared_ptr to the LC Buffer) plus the
 * cuda_tool accessor surface (size/offset/subview/copy_from/copy_to/fill) and an
 * implicit conversion to the LC view, so ported host code reads almost
 * unchanged and kernel argument passing is transparent:
 *
 *   LBufferView<Vector3> pos = positions.view();
 *   stream.record(kernel(pos, n).dispatch(n));  // pos -> BufferView<Vector3>
 *
 * Lifetime rule (plan risk R5): views are frame-scoped. Growing the owning
 * LBuffer reallocates and invalidates outstanding views; do not keep one across
 * a resize().
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

#include <memory>
#include <span>
#include <stdexcept>

#include <luisa_device/context.h>
#include <luisa_device/stream.h>

namespace uipc::backend::luisa::luisa_device
{
template <typename T>
class LBufferView;

// Read-only flavour: same implementation, the const-ness is advisory (LC has no
// const buffer element views). Kept as a distinct name so ported signatures can
// mirror cuda_tool::CBufferView<T>.
template <typename T>
using LCBufferView = LBufferView<T>;

template <typename T>
class LBufferView
{
  public:
    using value_type = T;

    LBufferView() = default;
    LBufferView(std::shared_ptr<lcrt::Buffer<T>> base, size_t offset, size_t size) noexcept
        : m_base{std::move(base)}
        , m_offset{offset}
        , m_size{size}
    {
    }

    [[nodiscard]] bool   valid() const noexcept { return m_base != nullptr; }
    [[nodiscard]] size_t size() const noexcept { return m_size; }
    [[nodiscard]] size_t offset() const noexcept { return m_offset; }
    [[nodiscard]] size_t total_size() const noexcept { return m_base ? m_base->size() : 0u; }
    [[nodiscard]] bool   empty() const noexcept { return m_size == 0u; }

    [[nodiscard]] const std::shared_ptr<lcrt::Buffer<T>>& base() const noexcept
    {
        return m_base;
    }

    // The LC view used as a kernel argument (keeps the subview offset).
    [[nodiscard]] lcrt::BufferView<T> native() const noexcept
    {
        if(!m_base)
        {
            return lcrt::BufferView<T>{};
        }
        return m_base->view(m_offset, m_size);
    }

    // Implicit conversion so `kernel(view, n)` works without a .native() call.
    operator lcrt::BufferView<T>() const noexcept { return native(); }
    // cuda_tool parity aliases
    [[nodiscard]] lcrt::BufferView<T> cview() const noexcept { return native(); }
    [[nodiscard]] lcrt::BufferView<T> cviewer() const noexcept { return native(); }
    [[nodiscard]] lcrt::BufferView<T> viewer() const noexcept { return native(); }

    [[nodiscard]] LBufferView<T> subview(size_t offset, size_t count = ~0ull) const
    {
        if(count == ~0ull)
        {
            count = (offset <= m_size) ? (m_size - offset) : 0u;
        }
        if(offset + count > m_size)
        {
            throw std::out_of_range{"LBufferView::subview overflow"};
        }
        return LBufferView<T>{m_base, m_offset + offset, count};
    }

    // ---- transfers (async on the main stream unless `s` is given) ----------
    void copy_from(std::span<const T> src, LStream* s = nullptr) const
    {
        if(src.size() != m_size)
        {
            throw std::invalid_argument{"LBufferView::copy_from size mismatch"};
        }
        if(m_size == 0u || !m_base)
        {
            // An empty (never allocated) view has no native handle to copy
            // through: nothing to do.
            return;
        }
        // An upload captures the *host* pointer, and a recorded upload is a
        // deferred memcpy: record_upload() submits it now, ordered after whatever
        // the frame has recorded so far (OP-A2).
        auto& stream = s ? *s : LStream::current();
        stream.record_upload(native().copy_from(src));
    }

    void copy_to(std::span<T> dst, LStream* s = nullptr) const
    {
        if(dst.size() != m_size)
        {
            throw std::invalid_argument{"LBufferView::copy_to size mismatch"};
        }
        if(m_size == 0u || !m_base)
        {
            return;
        }
        // Recorded, not submitted: the copy is ordered after the frame's recorded
        // kernels, and the caller synchronizes before reading `dst` (so `dst` must
        // stay alive until that synchronize).
        auto& stream = s ? *s : LStream::current();
        stream.record(native().copy_to(dst));
    }

    // device -> device
    void copy_from(const LBufferView<T>& src, LStream* s = nullptr) const
    {
        if(src.size() != m_size)
        {
            throw std::invalid_argument{"LBufferView::copy_from(d2d) size mismatch"};
        }
        if(m_size == 0u || !m_base || !src.m_base)
        {
            return;
        }
        // Device -> device: recorded, so it runs after the kernels that produce
        // `src` instead of jumping ahead of them (the live `z = r` bug this fixes).
        auto& stream = s ? *s : LStream::current();
        stream.record(native().copy_from(src.native()));
    }

    // host read-back with an implicit synchronize (cuda_tool parity for
    // DeviceVar-style scalar reads).
    [[nodiscard]] T read_host() const
    {
        T v{};
        auto& stream = LStream::current();
        stream.record(native().copy_to(lcrt::span{&v, 1u}));
        // Commits the frame list (the copy included), waits, and re-opens it.
        stream.synchronize();
        return v;
    }

    // WARNING: only valid for size() == 1 (single scalar view).
    void write_host(const T& v) const
    {
        auto& stream = LStream::current();
        stream.record_upload(native().copy_from(lcrt::span{&v, 1u}));
    }

    // Fill every element with `value` (device-side fill kernel).
    void fill(const T& value, LStream* s = nullptr) const;

  private:
    std::shared_ptr<lcrt::Buffer<T>> m_base;
    size_t                                      m_offset = 0u;
    size_t                                      m_size   = 0u;
};

// Lazily compiled "fill" kernel, cached per element type.
namespace detail
{
template <typename T>
[[nodiscard]] inline auto& fill_kernel()
{
    // Deliberately leaked (cookbook §7.10 trap 1): a Shader held by a
    // function-local static is destroyed during static destruction, i.e. after the
    // test harness's explicit Runtime::shutdown(), and destroying an LC resource
    // whose device is gone aborts the process (0xC0000409). Compiled shaders have
    // no destructor work worth doing, so the cache is a heap object.
    static auto* s_kernel = new lcrt::Shader<1u, lcrt::Buffer<T>, T>{device().compile<1>(
        lcrt::Kernel1D<lcrt::Buffer<T>, T>{[](lcrt::BufferVar<T> v, lcrt::Var<T> value) noexcept {
            v->write(lcrt::dispatch_x(), value);
        }})};
    return *s_kernel;
}
}// namespace detail

template <typename T>
void LBufferView<T>::fill(const T& value, LStream* s) const
{
    if(m_size == 0u)
    {
        return;
    }
    // A fill is a dispatch with device-resident arguments only: recordable.
    auto& stream = s ? *s : LStream::current();
    stream.record(detail::fill_kernel<T>()(native(), value).dispatch(static_cast<uint32_t>(m_size)));
}
}// namespace uipc::backend::luisa::luisa_device
