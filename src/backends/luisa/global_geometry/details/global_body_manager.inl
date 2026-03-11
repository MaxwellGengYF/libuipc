#pragma once

namespace uipc::backend::luisa
{
template <typename T>
BufferView<T> GlobalBodyManager::Impl::subview(Buffer<T>& buffer,
                                               SizeT index) const noexcept
{
    span<const IndexT> reporter_body_offsets = reporter_body_offsets_counts.offsets();
    span<const IndexT> reporter_body_counts = reporter_body_offsets_counts.counts();
    return buffer.view(reporter_body_offsets[index], reporter_body_counts[index]);
}
}  // namespace uipc::backend::luisa
