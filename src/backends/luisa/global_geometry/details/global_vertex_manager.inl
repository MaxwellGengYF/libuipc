namespace uipc::backend::luisa
{
template <typename T>
lud::LBufferView<T> GlobalVertexManager::Impl::subview(lud::LDeviceBuffer<T>& buffer,
                                                       SizeT index) const noexcept
{
    span<const IndexT> reporter_vertex_offsets = reporter_vertex_offsets_counts.offsets();
    span<const IndexT> reporter_vertex_counts = reporter_vertex_offsets_counts.counts();
    return buffer.view(reporter_vertex_offsets[index], reporter_vertex_counts[index]);
}
}  // namespace uipc::backend::luisa
