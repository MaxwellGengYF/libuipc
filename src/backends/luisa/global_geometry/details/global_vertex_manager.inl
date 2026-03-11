namespace uipc::backend::luisa
{
template <typename T>
luisa::compute::BufferView<T> GlobalVertexManager::Impl::subview(luisa::compute::Buffer<T>& buffer,
                                                                  SizeT index) const noexcept
{
    span<const IndexT> reporter_vertex_offsets = reporter_vertex_offsets_counts.offsets();
    span<const IndexT> reporter_vertex_counts = reporter_vertex_offsets_counts.counts();
    // In LuisaCompute, Buffer::view() returns a BufferView with offset and size
    return buffer.view(reporter_vertex_offsets[index], reporter_vertex_counts[index]);
}
}  // namespace uipc::backend::luisa
