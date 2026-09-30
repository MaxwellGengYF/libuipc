#include <collision_detection/atomic_counting_lbvh.h>

#include <frame_stream.h>

/*************************************************************************************************
 * T7.1a — AtomicCountingLBVH (port of
 * src/backends/cuda/collision_detection/atomic_counting_lbvh.{h,cu}).  The two
 * ticket-counting kernels and the `detect`/`query` bodies are templated on the
 * predicate and live in details/atomic_counting_lbvh.inl; this file is the reference's
 * whole .cu (16 lines) plus the stream accessor.
 *************************************************************************************************/
namespace uipc::backend::luisa
{
AtomicCountingLBVH::AtomicCountingLBVH(lud::LStream* stream) noexcept
    : m_stream{stream}
{
}

lud::LStream& AtomicCountingLBVH::stream() noexcept
{
    return m_stream ? *m_stream : frame_stream();
}

void AtomicCountingLBVH::build(lud::LCBufferView<LinearBVHAABB> aabbs)
{
    m_aabbs = aabbs;
    m_lbvh.build(aabbs, &stream());
}
}// namespace uipc::backend::luisa
