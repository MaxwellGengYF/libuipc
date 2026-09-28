#include <luisa_device/stream.h>

namespace uipc::backend::luisa::luisa_device
{
void LStream::begin()
{
    if(!m_list.has_value())
    {
        m_list.emplace(lcrt::CommandList::create());
    }
}

void LStream::commit()
{
    if(m_list.has_value())
    {
        (*m_stream) << m_list->commit();
        m_list.reset();
    }
}

void LStream::flush()
{
    commit();
    synchronize();
}

void LStream::synchronize()
{
    // Anything recorded but not submitted must be submitted first, otherwise a
    // following synchronize() would not observe the recorded work.
    commit();
    m_stream->synchronize();
}

LStream::~LStream()
{
    // The cached objects own device buffers, and destroying a buffer after
    // Runtime::shutdown() is fatal in the LC CUDA backend (it fast-fails the
    // process). A long-lived stream (a static one, or one owned by a system that
    // outlives the runtime teardown) therefore releases them only while the runtime
    // is still alive; afterwards it leaks them, which is harmless at exit.
    if(!Runtime::instance().initialized())
        return;
    for(auto& o : m_cached)
    {
        if(o.ptr)
            o.destroy(o.ptr);
    }
}

void* LStream::cached_object(const void* key, void* (*create)(), void (*destroy)(void*))
{
    for(auto& o : m_cached)
    {
        if(o.key == key)
            return o.ptr;
    }
    auto* ptr = create();
    m_cached.push_back(CachedObject{key, ptr, destroy});
    return ptr;
}
}// namespace uipc::backend::luisa::luisa_device
