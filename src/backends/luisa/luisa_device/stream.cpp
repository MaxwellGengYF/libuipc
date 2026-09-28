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
}// namespace uipc::backend::luisa::luisa_device
