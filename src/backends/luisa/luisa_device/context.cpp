#include <luisa_device/context.h>

#include <stdexcept>

namespace uipc::backend::luisa::luisa_device
{
Runtime& Runtime::instance()
{
    static Runtime s_instance;
    return s_instance;
}

Runtime::~Runtime()
{
    shutdown();
}

void Runtime::init(std::string_view backend, std::string_view program_path)
{
    if(initialized())
    {
        return;
    }
    m_backend      = std::string{backend};
    m_program_path = std::string{program_path};
    m_context      = std::make_unique<lcrt::Context>(m_program_path);
    m_device       = m_context->create_device(m_backend);
    m_stream       = m_device.create_stream();
}

void Runtime::shutdown()
{
    if(m_context == nullptr)
    {
        return;
    }
    // Explicit reset in destruction-safe order: stream, device, context.
    m_stream = lcrt::Stream{};
    m_device = lcrt::Device{};
    m_context.reset();
}

lcrt::Device& Runtime::device()
{
    if(!initialized())
    {
        throw std::runtime_error{"luisa_device::Runtime is not initialized"};
    }
    return m_device;
}

lcrt::Stream& Runtime::main_stream()
{
    if(!initialized())
    {
        throw std::runtime_error{"luisa_device::Runtime is not initialized"};
    }
    return m_stream;
}

lcrt::Context& Runtime::context()
{
    if(!initialized())
    {
        throw std::runtime_error{"luisa_device::Runtime is not initialized"};
    }
    return *m_context;
}

lcrt::Device& device()
{
    return Runtime::instance().device();
}

lcrt::Stream& main_stream()
{
    return Runtime::instance().main_stream();
}

}// namespace uipc::backend::luisa::luisa_device
