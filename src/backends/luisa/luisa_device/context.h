#pragma once
/********************************************************************
 * @file context.h
 * @brief Process-wide LuisaCompute runtime (Context + Device + main Stream).
 *
 * Replacement for the implicit CUDA runtime used by the reference backend.
 * The LC `Context` must outlive the `Device`, which must outlive every Buffer,
 * Shader and CommandList created from it. This singleton owns all three and is
 * therefore the single place that controls LC object lifetime:
 *
 *   Runtime::instance().init("cuda", argv[0]);
 *   ...
 *   Runtime::instance().shutdown();   // destroy device+stream before context
 *
 * Declaring the members in the order (context, device, stream) makes the
 * implicit destruction order exactly stream -> device -> context.
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>
#include <luisa_device/export.h>

#include <memory>
#include <string>
#include <string_view>

namespace uipc::backend::luisa::luisa_device
{
class UIPC_LUISA_DEVICE_API Runtime
{
  public:
    static Runtime& instance();

    // Create the LC context and device. `backend` is an LC backend name
    // ("cuda", "dx", "vk", ...). `program_path` must stay valid for the process
    // lifetime (pass argv[0]).
    void init(std::string_view backend, std::string_view program_path);

    // Tear everything down. Safe to call more than once, and safe to call when
    // never initialized.
    void shutdown();

    [[nodiscard]] bool initialized() const noexcept { return m_context != nullptr; }

    [[nodiscard]] lcrt::Device&       device();
    [[nodiscard]] lcrt::Stream&       main_stream();
    [[nodiscard]] lcrt::Context&      context();
    [[nodiscard]] std::string_view              backend_name() const noexcept { return m_backend; }

  private:
    Runtime()  = default;
    ~Runtime();
    Runtime(const Runtime&)            = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&)                 = delete;
    Runtime& operator=(Runtime&&)      = delete;

    std::unique_ptr<lcrt::Context> m_context;  // destroyed LAST
    lcrt::Device                   m_device;
    lcrt::Stream                   m_stream;   // destroyed FIRST
    std::string                              m_backend;
    std::string                              m_program_path;
};

// Convenience accessors (throw if the runtime was not initialized).
[[nodiscard]] UIPC_LUISA_DEVICE_API lcrt::Device& device();
[[nodiscard]] UIPC_LUISA_DEVICE_API lcrt::Stream& main_stream();

// Absolute path of the running executable. LC derives its runtime / builtin-data
// directories from the program path, so a backend loaded as a shared module
// (which never sees argv[0]) needs this to bring the Context up correctly.
[[nodiscard]] UIPC_LUISA_DEVICE_API std::string default_program_path();

}// namespace uipc::backend::luisa::luisa_device
