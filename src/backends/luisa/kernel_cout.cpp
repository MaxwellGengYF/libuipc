#include <kernel_cout.h>

#include <luisa_device/luisa_device.h>

#include <map>
#include <mutex>
#include <string_view>
#include <utility>

namespace uipc::backend::luisa
{
// The device print path itself is header-only: the kernel side is LC_LOG
// (luisa_device/log.h) and the delivery side is LC's per-stream log callback. What
// lives here is the *capture* policy — the reference had none (printf went
// straight to stdout), but the plan's T4.3 validation item is "kernel_cout prints
// captured on sync in a device test", and LC's per-stream callback is the only
// place where the formatted text is available before it reaches the log.
namespace
{
    /// The capture state of one LC stream. `installed` means the trampoline below
    /// is registered on it; `target` is where the prints currently go (null = the
    /// process log).
    struct CaptureSlot
    {
        KernelCout::Lines* target    = nullptr;
        bool               installed = false;
    };

    // Deliberately leaked: the trampoline is owned by the stream and outlives every
    // capture (and the LC runtime, which the process may bring up repeatedly).
    using Slots = std::map<std::uint64_t, CaptureSlot>;

    Slots& slots()
    {
        static auto* s = new Slots{};
        return *s;
    }

    std::mutex& slots_mutex()
    {
        static auto* m = new std::mutex{};
        return *m;
    }

    CaptureSlot& slot_for(std::uint64_t handle)
    {
        std::lock_guard lock{slots_mutex()};
        return slots()[handle];
    }

    /// Install the trampoline on `stream` the first time the stream is captured.
    void install_once(lcrt::Stream& stream)
    {
        auto& slot = slot_for(stream.handle());
        if(slot.installed)
        {
            return;
        }

        // LC warns when a stream's log callback is already set ("… more than once …
        // not thread-safe, you may want to synchronize the stream"). Installing it
        // exactly once per stream avoids both the warning and that race; the
        // synchronize() below covers the case where this stream is still busy.
        stream.synchronize();

        const auto handle = stream.handle();
        stream.set_log_callback(
            [handle](std::string_view line)
            {
                KernelCout::Lines* target = nullptr;
                {
                    std::lock_guard lock{slots_mutex()};
                    if(auto it = slots().find(handle); it != slots().end())
                    {
                        target = it->second.target;
                    }
                }
                if(target)
                {
                    target->emplace_back(std::string{line});
                }
                else
                {
                    // LC's own default path (shader_print_formatter.h:
                    // `LUISA_INFO("[DEVICE] {}", scratch)`), so clearing a capture
                    // restores the reference's "flushed to stdout" behaviour.
                    // NOTE: luisa::log_info takes an *already formatted* message
                    // (the LUISA_INFO macro is the spelling that formats), which is
                    // why this is the macro and not a literal format string.
                    LUISA_INFO("[DEVICE] {}", line);
                }
            });

        slot.installed = true;
    }
}// namespace

void KernelCout::set_capture(luisa_device::LStream& stream, Lines& lines)
{
    lines.clear();
    install_once(stream.stream());
    slot_for(stream.stream().handle()).target = &lines;
}

void KernelCout::clear_capture(luisa_device::LStream& stream)
{
    slot_for(stream.stream().handle()).target = nullptr;
}

KernelCoutCapture::KernelCoutCapture(luisa_device::LStream& stream)
    : m_stream{&stream}
{
    KernelCout::set_capture(*m_stream, m_lines);
    m_active = true;
}

KernelCoutCapture::~KernelCoutCapture()
{
    if(m_active && m_stream)
    {
        KernelCout::clear_capture(*m_stream);
    }
}
}// namespace uipc::backend::luisa
