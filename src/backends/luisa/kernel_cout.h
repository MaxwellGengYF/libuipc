#pragma once
/********************************************************************
 * @file kernel_cout.h
 * @brief Backend-level device console (reference: cuda/kernel_cout.h).
 *
 * The reference facade is
 *     my_kernel<<<grid, block>>>(KernelCout::viewer(), ...);
 *     // in kernel: cout << "x = " << x << "\n";
 * i.e. a *streaming* device-side logger whose fragments go out through device
 * printf and are flushed to stdout on the next sync (cuda_tool::LoggerViewer,
 * cuda_tool/logger.h).
 *
 * LuisaCompute has no device-side stream object: the one device print is
 * `device_log(fmt, args...)` whose format string is parsed by the runtime
 * (luisa_device/log.h). The reconciliation is therefore:
 *
 *  1. `KernelCout::viewer()` stays and returns `luisa_device::LoggerViewer`, a
 *     no-op device-capturable handle, so the reference's plumbing (threading a
 *     `cout_view` into a launch) ports unchanged. Streaming into it is compiled
 *     out; a kernel prints with LC_LOG instead:
 *         LC_LOG("x = {}", x);          // one {} per argument, no other specifier
 *  2. LC_LOG is a no-op unless UIPC_LUISA_KERNEL_PRINT is defined to 1 *in the
 *     translation unit that defines the kernel* (device print is not free — the
 *     same "debug-only aid" contract the reference had).
 *  3. The print is delivered, in dispatch order, when the stream reaches the
 *     print buffer's read-back, i.e. inside the next synchronize() — the exact
 *     semantics of the reference's "flushed to stdout on the next sync". By
 *     default it goes to the process log as `[DEVICE] <text>` (LC's own path);
 *     `set_capture()`/`KernelCoutCapture` route it into a string list instead,
 *     which is what makes the device output *observable* from a test.
 *
 * Capture mechanics (LC's per-stream log callback). LC delivers the formatted
 * print text to `Stream::set_log_callback`'s callback, and replacing a callback
 * that is already set logs "Setting CUDAStream::log_callback more than once …
 * not thread-safe". This facade therefore installs its trampoline **once per
 * stream** (the first capture on that stream, after a synchronize) and later
 * captures only retarget it; `clear_capture()` stops routing, after which the
 * trampoline reproduces LC's own default (the process log, `[DEVICE] <text>`).
 * One capture per stream at a time; a second one on the same stream takes over.
 *
 * Formatting note (verified against LC 0.10.0): LC accepts only `{}` per
 * argument (a width/precision specifier asserts), and renders values with fmt's
 * default formatter — doubles with the shortest round-trip spelling, not the
 * reference's printf("%f"). The reference's *text* is therefore not reproducible
 * token-for-token; the port's text is exact for a given LC_LOG call and the
 * device test asserts it exactly.
 *********************************************************************/
#include <type_define.h>
#include <luisa_device/log.h>
#include <luisa_device/stream.h>

#include <frame_stream.h>

#include <cstddef>
#include <string>
#include <vector>

namespace uipc::backend::luisa
{
class KernelCout
{
  public:
    /// One formatted device-print item (the text of a single LC_LOG call, with a
    /// trailing '\n' when the format string has one).
    using Lines = std::vector<std::string>;

    /// Kept for call-site compatibility (reference: cuda_tool::KernelCout::viewer()).
    [[nodiscard]] static luisa_device::LoggerViewer viewer() noexcept
    {
        return luisa_device::KernelCout::viewer();
    }

    /**
     * @brief Route `stream`'s device prints into `lines` (cleared first) instead of
     *        the process log.
     *
     * The registered callback runs on the host when the stream executes the print
     * buffer's read-back — i.e. during the next synchronize() (or when the frame
     * that contains the printing dispatch is committed and completed) — with one
     * call per print item and in dispatch order. It must be installed *before* the
     * printing dispatch is encoded (LC bakes the callback into the recorded
     * command), and `lines` must outlive the capture.
     */
    static void set_capture(luisa_device::LStream& stream, Lines& lines);

    /// Stop routing into the captured list; the stream's prints go back to the
    /// process log (`[DEVICE] <text>`).
    static void clear_capture(luisa_device::LStream& stream);
};

/**
 * @brief RAII capture of one stream's device prints (`KernelCout::set_capture`).
 *
 *     KernelCoutCapture cap;            // the shared frame stream
 *     dispatch_the_printing_kernel();
 *     cap.stream().synchronize();       // the prints arrive here
 *     CHECK(cap.lines().size() == 2);
 *
 * One capture per stream at a time: a second guard on the same stream takes the
 * routing over and the first one then sees nothing new.
 */
class KernelCoutCapture
{
  public:
    explicit KernelCoutCapture(luisa_device::LStream& stream = frame_stream());
    ~KernelCoutCapture();

    KernelCoutCapture(const KernelCoutCapture&)            = delete;
    KernelCoutCapture& operator=(const KernelCoutCapture&) = delete;

    [[nodiscard]] luisa_device::LStream& stream() noexcept { return *m_stream; }
    [[nodiscard]] const KernelCout::Lines& lines() const noexcept { return m_lines; }
    [[nodiscard]] std::size_t line_count() const noexcept { return m_lines.size(); }

  private:
    luisa_device::LStream* m_stream = nullptr;
    KernelCout::Lines      m_lines;
    bool                   m_active = false;
};
}// namespace uipc::backend::luisa
