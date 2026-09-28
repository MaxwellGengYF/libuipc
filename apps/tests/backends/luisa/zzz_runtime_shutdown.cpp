/********************************************************************
 * @file zzz_runtime_shutdown.cpp
 * @brief Explicit LC runtime teardown — must run LAST.
 *
 * LuisaCompute's Context has to be destroyed before the LC runtime modules are
 * unloaded; relying on the process-wide static destructors alone crashes during
 * DLL detach on Windows (0xC0000409 STATUS_STACK_BUFFER_OVERRUN).
 *
 * Catch2 runs test cases in declaration order, which across translation units
 * follows the order the static registrars are linked — i.e. the (sorted) source
 * glob. The `zzz_` file-name prefix keeps this translation unit last so that no
 * later test can observe a torn-down runtime.
 *
 * The case below only runs when it is *selected*: a filtered run
 * (`uipc_test_backend_luisa.exe "[linear]"`) skips it and would then exit with the
 * same 0xC0000409.  The end-of-run listener below is the backstop for that case —
 * it performs the identical teardown at the end of `Catch::Session::run()`, i.e.
 * still before any static teardown.  `Runtime::shutdown()` is idempotent, so the
 * two paths cannot conflict.
 *
 * The listener is in-tree only: the standalone harness has its own test registry
 * (`luisa_test_shim.h`, `standalone_main.cpp`) rather than Catch2, and runs its
 * cases unfiltered so the case below always executes there.
 *********************************************************************/
#include "luisa_test_runtime.h"

#if !defined(UIPC_LUISA_TESTS_STANDALONE)
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace
{
/// Performs the runtime teardown for *filtered* runs as well.
class RuntimeShutdownListener final : public Catch::EventListenerBase
{
  public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunEnded(const Catch::TestRunStats&) override
    {
        auto& rt = lud::Runtime::instance();
        if(rt.initialized())
        {
            rt.shutdown();
        }
    }
};
}  // namespace

CATCH_REGISTER_LISTENER(RuntimeShutdownListener)
#endif

TEST_CASE("zzz_runtime_shutdown", "[luisa][device]")
{
    auto& rt = lud::Runtime::instance();
    if(rt.initialized())
    {
        rt.shutdown();
    }
    CHECK(!rt.initialized());
}
