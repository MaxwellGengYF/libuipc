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
 *********************************************************************/
#include "luisa_test_runtime.h"

TEST_CASE("zzz_runtime_shutdown", "[luisa][device]")
{
    auto& rt = lud::Runtime::instance();
    if(rt.initialized())
    {
        rt.shutdown();
    }
    CHECK(!rt.initialized());
}
