#pragma once
/********************************************************************
 * @file luisa_test_runtime.h
 * @brief Shared test fixture for the luisa backend tests.
 *
 * `test_runtime()` lazily initialises the process-wide LC runtime (the first test
 * to touch the GPU does it) and `UIPC_TEST_PROGRAM_PATH` resolves the real
 * executable path LC needs to find its runtime/builtin data.
 *
 * The runtime is torn down explicitly by the `zzz_runtime_shutdown` case in its
 * own translation unit, which must be *last* — see that file for why.
 *********************************************************************/
#include "luisa_test_shim.h"

#include <luisa_device/luisa_device.h>

namespace lud = uipc::backend::luisa::luisa_device;

#if defined(UIPC_LUISA_TESTS_STANDALONE)
#define UIPC_TEST_PROGRAM_PATH ::uipc_luisa_test::program_path()
#else
#define UIPC_TEST_PROGRAM_PATH lud::default_program_path()
#endif

namespace uipc_luisa_test
{
/// Process-wide runtime accessor; initialises on first use.
inline lud::Runtime& test_runtime(const char* backend = "cuda")
{
    auto& rt = lud::Runtime::instance();
    if(!rt.initialized())
    {
        rt.init(backend, UIPC_TEST_PROGRAM_PATH);
    }
    return rt;
}
}// namespace uipc_luisa_test

/// Short spelling for test bodies.
using uipc_luisa_test::test_runtime;
