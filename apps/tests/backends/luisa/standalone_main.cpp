/********************************************************************
 * @file standalone_main.cpp
 * @brief Standalone test runner `main` for the luisa backend tests.
 *
 * Compiled to an empty translation unit in-tree (Catch2's catch_main provides
 * `main` there); in a standalone build it walks the cases registered by
 * `luisa_test_shim.h`. Keeping it in its own TU lets several test sources share
 * the shim without colliding on `main`.
 *********************************************************************/
#if defined(UIPC_LUISA_TESTS_STANDALONE)

#include "luisa_test_shim.h"

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: keep output on crash
    if(argc > 0 && argv && argv[0])
    {
        ::uipc_luisa_test::program_path() = argv[0];
    }
    const char* backend = (argc > 1 && argv[1]) ? argv[1] : "cuda";
    std::printf("== luisa backend tests (standalone) backend=%s ==\n", backend);
    int total = 0;
    for(auto& c : ::uipc_luisa_test::cases())
    {
        ++total;
        const int before = ::uipc_luisa_test::failures();
        std::printf("-- %s\n", c.name.c_str());
        c.fn();
        if(::uipc_luisa_test::failures() == before)
        {
            std::printf("   [ok]\n");
        }
    }
    const int failed = ::uipc_luisa_test::failures();
    std::printf("== %d cases, %d checks failed ==\n", total, failed);
    return failed == 0 ? 0 : 1;
}

#endif// UIPC_LUISA_TESTS_STANDALONE
