#pragma once
/********************************************************************
 * @file luisa_test_shim.h
 * @brief Lets the luisa backend tests build either in-tree (Catch2, via
 *        uipc_add_test) or standalone (against the LC install only).
 *
 * In-tree:  Catch2 provides TEST_CASE / CHECK / REQUIRE / main (catch_main).
 * Standalone (define UIPC_LUISA_TESTS_STANDALONE): this header provides minimal
 * equivalents plus a main(), so the same test source can be compiled without
 * vcpkg/libuipc present — which is how the ~luisa_device~ foundation layer is
 * verified during bring-up.
 *********************************************************************/
#if defined(UIPC_LUISA_TESTS_STANDALONE)

#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <string>
#include <vector>

namespace uipc_luisa_test
{
struct Case
{
    std::string name;
    void (*fn)();
};
inline std::vector<Case>& cases()
{
    static std::vector<Case> s;
    return s;
}
inline int& failures()
{
    static int s = 0;
    return s;
}
inline const char*& program_path()
{
    static const char* s = "luisa_backend_tests";
    return s;
}
struct Registrar
{
    Registrar(const char* name, void (*fn)()) { cases().push_back({name, fn}); }
};
}// namespace uipc_luisa_test

#define UIPC_TEST_CAT_(a, b) a##b
#define UIPC_TEST_CAT(a, b) UIPC_TEST_CAT_(a, b)

#define TEST_CASE(name, ...)                                              \
    static void UIPC_TEST_CAT(uipc_test_fn_, __LINE__)();                 \
    static ::uipc_luisa_test::Registrar UIPC_TEST_CAT(uipc_test_reg_,     \
                                                      __LINE__){          \
        name, &UIPC_TEST_CAT(uipc_test_fn_, __LINE__)};                   \
    static void UIPC_TEST_CAT(uipc_test_fn_, __LINE__)()

#define CHECK(cond)                                                          \
    do                                                                       \
    {                                                                        \
        if(!(cond))                                                          \
        {                                                                    \
            ++::uipc_luisa_test::failures();                                 \
            std::printf("    CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
        }                                                                    \
    } while(false)

#define REQUIRE(cond)                                                        \
    do                                                                       \
    {                                                                        \
        if(!(cond))                                                          \
        {                                                                    \
            ++::uipc_luisa_test::failures();                                 \
            std::printf("    REQUIRE failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            return;                                                          \
        }                                                                    \
    } while(false)

#define CHECK_THROWS(expr)                                                    \
    do                                                                       \
    {                                                                        \
        bool threw = false;                                                  \
        try                                                                  \
        {                                                                    \
            (void)(expr);                                                    \
        }                                                                    \
        catch(...)                                                           \
        {                                                                    \
            threw = true;                                                    \
        }                                                                    \
        if(!threw)                                                           \
        {                                                                    \
            ++::uipc_luisa_test::failures();                                 \
            std::printf("    CHECK_THROWS failed: %s\n", #expr);             \
        }                                                                    \
    } while(false)

// The standalone `main` lives in standalone_main.cpp (compiled only when
// UIPC_LUISA_TESTS_STANDALONE is defined) so that any number of test sources can
// share this header. In-tree, Catch2's catch_main supplies `main`.

#else// in-tree: Catch2

#include <catch2/catch_test_macros.hpp>

#endif
