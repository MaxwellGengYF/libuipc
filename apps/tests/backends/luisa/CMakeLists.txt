# ---------------------------------------------------------------------------
# LuisaCompute backend tests (uipc_test_backend_luisa)
#
# Plain C++20: the backend has no CUDA language, so this builds like any other
# host test. The LuisaCompute runtime DLLs / backend modules are deployed next
# to the module by src/backends/luisa/CMakeLists.txt.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE CONFIGURE_DEPENDS "*.cpp" "*.h")
# Exclude the throwaway standalone build tree (it contains CMake compiler probes).
list(FILTER SOURCE EXCLUDE REGEX "/standalone/")

uipc_add_test(backend_luisa ${SOURCE})
target_link_libraries(backend_luisa PRIVATE uipc::backend::luisa)
# The tests instantiate LC DSL templates directly, so they need LuisaCompute's
# import libraries on their own link line (not only via the module).
target_link_libraries(backend_luisa PRIVATE luisa::compute)

# LC headers need a conformant preprocessor on MSVC; /utf-8 keeps non-ASCII
# comment characters in the sources from being misread under a non-UTF-8 code page.
if(MSVC)
    target_compile_options(backend_luisa PRIVATE /utf-8)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(backend_luisa PRIVATE /Zc:preprocessor)
    endif()
endif()
