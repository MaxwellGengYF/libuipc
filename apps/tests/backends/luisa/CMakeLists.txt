# ---------------------------------------------------------------------------
# LuisaCompute backend tests (uipc_test_backend_luisa)
#
# Plain C++20: the backend has no CUDA language, so this builds like any other
# host test. The LuisaCompute runtime DLLs / backend modules are deployed next
# to the module by src/backends/luisa/CMakeLists.txt.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE CONFIGURE_DEPENDS "*.cpp" "*.h")

uipc_add_test(backend_luisa ${SOURCE})
target_link_libraries(backend_luisa PRIVATE uipc::backend::luisa)

# LC headers need a conformant preprocessor on MSVC.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(backend_luisa PRIVATE /Zc:preprocessor)
endif()
