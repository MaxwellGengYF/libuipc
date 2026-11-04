# ---------------------------------------------------------------------------
# Parity harness (S0 / T9.3a, .kimix_cache/plans/03-parity-harness.md)
#
# Two plain console executables, deliberately not Catch2 tests (uipc_add_test would
# inject catch_main.cpp):
#
#   uipc_parity_runner     one (scene, backend) leg of the comparison. It loads the
#                          backend module by name through core::Engine, so it must NOT
#                          link uipc::backend::luisa: one process holds exactly one
#                          backend, and both legs are separate processes driven by
#                          scripts/parity_run.py (plan §8 trap 1).
#   uipc_parity_reference  the Tier-1 host golden generator. It includes the reference
#                          headers under src/backends/cuda/... directly and never
#                          compiles CUDA, so it builds with UIPC_WITH_CUDA_BACKEND=OFF.
#
# Mirrors apps/tests/parity/xmake.lua. NOTE: this file is unverified by a build -- the
# machine has no vcpkg toolchain (AGENT_ENV_BRIEF.md:10), so it is written by mirroring
# apps/tests/backends/luisa/CMakeLists.txt literally.
# ---------------------------------------------------------------------------
function(uipc_parity_executable name)
    target_link_libraries(${name} PRIVATE
            app
            uipc::uipc)
    set_property(TARGET ${name} PROPERTY FOLDER "apps/tests")
    # uipc_parity_runner / uipc_parity_reference: no .exe-name drift between the two
    # build systems, because scripts/parity_run.py looks for the name.
    set_target_properties(${name} PROPERTIES OUTPUT_NAME "uipc_${name}")
    uipc_target_add_include_files(${name})
    # The harness headers are included with angle brackets (<parity_scene.h>), the same
    # way the backend's own headers are.
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    uipc_target_set_output_directory(${name})
    # Builds the backend modules first, and deploys them (with the LuisaCompute
    # runtime) next to the executable -- the runner loads uipc_backend_<name>.dll from
    # its own directory.
    uipc_target_add_backend_dependency(${name})
    if(MSVC)
        target_compile_options(${name} PRIVATE /utf-8)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${name} PRIVATE /Zc:preprocessor)
        endif()
    endif()
endfunction()

add_executable(parity_runner
        main.cpp
        parity_artifact.cpp
        parity_probe.cpp
        parity_scene.cpp)
uipc_target_set_relative_source_file(parity_runner)
uipc_parity_executable(parity_runner)

add_executable(parity_reference reference/main.cpp)
uipc_target_set_relative_source_file(parity_reference)
uipc_parity_executable(parity_reference)
