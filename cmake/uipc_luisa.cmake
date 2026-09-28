# ---------------------------------------------------------------------------
# LuisaCompute dependency resolution for the `luisa` backend.
#
# Called from the ROOT CMakeLists.txt (top-level scope) so the imported
# `luisa::compute` target is visible to every subdirectory that needs it
# (src/backends/luisa and apps/tests/backends/luisa).
#
# Strategy (decided in T0.1, see docs/luisa_porting_cookbook.md §1):
#   find_package(LuisaCompute CONFIG) against a pinned install prefix, with an
#   add_subdirectory fallback when the prefix is unavailable (plan risk R1).
# ---------------------------------------------------------------------------
function(uipc_luisa_resolve_dependency)
    if(NOT DEFINED LUISA_COMPUTE_ROOT OR LUISA_COMPUTE_ROOT STREQUAL "")
        set(LUISA_COMPUTE_ROOT "C:/dev/LuisaCompute" CACHE PATH
            "LuisaCompute source root (its install prefix is <root>/install)")
    endif()

    if(NOT DEFINED LUISA_COMPUTE_PREFIX OR LUISA_COMPUTE_PREFIX STREQUAL "")
        set(LUISA_COMPUTE_PREFIX "${LUISA_COMPUTE_ROOT}/install" CACHE PATH
            "Install prefix of LuisaCompute (must be built with the same ABI as libuipc)")
    endif()
    set(_uipc_luisa_prefix "${LUISA_COMPUTE_PREFIX}/lib/cmake/LuisaCompute")
    if(EXISTS "${_uipc_luisa_prefix}/LuisaComputeConfig.cmake")
        find_package(LuisaCompute REQUIRED CONFIG
                     PATHS "${_uipc_luisa_prefix}"
                     NO_DEFAULT_PATH)
        uipc_info("LuisaCompute ${LuisaCompute_VERSION} resolved from '${_uipc_luisa_prefix}' "
                  "(backends: ${LuisaCompute_BACKENDS}).")
    else()
        uipc_info("LuisaCompute install prefix not found at '${_uipc_luisa_prefix}'; "
                  "falling back to add_subdirectory(${LUISA_COMPUTE_ROOT}).")
        add_subdirectory("${LUISA_COMPUTE_ROOT}"
                         "${CMAKE_BINARY_DIR}/luisa_compute"
                         EXCLUDE_FROM_ALL)
    endif()

    if(NOT TARGET luisa::compute)
        uipc_error("LuisaCompute did not provide the 'luisa::compute' target.")
    endif()
endfunction()
