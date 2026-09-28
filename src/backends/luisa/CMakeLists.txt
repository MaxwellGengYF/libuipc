# ---------------------------------------------------------------------------
# LuisaCompute backend (uipc_backend_luisa)
#
# Plain C++20 — no CUDA language, no nvcc. LuisaCompute is consumed through
# find_package(LuisaCompute CONFIG) against a pinned install prefix; the device
# backend is selected at runtime (default "cuda"), see docs/luisa_porting_cookbook.md.
# ---------------------------------------------------------------------------

# Resolve the LuisaCompute dependency (strategy decided in T0.1).
function(uipc_luisa_resolve_dependency)
    if(NOT DEFINED LUISA_COMPUTE_ROOT OR LUISA_COMPUTE_ROOT STREQUAL "")
        set(LUISA_COMPUTE_ROOT "C:/dev/LuisaCompute" CACHE PATH
            "LuisaCompute source root (its install prefix is <root>/install)")
    endif()
    set(_uipc_luisa_prefix "${LUISA_COMPUTE_ROOT}/install/lib/cmake/LuisaCompute")
    if(EXISTS "${_uipc_luisa_prefix}/LuisaComputeConfig.cmake")
        find_package(LuisaCompute REQUIRED CONFIG PATHS "${_uipc_luisa_prefix}" NO_DEFAULT_PATH)
    else()
        # Contingency (plan risk R1): fall back to building LC as a subproject.
        uipc_info("LuisaCompute install prefix not found at '${_uipc_luisa_prefix}'; "
                  "falling back to add_subdirectory(${LUISA_COMPUTE_ROOT}).")
        add_subdirectory("${LUISA_COMPUTE_ROOT}" "${CMAKE_BINARY_DIR}/luisa_compute" EXCLUDE_FROM_ALL)
    endif()
endfunction()

uipc_luisa_resolve_dependency()

uipc_add_backend(luisa)

target_link_libraries(luisa PUBLIC
    uipc_geometry
    uipc_io
    luisa::compute
)

target_compile_features(luisa PUBLIC cxx_std_20)
target_include_directories(luisa PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
# <luisa_device/...> resolves from here (the analogue of the CUDA backend's
# <cuda_tool/...> include root).
target_include_directories(luisa PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/luisa_device")

# LuisaCompute headers require a conformant preprocessor (__VA_OPT__ in
# dsl/polymorphic.h). clang-cl/clang++ are fine; MSVC needs /Zc:preprocessor.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(luisa PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
endif()

# The LC backend modules and runtime DLLs are discovered next to the consuming
# executable; deploy them beside this module too (Windows keeps them in bin/).
if(EXISTS "${LUISA_COMPUTE_ROOT}/install/bin")
    add_custom_command(TARGET luisa POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${LUISA_COMPUTE_ROOT}/install/bin"
                "$<TARGET_FILE_DIR:luisa>"
        COMMENT "Deploying LuisaCompute runtime next to uipc_backend_luisa")
endif()

# No CUDA language, no cublas/cusparse/cusolver (unused — see plan §2.2).
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h" "*.inl")
target_sources(luisa PRIVATE ${SOURCES})
uipc_target_set_relative_source_file(luisa)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCES})
