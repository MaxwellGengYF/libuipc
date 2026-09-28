# ---------------------------------------------------------------------------
# Standalone build of the luisa_device foundation tests.
#
# This lets the foundation layer (src/backends/luisa/luisa_device) be built and
# run against an installed LuisaCompute *without* the libuipc vcpkg toolchain —
# the bring-up path used before UIPC_WITH_LUISA_BACKEND is wired into the main
# build. The same test source is registered in-tree via uipc_add_test.
#
#   cmake -S apps/tests/backends/luisa/standalone -B <build> -G Ninja ^
#         -DCMAKE_BUILD_TYPE=Release ^
#         -DLUISA_COMPUTE_ROOT=C:/dev/LuisaCompute
#   cmake --build <build>
#   <build>/luisa_device_tests.exe cuda
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.26)
project(luisa_device_tests LANGUAGES CXX)

cmake_minimum_required(VERSION 3.26)
project(luisa_device_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED LUISA_COMPUTE_ROOT)
    set(LUISA_COMPUTE_ROOT "C:/dev/LuisaCompute" CACHE PATH "LuisaCompute source root")
endif()
if(NOT DEFINED LUISA_COMPUTE_PREFIX)
    # LuisaCompute must be built with the same toolchain/ABI as its consumer
    # (see docs/luisa_porting_cookbook.md section 1.1), hence the -msvc default.
    set(LUISA_COMPUTE_PREFIX "${LUISA_COMPUTE_ROOT}/install-msvc"
        CACHE PATH "LuisaCompute install prefix")
endif()
# LC was configured with LUISA_COMPUTE_USE_SYSTEM_SPDLOG=ON, so its package config
# re-finds spdlog/fmt. Point at a vcpkg installed tree when one is given; it is
# only needed as a *prefix* (the standalone harness links no vcpkg libraries).
if(DEFINED UIPC_VCPKG_INSTALLED_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${UIPC_VCPKG_INSTALLED_DIR}/x64-windows")
endif()
find_package(LuisaCompute REQUIRED CONFIG
    PATHS "${LUISA_COMPUTE_PREFIX}/lib/cmake/LuisaCompute"
    NO_DEFAULT_PATH)

set(UIPC_LUISA_BACKEND_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../src/backends/luisa")

file(GLOB LUISA_DEVICE_SOURCES CONFIGURE_DEPENDS
    "${UIPC_LUISA_BACKEND_DIR}/luisa_device/*.cpp"
    "${UIPC_LUISA_BACKEND_DIR}/luisa_device/linear/*.cpp")

add_executable(luisa_device_tests
    "../standalone_main.cpp"
    "../luisa_device_tests.cpp"
    "../luisa_device_math_tests.cpp"
    "../luisa_device_matrix_tests.cpp"
    "../luisa_device_distance_tests.cpp"
      "../luisa_device_linear_tests.cpp"
        "../luisa_device_converter_tests.cpp"
      "../luisa_device_algorithms_tests.cpp"
    "../zzz_runtime_shutdown.cpp"
    ${LUISA_DEVICE_SOURCES})
target_include_directories(luisa_device_tests PRIVATE
    "${UIPC_LUISA_BACKEND_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_definitions(luisa_device_tests PRIVATE UIPC_LUISA_TESTS_STANDALONE=1 UIPC_LUISA_DEVICE_EXPORT=1)
# LC headers need a conformant preprocessor on MSVC; /utf-8 keeps non-ASCII
# characters in the sources from being misread under a non-UTF-8 code page.
if(MSVC)
    target_compile_options(luisa_device_tests PRIVATE /utf-8)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(luisa_device_tests PRIVATE /Zc:preprocessor)
    endif()
endif()
# LC's default preprocessor requirements are inherited from the LC package.
target_link_libraries(luisa_device_tests PRIVATE luisa::compute)
luisa_compute_deploy_runtime(TARGET luisa_device_tests)

# LC was built against vcpkg's spdlog/fmt (LUISA_COMPUTE_USE_SYSTEM_SPDLOG=ON), so
# those two DLLs must sit next to the executable as well -- the harness does not
# otherwise link anything from vcpkg.
if(DEFINED UIPC_VCPKG_INSTALLED_DIR)
    foreach(_dep spdlog fmt)
        find_file(UIPC_DEP_${_dep}_DLL
            NAMES ${_dep}.dll
            PATHS "${UIPC_VCPKG_INSTALLED_DIR}/x64-windows/bin"
            NO_DEFAULT_PATH)
        if(UIPC_DEP_${_dep}_DLL)
            add_custom_command(TARGET luisa_device_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${UIPC_DEP_${_dep}_DLL}"
                        "$<TARGET_FILE_DIR:luisa_device_tests>")
        else()
            message(WARNING "standalone harness: ${_dep}.dll not found under "
                            "${UIPC_VCPKG_INSTALLED_DIR}/x64-windows/bin")
        endif()
    endforeach()
endif()
luisa_compute_deploy_runtime(TARGET luisa_device_tests)
