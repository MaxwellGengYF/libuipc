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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED LUISA_COMPUTE_ROOT)
    set(LUISA_COMPUTE_ROOT "C:/dev/LuisaCompute" CACHE PATH "LuisaCompute source root")
endif()

find_package(LuisaCompute REQUIRED CONFIG
    PATHS "${LUISA_COMPUTE_ROOT}/install/lib/cmake/LuisaCompute"
    NO_DEFAULT_PATH)

set(UIPC_LUISA_BACKEND_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../src/backends/luisa")

file(GLOB LUISA_DEVICE_SOURCES CONFIGURE_DEPENDS
    "${UIPC_LUISA_BACKEND_DIR}/luisa_device/*.cpp")

add_executable(luisa_device_tests
    "../luisa_device_tests.cpp"
    ${LUISA_DEVICE_SOURCES})
target_include_directories(luisa_device_tests PRIVATE
    "${UIPC_LUISA_BACKEND_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_definitions(luisa_device_tests PRIVATE UIPC_LUISA_TESTS_STANDALONE=1)
# LC's default preprocessor requirements are inherited from the LC package.
target_link_libraries(luisa_device_tests PRIVATE luisa::compute)

luisa_compute_deploy_runtime(TARGET luisa_device_tests)
