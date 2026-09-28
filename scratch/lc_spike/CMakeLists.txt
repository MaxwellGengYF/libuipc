# ---------------------------------------------------------------------------
# T0.1 — Standalone LuisaCompute device spike (no libuipc integration yet).
#
# Purpose: prove that a Context/Device can be created against LC's CUDA backend,
# that a Kernel1D DSL kernel can be compiled/dispatched, that a CommandList can
# be recorded + committed, that device_log() output is retrievable, and that a
# warp reduction produces a correct value.
#
# LC is consumed through find_package(CONFIG) against a *pinned install prefix*
# (see docs/luisa_porting_cookbook.md "LC dependency strategy"). Point
# LuisaCompute_DIR at <prefix>/lib/cmake/LuisaCompute.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.26)
project(lc_spike LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED LUISA_COMPUTE_ROOT)
    set(LUISA_COMPUTE_ROOT "C:/dev/LuisaCompute" CACHE PATH "LuisaCompute source root")
endif()

find_package(LuisaCompute REQUIRED CONFIG
    PATHS "${LUISA_COMPUTE_ROOT}/install/lib/cmake/LuisaCompute"
    NO_DEFAULT_PATH)

message(STATUS "LuisaCompute found: backends=${LuisaCompute_BACKENDS} version=${LuisaCompute_VERSION}")

add_executable(lc_spike main.cpp)
target_link_libraries(lc_spike PRIVATE luisa::compute)
set_target_properties(lc_spike PROPERTIES CXX_STANDARD 20)

# Copies the LC runtime DLLs / backend modules next to the executable.
luisa_compute_deploy_runtime(TARGET lc_spike)
