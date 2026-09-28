# ---------------------------------------------------------------------------
# LuisaCompute backend (uipc_backend_luisa)
#
# Plain C++20 — no CUDA language, no nvcc. LuisaCompute is consumed through
# find_package(LuisaCompute CONFIG) against a pinned install prefix; the device
# backend is selected at runtime (default "cuda"), see docs/luisa_porting_cookbook.md.
# ---------------------------------------------------------------------------

# LuisaCompute is resolved at the top level (cmake/uipc_luisa.cmake) so that
# `luisa::compute` is visible to apps/tests as well.
if(NOT TARGET luisa::compute)
    uipc_error("luisa backend: luisa::compute is missing; configure with -DUIPC_WITH_LUISA_BACKEND=ON")
endif()

uipc_add_backend(luisa)
# T4.1: the backend's own unit tests link this module and use the backend's system
# classes (the global geometry managers + their reporter base classes), which is
# what `uipc_add_backend`'s "allowing backend-internal unit tests to link against
# the same binary" promises. On MSVC a DLL exports nothing by default, so the
# backend-internal symbols must be exported explicitly; WINDOWS_EXPORT_ALL_SYMBOLS
# generates the .def from this target's objects (the foundation layer keeps its
# own UIPC_LUISA_DEVICE_API export macro; this covers the system classes).
if(WIN32)
    set_target_properties(luisa PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# LuisaCompute must be PUBLIC: this backend's headers (luisa_device, type_define.h)
# expose LC types, and on Windows a module's consumers need the LC import
# libraries to resolve the inline/template DSL symbols they instantiate.
target_link_libraries(luisa PUBLIC
    uipc_geometry
    uipc_io
    luisa::compute
)

target_compile_features(luisa PUBLIC cxx_std_20)
# luisa_device has out-of-line definitions -> export them from the module DLL.
target_compile_definitions(luisa PRIVATE UIPC_LUISA_DEVICE_EXPORT=1)
target_include_directories(luisa PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
# <luisa_device/...> resolves from here (the analogue of the CUDA backend's
# <cuda_tool/...> include root).
target_include_directories(luisa PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/luisa_device")

# LuisaCompute headers require a conformant preprocessor (__VA_OPT__ in
# dsl/polymorphic.h). clang-cl/clang++ are fine; MSVC needs /Zc:preprocessor.
# /utf-8 keeps the sources' non-ASCII comment characters from being misread under
# a non-UTF-8 system code page (e.g. 936), which corrupts tokenization.
if(MSVC)
    target_compile_options(luisa PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(luisa PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    endif()
endif()

# The LC backend modules and runtime DLLs are discovered next to the consuming
# executable; deploy them beside this module too (Windows keeps them in bin/).
if(EXISTS "${LUISA_COMPUTE_PREFIX}/bin")
    add_custom_command(TARGET luisa POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                "${LUISA_COMPUTE_PREFIX}/bin"
                "$<TARGET_FILE_DIR:luisa>"
        COMMENT "Deploying LuisaCompute runtime next to uipc_backend_luisa")
endif()

# No CUDA language, no cublas/cusparse/cusolver (unused — see plan §2.2).
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h" "*.inl")
target_sources(luisa PRIVATE ${SOURCES})
uipc_target_set_relative_source_file(luisa)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCES})
