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
    # T6.3b3: WINDOWS_EXPORT_ALL_SYMBOLS used to export *every* symbol this module's objects
    # define.  Each kernel signature instantiates a large amount of LuisaCompute DSL
    # machinery (~345 symbols), so the auto-generated list grew to **69295** entries and the
    # link died with "fatal error LNK1189: cannot have more than 64K exports" (MSVC's export
    # table is capped at 65535; the T6.3b2 tree already sat at 65482, i.e. 53 below the
    # ceiling, so *any* new kernel tipped it over — this is a latent limit, not a T6.3b3
    # regression).  The module now exports an explicit, checked-in list instead: its C API
    # (uipc_create_engine / uipc_destroy_engine / uipc_init_module / uipc_query_module, which
    # are resolved with GetProcAddress and therefore imported by nobody) plus every symbol the
    # in-tree consumers import from it.  That file is
    # `src/backends/luisa/luisa_module.def`, maintained by `scripts/luisa_module_def.py`
    # (`--check` verifies it).  A test that uses a *new* backend symbol makes the link fail
    # with an unresolved external — the intended, loud signal: re-run that script and commit
    # the refreshed file.  Production consumers (apps/uipc core) only need the C API.
    set_target_properties(luisa PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    # NOTE: the four C entry points (`uipc_query_module` / `uipc_init_module` /
    # `uipc_create_engine` / `uipc_destroy_engine`) carry `UIPC_BACKEND_API`
    # (`__declspec(dllexport)`), so they are in the export table whatever this list says;
    # `scripts/luisa_check_module_abi.py` proves that dynamically (LoadLibrary +
    # GetProcAddress).  The list below is what the *C++* consumers need.
    target_link_options(luisa PRIVATE
        "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/luisa_module.def")
    # `/DEF:` alone is not a build dependency: without this the module is not relinked when
    # the export list changes, and a refreshed list would silently have no effect.
    set_property(TARGET luisa APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/luisa_module.def")
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
  #
  # This is an ALWAYS-RUN custom target rather than a POST_BUILD step of the module:
  # a POST_BUILD command only runs when the module is (re)linked, so rebuilding
  # LuisaCompute alone left a *stale* LC runtime next to the tests -- which silently
  # changes numerical results, because different LC revisions use different codegen
  # paths (measured in T5.2c: an older deployed luisa-backend-cuda.dll evaluated
  # double-precision log/exp/pow/trig through LC's AST path, i.e. in float, while the
  # current one lowers them to libdevice's double functions). See
  # docs/luisa_porting_cookbook.md section 1.6.
  if(EXISTS "${LUISA_COMPUTE_PREFIX}/bin")
      add_custom_target(luisa_runtime_deploy ALL
          COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different
                  "${LUISA_COMPUTE_PREFIX}/bin"
                  "$<TARGET_FILE_DIR:luisa>"
          COMMENT "Deploying LuisaCompute runtime next to uipc_backend_luisa")
      add_dependencies(luisa luisa_runtime_deploy)
  endif()


# No CUDA language, no cublas/cusparse/cusolver (unused — see plan §2.2).
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.h" "*.inl")
target_sources(luisa PRIVATE ${SOURCES})
uipc_target_set_relative_source_file(luisa)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCES})
