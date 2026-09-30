-- LuisaCompute backend tests (uipc_test_backend_luisa)
--
-- Plain C++20: the backend has no CUDA language, so this builds like any other host
-- test. Parity with apps/tests/backends/luisa/CMakeLists.txt:
--   * the throwaway standalone project (standalone/) is excluded,
--   * the tests instantiate LC DSL templates directly, so they need LuisaCompute's
--     import libraries -- inherited from the public links of the `luisa` dependency,
--   * <src> must be on the include path for the backend's own headers
--     (<sim_system.h>, <global_geometry/...>) and their <backends/common/...> includes,
--   * zzz_runtime_shutdown.cpp must stay last: `**.cpp` expands in sorted order, which
--     the zzz_ prefix relies on (the in-tree `main` comes from catch_main.cpp).
--
-- The executable links the LuisaCompute import libraries, so the LC runtime DLLs must sit
-- beside it; the "luisa.runtime" rule (defined in src/backends/luisa/xmake.lua) deploys
-- them. The LC include dirs, definitions and compile options are all public on the
-- `luisa` target, mirroring the INTERFACE properties of the imported luisa::compute
-- CMake target, so nothing extra is needed here.
target("backend_luisa")
    set_group("tests")
    add_rules("uipc_test", "luisa.runtime")
    add_deps("luisa")
    add_files("**.cpp")
    -- standalone/ is a separate CMake project; its build dir contains compiler probes
    -- that must not be compiled here.
    remove_files("standalone/**")
    add_includedirs(path.join(os.projectdir(), "src"))
    if is_plat("windows") then
        add_cxflags("/utf-8", "/Zc:preprocessor")
    end
    if has_config("dev") then
        add_rules("clangd")
    end
