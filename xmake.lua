set_xmakever("3.0.5")

option("pybind", {default = false, description = "Build pyuipc"})
option("examples", {default = true})
option("tests", {default = true})
option("benchmarks", {default = false})
option("dev", {default = true, description = "Enable developer mode"})
option("github_actions", {default = false})

option("backend_cuda", {default = true, description = "Build with CUDA backend"})
option("cuda_legacy_collision", {default = true, description = "Build legacy CUDA broad-phase trajectory filters"})
option("backend_luisa", {default = false, description = "Build with the LuisaCompute backend (no nvcc)"})
-- LuisaCompute resolution lives here, NOT in src/backends/luisa/xmake.lua: that file is
-- only included when backend_luisa is already enabled, so an option declared there would
-- never be registered and the CLI would reject --luisa_prefix/--luisa_root as unknown.
option("luisa_root", {default = "C:/dev/LuisaCompute",
    description = "LuisaCompute source root (install prefix defaults to <root>/install-msvc, then <root>/install)"})
option("luisa_prefix", {default = "",
    description = "LuisaCompute install prefix (overrides the <luisa_root> defaults)"})
-- LuisaCompute is built with LUISA_USE_SYSTEM_SPDLOG=ON, so its modules import the system
-- fmt runtime; that DLL is not part of the install prefix. Defaults are resolved in
-- src/backends/luisa/xmake.lua (LC build tree, then the repo's vcpkg tree).
option("luisa_deps_bin", {default = "",
    description = "Directory holding LuisaCompute's external runtime DLLs (fmt.dll)"})
option("usd", {default = false, description = "Build with OpenUSD support"})
option("vdb", {default = false, description = "Build with OpenVDB support"})

option("python_version", {default = "3.11.x", description = "Specify python version"})
option("python_system", {default = false, description = "Use system python"})


includes("src", "apps", "xmake/*.lua")

add_rules("mode.release", "mode.debug", "mode.releasedbg", "uipc.basic")

set_languages("c++20")

if is_plat("linux") then
    add_rpathdirs("$ORIGIN")
end

set_version("0.9.0")

-- Repository policy forbids compiler-cache wrappers because they make build
-- provenance and CUDA diagnostics harder to reproduce.
set_policy("build.ccache", false)

if has_config("dev") then
    set_policy("compatibility.version", "3.0")

    if is_plat("windows") then
        set_runtimes("MD")
    end
end
