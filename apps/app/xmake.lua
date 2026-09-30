-- Pin Catch2 to the same version the CMake build resolves through vcpkg
-- (build/vcpkg_installed/vcpkg/status: catch2 3.8.1). Newer Catch2 releases randomize
-- the default *execution order* of test cases; this suite is order-sensitive (the
-- per-file "summary" cases assert on counters accumulated by their sibling cases), so an
-- unpinned resolve silently shuffles the suite and breaks it.
add_requires("catch2 3.8.1", {configs = {main = false}})
target("app")
    set_kind("static")
    set_group("apps")
    add_files("*.cpp")
    add_headerfiles("*.h")
    add_includedirs("../",{public = true})
    add_rules("uipc_deps")
    add_packages("catch2",{public = true})
    local format_string = [[%s=R"(%s)"]]
    add_defines(format(format_string, "UIPC_ASSET_PATH", path.unix("$(projectdir)/assets/")))
    add_defines(format(format_string, "UIPC_OUTPUT_PATH", path.unix("$(projectdir)/output/")))
