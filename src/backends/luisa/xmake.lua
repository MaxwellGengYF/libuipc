-- LuisaCompute backend (uipc_backend_luisa)
--
-- Plain C++20: no CUDA language, no nvcc. LuisaCompute is consumed from a pinned
-- install prefix (same strategy as the CMake build, see
-- docs/luisa_porting_cookbook.md section 1):
--
--   xmake f --backend_luisa=y --backend_cuda=n \
--           --luisa_root=C:/dev/LuisaCompute \
--           --luisa_prefix=C:/dev/LuisaCompute/install-msvc
--
-- The --luisa_root / --luisa_prefix options are declared in the ROOT xmake.lua (see the
-- comment there: an option declared in this file would never be registered, because this
-- file is only included once backend_luisa is already enabled).
--
-- The prefix must have been built with the same toolchain/ABI as this build
-- (cookbook section 1.1), hence the <root>/install-msvc default on Windows -- the
-- same prefix the CMake build pins (build/CMakeCache.txt: LUISA_COMPUTE_PREFIX).

-- Resolve the LuisaCompute install prefix, preferring the -msvc install (the one the
-- CMake build uses) and falling back to a plain <root>/install. `local` on purpose:
-- xmake runs on_load/after_build closures in a sandbox where this file's *globals* are
-- gone, while upvalues captured by the closure keep working.
local function uipc_luisa_install_prefix()
    local explicit = get_config("luisa_prefix")
    if explicit and explicit ~= "" then
        return explicit
    end
    local root = get_config("luisa_root") or "C:/dev/LuisaCompute"
    for _, candidate in ipairs({path.join(root, "install-msvc"), path.join(root, "install")}) do
        if os.isfile(path.join(candidate, "lib", "cmake", "LuisaCompute",
                              "LuisaComputeTargets.cmake")) then
            return candidate
        end
    end
    return path.join(root, "install")
end

-- Resolve the directory holding LuisaCompute's *external* runtime DLLs. LC is built with
-- LUISA_USE_SYSTEM_SPDLOG=ON, so its modules import the system runtime: luisa-core.dll
-- imports fmt.dll (spdlog itself is header-only inside LC). The pinned install prefix
-- does not ship that DLL, hence the search order: an explicit --luisa_deps_bin, else LC's
-- own build tree, else the vcpkg tree the CMake build uses (the standalone harness
-- resolves it the same way, through UIPC_VCPKG_INSTALLED_DIR).
local function uipc_luisa_deps_bin()
    local explicit = get_config("luisa_deps_bin")
    if explicit and explicit ~= "" then
        return explicit
    end
    local root = get_config("luisa_root") or "C:/dev/LuisaCompute"
    for _, candidate in ipairs({path.join(root, "build-msvc", "bin"),
                                path.join(os.projectdir(), "build", "vcpkg_installed",
                                          "x64-windows", "bin")}) do
        if os.isfile(path.join(candidate, "fmt.dll")) then
            return candidate
        end
    end
    return nil
end

-- The bin directories of the packages a target links against. target:pkgs() only returns
-- a target's OWN packages, while the shared spdlog arrives as a *public* package of
-- uipc_core, so the deps are traversed exactly like xmake's own target:pkgenvs()
-- (core/project/target.lua: "we can also get package envs from deps (public package)").
local function uipc_luisa_package_bindirs(target)
    local seen = {}
    local dirs = {}
    local function collect(scope, opt)
        for _, pkgname in ipairs(table.wrap(scope:get("packages", opt))) do
            local pkg = scope:pkg(pkgname)
            if pkg and not seen[pkg:name()] then
                seen[pkg:name()] = true
                table.insert(dirs, path.join(pkg:installdir(), "bin"))
            end
        end
    end
    collect(target, nil)
    for _, dep in ipairs(target:orderdeps()) do
        collect(dep, nil)
        collect(dep, {interface = true})
    end
    return dirs
end

-- Deploy the LuisaCompute runtime + backend modules next to a target, so that
-- uipc_backend_luisa.dll can be dlopen'd and can resolve luisa-backend-cuda.dll at run
-- time (the tests link the LC import libraries directly, so they need the DLLs beside
-- the executable too). A *stale* LC runtime silently changes numerical results, because
-- different LC revisions lower double transcendental functions differently (cookbook
-- section 1.6), so every consumer target deploys its own copy.
--
-- This has to be a RULE, not a target-level after_build: target/option/package scripts
-- run in xmake's read-only "interpreter" sandbox, whose os module exports only read
-- interfaces (core/sandbox/modules/interpreter/os.lua: no cp/mkdir/exec). Rule scripts
-- get the full os module -- xmake's own rules/wdk/xmake.lua copies *.dll from an
-- after_build the same way. No os.mkdir: target:targetdir() exists by then. xmake does
-- not deploy the DLLs of dependent packages either (it aborts at run time with
-- STATUS_DLL_NOT_FOUND), so they are deployed here too.
rule("luisa.runtime")
    after_build(function (target)
        if not is_plat("windows") then
            return
        end
        local targetdir = target:targetdir()
        -- The LC runtime is not just DLLs: the CUDA backend JITs through the standalone
        -- NVRTC compiler `luisa_nvrtc.exe`, which lives in the same directory. The CMake
        -- build copies that whole directory (copy_directory_if_different), so match it.
        local function copy_runtimes(dir)
            if not dir or not os.isdir(dir) then
                return
            end
            for _, pattern in ipairs({"*.dll", "*.exe"}) do
                for _, file in ipairs(os.files(path.join(dir, pattern))) do
                    os.cp(file, targetdir)
                end
            end
        end

        -- 1. the LC runtime, its backend modules (luisa-backend-cuda.dll), glslang and
        --    the standalone NVRTC compiler
        copy_runtimes(path.join(uipc_luisa_install_prefix(), "bin"))

        -- 2. LuisaCompute's external runtime dependency (fmt.dll, see above)
        local depsbin = uipc_luisa_deps_bin()
        if depsbin and os.isfile(path.join(depsbin, "fmt.dll")) then
            os.cp(path.join(depsbin, "fmt.dll"), targetdir)
        end

        -- 3. last, so nothing overwrites them: the shared libraries of the packages this
        --    target links against (xmake's spdlog.dll). xmake's spdlog embeds fmt
        --    statically, so the vcpkg spdlog must NOT be deployed -- the module was linked
        --    against xmake's spdlog and its mangled exports differ between versions.
        for _, bindir in ipairs(uipc_luisa_package_bindirs(target)) do
            copy_runtimes(bindir)
        end
    end)

target("luisa")
    add_rules("backend", "luisa.runtime")
    add_files("**.cpp")
    add_headerfiles("**.h", "**.inl")
    add_includedirs(path.join(os.projectdir(), "src"))
    add_includedirs(os.scriptdir(), {public = true})
    -- luisa_device (device utilities) on the include path for <luisa_device/...>
    add_includedirs(path.join(os.scriptdir(), "luisa_device"), {public = true})
    -- luisa_device has out-of-line definitions -> export them from the module DLL
    -- (parity with target_compile_definitions(luisa PRIVATE UIPC_LUISA_DEVICE_EXPORT=1)).
    add_defines("UIPC_LUISA_DEVICE_EXPORT=1")
    add_deps("uipc_geometry", "uipc_io")
    if is_plat("windows") then
        -- Explicit export list instead of WINDOWS_EXPORT_ALL_SYMBOLS: every kernel
        -- signature instantiates a large amount of LC DSL machinery (~345 symbols),
        -- which pushed the auto-generated export table past MSVC's 65535-entry ceiling
        -- (LNK1189, T6.3b3). The list is maintained by scripts/luisa_module_def.py
        -- (--check verifies it); a test that uses a *new* backend symbol fails the link
        -- with an unresolved external, which is the intended loud signal.
        -- Fed through xmake's builtin "platform.windows.def" rule, which turns the file
        -- into /def:<path> for the shared link AND records it as a link dependency, so a
        -- refreshed list relinks the module (the CMake build needs an explicit
        -- LINK_DEPENDS for the same reason). add_ldflags()/target:add("ldflags", ...) do
        -- NOT work here: the flag is listed by `xmake show -t luisa` but never reaches
        -- the link command line (measured with -v).
        add_files("luisa_module.def")
        add_rules("platform.windows.def")
    end
    if has_config("dev") then
        add_rules("clangd")
    end
    on_load(function (target)
        local prefix = uipc_luisa_install_prefix()
        local inc = path.join(prefix, "include")
        local lib = path.join(prefix, "lib")
        if not os.isdir(inc) or not os.isdir(lib) then
            raise("LuisaCompute install prefix looks incomplete: " .. prefix ..
                  " (expected include/ and lib/; build+install LC first, then configure " ..
                  "with --luisa_prefix=<prefix>, see docs/luisa_porting_cookbook.md section 1)")
        end
        -- Public include surface (matches the imported luisa::compute CMake target).
        target:add("includedirs", inc, {public = true})
        target:add("includedirs", path.join(inc, "luisa", "ext"), {public = true})
        target:add("includedirs", path.join(inc, "luisa", "ext", "xxhash"), {public = true})
        -- Compile definitions used when LuisaCompute itself is built. These are the
        -- union of the INTERFACE_COMPILE_DEFINITIONS of the imported `luisa::compute`
        -- target chain (install-msvc/lib/cmake/LuisaCompute/LuisaComputeTargets.cmake);
        -- MARL_USE_SYSTEM_STL=1 in particular is what keeps marl from including EASTL
        -- (it pulls <EASTL/unique_ptr.h>, which is not part of the install tree).
        -- PUBLIC, exactly like the imported target's INTERFACE properties: the tests
        -- include the same LC headers and need the same definitions.
        target:add("defines",
            "LUISA_PLATFORM_WINDOWS=1",
            "LUISA_ENABLE_DSL=1",
            "LUISA_ENABLE_XIR=1",
            "LUISA_USE_SYSTEM_STL=1",
            "LUISA_USE_SYSTEM_SPDLOG=1",
            "LUISA_VSTL",
            "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR",
            "XXH_INLINE_ALL",
            "MARL_DLL=1",
            "MARL_USE_SYSTEM_STL=1",
            "REPROC_SHARED",
            "REPROCXX_SHARED",
            "YYJSON_IMPORTS=1",
            "GLSLANG_IS_SHARED_LIBRARY=1",
            "VK_NO_PROTOTYPES",
            {public = true})
        -- Public, like the links below: the tests name the same import libraries and
        -- therefore need LuisaCompute's lib directory on their own link line.
        target:add("linkdirs", lib, {public = true})
        -- PUBLIC: this backend's headers (luisa_device/, type_define.h) expose LC types,
        -- and on Windows a module's consumers need the LC import libraries to resolve
        -- the inline/template DSL symbols they instantiate. Parity with
        -- `target_link_libraries(luisa PUBLIC uipc_geometry uipc_io luisa::compute)`.
        target:add("links",
            "luisa-tile",
            "luisa-xir",
            "luisa-dsl",
            "luisa-runtime",
            "luisa-ast",
            "luisa-core",
            {public = true})
        if is_plat("windows") then
            -- LuisaCompute headers require a conformant preprocessor (__VA_OPT__ in
            -- dsl/polymorphic.h) and report __cplusplus correctly; these plus the two
            -- warning suppressions are the LC config's INTERFACE_COMPILE_OPTIONS, hence
            -- public. /utf-8 keeps the sources' non-ASCII comment characters from being
            -- misread under a non-UTF-8 code page (e.g. 936) -- same as the CMake build.
            target:add("cxflags", "/utf-8", "/Zc:preprocessor", "/Zc:__cplusplus",
                                  "/wd4068", "/wd5105", {public = true})
            -- LC's luisa-compute-ext chain links dbghelp on Windows (cpptrace symbols).
            target:add("syslinks", "dbghelp", {public = true})
        end
    end)
