-- LuisaCompute backend (uipc_backend_luisa)
--
-- Plain C++20: no CUDA language, no nvcc. LuisaCompute is consumed from a pinned
-- install prefix (same strategy as the CMake build, see
-- docs/luisa_porting_cookbook.md §1). Override the prefix root with
--   xmake f --luisa_root=<path-to-LuisaCompute-source>
-- whose install tree is expected at <root>/install.

option("luisa_root", {default = "C:/dev/LuisaCompute", description = "LuisaCompute source root (install at <root>/install)"})

target("luisa")
    add_rules("backend")
    add_files("**.cpp")
    add_headerfiles("**.h", "**.inl")
    add_includedirs(path.join(os.projectdir(), "src"))
    add_includedirs(os.scriptdir(), {public = true})
    -- luisa_device (device utilities) on the include path for <luisa_device/...>
    add_includedirs(path.join(os.scriptdir(), "luisa_device"), {public = true})
    add_deps("uipc_geometry", "uipc_io")

    if has_config("dev") then
        add_rules("clangd")
    end

    on_load(function (target)
        local root = target:extraconf("rules", "@luisa-root") or get_config("luisa_root") or "C:/dev/LuisaCompute"
        local prefix = path.join(root, "install")
        local inc = path.join(prefix, "include")
        local lib = path.join(prefix, "lib")
        local bin = path.join(prefix, "bin")

        if not os.isdir(prefix) then
            raise("LuisaCompute install prefix not found: " .. prefix ..
                  " (build+install LC first; see docs/luisa_porting_cookbook.md §1)")
        end

        -- Public include surface (matches the imported CMake target).
        target:add("includedirs", inc, {public = true})
        target:add("includedirs", path.join(inc, "luisa", "ext"), {public = true})
        target:add("includedirs", path.join(inc, "luisa", "ext", "xxhash"), {public = true})

        -- Compile definitions used when LuisaCompute itself is built.
        target:add("defines",
            "LUISA_ENABLE_DSL=1",
            "LUISA_ENABLE_XIR=1",
            "LUISA_USE_SYSTEM_STL=1",
            "LUISA_PLATFORM_WINDOWS=1",
            "XXH_INLINE_ALL",
            "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR",
            "NOMINMAX",
            "WIN32_LEAN_AND_MEAN")

        target:add("linkdirs", lib)
        target:add("links",
            "luisa-tile",
            "luisa-xir",
            "luisa-dsl",
            "luisa-runtime",
            "luisa-ast",
            "luisa-core")

        -- Deploy the LC runtime + backend modules beside the produced module so
        -- it can be dlopen'd and can find luisa-backend-cuda.dll at runtime.
        if os.isdir(bin) then
            target:add("installfiles", path.join(bin, "*.dll"))
        end
    end)
