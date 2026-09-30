rule("uipc_test")
    on_load(function (target)
        target:set("basename", "uipc_test_" .. target:name())
        target:set("kind","binary")
        target:add("deps","app", "none", {public = true})
        target:add("files", path.join(os.projectdir(), "apps/tests/catch_main.cpp"))
        local include_dir = path.join(os.projectdir(), "include")
        target:add("includedirs", include_dir)
        target:add("headerfiles", path.join(include_dir, "**.h"), path.join(include_dir, "**.inl"))
    end)

includes("common")
includes("core")
includes("geometry")
includes("sanity_check")
-- The backend test directories are per-backend (backends/xmake.lua gates cuda/luisa
-- itself), so they are included whenever *either* backend is enabled -- a
-- luisa-only configuration used to skip them entirely.
if has_config("backend_cuda") or has_config("backend_luisa") then
    includes("backends")
end

if has_config("backend_cuda") then
    includes("sim_case")
    includes("regression")
end
