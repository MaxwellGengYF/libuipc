-- Parity harness (S0 / T9.3a, .kimix_cache/plans/03-parity-harness.md, research
-- 03-parity-harness-research.md).
--
-- Two plain console executables, deliberately *not* Catch2 tests:
--
--   uipc_parity_runner     one (scene, backend) leg of the comparison. It loads the
--                          backend module by name through core::Engine, so it must
--                          NOT link the `luisa` module: one process holds exactly one
--                          backend, and both legs are separate processes driven by
--                          scripts/parity_run.py (a second Engine in one process breaks
--                          the LC stream -- plan §8 trap 1).
--   uipc_parity_reference  the Tier-1 host golden generator. It includes the
--                          reference headers under src/backends/cuda/... directly and
--                          never links or compiles CUDA, so it builds in a
--                          --backend_cuda=n configuration too.
--
-- Both are gated on "either backend is enabled", exactly like backends/
-- (apps/tests/xmake.lua), and mirror apps/tests/parity/CMakeLists.txt.
--
-- The `luisa.runtime` rule deploys the LC runtime DLLs (and luisa_nvrtc.exe) next to
-- the executable: the runner does not link LC, but the module it loads does, so a
-- runner without them next to it cannot construct an Engine (cookbook §1.5).

rule("uipc_parity")
    on_load(function (target)
        -- uipc_parity_runner / uipc_parity_reference: the two build systems must not
        -- drift on the .exe name, because scripts/parity_run.py looks for it.
        target:set("basename", "uipc_" .. target:name())
        target:set("kind", "binary")
        local root = os.projectdir()
        target:add("includedirs", path.join(root, "include"))
        target:add("includedirs", path.join(root, "apps", "tests", "parity"))
    end)

target("parity_runner")
    set_group("tests")
    add_rules("uipc_parity")
    -- reference/main.cpp has its own main() and belongs to the other target.
    add_files("main.cpp", "parity_*.cpp")
    add_deps("app")
    if has_config("backend_luisa") then
        add_rules("luisa.runtime")
    end
    if is_plat("windows") then
        add_cxflags("/utf-8", "/Zc:preprocessor")
    end
    if has_config("dev") then
        add_rules("clangd")
    end

target("parity_reference")
    set_group("tests")
    add_rules("uipc_parity")
    add_files("reference/main.cpp")
    -- `app` is what carries Eigen (the reference distance .inl files are pure Eigen)
    -- and the core libraries; nothing from src/backends/luisa is needed here.
    add_deps("app")
    if is_plat("windows") then
        add_cxflags("/utf-8", "/Zc:preprocessor")
    end
    if has_config("dev") then
        add_rules("clangd")
    end
