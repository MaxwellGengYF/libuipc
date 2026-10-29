-- Benchmarks (bench_plan.md §6.1, task B1).
--
-- Layout contract:
--   * one binary per backend: uipc_bench_t0_luisa / uipc_bench_t0_cuda, built from
--     the SAME sources -- bench_main.cpp, common/*.cpp and tier0/*.cpp are shared
--     verbatim, only the BenchDevice implementation TU differs
--     (luisa/bench_device_luisa.cpp vs cuda/bench_device_cuda.cu). "Same scenario,
--     two backends" is therefore enforced by the linker, not by discipline;
--   * never one binary linking both backends: B3 (a second Engine in one process
--     breaks the LC stream) and the two module DLLs are already separate;
--   * the Tier-0 benchmark targets do not depend on `app`/Catch2: they are plain
--     console programs so that a timed window has no test-framework code in it.
--     (The engine A/B driver and the Tier-1 kernel bench DO need the core libraries:
--     they drive the `luisa` module, not raw LuisaCompute.)
--
-- Naming: uipc_bench_<tier>_<backend> (see rule("uipc_bench")).

-- Compiler contract for every benchmark binary: the *only* per-backend switch is
-- which device TU is linked in. Anything else (a define, an include dir) must be
-- identical for both targets, otherwise the comparison is invalid.
rule("uipc_bench")
    on_load(function (target)
        target:set("basename", "uipc_bench_" .. target:name())
        target:set("kind", "binary")
        local bench_root = path.join(os.projectdir(), "apps", "benchmarks")
        target:add("includedirs", bench_root)
        target:add("includedirs", path.join(os.projectdir(), "src"))
        target:add("includedirs", path.join(os.projectdir(), "include"))
        -- The measured window must not be perturbed by debug/assert code: the
        -- benchmark always builds in release mode with NDEBUG (xmake's release
        -- rule already sets it; this documents the requirement).
        target:add("defines", "UIPC_BENCHMARK=1")
    end)

-- Sources shared by every tier and every backend. Kept as one list so the targets
-- cannot drift apart. (Tier 0's helpers in common/bench_case.cpp are written against
-- `BenchDevice`, so they belong to the Tier-0 list, not to this one.)
local bench_support_files = {
    "common/bench_app.cpp",
    "common/bench_env.cpp",
    "common/bench_report.cpp",
}

local bench_tier0_files = {
    "bench_main.cpp",
    "common/bench_case.cpp",
    "tier0/t0_cases.cpp",
}

-- LuisaCompute's include surface and define set. Restated for the targets that link
-- the `luisa` module *and* another dependency: the module publishes these from its
-- `on_load`, and a target with more than one dependency does not inherit them
-- (measured: without them the TU cannot find <luisa/luisa-compute.h>). The values are
-- the ones src/backends/luisa/xmake.lua sets; the backend test executable gets them
-- through the single `luisa` dependency.
function luisa_compute_surface()
    local lc_root = get_config("luisa_root") or "C:/dev/LuisaCompute"
    local lc_prefix = path.join(lc_root, "install-msvc")
    if not os.isdir(path.join(lc_prefix, "include")) then
        lc_prefix = path.join(lc_root, "install")
    end
    add_includedirs(path.join(lc_prefix, "include"))
    add_includedirs(path.join(lc_prefix, "include", "luisa", "ext"))
    add_includedirs(path.join(lc_prefix, "include", "luisa", "ext", "xxhash"))
    add_linkdirs(path.join(lc_prefix, "lib"))
    add_defines("LUISA_PLATFORM_WINDOWS=1", "LUISA_ENABLE_DSL=1", "LUISA_ENABLE_XIR=1",
                "LUISA_USE_SYSTEM_STL=1", "LUISA_USE_SYSTEM_SPDLOG=1", "LUISA_VSTL",
                "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR", "XXH_INLINE_ALL", "MARL_DLL=1",
                "MARL_USE_SYSTEM_STL=1", "REPROC_SHARED", "REPROCXX_SHARED",
                "YYJSON_IMPORTS=1", "GLSLANG_IS_SHARED_LIBRARY=1", "VK_NO_PROTOTYPES")
end

-- ---------------------------------------------------------------------------
-- Tier 0 — host submission and sync cost (bench_plan.md §4.1, H0.1..H0.7).
-- ---------------------------------------------------------------------------
if has_config("backend_cuda") then
    target("t0_cuda")
        set_group("benchmarks")
        add_rules("uipc_bench")
        add_files(table.join(bench_support_files, bench_tier0_files))
        add_files("cuda/bench_device_cuda.cu")
        -- Same dependency and the same .cu flags as the CUDA backend test target
        -- (apps/tests/backends/cuda/xmake.lua): inheriting them is the only way to
        -- prove the benchmark compiles under exactly the backend's CUDA settings.
        add_deps("cuda")
        if has_config("github_actions") then
            add_cugencodes("sm_89")
        else
            add_cugencodes("native")
        end
        add_cuflags("/wd4819", {tools = "cl"})
        add_cuflags("--extended-lambda", "--expt-relaxed-constexpr")
        if is_plat("windows") then
            add_cxflags("/utf-8", "/Zc:preprocessor")
            add_cuflags("-Xcompiler=/Zc:preprocessor")
        end
        if has_config("dev") then
            add_rules("clangd")
        end
end

if has_config("backend_luisa") then
    target("t0_luisa")
        set_group("benchmarks")
        -- "luisa.runtime" deploys the LC runtime DLLs beside the executable: a bench
        -- binary that picks up a different LC runtime than the module silently
        -- changes its numbers (cookbook §1.5).
        add_rules("uipc_bench", "luisa.runtime")
        add_files(table.join(bench_support_files, bench_tier0_files))
        add_files("luisa/bench_device_luisa.cpp")
        add_deps("luisa")
        if is_plat("windows") then
            add_cxflags("/utf-8", "/Zc:preprocessor")
        end
        if has_config("dev") then
            add_rules("clangd")
        end

    -- ---------------------------------------------------------------------------
    -- Engine-level A/B driver (optimize_plan.md OP-A1 / M-O1-M-O2): unlike the Tier-0
    -- binaries above, it drives the whole `uipc::core::World` pipeline through the
    -- `luisa` module and reports the engine's own per-frame counters
    -- (`Engine::frame_stats()`), which is what makes "commits/frame" measurable.
    -- It needs the core libraries (`app` -> uipc_deps) and the backend module, and it
    -- has its own `main`, so it must not see bench_main.cpp.
    -- ---------------------------------------------------------------------------
    target("engine_ab")
        set_group("benchmarks")
        add_rules("uipc_bench", "luisa.runtime")
        add_files("engine/ab_main.cpp")
        add_deps("app", "luisa")
        if is_plat("windows") then
            add_cxflags("/utf-8", "/Zc:preprocessor")
        end
        if has_config("dev") then
            add_rules("clangd")
        end

    -- ---------------------------------------------------------------------------
    -- Tier 1 — kernel / primitive microbenchmarks (bench_plan.md §4.2, task B3).
    --
    -- Structurally different from Tier 0 in exactly two places: the entry point
    -- (t1_main.cpp, because the case list is t1_registry()) and the device
    -- (luisa/bench_kernels_luisa.cpp: the *product* kernels, so this binary links the
    -- `luisa` module and includes the backend's headers, exactly like the backend test
    -- executable does).  It therefore also needs the core libraries and the LC define
    -- set (see luisa_compute_surface()).
    --
    -- Bench plan B4 (the CUDA counterpart, cuda/bench_kernels_cuda.cu) is not written
    -- yet: a Tier-1 case measured on one backend is still useful (absolute GB/s against
    -- K0's bandwidth band), but a luisa/cuda *ratio* needs both, so `t1_cuda` is
    -- deliberately absent instead of stubbed with a number that pretends to be
    -- comparable.
    -- ---------------------------------------------------------------------------
    -- PARKED: target("t1_luisa") -- see the note in tier1/t1_cases.cpp. The sources
    -- and the case list are written and compile; the binary aborts during static
    -- initialisation (0xC0000409) before main() when it links the `luisa` module and
    -- the uipc core together, which the backend test executable does successfully.
    -- Wiring it needs that startup difference understood first (t0_luisa links the
    -- same module and runs, so it is not the module itself).
        if has_config("dev") then
            add_rules("clangd")
        end
end
