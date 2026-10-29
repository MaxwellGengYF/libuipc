includes("components.lua")

target("cuda")
    add_rules("backend")
    uipc_add_cuda_component_sources()
    if has_config("dev") then
        add_rules("clangd")
    end
    add_headerfiles("**.h", "**.inl")
    -- public: the backend's own public headers (`cuda_tool/launch_base.h`) include the
    -- shared instrumentation header <backends/common/nvtx_range.h>, so every consumer of
    -- this module (the backend test target, the benchmark binaries) needs `src/` on its
    -- include path too. Before BN1 nothing in `cuda_tool/` reached outside it, and the
    -- `src/` entry was private, which broke the CUDA test target's build.
    add_includedirs(path.join(os.projectdir(), "src"), {public = true})
    add_includedirs(os.scriptdir(), {public = true})
    -- cuda_tool (self-written device utilities) on the include path for <cuda_tool/...>
    add_includedirs(path.join(os.scriptdir(), "cuda_tool"), {public = true})
    -- BN1 (bench_plan.md 13.3): NVTX ranges behind one compile-time switch, UIPC_NVTX.
    -- nvtx3 is header-only on this machine (CUDA 12.9 ships include/nvtx3 and no
    -- nvToolsExt64_1.lib/.dll; nvtxInit resolves libnvToolsExt with LoadLibraryW), so this
    -- adds an include dir and a define, never a link dependency. Off by default: without
    -- `xmake f --nvtx=y` nothing is added and UIPC_NVTX stays 0
    -- (src/backends/common/nvtx_range.h), so a throughput run is byte-identical to before.
    if has_config("nvtx") then
        local cuda_inc = path.join(os.getenv("CUDA_PATH") or
                                   "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9",
                                   "include")
        if not os.isdir(path.join(cuda_inc, "nvtx3")) then
            raise("--nvtx=y needs a CUDA toolkit include dir containing nvtx3/: " ..
                  cuda_inc .. " (set CUDA_PATH)")
        end
        add_includedirs(cuda_inc)
        add_defines("UIPC_NVTX=1")
    end
    if has_config("github_actions") then
        add_cugencodes("sm_89")
    else
        add_cugencodes("native")
    end
    add_rules("cuda_warning")
    add_rules("cuda_no_host_compiler_check")    
    add_cuflags("--expt-relaxed-constexpr")
    add_cuflags("--extended-lambda")
    -- RDC is required: affine_body/utils.cu defines UIPC_GENERIC free functions
    -- (e.g. q_to_transform) called from device code in other TUs; without
    -- separable compilation ptxas fails with "Unresolved extern".
    -- (parity with CUDA_SEPARABLE_COMPILATION ON in CMakeLists.txt)
    add_cuflags("-rdc=true")
    
    add_links(
        "cudart"
    )

    add_deps("uipc_geometry", "uipc_io")
    on_load(function(target)
        if target:is_plat('windows') then
            target:add('defines', '__NV_NO_HOST_COMPILER_CHECK', {public = true})
            target:add('cuflags', '-allow-unsupported-compiler', {public = true})
            -- Suppress MSVC C4819 for host compilation of CUDA sources.
            target:add('cuflags', '-Xcompiler=/wd4819', {public = true})
            -- CUDA >= 13 CCCL requires the standard-conforming MSVC preprocessor
            -- (parity with src/backends/cuda/CMakeLists.txt).
            target:add('cuflags', '-Xcompiler=/Zc:preprocessor', {public = true})
            target:set('toolchains', 'msvc')
        end
        target:set('toolchains', 'cuda')
    end)
