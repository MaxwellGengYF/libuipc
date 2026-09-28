# T0.1 — LuisaCompute build & device spike report

**Status: PASSED** (2026-09-28). Branch `feature/luisa-backend`, base `9c748a75`.

## Objective

Prove, before any porting, that LuisaCompute (LC) can be consumed by libuipc and
actually run a DSL kernel on the local NVIDIA GPU — and lock in the dependency
integration strategy (plan risk R1).

## What was run

```bat
cmake -S scratch\lc_spike -B scratch\lc_spike\build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CXX_COMPILER=D:/TestUE/clang/bin/clang++.exe ^
      -DLUISA_COMPUTE_ROOT=C:/dev/LuisaCompute
cmake --build scratch\lc_spike\build
scratch\lc_spike\build\lc_spike.exe cuda
```

## Result

```
== LC spike == backend=cuda
  device: cuda
-- checks --
  [ ok ] Kernel1D write: buf[i] == 1 + i
  warp/atomic sum = 524800.0 (expected 524800.0, rel err 0.000e+00)
  [ ok ] warp_active_sum + atomic_fetch_add reduction
== spike PASSED ==
```

Exit code **0**. Additionally, 32 `[DEVICE] warp[N] partial=…` lines were captured,
confirming `device_log()` (the `muda::Logger`/`KernelCout` replacement) works and is
retrieved on dispatch/sync.

## Coverage matrix (spike ↔ plan requirements)

| Requirement | LC mechanism exercised | Result |
|---|---|---|
| Context / Device / Stream creation | `Context`, `ctx.create_device("cuda")`, `create_stream()` | ✅ `device: cuda` |
| Kernel1D compile | `device.compile<1>(lambda)` | ✅ |
| Dispatch | `Kernel1D().dispatch(N)` | ✅ N=1024 |
| Explicit runtime command path | `CommandList::create()` + `<<` + `commit()` | ✅ |
| Device↔host copy | `buf.copy_to(span)` + `synchronize()` | ✅ values verified |
| Device print / logger | `device_log("…", expr)` | ✅ 32 lines captured |
| Warp reduction | `warp_active_sum(v)` | ✅ exact |
| Atomic on buffer element | `buf->atomic(i).fetch_add(w)` | ✅ exact |
| GPU target | LC-CUDA backend (NVRTC → PTX) on RTX 4060 | ✅ |

## Decisions recorded

1. **Dependency strategy:** `find_package(LuisaCompute CONFIG REQUIRED)` against the
   pinned install prefix `C:/dev/LuisaCompute/install`; option-gated by
   `UIPC_WITH_LUISA_BACKEND` (default OFF). Rationale + contingency in
   `docs/luisa_porting_cookbook.md` §1.
2. **Runtime device:** `"cuda"` by default (LC-CUDA), selectable via engine config.
3. **Toolchain:** consumers need a conformant preprocessor (`clang++/clang-cl`, or MSVC
   with `/Zc:preprocessor`); Release config (LC ships release import libs only).
4. **LC must be rebuilt after source changes.** The prebuilt `luisa-ast.dll` in this
   checkout was stale and did not export
   `luisa::compute::detail::debug_function_if_enabled`, so *every* consumer of
   `device.compile` failed to link. Rebuilding LC (source commit `112bf1daf` "Fix build")
   fixed it. Documented in cookbook §1.2.

## Artifacts

- `scratch/lc_spike/CMakeLists.txt`, `scratch/lc_spike/main.cpp` — the spike (reused as
  the template for `apps/tests/backends/luisa/hello_luisa` in T1.3).
- `scratch/spike_run_raw.txt` — raw captured run output.
- Cookbook §1 (strategy) and §Appendix (verified LC API cheat-sheet).

## Follow-ups carried into later tasks

- R9: confirm the remaining LC intrinsics used by the backend (shuffles, bit ops,
  fences, `Shared<T>` block reductions) exist — first needed in T2/T3 (`builtins.h`).
- The spike's `LBuffer`-style wrapper design (cookbook §3.2) is implemented in T1.2.
