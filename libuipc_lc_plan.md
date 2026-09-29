# libuipc LuisaCompute Backend — Master Implementation Plan

## 0. Metadata
- Repo: `D:\libuipc` (Windows; both CMake + xmake builds)
- New backend target dir: `D:\libuipc\src\backends\luisa\`
- Reference backend: `D:\libuipc\src\backends\cuda\` (205 `.cu` files, ~25 subsystems)
- Compute framework: **LuisaCompute (LC)** — see skill `D:\compute\.agents\skills\lc_api\SKILL.md`
- LC source tree assumed available at `D:\compute` (verify in T0.1; adjust paths if different)
- Build entry points to touch: `src\backends\CMakeLists.txt`, root `CMakeLists.txt`, `src\backends\xmake.lua`, root `xmake.lua`, `CMakePresets.json`

---

## 1. Scope & Deliverables (from requirement)

1. **Kernels**: every CUDA kernel currently compiled by nvcc and running on the GPU device → LuisaCompute `Kernel1D` DSL kernels.
2. **Runtime**: every CUDA/muda runtime command (kernel dispatch, device↔host copy, fill, stream wait, device reductions) → LuisaCompute runtime commands, expressed with LC `CommandList` + `Stream`.
3. **Tests**: port/author tests proving the Luisa re-write (unit kernels, backend GPU sanity, full sim-case regressions, and CUDA-vs-Luisa numerical equivalence).

This is a **paradigm migration**, not a line-by-line port. muda device lambdas + Eigen-on-device become LC AST-built kernels (`Var<T>`, DSL control flow). All ported sources are plain C++20 (`.cpp`) — **nvcc is eliminated entirely** for this backend.

---

## 2. Measured Current State (facts gathered by exploration)

### 2.1 Codebase inventory (src/backends/cuda)
| Area | Files (.cu) | Notes |
|---|---|---|
| engine/ | 8 | sim_engine + do_init/advance/sync/retrieve/backward + advance_ipc/al |
| finite_element/ | ~62 | incl. bdf/, codim constitutions, constitutions/ (12 constitutions + sym/*.inl), constraints/, reporters, preconditioners |
| affine_body/ | ~45 | incl. bdf/, constitutions/ (arap, ortho, 4 joint types + driving variants), constraints/, inter_* |
| collision_detection/ | ~12 | atomic_counting_lbvh, linear_bvh, stackless_bvh, info_stackless_bvh(_v0), 8 trajectory filters |
| contact_system/ | ~17 | IPC/codim/AL contact models, vertex-half-plane models, analytical_barrier_pFpx (pFpx PT/PP/PE/EE/parallel-*), exporters |
| linear_system/ | 10 | global_linear_system, pcg, linear_fused_pcg, spmv, iterative_solver, subsystems, preconditioners |
| global_geometry/ | 6 | vertex/body/simplicial-surface managers + reporters |
| finite_element/mas | (part of FE) | mas_preconditioner + mas_preconditioner_engine (22 muda refs) |
| misc systems | ~30 | diff_sim(4), dytopo(6), line_search(2), newton_tolerance(3), active_set(3), coupling(3), external_force(2), animator(2), joint_dof(2), time_integrator(4), pipeline flags(2), distance_system(1), inter_primitive(6), implicit_geometry(3), sanity_check(3), kernel_cout(1) |
| algorithm/ | headers/.inl only | matrix_converter, fast_segmental_reduce, givens.hpp, qr_svd.hpp (device headers) |
| **Total** | **205 .cu** | plus ~120 headers/.inl; `sym/*.inl` = SymEigen-generated elementwise math (~40 files) |

### 2.2 Technology dependencies used by CUDA backend
- **muda** (`muda::muda` target): `ParallelFor`, `DeviceBuffer<T>`, `DeviceVar<T>`, `BufferView/CBufferView`, `muda::wait_device`, `Debug::set_sync_callback`, `muda::Logger` (kernel printf).
- **muda ext/eigen**: `eigen_cxx20.h`, `ext/eigen/atomic.h` (atomicAdd on Eigen vectors), `ext/eigen/inverse.h`, `ext/eigen/evd.h` (used by `utils/make_spd.h`), `ext/eigen/log_proxy.h`.
- **muda ext/linear_system**: `DeviceTripletMatrix`, `DeviceDoubletVector`, `DeviceBCOOMatrix`, `DeviceBSRMatrix`, `DeviceBCOOVector`, plus matrix views (25 files, 115 refs).
- **CUB via muda**: `device_radix_sort`, `device_scan`, `device_reduce`, `device_select`, `device_partition`, `device_merge_sort`, `device_run_length_encode` (in matrix_converter + filters + line_search).
- **cuBLAS/cuSPARSE/cuSOLVER**: only linked (CMakeLists/xmake). Grep confirms **no runtime API usage remains** (`linear_fused_pcg.cu` is explicitly "cublas-free, device-only, CUB warp reduction"). → Do NOT link or port these.
- **Eigen**: host-side (scene/geometry attrs) and device-side (small fixed-size matrices). `sym/*.inl` generated functions are pure elementwise scalar arithmetic over `Eigen::Vector<T,9>`/`Matrix<T,9,9>` etc. — no linear-algebra calls inside the .inl bodies.
- Type system: `uipc::Float/Vector3/Matrix3x3/IndexT` via `type_define.h`; muda forces Eigen trivially-copyable traits.

### 2.3 Backend loading architecture
- `src/backends/common/` holds the shared engine framework (`SimEngine`, `ISimSystem`, `SimSystemCollection`, registration). `uipc_add_backend(name)` in `src/backends/CMakeLists.txt` builds each backend as MODULE (dyn-load) or SHARED (when `UIPC_BUILD_TESTS=ON`), always compiling `backends/common/*` into the backend.
- Backend entry: `entrance.cpp` exports `uipc_create_engine` / `uipc_destroy_engine` (`backends/common/module.h`).
- CUDA backend wraps common with thin namespaced headers: `sim_engine.h`, `sim_system.h`, `sim_action.h`, `sim_system_slot.h`, `sim_action_collection.h`, `sim_engine_state.h`, `type_define.h`, `sim_system.cpp`.
- Sim systems self-register via `REGISTER_SIM_SYSTEM(T)` (static registration) and wire up in `SimEngine::build()`.
- Engine pipeline state machine: `do_init → per-frame {advance(newton iters, line search, linear solve) → sync → retrieve}`, plus dump/recover.

### 2.4 Test infrastructure
- `apps/tests/` — Catch2 (`catch_main.cpp`). `uipc_add_test(name ...)` helper.
- `apps/tests/sim_case/` — 93 sim cases (gravity, contact, joints, shells, MAS, codim, recover…). They load a backend by name through the public engine API → backend-agnostic; can be re-targeted at `luisa`.
- `apps/tests/backends/cuda/` — GPU unit tests (hello_muda, gpu_sanity_check, lbvh, stackless/info_stackless_bvh, shell bending, ee_distance_flag).
- `apps/tests/sanity_check/`, `apps/tests/regression/`, `python/tests/sim_case/`.
- Backend selection: engine creation is by backend module name (see `apps/tests/core/engine.cpp`), so sim_case runs against any built backend.

---

## 3. Target Architecture & Migration Mapping

### 3.1 Core muda → LC mapping (the contract every port task follows)

| muda / CUDA concept | LuisaCompute replacement |
|---|---|
| `muda::ParallelFor().apply(N, [..] __device__(int i){...})` | `Kernel1D` lambda built & compiled once at system build (`device.compile<1>`), dispatched per frame: `stream << cmd_list;` `kernel(bufs...).dispatch(N)` |
| `__device__` lambda body | DSL lambda body operating on `luisa::compute::Var<T>` / DSL exprs; runtime control flow via `$if/$for/$while/$switch` (or `if_`/`for(dynamic_range)`) |
| `muda::DeviceBuffer<T>` | `luisa::compute::Buffer<T>` owned via `LBuffer<T>` wrapper with muda-like `resize/reserve/view/cviewer` |
| `muda::BufferView<T>` / `CBufferView<T>` | `LBufferView<T>` / `LCBufferView<T>` (custom view: `{shared_ptr<Buffer<T>> base, size_t offset, size_t size}`) — keeps downstream API signatures nearly identical |
| `muda::DeviceVar<T>` | `LBuffer<T>` of size 1 (or `LVar<T>`), same view pattern |
| `.viewer()/.cviewer()` (indexed read/write, `.name()`) | `BufferVar<T>` in kernel params: `buf->read(i)` / `buf->write(i,v)` (name() maps to LC kernel-arg naming/debug) |
| `muda::wait_device()` / stream sync | `stream.synchronize()` (or LC `barrier()`); engine holds one main `Stream` + `CommandList` |
| `muda::Logger` / `KernelCout` | LC device print: `device_log(...)` / `Printer` captured per stream; retrieval hook on sync |
| Eigen `Matrix3x3/Vector3` etc. in kernels | `lps::LMat<T,M,N>` / `lps::LVec<T,N>` fixed-size shim + LC scalar math (see §3.2) |
| `atomicAdd(&v, x)` on scalars | `buf.atomic(i).fetch_add(x)` / `atomic_fetch_add(&var, x)` |
| `atomicAdd` on Eigen `Vector3` | 3 scalar atomic fetch_adds (shim `lps::atomic_add`) |
| CUB `radix_sort/scan/reduce/select/rle/partition/merge_sort` | Custom LC kernels (bucket sort by row, block scan, warp reductions, stream-compaction) — see Phase 3 |
| `muda::ext::linear_system` containers | `lps::` device linear-algebra containers re-implemented on LC buffers (Phase 3) |
| `cuBLAS/cuSPARSE/cuSOLVER` | **not needed** (confirmed unused at API level) |

### 3.2 Key design decisions

**D1 — Math shim instead of Eigen-in-kernels (CRITICAL).** LC kernels cannot use Eigen (AST-built). `sym/*.inl` bodies are pure elementwise arithmetic, so they port by replacing `Eigen::Vector<T,N>`→`lps::LVec<T,N>`, `Eigen::Matrix<T,R,C>`→`lps::LMat<T,R,C>` and `std::pow/min/max...`→`lps::pow/min/max...`. The shim is a header-only, compile-time-sized type whose operators work with `T=float/double` (host tests) **and** `T=luisa::compute::Float/Double` DSL expressions (device). Provide: `operator()`, `operator[]`, `+ - * /` (scalar & mat-vec, mat-mat for sizes used), `.transpose()`, `.block()`, ` determinant/inverse` (3x3, 4x4), and `lps` math namespace (`pow, sqrt, exp, log, sin, cos, tan, asin, acos, atan, atan2, abs, floor, ceil, min, max, clamp, copysign, fmod, select`).

**D2 — Buffer/View wrapper preserving muda semantics.** LC `Buffer<T>` is fixed-size; muda `DeviceBuffer` resizes. `LBuffer<T>` reallocates on growth and hands out `LBufferView<T>` value types referencing a `shared_ptr<Buffer<T>>` + offset + size, so existing code that stores/passes views (very common) ports with minimal diff. All kernels take views (not raw buffers) as arguments. Resizing while views exist is a use-after-free risk — documented rule: views are short-lived (frame-scoped), matching muda usage.

**D3 — Compile-once, dispatch-per-frame.** Each kernel site compiles its `Kernel1D`/`Callable` in `do_build()`/`do_init()` (stored as `luisa::compute::Shader` handles), records dispatches into a per-frame `CommandList`, and commits to the engine's `Stream`. A small `lps::KernelCache` keyed per call-site avoids duplicate compiles.

**D4 — Linear algebra re-implementation (no CUB, no cuSPARSE).** Phase 3 delivers: doublet/triplet/BCOO/BSR containers + `MatrixConverter` (bucket/radix sort via per-row atomic counting + block scan; segment reduction for duplicate (i,j); RLE for row offsets) + SpMV kernels (incl. fused dot `rbk_sym_spmv_dot` via LC warp reductions: `warp_active_sum`, block reduce + `Shared<T>` scratch) + dense BLAS-ish kernels (dot, axpy, fill, norm2, segment reduce). Reference CPU implementations kept for tests.

**D5 — Backend shell mirrors CUDA shell.** Namespace `uipc::backend::luisa`. Same header names (`type_define.h`, `sim_system.h`, `sim_engine.h`, …) so ported `.cu` bodies change includes/namespaces but not structure. `REGISTER_SIM_SYSTEM` preserved.

**D6 — LC device = LC-CUDA backend.** The luisa backend uses LC's CUDA backend (NVRTC → PTX), still targeting NVIDIA GPUs. LC context creation string e.g. `"cuda"`; keep backend name `luisa` (independent of LC's internal backends). Optionally allow `"dx"`/`"vk"` later via config; default cuda.

**D7 — Reduction/sort performance parity is NOT a correctness gate.** CUB replacements may be slower initially; correctness first, then optimize (lc_optimize skill) only where profiles demand.

### 3.3 Risks
A full risk register (15 risks with likelihood / impact / mitigation / detection / contingency) is maintained in **Appendix A** and must be re-reviewed at every milestone. Top 5:
1. LC dependency & build-integration uncertainty (R1).
2. Hidden Eigen linear-algebra surface in kernels exceeds the Phase-0 inventory (R2).
3. LC DSL control-flow misuse silently flattening the AST (R3).
4. Numerical divergence vs CUDA breaking regression parity (R6).
5. Multi-session execution losing state (R15) — mitigated by the milestone & progress-tracking protocol in §5.0.

---

## 4. New Backend Directory Layout

```
src/backends/luisa/
  CMakeLists.txt                     (new; mirrors cuda/CMakeLists.txt, links lc targets)
  xmake.lua                          (new; mirrors cuda/xmake.lua)
  entrance.cpp                       (uipc_create_engine/destroy)
  type_define.h                      (Float/IndexT/Vector3/Matrix3x3 → LC-compatible aliases + LVec/LMat forward)
  sim_engine.h / sim_engine.cpp      (SimEngine port, namespace uipc::backend::luisa)
  sim_engine_state.h
  sim_system.h / sim_system.cpp
  sim_system_slot.h
  sim_action.h / sim_action_collection.h
  kernel_cout.h / kernel_cout.cpp    (device_log wrapper)
  luisa_device/                      (foundation runtime layer, Phase 1)
    context.h/.cpp                   (LC Context/Device/Stream lifecycle)
    buffer.h                         (LBuffer<T>, LBufferView<T>, LCBufferView<T>)
    kernel.h                         (KernelCache, parallel_for helper, compile-once helpers)
    stream.h                         (CommandList + Stream facade, wait())
    log.h                            (device print + retrieval)
    builtins.h                       (bit ops, warp shims, math lps:: aliases)
  luisa_device/math/                 (Phase 2)
    vec.h mat.h                      (LVec/LMat fixed-size, host+DSL)
    math.h                           (lps math namespace)
    matrix_ops.h                     (det/inverse/transpose/solve small, make_spd Jacobi-EVD)
    svd.h                            (3x3 SVD via Givens/QR port of algorithm/givens.hpp, qr_svd.hpp)
    distance.h                       (point-point/edge/triangle/edge-edge + mollifier + CCD)
    dihedral.h thickness.h friction.h
  luisa_device/linear/               (Phase 3)
    containers.h                     (doublet/triplet vector/matrix, BCOO/BSR, dense vectors + views)
    converter.h/.cpp                 (MatrixConverter port)
    spmv.h/.cpp                      (SpMV incl. fused-dot)
    blas.h/.cpp                      (dot/axpy/fill/norm/segment-reduce)
    algorithms.h/.cpp                (sort/scan/select/rle/reduce primitives)
  <all subsystem dirs same names as cuda backend>/
  engine/
  finite_element/ ... affine_body/ ... etc.
```

---

## 5. Phase & Subagent Task Breakdown

Conventions:
- **Task id**: T<phase>.<n>. Each task = one subagent assignment (or explicitly parallelizable batch).
- **Deps**: task ids that must complete first (their deliverables are inputs).
- **Files**: primary paths created (➕) / ported (⇄ from src/backends/cuda/...).
- **Done**: acceptance criteria the subagent must satisfy before marking complete.
- Porting authority: **`docs/luisa_porting_cookbook.md`** (authored in T1.0). Every port task must follow it; every new pattern discovered must be appended to it.

---

### 5.0 Progress Tracking, Validation & Milestone Protocol (MANDATORY)

This plan is too large for one session. State MUST be persisted so a fresh session can resume without re-deriving anything.

**Git workflow**
- Work on branch `feature/luisa-backend` (created in T1.0).
- Every completed task = one commit: message `[luisa-backend] T<x>.<y> <short title>`. The commit MUST include: ported code, its tests, cookbook additions, and the updated progress doc.
- When every task of a milestone passes validation: `git tag luisa-m<N>` (annotated, listing milestone exit criteria).

**Progress doc**
- `docs/luisa_backend_progress.md` (template in Appendix B) is the single source of truth. It contains: a global status header (current milestone, last completed task, commit sha, next action) and a per-task table (Task | Title | Milestone | Status ⬜ todo / 🟡 in-progress / ✅ done | Commit | Validation evidence | Notes).
- A subagent MUST update this file before reporting its task done. Validation evidence = the exact command(s) run and their pass/fail summary (or artifact path, e.g. ctest log, regression CSV).

**Resume protocol for a new session**
1. `git status` and `git log --oneline -20 --grep=luisa-backend`.
2. Read `docs/luisa_backend_progress.md` (global header + rows since last tag).
3. Read `docs/luisa_porting_cookbook.md`.
4. Continue the next ⬜ task in the current milestone (scheduling per §6).
5. Before committing, re-run the task's **Validate** line; attach evidence to the progress doc.

**Milestones (exit criteria = all tasks ✅ + validations green)**

| Milestone | Tasks | Exit criteria |
|---|---|---|
| M0 Spike & Inventory | T0.1, T0.2 | LC spike binary runs on GPU; dependency strategy decided; API inventory covers 100% symbols |
| M1 Foundation | T1.0–T1.4 | Backend skeleton builds (CMake+xmake), loads; device core tests green; shell system plumbing works |
| M2 Math layer | T2.1–T2.3 | LVec/LMat + ops/SVD/distance host+device parity vs Eigen/CUDA-ref within tol |
| M3 Linear algebra | T3.1–T3.4 | Containers, converter, SpMV, BLAS kernels verified vs references |
| M4 Base systems | T4.1–T4.3 | Geometry/manager/animator systems build + unit-tested |
| M5 FEM | T5.1–T5.4b | All FEM constitutions & dynamics verified (E/G/H vs CUDA); T5.5 re-scoped (AL reporters → M8, diff reporters → M9), MAS preconditioner → M6 |
| M6 ABD | T6.1–T6.4 | All ABD constitutions/joints/constraints verified |
| M7 Collision | T7.1–T7.2 | BVH + trajectory filters equivalent to CUDA |
| M8 Contact→Engine | T8.1–T8.5 | Contact models, coupling, PCG, full engine link; frame 0 of a gravity case advances |
| M9 Verification | T9.1–T9.4 | Unit + sim-case regression + equivalence harness green; baselines recorded |
| M10 Integration | T10.1–T10.2 | Clean builds both generators/options; CI green; docs merged |

---

### PHASE 0 — Validation Spike & Inventory (prerequisite for everything)

**T0.1 — LC build & device spike** (no libuipc integration yet)
- Verify LuisaCompute availability at `D:\compute` (LC source root), identify cmake package targets (`lc-core`, `lc-runtime`, `lc-dsl`, `lc-vstl`), and backend enable flags (`LUISA_COMPUTE_ENABLE_CUDA`).
- Create a scratch standalone CMake project that: creates `Context`, creates Device with `"cuda"`, compiles a trivial `Kernel1D` (buffer write), dispatches, copies back, verifies value; also test `Printer`/`device_log` retrieval and a `CommandList` commit.
- Deliverable: spike report + minimal working example committed under `apps/tests/backends/luisa/` later reused by T9.1. **Decision recorded: LC dependency integration strategy** (add_subdirectory vs find_package vs FetchContent) for `CMakeLists.txt` and `xmake.lua`.
- **Done**: spike binary runs on the local NVIDIA GPU; strategy doc note added to cookbook.
- **Validate**: run the spike executable on the local NVIDIA GPU → exit 0 and the kernel-written buffer value prints correct; `docs/luisa_porting_cookbook.md` contains the “LC dependency strategy” note.
- **Milestone**: M0 — commit `[luisa-backend] T0.1 lc-build-device-spike`; update progress doc.

**T0.2 — Full muda/Eigen/CUB API inventory**
- Grep all 205 `.cu` + headers for every `muda::` symbol, Eigen member used inside `__device__`/lambda bodies, CUB algorithm, and intrinsic (`__shfl`, `__ballot`, atomics, `__brev` etc.).
- Produce `docs/luisa_api_inventory.md`: table `symbol | file:line | LC replacement | shim needed?`.
- Deliver the definitive list the math shim (Phase 2) and linear layer (Phase 3) must cover.
- **Done**: inventory doc covers 100% of matches; each entry has a mapped replacement.
- **Validate**: re-run the grep/inventory script → 0 unmapped symbols; spot-check ≥20 random `file:line` entries for correct LC replacement.
- **Milestone**: M0 — commit `[luisa-backend] T0.2 muda-eigen-cub-api-inventory`; `git tag luisa-m0`.

---

### PHASE 1 — Foundation: Build, Device, Runtime Shell

**T1.0 — Author the Porting Cookbook + backend skeleton**
- Write `docs/luisa_porting_cookbook.md`: the §3.1 mapping table as copy-paste code patterns (muda snippet → luisa snippet), DSL control-flow rules, Eigen→shim rewrite rules, REGISTER_SIM_SYSTEM usage, naming/extension rules (`.cu`→`.cpp`, namespace `uipc::backend::luisa`), view-lifetime rule.
- Create `src/backends/luisa/` skeleton: all empty subsystem dirs mirroring cuda backend; placeholder `entrance.cpp` returning a stub engine (modeled on `src/backends/none/`).
- **Files**: ➕ `docs/luisa_porting_cookbook.md`; ➕ `src/backends/luisa/**` skeleton.
- **Done**: skeleton compiles as a MODULE/SHARED lib via a new option and loads through the existing backend loader.
- **Validate**: `dir src\backends\luisa` subsystem tree matches the CUDA backend list; cookbook has all sections in §3.1 as code patterns; backend module `uipc_backend_luisa` produced and loadable.
- **Milestone**: M1 — commit `[luisa-backend] T1.0 porting-cookbook-and-skeleton`.

**T1.1 — Build integration (CMake + xmake)**
- Root `CMakeLists.txt`: `option(UIPC_WITH_LUISA_BACKEND "... " OFF)`; `src/backends/CMakeLists.txt`: `if(UIPC_WITH_LUISA_BACKEND) add_subdirectory(luisa) endif()`. Wire LC dependency per T0.1 decision (keep it lazy: only when option ON).
- `src/backends/luisa/CMakeLists.txt`: `uipc_add_backend(luisa)`; link `lc-runtime lc-dsl lc-vstl lc-core`, `uipc_geometry`, `uipc_io`; `cxx_std_20`; GLOB `*.cpp/*.h/*.inl`; NO CUDA language.
- Port `src/backends/luisa/xmake.lua` mirroring cuda's (add_requires LC / package integration, files `**.cpp`, remove cuflags/cudagen/links cublas etc.).
- **Files**: ⇄ root `CMakeLists.txt`, `src/backends/CMakeLists.txt`, `src/backends/xmake.lua`; ➕ `src/backends/luisa/CMakeLists.txt`, `xmake.lua`.
- **Done**: both generators build the (stub) backend with the option ON; default build (option OFF) unchanged.
- **Validate**: `cmake -DUIPC_WITH_LUISA_BACKEND=ON` full build passes; option-OFF configure+build passes (parity check); `xmake build` passes. Record generator versions.
- **Milestone**: M1 — commit `[luisa-backend] T1.1 build-integration-cmake-xmake`.

**T1.2 — `luisa_device` core runtime** (depends T0.1, T1.1)
- Implement foundation layer (§4 `luisa_device/`):
  - `context.h/.cpp`: LC `Context` singleton + `Device` + main `Stream`; config from engine config JSON; shutdown order safe (LC teardown after all buffers die).
  - `buffer.h`: `LBuffer<T>` (resize/reserve/clear/copy_from/copy_to), `LBufferView<T>`/`LCBufferView<T>` with offset+count+shared ownership; `subview` helpers.
  - `kernel.h`: `KernelCache` (compile-once per call site), `parallel_for(N, DSL-lambda, views...)` facade used by ported code in place of `muda::ParallelFor`.
  - `stream.h`: `LStream` facade exposing `command_list()`, `commit()`, `wait()`; engine-wide frame command list.
  - `log.h`: kernel print capture (replace `muda::Logger`); sync-time retrieval like `kernel_cout.cu`.
  - `builtins.h`: bit ops, warp primitives shims (`warp_active_sum` etc.), atomic shims.
- **Files**: ➕ `src/backends/luisa/luisa_device/*`.
- **Done**: unit tests in T1.3 pass.
- **Validate**: `ctest -R luisa_device` green; a debug run shows no LC resource leaks after context teardown (log check).
- **Milestone**: M1 — commit `[luisa-backend] T1.2 luisa-device-core-runtime`.

**T1.3 — Foundation unit tests** (depends T1.2)
- ➕ `apps/tests/backends/luisa/CMakeLists.txt` mirroring `apps/tests/backends/cuda/CMakeLists.txt`; tests: `hello_luisa` (kernel write + copy back), `buffer_ops` (resize/fill/copy/subview), `parallel_for` (index correctness + reduction), `kernel_print` (device_log retrieval), `stream_commands` (command-list batching + event/wait). Test target links `uipc::backend::luisa`; register via `uipc_add_test`.
- **Done**: all green on the local GPU; tests run under ctest.
- **Validate**: `ctest -R backend_luisa --output-on-failure` → all tests pass; ctest log attached to progress doc as evidence.
- **Milestone**: M1 — commit `[luisa-backend] T1.3 foundation-unit-tests`.

**T1.4 — Backend shell headers + SimSystem plumbing** (depends T1.0)
- Port the thin shell: `type_define.h` (aliases `Float=uipc::Float`, `IndexT`, `Vector3`, `Matrix3x3`, `AABB`, plus LVec/LMat includes), `sim_system.h/.cpp`, `sim_system_slot.h`, `sim_action.h`, `sim_action_collection.h`, `sim_engine.h/.cpp` (stub do_* until Phase 8), `sim_engine_state.h`.
- ➕ `kernel_cout.h/.cpp` (LC device print facade).
- **Files**: ⇄ from `src/backends/cuda/*.h|cpp` (the 8 shell files listed in §2.3).
- **Done**: a `REGISTER_SIM_SYSTEM`-registered dummy system builds and `find<T>()`/`require<T>()` resolve inside a stub engine.
- **Validate**: compile+run a stub-engine test that registers a dummy system and calls `find`/`require` → both resolve; negative case (`require` missing system) throws `SimEngineException`.
- **Milestone**: M1 — commit `[luisa-backend] T1.4 backend-shell-sim-system`; `git tag luisa-m1`.

---

### PHASE 2 — Math Compatibility Layer (`luisa_device/math`) (depends T1.2)

**T2.1 — LVec/LMat + scalar math** (depends T0.2 inventory)
- Implement `vec.h`, `mat.h`, `math.h`: compile-time-sized `LVec<T,N>`, `LMat<T,R,C>` supporting host `T=float/double` and DSL `T=luisa::compute::Float/Double`; storage as plain array so `LUISA_STRUCT` registration can be added where buffers of them are needed; operators: indexing `v(i)`, `m(r,c)`, arithmetic scalar/mat/mat-vec, `transpose`, `block`, `norm`, `normalized`, `dot`, `cross`; `lps` math namespace.
- Host unit tests validating identical results vs Eigen for the sym-function use cases (vector9/matrix9x9 products).
- **Files**: ➕ `src/backends/luisa/luisa_device/math/{vec.h,mat.h,math.h}`; ➕ test file under `apps/tests/backends/luisa/`.
- **Done**: host-side parity with Eigen within 1e-6 for representative ops; DSL smoke kernel using LVec/LMat compiles & runs.
- **Validate**: `test_math_host` (LVec/LMat vs Eigen, 1e-6) and `test_math_smoke` (device kernel) both pass.
- **Milestone**: M2 — commit `[luisa-backend] T2.1 lvec-lmat-math-shim`.

**T2.2 — Small matrix ops + make_spd + SVD** (depends T2.1)
- `matrix_ops.h`: `det/inverse/transpose` for 2x2/3x3/4x4; `make_spd` via Jacobi eigenvalue iteration (port logic of `utils/make_spd.h` + `muda/ext/eigen/evd.h` usage); LDL/Cholesky fallback if simpler.
- `svd.h`: port `algorithm/givens.hpp` + `algorithm/qr_svd.hpp` 3x3 SVD/polar decomposition used by affine-body constitutions.
- `dihedral.h`, `thickness.h`, `friction.h`, `matrix_ops` tests.
- **Files**: ➕ `src/backends/luisa/luisa_device/math/{matrix_ops.h,svd.h,dihedral.h,thickness.h,friction.h}`.
- **Done**: unit tests vs Eigen (`Eigen::JacobiSVD`, `SelfAdjointEigenSolver`) within tolerance, host + one device kernel.
- **Validate**: `test_matrix_ops` host+device pass; `make_spd` output min eigenvalue ≥ -1e-8 on 200 random symmetric matrices; `svd` reproduces known polar decompositions.
- **Milestone**: M2 — commit `[luisa-backend] T2.2 matrix-ops-make-spd-svd`.

**T2.3 — Distance functions + CCD** (depends T2.1)
- Port `utils/distance/*` (`point_point.h`, `point_edge.h`, `point_triangle.h`, `edge_edge.h`, `edge_edge_mollifier.h`, `ccd.h` + `details/*.inl`, `distance_flagged.h`) onto LVec math with DSL control flow.
- **Files**: ➕ `src/backends/luisa/luisa_device/math/distance*` (mirror cuda's `utils/distance/` structure).
- **Done**: randomized host tests vs CUDA reference implementations (recompile cuda headers on host with double) within 1e-5; device smoke test.
- **Validate**: `test_distance` with ≥10k random PP/PE/PT/EE configs (incl. near-degenerate) → max abs error < 1e-5; device smoke kernel returns expected flags.
- **Milestone**: M2 — commit `[luisa-backend] T2.3 distance-ccd-port`; `git tag luisa-m2`.

---

### PHASE 3 — Linear Algebra Containers & Algorithms (depends T1.2, T2.1)

**T3.1 — Containers + views**
- Port `algorithm/matrix_converter.h` types + `muda/ext/linear_system` containers: `LTripletMatrix<T,N>`, `LDoubletVector<T,N>`, `LBCOOMatrix<T,N>`, `LBSRMatrix<T,N>`, `LBCOOVector<T,N>`, dense vectors, and their const/mutable view types with the same accessor surface used by the 25 referencing files.
- **Files**: ➕ `src/backends/luisa/luisa_device/linear/containers.h`.
- **Done**: containers compile; resize/clear/view semantics match muda; host round-trip tests.
- **Validate**: `test_linear_containers` round-trip passes (fill → view → device copy back → compare), covering doublet/triplet/BCOO/BSR + dense vectors, block sizes 1 and 3.
- **Milestone**: M3 — commit `[luisa-backend] T3.1 linear-containers-views`.

**T3.2 — Device algorithm primitives**
- Implement in `algorithms.h/.cpp`: bucket sort by row key (atomic per-row counts + block exclusive scan), segmented reduce (duplicate (i,j) combine), select-if (stream compaction), run-length encode, device reduce (block reduce + atomic fold), merge-sort only if inventory shows need beyond radix-by-key.
- **Files**: ➕ `src/backends/luisa/luisa_device/linear/algorithms.h/.cpp`.
- **Done**: each primitive has a randomized host-verified device test.
- **Validate**: `test_primitives` (bucket-sort, segmented-reduce, select-if, RLE, block reduce) randomized vs CPU reference passes for sizes 1..1e6.
- **Milestone**: M3 — commit `[luisa-backend] T3.2 device-algorithm-primitives`.

**T3.3 — MatrixConverter port** (depends T3.1, T3.2)
- Port `algorithm/details/matrix_converter.inl` + `matrix_converter.h`: Triplet→BCOO, BCOO→BSR, Doublet→BCOO conversions using T3.2 primitives.
- **Files**: ➕ `src/backends/luisa/luisa_device/linear/converter.h/.cpp`.
- **Done**: numerical equivalence tests vs the CUDA MatrixConverter on random sparse matrices (block sizes 3 and 1), incl. duplicate accumulation order-insensitive check.
- **Validate**: `test_matrix_converter` — random sparse inputs (density sweep 1%–50%, duplicates injected) → converted matrix equals CUDA-reference result within 1e-6; row-pointer/column-index arrays identical.
- **Milestone**: M3 — commit `[luisa-backend] T3.3 matrix-converter-port`.

**T3.4 — SpMV + dense BLAS kernels** (depends T3.1)
- Port `linear_system/spmv.h/.cu`: `sym_spmv`, `rbk_spmv`, `rbk_sym_spmv`, `rbk_sym_spmv_dot` (fused dot via warp+block reduction), `cpu_sym_spmv` debug fallback; plus `blas.h/.cpp` (dot, axpy, fill, norm2, segment reduce used by PCG).
- **Files**: ➕ `src/backends/luisa/luisa_device/linear/spmv.h/.cpp`, `blas.h/.cpp`.
- **Done**: SpMV results match CPU reference within 1e-5 for random symmetric BCOO; fused-dot kernel verified against separate dot.
- **Validate**: `test_spmv` vs CPU reference (1e-5) for all 4 SpMV variants; `rbk_sym_spmv_dot` fused dot equals separate dot within 1e-6; `blas` dot/axpy/fill/norm2 pass.
- **Milestone**: M3 — commit `[luisa-backend] T3.4 spmv-blas-kernels`; `git tag luisa-m3`.

---

### PHASE 4 — Base Geometry & Manager Systems (depends T1.4; parallelizable)

**T4.1 — global_geometry** (depends T1.2, T2.1)
- Port 6 files: `global_vertex_manager`, `global_body_manager`, `global_simplicial_surface_manager`, `vertex_reporter`, `body_reporter`, `simplicial_surface_reporter` (+ `details/*.inl`, `utils/offset_count_collection.*` used by them).
- These are mostly per-element kernels (attribute copy, displacement, AABB reduction) — good first real-system validation of the cookbook.
- **Done**: manager unit test under `apps/tests/backends/luisa/` exercising init/rebuild/dump/recover paths.
- **Validate**: `test_global_geometry` — init a small vertex/body/surface set, rebuild, dump to workspace, recover → attribute arrays byte-identical to pre-dump.
- **Milestone**: M4 — commit `[luisa-backend] T4.1 global-geometry`.

**T4.2 — implicit_geometry + animator + external_force + joint_dof + flags** (depends T1.4)
- Port: `implicit_geometry/half_plane*.{h,cu}` (+ `details/half_plane.inl`), `animator/animator`, `animator/global_animator` (+ `animator/utils.h`), `external_force/global_external_force_manager`, `external_force/external_force_reporter`, `joint_dof_system/global_joint_dof_manager`, `joint_dof_system/joint_dof_reporter`, `time_integrator/*` (4), `pipeline/*` (2 flags).
- **Done**: each builds; half-plane + animator covered by a small device test.
- **Validate**: `test_half_plane_animator` — animated keyframes displace vertices to expected positions (vs hand-computed); flags flip per frame.
- **Milestone**: M4 — commit `[luisa-backend] T4.2 implicit-animator-force-joint-flags`.

**T4.3 — kernel_cout + sanity_check + distance_system + dytopo base** (depends T1.4)
- Port `kernel_cout.*` (LC print), `sanity_check/*` (3 checkers), `distance_system/distance_diagnoser`, `dytopo_effect_system/*` (6 files: classify info, receivers/reporters, global manager).
- **Done**: builds; sanity checkers run in the sanity_check test app when wired in Phase 8.
- **Validate**: each ported file compiles; `kernel_cout` prints captured on sync in a device test; distance/dytopo kernels unit-run standalone.
- **Milestone**: M4 — commit `[luisa-backend] T4.3 kernel-cout-sanity-dytopo`; `git tag luisa-m4`.

---

### PHASE 5 — Finite Element System (largest subsystem, depends T2, T3, T4.1; parallelizable after T5.1)

**T5.1 — FEM base & utils** (depends T2.3, T3.1)
- Port: `finite_element/fem_utils.{h,cu}`, `matrix_utils.{h,cu}`, `shell_utils.{h,cu}`, `finite_element_method.{h,cu}`, `finite_element_state_accessor*` (2), reporters: `finite_element_body_reporter`, `finite_element_vertex_reporter`, `finite_element_surface_reporter`, `fem_linear_subsystem_reporter`, plus `details/finite_element_method.inl`.
- **Done**: FEM method builds; a single-element tet/shell energy/gradient/hessian test matches CUDA reference values.
- **Validate**: `test_fem_base` — one tet + one shell element: E, ∇E, ∇²E match CUDA-reference values (recorded constants) within 1e-5.
- **Milestone**: M5 — commit `[luisa-backend] T5.1 fem-base-utils`.

**T5.2 — FEM constitutions** (depends T5.1; internal batches parallelizable)
- Port constitution base (`fem_3d_constitution`, `codim_0d/1d/2d_constitution`, `finite_element_constitution`, `finite_element_extra_constitution`) and the 12 constitutions + their `sym/*.inl` + function headers:
  `arap_3d`, `stable_neo_hookean_3d` (+`detail/*.inl`), `discrete_shell_bending`, `strain_plastic_discrete_shell_bending`, `stress_plastic_discrete_shell_bending`, `neo_hookean_shell_2d`, `strain_limiting_baraff_witkin_shell_2d`, `kirchhoff_rod_bending`, `hookean_spring_1d`, `particle_0d`, `empty_0d`, `codim` variants. Each `sym/*.inl` is mechanically transformed per cookbook (Eigen→LVec/LMat, `std::`→`lps::`).
- Split into 3 subagent batches: (a) 3D volumetric (arap, snh + codim_0d/1d/2d), (b) shells (discrete bending + 2 plastics + neo-hookean shell + strain-limiting), (c) rods/springs/particles (kirchhoff, hookean, particle, empty).
- **Done**: per-constitution E/∇E/∇²E randomized finite-difference + vs-CUDA-reference tests pass.
- **Validate**: `test_constitution_*` per batch — analytic gradient/hessian vs central finite differences (h=1e-6, rel err <1e-4) AND vs recorded CUDA outputs; plastic shells also check history update.
- **Milestone**: M5 — commit `[luisa-backend] T5.2 fem-constitutions-a/b/c` (3 commits, one per batch).

**T5.3 — FEM dynamics, BDF, animator, elastics** (depends T5.1)
- Port: `finite_element_kinetic`, `fem_time_integrator`, `bdf/fem_bdf1_time_integrator`, `bdf/fem_bdf2_time_integrator`, `bdf/fem_bdf2_state`, `bdf/finite_element_bdf1_kinetic`, `bdf/finite_element_bdf2_kinetic` (+ `bdf/bdf2_function.h`, `sym/bdf2.inl`), `finite_element_animator` (+`details`), `finite_element_elastics`, `finite_element_external_force_manager`, `finite_element_external_force_reporter`.
- **Done**: builds; kinetic energy matches hand-computed reference on a small mesh.
- **Validate**: `test_fem_kinetic` — fixed 2-tet mesh, BDF1/BDF2 states: kinetic energy & inertia terms match hand-derived values within 1e-6.
- **Milestone**: M5 — commit `[luisa-backend] T5.3 fem-dynamics-bdf-animator`.

  **T5.4 — FEM linear subsystem + preconditioners** (depends T3, T5.1)
  - Port: `fem_linear_subsystem`, `diag_linear_subsystem` pattern owner, `fem_diag_preconditioner`, `fem_mas_preconditioner`, `mas_preconditioner_engine`, `finite_element_constraint`, `constraints/soft_position_constraint`, `constraints/finite_element_external_vertex_force_constraint`, `finite_element_external_vertex_force`.
  - **Done**: local Hessian assembly → converter → SpMV round-trip test passes vs reference.
  - **Validate**: `test_fem_linear` — assemble local Hessians for a 5-tet mesh → convert → SpMV → y matches direct dense A·x within 1e-5; MAS engine produces SPD preconditioner.
  - **Milestone**: M5 — commit `[luisa-backend] T5.4 fem-linear-subsystem-preconditioners`.
  - **EXECUTED AS T5.4a/T5.4b (measured, 2026-09):** T5.4a (`38052f30`) = `global_linear_system`'s *assembly half* + `diag_linear_subsystem` + `local_preconditioner` + `fem_linear_subsystem` + `fem_dytopo_effect_receiver` + the elastics/animator assembly fan-outs; validation met (5-tet → `MatrixConverter` → `SpMV` vs dense `A·x` **1.339e-15**, bound 1e-5). T5.4b = `fem_diag_preconditioner` + `finite_element_external_vertex_force` + `constraints/finite_element_external_vertex_force_constraint` + the preconditioner-apply half of `GlobalLinearSystem`. **Descriptor of the split, with reasons: (1) the reference's `GlobalLinearSystem::solve()` is what *triggers* the assembly (via `_update_subsystem_extent`/`_assemble_linear_system`), and `solve()` needs `IterativeSolver`/`LinearPCG`/`GlobalPreconditioner`/`OffDiagLinearSubsystem`/`CurrentFrameDiffDofReporter`, none of which exist before T8.3 — so T5.4a ports the assembly/layout half and leaves the solve path declared-but-undefined under `SCOPE NOTE (T8.3)` markers, and the assembly is reached from the test through header-level factored routines plus a standalone `GlobalLinearSystem::Impl` (see cookbook §7.19). (2) `mas_preconditioner_engine` + `fem_mas_preconditioner` are **deferred to M6 (new task)**: the engine is consumed by *both* FEM and ABD, so one validation in M6 covers both, it is ~2400 lines of warp-level/shared-memory device code (Eigen-in-shared-memory, float atomics, `cub::WarpReduce`) that is orthogonal to "FEM constitutions & dynamics", and its end-to-end purpose is exercised by T8.3 anyway. The `LocalPreconditioner` interface it plugs into is already ported, so the deferral is structural no-op.

  **T5.5 — FEM diff/extra constitutions + reporters** (depends T5.2) — **RE-SCOPED (measured, 2026-09): this task cannot complete inside M5.**
  - Port: `codim_*_constitution_diff_parm_reporter` (3), `fem_3d_constitution_diff_parm_reporter`, `finite_element_constitution_diff_parm_reporter`, `finite_element_extra_constitution`, `finite_element_extra_constitution_diff_parm_reporter`, `finite_element_diff_dof_reporter`, `finite_element_diff_parm_reporter`, `fem_active_set_reporter`, `fem_al_stiffness_estimator`, `fem_line_search_reporter`, `fem_line_search_subreporter`, `fem_dytopo_effect_receiver`.
  - **Measured dependency findings (they re-route the task):**
    * already delivered early (structural include cycles): `fem_line_search_reporter`/`fem_line_search_subreporter` + `finite_element_extra_constitution` in **T5.3** (`da86f09a`), `fem_dytopo_effect_receiver` in **T5.4a** (`38052f30`);
    * `fem_active_set_reporter` and `fem_al_stiffness_estimator` include `active_set_system/active_set_reporter.h` / `active_set_system/al_stiffness_estimator.h` — the AL active-set layer, which does not exist in this backend and belongs to the **M8** contact/AL work;
    * every `*_diff_*_reporter` (and the three `codim_*_constitution_diff_parm_reporter`) includes `diff_sim/diff_dof_reporter.h` / `diff_sim/diff_parm_reporter.h` — the **diff-sim/adjoint** layer, which belongs to **M9 (T9.4)** and drags in `<diff_sim/...>` (this is also why T5.3/T5.4a dropped the reference's unused diff-reporter includes).
  - **Therefore**: the remaining T5.5 files are re-assigned — the two AL reporters to M8 (with `active_set_system`), the diff reporters to M9 (with `diff_sim`). `git tag luisa-m5` is cut after **T5.4b**, not after T5.5: M5's stated deliverable (“all FEM constitutions & dynamics verified, E/G/H vs CUDA”) is exactly what T5.1–T5.4b deliver, and the re-assigned reporters are validation (shape/finiteness) shims over layers that do not exist yet.
  - **Milestone**: M5 — `git tag luisa-m5` after T5.4b; the re-assigned files are tracked in the M8/M9 task tables.

### PHASE 6 — Affine Body System (depends T2.2, T3, T5.1; parallelizable after T6.1)

**T6.1 — ABD base & utils** (depends T2.2, T5.1)
- Port: `affine_body/utils.{h,cu}`, `affine_body_dynamics.{h,cu}` (+`details`), `abd_energy`, `affine_body_kinetic`, `abd_time_integrator`, `affine_body_animator` (+`details`), `affine_body_state_accessor*` (2), reporters (`affine_body_body_reporter`, `affine_body_vertex_reporter`, `affine_body_surface_reporter`, `abd_linear_subsystem_reporter`, `affine_body_external_force_reporter`), `affine_body_external_force_manager`, `affine_body_constitution`, `affine_body_constraint`, base `inter_affine_body_constitution`, `inter_affine_body_constitution_manager` (+`details`), `inter_affine_body_animator`, `inter_affine_body_constraint`.
- **Done**: ABD dynamics builds; single-cube kinetic/potential matches reference.
- **Validate**: `test_abd_base` — one affine cube: kinetic + arap potential match recorded CUDA values within 1e-5; animator moves cube as scripted.
- **Milestone**: M6 — commit `[luisa-backend] T6.1 abd-base-utils`.

**T6.2 — ABD constitutions & joints** (depends T6.1; 2 parallel batches)
- Port: `constitutions/arap`, `constitutions/ortho_potential` (+ function headers + `sym/*.inl`), and joint constitutions `affine_body_fixed_joint`, `affine_body_prismatic_joint`, `affine_body_revolute_joint`, `affine_body_spherical_joint` (+ driving variants in `sym/`, `joint_limit_penalty.h`).
- Batches: (a) arap + ortho_potential, (b) 4 joint types (+limits). Each `sym/*.inl` mechanically transformed.
- **Done**: per-constitution E/G/H tests vs CUDA reference pass.
- **Validate**: `test_abd_constitution_*` per batch — E/G/H vs CUDA reference & finite-difference check; joint constitutions validated at multiple joint angles incl. limit-clamped cases.
- **Milestone**: M6 — commit `[luisa-backend] T6.2 abd-constitutions-joints-a/b` (2 commits).

**T6.3 — ABD constraints** (depends T6.1)
- Port: `constraints/external_articulation_constraint` (+ function header + `sym/*.inl`), `constraints/soft_transform_constraint`, `constraints/affine_body_external_body_force_constraint`, `affine_body_external_body_force`.
- **Done**: constraint gradient/hessian tests pass.
- **Validate**: `test_abd_constraints` — external articulation + soft transform constraints: G/H vs CUDA reference within 1e-5 across sampled configurations.
- **Milestone**: M6 — commit `[luisa-backend] T6.3 abd-constraints`.

**T6.4 — ABD BDF + solvers-side helpers** (depends T6.1, T3)
- Port: `bdf/abd_bdf1_time_integrator`, `bdf/abd_bdf2_time_integrator`, `bdf/abd_bdf2_state`, `bdf/affine_body_bdf1_kinetic`, `bdf/affine_body_bdf2_kinetic` (+ `bdf/sym/bdf2.inl`), `abd_linear_subsystem`, `abd_jacobi_matrix` (+`details`), `abd_diag_preconditioner`, `abd_tolerance_checker`, `abd_active_set_reporter`, `abd_al_stiffness_estimator`, `abd_line_search_reporter`, `abd_line_search_subreporter`, `abd_dytopo_effect_receiver`.
- **Done**: builds; jacobi-matrix conversion → SpMV test passes.
- **Validate**: `test_abd_linear` — jacobi matrix of a 2-cube chain converts and SpMV matches dense reference within 1e-5; BDF2 state advance matches BDF1 at steady state.
- **Milestone**: M6 — commit `[luisa-backend] T6.4 abd-bdf-linear-helpers`; `git tag luisa-m6`.

### PHASE 7 — Collision Detection (depends T2.3, T4.1; parallelizable after T7.1)

**T7.1 — BVH structures** (depends T3.2 algorithms, T4.1)
- Port: `collision_detection/atomic_counting_lbvh.{h,cu}` (+`details`), `linear_bvh.h` (+`details/linear_bvh.inl`), `stackless_bvh.h` (+`details`), `info_stackless_bvh.h` (+`details`), `info_stackless_bvh_v0.h` (+`details`), `aabb.h`.
- Heavy use of atomics, Morton codes (`cuda_device/bit_operation.h` → `luisa_device/builtins.h`), warp ops, and CUB-style compaction → validates T3.2 primitives at scale.
- **Done**: BVH build on random point/cloud matches CUDA-built BVH (same node order or equivalent traversal results); unit tests ported from `apps/tests/backends/cuda/lbvh.cu`, `stackless_bvh.cu`, `info_stackless_bvh.cu`.
- **Validate**: `test_bvh_*` — 100k random primitives: CUDA-exported BVH vs Luisa BVH produce identical traversal candidate sets; ported unit tests pass.
- **Milestone**: M7 — commit `[luisa-backend] T7.1 bvh-structures`.

**T7.2 — Trajectory filters** (depends T7.1)
- Port: `trajectory_filter`, `simplex_trajectory_filter` (+`details`), `vertex_half_plane_trajectory_filter`, `global_trajectory_filter`, and 7 filters: `lbvh_simplex_trajectory_filter`, `stackless_bvh_simplex_trajectory_filter`, `info_stackless_bvh_simplex_trajectory_filter`, `info_stackless_bvh_v0_simplex_trajectory_filter`, `easy_vertex_half_plane_trajectory_filter`, `al_vertex_half_plane_trajectory_filter`.
- **Done**: filter produces candidate pair sets equivalent to CUDA version on benchmark scenes (compare via exported pairs).
- **Validate**: `test_filters` — run CUDA filter & Luisa filter on 3 benchmark point clouds; exported candidate pairs identical (order-insensitive).
- **Milestone**: M7 — commit `[luisa-backend] T7.2 trajectory-filters`; `git tag luisa-m7`.

### PHASE 8 — Contact, Coupling, Solvers, Engine (depends Phase 5/6/7; mostly sequential at the end)

**T8.1 — Contact system core + models** (depends T7.2, T5.2, T6.2; 2 parallel batches)
- Core: `contact_system/global_contact_manager`, `contact_system_feature`, `contact_reporter`, `contact_exporter`, `contact_exporter_manager`, `adaptive_contact_parameter_reporter`, `adaptive_strategies/gipc_adaptive_parameter_strategy`, `contact_coeff.h`, `al_contact_function.h`, `al_vertex_half_plane_contact_function.h`.
- Models batch A: `simplex_normal_contact`, `simplex_frictional_contact`, `vertex_half_plane_normal_contact`, `vertex_half_plane_frictional_contact`, `contact_models/ipc_simplex_normal_contact`, `ipc_simplex_frictional_contact`, `ipc_vertex_half_plane_normal_contact`, `ipc_vertex_half_plane_frictional_contact` (+ function headers + `sym/*.inl` incl. `codim_ipc_*.inl`, `vertex_half_plane_distance.inl`).
- Models batch B (AL variants): `al_simplex_normal_contact`, `al_simplex_frictional_contact`, `al_vertex_half_plane_normal_contact`, `al_vertex_half_plane_frictional_contact`.
- Models batch C: `contact_models/analytical_barrier_pFpx/*` (pFpx PT/PP/PE/EE + parallel-EE/PE/PP + `details/*.inl`).
- **Done**: contact energy/gradient tests on synthetic PP/PE/PT/EE pairs match CUDA reference within tol.
- **Validate**: `test_contact_models` — synthetic pairs in contact & non-contact (incl. d→d_hat boundary, friction on/off): E/∇E vs CUDA within 1e-5; pFpx models check ∂²E blocks vs finite difference.
- **Milestone**: M8 — commit `[luisa-backend] T8.1 contact-core-models-a/b/c` (3 commits).

**T8.2 — Coupling system** (depends T5.4, T6.4)
- Port: `coupling_system/abd_fem_dytopo_effect_receiver`, `coupling_system/fem_abd_dytopo_effect_receiver`, `coupling_system/abd_fem_linear_subsystem`.
- **Done**: builds; coupling hessian assembly test passes.
- **Validate**: `test_coupling` — ABD-FEM coupled block matrix assembles and SpMV matches dense reference within 1e-5.
- **Milestone**: M8 — commit `[luisa-backend] T8.2 coupling-system`.

**T8.3 — Global linear system + solvers** (depends T3.4, T5.4, T6.4)
- Port: `linear_system/global_linear_system`, `iterative_solver`, `linear_pcg`, `linear_fused_pcg`, `local_preconditioner`, `global_preconditioner`, `off_diag_linear_subsystem`, `diag_linear_subsystem`, `current_frame_diff_dof_reporter`.
- **Done**: PCG solves a assembled FEM+ABD system to the same iteration count/residual as CUDA within tolerance on a small scene.
- **Validate**: `test_pcg` — same 2-cube+2-tet scene on both backends: iteration count equal, residual curve within 1e-4, solution x within 1e-5.
- **Milestone**: M8 — commit `[luisa-backend] T8.3 global-linear-system-solvers`.

**T8.4 — Engine assembly + advance pipeline** (depends all above)
- Port `engine/*` (8 files): `sim_engine.cu` → `sim_engine.cpp` + `sim_engine_do_init/advance/sync/retrieve/backward` + `advance_ipc` + `advance_al`; wire `SimEngine::build()` requiring all systems in the same order as CUDA; event collections (`event_init_scene` etc.); dump/recover paths; `Timer` sync func; `Debug::debug_sync_all` equivalent.
- **Done**: `luisa` engine advances frame 0 of `0_abd_gravity` without error and produces sane positions (full numerical check in Phase 9).
- **Validate**: run sim_case `0_abd_gravity` with luisa backend → completes frame 0, log shows all systems ran, output positions finite and visually plausible (attach plot/numbers to progress doc).
- **Milestone**: M8 — commit `[luisa-backend] T8.4 engine-assembly-advance`.

**T8.5 — Remaining systems** (depends T8.4; parallelizable small batch)
- Port: `diff_sim/*` (4), `line_search/*` (2), `newton_tolerance/*` (3), `active_set_system/*` (3), `engine` leftovers, `kernel_cout` final wiring, `sanity_check` registration.
- **Done**: whole backend links; full system list in `systems.json` matches CUDA backend's list.
- **Validate**: full link of `uipc_backend_luisa`; run engine init → `systems.json` diff (cuda vs luisa) is empty; `line_search`/`newton_tolerance`/`diff_sim` smoke-ran in a 2-frame sim.
- **Milestone**: M8 — commit `[luisa-backend] T8.5 remaining-systems-engine-complete`; `git tag luisa-m8`.

### PHASE 9 — Verification & Tests (depends T8.4)

**T9.1 — Backend unit tests** (parallel with phases)
- Port `apps/tests/backends/cuda/*` → `apps/tests/backends/luisa/*`: `hello_luisa` (from hello_muda), `gpu_sanity_check`, `lbvh`, `stackless_bvh`, `info_stackless_bvh`, `strain_plastic_discrete_shell_bending`, `stress_plastic_discrete_shell_bending`, `ee_distance_flag`.
- **Done**: all green.
- **Validate**: `ctest -R backend_luisa --output-on-failure` all tests pass; list attached to progress doc.
- **Milestone**: M9 — commit `[luisa-backend] T9.1 backend-unit-tests`.

**T9.2 — Sim-case regression** (depends T8.4)
- Ensure `apps/tests/sim_case` can target the `luisa` backend (engine selection by name); run representative subset: `0_abd_gravity`, `1_abd_contact_pt`, `4_abd_contact_pp`, `6_abd_cube_pile`, `13_fem_3d_gravity`, `14_fem_3d_ground_contact`, `17_fem_multi_constituion`, `18_abd_fem_contact`, `37_abd_revolute_joint`, `63_abd_codim_shell_gravity`, `76_abd_bdf2_rebound_higher_than_bdf1`, plus `sanity_check` app cases.
- Record per-case final positions/energies vs CUDA run artifacts in `docs/luisa_regression_baseline.md`.
- **Done**: subset completes; no NaNs; positions within tolerance vs CUDA baseline (translation/rotation epsilon defined per case).
- **Validate**: run the listed sim cases on both backends; compare final positions: mean per-vertex error < 1e-3 (rigid) / < 5e-3 (soft) or case-specific eps recorded in `docs/luisa_regression_baseline.md`; no NaN/Inf.
- **Milestone**: M9 — commit `[luisa-backend] T9.2 sim-case-regression`.

**T9.3 — CUDA↔Luisa equivalence harness** (depends T9.2)
- Add a test utility that runs the same scene JSON on both `cuda` and `luisa` backends (when both built) and compares retrieved positions/velocities/energies with tolerances; wire into regression CI script.
- **Done**: harness runs; divergence report documented.
- **Validate**: equivalence harness executes same scene on cuda+luisa and emits CSV of per-quantity deltas; report committed under `docs/`.
- **Milestone**: M9 — commit `[luisa-backend] T9.3 equivalence-harness`.

**T9.4 — dump/recover + diff-sim tests**
- Validate `do_dump`/`do_try_recover`/`do_apply_recover` round trip and diff reporters on `29_abd_fem_recover`, `84/87 *_recover` cases.
- **Done**: recover resumes to same state.
- **Validate**: run `29_abd_fem_recover` and `87_*_recover`: dump at frame N, recover in new process → positions match continuous run within 1e-6; diff-reporter outputs present.
- **Milestone**: M9 — commit `[luisa-backend] T9.4 dump-recover-diffsim`; `git tag luisa-m9`.

### PHASE 10 — Integration, CI & Docs (depends T9.2)

**T10.1 — Build polish & options**
- Verify `UIPC_WITH_LUISA_BACKEND` OFF-by-default builds are untouched; ON builds both CUDA and Luisa; presets updated (`CMakePresets.json`); xmake config parity; install rules include the new backend module; rpath/module naming (`uipc_backend_luisa`).
- **Done**: clean configure+build both generators on Windows.
- **Validate**: from clean tree: CMake option ON & OFF builds + `xmake` build all succeed; `uipc_backend_luisa` module installed; no cuda toolchain invoked (check build log lacks nvcc).
- **Milestone**: M10 — commit `[luisa-backend] T10.1 build-polish`.

**T10.2 — CI & docs**
- Add CI job matrix entry (optional/backend-luisa) running T9.1 + T9.2 subset; update `README.md`/docs backend list; porting cookbook finalized.
- **Done**: CI green; docs merged.
- **Validate**: CI job with `UIPC_WITH_LUISA_BACKEND=ON` runs T9.1+T9.2 subset green; README/docs list the backend; cookbook finalized.
- **Milestone**: M10 — commit `[luisa-backend] T10.2 ci-docs`; `git tag luisa-m10`.

---

## 6. Subagent Scheduling Waves

- **Wave 0 (sequential):** T0.1 → T0.2 → T1.0 → T1.1.
- **Wave 1 (sequential):** T1.2 → T1.3, T1.4 (T1.4 ∥ T1.2).
- **Wave 2 (sequential):** T2.1 → T2.2 → T2.3 (each with tests).
- **Wave 3 (sequential):** T3.1 → T3.2 → (T3.3 ∥ T3.4).
- **Wave 4 (∥ up to 3 agents):** T4.1, T4.2, T4.3.
- **Wave 5 (∥ after T5.1):** T5.1 → {T5.2a/b/c ∥, T5.3, T5.4} → T5.5.
- **Wave 6 (∥ after T6.1):** T6.1 → {T6.2a/b ∥, T6.3, T6.4}.
- **Wave 7 (∥ after T7.1):** T7.1 → T7.2.
- **Wave 8:** T8.1(a/b/c ∥) → T8.2 → T8.3 → T8.4 → T8.5.
- **Wave 9 (∥):** T9.1 (throughout), T9.2 → T9.3 → T9.4.
- **Wave 10:** T10.1 → T10.2.

Rules for subagents:
1. Never modify `src/backends/cuda/` or `src/backends/common/` (read-only references). All new code under `src/backends/luisa/` (+ tests under `apps/tests/`, docs under `docs/`).
2. Every ported file must follow the cookbook; every deviation/new pattern gets appended to the cookbook in the same PR/task.
3. Keep the public system interfaces (`SimSystem` subclasses, info structs, `muda::BufferView`-shaped signatures → `LBufferView`) identical so the engine and cross-system code compile unchanged.
4. One subsystem batch per subagent; run that batch's unit tests before reporting done.
5. Report format per task: files created/modified, tests run + results, deviations, cookbook additions.
6. **Before reporting done**: run the task's **Validate** line, paste evidence into `docs/luisa_backend_progress.md`, and commit everything (code+tests+doc) as `[luisa-backend] T<x>.<y> ...` on `feature/luisa-backend`. Do not leave uncommitted work — the next session resumes from git state only.

---

## 7. Definition of Done (whole project)

1. `src/backends/luisa/` builds via CMake and xmake behind `UIPC_WITH_LUISA_BACKEND=ON`; nvcc never invoked.
2. All 205 CUDA-kernel-bearing files have Luisa equivalents wired into the engine (`systems.json` parity).
3. All runtime dispatches/copies go through LC `CommandList` + `Stream`; kernel printf works.
4. T9.1 unit tests green; T9.2 sim-case subset within tolerance vs CUDA; dump/recover and diff-sim validated.
5. Cookbook + inventory + regression baseline docs committed.
6. `docs/luisa_backend_progress.md` shows every task ✅ with commit sha and validation evidence; milestones tagged `luisa-m0..m10`.


---

## Appendix A — Risk Register

Reviewed at every milestone tag. L = likelihood, I = impact (H/M/L).

| ID | Risk | L | I | Mitigation | Detection point | Contingency |
|---|---|---|---|---|---|---|
| R1 | LC dependency can't be integrated cleanly into libuipc CMake/xmake (version conflict, exported targets missing, vcpkg/FLAC-style pinning pain) | M | H | T0.1 spike decides strategy before any porting; pin LC commit; keep option OFF by default | M0 exit | Vendor LC as git submodule at known-good commit; worst case patch its CMake |
| R2 | Hidden Eigen linear-algebra surface (inverse/SVD/EVD/block ops) inside kernels exceeds the Phase-0 inventory | H | H | T0.2 inventory is exhaustive (grep every Eigen member in `__device__`/lambda scope); math shim grows test-first | T2.x validation gates | Write the missing op on the shim immediately; escalate to plan owner if a full dense solver is needed |
| R3 | LC DSL control-flow misuse: host `if/for` silently flattened into AST, or runtime loop duplicated at build time | M | H | Cookbook rule + review checklist; require `dynamic_range`/`$for`/`$if` for any runtime-bound flow; AST-print debug helper | Any ported kernel with loops/branching | Code-review rework; add `device_log` tracing test per suspicious kernel |
| R4 | Large fixed-size matrices (9x9/12x12) blow up AST size → very long JIT per kernel, or backend arg limits hit | M | M | Compile-once (D3); measure JIT time in T2.1 smoke; keep matrices as local arrays in kernels | T2.1/T5.2 timings | Split hessian kernels per-element; AOT/cached shader blobs; hoist matrices to buffers |
| R5 | `LBuffer` resize invalidates outstanding views → use-after-free (LC buffers are fixed-size) | M | H | D2 shared-ownership views; frame-scoped view discipline rule; ASAN-style debug build with view tracking | T4.1 manager tests | Strengthen view debugging (owner generation counter); audit offending system |
| R6 | Numerical divergence vs CUDA (atomic ordering, float contraction/FMA, fast-math, reduction order) breaks regression parity | H | H | Equivalence harness (T9.3) with explicit per-case eps; disable fast-math equivalents; deterministic reduction order where feasible | T9.2 baseline compare | Document eps per case; if systematic, add LC-side rounding mode/flags and re-baseline |
| R7 | CUB replacements (sort/scan/select) much slower → unusable frame times | M | M | Correctness first (D7); algorithmic choices reviewed in T3.2 | T9.2 wall-clock vs CUDA | Optimize hot primitives (better bucket sizes, warp paths) via lc_optimize skill; last resort CPU conversion for cold paths |
| R8 | `make_spd`/SVD/EVD parity failures (Jacobi iteration convergence, sign conventions) | M | H | T2.2 tests vs Eigen on 200+ random matrices incl. near-degenerate; clamping strategies identical to CUDA | T2.2 | Port exact CUDA algorithm step-for-step; add fallback reference path behind flag |
| R9 | LC CUDA backend feature gaps (warp intrinsics subset, device print buffering, byte buffers, atomic float3) | M | H | T0.1 spike exercises each; inventory maps every intrinsic before port waves | M0 / first use in T2-T3 | Emulate (e.g. 3 scalar atomics); contribute upstream; isolate behind `builtins.h` shim so fix is one place |
| R10 | muda async semantics mismatch (stream-ordered copies, `wait_device` granularity) → race/heisenbug | M | H | Single engine Stream; explicit `synchronize()` at the same points CUDA backend waits; documented in cookbook | T4.x manager tests under stress | Add stream event sync per frame; audit copy call-sites |
| R11 | Subagent inconsistency: parallel port batches drift from the cookbook → integration breakage | M | M | Cookbook is sole authority; deviation must be appended; milestone commit per batch; CI builds each batch | Every milestone | Rework to cookbook; assign integration-fix subagent |
| R12 | Test tolerance flakiness (randomized scenes, GPU nondeterminism) makes gates unreliable | M | M | Fixed seeds in tests; recorded reference artifacts; eps bands not single values | T9.1/T9.2 first runs | Re-seed/re-baseline; separate flaky tests from gating set |
| R13 | LC version drift (API changes in `Buffer`/`Stream`/DSL sugar) breaks the backend later | L | M | Pin LC commit in T0.1; upgrade only at milestone boundaries with full T9.1 run | Upgrade PRs | Stay pinned; budgeted upgrade task |
| R14 | Hard-to-debug device failures (DSL AST errors, no nvcc printf-style stepping) | H | M | `device_log` from day 1 (T1.2); small repro tests per kernel; LC validation layers on | Any failing port task | Minimize repro; binary-search kernel args; ask LC debug skill |
| R15 | Multi-session state loss: work resumes without context, tasks redone or skipped | H | H | This protocol: progress doc + per-task commits + milestone tags + resume procedure (§5.0) | Every session start | Session reads §5.0 resume protocol first; reconcile git log vs progress doc before coding |
| R16 | Kernel parameter count / buffer-view explosion exceeds LC backend limits for fat kernels (many views captured) | L | M | Pass base pointers + offsets via a small struct buffer when view count > threshold; cookbook pattern | T5.2/T8.1 compile | Pack args into device struct (SoA) kernels |
| R17 | Diff-sim/adjoint paths silently wrong (they're tested last) | M | H | T5.5/T9.4 explicit shape+value checks before engine milestone; equivalence harness covers gradient reporters | M8/M9 | Isolate diff reporters; compare against finite differences of the forward sim |

---

## Appendix B — Progress Doc Template (`docs/luisa_backend_progress.md`)

```markdown
# Luisa Backend Progress

## Global status (update every commit)
- Current milestone: M<n> <name>
- Last completed task: T<x>.<y> (<title>) — commit <sha>
- Next action: T<a>.<b> (<title>)
- Branch: feature/luisa-backend | Last tag: luisa-m<n-1>
- Open risks touched this milestone: R..

## Task ledger
| Task | Title | Milestone | Status | Commit | Validation evidence | Notes |
|---|---|---|---|---|---|---|
| T0.1 | LC build & device spike | M0 | ✅ | abc1234 | spike exe exit 0, buffer=42 | strategy: add_subdirectory |

## Milestone log
- luisa-m0 (date): exit criteria met …
```
