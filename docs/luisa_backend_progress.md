# libuipc LuisaCompute Backend — Progress

> **Single source of truth for the `feature/luisa-backend` work stream.**
> Read this first when resuming. Protocol: `libuipc_lc_plan.md` §5.0.

## Global status

| Field | Value |
|---|---|
| Branch | `feature/luisa-backend` |
| Base commit | `9c748a75` (main @ merge PR #497) |
| Current milestone | **M2 — Math layer** (in progress: T2.1 complete) |
| Last completed task | T2.1 |
| Last commit | _see `git log --oneline --grep=luisa-backend`_ |
| Milestone tags | `luisa-m0`, `luisa-m1` |
| Next action | T2.2 — `matrix_ops.h` (det/inverse), `make_spd`, 3x3 `svd.h`, dihedral/thickness/friction |
| LC dependency | `find_package(LuisaCompute CONFIG)` @ `LUISA_COMPUTE_PREFIX` = `C:/dev/LuisaCompute/install-msvc`, LC 0.10.0, backend `cuda`. **The prefix is the MSVC build** — see cookbook §1.1 |

### Environment facts (do not re-derive)

- Work dir: `C:\dev\libuipc`; LC source `C:\dev\LuisaCompute` (symlink `LuisaCompute/`), LC install `C:\dev\LuisaCompute\install`.
- LC built with clang++ (LLVM 19, MSVC ABI) at `D:/TestUE/clang/bin/clang++.exe`; CMake `D:\cmake\bin\cmake`, Ninja, NVCC 12.9, GPU RTX 4060.
- Build LC: `cmake --build C:/dev/LuisaCompute/build -j12 && cmake --install C:/dev/LuisaCompute/build --prefix C:/dev/LuisaCompute/install`.
- **LC must be built with `-DLUISA_COMPUTE_USE_SYSTEM_SPDLOG=ON`** against libuipc's vcpkg spdlog/fmt (see cookbook §1.3), otherwise `fmt::v10` is defined twice and every backend TU fails to compile.
- libuipc in-tree build: `CMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake`, `-DUIPC_USING_LOCAL_VCPKG=ON -DUIPC_WITH_LUISA_BACKEND=ON`; vcpkg deps installed in `build/vcpkg_installed`.
- The reference backend uses **`cuda_tool`**, not `muda` (see cookbook §0).
- Windows shell notes: invoke `cmd /c` from **Python** (`subprocess`), not from MSYS bash (argument mangling); `set VSLANG=1033` for English diagnostics.

## Task table

Status: ⬜ todo · 🟡 in-progress · ✅ done

| Task | Title | Milestone | Status | Commit | Validation evidence | Notes |
|---|---|---|---|---|---|---|
| T0.1 | LC build & device spike | M0 | ✅ | _see tag luisa-m0_ | `scratch/lc_spike/build/lc_spike.exe cuda` → RC 0; `[ ok ] Kernel1D write`; `[ ok ] warp_active_sum + atomic_fetch_add` (sum exact); 32 `[DEVICE] warp[N] partial=` lines captured | Spike at `scratch/lc_spike/`; strategy in cookbook §1 |
| T0.2 | muda/Eigen/CUB API inventory | M0 | ✅ | _see tag luisa-m0_ | `python scripts/luisa_inventory.py` → 8071 occurrences, 154 distinct, **0 unmapped**, exit 0 | `docs/luisa_api_inventory.md` |
| T1.0 | Porting cookbook + backend skeleton | M1 | ✅ | d93050a4 | cookbook is the porting contract (`docs/luisa_porting_cookbook.md`); `src/backends/luisa/` skeleton mirrors the CUDA shell | |
| T1.1 | Build integration (CMake+xmake) | M1 | ✅ | 718b9400 | ON: `cmake -DUIPC_WITH_LUISA_BACKEND=ON` → "Adding backend: [luisa]", `uipc_backend_luisa.dll` builds+links (MSVC). OFF: `-DUIPC_WITH_LUISA_BACKEND=OFF` → only `[none]`, `none` builds. No nvcc invoked. | LC resolved top-level via `cmake/uipc_luisa.cmake` (`LUISA_COMPUTE_PREFIX`) |
| T1.2 | luisa_device core runtime | M1 | ✅ | 92322e82 | `scratch/lc_spike/lc_spike2.exe cuda` → RC 0, 4/4 design checks (subview offsets, double atomics, Shared block reduce, runtime dispatch); `luisa_device_tests.exe cuda` exercises the layer | `src/backends/luisa/luisa_device/*`; R9 double-atomic gap closed in `atomic.h` |
| T1.3 | Foundation unit tests | M1 | ✅ | 92322e82 + M1 commit | `luisa_device_tests.exe cuda` → **7 cases, 0 checks failed**, exit 0 (runtime_init, buffer_ops, device_var_scalar, parallel_for, stream_commands, atomic_and_reductions, kernel_print) | `apps/tests/backends/luisa/`; runs standalone now, registered in-tree via `uipc_add_test` in T1.1 |
| T1.4 | Backend shell + SimSystem plumbing | M1 | ✅ | 0b83a2e4 + M1 commit | `ctest -R backend_luisa --output-on-failure` → **1/1 passed** (label `gpu`); binary: **all tests passed, 28 assertions in 10 test cases**, exit 0. Covers module load via the public Engine API, shell init/advance/sync/retrieve, and find()/require() resolution with a real dependency edge. | |
| T2.1 | LVec/LMat + scalar math | M2 | ✅ | _this commit_ | in-tree: `ctest -R backend_luisa --output-on-failure` → **1/1 passed** (label `gpu`); binary → **all tests passed, 114 assertions in 15 test cases**, exit 0 (new: `math_host_layout`, `math_host_views`, `math_host_ops`, `math_host_eigen_parity`, `math_device_smoke`). standalone (`scripts/luisa_standalone.py cuda`) → **12 cases, 0 checks failed**, exit 0. `math_host_eigen_parity` compares LVec/LMat against Eigen within 1e-6 for the Vector9/Matrix9x9 use cases; `math_device_smoke` runs **one** generic routine as `S=double` (host reference) and `S=lcrt::Var<double>` (GPU) and requires bit-identical results. | `src/backends/luisa/luisa_device/math/{math.h,vec.h,mat.h}`; helpers in `scripts/luisa_{build,check_compile,standalone}.py` |

### Execution note

T1.2/T1.3 were implemented **before** T1.0's skeleton because the `luisa_device`
layer is self-contained (LC + std only) and could therefore be built and validated
on the GPU immediately, without the libuipc vcpkg toolchain. The cookbook (T1.0)
was authored first and is already authoritative. Task ids keep the plan's meaning.

### Not started (planned)

T2.2–T2.3 (math), T3.1–T3.4 (linear algebra), T4.1–T4.3 (base systems),
T5.1–T5.5 (FEM), T6.1–T6.4 (ABD), T7.1–T7.2 (collision), T8.1–T8.5 (contact/engine),
T9.1–T9.4 (verification), T10.1–T10.2 (integration/CI).

## Milestone log

| Milestone | Tasks | Exit criteria | Status | Tag |
|---|---|---|---|---|
| M0 Spike & Inventory | T0.1, T0.2 | LC spike runs on GPU; dependency strategy decided; API inventory 100% mapped | ✅ | `luisa-m0` |
| M1 Foundation | T1.0–T1.4 | Backend skeleton builds (CMake), loads; device core tests green; shell works | ✅ | `luisa-m1` |
| M2 Math layer | T2.1–T2.3 | LVec/LMat + ops/SVD/distance host+device parity | 🟡 T2.1 ✅ | `luisa-m2` |
| M3 Linear algebra | T3.1–T3.4 | Containers, converter, SpMV, BLAS verified vs references | ⬜ | `luisa-m3` |
| M4 Base systems | T4.1–T4.3 | Geometry/manager/animator systems build + unit-tested | ⬜ | `luisa-m4` |
| M5 FEM | T5.1–T5.5 | All FEM constitutions & dynamics verified | ⬜ | `luisa-m5` |
| M6 ABD | T6.1–T6.4 | All ABD constitutions/joints/constraints verified | ⬜ | `luisa-m6` |
| M7 Collision | T7.1–T7.2 | BVH + trajectory filters equivalent | ⬜ | `luisa-m7` |
| M8 Contact→Engine | T8.1–T8.5 | Contact models, coupling, PCG, engine link; frame 0 advances | ⬜ | `luisa-m8` |
| M9 Verification | T9.1–T9.4 | Unit + sim-case regression + equivalence green | ⬜ | `luisa-m9` |
| M10 Integration | T10.1–T10.2 | Clean builds both generators; CI green; docs merged | ⬜ | `luisa-m10` |

## Risk register review

Reviewed at M0: R1 (LC dependency) → **mitigated**: `find_package` + pinned install prefix
verified by T0.1. R2 (hidden Eigen surface) → inventory T0.2 found 154 symbols, 0 unmapped;
Eigen surface is confined to fixed-size types/ops. R9 (LC CUDA backend gaps) → T0.1
exercised Kernel1D, CommandList, warp reduction, atomics, `device_log` — all OK.
Remaining top risks: R3 (DSL control flow), R5 (view lifetime), R6 (numerical parity).
