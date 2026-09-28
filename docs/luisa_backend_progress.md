# libuipc LuisaCompute Backend — Progress

> **Single source of truth for the `feature/luisa-backend` work stream.**
> Read this first when resuming. Protocol: `libuipc_lc_plan.md` §5.0.

## Global status

| Field | Value |
|---|---|
| Branch | `feature/luisa-backend` |
| Base commit | `9c748a75` (main @ merge PR #497) |
| Current milestone | **M0 — Spike & Inventory** (complete) → M1 in progress |
| Last completed task | T0.2 |
| Last commit | _see `git log --oneline --grep=luisa-backend`_ |
| Milestone tags | `luisa-m0` |
| Next action | T1.0 — backend skeleton + cookbook already authored; create `src/backends/luisa/**` |
| LC dependency | `find_package(LuisaCompute CONFIG)` @ `C:/dev/LuisaCompute/install`, LC 0.10.0, backend `cuda` |

### Environment facts (do not re-derive)

- Work dir: `C:\dev\libuipc`; LC source `C:\dev\LuisaCompute` (symlink `LuisaCompute/`), LC install `C:\dev\LuisaCompute\install`.
- LC built with clang++ (LLVM 19, MSVC ABI) at `D:/TestUE/clang/bin/clang++.exe`; CMake `D:\cmake\bin\cmake`, Ninja, NVCC 12.9, GPU RTX 4060.
- Build LC: `cmake --build C:/dev/LuisaCompute/build -j12 && cmake --install C:/dev/LuisaCompute/build --prefix C:/dev/LuisaCompute/install`.
- The reference backend uses **`cuda_tool`**, not `muda` (see cookbook §0).
- Windows shell notes: invoke `cmd /c` from **Python** (`subprocess`), not from MSYS bash (argument mangling); `set VSLANG=1033` for English diagnostics.

## Task table

Status: ⬜ todo · 🟡 in-progress · ✅ done

| Task | Title | Milestone | Status | Commit | Validation evidence | Notes |
|---|---|---|---|---|---|---|
| T0.1 | LC build & device spike | M0 | ✅ | _see tag luisa-m0_ | `scratch/lc_spike/build/lc_spike.exe cuda` → RC 0; `[ ok ] Kernel1D write`; `[ ok ] warp_active_sum + atomic_fetch_add` (sum exact); 32 `[DEVICE] warp[N] partial=` lines captured | Spike at `scratch/lc_spike/`; strategy in cookbook §1 |
| T0.2 | muda/Eigen/CUB API inventory | M0 | ✅ | _see tag luisa-m0_ | `python scripts/luisa_inventory.py` → 8071 occurrences, 154 distinct, **0 unmapped**, exit 0 | `docs/luisa_api_inventory.md` |
| T1.0 | Porting cookbook + backend skeleton | M1 | 🟡 | | cookbook authored (`docs/luisa_porting_cookbook.md`); skeleton pending build | |
| T1.1 | Build integration (CMake+xmake) | M1 | ⬜ | | | |
| T1.2 | luisa_device core runtime | M1 | ⬜ | | | |
| T1.3 | Foundation unit tests | M1 | ⬜ | | | |
| T1.4 | Backend shell + SimSystem plumbing | M1 | ⬜ | | | |

### Not started (planned)

T2.1–T2.3 (math), T3.1–T3.4 (linear algebra), T4.1–T4.3 (base systems),
T5.1–T5.5 (FEM), T6.1–T6.4 (ABD), T7.1–T7.2 (collision), T8.1–T8.5 (contact/engine),
T9.1–T9.4 (verification), T10.1–T10.2 (integration/CI).

## Milestone log

| Milestone | Tasks | Exit criteria | Status | Tag |
|---|---|---|---|---|
| M0 Spike & Inventory | T0.1, T0.2 | LC spike runs on GPU; dependency strategy decided; API inventory 100% mapped | ✅ | `luisa-m0` |
| M1 Foundation | T1.0–T1.4 | Backend skeleton builds (CMake+xmake), loads; device core tests green; shell works | 🟡 | `luisa-m1` |
| M2 Math layer | T2.1–T2.3 | LVec/LMat + ops/SVD/distance host+device parity | ⬜ | `luisa-m2` |
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
