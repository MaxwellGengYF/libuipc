# libuipc — LuisaCompute porting cookbook

> **Authority document for the `luisa` backend.** Every ported file must follow the
> patterns below. Any *new* pattern discovered during a port task must be appended to
> this file in the same commit.
>
> Companion documents:
> - `docs/luisa_api_inventory.md` — every device-API symbol and its LC replacement (T0.2).
> - `docs/luisa_backend_progress.md` — task status, commits, validation evidence.
> - `libuipc_lc_plan.md` — master plan (phases, milestones, risks).

---

## 0. TL;DR of the paradigm shift

The reference backend no longer uses `muda`; it uses an in-tree, hand-rolled
**`cuda_tool`** layer (`uipc::backend::cuda_tool`, `src/backends/cuda/cuda_tool/`).
The LuisaCompute backend mirrors that layer as **`luisa_device`**
(`uipc::backend::luisa::luisa_device`). Porting is therefore:

```
cuda_tool::{BufferView,DeviceBuffer,DeviceVar,...}   ->   luisa_device::{LBufferView,LDeviceBuffer,LDeviceVar,...}
__global__ void foo_kernel(...) { ... }              ->   Kernel1D built with device.compile<1>(...)
kernel<<<grid, block, smem, stream>>>(args...)       ->   stream << shader(args...).dispatch(n)  (via LStream)
atomicAdd(&v(i), x)                                  ->   v.atomic(i).fetch_add(x)
Eigen::Matrix<T,9,9> / Eigen::Vector<T,9>            ->   lc::math::LMat<T,9,9> / LVec<T,9>
CUB / cub::*                                         ->   luisa_device::linalg::* (Phase 3)
__shfl_down_sync / __ballot_sync / __popc ...        ->   luisa_device::builtins::* (warp ops)
```

Everything is **plain C++20 `.cpp`** compiled without nvcc.

---

## 1. LC dependency strategy (T0.1 decision)

**Decision: `find_package(LuisaCompute CONFIG REQUIRED)` against a pinned, locally
built install prefix; enabled by `UIPC_WITH_LUISA_BACKEND` (default OFF).**

| Aspect | Decision |
|---|---|
| Integration mechanism | `find_package(LuisaCompute CONFIG)` — *not* `add_subdirectory`, *not* `FetchContent` |
| Package prefix | `LUISA_COMPUTE_PREFIX` (default `<LUISA_COMPUTE_ROOT>/install`). **Use `C:/dev/LuisaCompute/install-msvc`** — LC must be built with MSVC, see §1.1 |
| Consumed targets | `luisa::compute` (umbrella INTERFACE target; pulls core/runtime/dsl/ast/xir/ext/backends) |
| LC version | 0.10.0 (`include/luisa/version.h`) |
| LC backends built | `cuda` only (`LUISA_COMPUTE_ENABLE_VULKAN=OFF`, `..._DX=OFF`, `..._GUI=OFF`) |
| Device at runtime | `"cuda"` (LC-CUDA / NVRTC → PTX), selected from the engine config JSON (default `"cuda"`) |
| Rebuild rule | LC install must post-date its sources. Rebuild with `cmake --build C:/dev/LuisaCompute/build && cmake --install C:/dev/LuisaCompute/build --prefix C:/dev/LuisaCompute/install` |
| Compiler | LC is built with **MSVC (cl.exe)** — see §1.1. Consumers must use a conformant preprocessor. |

**Why not `add_subdirectory`:** LC is large (glslang, SPIRV-Tools, …); building it inside
every libuipc configure would multiply CI cost. An installed, importable package keeps the
libuipc build fast and the dependency *pinnable* (R1, R13).

**Contingency (R1):** if the install prefix is unavailable, fall back to
`add_subdirectory(${LUISA_COMPUTE_ROOT})`; a `FetchContent` pin is the last resort.
The backend CMake isolates this choice behind `uipc_luisa_resolve_dependency()`.

### 1.1 Toolchain: LC must be built with the *same* ABI as libuipc

Two separate, hard requirements:

1. **Conformant preprocessor.** `include/luisa/dsl/polymorphic.h` uses `__VA_OPT__`.
   MSVC only supports it with `/Zc:preprocessor` (clang-cl/clang++ are fine).
   The `luisa` target and its tests pass both `/utf-8` (see below) and
   `/Zc:preprocessor`.

2. **Same compiler family for LC and its consumers (ABI).** A clang-built LC is
   **not link-compatible with an MSVC consumer**: clang mangles deduced-`auto`
   return types with an `?A?<auto>@@` placeholder
   (e.g. `?device@Resource@compute@luisa@@QEBA?A?<auto>@@XZ`) while MSVC emits the
   plain form (`?device@Resource@compute@luisa@@QEBA@XZ`). libuipc's headers hit
   several such members (`Resource::device/handle/native_handle`,
   `Device::backend_name`, `ShaderInvokeBase::_parallelize`, ...), so an MSVC
   consumer fails with `LNK2019` on symbols that *are* exported.

   Conversely libuipc's own core does **not** compile with clang-cl (clang rejects
   the non-const `AutoAttributeCollection` mutations in
   `src/core/geometry/simplicial_complex_attributes.cpp`).

   **Therefore: build LuisaCompute with MSVC (cl.exe), matching libuipc.**

   ```
   cmake -S LuisaCompute -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_TOOLCHAIN_FILE=<vcpkg.cmake> -DVCPKG_INSTALLED_DIR=<vcpkg_installed> \
         -DLUISA_COMPUTE_USE_SYSTEM_SPDLOG=ON \
         -DLUISA_COMPUTE_ENABLE_VULKAN=OFF -DLUISA_COMPUTE_ENABLE_DX=OFF \
         -DLUISA_COMPUTE_ENABLE_GUI=OFF -DLUISA_COMPUTE_BUILD_TESTS=OFF \
         -Dspdlog_DIR=<vcpkg_installed>/x64-windows/share/spdlog \
         -Dfmt_DIR=<vcpkg_installed>/x64-windows/share/fmt
   ```

3. **`/utf-8`.** Several luisa_device/backend headers carry non-ASCII characters
   in comments (em dashes, arrows). Under a non-UTF-8 system code page (e.g. 936)
   MSVC mis-decodes them and tokenization breaks in unrelated headers. Always
   compile the `luisa` target and its tests with `/utf-8`.

`LUISA_COMPUTE_ENABLE_VULKAN`/`DX` are off because their SPIR-V/HLSL compile steps
invoke helper executables that fail to start under a vcpkg-modified environment;
the backend only needs LC's CUDA backend (plan D6).

### 1.2 LC source fix applied locally

The checkout's prebuilt binaries were stale: `luisa::compute::detail::debug_function_if_enabled`
(defined in `src/ast/function_builder_debugger.cpp`, declared `LUISA_AST_API` in the header)
was **not exported** from `luisa-ast.dll`, so any consumer calling `define_kernel`
(`device.compile<1>`) failed to link. Rebuilding LC from source (commit `112bf1daf`,
"Fix build") exports the symbol and the spike links. **Always rebuild LC after updating it.**

### 1.3 fmt/spdlog must be unified (one `fmt::v10` per process)

LC bundles spdlog+fmt and `include/luisa/core/logging.h` includes `<spdlog/spdlog.h>`,
which resolves to LC's bundled copy. libuipc gets fmt/spdlog from vcpkg. Both are
fmt **v10**, so a translation unit that includes both (any ported backend file that
uses libuipc's logging *and* LuisaCompute) sees two conflicting `fmt::v10`
definitions and fails with hundreds of C7568/C2955 errors inside
`luisa/ext/spdlog/fmt/bundled/{chrono,xchar,ostream}.h`.

**Fix:** build LC with the *same* spdlog/fmt as libuipc:

```
cmake -S LuisaCompute -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg.cmake> \
      -DLUISA_COMPUTE_USE_SYSTEM_SPDLOG=ON \
      -Dspdlog_DIR=<vcpkg_installed>/x64-windows/share/spdlog \
      -Dfmt_DIR=<vcpkg_installed>/x64-windows/share/fmt
cmake --build build -j && cmake --install build --prefix <root>/install
```

This makes `<spdlog/spdlog.h>` resolve to vcpkg's spdlog (header-only target
`spdlog::spdlog_header_only`) so there is exactly one fmt in the process. Do **not**
  work around this by removing LC's `include/luisa/ext` from the include path — the LC
  logging header needs it.

### 1.4 **Applied local LC fix #2**: double-precision transcendentals were computed in float

LC's CUDA backend has **two** codegen paths and picks between them by environment variable
(`src/backends/cuda/cuda_device.cpp`):

| env | path |
|---|---|
| `LUISA_EXPERIMENTAL_LLVM_CODEGEN=1` | `llvm_codegen/` -- lowers `log`/`exp`/... to `_call_libdevice_unary_op(b, "log", v)`, i.e. **libdevice double** |
| `LUISA_EXPERIMENTAL_XIR_CODEGEN=1` (or a kernel needing autodiff) | XIR |
| neither (the **default**) | `cuda_codegen_ast.cpp`, which emits `lc_log`/`lc_log2`/`lc_log10`/... for `CallOp::LOG*` |

The default (AST) path lands in `cuda_builtin/cuda_device_math.h`, where the `lc_double`
overloads of **19** functions are the single-precision ones:

```cpp
__device__ inline lc_double lc_log  (lc_double x) noexcept { return logf(x); }
__device__ inline lc_double lc_acos (lc_double x) noexcept { return acosf(x); }
__device__ inline lc_double lc_exp  (lc_double x) noexcept { return expf(x); }  // + sin/cos/tan, the
...                                                                              // arc/hyperbolic ones,
__device__ inline lc_double lc_atan2(lc_double y, lc_double x) noexcept { return atan2f(y, x); }
                                                                                 // log2/log10/exp10,
// and powf_impl(lc_double, lc_double) falls back to                               // copysign
//     return y_int == y ? powi_impl(x, y_int) : powf(x, y);
```

(`sqrt/rsqrt/exp2/ceil/floor/trunc/round/rint/fma/fmod` are correctly double.) **So with the
default codegen every double `log`/`exp`/`pow`/trigonometric in a ported kernel is evaluated at
~1e-8 relative accuracy instead of ~1e-16**, while the reference (nvcc-compiled) code uses the
accurate double functions. Measured in-tree, before the fix:
`log(0.5)` -> `-0.69314712285995483` (rel 8.32e-08), `exp(0.5)` rel 3.19e-08, `acos(0.5)` rel
2.78e-08, `sin(0.5)` rel 2.26e-07; the rod/spring constitutions of T5.2c show the same ~1e-7 in
their per-element device-vs-host comparison.

**Fix (applied):**
```
python scripts/patch_lc_double_transcendentals.py        # idempotent; --check reports only
# rebuild + reinstall LC (cuda_builtin/*.h is embedded by luisa_embed_device_lib, so
# cuda_builtin_embedded.cpp regenerates, and the change reaches the runtime)
cmake --build C:/dev/LuisaCompute/build-msvc -j 12
cmake --install C:/dev/LuisaCompute/build-msvc --prefix C:/dev/LuisaCompute/install-msvc
```
After the fix the same kernel is **bit-exact** against `std::` for `log`, `pow(x,0.5)`, `exp`,
`acos` and 1.4e-16 for `sin`. This is guarded permanently by the
`device_double_math_accuracy` case in `apps/tests/backends/luisa/luisa_device_math_tests.cpp`
(bound 1e-13, which separates double from float unambiguously) -- if a future LC upgrade
switches codegen or drops the patch, that case fails loudly instead of quietly degrading every
constitutive routine.

### 1.5 Three deployment traps that silently change the numbers (all hit in T5.2c)

1. **A stale LC runtime next to the tests.** libuipc deploys `${LUISA_COMPUTE_PREFIX}/bin` next
   to the module. That used to be a `POST_BUILD` step of the `luisa` target, which only runs
   when the module is **relinked** -- so rebuilding LuisaCompute alone left an *older*
   `luisa-backend-cuda.dll` (and `.dll`s) beside `uipc_test_backend_luisa.exe`, and every
   in-tree measurement silently used that older LC (it was even a different build of the same
   version: its md5 matched neither `install/bin` nor `install-msvc/bin`). It is now an
   always-run `add_custom_target(luisa_runtime_deploy ALL)` with the module depending on it.
   After rebuilding LC, verify
   `md5sum build/Release/bin/luisa-backend-cuda.dll C:/dev/LuisaCompute/install-msvc/bin/luisa-backend-cuda.dll`
   -- they must match.
2. **`luisa_compute_deploy_runtime` remembers per target.** LC's own deploy function returns
   early once its `LUISA_COMPUTE_RUNTIME_DEPLOYED` property is set, so a standalone scratch
   project (e.g. `scratch/lc_math_probe`) keeps running whatever it deployed the first time.
   That produced a *wrong* conclusion during this investigation (a probe with a patched DLL
   looked unaffected by a patch that was in fact decisive). Delete the scratch build directory
   (or at least its LC DLLs and `.cache`) before re-measuring, and prefer asserting with
   `md5sum` that the consumer's DLL is the one you just built.
3. **The LC PTX cache is codegen-specific.** `<exe dir>/.cache/kernel_<hash>.ptx` is keyed by the
   kernel IR, not by the backend DLL or the builtin sources, so it can serve PTX produced by the
   *other* codegen path. After changing LC (or its codegen selection), purge
   `rm -rf build/Release/bin/.cache build/Release/bin/.data` -- it costs a full re-JIT (~17 min
   for the whole suite) and it is the only way to be sure nothing stale is reused.

## 2. Build & run recipes (Windows)

```bat
:: LC (only when its sources changed)
set VSLANG=1033
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake --build C:\dev\LuisaCompute\build -j 12
cmake --install C:\dev\LuisaCompute\build --prefix C:\dev\LuisaCompute\install
```

Standalone spike (T0.1, also the smoke test template for T1.3):

```bat
cmake -S scratch\lc_spike -B scratch\lc_spike\build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CXX_COMPILER=D:/TestUE/clang/bin/clang++.exe
cmake --build scratch\lc_spike\build
scratch\lc_spike\build\lc_spike.exe cuda
```

### 2.1 Helper scripts (use these instead of retyping the recipes)
All three go through `cmd /c` + `vcvars64.bat` (never MSYS bash, which mangles
`cmd //c` arguments) and set `VSLANG=1033`.

```bat
:: in-tree: build + run the backend test target (uipc_test_backend_luisa.exe)
python scripts/luisa_build.py build backend_luisa
python scripts/luisa_build.py run
python scripts/luisa_build.py ctest

:: standalone luisa_device harness (LC only, no vcpkg libraries)
python scripts/luisa_standalone.py cuda

:: single-TU syntax / template-instantiation check with the exact CMake flags
:: borrowed from build/compile_commands.json
python scripts/luisa_check_compile.py <source.cpp> --syntax
```

`luisa_check_compile.py` is the fast inner loop: it surfaces template
instantiations (including the DSL ones, which are ordinary host code) in a few
seconds instead of a full build + link.

---

## 3. Core mapping table (verified patterns)

### 3.1 Kernel definition & dispatch

| cuda_tool / CUDA | luisa_device |
|---|---|
| `__global__ void k(BufferView<T> v, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; ... }` | `Kernel1D k = device.compile<1>([&](LBufferView<T> v, ...) noexcept { auto i = dispatch_x(); $if (i >= n) { return_; }; ... });` |
| `k<<<best_grid_dim(n,k), best_block_dim(k)>>>(v, n)` | `stream << k(v, n).dispatch(n);` |
| `cudaStreamSynchronize(s)` | `stream << synchronize();` / `LStream::synchronize()` |
| `cudaMemcpyAsync(dst,h,n,H2D,s)` | `stream << buf.copy_from(span{h, n});` |
| host read back | `stream << buf.copy_to(span{h, n}) << synchronize();` |

**Rule K1.** `device.compile<1>` is called **once per call-site** and the resulting
`Kernel1D` (a `Shader`) is cached for the process lifetime (`luisa_device::KernelCache`).
Never compile inside a frame loop (R4).

**Rule K2.** Anything that depends on runtime values must use DSL control flow
(`$if`, `$while`, `$for`, `$switch`, `$invoke`) — never host `if`/`for`, which would be
evaluated at AST-build time (R3).

```cpp
// WRONG: host branch evaluated while building the AST
for (uint i = 0; i < n; ++i) { v->write(i, 0.0f); }   // n is a host value -> unrolled/ignored

// RIGHT: device loop
$for (i, 0u, n) { v->write(i, 0.0f); }
```

**Rule K3.** Kernels take **views** (`LBufferView<T>`), never raw `Buffer<T>` handles,
so call-sites keep the cuda_tool signature shape.

### 3.2 Buffers & views

LC's `luisa::compute::BufferView<T>` is a lightweight, non-owning handle
(handle + stride + byte offset + size) and is exactly the type a kernel invocation
accepts for a `Buffer<T>`/`BufferVar<T>` parameter — including **subviews**, so the
byte offset is honoured by the backend. `luisa_device::LBufferView<T>` adds
ownership (a `shared_ptr` to the LC Buffer) plus the cuda_tool accessor surface and
an implicit conversion to the LC view:

```cpp
// host: LDeviceBuffer<T> mirrors cuda_tool::DeviceVector
lud::LDeviceBuffer<Vector3> pos;
pos.resize(n);

// kernel: parameters are the LC prototype types
static auto k = lud::compile1<luisa::compute::Buffer<Vector3>, uint32_t>(
    [](luisa::compute::BufferVar<Vector3> p, luisa::compute::Var<uint32_t> n) noexcept {
        auto i = luisa::compute::dispatch_x();       // == dispatch_id().x
        $if (i < n) { p->write(i, p->read(i) * 2.0); };
    });

// dispatch: pass the (possibly offset) view directly
stream << k(pos.view(offset, count), count).dispatch(count);
```

**Rule B0 (kernel parameters).** A kernel lambda's buffer parameter must be
declared with the *prototype* type `luisa::compute::Buffer<T>` (equivalently
`BufferVar<T>` as the creation type). `BufferView<T>` is the **host-side invocation
argument**, never a parameter prototype — LC only maps `Buffer<T>` prototypes.

**Rule B1 (view lifetime — R5).** LC `Buffer<T>` has fixed capacity; growing an
`LBuffer` reallocates. **Views are frame-scoped**: never store an `LBufferView<T>`
across a `resize()`. This matches cuda_tool usage (views are short-lived).
Debug builds record a generation counter and assert in `LBufferView::use_check()`.

**Rule B2.** `LBuffer` growth over-allocates: `capacity = max(2*capacity, required)`,
so steady-state frames stop reallocating (avoids R5 in the common case).

### 3.3 Atomics

| CUDA | luisa_device |
|---|---|
| `atomicAdd(&v(i), x)` | `v.atomic(i).fetch_add(x)` |
| `atomicAdd(&eigen_vec.x(), x)` (3 scalars) | `math::atomic_add(v.atomic(i), LVec{...})` (3 fetch_adds) |
| `atomicMax(&v(i), x)` | `v.atomic(i).fetch_max(x)` |
| `atomicCAS(...)` | `v.atomic(i).compare_exchange(...)` |

**R9 gap closed — `AtomicRef<double>`.** LC's DSL specializes `AtomicRef<T>` for
int/uint/slong/ulong and float, but **not** for `double`; `v.atomic(i).fetch_add(x)`
on a `Buffer<double>` therefore fails to compile. Its CUDA codegen already maps
`ATOMIC_FETCH_ADD` to `atomicAdd`, which has a double overload, so only the DSL
type was missing. `luisa_device/atomic.h` adds the specialization (a legal explicit
specialization in `luisa::compute::detail`) — verified on the GPU by
`scratch/lc_spike/spike2.cpp` and `luisa_device_tests.cpp::atomic_and_reductions`.
If a non-CUDA LC backend later lacks double atomics, this one header is the place
to swap in a CAS-on-`ulong` emulation.

**R9 gap — no raw warp shuffles.** LC has no `__shfl_*_sync`; use the higher level
primitives (`warp_active_sum`, `warp_prefix_sum`, `warp_read_lane`) or the
`luisa_device::warp::shfl_*` shims in `builtins.h`, which implement them in terms
of `warp_read_lane`.

### 3.4 Warp / block intrinsics (`luisa_device/builtins.h`)

| CUDA | luisa_device |
|---|---|
| `__shfl_down_sync(m, v, o, w)` | `warp::shfl_down(v, o, w)` |
| `__shfl_up_sync`, `__shfl_xor_sync`, `__shfl_sync` | `warp::shfl_up/xor(/shfl)` |
| `__ballot_sync(m,p)` | `warp_active_bit_mask(p)` |
| `__any_sync` / `__all_sync` / `__activemask` | `warp_active_any/all/bit_mask` |
| warp sum reduction | `warp_active_sum(v)` |
| block sum reduction | `block_reduce_sum(v)` (uses `Shared<T>` scratch) |
| `__popc/__clz/__ffs/__brev` | `popcount/clz/find_first_set/reverse_bits` |
| `__float_as_int` & friends | `as<int>(f)` bit-cast |

### 3.5 Eigen → `lc::math` (Phase 2)

| Eigen | lc::math |
|---|---|
| `Eigen::Vector<T,N>` | `LVec<T,N>` |
| `Eigen::Matrix<T,R,C>` | `LMat<T,R,C>` |
| `Eigen::Vector3` / `Matrix3d` | `LVec<T,3>` / `LMat<double,3,3>` |
| `Eigen::Array` | `LVec` (element-wise via `lps::`) |
| `m.inverse()` | `math::inverse(m)` |
| `eigen::evd(m)` | `math::evd(m)` |
| `eigen::svd(m)` | `math::svd(m)` |
| `eigen::pd(m)` | `math::polar_decomposition(m)` |
| `std::pow/min/max/...` in kernels | `lc::math::pow/min/max/...` |

**Rule M1.** `LVec`/`LMat` must work for **host `T`** (`float`/`double`, used by the
reference unit tests) *and* **DSL `T`** (`Float`/`Double`). Storage is a plain array so
`LUISA_STRUCT` can register them where a buffer of them is needed.

**Rule M2.** `sym/*.inl` bodies are pure element-wise arithmetic. Port them
**mechanically**: `Eigen::Vector<T,N>`→`LVec<T,N>`, `Eigen::Matrix<T,R,C>`→`LMat<T,R,C>`,
`std::`→`lps::`. Do not rewrite the math.
**Resolved in T2.1** (see §7 for the concrete contract): the shim namespace is
`uipc::backend::luisa::luisa_device::math`, with the short alias
`uipc::backend::luisa::luisa_device::lps`, and the DSL element type is
`lcrt::Var<double>` (**not** `Expr` — `Expr` deletes copy assignment). Two further
rules came out of that work and apply to *every* ported kernel: RULE M1 (LC's DSL
scalar operators live at global scope, so a surrounding namespace that declares its
own operators shadows them — use `math::detail::{add,sub,mul,div,neg}` there) and
RULE M2 (no `Expr<bool>`/`Expr<uint>` may leak into host control flow).

### 3.6 Linear algebra (Phase 3)

`cuda_tool::DeviceTripletMatrix/DoubletVector/BCOOMatrix/BSRMatrix/…` and the
`*View` types are re-implemented as `luisa_device::linalg::L*` over LC buffers with the
same accessor surface. CUB (`DeviceRadixSort/Scan/Reduce/Select/Partition/…`) is
replaced by `luisa_device::linalg` primitives implemented with LC kernels (D4 — no
cuSPARSE/cuBLAS/CUB).

### 3.7 Logging / device print

| cuda_tool | luisa_device |
|---|---|
| `cuda_tool::KernelCout` | `luisa_device::KernelCout` (thin facade) |
| `printf` in kernel | `device_log("fmt {}", expr)` — LC prints to the log on dispatch |
| `cuda_tool::Debug::set_sync_callback` | `luisa_device::debug::set_sync_callback` |

`device_log` output is captured by LC and emitted at dispatch/sync time
(`[DEVICE] …` lines), verified by the T0.1 spike (32 warp lines). Kernel print is
**not** free — gate it behind `UIPC_LUISA_KERNEL_PRINT` like `kernel_cout.h` does.

---

## 4. Backend shell rules (Phase 1 / T1.4)

1. Namespace: `uipc::backend::luisa` (mirrors `uipc::backend::cuda`).
2. Same header *names* as the CUDA shell (`type_define.h`, `sim_system.h`,
   `sim_engine.h`, `sim_engine_state.h`, `sim_system_slot.h`, `sim_action.h`,
   `sim_action_collection.h`, `kernel_cout.h`).
3. Systems self-register with `REGISTER_SIM_SYSTEM(T)` in the `.cpp`.
4. Public system interfaces (info structs, `SimSystem` subclasses) stay **identical**
   so cross-system code compiles unchanged (plan §6 rule 3).
5. Never modify `src/backends/cuda/` or `src/backends/common/` (read-only references).

## 5. File / naming conventions

| Item | Convention |
|---|---|
| Extension | `.cu` → `.cpp`; device-only helpers → `.h`/`.inl` |
| Namespace | `uipc::backend::luisa` (was `uipc::backend::cuda`) |
| Device util namespace | `uipc::backend::luisa::luisa_device` (was `…::cuda_tool`) |
| Math namespace | `uipc::backend::luisa::luisa_device::math` (aliases: `lps` inside `luisa_device`, `lc::math` in backend code) — see §7 |
| LINEAR namespace | `uipc::backend::luisa::luisa_device::linalg` |
| Launch qualifier | the `UIPC_HOST/UIPC_DEVICE/UIPC_GLOBAL` macros are **removed**; kernels are DSL lambdas |
| Buffer view param | `LBufferView<T>` / `LCBufferView<T>` |

## 6. Review checklist (per ported kernel)

- [ ] Every runtime branch/loop uses `$if/$while/$for` (no host control flow).
- [ ] Kernels dispatched via `LStream`, batched into the frame `CommandList`.
- [ ] Views are frame-scoped (no view survives a resize).
- [ ] Atomically-updated scalars use `fetch_add`-style ops, in a deterministic order
      where the result is compared against CUDA (R6).
- [ ] No `cuda_tool::`, `Eigen::`, `cub::`, `__shfl*` or raw CUDA intrinsic remains.
- [ ] Unit test added under `apps/tests/backends/luisa/` (+ sim case where applicable).
- [ ] New patterns appended to this cookbook.

## 7. Numerics layer (`luisa_device/math`) — patterns established in T2.1
The device has no separate compilation: a kernel body is ordinary host C++ that
*records IR*, so a numeric routine is written **once**, as a template over the
scalar type. `UIPC_GENERIC Foo(...)` becomes `template <typename T> Foo(...)`, and
the scalar type is either

| instantiation | meaning |
|---|---|
| `double` | host evaluation (reference / unit tests) |
| `lcrt::Var<double>` | device evaluation (a DSL local variable) |

`uipc::Float` is `double`, so ported device code keeps **double precision on the
GPU**, exactly like the CUDA backend.

### 7.1 `LVec<T,N>` / `LMat<T,R,C>` (`math/vec.h`, `math/mat.h`)
```cpp
// host                                  // device (inside a kernel lambda)
using Vec3 = lmath::LVec<double, 3>;     using DVec3 = lmath::LVec<lcrt::Var<double>, 3>;
using Mat9 = lmath::LMat<double, 9, 9>;  using DMat9 = lmath::LMat<lcrt::Var<double>, 9, 9>;
```
* Element type is **`Var`, never `Expr`**: LC's `Expr` deletes copy assignment
  (`Expr &operator=(Expr) = delete`), so a container of `Expr` could not be
  assigned. `Var` is a mutable DSL local and *is* assignable.
* Storage is a plain `std::array<T,N>` **column-major** (`index(r,c) == c*R + r`),
  i.e. memcpy/Eigen compatible; the *constructor* is row-major, like Eigen's
  `Matrix(a, b, c, ...)`.
* Default construction is **zero** (Eigen leaves it uninitialised). For the DSL
  this is what `Var`'s default initialiser does anyway, so
  `Vector9 ret; ret.segment<3>(0) = ...;` ports 1:1.
* Everything Eigen returns as a *block expression* comes back as a lightweight
  **strided view** that is assignable and implicitly convertible to a value:
  ```cpp
  auto dEda1 = ret.segment<3>(0);  dEda1 = expr;   // writes through to ret
  H.block<3, 3>(0, 3) = M;                         // 9x9 tile assembly
  A.diagonal().array() += f;                       // `.array()` is element-wise
  ```
  A view holds a raw element pointer plus a stride. That is safe here *because*
  the enclosing `LVec`/`LMat` is a **host** object holding expression handles, so
  a pointer into it behaves like a pointer into a host array. Usual caveat: a view
  must not outlive its parent; a `const` parent — or an rvalue such as `(A * B)` —
  yields a **copy** instead of a view.
* Defined: indexing `v(i)` / `m(r,c)`, `x()/y()/z()/w()`, arithmetic (`+ - * /`,
  scalar and matrix/vector products, outer product `a * b.transpose()`),
  `transpose`, `block<>`, `head<>/tail<>`, `row/col/diagonal`, `segment<>`,
  `dot`, `cross`, `norm`, `squaredNorm`, `normalized`, `sum`, `trace`,
  `maxCoeff/minCoeff`, `Zero/Ones/Identity/Constant/Unit<I>`, `asDiagonal`,
  `Map(ptr)`, and `.array()` element-wise algebra.
* Deliberately **absent** here: `determinant`, `inverse`, `make_spd`, SVD — those
  belong to T2.2 (`matrix_ops.h`, `svd.h`).

### 7.2 RULE M1 — LC's DSL scalar operators live at **global** scope
`luisa/dsl/operators.h` declares the arithmetic/comparison operators for DSL values
in the **global namespace** (reachable only by unqualified lookup that actually
reaches global scope, and no built-in candidate exists for a DSL type). Therefore:
* Any header that declares *its own* operators inside a namespace that encloses the
  call site **shadows** them: unqualified `expr_a + expr_b` then fails with
  `C2676: 'const Var<double>' does not define this operator`.
* Inside the numerics layer this is solved once, in `math/math.h`, by routing
  **every** arithmetic operation on a possibly-DSL *scalar* through
  `math::detail::{add,sub,mul,div,neg}`, which forwards to `::operator OP(...)` for
  DSL values and to the built-in operators for host values.
* The assumption is asserted (`static_assert(requires(lcrt::Var<double> a,
  lcrt::Var<double> b) { ::operator+(a, b); })`) so an upstream change fails loudly
  there instead of subtly at a call site.
* Ported code elsewhere is unaffected: `uipc::backend::luisa` and the system
  namespaces declare no such operators, so ordinary lookup reaches global scope.
  **If a new namespace ever declares its own arithmetic operators, DSL arithmetic
  inside it breaks** — use the same trampolines there.

### 7.3 RULE M2 — no DSL boolean or index may leak into host control flow
An `Expr<bool>` cannot be a C++ `if` condition, and an `Expr<uint>` cannot index a
host container. In generic code comparisons are therefore performed only by the DSL
intrinsics (`lcrt::min/max/clamp/sign/select`), never by `?:` or `<` on the device
path; bounds are host `int`/`uint32_t` constants or come from `$for`/`dispatch_x()`.
Related: use plain `uint32_t` constants for byte/slot offsets inside kernels — an
unscoped `enum { ... }` is not accepted as an operand of `Expr<uint>` arithmetic.

### 7.4 Testing (`apps/tests/backends/luisa/`)
* **Host/device equivalence by construction.** Write the computation once as
  ```cpp
  template <typename S>
  Result compute(const LVec<S, 3>& a, const LVec<S, 3>& b);
  ```
  instantiate it in a host reference loop (`S = double`) *and* inside the kernel
  (`S = lcrt::Var<double>`), then compare the resulting buffers bit-for-bit. This
  is already one unit of the M9 equivalence harness.
* Eigen parity cases are guarded with `#if __has_include(<uipc/common/type_define.h>)`
  (not merely `<Eigen/Core>`), so the same source still builds in the standalone
  harness, which links no libuipc. `Eigen::Vector::cross` / `asDiagonal` also need
  `<Eigen/Geometry>`.
* Catch2 forbids `&&` inside an assertion — wrap the whole expression in an extra
  pair of parentheses: `CHECK((a && b));`.
  * **Test-case order matters.** `zzz_runtime_shutdown` tears LC down explicitly
    (without it Windows exits with 0xC0000409) and must run last. Catch2 runs cases
    in the sorted source-glob order, so the teardown lives alone in
    `zzz_runtime_shutdown.cpp`; never name another test file so that it sorts after.
    Because that teardown is a *case*, a **filtered** run
    (`uipc_test_backend_luisa.exe "[linear]"`, `scripts/luisa_build.py run "[tag]"`)
    skips it and then exits with the same `0xC0000409`; the `RuntimeShutdownListener`
    in the same file performs the identical teardown at the end of the run so that
    filtered runs also exit `0`. If you select a subset, check the exit code, not just
    the "All tests passed" line.

* Helpers: `scripts/luisa_build.py {build|run|ctest}`,
  `scripts/luisa_standalone.py [backend]`, and
  `scripts/luisa_check_compile.py <file> --syntax` (single-TU syntax/instantiation
  check that borrows the exact CMake flags from `build/compile_commands.json`).

### 7.5 `matrix_ops.h` / `evd.h` / `svd.h` (T2.2)
| Eigen / reference | `lc::math` |
|---|---|
| `m.determinant()` | `math::determinant(m)` / `m.determinant()` |
| `m.inverse()` | `math::inverse(m)` / `m.inverse()` (2x2/3x3/4x4 closed form) |
| `cuda_tool::eigen::inverse<T,N>(m)` | `math::inverse(m)` |
| `cuda_tool::eigen::evd` / `evd_jacobi` | `math::evd(...)` / `math::jacobi_evd(...)` |
| `cuda_tool::eigen::svd` / `math::qr_svd` | `math::svd(...)` / `math::qr_svd(...)` |
| `cuda_tool::eigen::pd` | `math::pd(...)` / `math::polar_decomposition(...)` |
| `utils/make_spd.h::make_spd` | `math::make_spd(H)` / `math::make_spd_inplace(H)` |

* **The adjugate inverses are transcribed term-for-term** from
  `cuda_tool/eigen/analytic_inverse.h`, including the left-to-right summation order
  of the 4x4 entry formulas, so the floating-point result stays comparable with the
  CUDA backend. `scratch/gen_adj4x4.py` regenerates and *proves* the 4x4 table
  against an exact adjugate before writing any C++ -- hand-simplifying those sign
  patterns is where the first version went wrong.
* **`evd.h` uses fixed-sweep cyclic Jacobi**, not the reference's
  `while (off-diagonal sum > 1e-6) { find max (p,q); rotate; }`: both the
  termination test and the pivot search are *device* values there, which cannot
  drive host control flow (RULE M2). 20 cyclic sweeps reach machine precision for
  N <= 12 and the routine is branchless, so one implementation serves host and
  device. `sort_eigensystem` is a device-safe odd-even transposition network (the
  comparison *pattern* is host-fixed; the results only feed `select`s).
* **`svd.h` implements one-sided Jacobi (Hestenes)** instead of porting the
  Givens + implicit-shift QR of `algorithm/qr_svd.hpp`. That reference is driven by
  device data at every decision (`while (|a_i| > tol)`, a 5-way `if / else if`
  termination, `if (y2 == 0)`, `if (tau > 0)`, ...), so a transcription would need
  `$if`/`$while` and would be device-only. One-sided Jacobi is fixed-sweep,
  branchless, has relative accuracy on `sigma_min`, and is therefore *one*
  implementation for both sides. The public surface (`svd`, `qr_svd`, `pd`,
  `polar_decomposition`) and the signed-SVD convention are preserved.
* **Signed-SVD convention.** The reference absorbs a reflection into the smallest
  singular value: `U`, `V` proper rotations, `sigma(2)` possibly negative. Here `U`
  is built as a right-handed frame (`u2 = cross(u0, u1)` => `det(U) = +1` by
  construction) and the sign of the *third column of B* is carried by `sigma(2)`;
  `V` is then flipped as a whole (`V(:,2) *= -1` **and** `sigma(2) *= -1`, which
  leaves `U diag(sigma) V^T` unchanged). Getting this wrong silently breaks
  `U diag(sigma) V^T == A` for `det(A) > 0` inputs -- check the *reconstruction*,
  not just the singular values.
* `make_spd` clamps eigenvalues with `select(lambda, 0, lambda < 0)` and skips the
  eigenvector sort (irrelevant to the projection, and the sort is the most
  expensive part of the generated IR).

### 7.6 Four traps found while building this layer (each cost a debugging round)
1. **`select(f, t, p)` is `p ? t : f`** -- *t* first, *f* second (LC has the same
   order). The reverse reads naturally and silently swaps the branches:
   `select(div(v, n), fallback, n > eps)` returns the *fallback* for healthy input.
2. **No short-circuit `&&`/`||` for scalar DSL booleans.** LC's `&&` demands
   *boolean vector* operands and fails with a static_assert telling you to use the
   bitwise spellings -- write `(a <= b) & (c <= d)` and `!(a | b)`.
3. **`math::select` must not accept containers.** LC's own `select` has a
   custom-struct overload that also matches `LVec`/`LMat`, so a container overload of
   ours is ambiguous -- or worse, silently builds a `Var<LVec<...>>` and blows up
   inside `Type::of`. The layer's `select` is therefore scalar-only and host-only
   (DSL scalars are served by LC's own, found via ADL), and containers are selected
   element-wise:
   ```cpp
   auto axis = make_vec<storage_of_t<T>, 3>([&](int i) {
       auto c0 = select(zero_of<T>(), u0(i), use0);
       auto c1 = select(c0, u1(i), use1);
       return select(c1, u2(i), use2);
   });
   ```
4. **Dependent static members need `template`**: `LVec<T,3>::template Unit<0>()`
   inside a template, otherwise MSVC parses `Unit < 0` as a comparison.

### 7.7 Verifying a ported algorithm
The fastest way to find a bug in this kind of branchless port is to **replicate the
algorithm in Python** (`scratch/debug_svd*.py`) and assert orthonormality,
reconstruction and eigen-residuals there. Both SVD bugs found in T2.2 (the rotation
sign convention and the inverted `select`) were localised this way in minutes,
whereas the C++ test only reported "check failed". Keep such a script for every
non-trivial ported routine: it is cheap and it is reusable across back ends.

### 7.8 Patterns established in T2.3 (distance / CCD)
* **Generated arithmetic is transcribed by a generator, never by hand.**
  The reference's gradient/Hessian bodies are Symbolic-Math-Toolbox output
  (`t17 = -v11 + v01;`). They are copied verbatim, only re-spelled through the
  `detail::{add,sub,mul,div,neg}` trampolines (RULE M1). `scratch/gen_distance.py`
  rewrites every `// >>>GENERATED:name>>>` … `// <<<GENERATED:name<<<` region
  straight from the reference (`--check` verifies the tree is in sync), so the
  arithmetic order cannot drift under hand editing. Typing the bodies by hand is
  exactly where T2.2's first 4x4 adjugate went wrong.
* **A `bool`-returning reference routine that reads device data needs two
  flavours.** `need_mollify` and the four `*_ccd` (also the 8 broad phases) return
  a predicate that is *device* data on the device path, and an `Expr<bool>` cannot
  be a host `bool` (RULE M2). The pinned declaration keeps its `bool` return for
  `T` arithmetic; a second, `requires(!std::is_arithmetic_v<T>)` overload returns
  `bool_of_t<T>` (= `lcrt::Expr<bool>`) with an identical parameter list. Both call
  one generic `_impl`, so the logic exists once. `bool_of_t` lives in
  `distance_common.h`.
* **Flag dispatch = host tables + predicated select, not branches.** A flag vector
  is device data, so `if(dim == 2) …` has to become "evaluate every candidate, then
  select". The reference's per-dim layouts (`pp_from_*`, `pe_from_pt`, `pe_from_ee`)
  become host constant tables, and each terminal case is evaluated on the
  *host-permuted* points (so it is literally the same canonical routine with the
  same inputs) with its result selected by a mutually exclusive predicate. Cost is
  `O(#cases)` extra IR instead of a branch; correctness is exact, because a select
  passes its chosen operand through untouched.
* **Predicated fixed-iteration loops (CCD).** `max_iter` is a *host* value in the
  reference, so the loop stays host-bounded (`detail::ccd_iterations`). Everything
  else is predicated on a `running` flag that a termination test clears; the state
  update is multiplied by 0 while it is clear, so the state freezes and the extra
  iterations are no-ops. The two `return`s are read off `running`/`hit` afterwards.
  Cost: the loop is unrolled `max_iter` times at trace time — pass a
  convergence-sized bound (not the reference's `constexpr Size_t max_iter = 1000`
  budget) at a device call site.
* **`pred_and` / `pred_or` for host/device-generic boolean chains.** LC requires
  `&`/`|` for DSL scalars (trap #2), but on the host `bool & bool` *promotes to
  `int`*, so a third operand mixes `int` and `bool` (MSVC C4805). Fold 3+-term
  predicate chains through `math::distance::detail::pred_and/pred_or`.
* **Comparing a large device dump against the host.** Give the generic routine
  `load(j)` / `put(v)` callbacks so the host and the kernel walk an identical,
  fixed dump order (the kernel's `put` just writes the next buffer slot), and make
  the comparison `NaN`-aware: a degenerate configuration legitimately produces the
  same non-finite value on both sides, while `0/0`-style garbage that *differs*
  between the compilers (FMA contraction) is only tolerated for well-conditioned
  inputs — keep exactly-degenerate configurations in the host reference-equivalence
  tests, not in the device smoke.

### 7.9 Patterns established in T3.1 (linear containers & views)
`src/backends/luisa/luisa_device/linear/{views.h,containers.h,linear.h}` port
`cuda_tool/linear_system.h` + `linear_system/views.h` into
`luisa_device::linalg` (alias `lla`). The five rules below are what the port
turned out to depend on.

1. A block is **flat scalar storage**, column-major inside the block.
   `DeviceTripletMatrix<T,M,N>::values()` becomes one `LDeviceBuffer<double>` of
   `nnz*M*N` elements with `index(i,r,c) == i*M*N + c*M + r`, i.e. the bytes of
   block `i` are exactly `Eigen::Matrix<T,M,N>` / `math::LMat<T,M,N>` bytes;
   `M == N == 1` stays a plain scalar with no indirection. Measured
   (scratch/lc_lin_spike) before choosing this: `lcrt::Matrix<double,3>`
   (`double3x3`) *is* a legal buffer element, but its stride is **96 B** — the
   columns are `alignas(16)`, starting at +0/+32/+64 — against Eigen's 72 B, so
   the device ABI and the Eigen/T3.3-comparison ABI differ; a `LUISA_STRUCT` POD
   element can be made 72 B (9 doubles, Eigen-compatible) and is writable through
   `Var<struct>` member assignment, but it needs one registered struct type per
   `(M,N)` shape and its accessor surface is not the `LMat` one — it cannot back
   the generic `<T,M,N>` block API. Flat scalars win on both layout and
   genericity.

2. A view owns **handles, not pointers**, and is never captured into a kernel.
   Views are built from `LBufferView<S>` handles (host / consumer side) or from
   `lcrt::Expr<lcrt::Buffer<S>>` handles (inside a kernel, from the kernel's
   arguments). Both reach the same accessor code through a *storage flavour*
   template parameter — `storage::buffers`, `storage::device` and
   `storage::memory` (raw host pointers, used by the unit tests and by
   host-reference evaluation of a ported body). A view is **passed as launch
   arguments**, never captured: a captured `BufferView` becomes a *bound
   argument* of the compiled shader, and LC bakes the buffer handle in at
   compile time. scratch/lc_lin_spike demonstrates the trap — after the owner
   reallocates, re-dispatching the same compiled shader rewrites the **old**
   buffer and leaves the new one untouched. So:

     ```
     static auto k = lud::compile1<lcrt::Buffer<int>, lcrt::Buffer<int>,
                                   lcrt::Buffer<double>, uint32_t>(
         [](BufferVar<int> rows, BufferVar<int> cols, BufferVar<double> vals,
            Var<uint32_t> n) noexcept {
             auto A = lla::LCTripletMatrixDeviceView<double, 3>::from_handles(
                 rows, cols, vals, n, n, n);
             ...
         });
     stream << k(A.row_indices(), A.col_indices(), A.values(), A.triplet_count())
                   .dispatch(A.triplet_count());
     ```
   This is the concrete form of R5 ("views are frame-scoped"): the host views
   carry a shared generation cell and `view.use_check()` throws when the owner
   resized after the view was taken.

3. Device indices are **signed** (`lcrt::Expr<std::int32_t>`), matching the
   reference's `int` indices, and every caller-supplied index goes through
   `Storage::as_index(...)` so a kernel may pass `dispatch_x()`
   (`Var<uint32_t>`), a buffer read or a literal without casting. Two related
   traps: `Expr<T>` has **no default constructor** (so `CTriplet`/`CDoublet` are
   aggregates without default member initialisers and are always aggregate
   initialised) and **no copy assignment** (so sub-views/`as_const()` are
   *constructed*, never copy-and-assign).

4. A `CView` has **no write path at all** — the write members exist only on the
   non-const flavour (`requires(!IsConst)`), and `operator()` on a const view
   returns the read-only aggregate (`CTriplet`/`CDoublet`) instead of a write
   proxy. The unit tests assert this with dependent `requires`-expressions
   (`static_assert(cannot_write_triplet<CV>::value)`); note that a
   *non-dependent* `!requires{...}` is diagnosed eagerly by MSVC, and that a
   host `LBufferView` view intentionally refuses *element access* entirely
   (`static_assert` in `storage::buffers::load`) — a host read of device memory
   would be an invisible synchronisation. Portable host-side element access is
   `storage::memory`.

5. Views are **not** registered as `vec_like`/`mat_like`, and `as_eigen()` is the
   view itself. Element access is a buffer load (a DSL expression), and RULE M1
   forbids a second arithmetic-operator namespace, so a kernel materialises a
   segment before using the numerics layer:
   `LVec<T,N> v = x.segment<N>(off).as_eigen();` — which is exactly the
   reference's `Eigen::Vector<T,N> v = seg.as_eigen();` step (an implicit
   conversion on the statically sized `segment<SegN>` view). `+=`, `-=`,
   `atomic_add(LVec)` and `to_vec<N>()` cover the mutable-map uses.

Two minimal additions were needed in the foundation (both crash/UB guards for
zero-size work, which the new containers hit with `nnz == 0`):
`LDeviceBuffer::copy_from` must not dereference the (never allocated) handle when
the span is empty, and `LBufferView::{copy_from,copy_to}` must return early for a
zero-size (or unallocated) view instead of enqueuing a copy through the invalid
native handle.

### 7.10 Patterns established in T3.2 (device algorithm primitives)
`src/backends/luisa/luisa_device/linear/algorithms.{h,cpp}` reimplements the CUB
surface the reference backend uses (`DeviceRadixSort::SortPairs`,
`DeviceScan::ExclusiveSum`, `DeviceRunLengthEncode::Encode`,
`DevicePartition::Flagged`, `DeviceReduce::Sum/Min/Max`, `cub::WarpReduce`'s
head-segmented reduction) on LC kernels. The rules below are what it depends on —
the first three are traps that each cost a debugging round.

1. **A compiled `Shader` held by a function-local static is destroyed *after*
   `Runtime::shutdown()`** and destroying an LC resource whose device is already
   gone aborts the process (`0xC0000409`, the teardown trap §7.4 documents). The
   kernel caches of the primitives are therefore *deliberately leaked*
   (`detail::cached_kernel<...>` returns a `Shader&` bound to a heap object), and
   any long-lived `LStream` is leaked too. Measured before/after: the full test
   suite printed "all tests passed" and then exited `0xC0000409` 3/3 runs; after
   the leak it exits 0 3/3 and `ctest -R backend_luisa` passes.

2. **A temporary buffer must outlive the commands that reference it.** LC's
   `CommandList` stores *raw* resource handles, so a scratch buffer that dies while
   the caller's frame list is still open leaves a dangling handle behind — the
   failure is a crash at **commit**, not a wrong number. Two consequences:
   * `LStream::cached_object(key, create, destroy)` (the one foundation addition of
     T3.2) hands the primitives a per-stream temporary set, so a primitive called
     without an explicit scratch is safe inside an open frame;
   * `detail::carve` *retires* a buffer it has to grow (it keeps the old allocation
     in the scratch's `retired` list) instead of letting `LDeviceBuffer::reserve`
     drop it: a grown scratch must never invalidate a handle an already recorded
     command still uses.
   `~LStream` releases the cached objects only while the runtime is still alive.

3. **`warp_prefix_count_bits` is *not* a segment label.** For "fold this warp's
   slice of a segment" logic (`HeadSegmentedReduce`) the label has to be constant
   for every lane of one segment and increase across segments; the exclusive ballot
   count differs between a head lane and the rest of *its own* segment (the head
   itself is counted for the later lanes). Use
   `warp_prefix_count_bits(is_head) + select(0, 1, is_head)`.
   Related, measured on the GPU by `scratch/t32_probe`: `warp_prefix_sum` is
   *exclusive* over the active lanes and `+ v` is the inclusive form;
   `warp_active_bit_mask(pred).x` is the 32-lane ballot (a `uint4`);
   `warp_active_count_bits(pred)` is its popcount; `LC_WARP_ACTIVE_MASK` is
   `__activemask()`, so **predicate the tail instead of branching** and dispatch a
   rounded-up element count so that every warp is full and the "last lane" tricks
   above stay valid.

4. **A stable radix sort without `__match_any_sync`.** LC exposes no match-any, so
   the within-warp rank of equal digits is a 32-iteration shuffle loop
   (`warp_read_lane(digit, j)` + `select`), i.e. ~1 shuffle per element per pass —
   measured acceptable. The rest of the pass is the standard two-level scheme, with
   the histogram stored *transposed* as `histogram[digit * block_count + block]` so
   that **one flat `exclusive_scan` yields both** the bucket offset
   (`P[digit*nb]`) and the earlier-blocks count
   (`P[digit*nb + block] - P[digit*nb]`); no per-digit scan is needed.

5. **Device-bounded `$for` works and is the right tool for flat-block loops.**
   `$for (c, 0u, components)` with `components` a kernel scalar argument compiles
   and runs (verified by the probe), which lets one kernel serve every T3.1 block
   shape (`segmented_reduce_blocks`, `permute_gather_strided`) instead of one
   instantiation per (M, N).

6. **`select(f, t, p)` is `p ? t : f` — and it silently returns the wrong branch.**
   Every clamp/validity select in T3.2 is written with an explicit
   `// p ? t : f` comment, and index clamping uses `lcrt::min/max` rather than a
   select where possible. (Trap #1 of §7.6, hit again in the T3.2 probe.)

7. **No `atomicMax/atomicMin` for `double` in CUDA.** The cross-warp branch of the
   segmented reduce is an *atomic* combine (as in the reference), so only
   `detail::Plus` is instantiated; a non-plus op needs a deterministic non-atomic
   fold and is rejected with a readable `static_assert`.

### 7.11 Patterns established in T3.3 (matrix converter)
src/backends/luisa/luisa_device/linear/converter.{h,cpp} ports
`cuda::MatrixConverter<T,N>` (Triplet->BCOO, BCOO->BSR, Doublet->BCOO vector,
ge2sym, sym2ge, reserve_ratio/loose_resize).  The nine rules below are what the
port turned out to depend on; the CUB -> T3.2 mapping is in the converter.h
header comment.

1. A namespace-scope `operator==` inside `linalg` hides the DSL's.  RULE M1 says
   LC's scalar operators live at global scope; the reference's
   `MatrixConverterIntPair` has a namespace-scope `operator==`, and copying that
   declaration into `luisa_device::linalg` made *every* unqualified comparison of
   DSL scalars in the namespace fail to compile — `Var<uint32_t> == Var<uint32_t>`
   reported C2678 ("no operator found") with the note that only
   `linalg::operator==(const MatrixConverterIntPair&, ...)` was a candidate, plus
   a cascade of C2672 inside `$if` conditions and `write()` calls.  Write such a
   comparison as a hidden friend (inside the struct) so only ADL finds it:
   `friend constexpr bool operator==(const A&, const A&)`.  This is the second
   concrete instance of RULE M1 (the first was the numerics layer's arithmetic).

2. `select`/`pick` evaluates *both* operands — keep the discarded one in range.
   sym2ge chooses between the diagonal and the off-diagonal index arrays with
   `pick(diag_index[i], offdiag_index[i - diag_count], is_lower)`.  The second
   index underflows to a huge `uint32_t` when `i < diag_count`, and it is
   *evaluated* (no short circuit), so the port clamps it:
   `off_i = i - min(i, diag_count)`.  One instruction; without it the kernel reads
   far out of bounds.

3. `sort_pairs_by_key`'s payload is one *element*, not a block.  The reference's
   `SortPairs(indices, indices_sorted, segments, segments_sorted, n)` on
   `Eigen::Vector<T,N>` payloads maps to `sort_index_pairs` +
   `permute_gather_strided<T>(..., components = N)`, because T3.2's
   `sort_pairs_by_key` gathers its payload with `permute_gather`
   (`components == 1`): with a flat multi-scalar payload it silently keeps only
   the first component of every element (the doublet test caught it at once;
   block size 1 is unaffected — it is the one shape where `sort_pairs_by_key` is
   the right primitive).

4. The reference's host read-back is a mid-frame synchronisation.  `int h_count =
   count;` on a device count is needed to size the next step.  Port pattern
   (`read_back_count`): remember `stream.frame_open()`, `synchronize()` (which
   commits the open CommandList and waits), read the single element through
   `stream.stream()`, then `begin()` again if the frame was open — the caller
   keeps batching across the read.  Read through the *caller's* LC stream, not
   `main_stream()`, so the ordering holds for any LStream.

5. T3.2 gap found here: `segmented_reduce_blocks` filled only `out_size` scalars.
   The primitive's identity fill covered `out_size` scalars instead of
   `out_size * components`, so with `components > 1` the slots of a segment that
   crosses a warp boundary — accumulated with `atomic_add` by its head warp (the
   reference's `is_cross_warp` path) — kept whatever the buffer held.  Invisible on
   a fresh, zero-initialised buffer (which is why T3.2's own test passed), wrong as
   soon as the out buffer is reused or pre-filled: T3.3's "convert twice into the
   same container" determinism check fails without the fix, as does the sentinel
   probe `converter_segmented_reduce_blocks_out_fill` (3/4 assertions fail before,
   4/4 pass after).  The reference fills the whole matrix buffer
   (`BufferLaunch().fill<Matrix>(out, zero)` — all M*N scalars of every element);
   the fix is `filled = out_size * components`.

6. A reused scratch grows like the reference's `loose_resize`, but retires the old
   allocation.  `reserve_ratio` is the reference's policy (`capacity < n ->
   reserve(n * ratio)`); the port exposes the whole capacity once (so
   `LDeviceBuffer::resize`'s value-initialisation of the tail is amortised) and
   pushes the displaced `LBufferView::base()` into an `m_retired` list — a command
   already recorded in the caller's open frame must not see its buffer freed (the
   `detail::carve` argument of §7.10).  This is load-bearing in sym2ge + the
   in-place sort, where one member holds a `sym_size` array and is then grown to
   `general_size`.

7. A "block copy" is a components loop or an LC copy command.  T3.1's flat layout
   turns the reference's whole-block assignments into either a device-bounded
   `$for (c, 0u, components)` (the reference's own kernels are the model) or — when
   it is a pure copy — a recorded
   `stream.record(dst.native().copy_from(src.native()))`.  Inside a conversion do
   not use `LDeviceBuffer::fill`/`copy_from`: they submit on the *main* stream,
   which is not batched into the caller's frame CommandList.  LC has no fill
   command, so a fill that must be ordered inside the frame is a small local
   kernel dispatched on the caller's stream.

8. `sym2ge` needs *two* `select_if` calls to reproduce
   `DevicePartition::Flagged`.  CUB's Flagged writes the selected elements to the
   front *and* the unselected ones to the back of the same array; T3.2's
   `select_if` writes only the selected prefix (the rest is untouched).  So the
   [Diag | Upper] halves are produced with its two flavours
   (`keep_when_zero = false` / `true`) into two arrays — same stable order, same
   split point (the returned count), no dependence on stale contents.  While
   there: the reference's split/shift uses `from.rows()`, which equals the diagonal
   count only for a well-formed packed-upper matrix; the port uses the count the
   partition actually returns, which is identical for every input the reference
   accepts and well-defined otherwise.  Its output size formula
   (`2 * (sym_size - rows) + rows`) is clamped to at least `sym_size`, because it
   underflows for an empty input.

9. The CPU model is the oracle, and "order-insensitive" needs exact values.  The
   reference converter is CUDA-only (it cannot be host-included), so T3.3's oracle
   is an independent CPU model written from the mathematical definition (group by
   (row, col) and sum; count the blocks per row and prefix-sum; mirror the
   off-diagonal blocks transposed).  A *bit-identical* result after shuffling the
   input is only achievable with exactly representable values (small integers): a
   float re-association changes the last bits, so the shuffled-input check uses
   integer-valued blocks while the general cases are compared within 1e-6.

### 7.12 Patterns established in T3.4 (SpMV + dense BLAS)
src/backends/luisa/luisa_device/linear/{spmv,blas}.{h,cpp} port
`linear_system/spmv.{h,cu}` (class `Spmv` → `lla::LSpmv<T,N>`, alias
`lla::Spmv = LSpmv<double,3>`) and `cuda_tool/linear_reduction.h` +
`cuda_tool::LinearSystemContext` (→ `lla::LLinearSystemContext`, alias
`LinearSystemContext`, plus the free-function spellings).  The nine rules below
are what the port turned out to depend on.

1. The symmetric convention, extracted from the reference kernels and preserved.
   A triplet `(i, j, B)` of the upper-packed block BCOO contributes
   `a * B * x_j` to output row `i`; when `i != j` **and only then** it *also*
   contributes `a * B^T * x_i` to output row `j`.  The diagonal is therefore
   neither halved nor doubled and has no mirrored copy; a duplicated `(i, j)`
   entry is an independent contribution and simply sums; `b` is applied to `y`
   *before* the kernel (`y = b*y`, or a zero fill when `b == 0`), so a row with no
   entry keeps its scaled value.  `rbk_sym_spmv_dot` folds the dot contribution
   with the same convention — `a * x_i^T B x_i` for a diagonal entry,
   `2*a * x_i^T B x_j` for an off-diagonal one — i.e. `x^T (a A_sym x)`,
   independent of `b`.  `rbk_spmv` is the *general* variant: it applies the stored
   entries as a plain COO matrix (a symmetric upper-packed input gives
   `a*A_upper*x + b*y`, not the symmetric product) and assumes one row's entries
   are consecutive (the converter's sorted output).

2. Capacities are not counts: clamp every index read past the device-side count.
   `rbk_sym_spmv_dot` reads its triplet count *on the device* and sizes the grid by
   `triplet_capacity`, so lanes with `dispatch_x() >= count` still run and read the
   reserved-but-unwritten tail of the row/col arrays (the reference's grid-stride
   loop skips those lanes entirely).  The *buffer* read is safe (the allocation
   covers the capacity), but the indices it yields are arbitrary device bytes; used
   as a segment offset (`x.segment<N>(col * N)`) they are an illegal memory access.
   Every such read goes through `min(max(i, 0), segments - 1)` with `segments`
   passed in as the real dof count, and the contribution is still masked by
   `valid`/gated on `is_head` so the value is unchanged.  Measured: without the
   clamp the full test suite aborted with `CUDA_ERROR_ILLEGAL_ADDRESS` when a
   reserved tail happened to hold stale bytes, while the same test passed in
   isolation (fresh allocations are zeroed); the T3.4 test now poisons the
   reserved tail (`poison_reserved_tail`) so the clamp is regression-tested.

3. The reference's atomics are kept for `y`, and R6 is met by the reductions.
   All four variants accumulate `y` with `atomic_add` exactly like the reference
   (the `rbk` variants differ only in that the entries of one output row are first
   folded by a warp-level head-segmented reduce, so one atomic per row segment
   instead of one per entry).  When several warps/blocks contribute to the same
   row the *order* of those atomics is not fixed by the programming model, so the
   last bits of `y` can differ between two runs (and between two launch shapes) —
   the reference has the same property.  The *reductions* are atomic-free
   two-pass folds (block partials + a fixed-order fold), so `dot`, `norm` and the
   fused `d_dot` are bitwise reproducible; the fused variant writes its per-block
   dot partials to a scratch array (`detail::carve`, retired on growth) and folds
   them in a second dispatch instead of `atomicAdd`-ing into `d_dot`, which is what
   makes the ``grid sized by capacity'' contract exact for `d_dot` (the extra
   all-zero partials cannot change a fixed-order sum).

4. `LVec<T,1>`'s single-component constructor hijacks copy-construction.
   `LVec`'s variadic constructor is constrained by
   `sizeof...(U) == N && (is_convertible_v<U, T> && ...)`, and `Var<T>`'s
   converting constructor is unconstrained, so for `N == 1` `LVec<Var<T>,1> v = w;`
   with `w` a **non-const lvalue** selects that constructor (a better ICS than the
   copy constructor's `const LVec&`) and fails inside `Type::of<LVec>` /
   `assign_impl`.  Two consequences for ported kernels: materialise a view with
   `view.to_vec<N>()` rather than `SomeVec v = view.as_eigen();` (the view's
   conversion operator and the LVec ctor are both viable, and `as_eigen()`'s
   `ThisView` is not `vec_like`), and copy an `LVec` parameter element-wise
   (`for(c) v(c) = src(c);`) instead of taking it by value.  Prvalue
   initialisation (`Vec v = expr;`) is fine.

5. A function-local static `Shader` anywhere in the foundation is a teardown trap.
   `view.h`'s `detail::fill_kernel<T>()` held its compiled shader in a
   function-local static, so the first `LBufferView::fill` destroyed it during
   static destruction — after the harness's explicit `Runtime::shutdown()` —
   and the process aborted with 0xC0000409 (the §7.10 trap 1).  Filtered runs that
   never call `fill` exited 0, which is why this survived T3.1–T3.3: the T3.4 BLAS
   test is the first filtered run to call it.  Fixed minimally (the one foundation
   addition of T3.4): the kernel is now a `Shader&` bound to a heap object,
   exactly like `detail::cached_kernel`.

6. A raw `ScaledSquareSum` buffer is two scalar buffers, and its combine is
   branchless.  LC buffers need a registered element type, so
   `details::LinearReductionStorage<T>` keeps `norm_scale_partials` /
   `norm_square_partials` (and the two result scalars) instead of a
   `DeviceVector<ScaledSquareSum<T>>` — the T3.3 precedent of splitting a reference
   struct buffer.  The reference's `ScaledSquareSumOp` branches on the two scales,
   which a device value cannot do (RULE M2), so the port combines with
   `min`/`max` + `ratio = (hi == lo) ? 1 : lo/hi` + a component-wise `select`,
   which reproduces all three finite cases (including the `0 vs 0` one, where the
   selected ratio is 1) and only turns the reference's NaN-propagation case into a
   finite operand — a diagnostic-only difference, since PCG tests `dot`.
   Measured: the norm of `{1e200, 1, 1e-200}` (whose naive square sum is `1e400`,
   i.e. inf in double) returns 1e200 within 1e-12 relative.

7. The reductions return a `T` and therefore synchronise — mid-frame like T3.3's
   read-back.  `dot`/`norm` follow the reference's host-facing contract and commit
   + wait, read the single result scalar through the *caller's* `LStream`, and
   re-open the frame if the caller had one (`read_back_scalar`, the same pattern as
   converter.h's `read_back_count`).  The free-function spellings keep their
   persistent storage in `LStream::cached_object` keyed per `LinearReductionStorage<T>`
   so a primitive called inside an open frame still has live partials (§7.10).

8. The PCG vector updates are FMA-contracted on the device, so an exact CPU
   comparison is wrong for them.  `axpy` / `axpby` / `update_xr` / `update_p`
   compute `y + a*x` (and friends); the GPU contracts that to one FMA while the
   host model rounds twice, so the results differ in the last bit even for
   well-conditioned inputs.  Compare them within 1e-15 relative, not `==`.
   `fill`, `scale` (one multiply) and `copy` stay exact.

9. `LinearReductionStorage`'s partials are carved, not resized.  A second call
   with a larger vector must not free the partials a command still recorded in the
   caller's open frame references, so the growth goes through `detail::carve`
   (retire the displaced allocation) exactly like the algorithm primitives —
   `LinearReductionStorage` carries its own `retired` list for that.

### 7.13 Patterns established in T4.1 (global geometry managers)
src/backends/luisa/global_geometry/* ports the reference's
global_{vertex,body,simplicial_surface}_manager + the three reporters, plus
utils/offset_count_collection and utils/dump_utils.  The rules below are what the
port turned out to depend on; the first one is a hard constraint of the device ABI,
not a choice.

1. **An Eigen type cannot be an LC buffer element.**  LC reflects a buffer element
   type through `detail::TypeDesc<T>` (member_reflect), which requires
   `std::is_aggregate_v<T>`; `Eigen::Matrix<double,3,1>` is not an aggregate, so
   `device.create_buffer<uipc::Vector3>()` does not compile.  The ported backend
   therefore stores every vector-valued attribute as `lps::LVec<T,N>`
   (`uipc::backend::luisa::LVector2/3/2i/3i`, see src/backends/luisa/type_define.h):
   a plain `std::array<T,N>`, i.e. the *same bytes* as the Eigen alias
   (LVector3 == 24 B == Vector3, verified by static_assert).  The four element
   types are registered as LC struct types with `LUISA_STRUCT(gg::LVector3, data){};`
   in that header, with three ordering constraints:
   * the aliases must be declared before the registration (the macro takes the
     type as its first argument and an unparenthesised comma inside `LVec<T, N>`
     would split its arguments);
   * `LUISA_STRUCT` must be invoked at *global* scope (inside
     `uipc::backend::luisa` the name `luisa` resolves to the enclosing backend
     namespace, §7.2/compat.h) and needs the `{};` suffix (the macro ends with
     `: luisa_compute_extension<S>`);
   * never write `LVec<::luisa::compute::...>` — `<:` is the `[` digraph and the
     tokeniser eats it (use the `lcrt` alias).
   A `LUISA_STRUCT` type is *not* `is_custom_struct`, so `create_buffer<T>()` stays
   available; host code converts with `to_luisa(Vector3)` / `to_eigen(LVector3)`
   (identical layout, unrelated types: the explicit conversion is well defined, a
   raw cast is not).  Inside a kernel an element reads as `Expr<LVector3>` whose
   components are `element.data[i]`; `load_vec3()/store_vec3()` (type_define.h)
   materialise it as `lps::LVec<Var<double>,3>` for the numerics layer.  Writing an
   element needs a `Var<LVector3>`: `dst->write(i, some_lvec)` does not convert.

2. **Ported systems have no stream parameter — they share one frame stream.**
   cuda_tool submits on the default CUDA stream, so reference system code never
   threads a stream through.  The port adds `uipc::backend::luisa::frame_stream()`
   (src/backends/luisa/frame_stream.{h,cpp}): one deliberately leaked LStream over
   `main_stream()`, shared by every ported system so that a frame opened by the
   engine batches *all* of their work.  Until Phase 8 opens it, `frame_open()` is
   false and every command is submitted immediately, exactly like cuda_tool.

3. **A device read-back inside a system is a mid-frame synchronisation (§7.11
   rule 4, restated for managers).**  `cuda_tool::copy_to` synchronises; an LC copy
   recorded on the main stream would *not* be ordered after commands still sitting
   in the caller's open frame CommandList, so the read-back must commit the frame
   first (`LStream::synchronize()`), copy, wait, and re-open if the frame had been
   open.  Used by the surface manager's `DeviceSelect::Flagged` count and by
   `compute_axis_max_displacement()`.

4. **A CUB reduce with a *vector* op has no counterpart: pack, then reduce
   scalars.**  The AABB reduction folded `Vector3` values with `cwiseMin/cwiseMax`
   and the |displacement| reduction folded the flat 3n array with a `MaxAbsOp`.
   linalg's `device_reduce_*` is scalar-typed, so the port writes the components
   transposed into one scratch array (`comps[c*n + i] = positions[i][c]`, a
   `detail::carve`d per-stream temporary) and runs 3 min + 3 max (or one max over
   per-element maxima) reductions.  min/max are exact and order-independent, so the
   result is identical to CUB's; the reduce *identity* also defines the n == 0
   case, which is why an empty scene yields `(+inf, -inf)` after the `AABB`'s cast
   to float (the reference behaves the same way).

5. **`DeviceSelect::Flagged` is `select_if(..., keep_when_zero = false)` plus a
   count read-back**, and its count is `uint32_t` (T3.2), not the reference's `int`;
   the manager stores it in an `LDeviceVar<uint32_t>` and casts on the host.

6. **Measured: a leaked static Shader survives `Runtime::shutdown()` + `init()`.**
   scratch/t41_probe/probe_shader.cpp dispatches one function-local-static compiled
   Shader, shuts the runtime down, brings it up again (a new device) and
   re-dispatches the same Shader: both rounds return the correct result (2/2).  So
   the §7.10 compile-once caches stay valid when a process creates several engines
   sequentially — every `~SimEngine` calls `Runtime::shutdown()`, and the T4.1 test
   creates a fresh engine per scene (including the dump/recover test's second
   engine).  This is what makes "dump in engine A, recover in engine B" testable.

7. **Backend-internal tests need two build settings on MSVC.**  A DLL exports
   nothing by default, so a test that uses the backend's *system* classes (the
   managers and their reporter base classes) cannot link against the module until
   the symbols are exported: src/backends/luisa/CMakeLists.txt sets
   `WINDOWS_EXPORT_ALL_SYMBOLS ON` (the foundation layer keeps its own
   `UIPC_LUISA_DEVICE_API` macro).  The test target also needs `<src>` on its
   include path because the backend headers include `<backends/common/...>`
   (apps/tests/backends/luisa/CMakeLists.txt, like apps/tests/common).

8. **Frontend facts a system port needs** (discovered in T4.1):
   * `geometry::Geometry::as<T>()` returns a raw `T*`, not a `S<T>`;
   * `geometry::SimplicialComplex` has no assignment operator (deleted): build it
     in place (a `[&]{...}()` immediately-invoked lambda returning by value);
   * `core::Scene::geometries()` is not iterable — reporters walk
     `world().scene().geometries()` (the `SceneVisitor` `span<S<GeometrySlot>>`);
   * `VertexReporter::vertex_offset()` is the reporter's *slice* offset in the
     global vertex array, not a per-geometry offset: a reporter that reports
     several geometries (all of them do) must accumulate the per-geometry offsets
     itself (prefix sum of the vertex counts);
   * the per-vertex attributes the reference reporters fill (`coindices`,
     `body_ids`, ...) are byte copies, so a test reporter can feed them with
     `LBufferView::copy_from(span<const T>)` (H2D) exactly where a real reporter
     copies device data;
   * `config()["sanity_check"]["enable"] = false` keeps the frontend sanity
     checkers out of a deliberately degenerate test scene;
   * the frontend scene config accepts backend-specific keys that are not in
     `Scene::default_config()`'s schema: `config["contact"]["d_hat_relative"] =
     <double>` is created as a `Float` attribute and read back with
     `find<Float>("contact/d_hat_relative")`, so the scene-relative d_hat override
     is testable straight from the scene JSON;
   * the manager's `*_bounding_box()/compute_*()` helpers are *private* (reference
     parity): a test drives the reduction kernels through
     `GlobalVertexManager::Impl`, which is a public nested class with public device
     methods, injected with synthetic buffers and no engine at all;
   * the reference's dump set is `positions` + `prev_positions` only; every other
     buffer is re-established by init/reporters, so a dump/recover test must assert
     "recovered" for those two and "fresh scene value" for the rest;
   * `GlobalVertexManager` has no velocity buffer at all (velocities live in the
     dynamics systems), so a vertex-attribute round-trip test covers them.

### 7.14 Patterns established in T4.2 (implicit geometry, animator, external force, joint dof, flags)
src/backends/luisa/{implicit_geometry,animator,external_force,joint_dof_system,time_integrator,pipeline}/*
port the reference's half plane + its two reporters, animator/global_animator/utils.h,
global_external_force_manager/external_force_reporter, global_joint_dof_manager/joint_dof_reporter,
time_integrator + time_integrator_manager + bdf1/bdf2_flag and ipc/al_ipc_pipeline_flag, plus the
engine shell wiring. The rules below are what the port turned out to depend on.

1. **The reference's `init_scene()` order makes the half plane report 0 bodies.** The
   reference calls `GlobalBodyManager::init()` in the "Before Common Scene Initialization"
   step and `event_init_scene()` afterwards; the HalfPlane fills the instance arrays its
   reporters read from an `on_init_scene()` action, i.e. in that second step. Consequences,
   all verified (`half_plane_geometry_and_vertex_sets`, measured):
   * `HalfPlaneBodyReporter` reports `h_positions.size()` — still 0 — so the half plane's
     body slice is empty and only `body_offset()` (its would-be start) is set;
   * `HalfPlaneVertexReporter` and the half-plane device buffers *are* filled (the
     `GlobalVertexManager`/surface initialisation runs even later), so its vertices get
     `body_ids = body_offset + i`, ids that lie beyond the reported body array — the same
     values the reference produces, and nothing in the reference consumes them;
   * the half-plane body reporter's iota kernel and the `self_collision().fill(0)` call are
     therefore never dispatched for a real scene (they need `n > 0`). Keep them verbatim;
     coverage of the same convention comes from the *vertex* reporter's kernel
     (`coindices(i) = i`, `positions(i) = src_positions(i)`, `body_ids(i) = body_offset + i`),
     which is exercised.
   Do not "fix" this by reordering `init_scene()`: the ordering is the reference's and later
   subsystems (ABD/FEM reporters, whose ranges *are* filled in step 1) depend on it.

2. **The engine hooks T4.2 added, and the per-frame order they imply.** `SimEngine` now has
   `build_engine_systems()` (renamed from T4.1's `build_geometry_systems()`: it also
   `require<>`s the TimeIntegratorManager and `find<>`s the optional GlobalAnimator /
   GlobalExternalForceManager, like the reference's `build()`), `set_pipeline_type()` (at the
   top of `init_scene()`, `find<ALIPCPipelineFlag>()`/`find<IPCPipelineFlag>()`, throwing
   `SimEngineException` when neither exists), `step_animation_and_external_forces()` (verbatim:
   `external_force_manager->clear()` → `animator->step()` → `external_force_manager->step()`)
   and an `advance()` carrying the reference's frame prologue/epilogue. Measured per-frame
   order (asserted by the T4.2 test with an event log):
   `vertex reporter attributes → force_clear → frontend animation callbacks → animator do_step
   → force_step → integrator predict_dof → integrator update_state`.
   Two consequences worth knowing:
   * a reporter's per-frame attributes are one frame behind the animation: `update_attributes()`
     runs in the prologue, the animation step later in the same frame, so after frame k the
     *device* holds A(k-1) while the *scene* already holds A(k) (measured with exact keyframe
     values);
   * `SimEngineState` is set to `RebuildScene`/`PredictMotion`/`UpdateVelocity` around the
     matching phases, exactly as the reference does.

3. **The animator's substep ratio is reset, not accumulated.** `GlobalAnimator::step()` calls
   the frontend `animator().update()` first and then sets `m_substep_ratio = 0` *before* fanning
   out to the backend animators, so a backend animator observes `substep_ratio() == 0` inside
   `do_step()` (measured). The reference then recomputes it at the top of every Newton iteration
   as `min((iter + 1) / substep, 1)`; with no solver yet the shell performs the `iter = 0`
   bookkeeping once per frame, so after a frame the ratio is `min(1 / substep, 1)` — e.g. `1.0`
   with the default substep 1 and `0.25` after `scene.animator().substep(4)`. Replace that call
   with the real loop in Phase 8.

4. **The animator's `lerp` must be qualified — a flat spelling silently picks the wrong law.**
   src/backends/cuda/animator/utils.h defines
   `lerp(src, dst, alpha) { alpha = clamp(alpha, 0, 1); return src * (1 - alpha) + dst * alpha; }`
   (the clamp is what *holds* the last keyframe past the end; `T{1} - alpha` with `T` first is
   the reference's rounding). `lps::lerp` already exists (`a + (b - a) * t`, **no clamp**) and it
   is found by ADL for every `lps::LVec`/`LMat` argument; partial ordering prefers a fully
   generic `(A&&, B&&, T&&)` template over a constrained `(const LVec<T,N>&, ..., T)` one, so a
   flat `uipc::backend::luisa::lerp` loses the overload resolution and a call site compiles to
   the *unclamped* law (measured: `lerp(a_lvec, b_lvec, 2.0)` returned the extrapolated value).
   The port therefore declares it in the nested namespace `uipc::backend::luisa::animator`;
   Phase 5/6 constraint ports call `animator::lerp(...)`. It is a template over the numerics
   layer (one overload for `LVec<T,N>`, one for `LMat<T,R,C>`, one generic scalar), so it needs
   no registered buffer element type; the DSL instantiation
   (`LVec<Var<double>,3>` + `Var<double>`) is exercised by the T4.2 test's device kernel.

5. **Flag systems are scene predicates created by `SimSystemCreator` specialisations, and
   `require<Flag>()` is the shutdown idiom.** `bdf1_flag.cpp`/`bdf2_flag.cpp`/
   `ipc_pipeline_flag.cpp`/`al_ipc_pipeline_flag.cpp` and `global_animator.cpp` specialise
   `template <> class backend::SimSystemCreator<luisa::X>` (at namespace `uipc::backend`, taking
   the *base* `SimEngine&` and `dynamic_cast`ing to `luisa::SimEngine`) and return `nullptr` when
   the scene config does not select them — the world visitor *is* available at that point
   (`backend::SimEngine::do_init(internal::World&)` installs it before `build_systems()`).
   A system that must not exist for the current scene declares `require<BDF1Flag>()` (etc.) in
   its build hook: the `SimSystemException` is caught by `SimSystemCollection::build_systems()`,
   the system is invalidated and every system requiring *it* is invalidated in turn. Verified
   both ways (`half_plane_pipeline_and_flags`, `half_plane_animator_edge_cases`): with
   `integrator/type = "bdf1"` only the BDF1-gated integrator is built, and `contact/constitution
   = "al-ipc"` makes the engine report `PipelineType::AugmentedLagrangian`. The flag families are
   per-*scene*, not per-frame: a dependency's flag never flips between frames.

6. **Everything a test registers is active in every scene of the whole test binary.** All
   luisa test sources are one executable, every `REGISTER_SIM_SYSTEM` is built for every
   `World::init()`, and a reporter reports whatever its `do_report_*` walks — so a test-local
   vertex/body/surface reporter that reports "every SimplicialComplex of the scene" changes the
   *other* test files' expectations. Measured: `luisa_global_geometry_test.cpp`'s global arrays
   grew from 8 to 16 entries (its 8 plus the new reporter's 8) and 23 of its checks failed.
   Two rules follow:
   * gate test reporters on a marker attribute the test creates on its own geometries
     (`meta().create<IndexT>("test_<task>_scene", 1)`) and walk only marked geometries;
   * the manager arrays are a concatenation of the reporters' *slices* in reporter-uid order, so
     a test asserts on its own slice via the public accessors (`vertex_offset()`,
     `vertex_count()`, `body_offset()`, `body_count()`) instead of assuming it owns the array,
     and uses a subsequence search for the surface slices (the surface manager exposes no
     per-reporter offsets).

7. **`#include <sim_engine.h>` is ambiguous in a test TU.** The backend test target's include
   path contains `src/backends/none` *before* `src/backends/luisa`, so a test that includes the
   shell header by its bare name silently gets `uipc::backend::none::SimEngine` and
   `luisa::SimEngine` stays incomplete (measured: C2027 at every `PipelineType` use). Use the
   qualified `<backends/luisa/sim_engine.h>` (the `src` include dir is on the test target's
   path, see §7.13 rule 7).

8. **Ported-but-unreachable code is kept verbatim.** The half-plane body reporter's kernel
   (rule 1), the joint-dof reporter base's `update_dof_attributes()` fan-out (the reference only
   calls it from the state-restore path, never from the frame loop — the T4.2 test calls it
   explicitly and checks the re-sync) and the `fill(0, &frame_stream())` zeroing all stay as
   they are; the report lists what is therefore not exercised on the device.

Appendix — Verified LC API cheat-sheet (from T0.1)
### 7.15 Patterns established in T4.3 (kernel_cout device print, sanity checker, dytopo effects, line search)
M4 commits: T4.1 `30f07811`, T4.2 `789f84b1`, T4.3 `<the parent's `[luisa-backend] T4.3 …` commit>`.
`src/backends/luisa/{kernel_cout.*,sanity_check/*,dytopo_effect_system/*,line_search/*,energy_component_flags.*}`
port `kernel_cout.h`, `sanity_check/{backend_sanity_checker.h,half_plane_vertex_distance_check.cu}`,
`dytopo_effect_system/*` (classify info, receiver/reporter bases, the line-search reporter, the global manager),
`line_search/{line_searcher,line_search_reporter}.{h,cu}` and `energy_component_flags.{h,cpp}`. The rules below are
what the port turned out to depend on.

1. **A device-print *capture* is LC's per-stream log callback, installed once and retargeted.**
   The reference's `KernelCout::viewer()` streaming is impossible on LC (there is no device-side stream object), so
   `LC_LOG(fmt, args…)` (→ `device_log`) is the print and `luisa_device::LoggerViewer` stays a no-op handle for
   plumbing parity. The formatted text is delivered to `Stream::set_log_callback`'s callback when the stream reaches
   the print buffer's read-back, i.e. *on the next synchronize* — exactly the reference's "flushed on the next sync".
   `KernelCout::set_capture(stream, lines)` / `KernelCoutCapture` route it into a string list.
   * LC warns (`Setting CUDAStream::log_callback more than once … not thread-safe`) whenever `set_log_callback` is
     called on a stream whose callback is already set — **including the call that clears it**. The port therefore
     installs one trampoline per stream (the first capture on that stream, after `synchronize()`) and later captures
     only retarget it; `clear_capture()` sets the target to null and the trampoline then reproduces LC's own default,
     `LUISA_INFO("[DEVICE] {}", line)` (shader_print_formatter.h). Measured: 4 captures in the test originally
     produced 4 warnings, none after the change.
   * `luisa::log_info("…{}", x)` **does not compile** (fmt 10's `format_string` cannot be constructed from the literal
     through the template — C7595 "call to immediate function is not a constant expression"), and neither does it in a
     lambda. LC's own spelling is the macro `LUISA_INFO(fmt, args…)`, which pre-formats with `fmt::format(FMT_STRING(…))`
     and passes the resulting string to `log_info`. Use the macro (or `log_info` with a single already-formatted
     argument) for any LC logging added to the backend.
   * LC accepts only `{}` per argument (a width/precision specifier asserts) and renders with fmt's default
     formatter — doubles in the shortest round-trip spelling, not the reference's `printf("%f")`. The reference's
     *text* is therefore not reproducible token-for-token; the port's text is exact for a given LC_LOG call.
     Measured: `LC_LOG("int={} uint={}", v, 7u)` → `int=-7 uint=7`, `LC_LOG("double={}", 0.5)` → `double=0.5`,
     `LC_LOG("mixed {} {} {}", -3, 42u, 2.25)` → `mixed -3 42 2.25` (dispatch order preserved), and the no-capture
     fallback prints `[DEVICE] t43_no_capture 1` through the process log.
   * Kernel print is compiled out unless `UIPC_LUISA_KERNEL_PRINT=1` *in the translation unit that defines the
     kernel* (luisa_device/log.h) — a per-TU macro, which is why a test enables it at the very top of its file.

2. **A backend sanity checker is registered in the module and replaces the CPU one by uid — and it needs a live
   device *before* `do_init()`.** `sanity_check/backend_sanity_checker.h` derives from `core::ISanityChecker`, takes
   the CPU sanity module's `core::ISanityCheckContext`, and self-registers with `REGISTER_BACKEND_SANITY_CHECKER`;
   `core::SanityChecker::check()` loads `uipc_sanity_check`, builds it (which flattens the scene into
   `scene_simplicial_surface()` and fills `sanity_check/{contact_element_id,subscene_contact_element_id,geometry_id,
   object_id,dim,self_collision}`), then lets the engine insert the backend's checkers so
   `SanityCheckerCollection::insert()` replaces the entry with the same id. Consequences:
   * `msg->name()` is `uipc::demangle(typeid(*this).name())` → the ported checker reports as
     `uipc::backend::luisa::HalfPlaneVertexDistanceCheck` (uid 2), which is how a test proves *its* checker ran and
     not the CPU one;
   * a checker whose `build()` throws `BackendSanityCheckerException` is simply not inserted (the reference's
     "contact is not enabled" gate);
   * **`World::init()` runs the sanity check before `engine->init()`**, i.e. before `SimEngine::do_init()` — the
     reference CUDA backend creates its device in the engine *constructor*, so the port does the same
     (`init_device(info->config)` in `SimEngine::SimEngine`, `Runtime::init()` being idempotent). Without it every
     device-side checker throws "luisa_device::Runtime is not initialized" (measured).
   * the checker's `map<Vector2i, Vector2i>` uses a *named* comparator (`Vec2iLess`) instead of specialising
     `std::less` for the Eigen type (what the reference does is formally UB; the ordering — and therefore the reported
     pair order — is identical).

3. **`Scene::validate_config()` rejects unknown config *groups* — a test-local system knob must live elsewhere.**
   `config["test_t43"]["mode"] = "fast"` throws `Assertion entries.contains(name) failed. Unknown mutable scene config
   attribute 'test_t43/mode' … Custom metadata belongs on geometry, not Scene::config()` in `World::init` (measured).
   Keys *inside* a known group (e.g. `contact/d_hat_relative`, `line_search/max_iter`) are fine. A test that needs to
   steer its own registered systems therefore uses a translation-unit-local variable (T4.3's `t43_dytopo::g_mode`),
   not the scene config.

4. **The dytopo manager's contract, and the two paths a receiver sees.** `GlobalDyTopoEffectManager` fans out to the
   `DyTopoEffectReporter`s (each declares `component_flags()` — which must be a *single* valid bit: `Contact` or
   `Complement`; `EnergyComponentFlags::All` fails the manager's `is_valid_flag` assert and is only for
   `ComputeDyTopoEffectInfo`), scans the reported extents, hands each reporter a `subview` of the collected gradient /
   hessian to fill, converts collected → sorted BCOO (duplicates summed, sorted by index / by `(row<<32)|col`) and then
   classifies into each receiver's reported ranges:
   * a **single diagonal receiver** whose range is exactly the leading dynamic prefix (`x == 0`, `y != 0`,
     `y <= global_vertex_manager->positions().size()`) gets the *collected* arrays, unconverted
     (`_convert_matrix()` is skipped) — the reference still *sizes* the sorted containers in `_assemble()`, so only the
     data (raw order, duplicates kept) distinguishes the paths;
   * otherwise the gradient is the contiguous partition of the *sorted* doublets whose index lies in the receiver's
     `gradient_i_range()`, and the hessian is the sorted selection of triplets with `i` and `j` inside its ranges
     (an off-diagonal `range(L, R)` leaves the gradient view empty). A diagonal range that does not start at 0 takes
     the distributed path.
   Port-specific deviations: `DeviceVar<Vector2i> gradient_range` is a flat two-element `IndexT` buffer (k1 writes x
   and y from *different* lanes, which one vector-element write cannot express) cleared with a two-element copy *on the
   frame stream*; and `selected_hessian[N]` (the exclusive scan's "last" slot) is cleared by thread 0 with the select
   kernel dispatched over `max(N, 1)` threads, so `N == 0` clears it too (the reference leaves it untouched and only
   happens to get a zeroed fresh allocation).

5. **dytopo/line-search type mapping** (all already ported): `DeviceTripletMatrix/DoubletVector/BCOOMatrix/BCOOVector`
   → T3.1 `lla::L*`, `CTripletMatrixView/CDoubletVectorView` → `lla::LCTripletMatrixView<T,M,N>` /
   `lla::LCDoubletVectorView<T,N>` (the *host* flavour — a reporter's `do_assemble` is host code that writes the
   slices with `row_indices()/col_indices()/values()` `copy_from`s), `MatrixConverter<Float,3>` → T3.3
   `lla::MatrixConverter` (its `convert(from, to, stream)` overload keeps the conversion on the frame stream),
   `DeviceScan().ExclusiveSum` → `lla::exclusive_scan`, `DeviceReduce().Sum` → `lla::device_reduce_sum`,
   `cuda_tool::VarView<Float>` (a one-scalar view) → a one-element `LBufferView<Float>` subview. Inside a kernel the
   views are rebuilt from the raw row/col/value buffers with `from_handles` — never captured (cookbook §7.9 rule 2).
   `GlobalDyTopoEffectManager` is `find<>`-able (a `SimSystemCreator` specialisation creates it only when
   `contact/enable` is set or the scene has an `InterPrimitive` constitution); `LineSearcher` is `require<>`d and is
   always registered. Both are initialised from `SimEngine::init_scene()` in the reference's order
   (vertex → surface → dytopo → contact → animator → external force → active set → line searcher → linear system →
   time integrator), and `LineSearcher::init()` reads `line_search/{report_energy,max_iter}` + `dt` from the scene
   config (the shell exposes only `max_iter()` until Phase 8 drives `record_start_point/step_forward/compute_energy`,
   which are private with `friend class SimEngine`, exactly like the reference).

6. **`lla` is a namespace alias inside `luisa_device`, so a backend *header* must re-declare it.** `sim_engine.h`,
   the ported managers and the test files all use `lla::…`; inside `uipc::backend::luisa` that only resolves if the
   file (or an included header) declares `namespace lla = luisa_device::linalg;`. `using lla = luisa_device::linalg;`
   is a *type* alias declaration and fails with C2061 "unrecognised linalg" — write `namespace lla = …`.

7. **`include <backends/luisa/sim_engine.h>`, not `<sim_engine.h>`, in any TU that may borrow the test target's
   include path** (rule 7 of §7.14 restated) — the single-TU checker `scripts/luisa_check_compile.py` borrows exactly
   those flags, so a bare `<sim_engine.h>` makes the fast inner loop compile the *none* backend's shell and report
   "SimEngine is incomplete"/`find<>` failures that the real module build never sees.

8. **What T4.3 did *not* port, and why** (kept in `sanity_check/backend_sanity_checker.h` and in the T4.3 row):
   `simplicial_surface_distance_check.cu` / `simplicial_surface_intersection_check.cu` need
   `InfoStacklessBVH::build()/query()` (T7.1); `distance_system/distance_diagnoser.*` needs
   `contact_system/contact_models/codim_ipc_contact_function.h` (+ `sym/codim_ipc_contact.inl`) and
   `utils/friction_utils.h` (T8.1), plus `utils/codim_thickness.h` / `utils/primitive_d_hat.h` whose first consumer is
   that diagnoser. They must not be stubbed.

### 7.16 Patterns established in T5.1 (FEM base & utils)

1. **DSL-callable utility headers are header-only with a *deduced* scalar type.** `finite_element/fem_utils.h` and
   `finite_element/matrix_utils.h` carry their bodies inline (no `.cu`/`.cpp`) because a DSL call needs the body at
   the call site to emit IR. Both are `template<class T>` / `template<class Scalar>` with `auto` returns, so
   `fem::dJdF<Float>(...)`-style reference call sites keep working while the same code instantiates for `double`
   (host) and `Var<float>`/`Expr` (device). Deviation from the reference's `.h`+`.cu` split, reported in the header
   comments. Two signatures *had* to be loosened to `is_vec_like_v<V> && vec_dim_v<V> == 3` (`project_F_gradient`,
   `project_F_hessian_block`) so that Eigen's `shape_gradients.col(i)` (`VecMap`) binds.
2. **Re-export `lps` math through `using`-declarations, never wrapper templates.**
   `using lps::svd; using lps::qr_svd; using lps::polar_decomposition; using lps::evd;` inside
   `uipc::backend::luisa`. A wrapper `template<class...A> auto svd(A&&...)` is *ambiguously* overload-resolved
   (measured **C2668**) once ADL also finds `lps::svd` for an `LMat` argument — the same trap as the `lla`/`detail`
   aliases of §7.10 rule 4.
3. **`Var<T>` arithmetic rules (restated for FEM):** no `operator+=`, no copy-assignment, `lps::stored()` is
   scalar-only (never call it on a container — `storage_of` maps non-arithmetic to `Var<…>`). Write every scalar
   expression through `lps::detail::{add,sub,mul,div,neg}` (alias `namespace detail = lps::detail;`) and let the
   LVec/LMat operators come from ADL. Ordering inside a header-only translation unit is now significant: `Ds`
   must be **defined before** `Dm_inv` (in the reference both are only declared in the `.h`, so order is free).
4. **`LDeviceBuffer` semantics you must not assume away.**
   (a) `resize()` zero-fills the tail, so **value-initialise the host staging array** (`std::vector<T> v(n, T{})`)
   before `copy_from`, exactly like `cuda_tool`'s alloc+fill.
   (b) **Copy-assignment shares the allocation, it does not copy**: `x_temps = xs;` aliases the same buffer, whereas
   `cuda_tool::DeviceBuffer` deep-copies. Every reference idiom of that shape must be spelled as an explicit
   deep copy (here: the `copy_device(dst, src)` helper) — the `dump_recover` test only reached bitwise 187/187
   after this fix.
   (c) `LBufferView::copy_from/copy_to` default to the **main stream** and are no-ops on empty views.
5. **`backend::BufferView::handle()` is a *host* address.** Documented deviation: the reference reinterprets it as a
   raw device pointer and does a D2D copy; LC exposes no raw device pointer and the pybind frontend hands host
   buffers, so `copy_to_buffer_view` does D2H + `stream << synchronize()` + element-wise `LVector3` conversion.
6. **Kernel asserts.** `UIPC_KERNEL_ASSERT` → `lcrt::device_assert(cond, "literal")`; LC cannot format device
   indices, so the tet-index part of the message is dropped. Placement is preserved verbatim, which matters:
   in the rest-volume kernel the check sits **after** the closed-form inverse (a zero/negative-volume tet therefore
   still computes `Dm_inv` first), and the shell rest-area kernel has **no** assert at all (zero area stays `0`).
7. **`UIPC_ASSERT` in the backend aborts, it does not throw** (`utils/report_extent_check.h` → logger + `uipc::abort()`).
   A Catch2 test can therefore never exercise the two invalid `check_report_extent` combinations; the test pins only
   the legal ones and the assertions are documented in the header.
8. **Interfaces owned by later tasks still have to *link* in T5.1.** Three sub-rules learned here:
   (a) a member whose **parameter** type is a nested class of a class that does not exist yet
   (`FEMLinearSubsystem::AssembleInfo`, `FEMLineSearchReporter::ComputeEnergyInfo`) cannot even be declared →
   it is deferred (elastics' `assemble`/`compute_energy`), because a nested type needs its enclosing class complete;
   (b) the method `dynamic_cast`s to the four dimension bases, so their final overrides must be **defined** now even
   though the concrete constitutions arrive in T5.2 (a concrete class's vtable references them);
   (c) `FEMLinearSubsystem::add_reporter` and `ReportExtentInfo::{gradient_count,hessian_count,check}` are defined in
   `fem_linear_subsystem_reporter.cpp` so the T5.1 reporter base links; T5.4 relocates them next to the subsystem they
   belong to (said so in-file).
9. **Keep the reference `friend` lists verbatim** rather than widening accessors — `Impl` state is reached only
   through the friends the reference already declares, and every later task port is then a mechanical edit.
10. **MSVC warning hygiene over LC buffers (reference-parity, not semantic change).** `std::iota`/`fill`/`transform`
    on a `span<T>` of an LC buffer raise **C4267** (`size_t`→`IndexT`) → put `static_cast<IndexT>(i)` in the lambda;
    `std::exclusive_scan` with a `0` init raises **C4244** → use `SizeT{0}`. Goal: module build with 0 warnings.
11. **Verifying a "clean rebuild" of the backend module needs a cmake re-run in between.** The module sets
    `WINDOWS_EXPORT_ALL_SYMBOLS`, so CMake writes the object-list file
    `build/src/backends/luisa/CMakeFiles/luisa.dir/exports.def.objs` at *generate* time. Deleting the `.dir` tree
    (the usual way to force a full recompile) therefore leaves the link step failing with
    `could not open object list file: …\luisa.dir\.\exports.def.objs`. The sequence that works is
    `rm -rf build/src/backends/luisa/CMakeFiles/luisa.dir build/apps/tests/backends/luisa/CMakeFiles/backend_luisa.dir`
    → `cmake -S . -B build` → `python scripts/luisa_build.py build backend_luisa` (measured: 81 CXX compile steps,
    rc 0, 0 warnings, ~85 s; keep the log — `scratch/t51_cleanbuild.log`).
12. **Test-side patterns (reusable by T5.2–T5.5).**
    (a) Keep hand-derived oracles in Eigen notation: a small test namespace of shims (`lv`, `Ds`, `Dm_inv`, `F`,
    `dFdx`, `tetrahedron_shape_gradients`) converts `Eigen::Matrix`→`lps::LMat`/`LVec`, so the oracle reads like the
    paper formula while the *ported* utils are still what runs.
    (b) One generic dumper beats N ad-hoc checks: `fem_util_round<S, Put>` serialises **187 doubles** (v9 9, ddot 1,
    inv2F 1, inv3F 1, Σ 3, inv2σ 1, inv3σ 1, inv4 1, inv5 1, dFdx 108, dJdF 9, shape grads 12, project grad 3,
    project Hessian block 9, Dm_inv 9, Ds 9, F 9) from any scalar backend; run it on host `double` and on device
    `Float` and compare **bitwise** → "187/187 bitwise equal, worst abs 0.000e+00" is a far stronger claim than a
    per-function epsilon. The device side is one `cached_kernel<Buffer<double>>` writing each entry with
    host-constant indices (traps 1/4 of §7.10 still apply: the compiled kernel and the result buffer are leaked).
    (c) Record the tolerances actually achieved so later tasks stop over-tightening: h=1e-6 FD gradient worst abs
    **1.069e-09** / rel **9.064e-10**; FD Hessian worst abs **6.685e-10** / rel **5.293e-10**; Hessian asymmetry
    **2.220e-16**; min eigenvalue **-2.980e-16** at λmax **9.437e+00**; `dFdx` vs FD rel **8.227e-11**; `dJdF` vs FD
    rel **6.884e-11**. Assert order-of-magnitude bounds, never exact values.
    (d) `type_define.h` keeps growing but generically: a `LMat` used by value (kernel arg, struct member) fails with
    **C2027** when only `<luisa_device/math/vec.h>` is included (it merely forward-declares `LMat`) → include
    `<luisa_device/math/mat.h>`. Any new element type needs `sizeof`/`alignof` static_asserts, `to_luisa`/`to_eigen`
    and a **global-scope** `LUISA_STRUCT(…)` registration, otherwise `buffer.h:85`'s `is_valid_buffer_element_v`
    static_assert stops the build (added here: `LVector4i`, `LMatrix3x3`).
13. **What T5.1 did *not* port, and why** (recorded in each file's scope note and in the T5.1 row): concrete
    constitutions + `sym/*.inl` (T5.2), shell bending/mass (T5.2/T5.3), `FiniteElementElastics::Impl::assemble` /
    `compute_energy` and its two nested reporter systems (T5.4/T5.5), the `FEMLinearSubsystem` system class and
    `ReportExtentInfo::Impl` (T5.4), the concrete extra constitutions (T5.5), and
    `finite_element_diff_parm_reporter` (needs `<diff_sim/…>`, not ported at all). None of them may be stubbed:
      a stub would silently turn a missing-feature build error into a runtime no-op.

### 7.17 Patterns established in T5.2 (FEM constitutions)

M5 batch **a** (3D volumetric) ports `arap_3d` and `stable_neo_hookean_3d` (their `*_function.h`,
`detail/*.inl` and one `sym/*.inl`) into `src/backends/luisa/finite_element/constitutions/`, plus
the shared `utils/matrix_assembler.h`. The rules below are what the port turned out to depend on.

1. **A *subview* passed as a kernel argument is bound at `base + offset_bytes`.** A constitution
   receives its slice of the method's buffers (`BaseInfo::indices/Dm_invs/rest_volumes`), so the
   per-element index arithmetic stays identical to the reference's. Verified in LC's CUDA backend:
   `CUDACommandEncoder` encodes a buffer argument with `CUDABuffer::binding(arg.offset, arg.size)`
   and that returns `Binding{device_address() + offset, size}` — i.e. index 0 *is* the subview's first
   element (`size_bytes` is the subview's, so `lc_buffer_size` and the debug bounds check follow).
2. **A doublet/triplet *view* cannot be a kernel argument** (§7.9 rule 2), so the port passes its flat
   buffers *pre-offset to the constitution's slice* and rebuilds the device view inside the kernel with
   `from_handles_full`. Two pieces of state cannot travel inside a buffer handle and are passed as
   scalars: the dof-vector `subvector_offset` (doublet) resp. `submatrix_offset.{x,y}` (triplet); the
   remaining `from_handles_full` arguments are host-assert material on the device path.
3. **`utils/matrix_assembler.h` is now shared by the constitution batches.** Ported as
   `DoubletVectorAssembler<View>` / `TripletMatrixAssembler<View>` (generic over the *view* type, so one
   implementation serves `storage::device`, `storage::buffers` and `storage::memory`). Three
   deviations, all measured: (a) `upper_LR` cannot branch on a device predicate (`l_indices(I) +
   offset.x < r_indices(J) + offset.y`), so both candidate blocks are evaluated with *host constant*
   indices and the block and the two global indices are `select`ed — a select passes its chosen operand
   through untouched, so the written `(row, col, block)` triples are exactly the reference's; (b) the
   reference's `offset++` becomes `m_I + counter` (a device index type has no compound assignment), and
   the assembler's index argument goes through `Storage::as_index`, so a kernel may pass `dispatch_x()`;
   (c) every `UIPC_KERNEL_ASSERT` is kept in place — `lcrt::device_assert` for the device flavour,
   `lla::detail::kernel_assert` (a host throw) for the others. The reference's `zero_out` and the
   `ignore`-flavoured `write` overloads are *not* ported (no 3D constitution uses them; their bodies
   branch on device data) — they arrive with their first consumer.
4. **A container `select` needs its own helper, and the scalar one has to be *dispatched*.** `lps::select`
   is host-scalar-only on purpose (§7.6 trap 3) and LC's `select` is only reachable by ADL, which a
   *nested* `ProxyRange` class does not get from a `using`-declaration in an enclosing namespace
   (measured C3861) — and a *qualified* call never gets ADL at all (measured C2672 in the first version).
   The port therefore dispatches explicitly:
   `if constexpr(is_dsl_value_v<…>) return lcrt::select(f, t, p); else return lps::select(f, t, p);`
   and selects containers element-wise through `make_mat`/`make_vec` (LC's own container `select`
   overload would silently build a `Var<LVec<…>>`).
5. **The reference's kernels are factored into generic per-element routines that the unit test
   instantiates too.** `arap_3d_detail` / `snh_3d_detail` (`tet_energy`, `tet_gradient`, `tet_hessian`)
   live in the `*_function.h` header, are templated on the scalar (`double` / `lcrt::Var<double>`), and
   are exactly the reference's kernel bodies minus the buffer plumbing — so the FD oracle, the
   device-vs-host comparison and the shipped kernel all run the *same* code (cookbook §7.4).
6. **`if(gradient_only) return;` becomes a predicate, not a `return`.** The reference's kernels return
   early on a *device* flag; a `return` inside `$if` is not a DSL operation, so the Hessian stage is
   wrapped in `$if(gradient_only == 0u) { … }` (RULE M2). The test asserts the predicated behaviour: the
   doublet rows/values are written, the triplet buffers keep their sentinels.
7. **The transcribed `.inl` files are generated by `scratch/gen_fem_cons_a.py`** (an idempotent re-run
   of T2.3's generator, with `--check` *and* `--selfcheck`). New in this generator: a *container-aware*
   emitter — an operand is a container if the signature/declarations say so, and container/scalar mixes
   keep the numerics layer's own operators (`mul(mu, …) * F`) while scalar arithmetic goes through the
   `lps::detail::{add,sub,mul,div,neg}` trampolines — plus `std::pow` → `lps::pow` (identical to
   `std::pow` on the host, `lcrt::pow` on the DSL) and *integer literals → double literals* (value
   identical here, and it keeps LC's `[[deprecated]]` fp/integral implicit-conversion helper out of the
   DSL instantiation, §7.16 rule 10). `--selfcheck` re-parses every generated statement and proves it is
   a faithful re-spelling (tree identity after un-transforming) and value-exact under 200 random inputs.
   Hand-written parts follow the reference literally: an Eigen comma initialiser becomes one element
   assignment per entry in Eigen's **row-major traversal order**, and
   `Eigen::Map<M3>(Q.data() + k) = M;` is a *flat* nine-element column-major write
   (`Q.data[k + i] = M.data[i]`), not a `block<3,3>`.
8. **`ARAP_Hessian` is the *unprojected true* Hessian, and the projection is `make_spd`.** Both
   constitutions are exact: the dof-FD comparison (h = 1e-6, 40 randomized admissible states) reaches
   grad rel **8.8e-11** / Hessian rel **1.2e-10** (ARAP) and **2.1e-10** / **2.9e-10** (SNK1) — but *only
   in the states where the local Hessian is already PSD*; both constitutions deliberately replace an
   indefinite local Hessian by its eigenvalue-clamped version (`make_spd` 9x9 for ARAP, Stiff's analytic
   `ddEddF_spd` for SNK1). A fidelity test must therefore (a) FD-compare in the PSD states and (b) in the
   indefinite ones prove the *eigenvalue-clamp identity* (`eig(analytic) == max(eig(true), 0)`), which the
   ported `make_spd` satisfies to **2.3e-10** and the SNK1 eigensystem to **1.3e-09**. Comparing the clamped spectrum against the unclamped FD one
   *including the 12-dof element space* is wrong: clamping happens in the 9x9 F-space, and
   `dFdx^T H dFdx` does not commute with it.
9. **Degenerate states, pinned from the source** (no input check exists in either constitution; the only
   volume check is T5.1's `Dm^-1` kernel): `ARAP_Hessian` divides by `sigma_i + sigma_j`, so a
   zero-volume state (`F = 0`) and a *reflection* (`sigma_1 + sigma_2 = 0`, e.g. `F = -I`) give a
   non-finite 9x9 Hessian while `E` and `dE/dF` stay finite (so the element Hessian is non-finite too —
   the port does not "repair" it); a singular-but-signed state (`diag(1,1,0)`) stays finite. SNK1 and the
   SMT sibling are finite everywhere, and SNK1's clamped Hessian stays PSD even for a collapsed element.
10. **A real bug in `luisa_device/math/evd.h` (T2.2) found here and fixed minimally.** `jacobi_rotate`
   used `t = sign(tau) / (|tau| + sqrt(1 + tau²))` with the layer's `sign(0) == 0`, so the rotation
   *vanished* exactly when `M(p,p) == M(q,q)` with `M(p,q) != 0` — the classical formula needs
   `sign(0) := +1` (a 45° rotation, `t = 1`). The symptom is not a small error: the sweep sequence
   leaves the off-diagonals alone and the "eigenvalues" are the diagonal, which made SNK1's 3x3 stretch
   block at an isotropic stretch (`F = s I`, whose `A` has equal diagonal entries) return 29.26 three
   times instead of `(-17.36, -17.36, 122.51)`. One line (`select(one_of<T>(), make_scalar<T>(-1),
   tau < zero_of<T>())`), no change for any other input; T2.2's `evd`/`make_spd`/`svd`/`pd` tests still
   pass unchanged. **Any ported routine that calls `evd` was exposed** (here: `make_spd`, the SNK1
   eigensystem).
11. **The class declaration is header-separated from the kernels (reported deviation).** The reference
   puts the constitution class, its kernels and `REGISTER_SIM_SYSTEM` in one `.cu`; the port keeps the
   kernels + out-of-line member definitions + registration in `<name>.cpp` but declares the class in
   `<name>.h`, so the unit test can reach the reference's *public* surface (the reference declares the
   five hooks and the parameter buffers in the class's `public:` section) and drive
   `do_compute_energy`/`do_compute_gradient_hessian` on the real method buffers. Without it, validation
   item "the constitution's build/init/report_extent/assemble/energy path end to end" is unreachable:
   `FiniteElementElastics::Impl::assemble` (the fan-out that calls those hooks) is a T5.3/T5.4
   deliverable and its parameter types do not even exist yet.
12. **A test-binary uid collision has to be *detected*, not assumed away.** `luisa_fem_base_test.cpp`
   registers a *test* dim-3 constitution whose uid is ARAP's (9), and
   `FiniteElementMethod::Impl::_build_base_constitution_infos` hands a `(dim, uid)` geometry group to
   the **first** system registered with that uid (the per-dim index is per *system*, the geometry group
   per *uid*). The T5.2 scene test therefore checks which of the two owns the group and reports it
   (StableNeoHookean's uid 10 is unique and carries the strong end-to-end checks).
13. **The cost of a *faithful* ARAP kernel is the JIT, not the GPU.** Inlining `make_spd`
   (`evd` = 20 cyclic sweeps × 36 rotations on a 9x9) makes LC's nvrtc compile the ARAP
   gradient/Hessian shader in **~13–16 min** on the first dispatch (measured: 962 s for one 16-element
   batch), while the SNK1 shader (3x3 `qr_svd` + 3x3 `evd` + 16 projections) compiles in **~10 s**.
   The energy kernels are cheap (no `evd`). If that ever matters, the lever is `make_spd`'s sweep
   count (`lps::make_spd_inplace(H, sweeps)`), which is T2.2's documented fidelity choice — not
   something T5.2 changed silently.
  14. **The batch-b generator *imports* batch a's transformer; it does not re-implement it.**
     `scratch/gen_fem_cons_b.py` adds `sym/discrete_shell_bending.inl` and
     `sym/neo_hookean_shell_2d.inl` to the same tokenizer/parser/emitter/`--selfcheck` machinery
     (`import gen_fem_cons_a as g`, reusing `g.DST_DIR`/`g.REF_DIR`/`g.rewrite`), so the two batches
     cannot drift apart. `--check --selfcheck` is green for both regions (3 + 763 statements). One
     *generator* fix was needed: `--selfcheck`'s value comparison draws each free variable uniformly
     from (0.5, 2.0), and the shell energy's `x29 = log((det IB) * (a signed determinant expression))`
     legitimately leaves the logarithm's domain there — both sides evaluate the *same* function, so the
     script installs a domain-safe `math.log` (one sentinel for `x <= 0`) instead of raising
     `ValueError`. The tree-identity check (the property transcription relies on) is unaffected.
  15. **`cross` is a free function in the numerics layer.** Eigen spells it `a.cross(b)`; `lps` has
     `lps::cross(a, b)` and *no* `LVec::cross` member (`dot`/`norm`/`squaredNorm`/`normalized` are
     members on both sides). Every transcription of the reference's dihedral-angle utility and of
     Baraff-Witkin's `Dm2x2` must re-spell it (otherwise measured C2039).
  16. **`uipc::Vector`/`uipc::Matrix` are `using Eigen::Vector/Matrix`** — a generic test-side shim
     therefore needs `Vector<Float, N>` / `Matrix<Float, R, C>` (three template arguments), never
     `Vector<N>`. Related MSVC parser trap in a *dependent* context: `span<const Dst>{host}` (a
     using-declared class template) is **not** parsed as a type-id — measured C2760/C3878
     "unexpected '{'; expected ')'" — and the parenthesised form `span<const Dst>(host)` is required.
     Non-dependent uses (`span<const Float>{h_kappas}`) are fine, which is why batch a never hit it.
  17. **`utils/scalar_util.h` is the "possibly-DSL scalar" toolkit batch b needed.** Three primitives
     the shell constitutions cannot express with a single layer call: `select_scalar(f, t, p)` (the same
     dispatcher `matrix_assembler.h` carries privately — `lps::select` is host-only by design and LC's
     `select` is ADL-only, so a *qualified* call never finds it), `pred_and`/`pred_or` (LC needs `&`/`|`
     for DSL scalars, the host needs `&&`/`||` to avoid `bool & bool -> int`), and
     `is_finite_scalar`/`is_finite_vec3` (`!(isnan | isinf)`: LC has no DSL `isfinite`, and the
     `x - x == 0` trick would rest on fast-math assumptions), plus `select_container` for element-wise
     LVec/LMat selects. Batch c can reuse all of it.
  18. **`select(f, t, p)` is `p ? t : f` — a reversed argument pair is a *silent* behaviour change.**
     One debugging round in batch b: `select_container(rotation_singular, rotation_general, singular)`
     returned the *singular* matrix for every non-singular input, i.e. the Baraff-Witkin rest metric came
     out as a fixed 180-degree rotation instead of the isometric projection. The "canonical triangle
     gives `IB == I`" and "arbitrary rest triangles give `F^T F == I`" invariants caught it immediately,
     while a gradient/Hessian-only test would have "passed" against a consistently wrong model.
  19. **The reference's `bool`-returning guard chains become `select`s, not early returns.** Both plastic
     constitutions guard every per-element routine (`safe_dihedral_angle`, `try_angle_delta`,
     `try_trial_state`, `augmented_response_from_angle_delta`, `update_plastic_state`) and *return* on
     failure. A device predicate cannot drive host control flow (RULE M2), so the port evaluates the whole
     computation (all of it NaN-safe: the cosine is clamped before `acos`), folds the guards into one
     predicate in the reference's own order and selects where the reference returns — `return 0.0` ->
     `select(0.0, value, ok)`, `G.setZero()` -> a container select, and for the out-parameter routines the
     outputs are selected back to their input value, so the caller's write becomes an identity.
     Equivalence argument: a select passes its chosen operand through untouched.
  20. **The plastic history is a *return mapping*, and it needs its own test.** `update_plastic_state`
     does `theta_bar += sign(delta) * (|delta| - yield)` — the new `theta_bar` is *not* `theta` (the
     first hand-computed sequence written for this batch was wrong for exactly that reason, and the
     oracle caught it). Two gates matter: `plastic_excess <= 1e-6`
     (`plasticity_write_threshold`) and, for the stress-plastic variant, the whole `try_trial_state`
     guard chain (`L0 <= 0`, `h_bar <= 0`, `yield_stress < 0`, non-finite inputs). The port's kernel
     writes both attributes unconditionally (`update_plastic_state` already returns the input value when
     its guard fails), which is the port's formulation of the reference's conditional write: an unmoved
     history is not a change. The unit test drives the reference's sequence step by step (host, with
     literal expectations including a "yield not met" step) *and* the constitution's own device buffers
     through the shipped generic routine (a test kernel with the same body as the TimeIntegrator's).
  21. **Three fidelity notes worth pinning rather than "fixing".**
     (a) The reference's `while(angle > Pi) angle -= TwoPi;` loops are device loops with data-dependent
     trip counts; the port uses a fixed 5-step predicated form (bit-identical to the loops for
     |a| <= 11 pi, asserted over 4000 random angles in ±11 pi) and every call site is provably inside
     |a| <= 3 pi.
     (b) `lps::min/max` compare as `x < y ? x : y` while `std::min/std::max` keep their *first* argument
     on a NaN comparison, so a fully degenerate stencil's `theta_bar` is `pi` (port) vs `0` (reference):
     both finite, and the energy is non-finite either way (`h_bar` is 0/0). Recorded in the
     degenerate-state case.
     (c) The extra constitutions' `ComputeEnergyInfo`/`ComputeGradientHessianInfo` cannot be reached from
     a test through the base (the hooks are `protected` there *and* the info types need the elastics'
     `Impl`); the batch-b scene test builds a test-local `FiniteElementElastics::Impl`
     (`constitution_*_offsets_counts` sized + `scan()`) plus a test-local
     `FiniteElementExtraConstitution::Impl` pointing at the *real* method, and calls the hooks through the
     *concrete* type (public there, like the reference's class layout). The dim-2 constitutions'
     `Compute*Info` constructors are public, so they need no such trick — but their hooks still have to be
     called through the concrete type (the base declares them `protected`).
  22. **The Baraff-Witkin Hessian is a *stabilised* Hessian, not the exact derivative.** Measured with an
     independent Eigen re-transcription *and* with central differences: `ddEddF` matches `d(dEdF)/dx` in
     the regime the model was derived for (`I6 == 0` with both anisotropic invariants >= 1) — rel 3.6e-10 —
     but deviates by 0.07–1.5 x the Hessian's own scale in sheared states and in the shrunk regime, where
     the *true* Hessian is indefinite (`I5 < 1` is the concave branch) and the reference keeps a PSD
     surrogate. The port is therefore pinned against the reference formula (independent Eigen
     transcription, 2.3e-13) rather than against FD, and the FD deviation is asserted only in the model's
     own regime. Consequence for later tasks: do not tighten a Baraff-Witkin gradient/Hessian FD bound
     below ~1e-2 in a general state.
  23. **`-use_fast_math` shows up as a device-vs-host bound, and only through `acos`.** LC compiles its
     CUDA shaders with `-use_fast_math`; the DSL's `+ - * / sqrt` on `double` stay bit-accurate (Baraff-
     Witkin's energy/gradient/Hessian agree with the host instantiation to 1.4e-13), but `acos` carries
     ~1e-8. Everything that turns the dihedral angle into a physical quantity inherits that: the
     discrete-shells energy agrees with the host to 7.7e-07 and the plastic *history* (a difference of two
     O(1) angles) to ~2e-7 on the device, while the *host* instantiation of the same generic routine is
     **bitwise** equal to the independent oracle (measured 0.0). Assert the device bound per routine and
     the host path bitwise.
24. **A stationary state needs its own FD step -- and the step has a *measured* optimum.** At the rest
state the bending gradient is `2 L0 kappa (theta - theta_bar) / h_bar`, a cancelling difference of two
O(1) angles, so an h = 1e-6 central-difference Hessian is pure round-off (its smallest eigenvalue goes
negative by 1e-4, measured -2.8e-04). A sweep over 40 random rest stencils (`kFdRestH` in
`luisa_fem_constitution_b_test.cpp`, |H| ~ 1.9e1) measures `|dH| = 7.3e-2 (h = 1e-6), 8.6e-6 (1e-4),
4.4e-5 (1e-3), 4.4e-3 (1e-2)`: the round-off falls as `1/h^2` while the truncation rises as `h^2`, so
1e-4 is the optimum. Compare it *relative to the analytic Hessian's own scale* and assert the
*analytic* Hessian's PSD-ness, not the FD one's.
25. **`triangle_thickness(t0, t1, t2)` returns `t0`; it is not an average -- and every membrane energy is
proportional to the thickness.** The reference's `utils/codim_thickness.h` helper returns the *first*
vertex's thickness, and its kernel asserts that the three agree (the frontend writes the thickness
attribute uniformly). An oracle that averages with `0.5 * (t0 + t1 + t2)` is therefore 1.5x the
thickness, i.e. 1.5x *every* energy, gradient and Hessian of a codim-2d constitution -- while
Baraff-Witkin's energy ignores the thickness buffer, so that constitution stays green and hides the
oracle bug. Read the reference's reduction, and assert the equality its kernel asserts.
26. **A *constant* ratio between the device and the oracle is never rounding: look for a multiplicative
*input*.** Measured symptom: the device's Neo-Hookean density was exactly 2/3 of the host's for all six
scene elements, while the state-dependent part of each agreed to 1e-7. Rounding cannot produce 2/3 of a
value in every state; only a factor every term is homogeneous in (`lambda`, `mu`, `rest_area`,
`thickness`, `dt` for a membrane) can. In this case it was the oracle's thickness (rule 25: `0.5 * 3 =
1.5` instead of `t0`) -- i.e. the *device* was right and the independently written oracle was wrong,
which is why the symptom had to be chased to the end instead of being "fixed" in the port.
27. **Split a mismatch with a *density* probe before blaming the formula.** `element_energy` is
`density * Vdt2`. A tiny kernel that calls the generated density (`sym::...::E`) and writes it per
element separates "the generated code is wrong" from "an argument/scaling is wrong" in a single run
(the scene's temporary `nh_density_kernel` did exactly that and localised the 2/3). Its permanent
version lives in the `device_vs_host` case (`nh_element_energy_kernel`): device vs host 1.9e-9 for the
generated Neo-Hookean density -- including states translated far from the origin, because the generated
expansion computes `det(A)` from *raw* coordinates, so a translated stencil exercises its cancellation.
28. **A device-vs-host bound belongs to the routine's *transcendentals*, not to the module.** Measured on
this port: Baraff-Witkin (no `acos`) 3.1e-15/3.6e-15/1.4e-13 (E/G/H), the discrete-shells *energy* (one
`acos`) 1.0e-6, the plastic *history* (a difference of two `acos`) 2.2e-7, the generated Neo-Hookean
density (one `log` of a 2x2 determinant) 1.9e-9. Do not carry one number across routines: assert ~1e-5
where `acos` is involved and ~1e-9 where only `log`/`sqrt` are.
29. **Run the *filtered* suite and the *full* suite: the RNG stream is process-wide.** The new test's
`std::mt19937` is a file-local singleton seeded once (`20240917u`), so a case's random states depend on
how many cases ran before it. Measured consequence: the same stationary-state FD check reads worst
`4.5e-7` alone but `1.8e-5` in `[constitution_b]` and in the full run. Bounds for random-state checks
must hold in *both* streams (and the filtered run is the one that has all six cases in one process, so
it is the sharper one).

M5 batch **c** (rods/springs/particles) ports `kirchhoff_rod_bending` (uid 15, an extra constitution),
`hookean_spring_1d` (uid 12), `particle_0d` (uid 13) and the four uid-0 placeholders
`empty_0d/1d/2d/3d`, plus their two generated `sym/*.inl` files. The rules below are what that port
turned out to depend on.

30. **A hand-derived oracle can be wrong in a way a "compare with the port" test cannot see — and both
the finite difference *and* an expression-level oracle are what catch it.** Batch c's Kirchhoff-rod
energy was first derived as `k pi r^4 (1 - cos theta)/L0`; the generated `x7 * bracket` actually
collapses to `k pi r^4 (1 - cos)/((1 + cos) L0)` = `k pi r^4 tan^2(theta/2)/L0` (the algebra slip is
`(1 - cos^2)/(1 + cos)^2` mis-cancelled to `(1 - cos)` — the *device* was right). Three consequences,
all acted on:
  (a) a **right-angle** hand-computed configuration cannot see it: `cos = 0` makes `(1 - c)` and
  `(1 - c)/(1 + c)` equal, which is exactly why the first version "passed" it — the batch-c hand case
  is now a **60 degree** hinge (the two forms differ by 3/2);
  (b) the FD comparison reported it as a *constant* ~0.5 relative error, i.e. rule 26's signature seen
  from the oracle's side;
  (c) `scratch/t52c_eval.py` settles such a dispute with no GPU: it re-parses the *generated* `.inl`
  with the transcriber's own parser (`gen_fem_cons_a.parse`), emits Python from the AST and evaluates
  it in double, so a hand formula is checked against the expression itself (measured: rod E rel 4.6e-14,
  rod `dEdX` 1.1e-13, spring E 1.1e-14, spring `dEdX` 5.5e-15). It also localised a **sign** slip in the
  derived rod gradient: with `dE/dc = -2K/(1 + c)^2` and `du, dv` the `c`-gradients w.r.t. `e0, e1`,
  `dE/dx0 = -dE/dc du`, `dE/dx1 = +dE/dc (du - dv)`, `dE/dx2 = +dE/dc dv`.
  Write the FD first and keep a cheap expression-level oracle.

31. **`make_spd`'s sweep count for the rod/spring element Hessians is 8, measured.** The reference's
`utils/make_spd.h` is a tolerance-driven Jacobi loop (device data, so the port needs a host-fixed count)
whose *purpose* is the eigenvalue clamp. Measured plateau of the layer's cyclic sweep against an exact
`eigen decompose + clamp + reconstruct` (30 random symmetric matrices per N, worst relative over the
matrix's own scale): N = 6/9/12 all reach ~1e-15 at **8** sweeps, while 5 gives 1.9e-12 (N = 6), 1.1e-6
(N = 9) and 2.2e-4 (N = 12) — the same 8 batch b chose for its 12x12 shells, at ~1/3 of the layer
default's (20) JIT cost. `krb_detail::kMakeSpdSweeps` and `hs1d_detail::kMakeSpdSweeps` are both 8,
spelled locally so that neither constitution's header depends on the other's. Measured on the *actual*
element Hessians: the projected 9x9 matches the exact clamp to **3.2e-15** (rod) / **2.9e-15** (spring)
relative, and **40/40** rod / **26/40** spring randomized states are *indefinite* before the projection,
i.e. the clamp is not a no-op.

32. **Split the Hessian check: unclamped analytic vs FD in *every* state, the clamp identity
separately.** Same shape as batch a's rule 8. Measured over 40 randomized states: rod FD gradient rel
**3.7e-10**, Hessian rel **9.6e-5** at h = 1e-6 and **2.0e-8** at h = 1e-4; spring **2.5e-10** /
**2.0e-4** / **2.7e-8**; Hessian asymmetry **1.7e-16** (rod) and **1.6e-16** (spring). The brief's
1e-4 / 1e-3 bounds hold at h = 1e-6 for both; h = 1e-4 is the round-off optimum (rule 24). A straight
rod's *analytic* Hessian is PSD (min/max eigenvalue −2.9e-16) and its FD at h = 1e-4 agrees to 2.0e-7.

33. **The degenerate states of this batch, pinned from the source** (neither constitution has any input
guard — no branch, no assert, no length clamp):
  * rod, collapsed current configuration (all three positions equal) with a healthy `L0`: `x7 =
  (...)^-2 = +inf` times a bracket that is *exactly* 0 ⇒ `E = NaN` (gradient and Hessian non-finite);
  * rod, zero *rest* length: `+inf` with a bent current state, `0/0 = NaN` with a straight one;
  * rod folded back on itself (antiparallel edges): the closed form's limit is `+inf`, but the
  generated `0 * inf` gives NaN (or a value whose last bits depend on `u.v + |u||v|`) ⇒ only
  non-finiteness is asserted;
  * spring, zero *current* length: the energy is finite and *exactly* `k Vdt2 / 2` (derived:
  `0.5 k (0 - L0)^2 / L0^2 * Vdt2`), the gradient is `0 * inf = NaN`, the Hessian non-finite;
  * spring, zero *rest* length: the generated density is `+inf` but the kernel's own `Vdt2 ∝ L0` scales
  it by 0 ⇒ the *kernel* energy is `NaN` (a case where "the density alone would be +inf" is the wrong
  assertion — check the value the kernel actually returns);
  * spring, compressed (`L < L0`): the unclamped Hessian has a negative eigenvalue (−3.0 here) and the
  shipped one is PSD — the model's own reason for `make_spd`.
  The one guard that *does* exist is `utils/codim_thickness.h`'s `edge_thickness(t0, t1)` RUNTIME_CHECK
  (an abort, not a throw), so it cannot be exercised from a test.

34. **LC's default CUDA codegen evaluates `double` transcendentals in SINGLE precision.** LC
picks its CUDA codegen by environment variable (see 1.4): the default is the **AST** path, which
emits `lc_log`/`lc_exp`/`lc_acos`/... , and `cuda_builtin/cuda_device_math.h` defines those
`lc_double` overloads as the **float** functions (`logf`, `expf`, `acosf`, ..., and `powf` for a
non-integral `pow` exponent). This, not fast-math rounding, is what bounds a device-vs-host
comparison in any routine that transcribes a generated expression containing `pow`/`log`/`exp`/
trigonometry: measured in-tree, `log(0.5)` came back as `-0.69314712285995483` (8.32e-08
relative), `sin(0.5)` at 2.26e-07, and the rod/spring constitutions showed ~1e-7 per element
while the *host* instantiation of the very same routine matched the oracle to ~1e-13.
    (a) this is **fixed** by the applied local LC patch of 1.4 (`scripts/patch_lc_double_
    transcendentals.py`), which makes the AST path use the double functions; the guard is the
    `device_double_math_accuracy` case (bound 1e-13);
    (b) if you measure before/after such a change, read 1.5 first -- a stale deployed LC runtime
    or a stale PTX cache will make a real fix look like a no-op (this cost two wrong conclusions
    during T5.2c, including one that withdrew a correct diagnosis);
    (c) assert a device-vs-host bound per routine and **record the measured value**; do not
    derive it from which transcendental the routine uses, and do not "fix" a transcription by
    re-spelling it as `lps::sqrt` (the reference writes `std::pow(..., 1.0/2.0)` and the
    generator's contract is operator spelling only).
## Appendix — Verified LC API cheat-sheet (from T0.1)

```cpp
#include <luisa/luisa-compute.h>
using namespace luisa::compute;

Context ctx{argv[0]};
Device  device = ctx.create_device("cuda");
Stream  stream = device.create_stream();

auto buf = device.create_buffer<float>(N);

Kernel1D k = device.compile<1>([&](BufferVar<float> v) noexcept {
    auto i = dispatch_id().x;          // or dispatch_x()
    v->write(i, 1.0f + cast<float>(i));
    auto w = warp_active_sum(v->read(i));
    $if (warp_lane_id() == 0u) { v->atomic(i).fetch_add(w); }
    device_log("i={}", i);
});

CommandList list = CommandList::create();
list << k().dispatch(N);
stream << list.commit();
stream << buf.copy_to(span{host.data(), N}) << synchronize();
```

Notes: `Buffer<T>` in a kernel supports `read/write/volatile_*/atomic/size`;
LC's `Buffer<T>::operator()` *is not* a reference accessor — that is why
`LBufferView<T>` supplies the proxy `operator()`.
