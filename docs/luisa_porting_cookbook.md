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
### 7.18 Patterns established in T5.3 (FEM dynamics, BDF, animator, external forces)

M5 batch **T5.3** ports the FEM dynamics layer: `fem_time_integrator.{h,cpp}`, the BDF
integrators (`bdf/fem_bdf{1,2}_time_integrator`), their kinetic systems
(`bdf/finite_element_bdf{1,2}_kinetic`), `bdf/bdf2_function.h` + `bdf/sym/bdf2.inl`,
`bdf/fem_bdf2_state.{h,cpp}`, `finite_element_kinetic.{h,cpp}`,
`finite_element_animator.{h,cpp}` + `details/finite_element_animator.inl`,
`finite_element_external_force_{manager,reporter}.{h,cpp}`, plus the three pieces the
dependency cycle forces T5.3 to pull forward (`fem_line_search_{subreporter,reporter}`,
`finite_element_constraint`, `constraints/soft_position_constraint`) and the *energy
half* of `finite_element_elastics`. The rules below are what that port turned out to
depend on; the first five are the ones a later task will hit again.

1. **The BDF kernels, the beta convention, and one asymmetry the brief got wrong.** The
   transcribed laws (`fx = is_fixed(i) != 0`, `x̃ = x_tilde`, `dx = x - x̃`, `h = dt`):
   ```
   BDF1 predict : x_prevs(i) = xs(i)                       (unconditional)
                  x̃ = x_prevs
                  if !fx                x̃ += (g + f_ext_acc) * dt * dt
                  if !fx && is_dynamic  x̃ += vs * dt
   BDF1 update  : vs = (xs - x_prevs) * (1 / dt)
   BDF2 predict : x_ns(i) = xs(i);  x_{n-1} = x_n_1s(i); v_n = vs(i); v_{n-1} = v_n_1s(i)
                  g = gravities(i) + f_ext_acc(i)
                  if fx              { x_{n-1} = x_n; v_n = 0; v_{n-1} = 0; g = 0 }
                  else if !is_dynamic { x_{n-1} = x_n; v_n = 0; v_{n-1} = 0 }
                  x̃ = 4/9 h² g + 4/3 x_n - 1/3 x_{n-1} + 2/9 h (4 v_n - v_{n-1})
   BDF2 update  : v = 1/(2h) (3 x - 4 x_n + x_{n-1})
                  v_n_1s = vs;  vs = v;  x_n_1s = x_n        (old values on the right!)
   kinetic (BDF1): K = fx ? 0 : 0.5 m dot(dx,dx);  G = fx ? 0 : m dx;  H = m I
   kinetic (BDF2): K = fx ? 0 : inv_beta/2 dot(dx, m dx);  G = fx ? 0 : inv_beta m dx;
                   H = (inv_beta m) I
   ```
   `beta = 4.0/9.0`, `inv_beta = Float{1}/beta = 9.0/4.0` (both exact in `double`; the unit
   test asserts the literals, which is what catches a `float` transcription).
   **The Hessian is NOT guarded by `is_fixed`** in either kinetic kernel (only `E` and `G`
   are): a fixed vertex still gets `m I` on its diagonal block. The T5.3 brief's wording
   ("fixed => H = 0") contradicts the reference's kernels and the reference wins; the test
   asserts the actual `m I`. By contrast `SoftPositionConstraint`'s Hessian *is* guarded
   (`fx ? 0 : (s m) I`) -- do not "unify" the two.
   Device form: the reference's `if` chains become `select`s of the already computed
   candidates (`scalar_detail::select_container` / `pred_and`), in the reference's own order;
   the selected operand passes through untouched, so the values are bit-identical.

2. **The BDF2 bootstrap copies the *method's* `x_prevs`/`vs`, not the current `xs`.** The
   reference's first `do_predict_dof` resizes `x_n_1s`/`v_n_1s` and copies
   `info.x_prevs()`/`info.vs()` into them. `x_prevs` is `x_n` *as of the previous frame*
   (the method initialises it from `xs` in `_build_on_device`), so writing `xs` externally
   (e.g. through `FiniteElementStateAccessorFeature::copy_from`, which touches `xs` and `vs`
   but **not** `x_prevs`) before the first frame makes the bootstrapped `x_{n-1}` differ from
   the current `x_n`. Measured with a seeded `xs = P0` and `x_prevs = P_orig`: the first
   frame's `x_tilde` came out `4/3 P0 - 1/3 P_orig + ...` (rel error 0.2495 against the
   "obvious" expectation) and `update_state` produced `v = (P_orig - P0)/dt` -- both exactly
   what the reference computes. The test therefore seeds **velocities only** (which *is* the
   `vs` the bootstrap reads, and is what makes the `v dt` / `2/3 h V0` terms visible instead
   of vanishing in a scene at rest).

3. **The dependency cycle, and exactly what T5.3 pulled forward.** The reference's FEM
   dynamics has a genuine include cycle across the plan's T5.3/T5.4/T5.5 split:
   `FiniteElementKinetic` declares members whose parameter types are
   `FEMLinearSubsystem::ComputeGradientHessianInfo` (T5.4) and
   `FEMLineSearchReporter::ComputeEnergyInfo` (T5.5); `FEMLineSearchReporter` calls
   `FiniteElementKinetic::compute_energy`; `FiniteElementAnimator::Impl::init` needs
   `FiniteElementConstraint` and the line-search subreporter. The port's resolution (marked
   in-file with SCOPE NOTE blocks):
   * **pulled forward and completed:** `FEMLinearSubsystem::ComputeGradientHessianInfo`
     (an *inline* nested class, transcribed verbatim into T5.1's scaffold),
     `FEMLinearSubsystem::Impl::kinetic` + `add_kinetic()` (+ `friend class
     FiniteElementKinetic`), `fem_line_search_subreporter`, `fem_line_search_reporter`,
     `finite_element_constraint`, `constraints/soft_position_constraint`,
     `FiniteElementElastics::Impl::compute_energy` + `FiniteElementElasticsLineSearchSubreporter`,
     and `FiniteElementAnimatorLineSearchSubreporter`;
   * **still T5.4:** `FEMLinearSubsystem`'s system class (`DiagLinearSubsystem` base,
     `do_build/do_init/do_report_extent/do_assemble/do_accuracy_check/do_retrieve_solution/
     do_diag_norm/do_mass_norm/do_report_init_extent/do_receive_init_dof_info/get_uid`), its
     dytopo receiver, its preconditioner hooks, its assembly kernels, and the four
     `AssembleInfo` accessor *definitions*; `FiniteElementElastics::Impl::assemble` +
     `FiniteElementElasticsLinearSubsystemReporter`; and
     `FiniteElementAnimator::Impl::assemble` (declared, deliberately **not defined**) +
     `FiniteElementAnimatorLinearSubsystemReporter`. An undefined member is fine as long as
     nothing ODR-uses it -- and nothing does until T5.4 lands -- so the interface is complete
     without a stub (T5.1's rule 13 still holds: never invent a placeholder *type*).
   * **still T5.5:** the diff-dof/diff-parm reporters, the active-set/stiffness pieces, and
     `constraints/finite_element_external_vertex_force_constraint.*` (uid 671, T5.4).

4. **Registering the animator changes every scene of the shared test binary -- and the
   `constraint_uids` assert is a hard abort.** `FiniteElementAnimator` requires
   `FiniteElementMethod` *and* `GlobalAnimator` (the base `Animator::do_build` requires the
   latter), so it exists exactly in scenes that have both a `FiniteElement` and a
   `Constraint` constitution type. Where it does exist, `GlobalAnimator::init()` calls
   `Impl::init()`, which walks **every FEM geometry's** `builtin::constraint_uids` and
   `UIPC_ASSERT`s that some registered `FiniteElementConstraint` claims each uid -- an
   **abort**, not an exception, so it cannot be caught by a test. Hence
   `constraints/soft_position_constraint` (uid 14) had to be ported with the animator: it is
   the only `Constraint` the ported frontend can write into a FEM geometry's
   `constraint_uids`. Two corrections to the T5.3 brief's planning warning, both measured:
   (a) `luisa_half_plane_animator_test.cpp`'s four `with_constraint` scenes have **no**
   `FiniteElement` constitution, so the method (and therefore the animator) is invalidated
   there and the assert cannot fire -- the constraint is still required (by the general
   case and by the brief), and the test asserts it by construction; (b) even a scene without
   the `Constraint` type builds the `SoftPositionConstraint` *object* (the registry
   specialisation is process-wide) but **invalidates** it (`FiniteElementConstraint::do_build`
   throws `SimSystemException` when uid 14 is not in the scene's `ConstitutionTabular`) --
   see rule 6.
   The constraint is pure energy: its `do_step` only refreshes its own
   `aim_positions`/`strength_ratios` buffers from the animation and never writes
   `xs`/`x_tildes`/`vs`, so the dynamics of a scene with it must be **bit-identical** to one
   without it (asserted, measured 0.0e+00).

5. **`cuda_tool` idioms that must not be transcribed literally.** `DeviceBuffer::operator=`
   deep-copies in the reference, so `fem().x_temps = fem().xs;` becomes `copy_device(...)`
   (T5.1 rule 4b, restated because the line-search reporter does exactly this);
   `resize_discard(n)` becomes `loose_resize(buffer, n)` with the reference's 1.5 growth
   policy (T4.3's spelling); `DeviceReduce().Sum(in, out, n)` becomes
   `lla::device_reduce_sum(frame_stream(), in, out, n)`, **but a zero-length reduction must
   not pass an empty (null-handle) view as a kernel argument** -- the port writes the reduce
   identity with a one-element copy recorded on the same stream (T4.3's
   dytopo_effect_line_search_reporter recipe, factored here as `reduce_sum_onto`).

6. **`find<T>()` returns an *invalidated* system; `require<T>()` throws.** The engine's
   probe idiom (`T53Probe`) reads optional systems with `find<>`, and T4.x's code (and the
   T5.3 brief's warning) treated a `nullptr` result as "not created". That is only true when
   the `SimSystemCreator` returned `nullptr` (e.g. a flag system whose scene config does not
   select it): a *registered-but-invalidated* system is still found
   (`SimSystemCollection::find` looks the type up in `m_sim_system_map` and never checks
   `is_valid()`, measured with `SoftPositionConstraint` in a scene that does not use it). A
   test that means "is this system in play?" must assert `is_valid()` (a public
   `ISimSystem` member), not null-ness. Also note `FEMBDF2State`'s `do_build()` only
   requires the method, so the state *system* exists in every FEM scene -- only its two
   history buffers are BDF2-specific (a bdf1 scene leaves them at size 0, asserted).

7. **A `uipc::vector` is a `std::pmr::vector`, so Catch2 comparisons need care.**
   `CHECK(backend_vec == std::vector<T>{...})` does **not** compile (no common
   `operator==` between the pmr and the default allocator flavours; measured C2678 with a
   `Catch::ExprLhs` in the diagnostic). Compare element-wise, or build the expected value as
   the same `uipc::vector<T>`.

8. **The linear-algebra views' index type is `int32_t`, `dispatch_x()` is `uint32_t`.** A
   constitution/reporter that writes its own doublet/triplet entry
   (`G3s(i).write(i, G)` / `H3x3s(i).write(i, i, H)`, the reference's own statements -- the
   device views' `write` takes the numerics layer's `LVec`/`LMat` by value, so no
   `store_vec3`/`store_mat3x3` bridge is needed) has to cast the dispatch index once
   (`lcrt::cast<std::int32_t>(i)`, the dytopo manager's idiom); passing a `Var<uint32_t>`
   into the view's `Storage::store` fails with C2672 inside `views.h`.

9. **Passing buffers to a kernel needs `.view()`/`.cview()`.** The `laa::detail::cached_kernel`
   invocation accepts an `LBufferView` (it converts implicitly to `lcrt::BufferView<T>`), but
   an owning `LDeviceBuffer` does not convert at all (measured C2664); a *buffer element* is
   still written as a `Var<LVector3>` built with `store_vec3` (T4.2's note, re-measured).

10. **`AtomicRef<LVector3>` has no component-wise atomic add.** LC derives an `AtomicRef`
    for a `LUISA_STRUCT` type whose members are `AtomicRef<member_type>`, and
    `AtomicRef<std::array<double,3>>` exposes only the base (no `fetch_add`), so the
    reference's `eigen::atomic_add(forces(vid), F)` has no direct DSL spelling for a
    vector-valued buffer. Consequence for T5.4: the real
    `FiniteElementExternalVertexForce` reporter needs either three scalar buffers or a
    per-component kernel; T5.3's test-local mock uses a read-modify-write (exactly equivalent
    while it is the only writer, which the `clear -> step` contract guarantees for the test).

11. **The external-force manager's contract is observable from the method's buffers.**
    `FEMExternalForceManager` (`UID = 671`, an `ExternalForceReporter`, so
    `GlobalExternalForceManager` drives it) clears `vertex_external_forces`, fans `step()` out
    to its reporters (scatter-add), then writes `vertex_external_force_accs = F / m`, and
    **exactly 0 when `m <= 0`** (a *device* test, so the branch is a select). A massless
    vertex is reachable in a scene by writing a per-vertex `mass_density` of 0 (the method's
    mass is `density * volume`), and the `clear -> step` order is provable by silencing the
    reporter for one frame and checking that the buffer still ends up all-zero.

12. **Test-side uids reserved for mocks** (so the next task does not collide; the test binary
    is shared, and a mock that claims a *real* uid steals the `(dim, uid)` geometry group):
    `0x5445535456455254` "TESTVERT" (T4.1), `0x543432414E494D` "T42ANIM",
    `0x543432464F5243` "T42FORC" (T4.2), `0x54343344594E54` "T43DYNT" (T4.3), and T5.3's
    `0x54353346454D58` "T53FEMX" (an `FiniteElementExternalForceReporter`, uid > 2^32).
    `FEMExternalForceManager` itself claims 671 (the same number as the reference's
    `FiniteElementExternalVertexForce` reporter and the frontend's
    `uipc::constitution::FiniteElementExternalForce`), which is *not* available for a mock.

13. **What T5.3 deviated from, and why** (all reported, all minimal):
    * The reference's `.cu`-only classes (`FEMBDF1Integrator`, `FEMBDF2Integrator`,
      `FiniteElementBDF{1,2}Kinetic`) get a matching `.h` with the class declaration and the
      generic per-vertex routines, exactly like T5.2 rule 11; the kernels, their cached
      shaders and `REGISTER_SIM_SYSTEM` stay in the `.cpp`. Without it the T5.3 brief's
      §2.1 rule ("one generic implementation per routine over the scalar type") cannot be
      satisfied, because the reference gets host+device from `__host__ __device__` and the
      DSL cannot.
    * `FiniteElementAnimator::BaseInfo::dt()` is *declared but never defined* in the
      reference; the port defines it (`return m_dt;`), which is the only sensible reading of
      the member it stores.
    * `finite_element_kinetic.cu` includes `<finite_element/finite_element_diff_dof_reporter.h>`
      without using anything from it; the port drops the include (the diff-dof reporter is
      T5.5 and drags in `<diff_sim/...>`).
    * `SoftPositionConstraint`'s class is declared in a header (the same T5.2 rule 11
      deviation) so the unit test can drive its hooks; its three host vectors and three
      device buffers are public, as in the reference.

14. **The scene oracle recipe (what makes an end-to-end dynamics test non-vacuous).**
    Build a small tet scene with `ARAP`, `contact/enable = false`,
    `sanity_check/enable = false` and `integrator/type` = `bdf1`/`bdf2`; write the per-vertex
    `is_fixed`/`is_dynamic`/`gravity`/`mass_density` attributes (find-or-create: the
    frontend's `apply_to` may already have created `is_fixed`); seed **velocities** through
    `FiniteElementStateAccessorFeature::copy_from` (rule 2); `advance()` once and compare the
    method's `x_prevs`/`x_tildes`/`vs` against the formulas of rule 1, with the BDF2 history
    bootstrapped from the same inputs. The shipped kernels are only reachable this way (they
    are anonymous-namespace-local), so this case is what proves the *plumbing*; the
    device-buffer cases prove the *routines* (the same `*_detail` functions the kernels call)
    over synthetic buffers, and `fem_dynamics_device_vs_host` proves they agree with their
    host instantiation bitwise (measured 61/61 bitwise, worst abs 0.0).

### 7.19 Patterns established in T5.4a (FEM linear subsystem & the global linear system's assembly half)
M5 batch **T5.4a** ports `linear_system/global_linear_system.{h,cpp}` (+
`global_linear_system_function.h`), `linear_system/diag_linear_subsystem.{h,cpp}`,
`linear_system/local_preconditioner.{h,cpp}`,
`finite_element/fem_dytopo_effect_receiver.{h,cpp}`,
`finite_element/fem_linear_subsystem.{h,cpp}` (+ `fem_linear_subsystem_function.h`), the
*assembly half* of `finite_element_elastics.{h,cpp}` (the fan-out + its reporter) and
`FiniteElementAnimator::assemble` + `FiniteElementAnimatorLinearSubsystemReporter`.  The rules
below are what the port turned out to depend on; the first one is the split the brief calls the
crux of the task.

1. **`GlobalLinearSystem` is ported as "everything except the solve path", and the split is a
   *reachability* decision, not a style one.**  In the reference the assembly is only ever
   triggered by `solve()` (`_update_subsystem_extent` -> `_assemble_linear_system` ->
   `_assemble_preconditioner` -> `solve_linear_system`), and `solve()` needs `IterativeSolver`,
   `LinearPCG` (+ the CUDA-graph fused variant), `GlobalPreconditioner`,
   `OffDiagLinearSubsystem` and `CurrentFrameDiffDofReporter` — none of which exist in this
   backend.  Ported **now**: the whole public surface (every nested `*Info`, the friend set,
   `add_*()`, `dof_count()`, `last_solve_iterations()`, `compute_gradient()`, `ctx()`, the type
   aliases) and `Impl::{init, build_linear_system, _update_subsystem_extent,
   _assemble_linear_system, _assemble_preconditioner, distribute_solution, accuracy_statisfied,
   compute_gradient, diag_norm, mass_norm}` + the `*Info::extent/dof_offset/dof_count`
   definitions.  **Deferred to T8.3, declared-but-undefined** (each with a `SCOPE NOTE (T8.3)` in
   `global_linear_system.h` and `global_linear_system.cpp`): `solve()`,
   `Impl::solve_linear_system/apply_preconditioner/spmv/spmv_dot/matrix_data_ptrs`,
   `_dump_A_b`/`_dump_x`, the two `SimEngine`-only norm entries `diag_norm`/`mass_norm`, the
   `need_debug_dump`/`debug_dump_path` config read (so `do_build()` is an empty body — it is a
   pure-virtual override) and the graph-stability key `triplet_count_dev` (its only consumer is
   `spmv_dot`), which is why `build_linear_system()` does *not* upload the nnz count.  The
   off-diag branches (`OffDiagLinearSubsystem` is T8.3's type; T5.1 rule 13 forbids a placeholder)
   are marked at each site and unreachable, and `SimEngine` is **not** wired (the reference's
   `do_init` require+init and the `advance()`-side solve stay T8.4/T8.3) — the engine's own
   comment already reserves it: "GlobalLinearSystem -> Phase 8".
2. **A nested `*Info`/`Impl` class is the test hook when the engine is not wired — plus exactly
   one access deviation.**  `GlobalLinearSystem::Impl` and `FEMLinearSubsystem::Impl` are *public*
   nested classes with public members, so a test can instantiate its own `Impl`, register the
   *real* subsystem into its `diag_subsystems` collection and drive `Impl::init()` /
   `_update_subsystem_extent()` / `_assemble_linear_system()` / `compute_gradient()` on the real
   systems (the `fem_linear_layout` case does exactly that and reads
   `subsystem_triplet_offsets_counts.counts()[0] == 31`).  The one thing that is *not* reachable
   that way is the real `GlobalLinearSystem`'s own `Impl`, so the port declares
   **`GlobalLinearSystem::init()` public** (the reference keeps it private + `friend class
   SimEngine`; T5.1's rule 9 forbids widening the friend list, and a test-only friend or hook
   would be worse).  `dof_count()` needs it (`UIPC_ASSERT(m_impl.initialized, ...)` **aborts**,
   not throws), and with it `dof_count()`, the geometry `dof_offset`/`dof_count` meta attributes
   and `compute_gradient()` all become reachable.  Everything else keeps the reference's access
   levels; `solve()` and the engine wiring stay private/T8.3.
3. **The assembly kernels are factored into `fem_linear_subsystem_function.h`, generic over the
   view flavour *and* the scalar type.**  Same recipe as T5.2's `utils/matrix_assembler.h` and
   T5.3's `*_detail` namespaces: the kernel lambda builds the device views and calls one
   `fem_linear_subsystem_detail::*` routine (`clear_fixed_gradient_row`, `fix_hessian_row`,
   `scatter_gradient_row`, `scatter_dytopo_gradient_row`, `copy_dytopo_hessian_row`,
   `retrieve_solution_row`, `diag_block_norm_row`, `mass_block_norm_row`), which the unit test
   also instantiates with `storage::memory` views over host arrays.  Two consequences worth
   remembering:
   * the routines take the `is_fixed` accessor as a **callable** (`is_fixed_at(i)`): the host
     flavour indexes a host array, the device flavour reads a buffer, and both are
     `[&](auto&& i) { return ...; }` at the call site;
   * a *read* must go through `read(i)`/`read_element(v, i)`, never `v(i)`: the non-const
     flavour's `operator()` hands out an *assignable proxy*, so `zero_of<decltype(mass(i))>()`
     fails to compile (measured C3313/C2440) — `mass.read(i)` is the value.
4. **Branchless where the predicate is device data — but a *store* can be load-bearing, so
   `diag_norm` returns the guards instead.**  `if(is_fixed(i)) return;` becomes "scatter exactly
   0" (the reference's own k1 zeroes those rows anyway), and "write `Zero()`/`Identity()`
   depending on `i != j`" becomes one element-wise select of two candidates.  The **one place
   that must not be branchless** is `diag_norm`: several triplets of one row address the same
   `diag_blocks_norm` segment, so a non-diagonal thread that wrote back the value it read would
   race with the diagonal thread's store.  `diag_block_norm_row` therefore *returns*
   `{index, value, out_of_window, is_diagonal}` and the caller gates the store (`$if` in the
   shipped kernel, an `if` on the host) — and it *clamps* the returned index into
   `[0, segment_count)`, because the reference's grid is sized by the triplet count while the
   segment window is the subsystem's dof window (cookbook §7.12 rule 2).
5. **The dof layout (`uid` order, offsets) and the extent budget are factored host routines.**
   `global_linear_system_detail::plan_dof_layout(uids, dof_counts)` is the reference's
   `std::ranges::sort(..., by uid)` + exclusive scan, and
   `fem_linear_subsystem_detail::plan_extent(vertex_count, gradient_only, has_complement,
   reporter_grad_counts, reporter_hess_counts, dytopo_grad, dytopo_hess)` is
   `Impl::report_extent`'s arithmetic (`[KineticG | OtherG ... | DyTopoG]` /
   `[KineticH | OtherH ... | DyTopoH]`), with `check_gradient_only(gradient_only)` as the
   reference's issue-#295 assertion turned into a *predicate* (a test may not call the aborting
   `UIPC_ASSERT`).  The shipped `Impl::init`/`report_extent` call them **and** assert their result
   equals the reference-order accumulation, so the two definitions cannot drift.  Measured on the
   two-body scene: hessian blocks 8 (kinetic) + 2x10 (elastics) + 0 (animator) + 3 (the test's
   mock) = **31**, dofs 24 — and the real chain reports 31.
 **Watch the uid collision for this one**: the shared test binary registers T5.1's *mock* dim-3
 constitution with ARAP's uid (9), and the method hands a `(dim, uid)` geometry group to the
 *first* system registered with that uid.  Measured: in a *full-suite* run ARAP owned no
 geometry in a scene that applied `ARAP`, the elastics reporter reported **1** Hessian block
 instead of 20, and the hand-derived 31 became 12 (the layout case *failed*, which is how it was
 found -- T5.1 rule 12 / T5.2 rule 12).  The scene cases therefore apply `StableNeoHookean`
 (uid 10, unique), which keeps the hand-derived numbers valid in both the filtered and the full
 run.
6. **`AssembleInfo::{gradients,hessians}()` is the fan-out slice, and the hessian slice is empty
   in gradient-only mode.**  `gradients()` = `reporter_gradients.view().subview(offset, count)`
   with `(offset, count)` from the reporter's entry in `reporter_gradient_offsets_counts`;
   `hessians()` slices the subsystem's Hessian window the same way.  A reporter must report
   `hessian_count == 0` when `gradient_only()` (both the subsystem and
   `FEMLinearSubsystemReporter::report_extent` assert it), so `subview(0, 0)` on the
   *default-constructed* (null-handle) hessian view is the call a gradient-only chain makes — it
   is legal (`LBufferView::subview(0, 0)` does not throw) and the Hessian stage is predicated
   away.
7. **A reporter's write needs the *slice* handles, not the view's.**  `AssembleInfo` hands out a
   *view* whose `row_indices()`/`values()` are still the whole-buffer handles and whose
   `triplet_index_offset()`/`subvector_offset()` carry the slice: the writer does
   `view.row_indices().subview(offset, count)` and `view.values().subview(offset * 9u, count *
   9u)` (T5.3's kinetic kernel is the same idiom) and then builds the device view with
   `from_handles_full(..., triplet_index_offset = 0, ...)` so the routine's local index space is
   the slice.  The test-local reporter in `luisa_fem_linear_test.cpp` does exactly that and writes
   its two doublets/three triplets through `write_element_gradient`/`write_element_hessian` — the
   same routines a constitution runs.
8. **`write_element_gradient`/`write_element_hessian` are the fan-out write with the `upper_LR`
   swap, and their offset parameter must be generic.**  They are thin, test-reachable wrappers
   over T5.2's `DoubletVectorAssembler`/`TripletMatrixAssembler` (`segment<N>`/`half_block<N>`), so
   a 5-tet mesh assembles into a `DeviceTripletMatrix` through the *ported* assembly code: 10
   `(row, col, Float[3][3])` triples per tet, at *global* vertex indices, ascending `(row, col)`
   (the swap), 50 triplets for 5 tets -> `ge2sym` (a no-op for an already upper-packed input,
   which is why `triplet_count()` is unchanged) -> `convert` -> 23 unique blocks (5 of them
   exactly zero, which the oracle reproduces) -> `rbk_sym_spmv`.  The offset parameter is
   `const OffsetT&`, not `IndexT`: the host passes an `int32_t`, a kernel passes its
   `Var<int32_t>` slice offset (measured C2672 with a fixed `IndexT`).
9. **`MatrixConverter` + `Spmv` are the end-to-end oracle, and the dense model must be built in a
   *different* order.**  `converter.ge2sym(A)` keeps `row <= col` entries in stable order (it does
   not double anything: the mirroring is `Spmv`'s symmetric convention, `a * B * x_j` into row `i`
   plus `a * B^T * x_i` into row `j` for `i != j`), so a dense `A = sum_e H_e` built with plain
   Eigen in element order and multiplied by the same `x` must equal the SpMV result.  Measured on
   the five-tet mesh (24 dofs, `|y|max` 2.66e2): worst **1.34e-15** relative, and the assembled
   COO blocks match the hand-derived sums to **2.18e-16**; the device triplet write equals the
   host instantiation **bitwise** (worst 0.0).
10. **The extent path and the *gradient* assembly are what a scene can prove; the Hessian
    assembly needs the test-local `Impl`.**  With the engine not wired, `compute_gradient()` is
    the only public entry that runs `Impl::assemble` on the *real* reporters: it reports extents
    with `gradient_only = true`, runs `_assemble_kinetic` (the kinetic doublets) +
    `_assemble_reporters` (the elastics/animator reporters -> the constitution fan-out) and then
    the shared scatters + the fixed-row clear.  Its observable is the assembled gradient, and it
 matches the frontend's own laws (kinetic `m (x - x~)` from T5.3 rule 1 + the element gradients
 scattered at the element's global vertex indices + the mock reporter's known writes) to
 **5.20e-15** relative at `max |g| = 1.008e+01`, with the fixed vertex *exactly* 0.  Two things
 make that number non-vacuous, and both cost one round to learn:
 * **deform the state first** (through `FiniteElementStateAccessorFeature`, T5.3 rule 2).  At
   the rest state `F == I` a well-behaved hyperelastic gradient is *exactly* zero (checked for
   both ARAP and SNH: `dE/dF = mu F - mu F^-T + lambda log(J) F^-T = 0` at `F = I`), so the
   constitution term would be invisible and the comparison would pin only the kinetic term.
   Seeding `x -> x + (0.03(i+1), -0.02 i, 0.01(i+2))` makes it load-bearing;
 * **read the *stored* moduli, not `apply_to`'s arguments.**  `StableNeoHookean::apply_to`
   reparameterises them (`snh_mu = 4 mu / 3`, `snh_lambda = lambda + 5 mu / 6`, see
   src/constitution/stable_neo_hookean.cpp); an oracle that feeds `moduli.mu()`/`lambda()` into
   `snh_3d_detail::tet_gradient` is off by **1.84e-2** relative -- 13 orders of magnitude above
   the bound, and the failure *pattern* (a per-element constant offset, identical for two
   *congruent* tets) is what identifies it as an input error rather than a formula error (T5.2
   rule 26's 'a constant ratio is never rounding').
 Use `compute_gradient()`: it is the cheapest end-to-end evidence for a fan-out chain, and it
 needs nothing but `init()` + one `advance()` (frame 0 is not usable -- `set_dof_info` asserts
 `frame > 0`).
11. **`frame()` is *not* on the backend `SimEngine`; it comes from `core::IEngine`.**  The
    reference's `sim_engine->frame()` therefore compiles unchanged in the port even though neither
    backend's `SimEngine` declares it (`get_frame()` is the private override) — no mapping was
    needed, but `Impl::assemble`'s `set_dof_info(frame, ...)` is the line to check when a future
    task touches the frame counter.
12. **What T5.4a could not verify (recorded so T8.3/T5.4b do not assume it).**  The assemble ->
    solve path (`solve()`, `build_linear_system`'s converter step, `distribute_solution`,
    `apply_preconditioner`, `spmv`, `diag_norm`/`mass_norm` *as shipped kernels*),
    `_assemble_linear_system`'s submatrix windows, the dytopo-effect branch (contact is disabled
    in every scene of this backend, so `FEMDyTopoEffectReceiver` is ported but
    `_assemble_dytopo_effect` never runs) and `_assemble_preconditioner`'s global branch.  Their
    *bodies* are covered by the factored routines' device-vs-host case (198 dumped values, bitwise;
    106 non-zero) except the submatrix-window arithmetic, which is T8.3's to exercise.
13. **The `upper_LR` swap is *not* a cheap perturbation target any more.**  Disabling it in
    `utils/matrix_assembler.h` changes the *constitution* kernels' IR (every constitution
    assembles through `half_block`), so the ARAP gradient/Hessian shader is re-JIT'd: the
    `[fem_linear]` run did not finish in **15 minutes** (T5.2 measured 962 s for that one shader).
    For a non-vacuity perturbation of the assembly, break something in *this* batch's own factored
    routines (the fixed-mask select in `scatter_gradient_row` costs ~48 s including the rebuild
    and fails exactly the named assertions).
14. **The MSVC `luisa_check_compile.py` inner loop needs `--filter /backends/luisa/` for a
    *backend* TU.**  The script borrows the flags of `luisa_device_tests.cpp` (the *test* target)
    by default, and that target's include order puts `src/backends/none` **before**
    `src/backends/luisa`, so a backend source's `#include <sim_engine.h>` resolves to the `none`
    backend's header and every `SimEngine` use reports C2027 "incomplete type" (a false positive:
    the real build is fine).  `python scripts/luisa_check_compile.py <backend-src> --syntax
    --filter /backends/luisa/` is the invocation to use.

### 7.20 Patterns established in T5.4b (the diagonal preconditioner, the preconditioner-apply path, the external vertex force)
M5 batch **T5.4b** ports the last M5 pieces: `finite_element/fem_diag_preconditioner.{h,cpp}`
(+ `fem_diag_preconditioner_function.h`), `linear_system/global_linear_system.cpp`'s
`Impl::apply_preconditioner`, `finite_element/constraints/finite_element_external_vertex_force_constraint.{h,cpp}`
and `finite_element/finite_element_external_vertex_force.{h,cpp}`.  It also closes the
two `SCOPE NOTE (T8.3)` items the T5.4a row listed for it (`LocalPreconditioner`'s
defined members — already done in T5.4a — and `apply_preconditioner`).  The rules
below are what the port turned out to depend on; the first one is the brief's §1.1
"one genuinely hard design problem".

1. **The vector-valued atomic add is the *scalar alias* of the AoS buffer
   (`BufferView::as<U>()` + three scalar atomics) — measured, not assumed.**  The
   reference's `eigen::atomic_add(forces(vid), F)` adds a `Vector3` into an
   `AoS BufferView<Vector3>`; rule 10 above says LC's `AtomicRef<LVector3>` exposes
   the struct's `std::array<double,3>` member as an `AtomicRef<std::array<double,3>>`,
   which has no `fetch_add`, so there is no direct DSL spelling.  The three
   candidate designs were (a) three scalar buffers, (b) a read-modify-write kernel,
   (c) something LC offers.  **(c) exists:** `BufferView<T>::as<U>()`
   (`luisa/runtime/buffer.h`) reinterprets a buffer view as another element type
   (same handle, same byte offset, element stride `sizeof(U)`, size
   `size_bytes / sizeof(U)`; it is `requires(!is_custom_struct_v<U>)`), and the
   dispatch path encodes only `(handle, offset_bytes, size_bytes)`
   (`ShaderInvokeBase::operator<<(BufferView<T>)` → `encode_buffer`), so a shader
   whose prototype is `Buffer<double>` accepts the alias.  Because
   `element_stride == sizeof(LVector3)` (24 B = 3 x 8) the flat index of component
   `c` of vertex `v` is `3 * v + c`, and the reference's add becomes the three
   per-component atomic adds of `finite_element_external_vertex_force_detail::scatter_add_vec3`.
   Measured evidence (all in `luisa_fem_preconditioner_test.cpp`):
   * the shipped `do_step` asserts the contract before dispatching and the test
     prints the measured stride: **`LVector3` buffer element stride 24 B
     (`sizeof(LVector3)` 24 B)** — i.e. LC does not pad the registered struct;
   * the **duplicate-vertex case**: a synthetic table adds three entries on vertex 3
     and two on vertex 0; the scatter reproduces the hand-derived sums exactly
     (`worst abs 0.000e+00`, vertex 3 = `(11.500, 20.250, 29.125)` =
     `(1.0+0.5+10.0, 0+0.25+20.0, -1+0.125+30.0)`) and nothing else is touched.
     This is exactly what design (b) could not do: a single writer performing three
     sequential whole-`LVector3` read-modify-writes on the *same* vertex would keep
     one of the three adds (and a parallel read-modify-write would lose an
     unspecified number of them);
   * the device-vs-host round evaluates the same routine over `storage::memory`
     views and gets the 9 doubles back exactly (`dev_aos[0] = (7, 8, 9)`).
   *Trade-off (documented, guarded):* the alias is a raw reinterpretation of the
   buffer's memory, so it is only valid while LC lays a custom-struct buffer out as
   `sizeof(T)`-strided contiguous elements.  That is asserted at the call site
   (`UIPC_ASSERT(native.stride() == sizeof(LVector3), ...)`) so a future LC that
   padded the element would trip the assert instead of silently writing to the wrong
   words.  Design (a) (three scalar buffers) was rejected because it would change
   T5.3's `FiniteElementMethod::vertex_external_forces` layout, which its clear/step
   kernels and the external-force contract already fix as AoS.
   *Portability note:* the same trick would work for any `LVectorK` AoS buffer whose
   element stride is `K * sizeof(scalar)`; it is the LC answer to "CUDA's
   `atomicAdd` on a user type" (`cuda_tool::eigen::atomic_add`).
2. **`Impl::apply_preconditioner` — the slice semantics and a *constant-true* copy
   fallback.**  Ported verbatim from `global_linear_system.cu:466-511`: for every
   registered `LocalPreconditioner`,
   `index = preconditioner->m_subsystem->m_index`,
   `(offset, count) = diag_dof_offsets_counts[index]`, and the info hands over
   `z.subview(offset, count)` / `r.subview(offset, count)` — so a preconditioner
   works in a **window-local index space** (`i * 3`, not `offset + i * 3`), which is
   exactly what the reference's `do_apply` kernel assumes.  Then the copy fallback
   runs over `no_precond_diag_subsystem_indices` (filled by `Impl::init()`), as
   `z_sub.buffer_view().copy_from(r_sub.buffer_view(), stream)`: the port's
   `LBufferView` *is* a windowed handle (`m_offset`/`m_size`), so the d2d copy copies
   the window — unlike `DenseVectorView::native()`-style whole-buffer handles.  Two
   deviations, both forced:
   * the reference's `if(!global_preconditioner)` **cannot be spelled**: testing the
     slot instantiates `SimSystemSlot<GlobalPreconditioner>::lazy_init()` on the
     incomplete (T8.3) type — measured C2027.  Since the slot's only writer is the
     T8.3 `add_preconditioner(GlobalPreconditioner*)` (a no-op here, T5.1 rule 13),
     the guard is constant-true and the fallback runs unconditionally; the comment
     above the loop records the reference's guard so T8.3 restores it in one line;
   * the global-preconditioner fan-out itself stays a `SCOPE NOTE (T8.3)` comment.
3. **What is still T8.3** (so T8.3 does not re-derive it): `solve()`,
   `Impl::solve_linear_system`, `spmv`, `spmv_dot`, `matrix_data_ptrs`, `_dump_A_b`/
   `_dump_x`, `need_debug_dump`/`debug_dump_path`, `triplet_count_dev`, the
   off-diag subsystem branches, the *global*-preconditioner branch of
   `apply_preconditioner`, and the `SimEngine` wiring.  `apply_preconditioner`,
   `build_linear_system`, `diag_norm`/`mass_norm` and every assembly kernel are now
   defined and test-reachable.
4. **The preconditioner's slice arithmetic is the reference's, including the
   signed-guard subtlety.**  `do_assemble` sizes `diag_inv` by the FEM vertex count
   (`xs().size()`, one 3x3 block per DoF block), dispatches over
   `info.A().triplet_count()` (the *whole* assembled matrix, not the FEM window) and
   converts `g_i - dof_offset()/3` into the window.  The guard is
   `!(i >= fem_segment_count || j >= fem_segment_count)` — the reference's two early
   `return`s.  Note it is **signed**: an `i < 0` passes the guard (the reference has
   the same hole), so the port's factored routine returns `{index, inverse(H3x3),
   in_window, is_diagonal}` and the caller gates the store (`$if` in the shipped
   kernel, an `if` on the host) — the same shape T5.4a chose for `diag_block_norm_row`
   (§7.19 rule 4).  The inverse is computed unconditionally (the discarded lanes are
   never stored), so a multi-subsystem layout computes an inverse of a zero block in
   the discarded lanes — harmless, and it keeps the routine a single shared body.
5. **The `converged != 0` early-out is a scalar read that wraps the whole body.**
   The reference's `if(*converged != 0) return;` is per-thread; the port reads the
   one-element `converged` buffer once and wraps the body in
   `$if(converged->read(0) == 0)`.  Measured: with `converged = 1` **all 24 z
   entries keep their pre-filled sentinel** (24/24 unchanged); with `converged = 0`
   the apply matches the oracle.
6. **The preconditioner inverts the *assembled* block-diagonal A, and a fixed
   vertex's diagonal block is `(1 + #incident tets) * Identity` — a measured
   reference quirk, not a port bug.**  The assembly writes the kinetic `m I`
   (unconditional, T5.3 rule 1) and every element's 10 upper-packed blocks as
   *separate triplets*, and `triplet -> BCOO` is a sort + a segmented **sum**, so
   duplicates accumulate (the CUDA reference's `MatrixConverter` is the same
   `FastSegmentalReduce`).  `fix_hessian_row` replaces the block of *every* triplet
   whose `(row, col)` touches a fixed vertex by `Identity()`, and those replacements
   then sum.  Measured on the 5-tet scene: vertex 0 (fixed, 3 incident tets) has an
   assembled diagonal block of exactly **`4.0 * Identity`** — one kinetic triplet +
   three elastic ones.  Consequence for the preconditioner: a fixed vertex's
   residual is mapped to `r / (1 + #incident tets)`, not to `r`.  The test's oracle
   models the reference's triplet arithmetic (not the "obvious" `Identity`), which is
   what makes the assembled block comparison 2.4e-15 instead of 7.5e-1.
7. **`diag_norm()` / `mass_norm()` are max-reductions over the assembled diagonal.**
   The reference's two kernels (`fem_linear_subsystem.cu`):
   `diag_norm`: `diag_blocks_norm(i) = is_fixed(i) ? 0 : max(|H_ii(0,0)|, |H_ii(1,1)|,
   |H_ii(2,2)|)` where `H_ii` is the *assembled* diagonal block;
   `mass_norm`: `diag_blocks_norm(i) = is_fixed(i) ? 0 : mass(i)`; then
   `GlobalLinearSystem::Impl::{diag_norm,mass_norm}` take the max over the diag
   subsystems (here: one), each a device max-reduction.  The `is_fixed` guard is a
   *device* predicate, so it is a select in the shipped kernels (T5.4a's factored
   routines).  Measured against hand-derived values built from the ported T5.2
   element Hessians and the frontend's mass law (`m = density * sum_e V_e / 4`):
   diag_norm **4.430390e+02** (hand-derived 4.430390e+02), mass_norm **2.578125e+02**
   (hand-derived 2.578125e+02) — exact, and the mass law itself matches the method's
   device masses to 0.000e+00.  These are the norms the AL pipeline scales `mu` by
   (T8.x); they are shipped kernels now.
8. **The external-force contract is observable end to end, and uid 671 lives in
   *three different registries*.**  The schedule is
   `SimEngine::step_animation_and_external_forces()`:
   `GlobalExternalForceManager::clear()` → `GlobalAnimator::step()` (→
   `FiniteElementAnimator` → the constraints' `do_step`, which re-upload the
   `external_force` attributes) → `GlobalExternalForceManager::step()` (→
   `FEMExternalForceManager`: every `FiniteElementExternalForceReporter` scatter-adds
   → `vertex_external_force_accs = F / m`, **0 when `m <= 0`**).  The three
   registries are: the *constraint-uid* registry
   (`FiniteElementExternalVertexForceConstraint::UID = 671`), the
   *external-force-reporter* registry (`FEMExternalForceManager::UID = 671`, T5.3)
   and the *constitution* registry
   (`uipc::constitution::FiniteElementExternalForce`, uid 671).  They are separate
   `(kind, uid)` namespaces, so the collision is by design (the reference's own
   numbering); it is *not* available for a test mock.  Measured on the frontend
   scene: 4 constrained vertices collected in vertex order {1, 3, 5, 6},
   `worst |F - expected| = 0.000e+00`, `worst |a - F/m| = 0.000e+00`, the massless
   vertex (`mass_density = 0`) has `|F| = 4.0e-7 > 0` and `a` **exactly 0**, and the
   unconstrained vertex keeps `F = 0` even though its `external_force` attribute was
   set (the frontend's contract: "in" is `is_constrained`, not the attribute's
   presence).
   *Frontend trap measured here:* `uipc::constitution::FiniteElementExternalForce::apply_to(sc, F)`
   creates `is_constrained` (**all zero**) if missing and fills `external_force`
   uniformly; a scene must create `is_constrained` *before* `apply_to` (it keeps the
   existing attribute) and overwrite `external_force` afterwards.
9. **The constraint's extents are 0/0/0 and one test seam was added for it.**
   `do_report_extent` declares zero energy/gradient/Hessian (the reference's comment:
   "forces are applied through x_tilde in the time integrator") and
   `do_init == do_step`.  `FiniteElementAnimator::ReportExtentInfo` had no read
   accessor (the animator is its only reader), so T5.4b added three const overloads
   (`energy_count()`, `gradient_count()`, `hessian_count()`) — a pure test seam, no
   behaviour change, the same rationale as T5.4a's public `GlobalLinearSystem::init()`.
   The constraint's hooks and `m_impl` are public for the same reason (T5.2 rule 11):
   the duplicate-vertex case writes a synthetic (vertex_id, force) table into the
   constraint's own device buffers, which is the only way to produce two entries on
   one vertex (the frontend gives every geometry vertex a distinct global id).
10. **Device-vs-host: integer outputs are *exact*, floating outputs agree to ~1 ulp
   because the CUDA codegen contracts into FMAs.**  Measured over one round of every
   factored routine (`diag_inv_row`, `diag_apply_row`, `scatter_add_vec3`):
   **65/69 doubles bitwise equal, worst 1.110e-16 (1 ulp)**, and **15/15
   guards/indices exactly equal** (`in_window`, `is_diagonal`, the local index).  T2.1
   /§7.16's "bitwise" convention therefore holds only where no `a*b+c` is contractible;
   `lps::inverse`'s adjugate and the matrix-vector product are not, and the assertion
   is stated at 1e-15 relative with the measured absolute/relative numbers printed.
11. **Two compile traps this batch measured** (both cost a round):
    * **`Var<Buffer<T>>` is move-only**: a generic routine must not take a kernel's
      `BufferVar<T>` parameter **by value** — measured C2280
      (`Var<Buffer<LMatrix3x3>>::Var(const Var<Buffer<LMatrix3x3>>&)` is deleted).
      Scalar/`double` buffer vars convert to `Expr<Buffer<T>>` (copyable) on the way
      into `from_handles`, which is why T5.4a's routines never hit it.  The batch's
      answer: the shared routines take a **view** (`DiagInvDeviceView` /
      `DiagInvHostMemoryView`, built at both call sites with `from_handles`), never a
      raw buffer var.
    * **`lps::inverse` needs `#include <luisa_device/math/matrix_ops.h>`**: neither
      `type_define.h` nor `luisa_device/linear/linear.h` pulls it in (measured C2039
      "`inverse` is not a member of `lps`"), because `mat.h` only *declares* it.
12. **A T5.4a test mesh cannot be reused for a *scene*.**  The five-tet connectivity of
    `luisa_fem_linear_test.cpp` has two negative-determinant tets
    (`{0,2,3,6}` -> -0.25, `{4,3,1,7}` -> -0.75), which `ARAP`'s/`SNH`'s `apply_to`
    accepts only through `compute_vertex_volume`, which **asserts `D > 0`** and aborts
    the test.  T5.4b's scene uses a 5-tet mesh with all-positive rest determinants
    (`{1,0,3,2}`, `{0,1,2,4}`, `{0,3,2,6}`, `{1,2,4,5}`, `{2,4,6,7}`), keeping one
    *non-ascending* even permutation so the `upper_LR` swap is still exercised.
13. **Test uids: T5.4b reserves none.**  It drives the *real* systems instead of mocks
    (the real `FEMDiagPreconditioner` registered into a standalone
    `GlobalLinearSystem::Impl`, the real constraint/reporter pair), and the new test
    registers only `T54bProbe` — a plain `SimSystem` with no uid.  The reservations
    from §7.18 rule 12 stay in force; in particular **671 is still not available**
    (it is the constraint/reporter/constitution uid above).
14. **Regression shape of this batch.**  `run` -> 757525 assertions / 139 test cases
    (= the T5.4a baseline 757421/135 + this file's 104/4) and `run "~[fem_precond]"`
    -> **757421 assertions / 135 test cases**, i.e. *no* pre-existing expectation
    moved even though two new systems (a `LocalPreconditioner` and a
    `FiniteElementConstraint` + its reporter) are now registered in every scene of the
    shared binary: the preconditioner is inert until `_assemble_preconditioner`/
    `apply_preconditioner` are called, the constraint throws `SimSystemException`
    unless uid 671 is in the scene's constitution table (and then its reporter is
    invalidated with it), and the filters `[half_plane]` 426/7, `[constitution_a]`
    2362/7, `[fem_dynamics]` 135/8 and `[fem_linear]` 323/4 are unchanged.

### 7.21 Patterns established in T6.1a (ABD core: types, jacobi matrices, dynamics, energy, kinetic, reporters)

M6 batch **T6.1a** opens the Affine Body Dynamics port: the new directory
`src/backends/luisa/affine_body/` gets `type_define.h`, `utils.{h,cpp}`,
`abd_jacobi_matrix.{h,cpp}` (+ `details/abd_jacobi_matrix.inl`),
`abd_energy.{h,cpp}`, `affine_body_dynamics.{h,cpp}` (+ `details/affine_body_dynamics.inl`),
`abd_time_integrator.{h,cpp}`, `affine_body_state_accessor_feature.{h,cpp}`,
`affine_body_state_accessor.cpp`, the five reporters
(`affine_body_{body,vertex,surface}_reporter.{h,cpp}`,
`affine_body_external_force_{reporter,manager}.{h,cpp}`) and the two interfaces
`affine_body_constitution.{h,cpp}` / `affine_body_constraint.{h,cpp}` — 19 files, 5940
lines — plus the engine wiring in `luisa/sim_engine.{h,cpp}` and the test
`apps/tests/backends/luisa/luisa_abd_base_test.cpp` (tag `[abd_base]`).

1. **The M6 re-partition (the plan does *not* contain it yet — this is the record of
   it).** The reference's `affine_body/` has structural include cycles, so M6 was
   re-partitioned: **T6.1a** = types / `utils` / `abd_jacobi_matrix` / `abd_energy` /
   `affine_body_dynamics(+inl)` / `abd_time_integrator` / `affine_body_state_accessor(+feature)` /
   reporters / `affine_body_constitution` + `affine_body_constraint` *interfaces*;
   **T6.1b** = `abd_linear_subsystem(+reporter)` + `abd_dytopo_effect_receiver` +
   `affine_body_animator(+details)` + the inter-affine-body layer; **T6.2** =
   `affine_body/bdf/`; **T6.3a/b** = `constitutions/` (arap, ortho_potential) + the four
   joints; **T6.4** = `constraints/` + `abd_diag_preconditioner` + `abd_tolerance_checker`
   + the shared `mas_preconditioner_engine`/`fem_mas_preconditioner`; the two AL files
   (`abd_active_set_reporter`, `abd_al_stiffness_estimator`) are M8's.  A batch may
   **declare** another batch's interface (rule 2), never its bodies.

2. **Pull-forward = declaration surface only, no stubs.**  T6.1a had to declare
   `ABDLineSearchReporter` + `ABDLineSearchSubreporter` (T6.2/T6.4 own the bodies),
   `ABDLinearSubsystem` (T6.1b), `AffineBodyAnimator` (T6.1b) and
   `AffineBodyKinetic`'s *interface* (the `bdf/` bodies are T6.2).  Each file carries a
   `SCOPE NOTE (T6.x)` naming exactly what is missing and who owns it, and nothing is
   stubbed: an unimplemented entry point stays a **link error** (rule 7.16-13).
   Measured closure (the three definitions T6.1a *did* have to provide because an
   in-scope, reachable call site needs them):
   * `AffineBodyAnimator::add_constraint` — `AffineBodyConstraint::do_build` calls it;
   * `AffineBodyAnimator::BaseInfo::dt()` — the reference declares it and never defines
     it (the same finding as `FiniteElementAnimator::BaseInfo::dt()` in T5.3);
   * `AffineBodyAnimator::ReportExtentInfo::check()` — `AffineBodyConstraint::report_extent`
     calls `info.check(name())` and that member is compiled here, so leaving it out was a
     **measured LNK2019**, not a "not implemented" signal.  Its body is the reference's
     one line (`check_report_extent(...)`).

3. **The ABD element types are aggregates with `LUISA_STRUCT` registrations, and they
   are byte-identical to the Eigen aliases.**  12-dof data is
   `LVector12 = lps::LVec<Float,12>` (96 B) and `LMatrix12x12 = lps::LMat<Float,12,12>`
   (1152 B, column-major = Eigen's default order).  The two *aggregates* the reference
   defines in `abd_jacobi_matrix.h` are
   `ABDJacobi{LVector3 x_bar;}` (24 B, the Eigen class is one `Vector3` member at offset
   0 with no base/virtuals) and
   `ABDJacobiDyadicMass{Float m_mass; LVector3 m_mass_times_x_bar;
   LMatrix3x3 m_mass_times_dyadic_x_bar;}` (104 B, offsets 0/8/32 — every member is
   8-byte aligned, so neither side pads).  All four sizes are `static_assert`ed in
   `affine_body/type_define.h` and re-checked against a *mirror* Eigen-layout struct in
   the test (`sizeof(ABDJacobiDyadicMass) == 104`,
   `offsetof(m_mass_times_dyadic_x_bar) == 32`, ...), because the ABI argument for a
   different-language port has to be measured, not asserted in prose.  Note the
   *alignment* differs on purpose (Eigen over-aligns a 96/1152-byte type to 16 B,
   `LVec`/`LMat` use 8 B): only size and storage order take part in the buffer ABI, and
   every host↔device crossing goes through `to_luisa()`/`to_eigen()` element-wise —
   never a `reinterpret_cast`.

4. **`LUISA_STRUCT` registration order is load-bearing, twice over.**
   (a) The registration *derives* the DSL member access (`Expr<S>`/`Ref<S>`/`Var<S>`),
   so any helper that reads `e.member` or writes `dst.member` must come **after**
   `LUISA_STRUCT(S, ...)` — the file's layout is aliases → generic views → aggregate
   element types → registrations → `load_*`/`store_*` helpers (measured C2908/C2766/C2039
   while writing it the other way round).  (b) The registration block itself must be at
   **global** scope (like the backend's own `type_define.h`), so the helpers live in a
   *second* `namespace uipc::backend::luisa` block after it.

5. **One generic body per routine, templated on the scalar (`S`), never a host/device
   pair.**  Every numeric routine is header-only and `S`-templated (`double` host,
   `lcrt::Var<double>` device); `utils.cpp`/`abd_energy.cpp` then hold the reference's
   *Eigen-typed* host signatures as thin forwards to `S = Float`.  This is the T5.1
   convention (§7.16 rule 1), and it is what makes the device-vs-host round *bitwise*
   (T6.1a measured **779/779 bitwise equal**, worst absolute difference 0.0).

6. **`q`/`A` conventions (the (body, vertex) index maps).**
   `q = [t; a1; a2; a3]` where **`a_i` is row `i` of `A`** (`q_to_A` reads
   `A(i,j) = q(3+3i+j)`), so the deformed point is `p = A x̄ + t` — *not* `Aᵀx̄ + t`.
   `ABDJacobi::to_mat()` is the 3x12 jacobian `J(a,a)=1`, `J(i,3+3i+j)=x̄_j`, and
   `ABDJacobiT::operator*(g)` is `Jᵀg`.  The port's T6.1a test initially wrote the
   transposed oracle and the *port* was right: the two forms differ by a transpose, and
   the FD/layout checks pin it down.
   The dynamics indexes bodies and vertices as
   `vertex_id = geo_info.vertex_offset + instance * (sc.vertices().size()) + local_vertex`
   and `body_id = geo_info.body_offset + instance`; `v2b` maps the former onto the
   latter, `vertex_id_to_J` holds the **rest** position of every vertex, and geometries
   are ordered by `constitution_uid` (stable) — i.e. by creation order when one
   constitution owns them all.

7. **`ABDJacobiStack<N>` had a wrong return type in the port until the test
   instantiated it.**  The reference has *two* different maps:
   `ABDJacobiStackT::operator*(const Vector<Float,3N>&)` returns **`Vector12`**
   (`sum_i J_iᵀ g_i`), while `ABDJacobiStack::operator*(const Vector12&)` returns
   `Vector<Float,3N>` (`q -> (J_1 q, ..., J_N q)`).  The port's first spelling gave the
   transpose map the stack-map return type; it compiled (nothing instantiated it) and the
   `abd_base_jacobi` case's `ABDJacobiStack2` round tripped it into a **C2440**.  The
   stack is dead code in the reference and its only call site here is the test, which is
   exactly why the test must touch every ported routine.

8. **A local variable named like the scalar template parameter silently breaks
   `lps::zero_of<S>()`.**  The reference's `skew` writes `S << ...` with a local `S`
   (fine for Eigen); the port's `lps::LMat<S,3,3> S;` shadows the template parameter, so
   `lps::zero_of<S>()` received a *variable* — measured **C2672 "invalid template
   argument for 'T', expected a type"** at the first device instantiation.  The local is
   now `ret`.

9. **`obj.template member(...)` is rejected by MSVC on a non-dependent member name.**
   The reference's `affine_body_dynamics.inl` calls
   `m_impl.template for_each(geo_slots, ...)`; MSVC reports
   **C2187 "unexpected 'template'"** *at instantiation* (it parses the body lazily, which
   is why the un-instantiated TU compiled).  Dropping the disambiguator is correct here
   (`m_impl` is a concrete type and the overloads differ by arity) and is what the FEM
   `.inl` already does — recorded as a port deviation.

10. **Eigen expressions bound with `auto` dangle (a *test* trap, but worth the rule).**
   `const auto Minv = <temporary>.inverse();` and
   `const auto H = 0.5 * f();` (a scalar times a by-value temporary) bind an
   `Eigen::Inverse<>`/`CwiseBinaryOp` **expression** that references a temporary which
   dies at the end of the statement.  The T6.1a test measured 144 `nan`s from exactly
   that, and an earlier version of the same lines passed "by luck"; every Eigen-valued
   local in the test is now explicitly typed.  (The ported code itself is unaffected: it
   returns concrete `lps::` containers.)

11. **The ABD layer's own LC idioms.**  Scalars have no `operator+=` /
   `operator-=` (`lps::detail::{add,sub,mul}`; §7.16-3), `h.block<3,1>(r,c)` is an
   `LMat<3,1>` (not an `LVec<3>`), so `h.block<3,1>(3,0).col(0) += m_x_bar` is the only
   spelling of the reference's `h.block<3,1>(r,c) += m_x_bar`; `norm/dot/squaredNorm/cross`
   are **free** functions for views (a `MatMap` row has no `.norm()`); `LMat::block<BR,BC>`
   is (rows, cols); `lps::inverse`/`determinant` exist for 2/3/4 only, so the 12x12 mass
   inverse is computed with **Eigen** on the host (`.inverse()` — which also needs
   `<Eigen/LU>`, else the call *compiles* and fails to link, measured **LNK2019** for
   both `Matrix12x12::inverse()` and a 3x3 `Block::determinant()`); device predicates in
   `abd_dyadic_mass_inertia_tensor_cm` and `torque_to_F` become `select`s (RULE M2),
   with the reference's warning routed through `LC_LOG`/`logger::warn`.

12. **The engine wiring is part of the batch that ports the dynamics, not of Phase 8.**
   `luisa::SimEngine` had no `AffineBodyDynamics` slot, so `world.init()` never called
   `AffineBodyDynamics::init()`: the system built, the constitutions registered, and the
   dynamics' `abd_body_count` stayed **0** (the (uid → geometry) classification is done
   in `Impl::init`).  The reference's `sim_engine_do_init.cu` step 1 order is
   `affine_body_dynamics -> inter_affine_body_constitution_manager -> finite_element_method
   -> global_body_manager`, and the port now mirrors it (`find<AffineBodyDynamics>()`
   beside `find<FiniteElementMethod>()`, `init()` before the FEM method's).  Symptom to
   remember: a *valid* system with a valid constitution and an empty body list means the
   engine never initialised it.

13. **Test-side findings that would repeat for any ABD test (recorded here so the next
   batch does not rediscover them).**
   * A *rotated/scaled* ABD cube is a different **rest** configuration (the mesh
     positions), not a different instance transform: the dyadic mass integrates the rest
     positions, the instance transform is the deformed state `q`.
   * The surface reporter reports **only what the scene labelled**: `label_surface(sc)`
     must be called on the tet mesh (`label_triangle_orient` in turn *requires* a
     tetmesh — it aborts on a trimesh).  A tetmesh built from raw connectivity has no
     triangles at all, so the surface arrays are empty without it.
   * `AffineBodyConstitution::apply_to(sc, kappa, mass, volume)` (the explicit-mass
     overload) is the only way to get a **dim-2** ABD body while T6.3a's shell
     constitution is missing: the geometry must carry `constitution_uid == 1` to be in
     the ported dynamics' group, and the density overload's dim-2 path needs a closed,
     *oriented* trimesh.
   * The shared-binary rule bites *harder* here than in M5: `luisa_global_geometry_test.cpp`'s
     reporters report **every** scene geometry as a body/vertex set, so this file's
     `body_offset()` is 1 and the global vertex array is 8 entries longer than the scene's
     — all assertions are made on the reporter's own slice.
   * The default scene config is already a **BDF1** scene
     (`integrator/type == "bdf1"`), so `BDF1Flag` exists for any default scene; the test
     sets it explicitly anyway and `find<>`s the flag.

14. **Test uids and what the mock proves.**  T6.1a has no backend ABD constitution
   (`constitutions/` is T6.3a's), so the file registers **`T61ABDMockConstitution` with
   uid 1** = the frontend `uipc::constitution::AffineBodyConstitution`'s
   "OrthoPotential" (the only alternative — an arbitrary reserved uid — is impossible:
   `ConstitutionTabular::insert(uid)` resolves through `ConstitutionUIDCollection::find(uid)`
   and throws for an unregistered uid).  This is the §7.18-12 precedent (T5.1's mock
   claims ARAP's uid 9) and it *must* be revisited when T6.3a lands: the mock's group is
   then contested and the scene case's own report (`do_init`'s recorded
   `geo_infos().size()` / `body_count()` / `vertex_count()`) is what says which system
   got it.  The new uid space: **666** is the ABD external-force manager's
   (`AffineBodyExternalForceManager::UID`, "the same UID as external force constitution"
   and a *reporter* uid here, like the FEM manager's 671); **1** is claimed by the test
   mock as above; 671 stays as §7.18-12 recorded it.

15. **Regression shape of this batch (the T5.4b baselines, exactly).**
   `run` -> **759668 assertions / 146 test cases** (= the T5.4b baseline 757525/139 +
   this file's 2143/7) and `run "~[abd_base]"` -> **757525 assertions / 139 test cases**,
   i.e. *no* pre-existing expectation moved although five new systems
   (`AffineBodyDynamics`, three reporters, the external-force manager, the state accessor
   and the time-integrator/constitution/constraint/animator declarations) are now
   registered in every scene of the shared binary — all of them are either invalidated
   by `require<AffineBodyDynamics>()`/`BDF1Flag` when the scene has no ABD, or inert.
   `ctest` 1/1 and the standalone build (69 cases, 0 checks failed) are unchanged.

16. **Non-vacuity evidence for the batch** (brief §3.7, three perturbations, each
   patched → built → run → reverted): moving `ABDJacobi::to_mat`'s block column
   `(0,3) -> (0,6)` fails 11 assertions (`J(a,b) vs J_oracle`, `Jq vs p_oracle`,
   `M6 vs J0/J1`); scaling the gravity's mass factor by 0.5 in
   `_build_geometry_on_host` fails 8 (`grav[0] vs expected`, `|grav_r| <= 1e-9`,
   `grav[0] vs g`); dropping the vertex reporter's `+ body_offset` fails 32
   (`gvm_body_ids == body_offset + inst`).  After the reverts the same commands are
   green again (2143 assertions, 0 failures).

### 7.22 Patterns established in T6.1b (ABD linear subsystem, animator, inter-affine-body layer)
M6 batch **T6.1b** completes the ABD layer M6 opened: `affine_body/abd_linear_subsystem.{h,cpp}`
(+ `abd_linear_subsystem_function.h`), `abd_linear_subsystem_reporter.{h,cpp}`,
`abd_dytopo_effect_receiver.{h,cpp}`, `affine_body_animator.{h,cpp}`
(+ `details/affine_body_animator.inl`), `abd_line_search_reporter.cpp` /
`abd_line_search_subreporter.cpp` (the two T6.4 bodies a vtable needs, see rule 1), the
inter-affine-body layer (`inter_affine_body_constitution.{h,cpp}`,
`inter_affine_body_constitution_manager.{h,cpp}` + its `.inl`, `inter_affine_body_animator.{h,cpp}`,
`inter_affine_body_constraint.{h,cpp}`), the `SimEngine` step-1 wiring for the inter-body
manager and the test `apps/tests/backends/luisa/luisa_abd_linear_test.cpp` (tag `[abd_linear]`).
1. **The dependency closure was measured, and only two T6.4 bodies were pulled forward.**
   `AffineBodyAnimatorLineSearchSubreporter` and
   `InterAffineBodyConstitutionLineSearchSubreporter` derive from T6.4's
   `ABDLineSearchSubreporter`, whose only virtual override plus the four forwarding hooks are
   needed for their *vtables*, and that `do_build` calls
   `ABDLineSearchReporter::add_reporter` — so `abd_line_search_subreporter.cpp` (4 bodies) and
   `abd_line_search_reporter.cpp` (1 body, `add_reporter`) exist here with SCOPE NOTEs naming
   T6.4 as the owner of the rest (the T5.3 / `fem_line_search_subreporter.cpp` precedent).  The
   subreporters are then *invalidated in every scene* by
   `require<ABDLineSearchReporter>()` — the explicit not-implemented signal, not a no-op.
2. **The ABD Jacobi triple's index conventions, and the `I * 16` that is easy to get wrong.**
   A body owns 12 dofs, so its 12x12 block is a **4x4 block of 3x3 blocks** and the global
   triplet buffer (the `DiagLinearSubsystem`/`GlobalLinearSystem` `DoFBlockSize == 3` layout) is
   a 3x3-block COO: the reference's `TripletMatrixAssembler::block<4,4>(I * 4 * 4).write(...)`
   *starts at the I-th body's 16-triplet run*, so the kernel's per-thread call must pass
   `I * kTripletsPerBodyBlock`, not `0`.  T6.1b shipped `0` first and the COO assertion caught
   it immediately: only thread 0's block landed and the rest of the array kept the `-1` sentinel
   (measured: 208 triplets, 17 distinct keys instead of 48).  The unpacking order is
   `counter = a * 4 + b` with the sub-block `H12.block<3,3>(3a, 3b)`, and the row/col indices
   written are the *submatrix-relative* blocked indices (`body * 4 + a`), which the view adds to
   its own `submatrix_offset`.
3. **`zero_out_lower` is part of the convention, not a detail.** Every *same-body* 12x12 block
   (the body term's kinetic+shape block *and* every reporter's diagonal block *and* the dytopo
   same-body pair) is stored **upper-triangular in 3x3 blocks**; for `body_i < body_j` all 16
   sub-blocks are written; for `body_i > body_j` the reference zeroes the block (its reporters
   only report the upper triangle).  The symmetric reconstruction is then performed by
   `ge2sym` + `MatrixConverter`: entries with `row > col` are dropped, off-diagonal blocks are
   mirrored transposed by the SpMV, and a *diagonal* (`row == col`) block is **not** mirrored
   (`spmv.h`'s contract).  Measured consequence, which a test oracle must model: the assembled
   ABD matrix is symmetric at the **3x3-block level only** — the within-body couplings that
   `zero_out_lower` removed stay zero.
4. **The dense oracle must `+=`, never `=`.**  The oracle that rebuilds the matrix from the
   hand-derived `(row, col, block)` sums walks a *map that still contains the zero-valued lower
   half* of each diagonal block; writing the mirror with `=` lets that zero entry clobber the
   real upper entry (measured: `A_eff(0,3)` 2.55 vs oracle 0.0).  With `+=` the measured
   agreement is **2.493e-14** against the effective operator extracted by 24 unit-vector
   `rbk_sym_spmv` solves, and `A*x` matches to **4.265e-16** (bound 1e-5) while the
   "no mirroring at all" convention is off by **5.074e-01** — i.e. the comparison pins the
   convention instead of accepting any assembly.
5. **What the COO assertion buys.**  `adj_linear_chain` reads the raw `triplet_A` (208 triplets)
   *before* the conversion, sums duplicates per `(row, col)` and compares against the
   hand-derived map: **worst |dev-oracle| 0.000e+00** with 432 values compared, and the
   hand-derived block sums are exact for every key.  After `ge2sym` + `convert` the BCOO is
   sorted/unique with the duplicate sums preserved (**worst 1.178e-14**, 36 unique blocks).
   The two conventions (mirror / upper-only) are also exposed as an `is_valid()`-style
   non-vacuity: the block map of the *body* term alone cannot distinguish them, the
   within-body lower keys can.
6. **`ComputeGradientHessianInfo`'s two buffer types were wrong in T6.1a's scaffold, and T6.1b
   fixed them.**  The reference's kinetic/constitution interface hands over **dense per-body
   buffers** (`cuda_tool::BufferView<Vector12>` / `BufferView<Matrix12x12>`); a constitution
   writes `gradients(body_i) = G12` (measured against
   `ortho_potential_compute_gradient_hessian_kernel`).  T6.1a had mapped them onto
   `LDoubletVectorView`/`LTripletMatrixView` — the *reporter* interface's shape (which is
   correct for `AssembleInfo` and for `AffineBodyAnimator::ComputeGradientHessianInfo`, where a
   constraint reports 12x12 `(body_i, body_j)` blocks).  In the port the dense buffers are
   `lud::LBufferView<LVector12>` / `lud::LBufferView<LMatrix12x12>` (windowed views), and a
   constitution's kernel is dispatched with them directly (the dispatch encodes the window).
7. **LC has no `fill` *command*, and `LBufferView::fill` runs on the main stream.**  The ABD
   subsystem's contact-only branch needed the reference's `gradients.buffer_view().fill(0)`
   and got a local zero-kernel on the frame stream instead (§7.11 rule 7), the T5.4a finding.
   The same trap is *live* in `GlobalLinearSystem::Impl::_assemble_linear_system`, whose
   `triplet_A.fill(0)/row_indices().fill(-1)` go to the main stream while the subsystem kernels
   are recorded on the frame stream: a test that reads the assembled buffers back must
   `gg::frame_stream().synchronize()` (this batch's test does), and **T5.4a's test never read the
   assembled content back, which is why the race was not visible before** — T8.3 should replace
   those three fills with recorded kernels when it ports `solve()`.
8. **The ABD assembly budget.**  `plan_extent` (the factored routine the unit test drives) is
   the reference's arithmetic: the body term contributes **one 12x12 block per body** only when
   the Hessian is assembled at all, the reporters contribute their 12x12 *gradient doublets*
   (always scanned — the slices are needed in gradient-only mode too) and their 12x12 Hessian
   blocks (only when the Hessian is assembled), the dytopo receiver contributes its 3x3
   *triplet* count, and the reported extent is `(sum of 12x12 blocks) * 16` 3x3 triplets with
   `dof_count = body_count * 12`.  The ABD gradient extent is the *dof count* (the gradient
   buffer is the global one), not the reporters' doublet count.
9. **The per-vertex -> per-body Jacobi maps are the dytopo end's whole content.**
   `_assemble_dytopo_effect` maps the classified effect with `v2b` and the per-vertex `J`:
   `G12 = J_i^T G3` scattered at `body_i * 12`, and for a vertex pair the reference's three
   cases — `body_i < body_j`: `J_i^T H J_j`; `body_i > body_j`: `J_j^T H^T J_i`;
   `body_i == body_j`: `J_i^T H J_j (+ J_j^T H^T J_i)` when `i != j`, one term when `i == j` —
   followed by `zero_out_lower` for the same-body case and a write at the swapped `(L, R)`
   (`L = min(body_i, body_j)`).  The receiver's `do_report` declares the *vertex* range
   `[vertex_offset, vertex_offset + vertex_count)`; with several receivers the manager partitions
   the effect by range (T4.3's coverage), while a **single** receiver takes the manager's fast
   path and receives the collected effect unfiltered.
10. **The inter-affine-body manager's two orderings.**  `Impl::init` sorts the registered
    constitutions by **uid** (`m_index` from that order) and fills two lists from the scene's
    geometries *in scene order*: `inter_geo_infos` (a geometry whose **base**
    `constitution_uid` is one of ours) and `constitution_inter_geo_infos` (base **plus**
    `extra_constitution_uids` hits, counted per constitution).  Both are `stable_sort`ed by
    constitution uid, so same-constitution geometries keep scene order and the offset/count
    slicing is sound (`UIPC_ASSERT(total == size)` at the end of `init`).  The manager is only
    *created* when the scene's constitution table carries the `InterAffineBody` type
    (`SimSystemCreator`), and `SimEngine::init_scene` now initialises it in the reference's
    step-1 order: `affine_body_dynamics -> inter_affine_body_constitution_manager ->
    finite_element_method -> global_body_manager`.
11. **The ABD animator's fan-outs and slice semantics.**  `Impl::init` sorts the constraints by
    uid (assigning `m_index`), associates each with the geometries whose `constraint_uids`
    mention its uid (aborting on an unclaimed uid — the reference's contract), and
    `Impl::step` runs `step` + `report_extent` in the *unsorted* `constraints.view()` order,
    writing the per-constraint energy/gradient/Hessian counts and re-scanning the offset
    tables.  A constraint's `ComputeGradientHessianInfo::gradients()/hessians()` are the
    *slices* `[offset, count)` of the animator's reporter buffer, so the constraint writes at
    slice-relative indices with `(body_i, body_j)` block indices — the same shape the
    `AssembleInfo` slices have.  `GlobalAnimator::step()` is private (only the engine drives
    it), so a test that wants the step fan-out must go through `World::advance()`.
12. **Test-side ordering matters when a system's gate lives in `do_init`.**  A reporter's
    `do_init` runs inside `ABDLinearSubsystem::Impl::init()`, which only runs when the *linear
    subsystem* is initialised (`GlobalLinearSystem::init()` or a test-local
    `GlobalLinearSystem::Impl`), so a test that asserts a reporter's `enabled` flag before
    calling `impl.init()` reads the *previous* value.  T6.1b's test asserts it after.
13. **`find<T>()` returns the registered-but-invalidated object (T5.3's finding, re-measured).**
    With the ABD animator now registered, T6.1a's `abd_base_scene` case had to stop asserting
    `find<AffineBodyAnimator>() == nullptr` and assert the *in-play* question instead
    (`animator == nullptr || !animator->is_valid()`), because the scene has no `Constraint`
    constitution and therefore no `GlobalAnimator` — the animator is invalidated by its own
    `require<>`.  This is the only pre-existing expectation T6.1b moved, and it is a *test*
    update (the file's own comment predicted it).
14. **Reserved uids this batch claims.**  The test-local `AffineBodyConstraint` claims uid **16**
    (`uipc::constitution::SoftTransformConstraint`) and the test-local
    `InterAffineBodyConstitution`/`InterAffineBodyConstraint` claim uid **25**
    (`uipc::constitution::AffineBodyFixedJoint`) — the same knowing reuse as T6.1a's uid-1 mock
    (`ConstitutionTabular::insert` resolves through `ConstitutionUIDCollection::find`, so an
    arbitrary uid is impossible).  T6.4 and T6.3b own the real bodies for 16 and 25 and must
    revisit these claims; the frontend `AffineBodyFixedJoint::create_geometry` is also what
    makes the `InterAffineBody` *type* appear in the scene (the manager's creation gate).
15. **Regression shape of this batch.**  `run` -> **760163 assertions / 153 test cases** (=
    the T6.1a baseline 759668/146 + this file's 495/7) and `run "~[abd_linear]"` ->
    **759668 / 146**, i.e. *no* pre-existing expectation moved (after the one T6.1a test update
    of rule 13, which changes an assertion's *form*, not its count).  `ctest` 1/1 and the
    standalone build are unchanged.
16. **Non-vacuity evidence for the batch.**  See the batch report: the `I * 16` perturbation of
    rule 2 (the COO assertion fails with a *named* assertion and a printed key list), the
    upper-only-convention perturbation of rule 4 (`worst_sym` jumps from 4.3e-16 to 1.19) and
    the `dyt_dbl_idx` perturbation of the device round (a host-memory `kernel_assert` abort).

### 7.23 Patterns established in T6.2 (ABD BDF integrators, states and kinetic)
M6 batch **T6.2** ports `affine_body/bdf/**` — 12 files: `bdf2_function.h` +
`sym/bdf2.inl` (the 12-dof helper templates), `abd_bdf1_time_integrator.{h,cpp}`,
`abd_bdf2_time_integrator.{h,cpp}`, `abd_bdf2_state.{h,cpp}` (the `q_n_1s`/`q_v_n_1s`
history + its dump/recover files), `affine_body_bdf{1,2}_kinetic.{h,cpp}` — plus
`apps/tests/backends/luisa/luisa_abd_dynamics_test.cpp` (tag `[abd_dynamics]`, 7
cases, 547 assertions).  It is a deliberate mirror of T5.3's FEM batch (§7.18); the
rules below are the ones where the ABD sibling is *not* a rename of the FEM one.

1. **The 12-dof BDF laws are the FEM laws with a wider container — and the *test*
   proves it component-wise.**  `sym::abd_bdf2::{compute_q_tilde (both overloads),
   compute_qv}` are `lps::LVec<T,12>` transcriptions of the very statements in
   `finite_element/bdf/sym/bdf2.inl` (same temporaries `c_qn/c_qn_1/c_v/c_g`, same
   association order, `compute_v` -> `compute_qv`); the handwritten reference has no
   generator, so they are transcribed by hand exactly like the FEM file.  Measured:
   driving both templates on the *same* numbers gives **|abd - fem| = 0.000e+00**
   (bitwise) over the first three components, 180/180 samples
   (`abd_dynamics_bdf2_host_oracle`).  The oracle itself is a hand-derived closed form
   (60 random 12-vectors x 3 step sizes, bound 1e-14, measured **0.000e+00** for all
   three templates).
2. **The mass operator is the difference that matters: `M = Jᵀ Mᵢ J` is a dense
   12x12, not the FEM scalar `m I`.**  `K = ½ dq·(M dq)`, `G = M dq`,
   `H = M.to_mat()` where `M` is the per-body `ABDJacobiDyadicMass` aggregate and
   `M dq` is the factored `abd_dyadic_mass_mul` (§7.21 rule 11).  The hand-derived
   `mass_matrix_oracle(m, m x̄, m x̄⊗x̄)` (written out in the test from the jacobian's
   own layout: `block(0,0) = m I`, `block(0,1+i)` non-zero only in row `i` with
   `m x̄ᵀ`, `block(1+i,1+i) = m x̄⊗x̄`) is the oracle: the ported energy/gradient/
   Hessian agree at **0.000e+00** over three hand-written triads, and the pinned
   entries `H(0,0) = m`, `H(0,3) = m x̄(0)`, `H(0,1) = 0` prove the matrix is *not*
   diagonal (the FEM kinetic's Hessian is).  The gradient/Hessian stage writes the
   **dense per-body** buffers T6.1b's corrected `ComputeGradientHessianInfo` hands over
   (§7.22 rule 6) — `gradients->write(i, G12)` / `hessians->write(i, H12)` are the
   reference's own statements — so no device doublet/triplet view has to be rebuilt
   (that is the FEM kernel's idiom), and there is no empty-Hessian-view host guard.
3. **Two kinetic asymmetries that must not be "unified".**  (a) The Hessian is **not**
   guarded by `is_fixed` in either ABD kinetic (only E and G are) — the same finding as
   the FEM batch (§7.18 rule 1): a fixed body still receives `M` on its diagonal.  (b)
   *The ABD energy has an extra guard the FEM one does not*:
   `K = (is_fixed || ext_kinetic) ? 0 : …` while `G = is_fixed ? 0 : M dq` has no
   `ext_kinetic` term.  Both are asserted (the device-vs-host dump pins `entry 122/293
   == 0.0` for an external-kinetic body and its *non-zero* gradient).
4. **The BDF2 predict kernel has no external-force argument — unlike its BDF1 sibling
   and unlike the FEM BDF2 twin.**  The reference's ABD BDF2 kernel takes `gravities()`
   alone (`g = gravity(i)`); BDF1 takes `gravities() + external_force_accs()`.  The port
   keeps that verbatim (`bdf2_detail::predict_dof(q_n, q_n_1, q_v_n, q_v_n_1, gravity,
   dt, is_fixed, is_dynamic)` — one fewer parameter than `bdf1_detail`), because
   "fixing" it would silently change the port's dynamics.  The *device-vs-host* round
   therefore has no external-force entry on the BDF2 side either.
5. **The bootstrap copies the *dynamics'* `q_prevs`/`q_vs`, and the history rolls with
   the old values on the right.**  `ABDBDF2State::q_n_1s`/`q_v_n_1s` are two
   `LDeviceBuffer<LVector12>` (+ their `BufferDump` files `q_n_1.<frame>` /
   `qv_n_1.<frame>`); the first `do_predict_dof` resizes them and deep-copies
   `info.q_prevs()`/`info.q_vs()` (`copy_device` on the *frame* stream, §7.16 rule 4b —
   the reference's `copy_from`).  `_build_on_device` initialises `q_tilde = q_prev = q`
   (§7.21 rule 16's step list, line 822-824), so the *first* BDF2 frame sees
   `q_{n-1} = q_n` and `q_v_{n-1} = q_v_n` and produces
   `q̃ = q + 2/3 h q_v + 4/9 h² g12` — measured in a real scene at rel
   **5.551e-17** — where BDF1 produces `q + h q_v + g12 h²` (rel `0.000e+00`).  The
   update kernel writes `q_v_{n-1} := q_v_n`, `q_v_n := q_v`, `q_{n-1} := q_n` **in the
   reference's order** (all three right-hand sides are read before any write; the port's
   device round pins each with a sentinel).  In a `bdf1` scene the state *system* is
   still built (`do_build` only requires the dynamics) but its buffers stay at size 0.
6. **The steady-state oracle (the plan's headline) and what it actually says.**  BDF2 is
   exact for a *consistent* steady state: with `q_{n-1} = q_n - V h`,
   `q_v_n = q_v_{n-1} = V` and `g = 0`, the shipped BDF2 law gives
   `4/3 q_n - 1/3 (q_n - Vh) + 2/9 h (4V - V) = q_n + V h`, i.e. **exactly BDF1's
   `q_prev + q_v dt`**, and the update `1/(2h)(3(q_n+Vh) - 4q_n + (q_n-Vh)) = V`
   reproduces the velocity.  Measured: `bdf1 rel 0.000e+00`, `bdf2 rel 1.110e-16`,
   `|bdf2 - bdf1| <= 1e-14` (bound 1e-14; one ULP apart because the two sides evaluate
   differently *associated* expressions).  With `g ≠ 0` the two **must** differ, by
   exactly `(1 - 4/9) h² g` — asserted to **1.110e-16** and equal to `2.178e-03` for
   `h = 0.02`, `g = 9.8` — which is what pins the `beta = 4/9` convention against
   BDF1's coefficient 1.  *State* the two `4/9`-vs-`1` coefficients: this equality is
   a property of the steady state, not of the schemes.  The case also asserts the
   transform invariants: the steady-state transform's rotational block is bitwise `A`
   (`|A_after - A| = 0.0`), `|AᵀA - I| = 5.551e-17`,
   `|q_to_transform(q̃)'s orthonormality defect| = 5.551e-17`, the last row of
   `q_v_to_transform_v` is exactly `0.0`, and `transform_to_q ∘ q_to_transform` is exact
   (`0.0`).
7. **Who owns `AffineBodyDynamics`' single kinetic slot is decided by a *name sort*
   (measured, and the reason T6.1b's test file changed).**  `SimEngine::build_systems()`
   sorts the `SimSystemAutoRegister` creators by **demangled type name** and
   `AffineBodyDynamics::add_kinetic()` overwrites the slot ("only one allowed"), so the
   **last system built wins** — and that order is a property of the class *names*, not of
   registration or dependency order.  T6.1b's test-local stand-in (`t61b::T61bKinetic`, a
   *global* namespace) therefore lost the slot to T6.2's
   `uipc::backend::luisa::AffineBodyBDF1Kinetic` ('t' < 'u') the moment the shipped
   batch landed, and `abd_linear_chain` + `abd_linear_animator` lost 8 assertions
   (measured: the device's `A_eff(0,0)` became `127.9 = 125` — the real dyadic mass
   `m = ρL³` — `+ 2.9`, instead of the hand-derived `2.5 + 2.9`, and the gradient's body
   term became `M(q - q̃) = 0` because the dynamics initialises `q̃ = q`).  The fix is
   test-side and one line: the registered system is now
   `zt61b::T61bKineticOwner : t61b::T61bKinetic`, whose name sorts **after** every
   `uipc::backend::luisa::*` class ('z' > 'u'), so the stand-in owns the slot again with
   **no change to any value T6.1b derives**.  Any future test-local kinetic must know
   this rule (a backend-namespace class always beats a globally-namespaced one with a
   name < 'u').
8. **The frontend's state accessor and the state's own buffers (test-side seams).**
   (a) `copy_to` exports `q`/`q_v` — *not* `q_tilde` — through
   `q_to_transform`/`q_v_to_transform_v`, and the *velocity* export **zeroes the last
   row** while the transform keeps `(0,0,0,1)` (a test comparing a read-back velocity
   with a `Matrix4x4::Identity`-based helper fails on entry (3,3); measured 1.0 vs 0.0).
   (b) `copy_from` writes the **whole** state: a test that means "re-seed the velocity
   between frames" must also re-write the transform, because the freshly created
   (default-identity) `builtin::transform` attribute otherwise wipes `qs` (measured: the
   frame-2 `q_tilde` mismatched in 5 of 12 components).  (c) `ABDBDF2State::q_n_1s()` /
   `q_v_n_1s()` have a **public const** overload and a **private non-const** one (the
   reference's layout: the mutable views belong to the integrator), so a test must read
   them through a `const` pointer (the FEM batch's `state_x_n_1s` helper).
9. **Test-side traps this batch measured** (all cheap to avoid next time):
   * an Eigen comma initializer **inside a braced init list** is mis-parsed by MSVC
     (`Matrix3x3() << …` in a `std::vector<Matrix3x3>{…}` initializer → C3329/C2059); use
     a nine-argument helper (`m3(...)`);
   * a ternary mixing `Vector12::Zero()` (an Eigen *nullary expression*) with a matrix
     product has no common type (C2446) — write it as two statements (§7.21 rule 10's
     trap, again);
   * `lps` **views** (`.diagonal()`, `MatMap`) have no member `norm()`/`squaredNorm()`;
     compare entry-wise or use the owning `LVec`/`LMat` (which do);
   * `AffineBodyDynamics::body_count()` is **not** a public accessor on the system (it
     exists on the *constitution-scoped* info and on `Impl`); use `qs().size()` in a
     test;
   * with both `using namespace uipc::constitution;` and
     `using namespace uipc::backend::luisa;` open, `AffineBodyConstitution` is ambiguous
     (C2872) — qualify the frontend one;
   * a `||`/`&&` of two comparisons inside one Catch2 `CHECK` is a *chained comparison*
     (C2338): materialise the `bool` first.
10. **Regression shape of this batch.**  `run` -> **760710 assertions / 160 test cases**
    (= the T6.1b baseline 760163/153 + this file's 547/7), `run "~[abd_dynamics]"` ->
    **760163 / 153** = the T6.1b baseline *exactly*, `ctest` **1/1 passed** (20.7 s,
    label gpu) and the standalone build (69 cases, 0 checks failed) unchanged.  The
    device-vs-host round of every ported routine is **462/462 bitwise equal**
    (worst abs `0.000e+00`).  Non-vacuity (3 perturbations, each patched -> built ->
    run -> reverted -> green): `beta := 1/3` fails the 2 literal checks and moves the
    measured `K_bdf2/K_bdf1` from 2.25 to 3.00; the BDF1 predictor `dt*dt -> dt` fails
    7 assertions across 3 cases (worst rel 1.83 on the scalar law); removing the
    kinetic's `ext_kinetic` guard fails 3 (`worst_k`, `ΣK1` and the dump's `entry 122`).

### 7.24 Patterns established in T6.3a (ABD constitutions: arap, ortho_potential)

M6 batch **T6.3a** ports the two ABD constitutions into the new directory
`src/backends/luisa/affine_body/constitutions/`: `arap.{h,cpp}` +
`arap_function.h` (uid 2), `ortho_potential.{h,cpp}` + `ortho_potential_function.h` +
the generated `sym/ortho_potential.inl` (uid 1) — 7 files — plus the transcriber
`scratch/gen_abd_cons_a.py`, the test
`apps/tests/backends/luisa/luisa_abd_constitution_test.cpp` (tag `[abd_constitution]`,
7 cases, 523 assertions) and **two test-side uid updates** in older files (rule 9).
The rules below are what the port turned out to depend on.

1. **The ABD constitution interface is *not* the FEM one: there are no
   `report_extent`/`assemble` hooks.**  An ABD constitution only *computes*
   energy/gradient/Hessian into the dense per-body windows the line-search
   (`abd_line_search_reporter.h`, T6.4's) and linear-subsystem
   (`ABDLinearSubsystem::ComputeGradientHessianInfo`, T6.1b's) infos hand over
   (§7.22 rule 6); every extent, block index and COO convention belongs to
   `ABDLinearSubsystem`.  A batch-brief sentence listing "report_extent / assemble"
   among "the reporter hooks" is therefore *not* a file list: read the reference's
   `affine_body_constitution.h`, whose four hooks are
   `do_build`/`do_init`/`do_compute_energy`/`do_compute_gradient_hessian`, and port
   exactly those.  Consequence for the test: the *assembly* half is driven through a
   test-local `GlobalLinearSystem::Impl` (§7.22 rule 12) and the energy hook is
   unreachable (rule 8).
2. **The ABD local variables and the two energy laws** (`q = [t; a1; a2; a3]`,
   `a_i` = *row* `i` of the 3x3 linear part `A`, so `A(i,j) = q(3 + 3i + j)`;
   §7.21 rule 6):
   * ARAP: `E = kappa ||A - R||_F^2` with `R = U V^T` from the signed SVD, i.e. the
     **dimensionless** density.  The kernel multiplies by `volume * dt^2` on the
     *outside* (`E * Vdt2`, `G9 * Vdt2`, `H9 * Vdt2`) — the FEM family puts `v` (and
     the `dt^2` as `kappa dt^2`) *inside* the local law (§7.17 rule 8).  Do not
     "unify" them; the ABD laws read `A` only, so they are exactly
     translation-invariant, which the test asserts (`|dG| = 0`, `|dH| = 0`).
   * ortho potential: the *generated* law is `E = kappa ||A^T A - I||_F^2`: the three
     `(|a_i|^2 - 1)^2` terms are the diagonal of `C = A^T A - I` and the three
     `2 (...)^2` terms its two off-diagonal entries; `dE/dA = 4 kappa A C`
     (verified against the generated *expression* below).  It is the reference's
     `abd_energy.cu` ("perpendicular volume") times `kappa` — the T6.1a analytic
     sibling `gg::shape_energy`/`_gradient`/`_hessian` is an *independent* oracle for
     the whole constitution (measured rel `1e-14 / 1e-13 / 1e-13` in the test, and
     `1.4e-17` for the hand-derived 9x9 against the generated `ddEddq` inside the FD
     case).  Use it.
3. **`sym/ortho_potential.inl` is generated by `scratch/gen_abd_cons_a.py`, which
   *imports* the T5.2 transcriber** (`import gen_fem_cons_a as g`, the §7.17 rule 14
   precedent) and supplies only the ABD `REF_DIR`/`DST_DIR` + its region list — one
   region, `abd_ortho_potential`, with the three reference functions `E`, `dEdq`,
   `ddEddq` (1 / 15 / 177 statements, all scalar).  `--check` is idempotent
   (`ok  sym/ortho_potential.inl <- sym/ortho_potential.inl (E, dEdq, ddEddq)`),
   `--check --selfcheck` re-parses every generated statement (tree identity after
   un-transforming) *and* is value-exact under 200 random inputs per scalar
   statement.  The generated file is the same shape as the FEM `sym/*.inl` (no
   includes allowed, `lps::detail::{add,sub,mul,div}`, `std::pow -> lps::pow`,
   integer literals -> double), so the two families cannot drift apart.
   **The expression-level oracle first** (§7.17 rule 30c): before writing any C++,
   the generated `E`/`dEdq`/`ddEddq` were re-parsed and evaluated in Python against
   the hand-derived `E = kappa ||C||^2`, `G = 4 kappa A C` and
   `H[(i,j),(r,s)] = 4 kappa [delta_ir C(s,j) + A(i,s) A(r,j) + (A A^T)(i,r) delta_js]`
   (the Hessian derived by differentiating `4 kappa A C` along one elementary
   direction): **2.5e-16 / 3.3e-16 / 4.3e-16** over 20 random states.  It also
   settled a *wrong* first guess (the symmetric-part form `2 kappa ((AC) + (AC)^T)`)
   in one run: with a generated body, check the expression, never the memory.
4. **`make_spd` is the ortho potential's, and only its.**  The reference's
   `arap.cu` includes `utils/make_spd.h` and never calls it (measured: the only call
   sites under `affine_body/constitutions/` are `ortho_potential.cu:66` and the four
   joints), so the port **drops the vestigial include and ships the unclamped exact
   ARAP Hessian**.  The clamp itself is applied in the *9x9 F-block*, **before** the
   `Vdt2` scaling and the 12x12 embedding (the reference's own order) — clamping the
   assembled 12x12 afterwards is a different (rounded, 12x12-Jacobi) computation.
   Measured in the test over 40 randomized states: the shipped 9x9 equals the exact
   Eigen clamp (`max(lambda,0)` + reconstruct) to **6.1e-15** while **29/40** states
   are indefinite *before* the clamp (so it is not a no-op); the port's local
   `kMakeSpdSweeps = 8` (§7.17 rule 31's measured plateau for N = 9) agrees with the
   layer default (20) to **0.0** — 8 sweeps are enough for this spectrum.  The ARAP
   side is pinned the other way: the shipped Hessian has `min eig = -2 kappa v dt^2`
   at the isotropic stretch `s = 0.5` and `-6.0e-5` at the singular state
   `diag(1,1,0)` — unclamped, indefinite, *as the reference ships it*.
5. **FINDING — the reference's ABD ARAP Hessian is spelled in the wrong basis
   relative to its own gradient (and the port keeps it).**  `dEdq` writes
   `gradients(i*3 + j) = kappa * dPsi_dF(i,j)`, i.e. the *dof* order
   (`q(3 + 3i + j) = A(i,j)`, row-major), while `ARAP_Hessian` builds its 9x9 from
   `vec(T)`, which is **column-major**.  The twist-mode rank-1 terms are invariant
   under the transpose permutation `P` only when the modes are antisymmetric, i.e.
   when `U == V`; in general `H` is the `P`-conjugate of the true Hessian
   (`P` = `transpose_permutation()`, `P vec(M) = vec(M^T)`), so the shipped
   `(g, H)` pair is *inconsistent* by a permutation.  Measured (numpy, then in the
   C++ test): `|H - dG/dq| = 7.4e-03` (unpermuted, ~4-40 % of `|H|`) versus
   `|H - P (dG/dq) P| = 2.3e-12` (the FD truncation level).  The **FEM** sibling is
   self-consistent — its `dEdF` writes `gradients.segment<3>(k) = dPsi_dF.col(k)`,
   which *is* the column-major order `vec` uses — so this is an ABD-only reference
   quirk, not a port bug.  A port must **not** "fix" it (§7.17 rule 34); the test
   asserts *both* the conjugated agreement and the measurable plain deviation, so a
   future reader cannot mistake it for a rounding artefact.  The generated ortho
   potential is self-consistent (`m = 3i + j` in both files), which the same case
   proves (ortho FD grad `1.4e-12` / Hessian `8.2e-12` unpermuted).
6. **The FD calibration of the two ABD families** (40 randomized admissible states
   each: `A = R (I + S)` with `S` symmetric, `sigma` in `[0.35, 2.5]`, `kappa`,
   `volume`, `dt` random — §7.17 rule 32's "split the Hessian check" applies):
   * ARAP (after the rule-5 conjugation): gradient **1.2e-12**, Hessian
     **2.3e-12** (h = 1e-6), **8.0e-12** (1e-4), **8.0e-10** (1e-3), **8.0e-8**
     (1e-2) — the truncation rises with `h`, so the sweep is not flat; asymmetry
     **0.0** (bitwise: the Hessian is `2 I - sum c t t^T`, i.e. exactly symmetric
     outer products); **34/40** states indefinite.
   * ortho: gradient **1.4e-12**, Hessian **8.2e-12** / **2.3e-10** / **2.3e-8** /
     **2.3e-6**; asymmetry **1.5e-16** (the *generated* statements are symmetric
     entry-by-entry — `R(0,1)` and `R(1,0)` are the *same* temporary — but the
     clamped `V D V^T` reconstruction is only symmetric to round-off, so assert
     ~1e-15 here and `0.0` only for ARAP).
   The test's assertions are `<= 1e-8` (h = 1e-6) and `<= 1e-6` (h = 1e-4) for both
   Hessians and `<= 1e-9` for both gradients — an order of magnitude above the
   measured worst in both the filtered and the full RNG stream (§7.17 rule 29).
7. **Degenerate inputs, pinned from the source** (neither constitution guards its
   input; the only guard is `lps::svd`'s own rank-loss completion, `svd.h`'s
   `normalized_completion` + `axis_orthogonal_to`):
   * ARAP, `A = 0`: `R` comes out as *some* rotation (the completion), so
     `E = kappa 3 v dt^2` and `G = -2 kappa R v dt^2` are finite; `ARAP_Hessian`
     divides by `sigma_0 + sigma_1 = 0` -> the Hessian is **not finite** (measured
     both ways in the test).
   * ARAP, `diag(1,1,0)` (singular): the source's branch gives `U = V = I` (every
     `gamma` is 0, so no rotation happens), so `R = I` exactly and
     `E = kappa v dt^2`, `dE/dA = -2 kappa e_2 e_2^T`, and every denominator is
     positive (`2`, `1`, `1`) -> **E/G/H are all finite**, but the Hessian is
     indefinite (`min eig = -2 kappa v dt^2`) and unclamped.
   * ARAP, a reflection `diag(-1,1,1)` (`det < 0`): the signed convention puts the
     sign in `sigma_2`, giving `U = diag(-1,1,-1)`, `V = I`, `R = diag(-1,1,-1)`,
     hence `E = 4 kappa v dt^2` and `G(11) = +4 kappa v dt^2` (finite, exact), while
     `sigma_1 + sigma_2 = 0` makes the Hessian non-finite.
   * ortho: a polynomial in the nine dofs (no division anywhere in the generated
     file), so `E/G/H` are finite in *every* state; the exact values are
     hand-derivable: `A = 0` -> `E = 3 kappa v dt^2`, `G = 0`,
     `H_raw = -4 kappa v dt^2 I` (negative definite — the reason the reference
     clamps, and the shipped one is then exactly `0`, `|H_shipped| = 0.0`);
     `diag(1,1,0)` -> `E = kappa v dt^2`, `G = 0`; an *orthogonal* `A` (any
     orientation, **including a reflection**) -> `E = 0` and `G = 0` exactly, because
     the potential measures the orthogonality defect only.  A zero *volume* zeroes all
     three kernel quantities while the local law stays `3 kappa` — assert the value
     the kernel returns, not the density (§7.17 rule 33).
8. **The energy hook is unreachable from a test in this batch; the gradient/Hessian
   hook is reachable.**  `AffineBodyConstitution::do_compute_energy` takes a
   `ComputeEnergyInfo` that wraps `ABDLineSearchReporter::ComputeEnergyInfo`, whose
   `energies` view is written only by `ABDLineSearchReporter` (a `friend`) — and that
   system is **T6.4's** (it has no `SimSystemCreator` here, so no instance exists to
   build the info).  `do_compute_gradient_hessian` needs no such friend: its
   `ABDLinearSubsystem::ComputeGradientHessianInfo` has a public constructor, so
   `ABDLinearSubsystem::Impl::init()` + `_assemble_linear_system()` drives the real
   kernels end to end (measured: the assembled shape block equals the hand-derived
   one to **3.3e-11** and the assembled gradient to **1.4e-15**).  The **energy law**
   is therefore validated through the same generic routine the kernel body calls
   (`abd_*_detail::body_energy`) plus the closed forms of cases 1-3, and this is
   recorded as a deviation, not papered over.
9. **A uid collision is resolved by *moving the claiming mock*, and the reason is a
   name-sort plus a last-one-wins map.**  `AffineBodyDynamics::Impl::_build_constitutions`
   sorts the registered constitutions by uid and fills
   `constitution_uid_to_index[uid] = i` in a loop, so with two equal-uid entries the
   **last** one owns the geometry group; the registration order is
   `SimEngine::build_systems()`'s **demangled-type-name** order (§7.23 rule 7), and
   every `uipc::backend::luisa::*` class sorts *after* a globally-namespaced test class
   with a name < 'u'.  T6.1a's mock (`T61ABDMockConstitution`) claiming uid 1 therefore
   loses the group to the real `OrthoPotential` the moment T6.3a lands — and would then
   read an empty group (its `rec().body_count` would be 0, i.e. T6.1a's scene case
   would fail *and* T6.1b's chain oracle would gain a shape term it was never written
   for).  Measured and resolved:
   * the mock moved to uid **8** — the last frontend-registered `AffineBody`-type uid
     (`src/constitution/affine_body_constitution.cpp` registers 1..8) and grep-proven
     unused by every backend system and every test — and each of that file's scene
     geometries is relabelled with a 5-line `claim_mock_uid(sc)` helper;
   * the same helper was applied to `luisa_abd_linear_test.cpp`'s scene bodies, whose
     oracle is the *assembly* and therefore needs the constitution term to stay
     exactly zero (the mock's hook writes nothing);
   * the new file does not *assume* the resolution: the assembly and scene cases prove
     ownership (the real constitution's `do_init` ran with exactly the scene's bodies —
     its `kappas` buffer holds one entry per body — and the assembled block equals the
     file's own hand-derived contribution), print a named owner line, and would fail
     with that line above the assertion rather than silently weakening it.
   Cost of the move: **zero** — the exclusion run reproduces the T6.2 baseline
   `760710/160` exactly (both files' assertion counts and values are unchanged).
   The reserved test uid space after this batch: **8** (T6.1a's mock), 16/25 (T6.1b's
   constraint/inter-body stubs), 666 (the ABD external-force manager), 671 (the FEM
   manager).  T6.3b/T6.4 must revisit 8 if they need it.
10. **The dof offsets are written by a *linear system's* init, not by `world.init()`.**
   `builtin::dof_offset`/`dof_count` are filled by
   `ABDLinearSubsystem::Impl::receive_init_dof_info` (driven from
   `GlobalLinearSystem::init()`), so a test that addresses bodies by their dof offset
   must first drive a *throwaway* test-local `GlobalLinearSystem::Impl`
   (`register_sim_system(*abd_linear); impl.init();`) — before that the attributes are
   still the frontend's `-1` (measured: `dof_offset/12` came out as `1.5e18`).
   Measured layout of the 2-body scene: uid 1 first (body 0, dofs 0..11), uid 2 second
   (body 1, 12..23) — `_build_geo_infos` *stable-sorts* the geo infos by
   `constitution_uid`, so the body order is the uid order, **not** the scene creation
   order.
11. **The ABD assembly's two conventions, restated for a constitution oracle.**
   The body term contributes one 12x12 block per body, unpacked into **16** 3x3
   triplets `(body*4 + a, body*4 + b)` in `counter = a*4 + b` order, with the diagonal
   block *upper-triangular in 3x3 blocks* (`zero_out_lower`: the strictly-lower `a > b`
   sub-blocks are present with value `0`, so the key set is all 16 per body while the
   assembled operator is only block-symmetric) — the oracle must therefore write the
   zeroed lower blocks too, at the *blocked* global indices
   (`body = dof_offset/12`).  Measured: 32 triplets / 32 distinct keys for two bodies,
   worst `|entry|` **3.27e-11** and `|block|` **4.51e-11** against the hand-derived
   dense map, gradient rel **1.39e-15** (`|b| = 4.98e3`) — and the *clamped* ortho
   Hessian is the right oracle: using the unclamped law is off by the clamp's own size,
   measured **1.64e+02** for a `kappa = 1e8` body (a "constant ratio" symptom, §7.17
   rule 26, seen from the oracle's side again).
12. **A one-`advance()` scene leaves the ABD *state* where it was; the integrator's
   output lands in `q_tilde`.**  The update path is T8.3's `solve()` + T6.4's line
   search, so after `advance()` the state `q` is still the scene's initial state while
   `abd.q_tildes()` holds `q + q_v dt + g dt^2` (measured: with a zero-gravity config
   and an isotropic `q_v = 0.5 I`, the predicted stretch is `1.4 + 0.5 dt = 1.405`
   exactly, and the state stays `1.4`).  A scene case that wants a *changed* state must
   therefore read `q_tildes` (or wait for T8.3); one that wants the *state's* energy
   must read `qs`.  A `bdf1` scene config with `gravity = 0` keeps the prediction a
   pure `q + q_v dt`, which is what makes a closed-form Hessian survive an `advance()`.
13. **`uipc::constitution::AffineBodyConstitution::apply_to` throws on a
   non-positive volume — check the *orientation* of the test's tetrahedralisation.**
   A six-tet cube decomposition copied from a file that uses a different vertex
   numbering gives three inverted tets and `compute_mesh_volume` returns `-0.0208`,
   and the frontend aborts with `Assertion volume > 0 failed` (measured, twice: both
   scene cases died before their first assertion).  Use the fan
   `{{0,1,3,7},{0,3,2,7},{0,2,6,7},{0,6,4,7},{0,4,5,7},{0,5,1,7}}` with the
   `x + 2y + 4z` vertex order that the other ABD test files use.
14. **Assert scale-relative bounds, not `== 0.0`, for SVD-derived zeros, and read
   Catch2's expansion carefully.**  `E` and `G` at a *rigid* map are zero only to the
   iterative SVD's round-off (measured `|E| = 4.2e-35` against a scale of `3.8e-4`,
   `|G|max = 1.1e-19` against `2.5e-4`), and Catch2 prints such values as `0.0` — so
   `CHECK(x == 0.0)` fails with the baffling expansion `0.0 == 0.0`.  Assert
   `|x| <= 1e-13 * scale`.  The mirror-image trap is the *helper*: a `dev(a,b)`
   normalising by `max(1, |a|, |b|)` silently becomes an *absolute* bound whenever the
   quantities are smaller than 1 (all of this batch's energies are `O(1e-4)`), which
   is why the clamp-identity checks in this batch are absolute by construction and
   their bound is `1e-13` rather than a "relative" one.
15. **The device-vs-host round of this batch is *not* bitwise — and that is the
   routine's transcendentals, not the module** (§7.17 rule 34).  The generic routines
   are the same statements, but the DSL lowers the `sqrt` inside `lps::svd` and the
   `pow` inside the generated ortho expression to `lcrt::*`: measured over 16 random
   states, ARAP `E/G/H` differ from the host by `1.1e-19 / 5.6e-19 / 6.5e-19`
   absolute and `1.7e-15 / 2.9e-15 / 8.0e-16` relative; ortho
   `5.4e-20 / 1.1e-19 / 5.6e-18` absolute and `2.5e-16 / 1.7e-16 / 1.3e-15` relative
   (bound asserted: `1e-12` relative).  T6.1a's `779/779 bitwise` does **not** transfer
   to a routine that calls the SVD — assert a per-routine bound and record it.
16. **Regression shape of this batch.**  `run "[abd_constitution]"` -> **523
   assertions in 7 test cases**, rc 0 — per case: `abd_constitution_closed_forms` 217,
   `abd_constitution_finite_difference` 16, `abd_constitution_degenerate_inputs` 76,
   `abd_constitution_assembly` 90, `abd_constitution_scene` 16,
   `abd_constitution_device_vs_host` 2, `abd_constitution_summary` (the accounting
   case: 6 cases, 417 `T63A_CHECK`s; the difference to 523 is the `REQUIRE`s).
   `run` -> **761233 assertions in 167 test cases** (= the T6.2 baseline 760710/160 +
   this file's 523/7), rc 0; `run "~[abd_constitution]"` -> **760710 assertions in 160
   test cases** = the T6.2 baseline *exactly*, rc 0 (i.e. the two uid updates of rule 9
   moved no expectation); `ctest` -> **1/1 passed** (`uipc.backend_luisa`, 21.52 s,
   label gpu), rc 0; `python scripts/luisa_standalone.py cuda` -> **69 cases, 0 checks
   failed**, rc 0.  **Cold-JIT cost:** the *first* filtered run took **2 m 18 s** (both
   constitutions' energy/gradient/Hessian shaders, including the ortho's inlined
   8-sweep `make_spd`), every later run ~5 s — i.e. ~30x cheaper than T5.2a's FEM 9x9
   ARAP (`~16 min`), because the ABD ARAP has no `evd` and the ortho clamps with 8
   sweeps instead of the layer's 20.
17. **Non-vacuity evidence for the batch** (brief §2.7, three perturbations, each
   patched -> built -> run -> reverted -> green; driver `scratch/t63a_perturb.py`,
   log `scratch/t63a_perturb.log`):
   * the ARAP `dEdq` coefficient `kappa -> 2 kappa` fails **23** assertions (the
     closed-form gradient rows, the FD gradient/Hessian, the degenerate `G(11)` pins,
     the assembly gradient);
   * a *generated* exponent, `pow(q(3), 2.0) -> pow(q(3), 3.0)` inside
     `sym/ortho_potential.inl`, fails **12** (the ortho closed-form gradient, the
     zero-gradient pins, the ortho FD gradient/Hessian, the assembly block);
   * `kMakeSpdSweeps 8 -> 2` fails **5** (the clamp identity `shipped == exact clamp`,
     its FD counterpart, the 8-vs-20 sweep agreement and the assembly block).
   After the reverts the same command is green again (523 assertions, rc 0) and the
   three files are byte-identical to the tested ones (the driver checks that too).

### 7.25 Patterns established in T6.3b1 (ABD joints, batch 1: fixed + spherical)

M6 batch **T6.3b1** ports the first two of the four ABD *joints* into
`src/backends/luisa/affine_body/constitutions/`: `affine_body_fixed_joint.{h,cpp}`
+ `affine_body_fixed_joint_function.h` + the generated
`sym/affine_body_fixed_joint.inl` (uid **25**) and the spherical joint's five
siblings (uid **26**), plus the generator regions and the test
`apps/tests/backends/luisa/luisa_abd_joint_a_test.cpp` (tag `[abd_joint_a]`, 8 cases,
223 assertions) — nine new files, no edits to any existing backend file except the
one additive type (`LVector6` in the global `type_define.h`, rule 1's note below).
`scratch/gen_abd_cons_a.py` grew two regions and the independent expression-level
(oracle) is `scratch/t63b1_oracle.py`.  Sub-milestone
`T6.3b2` (`T6.3b` was split) owes prismatic uid 20 + revolute uid 18 with their dof
reporters, time integrators, driving constraints (21/19), external-force constraints
(667/668), the two joint limits (669/670) and the remaining five `sym/*.inl`.
The rules below are the ones where a joint is *not* just another constitution.

1. **A joint is a plain `InterAffineBodyConstitution` with a *pair* fan-out, and the
   whole file split of T6.3a carries over unchanged.**  Four hooks
   (`do_build`/`do_init`/`do_report_*_extent`/`do_compute_*`), the class in the
   header, the kernels + `REGISTER_SIM_SYSTEM` in the `.cpp`, the arithmetic in
   `*_function.h` as `abd_{fixed,spherical}_joint_detail::body_{energy,gradient,hessian}`
   over the scalar `S`, and the `sym/*.inl` transcribed by the generator.  What is
   *new* is only the shape of the fan-out: per joint the reference reports
   `gradient_count(2 * n)` (a **two-body doublet**, `DVA.segment<2>(2 * I).write({bids0,
   bids1}, G24)`) and `hessian_count(HalfHessianSize * n)` with
   `HalfHessianSize = 2 * (2 + 1) / 2 == 3` (the **upper triangle of a 2x2 block of
   `TMA.half_block<2>(3 * I).write(indices, H24)`), i.e. 2 doublets and
   3 triplets per joint — not the constitutions' one-block-per-body.  The batch's one
   *additive* edit outside its nine files is the joint's device element type:
   `LVector6 = lps::LVec<Float,6>` (+ `to_luisa`/`to_eigen`/`load_vec6`) in
   `src/backends/luisa/type_define.h` — the reference's `DeviceBuffer<Vector6>` rest
   rows have no home without it, and the port had no 6-vector element type.

2. **The two joint laws, and the frames the reference stores for them.**  With the
   pair state `q = [ti ; A_i ; tj ; A_j]` (12 dofs each, `A(i,j) = q(3 + 3i + j)`, and
   note the *pair* layout: body i occupies 0..11 and body j 12..23, so **`A_j` starts
   at 15** — `q(12..14)` is `tj`; getting that off by three is rule 9):
   `Ft = A_i ci_bar + ti - (A_j cj_bar + tj)` (3-vector, the *ball* constraint) and
   `Fr = [A_i t - A_j t_j ; A_i n - A_j n_j ; A_i b - A_j b_j]` (9-vector, the frame
   mismatch), with `E = 1/2 kappa (||Ft||^2 + ||Fr||^2)`; the **spherical** joint keeps
   `Ft` alone.  The rest data is the frontend's per-vertex convention:
   `rest_cs = [ci_bar | cj_bar]` (the two *body-local* attachment points, taken
   verbatim from the `l_position`/`r_position` vertex attributes when the frontend
   wrote them, otherwise `LT.inverse() * Ps[i]` from the scene vertex) and
   `rest_ts`/`rest_ns`/`rest_bs` = `[L_i^{-1} e_x | L_j^{-1} e_x]` etc.  The port keeps
   the reference's member names, its `list<>`/`move_to`/`copy_from` staging, and the
   one non-obvious aliasing: `LT.rotation()` on an
   `Eigen::Transform<Float,3,Eigen::Affine>` is **`computeRotationScaling()`**, i.e. a
   *polar decomposition* (rule 3).

3. **FINDING — the joint's rest frames come from the *polar* rotation of the body's
   initial transform, not from its linear part.**  Eigen's `Transform<Mode>::rotation()`
   is `linear()` only for `Mode == Isometry`; for `Affine` it
   `computeRotationScaling`s, so a *stretched* initial state still contributes a
   non-zero rotation term at the scene's own state.  Measured on this batch's scene
   (body 0's transform = `R(0.6, axis) * diag(1.25, 0.9, 1.1)`):
   the kernel's `rest_t` came out `0.83706680 -0.28299237 -0.46822482` = the polar
   value while the linear-part value is `0.66965344 -0.31443597 -0.42565893`; with the
   polar frames the joint's fixed energy reproduces the kernel's `1.11354993e+04`
   exactly (the linear-part convention gives `8.79784244e+03`, i.e. `Er == 0`).
   A *test oracle* must therefore spell the frame source as `U V^T` of the SVD (the
   test's `polar_rotation`), which is what the scene case's
   `h_rest_*` comparison (`worst |rest - hand| = 0.000e+00`) pins.  Do **not**
   "fix" the reference's `LT.rotation()`.

4. **The joint Hessians are *constant* (the laws are quadratic), so the FD sweep has
   no truncation error and the round-off falls with h.**  `G = kappa (J_t^T Ft + J_r^T Fr)`
   with *constant* Jacobians, so `H = kappa (J_t^T J_t + J_r^T J_r)` does not depend on
   the state at all.  Measured over 40 states: gradient FD `2.1e-10` (fixed and
   spherical), Hessian `9.8e-11` (h = 1e-6), `9.8e-13` (1e-4), `1.1e-13` (1e-3),
   `8.7e-15` (1e-2) — the *opposite* of the constitutive families' growth-with-h and
   the reason the batch asserts `worst_h(1e-2) <= worst_h(1e-6)` instead of "the sweep
   rises"; asymmetry `2.1e-16` (fixed) / `0.0` (spherical: `J_t^T J_t` is an exactly
   symmetric rank-3 fan-out).  The test also pins the state-independence directly
   (`H(q) == H(q')` bitwise for two random states) — none of the constitutive
   families can say that.

5. **The rule-34 basis question, answered *negative* for these two laws (and pinned as
   such).**  Unlike the ABD ARAP (§7.24 rule 5, whose shipped Hessian is the
   `transpose_permutation()`-conjugate of its own `dEdq` derivative), the joints'
   generated fan-outs take their `H9x9`/`Ht3x3` in the **same** `Fr`/`Ft` order their
   `dEdF` siblings write, so the *plain* central-difference comparison is the
   agreement — measured `9.8e-11`/`1.3e-10` at h = 1e-6 with **no** permutation.  The
   case also re-derives the closed form from the hand-built Jacobians
   (`kappa (J_t^T J_t + J_r^T J_r)`, worst `1.4e-17` spherical) so the reader can see
   that the two statements are the same basis, not a rounding coincidence.

6. **`make_spd` is applied to both *element* Hessians, 9x9 then 3x3, *before* their
   fan-outs — a no-op for `kappa > 0`, the whole block for `kappa < 0`.**  The
   reference's `fixed_joint.cu` calls `make_spd(ddEdFr_val)` and
   `make_spd(ddEdFt_val)` (and the spherical one `make_spd(ddEdFt_val)`), and both
   element Hessians are exactly `kappa I` — so the clamp is invisible until a user
   passes a negative `strength_ratio`, where the shipped `H` collapses to `0` while
   `E` and `G` keep their unclamped (negative-energy) values.  Measured: `kappa = -3`,
   `max|H| = 0.0` while `|G| = 2.139e+00` and `E = -2.259e+00`; and the shipped H is
   *not* `dG/dq` there (deviation `1.000e+00`, asserted so it cannot be mistaken for
   round-off).  The port keeps `kMakeSpdSweeps = 8` (§7.17 rule 31's plateau) and the
   test measures it against the layer default (20 sweeps) → `0.0` difference.

7. **The degenerates are *exact* zeros, unlike the SVD families'.**  The joint laws
   contain no division and no SVD, so `E`, `G` and `H` are finite in **every** state
   (including a reflection: `E = 1.12125`, `|G| = 2.25`, all finite, from the same
   polynomial) and a coincident rest pair gives `E == 0.0` **and** `max|G| == 0.0`
   *bitwise*.  Two pins that are easy to get wrong by hand:
   * `A_i = A_j = 0` does **not** zero the `A`-blocks of `G`: `dFt(k)/dA_i(k,m) =
     ci_bar(m)` gives `(J_t^T Ft)(3 + 3k + m) = ci_bar(m) Ft(k)`, i.e. a
     `ci_bar (x) Ft` outer product (and `-cj_bar (x) Ft` on the right body) — measured
     `worst dev 0.000e+00` against that closed form, `|G_Ab| = 1.2`;
   * `kappa = 0` zeroes everything including the clamped `H` (the element Hessians are
     the zero matrix).
   A 1e12 coefficient with 1e-6 rest frames stays finite: there is no conditioning
   cliff to pin.

8. **The generator regions are the *whole* `.inl`, not the "interesting" functions.**
   The T6.3b brief listed three function names per joint (`Fr, JrT_Gr, JrT_Hr_Jr` /
   `Ft, JtT_Gt, JtT_Ht_Jt`), but the transcriber's contract is *one region == one
   sentinel pair == every generated function in it* — the trivial energy/derivative
   laws (`Er, dEdFr, ddEdFr, Et, dEdFt, ddEdFt`) are SymEigen output too and must be
   transcribed beside the fan-outs.  The two `REGIONS` entries therefore list **12** and
   **6** names.  Per-region statement counts (all scalar):
   fixed `Fr 9, JrT_Gr 24, JrT_Hr_Jr 738, Er 1, dEdFr 9, ddEdFr 81, Ft 3, JtT_Gt 24,
   JtT_Ht_Jt 834, Et 1, dEdFt 4, ddEdFt 10` (1738 total); spherical `Ft 3, JtT_Gt 24,
   JtT_Ht_Jt 834, Et 1, dEdFt 4, ddEdFt 10` (876).  `--check --selfcheck` is rc 0 and
   idempotent (tree identity after un-transforming, plus 200 random scalar inputs per
   scalar statement).

9. **The *pair* dof layout is the new "off by one" trap: `A_j` starts at 15, not 12.**
   `q = [ti(0..2); A_i(3..11); tj(12..14); A_j(15..23)]`.  The hand-derived Jacobians
   are
   `J_t(k, k) = 1, J_t(k, 12 + k) = -1, J_t(k, 3 + 3k + m) = ci_bar(m),
   J_t(k, 15 + 3k + m) = -cj_bar(m)` and
   `J_r(3a + k, 3 + 3k + m) = (d_a)_m, J_r(3a + k, 15 + 3k + m) = -(e_a)_m`
   (`d_a` = t/n/b, `e_a` = t_j/n_j/b_j; the row index `3a + k` is the group/component
   pair and the *column* is the `A`-*row* `k` — that is what makes the generated
   `JrT_Gr`'s `R(3) = G(0) t(0) + G(3) n(0) + G(6) b(0)` come out right).  Writing
   `12 + 3k + m` (the *body*-local offset instead of the pair offset) produced a
   *silently plausible* wrong oracle — the E comparison still passed (E does not use
   the Jacobians) while `G` was off by `1.17` and `H` by `0.52`; the two minutes spent
   comparing the hand Jacobians against a finite-difference Jacobian of the
   hand-derived law in Python (`scratch/t63b1_jac.py`) is the cheapest way to catch it.

10. **The expression-level oracle (before any C++), and what it settled.**  The two
   generated bodies were re-parsed and evaluated numerically (`scratch/t63b1_oracle.py`,
   the §7.17 rule 30c technique) against hand-derived laws that were written out
   independently *there* — worst relative residuals over 20 random states:
   `Fr 3.3e-16`, `Ft 2.6e-16`, `Er 2.2e-16`, `Et 2.2e-16`, `dEdFr/dEdFt/ddEdFr/ddEdFt
   0.0` (bitwise), `JrT_Gr 1.5e-10`, `JtT_Gt 2.1e-10`, `JrT_Hr_Jr 2.5e-10`,
   `JtT_Ht_Jt 2.8e-10` — the fan-out residuals are the *FD Jacobian's* truncation
   (`h = 1e-6`), not the transcription's.  The four fan-outs are compared against
   `J^T G` / `J^T H J` built from a **central-difference Jacobian of the hand-derived
   law**, so the *structure* of the fan-out (not only the scalar laws) is pinned
   before a line of C++ exists.  This is the batch's non-vacuity proof beside the
   generator's own `--selfcheck`.

11. **The assembly fan-out is checked by *differencing two real assemblies*.**
   Building the same 2-body scene with and without the joint geometries and
   differencing the two `triplet_A`/`b` dumps (via a test-local
   `GlobalLinearSystem::Impl`, §7.22 rule 12) isolates the joint's contribution
   *without* needing an oracle for the ARAP body term — a much smaller oracle than
   T6.3a's.  Measured: the extent budget `base 32 triplets / 2 blocks` vs
   `joint 176 / 11` (the three joints add `3 * 3 * 16 = 144` triplets), the 48 COO keys
   (32 diagonal + 16 distinct cross-body: the ascending and the *descending* pair share
   their `(min, max)` keys), `worst |entry| 7.276e-12`, `worst |block| 1.053e-11`,
   gradient `rel 7.431e-15` (joint-only, `|b_joint| = 9.086e+04`) and `2.182e-15`
   against the *absolute* base gradient plus the scatter.  The `zero_out_lower` of the
   two diagonal 12x12 blocks and the 16-way 3x3 unpacking are the T6.1b conventions
   (§7.24 rule 11); the new one is `upper_LR` (rule 12).

12. **`upper_LR` is load-bearing and needs a *device* select — build the scene with a
   descending pair.**  The reference writes the single off-diagonal 12x12 block at
   `(bids(L), bids(R))` where `(L, R) = (0, 1)` iff `bids(0) < bids(1)`, else
   `(1, 0)`, with the *same* rule for the block's content (`H.block(L, R)`); since the
   comparison is a device predicate it must become
   `scalar_detail::{select_scalar, select_container}` (RULE M2), not a branch.  The
   reporter interface only accepts *upper* block indices (`reporter_hessian_row`
   zeroes `body_i > body_j`), so a joint whose left body has the larger id must have
   its off-diagonal placed at `(min, max)` or its block is silently dropped — the test
   scene therefore contains one ascending and one **descending** pair, and the
   perturbation `ascending -> true` fails the COO comparison with `2.5e+04` (the whole
   block is replaced by zeros).

13. **The *energy* hook IS reachable here — through a test-built manager `Impl`.**
   Unlike the constitutions (T6.3a §7.24 rule 8: their energies view is writable only
   by T6.4's line-search reporter), a joint's `ComputeEnergyInfo` is an
   `InterAffineBodyConstitutionManager::EnergyInfo` whose constructor and the
   manager's `Impl` (with `affine_body_dynamics`, `dt_attr` and the
   `constitution_energy_offsets_counts` table) are **public**, so the test fills them
   and calls `do_compute_energy` on the real kernel: `worst rel 0.0` (assembly case)
   and `4.9e-16` (scene case) against the hand-derived energies.  The *extent* hooks
   stay unreadable (their counts are private), which is why the extent arithmetic is
   pinned through the assembled budgets instead.

14. **`make_spd`'s element Hessians are diagonal, so 8 sweeps == 20 == exact; but the
   two *offsets* in `from_handles_full` must be the assembly's, not zero.**  The
   joint kernel rebuilds the two *device* views from the handles the manager hands
   over (`gradients.indices()/values()`, `hessians.row_indices()/col_indices()/values()`)
   plus the view's *own* metadata, and passes `subvector_offset = 0` /
   `subvector_extent = total_extent()` (the doublet's segment index is the *global*
   body id, exactly as the reference's `upper_LR` assumes).  Handing the *rebased*
   view (offsets 0, the T6.1b test's idiom) is also correct — but then the kernel must
   not add the assembly's `doublet_index_offset`/`triplet_index_offset` again; the
   port keeps the assembly's offsets and writes at `2 * I` / `HalfHessianSize * I`
   *inside the slice*, which is the reference's own arithmetic.

15. **The uid-25 hazard, measured, and resolved the T6.3a way (moving the mock).**
   T6.1b's test-local `t61b::T61bInterConstitution` claimed uid 25 (the frontend's
   `AffineBodyFixedJoint`) because that was the only way to get an
   `InterAffineBody`-typed geometry into its scene then; T6.3b1 registers the *real*
   **Measured first:** with the real system registered but the
   mock still at 25, `run "[abd_linear]"` → rc **42**, `426 assertions / 2 failed`
   (`T61bInterConstitution::geo_info_count == 0u` and `geo_ids.size() == 0u` at
   `luisa_abd_linear_test.cpp:2340/2341`) and the **full suite** → rc **42**,
   `175 test cases (174 passed / 1 failed)`, `761387 assertions (2 failed)` — the
   *real* system won the group, exactly
   as `SimEngine::build_systems()`'s demangled-type-name sort predicts
   (`t61b::T61b...` < `uipc::backend::luisa::...`) plus the manager's
   "last equal-uid entry wins" map.  **Resolution:** the mock moved to uid **28**
   (`AffineBodyD6Joint`: frontend-registered, `InterAffineBody`-typed and
   grep-proven unused — the reference has no `ConstitutionUID = 28` under
   `src/backends/cuda`, and no test uses it), with a new 4-line
   `claim_mock_inter_uid(joint_geo)` that rewrites the joint geometry's
   `constitution_uid` to 28 while its `constraint_uids = {25}` stay (the *animator*
   is a different registry and `T61bInterConstraint` still claims 25 there).  The
   real joint is then *built* in that scene but owns no geometry and contributes
   zero, so the file's assembly oracle keeps its meaning.  **Cost:** zero —
   `run "[abd_linear]"` → **495 assertions in 7 test cases**, rc 0, the recorded
   T6.1b number exactly.  Reserved test uids after this batch: **8** (T6.1a's mock),
   **16** (T6.1b's `SoftTransformConstraint` stub; T6.4 owns the real one),
   **25** (the real fixed joint, and T6.1b's *constraint* stub), **26** (the real
   spherical joint), **28** (T6.1b's inter-body mock), 666/671 (the two external-force
   managers).  Note the shapes that are *not* free: 16, 18, 19, 20, 21, 23, 666, 667,
   668, 669, 670 are real uids of this milestone's remaining batches.

16. **The scene case's absolute block check must skip the same-body keys.**  The
   *absolute* `triplet_A` of a scene carries the ARAP body term (kappa = 1e8), so only
   the *cross-body* keys (16 of the 48) are an absolute statement about the joints;
   the differenced 48-key check lives in the assembly case.  The `[t61b]`/`[t63a]`
   lesson repeats: state which keys your oracle can speak about.

17. **Two Eigen traps this batch paid for (both measured).**  (a) `Eigen::Matrix`'s
   default constructor does **not** zero, so `BlockMap out; out[key] += blk;` reads an
   uninitialized 3x3 — the port's oracles must accumulate through
   `find`/`emplace` (measured: a `~1e257` entry, and an `inf` block norm that looked
   like a kernel bug for two runs).  (b) `Eigen::Transform<...,Affine>::rotation()` is
   a polar decomposition (rule 3), while `q_to_transform`/`transform_to_q` are plain
   linear-part spellings — a scene oracle must not mix the two.

18. **Regression shape of this batch, and the cold-JIT cost.**  `run "[abd_joint_a]"`
   → **223 assertions in 8 test cases**, rc 0 (per case: closed_forms 31,
   finite_difference 13, degenerate_inputs 23, assembly 16, scene 18, device_vs_host 2,
   uid_ownership 4, summary = the accounting case; the 7 scopes' 107 `T63B1_CHECK`s
   are the file's own accounting and the remaining assertions to the 223 total are the
   `REQUIRE`s and the summary's `CHECK`s).  `run ""` → **761456 assertions in 175 test cases**
   (= the T6.3a baseline 761233/167 + this file's 223/8), rc 0; `run "~[abd_joint_a]"`
   → **761233 assertions in 167 test cases** = the T6.3a baseline *exactly*, rc 0
   (re-measured four times — see the FINDINGS note in the batch report about the one
   0xC0000409 teardown flake); ctest → **1/1 passed** (`uipc.backend_luisa`, 22.61 s,
   label gpu); `python scripts/luisa_standalone.py cuda` → **69 cases, 0 checks
   failed**; `python scratch/gen_abd_cons_a.py --check --selfcheck` → rc 0.  The other
   ABD filters are unchanged: `[abd_base] 2143/7`, `[abd_linear] 495/7`,
   `[abd_dynamics] 547/7`, `[abd_constitution] 523/7`.  **The new file is picked up by
   the test target's `file(GLOB_RECURSE SOURCE CONFIGURE_DEPENDS "*.cpp" "*.h")` +
   `uipc_add_test(backend_luisa ${SOURCE})` (apps/tests/backends/luisa/CMakeLists.txt)
   and the two backend TUs by `file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp"
   "*.h" "*.inl")` + `target_sources(luisa PRIVATE ${SOURCES})`
   (src/backends/luisa/CMakeLists.txt:77)** — verified in the build log
   (`[138/142] ... luisa_abd_joint_a_test.cpp.obj`, `[35/142] ...
   affine_body_spherical_joint.cpp.obj`, `[54/142] ... affine_body_fixed_joint.cpp.obj`),
   not assumed.  **Cold JIT:** the first filtered run of the new file took **3 m 23 s**
   (the joints' energy/gradient/Hessian shaders plus the two inlined `make_spd`s),
   every later run ~5 s — the same order as T6.3a's constitutions.

19. **The `make_spd` sweep count is *unobservable* for these two joints (parent-verified
    negative result; see rule 7).**  Both joints' element Hessians are exactly
    `ddEdFr = kappa I_9` / `ddEdFt = kappa I_3` (diagonal), and `jacobi_evd` with
    `sweeps = 0` returns `values = diag(M)`, `vectors = I` (`luisa_device/math/evd.h:155-164`),
    so `make_spd(diag(kappa), 0) = diag(max(kappa, 0))` — *bitwise equal* to the 8-sweep and
    20-sweep results.  Measured: the parent perturbed `abd_fixed_joint_detail::kMakeSpdSweeps
    8 -> 0`, rebuilt, and `run "[abd_joint_a]"` stayed **223/8 green (rc 0)** — a genuinely
    value-invariant change, *not* a coverage gap: the pass/fail pins
    (`max_abs(H_neg) == 0` in *closed_forms*, `measured > 1e-4` in *degenerate_inputs (D4)*,
    `worst_spd == 0.0`) are all satisfied identically.  Consequence for T6.3b2: the
    **revolute** joint's element Hessian is *not* diagonal, so its clamp block *is*
    sweep-sensitive — T6.3b2's test must perturb that one (e.g. `kMakeSpdSweeps` or the
    pre-fan-out `make_spd` call) to pin the sweep count there.  Do not put a sweep-count
    perturbation into a *diagonal*-element-hessian batch expecting it to bite.
20. **Parent verification of this batch (what the independent pass added on top of the
    subagent's own perturbations).**  Beyond the agent's three axes (a *generated*
    exponent, the hessian's `upper_LR` predicate, the energy kernel's mass coefficient),
    the parent perturbed three further axes and reverted each byte-identically:
    (a) the **gradient doublet layout** (`write_pair_gradient`'s two payloads swapped:
    `base <- G[12:24]`, `base+1 <- G[0:12]`) → rc 42, 2 failures at test lines 1831/1846
    (the assembly's COO/gradient residual); (b) the **uid registry** (the fixed joint
    claiming `26ull` instead of `25ull`) → rc 42, 6 failures, with the diagnostic itself
    flipping to `uid25 = ANOTHER SYSTEM` (the ownership pin is therefore non-vacuous);
    (c) `kMakeSpdSweeps 8 -> 0` → **not** caught, and provably value-invariant (rule 19).
    The parent also re-ran the whole battery independently: `[abd_joint_a]` 223/8,
    `~[abd_joint_a]` 761233/167 (the T6.3a baseline exactly), the full suite 761456/175,
    `[abd_base]/[abd_linear]/[abd_dynamics]/[abd_constitution]` 2143/7 · 495/7 · 547/7 ·
    523/7, standalone 69/0, generator `--check --selfcheck` rc 0, and **ctest 1/1 passed**
    (22.60 s) — the first ctest attempt failed with LC's CUDA compiler reporting *"Failed to
    start the process: An operation on a socket could not be performed because the system
    lacked sufficient buffer space or because a queue was full"*
    (`LuisaCompute/src/backends/cuda/cuda_compiler.cpp:54`), a **transient host-resource
    error** (a parallel batch of GPU compilers had just run), not a defect: the immediate
    re-run passed.  Budget for it when several JIT-heavy runs are launched back to back.

### 7.26 Patterns established in T6.3b2 (ABD joints, batch 2: prismatic, driving, external force, limit)

Scope: the reference's `affine_body_prismatic_joint.{h,cpp}` (uid 20, with its dof reporter
and time integrator), the *driving* `affine_body_driving_prismatic_joint` (uid 21, an
`InterAffineBodyConstraint`), the external-force pair
`affine_body_prismatic_joint_external_force.{h,cpp}` (uid 667, constraint + reporter) and the
joint limit `affine_body_prismatic_joint_limit.{h,cpp}` (uid 669, a second
`InterAffineBodyConstitution` gated on `extra_constitution_uids`), plus
`joint_limit_penalty.h` and the shared arithmetic header
`affine_body_prismatic_joint_function.h`.  Two generated bodies:
`sym/affine_body_prismatic_joint.inl` (F01, J01T_G01, J01T_H01_J01, E01, dE01dF01, ddE01ddF01,
E23, dE23dQ, ddE23ddQ — 1707 statements) and `sym/affine_body_driving_prismatic_joint.inl`
(F01_q, J01T_G01, J01T_H01_J01, Distance, dxdF01, ddxdF01, E01, dE01dF01, ddE01ddF01 —
1159).  Test file `apps/tests/backends/luisa/luisa_abd_joint_b_test.cpp`: 8 cases, 402
assertions.

1. **One header/.cpp pair per uid — keep the joint's own dof reporter / time integrator in
   the joint's pair.**  The reference has one file per *uid*, so the split here is the
   reference's own (unlike T6.3b1's fixed/spherical pair): `AffineBodyPrismaticJoint` (20)
   registers the `AffineBodyPrismaticJointDofReporter` and the
   `AffineBodyPrismaticJointTimeIntegrator` in the *same* file as the constitution, the
   driving (21), the external force (667) and the limit (669) get one pair each, and every
   `REGISTER_SIM_SYSTEM` appears exactly once.  The two shared layers are the arithmetic
   header (all four systems' laws) and the branchless `joint_limit_penalty.h`.

2. **The prismatic `F01`'s *first three rows* are minus `Ft`.**  Measured, not assumed: the
   joint's `Ft` is the difference of the two bodies' attachment points, and the pair vector
   is `F01 = [-Ft ; A_i t_i ; A_j t_j]` — the `Ft` block is the *negated* one (the reference's
   `rest_c` enters the joint chain with the opposite sign to the driving chain's).  Getting it
   wrong keeps the *energy* comparison passing (E does not see `F01`'s sign) while G/H break
   at rel 1.0, i.e. exactly the §7.25 rule 9 species of transcription risk.

3. **`J01` is *constant* for the prismatic joint — and the banner has to be written from a
   measurement, not from the law's shape.**  `F01` is affine in `q`, so `dF01/dq` does not
   depend on the state; the file banner initially claimed the opposite and was corrected
   after the *analytic* J01 was checked against a finite-difference Jacobian
   (`scratch/t63b2_jac.py`: 1.9e-10 with the correct 9x24 layout, dev 1.0 for the tempting
   "body-local `12 + 3k + m`" spelling — the same trap as §7.25 rule 9, now caught before a
   line of C++ was written).  A constant Jacobian is also why the closed-forms case can pin
   `J01` at 1.3e-10 (10x the FD bound) instead of at 1e-12.

4. **The driving joint negates its *third* block — give it its own `J01q_of`.**
   `F01_q = [-Ft ; A_i T_i ; -(A_j T_j)]`, so the 9x24 map is *not* the prismatic one, and
   neither is the limit chain's (its `Distance` is a scalar of the same pair data).  Reusing
   `J01_of` for the driving or for the limit cost 0.07-1.83 relative error in the oracles —
   measured, then fixed by three explicitly named Jacobian helpers (`J01_of`, `J01q_of`,
   `Distance`'s own), each pinned against its own FD.

5. **The driving's `Distance` is the *half* of the axis difference, and its absolute value
   needs its own derivative chain.**  `Distance = a·(b - c)/2` (a = the `rest_t` axis, b/c =
   the two attachment points) and the limit consumes `x = |Distance| + offset` with
   `dx_dq` = `d|Distance|/dq`; the chain E -> dE -> d2E was re-derived by hand in the test
   and compared against the ported `compute_absolute_distance{,_derivative,_hessian}`
   (residuals 1.11e-16 / 1.11e-16 device-vs-host, 9.3e-11 in the oracle), so the *scalar*
   law, its gradient and its Hessian are all pinned independently of the generated bodies.

6. **The expression-level oracle first, and the two new non-diagonality flags it produced.**
   Both generated bodies were re-parsed and evaluated numerically
   (`scratch/t63b2_oracle.py`) against hand laws written out *there*: scalars ~2e-16, F01 /
   F01_q 3.3e-16, the fan-outs 1.4e-10-3.1e-10, `ddE01ddF01`/`ddE23ddQ` 1.3e-8-5.2e-8, the
   limit chain 9.3e-11 — and two *shape* flags worth recording before any C++ exists:
   `dpj.ddE01ddF01_nondiag 1.15`, `dpj.ddxdF01_nondiag 0.5` (the element tensors are *not*
   diagonal, which is what makes rule 10's clamp observable).

7. **`joint_limit_penalty.h` is branchless on purpose — with the branchy reference kept as
   the *test's* oracle.**  The reference's early `return`s (`if(inside) {...}`, the `w == 0`
   special case, the `dE_dx != 0` guard) have no DSL equivalent: LC's `Expr<bool>` cannot
   drive a host-and-device `if`, so the port is a `select`-only transcription (RULE M2).
   The differential proof is the *assertion* that all 11 branch combinations reproduce the
   branchy reference written out in the test: worst |dE| / |dE_dx| / |d2E_dx2| = 0.0
   (bitwise) — i.e. the ported select tree and the reference's control flow are the same
   function, and the test is the only place the *un*-ported form can be evaluated.

8. **A guard that is value-free is a guard to *drop*, and to *document*.**  The reference's
   `if(dE_dx != 0)` wrap around the limit Hessian's `dE_dx * dE_dx^T` term is value-free:
   the test evaluates `|H(0)|` for exactly the state the guard rejects and measures 0.0 (the
   term is a product of two `dE_dx`s, so the guard only skips an exact zero).  The port keeps
   the guard-free arithmetic and says so in the header — the deviation is recorded *with* the
   measurement that makes it a non-deviation.

9. **The uid table and the ownership *proof*.**  This batch claims 20 (prismatic joint), 21
   (driving constraint), 667 (external-force constraint **and** its reporter — the shared
   constraint/reporter registry, T5.4b's 671 precedent) and 669 (limit constitution).  The
   uid-ownership case does not merely query the names: it asserts, per geometry, that the
   limit-bearing geometry's base uid is 20 with `extra_constitution_uids = {669}` and that
   the geometry's constraint set is `{21, 667}` — a perturbation of the registry then flips
   the *diagnostic itself* to `uid669 = ANOTHER SYSTEM`, i.e. the pin is non-vacuous.
   Reserved before this batch: 8, 16, 25, 26, 28, 666, 671.

10. **`kMakeSpdSweeps` IS observable for these joints — this is §7.25 rule 19 flipped, and
    the sweep count is now *pinned by the test*.**  The element Hessians clamped on this
    path (the joint's 9x9 `ddE01ddF01`/`ddE23ddQ`, the driving's 9x9, the limit's `d2E_dx2
    dx_dq dx_dq^T` term) are non-diagonal and *indefinite* in 200/200 sampled states (min eig
    -5.597 for the raw 24x24, -4.082 for the element H01), so `make_spd` is a real projection:
    `||H_raw - H_shipped|| = 3.137e-1`, and the 0-vs-8-sweep distance is **3.454 / 2.765 /
    0.955** (joint / driving / limit) while 8-vs-20 is 0.0.  The closed-forms case therefore
    *pins* `sweeps == 8` (a `kMakeSpdSweeps 8 -> 0` perturbation is caught here, unlike in
    T6.3b1), and the finite-difference case asserts **both** ends: dev(H_raw, FD) = 1.8e-10
    (the transcribed arithmetic) and dev(H_shipped, FD) = 3.7e-1 (the projection) — a raw-Hessian
    check alone would silently accept a broken clamp, and a shipped-Hessian check alone a
    broken law.

11. **Isolate a *constraint*'s contribution by differencing two animator-stepped worlds.**
   A constraint's extents only exist after `InterAffineBodyAnimator::step()` has run, so
   differencing the same scene with and without the driving geometry (as §7.25 rule 11 did
   for constitutions) is the only oracle needed: the assembled joint-side budget is 224
   triplets / 14 blocks (2 joint + 12 limit) *before* the animator steps and the driving adds
   its own 6 blocks after, and the *differenced* driving contribution matches the hand-derived
   blocks at dev 9.346e-15 (48 keys), gradient rel 1.579e-15 (|b_driving| 4.502558e+04) and
   the cross-body hand-derived check at 5.093e-11.

12. **Driving a constraint's kernels *directly* needs a bare animator `Impl` — and the
   hessian *budget* is in 12x12 blocks while the view's payload is 144 doubles.**  The
   `GradientHessianInfo` the animator hands over is `subview(offset, count)` of the view,
   sliced by the constraint's own `constraint_hessian_offsets_counts` table, so the test
   builds an `InterAffineBodyAnimator::Impl` (public), fills the table with `counts()[0] = 3 *
   n_pairs` (3 *blocks* per pair: (i,i), the off-diagonal, (j,j)) and calls
   `do_compute_gradient_hessian`; the resulting triplets are compared 12x12 against
   `H.block(...)`, with the row/col indices the *raw body ids* (`bids`), not the assembled
   COO's `body * 4 + a` sub-blocks.  Measured trap: reading the payload as the assembled
   layer's 9-double 3x3 triplets (the §7.22 rule 8 unpacking) gave dev exactly 1.037 and only
   3 of 3 keys — the assembled COO and the reporter's view use *different* triplet payloads,
   and the budget table is in blocks.  After the fix: gradient 6.766e-16, Hessian 3.066e-16.

13. **`select_container(f, t, p)` means `p ? t : f` — the argument order is a silent,
   whole-block bug.**  The driving kernel's `constrained == 0` select was written as
   `select_container(zero_H, H, constrained == 0)`, which zeroes the Hessian for exactly the
   *constrained* pairs: the assembled contribution of a whole constraint became zero (the
   differenced assembled blocks dev 1.0, the differenced gradient rel 1.0 with |produced|
   0.0 against |b_driving| 4.502558e+04, the scene's cross-body worst 3.560e+04) while the
   *energy* kernel — whose select had been written the other way — kept matching at 4.0e-16,
   so the failure looked like a numerics problem in the Hessian.  The two lessons: spell the
   helper's polarity in the call site or assert it (`scratch` probe: `f=0,t=1,p=1 -> 1`), and
   keep at least one *differenced* diagnostic (rule 11) that cannot be satisfied by a
   zero contribution.

14. **The external-force pair is a constraint (uid 667) plus a reporter (uid 667) with a
   *symmetrized* axis pair.**  `symmetrized_axes` returns the two body-local axes the pair
   force is built from and `force_pair` returns `(F_i, F_j)`; the test asserts `t_i`, `t_j`,
   `F_i`, `F_j` against its own hand oracle on both the host path and the device round
   (dev <= 1e-12, measured ~1e-16), which is what makes the 667 sharing safe: the reporter
   inherits nothing the constraint's numbers do not already fix.

15. **The joint-limit's `ref_qs` needs a registered 24-vector: add `LVector24` to
   `affine_body/type_define.h`, and convert it *element-wise* in `do_init`.**  The limit's
   per-geometry state is a 24-vector (the pair dofs) and the LC buffer layer only accepts
   registered types, so `LVector24`/`Vector24`/`to_luisa`/`to_eigen`/`load_vec24` went into
   the *affine-body* type header (not the global one — §7.21's rule on scope).  Uploading it
   with `copy_host_to_device` fails to compile (C2668: a dependent unqualified call in an
   earlier template cannot see the later overload while Eigen's converting constructor makes
   all the older overloads viable) — the working spelling is an element-wise loop in
   `do_init`.

16. **Consume *every* slot of a device-vs-host bag, and assert the stride.**  The
   device-vs-host round packs each state's whole output bag into one flat buffer
   (`kRoundStride = 1862` doubles for 16 states here); the host side must consume the
   distance block's 24-entry `dx_dq` payload even when another check already covers it.  A
   missing 24-slot shifted every later field by exactly one block and the *later* deviations
   read O(1) (limit E 5.815e+03, penalty 1.1e4, force 1.5) while everything *before* the gap
   stayed at 1e-16 — the classic layout-mismatch signature, and the reason the case now ends
   with `T63B2_CHECK(c, kk == kRoundStride)`.  After the fix: 0.0 / 3.66e-16 / 4.25e-16.

17. **A device-vs-host round is per-routine and *not* bitwise.**  The generic host routines
   and the generated kernels are the same statements, but `lps::pow` lowers to `lcrt::pow`
   and `make_spd`'s sweeps reassociate, so the case asserts a per-routine bound (1e-12) and
   *records* the measurement (prismatic 4.4e-16/4.6e-16/5.4e-15, driving
   2.2e-16/4.0e-16/5.7e-15, distance 1.1e-16, limit 0.0/3.7e-16/4.3e-16, penalty 0.0,
   external force ~1e-16) — §7.17 rule 34 / §7.24 rule 15's discipline, now for a
   constraint + a constitution + an external-force pair in the same case.

18. **Regression shape of this batch, and the cold-JIT budget.**  `run "[abd_joint_b]"`:
   402 assertions / 8 cases; `run "~[abd_joint_b]"`: 761456/175 (the pre-batch baseline
   exactly); the whole suite: 761858/183; `[abd_base]`/`[abd_linear]`/`[abd_dynamics]`/
   `[abd_constitution]`/`[abd_joint_a]`: 2143/7 · 495/7 · 547/7 · 523/7 · 223/8; ctest 1/1;
   standalone 69/0; generator `--check --selfcheck` rc 0.  A *cold* `[abd_joint_b]` run costs
   ~26 min of Luisa JIT (the piecewise-constant Hessians make eight cases of fresh kernels),
   and **any** edit to a ported or generated kernel source re-pays it for the cases that use
   that kernel — so perturb with the *owning case* filter and re-verify the whole filter
   afterwards (the reverted sources hit the warm cache and run in seconds).

19. **Non-vacuity: four perturbations on four axes** (`scratch/t63b2_perturb.py`, the
   T6.3b1 driver pattern: patch -> build -> run -> revert -> sha256 -> re-verify).
   P1 the generated exponent (`sym/affine_body_prismatic_joint.inl`, the first
   `lps::pow(..., 2.0)` -> 3.0) — the *transcribed-arithmetic* axis;
   P2 the driving's `kMakeSpdSweeps` 8 -> 0 — the *clamping* axis (observable here, rule
   10);
   P3 `ascending -> always true` in `write_pair_half_hessian` — the *ordering/plumbing* axis
   (the scene deliberately holds a descending pair, §7.25 rule 12);
   P4 the driving `select_container` argument order (rule 13) — the *kernel-input* axis, i.e.
   the exact bug this batch paid for.
   Every one of them must exit the owning case with rc 42 and at least one `FAILED:` line,
   and the reverted tree must rebuild with 0 warnings and run the whole `[abd_joint_b]`
   filter green again (sha256-verified byte-identical revert).  Measured signatures (all four
   rc 42): P1 -> 3 failures, `closed (40 states)` H 6.016e-01 and `ddE01ddF01` 1.734e+00 (the
   *laws*, not the plumbing); P2 -> 5 failures, and the sweep pin itself *moves* — `0-vs-8`
   drops 2.765 (driving) to 0.000 while `8-vs-20` picks it up; P3 -> 6 failures, the
   differenced COO's `worst |entry|` jumps 5.821e-11 -> 6.775e+04 and the direct-drive
   Hessian shows a **4th** produced key (the descending pair's off-diagonal block lands on
   the wrong key); P4 -> 2 failures, the direct-drive Hessian dev is exactly 1.000 (the whole
   24x24 zeroed) while the *assembled* COO stays at 5.821e-11 — the driving's contribution in
   the pre-step world is invisible to the assembly case, which is why the direct-drive harness
   (rule 12) and the scene's post-`advance()` differencing (rule 11) are both needed.

20. **The tooling incident worth a rule: a heredoc can rewrite an entire file with its
   newlines mangled.**  A `python - <<'PY'` block whose source contains the C++ newline
   escape (backslash + `n`) had every such escape — *and* every real newline of the file it
   rewrote — turned into the two characters `/n`, leaving a 188 KB single-line file, while
   other backslash escapes survived.  Recovery recipe, in order: (a) keep a `.bak` copy the
   moment a file-rewriting probe is about to run; (b) restore with a *state scanner*
   (code / string / char / line-comment / block-comment) that maps `/n` back to a newline,
   with exactly two exceptions — a separator seen *inside* a string or char literal must have
   been the escape (a real newline cannot occur inside a literal), and a separator whose
   neighbours are identifier characters (or `ident` + `/ident`) is genuine prose (measured:
   `src/backends/none`, `zero/negative`, `rest_t/n/b` inside `///` comment lines);
   (c) re-run the *whole* battery — the byte-identical green result (402/8 here) is the proof
   that the repair did not silently change a single statement.  Prevention: never write a
   backslash-`n` inside a heredoc (use `chr(10)`), prefer the `edit`/`write` tools for file
   surgery, and prefer line-index or whitespace-tolerant (`\s+`/`re.S`) patches over
   multi-line literals copied from a display (the display strips the common indentation, so
   an "exact" anchor built by eye will not match).

21. **Parent verification of this batch — two further perturbation axes, and a harness fact.**
    Beyond the agent's four axes (a generated exponent; the driving `kMakeSpdSweeps 8 -> 0`; the
    hessian's `upper_LR`; the driving `select_container` order) the parent perturbed, on
    *different* axes, and reverted each byte-identically:
    (a) **the external-force sign axis** — `symmetrized_axes`' `t_i = -t_j` -> `t_i = t_j`:
    rc 42 with 7 failures spanning the FD (lines 2252/2255/2260), the scene (3365-3367) and the
    device-vs-host (3915) cases; (b) **the joint-limit's cross-constitution lookup** — `JointUID`
    20 -> a literal `26`: the reference-faithful host assert *"AffineBodyPrismaticJointLimit must
    be attached on base prismatic joint"* fires, and then the harness **aborts the process**
    (rc 3221226505 = 0xC0000409) rather than reporting a Catch2 failure. That abort is a
    pre-existing property of the test harness (the same code and shape as T6.3b1's once-seen
    teardown flake), so a broken cross-lookup cannot silently pass — but it also means a
    *deliberate* failure of a `UIPC_ASSERT` in `do_build`/`do_init` shows up as "the run is not
    green", not as a named Catch2 assertion.  Note the cost asymmetry: a perturbation that changes
    a *kernel* source (e.g. a function header included by the joint's TU) invalidates the PTX
    cache and re-JITs for ~25 min here, while a host-only constant (the limit's `JointUID`) is
    warm — budget accordingly when choosing a perturbation axis.
22. **Anchor-matching lesson (reinforces rule 20's whitespace warning).**  The reference and the
    port both *column-align* their `static constexpr U64` blocks, so a declaration reads
    `static constexpr U64   JointUID     = ...` with run-length padding.  A patch anchored on
    `"U64 JointUID = ..."` silently finds **zero** matches and — if the assert is skipped — a
    follow-up run can measure the *stale* binary instead of the perturbation (that happened here:
    one attempted axis reported "caught" from an unrelated earlier build).  Always match declarations
    with a whitespace-tolerant pattern (`\s+`), and treat "build rc != 0" as *inconclusive*, never
    as evidence.
23. **The parent's independent re-run (frozen tree, after every perturbation was reverted).**
    `[abd_joint_b]` 402/8 rc 0; `~[abd_joint_b]` 761456/175 rc 0 (the pre-batch baseline exactly);
    the full suite 761858/183 rc 0; `[abd_base]`/`[abd_linear]`/`[abd_dynamics]`/
    `[abd_constitution]`/`[abd_joint_a]` 2143/7 · 495/7 · 547/7 · 523/7 · 223/8; ctest 1/1 passed;
    standalone 69 cases / 0 checks failed; the generator `--check --selfcheck` rc 0; and the stray
    `_hf_probe.obj` a device-vs-host probe left in the repository root was removed before the
    commit (check `git status` for stray artefacts — the build dir is ignored, the root is not).

### 7.27 Patterns established in T6.3b3 (ABD joints, batch 3: revolute, driving, external force, limit)

   The last joint batch ports the **revolute** family: `AffineBodyRevoluteJoint` (uid 18) with its
   dof reporter and time integrator, `AffineBodyDrivingRevoluteJoint` (uid 19, the animator-driven
   `atan2` constraint), the external-force pair (`AffineBodyRevoluteJointExternalForceConstraint` /
   `AffineBodyRevoluteJointExternalForce`, both uid 668 — the shared constraint/reporter registry
   again) and `AffineBodyRevoluteJointLimit` (uid 670, whose `JointUID` cross-references
   `AffineBodyRevoluteJoint::ConstitutionUID`).  Seven classes as four `.h`/`.cpp` pairs plus the
   shared `affine_body_revolute_joint_function.h` and three generated `sym/*.inl`
   (`affine_body_revolute_joint.inl` 927 statements, `affine_body_driving_revolute_joint.inl`
   1192 — the project's largest body, 1.1 MB of SymEigen source — and
   `affine_body_revolute_joint_limit.inl` 1344), and `apps/tests/backends/luisa/
   luisa_abd_joint_c_test.cpp` (tag `[abd_joint_c]`, 8 cases / 383 assertions).  It reuses
   `joint_limit_penalty.h` unchanged, which is the b2 deliverable paying off.

1. **The *revolute* joint's element Hessian is DIAGONAL, so §7.26 rule 10 does not carry over to
   it — but it does carry over to its two companions.**  The joint's `Faxis` is a 6-vector of two
   coincident *point* differences, so `ddEaxisddFaxis = κ I₆` exactly (`|H| = κ·5` for
   `κ = 1`, the 6x6's only non-zeros are the six κ's) and `make_spd` is a pure no-op:
   0-vs-8 sweeps differ by **0.000e+00**, `‖H_raw − H_shipped‖ = 0.0`, a *negative* κ is clamped to
   exactly zero, and the FD case asserts `dev(H_raw, FD) == dev(H_shipped, FD) == 8.8e-11`
   (the transcribed arithmetic).  The **driving** (uid 19) and the **limit** (uid 670), however,
   are 12x12 and *indefinite* (the non-diagonal magnitudes are 1.589 / 0.690), so the sweep count
   is a real projection there: **0-vs-8 = 7.270e-01 (driving) / 7.237e-01 (limit)**, 8-vs-20 =
   0.000e+00, and the test *pins* `kMakeSpdSweeps == 8` for both while the FD case asserts both
   ends of the driving's Hessian (raw chain vs FD 1.1e-10, shipped vs FD 3.0e-01).  Read rule 10
   of §7.26 as a statement about *those* two, not about the joint.

2. **The revolute element Hessians are 12x12 where the prismatic ones were 9x9, and that size is
   what deadlocks LC's `Device::compile` (this batch's headline finding).**  A *test-local* kernel
   containing either the driving's `body_hessian` or the limit's `hessian` (both a 12x12
   `make_spd` chain) parks forever inside `luisa::compute::Device::compile`: process CPU time and
   GPU utilisation freeze at 0, memory stops growing, only `taskkill` ends it.  Measured by
   `#if 0` bisection plus per-constructor `stderr` markers, all reproducible in isolation:

   - one kernel with joint + driving + limit + external force -> **deadlock**;
   - disabling any *one* of {driving block, limit Hessian, external force} -> **still deadlock**;
   - disabling *two* of them -> compiles and runs (rc 42 against the now-wrong layout);
   - a five-kernel split (joint / driving E+G / limit chain / limit E+G / external force) ->
     the four "small" kernels compile, and the two `make_spd` kernels deadlock **on their own**.

   The *same* templates compile and run fine inside the shipped subsystem kernels — the assembly
   case drives the driving's real k1/k2 shaders directly and measures Hessian dev 1.709e-15 — so
   this is an LC compile-path limitation triggered by these particular test kernels, not a defect
   in the transcription.  Consequences the next batch must live with: (a) keep test-local kernels
   to *one* big generated chain each; (b) the device-vs-host round can cover every chain except
   those two Hessians, and their device instantiation must be evidenced through the *subsystem*
   path (the k1/k2 direct drive).  The `[abd_joint_c]` round says so in a comment at the point of
   exclusion rather than silently skipping a comparison.

3. **`unwrap_angle(wrapped, ref) = ref + remainder(wrapped − ref, 2π)`, and `lps` has no
   `remainder`.**  The reference's helper (in `affine_body_revolute_joint_function.h`, *before* the
   three `.inl` includes) is `::remainder(wrapped_angle − ref, 2*pi) + ref`; the port spells the
   IEEE identity `remainder(x, y) == x − y*round(x/y)` with `lps::round`.  The two agree everywhere
   except an exact half-quotient tie (where `std::remainder` rounds half-to-even and `lps::round`
   half-away), and the test measures that difference on every sampled state: **0.000e+00**.

4. **An oracle that spells that law with the operands swapped is off by exactly
   `(ref − wrapped)`, and it fails "plausibly".**  Two transcriptions of the same sentence are
   both natural and only one is the reference:
   `ref + remainder(wrapped − ref, 2π)` (correct) versus `wrapped + remainder(ref − wrapped, 2π)`
   (which *returns `ref`* whenever the two are within π — the most common case).  The
   `[abd_joint_c]` device-vs-host round had exactly this for one field and reported
   `unwrap 6.388e-01` — a deviation of the same order as the data, i.e. indistinguishable from a
   real numerics bug.  When an oracle and a device disagree by "a plausible amount" on a
   *wrapping* helper, re-read the reference's own argument order before touching the port.
   (The closed-forms case pins the identity in both spellings, which is what caught it.)

5. **The revolute function header includes all three `.inl` — the limit body included *there*, not
   in the limit `.cpp`.**  The reference's `affine_body_revolute_joint_function.h` pulls in
   `sym/affine_body_revolute_joint.inl`, `sym/affine_body_driving_revolute_joint.inl` and
   `<affine_body/constitutions/sym/affine_body_revolute_joint_limit.inl>`, and the port keeps that
   shape (quoted relative includes for all three).  Consequence for the reader: the joint's own
   `.cpp` parses the 12x12 chains of all three bodies even though it only *calls* the joint's, and
   the 1.1 MB driving source shrinks to ~120 KB once the SymEigen comments are dropped.  Compile
   time for that TU is dominated by the driving body — budget for it.

6. **The first *overloaded* generated function in the project, and the transcriber's occurrence
   suffix.**  `sym/affine_body_revolute_joint_limit.inl` defines `DeltaTheta` **twice** — a
   6-vector `basis_k/qk/qk_t` frame form and a 12-vector `F/F_t` form.  The region contract is
   "one sentinel pair == every generated function between them", so `rewrite()` needed no change at
   all; only the *verification* names needed to be addressable, so `scratch/gen_fem_cons_a.py`
   gained `split_name_spec("DeltaTheta#2")` + `find_function(text, name, occurrence)` and the
   region list names the second overload `DeltaTheta#2`.  Every other region in the project is
   occurrence 1, so the extension is backward compatible (verified: `--check --selfcheck` rc 0 with
   unchanged output for all pre-existing regions).

7. **`Vector12`/`Vector24`/`Vector6` staging stays element-wise (the §7.26 rule-15 trap), and the
   revolute joint hits it in `do_init`.**  Uploading `rest_positions` with `copy_host_to_device`
   is a C2668 (a dependent unqualified call in an earlier template cannot see the later overload
   while Eigen's converting constructor makes every older overload viable); the working spelling is
   an explicit `to_luisa` staging loop, which is what all four revolute classes do.

8. **The external-force pair (uid 668) uses the *edge's own* rest endpoints and `torque_to_F`
   contributes no translational dof.**  The reporter takes the axle from `l1 − l0` / `r1 − r0`
   (the joint's own two rest points, not the `A·l0 + t` world points), normalises per body and
   *symmetrises* (`e_i = −e_j`); `torque_to_F` then returns a 12-vector whose translational 3 dofs
   are exactly zero and whose rotational 3x3 is the torque's cross-product operator.  The
   uid-ownership case proves the sharing positively (asserting the geometry's constraint set is
   `{19, 668}` and that a registry perturbation flips the diagnostic to `uid668 = ANOTHER
   SYSTEM`), which is what makes 668-as-two-systems safe.

9. **The round's *cursor* is as load-bearing as its stride.**  §7.26 rule 16's "consume every slot"
   has a twin: when a field is deliberately reserved (see rule 2) the writer must *jump* the cursor
   over the reserved block instead of writing the next field contiguously.  The first split of
   `[abd_joint_c]`'s round kernel wrote the penalty scalars straight after the limit chain because
   the loop that used to sit between them had moved to another kernel — the layout then read
   `limit E 0.0 / G 1.0, penalty 3.2e+02`, exactly the signature of a shifted field.  Keep the
   reader's `kk == kRoundStride` assertion (it stays valid) *and* keep every writer's `k =` jump
   next to the constant it jumps to.

10. **Regression shape of this batch, the cold-JIT budget and the harness limits.**  `[abd_joint_c]`
    383 assertions / 8 cases rc 0; `~[abd_joint_c]` 761858/183 rc 0 — bit-identical to the T6.3b2
    full-suite count, i.e. the batch is purely additive; the whole suite 762241/191; `[abd_base]`/
    `[abd_linear]`/`[abd_dynamics]`/`[abd_constitution]`/`[abd_joint_a]`/`[abd_joint_b]` =
    2143/7 · 495/7 · 547/7 · 523/7 · 223/8 · 402/8; ctest 1/1; standalone 69 cases / 0 checks
    failed; `scratch/gen_abd_cons_a.py --check --selfcheck` rc 0; `scratch/t63b3_oracle.py` rc 0.
    A *cold* `[abd_joint_c]` run re-JITs the 12x12-shape kernels; once warm the whole 8-case filter
    runs in ~3 s, and `~[abd_joint_c]` in ~47 s.

11. **Non-vacuity: two perturbations on two axes** (`scratch/t63b3_perturb.py`, the T6.3b1/b2
    driver pattern: patch -> build -> run -> revert -> sha256 -> re-verify).
    **P1** the *host staging* axis (`affine_body_revolute_joint.cpp`,
    `init_angles_list.push_back(init_angle_view[i])` -> `+ 0.25`) — a pure host-side data-staging
    change, so the PTX cache stays valid and the owning case (`abd_joint_c_scene`) re-runs in
    seconds; **P2** the *transcribed-arithmetic* axis
    (`sym/affine_body_driving_revolute_joint.inl`, the first `lps::pow(x3, 2.0)` -> `3.0`) — the
    1.1 MB driving body, chosen to be observable through `abd_joint_c_closed_forms`, which drives
    only *host* (`S = double`) instantiations of the generated bodies and therefore re-JITs
    nothing.  Both exit with rc 42 and at least one `FAILED:` line, and the reverted tree rebuilds with
    0 warnings and runs the whole `[abd_joint_c]` filter green again with a byte-identical
    (`sha256`) revert.  Measured: **P1 -> 5 failures** in `abd_joint_c_scene` — the two staging
    pins (`h_init_angles[0/1] == 0.0` read `0.25 == 0.0`) plus the three angle-dependent checks
    they feed (`worst <= 1e-12` -> 0.25, `worst_adopt <= 1e-12` -> 0.25, `worst <= 1e-9` ->
    0.25000000000027711); **P2 -> 1 failure** in `abd_joint_c_closed_forms`, the driving's
    `dEdF01` oracle (`worst_dE <= 1e-12` -> 1.2480488231944847e-01), i.e. the transcribed
    arithmetic itself, not the plumbing.  Both reverts are byte-identical and both reverted
    `[abd_joint_c]` runs are 383/8 green.

12. **Disaster recovery: the agent session's `wire.jsonl` is a complete, replayable edit log.**
    A heredoc `Path.write_text(..., newline='\n')` truncated the brand-new 190 KB test file to zero
    bytes mid-batch (the heredoc turns the escape into a real newline, so the `newline` argument is
    left unterminated and the write happens on a partially-parsed payload).  Everything was
    recovered, byte-for-byte, from
    `.kimix_cache/<session-uuid>/wire.jsonl`, which records **every** tool call with its full
    arguments — `write` (with `mode`), `edit` (old/new strings) **and** `bash`, including inline
    `python -c "…"` edits.  The working recipe is a replay driver that walks the calls in order,
    applies `write`/`edit` verbatim, extracts each `python` heredoc / `-c` body out of the `bash`
    command and runs it in a *sandbox* directory holding only the file under repair.  Two lessons:
    replay `bash` too (the "missing" edits were all `python -c` blocks, not tool `edit` calls), and
    treat "the file compiles" as the recovery's *entry* criterion, never its exit criterion —
    only a green run pins it.


13. **The module export ceiling (LNK1189) and the explicit `luisa_module.def` that replaces
    `WINDOWS_EXPORT_ALL_SYMBOLS` — a latent limit this batch tripped.**  On Windows the `luisa`
    module used to set `WINDOWS_EXPORT_ALL_SYMBOLS ON`, which exports *every* symbol its objects
    define.  Each kernel signature instantiates ~345 symbols of LuisaCompute DSL machinery, so at
    T6.3b2 the auto-generated list was already **65482** entries — **53 below** MSVC's 65535-export
    ceiling — and T6.3b3's eleven new kernels pushed it to **69295**, where the link dies with
    `LINK : fatal error LNK1189: cannot have more than 64K exports` (measured: the generated
    `build/src/backends/luisa/CMakeFiles/luisa.dir/exports.def` has 69295 lines; the tree at
    T6.3b2 links only by luck).  **Fix in the tree:** `WINDOWS_EXPORT_ALL_SYMBOLS OFF` + an
    explicit, *checked-in* export list `src/backends/luisa/luisa_module.def` (passed as
    `/DEF:<abs path>`), generated and verified by `scripts/luisa_module_def.py`
    (`--check` → rc 1 when stale).  Contents = the module's four C entry points
    (`uipc_create_engine`, `uipc_destroy_engine`, `uipc_init_module`, `uipc_query_module` — resolved
    with `GetProcAddress`, so *nothing* imports them) **plus every symbol the in-tree consumers
    import** (mechanically: `dumpbin /imports` over the build's PE files, keeping only the sections
    whose DLL header is `uipc_backend_luisa.dll`; a naive whole-dump scan mis-attributes other
    DLLs' imports).  Measured: 361 consumer symbols + 4 C API = the `.def`; the rebuilt DLL
    exports **502** symbols (the extra ones are the foundation layer's own
    `UIPC_LUISA_DEVICE_EXPORT` API).  **Maintenance protocol:** a test that uses a *new* backend
    symbol now fails the **link** with an unresolved external — the intended loud signal; re-run
    `python scripts/luisa_module_def.py` and commit the refreshed file.  Both name-loading paths
    still pass on the explicit list (`ctest -R backend_luisa` 1/1 and
    `python scripts/luisa_standalone.py cuda` 69/0), which is the evidence that the list is
    complete for the module ABI.  **Rejected alternatives:** (a) moving the kernel TUs into a
    static library linked into the module — the in-tree tests would then link a *second* copy of
    every backend class, i.e. two sets of the `REGISTER_SIM_SYSTEM` registries and two type
    identities (RTTI/`dynamic_cast` across the boundary would break); (b) post-filtering CMake's
    auto-generated `.def` — CMake passes it to the linker itself, so a filter cannot be inserted
    between its generation and the link without a second, conflicting `/DEF`; (c) keeping the
    auto-export and pruning our own symbols — the noise is *LC's* template machinery, not ours.
14. **Parent verification of T6.3b3 (independent pass) — and one perturbation that turned out to be
    value-invariant, plus the coverage it exposed.**  The parent re-ran the whole battery on the
    frozen tree *after* re-establishing the build (the export policy of rule 13): build rc 0 with
    0 warnings; `[abd_joint_c]` **383/8** rc 0; `~[abd_joint_c]` **761858/183** rc 0 == the T6.3b2
    full-suite baseline exactly (so the batch is purely additive); the full suite **762241/191**
    rc 0; `[abd_base]`/`[abd_linear]`/`[abd_dynamics]`/`[abd_constitution]`/`[abd_joint_a]`/
    `[abd_joint_b]` = 2143/7 · 495/7 · 547/7 · 523/7 · 223/8 · 402/8 (all unchanged); ctest 1/1;
    standalone 69/0; generator `--check --selfcheck` rc 0.  Two further axes (not the agent's):
    **(A) the export list** — deleting the C API entries (and later the *loader's* two) from
    `luisa_module.def` changed **nothing** (build ok, ctest 1/1, standalone 69/0): the four C entry
    points carry `UIPC_BACKEND_API` (`__declspec(dllexport)`, `src/backends/common/module.h`), so
    the `.def` never controlled them — a genuine *value-invariant* axis, not a coverage gap.  That
    experiment did expose two real defects, both now fixed: (i) a `.def` is **not** a build
    dependency of the link step, so a refreshed list silently had no effect until
    `LINK_DEPENDS` was set (verified by watching the module's link step); (ii) **nothing in the
    battery exercised the module's dynamic-loading ABI** — the in-tree tests *link* the module and
    the standalone suite builds its own copy, so `uipc::core`'s `LoadLibrary` +
    `get_function<...>("uipc_query_module")` path (`src/core/core/internal/engine.cpp`) was never
    run.  `scripts/luisa_check_module_abi.py` now closes that gap (LoadLibrary + `GetProcAddress`
    on all four entry points, rc 0) and belongs in the battery after any export-policy change.
    **(B) the uid registry** — the revolute joint claiming `19` instead of `18`: **caught**, via the
    reference's own host assert ("Assertion uid && uid->view()[0] == JointUID failed.
    AffineBodyRevoluteJointLimit must be attached on base revolute joint"), after which the harness
    aborts with rc 3221226505 (0xC0000409) rather than reporting a Catch2 failure — the same
    abort-shaped failure mode §7.26 rule 21 records.
15. **Coverage gaps this batch records (do not read them as "verified").**  (i) Two 12x12
    `make_spd` kernel chains are **excluded** from the test-local device-vs-host round because
    holding them in one test kernel *deadlocks* `luisa::compute::Device::compile` (bisected: any one
    of the four chains alone compiles; two of them together deadlock; the two excluded ones deadlock
    even alone) — their device instantiation is evidenced only through the *subsystem* shaders (the
    assembly case drives the real reporters).  (ii) The scene case's second `Engine` is gone: after a
    second engine is initialised, the next `frame_stream().synchronize()` from the test reproducibly
    SIGSEGVs or hangs (`LStream::synchronize`), a **pre-existing multi-engine hazard** that T6.4's
    multi-engine tests will hit; the driving's world-level differencing is therefore not exercised
    (its kernels are pinned by the direct k1/k2 drive + the assembly differencing instead).  (iii)
    Every number above was measured on a relinked module exporting **502** symbols rather than 69k —
    same code, different export table.  (iv) The regression suite is **order-fragile**: `--order
    rand` fails one or two assertions at *every* seed on the T6.3b2 baseline too (default
    declaration order is green), so the battery's green runs are only meaningful in the default
    order — worth a look if a future batch ever sees a spurious single-assertion failure.

### 7.28 Patterns established in T6.4a (the Newton tolerance subsystem, the ABD diag preconditioner, the ABD tolerance checker, the completed ABD line-search reporter)

**T6.4 was split into three batches and T6.4a is the first**: (a) *this* batch — the solve
plumbing: `newton_tolerance/**` (not ported before, and a prerequisite: the ABD tolerance
checker derives from `NewtonToleranceChecker`), `affine_body/abd_diag_preconditioner.*`,
`affine_body/abd_tolerance_checker.*` and the *deferred bodies* of
`affine_body/abd_line_search_reporter.cpp`; (b) `affine_body/constraints/**`
(external-articulation uid 23, soft-transform uid 16, the ABD-external-body-force pair uid
666); (c) the deferred MAS preconditioner (`mas_preconditioner_engine.{h,cu}` +
`fem_mas_preconditioner.cu`, ~2400 lines of warp/shared-memory device code).  The split is
recorded in `libuipc_lc_plan.md`'s M6 paragraph.  T6.4a delivers 12 files (6 new headers/sources
in `newton_tolerance/`, 3 new `affine_body/` pairs + their two `*_function.h`, the completed
reporter pair) plus one new generic routine in the numerics layer, the engine wiring, the
refreshed module export list and `apps/tests/backends/luisa/luisa_abd_solve_test.cpp`
(tag `[abd_solve]`, 8 cases / 245 assertions).

1. **The tolerance subsystem is the Newton loop's convergence contract, and it is *inert*
   until Phase 8.**  `NewtonToleranceManager` (`require<>`d by the engine like the
   `LineSearcher`, and `init()`ed last in `init_scene`'s step-1 order, after the
   time-integrator manager) holds the checkers; `pre_newton(frame)` fans out to every
   checker and `check(ResultInfo&)` fans out **until the first non-converged checker**
   (the `break` is part of the contract) and logs the reference's report block.  Neither has
   an engine call site yet — `advance()` still runs only the prologue/epilogue — so the
   subsystem is inert in every existing scene (that is why adding it moved **no** pre-existing
   expectation: measured `~[abd_solve]` = the T6.3b3 baseline 762241/191 *exactly*).

2. **Two pinned reference quirks in the checker base, both transcribed verbatim.**
   (a) `NewtonToleranceChecker::pre_newton(PreNewtonInfo& info)` **ignores its argument** and
   constructs a *fresh default* `PreNewtonInfo` for its subclass, so every checker sees
   `frame() == 0` — the manager's `pre_newton(7)` reaches only the log line.  The port keeps
   that (`(void)info; PreNewtonInfo pre_info;`) and the test pins `MaxTranslationChecker::frame
   == 0` after `pre_newton(7)`.
   (b) `ABDToleranceChecker::do_check` declares a **local** `bool h_success`, which shadows the
   member `IndexT h_success = 1`: the local feeds `info.converged(...)`, the member is never
   written, so `do_report()` — which reads the *member* — always prints `Tol: < abs_tol`, even
   after a *failed* check.  The port keeps both names and the shadowing and pins the report
   string after a measured failure ("`Tol: < 0.0001`"; `h_success == 1`; the flag itself is 0).

3. **The `lps::inverse` gap for `N > 4`, and what the reference does with a singular input.**
   The ABD preconditioner inverts a 12x12 block per body, and the reference's
   `cuda_tool::eigen::inverse` dispatches on the size: the closed-form adjugate forms exist for
   2/3/4 only and everything wider goes through `GaussEliminationInverse`
   (`cuda_tool/eigen/gauss_elimination.h`).  Until T6.4a `lps::inverse` *static_asserted*
   `N == 2 || 3 || 4` (cookbook §7.21 rule 11), so this batch added the generic branch to the
   numerics layer (`luisa_device/math/matrix_ops.h`, `detail::gauss_elimination_inverse`) — the
   first *additive* change to that layer since T2.2.  Its two device-control-flow statements
   become `select`s (RULE M2): the pivot *search* becomes a running scan whose outcome is folded
   into each candidate row's weight (the reference **adds** the first row below whose |entry|
   exceeds `eps`, it does not swap), and the two `return result;` statements — which return an
   **uninitialized** `Eigen::Matrix` for a singular input — become a `missing` flag that masks
   the whole result to **zero** and protects the divisions.  The port's singular answer is
   therefore the deterministic zero matrix instead of UB; that is *pinned* four ways in the test
   (a duplicate-row block, a diagonal entry below `epsilon<double>()`, one ulp above it, and a
   singular block through the factored preconditioner row).  Every loop index is a host constant
   (LC's DSL has no dynamically-indexed local array), which is why the routine is
   select-and-unrolled; the cost is a ~9k-statement shader, and the *measured* cold-JIT for the
   whole new filter is seconds, not minutes (rule 14).

4. **The ABD diagonal preconditioner is the FEM sibling's shape with a different *input*.**
   `ABDDiagPreconditioner` is a `LocalPreconditioner` bound through `BuildInfo::connect` to
   `ABDLinearSubsystem`; `do_assemble` does **not** scan the assembled BCOO (`info.A()`, the FEM
   idiom) but inverts `ABDLinearSubsystem::diag_hessian()`, the subsystem's *dense per-body*
   12x12 buffer (one element per body: no `i == j` test, no dof-window arithmetic), and
   `do_apply` computes `z.segment<12>(i * 12) = diag_inv(i) * r.segment<12>(i * 12)` over the
   subsystem's *window*, with the reference's `if(*converged != 0) return;` early-out.  Both
   bodies live in `abd_diag_preconditioner_function.h` so the shipped shader and the unit test
   run the same code (the §7.20 rule 1 recipe), and the class keeps the reference's public
   surface (`abd_linear_subsystem`, `diag_inv`) in a header, exactly like T5.4b's FEM pair — the
   test registers the *real* preconditioner into a standalone `GlobalLinearSystem::Impl`.

5. **The tolerance checker's factored row, and the DSL's missing `atomic_exch`.**  The
   reference's `abd_tolerance_checker_do_check_kernel` is three statements: read the one-element
   `success` flag, `return` if it is 0 ("already failed stays failed"), then `for(c = 3; c < 12;
   ++c) if(abs(dq[c]) > abs_tol) { atomic_exch(success, 0); break; }` — the first three dofs are
   the translation and are **ignored on purpose**, and the comparison is **strict** (so
   `|dq_i| == abs_tol` converges).  `abd_tolerance_checker_function.h` factors the decision
   (`exceeded_row` = the nine-component OR, `failed_row` = that OR folded with
   `success_value != 0`), leaving only the `$if` guards and the store in the kernel — which is
   what makes the host instantiation (a `storage::memory` view) the test's oracle.  The
   atomic's DSL gap: LC has no `atomicExch` *function*, only the `AtomicOp` behind
   `AtomicRef<T>::exchange`, so `luisa_device/atomic.h` (the file that exists for exactly these
   gaps, §3.3) gained `atomic_exch(ref, v)`; the **flag read-back** is spelled
   `success.copy_to(&frame_stream()) != 0` rather than the reference's implicit
   `bool h_success = success;`, because a `LDeviceVar`'s implicit conversion reads on the *main*
   stream and can see a stale flag (§7.22 rule 7).

6. **The line-search reporter's bodies are the FEM sibling's, plus a shape-energy sum.**  T6.1a
   had declared the class and T6.1b had pulled `add_reporter` (and the subreporter's four
   forwarding hooks) forward for their *vtables*; T6.4a adds
   `Impl::{init, record_start_point, step_forward, compute_energy}` and the five `do_*` hooks +
   `REGISTER_SIM_SYSTEM`, and deletes the SCOPE NOTEs (in the reporter, the subreporter, the
   animator and the inter-body manager).  Three kernels: `step_forward` writes
   `q = q_temp + alpha * dq` for non-fixed bodies (`$if`-guarded store), the energy fan-out
   zeroes fixed/external-kinetic bodies' kinetic energy and combines *three* sums — kinetic
   (`abd().kinetic->compute_energy`, special-cased), shape (each constitution's slice) and the
   subreporters' (each `report_extent`/`compute_energy` slice, `scan`-ed into the offsets table)
   — and a one-thread combine kernel writes `info.energy()`.  `resize_discard` maps onto
   `LDeviceBuffer::resize` (the port *is* cuda_tool's shape: amortized growth + a
   value-initialised tail) and `DeviceReduce().Sum` onto `lla::device_reduce_sum` with the FEM's
   zero-length guard.  **Registration consequence:** the ABD animator's and the inter-body
   manager's line-search subreporters stop being invalidated and become live — the explicit
   not-implemented signal T6.1b left behind is gone.

7. **THE EXPORT LIST IS NOW PART OF EVERY BATCH THAT ADDS A TEST SYMBOL — AND THE GENERATOR HAS
   A BOOTSTRAP TRAP.**  `src/backends/luisa/luisa_module.def` is the module's explicit export
   list (T6.3b3's LNK1189 fix) and `scripts/luisa_module_def.py` derives it from **the symbols the
   in-tree consumers *import*** (a `dumpbin /imports` scan of the built PE files).  That is
   self-referential, and this batch hit the trap head-on: the first build after writing the test
   failed with **13 unresolved externals** (the intended loud signal), the symbols were merged
   into the `.def` by `scratch/t64a_def_bootstrap.py` (it parses the decorated names out of the
   `LNK2019`/`LNK2001` lines), the module and the test exe relinked — and when the *generator*
   was re-run it **dropped one of them again**: a `LNK2001` for
   `?do_report@NewtonToleranceChecker@...` whose only reference is a **vtable slot** (the test's
   derived mock checkers) is resolved from the module's import library *without* creating an
   import-table entry, so the consumer-import scan cannot see it and the next link fails again.
   The fix (in the *test*): make the reference a **used** import — the mocks' `do_report()`
   calls the base's own default body
   (`std::string{"mockA|"} + NewtonToleranceChecker::do_report()`) — after which the exe imports
   it, the generator keeps it, and the loop is idempotent.  **Recipe for the next batch:**
   build → read the `LNK2019`/`LNK2001` names → `python scratch/t64a_def_bootstrap.py <build
   log>` (or add the symbol as a *used* reference) → build → `python
   scripts/luisa_module_def.py` (**rewrite**, so the committed file is generator-produced) →
   `--check` must be rc 0 → and the two name-loading paths (`ctest -R backend_luisa`,
   `python scripts/luisa_standalone.py cuda`) plus `python scripts/luisa_check_module_abi.py`
   must stay green.  Measured: the list went 362 → **374** entries (370 consumer symbols + the 4
   C API entry points), and the module relinks on every `.def` change (it is a `LINK_DEPENDS`).

8. **A shared-binary hazard that *changes the assembly's numbers*: a sibling test file's
   slot-owning, marker-gated kinetic.**  `AffineBodyDynamics` has exactly **one** kinetic slot,
   and "the last system built wins" (§7.23 rule 7's demangled-name sort); T6.3b1's file
   registered `zt61b::T61bKineticOwner` *precisely* to sort after every `uipc::backend::luisa::`
   class and keep the slot.  Consequence, measured *here*: in a scene that does **not** carry
   that file's marker, `abd().kinetic` is that mock, its `do_compute_gradient_hessian` writes
   nothing, and the assembled ABD `diag_hessian` is the **shape Hessian alone** (measured:
   `body_id_to_kinetic_hessian` identically zero, the dyadic mass 125, the (t,t) 3x3 sub-block
   exactly 0, `|diag_hessian|max = κ · 1e-4`, `cond ≈ 1e296`).  This is *not* a port defect (the
   port's kernels and the reference's)
   `diag_hessian(I) = shape + (!external_kinetic ? kinetic : 0)` agree; it is the shared test
   binary's slot ownership.  **Lessons for test design:** (i) a scene that wants the *real*
   kinetic must own the slot itself (the `zt64a::…` name trick) or carry the sibling's marker;
   (ii) a *structural* zero mode follows: the shape-only ortho-potential Hessian has neither
   translation nor rotation curvature, so `D` is singular whenever the kinetic term is missing —
   so the preconditioner's **apply oracle must be built from a hand-written `diag_inv`**
   (`ABDDiagPreconditioner::diag_inv` is public, so the test writes its own rank-1 SPD inverses
   and compares the shipped shader's `z` with the closed form), not from the assembled block;
   (iii) the *device-vs-host* round stays valid anyway and is the strongest statement available:
   the shipped shader's `diag_inv` equals the host `lps::inverse(diag_hessian)` to **0.000e+00**
   and Eigen's `inverse()` to **0.000e+00** on the same input.

9. **Perturbation *pins* must change a value, not just a call count.**  The manager's
   short-circuit was pinned by asserting that the checkers after the failing mock keep their
   previous `abs_tol`; removing the `break` (a pure host change) was **not caught** the first
   time, because the two real checkers recompute the *same* tolerance from unchanged inputs.
   The pin became non-vacuous only after the test *changed the checkers' inputs* before the
   failing check (`newton_velocity_tol = 0.75`, `transrate_tol = 0.25`) so that a run would
   write a *different* `abs_tol` — then the perturbation fails 4 assertions.  The general rule
   (the §7.25 rule 19 family): a *value-invariant* perturbation is not evidence of coverage,
   and a *call-count* pin is value-invariant unless the callee's inputs change too.

10. **The frame stream vs the main stream, again, for *recorded* kernels.**  The preconditioner's
    `do_apply` and the line-search `step_forward` **record** their kernels on the frame stream
    (cuda_tool equivalence), while `LDenseVector::copy_to`/`LDeviceVar::copy_to` go through the
    main stream: a test that reads the output back right after the call can read the *sentinel*
    it wrote before (measured here: `worst |z - D^-1 r| = 1.0` with `|z|` equal to the sentinel
    instead of the expected 5.5e4 — the classic shifted/stale-field signature).  Fix: sync the
    frame stream before every read-back (the test's `read_device` helper does it; a bare
    `copy_to` does not), and where a `LDeviceVar`'s implicit conversion would be used, spell it
    `var.copy_to(&frame_stream())`.

11. **The ABD tolerance row is a pure OR over the *rows that see the flag still set*, and that is
    what the device round has to model.**  With the flag seeded 1 the kernel's per-row verdicts
    are exactly the host `failed_row` results (measured 64/64 on a random batch, 32/32 in the
    device-vs-host case, 1000/1000 host-side decisions vs a hand-written loop); with the flag
    seeded **0** every thread returns early, *every* recorded verdict is false and no exchange
    happens (measured: 0/64 decided), which is the "already failed stays failed" semantics in its
    observable form.  A batch whose tolerance sits *outside* the sampled range is useless — the
    first version sampled `|dq| ∈ [0, 0.35]` against `abs_tol = 0.12` and all 64 rows failed, so
    the non-vacuity `REQUIRE` ("both verdicts occur") fired.  Sample the rotational components
    *around* the tolerance (`[-0.10, 0.10]` against `abs_tol = 0.09`: measured 40/64 failed) and
    assert both verdicts.

12. **`LineSearcher::ComputeEnergyInfo` is constructible by a test — with a *real* `LineSearcher`
    pointer.**  `ComputeEnergyInfo(LineSearcher*, LBufferView<Float>)` is public, but `dt()` reads
    `m_impl->m_dt_attr`, which only `LineSearcher::init()` sets, so the test must
    `find<LineSearcher>()` in a real scene and hand it over; then `do_compute_energy` runs the
    whole fan-out and writes the one-element `info.energy()`.  With a *test-local subreporter*
    whose energy values are known, the fan-out becomes a **differenced** statement
    (`total(on) - total(off) == the mock's own values`; measured exactly 3000.0 with the mock
    contributing 1000 + 2000), and the subreporter's `report_extent`/`compute_energy` slicing is
    observable through the size of the view it is handed (`last_count == 2`).  The mock is
    **inert** in every other scene (a static flag), which is what keeps the shared binary's other
    files unchanged.

13. **The reporter's four forwarding bodies are only reachable through *public* hooks.**  The
    reference declares `ABDLineSearchReporter`'s five `do_*` in a `private:` section (its own
    `.cu` is the only reader) and `LineSearchReporter`'s `record_start_point`/`step_forward`/
    `compute_energy`/`init` are private with `friend class LineSearcher`, which no test can drive
    before Phase 8.  The port therefore makes the five hooks public (documented as a test seam,
    the FEM sibling's `protected:` shape) so the test can drive the *same* calls the engine will:
    measured `q = q_temp + alpha·dq` to **0.000e+00** (bitwise) for the moving body, the fixed
    body untouched **bitwise**, and a *second* identical `step_forward` call reproducing the same
    `q` **bitwise** — which is what proves the start-point snapshot happened (the reference
    always evaluates from `q_temp`, never from the already-stepped `q`).

14. **Regression shape, the cold-JIT budget and the reproduction recipe.**  `[abd_solve]` →
    **245 assertions in 8 test cases** rc 0 (per case: inverse12 24, diag_apply 24,
    tolerance_boundary 60, tolerance_device 9, manager 45, line_search 16, device_vs_host 13,
    summary = the accounting case); `~[abd_solve]` → **762241/191** = the T6.3b3 baseline
    *exactly* (so the batch is purely additive: the new systems are inert in every other scene);
    the whole suite → **762486/199**; `[abd_base]`/`[abd_linear]`/`[abd_dynamics]`/
    `[abd_constitution]`/`[abd_joint_a]`/`[abd_joint_b]`/`[abd_joint_c]` = 2143/7 · 495/7 · 547/7 ·
    523/7 · 223/8 · 402/8 · 383/8 (all unchanged); `ctest` 1/1 (30.6 s); the standalone suite 69
    cases / 0 checks failed; `python scripts/luisa_module_def.py --check` rc 0 (374 symbols);
    `python scripts/luisa_check_module_abi.py` rc 0; `python scratch/gen_abd_cons_a.py --check
    --selfcheck` rc 0.  **The 12x12 Gauss-Jordan shader is cheap to JIT here** (the first
    `[abd_solve]` run took ~14 s *including* the kernel compiles of six cases; later runs are
    2–4 s), and — deliberately — **no test-local 12x12 kernel is written anywhere in this file**:
    the new inverse's device instantiation is the *shipped* preconditioner shader (§7.27 rule 2
    is why).  The per-TU syntax checks (`python scripts/luisa_check_compile.py <src> --syntax
    --filter /backends/luisa/`) are the fast inner loop: all six new TUs and the four touched
    ones compile with 0 warnings in ~10 s each.

15. **Non-vacuity: three perturbations on three different axes, all caught and all reverted
    byte-identically** (`scratch/t64a_perturb.py`, the T6.3b1/b2/b3 driver pattern: patch →
    build → run the owning case → revert → sha256 → re-verify the whole filter green).
    **P1** the *strict-comparison* axis (`abd_tolerance_checker_function.h`: the first
    `lps::abs(dq(3)) > abs_tol` → `>=`) — the host-only `abd_solve_tolerance_boundary` case,
    which pins `|dq_i| == abs_tol` ⇒ converged: **2 failures**.  **P2** the *transcribed
    arithmetic* of the shipped preconditioner (`abd_diag_preconditioner_function.h`: the apply's
    `H * r12` → `H * r12 * 0.5`), owned by `abd_solve_diag_apply` (which drives the real shader
    against the closed-form `diag_inv * r`): **1 failure**.  **P3** the *host control flow* of
    the manager (`newton_tolerance_manager.cpp`: the short-circuit `break` removed), owned by
    `abd_solve_manager`: **4 failures** (and it took the rule-9 lesson above to make this axis
    observable at all).  After each revert the module rebuilds with 0 warnings and `[abd_solve]`
    is green again (245 assertions).

16. **What this batch *claims*, and what it does not.**  No uid is claimed (the classes here are
    uid-less; the ABD checker and the preconditioner report through their managers/registries —
    the FEM sibling's spelling).  Nothing is written to `src/backends/cuda/` or
    `src/backends/common/`, no existing assertion is weakened, and the two files the export list
    touches (`luisa_module.def`) are generator-produced.  **Deliberate deviations, all reported:**
    (i) `newton_tolerance_manager.h`/`newton_tolerance_checker.h` make `init`/`pre_newton`/`check`
    (and the checker's four forwards) public — the T5.4b "pure test seam" precedent (§7.20 rule 9);
    (ii) `max_translation_checker` and `abd_tolerance_checker` got *headers* (the reference defines
    both inside their `.cu`) so the unit test can name them and drive `do_check` directly;
    (iii) the reporter's five hooks are public (rule 13); (iv) `math::inverse` gained the `N > 4`
    branch (rule 3) and `luisa_device/atomic.h` gained `atomic_exch` (rule 5).  **Still open for
    T6.4b/c** (do not read this list as "ported"): `affine_body/constraints/**`, the uid 16/23/666
    systems and the whole MAS preconditioner engine; and the port's solve path (`GlobalLinear-
    System::solve()`, `spmv`, `spmv_dot`, the global-preconditioner branch) remains T8.3's, so
    *nothing* here exercises an actual Newton solve — the preconditioner is driven through a
    standalone `GlobalLinearSystem::Impl` and the line search's iteration loop is not run.

17. **The battery's two new steps are permanent.**  `python scripts/luisa_module_def.py --check`
    (rc 0 ⇔ the committed export list matches the consumers' imports) and `python
    scripts/luisa_check_module_abi.py` (LoadLibrary + GetProcAddress on the four C entry points)
    belong in every batch from T6.3b3 on; a batch that adds a test symbol must *rewrite* the list
    (rule 7) and commit it, and a batch that touches the module's link properties must re-run both
    (the C API symbols are `UIPC_BACKEND_API`-exported, so the `.def` never controlled *them* —
    §7.27 rule 14's value-invariant axis).

### 7.29 Patterns established in T6.4b (ABD constraints: external articulation 23/24, soft transform 16, external body force 666)

**Scope.** `src/backends/luisa/affine_body/constraints/**` (a new directory, mirroring the
reference's): `external_articulation_constraint.{h,cpp}` + `external_articulation_constraint_function.h`
+ the two new generated bodies `sym/external_articulation_{revolute,prismatic}_joint_constraint.inl`
(the constitution uid **23** + the constraint uid **24** + its time integrator),
`soft_transform_constraint.{h,cpp}` (uid **16**, the ABD sibling of T5.3's FEM
`SoftPositionConstraint` uid 14) and `constraints/affine_body_external_body_force_constraint.{h,cpp}`
+ `affine_body/affine_body_external_body_force.{h,cpp}` (the pair sharing uid **666**, the ABD
sibling of T5.4b's FEM 671 pair).  Test file `apps/tests/backends/luisa/luisa_abd_constraint_test.cpp`
(tag `[abd_constraint]`, 9 cases / **471** assertions) and the **uid-16 collision resolution** in
`apps/tests/backends/luisa/luisa_abd_linear_test.cpp`.  The two generated bodies are transcribed by
`scratch/gen_abd_cons_a.py` (a second `(ref-dir, dst-dir, regions)` group was added, because the
reference keeps the articulation's `sym/` beside the constraints' `.cu`, not beside the
constitutions'), and the law was re-derived independently *before* a line of C++ existed
(`scratch/t64b_oracle.py`, the §7.17 rule 30c technique).

1. **A uid *pair* is a design, not a typo — and the class name's typo is transcribed.**  The
   articulation is an empty `InterAffineBodyConstitution` (**23**) plus an
   `InterAffineBodyConstraint` (**24**): the constitution is the (deliberately empty) owner of the
   geometry group, the constraint is where every hook lives.  The reference's own class name for
   the former is `ExternalArticulationConstituion` (sic) — **kept**, per the "never fix a reference
   inconsistency" rule; the uid is the honest identifier.  (The reference's `.cu` also keeps the
   *empty* constitution's four hooks returning zero: they are transcribed, not omitted.)
2. **The cross-reference is data-level, and the port reproduces both directions.**  The
   articulation does *not* `require<>` the joint constitutions: it reads each referenced joint
   geometry's `builtin::constitution_uid` and asserts it is **18** (revolute) or **20** (prismatic)
   — the same shape the joint-limit constitutions use for their `JointUID` — and the *uid selects
   which of the two generated bodies a kernel calls*.  The test pins both sides: the geometry's
   constitution uid (18/20), the *absent* `constraint_uids` on a plain joint geometry (measured:
   a joint geometry only carries them when a driving/limit instance was applied), and the
   articulation's own `h_joint_id_to_uid[0] == 18`.
3. **The generated bodies share eight names with the joint-limit family and *are not* the same
   law.**  Both files define `F`, `DeltaTheta` (twice), `dDeltaTheta_dQ`, `JT_G`, `JT_H_J`,
   `dDeltaTheta_dF`, `ddDeltaTheta_ddF`, so the region list needs T6.3b3's `Name#2` occurrence
   suffix; and the *revolute* body's `F` is direction-only with the **wrapped `atan2`** law while
   the *prismatic* body's `F` carries the attachment **points** (`+ qk(0..2)`) and its
   `DeltaTheta` is the **linear** (quadratic in `F`) projected half-distance.  The register reports
   (statements): revolute `F` 12, `DeltaTheta` 29, `dDeltaTheta_dQ` 102, `JT_H_J` 792,
   `DeltaTheta#2` 11, `dDeltaTheta_dF` 44, `ddDeltaTheta_ddF` 330 — *identical counts* to the
   revolute joint-limit body (the same law); prismatic `DeltaTheta` 7, `dDeltaTheta_dQ` 63,
   `dDeltaTheta_dF` 18, `ddDeltaTheta_ddF` 144.  **Never** infer "same names ⇒ same body".
4. **The expression-level oracle first, and the two shape flags it produced.**  `scratch/t64b_oracle.py`
   re-parses the *reference* `.inl` with the transcriber's own tokenizer and evaluates it against
   hand laws written out there; worst relative residuals over 20 random states: revolute `F`
   1.7e-16, `DeltaTheta(F,F_t)` 5.0e-16, `DeltaTheta` (6-vector) 5.6e-16, `dDeltaTheta_dF` 1.9e-10,
   `dDeltaTheta_dQ` 2.6e-10, `ddDeltaTheta_ddF` 7.7e-07 (a *twice*-differenced transcendental law:
   the residual is the FD's roundoff floor, which is why the Python Hessian check uses `h = 1e-4`
   and a 1e-6 bound), `JT_G` 7.8e-11, `JT_H_J(bk,bl,bm,bn)` 2.1e-10, `JT_H_J(bk,bl,bk,bl)` 2.8e-10;
   prismatic the same order (`F` 2.2e-16, `ddDeltaTheta_ddF` 2.2e-08, `JT_G` 1.5e-10).  The flags:
   the **off-diagonal fraction of `ddDeltaTheta_ddF` is 1.000e+00** for both bodies — the element
   tensor is strongly non-diagonal, so the `make_spd` clamp on `G_theta * ddDeltaTheta_ddF` is a
   *real projection* and the sweep count is observable (§7.26 rule 10's axis).  The prismatic one is
   additionally *purely* off-diagonal (its diagonal fraction is 4.4e-08).
5. **The uid dispatch is a branchless select — and its two ABI assumptions are pinned by a device
   probe.**  The reference's `if(uid == RevoluteUID) … else if(uid == PrismaticUID) … else { assert;
   zero }` reads *device* data, so the port evaluates both candidates and selects, with the
   else-branch's `UIPC_KERNEL_ASSERT(false, …)` routed through the `LC_LOG` facade (compiled out by
   default) and its zero selected.  `abd_external_articulation_detail::select_if(p, a, b)` is
   `p ? a : b` — spelled once, with the argument names carrying the polarity, because
   `scalar_detail::select_container(f, t, p)` means `p ? t : f` (§7.26 rule 13).  The dispatch rests
   on two ABI facts that no shipped kernel shows directly, so the test *measures* them with a tiny
   test-local kernel (no `make_spd` chain, so §7.27 rule 2 does not apply): the two `U64` kernel
   arguments arrive in the declared order (`a 18 b 20`), and `lcrt::select(10, 20, p)` is
   `p ? 20 : 10` for both a `Var`-vs-`Var` and a *buffer-read* `Expr`-vs-`Var` predicate
   (measured `eq-> 10 ne-> 20 flag 0 (read==a)-> 20 eq 1 ne 0 v 18`).
6. **The `make_spd` sweep count is observable here (T6.3b1's rule 19 flipped twice).**  With
   `G_theta = -3.25` (a negative coefficient, so `HF = G_theta * dd` is indefinite — measured
   `min eig < -1e-3`), the shipped 8-sweep projection differs from the 0-sweep one by
   **6.331e-01** while 8-vs-20 is **1.181e-15**; `kMakeSpdSweeps == 8` is `REQUIRE`d, and the test
   asserts `|HF|max = 3.031306e+00`-scale distance stays > 1e-3.  (A 0-sweep `make_spd` is the
   clamped *diagonal*, so a non-diagonal `dd` makes the difference large — the same argument as
   §7.26 rule 10 / §7.27 rule 1.)
7. **`FastSegmentalReduce` has no LC equivalent that computes its value in-kernel: port two
   additions.**  `lla::segmented_reduce` reads *both* the sorted keys and the values from buffers,
   so the port (a) materialises `joint_joint_id_to_key` = `ij(0)` **once in `do_init`** (the layout
   never changes; the keys are ascending by construction because the articulations' joint offsets
   are a prefix scan) and (b) adds a `joint_joint_id_to_value` buffer written by a
   `..._compute_reduce_value_kernel` (`m_ij * (DeltaTheta_j - delta_theta_tilde_j)`), replacing the
   reference's in-kernel `GetKey`/`GetValue` functors.  The reduce's output `joint_id_to_G_theta`
   is checked against the hand sum, and it matched **exactly** (`-7.750149189e+01` revolute,
   `-5.921598575e+00` prismatic).
8. **The `ref_dof_prev` protocol needs a *padded* buffer and a clamped index.**  The reference's
   `ref_q_prev_ids[s] >= 0 ? ref_q_prevs(id) : q_prevs(body)` is a device predicate, so the port
   reads *both* candidates: the discarded one indexes the `-1` sentinel, hence the index is clamped
   to 0 **and `ref_q_prevs` is padded to at least one element in `do_init`** (an LC `BufferView`
   with a zero length cannot be a kernel argument at all).  The test drives the path with a
   `ref_dof_prev` attribute whose value *differs* from the body's own state — without it every
   scene's `DeltaTheta` is exactly 0 (the port's solve path is Phase 8's, so `q` never moves).
9. **The assembly needs the animator's *step*, and the differencing must be done in one engine.**
   An `InterAffineBodyConstraint`'s assembled contribution is **invisible before
   `InterAffineBodyAnimator::step()`** (§7.26 rule 12), so the assembly case calls `world.advance()`
   first; and instead of the two-engine idiom the [abd_joint_c] assembly case used (a second
   `Engine`'s teardown is where T6.3b1 recorded the multi-engine hazard), the *base* assembly here
   is the same world with the constraint's `joint_joint_id_to_mass` **device buffer zeroed between
   the two assemblies** — the extents depend on the counts, not the values, so the COO layout is
   identical and the difference is exactly the articulation's.  Measured: 128 triplets, 48/48 keys,
   revolute worst `|block|` dev **2.32e-14** and gradient rel **3.53e-14**, prismatic **1.25e-12**
   and **2.97e-13**; `|b| ≈ 7.96e4…9.30e4` (the *joint's* κ = 1e8 term dominates, so the
   difference is a small residual — the §7.25 rule 16 discipline: state which keys speak).
10. **The `dof_offset` meta is `-1` until the global linear system has been initialised — an oracle
    built from it writes out of bounds.**  `body_dof_offsets`-style oracles must call
    `prepare_dof_layout(...)` (a throwaway `GlobalLinearSystem::Impl` + `init()`, §7.22 rule 12)
    *before* reading the per-geometry `builtin::dof_offset` metas: read too early, the second body's
    offset is still `-1` and `want_full(-1 + …)` corrupts the heap — measured as a **0xC0000374 heap
    corruption** that killed the *rest of the filter* after the case, with the case's own assertions
    looking "plausible" (the shifted `want` matched 9 of the values at the wrong dofs).  The fix is
    both the ordering and an explicit `T64B_CHECK(c, offsets[1] == 12)`.
11. **A test-built `ComputeEnergyInfo` needs a *real* `global_animator`.**  `BaseInfo::substep_ratio()`
    dereferences `m_impl->global_animator`, so the reference's own shape gives a **SIGSEGV** (not a
    Catch2 failure) for a test-built `Impl`: the helper must carry the scene's
    `find<GlobalAnimator>()` and `REQUIRE` it.  The same helper drives the three shipped kernels
    through the *public* info constructors (§7.26 rule 12's harness) — the direct drive is the
    primary oracle for the k1/k2 write patterns (the 2 doublets' segment indices 0/1 and values, and
    the 3 twelve-by-twelve blocks at `(0,0)`, `(0,1)`, `(1,1)` — the reference's `upper_LR`), while
    the assembly case adds the COO/gradient differencing on top.
12. **`on_write_scene` is driven *only* by the engine's dump path.**  `SimEngine::event_write_scene()`
    is called from `do_dump` (measured in `sim_engine.cpp`; `do_retrieve()` is a no-op until
    Phase 8), so a test that wants a constraint's attribute write-back (`delta_theta`) must call
    `world.dump()` — doing so both exercises `write_scene` and pins the *same* device buffer the
    update-state kernel wrote (measured: the update-state device value and the frontend attribute
    both `-1.040960811e+00` == the hand law).  Without the dump call the attribute keeps its
    frontend default (measured `0.0`).
13. **The soft transform is the FEM sibling's contract with a different *input*.**  One
    `AffineBodyConstraint` (uid 16), the same `is_constrained` / `strength_ratio` / `aim_*`
    attribute contract, the same `is_fixed`-guarded energy/gradient/Hessian — but the dof is a
    **12-vector** (`q`, not a vertex position) and the aim is `transform_to_q(aim_transform)` from
    the *instances*' `Matrix4x4` (shared with `transforms` by the frontend, so writing it moves the
    geometry too — the *dynamics'* `q` is a copy taken at `init()`, which is how the case produces a
    `dq ≠ 0` without a Newton solve).  Its one non-obvious law is
    `M_c = s_r M + (s_t - s_r) M_cm` (`M_cm` = the ABD mass of a point mass at the centre of mass):
    the test pins the shipped routine against a hand rebuild of `ABDJacobiDyadicMass::to_mat()` from
    its documented `(m, m c, m c c^T)` layout, and the four identities `M_c(s,s) = s M`,
    `M_c(1,0) = M_cm`, `M_c(0,1) = M - M_cm`, `M_c(a,b) + M_c(b,a) = (a+b) M` — plus PSD for a
    *physically consistent* triple (`S_dy - m c c^T ⪰ 0`, a Schur-complement condition; an
    arbitrary symmetric `S_dy` is **not** PSD and the check must not be run on one — measured
    −1.6 eigenvalues).
14. **The external-body-force pair is a *third* 666.**  The constraint (an `AffineBodyConstraint`,
    reading the per-instance `external_force` **Vector12**) and its reporter (an
    `AffineBodyExternalForceReporter`) both claim 666 — a *different registry* from the frontend
    constitution's 666 (the T5.4b 671 / T6.3b2 667 / T6.3b3 668 precedent), and the manager claims
    666 too: the test prints and `REQUIRE`s **all three**.  The reporter's `eigen::atomic_add` is
    the *scalar alias* of the AoS buffer (`forces.native().as<double>()` + twelve `fetch_add`s), and
    the test pins the atomicity with a synthetic table holding **duplicate body ids** (measured:
    body 0 received the *sum* of its two entries, `3×` the frontend force).  The constraint's table
    is driven *differenced* (`after - before`) because the buffer is not cleared by the reporter.
15. **The reference's empty-upload quirk is transcribed, not fixed.**  The ABD constraint guards its
    two uploads with the reference's `if(!h_forces.empty())`, so an empty frame leaves the device
    buffers at their previous contents — the FEM sibling (671) deliberately *fixed* exactly this by
    always resizing; the port documents the divergence at the call site instead of importing the
    fix (`transcribe + pin`).
16. **The uid-16 collision, measured first (the T6.3b1 way).**  `luisa_abd_linear_test.cpp` claimed
    uid 16 for its synthetic `T61bConstraint` from T6.1b on.  With the *real*
    `SoftTransformConstraint` registered but the mock still at 16: `[abd_linear]` → **rc 42, 411
    assertions / 2 failed** at `luisa_abd_linear_test.cpp:2368/2369`
    (`T61bConstraint::geo_info_count == 2u` and `geo_ids.size() == 2u`, both read **0**) and the
    **full suite** → rc 42, 762402 assertions / 199 cases (2 failed) — the *real* system won the
    group, exactly as the animator's "last equal-uid entry wins" map predicts (both constraints sort
    equal under `std::sort(by uid)`, which is a stable-in-practice insertion sort at this size, so
    the *registration* order — `SimEngine::build_systems()`'s demangled-type-name order, where
    `t61b::T61bConstraint` < `uipc::backend::luisa::SoftTransformConstraint` — decides).
    **Resolution** (the T6.3a/T6.3b1 "move the mock" precedent, no assertion weakened): the mock
    moved to uid **27** (`uipc::constitution::AffineBodyDrivingSphericalJoint`, frontend-registered
    *Constraint*-typed, grep-proven unused under `src/backends/luisa`, `src/backends/cuda` and
    `apps/tests/backends/luisa`), and `claim_mock_constraint_uid(sc)` rewrites the bodies'
    `constraint_uids` entry 16 → 27 right after `SoftTransformConstraint::apply_to` wrote it.  16 is
    then *absent* from the scene's constitution tabular, so the real constraint is not even built
    (its `do_build` throws the "requires Constraint UID=16" `SimSystemException`, the port's
    explicit not-in-play signal) while the mock owns exactly the two bodies it owned before.
    **Re-measured: `[abd_linear]` → 495 assertions in 7 test cases, rc 0 — the recorded T6.1b number
    exactly.**
17. **Transcribing a *flag* into an oracle is a trap; name the parameter after what it means.**
    The first version of the new test passed its local `prismatic` bool into helpers whose first
    parameter meant `revolute`, so the hand oracle silently computed the *other* body's law — and
    the *assembly* case "failed" with a plausible-looking residual while the **port was right**.  The
    measurement that separated the two was three prints: (a) the *device* basis vs the hand basis
    (dev `0.000e+00`), (b) the hand `DeltaTheta` vs the device's *written* `joint_id_to_delta_theta`
    (equal), and (c) the *other* branch's hand value (which matched the device's *first* reading
    exactly).  Lesson: when an oracle and the kernel disagree, print the *intermediate* quantities
    (the basis, the reduce's output, the branch constants) before touching the port — and prefer a
    flag whose name matches the parameter it feeds.
18. **The export list again: 374 → 390 symbols, and the *vtable* trap did not repeat.**  16 new
    imports (`uid`, `forces`, `body_ids`, `do_report_extent`, `do_compute_energy`,
    `do_compute_gradient_hessian`, `do_step`, the constitution's extent hooks, the articulation's
    `do_compute_gradient_hessian`, …) failed the **link** loudly in two rounds (15 + 1) and were
    merged with `scratch/t64a_def_bootstrap.py`; because the test calls them **directly** (not
    through a vtable slot), `python scripts/luisa_module_def.py` (**rewrite**) kept every one of them
    and `--check` is rc 0 — i.e. §7.28 rule 7's bootstrap trap did not bite this time, which is the
    point of making the test's references *used* imports.
19. **Regression shape, the cold-JIT budget and the reproduction recipe.**  `[abd_constraint]` →
    **471 assertions in 9 test cases**, rc 0 (per case: closed_forms 93, finite_difference 34,
    assembly 86, soft_transform 65, external_body_force 21, uid_ownership 10, scene 34,
    device_vs_host 3, summary = the accounting case; the 346 `T64B_CHECK`s are the file's own
    accounting and the rest are `REQUIRE`s and the summary's checks).  `~[abd_constraint]` →
    **762486/199 = the T6.4a baseline *exactly*** (the batch is purely additive: `~[abd_constraint]`
    is the whole suite, since the tag did not exist before).  The whole suite → **762957/208**
    (= 762486+471 / 199+9).  `[abd_base]`/`[abd_linear]`/`[abd_dynamics]`/`[abd_constitution]`/
    `[abd_joint_a]`/`[abd_joint_b]`/`[abd_joint_c]`/`[abd_solve]` = 2143/7 · 495/7 · 547/7 · 523/7 ·
    223/8 · 402/8 · 383/8 · 245/8 (all unchanged).  ctest **1/1** (28.96 s), standalone **69/0**,
    `python scratch/gen_abd_cons_a.py --check --selfcheck` rc 0, `python scratch/t64b_oracle.py`
    rc 0, `python scripts/luisa_module_def.py --check` rc 0 (**390** symbols),
    `python scripts/luisa_check_module_abi.py` rc 0.  **Cold JIT:** the first filtered run of the new
    file took **~15 min** (the articulation's five kernels + the soft transform's three + the
    reporter's one, i.e. three fresh generated bodies — two of them 12x12 `make_spd` chains), every
    later run **3–4 s**; a *kernel-source* perturbation therefore costs a fresh JIT (~35 s per
    affected case in the warm-cache runs measured here) while a *host-only* one stays warm.
20. **Non-vacuity: two perturbations on two axes, both caught, both reverted byte-identically**
    (`scratch/t64b_perturb.py`, the T6.3b1/b2/b3 driver pattern: patch → build → run the *owning*
    case → revert → sha256 → re-verify the whole filter green).  **P1** the *host* registry axis
    (`external_articulation_constraint.h`: `ConstraintUID 24 → 25`) — the constraint's uid then
    leaves the scene's tabular and the *animator* aborts loudly (`rc 3221226505`,
    "InterAffineBodyAnimator: No responsible backend SimSystem registered for constraint uid 25") —
    the abort-shaped failure mode of §7.27 rule 15.  **P2** the *transcribed-arithmetic* axis
    (`sym/external_articulation_revolute_joint_constraint.inl`: the first `lps::detail::div(1.0, 2.0)`
    → `div(1.0, 3.0)`) — owned by the **closed-forms** case, which drives only host
    (`S = double`) instantiations and therefore re-JITs nothing: **rc 42, 93 assertions / 4 failed**
    (lines 1204 ×3, 1284).  Both reverts are byte-identical (sha256 `2de0ffa1…` / `afe4effb…`), the
    generated file's revert is proven by re-running the generator, `--check --selfcheck` is rc 0
    after each, and the reverted tree rebuilds (rc 0) and runs `[abd_constraint]` **471/9 green**.
    A third, *cheap* axis worth recording: the test's own *uid probe* kernel pins the two ABI
    assumptions that the dispatch rests on (rule 5), so a future reviewer does not have to trust them.

### 7.30 Patterns established in T6.4c (the MAS preconditioner engine + the FEM MAS preconditioner)

**Scope.** The last M6 batch and the hardest device-code one: `finite_element/mas_preconditioner_engine.{h,cpp}`
+ `mas_preconditioner_engine_function.h` (the port of the reference's 205 + 1726 lines), the
`finite_element/fem_mas_preconditioner.{h,cpp}` + `fem_mas_preconditioner_function.h` (the port of its
473-line driver), the `"mas"` entry of the config selection (the diagonal preconditioner's documented
deferral became a two-way contract), the refreshed module export list and
`apps/tests/backends/luisa/luisa_mas_preconditioner_test.cpp` (tag `[mas_preconditioner]`, **7 cases /
234 assertions**). The engine is the StiffGIPC MultiLevel Additive Schwarz preconditioner: 16-node banks,
a hierarchy of "banks of banks", a 48x48 Galerkin cluster matrix per bank per level, a column-oriented
48x48 Gauss-Jordan inverse per cluster and `z = M^{-1} r` with injection prolongation.

1. **The kernel map is the reference's, one for one.** The 14 named `__global__` functions of the
   reference (`build_connect_mask_L0/Lx`, `prepare_prefix_sum_L0`, `build_level1`, `next_level_cluster`,
   `prefix_sum_Lx`, `compute_next_level`, `aggregation`, `scatter_hessian_to_clusters_k1/k2`,
   `invert_cluster_matrices`, `build_multi_level_R`, `schwarz_local_solve`, `collect_final_Z`) became 14
   `cached_kernel` shaders in `mas_preconditioner_engine.cpp` and the class keeps the reference's whole
   public step list, so the unit test can drive the engine step by step. **Read the reference before
   writing**: the colouring (per-bank elected-lane prefixes), the level table and the two scatter passes are
   not reconstructable from the algorithm's description (the second pass's packed-warp-sum coarsening is
   only visible in the code).
2. **Storage deviations, all forced by the DSL, none behavioural.** `ClusterMatrixSymT<Scalar>`
   (`Eigen::Matrix<Scalar,3,3> M[136]`) is not an LC buffer element (a C-array member is not reflectable
   and a custom-struct element has no `fetch_add`), so the port stores the **same bytes** as one flat
   scalar buffer, indexed `cluster * 1224 + sym_index(br,bc) * 9 + (3*j + i)`; that is also what makes the
   reference's `cuda_tool::eigen::atomic_add` (9 scalar atomics) expressible as
   `DenseVectorView::atomic_add`. `Int2` and `LevelTable` are flattened the same way (`level_sizes` = 2
   ints per level, `coarse_tables` = MAX_LEVELS ints per vertex), and the multi-level `Vector3f`/`float3`
   buffers are flat floats (3 per node). The kernel *code* then reads like the reference's.
3. **`Var<Buffer<T>>` is move-only (again), now inside a helper signature.** `load_cluster_block(HView H,
   …)` with `HView` taken **by value** hit T5.4b's `C2280` at the first *kernel* call site; the helpers take
   `const HView&`. (§7.20 rule 11's trap is not just about the reference's own signature — every new helper
   that receives a `BufferVar` has to spell `const&`.)
4. **The generic-index gap: `sym_index`/`mat_entry`/`lanemask_lt` must be templated on the index flavour.**
   The kernels call them with **DSL** indices (`vert_col % 16`) while the host/oracle code calls them with
   `int`s, and a `constexpr int sym_index(int,int)` simply does not accept an `Expr`. The port's helpers are
   `template <typename R, typename C> auto sym_index(const R&, const C&)` with pure integer arithmetic, so
   one body serves both; `cluster_block_base` (host-only) had to drop `constexpr` for the same reason.
5. **Three more scalar gaps closed in `mas_preconditioner_engine_function.h`** (the file that exists for
   them): `popcount_scalar`/`ctz_scalar` (`lcrt::popcount`/`lcrt::ctz` vs `std::popcount`/`std::countr_zero`),
   `min_scalar`/`max_scalar` (`lcrt::min` is not reachable from host code, `std::min` is not a DSL
   intrinsic), and `as_u32`/`as_i32` (a `static_cast<uint32_t>` of a DSL value is a compile error — the
   first version of the engine had 32 of them and the fix is a two-line dispatcher). `__ffs(x) - 1` is
   `ctz_scalar`, and `ctz(0)` is 32 on both sides (the callers test `j < lane_count` instead of relying on
   it).
6. **RULE M2 for the reference's data-dependent loops: a fixed bound plus a `live` flag.** The closure
   loops (`while(connect_msk != ~0U) … __ffs(todo) - 1 …`) and the level walks (`while(next_id >= 0 &&
   level < level_num - 1)`) are rewritten as `$while`/`$for` loops with a host bound (BANKSIZE, MAX_LEVELS,
   `level_num`) and a `live` predicate. The closure's result is the **least fixpoint** of "merge the masks of
   the lanes you are connected to", which is order independent, so the fixed 16-iteration version computes
   exactly what the reference's early-exiting loop computes (pinned by the level-table case: 192/192 table
   entries and 64/64 `going_next` entries equal for three meshes, including a *reversed* partition).
7. **The surplus-thread gate is load-bearing (a real bug the port had to fix, measured).** The reference
   lets the surplus threads of a partial block `return` before any shared-memory or atomic work. The port
   dispatches whole blocks and predicates instead (it cannot return around a block-wide barrier), and the
   first version therefore let the *clamped* surplus lanes of `next_level_cluster` run their elected-lane
   accumulation: a clamped lane whose mask is only its own bit passes
   `popc(mask & lanemask_lt(lane)) == 0` and counts as an elected lane. Measured effect: **16** level-2
   nodes instead of 1, i.e. `level_sizes[3].x = 16` (the whole logical warp) — caught by the level-table
   oracle, fixed with `& in_range` on the two elected-lane accumulations (and on the level-size write).
   **Rule: every side effect of a clamped lane in a kernel with warp/shared collectives needs an
   `in_range` gate, and the portable way to *see* the mistake is an independent structural oracle.**
8. **A zero-length buffer view is not a kernel argument — pad it.** The engine's 1-vertex unit test (and
   any scene with no cross-bank edge) passes an empty neighbor list, and LC's `LBufferView::native()` of a
   never-allocated buffer is a *null* handle: the dispatch dies with a **SIGSEGV inside the test process**
   (not a CUDA error). The fix is T6.4b's "padded buffer + clamped index" idiom: the neighbor list keeps one
   unused element (§7.29 rule 8) and the host copy uses the exact subview.
9. **THE FLAT-VIEW TRAP: `A.values().size()` is not the triplet count.** The reference passes
   `CBufferView<Eigen::Matrix3d>` (one element per triplet) to `set_preconditioner` and takes
   `triplet_values.size()` as the dispatch count; the port's BCOO container exposes a **flat scalar** view
   (`9` doubles per triplet, T3.1's layout), so the same expression meant **9x too many "triplets"**. The
   surplus iterations read past `row_indices()`/`col_indices()` (whose tail is zero-filled by the geometric
   growth) and past the values (whose tail the converter leaves *uninitialised*), i.e. they scattered
   garbage blocks into the fine cluster. Measured symptom: the assembled cluster-0 matrix had `4.0` in the
   fixed vertex's block instead of the reader's `55.66`-scale values, and the fine-level apply was off by
   1.2e-1 (a 100% error) while the *synthetic* cases still passed — their garbage triplets were rejected by
   the `0 <= row < total_nodes` guards. The fix is `size() / 9` (documented at the call site), and the
   lesson is general: **a count taken from an linalg view must be divided by the block size.**
10. **`LocalPreconditioner::do_init` is only reached through `GlobalLinearSystem::Impl::init()` — so a
    scene alone leaves the MAS engine uninitialised.** `do_init` is where the hierarchy (the partition, the
    CSR, the level table) is built; in the *engine* flow only Phase 8's `SimEngine::do_init` calls
    `GlobalLinearSystem::init()` (the port's `sim_engine.cpp` still says so), so a scene builds the
    preconditioner's `do_build` (the config selection, the registration) but never its `do_init`. The test
    therefore drives the real preconditioner through a standalone `GlobalLinearSystem::Impl` exactly like
    T5.4b/T6.4a, and **asserts the uninitialised state first** so the contract is documented rather than
    implied. Consequence for a future reader: `m_has_partition`/`is_initialized()` are false in every scene
    of this port today — that is Phase 8's wiring, not a defect.
11. **The reference's fixed-row substitution is in the assembled BCOO, and the fixed vertex's row is
    *zero* off the diagonal (measured, not assumed).** `fix_hessian_row` (T5.4a) replaces a fixed vertex's
    diagonal block by `Identity()` and every off-diagonal block that touches it by `Zero()`, and the
    converter's segmented sum adds the (1 kinetic + #incident-tets) diagonal triplets into
    `(1 + #tets) * Identity` — measured here as exactly `4.0` for the 5-tet scene's vertex 0, with its five
    `(0,j)` triplets at `|H|max = 0.000e+00`. The test's cluster-matrix oracle models exactly that (and the
    engine agrees to 1.4e-14, i.e. a few ulps of a `4e2`-scale sum whose *atomic* order is not fixed —
    asserted at 1e-12, not 0, and that 1.4e-14 is *measured*, not a guess).
12. **The cluster-inverse oracles: hand-derived systems first, Eigen second, and the fix-up is part of the
    oracle.** For analytically known cases (a one-vertex cluster: `blockdiag(H^-1, I_42)`, exact to 0.0; a
    two-vertex cluster: the explicitly written 6x6 inverse, worst 2.6e-08 because the inverse is stored in
    **float**, GIPC parity) the expected value is written out in the test; for a random SPD 8-vertex stencil
    the oracle is (a) the reference's own scatter rule (an upper triplet adds `H`, a lower one adds `H^T` to
    the mirrored block), (b) the gather's symmetric reconstruction with the **zero-diagonal pin**
    (`if(row == col && s_mat == 0) s_mat = 1`), (c) Eigen's `inverse()` (LU with partial pivoting — a
    different algorithm from the ported column-oriented Gauss-Jordan) and (d) the reference's
    post-elimination 3x3-block fix-up applied to the *inverse*. Measured: the rebuilt dense matrix agrees
    with the hand-written SPD system **exactly** (0.0), the cluster inverse with Eigen's **1.3e-08**
    (`|D·inv − I|` diagonal max 4.9e-08), and a *zero* cluster gives the exact identity (the reference's
    deterministic singular answer). The engine's 48x48 Gauss-Jordan is thus pinned on three hand-derived
    systems and one random one; the 2x2 hand trace that proves the "odd" pivot-row division cancels out is
    recorded in the test's comments.
13. **The level table / colouring oracle is a union-find, and it exposed the two index spaces.** The
    test derives the hierarchy with a plain union-find over the same hand-written adjacency (per-bank
    components, the `(bank, lowest lane)` component ranking, the level boundaries, the per-vertex chain) —
    a completely different algorithm from the engine's bit-mask closure — and compares it against the
    engine's own buffers: **fine masks 32/32, `going_next` 64/64 and `coarse_tables` 192/192 equal** for a
    32-vertex chain, the same chain with a *reversed* partition and an 8x4 triangulated patch. The
    reversed-partition case is what pins the reference's two index spaces: `fine_connect_masks` is indexed
    by the **partition position** while `going_next`'s level-0 region and `coarse_space_tables` are indexed
    by the **real vertex index** (they only coincide for an identity partition, which is why the first
    version of the oracle — position-indexed — passed the identity case and failed the reversed one
    32/64). Also note the *post-closure* masks: a fully connected bank ends up with `0xffff` for every
    vertex, so the interesting hand-derivable quantities are the pre-closure masks (pinned separately).
14. **The coarse level is aggregation Galerkin, in two shapes.** With one connected component in a
    level-0 bank (`prefix == 1`) the reference's k2 sums every block of the fine cluster into one coarse
    node's diagonal block (the test's hand-derived 3x3 = `A + B + 2C + 2C^T`, matched to 1.6e-09); with
    more than one component each lane walks up on its own and the coarse block `(a,b)` accumulates the fine
    blocks whose ancestors are `a`/`b` — and that branch adds `mat3` under `cdx >= rdx` into
    `sym_index(rdx % 16, cdx % 16)`, which for *interleaved* components can be a **lower**-triangle slot
    (the formula is only the symmetric index for `row <= col`). The test keeps its meshes in the contiguous
    regime (a chain, a grid, two contiguous chains) where the two agree; the interleaved case is a *latent
    reference quirk* recorded here rather than pinned (no test constructs it).
15. **The apply oracle is the fine level only, and that is what makes it checkable.** The engine's
    `set_active_level_num(1)` (public, the reference's own API) makes the apply a pure per-cluster block
    Jacobi, so the test builds the oracle from the *read-back assembled BCOO*: for every vertex `v` in
    cluster `c`, `z_v = Σ_{q ∈ c} M_c^{-1}[i_v, i_q] r_q`. Measured on the 5-tet scene: worst absolute
    **4.9e-09** (scale 1.25e-01, `z^T r = +9.01e-01`), 24/24 entries bitwise identical across two runs, the
    `converged != 0` early-out keeps all 24 sentinels, the MAS result differs from the block-*diagonal*
    oracle by 1.1e-02 (**so the selection is not a silent fallback to the diagonal preconditioner**) and the
    full hierarchy differs from the fine level by 4.1e-04 (so the coarse levels really contribute). The
    same case also compares the engine's *assembled* cluster-0 matrix and its inverse with the oracle
    (1.4e-14 / 5.4e-10) — the same check case 1 makes on synthetic meshes, here on the real scene's A.
16. **The compile-deadlock trap was avoided by construction, and the barrier forced a padding.** No kernel
    here holds a `make_spd`/12x12 chain (the largest expression is a 3x3 block), so §7.27 rule 2 never bit;
    the 48x48 Gauss-Jordan is a *loop* (`$for(pivot, 0, 48)`), not an unrolled chain, and its four
    `sync_block()`s per pivot must be reached by every live thread — so the port **pads the cluster buffers**
    (≤ 1 cluster) to make `m_total_num_clusters * 3` a whole number of 96-thread blocks instead of letting
    the surplus lane return around the barrier: the padded lanes invert a zero cluster (whose diagonal the
    kernel pins to 1, i.e. an exact identity inverse) and their result is never read. The one *host* cost is
    a slightly larger allocation, documented at `init_matrix`/`invert_cluster_matrices`.
17. **The GPU block size must be a multiple of 32 — a *new* harness trap.** The banks are 16 lanes wide,
    but `set_block_size(16)` is rejected by LC at dispatch time with a **fatal
    `Assertion 'thread_count <= 1024u && thread_count % 32u == 0u' failed` (process abort, rc 3221226505)**;
    the test-local closure kernel therefore uses a 32-thread block (and predicates the idle lanes). The
    engine's own kernels use 256 and 96 (both multiples of 32).
18. **The config selection is a two-way contract with printed diagnostics.**
    `FEMMASPreconditioner::do_build` throws `SimSystemException` unless
    `linear_system/fem_preconditioner == "mas"`; `FEMDiagPreconditioner::do_build` throws when it *is*
    "mas". Measured on the same scene: with `"mas"` the MAS preconditioner is valid and the diagonal one is
    invalid (`valid=1 / valid=0`), without the key it is exactly the other way round (`0 / 1`), and the
    deferral reason is the debug line `SimSystemCollection::build_systems()` logs
    (`[{system}] shutdown, reason: …`) — the test raises the log level around the scene build so the
    diagnostic is visible in the log, i.e. never a silent fallback. `python scripts/luisa_module_def.py` +
    `--check` rc 0 and `scripts/luisa_check_module_abi.py` rc 0 stay in the battery (§7.28 rule 7): this
    batch needed **4** new exports (`MASPreconditionerEngine::{init_neighbor,init_matrix,set_preconditioner,
    level_size}`), bootstrapped from the `LNK2019` names and then rewritten by the generator (390 → **394**
    symbols); the `FEMMASPreconditioner` hooks are vtable-only, so they were *not* needed — the test drives
    them through the class's `do_*` methods, which are already exported for T5.4b's sibling.
19. **Non-vacuity: two perturbations on two axes, both caught, both reverted byte-identically** (below in
    the T6.4c report). The host axis (the config reader) is cheap; a *device* axis costs a fresh JIT of the
    affected engine kernels (the whole `[mas_preconditioner]` filter cold-JITs in ~40 s here, warm runs in
    ~3 s).
20. **Regression shape and the reproduction recipe.** `[mas_preconditioner]` → **234 assertions in 7 test
    cases** (per case: cluster_inverse 12, level_table 73, preconditioner_apply 18, config_mas 7,
    config_default 4, device_vs_host 46 + the accounting case); `~[mas_preconditioner]` → **762957/208 =
    the T6.4b baseline exactly** (the batch is purely additive: the new systems are inert unless a scene
    selects `"mas"`); the whole suite → **763191/215**; `[fem_precond]`/`[fem_linear]`/
    `[fem_dynamics]` = 104/4 · 323/4 · 135/8 (unchanged; note the filter is `[fem_precond]`, the
    T5.4b tag, not `[fem_preconditioner]`); `ctest` 1/1 (29.96 s); the standalone suite 69 cases / 0
    checks failed; `python scripts/luisa_module_def.py --check` rc 0 (**394** symbols);
    `python scripts/luisa_check_module_abi.py` rc 0. Total ~5200 new/changed port lines. **Reproduction:**
    `python scripts/luisa_build.py build backend_luisa` (rc 0, 0 warnings) then
    `python scripts/luisa_build.py run "[mas_preconditioner]"`; each case prints the numbers quoted above.

### 7.31 Patterns established in T7.1a (the collision-detection BVH batch: AtomicCountingLBVH, LinearBVH, StacklessBVH)

**Scope.** The first M7 batch: `collision_detection/aabb.h` (the host AABB **plus** the new device element
type), `collision_detection/linear_bvh.{h,cpp}` + `details/linear_bvh.inl` (the port of the reference's
268 + 716 lines), `collision_detection/atomic_counting_lbvh.{h,cpp}` + `details/atomic_counting_lbvh.inl`
(48 + 16 + 154), `collision_detection/stackless_bvh.{h,cpp}` + `details/stackless_bvh.inl` (195 + 1217),
the shared AABB-union reduction (`collision_detection/details/aabb_reduce.h` + `aabb_reduce.cpp`), the
refreshed module export list and `apps/tests/backends/luisa/luisa_bvh_test.cpp` (tag `[bvh]`, **8 cases /
219 assertions**). The three structures are what T7.2's trajectory filters and M8's contact detection query;
their *node ordering*, their Morton codes, their internal-AABB aggregation and their traversal order are the
contract, so every deviation below is either a transcription of the reference's own quirk or a strictly
stronger ordering, and the [bvh] test pins the values (Morton codes, the exact child-union property, the
leaf permutation, brute-force candidate sets including a 100k-primitive tree).

  1. **The kernel map is the reference's, one for one; the file split is not.** The reference is
     header-only *because CUDA requires it* ("Otherwise, if using dllexport/dllimport, cuda will throw
     `InvalidDeviceFunction`", its own NOTE). The port has no nvcc, so `linear_bvh.cpp` /
     `atomic_counting_lbvh.cpp` / `stackless_bvh.cpp` hold the `__global__` bodies as `lla::detail::cached_kernel`
     shaders (compile once, dispatch per frame) and the `details/*.inl` files keep only what a *kernel body*
     must see: the Morton arithmetic, the node predicates, the device-side traversal (`LinearBVHDeviceViewer`)
     and the predicate-templated traversal kernels. The reference's `namespace detail` helpers, its
     `uipc::culbvh` vendored helpers and its public API names are all preserved (a reader can diff the two
     file sets line by line).
  2. **The device AABB is a *new* element type and it is deliberately not LC's `float3`.** `AABB` stays the
     reference's host `Eigen::AlignedBox<float,3>`; device buffers hold `LAABB
     {LVec<float,3> min, max}` (24 B, align 4 == Eigen's bytes, `static_assert`ed). LC's `float3` is
     `Vector<float,3>`, which is `alignas(16)` and therefore **16 bytes**, so a `{float3,float3}` element
     would be 32 B and every `LinearBVHNode` would grow by 8 B of padding. `lps::LVec<float,3>` is a plain
     `std::array<float,3>`, which is exactly why the numerics layer exists; it needs its own
     `LUISA_STRUCT(uipc::backend::luisa::LVector3f, data)` registration, and `LAABB`'s registration must come
     **before** the first `Expr<LAABB>` use (an unregistered struct's `Expr` has no members — measured C2039
     in `scratch/t71a_spike.cpp`). Kernel-side access is the established `load_aabb`/`store_aabb` idiom, two
     levels deep (`e.min.data[k]`).
  3. **An LC store is a *whole element* store, so a per-member update becomes a read-modify-write — split
     the kernel instead of racing.** The reference's `linear_bvh_build_internal_nodes_kernel` writes the
     internal node's `object_idx`/`left_idx`/`right_idx` *and* its two children's `parent_idx`
     (`nodes(nodes(idx).left_idx).parent_idx = idx;` — a 4-byte store to a different *word* of the child,
     which the child's own thread may be writing concurrently: different words, so neither store is lost).
     LC cannot address a member of a buffer element, and a whole-struct read-modify-write in the same
     dispatch would lose the child's own fields. The port therefore keeps the reference's two *phases* as two
     dispatches — `internal_nodes_kernel` (the node's own fields) then `internal_parents_kernel` (the
     children's parent links) — which are stream-ordered and produce a bit-identical node array.
  4. **`Var<Struct>` default-initialises *every* member — the reference relies on the buffer's stale value.
     This was the batch's one real bug (a GPU hang, not a wrong number).** The reference's kernel writes only
     `nodes(idx).object_idx = 0xFFFFFFFF` and leaves `parent_idx` at whatever the buffer held: the build's
     `m_nodes.resize(num_nodes, default_node)` fill (`0xFFFFFFFF` everywhere). A `lcrt::Var<LinearBVHNode>`
     is `apply_default_initializer`-ed to **zero**, so the port's first K1 wrote `parent_idx = 0` for the
     internal nodes — including the root. The root then became "someone's child"; the internal-AABB walk's
     `while(parent != 0xFFFFFFFF)` never reached a top and the whole build spun on the GPU (measured: 100 %
     GPU utilisation, one busy host core, no console output, >30 min). The fix is one line
     (`node.parent_idx = 0xFFFFFFFFu;`) and the *lesson* is general: whenever the reference writes a subset
     of a struct element's members, the port must spell out every member it depends on. The isolated
     `bvh_tree` case reproduces both symptoms (without the fix: `roots == 0` and
     `nodes[0].parent_idx == 0`; with it: `roots == 1`).
  5. **Two Morton spellings, two different functions (transcribe both, do not unify).** `linear_bvh`'s
     `detail::morton_code` clamps the *normalized* centre into `[0,1]` (`fminf`/`fmaxf`) and multiplies by
     the **double** literal `1024.0` (`static_cast<uint32_t>(float * double)`); `culbvh::morton3D` multiplies
     by `1024.0f` (**float**) and clamps into `[0,1023]`. They disagree whenever `x * 1024.0f` is not exact
     (measured on 2 of 8 probe inputs, e.g. `x == 1.0` → 1024 vs 1023). The [bvh] test's oracle is an
     independent **bit-interleave loop** (cross-checked against the reference's magic constants on all 1024
     inputs) and pins each spelling separately, bit-for-bit on the device.
  6. **The degenerate point cloud is a *division by zero* and the reference's clamp makes it well defined.**
     With all centres identical the union box has zero size, so `(centre - min) / sizes` is `0/0 = NaN` for
     every component; Eigen's `mini<float>`/`maxi<float>` are `fminf`/`fmaxf`, i.e. `fminf(NaN, 1) == 1` and
     `fmaxf(1, 0) == 1`, so the code becomes `1024` (not NaN) and all Morton codes collapse to one value
     (measured: 64 identical points → 1 distinct code). LC's `lc_min`/`lc_max` on `float` are `fminf`/`fmaxf`
     too (`cuda_device_math.h`), so the device and a host oracle written with `std::fmin`/`std::fmax` agree
     exactly — this is why the clamp is written as `max(min(x, 1.0f), 0.0f)` and never as a ternary.
  7. **The 64-bit Morton word's low half is the *sorted position*, not the object index.** The reference's
     `linear_bvh_build_expand_morton64_kernel` takes `indices` and never reads it (`uint32_t idx = i;`), so
     `m_morton_idx[i] = (sorted_morton[i] << 32) | i`; `common_upper_bits` therefore also counts the
     *position* bits, and `find_split`'s `if(first_code == last_code)` branch is dead for any range longer
     than one leaf. Both are transcribed as they are (the port stores the word as a plain `uint64_t`, keeping
     `detail::morton_index(morton, idx)` as the constructor) and pinned by the leaf-permutation check: the
     leaves are exactly the **stable** Morton order of the inputs (the port's `sort_pairs_by_key` is stable by
     construction, matching CUB's `DeviceRadixSort::SortPairs`).
  8. **`__threadfence()` has no LC spelling, but `volatile_read`/`volatile_write` *are* it.** LC's CUDA
     codegen is `lc_buffer_volatile_read(buffer, i) = { __threadfence(); return buffer.ptr[i]; }` and
     `lc_buffer_volatile_write(buffer, i, v) = { buffer.ptr[i] = v; __threadfence(); }`
     (`cuda_builtin/cuda_device_resource.h`). The port's rule: wherever the reference has an explicit
     `__threadfence()` (`linear_bvh`'s acquire fence, `stackless_bvh`'s two fences in the internal-node
     protocol) or wherever a cross-thread protocol needs the store to be visible before the next atomic
     claim, use `volatile_write` for the store and `volatile_read` for the load that must observe it. The
     values are unchanged (a fence changes only visibility), the ordering is at least as strong as the
     reference's, and the reader's acquire is covered by the volatile *spin*
     (`$while(box_is_empty(load_aabb(lbox))) { lbox = aabbs->volatile_read(lidx); };`, the reference's
     `atomic_fetch`).
  9. **A zero-length buffer view is not a kernel argument — pad, and replace the size tests that used to
     mean "empty".** Both this rule and §7.30 rule 16 are about the same LC behaviour, but here it has a
     second consequence: the reference uses *buffer sizes* as predicates
     (`if(_flag.total_size() == 0) return;`, `if(flags.size() == 0) return;` for "there is only one
     primitive"). The port pads the internal-node arrays and `flags` to one element so they stay valid kernel
     arguments, and the predicates become explicit `size > 1` / `size <= 1` tests on the element counts. The
     same padding applies to `AtomicCountingLBVH::QueryBuffer::m_pairs`, which starts empty in the reference
     (ticket 0 overflows it — that is the whole point of its retry) but must be a valid handle for LC: the
     port grows it to one element before the first pass and the kernel's own
     `last < pairs_capacity` guard drops the surplus tickets.
 10. **The AABB *union* reduction is not a `linalg` primitive — and it cannot be done with atomics either.**
     The reference calls CUB `DeviceReduce(stream).Reduce(aabbs, out, n, merge, default_aabb)` for the tree's
     normalization box and `atomicMinf`/`atomicMaxf` (on the box's components) for the stackless scene box.
     `linalg::device_reduce` supports only sum/min/max over *scalars* (no custom merge functor), and an
     `LAABB` element is a registered struct, so its components are not addressable as `AtomicRef<float>` (the
     struct `AtomicRef` reaches members through `LUISA_STRUCT`'s per-member `AtomicRef<member_type>`, and
     `LVec<float,3>`'s member would need an `AtomicRef<std::array<float,3>>` that does not exist). The port
     therefore implements the union as `detail::reduce_aabb_union` (details/aabb_reduce.h + aabb_reduce.cpp):
     the same two-pass shape as `device_reduce` (a per-block tree reduction, then one block that folds the
     partials with a strided loop), atomic-free and deterministic. Component-wise min/max over floats is
     associative, commutative and exact, and the **empty box is its identity** (`min = +FLT_MAX`,
     `max = -FLT_MAX`, Eigen's `setEmpty()`), so the value equals the reference's whatever the association
     order — pinned by `top_aabb() == host_union(inputs)` on six clouds and by the brute-force traversal
     equality (which depends on the normalization).
 11. **The reference's shuffle tree survives with a *clamped source lane*.** The scene-box kernel's
     `__shfl_down_sync(0xffffffff, v, d)` leaves the value undefined for lanes with `lane + d >= width`
     (CUDA's `shfl_down` does not wrap); the port reads
     `lcrt::warp_read_lane(v, lcrt::min(lane + d, width - 1))`. Lane 0's final value is still the fold of the
     whole width-lane group (each step folds `j .. j+d-1` inside the group and the clamped reads only ever
     bring in lanes of the *same* group), which is what the reduction uses; the extra lanes hold the identity
     box (`fminf(+FLT_MAX, x) == x`).
 12. **A synchronous device read must commit the frame list first — and that is what makes the retry's
     resize safe.** The reference reads its counters implicitly (`int h_cp_num = m_cpNum;`) and every launch
     it ordered through the CUDA default stream is then complete. An `LStream` batches into the frame
     `CommandList`, so `copy_to` on the main stream would observe the value from *before* the recorded
     dispatches. The port adds one layer helper, `luisa_device::read_host_sync(stream, view)` (buffer.h:
     `stream.flush(); copy_to; synchronize()`), used by both `QueryBuffer` retry loops. It also removes the
     dangling-handle trap: by the time the pair buffer is grown (which frees the old LC buffer while the
     recorded dispatch still holds its raw handle), the pass that referenced it has completed.
 13. **A record-time predicate is baked into the cached shader — say so, and keep the predicates
     stateless.** The reference passes its `Pred` as a CUDA kernel argument (a device functor with embedded
     state). An LC kernel signature cannot carry a host callable, so `detect(pred, ...)`/`query(pred, ...)`
     are templated on the predicate's *type*, the shader is cached per type, and the predicate instance is
     captured by the factory (`stackless_cd_shared_self_kernel(pred)`). A predicate must therefore be
     stateless or have identical captures on every call; the port documents this at each kernel and in the
     class notes, and `DefaultQueryCallback::operator()` is made generic over its arguments so that the
     `$if(pred(i, j))` shape works with DSL components (a host `true` is a valid `$if` condition).
 14. **The traversal stack is an LC `Local<uint32_t>`, the push is bounds-checked, and `stack_overflow()`
     is not portable.** The reference's `uint32_t stack[64]` + `uint32_t* stack_ptr/stack_end` becomes
     `Local<uint32_t>` + an integer stack pointer (a do-while becomes `$while(sp > 0u)` with the pop at the
     top; the root is pushed before the loop). The reference pushes *unconditionally* and only then tests
     `stack_ptr >= stack_end`, which writes one element past the array when both children are pushed into a
     full stack (that is UB in CUDA too); the port guards the push with the same test, so an overflowing
     query truncates instead of corrupting — the reference's own comment already promises only "may be
     smaller than the ground truth". `LinearBVHViewer::stack_overflow()` is a `mutable int` member of a
     viewer *copied into* the kernel, so a kernel's write is invisible on the host: it is not ported, and the
     [bvh] test pins the consequence instead (a 512-box line + a 5-entry stack finds 0 of 512; the same
     5-entry stack on a 37-box cloud finds all 37).
 15. **The stackless structure's contract, and the one ordering that is *not* deterministic.** `Node` is
     `{int32 lc; int32 escape; LAABB bound;}` (32 B; `lc == -1` marks a leaf, `escape == -1` ends the walk);
     `nodes[0 .. n-1)` are the internal nodes in the reference's reordered (tkMap) order and
     `nodes[n-1 .. 2n-1)` the *sorted* leaves; `MAX_RES_PER_BLOCK = 1024` batches a block's results in shared
     memory, and a block that fills the batch flushes it with an `atomicAdd` ticket on the pair counter and
     *re-traverses* from the same node (so duplicates can occur, which is why the reference's own test
     compares pair *sets*). The port keeps the two `__syncthreads()` positions of each kernel exactly (the
     "Other" flavour has one more than the "Self" flavour) and uses `$shared<LVector2i>` +
     `Shared<int32>::atomic(i)` for the cache and the ticket. The reference's `unsorted_ids`-style
     `DeviceVector` fields, the `depths`/`count` aliasing (`count` is both the init kernel's `depths` and the
     `ExclusiveSum` input) and the `primMap`-becomes-the-inverse-mapping reuse are all transcribed.
 16. **The suite is order-fragile, and a kernel that writes out of range corrupts whatever was allocated
     after it — including another test's buffers.** A test-local kernel in this batch wrote
     `counts[n + i]` into a buffer sized `n + 1` (the block needs `2n + 1`: `counts[0]` is the ticket total
     and `counts[n + i]` the per-query count). Isolated `[bvh]` runs passed; the whole suite failed with 10
     *missing* candidate pairs on one cloud (`bvh_cross_structure`'s/`bvh_traversal`'s viewer oracle found
     2637 of 2647) because the stray stores landed in the next allocation's memory. The fix is the test's
     buffer size; the lesson is to run the *whole suite in one process* before believing an isolated filter,
     and to size every scratch buffer from the kernel's own index expressions.
 17. **Two more harness facts confirmed here.** (a) LC's dispatch aborts on a block size that is not a
     multiple of 32 (both structures use exactly 256, the reference's `K_THREADS`, and the reduce kernels
     use 256 as well); (b) one run died during setup with LC's own
     `Assertion 'begin != end' failed: Failed to parse NVRTC version` (rc 3221226505) immediately after the
     device was created — a transient, environment-level NVRTC probe failure, not a port bug: the very same
     binary ran green on the next invocation (do not chase it, re-run).
 18. **The export list: 394 → 403 symbols, bootstrapped then regenerated.** `python
     scripts/luisa_module_def.py` derives the list from the *import tables of the built PE files*, so a
     *failed* test link (which deletes the exe) leaves it with zero consumers — it then writes a 5-line file
     and every consumer's link breaks. Recovery: restore the committed `.def` (`git show HEAD:…` — a read),
     append the decorated names the linker reports (the 8 new member functions), rebuild to get a linkable
     exe, and only then re-run the generator, which rewrites the file from the real imports (403 = 394 + the
     8 + `LStream::flush`, which the new inline `read_host_sync` imports). The final file must be 100 %
     generator output (`--check` rc 0) — never ship the hand-seeded version.
 19. **The regression shape and the reproduction recipe.** `[bvh]` → **219 assertions in 8 test cases**
     (per case: morton_codes 11, tree 112, traversal 60, cross_structure 8, determinism 13,
     device_vs_host 9, stack_overflow 5, summary 1); `~[bvh]` → **763191/215 == the T6.4c baseline exactly**
     (the batch adds no system to any scene, so it is purely additive); the whole suite → **763410/223**
     (measured twice, identical); `ctest` 1/1; the standalone suite 69 cases / 0 checks failed; the ten
     sibling filters unchanged; `python scripts/luisa_module_def.py --check` rc 0 (403 symbols);
     `python scripts/luisa_check_module_abi.py` rc 0. Measured device-vs-host: 257 Morton codes and 67
     common-prefix counts bit-identical to the independent host oracles (0 mismatches); the tree's internal
     boxes *exactly* equal the union of their children on 6 clouds (1/2/3/1024/identical-37/line-129 leaves;
     node counts 2n-1, depths 0/1/2/14/6/8, exactly one root each); the candidate sets equal brute force on
     7 clouds (n = 1, 2, 37, 64 identical, 211 duplicates, 4096, **100k**) — the 100k tree answers 1563
     queries with exactly the 26075 brute-force pairs — and the two independent structures agree on 795729
     self-pairs at 100k primitives. **Reproduction:**
     `python scripts/luisa_build.py build backend_luisa` (rc 0, 0 warnings) then
     `python scripts/luisa_build.py run "[bvh]"`; every case prints the numbers quoted above.
   20. **What the next batch (T7.1b) still owed — and now delivers (§7.32).** `info_stackless_bvh.{h,cu}` +
       `details/info_stackless_bvh.inl` (1154 lines) and `info_stackless_bvh_v0.{h,cu}` +
       `details/info_stackless_bvh_v0.inl` (1134 lines) were the two "info" variants the deferred sanity checkers
       (`sanity_check/simplicial_surface_{distance,intersection}_check.cpp`) needed; T7.1b ported both, reusing
       the `culbvh` helpers, `LAABB` and pair type this batch established, ported and wired the two checkers and
       re-enabled them — see §7.32 for the info contract, the v1/v0 differences as measured, the predicate
       limitation and the checkers' host-filter shape. `LinearBVH::update`
       is ported and pinned (a rebuild-equivalence check), but the reference's `update()` warning ("if the
       AABBs are significantly changed, you may get a Tree with bad query performance") is only a comment; the
       port keeps its `UIPC_ASSERT` on the size contract (which aborts the process on misuse — not covered by a
       test for that reason).

### 7.32 Patterns established in T7.1b (the "info" BVH variants + the two deferred sanity checkers)

**Scope.** The second M7 batch: `collision_detection/info_stackless_bvh.{h,cpp}` + `details/info_stackless_bvh.inl`
(the port of the reference's 238 + 1154 lines), `collision_detection/info_stackless_bvh_v0.{h,cpp}` +
`details/info_stackless_bvh_v0.inl` (178 + 1134), **both deferred sanity checkers**
(`sanity_check/simplicial_surface_distance_check.cpp` → uid 3, `sanity_check/simplicial_surface_intersection_check.cpp`
→ uid 1, whose deferral note in `sanity_check/backend_sanity_checker.h` is gone), the refreshed module export list
(403 → **409** symbols) and `apps/tests/backends/luisa/luisa_info_bvh_test.cpp` (tag `[bvh_info]`, **9 cases / 226
assertions**). Baseline shape: `[bvh]` **219/8** unchanged, `~[bvh_info]` = **763410/223** = the T7.1a baseline
exactly, `[bvh_info]` 226/9, the whole suite 763410+226 / 223+9, the 13 sibling filters unchanged, ctest 1/1,
standalone 69/0, `scripts/luisa_module_def.py --check` rc 0, `scripts/luisa_check_module_abi.py` rc 0.

  1. **The info variants are T7.1a's `StacklessBVH` + metadata + a two-predicate contract, and the port reuses the
     first half verbatim.** The reference's `info_stackless_bvh.inl` is a near-duplicate of `stackless_bvh.inl`
     that vendors its *own* copy of the culbvh helpers under `uipc::info_stackless_detail`; the port keeps one
     shared copy and includes `collision_detection/stackless_bvh.h` (its `culbvh` namespace: `K_THREADS`,
     `MAX_RES_PER_BLOCK`, `morton3D`, `make_pair_var`, `make_ordered_pair`, `safe_copy_to`) plus T7.1a's
     `LAABB`/`LBox`/`aabb_reduce`. The build pipeline is `stackless_bvh.cpp` with exactly three additions —
     `buildPrimitivesFromBox` scatters `ext_bid`/`ext_cid`, `buildIntNodes` propagates the *common* bid/cid (or
     -1), `reorderNode` writes both into the node array — and `Node` grows from 32 B (`{lc, escape, bound}`) to
     **40 B** (`{lc, escape, bound, bid, cid}`). The reference's `propagateInformativeMetadata(int)` is declared
     and defined **empty** in both files; it is transcribed (a reader can diff the two file sets line by line).
  2. **A record-time predicate cannot read a device array — the batch's one structural deviation.** The
     reference's `NodePred`/`LeafPred` are CUDA device functors that *capture* `cuda_tool::CBufferView`s (and a
     `CDense2D` mask table). An LC kernel signature cannot carry a callable, and a captured `Buffer`/`BufferView`
     becomes a **bound** argument of the compiled shader (rule 2 in §7.9; `FunctionBuilder::buffer_binding`),
     so a cached shader would keep reading the *first* call's allocation. The port therefore keeps T7.1a's
     shape — the predicate is a host callable invoked at *record time* with LC expressions — and adds the one
     thing that shape lacked: the info structs are **templates over the scalar type**
     (`NodePredInfoT<I>` / `LeafPredInfoT<I>`, `I = IndexT` on the host, `lcrt::Var<IndexT>` in a kernel),
     because a device value is a DSL object and cannot live in an `IndexT` field. A ported predicate reads the
     struct's *fields* (scalars) only; `nlohmann`-style "read `bids[query_id]`" predicates (the reference's
     `BidOnlyNodeCull`) are not expressible. The two checkers therefore drive the four phases with keep-all
     predicates and evaluate the reference's predicate *bodies* as **host filters over the reported candidate
     pairs** — see rules 15/16.
  3. **The cache key is the predicate's *type* — a capture that varies between calls silently uses the first
     call's shader.** `cached_kernel` instantiates a `static Shader&` per (lambda type) == per predicate *type*,
     so two predicates of the same type with different captured values share the first one's shader. This bit the
     [bvh_info] test itself: a `NodeCidCull{IndexT cid}` member used in a `for(cid = 0..2)` loop culled with
     `cid == 0` in all three iterations (13 assertions failed until the constant moved into the type,
     `NodeCidCullT<Cid>`). Rule: **carry a predicate's constants in its type** (or make it truly stateless) —
     T7.1a's rule 13, now with a measured failure mode.
  4. **A predicate's return value is the *keep* condition — inverting the `-1` sentinel test is a silent
     everything-culled.** The port's `$if(node_cull(info))` descends/reports on `true` (the reference's
     `if(!node_cull(...)) { st = escape; continue; }`). The [bvh_info] test's first `NodeInvalidBidCull` wrote
     `info.node_bid == -1` (the *cull* condition) instead of `!= -1`: on a cloud where the exported array had
     **zero** nodes with `bid == -1`, every candidate vanished (`pairs=0` against `brute=430`) — the test's host
     walk agreed with the device, so only the *oracle* caught it. Polarity now lives in the names
     (`NodeValidBidOnlyCull`).
  5. **The node-cull oracle is *asymmetric*: only the reported pair's larger-sorted primitive has its node
     culled.** In the reference's self traversal the *query* primitive's own box is loaded before the walk;
     the reporter is whichever of the two has the smaller **sorted position**, and the walk descends to the
     other one — whose own node (and ancestors) are the only ones the predicate tests. So a cull on `cid == C`
     does **not** keep "the pairs whose two primitives both have `cid != C`" (the naive oracle): it keeps a pair
     iff the primitive with the larger sorted position has `cid != C`. Measured on a 512-cloud (1168 overlapping
     pairs, cids `i % 3`): the surviving counts are **777 / 771 / 788** for `C = 0 / 1 / 2`, not
     `1168·(2/3)² ≈ 519`; `brute_force_self_node_cull` in the test implements the exact rule and the device
     matches it. The `query` flavour is the symmetric case (the visited node is always the object's), which is
     why the checker's `CodimP vs AllP` oracle is simply `cid[object] != C`.
  6. **v1 vs v0: what actually differs (all pinned).** | | v1 | v0 | |---|---|---| | `NodePredInfo` | **5 fields**,
     `query_bid`/`query_cid` pre-loaded into shared memory | **3 fields** (`query_id`, `node_bid`, `node_cid`) |
     | `LeafPredInfo` | **6 fields**, the pair's bid/cid resolved | **absent** — `LeafPred(i, j)` | | traversal
     SMem | `s_qbid`/`s_qcid` pre-load once per thread | none | | self pair | `LeafPredInfo` with the query side
     first (ordered by `q_first`) | `LeafPred(ordered_pair)` + the reference's `j <= i` adapter guard | |
     `detect`/`query` | split into public `launch_*` + `prepare_query_result` | one inline body each | |
     `Impl::build` | no `ext_mark` | `ext_mark` allocated + initialised to 7 | | init kernel | `ext_lca(idx) =
     -1` for `idx <= num_objs`; **no** `int_bid`/`int_cid` init | `ext_lca = 0`/`ext_par = 0`/`ext_mark = 7` per
     leaf, `int_bid`/`int_cid = **-1**` per internal node | | `buildIntNodes` | no leaf prologue | `lvs_lca(idx)
     = -1; depths(idx) = 0;` prologue | | `stacklessOther` | takes the query BID/CID spans | takes neither |.
     **Measured**: the two builds produce **element-identical node arrays** (799/799 on a 400-primitive
     duplicate-heavy cloud, `bid`/`cid`/bounds/links all equal) and identical candidate sets (2048: 19161 =
     `StacklessBVH` = `AtomicCountingLBVH` = brute force; 100k: 794217 for all three) — so v1's missing
     `int_bid`/`int_cid` initialisation is *unobservable*: every read of an internal child's metadata happens
     after that child completed its own merge (the flag protocol + the fence order it).
  7. **The reference's `build(aabbs)` overload is a trap and it is transcribed as one.** With no BID/CID spans
     `m_BIDs`/`m_CIDs` stay empty, but `launch_detect`/`launch_query` then assert `m_aabbs.size() ==
     m_BIDs.size()` — i.e. the two-argument build is only usable with an empty input set. The port keeps the
     asserts verbatim (transcribe, don't fix) and avoids the trap internally by passing a **one-element
     `info_dummy`** span for the missing ids (LC rejects a zero-length buffer view as a kernel argument — T7.1a
     rule 9), with `has_info == 0` telling the kernels never to read it.
  8. **The v0 retry must re-clear the pair counter — the port's one *real* bug in this batch.** The reference's
     v0 `detect`/`query` clear `m_cpNum` **inside their `do_query` lambda**, so both the first pass and the
     retry start from 0; the port's first version cleared it only once and the retry appended to the first
     pass's total (measured: the 100k self-pair set came out at **51201** — exactly the initial `50 * 1024`
     capacity — instead of 794217, while v1, which clears inside `launch_detect`, was right). The v1 split
     (`launch_detect` = "clear + record", `prepare_query_result` = "read + grow") is the shape that makes this
     impossible to get wrong.
  9. **`prepare_query_result` and the capacity contract.** The pair count is read with `read_host_sync`
     (T7.1a's rule 12), and on `count > m_pairs.size()` the buffer is grown to `count * reserve_ratio` and the
     launch repeated — `retry == (count > 1)` for a queue reserved to 1 element, and `capacity = 5524` for
     `count = 4604` at the default `reserve_ratio = 1.2` (measured), `2.0 × count` with a caller-supplied
     `Config`. `QueryBuffer`'s initial capacity is the reference's `50 * 1024`.
 10. **The sanity checkers are wired exactly where the reference wires them, and the frontend replaces the CPU
     checker of the same uid.** `REGISTER_BACKEND_SANITY_CHECKER(SimplicialSurfaceDistanceCheck)` (uid **3**) and
     `...(SimplicialSurfaceIntersectionCheck)` (uid **1**) — both derive from the ported
     `BackendSanityChecker`, both gated by `contact/enable` in `build()` (throwing
     `BackendSanityCheckerException` skips the checker), both reading the CPU sanity module's
     `sanity_check/*` attributes (`dim`, `d_hat`, `instance_id`, `geometry_id`, `object_id`,
     `contact_element_id`, `subscene_contact_element_id`, `self_collision`) and the contact/subscene tabulars.
     Measured on a two-cube scene: with a 0.5-offset interpenetration both cubes are in ONE object → the
     message is `Geometry(0) in Object[t71b_overlap(0)] intersects with Geometry(1) in
     Object[t71b_overlap(0)]` + `Intersected mesh has 14 vertices, 12 edges, and 12 triangles.` +
     `Create mesh [intersected_mesh<SimplicialComplex>] for post-processing.`; `msg->name()` is
     `class uipc::backend::luisa::SimplicialSurfaceIntersectionCheck(1)` (the backend's, not the CPU's).
 11. **`d_hat` is the CPU context's, and it is 0.01 by default — the AABB expansions are *not* just the
     thickness.** `sanity_check/d_hat` comes from the scene/contact config (measured 0.01 with a default
     scene), so the distance checker's point boxes are expanded by `thickness + d_hat` and its triangle boxes
     by `3·thickness + (Σd_hat)/3`; the *violation* test is the reference's `D ≤ (t_i + t_j)²` on squared
     distances. Both are transcribed unchanged (the expansions only decide which pairs are *candidates*).
 12. **The reference's `NodePred` cull is *sound*: removing it cannot change the surviving set.** The node's
     `cid` is the common cid of its subtree, so `!mask(query_cid, node_cid)` fires only when every leaf below
     shares that cid — and the `LeafPred`'s per-vertex mask checks then reject each of those pairs anyway. That
     is why the port can drive the checkers with keep-all predicates and filter on the host without any
     behavioural loss (only the candidate count changes). The `self_collision` rule is *not* implied by the
     node cull and is applied at the leaf/host level.
 13. **"Distance" is vertex-triangle + edge-edge proximity, not interpenetration — pinned by a scene where it
     legitimately stays silent.** Two axis-aligned unit cubes offset by `(0.5, 0.5, 0.5)` interpenetrate, and
     the intersection checker reports 14/12/12 — but the *distance* checker reports **nothing**: the crossing
     happens as edges passing *through triangle interiors*, and with the offset's two fixed coordinates
     (`0/1` vs `0.5/1.5`) no edge-edge pair ever meets and no vertex-triangle pair comes closer than 0.25
     (≫ the 0.004 thickness sum). The [bvh_info] test pins both verdicts (`errors.find(1) != end`,
     `errors.find(3) == end`).
 14. **The reported "close mesh" flags *whole* triangles and *both* endpoints of an edge.** Two cubes separated
     by `1.5·thickness` (a genuine near-miss: uid 3 fires with `distance=0.0029999999999998916`,
     `thickness=0.004`) report `Close mesh has 16 vertices, 26 edges, and 20 triangles` — i.e. **all** 16 scene
     vertices, because every flagged triangle spans from the 0.003-away rim to the far side and every flagged
     edge is collinear with a near edge (the vertical edges of the two cubes are `0.003` apart *along their whole
     length*). The test asserts `nv == 16` with that explanation; the "only the facing faces" intuition is wrong.
 15. **The checkers' host filter is a transcription, not an approximation.** Each of the four distance phases
     (and the intersection check) transcribes its reference device predicate *body* to a host function over the
     same data: `distance::point_*_distance_flag` + `*_distance2` (the ported T2.3 layer, host `T = double`),
     `tri_edge_intersect` (the reference's `UIPC_GENERIC` Eigen function, host instantiation),
     `simplex_contact_mask_utils.h`'s four `allow_*_contact` helpers over the port's **flat** mask layout
     (`row * N + col`, §7.13). `close_geo_ids` (last-write-wins on a sorted-key `map`) and `close_geo_distances`
     (min) are order-insensitive, so re-deriving the violation list from the same surviving pairs reproduces the
     reference's message byte for byte — the test compares the geometry/object ids, both floats, the mesh-size
     lines and the `Create mesh [...]` tail.
 16. **A `quad` predicate is a *typed* host callable: write `operator()` as a template.** Because the info
     structs are templates over the scalar type, a ported predicate is
     `template <typename I> auto operator()(const InfoStacklessBVH::NodePredInfoT<I>&) const` — and in the
     `v0` leaf flavour a two-argument `template <typename A, typename B> auto operator()(const A&, const B&)`
     (the port folds the reference's `SelfPairPredAdapter` `j <= i` guard into the kernel, so a predicate that
     reads its indices still sees the ordered pair). Also: a *local* class cannot have template members (MSVC
     C2892), so the test's predicates live at namespace scope; and a predicate that needs a device *comparison*
     must return the `Expr<bool>` directly, never assign it to a host `bool` (rule 13 of §7.31).
 17. **The export-list step is part of this batch, and the `.def` must be bootstrapped before it can be
     regenerated.** Adding the test made the **link** fail with 6 unresolved externals
     (the three non-inline members of each structure: `QueryBuffer::build`, the public `InfoStacklessBVH::build` /
     `InfoStacklessBVHV0::build` and the two `Impl::build`s) — a *failed link deletes the test exe*, and
     `scripts/luisa_module_def.py` derives the export list from the built PE files, so regenerating first would
     have written a 5-line file. Recipe: append the LNK2019/LNK2001 mangled names to the committed `.def`,
     rebuild until the link succeeds, *then* run `python scripts/luisa_module_def.py` (403 → **409** symbols,
     `--check` rc 0) and finish with `scripts/luisa_check_module_abi.py` rc 0. (T7.1a's rule 18; no vtable-slot
     rebootstrap was needed here — the two new classes are leaf checkers with new mangled names, not new slots
     in an existing vtable.)
 18. **`--syntax` does not instantiate the test-local kernels; the [bvh_info] filter is the real syntax
     check.** `python scripts/luisa_check_compile.py <file> --syntax` compiles one TU: it catches the traversal
     templates only where a TU instantiates them (the test file does, via its predicates), and it takes ~6-10 s
     per file. It is the right inner loop — the whole-batch build is ~20 s warm, ~25 s with a JIT-cold test run.
 19. **Reproduction.** `python scripts/luisa_build.py build backend_luisa` (rc 0, 0 warnings) →
     `python scripts/luisa_build.py run "[bvh_info]"` (9 cases / 226 assertions, rc 0; every quoted number is
     printed by the test). The battery is
     `[bvh]` 219/8 → `[bvh_info]` 226/9 → `~[bvh_info]` 763410/223 → the whole suite → the 13 sibling filters →
     `[luisa][t43]` → ctest 1/1 → `python scripts/luisa_standalone.py cuda` 69/0 →
     `python scripts/luisa_module_def.py --check` rc 0 → `python scripts/luisa_check_module_abi.py` rc 0.
 20. **Two perturbations on two axes (one host, one device), both rc 42 and both byte-identically reverted.**
     P1 (**host/plumbing**): `info_stackless_bvh.h`'s `QueryBuffer` initial capacity `50 * 1024` → `64` — the two
     capacity-contract assertions in `info_determinism` fail (lines 1555/1564) while the pair *sets* stay correct
     (the retry grows the buffer), i.e. the perturbation is caught by the *contract* pin, not by a crash.
     P2 (**device/kernel**): the v1 self flavour's resolved `LeafPredInfo` `bid_i`/`bid_j` `select` operands
     swapped — the order-sensitive `leaf bid-pair` pins fail (line 703, both the `(3,5)`/`(5,3)` and the
     `(2,6)`/`(6,2)` comparisons, which are non-empty and *different* by construction). Both patches were
     reverted with a byte-identical `sha256`, rebuilt (rc 0) and the `[bvh_info]` filter re-run green — so the
     test really does pin the capacity contract *and* the resolved-pair assignment, and neither pin survives a
     guess.

### 7.33 Patterns established in T7.2a (the trajectory filters, batch 1: the filter bases + the vertex-half-plane filters)

**Scope.** The first M7 filter batch: `collision_detection/trajectory_filter.{h,cpp}` (the
interface), `collision_detection/global_trajectory_filter.{h,cpp}` (the manager),
`collision_detection/simplex_trajectory_filter.{h,cpp}` (the simplex-pair base),
`collision_detection/vertex_half_plane_trajectory_filter.{h,cpp}` (the vertex/half-plane base),
`collision_detection/filters/easy_vertex_half_plane_trajectory_filter.{h,cpp}` and
`collision_detection/filters/al_vertex_half_plane_trajectory_filter.{h,cpp}`, plus the **scoped
dependency port** `contact_system/global_contact_manager.{h,cpp}` (§7.33 rule 4), the wiring
(`sim_engine`, `global_vertex_manager`) and `apps/tests/backends/luisa/luisa_trajectory_filter_test.cpp`
(tag `[filter]`, **7 cases / 423 assertions**).  Baseline shape: `[filter]` **423/7**,
`~[filter]` = **763636/232** = the T7.1b baseline *exactly*, the whole suite **764059/239**
(measured twice: `763636 + 423` / `232 + 7`), every sibling filter unchanged (`[abd_base]` 2143/7,
`[abd_linear]` 495/7, `[abd_dynamics]` 547/7, `[abd_constitution]` 523/7, `[abd_solve]` 245/8,
`[abd_joint_a/b/c]` 223/8 · 402/8 · 383/8, `[bvh]` 219/8, `[bvh_info]` 226/9, `[mas_preconditioner]`
234/7, `[fem_precond]` 104/4, `[fem_linear]` 323/4, `[fem_dynamics]` 135/8, `[luisa][t43]` 238/4,
`[half_plane]` 426/7), ctest 1/1, standalone 69/0, the module export list **410 → 463** symbols
(`--check` rc 0, ABI probe rc 0).

  1. **The one structural decision of the batch: the trajectory filters need no "device data
     reaches the predicate" hook — the host-filter shape is exact.** The reference's *simplex*
     filters pass a device functor into `AtomicCountingLBVH::detect/query` or
     `StacklessBVH::detect/query`, but that functor is applied **inside the traversal callback**,
     i.e. only for objects whose AABB already passed the *fixed* node test:
     `LinearBVHViewer::query(aabb, callback)` hard-wires `Intersect = node.intersects(Q)` (the
     AABB overload's own lambda, `linear_bvh.h`), and `details/atomic_counting_lbvh.inl` calls the
     user predicate only there (`if(id > i && p(i, id))`); `StacklessBVH`'s contract is literally
     `template <std::invocable<IndexT, IndexT> Pred>` (two *object* indices, no node).  So the
     predicate cannot change which nodes/leaves are visited: it is a **selection** of the
     already-found (AABB-overlapping) pairs, and a host re-implementation of the predicate body
     over the reported pairs reproduces the surviving set exactly.  What it *does* change is the
     reported pair *count* — and therefore the pair buffer's growth/retry — which is host-visible
     but not semantic (§7.33 rule 6, and rule 12 pins the pruning exception).  The T7.1b decision
     (host filter over the reported candidates) therefore stands for the whole T7.2 family.
  2. **The two exceptions, stated for T7.2b.** (a) `InfoStacklessBVH(V0)`'s `NodePred` *is* a
     node-level cull (its `NodePredInfo{query_id, bid, cid}` reads per-node device metadata), so it
     prunes; the cull is sound because every cull condition is implied by the leaf predicate
     (the "common bid/cid, or -1" propagation means the cull fires only when *every* leaf below
     shares the value the leaf check rejects) — that is the T7.1b §7.32 rule 12 argument, pinned
     there on the host-walk oracle, and it is what makes the host-filter route sound for those two
     filters as well.  (b) the four BVH filters' `DetectInfo` also reads per-primitive device
     attributes (`v2b`, `codim_vertices`, `contact_element_ids`, the masks) — those arrive as
     *kernel arguments* in the ported shape (rule 10 below), never as predicate captures.
  3. **A record-time predicate that captures a device view is a hazard, not a hook.** Because the
     port bakes the predicate into the cached shader *and* the cache key is the lambda *type*
     (§7.32 rule 3), a captured `LBufferView` becomes a shader-bound buffer of the first call: two
     predicates of the same type with different captures silently share the first allocation.  That
     is *why* the design question has to be answered by "does the reference test device data at the
     node level?" (no, for this family) rather than by "can the port technically read a buffer in a
     predicate?" (it cannot, safely).
  4. **This batch needed a scoped `GlobalContactManager` — the batch's one piece of new
     infrastructure, and it is documented as such.** The six filter files `require<>` the contact
     manager and read exactly five things from it: `d_hat()`, `contact_mask_tabular()`,
     `subscene_mask_tabular()`, `cfl_enabled()` and `Impl::vert_is_active_contact`.  The port had
     no contact layer at all (T8.1 owns `contact_system/`), so T7.2a ports **that surface only** —
     the system creator (`contact/enable`), `do_build`, `Impl::init`'s scene-relative `d_hat`
     override, the two mask tabulars (built from the frontend's `contact_tabular()` /
     `subscene_tabular()` attributes) and the vertex activation buffer — and leaves the
     coefficient/reporter/CFL/feasible-step halves under explicit `SCOPE NOTE (T8.1)` markers (the
     T5.4a "port the half, mark the rest" precedent).  T8.1 completes the file in place; nothing in
     the filters deviates because of it.
  5. **The 2D → flat mask mapping, and the extent that has to travel with it.** The reference's
     `CBuffer2DView<IndexT>` masks are self-describing; the port's device layer has no 2D buffer, so
     the masks become flat `LBufferView<IndexT>` (`index = row * extent + col`, the same layout the
     T7.1b checkers use) **plus an explicit extent**: `GlobalContactManager::contact_mask_size()`
     / `subscene_mask_size()`, re-exported on both filter bases as
     `BaseInfo::contact_mask_size()` / `subscene_mask_size()`, and passed to the kernels as extra
     scalar arguments.  A T7.2b filter must do the same (`allow_XX_contact` reads
     `mask[cids[0] * extent + cids[1]]`).
  6. **The pair buffer's growth/retry contract, with exact numbers.** The kernel takes its ticket
     *before* the `last < max_count` guard, so `h_num_collisions` counts every surviving pair and
     the host always knows the true count; when it exceeds `PHs.size()` the host grows the buffer to
     `static_cast<size_t>(h_num_collisions * reserve_ratio)` and re-runs the **same** kernel from
     ticket 0 (a rewrite, not an append — pinned by comparing the retry set with the keep-all set),
     and `info.PHs(PHs.view(0, h_num_collisions))` then carries exactly the reported count.  With
     the reference's `reserve_ratio = 1.1f` the truncating cast is harmless (`count * 1.1 >= count`)
     but visible: **3 candidates → buffer size 3**; the test drives `reserve_ratio = 2.0` in its
     standalone pass so every quoted size is exact (**2 candidates → 4**), and pins both the
     "capacity already covers the count → no growth" and the "empty start → grow" paths.
  7. **A zero-length buffer view is not an LC kernel argument — the Easy filter's pair buffer has to
     be padded (T7.1a rule 9 again).** `Impl::filter_active` passes `PHs.view()` even when it is
     empty (the reference's `DeviceBuffer` starts empty and its CUDA pointer is never dereferenced
     because `max_count == 0` forbids every write); in the port that is an invalid shader argument,
     so the filter pads to one element before the query — the T7.1a `AtomicCountingLBVH::detect`
     fix, with the same "nothing observable changes" argument (the kernel's own guard drops the
     surplus ticket and the host retries with the grown buffer).  The AL filter's grid is empty only
     when the scene has no half-plane instance, so its dispatch (and its reduce) is guarded by
     `n > 0 && n_planes > 0` — equivalent to the reference's no-op kernel *and* to CUB's
     "zero-length reduce does not launch" (the ported `device_reduce_min` *writes* the identity for
     `n == 0`, so the guard is the faithful spelling of the reference's no-launch).
  8. **Ordering: the fills and reduces go on the frame stream, and the `toi` slot's pre-fill is the
     n == 0 fallback.** `info.toi().fill(1.1f, &frame_stream())` runs before the dispatch in the
     reference, so an empty phase leaves the slot at `1.1f`; the port keeps the fill first and the
     reduce (`lla::device_reduce_min(frame_stream(), tois, info.toi(), n)`) only when the kernel
     ran.  A fill on the *main* stream would invert the order (the T5.4a finding, §7.22 rule 7).
  9. **Two different `1.1` spellings in one buffer, both pinned.** The *Easy* kernel's `min_toi`
     starts at `Float min_toi = 1.1f` (the float literal, `1.10000002384185791015625`) and the min
     wins over any larger toi: a vertex whose raw toi is 19.8 reports `1.1f`, not 19.8 — a
     *sentinel*, and the test pins it with a vertex engineered for it (y=0.75 arriving slowly:
     `want == 1.1f` while the raw formula is `> 1.1`).  The *AL* kernel writes the **double** `1.1`
     for a pair that moves away, while the mask-skipped pairs keep the caller's `tois.fill(1.1f)`
     float value: the same grid therefore holds both, and the test asserts `tois[2] == 1.1` **and**
     `tois[2] != 1.1f` (they differ by 2.4e-8).  Transcribing `1.1` as `1.1f` anywhere would have
     made both checks silently vacuous.
 10. **The three measurable Easy-vs-AL differences (all pinned).** (a) the AL `filter_active` is the
     reference's explicit no-op, so `candidate_PHs()` is **empty** after the active stage and only
     becomes the *full* `surf_vertices × plane_instances` grid after `filter_toi` (the pair is
     written *before* the mask checks, so mask-disabled pairs appear in the grid too);
     (b) `eta = 0.001` (AL) vs `0.1` (Easy) — pinned by the exact toi values; (c) the AL toi is a
     per-*pair* grid min-reduced as a whole, the Easy toi is a per-*surface-vertex* minimum
     (mask-skipped or away-moving planes leave the `1.1f` initial value).  In an `ipc` scene only
     the Easy filter survives; in an `al-ipc` scene only the AL one (both derive from
     `VertexHalfPlaneTrajectoryFilter` and both are created in every scene — the base's `require<>`
     chain plus the derived `require<{I,AL}IPCPipelineFlag>()` is the reference's scene gating, and
     the engine's `SimSystemException` handler invalidates the loser before it registers).
 11. **`candidate_PHs()` and `PHs()` disagree on purpose.** `PHs()` is the *view the base stored*
     (`view(0, h_num_collisions)`, the reported count), while the derived filter's own
     `candidate_PHs()` returns its whole buffer, whose size is the retry-grown capacity (1 after the
     padding, `count * ratio` after a retry).  The reference has the same asymmetry (its
     `DeviceBuffer` view carries the capacity), so it is transcribed and pinned: an empty-candidate
     scene gives `PHs().empty()` **and** `candidate_PHs().size() == 1`.
 12. **The predicate-shape measurement, on the port's own structures.** `[filter]` drives
     `AtomicCountingLBVH::detect` with a keep-all predicate and with a selective one
     (`i % 2 == 0`): **keep-all 1139 == brute force 1139**, selective **602 == the host-filtered
     keep-all set** with `602 < 1139` — the selection/not-pruning argument of rule 1, measured
     rather than argued.  The *pruning* half of the family is pinned by `[bvh_info]` (node-cull vs
     host-walk vs the brute-force node-cull oracle, §7.32 rules 5/12); the two together are the
     evidence behind "a host filter reproduces the reference's surviving set, and only the candidate
     count (hence the pair-buffer retry) differs".
 13. **Wiring: three call sites, one of which is a no-op until T8.4.** `SimSystemCreator<GlobalContactManager>`
     and `SimSystemCreator<GlobalTrajectoryFilter>` both gate on `contact/enable`
     (`default_config()` has `contact/enable = 1`, so both exist in every scene of the suite);
     `SimEngine::build_engine_systems()` resolves them in the reference's order (dytopo → contact
     manager → trajectory filter → animator); `SimEngine::init_scene()`'s 3.1 forwards call
     `m_global_contact_manager->init()` where the reference does (after the dytopo manager, before
     the animator); `GlobalVertexManager::do_build()` resolves the `GlobalTrajectoryFilter` slot and
     its `require_discard_friction()` half is live again (the `if(auto& ...)` guard keeps it a no-op
     in a contact-less scene).  The Newton-loop calls (`detect`/`filter_active`/`filter_toi`) stay
     T8.4's: adding them to the ported frame shell would run the filters in *every* test's frames
     (and the reference's own `RUNTIME_CHECK(toi > 0)` is only sound after the CCD clamp).
 14. **The `run_stage()` test seam — the batch's only addition to the reference's API surface.**
     The six stage entry points are private with an explicit friend list (SimEngine /
     ContactExporterManager / GlobalContactManager) and none of those three callers exists before
     T8.4/T8.1, so without a way in the manager's own loops (`filter_active`'s CFL reset,
     `filter_toi`'s min-reduce and `toi > 0` assert, `record_friction_candidates`'s issue-#303
     discard) could not be driven at all.  `GlobalTrajectoryFilter::run_stage(Stage, alpha)` is a
     single public forwarder whose branches are the reference's own one-liners; it adds no behaviour
     and is removed when T8.4 wires the loop.  Everything else in the batch is driven through
     *public* API only (the two `Impl` structs are public nested classes, so the test constructs
     `VertexHalfPlaneTrajectoryFilter::Impl` / `Easy…::Impl` / `SimplexTrajectoryFilter::Impl` and
     calls their public `filter_active`/`filter_toi`/`record_friction_candidates` directly).
 15. **The test's scene is the ordinary frontend one, and its oracles are host re-implementations
     over the read-back device arrays.** A `HalfPlane` implicit geometry + a `SimplicialComplex`
     carrying the shared `test_*` vertex attributes (the T4.x convention: `test_dimension`,
     `test_contact_element_id`, `test_subscene_element_id`, `test_d_hat`, `test_thickness`,
     `test_displacement`) is enough — the sibling `luisa_global_geometry_test.cpp` reporters report
     every complex and every surface vertex, so the global arrays are populated without a
     test-local reporter.  Every oracle (`easy_candidate_oracle`, `easy_toi_oracle`,
     `al_toi_oracle`) is written out in the test from the same *device* inputs (`gather()` reads
     `surf_vertices`, positions, thicknesses, ids, `d_hats`, displacements, the plane arrays and the
     two flat masks back through the ported managers) — independent of the filter's outputs, and
     bit-exact because both sides evaluate the same double expressions in the same order (the
     test's `dot3` spells out `s = a0*b0; s += a1*b1; s += a2*b2` so no Eigen expression template
     can re-associate it).  The scene's numbers are dyadic where the test asserts exactness
     (`thickness = 1/16`, `d_hat = 1/8`, `alpha = 0.5`, displacements multiples of 1/64).
 16. **The global displacement array is the test's to fill (the Phase-8 fan-out is private).**
     `GlobalVertexManager::collect_vertex_displacements()` is private (friends: SimEngine,
     MaxTranslationChecker, GlobalTrajectoryFilter) and the ported engine shell does not call it, so
     `DetectInfo::displacements()` is all zeros at `world.init()`.  The test uploads the array
     through the public `displacements()` view (its `copy_from` is const'able) — the same thing the
     reporter fan-out does, and the only way to exercise the TOI stage before T8.4.
 17. **The CFL activation buffer is written by the label stage and cleared only when CFL is on.**
     `cfl/enable` defaults to **0**, so `filter_active` does *not* clear
     `vert_is_active_contact`; `label_active_vertices` then ORs in the vertices of the reported
     `PHs` (one thread per pair, one atomic exchange per member).  With `cfl/enable = 1` the clear
     happens first.  The buffer is reachable from a test through the public
     `LabelActiveVerticesInfo{&gtf_impl}` (its `Impl` pointer is a public member of a public nested
     class) — no friendship needed.  `[filter]` pins the accumulate semantics and the exact labelled
     set.
 18. **The export-list step: two bootstrap rounds this time (410 → 463).** Adding the test made the
     link fail with **38** unresolved externals (the bases' `BaseInfo` accessors, the two
     `GlobalContactManager` accessor groups, `run_stage`/`require_discard_friction`,
     `LabelActiveVerticesInfo::vert_is_active`, `SimplexTrajectoryFilter::Impl::record_friction_candidates`
     and `ISimSystem::is_building`), then **16** more after the first append (the `toi_PHs()` /
     `candidate_PHs()` virtuals and the `find<>` instantiation) — the linker stops after a batch, so
     the loop "append the reported mangled names, rebuild" has to run twice; only then did
     `python scripts/luisa_module_def.py` rewrite the file from the real import tables (463 symbols,
     `--check` rc 0) and `scripts/luisa_check_module_abi.py` stay rc 0.  A failed link deletes the exe,
     so the generator must not be run before the link succeeds (the T7.1b trap).
 19. **Two perturbations, two axes, both rc 42, both byte-identically reverted.**
     P1 (**host/plumbing**): `EasyVertexHalfPlaneTrajectoryFilter::Impl::reserve_ratio` `1.1f → 4.0f`
     — the capacity pins fail (`luisa_trajectory_filter_test.cpp(797)`: `12 == 3`;
     `(1197)`: `8 == 2`), i.e. the retry/growth contract is pinned, not just the surviving set.
     P2 (**device/kernel**): the AL kernel's `t0 = -dot(N, diff) - thickness` doubled
     (`- 2.0 * thickness`) — the AL toi grid pins fail (`(1147)`: `0.4995 == 0.74924999999999997`
     and `2.99699999999999989 == 3.49650000000000016`), with the values staying positive so the
     manager's `toi > 0` assert does not fire (a *value* pin, not a crash).  Both files were
     restored to their exact sha256 (`easy…h` `11c4ff4d…`, `al…cpp` `8964fa1f…`), rebuilt (rc 0) and
     re-run green (423/7).
 20. **Reproduction.** `python scripts/luisa_build.py build backend_luisa` (rc 0, 0 warnings) →
     `python scripts/luisa_build.py run "[filter]"` (7 cases / 423 assertions, rc 0; every quoted
     number is printed by the test: candidates 3 over 6 surf vertices, the two-plane band 2,
     `easy toi` 0.675 / 0.7875 / 0.9 / 1.1(away) / 1.1(sentinel) with the manager step 0.675,
     `al toi grid` 3 pairs, the predicate shape 1139 / 602 / 1139, the device-vs-host
     `D_range`/`is_active_D` 68 cases / 0 mismatches).  The battery is
     `[filter]` → `~[filter]` 763636/232 → the whole suite 764059/239 → the fifteen sibling filters →
     `[luisa][t43]` → ctest 1/1 → `python scripts/luisa_standalone.py cuda` 69/0 →
     `python scripts/luisa_module_def.py --check` rc 0 (463 symbols) →
     `python scripts/luisa_check_module_abi.py` rc 0.  **What T7.2b owes:** the four BVH simplex
     filters (`filters/{lbvh,stackless_bvh,info_stackless_bvh,info_stackless_bvh_v0}_simplex_trajectory_filter.*`,
     ~7200 reference lines) built on this batch's `SimplexTrajectoryFilter` contract, their
     `candidate_PTs/EEs` + `toi_PTs/EEs` exports (the `GlobalActiveSetManager`/`GlobalContactManager`
     consumers' shape), and — per rules 1/2 — either a host-filter equivalence proof per predicate or
     the port's first device-data hook if a *node-level* test turns out to need device data the leaf
     predicate does not imply.

### 7.34 Patterns established in T7.2b (the four BVH simplex trajectory filters)

Scope: `collision_detection/filters/{lbvh,stackless_bvh,info_stackless_bvh,info_stackless_bvh_v0}_simplex_trajectory_filter.{h,cpp}`
(the reference's ~7200 lines), the single shared kernel header
`collision_detection/details/simplex_trajectory_filter_kernels.h`, `utils/primitive_d_hat.h`, two
additions to `luisa_device/linear/algorithms.h` (`compact_by_flags` / `select_valid_pairs` +
`StructSelectScratch`) and `apps/tests/backends/luisa/luisa_bvh_filter_test.cpp` (tag
`[bvh_filter]`, 9 cases / **594 assertions**).  Baselines: `[bvh_filter]` 594/9 rc 0;
`~[bvh_filter]` **764059/239** rc 0 = the T7.2a baseline exactly; the whole suite **764653/248**;
`ctest` 1/1; `python scripts/luisa_standalone.py cuda` 69/0; the export list 463 → **504**
symbols (`python scripts/luisa_module_def.py --check` rc 0, `scripts/luisa_check_module_abi.py`
rc 0).

 1. **The batch's central decision: keep-all traversal + a device *selection pass*, never
    "device data reaches the predicate".**  In the reference the predicate is invoked *inside*
    the traversal callback, i.e. **after** the fixed node test — `AtomicCountingLBVH::detect`
    reports `(i, id)` and the caller's lambda does `if(id > i && p(i, id))`; `StacklessBVH`'s
    leaf callback is `$if(pred(idx, other))`; the `Info*` variants surface
    `launch_query/launch_detect` + `prepare_query_result` with the same shape.  A predicate can
    therefore only *select among already-reported pairs*: it cannot change the visited set (it
    can only change the reported **count**, i.e. the pair-buffer growth/retry — rule 3).  The
    port spells exactly that: the BVH runs with `sftd::KeepAll{}` (a host-true predicate) to
    produce the maximal swept-AABB overlap set, and the reference's predicate body is
    re-implemented as a **device pass** over the raw pairs
    (`detect_AllP_CodimP_pass_kernel<Dcd>` … `detect_AllE_AllE_pass_kernel<Dcd>`) that writes a
    per-pair flag, followed by `lla::compact_by_flags` (a *stable* compaction, so the survivors
    keep the traversal's ticket order).  Measured on the port's own structures in T7.2a
    (keep-all 1139 == brute force; the selective `i % 2 == 0` run 602 == the host-filtered
    keep-all set) and again here per family (below).  This is why the batch adds **no** new
    predicate hook: LC predicates are recorded shaders, and a captured device view becomes a
    shader-bound argument (cookbook §7.32 rule 3).
 2. **The `Info` variants' node cull is *implied* by the leaf predicate, so `KeepAllNodes` /
    `KeepAllLeaves` are exact, not an approximation.**  `InfoStacklessBVH`'s node metadata is
    `bid = -1` / `cid = -1` unless *all* leaves under the node share a body/contact id; the
    reference's `NodePred` cull tests `table[qcid * dim + node_cid]` (and the same for the
    subscene table, and the `node_bid != qbid || body_self_collision` term).  The corresponding
    leaf predicate `allow_*_contact` is a conjunction that contains that very same
    `table[qcid * dim + cid]` term, so a node culled by the reference has no surviving leaf
    under it — dropping the cull loses nothing.  The port therefore never needs the row
    metadata: `KeepAllNodes{}` / `KeepAllLeaves{}` (generic + the 2-argument overload `v0`
    uses) reproduce the reference's surviving set for all four variants (pinned per variant by
    `[bvh_filter]`: the `-1`-propagation is exercised through the mask batteries, where the
    info variants still agree with the oracle pair-for-pair).
 3. **What the predicate *does* change is the reported count — pin the *capacity/retry*
    contract, not just the surviving set.**  With `KeepAll{}` the port's raw pair counts are the
    AABB-overlap counts (strictly larger than the reference's), which is exactly what the query
    buffers must be sized for.  `[bvh_filter]` pins: the initial capacities (the
    stackless-family buffers reserve `50 * 1024` pairs, the `AtomicCountingLBVH`'s starts empty
    and is padded to 1 by its `detect`/`query`), the two `reserve_ratio`s (`1.2` for the
    stackless family, the LBVH's private `1.1`), and the *retry* itself — quoted from the run:
    "37 reported edge pairs == 37 brute-force overlap pairs (the buffer started empty, so the
    query ran twice)".
 4. **A host-truth predicate is a *type*, and types are the shader cache key.**  `KeepAll{}` /
    `KeepAllNodes{}` / `KeepAllLeaves{}` are empty structs; every kernel factory here is
    `inline` (one shader per process, the T7.1a rule) and the predicate must stay **stateless**
    — which is also why the port cannot pass a lambda that reads a device array.
 5. **The four-slice `tois` layout is part of the exported contract.**
    `Impl::filter_toi` sizes `tois` to `pp + pe + pt + ee` and fills the slices in that order;
    the *only* two slices the reference exposes are `toi_PTs() = tois.cview(pp + pe, pt)` and
    `toi_EEs() = tois.cview(pp + pe + pt, ee)`.  `candidate_PTs()` / `candidate_EEs()` return the
    **whole buffers** (capacity-sized, *not* count-sized) — the T7.2a rule 3 repeated, because
    the four `QueryBuffer`s are grown by the retry path and the pair buffers keep that capacity.
 6. **The PP/PE candidate counts are *not observable* through the pipeline.**  Only the PT/EE
    slices are exported, so a test that reads the two accessors cannot see the PP/PE
    survivors at all.  `[bvh_filter]` therefore pins them through the white-box route
    (`Impl::detect` at the DCD `alpha == 0`: raw and selected counts per family) *and* through
    the active sets, which are derived from them — quoted: pipeline `PP+PE=(not exported)`
    (the inferred `0` of the raw accessor reading was a red herring; the oracle's `4` is only
    visible in the white box, where the device answers `PP=1 PE=3 …` == the oracle `1/3`).
 7. **`alpha == 0` is a *branch* of the predicate, and only v1 has it.**  `info_stackless_bvh`'s
    four detect predicates evaluate the exact distance (`D2 < expand * expand`, via
    `*_distance2`) when `alpha == 0`, while the other three always use `*_ccd_broadphase`.  The
    port spells it `if constexpr(Dcd)` + a runtime `$if(alpha == 0.0)`, and the difference is
    measurable and pinned: at `alpha == 0` on the same scene the four-way DCD table is
    `PP/PE/EE/PT = 1/3/13/10` for `lbvh`, `stackless` and `infov0` (== the box/broadphase
    oracle) but **`1/2/13/7` for `infov1`** (== the exact-distance oracle, 3 PE + 3 PT pairs
    rejected that the broadphase keeps).  Every sub-scene battery repeats the split
    (identical-geometry `1/3/13/10` vs `1/2/8/6`), so a careless transcription of the branch
    fails loudly instead of silently.
 8. **The reference's `pe_from_ee` out-of-range read is transcribed and pinned as *undefined
    behaviour*, not "fixed".**  In the reference's k4 (edge-edge) `dim == 3` degeneracy
    classification, `pe_from_ee` is indexed by the *inactive* index and reads the ascending
    layout's slot 3, which `dim == 3` leaves at the "none" sentinel — an out-of-range read of
    the 4-element vertex list.  The device reports **no entry** for such pairs (measured: 3 EE
    candidates per variant), the test's oracle skips them and the run prints the quirk
    ("3 edge-edge candidate(s) classified through `pe_from_ee`'s out-of-range slot (undefined in
    the reference) -- the oracle skips them, and the device reports no entry for them").  Do not
    "repair" it: the port's job is to reproduce the reference's observable behaviour.
 9. **The sentinel is `1.1f` and it is written by three different paths.**
    `kLargeEnoughToi = 1.1` is (a) the kernels' initial value, (b) re-written after a CCD
    miss (`$if(!hit) toi = kLargeEnoughToi;`), and (c) the *host* fill
    `info.toi().fill(kLargeEnoughToi)` for an empty family (the faithful spelling of CUB's
    "an empty reduce does not launch").  `filter_toi` otherwise reduces with
    `lla::device_reduce_min` over the whole slot.  The `toi > 0` `RUNTIME_CHECK` in the manager
    (T7.2a) is what makes a `NaN`/zero transcription fail loudly.
10. **`kToiMaxIter = 16`, not the reference's `kCcdUnboundedIterations = 64`.**  The port's CCD
    unrolls once per iteration, so the 64-iteration kernels took **> 900 s** to JIT/run; the
    convergence-sized bound `kMaxIter = 1000` is what the port passes to the *host* oracle
    comparisons.  Pinned by measurement: `1000 vs 16 → 0 differing toi values` (per variant, in
    the *closing* probe where the values are non-trivial — see rule 16).
11. **The scene-config gate: three of the four filters are unreachable from a frontend scene.**
    This build's schema lists only `info_stackless_bvh` for `collision_detection/method`
    (`conditionalValues` drops the other three when `UIPC_WITH_CUDA_LEGACY_COLLISION == 0`), so
    the port's LBVH / stackless / v0 filters cannot be selected by a scene (and
    `apps/tests/core/scene_config.cpp` asserts `listed == legacy_collision_built`, so extending
    the enum would break a core test).  `[bvh_filter]` drives them with **test-local
    `SimplexTrajectoryFilter` subclasses** that own the variant's production `Impl`
    (`T72bVariantFilter<ImplT>` + the three concrete wrappers, `REGISTER_SIM_SYSTEM`), gated on
    a `test_t72b_variant` meta uid the scene builder sets on its geometries (`SimSystemException`
    otherwise) so no other test's scene is affected.
12. **Every ipc scene of this family needs a half-plane** (T7.2a's finding, restated because it
    bit again): the two interval filters' `filter_active` reads the plane's vertex slice and
    crashes on a plane-less scene, so `[bvh_filter]`'s scenes all carry one, parked 100 units
    below the geometry so it never contributes a candidate.
13. **Two teardown hazards the test has to work around (both measured, both *not* port bugs).**
    (a) Destroying a white-box `Impl` (the LBVH/v0 ones) aborts the *later* engine teardown, so
    the test leaks its probe `Impl`s; (b) writing the global contact-id array mid-case
    (`set_contact_ids`) crashes a later teardown, so the contact-mask direction is exercised in
    a scene of its own instead of by mutating a live one.
14. **The TOI comparison must be keyed by *pair*, never by index.**  The device's `toi_pt[k]` is
    parallel to `candidate_PTs()[k]` (traversal/compaction order), the host oracles are
    pair-keyed maps, and the two orders differ per structure (the same scene gave 13 vs 6
    "mismatches" between the LBVH and the stackless variants).  The batch's first version walked
    the slices element-wise — it *passed* only because every value in that scene was the same
    `1.1` sentinel, and it reported bogus mismatches the moment the pairs really collided.  Fix
    by re-keying the device's values (`toi_by_pair(candidates, tois)`), then compare per pair.
15. **A scene whose "displacement" rule matches nothing makes `alpha` invisible.**  The
    test's original assignment (`m.positions[0].y() > 1.0 ? {-0.01} : {0}`) matched *no* vertex
    (the upper plate sits at y = 0.15), so every `alpha`-dependent oracle was constant and the
    whole `alpha = 0.5` battery silently degenerated to a static scene.  Fixed with an explicit
    `is_upper_plate(p) { return p.y() > 0.05; }`.
16. **A *positive control* is mandatory: prove the probe exercises what it claims.**  Because
    of rule 15 the first version's "0 mismatches, max|d| = 0" was vacuous (sentinel vs
    sentinel).  The batch therefore adds a **closing probe** — the same pipeline re-run on a copy
    of the scene whose upper plate drops 0.35 (through the lower one) — and asserts its own
    meaningfulness: `non_sentinel > 0` (measured **14 of 23**), `p.step < 1` and
    `toi.min_all < 1`.  Its values are the batch's strongest numerical evidence: `min toi =
    0.12857142857142856 == step`, `0 mismatches`, `max|d| = 1.11e-16` (one ULP: the device's
    CCD and the host port of it differ only in evaluation order).  Without this probe the
    device/kernel perturbation of rule 18 stays invisible.
17. **Perturbation 1 — host/plumbing axis (the count bookkeeping): caught, rc 127, reverted
    byte-identically.**  `detail::compact_scatter_kernel`'s published total
    (`offsets[n-1] + flags[n-1]`) was replaced by the dispatch count `n`
    (`luisa_device/linear/algorithms.h`), i.e. every family exports the *raw* AABB set instead
    of the filtered set.  **34 assertions fail, rc 127**, in both directions of the pin:
    `luisa_bvh_filter_test.cpp(1806/1807)` (`CHECK(pt_cand == oracle.allp_allt)` /
    `(ee_cand == oracle.alle_alle)`: the device exports 11/14 pairs where the oracle has 10/13)
    and `(1856…1859)` (the four active-set multisets, e.g. `PPs {{?},{?}} == {{?}}`).  The file
    was restored to its exact sha256
    (`2ebbb8b3de5f60f9b21d5640e1cbc36b4a7d442dab56385559f4af0c8429aaee`), rebuilt (rc 0) and
    re-run green (594/9).
18. **Perturbation 2 — device/kernel axis (a TOI kernel's arithmetic): caught by the *closing*
    probe only, rc 42, reverted byte-identically.**  The PT TOI kernel lost the `alpha` scaling
    of the vertex displacement (`const auto dVP = alpha * load_vec3(dxs->read(V));` →
    `load_vec3(...)`, `simplex_trajectory_filter_kernels.h`), i.e. the CCD is solved over the
    full displacement instead of the alpha-scaled one.  **8 assertions fail, rc 42**, all of them
    in the closing probe: `(2005)` the pair-keyed mismatch count (4 of 23 toi values differ,
    `max|d| = 0.0643`) and `(2006)` the manager step (device `0.064285714285714279` vs oracle
    `0.12857142857142856`).  The *base* scene does **not** catch it — every one of its tois is
    the sentinel — which is exactly the argument for rule 16.  Restored to sha256
    `f26d2b3ff19fda05ccbf98e8d471f97637268ee799b693d57f10e461636f17a3`, rebuilt (rc 0), re-run
    green (594/9).
    **Perturbation 3 (host/plumbing again, and the most instructive one): the *production*
    accessor's offset.** `InfoStacklessBVHSimplexTrajectoryFilter::toi_EEs()`'s slice start dropped
    its PT term (`pp + pe + pt` → `pp + pe`).  The first attempt at this batch's test drove *every*
    variant through a test-local mirror whose accessors re-derive the same slices, so the
    production accessors were **never called** and this perturbation stayed invisible (measured:
    rc 0, green with the bug in place!).  That is the coverage gap the perturbation exercise is
    *for*: `check_variant_battery` now also looks up the variant's **production** filter
    (`gtf->find<ImplOwnerT>()` — non-empty for `info_stackless_bvh`, the only method a scene can
    select in this build, and the manager's stage loop drives every registered filter, so its
    `Impl` is filled by the same run) and compares its four exported slices with the mirror's.
    With that block in place the same perturbation fails loudly: **1 failed assertion, rc 42**, at
    `luisa_bvh_filter_test.cpp(2024)` (`CHECK(read_device(prod->toi_EEs()) == p.toi_ee)`).  For the
    other three variants the pin cannot exist by construction (their production filters are
    unregisterable — the config gate of rule 11), which is the honest limit of this check.
    Restored to sha256 `f0f44a415bbf29b2cc8ac502e4d4de846fb711e9e3781d5f85a820882bff8593`, rebuilt
    (rc 0), re-run green (594/9).
19. **Reproduction and battery.**  `python scripts/luisa_build.py build backend_luisa` (rc 0,
    0 warnings) → `run "[bvh_filter]"` (9 cases / 594 assertions, rc 0) → `run "~[bvh_filter]"`
    (764059/239 = the T7.2a baseline exactly) → the whole suite (764653/248 = +594/+9) → the
    seventeen sibling tags unchanged (`[filter]` 423/7, `[bvh]` 219/8, `[bvh_info]` 226/9,
    `[half_plane]` 426/7, `[mas_preconditioner]` 234/7, `[fem_precond]` 104/4, `[fem_linear]`
    323/4, `[fem_dynamics]` 135/8, `[abd_base]` 2143/7, `[abd_constitution]` 523/7,
    `[abd_constraint]` 471/9, `[abd_dynamics]` 547/7, `[abd_joint_a/b/c]` 223/8 · 402/8 · 383/8,
    `[abd_linear]` 495/7, `[abd_solve]` 245/8) plus `[luisa][t43]` 238/4 → `ctest` 1/1 →
    `python scripts/luisa_standalone.py cuda` 69/0 → `python scripts/luisa_module_def.py --check`
    rc 0 (504 symbols) → `python scripts/luisa_check_module_abi.py` rc 0.  The export list needed
    two bootstrap rounds again (24 then 17 unresolved mangled names, harvested from the LNK2019
    lines — take only the **first** `(?…)` match on a line: MSVC prints the symbol *and* the
    referencing function on the same line).
20. **What T7.2b owes the next milestones.**  The four filters are wired into
    `GlobalTrajectoryFilter` (T7.2a's registry) and export exactly the reference's surface
    (`candidate_PTs/EEs`, `toi_PTs/EEs`, plus the base's friction/active-set accessors); their
    Newton-loop callers are still T8.4's (`run_stage` remains the test seam), and the
    `GlobalActiveSetManager` / `GlobalContactManager` consumers of
    `candidate_PTs/EEs` + `toi_PTs/EEs` are T8.1/T8.5's.  M7 has no uid of its own: the port
    claims **none** in this batch.

### 7.35 Patterns established in T8.1a (the contact system core: the completed GlobalContactManager, the reporter/exporter layer, the adaptive-parameter layer)

**Scope.** The first M8 batch: the contact *core* — `contact_system/contact_coeff.h`,
`contact_reporter.{h,cpp}`, `contact_exporter.{h,cpp}`, `contact_exporter_manager.{h,cpp}`,
`contact_system_feature.{h,cpp}`, `adaptive_contact_parameter_reporter.{h,cpp}`,
`adaptive_strategies/gipc_adaptive_parameter_strategy.{h,cpp}` and the **completion in place** of
`global_contact_manager.{h,cpp}` (all eight `SCOPE NOTE (T8.1)` markers resolved), plus
`apps/tests/backends/luisa/luisa_contact_core_test.cpp` (tag `[contact_core]`, **8 cases / 296
assertions**).  Baselines: `[contact_core]` **296/8** rc 0; `~[contact_core]` **764653/248** rc 0 =
the T7.2b baseline exactly; the whole suite **764949/256**; `ctest` 1/1;
`python scripts/luisa_standalone.py cuda` 69/0; the export list 504 → **523** symbols
(`python scripts/luisa_module_def.py --check` rc 0, `scripts/luisa_check_module_abi.py` rc 0).
M8 is re-partitioned four ways (T8.1a this batch; T8.1b the IPC models + function headers +
exporter wiring; T8.1c `contact_models/analytical_barrier_pFpx/*`; T8.1d the AL subsystem);
T8.5 keeps only `diff_sim/*`, `line_search/*`, `newton_tolerance/*` and the engine leftovers.

  1. **Both `DeviceBuffer2D`s are flat buffers + an extent, and the `ContactCoeff` table is the
     same object the adaptive strategy swaps.**  `DeviceBuffer2D<IndexT>` /
     `<ContactCoeff>` → `lud::LDeviceBuffer<T>` with `index = row * N + col`, and the extent
     travels as `contact_mask_size()` / `subscene_mask_size()` / `contact_tabular_size()` (the
     reference's `CBuffer2DView::extent().width`).  The coefficient table keeps the reference's
     *shared-pointer shape* (`S<lud::LDeviceBuffer<ContactCoeff>>`) because
     `AdaptiveParameterInfo::exchange_contact_tabular` swaps that pointer with `std::exchange`.
     `ContactCoeff` is registered with `LUISA_STRUCT(..., kappa, mu)` in `contact_coeff.h`.
  2. **The default-kappa policy, verbatim — including a clamp the reference's own comment does
     not hint at.**  `min_kappa`/`max_kappa` start at the config
     `contact/adaptive/{min,max}_kappa` and are *replaced* by the corridor when it is computable;
     `default_kappa = resistance_view[0]`; if `!default_model_is_user_set()` it becomes
     `min_kappa`; else if `default_kappa >= 0 && (default_kappa < min_kappa || > max_kappa)` it
     is `std::clamp`ed (a *negative* kappa is never clamped — the adaptive opt-in).  The per-model
     loop repeats the same clamp with `kk = (i == 0) ? default_kappa : resistance[i]`, and writes
     the coefficient **and** the mask to `(x,y)` and `(y,x)` (the symmetric half).  Pinned by the
     test's hand-derived table: an enabled model with `resistance == 0` ends up at
     `min_kappa`, not at 0.
  3. **The scene-adaptive kappa corridor: formula, convention and the only way to make it not
     computable.**  `d = kappa_eval_scale * diag^2`, `dHat = d_hat^2`, `t = d - dHat`,
     `H_b = -2 ln(d/dHat) - 4 t/d + t^2/d^2`, `base = 1e11 / (4 d H_b)`,
     `base *= mean_mass` only when `mean_mass > 0`, `lower = base / dt^2`, `upper = 100 * lower`
     (the `dt^2` factor converts Stiff-GIPC's raw barrier to our convention).  `mean_mass` is
     `(Σ FEM vertex masses + Σ ABD body masses) / vertex_count` — **0 in a mesh-only scene**, so
     the mass term is skipped there.  **`contact/d_hat = 0` is not a legal config**
     (`exclusiveMinimum: 0`), so the "not computable" branch (`d <= 0 || dHat <= 0 || dt <= 0`,
     `!isfinite(H_b) || H_b <= 0`) is reached with a *large* `kappa_eval_scale` (measured:
     `1e6` → `d >> dHat` → `H_b < 0` → both bounds stay `-1.0`).  The sentinel is `-1.0`, not 0.
  4. **The six device kernels: two CFL norm kernels + the four feasible-step CCD kernels, and
     they are *all* the reference's.**  `compute_cfl_condition_kernel` writes
     `‖disps(surf_vertices(i))‖` (the *surface* set), `..._all_kernel` writes `‖disps(i)‖` for
     the no-surface-manager fallback; the four feasible-step kernels read one pair array each
     (`LVector4i/LVector4i/LVector3i/LVector2i`), sum the codim thicknesses
     (`PT/EE/PE/PP_thickness` from `utils/codim_thickness.h`), call the ported additive CCD
     (`luisa_device::math::distance::point_triangle_ccd` / `edge_edge_ccd` / `point_edge_ccd` /
     `point_point_ccd`) with `eta = 0.2` and write `select(1.0, toi, hit)`.  `cb::DeviceReduce::
     Min/Max` → `lla::device_reduce_min/max`, and the reference's *implicit* `DeviceVar` reads
     become `lud::read_host_sync(frame_stream(), var.view())`.
  5. **The batch's one difficult mapping: the reference's `max_iter = 1000` cannot be
     transcribed.**  `luisa_device/math/distance/details/ccd.inl`'s `ccd_loop` takes the
     iteration budget as a **host** constant (`ccd_iterations(max_iter)`) and *unrolls* it into
     the AST — the reference's `max_iter` is a device value, the port's is not.  Passing `1000`
     made the four kernels' first JIT die (SIGSEGV inside `compute_feasible_step`, measured), so
     the port passes `kFeasibleToiMaxIter = 16`, the same budget the ported trajectory filters
     pass to the *identical* CCD call for the *identical* loop
     (`details/simplex_trajectory_filter_kernels.h: kToiMaxIter`).  With `eta = 0.2` the loop's
     gap test fires within **two** iterations for every approaching pair and the non-approaching
     pairs exit on `toc_cur > toc_prev`, so the returned `toc` is unchanged; the test pins the
     closed form for all four kernels.
  6. **The feasible-step `toc` has a closed form worth pinning.**  For a simplex pair whose
     relative motion is *collinear* with the separation, `ccd_loop` steps
     `toc_lower = (1 - eta) (d^2 - th^2) / ((d + th) * maxDispMag)`, so the distance falls to
     `th + eta (d0 - th)` after one step and the gap test fires on the second — the returned
     `toc` is `(1 - eta) (d0 - th) / (d0 - d1)` (`d0`/`d1` the distances at t = 0 and t = 1,
     `maxDispMag = |d0 - d1|` for collinear motion).  The test lays out all four pairs with
     `d0 = 1`, `d1 = 0.1`, `thickness = 0.05` → `t* = 0.844444…`; the *device* value differs from
     the host formula in the last ulp (`0.84444444444444433` vs `…44`), so compare with a
     tolerance, never with `==`.  A `toc > 1.0` result is *rejected* by the loop (`toc_cur >
     toc_prev`) and the kernel writes the `1.0` sentinel; `compute_feasible_step` also clamps
     the reduced min to `<= 1.0`.
  7. **`distance::point_triangle_ccd` etc. are the T2.3 layer, and the *host* flavour is the
     oracle.**  `luisa_device::math::distance::*_ccd` has a host (`T = double`) and a device
     (`T = Var<double>`, `bool_of_t` result) overload over one shared `_impl`; the device
     overloads take `T max_iter` as an `int` argument.  `*_thickness` live in
     `utils/codim_thickness.h` and their `RUNTIME_CHECK` asserts become `lcrt::device_assert`
     on the device path (`codim_thickness_detail::assert_same_thickness`).
  8. **A `ContactReporter` registers with *two* managers, and that is the reporting pipeline's
     whole shape.**  It derives from `DyTopoEffectReporter` (so `GlobalDyTopoEffectManager`
     drives it through extent → assemble / energy) and overrides
     `do_build(DyTopoEffectReporter::BuildInfo&) final` to also call
     `GlobalContactManager::add_reporter(this)`; `component_flags()` is forced to
     `EnergyComponentFlags::Contact`, which is what partitions it out of the complement set.
     `GlobalContactManager::Impl::init` fans `init()` out and then writes `R->m_index = i` (the
     index contract lives in `m_index`, private with `friend GlobalContactManager`).
  9. **The exporter/feature seam is `ContactExporter` (a `SimSystem`) → `ContactExporterManager`
     → `core::ContactSystemFeature`.**  `ContactExporterManager::do_build` builds a
     `ContactSystemFeatureOverrider` + a `core::ContactSystemFeature` and `features().insert(...)`
     (`features()` is the `SimSystem` base's hook); `init()` builds the `prim_type -> exporter`
     map and the ordered `m_contact_prim_types`; `find_exporter` warns and returns `nullptr` for
     an unknown type, and `_create_prim_type_on_geo` writes the `prim_type` `std::string` meta
     onto the geometry the caller passes.  The frontend reads it back through
     `engine.features().find<core::ContactSystemFeature>()` — `contact_energy/gradient/hessian
     (prim_type, Geometry&)` and `contact_primitive_types()`.  The reference header's four
     `contact_system/*_contact.h` includes are dead in its own class body *and* are T8.1b's
     types, so the port drops them (the T5.3 rule).
 10. **`AdaptiveParameterInfo` is a true `std::exchange`, and the tabular pointer identity is
     the contract.**  `exchange_contact_tabular(new)` returns the *previous* `S<...>` and installs
     `new` (`std::exchange(m_impl->contact_tabular, new_buffer)`); `contact_tabular()` hands out
     the flat `LBufferView<ContactCoeff>`.  The test probe exchanges `nullptr` first (proving the
     returned handle is the manager's original), installs a replacement, reads it back through
     `GlobalContactManager::contact_tabular()`, then restores and asserts the swap returned the
     replacement — three pointer-identity pins.
 11. **The adaptive reporter slot is single, and the *scene* decides who owns it.**
     `AdaptiveContactParameterReporter::do_build` throws unless some resistance is `<= 0`;
     `GIPCAdaptiveParameterStrategy::do_build` throws unless some resistance is `< 0`; and
     `GlobalContactManager::add_reporter(AdaptiveContactParameterReporter*)` **asserts** the slot
     is empty.  So a scene with a model at `resistance == 0` makes the *probe* reporter valid and
     GIPC invalid, and a scene with `resistance < 0` does the opposite — exactly the two scenes
     the test builds (a second reporter in either scene would abort the process).
 12. **The reference's dead code was transcribed as dead code.**  `Impl::compute_d_hat()` /
     `Impl::compute_adaptive_kappa()` are empty `// TODO: Now do nothing` bodies with no caller in
     the whole reference backend (kept); the GIPC strategy's two `copy_to` host copies
     (`h_contact_gradient` / `h_non_contact_gradient`) are never read (kept, with the port's
     explicit `read_back_vector` because `LDenseVector` has no `copy_to(std::vector&)`); the
     reference's `ContactReporter::Impl` duplicates the base's three views (kept, unread).
     Nothing was "fixed".
 13. **Test seams.**  `GlobalContactManager` keeps the reference's private surface and adds, in
     the header's documented `T8.1a test seam` block, `Impl& impl()` and
     `run_compute_adaptive_parameters()` (the T7.2a `GlobalTrajectoryFilter::run_stage`
     precedent; the reference's `compute_cfl_condition` / `compute_feasible_step` /
     `compute_adaptive_parameters` are private with only the Phase-8 Newton loop as their caller).
     Everything else the test reads is public in the reference too (`contact_tabular()`,
     `kappa_lower_bound()/upper_bound()`, `d_hat()`, the `Impl` members).
 14. **Scene-gating a probe system in the shared test binary.**  `REGISTER_SIM_SYSTEM` is
     process-wide, so every test-local probe exists in *every* scene the shared
     `uipc_test_backend_luisa.exe` builds (including `[filter]` / `[bvh_filter]`) — a probe
     `ContactReporter` would otherwise pollute those scenes' dytopo-effect assembly, and a probe
     `SimplexTrajectoryFilter` would be found by `find<SimplexTrajectoryFilter>()` there.  The
     port's technique is the reference filters' own `throw SimSystemException` gate plus a
     **geometry-meta marker** (`t81a_probe` on the mesh / half-plane geometry) that only the
     `[contact_core]` scenes carry; `Scene::config()` rejects unknown keys and the
     `collision_detection/method` enum rejects unknown values, so a marker attribute on a
     *geometry* is the only place a test may put its own flag.
 15. **A scene that drives the trajectory pipeline needs a half plane -- FIXED at the filter.**
 `EasyVertexHalfPlaneTrajectoryFilter` is valid in *any* IPC scene (`do_build` only requires
 `IPCPipelineFlag`) and its `filter_active` / `filter_toi` dispatch their kernels with
 `plane_positions` / `plane_normals`; with **no** half-plane geometry those are zero-length buffer
 views, which LC rejects as kernel arguments (the recorded harness hazard: a SIGSEGV *inside* the
 filter, before its own log line).  T7.2a/T7.2b worked around it by parking a far-away half plane in
 every `run_stage()` scene (`t81a_feasible_step` still does; the `[bvh_filter]` scenes park one 100
 units below).  **The port now skips the dispatch when `n_planes == 0`**, which is exact: both
 kernels' whole bodies sit inside their `$for(j, 0u, n_planes)` loop, so they write nothing at all/r/n (not even `num_collisions`), and `filter_toi` skips its reduce with the dispatch -- otherwise the/r/n reduce would consume the *unwritten* `tois` and clobber the caller's `info.toi()` 1.1 fill, which/r/n *is* the reference's answer for a scene with no plane in range.  The idiom is/r/n `ALVertexHalfPlaneTrajectoryFilter`'s (it had it since T7.2a; the easy twin did not).  Proof: the/r/n new `t81a_plane_less_scene` fails with **SIGSEGV rc 3221225477** when the guard is reverted and is/r/n green when restored, and `~[contact_core]` stays at the previous baseline exactly, so the fix is/r/n behaviour-neutral for every existing scene./r/n 16. **The measured corridor numbers (mesh-only scenes, `scale = 1e-16`, `dt = 0.25`).**  a
     3-vertex unit mesh (`diag = sqrt(2)`) → `[8000, 800000]`, default kappa `8000`; a 4-vertex
     mesh (`diag = sqrt(3)`) → `[12000, 1.2e6]`; the earlier T7.2a scene (`diag = 1.4142135`,
     `dt = 0.25`) → `[8.0e3, 8.0e5]`.  The corridor **overrides** the config
     `[100 MPa, 100 GPa]` fallback in every computable scene, so a default trimesh scene's
     `ContactCoeff::kappa` is the corridor's lower bound, not `1e8`.
 17. **The export-list step, again.**  Adding the test made the link fail with **19**
     unresolved externals (the new `GlobalContactManager` accessors, the `AdaptiveParameterInfo`
     pair, `Impl::compute_cfl_condition/compute_feasible_step`, `ContactReporter::do_build/
     do_init`, `ContactExporter::do_build`, `AdaptiveContactParameterReporter::do_build`,
     `SimplexTrajectoryFilter::FilterActiveInfo::PTs/EEs/PEs/PPs`, and
     `DyTopoEffectLineSearchReporter::Impl::compute_energy` — the test drives the line-search
     reporter's `Impl::compute_energy` directly, which is a `friend` of the dytopo manager).
     Recipe as usual: harvest the **first** `(?…)` on each LNK2019/LNK2001 line, append them to
     the committed `.def`, relink (the exe must exist before the generator can read its import
     table), then `python scripts/luisa_module_def.py` (504 → **523**) and `--check` rc 0.
 18. **Uid hygiene.**  The batch claims **no uid**: every new system is a plain `SimSystem` /
     reporter (no `get_uid`, no `(dim, uid)` geometry group), and the five test-local probes
     claim none either, so no reserved mock uid (8/16/25/26/27/28/666/671) is touched.  The
     proof that adding the systems did not shift the system-build order is the exclusion run:
     `~[contact_core]` came out at **764653/248 exactly**, i.e. every previously green case
     still sees its own systems first.
 19. **Reproduction.**  `python scripts/luisa_build.py build backend_luisa` (rc 0, 0 warnings) →
     `python scripts/luisa_build.py run "[contact_core]"` (8 cases / 296 assertions, rc 0; the
     four printed lines carry every quoted number).  Battery: `[contact_core]` 296/8 →
     `~[contact_core]` 764653/248 → the whole suite 764949/256 → the 18 sibling tags
     (`[bvh_filter]` 594/9, `[filter]` 423/7, `[bvh]` 219/8, `[bvh_info]` 226/9, `[half_plane]`
     426/7, the nine `[abd_*]`, `[mas_preconditioner]` 234/7, `[fem_precond]` 104/4,
     `[fem_linear]` 323/4, `[fem_dynamics]` 135/8, `[luisa][t43]` 238/4) → `ctest` 1/1 →
     `python scripts/luisa_standalone.py cuda` 69/0 → `python scripts/luisa_module_def.py --check`
     rc 0 (523 symbols) → `python scripts/luisa_check_module_abi.py` rc 0.
 20. **What T8.1a does *not* cover, and T8.1b owes.**  The contact tabular is built and read, the
     corridor/CFL/feasible-step caps are driven over real scenes, and a probe reporter + probe
     exporter exercise the reporting pipeline end-to-end — but the *real* models are missing, so
     (i) `ContactExporterManager` has no concrete exporter until T8.1b, (ii) the `ContactReporter`
     gradient/Hessian *content* is pinned only as the slice sizes the manager hands out (the
     writers' numbers are the energies and the exporter path), and (iii) the GIPC strategy's
     `do_compute_parameters` is **not** driven here (it needs `GlobalLinearSystem` +
     `GlobalDyTopoEffectManager::compute_dytopo_effect`, i.e. the Phase-8 loop); the test pins its
     resolved state (`h_adaptive_kappa_index`, `test_contact_tabular`, the corridor/config bounds)
     and re-derives its clamp law on the host.  `contact_receivers` keeps the reference's
     **dangling forward declaration** (`class ContactReceiver;` is never defined nor registered in
     the reference either) and is asserted empty.
 21. **The per-vertex attribute plumbing in the *test target* is a `test_*` convention -- a
 plain `verts.create<Float>("thickness", ...)` is dead.**  The manager's per-vertex arrays are
 written by whichever `VertexReporter` wins the uid group; in this test executable that is the
 sibling `GlobalGeometryTestVertexReporter` (`luisa_global_geometry_test.cpp`), which reads the
 attributes named `test_dimension` / `test_contact_element_id` / `test_subscene_element_id` /
 `test_d_hat` / `test_thickness` / `test_displacement` and pushes them into the manager's buffers
 (`luisa_trajectory_filter_test.cpp` relies on the same convention; T8.1a measured it: a scratch
 mesh with no `test_d_hat` gets `d_hats == 0`, while the *half-plane* geometry's vertex -- an
 implicit geometry, so not in `scene_complexes()` -- keeps the manager's `resize_init` default
 `contact/d_hat == 0.01`).  A test that needs a value in the manager's arrays must therefore either
 create the `test_*` attribute or write the device buffer directly (`set_thicknesses` / the new
 `set_d_hats` in `[contact_core]`).  Consequence for a half-plane DCD test: with `d_hat == 0` the
 activation band `(thickness, thickness + d_hat)^2` is empty and `filter_active` reports **zero**
 pairs while `filter_toi` (which does not use `d_hat`) happily returns a TOI -- an easy way to write
 a positive control that passes for the wrong reason.
 22. **`num_pairs()` is a documented *test seam*, not a reference API.**  `candidate_PHs()` cannot
 express "zero pairs", because the port pads that pair buffer to one element so that it stays a
 valid LC kernel argument (rule 7), so a zero-pair and a one-pair run both report
 `candidate_PHs().size() == 1`.  `EasyVertexHalfPlaneTrajectoryFilter::num_pairs()` returns the
 `h_num_collisions` the last `filter_active` recorded; `t81a_plane_less_scene` is its only user (0
 for a plane-less scene, 2 for the same mesh with a plane 0.005 below it -- the triangle's third
 vertex sits 1.0 *above* the plane, i.e. outside the band, so the control pins an in-band *and* an
 out-of-band vertex; the port's `PHs` capacity/retry kept the count honest at 2, which the first
 spelling of the oracle got wrong).


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
