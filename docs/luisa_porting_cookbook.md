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
| Compiler | LC is built with **clang++ (LLVM 19, MSVC ABI)**. Consumers must use a conformant preprocessor. |

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

---

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
| Math namespace | `uipc::backend::luisa::luisa_device::math` (alias `lc::math` in backend code) |
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

---

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
