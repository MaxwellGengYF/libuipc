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

7.8 Patterns established in T2.3 (distance / CCD)
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
