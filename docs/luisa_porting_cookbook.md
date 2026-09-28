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
