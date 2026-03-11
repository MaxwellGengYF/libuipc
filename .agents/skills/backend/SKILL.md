---
name: backend
---

# UIPC CUDA Backend Skill

This skill provides guidance for working with the UIPC (Unified Incremental Potential Contact) CUDA backend implementation.

## Overview

The CUDA backend (`src/backends/cuda/`) is a GPU-accelerated physics simulation engine built on top of the [muda](https://github.com/MuGdxy/muda) library. It implements an IPC (Incremental Potential Contact) based simulation framework with Newton's method for solving nonlinear systems.

## Architecture

### Class Hierarchy

```
backend::SimEngine
    └── cuda::SimEngine           # Main CUDA engine (entrance.cpp)

backend::SimSystem
    └── cuda::SimSystem           # Base for all CUDA systems
        ├── GlobalVertexManager   # Global vertex management
        ├── GlobalSimplicialSurfaceManager
        ├── GlobalBodyManager
        ├── GlobalContactManager  # Contact handling
        ├── GlobalTrajectoryFilter # CCD/Broad-phase
        ├── AffineBodyDynamics    # ABD simulation system
        ├── FiniteElementMethod   # FEM simulation system
        ├── GlobalLinearSystem    # Linear solver
        ├── TimeIntegratorManager
        ├── LineSearcher
        └── ...
```

## Key Files

### Core Headers

| File | Purpose |
|------|---------|
| `sim_engine.h` | Main CUDA engine class |
| `sim_system.h` | Base class for all simulation systems |
| `sim_engine_state.h` | Engine state machine enum |
| `type_define.h` | Type definitions and Eigen/muda compatibility |
| `kernel_cout.h` | Kernel-side console output utility |
| `energy_component_flags.h` | Flags for energy components (Contact/Complement) |

### Entry Point

```cpp
// entrance.cpp
UIPC_BACKEND_API EngineInterface* uipc_create_engine(EngineCreateInfo* info)
{
    return new uipc::backend::cuda::SimEngine(info);
}
```

## Simulation Pipeline

The simulation follows a Newton-based IPC solver with the following pipeline (in `sim_engine_do_advance.cu`):

```
Frame Loop:
├── Rebuild Scene
│   └── event_rebuild_scene()
├── Simulation
│   ├── 1. Process External Changes
│   ├── 2. Record Friction Candidates
│   ├── 3. Predict Motion (x_tilde = x + v*dt)
│   ├── 4. Adaptive Parameter Calculation
│   └── 5. Newton Iteration Loop
│       ├── Compute Animation Substep
│       ├── Build Collision Pairs
│       ├── Compute Dynamic Topo Effect (G, H)
│       ├── Solve Global Linear System (dx = A^-1 * b)
│       ├── Collect Vertex Displacements
│       ├── Convergence Check
│       └── Line Search
│           ├── CCD Filter
│           ├── CFL Condition
│           └── Energy Decrease Check
└── Update Velocity (v = (x - x_0) / dt)
```

## Engine States

```cpp
enum class SimEngineState
{
    None = 0,
    BuildSystems,
    InitScene,
    RebuildScene,
    PredictMotion,
    ComputeDyTopoEffect,
    SolveGlobalLinearSystem,
    LineSearch,
    UpdateVelocity,
};
```

## Creating a New System

### 1. Inherit from `SimSystem`

```cpp
#pragma once
#include <sim_system.h>

namespace uipc::backend::cuda
{
class MySystem final : public SimSystem
{
  public:
    using SimSystem::SimSystem;  // Inherit constructor

  protected:
    virtual void do_build() override;  // Called during system build
    
    // Optional: Dump/Recover support
    virtual bool do_dump(DumpInfo& info) override;
    virtual bool do_try_recover(RecoverInfo& info) override;
    virtual void do_apply_recover(RecoverInfo& info) override;
    virtual void do_clear_recover(RecoverInfo& info) override;

  private:
    // Register lifecycle actions in do_build()
    void on_init_scene(std::function<void()>&& action) noexcept;
    void on_rebuild_scene(std::function<void()>&& action) noexcept;
    void on_write_scene(std::function<void()>&& action) noexcept;
    
    // Access engine and world
    SimEngine& engine() noexcept;
    WorldVisitor& world() noexcept;
    
    // Check current engine state
    void check_state(SimEngineState state, std::string_view function_name) noexcept;
};
}  // namespace uipc::backend::cuda
```

### 2. Register the System

```cpp
// MySystem.cpp
#include <MySystem.h>
#include <backends/common/module.h>

namespace uipc::backend::cuda
{
void MySystem::do_build()
{
    // Register init scene action
    on_init_scene([this]() {
        // Initialization code here
    });
    
    // Register rebuild scene action  
    on_rebuild_scene([this]() {
        // Rebuild code here
    });
}
}  // namespace uipc::backend::cuda

// Auto-register the system
REGISTER_SIM_SYSTEM(MySystem);
```

### 3. Find/Require Other Systems

```cpp
void MySystem::do_build()
{
    // Find optional system (returns nullptr if not found)
    auto* vertex_manager = find<GlobalVertexManager>();
    
    // Require mandatory system (throws if not found)
    auto& linear_system = require<GlobalLinearSystem>();
    
    // Access engine
    auto& eng = engine();
}
```

## Key Patterns

### 1. Device Buffer Management (muda)

```cpp
#include <muda/buffer/device_buffer.h>

// Device buffer declaration
muda::DeviceBuffer<Vector3> positions;
muda::DeviceBuffer<Float>   masses;

// View access (read-only)
muda::CBufferView<Vector3> pos_view = positions.view();

// View access (read-write)
muda::BufferView<Vector3> pos_view = positions.view();

// Subview for segmented access
template <typename T>
muda::BufferView<T> subview(muda::DeviceBuffer<T>& buffer, SizeT index) const noexcept;
```

### 2. Kernel Console Output

```cpp
#include <kernel_cout.h>

// In host code, launch kernel with cout
ParallelFor()
    .apply(N, 
[cout = KernelCout::viewer()] __device__ (int i) mutable
{ 
    cout << "Value at " << i << ": " << data[i] << "\n";
})
```

### 3. Linear System Integration

```cpp
// For diagonal subsystems (e.g., FEM, ABD)
class MyLinearSubsystem : public DiagLinearSubsystem
{
    virtual void do_build() override
    {
        // Extent DOF count
        InitDofExtentInfo extent_info;
        extent_info.extent(dof_count);
        
        // Extent Hessian blocks
        DiagExtentInfo diag_info;
        diag_info.extent(hessian_block_count, dof_count);
        diag_info.component_flags(EnergyComponentFlags::Complement);
    }
    
    virtual void do_compute_gradient(ComputeGradientInfo& info) override
    {
        auto grad_view = info.gradients();
        // Compute gradient...
    }
    
    virtual void do_assemble(DiagInfo& info) override
    {
        auto H = info.hessians();  // TripletMatrixView
        auto g = info.gradients(); // DenseVectorView
        // Assemble Hessian and gradient...
    }
};
```

### 4. Geometry Access Pattern

```cpp
// FEM/ABD systems use ForEach pattern
void for_each(span<S<geometry::GeometrySlot>> geo_slots,
              ViewGetterF&&                   getter,
              ForEachF&&                      for_each) const;

// Example usage:
for_each(geo_slots, 
[](SimplicialComplex& sc) {
    return sc.transforms().view();  // View getter
},
[](const ForEachInfo& I, const Matrix4x4& transform) {
    auto bodyI = I.global_index();
    // Process each body...
});
```

## Directory Structure

```
src/backends/cuda/
├── affine_body/           # Affine Body Dynamics (ABD)
│   ├── constitutions/     # ABD constitutions (ARAP, Joints)
│   └── constraints/       # ABD constraints
├── algorithm/             # GPU algorithms
├── animator/              # Animation system
├── collision_detection/   # CCD and broad-phase
├── contact_system/        # IPC contact models
├── coupling_system/       # ABD-FEM coupling
├── cuda_device/           # CUDA device utilities
├── diff_sim/              # Differentiable simulation
├── dytopo_effect_system/  # Dynamic topology effects
├── engine/                # Engine implementation files
├── external_force/        # External force management
├── finite_element/        # FEM system
│   ├── constitutions/     # FEM constitutions
│   └── bdf/               # BDF time integration
├── global_geometry/       # Global geometry management
├── implicit_geometry/     # Half-planes, etc.
├── inter_primitive_effect_system/  # Stitching, etc.
├── line_search/           # Line search implementation
├── linear_system/         # Linear solver (PCG)
├── newton_tolerance/      # Convergence checking
├── time_integrator/       # Time integration
└── utils/                 # Utilities (distance, CCD, etc.)
```

## Important Conventions

### 1. Namespace
```cpp
namespace uipc::backend::cuda {
// All CUDA backend code goes here
}
```

### 2. Type Definitions
- Use `Vector3`, `Vector12`, `Matrix3x3`, `Matrix12x12` from `type_define.h`
- Use `IndexT`, `SizeT`, `Float` for generic types
- Use `U64` for 64-bit identifiers

### 3. Memory Management
- Use `muda::DeviceBuffer<T>` for device arrays
- Use `muda::DeviceVar<T>` for device scalars
- Use `vector<T>` (pmr) for host arrays

### 4. Error Handling
```cpp
UIPC_ASSERT(condition, "Error message with {}", value);
UIPC_WARN_WITH_LOCATION("Warning message");
logger::info("Info message {}", value);
```

## Build Requirements

- CUDA Toolkit (compatible with project's requirement)
- muda library (header-only, included)
- Eigen3 (with CXX20 support via muda)
- magic_enum for enum operations

## Tips

1. **State Checking**: Always use `check_state()` in callbacks to ensure you're in the correct engine state
2. **Friend Classes**: Use friend declarations for tight coupling between related systems
3. **Impl Pattern**: Use Pimpl idiom (class `Impl`) to hide implementation details
4. **Dump/Recover**: Implement dump/recover for checkpoint/restart functionality
5. **Template Methods**: Use template `for_each` methods for geometry iteration
