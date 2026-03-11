# libuipc Project Structure

## Overview

**libuipc** is a Unified Incremental Potential Contact (UIPC) library for physics simulation. It provides a modular, extensible framework for simulating deformable objects, rigid bodies, and contact dynamics using the IPC (Incremental Potential Contact) method.

The project follows a layered architecture with clear separation between:
- **Frontend API** (public headers in `include/`)
- **Core Implementation** (in `src/core/`)
- **Backend Implementations** (in `src/backends/`)
- **Python Bindings** (in `src/pybind/`)

---

## Directory Structure

```
libuipc/
├── include/uipc/          # Public C++ headers (frontend API)
├── src/                   # Source code
│   ├── backends/          # Simulation backends
│   ├── constitution/      # Constitution implementations
│   ├── core/              # Core library implementation
│   ├── geometry/          # Geometry utilities implementation
│   ├── io/                # I/O operations implementation
│   ├── pybind/            # Python bindings (pyuipc)
│   ├── sanity_check/      # Sanity checking system
│   ├── usd/               # USD (Universal Scene Description) support
│   └── vdb/               # OpenVDB integration
├── python/                # Python-related files
├── apps/                  # Example applications
├── docs/                  # Documentation
├── cmake/                 # CMake modules
└── external/              # External dependencies
```

---

## Core Architecture

### 1. Frontend API (`include/uipc/`)

#### 1.1 Common Utilities (`include/uipc/common/`)
Base types, macros, and utility classes used throughout the library.

| File | Purpose |
|------|---------|
| `type_define.h` | Type aliases (Float, Vector3, Matrix4x4, etc.) |
| `smart_pointer.h` | Smart pointer definitions (S<T>, U<T>) |
| `vector.h`, `map.h`, `unordered_map.h` | STL container wrappers |
| `json.h`, `json_eigen.h` | JSON serialization |
| `log.h`, `logger.h` | Logging infrastructure |
| `dllexport.h` | DLL export macros |

**Key Types:**
- `Float = double` - Scalar type for simulation
- `Vector3` - 3D vector using Eigen
- `Matrix4x4` - 4x4 transformation matrix
- `S<T> = std::shared_ptr<T>`
- `U<T> = std::unique_ptr<T>`

#### 1.2 Geometry System (`include/uipc/geometry/`)
Attribute-based geometry representation supporting simplicial complexes.

**Core Classes:**
- `Geometry` - Base geometry class with meta and instance attributes
- `SimplicialComplex` - Mesh representation (vertices, edges, triangles, tetrahedra)
- `ImplicitGeometry` - Implicit surface representations
- `AttributeCollection` - Container for geometry attributes
- `Attribute<T>` - Typed attribute storage
- `AttributeSlot<T>` - Handle to access attributes

**Attribute System:**
```cpp
// Example: Working with vertex attributes
auto sc = SimplicialComplex();
auto vertices = sc.vertices();
auto pos = vertices.create<Vector3>("position");
auto vel = vertices.create<Vector3>("velocity");
```

**Utilities (`include/uipc/geometry/utils/`):**
- `extract_surface.h` - Extract surface mesh from volume
- `label_surface.h` - Mark surface elements
- `apply_transform.h` - Transform geometry
- `merge.h` - Merge multiple geometries
- `mesh_partition.h` - Mesh partitioning
- `factory.h` - Geometry creation utilities
- `bvh.h`, `octree.h` - Spatial acceleration structures

#### 1.3 Core Simulation (`include/uipc/core/`)
Main simulation framework classes.

| Class | Purpose |
|-------|---------|
| `Engine` - Simulation engine interface
| `World` - Simulation world managing time stepping
| `Scene` - Scene container with objects, contacts, constitutions
| `Object` - Simulation object containing geometries
| `Animator` - Animation/kinematic control
| `DiffSim` - Differentiable simulation support
| `SanityChecker` - Pre-simulation validation
| `ContactTabular` - Contact model management
| `ConstitutionTabular` - Constitution registration
| `Feature` / `FeatureCollection` - Backend feature queries

**Object-Geometry Relationship:**
```
Scene
├── Objects[]
│   └── Geometries[]
│       ├── Geometry (current state)
│       └── Rest Geometry (rest state)
├── ContactTabular
├── ConstitutionTabular
└── Animator
```

#### 1.4 Constitution System (`include/uipc/constitution/`)
Material models and constraints for deformable objects.

**Base Classes:**
- `IConstitution` - Base interface for all constitutions
- `FiniteElementConstitution` - FEM-based materials
- `FiniteElementExtraConstitution` - Additional FEM effects

**Material Models:**
| Constitution | Type | Description |
|--------------|------|-------------|
| `StableNeoHookean` | 3D FEM | Stable Neo-Hookean elastic solid |
| `ARAP` | 3D FEM | As-Rigid-As-Possible deformation |
| `NeoHookeanShell` | Shell | Neo-Hookean shell model |
| `BaraffWitkinShell` | Shell | Baraff-Witkin cloth model |
| `DiscreteShellBending` | Shell | Discrete shell bending |
| `KirchhoffRodBending` | Rod | Kirchhoff rod bending |
| `HookeanSpring` | Spring | Hookean spring forces |
| `Particle` | Particle | Particle system |

**Affine Body System:**
- `AffineBodyConstitution` - Affine body dynamics
- `AffineBodyRod`, `AffineBodyShell` - Codimensional affine bodies
- `AffineBodyRevoluteJoint`, `AffineBodyPrismaticJoint` - Joints
- `InterAffineBodyConstitution` - Inter-body interactions

**Constraints:**
- `SoftPositionConstraint` - Soft position constraints
- `SoftTransformConstraint` - Soft transformation constraints
- `SoftVertexStitch` - Vertex stitching
- `ExternalArticulationConstraint` - External articulation

#### 1.5 I/O System (`include/uipc/io/`)
File format support for geometry and scene data.

| Class | Formats |
|-------|---------|
| `SimplicialComplexIO` | .obj, .ply, .stl, .msh |
| `SceneIO` | Scene serialization |
| `SpreadSheetIO` | CSV data export |
| `URDFIO` | URDF robot description |
| `GLTFIO` | glTF format support |

#### 1.6 Built-in Definitions (`include/uipc/builtin/`)
Standard attribute names and type identifiers.

**Key Attributes (from `attribute_name.h`):**
- `position` - Vertex positions
- `velocity` - Vertex velocities
- `transform` - Instance transforms
- `contact_element_id` - Contact element identifier
- `constitution_uid` - Constitution unique ID
- `is_fixed` - Fixed flag
- `is_dynamic` - Dynamic flag
- `thickness` - Shell thickness
- `volume`, `mass_density` - Physical properties

#### 1.7 Backend Interface (`include/uipc/backend/`)
Backend communication interfaces.

| Class | Purpose |
|-------|---------|
| `Buffer` / `BufferView` | GPU/Backend buffer abstraction |
| `visitors/SceneVisitor` | Scene data access for backends |
| `visitors/WorldVisitor` | World state access |
| `visitors/AnimatorVisitor` | Animation data access |
| `visitors/DiffSimVisitor` | Differentiable sim access |

---

### 2. Implementation (`src/`)

#### 2.1 Core Implementation (`src/core/`)
Implements the frontend API classes.

**Structure:**
```
src/core/
├── common/              # Common utilities (logger, timer, etc.)
├── backend/             # Backend visitor implementations
│   └── visitors/
├── builtin/             # UID registration for constitutions
├── constitution/        # Base constitution classes
├── core/                # Core classes (Engine, World, Scene, Object)
├── diff_sim/            # Differentiable simulation
└── geometry/            # Geometry system implementation
```

#### 2.2 Constitution Implementations (`src/constitution/`)
Concrete implementations of material models.

Each constitution implements:
- UID registration with the builtin system
- `apply_to()` method to tag geometries
- Backend-specific energy/gradient/hessian computation

#### 2.3 Geometry Utilities (`src/geometry/`)
Implementation of geometry processing algorithms.

**Key Components:**
- `bvh/` - Bounding Volume Hierarchy
- `graph_coloring/` - Graph coloring algorithms
- `affine_body/` - Affine body utilities
- `implicit_geometries/` - Implicit geometry shapes

#### 2.4 I/O Implementation (`src/io/`)
File format parsers and writers.

Includes third-party implementations:
- `tiny_gltf_impl.cpp` - TinyGLTF library

---

### 3. Backends (`src/backends/`)

Backends implement the actual simulation algorithms. The library uses a plugin-style architecture where backends are loaded dynamically.

#### 3.1 Common Backend Framework (`src/backends/common/`)
Shared infrastructure for all backends.

| Class | Purpose |
|-------|---------|
| `SimEngine` - Base simulation engine |
| `ISimSystem` / `SimSystem` - Simulation subsystems |
| `SimSystemCollection` - System management |
| `SimAction` / `SimActionCollection` - Simulation actions |

**System Lifecycle:**
1. Registration - Systems register themselves
2. Build - Dependency resolution and validation
3. Init - Initialize from scene data
4. Advance - Time stepping loop
5. Sync/Retrieve - Data exchange with frontend

#### 3.2 CUDA Backend (`src/backends/cuda/`)
GPU-accelerated simulation backend using CUDA.

**Major Systems:**

| System | Purpose |
|--------|---------|
| `finite_element/` | FEM dynamics (tetrahedra, shells, rods) |
| `affine_body/` | Affine body dynamics |
| `contact_system/` | Contact detection and response |
| `collision_detection/` | Broad-phase collision detection |
| `linear_system/` | Linear solver (PCG) |
| `animator/` | Animation handling |
| `diff_sim/` | Differentiable simulation |

**FEM System Structure:**
```
finite_element/
├── finite_element_method.h       # Main FEM coordinator
├── fem_time_integrator.h         # Time integration
├── fem_linear_subsystem.h        # Linear system assembly
├── fem_line_search_reporter.h    # Line search for FEM
├── constitutions/                # FEM constitutions
│   ├── stable_neo_hookean_3d_function.h
│   ├── neo_hookean_shell_2d_function.h
│   ├── hookean_spring_1d_function.h
│   └── ...
└── ...
```

**Contact System:**
```
contact_system/
├── global_contact_manager.h      # Contact coordinator
├── contact_models/               # Contact formulations
│   ├── codim_ipc_simplex_normal_contact_function.h
│   ├── codim_ipc_simplex_frictional_contact_function.h
│   └── ipc_vertex_half_plane_contact_function.h
└── ...
```

**Collision Detection:**
```
collision_detection/
├── linear_bvh.h                  # LBVH broad phase
├── simplex_trajectory_filter.h   # Continuous collision detection
├── trajectory_filter.h           # Trajectory filtering
└── filters/                      # Specialized filters
```

#### 3.3 None Backend (`src/backends/none/`)
Null backend for testing and validation.

---

### 4. Python Bindings (`src/pybind/`)

Pybind11-based Python bindings exposing the C++ API as `pyuipc`.

**Structure mirrors C++ API:**
```
src/pybind/pyuipc/
├── module.cpp              # Main module definition
├── pyuipc.h                # Common binding utilities
├── exception.cpp           # Exception translation
├── common/                 # Common types (transform, unit, etc.)
├── builtin/                # Built-in attributes and types
├── geometry/               # Geometry bindings
├── core/                   # Core simulation bindings
├── constitution/           # Constitution bindings
├── diff_sim/               # Differentiable simulation
├── backend/                # Backend interface bindings
└── usd/                    # USD support bindings
```

**Example binding:**
```cpp
// From src/pybind/pyuipc/geometry/simplicial_complex.cpp
py::class_<SimplicialComplex, Geometry>(m, "SimplicialComplex")
    .def(py::init<>())
    .def("vertices", py::overload_cast<>(&SimplicialComplex::vertices))
    .def("positions", py::overload_cast<>(&SimplicialComplex::positions))
    // ...
```

---

### 5. Sanity Check System (`src/sanity_check/`)

Pre-simulation validation to catch common errors.

| Check | Description |
|-------|-------------|
| `simplicial_surface_intersection_check` | Self-intersection detection |
| `simplicial_surface_distance_check` | Minimum distance validation |
| `mesh_partition_check` | Mesh quality validation |
| `half_plane_vertex_distance_check` | Half-plane violation check |

---

### 6. Additional Modules

#### 6.1 USD Support (`src/usd/`)
Pixar's Universal Scene Description integration.

#### 6.2 OpenVDB Support (`src/vdb/`)
Volumetric data processing with OpenVDB.

---

## Data Flow

### Simulation Setup Flow
```
1. User creates Scene
   ├── Create Objects
   │   └── Add Geometries (SimplicialComplex)
   │       ├── Set vertex positions
   │       └── Apply constitutions
   ├── Configure ContactTabular
   │   ├── Create ContactElements
   │   └── Set friction/resistance
   └── Configure Animator (optional)

2. User creates Engine(backend_name)

3. User creates World(Engine)

4. World.init(Scene)
   └── Backend builds simulation systems
```

### Simulation Loop
```
while running:
    World.advance()  - Advance simulation one frame
        ├── Animator updates kinematic objects
        ├── Backend solves physics
        │   ├── Collision detection
        │   ├── Contact solving
        │   └── FEM/ABD dynamics
        └── State update
    
    World.sync()     - Sync frontend state to backend
    World.retrieve() - Retrieve results from backend
```

---

## Key Design Patterns

### 1. Attribute-Based Geometry
All geometry data is stored as named attributes, enabling:
- Flexible data layout
- Easy serialization
- Backend-specific data storage
- Runtime extensibility

### 2. Visitor Pattern
Backend visitors provide controlled access to frontend data:
- `SceneVisitor` - Read scene data
- `WorldVisitor` - Access world state
- `AnimatorVisitor` - Modify animated objects

### 3. Constitution Pattern
Materials are applied as "tags" to geometries:
- Frontend: Set `constitution_uid` attribute
- Backend: Query UID and apply appropriate physics

### 4. System-Based Backend
Modular backend architecture:
- Each physics aspect is a `SimSystem`
- Systems declare dependencies
- Automatic initialization order

---

## File Statistics

| Component | Headers (.h) | Implementation (.cpp) |
|-----------|-------------|----------------------|
| include/uipc/ | ~215 | - |
| src/core/ | ~50 | ~60 |
| src/constitution/ | - | ~35 |
| src/geometry/ | ~20 | ~35 |
| src/backends/cuda/ | ~150 | ~200+ |
| src/backends/common/ | ~10 | ~10 |
| src/io/ | ~5 | ~12 |
| src/pybind/ | ~50 | ~96 |
| src/sanity_check/ | ~8 | ~10 |

---

## Dependencies

- **Eigen** - Linear algebra
- **spdlog** - Logging
- **nlohmann/json** - JSON serialization
- **fmt** - String formatting
- **pybind11** - Python bindings
- **CUDA Toolkit** - GPU backend
- **OpenVDB** - Volumetric data (optional)
- **USD** - Scene description (optional)

---

## Build System

CMake-based build with presets in `CMakePresets.json`.

Key options:
- `UIPC_BUILD_CUDA_BACKEND` - Build CUDA backend
- `UIPC_BUILD_PYTHON_BINDINGS` - Build pyuipc
- `UIPC_BUILD_SANITY_CHECK` - Build sanity checker
- `UIPC_BUILD_TESTS` - Build unit tests
