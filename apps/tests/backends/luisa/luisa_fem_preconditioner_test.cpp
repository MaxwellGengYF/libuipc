/********************************************************************
 * @file luisa_fem_preconditioner_test.cpp
 * @brief T5.4b — the FEM diagonal preconditioner and the preconditioner-apply
 *        path, plus the per-vertex external-force constraint/reporter pair
 *        (LuisaCompute backend).
 *
 * Every oracle here is derived independently *in this file*: the reference backend
 * is CUDA-only and cannot be host-included, so nothing is copied from it.  The
 * oracles are
 *  (a) the *ported* T5.2 StableNeoHookean element routines
 *      (`snh_3d_detail::tet_hessian<double>`) evaluated on the host from the
 *      scene's own rest geometry and its *read-back* deformed state, summed in
 *      element order into a dense block-diagonal `D` (a different assembly order
 *      from the device's triplet fan-out + duplicate accumulation) and inverted
 *      with Eigen — never with the ported `lps::inverse`;
 *  (b) hand-derived scalar tuples (the per-vertex mass law
 *      `m = density * sum_e V_e / 4`, the vertex-`volume` law of the frontend's
 *      `compute_vertex_volume`, the diag/mass norm definitions);
 *  (c) the frontend's own laws for the external-force scene (`F/m` with the
 *      massless-vertex contract of T5.3) and a hand-written synthetic
 *      (vertex_id, force) table for the duplicate case;
 *  (d) the device-vs-host equivalence of the three factored routines
 *      (`fem_diag_preconditioner_detail::{diag_inv_row, diag_apply_row}`,
 *      `finite_element_external_vertex_force_detail::scatter_add_vec3`),
 *      asserted *bitwise* because the arithmetic is identical (T2.1/T5.1/T5.4a).
 *
 * ------------------------------- T5.4b --------------------------------------
 * Coverage map (case -> oracle):
 *  1) a 5-tet scene, the *real* FEMDiagPreconditioner registered into a standalone
 *     `GlobalLinearSystem::Impl`: init -> build_linear_system (extent, assemble,
 *     converter, _assemble_preconditioner) -> apply_preconditioner, against
 *     `z = D^{-1} r` with D built by Eigen from the ported element Hessians;
 *     plus the `converged != 0` early-out and the no-local-preconditioner
 *     `z <- r` fallback ................................. fem_precond_diag_apply
 *  2) `diag_norm()` / `mass_norm()` of the same scene against the hand-derived
 *     definitions (the AL pipeline's mu scaling) ....... fem_precond_diag_norm
 *  3) the constraint/reporter pair: (a) a synthetic table with *duplicates on one
 *     vertex* (the case that makes the vector atomic observable), (b) the
 *     frontend `uipc::constitution::FiniteElementExternalForce` scene through one
 *     `advance()` (`accs == F/m`, `0` when `m <= 0`), (c) the constraint's 0/0/0
 *     extents and the three uid-671 registries ... fem_precond_external_vertex_force
 *  4) every factored routine: host `double` vs device DSL, bitwise
 *     ................................................. fem_precond_device_vs_host
 *
 * Shared-binary rules (cookbook §7.14 rule 6, §7.18 rule 6): the systems
 * registered by the port are built for every scene of the whole test binary, so
 * the probe below only *reads* them; `find<T>()` returns a registered but
 * *invalidated* system, so "is it in play?" is asserted with `is_valid()`.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS FILE CANNOT REACH (stated so nothing is implied)
 * ---------------------------------------------------------------------------
 * `GlobalLinearSystem::solve()` is T8.3 and `SimEngine` is not wired, so the
 * assemble -> solve path is not exercised: `solve()`, `Impl::{spmv, spmv_dot}`,
 * `matrix_data_ptrs`, the debug dump, `distribute_solution`'s consumers and the
 * *global*-preconditioner branch of `apply_preconditioner` are all unreached
 * (the first three are declared-but-undefined; the last is T8.3's type).
 * Reached: `Impl::init`, `build_linear_system` (the whole extent -> assemble ->
 * converter -> `_assemble_preconditioner` chain), `apply_preconditioner`,
 * `diag_norm`, `mass_norm` and the shipped `FEMDiagPreconditioner` shaders, plus
 * the shipped `FiniteElementExternalVertexForce` shader on a real scene.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"
#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/elastic_moduli.h>
#include <uipc/constitution/finite_element_external_force.h>
#include <type_define.h>
#include <frame_stream.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/local_preconditioner.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/fem_diag_preconditioner.h>
#include <finite_element/fem_diag_preconditioner_function.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_animator.h>
#include <finite_element/finite_element_external_force_manager.h>
#include <finite_element/finite_element_external_vertex_force.h>
#include <finite_element/constraints/finite_element_external_vertex_force_constraint.h>
#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>
#include <finite_element/fem_utils.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;
namespace gg    = uipc::backend::luisa;
namespace lud   = uipc::backend::luisa::luisa_device;
namespace lps   = uipc::backend::luisa::luisa_device::math;
namespace lla   = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt  = ::luisa::compute;
namespace lfdi  = uipc::backend::luisa::fem_diag_preconditioner_detail;
namespace lfvfd = uipc::backend::luisa::finite_element_external_vertex_force_detail;

// The ported surface lives in `uipc::backend::luisa` (like the reference's does in
// `uipc::backend::cuda`), so a test body sees the same spellings a ported system does.
using namespace uipc::backend::luisa;

namespace t54b
{
// ---------------------------------------------------------------------------
//  small helpers (the T5.1/T5.3/T5.4a test files' idioms)
// ---------------------------------------------------------------------------
[[nodiscard]] inline double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// Read an LC buffer back to the host (frame stream first: a command may still be
/// recorded in an open frame).
template <typename T>
std::vector<T> read_device(const lud::LCBufferView<T>& view)
{
    std::vector<T> host(view.size());
    gg::frame_stream().synchronize();
    if(!host.empty())
    {
        view.copy_to(span<T>{host});
        lud::main_stream().synchronize();
    }
    return host;
}

[[nodiscard]] std::vector<Vector3> read_vec3(const lud::LCBufferView<gg::LVector3>& view)
{
    auto                 raw = read_device(view);
    std::vector<Vector3> host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

using HostTriplet  = lla::LTripletMatrixHostMemoryView<double, 3, 3>;
using HostDense    = lla::LDenseVectorHostMemoryView<double>;
using HostDiagInv  = lfdi::DiagInvHostMemoryView;
using TripletDeviceView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using DenseDeviceView   = lla::LDenseVectorDeviceView<lcrt::Var<double>>;

[[nodiscard]] inline lps::LVec<double, 3> vec3_at(const std::vector<Vector3>& v, int i)
{
    return lps::LVec<double, 3>{v[i](0), v[i](1), v[i](2)};
}

// ---------------------------------------------------------------------------
//  the five-tet mesh (written down here: connectivity + rest/deformed positions)
// ---------------------------------------------------------------------------
struct FiveTet
{
    std::vector<Vector3>  X;  // rest positions
    std::vector<Vector3>  x;  // the deformation the assembly evaluates the Hessians at
    std::vector<Vector4i> tets;
    double                dt = 1.0e-2;
};
/// 8 vertices, 5 tets.  `tets[0]` is a *non-ascending* (even) permutation of the
/// unit tet, which exercises the reference's `upper_LR` swap through the assembly;
/// every tet has a positive rest determinant -- the frontend's
/// `compute_vertex_volume` (reached by `StableNeoHookean::apply_to`) rejects an
/// inverted one (measured: the T5.4a five-tet connectivity has two negative tets,
/// so it cannot be used for a *scene*).
[[nodiscard]] inline FiveTet five_tet()
{
    FiveTet m;
    m.X = {{0.0, 0.0, 0.0},
           {1.0, 0.0, 0.0},
           {0.0, 1.0, 0.0},
           {0.0, 0.0, 1.0},
           {1.0, 1.0, 1.0},
           {2.0, 0.5, 0.25},
           {-0.25, 0.5, 1.5},
           {0.5, -0.25, -0.5}};
    m.tets = {{1, 0, 3, 2}, {0, 1, 2, 4}, {0, 3, 2, 6}, {1, 2, 4, 5}, {2, 4, 6, 7}};
    m.x    = {{0.05, -0.02, 0.01},
              {1.10, 0.05, -0.03},
              {-0.04, 1.05, 0.02},
              {0.03, 0.01, 0.95},
              {1.05, 1.10, 0.98},
              {2.05, 0.48, 0.29},
              {-0.28, 0.55, 1.45},
              {0.52, -0.20, -0.55}};
    return m;
}

[[nodiscard]] inline lps::LMat<double, 3, 3> element_Dm_inv(const FiveTet& m, SizeT e)
{
    const auto& t = m.tets[e];
    return fem::Dm_inv<double>(vec3_at(m.X, t(0)),
                               vec3_at(m.X, t(1)),
                               vec3_at(m.X, t(2)),
                               vec3_at(m.X, t(3)));
}

[[nodiscard]] inline double element_volume(const FiveTet& m, SizeT e)
{
    const auto& t  = m.tets[e];
    auto        Ds = fem::Ds<double>(vec3_at(m.X, t(0)),
                              vec3_at(m.X, t(1)),
                              vec3_at(m.X, t(2)),
                              vec3_at(m.X, t(3)));
    return lps::determinant(Ds) / 6.0;
}

// ---------------------------------------------------------------------------
//  the scene: one body, 5 tets, StableNeoHookean (uid 10).
//  NOTE: StableNeoHookean, *not* ARAP (uid 9): the shared test binary registers
//  T5.1's mock dim-3 constitution with ARAP's uid, so in a full-suite run ARAP
//  owns no geometry (T5.1 rule 12 / T5.4a's layout case measured it).
// ---------------------------------------------------------------------------
struct SceneDesc
{
    FiveTet              mesh;
    std::vector<IndexT>  is_fixed;
    std::vector<IndexT>  is_constrained;
    std::vector<Vector3> external_forces;
    std::vector<Float>   densities;
    bool                 with_external_force_constitution = false;
    std::string          integrator_type = "bdf1";
    Float                dt              = 1.0e-2;
    Float                density         = 1000.0;
    ElasticModuli        moduli          = ElasticModuli::youngs_poisson(120.0e3, 0.49);
};

[[nodiscard]] inline SceneDesc five_tet_desc()
{
    SceneDesc d;
    d.mesh     = five_tet();
    d.dt       = static_cast<Float>(d.mesh.dt);
    d.is_fixed.assign(d.mesh.X.size(), 0);
    d.is_fixed[0] = 1;  // one fixed vertex (its diagonal block must be Identity)
    d.is_constrained.assign(d.mesh.X.size(), 0);
    d.external_forces.assign(d.mesh.X.size(), Vector3::Zero());
    d.densities.assign(d.mesh.X.size(), d.density);
    return d;
}

Scene build_scene(const SceneDesc& desc)
{
    auto config = Scene::default_config();
    config["contact"]["enable"]      = false;
    config["sanity_check"]["enable"] = false;
    config["integrator"]["type"]     = desc.integrator_type;
    config["dt"]                     = desc.dt;
    config["gravity"]                = Vector3{0.0, -9.8, 0.0};

    Scene scene{config};
    auto  object = scene.objects().create("t54b");

    StableNeoHookean snh;
    auto mesh = tetmesh(span<const Vector3>{desc.mesh.X},
                        span<const Vector4i>{desc.mesh.tets});
    label_surface(mesh);
    snh.apply_to(mesh, desc.moduli, desc.density);

    {
        auto is_fixed = mesh.vertices().find<IndexT>(builtin::is_fixed);
        if(!is_fixed)
            is_fixed = mesh.vertices().create<IndexT>(builtin::is_fixed, 0);
        std::ranges::copy(desc.is_fixed, view(*is_fixed).begin());
    }
    {
        // per-vertex mass density: a 0 makes the vertex massless (the method
        // computes m = density * volume), which the external-force manager must
        // map to a zero acceleration
        auto density = mesh.vertices().find<Float>(builtin::mass_density);
        if(!density)
            density = mesh.vertices().create<Float>(builtin::mass_density, desc.density);
        std::ranges::copy(desc.densities, view(*density).begin());
    }

    if(desc.with_external_force_constitution)
    {
        // `uipc::constitution::FiniteElementExternalForce::apply_to` creates
        // `is_constrained` (default 0) when it is missing and *fills*
        // `external_force` uniformly, so create `is_constrained` first and
        // overwrite `external_force` per vertex afterwards.
        auto is_constrained = mesh.vertices().find<IndexT>(builtin::is_constrained);
        if(!is_constrained)
            is_constrained = mesh.vertices().create<IndexT>(builtin::is_constrained, 0);
        std::ranges::copy(desc.is_constrained, view(*is_constrained).begin());

        FiniteElementExternalForce ext_force;
        ext_force.apply_to(mesh, Vector3{0.0, 0.0, 0.0});

        auto force =
            mesh.vertices().find<Vector3>("external_force");  // created by apply_to
        REQUIRE(force != nullptr);
        std::ranges::copy(desc.external_forces, view(*force).begin());
        // re-assert the marker (apply_to kept the existing attribute untouched)
        std::ranges::copy(desc.is_constrained, view(*is_constrained).begin());
    }

    object->geometries().create(mesh);
    return scene;
}

void step_frame(World& world)
{
    world.advance();
    world.sync();
    gg::frame_stream().synchronize();
}

/// Deform the scene's device state through the public accessor feature (T5.3
/// rule 2), so the *constitution*'s Hessian is evaluated off the rest state.
void deform_state(World& world, const std::vector<Vector3>& xs)
{
    auto feature = world.features().find<FiniteElementStateAccessorFeature>();
    REQUIRE(feature != nullptr);
    auto state_geo = feature->create_geometry();
    auto pos = state_geo.vertices().create<Vector3>(builtin::position, Vector3::Zero());
    state_geo.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());
    feature->copy_to(state_geo);
    auto pos_view = view(*pos);
    REQUIRE(pos_view.size() == xs.size());
    for(SizeT i = 0; i < xs.size(); ++i)
        pos_view[i] = xs[i];
    feature->copy_from(state_geo);
    gg::frame_stream().synchronize();
}

// ---------------------------------------------------------------------------
//  the oracle: the assembled block-diagonal `D` (3x3 per vertex)
// ---------------------------------------------------------------------------
struct DiagOracle
{
    /// per-vertex diagonal block of the *assembled* A:
    ///   non-fixed: `m I + sum_e H_e[a][a]`
    ///   fixed    : `(1 + #incident tets) * Identity`  (see `fixed_identity_multiplier`)
    std::vector<Eigen::Matrix3d> D;
    /// hand-derived per-vertex mass `density * sum_e V_e / 4`
    std::vector<double>          masses;
    /// the multiplier the reference's fixed-row substitution produces (1 kinetic
    /// diagonal triplet + one elastic triplet per incident tet, each replaced by
    /// `Identity` and then summed by the converter's duplicate accumulation)
    double                       fixed_identity_multiplier = 1.0;
    /// the frontend's SNH reparameterisation (`StableNeoHookean::apply_to`):
    /// snh_mu = 4 mu / 3, snh_lambda = lambda + 5 mu / 6
    double                       snh_mu     = 0.0;
    double                       snh_lambda = 0.0;
};

/// Build the oracle from the *read-back* deformed state `xs` and the scene's own
/// rest geometry.  The element Hessians come from the ported T5.2 routine, summed
/// in element order (a different order from the device's triplet fan-out), and the
/// inversion/applications below use Eigen, never the ported numerics layer.
[[nodiscard]] DiagOracle build_diag_oracle(const SceneDesc&          desc,
                                           const std::vector<Vector3>& xs,
                                           const std::vector<IndexT>&  is_fixed)
{
    const auto& m     = desc.mesh;
    const auto  nv    = m.X.size();
    DiagOracle  o;
    o.snh_mu     = 4.0 * desc.moduli.mu() / 3.0;
    o.snh_lambda = desc.moduli.lambda() + 5.0 * desc.moduli.mu() / 6.0;

    // hand-derived vertex volumes (the frontend's `compute_vertex_volume` law:
    // sum of the incident tet volumes / 4) and masses m = density * volume
    std::vector<double> vertex_volume(nv, 0.0);
    for(SizeT e = 0; e < m.tets.size(); ++e)
    {
        const auto V = element_volume(m, e);
        for(int i = 0; i < 4; ++i)
            vertex_volume[static_cast<SizeT>(m.tets[e](i))] += V / 4.0;
    }
    o.masses.resize(nv, 0.0);
    for(SizeT v = 0; v < nv; ++v)
        o.masses[v] = static_cast<double>(desc.densities[v]) * vertex_volume[v];

    o.D.assign(nv, Eigen::Matrix3d::Zero());
    for(SizeT v = 0; v < nv; ++v)
    {
        if(is_fixed[v] != 0)
        {
            // The reference's `fix_hessian_row` replaces the block of *every*
            // triplet whose (row, col) touches a fixed vertex with `Identity()`,
            // and the triplet -> BCOO conversion accumulates duplicates, so a fixed
            // vertex's diagonal block ends up as
            //   (1 kinetic diagonal triplet + #incident tets) * Identity.
            SizeT n_incident = 0;
            for(SizeT e = 0; e < m.tets.size(); ++e)
                for(int i = 0; i < 4; ++i)
                    if(static_cast<SizeT>(m.tets[e](i)) == v)
                        ++n_incident;
            o.fixed_identity_multiplier = static_cast<double>(1 + n_incident);
            o.D[v] = o.fixed_identity_multiplier * Eigen::Matrix3d::Identity();
            continue;
        }
        // the kinetic Hessian is `m I` (unconditional), the elastics add their
        // diagonal 3x3 sub-blocks
        o.D[v] = o.masses[v] * Eigen::Matrix3d::Identity();
    }

    for(SizeT e = 0; e < m.tets.size(); ++e)
    {
        const auto& t = m.tets[e];
        const auto  H = snh_3d_detail::tet_hessian<double>(o.snh_mu,
                                                          o.snh_lambda,
                                                          static_cast<double>(desc.dt),
                                                          element_volume(m, e),
                                                          vec3_at(xs, t(0)),
                                                          vec3_at(xs, t(1)),
                                                          vec3_at(xs, t(2)),
                                                          vec3_at(xs, t(3)),
                                                          element_Dm_inv(m, e));
        for(int i = 0; i < 4; ++i)
        {
            const auto v = static_cast<SizeT>(t(i));
            if(is_fixed[v] != 0)
                continue;
            for(int r = 0; r < 3; ++r)
                for(int c = 0; c < 3; ++c)
                    o.D[v](r, c) += H(3 * i + r, 3 * i + c);
        }
    }
    return o;
}

/// Read the converted `bcoo_A`'s (row, col, block) triples back to the host and
/// return the per-block-index diagonal blocks of a `dof_block_count` system.
[[nodiscard]] std::vector<Eigen::Matrix3d> assembled_diagonal_blocks(
    const lla::LBCOOMatrix<double, 3, 3>& bcoo, SizeT dof_block_count)
{
    const auto nnz = static_cast<SizeT>(bcoo.triplet_count());
    std::vector<std::int32_t> rows(nnz), cols(nnz);
    std::vector<double>       vals(nnz * 9u);
    gg::frame_stream().synchronize();  // the assembly may still be in flight
    if(nnz != 0u)
    {
        bcoo.copy_to_row_indices(span<std::int32_t>{rows});
        bcoo.copy_to_col_indices(span<std::int32_t>{cols});
        bcoo.copy_to_values(span<double>{vals});
        lud::main_stream().synchronize();
    }
    std::vector<Eigen::Matrix3d> out(dof_block_count, Eigen::Matrix3d::Zero());
    for(SizeT k = 0; k < nnz; ++k)
    {
        if(rows[k] != cols[k])
            continue;
        const auto i = static_cast<SizeT>(rows[k]);
        if(i >= dof_block_count)
            continue;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                out[i](r, c) = vals[k * 9u + static_cast<SizeT>(c * 3 + r)];
    }
    return out;
}

// ---------------------------------------------------------------------------
//  the probe: grabs the port's public interfaces once they are built
// ---------------------------------------------------------------------------
class T54bProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::FiniteElementMethod*                    fem            = nullptr;
    inline static gg::GlobalLinearSystem*                     gls            = nullptr;
    inline static gg::FEMLinearSubsystem*                     femsub         = nullptr;
    inline static gg::FEMDiagPreconditioner*                  precond        = nullptr;
    inline static gg::FEMExternalForceManager*                ext_force      = nullptr;
    inline static gg::FiniteElementExternalVertexForce*       ext_vertex     = nullptr;
    inline static gg::FiniteElementExternalVertexForceConstraint* ext_vertex_constraint = nullptr;

    static void reset()
    {
        fem = nullptr;
        gls = nullptr;
        femsub = nullptr;
        precond = nullptr;
        ext_force = nullptr;
        ext_vertex = nullptr;
        ext_vertex_constraint = nullptr;
    }

  private:
    void do_build() override
    {
        fem      = &require<gg::FiniteElementMethod>();
        gls      = &require<gg::GlobalLinearSystem>();
        femsub   = &require<gg::FEMLinearSubsystem>();
        precond  = find<gg::FEMDiagPreconditioner>();
        ext_force = find<gg::FEMExternalForceManager>();
        ext_vertex = find<gg::FiniteElementExternalVertexForce>();
        ext_vertex_constraint = find<gg::FiniteElementExternalVertexForceConstraint>();
    }
};
}  // namespace t54b

REGISTER_SIM_SYSTEM(t54b::T54bProbe);

namespace t54b
{
// ---------------------------------------------------------------------------
//  the test-local device kernels (case 4): the *same* factored routines the
//  shipped shaders call, so host and device run one implementation
// ---------------------------------------------------------------------------
/// `fem_diag_preconditioner_detail::diag_inv_row` on the device: writes the
/// returned index, the two guard flags and `store_mat3x3(inverse)`.
auto& t54b_diag_inv_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // rows
                                   lcrt::Buffer<std::int32_t>,  // cols
                                   lcrt::Buffer<double>,        // flat block values
                                   lcrt::Buffer<LMatrix3x3>,    // out: inverse blocks
                                   lcrt::Buffer<std::int32_t>,  // out: in_window flags
                                   lcrt::Buffer<std::int32_t>,  // out: is_diagonal flags
                                   lcrt::Buffer<std::int32_t>,  // out: local block index
                                   std::int32_t,                // fem segment offset
                                   std::int32_t,                // fem segment count
                                   uint32_t>(                   // triplet count
            [](lcrt::BufferVar<std::int32_t> rows,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       vals,
               lcrt::BufferVar<LMatrix3x3>   inv_out,
               lcrt::BufferVar<std::int32_t> in_window_out,
               lcrt::BufferVar<std::int32_t> is_diagonal_out,
               lcrt::BufferVar<std::int32_t> index_out,
               lcrt::Var<std::int32_t>       seg_offset,
               lcrt::Var<std::int32_t>       seg_count,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto A = TripletDeviceView::from_handles(rows, cols, vals, n, n, n);
                    auto row = lfdi::diag_inv_row(
                        A, lcrt::cast<std::int32_t>(i), seg_offset, seg_count);

                    lcrt::Var<LMatrix3x3> out;
                    store_mat3x3(out, row.value);
                    inv_out->write(i, out);
                    // two separate flags (an `$if`-gated write of 1; the buffers are
                    // zeroed on the host) instead of a `select` bit field
                    $if(row.in_window)
                    {
                        in_window_out->write(i, std::int32_t{1});
                    };
                    $if(row.is_diagonal)
                    {
                        is_diagonal_out->write(i, std::int32_t{1});
                    };
                    index_out->write(i, row.index);
                };
            });
    return k;
}

/// `fem_diag_preconditioner_detail::diag_apply_row` on the device.
auto& t54b_diag_apply_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,        // r (flat)
                                                lcrt::Buffer<double>,        // z (flat)
                                                lcrt::Buffer<LMatrix3x3>,    // inverse blocks
                                                uint32_t>(                   // block count
        [](lcrt::BufferVar<double>       r,
           lcrt::BufferVar<double>       z,
           lcrt::BufferVar<LMatrix3x3>   diag_inv,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto R = DenseDeviceView::from_handles(r, n * 3u);
                auto Z = DenseDeviceView::from_handles(z, n * 3u);
                auto D = lfdi::DiagInvDeviceView::from_handles(diag_inv, n);
                lfdi::diag_apply_row(Z, R, D, lcrt::cast<std::int32_t>(i));
            };
        });
    return k;
}

/// `finite_element_external_vertex_force_detail::scatter_add_vec3` on the device,
/// over the *scalar alias* of an AoS `LVector3` force buffer.
auto& t54b_scatter_add_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,        // forces (scalar alias)
                                                lcrt::Buffer<std::int32_t>,  // vertex ids
                                                lcrt::Buffer<LVector3>,      // forces (AoS)
                                                uint32_t>(                   // n
        [](lcrt::BufferVar<double>       forces_flat,
           lcrt::BufferVar<std::int32_t> vertex_ids,
           lcrt::BufferVar<LVector3>     vertex_forces,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto vid = vertex_ids->read(i);
                auto f   = load_vec3(vertex_forces->read(i));
                auto forces = DenseDeviceView::from_handles(forces_flat, n * 3u);
                lfvfd::scatter_add_vec3(forces, vid, f);
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  the shared round data of case 4
// ---------------------------------------------------------------------------
constexpr int kRoundTriplets = 5;

struct RoundData
{
    // 5 triplets of a 5-block matrix; row 3 is out-of-window (>= seg_count = 4)
    // when the round uses seg_offset = 0 / seg_count = 4
    std::vector<std::int32_t> rows = {0, 1, 2, 3, 4};
    std::vector<std::int32_t> cols = {0, 2, 2, 3, 4};
    // per-triplet 3x3 block, column-major (the flat storage layout)
    std::vector<double> blocks;
    // per-entry r values
    std::vector<double> r;

    RoundData()
    {
        blocks.resize(static_cast<SizeT>(kRoundTriplets) * 9u);
        for(int k = 0; k < kRoundTriplets; ++k)
        {
            for(int c = 0; c < 3; ++c)
            {
                for(int rr = 0; rr < 3; ++rr)
                {
                    // diagonally dominant, non-symmetric, exactly representable
                    const double v = (rr == c) ? (2.0 + k) : (0.25 * (rr + 1) - 0.5 * c + k);
                    blocks[static_cast<SizeT>(k * 9 + c * 3 + rr)] = v;
                }
            }
        }
        r = {1.0, -2.0, 3.0, 0.5, 1.5, -1.25, 2.25, -3.5, 0.75, 1.0, 2.0, 3.0, -1.0, 0.5, 2.5};
    }
};

struct RoundOutput
{
    std::vector<Eigen::Matrix3d> inv;           // per triplet
    std::vector<std::int32_t>    in_window;     // per triplet
    std::vector<std::int32_t>    is_diagonal;   // per triplet
    std::vector<std::int32_t>    index;         // per triplet
    std::vector<double>          z;             // 3 per block
    std::vector<double>          forces;        // 3 per vertex
};

/// The host side of the round: the same routines over `storage::memory` views.
[[nodiscard]] RoundOutput round_host(const RoundData& d,
                                     std::int32_t     seg_offset,
                                     std::int32_t     seg_count)
{
    RoundOutput out;
    out.inv.assign(kRoundTriplets, Eigen::Matrix3d::Zero());
    out.in_window.assign(kRoundTriplets, 0);
    out.is_diagonal.assign(kRoundTriplets, 0);
    out.index.assign(kRoundTriplets, 0);

    std::vector<std::int32_t> rows = d.rows;
    std::vector<std::int32_t> cols = d.cols;
    std::vector<double>       vals = d.blocks;
    auto A = HostTriplet::from_handles(rows.data(),
                                       cols.data(),
                                       vals.data(),
                                       static_cast<std::int32_t>(kRoundTriplets),
                                       static_cast<std::int32_t>(kRoundTriplets),
                                       static_cast<std::int32_t>(kRoundTriplets));

    std::vector<LMatrix3x3> diag_inv(kRoundTriplets);
    for(int k = 0; k < kRoundTriplets; ++k)
    {
        auto row = lfdi::diag_inv_row(
            A, static_cast<std::int32_t>(k), seg_offset, seg_count);
        out.in_window[k]   = row.in_window ? 1 : 0;
        out.is_diagonal[k] = row.is_diagonal ? 1 : 0;
        out.index[k]       = row.index;
        for(int c = 0; c < 3; ++c)
            for(int rr = 0; rr < 3; ++rr)
                diag_inv[static_cast<SizeT>(k)](rr, c) = row.value(rr, c);
    }

    // diag_apply_row over memory views
    std::vector<double> r = d.r;
    std::vector<double> z(r.size(), 0.0);
    auto R = HostDense::from_handles(r.data(), static_cast<std::int32_t>(r.size()));
    auto Z = HostDense::from_handles(z.data(), static_cast<std::int32_t>(z.size()));
    auto D = HostDiagInv::from_handles(diag_inv.data(),
                                       static_cast<std::int32_t>(diag_inv.size()));
    for(int k = 0; k < kRoundTriplets; ++k)
        lfdi::diag_apply_row(Z, R, D, static_cast<std::int32_t>(k));
    out.z = z;

    for(int k = 0; k < kRoundTriplets; ++k)
        for(int c = 0; c < 3; ++c)
            for(int rr = 0; rr < 3; ++rr)
                out.inv[static_cast<SizeT>(k)](rr, c) = diag_inv[static_cast<SizeT>(k)](rr, c);

    // scatter_add_vec3 over memory views: a synthetic table *with duplicates*
    std::vector<std::int32_t> ids      = {0, 1, 0, 2, 0};
    std::vector<double>       fvals    = {1.0, 2.0, 3.0,   //
                                          4.0, 5.0, 6.0,   //
                                          7.0, 8.0, 9.0,   //
                                          10.0, 11.0, 12.0,  //
                                          -1.0, -2.0, -3.0};
    std::vector<double>       forces(3 * 3, 0.0);
    auto F = HostDense::from_handles(forces.data(), static_cast<std::int32_t>(forces.size()));
    for(int k = 0; k < 5; ++k)
    {
        lps::LVec<double, 3> f{fvals[static_cast<SizeT>(k * 3)],
                               fvals[static_cast<SizeT>(k * 3 + 1)],
                               fvals[static_cast<SizeT>(k * 3 + 2)]};
        lfvfd::scatter_add_vec3(F, ids[static_cast<SizeT>(k)], f);
    }
    out.forces = forces;
    return out;
}
}  // namespace t54b

// ===========================================================================
//  CASE 1 — the headline: the real FEMDiagPreconditioner through a standalone
//  GlobalLinearSystem::Impl
// ===========================================================================
TEST_CASE("fem_precond_diag_apply", "[luisa][fem_precond]")
{
    using namespace t54b;
    test_runtime();

    auto desc = five_tet_desc();
    T54bProbe::reset();

    auto   scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t54b_apply")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T54bProbe::fem != nullptr);
    REQUIRE(T54bProbe::gls != nullptr);
    REQUIRE(T54bProbe::femsub != nullptr);
    REQUIRE(T54bProbe::femsub->is_valid());
    // the *real* preconditioner is built in this scene (a FEM scene always has one)
    REQUIRE(T54bProbe::precond != nullptr);
    CHECK(T54bProbe::precond->is_valid());

    // frame > 0 is required by the assembly's set_dof_info
    step_frame(world);
    // deform the state so the *elastic* Hessian is off the rest state (at F == I a
    // hyperelastic Hessian is the make_spd projection of a non-zero tensor, but the
    // elastic *gradient* vanishes -- deforming makes the numbers unambiguous)
    deform_state(world, desc.mesh.x);

    auto xs = read_vec3(T54bProbe::fem->xs());
    auto isf = read_device(T54bProbe::fem->is_fixed());
    auto masses_device = read_device(T54bProbe::fem->masses());
    REQUIRE(xs.size() == 8u);

    const auto oracle = build_diag_oracle(desc, xs, isf);

    // the hand-derived mass law against the method's own device masses (an
    // independent cross-check of the oracle's input, not of the code under test)
    double worst_mass = 0.0;
    for(SizeT v = 0; v < 8u; ++v)
        worst_mass = std::max(worst_mass, rel_err(oracle.masses[v], masses_device[v]));
    std::printf("[t54b] apply: hand-derived masses vs device masses worst rel %.3e\n",
                worst_mass);
    CHECK(worst_mass <= 1e-12);

    // ---- the standalone Impl: the real subsystem + the real preconditioner ----
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T54bProbe::femsub);
    impl.local_preconditioners.register_sim_system(*T54bProbe::precond);
    impl.init();

    CHECK(impl.diag_dof_offsets_counts.total_count() == 24u);
    CHECK(impl.subsystem_infos.size() == 1u);
    CHECK(impl.subsystem_infos[0].is_diag);
    // the preconditioner covers the only diag subsystem
    CHECK(impl.no_precond_diag_subsystem_indices.empty());

    // the whole main path: extent -> assemble -> convert (ge2sym + triplet->BCOO)
    // -> _assemble_preconditioner.  T5.4a could not run it (it needs the solver's
    // assembly trigger); T5.4b reaches it through this Impl.
    impl.build_linear_system();
    CHECK(impl.empty_system == false);
    CHECK(impl.triplet_A.triplet_count() > 0);
    CHECK(impl.bcoo_A.triplet_count() > 0);
    CHECK(static_cast<SizeT>(T54bProbe::precond->diag_inv.size()) == 8u);

    // ---- the assembled diagonal blocks against the oracle ----
    const auto diag_assembled = assembled_diagonal_blocks(impl.bcoo_A, 8u);
    double     worst_assembled = 0.0;
    for(SizeT v = 0; v < 8u; ++v)
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                worst_assembled = std::max(worst_assembled,
                                           rel_err(diag_assembled[v](r, c), oracle.D[v](r, c)));
    double dmax = 0.0;
    for(SizeT v = 0; v < 8u; ++v)
        dmax = std::max(dmax, oracle.D[v].cwiseAbs().maxCoeff());
    std::printf("[t54b] apply: assembled diag blocks worst rel %.3e (max |D| %.3e, nnz %d)\n",
                worst_assembled,
                dmax,
                static_cast<int>(impl.bcoo_A.triplet_count()));
    CHECK(dmax > 1.0);  // non-vacuous
    CHECK(worst_assembled <= 1e-12);

    // THE REFERENCE'S FIXED-ROW QUIRK (measured, not assumed): `fix_hessian_row`
    // writes `Identity` into *every* triplet whose (row, col) block touches a fixed
    // vertex, and the triplet -> BCOO conversion accumulates duplicates (a sort +
    // a segmented sum), so a fixed vertex's diagonal block is
    // `(1 + #incident tets) * Identity` rather than `Identity`.
    // Vertex 0 is in 3 tets (1 kinetic diagonal triplet + 3 elastic ones -> 4
    // Identity blocks summed).  The CUDA reference has the same structure (its
    // `MatrixConverter::convert` is the same sort + `FastSegmentalReduce`), so the
    // port reproduces its numbers; the oracle above models exactly this.
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            CHECK(diag_assembled[0](r, c)
                  == (r == c ? static_cast<double>(oracle.fixed_identity_multiplier)
                             : 0.0));

    // ---- the preconditioner-apply oracle: z = D^{-1} r ----
    // a per-vertex-distinct r, so a wrong block (an off-diagonal one, or a
    // neighbouring vertex's) cannot pass
    std::vector<double> r(24, 0.0);
    for(SizeT v = 0; v < 8u; ++v)
        for(int c = 0; c < 3; ++c)
            r[3 * v + static_cast<SizeT>(c)] =
                0.5 * static_cast<double>(v + 1) + 0.25 * static_cast<double>(c) - 1.0;

    std::vector<double> z_expected(24, 0.0);
    for(SizeT v = 0; v < 8u; ++v)
    {
        Eigen::Vector3d rv(r[3 * v], r[3 * v + 1], r[3 * v + 2]);
        const Eigen::Vector3d zv = oracle.D[v].inverse() * rv;
        for(int c = 0; c < 3; ++c)
            z_expected[3 * v + static_cast<SizeT>(c)] = zv(c);
    }

    lla::LDenseVector<double> r_dev, z_dev;
    r_dev.resize(24);
    z_dev.resize(24);
    r_dev.copy_from(span<const double>{r});
    std::vector<double> sentinel(24, -12345.0);
    z_dev.copy_from(span<const double>{sentinel});

    lud::LDeviceVar<IndexT> converged;
    converged.copy_from(IndexT{0});

    gg::frame_stream().synchronize();
    impl.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    gg::frame_stream().synchronize();

    std::vector<double> z(24, 0.0);
    z_dev.copy_to(span<double>{z});

    double worst = 0.0, zmax = 0.0, worst_ident = 0.0;
    for(SizeT i = 0; i < 24u; ++i)
    {
        worst = std::max(worst, rel_err(z[i], z_expected[i]));
        zmax  = std::max(zmax, std::abs(z_expected[i]));
        // the "wrong block" sensitivity: the off-diagonal blocks of the same row
        // would give a different vector; quantify how different
        worst_ident = std::max(worst_ident, std::abs(r[i] - z_expected[i]));
    }
    std::printf("[t54b] apply: z vs D^{-1} r worst rel %.3e (max |z| %.3e, |r - D^-1 r|max %.3e)\n",
                worst,
                zmax,
                worst_ident);
    CHECK(zmax > 1.0e-2);
    CHECK(worst_ident > 1.0e-2);  // the preconditioner is not the identity -> non-vacuous
    CHECK(worst <= 1e-12);

    // a wrong *block* is not a rounding-level error: verify the assertion above can
    // see it by evaluating the oracle with each vertex's block replaced by its
    // neighbour's
    double worst_wrong = 0.0;
    for(SizeT v = 0; v < 8u; ++v)
    {
        const auto      wrong  = oracle.D[(v + 1u) % 8u];
        Eigen::Vector3d rv(r[3 * v], r[3 * v + 1], r[3 * v + 2]);
        const auto      zwrong = wrong.inverse() * rv;
        for(int c = 0; c < 3; ++c)
            worst_wrong = std::max(worst_wrong,
                                   std::abs(zwrong(c) - z_expected[3 * v + static_cast<SizeT>(c)]));
    }
    std::printf("[t54b] apply: neighbour-block divergence %.3e (a wrong block cannot hide)\n",
                worst_wrong);
    CHECK(worst_wrong > 1.0e-3);

    // ---- the `converged != 0` early-out ----
    z_dev.copy_from(span<const double>{sentinel});
    converged.copy_from(IndexT{1});
    gg::frame_stream().synchronize();
    impl.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    gg::frame_stream().synchronize();
    z_dev.copy_to(span<double>{z});
    int n_early_out = 0;
    for(SizeT i = 0; i < 24u; ++i)
        if(z[i] == sentinel[i])
            ++n_early_out;
    std::printf("[t54b] apply: converged != 0 early-out kept %d/24 entries unchanged\n",
                n_early_out);
    CHECK(n_early_out == 24);

    // ---- the no-local-preconditioner fallback (z <- r) ----
    gg::GlobalLinearSystem::Impl impl_fallback;
    impl_fallback.diag_subsystems.register_sim_system(*T54bProbe::femsub);
    impl_fallback.init();
    REQUIRE(impl_fallback.no_precond_diag_subsystem_indices.size() == 1u);
    CHECK(impl_fallback.no_precond_diag_subsystem_indices[0] == 0);

    z_dev.copy_from(span<const double>{sentinel});
    converged.copy_from(IndexT{0});
    gg::frame_stream().synchronize();
    impl_fallback.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    gg::frame_stream().synchronize();
    z_dev.copy_to(span<double>{z});
    int n_copied = 0;
    for(SizeT i = 0; i < 24u; ++i)
        if(z[i] == r[i])
            ++n_copied;
    std::printf("[t54b] apply: fallback copied r to z for %d/24 entries\n", n_copied);
    CHECK(n_copied == 24);
}

// ===========================================================================
//  CASE 2 — diag_norm() / mass_norm()
// ===========================================================================
TEST_CASE("fem_precond_diag_norm", "[luisa][fem_precond]")
{
    using namespace t54b;
    test_runtime();

    auto desc = five_tet_desc();
    T54bProbe::reset();

    auto   scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t54b_norm")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T54bProbe::femsub != nullptr);
    REQUIRE(T54bProbe::femsub->is_valid());

    step_frame(world);
    deform_state(world, desc.mesh.x);

    auto xs  = read_vec3(T54bProbe::fem->xs());
    auto isf = read_device(T54bProbe::fem->is_fixed());
    const auto oracle = build_diag_oracle(desc, xs, isf);

    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T54bProbe::femsub);
    impl.local_preconditioners.register_sim_system(*T54bProbe::precond);
    impl.init();
    impl.build_linear_system();
    REQUIRE(impl.empty_system == false);

    // the reference's definitions (src/backends/cuda/finite_element/fem_linear_subsystem.cu):
    //   diag_norm_kernel:  diag_blocks_norm(i) = is_fixed(i) ? 0
    //                        : max(|H_ii(0,0)|, |H_ii(1,1)|, |H_ii(2,2)|)
    //   mass_norm_kernel:  diag_blocks_norm(i) = is_fixed(i) ? 0 : mass(i)
    // and `GlobalLinearSystem::Impl::{diag_norm,mass_norm}` take the max over the
    // diag subsystems' device max-reductions (here: one subsystem).
    double expected_diag_norm = 0.0;
    double expected_mass_norm = 0.0;
    double min_mass = 1.0e300, max_mass = 0.0;
    for(SizeT v = 0; v < 8u; ++v)
    {
        min_mass = std::min(min_mass, oracle.masses[v]);
        max_mass = std::max(max_mass, oracle.masses[v]);
        if(isf[v] != 0)
            continue;
        const auto& D = oracle.D[v];
        expected_diag_norm =
            std::max(expected_diag_norm, std::max({std::abs(D(0, 0)),
                                                   std::abs(D(1, 1)),
                                                   std::abs(D(2, 2))}));
        expected_mass_norm = std::max(expected_mass_norm, oracle.masses[v]);
    }

    const auto diag_norm = impl.diag_norm();
    const auto mass_norm = impl.mass_norm();

    std::printf("[t54b] norm: diag_norm %.6e (hand-derived %.6e), mass_norm %.6e "
                "(hand-derived %.6e; per-vertex mass in [%.6e, %.6e])\n",
                diag_norm,
                expected_diag_norm,
                mass_norm,
                expected_mass_norm,
                min_mass,
                max_mass);

    CHECK(expected_diag_norm > 1.0);
    CHECK(expected_mass_norm > 1.0);
    CHECK(rel_err(diag_norm, expected_diag_norm) <= 1e-12);
    CHECK(rel_err(mass_norm, expected_mass_norm) <= 1e-12);
}

// ===========================================================================
//  CASE 3 — the constraint/reporter pair: duplicates, the frontend scene, the
//  extents and the uid
// ===========================================================================
TEST_CASE("fem_precond_external_vertex_force", "[luisa][fem_precond]")
{
    using namespace t54b;
    test_runtime();

    // ---- the scene: the frontend `uipc::constitution::FiniteElementExternalForce`
    auto desc = five_tet_desc();
    desc.with_external_force_constitution = true;
    // vertices 1, 3, 5 and 6 are constrained; vertex 5 is *massless* (density 0)
    // and vertex 7 carries a force but is not constrained (it must not be applied)
    desc.is_constrained = {0, 1, 0, 1, 0, 1, 1, 0};
    desc.densities      = {desc.density,
                           desc.density,
                           desc.density,
                           desc.density,
                           desc.density,
                           0.0,  // massless
                           desc.density,
                           desc.density};
    for(SizeT v = 0; v < desc.external_forces.size(); ++v)
        desc.external_forces[v] = Vector3{1.0 * static_cast<double>(v + 1),
                                          2.0 * static_cast<double>(v + 1),
                                          -3.0 * static_cast<double>(v + 1)};
    desc.external_forces[7] = Vector3{100.0, 100.0, 100.0};  // not constrained

    T54bProbe::reset();
    auto   scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t54b_extforce")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    REQUIRE(T54bProbe::ext_vertex_constraint != nullptr);
    CHECK(T54bProbe::ext_vertex_constraint->is_valid());
    REQUIRE(T54bProbe::ext_vertex != nullptr);
    CHECK(T54bProbe::ext_vertex->is_valid());
    REQUIRE(T54bProbe::ext_force != nullptr);
    CHECK(T54bProbe::ext_force->is_valid());
    REQUIRE(T54bProbe::fem != nullptr);

    // ---- (c) the uid story and the 0/0/0 extents ------------------------------
    // three *different* registries carry 671 (cookbook §7.20):
    //   * the constraint uid registry          -> FiniteElementExternalVertexForceConstraint
    //   * the external-force-reporter registry -> FEMExternalForceManager (T5.3)
    //   * the constitution uid registry        -> uipc::constitution::FiniteElementExternalForce
    CHECK(gg::FiniteElementExternalVertexForceConstraint::UID == 671ull);
    CHECK(gg::FiniteElementExternalVertexForce::UID == 671ull);
    CHECK(gg::FEMExternalForceManager::UID == 671ull);
    CHECK(T54bProbe::ext_vertex_constraint->uid() == 671ull);
    CHECK(T54bProbe::ext_vertex->uid() == 671ull);

    {
        FiniteElementAnimator::ReportExtentInfo info;
        CHECK(info.gradient_only() == false);
        T54bProbe::ext_vertex_constraint->do_report_extent(info);
        CHECK(info.energy_count() == 0u);
        CHECK(info.gradient_count() == 0u);
        CHECK(info.hessian_count() == 0u);
        // and the gradient-only flavour of the same call
        // (the reference's `check()` asserts hessian == 0 when gradient_only)
    }

    // ---- (a) the synthetic duplicate-vertex case ------------------------------
    // The constraint's own device buffers are overwritten with a table that adds
    // *three* entries on vertex 3 and two on vertex 0: that is what makes the
    // reporter's vector-valued atomic observable (a read-modify-write with the
    // same single writer would still copy the whole LVector3 three times and lose
    // two of the adds).
    {
        auto& impl = T54bProbe::ext_vertex_constraint->m_impl;
        std::vector<IndexT> ids      = {3, 0, 3, 7, 3};
        std::vector<Vector3> f_pairs = {{1.0, 0.0, -1.0},
                                       {2.0, 4.0, 6.0},
                                       {0.5, 0.25, 0.125},
                                       {-8.0, 1.0, 2.0},
                                       {10.0, 20.0, 30.0}};

        impl.vertex_ids.copy_from(span<const IndexT>{ids});
        std::vector<LVector3> lf;
        lf.reserve(f_pairs.size());
        for(auto& f : f_pairs)
            lf.push_back(gg::to_luisa(f));
        impl.forces.copy_from(span<const LVector3>{lf});
        gg::frame_stream().synchronize();

        CHECK(T54bProbe::ext_vertex_constraint->forces().size() == ids.size());
        CHECK(T54bProbe::ext_vertex_constraint->vertex_ids().size() == ids.size());

        // clear -> step: the manager's own contract (clear first, then every
        // reporter scatter-adds)
        gg::FEMExternalForceManager::Impl mgr_impl;
        mgr_impl.finite_element_method = T54bProbe::fem;
        mgr_impl.clear();
        gg::frame_stream().synchronize();

        gg::FEMExternalForceManager::ExternalForceInfo info{&mgr_impl};
        T54bProbe::ext_vertex->do_step(info);
        gg::frame_stream().synchronize();

        auto forces = read_vec3(T54bProbe::fem->vertex_external_forces());

        // hand-derived per-vertex sums
        std::vector<Vector3> expected(8, Vector3::Zero());
        for(SizeT k = 0; k < ids.size(); ++k)
            expected[static_cast<SizeT>(ids[k])] += f_pairs[k];

        double worst = 0.0;
        int    n_touched = 0;
        for(SizeT v = 0; v < 8u; ++v)
        {
            worst = std::max(worst, (forces[v] - expected[v]).norm());
            if(expected[v].norm() > 0.0)
                ++n_touched;
        }
        std::printf("[t54b] ext-force: duplicate-vertex scatter worst abs %.3e "
                    "(touched %d/8 vertices; vertex3 = (%.3f, %.3f, %.3f))\n",
                    worst,
                    n_touched,
                    forces[3].x(),
                    forces[3].y(),
                    forces[3].z());
        // the duplicated vertex got all three adds (a lost update would leave 1.0)
        CHECK(forces[3].x() == 11.5);
        CHECK(forces[3].y() == 20.25);
        CHECK(forces[3].z() == 29.125);
        CHECK(expected[3].x() == 11.5);
        CHECK(worst == 0.0);           // exactly representable values -> bitwise
        CHECK(n_touched == 3);         // 0, 3, 7 -- nothing else was touched
        for(SizeT v = 0; v < 8u; ++v)
            if(v != 0u && v != 3u && v != 7u)
                CHECK(forces[v].norm() == 0.0);

        // the scalar-alias contract: the LVector3 element stride is exactly
        // sizeof(LVector3), which is what makes `as<double>()` + 3*index + c exact
        auto native = T54bProbe::fem->vertex_external_forces().native();
        std::printf("[t54b] ext-force: LVector3 buffer element stride %zu B "
                    "(sizeof(LVector3) == %zu B)\n",
                    native.stride(),
                    sizeof(LVector3));
        CHECK(native.stride() == sizeof(LVector3));
        CHECK(sizeof(LVector3) == 24u);
    }

    // ---- (b) the frontend scene: one advance, accs == F/m ---------------------
    {
        // restore the constraint's buffers from the scene's own attributes by
        // running the animator's step once more through `advance()`
        step_frame(world);
        world.sync();
        gg::frame_stream().synchronize();

        auto forces   = read_vec3(T54bProbe::fem->vertex_external_forces());
        auto accs     = read_vec3(T54bProbe::fem->vertex_external_force_accs());
        auto masses   = read_device(T54bProbe::fem->masses());
        auto ids      = read_device(T54bProbe::ext_vertex_constraint->vertex_ids());

        std::printf("[t54b] ext-force scene: %zu constrained vertices, masses[5] = %.6e\n",
                    ids.size(),
                    masses[5]);

        // the constraint collected exactly the marked vertices, in vertex order
        REQUIRE(ids.size() == 4u);
        CHECK(ids[0] == 1);
        CHECK(ids[1] == 3);
        CHECK(ids[2] == 5);
        CHECK(ids[3] == 6);

        double worst_f = 0.0, worst_a = 0.0;
        for(SizeT v = 0; v < 8u; ++v)
        {
            const bool constrained = desc.is_constrained[v] != 0;
            const auto F = constrained ? desc.external_forces[v] : Vector3::Zero();
            worst_f      = std::max(worst_f, (forces[v] - F).norm());

            Vector3 a = Vector3::Zero();
            if(constrained && masses[v] > 0.0)
                a = F / masses[v];
            worst_a = std::max(worst_a, (accs[v] - a).norm());
        }
        std::printf("[t54b] ext-force scene: worst |F - expected| %.3e, "
                    "worst |a - F/m| %.3e (massless vertex 5 acc %.3e)\n",
                    worst_f,
                    worst_a,
                    accs[5].norm());
        CHECK(worst_f <= 1e-12);
        CHECK(worst_a <= 1e-12);
        // the massless vertex: force present, acceleration exactly 0
        CHECK(forces[5].norm() > 0.0);
        CHECK(accs[5].norm() == 0.0);
        // the unconstrained vertex 7: its attribute force was never collected
        CHECK(forces[7].norm() == 0.0);
        CHECK(accs[7].norm() == 0.0);
    }
}

// ===========================================================================
//  CASE 4 — every factored routine: host `double` vs device DSL, bitwise
// ===========================================================================
TEST_CASE("fem_precond_device_vs_host", "[luisa][fem_precond]")
{
    using namespace t54b;
    test_runtime();

    const RoundData d;
    constexpr std::int32_t seg_offset = 0;
    constexpr std::int32_t seg_count  = 4;  // row 4 -> i == 4 >= count (out of window)

    const auto host = round_host(d, seg_offset, seg_count);

    // ---- device side ---------------------------------------------------------
    lud::LDeviceBuffer<std::int32_t> rows_dev, cols_dev, in_win_dev, is_diag_dev, index_dev;
    lud::LDeviceBuffer<double>       vals_dev, r_dev, z_dev;
    lud::LDeviceBuffer<LMatrix3x3>   inv_dev;

    rows_dev.copy_from(span<const std::int32_t>{d.rows});
    cols_dev.copy_from(span<const std::int32_t>{d.cols});
    vals_dev.copy_from(span<const double>{d.blocks});
    r_dev.copy_from(span<const double>{d.r});
    std::vector<double> z_init(d.r.size(), 0.0);
    z_dev.copy_from(span<const double>{z_init});
    inv_dev.resize(static_cast<SizeT>(kRoundTriplets));
    std::vector<std::int32_t> zeros(kRoundTriplets, 0);
    in_win_dev.copy_from(span<const std::int32_t>{zeros});
    is_diag_dev.copy_from(span<const std::int32_t>{zeros});
    index_dev.copy_from(span<const std::int32_t>{zeros});
    gg::frame_stream().synchronize();

    {
        auto& k = t54b_diag_inv_kernel();
        gg::frame_stream().record(k(rows_dev.view(),
                                    cols_dev.view(),
                                    vals_dev.view(),
                                    inv_dev.view(),
                                    in_win_dev.view(),
                                    is_diag_dev.view(),
                                    index_dev.view(),
                                    seg_offset,
                                    seg_count,
                                    static_cast<uint32_t>(kRoundTriplets))
                                      .dispatch(kRoundTriplets));
    }
    {
        auto& k = t54b_diag_apply_kernel();
        gg::frame_stream().record(k(r_dev.cview(),
                                    z_dev.view(),
                                    inv_dev.cview(),
                                    static_cast<uint32_t>(kRoundTriplets))
                                      .dispatch(kRoundTriplets));
    }
    // the scatter-add over the *scalar alias* of an AoS LVector3 buffer, with
    // duplicates on vertex 0 (the same table the host side evaluates)
    lud::LDeviceBuffer<LVector3> aos_forces;
    std::vector<LVector3>        aos_init(3, LVector3{0.0, 0.0, 0.0});
    aos_forces.copy_from(span<const LVector3>{aos_init});
    std::vector<std::int32_t> sids      = {0, 1, 0, 2, 0};
    std::vector<double>       sfvals    = {1.0, 2.0, 3.0,   //
                                           4.0, 5.0, 6.0,   //
                                           7.0, 8.0, 9.0,   //
                                           10.0, 11.0, 12.0,  //
                                           -1.0, -2.0, -3.0};
    std::vector<LVector3>     sf_host;
    for(SizeT k = 0; k < sids.size(); ++k)
        sf_host.push_back(LVector3{sfvals[k * 3], sfvals[k * 3 + 1], sfvals[k * 3 + 2]});
    lud::LDeviceBuffer<std::int32_t> sids_dev;
    lud::LDeviceBuffer<LVector3>     sf_dev;
    sids_dev.copy_from(span<const std::int32_t>{sids});
    sf_dev.copy_from(span<const LVector3>{sf_host});
    gg::frame_stream().synchronize();

    {
        auto native = aos_forces.view().native();
        CHECK(native.stride() == sizeof(LVector3));  // the alias contract
        auto& k = t54b_scatter_add_kernel();
        gg::frame_stream().record(k(native.as<double>(),
                                    sids_dev.cview(),
                                    sf_dev.cview(),
                                    static_cast<uint32_t>(sids.size()))
                                      .dispatch(static_cast<uint32_t>(sids.size())));
    }
    gg::frame_stream().synchronize();

    // ---- read back and compare bitwise ---------------------------------------
    std::vector<LMatrix3x3> dev_inv(static_cast<SizeT>(kRoundTriplets));
    inv_dev.copy_to(span<LMatrix3x3>{dev_inv});
    auto dev_in_window = read_device(in_win_dev.cview());
    auto dev_is_diag   = read_device(is_diag_dev.cview());
    auto dev_index     = read_device(index_dev.cview());
    std::vector<double> dev_z(d.r.size(), 0.0);
    z_dev.copy_to(span<double>{dev_z});
    std::vector<LVector3> dev_aos(3);
    aos_forces.copy_to(span<LVector3>{dev_aos});

    int    n_vals = 0, n_val_mismatch = 0, n_flag_vals = 0, n_flag_mismatch = 0;
    double worst_abs = 0.0, worst_rel = 0.0;
    auto   compare_value = [&](double a, double b)
    {
        ++n_vals;
        if(a != b)
        {
            ++n_val_mismatch;
            worst_abs = std::max(worst_abs, std::abs(a - b));
            worst_rel = std::max(worst_rel, rel_err(a, b));
        }
    };
    auto compare_flag = [&](std::int32_t a, std::int32_t b)
    {
        ++n_flag_vals;
        if(a != b)
            ++n_flag_mismatch;
    };
    // diag_inv_row: 9 doubles + the two guards + the index, per triplet
    for(int k = 0; k < kRoundTriplets; ++k)
    {
        for(int c = 0; c < 3; ++c)
            for(int rr = 0; rr < 3; ++rr)
                compare_value(host.inv[static_cast<SizeT>(k)](rr, c),
                              dev_inv[static_cast<SizeT>(k)](rr, c));
        compare_flag(host.in_window[k], dev_in_window[k]);
        compare_flag(host.is_diagonal[k], dev_is_diag[k]);
        compare_flag(host.index[k], dev_index[k]);
    }
    // diag_apply_row: 3 doubles per block
    for(SizeT i = 0; i < dev_z.size(); ++i)
        compare_value(host.z[i], dev_z[i]);
    // scatter_add_vec3: 9 doubles
    for(SizeT v = 0; v < 3u; ++v)
        for(int c = 0; c < 3; ++c)
            compare_value(host.forces[v * 3 + static_cast<SizeT>(c)], dev_aos[v](c));

    std::printf("[t54b] device_vs_host: values %d/%d equal (worst abs %.3e, worst rel %.3e), "
                "guards+indices %d/%d equal (guards %d/%d/%d, duplicated vertex 0 = "
                "(%.1f, %.1f, %.1f))\n",
                n_vals - n_val_mismatch,
                n_vals,
                worst_abs,
                worst_rel,
                n_flag_vals - n_flag_mismatch,
                n_flag_vals,
                dev_in_window[0],
                dev_is_diag[0],
                dev_index[0],
                dev_aos[0](0),
                dev_aos[0](1),
                dev_aos[0](2));

    // the round is non-vacuous: the guards really differ across triplets and the
    // duplicates really were summed
    CHECK(dev_in_window[0] == 1);
    CHECK(dev_in_window[4] == 0);
    CHECK(dev_is_diag[0] == 1);
    CHECK(dev_is_diag[1] == 0);
    CHECK(dev_index[4] == 4);
    CHECK(dev_aos[0](0) == 1.0 + 7.0 - 1.0);
    CHECK(dev_aos[0](1) == 2.0 + 8.0 - 2.0);
    CHECK(dev_aos[0](2) == 3.0 + 9.0 - 3.0);
    // the integer-valued guard/index outputs are *exact*
    CHECK(n_flag_mismatch == 0);
    // the floating-point outputs are the same expression evaluated on both sides;
    // the CUDA codegen contracts `a * b + c` into an FMA, so the agreement is to
    // round-off (measured: 65/69 values bitwise, worst 1.110e-16 absolute = 1 ulp)
    // rather than bitwise for every entry -- the numbers are printed above and
    // asserted at 1e-15 relative
    CHECK(worst_rel <= 1e-15);
    CHECK(n_vals == 69);  // 5 x 9 inverse entries + 5 x 3 applied entries + 3 x 3 forces
}
