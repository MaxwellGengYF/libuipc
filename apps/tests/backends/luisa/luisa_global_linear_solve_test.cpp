/********************************************************************
 * @file luisa_global_linear_solve_test.cpp
 * @brief T8.3b — `GlobalLinearSystem::solve()`, the two norm wrappers, the debug dump
 *        and the engine's `init()` wiring, on the shipped chain.
 *
 * WHAT IS DRIVEN (the reference contract, src/backends/cuda/linear_system/
 * global_linear_system.cu:30-103)
 * --------------------------------------------------------------------
 *  * `do_build()` — the `extras/debug/dump_linear_system` config read;
 *  * `solve()` — the seven statements in the reference's order: reset the counter ->
 *    `build_linear_system()` -> empty early return -> [`_dump_A_b()`] ->
 *    `solve_linear_system()` -> [`_dump_x()`] -> `distribute_solution()`;
 *  * `_dump_A_b()` / `_dump_x()` — three `.mtx` files at the reference's
 *    `BackendPathTool` path, named `A.<frame>.<newton_iter>.mtx` / `b.…` / `x.…`;
 *  * `diag_norm()` / `mass_norm()` — `build_linear_system()` then the `Impl::` max over
 *    the diag subsystems;
 *  * the engine wiring: `SimEngine::build_engine_systems()` `require<>`s the system and
 *    `init_scene()` calls `init()` between the line searcher and the time-integrator
 *    manager (`sim_engine_do_init.cu:37`, `:130-132`), so a real scene reaches
 *    `dof_count() != 0` **without a test calling `init()` by hand**.
 *
 * THE SYSTEM
 * ----------
 * A 1-D chain of 8 block nodes (24 dofs): `A_ii = c_i * I` with
 * `c = (2, 3, 4, 6, 8, 12, 16, 24)` (heterogeneous and strictly diagonally dominant) and
 * `A_{i,i+1} = -I`, written upper-triangular only — `ge2sym` keeps `row <= col` and
 * `rbk_sym_spmv` mirrors the strict upper part.  The matrix is SPD and dyadic, so the
 * host oracle is exact.  A second, 0-dof `DiagLinearSubsystem` ("norm probe") exists only
 * to make the two norm wrappers' `max` over subsystems discriminating.
 *
 * ORACLES
 *  * `Eigen::LDLT` on the **symmetrised** dense reconstruction of the *imported* matrix
 *    (`A_eff = U + Uᵀ − D`: the stored diagonal blocks are whole, the strict upper blocks
 *    are mirrored-and-transposed — research §7 T2), for `x = A⁻¹ b` to 1e-10 relative.
 *    `tol_rate` is driven to 1e-16 in that case so PCG runs to its round-off floor: with
 *    the default 1e-3 the residual gate is only `sqrt(1e-3) ≈ 3e-2` and a 1e-10 *solution*
 *    comparison would be meaningless.
 *  * the dumped files are re-imported and compared to the assembled `bcoo_A` / `b`
 *    (Class A: `Δ == 0`), which is what makes "the file is the system the solver saw"
 *    checkable without a CUDA backend.
 *
 * EXPLICIT SKIPS (never silent passes)
 *  * `solve_from_cuda_mtx` — no CUDA ground truth in this environment.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"
#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <type_define.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/linear_pcg.h>
#include <utils/matrix_market.h>
// The test target's include path puts src/backends/none first: the shell header must be
// qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
namespace gg   = uipc::backend::luisa;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;

namespace t83s
{
namespace fs = std::filesystem;

// ===========================================================================
//  1) the block table — ONE source of truth for the device writes and the oracles
// ===========================================================================
constexpr int kBlockDim = 3;
constexpr int kNodes    = 8;
constexpr int kDofs     = kNodes * kBlockDim;  // 24

constexpr const char* kMarker = "t83s_solve_marker";
constexpr U64         kUidChain = 0x54385343ull;  // "T83SC"
constexpr U64         kUidProbe = 0x54385344ull;

/// The two subsystems' norm values.  They are deliberately *different* in both
/// directions, so `max` cannot be confused with "the last subsystem wins".
constexpr double kChainDiagNorm = 24.0;  // max_i c_i
constexpr double kChainMassNorm = 2.0;   // min_i c_i
constexpr double kProbeDiagNorm = 7.5;
constexpr double kProbeMassNorm = 3.25;

/// How many times the chain's `do_assemble` ran (the norm wrappers must rebuild first).
inline int g_assemble_calls = 0;

/// A 3x3 block, `v[r][c]` like `linalg::LMat::operator()(r, c)`.
struct Block
{
    double v[kBlockDim][kBlockDim]{};

    [[nodiscard]] double operator()(int r, int c) const { return v[r][c]; }
    void                set(int r, int c, double x) { v[r][c] = x; }
};

struct HostTriplet
{
    int   row = 0;
    int   col = 0;
    Block block;
};

[[nodiscard]] inline Block scaled_identity(double c)
{
    Block b;
    for(int i = 0; i < kBlockDim; ++i)
        b.set(i, i, c);
    return b;
}

/// 2, 3, 4, 6, 8, 12, 16, 24 — all exactly representable, all >= 2 so the chain is
/// strictly diagonally dominant (hence SPD): a *path Laplacian* (`c = 1` at the ends)
/// is singular and would divide by `pᵀAp ~ 0` in the first PCG step.
[[nodiscard]] inline double diag_coeff(int i)
{
    static constexpr double kDiag[kNodes] = {2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0};
    return kDiag[i];
}

/// The upper-triangular triplet table (the `ge2sym` packing): diagonal blocks first,
/// then the (i, i+1) couplings.
[[nodiscard]] inline std::vector<HostTriplet> chain_triplets()
{
    std::vector<HostTriplet> ts;
    ts.reserve(2u * kNodes - 1u);
    for(int i = 0; i < kNodes; ++i)
        ts.push_back({i, i, scaled_identity(diag_coeff(i))});
    for(int i = 0; i + 1 < kNodes; ++i)
        ts.push_back({i, i + 1, scaled_identity(-1.0)});
    return ts;
}

/// The right-hand side the fixture assembles into `b` (`gradients()`), dyadic and
/// deliberately *not* constant: a smooth (let alone constant) `b` is nearly orthogonal
/// to every row sum of a chain operator, which makes `bᵀAb` tiny and the first `alpha`
/// explode.
[[nodiscard]] inline std::vector<double> default_rhs()
{
    std::vector<double> b(kDofs);
    for(int i = 0; i < kDofs; ++i)
    {
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        b[static_cast<std::size_t>(i)] =
            sign * (1.0 + 0.125 * static_cast<double>((i * 5 + 3) % 7));
    }
    return b;
}

/// The dense matrix the solver applies: the symmetric closure of the stored table —
/// `A(i, j) = block`, `A(j, i) = blockᵀ` for `i != j`, the diagonal block as stored
/// (research §7 T2).  This is what `rbk_sym_spmv` computes with the upper-packed BCOO.
[[nodiscard]] inline Eigen::MatrixXd dense_A(const std::vector<HostTriplet>& ts)
{
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(kDofs, kDofs);
    for(const auto& t : ts)
    {
        for(int r = 0; r < kBlockDim; ++r)
            for(int c = 0; c < kBlockDim; ++c)
            {
                A(t.row * kBlockDim + r, t.col * kBlockDim + c) += t.block(r, c);
                if(t.row != t.col)
                    A(t.col * kBlockDim + c, t.row * kBlockDim + r) += t.block(r, c);
            }
    }
    return A;
}

// ---------------------------------------------------------------------------
//  small helpers (the other luisa test files' idioms)
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = fs::temp_directory_path() / name;
    fs::create_directories(p);
    return p.string() + "/";
}

/// `std::printf` + `fflush`: a UIPC_ASSERT aborts the process and a redirected stdout is
/// fully buffered, so a diagnostic without the flush is lost exactly when it is needed.
#define T83S_LOG(...) \
    do \
    { \
        std::printf(__VA_ARGS__); \
        std::fflush(stdout); \
    } while(false)

struct Counter
{
    int checks = 0;
};

struct CaseScope
{
    std::string name;
    Counter     counter;

    explicit CaseScope(std::string n)
        : name(std::move(n))
    {
    }
    ~CaseScope()
    {
        T83S_LOG("[t83s] %s: %d checks\n", name.c_str(), counter.checks);
    }
};

#define T83S_CHECK(counter, cond) \
    do \
    { \
        ++(counter).checks; \
        CHECK(cond); \
    } while(false)

[[nodiscard]] inline double max_abs_diff(const std::vector<double>& a,
                                         const std::vector<double>& b)
{
    if(a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double d = 0.0;
    for(std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

/// `max |a - b| / max(1, max |b|)` — the tolerance band of plans/03 §5.
[[nodiscard]] inline double rel_diff(const std::vector<double>& a,
                                     const std::vector<double>& b)
{
    if(a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double scale = 1.0, d = 0.0;
    for(std::size_t i = 0; i < b.size(); ++i)
    {
        scale = std::max(scale, std::abs(b[i]));
        d     = std::max(d, std::abs(a[i] - b[i]));
    }
    return d / scale;
}

[[nodiscard]] inline double inf_norm(const std::vector<double>& v)
{
    double n = 0.0;
    for(double x : v)
        n = std::max(n, std::abs(x));
    return n;
}

[[nodiscard]] inline double relative_residual(const Eigen::MatrixXd&     A,
                                              const std::vector<double>& b,
                                              const std::vector<double>& x)
{
    Eigen::Map<const Eigen::VectorXd> bv(b.data(), static_cast<Eigen::Index>(b.size()));
    Eigen::Map<const Eigen::VectorXd> xv(x.data(), static_cast<Eigen::Index>(x.size()));
    Eigen::VectorXd                   r = bv - A * xv;
    return r.norm() / bv.norm();
}

// ===========================================================================
//  2) the scene + the marker gate
// ===========================================================================
[[nodiscard]] inline bool has_marker(uipc::backend::WorldVisitor& world)
{
    for(auto& slot : world.scene().geometries())
    {
        if(auto* sc = slot->geometry().as<SimplicialComplex>())
        {
            if(sc->vertices().find<IndexT>(kMarker))
                return true;
        }
    }
    return false;
}

struct RunOptions
{
    /// `extras/debug/dump_linear_system`
    bool   dump       = false;
    /// whether the geometry carries the fixture marker (false -> a dof-less system)
    bool   marker     = true;
    double tol_rate   = 1e-3;
};

[[nodiscard]] inline Json base_config()
{
    Json config                      = Scene::default_config();
    config["contact"]["enable"]      = IndexT{0};
    config["sanity_check"]["enable"] = IndexT{0};
    config["integrator"]["type"]     = std::string{"bdf1"};
    config["dt"]                     = 0.01;
    return config;
}

[[nodiscard]] inline Scene build_scene(const Json& config, bool marker)
{
    Scene scene{config};
    auto  object = scene.objects().create("t83s");

    const std::vector<Vector3> positions = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    const std::vector<Vector4i> tets = {{0, 1, 2, 3}};

    auto mesh = tetmesh(span<const Vector3>{positions}, span<const Vector4i>{tets});
    label_surface(mesh);
    // no dof-producing constitution claims this mesh: the whole linear system of this
    // file's scenes is the fixture below
    if(marker)
        mesh.vertices().create<IndexT>(kMarker, 1);
    object->geometries().create(mesh);
    return scene;
}

// ===========================================================================
//  3) the fixtures
// ===========================================================================
class T83sChain;

/// The single 24-dof diag subsystem: it owns every dof and writes both halves of the
/// system — the upper-triangular Hessian table and the gradient (the right-hand side).
/// Values are uploaded from the host table instead of a trace-time kernel: an upload
/// cannot hide an arithmetic difference, and the dyadic table keeps the Class-A gates
/// exact.
class T83sChain final : public gg::DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    static inline bool enabled = false;

    /// What `retrieve_solution()` received, i.e. the distributed window of `x`.
    static inline std::vector<double> distributed;

    static void reset()
    {
        enabled = false;
        distributed.clear();
        g_assemble_calls = 0;
    }

  protected:
    [[nodiscard]] U64 get_uid() const noexcept override { return kUidChain; }

    void do_init(InitInfo& info) override
    {
        (void)info;
        enabled = has_marker(world());
    }

    void do_report_init_extent(gg::GlobalLinearSystem::InitDofExtentInfo& info) override
    {
        info.extent(enabled ? SizeT{kDofs} : SizeT{0});
    }

    void do_receive_init_dof_info(gg::GlobalLinearSystem::InitDofInfo& info) override
    {
        // the layout hands the *whole* system to this subsystem: offset 0, 24 dofs
        if(enabled)
        {
            UIPC_ASSERT(info.dof_offset() == 0 && info.dof_count() == kDofs,
                        "T83sChain: unexpected dof window ({}, {})",
                        info.dof_offset(),
                        info.dof_count());
        }
    }

    void do_report_extent(gg::GlobalLinearSystem::DiagExtentInfo& info) override
    {
        auto n = enabled ? chain_triplets().size() : SizeT{0};
        info.extent(n, enabled ? SizeT{kDofs} : SizeT{0});
    }

    void do_assemble(gg::GlobalLinearSystem::DiagInfo& info) override
    {
        if(!enabled)
            return;

        ++g_assemble_calls;

        // ---- the Hessian: the upper-triangular table into the windowed view ---------
        auto       hessians = info.hessians();
        const auto count    = static_cast<std::size_t>(hessians.triplet_count());
        const auto ts       = chain_triplets();
        if(count != ts.size())
            return;  // the caller's assertions then fail loudly

        // the window's submatrix offset: with one 24-dof subsystem it is (0, 0), and the
        // stored indices are *global* block indices (`submatrix()` adds the offset)
        const auto offset = hessians.submatrix_offset();
        const auto begin  = static_cast<std::size_t>(hessians.triplet_index_offset());

        std::vector<std::int32_t> rows(count);
        std::vector<std::int32_t> cols(count);
        std::vector<double>       vals(count * 9u, 0.0);
        for(std::size_t i = 0; i < count; ++i)
        {
            rows[i] = static_cast<std::int32_t>(offset.x + ts[i].row);
            cols[i] = static_cast<std::int32_t>(offset.y + ts[i].col);
            for(int c = 0; c < kBlockDim; ++c)
                for(int r = 0; r < kBlockDim; ++r)
                    // flat storage is column-major inside a block, like LMat
                    vals[i * 9u + static_cast<std::size_t>(c * kBlockDim + r)] =
                        ts[i].block(r, c);
        }

        hessians.row_indices().subview(begin, count).copy_from(span<const std::int32_t>{rows});
        hessians.col_indices().subview(begin, count).copy_from(span<const std::int32_t>{cols});
        hessians.values().subview(begin * 9u, count * 9u).copy_from(span<const double>{vals});

        // ---- the gradient (the RHS): `DiagInfo::gradients()` is this subsystem's
        // window of `b`, so the assembled system has a non-zero right-hand side without
        // the test injecting one *after* the build (which the rebuild would wipe).
        auto               gradients = info.gradients();
        const auto         rhs       = default_rhs();
        if(gradients.size() == rhs.size())
            gradients.buffer_view().copy_from(span<const double>{rhs});
    }

    void do_accuracy_check(gg::GlobalLinearSystem::AccuracyInfo& info) override
    {
        // the reference's `AccuracyInfo::satisfied(bool)` defaults to true; this fixture
        // has no Newton-level tolerance to model, so the device's stopping rule is
        // `|rz_k| <= tol*|rz_0|` alone.
        (void)info;
    }

    void do_retrieve_solution(gg::GlobalLinearSystem::SolutionInfo& info) override
    {
        if(!enabled)
            return;
        // `distribute_solution()` hands this subsystem its window of `x`; download it so
        // the test can prove the distribute step really ran with the solution.
        auto solution = info.solution();
        distributed.resize(solution.size());
        solution.buffer_view().copy_to(span<double>{distributed});
        gg::luisa_device::LStream::current().synchronize();
    }

    Float do_diag_norm(gg::GlobalLinearSystem::DiagNormInfo& info) override
    {
        (void)info;
        return enabled ? static_cast<Float>(kChainDiagNorm) : Float{0.0};
    }

    Float do_mass_norm(gg::GlobalLinearSystem::DiagNormInfo& info) override
    {
        (void)info;
        return enabled ? static_cast<Float>(kChainMassNorm) : Float{0.0};
    }
};

/// A 0-dof `DiagLinearSubsystem` whose only job is to make `diag_norm()`/`mass_norm()`
/// discriminating: the wrappers take a `max` over the registered diag subsystems, and
/// with one subsystem that is indistinguishable from "return the only value".
class T83sProbe final : public gg::DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    static inline bool enabled = false;
    /// the assemble counter observed *inside* `do_diag_norm` / `do_mass_norm`
    static inline int assemble_calls_at_norm = -1;

    static void reset()
    {
        enabled             = false;
        assemble_calls_at_norm = -1;
    }

  protected:
    [[nodiscard]] U64 get_uid() const noexcept override { return kUidProbe; }

    void do_init(InitInfo& info) override
    {
        (void)info;
        enabled = has_marker(world());
    }

    void do_report_init_extent(gg::GlobalLinearSystem::InitDofExtentInfo& info) override
    {
        info.extent(SizeT{0});  // owns no dofs
    }

    void do_receive_init_dof_info(gg::GlobalLinearSystem::InitDofInfo& info) override
    {
        (void)info;
    }

    void do_report_extent(gg::GlobalLinearSystem::DiagExtentInfo& info) override
    {
        info.extent(SizeT{0}, SizeT{0});  // and no Hessian blocks
    }

    void do_assemble(gg::GlobalLinearSystem::DiagInfo& info) override { (void)info; }

    void do_accuracy_check(gg::GlobalLinearSystem::AccuracyInfo& info) override
    {
        (void)info;
    }

    void do_retrieve_solution(gg::GlobalLinearSystem::SolutionInfo& info) override
    {
        (void)info;
    }

    Float do_diag_norm(gg::GlobalLinearSystem::DiagNormInfo& info) override
    {
        (void)info;
        assemble_calls_at_norm = g_assemble_calls;
        return enabled ? static_cast<Float>(kProbeDiagNorm) : Float{0.0};
    }

    Float do_mass_norm(gg::GlobalLinearSystem::DiagNormInfo& info) override
    {
        (void)info;
        return enabled ? static_cast<Float>(kProbeMassNorm) : Float{0.0};
    }
};

/// Captures the engine's systems once they are built.
class T83sProbeSystem final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::GlobalLinearSystem* gls   = nullptr;
    inline static gg::LinearPCG*          pcg   = nullptr;
    inline static T83sChain*              chain = nullptr;
    inline static T83sProbe*              probe = nullptr;

    static void reset()
    {
        gls   = nullptr;
        pcg   = nullptr;
        chain = nullptr;
        probe = nullptr;
    }

  private:
    void do_build() override
    {
        // `require<>` is the reference's registration path: it throws when the system
        // is absent or invalid, which is exactly the assertion this probe needs.
        gls   = &require<gg::GlobalLinearSystem>();
        pcg   = &require<gg::LinearPCG>();
        chain = &require<T83sChain>();
        probe = &require<T83sProbe>();
    }
};
}  // namespace t83s

REGISTER_SIM_SYSTEM(t83s::T83sChain);
REGISTER_SIM_SYSTEM(t83s::T83sProbe);
REGISTER_SIM_SYSTEM(t83s::T83sProbeSystem);

namespace t83s
{
// ===========================================================================
//  4) the driver
// ===========================================================================
/// Build a fresh world for `options` and hand the *engine's* `GlobalLinearSystem` (layout
/// done by the engine, solver registered) to `fn`, with everything alive.
///
/// NOTE (T8.3b, the point of the task): `fn` receives a system whose `init()` the
/// **engine** called — no test calls `gls->init()` here any more, which is what makes
/// `dof_count()` a real scene-level gate instead of a test-only one.
template <typename Fn>
void with_world(const RunOptions& options, Fn&& fn)
{
    T83sProbeSystem::reset();
    T83sChain::reset();
    T83sProbe::reset();

    Json config                       = base_config();
    config["linear_system"]["tol_rate"] = options.tol_rate;
    if(options.dump)
        config["extras"]["debug"]["dump_linear_system"] = IndexT{1};

    const auto workspace = workspace_path("uipc_luisa_t83s_solve");
    // the dump directory of the *previous* run must not be able to satisfy the
    // "the file exists" assertions of this one
    fs::remove_all(fs::path{workspace} / "debug");

    auto   scene = build_scene(config, options.marker);
    Engine engine{"luisa", workspace};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T83sProbeSystem::gls != nullptr);
    REQUIRE(T83sProbeSystem::pcg != nullptr);
    REQUIRE(T83sProbeSystem::chain != nullptr);

    fn(*T83sProbeSystem::gls, T83sProbeSystem::gls->impl(), workspace);
}

/// The reference's dump directory, spelled out independently of the implementation:
/// `<workspace>/debug/<backend>/<dir>/<file>/` (`BackendPathTool::workspace()` strips
/// `src/backends` and prefixes `debug/`).  `UIPC_RELATIVE_SOURCE_FILE` of the dump is the
/// *backend* TU, so the last component is `global_linear_system.cpp`.
[[nodiscard]] inline std::string expected_dump_dir(const std::string& workspace)
{
    return workspace + "debug/luisa/linear_system/global_linear_system.cpp/";
}

[[nodiscard]] inline std::vector<double> read_vector(const std::string& path)
{
    std::vector<double> values;
    std::ifstream       in{path};
    std::string         line;
    int                 seen_header = 0;
    int                 rows = 0, cols = 0;
    while(std::getline(in, line))
    {
        if(line.empty() || line[0] == '%')
            continue;
        if(seen_header == 0)
        {
            std::istringstream iss(line);
            iss >> rows >> cols;
            seen_header = 1;
            continue;
        }
        values.push_back(std::stod(line));
    }
    if(values.size() != static_cast<std::size_t>(rows))
        values.clear();
    return values;
}
}  // namespace t83s

// ===========================================================================
//  CASE 1 — the engine wiring: `dof_count() != 0` with no manual `init()`
// ===========================================================================
TEST_CASE("solve_engine_init_wired", "[luisa][solve]")
{
    using namespace t83s;
    test_runtime();
    CaseScope scope{"solve_engine_init_wired"};

    // No `gls.init()` call anywhere in this case: the layout has to come from
    // `SimEngine::init_scene()` (T8.3b), in the reference's position between the line
    // searcher and the time-integrator manager.
    with_world({},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl, const std::string&)
               {
                   auto& c = scope.counter;

                   T83S_LOG("[t83s] init: initialized %d, dof_count %zu, diag subsystems "
                            "%zu, x %zu, b %zu\n",
                            static_cast<int>(impl.initialized),
                            static_cast<std::size_t>(gls.dof_count()),
                            impl.diag_subsystems.view().size(),
                            impl.x.size(),
                            impl.b.size());

                   T83S_CHECK(c, impl.initialized);  // <- SimEngine::init_scene()
                   T83S_CHECK(c, gls.dof_count() == static_cast<SizeT>(kDofs));
                   T83S_CHECK(c,
                              gls.dof_count()
                                  == impl.diag_dof_offsets_counts.total_count());
                   T83S_CHECK(c, T83sChain::enabled);
                   T83S_CHECK(c, T83sProbe::enabled);

                   // Every *other* test TU of this binary registers its own marker-gated
                   // diag subsystem too, and in a scene without its marker it owns no dofs
                   // — so the collection is larger than this file's two subsystems, while
                   // the layout must still put all 24 dofs on this file's chain and 0 on
                   // everything else.  Locate the two by identity instead of assuming
                   // indices.
                   const auto subs    = impl.diag_subsystems.view();
                   const auto offsets = impl.diag_dof_offsets_counts.offsets();
                   const auto counts  = impl.diag_dof_offsets_counts.counts();

                   std::size_t chain_index   = subs.size();
                   std::size_t probe_index   = subs.size();
                   std::size_t owners        = 0;
                   for(std::size_t i = 0; i < subs.size(); ++i)
                   {
                       if(subs[i] == T83sProbeSystem::chain)
                           chain_index = i;
                       if(subs[i] == T83sProbeSystem::probe)
                           probe_index = i;
                       if(counts[i] != 0)
                           ++owners;
                   }

                   T83S_LOG("[t83s] init: %zu diag subsystems registered, chain at %zu "
                            "(offset %d, count %d), probe at %zu, dof owners %zu\n",
                            subs.size(),
                            chain_index,
                            static_cast<int>(offsets[chain_index]),
                            static_cast<int>(counts[chain_index]),
                            probe_index,
                            owners);

                   T83S_CHECK(c, subs.size() >= 2u);
                   T83S_CHECK(c, chain_index < subs.size());
                   T83S_CHECK(c, probe_index < subs.size());
                   T83S_CHECK(c, chain_index != probe_index);
                   // the uid-ordered layout: exactly one subsystem owns dofs, and the 24
                   // dofs start at offset 0 (issue #271's stable order)
                   T83S_CHECK(c, owners == 1u);
                   T83S_CHECK(c, offsets[chain_index] == 0);
                   T83S_CHECK(c, counts[chain_index] == static_cast<IndexT>(kDofs));
                   T83S_CHECK(c, counts[probe_index] == 0);

                   // `init()` is the layout pass only: nothing is allocated and nothing
                   // is solved yet (`build_linear_system()` owns the allocation)
                   T83S_CHECK(c, impl.x.size() == 0u);
                   T83S_CHECK(c, impl.b.size() == 0u);
                   T83S_CHECK(c, gls.last_solve_iterations() == 0u);
                   T83S_CHECK(c, impl.empty_system);  // the initial value

                   // no subsystem has a local preconditioner here, so every dof window is
                   // in the `z <- r` fallback set
                   T83S_CHECK(c,
                              impl.no_precond_diag_subsystem_indices.size() == subs.size());
               });
}

// ===========================================================================
//  CASE 2 — the solve chain's observables: counter, x, distribute, idempotence
// ===========================================================================
TEST_CASE("solve_chain_observables", "[luisa][solve]")
{
    using namespace t83s;
    test_runtime();
    CaseScope scope{"solve_chain_observables"};

    with_world({},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl, const std::string& ws)
               {
                   auto& c = scope.counter;

                   REQUIRE(T83sChain::distributed.empty());

                   gls.solve();  // the whole chain: build -> solve -> distribute

                   const auto iters1 = gls.last_solve_iterations();
                   const auto nnz    = impl.bcoo_A.triplet_count();
                   const auto dofs   = impl.b.size();

                   std::vector<double> x1(dofs);
                   impl.x.copy_to(span<double>{x1});

                   T83S_LOG("[t83s] solve #1: iters %zu, dofs %zu, nnz %zu, |x|inf %.6e, "
                            "distributed %zu\n",
                            static_cast<std::size_t>(iters1),
                            static_cast<std::size_t>(dofs),
                            static_cast<std::size_t>(nnz),
                            inf_norm(x1),
                            T83sChain::distributed.size());

                   T83S_CHECK(c, dofs == static_cast<SizeT>(kDofs));
                   T83S_CHECK(c, nnz == chain_triplets().size());  // 15 upper blocks
                   T83S_CHECK(c, iters1 > 0u);
                   T83S_CHECK(c, iters1 <= static_cast<SizeT>(2 * kDofs));  // max_iter = 2n
                   T83S_CHECK(c, inf_norm(x1) > 0.0);  // really solved, not left at zero

                   // ---- distribute_solution() ran with the solution -------------------
                   // (the trap: without it every geometry stays at the predicted position
                   // while `x` still looks right in the dump)
                   T83S_CHECK(c, T83sChain::distributed.size() == static_cast<std::size_t>(kDofs));
                   T83S_CHECK(c, max_abs_diff(T83sChain::distributed, x1) == 0.0);

                   // the residual gate the parity harness uses (unpreconditioned: the
                   // stopping rule is |rz| <= tol*|rz_0|, i.e. ||r||/||b|| <= sqrt(tol))
                   const auto A = dense_A(chain_triplets());
                   const auto b = default_rhs();
                   const auto residual = relative_residual(A, b, x1);
                   T83S_LOG("[t83s] solve #1: ||b - Ax||/||b|| = %.3e (gate %.3e)\n",
                            residual,
                            std::sqrt(1e-3));
                   T83S_CHECK(c, residual <= std::sqrt(1e-3));

                   // ---- the assembled b really is the fixture's RHS -------------------
                   std::vector<double> b_host(dofs);
                   impl.b.copy_to(span<double>{b_host});
                   T83S_CHECK(c, max_abs_diff(b_host, b) == 0.0);

                   // ---- idempotence: the same call again gives the same numbers -------
                   // NOTE (research §7 T7): `solve()` rebuilds, and the rebuild re-fills
                   // `b` from the subsystems' gradients, so nothing has to be re-injected
                   // here — that is precisely why the RHS lives in `do_assemble()`.
                   gls.solve();

                   const auto iters2 = gls.last_solve_iterations();
                   std::vector<double> x2(dofs);
                   impl.x.copy_to(span<double>{x2});

                   T83S_LOG("[t83s] solve #2: iters %zu, |x - x'|inf = %.3e, "
                            "distributed identical %d\n",
                            static_cast<std::size_t>(iters2),
                            max_abs_diff(x2, x1),
                            static_cast<int>(T83sChain::distributed == x1));

                   T83S_CHECK(c, iters2 == iters1);
                   T83S_CHECK(c, max_abs_diff(x2, x1) <= 1e-14);
                   T83S_CHECK(c, T83sChain::distributed == x1);

                   // `solve()` rebuilt the system again (the norm wrappers are not the
                   // only rebuilders) -- one assemble per solve
                   T83S_CHECK(c, g_assemble_calls == 2);
               });
}

// ===========================================================================
//  CASE 3 — the debug dump: the reference's path/names + a Class-A re-import
// ===========================================================================
TEST_CASE("solve_dump_matches_assembled", "[luisa][solve]")
{
    using namespace t83s;
    namespace mm = uipc::backend::luisa;
    test_runtime();
    CaseScope scope{"solve_dump_matches_assembled"};

    with_world({.dump = true},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl, const std::string& ws)
               {
                   auto& c = scope.counter;

                   // ---- the config read (do_build) ------------------------------------
                   T83S_CHECK(c, impl.need_debug_dump);

                   const auto dir = expected_dump_dir(ws);
                   const auto A_path = dir + "A.0.0.mtx";  // frame 0, newton_iter 0
                   const auto b_path = dir + "b.0.0.mtx";
                   const auto x_path = dir + "x.0.0.mtx";

                   // nothing exists before the solve: the dump is solve()'s, not init()'s
                   T83S_CHECK(c, !fs::exists(A_path));
                   T83S_CHECK(c, !fs::exists(b_path));
                   T83S_CHECK(c, !fs::exists(x_path));

                   gls.solve();

                   T83S_LOG("[t83s] dump: A %d, b %d, x %d at %s\n",
                            static_cast<int>(fs::exists(A_path)),
                            static_cast<int>(fs::exists(b_path)),
                            static_cast<int>(fs::exists(x_path)),
                            dir.c_str());

                   // ---- the filenames and the directory (Class A: exact strings) -------
                   REQUIRE(fs::exists(A_path));
                   REQUIRE(fs::exists(b_path));
                   REQUIRE(fs::exists(x_path));
                   T83S_CHECK(c, fs::file_size(A_path) > 0u);

                   // the reference's dims/nnz line, from the assembled matrix
                   {
                       std::ifstream in{A_path};
                       std::string   line;
                       while(std::getline(in, line) && !line.empty() && line[0] == '%')
                       {
                       }
                       T83S_CHECK(c, line == "24 24 135");  // 15 blocks * 9 scalars
                   }

                   // ---- re-import A and b, compare to the assembled system (Class A) ---
                   lla::LBCOOMatrix<double, 3, 3> from_file;
                   REQUIRE(mm::import_matrix_market(from_file, std::string_view{A_path}));
                   T83S_CHECK(c, from_file.rows() == kNodes);
                   T83S_CHECK(c, from_file.cols() == kNodes);
                   T83S_CHECK(c, from_file.triplet_count() == impl.bcoo_A.triplet_count());

                   std::vector<lla::MatrixTriplet<double, 3, 3>> file_triplets(
                       from_file.triplet_count());
                   from_file.copy_to_triplets(file_triplets);
                   std::vector<lla::MatrixTriplet<double, 3, 3>> assembled(
                       impl.bcoo_A.triplet_count());
                   impl.bcoo_A.copy_to_triplets(assembled);

                   // the writer preserves the assembled order, so the sequences must
                   // agree entry for entry (an (i, j) multiset comparison would also
                   // allow a reordering, which the reference does not do)
                   std::size_t index_mismatch = 0;
                   double      worst_value    = 0.0;
                   for(std::size_t i = 0; i < assembled.size(); ++i)
                   {
                       if(file_triplets[i].row_index != assembled[i].row_index
                          || file_triplets[i].col_index != assembled[i].col_index)
                       {
                           ++index_mismatch;
                           continue;
                       }
                       for(int r = 0; r < kBlockDim; ++r)
                           for(int cc = 0; cc < kBlockDim; ++cc)
                               worst_value = std::max(worst_value,
                                                      std::abs(file_triplets[i].value(r, cc)
                                                               - assembled[i].value(r, cc)));
                   }

                   lla::LDenseVector<double> b_from_file;
                   REQUIRE(mm::import_vector_market(b_from_file, std::string_view{b_path}));
                   std::vector<double> b_file;
                   b_from_file.copy_to(b_file);
                   std::vector<double> b_assembled(impl.b.size());
                   impl.b.copy_to(span<double>{b_assembled});

                   T83S_LOG("[t83s] dump: A index mismatches %zu, max |dA| = %.3e, "
                            "max |db| = %.3e\n",
                            index_mismatch,
                            worst_value,
                            max_abs_diff(b_file, b_assembled));

                   T83S_CHECK(c, index_mismatch == 0u);
                   T83S_CHECK(c, worst_value == 0.0);  // Class A
                   T83S_CHECK(c, b_file.size() == b_assembled.size());
                   T83S_CHECK(c, max_abs_diff(b_file, b_assembled) == 0.0);  // Class A

                   // ---- x.mtx is the *raw* solution: dumped before distribute ---------
                   // (a dump taken after `distribute_solution()` would still hold the same
                   // numbers here because the fixture copies rather than transforms, so
                   // the x file is compared to `impl.x`, the pre-distribute state)
                   const auto x_file = read_vector(x_path);
                   std::vector<double> x_host(impl.x.size());
                   impl.x.copy_to(span<double>{x_host});
                   T83S_CHECK(c, x_file.size() == x_host.size());
                   T83S_CHECK(c, max_abs_diff(x_file, x_host) == 0.0);  // Class A

                   // and the distribute step did hand the same numbers to the subsystem
                   T83S_CHECK(c, max_abs_diff(T83sChain::distributed, x_host) == 0.0);
               });
}

// ===========================================================================
//  CASE 4 — `x = A^-1 b` against a dense host oracle on the *dumped* system
// ===========================================================================
TEST_CASE("solve_linear_system_matches_host_dense", "[luisa][solve]")
{
    using namespace t83s;
    namespace mm = uipc::backend::luisa;
    test_runtime();
    CaseScope scope{"solve_linear_system_matches_host_dense"};

    // `tol_rate = 1e-16`: the stopping rule is `|rz| <= tol*|rz_0|`, i.e.
    // `||r||/||b|| <= sqrt(tol) = 1e-8`, and the *solution* error is then
    // `cond(A) * 1e-8` (~1e-6 for this chain) — far above the 1e-10 gate.  At 1e-16 the
    // loop runs to its floating-point floor instead (<= max_iter = 2n), and the x
    // comparison is meaningful.
    with_world({.dump = true, .tol_rate = 1e-16},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl, const std::string& ws)
               {
                   auto& c = scope.counter;

                   gls.solve();  // assembles, dumps, solves, distributes

                   const auto dir    = expected_dump_dir(ws);
                   const auto A_path = dir + "A.0.0.mtx";
                   const auto b_path = dir + "b.0.0.mtx";
                   REQUIRE(fs::exists(A_path));
                   REQUIRE(fs::exists(b_path));

                   // ---- solve again, this time over the *file's* A and b -------------
                   // `import_matrix_market` overwrites `bcoo_A`, so the solver is now
                   // reading the dumped data and not the assembly buffer; the test bypassed
                   // the converter, so `triplet_count_dev` has to be re-uploaded (it is
                   // `spmv_dot`'s device-side count).
                   lla::LBCOOMatrix<double, 3, 3> A_file;
                   REQUIRE(mm::import_matrix_market(A_file, std::string_view{A_path}));
                   std::vector<lla::MatrixTriplet<double, 3, 3>> file_triplets(
                       A_file.triplet_count());
                   A_file.copy_to_triplets(file_triplets);
                   impl.bcoo_A.copy_from_triplets(
                       std::span<const lla::MatrixTriplet<double, 3, 3>>{file_triplets});
                   impl.triplet_count_dev.copy_from(
                       static_cast<IndexT>(impl.bcoo_A.triplet_count()));

                   lla::LDenseVector<double> b_file;
                   REQUIRE(mm::import_vector_market(b_file, std::string_view{b_path}));
                   std::vector<double> b_host;
                   b_file.copy_to(b_host);
                   impl.b.copy_from(span<const double>{b_host});

                   impl.solve_linear_system();

                   const auto iters = gls.last_solve_iterations();
                   std::vector<double> x(impl.x.size());
                   impl.x.copy_to(span<double>{x});

                   // ---- the dense oracle: the symmetric closure of the file's A ------
                   // `A_eff = U + Uᵀ − D` (research §7 T2): the diagonal blocks are stored
                   // whole, the strict upper blocks are mirrored *and transposed*.
                   std::vector<HostTriplet> table;
                   table.reserve(file_triplets.size());
                   for(const auto& t : file_triplets)
                   {
                       HostTriplet h;
                       h.row = t.row_index;
                       h.col = t.col_index;
                       for(int r = 0; r < kBlockDim; ++r)
                           for(int cc = 0; cc < kBlockDim; ++cc)
                               h.block.set(r, cc, t.value(r, cc));
                       table.push_back(h);
                   }
                   const auto A_dense = dense_A(table);

                   Eigen::Map<const Eigen::VectorXd> bv(b_host.data(),
                                                        static_cast<Eigen::Index>(b_host.size()));
                   Eigen::LDLT<Eigen::MatrixXd>      ldlt(A_dense);
                   REQUIRE(ldlt.info() == Eigen::Success);
                   Eigen::VectorXd x_ref = ldlt.solve(bv);
                   std::vector<double> x_ref_host(x_ref.data(), x_ref.data() + x_ref.size());

                   std::vector<double> diff(x.size());
                   for(std::size_t i = 0; i < x.size() && i < x_ref_host.size(); ++i)
                       diff[i] = x[i] - x_ref_host[i];
                   const double d_x = inf_norm(diff) / inf_norm(x_ref_host);
                   const double residual = relative_residual(A_dense, b_host, x);

                   T83S_LOG("[t83s] oracle: iters %zu, |x - x_ref|inf/|x_ref|inf = %.3e, "
                            "||b - Ax||/||b|| = %.3e\n",
                            static_cast<std::size_t>(iters),
                            d_x,
                            residual);

                   T83S_CHECK(c, x.size() == static_cast<std::size_t>(kDofs));
                   T83S_CHECK(c, iters > 0u);
                   T83S_CHECK(c, iters <= static_cast<SizeT>(2 * kDofs));
                   T83S_CHECK(c, d_x <= 1e-10);          // the plan's gate
                   T83S_CHECK(c, residual <= 1e-12);     // non-vacuous: it really converged
                   T83S_CHECK(c, inf_norm(x_ref_host) > 0.5);  // measured ~0.9, not ~0

                   // ---- the file's A is the assembled A: solving either gives the same x
                   // (the dumped values are Class-A identical to `bcoo_A`, so only the
                   // summation order inside `rbk_sym_spmv` could differ)
                   gls.solve();  // rebuild -> dump -> solve -> distribute
                   std::vector<double> x_assembled(impl.x.size());
                   impl.x.copy_to(span<double>{x_assembled});
                   const double d_paths = rel_diff(x_assembled, x_ref_host);
                   T83S_LOG("[t83s] oracle: |x(assembled) - x_ref|/scale = %.3e\n", d_paths);
                   T83S_CHECK(c, d_paths <= 1e-10);
               });
}

// ===========================================================================
//  CASE 5 — the two norm wrappers: `build_linear_system()` first, then the max
// ===========================================================================
TEST_CASE("solve_norms_match_host", "[luisa][solve]")
{
    using namespace t83s;
    test_runtime();
    CaseScope scope{"solve_norms_match_host"};

    with_world({},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl, const std::string&)
               {
                   auto& c = scope.counter;

                   // `diag_norm()` / `mass_norm()` are the entry points `SimEngine`'s
                   // AL `mu_scale_mode == "diag_norm"` path uses; both rebuild the system
                   // first (the reference calls `build_linear_system()` inside them).
                   const int base = g_assemble_calls;
                   T83S_CHECK(c, base == 0);  // `init_scene()` is the layout pass only

                   const auto diag = gls.diag_norm();
                   T83S_CHECK(c, g_assemble_calls == base + 1);  // <- the wrapper rebuilt
                   T83S_CHECK(c, T83sProbe::assemble_calls_at_norm == base + 1);

                   const auto mass = gls.mass_norm();
                   T83S_CHECK(c, g_assemble_calls == base + 2);

                   T83S_LOG("[t83s] norms: diag %.6f (chain %.6f, probe %.6f), "
                            "mass %.6f (chain %.6f, probe %.6f)\n",
                            diag,
                            kChainDiagNorm,
                            kProbeDiagNorm,
                            mass,
                            kChainMassNorm,
                            kProbeMassNorm);

                   // the max over the diag subsystems, in both directions:
                   // diag: chain (24) wins over probe (7.5); mass: probe (3.25) wins over
                   // chain (2.0) — so "the last subsystem wins" and "return the first"
                   // both fail
                   T83S_CHECK(c, diag == static_cast<Float>(std::max(kChainDiagNorm, kProbeDiagNorm)));
                   T83S_CHECK(c,
                              mass == static_cast<Float>(std::max(kChainMassNorm, kProbeMassNorm)));
                   T83S_CHECK(c, diag != static_cast<Float>(kProbeDiagNorm));
                   T83S_CHECK(c, mass != static_cast<Float>(kChainMassNorm));

                   // `diag_norm()` is the max over the *Impl*'s subsystems, reachable
                   // directly (the same number, without the rebuild)
                   T83S_CHECK(c, impl.diag_norm() == diag);
                   T83S_CHECK(c, impl.mass_norm() == mass);
               });
}

// ===========================================================================
//  CASE 6 — the empty system: reset first, then the early return
// ===========================================================================
TEST_CASE("solve_empty_scene", "[luisa][solve]")
{
    using namespace t83s;
    test_runtime();
    CaseScope scope{"solve_empty_scene"};

    // a scene whose geometry does not carry the marker: the fixture owns no dofs, so
    // `_update_subsystem_extent()` reports an empty system and `solve()` must return
    // *after* resetting the counter and *before* the dump/solve/distribute.
    with_world({.dump = true, .marker = false},
               [&](gg::GlobalLinearSystem& gls, gg::GlobalLinearSystem::Impl& impl, const std::string& ws)
               {
                   auto& c = scope.counter;

                   T83S_CHECK(c, gls.dof_count() == 0u);
                   T83S_CHECK(c, !T83sChain::enabled);

                   // a stale value a *previous* frame could have left behind
                   impl.last_solve_iterations = 12345;

                   REQUIRE_NOTHROW(gls.solve());

                   T83S_LOG("[t83s] empty: empty_system %d, last_solve_iterations %zu, "
                            "b %zu, x %zu\n",
                            static_cast<int>(impl.empty_system),
                            static_cast<std::size_t>(gls.last_solve_iterations()),
                            impl.b.size(),
                            impl.x.size());

                   // the reset happens first, and nothing sets it back
                   T83S_CHECK(c, gls.last_solve_iterations() == 0u);
                   T83S_CHECK(c, impl.empty_system);
                   T83S_CHECK(c, impl.b.size() == 0u);
                   T83S_CHECK(c, impl.x.size() == 0u);

                   // and no dump was written even though the config asks for one (the
                   // early return precedes `_dump_A_b()`), and no subsystem was assembled
                   const auto dir = expected_dump_dir(ws);
                   T83S_CHECK(c, !fs::exists(dir + "A.0.0.mtx"));
                   T83S_CHECK(c, !fs::exists(dir + "b.0.0.mtx"));
                   T83S_CHECK(c, !fs::exists(dir + "x.0.0.mtx"));
                   T83S_CHECK(c, g_assemble_calls == 0);

                   // the same call twice stays harmless
                   REQUIRE_NOTHROW(gls.solve());
                   T83S_CHECK(c, gls.last_solve_iterations() == 0u);
               });
}

// ===========================================================================
//  CASE 7 — plan §6.1's cross-backend gate: NOT REACHABLE here
// ===========================================================================
TEST_CASE("solve_from_cuda_mtx", "[luisa][solve]")
{
    using namespace t83s;
    test_runtime();
    SKIP(
        "no CUDA ground truth: --backend_cuda=y fails at configure time "
        "(src/backends/cuda/components.lua:86: unknown interface: raise()), so there is no "
        "CUDA-dumped A.mtx/b.mtx/x.mtx to compare against, and the parity runner cannot "
        "produce one either. The dump writer exists since T8.3b, so the gate is reachable "
        "on a machine with the CUDA backend; the same-product gate is covered in-process "
        "by solve_linear_system_matches_host_dense (Eigen dense oracle, 1e-10 relative) "
        "and solve_dump_matches_assembled (Class-A re-import).");
}
