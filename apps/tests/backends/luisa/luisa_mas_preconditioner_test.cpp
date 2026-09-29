/********************************************************************
 * @file luisa_mas_preconditioner_test.cpp
 * @brief T6.4c — the MAS (MultiLevel Additive Schwarz) preconditioner engine and the
 *        FEM MAS preconditioner (LuisaCompute backend).
 *
 * The reference backend is CUDA-only and cannot be host-included, so every oracle here
 * is derived independently *in this file*.  The engine's purpose (an approximate
 * inverse used as a preconditioner) is exercised end to end only by the Phase-8
 * solve, so the oracles are the honest, decisive ones available now:
 *
 *  (a) **hand-derived cluster systems.**  The engine is driven *directly* — its
 *      `init_neighbor`/`init_matrix`/`set_preconditioner` take the CSR adjacency, the
 *      partition maps and the BCOO triplets, so the test writes them by hand and then
 *      compares the engine's per-cluster inverse against
 *        * a closed form for the analytically known cases (a one-vertex cluster, a
 *          two-vertex cluster: the 6x6 inverse of the explicitly written block matrix,
 *          padded with the identity on the 42 unused DOFs), and
 *        * an **Eigen** inverse of the hand-rebuilt 48x48 dense cluster matrix
 *          (Eigen's LU with partial pivoting — a different algorithm from the ported
 *          column-oriented Gauss-Jordan) for a random SPD stencil.
 *      The rebuild follows the reference's own scatter rule (an upper triplet adds
 *      `H` to its block, a lower one adds `H^T` to the mirrored block; the gather
 *      transposes the mirrored blocks back) and its zero-diagonal pin, and the
 *      comparison symmetrizes the *diagonal* 3x3 blocks of the inverse, which is what
 *      the reference's post-elimination fix-up does.
 *  (b) **an independently computed level table / colouring.**  A plain union-find over
 *      the same hand-written adjacency (a completely different algorithm from the
 *      engine's bit-mask closure) derives the per-bank connected components, the
 *      `(bank, lowest lane)` component ranking, the level boundaries and the
 *      per-vertex level chain; every one of them is compared against the engine's
 *      own buffers for a 32-vertex chain and an 8x4 grid.
 *  (c) **the preconditioner contract** through a standalone `GlobalLinearSystem::Impl`
 *      and the *real* `FEMMASPreconditioner` on a real scene: `z = M^{-1} r` against an
 *      Eigen oracle built from the *assembled* matrix the impl produced (fine-level
 *      mode, i.e. plain block-Jacobi per cluster), `z^T r > 0`, determinism across
 *      runs, the `converged != 0` early-out, and two *differenced* statements: the
 *      MAS result is not the block-*diagonal* result (so it is not a silent fallback)
 *      and the coarse levels change the answer (so the hierarchy is really used).
 *  (d) **the config selection**: with `linear_system/fem_preconditioner = "mas"` the
 *      MAS preconditioner is valid and the diagonal one is explicitly not in play; with
 *      the config absent it is exactly the other way round (both measured, with the
 *      diagnostics printed — never a silent fallback).
 *  (e) a **device round** for the factored generic routines: the transitive closure
 *      (`mas_preconditioner_detail::transitive_closure_mask`, host vs a test-local
 *      kernel over random mask sets) and the unpartitioned-vertex diagonal fallback
 *      (`fem_mas_preconditioner_detail::{unpart_diag_inv_row,unpart_diag_apply_row}`),
 *      plus the pure integer laws (`sym_index`, `bank_align`, `num_levels`,
 *      `level1_begin`) against hand-written values.
 *
 * Coverage map (case -> oracle):
 *  1) cluster inverses, hand-derived + Eigen ................. mas_cluster_inverse
 *  2) the level table/colouring (chain + grid, union-find) ... mas_level_table
 *  3) the preconditioner contract on a real scene ............ mas_preconditioner_apply
 *  4) the config selection with "mas" ....................... mas_config_mas
 *  5) the config selection without it ....................... mas_config_default
 *  6) every factored routine, device vs host ................ mas_device_vs_host
 *  7) per-case assertion accounting ......................... mas_summary
 *
 * Shared-binary rules (cookbook §7.14 rule 6, §7.18 rule 6): the systems the port
 * registers are built for *every* scene of the whole test binary; the probe below only
 * reads them, and the MAS preconditioner is inert unless a scene selects it.
 *
 * WHAT THIS FILE CANNOT REACH (stated so nothing is implied)
 * ----------------------------------------------------------
 * The Newton/PCG loop (`GlobalLinearSystem::solve()`, `spmv`, `spmv_dot`) is T8.3, so
 * the preconditioner is driven through the standalone `GlobalLinearSystem::Impl`
 * exactly like T5.4b's FEM sibling; the engine's `apply` is therefore never called
 * from a solver, and `active_level_num` is set by the test to isolate the fine level.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/elastic_moduli.h>

#include <finite_element/mas_preconditioner_engine.h>
#include <finite_element/mas_preconditioner_engine_function.h>
#include <finite_element/fem_mas_preconditioner.h>
#include <finite_element/fem_mas_preconditioner_function.h>
#include <finite_element/fem_diag_preconditioner.h>
#include <finite_element/fem_diag_preconditioner_function.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/local_preconditioner.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>
#include <luisa_device/linear/algorithms.h>
// The test target's include path puts src/backends/none first: the shell header must
// be qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;
namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;
namespace mpd  = uipc::backend::luisa::mas_preconditioner_detail;
namespace mfd  = uipc::backend::luisa::fem_mas_preconditioner_detail;

// The ported surface lives in `uipc::backend::luisa`, so a test body sees the same
// spellings a ported system does.
using namespace uipc::backend::luisa;

namespace t64c
{
// ---------------------------------------------------------------------------
//  small helpers
// ---------------------------------------------------------------------------
/// `printf` + flush: the diagnostics must survive a crash of the case that emits
/// them (stdout is a pipe under the harness, i.e. block-buffered).
template <typename... Args>
inline void note(const char* fmt, Args&&... args)
{
    std::printf(fmt, std::forward<Args>(args)...);
    std::fflush(stdout);
}

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

/// Read an LC buffer back to the host (frame stream first: the engine's commands are
/// recorded there and a bare main-stream read would see stale bytes — cookbook §7.28
/// rule 10).
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

template <typename T>
std::vector<T> read_buffer(const lud::LDeviceBuffer<T>& buffer)
{
    return read_device(buffer.cview());
}

// ---------------------------------------------------------------------------
//  case accounting (the sibling files' CaseScope idiom)
// ---------------------------------------------------------------------------
std::vector<std::pair<std::string, int>> g_case_counts;

struct CaseScope
{
    std::string name;
    int         checks = 0;

    explicit CaseScope(std::string n)
        : name(std::move(n))
    {
    }
    ~CaseScope() { g_case_counts.emplace_back(name, checks); }
};

#define T64C_CHECK(counter, cond) \
    do \
    { \
        ++(counter).checks; \
        CHECK(cond); \
    } while(false)

// ---------------------------------------------------------------------------
//  the hand-written engine input
// ---------------------------------------------------------------------------
struct HandMesh
{
    int                              vert_num = 0;
    std::vector<std::pair<int, int>> edges;
};

/// The CSR adjacency + the partition maps for a mesh whose partition order is the
/// vertex order (the partition blocks are the first `blocks` banks of 16).
struct EngineInput
{
    int                       vert_num      = 0;
    int                       part_map_size = 0;
    std::vector<unsigned int> nbr_list;
    std::vector<unsigned int> nbr_start;
    std::vector<unsigned int> nbr_num;
    std::vector<int>          part_to_real;
    std::vector<int>          real_to_part;
};

[[nodiscard]] EngineInput make_engine_input(const HandMesh& m, int blocks)
{
    EngineInput in;
    in.vert_num = m.vert_num;
    std::vector<std::set<int>> nb(static_cast<size_t>(m.vert_num));
    for(auto& [a, b] : m.edges)
    {
        if(a != b && a >= 0 && b >= 0 && a < m.vert_num && b < m.vert_num)
        {
            nb[static_cast<size_t>(a)].insert(b);
            nb[static_cast<size_t>(b)].insert(a);
        }
    }
    in.nbr_start.assign(static_cast<size_t>(m.vert_num), 0u);
    in.nbr_num.assign(static_cast<size_t>(m.vert_num), 0u);
    for(int v = 0; v < m.vert_num; ++v)
    {
        in.nbr_start[static_cast<size_t>(v)] = static_cast<unsigned int>(in.nbr_list.size());
        in.nbr_num[static_cast<size_t>(v)] =
            static_cast<unsigned int>(nb[static_cast<size_t>(v)].size());
        for(int n : nb[static_cast<size_t>(v)])
            in.nbr_list.push_back(static_cast<unsigned int>(n));
    }
    in.part_map_size = blocks * MASPreconditionerEngine::BANKSIZE;
    in.part_to_real.assign(static_cast<size_t>(in.part_map_size), -1);
    in.real_to_part.assign(static_cast<size_t>(m.vert_num), -1);
    REQUIRE(m.vert_num <= in.part_map_size);
    for(int v = 0; v < m.vert_num; ++v)
    {
        in.part_to_real[static_cast<size_t>(v)] = v;
        in.real_to_part[static_cast<size_t>(v)] = v;
    }
    return in;
}

// ---------------------------------------------------------------------------
//  hand-written BCOO triplets (the assembled Hessian's shape: an *upper* block set)
// ---------------------------------------------------------------------------
struct BlockTriplet
{
    int             row = 0;
    int             col = 0;
    Eigen::Matrix3d H   = Eigen::Matrix3d::Zero();
};

/// Drive the real engine: `init_neighbor` -> `init_matrix` -> `set_preconditioner`.
/// `dof_offset` is 0, i.e. the triplets' block indices are vertex indices.
void run_engine(MASPreconditionerEngine&        engine,
                const EngineInput&              in,
                const std::vector<BlockTriplet>& ts,
                int                             dof_offset = 0)
{
    engine.init_neighbor(static_cast<int>(in.vert_num),
                         static_cast<int>(in.nbr_list.size()),
                         in.part_map_size,
                         span<const unsigned int>{in.nbr_list},
                         span<const unsigned int>{in.nbr_start},
                         span<const unsigned int>{in.nbr_num},
                         span<const int>{in.part_to_real},
                         span<const int>{in.real_to_part});
    engine.init_matrix();

    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> cols;
    std::vector<double>       vals;
    for(const auto& t : ts)
    {
        rows.push_back(t.row);
        cols.push_back(t.col);
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                vals.push_back(t.H(r, c));  // column-major, the BCOO block layout
    }
    lud::LDeviceBuffer<std::int32_t> row_buf;
    lud::LDeviceBuffer<std::int32_t> col_buf;
    lud::LDeviceBuffer<double>       val_buf;
    if(!ts.empty())
    {
        row_buf.copy_from(span<const std::int32_t>{rows});
        col_buf.copy_from(span<const std::int32_t>{cols});
        val_buf.copy_from(span<const double>{vals});
    }
    engine.set_preconditioner(val_buf.cview(), row_buf.cview(), col_buf.cview(), dof_offset);
    gg::frame_stream().synchronize();
}

// ---------------------------------------------------------------------------
//  the cluster-matrix oracles
// ---------------------------------------------------------------------------
using Mat48 = Eigen::Matrix<double, 48, 48>;
using Blk3  = Eigen::Matrix3d;

/// The reference's `sym_index(row, col)`.
[[nodiscard]] inline int oracle_sym_index(int row, int col)
{
    return 16 * row - row * (row + 1) / 2 + col;
}

/// The 48x48 matrix the *level-0* cluster `cluster_id` is assembled into:
/// per triplet, `part(row) <= part(col)` adds `H` to the upper block, a lower one adds
/// `H^T` to the mirrored block (the reference's two scatter-rule branches); the matrix
/// is then the symmetric reconstruction the invert kernel gathers, with the
/// zero-diagonal pin applied exactly where the kernel applies it.
[[nodiscard]] Mat48 oracle_level0_dense(const std::vector<BlockTriplet>& ts,
                                        int                             cluster_id,
                                        const std::vector<int>&         real_to_part)
{
    Blk3 stored[16][16];
    for(int br = 0; br < 16; ++br)
        for(int bc = 0; bc < 16; ++bc)
            stored[br][bc] = Blk3::Zero();

    for(const auto& t : ts)
    {
        const int pr = real_to_part[static_cast<size_t>(t.row)];
        const int pc = real_to_part[static_cast<size_t>(t.col)];
        if(pr < 0 || pc < 0)
            continue;
        if(pr / 16 != cluster_id || pc / 16 != cluster_id)
            continue;  // a cross-bank triplet walks up instead (see the coarse oracle)
        if(pr <= pc)
            stored[pr % 16][pc % 16] += t.H;
        else
            stored[pc % 16][pr % 16] += t.H.transpose();
    }

    Mat48 D = Mat48::Zero();
    for(int br = 0; br < 16; ++br)
    {
        for(int bc = br; bc < 16; ++bc)
        {
            D.block<3, 3>(3 * br, 3 * bc) = stored[br][bc];
            D.block<3, 3>(3 * bc, 3 * br) = stored[br][bc].transpose();
        }
    }
    // the reference's `if(row == col && s_mat(row,col) == 0) s_mat(row,col) = 1.0`
    for(int k = 0; k < 16; ++k)
        for(int i = 0; i < 3; ++i)
            if(D(3 * k + i, 3 * k + i) == 0.0)
                D(3 * k + i, 3 * k + i) = 1.0;
    return D;
}

/// The expected *stored* cluster inverse: Eigen's inverse of the rebuilt dense matrix,
/// then the reference's post-elimination 3x3-block fix-up
/// (`if(col%3 < 2) M[col+1][col] = M[col][col+1]; else M[col][col-2] = M[col-2][col]`)
/// applied to the inverse.
[[nodiscard]] Mat48 oracle_cluster_inverse(const Mat48& D)
{
    Mat48 inv = D.inverse();
    for(int k = 0; k < 16; ++k)
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                if(i > j)
                    inv(3 * k + i, 3 * k + j) = inv(3 * k + j, 3 * k + i);
    return inv;
}

/// The engine's stored (upper-triangle) blocks of cluster `cluster_id`, as a 48x48
/// dense matrix (the lower blocks are the transposes, exactly as the kernel stores the
/// reference's symmetric array).
template <typename Scalar>
[[nodiscard]] Mat48 read_cluster_matrix_blocks(const std::vector<Scalar>& flat, int cluster_id)
{
    Mat48 D = Mat48::Zero();
    for(int br = 0; br < 16; ++br)
    {
        for(int bc = br; bc < 16; ++bc)
        {
            const int base = oracle_sym_index(br, bc) * 9;
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                {
                    const auto v = static_cast<double>(
                        flat[static_cast<size_t>(cluster_id)
                                 * MASPreconditionerEngine::CLUSTER_ENTRIES
                             + static_cast<size_t>(base + j * 3 + i)]);
                    D(3 * br + i, 3 * bc + j) = v;
                    D(3 * bc + j, 3 * br + i) = v;
                }
        }
    }
    return D;
}

/// The same reader for an `LMatrix3x3` cluster buffer (the port's fallback shape).
[[nodiscard]] Mat48 read_cluster_inverse_blocks(const std::vector<float>& flat,
                                               int                       cluster_id)
{
    Mat48 D = Mat48::Zero();
    for(int br = 0; br < 16; ++br)
    {
        for(int bc = br; bc < 16; ++bc)
        {
            const int base = oracle_sym_index(br, bc) * 9;
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                {
                    const auto v =
                        static_cast<double>(flat[static_cast<size_t>(cluster_id)
                                                 * MASPreconditionerEngine::CLUSTER_ENTRIES
                                                 + static_cast<size_t>(base + j * 3 + i)]);
                    D(3 * br + i, 3 * bc + j) = v;
                    D(3 * bc + j, 3 * br + i) = v;
                }
        }
    }
    return D;
}
}  // namespace t64c

// ===========================================================================
//  CASE 1 — the cluster inverses against hand-derived systems and Eigen
// ===========================================================================
namespace t64c
{
TEST_CASE("mas_cluster_inverse", "[luisa][mas_preconditioner]")
{
    CaseScope c{"cluster_inverse"};
    test_runtime();

    // ---- (a) a one-vertex cluster: a closed form -----------------------------
    // `H = diag(2,4,8)` in the single 3x3 block; the 42 unused DOFs are pinned to the
    // identity by the reference's zero-diagonal rule, so the cluster inverse is
    // `blockdiag(H^-1, I_42)` — no numerical tolerance needed beyond float storage.
    {
        HandMesh mesh;
        mesh.vert_num = 1;
        auto in       = make_engine_input(mesh, 1);

        BlockTriplet t;
        t.row = 0;
        t.col = 0;
        t.H   = Blk3::Zero();
        t.H(0, 0) = 2.0;
        t.H(1, 1) = 4.0;
        t.H(2, 2) = 8.0;

        MASPreconditionerEngine engine;
        run_engine(engine, in, {t});

        CHECK(engine.is_initialized());
        CHECK(engine.m_total_nodes == 1);
        CHECK(engine.m_total_map_nodes == 16);
        CHECK(engine.m_level_num == 2);
        CHECK(engine.level_size(1).x == 1);
        CHECK(engine.level_size(1).y == 16);
        CHECK(engine.num_cluster_blocks() == 2);

        const auto flat = read_buffer(engine.cluster_inverses);
        const auto inv  = read_cluster_inverse_blocks(flat, 0);

        Mat48 expected = Mat48::Identity();
        expected(0, 0)  = 0.5;
        expected(1, 1)  = 0.25;
        expected(2, 2)  = 0.125;

        double worst = 0.0;
        for(int r = 0; r < 48; ++r)
            for(int k = 0; k < 48; ++k)
                worst = std::max(worst, std::abs(inv(r, k) - expected(r, k)));
        note("[t64c] cluster_inverse(1 vertex): |inv - blockdiag(H^-1,I42)|max = %.3e\n",
                    worst);
        T64C_CHECK(c, worst <= 1e-6);

        // the coarse level of a one-component bank: the aggregate of the single block
        // is `H` itself, so the coarse cluster inverse is the same closed form
        const auto inv1 = read_cluster_inverse_blocks(flat, 1);
        double     worst1 = 0.0;
        for(int r = 0; r < 48; ++r)
            for(int k = 0; k < 48; ++k)
                worst1 = std::max(worst1, std::abs(inv1(r, k) - expected(r, k)));
        note("[t64c] cluster_inverse(1 vertex): coarse |inv - expected|max = %.3e\n",
                    worst1);
        T64C_CHECK(c, worst1 <= 1e-6);
    }

    // ---- (b) a two-vertex cluster: the explicit 6x6 inverse -------------------
    // Three triplets in both orders (the assembled upper set *and* its mirror, the
    // shape a symmetric BCOO assembly has), so both scatter branches are exercised:
    // the stored (0,1) block ends up as `2C`.
    {
        HandMesh mesh;
        mesh.vert_num = 2;
        mesh.edges    = {{0, 1}};
        auto in       = make_engine_input(mesh, 1);

        Blk3 A = Blk3::Zero(), B = Blk3::Zero(), C = Blk3::Zero();
        A(0, 0) = 2.0; A(1, 1) = 3.0; A(2, 2) = 4.0;
        B(0, 0) = 5.0; B(1, 1) = 6.0; B(2, 2) = 7.0;
        C(0, 0) = 0.5; C(1, 1) = 0.25; C(2, 2) = 0.125;

        std::vector<BlockTriplet> ts{{0, 0, A}, {1, 1, B}, {0, 1, C}, {1, 0, C}};

        MASPreconditionerEngine engine;
        run_engine(engine, in, ts);

        CHECK(engine.m_level_num == 2);
        CHECK(engine.level_size(1).x == 1);  // one connected component in the bank
        CHECK(engine.level_size(1).y == 16);

        const auto flat = read_buffer(engine.cluster_inverses);
        const auto inv  = read_cluster_inverse_blocks(flat, 0);

        // the hand-written 6x6 system, inverted with Eigen on the *explicit* numbers
        Eigen::Matrix<double, 6, 6> M6 = Eigen::Matrix<double, 6, 6>::Zero();
        M6.block<3, 3>(0, 0)           = A;
        M6.block<3, 3>(3, 3)           = B;
        M6.block<3, 3>(0, 3)           = 2.0 * C;
        M6.block<3, 3>(3, 0)           = 2.0 * C.transpose();
        const auto inv6                = M6.inverse();

        double worst = 0.0, scale = 0.0;
        for(int br = 0; br < 2; ++br)
            for(int bc = br; bc < 2; ++bc)
                for(int i = 0; i < 3; ++i)
                    for(int j = 0; j < 3; ++j)
                    {
                        const double want = inv6(3 * br + i, 3 * bc + j);
                        worst = std::max(worst, std::abs(inv(3 * br + i, 3 * bc + j) - want));
                        scale = std::max(scale, std::abs(want));
                    }
        note("[t64c] cluster_inverse(2 vertices): |inv6 - hand 6x6 inverse|max = %.3e "
                    "(scale %.3e)\n",
                    worst,
                    scale);
        T64C_CHECK(c, scale > 1.0e-2);  // non-vacuous
        T64C_CHECK(c, worst <= 1e-6);

        // the unused DOFs stay the identity
        double worst_id = 0.0;
        for(int k = 2; k < 16; ++k)
            for(int i = 0; i < 3; ++i)
                for(int j = 0; j < 3; ++j)
                    worst_id = std::max(worst_id,
                                        std::abs(inv(3 * k + i, 3 * k + j)
                                                 - (i == j ? 1.0 : 0.0)));
        T64C_CHECK(c, worst_id == 0.0);

        // the *coarse* level: with a single component in the bank, the reference's
        // `prefix == 1` path aggregates the whole cluster into one coarse node, whose
        // 3x3 block is the sum of every block of the fine matrix
        const auto S = A + B + 2.0 * C + 2.0 * C.transpose();
        const auto invS = S.inverse();
        const auto inv1 = read_cluster_inverse_blocks(flat, 1);
        double     worst_coarse = 0.0;
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                worst_coarse =
                    std::max(worst_coarse, std::abs(inv1(i, j) - invS(i, j)));
        note("[t64c] cluster_inverse(2 vertices): coarse |inv(aggregate) - hand|max "
                    "= %.3e\n",
                    worst_coarse);
        T64C_CHECK(c, worst_coarse <= 1e-6);
    }

    // ---- (c) a random SPD stencil (8 vertices, one cluster) ------------------
    // The triplets are the *upper* block set of a random SPD 24x24 system (the shape
    // the assembled BCOO has), and the oracle is Eigen's inverse of the rebuilt dense
    // cluster matrix.
    {
        HandMesh mesh;
        mesh.vert_num = 8;
        for(int i = 0; i < 8; ++i)
        {
            mesh.edges.push_back({i, (i + 1) % 8});
            mesh.edges.push_back({i, (i + 3) % 8});
        }
        auto in = make_engine_input(mesh, 1);

        std::mt19937                           rng(64u);
        std::uniform_real_distribution<double> val(-0.5, 0.5);
        Eigen::Matrix<double, 24, 24>          dense = Eigen::Matrix<double, 24, 24>::Zero();
        for(int r = 0; r < 24; ++r)
            for(int k = 0; k <= r; ++k)
                dense(r, k) = val(rng);
        Eigen::Matrix<double, 24, 24> spd = dense * dense.transpose();
        for(int r = 0; r < 24; ++r)
            spd(r, r) += 3.0;  // well conditioned

        std::vector<BlockTriplet> ts;
        for(int br = 0; br < 8; ++br)
            for(int bc = br; bc < 8; ++bc)
            {
                BlockTriplet t;
                t.row = br;
                t.col = bc;
                t.H   = spd.block<3, 3>(3 * br, 3 * bc);
                ts.push_back(t);
            }

        MASPreconditionerEngine engine;
        run_engine(engine, in, ts);

        const auto flat = read_buffer(engine.cluster_inverses);
        const auto inv  = read_cluster_inverse_blocks(flat, 0);

        const auto D       = oracle_level0_dense(ts, 0, in.real_to_part);
        const auto expected = oracle_cluster_inverse(D);
        // the rebuilt dense matrix must be the hand-written SPD system padded with the
        // pinned identity on the unused DOFs (a cross-check of the oracle itself)
        double worst_dense = 0.0;
        for(int r = 0; r < 24; ++r)
            for(int k = 0; k < 24; ++k)
                worst_dense = std::max(worst_dense, std::abs(D(r, k) - spd(r, k)));
        for(int k = 24; k < 48; ++k)
            worst_dense = std::max(worst_dense, std::abs(D(k, k) - 1.0));
        note("[t64c] cluster_inverse(stencil): rebuilt dense matrix vs hand SPD "
                    "worst %.3e\n",
                    worst_dense);
        T64C_CHECK(c, worst_dense == 0.0);

        double worst = 0.0, inv_scale = 0.0, resid = 0.0;
        for(int r = 0; r < 48; ++r)
        {
            for(int k = 0; k < 48; ++k)
            {
                worst = std::max(worst, std::abs(inv(r, k) - expected(r, k)));
                // the engine's stored inverse is the cluster inverse padded with the
                // identity on the unused DOFs, so the non-vacuity scale is taken over
                // the cluster's own 24x24 region
                if(r < 24 && k < 24)
                    inv_scale = std::max(inv_scale, std::abs(expected(r, k)));
            }
            double acc = 0.0;
            for(int k = 0; k < 48; ++k)
                acc += D(r, k) * inv(k, r);
            resid = std::max(resid, std::abs(acc - 1.0));
        }
        note("[t64c] cluster_inverse(stencil): |inv - Eigen inverse|max = %.3e "
                    "(|inv|max %.3e, |D*inv - I| diag max %.3e)\n",
                    worst,
                    inv_scale,
                    resid);
        T64C_CHECK(c, inv_scale > 1.0e-2);
        T64C_CHECK(c, worst <= 1e-5);   // the inverse is stored in float (GIPC parity)
        T64C_CHECK(c, resid <= 1e-6);

        // every stored block must be a real 3x3 block: no all-zero block may hide a
        // missing write (the 8-vertex stencil's blocks are all non-trivial)
        int zero_blocks = 0;
        for(int br = 0; br < 8; ++br)
            for(int bc = br; bc < 8; ++bc)
                if(inv.block<3, 3>(3 * br, 3 * bc).cwiseAbs().maxCoeff() < 1e-12)
                    ++zero_blocks;
        T64C_CHECK(c, zero_blocks == 0);
    }

    // ---- (d) a *zero* cluster: the reference's singular rule -----------------
    // No triplet at all -> the whole 48x48 is zero -> the kernel pins every diagonal
    // scalar to 1 -> the inverse is the exact identity (the reference's deterministic
    // answer for a singular cluster, cookbook §7.28 rule 3's sibling behaviour).
    {
        HandMesh mesh;
        mesh.vert_num = 1;
        auto in       = make_engine_input(mesh, 1);

        MASPreconditionerEngine engine;
        run_engine(engine, in, {});

        const auto flat = read_buffer(engine.cluster_inverses);
        const auto inv  = read_cluster_inverse_blocks(flat, 0);
        double     worst = 0.0;
        for(int r = 0; r < 48; ++r)
            for(int k = 0; k < 48; ++k)
                worst = std::max(worst, std::abs(inv(r, k) - (r == k ? 1.0 : 0.0)));
        note("[t64c] cluster_inverse(zero cluster): |inv - I|max = %.3e\n", worst);
        T64C_CHECK(c, worst == 0.0);
    }
}
}  // namespace t64c

// ===========================================================================
//  CASE 2 — the level table / colouring against an independent union-find oracle
// ===========================================================================
namespace t64c
{
/// The independently computed MAS hierarchy: a plain union-find over the same
/// hand-written adjacency (a different algorithm from the engine's bit-mask closure),
/// plus the reference's own lane/`bank` bookkeeping rules written out here.
struct LevelOracle
{
    int              level_num = 0;
    std::vector<int> level_begin;  // [0 .. level_num]
    std::vector<int> level_count;  // [0 .. level_num]; level_count[0] = part_map_size
    std::vector<int> fine_mask;    // [vert_num] post-closure (what the engine stores)
    std::vector<int> pre_mask;     // [part_map_size] the per-position masks *before* the closure
    std::vector<int> bank_count;   // [part_map_size/16]
    std::vector<int> going_next;   // the engine's buffer contents ([0, level_begin[level_num]))
    /// anc[l][v]: the *id* of vertex `v`'s ancestor at level l (l >= 1);
    /// `anc[0][v]` is the partition *position* (the fine node).
    std::vector<std::vector<int>> anc;
    /// The per-vertex level chain, as the engine's `coarse_tables` holds it.
    std::vector<int> table;  // [vert_num * MAX_LEVELS]
};

[[nodiscard]] LevelOracle build_level_oracle(const EngineInput& in, const HandMesh& m)
{
    const int  P = in.part_map_size;
    LevelOracle o;
    int         L = 1;
    {
        int sz = mpd::bank_align(P);
        while(sz > 16)
        {
            sz /= 16;
            L++;
            sz = mpd::bank_align(sz);
        }
        L++;
    }
    o.level_num = std::min(L, static_cast<int>(MASPreconditionerEngine::MAX_LEVELS));
    o.level_begin.assign(static_cast<size_t>(o.level_num) + 1u, 0);
    o.level_count.assign(static_cast<size_t>(o.level_num) + 1u, 0);
    o.level_count[0] = P;
    o.level_begin[1] = mpd::level1_begin(P, m.vert_num);

    // ---- the fine connect masks ---------------------------------------------
    // The masks *live* in the partition-position space (the own bit is the position's
    // lane) but the engine stores them at the **real vertex index**, and the closure
    // merges a whole bank, so every vertex of a component carries the component's
    // whole mask.
    std::vector<int> pre_mask(static_cast<size_t>(P), 0);
    for(int p = 0; p < P; ++p)
    {
        const int v = in.part_to_real[static_cast<size_t>(p)];
        if(v < 0)
            continue;
        int mask = 1 << (p % 16);
        for(unsigned int i = in.nbr_start[static_cast<size_t>(v)];
            i < in.nbr_start[static_cast<size_t>(v)] + in.nbr_num[static_cast<size_t>(v)];
            ++i)
        {
            const int q = in.real_to_part[in.nbr_list[static_cast<size_t>(i)]];
            if(q >= 0 && q / 16 == p / 16)
                mask |= 1 << (q % 16);
        }
        pre_mask[static_cast<size_t>(p)] = mask;
    }
    o.pre_mask = pre_mask;
    o.fine_mask.assign(static_cast<size_t>(m.vert_num), 0);

    o.anc.assign(static_cast<size_t>(o.level_num), std::vector<int>(static_cast<size_t>(m.vert_num), -1));
    for(int v = 0; v < m.vert_num; ++v)
        o.anc[0][static_cast<size_t>(v)] = in.real_to_part[static_cast<size_t>(v)];

    // ---- the components of the level-(l-1) space -> the level-l ids ----
    // `comp[l]` maps a level-(l-1) node id to its level-l id; the level-1 components
    // come from the *fine* masks (the position space), the higher ones from the
    // level-(l-1) ids' own masks (the bank test is on the level-(l-1) ids).
    std::vector<std::vector<int>> comp(static_cast<size_t>(o.level_num) + 1u);
    comp[1] = std::vector<int>(static_cast<size_t>(P), -1);

    auto components_of = [&](int n, const std::vector<std::pair<int, int>>& adj) noexcept
    {
        std::vector<int> parent(static_cast<size_t>(n));
        for(int i = 0; i < n; ++i)
            parent[static_cast<size_t>(i)] = i;
        std::function<int(int)> find = [&](int i) {
            while(parent[static_cast<size_t>(i)] != i)
            {
                parent[static_cast<size_t>(i)] = parent[static_cast<size_t>(parent[static_cast<size_t>(i)])];
                i = parent[static_cast<size_t>(i)];
            }
            return i;
        };
        for(auto& [a, b] : adj)
        {
            if(a < 0 || b < 0)
                continue;
            const int ra = find(a), rb = find(b);
            if(ra != rb)
                parent[static_cast<size_t>(std::max(ra, rb))] = std::min(ra, rb);
        }
        // the component index of an id is the rank of its component ordered by
        // (bank, lowest id) — which is exactly the exclusive scan of the per-bank
        // elected-lane counts the engine performs
        std::vector<int> lowest(static_cast<size_t>(n), -1);
        for(int i = 0; i < n; ++i)
        {
            const int r = find(i);
            if(lowest[static_cast<size_t>(r)] < 0 || i < lowest[static_cast<size_t>(r)])
                lowest[static_cast<size_t>(r)] = i;
        }
        std::vector<int> roots;
        for(int i = 0; i < n; ++i)
            if(find(i) == i)
                roots.push_back(i);
        std::sort(roots.begin(), roots.end(), [&](int a, int b) {
            const auto ka = std::pair<int, int>{lowest[static_cast<size_t>(a)] / 16,
                                                lowest[static_cast<size_t>(a)]};
            const auto kb = std::pair<int, int>{lowest[static_cast<size_t>(b)] / 16,
                                                lowest[static_cast<size_t>(b)]};
            return ka < kb;
        });
        std::vector<int> out(static_cast<size_t>(n), -1);
        for(size_t k = 0; k < roots.size(); ++k)
            for(int i = 0; i < n; ++i)
                if(find(i) == roots[k])
                    out[static_cast<size_t>(i)] = static_cast<int>(k);
        return out;
    };

    // the level-1 components: the fine masks' connectivity within each fine bank
    {
        std::vector<std::pair<int, int>> adj;
        for(const auto& e : m.edges)
        {
            const int pa = in.real_to_part[static_cast<size_t>(e.first)];
            const int pb = in.real_to_part[static_cast<size_t>(e.second)];
            if(pa >= 0 && pb >= 0 && pa / 16 == pb / 16)
                adj.emplace_back(pa, pb);
        }
        comp[1]          = components_of(P, adj);
        o.level_count[1] = 1 + *std::max_element(comp[1].begin(), comp[1].end());
        // per fine bank: the number of components whose *lowest* position is in it
        // (the reference's `prefix_original`, the elected-lane count of the bank)
        o.bank_count.assign(static_cast<size_t>(P / 16), 0);
        for(int p = 0; p < P; ++p)
        {
            if(comp[1][static_cast<size_t>(p)] < 0)
                continue;
            bool first = true;
            for(int q = 0; q < p; ++q)
                if(comp[1][static_cast<size_t>(q)] == comp[1][static_cast<size_t>(p)])
                {
                    first = false;
                    break;
                }
            if(first)
                ++o.bank_count[static_cast<size_t>(p / 16)];
        }
        // the post-closure masks (a component's members share their whole bit set)
        for(int p = 0; p < P; ++p)
        {
            const int v = in.part_to_real[static_cast<size_t>(p)];
            if(v < 0)
                continue;
            int mask = 0;
            for(int q = 0; q < P; ++q)
                if(comp[1][static_cast<size_t>(q)] == comp[1][static_cast<size_t>(p)])
                    mask |= pre_mask[static_cast<size_t>(q)];
            o.fine_mask[static_cast<size_t>(v)] = mask;
        }
        for(int v = 0; v < m.vert_num; ++v)
            o.anc[1][static_cast<size_t>(v)] = comp[1][static_cast<size_t>(in.real_to_part[static_cast<size_t>(v)])];
    }

    // the higher levels: the components of the level-(l) id space (l = 1 .. level_num-1)
    for(int l = 1; l + 1 <= o.level_num; ++l)
    {
        const int n = o.level_count[static_cast<size_t>(l)];
        std::vector<std::pair<int, int>> adj;
        for(const auto& e : m.edges)
        {
            const int i = o.anc[static_cast<size_t>(l)][static_cast<size_t>(e.first)];
            const int j = o.anc[static_cast<size_t>(l)][static_cast<size_t>(e.second)];
            if(i >= 0 && j >= 0 && i / 16 == j / 16)
                adj.emplace_back(i, j);
        }
        comp[static_cast<size_t>(l) + 1u] = components_of(n, adj);
        o.level_count[static_cast<size_t>(l) + 1u] =
            1 + *std::max_element(comp[static_cast<size_t>(l) + 1u].begin(),
                                  comp[static_cast<size_t>(l) + 1u].end());
        if(l + 1 <= o.level_num - 1)
        {
            for(int v = 0; v < m.vert_num; ++v)
                o.anc[static_cast<size_t>(l) + 1u][static_cast<size_t>(v)] =
                    comp[static_cast<size_t>(l) + 1u]
                        [static_cast<size_t>(o.anc[static_cast<size_t>(l)][static_cast<size_t>(v)])];
        }
    }

    // ---- the level boundaries ----
    for(int l = 1; l < o.level_num; ++l)
        o.level_begin[static_cast<size_t>(l) + 1u] =
            o.level_begin[static_cast<size_t>(l)]
            + mpd::bank_align(o.level_count[static_cast<size_t>(l)]);

    // ---- going_next ----
    o.going_next.assign(static_cast<size_t>(o.level_begin[static_cast<size_t>(o.level_num)]), -1);
    // NOTE: the engine's `going_next` (and `coarse_space_tables`) level-0 region is
    // indexed by the **real vertex index**, while the connect masks are indexed by
    // the **partition position** — the two index spaces only coincide for an identity
    // partition, which is why the oracle keeps them apart (see the reversed-partition
    // case below).
    for(int v = 0; v < m.vert_num; ++v)
    {
        if(o.anc[0][static_cast<size_t>(v)] >= 0)
            o.going_next[static_cast<size_t>(v)] =
                o.level_begin[1] + o.anc[1][static_cast<size_t>(v)];
    }
    for(int l = 1; l < o.level_num; ++l)
    {
        for(int id = 0; id < o.level_count[static_cast<size_t>(l)]; ++id)
            o.going_next[static_cast<size_t>(o.level_begin[static_cast<size_t>(l)] + id)] =
                o.level_begin[static_cast<size_t>(l) + 1u]
                + comp[static_cast<size_t>(l) + 1u][static_cast<size_t>(id)];
    }

    // ---- the per-vertex level chain (the engine's `coarse_tables`) ----
    o.table.assign(static_cast<size_t>(m.vert_num) * MASPreconditionerEngine::MAX_LEVELS, 0);
    for(int v = 0; v < m.vert_num; ++v)
    {
        const int pos = o.anc[0][static_cast<size_t>(v)];
        if(pos < 0)
            continue;
        int cur = o.going_next[static_cast<size_t>(v)];  // the level-1 position
        for(int l = 1; l <= o.level_num - 1; ++l)
        {
            o.table[static_cast<size_t>(v) * MASPreconditionerEngine::MAX_LEVELS
                    + static_cast<size_t>(l - 1)] = cur;
            cur = o.going_next[static_cast<size_t>(cur)];
        }
    }
    return o;
}

/// Compare the engine's own hierarchy buffers against the oracle.
void check_level_structure(CaseScope&                c,
                           MASPreconditionerEngine&  engine,
                           const EngineInput&        in,
                           const LevelOracle&        o,
                           std::string_view          label)
{
    const int P = in.part_map_size;

    CHECK(engine.level_count() == o.level_num);
    T64C_CHECK(c, engine.m_total_num_clusters == o.level_begin[static_cast<size_t>(o.level_num)]);
    T64C_CHECK(c, engine.m_total_map_nodes == P);
    T64C_CHECK(c, engine.m_total_nodes == in.vert_num);

    // the level sizes (count/offset per level)
    const auto level_sizes = read_buffer(engine.level_sizes);
    REQUIRE(level_sizes.size() >= static_cast<size_t>(o.level_num) + 1u);
    for(int l = 1; l <= o.level_num; ++l)
    {
        T64C_CHECK(c, level_sizes[static_cast<size_t>(2 * l)] == o.level_count[static_cast<size_t>(l)]);
        T64C_CHECK(c, level_sizes[static_cast<size_t>(2 * l + 1)]
                          == o.level_begin[static_cast<size_t>(l)]);
    }
    T64C_CHECK(c, static_cast<int>(level_sizes.size()) == 2 * (o.level_num + 1));

    // the fine connect masks + the per-bank component counts
    const auto masks  = read_buffer(engine.fine_connect_masks);
    const auto prefix = read_buffer(engine.prefix_original);
    int        mask_mismatch = 0;
    for(int v = 0; v < in.vert_num; ++v)
    {
        const auto want = static_cast<std::uint32_t>(o.fine_mask[static_cast<size_t>(v)]);
        mask_mismatch += (masks[static_cast<size_t>(v)] != want) ? 1 : 0;
    }
    note("[t64c] level_table(%s): fine masks %d/%d equal (mismatches %d)\n",
                std::string{label}.c_str(),
                in.vert_num - mask_mismatch,
                in.vert_num,
                mask_mismatch);
    T64C_CHECK(c, mask_mismatch == 0);

    int bank_mismatch = 0;
    for(size_t b = 0; b + 1 <= o.bank_count.size(); ++b)
        bank_mismatch += (prefix[b] == o.bank_count[b]) ? 0 : 1;
    T64C_CHECK(c, bank_mismatch == 0);
    T64C_CHECK(c, static_cast<int>(o.bank_count.size()) == P / 16);

    // going_next: the level-0 region (indexed by the *real* vertex index) and every
    // level region (indexed by the level's own positions)
    const auto gn = read_buffer(engine.going_next);
    REQUIRE(gn.size() >= o.going_next.size());
    int gn_mismatch = 0;
    for(size_t i = 0; i < o.going_next.size(); ++i)
        gn_mismatch += (gn[i] == o.going_next[i]) ? 0 : 1;
    note("[t64c] level_table(%s): going_next %d/%d equal\n",
                std::string{label}.c_str(),
                static_cast<int>(o.going_next.size()) - gn_mismatch,
                static_cast<int>(o.going_next.size()));
    T64C_CHECK(c, gn_mismatch == 0);

    // the per-vertex level chain
    const auto table = read_buffer(engine.coarse_tables);
    REQUIRE(table.size() == o.table.size());
    int table_mismatch = 0;
    for(size_t i = 0; i < o.table.size(); ++i)
        table_mismatch += (table[i] == o.table[i]) ? 0 : 1;
    note("[t64c] level_table(%s): coarse_tables %d/%d equal (%d levels, clusters %d)\n",
                std::string{label}.c_str(),
                static_cast<int>(o.table.size()) - table_mismatch,
                static_cast<int>(o.table.size()),
                o.level_num,
                engine.m_total_num_clusters);
    T64C_CHECK(c, table_mismatch == 0);

    // the level-count law itself, pinned on hand-written inputs
    T64C_CHECK(c, engine.m_level_num == o.level_num);
}

TEST_CASE("mas_level_table", "[luisa][mas_preconditioner]")
{
    CaseScope c{"level_table"};
    test_runtime();

    // ---- (a) a 32-vertex chain (2 banks), identity partition ----------------
    {
        HandMesh mesh;
        mesh.vert_num = 32;
        for(int i = 0; i + 1 < 32; ++i)
            mesh.edges.push_back({i, i + 1});
        auto in = make_engine_input(mesh, 2);
        auto o  = build_level_oracle(in, mesh);

        // the hand-derived expectations for this mesh
        T64C_CHECK(c, o.level_num == 3);
        T64C_CHECK(c, o.level_begin[1] == 32);
        T64C_CHECK(c, o.level_count[1] == 2);  // two connected chains, one per bank
        T64C_CHECK(c, o.level_count[2] == 1);  // the cross-bank edge joins them
        T64C_CHECK(c, o.level_begin[2] == 48);
        T64C_CHECK(c, o.level_begin[3] == 64);
        T64C_CHECK(c, o.going_next[0] == 32);   // vertex 0 -> the first level-1 node
        T64C_CHECK(c, o.going_next[15] == 32);
        T64C_CHECK(c, o.going_next[16] == 33);  // vertex 16 -> the second one
        T64C_CHECK(c, o.going_next[31] == 33);
        T64C_CHECK(c, o.going_next[32] == o.going_next[33]);  // both -> the single level-2 node
        // the *pre*-closure masks (the engine stores the post-closure ones, which
        // for a fully connected bank are the whole bank: 0xffff)
        T64C_CHECK(c, o.pre_mask[0] == 3);  // bits {0,1}
        T64C_CHECK(c, o.pre_mask[15] == ((1 << 14) | (1 << 15)));
        T64C_CHECK(c, o.pre_mask[16] == ((1 << 0) | (1 << 1)));
        T64C_CHECK(c, o.fine_mask[0] == 0xffff);
        T64C_CHECK(c, o.fine_mask[16] == 0xffff);

        BlockTriplet t;
        t.row = 0;
        t.col = 0;
        t.H   = Blk3::Identity();
        MASPreconditionerEngine engine;
        run_engine(engine, in, {t});
        check_level_structure(c, engine, in, o, "chain32");
    }

    // ---- (b) the same chain with a *reversed* partition order ----------------
    // `going_next`'s level-0 region is indexed by the *real* vertex index while the
    // connect masks are indexed by the *partition position* (the reference's two
    // index spaces); a non-identity partition is what separates them.
    {
        HandMesh mesh;
        mesh.vert_num = 32;
        for(int i = 0; i + 1 < 32; ++i)
            mesh.edges.push_back({i, i + 1});

        auto in = make_engine_input(mesh, 2);
        for(int p = 0; p < in.part_map_size; ++p)
            in.part_to_real[static_cast<size_t>(p)] = (p < mesh.vert_num) ? mesh.vert_num - 1 - p
                                                                         : -1;
        for(int v = 0; v < mesh.vert_num; ++v)
            in.real_to_part[static_cast<size_t>(v)] = mesh.vert_num - 1 - v;

        auto o = build_level_oracle(in, mesh);
        T64C_CHECK(c, o.level_num == 3);
        // vertex 0 sits at partition position 31 (the second bank), vertex 31 at 0
        T64C_CHECK(c, o.going_next[0] == 33);
        T64C_CHECK(c, o.going_next[31] == 32);

        BlockTriplet t;
        t.row   = 0;
        t.col   = 0;
        t.H     = Blk3::Identity();
        MASPreconditionerEngine engine;
        run_engine(engine, in, {t});
        check_level_structure(c, engine, in, o, "chain32_reversed");
    }

    // ---- (c) an 8x4 triangulated patch (2 banks) ----------------------------
    {
        constexpr int W = 8, H = 4;
        HandMesh      mesh;
        mesh.vert_num = W * H;
        auto idx      = [](int x, int y) { return y * W + x; };
        for(int y = 0; y < H; ++y)
            for(int x = 0; x < W; ++x)
            {
                if(x + 1 < W)
                    mesh.edges.push_back({idx(x, y), idx(x + 1, y)});
                if(y + 1 < H)
                    mesh.edges.push_back({idx(x, y), idx(x, y + 1)});
                if(x + 1 < W && y + 1 < H)
                {
                    // the two triangle diagonals of the patch
                    mesh.edges.push_back({idx(x, y), idx(x + 1, y + 1)});
                    mesh.edges.push_back({idx(x + 1, y), idx(x, y + 1)});
                }
            }
        auto in = make_engine_input(mesh, 2);
        auto o  = build_level_oracle(in, mesh);

        T64C_CHECK(c, o.level_num == 3);
        T64C_CHECK(c, o.level_count[1] == 2);
        T64C_CHECK(c, o.level_count[2] == 1);
        T64C_CHECK(c, o.level_begin[1] == 32);
        T64C_CHECK(c, o.bank_count[0] == 1);
        T64C_CHECK(c, o.bank_count[1] == 1);

        BlockTriplet t;
        t.row = 0;
        t.col = 0;
        t.H   = Blk3::Identity();
        MASPreconditionerEngine engine;
        run_engine(engine, in, {t});
        check_level_structure(c, engine, in, o, "patch8x4");
    }
}
}  // namespace t64c

// ===========================================================================
//  The probe (reads the port's systems in every scene) + the shared scene
// ===========================================================================
namespace t64c
{
class T64cProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::FiniteElementMethod*       fem    = nullptr;
    inline static gg::GlobalLinearSystem*        gls    = nullptr;
    inline static gg::FEMLinearSubsystem*        femsub = nullptr;
    inline static gg::FEMMASPreconditioner*      mas    = nullptr;
    inline static gg::FEMDiagPreconditioner*     diag   = nullptr;

    static void reset()
    {
        fem    = nullptr;
        gls    = nullptr;
        femsub = nullptr;
        mas    = nullptr;
        diag   = nullptr;
    }

  private:
    void do_build() override
    {
        fem    = &require<gg::FiniteElementMethod>();
        gls    = &require<gg::GlobalLinearSystem>();
        femsub = &require<gg::FEMLinearSubsystem>();
        mas    = find<gg::FEMMASPreconditioner>();
        diag   = find<gg::FEMDiagPreconditioner>();
    }
};

/// The 5-tet mesh of the T5.4b sibling: all rest determinants positive, one
/// non-ascending even permutation (so the `upper_LR` swap is exercised).
struct FiveTet
{
    std::vector<Vector3>  X;
    std::vector<Vector3>  x;
    std::vector<Vector4i> tets;
};

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

/// `mas_selected == true` sets `linear_system/fem_preconditioner = "mas"`; otherwise
/// the config carries no such key at all (the reference's "off" state).
[[nodiscard]] Scene build_scene(const FiveTet& mesh, bool mas_selected)
{
    auto config = Scene::default_config();
    config["contact"]["enable"]      = false;
    config["sanity_check"]["enable"] = false;
    config["integrator"]["type"]     = "bdf1";
    config["dt"]                     = 1.0e-2;
    config["gravity"]                = Vector3{0.0, -9.8, 0.0};
    if(mas_selected)
        config["linear_system"]["fem_preconditioner"] = "mas";

    Scene scene{config};
    auto  object = scene.objects().create("t64c");

    StableNeoHookean snh;
    auto             tet_mesh = tetmesh(span<const Vector3>{mesh.X},
                                        span<const Vector4i>{mesh.tets});
    label_surface(tet_mesh);
    snh.apply_to(tet_mesh, ElasticModuli::youngs_poisson(120.0e3, 0.49), 1000.0);
    {
        auto is_fixed = tet_mesh.vertices().find<IndexT>(builtin::is_fixed);
        if(!is_fixed)
            is_fixed = tet_mesh.vertices().create<IndexT>(builtin::is_fixed, 0);
        view(*is_fixed)[0] = 1;  // one fixed vertex (its diagonal block is replaced)
    }
    object->geometries().create(tet_mesh);
    return scene;
}

void step_frame(World& world)
{
    world.advance();
    world.sync();
    gg::frame_stream().synchronize();
}

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

/// Read the assembled BCOO matrix of a standalone impl back as block triplets.
[[nodiscard]] std::vector<BlockTriplet> read_assembled_triplets(
    const lla::LBCOOMatrix<double, 3, 3>& bcoo)
{
    const auto nnz = static_cast<SizeT>(bcoo.triplet_count());
    std::vector<std::int32_t> rows(nnz), cols(nnz);
    std::vector<double>       vals(nnz * 9u, 0.0);
    gg::frame_stream().synchronize();
    if(nnz != 0u)
    {
        bcoo.copy_to_row_indices(span<std::int32_t>{rows});
        bcoo.copy_to_col_indices(span<std::int32_t>{cols});
        bcoo.copy_to_values(span<double>{vals});
        lud::main_stream().synchronize();
    }
    std::vector<BlockTriplet> ts;
    ts.reserve(nnz);
    for(SizeT k = 0; k < nnz; ++k)
    {
        BlockTriplet t;
        t.row = rows[k];
        t.col = cols[k];
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                t.H(r, c) = vals[k * 9u + static_cast<SizeT>(c * 3 + r)];
        ts.push_back(t);
    }
    return ts;
}

/// The fine-level (block-Jacobi per cluster) preconditioner oracle:
/// for every vertex `v` in cluster `c`, `z_v = sum_{q in c} M_c^{-1}[i_v, i_q] r_q`.
[[nodiscard]] std::vector<double> oracle_block_jacobi_z(
    const std::vector<BlockTriplet>& ts,
    const std::vector<int>&          part_to_real,
    const std::vector<int>&          real_to_part,
    const std::vector<double>&       r,
    int                              n_clusters)
{
    const int          vert_num = static_cast<int>(r.size() / 3u);
    std::vector<double> z(static_cast<size_t>(vert_num) * 3u, 0.0);
    for(int c = 0; c < n_clusters; ++c)
    {
        // the cluster's members (partition positions -> vertices)
        std::vector<int> members;
        for(size_t p = 0; p < part_to_real.size(); ++p)
            if(part_to_real[p] >= 0 && static_cast<int>(p) / 16 == c)
                members.push_back(static_cast<int>(p));
        if(members.empty())
            continue;
        const auto D   = oracle_level0_dense(ts, c, real_to_part);
        const auto inv = oracle_cluster_inverse(D);
        for(int p : members)
        {
            const int v = part_to_real[static_cast<size_t>(p)];
            const int lr = p % 16;
            for(int i = 0; i < 3; ++i)
            {
                double acc = 0.0;
                for(int q : members)
                {
                    const int vq = part_to_real[static_cast<size_t>(q)];
                    const int lc = q % 16;
                    for(int j = 0; j < 3; ++j)
                        acc += inv(3 * lr + i, 3 * lc + j)
                               * r[static_cast<size_t>(3 * vq + j)];
                }
                z[static_cast<size_t>(3 * v + i)] = acc;
            }
        }
    }
    return z;
}

/// The block-*diagonal* (3x3 per vertex) oracle, i.e. what the diagonal
/// preconditioner would produce: `z_v = D_v^{-1} r_v` with `D_v` the assembled
/// diagonal block.
[[nodiscard]] std::vector<double> oracle_diagonal_z(const std::vector<BlockTriplet>& ts,
                                                    int                              vert_num,
                                                    const std::vector<double>&       r)
{
    std::vector<Blk3> D(static_cast<size_t>(vert_num), Blk3::Zero());
    for(const auto& t : ts)
    {
        if(t.row != t.col)
            continue;
        if(t.row < 0 || t.row >= vert_num)
            continue;
        D[static_cast<size_t>(t.row)] += t.H;
    }
    std::vector<double> z(static_cast<size_t>(vert_num) * 3u, 0.0);
    for(int v = 0; v < vert_num; ++v)
    {
        Eigen::Vector3d rv(r[static_cast<size_t>(3 * v)],
                           r[static_cast<size_t>(3 * v + 1)],
                           r[static_cast<size_t>(3 * v + 2)]);
        const auto      zv = D[static_cast<size_t>(v)].inverse() * rv;
        for(int i = 0; i < 3; ++i)
            z[static_cast<size_t>(3 * v + i)] = zv(i);
    }
    return z;
}
}  // namespace t64c

REGISTER_SIM_SYSTEM(t64c::T64cProbe);

// ===========================================================================
//  CASE 3 — the preconditioner contract on a real scene
// ===========================================================================
namespace t64c
{
TEST_CASE("mas_preconditioner_apply", "[luisa][mas_preconditioner]")
{
    CaseScope c{"preconditioner_apply"};
    test_runtime();

    const auto mesh = five_tet();
    T64cProbe::reset();

    auto   scene = build_scene(mesh, /*mas_selected=*/true);
    Engine engine{"luisa", workspace_path("uipc_luisa_t64c_apply")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T64cProbe::femsub != nullptr);
    REQUIRE(T64cProbe::femsub->is_valid());
    REQUIRE(T64cProbe::mas != nullptr);
    CHECK(T64cProbe::mas->is_valid());

    step_frame(world);
    deform_state(world, mesh.x);

    auto& precond    = *T64cProbe::mas;
    auto& mas_engine = precond.engine;
    // The engine's hierarchy is built in the preconditioner's `do_init`, which only
    // `GlobalLinearSystem::Impl::init()` reaches (`LocalPreconditioner::init()` is
    // private to GlobalLinearSystem).  In this port the *engine* does not call it yet
    // (Phase 8's `sim_engine.cpp` note), so a scene alone leaves the engine
    // uninitialised -- measured here, and the reason the standalone impl below exists.
    note("[t64c] apply: before the linear system's init: valid=%d initialized=%d "
         "(Phase 8 drives `GlobalLinearSystem::init()`)\n",
         static_cast<int>(precond.is_valid()),
         static_cast<int>(mas_engine.is_initialized()));

    // ---- the standalone impl: the real subsystem + the real preconditioner ----
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T64cProbe::femsub);
    impl.local_preconditioners.register_sim_system(precond);
    impl.init();  // reaches LocalPreconditioner::do_init -> the engine's hierarchy
    REQUIRE(mas_engine.is_initialized());
    T64C_CHECK(c, precond.m_has_partition);

    const auto part_to_real = read_buffer(mas_engine.part_to_real);
    const auto real_to_part = read_buffer(mas_engine.real_to_part);
    const int  vert_num     = static_cast<int>(T64cProbe::fem->xs().size());
    const int  n_clusters   = mas_engine.num_cluster_blocks();
    note("[t64c] apply: vert_num %d, part_map_size %d, level_num %d, clusters %d",
         vert_num,
         mas_engine.m_total_map_nodes,
         mas_engine.level_count(),
         n_clusters);
    for(int l = 1; l <= mas_engine.level_count(); ++l)
        note(" | L%d=%d@%d", l, mas_engine.level_size(l).x, mas_engine.level_size(l).y);
    note("\n");
    T64C_CHECK(c, mas_engine.m_total_map_nodes == 16);  // one bank of 16 for 8 vertices
    T64C_CHECK(c, n_clusters >= 2);  // the level-0 bank + the coarse levels

    T64C_CHECK(c, impl.diag_dof_offsets_counts.total_count() == static_cast<SizeT>(3 * vert_num));
    T64C_CHECK(c, impl.no_precond_diag_subsystem_indices.empty());
    impl.build_linear_system();
    T64C_CHECK(c, impl.bcoo_A.triplet_count() > 0);

    const auto ts = read_assembled_triplets(impl.bcoo_A);
    note("[t64c] apply: assembled %d triplets, |A|max %.3e\n",
                static_cast<int>(ts.size()),
                [&] {
                    double m = 0.0;
                    for(const auto& t : ts)
                        m = std::max(m, t.H.cwiseAbs().maxCoeff());
                    return m;
                }());

    // a per-vertex-distinct r so a wrong block cannot pass
    std::vector<double> r(static_cast<size_t>(3 * vert_num), 0.0);
    for(int v = 0; v < vert_num; ++v)
        for(int i = 0; i < 3; ++i)
            r[static_cast<size_t>(3 * v + i)] =
                0.5 * static_cast<double>(v + 1) + 0.25 * static_cast<double>(i) - 1.0;

    lla::LDenseVector<double> r_dev, z_dev;
    r_dev.resize(static_cast<SizeT>(3 * vert_num));
    z_dev.resize(static_cast<SizeT>(3 * vert_num));
    r_dev.copy_from(span<const double>{r});
    const std::vector<double> sentinel(static_cast<size_t>(3 * vert_num), -12345.0);
    z_dev.copy_from(span<const double>{sentinel});
    lud::LDeviceVar<IndexT> converged;
    converged.copy_from(IndexT{0});

    // fine-level-only mode: the engine then *is* a per-cluster block Jacobi, which is
    // an oracle this test can build with Eigen
    mas_engine.set_active_level_num(1);
    gg::frame_stream().synchronize();
    impl.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    gg::frame_stream().synchronize();
    std::vector<double> z(static_cast<size_t>(3 * vert_num), 0.0);
    z_dev.copy_to(span<double>{z});

    const auto z_oracle = oracle_block_jacobi_z(ts, part_to_real, real_to_part, r, n_clusters);
    const auto z_diag   = oracle_diagonal_z(ts, vert_num, r);
    {
        // the engine's assembled cluster matrix and its inverse vs the oracle (the
        // same comparison case 1 makes on synthetic meshes, here on the *real*
        // scene A: the fixed vertex's row, the assembled couplings and the coarse
        // level all take part)
        const auto D        = oracle_level0_dense(ts, 0, real_to_part);
        const auto inv_want = oracle_cluster_inverse(D);
        const auto hess     = read_buffer(mas_engine.cluster_hessians);
        const auto flat     = read_buffer(mas_engine.cluster_inverses);
        const auto hess_mat = read_cluster_matrix_blocks(hess, 0);
        // the kernel pins a zero diagonal cell to 1 while it *gathers* the 48x48
        // matrix (the stored buffer keeps the raw value), so the comparison applies
        // the same pin to the engine's matrix
        auto hess_pinned = hess_mat;
        for(int kk = 0; kk < 48; ++kk)
            if(hess_pinned(kk, kk) == 0.0)
                hess_pinned(kk, kk) = 1.0;
        const auto inv_mat  = read_cluster_matrix_blocks(flat, 0);
        double     worst_hess = 0.0, worst_inv = 0.0, scale = 0.0;
        for(int rr = 0; rr < 48; ++rr)
        {
            for(int kk = 0; kk < 48; ++kk)
            {
                worst_hess = std::max(worst_hess, std::abs(hess_pinned(rr, kk) - D(rr, kk)));
                worst_inv = std::max(worst_inv, std::abs(inv_mat(rr, kk) - inv_want(rr, kk)));
                if(rr < 24 && kk < 24)
                    scale = std::max(scale, std::abs(inv_want(rr, kk)));
            }
        }
        note("[t64c] apply: assembled cluster0 matrix vs oracle worst %.3e; its inverse vs "
             "the Eigen oracle worst %.3e (|inv|max %.3e)\n",
             worst_hess,
             worst_inv,
             scale);
        // the scatter is a plain sum of doubles; the *atomic* accumulation order is/n        // not fixed (measured 1.4e-14 absolute on |H| ~ 4e2, i.e. a few ulps)/n        T64C_CHECK(c, worst_hess <= 1e-12);
        T64C_CHECK(c, scale > 1.0e-2);     // non-vacuous
        T64C_CHECK(c, worst_inv <= 1e-5);  // the inverse is stored in float
    }
    double     worst = 0.0, z_scale = 0.0, rdot = 0.0, z_norm = 0.0;
    for(size_t i = 0; i < z.size(); ++i)
    {
        worst   = std::max(worst, std::abs(z[i] - z_oracle[i]));
        z_scale = std::max(z_scale, std::abs(z_oracle[i]));
        rdot += z[i] * r[i];
        z_norm += z[i] * z[i];
    }
    note("[t64c] apply: fine-level z vs Eigen block-Jacobi oracle worst abs %.3e "
                "(scale %.3e), z^T r = %.6e\n",
                worst,
                z_scale,
                rdot);
    for(int v = 0; v < 4; ++v)
        note("[t64c] apply: v%d r = (%+.6e %+.6e %+.6e) z_mas = (%+.6e %+.6e %+.6e) "
             "z_oracle = (%+.6e %+.6e %+.6e) z_diag = (%+.6e %+.6e %+.6e)\n",
             v,
             r[3 * v],
             r[3 * v + 1],
             r[3 * v + 2],
             z[3 * v],
             z[3 * v + 1],
             z[3 * v + 2],
             z_oracle[3 * v],
             z_oracle[3 * v + 1],
             z_oracle[3 * v + 2],
             z_diag[3 * v],
             z_diag[3 * v + 1],
             z_diag[3 * v + 2]);
    T64C_CHECK(c, z_scale > 1.0e-2);   // non-vacuous
    T64C_CHECK(c, worst <= 1e-5);      // the inverse is stored in float (GIPC parity)
    T64C_CHECK(c, z_norm > 0.0);
    T64C_CHECK(c, rdot > 0.0);         // the preconditioner is positive on r

    // ---- determinism ---------------------------------------------------------
    z_dev.copy_from(span<const double>{sentinel});
    gg::frame_stream().synchronize();
    impl.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    gg::frame_stream().synchronize();
    std::vector<double> z2(static_cast<size_t>(3 * vert_num), 0.0);
    z_dev.copy_to(span<double>{z2});
    int identical = 0;
    for(size_t i = 0; i < z.size(); ++i)
        identical += (z[i] == z2[i]) ? 1 : 0;
    note("[t64c] apply: determinism %d/%d entries bitwise equal\n",
                identical,
                static_cast<int>(z.size()));
    T64C_CHECK(c, identical == static_cast<int>(z.size()));

    // ---- the differenced statement vs the block-*diagonal* preconditioner -----
    double worst_diag = 0.0;
    for(size_t i = 0; i < z.size(); ++i)
        worst_diag = std::max(worst_diag, std::abs(z[i] - z_diag[i]));
    note("[t64c] apply: |z_mas - z_diag|max = %.3e (so MAS is not the diagonal one)\n",
                worst_diag);
    T64C_CHECK(c, worst_diag > 1.0e-3);

    // ---- the full hierarchy changes the answer (the coarse levels are used) ----
    mas_engine.set_active_level_num(0);  // 0 = the full hierarchy
    z_dev.copy_from(span<const double>{sentinel});
    gg::frame_stream().synchronize();
    impl.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    gg::frame_stream().synchronize();
    std::vector<double> z_full(static_cast<size_t>(3 * vert_num), 0.0);
    z_dev.copy_to(span<double>{z_full});
    double worst_coarse = 0.0, full_norm = 0.0;
    int    n_finite     = 0;
    for(size_t i = 0; i < z.size(); ++i)
    {
        worst_coarse = std::max(worst_coarse, std::abs(z_full[i] - z[i]));
        full_norm    = std::max(full_norm, std::abs(z_full[i]));
        n_finite += std::isfinite(z_full[i]) ? 1 : 0;
    }
    note("[t64c] apply: |z_full - z_fine|max = %.3e (|z_full|max %.3e, finite %d/%d)\n",
                worst_coarse,
                full_norm,
                n_finite,
                static_cast<int>(z.size()));
    T64C_CHECK(c, n_finite == static_cast<int>(z.size()));
    T64C_CHECK(c, worst_coarse > 1.0e-6);
    T64C_CHECK(c, full_norm > 1.0e-2);

    // ---- the `converged != 0` early-out -------------------------------------
    z_dev.copy_from(span<const double>{sentinel});
    converged.copy_from(IndexT{1});
    gg::frame_stream().synchronize();
    impl.apply_preconditioner(z_dev.view(), r_dev.cview(), converged.view(), nullptr);
    gg::frame_stream().synchronize();
    z_dev.copy_to(span<double>{z});
    int kept = 0;
    for(size_t i = 0; i < z.size(); ++i)
        kept += (z[i] == sentinel[i]) ? 1 : 0;
    note("[t64c] apply: converged != 0 kept %d/%d entries unchanged\n",
                kept,
                static_cast<int>(z.size()));
    T64C_CHECK(c, kept == static_cast<int>(z.size()));
}
}  // namespace t64c

// ===========================================================================
//  CASES 4 & 5 — the config selection (`linear_system/fem_preconditioner`)
// ===========================================================================
namespace t64c
{
TEST_CASE("mas_config_mas", "[luisa][mas_preconditioner]")
{
    CaseScope c{"config_mas"};
    test_runtime();

    const auto mesh = five_tet();
    T64cProbe::reset();

    // The deferral diagnostic is logged at *debug* level by
    // `SimSystemCollection::build_systems()` ("[{system}] shutdown, reason: {...}"), so
    // the case turns it on around the scene build and prints what to look for: with
    // "mas" selected the *diagonal* preconditioner must report its own deferral, and
    // the MAS one must build.
    const auto old_level = uipc::logger::get_level();
    uipc::logger::set_level(spdlog::level::debug);
    auto   scene = build_scene(mesh, /*mas_selected=*/true);
    Engine engine{"luisa", workspace_path("uipc_luisa_t64c_cfg_mas")};
    World  world{engine};
    world.init(scene);
    uipc::logger::set_level(old_level);

    REQUIRE(world.is_valid());
    REQUIRE(T64cProbe::mas != nullptr);
    REQUIRE(T64cProbe::diag != nullptr);
    REQUIRE(T64cProbe::femsub != nullptr);

    note("[t64c] config_mas: FEMMASPreconditioner valid=%d, FEMDiagPreconditioner "
                "valid=%d (the diagonal one's deferral is the debug line "
                "\"FEMDiagPreconditioner ... deferring to FEMMASPreconditioner\")\n",
                static_cast<int>(T64cProbe::mas->is_valid()),
                static_cast<int>(T64cProbe::diag->is_valid()));

    T64C_CHECK(c, T64cProbe::mas->is_valid());
    T64C_CHECK(c, !T64cProbe::diag->is_valid());
    T64C_CHECK(c, T64cProbe::femsub->is_valid());

    // the selected preconditioner really is the one registered with the linear system
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T64cProbe::femsub);
    impl.local_preconditioners.register_sim_system(*T64cProbe::mas);
    impl.init();
    T64C_CHECK(c, impl.no_precond_diag_subsystem_indices.empty());

    // the engine built its hierarchy from this scene's FEM geometry
    T64C_CHECK(c, T64cProbe::mas->m_has_partition);
    T64C_CHECK(c, T64cProbe::mas->engine.is_initialized());
    note("[t64c] config_mas: engine level_num %d, part_map_size %d, clusters %d\n",
                T64cProbe::mas->engine.level_count(),
                T64cProbe::mas->engine.m_total_map_nodes,
                T64cProbe::mas->engine.num_cluster_blocks());
    T64C_CHECK(c, T64cProbe::mas->engine.level_count() >= 2);
}

TEST_CASE("mas_config_default", "[luisa][mas_preconditioner]")
{
    CaseScope c{"config_default"};
    test_runtime();

    const auto mesh = five_tet();
    T64cProbe::reset();

    const auto old_level = uipc::logger::get_level();
    uipc::logger::set_level(spdlog::level::debug);
    auto   scene = build_scene(mesh, /*mas_selected=*/false);
    Engine engine{"luisa", workspace_path("uipc_luisa_t64c_cfg_default")};
    World  world{engine};
    world.init(scene);
    uipc::logger::set_level(old_level);

    REQUIRE(world.is_valid());
    REQUIRE(T64cProbe::mas != nullptr);
    REQUIRE(T64cProbe::diag != nullptr);

    note("[t64c] config_default: FEMMASPreconditioner valid=%d, FEMDiagPreconditioner "
                "valid=%d (the MAS one's own deferral is the debug line "
                "\"FEMMASPreconditioner: linear_system/fem_preconditioner != \"mas\"\")\n",
                static_cast<int>(T64cProbe::mas->is_valid()),
                static_cast<int>(T64cProbe::diag->is_valid()));

    T64C_CHECK(c, T64cProbe::diag->is_valid());  // the default path is unchanged
    T64C_CHECK(c, !T64cProbe::mas->is_valid());
    // the MAS engine must not have been initialized at all (its do_init returns early)
    T64C_CHECK(c, !T64cProbe::mas->engine.is_initialized());
    T64C_CHECK(c, !T64cProbe::mas->m_has_partition);
}
}  // namespace t64c

// ===========================================================================
//  CASE 6 — the factored routines: device vs host
// ===========================================================================
namespace t64c
{
/// The transitive closure on the device: the ported routine with the `$shared` mask
/// callback (the shape the shipped kernels use).
auto& t64c_closure_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::uint32_t>,  // masks (16)
                                   lcrt::Buffer<std::uint32_t>,  // out (16)
                                   std::uint32_t>(               // lane count
            [](lcrt::BufferVar<std::uint32_t> masks,
               lcrt::BufferVar<std::uint32_t> out,
               lcrt::Var<std::uint32_t>       lane_count) noexcept
            {
                // LC requires the *hardware* block size to be a multiple of 32
                // (16 lanes is the engine's logical-bank width, not a block size)
                lcrt::set_block_size(32u);
                $shared<std::uint32_t> s_masks{16u};
                auto tid = lcrt::dispatch_x();
                $if(tid < lane_count)
                {
                    s_masks[tid] = masks->read(tid);
                };
                lcrt::sync_block();
                $if(tid < lane_count)
                {
                    auto own = masks->read(tid);
                    auto res = mpd::transitive_closure_mask(
                        1u << tid,
                        own,
                        [&](auto j) noexcept { return s_masks[j]; },
                        lane_count);
                    out->write(tid, res);
                };
            });
    return k;
}

using UnpartDiagDeviceView = mfd::DiagInvDeviceView;
using UnpartFlagDeviceView = mfd::UnpartDeviceView;
using TripletDeviceView    = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using DenseDeviceView      = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
using CDenseDeviceView     = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;

/// `assemble_diag_inv_for_unpartitioned_kernel` (the shipped shader) for a
/// test-local batch, through the *shared* factored routine.
auto& t64c_unpart_assemble_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,
                                                lcrt::Buffer<std::int32_t>,
                                                lcrt::Buffer<double>,
                                                lcrt::Buffer<LMatrix3x3>,
                                                lcrt::Buffer<std::int32_t>,
                                                std::int32_t,
                                                std::int32_t,
                                                uint32_t>(
        [](lcrt::BufferVar<std::int32_t> rows,
           lcrt::BufferVar<std::int32_t> cols,
           lcrt::BufferVar<double>       vals,
           lcrt::BufferVar<LMatrix3x3>   diag_inv,
           lcrt::BufferVar<std::int32_t> unpart,
           lcrt::Var<std::int32_t>       total_cols,
           lcrt::Var<std::int32_t>       fem_count,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto A = TripletDeviceView::from_handles(rows, cols, vals, total_cols, total_cols, n);
                auto U = UnpartFlagDeviceView::from_handles(unpart, fem_count);
                // the reference's single `fem_offset` (0 here): a *device* zero of the
                // same type as `fem_count`, because the routine deduces one offset type
                lcrt::Var<std::int32_t> fem_offset = total_cols * std::int32_t{0};
                auto row = mfd::unpart_diag_inv_row(
                    A, U, lcrt::cast<std::int32_t>(I), fem_offset, fem_count);
                $if(row.in_window)
                {
                    $if(scalar_detail::pred_and(row.is_unpart, mfd::unpart_flag(U, row.index)))
                    {
                        lcrt::Var<LMatrix3x3> out;
                        store_mat3x3(out, row.value);
                        diag_inv->write(row.index, out);
                    };
                };
            };
        });
    return k;
}

/// `apply_diag_inv_for_unpartitioned_kernel` (the shipped shader) for a test-local
/// batch.
auto& t64c_unpart_apply_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,
                                                lcrt::Buffer<double>,
                                                lcrt::Buffer<std::int32_t>,
                                                lcrt::Buffer<LMatrix3x3>,
                                                lcrt::Buffer<std::int32_t>,
                                                uint32_t>(
        [](lcrt::BufferVar<double>       r,
           lcrt::BufferVar<double>       z,
           lcrt::BufferVar<std::int32_t> converged,
           lcrt::BufferVar<LMatrix3x3>   diag_inv,
           lcrt::BufferVar<std::int32_t> unpart,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                $if(converged->read(0) == 0)
                {
                    auto Z = DenseDeviceView::from_handles(z, n * 3u);
                    auto R = CDenseDeviceView::from_handles(r, n * 3u);
                    auto D = UnpartDiagDeviceView::from_handles(diag_inv, n);
                    auto U = UnpartFlagDeviceView::from_handles(unpart, n);
                    $if(mfd::unpart_flag(U, lcrt::cast<std::int32_t>(i)))
                    {
                        mfd::unpart_diag_apply_row(Z, R, D, U, lcrt::cast<std::int32_t>(i));
                    };
                };
            };
        });
    return k;
}

TEST_CASE("mas_device_vs_host", "[luisa][mas_preconditioner]")
{
    CaseScope c{"device_vs_host"};
    test_runtime();

    // ---- (a) the pure integer laws ------------------------------------------
    T64C_CHECK(c, mpd::sym_index(0, 0) == 0);
    T64C_CHECK(c, mpd::sym_index(0, 15) == 15);
    T64C_CHECK(c, mpd::sym_index(1, 1) == 16);
    T64C_CHECK(c, mpd::sym_index(1, 2) == 17);
    T64C_CHECK(c, mpd::sym_index(15, 15) == 135);
    T64C_CHECK(c, MASPreconditionerEngine::SYM_BLOCK_COUNT == 136);
    T64C_CHECK(c, mpd::bank_align(0) == 0);
    T64C_CHECK(c, mpd::bank_align(1) == 16);
    T64C_CHECK(c, mpd::bank_align(16) == 16);
    T64C_CHECK(c, mpd::bank_align(17) == 32);
    T64C_CHECK(c, mpd::num_levels(1) == 2);
    T64C_CHECK(c, mpd::num_levels(16) == 2);
    T64C_CHECK(c, mpd::num_levels(17) == 3);
    T64C_CHECK(c, mpd::num_levels(256) == 3);
    T64C_CHECK(c, mpd::num_levels(257) == 4);
    T64C_CHECK(c, mpd::num_levels(65537) == 6);  // MAX_LEVELS caps it
    T64C_CHECK(c, mpd::level1_begin(16, 1) == 16);
    T64C_CHECK(c, mpd::level1_begin(1, 16) == 16);
    T64C_CHECK(c, mpd::level1_begin(32, 40) == 48);
    T64C_CHECK(c, mpd::lanemask_lt(0u) == 0u);
    T64C_CHECK(c, mpd::lanemask_lt(4u) == 15u);
    T64C_CHECK(c, mpd::lanemask_lt(16u) == 65535u);
    T64C_CHECK(c, mpd::mat_entry(0, 1) == 3);
    T64C_CHECK(c, mpd::mat_entry(2, 2) == 8);

    // ---- (b) the transitive closure: device vs host -------------------------
    {
        std::mt19937                           rng(64u);
        std::uniform_int_distribution<uint32_t> bits(0u, 0xFFFFu);
        constexpr int                           kRounds = 24;
        int                                     total_lanes = 0, mismatches = 0;
        for(int round = 0; round < kRounds; ++round)
        {
            std::vector<std::uint32_t> masks(16u, 0u);
            for(auto& m : masks)
                m = bits(rng);
            // the own bit is always present in the reference's masks
            for(int lane = 0; lane < 16; ++lane)
                masks[static_cast<size_t>(lane)] |= 1u << lane;

            // the host instantiation of the *same* routine
            std::vector<std::uint32_t> host(16u, 0u);
            for(int lane = 0; lane < 16; ++lane)
                host[static_cast<size_t>(lane)] = mpd::transitive_closure_mask(
                    1u << lane,
                    masks[static_cast<size_t>(lane)],
                    [&](std::uint32_t j) noexcept { return masks[static_cast<size_t>(j)]; },
                    16u);

            lud::LDeviceBuffer<std::uint32_t> masks_dev;
            lud::LDeviceBuffer<std::uint32_t> out_dev;
            masks_dev.copy_from(span<const std::uint32_t>{masks});
            std::vector<std::uint32_t> zeros(16u, 0u);
            out_dev.copy_from(span<const std::uint32_t>{zeros});
            gg::frame_stream().synchronize();
            auto& k = t64c_closure_kernel();
            gg::frame_stream().record(k(masks_dev.cview(), out_dev.view(), 16u).dispatch(32u));
            const auto dev = read_device<std::uint32_t>(out_dev.cview());
            REQUIRE(dev.size() == 16u);
            for(int lane = 0; lane < 16; ++lane)
            {
                ++total_lanes;
                if(dev[static_cast<size_t>(lane)] != host[static_cast<size_t>(lane)])
                    ++mismatches;
            }
        }
        note("[t64c] device_vs_host: transitive closure %d/%d lanes equal\n",
                    total_lanes - mismatches,
                    total_lanes);
        T64C_CHECK(c, mismatches == 0);
        T64C_CHECK(c, total_lanes == kRounds * 16);
    }

    // ---- (c) the unpartitioned-vertex fallback row --------------------------
    {
        constexpr int     kTriplets = 5;
        std::vector<std::int32_t> rows = {0, 1, 2, 3, 4};
        std::vector<std::int32_t> cols = {0, 2, 2, 3, 4};
        std::vector<double>       blocks(static_cast<size_t>(kTriplets) * 9u, 0.0);
        for(int k = 0; k < kTriplets; ++k)
            for(int col = 0; col < 3; ++col)
                for(int row = 0; row < 3; ++row)
                    blocks[static_cast<size_t>(k * 9 + col * 3 + row)] =
                        (row == col) ? (2.0 + k) : (0.25 * (row + 1) - 0.5 * col + k);
        // vertex 1 is unpartitioned, vertex 0 is not; triplet 4 is out of the window
        std::vector<std::int32_t> unpart_flags = {0, 1, 0, 1, 1};
        const std::int32_t        fem_count     = 4;

        // the host round starts from the *same* sentinel-filled buffer the device
        // round used, so the rows the store does not fire on keep the same value
        std::vector<LMatrix3x3> diag_init(kTriplets, LMatrix3x3{});
        for(auto& m : diag_init)
            for(int c = 0; c < 3; ++c)
                m(c, c) = -7.0;
        std::vector<LMatrix3x3> diag_inv_host = diag_init;
        std::vector<std::int32_t> in_window_host(kTriplets, 0), is_unpart_host(kTriplets, 0),
            index_host(kTriplets, 0);
        auto host_triplet = lla::LCTripletMatrixHostMemoryView<double, 3, 3>::from_handles(
            rows.data(),
            cols.data(),
            blocks.data(),
            kTriplets,
            kTriplets,
            kTriplets);
        auto host_unpart = mfd::UnpartHostMemoryView::from_handles(unpart_flags.data(),
                                                                   static_cast<std::int32_t>(
                                                                       unpart_flags.size()));
        for(int k = 0; k < kTriplets; ++k)
        {
            auto row = mfd::unpart_diag_inv_row(
                host_triplet, host_unpart, k, std::int32_t{0}, fem_count);
            in_window_host[static_cast<size_t>(k)] = row.in_window ? 1 : 0;
            is_unpart_host[static_cast<size_t>(k)] = row.is_unpart ? 1 : 0;
            index_host[static_cast<size_t>(k)]     = row.index;
            if(row.in_window && row.is_unpart
               && unpart_flags[static_cast<size_t>(row.index)] == 1)
            {
                // the host instantiation of the *same* inverse (`lps::inverse`)
                for(int rr = 0; rr < 3; ++rr)
                    for(int cc = 0; cc < 3; ++cc)
                        diag_inv_host[static_cast<size_t>(row.index)](rr, cc) = row.value(rr, cc);
            }
        }
        // the device round (through the shipped shader's body)
        lud::LDeviceBuffer<std::int32_t> rows_dev, cols_dev, unpart_dev;
        lud::LDeviceBuffer<double>       vals_dev;
        lud::LDeviceBuffer<LMatrix3x3>   diag_dev;
        rows_dev.copy_from(span<const std::int32_t>{rows});
        cols_dev.copy_from(span<const std::int32_t>{cols});
        vals_dev.copy_from(span<const double>{blocks});
        unpart_dev.copy_from(span<const std::int32_t>{unpart_flags});
        diag_dev.copy_from(span<const LMatrix3x3>{diag_init});
        gg::frame_stream().synchronize();
        {
            auto& k = t64c_unpart_assemble_kernel();
            gg::frame_stream().record(
                k(rows_dev.cview(), cols_dev.cview(), vals_dev.cview(), diag_dev.view(),
                  unpart_dev.cview(), std::int32_t{kTriplets}, fem_count,
                  static_cast<uint32_t>(kTriplets))
                    .dispatch(kTriplets));
        }
        const auto diag_host_out = read_device<LMatrix3x3>(diag_dev.cview());

        // the oracle: the reference's own rule (i == j && unpart(i) == 1 -> inverse)
        int stored = 0;
        for(int k = 0; k < kTriplets; ++k)
        {
            const bool in_window = (rows[static_cast<size_t>(k)] >= 0
                                    && rows[static_cast<size_t>(k)] < fem_count
                                    && cols[static_cast<size_t>(k)] >= 0
                                    && cols[static_cast<size_t>(k)] < fem_count);
            // the routine's `is_unpart` is the reference's `i == j` (the flag read is
            // the *caller's*), and the store additionally needs unpart(i) == 1
            const bool is_unpart = rows[static_cast<size_t>(k)] == cols[static_cast<size_t>(k)];
            const bool fires    = in_window && is_unpart
                               && unpart_flags[static_cast<size_t>(rows[static_cast<size_t>(k)])] == 1;
            T64C_CHECK(c, in_window_host[static_cast<size_t>(k)] == (in_window ? 1 : 0));
            T64C_CHECK(c, is_unpart_host[static_cast<size_t>(k)] == (is_unpart ? 1 : 0));
            T64C_CHECK(c, index_host[static_cast<size_t>(k)] == rows[static_cast<size_t>(k)]);
            if(!fires)
                continue;
            ++stored;
            // the host instantiation of the same routine vs the shipped shader's body
            // (both write the inverse into `diag[row_index]`)
            const auto slot = static_cast<size_t>(rows[static_cast<size_t>(k)]);
            double     worst = 0.0;
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                    worst = std::max(worst,
                                     std::abs(static_cast<double>(
                                                  diag_host_out[slot](rr, cc))
                                              - static_cast<double>(
                                                  diag_inv_host[slot](rr, cc))));
            T64C_CHECK(c, worst <= 1e-15);
        }
        note("[t64c] device_vs_host: unpartitioned assemble stored %d/%d blocks "
                    "(4 out-of-window/is-partitioned triplets ignored)\n",
                    stored,
                    kTriplets);
        T64C_CHECK(c, stored == 1);  // only triplet 1 (vertex 1, unpartitioned) lands

        // the sentinel survives everywhere the store did not fire
        int sentinels = 0;
        for(int k = 0; k < kTriplets; ++k)
        {
            const bool fires = (rows[static_cast<size_t>(k)] == cols[static_cast<size_t>(k)])
                               && rows[static_cast<size_t>(k)] < fem_count
                               && unpart_flags[static_cast<size_t>(rows[static_cast<size_t>(k)])] == 1;
            if(!fires && diag_host_out[static_cast<size_t>(k)](0, 0) == -7.0)
                ++sentinels;
        }
        T64C_CHECK(c, sentinels == kTriplets - stored);

        // the apply row: host vs device over the *same* inverse blocks
        std::vector<double> r(static_cast<size_t>(3 * kTriplets), 0.0);
        for(int v = 0; v < kTriplets; ++v)
            for(int i = 0; i < 3; ++i)
                r[static_cast<size_t>(3 * v + i)] =
                    0.5 * static_cast<double>(v + 1) - 0.25 * static_cast<double>(i);
        // the same sentinel the device buffer starts from, so the entries neither
        // side writes compare equal
        std::vector<double> z_host(static_cast<size_t>(3 * kTriplets), -12345.0);
        {
            auto R = lla::LCDenseVectorHostMemoryView<double>::from_handles(
                r.data(), static_cast<std::int32_t>(r.size()));
            auto Z = lla::LDenseVectorHostMemoryView<double>::from_handles(
                z_host.data(), static_cast<std::int32_t>(z_host.size()));
            auto D = mfd::DiagInvHostMemoryView::from_handles(
                diag_inv_host.data(), static_cast<std::int32_t>(diag_inv_host.size()));
            for(int v = 0; v < kTriplets; ++v)
            {
                if(unpart_flags[static_cast<size_t>(v)] != 1)
                    continue;
                mfd::unpart_diag_apply_row(Z, R, D, host_unpart, v);
            }
        }
        lud::LDeviceVar<std::int32_t> conv_zero;
        conv_zero.copy_from(std::int32_t{0});
        lud::LDeviceBuffer<double> r_dev, z_dev;
        r_dev.copy_from(span<const double>{r});
        std::vector<double> z_init(static_cast<size_t>(3 * kTriplets), -12345.0);
        z_dev.copy_from(span<const double>{z_init});
        gg::frame_stream().synchronize();
        {
            auto& k = t64c_unpart_apply_kernel();
            gg::frame_stream().record(k(r_dev.cview(),
                                        z_dev.view(),
                                        conv_zero.cview(),
                                        diag_dev.cview(),
                                        unpart_dev.cview(),
                                        static_cast<uint32_t>(kTriplets))
                                         .dispatch(kTriplets));
        }
        gg::frame_stream().synchronize();
        const auto z_device = read_device<double>(z_dev.cview());
        for(int v = 0; v < kTriplets; ++v)
        note("[t64c] device_vs_host: v%d z_host = (%+.6e %+.6e %+.6e) z_dev = (%+.6e %+.6e %+.6e)\n",
                 v,
                 z_host[static_cast<size_t>(3 * v)],
                 z_host[static_cast<size_t>(3 * v + 1)],
                 z_host[static_cast<size_t>(3 * v + 2)],
                 z_device[static_cast<size_t>(3 * v)],
                 z_device[static_cast<size_t>(3 * v + 1)],
                 z_device[static_cast<size_t>(3 * v + 2)]);
        int        z_mismatch = 0, z_sentinels = 0;
        for(size_t i = 0; i < z_host.size(); ++i)
        {
            z_mismatch += (std::abs(z_device[i] - z_host[i]) > 1e-15) ? 1 : 0;
            z_sentinels += (z_device[i] == z_init[i]) ? 1 : 0;
        }
        note("[t64c] device_vs_host: unpartitioned apply %d/%d entries equal "
                    "(%d kept the sentinel)\n",
                    static_cast<int>(z_host.size()) - z_mismatch,
                    static_cast<int>(z_host.size()),
                    z_sentinels);
        T64C_CHECK(c, z_mismatch == 0);
        T64C_CHECK(c, z_sentinels > 0);
    }
}
}  // namespace t64c

// ===========================================================================
//  CASE 7 — per-case assertion accounting
// ===========================================================================
namespace t64c
{
TEST_CASE("mas_summary", "[luisa][mas_preconditioner]")
{
    int total = 0;
    for(const auto& [name, checks] : g_case_counts)
    {
        note("[t64c] case %-24s %5d checks\n", name.c_str(), checks);
        total += checks;
        CHECK(checks > 0);
    }
    note("[t64c] summary: %d cases, %d checks\n",
                static_cast<int>(g_case_counts.size()),
                total);
}
}  // namespace t64c
