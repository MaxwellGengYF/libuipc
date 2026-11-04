#include <linear_system/linear_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <frame_stream.h>
#include <uipc/common/log.h>
#include <uipc/common/timer.h>
#include <algorithm>
#include <cmath>
#include <string>

/*************************************************************************************************
* T8.3a-2 — LinearPCG (port of the host loop of
* src/backends/cuda/linear_system/linear_pcg.cu:234-320, plus the config half of
* LinearFusedPCG's do_build that the fused mapping needs).
*
* The loop is transcribed statement by statement.  What is *not* here, on purpose:
*   * `dump_r_z()` / `dump_p_Ap()` and their three `if(need_debug_dump)` call sites
*     — `export_vector_market` does not exist in this backend (plan 11 phase 2/4
*     owns the Matrix-Market path).  The flag still warns at build time;
*   * the six CUDA kernels (`update_xr_kernel`, `update_p_kernel`) — they are
*     `lla::update_xr` / `lla::update_p` in T3.1's blas layer, i.e. the same
*     arithmetic, compiled once.  No new device kernel is introduced by this task;
*   * `LinearFusedPCG` — see linear_pcg.h (one implementation, config mapping).
*
* Every value the host loop needs is read back through `ctx().dot(...)`, which is
* the reference's synchronous `LinearSystemContext::dot`.  The number of host
* read-backs is therefore the reference's: one per iteration (`alpha`, `rz_new`),
* i.e. `counters.syncs ≈ 4·iters + 2` — never "one submit per solve" (research
* §4.6).  Do not replace `spmv` + `dot` by `spmv_dot` here: the reference's
* `LinearPCG` uses the pair, and its residual sequence is the one the parity gate
* compares (§7 trap 6 of the research note).
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace linear_pcg_detail
{
std::string init_message(SizeT frame, SizeT newton_iter, Float rz, Float norm_r, Float norm_z)
{
    bool r_bad = !std::isfinite(norm_r);
    auto hint  = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                         "preconditioner failed, likely due to inverse matrix calculation failure";
    return fmt::format(
        "Frame {}, Newton {}, PCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. Hint: {}.",
        frame,
        newton_iter,
        rz,
        norm_r,
        norm_z,
        hint);
}

std::string iter_message(SizeT frame, SizeT newton_iter, SizeT k, Float rz, Float norm_r, Float norm_z)
{
    bool r_ok  = std::isfinite(norm_r);
    bool z_bad = !std::isfinite(norm_z);
    auto hint  = (r_ok && z_bad) ?
                     "preconditioner failed, likely due to inverse matrix calculation failure" :
                     "PCG iteration diverged";
    return fmt::format(
        "Frame {}, Newton {}, PCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. Hint: {}.",
        frame,
        newton_iter,
        k,
        rz,
        norm_r,
        norm_z,
        hint);
}
}  // namespace linear_pcg_detail

REGISTER_SIM_SYSTEM(LinearPCG);

void LinearPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto solver_attr = config.find<std::string>("linear_system/solver");
    UIPC_ASSERT(solver_attr, "linear_system/solver not found");
    m_solver_name = solver_attr->view()[0];

    // The schema's enum is {"fused_pcg", "linear_pcg"} and its *default* is
    // "fused_pcg" (src/core/core/scene_default_config.cpp:152-163).  Accepting only
    // "linear_pcg" (what the reference's linear_pcg.cu:47 does) would leave every
    // default scene without a solver, and `solve_linear_system()` would silently
    // skip.  Both names select this class: there is no CUDA graph to capture on
    // LuisaCompute, the frame CommandList already batches the iteration stream.
    if(m_solver_name != "linear_pcg" && m_solver_name != "fused_pcg")
    {
        throw SimSystemException(
            fmt::format("LinearPCG unused: linear_system/solver = \"{}\", accepted values "
                        "are \"linear_pcg\" and \"fused_pcg\".",
                        m_solver_name));
    }

    auto& global_linear_system = require<GlobalLinearSystem>();
    // the reference's `auto& global_linear_system = require<...>()` is the build-order
    // dependency edge only (it is never dereferenced there either); spelled so the
    // compiler does not report an unused variable
    (void)global_linear_system;

    // TODO: get info from the scene, now we just use the default value
    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<Float>("linear_system/tol_rate");
    UIPC_ASSERT(tol_rate_attr, "linear_system/tol_rate not found");
    global_tol_rate = tol_rate_attr->view()[0];

    auto dump_attr = config.find<IndexT>("extras/debug/dump_linear_pcg");
    UIPC_ASSERT(dump_attr, "extras/debug/dump_linear_pcg not found");
    need_debug_dump = dump_attr->view()[0] != 0;

    logger::info("LinearPCG: max_iter_ratio = {}, tol_rate = {}, debug_dump = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 need_debug_dump);

    // ---- the fused / CUDA-graph mapping (plan §3.3(2), research §3.4.2) --------
    // Read, validate, log — and let none of them change a number.
    auto check_attr = config.find<IndexT>("linear_system/check_interval");
    UIPC_ASSERT(check_attr, "linear_system/check_interval not found");
    m_check_interval = check_attr->view()[0];
    UIPC_ASSERT_THROW(m_check_interval >= 1,
                      "linear_system/check_interval must be >= 1, yours {}.",
                      m_check_interval);

    auto graph_attr = config.find<IndexT>("linear_system/use_cuda_graph");
    UIPC_ASSERT(graph_attr, "linear_system/use_cuda_graph not found");
    m_use_cuda_graph = graph_attr->view()[0];
    UIPC_ASSERT_THROW(m_use_cuda_graph >= 0 && m_use_cuda_graph <= 2,
                      "linear_system/use_cuda_graph must be 0, 1 or 2, yours {}.",
                      m_use_cuda_graph);

    // the reference's restriction (linear_fused_pcg.cu:189-194): graph replay is
    // only enabled for the default IPC pipeline.
    auto constitution_attr = config.find<std::string>("contact/constitution");
    if(constitution_attr && constitution_attr->view()[0] != "ipc" && m_use_cuda_graph != 0)
    {
        m_use_cuda_graph = 0;
        logger::info("LinearPCG: contact/constitution != ipc — CUDA graph replay disabled");
    }

    if(need_debug_dump)
        logger::warn(
            "LinearPCG: extras/debug/dump_linear_pcg is enabled but this backend has no "
            "Matrix-Market writer yet (plan 11 phase 2/4), so no PCG vectors are written. "
            "The residual curve is still available in memory "
            "(LinearPCG::residual_trace()).");

    logger::info(
        "LinearPCG: linear_system/solver = \"{}\" -> LinearPCG (the fused CUDA-graph "
        "variant has no LuisaCompute counterpart: one PCG, recorded into the frame "
        "CommandList); use_cuda_graph = {} and check_interval = {} are validated and "
        "have no effect on the iteration count or on any other number.",
        m_solver_name,
        m_use_cuda_graph,
        m_check_interval);
}

void LinearPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    x.buffer_view().fill(0);

    auto N = static_cast<std::size_t>(x.size());
    if(z.capacity() < N)
    {
        auto M = static_cast<std::size_t>(reserve_ratio * static_cast<Float>(N));
        z.reserve(M);
        p.reserve(M);
        r.reserve(M);
        Ap.reserve(M);
    }

    z.resize(N);
    p.resize(N);
    r.resize(N);
    Ap.resize(N);
    d_converged_false.copy_from(IndexT{0});

    r0 = r;

    auto max_iter = static_cast<SizeT>(max_iter_ratio * static_cast<Float>(b.size()));
    max_iter      = std::max(max_iter, SizeT{1});

    m_rz_trace.clear();

    auto iter = pcg(x, b, max_iter);

    logger::info("LinearPCG: frame={} newton_iter={} dof={} max_iter={} -> iters={}",
                 engine().frame(),
                 engine().newton_iter(),
                 N,
                 max_iter,
                 iter);

    if(iter >= max_iter)
        logger::warn(
            "LinearPCG: reached max_iter = {} (no early convergence); "
            "check preconditioner or linear_system/tol_rate.",
            max_iter);

    info.iter_count(iter);
}

void LinearPCG::check_init_rz_nan_inf(Float rz)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        UIPC_ASSERT(false,
                    "{}",
                    linear_pcg_detail::init_message(
                        engine().frame(), engine().newton_iter(), rz, norm_r, norm_z));
    }
}

void LinearPCG::check_iter_rz_nan_inf(Float rz, SizeT k)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        UIPC_ASSERT(false,
                    "{}",
                    linear_pcg_detail::iter_message(
                        engine().frame(), engine().newton_iter(), k, rz, norm_r, norm_z));
    }
}

SizeT LinearPCG::pcg(lla::LDenseVectorView<Float>  x,
                     lla::LCDenseVectorView<Float> b,
                     SizeT                         max_iter)
{
    Timer pcg_timer{"PCG"};

    SizeT k = 0;
    // r = b - A * x
    {
        // r = b;
        r.buffer_view().copy_from(b.buffer_view());

        // x == 0, so we don't need to do the following
        // r = - A * x + r
        //spmv(-1.0, x.as_const(), 1.0, r.view());
    }

    Float alpha, beta, rz, abs_rz0;

    // z = P * r (apply preconditioner)
    {
        Timer timer{"Apply Preconditioner"};
        apply_preconditioner(z.view(), r.cview(), d_converged_false.view());
    }

    // p = z
    p = z;

    // init rz
    // rz = r^T * z
    rz = ctx().dot(r.cview(), z.cview());
    check_init_rz_nan_inf(rz);

    abs_rz0 = std::abs(rz);

    // |rz_0| / |rz_0| == 1 by construction, so the curve costs no extra read
    m_rz_trace.push_back(abs_rz0 == Float{0.0} ? Float{0.0} : Float{1.0});

    // check convergence
    if(accuracy_statisfied(r.view()) && abs_rz0 == Float{0.0})
        return 0;

    for(k = 1; k < max_iter; ++k)
    {
        {
            Timer timer{"SpMV"};
            spmv(p.cview(), Ap.view());
        }

        // alpha = rz / p^T * Ap
        alpha = rz / ctx().dot(p.cview(), Ap.cview());

        // x = x + alpha * p
        // r = r - alpha * Ap
        lla::update_xr(alpha,
                       x.buffer_view(),
                       p.buffer_view(),
                       r.buffer_view(),
                       Ap.buffer_view());

        // z = P * r (apply preconditioner)
        {
            Timer timer{"Apply Preconditioner"};
            apply_preconditioner(z.view(), r.cview(), d_converged_false.view());
        }

        // rz_new = r^T * z
        Float rz_new = ctx().dot(r.cview(), z.cview());
        check_iter_rz_nan_inf(rz_new, k);

        m_rz_trace.push_back(abs_rz0 > Float{0.0} ? std::abs(rz_new) / abs_rz0 :
                                                    std::abs(rz_new));

        // check convergence
        if(accuracy_statisfied(r.view()) && std::abs(rz_new) <= global_tol_rate * abs_rz0)
            break;

        // beta = rz_new / rz
        beta = rz_new / rz;

        // p = z + beta * p
        lla::update_p(p.buffer_view(), z.buffer_view(), beta);

        rz = rz_new;
    }

    return k;
}
}  // namespace uipc::backend::luisa
