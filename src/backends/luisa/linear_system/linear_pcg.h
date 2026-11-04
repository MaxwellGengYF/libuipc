#pragma once
/********************************************************************
 * @file linear_pcg.h
 * @brief T8.3a-2 — `LinearPCG` (port of the *host loop* of
 *        src/backends/cuda/linear_system/linear_pcg.{h,cu}).
 *
 * THE ONE PCG OF THIS BACKEND
 * ---------------------------
 * The reference has two PCG implementations:
 *   * `LinearPCG` (linear_pcg.cu:234-320) — a plain host loop, one host read per
 *     iteration (alpha), an exact iteration count;
 *   * `LinearFusedPCG` (linear_fused_pcg.cu:386-689) — the whole iteration fused
 *     into one CUDA graph, device-side alpha/beta/convergence, and an iteration
 *     count that is only ever a multiple of `check_interval`.
 *
 * On LuisaCompute there is no graph to capture: a frame already owns one open
 * `CommandList` (`frame_stream()`), `record()` appends and `commit()` submits, so
 * "fused" has no distinct meaning (§3.3 of the plan, §3.4 of the research note).
 * This port therefore has *one* implementation — `LinearPCG` — and maps the
 * config onto it:
 *
 *   linear_system/solver == "linear_pcg"  -> this class (the reference's exact
 *                                            count; CUDA-comparable)
 *   linear_system/solver == "fused_pcg"   -> this class (the *default* value of
 *                                            the schema, so accepting only
 *                                            "linear_pcg" would leave every
 *                                            default scene silently unsolved)
 *   anything else                         -> UIPC_ASSERT_THROW (the schema's enum
 *                                            rejects it earlier, in
 *                                            Scene::validate_config)
 *   linear_system/use_cuda_graph  (0|1|2) -> read, validated, logged, no effect
 *   linear_system/check_interval          -> read, validated, logged, no effect
 *   contact/constitution != "ipc"         -> use_cuda_graph is forced to 0 (the
 *                                            reference's restriction), logged
 *
 * The consequence a reviewer must not miss: a CUDA run in `fused_pcg` mode reports
 * a coarse count, so any cross-backend iteration-count comparison has to set
 * `linear_system/solver = "linear_pcg"` on the CUDA side.  Do not "fix" a count
 * mismatch by widening `tol_rate` (plan §7): compare the residual trace instead,
 * which `residual_trace()` exposes for exactly that purpose.
 *
 * Reference details preserved verbatim (each one has caused a parity mismatch in
 * similar ports):
 *   * `max_iter = max_iter_ratio * b.size()` with `max_iter_ratio` hard-coded to
 *     2.0 and `max_iter = max(2n, 1)` — never a config value;
 *   * convergence needs *both* `accuracy_statisfied(r)` and
 *     `|rz_new| <= tol_rate * |rz_0|`;
 *   * `x0 == 0`, so `r = b` and the `r = -A*x + r` update is skipped;
 *   * the reserve quirk (`if(z.capacity() < N)` reserves *all four* vectors, so
 *     `p/r/Ap` may dodge the `reserve_ratio = 1.5` growth path);
 *   * `r0 = r` right after `r.resize(N)` (a write-only dead member in the
 *     reference — kept, it is part of what the counters measure);
 *   * the two NaN/Inf guards, with the reference's message shape and hints.
 *
 * DEVIATION (1): `extras/debug/dump_linear_pcg` writes no files yet.  There is no
 * `export_vector_market` in this backend (the reference's lives in
 * `src/backends/cuda/utils/matrix_market.h`, and the whole Matrix-Market path is
 * plan 11 phase 2/4), so the flag produces the *reference FusedPCG's* warning
 * ("does not support PCG vector dumps") instead of silently doing nothing.
 *
 * DEVIATION (2, additive): `residual_trace()` — `|rz_k| / |rz_0|` for
 * k = 0..iters, collected from the values the loop already reads back (no extra
 * synchronize, no extra kernel).  The CUDA reference has no such accessor; the
 * parity gate of plan §6.1 needs the residual *curve*, not just the count.
 *
 * `DeviceBCOOMatrix` / `DeviceBSRMatrix` (the reference header's other two type
 * aliases) are not needed: the algorithm never touches the matrix itself, only
 * the four `IterativeSolver` forwarders.
 *********************************************************************/
#include <linear_system/iterative_solver.h>
#include <luisa_device/linear/linear.h>
#include <cmath>
#include <string>
#include <vector>

namespace uipc::backend::luisa
{
/// The *decision* and *message* halves of the reference's two NaN/Inf guards,
/// factored out of the asserting path so a test can pin them without aborting the
/// process (a failing `UIPC_ASSERT` calls `abort()` — cookbook §7.22; the same
/// "host-callable predicate" idiom as `off_diag_layout_detail::check_unique_pairs`,
/// §7.39 rule 9).
///
/// `init_message` / `iter_message` produce the reference's text verbatim, including
/// its hint selection, so `assert(linear_pcg_detail::init_message(...))` is exactly
/// what a user sees on a bad scene:
///
///   Frame {}, Newton {}, PCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. Hint: {}.
///   Frame {}, Newton {}, PCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. Hint: {}.
namespace linear_pcg_detail
{
/// `!std::isfinite(rz)` — the guard's trigger, for both variants.
[[nodiscard]] inline bool rz_nan_inf(Float rz) noexcept
{
    return !std::isfinite(rz);
}

[[nodiscard]] std::string init_message(SizeT frame,
                                       SizeT newton_iter,
                                       Float rz,
                                       Float norm_r,
                                       Float norm_z);
[[nodiscard]] std::string iter_message(SizeT frame,
                                       SizeT newton_iter,
                                       SizeT k,
                                       Float rz,
                                       Float norm_r,
                                       Float norm_z);
}  // namespace linear_pcg_detail

class LinearPCG : public IterativeSolver
{
  public:
    using IterativeSolver::IterativeSolver;

    /// `|rz_k| / |rz_0|` for k = 0..`iters` of the *last* solve (the parity gate's
    /// residual curve).  Empty before the first solve.
    [[nodiscard]] const std::vector<Float>& residual_trace() const noexcept
    {
        return m_rz_trace;
    }

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) override;

  private:
    using DeviceDenseVector = lla::LDenseVector<Float>;

    SizeT pcg(lla::LDenseVectorView<Float>  x,
              lla::LCDenseVectorView<Float> b,
              SizeT                         max_iter);
    void  check_init_rz_nan_inf(Float rz);
    void  check_iter_rz_nan_inf(Float rz, SizeT k);

    DeviceDenseVector        r0;  // initial residual
    DeviceDenseVector        z;   // preconditioned residual
    DeviceDenseVector        r;   // residual
    DeviceDenseVector        p;   // search direction
    DeviceDenseVector        Ap;  // A*p
    lud::LDeviceVar<IndexT>  d_converged_false;

    Float max_iter_ratio  = 2.0;
    Float global_tol_rate = 1e-4;
    Float reserve_ratio   = 1.5;

    bool        need_debug_dump = false;
    std::string debug_dump_path;

    /// The residual curve of the last solve (deviation 2 above).
    std::vector<Float> m_rz_trace;

    // The effective configuration the fused mapping logs once at `do_build()`.
    // Read + validated + ignored: none of them may change a number (the
    // observable contract of plan §3.3 / research §3.4.2).
    std::string m_solver_name;
    IndexT      m_check_interval = 5;
    IndexT      m_use_cuda_graph = 0;
};
}  // namespace uipc::backend::luisa
