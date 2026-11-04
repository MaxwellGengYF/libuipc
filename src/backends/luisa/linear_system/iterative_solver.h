#pragma once
/********************************************************************
 * @file iterative_solver.h
 * @brief T8.3a-2 — the iterative-solver base (port of
 *        src/backends/cuda/linear_system/iterative_solver.h — namespace + types
 *        only; the class surface is the reference's, token for token).
 *
 * WHAT THIS IS
 * ------------
 * An `IterativeSolver` is a `SimSystem` whose whole life is two overrides:
 *   * `do_build(BuildInfo&)` — the derived solver validates its configuration and
 *     declares its dependencies;
 *   * `do_solve(GlobalLinearSystem::SolvingInfo&)` — it fills `info.x()` from
 *     `info.b()` and reports the iteration count.
 * `do_build()` (the `final` one overridden below) closes the loop: it resolves
 * the `GlobalLinearSystem`, calls the derived `do_build`, and *then* registers
 * itself through `add_solver(this)` — registration therefore cannot be
 * forgotten, and a solver that throws `SimSystemException("… unused")` from its
 * own `do_build` never registers.  `GlobalLinearSystem::Impl::solve_linear_system()`
 * reaches `solve()` (private + `friend class GlobalLinearSystem`, exactly as in
 * the reference: a nested class has its enclosing class's access rights,
 * [class.access.nest]).
 *
 * The protected utility block is the reference's, translated mechanically:
 *
 *   cuda_tool::CDenseVectorView<Float>  -> lla::LCDenseVectorView<Float>
 *   cuda_tool::DenseVectorView<Float>   -> lla::LDenseVectorView<Float>
 *   cuda_tool::VarView<Float>           -> lud::LBufferView<Float>
 *   cuda_tool::CVarView<IndexT>         -> lud::LCBufferView<IndexT>
 *   cudaStream_t                        -> lud::LStream*  (nullptr == the default
 *                                          stream, i.e. `lla::default_blas_stream()`
 *                                          == `LStream::current()` == the frame
 *                                          command list)
 *   cuda_tool::LinearSystemContext&     -> lla::LinearSystemContext&
 *
 * Every one of them forwards to `m_system->m_impl` — the reference's placement
 * (they live in global_linear_system.cpp, which T8.3a-2 also un-defers).  There
 * is no second implementation of SpMV anywhere in this backend (plan §"Luisa
 * target" wanted a `linear_spmv.*`; the research §4.1 is right that it would be
 * a duplicate — the primitives are `lla::LSpmv` + these forwarders).
 *
 * DEVIATION (1, cosmetic): the reference leaves `m_system` uninitialised; here it
 * is `nullptr`-initialised so a mis-ordered `solve()` faults instead of reading
 * an indeterminate pointer.  `do_build()` is the only writer, and it runs for
 * every live solver before anything can call `solve()`.
 *
 * DEVIATION (2): `matrix_data_ptrs()` is kept because it is the reference's
 * public surface (and `GlobalLinearSystem` uses it as the FusedPCG cache/dump
 * key), but it is an *identity token* here — see global_linear_system.h.  No raw
 * device pointer exists in a buffer DSL (§7.9), and there is no CUDA graph to
 * key.
 *********************************************************************/
#include <sim_system.h>
#include <luisa_device/linear/linear.h>
#include <linear_system/global_linear_system.h>
#include <array>

namespace uipc::backend::luisa
{
class IterativeSolver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;

    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) = 0;


    /**********************************************************************************************
    * Util functions for derived classes
    ***********************************************************************************************/

    void spmv(Float                        a,
              lla::LCDenseVectorView<Float> x,
              Float                         b,
              lla::LDenseVectorView<Float>  y);
    void spmv(lla::LCDenseVectorView<Float> x, lla::LDenseVectorView<Float> y);
    void spmv_dot(lla::LCDenseVectorView<Float> x,
                  lla::LDenseVectorView<Float>  y,
                  lud::LBufferView<Float>       d_dot,
                  lud::LStream*                 stream = nullptr);
    void apply_preconditioner(lla::LDenseVectorView<Float>  z,
                              lla::LCDenseVectorView<Float> r,
                              lud::LCBufferView<IndexT>     converged,
                              lud::LStream*                 stream = nullptr);
    // data pointers of the assembled system matrix (FusedPCG graph key)
    std::array<const void*, 3> matrix_data_ptrs() const;
    bool                       accuracy_statisfied(lla::LDenseVectorView<Float> r);
    lla::LinearSystemContext&  ctx() const;

  private:
    friend class GlobalLinearSystem;
    GlobalLinearSystem* m_system = nullptr;

    virtual void do_build() final override;

    void solve(GlobalLinearSystem::SolvingInfo& info);
};
}  // namespace uipc::backend::luisa
