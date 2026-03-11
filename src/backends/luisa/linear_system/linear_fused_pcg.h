#pragma once
#include <linear_system/iterative_solver.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
// Fused PCG: keeps dot-product scalars (rz, pAp, rz_new) on device
// to eliminate per-iteration host synchronizations.  The update kernels read
// alpha = rz/pAp and beta = rz_new/rz directly from device memory.
// SpMV and dot(p,Ap) are fused into a single kernel pass.
// Convergence is checked every `check_interval` iterations via a single D2H copy.
class LinearFusedPCG : public IterativeSolver
{
  public:
    using IterativeSolver::IterativeSolver;

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) override;

  private:
    SizeT fused_pcg(luisa::compute::BufferView<Float>       x,
                    luisa::compute::BufferView<const Float> b,
                    SizeT                                   max_iter);

    // Device buffers for PCG vectors (using LuisaCompute Buffer instead of muda DeviceDenseVector)
    luisa::compute::Buffer<Float> r;
    luisa::compute::Buffer<Float> z;
    luisa::compute::Buffer<Float> p;
    luisa::compute::Buffer<Float> Ap;

    // Device scalars for dot products (using 1-element buffers instead of muda DeviceVar)
    luisa::compute::Buffer<Float> d_rz;
    luisa::compute::Buffer<Float> d_pAp;
    luisa::compute::Buffer<Float> d_rz_new;

    Float max_iter_ratio  = 2.0;
    Float global_tol_rate = 1e-4;
    Float reserve_ratio   = 1.5;
    SizeT check_interval  = 5;
};
}  // namespace uipc::backend::luisa
