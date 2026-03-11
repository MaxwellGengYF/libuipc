#pragma once
#include <linear_system/iterative_solver.h>

namespace uipc::backend::luisa
{
class LinearPCG : public IterativeSolver
{
  public:
    using IterativeSolver::IterativeSolver;


  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) override;

  private:
    // LuisaCompute Buffer types for linear algebra
    using BufferFloat       = luisa::compute::Buffer<Float>;
    using BufferViewFloat   = luisa::compute::BufferView<Float>;
    using CBufferViewFloat  = luisa::compute::BufferView<const Float>;

    SizeT pcg(BufferViewFloat x, CBufferViewFloat b, SizeT max_iter);
    void dump_r_z(SizeT k);
    void dump_p_Ap(SizeT k);
    void check_init_rz_nan_inf(Float rz);
    void check_iter_rz_nan_inf(Float rz, SizeT k);

    BufferFloat r0;  // initial residual
    BufferFloat z;   // preconditioned residual
    BufferFloat r;   // residual
    BufferFloat p;   // search direction
    BufferFloat Ap;  // A*p

    Float max_iter_ratio  = 2.0;
    Float global_tol_rate = 1e-4;
    Float reserve_ratio   = 1.5;

    bool        need_debug_dump = false;
    std::string debug_dump_path;
};
}  // namespace uipc::backend::luisa
