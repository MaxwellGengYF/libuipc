#pragma once
#include <sim_system.h>
#include <luisa/runtime/buffer.h>
#include <linear_system/global_linear_system.h>

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

    void spmv(Float a, luisa::compute::BufferView<const Float> x, Float b, luisa::compute::BufferView<Float> y);
    void spmv(luisa::compute::BufferView<const Float> x, luisa::compute::BufferView<Float> y);
    void spmv_dot(luisa::compute::BufferView<const Float> x, luisa::compute::BufferView<Float> y, luisa::compute::BufferView<Float> d_dot);
    void apply_preconditioner(luisa::compute::BufferView<Float>  z,
                              luisa::compute::BufferView<const Float> r);
    bool accuracy_statisfied(luisa::compute::BufferView<Float> r);

  private:
    friend class GlobalLinearSystem;
    GlobalLinearSystem* m_system;

    virtual void do_build() final override;

    void solve(GlobalLinearSystem::SolvingInfo& info);
};
}  // namespace uipc::backend::luisa
