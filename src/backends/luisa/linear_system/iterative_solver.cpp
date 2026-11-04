#include <linear_system/iterative_solver.h>
#include <linear_system/global_linear_system.h>
#include <array>

/*************************************************************************************************
* T8.3a-2 — IterativeSolver (port of src/backends/cuda/linear_system/iterative_solver.cu).
*
* Every body below is the reference's, one line per member.  The four
* `m_system->m_impl.*` forwarders are why this task also had to land
* `GlobalLinearSystem::Impl::{spmv, spmv_dot, matrix_data_ptrs,
* solve_linear_system}` (global_linear_system.cpp): defining them here ODR-uses
* those bodies, so they cannot stay `SCOPE NOTE (T8.3)`.
*
* `m_system` is only reachable because `IterativeSolver` is in
* `GlobalLinearSystem`'s friend list (the reference's list, unchanged).
*************************************************************************************************/
namespace uipc::backend::luisa
{
void IterativeSolver::do_build()
{
    m_system = &require<GlobalLinearSystem>();

    BuildInfo info;
    do_build(info);

    m_system->add_solver(this);
}

void IterativeSolver::spmv(Float                        a,
                           lla::LCDenseVectorView<Float> x,
                           Float                         b,
                           lla::LDenseVectorView<Float>  y)
{
    m_system->m_impl.spmv(a, x, b, y);
}

void IterativeSolver::spmv(lla::LCDenseVectorView<Float> x, lla::LDenseVectorView<Float> y)
{
    spmv(1.0, x, 0.0, y);
}

void IterativeSolver::spmv_dot(lla::LCDenseVectorView<Float> x,
                               lla::LDenseVectorView<Float>  y,
                               lud::LBufferView<Float>       d_dot,
                               lud::LStream*                 stream)
{
    m_system->m_impl.spmv_dot(x, y, d_dot, stream);
}

void IterativeSolver::apply_preconditioner(lla::LDenseVectorView<Float>  z,
                                           lla::LCDenseVectorView<Float> r,
                                           lud::LCBufferView<IndexT>     converged,
                                           lud::LStream*                 stream)
{
    m_system->m_impl.apply_preconditioner(z, r, converged, stream);
}

std::array<const void*, 3> IterativeSolver::matrix_data_ptrs() const
{
    return m_system->m_impl.matrix_data_ptrs();
}


bool IterativeSolver::accuracy_statisfied(lla::LDenseVectorView<Float> r)
{
    return m_system->m_impl.accuracy_statisfied(r);
}

lla::LinearSystemContext& IterativeSolver::ctx() const
{
    return m_system->m_impl.ctx;
}

void IterativeSolver::solve(GlobalLinearSystem::SolvingInfo& info)
{
    do_solve(info);
}
}  // namespace uipc::backend::luisa
