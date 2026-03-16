#include <linear_system/iterative_solver.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::luisa
{
void IterativeSolver::do_build()
{
    m_system = &require<GlobalLinearSystem>();

    BuildInfo info;
    do_build(info);

    m_system->add_solver(this);
}

void IterativeSolver::spmv(Float                                  a,
                           luisa::compute::BufferView<const Float> x,
                           Float                                  b,
                           luisa::compute::BufferView<Float>       y)
{
    m_system->m_impl.spmv(a, x, b, y);
}

void IterativeSolver::spmv(luisa::compute::BufferView<const Float> x, luisa::compute::BufferView<Float> y)
{
    spmv(1.0, x, 0.0, y);
}

void IterativeSolver::spmv_dot(luisa::compute::BufferView<const Float> x,
                               luisa::compute::BufferView<Float>       y,
                               luisa::compute::BufferView<Float>       d_dot)
{
    m_system->m_impl.spmv_dot(x, y, d_dot);
}

void IterativeSolver::apply_preconditioner(luisa::compute::BufferView<Float>       z,
                                           luisa::compute::BufferView<const Float> r)
{
    m_system->m_impl.apply_preconditioner(z, r);
}

bool IterativeSolver::accuracy_statisfied(luisa::compute::BufferView<Float> r)
{
    return m_system->m_impl.accuracy_statisfied(r);
}

void IterativeSolver::solve(GlobalLinearSystem::SolvingInfo& info)
{
    do_solve(info);
}
}  // namespace uipc::backend::luisa
