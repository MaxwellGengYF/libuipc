#pragma once
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa::analyticalBarrier
{
using namespace luisa;
using namespace luisa::compute;

/**
 * @brief Compute analytical partial derivatives for parallel edge-edge contact.
 * 
 * @param e0 First endpoint of edge 0 (Var<float3>)
 * @param e1 Second endpoint of edge 0 (Var<float3>)
 * @param e2 First endpoint of edge 1 (Var<float3>)
 * @param e3 Second endpoint of edge 1 (Var<float3>)
 * @param d_hatSqrt Square root of the barrier distance (Float)
 * @param result Output 12x9 Jacobian matrix (Float result[12][9])
 */
void analytical_parallel_edge_edge_pFpx(Expr<float3> e0,
                                        Expr<float3> e1,
                                        Expr<float3> e2,
                                        Expr<float3> e3,
                                        Expr<float>  d_hatSqrt,
                                        Float        result[12][9]) noexcept;

}  // namespace uipc::backend::luisa::analyticalBarrier

#include "details/pFpx_parallel_EE_contact.inl"
