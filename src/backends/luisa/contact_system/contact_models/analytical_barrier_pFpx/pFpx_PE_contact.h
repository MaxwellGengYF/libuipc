#pragma once
#include <backends/luisa/type_define.h>

namespace uipc::backend::luisa::analyticalBarrier
{


template <class T>
LC_DEVICE void analytical_point_edge_pFpx(const luisa::float3& p,
                                             const luisa::float3& e1,
                                             const luisa::float3& e2,
                                             T d_hatSqrt,
                                             T result[9][4]);


}

#include "details/pFpx_PE_contact.inl"
