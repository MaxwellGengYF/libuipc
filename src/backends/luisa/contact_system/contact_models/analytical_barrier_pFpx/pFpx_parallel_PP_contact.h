#pragma once
#include <backends/luisa/type_define.h>

namespace uipc::backend::luisa::analyticalBarrier
{


template <class T>
LC_DEVICE void analytical_parallel_point_point_pFpx(const luisa::float3& e0,
                                                       const luisa::float3& e2,
                                                       const luisa::float3& e1,
                                                       const luisa::float3& e3,
                                                       T d_hatSqrt,
                                                       T result[12][9]);


}

#include "details/pFpx_parallel_PP_contact.inl"
