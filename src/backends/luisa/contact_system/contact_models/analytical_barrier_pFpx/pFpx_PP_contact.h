#pragma once
#include <backends/luisa/type_define.h>

namespace uipc::backend::luisa::analyticalBarrier
{


template <class T>
LC_DEVICE void analytical_point_point_pFpx(const luisa::float3& p0,
                                              const luisa::float3& p1,
                                              T d_hatSqrt,
                                              T result[6]);


}

#include "details/pFpx_PP_contact.inl"
