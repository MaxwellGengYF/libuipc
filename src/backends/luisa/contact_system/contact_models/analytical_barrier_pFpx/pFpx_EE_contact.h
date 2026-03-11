#pragma once
#include <backends/luisa/type_define.h>

namespace uipc::backend::luisa::analyticalBarrier
{


template <class T>
LC_DEVICE void analytical_edge_edge_pFpx(const luisa::float3& e0,
                                            const luisa::float3& e1,
                                            const luisa::float3& e2,
                                            const luisa::float3& e3,
                                            T d_hatSqrt,
                                            T result[12][9]);


}

#include "details/pFpx_EE_contact.inl"
