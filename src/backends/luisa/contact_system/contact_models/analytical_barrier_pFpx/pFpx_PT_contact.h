#pragma once
#include <luisa/core/basic_types.h>

namespace uipc::backend::luisa::analyticalBarrier
{

template <class T>
void analytical_point_triangle_pFpx(const luisa::Vector<T, 3>& p,
                                    const luisa::Vector<T, 3>& t1,
                                    const luisa::Vector<T, 3>& t2,
                                    const luisa::Vector<T, 3>& t3,
                                    T d_hatSqrt,
                                    T result[12][9]);

}

#include "details/pFpx_PT_contact.inl"
