#pragma once
#include <luisa/runtime/buffer.h>
#include "type_define.h"

namespace uipc::backend::luisa::distance {

namespace details {
    template <typename T>
    LC_GPU_CALLABLE void g_EECN2(T v01,
                                 T v02,
                                 T v03,
                                 T v11,
                                 T v12,
                                 T v13,
                                 T v21,
                                 T v22,
                                 T v23,
                                 T v31,
                                 T v32,
                                 T v33,
                                 T g[12]);

    template <typename T>
    LC_GPU_CALLABLE void H_EECN2(T v01,
                                 T v02,
                                 T v03,
                                 T v11,
                                 T v12,
                                 T v13,
                                 T v21,
                                 T v22,
                                 T v23,
                                 T v31,
                                 T v32,
                                 T v33,
                                 T H[144]);

    template <typename T>
    LC_GPU_CALLABLE void EEM(T input, T eps_x, T& e);

    template <typename T>
    LC_GPU_CALLABLE void g_EEM(T input, T eps_x, T& g);

    template <typename T>
    LC_GPU_CALLABLE void H_EEM(T input, T eps_x, T& H);
}  // namespace details

template <typename T>
LC_GPU_CALLABLE bool need_mollify(const luisa::Vector<T, 3>& ea0,
                                  const luisa::Vector<T, 3>& ea1,
                                  const luisa::Vector<T, 3>& eb0,
                                  const luisa::Vector<T, 3>& eb1,
                                  T                          eps_x);

template <typename T>
LC_GPU_CALLABLE void edge_edge_cross_norm2(const luisa::Vector<T, 3>& ea0,
                                           const luisa::Vector<T, 3>& ea1,
                                           const luisa::Vector<T, 3>& eb0,
                                           const luisa::Vector<T, 3>& eb1,
                                           T&                         result);

template <typename T>
LC_GPU_CALLABLE void edge_edge_cross_norm2_gradient(const luisa::Vector<T, 3>& ea0,
                                                    const luisa::Vector<T, 3>& ea1,
                                                    const luisa::Vector<T, 3>& eb0,
                                                    const luisa::Vector<T, 3>& eb1,
                                                    luisa::Vector<T, 12>&      grad);

template <typename T>
LC_GPU_CALLABLE void edge_edge_cross_norm2_hessian(const luisa::Vector<T, 3>& ea0,
                                                   const luisa::Vector<T, 3>& ea1,
                                                   const luisa::Vector<T, 3>& eb0,
                                                   const luisa::Vector<T, 3>& eb1,
                                                   luisa::Matrix<T, 12, 12>&  Hessian);

template <typename T>
LC_GPU_CALLABLE void edge_edge_mollifier(const luisa::Vector<T, 3>& ea0,
                                         const luisa::Vector<T, 3>& ea1,
                                         const luisa::Vector<T, 3>& eb0,
                                         const luisa::Vector<T, 3>& eb1,
                                         T                          eps_x,
                                         T&                         e);

template <typename T>
LC_GPU_CALLABLE void edge_edge_mollifier_gradient(const luisa::Vector<T, 3>& ea0,
                                                  const luisa::Vector<T, 3>& ea1,
                                                  const luisa::Vector<T, 3>& eb0,
                                                  const luisa::Vector<T, 3>& eb1,
                                                  T                          eps_x,
                                                  luisa::Vector<T, 12>&      g);

template <typename T>
LC_GPU_CALLABLE void edge_edge_mollifier_hessian(const luisa::Vector<T, 3>& ea0,
                                                 const luisa::Vector<T, 3>& ea1,
                                                 const luisa::Vector<T, 3>& eb0,
                                                 const luisa::Vector<T, 3>& eb1,
                                                 T                          eps_x,
                                                 luisa::Matrix<T, 12, 12>&  H);

template <typename T>
LC_GPU_CALLABLE void edge_edge_mollifier_threshold(const luisa::Vector<T, 3>& ea0_rest,
                                                   const luisa::Vector<T, 3>& ea1_rest,
                                                   const luisa::Vector<T, 3>& eb0_rest,
                                                   const luisa::Vector<T, 3>& eb1_rest,
                                                   Float                    coeff,
                                                   T&                         eps_x);

}  // namespace uipc::backend::luisa::distance

#include "details/edge_edge_mollifier.inl"
