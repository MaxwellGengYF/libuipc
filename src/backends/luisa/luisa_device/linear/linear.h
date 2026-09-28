#pragma once
/********************************************************************
 * @file linear.h
 * @brief Umbrella header of the linear-algebra layer (T3.1).
 *
 * Mirrors `cuda_tool/linear_system.h`, which is what the referencing files
 * include: containers + views in one stop.  Ported consumer code therefore only
 * changes its include
 *
 *     #include <cuda_tool/linear_system.h>          ->  #include <luisa_device/linear/linear.h>
 *     cuda_tool::DeviceTripletMatrix<Float,3>        ->  lla::LTripletMatrix<Float,3>
 *                                                        (or the alias DeviceTripletMatrix)
 *
 * The namespace is `uipc::backend::luisa::luisa_device::linalg` (cookbook §5),
 * with the short alias `lla` (next to `lps` for the numerics layer).
 *
 * Out of scope in T3.1 (deliberately not ported here)
 * ---------------------------------------------------
 *  * `details::LinearReductionStorage` / `LinearSystemContext` — the persistent
 *    dot/norm reduction storage and its CUB-backed reductions.  That is the T3.4
 *    blas/spmv work (`linear/blas.h`, `linear/spmv.h`).
 *  * The cusparse descriptor carriers (`COOMatrixViewT`, `COOVectorViewBase`) —
 *    LC has no cuSPARSE; the descriptors only existed to hand handles to
 *    cuSPARSE calls that D4 replaces with LC kernels.
 *  * `cuda_tool::VarView`, `CDense` — outside the linear_system headers (T1.2's
 *    `LDeviceVar`/`LBufferView` are their replacement).
 *  * Raw element pointers (`view.data()`, `values().data()`): a buffer DSL has no
 *    host-visible device pointer.  Pass the handles as kernel arguments and index
 *    them inside the kernel (`view.value_at(i, r, c)`, `view.block(i)`).
 *********************************************************************/
#include <luisa_device/linear/containers.h>
#include <luisa_device/linear/views.h>

namespace uipc::backend::luisa::luisa_device
{
/// Short alias for the linear-algebra layer (`lla::LTripletMatrix<Float,3>`).
namespace lla = linalg;
}// namespace uipc::backend::luisa::luisa_device
