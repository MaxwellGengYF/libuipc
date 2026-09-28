#pragma once
/********************************************************************
 * @file point_point.inl
 * @brief T2.3 — point-point squared distance, gradient and Hessian.
 *
 * Ported from `src/backends/cuda/utils/distance/details/point_point.inl` (which
 * is Eigen-only, 29 lines) onto `LVec<T,N>`/`LMat<T,N,N>`:
 *
 *   (a - b).squaredNorm()            -> math::squaredNorm(a - b)
 *   Eigen::Vector<T,6>               -> LVec<T,6>
 *   Hessian.setZero() / .diagonal()  -> the same member spellings of LMat/LVec
 *
 * The scalar literals go through `math::make_scalar<T>` because a plain `2.0`
 * assigned to a `Var<double>` reference is only convertible implicitly; being
 * explicit keeps the DSL path obvious.
 *********************************************************************/
namespace uipc::backend::luisa::luisa_device::math::distance
{
template <typename T>
void point_point_distance2(const LVec<T, 3>& a, const LVec<T, 3>& b, T& dist2)
{
    // (a - b).squaredNorm(); `auto` materialises the difference once, so the DSL
    // path does not build the same sub-expressions twice.
    auto d = a - b;
    dist2  = math::squaredNorm(d);
}

template <typename T>
void point_point_distance2_gradient(const LVec<T, 3>& a,
                                   const LVec<T, 3>& b,
                                   LVec<T, 6>&       grad)
{
    grad.template segment<3>(0) = 2.0 * (a - b);
    grad.template segment<3>(3) = -grad.template segment<3>(0);
}

template <typename T>
void point_point_distance2_hessian(const LVec<T, 3>& a,
                                   const LVec<T, 3>& b,
                                   LMat<T, 6, 6>&    Hessian)
{
    Hessian.setZero();
    Hessian.diagonal().setConstant(math::make_scalar<T>(2.0));
    Hessian(0, 3) = Hessian(1, 4) = Hessian(2, 5) = Hessian(3, 0) = Hessian(4, 1) =
        Hessian(5, 2) = make_scalar<T>(-2.0);
}
}  // namespace uipc::backend::luisa::luisa_device::math::distance
