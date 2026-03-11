#pragma once
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
namespace distance
{
    // Kappa Barrier functions
    template <typename T>
    inline void KappaBarrier(T& R, const T& kappa, const T& D, const T& dHat, const T& xi)
    {
        auto x0 = xi * xi;
        auto x1 = dHat * dHat + 2.0f * dHat * xi;
        R = -kappa * pow(D - x0 - x1, 2) * log((D - x0) / x1);
    }

    template <typename T>
    inline void dKappaBarrierdD(T& R, const T& kappa, const T& D, const T& dHat, const T& xi)
    {
        auto x0 = xi * xi;
        auto x1 = D - x0;
        auto x2 = dHat * dHat;
        auto x3 = dHat * xi;
        auto x4 = x2 + 2.0f * x3;
        R = -kappa * (2.0f * D - 2.0f * x0 - 2.0f * x2 - 4.0f * x3) * log(x1 / x4) - kappa * pow(D - x0 - x4, 2) / x1;
    }

    template <typename T>
    inline void ddKappaBarrierddD(T& R, const T& kappa, const T& D, const T& dHat, const T& xi)
    {
        auto x0 = xi * xi;
        auto x1 = D - x0;
        auto x2 = dHat * dHat;
        auto x3 = dHat * xi;
        auto x4 = x2 + 2.0f * x3;
        auto x5 = 2.0f * kappa;
        R = kappa * pow(D - x0 - x4, 2) / (x1 * x1) - x5 * log(x1 / x4) - x5 * (2.0f * D - 2.0f * x0 - 2.0f * x2 - 4.0f * x3) / x1;
    }

}  // namespace distance
}  // namespace uipc::backend::luisa
