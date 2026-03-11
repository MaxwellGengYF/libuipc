#pragma once
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
    struct ContactCoeff
    {
        Float kappa;  // Contact stiffness
        Float mu;     // Friction coefficient
    };
}  // namespace uipc::backend::luisa
