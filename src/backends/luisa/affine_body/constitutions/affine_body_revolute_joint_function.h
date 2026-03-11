#pragma once
#include <type_define.h>

namespace uipc::backend::luisa
{
namespace sym::affine_body_revolute_joint
{
    inline LUISA_GENERIC Float E(const luisa::Vector<Float, 12>& X)
    {
        // X.segment<3>(0) - X.segment<3>(6)
        luisa::Vector<Float, 3> seg0{X[0] - X[6], X[1] - X[7], X[2] - X[8]};
        // X.segment<3>(3) - X.segment<3>(9)
        luisa::Vector<Float, 3> seg1{X[3] - X[9], X[4] - X[10], X[5] - X[11]};
        
        Float E0 = luisa::dot(seg0, seg0);
        Float E1 = luisa::dot(seg1, seg1);
        return (E0 + E1) / 2;
    }

#include "sym/affine_body_revolute_joint.inl"
}  // namespace sym::affine_body_revolute_joint
}  // namespace uipc::backend::luisa
