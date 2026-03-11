#pragma once
#include <backends/luisa/type_define.h>
#include <Eigen/Geometry>

namespace uipc::backend::luisa
{
// float based AABB
using AABB = Eigen::AlignedBox<float, 3>;
}  // namespace uipc::backend::luisa

// Make AABB available for luisa-compute DSL
LUISA_STRUCT(uipc::backend::luisa::AABB, m_min, m_max) {};
