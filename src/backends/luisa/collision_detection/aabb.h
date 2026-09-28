#pragma once
/********************************************************************
 * @file aabb.h
 * @brief Axis-aligned bounding box (port of the reference's
 *        src/backends/cuda/collision_detection/aabb.h).
 *
 * Host-only Eigen type, so the port is a namespace change. Kept at the same
 * relative path as the reference (`collision_detection/aabb.h`) so later
 * collision-detection ports compile by changing only the include prefix.
 *********************************************************************/
#include <type_define.h>
#include <Eigen/Geometry>

namespace uipc::backend::luisa
{
// float based AABB
using AABB = Eigen::AlignedBox<float, 3>;
}  // namespace uipc::backend::luisa
