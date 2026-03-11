#pragma once
#include "../point_point.h"

namespace uipc::backend::luisa::distance
{
// Compute squared distance between two 3D points
// dist2 = ||a - b||^2
inline void point_point_distance2(const Vector3& a,
                                  const Vector3& b,
                                  Float&         dist2)
{
    auto diff = a - b;
    dist2     = luisa::dot(diff, diff);
}

// Compute gradient of squared distance
// grad = [2*(a-b), -2*(a-b)]
inline void point_point_distance2_gradient(const Vector3& a,
                                           const Vector3& b,
                                           float          grad[6])
{
    auto diff = 2.0f * (a - b);
    
    // First 3 elements: 2*(a - b)
    grad[0] = diff.x;
    grad[1] = diff.y;
    grad[2] = diff.z;
    
    // Last 3 elements: -2*(a - b)
    grad[3] = -diff.x;
    grad[4] = -diff.y;
    grad[5] = -diff.z;
}

// Compute Hessian of squared distance
// Hessian = [[2*I, -2*I], [-2*I, 2*I]] where I is 3x3 identity
inline void point_point_distance2_hessian(const Vector3& a,
                                          const Vector3& b,
                                          float          Hessian[36])
{
    // Initialize all elements to zero
    for (int i = 0; i < 36; ++i)
    {
        Hessian[i] = 0.0f;
    }
    
    // Set diagonal elements to 2.0
    // (0,0), (1,1), (2,2), (3,3), (4,4), (5,5)
    for (int i = 0; i < 6; ++i)
    {
        Hessian[i * 6 + i] = 2.0f;
    }
    
    // Set off-diagonal coupling terms to -2.0
    // Hessian(0, 3) = Hessian(3, 0) = -2.0
    // Hessian(1, 4) = Hessian(4, 1) = -2.0
    // Hessian(2, 5) = Hessian(5, 2) = -2.0
    Hessian[0 * 6 + 3] = -2.0f;  // (3, 0) in row-major = [0][3]
    Hessian[1 * 6 + 4] = -2.0f;  // (4, 1) in row-major = [1][4]
    Hessian[2 * 6 + 5] = -2.0f;  // (5, 2) in row-major = [2][5]
    Hessian[3 * 6 + 0] = -2.0f;  // (0, 3) in row-major = [3][0]
    Hessian[4 * 6 + 1] = -2.0f;  // (1, 4) in row-major = [4][1]
    Hessian[5 * 6 + 2] = -2.0f;  // (2, 5) in row-major = [5][2]
}

}  // namespace uipc::backend::luisa::distance
