#pragma once
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
namespace friction
{
    /**
     * @brief Clamp a value to [0, 1] range
     */
    inline Float clamp01(Float x)
    {
        return clamp(x, 0.0f, 1.0f);
    }
    
    /**
     * @brief Compute squared distance from point to half-plane
     * @param D Output squared signed distance (negative if inside)
     * @param P Point position
     * @param plane_p Plane reference point
     * @param plane_n Plane normal (should be normalized)
     */
    inline void point_halfplane_distance2(Float& D, const Float3& P, 
                                          const Float3& plane_p, const Float3& plane_n)
    {
        Float dist = dot(P - plane_p, plane_n);
        D = dist * dist;
    }
    
    /**
     * @brief Compute tangent basis for half-plane contact
     * @param basis Output 2x3 tangent basis (basis[0] = e1, basis[1] = e2)
     * @param plane_n Plane normal
     */
    inline void halfplane_tangent_basis(Float3& e1, Float3& e2, const Float3& plane_n)
    {
        // Choose a trial vector not parallel to plane_n
        Float3 trial = make_float3(1.0f, 0.0f, 0.0f);
        if(abs(dot(plane_n, trial)) > 0.9f) {
            trial = make_float3(0.0f, 0.0f, 1.0f);
        }
        
        // First tangent vector
        e1 = normalize(cross(trial, plane_n));
        
        // Second tangent vector
        e2 = cross(plane_n, e1);
    }
    
    /**
     * @brief Compute tangent relative displacement for half-plane contact
     * @param tan_rel_dx Output 2D tangent displacement
     * @param dx 3D displacement at contact point
     * @param e1 First tangent basis vector
     * @param e2 Second tangent basis vector
     */
    inline void halfplane_tan_rel_dx(Float2& tan_rel_dx, const Float3& dx,
                                     const Float3& e1, const Float3& e2)
    {
        tan_rel_dx.x = dot(dx, e1);
        tan_rel_dx.y = dot(dx, e2);
    }
    
    /**
     * @brief Compute Jacobian for half-plane contact (3 DOFs for point)
     * @param J Output 2x3 Jacobian matrix (as array: J[row][col])
     * @param e1 First tangent basis vector
     * @param e2 Second tangent basis vector
     */
    inline void halfplane_jacobi(Float J[2][3], const Float3& e1, const Float3& e2)
    {
        // J maps from point displacement to tangent displacement
        J[0][0] = e1.x; J[0][1] = e1.y; J[0][2] = e1.z;
        J[1][0] = e2.x; J[1][1] = e2.y; J[1][2] = e2.z;
    }
    
}  // namespace friction
}  // namespace uipc::backend::luisa
