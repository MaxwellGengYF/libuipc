#pragma once
#include <luisa/luisa-compute.h>
#include <utils/dihedral_angle.h>

//ref: https://www.cs.columbia.edu/cg/pdfs/10_ds.pdf
namespace uipc::backend::luisa
{
namespace sym::discrete_shell_bending
{
#include "sym/discrete_shell_bending.inl"

    inline void compute_constants(float&         L0,
                                  float&         h_bar,
                                  float&         theta_bar,
                                  float&         V_bar,
                                  const float3& x0_bar,
                                  const float3& x1_bar,
                                  const float3& x2_bar,
                                  const float3& x3_bar,
                                  float          thickness0,
                                  float          thickness1,
                                  float          thickness2,
                                  float          thickness3)

    {
        using luisa::cross;
        using luisa::length;

        L0         = length(x2_bar - x1_bar);
        float3 n1 = cross(x1_bar - x0_bar, x2_bar - x0_bar);
        float3 n2 = cross(x2_bar - x3_bar, x1_bar - x3_bar);
        float   A  = (length(n1) + length(n2)) / 2.0f;
        h_bar      = A / 3.0f / L0;
        dihedral_angle(x0_bar, x1_bar, x2_bar, x3_bar, theta_bar);

        float thickness = (thickness0 + thickness1 + thickness2 + thickness3) / 4.0f;
        V_bar = A * thickness;
    }

    inline float E(const float3& x0,
                   const float3& x1,
                   const float3& x2,
                   const float3& x3,
                   float          L0,
                   float          h_bar,
                   float          theta_bar,
                   float          kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        float R;
        DSB::E(R, kappa, theta, theta_bar, L0, h_bar);

        return R;
    }

    inline void dEdx(luisa::Vector<float, 12>& G,
                     const float3& x0,
                     const float3& x1,
                     const float3& x2,
                     const float3& x3,
                     float          L0,
                     float          h_bar,
                     float          theta_bar,
                     float          kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        float dEdtheta;
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);

        luisa::Vector<float, 12> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        G = dthetadx * dEdtheta;
    }

    inline void ddEddx(luisa::Matrix<float, 12>&   H,
                       const float3& x0,
                       const float3& x1,
                       const float3& x2,
                       const float3& x3,
                       float          L0,
                       float          h_bar,
                       float          theta_bar,
                       float          kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        float dEdtheta;
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);

        float ddEddtheta;
        DSB::ddEddtheta(ddEddtheta, kappa, theta, theta_bar, L0, h_bar);

        luisa::Vector<float, 12> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        luisa::Matrix<float, 12> ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        // H = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;
        // We need to compute: outer_product(dthetadx, dthetadx) * ddEddtheta + ddthetaddx * dEdtheta
        
        // First term: ddEddtheta * dthetadx * dthetadx^T
        // Second term: dEdtheta * ddthetaddx
        
        // Compute outer product of dthetadx with itself, scaled by ddEddtheta
        for(int i = 0; i < 12; i++) {
            for(int j = 0; j < 12; j++) {
                H[i][j] = ddEddtheta * dthetadx[i] * dthetadx[j] + dEdtheta * ddthetaddx[i][j];
            }
        }
    }

}  // namespace sym::discrete_shell_bending
}  // namespace uipc::backend::luisa
