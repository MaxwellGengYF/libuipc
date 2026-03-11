#pragma once
#include <luisa/luisa-compute.h>
#include <utils/distance.h>
#include <utils/friction_utils.h>

namespace uipc::backend::luisa
{
namespace sym::codim_ipc_contact
{
#include "sym/codim_ipc_contact.inl"

    // C1 clamping
    inline void f0(Float x2, Float epsvh, Float& f0)
    {
        if(x2 >= epsvh * epsvh)
        {
            //tex: $$y$$
            f0 = sqrt(x2);
        }
        else
        {
            //tex: $$\frac{y^{2}}{\epsilon_{x}} + \frac{1}{3 \epsilon_{x}} - \frac{y^{3}}{3 \epsilon_{x}^{2}}$$
            f0 = x2 * (-sqrt(x2) / 3.0f + epsvh) / (epsvh * epsvh) + epsvh / 3.0f;
        }
    }

    inline void f1_div_rel_dx_norm(Float x2, Float epsvh, Float& result)
    {
        if(x2 >= epsvh * epsvh)
        {
            //tex: $$ \frac{1}{y}$$
            result = 1.0f / sqrt(x2);
        }
        else
        {
            //tex: $$ \frac{2 \epsilon_{x} - y}{ \epsilon_{x}^{2}}$$
            result = (-sqrt(x2) + 2.0f * epsvh) / (epsvh * epsvh);
        }
    }

    inline void f2_term(Float x2, Float epsvh, Float& term)
    {
        term = -1.0f / (epsvh * epsvh);
        // same for x2 >= epsvh * epsvh for C1 clamped friction
    }


    inline Float friction_energy(Float mu, Float lambda, Float eps_vh, Float2 tan_rel_x)
    {
        Float f0_val;
        f0(luisa::dot(tan_rel_x, tan_rel_x), eps_vh, f0_val);
        return mu * lambda * f0_val;
    }

    inline void friction_gradient(Float2& G2, Float mu, Float lambda, Float eps_vh, Float2 tan_rel_x)
    {
        Float f1_val;
        f1_div_rel_dx_norm(luisa::dot(tan_rel_x, tan_rel_x), eps_vh, f1_val);
        G2 = mu * lambda * f1_val * tan_rel_x;
    }

    inline void make_spd(Float2x2& mat)
    {
        // For luisa-compute, we need to implement SPD projection
        // Using simple eigenvalue clamping via explicit formula for 2x2
        Float a = mat[0][0];
        Float b = mat[0][1];
        Float c = mat[1][0];
        Float d = mat[1][1];
        
        // Compute eigenvalues of symmetric 2x2 matrix
        Float trace = a + d;
        Float det = a * d - b * c;
        Float discriminant = sqrt(max(0.0f, trace * trace - 4.0f * det));
        
        Float lambda1 = (trace + discriminant) * 0.5f;
        Float lambda2 = (trace - discriminant) * 0.5f;
        
        // Clamp to non-negative
        lambda1 = max(lambda1, 0.0f);
        lambda2 = max(lambda2, 0.0f);
        
        // Reconstruct: mat = V * diag(lambda) * V^T
        // For 2x2, we use the explicit formula
        if(discriminant > 1e-6f) {
            Float diff = lambda1 - lambda2;
            Float v11 = b;
            Float v12 = lambda1 - a;
            Float v21 = b;
            Float v22 = lambda2 - a;
            
            Float n1 = sqrt(v11 * v11 + v12 * v12);
            Float n2 = sqrt(v21 * v21 + v22 * v22);
            
            if(n1 > 1e-6f && n2 > 1e-6f) {
                v11 /= n1; v12 /= n1;
                v21 /= n2; v22 /= n2;
                
                mat[0][0] = lambda1 * v11 * v11 + lambda2 * v21 * v21;
                mat[0][1] = lambda1 * v11 * v12 + lambda2 * v21 * v22;
                mat[1][0] = mat[0][1];
                mat[1][1] = lambda1 * v12 * v12 + lambda2 * v22 * v22;
            } else {
                // Fallback to diagonal
                mat[0][0] = lambda1;
                mat[0][1] = 0.0f;
                mat[1][0] = 0.0f;
                mat[1][1] = lambda2;
            }
        } else {
            // Matrix is already diagonal-dominant or scalar
            mat[0][0] = lambda1;
            mat[0][1] = 0.0f;
            mat[1][0] = 0.0f;
            mat[1][1] = lambda1;
        }
    };

    inline void friction_hessian(Float2x2& H2x2, Float mu, Float lambda, Float eps_vh, Float2 tan_rel_x)
    {
        Float sq_norm = luisa::dot(tan_rel_x, tan_rel_x);
        Float epsvh2  = eps_vh * eps_vh;
        Float f1_div_rel_dx_norm_val;
        f1_div_rel_dx_norm(sq_norm, eps_vh, f1_div_rel_dx_norm_val);

        if(sq_norm >= epsvh2)
        {
            // no SPD projection needed
            Float2 ubar = make_float2(-tan_rel_x.y, tan_rel_x.x);
            // H2x2 = ubar * (mu * lambda * f1_div_rel_dx_norm_val / sq_norm) * ubar.transpose();
            Float scale = mu * lambda * f1_div_rel_dx_norm_val / sq_norm;
            H2x2[0][0] = scale * ubar.x * ubar.x;
            H2x2[0][1] = scale * ubar.x * ubar.y;
            H2x2[1][0] = scale * ubar.y * ubar.x;
            H2x2[1][1] = scale * ubar.y * ubar.y;
        }
        else
        {
            if(sq_norm < 1e-12f)
            {
                // no SPD projection needed
                H2x2 = make_float2x2(f1_div_rel_dx_norm_val);  // Diagonal matrix
            }
            else
            {
                Float f2_term_val;
                f2_term(sq_norm, eps_vh, f2_term_val);

                Float relDXNorm = sqrt(sq_norm);

                // only need to project the inner 2x2 matrix to SPD
                Float scale = f2_term_val / relDXNorm;
                H2x2[0][0] = tan_rel_x.x * scale * tan_rel_x.x + f1_div_rel_dx_norm_val;
                H2x2[0][1] = tan_rel_x.x * scale * tan_rel_x.y;
                H2x2[1][0] = tan_rel_x.y * scale * tan_rel_x.x;
                H2x2[1][1] = tan_rel_x.y * scale * tan_rel_x.y + f1_div_rel_dx_norm_val;

                make_spd(H2x2);

                H2x2[0][0] *= mu * lambda;
                H2x2[0][1] *= mu * lambda;
                H2x2[1][0] *= mu * lambda;
                H2x2[1][1] *= mu * lambda;
            }
        }
    }

    inline Float normal_force(Float kappa, Float d_hat, Float thickness, Float D)
    {
        using namespace distance;
        Float d = sqrt(D);
        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        Float dBdd = dBdD * 2.0f * d;
        return -dBdd;  // > 0
    }
}  // namespace sym::codim_ipc_contact
}  // namespace uipc::backend::luisa
