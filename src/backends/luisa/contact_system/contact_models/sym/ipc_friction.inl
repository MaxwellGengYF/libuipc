// ipc_friction.inl - Generated symbolic friction functions for IPC contact
// This file provides inline functions for friction energy computation

#pragma once

namespace uipc::backend::luisa
{
namespace sym::ipc_contact
{

    /**
     * @brief Derivative of kappa barrier with respect to distance
     * 
     * R = -kappa * (2*D - 2*dHat) * log(D/dHat) - kappa * (D - dHat)^2 / D
     * 
     * @param R      Output: derivative value
     * @param kappa  Contact stiffness  
     * @param D      Squared distance
     * @param dHat   Squared contact distance threshold
     */
    inline LC_GPU_CALLABLE void dKappaBarrierdD(Float& R, Float kappa, Float D, Float dHat)
    {
        Float x0 = D - dHat;
        Float x1 = D / dHat;
        // Simplified: d/dD [-kappa*(D-dHat)^2*log(D/dHat)]
        R = -kappa * (2.0f * x0 * log(x1) + x0 * x0 / D);
    }

    /**
     * @brief Compute friction energy
     * 
     * Uses the smooth friction model from IPC
     * 
     * @param F      Output: friction energy
     * @param lam_mu lambda * mu (normal force * friction coefficient)
     * @param eps_v  Velocity threshold
     * @param dt     Time step
     * @param y      Tangential displacement magnitude (||vk|| * dt)
     */
    inline LC_GPU_CALLABLE void FrictionEnergy(Float& F, Float lam_mu, Float eps_v, Float dt, Float y)
    {
        if(y < eps_v) {
            // Smooth region: quadratic
            Float ratio = y / eps_v;
            F = lam_mu * eps_v * dt * (-ratio * ratio * ratio / 3.0f + ratio * ratio - ratio + 1.0f / 3.0f);
        } else {
            // Linear region
            F = lam_mu * eps_v * dt * (y / eps_v - 1.0f / 3.0f);
        }
    }

    /**
     * @brief Derivative of friction energy with respect to velocity V
     * 
     * @param dFdV   Output: 6x1 gradient vector (dF/dV)
     * @param lam_mu lambda * mu
     * @param Tk     6x3 tangent basis matrix
     * @param eps_v  Velocity threshold
     * @param dt     Time step
     * @param vk     3x1 tangential velocity (Tk^T * V)
     */
    inline LC_GPU_CALLABLE void dFrictionEnergydV(friction::SmallVector<Float, 6>& dFdV,
                                                   Float lam_mu,
                                                   const friction::SmallMatrix<Float, 6, 3>& Tk,
                                                   Float eps_v,
                                                   Float dt,
                                                   const Vector3& vk)
    {
        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        
        Float dFdy;
        Float y = vk_norm * dt;
        
        if(y < 1e-12f) {
            // Near zero velocity, gradient is zero
            for(int i = 0; i < 6; ++i) {
                dFdV[i] = 0.0f;
            }
            return;
        }
        
        if(y < eps_v) {
            // Smooth region
            Float ratio = y / eps_v;
            dFdy = lam_mu * eps_v * dt * (-ratio * ratio + 2.0f * ratio - 1.0f) / eps_v;
        } else {
            // Linear region
            dFdy = lam_mu * dt;
        }
        
        // dFdV = Tk * (dFdy * vk / vk_norm * dt)
        Float scale = dFdy / vk_norm * dt;
        Vector3 dFdvk;
        dFdvk.x = scale * vk.x;
        dFdvk.y = scale * vk.y;
        dFdvk.z = scale * vk.z;
        
        // dFdV = Tk * dFdvk
        for(int i = 0; i < 6; ++i) {
            dFdV[i] = Tk[i][0] * dFdvk.x + Tk[i][1] * dFdvk.y + Tk[i][2] * dFdvk.z;
        }
    }

    /**
     * @brief Second derivative of friction energy with respect to velocity V
     * 
     * @param ddFddV Output: 6x6 Hessian matrix (d2F/dV2)
     * @param lam_mu lambda * mu
     * @param Tk     6x3 tangent basis matrix
     * @param eps_v  Velocity threshold
     * @param dt     Time step
     * @param vk     3x1 tangential velocity
     */
    inline LC_GPU_CALLABLE void ddFrictionEnergyddV(friction::SmallMatrix<Float, 6, 6>& ddFddV,
                                                     Float lam_mu,
                                                     const friction::SmallMatrix<Float, 6, 3>& Tk,
                                                     Float eps_v,
                                                     Float dt,
                                                     const Vector3& vk)
    {
        Float vk_norm_sq = vk.x * vk.x + vk.y * vk.y + vk.z * vk.z;
        Float vk_norm = luisa::sqrt(vk_norm_sq);
        Float y = vk_norm * dt;
        
        // Initialize to zero
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 6; ++j) {
                ddFddV[i][j] = 0.0f;
            }
        }
        
        if(y < 1e-12f) {
            // Near zero, use regularization or return zero
            // For numerical stability, use eps_v as regularization
            Float reg = lam_mu * dt * dt / eps_v;
            
            // ddFddV = reg * Tk * Tk^T
            for(int i = 0; i < 6; ++i) {
                for(int j = 0; j < 6; ++j) {
                    for(int k = 0; k < 3; ++k) {
                        ddFddV[i][j] += reg * Tk[i][k] * Tk[j][k];
                    }
                }
            }
            return;
        }
        
        Float d2Fdy2;
        if(y < eps_v) {
            // Smooth region
            Float ratio = y / eps_v;
            d2Fdy2 = lam_mu * dt * (-2.0f * ratio + 2.0f) / eps_v;
        } else {
            // Linear region - zero second derivative
            // Projected Hessian for linear region
            d2Fdy2 = 0.0f;
        }
        
        // Compute H = dt^2 * (d2F/dy2 * (vk * vk^T) / vk_norm^2 + dF/dy * (I - vk*vk^T/vk_norm^2) / vk_norm)
        // For the projected friction model
        Float dFdy;
        if(y < eps_v) {
            Float ratio = y / eps_v;
            dFdy = lam_mu * eps_v * dt * (-ratio * ratio + 2.0f * ratio - 1.0f) / eps_v;
        } else {
            dFdy = lam_mu * dt;
        }
        
        // Projected Hessian in 3D: H_3d = (dFdy / vk_norm) * (I - vk*vk^T/vk_norm^2) + d2Fdy2 * vk*vk^T/vk_norm^2
        // Then ddFddV = Tk * H_3d * Tk^T
        
        // First compute H_3d (3x3)
        friction::SmallMatrix<Float, 3, 3> H_3d{};
        
        Float scale1 = dFdy / vk_norm;
        Float scale2 = d2Fdy2 - scale1;
        
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float vi = (i == 0) ? vk.x : (i == 1) ? vk.y : vk.z;
                float vj = (j == 0) ? vk.x : (j == 1) ? vk.y : vk.z;
                
                if(i == j) {
                    H_3d[i][j] = scale1 + scale2 * vi * vj / vk_norm_sq;
                } else {
                    H_3d[i][j] = scale2 * vi * vj / vk_norm_sq;
                }
            }
        }
        
        // ddFddV = Tk * H_3d * Tk^T (6x3 * 3x3 * 3x6 -> 6x6)
        // First compute H_3d * Tk^T (3x6)
        friction::SmallMatrix<Float, 3, 6> temp{};
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 6; ++j) {
                temp[i][j] = 0;
                for(int k = 0; k < 3; ++k) {
                    temp[i][j] += H_3d[i][k] * Tk[j][k];  // Tk^T[k][j] = Tk[j][k]
                }
            }
        }
        
        // Then compute Tk * temp (6x6)
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 6; ++j) {
                ddFddV[i][j] = 0;
                for(int k = 0; k < 3; ++k) {
                    ddFddV[i][j] += Tk[i][k] * temp[k][j];
                }
            }
        }
    }

}  // namespace sym::ipc_contact
}  // namespace uipc::backend::luisa
