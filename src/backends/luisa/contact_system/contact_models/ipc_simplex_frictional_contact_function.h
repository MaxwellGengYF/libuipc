#pragma once
#include <type_define.h>
#include <utils/distance.h>
#include <utils/friction_utils.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::luisa
{
namespace sym::ipc_contact
{

    // Include the inline friction functions (to be implemented in sym/ipc_friction.inl)
    #include "sym/ipc_friction.inl"

    /**
     * @brief Compute Point-Triangle friction energy
     * 
     * @param kappa          Contact stiffness
     * @param squared_d_hat  Squared contact distance threshold
     * @param mu             Friction coefficient
     * @param dt             Time step
     * @param P              Current point position
     * @param T0, T1, T2     Current triangle vertex positions
     * @param prev_P         Previous point position
     * @param prev_T0, prev_T1, prev_T2  Previous triangle vertex positions
     * @param eps_v          Velocity threshold for friction
     * @return Friction energy
     */
    inline LC_GPU_CALLABLE Float PT_friction_energy(Float          kappa,
                                                    Float          squared_d_hat,
                                                    Float          mu,
                                                    Float          dt,
                                                    const Vector3& P,
                                                    const Vector3& T0,
                                                    const Vector3& T1,
                                                    const Vector3& T2,
                                                    const Vector3& prev_P,
                                                    const Vector3& prev_T0,
                                                    const Vector3& prev_T1,
                                                    const Vector3& prev_T2,
                                                    Float          eps_v)
    {
        using namespace distance;
        using namespace sym::ipc_contact;
        using namespace friction;

        Float D;
        point_triangle_distance2(prev_P, prev_T0, prev_T1, prev_T2, D);
        // Note: LC_ASSERT or assert should be used instead of MUDA_ASSERT
        // For luisa-compute, we typically skip assertions in release builds
        Vector12 GradD;
        point_triangle_distance2_gradient(prev_P, prev_T0, prev_T1, prev_T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        
        // Compute lambda (normal force magnitude)
        Float grad_norm = luisa::sqrt(GradD[0] * GradD[0] + GradD[1] * GradD[1] + GradD[2] * GradD[2]);
        Float lam = -dBdD * grad_norm;
        
        // Compute normal
        Vector3 n = luisa::cross(prev_T0 - prev_T1, prev_T0 - prev_T2);
        Float n_norm = luisa::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        Vector3 normal = n / n_norm;

        // Build tangent basis matrix Tk (6x3)
        SmallMatrix<Float, 6, 3> Tk;
        Matrix3x3 I = luisa::identity<3, float>();
        
        // I - normal * normal.transpose()
        Matrix3x3 I_nnt;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                I_nnt[i][j] = I[i][j] - ni * nj;
            }
        }
        
        // normal * normal.transpose() - I
        Matrix3x3 nnt_I;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                nnt_I[i][j] = ni * nj - I[i][j];
            }
        }
        
        // Fill Tk
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                Tk[i][j] = I_nnt[i][j];
                Tk[i + 3][j] = nnt_I[i][j];
            }
        }

        Vector3 v1 = (P - prev_P) / dt;

        // Compute barycentric coordinates
        Vector3 base_col0 = prev_T1 - prev_T0;
        Vector3 base_col1 = prev_T2 - prev_T0;
        
        // Lhs = base.transpose() * base (2x2)
        Float2x2 Lhs;
        Lhs[0][0] = luisa::dot(base_col0, base_col0);
        Lhs[0][1] = Lhs[1][0] = luisa::dot(base_col0, base_col1);
        Lhs[1][1] = luisa::dot(base_col1, base_col1);
        
        // rhs = base.transpose() * (prev_P - prev_T0)
        Vector3 diff = prev_P - prev_T0;
        Vector2 rhs;
        rhs.x = luisa::dot(base_col0, diff);
        rhs.y = luisa::dot(base_col1, diff);
        
        // Solve for t
        Float2x2 Lhs_inv = luisa::inverse(Lhs);
        Vector2 t = Lhs_inv * rhs;
        Float t1 = t.x;
        Float t2 = t.y;
        Float t0 = 1.0f - t1 - t2;

        // Compute velocity vector V (6x1)
        SmallVector<Float, 6> V;
        V[0] = v1.x; V[1] = v1.y; V[2] = v1.z;
        Vector3 edge_v = t0 * (T0 - prev_T0) / dt + t1 * (T1 - prev_T1) / dt
                       + t2 * (T2 - prev_T2) / dt;
        V[3] = edge_v.x; V[4] = edge_v.y; V[5] = edge_v.z;

        // vk = Tk.transpose() * V (3x1)
        Vector3 vk;
        vk.x = Tk[0][0] * V[0] + Tk[1][0] * V[1] + Tk[2][0] * V[2]
             + Tk[3][0] * V[3] + Tk[4][0] * V[4] + Tk[5][0] * V[5];
        vk.y = Tk[0][1] * V[0] + Tk[1][1] * V[1] + Tk[2][1] * V[2]
             + Tk[3][1] * V[3] + Tk[4][1] * V[4] + Tk[5][1] * V[5];
        vk.z = Tk[0][2] * V[0] + Tk[1][2] * V[1] + Tk[2][2] * V[2]
             + Tk[3][2] * V[3] + Tk[4][2] * V[4] + Tk[5][2] * V[5];

        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        Float y = vk_norm * dt;
        
        Float F;
        FrictionEnergy(F, lam * mu, eps_v, dt, y);
        return F;
    }

    /**
     * @brief Compute Point-Triangle friction gradient and Hessian
     */
    inline LC_GPU_CALLABLE void PT_friction_gradient_hessian(Vector12&      G,
                                                             Matrix12x12&   H,
                                                             Float          kappa,
                                                             Float          squared_d_hat,
                                                             Float          mu,
                                                             Float          dt,
                                                             const Vector3& P,
                                                             const Vector3& T0,
                                                             const Vector3& T1,
                                                             const Vector3& T2,
                                                             const Vector3& prev_P,
                                                             const Vector3& prev_T0,
                                                             const Vector3& prev_T1,
                                                             const Vector3& prev_T2,
                                                             Float          eps_v)
    {
        using namespace distance;
        using namespace sym::ipc_contact;
        using namespace friction;

        Float D;
        point_triangle_distance2(prev_P, prev_T0, prev_T1, prev_T2, D);
        Vector12 GradD;
        point_triangle_distance2_gradient(prev_P, prev_T0, prev_T1, prev_T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);

        Float grad_norm = luisa::sqrt(GradD[0] * GradD[0] + GradD[1] * GradD[1] + GradD[2] * GradD[2]);
        Float lam = -dBdD * grad_norm;

        Vector3 n = luisa::cross(prev_T0 - prev_T1, prev_T0 - prev_T2);
        Float n_norm = luisa::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        Vector3 normal = n / n_norm;

        // Build Tk
        SmallMatrix<Float, 6, 3> Tk;
        Matrix3x3 I = luisa::identity<3, float>();
        
        Matrix3x3 I_nnt, nnt_I;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                I_nnt[i][j] = I[i][j] - ni * nj;
                nnt_I[i][j] = ni * nj - I[i][j];
            }
        }
        
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                Tk[i][j] = I_nnt[i][j];
                Tk[i + 3][j] = nnt_I[i][j];
            }
        }

        Vector3 v1 = (P - prev_P) / dt;

        // Compute barycentric coordinates
        Vector3 base_col0 = prev_T1 - prev_T0;
        Vector3 base_col1 = prev_T2 - prev_T0;
        
        Float2x2 Lhs;
        Lhs[0][0] = luisa::dot(base_col0, base_col0);
        Lhs[0][1] = Lhs[1][0] = luisa::dot(base_col0, base_col1);
        Lhs[1][1] = luisa::dot(base_col1, base_col1);
        
        Vector3 diff = prev_P - prev_T0;
        Vector2 rhs;
        rhs.x = luisa::dot(base_col0, diff);
        rhs.y = luisa::dot(base_col1, diff);
        
        Float2x2 Lhs_inv = luisa::inverse(Lhs);
        Vector2 t = Lhs_inv * rhs;
        Float t1 = t.x;
        Float t2 = t.y;
        Float t0 = 1.0f - t1 - t2;

        // Compute velocity vector V
        SmallVector<Float, 6> V;
        V[0] = v1.x; V[1] = v1.y; V[2] = v1.z;
        Vector3 edge_v = t0 * (T0 - prev_T0) / dt + t1 * (T1 - prev_T1) / dt
                       + t2 * (T2 - prev_T2) / dt;
        V[3] = edge_v.x; V[4] = edge_v.y; V[5] = edge_v.z;

        // vk = Tk.transpose() * V
        Vector3 vk;
        for(int i = 0; i < 3; ++i) {
            vk[i] = 0;
            for(int j = 0; j < 6; ++j) {
                vk[i] += Tk[j][i] * V[j];
            }
        }

        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        Float y = vk_norm * dt;
        
        // dFdV (6x1)
        SmallVector<Float, 6> dFdV;
        dFrictionEnergydV(dFdV, lam * mu, Tk, eps_v, dt, vk);

        // GradV (6x12)
        SmallMatrix<Float, 6, 12> GradV{};
        // GradV.block<3, 3>(0, 0) = I / dt
        GradV[0][0] = 1.0f / dt; GradV[0][1] = 0; GradV[0][2] = 0;
        GradV[1][0] = 0; GradV[1][1] = 1.0f / dt; GradV[1][2] = 0;
        GradV[2][0] = 0; GradV[2][1] = 0; GradV[2][2] = 1.0f / dt;
        
        // GradV.block<3, 3>(3, 3) = I * t0 / dt
        GradV[3][3] = t0 / dt; GradV[3][4] = 0; GradV[3][5] = 0;
        GradV[4][3] = 0; GradV[4][4] = t0 / dt; GradV[4][5] = 0;
        GradV[5][3] = 0; GradV[5][4] = 0; GradV[5][5] = t0 / dt;
        
        // GradV.block<3, 3>(3, 6) = I * t1 / dt
        GradV[3][6] = t1 / dt; GradV[3][7] = 0; GradV[3][8] = 0;
        GradV[4][6] = 0; GradV[4][7] = t1 / dt; GradV[4][8] = 0;
        GradV[5][6] = 0; GradV[5][7] = 0; GradV[5][8] = t1 / dt;
        
        // GradV.block<3, 3>(3, 9) = I * t2 / dt
        GradV[3][9] = t2 / dt; GradV[3][10] = 0; GradV[3][11] = 0;
        GradV[4][9] = 0; GradV[4][10] = t2 / dt; GradV[4][11] = 0;
        GradV[5][9] = 0; GradV[5][10] = 0; GradV[5][11] = t2 / dt;

        // G = GradV.transpose() * dFdV (12x1)
        for(int i = 0; i < 12; ++i) {
            G[i] = 0;
            for(int j = 0; j < 6; ++j) {
                G[i] += GradV[j][i] * dFdV[j];
            }
        }

        // ddFddV (6x6)
        SmallMatrix<Float, 6, 6> ddFddV;
        ddFrictionEnergyddV(ddFddV, lam * mu, Tk, eps_v, dt, vk);

        // H = GradV.transpose() * ddFddV * GradV (12x12)
        // First compute ddFddV * GradV (6x12)
        SmallMatrix<Float, 6, 12> temp{};
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 12; ++j) {
                temp[i][j] = 0;
                for(int k = 0; k < 6; ++k) {
                    temp[i][j] += ddFddV[i][k] * GradV[k][j];
                }
            }
        }
        
        // Then compute GradV.transpose() * temp
        for(int i = 0; i < 12; ++i) {
            for(int j = 0; j < 12; ++j) {
                H[i][j] = 0;
                for(int k = 0; k < 6; ++k) {
                    H[i][j] += GradV[k][i] * temp[k][j];
                }
            }
        }
    }

    /**
     * @brief Compute Edge-Edge friction energy
     */
    inline LC_GPU_CALLABLE Float EE_friction_energy(Float          kappa,
                                                    Float          squared_d_hat,
                                                    Float          mu,
                                                    Float          dt,
                                                    const Vector3& P0,
                                                    const Vector3& P1,
                                                    const Vector3& Q0,
                                                    const Vector3& Q1,
                                                    const Vector3& prev_P0,
                                                    const Vector3& prev_P1,
                                                    const Vector3& prev_Q0,
                                                    const Vector3& prev_Q1,
                                                    Float          eps_v)
    {
        using namespace distance;
        using namespace sym::ipc_contact;
        using namespace friction;

        Float D;
        edge_edge_distance2(prev_P0, prev_P1, prev_Q0, prev_Q1, D);
        Vector12 GradD;
        edge_edge_distance2_gradient(prev_P0, prev_P1, prev_Q0, prev_Q1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);

        Vector3 n = luisa::cross(prev_P0 - prev_P1, prev_Q0 - prev_Q1);
        Float n_norm = luisa::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        Vector3 normal = n / n_norm;

        // Build Tk
        SmallMatrix<Float, 6, 3> Tk;
        Matrix3x3 I = luisa::identity<3, float>();
        
        Matrix3x3 I_nnt, nnt_I;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                I_nnt[i][j] = I[i][j] - ni * nj;
                nnt_I[i][j] = ni * nj - I[i][j];
            }
        }
        
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                Tk[i][j] = I_nnt[i][j];
                Tk[i + 3][j] = nnt_I[i][j];
            }
        }

        // Compute edge parameters
        Vector3 base_col0 = prev_P1 - prev_P0;
        Vector3 base_col1 = prev_Q1 - prev_Q0;
        
        Float2x2 X{};
        X[0][0] = -1.0f;
        X[1][1] = 1.0f;
        
        // Lhs = base.transpose() * base * X
        Float2x2 base_T_base;
        base_T_base[0][0] = luisa::dot(base_col0, base_col0);
        base_T_base[0][1] = luisa::dot(base_col0, base_col1);
        base_T_base[1][0] = base_T_base[0][1];
        base_T_base[1][1] = luisa::dot(base_col1, base_col1);
        
        Float2x2 Lhs;
        Lhs[0][0] = base_T_base[0][0] * X[0][0] + base_T_base[0][1] * X[1][0];
        Lhs[0][1] = base_T_base[0][0] * X[0][1] + base_T_base[0][1] * X[1][1];
        Lhs[1][0] = base_T_base[1][0] * X[0][0] + base_T_base[1][1] * X[1][0];
        Lhs[1][1] = base_T_base[1][0] * X[0][1] + base_T_base[1][1] * X[1][1];
        
        Vector3 diff = prev_P0 - prev_Q0;
        Vector2 rhs;
        rhs.x = luisa::dot(base_col0, diff);
        rhs.y = luisa::dot(base_col1, diff);
        
        Float2x2 Lhs_inv = luisa::inverse(Lhs);
        Vector2 t = Lhs_inv * rhs;
        Float t0 = t.x;
        Float t1 = t.y;

        // Compute velocity V
        SmallVector<Float, 6> V;
        Vector3 v0 = (P0 - prev_P0) * (1.0f - t0) / dt + (P1 - prev_P1) * t0 / dt;
        Vector3 v1 = (Q0 - prev_Q0) * (1.0f - t1) / dt + (Q1 - prev_Q1) * t1 / dt;
        V[0] = v0.x; V[1] = v0.y; V[2] = v0.z;
        V[3] = v1.x; V[4] = v1.y; V[5] = v1.z;

        // vk = Tk.transpose() * V
        Vector3 vk;
        for(int i = 0; i < 3; ++i) {
            vk[i] = 0;
            for(int j = 0; j < 6; ++j) {
                vk[i] += Tk[j][i] * V[j];
            }
        }

        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        Float y = vk_norm * dt;
        
        // Compute lambda
        Vector3 grad_seg0;
        grad_seg0.x = GradD[0]; grad_seg0.y = GradD[1]; grad_seg0.z = GradD[2];
        Vector3 grad_seg1;
        grad_seg1.x = GradD[3]; grad_seg1.y = GradD[4]; grad_seg1.z = GradD[5];
        Vector3 weighted_grad = grad_seg0 * (1.0f - t0) + grad_seg1 * t0;
        Float grad_norm = luisa::sqrt(weighted_grad.x * weighted_grad.x 
                                    + weighted_grad.y * weighted_grad.y 
                                    + weighted_grad.z * weighted_grad.z);
        Float lam = -dBdD * grad_norm;

        Float F;
        FrictionEnergy(F, lam * mu, eps_v, dt, y);
        return F;
    }

    /**
     * @brief Compute Edge-Edge friction gradient and Hessian
     */
    inline LC_GPU_CALLABLE void EE_friction_gradient_hessian(Vector12&      G,
                                                             Matrix12x12&   H,
                                                             Float          kappa,
                                                             Float          squared_d_hat,
                                                             Float          mu,
                                                             Float          dt,
                                                             const Vector3& P0,
                                                             const Vector3& P1,
                                                             const Vector3& Q0,
                                                             const Vector3& Q1,
                                                             const Vector3& prev_P0,
                                                             const Vector3& prev_P1,
                                                             const Vector3& prev_Q0,
                                                             const Vector3& prev_Q1,
                                                             Float          eps_v)
    {
        using namespace distance;
        using namespace sym::ipc_contact;
        using namespace friction;

        Float D;
        edge_edge_distance2(prev_P0, prev_P1, prev_Q0, prev_Q1, D);
        Vector12 GradD;
        edge_edge_distance2_gradient(prev_P0, prev_P1, prev_Q0, prev_Q1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);

        Vector3 n = luisa::cross(prev_P0 - prev_P1, prev_Q0 - prev_Q1);
        Float n_norm = luisa::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        Vector3 normal = n / n_norm;

        // Build Tk
        SmallMatrix<Float, 6, 3> Tk;
        Matrix3x3 I = luisa::identity<3, float>();
        
        Matrix3x3 I_nnt, nnt_I;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                I_nnt[i][j] = I[i][j] - ni * nj;
                nnt_I[i][j] = ni * nj - I[i][j];
            }
        }
        
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                Tk[i][j] = I_nnt[i][j];
                Tk[i + 3][j] = nnt_I[i][j];
            }
        }

        // Compute edge parameters
        Vector3 base_col0 = prev_P1 - prev_P0;
        Vector3 base_col1 = prev_Q1 - prev_Q0;
        
        Float2x2 X{};
        X[0][0] = -1.0f;
        X[1][1] = 1.0f;
        
        Float2x2 base_T_base;
        base_T_base[0][0] = luisa::dot(base_col0, base_col0);
        base_T_base[0][1] = luisa::dot(base_col0, base_col1);
        base_T_base[1][0] = base_T_base[0][1];
        base_T_base[1][1] = luisa::dot(base_col1, base_col1);
        
        Float2x2 Lhs;
        Lhs[0][0] = base_T_base[0][0] * X[0][0] + base_T_base[0][1] * X[1][0];
        Lhs[0][1] = base_T_base[0][0] * X[0][1] + base_T_base[0][1] * X[1][1];
        Lhs[1][0] = base_T_base[1][0] * X[0][0] + base_T_base[1][1] * X[1][0];
        Lhs[1][1] = base_T_base[1][0] * X[0][1] + base_T_base[1][1] * X[1][1];
        
        Vector3 diff = prev_P0 - prev_Q0;
        Vector2 rhs;
        rhs.x = luisa::dot(base_col0, diff);
        rhs.y = luisa::dot(base_col1, diff);
        
        Float2x2 Lhs_inv = luisa::inverse(Lhs);
        Vector2 t = Lhs_inv * rhs;
        Float t0 = t.x;
        Float t1 = t.y;

        // Compute velocity V
        SmallVector<Float, 6> V;
        Vector3 v0 = (P0 - prev_P0) * (1.0f - t0) / dt + (P1 - prev_P1) * t0 / dt;
        Vector3 v1 = (Q0 - prev_Q0) * (1.0f - t1) / dt + (Q1 - prev_Q1) * t1 / dt;
        V[0] = v0.x; V[1] = v0.y; V[2] = v0.z;
        V[3] = v1.x; V[4] = v1.y; V[5] = v1.z;

        // vk = Tk.transpose() * V
        Vector3 vk;
        for(int i = 0; i < 3; ++i) {
            vk[i] = 0;
            for(int j = 0; j < 6; ++j) {
                vk[i] += Tk[j][i] * V[j];
            }
        }

        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        Float y = vk_norm * dt;
        
        // Compute lambda
        Vector3 grad_seg0, grad_seg1;
        grad_seg0.x = GradD[0]; grad_seg0.y = GradD[1]; grad_seg0.z = GradD[2];
        grad_seg1.x = GradD[3]; grad_seg1.y = GradD[4]; grad_seg1.z = GradD[5];
        Vector3 weighted_grad = grad_seg0 * (1.0f - t0) + grad_seg1 * t0;
        Float grad_norm = luisa::sqrt(weighted_grad.x * weighted_grad.x 
                                    + weighted_grad.y * weighted_grad.y 
                                    + weighted_grad.z * weighted_grad.z);
        Float lam = -dBdD * grad_norm;

        // dFdV
        SmallVector<Float, 6> dFdV;
        dFrictionEnergydV(dFdV, lam * mu, Tk, eps_v, dt, vk);

        // GradV (6x12)
        SmallMatrix<Float, 6, 12> GradV{};
        // GradV.block<3, 3>(0, 0) = I * (1 - t0) / dt
        GradV[0][0] = (1.0f - t0) / dt; GradV[0][1] = 0; GradV[0][2] = 0;
        GradV[1][0] = 0; GradV[1][1] = (1.0f - t0) / dt; GradV[1][2] = 0;
        GradV[2][0] = 0; GradV[2][1] = 0; GradV[2][2] = (1.0f - t0) / dt;
        
        // GradV.block<3, 3>(0, 3) = I * t0 / dt
        GradV[0][3] = t0 / dt; GradV[0][4] = 0; GradV[0][5] = 0;
        GradV[1][3] = 0; GradV[1][4] = t0 / dt; GradV[1][5] = 0;
        GradV[2][3] = 0; GradV[2][4] = 0; GradV[2][5] = t0 / dt;
        
        // GradV.block<3, 3>(3, 6) = I * (1 - t1) / dt
        GradV[3][6] = (1.0f - t1) / dt; GradV[3][7] = 0; GradV[3][8] = 0;
        GradV[4][6] = 0; GradV[4][7] = (1.0f - t1) / dt; GradV[4][8] = 0;
        GradV[5][6] = 0; GradV[5][7] = 0; GradV[5][8] = (1.0f - t1) / dt;
        
        // GradV.block<3, 3>(3, 9) = I * t1 / dt
        GradV[3][9] = t1 / dt; GradV[3][10] = 0; GradV[3][11] = 0;
        GradV[4][9] = 0; GradV[4][10] = t1 / dt; GradV[4][11] = 0;
        GradV[5][9] = 0; GradV[5][10] = 0; GradV[5][11] = t1 / dt;

        // G = GradV.transpose() * dFdV
        for(int i = 0; i < 12; ++i) {
            G[i] = 0;
            for(int j = 0; j < 6; ++j) {
                G[i] += GradV[j][i] * dFdV[j];
            }
        }

        // ddFddV
        SmallMatrix<Float, 6, 6> ddFddV;
        ddFrictionEnergyddV(ddFddV, lam * mu, Tk, eps_v, dt, vk);

        // H = GradV.transpose() * ddFddV * GradV
        SmallMatrix<Float, 6, 12> temp{};
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 12; ++j) {
                temp[i][j] = 0;
                for(int k = 0; k < 6; ++k) {
                    temp[i][j] += ddFddV[i][k] * GradV[k][j];
                }
            }
        }
        
        for(int i = 0; i < 12; ++i) {
            for(int j = 0; j < 12; ++j) {
                H[i][j] = 0;
                for(int k = 0; k < 6; ++k) {
                    H[i][j] += GradV[k][i] * temp[k][j];
                }
            }
        }
    }

    /**
     * @brief Compute Point-Edge friction energy
     */
    inline LC_GPU_CALLABLE Float PE_friction_energy(Float          kappa,
                                                    Float          squared_d_hat,
                                                    Float          mu,
                                                    Float          dt,
                                                    const Vector3& P,
                                                    const Vector3& E0,
                                                    const Vector3& E1,
                                                    const Vector3& prev_P,
                                                    const Vector3& prev_E0,
                                                    const Vector3& prev_E1,
                                                    Float          eps_v)
    {
        using namespace distance;
        using namespace sym::ipc_contact;
        using namespace friction;

        Float D;
        point_edge_distance2(prev_P, prev_E0, prev_E1, D);
        Vector9 GradD;
        point_edge_distance2_gradient(prev_P, prev_E0, prev_E1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        
        Float grad_norm = luisa::sqrt(GradD[0] * GradD[0] + GradD[1] * GradD[1] + GradD[2] * GradD[2]);
        Float lam = -dBdD * grad_norm;

        // Compute barycentric coordinate t0
        Vector3 edge = prev_E0 - prev_E1;
        Float edge_len_sq = luisa::dot(edge, edge);
        Float t0 = luisa::dot(prev_P - prev_E1, edge) / edge_len_sq;
        Float t1 = 1.0f - t0;

        Vector3 prev_P0 = t0 * prev_E0 + t1 * prev_E1;
        Vector3 n = prev_P0 - prev_P;
        Float n_norm = luisa::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        Vector3 normal = n / n_norm;

        // Build Tk
        SmallMatrix<Float, 6, 3> Tk;
        Matrix3x3 I = luisa::identity<3, float>();
        
        Matrix3x3 I_nnt, nnt_I;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                I_nnt[i][j] = I[i][j] - ni * nj;
                nnt_I[i][j] = ni * nj - I[i][j];
            }
        }
        
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                Tk[i][j] = I_nnt[i][j];
                Tk[i + 3][j] = nnt_I[i][j];
            }
        }

        Vector3 v1 = (P - prev_P) / dt;

        // Compute velocity V
        SmallVector<Float, 6> V;
        V[0] = v1.x; V[1] = v1.y; V[2] = v1.z;
        Vector3 edge_v = t0 * (E0 - prev_E0) / dt + t1 * (E1 - prev_E1) / dt;
        V[3] = edge_v.x; V[4] = edge_v.y; V[5] = edge_v.z;

        // vk = Tk.transpose() * V
        Vector3 vk;
        for(int i = 0; i < 3; ++i) {
            vk[i] = 0;
            for(int j = 0; j < 6; ++j) {
                vk[i] += Tk[j][i] * V[j];
            }
        }

        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        Float y = vk_norm * dt;

        Float F;
        FrictionEnergy(F, lam * mu, eps_v, dt, y);
        return F;
    }

    /**
     * @brief Compute Point-Edge friction gradient and Hessian
     */
    inline LC_GPU_CALLABLE void PE_friction_gradient_hessian(Vector9&       G,
                                                             Matrix9x9&     H,
                                                             Float          kappa,
                                                             Float          squared_d_hat,
                                                             Float          mu,
                                                             Float          dt,
                                                             const Vector3& P,
                                                             const Vector3& E0,
                                                             const Vector3& E1,
                                                             const Vector3& prev_P,
                                                             const Vector3& prev_E0,
                                                             const Vector3& prev_E1,
                                                             Float          eps_v)
    {
        using namespace distance;
        using namespace sym::ipc_contact;
        using namespace friction;

        Float D;
        point_edge_distance2(prev_P, prev_E0, prev_E1, D);
        Vector9 GradD{};
        point_edge_distance2_gradient(prev_P, prev_E0, prev_E1, GradD);

        Float dBdD = 0;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);

        Float grad_norm = luisa::sqrt(GradD[0] * GradD[0] + GradD[1] * GradD[1] + GradD[2] * GradD[2]);
        Float lam = -dBdD * grad_norm;

        // Compute barycentric coordinate t0
        Vector3 edge = prev_E0 - prev_E1;
        Float edge_len_sq = luisa::dot(edge, edge);
        Float t0 = luisa::dot(prev_P - prev_E1, edge) / edge_len_sq;
        Float t1 = 1.0f - t0;

        Vector3 prev_P0 = t0 * prev_E0 + t1 * prev_E1;
        Vector3 n = prev_P0 - prev_P;
        Float n_norm = luisa::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        Vector3 normal = n / n_norm;

        // Build Tk
        SmallMatrix<Float, 6, 3> Tk;
        Matrix3x3 I = luisa::identity<3, float>();
        
        Matrix3x3 I_nnt, nnt_I;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                I_nnt[i][j] = I[i][j] - ni * nj;
                nnt_I[i][j] = ni * nj - I[i][j];
            }
        }
        
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                Tk[i][j] = I_nnt[i][j];
                Tk[i + 3][j] = nnt_I[i][j];
            }
        }

        Vector3 v1 = (P - prev_P) / dt;

        // Compute velocity V
        SmallVector<Float, 6> V;
        V[0] = v1.x; V[1] = v1.y; V[2] = v1.z;
        Vector3 edge_v = t0 * (E0 - prev_E0) / dt + t1 * (E1 - prev_E1) / dt;
        V[3] = edge_v.x; V[4] = edge_v.y; V[5] = edge_v.z;

        // vk = Tk.transpose() * V
        Vector3 vk;
        for(int i = 0; i < 3; ++i) {
            vk[i] = 0;
            for(int j = 0; j < 6; ++j) {
                vk[i] += Tk[j][i] * V[j];
            }
        }

        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        Float y = vk_norm * dt;

        // dFdV
        SmallVector<Float, 6> dFdV;
        dFrictionEnergydV(dFdV, lam * mu, Tk, eps_v, dt, vk);

        // GradV (6x9)
        SmallMatrix<Float, 6, 9> GradV{};
        // GradV.block<3, 3>(0, 0) = I / dt
        GradV[0][0] = 1.0f / dt; GradV[0][1] = 0; GradV[0][2] = 0;
        GradV[1][0] = 0; GradV[1][1] = 1.0f / dt; GradV[1][2] = 0;
        GradV[2][0] = 0; GradV[2][1] = 0; GradV[2][2] = 1.0f / dt;
        
        // GradV.block<3, 3>(3, 3) = I * t0 / dt
        GradV[3][3] = t0 / dt; GradV[3][4] = 0; GradV[3][5] = 0;
        GradV[4][3] = 0; GradV[4][4] = t0 / dt; GradV[4][5] = 0;
        GradV[5][3] = 0; GradV[5][4] = 0; GradV[5][5] = t0 / dt;
        
        // GradV.block<3, 3>(3, 6) = I * t1 / dt
        GradV[3][6] = t1 / dt; GradV[3][7] = 0; GradV[3][8] = 0;
        GradV[4][6] = 0; GradV[4][7] = t1 / dt; GradV[4][8] = 0;
        GradV[5][6] = 0; GradV[5][7] = 0; GradV[5][8] = t1 / dt;

        // G = GradV.transpose() * dFdV (9x1)
        for(int i = 0; i < 9; ++i) {
            G[i] = 0;
            for(int j = 0; j < 6; ++j) {
                G[i] += GradV[j][i] * dFdV[j];
            }
        }

        // ddFddV
        SmallMatrix<Float, 6, 6> ddFddV;
        ddFrictionEnergyddV(ddFddV, lam * mu, Tk, eps_v, dt, vk);

        // H = GradV.transpose() * ddFddV * GradV
        SmallMatrix<Float, 6, 9> temp{};
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 9; ++j) {
                temp[i][j] = 0;
                for(int k = 0; k < 6; ++k) {
                    temp[i][j] += ddFddV[i][k] * GradV[k][j];
                }
            }
        }
        
        for(int i = 0; i < 9; ++i) {
            for(int j = 0; j < 9; ++j) {
                H[i][j] = 0;
                for(int k = 0; k < 6; ++k) {
                    H[i][j] += GradV[k][i] * temp[k][j];
                }
            }
        }
    }

    /**
     * @brief Compute Point-Point friction energy
     */
    inline LC_GPU_CALLABLE Float PP_friction_energy(Float          kappa,
                                                    Float          squared_d_hat,
                                                    Float          mu,
                                                    Float          dt,
                                                    const Vector3& P,
                                                    const Vector3& Q,
                                                    const Vector3& prev_P,
                                                    const Vector3& prev_Q,
                                                    Float          eps_v)
    {
        using namespace distance;
        using namespace sym::ipc_contact;
        using namespace friction;

        Float D;
        point_point_distance2(prev_P, prev_Q, D);
        Vector6 GradD;
        point_point_distance2_gradient(prev_P, prev_Q, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        
        Float grad_norm = luisa::sqrt(GradD[0] * GradD[0] + GradD[1] * GradD[1] + GradD[2] * GradD[2]);
        Float lam = -dBdD * grad_norm;

        Vector3 n = prev_Q - prev_P;
        Float n_norm = luisa::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        Vector3 normal = n / n_norm;

        // Build Tk
        SmallMatrix<Float, 6, 3> Tk;
        Matrix3x3 I = luisa::identity<3, float>();
        
        Matrix3x3 I_nnt, nnt_I;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                I_nnt[i][j] = I[i][j] - ni * nj;
                nnt_I[i][j] = ni * nj - I[i][j];
            }
        }
        
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                Tk[i][j] = I_nnt[i][j];
                Tk[i + 3][j] = nnt_I[i][j];
            }
        }

        // Compute velocity V
        SmallVector<Float, 6> V;
        Vector3 v0 = (P - prev_P) / dt;
        Vector3 v1 = (Q - prev_Q) / dt;
        V[0] = v0.x; V[1] = v0.y; V[2] = v0.z;
        V[3] = v1.x; V[4] = v1.y; V[5] = v1.z;

        // vk = Tk.transpose() * V
        Vector3 vk;
        for(int i = 0; i < 3; ++i) {
            vk[i] = 0;
            for(int j = 0; j < 6; ++j) {
                vk[i] += Tk[j][i] * V[j];
            }
        }

        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        Float y = vk_norm * dt;

        Float F;
        FrictionEnergy(F, lam * mu, eps_v, dt, y);
        return F;
    }

    /**
     * @brief Compute Point-Point friction gradient and Hessian
     */
    inline LC_GPU_CALLABLE void PP_friction_gradient_hessian(Vector6&       G,
                                                             Matrix6x6&     H,
                                                             Float          kappa,
                                                             Float          squared_d_hat,
                                                             Float          mu,
                                                             Float          dt,
                                                             const Vector3& P,
                                                             const Vector3& Q,
                                                             const Vector3& prev_P,
                                                             const Vector3& prev_Q,
                                                             Float          eps_v)
    {
        using namespace distance;
        using namespace sym::ipc_contact;
        using namespace friction;

        Float D;
        point_point_distance2(prev_P, prev_Q, D);
        Vector6 GradD;
        point_point_distance2_gradient(prev_P, prev_Q, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        
        Float grad_norm = luisa::sqrt(GradD[0] * GradD[0] + GradD[1] * GradD[1] + GradD[2] * GradD[2]);
        Float lam = -dBdD * grad_norm;

        Vector3 n = prev_Q - prev_P;
        Float n_norm = luisa::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        Vector3 normal = n / n_norm;

        // Build Tk
        SmallMatrix<Float, 6, 3> Tk;
        Matrix3x3 I = luisa::identity<3, float>();
        
        Matrix3x3 I_nnt, nnt_I;
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                float ni = (i == 0) ? normal.x : (i == 1) ? normal.y : normal.z;
                float nj = (j == 0) ? normal.x : (j == 1) ? normal.y : normal.z;
                I_nnt[i][j] = I[i][j] - ni * nj;
                nnt_I[i][j] = ni * nj - I[i][j];
            }
        }
        
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                Tk[i][j] = I_nnt[i][j];
                Tk[i + 3][j] = nnt_I[i][j];
            }
        }

        // Compute velocity V
        SmallVector<Float, 6> V;
        Vector3 v0 = (P - prev_P) / dt;
        Vector3 v1 = (Q - prev_Q) / dt;
        V[0] = v0.x; V[1] = v0.y; V[2] = v0.z;
        V[3] = v1.x; V[4] = v1.y; V[5] = v1.z;

        // vk = Tk.transpose() * V
        Vector3 vk;
        for(int i = 0; i < 3; ++i) {
            vk[i] = 0;
            for(int j = 0; j < 6; ++j) {
                vk[i] += Tk[j][i] * V[j];
            }
        }

        Float vk_norm = luisa::sqrt(vk.x * vk.x + vk.y * vk.y + vk.z * vk.z);
        Float y = vk_norm * dt;

        // dFdV
        SmallVector<Float, 6> dFdV;
        dFrictionEnergydV(dFdV, lam * mu, Tk, eps_v, dt, vk);
        
        // GradV = Identity for PP case, so G = dFdV
        for(int i = 0; i < 6; ++i) {
            G[i] = dFdV[i];
        }

        // ddFddV
        SmallMatrix<Float, 6, 6> ddFddV;
        ddFrictionEnergyddV(ddFddV, lam * mu, Tk, eps_v, dt, vk);
        
        // H = ddFddV for PP case
        for(int i = 0; i < 6; ++i) {
            for(int j = 0; j < 6; ++j) {
                H[i][j] = ddFddV[i][j];
            }
        }
    }

}  // namespace sym::ipc_contact
}  // namespace uipc::backend::luisa
