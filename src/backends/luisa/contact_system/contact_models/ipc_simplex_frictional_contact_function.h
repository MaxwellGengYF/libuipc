#pragma once
/********************************************************************
 * @file contact_models/ipc_simplex_frictional_contact_function.h
 * @brief T8.1b — **DEAD CODE, TRANSCRIBED VERBATIM: the reference's own legacy
 * IPC-simplex friction formulation** (port of
 * src/backends/cuda/contact_system/contact_models/ipc_simplex_frictional_contact_function.h,
 * 487 reference lines).
 *
 * READ THIS BEFORE TOUCHING THE FILE
 * ----------------------------------
 *  1. **No translation unit includes it, here or in the reference.**  The live IPC simplex
 *     friction is the *codim* formulation (`codim_ipc_simplex_frictional_contact_function.h`,
 *     namespace `sym::codim_ipc_contact`), which is what
 *     `contact_models/ipc_simplex_frictional_contact.cpp` includes; `grep` over the whole
 *     reference tree finds this file only through its own body.
 *  2. **It cannot compile — in the reference either.**  Its two dependencies
 *     `#include <contact_system/contact_models/ipc_contact_function.h>` and
 *     `#include "sym/ipc_friction.inl"` **do not exist anywhere in the reference tree**
 *     (measured: `grep -rn "ipc_contact_function.h\|ipc_friction.inl" src/` finds only this
 *     file's own two include lines; `src/backends/cuda/contact_system/contact_models/sym/`
 *     holds exactly `codim_ipc_contact.inl` and `vertex_half_plane_distance.inl`), so the
 *     symbols the bodies call — `FrictionEnergy`, `dFrictionEnergydV`,
 *     `ddFrictionEnergyddV` and the **4-argument** `dKappaBarrierdD(R, kappa, D, dHat)`
 *     (the live codim one takes a fifth `xi` argument) — have no declaration.  It is a
 *     leftover of the pre-codim (2023 IPC) friction design, kept in the tree for reference
 *     and never compiled.
 *  3. **Therefore the port transcribes it as data, not as a compilable header**: the two
 *     impossible includes are *commented out* right here (with the reason), and no TU ever
 *     includes this file, so no compiler ever sees the undeclared names.  That is the
 *     cookbook §7.36 "transcribe the quirk, do not fix it" rule applied to a dead file — the
 *     alternative (dropping the file) would silently lose a 487-line reference artefact that
 *     later batches (T8.1d's AL subsystem, which has its own `al_simplex_*` family) may want
 *     to consult.  The [contact_model] test pins the *live* sibling and the cookbook rule
 *     pins this one's status.
 *
 * TRANSCRIPTION RULES (identical to the live headers): `Float` -> `T` (the routine is
 * `template <typename T>` so the test can evaluate it on the host, although nothing
 * instantiates it), `Vector3/6/9/12` -> `lps::LVec<T,3/{6,9,12}>`,
 * `Matrix6x6/9x9/12x12` -> `lps::LMat<T,6,6>/<T,9,9>/<T,12,12>`,
 * `Eigen::Matrix<Float,R,C>` -> `lps::LMat<T,R,C>`, `Eigen::Vector<Float,N>` ->
 * `lps::LVec<T,N>`, `a.cross(b)` -> `lps::cross(a, b)`, `std::pow/log/sqrt` ->
 * `lps::pow/log/sqrt`, `.block<>`/`.segment<>`/`.head<>` keep Eigen's spelling (the port's
 * strided views; `template` is needed in a dependent context).  `UIPC_KERNEL_ASSERT` is kept
 * *as written*: it is one more symbol the reference file cannot resolve on its own.
 *
 * WHAT THE DEAD FILE SAYS (for the record): the older formulation builds, per pair, the
 * projection `Tk = [I - n n^T, n n^T - I]` onto the contact plane, the relative velocity
 * `V` (current minus previous, over dt), its tangential part `vk = Tk^T V` and the
 * *scalar* `lam = -dB/dD * |GradD.head(3)|`, then calls the (missing)
 * `FrictionEnergy(dFdV, lam * mu, eps_v, dt, y)` family with `y = |vk| dt`; its gradient
 * and Hessian are `G = GradV^T dFdV`, `H = GradV^T ddFddV GradV` with the hand-written
 * `GradV` blocks.  The PP variant's chain rule degenerates to `G = dFdV`,
 * `H = ddFddV` (the dead file keeps the commented-out `GradV = Identity()` line).
 *********************************************************************/
#include <type_define.h>
#include <utils/distance.h>
// The reference's two dependencies are DEAD — these files do not exist in the reference
// tree either, so the whole header is un-compilable there (see the file header, item 2):
//   #include <contact_system/contact_models/ipc_contact_function.h>
// and, further down inside the namespace, `#include "sym/ipc_friction.inl"`.

namespace uipc::backend::luisa
{
namespace sym::ipc_contact
{
    // reference: #include "sym/ipc_friction.inl"   <-- does not exist (see the header note)

    namespace ldist = luisa_device::math::distance;
    using lps::LMat;
    using lps::LVec;

    template <typename T>
    [[nodiscard]] inline T PT_friction_energy(const T&          kappa,
                                              const T&          squared_d_hat,
                                              const T&          mu,
                                              const T&          dt,
                                              const LVec<T, 3>& P,
                                              const LVec<T, 3>& T0,
                                              const LVec<T, 3>& T1,
                                              const LVec<T, 3>& T2,
                                              const LVec<T, 3>& prev_P,
                                              const LVec<T, 3>& prev_T0,
                                              const LVec<T, 3>& prev_T1,
                                              const LVec<T, 3>& prev_T2,
                                              const T&          eps_v)
    {
        T D;
        ldist::point_triangle_distance2(prev_P, prev_T0, prev_T1, prev_T2, D);
        // reference: UIPC_KERNEL_ASSERT(D <= squared_d_hat);
        LVec<T, 12> GradD;
        ldist::point_triangle_distance2_gradient(prev_P, prev_T0, prev_T1, prev_T2, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        T lam = -dBdD * GradD.template head<3>().norm();
        LVec<T, 3> n      = lps::cross(prev_T0 - prev_T1, prev_T0 - prev_T2);
        LVec<T, 3> normal = n / n.norm();
        LMat<T, 6, 3> Tk;
        LMat<T, 3, 3> I = LMat<T, 3, 3>::Identity();
        Tk.template block<3, 3>(0, 0) = I - normal * normal.transpose();
        Tk.template block<3, 3>(3, 0) = normal * normal.transpose() - I;
        LVec<T, 3> v1 = (P - prev_P) / dt;

        // suppose P0 = t0 * T0 + t1 * T1 + t2 * T2
        LMat<T, 3, 2> base;
        base.template block<3, 1>(0, 0) = prev_T1 - prev_T0;
        base.template block<3, 1>(0, 1) = prev_T2 - prev_T0;
        LMat<T, 2, 2> Lhs = base.transpose() * base;
        LVec<T, 2>    rhs = base.transpose() * (prev_P - prev_T0);
        LMat<T, 2, 2> Lhs_inv;
        T             det = Lhs(0, 0) * Lhs(1, 1) - Lhs(0, 1) * Lhs(1, 0);
        Lhs_inv(0, 0)     = Lhs(1, 1) / det;
        Lhs_inv(0, 1)     = -Lhs(0, 1) / det;
        Lhs_inv(1, 0)     = -Lhs(1, 0) / det;
        Lhs_inv(1, 1)     = Lhs(0, 0) / det;
        LVec<T, 2> t      = Lhs_inv * rhs;
        T          t1     = t(0);
        T          t2     = t(1);
        T          t0     = 1 - t1 - t2;

        LVec<T, 6> V;
        V.template segment<3>(0) = v1;
        V.template segment<3>(3) = t0 * (T0 - prev_T0) / dt + t1 * (T1 - prev_T1) / dt
                                  + t2 * (T2 - prev_T2) / dt;
        LVec<T, 3> vk = Tk.transpose() * V;
        T          y  = vk.norm() * dt;
        T          F;
        FrictionEnergy(F, lam * mu, eps_v, dt, y);
        return F;
    }

    template <typename T>
    inline void PT_friction_gradient_hessian(LVec<T, 12>&      G,
                                             LMat<T, 12, 12>&  H,
                                             const T&          kappa,
                                             const T&          squared_d_hat,
                                             const T&          mu,
                                             const T&          dt,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& T0,
                                             const LVec<T, 3>& T1,
                                             const LVec<T, 3>& T2,
                                             const LVec<T, 3>& prev_P,
                                             const LVec<T, 3>& prev_T0,
                                             const LVec<T, 3>& prev_T1,
                                             const LVec<T, 3>& prev_T2,
                                             const T&          eps_v)
    {
        T D;
        ldist::point_triangle_distance2(prev_P, prev_T0, prev_T1, prev_T2, D);
        // reference: UIPC_KERNEL_ASSERT(D <= squared_d_hat);
        LVec<T, 12> GradD;
        ldist::point_triangle_distance2_gradient(prev_P, prev_T0, prev_T1, prev_T2, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);

        T lam = -dBdD * GradD.template head<3>().norm();
        LVec<T, 3> n      = lps::cross(prev_T0 - prev_T1, prev_T0 - prev_T2);
        LVec<T, 3> normal = n / n.norm();
        LMat<T, 6, 3> Tk;
        LMat<T, 3, 3> I = LMat<T, 3, 3>::Identity();
        Tk.template block<3, 3>(0, 0) = I - normal * normal.transpose();
        Tk.template block<3, 3>(3, 0) = normal * normal.transpose() - I;
        LVec<T, 3> v1 = (P - prev_P) / dt;

        // suppose prev_P0 = t0 * prev_T0 + t1 * prev_T1 + t2 * prev_T2
        LMat<T, 3, 2> base;
        base.template block<3, 1>(0, 0) = prev_T1 - prev_T0;
        base.template block<3, 1>(0, 1) = prev_T2 - prev_T0;
        LMat<T, 2, 2> Lhs = base.transpose() * base;
        LVec<T, 2>    rhs = base.transpose() * (prev_P - prev_T0);
        LMat<T, 2, 2> Lhs_inv;
        T             det = Lhs(0, 0) * Lhs(1, 1) - Lhs(0, 1) * Lhs(1, 0);
        Lhs_inv(0, 0)     = Lhs(1, 1) / det;
        Lhs_inv(0, 1)     = -Lhs(0, 1) / det;
        Lhs_inv(1, 0)     = -Lhs(1, 0) / det;
        Lhs_inv(1, 1)     = Lhs(0, 0) / det;
        LVec<T, 2> t      = Lhs_inv * rhs;
        T          t1     = t(0);
        T          t2     = t(1);
        T          t0     = 1 - t1 - t2;

        LVec<T, 6> V;
        V.template segment<3>(0) = v1;
        V.template segment<3>(3) = t0 * (T0 - prev_T0) / dt + t1 * (T1 - prev_T1) / dt
                                  + t2 * (T2 - prev_T2) / dt;
        LVec<T, 3> vk = Tk.transpose() * V;
        T          y  = vk.norm() * dt;
        LVec<T, 6> dFdV;
        dFrictionEnergydV(dFdV, lam * mu, Tk, eps_v, dt, vk);
        LMat<T, 6, 12> GradV;
        GradV.setZero();
        GradV.template block<3, 3>(0, 0) = I / dt;
        GradV.template block<3, 3>(3, 3) = I * t0 / dt;
        GradV.template block<3, 3>(3, 6) = I * t1 / dt;
        GradV.template block<3, 3>(3, 9) = I * t2 / dt;
        LMat<T, 12, 6> GradV_transpose = GradV.transpose();
        G                              = GradV_transpose * dFdV;

        LMat<T, 6, 6> ddFddV;
        ddFrictionEnergyddV(ddFddV, lam * mu, Tk, eps_v, dt, vk);
        H = GradV_transpose * ddFddV * GradV;
    }

    template <typename T>
    [[nodiscard]] inline T EE_friction_energy(const T&          kappa,
                                              const T&          squared_d_hat,
                                              const T&          mu,
                                              const T&          dt,
                                              const LVec<T, 3>& P0,
                                              const LVec<T, 3>& P1,
                                              const LVec<T, 3>& Q0,
                                              const LVec<T, 3>& Q1,
                                              const LVec<T, 3>& prev_P0,
                                              const LVec<T, 3>& prev_P1,
                                              const LVec<T, 3>& prev_Q0,
                                              const LVec<T, 3>& prev_Q1,
                                              const T&          eps_v)
    {
        T D;
        ldist::edge_edge_distance2(prev_P0, prev_P1, prev_Q0, prev_Q1, D);
        // reference: UIPC_KERNEL_ASSERT(D <= squared_d_hat);
        LVec<T, 12> GradD;
        ldist::edge_edge_distance2_gradient(prev_P0, prev_P1, prev_Q0, prev_Q1, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        LVec<T, 3> n      = lps::cross(prev_P0 - prev_P1, prev_Q0 - prev_Q1);
        LVec<T, 3> normal = n / n.norm();
        LMat<T, 6, 3> Tk;
        LMat<T, 3, 3> I = LMat<T, 3, 3>::Identity();
        Tk.template block<3, 3>(0, 0) = I - normal * normal.transpose();
        Tk.template block<3, 3>(3, 0) = normal * normal.transpose() - I;

        // suppose prev_P_project = prev_P0 + t0 * (prev_P1 - prev_P0)
        // prev_Q_project = prev_Q0 + t1 * (prev_Q1 - prev_Q0)
        LMat<T, 3, 2> base;
        base.template block<3, 1>(0, 0) = prev_P1 - prev_P0;
        base.template block<3, 1>(0, 1) = prev_Q1 - prev_Q0;
        LMat<T, 2, 2> X;
        X.setZero();
        X(0, 0)       = -1;
        X(1, 1)       = 1;
        LMat<T, 2, 2> Lhs = base.transpose() * base * X;
        LVec<T, 2>    rhs = base.transpose() * (prev_P0 - prev_Q0);
        LMat<T, 2, 2> Lhs_inv;
        T             det = Lhs(0, 0) * Lhs(1, 1) - Lhs(0, 1) * Lhs(1, 0);
        Lhs_inv(0, 0)     = Lhs(1, 1) / det;
        Lhs_inv(0, 1)     = -Lhs(0, 1) / det;
        Lhs_inv(1, 0)     = -Lhs(1, 0) / det;
        Lhs_inv(1, 1)     = Lhs(0, 0) / det;
        LVec<T, 2> t      = Lhs_inv * rhs;
        T          t0     = t(0);
        T          t1     = t(1);

        LVec<T, 6> V;
        V.template segment<3>(0) = (P0 - prev_P0) * (1 - t0) / dt + (P1 - prev_P1) * t0 / dt;
        V.template segment<3>(3) = (Q0 - prev_Q0) * (1 - t1) / dt + (Q1 - prev_Q1) * t1 / dt;
        LVec<T, 3> vk = Tk.transpose() * V;
        T          y  = vk.norm() * dt;
        T          lam = -dBdD
                        * (GradD.template segment<3>(0) * (1 - t0)
                           + GradD.template segment<3>(3) * t0)
                              .norm();
        T F;
        FrictionEnergy(F, lam * mu, eps_v, dt, y);
        return F;
    }

    template <typename T>
    inline void EE_friction_gradient_hessian(LVec<T, 12>&      G,
                                             LMat<T, 12, 12>&  H,
                                             const T&          kappa,
                                             const T&          squared_d_hat,
                                             const T&          mu,
                                             const T&          dt,
                                             const LVec<T, 3>& P0,
                                             const LVec<T, 3>& P1,
                                             const LVec<T, 3>& Q0,
                                             const LVec<T, 3>& Q1,
                                             const LVec<T, 3>& prev_P0,
                                             const LVec<T, 3>& prev_P1,
                                             const LVec<T, 3>& prev_Q0,
                                             const LVec<T, 3>& prev_Q1,
                                             const T&          eps_v)
    {
        T D;
        ldist::edge_edge_distance2(prev_P0, prev_P1, prev_Q0, prev_Q1, D);
        // reference: UIPC_KERNEL_ASSERT(D <= squared_d_hat);
        LVec<T, 12> GradD;
        ldist::edge_edge_distance2_gradient(prev_P0, prev_P1, prev_Q0, prev_Q1, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        LVec<T, 3> n      = lps::cross(prev_P0 - prev_P1, prev_Q0 - prev_Q1);
        LVec<T, 3> normal = n / n.norm();
        LMat<T, 6, 3> Tk;
        LMat<T, 3, 3> I = LMat<T, 3, 3>::Identity();
        Tk.template block<3, 3>(0, 0) = I - normal * normal.transpose();
        Tk.template block<3, 3>(3, 0) = normal * normal.transpose() - I;

        // suppose prev_P_project = prev_P0 + t0 * (prev_P1 - prev_P0)
        // prev_Q_project = prev_Q0 + t1 * (prev_Q1 - prev_Q0)
        LMat<T, 3, 2> base;
        base.template block<3, 1>(0, 0) = prev_P1 - prev_P0;
        base.template block<3, 1>(0, 1) = prev_Q1 - prev_Q0;
        LMat<T, 2, 2> X;
        X.setZero();
        X(0, 0)       = -1;
        X(1, 1)       = 1;
        LMat<T, 2, 2> Lhs = base.transpose() * base * X;
        LVec<T, 2>    rhs = base.transpose() * (prev_P0 - prev_Q0);
        LMat<T, 2, 2> Lhs_inv;
        T             det = Lhs(0, 0) * Lhs(1, 1) - Lhs(0, 1) * Lhs(1, 0);
        Lhs_inv(0, 0)     = Lhs(1, 1) / det;
        Lhs_inv(0, 1)     = -Lhs(0, 1) / det;
        Lhs_inv(1, 0)     = -Lhs(1, 0) / det;
        Lhs_inv(1, 1)     = Lhs(0, 0) / det;
        LVec<T, 2> t      = Lhs_inv * rhs;
        T          t0     = t(0);
        T          t1     = t(1);

        LVec<T, 6> V;
        V.template segment<3>(0) = (P0 - prev_P0) * (1 - t0) / dt + (P1 - prev_P1) * t0 / dt;
        V.template segment<3>(3) = (Q0 - prev_Q0) * (1 - t1) / dt + (Q1 - prev_Q1) * t1 / dt;
        LVec<T, 3> vk = Tk.transpose() * V;
        T          y  = vk.norm() * dt;
        T          lam = -dBdD
                        * (GradD.template segment<3>(0) * (1 - t0)
                           + GradD.template segment<3>(3) * t0)
                              .norm();

        LVec<T, 6> dFdV;
        dFrictionEnergydV(dFdV, lam * mu, Tk, eps_v, dt, vk);
        LMat<T, 6, 12> GradV;
        GradV.setZero();
        GradV.template block<3, 3>(0, 0) = I * (1 - t0) / dt;
        GradV.template block<3, 3>(0, 3) = I * t0 / dt;
        GradV.template block<3, 3>(3, 6) = I * (1 - t1) / dt;
        GradV.template block<3, 3>(3, 9) = I * t1 / dt;
        LMat<T, 12, 6> GradV_transpose = GradV.transpose();
        G                              = GradV_transpose * dFdV;

        LMat<T, 6, 6> ddFddV;
        ddFrictionEnergyddV(ddFddV, lam * mu, Tk, eps_v, dt, vk);
        H = GradV_transpose * ddFddV * GradV;
    }

    template <typename T>
    [[nodiscard]] inline T PE_friction_energy(const T&          kappa,
                                              const T&          squared_d_hat,
                                              const T&          mu,
                                              const T&          dt,
                                              const LVec<T, 3>& P,
                                              const LVec<T, 3>& E0,
                                              const LVec<T, 3>& E1,
                                              const LVec<T, 3>& prev_P,
                                              const LVec<T, 3>& prev_E0,
                                              const LVec<T, 3>& prev_E1,
                                              const T&          eps_v)
    {
        T D;
        ldist::point_edge_distance2(prev_P, prev_E0, prev_E1, D);
        // reference: UIPC_KERNEL_ASSERT(D <= squared_d_hat);
        LVec<T, 9> GradD;
        ldist::point_edge_distance2_gradient(prev_P, prev_E0, prev_E1, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        T lam = -dBdD * GradD.template head<3>().norm();
        // suppose prev_P0 = t0 * prev_E0 + t1 * prev_E1
        T t0 = lps::dot(prev_P - prev_E1, prev_E0 - prev_E1)
               / (prev_E0 - prev_E1).squaredNorm();
        T t1 = 1 - t0;

        LVec<T, 3> prev_P0 = t0 * prev_E0 + t1 * prev_E1;
        LVec<T, 3> n       = prev_P0 - prev_P;
        LVec<T, 3> normal  = n / n.norm();
        LMat<T, 6, 3> Tk;
        LMat<T, 3, 3> I = LMat<T, 3, 3>::Identity();
        Tk.template block<3, 3>(0, 0) = I - normal * normal.transpose();
        Tk.template block<3, 3>(3, 0) = normal * normal.transpose() - I;
        LVec<T, 3> v1 = (P - prev_P) / dt;

        LVec<T, 6> V;
        V.template segment<3>(0) = v1;
        V.template segment<3>(3) = t0 * (E0 - prev_E0) / dt + t1 * (E1 - prev_E1) / dt;
        LVec<T, 3> vk = Tk.transpose() * V;
        T          y  = vk.norm() * dt;
        T          F;
        FrictionEnergy(F, lam * mu, eps_v, dt, y);
        return F;
    }

    template <typename T>
    inline void PE_friction_gradient_hessian(LVec<T, 9>&       G,
                                             LMat<T, 9, 9>&    H,
                                             const T&          kappa,
                                             const T&          squared_d_hat,
                                             const T&          mu,
                                             const T&          dt,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& E0,
                                             const LVec<T, 3>& E1,
                                             const LVec<T, 3>& prev_P,
                                             const LVec<T, 3>& prev_E0,
                                             const LVec<T, 3>& prev_E1,
                                             const T&          eps_v)
    {
        T D;
        ldist::point_edge_distance2(prev_P, prev_E0, prev_E1, D);
        // reference: UIPC_KERNEL_ASSERT(D <= squared_d_hat);
        LVec<T, 9> GradD = LVec<T, 9>::Zero();
        ldist::point_edge_distance2_gradient(prev_P, prev_E0, prev_E1, GradD);

        T dBdD = lps::zero_of<T>();
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);

        T lam = -dBdD * GradD.template head<3>().norm();

        // suppose prev_P0 = t0 * prev_E0 + t1 * prev_E1
        T t0 = lps::dot(prev_P - prev_E1, prev_E0 - prev_E1)
               / (prev_E0 - prev_E1).squaredNorm();
        T t1 = 1 - t0;

        LVec<T, 3> prev_P0 = t0 * prev_E0 + t1 * prev_E1;
        LVec<T, 3> n       = prev_P0 - prev_P;
        LVec<T, 3> normal  = n / n.norm();
        LMat<T, 6, 3> Tk;
        LMat<T, 3, 3> I = LMat<T, 3, 3>::Identity();
        Tk.template block<3, 3>(0, 0) = I - normal * normal.transpose();
        Tk.template block<3, 3>(3, 0) = normal * normal.transpose() - I;
        LVec<T, 3> v1 = (P - prev_P) / dt;

        LVec<T, 6> V;
        V.template segment<3>(0) = v1;
        V.template segment<3>(3) = t0 * (E0 - prev_E0) / dt + t1 * (E1 - prev_E1) / dt;
        LVec<T, 3> vk = Tk.transpose() * V;
        T          y  = vk.norm() * dt;
        LVec<T, 6> dFdV;
        dFrictionEnergydV(dFdV, lam * mu, Tk, eps_v, dt, vk);
        LMat<T, 6, 9> GradV;
        GradV.setZero();
        GradV.template block<3, 3>(0, 0) = I / dt;
        GradV.template block<3, 3>(3, 3) = I * t0 / dt;
        GradV.template block<3, 3>(3, 6) = I * t1 / dt;
        LMat<T, 9, 6> GradV_transpose = GradV.transpose();
        G                              = GradV_transpose * dFdV;

        LMat<T, 6, 6> ddFddV;
        ddFrictionEnergyddV(ddFddV, lam * mu, Tk, eps_v, dt, vk);
        H = GradV_transpose * ddFddV * GradV;
    }

    template <typename T>
    [[nodiscard]] inline T PP_friction_energy(const T&          kappa,
                                              const T&          squared_d_hat,
                                              const T&          mu,
                                              const T&          dt,
                                              const LVec<T, 3>& P,
                                              const LVec<T, 3>& Q,
                                              const LVec<T, 3>& prev_P,
                                              const LVec<T, 3>& prev_Q,
                                              const T&          eps_v)
    {
        T D;
        ldist::point_point_distance2(prev_P, prev_Q, D);
        // reference: UIPC_KERNEL_ASSERT(D <= squared_d_hat);
        LVec<T, 6> GradD;
        ldist::point_point_distance2_gradient(prev_P, prev_Q, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        T lam = -dBdD * GradD.template head<3>().norm();

        LVec<T, 3>    n      = prev_Q - prev_P;
        LVec<T, 3>    normal = n / n.norm();
        LMat<T, 6, 3> Tk;
        LMat<T, 3, 3> I = LMat<T, 3, 3>::Identity();
        Tk.template block<3, 3>(0, 0) = I - normal * normal.transpose();
        Tk.template block<3, 3>(3, 0) = normal * normal.transpose() - I;

        LVec<T, 6> V;
        V.template segment<3>(0) = (P - prev_P) / dt;
        V.template segment<3>(3) = (Q - prev_Q) / dt;
        LVec<T, 3> vk = Tk.transpose() * V;
        T          y  = vk.norm() * dt;
        T          F;
        FrictionEnergy(F, lam * mu, eps_v, dt, y);
        return F;
    }

    template <typename T>
    inline void PP_friction_gradient_hessian(LVec<T, 6>&       G,
                                             LMat<T, 6, 6>&    H,
                                             const T&          kappa,
                                             const T&          squared_d_hat,
                                             const T&          mu,
                                             const T&          dt,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& Q,
                                             const LVec<T, 3>& prev_P,
                                             const LVec<T, 3>& prev_Q,
                                             const T&          eps_v)
    {
        T D;
        ldist::point_point_distance2(prev_P, prev_Q, D);
        // reference: UIPC_KERNEL_ASSERT(D <= squared_d_hat);
        LVec<T, 6> GradD;
        ldist::point_point_distance2_gradient(prev_P, prev_Q, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, squared_d_hat);
        T lam = -dBdD * GradD.template head<3>().norm();

        LVec<T, 3>    n      = prev_Q - prev_P;
        LVec<T, 3>    normal = n / n.norm();
        LMat<T, 6, 3> Tk;
        LMat<T, 3, 3> I = LMat<T, 3, 3>::Identity();
        Tk.template block<3, 3>(0, 0) = I - normal * normal.transpose();
        Tk.template block<3, 3>(3, 0) = normal * normal.transpose() - I;

        LVec<T, 6> V;
        V.template segment<3>(0) = (P - prev_P) / dt;
        V.template segment<3>(3) = (Q - prev_Q) / dt;
        LVec<T, 3> vk = Tk.transpose() * V;
        T          y  = vk.norm() * dt;
        LVec<T, 6> dFdV;
        dFrictionEnergydV(dFdV, lam * mu, Tk, eps_v, dt, vk);
        // GradV = LMat<T, 6, 6>::Identity();
        G = dFdV;

        LMat<T, 6, 6> ddFddV;
        ddFrictionEnergyddV(ddFddV, lam * mu, Tk, eps_v, dt, vk);
        H = ddFddV;
    }
}  // namespace sym::ipc_contact
}  // namespace uipc::backend::luisa
