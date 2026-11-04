#pragma once
/********************************************************************
 * @file contact_models/ipc_vertex_half_plane_contact_function.h
 * @brief T8.1b — the IPC vertex/half-plane contact device math (port of
 * src/backends/cuda/contact_system/contact_models/ipc_vertex_half_plane_contact_function.h,
 * 204 reference lines).
 *
 * Namespace `sym::ipc_vertex_half_contact`, and — exactly like the reference — it pulls the
 * generated half-plane distance family (`sym/vertex_half_plane_distance.inl`) and the two
 * codim IPC headers it builds on.  The four laws:
 *   D = HalfPlaneD(v, P, N)                     (the squared gap to the plane)
 *   E = KappaBarrier(kappa, D, d_hat, thickness)                       (normal barrier)
 *   f = normal_force(kappa, d_hat, thickness, prev_D)                  (previous-step force)
 *   friction_energy/gradient/hessian(mu, f, eps_vh, TR(v, prev_v, e1, e2))
 * with the tangential basis `(e1, e2)` of `compute_tan_basis` and the Jacobian `dTRdx`.
 *
 * TYPE MAPPING / DEVIATIONS: `Eigen::Vector<T,3>` -> `lps::LVec<T,3>`, `Matrix<T,3,3>` ->
 * `lps::LMat<T,3,3>`, `Matrix<T,2,3>` -> `lps::LMat<T,2,3>`, `.dot()/.cross()/.normalized()`
 * -> `lps::dot` / `lps::cross` / the member.  The one structural change is
 * `compute_tan_basis`: the reference branches on `N.dot(Vector3::UnitX()) > 0.9`, a *device*
 * predicate, and both branches differ only in `trial` — so the port selects `trial` first
 * and then performs the (single) shared computation, which is bit-identical to either
 * reference branch (RULE M2, cookbook §7.36).
 *********************************************************************/
#include <type_define.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>

namespace uipc::backend::luisa
{
namespace sym::ipc_vertex_half_contact
{
#include "sym/vertex_half_plane_distance.inl"

    namespace ldist = luisa_device::math::distance;
    using lps::LMat;
    using lps::LVec;

    template <typename T>
    [[nodiscard]] inline T PH_barrier_energy(const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const LVec<T, 3>& v,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& N)
    {
        T D;
        HalfPlaneD(D, v, P, N);
        T E = lps::zero_of<T>();
        sym::codim_ipc_contact::KappaBarrier(E, kappa, D, d_hat, thickness);

        return E;
    }

    template <typename T>
    inline void PH_barrier_gradient_hessian(LVec<T, 3>&       G,
                                            LMat<T, 3, 3>&    H,
                                            const T&          kappa,
                                            const T&          d_hat,
                                            const T&          thickness,
                                            const LVec<T, 3>& v,
                                            const LVec<T, 3>& P,
                                            const LVec<T, 3>& N)
    {
        T D;
        HalfPlaneD(D, v, P, N);

        T dBdD = lps::zero_of<T>();
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        LVec<T, 3> dDdx;
        dHalfPlaneDdx(dDdx, v, P, N);

        G = dBdD * dDdx;

        T ddBddD = lps::zero_of<T>();
        sym::codim_ipc_contact::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        LMat<T, 3, 3> ddDddx;
        ddHalfPlaneDddx(ddDddx, v, P, N);

        H = ddBddD * dDdx * dDdx.transpose() + dBdD * ddDddx;
    }

    template <typename T>
    inline void PH_barrier_gradient(LVec<T, 3>&       G,
                                    const T&          kappa,
                                    const T&          d_hat,
                                    const T&          thickness,
                                    const LVec<T, 3>& v,
                                    const LVec<T, 3>& P,
                                    const LVec<T, 3>& N)
    {
        T D;
        HalfPlaneD(D, v, P, N);

        T dBdD = lps::zero_of<T>();
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        LVec<T, 3> dDdx;
        dHalfPlaneDdx(dDdx, v, P, N);

        G = dBdD * dDdx;
    }

    template <typename T>
    inline void compute_tan_basis(LVec<T, 3>& e1, LVec<T, 3>& e2, const LVec<T, 3>& N)
    {
        // reference: trial = UnitX(); if(N.dot(trial) > 0.9) trial = UnitZ();
        //            e1 = trial.cross(N).normalized(); e2 = N.cross(e1);
        // -- the branch's predicate is a *device* value, and the two bodies differ only in
        // `trial`, so the port selects `trial` and runs the (single) shared body.
        LVec<T, 3> trial = ldist::detail::select_vec(
            LVec<T, 3>::template Unit<0>(),
            LVec<T, 3>::template Unit<2>(),
            lps::dot(N, LVec<T, 3>::template Unit<0>()) > 0.9);
        e1 = lps::cross(trial, N).normalized();
        e2 = lps::cross(N, e1);
    }

    template <typename T>
    [[nodiscard]] inline T PH_friction_energy(const T&          kappa,
                                              const T&          d_hat,
                                              const T&          thickness,
                                              const T&          mu,
                                              const T&          eps_vh,
                                              const LVec<T, 3>& prev_v,
                                              const LVec<T, 3>& v,
                                              const LVec<T, 3>& P,
                                              const LVec<T, 3>& N)
    {
        T prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        T f = sym::codim_ipc_contact::normal_force(kappa, d_hat, thickness, prev_D);

        LVec<T, 3> e1, e2;
        compute_tan_basis(e1, e2, N);

        LVec<T, 2> tan_dV;

        TR(tan_dV, v, prev_v, e1, e2);

        return sym::codim_ipc_contact::friction_energy(mu, f, eps_vh, tan_dV);
    }

    template <typename T>
    inline void PH_friction_gradient_hessian(LVec<T, 3>&       G,
                                             LMat<T, 3, 3>&    H,
                                             const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const T&          mu,
                                             const T&          eps_vh,
                                             const LVec<T, 3>& prev_v,
                                             const LVec<T, 3>& v,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& N)
    {
        T prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        T f = sym::codim_ipc_contact::normal_force(kappa, d_hat, thickness, prev_D);

        LVec<T, 3> e1, e2;
        compute_tan_basis(e1, e2, N);

        LVec<T, 2> tan_dV;

        TR(tan_dV, v, prev_v, e1, e2);

        LVec<T, 2> G2;
        sym::codim_ipc_contact::friction_gradient(G2, mu, f, eps_vh, tan_dV);

        LMat<T, 2, 3> J;
        dTRdx(J, v, prev_v, e1, e2);

        G = J.transpose() * G2;

        LMat<T, 2, 2> H2x2;
        sym::codim_ipc_contact::friction_hessian(H2x2, mu, f, eps_vh, tan_dV);

        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline void PH_friction_gradient(LVec<T, 3>&       G,
                                     const T&          kappa,
                                     const T&          d_hat,
                                     const T&          thickness,
                                     const T&          mu,
                                     const T&          eps_vh,
                                     const LVec<T, 3>& prev_v,
                                     const LVec<T, 3>& v,
                                     const LVec<T, 3>& P,
                                     const LVec<T, 3>& N)
    {
        T prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        T f = sym::codim_ipc_contact::normal_force(kappa, d_hat, thickness, prev_D);

        LVec<T, 3> e1, e2;
        compute_tan_basis(e1, e2, N);

        LVec<T, 2> tan_dV;
        TR(tan_dV, v, prev_v, e1, e2);

        LVec<T, 2> G2;
        sym::codim_ipc_contact::friction_gradient(G2, mu, f, eps_vh, tan_dV);

        LMat<T, 2, 3> J;
        dTRdx(J, v, prev_v, e1, e2);

        G = J.transpose() * G2;
    }
}  // namespace sym::ipc_vertex_half_contact
}  // namespace uipc::backend::luisa
