#pragma once
/********************************************************************
 * @file contact_models/codim_ipc_simplex_normal_contact_function.h
 * @brief T8.1b — the codim IPC simplex (PT/EE/PE/PP) normal-barrier device math (port of
 * src/backends/cuda/contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h,
 * 514 reference lines).  Also the reference's *only* home of the contact-table kappa
 * accessors (`PT_kappa`/`EE_kappa`/`PE_kappa`/`PP_kappa`), which the frictional sibling
 * re-states in its own `*_contact_coeff` form.
 *
 * WHAT IT IS.  For each primitive type: the energy (a flag-dispatched squared distance
 * composed with the barrier law of `codim_ipc_contact_function.h`), the gradient and the
 * Hessian-by-chain-rule
 *     G = dB/dD * dD/dx
 *     H = d2B/dD2 * dD/dx * dD/dx^T + dB/dD * d2D/dx^2
 * plus the EE mollification (`e_k` and its derivatives, the near-parallel guard) and the
 * two *non-flagged* overloads the reference keeps for its own convenience
 * (`PT_barrier_energy(kappa, ...)` and `PT_barrier_gradient_hessian(kappa, ...)`, which
 * call the unflagged `distance::point_triangle_distance2*`).
 *
 * TYPE MAPPING (cookbook §3/§5): `Vector3/Vector4i/...` -> `lps::LVec<T,3>` /
 * `lps::LVec<IndexT,4>`; `Vector12` -> `lps::LVec<T,12>`; `Matrix12x12` ->
 * `lps::LMat<T,12,12>` (the same element-wise layout as Eigen's column-major matrix);
 * `cuda_tool::CDense2D<ContactCoeff>::operator()(r, c)` -> a *flat* table
 * (`index = row * extent + col`) plus the explicit extent (cookbook §7.35 rule 1 — the
 * port's device layer has no 2D buffer/view), which is why every kernel here takes
 * `contact_tabular_size` as an extra scalar argument.
 *
 * DEVIATIONS (all reported):
 *  * the four kappa accessors take `(table, extent, cids)` and return `T` (the reference's
 *    `Float`); the `_contact_coeff` pair of the frictional sibling returns *two* out-values
 *    for the same reason.  The *arithmetic* (the member sums and their `/3`, `/4`, `/2`
 *    scalings) is transcribed verbatim, and the table is read-only inside a kernel, so the
 *    value is identical.
 *  * nothing else deviates: no branch of this header reads a device predicate (the
 *    flagged distances and the mollifier are already the port's select-based spellings).
 *  * `distance::*_thickness`/`*_d_hat` are **not** called here (the reference calls them in
 *    its kernels, so the port does too, via `utils/codim_thickness.h` and
 *    `utils/primitive_d_hat.h`).
 *********************************************************************/
#include <type_define.h>
#include <contact_system/contact_coeff.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::luisa
{
namespace sym::codim_ipc_simplex_contact
{
    namespace ldist = luisa_device::math::distance;
    namespace lcrt  = ::luisa::compute;
    using lps::LMat;
    using lps::LVec;

    // ---------------------------------------------------------------------------
    //  The contact-table kappa accessors (the reference's `CDense2D` reads)
    // ---------------------------------------------------------------------------
    template <typename Cids>
    [[nodiscard]] inline lcrt::Var<Float> PT_kappa(
        const lcrt::BufferVar<ContactCoeff>& table,
        const lcrt::Var<std::int32_t>&       extent,
        const Cids&                          cids) noexcept
    {
        lcrt::Var<Float> kappa = 0.0;
        for(int j = 1; j < 4; ++j)
        {
            kappa = kappa + table->read(cids(0) * extent + cids(j)).kappa;
        }
        return kappa / 3.0;
    }

    template <typename Cids>
    [[nodiscard]] inline lcrt::Var<Float> EE_kappa(
        const lcrt::BufferVar<ContactCoeff>& table,
        const lcrt::Var<std::int32_t>&       extent,
        const Cids&                          cids) noexcept
    {
        lcrt::Var<Float> kappa = 0.0;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                kappa = kappa + table->read(cids(j) * extent + cids(k)).kappa;
            }
        }
        return kappa / 4.0;
    }

    template <typename Cids>
    [[nodiscard]] inline lcrt::Var<Float> PE_kappa(
        const lcrt::BufferVar<ContactCoeff>& table,
        const lcrt::Var<std::int32_t>&       extent,
        const Cids&                          cids) noexcept
    {
        lcrt::Var<Float> kappa = 0.0;
        for(int j = 1; j < 3; ++j)
        {
            kappa = kappa + table->read(cids(0) * extent + cids(j)).kappa;
        }
        return kappa / 2.0;
    }

    template <typename Cids>
    [[nodiscard]] inline lcrt::Var<Float> PP_kappa(
        const lcrt::BufferVar<ContactCoeff>& table,
        const lcrt::Var<std::int32_t>&       extent,
        const Cids&                          cids) noexcept
    {
        return table->read(cids(0) * extent + cids(1)).kappa;
    }

    // ---------------------------------------------------------------------------
    //  Point - Triangle
    // ---------------------------------------------------------------------------
    template <typename T>
    [[nodiscard]] inline T PT_barrier_energy(const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& T0,
                                             const LVec<T, 3>& T1,
                                             const LVec<T, 3>& T2)
    {
        namespace d = lps::detail;
        T D;
        ldist::point_triangle_distance2(P, T0, T1, T2, D);
        T B;
        sym::codim_ipc_contact::KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    template <typename T, typename Flag>
    [[nodiscard]] inline T PT_barrier_energy(const Flag&       flag,
                                             const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& T0,
                                             const LVec<T, 3>& T1,
                                             const LVec<T, 3>& T2)
    {
        T D;
        ldist::point_triangle_distance2(flag, P, T0, T1, T2, D);
        T B;
        sym::codim_ipc_contact::KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    template <typename T>
    inline void PT_barrier_gradient_hessian(LVec<T, 12>&      G,
                                            LMat<T, 12, 12>&  H,
                                            const T&          kappa,
                                            const T&          d_hat,
                                            const T&          thickness,
                                            const LVec<T, 3>& P,
                                            const LVec<T, 3>& T0,
                                            const LVec<T, 3>& T1,
                                            const LVec<T, 3>& T2)
    {
        namespace d = lps::detail;
        T D;
        ldist::point_triangle_distance2(P, T0, T1, T2, D);

        LVec<T, 12> GradD;
        ldist::point_triangle_distance2_gradient(P, T0, T1, T2, GradD);

        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x} $$
        G = dBdD * GradD;

        T ddBddD;
        sym::codim_ipc_contact::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        LMat<T, 12, 12> HessD;
        ldist::point_triangle_distance2_hessian(P, T0, T1, T2, HessD);

        //tex: $$ H = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T
        //            + \frac{\partial B}{\partial D} \nabla^2 D $$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    template <typename T, typename Flag>
    inline void PT_barrier_gradient_hessian(LVec<T, 12>&      G,
                                            LMat<T, 12, 12>&  H,
                                            const Flag&       flag,
                                            const T&          kappa,
                                            const T&          d_hat,
                                            const T&          thickness,
                                            const LVec<T, 3>& P,
                                            const LVec<T, 3>& T0,
                                            const LVec<T, 3>& T1,
                                            const LVec<T, 3>& T2)
    {
        namespace d = lps::detail;
        T D;
        ldist::point_triangle_distance2(flag, P, T0, T1, T2, D);

        LVec<T, 12> GradD;
        ldist::point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x} $$
        G = dBdD * GradD;

        T ddBddD;
        sym::codim_ipc_contact::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        LMat<T, 12, 12> HessD;
        ldist::point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);

        //tex: $$ H = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T
        //            + \frac{\partial B}{\partial D} \nabla^2 D $$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    template <typename T, typename Flag>
    inline void PT_barrier_gradient(LVec<T, 12>&      G,
                                    const Flag&       flag,
                                    const T&          kappa,
                                    const T&          d_hat,
                                    const T&          thickness,
                                    const LVec<T, 3>& P,
                                    const LVec<T, 3>& T0,
                                    const LVec<T, 3>& T1,
                                    const LVec<T, 3>& T2)
    {
        T D;
        ldist::point_triangle_distance2(flag, P, T0, T1, T2, D);

        LVec<T, 12> GradD;
        ldist::point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    // ---------------------------------------------------------------------------
    //  Edge - Edge (mollified)
    // ---------------------------------------------------------------------------
    template <typename T, typename Flag>
    [[nodiscard]] inline T mollified_EE_barrier_energy(const Flag&       flag,
                                                       const T&          kappa,
                                                       const T&          d_hat,
                                                       const T&          thickness,
                                                       const LVec<T, 3>& t0_Ea0,
                                                       const LVec<T, 3>& t0_Ea1,
                                                       const LVec<T, 3>& t0_Eb0,
                                                       const LVec<T, 3>& t0_Eb1,
                                                       const LVec<T, 3>& Ea0,
                                                       const LVec<T, 3>& Ea1,
                                                       const LVec<T, 3>& Eb0,
                                                       const LVec<T, 3>& Eb1)
    {
        // using mollifier to improve the smoothness of the edge-edge barrier
        T D;
        ldist::edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);
        T B;
        sym::codim_ipc_contact::KappaBarrier(B, kappa, D, d_hat, thickness);

        T eps_x;
        ldist::edge_edge_mollifier_threshold(t0_Ea0,
                                            t0_Ea1,
                                            t0_Eb0,
                                            t0_Eb1,
                                            static_cast<double>(1e-3),
                                            eps_x);

        T ek;
        ldist::edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        return ek * B;
    }

    template <typename T, typename Flag>
    inline void mollified_EE_barrier_gradient_hessian(LVec<T, 12>&     G,
                                                      LMat<T, 12, 12>& H,
                                                      const Flag&      flag,
                                                      const T&         kappa,
                                                      const T&         d_hat,
                                                      const T&         thickness,
                                                      const LVec<T, 3>& t0_Ea0,
                                                      const LVec<T, 3>& t0_Ea1,
                                                      const LVec<T, 3>& t0_Eb0,
                                                      const LVec<T, 3>& t0_Eb1,
                                                      const LVec<T, 3>& Ea0,
                                                      const LVec<T, 3>& Ea1,
                                                      const LVec<T, 3>& Eb0,
                                                      const LVec<T, 3>& Eb1)
    {
        namespace d = lps::detail;
        T D;
        ldist::edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        //tex: $$ \nabla D $$
        LVec<T, 12> GradD;
        ldist::edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        //tex: $$ \nabla^2 D $$
        LMat<T, 12, 12> HessD;
        ldist::edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);

        T B;
        sym::codim_ipc_contact::KappaBarrier(B, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial B}{\partial D} $$
        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial^2 B}{\partial D^2} $$
        T ddBddD;
        sym::codim_ipc_contact::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex: $$ \nabla B = \frac{\partial B}{\partial D} \nabla D $$
        LVec<T, 12> GradB = dBdD * GradD;

        //tex: $$ \nabla^2 B = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T
        //                    + \frac{\partial B}{\partial D} \nabla^2 D $$
        LMat<T, 12, 12> HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;

        //tex: $$ \epsilon_x $$
        T eps_x;
        ldist::edge_edge_mollifier_threshold(t0_Ea0,
                                            t0_Ea1,
                                            t0_Eb0,
                                            t0_Eb1,
                                            static_cast<double>(1e-3),
                                            eps_x);

        //tex: $$ e_k $$
        T ek;
        ldist::edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        //tex: $$ \nabla e_k $$
        LVec<T, 12> Gradek;
        ldist::edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        //tex: $$ \nabla^2 e_k $$
        LMat<T, 12, 12> Hessek;
        ldist::edge_edge_mollifier_hessian(Ea0, Ea1, Eb0, Eb1, eps_x, Hessek);

        //tex: $$ G = \nabla e_k B + e_k \nabla B $$
        G = Gradek * B + ek * GradB;

        //tex: $$ H = \nabla^2 e_k B + \nabla e_k \nabla B^T + \nabla B \nabla e_k^T
        //            + e_k \nabla^2 B $$
        H = Hessek * B + Gradek * GradB.transpose() + GradB * Gradek.transpose() + ek * HessB;
    }

    template <typename T, typename Flag>
    inline void mollified_EE_barrier_gradient(LVec<T, 12>&     G,
                                              const Flag&      flag,
                                              const T&         kappa,
                                              const T&         d_hat,
                                              const T&         thickness,
                                              const LVec<T, 3>& t0_Ea0,
                                              const LVec<T, 3>& t0_Ea1,
                                              const LVec<T, 3>& t0_Eb0,
                                              const LVec<T, 3>& t0_Eb1,
                                              const LVec<T, 3>& Ea0,
                                              const LVec<T, 3>& Ea1,
                                              const LVec<T, 3>& Eb0,
                                              const LVec<T, 3>& Eb1)
    {
        T D;
        ldist::edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        LVec<T, 12> GradD;
        ldist::edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        T B;
        sym::codim_ipc_contact::KappaBarrier(B, kappa, D, d_hat, thickness);

        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        LVec<T, 12> GradB = dBdD * GradD;

        T eps_x;
        ldist::edge_edge_mollifier_threshold(t0_Ea0,
                                            t0_Ea1,
                                            t0_Eb0,
                                            t0_Eb1,
                                            static_cast<double>(1e-3),
                                            eps_x);

        T ek;
        ldist::edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        LVec<T, 12> Gradek;
        ldist::edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        G = Gradek * B + ek * GradB;
    }

    // ---------------------------------------------------------------------------
    //  Point - Edge
    // ---------------------------------------------------------------------------
    template <typename T, typename Flag>
    [[nodiscard]] inline T PE_barrier_energy(const Flag&       flag,
                                             const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& E0,
                                             const LVec<T, 3>& E1)
    {
        T D = lps::zero_of<T>();
        ldist::point_edge_distance2(flag, P, E0, E1, D);
        T E = lps::zero_of<T>();
        sym::codim_ipc_contact::KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    template <typename T, typename Flag>
    inline void PE_barrier_gradient_hessian(LVec<T, 9>&       G,
                                            LMat<T, 9, 9>&    H,
                                            const Flag&       flag,
                                            const T&          kappa,
                                            const T&          d_hat,
                                            const T&          thickness,
                                            const LVec<T, 3>& P,
                                            const LVec<T, 3>& E0,
                                            const LVec<T, 3>& E1)
    {
        T D = lps::zero_of<T>();
        ldist::point_edge_distance2(flag, P, E0, E1, D);

        LVec<T, 9> GradD;
        ldist::point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        LMat<T, 9, 9> HessD;
        ldist::point_edge_distance2_hessian(flag, P, E0, E1, HessD);

        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x} $$
        G = dBdD * GradD;

        T ddBddD;
        sym::codim_ipc_contact::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex: $$ H = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T
        //            + \frac{\partial B}{\partial D} \nabla^2 D $$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    template <typename T, typename Flag>
    inline void PE_barrier_gradient(LVec<T, 9>&       G,
                                    const Flag&       flag,
                                    const T&          kappa,
                                    const T&          d_hat,
                                    const T&          thickness,
                                    const LVec<T, 3>& P,
                                    const LVec<T, 3>& E0,
                                    const LVec<T, 3>& E1)
    {
        T D = lps::zero_of<T>();
        ldist::point_edge_distance2(flag, P, E0, E1, D);

        LVec<T, 9> GradD;
        ldist::point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    // ---------------------------------------------------------------------------
    //  Point - Point
    // ---------------------------------------------------------------------------
    template <typename T, typename Flag>
    [[nodiscard]] inline T PP_barrier_energy(const Flag&       flag,
                                             const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const LVec<T, 3>& P0,
                                             const LVec<T, 3>& P1)
    {
        T D = lps::zero_of<T>();
        ldist::point_point_distance2(flag, P0, P1, D);
        T E = lps::zero_of<T>();
        sym::codim_ipc_contact::KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    template <typename T, typename Flag>
    inline void PP_barrier_gradient_hessian(LVec<T, 6>&       G,
                                            LMat<T, 6, 6>&    H,
                                            const Flag&       flag,
                                            const T&          kappa,
                                            const T&          d_hat,
                                            const T&          thickness,
                                            const LVec<T, 3>& P0,
                                            const LVec<T, 3>& P1)
    {
        T D = lps::zero_of<T>();
        ldist::point_point_distance2(flag, P0, P1, D);

        LVec<T, 6> GradD;
        ldist::point_point_distance2_gradient(flag, P0, P1, GradD);

        LMat<T, 6, 6> HessD;
        ldist::point_point_distance2_hessian(flag, P0, P1, HessD);

        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x} $$
        G = dBdD * GradD;

        T ddBddD;
        sym::codim_ipc_contact::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex: $$ H = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T
        //            + \frac{\partial B}{\partial D} \nabla^2 D $$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    template <typename T, typename Flag>
    inline void PP_barrier_gradient(LVec<T, 6>&       G,
                                    const Flag&       flag,
                                    const T&          kappa,
                                    const T&          d_hat,
                                    const T&          thickness,
                                    const LVec<T, 3>& P0,
                                    const LVec<T, 3>& P1)
    {
        T D = lps::zero_of<T>();
        ldist::point_point_distance2(flag, P0, P1, D);

        LVec<T, 6> GradD;
        ldist::point_point_distance2_gradient(flag, P0, P1, GradD);

        T dBdD;
        sym::codim_ipc_contact::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }
}  // namespace sym::codim_ipc_simplex_contact
}  // namespace uipc::backend::luisa
