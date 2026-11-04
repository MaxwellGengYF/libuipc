#pragma once
/********************************************************************
 * @file contact_models/codim_ipc_simplex_frictional_contact_function.h
 * @brief T8.1b — the codim IPC simplex friction device math (port of
 * src/backends/cuda/contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h,
 * 771 reference lines).
 *
 * It lives in the *same* namespace as `codim_ipc_contact_function.h`
 * (`sym::codim_ipc_contact`) exactly like the reference, so the friction law
 * (`friction_energy` / `friction_gradient` / `friction_hessian`), the local `make_spd` and
 * `normal_force` are all in scope unqualified below.
 *
 * STRUCTURE (per primitive type): the reference splits each pair into
 *   `*_friction_basis`  — the previous-step projection (`*_closest_point`), the previous
 *                         gap's normal force `f = normal_force(...)` and the tangential
 *                         relative displacement `tan_rel_dx = basis^T * rel_dx`;
 *   `*_friction_energy` — `friction_energy(mu, f, eps_vh, tan_rel_dx)`;
 *   `*_friction_gradient{,_hessian}` — the reference's `J^T G2` / `J^T H2x2 J` with the
 *                         primitive's tangential Jacobian (`*_jacobi`).
 * The `prev_*` arguments are the *previous* Newton iterate's positions (the same arrays the
 * manager's `prev_positions()` view exposes) and `eps_vh` is the caller's
 * `eps_velocity() * dt` (see `ipc_simplex_frictional_contact.cpp`).
 *
 * TYPE MAPPING / SPELLING: `Eigen::Matrix<T,R,C>` -> `lps::LMat<T,R,C>`,
 * `Eigen::Vector<T,N>` -> `lps::LVec<T,N>`, `cuda_tool::CDense2D<ContactCoeff>` reads ->
 * `(flat table, extent)` with the two members returned through out-parameters (cookbook
 * §7.35 rule 1 and the header note of the normal sibling); the reference's
 * `using namespace friction;` becomes an explicit `friction::` (the port's friction layer
 * is `uipc::backend::luisa::friction`, which the reference's
 * `uipc::backend::cuda::friction` spelling cannot name from here).  No branch of this
 * header reads a device predicate, so nothing had to be recast as a select.
 *********************************************************************/
#include <type_define.h>
#include <contact_system/contact_coeff.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::luisa
{
namespace sym::codim_ipc_contact
{
    namespace ldist = luisa_device::math::distance;
    namespace lcrt  = ::luisa::compute;
    namespace friction = uipc::backend::luisa::friction;
    using lps::LMat;
    using lps::LVec;

    // ---------------------------------------------------------------------------
    //  The contact-table coefficient accessors (the reference's `CDense2D` reads)
    // ---------------------------------------------------------------------------
    template <typename Cids>
    inline void PT_contact_coeff(lcrt::Var<Float>&                    kappa,
                                 lcrt::Var<Float>&                    mu,
                                 const lcrt::BufferVar<ContactCoeff>& table,
                                 const lcrt::Var<std::int32_t>&       extent,
                                 const Cids&                          cids) noexcept
    {
        kappa = 0.0;
        mu    = 0.0;
        for(int j = 1; j < 4; ++j)
        {
            const auto coeff = table->read(cids(0) * extent + cids(j));
            kappa            = kappa + coeff.kappa;
            mu               = mu + coeff.mu;
        }
        kappa = kappa / 3.0;
        mu    = mu / 3.0;
    }

    template <typename Cids>
    inline void EE_contact_coeff(lcrt::Var<Float>&                    kappa,
                                 lcrt::Var<Float>&                    mu,
                                 const lcrt::BufferVar<ContactCoeff>& table,
                                 const lcrt::Var<std::int32_t>&       extent,
                                 const Cids&                          cids) noexcept
    {
        kappa = 0.0;
        mu    = 0.0;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                const auto coeff = table->read(cids(j) * extent + cids(k));
                kappa            = kappa + coeff.kappa;
                mu               = mu + coeff.mu;
            }
        }
        kappa = kappa / 4.0;
        mu    = mu / 4.0;
    }

    template <typename Cids>
    inline void PE_contact_coeff(lcrt::Var<Float>&                    kappa,
                                 lcrt::Var<Float>&                    mu,
                                 const lcrt::BufferVar<ContactCoeff>& table,
                                 const lcrt::Var<std::int32_t>&       extent,
                                 const Cids&                          cids) noexcept
    {
        kappa = 0.0;
        mu    = 0.0;
        for(int j = 1; j < 3; ++j)
        {
            const auto coeff = table->read(cids(0) * extent + cids(j));
            kappa            = kappa + coeff.kappa;
            mu               = mu + coeff.mu;
        }
        kappa = kappa / 2.0;
        mu    = mu / 2.0;
    }

    template <typename Cids>
    inline void PP_contact_coeff(lcrt::Var<Float>&                    kappa,
                                 lcrt::Var<Float>&                    mu,
                                 const lcrt::BufferVar<ContactCoeff>& table,
                                 const lcrt::Var<std::int32_t>&       extent,
                                 const Cids&                          cids) noexcept
    {
        const auto coeff = table->read(cids(0) * extent + cids(1));
        kappa            = coeff.kappa;
        mu               = coeff.mu;
    }

    // ---------------------------------------------------------------------------
    //  Point - Triangle
    // ---------------------------------------------------------------------------
    template <typename T>
    inline void PT_friction_basis(
        // out
        T&                f,
        LVec<T, 2>&       beta,
        LMat<T, 3, 2>&    basis,
        LVec<T, 2>&       tan_rel_dx,
        // in
        const T&          kappa,
        const T&          d_hat,
        const T&          thickness,
        const LVec<T, 3>& prev_P,
        const LVec<T, 3>& prev_T0,
        const LVec<T, 3>& prev_T1,
        const LVec<T, 3>& prev_T2,
        const LVec<T, 3>& P,
        const LVec<T, 3>& T0,
        const LVec<T, 3>& T1,
        const LVec<T, 3>& T2)
    {
        // using the prev values to compute normal force
        friction::point_triangle_closest_point(prev_P, prev_T0, prev_T1, prev_T2, beta);
        friction::point_triangle_tangent_basis(prev_P, prev_T0, prev_T1, prev_T2, basis);
        T prev_D;
        ldist::point_triangle_distance2(prev_P, prev_T0, prev_T1, prev_T2, prev_D);

        f = normal_force(kappa, d_hat, thickness, prev_D);

        LVec<T, 3> dP  = P - prev_P;
        LVec<T, 3> dT0 = T0 - prev_T0;
        LVec<T, 3> dT1 = T1 - prev_T1;
        LVec<T, 3> dT2 = T2 - prev_T2;

        friction::point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, beta, tan_rel_dx);
    }

    template <typename T>
    [[nodiscard]] inline T PT_friction_energy(const T&          kappa,
                                              const T&          d_hat,
                                              const T&          thickness,
                                              const T&          mu,
                                              const T&          eps_vh,
                                              const LVec<T, 3>& prev_P,
                                              const LVec<T, 3>& prev_T0,
                                              const LVec<T, 3>& prev_T1,
                                              const LVec<T, 3>& prev_T2,
                                              const LVec<T, 3>& P,
                                              const LVec<T, 3>& T0,
                                              const LVec<T, 3>& T1,
                                              const LVec<T, 3>& T2)
    {
        T             f;
        LVec<T, 2>    beta;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PT_friction_basis(
            // out
            f,
            beta,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P,
            prev_T0,
            prev_T1,
            prev_T2,
            P,
            T0,
            T1,
            T2);

        T E = friction_energy(mu, f, eps_vh, tan_rel_dx);

        return E;
    }

    template <typename T>
    inline void PT_friction_gradient_hessian(LVec<T, 12>&      G,
                                             LMat<T, 12, 12>&  H,
                                             const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const T&          mu,
                                             const T&          eps_vh,
                                             const LVec<T, 3>& prev_P,
                                             const LVec<T, 3>& prev_T0,
                                             const LVec<T, 3>& prev_T1,
                                             const LVec<T, 3>& prev_T2,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& T0,
                                             const LVec<T, 3>& T1,
                                             const LVec<T, 3>& T2)
    {
        T             f;
        LVec<T, 2>    beta;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PT_friction_basis(
            // out
            f,
            beta,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P,
            prev_T0,
            prev_T1,
            prev_T2,
            P,
            T0,
            T1,
            T2);

        LMat<T, 2, 12> J;
        friction::point_triangle_jacobi(basis, beta, J);

        LVec<T, 2> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        //apply_point_triangle_jacobi(G2, basis, beta, G);
        G = J.transpose() * G2;

        LMat<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline void PT_friction_gradient(LVec<T, 12>&      G,
                                     const T&          kappa,
                                     const T&          d_hat,
                                     const T&          thickness,
                                     const T&          mu,
                                     const T&          eps_vh,
                                     const LVec<T, 3>& prev_P,
                                     const LVec<T, 3>& prev_T0,
                                     const LVec<T, 3>& prev_T1,
                                     const LVec<T, 3>& prev_T2,
                                     const LVec<T, 3>& P,
                                     const LVec<T, 3>& T0,
                                     const LVec<T, 3>& T1,
                                     const LVec<T, 3>& T2)
    {
        T             f;
        LVec<T, 2>    beta;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PT_friction_basis(
            // out
            f,
            beta,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P,
            prev_T0,
            prev_T1,
            prev_T2,
            P,
            T0,
            T1,
            T2);

        LMat<T, 2, 12> J;
        friction::point_triangle_jacobi(basis, beta, J);

        LVec<T, 2> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;
    }

    // ---------------------------------------------------------------------------
    //  Edge - Edge
    // ---------------------------------------------------------------------------
    template <typename T>
    inline void EE_friction_basis(
        // out
        T&                f,
        LVec<T, 2>&       gamma,
        LMat<T, 3, 2>&    basis,
        LVec<T, 2>&       tan_rel_dx,
        // in
        const T&          kappa,
        const T&          d_hat,
        const T&          thickness,
        const LVec<T, 3>& prev_Ea0,
        const LVec<T, 3>& prev_Ea1,
        const LVec<T, 3>& prev_Eb0,
        const LVec<T, 3>& prev_Eb1,
        const LVec<T, 3>& Ea0,
        const LVec<T, 3>& Ea1,
        const LVec<T, 3>& Eb0,
        const LVec<T, 3>& Eb1)
    {
        // using the prev values to compute normal force
        friction::edge_edge_closest_point(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, gamma);
        friction::edge_edge_tangent_basis(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, basis);

        T prev_D;
        ldist::edge_edge_distance2(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, prev_D);

        f = normal_force(kappa, d_hat, thickness, prev_D);

        LVec<T, 3> dEa0 = Ea0 - prev_Ea0;
        LVec<T, 3> dEa1 = Ea1 - prev_Ea1;
        LVec<T, 3> dEb0 = Eb0 - prev_Eb0;
        LVec<T, 3> dEb1 = Eb1 - prev_Eb1;

        friction::edge_edge_tan_rel_dx(dEa0, dEa1, dEb0, dEb1, basis, gamma, tan_rel_dx);
    }

    template <typename T>
    [[nodiscard]] inline T EE_friction_energy(const T&          kappa,
                                              const T&          d_hat,
                                              const T&          thickness,
                                              const T&          mu,
                                              const T&          eps_vh,
                                              const LVec<T, 3>& prev_Ea0,
                                              const LVec<T, 3>& prev_Ea1,
                                              const LVec<T, 3>& prev_Eb0,
                                              const LVec<T, 3>& prev_Eb1,
                                              const LVec<T, 3>& Ea0,
                                              const LVec<T, 3>& Ea1,
                                              const LVec<T, 3>& Eb0,
                                              const LVec<T, 3>& Eb1)
    {
        T             f;
        LVec<T, 2>    gamma;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        EE_friction_basis(
            // out
            f,
            gamma,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_Ea0,
            prev_Ea1,
            prev_Eb0,
            prev_Eb1,
            Ea0,
            Ea1,
            Eb0,
            Eb1);

        T E = friction_energy(mu, f, eps_vh, tan_rel_dx);

        return E;
    }

    template <typename T>
    inline void EE_friction_gradient_hessian(LVec<T, 12>&      G,
                                             LMat<T, 12, 12>&  H,
                                             const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const T&          mu,
                                             const T&          eps_vh,
                                             const LVec<T, 3>& prev_Ea0,
                                             const LVec<T, 3>& prev_Ea1,
                                             const LVec<T, 3>& prev_Eb0,
                                             const LVec<T, 3>& prev_Eb1,
                                             const LVec<T, 3>& Ea0,
                                             const LVec<T, 3>& Ea1,
                                             const LVec<T, 3>& Eb0,
                                             const LVec<T, 3>& Eb1)
    {
        T             f;
        LVec<T, 2>    gamma;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        EE_friction_basis(
            // out
            f,
            gamma,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_Ea0,
            prev_Ea1,
            prev_Eb0,
            prev_Eb1,
            Ea0,
            Ea1,
            Eb0,
            Eb1);

        LMat<T, 2, 12> J;
        friction::edge_edge_jacobi(basis, gamma, J);

        LVec<T, 2> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        //apply_edge_edge_jacobi(G2, basis, gamma, G);
        G = J.transpose() * G2;

        LMat<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline void EE_friction_gradient(LVec<T, 12>&      G,
                                     const T&          kappa,
                                     const T&          d_hat,
                                     const T&          thickness,
                                     const T&          mu,
                                     const T&          eps_vh,
                                     const LVec<T, 3>& prev_Ea0,
                                     const LVec<T, 3>& prev_Ea1,
                                     const LVec<T, 3>& prev_Eb0,
                                     const LVec<T, 3>& prev_Eb1,
                                     const LVec<T, 3>& Ea0,
                                     const LVec<T, 3>& Ea1,
                                     const LVec<T, 3>& Eb0,
                                     const LVec<T, 3>& Eb1)
    {
        T             f;
        LVec<T, 2>    gamma;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        EE_friction_basis(
            // out
            f,
            gamma,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_Ea0,
            prev_Ea1,
            prev_Eb0,
            prev_Eb1,
            Ea0,
            Ea1,
            Eb0,
            Eb1);

        LMat<T, 2, 12> J;
        friction::edge_edge_jacobi(basis, gamma, J);

        LVec<T, 2> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;
    }

    // ---------------------------------------------------------------------------
    //  Point - Edge
    // ---------------------------------------------------------------------------
    template <typename T>
    inline void PE_friction_basis(
        // out
        T&                f,
        T&                eta,
        LMat<T, 3, 2>&    basis,
        LVec<T, 2>&       tan_rel_dx,
        // in
        const T&          kappa,
        const T&          d_hat,
        const T&          thickness,
        const LVec<T, 3>& prev_P,
        const LVec<T, 3>& prev_E0,
        const LVec<T, 3>& prev_E1,
        const LVec<T, 3>& P,
        const LVec<T, 3>& E0,
        const LVec<T, 3>& E1)
    {
        // using the prev values to compute normal force
        friction::point_edge_closest_point(prev_P, prev_E0, prev_E1, eta);
        friction::point_edge_tangent_basis(prev_P, prev_E0, prev_E1, basis);

        T prev_D;
        ldist::point_edge_distance2(prev_P, prev_E0, prev_E1, prev_D);

        f = normal_force(kappa, d_hat, thickness, prev_D);

        LVec<T, 3> dP  = P - prev_P;
        LVec<T, 3> dE0 = E0 - prev_E0;
        LVec<T, 3> dE1 = E1 - prev_E1;

        friction::point_edge_tan_rel_dx(dP, dE0, dE1, basis, eta, tan_rel_dx);
    }

    template <typename T>
    [[nodiscard]] inline T PE_friction_energy(const T&          kappa,
                                              const T&          d_hat,
                                              const T&          thickness,
                                              const T&          mu,
                                              const T&          eps_vh,
                                              const LVec<T, 3>& prev_P,
                                              const LVec<T, 3>& prev_E0,
                                              const LVec<T, 3>& prev_E1,
                                              const LVec<T, 3>& P,
                                              const LVec<T, 3>& E0,
                                              const LVec<T, 3>& E1)
    {
        T             f;
        T             eta;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PE_friction_basis(
            // out
            f,
            eta,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P,
            prev_E0,
            prev_E1,
            P,
            E0,
            E1);

        T E = friction_energy(mu, f, eps_vh, tan_rel_dx);

        return E;
    }

    template <typename T>
    inline void PE_friction_gradient_hessian(LVec<T, 9>&       G,
                                             LMat<T, 9, 9>&    H,
                                             const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const T&          mu,
                                             const T&          eps_vh,
                                             const LVec<T, 3>& prev_P,
                                             const LVec<T, 3>& prev_E0,
                                             const LVec<T, 3>& prev_E1,
                                             const LVec<T, 3>& P,
                                             const LVec<T, 3>& E0,
                                             const LVec<T, 3>& E1)
    {
        T             f;
        T             eta;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PE_friction_basis(
            // out
            f,
            eta,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P,
            prev_E0,
            prev_E1,
            P,
            E0,
            E1);

        LMat<T, 2, 9> J;
        friction::point_edge_jacobi(basis, eta, J);

        LVec<T, 2> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        //apply_point_edge_jacobi(G2, basis, eta, G);
        G = J.transpose() * G2;

        LMat<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline void PE_friction_gradient(LVec<T, 9>&       G,
                                     const T&          kappa,
                                     const T&          d_hat,
                                     const T&          thickness,
                                     const T&          mu,
                                     const T&          eps_vh,
                                     const LVec<T, 3>& prev_P,
                                     const LVec<T, 3>& prev_E0,
                                     const LVec<T, 3>& prev_E1,
                                     const LVec<T, 3>& P,
                                     const LVec<T, 3>& E0,
                                     const LVec<T, 3>& E1)
    {
        T             f;
        T             eta;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PE_friction_basis(
            // out
            f,
            eta,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P,
            prev_E0,
            prev_E1,
            P,
            E0,
            E1);

        LMat<T, 2, 9> J;
        friction::point_edge_jacobi(basis, eta, J);

        LVec<T, 2> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;
    }

    // ---------------------------------------------------------------------------
    //  Point - Point
    // ---------------------------------------------------------------------------
    template <typename T>
    inline void PP_friction_basis(
        // out
        T&                f,
        LMat<T, 3, 2>&    basis,
        LVec<T, 2>&       tan_rel_dx,
        // in
        const T&          kappa,
        const T&          d_hat,
        const T&          thickness,
        const LVec<T, 3>& prev_P0,
        const LVec<T, 3>& prev_P1,
        const LVec<T, 3>& P0,
        const LVec<T, 3>& P1)
    {
        // using the prev values to compute normal force
        friction::point_point_tangent_basis(prev_P0, prev_P1, basis);

        T prev_D;
        ldist::point_point_distance2(prev_P0, prev_P1, prev_D);

        f = normal_force(kappa, d_hat, thickness, prev_D);

        LVec<T, 3> dP0 = P0 - prev_P0;
        LVec<T, 3> dP1 = P1 - prev_P1;

        friction::point_point_tan_rel_dx(dP0, dP1, basis, tan_rel_dx);
    }

    template <typename T>
    [[nodiscard]] inline T PP_friction_energy(const T&          kappa,
                                              const T&          d_hat,
                                              const T&          thickness,
                                              const T&          mu,
                                              const T&          eps_vh,
                                              const LVec<T, 3>& prev_P0,
                                              const LVec<T, 3>& prev_P1,
                                              const LVec<T, 3>& P0,
                                              const LVec<T, 3>& P1)
    {
        T             f;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PP_friction_basis(
            // out
            f,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P0,
            prev_P1,
            P0,
            P1);

        T E = friction_energy(mu, f, eps_vh, tan_rel_dx);

        return E;
    }

    template <typename T>
    inline void PP_friction_gradient_hessian(LVec<T, 6>&       G,
                                             LMat<T, 6, 6>&    H,
                                             const T&          kappa,
                                             const T&          d_hat,
                                             const T&          thickness,
                                             const T&          mu,
                                             const T&          eps_vh,
                                             const LVec<T, 3>& prev_P0,
                                             const LVec<T, 3>& prev_P1,
                                             const LVec<T, 3>& P0,
                                             const LVec<T, 3>& P1)
    {
        T             f;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PP_friction_basis(
            // out
            f,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P0,
            prev_P1,
            P0,
            P1);

        LMat<T, 2, 6> J;
        friction::point_point_jacobi(basis, J);

        LVec<T, 2> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        //apply_point_point_jacobi(G2, basis, G);
        G = J.transpose() * G2;

        LMat<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline void PP_friction_gradient(LVec<T, 6>&       G,
                                     const T&          kappa,
                                     const T&          d_hat,
                                     const T&          thickness,
                                     const T&          mu,
                                     const T&          eps_vh,
                                     const LVec<T, 3>& prev_P0,
                                     const LVec<T, 3>& prev_P1,
                                     const LVec<T, 3>& P0,
                                     const LVec<T, 3>& P1)
    {
        T             f;
        LMat<T, 3, 2> basis;
        LVec<T, 2>    tan_rel_dx;

        PP_friction_basis(
            // out
            f,
            basis,
            tan_rel_dx,
            // in
            kappa,
            d_hat,
            thickness,
            prev_P0,
            prev_P1,
            P0,
            P1);

        LMat<T, 2, 6> J;
        friction::point_point_jacobi(basis, J);

        LVec<T, 2> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;
    }
}  // namespace sym::codim_ipc_contact
}  // namespace uipc::backend::luisa
