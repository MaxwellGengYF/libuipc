#pragma once
#include <luisa/luisa-compute.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>

namespace uipc::backend::luisa
{
namespace sym::ipc_vertex_half_contact
{
#include "sym/vertex_half_plane_distance.inl"

    // PH_barrier_energy: Returns Float (by value), so no Var<> wrapper needed for return type
    inline Float PH_barrier_energy(Float          kappa,
                                   Float          d_hat,
                                   Float          thickness,
                                   const Float3&  v,
                                   const Float3&  P,
                                   const Float3&  N)
    {
        using namespace codim_ipc_contact;

        Float D;
        HalfPlaneD(D, v, P, N);
        Float E = 0.0f;
        KappaBarrier(E, kappa, D, d_hat, thickness);

        return E;
    }

    inline void PH_barrier_gradient_hessian(Float3&        G,
                                            Float3x3&      H,
                                            Float          kappa,
                                            Float          d_hat,
                                            Float          thickness,
                                            const Float3&  v,
                                            const Float3&  P,
                                            const Float3&  N)
    {
        using namespace codim_ipc_contact;

        Float D;
        HalfPlaneD(D, v, P, N);

        Float dBdD = 0.0f;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Float3 dDdx;
        dHalfPlaneDdx(dDdx, v, P, N);

        G = dBdD * dDdx;

        Float ddBddD = 0.0f;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Float3x3 ddDddx;
        ddHalfPlaneDddx(ddDddx, v, P, N);

        // H = ddBddD * dDdx * dDdx.transpose() + dBdD * ddDddx
        H = ddBddD * luisa::outer_product(dDdx, dDdx) + dBdD * ddDddx;
    }

    inline void PH_barrier_gradient(Float3&        G,
                                    Float          kappa,
                                    Float          d_hat,
                                    Float          thickness,
                                    const Float3&  v,
                                    const Float3&  P,
                                    const Float3&  N)
    {
        using namespace codim_ipc_contact;

        Float D;
        HalfPlaneD(D, v, P, N);

        Float dBdD = 0.0f;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Float3 dDdx;
        dHalfPlaneDdx(dDdx, v, P, N);

        G = dBdD * dDdx;
    }

    inline void compute_tan_basis(Float3& e1, Float3& e2, const Float3& N)
    {
        using namespace codim_ipc_contact;

        Float3 trial = make_float3(1.0f, 0.0f, 0.0f);  // UnitX
        if(dot(N, trial) > 0.9f)
        {
            trial = make_float3(0.0f, 0.0f, 1.0f);  // UnitZ
            e1    = normalize(cross(trial, N));
        }
        else
        {
            e1 = normalize(cross(trial, N));
        }
        e2 = cross(N, e1);
    }


    inline Float PH_friction_energy(Float          kappa,
                                    Float          d_hat,
                                    Float          thickness,
                                    Float          mu,
                                    Float          eps_vh,
                                    const Float3&  prev_v,
                                    const Float3&  v,
                                    const Float3&  P,
                                    const Float3&  N)
    {
        using namespace codim_ipc_contact;

        Float prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        Float f = normal_force(kappa, d_hat, thickness, prev_D);

        Float3 dV = v - prev_v;


        Float3 e1, e2;
        compute_tan_basis(e1, e2, N);

        Float2 tan_dV;

        TR(tan_dV, v, prev_v, e1, e2);

        return friction_energy(mu, f, eps_vh, tan_dV);
    }

    inline void PH_friction_gradient_hessian(Float3&        G,
                                             Float3x3&      H,
                                             Float          kappa,
                                             Float          d_hat,
                                             Float          thickness,
                                             Float          mu,
                                             Float          eps_vh,
                                             const Float3&  prev_v,
                                             const Float3&  v,
                                             const Float3&  P,
                                             const Float3&  N)
    {
        using namespace codim_ipc_contact;

        Float prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        Float f = normal_force(kappa, d_hat, thickness, prev_D);

        Float3 dV = v - prev_v;

        Float3 e1, e2;
        compute_tan_basis(e1, e2, N);

        Float2 tan_dV;

        TR(tan_dV, v, prev_v, e1, e2);

        Float2 G2;
        friction_gradient(G2, mu, f, eps_vh, tan_dV);

        // Matrix<Float, 2, 3> J;
        Float2x3 J;
        dTRdx(J, v, prev_v, e1, e2);

        // G = J.transpose() * G2
        G = J[0] * G2.x + J[1] * G2.y;

        Float2x2 H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_dV);

        // H = J.transpose() * H2x2 * J
        // J is 3x2 (3 rows, 2 cols) in row-major view, but luisa stores as 2 cols x 3 rows
        // H = H2x2[0][0] * J[:,0] * J[:,0]^T + H2x2[0][1] * J[:,0] * J[:,1]^T
        //   + H2x2[1][0] * J[:,1] * J[:,0]^T + H2x2[1][1] * J[:,1] * J[:,1]^T
        H = H2x2[0][0] * luisa::outer_product(J[0], J[0])
          + H2x2[0][1] * luisa::outer_product(J[0], J[1])
          + H2x2[1][0] * luisa::outer_product(J[1], J[0])
          + H2x2[1][1] * luisa::outer_product(J[1], J[1]);
    }

    inline void PH_friction_gradient(Float3&        G,
                                     Float          kappa,
                                     Float          d_hat,
                                     Float          thickness,
                                     Float          mu,
                                     Float          eps_vh,
                                     const Float3&  prev_v,
                                     const Float3&  v,
                                     const Float3&  P,
                                     const Float3&  N)
    {
        using namespace codim_ipc_contact;

        Float prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        Float f = normal_force(kappa, d_hat, thickness, prev_D);

        Float3 dV = v - prev_v;

        Float3 e1, e2;
        compute_tan_basis(e1, e2, N);

        Float2 tan_dV;
        TR(tan_dV, v, prev_v, e1, e2);

        Float2 G2;
        friction_gradient(G2, mu, f, eps_vh, tan_dV);

        // Matrix<Float, 2, 3> J;
        Float2x3 J;
        dTRdx(J, v, prev_v, e1, e2);

        // G = J.transpose() * G2
        G = J[0] * G2.x + J[1] * G2.y;
    }
}  // namespace sym::ipc_vertex_half_contact
}  // namespace uipc::backend::luisa
