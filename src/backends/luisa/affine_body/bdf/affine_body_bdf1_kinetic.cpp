#include <affine_body/affine_body_kinetic.h>
#include <time_integrator/bdf1_flag.h>
#include <sim_engine.h>
#include <luisa/dsl/dsl.h>
#include <kernel_cout.h>

namespace uipc::backend::luisa
{
/**
 * @brief BDF1 kinetic energy computation for Affine Body Dynamics
 * 
 * Computes the kinetic energy and its gradient/hessian for BDF1 time integration.
 * The kinetic energy is defined as:
 *   K = 0.5 * (q - q_tilde)^T * M * (q - q_tilde)
 * 
 * Gradient:
 *   G = M * (q - q_tilde)
 * 
 * Hessian:
 *   H = M
 */
class AffineBodyBDF1Kinetic final : public AffineBodyKinetic
{
  public:
    using AffineBodyKinetic::AffineBodyKinetic;

    virtual void do_build(BuildInfo& info) override
    {
        // need BDF1 flag for BDF1 time integration
        require<BDF1Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace luisa::compute;

        auto& engine = static_cast<SimEngine&>(world().sim_engine());
        auto& device = engine.device();
        auto  stream = engine.compute_stream();

        // Get buffer views
        auto is_fixed_view = info.is_fixed();
        auto ext_kinetic_view = info.external_kinetic();
        auto qs_view = info.qs();
        auto q_tildes_view = info.q_tildes();
        auto masses_view = info.masses();
        auto energies_view = info.energies();
        
        SizeT count = qs_view.size();

        // Kernel for computing kinetic energy
        Kernel1D compute_energy_kernel = [&](BufferVar<const IndexT> is_fixed,
                                              BufferVar<const IndexT> ext_kinetic,
                                              BufferVar<const Vector12> qs,
                                              BufferVar<const Vector12> q_tildes,
                                              BufferVar<const ABDJacobiDyadicMass> masses,
                                              BufferVar<Float> energies) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < count)
            {
                $if(is_fixed.read(i) != 0 || ext_kinetic.read(i) != 0)
                {
                    energies.write(i, 0.0);
                }
                $else
                {
                    Vector12 q = qs.read(i);
                    Vector12 q_tilde = q_tildes.read(i);
                    ABDJacobiDyadicMass M = masses.read(i);

                    // dq = q - q_tilde
                    Vector12 dq;
                    for(int k = 0; k < 12; ++k)
                    {
                        dq[k] = q[k] - q_tilde[k];
                    }

                    // K = 0.5 * dq.dot(M * dq)
                    Vector12 M_dq = M * dq;
                    Float K = 0.0;
                    for(int k = 0; k < 12; ++k)
                    {
                        K += dq[k] * M_dq[k];
                    }
                    K *= 0.5;

                    energies.write(i, K);
                };
            };
        };

        auto shader = device.compile(compute_energy_kernel);
        stream << shader(is_fixed_view,
                         ext_kinetic_view,
                         qs_view,
                         q_tildes_view,
                         masses_view,
                         energies_view).dispatch(count);
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace luisa::compute;

        auto& engine = static_cast<SimEngine&>(world().sim_engine());
        auto& device = engine.device();
        auto  stream = engine.compute_stream();

        // Get buffer views
        auto is_fixed_view = info.is_fixed();
        auto qs_view = info.qs();
        auto q_tildes_view = info.q_tildes();
        auto masses_view = info.masses();
        auto gradients_view = info.gradients();
        auto hessians_view = info.hessians();
        
        SizeT count = qs_view.size();
        bool gradient_only = info.gradient_only();

        // Kernel for computing gradient and hessian
        Kernel1D compute_gradient_hessian_kernel = [&](BufferVar<const IndexT> is_fixed,
                                                       BufferVar<const Vector12> qs,
                                                       BufferVar<const Vector12> q_tildes,
                                                       BufferVar<const ABDJacobiDyadicMass> masses,
                                                       BufferVar<Vector12> gradients,
                                                       BufferVar<Matrix12x12> hessians,
                                                       Bool grad_only) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < count)
            {
                Vector12 q = qs.read(i);
                Vector12 q_tilde = q_tildes.read(i);
                ABDJacobiDyadicMass M = masses.read(i);

                // G = M * (q - q_tilde)
                Vector12 dq;
                for(int k = 0; k < 12; ++k)
                {
                    dq[k] = q[k] - q_tilde[k];
                }
                Vector12 G = M * dq;

                // Fixed bodies have zero gradient
                $if(is_fixed.read(i) != 0)
                {
                    for(int k = 0; k < 12; ++k)
                    {
                        G[k] = 0.0;
                    }
                };

                gradients.write(i, G);

                $if(!grad_only)
                {
                    // H = M (as Matrix12x12)
                    Matrix12x12 H = M.to_mat();
                    hessians.write(i, H);
                };
            };
        };

        auto shader = device.compile(compute_gradient_hessian_kernel);
        stream << shader(is_fixed_view,
                         qs_view,
                         q_tildes_view,
                         masses_view,
                         gradients_view,
                         hessians_view,
                         gradient_only).dispatch(count);
    }
};

REGISTER_SIM_SYSTEM(AffineBodyBDF1Kinetic);
}  // namespace uipc::backend::luisa
