#include <time_integrator/bdf1_flag.h>
#include <finite_element/finite_element_kinetic.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class FiniteElementBDF1Kinetic final : public FiniteElementKinetic
{
  public:
    using FiniteElementKinetic::FiniteElementKinetic;

    virtual void do_build(BuildInfo& info) override
    {
        // require BDF1 integration flag
        require<BDF1Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        auto is_fixed = info.is_fixed();
        auto xs       = info.xs();
        auto x_tildes = info.x_tildes();
        auto masses   = info.masses();
        auto energies = info.energies();

        Kernel1D compute_energy_kernel = [&](BufferVar<int> is_fixed_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Vector3> x_tildes_buf,
                                             BufferVar<Float> masses_buf,
                                             BufferVar<Float> energies_buf) noexcept
        {
            auto i = dispatch_x();
            $if(i < xs_buf.size())
            {
                $if(is_fixed_buf.read(i) != 0)
                {
                    energies_buf.write(i, 0.0);
                }
                $else
                {
                    Vector3 x       = xs_buf.read(i);
                    Vector3 x_tilde = x_tildes_buf.read(i);
                    Float   M       = masses_buf.read(i);
                    Vector3 dx      = x - x_tilde;
                    Float   K       = 0.5 * M * dot(dx, dx);
                    energies_buf.write(i, K);
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(is_fixed, xs, x_tildes, masses, energies).dispatch(xs.size());
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        auto is_fixed     = info.is_fixed();
        auto xs           = info.xs();
        auto x_tildes     = info.x_tildes();
        auto masses       = info.masses();
        auto gradients    = info.gradients();
        auto hessians     = info.hessians();
        Bool gradient_only = info.gradient_only();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<int> is_fixed_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Vector3> x_tildes_buf,
                                                BufferVar<Float> masses_buf,
                                                BufferVar<Float3> G3s_buf,
                                                BufferVar<Matrix3x3> H3x3s_buf,
                                                Bool gradient_only_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < xs_buf.size())
            {
                Float   m       = masses_buf.read(i);
                Vector3 x       = xs_buf.read(i);
                Vector3 x_tilde = x_tildes_buf.read(i);

                Vector3 G;

                $if(is_fixed_buf.read(i) != 0)
                {
                    G = Vector3::Zero();
                }
                $else
                {
                    G = m * (x - x_tilde);
                };

                // Write gradient: G3s(i).write(i, G)
                G3s_buf.atomic(i).x.fetch_add(G.x);
                G3s_buf.atomic(i).y.fetch_add(G.y);
                G3s_buf.atomic(i).z.fetch_add(G.z);

                $if(!gradient_only_val)
                {
                    Matrix3x3 H = m * Matrix3x3::Identity();
                    // Write hessian: H3x3s(i).write(i, i, H)
                    // For diagonal block at (i, i)
                    H3x3s_buf.atomic(i * 3 + 0)[0].x.fetch_add(H(0, 0));
                    H3x3s_buf.atomic(i * 3 + 0)[0].y.fetch_add(H(1, 0));
                    H3x3s_buf.atomic(i * 3 + 0)[0].z.fetch_add(H(2, 0));
                    H3x3s_buf.atomic(i * 3 + 1)[1].x.fetch_add(H(0, 1));
                    H3x3s_buf.atomic(i * 3 + 1)[1].y.fetch_add(H(1, 1));
                    H3x3s_buf.atomic(i * 3 + 1)[1].z.fetch_add(H(2, 1));
                    H3x3s_buf.atomic(i * 3 + 2)[2].x.fetch_add(H(0, 2));
                    H3x3s_buf.atomic(i * 3 + 2)[2].y.fetch_add(H(1, 2));
                    H3x3s_buf.atomic(i * 3 + 2)[2].z.fetch_add(H(2, 2));
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_grad_hess_kernel);
        engine().stream() << shader(is_fixed,
                                    xs,
                                    x_tildes,
                                    masses,
                                    gradients.buffer(),
                                    hessians.buffer(),
                                    gradient_only)
                                .dispatch(xs.size());
    }
};

REGISTER_SIM_SYSTEM(FiniteElementBDF1Kinetic);
}  // namespace uipc::backend::luisa
