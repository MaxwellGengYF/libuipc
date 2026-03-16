#include <finite_element/fem_time_integrator.h>
#include <time_integrator/bdf1_flag.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class FEMBDF1Integrator final : public FEMTimeIntegrator
{
  public:
    using FEMTimeIntegrator::FEMTimeIntegrator;

    void do_build(BuildInfo& info) override
    {
        // require the BDF1 flag
        require<BDF1Flag>();
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_predict_dof(PredictDofInfo& info) override
    {
        auto is_fixed   = info.is_fixed();
        auto is_dynamic = info.is_dynamic();
        auto x_prevs    = info.x_prevs();
        auto xs         = info.xs();
        auto vs         = info.vs();
        auto x_tildes   = info.x_tildes();
        auto gravities  = info.gravities();
        Float dt        = info.dt();

        Kernel1D predict_dof_kernel = [&](BufferVar<Vector3> x_prevs_buf,
                                          BufferVar<Vector3> xs_buf,
                                          BufferVar<Vector3> vs_buf,
                                          BufferVar<Vector3> x_tildes_buf,
                                          BufferVar<Vector3> gravities_buf,
                                          BufferVar<int> is_fixed_buf,
                                          BufferVar<int> is_dynamic_buf,
                                          Float dt_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < xs_buf.size())
            {
                // record previous position
                Vector3 x_prev = xs_buf.read(i);
                x_prevs_buf.write(i, x_prev);

                Vector3 v = vs_buf.read(i);

                // 0) fixed: x_tilde = x_prev
                Vector3 x_tilde = x_prev;

                $if(is_fixed_buf.read(i) == 0)
                {
                    Vector3 g = gravities_buf.read(i);

                    // 1) static problem: x_tilde = x_prev + g * dt * dt
                    x_tilde = x_tilde + g * dt_val * dt_val;

                    // 2) dynamic problem: x_tilde = x_prev + v * dt + g * dt * dt
                    $if(is_dynamic_buf.read(i) != 0)
                    {
                        x_tilde = x_tilde + v * dt_val;
                    };
                };

                x_tildes_buf.write(i, x_tilde);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(predict_dof_kernel);
        engine().stream() << shader(x_prevs,
                                    xs,
                                    vs,
                                    x_tildes,
                                    gravities,
                                    is_fixed,
                                    is_dynamic,
                                    dt)
                                .dispatch(xs.size());
    }

    virtual void do_update_state(UpdateVelocityInfo& info) override
    {
        auto xs      = info.xs();
        auto vs      = info.vs();
        auto x_prevs = info.x_prevs();
        Float dt     = info.dt();

        Kernel1D update_state_kernel = [&](BufferVar<Vector3> xs_buf,
                                           BufferVar<Vector3> vs_buf,
                                           BufferVar<Vector3> x_prevs_buf,
                                           Float dt_val) noexcept
        {
            auto i = dispatch_x();
            $if(i < xs_buf.size())
            {
                Vector3 x      = xs_buf.read(i);
                Vector3 x_prev = x_prevs_buf.read(i);
                Vector3 v      = (x - x_prev) * (1.0 / dt_val);
                vs_buf.write(i, v);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(update_state_kernel);
        engine().stream() << shader(xs, vs, x_prevs, dt).dispatch(xs.size());
    }
};

REGISTER_SIM_SYSTEM(FEMBDF1Integrator);
}  // namespace uipc::backend::luisa
