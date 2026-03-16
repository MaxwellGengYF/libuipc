#include <affine_body/abd_time_integrator.h>
#include <time_integrator/bdf1_flag.h>
#include <sim_engine.h>
#include <luisa/dsl/dsl.h>

namespace uipc::backend::luisa
{
/**
 * @brief BDF1 time integrator for Affine Body Dynamics (ABD)
 * 
 * Implements the first-order Backward Differentiation Formula (BDF1) for time integration
 * of affine body dynamics. This is an implicit integrator that is unconditionally stable.
 * 
 * The prediction formula is:
 * - Fixed bodies: q_tilde = q_prev
 * - Static (kinematic) bodies: q_tilde = q_prev + (g + f_ext) * dt^2
 * - Dynamic bodies: q_tilde = q_prev + q_v * dt + (g + f_ext) * dt^2
 * 
 * Velocity update:
 * - q_v = (q - q_prev) / dt
 */
class ABDBDF1Integrator final : public ABDTimeIntegrator
{
  public:
    using ABDTimeIntegrator::ABDTimeIntegrator;

    void do_build(BuildInfo& info) override
    {
        // require the BDF1 flag
        require<BDF1Flag>();
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_predict_dof(PredictDofInfo& info) override
    {
        using namespace luisa::compute;

        auto& engine = static_cast<SimEngine&>(world().sim_engine());
        auto& device = engine.device();
        auto  stream = engine.compute_stream();

        // Get buffer views
        auto is_fixed_view = info.is_fixed();
        auto is_dynamic_view = info.is_dynamic();
        auto qs_view = info.qs();
        auto q_prevs_view = info.q_prevs();
        auto q_vs_view = info.q_vs();
        auto q_tildes_view = info.q_tildes();
        auto gravities_view = info.gravities();
        auto external_force_accs_view = info.external_force_accs();
        
        SizeT count = qs_view.size();
        Float dt = info.dt();

        // Kernel for predicting DOF values
        Kernel1D predict_kernel = [&](BufferVar<const IndexT> is_fixed,
                                       BufferVar<const IndexT> is_dynamic,
                                       BufferVar<const Vector12> qs,
                                       BufferVar<Vector12> q_prevs,
                                       BufferVar<const Vector12> q_vs,
                                       BufferVar<Vector12> q_tildes,
                                       BufferVar<const Vector12> gravities,
                                       BufferVar<const Vector12> external_force_accs,
                                       Float dt_val) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < count)
            {
                // Read current q and record as previous
                Vector12 q = qs.read(i);
                q_prevs.write(i, q);

                // Read velocity, gravity, and external force
                Vector12 q_v = q_vs.read(i);
                Vector12 g = gravities.read(i);
                Vector12 f_ext_acc = external_force_accs.read(i);

                // Default: q_tilde = q_prev (for fixed bodies)
                Vector12 q_tilde = q;

                $if(is_fixed.read(i) == 0)
                {
                    // Non-fixed body: add gravity and external force contribution
                    // q_tilde = q_prev + (g + f_ext_acc) * dt^2
                    for(int k = 0; k < 12; ++k)
                    {
                        q_tilde[k] = q[k] + (g[k] + f_ext_acc[k]) * dt_val * dt_val;
                    }

                    // Dynamic body: also add velocity contribution
                    // q_tilde += q_v * dt
                    $if(is_dynamic.read(i) != 0)
                    {
                        for(int k = 0; k < 12; ++k)
                        {
                            q_tilde[k] += q_v[k] * dt_val;
                        }
                    };
                };

                q_tildes.write(i, q_tilde);
            };
        };

        auto shader = device.compile(predict_kernel);
        stream << shader(is_fixed_view,
                         is_dynamic_view,
                         qs_view,
                         q_prevs_view,
                         q_vs_view,
                         q_tildes_view,
                         gravities_view,
                         external_force_accs_view,
                         dt).dispatch(count);
    }

    virtual void do_update_state(UpdateVelocityInfo& info) override
    {
        using namespace luisa::compute;

        auto& engine = static_cast<SimEngine&>(world().sim_engine());
        auto& device = engine.device();
        auto  stream = engine.compute_stream();

        // Get buffer views
        auto qs_view = info.qs();
        auto q_vs_view = info.q_vs();
        auto q_prevs_view = info.q_prevs();
        
        SizeT count = qs_view.size();
        Float dt = info.dt();
        Float inv_dt = 1.0 / dt;

        // Kernel for updating velocities
        Kernel1D update_velocity_kernel = [&](BufferVar<const Vector12> qs,
                                               BufferVar<Vector12> q_vs,
                                               BufferVar<const Vector12> q_prevs,
                                               Float inv_dt_val) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < count)
            {
                Vector12 q = qs.read(i);
                Vector12 q_prev = q_prevs.read(i);
                Vector12 q_v;

                // q_v = (q - q_prev) / dt
                for(int k = 0; k < 12; ++k)
                {
                    q_v[k] = (q[k] - q_prev[k]) * inv_dt_val;
                }

                q_vs.write(i, q_v);
            };
        };

        auto shader = device.compile(update_velocity_kernel);
        stream << shader(qs_view,
                         q_vs_view,
                         q_prevs_view,
                         inv_dt).dispatch(count);
    }
};

REGISTER_SIM_SYSTEM(ABDBDF1Integrator);
}  // namespace uipc::backend::luisa
