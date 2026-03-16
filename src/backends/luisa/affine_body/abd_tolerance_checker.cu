#include <newton_tolerance/newton_tolerance_checker.h>
#include <affine_body/affine_body_dynamics.h>
#include <luisa/dsl/dsl.h>

namespace uipc::backend::luisa
{
class ABDToleranceChecker final : public NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
    Float                             abs_tol = 0.0;
    luisa::compute::Buffer<IndexT>    success;
    IndexT h_success = 1;  // 1 means success, 0 means failure

    // Inherited via NewtonToleranceChecker
    void do_build(BuildInfo& info) override
    {
        affine_body_dynamics     = require<AffineBodyDynamics>();
        auto& config             = world().scene().config();
        auto  dt_attr            = config.find<Float>("dt");
        Float dt                 = dt_attr->view()[0];
        auto  transrate_tol_attr = config.find<Float>("newton/transrate_tol");
        Float transrate_tol      = transrate_tol_attr->view()[0];
        abs_tol                  = transrate_tol * dt;

        // Create success flag buffer
        success = device().create_buffer<IndexT>(1);
    }

    void do_init(InitInfo& info) override {}

    void do_pre_newton(PreNewtonInfo& info) override {}

    void do_check(CheckResultInfo& info) override
    {
        using namespace luisa::compute;

        auto dqs = affine_body_dynamics->dqs();

        // Reset success flag to 1
        Kernel1D reset_kernel = [&](BufferVar<IndexT> success_buf) noexcept
        {
            success_buf.write(0, 1);
        };
        auto reset_shader = device().compile(reset_kernel);
        stream() << reset_shader(success.view()).dispatch(1);

        // Check tolerance for each body
        Kernel1D check_kernel = [&](BufferVar<const Vector12> dqs_buf,
                                     BufferVar<IndexT> success_buf,
                                     Float tol) noexcept
        {
            auto I = dispatch_id().x;
            $if(I < dqs_buf.size())
            {
                Vector12 dq = dqs_buf.read(I);
                IndexT success_value = success_buf.read(0);

                // if success is already marked as failed, skip
                $if(success_value != 0)
                {
                    // the first 3 components are translation, ignore
                    // the rest 9 components are rotation/scaling/shear, take
                    $for(i, 3, 12)
                    {
                        $if(abs(dq[i]) > tol)
                        {
                            success_buf.write(0, 0);
                            $break;
                        };
                    };
                };
            };
        };

        auto check_shader = device().compile(check_kernel);
        stream() << check_shader(dqs,
                                  success.view(),
                                  abs_tol).dispatch(dqs.size());

        // copy from device to host
        std::vector<IndexT> h_success_vec(1);
        stream() << success.copy_to(h_success_vec.data())
                 << synchronize();
        h_success = h_success_vec[0];
        info.converged(h_success != 0);
    }

    std::string do_report() override
    {
        return fmt::format("Tol: {}{}", (h_success ? "< " : "> "), abs_tol);
    }
};

REGISTER_SIM_SYSTEM(ABDToleranceChecker);
}  // namespace uipc::backend::luisa
