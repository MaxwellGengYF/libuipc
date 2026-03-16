#include <finite_element/codim_2d_constitution.h>

namespace uipc::backend::luisa
{
class Empty2D final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 2ull;

    using Codim2DConstitution::Codim2DConstitution;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(0);
        info.gradient_count(0);
        info.hessian_count(0);
    }

    virtual void do_build(BuildInfo& info) override
    {
        // do nothing
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        // do nothing
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        // do nothing
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        // do nothing
    }
};

REGISTER_SIM_SYSTEM(Empty2D);
}  // namespace uipc::backend::luisa
