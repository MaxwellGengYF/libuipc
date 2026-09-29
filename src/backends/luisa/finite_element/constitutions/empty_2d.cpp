/********************************************************************
 * @file empty_2d.cpp
 * @brief T5.2 (batch c) — Empty2D, the "no constitution" placeholder for 2d
 *        geometries (uid 0).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/empty_2d.cu.
 * See `empty_0d.cpp` for why the uid-0 placeholders are load-bearing and
 * `empty_1d.cpp` for the two spelling deviations (`override`, dropped stray includes:
 * the reference includes hookean_spring_1d_function.h / kernel_cout.h / cuda_tool /
 * Eigen / codim_thickness, none of which its body uses).
 ********************************************************************/
#include <finite_element/codim_2d_constitution.h>

namespace uipc::backend::luisa
{
class Empty2D final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 0ull;

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
        (void)info;
        // do nothing
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        (void)info;
        // do nothing
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        (void)info;
        // do nothing
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        (void)info;
        // do nothing
    }
};

REGISTER_SIM_SYSTEM(Empty2D);
}  // namespace uipc::backend::luisa
