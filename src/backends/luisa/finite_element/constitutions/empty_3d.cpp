/********************************************************************
 * @file empty_3d.cpp
 * @brief T5.2 (batch c) — Empty3D, the "no constitution" placeholder for 3d
 *        geometries (uid 0).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/empty_3d.cu.
 * See `empty_0d.cpp` for why the uid-0 placeholders are load-bearing and
 * `empty_1d.cpp` for the two spelling deviations (`override`, dropped stray includes:
 * the reference includes hookean_spring_1d_function.h / kernel_cout.h / cuda_tool /
 * Eigen, none of which its body uses — note it does *not* include codim_thickness.h,
 * which the 1d/2d siblings do).
 ********************************************************************/
#include <finite_element/fem_3d_constitution.h>

namespace uipc::backend::luisa
{
class Empty3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 0ull;

    using FEM3DConstitution::FEM3DConstitution;

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

REGISTER_SIM_SYSTEM(Empty3D);
}  // namespace uipc::backend::luisa
