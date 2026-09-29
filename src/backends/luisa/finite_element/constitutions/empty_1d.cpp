/********************************************************************
 * @file empty_1d.cpp
 * @brief T5.2 (batch c) — Empty1D, the "no constitution" placeholder for 1d
 *        geometries (uid 0).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/empty_1d.cu.
 * See `empty_0d.cpp` for why the uid-0 placeholders are load-bearing.
 *
 * A direct port.  The reference declares `do_report_extent` publicly and without
 * `override`; the port keeps the public access and spells the `override` keyword
 * (semantically identical — the base's hook is a pure virtual).  It sets all three
 * extents to 0 *unconditionally*, which is legal: `ReportExtentInfo::check` only
 * forbids a *non-zero* Hessian count while `gradient_only` is set.
 ********************************************************************/
#include <finite_element/codim_1d_constitution.h>

namespace uipc::backend::luisa
{
class Empty1D final : public Codim1DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 0ull;

    using Codim1DConstitution::Codim1DConstitution;

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

REGISTER_SIM_SYSTEM(Empty1D);
}  // namespace uipc::backend::luisa
