#include <finite_element/finite_element_elastics.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/fem_line_search_subreporter.h>
#include <utils/report_extent_check.h>

/*************************************************************************************************
* See the scope note in finite_element_elastics.h: T5.1 ports the system, its
* extent bookkeeping and the subview accessors (which FiniteElementMethod::init()
* needs and which the constitution bases alias); T5.3 adds the *energy half* of the
* reference's two fan-outs (`Impl::compute_energy` and the
* `FiniteElementElasticsLineSearchSubreporter` that drives it).  T5.4 adds
* `Impl::assemble()` and `FiniteElementElasticsLinearSubsystemReporter`.
*************************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(FiniteElementElastics);

void FiniteElementElastics::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}

void FiniteElementElastics::do_build()
{
    m_impl.finite_element_method = require<FiniteElementMethod>();
    m_impl.finite_element_method->add_elastics(this);
}

void FiniteElementElastics::init()
{
    auto  constitutions = m_impl.fem().constitutions.view();
    auto  extras        = m_impl.fem().extra_constitutions.view();
    SizeT N             = constitutions.size() + extras.size();

    m_impl.constitution_energy_offsets_counts.resize(N);
    m_impl.constitution_gradient_offsets_counts.resize(N);
    m_impl.constitution_hessian_offsets_counts.resize(N);

    auto energy_span = m_impl.constitution_energy_offsets_counts.counts();
    auto grad_span   = m_impl.constitution_gradient_offsets_counts.counts();
    auto hess_span   = m_impl.constitution_hessian_offsets_counts.counts();

    auto offset = 0;
    for(auto&& [I, c] : enumerate(constitutions))
    {
        ReportExtentInfo this_info;
        c->report_extent(this_info);
        energy_span[I] = this_info.m_energy_count;
        grad_span[I]   = this_info.m_gradient_count;
        hess_span[I]   = this_info.m_hessian_count;
    }

    offset += constitutions.size();
    for(auto&& [I, c] : enumerate(extras))
    {
        ReportExtentInfo this_info;
        c->report_extent(this_info);
        energy_span[offset + I] = this_info.m_energy_count;
        grad_span[offset + I]   = this_info.m_gradient_count;
        hess_span[offset + I]   = this_info.m_hessian_count;
    }
    offset += extras.size();

    UIPC_ASSERT(offset == N, "offset mismatching, offset = {}, expected = {}", offset, N);

    m_impl.constitution_energy_offsets_counts.scan();
    m_impl.constitution_gradient_offsets_counts.scan();
    m_impl.constitution_hessian_offsets_counts.scan();
}

lla::LDoubletVectorView<Float, 3> FiniteElementElastics::ComputeGradientHessianInfo::gradients() const noexcept
{
    auto [offset, count] = m_impl->constitution_gradient_offsets_counts[m_index];
    return m_gradients.subview(offset, count);
}

lla::LTripletMatrixView<Float, 3, 3> FiniteElementElastics::ComputeGradientHessianInfo::hessians() const noexcept
{
    if(m_gradient_only)
        return lla::LTripletMatrixView<Float, 3, 3>{};

    auto [offset, count] = m_impl->constitution_hessian_offsets_counts[m_index];
    return m_hessians.subview(offset, count);
}

// ---------------------------------------------------------------------------
// T5.3: the energy half of the reference's two fan-outs.
// `Impl::assemble(...)` and `FiniteElementElasticsLinearSubsystemReporter` below
// are T5.4's -- they need `FEMLinearSubsystem::AssembleInfo`'s accessor
// definitions and the linear subsystem's Impl members, which T5.4 owns.
// ---------------------------------------------------------------------------
void FiniteElementElastics::Impl::compute_energy(FEMLineSearchReporter::ComputeEnergyInfo& info)
{
    auto constitutions = fem().constitutions.view();
    auto extras        = fem().extra_constitutions.view();
    auto offset        = 0;
    for(auto&& [I, c] : enumerate(constitutions))
    {
        ComputeEnergyInfo this_info{this, I, info.dt(), info.energies()};
        c->compute_energy(this_info);
    }
    offset += constitutions.size();
    for(auto&& [I, c] : enumerate(extras))
    {
        ComputeEnergyInfo this_info{this, offset + I, info.dt(), info.energies()};
        c->compute_energy(this_info);
    }
    offset += extras.size();

    UIPC_ASSERT(offset == constitutions.size() + extras.size(),
                "offset mismatching, offset = {}, expected = {}",
                offset,
                constitutions.size() + extras.size());
}

class FiniteElementElasticsLineSearchSubreporter final : public FEMLineSearchSubreporter
{
  public:
    using FEMLineSearchSubreporter::FEMLineSearchSubreporter;

    SimSystemSlot<FiniteElementElastics> elastics;

    FiniteElementElastics::Impl& el() { return elastics->m_impl; }

    void do_build(BuildInfo& info) override
    {
        (void)info;
        elastics = require<FiniteElementElastics>();
    }

    void do_init(InitInfo& info) override { (void)info; }

    void do_report_extent(ReportExtentInfo& info) override
    {
        auto energy_count = el().constitution_energy_offsets_counts.total_count();
        info.energy_count(energy_count);
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        el().compute_energy(info);
    }
};

REGISTER_SIM_SYSTEM(FiniteElementElasticsLineSearchSubreporter);

// ---------------------------------------------------------------------------
// T5.4 continues here: `FiniteElementElasticsLinearSubsystemReporter`
// (reporting the constitution gradient/Hessian extents and calling
// `el().assemble(info)`) and `FiniteElementElastics::Impl::assemble(...)`.
// ---------------------------------------------------------------------------
}  // namespace uipc::backend::luisa
