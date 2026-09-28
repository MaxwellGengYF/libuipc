#include <finite_element/finite_element_elastics.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <utils/report_extent_check.h>

/*************************************************************************************************
* See the scope note in finite_element_elastics.h: T5.1 ports the system, its
* extent bookkeeping and the subview accessors (which FiniteElementMethod::init()
* needs and which the constitution bases alias); T5.3 adds Impl::assemble() /
* Impl::compute_energy() and T5.4/T5.5 add the two nested reporter systems.
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
// T5.3 (Impl::assemble / Impl::compute_energy) and T5.4/T5.5 (the two nested
// reporter systems) continue here. They are *not* ported yet on purpose:
//   Impl::assemble(FEMLinearSubsystem::AssembleInfo&)      -- T5.4 owns the
//       parameter type; the body is the reference's fan-out over
//       `ComputeGradientHessianInfo`.
//   Impl::compute_energy(FEMLineSearchReporter::ComputeEnergyInfo&) -- T5.5.
// Both stay declared (and unused) until then, so nothing is dropped silently.
// ---------------------------------------------------------------------------
}  // namespace uipc::backend::luisa
