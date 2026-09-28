#include <finite_element/codim_1d_constitution.h>

namespace uipc::backend::luisa
{
void Codim1DConstitution::do_build(FiniteElementConstitution::BuildInfo& info)
{
    Codim1DConstitution::BuildInfo this_info;
    do_build(this_info);
}

void Codim1DConstitution::do_compute_energy(FiniteElementConstitution::ComputeEnergyInfo& info)
{
    Codim1DConstitution::ComputeEnergyInfo this_info{
        this, m_index_in_dim, info.dt(), info.energies()};
    do_compute_energy(this_info);
}

void Codim1DConstitution::do_compute_gradient_hessian(FiniteElementConstitution::ComputeGradientHessianInfo& info)
{
    Codim1DConstitution::ComputeGradientHessianInfo this_info{
        this, m_index_in_dim, info.gradient_only(), info.dt(), info.gradients(), info.hessians()};
    do_compute_gradient_hessian(this_info);
}

IndexT Codim1DConstitution::get_dim() const noexcept
{
    return 1;
}

lud::LCBufferView<LVector3> Codim1DConstitution::BaseInfo::xs() const noexcept
{
    return m_impl->fem().xs.cview();  // must return full buffer, because the indices index into the full buffer
}

lud::LCBufferView<LVector3> Codim1DConstitution::BaseInfo::x_bars() const noexcept
{
    return m_impl->fem().x_bars.cview();  // must return full buffer, because the indices index into the full buffer
}

lud::LCBufferView<Float> Codim1DConstitution::BaseInfo::rest_lengths() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().rest_lengths.cview(info.primitive_offset, info.primitive_count);
}

lud::LCBufferView<Float> Codim1DConstitution::BaseInfo::thicknesses() const noexcept
{
    return m_impl->fem().thicknesses.cview();
}

lud::LCBufferView<LVector2i> Codim1DConstitution::BaseInfo::indices() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().codim_1ds.cview(info.primitive_offset, info.primitive_count);
}
lud::LCBufferView<IndexT> Codim1DConstitution::BaseInfo::is_fixed() const noexcept
{
    return m_impl->fem().is_fixed.cview();  // must return full buffer, because the indices index into the full buffer
}

const FiniteElementMethod::ConstitutionInfo& Codim1DConstitution::BaseInfo::constitution_info() const noexcept
{
    return m_impl->fem().codim_1d_constitution_infos[m_index_in_dim];
}

Float Codim1DConstitution::BaseInfo::dt() const noexcept
{
    return m_dt;
}
}  // namespace uipc::backend::luisa
