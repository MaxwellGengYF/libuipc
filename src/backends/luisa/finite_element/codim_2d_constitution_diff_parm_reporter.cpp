#include <finite_element/codim_2d_constitution_diff_parm_reporter.h>

namespace uipc::backend::luisa
{
void Codim2DConstitutionDiffParmReporter::do_assemble(Base::DiffParmInfo& info)
{
    DiffParmInfo this_info(this, info);
    do_assemble(this_info);
}

IndexT Codim2DConstitutionDiffParmReporter::get_dim() const noexcept
{
    return 2;
}

luisa::compute::BufferView<const luisa::float3> Codim2DConstitutionDiffParmReporter::DiffParmInfo::xs() const noexcept
{
    return m_impl->fem().xs.view();  // must return full buffer, because the indices index into the full buffer
}

luisa::compute::BufferView<const luisa::float3> Codim2DConstitutionDiffParmReporter::DiffParmInfo::x_bars() const noexcept
{
    return m_impl->fem().x_bars.view();  // must return full buffer, because the indices index into the full buffer
}

luisa::compute::BufferView<const float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::rest_areas() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().rest_areas.view(info.primitive_offset, info.primitive_count);
}

luisa::compute::BufferView<const float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::thicknesses() const noexcept
{
    return m_impl->fem().thicknesses.view();
}

luisa::compute::BufferView<const luisa::int3> Codim2DConstitutionDiffParmReporter::DiffParmInfo::indices() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().codim_2ds.view(info.primitive_offset, info.primitive_count);
}

luisa::compute::BufferView<const int> Codim2DConstitutionDiffParmReporter::DiffParmInfo::is_fixed() const noexcept
{
    return m_impl->fem().is_fixed.view();  // must return full buffer, because the indices index into the full buffer
}

const FiniteElementMethod::ConstitutionInfo& Codim2DConstitutionDiffParmReporter::DiffParmInfo::constitution_info() const noexcept
{
    return m_impl->constitution_info();
}

SizeT Codim2DConstitutionDiffParmReporter::DiffParmInfo::frame() const
{
    return m_diff_parm_info.frame();
}

IndexT Codim2DConstitutionDiffParmReporter::DiffParmInfo::dof_offset(SizeT frame) const
{
    return m_diff_parm_info.dof_offset(frame);
}

IndexT Codim2DConstitutionDiffParmReporter::DiffParmInfo::dof_count(SizeT frame) const
{
    return m_diff_parm_info.dof_count(frame);
}

luisa::span<const int> Codim2DConstitutionDiffParmReporter::DiffParmInfo::pGpP_rows() const
{
    auto pGpP = m_diff_parm_info.pGpP();
    auto rows = pGpP.row_indices_view();
    return luisa::span<const int>(rows.data(), rows.size());
}

luisa::span<const int> Codim2DConstitutionDiffParmReporter::DiffParmInfo::pGpP_cols() const
{
    auto pGpP = m_diff_parm_info.pGpP();
    auto cols = pGpP.col_indices_view();
    return luisa::span<const int>(cols.data(), cols.size());
}

luisa::span<const float> Codim2DConstitutionDiffParmReporter::DiffParmInfo::pGpP_values() const
{
    auto pGpP = m_diff_parm_info.pGpP();
    auto vals = pGpP.values_view();
    // For block size 1, each value is a 1x1 Eigen matrix, so we get the scalar
    return luisa::span<const float>(reinterpret_cast<const float*>(vals.data()), vals.size());
}

float Codim2DConstitutionDiffParmReporter::DiffParmInfo::dt() const
{
    return m_diff_parm_info.dt();
}
}  // namespace uipc::backend::luisa
