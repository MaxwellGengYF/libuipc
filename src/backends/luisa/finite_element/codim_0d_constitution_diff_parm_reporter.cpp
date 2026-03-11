#include <finite_element/codim_0d_constitution_diff_parm_reporter.h>

namespace uipc::backend::luisa
{
void Codim0DConstitutionDiffParmReporter::do_assemble(Base::DiffParmInfo& info)
{
    DiffParmInfo this_info(this, info);
    do_assemble(this_info);
}

IndexT Codim0DConstitutionDiffParmReporter::get_dim() const noexcept
{
    return 0;
}

luisa::compute::BufferView<const luisa::float3> Codim0DConstitutionDiffParmReporter::DiffParmInfo::xs() const noexcept
{
    return m_impl->fem().xs.view();  // must return full buffer, because the indices index into the full buffer
}

luisa::compute::BufferView<const luisa::float3> Codim0DConstitutionDiffParmReporter::DiffParmInfo::x_bars() const noexcept
{
    return m_impl->fem().x_bars.view();  // must return full buffer, because the indices index into the full buffer
}

luisa::compute::BufferView<const int> Codim0DConstitutionDiffParmReporter::DiffParmInfo::indices() const noexcept
{
    auto& info = constitution_info();
    return m_impl->fem().codim_0ds.view(info.primitive_offset, info.primitive_count);
}

luisa::compute::BufferView<const float> Codim0DConstitutionDiffParmReporter::DiffParmInfo::thicknesses() const noexcept
{
    return m_impl->fem().thicknesses.view();
}

luisa::compute::BufferView<const int> Codim0DConstitutionDiffParmReporter::DiffParmInfo::is_fixed() const noexcept
{
    return m_impl->fem().is_fixed.view();  // must return full buffer, because the indices index into the full buffer
}

const FiniteElementMethod::ConstitutionInfo& Codim0DConstitutionDiffParmReporter::DiffParmInfo::constitution_info() const noexcept
{
    return m_impl->constitution_info();
}

SizeT Codim0DConstitutionDiffParmReporter::DiffParmInfo::frame() const
{
    return m_diff_parm_info.frame();
}

IndexT Codim0DConstitutionDiffParmReporter::DiffParmInfo::dof_offset(SizeT frame) const
{
    return m_diff_parm_info.dof_offset(frame);
}

IndexT Codim0DConstitutionDiffParmReporter::DiffParmInfo::dof_count(SizeT frame) const
{
    return m_diff_parm_info.dof_count(frame);
}

luisa::span<const int> Codim0DConstitutionDiffParmReporter::DiffParmInfo::pGpP_rows() const
{
    auto pGpP = m_diff_parm_info.pGpP();
    auto rows = pGpP.row_indices_view();
    return luisa::span<const int>(rows.data(), rows.size());
}

luisa::span<const int> Codim0DConstitutionDiffParmReporter::DiffParmInfo::pGpP_cols() const
{
    auto pGpP = m_diff_parm_info.pGpP();
    auto cols = pGpP.col_indices_view();
    return luisa::span<const int>(cols.data(), cols.size());
}

luisa::span<const float> Codim0DConstitutionDiffParmReporter::DiffParmInfo::pGpP_values() const
{
    auto pGpP = m_diff_parm_info.pGpP();
    auto vals = pGpP.values_view();
    // For block size 1, each value is a 1x1 Eigen matrix, so we get the scalar
    return luisa::span<const float>(reinterpret_cast<const float*>(vals.data()), vals.size());
}

float Codim0DConstitutionDiffParmReporter::DiffParmInfo::dt() const
{
    return m_diff_parm_info.dt();
}
}  // namespace uipc::backend::luisa
