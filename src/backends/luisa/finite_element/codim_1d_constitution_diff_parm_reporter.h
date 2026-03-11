#pragma once
#include <finite_element/finite_element_constitution_diff_parm_reporter.h>
#include <finite_element/finite_element_method.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
class Codim1DConstitutionDiffParmReporter : public FiniteElementConstitutionDiffParmReporter
{
  public:
    using Base = FiniteElementConstitutionDiffParmReporter;
    using Base::Base;

    class DiffParmInfo
    {
      public:
        DiffParmInfo(Codim1DConstitutionDiffParmReporter* impl, Base::DiffParmInfo& diff_parm_info)
            : m_impl(impl)
            , m_diff_parm_info(diff_parm_info)
        {
        }

        luisa::compute::BufferView<const luisa::float3> xs() const noexcept;
        luisa::compute::BufferView<const luisa::float3> x_bars() const noexcept;
        luisa::compute::BufferView<const float>         rest_lengths() const noexcept;
        luisa::compute::BufferView<const luisa::int2>   indices() const noexcept;
        luisa::compute::BufferView<const float>         thicknesses() const noexcept;
        luisa::compute::BufferView<const int>           is_fixed() const noexcept;
        const FiniteElementMethod::ConstitutionInfo&    constitution_info() const noexcept;

        SizeT  frame() const;
        IndexT dof_offset(SizeT frame) const;
        IndexT dof_count(SizeT frame) const;

        // Triplet matrix view for sparse matrix representation
        // Using luisa::span for raw data access with row indices, column indices, and values
        luisa::span<const int>   pGpP_rows() const;
        luisa::span<const int>   pGpP_cols() const;
        luisa::span<const float> pGpP_values() const;
        float                    dt() const;

      protected:
        Codim1DConstitutionDiffParmReporter* m_impl = nullptr;
        Base::DiffParmInfo&                  m_diff_parm_info;
    };

  protected:
    virtual void do_assemble(DiffParmInfo& info) = 0;

  private:
    virtual void do_assemble(Base::DiffParmInfo& info) final override;
    IndexT       get_dim() const noexcept final override;
};
}  // namespace uipc::backend::luisa
