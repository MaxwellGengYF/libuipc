#pragma once
#include <finite_element/finite_element_constitution.h>

namespace uipc::backend::luisa
{
class Codim2DConstitution : public FiniteElementConstitution
{
  public:
    using FiniteElementConstitution::FiniteElementConstitution;

    class BuildInfo
    {
      public:
    };

    class BaseInfo
    {
      public:
        BaseInfo(Codim2DConstitution* impl, SizeT index_in_dim, Float dt)
            : m_impl(impl)
            , m_index_in_dim(index_in_dim)
            , m_dt(dt)
        {
        }

        luisa::compute::BufferView<const Vector3>  xs() const;
        luisa::compute::BufferView<const Vector3>  x_bars() const;
        luisa::compute::BufferView<const Float>    rest_areas() const;
        luisa::compute::BufferView<const Float>    thicknesses() const;
        luisa::compute::BufferView<const Vector3i> indices() const;
        const FiniteElementMethod::ConstitutionInfo& constitution_info() const;
        Float dt() const;

      protected:
        SizeT                m_index_in_dim = ~0ull;
        Codim2DConstitution* m_impl         = nullptr;
        Float                m_dt           = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Codim2DConstitution*    impl,
                          SizeT                   index_in_dim,
                          Float                   dt,
                          luisa::compute::BufferView<Float> energies)
            : BaseInfo(impl, index_in_dim, dt)
            , m_energies(energies)
        {
        }

        auto energies() const { return m_energies; }

      private:
        luisa::compute::BufferView<Float> m_energies;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(Codim2DConstitution* impl,
                                   SizeT                index_in_dim,
                                   bool                 gradient_only,
                                   Float                dt,
                                   luisa::compute::BufferView<Float> gradients,
                                   luisa::compute::BufferView<Float> hessians)
            : BaseInfo(impl, index_in_dim, dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
        {
        }

        auto gradients() const { return m_gradients; }
        auto hessians() const { return m_hessians; }
        auto gradient_only() const { return m_gradient_only; }

      private:
        luisa::compute::BufferView<Float> m_gradients;
        luisa::compute::BufferView<Float> m_hessians;
        bool                              m_gradient_only = false;
    };

  protected:
    virtual void do_build(BuildInfo& info)                  = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) = 0;
    const FiniteElementMethod::ConstitutionInfo& constitution_info() const;

  private:
    friend class FiniteElementMethod;
    virtual void do_build(FiniteElementConstitution::BuildInfo& info) override final;
    virtual void do_compute_energy(FiniteElementConstitution::ComputeEnergyInfo& info) override final;
    virtual void do_compute_gradient_hessian(
        FiniteElementConstitution::ComputeGradientHessianInfo& info) override final;
    virtual IndexT get_dim() const override final;
};
}  // namespace uipc::backend::luisa
