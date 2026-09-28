/********************************************************************
 * @file fem_3d_constitution.h
 * @brief T5.1/T5.2 — FEM3DConstitution, the dimension-3d constitution base (port of
 * src/backends/cuda/finite_element/fem_3d_constitution.h).
 *
 * SCOPE NOTE (T5.1): ported in T5.1 together with finite_element_constitution.h
 * because `FiniteElementMethod::_classify_base_constitutions()` `dynamic_cast`s
 * every constitution to its dimension base and the base chain's virtual overrides
 * must be *defined* (they are what fills the vtable of a concrete constitution).
 * The bodies are the reference's pure plumbing (type dispatch + the BaseInfo
 * accessors); T5.2 adds the concrete constitutions on top.
 *********************************************************************/
#pragma once
#include <finite_element/finite_element_constitution.h>

namespace uipc::backend::luisa
{
class FEM3DConstitution : public FiniteElementConstitution
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
        BaseInfo(FEM3DConstitution* impl, SizeT index_in_dim, Float dt)
            : m_impl(impl)
            , m_index_in_dim(index_in_dim)
            , m_dt(dt)
        {
        }

        lud::LCBufferView<LVector3> xs() const noexcept;
        lud::LCBufferView<LVector3> x_bars() const noexcept;
        lud::LCBufferView<LMatrix3x3> Dm_invs() const noexcept;
        lud::LCBufferView<Float> rest_volumes() const noexcept;
        lud::LCBufferView<LVector4i> indices() const noexcept;
        const FiniteElementMethod::ConstitutionInfo& constitution_info() const noexcept;
        auto dt() const noexcept { return m_dt; }

      protected:
        SizeT m_index_in_dim = ~0ull;
        FEM3DConstitution* m_impl = nullptr;
        Float m_dt = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(FEM3DConstitution* impl,
                          SizeT index_in_dim,
                          Float dt,
                          lud::LBufferView<Float> energies)
            : BaseInfo(impl, index_in_dim, dt)
            , m_energies(energies)
        {
        }

        auto energies() const noexcept { return m_energies; }

      private:
        lud::LBufferView<Float> m_energies;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(FEM3DConstitution* impl,
                                   SizeT index_in_dim,
                                   bool gradient_only,
                                   Float dt,
                                   lla::LDoubletVectorView<Float, 3> gradients,
                                   lla::LTripletMatrixView<Float, 3, 3> hessians)
            : BaseInfo(impl, index_in_dim, dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
        {
        }

        auto gradients() const noexcept { return m_gradients; }
        auto hessians() const noexcept { return m_hessians; }
        auto gradient_only() const noexcept { return m_gradient_only; }

      private:
        lla::LDoubletVectorView<Float, 3> m_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_hessians;
        bool m_gradient_only = false;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) = 0;

  private:
    friend class FiniteElementMethod;
    virtual void do_build(FiniteElementConstitution::BuildInfo& info) override final;
    virtual void do_compute_energy(FiniteElementConstitution::ComputeEnergyInfo& info) override final;
    virtual void do_compute_gradient_hessian(
        FiniteElementConstitution::ComputeGradientHessianInfo& info) override final;
    virtual IndexT get_dim() const noexcept override final;
};
}  // namespace uipc::backend::luisa
