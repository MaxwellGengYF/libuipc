#pragma once
#include <luisa/luisa-compute.h>
#include <backends/luisa/finite_element/codim_2d_constitution.h>

namespace uipc::backend::luisa
{
class NeoHookeanShell2D final : public Codim2DConstitution
{
  public:
    using Codim2DConstitution::Codim2DConstitution;

    [[nodiscard]] std::string_view get_name() const noexcept override;

    void do_init(FiniteElementAnimator::ScopedInitState& state) override;

    void do_compute_energy(ComputeEnergyInfo& info) override;

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;

    void do_step(StepInfo& info) override;

    class Impl;

  private:
    luisa::unique_ptr<Impl> m_impl;
};
}  // namespace uipc::backend::luisa
