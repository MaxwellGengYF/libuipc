#pragma once
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <affine_body/affine_body_vertex_reporter.h>

namespace uipc::backend::luisa
{
/**
 * @brief Receiver for dynamic topology effects in Affine Body Dynamics (ABD)
 * 
 * This class receives gradient and Hessian contributions from dynamic topology changes
 * and integrates them into the ABD linear subsystem.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class ABDDyTopoEffectReceiver final : public DyTopoEffectReceiver
{
  public:
    using DyTopoEffectReceiver::DyTopoEffectReceiver;

    /**
     * @brief Implementation details (PIMPL pattern)
     */
    class Impl
    {
      public:
        void receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info);

        AffineBodyVertexReporter* affine_body_vertex_reporter = nullptr;

        // LuisaCompute view for gradient doublets (index, float3 value)
        // Equivalent to muda::CDoubletVectorView<Float, 3>
        DoubletVector3 gradients;
        // LuisaCompute view for Hessian triplets (row, col, float3x3 block)
        // Equivalent to muda::CTripletMatrixView<Float, 3>
        TripletMatrix3 hessians;
    };


  protected:
    virtual void do_build(DyTopoEffectReceiver::BuildInfo& info) override;

  private:
    friend class ABDLinearSubsystem;
    auto gradients() const noexcept { return m_impl.gradients; }
    auto hessians() const noexcept { return m_impl.hessians; }
    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) override;
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) override;
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
