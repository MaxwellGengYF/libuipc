#pragma once
#include <dytopo_effect_system/dytopo_effect_receiver.h>

namespace uipc::backend::luisa
{
// Forward declarations
class FiniteElementVertexReporter;
class FEMLinearSubsystem;

/**
 * @brief Receiver for dynamic topology effects in Finite Element Method (FEM)
 * 
 * This class receives gradient and Hessian contributions from dynamic topology changes
 * and integrates them into the FEM linear subsystem.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class FEMDyTopoEffectReceiver final : public DyTopoEffectReceiver
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

        FiniteElementVertexReporter* finite_element_vertex_reporter = nullptr;

        // LuisaCompute DoubletVector3 for gradients (index, float3 value pairs)
        // Using const view (equivalent to muda::CDoubletVectorView<Float, 3>)
        DoubletVector3 gradients;
        
        // LuisaCompute TripletMatrix3 for Hessians (row, col, float3x3 block)
        // Using const view (equivalent to muda::CTripletMatrixView<Float, 3>)
        TripletMatrix3 hessians;
    };

    /**
     * @brief Get the gradient doublet vector view
     * @return Const view of gradient doublets
     */
    auto gradients() const noexcept { return m_impl.gradients; }
    
    /**
     * @brief Get the Hessian triplet matrix view
     * @return Const view of Hessian triplets
     */
    auto hessians() const noexcept { return m_impl.hessians; }

  protected:
    /**
     * @brief Build the receiver system
     * @param info Build information
     */
    virtual void do_build(DyTopoEffectReceiver::BuildInfo& info) override;

  private:
    friend class FEMLinearSubsystem;

    /**
     * @brief Report classification info for dynamic topology effects
     * @param info Classification information to fill
     */
    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) override;
    
    /**
     * @brief Receive classified gradient and Hessian contributions
     * @param info Classified effect information containing gradients and Hessians
     */
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) override;
    
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
