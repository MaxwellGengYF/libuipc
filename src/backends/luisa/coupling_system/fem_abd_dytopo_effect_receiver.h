#pragma once
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <contact_system/simplex_frictional_contact.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
/**
 * @brief Receiver for FEM-to-ABD dynamic topology effects in LuisaCompute backend
 * 
 * This class receives gradient and Hessian contributions from dynamic topology changes
 * occurring in FEM elements that affect ABD bodies (Finite Element -> Affine Body).
 * It integrates these effects into the global linear subsystem.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 * Replaces muda::CTripletMatrixView<Float, 3> with BufferView<const TripletMatrix3>
 * and uses BufferView<const DoubletVector3> for gradients.
 */
class FEMABDDyTopoEffectReceiver final : public DyTopoEffectReceiver
{
  public:
    using DyTopoEffectReceiver::DyTopoEffectReceiver;
    
    /**
     * @brief Implementation details (PIMPL pattern)
     */
    class Impl
    {
      public:
        /// Pointer to the affine body vertex reporter for coordinate mapping
        AffineBodyVertexReporter*    affine_body_vertex_reporter    = nullptr;
        /// Pointer to the finite element vertex reporter for coordinate mapping  
        FiniteElementVertexReporter* finite_element_vertex_reporter = nullptr;
        
        /**
         * @brief Hessian contributions from dynamic topology effects
         * 
         * Stores 3x3 Hessian blocks as triplets (row, col, value) using
         * LuisaCompute's BufferView for device memory access.
         * Equivalent to muda::CTripletMatrixView<Float, 3> in CUDA backend.
         */
        luisa::compute::BufferView<const TripletMatrix3> hessians;
    };

    /**
     * @brief Get the Hessian buffer view
     * @return Buffer view of const TripletMatrix3 containing Hessian contributions
     */
    auto hessians() const noexcept { return m_impl.hessians; }

  private:
    /**
     * @brief Report classification info for dynamic topology effects
     * @param info Classification info to fill
     */
    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) override;
    
    /**
     * @brief Receive classified dynamic topology effect data
     * @param info Classified effect info containing gradients and Hessians
     */
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) override;

    Impl m_impl;

    /**
     * @brief Build the receiver system
     * @param info Build information
     */
    void do_build(BuildInfo& info) override;
};
}  // namespace uipc::backend::luisa
