#pragma once
/********************************************************************
 * @file affine_body/abd_dytopo_effect_receiver.h
 * @brief T6.1b — the ABD dynamic-topology effect receiver (port of
 *        src/backends/cuda/affine_body/abd_dytopo_effect_receiver.{h,cu} —
 *        namespace + T3.1 view-type change only).
 *
 * The ABD end of the dytopo effect system: it declares the ABD *vertices'* global
 * range to the `GlobalDyTopoEffectManager` and caches the classified
 * gradient/Hessian buffers the manager hands it, which
 * `ABDLinearSubsystem::Impl::_assemble_dytopo_effect` then maps through the
 * per-vertex jacobians (`J^T G`, `J_i^T H J_j`) into the body dofs.
 *
 * The `find<>` in `ABDLinearSubsystem::do_build` makes it optional: with contact
 * disabled the dytopo manager is never built, and with it this receiver, so the
 * subsystem's dytopo branch is simply skipped (the same arrangement as T5.4a's
 * `FEMDyTopoEffectReceiver`).
 *********************************************************************/
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <affine_body/affine_body_vertex_reporter.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class ABDDyTopoEffectReceiver final : public DyTopoEffectReceiver
{
  public:
    using DyTopoEffectReceiver::DyTopoEffectReceiver;

    class Impl
    {
      public:
        void receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info);

        AffineBodyVertexReporter* affine_body_vertex_reporter = nullptr;

        // cuda_tool::CDoubletVectorView<Float,3> / CTripletMatrixView<Float,3>
        lla::LCDoubletVectorView<Float, 3>    gradients;
        lla::LCTripletMatrixView<Float, 3, 3> hessians;
    };

    auto gradients() const noexcept { return m_impl.gradients; }
    auto hessians() const noexcept { return m_impl.hessians; }

  protected:
    virtual void do_build(DyTopoEffectReceiver::BuildInfo& info) override;

  private:
    friend class ABDLinearSubsystem;

    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) override;
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) override;
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
