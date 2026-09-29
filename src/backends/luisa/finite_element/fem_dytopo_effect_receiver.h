#pragma once
/********************************************************************
 * @file fem_dytopo_effect_receiver.h
 * @brief T5.4a — the FEM dynamic-topology effect receiver (port of
 *        src/backends/cuda/finite_element/fem_dytopo_effect_receiver.{h,cu} —
 *        namespace + T3.1 view-type change only).
 *
 * Pulled forward from T5.5 because `FEMLinearSubsystem`'s assembly needs it: the
 * receiver is the FEM end of the dytopo effect system — it declares the FEM
 * vertices' global range to the `GlobalDyTopoEffectManager` and caches the
 * classified gradient/Hessian buffers the manager hands it, which
 * `FEMLinearSubsystem::Impl::_assemble_dytopo_effect` then scatters into the
 * global linear system.
 *
 * The `find<>` in `FEMLinearSubsystem::do_build` makes it optional: with contact
 * disabled the dytopo manager (and with it this receiver) is never built, and the
 * subsystem's dytopo branch is simply skipped.
 *********************************************************************/
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class FEMDyTopoEffectReceiver final : public DyTopoEffectReceiver
{
  public:
    using DyTopoEffectReceiver::DyTopoEffectReceiver;

    class Impl
    {
      public:
        void receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info);

        FiniteElementVertexReporter* finite_element_vertex_reporter = nullptr;

        // cuda_tool::CDoubletVectorView<Float,3> / CTripletMatrixView<Float,3>
        lla::LCDoubletVectorView<Float, 3>    gradients;
        lla::LCTripletMatrixView<Float, 3, 3> hessians;
    };

    auto gradients() const noexcept { return m_impl.gradients; }
    auto hessians() const noexcept { return m_impl.hessians; }

  protected:
    virtual void do_build(DyTopoEffectReceiver::BuildInfo& info) override;

  private:
    friend class FEMLinearSubsystem;

    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) override;
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) override;
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
