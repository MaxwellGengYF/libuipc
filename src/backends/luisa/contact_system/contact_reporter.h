#pragma once
/********************************************************************
 * @file contact_reporter.h
 * @brief The contact energy reporter base (port of
 * src/backends/cuda/contact_system/contact_reporter.h, 47 lines).
 *
 * A `ContactReporter` *is a* `DyTopoEffectReporter` (it reports an energy / gradient /
 * Hessian that flows through the dynamic-topology effect pipeline) **and** a member of
 * `GlobalContactManager::contact_reporters`.  Its `do_build` therefore registers with
 * *both* managers: the base's private `do_build()` registers it with
 * `GlobalDyTopoEffectManager` and calls the protected `do_build(BuildInfo&)`; the
 * override below then registers it with `GlobalContactManager` (which is why the
 * reference overrides `do_build(DyTopoEffectReporter::BuildInfo&)` `final`).
 *
 * It forces `component_flags() == Contact`, so the dytopo-effect manager partitions it
 * into the contact slice of `ComputeDyTopoEffectInfo::component_flags`.
 *
 * Type mapping (cookbook §5, identical to dytopo_effect_system/dytopo_effect_reporter.h):
 * `cuda_tool::CBufferView<Float>` -> `lud::LCBufferView<Float>`;
 * `CDoubletVectorView<Float,3>` -> `lla::LCDoubletVectorView<Float,3>`;
 * `CTripletMatrixView<Float,3>` -> `lla::LCTripletMatrixView<Float,3,3>`.
 *
 * The reference re-declares its own `class Impl` (with the same three views the base
 * already owns) and its own `Impl m_impl;` that nothing ever reads; transcribed verbatim
 * (rule 34 — do not "fix" a reference quirk).
 *********************************************************************/
#include <sim_system.h>
#include <contact_system/global_contact_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>

namespace uipc::backend::luisa
{
class ContactReporter : public DyTopoEffectReporter
{
  public:
    using DyTopoEffectReporter::DyTopoEffectReporter;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    class Impl
    {
      public:
        lud::LCBufferView<Float> energies;
        lla::LCDoubletVectorView<Float, 3> gradients;
        lla::LCTripletMatrixView<Float, 3, 3> hessians;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo& info);

  private:
    friend class GlobalContactManager;
    void         init();  // only be called by GlobalContactManager
    virtual void do_build(DyTopoEffectReporter::BuildInfo&) final override;

    virtual EnergyComponentFlags component_flags() final override
    {
        return EnergyComponentFlags::Contact;
    }
    SizeT m_index = ~0ull;
    Impl  m_impl;
};
}  // namespace uipc::backend::luisa
