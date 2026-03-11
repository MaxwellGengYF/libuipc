#pragma once
#include <sim_system.h>
#include <contact_system/global_contact_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>

namespace uipc::backend::luisa
{
/**
 * @brief Base class for contact reporters in the LuisaCompute backend
 * 
 * Contact reporters are responsible for computing and reporting contact-related
 * energies, gradients, and Hessians to the global contact manager.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
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

    /**
     * @brief Implementation data for contact reporter
     * 
     * Uses LuisaCompute BufferViews instead of muda views.
     * These views provide read-only access to device memory buffers.
     */
    class Impl
    {
      public:
        luisa::compute::BufferView<Float>           energies;
        luisa::compute::BufferView<DoubletVector3>  gradients;
        luisa::compute::BufferView<TripletMatrix3>  hessians;
    };

  protected:
    /**
     * @brief Override to build the contact reporter
     * 
     * @param info Build information
     */
    virtual void do_build(BuildInfo& info) = 0;

    /**
     * @brief Override to initialize the contact reporter
     * 
     * @param info Initialization information
     */
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
