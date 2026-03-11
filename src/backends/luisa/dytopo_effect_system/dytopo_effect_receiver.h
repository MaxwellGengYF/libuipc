#pragma once
#include <sim_system.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>

namespace uipc::backend::luisa
{
/**
 * @brief Base class for receiving dynamic topology effects
 * 
 * Receivers get gradient and Hessian contributions from dynamic topology changes
 * and integrate them into their respective subsystems.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class DyTopoEffectReceiver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

  protected:
    virtual void do_init(InitInfo&) {}
    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) = 0;
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) = 0;
    virtual void do_build(BuildInfo& info) = 0;

  private:
    friend class GlobalDyTopoEffectManager;
    virtual void do_build() final override;
    void         init();  // only be called by GlobalDyTopoEffectManager
    void         report(GlobalDyTopoEffectManager::ClassifyInfo& info);
    void         receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info);
    SizeT        m_index = ~0ull;
};
}  // namespace uipc::backend::luisa
