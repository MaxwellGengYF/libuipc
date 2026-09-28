#pragma once
/********************************************************************
 * @file dytopo_effect_receiver.h
 * @brief Base class of the dynamic-topology effect receivers (port of
 *        src/backends/cuda/dytopo_effect_system/dytopo_effect_receiver.h —
 *        namespace change only).
 *********************************************************************/
#include <sim_system.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>

namespace uipc::backend::luisa
{
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
    virtual void do_init(InitInfo&);
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
}// namespace uipc::backend::luisa
