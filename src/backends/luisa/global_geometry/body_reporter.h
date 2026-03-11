#pragma once
#include <sim_system.h>
#include <global_geometry/global_body_manager.h>

namespace uipc::backend::luisa
{
/**
 * @brief Base class for body reporters
 * 
 * Reports body counts and attributes to the global body manager.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class BodyReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    using BodyCountInfo     = GlobalBodyManager::BodyCountInfo;
    using BodyAttributeInfo = GlobalBodyManager::BodyAttributeInfo;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    IndexT body_offset() const { return m_body_offset; }
    IndexT body_count() const { return m_body_count; }

  protected:
    virtual void do_init(InitInfo& info)                       = 0;
    virtual void do_build(BuildInfo& info)                     = 0;
    virtual void do_report_count(BodyCountInfo& info)          = 0;
    virtual void do_report_attributes(BodyAttributeInfo& info) = 0;

  private:
    friend class GlobalBodyManager;
    virtual void do_build() final override;
    void         init();
    void         report_count(BodyCountInfo& info);
    void         report_attributes(BodyAttributeInfo& info);

    SizeT  m_index       = ~0ull;
    IndexT m_body_offset = -1;
    IndexT m_body_count  = -1;
};
}  // namespace uipc::backend::luisa

#include "details/global_body_manager.inl"
