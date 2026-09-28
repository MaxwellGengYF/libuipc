#pragma once
#include <sim_system.h>
#include <global_geometry/global_simplicial_surface_manager.h>

namespace uipc::backend::luisa
{
/**
 * @brief Base class of every system that reports a simplicial surface to the
 *        GlobalSimplicialSurfaceManager (port of the reference's
 *        simplicial_surface_reporter.h).
 */
class SimplicialSurfaceReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    class BuildInfo
    {
      public:
    };

    using SurfaceInitInfo       = GlobalSimplicialSurfaceManager::SurfaceInitInfo;
    using SurfaceCountInfo      = GlobalSimplicialSurfaceManager::SurfaceCountInfo;
    using SurfaceAttributeInfo  = GlobalSimplicialSurfaceManager::SurfaceAttributeInfo;

  protected:
    virtual void do_build(BuildInfo& info)                            = 0;
    virtual void do_init(SurfaceInitInfo& info)                       = 0;
    virtual void do_report_count(SurfaceCountInfo& info)              = 0;
    virtual void do_report_attributes(SurfaceAttributeInfo& info)     = 0;

  private:
    friend class GlobalSimplicialSurfaceManager;
    virtual void do_build() final override;

    void init(SurfaceInitInfo& info);
    void report_count(SurfaceCountInfo& info);
    void report_attributes(SurfaceAttributeInfo& info);

    friend class GlobalSimplicialSurfaceManager;
    SizeT m_index = ~0ull;
};
}  // namespace uipc::backend::luisa
