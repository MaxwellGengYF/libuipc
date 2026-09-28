#pragma once
/********************************************************************
 * @file animator.h
 * @brief Port of src/backends/cuda/animator/animator.h (T4.2).
 *
 * Identical shape: `do_build()` is final and resolves the GlobalAnimator, then
 * calls the subclass hook and registers the animator with it. The only changes
 * are the namespace and the include prefix.
 *********************************************************************/
#include <sim_system.h>
#include <animator/global_animator.h>

namespace uipc::backend::luisa
{
class Animator : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

  private:
    friend class GlobalAnimator;
    void         init();
    void         step();
    virtual void do_build() override final;

    GlobalAnimator* m_global_animator = nullptr;

  protected:
    Float        substep_ratio() const noexcept;
    virtual void do_init()                     = 0;
    virtual void do_step()                     = 0;
    virtual void do_build(BuildInfo& info)     = 0;
};
}  // namespace uipc::backend::luisa
