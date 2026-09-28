#pragma once
/********************************************************************
 * @file bdf1_flag.h
 * @brief Port of src/backends/cuda/time_integrator/bdf1_flag.h (T4.2).
 *
 * Identical shape. The system only exists when the scene's `integrator/type` is
 * "bdf1" (see the SimSystemCreator specialization in the .cpp), which is how a
 * dependent system shuts itself down: `require<BDF1Flag>()` throws
 * SimSystemException when the flag was not created, and
 * SimSystemCollection::build_systems() then invalidates the dependent system
 * (and, transitively, everything that requires *it*).
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
/*
* @brief FEM BDF1 Flag
 *
 * This is a flag class, by `require<BDF1Flag>()` you can easily shutdown you dependent
 * systems if the BDF1 integration method is not enabled.
*/
class BDF1Flag final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  private:
    void do_build() override;
};
}  // namespace uipc::backend::luisa
