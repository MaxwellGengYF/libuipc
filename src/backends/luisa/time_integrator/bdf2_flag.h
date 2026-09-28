#pragma once
/********************************************************************
 * @file bdf2_flag.h
 * @brief Port of src/backends/cuda/time_integrator/bdf2_flag.h (T4.2).
 *
 * Identical shape. The system only exists when the scene's
 * `integrator/type` is "bdf2" (see the SimSystemCreator specialization in the
 * .cpp), which is how a dependent system shuts itself down:
 * `require<BDF2Flag>()` throws SimSystemException when the flag was not created,
 * and SimSystemCollection::build_systems() then invalidates the dependent
 * system (and, transitively, everything that requires *it*).
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
/*
* @brief FEM/ABD BDF2 Flag
*
* This is a flag class, by `require<BDF2Flag>()` you can easily shutdown your dependent
* systems if the BDF2 integration method is not enabled.
*/
class BDF2Flag final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  private:
    void do_build() override;
};
}  // namespace uipc::backend::luisa
