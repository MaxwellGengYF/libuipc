#pragma once
/********************************************************************
 * @file al_ipc_pipeline_flag.h
 * @brief Port of src/backends/cuda/pipeline/al_ipc_pipeline_flag.h (T4.2).
 *
 * Identical shape. The system only exists when the scene's
 * `contact/constitution` is "al-ipc" (see the SimSystemCreator specialization in the
 * .cpp), which is how a dependent system shuts itself down:
 * `require<ALIPCPipelineFlag>()` throws SimSystemException when the flag was not created,
 * and SimSystemCollection::build_systems() then invalidates the dependent
 * system (and, transitively, everything that requires *it*).
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
/*
* @brief AL-IPC Pipeline Flag
*
* This is a flag class, by `require<ALIPCPipelineFlag>()` you can easily shutdown your dependent
* systems if the AL-IPC pipeline is not enabled.
*/
class ALIPCPipelineFlag final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  private:
    void do_build() override;
};
}  // namespace uipc::backend::luisa
