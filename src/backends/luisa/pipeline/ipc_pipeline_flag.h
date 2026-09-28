#pragma once
/********************************************************************
 * @file ipc_pipeline_flag.h
 * @brief Port of src/backends/cuda/pipeline/ipc_pipeline_flag.h (T4.2).
 *
 * Identical shape. The system only exists when the scene's
 * `contact/constitution` is "ipc" (see the SimSystemCreator specialization in the
 * .cpp), which is how a dependent system shuts itself down:
 * `require<IPCPipelineFlag>()` throws SimSystemException when the flag was not created,
 * and SimSystemCollection::build_systems() then invalidates the dependent
 * system (and, transitively, everything that requires *it*).
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
/*
* @brief IPC Pipeline Flag
*
* This is a flag class, by `require<IPCPipelineFlag>()` you can easily shutdown your dependent
* systems if the IPC pipeline is not enabled.
*/
class IPCPipelineFlag final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  private:
    void do_build() override;
};
}  // namespace uipc::backend::luisa
