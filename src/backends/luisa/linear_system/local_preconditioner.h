#pragma once
#include <sim_system.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::luisa
{
class DiagLinearSubsystem;

/**
 * @brief Local Preconditioner for the linear system solver.
 * 
 * This class provides an interface for implementing local preconditioners
 * used in the PCG (Preconditioned Conjugate Gradient) solver.
 * 
 * Refactored for luisa-compute backend.
 */
class LocalPreconditioner : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    /**
     * @brief Initialization information for the preconditioner.
     */
    class InitInfo
    {
      public:
    };

    /**
     * @brief Build information for connecting with diagonal linear subsystems.
     */
    class BuildInfo
    {
      public:
        /**
         * @brief Connect this preconditioner to a diagonal linear subsystem.
         * @param system The diagonal linear subsystem to connect to.
         */
        void connect(DiagLinearSubsystem* system);

      private:
        friend class LocalPreconditioner;
        DiagLinearSubsystem* m_subsystem = nullptr;
    };

  protected:
    /**
     * @brief Build the preconditioner. Called during system construction.
     * @param info Build information.
     */
    virtual void do_build(BuildInfo& info) = 0;

    /**
     * @brief Initialize the preconditioner.
     * @param info Initialization information.
     */
    virtual void do_init(InitInfo& info) = 0;

    /**
     * @brief Assemble the local preconditioner matrix.
     * @param info Assembly information from the global linear system.
     * 
     * Note: Uses luisa::compute::Buffer for device memory management.
     */
    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) = 0;

    /**
     * @brief Apply the preconditioner to a vector.
     * @param info Application information containing input/output buffers.
     * 
     * Note: Uses luisa::compute::BufferView for kernel parameter passing.
     */
    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) = 0;

  private:
    friend class GlobalLinearSystem;

    virtual void do_build() final override;
    virtual void init();

    void assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info);
    void apply(GlobalLinearSystem::ApplyPreconditionerInfo& info);
    
    DiagLinearSubsystem* m_subsystem = nullptr;
};
}  // namespace uipc::backend::luisa
