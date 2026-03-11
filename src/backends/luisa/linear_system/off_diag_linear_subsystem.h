#pragma once
#include <sim_system.h>
#include <linear_system/diag_linear_subsystem.h>

namespace uipc::backend::luisa
{
/**
 * @brief An off diag linear subsystem represents a submatrix of the global hessian
 * 
 * This is the luisa-compute backend implementation of OffDiagLinearSubsystem.
 * It provides the same interface as the CUDA backend version but uses luisa-compute
 * for GPU acceleration.
 * 
 * Off-diagonal subsystems represent coupling terms between two diagonal subsystems
 * (l and r), storing the Hessian blocks that connect them in the global linear system.
 */
class OffDiagLinearSubsystem : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
        /**
         * @brief Connect this off-diagonal subsystem to two diagonal subsystems
         * 
         * @param l The left diagonal subsystem
         * @param r The right diagonal subsystem
         */
        void connect(DiagLinearSubsystem* l, DiagLinearSubsystem* r)
        {
            m_diag_l = l;
            m_diag_r = r;
        }

      private:
        friend class OffDiagLinearSubsystem;
        DiagLinearSubsystem* m_diag_l = nullptr;
        DiagLinearSubsystem* m_diag_r = nullptr;
    };

    class InitInfo
    {
      public:
    };

    /**
     * @brief Get the unique identifier for this subsystem (pair of connected diag subsystem UIDs)
     * 
     * @return A tuple containing the UIDs of the left and right diagonal subsystems
     */
    std::tuple<U64, U64> uid() const noexcept;

  protected:
    /**
     * @brief Report the extent (size) of the off-diagonal blocks
     * 
     * Called during the linear system setup to determine memory requirements
     * for the off-diagonal Hessian blocks.
     * 
     * @param info The extent info structure to populate with block counts
     */
    virtual void report_extent(GlobalLinearSystem::OffDiagExtentInfo& info) = 0;

    /**
     * @brief Assemble the off-diagonal Hessian blocks into the global system
     * 
     * Called during linear system assembly to fill in the off-diagonal
     * coupling terms between the connected diagonal subsystems.
     * 
     * @param info The assembly info containing buffers for lr_hessian and rl_hessian
     */
    virtual void assemble(GlobalLinearSystem::OffDiagInfo&) = 0;

    /**
     * @brief Build the subsystem with the given build info
     * 
     * @param info Build information including connected diagonal subsystems
     */
    virtual void do_build(BuildInfo& info) = 0;

    /**
     * @brief Initialize the subsystem
     * 
     * @param info Initialization information
     */
    virtual void do_init(InitInfo& info);

  private:
    friend class GlobalLinearSystem;
    DiagLinearSubsystem* m_l = nullptr;
    DiagLinearSubsystem* m_r = nullptr;
    virtual void         do_build() final override;
    void                 init();  // only be called by GlobalLinearSystem
};
}  // namespace uipc::backend::luisa
