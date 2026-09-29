#pragma once
/********************************************************************
 * @file diag_linear_subsystem.h
 * @brief T5.4a — the diag linear-subsystem base (port of
 *        src/backends/cuda/linear_system/diag_linear_subsystem.h — namespace +
 *        `cuda_tool` include change only).
 *
 * A diag linear subsystem represents a submatrix of the global Hessian and a
 * subvector of the global gradient: it registers itself with the
 * `GlobalLinearSystem` (in its `do_build()`), reports its own dof/block extent and
 * assembles its contribution into the shared triplet/doublet buffers.
 *
 * The private access surface is the reference's: `GlobalLinearSystem` (and, via
 * C++11 nested-class access, its `Impl`) is the only caller of `init()` /
 * `report_extent()` / `assemble()` / `accuracy_check()` / `retrieve_solution()` /
 * `diag_norm()` / `mass_norm()`, while `uid()` / `dof_offset()` / `dof_count()`
 * are public (a test and the reporters use them).
 *********************************************************************/
#include <sim_system.h>
#include <linear_system/global_linear_system.h>
namespace uipc::backend::luisa
{
/**
 * @brief A diag linear subsystem represents a submatrix of the global hessian and a subvector of the global gradient.
 */
class DiagLinearSubsystem : public SimSystem
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

    U64 uid() const noexcept;

    IndexT dof_offset() const noexcept;
    IndexT dof_count() const noexcept;

  protected:
    virtual void do_build(BuildInfo& info);
    virtual void do_init(InitInfo& info) = 0;

    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) = 0;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) = 0;

    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) = 0;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info) = 0;
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) = 0;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) = 0;

    virtual Float do_diag_norm(GlobalLinearSystem::DiagNormInfo& info) = 0;
    virtual Float do_mass_norm(GlobalLinearSystem::DiagNormInfo& info) = 0;
    virtual U64 get_uid() const noexcept = 0;

  private:
    friend class GlobalLinearSystem;
    virtual void do_build() final override;

    void init();  // only be called by GlobalLinearSystem

    void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
    void receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info);

    void report_extent(GlobalLinearSystem::DiagExtentInfo& info);
    void assemble(GlobalLinearSystem::DiagInfo& info);
    void accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
    void retrieve_solution(GlobalLinearSystem::SolutionInfo& info);

    Float diag_norm(GlobalLinearSystem::DiagNormInfo& info);
    Float mass_norm(GlobalLinearSystem::DiagNormInfo& info);

    SizeT m_index = ~0ull;

    SimSystemSlot<GlobalLinearSystem> m_global_linear_system;
};
}  // namespace uipc::backend::luisa
