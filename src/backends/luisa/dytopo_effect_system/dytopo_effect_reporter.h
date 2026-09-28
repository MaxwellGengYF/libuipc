#pragma once
/********************************************************************
 * @file dytopo_effect_reporter.h
 * @brief Base class of the dynamic-topology effect reporters (port of
 *        src/backends/cuda/dytopo_effect_system/dytopo_effect_reporter.h).
 *
 * Type mapping (cookbook §5): cuda_tool::CBufferView<Float> ->
 * lud::LBufferView<Float>; cuda_tool::CDoubletVectorView<Float,3> ->
 * lla::LCDoubletVectorView<Float,3>; cuda_tool::CTripletMatrixView<Float,3> ->
 * lla::LCTripletMatrixView<Float,3,3>. The `Impl` is the reference's
 * "last assembled views" cache (filled in assemble()/compute_energy()).
 *********************************************************************/
#include <sim_system.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <energy_component_flags.h>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

class DyTopoEffectReporter : public SimSystem
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

    class Impl
    {
      public:
        lud::LCBufferView<Float>              energies;
        lla::LCDoubletVectorView<Float, 3>    gradients;
        lla::LCTripletMatrixView<Float, 3, 3> hessians;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo&);
    virtual void do_report_energy_extent(GlobalDyTopoEffectManager::EnergyExtentInfo& info) = 0;
    virtual void do_report_gradient_hessian_extent(
        GlobalDyTopoEffectManager::GradientHessianExtentInfo& info) = 0;
    virtual void do_assemble(GlobalDyTopoEffectManager::GradientHessianInfo& info) = 0;
    virtual void do_compute_energy(GlobalDyTopoEffectManager::EnergyInfo& info) = 0;
    virtual EnergyComponentFlags component_flags() = 0;

  private:
    friend class GlobalDyTopoEffectManager;
    friend class DyTopoEffectLineSearchReporter;
    void init();  // only be called by GlobalDyTopoEffectManager
    virtual void do_build() final override;
    void report_energy_extent(GlobalDyTopoEffectManager::EnergyExtentInfo& info);
    void report_gradient_hessian_extent(GlobalDyTopoEffectManager::GradientHessianExtentInfo& info);
    void assemble(GlobalDyTopoEffectManager::GradientHessianInfo& info);
    void compute_energy(GlobalDyTopoEffectManager::EnergyInfo& info);
    SizeT m_index = ~0ull;
    Impl  m_impl;
};
}// namespace uipc::backend::luisa
