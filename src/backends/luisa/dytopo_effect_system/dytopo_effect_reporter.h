#pragma once
#include <sim_system.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <energy_component_flags.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/basic_types.h>

namespace uipc::backend::luisa
{
// Forward declarations for sparse matrix/vector types that will be used by the linear system
// These are LuisaCompute-compatible replacements for muda's linear system types

template<typename T, int N>
struct Triplet
{
    uint32_t row;
    uint32_t col;
    luisa::Vector<T, N> value;
};

template<typename T, int N>
struct Doublet
{
    uint32_t index;
    luisa::Vector<T, N> value;
};

// View types for sparse data (constant/read-only views)
template<typename T, int N>
using CTripletMatrixView = luisa::compute::BufferView<Triplet<T, N>>;

template<typename T, int N>
using CDoubletVectorView = luisa::compute::BufferView<Doublet<T, N>>;

// Regular (mutable) view types
template<typename T, int N>
using TripletMatrixView = luisa::compute::BufferView<Triplet<T, N>>;

template<typename T, int N>
using DoubletVectorView = luisa::compute::BufferView<Doublet<T, N>>;

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
        luisa::compute::BufferView<Float> energies;
        CDoubletVectorView<Float, 3>      gradients;
        CTripletMatrixView<Float, 3>      hessians;
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
    void do_build() final override;
    void report_energy_extent(GlobalDyTopoEffectManager::EnergyExtentInfo& info);
    void report_gradient_hessian_extent(GlobalDyTopoEffectManager::GradientHessianExtentInfo& info);
    void  assemble(GlobalDyTopoEffectManager::GradientHessianInfo& info);
    void  compute_energy(GlobalDyTopoEffectManager::EnergyInfo& info);
    SizeT m_index = ~0ull;
    Impl  m_impl;
};
}  // namespace uipc::backend::luisa
