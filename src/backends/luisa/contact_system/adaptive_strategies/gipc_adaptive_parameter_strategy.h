#pragma once
#include <contact_system/adaptive_contact_parameter_reporter.h>
#include <contact_system/contact_coeff.h>
#include <energy_component_flags.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
class GlobalContactManager;
class GlobalVertexManager;
class GlobalSimplicialSurfaceManager;
class GlobalLinearSystem;
class GlobalDyTopoEffectManager;

/**
 * @brief GIPC adaptive contact parameter strategy
 * 
 * Adjusts kappa (contact stiffness) values based on the projection of 
 * contact gradient onto non-contact gradient.
 * 
 * Reference: "GIPC: GPU-Accelerated Incremental Potential Contact" 
 * (SIGGRAPH Asia 2022)
 */
class GIPCAdaptiveParameterStrategy final : public AdaptiveContactParameterReporter
{
  public:
    using AdaptiveContactParameterReporter::AdaptiveContactParameterReporter;

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_init(InitInfo& info) override;
    virtual void do_compute_parameters(AdaptiveParameterInfo& info) override;

  private:
    void compute_gradient(AdaptiveParameterInfo& info);
    void write_scene();

    // Config parameters
    Float min_kappa  = 0.0f;
    Float init_kappa = 0.0f;
    Float max_kappa  = 0.0f;
    Float new_kappa  = 0.0f;

    // System slots
    SimSystemSlot<GlobalContactManager>           contact_manager;
    SimSystemSlot<GlobalVertexManager>            vertex_manager;
    SimSystemSlot<GlobalSimplicialSurfaceManager> surface_manager;
    SimSystemSlot<GlobalLinearSystem>             linear_system;
    SimSystemSlot<GlobalDyTopoEffectManager>      dytopo_effect_manager;

    // Adaptive kappa tracking
    std::vector<IndexT> h_adaptive_kappa_index;
    luisa::compute::Buffer<Vector2i> adaptive_topos;
    S<luisa::compute::Buffer<ContactCoeff>> test_contact_tabular;

    // Gradient buffers
    luisa::compute::Buffer<Float> contact_gradient;
    luisa::compute::Buffer<Float> non_contact_gradient;
};
}  // namespace uipc::backend::luisa
