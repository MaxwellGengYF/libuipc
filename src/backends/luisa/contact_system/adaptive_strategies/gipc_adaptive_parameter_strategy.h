#pragma once
/********************************************************************
 * @file gipc_adaptive_parameter_strategy.h
 * @brief The GIPC-style adaptive contact-parameter strategy (port of
 * src/backends/cuda/contact_system/adaptive_strategies/gipc_adaptive_parameter_strategy.cu,
 * 250 lines).
 *
 * ref: https://github.com/KemengHuang/Stiff-GIPC
 *
 * The reference defines the class *inside* its `.cu` (all of its device work is two small
 * kernels); the port gives it a header exactly like the other ported `.cu`-only systems
 * (T5.2 rule 11 / T5.3 rule 13): the class and its two kernels' host surface live here,
 * the kernels, their cached shaders and `REGISTER_SIM_SYSTEM` live in the `.cpp`.
 *
 * It is the scene's single `AdaptiveContactParameterReporter` in an adaptive-kappa scene
 * (a contact model with `resistance <= 0`): `do_build` throws `SimSystemException` in a
 * scene with no adaptive kappa (which is how the engine invalidates it), and
 * `GlobalContactManager::add_reporter` asserts it is the only one.
 *
 * `do_compute_parameters` estimates the contact stiffness from the projected Newton
 * direction: `proj_kappa = -(g_c . g_nc) / (g_c . g_c)` (`init_kappa` when
 * `g_c . g_c == 0`), clamped to the manager's scene-adaptive corridor
 * (`kappa_lower_bound()/kappa_upper_bound()`, falling back to the config
 * `contact/adaptive/{min,max}_kappa` when the corridor is not computable), then writes
 * it into the adaptive models' entries of the contact tabular.
 *
 * Type mappings (cookbook §3, the batch's standard)
 * -----------------------------------------------
 *  * `cuda_tool::DeviceBuffer<Vector2i>` -> `lud::LDeviceBuffer<LVector2i>`;
 *  * `cuda_tool::DeviceBuffer2D<ContactCoeff>` -> a flat
 *    `lud::LDeviceBuffer<ContactCoeff>` + the extent (the port's device layer has no 2D
 *    buffer; the extent is `GlobalContactManager::contact_tabular_size()`), and the two
 *    kernels take it as an extra scalar argument;
 *  * `cuda_tool::DeviceDenseVector<Float>` -> `lla::LDenseVector<Float>`;
 *    `cuda_tool::DenseVectorView<Float>` -> `lla::LDenseVectorView<Float>`;
 *    `cuda_tool::LinearSystemContext` -> `lla::LinearSystemContext`.
 *
 * The class's members are public exactly as in the reference (that is also how the
 * `[contact_core]` test reads the strategy's resolved config/index state back).
 *********************************************************************/
#include <contact_system/adaptive_contact_parameter_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <linear_system/global_linear_system.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

// ref: https://github.com/KemengHuang/Stiff-GIPC
class GIPCAdaptiveParameterStrategy : public AdaptiveContactParameterReporter
{
  public:
    using AdaptiveContactParameterReporter::AdaptiveContactParameterReporter;

    SimSystemSlot<GlobalContactManager> contact_manager;
    SimSystemSlot<GlobalVertexManager> vertex_manager;
    SimSystemSlot<GlobalSimplicialSurfaceManager> surface_manager;
    SimSystemSlot<GlobalLinearSystem> linear_system;
    SimSystemSlot<GlobalDyTopoEffectManager> dytopo_effect_manager;

    std::vector<IndexT> h_adaptive_kappa_index;
    lud::LDeviceBuffer<LVector2i> adaptive_topos;
    S<lud::LDeviceBuffer<ContactCoeff>> test_contact_tabular;

    Float min_kappa  = 0.0;
    Float init_kappa = 0.0;
    Float max_kappa  = 0.0;
    Float new_kappa  = 0.0;

    void do_build(BuildInfo& info) override;

    virtual void do_init(InitInfo& info) override;

    lla::LDenseVector<Float> contact_gradient;
    lla::LDenseVector<Float> non_contact_gradient;

    void compute_gradient(AdaptiveParameterInfo& info);

    virtual void do_compute_parameters(AdaptiveParameterInfo& info) override;

    vector<Float> h_kappas;

    void write_scene();
};
}  // namespace uipc::backend::luisa
