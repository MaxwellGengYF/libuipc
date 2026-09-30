#pragma once
/********************************************************************
 * @file global_contact_manager.h
 * @brief Global contact manager — the contact core (port of
 * src/backends/cuda/contact_system/global_contact_manager.{h,cu}, 156 + 701 lines).
 *
 * It is the registry and driver of the four things the contact system needs:
 *  * the contact / subscene **mask tabulars** and the `ContactCoeff` **contact tabular**
 *    (built in `init()`; the masks gate the trajectory filters, the coefficient table
 *    feeds the contact models);
 *  * the scene-adaptive **kappa corridor** (Stiff-GIPC `suggestKappa`/`upperBoundKappa`,
 *    with the dt^2 convention conversion) and the CFL/feasible-step step caps;
 *  * the **contact reporters** (`ContactReporter`, themselves `DyTopoEffectReporter`s)
 *    and the single **adaptive contact-parameter reporter**;
 *  * the `ContactCoeff`-table **exchange** the adaptive strategy needs
 *    (`AdaptiveParameterInfo`).
 *
 * HISTORY: T7.2a ported only the slice the trajectory filters read (the two mask
 * tabulars, `d_hat`/`eps_velocity`/CFL state, `vert_is_active_contact`, the system's
 * creation and init wiring) under the `T8.1` scope markers.  T8.1a completes the class
 * in place; every marker is gone.
 *
 * TYPE MAPPINGS (the port's standard, cookbook §3/§7.13)
 * -----------------------------------------------------
 * * `cuda_tool::DeviceBuffer2D<IndexT>` / `<ContactCoeff>` -> a **flat**
 *   `lud::LDeviceBuffer<T>` (`index = row * extent + col`) plus the extent, because the
 *   port's device layer has no 2D buffer/view (T3.1/T3.2 decision, the same one the
 *   T7.1b sanity checkers use).  `contact_mask_size()` / `subscene_mask_size()` /
 *   `contact_tabular_size()` carry the extents; a kernel gets one as an extra scalar
 *   argument.  The reference's `S<DeviceBuffer2D<ContactCoeff>> contact_tabular` keeps
 *   its shared-pointer *shape* (`S<lud::LDeviceBuffer<ContactCoeff>>`), because
 *   `AdaptiveParameterInfo::exchange_contact_tabular` swaps that pointer.
 * * `cuda_tool::CBuffer2DView<T>` -> `lud::LCBufferView<T>` (flat) + the extent.
 * * `cuda_tool::DeviceVar<Float>` -> `lud::LDeviceVar<Float>` (the T6.x/T7.1 spelling;
 *   the implicit host read-back is made explicit with `lud::read_host_sync` where the
 *   reference relies on `operator T()`, see the .cpp).
 * * `cub::DeviceReduce::Min/Max` -> `lua::device_reduce_min/max` (luisa_device/linear/
 *   algorithms.h).
 * * `cuda_tool::DeviceVector::resize(n, v)` -> `resize(n)` + `fill(v)` where a fresh
 *   buffer is meant.
 *
 * The public system interface is identical to the reference's (plan rule 3), plus the
 * accessors the port needs because its 2D views are flat (`*_size()`) and one documented
 * test seam (`impl()`, the T7.2a `GlobalTrajectoryFilter::run_stage` precedent).
 *********************************************************************/
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <contact_system/contact_coeff.h>
#include <type_define.h>

namespace uipc::backend::luisa
{
class ContactReporter;
class ContactReceiver;
class GlobalTrajectoryFilter;
class GlobalSimplicialSurfaceManager;
class AdaptiveContactParameterReporter;
class FiniteElementMethod;
class AffineBodyDynamics;

class GlobalContactManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    using GradientHessianExtentInfo = GlobalDyTopoEffectManager::GradientHessianExtentInfo;

    using GradientHessianInfo = GlobalDyTopoEffectManager::GradientHessianInfo;

    using EnergyExtentInfo = GlobalDyTopoEffectManager::EnergyExtentInfo;

    using EnergyInfo = GlobalDyTopoEffectManager::EnergyInfo;

    class Impl;

    class AdaptiveParameterInfo
    {
      public:
        AdaptiveParameterInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        /// the flat contact tabular (index = row * contact_tabular_size() + col)
        lud::LBufferView<ContactCoeff> contact_tabular() const noexcept;

        S<lud::LDeviceBuffer<ContactCoeff>> exchange_contact_tabular(
            S<lud::LDeviceBuffer<ContactCoeff>> new_buffer) const noexcept;

      private:
        Impl* m_impl;
    };

    class Impl
    {
      public:
        void  init(WorldVisitor& world);
        void  _build_contact_tabular(WorldVisitor& world);
        void  _build_subscene_tabular(WorldVisitor& world);
        void  compute_d_hat();
        void  compute_adaptive_kappa();
        Float compute_cfl_condition();
        Float compute_feasible_step();

        SimSystemSlot<GlobalVertexManager> global_vertex_manager;
        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<GlobalSimplicialSurfaceManager> global_surface_manager;
        SimSystemSlot<FiniteElementMethod> finite_element_method;
        SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

        bool cfl_enabled = false;

        S<lud::LDeviceBuffer<ContactCoeff>> contact_tabular;

        /// The extent N of the flat contact tabular (`index = row * N + col`); the
        /// reference's `contact_tabular->extent().width`.
        IndexT contact_tabular_size = 0;

        vector<ContactCoeff> h_contact_tabular;
        vector<IndexT> h_contact_mask_tabular;
        vector<IndexT> h_subcene_mask_tabular;
        lud::LDeviceBuffer<IndexT> contact_mask_tabular;
        lud::LDeviceBuffer<IndexT> subscene_mask_tabular;
        /// the extents of the flat mask layouts (`index = row * size + col`)
        IndexT contact_mask_size  = 0;
        IndexT subscene_mask_size = 0;
        Float  reserve_ratio      = 1.1;

        Float d_hat = 0.0;
        Float kappa = 0.0;
        S<const geometry::AttributeSlot<Float>> dt_attr;
        Float eps_velocity = 0.0;

        // Scene-adaptive kappa corridor, ported from Stiff-GIPC's
        // suggestKappa/upperBoundKappa (GIPC.cu:9850-9888) with the dt^2
        // convention conversion. Computed once in init(); <= 0 means "not
        // computable for this scene" and callers fall back to the
        // contact/adaptive/{min,max}_kappa config values.
        Float kappa_lower_bound = -1.0;
        Float kappa_upper_bound = -1.0;

        /***********************************************************************
        * Global Vertex Contact Info *
        ***********************************************************************/

        lud::LDeviceBuffer<IndexT> vert_is_active_contact;
        lud::LDeviceBuffer<Float>  vert_disp_norms;
        lud::LDeviceVar<Float> max_disp_norm;

        lud::LDeviceBuffer<Float> feasible_tois;
        lud::LDeviceVar<Float> min_feasible_toi;

        SimSystemSlotCollection<ContactReporter> contact_reporters;
        SimSystemSlotCollection<ContactReceiver> contact_receivers;
        SimSystemSlot<AdaptiveContactParameterReporter> adaptive_contact_parameter_reporter;
    };

    Float d_hat() const;
    Float eps_velocity() const;
    bool  cfl_enabled() const;

    // Scene-adaptive kappa corridor (Stiff-GIPC suggestKappa/upperBoundKappa
    // with the dt^2 conversion); <= 0 = not computable, use config fallback
    Float kappa_lower_bound() const;
    Float kappa_upper_bound() const;

    lud::LCBufferView<ContactCoeff> contact_tabular() const noexcept;
    lud::LCBufferView<IndexT> contact_mask_tabular() const noexcept;
    lud::LCBufferView<IndexT> subscene_mask_tabular() const noexcept;

    /// The extents of the flat mask / coefficient layouts (see the header note).
    IndexT contact_tabular_size() const noexcept;
    IndexT contact_mask_size() const noexcept;
    IndexT subscene_mask_size() const noexcept;

  protected:
    virtual void do_build() override;

  private:
    friend class SimEngine;
    friend class ContactLineSearchReporter;
    friend class GlobalTrajectoryFilter;
    friend class ContactExporterManager;

    void init();

    void compute_adaptive_parameters();

    Float compute_cfl_condition();

    // Stiff-GIPC-style feasible-step pre-cap: over the current active contact
    // set, the largest Newton step that keeps every pair's gap >= (1-slackness)
    // of its current value. Applied before trajectory candidate generation to
    // shrink the swept-AABB candidate set.
    Float compute_feasible_step();

    friend class ContactReporter;
    void add_reporter(ContactReporter* reporter);

    friend class AdaptiveContactParameterReporter;
    void add_reporter(AdaptiveContactParameterReporter* reporter);

    Impl m_impl;

  public:
    // -------------------------------------------------------------------------
    // T8.1a test seam — the T7.2a `GlobalTrajectoryFilter::run_stage` precedent.
    // The reference's compute_cfl_condition()/compute_feasible_step()/
    // compute_adaptive_parameters() and its Impl are private, with the Newton loop
    // (advance_ipc.cu, Phase 8 / T8.4) as their only caller; the port's engine shell
    // does not run it yet, so without a way in none of them could be driven (nor the
    // `feasible_tois` / `min_feasible_toi` / `vert_disp_norms` read back).  This is
    // additive (no state, no behaviour) and is removed when T8.4 wires the loop.
    // -------------------------------------------------------------------------
    Impl& impl() noexcept { return m_impl; }
    /// the Newton loop's adaptive-parameter fan-out (advance_ipc.cu), exposed
    void run_compute_adaptive_parameters() { compute_adaptive_parameters(); }
};
}  // namespace uipc::backend::luisa
