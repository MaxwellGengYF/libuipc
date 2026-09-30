#pragma once
/********************************************************************
 * @file global_contact_manager.h
 * @brief Global contact manager — **the M7 (T7.2a) slice only**
 *        (scoped port of src/backends/cuda/contact_system/global_contact_manager.{h,cu},
 *        156 + 701 lines).
 *
 * WHY THIS FILE EXISTS IN M7
 * --------------------------
 * The reference's trajectory filters (`collision_detection/trajectory_filter.h`,
 * `simplex_trajectory_filter.*`, `vertex_half_plane_trajectory_filter.*`,
 * `filters/{easy,al}_vertex_half_plane_trajectory_filter.*`) `require<>` this system
 * and read exactly five things from it:
 *
 *   * `require<GlobalContactManager>()`   (the base classes' do_build)
 *   * `->d_hat()`                         (`SimplexTrajectoryFilter::BaseInfo`)
 *   * `->contact_mask_tabular()`          (both bases' BaseInfo)
 *   * `->subscene_mask_tabular()`         (both bases' BaseInfo)
 *   * `->cfl_enabled()` / `->m_impl.vert_is_active_contact`
 *                                         (`GlobalTrajectoryFilter::filter_active()`
 *                                          and `LabelActiveVerticesInfo::vert_is_active()`)
 *
 * The port had no contact layer at all (the reference's contact_system/ is T8.1 of the
 * plan: contact models, reporters, pFpx, AL variants), so T7.2a ports **only that
 * surface** — the contact/subscene *mask tabulars*, the vertex contact activation
 * buffer, the scalar `d_hat`/`eps_velocity`/CFL state and the system's creation and
 * init wiring — and leaves every coefficient/reporter/CFL-feasible-step part declared
 * under a `SCOPE NOTE (T8.1)` marker (the T5.4a precedent for "port the half a later
 * task needs, mark the rest").  T8.1 completes this class in place; nothing here is a
 * deviation from the reference, only a *prefix* of it.
 *
 * TYPE MAPPINGS (the port's standard, cookbook §3/§7.13)
 * -----------------------------------------------------
 *   * `cuda_tool::DeviceBuffer2D<IndexT>` -> a **flat** `lud::LDeviceBuffer<IndexT>`
 *     (`index = row * extent + col`) plus the extent, because the port's device layer
 *     has no 2D buffer/view (T3.1/T3.2 decision, the same one the T7.1b sanity checkers
 *     and the flat contact-mask helpers use).  The extent travels with the view as
 *     `contact_mask_size()` / `subscene_mask_size()`; a kernel gets it as an extra
 *     scalar argument.
 *   * `cuda_tool::DeviceBuffer<T>` -> `lud::LDeviceBuffer<T>`;
 *     `CBuffer2DView<IndexT>` -> `lud::LCBufferView<IndexT>`.
 *   * `cuda_tool::DeviceVector::resize(n, v)` (fill *all* elements with `v`) ->
 *     `resize(n)` (which value-initialises the new tail) + `fill(v)`; identical for the
 *     fresh buffer this is called on, and defined for a re-init.
 *
 * WHAT THE FILTERS ARE *NOT* ALLOWED TO ASSUME FROM THIS FILE
 * ----------------------------------------------------------
 * The reference's masks are `CBuffer2DView`s (self-describing); the port's are flat, so
 * a filter must pass `contact_mask_size()` alongside the view.  `LVector*`-valued
 * device attributes are not used here (the masks are `IndexT`).
 *********************************************************************/
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <type_define.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class GlobalTrajectoryFilter;
class GlobalSimplicialSurfaceManager;
class FiniteElementMethod;
class AffineBodyDynamics;

class GlobalContactManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class Impl
    {
      public:
        void init(WorldVisitor& world);
        void _build_contact_tabular(WorldVisitor& world);
        void _build_subscene_tabular(WorldVisitor& world);

        // -------------------------------------------------------------------
        // SCOPE NOTE (T8.1): the reference's Impl also owns
        //   contact_tabular (DeviceBuffer2D<ContactCoeff>) + h_contact_tabular,
        //   kappa / kappa_lower_bound / kappa_upper_bound + the scene-adaptive
        //   corridor, contact_reporters / contact_receivers /
        //   adaptive_contact_parameter_reporter, vert_disp_norms / max_disp_norm,
        //   feasible_tois / min_feasible_toi, compute_d_hat(),
        //   compute_adaptive_kappa(), compute_cfl_condition(),
        //   compute_feasible_step().
        // None of them is read by the trajectory filters; T8.1 adds them here.
        // -------------------------------------------------------------------

        SimSystemSlot<GlobalVertexManager>            global_vertex_manager;
        SimSystemSlot<GlobalTrajectoryFilter>         global_trajectory_filter;
        SimSystemSlot<GlobalSimplicialSurfaceManager> global_surface_manager;
        SimSystemSlot<FiniteElementMethod>            finite_element_method;
        SimSystemSlot<AffineBodyDynamics>             affine_body_dynamics;

        bool cfl_enabled = false;

        // flat `IndexT[row * size + col]` masks + their extent (see the header note)
        IndexT                     contact_mask_size = 0;
        IndexT                     subscene_mask_size = 0;
        vector<IndexT>             h_contact_mask_tabular;
        vector<IndexT>             h_subcene_mask_tabular;
        lud::LDeviceBuffer<IndexT> contact_mask_tabular;
        lud::LDeviceBuffer<IndexT> subscene_mask_tabular;

        Float reserve_ratio = 1.1;

        Float d_hat        = 0.0;
        Float eps_velocity = 0.0;

        // `dt` is read (and asserted) exactly like the reference; the corridor that
        // consumes it is T8.1's.
        S<const geometry::AttributeSlot<Float>> dt_attr;

        /***********************************************************************
        *                     Global Vertex Contact Info                       *
        ***********************************************************************/
        lud::LDeviceBuffer<IndexT> vert_is_active_contact;
    };

    Float d_hat() const;
    Float eps_velocity() const;
    bool  cfl_enabled() const;

    lud::LCBufferView<IndexT> contact_mask_tabular() const noexcept;
    lud::LCBufferView<IndexT> subscene_mask_tabular() const noexcept;
    /// The extent of the flat mask layouts (the reference's `CBuffer2DView::extent().width`).
    IndexT contact_mask_size() const noexcept;
    IndexT subscene_mask_size() const noexcept;

  protected:
    virtual void do_build() override;

  private:
    friend class SimEngine;
    friend class GlobalTrajectoryFilter;

    // only called by SimEngine (the reference's step-3.1 forwards)
    void init();

    Impl m_impl;
};
}// namespace uipc::backend::luisa
