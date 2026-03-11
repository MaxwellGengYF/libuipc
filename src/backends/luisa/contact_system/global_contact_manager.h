#pragma once
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <contact_system/contact_coeff.h>
#include <algorithm/matrix_converter.h>
#include <utils/offset_count_collection.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
class ContactReporter;
class ContactReceiver;
class GlobalTrajectoryFilter;
class AdaptiveContactParameterReporter;

class GlobalContactManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    using GradientHessianExtentInfo = GlobalDyTopoEffectManager::GradientHessianExtentInfo;

    using GradientHessianInfo = GlobalDyTopoEffectManager::GradientHessianInfo;

    using EnergyExtentInfo = GlobalDyTopoEffectManager::EnergyExtentInfo;

    using EnergyInfo = GlobalDyTopoEffectManager::EnergyInfo;

    class AdaptiveParameterInfo
    {
      public:
        AdaptiveParameterInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        /**
         * @brief Get a view of the contact tabular buffer
         * 
         * The contact tabular is a 2D table stored as a 1D buffer.
         * Access element at (i, j) as: contact_tabular()[i * N + j]
         * where N is the number of columns.
         */
        luisa::compute::BufferView<ContactCoeff> contact_tabular() const noexcept;

        /**
         * @brief Exchange the contact tabular buffer with a new one
         * 
         * Returns the old buffer and sets the internal buffer to the new one.
         */
        S<luisa::compute::Buffer<ContactCoeff>> exchange_contact_tabular(
            S<luisa::compute::Buffer<ContactCoeff>> new_buffer) const noexcept;

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

        SimSystemSlot<GlobalVertexManager>    global_vertex_manager;
        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;

        bool cfl_enabled = false;

        // Contact tabular stored as 1D buffer (2D table flattened)
        // Size: contact_type_count * contact_type_count
        S<luisa::compute::Buffer<ContactCoeff>> contact_tabular;
        SizeT                                   contact_tabular_width = 0;

        vector<ContactCoeff> h_contact_tabular;
        vector<IndexT>       h_contact_mask_tabular;
        vector<IndexT>       h_subcene_mask_tabular;
        
        // Contact mask tabular stored as 1D buffer
        // Size: contact_type_count * contact_type_count
        luisa::compute::Buffer<IndexT> contact_mask_tabular;
        luisa::compute::Buffer<IndexT> subscene_mask_tabular;
        
        Float reserve_ratio = 1.1;

        Float d_hat        = 0.0;
        Float kappa        = 0.0;
        Float dt           = 0.0;
        Float eps_velocity = 0.0;


        /***********************************************************************
        *                     Global Vertex Contact Info                       *
        ***********************************************************************/

        luisa::compute::Buffer<IndexT> vert_is_active_contact;
        luisa::compute::Buffer<Float>  vert_disp_norms;
        luisa::compute::Buffer<Float>  max_disp_norm;  // Using buffer with size 1 for scalar

        SimSystemSlotCollection<ContactReporter>        contact_reporters;
        SimSystemSlotCollection<ContactReceiver>        contact_receivers;
        SimSystemSlot<AdaptiveContactParameterReporter> adaptive_contact_parameter_reporter;
    };

    Float d_hat() const;
    Float eps_velocity() const;
    bool  cfl_enabled() const;

    /**
     * @brief Get a view of the contact tabular buffer
     */
    luisa::compute::BufferView<ContactCoeff> contact_tabular() const noexcept;
    
    /**
     * @brief Get a view of the contact mask tabular buffer
     */
    luisa::compute::BufferView<IndexT> contact_mask_tabular() const noexcept;
    
    /**
     * @brief Get a view of the subscene mask tabular buffer
     */
    luisa::compute::BufferView<IndexT> subscene_mask_tabular() const noexcept;



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

    friend class ContactReporter;
    void add_reporter(ContactReporter* reporter);

    friend class AdaptiveContactParameterReporter;
    void add_reporter(AdaptiveContactParameterReporter* reporter);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
