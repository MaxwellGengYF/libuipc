#pragma once
#include <contact_system/contact_reporter.h>
#include <line_search/line_searcher.h>
#include <contact_system/contact_coeff.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>

namespace uipc::backend::luisa
{
class GlobalTrajectoryFilter;
class VertexHalfPlaneTrajectoryFilter;

class VertexHalfPlaneFrictionalContact : public ContactReporter
{
  public:
    using ContactReporter::ContactReporter;

    class Impl;

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        // LuisaCompute: BufferView for read-only buffer access
        luisa::compute::BufferView<ContactCoeff> contact_tabular() const;
        luisa::compute::BufferView<Vector2i>     friction_PHs() const;
        luisa::compute::BufferView<Vector3>      positions() const;
        luisa::compute::BufferView<Float>        thicknesses() const;
        luisa::compute::BufferView<Vector3>      prev_positions() const;
        luisa::compute::BufferView<Vector3>      rest_positions() const;
        luisa::compute::BufferView<IndexT>       contact_element_ids() const;
        luisa::compute::BufferView<IndexT>       subscene_element_ids() const;
        Float                                    d_hat() const;
        luisa::compute::BufferView<Float>        d_hats() const;
        Float                                    dt() const;
        Float                                    eps_velocity() const;
        IndexT                                   half_plane_vertex_offset() const;

      private:
        friend class VertexHalfPlaneFrictionalContact;
        Impl* m_impl;
    };

    class ContactInfo : public BaseInfo
    {
      public:
        ContactInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }

        // Return BufferView for gradient/hessian data
        // Gradient: stored as (index, x, y, z) tuples in Float array
        // Hessian: stored as (row, col, 3x3 matrix) in Float array
        auto gradients() const noexcept { return m_gradients; }
        auto hessians() const noexcept { return m_hessians; }
        bool gradient_only() const noexcept { return m_gradient_only; }

      private:
        friend class VertexHalfPlaneFrictionalContact;

        // LuisaCompute: BufferView for sparse vector/matrix storage
        luisa::compute::BufferView<Float> m_gradients;  // Doublet format
        luisa::compute::BufferView<Float> m_hessians;   // Triplet format
        bool                              m_gradient_only = false;
    };

    class BuildInfo
    {
      public:
    };

    class EnergyInfo : public BaseInfo
    {
      public:
        EnergyInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }

        luisa::compute::BufferView<Float> energies() const noexcept;

      private:
        friend class VertexHalfPlaneFrictionalContact;
        luisa::compute::BufferView<Float> m_energies;
    };

    class Impl
    {
      public:
        void compute_energy(EnergyInfo& info);

        SimSystemSlot<GlobalTrajectoryFilter>          global_trajectory_filter;
        SimSystemSlot<GlobalContactManager>            global_contact_manager;
        SimSystemSlot<GlobalVertexManager>             global_vertex_manager;
        SimSystemSlot<VertexHalfPlaneTrajectoryFilter> veretx_half_plane_trajectory_filter;
        SimSystemSlot<HalfPlaneVertexReporter>         vertex_reporter;

        SizeT PH_count = 0;
        Float dt       = 0.0;

        // LuisaCompute: Device buffers
        // Energy per contact pair
        luisa::compute::Buffer<Float> energies;
        // Gradient in sparse doublet format: (index, x, y, z) per entry
        luisa::compute::Buffer<Float> gradients;
        // Hessian in sparse triplet format: (row, col, 3x3 matrix) per entry
        luisa::compute::Buffer<Float> hessians;
    };

    // Read-only buffer views for external access
    luisa::compute::BufferView<Vector2i> PHs() const noexcept;
    luisa::compute::BufferView<Float>    energies() const noexcept;
    luisa::compute::BufferView<Float>    gradients() const noexcept;
    luisa::compute::BufferView<Float>    hessians() const noexcept;

  protected:
    virtual void do_build(BuildInfo& info)           = 0;
    virtual void do_compute_energy(EnergyInfo& info) = 0;
    virtual void do_assemble(ContactInfo& info)      = 0;

  private:
    virtual void do_report_energy_extent(GlobalContactManager::EnergyExtentInfo& info) override final;
    virtual void do_compute_energy(GlobalContactManager::EnergyInfo& info) override final;
    virtual void do_report_gradient_hessian_extent(
        GlobalContactManager::GradientHessianExtentInfo& info) override final;
    virtual void do_assemble(GlobalContactManager::GradientHessianInfo& info) override final;
    virtual void do_build(ContactReporter::BuildInfo& info) override final;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
