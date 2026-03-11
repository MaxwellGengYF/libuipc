#pragma once
#include <contact_system/contact_reporter.h>
#include <line_search/line_searcher.h>
#include <contact_system/contact_coeff.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>

namespace uipc::backend::luisa
{
class GlobalTrajectoryFilter;
class VertexHalfPlaneTrajectoryFilter;

/**
 * @brief Contact reporter for vertex-half-plane normal contact
 * 
 * This class handles contact detection and response between vertices and half-planes
 * (implicit planes) in the normal direction. It computes contact energies, gradients,
 * and Hessians for the IPC (Incremental Potential Contact) solver.
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class VertexHalfPlaneNormalContact : public ContactReporter
{
  public:
    using ContactReporter::ContactReporter;

    class Impl;

    /**
     * @brief Base information class providing access to common contact data
     * 
     * Provides read-only access to contact-related buffers via LuisaCompute BufferViews.
     */
    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        /**
         * @brief Get the contact tabular buffer view
         * 
         * The contact tabular is a 2D table stored as a 1D buffer containing
         * contact coefficients (kappa, mu) for each contact type pair.
         * Access element at (i, j) as: contact_tabular()[i * N + j]
         * where N is the number of contact types.
         */
        luisa::compute::BufferView<ContactCoeff> contact_tabular() const;

        /**
         * @brief Get the vertex-half-plane pairs (PHs) buffer view
         * 
         * Each element is a Vector2i containing:
         * - x: vertex index
         * - y: half-plane index
         */
        luisa::compute::BufferView<Vector2i> PHs() const;

        /**
         * @brief Get the current positions buffer view
         */
        luisa::compute::BufferView<Vector3> positions() const;

        /**
         * @brief Get the previous positions buffer view
         */
        luisa::compute::BufferView<Vector3> prev_positions() const;

        /**
         * @brief Get the rest positions buffer view
         */
        luisa::compute::BufferView<Vector3> rest_positions() const;

        /**
         * @brief Get the thicknesses buffer view
         */
        luisa::compute::BufferView<Float> thicknesses() const;

        /**
         * @brief Get the contact element IDs buffer view
         */
        luisa::compute::BufferView<IndexT> contact_element_ids() const;

        /**
         * @brief Get the subscene element IDs buffer view
         */
        luisa::compute::BufferView<IndexT> subscene_element_ids() const;

        /**
         * @brief Get the global d_hat value (barrier activation distance)
         */
        Float d_hat() const;

        /**
         * @brief Get the per-element d_hats buffer view
         */
        luisa::compute::BufferView<Float> d_hats() const;

        /**
         * @brief Get the time step size
         */
        Float dt() const;

        /**
         * @brief Get the epsilon velocity for friction
         */
        Float eps_velocity() const;

        /**
         * @brief Get the offset for half-plane vertices in the global vertex array
         */
        IndexT half_plane_vertex_offset() const;

      private:
        friend class VertexHalfPlaneNormalContact;
        Impl* m_impl;
    };

    /**
     * @brief Contact information class for gradient and Hessian assembly
     */
    class ContactInfo : public BaseInfo
    {
      public:
        ContactInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }

        /**
         * @brief Get the gradient buffer view (Doublet format)
         * 
         * Gradient is stored as (index, value) pairs in a flattened buffer.
         */
        auto gradients() const noexcept { return m_gradients; }

        /**
         * @brief Get the Hessian buffer view (Triplet format)
         * 
         * Hessian is stored as (row, col, value) triples in a flattened buffer.
         */
        auto hessians() const noexcept { return m_hessians; }

        /**
         * @brief Check if only gradient should be computed (no Hessian)
         */
        auto gradient_only() const noexcept { return m_gradient_only; }

      private:
        friend class VertexHalfPlaneNormalContact;

        // LuisaCompute: sparse vector/matrix views for gradient/hessian
        // Stored as flattened buffers in Doublet/Triplet format
        luisa::compute::BufferView<Float> m_gradients;  // (index, value) pairs
        luisa::compute::BufferView<Float> m_hessians;   // (row, col, value) triples
        bool                              m_gradient_only = false;
    };

    class BuildInfo
    {
      public:
    };

    /**
     * @brief Energy information class for computing contact energies
     */
    class EnergyInfo : public BaseInfo
    {
      public:
        EnergyInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }

        /**
         * @brief Get the energy buffer view for storing computed energies
         */
        luisa::compute::BufferView<Float> energies() const noexcept;

      private:
        friend class VertexHalfPlaneNormalContact;
        luisa::compute::BufferView<Float> m_energies;
    };

    /**
     * @brief Implementation data for vertex-half-plane normal contact
     */
    class Impl
    {
      public:
        void compute_energy(EnergyInfo& info);

        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<GlobalContactManager>   global_contact_manager;
        SimSystemSlot<GlobalVertexManager>    global_vertex_manager;
        SimSystemSlot<VertexHalfPlaneTrajectoryFilter> veretx_half_plane_trajectory_filter;
        SimSystemSlot<HalfPlaneVertexReporter> vertex_reporter;

        SizeT PH_count = 0;
        Float dt       = 0.0;

        // LuisaCompute: device buffers for energy, gradient, hessian
        // Using Buffer<Float> for sparse data stored in flattened format
        luisa::compute::Buffer<Float> energies;
        luisa::compute::Buffer<Float> gradients;  // Doublet format: (index, value) pairs
        luisa::compute::Buffer<Float> hessians;   // Triplet format: (row, col, value) triples
    };

    /**
     * @brief Get the vertex-half-plane pairs buffer view
     */
    luisa::compute::BufferView<Vector2i> PHs() const noexcept;

    /**
     * @brief Get the energy buffer view
     */
    luisa::compute::BufferView<Float> energies() const noexcept;

    /**
     * @brief Get the gradient buffer view (Doublet format)
     */
    luisa::compute::BufferView<Float> gradients() const noexcept;

    /**
     * @brief Get the Hessian buffer view (Triplet format)
     */
    luisa::compute::BufferView<Float> hessians() const noexcept;

  protected:
    /**
     * @brief Override to build the contact reporter
     * 
     * @param info Build information
     */
    virtual void do_build(BuildInfo& info) = 0;

    /**
     * @brief Override to compute contact energy
     * 
     * @param info Energy computation information
     */
    virtual void do_compute_energy(EnergyInfo& info) = 0;

    /**
     * @brief Override to assemble gradient and Hessian
     * 
     * @param info Assembly information
     */
    virtual void do_assemble(ContactInfo& info) = 0;

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
