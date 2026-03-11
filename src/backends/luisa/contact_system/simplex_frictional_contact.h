#pragma once
#include <contact_system/contact_reporter.h>
#include <line_search/line_searcher.h>
#include <contact_system/contact_coeff.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
/**
 * @brief Doublet vector entry for 3D vectors (index + value pair)
 * 
 * Equivalent to muda::DoubletVectorView entry.
 * Used for storing sparse gradients with 3D vector values.
 */
struct DoubletVector3
{
    IndexT   index;   // Global vertex index
    Vector3  value;   // 3D gradient vector
};

/**
 * @brief Triplet matrix entry for 3x3 blocks (row, col + value)
 * 
 * Equivalent to muda::TripletMatrixView entry.
 * Used for storing sparse Hessian blocks with 3x3 matrix values.
 */
struct TripletMatrix3
{
    IndexT   row;     // Row index (global vertex index)
    IndexT   col;     // Column index (global vertex index)  
    Matrix3x3 value;  // 3x3 Hessian block
};

/**
 * @brief Base class for simplex-based frictional contact in LuisaCompute backend
 * 
 * This class provides the interface for computing frictional contact energies,
 * gradients, and Hessians for simplex collision pairs (PT, EE, PE, PP).
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 */
class SimplexFrictionalContact : public ContactReporter
{
  public:
    using ContactReporter::ContactReporter;
    
    // Half Hessian sizes for symmetric matrices
    constexpr static SizeT PTHalfHessianSize = 4 * (4 + 1) / 2;  // 4 vertices, symmetric matrix
    constexpr static SizeT EEHalfHessianSize = 4 * (4 + 1) / 2;  // 4 vertices, symmetric matrix
    constexpr static SizeT PEHalfHessianSize = 3 * (3 + 1) / 2;  // 3 vertices, symmetric matrix
    constexpr static SizeT PPHalfHessianSize = 2 * (2 + 1) / 2;  // 2 vertices, symmetric matrix

    class Impl;

    /**
     * @brief Base information class providing access to common contact data
     */
    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        /**
         * @brief Get contact tabular buffer (flattened 2D table)
         * 
         * Access element at (i, j) as: contact_tabular()[i * N + j]
         * where N is the number of contact types.
         */
        luisa::compute::BufferView<ContactCoeff> contact_tabular() const;
        
        luisa::compute::BufferView<Vector4i> friction_PTs() const;
        luisa::compute::BufferView<Vector4i> friction_EEs() const;
        luisa::compute::BufferView<Vector3i> friction_PEs() const;
        luisa::compute::BufferView<Vector2i> friction_PPs() const;
        
        luisa::compute::BufferView<Vector3> positions() const;
        luisa::compute::BufferView<Vector3> prev_positions() const;
        luisa::compute::BufferView<Vector3> rest_positions() const;
        luisa::compute::BufferView<Float>   thicknesses() const;
        luisa::compute::BufferView<IndexT>  contact_element_ids() const;
        
        Float d_hat() const;
        luisa::compute::BufferView<Float> d_hats() const;
        Float dt() const;
        Float eps_velocity() const;

      private:
        friend class SimplexFrictionalContact;
        Impl* m_impl;
    };

    /**
     * @brief Contact information for assembling gradients and Hessians
     */
    class ContactInfo : public BaseInfo
    {
      public:
        ContactInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }
        
        luisa::compute::BufferView<DoubletVector3> friction_PT_gradients() const noexcept { return m_PT_gradients; }
        luisa::compute::BufferView<TripletMatrix3> friction_PT_hessians() const noexcept { return m_PT_hessians; }
        
        luisa::compute::BufferView<DoubletVector3> friction_EE_gradients() const noexcept { return m_EE_gradients; }
        luisa::compute::BufferView<TripletMatrix3> friction_EE_hessians() const noexcept { return m_EE_hessians; }
        
        luisa::compute::BufferView<DoubletVector3> friction_PE_gradients() const noexcept { return m_PE_gradients; }
        luisa::compute::BufferView<TripletMatrix3> friction_PE_hessians() const noexcept { return m_PE_hessians; }
        
        luisa::compute::BufferView<DoubletVector3> friction_PP_gradients() const noexcept { return m_PP_gradients; }
        luisa::compute::BufferView<TripletMatrix3> friction_PP_hessians() const noexcept { return m_PP_hessians; }
        
        bool gradient_only() const noexcept { return m_gradient_only; }

      private:
        friend class SimplexFrictionalContact;
        
        luisa::compute::BufferView<DoubletVector3> m_PT_gradients;
        luisa::compute::BufferView<TripletMatrix3> m_PT_hessians;

        luisa::compute::BufferView<DoubletVector3> m_EE_gradients;
        luisa::compute::BufferView<TripletMatrix3> m_EE_hessians;

        luisa::compute::BufferView<DoubletVector3> m_PE_gradients;
        luisa::compute::BufferView<TripletMatrix3> m_PE_hessians;

        luisa::compute::BufferView<DoubletVector3> m_PP_gradients;
        luisa::compute::BufferView<TripletMatrix3> m_PP_hessians;
        
        bool m_gradient_only = false;
    };

    class BuildInfo
    {
      public:
    };

    /**
     * @brief Energy information for computing friction energies
     */
    class EnergyInfo : public BaseInfo
    {
      public:
        EnergyInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }

        luisa::compute::BufferView<Float> friction_PT_energies() const noexcept
        {
            return m_PT_energies;
        }
        luisa::compute::BufferView<Float> friction_EE_energies() const noexcept
        {
            return m_EE_energies;
        }
        luisa::compute::BufferView<Float> friction_PE_energies() const noexcept
        {
            return m_PE_energies;
        }
        luisa::compute::BufferView<Float> friction_PP_energies() const noexcept
        {
            return m_PP_energies;
        }

      private:
        friend class SimplexFrictionalContact;
        
        luisa::compute::BufferView<Float> m_PT_energies;
        luisa::compute::BufferView<Float> m_EE_energies;
        luisa::compute::BufferView<Float> m_PE_energies;
        luisa::compute::BufferView<Float> m_PP_energies;
    };

    /**
     * @brief Implementation data for SimplexFrictionalContact
     * 
     * Uses LuisaCompute Buffer and BufferView types instead of muda types.
     */
    class Impl
    {
      public:
        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<GlobalContactManager>   global_contact_manager;
        SimSystemSlot<GlobalVertexManager>    global_vertex_manager;

        SimSystemSlot<SimplexTrajectoryFilter> simplex_trajectory_filter;

        SizeT PT_count = 0;
        SizeT EE_count = 0;
        SizeT PE_count = 0;
        SizeT PP_count = 0;
        Float dt       = 0;

        // PT (Point-Triangle) friction data
        luisa::compute::BufferView<Float> PT_energies;
        luisa::compute::BufferView<DoubletVector3> PT_gradients;
        luisa::compute::BufferView<TripletMatrix3> PT_hessians;

        // EE (Edge-Edge) friction data
        luisa::compute::BufferView<Float> EE_energies;
        luisa::compute::BufferView<DoubletVector3> EE_gradients;
        luisa::compute::BufferView<TripletMatrix3> EE_hessians;

        // PE (Point-Edge) friction data
        luisa::compute::BufferView<Float> PE_energies;
        luisa::compute::BufferView<DoubletVector3> PE_gradients;
        luisa::compute::BufferView<TripletMatrix3> PE_hessians;

        // PP (Point-Point) friction data
        luisa::compute::BufferView<Float> PP_energies;
        luisa::compute::BufferView<DoubletVector3> PP_gradients;
        luisa::compute::BufferView<TripletMatrix3> PP_hessians;
    };

    // Accessor methods for PT data
    luisa::compute::BufferView<Vector4i> PTs() const;
    luisa::compute::BufferView<Float> PT_energies() const;
    luisa::compute::BufferView<DoubletVector3> PT_gradients() const;
    luisa::compute::BufferView<TripletMatrix3> PT_hessians() const;

    // Accessor methods for EE data
    luisa::compute::BufferView<Vector4i> EEs() const;
    luisa::compute::BufferView<Float> EE_energies() const;
    luisa::compute::BufferView<DoubletVector3> EE_gradients() const;
    luisa::compute::BufferView<TripletMatrix3> EE_hessians() const;

    // Accessor methods for PE data
    luisa::compute::BufferView<Vector3i> PEs() const;
    luisa::compute::BufferView<Float> PE_energies() const;
    luisa::compute::BufferView<DoubletVector3> PE_gradients() const;
    luisa::compute::BufferView<TripletMatrix3> PE_hessians() const;

    // Accessor methods for PP data
    luisa::compute::BufferView<Vector2i> PPs() const;
    luisa::compute::BufferView<Float> PP_energies() const;
    luisa::compute::BufferView<DoubletVector3> PP_gradients() const;
    luisa::compute::BufferView<TripletMatrix3> PP_hessians() const;

  protected:
    /**
     * @brief Override to build the frictional contact system
     */
    virtual void do_build(BuildInfo& info) = 0;
    
    /**
     * @brief Override to compute friction energies
     */
    virtual void do_compute_energy(EnergyInfo& info) = 0;
    
    /**
     * @brief Override to assemble friction gradients and Hessians
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
