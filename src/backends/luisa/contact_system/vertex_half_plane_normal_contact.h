#pragma once
/********************************************************************
 * @file vertex_half_plane_normal_contact.h
 * @brief T8.1b — the IPC contact-model *base* for the vertex/half-plane normal contact
 * (port of src/backends/cuda/contact_system/vertex_half_plane_normal_contact.h, 126
 * reference lines).
 *
 * The half-plane twin of `simplex_normal_contact.h`, and structurally simpler: there is
 * **one** primitive type (PH = `{vertex, half_plane_index}`), so each pair contributes one
 * energy, one gradient doublet (3 components, a single vertex) and one Hessian triplet (a
 * full 3x3 block — the assembler writes it with `Grad(I).write(vI, G)` /
 * `Hess(I).write(vI, vI, H)`, i.e. *one* doublet/triplet per pair, not 4/10 like PT).
 * Consequently `do_report_gradient_hessian_extent` publishes `count` gradients and (unless
 * `gradient_only`) `count` Hessians, and `do_assemble` hands the whole manager views to
 * `do_assemble(ContactInfo&)` without slicing.
 *
 * The candidates come from `VertexHalfPlaneTrajectoryFilter::PHs()` (the T7.2a filter), and
 * the *half-plane instance's* vertex index has to be biased by
 * `HalfPlaneVertexReporter::vertex_offset()` before it can index the manager's per-vertex
 * arrays — the reference does that inside its kernels through
 * `BaseInfo::half_plane_vertex_offset()`, and it is why this model requires the
 * `HalfPlaneVertexReporter` slot.
 *
 * TYPE MAPPINGS: identical to `simplex_normal_contact.h` (see its header).  The one addition is
 * `BaseInfo::contact_tabular_size()` (the flat contact tabular's extent, cookbook §7.35
 * rule 1).
 *********************************************************************/
#include <contact_system/contact_reporter.h>
#include <line_search/line_searcher.h>
#include <contact_system/contact_coeff.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>

namespace uipc::backend::luisa
{
class GlobalTrajectoryFilter;
class VertexHalfPlaneTrajectoryFilter;

class VertexHalfPlaneNormalContact : public ContactReporter
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

        lud::LCBufferView<ContactCoeff> contact_tabular() const;
        /// the extent of the flat contact tabular (see the header note)
        IndexT                          contact_tabular_size() const;
        lud::LCBufferView<LVector2i>    PHs() const;
        lud::LCBufferView<LVector3>     positions() const;
        lud::LCBufferView<LVector3>     prev_positions() const;
        lud::LCBufferView<LVector3>     rest_positions() const;
        lud::LCBufferView<Float>        thicknesses() const;
        lud::LCBufferView<IndexT>       contact_element_ids() const;
        lud::LCBufferView<IndexT>       subscene_element_ids() const;
        Float                           d_hat() const;
        lud::LCBufferView<Float>        d_hats() const;
        Float                           dt() const;
        Float                           eps_velocity() const;
        IndexT                          half_plane_vertex_offset() const;

      private:
        friend class VertexHalfPlaneNormalContact;
        Impl* m_impl;
    };

    class ContactInfo : public BaseInfo
    {
      public:
        ContactInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }

        auto gradients() const noexcept { return m_gradients; }
        auto hessians() const noexcept { return m_hessians; }
        auto gradient_only() const noexcept { return m_gradient_only; }

      private:
        friend class VertexHalfPlaneNormalContact;

        lla::LDoubletVectorView<Float, 3> m_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_hessians;
        bool                                   m_gradient_only = false;
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

        lud::LBufferView<Float> energies() const noexcept;

      private:
        friend class VertexHalfPlaneNormalContact;
        lud::LBufferView<Float> m_energies;
    };

    class Impl
    {
      public:
        void compute_energy(EnergyInfo& info);

        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<GlobalContactManager>   global_contact_manager;
        SimSystemSlot<GlobalVertexManager>    global_vertex_manager;
        SimSystemSlot<VertexHalfPlaneTrajectoryFilter> veretx_half_plane_trajectory_filter;
        SimSystemSlot<HalfPlaneVertexReporter> vertex_reporter;

        SizeT                                   PH_count = 0;
        S<const geometry::AttributeSlot<Float>> dt_attr;

        lud::LCBufferView<Float>              energies;
        lla::LCDoubletVectorView<Float, 3>    gradients;
        lla::LCTripletMatrixView<Float, 3, 3> hessians;
    };

    lud::LCBufferView<LVector2i> PHs() const noexcept;
    lud::LCBufferView<Float>     energies() const noexcept;
    lla::LCDoubletVectorView<Float, 3> gradients() const noexcept;
    lla::LCTripletMatrixView<Float, 3, 3> hessians() const noexcept;

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
