/********************************************************************
 * @file simplex_frictional_contact.h
 * @brief T8.1b — the IPC contact-model *base* for the simplex (PT/EE/PE/PP) frictional
 * contact (port of src/backends/cuda/contact_system/simplex_frictional_contact.h, 187
 * reference lines).
 *
 * It is the frictional twin of `simplex_normal_contact.h`: the same four forwarding
 * overrides (`do_report_energy_extent` / `do_compute_energy` /
 * `do_report_gradient_hessian_extent` / `do_assemble`), the same half-Hessian block sizes
 * (`4*(4+1)/2`, `4*(4+1)/2`, `3*(3+1)/2`, `2*(2+1)/2`) and the same manager-slice layout —
 * the only differences are
 *  * `do_build` first reads `contact/friction/enable` and `throw
 *    SimSystemException("Frictional contact is disabled")` when it is 0 (so the model simply
 *    does not exist in a friction-less scene — the reference's scene gate, cookbook §7.35
 *    rule 11's mechanism), and
 *  * every candidate comes from the trajectory filter's **`friction_*`** accessors
 *    (`BaseInfo::friction_PTs()/...`), i.e. the friction-candidate copies the filters make in
 *    `record_friction_candidates` (T7.2a/T7.2b), not the active-contact set.
 * The reference's `SimplexFrictionalContact::Impl` has **no** `reserve_ratio` /
 * `loose_resize` / `selected_count` (the normal model's leftovers); they are absent here too.
 *
 * TYPE MAPPINGS: identical to `simplex_normal_contact.h` (see its header).
 *********************************************************************/
#include <contact_system/contact_reporter.h>
#include <line_search/line_searcher.h>
#include <contact_system/contact_coeff.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::luisa
{
class SimplexFrictionalContact : public ContactReporter
{
  public:
    using ContactReporter::ContactReporter;
    constexpr static SizeT PTHalfHessianSize = 4 * (4 + 1) / 2;  // 4 vertices, symmetric matrix
    constexpr static SizeT EEHalfHessianSize = 4 * (4 + 1) / 2;  // 4 vertices, symmetric matrix
    constexpr static SizeT PEHalfHessianSize = 3 * (3 + 1) / 2;  // 3 vertices, symmetric matrix
    constexpr static SizeT PPHalfHessianSize = 2 * (2 + 1) / 2;  // 2 vertices, symmetric matrix

    class Impl;

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        lud::LCBufferView<ContactCoeff> contact_tabular() const;
        /// the extent of the flat contact tabular (`index = row * extent + col`); the
        /// reference's `contact_tabular().extent().width` (see the header note)
        IndexT                          contact_tabular_size() const;
        lud::LCBufferView<LVector4i>    friction_PTs() const;
        lud::LCBufferView<LVector4i>    friction_EEs() const;
        lud::LCBufferView<LVector3i>    friction_PEs() const;
        lud::LCBufferView<LVector2i>    friction_PPs() const;

        lud::LCBufferView<Float>   thicknesses() const;
        lud::LCBufferView<LVector3> positions() const;
        lud::LCBufferView<LVector3> prev_positions() const;
        lud::LCBufferView<LVector3> rest_positions() const;
        lud::LCBufferView<IndexT>  contact_element_ids() const;
        Float                      d_hat() const;
        lud::LCBufferView<Float>   d_hats() const;
        Float                      dt() const;
        Float                      eps_velocity() const;

      private:
        friend class SimplexFrictionalContact;
        Impl* m_impl;
    };

    class ContactInfo : public BaseInfo
    {
      public:
        ContactInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }
        auto friction_PT_gradients() const noexcept { return m_PT_gradients; }
        auto friction_PT_hessians() const noexcept { return m_PT_hessians; }
        auto friction_EE_gradients() const noexcept { return m_EE_gradients; }
        auto friction_EE_hessians() const noexcept { return m_EE_hessians; }
        auto friction_PE_gradients() const noexcept { return m_PE_gradients; }
        auto friction_PE_hessians() const noexcept { return m_PE_hessians; }
        auto friction_PP_gradients() const noexcept { return m_PP_gradients; }
        auto friction_PP_hessians() const noexcept { return m_PP_hessians; }
        bool gradient_only() const noexcept { return m_gradient_only; }

      private:
        friend class SimplexFrictionalContact;
        lla::LDoubletVectorView<Float, 3> m_PT_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_PT_hessians;

        lla::LDoubletVectorView<Float, 3> m_EE_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_EE_hessians;

        lla::LDoubletVectorView<Float, 3> m_PE_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_PE_hessians;

        lla::LDoubletVectorView<Float, 3> m_PP_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_PP_hessians;
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

        lud::LBufferView<Float> friction_PT_energies() const noexcept
        {
            return m_PT_energies;
        }
        lud::LBufferView<Float> friction_EE_energies() const noexcept
        {
            return m_EE_energies;
        }
        lud::LBufferView<Float> friction_PE_energies() const noexcept
        {
            return m_PE_energies;
        }
        lud::LBufferView<Float> friction_PP_energies() const noexcept
        {
            return m_PP_energies;
        }

      private:
        friend class SimplexFrictionalContact;
        lud::LBufferView<Float> m_PT_energies;
        lud::LBufferView<Float> m_EE_energies;
        lud::LBufferView<Float> m_PE_energies;
        lud::LBufferView<Float> m_PP_energies;
    };

    class Impl
    {
      public:
        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<GlobalContactManager>   global_contact_manager;
        SimSystemSlot<GlobalVertexManager>    global_vertex_manager;

        SimSystemSlot<SimplexTrajectoryFilter> simplex_trajectory_filter;

        // constraint count

        SizeT PT_count = 0;
        SizeT EE_count = 0;
        SizeT PE_count = 0;
        SizeT PP_count = 0;

        S<const geometry::AttributeSlot<Float>> dt_attr;

        lud::LCBufferView<Float>           PT_energies;
        lla::LCDoubletVectorView<Float, 3> PT_gradients;
        lla::LCTripletMatrixView<Float, 3, 3> PT_hessians;

        lud::LCBufferView<Float>           EE_energies;
        lla::LCDoubletVectorView<Float, 3> EE_gradients;
        lla::LCTripletMatrixView<Float, 3, 3> EE_hessians;

        lud::LCBufferView<Float>           PE_energies;
        lla::LCDoubletVectorView<Float, 3> PE_gradients;
        lla::LCTripletMatrixView<Float, 3, 3> PE_hessians;

        lud::LCBufferView<Float>           PP_energies;
        lla::LCDoubletVectorView<Float, 3> PP_gradients;
        lla::LCTripletMatrixView<Float, 3, 3> PP_hessians;
    };

    lud::LCBufferView<LVector4i> PTs() const;
    lud::LCBufferView<Float>     PT_energies() const;
    lla::LCDoubletVectorView<Float, 3> PT_gradients() const;
    lla::LCTripletMatrixView<Float, 3, 3> PT_hessians() const;

    lud::LCBufferView<LVector4i> EEs() const;
    lud::LCBufferView<Float>     EE_energies() const;
    lla::LCDoubletVectorView<Float, 3> EE_gradients() const;
    lla::LCTripletMatrixView<Float, 3, 3> EE_hessians() const;

    lud::LCBufferView<LVector3i> PEs() const;
    lud::LCBufferView<Float>     PE_energies() const;
    lla::LCDoubletVectorView<Float, 3> PE_gradients() const;
    lla::LCTripletMatrixView<Float, 3, 3> PE_hessians() const;

    lud::LCBufferView<LVector2i> PPs() const;
    lud::LCBufferView<Float>     PP_energies() const;
    lla::LCDoubletVectorView<Float, 3> PP_gradients() const;
    lla::LCTripletMatrixView<Float, 3, 3> PP_hessians() const;

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
