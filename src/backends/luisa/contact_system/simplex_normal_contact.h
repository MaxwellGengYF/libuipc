#pragma once
/********************************************************************
 * @file simplex_normal_contact.h
 * @brief T8.1b — the IPC contact-model *base* for the simplex (PT/EE/PE/PP) normal
 * contact (port of src/backends/cuda/contact_system/simplex_normal_contact.h, 203
 * reference lines).
 *
 * WHAT IT IS.  A `ContactReporter` (T8.1a) whose four final overrides forward to the two
 * virtuals the *model* implements:
 *   do_report_energy_extent  -> reads the trajectory filter's four candidate counts and
 *                               publishes their sum;
 *   do_compute_energy        -> slices the manager's energy buffer into the four
 *                               per-type sub-views and calls `do_compute_energy(EnergyInfo&)`;
 *   do_report_gradient_hessian_extent -> publishes `4*n_pt + 4*n_ee + 3*n_pe + 2*n_pp`
 *                               gradients and `n_pt*10 + n_ee*10 + n_pe*6 + n_pp*3`
 *                               Hessian triplets (or 0 Hessians when `gradient_only`);
 *   do_assemble              -> slices the manager's gradient/Hessian buffers and calls
 *                               `do_assemble(ContactInfo&)`.
 * The half-Hessian block sizes are the reference's: `4*(4+1)/2 = 10` for PT and EE (four
 * 3-vectors), `3*(3+1)/2 = 6` for PE and `2*(2+1)/2 = 3` for PP — i.e. the *upper triangle*
 * of the per-pair block matrix, which is what `TripletMatrixAssembler::half_block<N>` writes.
 * The `*_gradients` are `DoubletVectorView`s (`4/4/3/2` doublets per pair, one per vertex),
 * the `*_hessians` `TripletMatrixView`s.
 *
 * ITS CANDIDATES COME FROM THE T7.x FILTERS: `SimplexTrajectoryFilter::PTs()/EEs()/PEs()/PPs()`
 * for the normal model, the same base's `friction_*` accessors for the frictional one — that
 * is what M7 was for.  The slot is resolved in `on_init_scene` (the reference's ordering
 * requirement: `GlobalTrajectoryFilter::find<SimplexTrajectoryFilter>()` must run after every
 * filter registered) and the pointer is *not* re-resolved per frame (the reference's shape).
 *
 * TYPE MAPPINGS (cookbook §3/§5, identical to contact_reporter.h):
 * `cuda_tool::CBufferView<T>` -> `lud::LCBufferView<T>` (`LBufferView<T>`; the "C" is the
 * reference's naming for the const flavour, the port has one type),
 * `cuda_tool::BufferView<T>` -> `lud::LBufferView<T>`,
 * `cuda_tool::DoubletVectorView<Float,3>` -> `lla::LDoubletVectorView<Float,3>`,
 * `cuda_tool::CDoubletVectorView<Float,3>` -> `lla::LCDoubletVectorView<Float,3>`,
 * `cuda_tool::TripletMatrixView<Float,3>` -> `lla::LTripletMatrixView<Float,3,3>`,
 * `cuda_tool::CTripletMatrixView<Float,3>` -> `lla::LCTripletMatrixView<Float,3,3>`,
 * `S<const geometry::AttributeSlot<Float>>` (the `dt` config attribute) unchanged,
 * `cuda_tool::DeviceVar<IndexT> selected_count` -> `lud::LDeviceVar<IndexT>`.
 *
 * DOCUMENTED ADDITIONS (all additive, none changes the reference's behaviour):
 *  * `BaseInfo::contact_tabular_size()` — the reference's `CBuffer2DView<ContactCoeff>` was
 *    self-describing; the port's tabular is *flat* (`index = row * extent + col`), so the
 *    extent has to travel with it (cookbook §7.35 rule 1).  The kernels take it as an extra
 *    scalar argument.
 *  * everything else in this header is the reference's surface verbatim, including the
 *    unused `SimSystemSlot<GlobalVertexManager>`/`SimSystemSlot<GlobalTrajectoryFilter>`
 *    members and the `Int::loose_resize` growth policy (`reserve(size * 1.1)`, the T7.2a
 *    `resize_discard` spelling).
 *********************************************************************/
#include <contact_system/contact_reporter.h>
#include <line_search/line_searcher.h>
#include <contact_system/contact_coeff.h>
#include <collision_detection/simplex_trajectory_filter.h>

namespace uipc::backend::luisa
{
class SimplexNormalContact : public ContactReporter
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
        lud::LCBufferView<LVector4i>    PTs() const;
        lud::LCBufferView<LVector4i>    EEs() const;
        lud::LCBufferView<LVector3i>    PEs() const;
        lud::LCBufferView<LVector2i>    PPs() const;

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
        friend class SimplexNormalContact;
        Impl* m_impl;
    };

    class ContactInfo : public BaseInfo
    {
      public:
        ContactInfo(Impl* impl) noexcept
            : BaseInfo(impl)
        {
        }
        auto PT_gradients() const noexcept { return m_PT_gradients; }
        auto PT_hessians() const noexcept { return m_PT_hessians; }
        auto EE_gradients() const noexcept { return m_EE_gradients; }
        auto EE_hessians() const noexcept { return m_EE_hessians; }
        auto PE_gradients() const noexcept { return m_PE_gradients; }
        auto PE_hessians() const noexcept { return m_PE_hessians; }
        auto PP_gradients() const noexcept { return m_PP_gradients; }
        auto PP_hessians() const noexcept { return m_PP_hessians; }
        bool gradient_only() const noexcept { return m_gradient_only; }

      private:
        friend class SimplexNormalContact;
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

        lud::LBufferView<Float> PT_energies() const noexcept
        {
            return m_PT_energies;
        }
        lud::LBufferView<Float> EE_energies() const noexcept
        {
            return m_EE_energies;
        }
        lud::LBufferView<Float> PE_energies() const noexcept
        {
            return m_PE_energies;
        }
        lud::LBufferView<Float> PP_energies() const noexcept
        {
            return m_PP_energies;
        }

      private:
        friend class SimplexNormalContact;
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
        lud::LDeviceVar<IndexT>                 selected_count;

        Float reserve_ratio = 1.1;

        template <typename T>
        void loose_resize(lud::LDeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
        }

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
