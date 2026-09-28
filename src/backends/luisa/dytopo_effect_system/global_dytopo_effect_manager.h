#pragma once
/********************************************************************
 * @file global_dytopo_effect_manager.h
 * @brief Global dynamic-topology effect manager (port of
 *        src/backends/cuda/dytopo_effect_system/global_dytopo_effect_manager.h).
 *
 * The public surface (class names, info structs, Impl members, accessors) is kept
 * identical, so cross-system code (the FEM/ABD dytopo receivers, the contact
 * system, the line search) ports by changing only the namespace / include prefix.
 * Device types map per cookbook §3 / §5:
 *
 *   cuda_tool::DeviceTripletMatrix<T,3> -> lla::LTripletMatrix<T,3>
 *   cuda_tool::DeviceDoubletVector<T,3> -> lla::LDoubletVector<T,3>
 *   cuda_tool::DeviceBCOOMatrix<T,3>    -> lla::LBCOOMatrix<T,3>
 *   cuda_tool::DeviceBCOOVector<T,3>    -> lla::LBCOOVector<T,3>
 *   cuda_tool::CTripletMatrixView<T,3>  -> lla::LCTripletMatrixView<T,3,3>
 *   cuda_tool::CDoubletVectorView<T,3>  -> lla::LCDoubletVectorView<T,3>
 * cuda_tool::CBCOOMatrixView<T,3> -> lla::CBCOOMatrixView<T,3,3>
 * cuda_tool::CBCOOVectorView<T,3> -> lla::CBCOOVectorView<T,3>
 *   cuda_tool::DeviceVar<Vector2i>      -> two scalar slots in
 *                                          `Impl::gradient_range` (see below)
 *   MatrixConverter<Float,3>            -> lla::MatrixConverter<Float,3> (T3.3)
 *
 * Deliberate deviations (all reported in the T4.3 report):
 *  - `gradient_range` (a `DeviceVar<Vector2i>`) becomes a two-element
 *    `LDeviceBuffer<IndexT>` holding {x, y}: an Eigen/`LVec` *element* written by
 *    two different threads (k1 writes x and y from different lanes) cannot be
 *    expressed with one register-wide write, and the T3.4 precedent is to split a
 *    small vector-valued device scalar into flat scalars.
 *  - `selected_hessian[N]` (the reference's `VarView<IndexT>{data + N}` "last"
 *    slot) is written by the same thread that sees I == 0, and the select kernel
 *    is dispatched over `max(N, 1)` threads so that the N == 0 case still clears
 *    it (the reference leaves it untouched, which is only safe because its fresh
 *    allocation happens to be zeroed).
 *********************************************************************/
#include <sim_system.h>
#include <energy_component_flags.h>
#include <global_geometry/global_vertex_manager.h>
#include <luisa_device/linear/linear.h>
#include <utils/offset_count_collection.h>
#include <dytopo_effect_system/dytopo_classify_info.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (the alias also exists inside luisa_device; this one
// makes the `lla::` spellings below and in the ported receivers resolve here).
namespace lla = luisa_device::linalg;

class DyTopoEffectReporter;
class DyTopoEffectReceiver;

class GlobalDyTopoEffectManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class GradientHessianExtentInfo
    {
      public:
        bool gradient_only() const { return m_gradient_only; }
        void gradient_count(SizeT count) noexcept { m_gradient_count = count; }
        void hessian_count(SizeT count) noexcept { m_hessian_count = count; }

      private:
        friend class Impl;
        friend class DyTopoEffectReporter;

        bool  m_gradient_only  = false;
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count  = 0;
    };

    class GradientHessianInfo
    {
      public:
        bool gradient_only() const noexcept { return m_gradient_only; }
        lla::LDoubletVectorView<Float, 3> gradients() const noexcept
        {
            return m_gradients;
        }
        lla::LTripletMatrixView<Float, 3, 3> hessians() const noexcept
        {
            return m_hessians;
        }

      private:
        friend class Impl;
        bool                                 m_gradient_only = false;
        lla::LDoubletVectorView<Float, 3>    m_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_hessians;
    };

    class EnergyExtentInfo
    {
      public:
        void energy_count(SizeT count) noexcept { m_energy_count = count; }

      private:
        friend class Impl;
        friend class DyTopoEffectLineSearchReporter;
        SizeT m_energy_count = 0;
    };

    class EnergyInfo
    {
      public:
        lud::LBufferView<Float> energies() const { return m_energies; }
        bool                    is_initial() const { return m_is_initial; }

      private:
        friend class DyTopoEffectLineSearchReporter;
        lud::LBufferView<Float> m_energies;
        bool                    m_is_initial = false;
    };

    using ClassifyInfo = DyTopoClassifyInfo;

    class ClassifiedDyTopoEffectInfo
    {
      public:
        lla::LCDoubletVectorView<Float, 3> gradients() const noexcept
        {
            return m_gradients;
        }
        lla::LCTripletMatrixView<Float, 3, 3> hessians() const noexcept
        {
            return m_hessians;
        }

      private:
        friend class Impl;
        lla::LCDoubletVectorView<Float, 3>    m_gradients;
        lla::LCTripletMatrixView<Float, 3, 3> m_hessians;
    };

    class ComputeDyTopoEffectInfo
    {
      public:
        void gradient_only(bool v) noexcept { m_gradient_only = v; }
        void component_flags(EnergyComponentFlags v) noexcept
        {
            m_component_flags = v;
        }

      private:
        friend class Impl;
        bool                 m_gradient_only   = false;
        EnergyComponentFlags m_component_flags = EnergyComponentFlags::All;
    };

    class Impl
    {
      public:
        void init(WorldVisitor& world);
        void compute_dytopo_effect(ComputeDyTopoEffectInfo& info);
        void _assemble(ComputeDyTopoEffectInfo& info);
        void _convert_matrix();
        void _distribute(ComputeDyTopoEffectInfo& info);

        SimSystemSlot<GlobalVertexManager> global_vertex_manager;

        Float reserve_ratio = 1.1;


        /***********************************************************************
        *                              Reporter                                *
        ***********************************************************************/

        SimSystemSlotCollection<DyTopoEffectReporter> dytopo_effect_reporters;
        SimSystemSlotCollection<DyTopoEffectReporter> contact_reporters;
        SimSystemSlotCollection<DyTopoEffectReporter> non_contact_reporters;

        OffsetCountCollection<IndexT> reporter_energy_offsets_counts;
        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        lla::LTripletMatrix<Float, 3> collected_dytopo_effect_hessian;
        lla::LDoubletVector<Float, 3> collected_dytopo_effect_gradient;

        lla::MatrixConverter<Float, 3> matrix_converter;
        lla::LBCOOMatrix<Float, 3>     sorted_dytopo_effect_hessian;
        lla::LBCOOVector<Float, 3>     sorted_dytopo_effect_gradient;

        /***********************************************************************
        *                               Receiver                               *
        ***********************************************************************/

        SimSystemSlotCollection<DyTopoEffectReceiver> dytopo_effect_receivers;

        /// {x, y} of the reference's `DeviceVar<Vector2i> gradient_range`.
        ///
        /// k1 writes x and y from *different* lanes (`gradient_range->x() = I` /
        /// `->y() = I + 1`), which two scalar slots express directly; a single
        /// vector element would need a whole-element write and the two writes would
        /// clobber each other.
        lud::LDeviceBuffer<IndexT> gradient_range;
        lud::LDeviceBuffer<IndexT> selected_hessian;
        lud::LDeviceBuffer<IndexT> selected_hessian_offsets;

        vector<lla::LTripletMatrix<Float, 3>> classified_dytopo_effect_hessians;
        vector<lla::LDoubletVector<Float, 3>> classified_dytopo_effect_gradients;

        void loose_resize_entries(lla::LTripletMatrix<Float, 3>& m, SizeT size);
        void loose_resize_entries(lla::LDoubletVector<Float, 3>& v, SizeT size);

        template <typename T>
        void loose_resize(lud::LDeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
        }
    };

    lla::CBCOOVectorView<Float, 3> gradients() const noexcept;
    lla::CBCOOMatrixView<Float, 3, 3> hessians() const noexcept;

    void compute_dytopo_effect(ComputeDyTopoEffectInfo& info);

  protected:
    virtual void do_build() override;

  private:
    friend class DyTopoEffectLineSearchReporter;
    void init();

    friend class SimEngine;
    // only be called by SimEngine
    void compute_dytopo_effect();

    friend class DyTopoEffectReporter;
    void add_reporter(DyTopoEffectReporter* reporter);
    friend class DyTopoEffectReceiver;
    void add_receiver(DyTopoEffectReceiver* receiver);

    Impl m_impl;
};
}// namespace uipc::backend::luisa
