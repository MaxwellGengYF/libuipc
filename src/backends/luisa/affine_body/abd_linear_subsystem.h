#pragma once
/********************************************************************
 * @file affine_body/abd_linear_subsystem.h
 * @brief SCOPE NOTE (T6.1b) — the ABD diag linear subsystem's *declaration
 * surface* (port of src/backends/cuda/affine_body/abd_linear_subsystem.h).
 *
 * WHY THIS FILE EXISTS IN T6.1a
 * -----------------------------
 * `AffineBodyConstitution` (in scope here) and `AffineBodyKinetic` (in scope here)
 * declare members whose *parameter types* are nested classes of
 * `ABDLinearSubsystem` — `ComputeGradientHessianInfo` and, for the animator,
 * `AssembleInfo` / `ReportExtentInfo`.  A nested type needs its enclosing class
 * complete, so the interface cannot be deferred (the T5.1 §7.16 rule 8a / T5.3
 * §7.18 rule 3 dependency-cycle policy, same as the FEM scaffold T5.1 carried for
 * T5.4).
 *
 * WHAT IS PORTED HERE (the reference's inline surface)
 *  * the class declaration and its `DiagLinearSubsystem` base,
 *  * `ComputeGradientHessianInfo` — complete: the reference defines its ctor and
 *    accessors *inline in the header*, so nothing is deferred.  **T6.1b corrected
 *    its two buffer types** to the reference's dense `BufferView<Vector12>` /
 *    `BufferView<Matrix12x12>` (T6.1a had used the reporter interface's
 *    doublet/triplet views; the kinetic and the constitutions write
 *    `gradients(body_i) = G12`, so the dense shape is the load-bearing one),
 *  * `diag_hessian()` (an inline accessor over `Impl`),
 *  * the nested `Impl` / `ReportExtentInfo` / `AssembleInfo` *declarations*.
 *
 * WHAT T6.1b ADDED (all of it defined in abd_linear_subsystem.cpp):
 *  * `ReportExtentInfo::{gradient_count, hessian_count, check}`,
 *  * `AssembleInfo`'s ctor + `gradients()/hessians()/gradient_only()`,
 *  * every `Impl` method and every `do_*` override, and
 *  * `REGISTER_SIM_SYSTEM` + `get_uid()` (the reference's `ABDLinearSubsystemUID`
 *    == 0, issue #271).
 *
 * With this file the ABD subsystem is registered, so the global linear system has
 * ABD dofs in every ABD scene: `uid 0 < FEM's uid 1`, i.e. the ABD block comes
 * first in the shared dof space.
 *********************************************************************/
#include <sim_system.h>
#include <linear_system/diag_linear_subsystem.h>
#include <affine_body/affine_body_dynamics.h>
#include <utils/offset_count_collection.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class ABDLinearSubsystemReporter;  // T6.1b
class ABDDyTopoEffectReceiver;     // T6.1b
class AffineBodyVertexReporter;

class ABDLinearSubsystem final : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    lud::LCBufferView<LMatrix12x12> diag_hessian() const noexcept
    {
        return m_impl.diag_hessian.cview();
    }

    /// The info the kinetic system and the constitutions write their element
    /// gradient/Hessian into: **dense per-body buffers** (the reference's
    /// `cuda_tool::BufferView<Vector12>` / `BufferView<Matrix12x12>`), indexed by
    /// the *body* index within the subsystem's window.  T6.1a had mapped these two
    /// members onto `LDoubletVectorView`/`LTripletMatrixView` (the *reporter*
    /// interface's shape); T6.1b corrects that to the reference's dense buffers,
    /// because the kinetic/constitutions write `gradients(body_i) = G12` (measured
    /// against the reference's `ortho_potential_compute_gradient_hessian_kernel`).
    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(bool                           gradient_only,
                                   lud::LBufferView<LVector12>    gradient,
                                   lud::LBufferView<LMatrix12x12> hessians,
                                   Float                          dt) noexcept
            : m_gradient_only(gradient_only)
            , m_gradients(gradient)
            , m_hessians(hessians)
            , m_dt(dt)
        {
        }

        auto gradient_only() const noexcept { return m_gradient_only; }
        auto hessians() const noexcept { return m_hessians; }
        auto gradients() const noexcept { return m_gradients; }
        auto dt() const noexcept { return m_dt; }

      private:
        bool                           m_gradient_only = false;
        lud::LBufferView<LMatrix12x12> m_hessians;
        lud::LBufferView<LVector12>    m_gradients;
        Float                          m_dt = 0.0;
    };

    /// SCOPE NOTE (T6.1b, closed): the three member bodies are the reference's .cu
    /// definitions (`check()` is the `check_report_extent` guard).
    class ReportExtentInfo
    {
      public:
        // DoubletVector12 count
        void gradient_count(SizeT size);
        // TripletMatrix12x12 count
        void hessian_count(SizeT size);
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class ABDLinearSubsystem;
        friend class ABDLinearSubsystemReporter;
        SizeT        m_gradient_count        = 0;
        SizeT        m_hessian_count         = 0;
        bool         m_gradient_only         = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl;

    /// SCOPE NOTE (T6.1b, closed): ctor + accessors are the reference's .cu
    /// definitions.  This is the *reporter* interface: 12x12 blocks with explicit
    /// (body_i, body_j) indices.
    class AssembleInfo
    {
      public:
        AssembleInfo(Impl* impl, IndexT index, bool gradient_only) noexcept;
        lla::LDoubletVectorView<Float, 12>     gradients() const;
        lla::LTripletMatrixView<Float, 12, 12> hessians() const;
        bool                                   gradient_only() const noexcept;

      private:
        friend class ABDLinearSubsystem;

        Impl*  m_impl          = nullptr;
        IndexT m_index         = ~0;
        bool   m_gradient_only = false;
    };

    /// SCOPE NOTE (T6.1b, closed): the whole Impl (the assembly kernels, the
    /// gradient/Hessian windows, the dytopo receiver, the norms) is ported; the
    /// member list is the reference's, so the port is a mechanical edit there.
    class Impl
    {
      public:
        void init();
        void report_extent(GlobalLinearSystem::DiagExtentInfo& info);

        void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
        void receive_init_dof_info(WorldVisitor& w, GlobalLinearSystem::InitDofInfo& info);

        void assemble(GlobalLinearSystem::DiagInfo& info);
        void _assemble_kinetic_shape(IndexT& offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_reporters(IndexT& offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_dytopo_effect(IndexT& offset, GlobalLinearSystem::DiagInfo& info);

        void  accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
        void  retrieve_solution(GlobalLinearSystem::SolutionInfo& info);
        Float diag_norm();
        Float mass_norm();

        SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
        AffineBodyDynamics::Impl& abd() const noexcept;
        SimSystemSlot<ABDDyTopoEffectReceiver>  dytopo_effect_receiver;
        SimSystemSlot<AffineBodyVertexReporter> affine_body_vertex_reporter;

        Float reserve_ratio = 1.5;

        SimSystemSlotCollection<ABDLinearSubsystemReporter> reporters;
        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        lla::LTripletMatrix<Float, 12, 12> reporter_hessians;
        lla::LDoubletVector<Float, 12>     reporter_gradients;

        // intermediate gradient/hessian buffers for kinetic/shape
        lud::LDeviceBuffer<LMatrix12x12> body_id_to_shape_hessian;
        lud::LDeviceBuffer<LVector12>    body_id_to_shape_gradient;
        lud::LDeviceBuffer<LMatrix12x12> body_id_to_kinetic_hessian;
        lud::LDeviceBuffer<LVector12>    body_id_to_kinetic_gradient;

        // diag hessian for preconditioner
        lud::LDeviceBuffer<LMatrix12x12> diag_hessian;
        lud::LDeviceBuffer<Float>        block_norm;
        lud::LDeviceVar<Float>           reduced_norm;

        S<const geometry::AttributeSlot<Float>> dt_attr;
    };

  private:
    virtual void do_build(DiagLinearSubsystem::BuildInfo& info) override;
    virtual void do_init(InitInfo& info) override;

    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) override;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) override;

    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) override;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info) override;
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) override;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) override;
    virtual Float do_diag_norm(GlobalLinearSystem::DiagNormInfo& info) override;
    virtual Float do_mass_norm(GlobalLinearSystem::DiagNormInfo& info) override;

    virtual U64 get_uid() const noexcept override;

    friend class ABDLinearSubsystemReporter;
    void add_reporter(ABDLinearSubsystemReporter* reporter);  // only by ABDLinearSubsystemReporter

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
