#pragma once
/********************************************************************
 * @file fem_linear_subsystem.h
 * @brief T5.4a — the FEM diag linear subsystem (port of
 *        src/backends/cuda/finite_element/fem_linear_subsystem.h).
 *
 * This replaces T5.1's reporter-facing scaffold (whose scope note listed exactly
 * this work) and completes the pieces T5.3 could only declare.
 *
 * WHAT IT IS
 * ----------
 * A `DiagLinearSubsystem` that owns the FEM method's slice of the global linear
 * system: it reports the FEM dof extent (`3 * n_vertices`), lays out the shared
 * gradient/Hessian buffers as
 *   `[KineticG | OtherG ... | DyTopoG]` / `[KineticH | OtherH ... | DyTopoH]`
 * (kinetic first, then the reporters in registration order, then the dytopo
 * effect receiver), fans the assembly out to the kinetic system and the
 * reporters, and implements the norm/accuracy/solution hooks.
 *
 * TEST-REACHABLE VS SOLVER-ONLY (the brief's §2)
 * ---------------------------------------------
 * `FEMLinearSubsystem::Impl::{report_init_extent, receive_init_dof_info}` run from
 * `GlobalLinearSystem::init()` (reachable: the port's `init()` is public, see
 * global_linear_system.h), and `Impl::{report_extent, assemble,
 * _assemble_kinetic, _assemble_reporters, _assemble_dytopo_effect}` run from
 * `GlobalLinearSystem::compute_gradient()` (public) in *gradient-only* mode; the
 * full (Hessian) extent/assembly path is reached from `solve()` (T8.3) or from a
 * test-local `GlobalLinearSystem::Impl` instance.
 *
 * DEVIATIONS (T5.4a, all reported):
 *  1. `Impl::assemble`'s `fem().set_dof_info(frame, ...)` uses the port's
 *     `sim_engine->frame()` (the reference's own spelling) — the luisa
 *     `SimEngine` inherits `core::IEngine::frame()`, so the line is unchanged.
 *  2. `diag_norm()`/`mass_norm()` need a device max-reduction: the reference's
 *     `cuda_tool::DeviceReduce().Max` becomes T3.2's
 *     `lla::device_reduce_max(frame_stream(), in, out, n)`.
 *  3. The kinetic/reporter/dytopo scatter kernels are the *shared* generic
 *     routines of `fem_linear_subsystem_function.h` (the brief's §2 testability
 *     requirement), so the kernel lambdas here are thin wrappers.
 *  4. `UID = 1` (the reference's `FEMLinearSubsystemUID`, issue #271) is kept as
 *     a `constexpr` in the .cpp — it is what the uid-ordering rule sorts by.
 *********************************************************************/
#include <sim_system.h>
#include <linear_system/diag_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <utils/offset_count_collection.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class FiniteElementKinetic;
class FEMDyTopoEffectReceiver;
class FEMLinearSubsystemReporter;

class FEMLinearSubsystem final : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    /// T5.3 (pulled forward, verbatim from the reference's
    /// fem_linear_subsystem.h): the info the kinetic system and the constitutions
    /// write their element gradient/Hessian into.  Every accessor is inline in the
    /// reference, so nothing has to be defined in a .cpp.
    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(bool gradient_only,
                                   lla::LDoubletVectorView<Float, 3> gradients,
                                   lla::LTripletMatrixView<Float, 3, 3> hessians,
                                   Float dt) noexcept
            : m_gradient_only(gradient_only)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_dt(dt)
        {
        }

        auto gradient_only() const noexcept { return m_gradient_only; }
        auto gradients() const noexcept { return m_gradients; }
        auto hessians() const noexcept { return m_hessians; }
        auto dt() const noexcept { return m_dt; }

      private:
        bool m_gradient_only = false;
        lla::LDoubletVectorView<Float, 3>    m_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_hessians;
        Float                                m_dt = 0.0;
    };

    class ReportExtentInfo
    {
      public:
        // DoubletVector3 count
        void gradient_count(SizeT size);
        // TripletMatrix3x3 count
        void hessian_count(SizeT size);
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class FEMLinearSubsystem;
        friend class FEMLinearSubsystemReporter;
        SizeT        m_gradient_count        = 0;
        SizeT        m_hessian_count         = 0;
        bool         m_gradient_only         = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl;

    class AssembleInfo
    {
      public:
        AssembleInfo(Impl*                                 impl,
                     IndexT                                index,
                     lla::LTripletMatrixView<Float, 3, 3> hessians,
                     bool                                 gradient_only) noexcept
            : m_impl(impl)
            , m_index(index)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
        {
        }

        lla::LDoubletVectorView<Float, 3>    gradients() const;
        lla::LTripletMatrixView<Float, 3, 3> hessians() const;
        Float                                dt() const noexcept;
        bool                                 gradient_only() const noexcept;

      private:
        friend class FEMLinearSubsystem;

        Impl*                                m_impl  = nullptr;
        IndexT                               m_index = ~0;
        lla::LTripletMatrixView<Float, 3, 3> m_hessians;
        bool                                 m_gradient_only = false;
    };

    class Impl
    {
      public:
        void init();
        void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
        void receive_init_dof_info(WorldVisitor& w, GlobalLinearSystem::InitDofInfo& info);

        void report_extent(GlobalLinearSystem::DiagExtentInfo& info);

        void assemble(GlobalLinearSystem::DiagInfo& info);
        void _assemble_kinetic(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_reporters(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_dytopo_effect(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);

        void  accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
        void  retrieve_solution(GlobalLinearSystem::SolutionInfo& info);
        Float diag_norm(GlobalLinearSystem::DiagNormInfo& info);
        Float mass_norm(GlobalLinearSystem::DiagNormInfo& info);

        SimEngine* sim_engine = nullptr;

        SimSystemSlot<FiniteElementMethod> finite_element_method;
        FiniteElementMethod::Impl&         fem() noexcept
        {
            return finite_element_method->m_impl;
        }

        SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;

        SimSystemSlot<FEMDyTopoEffectReceiver>                  dytopo_effect_receiver;
        SimSystemSlotCollection<FEMLinearSubsystemReporter>     reporters;

        SimSystemSlot<FiniteElementKinetic> kinetic;


        S<const geometry::AttributeSlot<Float>> dt_attr;
        Float                                   reserve_ratio = 1.5;

        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        lla::LDoubletVector<Float, 3> kinetic_gradients;
        lla::LDoubletVector<Float, 3> reporter_gradients;
        lud::LDeviceBuffer<Float>     diag_blocks_norm;
        lud::LDeviceVar<Float>        reduced_diag_norm;

        void loose_resize_entries(lla::LDoubletVector<Float, 3>& v, SizeT size);
    };

  private:
    virtual void do_build(DiagLinearSubsystem::BuildInfo& info) override;
    virtual void do_init(DiagLinearSubsystem::InitInfo& info) override;
    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) override;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info) override;
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) override;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) override;
    virtual Float do_diag_norm(GlobalLinearSystem::DiagNormInfo& info) override;
    virtual Float do_mass_norm(GlobalLinearSystem::DiagNormInfo& info) override;
    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) override;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) override;
    virtual U64 get_uid() const noexcept override;

    friend class FEMLinearSubsystemReporter;
    void add_reporter(FEMLinearSubsystemReporter* reporter);

    friend class FiniteElementKinetic;
    // defined in finite_element_kinetic.cpp, next to its only caller (T5.3)
    void add_kinetic(FiniteElementKinetic* kinetic);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
