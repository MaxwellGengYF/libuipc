#pragma once
/********************************************************************
 * @file fem_linear_subsystem.h
 * @brief T5.1 (reporter-facing part) — port of
 * src/backends/cuda/finite_element/fem_linear_subsystem.h.
 *
 * SCOPE NOTE (T5.1): `fem_linear_subsystem_reporter` (a *reporter base class*)
 * is in T5.1's file list, and it is defined in terms of the two nested info types
 * below plus `FEMLinearSubsystem::add_reporter()`. `FEMLinearSubsystem` itself —
 * a `DiagLinearSubsystem` with its kernels, the dytopo receiver, the
 * preconditioner hooks and the whole `Impl` — is T5.4's task.
 *
 * T5.1 therefore ports *only* the reporter-facing surface:
 *   - `ReportExtentInfo` (counts + the "gradient_only must be queried" check),
 *   - `AssembleInfo` (the assembly view the reporters fill),
 *   - `add_reporter()`,
 *   - an `Impl` that holds just the reporters collection (T5.4 extends this class
 *     with the rest of the reference's Impl).
 * and defines the four trivial members in fem_linear_subsystem_reporter.cpp
 * (T5.4 moves those definitions into its own fem_linear_subsystem.cpp).
 * Everything else of this header is *not* dropped: it is T5.4's port work.
 *
 * ---------------------------------------------------------------------------
 * T5.3 EXTENSION (the dependency-cycle policy of the T5.3 brief's §1.2)
 * ---------------------------------------------------------------------------
 * `FiniteElementKinetic` cannot even be *declared* without two more members of
 * this class, so T5.3 pulls exactly those forward and nothing else:
 *   - the nested `ComputeGradientHessianInfo` (verbatim from the reference header
 *     with `cuda_tool::` -> `lla::` view types; fully inline there, so no
 *     definition moved),
 *   - `Impl::kinetic` (`SimSystemSlot<FiniteElementKinetic>`),
 *   - `friend class FiniteElementKinetic` + `add_kinetic()` (defined in
 *     finite_element_kinetic.cpp, next to its only caller).
 * T5.4 still owns: `DiagLinearSubsystem` as the base, `do_build/do_init/
 * do_report_extent/do_assemble/do_accuracy_check/do_retrieve_solution/
 * do_diag_norm/do_mass_norm/do_report_init_extent/do_receive_init_dof_info/
 * get_uid`, the dytopo receiver, the preconditioner hooks, every assembly kernel,
 * the definitions of the four `AssembleInfo` accessors, and the move of T5.1's
 * four trivial definitions into its own fem_linear_subsystem.cpp.
 *********************************************************************/
#include <sim_system.h>
#include <luisa_device/linear/linear.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class FEMLinearSubsystemReporter;
class FEMDyTopoEffectReceiver;  // T5.5
class FiniteElementKinetic;     // T5.3

// NOTE: the reference derives this from `DiagLinearSubsystem` (T5.4). T5.1 keeps
// the SimSystem base so that `require<FEMLinearSubsystem>()` (which the reporter
// base calls in do_build()) stays well-formed; T5.4 replaces the base with
// DiagLinearSubsystem, which is itself a SimSystem.
class FEMLinearSubsystem : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    // T5.4: `using DiagLinearSubsystem::DiagLinearSubsystem;`

    /// T5.3: verbatim from the reference's fem_linear_subsystem.h, with the
    /// cuda_tool view types mapped to T3.1's `lla` views (`LDoubletVectorView` is
    /// the reference's `DoubletVectorView`, `LTripletMatrixView<Float,3,3>` its
    /// `TripletMatrixView<Float,3>`).  Every accessor is inline in the reference,
    /// so nothing has to be defined in a .cpp.
    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(bool                                          gradient_only,
                                   lla::LDoubletVectorView<Float, 3>             gradients,
                                   lla::LTripletMatrixView<Float, 3, 3>          hessians,
                                   Float                                         dt) noexcept
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
        friend class FEMLinearSubsystemReporter;
        friend class FEMLinearSubsystem;  // T5.4
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count = 0;
        bool m_gradient_only = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl;

    class AssembleInfo
    {
      public:
        AssembleInfo(Impl* impl,
                     IndexT index,
                     lla::LTripletMatrixView<Float, 3, 3> hessians,
                     bool gradient_only) noexcept
            : m_impl(impl)
            , m_index(index)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
        {
        }

        lla::LDoubletVectorView<Float, 3> gradients() const;
        lla::LTripletMatrixView<Float, 3, 3> hessians() const;
        Float dt() const noexcept;
        bool gradient_only() const noexcept;

      private:
        friend class FEMLinearSubsystem;

        Impl*  m_impl = nullptr;
        IndexT m_index = ~0;
        lla::LTripletMatrixView<Float, 3, 3> m_hessians;
        bool m_gradient_only = false;
    };

    /// T5.1 part of the reference's Impl: the reporters collection that
    /// `add_reporter()` fills. T5.3 adds `kinetic` (it is needed to *declare*
    /// `FiniteElementKinetic`, see the file's scope note).  T5.4 adds every other
    /// member of the reference's Impl (gradients, offset-count collections, the
    /// device containers, `sim_engine`, the vertex reporter, the dytopo receiver,
    /// `dt_attr`, `reserve_ratio`, ...).
    class Impl
    {
      public:
        SimSystemSlotCollection<FEMLinearSubsystemReporter> reporters;

        // T5.3 (see the scope note)
        SimSystemSlot<FiniteElementKinetic> kinetic;
    };

    // !!! T5.4 ports the system class itself (BuildInfo/InitInfo, do_build(), ...)
    // and the remaining Impl members, and moves the definitions of the four
    // members that T5.1 defines in fem_linear_subsystem_reporter.cpp.

  private:
    friend class FEMLinearSubsystemReporter;
    void add_reporter(FEMLinearSubsystemReporter* reporter);

    // T5.3: the reference's `friend class FiniteElementKinetic; void
    // add_kinetic(FiniteElementKinetic*);` (defined in finite_element_kinetic.cpp).
    friend class FiniteElementKinetic;
    void add_kinetic(FiniteElementKinetic* kinetic);

    // T5.4's `FEMLinearSubsystem` owns the real Impl; T5.1 needs an instance to
    // hold `reporters`.
    Impl m_impl;
};
}  // namespace uipc::backend::luisa
