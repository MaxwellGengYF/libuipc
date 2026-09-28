#pragma once
/********************************************************************
 * @file finite_element_elastics.h
 * @brief T5.1 — the FEM energy-aggregating system (port of
 * src/backends/cuda/finite_element/finite_element_elastics.h).
 *
 * SCOPE NOTE (T5.1): this file is part of T5.3's *task*, but
 * `FiniteElementMethod::Impl::init()` calls `elastics->init()` (a
 * `SimSystemSlot` dereference, so the system must exist and be valid), and the
 * T5.2 constitution bases are *defined* in terms of this class's info types
 * (`FiniteElementConstitution` aliases its ReportExtentInfo/Compute*Info). T5.1
 * therefore ports the *interface plus the offsets plumbing*
 * (finite_element_elastics.cpp: do_build + init + the subview accessors).
 * Deliberately left to the later tasks, and named here so nothing is dropped:
   * - T5.3: `Impl::assemble(...)` and `Impl::compute_energy(...)` (pure fan-outs
   *   over the constitution collections; the reference defines them in this same
   *   .cu). They are *not declared here*: their parameter types are
   *   `FEMLinearSubsystem::AssembleInfo` /
   *   `FEMLineSearchReporter::ComputeEnergyInfo` -- nested classes of the T5.4/T5.5
   *   headers -- and a nested type cannot be named before its enclosing class is
   *   complete (cookbook §7.16 rule 8a). T5.3 adds the members together with those
   *   headers.
 *   - T5.4/T5.5: `FiniteElementElasticsLinearSubsystemReporter` and
 *     `FiniteElementElasticsLineSearchSubreporter` (the two nested reporter
 *     classes at the bottom of the reference's .cu).
 * The later-task parameter types are forward-declared instead of included, so
 * this header does not depend on the T5.4/T5.5 headers.
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/finite_element_method.h>
#include <luisa_device/linear/linear.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (the alias also exists inside luisa_device; a
// backend header must re-declare it, cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class FiniteElementMethod;
// T5.4 / T5.5 (see the scope note above).
class FEMLinearSubsystem;
class FEMLineSearchReporter;
class FiniteElementElasticsLinearSubsystemReporter;
class FiniteElementElasticsLineSearchSubreporter;

class FiniteElementElastics final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class ReportExtentInfo
    {
      public:
        void energy_count(SizeT count) noexcept { m_energy_count = count; }
        void gradient_count(SizeT count) noexcept { m_gradient_count = count; }
        void hessian_count(SizeT count) noexcept { m_hessian_count = count; }
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class FiniteElementElastics;
        friend class FiniteElementConstitution;
        friend class FiniteElementExtraConstitution;

        SizeT m_energy_count = 0;
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count = 0;
        bool m_gradient_only = false;
        mutable bool m_gradient_only_checked = false;
    };

    class ComputeEnergyInfo
    {
      public:
        ComputeEnergyInfo(Impl* impl, SizeT index, Float dt, lud::LBufferView<Float> energies)
            : m_impl(impl)
            , m_index(index)
            , m_dt(dt)
            , m_energies(energies)
        {
        }

        auto energies() const noexcept
        {
            auto [offset, count] = m_impl->constitution_energy_offsets_counts[m_index];
            return m_energies.subview(offset, count);
        }
        auto dt() const noexcept { return m_dt; }

      private:
        Impl*                 m_impl  = nullptr;
        SizeT                 m_index = 0;
        Float                 m_dt    = 0.0;
        lud::LBufferView<Float> m_energies;
    };

    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(Impl*                       impl,
                                   SizeT                       index,
                                   bool                        gradient_only,
                                   Float                       dt,
                                   lla::LDoubletVectorView<Float, 3> gradients,
                                   lla::LTripletMatrixView<Float, 3, 3> hessians)
            : m_impl(impl)
            , m_index(index)
            , m_gradient_only(gradient_only)
            , m_dt(dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
        {
        }

        auto gradient_only() const noexcept { return m_gradient_only; }
        lla::LDoubletVectorView<Float, 3> gradients() const noexcept;
        lla::LTripletMatrixView<Float, 3, 3> hessians() const noexcept;

        auto dt() const noexcept { return m_dt; }

      private:
        Impl*  m_impl = nullptr;
        SizeT  m_index = 0;
        bool   m_gradient_only = false;
        Float  m_dt = 0.0;
        lla::LDoubletVectorView<Float, 3> m_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_hessians;
    };

    class Impl
    {
      public:
        SimSystemSlot<FiniteElementMethod> finite_element_method;
        FiniteElementMethod::Impl& fem()
        {
            return finite_element_method->m_impl;
        }

        OffsetCountCollection<IndexT> constitution_energy_offsets_counts;
        OffsetCountCollection<IndexT> constitution_gradient_offsets_counts;
        OffsetCountCollection<IndexT> constitution_hessian_offsets_counts;

        // T5.3/T5.4/T5.5 add the two fan-out members the reference declares here:
        //   void assemble(FEMLinearSubsystem::AssembleInfo& info);            // T5.4
        //   void compute_energy(FEMLineSearchReporter::ComputeEnergyInfo&);   // T5.5
        // A *nested* parameter type needs its enclosing class to be complete, and
        // neither FEMLinearSubsystem (T5.4) nor FEMLineSearchReporter (T5.5) is
        // ported yet, so T5.1 omits the two declarations instead of inventing
        // placeholder types. (Precedent: the reference declares them here and
        // defines them in this same .cu -- T5.3's port of finite_element_elastics
        // adds both the declarations and their bodies.)
    };

  private:
    friend class FiniteElementElasticsLinearSubsystemReporter;  // T5.4
    friend class FiniteElementElasticsLineSearchSubreporter;    // T5.5

    virtual void do_build() override;

    friend class FiniteElementMethod;
    void init();  // only be called in FiniteElementMethod

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
