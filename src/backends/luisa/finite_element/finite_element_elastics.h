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
 *   - T5.3 ADDS (see the T5.3 note below): `Impl::compute_energy(...)` and the
 *     `FiniteElementElasticsLineSearchSubreporter` that drives it -- the energy
 *     half of the reference's .cu.  The line-search reporter path is complete in
 *     T5.3 (`FEMLineSearchReporter`/`FEMLineSearchSubreporter` are ported).
 *   - T5.4: `Impl::assemble(...)` and `FiniteElementElasticsLinearSubsystemReporter`
 *     (the reference's first nested reporter class): both need
 *     `FEMLinearSubsystem::AssembleInfo`'s accessor *definitions* and the linear
 *     subsystem's Impl members, which T5.4 owns.  `assemble` is still **not
 *     declared here** (nothing in T5.3 ODR-uses it, and declaring it would add a
 *     hard dependency on the T5.4 header from every constitution).
 * The later-task parameter types are forward-declared instead of included, so
 * this header does not depend on the T5.4/T5.5 headers.
 *
 * ---------------------------------------------------------------------------
 * T5.3 NOTE (the energy half)
 * ---------------------------------------------------------------------------
 * `Impl::compute_energy(FEMLineSearchReporter::ComputeEnergyInfo&)` needs the
 * line-search reporter to be *complete*, so this header now includes
 * `<finite_element/fem_line_search_reporter.h>` (ported by T5.3 itself; it does
 * not include back into the elastics, so there is no cycle).  The body is the
 * reference's pure fan-out over the constitution collections, in the same
 * constitution-then-extra order the offsets were laid out in `init()`.
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/fem_line_search_reporter.h>
#include <luisa_device/linear/linear.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (the alias also exists inside luisa_device; a
// backend header must re-declare it, cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class FiniteElementMethod;
// T5.4 (see the scope note above).
class FEMLinearSubsystem;
class FiniteElementElasticsLinearSubsystemReporter;

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

        // T5.3 adds the energy half of the reference's two fan-outs; T5.4 adds
        //   void assemble(FEMLinearSubsystem::AssembleInfo& info);
        // (see the file's scope note: its parameter type's accessors are T5.4's).
        void compute_energy(FEMLineSearchReporter::ComputeEnergyInfo& info);
    };

  private:
    friend class FiniteElementElasticsLinearSubsystemReporter;  // T5.4
    friend class FiniteElementElasticsLineSearchSubreporter;    // T5.3

    virtual void do_build() override;

    friend class FiniteElementMethod;
    void init();  // only be called in FiniteElementMethod

    Impl m_impl;
};
}  // namespace uipc::backend::luisa
