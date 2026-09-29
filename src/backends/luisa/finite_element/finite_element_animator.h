#pragma once
/********************************************************************
 * @file finite_element_animator.h
 * @brief T5.3 — the FEM animator (port of
 * src/backends/cuda/finite_element/finite_element_animator.h).
 *
 * The FEM end of the T4.2 animator: `GlobalAnimator::init()`/`step()` call it once
 * per scene / frame, and it fans `init`/`step` out to its own
 * `FiniteElementConstraint`s (sorted by uid), maintaining the
 * constraint → geometry-range offset tables those constraints index their
 * per-constraint views with.
 *
 * TWO CONSEQUENCES OF REGISTERING IT (cookbook §7.18):
 *  1. It is built for *every* scene of the shared test binary in which the
 *     `FiniteElement` + `Constraint` constitution types both exist.  Its
 *     `do_build` `require<>`s `GlobalAnimator` (through the `Animator` base) and
 *     `FiniteElementMethod`, so a scene with only one of the two invalidates it
 *     (the reference's shutdown idiom).
 *  2. `Impl::init()` walks every FEM geometry's `builtin::constraint_uids` and
 *     **aborts** on a uid no registered `FiniteElementConstraint` claims.  That is
 *     why `constraints/soft_position_constraint.{h,cpp}` (uid 14) had to come with
 *     this file: it is the only `Constraint`-type constitution the ported frontend
 *     can write into a FEM geometry's `constraint_uids` in this task.
 *
 * SCOPE NOTE (T5.3, closed by T5.4a): the reference's `assemble()` was *declared*
 * here but not defined in T5.3, because its body calls
 * `AssembleInfo::{gradients,hessians,dt,gradient_only}()` and those four were
 * declared-but-undefined in T5.1's `fem_linear_subsystem.h` scaffold; nothing
 * ODR-used it then, so the class linked.  **T5.4a defines it** (and the
 * `FiniteElementAnimatorLinearSubsystemReporter` that drives it) at the bottom of
 * finite_element_animator.cpp, next to the line-search subreporter T5.3 had
 * already ported.  The declaration below is unchanged.
 * Everything else -- the uid→index map, the geometry-range offsets, the energy /
 * gradient / Hessian extents and `compute_energy` -- is the reference's.
 *********************************************************************/
#include <animator/animator.h>
#include <finite_element/finite_element_method.h>
#include <line_search/line_searcher.h>
#include <utils/offset_count_collection.h>
#include <finite_element/fem_linear_subsystem_reporter.h>
#include <finite_element/fem_line_search_subreporter.h>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (the alias also exists inside luisa_device; a
// backend header must re-declare it, cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

class FiniteElementConstraint;
class FiniteElementAnimator final : public Animator
{
  public:
    using Animator::Animator;

    using AnimatedGeoInfo = FiniteElementMethod::GeoInfo;

    class Impl;

    class FilteredInfo
    {
      public:
        FilteredInfo(Impl* impl, SizeT index)
            : m_impl(impl)
            , m_index(index)
        {
        }

        span<const AnimatedGeoInfo> anim_geo_infos() const;

        template <typename ForEach, typename ViewGetter>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ViewGetter&&                    view_getter,
                      ForEach&&                       for_each_action) noexcept;

      private:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl, SizeT index, Float dt)
            : m_impl(impl)
            , m_index(index)
            , m_dt(dt)
        {
        }
        Float substep_ratio() const noexcept;
        Float dt() const noexcept;
        lud::LCBufferView<LVector3> xs() const noexcept;
        lud::LCBufferView<LVector3> x_prevs() const noexcept;
        lud::LCBufferView<Float>    masses() const noexcept;
        lud::LCBufferView<IndexT>   is_fixed() const noexcept;

      protected:
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
        Float m_dt    = 0.0;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Impl* impl, SizeT index, Float dt, lud::LBufferView<Float> energies)
            : BaseInfo(impl, index, dt)
            , m_energies(energies)
        {
        }

        lud::LBufferView<Float> energies() const noexcept;

      private:
        lud::LBufferView<Float> m_energies;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(Impl* impl,
                                   SizeT index,
                                   Float dt,
                                   lla::LDoubletVectorView<Float, 3>    gradients,
                                   lla::LTripletMatrixView<Float, 3, 3> hessians,
                                   bool                                 gradient_only)
            : BaseInfo(impl, index, dt)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
        {
        }
        lla::LDoubletVectorView<Float, 3>    gradients() const noexcept;
        lla::LTripletMatrixView<Float, 3, 3> hessians() const noexcept;
        bool                                 gradient_only() const noexcept;

      private:
        lla::LDoubletVectorView<Float, 3>    m_gradients;
        lla::LTripletMatrixView<Float, 3, 3> m_hessians;
        bool                                 m_gradient_only = false;
    };

    class ReportExtentInfo
    {
      public:
        void hessian_count(SizeT count) noexcept;
        void gradient_count(SizeT count) noexcept;
        void energy_count(SizeT count) noexcept;
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

        /// DEVIATION (T5.4b): read-only accessors for the three counts.  The
        /// reference has none (the animator is its only reader), but a unit test
        /// that drives a constraint's `do_report_extent` directly has no way to
        /// observe the 0/0/0 contract otherwise; the setters above keep the
        /// reference's names and signatures.
        [[nodiscard]] SizeT energy_count() const noexcept { return m_energy_count; }
        [[nodiscard]] SizeT gradient_count() const noexcept { return m_gradient_segment_count; }
        [[nodiscard]] SizeT hessian_count() const noexcept { return m_hessian_block_count; }

      private:
        friend class FiniteElementAnimator;
        friend class FiniteElementConstraint;
        SizeT m_hessian_block_count = 0;
        SizeT m_gradient_segment_count = 0;
        SizeT m_energy_count = 0;
        bool  m_gradient_only = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl
    {
      public:
        void init(backend::WorldVisitor& world);
        void step();

        FiniteElementMethod* finite_element_method = nullptr;
        FiniteElementMethod::Impl& fem() const noexcept
        {
            return finite_element_method->m_impl;
        }

        GlobalAnimator* global_animator = nullptr;
        SimSystemSlotCollection<FiniteElementConstraint> constraints;
        unordered_map<U64, SizeT> uid_to_constraint_index;

        vector<AnimatedGeoInfo>       anim_geo_infos;
        OffsetCountCollection<IndexT> constraint_geo_info_offsets_counts;

        OffsetCountCollection<IndexT> constraint_energy_offsets_counts;
        OffsetCountCollection<IndexT> constraint_gradient_offsets_counts;
        OffsetCountCollection<IndexT> constraint_hessian_offsets_counts;
    };

  private:
    friend class FiniteElementConstraint;
    void add_constraint(FiniteElementConstraint* constraint);  // only be called by FiniteElementConstraint

    friend class FiniteElementAnimatorLineSearchSubreporter;
    void compute_energy(FEMLineSearchSubreporter::ComputeEnergyInfo& info);  // only be called by FiniteElementAnimatorLineSearchSubreporter

    friend class FiniteElementAnimatorLinearSubsystemReporter;
    // defined in finite_element_animator.cpp (T5.4a; see the scope note above)
    void assemble(FEMLinearSubsystemReporter::AssembleInfo& info);  // only be called by FEMLinearSubsystem

    Impl m_impl;

    void do_init() override;
    void do_step() override;
    void do_build(BuildInfo& info) override;
};
}  // namespace uipc::backend::luisa


#include "details/finite_element_animator.inl"
