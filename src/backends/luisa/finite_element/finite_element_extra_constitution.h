#pragma once
/********************************************************************
 * @file finite_element_extra_constitution.h
 * @brief T5.1/T5.5 — the FEM extra-constitution base (port of
 * src/backends/cuda/finite_element/finite_element_extra_constitution.h).
 *
 * SCOPE NOTE (T5.1): the plan assigns this file to T5.5, but
 * `FiniteElementMethod::_init_extra_constitutions()` iterates the registered
 * extras and calls `c->init()`/`c->uid()`, and the surface/bending constitutions
 * of the frontend land here, so T5.1 ports the interface plus the
 * scene-membership plumbing (which is all the reference's .cu contains apart from
 * the pure info forwarding). T5.5 adds the concrete extra constitutions and the
 * diff reporters on top.
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_elastics.h>

namespace uipc::backend::luisa
{
class FiniteElementExtraConstitution : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        void init(U64 uid, backend::WorldVisitor& world);

        FiniteElementMethod* finite_element_method = nullptr;
        vector<FiniteElementMethod::GeoInfo> geo_infos;
        FiniteElementMethod::Impl& fem() noexcept
        {
            return finite_element_method->m_impl;
        }
    };

    class BuildInfo
    {
      public:
    };

    using ReportExtentInfo = FiniteElementElastics::ReportExtentInfo;

    class FilteredInfo
    {
      public:
        FilteredInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        span<const FiniteElementMethod::GeoInfo> geo_infos() const noexcept;

        template <typename ForEach, typename ViewGetter>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ViewGetter&& view_getter,
                      ForEach&& for_each_action);

        template <typename ForEach>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots, ForEach&& for_each_action);

        span<const Vector3> positions() noexcept;
        span<const Vector3> rest_positions() noexcept;
        span<const Float> thicknesses() noexcept;

      private:
        Impl* m_impl = nullptr;
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl, Float dt)
            : m_impl(impl)
            , m_dt(dt)
        {
        }

        Float dt() const noexcept;

        lud::LCBufferView<LVector3> xs() const noexcept;
        lud::LCBufferView<LVector3> x_bars() const noexcept;
        lud::LCBufferView<IndexT>   is_fixed() const noexcept;
        lud::LCBufferView<Float>    thicknesses() const noexcept;

      protected:
        Impl* m_impl = nullptr;
        Float m_dt;
    };

    class ComputeEnergyInfo : public BaseInfo
    {
      public:
        ComputeEnergyInfo(Impl* impl, FiniteElementElastics::ComputeEnergyInfo* base_info, Float dt)
            : BaseInfo(impl, dt)
            , base_info(base_info)
        {
        }

        auto energies() const noexcept { return base_info->energies(); }

      private:
        FiniteElementElastics::ComputeEnergyInfo* base_info = nullptr;
    };

    class ComputeGradientHessianInfo : public BaseInfo
    {
      public:
        ComputeGradientHessianInfo(Impl* impl,
                                   FiniteElementElastics::ComputeGradientHessianInfo* base_info,
                                   Float dt)
            : BaseInfo(impl, dt)
            , base_info(base_info)
        {
        }

        auto gradient_only() const noexcept
        {
            return base_info->gradient_only();
        }
        auto gradients() const noexcept { return base_info->gradients(); }
        auto hessians() const noexcept { return base_info->hessians(); }

      private:
        FiniteElementElastics::ComputeGradientHessianInfo* base_info = nullptr;
    };

    U64 uid() const noexcept;

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual U64  get_uid() const noexcept = 0;
    virtual void do_init(FilteredInfo& info) = 0;
    virtual void do_report_extent(ReportExtentInfo& info) = 0;
    virtual void do_compute_energy(ComputeEnergyInfo& info) = 0;
    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) = 0;

    friend class FiniteElementExtraConstitutionDiffParmReporter;  // T5.5
    span<const FiniteElementMethod::GeoInfo> geo_infos() const noexcept;

  private:
    friend class FiniteElementConstitutionLinearSubsystemReporter;  // T5.4
    friend class FiniteElementConstitutionLineSearchSubreporter;    // T5.5

    friend class FiniteElementMethod;
    void init();  // only be called by FiniteElementMethod

    virtual void do_build() override final;

    friend class FiniteElementElastics;
    void report_extent(ReportExtentInfo& info);
    void compute_energy(FiniteElementElastics::ComputeEnergyInfo& info);
    void compute_gradient_hessian(FiniteElementElastics::ComputeGradientHessianInfo& info);

    Impl  m_impl;
    SizeT m_index = 0;
};
}  // namespace uipc::backend::luisa

#include "details/finite_element_extra_constitution.inl"
