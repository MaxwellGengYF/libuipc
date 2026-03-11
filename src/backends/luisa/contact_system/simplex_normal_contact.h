#pragma once
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

        luisa::compute::BufferView<const ContactCoeff> contact_tabular() const;
        luisa::compute::BufferView<const Vector4i>     PTs() const;
        luisa::compute::BufferView<const Vector4i>     EEs() const;
        luisa::compute::BufferView<const Vector3i>     PEs() const;
        luisa::compute::BufferView<const Vector2i>     PPs() const;

        luisa::compute::BufferView<const Float>   thicknesses() const;
        luisa::compute::BufferView<const Vector3> positions() const;
        luisa::compute::BufferView<const Vector3> prev_positions() const;
        luisa::compute::BufferView<const Vector3> rest_positions() const;
        luisa::compute::BufferView<const IndexT>  contact_element_ids() const;
        Float                                     d_hat() const;
        luisa::compute::BufferView<const Float>   d_hats() const;
        Float                                     dt() const;
        Float                                     eps_velocity() const;

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
        luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> m_PT_gradients;
        luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> m_PT_hessians;

        luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> m_EE_gradients;
        luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> m_EE_hessians;

        luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> m_PE_gradients;
        luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> m_PE_hessians;

        luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> m_PP_gradients;
        luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> m_PP_hessians;
        bool m_gradient_only = false;
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

        luisa::compute::BufferView<Float> PT_energies() const noexcept
        {
            return m_PT_energies;
        }
        luisa::compute::BufferView<Float> EE_energies() const noexcept
        {
            return m_EE_energies;
        }
        luisa::compute::BufferView<Float> PE_energies() const noexcept
        {
            return m_PE_energies;
        }
        luisa::compute::BufferView<Float> PP_energies() const noexcept
        {
            return m_PP_energies;
        }

      private:
        friend class SimplexNormalContact;
        luisa::compute::BufferView<Float> m_PT_energies;
        luisa::compute::BufferView<Float> m_EE_energies;
        luisa::compute::BufferView<Float> m_PE_energies;
        luisa::compute::BufferView<Float> m_PP_energies;
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

        Float dt = 0;
        luisa::compute::Buffer<IndexT> selected_count;

        Float reserve_ratio = 1.1;

        template <typename T>
        void loose_resize(luisa::compute::Buffer<T>& buffer, SizeT size)
        {
            if(size > buffer.size())
            {
                // In LuisaCompute, we need to recreate the buffer if we need more capacity
                // The old buffer will be released when the new one is assigned
                buffer = buffer.device().create_buffer<T>(static_cast<size_t>(size * reserve_ratio));
            }
            // Note: LuisaCompute buffers don't have resize like std::vector,
            // so we track logical size separately or recreate with exact size
        }

        luisa::compute::BufferView<const Float>                      PT_energies;
        luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> PT_gradients;
        luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> PT_hessians;

        luisa::compute::BufferView<const Float>                      EE_energies;
        luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> EE_gradients;
        luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> EE_hessians;

        luisa::compute::BufferView<const Float>                      PE_energies;
        luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> PE_gradients;
        luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> PE_hessians;

        luisa::compute::BufferView<const Float>                      PP_energies;
        luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> PP_gradients;
        luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> PP_hessians;
    };

    luisa::compute::BufferView<const Vector4i> PTs() const;
    luisa::compute::BufferView<const Float>    PT_energies() const;
    luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> PT_gradients() const;
    luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> PT_hessians() const;

    luisa::compute::BufferView<const Vector4i> EEs() const;
    luisa::compute::BufferView<const Float>    EE_energies() const;
    luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> EE_gradients() const;
    luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> EE_hessians() const;

    luisa::compute::BufferView<const Vector3i> PEs() const;
    luisa::compute::BufferView<const Float>    PE_energies() const;
    luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> PE_gradients() const;
    luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> PE_hessians() const;

    luisa::compute::BufferView<const Vector2i> PPs() const;
    luisa::compute::BufferView<const Float>    PP_energies() const;
    luisa::compute::BufferView<const luisa::compute::Doublet<Vector3>> PP_gradients() const;
    luisa::compute::BufferView<const luisa::compute::Triplet<Matrix3>> PP_hessians() const;

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
