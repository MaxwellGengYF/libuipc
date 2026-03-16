#include <affine_body/affine_body_animator.h>
#include <affine_body/affine_body_constraint.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/abd_line_search_reporter.h>
#include <utils/report_extent_check.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(AffineBodyAnimator);

void AffineBodyAnimator::do_build(BuildInfo& info)
{
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
    m_impl.global_animator      = &require<GlobalAnimator>();
    auto dt_attr                = world().scene().config().find<Float>("dt");
    m_impl.dt                   = dt_attr->view()[0];
}

void AffineBodyAnimator::add_constraint(AffineBodyConstraint* constraint)
{
    m_impl.constraints.register_sim_system(*constraint);
}

void AffineBodyAnimator::do_init()
{
    m_impl.init(world());
}

void AffineBodyAnimator::do_step()
{
    m_impl.step();
}

void AffineBodyAnimator::Impl::init(backend::WorldVisitor& world)
{
    // sort the constraints by uid
    auto constraint_view = constraints.view();

    std::sort(constraint_view.begin(),
              constraint_view.end(),
              [](const AffineBodyConstraint* a, const AffineBodyConstraint* b)
              { return a->uid() < b->uid(); });

    // setup constraint index and the mapping from uid to index
    for(auto&& [i, constraint] : enumerate(constraint_view))
    {
        auto uid                     = constraint->uid();
        uid_to_constraint_index[uid] = i;
        constraint->m_index          = i;
    }

    constraint_geo_info_offsets_counts.resize(device(), constraints.view().size());
    span<IndexT> constraint_geo_info_counts = constraint_geo_info_offsets_counts.counts();

    auto  geo_slots = world.scene().geometries();
    auto& geo_infos = affine_body_dynamics->m_impl.geo_infos;
    vector<list<AnimatedGeoInfo>> anim_geo_info_buffer;
    anim_geo_info_buffer.resize(constraint_view.size());

    for(auto& info : geo_infos)
    {
        auto  geo_slot = geo_slots[info.geo_slot_index];
        auto& geo      = geo_slot->geometry();
        auto  uids     = geo.meta().find<VectorXu64>(builtin::constraint_uids);

        if(uids)
        {
            const auto& uid_values = uids->view().front();
            for(auto&& uid_value : uid_values)
            {
                auto it = uid_to_constraint_index.find(uid_value);
                UIPC_ASSERT(it != uid_to_constraint_index.end(),
                            "AffineBodyAnimator: No responsible backend SimSystem registered for constraint uid {}",
                            uid_value);
                auto index = it->second;
                anim_geo_info_buffer[index].push_back(info);
            }
        }
    }

    // fill in the counts
    std::ranges::transform(anim_geo_info_buffer,
                           constraint_geo_info_counts.begin(),
                           [](const auto& infos)
                           { return static_cast<IndexT>(infos.size()); });

    // scan to get offsets and total count
    constraint_geo_info_offsets_counts.scan(stream());

    auto total_anim_geo_info_count = constraint_geo_info_offsets_counts.total_count();
    anim_geo_infos.reserve(total_anim_geo_info_count);

    for(auto& infos : anim_geo_info_buffer)
    {
        for(auto& info : infos)
        {
            anim_geo_infos.push_back(info);
        }
    }

    // initialize the constraints
    for(auto constraint : constraint_view)
    {
        FilteredInfo info{this, constraint->m_index};
        constraint->init(info);
    }

    constraint_energy_offsets_counts.resize(device(), constraint_view.size());
    constraint_gradient_offsets_counts.resize(device(), constraint_view.size());
    constraint_hessian_offsets_counts.resize(device(), constraint_view.size());
}

void AffineBodyAnimator::Impl::step()
{
    // Step constraints
    for(auto constraint : constraints.view())
    {
        FilteredInfo info{this, constraint->m_index};
        constraint->step(info);
    }

    span<IndexT> constraint_energy_counts = constraint_energy_offsets_counts.counts();
    span<IndexT> constraint_gradient_counts = constraint_gradient_offsets_counts.counts();
    span<IndexT> constraint_hessian_counts = constraint_hessian_offsets_counts.counts();

    for(auto&& [i, constraint] : enumerate(constraints.view()))
    {
        ReportExtentInfo this_info;
        constraint->report_extent(this_info);

        constraint_energy_counts[i]   = this_info.m_energy_count;
        constraint_gradient_counts[i] = this_info.m_gradient_segment_count;
        constraint_hessian_counts[i]  = this_info.m_hessian_block_count;
    }

    // update the offsets
    constraint_energy_offsets_counts.scan(stream());
    constraint_gradient_offsets_counts.scan(stream());
    constraint_hessian_offsets_counts.scan(stream());
}

void AffineBodyAnimator::compute_energy(ABDLineSearchReporter::ComputeEnergyInfo& info)
{
    for(auto constraint : m_impl.constraints.view())
    {
        ComputeEnergyInfo this_info{&m_impl, constraint->m_index, m_impl.dt, info.energies()};
        constraint->compute_energy(this_info);
    }
}

void AffineBodyAnimator::compute_gradient_hessian(ABDLinearSubsystem::AssembleInfo& info)
{
    // Get the gradient and hessian views from AssembleInfo
    auto grad_view = info.gradients();
    auto hess_view = info.hessians();
    
    for(auto constraint : m_impl.constraints.view())
    {
        // The actual gradient/hessian computation is done by the constraint
        // Pass the raw buffer views - constraints will use subviews based on their offsets
        ComputeGradientHessianInfo this_info{&m_impl,
                                             constraint->m_index,
                                             m_impl.dt,
                                             grad_view.values,
                                             grad_view.indices,
                                             hess_view.values,
                                             hess_view.row_indices,
                                             hess_view.col_indices,
                                             info.gradient_only()};
        constraint->compute_gradient_hessian(this_info);
    }
}

auto AffineBodyAnimator::FilteredInfo::anim_geo_infos() const noexcept
    -> span<const AnimatedGeoInfo>
{
    auto [offset, count] = m_impl->constraint_geo_info_offsets_counts[m_index];

    return span<const AnimatedGeoInfo>{m_impl->anim_geo_infos}.subspan(offset, count);
}

Float AffineBodyAnimator::BaseInfo::substep_ratio() const noexcept
{
    return m_impl->global_animator->substep_ratio();
}

Float AffineBodyAnimator::BaseInfo::dt() const noexcept
{
    return m_dt;
}

luisa::compute::BufferView<Vector12> AffineBodyAnimator::BaseInfo::qs() const noexcept
{
    return m_impl->affine_body_dynamics->m_impl.body_id_to_q.view();
}

luisa::compute::BufferView<Vector12> AffineBodyAnimator::BaseInfo::q_prevs() const noexcept
{
    return m_impl->affine_body_dynamics->m_impl.body_id_to_q_prev.view();
}

luisa::compute::BufferView<ABDJacobiDyadicMass> AffineBodyAnimator::BaseInfo::body_masses() const noexcept
{
    return m_impl->affine_body_dynamics->m_impl.body_id_to_abd_mass.view();
}

luisa::compute::BufferView<IndexT> AffineBodyAnimator::BaseInfo::is_fixed() const noexcept
{
    return m_impl->affine_body_dynamics->m_impl.body_id_to_is_fixed.view();
}

luisa::compute::BufferView<Float> AffineBodyAnimator::ComputeEnergyInfo::energies() const noexcept
{
    auto [offset, count] = m_impl->constraint_energy_offsets_counts[m_index];
    return m_energies.subview(offset, count);
}

luisa::compute::BufferView<Vector12> AffineBodyAnimator::ComputeGradientHessianInfo::gradients() const noexcept
{
    auto [offset, count] = m_impl->constraint_gradient_offsets_counts[m_index];
    return m_gradients.subview(offset, count);
}

luisa::compute::BufferView<IndexT> AffineBodyAnimator::ComputeGradientHessianInfo::gradient_indices() const noexcept
{
    auto [offset, count] = m_impl->constraint_gradient_offsets_counts[m_index];
    return m_gradient_indices.subview(offset, count);
}

luisa::compute::BufferView<Matrix12x12> AffineBodyAnimator::ComputeGradientHessianInfo::hessians() const noexcept
{
    auto [offset, count] = m_impl->constraint_hessian_offsets_counts[m_index];
    return m_hessians.subview(offset, count);
}

luisa::compute::BufferView<IndexT> AffineBodyAnimator::ComputeGradientHessianInfo::hessian_row_indices() const noexcept
{
    auto [offset, count] = m_impl->constraint_hessian_offsets_counts[m_index];
    return m_hessian_row_indices.subview(offset, count);
}

luisa::compute::BufferView<IndexT> AffineBodyAnimator::ComputeGradientHessianInfo::hessian_col_indices() const noexcept
{
    auto [offset, count] = m_impl->constraint_hessian_offsets_counts[m_index];
    return m_hessian_col_indices.subview(offset, count);
}

bool AffineBodyAnimator::ComputeGradientHessianInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void AffineBodyAnimator::ReportExtentInfo::hessian_count(SizeT count) noexcept
{
    m_hessian_block_count = count;
}

void AffineBodyAnimator::ReportExtentInfo::gradient_count(SizeT count) noexcept
{
    m_gradient_segment_count = count;
}

void AffineBodyAnimator::ReportExtentInfo::energy_count(SizeT count) noexcept
{
    m_energy_count = count;
}

void AffineBodyAnimator::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_block_count, name);
}
}  // namespace uipc::backend::luisa

namespace uipc::backend::luisa
{
class AffineBodyAnimatorLinearSubsystemReporter final : public ABDLinearSubsystemReporter
{
  public:
    using ABDLinearSubsystemReporter::ABDLinearSubsystemReporter;
    SimSystemSlot<AffineBodyAnimator> animator;

    virtual void do_build(BuildInfo& info) override
    {
        animator = require<AffineBodyAnimator>();
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        SizeT gradient_count = 0;
        SizeT hessian_count  = 0;

        gradient_count =
            animator->m_impl.constraint_gradient_offsets_counts.total_count();
        if(!info.gradient_only())
            hessian_count =
                animator->m_impl.constraint_hessian_offsets_counts.total_count();

        info.gradient_count(gradient_count);
        info.hessian_count(hessian_count);
    }

    virtual void do_assemble(AssembleInfo& info) override
    {
        animator->compute_gradient_hessian(info);
    }
};

REGISTER_SIM_SYSTEM(AffineBodyAnimatorLinearSubsystemReporter);

class AffineBodyAnimatorLineSearchSubreporter final : public ABDLineSearchSubreporter
{
  public:
    using ABDLineSearchSubreporter::ABDLineSearchSubreporter;
    SimSystemSlot<AffineBodyAnimator> animator;

    virtual void do_build(BuildInfo& info) override
    {
        animator = require<AffineBodyAnimator>();
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        SizeT energy_count =
            animator->m_impl.constraint_energy_offsets_counts.total_count();
        info.energy_count(energy_count);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        animator->compute_energy(info);
    }
};

REGISTER_SIM_SYSTEM(AffineBodyAnimatorLineSearchSubreporter);
}  // namespace uipc::backend::luisa
