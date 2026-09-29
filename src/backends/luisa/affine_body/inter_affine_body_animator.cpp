#include <affine_body/inter_affine_body_animator.h>
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/abd_linear_subsystem_reporter.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <utils/report_extent_check.h>
#include <algorithm>
#include <ranges>

/*************************************************************************************************
* T6.1b — InterAffineBodyAnimator (port of
* src/backends/cuda/affine_body/inter_affine_body_animator.cu; the two
* sub-reporters at the bottom included).
*
* Notes:
*  * `Impl::init` sorts the constraints by uid (assigning `m_index` from the sorted
*    order), then walks the *manager's* `inter_geo_infos` (the legacy ordering, see
*    the manager's scope note) and pushes each constraint's matching geometries
*    into a per-constraint list, which is concatenated in constraint order and
*    indexed through `constraint_geo_info_offsets_counts`.  The `UIPC_ASSERT` on an
*    unclaimed constraint uid is the reference's, verbatim.
*  * `Impl::step` fans `step`/`report_extent` out in the *unsorted*
*    `constraints.view()` order (`m_index` is the sorted index the offset tables
*    are keyed by) -- preserved verbatim.
*  * `FilteredInfo`'s five accessors delegate to the manager's dynamics data with
*    the manager's own bounds checks (the reference's bodies).
*************************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(InterAffineBodyAnimator);

void InterAffineBodyAnimator::do_build(BuildInfo& info)
{
    (void)info;
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
    m_impl.manager              = &require<InterAffineBodyConstitutionManager>();
    m_impl.global_animator      = &require<GlobalAnimator>();
    m_impl.dt_attr              = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");
}

void InterAffineBodyAnimator::add_constraint(InterAffineBodyConstraint* constraint)
{
    m_impl.constraints.register_sim_system(*constraint);
}

void InterAffineBodyAnimator::do_init()
{
    m_impl.init(world());
}

void InterAffineBodyAnimator::do_step()
{
    m_impl.step();
}

void InterAffineBodyAnimator::Impl::init(backend::WorldVisitor& world)
{
    // sort the constraints by uid
    auto constraint_view = constraints.view();

    std::sort(constraint_view.begin(),
              constraint_view.end(),
              [](const InterAffineBodyConstraint* a, const InterAffineBodyConstraint* b)
              { return a->uid() < b->uid(); });

    // setup constraint index and the mapping from uid to index
    for(auto&& [i, constraint] : enumerate(constraint_view))
    {
        auto uid                     = constraint->uid();
        uid_to_constraint_index[uid] = i;
        constraint->m_index          = i;
    }

    constraint_geo_info_offsets_counts.resize(constraints.view().size());

    auto  geo_slots       = world.scene().geometries();
    auto& inter_geo_infos = manager->m_impl.inter_geo_infos;

    vector<list<AnimatedInterGeoInfo>> anim_geo_info_buffer(constraint_view.size());
    span<IndexT> constraint_geo_info_counts = constraint_geo_info_offsets_counts.counts();

    // filter-out the geo infos for each constraint
    for(auto& info : inter_geo_infos)
    {
        auto  geo_slot = geo_slots[info.geo_slot_index];
        auto& geo      = geo_slot->geometry();
        auto  uids     = geo.meta().find<VectorXu64>(builtin::constraint_uids);
        if(uids)
        {
            auto uid_values = uids->view().front();
            for(auto uid_value : uid_values)
            {
                auto it = uid_to_constraint_index.find(uid_value);
                UIPC_ASSERT(it != uid_to_constraint_index.end(),
                            "InterAffineBodyAnimator: No responsible backend SimSystem registered for constraint uid {}",
                            uid_value);
                auto index = it->second;
                anim_geo_info_buffer[index].push_back(info);
            }
        }
    }

    // count the geo infos for each constraint
    std::ranges::transform(anim_geo_info_buffer,
                           constraint_geo_info_counts.begin(),
                           [](const list<AnimatedInterGeoInfo>& infos)
                           { return static_cast<IndexT>(infos.size()); });

    // get offset
    constraint_geo_info_offsets_counts.scan();

    auto total_count = constraint_geo_info_offsets_counts.total_count();
    anim_geo_infos.reserve(total_count);

    // copy the geo infos to the flatten buffer
    for(auto&& infos : anim_geo_info_buffer)
    {
        for(auto& info : infos)
        {
            anim_geo_infos.push_back(info);
        }
    }

    // initialize the constraints
    for(auto constraint : constraints.view())
    {
        FilteredInfo info{this, constraint->m_index};
        constraint->init(info);
    }

    constraint_energy_offsets_counts.resize(constraint_view.size());
    constraint_gradient_offsets_counts.resize(constraint_view.size());
    constraint_hessian_offsets_counts.resize(constraint_view.size());
}

void InterAffineBodyAnimator::Impl::step()
{
    for(auto constraint : constraints.view())
    {
        FilteredInfo info{this, constraint->m_index};
        constraint->step(info);
    }

    span<IndexT> constraint_energy_counts   = constraint_energy_offsets_counts.counts();
    span<IndexT> constraint_gradient_counts = constraint_gradient_offsets_counts.counts();
    span<IndexT> constraint_hessian_counts  = constraint_hessian_offsets_counts.counts();

    for(auto&& [i, constraint] : enumerate(constraints.view()))
    {
        ReportExtentInfo this_info;
        constraint->report_extent(this_info);

        constraint_energy_counts[i]   = this_info.m_energy_count;
        constraint_gradient_counts[i] = this_info.m_gradient_segment_count;
        constraint_hessian_counts[i]  = this_info.m_hessian_block_count;
    }

    // update the offsets
    constraint_energy_offsets_counts.scan();
    constraint_gradient_offsets_counts.scan();
    constraint_hessian_offsets_counts.scan();
}

void InterAffineBodyAnimator::compute_energy(ABDLineSearchReporter::ComputeEnergyInfo& info)
{
    Float dt = m_impl.dt_attr->view()[0];
    for(auto constraint : m_impl.constraints.view())
    {
        ComputeEnergyInfo this_info{&m_impl, constraint->m_index, dt, info.energies()};
        constraint->compute_energy(this_info);
    }
}

void InterAffineBodyAnimator::compute_gradient_hessian(ABDLinearSubsystem::AssembleInfo& info)
{
    Float dt = m_impl.dt_attr->view()[0];
    for(auto constraint : m_impl.constraints.view())
    {
        GradientHessianInfo this_info{
            &m_impl, constraint->m_index, dt, info.gradients(), info.hessians(), info.gradient_only()};
        constraint->compute_gradient_hessian(this_info);
    }
}

span<const InterAffineBodyAnimator::AnimatedInterGeoInfo> InterAffineBodyAnimator::FilteredInfo::anim_inter_geo_infos() const noexcept
{
    auto [offset, count] = m_impl->constraint_geo_info_offsets_counts[m_index];

    return span<const AnimatedInterGeoInfo>{m_impl->anim_geo_infos}.subspan(offset, count);
}

span<const AffineBodyDynamics::GeoInfo> InterAffineBodyAnimator::FilteredInfo::geo_infos() const noexcept
{
    return m_impl->affine_body_dynamics->m_impl.geo_infos;
}

const AffineBodyDynamics::GeoInfo& InterAffineBodyAnimator::FilteredInfo::geo_info(IndexT geo_id) const noexcept
{
    auto& gid_to_ginfo_index = m_impl->affine_body_dynamics->m_impl.geo_id_to_geo_info_index;
    auto  it                 = gid_to_ginfo_index.find(static_cast<U64>(geo_id));
    UIPC_ASSERT(it != gid_to_ginfo_index.end(),
                "Geometry ID {} does not have any affine body constitution",
                geo_id);
    auto geo_info_index = it->second;
    return m_impl->affine_body_dynamics->m_impl.geo_infos[static_cast<SizeT>(geo_info_index)];
}

IndexT InterAffineBodyAnimator::FilteredInfo::body_id(IndexT geo_id) const noexcept
{
    const auto& info = geo_info(geo_id);
    UIPC_ASSERT(info.body_count == 1, "Body count in geometry must be 1");
    return static_cast<IndexT>(info.body_offset);
}

IndexT InterAffineBodyAnimator::FilteredInfo::body_id(IndexT geo_id, IndexT instance_id) const noexcept
{
    const auto& info = geo_info(geo_id);
    UIPC_ASSERT(instance_id >= 0 && instance_id < static_cast<IndexT>(info.body_count),
                "Instance ID {} is out of range [0, {}) for geometry {}",
                instance_id,
                info.body_count,
                geo_id);
    return static_cast<IndexT>(info.body_offset) + instance_id;
}

geometry::SimplicialComplex* InterAffineBodyAnimator::FilteredInfo::body_geo(
    span<S<geometry::GeometrySlot>> geo_slots, IndexT geo_id) const noexcept
{
    const auto& info = geo_info(geo_id);
    UIPC_ASSERT(info.geo_slot_index < static_cast<IndexT>(geo_slots.size()),
                "Geometry slot index {} out of range for geo slots size {}, why can it happen?",
                info.geo_slot_index,
                geo_slots.size());
    return geo_slots[static_cast<SizeT>(info.geo_slot_index)]
        ->geometry()
        .as<geometry::SimplicialComplex>();
}

Float InterAffineBodyAnimator::BaseInfo::substep_ratio() const noexcept
{
    return m_impl->global_animator->substep_ratio();
}

Float InterAffineBodyAnimator::BaseInfo::dt() const noexcept
{
    return m_dt;
}

lud::LCBufferView<LVector12> InterAffineBodyAnimator::BaseInfo::qs() const noexcept
{
    return m_impl->affine_body_dynamics->qs();
}

lud::LCBufferView<LVector12> InterAffineBodyAnimator::BaseInfo::q_prevs() const noexcept
{
    return m_impl->affine_body_dynamics->q_prevs();
}

lud::LCBufferView<ABDJacobiDyadicMass> InterAffineBodyAnimator::BaseInfo::body_masses() const noexcept
{
    return m_impl->affine_body_dynamics->body_masses();
}

lud::LCBufferView<IndexT> InterAffineBodyAnimator::BaseInfo::is_fixed() const noexcept
{
    return m_impl->affine_body_dynamics->body_is_fixed();
}

lud::LBufferView<Float> InterAffineBodyAnimator::ComputeEnergyInfo::energies() const noexcept
{
    auto [offset, count] = m_impl->constraint_energy_offsets_counts[m_index];
    return m_energies.subview(offset, count);
}

lla::LDoubletVectorView<Float, 12> InterAffineBodyAnimator::GradientHessianInfo::gradients() const noexcept
{
    auto [offset, count] = m_impl->constraint_gradient_offsets_counts[m_index];
    return m_gradients.subview(offset, count);
}

lla::LTripletMatrixView<Float, 12, 12> InterAffineBodyAnimator::GradientHessianInfo::hessians() const noexcept
{
    auto [offset, count] = m_impl->constraint_hessian_offsets_counts[m_index];
    return m_hessians.subview(offset, count);
}

bool InterAffineBodyAnimator::GradientHessianInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void InterAffineBodyAnimator::ReportExtentInfo::hessian_count(SizeT count) noexcept
{
    m_hessian_block_count = count;
}

void InterAffineBodyAnimator::ReportExtentInfo::gradient_count(SizeT count) noexcept
{
    m_gradient_segment_count = count;
}

void InterAffineBodyAnimator::ReportExtentInfo::energy_count(SizeT count) noexcept
{
    m_energy_count = count;
}

void InterAffineBodyAnimator::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_block_count, name);
}
}  // namespace uipc::backend::luisa

namespace uipc::backend::luisa
{
class InterAffineBodyAnimatorLinearSubsystemReporter final : public ABDLinearSubsystemReporter
{
  public:
    using ABDLinearSubsystemReporter::ABDLinearSubsystemReporter;
    SimSystemSlot<InterAffineBodyAnimator> animator;

    void do_build(BuildInfo& info) override
    {
        (void)info;
        animator = require<InterAffineBodyAnimator>();
    }

    void do_init(InitInfo& info) override { (void)info; }

    void do_report_extent(ReportExtentInfo& info) override
    {
        SizeT gradient_count =
            animator->m_impl.constraint_gradient_offsets_counts.total_count();

        SizeT hessian_count = 0;
        if(!info.gradient_only())
            hessian_count = animator->m_impl.constraint_hessian_offsets_counts.total_count();

        info.gradient_count(gradient_count);
        info.hessian_count(hessian_count);
    }

    void do_assemble(AssembleInfo& info) override
    {
        animator->compute_gradient_hessian(info);
    }
};

REGISTER_SIM_SYSTEM(InterAffineBodyAnimatorLinearSubsystemReporter);

class InterAffineBodyAnimatorLineSearchSubreporter final : public ABDLineSearchSubreporter
{
  public:
    using ABDLineSearchSubreporter::ABDLineSearchSubreporter;
    SimSystemSlot<InterAffineBodyAnimator> animator;

    void do_build(BuildInfo& info) override
    {
        (void)info;
        animator = require<InterAffineBodyAnimator>();
    }

    void do_init(InitInfo& info) override { (void)info; }

    void do_report_extent(ReportExtentInfo& info) override
    {
        SizeT energy_count =
            animator->m_impl.constraint_energy_offsets_counts.total_count();
        info.energy_count(energy_count);
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        animator->compute_energy(info);
    }
};

REGISTER_SIM_SYSTEM(InterAffineBodyAnimatorLineSearchSubreporter);
}  // namespace uipc::backend::luisa
