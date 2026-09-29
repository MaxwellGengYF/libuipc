#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/abd_linear_subsystem_reporter.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/set.h>
// The test target's include path puts src/backends/none first, so the shell header
// must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <algorithm>
#include <ranges>

/*************************************************************************************************
* T6.1b — InterAffineBodyConstitutionManager (port of
* src/backends/cuda/affine_body/inter_affine_body_constitution_manager.cu).
*
* The manager's creation is gated on the scene's constitution *type* table
* (`InterAffineBody`), which is how the reference's SimSystemCreator behaves: a
* scene without a joint/articulation constitution has no manager at all (so the
* joint machinery costs nothing), and with one the manager is built and
* `SimEngine::init_scene()` initialises it in the reference's step-1 order
* (affine_body_dynamics -> inter_affine_body_constitution_manager ->
* finite_element_method -> global_body_manager).
*
* `Impl::init`'s two orderings and their invariants are documented in the header;
* the bodies are the reference's verbatim, plus the two reporters at the bottom of
* this file (`InterAffineBodyConstitutionLinearSubsystemReporter` ->
* `ABDLinearSubsystemReporter`, and `InterAffineBodyConstitutionLineSearchSubreporter`
* -> T6.4a's `ABDLineSearchSubreporter`).  The line-search subreporter was
* *invalidated* by `require<ABDLineSearchReporter>()` for as long as that reporter
* had no bodies (T6.1b's explicit not-implemented signal); T6.4a completed the
* reporter, so both subreporters in this file are now live and the
* linear-subsystem one is what the T6.1b test drives.
*************************************************************************************************/
namespace uipc::backend
{
template <>
class backend::SimSystemCreator<luisa::InterAffineBodyConstitutionManager>
{
  public:
    static U<luisa::InterAffineBodyConstitutionManager> create(SimEngine& engine)
    {
        auto  scene = dynamic_cast<luisa::SimEngine&>(engine).world().scene();
        auto& types = scene.constitution_tabular().types();

        // inter affine body constitution manager requires
        // affine body and constraint types to be present
        if(types.find(std::string{builtin::InterAffineBody}) == types.end())
        {
            return nullptr;
        }

        return uipc::make_unique<luisa::InterAffineBodyConstitutionManager>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(InterAffineBodyConstitutionManager);

void InterAffineBodyConstitutionManager::do_build()
{
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
    m_impl.dt_attr              = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");
}

void InterAffineBodyConstitutionManager::Impl::init(SceneVisitor& scene)
{
    auto constitution_view = constitutions.view();

    std::ranges::sort(constitution_view,
                      [](const InterAffineBodyConstitution* a, const InterAffineBodyConstitution* b)
                      { return a->uid() < b->uid(); });

    for(auto&& [i, c] : enumerate(constitution_view))
    {
        c->m_index             = i;  // assign index for each constitution
        uid_to_index[c->uid()] = i;  // build map
    }

    constitution_geo_info_offsets_counts.resize(constitution_view.size());
    auto geo_slots = scene.geometries();
    inter_geo_infos.clear();
    constitution_inter_geo_infos.clear();
    inter_geo_infos.reserve(geo_slots.size());

    span<IndexT> constitution_geo_info_counts =
        constitution_geo_info_offsets_counts.counts();

    for(auto&& [i, slot] : enumerate(geo_slots))
    {
        auto& geo      = slot->geometry();
        auto  base_uid = geo.meta().find<U64>(builtin::constitution_uid);

        if(base_uid)
        {
            U64 uid_value = base_uid->view()[0];
            if(uid_to_index.contains(uid_value))
            {
                InterGeoInfo geo_info;
                geo_info.geo_slot_index   = static_cast<IndexT>(i);
                geo_info.geo_id           = slot->id();
                geo_info.constitution_uid = uid_value;
                inter_geo_infos.push_back(geo_info);
            }
        }

        set<U64> constitution_uids;
        if(base_uid)
            constitution_uids.insert(base_uid->view()[0]);

        auto extra_uids = geo.meta().find<VectorXu64>(builtin::extra_constitution_uids);
        if(extra_uids)
        {
            for(auto uid : extra_uids->view().front())
                constitution_uids.insert(uid);
        }

        for(auto uid_value : constitution_uids)
        {
            auto it = uid_to_index.find(uid_value);
            if(it == uid_to_index.end())
                continue;

            InterGeoInfo geo_info;
            geo_info.geo_slot_index   = static_cast<IndexT>(i);
            geo_info.geo_id           = slot->id();
            geo_info.constitution_uid = uid_value;
            constitution_geo_info_counts[it->second]++;
            constitution_inter_geo_infos.push_back(geo_info);
        }
    }

    // Keep the legacy ordering for animator/constraint pipeline.
    std::ranges::stable_sort(inter_geo_infos,
                             [](const InterGeoInfo& a, const InterGeoInfo& b)
                             { return a.constitution_uid < b.constitution_uid; });

    // Stable sort by constitution uid to ensure geometries from the same constitution are grouped together.
    std::ranges::stable_sort(constitution_inter_geo_infos,
                             [](const InterGeoInfo& a, const InterGeoInfo& b)
                             { return a.constitution_uid < b.constitution_uid; });

    // compute offsets
    constitution_geo_info_offsets_counts.scan();

    // thus, the geo_infos are sorted by constitution uid and can use offset/count
    // to get a subspan for each constitution

    UIPC_ASSERT(constitution_geo_info_offsets_counts.total_count()
                    == constitution_inter_geo_infos.size(),
                "Mismatch in geometry count for constitutions: expected {}, found {}",
                constitution_geo_info_offsets_counts.total_count(),
                constitution_inter_geo_infos.size());

    // count geometries

    for(auto&& c : constitution_view)
    {
        FilteredInfo info{this, c->m_index};
        c->init(info);
    }

    constitution_energy_offsets_counts.resize(constitution_view.size());
    constitution_gradient_offsets_counts.resize(constitution_view.size());
    constitution_hessian_offsets_counts.resize(constitution_view.size());
}

void InterAffineBodyConstitutionManager::Impl::report_energy_extent(
    ABDLineSearchReporter::ReportExtentInfo& info)
{
    auto constitution_view = constitutions.view();

    auto counts = constitution_energy_offsets_counts.counts();

    for(auto&& [i, c] : enumerate(constitution_view))
    {
        EnergyExtentInfo extent_info;
        c->report_energy_extent(extent_info);
        counts[i] = static_cast<IndexT>(extent_info.m_energy_count);
    }

    constitution_energy_offsets_counts.scan();
    info.energy_count(constitution_energy_offsets_counts.total_count());
}

void InterAffineBodyConstitutionManager::Impl::compute_energy(
    ABDLineSearchReporter::ComputeEnergyInfo& info)
{
    Float dt                = dt_attr->view()[0];
    auto  constitution_view = constitutions.view();
    for(auto&& c : constitution_view)
    {
        EnergyInfo this_info{this, static_cast<IndexT>(c->m_index), dt, info.energies()};
        c->compute_energy(this_info);
    }
}

void InterAffineBodyConstitutionManager::Impl::report_gradient_hessian_extent(
    ABDLinearSubsystem::ReportExtentInfo& info)
{
    auto constitution_view = constitutions.view();

    auto gradient_counts = constitution_gradient_offsets_counts.counts();
    auto hessian_counts  = constitution_hessian_offsets_counts.counts();

    for(auto&& [i, c] : enumerate(constitution_view))
    {
        GradientHessianExtentInfo extent_info;
        extent_info.m_gradient_only = info.gradient_only();
        c->report_gradient_hessian_extent(extent_info);
        gradient_counts[i] = static_cast<IndexT>(extent_info.m_gradient_count);
        hessian_counts[i]  = static_cast<IndexT>(extent_info.m_hessian_count);
    }

    constitution_gradient_offsets_counts.scan();
    constitution_hessian_offsets_counts.scan();

    info.gradient_count(constitution_gradient_offsets_counts.total_count());
    info.hessian_count(constitution_hessian_offsets_counts.total_count());
}

void InterAffineBodyConstitutionManager::Impl::compute_gradient_hessian(
    ABDLinearSubsystem::AssembleInfo& info)
{
    Float dt                = dt_attr->view()[0];
    auto  constitution_view = constitutions.view();
    for(auto&& c : constitution_view)
    {
        GradientHessianInfo this_info{
            this, static_cast<IndexT>(c->m_index), dt, info.gradients(), info.hessians(), info.gradient_only()};
        c->compute_gradient_hessian(this_info);
    }
}

void InterAffineBodyConstitutionManager::init()
{
    auto scene = world().scene();
    m_impl.init(scene);
}

void InterAffineBodyConstitutionManager::add_constitution(
    InterAffineBodyConstitution* constitution) noexcept
{
    UIPC_ASSERT(constitution != nullptr, "Constitution must not be null");
    check_state(SimEngineState::BuildSystems, "add_constitution");
    m_impl.constitutions.register_sim_system(*constitution);
}

void InterAffineBodyConstitutionManager::GradientHessianExtentInfo::hessian_count(SizeT count) noexcept
{
    m_hessian_count = count;
}

void InterAffineBodyConstitutionManager::GradientHessianExtentInfo::gradient_count(SizeT count) noexcept
{
    m_gradient_count = count;
}

void InterAffineBodyConstitutionManager::EnergyExtentInfo::energy_count(SizeT count) noexcept
{
    m_energy_count = count;
}

lud::LBufferView<Float> InterAffineBodyConstitutionManager::EnergyInfo::energies() const noexcept
{
    auto [offset, count] = m_impl->constitution_energy_offsets_counts[m_index];
    return m_energies.subview(offset, count);
}

lla::LDoubletVectorView<Float, 12> InterAffineBodyConstitutionManager::GradientHessianInfo::gradients() const noexcept
{
    auto [offset, count] = m_impl->constitution_gradient_offsets_counts[m_index];
    return m_gradients.subview(offset, count);
}

lla::LTripletMatrixView<Float, 12, 12> InterAffineBodyConstitutionManager::GradientHessianInfo::hessians() const noexcept
{
    auto [offset, count] = m_impl->constitution_hessian_offsets_counts[m_index];
    return m_hessians.subview(offset, count);
}

bool InterAffineBodyConstitutionManager::GradientHessianInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

span<const InterAffineBodyConstitutionManager::InterGeoInfo> InterAffineBodyConstitutionManager::FilteredInfo::inter_geo_infos() const noexcept
{
    auto [offset, count] = m_impl->constitution_geo_info_offsets_counts[m_index];
    return span{m_impl->constitution_inter_geo_infos}.subspan(offset, count);
}

span<const AffineBodyDynamics::GeoInfo> InterAffineBodyConstitutionManager::FilteredInfo::geo_infos() const noexcept
{
    return m_impl->affine_body_dynamics->m_impl.geo_infos;
}

const AffineBodyDynamics::GeoInfo& InterAffineBodyConstitutionManager::FilteredInfo::geo_info(IndexT geo_id) const noexcept
{
    auto& gid_to_ginfo_index = m_impl->affine_body_dynamics->m_impl.geo_id_to_geo_info_index;
    auto  it                 = gid_to_ginfo_index.find(static_cast<U64>(geo_id));
    UIPC_ASSERT(it != gid_to_ginfo_index.end(),
                "Geometry ID {} does not have any affine body constitution",
                geo_id);
    auto geo_info_index = it->second;
    return m_impl->affine_body_dynamics->m_impl.geo_infos[static_cast<SizeT>(geo_info_index)];
}

IndexT InterAffineBodyConstitutionManager::FilteredInfo::body_id(IndexT geo_id) const noexcept
{
    const auto& info = geo_info(geo_id);
    UIPC_ASSERT(info.body_count == 1, "Body count in geometry must be 1");
    return static_cast<IndexT>(info.body_offset);
}

IndexT InterAffineBodyConstitutionManager::FilteredInfo::body_id(IndexT   geo_id,
                                                                 IndexT instance_id) const noexcept
{
    const auto& info = geo_info(geo_id);
    UIPC_ASSERT(instance_id >= 0 && instance_id < static_cast<IndexT>(info.body_count),
                "Instance ID {} is out of range [0, {}) for geometry {}",
                instance_id,
                info.body_count,
                geo_id);
    return static_cast<IndexT>(info.body_offset) + instance_id;
}

geometry::SimplicialComplex* InterAffineBodyConstitutionManager::FilteredInfo::body_geo(
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

Float InterAffineBodyConstitutionManager::BaseInfo::dt() const noexcept
{
    return m_impl->dt_attr->view()[0];
}

lud::LCBufferView<LVector12> InterAffineBodyConstitutionManager::BaseInfo::qs() const noexcept
{
    return m_impl->affine_body_dynamics->qs();
}

lud::LCBufferView<LVector12> InterAffineBodyConstitutionManager::BaseInfo::q_prevs() const noexcept
{
    return m_impl->affine_body_dynamics->q_prevs();
}

lud::LCBufferView<ABDJacobiDyadicMass> InterAffineBodyConstitutionManager::BaseInfo::body_masses() const noexcept
{
    return m_impl->affine_body_dynamics->body_masses();
}

lud::LCBufferView<IndexT> InterAffineBodyConstitutionManager::BaseInfo::is_fixed() const noexcept
{
    return m_impl->affine_body_dynamics->body_is_fixed();
}
}  // namespace uipc::backend::luisa

namespace uipc::backend::luisa
{
class InterAffineBodyConstitutionLinearSubsystemReporter final : public ABDLinearSubsystemReporter
{
  public:
    using ABDLinearSubsystemReporter::ABDLinearSubsystemReporter;
    SimSystemSlot<InterAffineBodyConstitutionManager> manager;

    void do_build(BuildInfo& info) override
    {
        (void)info;
        manager = require<InterAffineBodyConstitutionManager>();
    }

    void do_init(InitInfo& info) override { (void)info; }

    void do_report_extent(ReportExtentInfo& info) override
    {
        manager->m_impl.report_gradient_hessian_extent(info);
    }

    void do_assemble(AssembleInfo& info) override
    {
        manager->m_impl.compute_gradient_hessian(info);
    }
};

REGISTER_SIM_SYSTEM(InterAffineBodyConstitutionLinearSubsystemReporter);

class InterAffineBodyConstitutionLineSearchSubreporter final : public ABDLineSearchSubreporter
{
  public:
    using ABDLineSearchSubreporter::ABDLineSearchSubreporter;

    SimSystemSlot<InterAffineBodyConstitutionManager> manager;

    void do_build(BuildInfo& info) override
    {
        (void)info;
        manager = require<InterAffineBodyConstitutionManager>();
    }

    void do_init(InitInfo& info) override { (void)info; }

    void do_report_extent(ReportExtentInfo& info) override
    {
        manager->m_impl.report_energy_extent(info);
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        manager->m_impl.compute_energy(info);
    }
};

REGISTER_SIM_SYSTEM(InterAffineBodyConstitutionLineSearchSubreporter);
}  // namespace uipc::backend::luisa
