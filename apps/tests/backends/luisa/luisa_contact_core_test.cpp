/********************************************************************
 * @file luisa_contact_core_test.cpp
 * @brief T8.1a — the contact system core: the completed GlobalContactManager
 * (contact/subscene tabulars, the ContactCoeff table, the scene-adaptive kappa
 * corridor, the CFL / feasible-step caps), the contact reporter/exporter layer and
 * the adaptive-parameter layer (LuisaCompute backend).  Tag [contact_core].
 *
 * Coverage map (brief §Tests):
 * 1) the ContactCoeff tabular oracle (default policy, corridor, symmetric half,
 *    disabled entries, the not-computable fallback) ..... t81a_contact_tabular_oracle
 * 2) the mask semantics (symmetric contact mask + identity subscene default)
 *    ..................................................... t81a_mask_semantics
 * 3) the kappa corridor (Stiff-GIPC suggestKappa/upperBoundKappa re-derived) ...
 *    ..................................................... t81a_kappa_corridor
 * 4) compute_cfl_condition device-vs-host, both branches (surface / all vertices)
 *    ..................................................... t81a_cfl_condition
 * 5) compute_feasible_step: the four CCD kernels, the no-approach case, the
 *    thickness dependence, the min-reduce ................ t81a_feasible_step
 * 6) the reporting pipeline: a probe ContactReporter through extent -> assemble /
 *    compute-energy, and a probe ContactExporter through the exporter manager + the
 *    frontend ContactSystemFeature readback ...... t81a_reporting_pipeline
 * 7) the adaptive-parameter layer: the base reporter + AdaptiveParameterInfo's
 *    exchange semantics, and the GIPC strategy's resolved state
 *    ..................................................... t81a_adaptive_parameters
 * 8) determinism / repeat-call, the empty case, and the quoted numbers
 *    ..................................................... t81a_determinism_and_empty
 *
 * HOW THE SCENES ARE READ
 * -----------------------
 * Every expectation is derived independently in this file (from the scene inputs the
 * test itself fed in), never from the CUDA backend (which is not host-includable).
 * The ported managers are reached through a probe `SimSystem` (the sibling tests'
 * pattern) and through `GlobalContactManager::impl()` (the documented T8.1a test seam:
 * the reference's CFL / feasible-step / adaptive fan-out entry points are private and
 * only the Phase-8 Newton loop calls them).
 *
 * SCENE GATING (why this file is safe in the shared test binary)
 * ------------------------------------------------------------
 * `REGISTER_SIM_SYSTEM` is process-wide, so every probe system here exists in *every*
 * scene the shared `uipc_test_backend_luisa.exe` builds (including the [filter] and
 * [bvh_filter] scenes).  Each probe system therefore refuses to build unless the scene
 * carries a marker geometry-meta attribute (`t81a_probe`), which is exactly the
 * reference filters' `throw SimSystemException` scene-gating mechanism.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/unit.h>
#include <uipc/core/contact_tabular.h>
#include <uipc/core/subscene_tabular.h>
#include <uipc/core/contact_system_feature.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>

#include <type_define.h>
#include <frame_stream.h>
#include <contact_system/global_contact_manager.h>
#include <contact_system/contact_reporter.h>
#include <contact_system/contact_exporter.h>
#include <contact_system/contact_exporter_manager.h>
#include <contact_system/contact_system_feature.h>
#include <contact_system/adaptive_contact_parameter_reporter.h>
#include <contact_system/adaptive_strategies/gipc_adaptive_parameter_strategy.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h>
#include <dytopo_effect_system/dytopo_effect_line_search_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_body_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <utils/codim_thickness.h>
#include <luisa_device/math/distance/ccd.h>
// The test target's include path contains src/backends/none before src/backends/luisa,
// so the bare <sim_engine.h> would silently resolve to the none backend's shell.
#include <backends/luisa/sim_engine.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
namespace gg  = uipc::backend::luisa;
namespace lud = uipc::backend::luisa::luisa_device;

namespace t81a
{
namespace lcrt = ::luisa::compute;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;

// ===========================================================================
//  Helpers
// ===========================================================================
std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

template <typename T>
std::vector<T> read_device(const lud::LCBufferView<T>& view)
{
    std::vector<T> host(view.size());
    gg::frame_stream().synchronize();
    if(!host.empty())
    {
        view.copy_to(span<T>{host});
        lud::main_stream().synchronize();
    }
    return host;
}

template <typename T>
std::vector<T> read_device_view(const lud::LBufferView<T>& view)
{
    return read_device(view);
}

std::vector<Vector3> read_vector3(const lud::LCBufferView<gg::LVector3>& view)
{
    std::vector<Vector3> host;
    auto raw = read_device(view);
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

/// `LDeviceBuffer::copy_from` / `LBufferView::copy_from` with the test's stream.
template <typename T>
void write_device(const lud::LBufferView<T>& view, std::span<const T> host)
{
    view.copy_from(host, &gg::frame_stream());
    gg::frame_stream().flush();
}

constexpr std::string_view kProbeMarker = "t81a_probe";

/// The marker gate: true only in a [contact_core] scene (see the file header).
bool has_probe_marker(backend::WorldVisitor& world)
{
    for(auto& slot : world.scene().geometries())
    {
        if(slot->geometry().meta().find<IndexT>(std::string{kProbeMarker}))
            return true;
    }
    return false;
}

// ===========================================================================
//  The probe systems (all marker-gated so the shared test binary is unaffected)
// ===========================================================================

/// 1) the reporter probe: a real `ContactReporter` (a `DyTopoEffectReporter`), so it
///    goes through the dytopo-effect manager's extent -> assemble / compute-energy
///    pipeline *and* the contact manager's reporter list.
class ContactCoreProbeReporter final : public gg::ContactReporter
{
  public:
    using ContactReporter::ContactReporter;

    // the reporter's oracle: what it reports, and what it was handed ----
    static SizeT reported_energy_count;
    static SizeT reported_gradient_count;
    static SizeT reported_hessian_count;
    static SizeT seen_energy_size;
    static SizeT seen_gradient_size;
    static SizeT seen_hessian_size;
    static SizeT seen_gradient_offset;
    static SizeT seen_hessian_offset;
    static bool  seen_gradient_only;
    static bool  seen_is_initial;
    static std::vector<Float> energies;

    static void reset()
    {
        reported_energy_count   = 0;
        reported_gradient_count = 0;
        reported_hessian_count  = 0;
        seen_energy_size        = 0;
        seen_gradient_size      = 0;
        seen_hessian_size       = 0;
        seen_gradient_offset    = 0;
        seen_hessian_offset     = 0;
        seen_gradient_only      = false;
        seen_is_initial         = false;
        energies.clear();
    }

  protected:
    void do_build(BuildInfo& info) override
    {
        if(!has_probe_marker(world()))
            throw uipc::backend::SimSystemException("not a [contact_core] probe scene");
    }

    void do_init(InitInfo& info) override {}

    virtual void do_report_energy_extent(gg::GlobalDyTopoEffectManager::EnergyExtentInfo& info) override
    {
        info.energy_count(reported_energy_count);
    }

    virtual void do_report_gradient_hessian_extent(
        gg::GlobalDyTopoEffectManager::GradientHessianExtentInfo& info) override
    {
        seen_gradient_only = info.gradient_only();
        info.gradient_count(reported_gradient_count);
        info.hessian_count(info.gradient_only() ? 0 : reported_hessian_count);
    }

    virtual void do_assemble(gg::GlobalDyTopoEffectManager::GradientHessianInfo& info) override
    {
        auto gradients = info.gradients();
        auto hessians  = info.hessians();
        seen_gradient_size   = static_cast<SizeT>(gradients.doublet_count());
        seen_gradient_offset = static_cast<SizeT>(gradients.doublet_index_offset());
        seen_hessian_size    = static_cast<SizeT>(hessians.triplet_count());
        seen_hessian_offset  = static_cast<SizeT>(hessians.triplet_index_offset());

        // write the reporter's known numbers into the slice the manager handed out
        // (the FEM linear test's host `copy_from` idiom: indices/values are flat)
        if(seen_gradient_size > 0)
        {
            std::vector<std::int32_t> idx(seen_gradient_size);
            std::vector<Float>        vals(seen_gradient_size * 3u);
            for(SizeT i = 0; i < seen_gradient_size; ++i)
            {
                idx[i]                       = static_cast<std::int32_t>(i);
                vals[i * 3u + 0]             = 1.0 + static_cast<Float>(i);
                vals[i * 3u + 1]             = 2.0 + static_cast<Float>(i);
                vals[i * 3u + 2]             = 3.0 + static_cast<Float>(i);
            }
            gradients.indices().subview(seen_gradient_offset, seen_gradient_size)
                .copy_from(span<const std::int32_t>{idx}, &gg::frame_stream());
            gradients.values().subview(seen_gradient_offset * 3u, seen_gradient_size * 3u)
                .copy_from(span<const Float>{vals}, &gg::frame_stream());
        }
        if(seen_hessian_size > 0)
        {
            std::vector<std::int32_t> rows(seen_hessian_size);
            std::vector<std::int32_t> cols(seen_hessian_size);
            std::vector<Float>        vals(seen_hessian_size * 9u, 0.0);
            for(SizeT i = 0; i < seen_hessian_size; ++i)
            {
                rows[i]       = static_cast<std::int32_t>(i);
                cols[i]       = static_cast<std::int32_t>(i);
                vals[i * 9u]  = 7.0 + static_cast<Float>(i);
            }
            hessians.row_indices().subview(seen_hessian_offset, seen_hessian_size)
                .copy_from(span<const std::int32_t>{rows}, &gg::frame_stream());
            hessians.col_indices().subview(seen_hessian_offset, seen_hessian_size)
                .copy_from(span<const std::int32_t>{cols}, &gg::frame_stream());
            hessians.values().subview(seen_hessian_offset * 9u, seen_hessian_size * 9u)
                .copy_from(span<const Float>{vals}, &gg::frame_stream());
        }
    }

    virtual void do_compute_energy(gg::GlobalDyTopoEffectManager::EnergyInfo& info) override
    {
        seen_is_initial = info.is_initial();
        auto e          = info.energies();
        seen_energy_size = static_cast<SizeT>(e.size());
        if(!energies.empty() && e.size() >= energies.size())
        {
            e.subview(0, energies.size())
                .copy_from(span<const Float>{energies}, &gg::frame_stream());
        }
    }
};
REGISTER_SIM_SYSTEM(ContactCoreProbeReporter);

SizeT ContactCoreProbeReporter::reported_energy_count   = 0;
SizeT ContactCoreProbeReporter::reported_gradient_count = 0;
SizeT ContactCoreProbeReporter::reported_hessian_count  = 0;
SizeT ContactCoreProbeReporter::seen_energy_size        = 0;
SizeT ContactCoreProbeReporter::seen_gradient_size      = 0;
SizeT ContactCoreProbeReporter::seen_hessian_size       = 0;
SizeT ContactCoreProbeReporter::seen_gradient_offset    = 0;
SizeT ContactCoreProbeReporter::seen_hessian_offset     = 0;
bool  ContactCoreProbeReporter::seen_gradient_only      = false;
bool  ContactCoreProbeReporter::seen_is_initial         = false;
std::vector<Float> ContactCoreProbeReporter::energies;

/// 2) the exporter probe: one primitive type, writing known numbers per readback.
class ContactCoreProbeExporter final : public gg::ContactExporter
{
  public:
    using ContactExporter::ContactExporter;

    static constexpr std::string_view kPrimType = "t81a_probe_prim";
    static Float energy_value;
    static Float gradient_value;
    static Float hessian_value;
    static std::string seen_energy_prim;
    static std::string seen_gradient_prim;
    static std::string seen_hessian_prim;
    static void reset()
    {
        energy_value   = 0.0;
        gradient_value = 0.0;
        hessian_value  = 0.0;
        seen_energy_prim.clear();
        seen_gradient_prim.clear();
        seen_hessian_prim.clear();
    }

  protected:
    void do_build(BuildInfo& info) override
    {
        if(!has_probe_marker(world()))
            throw uipc::backend::SimSystemException("not a [contact_core] probe scene");
    }

    std::string_view get_prim_type() const noexcept override { return kPrimType; }

    void get_contact_energy(std::string_view prim_type, geometry::Geometry& energy_geo) override
    {
        seen_energy_prim = std::string{prim_type};
        energy_geo.meta().create<Float>("t81a_energy", energy_value);
    }

    void get_contact_gradient(std::string_view prim_type, geometry::Geometry& vert_grad) override
    {
        seen_gradient_prim = std::string{prim_type};
        vert_grad.meta().create<Float>("t81a_gradient", gradient_value);
    }

    void get_contact_hessian(std::string_view prim_type, geometry::Geometry& vert_hess) override
    {
        seen_hessian_prim = std::string{prim_type};
        vert_hess.meta().create<Float>("t81a_hessian", hessian_value);
    }
};
REGISTER_SIM_SYSTEM(ContactCoreProbeExporter);

Float ContactCoreProbeExporter::energy_value   = 0.0;
Float ContactCoreProbeExporter::gradient_value = 0.0;
Float ContactCoreProbeExporter::hessian_value  = 0.0;
std::string ContactCoreProbeExporter::seen_energy_prim;
std::string ContactCoreProbeExporter::seen_gradient_prim;
std::string ContactCoreProbeExporter::seen_hessian_prim;

/// 3) the adaptive reporter probe: exercises the base class contract and
///    `AdaptiveParameterInfo` (incl. the tabular exchange).  Refuses to build in a
///    scene with a *negative* resistance (there the real GIPC strategy owns the slot;
///    two adaptive reporters would trip the manager's one-reporter assert).
class ContactCoreProbeAdaptiveReporter final : public gg::AdaptiveContactParameterReporter
{
  public:
    using AdaptiveContactParameterReporter::AdaptiveContactParameterReporter;

    static bool  called;
    static SizeT seen_tabular_size;
    static Float seen_tabular_kappa_0;
    static bool  first_exchange_returned_original;
    static bool  second_exchange_returned_replacement;
    static bool  manager_saw_new_tabular;
    static bool  restored;
    static gg::GlobalContactManager* contact_manager;
    static void  reset()
    {
        called                                = false;
        seen_tabular_size                     = 0;
        seen_tabular_kappa_0                  = 0.0;
        first_exchange_returned_original       = false;
        second_exchange_returned_replacement   = false;
        manager_saw_new_tabular                = false;
        restored                               = false;
    }

  protected:
    void do_build(BuildInfo& info) override
    {
        if(!has_probe_marker(world()))
            throw uipc::backend::SimSystemException("not a [contact_core] probe scene");
        auto kappas = world().scene().contact_tabular().contact_models().find<Float>("resistance");
        for(auto k : kappas->view())
        {
            if(k < 0.0)
                throw uipc::backend::SimSystemException("adaptive strategy scene, not the probe's");
        }
        contact_manager = &require<gg::GlobalContactManager>();
    }

    void do_init(InitInfo& info) override {}

    void do_compute_parameters(AdaptiveParameterInfo& info) override
    {
        called = true;
        // the manager's tabular, as the adaptive strategy sees it
        auto tabular      = info.contact_tabular();
        seen_tabular_size = static_cast<SizeT>(tabular.size());
        if(tabular.size() > 0)
        {
            auto host            = read_device(tabular);
            seen_tabular_kappa_0 = host[0].kappa;
        }

        // exchange: the returned buffer must be the *previous* one, and the new one
        // must be visible to the manager afterwards
        auto original = info.exchange_contact_tabular(nullptr);
        first_exchange_returned_original = (original != nullptr);

        auto replacement = std::make_shared<lud::LDeviceBuffer<gg::ContactCoeff>>();
        replacement->resize(static_cast<SizeT>(tabular.size()));
        replacement->fill(gg::ContactCoeff{.kappa = 1234.5, .mu = 0.25});
        auto previous = info.exchange_contact_tabular(replacement);
        second_exchange_returned_replacement = (previous == nullptr);

        // read back through the *manager's* accessor (not through `info`)
        if(contact_manager)
        {
            auto now                  = read_device(contact_manager->contact_tabular());
            manager_saw_new_tabular   = !now.empty() && now[0].kappa == 1234.5 && now[0].mu == 0.25;
        }

        // restore the original so the scene's tabular is unchanged (the exchange
        // must hand back the replacement)
        auto back = info.exchange_contact_tabular(original);
        restored  = (back == replacement);
    }
};
REGISTER_SIM_SYSTEM(ContactCoreProbeAdaptiveReporter);

bool  ContactCoreProbeAdaptiveReporter::called                              = false;
SizeT ContactCoreProbeAdaptiveReporter::seen_tabular_size                   = 0;
Float ContactCoreProbeAdaptiveReporter::seen_tabular_kappa_0                = 0.0;
bool  ContactCoreProbeAdaptiveReporter::first_exchange_returned_original    = false;
bool  ContactCoreProbeAdaptiveReporter::second_exchange_returned_replacement = false;
bool  ContactCoreProbeAdaptiveReporter::manager_saw_new_tabular             = false;
bool  ContactCoreProbeAdaptiveReporter::restored                            = false;
gg::GlobalContactManager* ContactCoreProbeAdaptiveReporter::contact_manager = nullptr;

// ===========================================================================
//  The probe filter: publishes exactly the analytic pairs the test wants, so the
//  four feasible-step CCD kernels can be driven with closed-form oracles.
// ===========================================================================
class ContactCoreProbeFilter final : public gg::SimplexTrajectoryFilter
{
  public:
    using SimplexTrajectoryFilter::SimplexTrajectoryFilter;

    static std::vector<Vector4i> PTs_data;
    static std::vector<Vector4i> EEs_data;
    static std::vector<Vector3i> PEs_data;
    static std::vector<Vector2i> PPs_data;
    static void                  clear_pairs()
    {
        PTs_data.clear();
        EEs_data.clear();
        PEs_data.clear();
        PPs_data.clear();
    }

  protected:
    void do_build(BuildInfo& info) override
    {
        if(!has_probe_marker(world()))
            throw uipc::backend::SimSystemException("not a [contact_core] probe scene");
    }

    void do_detect(DetectInfo& info) override {}

    void do_filter_active(FilterActiveInfo& info) override
    {
        auto to_lv4 = [](const std::vector<Vector4i>& src)
        {
            std::vector<gg::LVector4i> out;
            out.reserve(src.size());
            for(auto& v : src)
                out.push_back(gg::to_luisa(v));
            return out;
        };
        auto to_lv3 = [](const std::vector<Vector3i>& src)
        {
            std::vector<gg::LVector3i> out;
            out.reserve(src.size());
            for(auto& v : src)
                out.push_back(gg::to_luisa(v));
            return out;
        };
        auto to_lv2 = [](const std::vector<Vector2i>& src)
        {
            std::vector<gg::LVector2i> out;
            out.reserve(src.size());
            for(auto& v : src)
                out.push_back(gg::to_luisa(v));
            return out;
        };

        auto pt = to_lv4(PTs_data);
        auto ee = to_lv4(EEs_data);
        auto pe = to_lv3(PEs_data);
        auto pp = to_lv2(PPs_data);

        m_pt.copy_from(span<const gg::LVector4i>{pt});
        m_ee.copy_from(span<const gg::LVector4i>{ee});
        m_pe.copy_from(span<const gg::LVector3i>{pe});
        m_pp.copy_from(span<const gg::LVector2i>{pp});

        info.PTs(m_pt.cview());
        info.EEs(m_ee.cview());
        info.PEs(m_pe.cview());
        info.PPs(m_pp.cview());
    }

    void do_filter_toi(FilterTOIInfo& info) override {}

    lud::LCBufferView<gg::LVector2i> candidate_PTs() const noexcept override { return {}; }
    lud::LCBufferView<gg::LVector2i> candidate_EEs() const noexcept override { return {}; }
    lud::LCBufferView<Float>         toi_PTs() const noexcept override { return {}; }
    lud::LCBufferView<Float>         toi_EEs() const noexcept override { return {}; }

  private:
    lud::LDeviceBuffer<gg::LVector4i> m_pt;
    lud::LDeviceBuffer<gg::LVector4i> m_ee;
    lud::LDeviceBuffer<gg::LVector3i> m_pe;
    lud::LDeviceBuffer<gg::LVector2i> m_pp;
};
REGISTER_SIM_SYSTEM(ContactCoreProbeFilter);

std::vector<Vector4i> ContactCoreProbeFilter::PTs_data;
std::vector<Vector4i> ContactCoreProbeFilter::EEs_data;
std::vector<Vector3i> ContactCoreProbeFilter::PEs_data;
std::vector<Vector2i> ContactCoreProbeFilter::PPs_data;

// ===========================================================================
//  The manager probe: hands the ported managers to the test
// ===========================================================================
class ContactCoreProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    static gg::GlobalVertexManager*              vertex_manager;
    static gg::GlobalSimplicialSurfaceManager*   surface_manager;
    static gg::GlobalBodyManager*                body_manager;
    static gg::GlobalContactManager*             contact_manager;
    static gg::GlobalTrajectoryFilter*           trajectory_filter;
    static gg::GlobalDyTopoEffectManager*        dytopo_effect_manager;
    static gg::ContactExporterManager*           exporter_manager;
    static gg::HalfPlane*                        half_plane;
    static gg::HalfPlaneVertexReporter*          half_plane_vertex_reporter;
    static gg::EasyVertexHalfPlaneTrajectoryFilter* easy_half_plane_filter;

  protected:
    void do_build() override
    {
        vertex_manager               = find<gg::GlobalVertexManager>();
        surface_manager              = find<gg::GlobalSimplicialSurfaceManager>();
        body_manager                 = find<gg::GlobalBodyManager>();
        contact_manager              = find<gg::GlobalContactManager>();
        trajectory_filter            = find<gg::GlobalTrajectoryFilter>();
        dytopo_effect_manager        = find<gg::GlobalDyTopoEffectManager>();
        exporter_manager             = find<gg::ContactExporterManager>();
        half_plane                   = find<gg::HalfPlane>();
        half_plane_vertex_reporter   = find<gg::HalfPlaneVertexReporter>();
        easy_half_plane_filter       = find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
    }
};
REGISTER_SIM_SYSTEM(ContactCoreProbe);

gg::GlobalVertexManager*            ContactCoreProbe::vertex_manager             = nullptr;
gg::GlobalSimplicialSurfaceManager* ContactCoreProbe::surface_manager            = nullptr;
gg::GlobalBodyManager*              ContactCoreProbe::body_manager               = nullptr;
gg::GlobalContactManager*           ContactCoreProbe::contact_manager            = nullptr;
gg::GlobalTrajectoryFilter*         ContactCoreProbe::trajectory_filter          = nullptr;
gg::GlobalDyTopoEffectManager*      ContactCoreProbe::dytopo_effect_manager      = nullptr;
gg::ContactExporterManager*         ContactCoreProbe::exporter_manager           = nullptr;
gg::HalfPlane*                      ContactCoreProbe::half_plane                 = nullptr;
gg::HalfPlaneVertexReporter*        ContactCoreProbe::half_plane_vertex_reporter = nullptr;
gg::EasyVertexHalfPlaneTrajectoryFilter* ContactCoreProbe::easy_half_plane_filter   = nullptr;

// ===========================================================================
//  Scene description + builder
// ===========================================================================
struct HPDesc
{
    Vector3 P{0.0, 0.0, 0.0};
    Vector3 N{0.0, 1.0, 0.0};
};

struct MeshDesc
{
    std::vector<Vector3>  positions;
    std::vector<Vector3i> triangles;
    std::vector<Float>    thicknesses;
    std::vector<Float>    d_hats;
};

struct SceneDesc
{
    std::vector<HPDesc> half_planes;
    MeshDesc            mesh;
    bool                has_mesh = true;
    bool                probe_marker = true;

    Float d_hat            = 0.01;
    Float dt               = 0.25;
    Float kappa_eval_scale = 1e-16;
    bool  cfl_enable = false;
    std::string constitution = "ipc";

    // contact tabular
    int   n_contact_elements = 0;  // create this many *user* elements (besides default)
    bool  insert_pair_model  = false;  // insert a model for (e1, e2)
    bool  insert_self_model  = false;  // insert a model for (e1, e1)
    Float model_friction     = 0.5;
    Float model_resistance   = 1.0_GPa;
    bool  model_enable       = true;

    bool  default_user_set     = false;
    Float default_friction     = 0.5;
    Float default_resistance   = 1.0_GPa;
    bool  default_enable       = true;

    // subscene tabular
    int   n_subscenes         = 1;   // user subscenes (a default one always exists)
    bool  enable_subscene_1_2 = false;
};

Scene build_scene(const SceneDesc& desc)
{
    auto config = Scene::default_config();
    config["sanity_check"]["enable"] = 0;
    config["dt"]                      = desc.dt;
    config["contact"]["constitution"] = desc.constitution;
    config["contact"]["d_hat"]        = desc.d_hat;
    config["contact"]["adaptive"]["kappa_eval_scale"] = desc.kappa_eval_scale;
    config["cfl"]["enable"]           = desc.cfl_enable ? 1 : 0;
    Scene scene{config};

    {
        auto& ct = scene.contact_tabular();
        std::vector<ContactElement> elements;
        for(int i = 0; i < desc.n_contact_elements; ++i)
            elements.push_back(ct.create("t81a_e" + std::to_string(i + 1)));

        if(desc.insert_self_model && !elements.empty())
        {
            ct.insert(elements[0], elements[0], desc.model_friction, desc.model_resistance, desc.model_enable, Json::object());
        }
        if(desc.insert_pair_model && elements.size() >= 2)
        {
            ct.insert(elements[0], elements[1], desc.model_friction, desc.model_resistance, desc.model_enable, Json::object());
        }
        if(desc.default_user_set)
        {
            ct.default_model(desc.default_friction, desc.default_resistance, desc.default_enable, Json::object());
        }
    }

    if(desc.n_subscenes > 1)
    {
        auto& st = scene.subscene_tabular();
        std::vector<SubsceneElement> subs;
        for(int i = 0; i < desc.n_subscenes; ++i)
            subs.push_back(st.create("t81a_s" + std::to_string(i + 1)));
        // create() ids start after the always-present default element (id 0), so the
        // user subscenes are 1, 2, ...; an *enabled* (s1, s2) pair differs from the
        // identity default the manager writes first.
        if(desc.enable_subscene_1_2 && subs.size() >= 2)
            st.insert(subs[0], subs[1], true, Json::object());
    }

    for(size_t i = 0; i < desc.half_planes.size(); ++i)
    {
        auto object = scene.objects().create("t81a_ground_" + std::to_string(i));
        const auto& hp = desc.half_planes[i];
        auto ig = halfplane(hp.P, hp.N);
        ig.meta().create<IndexT>(builtin::contact_element_id, 0);
        ig.meta().create<IndexT>(builtin::subscene_element_id, 0);
        if(desc.probe_marker)
            ig.meta().create<IndexT>(std::string{kProbeMarker}, 1);
        object->geometries().create(ig);
    }

    if(desc.has_mesh)
    {
        const auto& m = desc.mesh;
        auto complex = trimesh(span<const Vector3>{m.positions}, span<const Vector3i>{m.triangles});
        label_surface(complex);
        auto verts = complex.vertices();
        auto thickness = verts.create<Float>("test_thickness", 0.0);
        {
            auto v = view(*thickness);
            for(size_t i = 0; i < v.size(); ++i)
                v[i] = i < m.thicknesses.size() ? m.thicknesses[i] : 0.0;
        }
        if(desc.probe_marker)
            complex.meta().create<IndexT>(std::string{kProbeMarker}, 1);
        auto object = scene.objects().create("t81a_mesh");
        object->geometries().create(complex);
    }

    return scene;
}

/// Set the global displacement buffer (the ported engine shell does not run the
/// Newton loop yet, so the test plays the reporter/solver's role).
void set_displacements(const std::vector<Vector3>& per_vertex)
{
    auto* vm = ContactCoreProbe::vertex_manager;
    auto  v  = vm->displacements();
    REQUIRE(v.size() == per_vertex.size());
    std::vector<gg::LVector3> host;
    host.reserve(per_vertex.size());
    for(auto& p : per_vertex)
        host.push_back(gg::to_luisa(p));
    write_device(v, span<const gg::LVector3>{host});
}

void set_thicknesses(const std::vector<Float>& per_vertex)
{
    auto* vm = ContactCoreProbe::vertex_manager;
    auto  t  = vm->thicknesses();
    REQUIRE(t.size() == per_vertex.size());
    write_device(t, span<const Float>{per_vertex});
}

/// The manager's per-vertex d_hat is a *device* buffer filled from the scene's
/// vertex attributes by the vertex reporters (in this test target the sibling
/// `GlobalGeometryTestVertexReporter` reads the `test_d_hat` convention), so a scratch
/// scene's mesh keeps whatever that reporter wrote -- this writes the buffer directly
/// instead, exactly like set_thicknesses().
void set_d_hats(const std::vector<Float>& per_vertex)
{
    auto* vm = ContactCoreProbe::vertex_manager;
    auto d  = vm->d_hats();
    REQUIRE(d.size() == per_vertex.size());
    write_device(d, span<const Float>{per_vertex});
}

// ===========================================================================
//  The independent oracles
// ===========================================================================
struct TabularOracle
{
    std::vector<gg::ContactCoeff> table;  // row-major N x N
    std::vector<IndexT>           mask;
    IndexT                        N = 0;
};

/// Re-derive the reference's `_build_contact_tabular` from the scene's own model
/// attributes + the manager's kappa corridor + the config fallbacks.
TabularOracle tabular_oracle(Scene&       scene,
                             Float        kappa_lower_bound,
                             Float        kappa_upper_bound,
                             Float        cfg_min_kappa,
                             Float        cfg_max_kappa)
{
    auto& ct            = scene.contact_tabular();
    auto  models        = ct.contact_models();
    auto  topo          = models.find<Vector2i>("topo")->view();
    auto  resistance    = models.find<Float>("resistance")->view();
    auto  friction      = models.find<Float>("friction_rate")->view();
    auto  enabled       = models.find<IndexT>("is_enabled")->view();
    auto  N             = ct.element_count();

    TabularOracle out;
    out.N = static_cast<IndexT>(N);
    out.mask.assign(N * N, enabled[0]);

    Float min_kappa = cfg_min_kappa;
    Float max_kappa = cfg_max_kappa;
    if(kappa_lower_bound > 0.0)
        min_kappa = kappa_lower_bound;
    if(kappa_upper_bound > 0.0)
        max_kappa = kappa_upper_bound;

    Float default_kappa = resistance[0];
    if(!ct.default_model_is_user_set())
        default_kappa = min_kappa;
    else if(default_kappa >= 0.0 && (default_kappa < min_kappa || default_kappa > max_kappa))
        default_kappa = std::clamp(default_kappa, min_kappa, max_kappa);

    out.table.assign(N * N, gg::ContactCoeff{.kappa = default_kappa, .mu = friction[0]});

    for(size_t i = 0; i < topo.size(); ++i)
    {
        auto  ids = topo[i];
        Float kk  = (i == 0) ? default_kappa : resistance[i];
        if(kk >= 0.0 && (kk < min_kappa || kk > max_kappa))
            kk = std::clamp(kk, min_kappa, max_kappa);
        gg::ContactCoeff coeff{.kappa = kk, .mu = friction[i]};
        out.table[static_cast<size_t>(ids.x() * N + ids.y())] = coeff;
        out.mask[static_cast<size_t>(ids.x() * N + ids.y())]  = enabled[i];
        out.table[static_cast<size_t>(ids.y() * N + ids.x())] = coeff;
        out.mask[static_cast<size_t>(ids.y() * N + ids.x())]  = enabled[i];
    }
    return out;
}

/// recompute Stiff-GIPC's suggestKappa/upperBoundKappa with the dt^2 conversion
struct CorridorOracle
{
    Float lower = -1.0;
    Float upper = -1.0;
};

CorridorOracle corridor_oracle(Float diag, Float d_hat, Float scale, Float dt, Float mean_mass)
{
    constexpr Float min_kappa_coef = 1e11;
    CorridorOracle  out;
    if(!(diag > 0.0))
        return out;
    Float d    = scale * diag * diag;
    Float dHat = d_hat * d_hat;
    if(!(d > 0.0 && dHat > 0.0 && dt > 0.0))
        return out;
    Float t   = d - dHat;
    Float H_b = -2.0 * std::log(d / dHat) - 4.0 * t / d + (t * t) / (d * d);
    if(!(std::isfinite(H_b) && H_b > 0.0))
        return out;
    Float base = min_kappa_coef / (4.0 * d * H_b);
    if(mean_mass > 0.0)
        base *= mean_mass;
    out.lower = base / (dt * dt);
    out.upper = 100.0 * out.lower;
    return out;
}

/// the feasible-step closed form, re-derived from the additive CCD loop:
/// the loop steps by `(1 - eta) * (d - thickness) / maxDispMag` and stops one step
/// after the gap drops below `eta * gap0`, so the returned `toc` is
///     (1 - eta) * (d0 - thickness) / maxDispMag = (1 - eta) * (d0 - thickness) / (d0 - d1)
Float feasible_toi_oracle(Float d0, Float d1, Float thickness, Float eta = 0.2)
{
    return (1.0 - eta) * (d0 - thickness) / (d0 - d1);
}
}// namespace t81a

using namespace t81a;

// ===========================================================================
//  1) the ContactCoeff tabular oracle
// ===========================================================================
TEST_CASE("t81a_contact_tabular_oracle", "[contact_core]")
{
    test_runtime();

    // ---- (a) no model, default not user-set, corridor computable ----------
    {
        SceneDesc desc;
        desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_tabular_a")};
        World  world{engine};
        world.init(scene);

        auto* cm = ContactCoreProbe::contact_manager;
        REQUIRE(cm != nullptr);

        auto cfg_min = scene.config().find<Float>("contact/adaptive/min_kappa")->view()[0];
        auto cfg_max = scene.config().find<Float>("contact/adaptive/max_kappa")->view()[0];

        // the corridor is computable in this scene (d_hat > 0, diag > 0, dt > 0)
        CHECK(cm->kappa_lower_bound() > 0.0);
        CHECK(cm->kappa_upper_bound() > 0.0);
        CHECK(cm->kappa_upper_bound() == 100.0 * cm->kappa_lower_bound());

        auto oracle = tabular_oracle(scene, cm->kappa_lower_bound(), cm->kappa_upper_bound(), cfg_min, cfg_max);
        auto N      = cm->contact_tabular_size();
        CHECK(N == 1);
        CHECK(oracle.N == 1);

        auto table = read_device(cm->contact_tabular());
        auto mask  = read_device(cm->contact_mask_tabular());
        REQUIRE(table.size() == 1);
        REQUIRE(mask.size() == 1);

        // default not user-set -> default kappa = the corridor's lower bound
        CHECK(table[0].kappa == cm->kappa_lower_bound());
        CHECK(table[0].kappa == oracle.table[0].kappa);
        CHECK(table[0].mu == oracle.table[0].mu);
        CHECK(table[0].mu == 0.5);      // the frontend's default friction_rate
        CHECK(mask[0] == oracle.mask[0]);
        CHECK(mask[0] == 1);            // default model enabled
        CHECK(cm->contact_mask_size() == 1);
        CHECK(cm->subscene_mask_size() == 1);
        // the default kappa is nowhere near the config fallback in this scene
        CHECK(table[0].kappa != cfg_min);
        std::printf("[contact_core] tabular(a): kappa=%g corridor=[%g,%g] mu=%g\n",
                    table[0].kappa, cm->kappa_lower_bound(), cm->kappa_upper_bound(), table[0].mu);
    }

    // ---- (b) the corridor is NOT computable -> the config fallback rules ----
    {
        SceneDesc desc;
        desc.kappa_eval_scale = 1e6;  // d >> dHat -> H_b < 0 -> not computable
        desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_tabular_b")};
        World  world{engine};
        world.init(scene);

        auto* cm = ContactCoreProbe::contact_manager;
        REQUIRE(cm != nullptr);
        CHECK(cm->d_hat() == 0.01);
        CHECK(cm->kappa_lower_bound() == -1.0);  // the "not computable" sentinel
        CHECK(cm->kappa_upper_bound() == -1.0);

        auto cfg_min = scene.config().find<Float>("contact/adaptive/min_kappa")->view()[0];
        auto cfg_max = scene.config().find<Float>("contact/adaptive/max_kappa")->view()[0];
        CHECK(cfg_min == 100.0_MPa);
        CHECK(cfg_max == 100.0_GPa);

        auto oracle = tabular_oracle(scene, cm->kappa_lower_bound(), cm->kappa_upper_bound(), cfg_min, cfg_max);
        auto table  = read_device(cm->contact_tabular());
        REQUIRE(table.size() == 1);
        CHECK(table[0].kappa == cfg_min);   // default not user-set -> fallback min
        CHECK(table[0].kappa == oracle.table[0].kappa);
    }

    // ---- (c) default user-set, in-range / out-of-range / adaptive ----------
    {
        SceneDesc desc;
        desc.kappa_eval_scale    = 1e6;  // isolate the *default policy* from the corridor
        desc.default_user_set    = true;
        desc.default_friction    = 0.7;
        desc.default_resistance  = 1.0_GPa;  // above cfg_max=100 GPa? no: in range
        desc.default_enable      = false;
        desc.mesh.positions      = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles      = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_tabular_c")};
        World  world{engine};
        world.init(scene);

        auto* cm = ContactCoreProbe::contact_manager;
        REQUIRE(cm != nullptr);
        auto cfg_min = scene.config().find<Float>("contact/adaptive/min_kappa")->view()[0];
        auto cfg_max = scene.config().find<Float>("contact/adaptive/max_kappa")->view()[0];
        CHECK(scene.contact_tabular().default_model_is_user_set());

        auto oracle = tabular_oracle(scene, cm->kappa_lower_bound(), cm->kappa_upper_bound(), cfg_min, cfg_max);
        auto table  = read_device(cm->contact_tabular());
        auto mask   = read_device(cm->contact_mask_tabular());
        REQUIRE(table.size() == 1);
        // 1 GPa is inside [100 MPa, 100 GPa] -> no clamp; the *default model* entry
        // still takes the user's value; enable == false -> mask 0
        CHECK(table[0].kappa == desc.default_resistance);
        CHECK(table[0].mu == desc.default_friction);
        CHECK(table[0].kappa == oracle.table[0].kappa);
        CHECK(mask[0] == 0);
        CHECK(mask[0] == oracle.mask[0]);
    }
    {
        SceneDesc desc;
        desc.kappa_eval_scale   = 1e6;
        desc.default_user_set   = true;
        desc.default_resistance = 1000.0_GPa;  // above cfg_max -> clamp to cfg_max
        desc.mesh.positions     = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles     = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_tabular_clamp")};
        World  world{engine};
        world.init(scene);

        auto* cm     = ContactCoreProbe::contact_manager;
        auto  cfg_max = scene.config().find<Float>("contact/adaptive/max_kappa")->view()[0];
        auto  table   = read_device(cm->contact_tabular());
        REQUIRE(table.size() == 1);
        CHECK(table[0].kappa == cfg_max);  // clamped
    }
    {
        SceneDesc desc;
        desc.kappa_eval_scale   = 1e6;
        desc.default_user_set   = true;
        desc.default_resistance = -3.0;  // adaptive opt-in: never clamped
        desc.mesh.positions     = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles     = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_tabular_negative")};
        World  world{engine};
        world.init(scene);

        auto* cm    = ContactCoreProbe::contact_manager;
        auto  table = read_device(cm->contact_tabular());
        REQUIRE(table.size() == 1);
        CHECK(table[0].kappa == -3.0);
    }

    // ---- (d) the symmetric half: two elements, a model on (e1, e2) ---------
    {
        SceneDesc desc;
        desc.kappa_eval_scale    = 1e6;
        desc.n_contact_elements  = 2;
        desc.insert_pair_model   = true;
        desc.model_friction      = 0.9;
        desc.model_resistance    = 5.0_MPa;
        desc.model_enable        = false;
        desc.mesh.positions      = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles      = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_tabular_sym")};
        World  world{engine};
        world.init(scene);

        auto* cm = ContactCoreProbe::contact_manager;
        REQUIRE(cm != nullptr);
        auto cfg_min = scene.config().find<Float>("contact/adaptive/min_kappa")->view()[0];
        auto cfg_max = scene.config().find<Float>("contact/adaptive/max_kappa")->view()[0];
        CHECK(cm->contact_tabular_size() == 3);

        auto oracle = tabular_oracle(scene, cm->kappa_lower_bound(), cm->kappa_upper_bound(), cfg_min, cfg_max);
        auto table  = read_device(cm->contact_tabular());
        auto mask   = read_device(cm->contact_mask_tabular());
        REQUIRE(table.size() == 9);
        REQUIRE(oracle.table.size() == 9);
        CHECK(oracle.N == 3);

        // the default entry is (0,0) with the fallback min kappa
        CHECK(table[0].kappa == cfg_min);
        CHECK(table[0].mu == 0.5);
        // the model on (1,2): 5 MPa is *below* cfg_min=100 MPa -> clamped up
        CHECK(table[1 * 3 + 2].kappa == cfg_min);
        CHECK(table[1 * 3 + 2].mu == desc.model_friction);
        // ... and the symmetric half is identical
        CHECK(table[2 * 3 + 1].kappa == table[1 * 3 + 2].kappa);
        CHECK(table[2 * 3 + 1].mu == table[1 * 3 + 2].mu);
        CHECK(mask[1 * 3 + 2] == 0);
        CHECK(mask[2 * 3 + 1] == 0);
        // other entries keep the default coefficient
        CHECK(table[0 * 3 + 1].kappa == cfg_min);
        CHECK(table[1 * 3 + 0].kappa == cfg_min);
        CHECK(table[2 * 3 + 2].kappa == cfg_min);
        CHECK(mask[0 * 3 + 1] == 1);
        CHECK(mask[2 * 3 + 2] == 1);
        // element-wise against the oracle
        for(size_t i = 0; i < table.size(); ++i)
        {
            CHECK(table[i].kappa == oracle.table[i].kappa);
            CHECK(table[i].mu == oracle.table[i].mu);
            CHECK(mask[i] == oracle.mask[i]);
        }
    }
}

// ===========================================================================
//  2) the mask semantics
// ===========================================================================
TEST_CASE("t81a_mask_semantics", "[contact_core]")
{
    test_runtime();

    // contact mask: the symmetric half + the disabled-by-default entry
    {
        SceneDesc desc;
        desc.kappa_eval_scale   = 1e6;
        desc.n_contact_elements = 2;
        desc.insert_pair_model  = true;
        desc.model_enable       = false;
        desc.default_enable     = true;
        desc.mesh.positions     = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles     = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_mask_contact")};
        World  world{engine};
        world.init(scene);

        auto* cm   = ContactCoreProbe::contact_manager;
        auto  mask = read_device(cm->contact_mask_tabular());
        REQUIRE(mask.size() == 9);
        CHECK(cm->contact_mask_size() == 3);
        // every entry defaults to the default model's enabled flag (1), except the
        // explicitly disabled (1,2) pair (written in both orders)
        for(IndexT i = 0; i < 3; ++i)
        {
            for(IndexT j = 0; j < 3; ++j)
            {
                auto expected = ((i == 1 && j == 2) || (i == 2 && j == 1)) ? 0 : 1;
                CHECK(mask[static_cast<size_t>(i * 3 + j)] == expected);
            }
        }
        CHECK(mask[1 * 3 + 2] == 0);
        CHECK(mask[2 * 3 + 1] == 0);
    }

    // the subscene mask: identity by default, an explicitly enabled pair flips both
    // triangles
    {
        SceneDesc desc;
        desc.kappa_eval_scale    = 1e6;
        desc.n_subscenes         = 3;  // + the always-present default -> SN = 4
        desc.enable_subscene_1_2 = true;
        desc.mesh.positions      = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles      = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_mask_subscene")};
        World  world{engine};
        world.init(scene);

        auto* cm   = ContactCoreProbe::contact_manager;
        auto  mask = read_device(cm->subscene_mask_tabular());
        const IndexT SN = 4;
        REQUIRE(mask.size() == SN * SN);
        CHECK(cm->subscene_mask_size() == SN);
        // identity: self-subscene contact is on, everything else off, except the
        // explicitly enabled (s1, s2) pair (user ids 1 and 2)
        for(IndexT i = 0; i < SN; ++i)
        {
            for(IndexT j = 0; j < SN; ++j)
            {
                auto expected = (i == j) ? 1
                                        : (((i == 1 && j == 2) || (i == 2 && j == 1)) ? 1 : 0);
                CHECK(mask[static_cast<size_t>(i * SN + j)] == expected);
            }
        }
        CHECK(mask[1 * SN + 2] == 1);
        CHECK(mask[2 * SN + 1] == 1);
    }
}

// ===========================================================================
//  3) the kappa corridor
// ===========================================================================
TEST_CASE("t81a_kappa_corridor", "[contact_core]")
{
    test_runtime();

    // (a) corridor computable: re-derive the formula from the scene inputs
    {
        SceneDesc desc;
        desc.d_hat  = 0.01;
        desc.dt     = 0.25;
        desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}};
        desc.mesh.triangles = {{0, 1, 2}, {1, 3, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_corridor_a")};
        World  world{engine};
        world.init(scene);

        auto* cm   = ContactCoreProbe::contact_manager;
        auto* vm   = ContactCoreProbe::vertex_manager;
        auto  diag = vm->scene_diagonal();
        auto  scale = scene.config().find<Float>("contact/adaptive/kappa_eval_scale")->view()[0];
        auto  dt    = scene.config().find<Float>("dt")->view()[0];
        CHECK(scale == 1e-16);
        CHECK(dt == 0.25);
        CHECK(diag > 0.0);

        auto oracle = corridor_oracle(diag, desc.d_hat, scale, dt, 0.0);  // no FEM/ABD mass
        REQUIRE(oracle.lower > 0.0);
        CHECK(cm->kappa_lower_bound() == oracle.lower);
        CHECK(cm->kappa_upper_bound() == oracle.upper);
        CHECK(cm->kappa_upper_bound() == 100.0 * cm->kappa_lower_bound());
        std::printf("[contact_core] corridor: [%g, %g] diag=%g d_hat=%g dt=%g\n",
                    cm->kappa_lower_bound(), cm->kappa_upper_bound(), diag, desc.d_hat, dt);
    }

    // (b) a second scene: a different dt and d_hat must change both bounds
    {
        SceneDesc desc;
        desc.d_hat  = 0.02;
        desc.dt     = 0.5;
        desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_corridor_b")};
        World  world{engine};
        world.init(scene);

        auto* cm    = ContactCoreProbe::contact_manager;
        auto* vm    = ContactCoreProbe::vertex_manager;
        auto  diag  = vm->scene_diagonal();
        auto  scale = scene.config().find<Float>("contact/adaptive/kappa_eval_scale")->view()[0];
        auto  dt    = scene.config().find<Float>("dt")->view()[0];
        auto  oracle = corridor_oracle(diag, desc.d_hat, scale, dt, 0.0);
        CHECK(cm->kappa_lower_bound() == oracle.lower);
        CHECK(cm->kappa_upper_bound() == oracle.upper);
    }

    // (c) not computable: every guard fires the -1 sentinel
    {
        struct Case
        {
            Float d_hat;
            Float dt;
        };
        std::vector<Case> cases = {{0.0, 0.25}, {0.01, 0.25}};  // d_hat==0 is the observable one
        SceneDesc desc;
        desc.kappa_eval_scale = 1e6;
        desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_corridor_none")};
        World  world{engine};
        world.init(scene);
        auto* cm = ContactCoreProbe::contact_manager;
        CHECK(cm->kappa_lower_bound() == -1.0);
        CHECK(cm->kappa_upper_bound() == -1.0);
        auto oracle = corridor_oracle(1.0, 0.01, 1e6, 0.25, 0.0);
        CHECK(oracle.lower == -1.0);
        CHECK(oracle.upper == -1.0);
    }
}

// ===========================================================================
//  4) compute_cfl_condition, both branches
// ===========================================================================
TEST_CASE("t81a_cfl_condition", "[contact_core]")
{
    test_runtime();

    SceneDesc desc;
    desc.cfl_enable = true;
    desc.d_hat      = 0.01;
    desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
    desc.mesh.positions = {{0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}, {0.0, 1.5, 0.0}, {1.0, 1.5, 0.0}};
    desc.mesh.triangles = {{0, 1, 2}, {1, 3, 2}};
    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("u81a_cfl")};
    World  world{engine};
    world.init(scene);

    auto* cm = ContactCoreProbe::contact_manager;
    auto* vm = ContactCoreProbe::vertex_manager;
    auto* sm = ContactCoreProbe::surface_manager;
    auto* gtf = ContactCoreProbe::trajectory_filter;
    REQUIRE(cm != nullptr);
    REQUIRE(vm != nullptr);
    REQUIRE(sm != nullptr);
    REQUIRE(gtf != nullptr);
    CHECK(cm->cfl_enabled());

    const auto n_vertices = vm->positions().size();
    CHECK(n_vertices == 5);  // the half plane's own vertex (id 0) + 4 mesh vertices
    auto surf = read_device(sm->surf_vertices());
    CHECK(surf.size() == 4);

    // a deliberately unbalanced displacement: the half-plane vertex (a NON-surface
    // vertex) has by far the largest norm, so the two kernels give different answers
    std::vector<Vector3> disps;
    disps.push_back(Vector3(-10.0, 0.0, 0.0));  // vertex 0 (not a surface vertex)
    disps.push_back(Vector3(0.001, 0.0, 0.0));
    disps.push_back(Vector3(0.0, 0.002, 0.0));
    disps.push_back(Vector3(0.0, 0.0, 0.004));
    disps.push_back(Vector3(0.008, 0.0, 0.0));
    REQUIRE(disps.size() == n_vertices);
    set_displacements(disps);
    // ---- the surface-vertex branch (the real manager's Impl has a surface mgr) ----
    {
        auto max_surf = 0.0;
        for(auto v : surf)
            max_surf = std::max(max_surf, disps[v].norm());
        CHECK(std::abs(max_surf - 0.008) < 1e-15);
        auto expected = std::min(0.5 * cm->d_hat() / max_surf, 1.0);
        auto got      = cm->impl().compute_cfl_condition();
        CHECK(got == expected);
        CHECK(std::abs(got - 0.625) < 1e-15);
        // the per-vertex norms the kernel wrote are the *surface* vertices' norms
        auto norms = read_device(cm->impl().vert_disp_norms.cview());
        REQUIRE(norms.size() == surf.size());
        for(size_t i = 0; i < surf.size(); ++i)
            CHECK(std::abs(norms[i] - disps[surf[i]].norm()) < 1e-15);
        CHECK(read_device_view(cm->impl().max_disp_norm.view())[0] == max_surf);
    }

    // ---- the all-vertex branch (a standalone Impl with no surface manager) ----
    {
        gg::GlobalContactManager::Impl impl;
        impl.global_vertex_manager      = *vm;
        impl.global_trajectory_filter   = *gtf;
        impl.cfl_enabled                = true;
        impl.d_hat                      = cm->d_hat();
        auto max_all = 0.0;
        for(auto& d : disps)
            max_all = std::max(max_all, d.norm());
        CHECK(max_all == 10.0);
        auto expected = std::min(0.5 * impl.d_hat / max_all, 1.0);
        auto got      = impl.compute_cfl_condition();
        CHECK(got == expected);
        CHECK(got != cm->impl().compute_cfl_condition());  // the argmax matters
        auto norms = read_device(impl.vert_disp_norms.cview());
        REQUIRE(norms.size() == n_vertices);
        for(size_t i = 0; i < n_vertices; ++i)
            CHECK(std::abs(norms[i] - disps[i].norm()) < 1e-15);
    }

    // ---- the h_max == 0 -> 1.0 guard and the min(.., 1.0) cap ----
    {
        std::vector<Vector3> zeros(n_vertices, Vector3::Zero());
        set_displacements(zeros);
        CHECK(cm->impl().compute_cfl_condition() == 1.0);

        std::vector<Vector3> tiny(n_vertices, Vector3::Zero());
        tiny[1] = Vector3(1e-9, 0.0, 0.0);  // 0.5*d_hat/1e-9 >> 1 -> capped at 1.0
        set_displacements(tiny);
        CHECK(cm->impl().compute_cfl_condition() == 1.0);
    }

    // ---- cfl disabled -> 1.0 (the first guard) ----
    {
        gg::GlobalContactManager::Impl impl;
        impl.global_vertex_manager    = *vm;
        impl.global_trajectory_filter = *gtf;
        impl.cfl_enabled              = false;
        impl.d_hat                    = cm->d_hat();
        CHECK(impl.compute_cfl_condition() == 1.0);
    }

    // ---- no trajectory filter -> 1.0 (the second guard) ----
    {
        gg::GlobalContactManager::Impl impl;
        impl.global_vertex_manager = *vm;
        impl.cfl_enabled           = true;
        impl.d_hat                 = cm->d_hat();
        CHECK(impl.compute_cfl_condition() == 1.0);
    }
}

// ===========================================================================
//  5) compute_feasible_step: the four kernels
// ===========================================================================
TEST_CASE("t81a_feasible_step", "[contact_core]")
{
    test_runtime();

    // a mesh whose vertices are global ids 0..8; the pairs reference them, but only
    // the geometry positions matter for the kernels.  Every pair is laid out so the
    // relative motion is *collinear* with the separation, which is what makes the
    // additive CCD loop's time-of-contact closed-form.
    SceneDesc desc;
    desc.d_hat = 0.01;
    // A half plane is *required* whenever run_stage() is used: the (always valid in an
    // IPC scene) EasyVertexHalfPlaneTrajectoryFilter dispatches its kernel with the
    // plane arrays, and LC rejects a zero-length buffer view as a kernel argument
    // (the recorded harness hazard) -- with no half plane that filter crashes.  Its
    // own vertex becomes global id 0, so every mesh vertex shifts by one.
    desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, -5.0}, .N = {0.0, 1.0, 0.0}});
    desc.mesh.positions = {{0.0, 0.0, 0.0},      // g1  (PP pair's first point)
                           {0.0, 0.0, 1.0},      // g2  (the moving point: PT/PE/PP)
                           {-1.0, -1.0, 0.0},    // g3  (PT triangle)
                           {1.0, -1.0, 0.0},     // g4
                           {0.0, 1.0, 0.0},      // g5
                           {-1.0, 0.0, 1.0},     // g6  (EE edge A)
                           {1.0, 0.0, 1.0},      // g7
                           {0.0, -1.0, 0.0},     // g8  (EE edge B / PE edge)
                           {0.0, 1.0, 0.0}};     // g9
    desc.mesh.triangles = {{2, 3, 4}};           // just to have a surface
    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("u81a_feasible")};
    World  world{engine};
    world.init(scene);

    auto* cm  = ContactCoreProbe::contact_manager;
    auto* vm  = ContactCoreProbe::vertex_manager;
    auto* gtf = ContactCoreProbe::trajectory_filter;
    REQUIRE(cm != nullptr);
    REQUIRE(vm != nullptr);
    REQUIRE(gtf != nullptr);

    auto n = vm->positions().size();
    CHECK(n == 10);  // the half plane's vertex (id 0) + the 9 mesh vertices

    // only this filter is valid (the four BVH filters are not gated for this scene)
    auto simplex = gtf->find<gg::SimplexTrajectoryFilter>();
    REQUIRE(static_cast<bool>(simplex));
    CHECK(dynamic_cast<ContactCoreProbeFilter*>(simplex.view()) != nullptr);

    // ---- (a) the analytic linear cases ---------------------------------------
    // every case: the moving simplex translates along -z by 0.9 over the full step
    // from d0 = 1.0 to d1 = 0.1, so maxDispMag = 0.9 and
    //     t* = (1 - eta) * (d0 - thickness) / (d0 - d1),  eta = 0.2
    const Float th = 0.05;
    std::vector<Float> thicknesses(n, th / 2.0);  // the codim thicknesses sum to th
    set_thicknesses(thicknesses);

    std::vector<Vector3> disps(n, Vector3::Zero());
    disps[2] = Vector3(0.0, 0.0, -0.9);  // PT/PE/PP point
    disps[6] = Vector3(0.0, 0.0, -0.9);  // EE edge A (both endpoints)
    disps[7] = Vector3(0.0, 0.0, -0.9);
    set_displacements(disps);

    // record the d0/d1 the oracle uses (host-side, from the scene positions)
    auto pos = read_vector3(vm->positions());
    // PT: the point is 1 above the triangle plane, and its projection (0,0,0) is
    // inside the triangle (its signed volume is positive on all three edges)
    CHECK(std::abs(pos[2].z() - 1.0) < 1e-15);
    // EE: the crossed edges' closest points are their midpoints, which are 1 apart
    CHECK(std::abs((0.5 * (pos[6] + pos[7]) - 0.5 * (pos[8] + pos[9])).norm() - 1.0) < 1e-15);
    // PE: the perpendicular foot of the point on the edge is the origin -> distance 1
    CHECK(std::abs(pos[2].z() - 1.0) < 1e-15);
    CHECK(std::abs(pos[8].x()) < 1e-15);
    CHECK(std::abs(pos[9].x()) < 1e-15);
    // PP
    CHECK(std::abs((pos[2] - pos[1]).norm() - 1.0) < 1e-15);

    ContactCoreProbeFilter::clear_pairs();
    ContactCoreProbeFilter::PTs_data = {Vector4i{2, 3, 4, 5}};
    ContactCoreProbeFilter::EEs_data = {Vector4i{6, 7, 8, 9}};
    ContactCoreProbeFilter::PEs_data = {Vector3i{2, 8, 9}};
    ContactCoreProbeFilter::PPs_data = {Vector2i{2, 1}};

    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
    CHECK(static_cast<ContactCoreProbeFilter*>(simplex.view())->PTs().size() == 1);
    CHECK(static_cast<ContactCoreProbeFilter*>(simplex.view())->EEs().size() == 1);
    CHECK(static_cast<ContactCoreProbeFilter*>(simplex.view())->PEs().size() == 1);
    CHECK(static_cast<ContactCoreProbeFilter*>(simplex.view())->PPs().size() == 1);

    auto got = cm->impl().compute_feasible_step();

    // all four share d0 = 1, d1 = 0.1, thickness = 0.05
    auto t_star = feasible_toi_oracle(1.0, 0.1, th);
    CHECK(std::abs(t_star - 0.8444444444444444) < 1e-12);

    auto tois = read_device_view(cm->impl().feasible_tois.view());
    REQUIRE(tois.size() == 4);
    for(size_t i = 0; i < 4; ++i)
        CHECK(std::abs(tois[i] - t_star) < 1e-12);  // PT, EE, PE, PP

    // the min-reduce picks the smallest, and the outer accessor clamps to <= 1
    auto min_toi = *std::min_element(tois.begin(), tois.end());
    CHECK(got == min_toi);
    CHECK(read_device_view(cm->impl().min_feasible_toi.view())[0] == min_toi);
    CHECK(std::abs(got - t_star) < 1e-12);
    std::printf("[contact_core] feasible(a): tois=[%g,%g,%g,%g] min=%g\n",
                tois[0], tois[1], tois[2], tois[3], got);

    // ---- (b) the thickness argument must change t* ---------------------------
    {
        // same geometry, thickness 0.5 instead of 0.05
        std::vector<Float> thick(n, 0.25);
        set_thicknesses(thick);
        ContactCoreProbeFilter::PTs_data = {Vector4i{2, 3, 4, 5}};
        ContactCoreProbeFilter::EEs_data.clear();
        ContactCoreProbeFilter::PEs_data.clear();
        ContactCoreProbeFilter::PPs_data.clear();
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        auto got2 = cm->impl().compute_feasible_step();
        auto t2   = feasible_toi_oracle(1.0, 0.1, 0.5);
        CHECK(std::abs(t2 - 0.4444444444444444) < 1e-12);
        CHECK(std::abs(got2 - t2) < 1e-12);
        CHECK(got2 < got);
        set_thicknesses(thicknesses);  // restore
    }

    // ---- (c) no approach: every toi == 1.0 -----------------------------------
    {
        std::vector<Vector3> away(n, Vector3::Zero());
        away[2] = Vector3(0.0, 0.0, 1.0);  // the point moves away
        away[6] = Vector3(0.0, 0.0, 1.0);  // the edge moves away
        away[7] = Vector3(0.0, 0.0, 1.0);
        set_displacements(away);
        ContactCoreProbeFilter::PTs_data = {Vector4i{2, 3, 4, 5}};
        ContactCoreProbeFilter::EEs_data = {Vector4i{6, 7, 8, 9}};
        ContactCoreProbeFilter::PEs_data = {Vector3i{2, 8, 9}};
        ContactCoreProbeFilter::PPs_data.clear();
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        auto got3 = cm->impl().compute_feasible_step();
        CHECK(got3 == 1.0);
        auto tois3 = read_device_view(cm->impl().feasible_tois.view());
        REQUIRE(tois3.size() == 3);
        for(auto t : tois3)
            CHECK(t == 1.0);
        set_displacements(disps);  // restore
    }

    // ---- (d) a single PP pair: the argmin is the pair ------------------------
    {
        ContactCoreProbeFilter::PTs_data.clear();
        ContactCoreProbeFilter::EEs_data.clear();
        ContactCoreProbeFilter::PEs_data.clear();
        ContactCoreProbeFilter::PPs_data = {Vector2i{2, 1}};
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        auto got4 = cm->impl().compute_feasible_step();
        CHECK(std::abs(got4 - feasible_toi_oracle(1.0, 0.1, th)) < 1e-12);
        CHECK(read_device_view(cm->impl().feasible_tois.view()).size() == 1);
    }

    // ---- (e) zero pairs -> 1.0 (the total == 0 early return) -----------------
    {
        ContactCoreProbeFilter::clear_pairs();
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        CHECK(cm->impl().compute_feasible_step() == 1.0);
    }

    // ---- (f) no trajectory filter at all -> 1.0 ------------------------------
    {
        gg::GlobalContactManager::Impl impl;
        impl.global_vertex_manager = *vm;
        CHECK(impl.compute_feasible_step() == 1.0);
    }
}

// ===========================================================================
//  6) a plane-less scene: the whole trajectory stage must still run
// ===========================================================================
TEST_CASE("t81a_plane_less_scene", "[contact_core]")
{
    test_runtime();

    // `EasyVertexHalfPlaneTrajectoryFilter` (and its AL twin) registers in *every* IPC
    // scene, but its kernels take the half-plane geometry's arrays as arguments.  In a
    // scene with **no** half-plane geometry those views are empty, and LC rejects a
    // zero-length buffer view as a kernel argument (T7.1a rule 9) -- measured here as a
    // SIGSEGV inside `filter_active` before this batch's fix.  The fix skips the dispatch
    // when `n_planes == 0`, which is exact because both kernels' whole bodies sit inside
    // their `$for(j, 0u, n_planes)` loops (so they write nothing at all), and for
    // `filter_toi` it skips the reduce with it (the caller's `info.toi()` 1.1 fill already
    // *is* the answer).  This case is the guard: the same mesh through all three stages
    // with and without a plane, pinning the zero-pair and the in-contact answers.
    struct Fixture
    {
        IndexT pairs = -1;
        Float  step  = -1.0;
    };

    auto run_fixture = [](bool with_plane, std::string_view tag) -> Fixture
    {
        SceneDesc desc;
        desc.d_hat = 0.01;
        // the test-local *simplex* probe filter would be registered too (its `toi_*` views
        // are empty, which the global filter's `filter_toi` asserts on); the marker gate
        // invalidates it, and this case only needs the manager probe
        desc.probe_marker = false;
        if(with_plane)
            // 0.005 below the mesh plane: inside d_hat = 0.01
            desc.half_planes.push_back(HPDesc{.P = {0.0, -0.005, 0.0}, .N = {0.0, 1.0, 0.0}});
        desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        auto scene          = build_scene(desc);
        Engine engine{"luisa", workspace_path(tag)};
        World  world{engine};
        world.init(scene);

        auto* gtf  = ContactCoreProbe::trajectory_filter;
        auto* easy = ContactCoreProbe::easy_half_plane_filter;
                REQUIRE(gtf != nullptr);
        // the filter is created in a plane-less scene too -- that is exactly the point
        REQUIRE(easy != nullptr);

        // the mesh vertices fall straight down (``y`` only), 0.05 * alpha towards the
        // plane; the half-plane geometry contributes one vertex (id 0 in scene (b))
        // the mesh's per-vertex d_hat: the scene default (0.01), so the band
        // (0, (thickness + d_hat)^2) contains the 0.005 gap below
        auto n = ContactCoreProbe::vertex_manager->positions().size();
        std::vector<Float> d_hats(n, static_cast<Float>(desc.d_hat));
        set_d_hats(d_hats);
        std::vector<Vector3> disps(n, Vector3(0.0, -0.1, 0.0));
        set_displacements(disps);

        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::Detect, 0.5) == 1.0);
        CHECK(gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive) == 1.0);

        Fixture f;
        f.pairs = easy->num_pairs();
        f.step  = gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterTOI, 0.5);
        CHECK(f.step <= 1.0);
        return f;
    };

    // (a) no half-plane geometry at all: the stage completes, no pair, no TOI cap
    {
        auto f = run_fixture(false, "u81a_plane_less");
        CHECK(f.pairs == 0);
        CHECK(f.step == 1.0);  // the 1.1 sentinel, clamped -- *not* a skipped stage
        std::printf("[contact_core] plane-less: pairs=%d step=%g\n",
                    static_cast<int>(f.pairs),
                    f.step);
    }
    // (b) the positive control: the same mesh with a plane 0.005 below it (in contact)
    {
        auto f = run_fixture(true, "u81a_plane_touching");
        // the plane touches exactly the two mesh vertices that lie *in* the mesh plane;
        // the triangle's third vertex ((0,1,0)) sits 1.0 above it, far outside the
        // (thickness, thickness + d_hat)^2 = (0, 1e-4) band -- so this control pins an
        // in-band *and* an out-of-band vertex
        CHECK(f.pairs == 2);
        // toi = (-N.(P - x) - thickness) / t * (1 - eta), eta = 0.1, t = -N.dx =
        // 0.1 * alpha = 0.05, -N.(P - x) = 0.005, thickness = 0  =>  0.005/0.05*0.9
        CHECK(std::abs(f.step - 0.09) < 1e-5);
        std::printf("[contact_core] plane-touching: pairs=%d step=%g\n",
                    static_cast<int>(f.pairs),
                    f.step);
    }
}

// ===========================================================================
//  7) the reporting pipeline: reporter fan-out + exporter/feature readback
// ===========================================================================
TEST_CASE("t81a_reporting_pipeline", "[contact_core]")
{
    test_runtime();

    ContactCoreProbeReporter::reset();
    ContactCoreProbeExporter::reset();
    ContactCoreProbeReporter::reported_energy_count   = 3;
    ContactCoreProbeReporter::reported_gradient_count = 2;
    ContactCoreProbeReporter::reported_hessian_count  = 1;
    ContactCoreProbeReporter::energies                = {10.0, 20.0, 30.0};

    ContactCoreProbeExporter::energy_value   = 42.5;
    ContactCoreProbeExporter::gradient_value = -7.25;
    ContactCoreProbeExporter::hessian_value  = 3.5;

    SceneDesc desc;
    desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    desc.mesh.triangles = {{0, 1, 2}};
    auto scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("u81a_pipeline")};
    World  world{engine};
    world.init(scene);

    auto* cm  = ContactCoreProbe::contact_manager;
    auto* dyt = ContactCoreProbe::dytopo_effect_manager;
    auto* em  = ContactCoreProbe::exporter_manager;
    REQUIRE(cm != nullptr);
    REQUIRE(dyt != nullptr);
    REQUIRE(em != nullptr);

    // ---- (a) the contact reporter is registered and indexed ------------------
    auto reporters = cm->impl().contact_reporters.view();
    REQUIRE(reporters.size() == 1);
    CHECK(dynamic_cast<ContactCoreProbeReporter*>(reporters[0]) != nullptr);
    // (the reporter is also registered with the dytopo-effect manager -- provable
    //  by construction: only a registered reporter is reached by `_assemble` below)

    // ---- (b) extent -> assemble through the dytopo manager -------------------
    gg::GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo einfo;
    einfo.component_flags(gg::EnergyComponentFlags::Contact);
    einfo.gradient_only(false);
    dyt->compute_dytopo_effect(einfo);

    CHECK(ContactCoreProbeReporter::seen_gradient_only == false);
    CHECK(ContactCoreProbeReporter::seen_gradient_size == 2);
    CHECK(ContactCoreProbeReporter::seen_hessian_size == 1);
    CHECK(ContactCoreProbeReporter::seen_gradient_offset == 0);
    CHECK(ContactCoreProbeReporter::seen_hessian_offset == 0);
    // the manager aggregated exactly the reporter's counts
    CHECK(dyt->gradients().doublet_count() == 2);
    CHECK(dyt->hessians().triplet_count() == 1);

    // ---- (c) compute-energy through the (public-Impl) line-search reporter ---
    gg::DyTopoEffectLineSearchReporter::Impl lsr;
    lsr.global_dytopo_effect_manager = *dyt;
    lsr.compute_energy(false);
    CHECK(ContactCoreProbeReporter::seen_energy_size == 3);
    CHECK(ContactCoreProbeReporter::seen_is_initial == false);
    auto energies = read_device_view(lsr.energies.view());
    REQUIRE(energies.size() == 3);
    for(size_t i = 0; i < 3; ++i)
        CHECK(energies[i] == ContactCoreProbeReporter::energies[i]);
    // `is_initial` is forwarded verbatim
    lsr.compute_energy(true);
    CHECK(ContactCoreProbeReporter::seen_is_initial == true);

    // ---- (d) the exporter manager + the frontend feature --------------------
    std::string prim{ContactCoreProbeExporter::kPrimType};
    // the feature the exporter manager inserted into the scene
    auto feature = engine.features().find<core::ContactSystemFeature>();
    REQUIRE(feature != nullptr);
    auto types = feature->contact_primitive_types();
    REQUIRE(types.size() == 1);
    CHECK(types[0] == prim);

    {
        geometry::Geometry geo;
        feature->contact_energy(prim, geo);
        CHECK(ContactCoreProbeExporter::seen_energy_prim == prim);
        auto prim_type = geo.meta().find<std::string>("prim_type");
        REQUIRE(prim_type != nullptr);
        CHECK(view(*prim_type)[0] == prim);
        auto value = geo.meta().find<Float>("t81a_energy");
        REQUIRE(value != nullptr);
        CHECK(view(*value)[0] == ContactCoreProbeExporter::energy_value);
    }
    {
        geometry::Geometry geo;
        feature->contact_gradient(prim, geo);
        CHECK(ContactCoreProbeExporter::seen_gradient_prim == prim);
        auto value = geo.meta().find<Float>("t81a_gradient");
        REQUIRE(value != nullptr);
        CHECK(view(*value)[0] == ContactCoreProbeExporter::gradient_value);
    }
    {
        geometry::Geometry geo;
        feature->contact_hessian(prim, geo);
        CHECK(ContactCoreProbeExporter::seen_hessian_prim == prim);
        auto value = geo.meta().find<Float>("t81a_hessian");
        REQUIRE(value != nullptr);
        CHECK(view(*value)[0] == ContactCoreProbeExporter::hessian_value);
    }

    // ---- (e) an unknown primitive type: the warn-and-return path ------------
    {
        geometry::Geometry geo;
        feature->contact_energy("t81a_no_such_type", geo);
        CHECK(geo.meta().find<std::string>("prim_type") == nullptr);
        CHECK(ContactCoreProbeExporter::seen_energy_prim == prim);  // unchanged
    }
}

// ===========================================================================
//  8) the adaptive-parameter layer
// ===========================================================================
TEST_CASE("t81a_adaptive_parameters", "[contact_core]")
{
    test_runtime();

    // ---- (a) the base reporter + AdaptiveParameterInfo exchange semantics ----
    {
        ContactCoreProbeAdaptiveReporter::reset();
        SceneDesc desc;
        desc.kappa_eval_scale   = 1e6;
        desc.n_contact_elements = 1;
        desc.insert_self_model  = true;
        desc.model_resistance   = 0.0;  // <= 0: the base accepts it, GIPC does not
        desc.mesh.positions     = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles     = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_adaptive_base")};
        World  world{engine};
        world.init(scene);

        auto* cm = ContactCoreProbe::contact_manager;
        REQUIRE(cm != nullptr);
        // the probe reporter owns the single adaptive slot (GIPC was invalidated)
        REQUIRE(static_cast<bool>(cm->impl().adaptive_contact_parameter_reporter));
        CHECK(dynamic_cast<ContactCoreProbeAdaptiveReporter*>(
                  cm->impl().adaptive_contact_parameter_reporter.view())
              != nullptr);

        auto cfg_min = scene.config().find<Float>("contact/adaptive/min_kappa")->view()[0];
        (void)cfg_min;

        CHECK(ContactCoreProbeAdaptiveReporter::called == false);
        cm->run_compute_adaptive_parameters();
        CHECK(ContactCoreProbeAdaptiveReporter::called == true);
        CHECK(ContactCoreProbeAdaptiveReporter::seen_tabular_size == 4);  // N=2 -> 2x2
        // the manager's tabular at (0,0): the default model (not user-set) -> fallback
        CHECK(ContactCoreProbeAdaptiveReporter::seen_tabular_kappa_0 == 100.0_MPa);
        // the exchange handed back the manager's previous buffer, the replacement was
        // visible to the manager, and the restore swap returned the replacement
        CHECK(ContactCoreProbeAdaptiveReporter::first_exchange_returned_original == true);
        CHECK(ContactCoreProbeAdaptiveReporter::second_exchange_returned_replacement == true);
        CHECK(ContactCoreProbeAdaptiveReporter::manager_saw_new_tabular == true);
        CHECK(ContactCoreProbeAdaptiveReporter::restored == true);
        // the probe restored the original, so the manager's tabular is unchanged
        auto table = read_device(cm->contact_tabular());
        REQUIRE(table.size() == 4);
        CHECK(table[0].kappa == 100.0_MPa);
        CHECK(table[0].mu == 0.5);
        // the resistance == 0 model is >= 0, so the reference clamps it up to the
        // config lower bound (only a *negative* resistance skips the clamp)
        CHECK(table[1 * 2 + 1].kappa == 100.0_MPa);
    }

    // ---- (b) the GIPC strategy's resolved state on a synthetic tabular -------
    {
        ContactCoreProbeAdaptiveReporter::reset();  // the probe must NOT run here
        SceneDesc desc;
        desc.kappa_eval_scale   = 1e6;
        desc.n_contact_elements = 2;
        desc.insert_self_model  = true;
        desc.model_resistance   = -2.0;  // < 0: adaptive, so GIPC is valid
        desc.mesh.positions     = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles     = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_adaptive_gipc")};
        World  world{engine};
        world.init(scene);

        auto* cm = ContactCoreProbe::contact_manager;
        REQUIRE(cm != nullptr);
        REQUIRE(static_cast<bool>(cm->impl().adaptive_contact_parameter_reporter));
        auto* gipc = dynamic_cast<gg::GIPCAdaptiveParameterStrategy*>(
            cm->impl().adaptive_contact_parameter_reporter.view());
        REQUIRE(gipc != nullptr);

        // the config values it resolved (the clamp's fallback bounds)
        CHECK(gipc->min_kappa == 100.0_MPa);
        CHECK(gipc->init_kappa == 1.0_GPa);
        CHECK(gipc->max_kappa == 100.0_GPa);

        // the adaptive model's index set: element 1's (e1,e1) model has ids (1,1),
        // collected because its resistance is negative
        REQUIRE(gipc->h_adaptive_kappa_index.size() == 1);
        CHECK(static_cast<IndexT>(gipc->h_adaptive_kappa_index[0]) == 1);
        CHECK(gipc->adaptive_topos.size() == 1);

        // the synthetic test tabular: adaptive entries kappa=1, everything else 0
        REQUIRE(gipc->test_contact_tabular != nullptr);
        auto N = gipc->test_contact_tabular->size();
        CHECK(N == 9);  // 3 contact elements -> 3x3
        auto tab = read_device(gipc->test_contact_tabular->cview());
        REQUIRE(tab.size() == 9);
        CHECK(tab[1 * 3 + 1].kappa == 1.0);
        CHECK(tab[1 * 3 + 1].mu == 0.0);
        CHECK(tab[0].kappa == 0.0);  // the default entry is not adaptive
        CHECK(tab[2 * 3 + 2].kappa == 0.0);

        // the scaling law the strategy implements, pinned as a host re-derivation:
        //   proj_kappa = (g_c . g_c == 0) ? init_kappa : -(g_c . g_nc) / (g_c . g_c)
        //   new_kappa  = clamp(proj_kappa, lo, hi)
        // with lo/hi the manager's corridor when computable, else the config values.
        auto law = [&](Float contact2, Float proj, Float lo, Float hi)
        {
            auto proj_kappa = (contact2 == 0.0) ? gipc->init_kappa : -proj / contact2;
            return std::clamp(proj_kappa, lo, hi);
        };
        // here the corridor is not computable (large eval scale) -> config min/max
        CHECK(cm->kappa_lower_bound() == -1.0);
        CHECK(cm->kappa_upper_bound() == -1.0);
        CHECK(law(0.0, 1.0, gipc->min_kappa, gipc->max_kappa) == gipc->init_kappa);
        CHECK(law(4.0, -8.0, gipc->min_kappa, gipc->max_kappa) == gipc->min_kappa);  // 2 clamped up
        CHECK(law(1e-3, -1.0, gipc->min_kappa, gipc->max_kappa) == 100.0_MPa);  // clamped up
        CHECK(law(1.0, -1.0e14, gipc->min_kappa, gipc->max_kappa) == 100.0_GPa); // clamped down

        // the probe reporter refused to build here (so exactly one adaptive reporter)
        CHECK(ContactCoreProbeAdaptiveReporter::called == false);
    }
}

// ===========================================================================
//  9) determinism / the empty case / the quoted numbers
// ===========================================================================
TEST_CASE("t81a_determinism_and_empty", "[contact_core]")
{
    test_runtime();

    // ---- (a) repeat calls are deterministic ---------------------------------
    {
        SceneDesc desc;
        desc.cfl_enable = true;
        desc.half_planes.push_back(HPDesc{.P = {0.0, 0.0, 0.0}, .N = {0.0, 1.0, 0.0}});
        desc.mesh.positions = {{0.0, 0.0625, 0.0}, {1.0, 0.0625, 0.0}, {0.0, 0.25, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_determinism")};
        World  world{engine};
        world.init(scene);

        auto* cm = ContactCoreProbe::contact_manager;
        auto* vm = ContactCoreProbe::vertex_manager;
        auto* gtf = ContactCoreProbe::trajectory_filter;
        REQUIRE(cm != nullptr);

        std::vector<Vector3> disps(vm->positions().size(), Vector3(0.003, 0.004, 0.0));
        set_displacements(disps);
        auto a = cm->impl().compute_cfl_condition();
        auto b = cm->impl().compute_cfl_condition();
        CHECK(a == b);

        ContactCoreProbeFilter::clear_pairs();
        ContactCoreProbeFilter::PPs_data = {Vector2i{1, 2}};
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        auto c = cm->impl().compute_feasible_step();
        auto e = cm->impl().compute_feasible_step();
        CHECK(c == e);
        CHECK(read_device_view(cm->impl().feasible_tois.view())
              == read_device_view(cm->impl().feasible_tois.view()));
    }

    // ---- (b) the empty case: no probe marker -> zero contact reporters, zero pairs
    {
        SceneDesc desc;
        desc.probe_marker   = false;
        desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_empty")};
        World  world{engine};
        world.init(scene);

        auto* cm  = ContactCoreProbe::contact_manager;
        auto* gtf = ContactCoreProbe::trajectory_filter;
        REQUIRE(cm != nullptr);
        // the probe reporter was invalidated by its marker gate
        CHECK(cm->impl().contact_reporters.view().empty());
        CHECK(cm->impl().adaptive_contact_parameter_reporter == false);
        // the probe filter was invalidated by its marker gate (the scene's default
        // `collision_detection/method` keeps exactly one *other* simplex filter valid)
        REQUIRE(gtf != nullptr);
        CHECK(gtf->find<ContactCoreProbeFilter>() == false);
        CHECK(cm->impl().compute_feasible_step() == 1.0);
        // the manager still built its (single-element) tabulars
        CHECK(cm->contact_tabular_size() == 1);
        CHECK(cm->contact_mask_size() == 1);
    }

    // ---- (c) the quoted numbers ---------------------------------------------
    {
        SceneDesc desc;
        desc.d_hat  = 0.01;
        desc.dt     = 0.25;
        desc.mesh.positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        desc.mesh.triangles = {{0, 1, 2}};
        auto scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81a_numbers")};
        World  world{engine};
        world.init(scene);
        auto* cm = ContactCoreProbe::contact_manager;
        auto  table = read_device(cm->contact_tabular());
        std::printf("[contact_core] default trimesh scene: diag=%g corridor=[%g,%g] default_kappa=%g\n",
                    ContactCoreProbe::vertex_manager->scene_diagonal(),
                    cm->kappa_lower_bound(),
                    cm->kappa_upper_bound(),
                    table[0].kappa);
        CHECK(std::abs(cm->kappa_lower_bound() - 8000.0) < 0.01);
        CHECK(std::abs(cm->kappa_upper_bound() - 800000.0) < 1.0);
        CHECK(table[0].kappa == cm->kappa_lower_bound());
        CHECK(cm->d_hat() == 0.01);
        CHECK(cm->eps_velocity() == 0.01);
    }
}
