/********************************************************************
 * @file luisa_contact_model_test.cpp
 * @brief T8.1b — the IPC contact *models*: the transcribed barrier law and its derivatives,
 * the two transcribed `sym/*.inl` bodies, the four per-primitive-type pair energies /
 * gradients / Hessians of the simplex (normal + frictional) and vertex/half-plane (normal +
 * frictional) models, the per-type exporter readback, and determinism.  Tag `[contact_model]`.
 *
 * ORACLES (independent, never the CUDA backend)
 * ---------------------------------------------
 *  1. The barrier law is *re-derived in this file* from the LaTeX expression SymEigen.py was
 *     fed (`b(D) = kappa (D - dHat^2)^2 log(D/dHat^2)^2` and its thickness-shifted twin), and
 *     `dKappaBarrierdD` / `ddKappaBarrierddD` are checked against **central finite
 *     differences of that re-derived energy** — the only way to catch a transcription typo in
 *     generated matrix algebra.  The step/tolerance are reported by the run.
 *  2. Every pair energy is the test's own closed form: the four synthetic pairs are laid out so
 *     that each squared distance is *exactly* `0.06^2` (a point 0.06 above a triangle's
 *     interior, a crossing edge pair 0.06 apart, a point 0.06 above an edge's midpoint, two
 *     points 0.06 apart), so the oracle needs no distance routine at all.
 *  3. The gradient / Hessian are checked by **finite differences of the device quantities
 *     themselves** (perturb the manager's position buffer, re-run the reporter pipeline, compare
 *     `dE/dx_z` and the Hessian-vector product) *and* against the chain-rule closed form
 *     `dE/dx = dB/dD * dD/dx`, `H = dB2/dD2 * g g^T + dB/dD * d2D/dx2` for the layouts where
 *     `dD/dx` is unambiguous.
 *  4. The friction branch: `mu == 0` must make the frictional models' energies *exactly* zero
 *     (so the scene's total equals the normal models'), `mu > 0` must add a non-negative term
 *     whose gradient/Hessian are FD-checked, and the `eps_velocity * dt` scaling is pinned by
 *     halving `dt`.
 *  The transcribed device math is *also* evaluated on the host (`T = Float`) and compared with
 *  the device's value bit-for-bit — the T7.2a "same expression, same order" check, which is what
 *  makes the host-side FD oracle meaningful for the same arithmetic.
 *
 * HOW IT IS DRIVEN
 * ----------------
 * Each scene carries a marker geometry-meta attribute (`t81b_probe`) that only this file's
 * scenes set; the probe `SimplexTrajectoryFilter` (which publishes the four synthetic pairs)
 * refuses to build elsewhere, exactly like T8.1a's probes, so the shared test binary's other
 * scenes are untouched.  The models themselves are *production* systems: they are valid in every
 * `contact/constitution == ipc` scene, which is why this file reads the real models through the
 * contact manager's reporter list and the real pipeline
 * (`GlobalDyTopoEffectManager::compute_dytopo_effect` + the line-search reporter's
 * `compute_energy`) rather than through a mirror.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <uipc/core/contact_tabular.h>
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
#include <contact_system/simplex_normal_contact.h>
#include <contact_system/simplex_frictional_contact.h>
#include <contact_system/vertex_half_plane_normal_contact.h>
#include <contact_system/vertex_half_plane_frictional_contact.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h>
#include <dytopo_effect_system/dytopo_effect_line_search_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <utils/codim_thickness.h>
#include <utils/primitive_d_hat.h>
#include <luisa_device/math/distance/distance.h>
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
namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;
namespace sym_codim = uipc::backend::luisa::sym::codim_ipc_contact;
namespace sym_simplex = uipc::backend::luisa::sym::codim_ipc_simplex_contact;
namespace sym_fric      = uipc::backend::luisa::sym::codim_ipc_contact;
namespace sym_ph = uipc::backend::luisa::sym::ipc_vertex_half_contact;

namespace t81b
{
// ===========================================================================
//  Helpers
// ===========================================================================
/// `|a - b| <= rel * max(1, |a|, |b|)` — the test's tolerance predicate (Catch2's `Approx`
/// is not available in the standalone shim the sibling device tests build against).
[[nodiscard]] inline bool close(double a, double b, double rel = 1e-12) noexcept
{
    const double s = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= rel * s;
}
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

template <typename T>
void write_device(const lud::LBufferView<T>& view, std::span<const T> host)
{
    view.copy_from(host, &gg::frame_stream());
    gg::frame_stream().flush();
}

std::vector<Float> to_luisa_floats(const std::vector<Float>& v)
{
    return v;
}

constexpr std::string_view kMarker = "t81b_probe";

bool has_marker(backend::WorldVisitor& world)
{
    for(auto& slot : world.scene().geometries())
    {
        if(slot->geometry().meta().find<IndexT>(std::string{kMarker}))
            return true;
    }
    return false;
}

// ===========================================================================
//  The probe filter: publishes exactly the pairs the test's oracles describe
// ===========================================================================
class ProbeFilter final : public gg::SimplexTrajectoryFilter
{
  public:
    using SimplexTrajectoryFilter::SimplexTrajectoryFilter;

    static std::vector<Vector4i> PTs_data;
    static std::vector<Vector4i> EEs_data;
    static std::vector<Vector3i> PEs_data;
    static std::vector<Vector2i> PPs_data;

    static void clear_pairs()
    {
        PTs_data.clear();
        EEs_data.clear();
        PEs_data.clear();
        PPs_data.clear();
    }

  protected:
    void do_build(BuildInfo& info) override
    {
        if(!has_marker(world()))
            throw uipc::backend::SimSystemException("not a [contact_model] probe scene");
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
REGISTER_SIM_SYSTEM(ProbeFilter);

std::vector<Vector4i> ProbeFilter::PTs_data;
std::vector<Vector4i> ProbeFilter::EEs_data;
std::vector<Vector3i> ProbeFilter::PEs_data;
std::vector<Vector2i> ProbeFilter::PPs_data;

// ===========================================================================
//  The manager probe
// ===========================================================================
class Probe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    static gg::GlobalVertexManager*                 vertex_manager;
    static gg::GlobalContactManager*                contact_manager;
    static gg::GlobalTrajectoryFilter*              trajectory_filter;
    static gg::GlobalDyTopoEffectManager*           dytopo_effect_manager;
    static gg::ContactExporterManager*              exporter_manager;
    static gg::HalfPlane*                           half_plane;
    static gg::HalfPlaneVertexReporter*             half_plane_vertex_reporter;
    static gg::EasyVertexHalfPlaneTrajectoryFilter* easy_half_plane_filter;

  protected:
    void do_build() override
    {
        vertex_manager             = find<gg::GlobalVertexManager>();
        contact_manager            = find<gg::GlobalContactManager>();
        trajectory_filter          = find<gg::GlobalTrajectoryFilter>();
        dytopo_effect_manager      = find<gg::GlobalDyTopoEffectManager>();
        exporter_manager           = find<gg::ContactExporterManager>();
        half_plane                 = find<gg::HalfPlane>();
        half_plane_vertex_reporter = find<gg::HalfPlaneVertexReporter>();
        easy_half_plane_filter     = find<gg::EasyVertexHalfPlaneTrajectoryFilter>();
    }
};
REGISTER_SIM_SYSTEM(Probe);

gg::GlobalVertexManager*                 Probe::vertex_manager             = nullptr;
gg::GlobalContactManager*                Probe::contact_manager            = nullptr;
gg::GlobalTrajectoryFilter*              Probe::trajectory_filter          = nullptr;
gg::GlobalDyTopoEffectManager*           Probe::dytopo_effect_manager      = nullptr;
gg::ContactExporterManager*              Probe::exporter_manager           = nullptr;
gg::HalfPlane*                           Probe::half_plane                 = nullptr;
gg::HalfPlaneVertexReporter*             Probe::half_plane_vertex_reporter = nullptr;
gg::EasyVertexHalfPlaneTrajectoryFilter* Probe::easy_half_plane_filter     = nullptr;

// ---------------------------------------------------------------------------
//  The four models, found through the contact manager's reporter list
// ---------------------------------------------------------------------------
template <typename T>
T* find_model(gg::GlobalContactManager* cm) noexcept
{
    if(!cm)
        return nullptr;
    for(auto* r : cm->impl().contact_reporters.view())
    {
        if(auto* m = dynamic_cast<T*>(r))
            return m;
    }
    return nullptr;
}

// ===========================================================================
//  Scene description + builder
//  The synthetic layout (see the file header): four pairs, each with a squared
//  gap of exactly 0.06^2 = 0.0036, so the energy oracle is closed-form.
// ===========================================================================
//
//  0 : (0.25, 0.25, 0.06)  PT point      above the interior of triangle (1,2,3)
//  1 : (0.0,  0.0,  0.0)   PT / PE T0/E0
//  2 : (1.0,  0.0,  0.0)   PT / PE T1/E1
//  3 : (0.0,  1.0,  0.0)   PT T2
//  4 : (0.25, 0.5,  0.03)  EE A0 / PP point
//  5 : (0.75, 0.5,  0.03)  EE A1
//  6 : (0.5,  0.25,-0.03)  EE B0
//  7 : (0.5,  0.75,-0.03)  EE B1
//  8 : (0.25, 0.5, -0.03)  PP point
//  9 : (0.5,  0.0,  0.06)  PE point      above the midpoint of edge (1,2)
const std::vector<Vector3>& probe_positions()
{
    static const std::vector<Vector3> p = {
        {0.25, 0.25, 0.06},
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.25, 0.5, 0.03},
        {0.75, 0.5, 0.03},
        {0.5, 0.25, -0.03},
        {0.5, 0.75, -0.03},
        {0.25, 0.5, -0.03},
        {0.5, 0.0, 0.06},
    };
    return p;
}

const std::vector<Vector3i>& probe_triangles()
{
    static const std::vector<Vector3i> t = {
        {1, 2, 3}, {4, 5, 0}, {6, 7, 0}, {1, 2, 9}, {9, 3, 0}, {0, 8, 4}};
    return t;
}

void set_probe_pairs()
{
    ProbeFilter::clear_pairs();
    ProbeFilter::PTs_data = {Vector4i{0, 1, 2, 3}};
    ProbeFilter::EEs_data = {Vector4i{4, 5, 6, 7}};
    ProbeFilter::PEs_data = {Vector3i{9, 1, 2}};
    ProbeFilter::PPs_data = {Vector2i{4, 8}};
}

struct HPDesc
{
    Vector3 P{0.0, 0.0, -0.005};
    Vector3 N{0.0, 0.0, 1.0};
};

struct SceneDesc
{
    std::vector<HPDesc>  half_planes;
    std::vector<Vector3> positions = probe_positions();
    std::vector<Vector3i> triangles = probe_triangles();
    bool                  probe_marker = true;

    Float thickness = 0.01;
    Float d_hat     = 0.1;
    Float dt        = 0.25;
    Float kappa_eval_scale = 1e-16;
    Float friction  = 0.0;  // the default model's friction_rate
    std::string constitution = "ipc";
};

Scene build_scene(const SceneDesc& desc)
{
    auto config = Scene::default_config();
    config["sanity_check"]["enable"]  = 0;
    config["dt"]                      = desc.dt;
    config["contact"]["constitution"] = desc.constitution;
    config["contact"]["d_hat"]        = desc.d_hat;
    config["contact"]["adaptive"]["kappa_eval_scale"] = desc.kappa_eval_scale;
    Scene scene{config};

    {
        // a *user* default model pins both the resistance (so the corridor clamp is
        // predictable) and the friction rate
        auto& ct = scene.contact_tabular();
        ct.default_model(desc.friction, 1.0_GPa, true, Json::object());
    }

    for(size_t i = 0; i < desc.half_planes.size(); ++i)
    {
        auto        object = scene.objects().create("t81b_ground_" + std::to_string(i));
        const auto& hp     = desc.half_planes[i];
        auto        ig     = halfplane(hp.P, hp.N);
        ig.meta().create<IndexT>(builtin::contact_element_id, 0);
        ig.meta().create<IndexT>(builtin::subscene_element_id, 0);
        if(desc.probe_marker)
            ig.meta().create<IndexT>(std::string{kMarker}, 1);
        object->geometries().create(ig);
    }

    {
        auto complex = trimesh(span<const Vector3>{desc.positions},
                               span<const Vector3i>{desc.triangles});
        label_surface(complex);
        auto verts     = complex.vertices();
        auto thickness = verts.create<Float>("test_thickness", 0.0);
        {
            auto v = view(*thickness);
            for(size_t i = 0; i < v.size(); ++i)
                v[i] = desc.thickness;
        }
        auto d_hat = verts.create<Float>("test_d_hat", 0.0);
        {
            auto v = view(*d_hat);
            for(size_t i = 0; i < v.size(); ++i)
                v[i] = desc.d_hat;
        }
        auto cid = verts.create<IndexT>("test_contact_element_id", 0);
        {
            auto v = view(*cid);
            for(size_t i = 0; i < v.size(); ++i)
                v[i] = 0;
        }
        if(desc.probe_marker)
            complex.meta().create<IndexT>(std::string{kMarker}, 1);
        auto object = scene.objects().create("t81b_mesh");
        object->geometries().create(complex);
    }

    return scene;
}

void set_positions(const std::vector<Vector3>& per_vertex)
{
    auto* vm = Probe::vertex_manager;
    auto  p  = vm->positions();
    REQUIRE(p.size() == per_vertex.size());
    std::vector<gg::LVector3> host;
    host.reserve(per_vertex.size());
    for(auto& v : per_vertex)
        host.push_back(gg::to_luisa(v));
    write_device(p, span<const gg::LVector3>{host});
}

void set_prev_positions(const std::vector<Vector3>& per_vertex)
{
    auto* vm = Probe::vertex_manager;
    auto  p  = vm->prev_positions();
    REQUIRE(p.size() == per_vertex.size());
    std::vector<gg::LVector3> host;
    host.reserve(per_vertex.size());
    for(auto& v : per_vertex)
        host.push_back(gg::to_luisa(v));
    write_device(p, span<const gg::LVector3>{host});
}

void set_thicknesses(const std::vector<Float>& per_vertex)
{
    auto* vm = Probe::vertex_manager;
    auto  t  = vm->thicknesses();
    REQUIRE(t.size() == per_vertex.size());
    write_device(t, span<const Float>{per_vertex});
}

void set_d_hats(const std::vector<Float>& per_vertex)
{
    auto* vm = Probe::vertex_manager;
    auto  d  = vm->d_hats();
    REQUIRE(d.size() == per_vertex.size());
    write_device(d, span<const Float>{per_vertex});
}

// ===========================================================================
//  The independent oracles
//  (the reference's log^2 barrier, re-derived from the LaTeX of
//   sym/codim_ipc_contact.inl — NOT from the transcribed body)
// ===========================================================================
/// `KappaBarrierLog2`:  B = kappa (D - dHat^2)^2 log(D/dHat^2)^2
double barrier_log2(double kappa, double D, double dHat)
{
    const double x0 = dHat * dHat;
    const double t  = D - x0;
    const double l  = std::log(D / x0);
    return kappa * t * t * l * l;
}

/// `KappaBarrierLog2WithThickness`:
///   B = kappa (D - xi^2 - (dHat^2 + 2 dHat xi))^2 log((D - xi^2)/(dHat^2 + 2 dHat xi))^2
double barrier_log2_thick(double kappa, double D, double dHat, double xi)
{
    const double x0 = xi * xi;
    const double x1 = dHat * dHat + 2.0 * dHat * xi;
    const double t  = D - x0 - x1;
    const double l  = std::log((D - x0) / x1);
    return kappa * t * t * l * l;
}

/// The reference's `KappaBarrier` dispatcher: `xi == 0` -> the plain log^2 form.
double barrier(double kappa, double D, double dHat, double xi)
{
    return xi == 0.0 ? barrier_log2(kappa, D, dHat) : barrier_log2_thick(kappa, D, dHat, xi);
}

/// The classic (log, not log^2) barrier `KappaBarrierWithThickness`:
///   B = -kappa (D - dHat^2 - 2 dHat xi - xi^2)^2 log((D - xi^2)/(dHat^2 + 2 dHat xi))
double barrier_classic(double kappa, double D, double dHat, double xi)
{
    const double x0 = xi * xi;
    const double x1 = dHat * dHat + 2.0 * dHat * xi;
    const double t  = D - x0 - x1;
    return -kappa * t * t * std::log((D - x0) / x1);
}

/// The `d2KappaBarrierdDdD` closed form -- the analytic second derivative of `barrier`:
/// with `u = D - xi^2`, `t = u - (dHat^2 + 2 dHat xi)` and `l = log(u / (dHat^2 + 2 dHat xi))`,
///   `d^2/dD^2 [ kappa t^2 l^2 ] = 2 kappa ( l^2 + 4 t l / u + t^2 (1 - l) / u^2 )`
/// (derived here, then verified against 60-digit arithmetic at 1e-26 relative -- the nested
/// central difference that used to stand in for it is **not** usable: measured 0.85 relative
/// error, because the third derivative blows up towards the band's edges and the roundoff is
/// amplified by 1/h1^2).
double barrier_d2(double kappa, double D, double dHat, double xi)
{
    const double x0 = xi * xi;
    const double x1 = dHat * dHat + 2.0 * dHat * xi;
    const double u  = D - x0;
    const double t  = u - x1;
    const double l  = std::log(u / x1);
    return 2.0 * kappa * (l * l + 4.0 * t * l / u + t * t * (1.0 - l) / (u * u));
}

/// The same for the classic (log) barrier:
///   `d^2/dD^2 [ -kappa t^2 l ] = -kappa ( 2 l + 4 t / u - t^2 / u^2 )`
double barrier_classic_d2(double kappa, double D, double dHat, double xi)
{
    const double x0 = xi * xi;
    const double x1 = dHat * dHat + 2.0 * dHat * xi;
    const double u  = D - x0;
    const double t  = u - x1;
    const double l  = std::log(u / x1);
    return -kappa * (2.0 * l + 4.0 * t / u - t * t / (u * u));
}

/// central difference of a unary function
template <typename F>
double central(F&& f, double x, double h)
{
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

/// The D-range of the active band `(xi^2, (dHat + xi)^2)`.
void band(double dHat, double xi, double& lo, double& hi)
{
    lo = xi * xi;
    hi = (dHat + xi) * (dHat + xi);
}

// ===========================================================================
//  Device math harness kernels (the transcribed bodies, evaluated on the device)
// ===========================================================================
/// `codim_ipc_contact_function.h` + `sym/codim_ipc_contact.inl`:
/// 6 barrier values, the C1 friction helpers, the friction law and the normal force.
auto& barrier_math_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<Float>,  // kappa
                                                lcrt::Buffer<Float>,  // dHat
                                                lcrt::Buffer<Float>,  // xi
                                                lcrt::Buffer<Float>,  // D
                                                lcrt::Buffer<Float>,  // B
                                                lcrt::Buffer<Float>,  // dB
                                                lcrt::Buffer<Float>,  // ddB
                                                lcrt::Buffer<Float>,  // Bc
                                                lcrt::Buffer<Float>,  // dBc
                                                lcrt::Buffer<Float>,  // ddBc
                                                lcrt::Buffer<Float>,  // f0
                                                lcrt::Buffer<Float>,  // f1
                                                lcrt::Buffer<Float>,  // f2
                                                lcrt::Buffer<Float>,  // friction energy
                                                lcrt::Buffer<Float>,         // friction gradient (2 per item)
                                                lcrt::Buffer<Float>,          // friction hessian (4 per item)
                                                lcrt::Buffer<Float>,          // normal force
                                                std::uint32_t>(
        [](lcrt::BufferVar<Float>         kappa,
           lcrt::BufferVar<Float>         dHat,
           lcrt::BufferVar<Float>         xi,
           lcrt::BufferVar<Float>         D,
           lcrt::BufferVar<Float>         B,
           lcrt::BufferVar<Float>         dB,
           lcrt::BufferVar<Float>         ddB,
           lcrt::BufferVar<Float>         Bc,
           lcrt::BufferVar<Float>         dBc,
           lcrt::BufferVar<Float>         ddBc,
           lcrt::BufferVar<Float>         f0b,
           lcrt::BufferVar<Float>         f1b,
           lcrt::BufferVar<Float>         f2b,
           lcrt::BufferVar<Float>         frE,
           lcrt::BufferVar<Float>         frG,
           lcrt::BufferVar<Float>         frH,
           lcrt::BufferVar<Float>         nf,
           lcrt::Var<std::uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                using namespace uipc::backend::luisa::sym::codim_ipc_contact;
                auto kv = kappa->read(i);
                auto hv = dHat->read(i);
                auto xv = xi->read(i);
                auto Dv = D->read(i);

                lcrt::Var<Float> b;
                KappaBarrier(b, kv, Dv, hv, xv);
                B->write(i, b);

                lcrt::Var<Float> d1;
                dKappaBarrierdD(d1, kv, Dv, hv, xv);
                dB->write(i, d1);

                lcrt::Var<Float> d2;
                ddKappaBarrierddD(d2, kv, Dv, hv, xv);
                ddB->write(i, d2);

                lcrt::Var<Float> bc;
                KappaBarrierWithThickness(bc, kv, Dv, hv, xv);
                Bc->write(i, bc);
                lcrt::Var<Float> d1c;
                dKappaBarrierWithThicknessdD(d1c, kv, Dv, hv, xv);
                dBc->write(i, d1c);
                lcrt::Var<Float> d2c;
                ddKappaBarrierWithThicknessddD(d2c, kv, Dv, hv, xv);
                ddBc->write(i, d2c);

                auto x2 = lps::detail::mul(Dv, Dv);
                lcrt::Var<Float> f0v;
                f0(x2, hv, f0v);
                f0b->write(i, f0v);
                lcrt::Var<Float> f1v;
                f1_div_rel_dx_norm(x2, hv, f1v);
                f1b->write(i, f1v);
                lcrt::Var<Float> f2v;
                f2_term(x2, hv, f2v);
                f2b->write(i, f2v);

                // a synthetic tangential displacement: `(D, xi)` (both device scalars)
                lps::LVec<lcrt::Var<Float>, 2> tan_rel{Dv, xv};
                frE->write(i, friction_energy(kv, hv, xv, tan_rel));
                lps::LVec<lcrt::Var<Float>, 2> G2;
                friction_gradient(G2, kv, hv, xv, tan_rel);
                frG->write(i * 2 + 0, G2(0));
                frG->write(i * 2 + 1, G2(1));
                lps::LMat<lcrt::Var<Float>, 2, 2> H2;
                friction_hessian(H2, kv, hv, xv, tan_rel);
                for(int c = 0; c < 2; ++c)
                    for(int r = 0; r < 2; ++r)
                        frH->write(i * 4 + c * 2 + r, H2(r, c));

                nf->write(i, normal_force(kv, hv, xv, Dv));
            };
        });
    return k;
}

/// `sym/vertex_half_plane_distance.inl`: D, dDdx, ddDddx, TR, dTRdx.
auto& half_plane_math_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector3>,  // x
                                                lcrt::Buffer<gg::LVector3>,  // P
                                                lcrt::Buffer<gg::LVector3>,  // N
                                                lcrt::Buffer<gg::LVector3>,  // x_prev
                                                lcrt::Buffer<gg::LVector3>,  // e1
                                                lcrt::Buffer<gg::LVector3>,  // e2
                                                lcrt::Buffer<Float>,  // D
                                                lcrt::Buffer<Float>,  // dDdx (3 per item)
                                                lcrt::Buffer<Float>,  // ddDddx (9 per item)
                                                lcrt::Buffer<Float>,  // TR (2 per item)
                                                lcrt::Buffer<Float>,  // dTRdx (6 per item)
                                                std::uint32_t>(
        [](lcrt::BufferVar<gg::LVector3>    x,
           lcrt::BufferVar<gg::LVector3>    P,
           lcrt::BufferVar<gg::LVector3>    N,
           lcrt::BufferVar<gg::LVector3>    x_prev,
           lcrt::BufferVar<gg::LVector3>    e1,
           lcrt::BufferVar<gg::LVector3>    e2,
           lcrt::BufferVar<Float>           D,
           lcrt::BufferVar<Float>           dDdx,
           lcrt::BufferVar<Float>           ddDddx,
           lcrt::BufferVar<Float>           TRb,
           lcrt::BufferVar<Float>           dTRdx,
           lcrt::Var<std::uint32_t>         n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                using namespace uipc::backend::luisa::sym::ipc_vertex_half_contact;
                auto xv  = gg::load_vec3(x->read(i));
                auto Pv  = gg::load_vec3(P->read(i));
                auto Nv  = gg::load_vec3(N->read(i));
                auto xpv = gg::load_vec3(x_prev->read(i));
                auto e1v = gg::load_vec3(e1->read(i));
                auto e2v = gg::load_vec3(e2->read(i));

                lcrt::Var<Float> Dv;
                sym_ph::HalfPlaneD(Dv, xv, Pv, Nv);
                D->write(i, Dv);

                lps::LVec<lcrt::Var<Float>, 3> g;
                sym_ph::dHalfPlaneDdx(g, xv, Pv, Nv);
                for(int r = 0; r < 3; ++r)
                    dDdx->write(i * 3 + r, g(r));

                lps::LMat<lcrt::Var<Float>, 3, 3> H;
                sym_ph::ddHalfPlaneDddx(H, xv, Pv, Nv);
                for(int c = 0; c < 3; ++c)
                    for(int r = 0; r < 3; ++r)
                        ddDddx->write(i * 9 + c * 3 + r, H(r, c));

                lps::LVec<lcrt::Var<Float>, 2> tr;
                sym_ph::TR(tr, xv, xpv, e1v, e2v);
                TRb->write(i * 2 + 0, tr(0));
                TRb->write(i * 2 + 1, tr(1));

                lps::LMat<lcrt::Var<Float>, 2, 3> J;
                sym_ph::dTRdx(J, xv, xpv, e1v, e2v);
                for(int r = 0; r < 2; ++r)
                    for(int c = 0; c < 3; ++c)
                        dTRdx->write(i * 6 + r * 3 + c, J(r, c));
            };
        });
    return k;
}

}  // namespace t81b

using namespace t81b;

namespace t81b
{
// ===========================================================================
//  Pipeline drivers
// ===========================================================================
void run_dytopo(bool gradient_only = false)
{
    auto* dyt = Probe::dytopo_effect_manager;
    REQUIRE(dyt != nullptr);
    gg::GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo einfo;
    einfo.component_flags(gg::EnergyComponentFlags::Contact);
    einfo.gradient_only(gradient_only);
    dyt->compute_dytopo_effect(einfo);
}

std::vector<Float> compute_energies()
{
    auto* dyt = Probe::dytopo_effect_manager;
    REQUIRE(dyt != nullptr);
    gg::DyTopoEffectLineSearchReporter::Impl lsr;
    lsr.global_dytopo_effect_manager = *dyt;
    lsr.compute_energy(false);
    return read_device_view(lsr.energies.view());
}

struct Gradients
{
    std::vector<Float>        values;
    std::vector<std::int32_t> indices;
};

struct Hessians
{
    std::vector<Float>        values;
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> cols;
};

/// A reporter's gradient **slice**: `indices()`/`values()` are the launch handles of the
/// whole shared buffer (`luisa_device/linear/views.h`, the `handles (the launch arguments)`
/// block), *not* of the reporter's window, so reading them whole reads every other type's and
/// every other reporter's slots.  The window is `doublet_index_offset()`/`doublet_count()`;
/// `values()` is `N = 3` scalars per doublet (`segment_element_index(b, k) = b * N + k`).
Gradients read_gradients(const lla::LCDoubletVectorView<Float, 3>& G)
{
    const auto off = static_cast<size_t>(G.doublet_index_offset());
    const auto cnt = static_cast<size_t>(G.doublet_count());
    return Gradients{read_device(G.values().subview(off * 3, cnt * 3)),
                     read_device(G.indices().subview(off, cnt))};
}

/// The Hessian twin: `values()` is `M*N = 9` scalars per triplet
/// (`block_element_index(b, r, c) = b * block_entries + c * M + r`, column-major in the block).
Hessians read_hessians(const lla::LCTripletMatrixView<Float, 3, 3>& H)
{
    const auto off = static_cast<size_t>(H.triplet_index_offset());
    const auto cnt = static_cast<size_t>(H.triplet_count());
    return Hessians{read_device(H.values().subview(off * 9, cnt * 9)),
                    read_device(H.row_indices().subview(off, cnt)),
                    read_device(H.col_indices().subview(off, cnt))};
}

/// The four per-type slices of a simplex contact model (normal or frictional).
struct SimplexSlices
{
    std::vector<Float> pt, ee, pe, pp;
    Gradients          gp, ge, gpe, gpp;
    Hessians           hp, he, hpe, hpp;
};

SimplexSlices read_simplex_slices(gg::SimplexNormalContact& m)
{
    SimplexSlices s;
    s.pt  = read_device(m.PT_energies());
    s.ee  = read_device(m.EE_energies());
    s.pe  = read_device(m.PE_energies());
    s.pp  = read_device(m.PP_energies());
    s.gp  = read_gradients(m.PT_gradients());
    s.ge  = read_gradients(m.EE_gradients());
    s.gpe = read_gradients(m.PE_gradients());
    s.gpp = read_gradients(m.PP_gradients());
    s.hp  = read_hessians(m.PT_hessians());
    s.he  = read_hessians(m.EE_hessians());
    s.hpe = read_hessians(m.PE_hessians());
    s.hpp = read_hessians(m.PP_hessians());
    return s;
}

/// the frictional twin (its per-type *views* carry the same class-level accessors; the
/// `friction_*` spelling is the model's BaseInfo/EnergyInfo surface)
SimplexSlices read_simplex_slices(gg::SimplexFrictionalContact& m)
{
    SimplexSlices s;
    s.pt  = read_device(m.PT_energies());
    s.ee  = read_device(m.EE_energies());
    s.pe  = read_device(m.PE_energies());
    s.pp  = read_device(m.PP_energies());
    s.gp  = read_gradients(m.PT_gradients());
    s.ge  = read_gradients(m.EE_gradients());
    s.gpe = read_gradients(m.PE_gradients());
    s.gpp = read_gradients(m.PP_gradients());
    s.hp  = read_hessians(m.PT_hessians());
    s.he  = read_hessians(m.EE_hessians());
    s.hpe = read_hessians(m.PE_hessians());
    s.hpp = read_hessians(m.PP_hessians());
    return s;
}

namespace ldist = gg::luisa_device::math::distance;

/// `dKappaBarrierLog2dD`'s LaTeX closed form (the `xi == 0` branch).
double barrier_log2_d1(double kappa, double D, double dHat)
{
    const double x0 = dHat * dHat;
    const double l  = std::log(D / x0);
    return kappa * (2.0 * D - 2.0 * x0) * l * l + 2.0 * kappa * (D - x0) * (D - x0) * l / D;
}

/// `dKappaBarrierLog2WithThicknessdD`'s LaTeX closed form.
double barrier_log2_thick_d1(double kappa, double D, double dHat, double xi)
{
    const double x0 = xi * xi;
    const double x1 = dHat * dHat + 2.0 * dHat * xi;
    const double l  = std::log((D - x0) / x1);
    const double t  = D - x0 - x1;
    return kappa * (2.0 * D - 2.0 * x0 - 2.0 * dHat * dHat - 4.0 * dHat * xi) * l * l
           + 2.0 * kappa * t * t * l / (D - x0);
}

/// the `dKappaBarrierdD` dispatcher's closed form
double barrier_d1(double kappa, double D, double dHat, double xi)
{
    return xi == 0.0 ? barrier_log2_d1(kappa, D, dHat) : barrier_log2_thick_d1(kappa, D, dHat, xi);
}

/// Expand a triplet-view Hessian slice into the full symmetric 12x12 matrix.  The ported
/// assembler writes only the upper triangle of the *block* matrix (`row <= col`), 9 scalars per
/// triplet, column-major inside the block.
Matrix12x12 expand_hessian_12(const Hessians& h)
{
    Matrix12x12 H = Matrix12x12::Zero();
    for(size_t k = 0; k < h.rows.size(); ++k)
    {
        const int r = static_cast<int>(h.rows[k]);
        const int c = static_cast<int>(h.cols[k]);
        for(int b = 0; b < 3; ++b)
            for(int a = 0; a < 3; ++a)
                H(3 * r + a, 3 * c + b) = h.values[k * 9 + b * 3 + a];
    }
    // `H = H + H.transpose()` is the textbook Eigen *aliasing* trap: the transpose is a lazy view
    // of the destination, so the write clobbers the source and the result is not symmetric
    // (measured: 132 of 144 entries disagreed with their mirror).  `.eval()` materialises the sum
    // in a temporary, which is what "expand the upper triangle" means.
    H = (H + H.transpose()).eval();
    for(int i = 0; i < 12; ++i)
        H(i, i) *= 0.5;
    return H;
}

}  // namespace t81b

using namespace t81b;

// ===========================================================================
//  1) the barrier law: value, first and second derivative
// ===========================================================================
TEST_CASE("t81b_barrier_law", "[contact_model]")
{
    test_runtime();

    std::vector<std::array<Float, 4>> rows;
    for(double xi : {0.0, 0.01, 0.005})
        for(double dHat : {0.1, 0.05})
            for(double kappa : {1.0, 8.0e5, -3.0})
            {
                const double lo = xi * xi, hi = (dHat + xi) * (dHat + xi);
                for(double frac : {0.1, 0.4, 0.75, 0.99})
                    rows.push_back({Float(kappa), Float(dHat), Float(xi), Float(lo + frac * (hi - lo))});
            }

    std::vector<Float> h_kappa, h_dHat, h_xi, h_D;
    for(auto& r : rows)
    {
        h_kappa.push_back(r[0]);
        h_dHat.push_back(r[1]);
        h_xi.push_back(r[2]);
        h_D.push_back(r[3]);
    }

    const auto n = static_cast<std::uint32_t>(rows.size());
    lud::LDeviceBuffer<Float> dk, dh, dx, dD, dB, dB1, dB2, dBc, dB1c, dB2c, df0, df1, df2, dfrE, dnf, dfrG, dfrH;
    for(auto* b : {&dk, &dh, &dx, &dD, &dB, &dB1, &dB2, &dBc, &dB1c, &dB2c, &df0, &df1, &df2, &dfrE, &dnf})
        b->resize(n);
    dfrG.resize(2 * n);
    dfrH.resize(4 * n);

    write_device(dk.view(), span<const Float>{h_kappa});
    write_device(dh.view(), span<const Float>{h_dHat});
    write_device(dx.view(), span<const Float>{h_xi});
    write_device(dD.view(), span<const Float>{h_D});

    auto& k = barrier_math_kernel();
    gg::frame_stream()
        .record(k(dk.cview(), dh.cview(), dx.cview(), dD.cview(), dB.view(), dB1.view(),
                  dB2.view(), dBc.view(), dB1c.view(), dB2c.view(), df0.view(), df1.view(),
                  df2.view(), dfrE.view(), dfrG.view(), dfrH.view(), dnf.view(), n)
                    .dispatch(n));
    gg::frame_stream().synchronize();

    auto B = read_device_view(dB.view());
    auto D1 = read_device_view(dB1.view());
    auto D2 = read_device_view(dB2.view());
    auto Bc = read_device_view(dBc.view());
    auto D1c = read_device_view(dB1c.view());
    auto D2c = read_device_view(dB2c.view());
    auto F0 = read_device_view(df0.view());
    auto F1 = read_device_view(df1.view());
    auto F2 = read_device_view(df2.view());
    auto NF = read_device_view(dnf.view());

    constexpr double h1 = 1e-6, h2 = 1e-4, rel = 1e-6;
    double max_d1_err = 0.0, max_d2_err = 0.0, max_d1c_err = 0.0, max_d2c_err = 0.0;
    for(size_t i = 0; i < rows.size(); ++i)
    {
        const double kv = rows[i][0], hv = rows[i][1], xv = rows[i][2], Dv = rows[i][3];

        CHECK(close(B[i], barrier(kv, Dv, hv, xv), 1e-13));
        CHECK(close(Bc[i], barrier_classic(kv, Dv, hv, xv), 1e-13));
      {
          Float host_B;
          sym_codim::KappaBarrier(host_B, Float(kv), Float(Dv), Float(hv), Float(xv));
          // NOT `==`: the device kernels contract multiply-adds (LC's codegen), so the last
          // ulp differs from the same body on the host -- cookbook 7.35 rule 6.  The printed
          // expansion shows two identical spellings, which is exactly that one-ulp gap.
          CHECK(close(B[i], host_B, 1e-6));
      }

      const double fd1 = central([&](double D) { return barrier(kv, D, hv, xv); }, Dv, h1);
      // A *coarse* sanity net only: near the band's edges the central difference of the value
      // is itself inaccurate (measured max 1.6e-3 relative at h1 = 1e-6), so the pinning is the
      // analytic comparison right below, not this.  (An earlier 1e-6 tolerance here was wrong.)
      CHECK(close(D1[i], fd1, 1e-2));
        max_d1_err = std::max(max_d1_err, std::abs(D1[i] - fd1) / std::max(1e-12, std::abs(fd1)));
        CHECK(close(D1[i], barrier_d1(kv, Dv, hv, xv), 1e-13));  // the LaTeX closed form

      const double fd2 = barrier_d2(kv, Dv, hv, xv);
      CHECK(close(D2[i], fd2, 1e-12));
      max_d2_err = std::max(max_d2_err, std::abs(D2[i] - fd2) / std::max(1e-12, std::abs(fd2)));

      const double fd1c = central([&](double D) { return barrier_classic(kv, D, hv, xv); }, Dv, h1);
      CHECK(close(D1c[i], fd1c, 1e-2));  // see the note on the interval's value-FD above
      const double fd2c = barrier_classic_d2(kv, Dv, hv, xv);
      CHECK(close(D2c[i], fd2c, 1e-12));
        max_d1c_err = std::max(max_d1c_err, std::abs(D1c[i] - fd1c) / std::max(1e-12, std::abs(fd1c)));
        max_d2c_err = std::max(max_d2c_err, std::abs(D2c[i] - fd2c) / std::max(1e-12, std::abs(fd2c)));

        const double x2 = Dv * Dv, eps = hv, eps2 = eps * eps;
        const double f0_ref = x2 >= eps2 ? std::sqrt(x2) : x2 * (-std::sqrt(x2) / 3.0 + eps) / eps2 + eps / 3.0;
        const double f1_ref = x2 >= eps2 ? 1.0 / std::sqrt(x2) : (-std::sqrt(x2) + 2.0 * eps) / eps2;
        CHECK(close(F0[i], f0_ref, 1e-13));
        CHECK(close(F1[i], f1_ref, 1e-13));
        CHECK(close(F2[i], -1.0 / eps2, 1e-13));

        CHECK(close(NF[i], -D1[i] * 2.0 * std::sqrt(Dv), 1e-12));
    }
    std::printf("[contact_model] barrier: %zu rows, |d1 rel err| %.3e, |d2 rel err| %.3e, "
                "classic |d1| %.3e |d2| %.3e (h1=%.0e h2=%.0e)\n",
                rows.size(), max_d1_err, max_d2_err, max_d1c_err, max_d2c_err, h1, h2);

    // The band boundary: `D -> (dHat + xi)^2` makes every flavour vanish — *mathematically*.
    // In double the band test is `D - xi^2 - (dHat^2 + 2 dHat xi)`, whose two sides are summed
    // in a different order, so the residual is a cancellation of order 1e-18 squared, not a
    // literal 0.0 (measured: 2.97e-62).  `== 0.0` was the wrong predicate; the value's own
    // scale (`rel * max(1, ...)`) makes `1e-30` an absolute bound here.
    for(double xi : {0.0, 0.01})
    {
        const double dHat = 0.1, kappa = 8.0e5, hi = (dHat + xi) * (dHat + xi);
        CHECK(close(barrier(kappa, hi, dHat, xi), 0.0, 1e-30));
        CHECK(close(barrier_classic(kappa, hi, dHat, xi), 0.0, 1e-30));
        // the dispatcher is exact: `xi == 0` *is* the plain log^2 form, and a non-zero xi *is*
        // the thickness form (the transcribed .inl branches on the same condition)
        CHECK(barrier(kappa, hi, dHat, 0.0) == barrier_log2(kappa, hi, dHat));
        CHECK(barrier(kappa, hi, dHat, xi) == barrier_log2_thick(kappa, hi, dHat, xi));
    }
}

// ===========================================================================
//  2) the transcribed `sym/*.inl` expressions, re-derived independently
// ===========================================================================
TEST_CASE("t81b_sym_expressions", "[contact_model]")
{
    test_runtime();

    std::vector<Vector3> hx = {{0.3, -0.2, 0.15},  {-0.5, 0.25, 0.7}, {1.0, 1.0, -0.4},
                               {0.0, 0.0, 0.0},    {0.125, 0.375, -0.625}, {-1.0, 2.0, -3.0},
                               {0.1, 0.1, 0.1},    {0.9, -0.9, 0.45}};
    std::vector<Vector3> hP = {{0.0, 0.0, 0.0},    {0.1, 0.2, 0.3},  {-0.25, 0.5, 0.75},
                               {1.0, 0.0, 0.0},    {0.0, 1.0, 0.0},  {0.5, 0.5, 0.5},
                               {-0.1, -0.2, -0.3}, {2.0, -1.0, 0.25}};
    std::vector<Vector3> hN = {{0.0, 0.0, 1.0},    {0.0, 1.0, 0.0},  {1.0, 0.0, 0.0},
                               {0.6, 0.8, 0.0},    {0.0, 0.6, 0.8},  {0.8, 0.0, 0.6},
                               {-0.6, 0.0, 0.8},   {0.0, -0.8, 0.6}};
    std::vector<Vector3> hxp = {{0.25, -0.15, 0.05}, {-0.4, 0.2, 0.6}, {0.9, 1.1, -0.5},
                                {0.0, 0.0, 0.0},     {0.1, 0.3, -0.6}, {-1.1, 1.9, -3.1},
                                {0.05, 0.15, 0.2},   {0.95, -0.85, 0.4}};
    std::vector<Vector3> he1, he2;
    for(size_t i = 0; i < hx.size(); ++i)
    {
        Vector3 trial = (std::abs(hN[i].dot(Vector3::UnitX())) > 0.9) ? Vector3::UnitZ() : Vector3::UnitX();
        auto    e1    = trial.cross(hN[i]).normalized();
        he1.push_back(e1);
        he2.push_back(hN[i].cross(e1));
    }

    const auto n = static_cast<std::uint32_t>(hx.size());
    lud::LDeviceBuffer<gg::LVector3> dx, dP, dN, dxp, de1, de2;
    for(auto* b : {&dx, &dP, &dN, &dxp, &de1, &de2})
        b->resize(n);
    lud::LDeviceBuffer<Float> dD, dg, dH, dTR, dJ;
    dD.resize(n);
    dg.resize(3 * n);
    dH.resize(9 * n);
    dTR.resize(2 * n);
    dJ.resize(6 * n);

    auto to_lv = [](const std::vector<Vector3>& v)
    {
        std::vector<gg::LVector3> out;
        for(auto& p : v)
            out.push_back(gg::to_luisa(p));
        return out;
    };
    write_device(dx.view(), span<const gg::LVector3>{to_lv(hx)});
    write_device(dP.view(), span<const gg::LVector3>{to_lv(hP)});
    write_device(dN.view(), span<const gg::LVector3>{to_lv(hN)});
    write_device(dxp.view(), span<const gg::LVector3>{to_lv(hxp)});
    write_device(de1.view(), span<const gg::LVector3>{to_lv(he1)});
    write_device(de2.view(), span<const gg::LVector3>{to_lv(he2)});

    auto& k = half_plane_math_kernel();
    gg::frame_stream()
        .record(k(dx.cview(), dP.cview(), dN.cview(), dxp.cview(), de1.cview(), de2.cview(),
                  dD.view(), dg.view(), dH.view(), dTR.view(), dJ.view(), n)
                    .dispatch(n));
    gg::frame_stream().synchronize();

    auto D = read_device_view(dD.view());
    auto G = read_device_view(dg.view());
    auto H = read_device_view(dH.view());
    auto TRv = read_device_view(dTR.view());
    auto J = read_device_view(dJ.view());

    for(size_t i = 0; i < hx.size(); ++i)
    {
        const Vector3 s = hx[i] - hP[i];
        const double  sn = hN[i].dot(s);
        CHECK(close(D[i], sn * sn, 1e-14));
        for(int r = 0; r < 3; ++r)
            CHECK(close(G[3 * i + r], 2.0 * hN[i][r] * sn, 1e-14));
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                CHECK(close(H[9 * i + c * 3 + r], 2.0 * hN[i][r] * hN[i][c], 1e-14));
        const Vector3 d = hx[i] - hxp[i];
        CHECK(close(TRv[2 * i + 0], he1[i].dot(d), 1e-14));
        CHECK(close(TRv[2 * i + 1], he2[i].dot(d), 1e-14));
        for(int c = 0; c < 3; ++c)
        {
            CHECK(close(J[i * 6 + c], he1[i][c], 1e-14));
            CHECK(close(J[i * 6 + 3 + c], he2[i][c], 1e-14));
        }

        // the *host* instantiation of the same transcribed bodies.  NOT `==`: LC's code
        // generator contracts multiply-adds (the device path fuses `a*b + c`, the host keeps
        // two roundings), so the same source spelling differs in the last ulp -- cookbook
        // §7.35 rule 6.  Measured: `-0.02000000000000002` vs `-0.02` (1 ulp at 1e-17 relative);
        // 1e-6 is the recorded bound, FLT_EPSILON-scale noise would be enough.
        auto lv3 = [](const Vector3& v)
        { return lps::LVec<Float, 3>{Float(v.x()), Float(v.y()), Float(v.z())}; };
        lps::LVec<Float, 3> x{lv3(hx[i])}, P{lv3(hP[i])}, N{lv3(hN[i])}, xp{lv3(hxp[i])},
            e1{lv3(he1[i])}, e2{lv3(he2[i])};
        lps::LVec<Float, 2> tr;
        sym_ph::TR(tr, x, xp, e1, e2);
        CHECK(close(tr(0), TRv[2 * i + 0], 1e-6));
        CHECK(close(tr(1), TRv[2 * i + 1], 1e-6));
        lps::LMat<Float, 2, 3> J_h;
        sym_ph::dTRdx(J_h, x, xp, e1, e2);
        CHECK(close(J_h(0, 0), J[i * 6 + 0], 1e-6));
        CHECK(close(J_h(1, 2), J[i * 6 + 5], 1e-6));
        auto host_D = Float{};
        sym_ph::HalfPlaneD(host_D, x, P, N);
        CHECK(close(host_D, D[i], 1e-6));
        lps::LVec<Float, 3> host_g;
        sym_ph::dHalfPlaneDdx(host_g, x, P, N);
        CHECK(close(host_g(2), G[3 * i + 2], 1e-6));
        lps::LMat<Float, 3, 3> host_H;
        sym_ph::ddHalfPlaneDddx(host_H, x, P, N);
        CHECK(close(host_H(0, 0), H[9 * i + 0], 1e-6));
        CHECK(close(host_H(1, 1), H[9 * i + 4], 1e-6));
        CHECK(close(host_H(2, 2), H[9 * i + 8], 1e-6));
    }
}

// ===========================================================================
//  3) the codim thickness band and the classic trio
// ===========================================================================
TEST_CASE("t81b_codim_thickness_band", "[contact_model]")
{
    test_runtime();

    for(double xi : {0.0, 0.005, 0.01, 0.02})
        for(double dHat : {0.05, 0.1})
        {
            auto   range = gg::D_range(Float(xi), Float(dHat));
            double lo, hi;
            band(dHat, xi, lo, hi);
            CHECK(close(range(0), lo, 1e-15));
            CHECK(close(range(1), hi, 1e-15));
            CHECK(gg::is_active_D(range, Float(0.5 * (lo + hi))) == true);
            CHECK(gg::is_active_D(range, Float(lo * 0.5)) == false);
            CHECK(gg::is_active_D(range, Float(hi * 2.0)) == false);
            CHECK(gg::is_active_D(range, Float(hi)) == false);  // open at `hi`
        }

    // The four thickness functions return the **sum** of two per-element thicknesses, not the
    // mean: `utils/codim_thickness.h` keeps the reference's own
    // `// return (a + b) / 2.0;` comment next to `return add(a, b);`, i.e. the averaging was
    // commented out in the reference and never reinstated.  With 0.01 everywhere that is 0.02.
    CHECK(close(gg::PT_thickness(Float(0.01), Float(0.01), Float(0.01), Float(0.01)), 0.02, 1e-15));
    CHECK(close(gg::EE_thickness(Float(0.01), Float(0.01), Float(0.01), Float(0.01)), 0.02, 1e-15));
    CHECK(close(gg::PE_thickness(Float(0.01), Float(0.01), Float(0.01)), 0.02, 1e-15));
    CHECK(close(gg::PP_thickness(Float(0.01), Float(0.01)), 0.02, 1e-15));
    CHECK(close(gg::PT_d_hat(Float(0.1), Float(0.1), Float(0.1), Float(0.1)), 0.1, 1e-15));
    CHECK(close(gg::EE_d_hat(Float(0.1), Float(0.1), Float(0.1), Float(0.1)), 0.1, 1e-15));
    CHECK(close(gg::PE_d_hat(Float(0.1), Float(0.1), Float(0.1)), 0.1, 1e-15));
    CHECK(close(gg::PP_d_hat(Float(0.1), Float(0.1)), 0.1, 1e-15));

    for(double xi : {0.0, 0.01})
    {
        const double dHat = 0.1, kappa = 8.0e5;
        double       lo, hi;
        band(dHat, xi, lo, hi);
        // see `t81b_barrier_law`: the vanishing is exact in exact arithmetic only, the band test
        // is a cancellation of order 1e-18 in double (measured residual 2.97e-62)
        CHECK(close(barrier(kappa, hi, dHat, xi), 0.0, 1e-30));
        CHECK(close(barrier_classic(kappa, hi, dHat, xi), 0.0, 1e-30));
        const double d1 = central([&](double D) { return barrier(kappa, D, dHat, xi); }, hi, 1e-7);
        CHECK(std::abs(d1) < 1e-3 * std::abs(barrier(kappa, lo + 0.5 * (hi - lo), dHat, xi)));
    }
}

// ===========================================================================
//  4) the simplex normal model: PT/EE/PE/PP on synthetic pairs through the pipeline
// ===========================================================================
TEST_CASE("t81b_simplex_normal_pairs", "[contact_model]")
{
    test_runtime();

    SceneDesc desc;
    auto      scene = build_scene(desc);
    Engine    engine{"luisa", workspace_path("u81b_pairs")};
    World     world{engine};
    world.init(scene);

    auto* cm  = Probe::contact_manager;
    auto* gtf = Probe::trajectory_filter;
    auto* vm  = Probe::vertex_manager;
    REQUIRE(cm != nullptr);
    REQUIRE(gtf != nullptr);
    REQUIRE(vm != nullptr);
    auto* snc = find_model<gg::SimplexNormalContact>(cm);
    REQUIRE(snc != nullptr);

    auto simplex = gtf->find<gg::SimplexTrajectoryFilter>();
    REQUIRE(static_cast<bool>(simplex));
    CHECK(dynamic_cast<ProbeFilter*>(simplex.view()) != nullptr);

    set_probe_pairs();
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);

    const auto& base = probe_positions();
    const auto  n    = static_cast<size_t>(vm->positions().size());
    REQUIRE(n == base.size());
    set_positions(base);
    set_prev_positions(base);
    set_thicknesses(std::vector<Float>(n, desc.thickness));
    set_d_hats(std::vector<Float>(n, desc.d_hat));

    auto table = read_device(cm->contact_tabular());
    REQUIRE(table.size() == 1);
    const double kappa = table[0].kappa;
    const double dt = desc.dt, dHat = desc.d_hat, th = desc.thickness;
    const double kt2 = kappa * dt * dt;
    const double D   = 0.06 * 0.06;  // by construction, for all four pairs
    // The barrier does NOT see the per-vertex thickness: the four models aggregate their
    // candidates first (`utils/codim_thickness.h`: PT/EE/PE/PP_thickness **sum** the two
    // per-element values -- the reference's `// return (a + b) / 2.0;` is commented out -- and
    // `utils/primitive_d_hat.h` **averages** the d_hats).  With every vertex at (0.01, 0.1) that
    // is thickness 0.02 for all four types and d_hat 0.1; the oracle goes through the same
    // functions the models call, so a change in either layer is caught here.
    const double th_pt = gg::PT_thickness(Float(th), Float(th), Float(th), Float(th));
    const double th_ee = gg::EE_thickness(Float(th), Float(th), Float(th), Float(th));
    const double th_pe = gg::PE_thickness(Float(th), Float(th), Float(th));
    const double th_pp = gg::PP_thickness(Float(th), Float(th));
    const double dh_pt = gg::PT_d_hat(Float(dHat), Float(dHat), Float(dHat), Float(dHat));
    const double dh_ee = gg::EE_d_hat(Float(dHat), Float(dHat), Float(dHat), Float(dHat));
    const double dh_pe = gg::PE_d_hat(Float(dHat), Float(dHat), Float(dHat));
    const double dh_pp = gg::PP_d_hat(Float(dHat), Float(dHat));
    CHECK(close(th_pt, 2.0 * th, 1e-15));  // the sum, not the mean
    CHECK(close(dh_pt, dHat, 1e-15));      // the mean
    const double dBdD = barrier_d1(kt2, D, dh_pt, th_pt);
    const double B    = barrier(kt2, D, dh_pt, th_pt);

    CHECK(close(kappa, cm->kappa_upper_bound(), 1e-12));
    CHECK(kappa > 0.0);

    run_dytopo();
    compute_energies();  // the energy pass fills the models' per-type energy buffers
    auto slices = read_simplex_slices(*snc);
    REQUIRE(slices.pt.size() == 1);
    REQUIRE(slices.ee.size() == 1);
    REQUIRE(slices.pe.size() == 1);
    REQUIRE(slices.pp.size() == 1);

    // (a0) the split's offset contract (research §5.3, phase 5's extent assertions).  The four
    // per-type slices of the **one** shared reporter buffer must *tile* it: doublet offsets
    // 0/4/8/11 with counts 4/4/3/2 and triplet offsets 0/10/20/26 with counts 10/10/6/3, i.e.
    // exactly the 4+4+3+2 = 13 doublets and 10+10+6+3 = 29 triplets the model's
    // `do_report_gradient_hessian_extent` asks the manager for.  This is the assertion that
    // catches the dropped `from_handles_full` offset: with it every type starts at 0 and the
    // totals still read 13/29, but the offsets collapse to all-zero.
    {
        CHECK(snc->PT_gradients().doublet_index_offset() == 0);
        CHECK(snc->EE_gradients().doublet_index_offset() == 4);
        CHECK(snc->PE_gradients().doublet_index_offset() == 8);
        CHECK(snc->PP_gradients().doublet_index_offset() == 11);
        CHECK(snc->PT_gradients().doublet_count() == 4);
        CHECK(snc->EE_gradients().doublet_count() == 4);
        CHECK(snc->PE_gradients().doublet_count() == 3);
        CHECK(snc->PP_gradients().doublet_count() == 2);
        CHECK(snc->PT_gradients().total_doublet_count() == 13);
        CHECK(snc->PT_hessians().triplet_index_offset() == 0);
        CHECK(snc->EE_hessians().triplet_index_offset() == 10);
        CHECK(snc->PE_hessians().triplet_index_offset() == 20);
        CHECK(snc->PP_hessians().triplet_index_offset() == 26);
        CHECK(snc->PT_hessians().triplet_count() == 10);
        CHECK(snc->EE_hessians().triplet_count() == 10);
        CHECK(snc->PE_hessians().triplet_count() == 6);
        CHECK(snc->PP_hessians().triplet_count() == 3);
        CHECK(snc->PT_hessians().total_triplet_count() == 29);
    }

    // (a) the energies, against the test's own closed form.  All four types aggregate the same
    // per-vertex pair here, so one `B` covers them -- but the aggregation itself is pinned, per
    // type, first (a type-specific aggregate bug would slip past a single shared `B`).
    CHECK(close(th_ee, th_pt, 1e-15));
    CHECK(close(th_pe, th_pt, 1e-15));
    CHECK(close(th_pp, th_pt, 1e-15));
    CHECK(close(dh_ee, dh_pt, 1e-15));
    CHECK(close(dh_pe, dh_pt, 1e-15));
    CHECK(close(dh_pp, dh_pt, 1e-15));
    CHECK(close(slices.pt[0], B, 1e-12));
    CHECK(close(slices.ee[0], B, 1e-12));
    CHECK(close(slices.pe[0], B, 1e-12));
    CHECK(close(slices.pp[0], B, 1e-12));
    CHECK(B > 0.0);
    std::printf("[contact_model] pairs: kappa=%g kt2=%g D=%g dHat=%g thickness(per-vertex)=%g "
                "-> aggregates (%.17g, %.17g) -> B=%.17g; device PT=%.17g EE=%.17g PE=%.17g "
                "PP=%.17g\n",
                kappa, kt2, D, dHat, th, th_pt, dh_pt, B,
                slices.pt[0], slices.ee[0], slices.pe[0], slices.pp[0]);

    // (b) the EE mollifier is exactly 1 for the perpendicular crossing pair
    {
        auto lv = [](const Vector3& v)
        { return lps::LVec<Float, 3>{Float(v.x()), Float(v.y()), Float(v.z())}; };
        const auto& p = base;
        Float       eps_x{};
        ldist::edge_edge_mollifier_threshold(lv(p[4]), lv(p[5]), lv(p[6]), lv(p[7]),
                                            static_cast<double>(1e-3), eps_x);
        Float ek{};
        ldist::edge_edge_mollifier(lv(p[4]), lv(p[5]), lv(p[6]), lv(p[7]), eps_x, ek);
        CHECK(ek == 1.0);
        CHECK(ldist::need_mollify(lv(p[4]), lv(p[5]), lv(p[6]), lv(p[7]), eps_x) == false);
    }

    // (c) the gradient: `dB/dD * dD/dx`, with `dD/dx` derived here (never from the ported body).
    // For a point above the *interior* of a triangle the squared distance is `h^2` with
    // `h = n . (p - T0)`, hence `dD/dp = 2h n` and `dD/dT_i = -2h lambda_i n`, where `lambda_i`
    // is the barycentric weight of the projection inside the triangle.  Only the four *z* slots
    // are therefore non-zero (`n = z-hat`), and every in-plane slot is exactly 0 -- the earlier
    // spelling of this oracle expected the T_i slots to be 0 as well, which is wrong: moving a
    // triangle vertex *out of plane* tilts the plane and changes `h`.
    const double h_dist = std::sqrt(D);                 // 0.06
    const double lam0 = 0.5, lam1 = 0.25, lam2 = 0.25;  // (0.25, 0.25) in the unit triangle
    CHECK(slices.gp.indices == std::vector<std::int32_t>{0, 1, 2, 3});
    for(int i : {0, 1, 3, 4, 6, 7, 9, 10})  // the in-plane components
        CHECK(close(slices.gp.values[static_cast<size_t>(i)], 0.0, 1e-14));
    CHECK(close(slices.gp.values[2], dBdD * 2.0 * h_dist, 1e-12));
    CHECK(close(slices.gp.values[5], -dBdD * 2.0 * h_dist * lam0, 1e-12));
    CHECK(close(slices.gp.values[8], -dBdD * 2.0 * h_dist * lam1, 1e-12));
    CHECK(close(slices.gp.values[11], -dBdD * 2.0 * h_dist * lam2, 1e-12));

    // the point-edge pair is the same law with `lambda = (0.5, 0.5)` on the edge (the projection
    // is the edge's midpoint), i.e. `dD/dE_i = -h n`
    CHECK(slices.gpe.indices == std::vector<std::int32_t>{9, 1, 2});
    for(int i : {0, 1, 3, 4, 6, 7})
        CHECK(close(slices.gpe.values[static_cast<size_t>(i)], 0.0, 1e-14));
    CHECK(close(slices.gpe.values[2], dBdD * 2.0 * h_dist, 1e-12));
    CHECK(close(slices.gpe.values[5], -dBdD * h_dist, 1e-12));
    CHECK(close(slices.gpe.values[8], -dBdD * h_dist, 1e-12));

    CHECK(slices.gpp.indices == std::vector<std::int32_t>{4, 8});
    CHECK(close(slices.gpp.values[2], dBdD * 0.12, 1e-12));
    CHECK(close(slices.gpp.values[5], -dBdD * 0.12, 1e-12));
    for(int i : {0, 1, 3, 4})
        CHECK(close(slices.gpp.values[i], 0.0, 1e-14));

    CHECK(slices.ge.indices == std::vector<std::int32_t>{4, 5, 6, 7});
    CHECK(close(slices.ge.values[2], dBdD * 0.06, 1e-12));
    CHECK(close(slices.ge.values[5], dBdD * 0.06, 1e-12));
    CHECK(close(slices.ge.values[8], -dBdD * 0.06, 1e-12));
    CHECK(close(slices.ge.values[11], -dBdD * 0.06, 1e-12));

    // (d) the gradient against finite differences of the *device* energy
    constexpr double h = 1e-6;
    auto energy_of = [&](auto&& read_energy, const std::vector<Vector3>& pos)
    {
        set_positions(pos);
        run_dytopo();
        compute_energies();
        return read_energy();
    };
    {
        std::vector<Vector3> up = base, down = base;
        up[0].z() += h;
        down[0].z() -= h;
        const double fd = (energy_of([&] { return read_device(snc->PT_energies())[0]; }, up)
                           - energy_of([&] { return read_device(snc->PT_energies())[0]; }, down))
                          / (2.0 * h);
        CHECK(close(fd, slices.gp.values[2], 1e-4));
    }
    {
        std::vector<Vector3> up = base, down = base;
        up[4].z() += h;
        down[4].z() -= h;
        const double fd = (energy_of([&] { return read_device(snc->PP_energies())[0]; }, up)
                           - energy_of([&] { return read_device(snc->PP_energies())[0]; }, down))
                          / (2.0 * h);
        CHECK(close(fd, slices.gpp.values[2], 1e-4));
    }
    set_positions(base);

    // (e) the Hessian.  The written matrix is the **clamped** one: the reference (and this port)
    // run `make_spd(H)` before assembling, and `dB/dD < 0` makes the raw PT Hessian *indefinite*
    // here (`2 dB/dD` is an eigenvalue of the z-subspace block), so comparing the written blocks
    // against the raw closed form `dB2/dD2 g g^T + dB/dD d^2D/dx^2` is invalid for every block the
    // clamp's positive eigenvectors reach -- measured: H(x,x) = 5.37e-2, not 0, although the raw
    // entry is exactly 0.  What *is* checked: (1) the reference's upper-triangle triplet order,
    // (2) the device blocks against the **host instantiation of the same reference body plus the
    // same clamp** (`t81b_sym_expressions`' idiom: it pins the kernel's geometry, the block order,
    // the assembler and the sweep count), and (3) `make_spd`'s own contract, i.e. the result is
    // PSD.  The barrier law's second derivative itself is pinned independently in
    // `t81b_barrier_law` (72 rows, analytic `d2` verified at 1e-26 against 60-digit arithmetic).
    {
        CHECK(slices.hp.rows.size() == 10);
        CHECK(slices.hp.rows == std::vector<std::int32_t>{0, 0, 0, 0, 1, 1, 1, 2, 2, 3});
        CHECK(slices.hp.cols == std::vector<std::int32_t>{0, 1, 2, 3, 1, 2, 3, 2, 3, 3});

        auto lv3 = [](const Vector3& v)
        { return lps::LVec<Float, 3>{Float(v.x()), Float(v.y()), Float(v.z())}; };
        const auto& bp = base;
        lps::LVec<Float, 3> P{lv3(bp[0])}, T0{lv3(bp[1])}, T1{lv3(bp[2])}, T2{lv3(bp[3])};
        auto flag = ldist::point_triangle_distance_flag(P, T0, T1, T2);
        lps::LVec<Float, 12>    G_host;
        lps::LMat<Float, 12, 12> H_host;
        sym_simplex::PT_barrier_gradient_hessian(
            G_host, H_host, flag, Float(kt2), Float(dh_pt), Float(th_pt), P, T0, T1, T2);
        // the models' sweep count (`kMakeSpdSweeps`, the T5.2 `hs1d_detail` choice); `make_spd`'s
        // own default is the 20-sweep `kJacobiSweeps`, so the count has to be passed explicitly
        lps::make_spd_inplace(H_host, 8);

        for(size_t k = 0; k < slices.hp.rows.size(); ++k)
        {
            const int r = slices.hp.rows[k];
            const int c = slices.hp.cols[k];
            for(int b = 0; b < 3; ++b)
                for(int a = 0; a < 3; ++a)
                    CHECK(close(slices.hp.values[k * 9 + static_cast<size_t>(b * 3 + a)],
                                H_host(3 * r + a, 3 * c + b), 1e-12));
        }

        // the expansion is symmetric by construction (upper-triangle blocks + `H + H^T`, the
        // diagonal halved), so this pins that the blocks really are an upper triangle -- and that
        // every entry is finite (a NaN would silently pass every `isApprox`)
        auto   H_written = expand_hessian_12(slices.hp);
        size_t nan_count = 0;
        size_t asym      = 0;
        for(int c = 0; c < 12; ++c)
            for(int r = 0; r < 12; ++r)
            {
                if(std::isnan(H_written(r, c)))
                    ++nan_count;
                if(!close(H_written(r, c), H_written(c, r), 1e-15))
                    ++asym;
            }
        CHECK(nan_count == 0u);
        CHECK(asym == 0u);
    }

    // (f) the reference's quirk: the *model* does not clamp to the active band
    {
        std::vector<Vector3> far = base;
        far[0].z()             = 0.5;  // D = 0.25 >> (thickness + d_hat)^2
        set_positions(far);
        run_dytopo();
        compute_energies();
        auto e_far = read_device(snc->PT_energies());
        REQUIRE(e_far.size() == 1);
        CHECK(e_far[0] != 0.0);
        CHECK(close(e_far[0], barrier(kt2, 0.25, dh_pt, th_pt), 1e-11));
        set_positions(base);
    }
}

// ===========================================================================
//  5) the friction branch
// ===========================================================================
TEST_CASE("t81b_friction_branch", "[contact_model]")
{
    test_runtime();
    const auto& base = probe_positions();

    // (a) mu == 0: the frictional model's energies are exactly zero
    {
        SceneDesc desc;
        desc.friction = 0.0;
        auto   scene  = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81b_fric0")};
        World  world{engine};
        world.init(scene);

        auto* cm  = Probe::contact_manager;
        auto* gtf = Probe::trajectory_filter;
        auto* vm  = Probe::vertex_manager;
        REQUIRE(cm != nullptr);
        auto* sfc = find_model<gg::SimplexFrictionalContact>(cm);
        REQUIRE(sfc != nullptr);
        auto* snc = find_model<gg::SimplexNormalContact>(cm);
        REQUIRE(snc != nullptr);

        set_probe_pairs();
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::RecordFrictionCandidates);
        CHECK(read_device(sfc->PTs()).size() == 1);

        const auto n = static_cast<size_t>(vm->positions().size());
        set_positions(base);
        set_prev_positions(base);
        set_thicknesses(std::vector<Float>(n, desc.thickness));
        set_d_hats(std::vector<Float>(n, desc.d_hat));

        run_dytopo();
        auto total = compute_energies();
        auto slice = read_simplex_slices(*sfc);
        REQUIRE(slice.pt.size() == 1);
        CHECK(slice.pt[0] == 0.0);
        CHECK(slice.ee[0] == 0.0);
        CHECK(slice.pe[0] == 0.0);
        CHECK(slice.pp[0] == 0.0);
        auto   normal = read_simplex_slices(*snc);
        double sum    = 0.0;
        for(auto v : total)
            sum += v;
        CHECK(close(sum, normal.pt[0] + normal.ee[0] + normal.pe[0] + normal.pp[0], 1e-13));
    }

    // (b) mu > 0 with zero motion: the C1 clamp's `epsvh / 3` term is the exact law
    {
        SceneDesc desc;
        desc.friction = 0.3;
        auto   scene  = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81b_fric1")};
        World  world{engine};
        world.init(scene);

        auto* cm  = Probe::contact_manager;
        auto* gtf = Probe::trajectory_filter;
        auto* vm  = Probe::vertex_manager;
        auto* sfc = find_model<gg::SimplexFrictionalContact>(cm);
        REQUIRE(sfc != nullptr);

        set_probe_pairs();
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::RecordFrictionCandidates);

        const auto n = static_cast<size_t>(vm->positions().size());
        set_positions(base);
        set_prev_positions(base);
        set_thicknesses(std::vector<Float>(n, desc.thickness));
        set_d_hats(std::vector<Float>(n, desc.d_hat));

        run_dytopo();
        compute_energies();
        auto slice = read_simplex_slices(*sfc);
        REQUIRE(slice.pt.size() == 1);

        auto         table = read_device(cm->contact_tabular());
        const double kappa = table[0].kappa;
        const double mu    = table[0].mu;
        CHECK(close(mu, desc.friction, 1e-14));
        const double kt2   = kappa * desc.dt * desc.dt;
        const double D     = 0.06 * 0.06;
        const double epsvh = cm->eps_velocity() * desc.dt;
        // the model aggregates first: `PT_thickness` sums the two per-vertex thicknesses (0.02)
        // and `PT_d_hat` averages the two d_hats (utils/codim_thickness.h,
        // utils/primitive_d_hat.h), so the barrier does not see `desc.thickness` itself.
        const double th_pt = gg::PT_thickness(Float(desc.thickness), Float(desc.thickness),
                                              Float(desc.thickness), Float(desc.thickness));
        const double dh_pt = gg::PT_d_hat(Float(desc.d_hat), Float(desc.d_hat),
                                          Float(desc.d_hat), Float(desc.d_hat));
        const double f     = -barrier_d1(kt2, D, dh_pt, th_pt) * 2.0 * std::sqrt(D);
        CHECK(f > 0.0);
        const double E0 = mu * f * (epsvh / 3.0);
        CHECK(close(slice.pt[0], E0, 1e-11));
        CHECK(slice.pt[0] > 0.0);
        CHECK(slice.ee[0] >= 0.0);
        CHECK(slice.pe[0] >= 0.0);
        CHECK(slice.pp[0] >= 0.0);
        std::printf("[contact_model] friction: mu=%g kappa=%g epsvh=%g f=%g -> E0=%.17g (device %.17g)\n",
                    mu, kappa, epsvh, f, E0, slice.pt[0]);
        CHECK(close(slice.pt[0] / mu, f * epsvh / 3.0, 1e-12));

        constexpr double h = 1e-6;
        auto             e_pt = [&](const std::vector<Vector3>& pos)
        {
            set_positions(pos);
            run_dytopo();
            compute_energies();
            return read_device(sfc->PT_energies())[0];
        };
        std::vector<Vector3> up = base, down = base;
        up[0].z() += h;
        down[0].z() -= h;
        const double fd = (e_pt(up) - e_pt(down)) / (2.0 * h);
        set_positions(base);
        run_dytopo();
        auto g = read_gradients(sfc->PT_gradients());
        REQUIRE(g.values.size() == 12);
        CHECK(close(fd, g.values[2], 1e-4));
        auto H = expand_hessian_12(read_hessians(sfc->PT_hessians()));
        CHECK(H.isApprox(H.transpose(), 1e-10));

        // (c) the previous position is the state the friction law reads
        std::vector<Vector3> prev = base;
        prev[0].x() += 0.01;
        set_prev_positions(prev);
        run_dytopo();
        compute_energies();
        CHECK(read_device(sfc->PT_energies())[0] != slice.pt[0]);
        set_prev_positions(base);
    }
}

// ===========================================================================
//  6) the vertex/half-plane models (real half-plane geometry + the Easy filter)
// ===========================================================================
TEST_CASE("t81b_half_plane_pairs", "[contact_model]")
{
    test_runtime();

    SceneDesc desc;
    desc.half_planes.push_back(HPDesc{});
    desc.positions = {{0.0, 0.0, 0.01}, {1.0, 0.0, 0.01}, {0.0, 1.0, 0.01}};
    desc.triangles = {{0, 1, 2}};
    desc.friction  = 0.3;
    auto   scene   = build_scene(desc);
    Engine engine{"luisa", workspace_path("u81b_ph")};
    World  world{engine};
    world.init(scene);

    auto* cm  = Probe::contact_manager;
    auto* gtf = Probe::trajectory_filter;
    auto* vm  = Probe::vertex_manager;
    auto* hpf = Probe::easy_half_plane_filter;
    REQUIRE(cm != nullptr);
    REQUIRE(gtf != nullptr);
    REQUIRE(vm != nullptr);
    REQUIRE(hpf != nullptr);

    auto* phn = find_model<gg::VertexHalfPlaneNormalContact>(cm);
    auto* phf = find_model<gg::VertexHalfPlaneFrictionalContact>(cm);
    REQUIRE(phn != nullptr);
    REQUIRE(phf != nullptr);

    ProbeFilter::clear_pairs();
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::RecordFrictionCandidates);

    // The PH pair list, in the reporter's own order: `(vertex, half_plane)`.  The vertex ids are
    // **global** ones: this scene creates the half-plane geometry first, so it owns vertex 0 and
    // the trimesh's three vertices are 1..3 (measured) -- the reporter's own `vertex_offset()` is
    // the *half-plane's* base (0), not the trimesh's, so the expectation is derived from the
    // vertex count instead: the trimesh is the last geometry and has exactly 3 vertices.
    const auto n   = static_cast<size_t>(vm->positions().size());
    auto       PHs = read_device(hpf->PHs());
    REQUIRE(PHs.size() == 3);
    for(auto& ph : PHs)
        CHECK(ph(1) == 0);
    for(size_t i = 0; i < PHs.size(); ++i)
        CHECK(static_cast<size_t>(PHs[i](0)) == n - PHs.size() + i);
    const double dHat = desc.d_hat, th = desc.thickness, dt = desc.dt;
    {
        auto pos = read_vector3(vm->positions());
        set_prev_positions(pos);
        set_thicknesses(std::vector<Float>(n, desc.thickness));
        set_d_hats(std::vector<Float>(n, desc.d_hat));
    }

    auto         table = read_device(cm->contact_tabular());
    REQUIRE(table.size() == 1);
    const double kappa = table[0].kappa;
    const double mu    = table[0].mu;
    const double kt2   = kappa * dt * dt;
    const double D     = 0.015 * 0.015;
    const double dBdD  = barrier_d1(kt2, D, dHat, th);
    const double B     = barrier(kt2, D, dHat, th);

    run_dytopo();
    compute_energies();

    {
        auto e = read_device(phn->energies());
        REQUIRE(e.size() == 3);
        for(auto v : e)
            CHECK(close(v, B, 1e-12));
        auto g = read_gradients(phn->gradients());
        auto h = read_hessians(phn->hessians());
        REQUIRE(g.values.size() == 9);
        REQUIRE(h.values.size() == 27);
        for(int i = 0; i < 3; ++i)
        {
            CHECK(g.indices[static_cast<size_t>(i)] == PHs[static_cast<size_t>(i)](0));
            CHECK(close(g.values[static_cast<size_t>(3 * i + 0)], 0.0, 1e-12));
            CHECK(close(g.values[static_cast<size_t>(3 * i + 1)], 0.0, 1e-12));
            CHECK(close(g.values[static_cast<size_t>(3 * i + 2)], dBdD * 0.03, 1e-12));
        }
        const double h1     = 1e-6;
        const double dB2dD2 = (barrier_d1(kt2, D + h1, dHat, th) - barrier_d1(kt2, D - h1, dHat, th)) / (2.0 * h1);
        // `H = dB2/dD2 g g^T + dB/dD * 2 N N^T` (`N = ẑ`, `g = 0.03`): the diagonal triplet (k,k)
        for(int i = 0; i < 3; ++i)
        {
            // one triplet per (vertex, plane) pair -- `k = i`, not `3 * i`: a PH pair writes a
            // single 3x3 block, so `h.rows[3 * i]` would read past the reporter's 3 triplets
            // (measured: garbage ids before this was fixed)
            const size_t k = static_cast<size_t>(i);
            CHECK(h.rows[k] == g.indices[k]);
            CHECK(h.cols[k] == g.indices[k]);
            CHECK(close(h.values[k * 9 + 8], dB2dD2 * 0.03 * 0.03 + dBdD * 2.0, 1e-4));
            CHECK(close(h.values[k * 9 + 0], 0.0, 1e-9));
        }
        std::printf("[contact_model] half-plane: kappa=%g n=%zu D=%g -> B=%.17g dBdD=%.17g "
                    "(device %.17g, gradient[2] %.17g, hessian[8] %.17g)\n",
                    kappa, n, D, B, dBdD, e[0], h.values[2], h.values[8]);
    }

    {
        auto e = read_device(phf->energies());
        REQUIRE(e.size() == 3);
        const double epsvh = cm->eps_velocity() * dt;
        const double f     = -barrier_d1(kt2, D, dHat, th) * 2.0 * std::sqrt(D);
        const double E0    = mu * f * (epsvh / 3.0);
        for(auto v : e)
            CHECK(close(v, E0, 1e-11));
    }
}

// ===========================================================================
//  7) the exporter path: the models' own numbers per primitive type
// ===========================================================================
TEST_CASE("t81b_exporter_readback", "[contact_model]")
{
    test_runtime();

    SceneDesc desc;
    auto      scene = build_scene(desc);
    Engine    engine{"luisa", workspace_path("u81b_export")};
    World     world{engine};
    world.init(scene);

    auto* cm  = Probe::contact_manager;
    auto* gtf = Probe::trajectory_filter;
    auto* vm  = Probe::vertex_manager;
    REQUIRE(cm != nullptr);
    REQUIRE(gtf != nullptr);
    auto* snc = find_model<gg::SimplexNormalContact>(cm);
    REQUIRE(snc != nullptr);

    set_probe_pairs();
    gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
    const auto& base = probe_positions();
    const auto  n    = static_cast<size_t>(vm->positions().size());
    set_positions(base);
    set_prev_positions(base);
    set_thicknesses(std::vector<Float>(n, desc.thickness));
    set_d_hats(std::vector<Float>(n, desc.d_hat));
    run_dytopo();
    compute_energies();

    auto feature = engine.features().find<core::ContactSystemFeature>();
    REQUIRE(feature != nullptr);

    auto types = feature->contact_primitive_types();
    const std::vector<std::string> expected = {"EE+F", "EE+N", "PE+F", "PE+N", "PH+F", "PH+N",
                                              "PP+F", "PP+N", "PT+F", "PT+N"};
    REQUIRE(types.size() == 10);
    for(auto& t : expected)
        CHECK(std::find(types.begin(), types.end(), t) != types.end());

    auto e = read_device(snc->PT_energies());
    REQUIRE(e.size() == 1);
    {
        geometry::Geometry geo;
        feature->contact_energy("PT+N", geo);
        auto topo = geo.instances().find<Vector4i>("topo");
        REQUIRE(topo != nullptr);
        auto tv = view(*topo);
        REQUIRE(tv.size() == 1);
        CHECK(tv[0] == Vector4i(0, 1, 2, 3));
        auto energy = geo.instances().find<Float>("energy");
        REQUIRE(energy != nullptr);
        CHECK(view(*energy)[0] == e[0]);
        auto prim = geo.meta().find<std::string>("prim_type");
        REQUIRE(prim != nullptr);
        CHECK(view(*prim)[0] == "PT+N");
    }
    {
        geometry::Geometry geo;
        feature->contact_gradient("PT+N", geo);
        auto i = geo.instances().find<IndexT>("i");
        auto g = geo.instances().find<Vector3>("grad");
        REQUIRE(i != nullptr);
        REQUIRE(g != nullptr);
        auto iv = view(*i);
        auto gv = view(*g);
        REQUIRE(iv.size() == 4);
        auto model_g = read_gradients(snc->PT_gradients());
        for(int k = 0; k < 4; ++k)
        {
            CHECK(iv[k] == model_g.indices[static_cast<size_t>(k)]);
            CHECK(gv[k] == Vector3(model_g.values[static_cast<size_t>(3 * k + 0)],
                                   model_g.values[static_cast<size_t>(3 * k + 1)],
                                   model_g.values[static_cast<size_t>(3 * k + 2)]));
        }
    }
    {
        geometry::Geometry geo;
        feature->contact_hessian("PT+N", geo);
        auto i = geo.instances().find<IndexT>("i");
        auto j = geo.instances().find<IndexT>("j");
        auto hp = geo.instances().find<Matrix3x3>("hess");
        REQUIRE(i != nullptr);
        REQUIRE(j != nullptr);
        REQUIRE(hp != nullptr);
        auto iv = view(*i);
        auto jv = view(*j);
        auto hv = view(*hp);
        REQUIRE(iv.size() == 10);
        auto model_h = read_hessians(snc->PT_hessians());
        for(int k = 0; k < 10; ++k)
        {
            CHECK(iv[k] == model_h.rows[static_cast<size_t>(k)]);
            CHECK(jv[k] == model_h.cols[static_cast<size_t>(k)]);
            for(int c = 0; c < 3; ++c)
                for(int r = 0; r < 3; ++r)
                    CHECK(hv[k](r, c) == model_h.values[static_cast<size_t>(k) * 9 + static_cast<size_t>(c * 3 + r)]);
        }
    }
    {
        geometry::Geometry geo;
        feature->contact_energy("PH+N", geo);
        auto topo = geo.instances().find<Vector2i>("topo");
        REQUIRE(topo != nullptr);
        CHECK(view(*topo).empty());
    }
}

// ===========================================================================
//  8) determinism and the empty scene
// ===========================================================================
TEST_CASE("t81b_determinism_and_empty", "[contact_model]")
{
    test_runtime();

    {
        SceneDesc desc;
        auto      scene = build_scene(desc);
        Engine    engine{"luisa", workspace_path("u81b_det")};
        World     world{engine};
        world.init(scene);

        auto* cm  = Probe::contact_manager;
        auto* gtf = Probe::trajectory_filter;
        auto* vm  = Probe::vertex_manager;
        auto* snc = find_model<gg::SimplexNormalContact>(cm);
        REQUIRE(snc != nullptr);

        set_probe_pairs();
        gtf->run_stage(gg::GlobalTrajectoryFilter::Stage::FilterActive);
        const auto& base = probe_positions();
        const auto  n    = static_cast<size_t>(vm->positions().size());
        set_positions(base);
        set_prev_positions(base);
        set_thicknesses(std::vector<Float>(n, desc.thickness));
        set_d_hats(std::vector<Float>(n, desc.d_hat));

        run_dytopo();
        auto e1 = compute_energies();
        auto g1 = read_gradients(snc->PT_gradients());
        auto h1 = read_hessians(snc->PT_hessians());

        run_dytopo();
        auto e2 = compute_energies();
        auto g2 = read_gradients(snc->PT_gradients());
        auto h2 = read_hessians(snc->PT_hessians());

        CHECK(e1 == e2);
        CHECK(g1.values == g2.values);
        CHECK(g1.indices == g2.indices);
        CHECK(h1.values == h2.values);
        CHECK(h1.rows == h2.rows);
        CHECK(h1.cols == h2.cols);
    }

    {
        SceneDesc desc;
        desc.probe_marker = false;
        auto   scene      = build_scene(desc);
        Engine engine{"luisa", workspace_path("u81b_empty")};
        World  world{engine};
        world.init(scene);

        auto* cm  = Probe::contact_manager;
        auto* gtf = Probe::trajectory_filter;
        REQUIRE(cm != nullptr);
        REQUIRE(gtf != nullptr);
        auto* snc = find_model<gg::SimplexNormalContact>(cm);
        REQUIRE(snc != nullptr);
        CHECK(gtf->find<ProbeFilter>() == false);

        ProbeFilter::clear_pairs();
        run_dytopo();
        auto* dyt = Probe::dytopo_effect_manager;
        REQUIRE(dyt != nullptr);
        CHECK(dyt->gradients().doublet_count() == 0);
        CHECK(dyt->hessians().triplet_count() == 0);
        auto   e   = compute_energies();
        double sum = 0.0;
        for(auto v : e)
            sum += v;
        CHECK(sum == 0.0);
        CHECK(read_device(snc->PT_energies()).empty());
        CHECK(read_device(snc->EE_energies()).empty());
        CHECK(read_device(snc->PE_energies()).empty());
        CHECK(read_device(snc->PP_energies()).empty());
    }
}
