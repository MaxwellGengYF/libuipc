/********************************************************************
 * @file luisa_fem_dynamics_test.cpp
 * @brief T5.3 — FEM dynamics: the BDF integrators and their states, the kinetic
 * system, the animator, the external-force manager/reporter and the
 * pulled-forward constraint surface (LuisaCompute backend).
 *
 * Every oracle in this file is derived independently *here*: the reference backend
 * is CUDA-only and cannot be host-included, so nothing is copied from it.  The
 * oracles are
 *  (a) hand-derived closed forms for the BDF2 helper functions
 *      (`sym::fem_bdf2::{compute_x_tilde, compute_v}`),
 *  (b) hand-derived closed forms for the kinetic energy/gradient/Hessian laws,
 *  (c) an independent host model of the integrators' per-vertex formulas, driven
 *      side by side with the *ported* per-vertex routines over the same device
 *      buffers (cookbook §7.4: the shipped kernel and the test run the same code),
 *  (d) a real scene through the public `Engine`/`World` API, whose state is seeded
 *      with known positions/velocities through the ported
 *      `FiniteElementStateAccessorFeature` and then compared against the same
 *      hand-written formulas -- this is the only path that exercises the *shipped*
 *      kernels (and the BDF2 history bootstrap) end to end,
 *  (e) the reference's own contract for the external-force manager
 *      (`clear` -> reporters scatter-add -> `accs = F/m`, exactly 0 for a
 *      massless vertex), read back from the method's buffers, and
 *  (f) the device-vs-host equivalence of the ported routines (`double` vs
 *      `lcrt::Var<double>` in a `compile1` kernel), expected *bitwise* because the
 *      arithmetic is identical and no transcendental is involved (cookbook §7.4 /
 *      T2.1's convention).
 *
 * ---------------------------------- T5.3 ----------------------------------
 * Coverage map (case -> oracle):
 *  1) the three BDF2 helper templates ................ fem_dynamics_bdf2_host_oracle
 *  2) kinetic energy/gradient/Hessian (BDF1+BDF2) .... fem_dynamics_kinetic_host_oracle
 *  3) BDF1 predict/update through device buffers ..... fem_dynamics_bdf1_integrator
 *  4) BDF2 predict/update (fixed/static/dynamic) ..... fem_dynamics_bdf2_integrator
 *  5) a real scene, bdf1 + bdf2 (+ the BDF2 bootstrap,
 *     + FEMBDF2State dump/recover) ................... fem_dynamics_scene
 *  6) the external-force manager contract ............ fem_dynamics_external_force_manager
 *  7) device == host for every ported routine ........ fem_dynamics_device_vs_host
 *  8) the pulled-forward SoftPositionConstraint ...... fem_dynamics_soft_position_constraint
 *
 * Shared-binary rule (cookbook §7.14 rule 6): every system registered here is
 * built for every scene of the whole test binary, so the test-local external-force
 * reporter gates itself on a marker attribute of *its own* geometry and the probe
 * is invalidated in any scene without a FiniteElement constitution.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"
#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/arap.h>
#include <uipc/constitution/soft_position_constraint.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <type_define.h>
#include <frame_stream.h>
#include <finite_element/fem_time_integrator.h>
#include <finite_element/bdf/bdf2_function.h>
#include <finite_element/bdf/fem_bdf1_time_integrator.h>
#include <finite_element/bdf/fem_bdf2_time_integrator.h>
#include <finite_element/bdf/finite_element_bdf1_kinetic.h>
#include <finite_element/bdf/finite_element_bdf2_kinetic.h>
#include <finite_element/bdf/fem_bdf2_state.h>
#include <finite_element/finite_element_kinetic.h>
#include <finite_element/finite_element_animator.h>
#include <finite_element/finite_element_constraint.h>
#include <finite_element/finite_element_external_force_manager.h>
#include <finite_element/finite_element_external_force_reporter.h>
#include <finite_element/fem_line_search_reporter.h>
#include <finite_element/fem_line_search_subreporter.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/constraints/soft_position_constraint.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;
namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;

// The ported dynamics surface lives in `uipc::backend::luisa` (like the reference
// backend's does in `uipc::backend::cuda`), so a test body sees the same
// unqualified spellings a ported system does: `bdf1_detail::predict_dof`,
// `bdf2_detail::predict_dof`, `bdf1_kinetic_detail::energy`, `sym::fem_bdf2`,
// `load_vec3`/`store_vec3`, ...
using namespace uipc::backend::luisa;

namespace t53
{
/// The BDF2 kinetic system carries the `beta`/`inv_beta` convention.
using BDF2Kinetic = gg::FiniteElementBDF2Kinetic;

// ---------------------------------------------------------------------------
//  small helpers (the T5.1/T5.2 test files' idioms)
// ---------------------------------------------------------------------------
/// Scale-aware relative error (identical to the T5.1 helper).
[[nodiscard]] double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

[[nodiscard]] std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// Read an LC buffer back to the host (frame-stream first: a command may still be
/// recorded in an open frame).
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

[[nodiscard]] std::vector<Vector3> read_vec3(const lud::LCBufferView<gg::LVector3>& view)
{
    auto                 raw = read_device(view);
    std::vector<Vector3> host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

[[nodiscard]] lps::LVec<double, 3> lv(const Vector3& v) noexcept
{
    return lps::LVec<double, 3>{v(0), v(1), v(2)};
}

/// `lv3<S>(a, b, c)` for a generic scalar `S` (`double` on the host, a DSL
/// `Var<double>` inside a kernel) -- the device-vs-host round needs the same
/// literal on both sides.
template <typename S>
[[nodiscard]] lps::LVec<S, 3> lv3(double a, double b, double c) noexcept
{
    return lps::make_vec<S, 3>(
        [&](int i) noexcept
        {
            const double v = (i == 0) ? a : (i == 1 ? b : c);
            return lps::make_scalar<S>(v);
        });
}

// ---------------------------------------------------------------------------
//  the probe: grabs the FEM systems' public interfaces once they are built (the
//  T5.1 T51FEMProbe idiom).  A scene without a FiniteElement constitution
//  invalidates it (require<> throws SimSystemException), so other test files are
//  unaffected; the test resets the static pointers before each scene.
// ---------------------------------------------------------------------------
class T53Probe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;
    inline static gg::FiniteElementMethod*   fem          = nullptr;
    inline static gg::FEMBDF2State*          bdf2_state   = nullptr;
    inline static gg::FiniteElementAnimator*  animator = nullptr;
    inline static gg::SoftPositionConstraint* spc      = nullptr;

    static void reset()
    {
        fem        = nullptr;
        bdf2_state = nullptr;
        animator   = nullptr;
        spc        = nullptr;
    }

  private:
    void do_build() override
    {
        fem = &require<gg::FiniteElementMethod>();
        // optional: only a bdf2 scene builds the state; the animator needs both a
        // FiniteElement and a Constraint constitution type
        bdf2_state = find<gg::FEMBDF2State>();
        animator   = find<gg::FiniteElementAnimator>();
        spc        = find<gg::SoftPositionConstraint>();
    }
};
}  // namespace t53
REGISTER_SIM_SYSTEM(t53::T53Probe);

namespace t53
{
// ---------------------------------------------------------------------------
//  test-local external-force reporter
//
//  Reserved uid 0x54353346454D58 == "T53FEMX" (7 bytes, > 2^32).  The uids in use
//  elsewhere are 0x5445535456455254 ("TESTVERT"), 0x543432414E494D ("T42ANIM"),
//  0x543432464F5243 ("T42FORC"), 0x54343344594E54 ("T43DYNT") and
//  FEMExternalForceManager's own 671 -- none collides (cookbook §7.18).
//
//  The reference's per-frame `FiniteElementExternalVertexForce::do_step` does an
//  `eigen::atomic_add(forces(vid), F)`.  LC cannot express a *component-wise*
//  atomic add on a struct element (its `AtomicRef<LVector3>` exposes only
//  whole-element ops); this mock therefore uses a read-modify-write, which is
//  observationally identical while it is the only writer.  The real T5.4 reporter
//  hits the same limitation -- the lever there is a scalar/per-component layout.
// ---------------------------------------------------------------------------
class T53ExternalForceReporter final : public gg::FiniteElementExternalForceReporter
{
  public:
    using FiniteElementExternalForceReporter::FiniteElementExternalForceReporter;

    static constexpr U64 kUid = 0x54353346454D58ull;  // "T53FEMX"

    static inline int                  init_count  = 0;
    static inline int                  step_count  = 0;
    static inline bool                 write_enabled = true;
    static inline std::vector<IndexT>  collected_indices;
    static inline std::vector<Vector3> collected_forces;

    static void reset()
    {
        init_count  = 0;
        step_count  = 0;
        write_enabled = true;
        collected_indices.clear();
        collected_forces.clear();
    }

  protected:
    void do_build(BuildInfo& info) override { (void)info; }
    U64  get_uid() const noexcept override { return kUid; }

    void do_init() override
    {
        ++init_count;
        m_indices_host.clear();
        m_forces_host.clear();

        // gate on the test's own marker attributes (cookbook §7.14 rule 6): the
        // reporter is built for *every* FEM scene of the shared test binary
        auto geo_slots = world().scene().geometries();
        for(auto& geo_slot : geo_slots)
        {
            auto sc = geo_slot->geometry().as<SimplicialComplex>();
            if(!sc)
                continue;
            auto force = sc->vertices().find<Vector3>("t53_force");
            if(!force)
                continue;
            auto enabled     = sc->vertices().find<IndexT>("t53_force_enabled");
            auto offset_attr = sc->meta().find<IndexT>(builtin::backend_fem_vertex_offset);
            if(!enabled || !offset_attr)
                continue;

            const auto offset = offset_attr->view().front();
            auto       fv     = view(*force);
            auto       ev     = view(*enabled);
            for(size_t v = 0; v < ev.size(); ++v)
            {
                if(ev[v])
                {
                    m_indices_host.push_back(offset + static_cast<IndexT>(v));
                    m_forces_host.push_back(fv[v]);
                }
            }
        }

        m_indices.resize(m_indices_host.size());
        m_forces.resize(m_forces_host.size());
        if(!m_indices_host.empty())
        {
            m_indices.view().copy_from(span<const IndexT>{m_indices_host}, &gg::frame_stream());
            std::vector<gg::LVector3> lf;
            lf.reserve(m_forces_host.size());
            for(auto& f : m_forces_host)
                lf.push_back(gg::to_luisa(f));
            m_forces.view().copy_from(span<const gg::LVector3>{lf}, &gg::frame_stream());
        }

        collected_indices = m_indices_host;
        collected_forces  = m_forces_host;
    }

    void do_step(ExternalForceInfo& info) override
    {
        ++step_count;
        auto n = static_cast<uint32_t>(m_indices.size());
        if(n == 0 || !write_enabled)
            return;  // nothing marked (other scenes), or the test asked to skip
        auto& k = scatter_force_kernel();
        gg::frame_stream().record(
            k(info.external_forces(), m_indices.cview(), m_forces.cview(), n).dispatch(n));
    }

  private:
    std::vector<IndexT>          m_indices_host;
    std::vector<Vector3>         m_forces_host;
    lud::LDeviceBuffer<IndexT>       m_indices;
    lud::LDeviceBuffer<gg::LVector3> m_forces;

    static auto& scatter_force_kernel()
    {
        static auto& k =
            lla::detail::cached_kernel<lcrt::Buffer<gg::LVector3>,  // forces (rw)
                                       lcrt::Buffer<int32_t>,       // indices
                                       lcrt::Buffer<gg::LVector3>,  // values
                                       uint32_t>(                   // n
                [](lcrt::BufferVar<gg::LVector3> forces,
                   lcrt::BufferVar<int32_t>       indices,
                   lcrt::BufferVar<gg::LVector3>  values,
                   lcrt::Var<uint32_t>            n) noexcept
                {
                    auto i = lcrt::dispatch_x();
                    $if(i < n)
                    {
                        auto idx  = indices->read(i);
                        auto prev = load_vec3(forces->read(idx));
                        auto add  = load_vec3(values->read(i));
                        lcrt::Var<gg::LVector3> out;
                        store_vec3(out, prev + add);
                        forces->write(idx, out);
                    };
                });
        return k;
    }
};
}  // namespace t53
REGISTER_SIM_SYSTEM(t53::T53ExternalForceReporter);

namespace t53
{
// ---------------------------------------------------------------------------
//  The two integrators' kernels, transcribed from the shipped ones
//  (src/backends/luisa/finite_element/bdf/fem_bdf{1,2}_time_integrator.cpp).
//
//  The shipped kernels live in an anonymous namespace, so a unit test cannot
//  dispatch them directly; the *routines* they call (`bdf1_detail`/`bdf2_detail`)
//  and the launch-argument list are the same, which is what this transcribes.
//  The shipped kernels themselves are exercised end to end by the
//  `fem_dynamics_scene` case, which compares the method's device buffers after
//  `engine.advance()`.
// ---------------------------------------------------------------------------
auto& bdf1_predict_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<int32_t>      is_fixed,
           lcrt::BufferVar<int32_t>      is_dynamic,
           lcrt::BufferVar<gg::LVector3> x_prevs,
           lcrt::BufferVar<gg::LVector3> xs,
           lcrt::BufferVar<gg::LVector3> vs,
           lcrt::BufferVar<gg::LVector3> x_tildes,
           lcrt::BufferVar<gg::LVector3> gravities,
           lcrt::BufferVar<gg::LVector3> external_force_accs,
           lcrt::Var<double>             dt,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                x_prevs->write(i, xs->read(i));
                auto x_tilde = bdf1_detail::predict_dof(load_vec3(xs->read(i)),
                                                        load_vec3(vs->read(i)),
                                                        load_vec3(gravities->read(i)),
                                                        load_vec3(external_force_accs->read(i)),
                                                        dt,
                                                        is_fixed->read(i),
                                                        is_dynamic->read(i));
                lcrt::Var<gg::LVector3> out;
                store_vec3(out, x_tilde);
                x_tildes->write(i, out);
            };
        });
    return k;
}

auto& bdf1_update_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<gg::LVector3> xs,
           lcrt::BufferVar<gg::LVector3> vs,
           lcrt::BufferVar<gg::LVector3> x_prevs,
           lcrt::Var<double>             dt,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto v = bdf1_detail::update_state(
                    load_vec3(xs->read(i)), load_vec3(x_prevs->read(i)), dt);
                lcrt::Var<gg::LVector3> out;
                store_vec3(out, v);
                vs->write(i, out);
            };
        });
    return k;
}

auto& bdf2_predict_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<int32_t>      is_fixed,
           lcrt::BufferVar<int32_t>      is_dynamic,
           lcrt::BufferVar<gg::LVector3> xs,
           lcrt::BufferVar<gg::LVector3> x_ns,
           lcrt::BufferVar<gg::LVector3> x_n_1s,
           lcrt::BufferVar<gg::LVector3> v_ns,
           lcrt::BufferVar<gg::LVector3> v_n_1s,
           lcrt::BufferVar<gg::LVector3> x_tildes,
           lcrt::BufferVar<gg::LVector3> gravity,
           lcrt::BufferVar<gg::LVector3> external_force_accs,
           lcrt::Var<double>             dt,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                x_ns->write(i, xs->read(i));
                auto x_tilde = bdf2_detail::predict_dof(load_vec3(xs->read(i)),
                                                        load_vec3(x_n_1s->read(i)),
                                                        load_vec3(v_ns->read(i)),
                                                        load_vec3(v_n_1s->read(i)),
                                                        load_vec3(gravity->read(i)),
                                                        load_vec3(external_force_accs->read(i)),
                                                        dt,
                                                        is_fixed->read(i),
                                                        is_dynamic->read(i));
                lcrt::Var<gg::LVector3> out;
                store_vec3(out, x_tilde);
                x_tildes->write(i, out);
            };
        });
    return k;
}

auto& bdf2_update_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                lcrt::Buffer<gg::LVector3>,
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<gg::LVector3> xs,
           lcrt::BufferVar<gg::LVector3> x_ns,
           lcrt::BufferVar<gg::LVector3> x_n_1s,
           lcrt::BufferVar<gg::LVector3> v_ns,
           lcrt::BufferVar<gg::LVector3> v_n_1s,
           lcrt::Var<double>             dt,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto x   = load_vec3(xs->read(i));
                auto x_n = load_vec3(x_ns->read(i));
                lps::LVec<lcrt::Var<double>, 3> v;
                sym::fem_bdf2::compute_v(v, x, x_n, load_vec3(x_n_1s->read(i)), dt);

                v_n_1s->write(i, v_ns->read(i));
                lcrt::Var<gg::LVector3> v_out;
                store_vec3(v_out, v);
                v_ns->write(i, v_out);
                x_n_1s->write(i, x_ns->read(i));
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  Device-vs-host round: one deterministic evaluation of every ported routine,
//  serialised through `put` so the host reference and the device kernel walk an
//  identical, fixed order (cookbook §7.8 / §7.16 rule 12b).
//
//  Entry map (61 doubles):
//    0..2    sym::fem_bdf2::compute_x_tilde (no gravity term)
//    3..5    sym::fem_bdf2::compute_x_tilde (with gravity term)
//    6..8    sym::fem_bdf2::compute_v
//    9..17   bdf1_detail::predict_dof  x {dynamic, static, fixed}
//   18..26   bdf2_detail::predict_dof  x {dynamic, static, fixed}
//   27       bdf1 kinetic energy, free vertex
//   28       bdf1 kinetic energy, fixed vertex
//   29..31   bdf1 kinetic gradient, free vertex
//   32..34   bdf1 kinetic gradient, fixed vertex (all zero)
//   35..43   bdf1 kinetic Hessian (row major)
//   44       bdf2 kinetic energy, free vertex
//   45       bdf2 kinetic energy, fixed vertex
//   46..48   bdf2 kinetic gradient, free
//   49..51   bdf2 kinetic gradient, fixed
//   52..60   bdf2 kinetic Hessian
// ---------------------------------------------------------------------------
constexpr int kRoundCount = 3 + 3 + 3 + 9 + 9 + 1 + 1 + 3 + 3 + 9 + 1 + 1 + 3 + 3 + 9;

template <typename S, typename I, typename Put>
void fem_dynamics_round(Put&& put)
{
    const auto h     = lps::make_scalar<S>(0.0125);
    const auto dt    = lps::make_scalar<S>(0.0125);
    const auto x     = lv3<S>(1.0, 0.5, -0.25);
    const auto x_n   = lv3<S>(1.25, -0.5, 2.0);
    const auto x_n_1 = lv3<S>(-0.75, 0.25, 1.5);
    const auto v_n   = lv3<S>(0.5, -1.25, 0.75);
    const auto v_n_1 = lv3<S>(-0.25, 0.5, -0.5);
    const auto g     = lv3<S>(0.0, -9.8, 0.3);
    const auto f_acc = lv3<S>(0.1, 0.2, -0.3);
    const auto m     = lps::make_scalar<S>(0.375);

    // 1) sym::fem_bdf2::compute_x_tilde (no gravity term)
    {
        lps::LVec<S, 3> r;
        sym::fem_bdf2::compute_x_tilde(r, x_n, x_n_1, v_n, v_n_1, h);
        for(int i = 0; i < 3; ++i)
            put(r(i));
    }
    // 2) sym::fem_bdf2::compute_x_tilde (with gravity term)
    {
        lps::LVec<S, 3> r;
        sym::fem_bdf2::compute_x_tilde(r, x_n, x_n_1, v_n, v_n_1, g, h);
        for(int i = 0; i < 3; ++i)
            put(r(i));
    }
    // 3) sym::fem_bdf2::compute_v
    {
        lps::LVec<S, 3> r;
        sym::fem_bdf2::compute_v(r, x, x_n, x_n_1, h);
        for(int i = 0; i < 3; ++i)
            put(r(i));
    }
    // 4) bdf1 / bdf2 per-vertex predict for the three flag combinations
    const I not_fixed = I{0};
    const I is_fixed  = I{1};
    const I is_dyn    = I{1};
    const I is_static = I{0};
    {
        auto r = bdf1_detail::predict_dof(x, v_n, g, f_acc, dt, not_fixed, is_dyn);
        for(int i = 0; i < 3; ++i)
            put(r(i));
        r = bdf1_detail::predict_dof(x, v_n, g, f_acc, dt, not_fixed, is_static);
        for(int i = 0; i < 3; ++i)
            put(r(i));
        r = bdf1_detail::predict_dof(x, v_n, g, f_acc, dt, is_fixed, is_dyn);
        for(int i = 0; i < 3; ++i)
            put(r(i));
    }
    {
        auto r = bdf2_detail::predict_dof(x_n, x_n_1, v_n, v_n_1, g, f_acc, dt, not_fixed, is_dyn);
        for(int i = 0; i < 3; ++i)
            put(r(i));
        r = bdf2_detail::predict_dof(x_n, x_n_1, v_n, v_n_1, g, f_acc, dt, not_fixed, is_static);
        for(int i = 0; i < 3; ++i)
            put(r(i));
        r = bdf2_detail::predict_dof(x_n, x_n_1, v_n, v_n_1, g, f_acc, dt, is_fixed, is_dyn);
        for(int i = 0; i < 3; ++i)
            put(r(i));
    }
    // 5) BDF1 kinetic energy / gradient / Hessian
    {
        put(bdf1_kinetic_detail::energy(x, x_n, m, not_fixed));
        put(bdf1_kinetic_detail::energy(x, x_n, m, is_fixed));
        auto G = bdf1_kinetic_detail::gradient(x, x_n, m, not_fixed);
        for(int i = 0; i < 3; ++i)
            put(G(i));
        auto G0 = bdf1_kinetic_detail::gradient(x, x_n, m, is_fixed);
        for(int i = 0; i < 3; ++i)
            put(G0(i));
        auto H = bdf1_kinetic_detail::hessian(m);
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                put(H(r, c));
    }
    // 6) BDF2 kinetic energy / gradient / Hessian
    {
        const auto inv_beta = lps::make_scalar<S>(BDF2Kinetic::inv_beta);
        put(bdf2_kinetic_detail::energy(x, x_n, m, inv_beta, not_fixed));
        put(bdf2_kinetic_detail::energy(x, x_n, m, inv_beta, is_fixed));
        auto G = bdf2_kinetic_detail::gradient(x, x_n, m, inv_beta, not_fixed);
        for(int i = 0; i < 3; ++i)
            put(G(i));
        auto G0 = bdf2_kinetic_detail::gradient(x, x_n, m, inv_beta, is_fixed);
        for(int i = 0; i < 3; ++i)
            put(G0(i));
        auto H = bdf2_kinetic_detail::hessian(m, inv_beta);
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                put(H(r, c));
    }
}

auto& fem_dynamics_round_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>>(
        [](lcrt::BufferVar<double> out) noexcept
        {
            uint32_t index = 0;
            fem_dynamics_round<lcrt::Var<double>, lcrt::Var<int32_t>>(
                [&](auto&& value) noexcept { out->write(index++, value); });
        });
    return k;
}

// ---------------------------------------------------------------------------
//  scene description (hand-written, no assets)
// ---------------------------------------------------------------------------
struct SceneDesc
{
    std::vector<Vector3>  positions;
    std::vector<Vector4i> tets;
    std::vector<IndexT>   is_fixed;
    std::vector<IndexT>   is_dynamic;
    std::vector<Vector3>  gravities;  // per-vertex gravity override
    std::vector<Float>    densities;  // per-vertex mass density override
    std::vector<Vector3>  forces;     // per-vertex external force for the mock
    std::vector<IndexT>   force_enabled;
    // the vertex attribute the SoftPositionConstraint reads; only meaningful when
    // `with_constraint` is set
    std::vector<IndexT>   is_constrained;
    bool                  with_constraint = false;
    std::string           integrator_type = "bdf1";
    Float                 dt              = 0.01;
    Float                 density         = 1000.0;
};

/// Two unit tets (each of volume 1/6), so the vertex count is 8 and the method's
/// geometry order is unambiguous.
SceneDesc basic_desc()
{
    SceneDesc d;
    d.positions = {{0.0, 0.0, 0.0},
                   {1.0, 0.0, 0.0},
                   {0.0, 1.0, 0.0},
                   {0.0, 0.0, 1.0},  // tet 0
                   {3.0, 0.0, 0.0},
                   {4.0, 0.0, 0.0},
                   {3.0, 1.0, 0.0},
                   {3.0, 0.0, 1.0}};  // tet 1
    d.tets          = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    const size_t n  = d.positions.size();
    d.is_fixed.resize(n, 0);
    d.is_dynamic.resize(n, 1);
    d.gravities.resize(n, Vector3{0.0, -9.8, 0.0});
    d.densities.resize(n, 1000.0);
    d.forces.resize(n, Vector3::Zero());
    d.force_enabled.resize(n, 0);
    d.is_constrained.resize(n, 0);
    return d;
}

Scene build_scene(const SceneDesc& desc)
{
    auto config = Scene::default_config();
    // the dytopo effect manager is only created when contact is enabled, and
    // contact is not ported; the FEM method does not need it (T5.1's scene)
    config["contact"]["enable"]      = false;
    config["sanity_check"]["enable"] = false;
    config["integrator"]["type"]     = desc.integrator_type;
    config["dt"]                     = desc.dt;
    config["gravity"]                = Vector3{0.0, -9.8, 0.0};

    Scene scene{config};
    auto  object = scene.objects().create("t53");

    ARAP arap;
    auto mesh = tetmesh(span<const Vector3>{desc.positions}, span<const Vector4i>{desc.tets});
    label_surface(mesh);
    arap.apply_to(mesh, 1.0e6, desc.density);

    {
        // `apply_to` may already have created some of these (the frontend's FEM
        // constitutions create `is_fixed`), so find-or-create
        auto is_fixed = mesh.vertices().find<IndexT>(builtin::is_fixed);
        if(!is_fixed)
            is_fixed = mesh.vertices().create<IndexT>(builtin::is_fixed, 0);
        std::ranges::copy(desc.is_fixed, view(*is_fixed).begin());
    }
    {
        auto is_dynamic = mesh.vertices().find<IndexT>(builtin::is_dynamic);
        if(!is_dynamic)
            is_dynamic = mesh.vertices().create<IndexT>(builtin::is_dynamic, 1);
        std::ranges::copy(desc.is_dynamic, view(*is_dynamic).begin());
    }
    {
        auto gravity = mesh.vertices().find<Vector3>(builtin::gravity);
        if(!gravity)
            gravity = mesh.vertices().create<Vector3>(builtin::gravity, Vector3::Zero());
        std::ranges::copy(desc.gravities, view(*gravity).begin());
    }
    {
        // per-vertex mass density: a 0 makes the vertex massless (the method
        // computes m = density * volume), which the external-force manager must
        // map to a zero acceleration
        auto density = mesh.vertices().find<Float>(builtin::mass_density);
        if(!density)
            density = mesh.vertices().create<Float>(builtin::mass_density, desc.density);
        std::ranges::copy(desc.densities, view(*density).begin());
    }
    {
        // the mock external-force reporter's inputs, gated on its own attributes
        auto force = mesh.vertices().find<Vector3>("t53_force");
        if(!force)
            force = mesh.vertices().create<Vector3>("t53_force", Vector3::Zero());
        std::ranges::copy(desc.forces, view(*force).begin());
        auto enabled = mesh.vertices().find<IndexT>("t53_force_enabled");
        if(!enabled)
            enabled = mesh.vertices().create<IndexT>("t53_force_enabled", 0);
        std::ranges::copy(desc.force_enabled, view(*enabled).begin());
    }
    if(desc.with_constraint)
    {
        // `is_constrained` must exist *before* apply_to (which deliberately does
        // not fill it when it is already there)
        auto is_constrained = mesh.vertices().find<IndexT>(builtin::is_constrained);
        if(!is_constrained)
            is_constrained = mesh.vertices().create<IndexT>(builtin::is_constrained, 0);
        std::ranges::copy(desc.is_constrained, view(*is_constrained).begin());

        uipc::constitution::SoftPositionConstraint spc;
        spc.apply_to(mesh, 100.0);
    }

    object->geometries().create(mesh);
    return scene;
}

/// "step the buffer forward one frame", with every stream synchronized.
void step_frame(World& world)
{
    world.advance();
    world.sync();
    gg::frame_stream().synchronize();
}

void copy_vec3_to_device(lud::LDeviceBuffer<gg::LVector3>& dst, const std::vector<Vector3>& src)
{
    std::vector<gg::LVector3> host;
    host.reserve(src.size());
    for(auto& v : src)
        host.push_back(gg::to_luisa(v));
    dst.resize(host.size());
    if(!host.empty())
        dst.view().copy_from(span<const gg::LVector3>{host});
}

void copy_index_to_device(lud::LDeviceBuffer<IndexT>& dst, const std::vector<IndexT>& src)
{
    dst.resize(src.size());
    if(!src.empty())
        dst.view().copy_from(span<const IndexT>{src});
}

/// `FEMBDF2State::x_n_1s()/v_n_1s()` have a public *const* overload and a private
/// non-const one (the reference's layout: the mutable views are the integrator's),
/// so a test has to read them through a `const` pointer.
[[nodiscard]] lud::LCBufferView<gg::LVector3> state_x_n_1s(const gg::FEMBDF2State* state)
{
    return state->x_n_1s();
}
[[nodiscard]] lud::LCBufferView<gg::LVector3> state_v_n_1s(const gg::FEMBDF2State* state)
{
    return state->v_n_1s();
}

/// Overwrite the method's `xs`/`vs` through the *public* state-accessor feature,
/// so a scene can be seeded with known positions and velocities.
void seed_state(World&                            world,
                const std::vector<Vector3>&       positions,
                const std::vector<Vector3>&       velocities)
{
    auto feature = world.features().find<FiniteElementStateAccessorFeature>();
    REQUIRE(feature != nullptr);
    auto state_geo = feature->create_geometry();
    auto pos = state_geo.vertices().create<Vector3>(builtin::position, Vector3::Zero());
    auto vel = state_geo.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());
    feature->copy_to(state_geo);
    auto pos_view = view(*pos);
    auto vel_view = view(*vel);
    REQUIRE(pos_view.size() == positions.size());
    REQUIRE(vel_view.size() == velocities.size());
    for(size_t i = 0; i < pos_view.size(); ++i)
    {
        pos_view[i] = positions[i];
        vel_view[i] = velocities[i];
    }
    feature->copy_from(state_geo);
}
}  // namespace t53

// ===========================================================================
//  1) the three BDF2 helper templates against hand-derived values
// ===========================================================================
TEST_CASE("fem_dynamics_bdf2_host_oracle", "[luisa][fem_dynamics]")
{
    using namespace t53;
    test_runtime();

    std::mt19937                           rng(20240917u);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    const double                           steps[3] = {1e-3, 1e-2, 0.1};

    double worst_x_tilde   = 0.0;
    double worst_x_tilde_g = 0.0;
    double worst_v         = 0.0;
    int    samples         = 0;

    for(int trial = 0; trial < 60; ++trial)
    {
        const Vector3 Xn{dist(rng), dist(rng), dist(rng)};
        const Vector3 Xn1{dist(rng), dist(rng), dist(rng)};
        const Vector3 Vn{dist(rng), dist(rng), dist(rng)};
        const Vector3 Vn1{dist(rng), dist(rng), dist(rng)};
        const Vector3 G{dist(rng), dist(rng), dist(rng)};
        const Vector3 X{dist(rng), dist(rng), dist(rng)};

        for(double h : steps)
        {
            // ---- compute_x_tilde without the gravity term
            // R = 4/3 x_n - 1/3 x_{n-1} + 2/9 h (4 v_n - v_{n-1})
            {
                lps::LVec<double, 3> r;
                sym::fem_bdf2::compute_x_tilde(r, lv(Xn), lv(Xn1), lv(Vn), lv(Vn1), h);

                const double  c_xn   = 4.0 / 3.0;
                const double  c_xn_1 = 1.0 / 3.0;
                const double  c_v    = (2.0 / 9.0) * h;
                const Vector3 expected = c_xn * Xn - c_xn_1 * Xn1 + c_v * (4.0 * Vn - Vn1);

                for(int i = 0; i < 3; ++i)
                    worst_x_tilde = std::max(worst_x_tilde, rel_err(r(i), expected(i)));
            }
            // ---- compute_x_tilde with the gravity term
            // R = 4/9 h^2 g + 4/3 x_n - 1/3 x_{n-1} + 2/9 h (4 v_n - v_{n-1})
            {
                lps::LVec<double, 3> r;
                sym::fem_bdf2::compute_x_tilde(r, lv(Xn), lv(Xn1), lv(Vn), lv(Vn1), lv(G), h);

                const double  c_xn   = 4.0 / 3.0;
                const double  c_xn_1 = 1.0 / 3.0;
                const double  c_v    = (2.0 / 9.0) * h;
                const double  c_g    = (4.0 / 9.0) * h * h;
                const Vector3 expected =
                    c_g * G + c_xn * Xn - c_xn_1 * Xn1 + c_v * (4.0 * Vn - Vn1);

                for(int i = 0; i < 3; ++i)
                    worst_x_tilde_g = std::max(worst_x_tilde_g, rel_err(r(i), expected(i)));
            }
            // ---- compute_v: R = 1/(2h) (3 x - 4 x_n + x_{n-1})
            {
                lps::LVec<double, 3> r;
                sym::fem_bdf2::compute_v(r, lv(X), lv(Xn), lv(Xn1), h);

                const double  c        = 0.5 / h;
                const Vector3 expected = c * (3.0 * X - 4.0 * Xn + Xn1);

                for(int i = 0; i < 3; ++i)
                    worst_v = std::max(worst_v, rel_err(r(i), expected(i)));
            }
            ++samples;
        }
    }

    std::printf("[t53] bdf2 host oracle (%d samples): x_tilde rel %.3e  "
                "x_tilde_with_g rel %.3e  v rel %.3e\n",
                samples,
                worst_x_tilde,
                worst_x_tilde_g,
                worst_v);

    // the three formulas are exact rational multiples of their inputs, so the
    // ported routine and the hand-derived expression agree to round-off
    CHECK(worst_x_tilde <= 1e-14);
    CHECK(worst_x_tilde_g <= 1e-14);
    CHECK(worst_v <= 1e-14);
    CHECK(samples == 60 * 3);
}

// ===========================================================================
//  2) the kinetic energy/gradient/Hessian laws
// ===========================================================================
TEST_CASE("fem_dynamics_kinetic_host_oracle", "[luisa][fem_dynamics]")
{
    using namespace t53;
    test_runtime();

    // ---- the beta convention, as *doubles* (a float literal would fail here)
    CHECK(BDF2Kinetic::beta == 4.0 / 9.0);
    CHECK(BDF2Kinetic::inv_beta == 9.0 / 4.0);
    CHECK(BDF2Kinetic::inv_beta == 1.0 / BDF2Kinetic::beta);
    static_assert(std::is_same_v<decltype(BDF2Kinetic::beta), const double>);
    static_assert(std::is_same_v<gg::Float, double>);

    // ---- n = 6 hand-written vertices
    const int n = 6;
    const std::vector<Vector3> x       = {{0.5, 0.0, 0.0},
                                          {1.0, 0.5, 0.0},
                                          {0.0, 1.0, 0.0},
                                          {1.0, 1.0, 1.0},
                                          {-1.0, 0.25, 2.0},
                                          {0.75, -0.5, 0.5}};
    const std::vector<Vector3> x_tilde = {{0.0, 0.0, 0.0},
                                          {0.5, 0.0, 0.0},
                                          {0.25, 0.75, 0.0},
                                          {2.0, 0.0, 1.0},
                                          {-1.5, 1.25, 1.0},
                                          {0.5, 0.5, 0.25}};
    const std::vector<double>  m       = {1.0, 2.5, 0.125, 4.0, 0.5, 8.0};
    // vertex 3 and 5 are fixed, the other four are free
    const std::vector<IndexT> is_fixed = {0, 0, 0, 1, 0, 1};

    double worst_k = 0.0, worst_g = 0.0, worst_h = 0.0;
    double k_bdf1 = 0.0, k_bdf2 = 0.0;

    for(int i = 0; i < n; ++i)
    {
        const Vector3 d  = x[i] - x_tilde[i];
        const bool    fx = is_fixed[i] != 0;

        // K = is_fixed ? 0 : 0.5 m dot(d, d)
        const double e1 =
            bdf1_kinetic_detail::energy(lv(x[i]), lv(x_tilde[i]), m[i], is_fixed[i]);
        const double expected_e1 = fx ? 0.0 : 0.5 * m[i] * d.dot(d);
        worst_k                  = std::max(worst_k, rel_err(e1, expected_e1));

        // K = is_fixed ? 0 : inv_beta / 2 * dot(d, m d)   (inv_beta = 9/4)
        const double e2 = bdf2_kinetic_detail::energy(
            lv(x[i]), lv(x_tilde[i]), m[i], BDF2Kinetic::inv_beta, is_fixed[i]);
        const double expected_e2 = fx ? 0.0 : BDF2Kinetic::inv_beta / 2.0 * d.dot(m[i] * d);
        worst_k                  = std::max(worst_k, rel_err(e2, expected_e2));

        // G = is_fixed ? 0 : m d            (BDF1)
        // G = is_fixed ? 0 : inv_beta m d   (BDF2)
        auto g1 = bdf1_kinetic_detail::gradient(lv(x[i]), lv(x_tilde[i]), m[i], is_fixed[i]);
        auto g2 = bdf2_kinetic_detail::gradient(
            lv(x[i]), lv(x_tilde[i]), m[i], BDF2Kinetic::inv_beta, is_fixed[i]);
        const Vector3 zero3         = Vector3::Zero();
        const Vector3 expected_g1   = fx ? zero3 : m[i] * d;
        const Vector3 expected_g2   = fx ? zero3 : (BDF2Kinetic::inv_beta * m[i]) * d;
        for(int c = 0; c < 3; ++c)
        {
            worst_g = std::max(worst_g, rel_err(g1(c), expected_g1(c)));
            worst_g = std::max(worst_g, rel_err(g2(c), expected_g2(c)));
        }

        // H = m I (BDF1) / (inv_beta m) I (BDF2).
        //
        // NOTE: the reference's Hessian has **no** `is_fixed` guard -- see
        // src/backends/cuda/finite_element/bdf/finite_element_bdf{1,2}_kinetic.cu:
        //   `Matrix3x3 H = masses(i) * Matrix3x3::Identity();` is written
        //   unconditionally; only E and G are branched on is_fixed.
        // A fixed vertex therefore still receives `m I` on the diagonal (the
        // implicit system's mass regularisation), NOT zero.  That is asserted
        // deliberately here: the T5.3 brief's wording ("fixed => H = 0")
        // contradicts the reference, and the reference wins (see the report).
        auto       h1          = bdf1_kinetic_detail::hessian(m[i]);
        auto       h2          = bdf2_kinetic_detail::hessian(m[i], BDF2Kinetic::inv_beta);
        const auto expected_h1 = m[i] * Matrix3x3::Identity();
        const auto expected_h2 = (BDF2Kinetic::inv_beta * m[i]) * Matrix3x3::Identity();
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
            {
                worst_h = std::max(worst_h, rel_err(h1(r, c), expected_h1(r, c)));
                worst_h = std::max(worst_h, rel_err(h2(r, c), expected_h2(r, c)));
            }
        // ... and the off-diagonal *blocks* of the assembled Hessian are zero by
        // construction: the kernel writes only the (i, i) triplet per vertex, so a
        // hand-written "H_ij = 0 for i != j" holds trivially.  The diagonal blocks
        // of the two fixed vertices are pinned separately:
        if(fx)
        {
            CHECK(rel_err(h1(1, 1), m[i]) <= 1e-12);
            CHECK(h1(0, 1) == 0.0);
        }

        k_bdf1 += e1;
        k_bdf2 += e2;
    }

    // ---- the same Σ by hand over the whole mesh
    double K1 = 0.0, K2 = 0.0;
    for(int i = 0; i < n; ++i)
    {
        if(is_fixed[i])
            continue;
        const Vector3 d = x[i] - x_tilde[i];
        K1 += 0.5 * m[i] * d.dot(d);
        K2 += BDF2Kinetic::inv_beta / 2.0 * d.dot(m[i] * d);
    }
    CHECK(rel_err(k_bdf1, K1) <= 1e-12);
    CHECK(rel_err(k_bdf2, K2) <= 1e-12);
    // the BDF2 kinetic energy is inv_beta times the BDF1 one, which pins the
    // scaling independently of the per-vertex values
    CHECK(rel_err(k_bdf2, BDF2Kinetic::inv_beta * K1) <= 1e-12);
    CHECK(K1 > 0.0);

    std::printf("[t53] kinetic oracle (n=%d): K rel %.3e  G rel %.3e  H rel %.3e  "
                "(K_bdf1 %.6f, K_bdf2 %.6f, bdf2/bdf1 %.6f)\n",
                n,
                worst_k,
                worst_g,
                worst_h,
                k_bdf1,
                k_bdf2,
                k_bdf2 / k_bdf1);

    CHECK(worst_k <= 1e-12);
    CHECK(worst_g <= 1e-12);
    CHECK(worst_h <= 1e-12);
}

// ===========================================================================
//  3) the BDF1 predict/update pair over device buffers
// ===========================================================================
TEST_CASE("fem_dynamics_bdf1_integrator", "[luisa][fem_dynamics]")
{
    using namespace t53;
    test_runtime();
    auto& stream = gg::frame_stream();

    const int n = 8;
    // 0,1 dynamic; 2 static; 3,4,5 fixed (3 dynamic, 4 static, 5 dynamic); 6,7 dynamic
    const std::vector<IndexT> is_fixed_h   = {0, 0, 0, 1, 1, 1, 0, 0};
    const std::vector<IndexT> is_dynamic_h = {1, 1, 0, 1, 0, 1, 1, 1};

    const std::vector<Vector3> xs_h     = {{0.0, 0.0, 0.0},
                                           {1.0, 0.0, 0.0},
                                           {0.0, 1.0, 0.0},
                                           {0.0, 0.0, 1.0},
                                           {1.0, 1.0, 0.0},
                                           {1.0, 0.0, 1.0},
                                           {0.0, 1.0, 1.0},
                                           {1.0, 1.0, 1.0}};
    const std::vector<Vector3> vs_h     = {{0.5, -0.25, 0.125},
                                           {-1.0, 0.5, 2.0},
                                           {0.0, 0.0, 0.0},
                                           {3.0, -3.0, 0.5},
                                           {0.25, 0.25, -0.25},
                                           {-0.5, 1.5, -1.5},
                                           {2.0, 2.0, 0.0},
                                           {0.0, -1.0, -1.0}};
    const std::vector<Vector3> g_h      = {{0.0, -9.8, 0.0},
                                           {0.0, -9.8, 0.5},
                                           {0.25, -9.8, -0.25},
                                           {0.0, -9.8, 0.0},
                                           {1.0, 1.0, 1.0},
                                           {0.0, -9.8, 2.0},
                                           {-1.0, 0.5, 0.0},
                                           {0.0, 0.0, -9.8}};
    const std::vector<Vector3> a_h      = {{0.1, 0.2, 0.3},
                                           {-0.1, -0.2, -0.3},
                                           {0.5, 0.5, 0.5},
                                           {0.0, 0.0, 0.0},
                                           {1.0, -1.0, 0.0},
                                           {0.25, 0.0, -0.25},
                                           {0.0, 0.0, 0.0},
                                           {-0.5, -0.5, -0.5}};
    const double dt = 0.02;

    lud::LDeviceBuffer<IndexT>       d_is_fixed, d_is_dynamic;
    lud::LDeviceBuffer<gg::LVector3> d_xs, d_vs, d_g, d_a, d_x_prevs, d_x_tildes;
    copy_index_to_device(d_is_fixed, is_fixed_h);
    copy_index_to_device(d_is_dynamic, is_dynamic_h);
    copy_vec3_to_device(d_xs, xs_h);
    copy_vec3_to_device(d_vs, vs_h);
    copy_vec3_to_device(d_g, g_h);
    copy_vec3_to_device(d_a, a_h);
    // sentinels: a fixed vertex must keep x_prev (never the sentinel)
    copy_vec3_to_device(d_x_prevs, std::vector<Vector3>(n, Vector3{7.0, 7.0, 7.0}));
    copy_vec3_to_device(d_x_tildes, std::vector<Vector3>(n, Vector3{-7.0, -7.0, -7.0}));

    const auto nn = static_cast<uint32_t>(n);
    {
        auto& k = bdf1_predict_probe_kernel();
        stream.record(k(d_is_fixed.cview(),
                        d_is_dynamic.cview(),
                        d_x_prevs.view(),
                        d_xs.cview(),
                        d_vs.cview(),
                        d_x_tildes.view(),
                        d_g.cview(),
                        d_a.cview(),
                        dt,
                        nn)
                          .dispatch(nn));
    }
    auto x_prevs  = read_vec3(d_x_prevs.cview());
    auto x_tildes = read_vec3(d_x_tildes.cview());

    REQUIRE(x_prevs.size() == static_cast<size_t>(n));
    REQUIRE(x_tildes.size() == static_cast<size_t>(n));

    double worst_prev = 0.0, worst_tilde = 0.0, worst_law = 0.0;
    for(int i = 0; i < n; ++i)
    {
        // 0) every vertex records x_prev = x
        for(int c = 0; c < 3; ++c)
            worst_prev = std::max(worst_prev, rel_err(x_prevs[i](c), xs_h[i](c)));

        // the reference's branch chain, hand-derived:
        //   x_tilde = x_prev; if !fixed: += (g + a) dt^2; if !fixed && dynamic: += v dt
        Vector3 expected = xs_h[i];
        if(!is_fixed_h[i])
        {
            expected += (g_h[i] + a_h[i]) * dt * dt;
            if(is_dynamic_h[i])
                expected += vs_h[i] * dt;
        }
        for(int c = 0; c < 3; ++c)
            worst_tilde = std::max(worst_tilde, rel_err(x_tildes[i](c), expected(c)));

        // the *scalar law* the brief names, recomputed from the ported output:
        //   (x_tilde - x_prev) / dt^2 == g + a           (static)
        //                               == v/dt + g + a  (dynamic)
        //                               == 0             (fixed)
        const Vector3 delta = (x_tildes[i] - x_prevs[i]) / (dt * dt);
        Vector3       law;
        if(is_fixed_h[i])
            law = Vector3::Zero();
        else if(is_dynamic_h[i])
            law = vs_h[i] / dt + g_h[i] + a_h[i];
        else
            law = g_h[i] + a_h[i];
        for(int c = 0; c < 3; ++c)
            worst_law = std::max(worst_law, rel_err(delta(c), law(c)));
    }

    // ---- update_state: v = (x - x_prev) / dt, with x modified in the buffer
    auto xs_after_h = xs_h;
    for(int i = 0; i < n; ++i)
        xs_after_h[i] += Vector3{0.25 * (i + 1), -0.5, 0.125 * (i + 1)};
    copy_vec3_to_device(d_xs, xs_after_h);
    {
        auto& k = bdf1_update_probe_kernel();
        stream.record(k(d_xs.cview(), d_vs.view(), d_x_prevs.cview(), dt, nn).dispatch(nn));
    }
    auto vs_new = read_vec3(d_vs.cview());

    double worst_v = 0.0;
    for(int i = 0; i < n; ++i)
    {
        const Vector3 expected = (xs_after_h[i] - xs_h[i]) * (1.0 / dt);
        for(int c = 0; c < 3; ++c)
            worst_v = std::max(worst_v, rel_err(vs_new[i](c), expected(c)));
    }

    std::printf("[t53] bdf1 integrator: x_prev rel %.3e  x_tilde rel %.3e  "
                "scalar law rel %.3e  v rel %.3e  (fixed vertices %d)\n",
                worst_prev,
                worst_tilde,
                worst_law,
                worst_v,
                static_cast<int>(std::count(is_fixed_h.begin(), is_fixed_h.end(), IndexT{1})));

    CHECK(worst_prev == 0.0);
    CHECK(worst_tilde <= 1e-12);
    CHECK(worst_law <= 1e-12);
    CHECK(worst_v <= 1e-12);
}

// ===========================================================================
//  4) the BDF2 predict/update pair over device buffers
// ===========================================================================
TEST_CASE("fem_dynamics_bdf2_integrator", "[luisa][fem_dynamics]")
{
    using namespace t53;
    test_runtime();
    auto& stream = gg::frame_stream();

    const int n = 8;
    // 0,1 dynamic; 2 static; 3,4,5 fixed (3 dynamic, 4 static, 5 dynamic); 6,7 dynamic
    const std::vector<IndexT> is_fixed_h   = {0, 0, 0, 1, 1, 1, 0, 0};
    const std::vector<IndexT> is_dynamic_h = {1, 1, 0, 1, 0, 1, 1, 1};

    const std::vector<Vector3> xs_h     = {{0.0, 0.0, 0.0},
                                           {1.0, 0.0, 0.0},
                                           {0.0, 1.0, 0.0},
                                           {0.0, 0.0, 1.0},
                                           {1.0, 1.0, 0.0},
                                           {1.0, 0.0, 1.0},
                                           {0.0, 1.0, 1.0},
                                           {1.0, 1.0, 1.0}};
    const std::vector<Vector3> x_n_1_h  = {{-0.5, 0.25, 0.125},
                                           {0.5, -0.5, 1.0},
                                           {-1.0, 0.0, 0.5},
                                           {4.0, 4.0, 4.0},
                                           {5.0, 5.0, 5.0},
                                           {6.0, 6.0, 6.0},
                                           {0.25, 0.25, 0.25},
                                           {-1.0, -1.0, -1.0}};
    const std::vector<Vector3> v_n_h    = {{0.5, -0.25, 0.125},
                                           {-1.0, 0.5, 2.0},
                                           {1.0, 1.0, 1.0},
                                           {9.0, 9.0, 9.0},
                                           {8.0, 8.0, 8.0},
                                           {7.0, 7.0, 7.0},
                                           {2.0, 2.0, 0.0},
                                           {0.0, -1.0, -1.0}};
    const std::vector<Vector3> v_n_1_h  = {{0.25, 0.25, -0.25},
                                           {0.5, 0.5, 0.5},
                                           {-1.0, 2.0, 0.0},
                                           {1.0, 1.0, 1.0},
                                           {2.0, 2.0, 2.0},
                                           {3.0, 3.0, 3.0},
                                           {-0.5, 0.5, 0.0},
                                           {1.0, 1.0, 1.0}};
    const std::vector<Vector3> g_h      = {{0.0, -9.8, 0.0},
                                           {0.0, -9.8, 0.5},
                                           {0.25, -9.8, -0.25},
                                           {0.0, -9.8, 0.0},
                                           {1.0, 1.0, 1.0},
                                           {0.0, -9.8, 2.0},
                                           {-1.0, 0.5, 0.0},
                                           {0.0, 0.0, -9.8}};
    const std::vector<Vector3> a_h      = {{0.1, 0.2, 0.3},
                                           {-0.1, -0.2, -0.3},
                                           {0.5, 0.5, 0.5},
                                           {0.0, 0.0, 0.0},
                                           {1.0, -1.0, 0.0},
                                           {0.25, 0.0, -0.25},
                                           {0.0, 0.0, 0.0},
                                           {-0.5, -0.5, -0.5}};
    const double dt = 0.02;
    const auto   nn = static_cast<uint32_t>(n);

    lud::LDeviceBuffer<IndexT>       d_is_fixed, d_is_dynamic;
    lud::LDeviceBuffer<gg::LVector3> d_xs, d_x_ns, d_x_n_1s, d_v_ns, d_v_n_1s;
    lud::LDeviceBuffer<gg::LVector3> d_x_tildes, d_g, d_a;
    copy_index_to_device(d_is_fixed, is_fixed_h);
    copy_index_to_device(d_is_dynamic, is_dynamic_h);
    copy_vec3_to_device(d_xs, xs_h);
    copy_vec3_to_device(d_x_n_1s, x_n_1_h);
    copy_vec3_to_device(d_v_ns, v_n_h);
    copy_vec3_to_device(d_v_n_1s, v_n_1_h);
    copy_vec3_to_device(d_g, g_h);
    copy_vec3_to_device(d_a, a_h);
    copy_vec3_to_device(d_x_ns, std::vector<Vector3>(n, Vector3{7.0, 7.0, 7.0}));
    copy_vec3_to_device(d_x_tildes, std::vector<Vector3>(n, Vector3{-7.0, -7.0, -7.0}));

    {
        auto& k = bdf2_predict_probe_kernel();
        stream.record(k(d_is_fixed.cview(),
                        d_is_dynamic.cview(),
                        d_xs.cview(),
                        d_x_ns.view(),
                        d_x_n_1s.cview(),
                        d_v_ns.cview(),
                        d_v_n_1s.cview(),
                        d_x_tildes.view(),
                        d_g.cview(),
                        d_a.cview(),
                        dt,
                        nn)
                          .dispatch(nn));
    }
    auto x_ns     = read_vec3(d_x_ns.cview());
    auto x_tildes = read_vec3(d_x_tildes.cview());

    double worst_n = 0.0, worst_t = 0.0;
    for(int i = 0; i < n; ++i)
    {
        // x_n is recorded into x_prevs for the current step
        for(int c = 0; c < 3; ++c)
            worst_n = std::max(worst_n, rel_err(x_ns[i](c), xs_h[i](c)));

        // hand-derived from the reference's kernel:
        Vector3 x_n   = xs_h[i];
        Vector3 x_n_1 = x_n_1_h[i];
        Vector3 v_n   = v_n_h[i];
        Vector3 v_n_1 = v_n_1_h[i];
        Vector3 gtot  = g_h[i] + a_h[i];
        if(is_fixed_h[i])
        {
            x_n_1 = x_n;
            v_n.setZero();
            v_n_1.setZero();
            gtot.setZero();
        }
        else if(!is_dynamic_h[i])
        {
            x_n_1 = x_n;
            v_n.setZero();
            v_n_1.setZero();
        }
        const Vector3 expected = (4.0 / 9.0) * dt * dt * gtot + (4.0 / 3.0) * x_n
                                 - (1.0 / 3.0) * x_n_1
                                 + (2.0 / 9.0) * dt * (4.0 * v_n - v_n_1);
        for(int c = 0; c < 3; ++c)
            worst_t = std::max(worst_t, rel_err(x_tildes[i](c), expected(c)));

        // the brief's named identities
        if(is_fixed_h[i])
        {
            for(int c = 0; c < 3; ++c)
                CHECK(rel_err(x_tildes[i](c), x_ns[i](c)) <= 1e-12);
        }
        else if(!is_dynamic_h[i])
        {
            // (x_tilde - x_n) / dt^2 == 4/9 g, because the kernel zeroes both
            // velocity histories (and x_{n-1} := x_n) on the static branch
            const Vector3 d = (x_tildes[i] - x_ns[i]) / (dt * dt);
            for(int c = 0; c < 3; ++c)
                CHECK(rel_err(d(c), (4.0 / 9.0) * (g_h[i](c) + a_h[i](c))) <= 1e-12);
        }
    }

    // ---- update_state
    auto xs_after_h = xs_h;
    for(int i = 0; i < n; ++i)
        xs_after_h[i] += Vector3{0.25 * (i + 1), -0.5, 0.125 * (i + 1)};
    copy_vec3_to_device(d_xs, xs_after_h);
    {
        auto& k = bdf2_update_probe_kernel();
        stream.record(k(d_xs.cview(),
                        d_x_ns.view(),
                        d_x_n_1s.view(),
                        d_v_ns.view(),
                        d_v_n_1s.view(),
                        dt,
                        nn)
                          .dispatch(nn));
    }
    auto v_ns_new   = read_vec3(d_v_ns.cview());
    auto v_n_1s_new = read_vec3(d_v_n_1s.cview());
    auto x_n_1s_new = read_vec3(d_x_n_1s.cview());

    double worst_v = 0.0, worst_hist = 0.0;
    for(int i = 0; i < n; ++i)
    {
        const Vector3 expected_v =
            (0.5 / dt) * (3.0 * xs_after_h[i] - 4.0 * xs_h[i] + x_n_1_h[i]);
        for(int c = 0; c < 3; ++c)
        {
            worst_v = std::max(worst_v, rel_err(v_ns_new[i](c), expected_v(c)));
            // the history: v_{n-1} := the *old* v_n, x_{n-1} := the old x_n
            worst_hist = std::max(worst_hist, rel_err(v_n_1s_new[i](c), v_n_h[i](c)));
            worst_hist = std::max(worst_hist, rel_err(x_n_1s_new[i](c), xs_h[i](c)));
        }
    }

    std::printf("[t53] bdf2 integrator: x_n rel %.3e  x_tilde rel %.3e  "
                "v rel %.3e  history rel %.3e\n",
                worst_n,
                worst_t,
                worst_v,
                worst_hist);

    CHECK(worst_n == 0.0);
    CHECK(worst_t <= 1e-12);
    CHECK(worst_v <= 1e-12);
    CHECK(worst_hist <= 1e-12);
}

// ===========================================================================
//  5) a real scene through the public API (both integrators, the BDF2 bootstrap
//     and FEMBDF2State's dump/recover round trip)
// ===========================================================================
TEST_CASE("fem_dynamics_scene", "[luisa][fem_dynamics]")
{
    using namespace t53;
    test_runtime();

    const int n = 8;
    // vertex 0 fixed, vertex 2 static (not dynamic), the rest dynamic
    const std::vector<IndexT> is_fixed   = {1, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<IndexT> is_dynamic = {1, 1, 0, 1, 1, 1, 1, 1};

    auto desc            = basic_desc();
    desc.is_fixed        = is_fixed;
    desc.is_dynamic      = is_dynamic;
    desc.dt              = 0.02;
    for(int i = 0; i < n; ++i)
        desc.gravities[i] = Vector3{0.0, -9.8, 0.1 * static_cast<double>(i)};

    // The scenes are seeded with a *velocity* through the public state-accessor
    // feature: at rest the integrators' velocity term would be invisible (and
    // BDF2's first-frame history would be all-zero).  The *positions* are left at
    // the scene's own values on purpose -- BDF2's bootstrap copies the method's
    // `x_prevs`, i.e. `x_n` as of the previous frame, which the method initialises
    // from `xs` at init; writing `xs` externally before the first frame would
    // therefore make the bootstrapped `x_{n-1}` differ from the current `x_n`
    // (measured, and a genuine consequence of the reference's bootstrap -- see
    // the T5.3 report).
    std::vector<Vector3> V0(n);
    for(int i = 0; i < n; ++i)
    {
        V0[i] = Vector3{0.25 * (i + 1), -0.5, 0.75 - 0.1 * i};
    }

    const double dt = desc.dt;

    // =======================================================================
    //  BDF1
    // =======================================================================
    {
        T53Probe::reset();
        T53ExternalForceReporter::reset();
        auto   scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_t53_scene_bdf1")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T53Probe::fem != nullptr);
        seed_state(world, desc.positions, V0);
        step_frame(world);

        auto& fern   = *T53Probe::fem;
        auto  xs     = read_vec3(fern.xs());
        auto  x_prev = read_vec3(fern.x_prevs());
        auto  x_til  = read_vec3(fern.x_tildes());
        auto  vs     = read_vec3(fern.vs());
        REQUIRE(x_prev.size() == static_cast<size_t>(n));
        REQUIRE(x_til.size() == static_cast<size_t>(n));
        REQUIRE(vs.size() == static_cast<size_t>(n));

        double worst_prev = 0.0, worst_til = 0.0, worst_v = 0.0;
        for(int i = 0; i < n; ++i)
        {
            // x_prev := xs, and the scene never moves xs (there is no solver yet)
            for(int c = 0; c < 3; ++c)
                worst_prev = std::max(worst_prev, rel_err(x_prev[i](c), desc.positions[i](c)));

            // the reference's chain: x_tilde = x_prev; if !fixed += (g+a) dt^2;
            // if !fixed && dynamic += v dt  (a == 0: no external force here)
            Vector3 expected = desc.positions[i];
            if(!is_fixed[i])
            {
                expected += desc.gravities[i] * dt * dt;
                if(is_dynamic[i])
                    expected += V0[i] * dt;
            }
            for(int c = 0; c < 3; ++c)
                worst_til = std::max(worst_til, rel_err(x_til[i](c), expected(c)));

            // update_state: v = (x - x_prev)/dt == 0 because xs did not move
            for(int c = 0; c < 3; ++c)
                worst_v = std::max(worst_v, std::abs(vs[i](c)));
            (void)xs;
        }
        std::printf("[t53] scene bdf1: x_prev rel %.3e  x_tilde rel %.3e  |v| %.3e\n",
                    worst_prev,
                    worst_til,
                    worst_v);
        CHECK(worst_prev == 0.0);
        CHECK(worst_til <= 1e-12);
        CHECK(worst_v <= 1e-12);
        // The BDF2 *state system* is built for every FEM scene (the reference's
        // `FEMBDF2State::do_build()` only requires the method -- which is what
        // makes the dump/recover path available), but a bdf1 scene never runs the
        // BDF2 integrator, so the state's history buffers stay at size 0:
        REQUIRE(T53Probe::bdf2_state != nullptr);
        // NOTE: `find<>` returns the registered object even when it was
        // invalidated, so the check is on validity (and the state is valid here
        // because its `do_build()` only requires the method)
        CHECK(T53Probe::bdf2_state->is_valid());
        CHECK(state_x_n_1s(T53Probe::bdf2_state).size() == 0);
        CHECK(state_v_n_1s(T53Probe::bdf2_state).size() == 0);
    }

    // =======================================================================
    //  BDF2 -- the bootstrap, then the state's dump/recover round trip
    // =======================================================================
    std::vector<Vector3> dumped_x_n_1s, dumped_v_n_1s;
    {
        auto d            = desc;
        d.integrator_type = "bdf2";
        T53Probe::reset();
        T53ExternalForceReporter::reset();
        auto   scene = build_scene(d);
        Engine engine{"luisa", workspace_path("uipc_luisa_t53_scene_bdf2_dump")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T53Probe::bdf2_state != nullptr);
        seed_state(world, desc.positions, V0);
        step_frame(world);

        auto& fern = *T53Probe::fem;
        auto  x_prev = read_vec3(fern.x_prevs());
        auto  x_til  = read_vec3(fern.x_tildes());
        auto  vs     = read_vec3(fern.vs());
        auto  x_n_1s = read_vec3(state_x_n_1s(T53Probe::bdf2_state));
        auto  v_n_1s = read_vec3(state_v_n_1s(T53Probe::bdf2_state));
        REQUIRE(x_n_1s.size() == static_cast<size_t>(n));
        REQUIRE(v_n_1s.size() == static_cast<size_t>(n));

        double worst_til = 0.0, worst_boot = 0.0, worst_v = 0.0;
        for(int i = 0; i < n; ++i)
        {
            // the bootstrap: on the first predict_dof the history is filled from
            // the method's x_prevs (== x_n of the previous frame == the scene's
            // positions here) and from vs (== V0)
            const Vector3 X = desc.positions[i];
            for(int c = 0; c < 3; ++c)
            {
                worst_boot = std::max(worst_boot, rel_err(x_n_1s[i](c), X(c)));
                worst_boot = std::max(worst_boot, rel_err(v_n_1s[i](c), V0[i](c)));
            }

            // the reference's chain with the bootstrapped history:
            //  fixed:          x_{n-1} = x_n, v = 0, g = 0  => x_tilde = x_n
            //  !dynamic:       x_{n-1} = x_n, v = 0        => x_tilde = x_n + 4/9 h^2 g
            //  dynamic: x_{n-1} = x_n = X, v_n = v_{n-1} = V0
            //           => x_tilde = X + 2/3 h V0 + 4/9 h^2 g, because
            //              2/9 h (4 V0 - V0) = 2/9 h 3 V0 = 2/3 h V0
            Vector3 expected;
            if(is_fixed[i])
                expected = X;
            else if(!is_dynamic[i])
                expected = X + (4.0 / 9.0) * dt * dt * desc.gravities[i];
            else
                expected = (4.0 / 9.0) * dt * dt * desc.gravities[i]
                           + (4.0 / 3.0) * X - (1.0 / 3.0) * X
                           + (2.0 / 9.0) * dt * (4.0 * V0[i] - V0[i]);
            for(int c = 0; c < 3; ++c)
                worst_til = std::max(worst_til, rel_err(x_til[i](c), expected(c)));

            // update_state: x_{n-1} == x_n == xs => v == 0
            for(int c = 0; c < 3; ++c)
                worst_v = std::max(worst_v, std::abs(vs[i](c)));
        }
        std::printf("[t53] scene bdf2 frame 1: bootstrap rel %.3e  x_tilde rel %.3e  "
                    "|v| %.3e\n",
                    worst_boot,
                    worst_til,
                    worst_v);
        CHECK(worst_boot <= 1e-12);
        CHECK(worst_til <= 1e-12);
        CHECK(worst_v <= 1e-12);
        // x_prevs is the frame's x_n (xs, i.e. the scene's own positions)
        for(int i = 0; i < n; ++i)
            CHECK(rel_err((x_prev[i] - desc.positions[i]).norm(), 0.0) <= 1e-12);

        REQUIRE(world.dump());
        dumped_x_n_1s = read_vec3(state_x_n_1s(T53Probe::bdf2_state));
        dumped_v_n_1s = read_vec3(state_v_n_1s(T53Probe::bdf2_state));
        // the dumped history is *not* what a fresh scene would produce (v != 0)
        bool nonzero = false;
        for(auto& v : dumped_v_n_1s)
            nonzero = nonzero || v.norm() > 1e-9;
        CHECK(nonzero);
    }
    {
        auto d            = desc;
        d.integrator_type = "bdf2";
        T53Probe::reset();
        T53ExternalForceReporter::reset();
        auto   scene = build_scene(d);
        Engine engine{"luisa", workspace_path("uipc_luisa_t53_scene_bdf2_dump")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T53Probe::bdf2_state != nullptr);
        step_frame(world);  // a *fresh* scene (vs == 0), so the history differs

        auto fresh_v = read_vec3(state_v_n_1s(T53Probe::bdf2_state));
        REQUIRE(fresh_v.size() == dumped_v_n_1s.size());
        bool differs = false;
        for(size_t i = 0; i < fresh_v.size(); ++i)
            differs = differs || (fresh_v[i] - dumped_v_n_1s[i]).norm() > 1e-9;
        CHECK(differs);

        CHECK(world.recover());
        auto recovered_x = read_vec3(state_x_n_1s(T53Probe::bdf2_state));
        auto recovered_v = read_vec3(state_v_n_1s(T53Probe::bdf2_state));
        REQUIRE(recovered_x.size() == dumped_x_n_1s.size());
        REQUIRE(recovered_v.size() == dumped_v_n_1s.size());
        bool x_bitwise = true, v_bitwise = true;
        for(size_t i = 0; i < recovered_x.size(); ++i)
        {
            x_bitwise = x_bitwise
                        && std::memcmp(&recovered_x[i], &dumped_x_n_1s[i], sizeof(Vector3)) == 0;
            v_bitwise = v_bitwise
                        && std::memcmp(&recovered_v[i], &dumped_v_n_1s[i], sizeof(Vector3)) == 0;
        }
        std::printf("[t53] FEMBDF2State dump/recover: x_n_1s bitwise %d, v_n_1s bitwise %d\n",
                    static_cast<int>(x_bitwise),
                    static_cast<int>(v_bitwise));
        CHECK(x_bitwise);
        CHECK(v_bitwise);
    }
}

// ===========================================================================
//  6) the FEM external-force manager / reporter contract
// ===========================================================================
TEST_CASE("fem_dynamics_external_force_manager", "[luisa][fem_dynamics]")
{
    using namespace t53;
    test_runtime();

    const int n = 8;
    auto      desc     = basic_desc();
    desc.dt            = 0.02;
    const std::vector<Vector3> forces = {{10.0, 0.0, -5.0},
                                         {0.0, 0.0, 0.0},
                                         {0.0, 0.0, 0.0},
                                         {0.0, -20.0, 0.0},
                                         {1.0, 2.0, 3.0},
                                         {0.0, 0.0, 0.0},
                                         {0.0, 0.0, 0.0},
                                         {0.5, 0.5, 0.5}};
    const std::vector<IndexT>  enabled = {1, 0, 0, 1, 1, 0, 0, 1};
    desc.forces                        = forces;
    desc.force_enabled                 = enabled;
    desc.densities[4]                  = 0.0;  // m = 0 for vertex 4

    T53Probe::reset();
    T53ExternalForceReporter::reset();
    auto   scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t53_force_test")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T53Probe::fem != nullptr);
    auto& fern = *T53Probe::fem;

    // the reporter's init ran once during init_scene and collected the marked set
    CHECK(T53ExternalForceReporter::init_count == 1);
    CHECK(T53ExternalForceReporter::step_count == 0);
    CHECK(T53ExternalForceReporter::collected_indices == std::vector<IndexT>{0, 3, 4, 7});

    // ---- frame 1: the reporter writes
    step_frame(world);
    CHECK(T53ExternalForceReporter::step_count == 1);

    const auto masses = read_device(fern.masses());
    auto       force  = read_vec3(fern.vertex_external_forces());
    auto       accs   = read_vec3(fern.vertex_external_force_accs());
    REQUIRE(masses.size() == static_cast<size_t>(n));
    REQUIRE(force.size() == static_cast<size_t>(n));
    REQUIRE(accs.size() == static_cast<size_t>(n));
    REQUIRE(masses[4] == 0.0);  // the massless vertex really is massless

    double worst_f = 0.0, worst_a = 0.0;
    int    massless = 0;
    for(int i = 0; i < n; ++i)
    {
        // clear() -> the buffer holds exactly the reporter's scatter-add
        const Vector3 expected_f = enabled[i] ? forces[i] : Vector3::Zero();
        for(int c = 0; c < 3; ++c)
            worst_f = std::max(worst_f, rel_err(force[i](c), expected_f(c)));

        // accs = F / m, and exactly 0 when m <= 0 (the reference's guard)
        Vector3 expected_a;
        if(masses[i] > 0.0)
            expected_a = expected_f / masses[i];
        else
        {
            expected_a = Vector3::Zero();
            ++massless;
        }
        for(int c = 0; c < 3; ++c)
            worst_a = std::max(worst_a, rel_err(accs[i](c), expected_a(c)));
    }
    CHECK(massless == 1);
    CHECK(accs[4] == Vector3::Zero());
    CHECK(worst_f <= 1e-12);
    CHECK(worst_a <= 1e-12);
    std::printf("[t53] external force: forces rel %.3e  accs rel %.3e  "
                "(massless vertices %d)\n",
                worst_f,
                worst_a,
                massless);

    // ---- frame 2: the reporter is silenced, so ONLY the manager's clear() can
    // zero the buffer -- this is what proves the clear -> step order
    T53ExternalForceReporter::write_enabled = false;
    step_frame(world);
    CHECK(T53ExternalForceReporter::step_count == 2);
    force = read_vec3(fern.vertex_external_forces());
    accs  = read_vec3(fern.vertex_external_force_accs());
    bool all_zero_force = true, all_zero_acc = true;
    for(int i = 0; i < n; ++i)
    {
        all_zero_force = all_zero_force && force[i] == Vector3::Zero();
        all_zero_acc   = all_zero_acc && accs[i] == Vector3::Zero();
    }
    CHECK(all_zero_force);
    CHECK(all_zero_acc);
    std::printf("[t53] external force clear(): forces all zero %d, accs all zero %d\n",
                static_cast<int>(all_zero_force),
                static_cast<int>(all_zero_acc));
}

// ===========================================================================
//  7) device == host for every ported routine
// ===========================================================================
TEST_CASE("fem_dynamics_device_vs_host", "[luisa][fem_dynamics]")
{
    using namespace t53;
    test_runtime();

    std::vector<double> host;
    host.reserve(kRoundCount);
    fem_dynamics_round<double, IndexT>([&](double value) { host.push_back(value); });
    REQUIRE(static_cast<int>(host.size()) == kRoundCount);

    lud::LDeviceBuffer<double> device(kRoundCount);
    auto&                      stream = gg::frame_stream();
    stream.record(fem_dynamics_round_kernel()(device.view()).dispatch(1u));
    auto device_host = read_device(device.cview());
    REQUIRE(device_host.size() == host.size());

    int    bitwise_equal = 0;
    double worst_abs     = 0.0;
    int    worst_idx     = -1;
    for(int i = 0; i < kRoundCount; ++i)
    {
        if(std::memcmp(&device_host[i], &host[i], sizeof(double)) == 0)
            ++bitwise_equal;
        const double diff = std::abs(device_host[i] - host[i]);
        if(diff > worst_abs)
        {
            worst_abs = diff;
            worst_idx = i;
        }
    }
    std::printf("[t53] device-vs-host round: %d/%d bitwise equal, worst abs %.3e (entry %d)\n",
                bitwise_equal,
                kRoundCount,
                worst_abs,
                worst_idx);
    // no transcendental is involved and the arithmetic is the same code on both
    // sides, so the result must be bit-identical (T2.1's convention); the bound is
    // the documented fallback if a future backend contraction (FMA) changes it
    CHECK(bitwise_equal == kRoundCount);
    CHECK(worst_abs <= 1e-12);

    // the BDF2 kinetic inv_beta scaling is visible in the dump: entry 44 is the
    // BDF2 free-vertex kinetic energy and entry 27 the BDF1 one
    CHECK(rel_err(host[44], BDF2Kinetic::inv_beta * host[27]) <= 1e-12);
    // entry 28 (BDF1 fixed-vertex energy) and 45 (BDF2 fixed-vertex energy) are
    // exactly zero
    CHECK(host[28] == 0.0);
    CHECK(host[45] == 0.0);
    // ... while the fixed-vertex Hessian blocks (35..43 / 52..60) are m I, not 0
    CHECK(rel_err(host[35], 0.375) <= 1e-15);
    CHECK(rel_err(host[52], BDF2Kinetic::inv_beta * 0.375) <= 1e-15);
}

// ===========================================================================
//  8) the pulled-forward SoftPositionConstraint is built *and* inert
// ===========================================================================
TEST_CASE("fem_dynamics_soft_position_constraint", "[luisa][fem_dynamics]")
{
    using namespace t53;
    test_runtime();

    const int n = 8;
    auto      desc = basic_desc();
    desc.dt        = 0.02;
    for(int i = 0; i < n; ++i)
        desc.gravities[i] = Vector3{0.0, -9.8, 0.1 * static_cast<double>(i)};
    // vertices 1 and 6 are the constrained ones: the constraint's `do_step` is
    // observable through the host/device set it collects from the geometry
    desc.is_constrained = {0, 1, 0, 0, 0, 0, 1, 0};

    struct Result
    {
        std::vector<Vector3> xs, x_prevs, x_tildes, vs;
    };

    auto run = [&](bool with_constraint) -> Result
    {
        T53Probe::reset();
        T53ExternalForceReporter::reset();
        auto d            = desc;
        d.with_constraint = with_constraint;
        auto   scene      = build_scene(d);
        Engine engine{"luisa",
                      workspace_path(with_constraint ? "uipc_luisa_t53_spc" : "uipc_luisa_t53_nospc")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T53Probe::fem != nullptr);
        // With the constraint applied the geometry carries `constraint_uids` =
        // {14}, so `FiniteElementAnimator::Impl::init()` -- reached through
        // `GlobalAnimator::init()` -- looks uid 14 up in its uid -> constraint map
        // and *aborts* when no `FiniteElementConstraint` claims it.  That is why
        // the pulled-forward SoftPositionConstraint is mandatory; its `do_step` is
        // then observable through the set it collects (and it must leave the
        // dynamics untouched: it is pure energy until a Phase-8 solver asks for
        // its energy/gradient/Hessian).
        CHECK(T53Probe::animator != nullptr);
        // The SoftPositionConstraint *system* is created for every scene (the
        // registry specialisation is process-wide) but is invalidated whenever uid
        // 14 is not part of the scene's ConstitutionTabular -- and `find<>` still
        // hands back the invalidated object, so this check is on validity.
        REQUIRE(T53Probe::spc != nullptr);
        CHECK(T53Probe::spc->is_valid() == with_constraint);
        for(int f = 0; f < 2; ++f)
            step_frame(world);
        if(with_constraint)
        {
            REQUIRE(T53Probe::spc != nullptr);
            // exactly the two marked vertices, in geometry/vertex order, with
            // `apply_to(sc, 100.0)`'s strength ratio and the aim position shared
            // from the geometry's own positions
            // `uipc::vector` is `std::pmr::vector`, so it is compared element-wise
            // (a `std::vector` has no operator== against the pmr flavour)
            REQUIRE(T53Probe::spc->h_constrained_vertices.size() == 2);
            CHECK(T53Probe::spc->h_constrained_vertices[0] == 1);
            CHECK(T53Probe::spc->h_constrained_vertices[1] == 6);
            REQUIRE(T53Probe::spc->h_strength_ratios.size() == 2);
            CHECK(T53Probe::spc->h_strength_ratios[0] == 100.0);
            CHECK(T53Probe::spc->h_strength_ratios[1] == 100.0);
            REQUIRE(T53Probe::spc->h_aim_positions.size() == 2);
            CHECK(rel_err((T53Probe::spc->h_aim_positions[0] - desc.positions[1]).norm(), 0.0)
                  <= 1e-12);
            CHECK(rel_err((T53Probe::spc->h_aim_positions[1] - desc.positions[6]).norm(), 0.0)
                  <= 1e-12);
            CHECK(T53Probe::spc->constrained_vertices.size() == 2);
            CHECK(T53Probe::spc->strength_ratios.size() == 2);
            CHECK(T53Probe::spc->aim_positions.size() == 2);
        }
        Result r;
        auto&  fern = *T53Probe::fem;
        r.xs        = read_vec3(fern.xs());
        r.x_prevs   = read_vec3(fern.x_prevs());
        r.x_tildes  = read_vec3(fern.x_tildes());
        r.vs        = read_vec3(fern.vs());
        return r;
    };

    auto plain = run(false);
    auto with  = run(true);

    REQUIRE(plain.xs.size() == with.xs.size());
    double worst = 0.0;
    for(size_t i = 0; i < plain.xs.size(); ++i)
    {
        worst = std::max(worst, (plain.xs[i] - with.xs[i]).norm());
        worst = std::max(worst, (plain.x_prevs[i] - with.x_prevs[i]).norm());
        worst = std::max(worst, (plain.x_tildes[i] - with.x_tildes[i]).norm());
        worst = std::max(worst, (plain.vs[i] - with.vs[i]).norm());
    }
    std::printf("[t53] SoftPositionConstraint inertness: worst |state difference| %.3e\n", worst);
    // the constraint is pure energy: its do_step only refreshes its own
    // aim_positions/strength_ratio buffers and never writes xs/x_tildes/v, so a
    // scene with it must behave *identically* to one without it
    CHECK(worst == 0.0);
}
