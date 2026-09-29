/********************************************************************
 * @file luisa_abd_dynamics_test.cpp
 * @brief T6.2 — ABD dynamics: the two BDF integrators, the BDF2 history state and
 * the two kinetic systems (LuisaCompute backend).
 *
 * Every oracle in this file is derived independently *here*: the reference backend
 * is CUDA-only and cannot be host-included, so nothing is copied from it.  The
 * oracles are
 *  (a) hand-derived closed forms for the BDF2 helper templates
 *      (`sym::abd_bdf2::{compute_q_tilde, compute_qv}`), which are the 12-dof
 *      siblings of T5.3's `sym::fem_bdf2::*` -- the file cross-checks the two
 *      templates component-wise, which is what pins the transcription,
 *  (b) hand-derived closed forms for the two kinetic laws with the **ABD dyadic
 *      mass** (not the FEM scalar mass): `M = Jᵀ Mᵢ J` written out explicitly
 *      (`mass_matrix_oracle`), including the *unguarded* Hessian,
 *  (c) an independent host model of the integrators' per-body formulas, driven side
 *      by side with the *ported* per-body routines over the same device buffers
 *      (cookbook §7.4: the shipped kernel and the test run the same code),
 *  (d) the plan's headline steady-state oracle: a *consistent* constant-velocity
 *      history makes BDF2's advance equal BDF1's *and* equal the analytic value
 *      (`Q + V h`, `V`), plus the exact `(1 - 4/9) h² g` by which they differ when
 *      gravity is on (which is why the equality needs `g = 0`),
 *  (e) a real scene through the public `Engine`/`World` API, seeded with a known
 *      velocity through the ported `AffineBodyStateAccessorFeature` and read back
 *      through the same feature -- this is the only path that exercises the *shipped*
 *      kernels (predict_dof/update_state, the BDF2 history bootstrap) end to end --
 *      plus `ABDBDF2State`'s dump/recover round trip (bitwise),
 *  (f) the device-vs-host equivalence of every ported routine (`double` vs
 *      `lcrt::Var<double>` in a `compile1` kernel), expected *bitwise* where the
 *      arithmetic is identical (cookbook §7.4).
 *
 * ---------------------------------- T6.2 ----------------------------------
 * Coverage map (case -> oracle):
 *  1) the three BDF2 helper templates (+ the 3-dof sibling) ... abd_dynamics_bdf2_host_oracle
 *  2) kinetic energy/gradient/Hessian, both kinetics,
 *     the beta/inv_beta literals, the unguarded H ......... abd_dynamics_kinetic
 *  3) BDF1/BDF2 predict+update over device buffers ........ abd_dynamics_integrators
 *  4) the steady-state BDF2 == BDF1 oracle + transform
 *     invariants ......................................... abd_dynamics_steady_state
 *  5) a real scene, bdf1 + bdf2, the frontend read-back
 *     + ABDBDF2State dump/recover ........................ abd_dynamics_scene
 *  6) device == host for every ported routine ............. abd_dynamics_device_vs_host
 *  7) per-case assertion accounting ...................... abd_dynamics_summary
 *
 * Shared-binary rules this file obeys:
 *  * every system it registers is built for every scene of the whole test binary, so
 *    the probe only reads the ABD dynamics (invalidated by `require<>` elsewhere) and
 *    the scenes are ABD-only (cookbook §7.14 rule 6);
 *  * the file registers **no** `AffineBodyConstitution` and **no** kinetic, so it does
 *    not contest any uid (T6.1a's uid-1 mock) or the dynamics' single kinetic slot
 *    (T6.1b's stand-in, see the T6.2 note in luisa_abd_linear_test.cpp);
 *  * the scene case reads the dynamics' own device buffers through the probe *and*
 *    the frontend's `AffineBodyStateAccessorFeature`, never a global array whose
 *    length other test files' reporters also contribute to.
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"
#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/core/affine_body_state_accessor_feature.h>
#include <type_define.h>
#include <frame_stream.h>
#include <affine_body/type_define.h>
#include <affine_body/utils.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/abd_time_integrator.h>
#include <affine_body/bdf/bdf2_function.h>
#include <affine_body/bdf/abd_bdf1_time_integrator.h>
#include <affine_body/bdf/abd_bdf2_time_integrator.h>
#include <affine_body/bdf/abd_bdf2_state.h>
#include <affine_body/bdf/affine_body_bdf1_kinetic.h>
#include <affine_body/bdf/affine_body_bdf2_kinetic.h>
// The 3-dof sibling of the very same BDF2 formulas (T5.3), used by case 1 to pin the
// transcription component-wise.
#include <finite_element/bdf/bdf2_function.h>
#include <utils/scalar_util.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the shell header must
// be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>
#include <array>
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
// backend's does in `uipc::backend::cuda`), so a test body sees the same unqualified
// spellings a ported system does: `bdf1_detail::predict_dof`,
// `bdf2_detail::predict_dof`, `bdf1_kinetic_detail::energy`, `sym::abd_bdf2`,
// `load_vec12`/`store_vec12`, ...
using namespace uipc::backend::luisa;

namespace t62
{
/// The two kinetic systems that carry the `beta`/`inv_beta` convention.
using BDF2Kinetic = gg::AffineBodyBDF2Kinetic;

// ---------------------------------------------------------------------------
//  small helpers (the T5.3/T6.1 test files' idioms)
// ---------------------------------------------------------------------------
/// Scale-aware relative error (identical to the T5.3 helper).
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

[[nodiscard]] std::vector<Vector12> read_vec12(const lud::LCBufferView<gg::LVector12>& view)
{
    auto                   raw = read_device(view);
    std::vector<Vector12>  host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

[[nodiscard]] std::vector<IndexT> read_index(const lud::LCBufferView<IndexT>& view)
{
    return read_device(view);
}

/// `q`/`A` conventions (cookbook §7.21 rule 6): `q = [t; a1; a2; a3]` with `a_i` =
/// *row* `i` of `A`, so the deformed point is `p = A x̄ + t`.
[[nodiscard]] Matrix3x3 A_oracle(const Vector12& q) noexcept
{
    Matrix3x3 A;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            A(i, j) = q(3 + 3 * i + j);
    return A;
}

[[nodiscard]] Vector12 make_q(const Matrix3x3& A, const Vector3& t) noexcept
{
    Vector12 q = Vector12::Zero();
    q.segment<3>(0) = t;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            q(3 + 3 * i + j) = A(i, j);
    return q;
}

[[nodiscard]] Matrix4x4 make_transform(const Matrix3x3& A, const Vector3& t) noexcept
{
    Matrix4x4 T = Matrix4x4::Identity();
    T.block<3, 3>(0, 0) = A;
    T.block<3, 1>(0, 3) = t;
    return T;
}

/// The frontend stores velocities as 4x4 matrices whose last row is unused for `q_v`
/// (`transform_to_q` reads the 3x3 block and the translation only).
[[nodiscard]] Matrix4x4 make_velocity(const Matrix3x3& A, const Vector3& t) noexcept
{
    return make_transform(A, t);
}

/// A 3x3 built from nine literals.  (The Eigen comma initializer cannot be used
/// inside a braced init list: MSVC mis-parses the `<<`/`,` sequence there, measured
/// C3329/C2059 -- a helper keeps the literals readable.)
[[nodiscard]] Matrix3x3 m3(double a00,
                           double a01,
                           double a02,
                           double a10,
                           double a11,
                           double a12,
                           double a20,
                           double a21,
                           double a22) noexcept
{
    Matrix3x3 M;
    M << a00, a01, a02, a10, a11, a12, a20, a21, a22;
    return M;
}

[[nodiscard]] lps::LVec<double, 3> lv3d(const Vector3& v) noexcept
{
    return lps::LVec<double, 3>{v(0), v(1), v(2)};
}

[[nodiscard]] lps::LVec<double, 12> lv12d(const Vector12& v) noexcept
{
    return lps::LVec<double, 12>{v(0),
                                 v(1),
                                 v(2),
                                 v(3),
                                 v(4),
                                 v(5),
                                 v(6),
                                 v(7),
                                 v(8),
                                 v(9),
                                 v(10),
                                 v(11)};
}

/// `lv3<S>(a, b, c)` for a generic scalar `S` (`double` on the host, a DSL
/// `Var<double>` inside a kernel) -- the device-vs-host round needs the same literal
/// on both sides.
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

template <typename S>
[[nodiscard]] lps::LVec<S, 12> lv12(const std::array<double, 12>& a) noexcept
{
    return lps::make_vec<S, 12>([&](int i) noexcept
                                { return lps::make_scalar<S>(a[i]); });
}

/// Generic (`S`-templated) per-body dyadic-mass triad, so the same literals are used
/// by the host oracle and the device kernel.
template <typename S>
[[nodiscard]] ABDJacobiDyadicMassT<S> mass_triad(double                      m,
                                                 const std::array<double, 3>& mx,
                                                 const std::array<double, 9>& mxx) noexcept
{
    ABDJacobiDyadicMassT<S> r;
    r.m_mass = lps::make_scalar<S>(m);
    r.m_mass_times_x_bar = lv3<S>(mx[0], mx[1], mx[2]);
    r.m_mass_times_dyadic_x_bar =
        lps::make_mat<S, 3, 3>([&](int i, int j) noexcept
                               { return lps::make_scalar<S>(mxx[3 * i + j]); });
    return r;
}

/// The hand-derived ABD mass operator `M = Jᵀ Mᵢ J`, written out from the jacobian's
/// own layout (`J(a,a) = 1`, `J(i, 3+3i+j) = x̄_j`), so:
///   block(0, 0)        = m I₃
///   block(0, 1+i)      = only row i is non-zero, and it is `m x̄ᵀ`
///                        (equivalently: column i of `block(1+i, 0)` is `m x̄`)
///   block(1+i, 1+i)    = m x̄ ⊗ x̄ ,  off-diagonal 3x3 blocks zero
/// This is *the* difference from the FEM kinetic: there the operator is the scalar
/// `m I`, here it is a dense 12x12 with three non-trivial block bands.
[[nodiscard]] Matrix12x12 mass_matrix_oracle(double          m,
                                             const Vector3&  mx,
                                             const Matrix3x3& mxx) noexcept
{
    Matrix12x12 M = Matrix12x12::Zero();
    for(int a = 0; a < 3; ++a)
        M(a, a) = m;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            M(i, 3 + 3 * i + j) = mx(j);  // block(0, 1+i), row i only
            M(3 + 3 * i + j, i) = mx(j);  // block(1+i, 0), column i only
        }
        M.block<3, 3>(3 + 3 * i, 3 + 3 * i) = mxx;  // block(1+i, 1+i)
    }
    return M;
}

// ---------------------------------------------------------------------------
//  box geometry + its region moments (brief §3 of T6.1a; re-derived here)
//  int x_a dV = L^4/2 -> m x = m (L/2) (1,1,1)
//  int x_a^2 dV = L^5/3 -> mxx_aa = m L^2/3 ;  int x_a x_b dV = L^6/4 -> m L^2/4
// ---------------------------------------------------------------------------
struct BoxMoments
{
    double    volume = 0.0;
    double    m      = 0.0;
    Vector3   mx     = Vector3::Zero();
    Matrix3x3 mxx    = Matrix3x3::Zero();
};

[[nodiscard]] BoxMoments box_moments(double L, double rho) noexcept
{
    BoxMoments o;
    o.volume = L * L * L;
    o.m      = rho * o.volume;
    o.mx     = o.m * Vector3{L / 2.0, L / 2.0, L / 2.0};
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
            o.mxx(a, b) = o.m * ((a == b) ? (L * L / 3.0) : (L * L / 4.0));
    return o;
}

/// The six-tet decomposition of the cube `[0,L]^3` (all positively oriented; vertex
/// index = x + 2y + 4z).
[[nodiscard]] std::array<Vector4i, 6> cube_tets() noexcept
{
    return {{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}}};
}

[[nodiscard]] std::array<Vector3, 8> cube_vertices(double L) noexcept
{
    std::array<Vector3, 8> v;
    for(int i = 0; i < 8; ++i)
    {
        const double x = (i & 1) ? L : 0.0;
        const double y = (i & 2) ? L : 0.0;
        const double z = (i & 4) ? L : 0.0;
        v[i]           = Vector3{x, y, z};
    }
    return v;
}

/// One ABD body: the frontend's `AffineBodyConstitution` (uid 1) applied to a tet cube
/// with `body_count` instances.  `rest_map` moves the *material* configuration (a
/// rotated box is a different **rest** state -- the dyadic mass integrates the rest
/// positions -- while the instance transform carries the deformed state `q`).
[[nodiscard]] SimplicialComplex make_box_body(double                         L,
                                              double                         rho,
                                              double                         kappa,
                                              SizeT                          body_count,
                                              const std::vector<Matrix4x4>&  transforms,
                                              const std::vector<Matrix4x4>&  velocities = {})
{
    auto vs = cube_vertices(L);
    auto ts = cube_tets();
    auto sc = tetmesh(span<const Vector3>{vs.data(), vs.size()},
                      span<const Vector4i>{ts.data(), ts.size()});
    label_surface(sc);
    sc.instances().resize(body_count);

    // qualified: `uipc::constitution::AffineBodyConstitution` (the *frontend* type)
    // is ambiguous with the backend's `uipc::backend::luisa::AffineBodyConstitution`
    // once both namespaces are open (measured C2872)
    uipc::constitution::AffineBodyConstitution abd;
    abd.apply_to(sc, kappa, rho);

    auto T = view(sc.transforms());
    for(SizeT i = 0; i < body_count; ++i)
        T[i] = (i < transforms.size()) ? transforms[i] : Matrix4x4::Identity();

    if(!velocities.empty())
    {
        auto V = view(*sc.instances().find<Matrix4x4>(builtin::velocity));
        for(SizeT i = 0; i < body_count && i < velocities.size(); ++i)
            V[i] = velocities[i];
    }
    return sc;
}

// ---------------------------------------------------------------------------
//  the probe: grabs the ABD systems' public interfaces once they are built.  A scene
//  without an ABD body invalidates it (`require<>` throws SimSystemException), so
//  other test files are unaffected; the test resets the static pointers per case.
// ---------------------------------------------------------------------------
class T62Probe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;
    inline static gg::AffineBodyDynamics* abd        = nullptr;
    inline static gg::ABDBDF2State*       bdf2_state = nullptr;

    static void reset()
    {
        abd        = nullptr;
        bdf2_state = nullptr;
    }

  private:
    void do_build() override
    {
        abd        = &require<gg::AffineBodyDynamics>();
        bdf2_state = find<gg::ABDBDF2State>();
    }
};

// ---------------------------------------------------------------------------
//  per-case assertion accounting (brief §5.3)
// ---------------------------------------------------------------------------
struct Counter
{
    int checks = 0;
};

std::vector<std::pair<std::string, int>> g_case_counts;

struct CaseScope
{
    std::string name;
    Counter     counter;

    explicit CaseScope(std::string n)
        : name(std::move(n))
    {
    }
    ~CaseScope() { g_case_counts.emplace_back(name, counter.checks); }
};

#define T62_CHECK(counter, cond) \
    do                           \
    {                            \
        ++(counter).checks;      \
        CHECK(cond);             \
    } while(false)

}  // namespace t62
REGISTER_SIM_SYSTEM(t62::T62Probe);

namespace t62
{
// ---------------------------------------------------------------------------
//  The shipped kernels' host-side twins
//
//  The ABD BDF kernels live in an anonymous namespace inside the two `.cpp` files, so
//  a unit test cannot dispatch them; the *routines* they call (`bdf1_detail`,
//  `bdf2_detail`, `sym::abd_bdf2`) and the launch-argument list are the same, which is
//  what these four probe kernels transcribe.  The shipped kernels themselves are
//  exercised end to end by `abd_dynamics_scene`.
// ---------------------------------------------------------------------------
auto& bdf1_predict_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,    // is_fixed
                                                lcrt::Buffer<int32_t>,    // is_dynamic
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_vs
                                                lcrt::Buffer<LVector12>,  // q_tildes
                                                lcrt::Buffer<LVector12>,  // gravities
                                                lcrt::Buffer<LVector12>,  // external_force_accs
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<int32_t>   is_fixed,
           lcrt::BufferVar<int32_t>   is_dynamic,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_vs,
           lcrt::BufferVar<LVector12> q_tildes,
           lcrt::BufferVar<LVector12> gravities,
           lcrt::BufferVar<LVector12> external_force_accs,
           lcrt::Var<double>          dt,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                q_prevs->write(i, qs->read(i));
                auto q_tilde = bdf1_detail::predict_dof(load_vec12(qs->read(i)),
                                                       load_vec12(q_vs->read(i)),
                                                       load_vec12(gravities->read(i)),
                                                       load_vec12(external_force_accs->read(i)),
                                                       dt,
                                                       is_fixed->read(i),
                                                       is_dynamic->read(i));
                lcrt::Var<LVector12> out;
                store_vec12(out, q_tilde);
                q_tildes->write(i, out);
            };
        });
    return k;
}

auto& bdf1_update_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_vs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_vs,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::Var<double>          dt,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto q_v = bdf1_detail::update_state(
                    load_vec12(qs->read(i)), load_vec12(q_prevs->read(i)), dt);
                lcrt::Var<LVector12> out;
                store_vec12(out, q_v);
                q_vs->write(i, out);
            };
        });
    return k;
}

auto& bdf2_predict_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,    // is_fixed
                                                lcrt::Buffer<int32_t>,    // is_dynamic
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_ns
                                                lcrt::Buffer<LVector12>,  // q_n_1s
                                                lcrt::Buffer<LVector12>,  // q_v_ns
                                                lcrt::Buffer<LVector12>,  // q_v_n_1s
                                                lcrt::Buffer<LVector12>,  // q_tildes
                                                lcrt::Buffer<LVector12>,  // gravity
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<int32_t>   is_fixed,
           lcrt::BufferVar<int32_t>   is_dynamic,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_ns,
           lcrt::BufferVar<LVector12> q_n_1s,
           lcrt::BufferVar<LVector12> q_v_ns,
           lcrt::BufferVar<LVector12> q_v_n_1s,
           lcrt::BufferVar<LVector12> q_tildes,
           lcrt::BufferVar<LVector12> gravity,
           lcrt::Var<double>          dt,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                q_ns->write(i, qs->read(i));
                auto q_tilde = bdf2_detail::predict_dof(load_vec12(qs->read(i)),
                                                        load_vec12(q_n_1s->read(i)),
                                                        load_vec12(q_v_ns->read(i)),
                                                        load_vec12(q_v_n_1s->read(i)),
                                                        load_vec12(gravity->read(i)),
                                                        dt,
                                                        is_fixed->read(i),
                                                        is_dynamic->read(i));
                lcrt::Var<LVector12> out;
                store_vec12(out, q_tilde);
                q_tildes->write(i, out);
            };
        });
    return k;
}

auto& bdf2_update_probe_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_ns
                                                lcrt::Buffer<LVector12>,  // q_n_1s
                                                lcrt::Buffer<LVector12>,  // q_v_ns
                                                lcrt::Buffer<LVector12>,  // q_v_n_1s
                                                double,
                                                uint32_t>(
        [](lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_ns,
           lcrt::BufferVar<LVector12> q_n_1s,
           lcrt::BufferVar<LVector12> q_v_ns,
           lcrt::BufferVar<LVector12> q_v_n_1s,
           lcrt::Var<double>          dt,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto q   = load_vec12(qs->read(i));
                auto q_n = load_vec12(q_ns->read(i));
                lps::LVec<lcrt::Var<double>, 12> qv;
                sym::abd_bdf2::compute_qv(qv, q, q_n, load_vec12(q_n_1s->read(i)), dt);
                q_v_n_1s->write(i, q_v_ns->read(i));
                lcrt::Var<LVector12> qv_out;
                store_vec12(qv_out, qv);
                q_v_ns->write(i, qv_out);
                q_n_1s->write(i, q_ns->read(i));
            };
        });
    return k;
}

/// `ABDBDF2State`'s history views have a public *const* overload and a private
/// non-const one (the reference's layout: the mutable views belong to the
/// integrator), so a test has to read them through a `const` pointer -- the same
/// helper shape as T5.3's `state_x_n_1s`.
[[nodiscard]] lud::LCBufferView<gg::LVector12> state_q_n_1s(const gg::ABDBDF2State* state)
{
    return state->q_n_1s();
}
[[nodiscard]] lud::LCBufferView<gg::LVector12> state_q_v_n_1s(const gg::ABDBDF2State* state)
{
    return state->q_v_n_1s();
}

void copy_vec12_to_device(lud::LDeviceBuffer<gg::LVector12>& dst, const std::vector<Vector12>& src)
{
    std::vector<gg::LVector12> host;
    host.reserve(src.size());
    for(auto& v : src)
        host.push_back(gg::to_luisa(v));
    dst.resize(host.size());
    if(!host.empty())
        dst.view().copy_from(span<const gg::LVector12>{host});
    lud::main_stream().synchronize();
}

void copy_index_to_device(lud::LDeviceBuffer<IndexT>& dst, const std::vector<IndexT>& src)
{
    dst.resize(src.size());
    if(!src.empty())
        dst.view().copy_from(span<const IndexT>{src});
    lud::main_stream().synchronize();
}

/// A *device* sentinel 12-vector, so "the kernel did not write this" is visible.
[[nodiscard]] std::vector<Vector12> sentinel12(size_t n, double v)
{
    Vector12 s;
    for(int i = 0; i < 12; ++i)
        s(i) = v * (1.0 + 0.5 * i);
    return std::vector<Vector12>(n, s);
}
}  // namespace t62

namespace t62
{
// ===========================================================================
//  1) the three BDF2 helper templates against hand-derived values
// ===========================================================================
TEST_CASE("abd_dynamics_bdf2_host_oracle", "[luisa][abd_dynamics]")
{
    test_runtime();
    CaseScope scope{"abd_dynamics_bdf2_host_oracle"};
    auto&     c = scope.counter;

    std::mt19937                           rng(20240917u);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    const double                           steps[3] = {1e-3, 1e-2, 0.1};

    double worst_q_tilde   = 0.0;
    double worst_q_tilde_g = 0.0;
    double worst_qv        = 0.0;
    double worst_sibling   = 0.0;
    int    samples         = 0;
    int    sibling_samples = 0;

    for(int trial = 0; trial < 60; ++trial)
    {
        Vector12 Qn, Qn1, Qvn, Qvn1, G, Q;
        for(int i = 0; i < 12; ++i)
        {
            Qn(i)   = dist(rng);
            Qn1(i)  = dist(rng);
            Qvn(i)  = dist(rng);
            Qvn1(i) = dist(rng);
            G(i)    = dist(rng);
            Q(i)    = dist(rng);
        }

        for(double h : steps)
        {
            // ---- compute_q_tilde without the gravity term
            // R = 4/3 q_n - 1/3 q_{n-1} + 2/9 h (4 qv_n - qv_{n-1})
            {
                lps::LVec<double, 12> r;
                sym::abd_bdf2::compute_q_tilde(r, lv12d(Qn), lv12d(Qn1), lv12d(Qvn), lv12d(Qvn1), h);

                const double  c_qn   = 4.0 / 3.0;
                const double  c_qn_1 = 1.0 / 3.0;
                const double  c_v    = (2.0 / 9.0) * h;
                const Vector12 expected = c_qn * Qn - c_qn_1 * Qn1 + c_v * (4.0 * Qvn - Qvn1);

                for(int i = 0; i < 12; ++i)
                    worst_q_tilde = std::max(worst_q_tilde, rel_err(r(i), expected(i)));
            }
            // ---- compute_q_tilde with the gravity term
            // R = 4/9 h^2 g + 4/3 q_n - 1/3 q_{n-1} + 2/9 h (4 qv_n - qv_{n-1})
            {
                lps::LVec<double, 12> r;
                sym::abd_bdf2::compute_q_tilde(
                    r, lv12d(Qn), lv12d(Qn1), lv12d(Qvn), lv12d(Qvn1), lv12d(G), h);

                const double  c_qn   = 4.0 / 3.0;
                const double  c_qn_1 = 1.0 / 3.0;
                const double  c_v    = (2.0 / 9.0) * h;
                const double  c_g    = (4.0 / 9.0) * h * h;
                const Vector12 expected =
                    c_g * G + c_qn * Qn - c_qn_1 * Qn1 + c_v * (4.0 * Qvn - Qvn1);

                for(int i = 0; i < 12; ++i)
                    worst_q_tilde_g = std::max(worst_q_tilde_g, rel_err(r(i), expected(i)));
            }
            // ---- compute_qv: R = 1/(2h) (3 q - 4 q_n + q_{n-1})
            {
                lps::LVec<double, 12> r;
                sym::abd_bdf2::compute_qv(r, lv12d(Q), lv12d(Qn), lv12d(Qn1), h);

                const double   c        = 0.5 / h;
                const Vector12 expected = c * (3.0 * Q - 4.0 * Qn + Qn1);

                for(int i = 0; i < 12; ++i)
                    worst_qv = std::max(worst_qv, rel_err(r(i), expected(i)));
            }
            // ---- and the *same* formulas through the 3-dof FEM sibling: the ABD
            // transcription must agree component-wise *bitwise* (same statements, same
            // association order -- the only difference is the container width)
            {
                lps::LVec<double, 12> r12;
                lps::LVec<double, 3>  r3;
                sym::abd_bdf2::compute_q_tilde(r12,
                                               lv12d(Qn),
                                               lv12d(Qn1),
                                               lv12d(Qvn),
                                               lv12d(Qvn1),
                                               lv12d(G),
                                               h);
                sym::fem_bdf2::compute_x_tilde(r3,
                                               lv3d(Qn.head<3>()),
                                               lv3d(Qn1.head<3>()),
                                               lv3d(Qvn.head<3>()),
                                               lv3d(Qvn1.head<3>()),
                                               lv3d(G.head<3>()),
                                               h);
                for(int i = 0; i < 3; ++i)
                    worst_sibling = std::max(worst_sibling, std::abs(r12(i) - r3(i)));

                lps::LVec<double, 12> rv12;
                lps::LVec<double, 3>  rv3;
                sym::abd_bdf2::compute_qv(rv12, lv12d(Q), lv12d(Qn), lv12d(Qn1), h);
                sym::fem_bdf2::compute_v(rv3, lv3d(Q.head<3>()), lv3d(Qn.head<3>()), lv3d(Qn1.head<3>()), h);
                for(int i = 0; i < 3; ++i)
                    worst_sibling = std::max(worst_sibling, std::abs(rv12(i) - rv3(i)));
                ++sibling_samples;
            }
            ++samples;
        }
    }

    std::printf("[t62] bdf2 host oracle (%d samples, %d sibling): q_tilde rel %.3e  "
                "q_tilde_with_g rel %.3e  qv rel %.3e  |abd-fem| %.3e\n",
                samples,
                sibling_samples,
                worst_q_tilde,
                worst_q_tilde_g,
                worst_qv,
                worst_sibling);

    // the three formulas are exact rational multiples of their inputs, so the ported
    // routine and the hand-derived expression agree to round-off
    T62_CHECK(c, worst_q_tilde <= 1e-14);
    T62_CHECK(c, worst_q_tilde_g <= 1e-14);
    T62_CHECK(c, worst_qv <= 1e-14);
    T62_CHECK(c, samples == 60 * 3);
    T62_CHECK(c, sibling_samples == 60 * 3);
    // the 12-dof transcription and the 3-dof sibling run the same statements on the
    // same numbers, so their first three components are *identical*, not just close
    T62_CHECK(c, worst_sibling == 0.0);
}

// ===========================================================================
//  2) the kinetic energy/gradient/Hessian laws with the ABD mass operator
// ===========================================================================
TEST_CASE("abd_dynamics_kinetic", "[luisa][abd_dynamics]")
{
    test_runtime();
    CaseScope scope{"abd_dynamics_kinetic"};
    auto&     c = scope.counter;

    // ---- the beta convention, as *doubles* (a float literal would fail here)
    T62_CHECK(c, BDF2Kinetic::beta == 4.0 / 9.0);
    T62_CHECK(c, BDF2Kinetic::inv_beta == 9.0 / 4.0);
    T62_CHECK(c, BDF2Kinetic::inv_beta == 1.0 / BDF2Kinetic::beta);
    static_assert(std::is_same_v<decltype(BDF2Kinetic::beta), const double>);
    static_assert(std::is_same_v<gg::Float, double>);

    // ---- three hand-written bodies with hand-written rest triads
    const int                  n = 3;
    const std::vector<double>  m   = {2.0, 0.375, 6.25};
    const std::vector<Vector3> mx  = {Vector3{0.5, -0.25, 1.0},
                                      Vector3{-0.75, 0.125, 0.0},
                                      Vector3{2.0, -1.5, 0.25}};
    const std::vector<Matrix3x3> mxx = {m3(0.5, 0.125, -0.25, 0.125, 1.5, 0.75, -0.25, 0.75, 0.25),
                                       m3(0.25, 0.0, 0.125, 0.0, 0.125, -0.5, 0.125, -0.5, 2.0),
                                       m3(3.0, 1.25, 0.0, 1.25, 0.5, -0.75, 0.0, -0.75, 1.75)};

    const std::vector<Vector12> q       = {make_q(Matrix3x3::Identity(), Vector3{0.25, -0.5, 1.0}),
                                           make_q(Matrix3x3::Identity(), Vector3{-2.0, 0.75, 0.0}),
                                           make_q(Matrix3x3::Identity(), Vector3{0.0, 0.0, -3.0})};
    const std::vector<Vector12> q_tilde = {make_q(Matrix3x3::Identity(), Vector3{0.0, 0.0, 0.0}),
                                           make_q(Matrix3x3::Identity(), Vector3{1.0, 1.0, 1.0}),
                                           make_q(Matrix3x3::Identity(), Vector3{0.5, -0.25, 0.125})};
    // body 1 is fixed; body 2 is external-kinetic; body 0 is free
    const std::vector<IndexT> is_fixed    = {0, 1, 0};
    const std::vector<IndexT> ext_kinetic = {0, 0, 1};

    double worst_k = 0.0, worst_g = 0.0, worst_h = 0.0;
    double k_bdf1 = 0.0, k_bdf2 = 0.0;
    double grad_norm_free = 0.0;

    for(int i = 0; i < n; ++i)
    {
        const auto          mass = to_generic(to_abd_dyadic_mass(m[i], mx[i], mxx[i]));
        const Matrix12x12   M    = mass_matrix_oracle(m[i], mx[i], mxx[i]);
        const Vector12      dq   = q[i] - q_tilde[i];
        const Vector12      Mdq  = M * dq;
        const bool          fx   = is_fixed[i] != 0;
        const bool          ext  = ext_kinetic[i] != 0;
        const bool          dead = fx || ext;

        // K = (is_fixed || ext_kinetic) ? 0 : 0.5 dq . (M dq)
        const double e1 = bdf1_kinetic_detail::energy(
            lv12d(q[i]), lv12d(q_tilde[i]), mass, is_fixed[i], ext_kinetic[i]);
        const double expected_e1 = dead ? 0.0 : 0.5 * dq.dot(Mdq);
        worst_k                  = std::max(worst_k, rel_err(e1, expected_e1));

        // K = (is_fixed || ext_kinetic) ? 0 : (0.5 inv_beta) dq . (M dq)
        const double e2 = bdf2_kinetic_detail::energy(
            lv12d(q[i]), lv12d(q_tilde[i]), mass, BDF2Kinetic::inv_beta, is_fixed[i], ext_kinetic[i]);
        const double expected_e2 = dead ? 0.0 : 0.5 * BDF2Kinetic::inv_beta * dq.dot(Mdq);
        worst_k                  = std::max(worst_k, rel_err(e2, expected_e2));

        // G = is_fixed ? 0 : M dq          (BDF1)
        // G = is_fixed ? 0 : inv_beta M dq (BDF2)   -- *not* guarded by ext_kinetic
        auto g1 = bdf1_kinetic_detail::gradient(
            lv12d(q[i]), lv12d(q_tilde[i]), mass, is_fixed[i]);
        auto g2 = bdf2_kinetic_detail::gradient(
            lv12d(q[i]), lv12d(q_tilde[i]), mass, BDF2Kinetic::inv_beta, is_fixed[i]);
        // NOTE: written as two statements -- a ternary mixing `Vector12::Zero()`
        // (an Eigen nullary expression) with a matrix product has *no* common type
        // (measured C2446, the T6.1a rule 10 trap)
        Vector12 expected_g1 = Vector12::Zero();
        Vector12 expected_g2 = Vector12::Zero();
        if(!fx)
        {
            expected_g1 = Mdq;
            expected_g2 = BDF2Kinetic::inv_beta * Mdq;
        }
        for(int a = 0; a < 12; ++a)
        {
            worst_g = std::max(worst_g, rel_err(g1(a), expected_g1(a)));
            worst_g = std::max(worst_g, rel_err(g2(a), expected_g2(a)));
        }
        if(ext)
        {
            // the external-kinetic body's *gradient* is NOT zero (only K is guarded):
            // the reference only branches on is_fixed in the gradient stage
            double norm = 0.0;
            for(int a = 0; a < 12; ++a)
                norm = std::max(norm, std::abs(g1(a) - Mdq(a)));
            T62_CHECK(c, norm <= 1e-12);
            T62_CHECK(c, Mdq.norm() > 0.0);
        }
        if(!dead)
            grad_norm_free = std::max(grad_norm_free, Mdq.norm());

        // H = M (BDF1) / inv_beta M (BDF2) -- and the Hessian has **no** `is_fixed`
        // guard in the reference: a fixed body still receives its mass block (the
        // implicit system's mass regularisation).  The T6.2 brief's "fixed => H = 0"
        // wording contradicts the reference's kernels and the reference wins.
        auto       h1 = bdf1_kinetic_detail::hessian(mass);
        auto       h2 = bdf2_kinetic_detail::hessian(mass, BDF2Kinetic::inv_beta);
        const auto expected_h1 = M;
        const auto expected_h2 = BDF2Kinetic::inv_beta * M;
        for(int r = 0; r < 12; ++r)
            for(int cc = 0; cc < 12; ++cc)
            {
                worst_h = std::max(worst_h, rel_err(h1(r, cc), expected_h1(r, cc)));
                worst_h = std::max(worst_h, rel_err(h2(r, cc), expected_h2(r, cc)));
            }
        if(dead)
        {
            // explicitly pin the asymmetry for the two "dead" bodies: H == M (not 0);
            // compared entry-wise, because an `LMat` view's `diagonal()` is a *view*
            // and views have no member `norm()` (cookbook §7.21 rule 11))
            double diag_dev = 0.0;
            double frob2    = 0.0;
            for(int a = 0; a < 12; ++a)
                diag_dev = std::max(diag_dev, std::abs(h1(a, a) - M(a, a)));
            for(int r0 = 0; r0 < 12; ++r0)
                for(int c0 = 0; c0 < 12; ++c0)
                    frob2 += h1(r0, c0) * h1(r0, c0);
            T62_CHECK(c, diag_dev <= 1e-15);
            T62_CHECK(c, rel_err(h1(0, 0), m[i]) <= 1e-15);
            T62_CHECK(c, rel_err(h1(3 + 0, 3 + 0), mxx[i](0, 0)) <= 1e-15);
            T62_CHECK(c, frob2 > 0.0);
        }
        // the mass operator is *not* diagonal (unlike the FEM `m I`): the (0, 3+i)
        // band is `m x̄` and the (1+i, 1+i) block is `m x̄ ⊗ x̄`
        T62_CHECK(c, rel_err(h1(0, 3), mx[i](0)) <= 1e-15);
        T62_CHECK(c, rel_err(h1(3, 0), mx[i](0)) <= 1e-15);
        T62_CHECK(c, rel_err(h1(3, 3), mxx[i](0, 0)) <= 1e-15);
        T62_CHECK(c, h1(0, 1) == 0.0);
        T62_CHECK(c, h1(0, 6) == 0.0);

        k_bdf1 += e1;
        k_bdf2 += e2;
    }

    // ---- the same total by hand over all three bodies
    double K1 = 0.0, K2 = 0.0;
    for(int i = 0; i < n; ++i)
    {
        if(is_fixed[i] || ext_kinetic[i])
            continue;
        const auto        M   = mass_matrix_oracle(m[i], mx[i], mxx[i]);
        const Vector12    dq  = q[i] - q_tilde[i];
        K1 += 0.5 * dq.dot(M * dq);
        K2 += 0.5 * BDF2Kinetic::inv_beta * dq.dot(M * dq);
    }
    T62_CHECK(c, rel_err(k_bdf1, K1) <= 1e-12);
    T62_CHECK(c, rel_err(k_bdf2, K2) <= 1e-12);
    // the BDF2 kinetic energy is inv_beta times the BDF1 one, which pins the scaling
    // independently of the per-body values
    T62_CHECK(c, rel_err(k_bdf2, BDF2Kinetic::inv_beta * K1) <= 1e-12);
    T62_CHECK(c, K1 > 0.0);
    T62_CHECK(c, grad_norm_free > 0.0);

    std::printf("[t62] kinetic oracle (n=%d): K rel %.3e  G rel %.3e  H rel %.3e  "
                "(K_bdf1 %.6f, K_bdf2 %.6f, bdf2/bdf1 %.6f)\n",
                n,
                worst_k,
                worst_g,
                worst_h,
                k_bdf1,
                k_bdf2,
                k_bdf2 / k_bdf1);

    T62_CHECK(c, worst_k <= 1e-12);
    T62_CHECK(c, worst_g <= 1e-12);
    T62_CHECK(c, worst_h <= 1e-12);
}

// ===========================================================================
//  3) the BDF1/BDF2 predict/update pairs over device buffers
// ===========================================================================
TEST_CASE("abd_dynamics_integrators", "[luisa][abd_dynamics]")
{
    test_runtime();
    CaseScope scope{"abd_dynamics_integrators"};
    auto&     c      = scope.counter;
    auto&     stream = gg::frame_stream();

    const int n = 4;
    // body 0 dynamic, body 1 static (not dynamic), body 2 fixed, body 3 dynamic
    const std::vector<IndexT> is_fixed_h   = {0, 0, 1, 0};
    const std::vector<IndexT> is_dynamic_h = {1, 0, 1, 1};

    std::vector<Vector12> qs_h, qv_h, g_h, a_h;
    for(int i = 0; i < n; ++i)
    {
        Vector12 q = Vector12::Zero();
        q.segment<3>(0) = Vector3{0.25 * (i + 1), -0.5, 0.125};
        for(int a = 0; a < 3; ++a)
            q(3 + 3 * a + a) = 1.0 + 0.1 * i;
        qs_h.push_back(q);

        Vector12 v = Vector12::Zero();
        v.segment<3>(0) = Vector3{0.5 * (i + 1), 1.0, -0.25};
        for(int a = 0; a < 3; ++a)
            v(3 + 3 * a + a) = 0.05 * (i + 1);
        qv_h.push_back(v);

        Vector12 g = Vector12::Zero();
        g.segment<3>(0) = Vector3{0.1 * i, -9.8, 0.3};
        g_h.push_back(g);

        Vector12 a = Vector12::Zero();
        a.segment<3>(0) = Vector3{-0.1, 0.2, 0.05 * i};
        a_h.push_back(a);
    }
    const double dt = 0.02;
    const auto   nn = static_cast<uint32_t>(n);

    // =======================================================================
    //  BDF1
    // =======================================================================
    {
        lud::LDeviceBuffer<IndexT>       d_is_fixed, d_is_dynamic;
        lud::LDeviceBuffer<gg::LVector12> d_qs, d_qv, d_g, d_a, d_q_prevs, d_q_tildes;
        copy_index_to_device(d_is_fixed, is_fixed_h);
        copy_index_to_device(d_is_dynamic, is_dynamic_h);
        copy_vec12_to_device(d_qs, qs_h);
        copy_vec12_to_device(d_qv, qv_h);
        copy_vec12_to_device(d_g, g_h);
        copy_vec12_to_device(d_a, a_h);
        // sentinels: a fixed body must keep q_prev, never the sentinel
        copy_vec12_to_device(d_q_prevs, sentinel12(n, 7.0));
        copy_vec12_to_device(d_q_tildes, sentinel12(n, -7.0));

        {
            auto& k = bdf1_predict_probe_kernel();
            stream.record(k(d_is_fixed.cview(),
                            d_is_dynamic.cview(),
                            d_q_prevs.view(),
                            d_qs.cview(),
                            d_qv.cview(),
                            d_q_tildes.view(),
                            d_g.cview(),
                            d_a.cview(),
                            dt,
                            nn)
                              .dispatch(nn));
        }
        auto q_prevs  = read_vec12(d_q_prevs.cview());
        auto q_tildes = read_vec12(d_q_tildes.cview());
        REQUIRE(q_prevs.size() == static_cast<size_t>(n));
        REQUIRE(q_tildes.size() == static_cast<size_t>(n));

        double worst_prev = 0.0, worst_tilde = 0.0, worst_law = 0.0;
        for(int i = 0; i < n; ++i)
        {
            // 0) every body records q_prev = qs
            for(int a = 0; a < 12; ++a)
                worst_prev = std::max(worst_prev, rel_err(q_prevs[i](a), qs_h[i](a)));

            // the reference's branch chain, hand-derived:
            //   q_tilde = q_prev; if !fixed: += (g + a) dt^2;
            //                     if !fixed && dynamic: += q_v dt
            Vector12 expected = qs_h[i];
            if(!is_fixed_h[i])
            {
                expected += (g_h[i] + a_h[i]) * (dt * dt);
                if(is_dynamic_h[i])
                    expected += qv_h[i] * dt;
            }
            for(int a = 0; a < 12; ++a)
                worst_tilde = std::max(worst_tilde, rel_err(q_tildes[i](a), expected(a)));

            // the *scalar law* the brief names, recomputed from the ported output:
            //   (q_tilde - q_prev) / dt^2 == g + a             (static)
            //                             == q_v/dt + g + a    (dynamic)
            //                             == 0                 (fixed)
            const Vector12 delta = (q_tildes[i] - q_prevs[i]) / (dt * dt);
            Vector12       law;
            if(is_fixed_h[i])
                law = Vector12::Zero();
            else if(is_dynamic_h[i])
                law = qv_h[i] / dt + g_h[i] + a_h[i];
            else
                law = g_h[i] + a_h[i];
            for(int a = 0; a < 12; ++a)
                worst_law = std::max(worst_law, rel_err(delta(a), law(a)));

            if(is_fixed_h[i])
            {
                // a fixed body's q_tilde is *exactly* q_prev (no dt^2 g term)
                for(int a = 0; a < 12; ++a)
                    T62_CHECK(c, q_tildes[i](a) == q_prevs[i](a));
            }
        }

        // ---- update_state: q_v = (q - q_prev) / dt, with q modified in the buffer
        auto qs_after_h = qs_h;
        for(int i = 0; i < n; ++i)
            qs_after_h[i] += Vector12::Constant(0.25 * (i + 1));
        copy_vec12_to_device(d_qs, qs_after_h);
        {
            auto& k = bdf1_update_probe_kernel();
            stream.record(k(d_qs.cview(), d_qv.view(), d_q_prevs.cview(), dt, nn).dispatch(nn));
        }
        auto qv_new = read_vec12(d_qv.cview());

        double worst_v = 0.0;
        for(int i = 0; i < n; ++i)
        {
            // the reference's update kernel is *unconditional*: even a fixed (or
            // static) body's velocity is overwritten by (q - q_prev)/dt
            const Vector12 expected = (qs_after_h[i] - qs_h[i]) * (1.0 / dt);
            for(int a = 0; a < 12; ++a)
                worst_v = std::max(worst_v, rel_err(qv_new[i](a), expected(a)));
        }

        std::printf("[t62] bdf1 integrator: q_prev rel %.3e  q_tilde rel %.3e  "
                    "scalar law rel %.3e  q_v rel %.3e\n",
                    worst_prev,
                    worst_tilde,
                    worst_law,
                    worst_v);
        T62_CHECK(c, worst_prev == 0.0);
        T62_CHECK(c, worst_tilde <= 1e-12);
        T62_CHECK(c, worst_law <= 1e-12);
        T62_CHECK(c, worst_v <= 1e-12);
    }

    // =======================================================================
    //  BDF2
    // =======================================================================
    std::vector<Vector12> q_n_1_h, qv_n_1_h;
    for(int i = 0; i < n; ++i)
    {
        Vector12 qn1 = qs_h[i];
        qn1(0) -= 0.5;
        qn1(5) += 1.25;
        // the *fixed* body's history is garbage on purpose: the kernel replaces it
        if(is_fixed_h[i])
            qn1 = sentinel12(1, 4.0)[0];
        q_n_1_h.push_back(qn1);

        Vector12 vn1 = qv_h[i];
        vn1(1) *= -1.0;
        if(is_fixed_h[i])
            vn1 = sentinel12(1, -4.0)[0];
        qv_n_1_h.push_back(vn1);
    }
    {
        lud::LDeviceBuffer<IndexT>        d_is_fixed, d_is_dynamic;
        lud::LDeviceBuffer<gg::LVector12> d_qs, d_q_ns, d_q_n_1s, d_q_v_ns, d_q_v_n_1s, d_q_tildes, d_g;
        copy_index_to_device(d_is_fixed, is_fixed_h);
        copy_index_to_device(d_is_dynamic, is_dynamic_h);
        copy_vec12_to_device(d_qs, qs_h);
        copy_vec12_to_device(d_q_n_1s, q_n_1_h);
        copy_vec12_to_device(d_q_v_ns, qv_h);
        copy_vec12_to_device(d_q_v_n_1s, qv_n_1_h);
        copy_vec12_to_device(d_g, g_h);
        copy_vec12_to_device(d_q_ns, sentinel12(n, 7.0));
        copy_vec12_to_device(d_q_tildes, sentinel12(n, -7.0));

        {
            auto& k = bdf2_predict_probe_kernel();
            stream.record(k(d_is_fixed.cview(),
                            d_is_dynamic.cview(),
                            d_qs.cview(),
                            d_q_ns.view(),
                            d_q_n_1s.cview(),
                            d_q_v_ns.cview(),
                            d_q_v_n_1s.cview(),
                            d_q_tildes.view(),
                            d_g.cview(),
                            dt,
                            nn)
                              .dispatch(nn));
        }
        auto q_ns     = read_vec12(d_q_ns.cview());
        auto q_tildes = read_vec12(d_q_tildes.cview());

        double worst_n = 0.0, worst_t = 0.0;
        for(int i = 0; i < n; ++i)
        {
            // q_n is recorded into q_ns (== q_prevs) for the current step
            for(int a = 0; a < 12; ++a)
                worst_n = std::max(worst_n, rel_err(q_ns[i](a), qs_h[i](a)));

            // hand-derived from the reference's kernel:
            Vector12 q_n   = qs_h[i];
            Vector12 q_n_1 = q_n_1_h[i];
            Vector12 v_n   = qv_h[i];
            Vector12 v_n_1 = qv_n_1_h[i];
            Vector12 gtot  = g_h[i];
            if(is_fixed_h[i])
            {
                q_n_1 = q_n;
                v_n.setZero();
                v_n_1.setZero();
                gtot.setZero();
            }
            else if(!is_dynamic_h[i])
            {
                q_n_1 = q_n;
                v_n.setZero();
                v_n_1.setZero();
            }
            const Vector12 expected = (4.0 / 9.0) * (dt * dt) * gtot + (4.0 / 3.0) * q_n
                                      - (1.0 / 3.0) * q_n_1
                                      + (2.0 / 9.0) * dt * (4.0 * v_n - v_n_1);
            for(int a = 0; a < 12; ++a)
                worst_t = std::max(worst_t, rel_err(q_tildes[i](a), expected(a)));

            if(is_fixed_h[i])
            {
                // the fixed branch removes every influence, and its garbage history
                // must not show up: q_tilde == q_n exactly
                for(int a = 0; a < 12; ++a)
                    T62_CHECK(c, q_tildes[i](a) == q_ns[i](a));
            }
            else if(!is_dynamic_h[i])
            {
                // the static branch zeroes both velocity histories and sets
                // q_{n-1} := q_n, so (q_tilde - q_n) / dt^2 == 4/9 g exactly
                const Vector12 d = (q_tildes[i] - q_ns[i]) / (dt * dt);
                for(int a = 0; a < 12; ++a)
                    T62_CHECK(c, rel_err(d(a), (4.0 / 9.0) * g_h[i](a)) <= 1e-12);
            }
        }

        // ---- update_state, with the three history writes' order load-bearing
        auto qs_after_h = qs_h;
        for(int i = 0; i < n; ++i)
            qs_after_h[i] += Vector12::Constant(0.25 * (i + 1));
        copy_vec12_to_device(d_qs, qs_after_h);
        {
            auto& k = bdf2_update_probe_kernel();
            stream.record(k(d_qs.cview(),
                            d_q_ns.view(),
                            d_q_n_1s.view(),
                            d_q_v_ns.view(),
                            d_q_v_n_1s.view(),
                            dt,
                            nn)
                              .dispatch(nn));
        }
        auto q_v_ns_new   = read_vec12(d_q_v_ns.cview());
        auto q_v_n_1s_new = read_vec12(d_q_v_n_1s.cview());
        auto q_n_1s_new   = read_vec12(d_q_n_1s.cview());

        double worst_v = 0.0, worst_hist = 0.0;
        for(int i = 0; i < n; ++i)
        {
            const Vector12 expected_v =
                (0.5 / dt) * (3.0 * qs_after_h[i] - 4.0 * qs_h[i] + q_n_1_h[i]);
            for(int a = 0; a < 12; ++a)
            {
                worst_v = std::max(worst_v, rel_err(q_v_ns_new[i](a), expected_v(a)));
                // the history: q_v_{n-1} := the *old* q_v_n, q_{n-1} := the old q_n
                worst_hist = std::max(worst_hist, rel_err(q_v_n_1s_new[i](a), qv_h[i](a)));
                worst_hist = std::max(worst_hist, rel_err(q_n_1s_new[i](a), qs_h[i](a)));
            }
            // the *old* q_n (not the new one) is what lands in q_{n-1}
            T62_CHECK(c, rel_err((q_n_1s_new[i] - qs_h[i]).norm(), 0.0) <= 1e-12);
            for(int a = 0; a < 12; ++a)
                T62_CHECK(c, q_v_n_1s_new[i](a) == qv_h[i](a));
        }

        std::printf("[t62] bdf2 integrator: q_n rel %.3e  q_tilde rel %.3e  "
                    "q_v rel %.3e  history rel %.3e\n",
                    worst_n,
                    worst_t,
                    worst_v,
                    worst_hist);
        T62_CHECK(c, worst_n == 0.0);
        T62_CHECK(c, worst_t <= 1e-12);
        T62_CHECK(c, worst_v <= 1e-12);
        T62_CHECK(c, worst_hist <= 1e-12);
    }
}

// ===========================================================================
//  4) the plan's headline: at steady state a BDF2 advance matches BDF1
// ===========================================================================
TEST_CASE("abd_dynamics_steady_state", "[luisa][abd_dynamics]")
{
    test_runtime();
    CaseScope scope{"abd_dynamics_steady_state"};
    auto&     c      = scope.counter;
    auto&     stream = gg::frame_stream();

    const double dt = 0.02;
    const double h  = dt;

    // ---- the steady state: constant velocity V, consistent history -------------
    //   Q            = the affine state at t_n   (a *rotation* + a translation)
    //   V            = the 12-dof velocity       (pure translation here, so the
    //                    steady-state transform's rotational block stays exactly A)
    //   history      = q_{n-1} = Q - V h, q_v_n = q_v_{n-1} = V
    //   next state   = Q + V h                (the analytic continuation)
    // BDF1 (dynamic): q_tilde = Q + V h + g dt^2
    // BDF2          : q_tilde = 4/9 h^2 g + 4/3 Q - 1/3 (Q - Vh) + 2/9 h (4V - V)
    //                        = Q + V h + 4/9 h^2 g
    // so with g = 0 the two *agree*, and with g != 0 they differ by exactly
    // (1 - 4/9) h^2 g -- which is what pins the beta = 4/9 convention against the
    // BDF1 predictor's coefficient 1.
    const Matrix3x3 A =
        Eigen::AngleAxisd(0.7, Vector3{1.0, 2.0, -0.5}.normalized()).toRotationMatrix();
    const Vector3  t{0.25, -1.5, 2.0};
    const Vector3  v{0.5, -0.75, 1.25};

    const Vector12 Q = make_q(A, t);
    const Vector12 V = make_q(Matrix3x3::Zero(), v);  // pure translation, no rotation
    const Vector12 Q_prev = Q - V * h;                // the consistent history
    const Vector12 Q_next = Q + V * h;                // the analytic continuation

    Vector12 G12             = Vector12::Zero();
    G12.segment<3>(0)        = Vector3{0.0, -9.8, 0.0};

    const auto nn = 1u;
    std::vector<IndexT> is_fixed_h{0};
    std::vector<IndexT> is_dynamic_h{1};

    lud::LDeviceBuffer<IndexT>        d_is_fixed, d_is_dynamic;
    lud::LDeviceBuffer<gg::LVector12> d_qs, d_q_prevs, d_qv, d_g, d_a, d_q_tildes;
    lud::LDeviceBuffer<gg::LVector12> d_q_ns, d_q_n_1s, d_q_v_ns, d_q_v_n_1s;
    copy_index_to_device(d_is_fixed, is_fixed_h);
    copy_index_to_device(d_is_dynamic, is_dynamic_h);
    copy_vec12_to_device(d_a, std::vector<Vector12>{Vector12::Zero()});

    double worst_bdf1 = 0.0, worst_bdf2 = 0.0, worst_gap = 0.0, worst_v_bdf1 = 0.0,
           worst_v_bdf2 = 0.0;

    for(int g_case = 0; g_case < 2; ++g_case)
    {
        // case 0: g = 0 -> the two integrators must agree (and match the analytic
        //         steady state); case 1: g != 0 -> they differ by exactly 5/9 h^2 g
        const Vector12 G = (g_case == 0) ? Vector12::Zero() : G12;

        // ---------------- BDF1 ----------------
        Vector12 q_tilde_bdf1, q_v_bdf1;
        {
            copy_vec12_to_device(d_qs, std::vector<Vector12>{Q});
            copy_vec12_to_device(d_q_prevs, std::vector<Vector12>{Q_prev});
            copy_vec12_to_device(d_qv, std::vector<Vector12>{V});
            copy_vec12_to_device(d_g, std::vector<Vector12>{G});
            copy_vec12_to_device(d_q_tildes, sentinel12(1, -7.0));
            auto& k = bdf1_predict_probe_kernel();
            stream.record(k(d_is_fixed.cview(),
                            d_is_dynamic.cview(),
                            d_q_prevs.view(),
                            d_qs.cview(),
                            d_qv.cview(),
                            d_q_tildes.view(),
                            d_g.cview(),
                            d_a.cview(),
                            dt,
                            nn)
                              .dispatch(nn));
            q_tilde_bdf1 = read_vec12(d_q_tildes.cview()).front();
            // the shipped BDF1 law: q_tilde = q_prev + (g + f_ext_acc) dt^2, then
            // += q_v dt -- and the kernel's `q_prev` is the value it has *just*
            // written (`qs`), not the input history:
            const Vector12 expected = Q + (G * dt) * dt + V * dt;
            for(int a = 0; a < 12; ++a)
                worst_bdf1 = std::max(worst_bdf1, rel_err(q_tilde_bdf1(a), expected(a)));

            // update on the steady-state continuation: q_v == V
            copy_vec12_to_device(d_qs, std::vector<Vector12>{Q_next});
            auto& ku = bdf1_update_probe_kernel();
            stream.record(ku(d_qs.cview(), d_qv.view(), d_q_prevs.cview(), dt, nn).dispatch(nn));
            q_v_bdf1 = read_vec12(d_qv.cview()).front();
            for(int a = 0; a < 12; ++a)
                worst_v_bdf1 = std::max(worst_v_bdf1, rel_err(q_v_bdf1(a), V(a)));
        }

        // ---------------- BDF2 ----------------
        Vector12 q_tilde_bdf2, q_v_bdf2;
        {
            copy_vec12_to_device(d_qs, std::vector<Vector12>{Q});
            copy_vec12_to_device(d_q_n_1s, std::vector<Vector12>{Q_prev});
            copy_vec12_to_device(d_q_v_ns, std::vector<Vector12>{V});
            copy_vec12_to_device(d_q_v_n_1s, std::vector<Vector12>{V});
            copy_vec12_to_device(d_g, std::vector<Vector12>{G});
            copy_vec12_to_device(d_q_ns, sentinel12(1, 7.0));
            copy_vec12_to_device(d_q_tildes, sentinel12(1, -7.0));
            auto& k = bdf2_predict_probe_kernel();
            stream.record(k(d_is_fixed.cview(),
                            d_is_dynamic.cview(),
                            d_qs.cview(),
                            d_q_ns.view(),
                            d_q_n_1s.cview(),
                            d_q_v_ns.cview(),
                            d_q_v_n_1s.cview(),
                            d_q_tildes.view(),
                            d_g.cview(),
                            dt,
                            nn)
                              .dispatch(nn));
            q_tilde_bdf2 = read_vec12(d_q_tildes.cview()).front();
            // the shipped BDF2 law *evaluated by hand* on the steady-state history
            // (the same statements as sym::abd_bdf2::compute_q_tilde):
            {
                const double  c_g = (4.0 / 9.0) * h * h;
                const double  c_v = (2.0 / 9.0) * h;
                const Vector12 expected =
                    c_g * G + (4.0 / 3.0) * Q - (1.0 / 3.0) * Q_prev + c_v * (4.0 * V - V);
                for(int a = 0; a < 12; ++a)
                    worst_bdf2 = std::max(worst_bdf2, rel_err(q_tilde_bdf2(a), expected(a)));
                // ... and the *analytic* steady-state value the brief asks for:
                // 4/3 Q - 1/3 (Q - Vh) = Q + 1/3 Vh and 2/9 h (4V - V) = 2/3 Vh add up
                // to Q + V h, plus the 4/9 h^2 g term
                const Vector12 analytic = Q + V * h + (4.0 / 9.0) * (h * h) * G;
                double analytic_err = 0.0;
                for(int a = 0; a < 12; ++a)
                    analytic_err = std::max(analytic_err, rel_err(expected(a), analytic(a)));
                T62_CHECK(c, analytic_err <= 1e-15);
            }

            // update: 1/(2h) (3 (Q + Vh) - 4 Q + (Q - Vh)) == V
            copy_vec12_to_device(d_qs, std::vector<Vector12>{Q_next});
            auto& ku = bdf2_update_probe_kernel();
            stream.record(ku(d_qs.cview(),
                             d_q_ns.view(),
                             d_q_n_1s.view(),
                             d_q_v_ns.view(),
                             d_q_v_n_1s.view(),
                             dt,
                             nn)
                              .dispatch(nn));
            q_v_bdf2 = read_vec12(d_q_v_ns.cview()).front();
            for(int a = 0; a < 12; ++a)
                worst_v_bdf2 = std::max(worst_v_bdf2, rel_err(q_v_bdf2(a), V(a)));
        }

        // ---- the headline: at steady state the BDF2 advance matches the BDF1 advance
        for(int a = 0; a < 12; ++a)
        {
            worst_gap = std::max(worst_gap, std::abs(q_tilde_bdf2(a) - q_tilde_bdf1(a)));
            T62_CHECK(c, rel_err(q_v_bdf2(a), q_v_bdf1(a)) <= 1e-14);
        }
        T62_CHECK(c, rel_err((q_v_bdf2 - q_v_bdf1).norm(), 0.0) <= 1e-14);

        if(g_case == 0)
        {
            // g == 0: q_tilde_BDF2 == q_tilde_BDF1 == Q + V h  (bitwise-close: the two
            // sides evaluate different expressions, so the bound is round-off)
            for(int a = 0; a < 12; ++a)
            {
                T62_CHECK(c, rel_err(q_tilde_bdf2(a), Q_next(a)) <= 1e-14);
                T62_CHECK(c, rel_err(q_tilde_bdf1(a), Q_next(a)) <= 1e-14);
                // one ULP: the two sides evaluate differently associated
                // (mathematically identical) expressions
                T62_CHECK(c, std::abs(q_tilde_bdf2(a) - q_tilde_bdf1(a)) <= 1e-14);
            }
            // ... and V == 0 would make the whole check vacuous: the velocity is real
            T62_CHECK(c, V.norm() > 1.0);
        }
        else
        {
            // g != 0: the two *must* differ, by exactly (1 - 4/9) h^2 g = 5/9 h^2 g --
            // which is what pins the beta = 4/9 convention against BDF1's coefficient 1
            double worst_beta = 0.0;
            for(int a = 0; a < 12; ++a)
            {
                const double expected = (1.0 - 4.0 / 9.0) * (h * h) * G(a);
                worst_beta = std::max(
                    worst_beta, std::abs((q_tilde_bdf1(a) - q_tilde_bdf2(a)) - expected));
                if(G(a) == 0.0)
                    T62_CHECK(c, std::abs(q_tilde_bdf1(a) - q_tilde_bdf2(a)) <= 1e-15);
            }
            T62_CHECK(c, worst_beta <= 1e-15);
            // ... and the gap is not a coincidence: the gravity amplitude is real
            T62_CHECK(c, std::abs(q_tilde_bdf1(1) - q_tilde_bdf2(1)) > 1e-6);
            std::printf("[t62] steady state (g != 0): (1-4/9) h^2 g check worst %.3e\n",
                        worst_beta);
        }
    }

    // ---- the ABD transform invariants ------------------------------------------
    // The steady-state advance only adds the *translation* V h to Q, so the rotational
    // block of the resulting transform is exactly A: q_to_transform's 3x3 part stays
    // orthonormal to round-off, and `q_v_to_transform_v` keeps the last row zero.
    {
        std::vector<Vector12> steady{Q + V * h};
        const Matrix3x3       A_after = A_oracle(steady[0]);
        double                ortho   = 0.0;
        // NOTE: an explicit type for every Eigen product -- `const auto X = <temp>
        // .transpose() * <temp>` binds an *expression* that references temporaries
        // (the T6.1a rule 10 trap, measured 144 nans there)
        const Matrix3x3 AtA = Matrix3x3{A_after.transpose() * A_after};
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                ortho = std::max(ortho, std::abs(AtA(i, j) - (i == j ? 1.0 : 0.0)));
        // and the rotational block must be *bitwise* the steady state's A
        double rot_diff = 0.0;
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                rot_diff = std::max(rot_diff, std::abs(A_after(i, j) - A(i, j)));

        const Matrix4x4 T_after = gg::q_to_transform(steady[0]);
        const Matrix4x4 T_v     = gg::q_v_to_transform_v(V);
        const Matrix3x3 T3      = Matrix3x3{T_after.block<3, 3>(0, 0)};
        const Matrix3x3 T3tT3   = Matrix3x3{T3.transpose() * T3};
        double          a_after_ortho4 = 0.0;
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                a_after_ortho4 = std::max(a_after_ortho4,
                                          std::abs(T3tT3(i, j) - (i == j ? 1.0 : 0.0)));
        double last_row = 0.0;
        for(int j = 0; j < 4; ++j)
            last_row = std::max(last_row, std::abs(T_v(3, j)));
        // the transform round trip (frontend <-> q) is exact for this q
        const Vector12 rt = gg::transform_to_q(T_after);
        double         rt_diff = 0.0;
        for(int a = 0; a < 12; ++a)
            rt_diff = std::max(rt_diff, std::abs(rt(a) - steady[0](a)));

        std::printf("[t62] steady state transform: |A^T A - I| %.3e  |A_after - A| %.3e  "
                    "|q_to_transform ortho| %.3e  |last row of q_v_to_transform_v| %.3e  "
                    "|transform_to_q round trip| %.3e\n",
                    ortho,
                    rot_diff,
                    a_after_ortho4,
                    last_row,
                    rt_diff);

        T62_CHECK(c, ortho <= 1e-15);
        T62_CHECK(c, rot_diff == 0.0);
        T62_CHECK(c, a_after_ortho4 <= 1e-15);
        T62_CHECK(c, last_row == 0.0);
        T62_CHECK(c, rt_diff == 0.0);
        // the transform's translation is the analytic t + v h
        for(int a = 0; a < 3; ++a)
            T62_CHECK(c, rel_err(T_after(a, 3), t(a) + v(a) * h) <= 1e-15);
        T62_CHECK(c, std::abs(A.determinant() - 1.0) <= 1e-15);
    }

    std::printf("[t62] steady state: bdf1 rel %.3e  bdf2 rel %.3e  max|bdf2-bdf1| %.3e  "
                "(q_v: bdf1 rel %.3e, bdf2 rel %.3e)\n",
                worst_bdf1,
                worst_bdf2,
                worst_gap,
                worst_v_bdf1,
                worst_v_bdf2);

    T62_CHECK(c, worst_bdf1 <= 1e-14);
    T62_CHECK(c, worst_bdf2 <= 1e-14);
    T62_CHECK(c, worst_v_bdf1 <= 1e-14);
    T62_CHECK(c, worst_v_bdf2 <= 1e-14);
    // with the *consistent* history the two agree in case 0 (g = 0) and differ by
    // 5/9 h^2 g in case 1; worst_gap is the larger of the two, so it is not a bound
    T62_CHECK(c, worst_gap >= (1.0 - 4.0 / 9.0) * (h * h) * 9.8 * 0.99);
}

// ===========================================================================
//  5) a real scene through the public API (both integrators, the frontend
//     read-back and ABDBDF2State's dump/recover round trip)
// ===========================================================================
TEST_CASE("abd_dynamics_scene", "[luisa][abd_dynamics]")
{
    test_runtime();
    CaseScope scope{"abd_dynamics_scene"};
    auto&     c = scope.counter;

    const double L     = 0.5;
    const double rho   = 1000.0;
    const double kappa = 1.0e8;
    const double dt    = 0.02;
    const Vector3 g{0.0, -9.8, 0.0};

    // ---- the hand-derived 12-dof gravity (cookbook §7.21 rule 16) --------------
    // The dynamics stores `M^-1 G` with `G = [m g; g_x m x̄; g_y m x̄; g_z m x̄]`, i.e.
    // the *generalized acceleration*: a rigid body's rotational block vanishes and its
    // translation block is exactly `g`.
    const auto     o = box_moments(L, rho);
    const Matrix12x12 M = mass_matrix_oracle(o.m, o.mx, o.mxx);
    Vector12       G12  = Vector12::Zero();
    G12.segment<3>(0)   = o.m * g;
    G12.segment<3>(3)   = g(0) * o.mx;
    G12.segment<3>(6)   = g(1) * o.mx;
    G12.segment<3>(9)   = g(2) * o.mx;
    const Vector12 g12  = M.inverse() * G12;
    for(int a = 3; a < 12; ++a)
        T62_CHECK(c, std::abs(g12(a)) <= 1e-9);
    for(int a = 0; a < 3; ++a)
        T62_CHECK(c, rel_err(g12(a), g(a)) <= 1e-12);

    // ---- the scene's q (a rotation + a translation) and the seeded velocity -----
    const Matrix3x3 A = Eigen::AngleAxisd(0.35, Vector3{0.0, 1.0, 0.0}.normalized())
                            .toRotationMatrix();
    const Vector3    t{0.0, 0.4, 0.0};
    const Vector3    v{0.5, -0.25, 0.75};
    // a velocity with *both* a rotational rate and a translation, so the `q_v dt`
    // term is visible in all 12 components
    const Matrix3x3 V3 = m3(0.0, -0.3, 0.1, 0.3, 0.0, -0.2, -0.1, 0.2, 0.0);
    const Vector12  Q0 = make_q(A, t);
    const Vector12  V0 = make_q(V3, v);

    const auto make_scene = [&](std::string_view integrator)
    {
        auto config = Scene::default_config();
        config["sanity_check"]["enable"] = false;
        config["contact"]["enable"]      = false;
        config["integrator"]["type"]     = std::string{integrator};
        config["dt"]                     = dt;
        config["gravity"]                = g;

        Scene scene{config};
        auto  object = scene.objects().create("t62_scene");
        object->geometries().create(
            make_box_body(L, rho, kappa, 1, {make_transform(A, t)}));
        return scene;
    };

    /// Write the body's transform/velocity through the *public* feature (the only
    /// supported way in), find-or-create so both attributes exist.
    const auto seed = [&](World& world)
    {
        auto feature = world.features().find<AffineBodyStateAccessorFeature>();
        REQUIRE(feature != nullptr);
        auto state_geo = feature->create_geometry();
        auto T         = state_geo.instances().find<Matrix4x4>(builtin::transform);
        if(!T)
            T = state_geo.instances().create<Matrix4x4>(builtin::transform, Matrix4x4::Identity());
        auto V = state_geo.instances().find<Matrix4x4>(builtin::velocity);
        if(!V)
            V = state_geo.instances().create<Matrix4x4>(builtin::velocity, Matrix4x4::Zero());
        REQUIRE(view(*T).size() == 1u);
        view(*T)[0] = make_transform(A, t);
        view(*V)[0] = make_velocity(V3, v);
        feature->copy_from(state_geo);
    };

    /// Read the body's state back through the feature (transform + velocity).
    const auto read_back = [&](World& world)
    {
        auto feature = world.features().find<AffineBodyStateAccessorFeature>();
        REQUIRE(feature != nullptr);
        auto state_geo = feature->create_geometry();
        auto T         = state_geo.instances().find<Matrix4x4>(builtin::transform);
        if(!T)
            T = state_geo.instances().create<Matrix4x4>(builtin::transform, Matrix4x4::Identity());
        auto V = state_geo.instances().find<Matrix4x4>(builtin::velocity);
        if(!V)
            V = state_geo.instances().create<Matrix4x4>(builtin::velocity, Matrix4x4::Zero());
        feature->copy_to(state_geo);
        return std::pair<Matrix4x4, Matrix4x4>{view(*T)[0], view(*V)[0]};
    };

    const auto step_frame = [](World& world)
    {
        world.advance();
        world.sync();
        gg::frame_stream().synchronize();
    };

    std::vector<Vector12> bdf1_tilde;
    std::vector<Vector12> bdf2_tilde;

    // =======================================================================
    //  BDF1
    // =======================================================================
    {
        T62Probe::reset();
        auto scene = make_scene("bdf1");
        Engine engine{"luisa", workspace_path("uipc_luisa_t62_scene_bdf1")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T62Probe::abd != nullptr);
        REQUIRE(T62Probe::abd->is_valid());
        T62_CHECK(c, T62Probe::abd->qs().size() == 1u);

        seed(world);

        // (a) before the frame: the frontend's own write/read round trip
        {
            const auto [T_seed, V_seed] = read_back(world);
            // the *transform* keeps the affine last row (0,0,0,1); the *velocity* is
            // exported through `q_v_to_transform_v`, whose last row is *zeroed*
            const Matrix4x4 V_expected = gg::q_v_to_transform_v(V0);
            for(int a = 0; a < 4; ++a)
                for(int b = 0; b < 4; ++b)
                {
                    T62_CHECK(c, std::abs(T_seed(a, b) - make_transform(A, t)(a, b)) <= 1e-14);
                    T62_CHECK(c, std::abs(V_seed(a, b) - V_expected(a, b)) <= 1e-14);
                }
            for(int b = 0; b < 4; ++b)
                T62_CHECK(c, V_expected(3, b) == 0.0);
            // `copy_from` accepted the 3x3 block + the translation and ignored the last
            // row: `transform_v_to_q_v` *is* `transform_to_q`
            T62_CHECK(c,
                      rel_err((gg::transform_v_to_q_v(make_velocity(V3, v)) - V0).norm(), 0.0)
                          <= 1e-15);
        }
        // ... and the dynamics' own buffers agree with the hand-derived model
        {
            auto qs   = read_vec12(T62Probe::abd->qs());
            auto q_vs = read_vec12(T62Probe::abd->q_vs());
            auto grav = read_vec12(T62Probe::abd->body_gravities());
            auto q_til = read_vec12(T62Probe::abd->q_tildes());
            auto q_prev = read_vec12(T62Probe::abd->q_prevs());
            REQUIRE(qs.size() == 1u);
            for(int a = 0; a < 12; ++a)
            {
                T62_CHECK(c, rel_err(qs[0](a), Q0(a)) <= 1e-14);
                T62_CHECK(c, rel_err(q_vs[0](a), V0(a)) <= 1e-14);
                T62_CHECK(c, rel_err(grav[0](a), g12(a)) <= 1e-12);
                // at init the dynamics sets q_tilde = q_prev = q
                T62_CHECK(c, rel_err(q_til[0](a), Q0(a)) <= 1e-14);
                T62_CHECK(c, rel_err(q_prev[0](a), Q0(a)) <= 1e-14);
            }
            auto is_fixed   = read_index(T62Probe::abd->body_is_fixed());
            auto is_dynamic = read_index(T62Probe::abd->body_is_dynamic());
            auto ext        = read_index(T62Probe::abd->body_external_kinetic());
            REQUIRE(is_fixed.size() == 1u);
            T62_CHECK(c, is_fixed[0] == 0);
            T62_CHECK(c, is_dynamic[0] == 1);
            T62_CHECK(c, ext[0] == 0);
        }

        // (b) one frame
        step_frame(world);
        {
            auto qs     = read_vec12(T62Probe::abd->qs());
            auto q_prev = read_vec12(T62Probe::abd->q_prevs());
            auto q_til  = read_vec12(T62Probe::abd->q_tildes());
            auto q_vs   = read_vec12(T62Probe::abd->q_vs());
            REQUIRE(qs.size() == 1u);

            // the shipped BDF1 kernels' hand-derived result:
            //   q_prev := qs (== the scene's q), q_tilde = q_prev + g12 dt^2 + q_v dt
            //   and the update q_v = (q - q_prev) / dt == 0 (nothing moves q)
            const Vector12 expected_tilde = Q0 + (g12 * dt) * dt + V0 * dt;
            for(int a = 0; a < 12; ++a)
            {
                T62_CHECK(c, rel_err(q_prev[0](a), Q0(a)) <= 1e-15);
                T62_CHECK(c, rel_err(q_til[0](a), expected_tilde(a)) <= 1e-12);
                T62_CHECK(c, std::abs(q_vs[0](a)) <= 1e-15);
            }
            // the scalar law, recomputed from the shipped output
            const Vector12 law = (q_til[0] - q_prev[0]) / (dt * dt);
            for(int a = 0; a < 12; ++a)
                T62_CHECK(c, rel_err(law(a), g12(a) + V0(a) / dt) <= 1e-11);
            bdf1_tilde = q_til;

            // the frontend read-back after the frame: `q` is untouched (no solver
            // writes it back) and the velocity is what `update_state` produced
            const auto [T_now, V_now] = read_back(world);
            for(int a = 0; a < 4; ++a)
                for(int b = 0; b < 4; ++b)
                {
                    T62_CHECK(c, std::abs(T_now(a, b) - make_transform(A, t)(a, b)) <= 1e-14);
                    T62_CHECK(c, std::abs(V_now(a, b)) <= 1e-15);
                }
        }

        // (c) a *second* frame with a velocity re-seeded through the frontend only
        {
            const Matrix3x3 V3b = m3(0.0, 0.4, -0.2, -0.4, 0.0, 0.5, 0.2, -0.5, 0.0);
            const Vector3   vb{1.0, 0.5, -0.75};
            const Vector12  V0b = make_q(V3b, vb);
            auto            feature = world.features().find<AffineBodyStateAccessorFeature>();
            REQUIRE(feature != nullptr);
            auto state_geo = feature->create_geometry();
            auto T = state_geo.instances().find<Matrix4x4>(builtin::transform);
            if(!T)
                T = state_geo.instances().create<Matrix4x4>(builtin::transform, Matrix4x4::Identity());
            auto V = state_geo.instances().find<Matrix4x4>(builtin::velocity);
            if(!V)
                V = state_geo.instances().create<Matrix4x4>(builtin::velocity, Matrix4x4::Zero());
            // NOTE: `copy_from` writes the *whole* state -- the transform included --
            // so the freshly created (identity-valued) transform attribute has to be
            // filled with the state's own transform, or `qs` would be wiped
            view(*T)[0] = make_transform(A, t);
            view(*V)[0] = make_velocity(V3b, vb);
            feature->copy_from(state_geo);

            auto q_vs = read_vec12(T62Probe::abd->q_vs());
            auto qs2  = read_vec12(T62Probe::abd->qs());
            for(int a = 0; a < 12; ++a)
            {
                T62_CHECK(c, rel_err(q_vs[0](a), V0b(a)) <= 1e-14);
                T62_CHECK(c, rel_err(qs2[0](a), Q0(a)) <= 1e-14);
            }

            step_frame(world);
            auto q_til = read_vec12(T62Probe::abd->q_tildes());
            const Vector12 expected_tilde = Q0 + (g12 * dt) * dt + V0b * dt;
            for(int a = 0; a < 12; ++a)
                T62_CHECK(c, rel_err(q_til[0](a), expected_tilde(a)) <= 1e-12);
            // ... which is *not* the first frame's q_tilde (the frame really re-ran)
            double gap = 0.0;
            for(int a = 0; a < 12; ++a)
                gap = std::max(gap, std::abs(q_til[0](a) - bdf1_tilde[0](a)));
            T62_CHECK(c, gap > 1e-3);
            std::printf("[t62] scene bdf1: frame-to-frame |q_tilde| gap %.3e\n", gap);
        }

        // the BDF2 *state system* exists in a bdf1 scene (its `do_build` only requires
        // the dynamics) but a bdf1 scene never runs the BDF2 integrator, so its two
        // history buffers stay at size 0
        REQUIRE(T62Probe::bdf2_state != nullptr);
        T62_CHECK(c, T62Probe::bdf2_state->is_valid());
        T62_CHECK(c, state_q_n_1s(T62Probe::bdf2_state).size() == 0u);
        T62_CHECK(c, state_q_v_n_1s(T62Probe::bdf2_state).size() == 0u);
    }

    // =======================================================================
    //  BDF2 -- the bootstrap, then the state's dump/recover round trip
    // =======================================================================
    std::vector<Vector12> dumped_q_n_1s, dumped_q_v_n_1s;
    {
        T62Probe::reset();
        auto scene = make_scene("bdf2");
        Engine engine{"luisa", workspace_path("uipc_luisa_t62_scene_bdf2")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T62Probe::bdf2_state != nullptr);
        REQUIRE(T62Probe::bdf2_state->is_valid());
        REQUIRE(T62Probe::abd != nullptr);
        T62_CHECK(c, T62Probe::abd->qs().size() == 1u);

        seed(world);
        step_frame(world);

        auto q_prev = read_vec12(T62Probe::abd->q_prevs());
        auto q_til  = read_vec12(T62Probe::abd->q_tildes());
        auto q_vs   = read_vec12(T62Probe::abd->q_vs());
        auto q_n_1s = read_vec12(state_q_n_1s(T62Probe::bdf2_state));
        auto q_v_n_1s = read_vec12(state_q_v_n_1s(T62Probe::bdf2_state));
        REQUIRE(q_n_1s.size() == 1u);
        REQUIRE(q_v_n_1s.size() == 1u);

        // the bootstrap: on the first predict_dof the history is filled from the
        // *dynamics'* q_prevs (== q at init) and q_vs (== the seeded V0)
        double worst_boot = 0.0;
        for(int a = 0; a < 12; ++a)
        {
            worst_boot = std::max(worst_boot, rel_err(q_n_1s[0](a), Q0(a)));
            worst_boot = std::max(worst_boot, rel_err(q_v_n_1s[0](a), V0(a)));
        }
        // the shipped BDF2 kernel with the bootstrapped history (q_{n-1} == q_n == Q0
        // and q_v_n == q_v_{n-1} == V0), hand-derived:
        //   q_tilde = 4/3 Q0 - 1/3 Q0 + 2/9 h (4 V0 - V0) + 4/9 h^2 g12
        //           = Q0 + 2/3 h V0 + 4/9 h^2 g12
        double worst_til = 0.0;
        {
            const double  c_g = (4.0 / 9.0) * dt * dt;
            const double  c_v = (2.0 / 9.0) * dt;
            const Vector12 expected =
                c_g * g12 + (4.0 / 3.0) * Q0 - (1.0 / 3.0) * Q0 + c_v * (4.0 * V0 - V0);
            for(int a = 0; a < 12; ++a)
                worst_til = std::max(worst_til, rel_err(q_til[0](a), expected(a)));
            // ... which is exactly the analytic Q0 + 2/3 h V0 + 4/9 h^2 g12
            const Vector12 analytic = Q0 + (2.0 / 3.0) * dt * V0 + c_g * g12;
            for(int a = 0; a < 12; ++a)
                T62_CHECK(c, rel_err(expected(a), analytic(a)) <= 1e-15);
        }
        // the update: q_v = 1/(2h)(3 q - 4 q_n + q_{n-1}) == 0 (q == q_n == q_{n-1}),
        // and the history rolls: q_v_{n-1} := the old q_v_n == V0, q_{n-1} := q_n == Q0
        double worst_v = 0.0, worst_hist = 0.0;
        for(int a = 0; a < 12; ++a)
        {
            worst_v    = std::max(worst_v, std::abs(q_vs[0](a)));
            worst_hist = std::max(worst_hist, rel_err(q_v_n_1s[0](a), V0(a)));
            worst_hist = std::max(worst_hist, rel_err(q_n_1s[0](a), Q0(a)));
            T62_CHECK(c, rel_err(q_prev[0](a), Q0(a)) <= 1e-15);
        }

        // the two integrators are *different* implementations: at the first frame the
        // BDF2 bootstrap transient gives 2/3 h V0 where BDF1 gives h V0
        {
            REQUIRE(bdf1_tilde.size() == 1u);
            double gap = 0.0;
            for(int a = 0; a < 12; ++a)
                gap = std::max(gap, std::abs(q_til[0](a) - bdf1_tilde[0](a)));
            T62_CHECK(c, gap > 1e-3);
            std::printf("[t62] scene bdf1-vs-bdf2 first frame: max|q_tilde diff| %.3e\n", gap);
        }

        std::printf("[t62] scene bdf2 frame 1: bootstrap rel %.3e  q_tilde rel %.3e  "
                    "|q_v| %.3e  history rel %.3e\n",
                    worst_boot,
                    worst_til,
                    worst_v,
                    worst_hist);
        T62_CHECK(c, worst_boot <= 1e-14);
        T62_CHECK(c, worst_til <= 1e-12);
        T62_CHECK(c, worst_v <= 1e-15);
        T62_CHECK(c, worst_hist <= 1e-14);

        // ---- the state's dump/recover round trip -----------------------------
        REQUIRE(world.dump());
        dumped_q_n_1s  = read_vec12(state_q_n_1s(T62Probe::bdf2_state));
        dumped_q_v_n_1s = read_vec12(state_q_v_n_1s(T62Probe::bdf2_state));
        REQUIRE(dumped_q_n_1s.size() == 1u);
        // the dumped history is *not* what a fresh scene would produce (q_v_n_1s != 0)
        T62_CHECK(c, dumped_q_v_n_1s[0].norm() > 1e-9);
    }
    {
        T62Probe::reset();
        auto scene = make_scene("bdf2");
        Engine engine{"luisa", workspace_path("uipc_luisa_t62_scene_bdf2")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T62Probe::bdf2_state != nullptr);
        step_frame(world);  // a *fresh* scene (q_v == 0), so its history differs

        auto fresh_v = read_vec12(state_q_v_n_1s(T62Probe::bdf2_state));
        REQUIRE(fresh_v.size() == dumped_q_v_n_1s.size());
        double differs = 0.0;
        for(size_t i = 0; i < fresh_v.size(); ++i)
            differs = std::max(differs, (fresh_v[i] - dumped_q_v_n_1s[i]).norm());
        T62_CHECK(c, differs > 1e-9);

        REQUIRE(world.recover());
        auto recovered_x = read_vec12(state_q_n_1s(T62Probe::bdf2_state));
        auto recovered_v = read_vec12(state_q_v_n_1s(T62Probe::bdf2_state));
        REQUIRE(recovered_x.size() == dumped_q_n_1s.size());
        REQUIRE(recovered_v.size() == dumped_q_v_n_1s.size());
        bool x_bitwise = true, v_bitwise = true;
        for(size_t i = 0; i < recovered_x.size(); ++i)
        {
            x_bitwise = x_bitwise
                        && std::memcmp(&recovered_x[i], &dumped_q_n_1s[i], sizeof(Vector12)) == 0;
            v_bitwise = v_bitwise
                        && std::memcmp(&recovered_v[i], &dumped_q_v_n_1s[i], sizeof(Vector12)) == 0;
        }
        std::printf("[t62] ABDBDF2State dump/recover: q_n_1s bitwise %d, q_v_n_1s bitwise %d\n",
                    static_cast<int>(x_bitwise),
                    static_cast<int>(v_bitwise));
        T62_CHECK(c, x_bitwise);
        T62_CHECK(c, v_bitwise);
    }
}

// ===========================================================================
//  6) device == host for every ported routine
// ===========================================================================
/// One deterministic evaluation of every T6.2 routine, serialised through `put` so
/// the host reference and the device kernel walk an identical, fixed order (cookbook
/// §7.8 / §7.16 rule 12b).
///
/// Entry map (462 doubles):
///    0.. 11  sym::abd_bdf2::compute_q_tilde (no gravity term)
///   12.. 23  sym::abd_bdf2::compute_q_tilde (with gravity term)
///   24.. 35  sym::abd_bdf2::compute_qv
///   36.. 71  bdf1_detail::predict_dof  x {dynamic, static, fixed}
///   72..107  bdf2_detail::predict_dof  x {dynamic, static, fixed}
///  108..119  bdf1_detail::update_state
///  120       bdf1 kinetic energy, free body
///  121       bdf1 kinetic energy, fixed body
///  122       bdf1 kinetic energy, external-kinetic body
///  123..134  bdf1 kinetic gradient, free body
///  135..146  bdf1 kinetic gradient, fixed body (all zero)
///  147..290  bdf1 kinetic Hessian (row major)
///  291       bdf2 kinetic energy, free body
///  292       bdf2 kinetic energy, fixed body
///  293       bdf2 kinetic energy, external-kinetic body
///  294..305  bdf2 kinetic gradient, free
///  306..317  bdf2 kinetic gradient, fixed
///  318..461  bdf2 kinetic Hessian
constexpr int kRoundCount = 12 * 3 + 36 + 36 + 12 + 3 + 24 + 144 + 3 + 24 + 144;

template <typename S, typename I, typename Put>
void abd_dynamics_round(Put&& put)
{
    const auto h  = lps::make_scalar<S>(0.0125);
    const auto dt = lps::make_scalar<S>(0.0125);

    const std::array<double, 12> a_q_n{1.25, -0.5, 2.0, 0.5, 0.25, -0.75, 1.0, -0.5, 0.5,
                                       -0.25, 1.5, 0.75};
    const std::array<double, 12> a_q_n_1{-0.75, 0.25, 1.5, -0.5, 0.5, 0.25, 0.75, -1.0, 0.25,
                                         1.0, 0.5, -0.5};
    const std::array<double, 12> a_q_v_n{0.5, -1.25, 0.75, 0.25, -0.5, 0.125, -0.75, 0.5, 1.0,
                                         0.5, -0.25, 0.75};
    const std::array<double, 12> a_q_v_n_1{-0.25, 0.5, -0.5, 0.125, -0.25, 0.5, 1.0, -0.75, 0.25,
                                           -0.5, 0.25, -1.0};
    const std::array<double, 12> a_g{0.0, -9.8, 0.3, 0.1, 0.2, -0.3, 0.25, 0.5, -0.125, 0.0,
                                     0.0, 0.0};
    const std::array<double, 12> a_f_acc{0.1, 0.2, -0.3, -0.1, -0.2, 0.3, 0.5, 0.5, 0.5, 0.0,
                                         0.0, 0.0};
    const std::array<double, 12> a_q{1.0, 0.5, -0.25, 0.5, 0.25, -0.75, 1.0, -0.5, 0.5, -0.25,
                                     1.5, 0.75};

    const auto q_n     = lv12<S>(a_q_n);
    const auto q_n_1   = lv12<S>(a_q_n_1);
    const auto q_v_n   = lv12<S>(a_q_v_n);
    const auto q_v_n_1 = lv12<S>(a_q_v_n_1);
    const auto g       = lv12<S>(a_g);
    const auto f_acc   = lv12<S>(a_f_acc);
    const auto q       = lv12<S>(a_q);

    const auto m = mass_triad<S>(0.375,
                                 {0.15, -0.25, 0.5},
                                 {0.5, 0.125, -0.25, 0.125, 1.5, 0.75, -0.25, 0.75, 0.25});

    const I not_fixed = I{0};
    const I is_fixed  = I{1};
    const I is_dyn    = I{1};
    const I is_static = I{0};

    // 1) sym::abd_bdf2::compute_q_tilde (no gravity term)
    {
        lps::LVec<S, 12> r;
        sym::abd_bdf2::compute_q_tilde(r, q_n, q_n_1, q_v_n, q_v_n_1, h);
        for(int i = 0; i < 12; ++i)
            put(r(i));
    }
    // 2) sym::abd_bdf2::compute_q_tilde (with gravity term)
    {
        lps::LVec<S, 12> r;
        sym::abd_bdf2::compute_q_tilde(r, q_n, q_n_1, q_v_n, q_v_n_1, g, h);
        for(int i = 0; i < 12; ++i)
            put(r(i));
    }
    // 3) sym::abd_bdf2::compute_qv
    {
        lps::LVec<S, 12> r;
        sym::abd_bdf2::compute_qv(r, q, q_n, q_n_1, h);
        for(int i = 0; i < 12; ++i)
            put(r(i));
    }
    // 4) the BDF1 per-body predict for the three flag combinations
    {
        auto r = bdf1_detail::predict_dof(q, q_v_n, g, f_acc, dt, not_fixed, is_dyn);
        for(int i = 0; i < 12; ++i)
            put(r(i));
        r = bdf1_detail::predict_dof(q, q_v_n, g, f_acc, dt, not_fixed, is_static);
        for(int i = 0; i < 12; ++i)
            put(r(i));
        r = bdf1_detail::predict_dof(q, q_v_n, g, f_acc, dt, is_fixed, is_dyn);
        for(int i = 0; i < 12; ++i)
            put(r(i));
    }
    // 5) the BDF2 per-body predict (gravity alone -- the reference's ABD BDF2 kernel
    //    has no external-force argument, unlike its BDF1 sibling and its FEM twin)
    {
        auto r = bdf2_detail::predict_dof(q_n, q_n_1, q_v_n, q_v_n_1, g, dt, not_fixed, is_dyn);
        for(int i = 0; i < 12; ++i)
            put(r(i));
        r = bdf2_detail::predict_dof(q_n, q_n_1, q_v_n, q_v_n_1, g, dt, not_fixed, is_static);
        for(int i = 0; i < 12; ++i)
            put(r(i));
        r = bdf2_detail::predict_dof(q_n, q_n_1, q_v_n, q_v_n_1, g, dt, is_fixed, is_dyn);
        for(int i = 0; i < 12; ++i)
            put(r(i));
    }
    // 6) bdf1_detail::update_state
    {
        auto r = bdf1_detail::update_state(q, q_n, dt);
        for(int i = 0; i < 12; ++i)
            put(r(i));
    }
    // 7) the BDF1 kinetic: energy x {free, fixed, external-kinetic}, gradient x
    //    {free, fixed}, Hessian
    {
        put(bdf1_kinetic_detail::energy(q, q_n, m, not_fixed, not_fixed));
        put(bdf1_kinetic_detail::energy(q, q_n, m, is_fixed, not_fixed));
        put(bdf1_kinetic_detail::energy(q, q_n, m, not_fixed, is_fixed));
        auto G = bdf1_kinetic_detail::gradient(q, q_n, m, not_fixed);
        for(int i = 0; i < 12; ++i)
            put(G(i));
        auto G0 = bdf1_kinetic_detail::gradient(q, q_n, m, is_fixed);
        for(int i = 0; i < 12; ++i)
            put(G0(i));
        auto H = bdf1_kinetic_detail::hessian(m);
        for(int r = 0; r < 12; ++r)
            for(int c = 0; c < 12; ++c)
                put(H(r, c));
    }
    // 8) the BDF2 kinetic (with the inv_beta scaling)
    {
        const auto inv_beta = lps::make_scalar<S>(BDF2Kinetic::inv_beta);
        put(bdf2_kinetic_detail::energy(q, q_n, m, inv_beta, not_fixed, not_fixed));
        put(bdf2_kinetic_detail::energy(q, q_n, m, inv_beta, is_fixed, not_fixed));
        put(bdf2_kinetic_detail::energy(q, q_n, m, inv_beta, not_fixed, is_fixed));
        auto G = bdf2_kinetic_detail::gradient(q, q_n, m, inv_beta, not_fixed);
        for(int i = 0; i < 12; ++i)
            put(G(i));
        auto G0 = bdf2_kinetic_detail::gradient(q, q_n, m, inv_beta, is_fixed);
        for(int i = 0; i < 12; ++i)
            put(G0(i));
        auto H = bdf2_kinetic_detail::hessian(m, inv_beta);
        for(int r = 0; r < 12; ++r)
            for(int c = 0; c < 12; ++c)
                put(H(r, c));
    }
}

auto& abd_dynamics_round_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>>(
        [](lcrt::BufferVar<double> out) noexcept
        {
            uint32_t index = 0;
            abd_dynamics_round<lcrt::Var<double>, lcrt::Var<int32_t>>(
                [&](auto&& value) noexcept { out->write(index++, value); });
        });
    return k;
}

TEST_CASE("abd_dynamics_device_vs_host", "[luisa][abd_dynamics]")
{
    test_runtime();
    CaseScope scope{"abd_dynamics_device_vs_host"};
    auto&     c = scope.counter;

    std::vector<double> host;
    host.reserve(kRoundCount);
    abd_dynamics_round<double, IndexT>([&](double value) { host.push_back(value); });
    REQUIRE(static_cast<int>(host.size()) == kRoundCount);

    lud::LDeviceBuffer<double> device(kRoundCount);
    auto&                      stream = gg::frame_stream();
    stream.record(abd_dynamics_round_kernel()(device.view()).dispatch(1u));
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
    std::printf("[t62] device-vs-host round: %d/%d bitwise equal, worst abs %.3e (entry %d)\n",
                bitwise_equal,
                kRoundCount,
                worst_abs,
                worst_idx);

    // no transcendental is involved and the arithmetic is the same code on both sides,
    // so the result must be bit-identical (T2.1's convention); the bound is the
    // documented fallback if a future backend contraction (FMA) changes it
    T62_CHECK(c, bitwise_equal == kRoundCount);
    T62_CHECK(c, worst_abs <= 1e-12);

    // ---- the entries the layout above pins ------------------------------------
    // entry 120/291: the BDF1/BDF2 free-body kinetic energy
    T62_CHECK(c, rel_err(host[291], BDF2Kinetic::inv_beta * host[120]) <= 1e-12);
    // entries 121/122/292/293: a fixed *or* external-kinetic body has zero kinetic
    // energy -- the external-kinetic guard is T6.2's own (the FEM kinetic has none)
    T62_CHECK(c, host[121] == 0.0);
    T62_CHECK(c, host[122] == 0.0);
    T62_CHECK(c, host[292] == 0.0);
    T62_CHECK(c, host[293] == 0.0);
    // entry 135..137 is the fixed body's *gradient*: zero, while its Hessian
    // (147..290) is the mass operator, not zero
    T62_CHECK(c, host[135] == 0.0);
    T62_CHECK(c, host[146] == 0.0);
    // H(0,0) = m = 0.375 and H(0,3) = m x̄(0) = 0.15 -- the *dense* ABD mass operator
    // (the FEM kinetic's Hessian is diagonal, this one is not)
    T62_CHECK(c, rel_err(host[147], 0.375) <= 1e-15);
    T62_CHECK(c, rel_err(host[147 + 3], 0.15) <= 1e-15);
    T62_CHECK(c, host[147 + 1] == 0.0);
    T62_CHECK(c, rel_err(host[318], BDF2Kinetic::inv_beta * 0.375) <= 1e-15);
    T62_CHECK(c, rel_err(host[318 + 3], BDF2Kinetic::inv_beta * 0.15) <= 1e-15);
    // the free-body gradient is not zero (the check above is not vacuous).  NOTE: a
    // `||` of two comparisons inside one `CHECK` is a Catch2 *chained comparison*
    // (measured C2338), so the bool is materialised first.
    const bool free_gradient_nonzero = (std::abs(host[123]) > 0.0)
                                       || (std::abs(host[124]) > 0.0);
    T62_CHECK(c, free_gradient_nonzero);
}

// ===========================================================================
//  7) per-case assertion accounting (brief §5.3)
// ===========================================================================
TEST_CASE("abd_dynamics_summary", "[luisa][abd_dynamics]")
{
    int total = 0;
    for(auto& [name, checks] : t62::g_case_counts)
    {
        total += checks;
        std::printf("[t62] case %-32s %6d assertions\n", name.c_str(), checks);
    }
    std::printf("[t62] summary: %zu cases, %d assertions\n", t62::g_case_counts.size(), total);
    CHECK(t62::g_case_counts.size() == 6u);
    CHECK(total > 0);
}
}  // namespace t62
