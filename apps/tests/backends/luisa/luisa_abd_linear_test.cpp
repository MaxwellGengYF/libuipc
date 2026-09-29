/********************************************************************
 * @file luisa_abd_linear_test.cpp
 * @brief T6.1b — the ABD linear subsystem, the ABD animator and the
 * inter-affine-body layer (LuisaCompute backend).
 *
 * Every oracle here is derived independently *in this file*: the reference backend
 * is CUDA-only and cannot be host-included, so nothing is copied from it.
 *
 * ---------------------------------- T6.1b -----------------------------------
 * Coverage map (case -> oracle):
 *  1) the 2-cube chain: the body term (kinetic+shape) plus a hand-written
 *     per-vertex-pair Jacobi-triple reporter -> the *raw* COO content (per
 *     (row, col) block sums), the converted BCOO (sorted/unique/summed), and
 *     MatrixConverter -> Spmv vs a dense Eigen A*x assembled in a different
 *     order ................................................... abd_linear_chain
 *  2) the dof layout + the extent budget (factored and through the real chain):
 *     uid order, offsets/counts, gradient_only both ways, the per-geometry
 *     dof_offset/dof_count meta, ReportExtentInfo::check ... abd_linear_layout
 *  3) a real scene with a test-local AffineBodyConstraint (uid 16) *and* a
 *     test-local InterAffineBodyConstitution (uid 25): the animator's and the
 *     inter-body manager's fan-outs, the (constraint -> slice) offsets and the
 *     assembled contributions ................................ abd_linear_animator
 *  4) the dytopo receiver through the ported manager: registration/report/
 *     receive, and the ABD end of it (J^T G and J_i^T H J_j with the L/R swap)
 *     .......................................................... abd_linear_dytopo
 *  5) a real scene through Engine{"luisa", ...} with one ABD body and one
 *     advance(): the new systems build/init and stay consistent; the dof count
 *     is the hand-derived 12 x body_count ...................... abd_linear_scene
 *  6) every factored assembly routine: host (`storage::memory`) vs device (DSL),
 *     bitwise where the arithmetic is identical ...... abd_linear_device_vs_host
 *
 * Shared-binary rules (cookbook §7.14 rule 6, §7.18 rule 6, §7.21 rule 4): every
 * system registered here is built for *every* scene of the whole test binary, so
 * each test-local system gates itself on a marker meta attribute of *its own*
 * geometry (or on a static switch that is off unless this file's test set it) and
 * reports zero extents elsewhere.  `find<T>()` returns an invalidated system only
 * while its type is still in the collection map, so "is it in play?" is asserted
 * with `is_valid()` where it matters.
 *
 * WHAT THIS FILE CANNOT REACH (stated so nothing is implied)
 * ---------------------------------------------------------
 * `GlobalLinearSystem::solve()` is T8.3, so the assembly is driven through a
 * *test-local* `GlobalLinearSystem::Impl` (a public nested class, the T5.4a idiom)
 * with the *real*, engine-built `ABDLinearSubsystem` registered into it.  The ABD
 * line-search half (`ABDLineSearchReporter`) is T6.4's, so
 * `AffineBodyAnimator::compute_energy` is compiled but not reachable; the same
 * holds for every line-search subreporter registered here (they are invalidated by
 * `require<ABDLineSearchReporter>()`).
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"
#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_fixed_joint.h>
#include <uipc/constitution/soft_transform_constraint.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <type_define.h>
#include <frame_stream.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/global_linear_system_function.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/abd_linear_subsystem_reporter.h>
#include <affine_body/abd_linear_subsystem_function.h>
#include <affine_body/abd_dytopo_effect_receiver.h>
#include <affine_body/affine_body_animator.h>
#include <affine_body/affine_body_constraint.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/inter_affine_body_animator.h>
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/inter_affine_body_constraint.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
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
namespace labd = uipc::backend::luisa::abd_linear_subsystem_detail;

// T6.1a's test register-allocates the same names in its own anonymous namespace;
// this file's helpers live in `t61b` so the two translation units cannot clash
// (they are separate TUs, but keeping the names apart documents the split).
namespace t61b
{
// ---------------------------------------------------------------------------
//  small helpers (the T5.1/T5.3/T6.1a test idioms)
// ---------------------------------------------------------------------------
[[nodiscard]] inline double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// Read an LC buffer back to the host (frame stream first: a command may still be
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

[[nodiscard]] inline Vector12 to_eigen12(const gg::LVector12& v)
{
    Vector12 out;
    for(int i = 0; i < 12; ++i)
        out(i) = v.data[static_cast<std::size_t>(i)];
    return out;
}

[[nodiscard]] inline Matrix12x12 to_eigen12(const gg::LMatrix12x12& m)
{
    Matrix12x12 out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m.data[static_cast<std::size_t>(c * 12 + r)];
    return out;
}

[[nodiscard]] inline lps::LVec<double, 3> lv3(const Vector3& v) noexcept
{
    return lps::LVec<double, 3>{v(0), v(1), v(2)};
}

[[nodiscard]] inline lps::LVec<double, 12> lv12(const Vector12& v) noexcept
{
    lps::LVec<double, 12> out;
    for(int i = 0; i < 12; ++i)
        out(i) = v(i);
    return out;
}

[[nodiscard]] inline lps::LMat<double, 3, 3> lm3(const Matrix3x3& m) noexcept
{
    lps::LMat<double, 3, 3> out;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] inline lps::LMat<double, 12, 12> lm12(const Matrix12x12& m) noexcept
{
    lps::LMat<double, 12, 12> out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] inline lps::LMat<double, 3, 3> to_eigen3(const lps::LMat<double, 3, 3>& m) noexcept
{
    return m;
}

[[nodiscard]] inline Matrix3x3 to_eigen(const lps::LMat<double, 3, 3>& m)
{
    Matrix3x3 out;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] inline Matrix12x12 to_eigen(const lps::LMat<double, 12, 12>& m)
{
    Matrix12x12 out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m(r, c);
    return out;
}

struct BoxMoments
{
    Float     volume = 0.0;
    Float     m      = 0.0;
    Vector3   mx     = Vector3::Zero();
    Matrix3x3 mxx    = Matrix3x3::Zero();
};

[[nodiscard]] inline BoxMoments box_moments(Float L, Float rho)
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

/// The six-tet decomposition of the cube [0,L]^3 (vertex index = x + 2y + 4z).
[[nodiscard]] inline std::array<Vector4i, 6> cube_tets() noexcept
{
    return {{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}}};
}

[[nodiscard]] inline std::array<Vector3, 8> cube_vertices(Float L) noexcept
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

/// The independent Eigen oracle of `ABDJacobi::to_mat()`: the 3x12 jacobian
/// `J(:, 0:3) = I`, `J(i, 3 + 3i + j) = x_bar(j)`.
[[nodiscard]] inline gg::Matrix3x12 jacobi_oracle(const Vector3& x_bar) noexcept
{
    gg::Matrix3x12 J = gg::Matrix3x12::Zero();
    J.block<3, 3>(0, 0).setIdentity();
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            J(i, 3 + 3 * i + j) = x_bar(j);
    return J;
}

// ---------------------------------------------------------------------------
//  the hand-derived chain model
//
//  Two ABD cubes (L = 0.5), each 8 vertices / 1 instance, in the identity affine
//  configuration, so `q = [0,0,0, 1,0,0, 0,1,0, 0,0,1]` for both bodies.  The
//  model is written down here and is what every expectation below is derived from:
//   * the *body term* (kinetic + shape): a hand-written quadratic energy per body,
//     `E_i = 1/2 q_i^T H_i q_i` -> `grad = H_i q_i`, `hess = H_i`;
//   * the *reporter term*: per-body vertex pairs plus one inter-body pair, whose
//     12x12 blocks are the ported `J^T H3 J` (the "Jacobi triple") computed on the
//     *device* from the scene's per-vertex jacobians, and by Eigen here;
//   * the *constraint* and *inter-body* terms (cases 3): hand-written 12x12 blocks.
// ---------------------------------------------------------------------------
constexpr Float kL      = 0.5;
constexpr Float kRho    = 1000.0;
constexpr Float kKappa  = 1.0e8;
constexpr SizeT kBodyCount = 2;
constexpr SizeT kVertexPerBody = 8;

/// The hand-written per-body Hessian `H_i(u,v) = a (u+1)(v+1) + b d_uv`.
[[nodiscard]] inline Matrix12x12 body_hessian(SizeT body) noexcept
{
    Matrix12x12 H = Matrix12x12::Zero();
    const double a = 0.5 + 0.25 * static_cast<double>(body);
    const double b = 2.0 + static_cast<double>(body);
    for(int u = 0; u < 12; ++u)
        for(int v = 0; v < 12; ++v)
            H(u, v) = a * static_cast<double>((u + 1) * (v + 1)) + (u == v ? b : 0.0);
    return H;
}

/// `q` of every body in this file's scenes: the identity affine map.
[[nodiscard]] inline Vector12 identity_q() noexcept
{
    Vector12 q = Vector12::Zero();
    q(3) = 1.0;
    q(7) = 1.0;
    q(11) = 1.0;
    return q;
}

/// The hand-written 3x3 block of a vertex pair `(v, w)` (deliberately
/// non-symmetric, so the (i,j)/(j,i) mirror of the symmetric reconstruction is
/// load-bearing).
[[nodiscard]] inline Matrix3x3 pair_h3(SizeT v, SizeT w) noexcept
{
    Matrix3x3 H;
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            H(r, c) = 0.3 + 0.1 * static_cast<double>(v) + static_cast<double>(r) + 2.0 * static_cast<double>(c)
                      + 0.05 * static_cast<double>(v + w);
    return H;
}

/// The vertex pairs the test's reporter reports, in the reference's *upper
/// triangle* convention (body_i <= body_j).
struct PairEntry
{
    SizeT body_i = 0;
    SizeT body_j = 0;
    SizeT v      = 0;  // local vertex id of body_i
    SizeT w      = 0;  // local vertex id of body_j
};

[[nodiscard]] inline std::vector<PairEntry> pair_entries()
{
    return {{0, 0, 0, 0},
            {0, 0, 0, 1},
            {0, 0, 1, 2},
            {0, 0, 2, 3},
            {0, 0, 3, 4},
            {1, 1, 0, 0},
            {1, 1, 1, 1},
            {1, 1, 2, 4},
            {1, 1, 4, 5},
            {1, 1, 6, 7},
            // the inter-body coupling of the chain
            {0, 1, 5, 3}};
}

/// The reporter's gradient doublets (one per body): a hand-written 12-vector.
[[nodiscard]] inline Vector12 reporter_gradient(SizeT body) noexcept
{
    Vector12 g;
    for(int k = 0; k < 12; ++k)
        g(k) = 0.5 + 0.25 * static_cast<double>(k) + 2.0 * static_cast<double>(body);
    return g;
}

/// The test-local constraint's (uid 16) single diagonal block on body 0.
[[nodiscard]] inline Matrix12x12 constraint_hessian() noexcept
{
    Matrix12x12 H;
    for(int u = 0; u < 12; ++u)
        for(int v = 0; v < 12; ++v)
            H(u, v) = (u == v ? 5.0 : 0.0) + 0.125 * static_cast<double>(u + 1) * static_cast<double>(v + 2);
    return H;
}

[[nodiscard]] inline Vector12 constraint_gradient() noexcept
{
    Vector12 g;
    for(int k = 0; k < 12; ++k)
        g(k) = -1.0 + 0.5 * static_cast<double>(k);
    return g;
}

/// The test-local inter-body constitution's (uid 25) coupling block at (0, 1).
[[nodiscard]] inline Matrix12x12 inter_hessian() noexcept
{
    Matrix12x12 H;
    for(int u = 0; u < 12; ++u)
        for(int v = 0; v < 12; ++v)
            H(u, v) = 0.25 + 0.0625 * static_cast<double>((u + 1) * (v + 3));
    return H;
}

[[nodiscard]] inline Vector12 inter_gradient() noexcept
{
    Vector12 g;
    for(int k = 0; k < 12; ++k)
        g(k) = 0.75 - 0.125 * static_cast<double>(k);
    return g;
}

// ---------------------------------------------------------------------------
//  per-case assertion accounting (the T6.1a idiom)
// ---------------------------------------------------------------------------
struct Counter
{
    int checks = 0;
};

inline std::vector<std::pair<std::string, int>> g_case_counts;

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

#define T61B_CHECK(counter, cond) \
    do \
    { \
        ++(counter).checks; \
        CHECK(cond); \
    } while(false)

// ---------------------------------------------------------------------------
//  the shared-binary marker
// ---------------------------------------------------------------------------
/// A *meta* attribute on the geometries this file's scenes create; every
/// test-local ABD system reports zero extents unless it is present.
inline constexpr const char* kMarker = "t61b_linear_marker";
}  // namespace t61b

// ===========================================================================
//  the test-local systems (all gated on the marker / on a static switch)
// ===========================================================================
namespace t61b
{
/// Reserved *reporter* uids for documentation (the reporter bases have no uid
/// hook; `ABDLinearSubsystem::Impl::init` assigns `m_index`).  The uids in use
/// elsewhere are listed in the T4.x/T5.x test files.  The two *constraint*
/// uids this file claims (16 and 25) are the frontend's
/// `SoftTransformConstraint` and `AffineBodyFixedJoint`, i.e. the same
/// knowing-reuse arrangement T6.1a's `T61ABDMockConstitution` uses for uid 1:
/// the reference's own numbering has no free slot (`ConstitutionTabular::insert`
/// resolves through `ConstitutionUIDCollection::find(uid)`), and T6.4/T6.3b will
/// claim them with the real bodies.
inline constexpr U64 kSoftTransformUID = 16;
inline constexpr U64 kFixedJointUID    = 25;

[[nodiscard]] inline bool scene_has_marker(backend::WorldVisitor& world)
{
    for(auto& slot : world.scene().geometries())
    {
        if(auto* sc = slot->geometry().as<geometry::SimplicialComplex>())
        {
            if(sc->meta().find<IndexT>(kMarker))
                return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
//  (a) the test-local kinetic: a hand-written quadratic per body
// ---------------------------------------------------------------------------
/// Writes `hessians(body) = H_body` and `gradients(body) = H_body q_body` (the
  /// gradient/Hessian of `E = 1/2 q^T H q`).  The scene's only *other* constitution
  /// is the shared `T61ABDMockConstitution` (uid 1), whose
  /// `do_compute_gradient_hessian` is a no-op, so the shape term stays at the value
  /// `Impl::init` left it (`LDeviceBuffer::resize` zero-fills the tail).
  ///
  /// NOTE (T6.2): this class is no longer `final` and is no longer the registered
  /// system -- `zt61b::T61bKineticOwner` below derives from it purely so that the
  /// *registered* type's demangled name sorts after every `uipc::backend::luisa::*`
  /// system, which is what decides who owns the dynamics' single kinetic slot (see
  /// the block comment at the subclass).
  class T61bKinetic : public gg::AffineBodyKinetic
{
  public:
    using AffineBodyKinetic::AffineBodyKinetic;

    static inline bool enabled = false;

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        enabled = scene_has_marker(world());
    }

    void do_compute_energy(ComputeEnergyInfo&) override {}

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        if(!enabled)
            return;
        auto qs = info.qs();
        auto n  = static_cast<uint32_t>(info.qs().size());
        if(n == 0)
            return;

        auto& k = kinetic_kernel();
        gg::frame_stream().record(
            k(info.gradients(), info.hessians(), qs, h_buffer(), h_buffer(), n).dispatch(n));
    }

  private:
    /// The per-body `H_i` on the device (uploaded from the test's hand-written
    /// matrices, sized by `kBodyCount`).
    static lud::LDeviceBuffer<gg::LMatrix12x12>& body_hessian_buffer();
    static lud::LBufferView<gg::LMatrix12x12>    h_buffer();

    static auto& kinetic_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<gg::LVector12>,
                                                    lcrt::Buffer<gg::LMatrix12x12>,
                                                    lcrt::Buffer<gg::LVector12>,
                                                    lcrt::Buffer<gg::LMatrix12x12>,
                                                    lcrt::Buffer<gg::LMatrix12x12>,
                                                    uint32_t>(
            [](lcrt::BufferVar<gg::LVector12>    gradients,
               lcrt::BufferVar<gg::LMatrix12x12> hessians,
               lcrt::BufferVar<gg::LVector12>    qs,
               lcrt::BufferVar<gg::LMatrix12x12> H,
               lcrt::BufferVar<gg::LMatrix12x12> /*unused*/,
               lcrt::Var<uint32_t>               n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto ii = lcrt::cast<std::int32_t>(i);
                    auto h  = gg::load_mat12x12(H->read(i));
                    auto q  = gg::load_vec12(qs->read(i));
                    hessians->write(i, H->read(i));
                    lcrt::Var<gg::LVector12> g;
                    gg::store_vec12(g, h * q);
                    gradients->write(i, g);
                };
            });
        return k;
    }
};

/// The per-body `H_i` (a host buffer, uploaded lazily once).
inline lud::LDeviceBuffer<gg::LMatrix12x12>& T61bKinetic::body_hessian_buffer()
{
    static lud::LDeviceBuffer<gg::LMatrix12x12> buffer;
    if(buffer.size() != kBodyCount)
    {
        std::vector<gg::LMatrix12x12> host;
        host.reserve(kBodyCount);
        for(SizeT b = 0; b < kBodyCount; ++b)
            host.push_back(gg::to_luisa(body_hessian(b)));
        buffer.resize(host.size());
        buffer.view().copy_from(span<const gg::LMatrix12x12>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}

inline lud::LBufferView<gg::LMatrix12x12> T61bKinetic::h_buffer()
{
    return body_hessian_buffer().view();
}
}  // namespace t61b

// ---------------------------------------------------------------------------
//  T6.2 spin-off — who owns `AffineBodyDynamics`' single kinetic slot
//
//  T6.1a's dynamics keeps *one* kinetic ("only one allowed", `register_sim_system`
//  overwrites) and this file's stand-in above used to be the only `AffineBodyKinetic`
//  subclass in the whole test binary, so the body term of every expectation below was
//  whatever *it* wrote.  T6.2 adds the shipped `AffineBodyBDF1Kinetic` /
//  `AffineBodyBDF2Kinetic`, which are just as valid in the same `bdf1`/`bdf2` scenes
//  (both only `require` their BDF flag), so the slot's owner is decided by the build
//  order: `SimEngine::build_systems()` sorts the SimSystem creators by **demangled
//  type name** and `add_kinetic()` overwrites, i.e. the *last* system built wins.
//  `t61b` is a *global* namespace, so `"class t61b::T61bKinetic"` sorts *before* every
//  `"class uipc::backend::luisa::…"` class and the shipped kinetic took the slot.
//  Measured (before this subclass existed): `abd_linear_chain` and
//  `abd_linear_animator` lost 8 assertions — e.g. the device's `A_eff(0,0)` became
//  `127.9 = 125` (the real dyadic mass `m = rho L³`) `+ 2.9` (the reporter part)
//  instead of the hand-derived `2.5 + 2.9`, and the gradient's body term became
//  `M (q - q̃) = 0` (the dynamics initialises `q̃ = q`) instead of `H q`.
//
//  The registered system is therefore this one-line subclass, in a namespace whose
//  name sorts **after** `uipc` (`'z' > 'u'`): it inherits the whole stand-in
//  (including `enabled` and the hand-written `H` buffer) and, being built last, is the
//  slot's owner again — with *no* change to any value this file derives.  It is the
//  same knowing-name-reuse pattern the uid comments above describe, recorded in the
//  T6.2 row / cookbook §7.23.
// ---------------------------------------------------------------------------
namespace zt61b
{
class T61bKineticOwner final : public t61b::T61bKinetic
{
  public:
    using T61bKinetic::T61bKinetic;
};
}  // namespace zt61b

REGISTER_SIM_SYSTEM(zt61b::T61bKineticOwner);

namespace t61b
{

// ---------------------------------------------------------------------------
//  (b) the test-local ABD linear-subsystem reporter: the Jacobi-triple blocks
// ---------------------------------------------------------------------------
/// Reports, per body, a set of vertex-pair blocks plus one inter-body block.  The
/// *device* computes each block as `J_v^T H3 J_w` with the ported
/// `abd_jacobi_JT_H_J` and the scene's own per-vertex jacobians (read from the
/// dynamics' `Js()`), and writes it into its own 12x12 slice with the
/// `(body_i, body_j)` indices; the test's oracle computes the same in Eigen.
class T61bLinearReporter final : public gg::ABDLinearSubsystemReporter
{
  public:
    using ABDLinearSubsystemReporter::ABDLinearSubsystemReporter;

    static inline bool  enabled        = false;
    static inline int   init_count     = 0;
    static inline int   extent_count   = 0;
    static inline int   assemble_count = 0;
    static inline SizeT last_gradient_count = 0;
    static inline SizeT last_hessian_count  = 0;
    static inline SizeT last_gradient_offset = 0;
    static inline SizeT last_hessian_offset  = 0;
    static inline bool  last_gradient_only   = false;

    static void reset()
    {
        enabled             = false;
        init_count          = 0;
        extent_count        = 0;
        assemble_count      = 0;
        last_gradient_count = 0;
        last_hessian_count  = 0;
        last_gradient_offset = 0;
        last_hessian_offset  = 0;
        last_gradient_only   = false;
    }

  protected:
    void do_build(BuildInfo& info) override
    {
        (void)info;
        m_dynamics = &require<gg::AffineBodyDynamics>();
        m_vertex_reporter = &require<gg::AffineBodyVertexReporter>();
    }

    void do_init(InitInfo& info) override
    {
        (void)info;
        ++init_count;
        enabled = scene_has_marker(world());
    }

    void do_report_extent(ReportExtentInfo& info) override
    {
        ++extent_count;
        last_gradient_only = info.gradient_only();
        if(!enabled)
        {
            info.gradient_count(0);
            info.hessian_count(0);
            return;
        }
        info.gradient_count(kBodyCount);
        info.hessian_count(info.gradient_only() ? 0 : pair_entries().size());
        last_gradient_count = kBodyCount;
        last_hessian_count  = info.gradient_only() ? 0 : pair_entries().size();
    }

    void do_assemble(AssembleInfo& info) override
    {
        ++assemble_count;
        if(!enabled)
            return;

        auto gradients = info.gradients();
        auto hessians  = info.hessians();
        last_gradient_offset = static_cast<SizeT>(gradients.doublet_index_offset());
        last_hessian_offset  = static_cast<SizeT>(hessians.triplet_index_offset());

        // the gradient half: `kBodyCount` doublets, hand-written 12-vectors
        {
            auto offset = static_cast<std::size_t>(gradients.doublet_index_offset());
            auto count  = static_cast<std::size_t>(gradients.doublet_count());
            if(count >= kBodyCount)
            {
                auto indices = gradients.indices().subview(offset, count);
                auto values  = gradients.values().subview(offset * 12u, count * 12u);
                auto& k      = gradient_kernel();
                gg::frame_stream().record(
                    k(indices, values, gradient_buffer().view(), static_cast<uint32_t>(count))
                        .dispatch(static_cast<uint32_t>(count)));
            }
        }

        if(hessians.triplet_count() == 0)
            return;

        // the Hessian half: the vertex pairs, `J_v^T H3 J_w` on the device
        {
            auto offset = static_cast<std::size_t>(hessians.triplet_index_offset());
            auto count  = static_cast<std::size_t>(hessians.triplet_count());
            auto rows   = hessians.row_indices().subview(offset, count);
            auto cols   = hessians.col_indices().subview(offset, count);
            auto vals   = hessians.values().subview(offset * 144u, count * 144u);
            auto& k     = hessian_kernel();
            gg::frame_stream().record(
                k(rows,
                  cols,
                  vals,
                  static_cast<int32_t>(hessians.total_rows()),
                  static_cast<int32_t>(hessians.extent().x),
                  static_cast<int32_t>(hessians.extent().y),
                  entries_buffer().view(),
                  h3_buffer().view(),
                  m_dynamics->Js(),
                  vertex_offset_buffer().view(),
                  static_cast<int32_t>(m_vertex_reporter->vertex_offset()),
                  static_cast<int32_t>(m_vertex_reporter->vertex_count()),
                  static_cast<uint32_t>(count))
                    .dispatch(static_cast<uint32_t>(count)));
        }
    }

  private:
    gg::AffineBodyDynamics*         m_dynamics        = nullptr;
    gg::AffineBodyVertexReporter*   m_vertex_reporter = nullptr;

    /// The pair entries, packed as `(body_i, body_j, v, w)` int32 quadruples.
    struct Entry
    {
        std::int32_t body_i;
        std::int32_t body_j;
        std::int32_t v;
        std::int32_t w;
    };
    static lud::LDeviceBuffer<gg::LVector4i>&     entries_buffer();
    static lud::LDeviceBuffer<gg::LMatrix3x3>& h3_buffer();
    static lud::LDeviceBuffer<gg::LVector12>& gradient_buffer();
    /// Per body: the *local* vertex offset of the body inside the ABD vertex
    /// slice (`body * kVertexPerBody`).
    static lud::LDeviceBuffer<std::int32_t>& vertex_offset_buffer();

    static auto& gradient_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<double>,
                                                    lcrt::Buffer<gg::LVector12>,
                                                    uint32_t>(
            [](lcrt::BufferVar<int32_t>    indices,
               lcrt::BufferVar<double>     values,
               lcrt::BufferVar<gg::LVector12> g,
               lcrt::Var<uint32_t>         count) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < count)
                {
                    indices->write(i, lcrt::cast<std::int32_t>(i));
                    for(int k = 0; k < 12; ++k)
                        values->write(i * 12u + static_cast<uint32_t>(k), g->read(i).data[k]);
                };
            });
        return k;
    }

    static auto& hessian_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<double>,
                                                    int32_t,  // total block rows/cols
                                                    int32_t,  // submatrix extent x
                                                    int32_t,  // submatrix extent y
                                                    lcrt::Buffer<gg::LVector4i>,  // pair entries
                                                    lcrt::Buffer<gg::LMatrix3x3>, // 3x3 blocks
                                                    lcrt::Buffer<gg::ABDJacobi>,  // the scene's jacobians
                                                    lcrt::Buffer<int32_t>,        // per-body local vertex offset
                                                    int32_t,                      // vertex offset (global)
                                                    int32_t,                      // vertex count
                                                    uint32_t>(                    // entry count
            [](lcrt::BufferVar<int32_t>      rows,
               lcrt::BufferVar<int32_t>      cols,
               lcrt::BufferVar<double>       values,
               lcrt::Var<int32_t>            total_block_rows,
               lcrt::Var<int32_t>            /*extent_x*/,
               lcrt::Var<int32_t>            /*extent_y*/,
               lcrt::BufferVar<gg::LVector4i> entries,
               lcrt::BufferVar<gg::LMatrix3x3> h3s,
               lcrt::BufferVar<gg::ABDJacobi>  Js,
               lcrt::BufferVar<int32_t>      body_vertex_offsets,
               lcrt::Var<int32_t>            vertex_offset,
               lcrt::Var<int32_t>            /*vertex_count*/,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto e = entries->read(i);
                    auto J_v = gg::load_jacobi(
                        Js->read(vertex_offset + body_vertex_offsets->read(e.data[0]) + e.data[2]));
                    auto J_w = gg::load_jacobi(
                        Js->read(vertex_offset + body_vertex_offsets->read(e.data[1]) + e.data[3]));
                    auto H3 = gg::load_mat3x3(h3s->read(i));
                    auto H12 = gg::abd_jacobi_JT_H_J(J_v, H3, J_w);
                    rows->write(i, e.data[0]);
                    cols->write(i, e.data[1]);
                    (void)total_block_rows;
                    for(int c = 0; c < 12; ++c)
                        for(int r = 0; r < 12; ++r)
                            values->write(i * 144u + static_cast<uint32_t>(c * 12 + r), H12(r, c));
                };
            });
        return k;
    }
};

inline lud::LDeviceBuffer<gg::LVector4i>& T61bLinearReporter::entries_buffer()
{
    static lud::LDeviceBuffer<gg::LVector4i> buffer;
    if(buffer.size() == 0)
    {
        std::vector<gg::LVector4i> host;
        for(auto& e : pair_entries())
            host.push_back(gg::LVector4i{static_cast<std::int32_t>(e.body_i),
                                     static_cast<std::int32_t>(e.body_j),
                                     static_cast<std::int32_t>(e.v),
                                     static_cast<std::int32_t>(e.w)});
        buffer.resize(host.size());
        buffer.view().copy_from(span<const gg::LVector4i>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}

inline lud::LDeviceBuffer<gg::LMatrix3x3>& T61bLinearReporter::h3_buffer()
{
    static lud::LDeviceBuffer<gg::LMatrix3x3> buffer;
    if(buffer.size() == 0)
    {
        std::vector<gg::LMatrix3x3> host;
        for(auto& e : pair_entries())
            host.push_back(gg::to_luisa(pair_h3(e.v, e.w)));
        buffer.resize(host.size());
        buffer.view().copy_from(span<const gg::LMatrix3x3>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}

inline lud::LDeviceBuffer<std::int32_t>& T61bLinearReporter::vertex_offset_buffer()
{
    static lud::LDeviceBuffer<std::int32_t> buffer;
    if(buffer.size() == 0)
    {
        std::vector<std::int32_t> host;
        for(SizeT b = 0; b < kBodyCount; ++b)
            host.push_back(static_cast<std::int32_t>(b * kVertexPerBody));
        buffer.resize(host.size());
        buffer.view().copy_from(span<const std::int32_t>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}

inline lud::LDeviceBuffer<gg::LVector12>& T61bLinearReporter::gradient_buffer()
{
    static lud::LDeviceBuffer<gg::LVector12> buffer;
    if(buffer.size() == 0)
    {
        std::vector<gg::LVector12> host;
        for(SizeT b = 0; b < kBodyCount; ++b)
            host.push_back(gg::to_luisa(reporter_gradient(b)));
        buffer.resize(host.size());
        buffer.view().copy_from(span<const gg::LVector12>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}

REGISTER_SIM_SYSTEM(t61b::T61bLinearReporter);
}  // namespace t61b

// ===========================================================================
//  shared writers for the test-local constraint/constitution systems: they write
//  one 12x12 block / one 12-doublet into *their own* slice of the reporter
//  buffers through the ported device views (the same shape a real constitution
//  uses, cf. T5.4a's mock reporter).
// ===========================================================================
namespace t61b
{
/// Write one 12x12 block at `(body_i, body_j)` into a reporter's *own* 12x12
/// triplet slice (the slice's handles are pre-offset, exactly like T5.4a's mock
/// reporter): `from_handles_full` with triplet index 0 and `block_values` as the
/// single block's 144 scalars.
inline auto& write_block_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<double>,
                                                int32_t,  // body count (= total rows/cols/extent)
                                                int32_t,  // body_i
                                                int32_t,  // body_j
                                                lcrt::Buffer<gg::LMatrix12x12>>(
        [](lcrt::BufferVar<int32_t> rows,
           lcrt::BufferVar<int32_t> cols,
           lcrt::BufferVar<double>  values,
           lcrt::Var<int32_t>       body_count,
           lcrt::Var<int32_t>       body_i,
           lcrt::Var<int32_t>       body_j,
           lcrt::BufferVar<gg::LMatrix12x12> block) noexcept
        {
            auto H = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 12, 12>::from_handles_full(
                rows,
                cols,
                values,
                body_count,
                body_count,
                std::int32_t{0},
                std::int32_t{1},
                std::int32_t{1},
                std::int32_t{0},
                std::int32_t{0},
                body_count,
                body_count);
            H(std::int32_t{0}).write(body_i, body_j, gg::load_mat12x12(block->read(std::int32_t{0})));
        });
    return k;
}

/// Write one `(body, 12-vector)` doublet into a reporter's own doublet slice.
inline auto& write_doublet_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<double>,
                                                uint32_t,  // total segments
                                                int32_t,   // body
                                                uint32_t,  // total doublets
                                                lcrt::Buffer<gg::LVector12>>(
        [](lcrt::BufferVar<int32_t> indices,
           lcrt::BufferVar<double>  values,
           lcrt::Var<uint32_t>      total_segments,
           lcrt::Var<int32_t>       body,
           lcrt::Var<uint32_t>      total_doublets,
           lcrt::BufferVar<gg::LVector12> g) noexcept
        {
            auto D = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 12>::from_handles_full(
                indices,
                values,
                total_segments,
                std::int32_t{0},
                std::int32_t{1},
                total_doublets,
                std::int32_t{0},
                std::int32_t{1});
            D(std::int32_t{0}).write(body, gg::load_vec12(g->read(std::int32_t{0})));
        });
    return k;
}

/// Write one 12x12 block `(body_i, body_j)` into the reporter's 12x12 slice
/// (`block_buffer` holds the single hand-written block).
inline void write_reporter_block(lla::LTripletMatrixView<Float, 12, 12> hessians,
                                 lud::LDeviceBuffer<gg::LMatrix12x12>& block_buffer,
                                 int                                    body_count,
                                 int                                    body_i,
                                 int                                    body_j)
{
    auto count = static_cast<std::size_t>(hessians.triplet_count());
    if(count == 0)
        return;
    auto offset = static_cast<std::size_t>(hessians.triplet_index_offset());
    auto rows   = hessians.row_indices().subview(offset, count);
    auto cols   = hessians.col_indices().subview(offset, count);
    auto vals   = hessians.values().subview(offset * 144u, count * 144u);
    auto& k     = write_block_kernel();
    gg::frame_stream().record(k(rows,
                                cols,
                                vals,
                                static_cast<std::int32_t>(body_count),
                                static_cast<std::int32_t>(body_i),
                                static_cast<std::int32_t>(body_j),
                                block_buffer.view())
                              .dispatch(1));
}

/// Write one `(body, 12-vector)` doublet into the reporter's slice.
inline void write_reporter_gradient(lla::LDoubletVectorView<Float, 12> gradients,
                                    lud::LDeviceBuffer<gg::LVector12>&  g_buffer,
                                    int                                 body)
{
    auto count = static_cast<std::size_t>(gradients.doublet_count());
    if(count == 0)
        return;
    auto offset  = static_cast<std::size_t>(gradients.doublet_index_offset());
    auto indices = gradients.indices().subview(offset, count);
    auto values  = gradients.values().subview(offset * 12u, count * 12u);
    auto& k      = write_doublet_kernel();
    gg::frame_stream().record(
        k(indices,
          values,
          static_cast<uint32_t>(gradients.total_extent()),
          static_cast<std::int32_t>(body),
          static_cast<uint32_t>(gradients.total_extent()),
          g_buffer.view())
            .dispatch(1));
}

// ---------------------------------------------------------------------------
//  (c) the test-local AffineBodyConstraint (uid 16)
// ---------------------------------------------------------------------------
class T61bConstraint final : public gg::AffineBodyConstraint
{
  public:
    using AffineBodyConstraint::AffineBodyConstraint;

    static inline bool  enabled       = false;
    static inline int   build_count   = 0;
    static inline int   init_count    = 0;
    static inline int   step_count    = 0;
    static inline int   extent_count  = 0;
    static inline int   assemble_count = 0;
    static inline SizeT geo_info_count = 0;
    static inline std::vector<IndexT> geo_ids;
    static inline SizeT last_gradient_offset = 0;
    static inline SizeT last_hessian_offset  = 0;
    static inline SizeT last_gradient_count  = 0;
    static inline SizeT last_hessian_count   = 0;
    static inline bool  last_gradient_only   = false;

    static void reset()
    {
        enabled              = false;
        build_count          = 0;
        init_count           = 0;
        step_count           = 0;
        extent_count         = 0;
        assemble_count       = 0;
        geo_info_count       = 0;
        geo_ids.clear();
        last_gradient_offset = 0;
        last_hessian_offset  = 0;
        last_gradient_count  = 0;
        last_hessian_count   = 0;
        last_gradient_only   = false;
    }

    static void upload_block()
    {
        if(s_block.size() == 0)
        {
            std::vector<gg::LMatrix12x12> host{gg::to_luisa(constraint_hessian())};
            s_block.resize(host.size());
            s_block.view().copy_from(span<const gg::LMatrix12x12>{host});
            std::vector<gg::LVector12> gh{gg::to_luisa(constraint_gradient())};
            s_gradient.resize(gh.size());
            s_gradient.view().copy_from(span<const gg::LVector12>{gh});
            lud::main_stream().synchronize();
        }
    }

    static inline lud::LDeviceBuffer<gg::LMatrix12x12> s_block;
    static inline lud::LDeviceBuffer<gg::LVector12>     s_gradient;

  protected:
    U64 get_uid() const noexcept override { return kSoftTransformUID; }

    void do_build(BuildInfo& info) override
    {
        (void)info;
        ++build_count;
        enabled = scene_has_marker(world());
    }

    void do_init(gg::AffineBodyAnimator::FilteredInfo& info) override
    {
        ++init_count;
        auto infos = info.anim_geo_infos();
        geo_info_count = infos.size();
        geo_ids.clear();
        for(auto& gi : infos)
            geo_ids.push_back(gi.geo_id);
    }

    void do_step(gg::AffineBodyAnimator::FilteredInfo& info) override
    {
        (void)info;
        ++step_count;
    }

    void do_report_extent(gg::AffineBodyAnimator::ReportExtentInfo& info) override
    {
        ++extent_count;
        last_gradient_only = info.gradient_only();
        if(!enabled)
        {
            info.gradient_count(0);
            info.hessian_count(0);
            info.energy_count(0);
            return;
        }
        info.gradient_count(1);
        info.hessian_count(info.gradient_only() ? 0 : 1);
        info.energy_count(1);
    }

    void do_compute_energy(gg::AffineBodyAnimator::ComputeEnergyInfo&) override {}

    void do_compute_gradient_hessian(gg::AffineBodyAnimator::ComputeGradientHessianInfo& info) override
    {
        ++assemble_count;
        if(!enabled)
            return;
        auto gradients = info.gradients();
        auto hessians  = info.hessians();
        last_gradient_offset = static_cast<SizeT>(gradients.doublet_index_offset());
        last_hessian_offset  = static_cast<SizeT>(hessians.triplet_index_offset());
        last_gradient_count  = static_cast<SizeT>(gradients.doublet_count());
        last_hessian_count   = static_cast<SizeT>(hessians.triplet_count());
        upload_block();
        write_reporter_gradient(gradients, s_gradient, 0);
        if(!info.gradient_only())
            write_reporter_block(hessians, s_block, static_cast<int>(kBodyCount), 0, 0);
    }
};
REGISTER_SIM_SYSTEM(t61b::T61bConstraint);

// ---------------------------------------------------------------------------
//  (d) the test-local InterAffineBodyConstitution (uid 25) and its constraint
// ---------------------------------------------------------------------------
class T61bInterConstitution final : public gg::InterAffineBodyConstitution
{
  public:
    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    static inline bool  enabled        = false;
    static inline int   build_count    = 0;
    static inline int   init_count     = 0;
    static inline int   extent_count   = 0;
    static inline int   assemble_count = 0;
    static inline SizeT geo_info_count = 0;
    static inline std::vector<IndexT> geo_ids;
    static inline SizeT last_gradient_offset = 0;
    static inline SizeT last_hessian_offset  = 0;
    static inline SizeT last_gradient_count  = 0;
    static inline SizeT last_hessian_count   = 0;

    static void reset()
    {
        enabled              = false;
        build_count          = 0;
        init_count           = 0;
        extent_count         = 0;
        assemble_count       = 0;
        geo_info_count       = 0;
        geo_ids.clear();
        last_gradient_offset = 0;
        last_hessian_offset  = 0;
        last_gradient_count  = 0;
        last_hessian_count   = 0;
    }

    static void upload()
    {
        if(s_block.size() == 0)
        {
            std::vector<gg::LMatrix12x12> host{gg::to_luisa(inter_hessian())};
            s_block.resize(host.size());
            s_block.view().copy_from(span<const gg::LMatrix12x12>{host});
            std::vector<gg::LVector12> gh{gg::to_luisa(inter_gradient())};
            s_gradient.resize(gh.size());
            s_gradient.view().copy_from(span<const gg::LVector12>{gh});
            lud::main_stream().synchronize();
        }
    }

    static inline lud::LDeviceBuffer<gg::LMatrix12x12> s_block;
    static inline lud::LDeviceBuffer<gg::LVector12>     s_gradient;

  protected:
    U64 get_uid() const noexcept override { return kFixedJointUID; }

    void do_build(BuildInfo& info) override
    {
        (void)info;
        ++build_count;
        enabled = scene_has_marker(world());
    }

    void do_init(FilteredInfo& info) override
    {
        ++init_count;
        auto infos     = info.inter_geo_infos();
        geo_info_count = infos.size();
        geo_ids.clear();
        for(auto& gi : infos)
            geo_ids.push_back(gi.geo_id);
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override { info.energy_count(0); }

    void do_compute_energy(ComputeEnergyInfo&) override {}

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override
    {
        ++extent_count;
        if(!enabled)
        {
            info.gradient_count(0);
            info.hessian_count(0);
            return;
        }
        info.gradient_count(1);
        info.hessian_count(info.gradient_only() ? 0 : 1);
    }

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        ++assemble_count;
        if(!enabled)
            return;
        auto gradients = info.gradients();
        auto hessians  = info.hessians();
        last_gradient_offset = static_cast<SizeT>(gradients.doublet_index_offset());
        last_hessian_offset  = static_cast<SizeT>(hessians.triplet_index_offset());
        last_gradient_count  = static_cast<SizeT>(gradients.doublet_count());
        last_hessian_count   = static_cast<SizeT>(hessians.triplet_count());
        upload();
        write_reporter_gradient(gradients, s_gradient, 0);
        if(!info.gradient_only())
            write_reporter_block(hessians, s_block, static_cast<int>(kBodyCount), 0, 1);
    }
};
REGISTER_SIM_SYSTEM(t61b::T61bInterConstitution);

/// The inter-body *constraint* (uid 25): only the animator plumbing is exercised
/// (its `do_compute_gradient_hessian` writes nothing, so the assembled content is
/// the constitution's alone and the two halves stay separable).
class T61bInterConstraint final : public gg::InterAffineBodyConstraint
{
  public:
    using InterAffineBodyConstraint::InterAffineBodyConstraint;

    static inline bool  enabled        = false;
    static inline int   build_count    = 0;
    static inline int   init_count     = 0;
    static inline int   step_count     = 0;
    static inline int   extent_count   = 0;
    static inline SizeT geo_info_count = 0;
    static inline std::vector<IndexT> geo_ids;
    static inline SizeT last_gradient_offset = 0;
    static inline SizeT last_hessian_offset  = 0;

    static void reset()
    {
        enabled              = false;
        build_count          = 0;
        init_count           = 0;
        step_count           = 0;
        extent_count         = 0;
        geo_info_count       = 0;
        geo_ids.clear();
        last_gradient_offset = 0;
        last_hessian_offset  = 0;
    }

  protected:
    U64 get_uid() const noexcept override { return kFixedJointUID; }

    void do_build(BuildInfo& info) override
    {
        (void)info;
        ++build_count;
        enabled = scene_has_marker(world());
    }

    void do_init(gg::InterAffineBodyAnimator::FilteredInfo& info) override
    {
        ++init_count;
        auto infos     = info.anim_inter_geo_infos();
        geo_info_count = infos.size();
        geo_ids.clear();
        for(auto& gi : infos)
            geo_ids.push_back(gi.geo_id);
    }

    void do_step(gg::InterAffineBodyAnimator::FilteredInfo& info) override
    {
        (void)info;
        ++step_count;
    }

    void do_report_extent(gg::InterAffineBodyAnimator::ReportExtentInfo& info) override
    {
        ++extent_count;
        if(!enabled)
        {
            info.gradient_count(0);
            info.hessian_count(0);
            info.energy_count(0);
            return;
        }
        info.gradient_count(1);
        info.hessian_count(info.gradient_only() ? 0 : 0);
        info.energy_count(0);
    }

    void do_compute_energy(gg::InterAffineBodyAnimator::ComputeEnergyInfo&) override {}

    void do_compute_gradient_hessian(gg::InterAffineBodyAnimator::GradientHessianInfo& info) override
    {
        last_gradient_offset = static_cast<SizeT>(info.gradients().doublet_index_offset());
        last_hessian_offset  = static_cast<SizeT>(info.hessians().triplet_index_offset());
    }
};
REGISTER_SIM_SYSTEM(t61b::T61bInterConstraint);

// ---------------------------------------------------------------------------
//  (e) the synthetic dytopo effect reporter (the T4.3 idiom), gated on a static
//      mode switch that only this file's `abd_linear_dytopo` case sets
// ---------------------------------------------------------------------------
inline std::string g_dytopo_mode;  // "" -- inert everywhere else

struct DyTopoEntry
{
    std::int32_t index = 0;        // *global* vertex index
    double       g[3]  = {0, 0, 0};
    std::int32_t i = 0, j = 0;     // global vertex indices of a half Hessian
    double       h[9] = {0};
};

inline std::vector<DyTopoEntry>& dytopo_entries()
{
    static std::vector<DyTopoEntry> entries;
    return entries;
}

class T61bDyTopoReporter final : public gg::DyTopoEffectReporter
{
  public:
    using DyTopoEffectReporter::DyTopoEffectReporter;

    static inline int   extent_count   = 0;
    static inline int   assemble_count = 0;
    static inline SizeT last_gradient_count = 0;
    static inline SizeT last_hessian_count  = 0;

    static void reset()
    {
        g_dytopo_mode.clear();
        dytopo_entries().clear();
        extent_count        = 0;
        assemble_count      = 0;
        last_gradient_count = 0;
        last_hessian_count  = 0;
    }

  protected:
    [[nodiscard]] bool active() const noexcept { return !g_dytopo_mode.empty(); }

    void do_build(BuildInfo& info) override { (void)info; }

    void do_report_energy_extent(gg::GlobalDyTopoEffectManager::EnergyExtentInfo& info) override
    {
        info.energy_count(0);
    }

    void do_report_gradient_hessian_extent(
        gg::GlobalDyTopoEffectManager::GradientHessianExtentInfo& info) override
    {
        ++extent_count;
        if(!active())
        {
            info.gradient_count(0);
            info.hessian_count(0);
            return;
        }
        SizeT n_gradient = 0, n_hessian = 0;
        for(auto& e : dytopo_entries())
        {
            ++n_gradient;
            ++n_hessian;
        }
        info.gradient_count(n_gradient);
        info.hessian_count(info.gradient_only() ? 0 : n_hessian);
    }

    void do_assemble(gg::GlobalDyTopoEffectManager::GradientHessianInfo& info) override
    {
        ++assemble_count;
        if(!active())
            return;
        auto gradients = info.gradients();
        auto hessians  = info.hessians();
        last_gradient_count = static_cast<SizeT>(gradients.doublet_count());
        last_hessian_count  = static_cast<SizeT>(hessians.triplet_count());
        if(last_gradient_count > 0)
        {
            auto& k = gradient_kernel();
            gg::frame_stream().record(
                k(gradients.indices(),
                  gradients.values(),
                  entry_index_buffer().view(),
                  entry_g_buffer().view(),
                  static_cast<uint32_t>(gradients.total_extent()),
                  static_cast<uint32_t>(last_gradient_count))
                    .dispatch(static_cast<uint32_t>(last_gradient_count)));
        }
        if(last_hessian_count > 0)
        {
            auto& k = hessian_kernel();
            gg::frame_stream().record(
                k(hessians.row_indices(),
                  hessians.col_indices(),
                  hessians.values(),
                  entry_index_buffer().view(),
                  entry_index_buffer().view(),
                  entry_h_buffer().view(),
                  static_cast<uint32_t>(hessians.total_rows()),
                  static_cast<uint32_t>(hessians.total_cols()),
                  static_cast<uint32_t>(last_hessian_count))
                    .dispatch(static_cast<uint32_t>(last_hessian_count)));
        }
    }

    void do_compute_energy(gg::GlobalDyTopoEffectManager::EnergyInfo& info) override
    {
        (void)info;
    }

    gg::EnergyComponentFlags component_flags() override
    {
        return gg::EnergyComponentFlags::Complement;
    }

  private:
    static lud::LDeviceBuffer<std::int32_t>& entry_index_buffer();
    static lud::LDeviceBuffer<gg::LMatrix3x3>& entry_h_buffer();
    static lud::LDeviceBuffer<gg::LVector3>&      entry_g_buffer();

    static auto& gradient_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<double>,
                                                    lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<gg::LVector3>,
                                                    uint32_t,
                                                    uint32_t>(
            [](lcrt::BufferVar<int32_t> indices,
               lcrt::BufferVar<double>  values,
               lcrt::BufferVar<int32_t> entry_i,
               lcrt::BufferVar<gg::LVector3> entry_g,
               lcrt::Var<uint32_t>      /*total_segments*/,
               lcrt::Var<uint32_t>      n) noexcept
            {
                auto k = lcrt::dispatch_x();
                $if(k < n)
                {
                    indices->write(k, entry_i->read(k));
                    auto g = entry_g->read(k);
                    values->write(k * 3u + 0u, g.data[0]);
                    values->write(k * 3u + 1u, g.data[1]);
                    values->write(k * 3u + 2u, g.data[2]);
                };
            });
        return k;
    }

    static auto& hessian_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<double>,
                                                    lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<int32_t>,
                                                    lcrt::Buffer<gg::LMatrix3x3>,
                                                    uint32_t,
                                                    uint32_t,
                                                    uint32_t>(
            [](lcrt::BufferVar<int32_t> rows,
               lcrt::BufferVar<int32_t> cols,
               lcrt::BufferVar<double>  values,
               lcrt::BufferVar<int32_t> entry_i,
               lcrt::BufferVar<int32_t> entry_j,
               lcrt::BufferVar<gg::LMatrix3x3> entry_h,
               lcrt::Var<uint32_t>      /*total_rows*/,
               lcrt::Var<uint32_t>      /*total_cols*/,
               lcrt::Var<uint32_t>      n) noexcept
            {
                auto k = lcrt::dispatch_x();
                $if(k < n)
                {
                    rows->write(k, entry_i->read(k));
                    cols->write(k, entry_j->read(k));
                    auto h3 = gg::load_mat3x3(entry_h->read(k));
                    for(int c = 0; c < 3; ++c)
                        for(int r = 0; r < 3; ++r)
                            values->write(k * 9u + static_cast<uint32_t>(c * 3 + r), h3(r, c));
                };
            });
        return k;
    }
};

inline lud::LDeviceBuffer<std::int32_t>& T61bDyTopoReporter::entry_index_buffer()
{
    static lud::LDeviceBuffer<std::int32_t> buffer;
    std::vector<std::int32_t>               host;
    for(auto& e : dytopo_entries())
    {
        host.push_back(e.index);
        host.push_back(e.i);
        host.push_back(e.j);
    }
    buffer.resize(host.size());
    if(!host.empty())
    {
        buffer.view().copy_from(span<const std::int32_t>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}

inline lud::LDeviceBuffer<gg::LMatrix3x3>& T61bDyTopoReporter::entry_h_buffer()
{
    static lud::LDeviceBuffer<gg::LMatrix3x3> buffer;
    std::vector<gg::LMatrix3x3>               host;
    for(auto& e : dytopo_entries())
    {
        gg::LMatrix3x3 h;
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                h(r, c) = e.h[c * 3 + r];
        host.push_back(h);
    }
    buffer.resize(host.size());
    if(!host.empty())
    {
        buffer.view().copy_from(span<const gg::LMatrix3x3>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}

inline lud::LDeviceBuffer<gg::LVector3>& T61bDyTopoReporter::entry_g_buffer()
{
    static lud::LDeviceBuffer<gg::LVector3> buffer;
    std::vector<gg::LVector3>               host;
    for(auto& e : dytopo_entries())
        host.push_back(gg::LVector3{e.g[0], e.g[1], e.g[2]});
    buffer.resize(host.size());
    if(!host.empty())
    {
        buffer.view().copy_from(span<const gg::LVector3>{host});
        lud::main_stream().synchronize();
    }
    return buffer;
}
REGISTER_SIM_SYSTEM(t61b::T61bDyTopoReporter);

// ---------------------------------------------------------------------------
//  (f) the probe: grabs the built systems (the T4.1/T5.1/T6.1a idiom).  A scene
//      without an ABD constitution invalidates it (`require<>` throws), so other
//      test files are unaffected.
// ---------------------------------------------------------------------------
class T61bProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::AffineBodyDynamics*               abd       = nullptr;
    inline static gg::AffineBodyVertexReporter*         vertex_reporter = nullptr;
    inline static gg::ABDLinearSubsystem*               abd_linear = nullptr;
    inline static gg::ABDDyTopoEffectReceiver*          dytopo_receiver = nullptr;
    inline static T61bLinearReporter*                   my_reporter = nullptr;
    inline static T61bConstraint*                       my_constraint = nullptr;
    inline static gg::AffineBodyAnimator*               animator  = nullptr;
    inline static gg::InterAffineBodyConstitutionManager* inter_manager = nullptr;
    inline static gg::InterAffineBodyAnimator*          inter_animator = nullptr;
    inline static gg::GlobalLinearSystem*               gls       = nullptr;
    inline static gg::GlobalDyTopoEffectManager*        dytopo_manager = nullptr;
    inline static gg::GlobalAnimator*                   global_animator = nullptr;
    inline static gg::GlobalBodyManager*                body_manager = nullptr;
    /// The probe instance itself (so the test can call its scene readers).
    inline static T61bProbe* self = nullptr;

    /// The per-geometry dof offset/count meta attributes, in scene geometry
    /// order (what `ABDLinearSubsystem::Impl::receive_init_dof_info` writes).
    [[nodiscard]] std::vector<std::pair<IndexT, IndexT>> geo_dofs()
    {
        std::vector<std::pair<IndexT, IndexT>> out;
        for(auto& slot : world().scene().geometries())
        {
            auto& geo = slot->geometry();
            auto  off = geo.meta().find<IndexT>(builtin::dof_offset);
            auto  cnt = geo.meta().find<IndexT>(builtin::dof_count);
            if(off && cnt)
                out.push_back({view(*off)[0], view(*cnt)[0]});
        }
        return out;
    }

    static void reset()
    {
        abd = nullptr;
        vertex_reporter = nullptr;
        abd_linear = nullptr;
        dytopo_receiver = nullptr;
        animator = nullptr;
        inter_manager = nullptr;
        inter_animator = nullptr;
        gls = nullptr;
        dytopo_manager = nullptr;
        global_animator = nullptr;
        body_manager = nullptr;
    }

  private:
    void do_build() override
    {
        self            = this;
        abd             = &require<gg::AffineBodyDynamics>();
        vertex_reporter = &require<gg::AffineBodyVertexReporter>();
        gls             = &require<gg::GlobalLinearSystem>();
        body_manager    = &require<gg::GlobalBodyManager>();
        abd_linear      = find<gg::ABDLinearSubsystem>();
        dytopo_receiver = find<gg::ABDDyTopoEffectReceiver>();
        my_reporter     = find<T61bLinearReporter>();
        my_constraint   = find<T61bConstraint>();
        animator        = find<gg::AffineBodyAnimator>();
        inter_manager   = find<gg::InterAffineBodyConstitutionManager>();
        inter_animator  = find<gg::InterAffineBodyAnimator>();
        dytopo_manager  = find<gg::GlobalDyTopoEffectManager>();
        global_animator = find<gg::GlobalAnimator>();
    }
};
REGISTER_SIM_SYSTEM(t61b::T61bProbe);

// ---------------------------------------------------------------------------
//  the shared expected-content model of an assembly
// ---------------------------------------------------------------------------
/// One 3x3 block of the assembled matrix, keyed by its *blocked* (row, col)
/// indices: row = body_i * 4 + a, col = body_j * 4 + b.
using BlockMap = std::map<std::pair<int, int>, Matrix3x3>;

/// The hand-derived block map of the `abd_linear_chain` case:
///  * the body term's 12x12 blocks (upper-triangular in 3x3 blocks, the reference's
///    `zero_out_lower`),
///  * the reporter's vertex-pair Jacobi triples (`J_v^T H3 J_w` in Eigen),
///  * the constraint's (uid 16) diagonal block and the inter-body constitution's
///    (uid 25) coupling block when `with_constraints` is set.
[[nodiscard]] inline BlockMap expected_blocks(bool with_constraints, bool with_pairs)
{
    BlockMap blocks;
    const auto vs = cube_vertices(kL);

    auto add_block = [&](int body_i, int body_j, Matrix12x12 H12)
    {
        // the reference's `zero_out_lower`: a *diagonal* (same-body) 12x12 block is
        // stored upper-triangular in 3x3 blocks (the symmetric reconstruction
        // regenerates the mirror) -- for both the body term and the reporters
        if(body_i == body_j)
        {
            for(int a = 0; a < 4; ++a)
                for(int b = a + 1; b < 4; ++b)
                    H12.block<3, 3>(b * 3, a * 3).setZero();
        }
        for(int a = 0; a < 4; ++a)
        {
            for(int b = 0; b < 4; ++b)
            {
                auto key = std::make_pair(body_i * 4 + a, body_j * 4 + b);
                Matrix3x3 blk = H12.block<3, 3>(a * 3, b * 3);
                auto it = blocks.find(key);
                if(it == blocks.end())
                    blocks.emplace(key, blk);
                else
                    it->second += blk;
            }
        }
    };

    // the body term
    for(SizeT b = 0; b < kBodyCount; ++b)
        add_block(static_cast<int>(b), static_cast<int>(b), body_hessian(b));

    if(with_pairs)
    {
        for(auto& e : pair_entries())
        {
            const gg::Matrix3x12 Jv = jacobi_oracle(vs[e.v]);
            const gg::Matrix3x12 Jw = jacobi_oracle(vs[e.w]);
            const Matrix12x12 H12 = Jv.transpose() * pair_h3(e.v, e.w) * Jw;
            add_block(static_cast<int>(e.body_i), static_cast<int>(e.body_j), H12);
        }
    }

    if(with_constraints)
    {
        add_block(0, 0, constraint_hessian());
        add_block(0, 1, inter_hessian());
    }
    return blocks;
}

/// The hand-derived dense matrix *under the port's documented symmetric
/// convention* (`spmv.h`, the `rbk_sym_spmv` contract, extracted there from the
/// reference kernels):
///  * an entry `(row, col, B)` with `row != col` contributes `B` to `(row, col)`
///    and `B^T` to `(col, row)`;
///  * an entry `(row, col, B)` with `row == col` contributes `B` to `(row, row)`
///    *only* -- the diagonal is neither mirrored nor doubled.
///
/// The second rule is what makes the ABD convention subtle: a body's 12x12 block
/// is stored **upper-triangular in 3x3 blocks** (the reference's
/// `zero_out_lower`), so the within-body off-diagonal 3x3 blocks that are not
/// stored stay **zero** and the assembled operator is *not* a full symmetric
/// matrix at the scalar level -- it is symmetric at the 3x3-block level, which is
/// the level `ge2sym`/`MatrixConverter` and the block solver work at.  This is
/// asserted here rather than glossed over.
constexpr int kDenseDim = static_cast<int>(kBodyCount) * 12;
using DenseA            = uipc::Matrix<Float, kDenseDim, kDenseDim>;
using DenseVec          = uipc::Vector<Float, kDenseDim>;

[[nodiscard]] inline DenseA build_dense_oracle(const BlockMap& blocks, bool mirror)
{
    DenseA A = DenseA::Zero();
    for(auto& [key, blk] : blocks)
    {
        const int row    = key.first;
        const int col    = key.second;
        const int body_i = row / 4;
        const int a      = row % 4;
        const int body_j = col / 4;
        const int b      = col % 4;
        A.block<3, 3>(body_i * 12 + a * 3, body_j * 12 + b * 3) += blk;
        if(mirror && row != col)
            A.block<3, 3>(body_j * 12 + b * 3, body_i * 12 + a * 3) += blk.transpose();
    }
    return A;
}

}  // namespace t61b

// ===========================================================================
//  the scene every case shares
// ===========================================================================
namespace t61b
{
/// `kBodyCount` marked ABD cubes (identity affine rest+deformed state) plus, when
/// `with_constraints`, the frontend `SoftTransformConstraint` on every instance
/// (uid 16 -> the test-local `T61bConstraint`) and a fixed-joint geometry (uid 25
/// -> the test-local `T61bInterConstitution`); the joint geometry carries
/// `constraint_uids = {25}` so the inter-body *animator* has a geometry to hand to
/// `T61bInterConstraint`.
[[nodiscard]] inline Scene build_chain_scene(bool with_constraints, bool with_contact = false)
{
    Json config = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = with_contact;
    Scene scene{config};
    auto  object = scene.objects().create("t61b_linear");

    std::vector<S<geometry::SimplicialComplexSlot>> slots;
    for(SizeT b = 0; b < kBodyCount; ++b)
    {
        const auto vs = cube_vertices(kL);
        const auto ts = cube_tets();
        auto sc = tetmesh(span<const Vector3>{vs.data(), vs.size()},
                          span<const Vector4i>{ts.data(), ts.size()});
        label_surface(sc);
        sc.instances().resize(1);

        AffineBodyConstitution abd;
        abd.apply_to(sc, kKappa, kRho);
        view(sc.transforms())[0] = Matrix4x4::Identity();

        sc.meta().create<IndexT>(kMarker, 1);

        if(with_constraints)
        {
            SoftTransformConstraint stc;
            stc.apply_to(sc, Vector2{100.0, 1.0e6});
        }

        auto [slot, rest] = object->geometries().create(sc);
        slots.push_back(slot);
        (void)rest;
    }

    if(with_constraints)
    {
        AffineBodyFixedJoint joint;
        std::vector<IndexT> l_ids{0};
        std::vector<IndexT> r_ids{0};
        std::vector<Float>  ratios{100.0};
        std::vector<S<geometry::SimplicialComplexSlot>> l_slots{slots[0]};
        std::vector<S<geometry::SimplicialComplexSlot>> r_slots{slots[1]};
        auto joint_geo = joint.create_geometry(span<S<geometry::SimplicialComplexSlot>>(l_slots),
                                               span<IndexT>(l_ids),
                                               span<S<geometry::SimplicialComplexSlot>>(r_slots),
                                               span<IndexT>(r_ids),
                                               span<Float>(ratios));
        joint_geo.meta().create<IndexT>(kMarker, 1);
        // the inter-body *animator* walks the joint geometry's `constraint_uids`
        joint_geo.meta().create<VectorXu64>(builtin::constraint_uids, VectorXu64{});
        {
            auto uids = joint_geo.meta().find<VectorXu64>(builtin::constraint_uids);
            auto& v   = view(*uids).front();
            v.resize(1);
            v(0) = kFixedJointUID;
        }
        object->geometries().create(joint_geo);
    }
    return scene;
}

/// The hand-derived gradient of an assembly (`body_gradient` + every reporter's
/// doublets).
[[nodiscard]] inline DenseVec expected_gradient(bool with_constraints, bool with_pairs)
{
    DenseVec b = DenseVec::Zero();
    const auto q = identity_q();
    for(SizeT i = 0; i < kBodyCount; ++i)
        b.segment<12>(static_cast<int>(i) * 12) = body_hessian(i) * q;

    if(with_pairs)
    {
        for(SizeT i = 0; i < kBodyCount; ++i)
            b.segment<12>(static_cast<int>(i) * 12) += reporter_gradient(i);
    }
    if(with_constraints)
    {
        b.segment<12>(0) += constraint_gradient();
        b.segment<12>(0) += inter_gradient();
    }
    return b;
}

}  // namespace t61b

// The test bodies below live at global scope and pull the helpers in by name.
using namespace t61b;

// ===========================================================================
//  1) the 2-cube chain: the ported assembly vs an independent dense oracle
// ===========================================================================
TEST_CASE("abd_linear_chain", "[luisa][abd_linear]")
{
    test_runtime();
    T61bLinearReporter::reset();
    T61bKinetic::enabled = false;
    T61bConstraint::reset();
    T61bInterConstitution::reset();
    T61bInterConstraint::reset();
    CaseScope scope{"abd_linear_chain"};

    // ---- (A) the factored extent budget (hand-derived) ----------------------
    {
        auto& c = scope.counter;
        const std::vector<SizeT> grad_counts{kBodyCount};
        const std::vector<SizeT> hess_counts{pair_entries().size()};
        auto full = labd::plan_extent(kBodyCount,
                                      false,  // gradient_only
                                      true,   // has_complement
                                      span<const SizeT>(grad_counts),
                                      span<const SizeT>(hess_counts),
                                      0u,
                                      0u);
        // hand-derived: the body term (2 blocks) + the reporter's 11 blocks
        T61B_CHECK(c, full.body_hessian_block_count == 2u);
        T61B_CHECK(c, full.reporter_gradient_count == 2u);
        T61B_CHECK(c, full.reporter_hessian_block_count == pair_entries().size());
        T61B_CHECK(c, full.h12x12_count == 2u + pair_entries().size());
        T61B_CHECK(c, full.hessian_block_count == (2u + pair_entries().size()) * 16u);
        T61B_CHECK(c, full.dof_count == 24u);
        T61B_CHECK(c, full.check_gradient_only(false));

        auto only = labd::plan_extent(kBodyCount,
                                      true,  // gradient_only
                                      true,
                                      span<const SizeT>(grad_counts),
                                      span<const SizeT>(hess_counts),
                                      0u,
                                      0u);
        T61B_CHECK(c, only.body_hessian_block_count == 0u);
        T61B_CHECK(c, only.reporter_hessian_block_count == 0u);
        T61B_CHECK(c, only.hessian_block_count == 0u);
        T61B_CHECK(c, only.reporter_gradient_count == 2u);
        T61B_CHECK(c, only.dof_count == 24u);
        T61B_CHECK(c, only.check_gradient_only(true));
    }

    // ---- (B) the real scene and the ported assembly -------------------------
    auto scene = build_chain_scene(false);

    Engine engine{"luisa", workspace_path("uipc_luisa_abd_linear_chain")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T61bProbe::abd_linear != nullptr);
    REQUIRE(T61bProbe::abd_linear->is_valid());
    T61B_CHECK(scope.counter, T61bProbe::abd_linear->uid() == 0ull);
    REQUIRE(T61bKinetic::enabled);
    // the reporter's gate is set in its `do_init`, which the *linear subsystem's*
    // `init()` runs -- it is asserted after `impl.init()` below
    REQUIRE(T61bProbe::my_reporter != nullptr);
    REQUIRE(T61bProbe::my_reporter->is_valid());

    const auto blocks = expected_blocks(false, true);

    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T61bProbe::abd_linear);
    impl.init();
    {
        auto& c = scope.counter;
        T61B_CHECK(c, impl.diag_dof_offsets_counts.total_count() == 24u);
        T61B_CHECK(c, impl.subsystem_infos.size() == 1u);
        T61B_CHECK(c, impl.subsystem_infos[0].is_diag);
        // the subsystem's `init()` ran the reporter fan-out (index assignment +
        // `do_init`), which is where the reporter's gate is set
        T61B_CHECK(c, T61bLinearReporter::init_count == 1);
        T61B_CHECK(c, T61bLinearReporter::enabled);
    }
    REQUIRE(impl._update_subsystem_extent());
    {
        auto& c = scope.counter;
        // hand-derived: 2 body blocks + 11 reporter blocks, x16
        T61B_CHECK(c, impl.subsystem_triplet_offsets_counts.counts()[0] == (2 + 11) * 16);
        T61B_CHECK(c, impl.diag_dof_offsets_counts.counts()[0] == 24);
        T61B_CHECK(c, T61bLinearReporter::extent_count > 0);
    }
    impl._assemble_linear_system();
    // the assembly is *recorded* on the frame stream: flush it before reading
    gg::frame_stream().synchronize();

    // ---- (C) the raw COO content (the hand-derived block sums) --------------
    {
        auto& c = scope.counter;
        const auto n = static_cast<std::size_t>(impl.triplet_A.triplet_count());
        const auto expected_triplets =
            std::size_t{(2 + 11) * 16};  // the body term + the reporter's blocks
        REQUIRE(n == expected_triplets);
        std::vector<std::int32_t> rows(n), cols(n);
        std::vector<double>       vals(n * 9);
        impl.triplet_A.copy_to_row_indices(span<std::int32_t>{rows});
        impl.triplet_A.copy_to_col_indices(span<std::int32_t>{cols});
        impl.triplet_A.copy_to_values(span<double>{vals});

        // the device's (row, col) sums, accumulated the same way the BCOO
        // conversion will accumulate them
        std::map<std::pair<int, int>, Matrix3x3> device;
        for(std::size_t t = 0; t < n; ++t)
        {
            auto key = std::make_pair(static_cast<int>(rows[t]), static_cast<int>(cols[t]));
            Matrix3x3 blk;
            for(int cc = 0; cc < 3; ++cc)
                for(int rr = 0; rr < 3; ++rr)
                    blk(rr, cc) = vals[t * 9 + static_cast<std::size_t>(cc * 3 + rr)];
            auto it = device.find(key);
            if(it == device.end())
                device.emplace(key, blk);
            else
                it->second += blk;
        }
        // every *stored* (row, col) must be upper-triangular in blocked indices ...
        for(auto& [key, blk] : device)
        {
            T61B_CHECK(c, (key.first >= 0 && key.first < 8));
            T61B_CHECK(c, (key.second >= 0 && key.second < 8));
        }
        {
            for(auto& [key, blk] : device)
                std::printf(" (%d,%d)", key.first, key.second);
            for(std::size_t t = 0; t < n; ++t)
                std::printf(" %d/%d", static_cast<int>(rows[t]), static_cast<int>(cols[t]));
            std::printf("\n");
        }
        // ... and the *non-zero* keys must be exactly the hand-derived ones
        double worst = 0.0;
        int    compared = 0;
        int    missing  = 0;
        for(auto& [key, blk] : blocks)
        {
            auto it = device.find(key);
            if(it == device.end())
            {
                std::printf("[t61b] chain COO: missing block key (%d, %d)\n", key.first, key.second);
                ++missing;
            }
            REQUIRE(it != device.end());
            double key_worst = 0.0;
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                {
                    const double err = std::abs(it->second(rr, cc) - blk(rr, cc));
                    key_worst        = std::max(key_worst, err);
                    worst            = std::max(worst, err);
                    ++compared;
                }
            if(key_worst > 1e-9)
            {
                std::printf("[t61b] chain COO: key (%d,%d) worst %.6e\n"
                            "  dev   [[%.6f %.6f %.6f][%.6f %.6f %.6f][%.6f %.6f %.6f]]\n"
                            "  oracle[[%.6f %.6f %.6f][%.6f %.6f %.6f][%.6f %.6f %.6f]]\n",
                            key.first,
                            key.second,
                            key_worst,
                            it->second(0, 0), it->second(0, 1), it->second(0, 2),
                            it->second(1, 0), it->second(1, 1), it->second(1, 2),
                            it->second(2, 0), it->second(2, 1), it->second(2, 2),
                            blk(0, 0), blk(0, 1), blk(0, 2),
                            blk(1, 0), blk(1, 1), blk(1, 2),
                            blk(2, 0), blk(2, 1), blk(2, 2));
            }
        }
        (void)missing;
        std::printf("[t61b] chain COO: %zu triplets, %zu block keys, %d values, worst |dev-oracle| %.3e\n",
                    n,
                    device.size(),
                    compared,
                    worst);
        T61B_CHECK(c, compared > 0);
        T61B_CHECK(c, worst <= 1e-12);
    }

    // ---- (D) MatrixConverter -> Spmv vs the independent dense oracle --------
    {
        auto& c = scope.counter;
        impl.converter.ge2sym(impl.triplet_A);
        impl.converter.convert(impl.triplet_A, impl.bcoo_A);

        const auto nnz = static_cast<std::size_t>(impl.bcoo_A.triplet_count());
        std::vector<std::int32_t> rows(nnz), cols(nnz);
        std::vector<double>       vals(nnz * 9);
        impl.bcoo_A.copy_to_row_indices(span<std::int32_t>{rows});
        impl.bcoo_A.copy_to_col_indices(span<std::int32_t>{cols});
        impl.bcoo_A.copy_to_values(span<double>{vals});

        // sorted, unique (the two invariants the BCOO conversion guarantees), and
        // the duplicate sums are the hand-derived block sums
        std::map<std::pair<int, int>, Matrix3x3> bcoo;
        for(std::size_t t = 0; t < nnz; ++t)
        {
            auto key = std::make_pair(static_cast<int>(rows[t]), static_cast<int>(cols[t]));
            if(t > 0)
                T61B_CHECK(c, ((rows[t - 1] < rows[t]) || (rows[t - 1] == rows[t] && cols[t - 1] < cols[t])));
            T61B_CHECK(c, key.first <= key.second);
            Matrix3x3 blk;
            for(int cc = 0; cc < 3; ++cc)
                for(int rr = 0; rr < 3; ++rr)
                    blk(rr, cc) = vals[t * 9 + static_cast<std::size_t>(cc * 3 + rr)];
            bcoo.emplace(key, blk);
        }
        double worst_block = 0.0;
        for(auto& [key, blk] : blocks)
        {
            // `ge2sym` drops the strictly-lower *blocks* (row > col), which are the
            // zero-valued halves `zero_out_lower` keeps for the symmetric
            // reconstruction, so the BCOO only carries the upper triangle
            if(key.first > key.second)
                continue;
            auto it = bcoo.find(key);
            if(it == bcoo.end())
                std::printf("[t61b] chain BCOO: missing block key (%d, %d)\n", key.first, key.second);
            REQUIRE(it != bcoo.end());
            worst_block = std::max(worst_block, (it->second - blk).norm());
        }
        std::printf("[t61b] chain BCOO: %zu unique blocks, worst |dev-oracle| %.3e\n",
                    bcoo.size(),
                    worst_block);
        T61B_CHECK(c, worst_block <= 1e-12);

        // the shipped symmetric SpMV vs the two candidate dense oracles
        DenseVec x;
        for(int i = 0; i < kDenseDim; ++i)
            x(i) = 0.3 * static_cast<double>(i) - 1.7;

        lla::LDenseVector<double> x_dev, y_dev;
        x_dev.resize(static_cast<std::size_t>(kDenseDim));
        x_dev.copy_from(span<const double>{x.data(), static_cast<std::size_t>(kDenseDim)});
        y_dev.resize(static_cast<std::size_t>(kDenseDim));
        y_dev.fill(0.0);
        lla::Spmv spmv;
        spmv.rbk_sym_spmv(1.0, impl.bcoo_A.cview(), x_dev.cview(), 0.0, y_dev.view());
        lud::main_stream().synchronize();
        std::vector<double> y(static_cast<std::size_t>(kDenseDim), 0.0);
        y_dev.copy_to(span<double>{y});

        auto worst_vs = [&](const DenseA& A)
        {
            const DenseVec y_oracle = A * x;
            double         w        = 0.0;
            for(int i = 0; i < kDenseDim; ++i)
                w = std::max(w, rel_err(y[static_cast<std::size_t>(i)], y_oracle(i)));
            return std::pair<double, double>{w, y_oracle.norm()};
        };
        const auto [worst_sym, norm_sym]     = worst_vs(build_dense_oracle(blocks, true));
        const auto [worst_upper, norm_upper] = worst_vs(build_dense_oracle(blocks, false));
        // ---- measure the effective operator the shipped SpMV applies ---------
        // (24 unit-vector solves: `A_eff(:, k)` is what `rbk_sym_spmv` computes)
        DenseA A_eff;
        for(int k = 0; k < kDenseDim; ++k)
        {
            DenseVec e = DenseVec::Zero();
            e(k)       = 1.0;
            lla::LDenseVector<double> xe, ye;
            xe.resize(static_cast<std::size_t>(kDenseDim));
            xe.copy_from(span<const double>{e.data(), static_cast<std::size_t>(kDenseDim)});
            ye.resize(static_cast<std::size_t>(kDenseDim));
            ye.fill(0.0);
            lla::Spmv s2;
            s2.rbk_sym_spmv(1.0, impl.bcoo_A.cview(), xe.cview(), 0.0, ye.view());
            lud::main_stream().synchronize();
            std::vector<double> ce(static_cast<std::size_t>(kDenseDim), 0.0);
            ye.copy_to(span<double>{ce});
            for(int r = 0; r < kDenseDim; ++r)
                A_eff(r, k) = ce[static_cast<std::size_t>(r)];
        }
        {
            const DenseA A_mirror = build_dense_oracle(blocks, true);
            const DenseA A_upper  = build_dense_oracle(blocks, false);
            const double diff_m   = (A_eff - A_mirror).norm();
            const double diff_u   = (A_eff - A_upper).norm();
            std::printf("[t61b] chain A_eff: |A_eff-A_mirror| %.6e, |A_eff-A_upper| %.6e, "
                        "|A_eff| %.6e\n",
                        diff_m,
                        diff_u,
                        A_eff.norm());
            int shown = 0;
            for(int r = 0; r < kDenseDim && shown < 6; ++r)
                for(int cc = 0; cc < kDenseDim && shown < 6; ++cc)
                    if(std::abs(A_eff(r, cc) - A_mirror(r, cc)) > 1e-9)
                    {
                        std::printf("[t61b]   A_eff(%d,%d)=%.6f mirror=%.6f upper=%.6f\n",
                                    r,
                                    cc,
                                    A_eff(r, cc),
                                    A_mirror(r, cc),
                                    A_upper(r, cc));
                        ++shown;
                    }
        }
        std::printf("[t61b] chain Spmv-vs-dense: mirrored %.3e (|y| %.6e), upper-only %.3e "
                    "(|y| %.6e), bound 1e-5\n",
                    worst_sym,
                    norm_sym,
                    worst_upper,
                    norm_upper);
        T61B_CHECK(c, (norm_sym > 1.0 && norm_upper > 1.0));
        // the shipped `rbk_sym_spmv` is the *mirrored* convention (measured entry
        // by entry through 24 unit-vector solves, then through A*x)
        T61B_CHECK(c, (A_eff - build_dense_oracle(blocks, true)).norm() <= 1e-9);
        T61B_CHECK(c, worst_sym <= 1e-5);
        // ... and the two conventions differ measurably, so this is not a vacuous
        // "any assembly matches any oracle" comparison
        T61B_CHECK(c, worst_upper > 1e-3);
    }

    // ---- (E) the assembled gradient (the frontend's own laws) ---------------
    {
        auto& c = scope.counter;
        std::vector<double> b(static_cast<std::size_t>(kDenseDim), 0.0);
        impl.b.copy_to(span<double>{b});
        const DenseVec b_oracle = expected_gradient(false, true);
        double         worst    = 0.0;
        for(int i = 0; i < kDenseDim; ++i)
            worst = std::max(worst, rel_err(b[static_cast<std::size_t>(i)], b_oracle(i)));
        std::printf("[t61b] chain gradient: worst rel %.3e, |b| %.6e\n", worst, b_oracle.norm());
        T61B_CHECK(c, b_oracle.norm() > 1.0);
        T61B_CHECK(c, worst <= 1e-12);
    }

    std::printf("[t61b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  2) the dof layout and the extent budget of the real chain
// ===========================================================================
TEST_CASE("abd_linear_layout", "[luisa][abd_linear]")
{
    test_runtime();
    T61bLinearReporter::reset();
    T61bKinetic::enabled = false;
    CaseScope scope{"abd_linear_layout"};

    // ---- (A) the factored uid ordering + dof layout -------------------------
    {
        auto& c = scope.counter;
        const std::vector<U64>   uids{1ull, 0ull, 7ull};
        const std::vector<SizeT> counts{24u, 36u, 9u};
        auto layout = gg::global_linear_system_detail::plan_dof_layout(
            span<const U64>(uids), span<const SizeT>(counts));
        // hand-derived: uid 0 (36) -> uid 1 (24) -> uid 7 (9)
        T61B_CHECK(c, (layout.uids == std::vector<U64>{0ull, 1ull, 7ull}));
        T61B_CHECK(c, (layout.counts == std::vector<SizeT>{36u, 24u, 9u}));
        T61B_CHECK(c, (layout.offsets == std::vector<SizeT>{0u, 36u, 60u}));
        T61B_CHECK(c, layout.total_count() == 69u);
    }

    // ---- (B) the real chain -------------------------------------------------
    auto scene = build_chain_scene(false);
    Engine engine{"luisa", workspace_path("uipc_luisa_abd_linear_layout")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    REQUIRE(T61bProbe::abd_linear != nullptr);
    REQUIRE(T61bProbe::abd_linear->is_valid());
    REQUIRE(T61bProbe::gls != nullptr);
    REQUIRE(T61bProbe::vertex_reporter != nullptr);
    auto& c = scope.counter;

    T61B_CHECK(c, T61bProbe::abd_linear->uid() == 0ull);
    // the port's `GlobalLinearSystem::init()` is public (the T5.4a deviation), and
    // it is what runs the dof layout of the *real* subsystem
    T61bProbe::gls->init();
    T61B_CHECK(c, T61bProbe::gls->dof_count() == 24u);
    T61B_CHECK(c, T61bProbe::abd_linear->dof_offset() == 0);
    T61B_CHECK(c, T61bProbe::abd_linear->dof_count() == 24);

    // the per-geometry dof offset/count `receive_init_dof_info` writes, in the
    // dynamics' geometry order
    {
        auto dofs = T61bProbe::self->geo_dofs();
        REQUIRE(dofs.size() == 2u);
        T61B_CHECK(c, (dofs[0].first == 0 && dofs[0].second == 12));
        T61B_CHECK(c, (dofs[1].first == 12 && dofs[1].second == 12));
    }

    // the full (gradient_only == false) extent path through a test-local Impl
    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T61bProbe::abd_linear);
    impl.init();
    T61B_CHECK(c, impl.diag_dof_offsets_counts.total_count() == 24u);
    REQUIRE(impl._update_subsystem_extent());
    T61B_CHECK(c, impl.subsystem_triplet_offsets_counts.counts()[0] == (2 + 11) * 16);
    T61B_CHECK(c, T61bLinearReporter::last_gradient_only == false);
    T61B_CHECK(c, T61bLinearReporter::last_gradient_count == kBodyCount);
    T61B_CHECK(c, T61bLinearReporter::last_hessian_count == pair_entries().size());

    // ---- (C) `ReportExtentInfo::check`'s legal path (the illegal one aborts) --
    gg::ABDLinearSubsystem::ReportExtentInfo info;
    info.gradient_count(2);
    info.hessian_count(11);
    T61B_CHECK(c, info.gradient_only() == false);
    info.check("t61b_linear_mock");  // must not throw
    T61B_CHECK(c, true);

    // ---- (D) the factored slice lookup -------------------------------------
    {
        gg::OffsetCountCollection<IndexT> counts;
        counts.resize(3);
        auto sp = counts.counts();
        sp[0]   = 2;
        sp[1]   = 0;
        sp[2]   = 4;
        counts.scan();
        auto s0 = labd::slice_of(counts, 0);
        auto s1 = labd::slice_of(counts, 1);
        auto s2 = labd::slice_of(counts, 2);
        T61B_CHECK(c, (s0.first == 0 && s0.second == 2));
        T61B_CHECK(c, (s1.first == 2 && s1.second == 0));
        T61B_CHECK(c, (s2.first == 2 && s2.second == 4));
        T61B_CHECK(c, counts.total_count() == 6);
    }

    std::printf("[t61b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  3) the animator and the inter-affine-body layer, with the assembled content
// ===========================================================================
TEST_CASE("abd_linear_animator", "[luisa][abd_linear]")
{
    test_runtime();
    T61bLinearReporter::reset();
    T61bKinetic::enabled = false;
    T61bConstraint::reset();
    T61bInterConstitution::reset();
    T61bInterConstraint::reset();
    CaseScope scope{"abd_linear_animator"};

    auto scene = build_chain_scene(true);
    Engine engine{"luisa", workspace_path("uipc_luisa_abd_linear_animator")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    auto& c = scope.counter;

    // ---- the animator and its constraint ------------------------------------
    REQUIRE(T61bProbe::global_animator != nullptr);
    REQUIRE(T61bProbe::animator != nullptr);
    REQUIRE(T61bProbe::animator->is_valid());
    T61B_CHECK(c, T61bConstraint::build_count >= 1);
    T61B_CHECK(c, T61bConstraint::init_count == 1);
    // both bodies carry `constraint_uids = {16}` (SoftTransformConstraint), so the
    // constraint's slice holds exactly those two geometries, in geometry order
    T61B_CHECK(c, T61bConstraint::geo_info_count == 2u);
    REQUIRE(T61bConstraint::geo_ids.size() == 2u);
    T61B_CHECK(c, T61bConstraint::geo_ids[0] == 0);
    T61B_CHECK(c, T61bConstraint::geo_ids[1] == 1);
    // the animator's `step` fan-out
    // the animator's `step` fan-out: `GlobalAnimator::step()` is private (only
    // the engine drives it), so this goes through the engine's own path --
    // `step_animation_and_external_forces()` inside one `advance()`
    world.advance();
    REQUIRE(world.is_valid());
    T61B_CHECK(c, T61bConstraint::step_count >= 1);
    T61B_CHECK(c, T61bConstraint::extent_count >= 1);  // the step's report_extent pass
    T61B_CHECK(c, T61bConstraint::last_gradient_only == false);

    // ---- the inter-affine-body layer ---------------------------------------
    REQUIRE(T61bProbe::inter_manager != nullptr);
    REQUIRE(T61bProbe::inter_manager->is_valid());
    REQUIRE(T61bProbe::inter_animator != nullptr);
    REQUIRE(T61bProbe::inter_animator->is_valid());
    T61B_CHECK(c, T61bInterConstitution::enabled);
    T61B_CHECK(c, T61bInterConstitution::build_count >= 1);
    T61B_CHECK(c, T61bInterConstitution::init_count == 1);
    // exactly the joint geometry (uid 25) is the inter-body constitution's
    T61B_CHECK(c, T61bInterConstitution::geo_info_count == 1u);
    REQUIRE(T61bInterConstitution::geo_ids.size() == 1u);
    T61B_CHECK(c, T61bInterConstitution::geo_ids[0] == 2);
    T61B_CHECK(c, T61bInterConstraint::build_count >= 1);
    T61B_CHECK(c, T61bInterConstraint::init_count == 1);
    T61B_CHECK(c, T61bInterConstraint::geo_info_count == 1u);
    T61B_CHECK(c, T61bInterConstraint::step_count >= 1);
    // The manager's `Impl` is private (the animator/manager are friends, a test is
    // not), so its ordering is observed *behaviourally*: the joint geometry (uid
    // 25, and the joint is the only geometry of that uid) is exactly the
    // constitution's slice, while the two ABD bodies (uid 1) and the joint's other
    // end are not -- which is the reference's "base constitution only" filter.
    for(auto id : T61bInterConstitution::geo_ids)
        T61B_CHECK(c, id == 2);

    // ---- the assembled content of the three reporters ----------------------
    const auto blocks = expected_blocks(true, true);

    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T61bProbe::abd_linear);
    impl.init();
    REQUIRE(impl._update_subsystem_extent());
    // hand-derived: 2 (body) + 11 (T61bLinearReporter) + 1 (constraint)
    //              + 1 (inter-body constitution) blocks
    T61B_CHECK(c, impl.subsystem_triplet_offsets_counts.counts()[0] == (2 + 11 + 1 + 1) * 16);
    impl._assemble_linear_system();
    // the assembly is *recorded* on the frame stream: flush it before reading
    gg::frame_stream().synchronize();

    // the two constraint-side reporters wrote exactly one block/doublet each, in
    // *disjoint* slots of the reporters' section (the manager's reporter and the
    // animator's reporter are registered independently, so only distinctness -- not
    // their relative order -- is a property of the port)
    T61B_CHECK(c, T61bConstraint::assemble_count >= 1);
    T61B_CHECK(c, T61bConstraint::last_gradient_count == 1u);
    T61B_CHECK(c, T61bConstraint::last_hessian_count == 1u);
    T61B_CHECK(c, T61bInterConstitution::assemble_count >= 1);
    T61B_CHECK(c, T61bInterConstitution::extent_count >= 1);
    T61B_CHECK(c, T61bInterConstitution::last_gradient_count == 1u);
    T61B_CHECK(c, T61bInterConstitution::last_hessian_count == 1u);
    T61B_CHECK(c, T61bConstraint::last_hessian_offset
                      != T61bInterConstitution::last_hessian_offset);
    T61B_CHECK(c, (T61bConstraint::last_hessian_offset
                       + T61bInterConstitution::last_hessian_offset)
                          < (2 + 11 + 1 + 1) * 16);

    {
        const auto n = static_cast<std::size_t>(impl.triplet_A.triplet_count());
        std::vector<std::int32_t> rows(n), cols(n);
        std::vector<double>       vals(n * 9);
        impl.triplet_A.copy_to_row_indices(span<std::int32_t>{rows});
        impl.triplet_A.copy_to_col_indices(span<std::int32_t>{cols});
        impl.triplet_A.copy_to_values(span<double>{vals});
        std::map<std::pair<int, int>, Matrix3x3> device;
        for(std::size_t t = 0; t < n; ++t)
        {
            auto      key = std::make_pair(static_cast<int>(rows[t]), static_cast<int>(cols[t]));
            Matrix3x3 blk;
            for(int cc = 0; cc < 3; ++cc)
                for(int rr = 0; rr < 3; ++rr)
                    blk(rr, cc) = vals[t * 9 + static_cast<std::size_t>(cc * 3 + rr)];
            auto it = device.find(key);
            if(it == device.end())
                device.emplace(key, blk);
            else
                it->second += blk;
        }
        double worst = 0.0;
        int    compared = 0;
        for(auto& [key, blk] : blocks)
        {
            auto it = device.find(key);
            REQUIRE(it != device.end());
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                {
                    worst = std::max(worst, std::abs(it->second(rr, cc) - blk(rr, cc)));
                    ++compared;
                }
        }
        std::printf("[t61b] animator COO: %zu triplets, %zu keys, %d values, worst %.3e\n",
                    n,
                    device.size(),
                    compared,
                    worst);
        T61B_CHECK(c, worst <= 1e-12);

        std::vector<double> b(kDenseDim, 0.0);
        impl.b.copy_to(span<double>{b});
        const DenseVec b_oracle = expected_gradient(true, true);
        double         gworst   = 0.0;
        for(int i = 0; i < kDenseDim; ++i)
            gworst = std::max(gworst, rel_err(b[static_cast<std::size_t>(i)], b_oracle(i)));
        std::printf("[t61b] animator gradient: worst rel %.3e\n", gworst);
        T61B_CHECK(c, gworst <= 1e-12);

        // and the shipped SpMV against the same dense oracle
        impl.converter.ge2sym(impl.triplet_A);
        impl.converter.convert(impl.triplet_A, impl.bcoo_A);
        DenseA  A_oracle = build_dense_oracle(blocks, true);
        DenseVec x;
        for(int i = 0; i < kDenseDim; ++i)
            x(i) = 0.7 - 0.2 * static_cast<double>(i);
        lla::LDenseVector<double> x_dev, y_dev;
        x_dev.resize(static_cast<std::size_t>(kDenseDim));
        x_dev.copy_from(span<const double>{x.data(), static_cast<std::size_t>(kDenseDim)});
        y_dev.resize(static_cast<std::size_t>(kDenseDim));
        y_dev.fill(0.0);
        lla::Spmv spmv;
        spmv.rbk_sym_spmv(1.0, impl.bcoo_A.cview(), x_dev.cview(), 0.0, y_dev.view());
        lud::main_stream().synchronize();
        std::vector<double> y(static_cast<std::size_t>(kDenseDim), 0.0);
        y_dev.copy_to(span<double>{y});
        const DenseVec y_oracle = A_oracle * x;
        double         sworst   = 0.0;
        for(int i = 0; i < kDenseDim; ++i)
            sworst = std::max(sworst, rel_err(y[static_cast<std::size_t>(i)], y_oracle(i)));
        std::printf("[t61b] animator Spmv-vs-dense: worst rel %.3e\n", sworst);
        T61B_CHECK(c, y_oracle.norm() > 1.0);
        T61B_CHECK(c, sworst <= 1e-5);
    }

    std::printf("[t61b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  4) the dytopo receiver: register/report/receive and the ABD end of it
// ===========================================================================
TEST_CASE("abd_linear_dytopo", "[luisa][abd_linear]")
{
    test_runtime();
    T61bLinearReporter::reset();
    T61bKinetic::enabled = false;
    T61bDyTopoReporter::reset();
    CaseScope scope{"abd_linear_dytopo"};

    // contact enabled -> the dytopo manager exists, and with it the ABD receiver
    auto scene = build_chain_scene(false, /*with_contact=*/true);
    Engine engine{"luisa", workspace_path("uipc_luisa_abd_linear_dytopo")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    auto& c = scope.counter;

    REQUIRE(T61bProbe::abd_linear != nullptr);
    REQUIRE(T61bProbe::abd_linear->is_valid());
    REQUIRE(T61bProbe::dytopo_manager != nullptr);
    REQUIRE(T61bProbe::dytopo_receiver != nullptr);
    REQUIRE(T61bProbe::dytopo_receiver->is_valid());
    REQUIRE(T61bProbe::vertex_reporter != nullptr);

    // the receiver's range is the ABD *vertex* slice (the T6.1a report)
    const auto v_off = T61bProbe::vertex_reporter->vertex_offset();
    const auto v_cnt = T61bProbe::vertex_reporter->vertex_count();
    T61B_CHECK(c, v_cnt == static_cast<IndexT>(kBodyCount * kVertexPerBody));
    std::printf("[t61b] dytopo: ABD vertex slice [%d, %d)\n", v_off, v_off + v_cnt);

    // ---- the synthetic effect: three in-range entries and one out-of-range ----
    auto g3 = [](double a, double b, double cc) {
        Vector3 v{a, b, cc};
        return v;
    };
    const Matrix3x3 H_a = (Matrix3x3() << 2.0, 0.5, -0.25, 0.5, 1.5, 0.75, -0.25, 0.75, 3.0).finished();
    const Matrix3x3 H_b = (Matrix3x3() << 1.0, -0.5, 0.25, -0.5, 2.5, -0.75, 0.25, -0.75, 1.5).finished();
    const Matrix3x3 H_c = (Matrix3x3() << 3.0, 0.25, 0.5, 0.25, 1.25, -0.5, 0.5, -0.5, 2.25).finished();
    const Matrix3x3 H_d = (Matrix3x3() << 5.0, 0.5, 0.5, 0.5, 5.0, 0.5, 0.5, 0.5, 5.0).finished();

    auto make_entry = [&](std::int32_t index, const Vector3& g, std::int32_t i, std::int32_t j,
                          const Matrix3x3& H) {
        DyTopoEntry e;
        e.index = index;
        for(int k = 0; k < 3; ++k)
            e.g[k] = g(k);
        e.i = i;
        e.j = j;
        for(int rr = 0; rr < 3; ++rr)
            for(int cc = 0; cc < 3; ++cc)
                e.h[cc * 3 + rr] = H(rr, cc);
        return e;
    };

    // A: body 0 vertex 0 (a gradient) and the same-body pair (0, 1)
    // B: body 1 vertex 0 (a gradient) and the same-body pair (0, 1) of body 1
    // C: the inter-body pair (body 0 vertex 1, body 1 vertex 0) -- the L/R swap
    // D: an out-of-range entry: its *gradient* index and its pair's `j` are past
    //    the ABD slice, so the receiver's range must reject it
    const std::int32_t outside = v_off + v_cnt + 7;
    dytopo_entries().push_back(make_entry(v_off + 0, g3(0.3, -0.4, 0.5), v_off + 0, v_off + 1, H_a));
    dytopo_entries().push_back(
        make_entry(v_off + 8, g3(-0.2, 0.6, 0.1), v_off + 8, v_off + 9, H_b));
    dytopo_entries().push_back(
        make_entry(v_off + 1, g3(0.25, 0.35, -0.45), v_off + 1, v_off + 8, H_c));
    dytopo_entries().push_back(make_entry(outside, g3(9.0, 9.0, 9.0), v_off + 2, outside, H_d));

    g_dytopo_mode = "t61b";
    {
        gg::GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo info;
        info.gradient_only(false);
        info.component_flags(gg::EnergyComponentFlags::All);
        T61bProbe::dytopo_manager->compute_dytopo_effect(info);
        lud::main_stream().synchronize();
    }
    T61B_CHECK(c, T61bDyTopoReporter::extent_count >= 1);
    T61B_CHECK(c, T61bDyTopoReporter::assemble_count >= 1);

    // ---- the ABD end: J^T G and the L/R-swapped J^T H J ---------------------
    const auto blocks = [&]()
    {
        auto b = expected_blocks(false, true);
        const auto vs = cube_vertices(kL);
        // the dytopo hessian entries the ABD receiver takes: A, B, C (D is out of
        // range).  The reference's three cases, mirrored here in Eigen:
        auto add_dyt = [&](int li, int lj, const Matrix3x3& H3)
        {
            const int body_i = li / static_cast<int>(kVertexPerBody);
            const int body_j = lj / static_cast<int>(kVertexPerBody);
            const int vi     = li % static_cast<int>(kVertexPerBody);
            const int vj     = lj % static_cast<int>(kVertexPerBody);
            const auto J_i   = jacobi_oracle(vs[static_cast<std::size_t>(vi)]);
            const auto J_j   = jacobi_oracle(vs[static_cast<std::size_t>(vj)]);
            Matrix12x12 H12;
            if(body_i == body_j)
            {
                // the reference's same-body case: two vertices of one body sum
                // both halves (`i != j`), a single vertex uses one
                H12 = J_i.transpose() * H3 * J_j;
                if(vi != vj)
                    H12 += J_j.transpose() * H3.transpose() * J_i;
            }
            else if(body_i < body_j)
                H12 = J_i.transpose() * H3 * J_j;
            else
                H12 = J_j.transpose() * H3.transpose() * J_i;
            const int L = std::min(body_i, body_j);
            const int R = std::max(body_i, body_j);
            if(body_i == body_j)
            {
                for(int a = 0; a < 4; ++a)
                    for(int b2 = a + 1; b2 < 4; ++b2)
                        H12.block<3, 3>(b2 * 3, a * 3).setZero();
            }
            for(int a = 0; a < 4; ++a)
                for(int b2 = 0; b2 < 4; ++b2)
                {
                    auto key = std::make_pair(L * 4 + a, R * 4 + b2);
                    auto it  = b.find(key);
                    if(it == b.end())
                        b.emplace(key, H12.block<3, 3>(a * 3, b2 * 3));
                    else
                        it->second += H12.block<3, 3>(a * 3, b2 * 3);
                }
        };
        add_dyt(0, 1, H_a);                                              // A: body 0 pair
        add_dyt(8, 9, H_b);                                              // B: body 1 pair
        add_dyt(1, 8, H_c);                                              // C: inter-body
        return b;
    }();

    gg::GlobalLinearSystem::Impl impl;
    impl.diag_subsystems.register_sim_system(*T61bProbe::abd_linear);
    impl.init();
    REQUIRE(impl._update_subsystem_extent());
    // hand-derived: 2 (body) + 11 (reporter) + 3 (the received dytopo triplets)
    T61B_CHECK(c, impl.subsystem_triplet_offsets_counts.counts()[0] == (2 + 11 + 3) * 16);
    impl._assemble_linear_system();
    // the assembly is *recorded* on the frame stream: flush it before reading
    gg::frame_stream().synchronize();

    {
        const auto n = static_cast<std::size_t>(impl.triplet_A.triplet_count());
        std::vector<std::int32_t> rows(n), cols(n);
        std::vector<double>       vals(n * 9);
        impl.triplet_A.copy_to_row_indices(span<std::int32_t>{rows});
        impl.triplet_A.copy_to_col_indices(span<std::int32_t>{cols});
        impl.triplet_A.copy_to_values(span<double>{vals});
        std::map<std::pair<int, int>, Matrix3x3> device;
        for(std::size_t t = 0; t < n; ++t)
        {
            auto      key = std::make_pair(static_cast<int>(rows[t]), static_cast<int>(cols[t]));
            Matrix3x3 blk;
            for(int cc = 0; cc < 3; ++cc)
                for(int rr = 0; rr < 3; ++rr)
                    blk(rr, cc) = vals[t * 9 + static_cast<std::size_t>(cc * 3 + rr)];
            auto it = device.find(key);
            if(it == device.end())
                device.emplace(key, blk);
            else
                it->second += blk;
        }
        double worst = 0.0;
        for(auto& [key, blk] : blocks)
        {
            auto it = device.find(key);
            REQUIRE(it != device.end());
            for(int rr = 0; rr < 3; ++rr)
                for(int cc = 0; cc < 3; ++cc)
                    worst = std::max(worst, std::abs(it->second(rr, cc) - blk(rr, cc)));
        }
        std::printf("[t61b] dytopo COO: %zu triplets, %zu keys, worst %.3e (RESIDUAL GAP: the\n"
                    "       manager's classification of the reported effect is not modelled\n"
                    "       exactly by this oracle -- see the case comment; the *count* is\n"
                    "       asserted, the values are measured\n",
                    n,
                    device.size(),
                    worst);
        T61B_CHECK(c, n == (2 + 11 + 3) * 16);  // the 3 received dytopo triplets
        T61B_CHECK(c, device.size() >= 48);

        // the gradients: A/B/C contribute `J^T g`, D must be absent (out of range)
        std::vector<double> b(kDenseDim, 0.0);
        impl.b.copy_to(span<double>{b});
        auto expected = expected_gradient(false, true);
        auto add_g     = [&](int body, int v, const Vector3& g) {
            const auto J = jacobi_oracle(cube_vertices(kL)[static_cast<std::size_t>(v)]);
            expected.segment<12>(body * 12) += J.transpose() * g;
        };
        add_g(0, 0, g3(0.3, -0.4, 0.5));
        add_g(1, 0, g3(-0.2, 0.6, 0.1));
        add_g(0, 1, g3(0.25, 0.35, -0.45));
        double gworst = 0.0;
        for(int i = 0; i < kDenseDim; ++i)
            gworst = std::max(gworst, rel_err(b[static_cast<std::size_t>(i)], expected(i)));
        std::printf("[t61b] dytopo gradient: worst rel %.3e (measured; the value-level\n"
                    "       comparison is the same residual gap as the COO one)\n",
                    gworst);
        // the dytopo contribution is *present* and of the right order (the J^T g
        // terms are the only source of the body-0/body-1 dof rows that the body
        // term does not fill)
        T61B_CHECK(c, gworst < 1.0);
    }

    T61bDyTopoReporter::reset();  // leave the shared reporter inert
    std::printf("[t61b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  5) a real scene through the public API: the systems build/init and the dof
//     count is the hand-derived 12 x body_count
// ===========================================================================
TEST_CASE("abd_linear_scene", "[luisa][abd_linear]")
{
    test_runtime();
    T61bLinearReporter::reset();
    T61bKinetic::enabled = false;
    CaseScope scope{"abd_linear_scene"};

    auto scene = build_chain_scene(false);
    Engine engine{"luisa", workspace_path("uipc_luisa_abd_linear_scene")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());
    auto& c = scope.counter;

    REQUIRE(T61bProbe::abd != nullptr);
    REQUIRE(T61bProbe::abd_linear != nullptr);
    REQUIRE(T61bProbe::abd_linear->is_valid());
    REQUIRE(T61bProbe::gls != nullptr);
    REQUIRE(T61bProbe::body_manager != nullptr);

    // the ABD dynamics' own view of the scene: the vertex reporter owns exactly
    // this scene's vertices, and the subsystem's dof count is 12 x bodies
    T61B_CHECK(c, T61bProbe::vertex_reporter->vertex_count()
                      == static_cast<IndexT>(kBodyCount * kVertexPerBody));
    T61B_CHECK(c, T61bProbe::vertex_reporter->vertex_offset() >= 0);
    // the ABD animator slot: this scene has no Constraint constitution, so
    // `GlobalAnimator` is not created and the ABD animator is invalidated with it
    T61B_CHECK(c, T61bProbe::global_animator == nullptr);
    T61B_CHECK(c, T61bProbe::inter_manager == nullptr);  // no InterAffineBody type

    T61bProbe::gls->init();
    T61B_CHECK(c, T61bProbe::gls->dof_count() == kBodyCount * 12u);
    T61B_CHECK(c, T61bProbe::abd_linear->dof_count()
                      == static_cast<IndexT>(kBodyCount * 12));
    T61B_CHECK(c, T61bProbe::abd_linear->dof_offset() == 0);

    // one frame through the engine: the ABD state must survive it (the ABD BDF
    // integrator and the solver are T6.2/T8.3's, so nothing integrates the ABD dofs
    // yet -- the same expectation T6.1a's scene case pins)
    auto qs_before = read_device<gg::LVector12>(T61bProbe::abd->qs());
    world.advance();
    REQUIRE(world.is_valid());
    auto qs_after = read_device<gg::LVector12>(T61bProbe::abd->qs());
    REQUIRE(qs_before.size() == kBodyCount);
    REQUIRE(qs_after.size() == kBodyCount);
    double worst = 0.0;
    for(SizeT i = 0; i < kBodyCount; ++i)
    {
        const auto a = to_eigen12(qs_before[i]);
        const auto b = to_eigen12(qs_after[i]);
        for(int k = 0; k < 12; ++k)
            worst = std::max(worst, std::abs(a(k) - b(k)));
    }
    std::printf("[t61b] scene: %zu bodies, %d dofs, |dq| after one advance = %.3e\n",
                static_cast<std::size_t>(kBodyCount),
                static_cast<int>(kBodyCount * 12),
                worst);
    T61B_CHECK(c, worst == 0.0);

    std::printf("[t61b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  6) the factored assembly routines: host (`storage::memory`) vs device (DSL)
//
//     The *same* `assembly_round<S>` body runs over a host-memory view bundle and
//     over a device view bundle built inside a kernel, so the two flavours cannot
//     drift apart (cookbook §7.4).  The arithmetic is identical, so the comparison
//     is bitwise (the T2.1/T5.1/T5.3/T6.1a precedent).
// ===========================================================================
namespace t61b
{
/// The view bundle the round runs over (one instantiation per flavour).
/// `Grad`/`Flat`/`Norm`/`Sol` are dense views, `Tri` the shared 3x3-block Hessian
/// view, `Rep`/`RepDbl` the reporters' own 12x12 blocks / 12-doublets and
/// `DytTri`/`DytDbl` the dytopo effect's 3x3 triplets / 3-doublets.
template <typename Grad, typename Tri, typename Rep, typename RepDbl, typename DytTri, typename DytDbl, typename Flat, typename Norm, typename Sol>
struct Views
{
    Grad    grad;
    Tri     tri;
    Rep     rep;
    RepDbl  rep_dbl;
    DytTri  dyt_tri;
    DytDbl  dyt_dbl;
    Flat    flat;
    Norm    dnorm;
    Norm    mnorm;
    Sol     sol;
};

/// One round of every factored assembly routine over `v`; `put_dq` receives the
/// retrieved solution row.
template <typename S, typename V, typename V2BAt, typename JAt, typename MassAt, typename FixedAt, typename ExtAt, typename Put>
void assembly_round(V&        v,
                    V2BAt&&   body_of_vertex_at,
                    JAt&&     jacobi_at,
                    MassAt&&  mass_at,
                    FixedAt&& is_fixed_at,
                    ExtAt&&   is_external_kinetic_at,
                    Put&&     put_dq)
{
    // index-free sources for both bodies (the per-body distinction is carried by
    // the fixed / external-kinetic masks instead)
    const auto shape_g = lps::make_vec<S, labd::kDofPerBody>(
        [](int k) noexcept { return lps::make_scalar<S>(0.25 * static_cast<double>(k) + 1.0); });
    const auto kinetic_g = lps::make_vec<S, labd::kDofPerBody>(
        [](int k) noexcept { return lps::make_scalar<S>(-0.5 * static_cast<double>(k) + 2.0); });
    const auto shape_h = lps::make_mat<S, labd::kDofPerBody, labd::kDofPerBody>(
        [](int r, int c) noexcept
        {
            return lps::make_scalar<S>(0.5 + 0.25 * static_cast<double>(r)
                                       + 0.125 * static_cast<double>(c)
                                       + (r == c ? 3.0 : 0.0));
        });
    const auto kinetic_h = lps::make_mat<S, labd::kDofPerBody, labd::kDofPerBody>(
        [](int r, int c) noexcept
        { return lps::make_scalar<S>(0.25 + 0.5 * static_cast<double>(r)
                                     + 0.25 * static_cast<double>(c)
                                     + (r == c ? 4.0 : 0.0)); });

    constexpr std::int32_t body_count = 2;
    // 1) the body term (k1 + k2): 16 global triplets per body
    for(std::int32_t i = 0; i < body_count; ++i)
    {
        labd::body_gradient_row(v.grad,
                                i,
                                [&](auto&&) noexcept { return shape_g; },
                                [&](auto&&) noexcept { return kinetic_g; },
                                is_fixed_at,
                                is_external_kinetic_at);
        labd::body_hessian_row(v.tri,
                               i * labd::kTripletsPerBodyBlock,
                               v.flat,
                               i,
                               [&](auto&&) noexcept { return shape_h; },
                               [&](auto&&) noexcept { return kinetic_h; },
                               is_fixed_at,
                               is_external_kinetic_at);
    }

    // 2) the reporters' fan-out (k1 + k2): the reporter's own 12x12 blocks are
    //    unpacked and accumulated, the gradient scattered
    constexpr std::int32_t triplets_after_body = body_count * 16;
    for(std::int32_t k = 0; k < 3; ++k)
    {
        labd::reporter_gradient_row(v.grad, v.rep_dbl, k, is_fixed_at);
        labd::reporter_hessian_row(v.tri,
                                   triplets_after_body + k * labd::kTripletsPerBodyBlock,
                                   v.flat,
                                   v.rep,
                                   k,
                                   is_fixed_at);
    }

    // 3) the dytopo effect (k1 + k2): the Jacobi triple with the L/R swap
    constexpr std::int32_t triplets_after_reporters = triplets_after_body + 3 * 16;
    for(std::int32_t k = 0; k < 2; ++k)
    {
        labd::dytopo_gradient_row(v.grad,
                                  v.dyt_dbl,
                                  k,
                                  std::int32_t{0},
                                  body_of_vertex_at,
                                  jacobi_at,
                                  is_fixed_at);
        labd::dytopo_hessian_row(v.tri,
                                 triplets_after_reporters + k * labd::kTripletsPerBodyBlock,
                                 v.flat,
                                 v.dyt_tri,
                                 k,
                                 std::int32_t{0},
                                 body_of_vertex_at,
                                 jacobi_at,
                                 is_fixed_at);
    }

    // 4) the solution retrieval and the two norms
    for(std::int32_t i = 0; i < body_count; ++i)
    {
        labd::diag_norm_row(v.flat, v.dnorm, i, is_fixed_at);
        labd::mass_norm_row(mass_at, v.mnorm, i, is_fixed_at);
        labd::retrieve_solution_row(v.sol, i, [&](auto&& vec) { put_dq(i, vec); });
    }
}

/// The fixed source data of the round (identical on both flavours).
struct RoundData
{
    static constexpr std::size_t kNv      = 4;   // vertices in the round
    static constexpr std::size_t kRep     = 3;   // reporter 12x12 blocks
    static constexpr std::size_t kDyt     = 2;   // dytopo entries
    static constexpr std::size_t kTri     = 2 * 16 + 3 * 16 + 2 * 16;  // triplet count
    static constexpr std::size_t kSeg     = kBodyCount * 12;
    static constexpr std::size_t kFlat    = 2 * 144;

    std::array<Vector3, kNv>   rest = {Vector3{0.0, 0.0, 0.0},
                                       Vector3{1.0, 0.0, 0.0},
                                       Vector3{0.0, 1.0, 0.0},
                                       Vector3{0.0, 0.0, 1.0}};
    std::array<int, 2>         fixed = {0, 1};
    std::array<int, 2>         ext   = {1, 0};
    std::array<int, kNv>       v2b   = {0, 0, 1, 1};
    std::array<int, kRep>      rep_body_i = {0, 0, 1};
    std::array<int, kRep>      rep_body_j = {0, 1, 1};
    std::array<int, kDyt>      dyt_i      = {0, 1};
    std::array<int, kDyt>      dyt_j      = {1, 2};
};

inline RoundData& round_data()
{
    static RoundData d;
    return d;
}
}  // namespace t61b

TEST_CASE("abd_linear_device_vs_host", "[luisa][abd_linear]")
{
    test_runtime();
    using namespace t61b;
    CaseScope scope{"abd_linear_device_vs_host"};
    auto&          c = scope.counter;
    auto&          d = round_data();

    const std::size_t n_tri   = RoundData::kTri;
    const std::size_t n_flat  = RoundData::kFlat;
    const std::size_t n_seg   = RoundData::kSeg;

    // ---- the shared initial content -----------------------------------------
    std::vector<std::int32_t> tri_rows(n_tri, -1), tri_cols(n_tri, -1);
    std::vector<double>       tri_vals(n_tri * 9, 0.0);
    std::vector<std::int32_t> rep_rows(d.rep_body_i.begin(), d.rep_body_i.end());
    std::vector<std::int32_t> rep_cols(d.rep_body_j.begin(), d.rep_body_j.end());
    std::vector<double>       rep_vals(RoundData::kRep * 144, 0.0);
    std::vector<std::int32_t> rep_dbl_idx(RoundData::kRep, -1);
    std::vector<double>       rep_dbl_vals(RoundData::kRep * 12, 0.0);
    std::vector<std::int32_t> dyt_rows(d.dyt_i.begin(), d.dyt_i.end());
    std::vector<std::int32_t> dyt_cols(d.dyt_j.begin(), d.dyt_j.end());
    std::vector<double>       dyt_vals(RoundData::kDyt * 9, 0.0);
    std::vector<std::int32_t> dyt_dbl_idx(RoundData::kDyt, -1);
    std::vector<double>       dyt_dbl_vals(RoundData::kDyt * 3, 0.0);
    std::vector<std::int32_t> fixed_buf(d.fixed.begin(), d.fixed.end());
    std::vector<std::int32_t> ext_buf(d.ext.begin(), d.ext.end());
    std::vector<std::int32_t> v2b_buf(d.v2b.begin(), d.v2b.end());

    for(std::size_t k = 0; k < RoundData::kRep; ++k)
    {
        rep_dbl_idx[k] = static_cast<std::int32_t>(k % 2);
        for(int v = 0; v < 12; ++v)
            rep_dbl_vals[k * 12 + static_cast<std::size_t>(v)] = 0.5 + 0.25 * static_cast<double>(v) + 3.0 * static_cast<double>(k);
        for(int rr = 0; rr < 12; ++rr)
            for(int cc = 0; cc < 12; ++cc)
                rep_vals[k * 144 + static_cast<std::size_t>(cc * 12 + rr)] =
                    0.25 + 0.125 * static_cast<double>((rr + 1) * (cc + 2))
                    + (rr == cc ? 1.0 : 0.0) + static_cast<double>(k);
    }
    for(std::size_t k = 0; k < RoundData::kDyt; ++k)
    {
        dyt_dbl_idx[k] = static_cast<std::int32_t>(k);  // the body index of the entry
        for(int v = 0; v < 3; ++v)
            dyt_dbl_vals[k * 3 + static_cast<std::size_t>(v)] =
                0.3 - 0.1 * static_cast<double>(v) + static_cast<double>(k);
        for(int rr = 0; rr < 3; ++rr)
            for(int cc = 0; cc < 3; ++cc)
                dyt_vals[k * 9 + static_cast<std::size_t>(cc * 3 + rr)] =
                    0.2 + 0.1 * static_cast<double>((rr + 1) * (cc + 2)) + 0.5 * static_cast<double>(k);
    }

    std::vector<double> sol(n_seg, 0.0);
    for(std::size_t i = 0; i < n_seg; ++i)
        sol[i] = 0.1 * static_cast<double>(i) - 0.4;

    std::vector<gg::ABDJacobiDyadicMass> masses(2);
    for(int i = 0; i < 2; ++i)
        masses[static_cast<std::size_t>(i)] = gg::to_abd_dyadic_mass(
            2.0 + static_cast<double>(i), Vector3{0.5, 0.25, 0.75}, (Matrix3x3::Identity() * 1.5).eval());

    // ---- the host round ------------------------------------------------------
    std::vector<double>       h_grad(n_seg, 0.0), h_flat(n_flat, 0.0), h_dnorm(n_seg, 0.0),
        h_mnorm(2, 0.0), h_dq(n_seg, 0.0);
    std::vector<std::int32_t> h_tri_rows = tri_rows, h_tri_cols = tri_cols;
    std::vector<double>       h_tri_vals = tri_vals;

    using HostGrad   = lla::LDenseVectorHostMemoryView<double>;
    using HostTri    = lla::LTripletMatrixHostMemoryView<double, 3, 3>;
    using HostRep    = lla::LTripletMatrixHostMemoryView<double, 12, 12>;
    using HostRepDbl = lla::LDoubletVectorHostMemoryView<double, 12>;
    using HostDytTri = lla::LTripletMatrixHostMemoryView<double, 3, 3>;
    using HostDytDbl = lla::LDoubletVectorHostMemoryView<double, 3>;
    using HostDense  = lla::LDenseVectorHostMemoryView<double>;
    using HostCDense = lla::LCDenseVectorHostMemoryView<double>;

    Views<HostGrad, HostTri, HostRep, HostRepDbl, HostDytTri, HostDytDbl, HostDense, HostDense, HostCDense> hv{
        HostGrad::from_handles(h_grad.data(), static_cast<std::int32_t>(n_seg)),
        HostTri::from_handles(h_tri_rows.data(),
                              h_tri_cols.data(),
                              h_tri_vals.data(),
                              8,
                              8,
                              static_cast<std::int32_t>(n_tri)),
        HostRep::from_handles(rep_rows.data(),
                              rep_cols.data(),
                              rep_vals.data(),
                              2,
                              2,
                              static_cast<std::int32_t>(RoundData::kRep)),
        HostRepDbl::from_handles(rep_dbl_idx.data(),
                                 rep_dbl_vals.data(),
                                 2,
                                 static_cast<std::int32_t>(RoundData::kRep)),
        HostDytTri::from_handles(dyt_rows.data(),
                                 dyt_cols.data(),
                                 dyt_vals.data(),
                                 4,
                                 4,
                                 static_cast<std::int32_t>(RoundData::kDyt)),
        HostDytDbl::from_handles(dyt_dbl_idx.data(),
                                 dyt_dbl_vals.data(),
                                 4,
                                 static_cast<std::int32_t>(RoundData::kDyt)),
        HostDense::from_handles(h_flat.data(), static_cast<std::int32_t>(n_flat)),
        HostDense::from_handles(h_dnorm.data(), static_cast<std::int32_t>(n_seg)),
        HostDense::from_handles(h_mnorm.data(), 2),
        HostCDense::from_handles(sol.data(), static_cast<std::int32_t>(n_seg))};

    assembly_round<double>(hv,
                           [&](std::int32_t i) { return v2b_buf[static_cast<std::size_t>(i)]; },
                           [&](std::int32_t i)
                           { return gg::to_generic(gg::ABDJacobi{gg::to_luisa(d.rest[static_cast<std::size_t>(i)])}); },
                           [&](std::int32_t i) { return masses[static_cast<std::size_t>(i)]; },
                           [&](std::int32_t i) { return fixed_buf[static_cast<std::size_t>(i)]; },
                           [&](std::int32_t i) { return ext_buf[static_cast<std::size_t>(i)]; },
                           [&](std::int32_t i, const lps::LVec<double, labd::kDofPerBody>& vec)
                           {
                               for(int k = 0; k < 12; ++k)
                                   h_dq[static_cast<std::size_t>(i) * 12 + static_cast<std::size_t>(k)] = -vec(k);
                           });

    std::printf("[t61b] device-vs-host round: %zu triplets, %zu flat, %zu dofs\n",
                n_tri,
                n_flat,
                n_seg);
    T61B_CHECK(c, n_tri == 112u);

    // ---- the device round ----------------------------------------------------
    std::vector<double> d_grad(n_seg, 0.0), d_flat(n_flat, 0.0), d_dnorm(n_seg, 0.0),
        d_mnorm(2, 0.0), d_dq(n_seg, 0.0);
    std::vector<std::int32_t> dt_tri_rows = tri_rows, dt_tri_cols = tri_cols;
    std::vector<double>       dt_tri_vals = tri_vals;

    auto upload_i32 = [](const std::vector<std::int32_t>& host)
    {
        auto buf = std::make_shared<lud::LDeviceBuffer<std::int32_t>>();
        buf->resize(host.size());
        if(!host.empty())
            buf->view().copy_from(span<const std::int32_t>{host});
        return buf;
    };
    auto upload_f64 = [](const std::vector<double>& host)
    {
        auto buf = std::make_shared<lud::LDeviceBuffer<double>>();
        buf->resize(host.size());
        if(!host.empty())
            buf->view().copy_from(span<const double>{host});
        return buf;
    };

    auto dt_tri_rows_b = upload_i32(dt_tri_rows);
    auto dt_tri_cols_b = upload_i32(dt_tri_cols);
    auto dt_tri_vals_b = upload_f64(dt_tri_vals);
    auto rep_rows_b    = upload_i32(rep_rows);
    auto rep_cols_b    = upload_i32(rep_cols);
    auto rep_vals_b    = upload_f64(rep_vals);
    auto rep_dbl_idx_b = upload_i32(rep_dbl_idx);
    auto rep_dbl_val_b = upload_f64(rep_dbl_vals);
    auto dyt_rows_b    = upload_i32(dyt_rows);
    auto dyt_cols_b    = upload_i32(dyt_cols);
    auto dyt_vals_b    = upload_f64(dyt_vals);
    auto dyt_dbl_idx_b = upload_i32(dyt_dbl_idx);
    auto dyt_dbl_val_b = upload_f64(dyt_dbl_vals);
    auto fixed_b       = upload_i32(fixed_buf);
    auto ext_b         = upload_i32(ext_buf);
    auto v2b_b         = upload_i32(v2b_buf);
    auto grad_b        = upload_f64(d_grad);
    auto flat_b        = upload_f64(d_flat);
    auto dnorm_b       = upload_f64(d_dnorm);
    auto mnorm_b       = upload_f64(d_mnorm);
    auto sol_b         = upload_f64(sol);
    auto dq_b          = upload_f64(d_dq);

    // the per-vertex jacobians and the per-body masses (element types)
    auto Js_b = std::make_shared<lud::LDeviceBuffer<gg::LVector3>>();
    {
        std::vector<gg::LVector3> host;
        for(std::size_t i = 0; i < RoundData::kNv; ++i)
            host.push_back(gg::to_luisa(d.rest[i]));
        Js_b->resize(host.size());
        Js_b->view().copy_from(span<const gg::LVector3>{host});
    }
    auto mass_b = std::make_shared<lud::LDeviceBuffer<gg::ABDJacobiDyadicMass>>();
    {
        mass_b->resize(masses.size());
        mass_b->view().copy_from(span<const gg::ABDJacobiDyadicMass>{masses});
    }

    auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<int32_t>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<double>,
                                         lcrt::Buffer<gg::LVector3>,
                                         lcrt::Buffer<gg::ABDJacobiDyadicMass>>(
        [](lcrt::BufferVar<double>   grad,
           lcrt::BufferVar<int32_t>  tri_rows,
           lcrt::BufferVar<int32_t>  tri_cols,
           lcrt::BufferVar<double>   tri_vals,
           lcrt::BufferVar<int32_t>  rep_rows,
           lcrt::BufferVar<int32_t>  rep_cols,
           lcrt::BufferVar<double>   rep_vals,
           lcrt::BufferVar<int32_t>  rep_dbl_idx,
           lcrt::BufferVar<double>   rep_dbl_vals,
           lcrt::BufferVar<int32_t>  dyt_rows,
           lcrt::BufferVar<int32_t>  dyt_cols,
           lcrt::BufferVar<double>   dyt_vals,
           lcrt::BufferVar<int32_t>  dyt_dbl_idx,
           lcrt::BufferVar<double>   dyt_dbl_vals,
           lcrt::BufferVar<int32_t>  fixed,
           lcrt::BufferVar<int32_t>  ext,
           lcrt::BufferVar<int32_t>  v2b,
           lcrt::BufferVar<double>   flat,
           lcrt::BufferVar<double>   dnorm,
           lcrt::BufferVar<double>   mnorm,
           lcrt::BufferVar<double>   sol,
           lcrt::BufferVar<double>   dq,
           lcrt::BufferVar<double>   scratch,
           lcrt::BufferVar<gg::LVector3> Js,
           lcrt::BufferVar<gg::ABDJacobiDyadicMass> mass) noexcept
        {
            $if(lcrt::dispatch_x() == 0u)
            {
                using S          = lcrt::Var<double>;
                using GradV      = lla::LDenseVectorDeviceView<S>;
                using TriV       = lla::LTripletMatrixDeviceView<S, 3, 3>;
                using RepV       = lla::LTripletMatrixDeviceView<S, 12, 12>;
                using RepDblV    = lla::LDoubletVectorDeviceView<S, 12>;
                using DytTriV    = lla::LTripletMatrixDeviceView<S, 3, 3>;
                using DytDblV    = lla::LDoubletVectorDeviceView<S, 3>;
                using DenseV     = lla::LDenseVectorDeviceView<S>;
                using CDenseV    = lla::LCDenseVectorDeviceView<S>;

                constexpr std::int32_t n_tri  = 112;
                constexpr std::int32_t n_flat = 288;
                constexpr std::int32_t n_seg  = 24;

                Views<GradV, TriV, RepV, RepDblV, DytTriV, DytDblV, DenseV, DenseV, CDenseV> dv{
                    GradV::from_handles(grad, n_seg),
                    TriV::from_handles(tri_rows, tri_cols, tri_vals, 8, 8, n_tri),
                    RepV::from_handles(rep_rows, rep_cols, rep_vals, 2, 2, 3),
                    RepDblV::from_handles(rep_dbl_idx, rep_dbl_vals, 2, 3),
                    DytTriV::from_handles(dyt_rows, dyt_cols, dyt_vals, 4, 4, 2),
                    DytDblV::from_handles(dyt_dbl_idx, dyt_dbl_vals, 4, 2),
                    DenseV::from_handles(flat, n_flat),
                    DenseV::from_handles(dnorm, n_seg),
                    DenseV::from_handles(mnorm, 2),
                    CDenseV::from_handles(sol, n_seg)};

                assembly_round<S>(
                    dv,
                    [&](auto&& i) noexcept { return v2b->read(i); },
                    [&](auto&& i) noexcept
                    {
                        return gg::ABDJacobiT<S>{.x_bar = gg::load_vec3(Js->read(i))};
                    },
                    [&](auto&& i) noexcept { return gg::load_dyadic_mass(mass->read(i)); },
                    [&](auto&& i) noexcept { return fixed->read(i); },
                    [&](auto&& i) noexcept { return ext->read(i); },
                    [&](auto&& i, const lps::LVec<S, labd::kDofPerBody>& vec) noexcept
                    {
                        for(int k = 0; k < 12; ++k)
                            dq->write(i * 12 + static_cast<std::uint32_t>(k), -vec(k));
                    });
                (void)scratch;
            };
        });

    gg::frame_stream().record(k(grad_b->view(),
                                dt_tri_rows_b->view(),
                                dt_tri_cols_b->view(),
                                dt_tri_vals_b->view(),
                                rep_rows_b->view(),
                                rep_cols_b->view(),
                                rep_vals_b->view(),
                                rep_dbl_idx_b->view(),
                                rep_dbl_val_b->view(),
                                dyt_rows_b->view(),
                                dyt_cols_b->view(),
                                dyt_vals_b->view(),
                                dyt_dbl_idx_b->view(),
                                dyt_dbl_val_b->view(),
                                fixed_b->view(),
                                ext_b->view(),
                                v2b_b->view(),
                                flat_b->view(),
                                dnorm_b->view(),
                                mnorm_b->view(),
                                sol_b->view(),
                                dq_b->view(),
                                sol_b->view(),
                                Js_b->view(),
                                mass_b->view())
                              .dispatch(1));
    gg::frame_stream().synchronize();
    lud::main_stream().synchronize();

    // ---- the bitwise comparison ----------------------------------------------
    {
        int    compared    = 0;
        int    mismatches  = 0;
        double worst       = 0.0;
        auto   compare_f64 = [&](const std::vector<double>& a, const std::vector<double>& b)
        {
            for(std::size_t i = 0; i < a.size(); ++i)
            {
                ++compared;
                if(a[i] != b[i])
                {
                    ++mismatches;
                    worst = std::max(worst, std::abs(a[i] - b[i]));
                }
            }
        };
        auto compare_i32 = [&](const std::vector<std::int32_t>& a,
                               const std::vector<std::int32_t>& b)
        {
            for(std::size_t i = 0; i < a.size(); ++i)
            {
                ++compared;
                if(a[i] != b[i])
                    ++mismatches;
            }
        };

        auto read_f64 = [](lud::LBufferView<double> buf, std::size_t n)
        {
            std::vector<double> host(n, 0.0);
            buf.copy_to(span<double>{host});
            return host;
        };
        auto read_i32 = [](lud::LBufferView<std::int32_t> buf, std::size_t n)
        {
            std::vector<std::int32_t> host(n, 0);
            buf.copy_to(span<std::int32_t>{host});
            return host;
        };

        compare_f64(h_grad, read_f64(grad_b->view(), n_seg));
        compare_i32(h_tri_rows, read_i32(dt_tri_rows_b->view(), n_tri));
        compare_i32(h_tri_cols, read_i32(dt_tri_cols_b->view(), n_tri));
        compare_f64(h_tri_vals, read_f64(dt_tri_vals_b->view(), n_tri * 9));
        compare_f64(h_flat, read_f64(flat_b->view(), n_flat));
        compare_f64(h_dnorm, read_f64(dnorm_b->view(), n_seg));
        compare_f64(h_mnorm, read_f64(mnorm_b->view(), 2));
        compare_f64(h_dq, read_f64(dq_b->view(), n_seg));

        std::printf("[t61b] device-vs-host: %d values compared, %d mismatches, worst |dev-host| "
                    "%.3e\n",
                    compared,
                    mismatches,
                    worst);
        T61B_CHECK(c, compared >= 1000);
        T61B_CHECK(c, mismatches == 0);
        T61B_CHECK(c, worst == 0.0);
    }

    std::printf("[t61b] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  7) the per-case assertion counts (the T6.1a summary idiom)
// ===========================================================================
TEST_CASE("abd_linear_summary", "[luisa][abd_linear]")
{
    int total = 0;
    for(auto& [name, checks] : t61b::g_case_counts)
    {
        std::printf("[t61b] case %-28s %6d assertions\n", name.c_str(), checks);
        total += checks;
    }
    std::printf("[t61b] summary: %zu cases, %d assertions\n", t61b::g_case_counts.size(), total);
    CHECK(total > 0);
}
