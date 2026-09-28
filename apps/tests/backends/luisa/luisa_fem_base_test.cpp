/********************************************************************
 * @file luisa_fem_base_test.cpp
 * @brief T5.1 — FEM base, utilities, state accessors and reporter bases
 *        (LuisaCompute backend).
 *
 * Every oracle in this file is derived independently *here*: the reference
 * backend is CUDA-only and cannot be host-included, so nothing is copied from it.
 * The oracles are
 *   (a) closed-form values for a single tetrahedron (unit tet and a
 *       scaled/rotated/stretched one) and a single (rotated) shell triangle,
 *       derived by hand in the comments below,
 *   (b) central finite differences (h = 1e-6) on a random admissible state,
 *   (c) structural invariants (partition of unity, rigid-translation nullspace,
 *       symmetry, positive semidefiniteness, determinant/cofactor identity),
 *   (d) the device-vs-host equivalence of the ported numeric templates
 *       (S = double vs S = lcrt::Var<double> in a Kernel1D), and
 *   (e) a real scene driven through the public Engine/World API, where the
 *       method's device buffers are compared against the same hand-computed
 *       values.
 *
 * Coverage map:
 *  1) tet closed forms + the utility dump .......... fem_base_utils_host_oracle
 *  2) affine-map recovery (rot/stretch/reflect) ..... fem_base_utils_affine_map
 *  3) finite differences + symmetry + PSD .......... fem_base_utils_finite_difference
 *  4) degenerates pinned from the reference ......... fem_base_utils_degenerate_inputs
 *  5) device == host for the utility templates ...... fem_base_utils_device_vs_host
 *  6) real scene: reporters + method buffers ........ fem_base_scene_reporters
 *  7) state accessor (copy_from/copy_to/exports) .... fem_base_state_accessor
 *  8) FEM dump/recover round trip ................... fem_base_dump_recover
 *
 * Shared-binary rule (cookbook §7.14 rule 6): every system registered by every
 * test file is built for every scene, so this test asserts only on *its own*
 * slices (via VertexReporter::vertex_offset()/BodyReporter::body_offset() and the
 * reporter uids) and marks its own geometries.
 *********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/geometry/utils/compute_vertex_volume.h>
#include <uipc/constitution/arap.h>
#include <uipc/constitution/discrete_shell_bending.h>
#include <uipc/constitution/hookean_spring.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <uipc/constitution/particle.h>
#include <uipc/core/finite_element_state_accessor_feature.h>

#include <type_define.h>
#include <frame_stream.h>
#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/finite_element_elastics.h>
#include <finite_element/fem_3d_constitution.h>
#include <finite_element/codim_2d_constitution.h>
#include <finite_element/codim_1d_constitution.h>
#include <finite_element/codim_0d_constitution.h>
#include <finite_element/finite_element_body_reporter.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <finite_element/finite_element_surface_reporter.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <utils/offset_count_collection.h>
#include <utils/report_extent_check.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;

namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace fem  = uipc::backend::luisa::fem;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;

// The ported utility surface lives in uipc::backend::luisa (like the reference's
// matrix_utils), so a test body sees the same unqualified spellings a ported
// constitution does (the reference's constitutions call `svd`, `flatten`, ...).
using gg::clamp_to_spd;
using gg::ddot;
using gg::evd;
using gg::flatten;
using gg::polar_decomposition;
using gg::svd;
using gg::unflatten;

namespace t51fem
{
// ---------------------------------------------------------------------------
//  Eigen-facing shims.
//
//  The *hand-derived* oracles in this file are written in Eigen notation (the
//  same notation the derivation comments use), while the ported FEM utilities
//  work on the numerics layer's types -- exactly like the reference's
//  `UIPC_GENERIC` routines work on Eigen's. These shims convert at the boundary
//  (`LVector3`/`LMatrix3x3` are byte-identical to `Vector3`/`Matrix3x3`, see
//  type_define.h) so that both sides read the same way.
// ---------------------------------------------------------------------------
[[nodiscard]] inline lps::LVec<double, 3> lv(const Vector3& v) noexcept
{
    return lps::LVec<double, 3>{v(0), v(1), v(2)};
}

[[nodiscard]] inline lps::LMat<double, 3, 3> Ds(const Vector3& a,
                                                const Vector3& b,
                                                const Vector3& c,
                                                const Vector3& d) noexcept
{
    return fem::Ds(lv(a), lv(b), lv(c), lv(d));
}

[[nodiscard]] inline lps::LMat<double, 3, 3> Dm_inv(const Vector3& a,
                                                    const Vector3& b,
                                                    const Vector3& c,
                                                    const Vector3& d) noexcept
{
    return fem::Dm_inv(lv(a), lv(b), lv(c), lv(d));
}

[[nodiscard]] inline lps::LMat<double, 3, 3> F(const Vector3&              a,
                                               const Vector3&              b,
                                               const Vector3&              c,
                                               const Vector3&              d,
                                               const lps::LMat<double, 3, 3>& DmInv) noexcept
{
    return fem::F(lv(a), lv(b), lv(c), lv(d), DmInv);
}

[[nodiscard]] inline lps::LMat<double, 9, 12> dFdx(
    const lps::LMat<double, 3, 3>& m) noexcept
{
    return fem::dFdx(m);
}

[[nodiscard]] inline lps::LMat<double, 3, 4> tetrahedron_shape_gradients(
    const lps::LMat<double, 3, 3>& m) noexcept
{
    return fem::tetrahedron_shape_gradients(m);
}
}  // namespace t51fem

namespace
{
// ---------------------------------------------------------------------------
//  small helpers
// ---------------------------------------------------------------------------
constexpr double kEps = 1e-12;

double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

std::string workspace_path(std::string_view name)
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

std::vector<Vector3> read_vec3(const lud::LCBufferView<gg::LVector3>& view)
{
    auto                  raw = read_device(view);
    std::vector<Vector3> host;
    host.reserve(raw.size());
    for(auto& v : raw)
    {
        host.push_back(gg::to_eigen(v));
    }
    return host;
}

std::vector<Matrix3x3> read_mat3(const lud::LCBufferView<gg::LMatrix3x3>& view)
{
    auto                    raw = read_device(view);
    std::vector<Matrix3x3>  host;
    host.reserve(raw.size());
    for(auto& m : raw)
    {
        host.push_back(gg::to_eigen(m));
    }
    return host;
}

std::vector<Vector2i> read_vec2i(const lud::LCBufferView<gg::LVector2i>& view)
{
    auto                  raw = read_device(view);
    std::vector<Vector2i> host;
    host.reserve(raw.size());
    for(auto& v : raw)
    {
        host.push_back(gg::to_eigen(v));
    }
    return host;
}

template <typename T>
bool arrays_equal(const std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
    {
        return false;
    }
    for(size_t i = 0; i < a.size(); ++i)
    {
        if constexpr(requires { (a[i] - b[i]).norm(); })
        {
            if(!((a[i] - b[i]).norm() <= 1e-12))
            {
                return false;
            }
        }
        else
        {
            if(!(std::abs(a[i] - b[i]) <= 1e-12))
            {
                return false;
            }
        }
    }
    return true;
}

std::vector<Vector4i> read_vec4i(const lud::LCBufferView<gg::LVector4i>& view)
{
    auto                  raw = read_device(view);
    std::vector<Vector4i> host;
    host.reserve(raw.size());
    for(auto& v : raw)
    {
        host.push_back(gg::to_eigen(v));
    }
    return host;
}

std::vector<Vector3i> read_vec3i(const lud::LCBufferView<gg::LVector3i>& view)
{
    auto                  raw = read_device(view);
    std::vector<Vector3i> host;
    host.reserve(raw.size());
    for(auto& v : raw)
    {
        host.push_back(gg::to_eigen(v));
    }
    return host;
}

// ---------------------------------------------------------------------------
//  hand-derived tetrahedron oracle
//
//  Reference tet ("unit tet"): X0 = (0,0,0), X1 = (1,0,0), X2 = (0,1,0),
//  X3 = (0,0,1).  Then
//      Dm  = Ds = [X1-X0 | X2-X0 | X3-X0] = I
//      Dm^-1 = I
//      shape gradients: grad_i = column i of Dm^-1 for i=1..3 and
//                       grad_0 = -(grad_1+grad_2+grad_3)
//                     => grads = [-1 -1 -1; 1 0 0; 0 1 0; 0 0 1] (rows = vertex)
//      V = det(Ds)/6 = 1/6
//  The "B matrix" of the FEM method is `dFdx` (9x12): dF/dx_l is column l of
//  dFdx, and with Dm^-1 = I its non-zero entries are the shape gradients (see
//  the assertions below).
// ---------------------------------------------------------------------------
struct TetDesc
{
    // material (rest) configuration
    Vector3 X0, X1, X2, X3;
    // deformed configuration
    Vector3 x0, x1, x2, x3;
};

TetDesc unit_tet()
{
    TetDesc t;
    t.X0 = Vector3{0.0, 0.0, 0.0};
    t.X1 = Vector3{1.0, 0.0, 0.0};
    t.X2 = Vector3{0.0, 1.0, 0.0};
    t.X3 = Vector3{0.0, 0.0, 1.0};
    t.x0 = t.X0;
    t.x1 = t.X1;
    t.x2 = t.X2;
    t.x3 = t.X3;
    return t;
}

/// `x -> F x + t` applied to a tet (the affine-map oracle of brief §4.3).
TetDesc affine_tet(const TetDesc& rest, const Matrix3x3& F, const Vector3& t)
{
    auto map = [&](const Vector3& X) -> Vector3 { return F * X + t; };
    TetDesc d;
    d.X0 = rest.X0;
    d.X1 = rest.X1;
    d.X2 = rest.X2;
    d.X3 = rest.X3;
    d.x0 = map(d.X0);
    d.x1 = map(d.X1);
    d.x2 = map(d.X2);
    d.x3 = map(d.X3);
    return d;
}

Matrix3x3 rotation_matrix(const Vector3& axis, double angle)
{
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

// ---------------------------------------------------------------------------
//  hand-derived shell (codim-2d) oracle
//
//  A shell triangle (p0, p1, p2) has
//      rest length of edge (i, j) = |p_i - p_j|
//      rest area = 0.5 * |(p1 - p0) x (p2 - p0)|
//  both of which are rigid-motion invariant, so the *frame* of the triangle does
//  not change them; the frame-dependent rest basis (IB) is T5.2's material. What
//  the checks below buy is: (i) the axis-aligned triangle's area/lengths are the
//  hand-computed ones, (ii) a rotated/reflected instance gives exactly the same
//  values (so a bug that reads the wrong frame or an axis-aligned shortcut shows
//  up), (iii) scaling by s scales lengths by s and areas by s^2.
// ---------------------------------------------------------------------------
struct TriDesc
{
    Vector3 p0, p1, p2;
};

TriDesc flat_triangle()
{
    // legs 3 and 2, right angle at p0:  area = 3,  edges = 3, sqrt(5), 2*sqrt(2)
    return TriDesc{Vector3{0.0, 0.0, 0.0}, Vector3{3.0, 0.0, 0.0}, Vector3{1.0, 2.0, 0.0}};
}

TriDesc transform_tri(const TriDesc& t, const Matrix3x3& F, const Vector3& translation)
{
    return TriDesc{F * t.p0 + translation, F * t.p1 + translation, F * t.p2 + translation};
}

double tri_area(const TriDesc& t)
{
    return 0.5 * ((t.p1 - t.p0).cross(t.p2 - t.p0)).norm();
}

// ---------------------------------------------------------------------------
//  The ported numeric templates, run as one generic routine (the T2.x pattern):
//  S = double (host) and S = lcrt::Var<double> (device, inside a Kernel1D).
//  `put(host_index, value)` is the only side effect, so the two runs walk an
//  identical fixed dump order and can be compared element by element.
// ---------------------------------------------------------------------------
constexpr int kDumpCount = 187;

template <typename S>
lps::LMat<S, 3, 3> fixed_F()
{
    // a well-conditioned, non-symmetric matrix
    lps::LMat<S, 3, 3> F;
    const double       v[9] = {1.2, 0.3, -0.2, 0.1, 0.9, 0.4, -0.3, 0.2, 1.4};
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            F(r, c) = lps::make_scalar<S>(v[c * 3 + r]);
    return F;
}

template <typename S>
lps::LMat<S, 9, 9> fixed_ddEddF()
{
    lps::LMat<S, 9, 9> H;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
        {
            const double value =
                (r == c) ? (1.0 + 0.25 * r) : (0.1 * ((r + c) % 5));
            H(r, c) = lps::make_scalar<S>(value);
        }
    return H;
}

template <typename S, typename Put>
void fem_util_round(Put&& put)
{
    int k = 0;

    // material space of the *stretched* reference tet:
    //   Dm = diag(2, 3, 4)  ->  Dm^-1 = diag(0.5, 1/3, 0.25)
    lps::LMat<S, 3, 3> Dm;
    Dm(0, 0) = lps::make_scalar<S>(2.0);
    Dm(1, 1) = lps::make_scalar<S>(3.0);
    Dm(2, 2) = lps::make_scalar<S>(4.0);
    auto DmInv = lps::inverse(Dm);

    auto F = fixed_F<S>();
    lps::LVec<S, 3> a;
    a(0) = lps::make_scalar<S>(0.5);
    a(1) = lps::make_scalar<S>(-1.5);
    a(2) = lps::make_scalar<S>(2.0);

    // deformed tet: x = F * (X - X0) + X0 * 0.25 with X_i from Ds
    lps::LVec<S, 3> x0;
    x0(0) = lps::make_scalar<S>(0.1);
    x0(1) = lps::make_scalar<S>(-0.2);
    x0(2) = lps::make_scalar<S>(0.3);
    lps::LVec<S, 3> x1, x2, x3;
    {
        auto ds0        = lps::LVec<S, 3>{};
        ds0(0)          = lps::make_scalar<S>(2.0);
        auto ds1        = lps::LVec<S, 3>{};
        ds1(1)          = lps::make_scalar<S>(3.0);
        auto ds2        = lps::LVec<S, 3>{};
        ds2(2)          = lps::make_scalar<S>(4.0);
        x1              = x0 + F * ds0;
        x2              = x0 + F * ds1;
        x3              = x0 + F * ds2;
    }

    auto v9 = flatten(F);
    for(int i = 0; i < 9; ++i)
        put(k++, v9(i));

    put(k++, ddot(F, F));
    put(k++, fem::invariant2(F));
    put(k++, fem::invariant3(F));

    lps::LMat<S, 3, 3> U, V;
    lps::LVec<S, 3>    Sigma;
    gg::svd(F, U, Sigma, V);
    for(int i = 0; i < 3; ++i)
        put(k++, Sigma(i));

    put(k++, fem::invariant2(Sigma));
    put(k++, fem::invariant3(Sigma));
    put(k++, fem::invariant4(F, a));
    put(k++, fem::invariant5(F, a));

    auto dFdX = fem::dFdx(DmInv);
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 9; ++r)
            put(k++, dFdX(r, c));

    auto dJ = fem::dJdF(F);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            put(k++, dJ(r, c));

    auto sg = fem::tetrahedron_shape_gradients(DmInv);
    for(int c = 0; c < 4; ++c)
        for(int r = 0; r < 3; ++r)
            put(k++, sg(r, c));

    auto grad = fem::project_F_gradient(sg.col(0), v9);
    for(int i = 0; i < 3; ++i)
        put(k++, grad(i));

    auto ddEddF = fixed_ddEddF<S>();
    auto H      = fem::project_F_hessian_block(sg.col(0), ddEddF, sg.col(1));
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            put(k++, H(r, c));

    auto DmInvT = fem::Dm_inv(x0, x1, x2, x3);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            put(k++, DmInvT(r, c));

    auto ds = fem::Ds(x0, x1, x2, x3);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            put(k++, ds(r, c));

    auto Fm   = fem::F(x0, x1, x2, x3, DmInvT);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            put(k++, Fm(r, c));
}

/// The same routine on the device; the single thread writes the whole dump.
auto& util_dump_kernel()
{
    static auto& k = lud::linalg::detail::cached_kernel<lcrt::Buffer<double>>(
        [](lcrt::BufferVar<double> out) noexcept
        {
            fem_util_round<lcrt::Var<double>>(
                [&](int i, auto&& value) noexcept { out->write(i, value); });
        });
    return k;
}
}  // namespace
namespace t51
{
// ---------------------------------------------------------------------------
//  The test constitutions' uids are the *frontend* constitutions' uids: the
//  scene's ConstitutionTabular is built from the geometries' `constitution_uid`
//  meta attribute and resolved through builtin::ConstitutionUIDCollection, so a
//  backend constitution must report exactly the uid the frontend wrote (proving
//  the reference's uid/dim contract).
// ---------------------------------------------------------------------------
enum class Dim
{
    d0 = 0,
    d1 = 1,
    d2 = 2,
    d3 = 3,
    extra = 4,
};

struct Uids
{
    U64 value[5] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
};

Uids& uids()
{
    static Uids u;
    return u;
}

struct Record
{
    int   build_count         = 0;
    int   init_count          = 0;
    int   report_extent_count = 0;
    SizeT vertex_count        = 0;
    SizeT primitive_count     = 0;
    SizeT geo_info_count      = 0;
    SizeT energy_count        = 0;
    SizeT gradient_count      = 0;
    SizeT hessian_count       = 0;
    bool  gradient_only_queried = false;
    // values gathered inside do_init through the FilteredInfo / BaseInfo surface
    std::vector<IndexT> for_each_is_fixed;
    SizeT               base_indices_size = 0;
    SizeT               base_dm_size      = 0;
    SizeT               base_volume_size  = 0;
    SizeT               xs_size           = 0;
    SizeT               xbar_size         = 0;
};

Record& rec(Dim d)
{
    static Record r[5];
    return r[static_cast<int>(d)];
}

int idx(Dim d)
{
    return static_cast<int>(d);
}

void reset_records()
{
    for(int i = 0; i < 5; ++i)
    {
        rec(static_cast<Dim>(i)) = Record{};
    }
}
}  // namespace t51

// ---------------------------------------------------------------------------
//  Test constitutions: one per dimension bucket + one extra constitution.
//  They are the T4.1 "test reporter" idiom applied to the constitution surface:
//  they declare the uid/dim the *scene* needs and record what the FEM method
//  reported to them, so the test can assert the whole classification plumbing
//  (do_build -> add_constitution -> classify by (dim, uid) -> init ->
//  report_extent) without porting any real constitution (that is T5.2).
// ---------------------------------------------------------------------------
class T51FEM3DConstitution final : public gg::FEM3DConstitution
{
  public:
    using FEM3DConstitution::FEM3DConstitution;

  protected:
    U64 get_uid() const noexcept override { return t51::uids().value[t51::idx(t51::Dim::d3)]; }

    void do_build(BuildInfo& info) override
    {
        (void)info;
        t51::rec(t51::Dim::d3).build_count++;
    }

    void do_init(gg::FiniteElementMethod::FilteredInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::d3);
        r.init_count++;
        r.vertex_count    = info.vertex_count();
        r.primitive_count = info.primitive_count();
        r.geo_info_count  = info.geo_infos().size();

        // FilteredInfo::for_each (the reference's per-primitive walk over the
        // constitution's own geometries)
        r.for_each_is_fixed.assign(r.vertex_count, -1);
        info.for_each(  //
            world().scene().geometries(),
            [](SimplicialComplex& sc)
            { return sc.vertices().find<IndexT>(builtin::is_fixed)->view(); },
            [&](const gg::FiniteElementMethod::ForEachInfo& I, IndexT v)
            { r.for_each_is_fixed[I.global_index()] = v; });

        // The dimension base's BaseInfo surface (T5.2 builds on it): a single
        // dim-3 constitution with one tet geometry has index_in_dim == 0.
        gg::FEM3DConstitution::ComputeGradientHessianInfo gh{
            this,
            0,
            false,
            0.0,
            lla::LDoubletVectorView<Float, 3>{},
            lla::LTripletMatrixView<Float, 3, 3>{}};
        r.base_indices_size = gh.indices().size();
        r.base_dm_size      = gh.Dm_invs().size();
        r.base_volume_size  = gh.rest_volumes().size();
        r.xs_size           = gh.xs().size();
        r.xbar_size         = gh.x_bars().size();
    }

    void do_report_extent(ReportExtentInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::d3);
        r.report_extent_count++;
        // gradient_only() must be queried (the reference's convention, checked by
        // FiniteElementElastics::ReportExtentInfo::check)
        r.gradient_only_queried = true;
        bool gradient_only      = info.gradient_only();
        info.energy_count(1);
        info.gradient_count(1);
        info.hessian_count(gradient_only ? 0 : 1);
        r.energy_count   = 1;
        r.gradient_count = 1;
        r.hessian_count  = gradient_only ? 0 : 1;
    }

    void do_compute_energy(ComputeEnergyInfo& info) override { (void)info; }
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override { (void)info; }
};
REGISTER_SIM_SYSTEM(T51FEM3DConstitution);

class T51Codim2DConstitution final : public gg::Codim2DConstitution
{
  public:
    using Codim2DConstitution::Codim2DConstitution;

  protected:
    U64 get_uid() const noexcept override { return t51::uids().value[t51::idx(t51::Dim::d2)]; }
    void do_build(BuildInfo& info) override
    {
        (void)info;
        t51::rec(t51::Dim::d2).build_count++;
    }
    void do_init(gg::FiniteElementMethod::FilteredInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::d2);
        r.init_count++;
        r.vertex_count    = info.vertex_count();
        r.primitive_count = info.primitive_count();
        r.geo_info_count  = info.geo_infos().size();
    }
    void do_report_extent(ReportExtentInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::d2);
        r.report_extent_count++;
        r.gradient_only_queried = true;
        bool gradient_only      = info.gradient_only();
        info.energy_count(1);
        info.gradient_count(1);
        info.hessian_count(gradient_only ? 0 : 1);
    }
    void do_compute_energy(ComputeEnergyInfo& info) override { (void)info; }
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override { (void)info; }
};
REGISTER_SIM_SYSTEM(T51Codim2DConstitution);

class T51Codim1DConstitution final : public gg::Codim1DConstitution
{
  public:
    using Codim1DConstitution::Codim1DConstitution;

  protected:
    U64 get_uid() const noexcept override { return t51::uids().value[t51::idx(t51::Dim::d1)]; }
    void do_build(BuildInfo& info) override
    {
        (void)info;
        t51::rec(t51::Dim::d1).build_count++;
    }
    void do_init(gg::FiniteElementMethod::FilteredInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::d1);
        r.init_count++;
        r.vertex_count    = info.vertex_count();
        r.primitive_count = info.primitive_count();
        r.geo_info_count  = info.geo_infos().size();
    }
    void do_report_extent(ReportExtentInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::d1);
        r.report_extent_count++;
        r.gradient_only_queried = true;
        bool gradient_only      = info.gradient_only();
        info.energy_count(2);
        info.gradient_count(2);
        info.hessian_count(gradient_only ? 0 : 2);
    }
    void do_compute_energy(ComputeEnergyInfo& info) override { (void)info; }
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override { (void)info; }
};
REGISTER_SIM_SYSTEM(T51Codim1DConstitution);

class T51Codim0DConstitution final : public gg::Codim0DConstitution
{
  public:
    using Codim0DConstitution::Codim0DConstitution;

  protected:
    U64 get_uid() const noexcept override { return t51::uids().value[t51::idx(t51::Dim::d0)]; }
    void do_build(BuildInfo& info) override
    {
        (void)info;
        t51::rec(t51::Dim::d0).build_count++;
    }
    void do_init(gg::FiniteElementMethod::FilteredInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::d0);
        r.init_count++;
        r.vertex_count    = info.vertex_count();
        r.primitive_count = info.primitive_count();
        r.geo_info_count  = info.geo_infos().size();
    }
    // do_report_extent is `final` in Codim0DConstitution (it reports all zeros),
    // exactly like the reference.
};
REGISTER_SIM_SYSTEM(T51Codim0DConstitution);

class T51ExtraConstitution final : public gg::FiniteElementExtraConstitution
{
  public:
    using FiniteElementExtraConstitution::FiniteElementExtraConstitution;

  protected:
    U64 get_uid() const noexcept override { return t51::uids().value[t51::idx(t51::Dim::extra)]; }
    void do_build(BuildInfo& info) override
    {
        (void)info;
        t51::rec(t51::Dim::extra).build_count++;
    }
    void do_init(FilteredInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::extra);
        r.init_count++;
        r.geo_info_count = info.geo_infos().size();
        // the extra constitution resolves its geometries through the *geometry
        // meta* `extra_constitution_uids` list
        SizeT vertex_count = 0;
        for(auto& geo_info : info.geo_infos())
        {
            vertex_count += geo_info.vertex_count;
        }
        r.vertex_count = vertex_count;
    }
    void do_report_extent(ReportExtentInfo& info) override
    {
        auto& r = t51::rec(t51::Dim::extra);
        r.report_extent_count++;
        r.gradient_only_queried = true;
        bool gradient_only      = info.gradient_only();
        info.energy_count(1);
        info.gradient_count(1);
        info.hessian_count(gradient_only ? 0 : 1);
    }
    void do_compute_energy(ComputeEnergyInfo& info) override { (void)info; }
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override { (void)info; }
};
REGISTER_SIM_SYSTEM(T51ExtraConstitution);

/// Grabs the FEM systems' public interfaces once they are built, exactly like
/// T4.1's probe. A scene without a FiniteElement constitution invalidates it
/// (require<> throws SimSystemException), so other test files are unaffected.
class T51FEMProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::FiniteElementMethod*         fem = nullptr;
    inline static gg::FiniteElementVertexReporter* vertex_reporter = nullptr;
    inline static gg::FiniteElementBodyReporter*   body_reporter = nullptr;
    inline static gg::FiniteElementSurfaceReporter* surface_reporter = nullptr;
    inline static gg::FiniteElementElastics*       elastics = nullptr;
    // the managers the reporters feed, for the slice assertions
    inline static gg::GlobalVertexManager*             vertex_manager = nullptr;
    inline static gg::GlobalBodyManager*               body_manager = nullptr;
    inline static gg::GlobalSimplicialSurfaceManager*  surface_manager = nullptr;

  private:
    void do_build() override
    {
        fem              = &require<gg::FiniteElementMethod>();
        vertex_reporter  = &require<gg::FiniteElementVertexReporter>();
        body_reporter    = &require<gg::FiniteElementBodyReporter>();
        surface_reporter = &require<gg::FiniteElementSurfaceReporter>();
        elastics         = &require<gg::FiniteElementElastics>();

        vertex_manager = &require<gg::GlobalVertexManager>();
        body_manager   = &require<gg::GlobalBodyManager>();
        surface_manager = find<gg::GlobalSimplicialSurfaceManager>();
    }
};
REGISTER_SIM_SYSTEM(T51FEMProbe);

namespace
{
// ---------------------------------------------------------------------------
//  Scene description (hand-written, no assets)
// ---------------------------------------------------------------------------
struct SceneDesc
{
    std::vector<Vector3>  tet_positions;
    std::vector<Vector4i> tets;
    Float                 tet_density = 1000.0;

    std::vector<Vector3>  shell_positions;
    std::vector<Vector3i> shell_triangles;
    Float                 shell_density = 500.0;
    Float                 shell_thickness = 0.01;
    Matrix3x3             shell_transform = Matrix3x3::Identity();

    std::vector<Vector3>  rod_positions;
    std::vector<Vector2i> rod_edges;
    Float                 rod_density     = 250.0;
    Float                 rod_thickness   = 0.02;

    std::vector<Vector3> point_positions;
    Float                point_density = 100.0;
};

/// unit tet (volume 1/6) + a non-axis-aligned (rotated + shifted) shell triangle
/// + a 2-edge rod with irrational edge lengths + a 2-point cloud.
SceneDesc t51_scene_desc()
{
    SceneDesc d;

    d.tet_positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    d.tets          = {{0, 1, 2, 3}};
    d.tet_density   = 1000.0;

    // the hand-derived triangle (area 3, edges 3, sqrt(5), 2*sqrt(2)), rotated by
    // 30 deg about x and 45 deg about z and shifted
    auto tri       = flat_triangle();
    auto rot       = rotation_matrix(Vector3{0.0, 0.0, 1.0}, 0.7853981633974483)
               * rotation_matrix(Vector3{1.0, 0.0, 0.0}, 0.5235987755982988);
    auto shifted   = transform_tri(tri, rot, Vector3{0.3, -0.2, 0.5});
    d.shell_positions = {shifted.p0, shifted.p1, shifted.p2};
    d.shell_triangles = {{0, 1, 2}};
    d.shell_density   = 500.0;
    d.shell_thickness = 0.01;
    // the *mesh transform* path: the method applies it to the positions and resets
    // it to identity (with a warning) at init
    d.shell_transform = rotation_matrix(Vector3{1.0, 1.0, 0.0}, 0.4);

    d.rod_positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
    d.rod_edges     = {{0, 1}, {1, 2}};
    d.rod_density   = 250.0;
    d.rod_thickness = 0.02;

    d.point_positions = {{5.0, 0.0, 0.0}, {5.5, 0.0, 0.0}};
    d.point_density   = 100.0;

    return d;
}

/// The four constitutive objects of the scene, with the frontend constitutions
/// applied in the reference's order (membrane before extra bending).
Scene build_scene(const SceneDesc& desc)
{
    auto config = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    // contact is not ported (T8.x) and the dytopo effect manager is only created
    // when contact is enabled; the FEM method does not need it.
    config["contact"]["enable"] = false;

    Scene scene{config};
    auto  object = scene.objects().create("t51");

    ARAP arap;
    t51::uids().value[t51::idx(t51::Dim::d3)] = arap.uid();

    auto tet_mesh = tetmesh(span<const Vector3>{desc.tet_positions},
                            span<const Vector4i>{desc.tets});
    label_surface(tet_mesh);
    arap.apply_to(tet_mesh, 1.0e6, desc.tet_density);
    object->geometries().create(tet_mesh);

    NeoHookeanShell nhs;
    t51::uids().value[t51::idx(t51::Dim::d2)] = nhs.uid();

    auto shell_mesh = trimesh(span<const Vector3>{desc.shell_positions},
                              span<const Vector3i>{desc.shell_triangles});
    label_surface(shell_mesh);
    nhs.apply_to(shell_mesh,
                 ElasticModuli2D::youngs_poisson(1.0e6, 0.49),
                 desc.shell_density,
                 desc.shell_thickness);
    {
        // the frontend stores affine transforms as 4x4 matrices
        Matrix4x4 T       = Matrix4x4::Identity();
        T.block<3, 3>(0, 0) = desc.shell_transform;
        view(shell_mesh.transforms())[0] = T;
    }
    // an *extra* constitution on the same geometry (T5.1 provides its base/plumbing)
    DiscreteShellBending dsb;
    t51::uids().value[t51::idx(t51::Dim::extra)] = dsb.uid();
    dsb.apply_to(shell_mesh, 100.0);
    object->geometries().create(shell_mesh);

    HookeanSpring spring;
    t51::uids().value[t51::idx(t51::Dim::d1)] = spring.uid();

    auto rod_mesh = linemesh(span<const Vector3>{desc.rod_positions},
                             span<const Vector2i>{desc.rod_edges});
    label_surface(rod_mesh);
    spring.apply_to(rod_mesh, 1.0e6, desc.rod_density, desc.rod_thickness);
    // a per-vertex gravity override (the method copies it over the config default)
    {
        auto gravity  = rod_mesh.vertices().create<Vector3>(builtin::gravity, Vector3::Zero());
        auto gravity_view = view(*gravity);
        for(size_t i = 0; i < gravity_view.size(); ++i)
        {
            gravity_view[i] = Vector3{0.0, -1.0 * (double)i, 0.5};
        }
    }
    object->geometries().create(rod_mesh);

    Particle particle;
    t51::uids().value[t51::idx(t51::Dim::d0)] = particle.uid();

    auto point_mesh = pointcloud(span<const Vector3>{desc.point_positions});
    label_surface(point_mesh);
    particle.apply_to(point_mesh, desc.point_density, 0.0);
    object->geometries().create(point_mesh);

    return scene;
}

// ---------------------------------------------------------------------------
//  Host model of the FEM method's buffers: the (dim, uid)-sorted geometry order
//  the method builds internally, and the per-vertex/per-element values derived
//  from the same closed forms used by the SDK's frontend volume pass.
// ---------------------------------------------------------------------------
struct FEMHostModel
{
    // the method's geometry order: sorted by (dim, uid) => d0, d1, d2, d3
    std::vector<Vector3>  positions;       // the method's xs/x_bars order
    std::vector<IndexT>   body_ids;        // h_vertex_body_id (geo index in that order)
    std::vector<IndexT>   dimensions;
    std::vector<Float>    thicknesses;
    std::vector<Float>    masses;
    std::vector<Vector3>  gravities;

    std::vector<Vector4i> tet_indices;     // already offset
    std::vector<Float>    tet_volumes;     // det(Ds)/6
    std::vector<Matrix3x3> tet_Dm_inv;
    std::vector<Vector3i> tri_indices;
    std::vector<Float>    tri_areas;
    std::vector<Vector2i> edge_indices;
    std::vector<Float>    edge_lengths;
    std::vector<IndexT>   point_indices;

    /// (scene slot index, method-local vertex_offset), in the method's geometry
    /// order (dim, uid) -- what the vertex reporter writes into each geometry's
    /// `global_vertex_offset` meta attribute at frame 0.
    std::vector<std::pair<IndexT, SizeT>> geo_offsets;
};

FEMHostModel compute_fem_host_model(const SceneDesc& desc)
{
    FEMHostModel m;

    auto push_vertices = [&](const std::vector<Vector3>& pos,
                             Float                       density,
                             IndexT                      body_id,
                             const std::vector<Float>&   volume) {
        for(size_t i = 0; i < pos.size(); ++i)
        {
            m.positions.push_back(pos[i]);
            m.body_ids.push_back(body_id);
            m.masses.push_back(density * volume[i]);
            m.gravities.push_back(Vector3{0.0, -9.8, 0.0});
        }
    };

    // ---- dim 0 (Particle: uid 13) -----------------------------------------
    {
        const IndexT body_id = 0;
        m.geo_offsets.push_back({3, 0});  // the point object is the 4th created
        std::vector<Float> volume(desc.point_positions.size(), 1.0);  // thickness 0 -> 1
        push_vertices(desc.point_positions, desc.point_density, body_id, volume);
        for(size_t i = 0; i < desc.point_positions.size(); ++i)
        {
            m.dimensions.push_back(0);
            m.thicknesses.push_back(0.0);
            m.point_indices.push_back(static_cast<IndexT>(i));
        }
    }

    // ---- dim 1 (HookeanSpring: uid 12) ------------------------------------
    {
        const IndexT body_id     = 1;
        const SizeT  base_vertex = m.positions.size();
        m.geo_offsets.push_back({2, base_vertex});
        // edge_volume = l * pi * r^2, vertex volume = sum(edge_volume / 2)
        std::vector<Float> volume(desc.rod_positions.size(), 0.0);
        for(auto& e : desc.rod_edges)
        {
            const auto l = (desc.rod_positions[e[1]] - desc.rod_positions[e[0]]).norm();
            const auto ev = l * 3.14159265358979323846 * desc.rod_thickness * desc.rod_thickness;
            volume[e[0]] += ev / 2.0;
            volume[e[1]] += ev / 2.0;
        }
        push_vertices(desc.rod_positions, desc.rod_density, body_id, volume);
        for(size_t i = 0; i < desc.rod_positions.size(); ++i)
        {
            m.dimensions.push_back(1);
            m.thicknesses.push_back(desc.rod_thickness);
        }
        for(auto& e : desc.rod_edges)
        {
            m.edge_indices.push_back(Vector2i{e[0] + static_cast<IndexT>(base_vertex),
                                              e[1] + static_cast<IndexT>(base_vertex)});
            m.edge_lengths.push_back(
                (desc.rod_positions[e[1]] - desc.rod_positions[e[0]]).norm());
        }
        // the per-vertex gravity override
        for(size_t i = 0; i < desc.rod_positions.size(); ++i)
        {
            m.gravities[base_vertex + i] = Vector3{0.0, -1.0 * static_cast<double>(i), 0.5};
        }
    }

    // ---- dim 2 (NeoHookeanShell: uid 11) ----------------------------------
    {
        const IndexT body_id     = 2;
        const SizeT  base_vertex = m.positions.size();
        m.geo_offsets.push_back({1, base_vertex});
        // the mesh transform is applied to the positions at init (and reset)
        std::vector<Vector3> pos;
        for(auto& p : desc.shell_positions)
        {
            pos.push_back(desc.shell_transform * p);
        }
        const Float r = desc.shell_thickness;
        const auto  area =
            0.5 * ((pos[1] - pos[0]).cross(pos[2] - pos[0])).norm();
        const Float tri_volume = (r == 0.0) ? area : area * (2.0 * r);
        std::vector<Float> volume(pos.size(), tri_volume / 3.0);
        push_vertices(pos, desc.shell_density, body_id, volume);
        for(size_t i = 0; i < pos.size(); ++i)
        {
            m.dimensions.push_back(2);
            m.thicknesses.push_back(r);
        }
        m.tri_indices.push_back(Vector3i{0 + static_cast<IndexT>(base_vertex),
                                         1 + static_cast<IndexT>(base_vertex),
                                         2 + static_cast<IndexT>(base_vertex)});
        m.tri_areas.push_back(area);
    }

    // ---- dim 3 (ARAP: uid 9) ---------------------------------------------
    {
        const IndexT body_id     = 3;
        const SizeT  base_vertex = m.positions.size();
        m.geo_offsets.push_back({0, base_vertex});
        const auto&  p           = desc.tet_positions;
        const Float  V           = (p[1] - p[0]).cross(p[2] - p[0]).dot(p[3] - p[0]) / 6.0;
        std::vector<Float> volume(p.size(), V / 4.0);
        push_vertices(p, desc.tet_density, body_id, volume);
        for(size_t i = 0; i < p.size(); ++i)
        {
            m.dimensions.push_back(3);
            m.thicknesses.push_back(0.0);
        }
        for(auto& t : desc.tets)
        {
            m.tet_indices.push_back(Vector4i{t[0] + static_cast<IndexT>(base_vertex),
                                             t[1] + static_cast<IndexT>(base_vertex),
                                             t[2] + static_cast<IndexT>(base_vertex),
                                             t[3] + static_cast<IndexT>(base_vertex)});
            Matrix3x3 Dm;
            Dm.col(0) = p[t[1]] - p[t[0]];
            Dm.col(1) = p[t[2]] - p[t[0]];
            Dm.col(2) = p[t[3]] - p[t[0]];
            m.tet_volumes.push_back(Dm.determinant() / 6.0);
            m.tet_Dm_inv.push_back(Dm.inverse());
        }
    }

    return m;
}
}  // namespace
namespace
{
/// h for the central differences (brief §4.4).
constexpr double kFDStep = 1e-6;

/// `E(x) = Q(F(x))` with a fixed quadratic `Q`, so that dE/dF and d2E/dF2 have
/// hand-written closed forms and the projected gradient/Hessian can be checked
/// against central differences of `E` itself. All constants are host doubles.
struct QuadraticEnergy
{
    // w[a*3+b] = 0.7 + 0.1*(a*3+b), c[a*3+b] = 0.3*sin(a*3+b+1), lambda = 1.3
    double w[9];
    double c[9];
    double lambda = 1.3;

    QuadraticEnergy()
    {
        for(int k = 0; k < 9; ++k)
        {
            w[k] = 0.7 + 0.1 * k;
            c[k] = 0.3 * std::sin(static_cast<double>(k) + 1.0);
        }
    }

    double E(const Matrix3x3& F) const
    {
        double e = 0.0;
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
            {
                const int k = b * 3 + a;
                e += 0.5 * w[k] * F(a, b) * F(a, b) + c[k] * F(a, b);
            }
        const double tr = F(0, 0) + F(1, 1) + F(2, 2);
        e += 0.5 * lambda * tr * tr;
        return e;
    }

    /// dQ/dF flattened exactly like `flatten(F)` (index = column * 3 + row).
    Vector9 dEdF(const Matrix3x3& F) const
    {
        Vector9 g;
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
            {
                const int k = b * 3 + a;
                g[k]       = w[k] * F(a, b) + c[k];
            }
        const double tr = F(0, 0) + F(1, 1) + F(2, 2);
        for(int a = 0; a < 3; ++a)
            g[a * 3 + a] += lambda * tr;
        return g;
    }

    /// d2Q/dF2 in the same flattening, as used by `project_F_hessian_block`
    /// (`ddEddF(3 * column + row, 3 * column' + row')`).
    Matrix9x9 ddEddF() const
    {
        Matrix9x9 H = Matrix9x9::Zero();
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
            {
                const int k = b * 3 + a;
                H(k, k) += w[k];
            }
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
                H(a * 3 + a, b * 3 + b) += lambda;
        return H;
    }
};

/// A fixed, well-conditioned, non-axis-aligned reference tet.
TetDesc fd_tet()
{
    TetDesc t;
    t.X0 = Vector3{0.1, -0.2, 0.3};
    t.X1 = Vector3{1.3, 0.1, -0.2};
    t.X2 = Vector3{-0.2, 1.4, 0.25};
    t.X3 = Vector3{0.3, -0.1, 1.15};
    t.x0 = t.X0 + Vector3{0.05, -0.03, 0.02};
    t.x1 = Vector3{1.1, 0.2, 0.1};
    t.x2 = Vector3{-0.1, 1.2, 0.35};
    t.x3 = Vector3{0.25, 0.05, 1.0};
    return t;
}

/// E(x) for the FD tet: F(x) = Ds(x) * DmInv (the rest tet fixes DmInv, so E is a
/// quadratic function of the 12 unknown vertex coordinates).
double energy_of(const TetDesc& rest,
                 const Vector3&  x0,
                 const Vector3&  x1,
                 const Vector3&  x2,
                 const Vector3&  x3,
                 const QuadraticEnergy& q)
{
    const auto DmInv = gg::to_eigen(t51fem::Dm_inv(rest.X0, rest.X1, rest.X2, rest.X3));
    Matrix3x3  F;
    F.col(0) = x1 - x0;
    F.col(1) = x2 - x0;
    F.col(2) = x3 - x0;
    return q.E(F * DmInv);
}

Vector3* vertex_ptr(TetDesc& t, int i)
{
    switch(i)
    {
        case 0: return &t.x0;
        case 1: return &t.x1;
        case 2: return &t.x2;
        default: return &t.x3;
    }
}

const Vector3* vertex_cptr(const TetDesc& t, int i)
{
    switch(i)
    {
        case 0: return &t.x0;
        case 1: return &t.x1;
        case 2: return &t.x2;
        default: return &t.x3;
    }
}

/// The expected cell sizes of the surface reporter's host model (a *host model*
/// derived from the scene's `is_surf` attributes, not a hardcoded constant: the
/// tetmesh/linemesh factories do not necessarily create the sub-simplices).
struct SurfModel
{
    std::vector<IndexT>    vertices;
    std::vector<Vector2i>  edges;
    std::vector<Vector3i>  triangles;
};

/// The FEM surface reporter walks the *method's* geometry order (dim, uid) and
/// offsets each geometry's surface indices by the method-local vertex offset plus
/// its own slice offset in the global vertex array. The host model must do the
/// same (`geos` is the method-order (slot index, method vertex offset) list).
SurfModel compute_surf_model(uipc::backend::WorldVisitor&                 world,
                             const std::vector<std::pair<IndexT, SizeT>>& geos,
                             IndexT                                        slice_offset)
{
    SurfModel m;
    for(auto& [slot_index, local_offset] : geos)
    {
        auto sc = world.scene()
                      .geometries()[slot_index]
                      ->geometry()
                      .as<SimplicialComplex>();
        if(!sc)
        {
            continue;
        }
        const IndexT vertex_base = static_cast<IndexT>(local_offset) + slice_offset;

        if(auto attr = sc->vertices().find<IndexT>(builtin::is_surf))
        {
            auto v = attr->view();
            for(size_t i = 0; i < v.size(); ++i)
                if(v[i])
                    m.vertices.push_back(static_cast<IndexT>(i) + vertex_base);
        }
        if(auto attr = sc->edges().find<IndexT>(builtin::is_surf))
        {
            auto v     = attr->view();
            auto edges = sc->edges().topo().view();
            for(size_t i = 0; i < v.size(); ++i)
                if(v[i])
                {
                    m.edges.push_back(Vector2i{edges[i][0] + vertex_base,
                                               edges[i][1] + vertex_base});
                }
        }
        if(auto attr = sc->triangles().find<IndexT>(builtin::is_surf))
        {
            auto v    = attr->view();
            auto tris = sc->triangles().topo().view();
            for(size_t i = 0; i < v.size(); ++i)
                if(v[i])
                {
                    m.triangles.push_back(Vector3i{tris[i][0] + vertex_base,
                                                   tris[i][1] + vertex_base,
                                                   tris[i][2] + vertex_base});
                }
        }
    }
    return m;
}

template <typename T>
bool subsequence(const std::vector<T>& needle, const std::vector<T>& haystack)
{
    if(needle.empty())
    {
        return true;
    }
    for(size_t start = 0; start + needle.size() <= haystack.size(); ++start)
    {
        bool all = true;
        for(size_t i = 0; i < needle.size(); ++i)
        {
            if(!(needle[i] == haystack[start + i]))
            {
                all = false;
                break;
            }
        }
        if(all)
        {
            return true;
        }
    }
    return false;
}
}  // namespace
// ---------------------------------------------------------------------------
//  1) closed forms for the unit tet + the utility dump
// ---------------------------------------------------------------------------
TEST_CASE("fem_base_utils_host_oracle", "[luisa][fem]")
{
    const auto tet = unit_tet();

    // ---- Ds / Dm^-1 / shape gradients (hand-derived) ----------------------
    auto Dm = t51fem::Ds(tet.X0, tet.X1, tet.X2, tet.X3);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(Dm(r, c), (r == c) ? 1.0 : 0.0) <= kEps);

    auto DmInv = t51fem::Dm_inv(tet.X0, tet.X1, tet.X2, tet.X3);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(DmInv(r, c), (r == c) ? 1.0 : 0.0) <= kEps);

    auto sg = t51fem::tetrahedron_shape_gradients(DmInv);
    // grad_0 = -(1,1,1), grad_1 = (1,0,0), grad_2 = (0,1,0), grad_3 = (0,0,1)
    const double sg_expected[3][4] = {
        {-1.0, 1.0, 0.0, 0.0}, {-1.0, 0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0, 1.0}};
    for(int r = 0; r < 3; ++r)
        for(int v = 0; v < 4; ++v)
            CHECK(rel_err(sg(r, v), sg_expected[r][v]) <= kEps);

    // partition of unity: the shape gradients sum to zero (so a rigid translation
    // has zero energy gradient)
    for(int r = 0; r < 3; ++r)
    {
        double s = 0.0;
        for(int v = 0; v < 4; ++v)
            s += sg(r, v);
        CHECK(std::abs(s) <= kEps);
    }

    // ---- rest volume V = det(Ds)/6 = 1/6 ---------------------------------
    const double V = Dm.determinant() / 6.0;
    CHECK(rel_err(V, 1.0 / 6.0) <= kEps);

    // ---- deformation gradient of the undeformed tet is the identity -------
    auto F = t51fem::F(tet.x0, tet.x1, tet.x2, tet.x3, DmInv);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(F(r, c), (r == c) ? 1.0 : 0.0) <= kEps);

    // ---- B matrix = dFdx (9x12): B(r, c) = -1 for c == r%3, +1 for c == r+3 ---
    auto B = t51fem::dFdx(DmInv);
    for(int r = 0; r < 9; ++r)
    {
        for(int c = 0; c < 12; ++c)
        {
            double expected = 0.0;
            if(c == r % 3)
                expected = -1.0;
            else if(c == r + 3)
                expected = 1.0;
            CHECK(rel_err(B(r, c), expected) <= kEps);
        }
        double row_sum = 0.0;
        for(int c = 0; c < 12; ++c)
            row_sum += B(r, c);
        CHECK(std::abs(row_sum) <= kEps);
    }

    // ---- flatten / unflatten / ddot --------------------------------------
    lps::LMat<double, 3, 3> A;
    double                  A_values[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            A(r, c) = A_values[r * 3 + c];
    auto v9 = gg::flatten(A);
    // column-major (Eigen's coefficient order)
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(v9[c * 3 + r], A(r, c)) <= kEps);
    auto A2 = gg::unflatten(v9);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(A2(r, c), A(r, c)) <= kEps);
    CHECK(rel_err(gg::ddot(A, A), (A.transpose() * A).trace()) <= kEps);

    // ---- invariants and dJ/dF on F = I -----------------------------------
    CHECK(rel_err(fem::invariant2(F), 3.0) <= kEps);
    CHECK(rel_err(fem::invariant3(F), 1.0) <= kEps);
    lps::LVec<double, 3> a{0.5, -1.5, 2.0};
    CHECK(rel_err(fem::invariant4(F, a), lps::dot(a, a)) <= kEps);
    CHECK(rel_err(fem::invariant5(F, a), lps::dot(a, a)) <= kEps);
    auto dJ = fem::dJdF(F);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(dJ(r, c), (r == c) ? 1.0 : 0.0) <= kEps);
    // dJdF is the cofactor matrix: F^T * dJdF = det(F) * I
    auto cof = F.transpose() * dJ;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(cof(r, c), (r == c) ? 1.0 : 0.0) <= kEps);

    // ---- SVD / polar decomposition (T2.2 wrappers) -----------------------
    lps::LMat<double, 3, 3> U, Vm, R, S, M;
    lps::LVec<double, 3>    Sigma;
    double                  M_values[9] = {1.2, 0.3, -0.2, 0.1, 0.9, 0.4, -0.3, 0.2, 1.4};
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            M(r, c) = M_values[r * 3 + c];
    gg::svd(M, U, Sigma, Vm);
    CHECK(rel_err(lps::determinant(U), 1.0) <= 1e-10);
    CHECK(rel_err(lps::determinant(Vm), 1.0) <= 1e-10);
    auto reconstructed = U * Sigma.asDiagonal() * Vm.transpose();
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(reconstructed(r, c), M(r, c)) <= 1e-10);
    gg::polar_decomposition(M, R, S);
    auto RS = R * S;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(rel_err(RS(r, c), M(r, c)) <= 1e-10);
    CHECK(rel_err((S - S.transpose()).norm(), 0.0) <= 1e-10);
    CHECK(rel_err((R * R.transpose() - lps::LMat<double, 3, 3>::Identity()).norm(), 0.0)
          <= 1e-10);

    // ---- clamp_to_spd: eigenvalues clamped at 0, reconstruction exact -----
    lps::LMat<double, 9, 9> Q;
    lps::LVec<double, 9>    lambda;
    gg::evd(lps::LMat<double, 9, 9>::Identity(), lambda, Q);
    // make it indefinite by flipping one eigenvalue
    lambda(0) = -3.0;
    lambda(1) = 0.5;
    auto                                   Hindef = Q * lambda.asDiagonal() * Q.transpose();
    auto                                   Hspd   = gg::clamp_to_spd(Hindef);
    lps::LVec<double, 9>                   ls;
    lps::LMat<double, 9, 9>                Qs;
    gg::evd(Hspd, ls, Qs);
    for(int i = 0; i < 9; ++i)
        CHECK(ls(i) >= -1e-9);
    CHECK(rel_err(ls(0), 0.0) <= 1e-9);
    CHECK(rel_err(ls(1), 0.5) <= 1e-9);
    // clamping a PSD matrix is the identity
    lps::LVec<double, 9> two;
    for(int i = 0; i < 9; ++i)
        two(i) = 2.0;
    auto Hpsd_target = Q * two.asDiagonal() * Q.transpose();
    CHECK(rel_err(gg::clamp_to_spd(Hpsd_target).norm(), Hpsd_target.norm()) <= 1e-9);
    // ---- the frontend volume pass, cross-checked against V/4 -------------
    {
        std::vector<Vector3>  Vs{tet.X0, tet.X1, tet.X2, tet.X3};
        std::vector<Vector4i> Ts{Vector4i{0, 1, 2, 3}};
        auto mesh = tetmesh(span<const Vector3>{Vs}, span<const Vector4i>{Ts});
        auto volume = compute_vertex_volume(mesh);
        auto vv     = view(*volume);
        for(size_t i = 0; i < vv.size(); ++i)
        {
            CHECK(rel_err(vv[i], V / 4.0) <= kEps);
        }
    }

    // ---- the shared generic dump routine walks exactly kDumpCount values -
    {
        std::vector<double> host(kDumpCount, 0.0);
        int                 k = 0;
        fem_util_round<double>(
            [&](int i, double value)
            {
                host[i] = value;
                ++k;
            });
        CHECK(k == kDumpCount);
    }
}

// ---------------------------------------------------------------------------
//  2) affine-map recovery: F(x) = applied F, independent of the translation,
//     for a rotation / a stretch / a reflection (det F < 0)
// ---------------------------------------------------------------------------
TEST_CASE("fem_base_utils_affine_map", "[luisa][fem]")
{
    const auto rest = unit_tet();
    const auto DmInv = t51fem::Dm_inv(rest.X0, rest.X1, rest.X2, rest.X3);

    const auto R = rotation_matrix(Vector3{0.3, -0.7, 0.5}, 0.9);

    Matrix3x3 stretch = Matrix3x3::Zero();
    stretch(0, 0)     = 1.5;
    stretch(1, 1)     = 0.5;
    stretch(2, 2)     = 2.0;
    stretch           = R * stretch * R.transpose();

    Matrix3x3 reflection = Matrix3x3::Identity();
    reflection(2, 2)     = -1.0;  // det = -1

    struct Case
    {
        const char* name;
        Matrix3x3   F;
    };
    const Case cases[] = {{"rotation", R}, {"stretch", stretch}, {"reflection", reflection}};

    for(const auto& c : cases)
    {
        const auto deformed_a = affine_tet(rest, c.F, Vector3{0.0, 0.0, 0.0});
        const auto deformed_b = affine_tet(rest, c.F, Vector3{7.5, -3.25, 11.0});

        auto Fa = t51fem::F(deformed_a.x0, deformed_a.x1, deformed_a.x2, deformed_a.x3, DmInv);
        auto Fb = t51fem::F(deformed_b.x0, deformed_b.x1, deformed_b.x2, deformed_b.x3, DmInv);

        for(int r = 0; r < 3; ++r)
            for(int col = 0; col < 3; ++col)
            {
                // the recovered F equals the applied F ...
                CHECK(rel_err(Fa(r, col), c.F(r, col)) <= 1e-12);
                // ... and the translation cancels exactly
                CHECK(rel_err(Fb(r, col), Fa(r, col)) <= 1e-12);
            }

        // invariant3 tracks the sign of the applied determinant: the reference
        // does *not* reject a reflected (det F < 0) deformation here -- only the
        // rest volume is checked, in the method's k3 kernel.
        CHECK(rel_err(fem::invariant3(Fa), c.F.determinant()) <= 1e-12);
        if(c.F.determinant() < 0.0)
        {
            CHECK(fem::invariant3(Fa) < 0.0);
            // the signed-SVD convention absorbs the reflection into Sigma(2)
            lps::LMat<double, 3, 3> U, Vm;
            lps::LVec<double, 3>    Sigma;
            gg::svd(Fa, U, Sigma, Vm);
            CHECK(rel_err(lps::determinant(U), 1.0) <= 1e-10);
            CHECK(rel_err(lps::determinant(Vm), 1.0) <= 1e-10);
            CHECK(Sigma(2) < 0.0);
            auto rec = U * Sigma.asDiagonal() * Vm.transpose();
            for(int r = 0; r < 3; ++r)
                for(int col = 0; col < 3; ++col)
                    CHECK(rel_err(rec(r, col), c.F(r, col)) <= 1e-9);
        }
    }

    // A non-axis-aligned *rest* tet with the identity deformation: Ds != I but
    // the recovered F must be exactly I. A bug that ignores DmInv (e.g. using
    // `Ds` as the deformation gradient) would return Ds here.
    {
        const auto rest2   = fd_tet();
        const auto DmInv2  = t51fem::Dm_inv(rest2.X0, rest2.X1, rest2.X2, rest2.X3);
        const auto ident   = affine_tet(rest2, Matrix3x3::Identity(), Vector3{1.0, 2.0, 3.0});
        auto       F       = t51fem::F(ident.x0, ident.x1, ident.x2, ident.x3, DmInv2);
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                CHECK(rel_err(F(r, c), (r == c) ? 1.0 : 0.0) <= 1e-12);

        // ... while its rest shape matrix is far from the identity
        auto Ds2 = t51fem::Ds(rest2.X0, rest2.X1, rest2.X2, rest2.X3);
        double off_identity = 0.0;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                off_identity += std::abs(Ds2(r, c) - ((r == c) ? 1.0 : 0.0));
        CHECK(off_identity > 1.0);
    }
}

// ---------------------------------------------------------------------------
//  3) finite-difference oracles (central differences, h = 1e-6)
// ---------------------------------------------------------------------------
TEST_CASE("fem_base_utils_finite_difference", "[luisa][fem]")
{
    const auto            rest = fd_tet();
    const QuadraticEnergy q;

    const auto DmInv   = t51fem::Dm_inv(rest.X0, rest.X1, rest.X2, rest.X3);
    const auto DmInv_e = gg::to_eigen(DmInv);  // Eigen notation for the hand math
    const auto sg      = t51fem::tetrahedron_shape_gradients(DmInv);

    auto current_F = [&]() -> Matrix3x3
    {
        Matrix3x3 F;
        F.col(0) = rest.x1 - rest.x0;
        F.col(1) = rest.x2 - rest.x0;
        F.col(2) = rest.x3 - rest.x0;
        return F * DmInv_e;
    };

    const Matrix3x3 F = current_F();
    const Vector9   dEdF = q.dEdF(F);
    lps::LMat<double, 9, 9> ddEddF;
    {
        const Matrix9x9 H_e = q.ddEddF();
        for(int r = 0; r < 9; ++r)
            for(int c = 0; c < 9; ++c)
                ddEddF(r, c) = H_e(r, c);
    }

    // ---- gradient: project_F_gradient vs central differences of E ---------
    double worst_grad_abs = 0.0;
    double worst_grad_rel = 0.0;
    for(int i = 0; i < 4; ++i)
    {
        lps::LVec<double, 9> dEdF_v;
        for(int k = 0; k < 9; ++k)
            dEdF_v(k) = dEdF[k];
        const auto analytic = fem::project_F_gradient(sg.col(i), dEdF_v);

        for(int k = 0; k < 3; ++k)
        {
            auto plus  = rest;
            auto minus = rest;
            (*vertex_ptr(plus, i))(k) += kFDStep;
            (*vertex_ptr(minus, i))(k) -= kFDStep;

            const double ep = energy_of(rest, plus.x0, plus.x1, plus.x2, plus.x3, q);
            const double em = energy_of(rest, minus.x0, minus.x1, minus.x2, minus.x3, q);
            const double fd = (ep - em) / (2.0 * kFDStep);

            const double diff = std::abs(fd - analytic(k));
            worst_grad_abs    = std::max(worst_grad_abs, diff);
            worst_grad_rel    = std::max(worst_grad_rel, rel_err(fd, analytic(k)));
        }
    }
    std::printf("[t51] FD gradient: worst abs %.3e, worst rel %.3e\n",
                worst_grad_abs,
                worst_grad_rel);
    CHECK(worst_grad_rel <= 1e-4);

    // ---- Hessian: project_F_hessian_block vs central differences of grad ---
    // The (i, j) block of the exact Hessian, assembled from the projections.
    Matrix12x12 H = Matrix12x12::Zero();
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
        {
            const auto block = fem::project_F_hessian_block(sg.col(i), ddEddF, sg.col(j));
            for(int r = 0; r < 3; ++r)
                for(int c = 0; c < 3; ++c)
                    H(i * 3 + r, j * 3 + c) = block(r, c);
        }

    // symmetry (H == H^T) -- an exact structural property
    double worst_sym = 0.0;
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            worst_sym = std::max(worst_sym, std::abs(H(r, c) - H(c, r)));
    std::printf("[t51] Hessian asymmetry: %.3e\n", worst_sym);
    CHECK(worst_sym <= 1e-12);

    // positive semidefiniteness: Q is convex (w > 0, lambda > 0), so the exact
    // Hessian must be PSD. The reference guarantees it for the FEM energy
    // (it is a Gram matrix of shape gradients).
    Eigen::SelfAdjointEigenSolver<Matrix12x12> es(H);
    const double min_eig = es.eigenvalues().minCoeff();
    std::printf("[t51] Hessian min eigenvalue: %.3e (lambda_max %.3e)\n",
                min_eig,
                es.eigenvalues().maxCoeff());
    CHECK(min_eig >= -1e-9);

    // finite differences of the projected gradient
    double worst_hess_abs = 0.0;
    double worst_hess_rel = 0.0;
    for(int i = 0; i < 2; ++i)  // two blocks is enough to cover the pattern
    {
        for(int k = 0; k < 3; ++k)
        {
            auto plus  = rest;
            auto minus = rest;
            (*vertex_ptr(plus, i))(k) += kFDStep;
            (*vertex_ptr(minus, i))(k) -= kFDStep;

            // gradient of the *plus/minus* state, at the other vertex j
            auto grad_of = [&](const TetDesc& t, int j) -> lps::LVec<double, 3>
            {
                Matrix3x3 Ft;
                Ft.col(0) = t.x1 - t.x0;
                Ft.col(1) = t.x2 - t.x0;
                Ft.col(2) = t.x3 - t.x0;
                const Matrix3x3      Fv = Ft * DmInv_e;
                lps::LVec<double, 9> g;
                const Vector9        d = q.dEdF(Fv);
                for(int m = 0; m < 9; ++m)
                    g(m) = d[m];
                return fem::project_F_gradient(sg.col(j), g);
            };

            for(int j = 0; j < 4; ++j)
            {
                const auto gp = grad_of(plus, j);
                const auto gm = grad_of(minus, j);
                for(int c = 0; c < 3; ++c)
                {
                    const double fd = (gp(c) - gm(c)) / (2.0 * kFDStep);
                    const double an = H(i * 3 + k, j * 3 + c);
                    worst_hess_abs  = std::max(worst_hess_abs, std::abs(fd - an));
                    worst_hess_rel  = std::max(worst_hess_rel, rel_err(fd, an));
                }
            }
        }
    }
    std::printf("[t51] FD Hessian: worst abs %.3e, worst rel %.3e\n",
                worst_hess_abs,
                worst_hess_rel);
    CHECK(worst_hess_rel <= 1e-3);

    // ---- dFdx vs central differences of F (exact for an affine map) --------
    {
        const auto B       = t51fem::dFdx(DmInv);
        double     worst   = 0.0;
        for(int l = 0; l < 12; ++l)
        {
            const int  vertex = l / 3;
            const int  comp   = l % 3;
            auto       plus   = rest;
            (*vertex_ptr(plus, vertex))(comp) += kFDStep;

            Matrix3x3 Fplus;
            Fplus.col(0) = plus.x1 - plus.x0;
            Fplus.col(1) = plus.x2 - plus.x0;
            Fplus.col(2) = plus.x3 - plus.x0;
            Matrix3x3 F0;
            F0.col(0) = rest.x1 - rest.x0;
            F0.col(1) = rest.x2 - rest.x0;
            F0.col(2) = rest.x3 - rest.x0;

            const Matrix3x3 dF = (Fplus - F0) * DmInv_e / kFDStep;
            for(int r = 0; r < 9; ++r)
            {
                worst = std::max(worst, rel_err(dF(r % 3, r / 3), B(r, l)));
            }
        }
        std::printf("[t51] dFdx vs FD: worst rel %.3e\n", worst);
        CHECK(worst <= 1e-8);
    }

    // ---- dJdF vs central differences of det(F) ----------------------------
    {
        const auto dJ    = fem::dJdF(gg::to_luisa(F));
        double     worst = 0.0;
        for(int k = 0; k < 9; ++k)
        {
            auto plus  = F;
            auto minus = F;
            plus(k / 3, k % 3) += kFDStep;
            minus(k / 3, k % 3) -= kFDStep;
            const double fd = (plus.determinant() - minus.determinant()) / (2.0 * kFDStep);
            worst           = std::max(worst, rel_err(fd, dJ(k / 3, k % 3)));
        }
        std::printf("[t51] dJdF vs FD: worst rel %.3e\n", worst);
        CHECK(worst <= 1e-8);
    }
}

// ---------------------------------------------------------------------------
//  4) degenerate inputs: the reference's behaviour, pinned from the source
// ---------------------------------------------------------------------------
TEST_CASE("fem_base_utils_degenerate_inputs", "[luisa][fem]")
{
    const auto good = unit_tet();

    // ---- valid input: V > 0 so the k3 device assert is NOT vacuous ---------
    {
        auto Dm = t51fem::Ds(good.X0, good.X1, good.X2, good.X3);
        CHECK(Dm.determinant() / 6.0 > 0.0);
    }

    // ---- zero-volume tet (X3 == X0): det(Ds) == 0 --------------------------
    // Reference behaviour (finite_element_method.cu kernel k3):
    //     Dm9x9_inv(i) = t51fem::Dm_inv(...);            // 3x3 closed-form inverse
    //     Float V = t51fem::Ds(...).determinant() / 6.0;
    //     UIPC_KERNEL_ASSERT(V > 0.0, "Negative volume tetrahedron (%d,...)");
    // i.e. (a) the inverse is taken *first* and divides by det == 0, and (b) the
    // assert then aborts the dispatch. The port keeps both: `lps::inverse` is the
    // same divide-by-determinant closed form, and `UIPC_KERNEL_ASSERT` maps to
    // LC's real device assert (`lcrt::device_assert`). Dispatching this case
    // would abort the process on both backends, so the test asserts the
    // *condition* and the pre-assert arithmetic on the host instead.
    {
        auto degenerate = good;
        degenerate.X3   = degenerate.X0;
        auto Dm         = t51fem::Ds(degenerate.X0, degenerate.X1, degenerate.X2, degenerate.X3);
        CHECK(Dm.determinant() == 0.0);
        CHECK(!(Dm.determinant() / 6.0 > 0.0));  // the assert condition is false

        auto DmInv = t51fem::Dm_inv(degenerate.X0, degenerate.X1, degenerate.X2, degenerate.X3);
        bool any_non_finite = false;
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                any_non_finite = any_non_finite || !std::isfinite(DmInv(r, c));
        CHECK(any_non_finite);
    }

    // ---- inverted tet (det(Ds) < 0) ---------------------------------------
    {
        auto inverted = good;
        inverted.X3.z() = -1.0;
        const double V  = t51fem::Ds(inverted.X0, inverted.X1, inverted.X2, inverted.X3).determinant()
                        / 6.0;
        CHECK(V < 0.0);
        CHECK(!(V > 0.0));
        // the reference logs/aborts with the four tet indices; the port's
        // device_assert keeps the check but cannot format device values into the
        // message (LC takes a fixed string), see finite_element_method.cpp
        CHECK(std::abs(V) == 1.0 / 6.0);
    }

    // ---- degenerate triangle (duplicated vertices): area == 0, no branch ---
    // The reference's shell kernel is unconditional:
    //     rest_areas(i) = 0.5 * E01.cross(E02).norm();
    // so a zero-area (or duplicated-vertex) triangle simply stores 0 and the
    // material basis is never checked -- there is no assert for shells.
    {
        TriDesc t{Vector3{0.0, 0.0, 0.0}, Vector3{1.0, 0.0, 0.0}, Vector3{1.0, 0.0, 0.0}};
        CHECK(tri_area(t) == 0.0);
        TriDesc dup{Vector3{2.0, 2.0, 2.0}, Vector3{2.0, 2.0, 2.0}, Vector3{2.0, 2.0, 2.0}};
        CHECK(tri_area(dup) == 0.0);
    }

    // ---- duplicated edge endpoints: rest length == 0, no branch -----------
    {
        const Vector3 p{1.0, 2.0, 3.0};
        CHECK((p - p).norm() == 0.0);
    }

    // ---- the reference's *extent* check is a real branch -------------------
    // (utils/report_extent_check.h): a reporter that returns `hessian_count != 0`
    // without querying `gradient_only()` and a gradient-only reporter with a
    // hessian both abort (UIPC_ASSERT -> uipc::abort()). Only the valid
    // combinations are exercised here.
    CHECK_NOTHROW(gg::check_report_extent(true, false, 7, "t51_valid"));
    CHECK_NOTHROW(gg::check_report_extent(true, true, 0, "t51_valid_gradient_only"));
}
// ---------------------------------------------------------------------------
//  5) device == host for the whole ported numeric surface
//     (the same generic routine instantiated as S = double and as
//      S = lcrt::Var<double> inside a Kernel1D, compared element by element)
// ---------------------------------------------------------------------------
TEST_CASE("fem_base_utils_device_vs_host", "[luisa][fem]")
{
    test_runtime();

    std::vector<double> host(kDumpCount, 0.0);
    int                 k = 0;
    fem_util_round<double>(
        [&](int i, double value)
        {
            host[i] = value;
            ++k;
        });
    REQUIRE(k == kDumpCount);

    lud::LDeviceBuffer<double> device(kDumpCount);
    auto&                      stream = gg::frame_stream();
    stream.record(util_dump_kernel()(device.view()).dispatch(1u));

    auto device_host = read_device(device.cview());
    REQUIRE(device_host.size() == host.size());

    int    bitwise_equal = 0;
    double worst_abs     = 0.0;
    double worst_idx     = -1;
    for(int i = 0; i < kDumpCount; ++i)
    {
        if(std::memcmp(&device_host[i], &host[i], sizeof(double)) == 0)
        {
            ++bitwise_equal;
        }
        const double diff = std::abs(device_host[i] - host[i]);
        if(diff > worst_abs)
        {
            worst_abs = diff;
            worst_idx = i;
        }
    }
    std::printf("[t51] device-vs-host dump: %d/%d bitwise equal, worst abs %.3e (entry %.0f)\n",
                bitwise_equal,
                kDumpCount,
                worst_abs,
                worst_idx);

    // The arithmetic is the same code on both sides and the measured result is
    // *bitwise* identical (T2.1's convention: "requires bit-identical results"),
    // so the requirement stays bitwise. If a future backend contraction
    // (FMA, T3.4 rule 8) makes that too strict, the bound printed above is the
    // fallback -- measured 0.0 here, so 1e-12 is still orders of magnitude
    // tighter than any real discrepancy.
    CHECK(bitwise_equal == kDumpCount);
    CHECK(worst_abs <= 1e-12);

    // entry 9 is `ddot(F, F)`: hand-derived from the fixed F of the dump routine
    // (1.2^2 + 0.3^2 + (-0.2)^2 + 0.1^2 + 0.9^2 + 0.4^2 + (-0.3)^2 + 0.2^2 + 1.4^2
    //  = 4.64, the Frobenius norm squared of a *non-symmetric* matrix, so the
    // transpose convention is covered too)
    {
        const double entries[9] = {1.2, 0.3, -0.2, 0.1, 0.9, 0.4, -0.3, 0.2, 1.4};
        double       expected   = 0.0;
        for(double x : entries)
            expected += x * x;
        CHECK(rel_err(expected, 4.64) <= 1e-12);
        CHECK(rel_err(device_host[9], expected) <= 1e-12);
    }
}

// ---------------------------------------------------------------------------
//  6) a real scene: the method's device buffers, the reporters' slices and the
//     state the scene describes. All expectations are independent host models.
// ---------------------------------------------------------------------------
TEST_CASE("fem_base_scene_reporters", "[luisa][fem]")
{
    test_runtime();

    const auto desc = t51_scene_desc();
    t51::reset_records();
    const auto model = compute_fem_host_model(desc);

    auto  scene = build_scene(desc);
    auto  workspace = workspace_path("uipc_luisa_fem_base_test");
    Engine engine{"luisa", workspace};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    REQUIRE(T51FEMProbe::fem != nullptr);
    auto& fem_method = *T51FEMProbe::fem;
    // World exposes the scene through the backend visitor (the reference's
    // systems use the same accessor)
    auto visitor = uipc::backend::WorldVisitor{world};

    // ---- the test constitutions saw the scene -----------------------------
    for(int i = 0; i < 4; ++i)
    {
        auto& r = t51::rec(static_cast<t51::Dim>(i));
        CHECK(r.build_count == 1);
        CHECK(r.init_count == 1);
        if(i == 0)
        {
            // Codim0DConstitution::do_report_extent is `final` in the reference
            // (it reports all-zero extents), so a 0d constitution's own override
            // is never called -- the base's `report_extent` calls it *and* then
            // `info.check(name())`, which passes because hessian_count == 0
            // without `gradient_only()` having been queried.
            CHECK(r.report_extent_count == 0);
            CHECK(!r.gradient_only_queried);
        }
        else
        {
            CHECK(r.report_extent_count == 1);
            CHECK(r.gradient_only_queried);
        }
    }
    // dim 3: one tet geometry (4 vertices, 1 primitive)
    CHECK(t51::rec(t51::Dim::d3).vertex_count == 4);
    CHECK(t51::rec(t51::Dim::d3).primitive_count == 1);
    CHECK(t51::rec(t51::Dim::d3).geo_info_count == 1);
    // dim 2: the shell
    CHECK(t51::rec(t51::Dim::d2).vertex_count == 3);
    CHECK(t51::rec(t51::Dim::d2).primitive_count == 1);
    // dim 1: the rod (2 edges)
    CHECK(t51::rec(t51::Dim::d1).vertex_count == 3);
    CHECK(t51::rec(t51::Dim::d1).primitive_count == 2);
    // dim 0: the two particles
    CHECK(t51::rec(t51::Dim::d0).vertex_count == 2);
    CHECK(t51::rec(t51::Dim::d0).primitive_count == 2);
    // the extra constitution found the shell through `extra_constitution_uids`
    CHECK(t51::rec(t51::Dim::extra).build_count == 1);
    CHECK(t51::rec(t51::Dim::extra).init_count == 1);
    CHECK(t51::rec(t51::Dim::extra).geo_info_count == 1);
    CHECK(t51::rec(t51::Dim::extra).report_extent_count == 1);

    // FilteredInfo::for_each: `is_fixed` is 0 for every FEM vertex of the tet
    // constitution (the frontend's default)
    CHECK(t51::rec(t51::Dim::d3).for_each_is_fixed.size() == 4);
    for(auto v : t51::rec(t51::Dim::d3).for_each_is_fixed)
        CHECK(v == 0);

    // the dimension base's BaseInfo surface: a view into the method's buffers at
    // this constitution's offsets, plus the *full* vertex buffers
    CHECK(t51::rec(t51::Dim::d3).base_indices_size == 1);
    CHECK(t51::rec(t51::Dim::d3).base_dm_size == 1);
    CHECK(t51::rec(t51::Dim::d3).base_volume_size == 1);
    CHECK(t51::rec(t51::Dim::d3).xs_size == model.positions.size());
    CHECK(t51::rec(t51::Dim::d3).xbar_size == model.positions.size());

    // ---- the method's device buffers --------------------------------------
    auto xs    = read_vec3(fem_method.xs());
    auto xbars = read_vec3(fem_method.x_bars());
    REQUIRE(xs.size() == model.positions.size());
    REQUIRE(xbars.size() == model.positions.size());
    for(size_t i = 0; i < xs.size(); ++i)
    {
        // xs: the method applied the shell's mesh transform to its positions
        CHECK(rel_err((xs[i] - model.positions[i]).norm(), 0.0) <= 1e-12);
    }
    // x_bars comes from the *rest* geometry, whose positions were never
    // transformed, so only the shell's entries differ from xs
    for(size_t i = 0; i < xbars.size(); ++i)
    {
        if(std::abs((xbars[i] - xs[i]).norm()) > 1e-12)
        {
            // method order: points (2), rod (3), shell (3), tet (4) => the shell
            // vertices are the only ones whose rest copy bypassed the mesh
            // transform, so only they may differ
            CHECK(i >= 2 + 3);
            CHECK(i < 2 + 3 + 3);
        }
    }

    auto thicknesses = read_device(fem_method.thicknesses());
    auto is_dynamic  = read_device(fem_method.is_dynamic());
    auto is_fixed    = read_device(fem_method.is_fixed());
    auto gravities   = read_vec3(fem_method.gravities());
    auto masses      = read_device(fem_method.masses());
    REQUIRE(thicknesses.size() == model.thicknesses.size());
    REQUIRE(masses.size() == model.masses.size());
    for(size_t i = 0; i < masses.size(); ++i)
    {
        CHECK(rel_err(thicknesses[i], model.thicknesses[i]) <= 1e-12);
        CHECK(rel_err(masses[i], model.masses[i]) <= 1e-12);
        CHECK(rel_err((gravities[i] - model.gravities[i]).norm(), 0.0) <= 1e-12);
        CHECK(is_fixed[i] == 0);
        CHECK(is_dynamic[i] == 1);
    }

    auto tets        = read_vec4i(fem_method.tets());
    auto rest_volumes = read_device(fem_method.rest_volumes());
    auto Dm3x3_invs  = read_mat3(fem_method.Dm3x3_invs());
    REQUIRE(tets.size() == model.tet_indices.size());
    REQUIRE(rest_volumes.size() == model.tet_volumes.size());
    REQUIRE(Dm3x3_invs.size() == model.tet_Dm_inv.size());
    for(size_t i = 0; i < tets.size(); ++i)
    {
        for(int c = 0; c < 4; ++c)
            CHECK(tets[i][c] == model.tet_indices[i][c]);
        // V = det(Ds)/6, hand-derived for the unit tet: 1/6
        CHECK(rel_err(rest_volumes[i], model.tet_volumes[i]) <= 1e-12);
        CHECK(rel_err(rest_volumes[i], 1.0 / 6.0) <= 1e-12);
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                CHECK(rel_err(Dm3x3_invs[i](r, c), model.tet_Dm_inv[i](r, c)) <= 1e-12);
    }

    auto tris       = read_vec3i(fem_method.codim_2ds());
    auto rest_areas = read_device(fem_method.rest_areas());
    REQUIRE(tris.size() == model.tri_indices.size());
    REQUIRE(rest_areas.size() == model.tri_areas.size());
    for(size_t i = 0; i < tris.size(); ++i)
    {
        for(int c = 0; c < 3; ++c)
            CHECK(tris[i][c] == model.tri_indices[i][c]);
        // the hand-derived area (3) of the rotated triangle
        CHECK(rel_err(rest_areas[i], model.tri_areas[i]) <= 1e-12);
        CHECK(rel_err(rest_areas[i], 3.0) <= 1e-12);
    }

    auto edges         = read_device(fem_method.codim_1ds());
    auto rest_lengths  = read_device(fem_method.rest_lengths());
    REQUIRE(edges.size() == model.edge_indices.size());
    REQUIRE(rest_lengths.size() == model.edge_lengths.size());
    for(size_t i = 0; i < edges.size(); ++i)
    {
        CHECK(edges[i][0] == model.edge_indices[i][0]);
        CHECK(edges[i][1] == model.edge_indices[i][1]);
        CHECK(rel_err(rest_lengths[i], model.edge_lengths[i]) <= 1e-12);
    }
    // the two hand-computed edge lengths of the rod (1 and sqrt(2))
    CHECK(rel_err(rest_lengths[0], 1.0) <= 1e-12);
    CHECK(rel_err(rest_lengths[1], std::sqrt(2.0)) <= 1e-12);

    auto points = read_device(fem_method.codim_0ds());
    REQUIRE(points.size() == model.point_indices.size());
    for(size_t i = 0; i < points.size(); ++i)
        CHECK(points[i] == model.point_indices[i]);

    // the shell mesh's transform was applied and reset to identity
    {
        auto geo_slots = visitor.scene().geometries();
        auto shell_sc  = geo_slots[1]->geometry().as<SimplicialComplex>();
        REQUIRE(shell_sc != nullptr);
        auto transforms = shell_sc->transforms().view();
        CHECK(rel_err((transforms[0] - Matrix4x4::Identity()).norm(), 0.0) <= 1e-12);
        auto pos = shell_sc->positions().view();
        for(size_t i = 0; i < pos.size(); ++i)
        {
            const Vector3 expected = desc.shell_transform * desc.shell_positions[i];
            CHECK(rel_err((pos[i] - expected).norm(), 0.0) <= 1e-12);
        }
    }

    // ---- the reporters' slices -------------------------------------------
    REQUIRE(T51FEMProbe::vertex_reporter != nullptr);
    REQUIRE(T51FEMProbe::body_reporter != nullptr);
    auto& vertex_reporter = *T51FEMProbe::vertex_reporter;
    auto& body_reporter   = *T51FEMProbe::body_reporter;

    CHECK(vertex_reporter.vertex_count() == static_cast<IndexT>(model.positions.size()));
    CHECK(body_reporter.body_count() == 4);

    // the global vertex array's slice owned by the FEM reporter holds exactly the
    // method's positions, in the method's (dim, uid) order
    REQUIRE(T51FEMProbe::vertex_manager != nullptr);
    auto& vertex_manager = *T51FEMProbe::vertex_manager;
    auto  gvm_positions  = read_vec3(vertex_manager.positions());
    auto  gvm_body_ids   = read_device(vertex_manager.body_ids());
    const auto vertex_offset = vertex_reporter.vertex_offset();
    const auto body_offset   = body_reporter.body_offset();
    REQUIRE(gvm_positions.size() >= model.positions.size());
    for(size_t i = 0; i < model.positions.size(); ++i)
    {
        CHECK(rel_err((gvm_positions[vertex_offset + i] - xs[i]).norm(), 0.0) <= 1e-12);
        CHECK(gvm_body_ids[vertex_offset + i] == body_offset + model.body_ids[i]);
    }

    // per-geometry `global_vertex_offset` written at frame 0
    {
        auto geo_slots = visitor.scene().geometries();
        for(auto& [slot_index, local_offset] : model.geo_offsets)
        {
            auto sc = geo_slots[slot_index]->geometry().as<SimplicialComplex>();
            REQUIRE(sc != nullptr);
            auto attr = sc->meta().find<IndexT>(builtin::global_vertex_offset);
            REQUIRE(attr != nullptr);
            CHECK(view(*attr)[0]
                  == static_cast<IndexT>(local_offset) + vertex_offset);
        }
    }

    // the surface reporter's slices (*global* sample indices, so they are located
    // inside the manager's concatenated arrays rather than assumed to own them,
    // cookbook §7.14 rule 6)
    {
        auto expected = compute_surf_model(visitor, model.geo_offsets, vertex_offset);

        REQUIRE(T51FEMProbe::surface_manager != nullptr);
        auto& surface_manager = *T51FEMProbe::surface_manager;
        auto  surf_vertices   = read_device(surface_manager.surf_vertices());
        auto  surf_edges      = read_vec2i(surface_manager.surf_edges());
        auto  surf_triangles  = read_vec3i(surface_manager.surf_triangles());

        CHECK(subsequence(expected.vertices, surf_vertices));
        CHECK(subsequence(expected.edges, surf_edges));
        CHECK(subsequence(expected.triangles, surf_triangles));
        // the shell's single triangle must be there in global vertex numbering
        if(!expected.triangles.empty())
        {
            CHECK(subsequence(expected.triangles, surf_triangles));
        }
    }

    // ---- advance() keeps the rest state (no dynamics in T5.1) -------------
    {
        auto before_xbars = read_vec3(fem_method.x_bars());
        auto before_volumes = read_device(fem_method.rest_volumes());
        world.advance();
        REQUIRE(world.is_valid());
        world.sync();
        auto after_xbars = read_vec3(fem_method.x_bars());
        auto after_volumes = read_device(fem_method.rest_volumes());
        CHECK(arrays_equal(before_xbars, after_xbars));
        CHECK(arrays_equal(before_volumes, after_volumes));
    }
}

// ---------------------------------------------------------------------------
//  7) the FEM state accessor (the public feature API)
// ---------------------------------------------------------------------------
TEST_CASE("fem_base_state_accessor", "[luisa][fem]")
{
    test_runtime();

    const auto desc = t51_scene_desc();
    t51::reset_records();
    const auto model = compute_fem_host_model(desc);

    auto   scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_fem_state_test")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    REQUIRE(T51FEMProbe::fem != nullptr);
    auto& fem_method = *T51FEMProbe::fem;

    auto feature = world.features().find<FiniteElementStateAccessorFeature>();
    REQUIRE(feature != nullptr);
    const SizeT vertex_count = fem_method.xs().size();
    CHECK(feature->vertex_count() == vertex_count);

    auto state_geo = feature->create_geometry();
    CHECK(state_geo.vertices().size() == vertex_count);
    // the feature creates neither position nor velocity by default
    CHECK(state_geo.vertices().find<Vector3>(builtin::position) == nullptr);
    auto pos_attr = state_geo.vertices().create<Vector3>(builtin::position, Vector3::Zero());
    auto vel_attr = state_geo.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());

    // ---- copy_to: the device state lands in the geometry ------------------
    feature->copy_to(state_geo);
    auto xs = read_vec3(fem_method.xs());
    {
        auto pos = view(*pos_attr);
        REQUIRE(pos.size() == model.positions.size());
        for(size_t i = 0; i < pos.size(); ++i)
            CHECK(rel_err((pos[i] - xs[i]).norm(), 0.0) <= 1e-12);
    }

    // ---- copy_from: a modified geometry is written back to the device -----
    {
        auto pos = view(*pos_attr);
        auto vel = view(*vel_attr);
        for(size_t i = 0; i < pos.size(); ++i)
        {
            pos[i] += Vector3{0.25, -0.5, 1.0};
            vel[i] = Vector3{1.0 + static_cast<double>(i), 0.0, -1.0};
        }
    }
    feature->copy_from(state_geo);
    {
        auto xs_after = read_vec3(fem_method.xs());
        auto vs_after = read_vec3(fem_method.vs());
        REQUIRE(xs_after.size() == xs.size());
        for(size_t i = 0; i < xs.size(); ++i)
        {
            CHECK(rel_err((xs_after[i] - (xs[i] + Vector3{0.25, -0.5, 1.0})).norm(), 0.0)
                  <= 1e-12);
            const Vector3 expected_vel{1.0 + static_cast<double>(i), 0.0, -1.0};
            CHECK(rel_err((vs_after[i] - expected_vel).norm(), 0.0) <= 1e-12);
        }
    }

    // ---- the externally-owned buffer exports ------------------------------
    // `backend::BufferView::handle()` is a caller-owned *host* address (that is
    // what the pybind frontend passes); see the deviation note in
    // finite_element_state_accessor_feature.cpp.
    {
        std::vector<Vector3> host_out(vertex_count, Vector3::Zero());
        // copy the first half only, through a ByteView whose handle points at the
        // host array
        const IndexT half = static_cast<IndexT>(vertex_count / 2);
        uipc::backend::BufferView buffer_view{
            reinterpret_cast<uipc::backend::HandleT>(host_out.data() + 0),
            0,
            static_cast<SizeT>(half),
            sizeof(Vector3),
            sizeof(Vector3),
            "luisa"};
        feature->copy_position_to(buffer_view, 0, static_cast<SizeT>(half));
        auto xs_now = read_vec3(fem_method.xs());
        for(IndexT i = 0; i < half; ++i)
            CHECK(rel_err((host_out[i] - xs_now[i]).norm(), 0.0) <= 1e-12);

        std::vector<Vector3> host_vel(vertex_count, Vector3::Zero());
        uipc::backend::BufferView vel_view{
            reinterpret_cast<uipc::backend::HandleT>(host_vel.data()),
                                     0,
                                     static_cast<SizeT>(half),
            sizeof(Vector3),
            sizeof(Vector3),
            "luisa"};
        feature->copy_velocity_to(vel_view, 0, static_cast<SizeT>(half));
        auto vs_now = read_vec3(fem_method.vs());
        for(IndexT i = 0; i < half; ++i)
            CHECK(rel_err((host_vel[i] - vs_now[i]).norm(), 0.0) <= 1e-12);
    }

    // ---- the write-back is picked up by the vertex reporter next frame -----
    {
        REQUIRE(T51FEMProbe::vertex_manager != nullptr);
        REQUIRE(T51FEMProbe::vertex_reporter != nullptr);
        auto& vertex_manager = *T51FEMProbe::vertex_manager;
        const auto offset    = T51FEMProbe::vertex_reporter->vertex_offset();

        world.advance();
        REQUIRE(world.is_valid());
        world.sync();

        auto gvm_positions = read_vec3(vertex_manager.positions());
        auto xs_now        = read_vec3(fem_method.xs());
        for(size_t i = 0; i < xs_now.size(); ++i)
            CHECK(rel_err((gvm_positions[offset + i] - xs_now[i]).norm(), 0.0) <= 1e-12);
    }
}

// ---------------------------------------------------------------------------
//  8) dump / recover of the FEM state
// ---------------------------------------------------------------------------
TEST_CASE("fem_base_dump_recover", "[luisa][fem]")
{
    test_runtime();

    const auto desc = t51_scene_desc();
    t51::reset_records();

    const auto workspace = workspace_path("uipc_luisa_fem_dump_test");

    std::vector<Vector3> dumped_positions;
    std::vector<Vector3> dumped_velocities;
    {
        auto   scene = build_scene(desc);
        Engine engine{"luisa", workspace};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());

        REQUIRE(T51FEMProbe::fem != nullptr);
        auto& fem_method = *T51FEMProbe::fem;

        // make the state distinguishable from a fresh initialization
        {
            auto feature = world.features().find<FiniteElementStateAccessorFeature>();
            REQUIRE(feature != nullptr);
            auto state_geo = feature->create_geometry();
            auto pos = state_geo.vertices().create<Vector3>(builtin::position, Vector3::Zero());
            auto vel = state_geo.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());
            feature->copy_to(state_geo);
            auto pos_view = view(*pos);
            auto vel_view = view(*vel);
            for(size_t i = 0; i < pos_view.size(); ++i)
            {
                pos_view[i] += Vector3{3.0, 4.0, 5.0};
                vel_view[i] = Vector3{0.5 * static_cast<double>(i), -1.0, 2.0};
            }
            feature->copy_from(state_geo);
        }

        REQUIRE(world.dump());
        dumped_positions  = read_vec3(fem_method.xs());
        dumped_velocities = read_vec3(fem_method.vs());
    }

    // ---- engine 2: a fresh scene, then recover ---------------------------
    {
        auto   scene = build_scene(desc);
        Engine engine{"luisa", workspace};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());

        REQUIRE(T51FEMProbe::fem != nullptr);
        auto& fem_method = *T51FEMProbe::fem;

        auto fresh_positions = read_vec3(fem_method.xs());
        auto fresh_xbars     = read_vec3(fem_method.x_bars());
        REQUIRE(fresh_positions.size() == dumped_positions.size());
        bool differs = false;
        for(size_t i = 0; i < fresh_positions.size(); ++i)
            differs = differs || (fresh_positions[i] - dumped_positions[i]).norm() > 1e-9;
        CHECK(differs);

        CHECK(world.recover());
        auto recovered_positions  = read_vec3(fem_method.xs());
        auto recovered_velocities = read_vec3(fem_method.vs());
        CHECK(arrays_equal(recovered_positions, dumped_positions));
        CHECK(arrays_equal(recovered_velocities, dumped_velocities));

        // the reference dumps xs/vs/x_prevs only: x_bars and the derived volumes
        // are re-established by init() from the fresh scene (asserted explicitly
        // rather than expecting a recovery the reference does not perform)
        CHECK(arrays_equal(read_vec3(fem_method.x_bars()), fresh_xbars));
        // ... and the recovered positions really are the dumped ones (not the
        // fresh scene's), so the check above is not vacuous
        CHECK(rel_err((recovered_positions[0] - fresh_positions[0]).norm(), 0.0) > 1e-9);
    }
}
