/********************************************************************
 * @file luisa_abd_base_test.cpp
 * @brief T6.1a — ABD base: types, jacobi matrices, dynamics, energy,
 *        kinetic/constitution interfaces, reporters and the state accessor
 *        (LuisaCompute backend).
 *
 * Every oracle in this file is derived independently *here*: the reference backend
 * is CUDA-only and cannot be host-included, so nothing is copied from it.  The
 * oracles are
 *   (a) closed forms for a box: for [0,L]^3 with density rho,
 *           m    = rho L^3
 *           mx   = m (L/2)(1,1,1)
 *           mxx  = m L^2/3 on the diagonal, m L^2/4 off it
 *       (the integrals int_rho x_a x_b dV, evaluated exactly), which hold for *any*
 *       tetrahedralisation of the box -- so the frontend's per-tet sum is checked
 *       against the region integral, not against a repetition of it;
 *   (b) the *definition* of the ABD mass matrix, M = int rho J^T J dV with
 *       J = dp/dq, expanded by hand into its (m, mx, mxx) blocks (the derivation is
 *       in `mass_matrix_oracle`);
 *   (c) the ABD shape energy in its operator form,  kappa v ||A A^T - I||_F^2,
 *       which is what the ported `shape_energy` (the sum form) must equal;
 *   (d) central finite differences (h = 1e-6) on (c) for the energy gradient and
 *       the Hessian, plus the Hessian symmetry invariant;
 *   (e) structural invariants (jacobi/point-map agreement, M vs J^T J assembly,
 *       rigid-translation and rotation laws of the moments, M inverse identity);
 *   (f) the device-vs-host equivalence of every ported routine
 *       (S = double vs S = lcrt::Var<double> inside a Kernel1D), and
 *   (g) a real scene driven through the public Engine/World API, where the ABD
 *       dynamics' device buffers and the four reporters are compared against the
 *       same host model.
 *
 * Coverage map:
 *  1) box volume/mass/mass-inverse/gravity/q, rotated+scaled box, dim 2 vs 3
 *     ............................................... abd_base_single_cube
 *  2) ABDJacobi/ABDJacobiDyadicMass vs hand-derived + the flag paths
 *     ............................................... abd_base_jacobi
 *  3) shape energy / kinetic terms + finite differences
 *     ............................................... abd_base_kinetic_and_energy
 *  4) state accessor (copy_from/copy_to/the two exports)
 *     ............................................... abd_base_state_accessor
 *  5) real scene: one ABD body, reporters, one advance()
 *     ............................................... abd_base_scene
 *  6) device == host for the whole ported routine surface
 *     ............................................... abd_base_device_vs_host
 *
 * Reachable uids: the frontend's `uipc::constitution::AffineBodyConstitution`
 * writes uid **1** ("OrthoPotential") into the geometry's `constitution_uid` meta
 * attribute, and T6.3a owns the *backend* constitution of that uid.  T6.1a has no
 * backend ABD constitution at all, so this file registers a test-local mock that
 * claims uid 1 and does the group's work -- exactly the T5.1/T5.2 arrangement for
 * the FEM uids (T5.1's mock claims ARAP's uid 9 and T5.2 reports the collision).
 * When T6.3a lands, the group goes to whichever system registered first and this
 * file's scene case will report which one owns it instead of asserting blindly
 * (the mock records what `do_init` handed *it*).
 *
 * Shared-binary rule (cookbook §7.14 rule 6): every system registered by every test
 * file is built for every scene, so this file asserts only on *its own* slices (via
 * `VertexReporter::vertex_offset()`/`BodyReporter::body_offset()`) and its own
 * scenes are ABD-only.
 *********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/geometry/utils/label_triangle_orient.h>
#include <uipc/geometry/utils/compute_mesh_volume.h>
#include <uipc/geometry/utils/affine_body/compute_dyadic_mass.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/core/affine_body_state_accessor_feature.h>

#include <affine_body/type_define.h>
#include <affine_body/utils.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <affine_body/abd_energy.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_constitution.h>
#include <affine_body/affine_body_body_reporter.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <affine_body/affine_body_surface_reporter.h>
#include <affine_body/affine_body_external_force_manager.h>
#include <affine_body/affine_body_animator.h>
#include <animator/global_animator.h>
#include <global_geometry/global_body_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <time_integrator/bdf1_flag.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <numbers>
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

namespace
{
// ---------------------------------------------------------------------------
//  small helpers
// ---------------------------------------------------------------------------
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

std::vector<Vector12> read_vec12(const lud::LCBufferView<gg::LVector12>& view)
{
    auto                  raw = read_device(view);
    std::vector<Vector12> host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

std::vector<Matrix12x12> read_mat12(const lud::LCBufferView<gg::LMatrix12x12>& view)
{
    auto                     raw = read_device(view);
    std::vector<Matrix12x12> host;
    host.reserve(raw.size());
    for(auto& m : raw)
        host.push_back(gg::to_eigen(m));
    return host;
}

std::vector<IndexT> read_index(const lud::LCBufferView<IndexT>& view)
{
    return read_device(view);
}

std::vector<Vector3> read_vec3(const lud::LCBufferView<gg::LVector3>& view)
{
    auto                  raw = read_device(view);
    std::vector<Vector3> host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

std::vector<Vector2i> read_vec2i(const lud::LCBufferView<gg::LVector2i>& view)
{
    auto                  raw = read_device(view);
    std::vector<Vector2i> host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

std::vector<Vector3i> read_vec3i(const lud::LCBufferView<gg::LVector3i>& view)
{
    auto                  raw = read_device(view);
    std::vector<Vector3i> host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

/// Every entry of `sub` appears in `all`, in order (the reporters' slices are
/// *global* index ranges, cookbook §7.14 rule 6).
template <typename T>
bool subsequence(const std::vector<T>& sub, const std::vector<T>& all)
{
    if(sub.size() > all.size())
        return false;
    size_t i = 0;
    for(const auto& v : all)
    {
        if(i == sub.size())
            break;
        if(v == sub[i])
            ++i;
    }
    return i == sub.size();
}

// ---------------------------------------------------------------------------
//  Eigen <-> numerics-layer shims (the T5.1 convention): the hand-derived oracles
//  are written in Eigen notation, the ported routines work on the numerics layer's
//  containers.
// ---------------------------------------------------------------------------
[[nodiscard]] lps::LVec<double, 3> lv(const Vector3& v) noexcept
{
    return lps::LVec<double, 3>{v(0), v(1), v(2)};
}

[[nodiscard]] lps::LVec<double, 12> lv12(const Vector12& q) noexcept
{
    lps::LVec<double, 12> out;
    for(int i = 0; i < 12; ++i)
        out(i) = q(i);
    return out;
}

[[nodiscard]] lps::LMat<double, 3, 3> lm3(const Matrix3x3& m) noexcept
{
    lps::LMat<double, 3, 3> out;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] Matrix3x3 to_eigen3(const lps::LMat<double, 3, 3>& m) noexcept
{
    Matrix3x3 out;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] Vector3 to_eigen3(const lps::LVec<double, 3>& v) noexcept
{
    return Vector3{v(0), v(1), v(2)};
}

[[nodiscard]] Vector12 to_eigen12(const lps::LVec<double, 12>& q) noexcept
{
    Vector12 out;
    for(int i = 0; i < 12; ++i)
        out(i) = q(i);
    return out;
}

[[nodiscard]] Matrix12x12 to_eigen12(const lps::LMat<double, 12, 12>& m) noexcept
{
    Matrix12x12 out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] gg::Matrix3x12 to_eigen3x12(const lps::LMat<double, 3, 12>& m) noexcept
{
    gg::Matrix3x12 out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 3; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] Matrix9x9 to_eigen9(const lps::LMat<double, 9, 9>& m) noexcept
{
    Matrix9x9 out;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
            out(r, c) = m(r, c);
    return out;
}

// ---------------------------------------------------------------------------
//  q <-> A conventions (hand-derived from the reference's own layout)
//
//  q = [ t ; a1 ; a2 ; a3 ]  with a_i = ***row i of A***
//  (affine_body/utils.h `q_to_A` reads A(i,j) = q(3 + 3i + j) and
//   `abd_jacobi_point` computes p = t + (x.a1, x.a2, x.a3) = t + A^T x)
// ---------------------------------------------------------------------------
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
    Vector12 q;
    q.segment<3>(0) = t;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            q(3 + 3 * i + j) = A(i, j);
    return q;
}

/// p = A x + t  (the reference's `ABDJacobi::point_x` / `operator*`:
/// `{x.a1, x.a2, x.a3} + t` with a_i = row i of A, i.e. the *row* form `A x + t`)
[[nodiscard]] Vector3 point_oracle(const Vector3& x_bar, const Vector12& q) noexcept
{
    return A_oracle(q) * x_bar + q.segment<3>(0);
}

/// J = dp/dq for the convention above:  J(a, a) = 1, J(i, 3+3i+j) = x_j.
[[nodiscard]] gg::Matrix3x12 jacobi_oracle(const Vector3& x_bar) noexcept
{
    gg::Matrix3x12 J = gg::Matrix3x12::Zero();
    for(int a = 0; a < 3; ++a)
        J(a, a) = 1.0;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            J(i, 3 + 3 * i + j) = x_bar(j);
    return J;
}

// ---------------------------------------------------------------------------
//  box oracle (brief §3.1): the *region* integrals, not the per-tet sum
//
//  int_{[0,L]^3} x_a dV      = L^4/2      -> mx = rho L^4/2 (1,1,1) = m (L/2)(1,1,1)
//  int_{[0,L]^3} x_a^2 dV    = L^5/3      -> mxx_aa = rho L^5/3      = m L^2/3
//  int_{[0,L]^3} x_a x_b dV  = L^2(L^2/2)(L^2/2) = L^6/4 -> mxx_ab = m L^2/4
// ---------------------------------------------------------------------------
struct BoxMoments
{
    Float     volume = 0.0;
    Float     m      = 0.0;
    Vector3   mx     = Vector3::Zero();
    Matrix3x3 mxx    = Matrix3x3::Zero();
};

BoxMoments box_moments(Float L, Float rho)
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

/// The six-tet decomposition of the cube [0,L]^3 (all positively oriented, each
/// tet a monotone path from v0 to v7; the determinants are +L^3, checked in the
/// test).  Vertex index = x + 2y + 4z.
std::array<Vector4i, 6> cube_tets() noexcept
{
    return {{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}}};
}

std::array<Vector3, 8> cube_vertices(Float L) noexcept
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

/// The 12 triangles of the cube surface (outward oriented, [0,L]^3), i.e. the same
/// region's boundary -- used for the dim-2 ABD body.
/// The boundary of the cube's six-tet decomposition: every tet's four faces, with
/// the *shared* ones (count 2) dropped -- i.e. the cube surface, triangulated by the
/// decomposition's own face diagonals.  This is the independent model of what
/// `label_surface` hands to the surface reporter (`label_surface`'s orientation
/// convention is not ours to predict, so triangles are compared canonically).
std::pair<std::vector<IndexT>, std::vector<Vector3i>> boundary_oracle(IndexT vertex_offset)
{
    const std::array<Vector4i, 6> ts     = cube_tets();
    const int                     faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

    std::map<std::array<IndexT, 3>, int> counts;
    for(auto& t : ts)
    {
        for(auto& f : faces)
        {
            std::array<IndexT, 3> key{t(f[0]) + vertex_offset,
                                      t(f[1]) + vertex_offset,
                                      t(f[2]) + vertex_offset};
            std::sort(key.begin(), key.end());
            counts[key] += 1;
        }
    }

    std::vector<IndexT>   vertices;
    std::vector<Vector3i> triangles;
    for(auto& [k, c] : counts)
    {
        if(c != 1)
            continue;  // shared face: interior
        triangles.push_back(Vector3i{k[0], k[1], k[2]});
        for(auto v : k)
        {
            if(std::find(vertices.begin(), vertices.end(), v) == vertices.end())
                vertices.push_back(v);
        }
    }
    std::sort(vertices.begin(), vertices.end());
    std::sort(triangles.begin(), triangles.end(), [](const Vector3i& a, const Vector3i& b)
              { return std::tie(a(0), a(1), a(2)) < std::tie(b(0), b(1), b(2)); });
    return {vertices, triangles};
}

/// Canonical (vertex-sorted) copy of a triangle list, sorted lexicographically.
std::vector<Vector3i> canonical(std::vector<Vector3i> tris)
{
    for(auto& t : tris)
    {
        std::array<IndexT, 3> k{t(0), t(1), t(2)};
        std::sort(k.begin(), k.end());
        t = Vector3i{k[0], k[1], k[2]};
    }
    std::sort(tris.begin(), tris.end(), [](const Vector3i& a, const Vector3i& b)
              { return std::tie(a(0), a(1), a(2)) < std::tie(b(0), b(1), b(2)); });
    return tris;
}

bool same_triangles(const std::vector<Vector3i>& a, const std::vector<Vector3i>& b)
{
    if(a.size() != b.size())
        return false;
    for(size_t i = 0; i < a.size(); ++i)
        for(int k = 0; k < 3; ++k)
            if(a[i](k) != b[i](k))
                return false;
    return true;
}

std::array<Vector3i, 12> cube_triangles() noexcept
{
    return {{{0, 2, 3}, {0, 3, 1},   // z = 0
             {4, 5, 7}, {4, 7, 6},   // z = L
             {0, 1, 5}, {0, 5, 4},   // y = 0
             {2, 6, 7}, {2, 7, 3},   // y = L
             {0, 4, 6}, {0, 6, 2},   // x = 0
             {1, 3, 7}, {1, 7, 5}}}; // x = L
}

// ---------------------------------------------------------------------------
//  the ABD mass matrix, derived from its definition
//
//  M = int rho J(x)^T J(x) dV,  J = dp/dq  (3x12),
//  with  J(a,a) = 1  and  J(i, 3+3i+j) = x_j   for the q convention above:
//
//    M(a,b)               = int rho  d_{a,b} dV                       = m d_{a,b}
//    M(a, 3+3i+j)         = int rho  d_{a,i} x_j dV                   = d_{a,i} mx_j
//    M(3+3i+j, 3+3k+l)    = int rho  d_{i,k} x_j x_l dV               = d_{i,k} mxx_{j,l}
//
//  i.e.  block(0,0) = m I3,  block(0,1+i) = e_i mx^T,
//        block(1+i,0) = mx e_i^T,  block(1+i,1+k) = d_{i,k} Mxx.
// ---------------------------------------------------------------------------
Matrix12x12 mass_matrix_oracle(Float m, const Vector3& mx, const Matrix3x3& mxx)
{
    Matrix12x12 M = Matrix12x12::Zero();
    for(int a = 0; a < 3; ++a)
        M(a, a) = m;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            // block(0, 1+i): only row i is non-zero
            M(i, 3 + 3 * i + j) = mx(j);
            // block(1+i, 0): only column i is non-zero
            M(3 + 3 * i + j, i) = mx(j);
        }
        // block(1+i, 1+i)
        M.block<3, 3>(3 + 3 * i, 3 + 3 * i) = mxx;
    }
    return M;
}

/// The ABD shape energy in operator form:  kappa v ||A A^T - I||_F^2  (the
/// reference's `abd_energy` sum form equals it, see `abd_base_kinetic_and_energy`).
double shape_energy_oracle(const Vector12& q, double kappa, double volume)
{
    const auto A = A_oracle(q);
    return kappa * volume * (A * A.transpose() - Matrix3x3::Identity()).squaredNorm();
}

/// The kinetic term  T = 1/2 q_v^T M q_v  with M built by `mass_matrix_oracle`.
double kinetic_energy_oracle(const Vector12& q_v, const Matrix12x12& M)
{
    return 0.5 * q_v.dot(M * q_v);
}

// ---------------------------------------------------------------------------
//  THE FIXED DUMP INPUTS of `abd_round` (shared by the host and the device run)
// ---------------------------------------------------------------------------
template <class S>
lps::LVec<S, 12> fixed_q()
{
    // A = R(rotation) * diag(1.3, 0.9, 1.1) + a small upper-triangular perturbation
    // (a deformed, well-conditioned affine state) and t = (0.1, -0.2, 0.3)
    Matrix3x3 A = (Eigen::AngleAxisd(0.7, Vector3{1.0, 2.0, -0.5}.normalized())
                   * Vector3{1.3, 0.9, 1.1}.asDiagonal());
    A(0, 1) += 0.05;
    A(1, 2) -= 0.03;
    const Vector12 q = make_q(A, Vector3{0.1, -0.2, 0.3});

    lps::LVec<S, 12> out;
    for(int i = 0; i < 12; ++i)
        out(i) = lps::make_scalar<S>(q(i));
    return out;
}

template <class S>
lps::LVec<S, 3> fixed_v3(double x, double y, double z)
{
    lps::LVec<S, 3> out;
    out(0) = lps::make_scalar<S>(x);
    out(1) = lps::make_scalar<S>(y);
    out(2) = lps::make_scalar<S>(z);
    return out;
}

template <class S>
lps::LMat<S, 3, 3> fixed_m3(const double v[9])
{
    lps::LMat<S, 3, 3> out;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            out(r, c) = lps::make_scalar<S>(v[c * 3 + r]);
    return out;
}

/// A well-conditioned, non-symmetric "Hessian"
template <class S>
lps::LMat<S, 3, 3> fixed_H()
{
    const double v[9] = {2.1, 0.3, -0.2, 0.3, 1.7, 0.4, -0.2, 0.4, 2.9};
    return fixed_m3<S>(v);
}

/// The dyadic mass of the unit box with L = 0.5, rho = 1000 (a real three-moment
/// triple, not a made-up one), scaled by 1e-3 so the dump stays in a sane range.
template <class S>
gg::ABDJacobiDyadicMassT<S> fixed_dyadic_mass()
{
    const auto o = box_moments(0.5, 1000.0);
    gg::ABDJacobiDyadicMassT<S> m;
    m.m_mass                    = lps::make_scalar<S>(o.m);
    m.m_mass_times_x_bar        = fixed_v3<S>(o.mx(0), o.mx(1), o.mx(2));
    m.m_mass_times_dyadic_x_bar = fixed_m3<S>(o.mxx.data());
    return m;
}

template <class S>
gg::ABDJacobiDyadicMassT<S> fixed_dyadic_mass2()
{
    const auto o = box_moments(0.2, 250.0);
    gg::ABDJacobiDyadicMassT<S> m;
    m.m_mass                    = lps::make_scalar<S>(o.m);
    m.m_mass_times_x_bar        = fixed_v3<S>(o.mx(0), o.mx(1), o.mx(2));
    m.m_mass_times_dyadic_x_bar = fixed_m3<S>(o.mxx.data());
    return m;
}

/// The dump order (brief §3.6): every ported routine, on one fixed input set.
template <typename S, typename Put>
void abd_round(Put&& put)
{
    int k = 0;

    const auto q = fixed_q<S>();
    const auto J1 = gg::ABDJacobiT<S>{.x_bar = fixed_v3<S>(0.3, -0.7, 1.1)};
    const auto J2 = gg::ABDJacobiT<S>{.x_bar = fixed_v3<S>(-0.25, 0.4, 0.6)};
    const auto g  = fixed_v3<S>(0.9, -1.3, 0.2);
    const auto H  = fixed_H<S>();

    // ABDJacobi
    {
        auto p = gg::abd_jacobi_point(J1, q);
        for(int i = 0; i < 3; ++i)
            put(k++, p(i));
    }
    {
        auto v = gg::abd_jacobi_vec_x(J1, q);
        for(int i = 0; i < 3; ++i)
            put(k++, v(i));
    }
    {
        auto m = gg::abd_jacobi_to_mat(J1);
        for(int c = 0; c < 12; ++c)
            for(int r = 0; r < 3; ++r)
                put(k++, m(r, c));
    }
    {
        auto v = gg::abd_jacobi_T_mul(J1, g);
        for(int i = 0; i < 12; ++i)
            put(k++, v(i));
    }
    {
        auto m = gg::abd_jacobi_JT_H_J(J1, H, J2);
        for(int c = 0; c < 12; ++c)
            for(int r = 0; r < 12; ++r)
                put(k++, m(r, c));
    }

    // ABDJacobiDyadicMass
    const auto M1 = fixed_dyadic_mass<S>();
    const auto M2 = fixed_dyadic_mass2<S>();
    {
        auto v = gg::abd_dyadic_mass_mul(M1, q);
        for(int i = 0; i < 12; ++i)
            put(k++, v(i));
    }
    {
        const double hv[144] = {0};
        lps::LMat<S, 12, 12> h;
        for(int c = 0; c < 12; ++c)
            for(int r = 0; r < 12; ++r)
                h(r, c) = lps::make_scalar<S>(hv[c * 12 + r]);
        gg::abd_dyadic_mass_add_to(M1, h);
        for(int c = 0; c < 12; ++c)
            for(int r = 0; r < 12; ++r)
                put(k++, h(r, c));
    }
    {
        auto m = gg::abd_dyadic_mass_to_mat(M1);
        for(int c = 0; c < 12; ++c)
            for(int r = 0; r < 12; ++r)
                put(k++, m(r, c));
    }
    {
        auto c = gg::abd_dyadic_mass_center_of_mass(M1);
        for(int i = 0; i < 3; ++i)
            put(k++, c(i));
    }
    {
        auto I = gg::abd_dyadic_mass_inertia_tensor_cm(M1);
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                put(k++, I(r, c));
    }
    {
        auto acc = M1;
        gg::abd_dyadic_mass_add(acc, M2);
        put(k++, acc.m_mass);
        for(int i = 0; i < 3; ++i)
            put(k++, acc.m_mass_times_x_bar(i));
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                put(k++, acc.m_mass_times_dyadic_x_bar(r, c));
    }

    // affine_body/utils.h
    const auto A = gg::q_to_A(q);
    {
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                put(k++, A(r, c));
    }
    {
        auto v = gg::A_to_q(A);
        for(int i = 0; i < 9; ++i)
            put(k++, v(i));
    }
    {
        lps::LVec<S, 9> F;
        for(int i = 0; i < 9; ++i)
            F(i) = lps::make_scalar<S>(0.4 + 0.3 * i);
        auto v = gg::F_to_A(F);
        for(int i = 0; i < 9; ++i)
            put(k++, v(i));
    }
    {
        const double hf[81] = {0};
        lps::LMat<S, 9, 9> HF;
        for(int c = 0; c < 9; ++c)
            for(int r = 0; r < 9; ++r)
                HF(r, c) = lps::make_scalar<S>(hf[c * 9 + r]);
        auto HA = gg::HF_to_HA(HF);
        for(int c = 0; c < 9; ++c)
            for(int r = 0; r < 9; ++r)
                put(k++, HA(r, c));
    }
    {
        auto S_skew = gg::skew(fixed_v3<S>(0.7, -0.35, 1.25));
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                put(k++, S_skew(r, c));
    }
    {
        auto t = fixed_v3<S>(0.2, 0.9, -0.4);
        lps::LVec<S, 3> n, b;
        gg::orthonormal_basis(t, n, b);
        for(int i = 0; i < 3; ++i)
            put(k++, t(i));
        for(int i = 0; i < 3; ++i)
            put(k++, n(i));
        for(int i = 0; i < 3; ++i)
            put(k++, b(i));
    }
    {
        auto F = gg::torque_to_F(lps::make_scalar<S>(2.5),
                                 fixed_v3<S>(0.0, 0.0, 1.0),
                                 q);
        for(int i = 0; i < 12; ++i)
            put(k++, F(i));
    }

    // abd_energy
    put(k++, gg::shape_energy(q));
    {
        auto gr = gg::shape_energy_gradient(q);
        for(int i = 0; i < 9; ++i)
            put(k++, gr(i));
    }
    {
        auto hess = gg::shape_energy_hessian(q);
        for(int c = 0; c < 9; ++c)
            for(int r = 0; r < 9; ++r)
                put(k++, hess(r, c));
    }
    {
        lps::LMat<S, 3, 3> d1, d2, d3, c12, c13, c23;
        gg::shape_energy_hessian(q, d1, d2, d3, c12, c13, c23);
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                put(k++, d1(r, c));
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                put(k++, c12(r, c));
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                put(k++, c23(r, c));
    }
}

/// The number of entries `abd_round` writes (checked below, so a silently
/// truncated dump cannot pass).
constexpr int kABDDumpCount = 779;

/// The same routine on the device; the single thread writes the whole dump.
auto& abd_dump_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>>(
        [](lcrt::BufferVar<double> out) noexcept
        {
            abd_round<lcrt::Var<double>>(
                [&](int i, auto&& value) noexcept { out->write(i, value); });
        });
    return k;
}
}  // namespace

// ---------------------------------------------------------------------------
//  The test-local ABD constitution (uid 1, the frontend's "OrthoPotential") and
//  the probe that grabs the built systems.
//
//  The mock records exactly what the *public* hooks handed it, so the test can
//  assert the (uid -> group) classification without touching the dynamics'
//  internals from a subclass (the constitution's `m_impl` is private, by design).
// ---------------------------------------------------------------------------
namespace t61
{
inline constexpr U64 kOrthoPotentialUID = 1;  // uipc::constitution::AffineBodyConstitution

struct Rec
{
    int   build_count  = 0;
    int   init_count   = 0;
    SizeT geo_count    = 0;
    SizeT body_count   = 0;
    SizeT vertex_count = 0;

    SizeT for_each_count = 0;
    // per body, in `ForEachInfo::global_index()` order
    std::vector<SizeT>    global_indices;
    std::vector<SizeT>    geo_body_offsets;
    std::vector<Vector12> qs;      // from the geometry's transform
    std::vector<Vector12> q_vs;    // from the geometry's velocity
    std::vector<Float>    kappas;  // per-instance attribute
    std::vector<Float>    volumes; // meta
    std::vector<Float>    shape_energies;    // kappa * v * shape_energy(q)
    std::vector<Float>    kinetic_energies;  // 1/2 q_v^T M q_v
};

inline Rec& rec()
{
    static Rec r;
    return r;
}

inline void reset()
{
    rec() = Rec{};
}
}  // namespace t61

class T61ABDMockConstitution final : public gg::AffineBodyConstitution
{
  public:
    using AffineBodyConstitution::AffineBodyConstitution;

  protected:
    U64 get_uid() const noexcept override { return t61::kOrthoPotentialUID; }

    void do_build(BuildInfo& info) override
    {
        (void)info;
        t61::rec().build_count++;
    }

    void do_init(gg::AffineBodyDynamics::FilteredInfo& info) override
    {
        auto& r = t61::rec();
        r.init_count++;
        r.geo_count    = info.geo_infos().size();
        r.body_count   = info.body_count();
        r.vertex_count = info.vertex_count();

        // Walk the constitution's own bodies through `FilteredInfo::for_each` and
        // evaluate the model energy/kinetic term from the *geometry's* data
        // (kappa/volume/abd_mass meta + the instance transform/velocity) using the
        // ported routines: an end-to-end pass over the same arithmetic the test
        // checks against its own oracle below.
        auto geo_slots = world().scene().geometries();

        r.for_each_count = 0;
        r.global_indices.clear();
        r.geo_body_offsets.clear();
        r.qs.clear();
        r.q_vs.clear();
        r.kappas.clear();
        r.volumes.clear();
        r.shape_energies.clear();
        r.kinetic_energies.clear();

        info.for_each(  //
            geo_slots,
            [](SimplicialComplex& sc) { return sc.transforms().view(); },
            [&](const gg::AffineBodyDynamics::ForEachInfo& I, const Matrix4x4& trans)
            {
                const auto& geo_info = I.geo_info();
                auto        sc_slot  = geo_slots[geo_info.geo_slot_index];
                auto&       sc       = *sc_slot->geometry().as<SimplicialComplex>();

                auto kappa_attr = sc.instances().find<Float>("kappa");
                auto vol_attr   = sc.meta().find<Float>(builtin::volume);
                auto mass_attr  = sc.meta().find<Float>(builtin::abd_mass);
                auto mx_attr    = sc.meta().find<Vector3>(builtin::abd_mass_x_bar);
                auto mxx_attr   = sc.meta().find<Matrix3x3>(builtin::abd_mass_x_bar_x_bar);

                UIPC_ASSERT(kappa_attr && vol_attr && mass_attr && mx_attr && mxx_attr,
                            "T61ABDMockConstitution: missing ABD meta on the geometry");

                const auto local   = I.local_index();
                const auto kappa   = kappa_attr->view()[local];
                const auto volume  = vol_attr->view().front();
                const auto m       = mass_attr->view().front();
                const auto m_x_bar = mx_attr->view().front();
                const auto mxx     = mxx_attr->view().front();

                const Vector12 q = gg::transform_to_q(trans);

                auto vel_attr = sc.instances().find<Matrix4x4>(builtin::velocity);
                const Vector12 q_v =
                    vel_attr ? gg::transform_to_q(vel_attr->view()[local]) : Vector12::Zero();

                // shape energy: kappa * v * shape_energy(q)  (the ported routine)
                const auto E = static_cast<Float>(kappa * volume)
                               * gg::shape_energy(gg::to_luisa(q));

                // kinetic term: 1/2 q_v^T M q_v, M from the *same* meta triple
                const auto M =
                    gg::abd_dyadic_mass_to_mat_eigen(gg::make_abd_dyadic_mass(m, m_x_bar, mxx));
                const auto T = static_cast<Float>(0.5) * q_v.dot(M * q_v);

                r.for_each_count++;
                r.global_indices.push_back(I.global_index());
                r.geo_body_offsets.push_back(geo_info.body_offset);
                r.qs.push_back(q);
                r.q_vs.push_back(q_v);
                r.kappas.push_back(kappa);
                r.volumes.push_back(volume);
                r.shape_energies.push_back(E);
                r.kinetic_energies.push_back(T);
            });
    }

    // The energy/gradient hooks of the constitution interface take the
    // line-search/linear-subsystem info classes, which are T6.4's: T6.1a only
    // needs them to be *declared* (their definitions land with their systems), so
    // the mock leaves the bodies empty and the interface's plumbing is covered by
    // the T6.1b/T6.4 batches.
    void do_compute_energy(ComputeEnergyInfo& info) override { (void)info; }
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override { (void)info; }
};
REGISTER_SIM_SYSTEM(T61ABDMockConstitution);

/// Grabs the ABD systems' public interfaces once they are built (the T4.1/T5.1
/// probe idiom).  A scene without an ABD constitution invalidates it (`require<>`
/// throws SimSystemException), so other test files are unaffected.
class T61ABDProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::AffineBodyDynamics*                  abd = nullptr;
    inline static gg::AffineBodyVertexReporter*            vertex_reporter = nullptr;
    inline static gg::AffineBodyBodyReporter*              body_reporter = nullptr;
    inline static gg::AffinebodySurfaceReporter*           surface_reporter = nullptr;
    inline static gg::AffineBodyExternalForceManager*      external_force_manager = nullptr;
    inline static gg::GlobalVertexManager*                 vertex_manager = nullptr;
    inline static gg::GlobalBodyManager*                   body_manager = nullptr;
    inline static gg::GlobalSimplicialSurfaceManager*      surface_manager = nullptr;
    inline static gg::BDF1Flag*                            bdf1_flag = nullptr;
    inline static gg::AffineBodyStateAccessorFeatureOverrider* state_overrider = nullptr;
    // T6.1b owns AffineBodyAnimator's body; the reference only creates it (through
    // GlobalAnimator) for a scene whose constitution table has a Constraint type.
    // Both are *found*, never required, so their absence cannot invalidate anything.
    inline static gg::AffineBodyAnimator* animator        = nullptr;
    inline static gg::GlobalAnimator*     global_animator = nullptr;

  private:
    void do_build() override
    {
        abd              = &require<gg::AffineBodyDynamics>();
        vertex_reporter  = &require<gg::AffineBodyVertexReporter>();
        body_reporter    = &require<gg::AffineBodyBodyReporter>();
        surface_reporter = &require<gg::AffinebodySurfaceReporter>();
        external_force_manager = &require<gg::AffineBodyExternalForceManager>();

        vertex_manager  = &require<gg::GlobalVertexManager>();
        body_manager    = &require<gg::GlobalBodyManager>();
        surface_manager = find<gg::GlobalSimplicialSurfaceManager>();
        // the integrator flag is only created for a "bdf1" scene (its
        // SimSystemCreator returns nullptr otherwise), so `find` and report
        bdf1_flag = find<gg::BDF1Flag>();
        animator        = find<gg::AffineBodyAnimator>();
        global_animator = find<gg::GlobalAnimator>();
    }
};
REGISTER_SIM_SYSTEM(T61ABDProbe);

namespace
{
// ---------------------------------------------------------------------------
//  scene helpers
// ---------------------------------------------------------------------------
/// The standard ABD test scene config: no ASSET files, no contact (T8.x), no
/// sanity check, gravity left at the default (0, -9.8, 0).
Json abd_scene_config()
{
    auto config                          = Scene::default_config();
    config["sanity_check"]["enable"]     = false;
    config["contact"]["enable"]          = false;
    return config;
}

/// One ABD body: a `SimplicialComplex` with the frontend constitution applied.
struct ABDBodySpec
{
    std::string            name;
    SimplicialComplex      sc;
    SizeT                  body_count = 1;
};

/// Applies the *frontend* ABD constitution (uid 1) to a box tetmesh of side L and
/// density rho, with `body_count` instances; the transform of instance i is set to
/// `transforms[i]` (identity when the vector is empty).
SimplicialComplex make_box_body(Float                          L,
                                Float                          rho,
                                Float                          kappa,
                                SizeT                          body_count,
                                const std::vector<Matrix4x4>&  transforms,
                                const std::vector<Matrix4x4>&  velocities = {},
                                const Matrix3x3&               rest_map   = Matrix3x3::Identity())
{
    // `rest_map` moves the *material* configuration: a rotated/scaled box is a
    // different rest state (the dyadic mass integrates the rest positions), while
    // the instance transform carries the deformed state `q`.
    auto vs = cube_vertices(L);
    if(rest_map != Matrix3x3::Identity())
    {
        for(auto& v : vs)
            v = rest_map * v;
    }
    auto ts = cube_tets();
    auto sc = tetmesh(span<const Vector3>{vs}, span<const Vector4i>{ts});
    // the frontend's own labelling step (T5.1's scene does the same): the ABD
    // dynamics does not need it, the *surface reporter* does (`is_surf`).
    label_surface(sc);
    sc.instances().resize(body_count);

    AffineBodyConstitution abd;
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

/// A dim-2 ABD body: the *surface* of the same box, with an explicit mass matrix
/// (the `apply_to(sc, kappa, mass, volume)` overload).  Used by the dim 2 vs 3
/// check only: T6.1a has no shell constitution, and the dim-2 path of
/// `compute_dyadic_mass` needs a closed, oriented trimesh (T6.3a's shell).
SimplicialComplex make_trimesh_body(Float                          L,
                                    Float                          kappa,
                                    SizeT                          body_count,
                                    const std::vector<Matrix4x4>&  transforms)
{
    auto vs = cube_vertices(L);
    auto fs = cube_triangles();
    auto sc = trimesh(span<const Vector3>{vs}, span<const Vector3i>{fs});
    label_surface(sc);
    sc.instances().resize(body_count);

    const auto o = box_moments(L, 1000.0);
    AffineBodyConstitution abd;
    abd.apply_to(sc, kappa, mass_matrix_oracle(o.m, o.mx, o.mxx), o.volume);

    auto T = view(sc.transforms());
    for(SizeT i = 0; i < body_count; ++i)
        T[i] = (i < transforms.size()) ? transforms[i] : Matrix4x4::Identity();
    return sc;
}

Matrix4x4 make_transform(const Matrix3x3& A, const Vector3& t)
{
    Matrix4x4 T = Matrix4x4::Identity();
    T.block<3, 3>(0, 0) = A;
    T.block<3, 1>(0, 3) = t;
    return T;
}

/// The frontend stores velocities as 4x4 matrices whose last row is unused for
/// `q_v` (`transform_to_q` reads the 3x3 block and the translation only).
Matrix4x4 make_velocity(const Matrix3x3& A, const Vector3& t)
{
    return make_transform(A, t);
}

// ---------------------------------------------------------------------------
//  per-case assertion accounting (brief §6.3)
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

#define T61_CHECK(counter, cond) \
    do                           \
    {                            \
        ++(counter).checks;      \
        CHECK(cond);             \
    } while(false)

}  // namespace

// ===========================================================================
//  1) single cube: volume, dyadic mass, M^-1, gravity, q/q_v, dim 2 vs 3
// ===========================================================================
TEST_CASE("abd_base_single_cube", "[luisa][abd_base]")
{
    test_runtime();
    t61::reset();
    CaseScope scope{"abd_base_single_cube"};

    const Float L     = 0.5;
    const Float rho   = 1000.0;
    const Float kappa = 1.0e8;

    const auto o = box_moments(L, rho);

    // a rotated + scaled box in a *separate* geometry, same density (the moments
    // must follow the rotation/scale laws, which the region integral predicts
    // independently of the tetrahedralisation)
    const double  s   = 1.7;
    const Matrix3x3 R = Eigen::AngleAxisd(0.6, Vector3{1.0, 1.0, 0.0}.normalized())
                            .toRotationMatrix();
    const auto o_rs = box_moments(L * s, rho);

    // instance transforms of body 0 (translation only: q must equal the hand q)
    const Vector3 t0{0.1, -0.2, 0.3};
    // body 1 is the *same* box rotated and scaled in its **material** frame (the
    // geometry's rest positions), so its dyadic mass is the transformed moment
    // triple of `o`; its instance transform is a pure translation.
    const Matrix3x3 A1 = Matrix3x3::Identity();
    const Vector3   t1{-0.4, 0.5, 0.05};

    // velocities: q_v is the transform's 3x3 block + translation
    const Matrix3x3 V0 = (0.1 * Vector3{1.0, 2.0, -1.0}.asDiagonal());
    const Vector3   v0{0.3, -0.1, 0.2};
    const Matrix3x3 V1 = (-0.05 * Vector3{0.5, 1.5, 1.0}.asDiagonal());
    const Vector3   v1{0.0, 0.25, -0.15};

    Json  config = abd_scene_config();
    Scene scene{config};
    auto  object = scene.objects().create("abd_base_single_cube");

    auto body0 = make_box_body(L, rho, kappa, 1, {make_transform(Matrix3x3::Identity(), t0)},
                               {make_velocity(V0, v0)});
    object->geometries().create(body0);

    auto body1 = make_box_body(L, rho, kappa, 1, {make_transform(A1, t1)},
                               {make_velocity(V1, v1)}, s * R);
    object->geometries().create(body1);

    auto body2 = make_trimesh_body(L, kappa, 1, {make_transform(Matrix3x3::Identity(), Vector3{1.0, 0.0, 0.0})});
    object->geometries().create(body2);

    // ---- the frontend wrote the meta we are about to check ------------------
    {
        auto& c = scope.counter;
        // the region integrals vs the frontend's own per-tet sum
        auto m_attr  = body0.meta().find<Float>(builtin::abd_mass);
        auto mx_attr = body0.meta().find<Vector3>(builtin::abd_mass_x_bar);
        auto mxx_attr = body0.meta().find<Matrix3x3>(builtin::abd_mass_x_bar_x_bar);
        auto vol_attr = body0.meta().find<Float>(builtin::volume);
        REQUIRE((m_attr && mx_attr && mxx_attr && vol_attr));

        T61_CHECK(c, rel_err(vol_attr->view().front(), o.volume) <= 1e-14);
        T61_CHECK(c, rel_err(vol_attr->view().front(), 0.125) <= 1e-14);
        T61_CHECK(c, rel_err(m_attr->view().front(), o.m) <= 1e-14);
        T61_CHECK(c, rel_err(m_attr->view().front(), 125.0) <= 1e-13);
        for(int a = 0; a < 3; ++a)
            T61_CHECK(c, rel_err(mx_attr->view().front()(a), o.mx(a)) <= 1e-14);
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
                T61_CHECK(c,
                          rel_err(mxx_attr->view().front()(a, b), o.mxx(a, b)) <= 1e-14);

        // the rotated + scaled box: m scales as s^3, mx as s^4 with the rotation,
        // mxx as s^5 with R Mxx R^T (all predicted by the region integral)
        auto m1   = body1.meta().find<Float>(builtin::abd_mass)->view().front();
        auto mx1  = body1.meta().find<Vector3>(builtin::abd_mass_x_bar)->view().front();
        auto mxx1 = body1.meta().find<Matrix3x3>(builtin::abd_mass_x_bar_x_bar)->view().front();
        const auto s3 = s * s * s;
        T61_CHECK(c, rel_err(m1, o.m * s3) <= 1e-13);
        // mx1 = R * (rho * s^4 * L^5/2) (1,1,1) / 3?  -- direct: int rho (s R x) dV
        // over the scaled box = s^4 R (int rho x dV) = s^4 R (m L/2) (1,1,1)
        for(int a = 0; a < 3; ++a)
            T61_CHECK(c, rel_err(mx1(a), s3 * s * (R * o.mx)(a)) <= 1e-13);
        const Matrix3x3 expected_mxx1 = s3 * s * s * R * o.mxx * R.transpose();
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
                T61_CHECK(c, rel_err(mxx1(a, b), expected_mxx1(a, b)) <= 1e-13);
    }

    Engine engine{"luisa", workspace_path("uipc_luisa_abd_base_single_cube")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    REQUIRE(T61ABDProbe::abd != nullptr);
    auto& abd = *T61ABDProbe::abd;

    // the mock did the group's work (or reports what it was handed)
    REQUIRE(t61::rec().init_count == 1);
    {
        auto& c = scope.counter;
        T61_CHECK(c, t61::rec().build_count == 1);
        T61_CHECK(c, t61::rec().geo_count == 3);
        T61_CHECK(c, t61::rec().body_count == 3);
        // 8 vertices per box x (1 + 1 + 1) bodies
        T61_CHECK(c, t61::rec().vertex_count == 24);
    }

    // ---- the dynamics' device buffers ---------------------------------------
    auto qs   = read_vec12(abd.qs());
    auto q_vs = read_vec12(abd.q_vs());
    auto dims = read_index(abd.body_dim());
    REQUIRE(qs.size() == 3);
    REQUIRE(q_vs.size() == 3);
    REQUIRE(dims.size() == 3);

    {
        auto& c = scope.counter;

        // q of body 0: identity 3x3 + the translation
        const Vector12 q0 = make_q(Matrix3x3::Identity(), t0);
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(qs[0](i), q0(i)) <= 1e-14);
        // q of body 1: A1 (column-major in the transform!) + the translation
        const Vector12 q1 = make_q(A1, t1);
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(qs[1](i), q1(i)) <= 1e-14);
        // q_v of body 0/1: from the instance velocity attribute
        const Vector12 qv0 = make_q(V0, v0);
        const Vector12 qv1 = make_q(V1, v1);
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(q_vs[0](i), qv0(i)) <= 1e-14);
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(q_vs[1](i), qv1(i)) <= 1e-14);
        // q_v of body 2 was never set: zero
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, q_vs[2](i) == 0.0);

        // dim 2 vs 3
        T61_CHECK(c, dims[0] == 3);
        T61_CHECK(c, dims[1] == 3);
        T61_CHECK(c, dims[2] == 2);
    }

    // ---- volumes ------------------------------------------------------------
    {
        auto&      c      = scope.counter;
        auto       volumes = read_device(abd.body_volumes());
        REQUIRE(volumes.size() == 3);
        T61_CHECK(c, rel_err(volumes[0], o.volume) <= 1e-14);
        T61_CHECK(c, rel_err(volumes[1], o_rs.volume) <= 1e-13);
        T61_CHECK(c, rel_err(volumes[2], o.volume) <= 1e-14);
    }

    // ---- dyadic masses and the 12x12 mass matrix ----------------------------
    {
        auto& c     = scope.counter;
        auto  masses = read_device(abd.body_masses());
        REQUIRE(masses.size() == 3);
        T61_CHECK(c, rel_err(masses[0].m_mass, o.m) <= 1e-14);
        for(int a = 0; a < 3; ++a)
            T61_CHECK(c, rel_err(masses[0].m_mass_times_x_bar(a), o.mx(a)) <= 1e-14);
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
                T61_CHECK(c,
                          rel_err(masses[0].m_mass_times_dyadic_x_bar(a, b), o.mxx(a, b))
                              <= 1e-14);
    }

    // M^-1 from the device must satisfy M^-1 M = I (the *hand-built* M)
    {
        auto&         c    = scope.counter;
        auto          Minv = read_mat12(abd.body_mass_invs());
        REQUIRE(Minv.size() == 3);
        const auto M = mass_matrix_oracle(o.m, o.mx, o.mxx);
        const auto I = Minv[0] * M;
        for(int a = 0; a < 12; ++a)
            for(int b = 0; b < 12; ++b)
                T61_CHECK(c,
                          std::abs(I(a, b) - (a == b ? 1.0 : 0.0))
                              <= 1e-11 * std::max(1.0, std::abs(M(a, b))));
        // the ported routine agrees with the definition-derived matrix
        const auto M_port = gg::abd_dyadic_mass_to_mat_eigen(
            gg::to_abd_dyadic_mass(o.m, o.mx, o.mxx));
        for(int a = 0; a < 12; ++a)
            for(int b = 0; b < 12; ++b)
                T61_CHECK(c, rel_err(M_port(a, b), M(a, b)) <= 1e-14);
    }

    // ---- the 12-dof gravity --------------------------------------------------
    {
        auto& c = scope.counter;
        // G = [ m g ; g_x mx ; g_y mx ; g_z mx ] with g = (0, -9.8, 0) (the scene
        // default), and the stored gravity is M^-1 G
        const Vector3 g{0.0, -9.8, 0.0};
        Vector12      G = Vector12::Zero();
        G.segment<3>(0) = o.m * g;
        G.segment<3>(3) = g(0) * o.mx;
        G.segment<3>(6) = g(1) * o.mx;
        G.segment<3>(9) = g(2) * o.mx;
        // NOTE: the type is explicit -- `auto` would bind an Eigen *expression*
        // that references the temporary returned by `inverse()`.
        const Vector12 expected = mass_matrix_oracle(o.m, o.mx, o.mxx).inverse() * G;

        auto grav = read_vec12(abd.body_gravities());
        REQUIRE(grav.size() == 3);
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(grav[0](i), expected(i)) <= 1e-12);
        // a rigid body's gravity is acceleration-like: the rotation block of q_ddot
        // must vanish, i.e. the rotation part of M^-1 G is (numerically) zero and
        // the translation part is exactly g
        for(int i = 3; i < 12; ++i)
            T61_CHECK(c, std::abs(grav[0](i)) <= 1e-9);
        for(int i = 0; i < 3; ++i)
            T61_CHECK(c, rel_err(grav[0](i), g(i)) <= 1e-12);
    }

    std::printf("[t61] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  2) ABDJacobi / ABDJacobiDyadicMass vs hand-derived + the flag paths
// ===========================================================================
TEST_CASE("abd_base_jacobi", "[luisa][abd_base]")
{
    test_runtime();
    t61::reset();
    CaseScope scope{"abd_base_jacobi"};

    const Float L     = 0.4;
    const Float rho   = 500.0;
    const Float kappa = 1.0e7;
    const auto  o     = box_moments(L, rho);

    // ---- the two element types, checked field by field against an
    //      Eigen-layout struct (the buffer ABI argument of type_define.h) -------
    {
        auto& c = scope.counter;

        struct MirrorJacobi
        {
            Vector3 x_bar;
        };
        struct MirrorDyadicMass
        {
            double    m_mass;
            Vector3   m_mass_times_x_bar;
            Matrix3x3 m_mass_times_dyadic_x_bar;
        };

        T61_CHECK(c, sizeof(gg::ABDJacobi) == sizeof(MirrorJacobi));
        T61_CHECK(c, alignof(gg::ABDJacobi) <= alignof(MirrorJacobi));
        T61_CHECK(c, sizeof(gg::ABDJacobiDyadicMass) == sizeof(MirrorDyadicMass));
        T61_CHECK(c, offsetof(gg::ABDJacobiDyadicMass, m_mass) == offsetof(MirrorDyadicMass, m_mass));
        T61_CHECK(c,
                  offsetof(gg::ABDJacobiDyadicMass, m_mass_times_x_bar)
                      == offsetof(MirrorDyadicMass, m_mass_times_x_bar));
        T61_CHECK(c,
                  offsetof(gg::ABDJacobiDyadicMass, m_mass_times_dyadic_x_bar)
                      == offsetof(MirrorDyadicMass, m_mass_times_dyadic_x_bar));
        T61_CHECK(c, sizeof(gg::ABDJacobiDyadicMass) == 104);
        // the 12-dof aggregates are byte-sized like Eigen's
        T61_CHECK(c, sizeof(gg::LVector12) == sizeof(Vector12));
        T61_CHECK(c, sizeof(gg::LMatrix12x12) == sizeof(Matrix12x12));
        T61_CHECK(c, sizeof(gg::LVector12) == 96);
        T61_CHECK(c, sizeof(gg::LMatrix12x12) == 1152);
    }

    // ---- the jacobi routines (host, S = double) ----------------------------
    const Vector3   x_bar{0.3, -0.7, 1.1};
    const Matrix3x3 A = Eigen::AngleAxisd(0.4, Vector3{0.0, 1.0, 0.0}.normalized())
                        * Vector3{1.2, 0.8, 1.0}.asDiagonal();
    const Vector3 t{0.5, 0.25, -0.75};
    const Vector12 q = make_q(A, t);

    const gg::ABDJacobi j{gg::to_luisa(x_bar)};
    {
        auto& c = scope.counter;

        // p = t + A^T x
        const auto p = gg::abd_jacobi_point_x_eigen(j, q);
        const auto p_oracle = point_oracle(x_bar, q);
        for(int i = 0; i < 3; ++i)
            T61_CHECK(c, rel_err(p(i), p_oracle(i)) <= 1e-14);

        // vec_x: no translation (the same `A x` row form)
        const auto vx = to_eigen3(gg::abd_jacobi_vec_x<double>(gg::to_generic(j), lv12(q)));
        const auto vx_oracle = A * x_bar;
        for(int i = 0; i < 3; ++i)
            T61_CHECK(c, rel_err(vx(i), vx_oracle(i)) <= 1e-14);

        // to_mat: J(a,a) = 1, J(i, 3+3i+j) = x_j
        const auto J = gg::abd_jacobi_to_mat_eigen(j);
        const auto J_oracle = jacobi_oracle(x_bar);
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 12; ++b)
                T61_CHECK(c, std::abs(J(a, b) - J_oracle(a, b)) <= 1e-15);
        // J * q == point_x (the two spellings of the same map)
        const auto Jq = J * q;
        for(int i = 0; i < 3; ++i)
            T61_CHECK(c, rel_err(Jq(i), p_oracle(i)) <= 1e-14);

        // J^T g (against the transposed oracle matrix)
        const Vector3 g{0.9, -1.3, 0.2};
        const auto JT_g = to_eigen12(gg::abd_jacobi_T_mul<double>(gg::to_generic(j), lv(g)));
        const auto JT_g_oracle = J_oracle.transpose() * g;
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(JT_g(i), JT_g_oracle(i)) <= 1e-14);

        // J^T H J
        const Matrix3x3 H = (Matrix3x3() << 2.1, 0.3, -0.2,
                             0.3, 1.7, 0.4,
                             -0.2, 0.4, 2.9).finished();
        const Vector3   y{ -0.25, 0.4, 0.6 };
        const auto       J2 = jacobi_oracle(y);
        const auto       JT_H_J = to_eigen12(gg::abd_jacobi_JT_H_J<double>(
            gg::to_generic(j), lm3(H), gg::to_generic(gg::ABDJacobi{gg::to_luisa(y)})));
        const Matrix12x12 JT_H_J_oracle = J_oracle.transpose() * H * J2;
        for(int a = 0; a < 12; ++a)
            for(int b = 0; b < 12; ++b)
                T61_CHECK(c, rel_err(JT_H_J(a, b), JT_H_J_oracle(a, b)) <= 1e-13);
    }

    // ---- the dyadic mass routines (host) -----------------------------------
    {
        auto& c    = scope.counter;
        auto  M    = gg::make_abd_dyadic_mass(o.m, o.mx, o.mxx);

        // mass / center of mass / inertia about the center of mass
        T61_CHECK(c, rel_err(gg::abd_dyadic_mass_mass(M), o.m) <= 1e-15);
        const auto com = gg::abd_dyadic_mass_center_of_mass_eigen(M);
        for(int i = 0; i < 3; ++i)
            T61_CHECK(c, rel_err(com(i), o.mx(i) / o.m) <= 1e-14);
        // a box's inertia about its own centre of mass is  m L^2/6 I
        const auto I_cm = gg::abd_dyadic_mass_inertia_tensor_cm_eigen(M);
        const double expected_I = o.m * L * L / 6.0;
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b)
                T61_CHECK(c,
                          rel_err(I_cm(a, b), (a == b ? expected_I : 0.0)) <= 1e-13);

        // to_mat == the definition-derived matrix, and operator* == to_mat * p
        const auto M_mat = gg::abd_dyadic_mass_to_mat_eigen(M);
        const auto M_oracle = mass_matrix_oracle(o.m, o.mx, o.mxx);
        for(int a = 0; a < 12; ++a)
            for(int b = 0; b < 12; ++b)
                T61_CHECK(c, rel_err(M_mat(a, b), M_oracle(a, b)) <= 1e-14);

        const Vector12 p = q;  // any 12-vector
        const auto     Mp = to_eigen12(gg::abd_dyadic_mass_mul<double>(gg::to_generic(M), lv12(p)));
        const auto     Mp_oracle = M_oracle * p;
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(Mp(i), Mp_oracle(i)) <= 1e-13);
        // the kinetic term uses the same operator: T = 1/2 p^T (M p)
        T61_CHECK(c,
                  rel_err(0.5 * p.dot(Mp), kinetic_energy_oracle(p, M_oracle)) <= 1e-13);

        // M is symmetric and positive definite (it is a Gram matrix)
        T61_CHECK(c, (M_oracle - M_oracle.transpose()).norm() <= 1e-14);
        Eigen::SelfAdjointEigenSolver<Matrix12x12> es(M_oracle);
        T61_CHECK(c, es.eigenvalues().minCoeff() > 0.0);

        // acc = M + 2*(m, mx, mxx)  (the reference's `operator+=` is the free
        // `abd_dyadic_mass_add`; the element types are plain aggregates, and the
        // *generic* view is what the factored routines take)
        {
            auto acc = gg::to_generic(M);
            gg::abd_dyadic_mass_add(acc,
                                    gg::to_generic(gg::make_abd_dyadic_mass(
                                        2.0 * o.m, 2.0 * o.mx, 2.0 * o.mxx)));
            T61_CHECK(c, rel_err(acc.m_mass, 3.0 * o.m) <= 1e-14);
            const auto acc_mat = to_eigen12(gg::abd_dyadic_mass_to_mat<double>(acc));
            for(int a = 0; a < 12; ++a)
                for(int b = 0; b < 12; ++b)
                    T61_CHECK(c, rel_err(acc_mat(a, b), 3.0 * M_oracle(a, b)) <= 1e-13);
        }

        // the massless guard of `inertia_tensor_cm` selects zero (the reference's
        // early return; RULE M2 forces a select)
        {
            auto zero_mass = gg::make_abd_dyadic_mass(0.0, Vector3::Zero(), Matrix3x3::Zero());
            const auto I0  = gg::abd_dyadic_mass_inertia_tensor_cm_eigen(zero_mass);
            T61_CHECK(c, I0.norm() == 0.0);
        }
    }

    // ---- ABDJacobiStack<N> (dead code in the reference; the port keeps it) --
    {
        auto& c = scope.counter;

        // the stack holds raw pointers to jacobi elements, so it is exercised on
        // the host (it is dead code in the reference; the port keeps it for parity)
        const Vector3         y0{1.0, 0.0, 0.0};
        const Vector3         y1{0.0, 2.0, 0.0};
        const gg::ABDJacobi   j0{gg::to_luisa(y0)};
        const gg::ABDJacobi   j1{gg::to_luisa(y1)};
        gg::ABDJacobiStack2   st{j0, j1};

        // the stacked map: [J0 q ; J1 q]
        const auto p6 = st * lv12(q);
        const auto p0 = point_oracle(y0, q);
        const auto p1 = point_oracle(y1, q);
        for(int i = 0; i < 3; ++i)
        {
            T61_CHECK(c, rel_err(p6(i), p0(i)) <= 1e-14);
            T61_CHECK(c, rel_err(p6(3 + i), p1(i)) <= 1e-14);
        }
        // to_mat stacks the two 3x12 jacobis
        const auto M6 = st.to_mat<double>();
        const auto J0 = jacobi_oracle(y0);
        const auto J1 = jacobi_oracle(y1);
        for(int a = 0; a < 3; ++a)
        {
            for(int b = 0; b < 12; ++b)
            {
                T61_CHECK(c, rel_err(M6(a, b), J0(a, b)) <= 1e-14);
                T61_CHECK(c, rel_err(M6(3 + a, b), J1(a, b)) <= 1e-14);
            }
        }
        // the transposed map J^T g (the `.T()` adapter)
        const lps::LVec<double, 6> g6{0.1, -0.2, 0.3, 0.4, 0.5, -0.6};
        const auto                 JTg = st.T() * g6;
        const Vector3              g0{g6(0), g6(1), g6(2)};
        const Vector3              g1{g6(3), g6(4), g6(5)};
        const auto expect = J0.transpose() * g0 + J1.transpose() * g1;
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(JTg(i), expect(i)) <= 1e-13);
    }

    // ---- a scene: the per-vertex jacobi/body maps and the flag paths --------
    {
        Json  config = abd_scene_config();
        Scene scene{config};
        auto  object = scene.objects().create("abd_base_jacobi");

        // one box with 3 instances: dynamic / fixed / external-kinetic
        auto body0 = make_box_body(L, rho, kappa, 3,
                                   {make_transform(Matrix3x3::Identity(), Vector3{0.0, 0.0, 0.0}),
                                    make_transform(Matrix3x3::Identity(), Vector3{1.0, 0.0, 0.0}),
                                    make_transform(Matrix3x3::Identity(), Vector3{0.0, 1.0, 0.0})});
        {
            view(*body0.instances().find<IndexT>(builtin::is_fixed))[1] = 1;  // fixed
            view(*body0.instances().find<IndexT>(builtin::is_dynamic))[1] = 0;
            view(*body0.instances().find<IndexT>(builtin::external_kinetic))[2] = 1;
        }
        object->geometries().create(body0);

        Engine engine{"luisa", workspace_path("uipc_luisa_abd_base_jacobi")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T61ABDProbe::abd != nullptr);
        REQUIRE(T61ABDProbe::vertex_reporter != nullptr);
        REQUIRE(T61ABDProbe::body_reporter != nullptr);

        auto& abd = *T61ABDProbe::abd;
        auto& c   = scope.counter;

        T61_CHECK(c, t61::rec().body_count == 3);
        T61_CHECK(c, t61::rec().vertex_count == 24);

        // vertex_id_to_J: one rest position per vertex, per instance
        const auto vs = cube_vertices(L);
        auto       Js = read_device(abd.Js());
        REQUIRE(Js.size() == 24);
        for(SizeT inst = 0; inst < 3; ++inst)
        {
            for(SizeT v = 0; v < 8; ++v)
            {
                const auto x = gg::to_eigen(Js[inst * 8 + v]);
                for(int a = 0; a < 3; ++a)
                    T61_CHECK(c, rel_err(x(a), vs[v](a)) <= 1e-15);
            }
        }

        // v2b: the vertex -> body map (8 vertices per body)
        auto v2b = read_index(abd.v2b());
        REQUIRE(v2b.size() == 24);
        for(SizeT inst = 0; inst < 3; ++inst)
            for(SizeT v = 0; v < 8; ++v)
                T61_CHECK(c, v2b[inst * 8 + v] == static_cast<IndexT>(inst));

        // the per-body dyadic mass is the *geometry's* (all three instances share
        // it) and dim/volume follow the geometry
        auto masses = read_device(abd.body_masses());
        REQUIRE(masses.size() == 3);
        for(SizeT i = 0; i < 3; ++i)
        {
            T61_CHECK(c, rel_err(masses[i].m_mass, o.m) <= 1e-14);
            for(int a = 0; a < 3; ++a)
                T61_CHECK(c, rel_err(masses[i].m_mass_times_x_bar(a), o.mx(a)) <= 1e-14);
        }

        // the flag paths (from the instance attributes)
        auto is_fixed    = read_index(abd.body_is_fixed());
        auto is_dynamic  = read_index(abd.body_is_dynamic());
        auto ext_kinetic = read_index(abd.body_external_kinetic());
        REQUIRE(is_fixed.size() == 3);
        REQUIRE(is_dynamic.size() == 3);
        REQUIRE(ext_kinetic.size() == 3);
        T61_CHECK(c, (is_fixed[0] == 0 && is_fixed[1] == 1 && is_fixed[2] == 0));
        T61_CHECK(c, (is_dynamic[0] == 1 && is_dynamic[1] == 0 && is_dynamic[2] == 1));
        T61_CHECK(c, (ext_kinetic[0] == 0 && ext_kinetic[1] == 0 && ext_kinetic[2] == 1));

        // the reporter's own slice: 24 vertices / 3 bodies.  The *shared binary*
        // has other test files' reporters registered too (they also report this
        // scene's geometry as a body), so only the slice is asserted -- never an
        // absolute offset (cookbook §7.14 rule 6).
        T61_CHECK(c, T61ABDProbe::vertex_reporter->vertex_count() == 24);
        T61_CHECK(c, T61ABDProbe::body_reporter->body_count() == 3);
        const auto slice_vertex_offset = T61ABDProbe::vertex_reporter->vertex_offset();
        const auto slice_body_offset   = T61ABDProbe::body_reporter->body_offset();
        T61_CHECK(c, slice_vertex_offset >= 0);
        T61_CHECK(c, slice_body_offset >= 0);

        REQUIRE(T61ABDProbe::vertex_manager != nullptr);
        auto& vertex_manager = *T61ABDProbe::vertex_manager;
        auto  gvm_positions  = read_vec3(vertex_manager.positions());
        auto  gvm_body_ids   = read_index(vertex_manager.body_ids());
        const auto vo        = static_cast<SizeT>(slice_vertex_offset);
        REQUIRE(gvm_positions.size() >= vo + 24u);
        REQUIRE(gvm_body_ids.size() >= vo + 24u);
        for(SizeT inst = 0; inst < 3; ++inst)
        {
            for(SizeT v = 0; v < 8; ++v)
            {
                // the *deformed* position: J * q(inst) = A(inst) x + t(inst); here
                // every instance is a pure translation, so p = x + t
                const Vector3 t_inst{inst == 1 ? 1.0 : 0.0, inst == 2 ? 1.0 : 0.0, 0.0};
                const auto    p   = vs[v] + t_inst;
                const auto    idx = vo + inst * 8 + v;
                for(int a = 0; a < 3; ++a)
                    T61_CHECK(c, rel_err(gvm_positions[idx](a), p(a)) <= 1e-14);
                T61_CHECK(c,
                          gvm_body_ids[idx]
                              == slice_body_offset + static_cast<IndexT>(inst));
            }
        }

        // the body manager sees the same three bodies
        REQUIRE(T61ABDProbe::body_manager != nullptr);
        auto& body_manager = *T61ABDProbe::body_manager;
        {
            auto coindices      = read_index(body_manager.coindices());
            auto self_collision = read_index(body_manager.self_collision());
            const auto bo       = static_cast<SizeT>(slice_body_offset);
            REQUIRE(coindices.size() >= bo + 3u);
            REQUIRE(self_collision.size() >= bo + 3u);
            for(SizeT i = 0; i < 3; ++i)
            {
                T61_CHECK(c, coindices[bo + i] == static_cast<IndexT>(i));
                // the frontend's default is 0 for every ABD body
                T61_CHECK(c, self_collision[bo + i] == 0);
            }
        }
    }

    std::printf("[t61] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  3) shape energy + kinetic terms, finite differences, and the constitution
//     interface end to end (a deformed scene)
// ===========================================================================
TEST_CASE("abd_base_kinetic_and_energy", "[luisa][abd_base]")
{
    test_runtime();
    t61::reset();
    CaseScope scope{"abd_base_kinetic_and_energy"};

    const Float L     = 0.5;
    const Float rho   = 1000.0;
    const Float kappa = 2.0e7;
    const auto  o     = box_moments(L, rho);

    // a deformed affine state (a rotation with a stretch, i.e. A A^T != I)
    const Matrix3x3 R = Eigen::AngleAxisd(0.9, Vector3{0.3, 1.0, -0.4}.normalized())
                            .toRotationMatrix();
    const Matrix3x3 A = R * Vector3{1.25, 0.85, 1.05}.asDiagonal();
    const Vector3   t{0.2, -0.35, 0.15};
    const Vector12  q = make_q(A, t);

    // ---- (c)/(d) energy in operator form vs the ported sum form, plus FD ----
    {
        auto& c = scope.counter;

        // the two forms of the energy are equal *by identity*
        //   sum_i (a_i.a_i - 1)^2 + sum_{i!=j} (a_i.a_j)^2  ==  ||A A^T - I||_F^2
        const double E_oracle = shape_energy_oracle(q, kappa, o.volume);
        const double E_port   = kappa * o.volume * gg::shape_energy(q);
        T61_CHECK(c, rel_err(E_port, E_oracle) <= 1e-15);

        // the energy is translation invariant (q[0..2] does not enter)
        {
            Vector12 q_shifted        = q;
            q_shifted.segment<3>(0) += Vector3{3.0, -2.0, 1.5};
            T61_CHECK(c, gg::shape_energy(q_shifted) == gg::shape_energy(q));
        }

        // central FD on the *oracle*:  gradient and Hessian
        constexpr double h = 1e-6;
        Vector9          grad_fd;
        for(int i = 0; i < 9; ++i)
        {
            Vector12 qp = q, qm = q;
            qp(3 + i) += h;
            qm(3 + i) -= h;
            grad_fd(i) = (shape_energy_oracle(qp, kappa, o.volume)
                          - shape_energy_oracle(qm, kappa, o.volume))
                         / (2.0 * h);
        }
        // explicit types: `kappa * o.volume * <temporary>` is an Eigen expression
        // that would dangle if it were bound with `auto`
        const Vector9 grad_port = kappa * o.volume * gg::shape_energy_gradient(q);
        double     worst_grad = 0.0;
        for(int i = 0; i < 9; ++i)
        {
            const double err = rel_err(grad_port(i), grad_fd(i));
            worst_grad       = std::max(worst_grad, err);
            T61_CHECK(c, err <= 1e-8);
        }
        // the FD check is *non-vacuous*: the gradient is not (numerically) zero
        T61_CHECK(c, grad_fd.norm() > 1e-3);
        // and the translation dofs have no gradient at all: the energy does not
        // depend on q[0..2], so a shift of the body must leave it unchanged
        {
            Vector12 q_shifted = q;
            q_shifted.segment<3>(0) += Vector3{1.0, 2.0, -3.0};
            T61_CHECK(c, gg::shape_energy(q_shifted) == gg::shape_energy(q));
            T61_CHECK(c, gg::shape_energy_gradient(q_shifted) == gg::shape_energy_gradient(q));
        }

        const Matrix9x9 H_port = kappa * o.volume * gg::shape_energy_hessian(q);
        // symmetry (brief §3.3)
        const double asym = (H_port - H_port.transpose()).norm()
                            / std::max(1.0, H_port.norm());
        T61_CHECK(c, asym <= 1e-12);

        // FD of the *oracle* gradient vs the ported Hessian
        double worst_hess = 0.0;
        for(int j = 0; j < 9; ++j)
        {
            Vector12 qp = q, qm = q;
            qp(3 + j) += h;
            qm(3 + j) -= h;
            const auto Gp = kappa * o.volume * gg::shape_energy_gradient(qp);
            const auto Gm = kappa * o.volume * gg::shape_energy_gradient(qm);
            const Vector9 col = (Gp - Gm) / (2.0 * h);
            for(int i = 0; i < 9; ++i)
            {
                const double err = rel_err(H_port(i, j), col(i));
                worst_hess       = std::max(worst_hess, err);
                T61_CHECK(c, err <= 1e-6);
            }
        }
        // the Hessian is non-trivial
        T61_CHECK(c, H_port.norm() > 1e-3);

        // the 6-out-parameter form and the assembled 9x9 agree
        Matrix3x3 d1, d2, d3, c12, c13, c23;
        gg::shape_energy_hessian(q, d1, d2, d3, c12, c13, c23);
        Matrix9x9 H6 = Matrix9x9::Zero();
        H6.block<3, 3>(0, 0) = d1;
        H6.block<3, 3>(3, 3) = d2;
        H6.block<3, 3>(6, 6) = d3;
        H6.block<3, 3>(0, 3) = c12;
        H6.block<3, 3>(3, 0) = c12.transpose();
        H6.block<3, 3>(0, 6) = c13;
        H6.block<3, 3>(6, 0) = c13.transpose();
        H6.block<3, 3>(3, 6) = c23;
        H6.block<3, 3>(6, 3) = c23.transpose();
        for(int a = 0; a < 9; ++a)
            for(int b = 0; b < 9; ++b)
                T61_CHECK(c, std::abs(H6(a, b) - gg::shape_energy_hessian(q)(a, b)) <= 1e-15);

        std::printf("[t61]   energy: worst grad rel %.3e, worst Hessian rel %.3e, asymmetry %.3e\n",
                    worst_grad,
                    worst_hess,
                    asym);
    }

    // ---- the kinetic term ---------------------------------------------------
    {
        auto& c = scope.counter;

        const Matrix3x3 V = 0.2 * Vector3{1.0, -2.0, 0.5}.asDiagonal();
        const Vector3   v{0.4, 0.1, -0.3};
        const Vector12  q_v = make_q(V, v);

        const auto M = mass_matrix_oracle(o.m, o.mx, o.mxx);
        // T = 1/2 q_v^T M q_v  (the ported operator == the hand-built matrix)
        const auto Mp = gg::abd_dyadic_mass_mul<double>(
            gg::to_generic(gg::to_abd_dyadic_mass(o.m, o.mx, o.mxx)), lv12(q_v));
        const double T_port = 0.5 * to_eigen12(Mp).dot(q_v);
        T61_CHECK(c, rel_err(T_port, kinetic_energy_oracle(q_v, M)) <= 1e-13);
        // the gradient of T w.r.t. q_v is M q_v, the Hessian is M
        const auto grad_T = M * q_v;
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(to_eigen12(Mp)(i), grad_T(i)) <= 1e-13);
        // FD of T (the oracle form) for the same statement
        constexpr double h2 = 1e-6;
        for(int i = 0; i < 12; ++i)
        {
            Vector12 qp = q_v, qm = q_v;
            qp(i) += h2;
            qm(i) -= h2;
            const double fd = (kinetic_energy_oracle(qp, M) - kinetic_energy_oracle(qm, M))
                              / (2.0 * h2);
            T61_CHECK(c, rel_err(fd, grad_T(i)) <= 1e-8);
        }
        // T is positive for a nonzero velocity and zero at rest
        T61_CHECK(c, T_port > 0.0);
        T61_CHECK(c, kinetic_energy_oracle(Vector12::Zero(), M) == 0.0);
    }

    // ---- the constitution interface, end to end on a deformed scene ---------
    {
        // two bodies with different deformations and velocities.
        // A0 is *slightly stretched*: a pure rotation has A A^T = I, i.e. exactly
        // zero shape energy, which would make the energy checks vacuous.
        const Matrix3x3 A0 = R * Vector3{1.05, 0.95, 1.02}.asDiagonal();
        const Vector3   t0{0.0, 0.0, 0.0};
        const Matrix3x3 A1 = R * Vector3{1.4, 0.7, 1.1}.asDiagonal();
        const Vector3   t1{1.0, 0.0, 0.0};
        const Matrix3x3 V0 = (0.15 * Vector3{1.0, 1.0, -1.0}.asDiagonal());
        const Vector3   vv0{0.2, 0.0, -0.1};
        const Matrix3x3 V1 = (-0.3 * Vector3{0.5, 0.5, 2.0}.asDiagonal());
        const Vector3   vv1{0.0, -0.25, 0.0};

        Json  config = abd_scene_config();
        Scene scene{config};
        auto  object = scene.objects().create("abd_base_energy");

        auto body0 =
            make_box_body(L, rho, kappa, 1, {make_transform(A0, t0)}, {make_velocity(V0, vv0)});
        object->geometries().create(body0);
        auto body1 =
            make_box_body(L, rho, kappa, 1, {make_transform(A1, t1)}, {make_velocity(V1, vv1)});
        object->geometries().create(body1);

        Engine engine{"luisa", workspace_path("uipc_luisa_abd_base_energy")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());

        auto& c = scope.counter;
        // `do_init` walked both bodies through `FilteredInfo::for_each`
        REQUIRE(t61::rec().init_count == 1);
        T61_CHECK(c, t61::rec().for_each_count == 2);
        T61_CHECK(c, (t61::rec().global_indices == std::vector<SizeT>{0, 1}));
        T61_CHECK(c, (t61::rec().geo_body_offsets == std::vector<SizeT>{0, 1}));

        // the recorded q/q_v are the scene's
        REQUIRE(t61::rec().qs.size() == 2);
        const Vector12 q0 = make_q(A0, t0);
        const Vector12 q1 = make_q(A1, t1);
        for(int i = 0; i < 12; ++i)
        {
            T61_CHECK(c, rel_err(t61::rec().qs[0](i), q0(i)) <= 1e-14);
            T61_CHECK(c, rel_err(t61::rec().qs[1](i), q1(i)) <= 1e-14);
        }
        for(int i = 0; i < 12; ++i)
        {
            T61_CHECK(c, rel_err(t61::rec().q_vs[0](i), make_q(V0, vv0)(i)) <= 1e-14);
            T61_CHECK(c, rel_err(t61::rec().q_vs[1](i), make_q(V1, vv1)(i)) <= 1e-14);
        }
        // and kappa/volume come from the geometry
        for(size_t i = 0; i < 2; ++i)
        {
            T61_CHECK(c, rel_err(t61::rec().kappas[i], kappa) <= 1e-15);
            T61_CHECK(c, rel_err(t61::rec().volumes[i], o.volume) <= 1e-14);
        }

        // the shape energies are the oracle's (operator form), computed from the
        // same transforms
        const auto M = mass_matrix_oracle(o.m, o.mx, o.mxx);
        const double E0_oracle = shape_energy_oracle(q0, kappa, o.volume);
        const double E1_oracle = shape_energy_oracle(q1, kappa, o.volume);
        T61_CHECK(c, rel_err(t61::rec().shape_energies[0], E0_oracle) <= 1e-14);
        T61_CHECK(c, rel_err(t61::rec().shape_energies[1], E1_oracle) <= 1e-14);
        // both are non-zero, i.e. the check is not vacuous
        T61_CHECK(c, std::abs(E0_oracle) > 1e-6);
        T61_CHECK(c, std::abs(E1_oracle) > 1e-6);
        T61_CHECK(c, std::abs(E0_oracle - E1_oracle) > 1e-6);

        const double T0_oracle = kinetic_energy_oracle(make_q(V0, vv0), M);
        const double T1_oracle = kinetic_energy_oracle(make_q(V1, vv1), M);
        T61_CHECK(c, rel_err(t61::rec().kinetic_energies[0], T0_oracle) <= 1e-13);
        T61_CHECK(c, rel_err(t61::rec().kinetic_energies[1], T1_oracle) <= 1e-13);
        T61_CHECK(c, T0_oracle > 0.0);
        T61_CHECK(c, T1_oracle > 0.0);

        // the dynamics' own buffers agree with the scene
        REQUIRE(T61ABDProbe::abd != nullptr);
        auto qs = read_vec12(T61ABDProbe::abd->qs());
        REQUIRE(qs.size() == 2);
        for(int i = 0; i < 12; ++i)
        {
            T61_CHECK(c, rel_err(qs[0](i), q0(i)) <= 1e-14);
            T61_CHECK(c, rel_err(qs[1](i), q1(i)) <= 1e-14);
        }
    }

    std::printf("[t61] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  4) the state accessor feature
// ===========================================================================
TEST_CASE("abd_base_state_accessor", "[luisa][abd_base]")
{
    test_runtime();
    t61::reset();
    CaseScope scope{"abd_base_state_accessor"};

    const Float L     = 0.5;
    const Float rho   = 1000.0;
    const Float kappa = 1.0e8;

    const Matrix3x3 A0 = Matrix3x3::Identity();
    const Vector3   t0{0.25, 0.0, -0.5};
    const Matrix3x3 A1 = Eigen::AngleAxisd(0.5, Vector3{0.0, 0.0, 1.0}.normalized())
                             .toRotationMatrix();
    const Vector3 t1{1.0, 2.0, 3.0};

    Json  config = abd_scene_config();
    Scene scene{config};
    auto  object = scene.objects().create("abd_base_state_accessor");
    object->geometries().create(make_box_body(L, rho, kappa, 1, {make_transform(A0, t0)}));
    object->geometries().create(make_box_body(L, rho, kappa, 1, {make_transform(A1, t1)}));

    Engine engine{"luisa", workspace_path("uipc_luisa_abd_base_state_accessor")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    auto& c       = scope.counter;
    auto  feature = world.features().find<AffineBodyStateAccessorFeature>();
    REQUIRE(feature != nullptr);
    T61_CHECK(c, feature->body_count() == 2);

    auto state_geo = feature->create_geometry();
    T61_CHECK(c, state_geo.instances().size() == 2);
    {
        auto offset_attr = state_geo.meta().find<IndexT>(builtin::backend_abd_body_offset);
        REQUIRE(offset_attr != nullptr);
        T61_CHECK(c, offset_attr->view()[0] == 0);
    }
    // the feature creates neither transform nor velocity by default
    T61_CHECK(c, state_geo.instances().find<Matrix4x4>(builtin::transform) == nullptr);
    auto trans_attr = state_geo.instances().create<Matrix4x4>(builtin::transform, Matrix4x4::Identity());
    auto vel_attr   = state_geo.instances().create<Matrix4x4>(builtin::velocity, Matrix4x4::Zero());

    // ---- copy_to: the device q/q_v land in the geometry --------------------
    feature->copy_to(state_geo);
    {
        auto trans = view(*trans_attr);
        auto vel   = view(*vel_attr);
        const std::array<Matrix3x3, 2> As{A0, A1};
        const std::array<Vector3, 2>   ts{t0, t1};
        for(int i = 0; i < 2; ++i)
        {
            const auto expected = make_transform(As[i], ts[i]);
            for(int a = 0; a < 4; ++a)
                for(int b = 0; b < 4; ++b)
                    T61_CHECK(c, std::abs(trans[i](a, b) - expected(a, b)) <= 1e-14);
            // the velocity was never set: a zero q_v maps to the zero transform
            for(int a = 0; a < 4; ++a)
                for(int b = 0; b < 4; ++b)
                    T61_CHECK(c, vel[i](a, b) == 0.0);
        }
    }

    // ---- copy_from: a modified geometry is written back to the device -------
    {
        const Matrix3x3 A0b = Eigen::AngleAxisd(1.1, Vector3{1.0, 1.0, 1.0}.normalized())
                                  .toRotationMatrix()
                              * Vector3{1.05, 0.95, 1.0}.asDiagonal();
        const Vector3 t0b{0.5, -0.75, 0.25};
        const Matrix3x3 V0b = 0.3 * Vector3{1.0, 2.0, 3.0}.asDiagonal();
        const Vector3   v0b{0.1, -0.2, 0.3};
        const Matrix3x3 V1b = 0.1 * Vector3{-1.0, 0.5, 1.5}.asDiagonal();
        const Vector3   v1b{-0.4, 0.2, 0.0};

        auto trans = view(*trans_attr);
        auto vel   = view(*vel_attr);
        trans[0]   = make_transform(A0b, t0b);
        vel[0]     = make_velocity(V0b, v0b);
        vel[1]     = make_velocity(V1b, v1b);

        feature->copy_from(state_geo);

        REQUIRE(T61ABDProbe::abd != nullptr);
        auto qs   = read_vec12(T61ABDProbe::abd->qs());
        auto q_vs = read_vec12(T61ABDProbe::abd->q_vs());
        REQUIRE(qs.size() == 2);
        REQUIRE(q_vs.size() == 2);

        const Vector12 q0b = make_q(A0b, t0b);
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(qs[0](i), q0b(i)) <= 1e-13);
        // body 1's transform was not modified
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(qs[1](i), make_q(A1, t1)(i)) <= 1e-14);
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(q_vs[0](i), make_q(V0b, v0b)(i)) <= 1e-13);
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(q_vs[1](i), make_q(V1b, v1b)(i)) <= 1e-13);

        // copy_from requested a vertex-attribute update: the next frame's reporter
        // pass must re-derive the global positions from the new q
        world.advance();
        REQUIRE(world.is_valid());
        world.sync();

        REQUIRE(T61ABDProbe::vertex_reporter != nullptr);
        REQUIRE(T61ABDProbe::vertex_manager != nullptr);
        auto& vertex_manager = *T61ABDProbe::vertex_manager;
        auto  gvm_positions  = read_vec3(vertex_manager.positions());
        const auto offset    = T61ABDProbe::vertex_reporter->vertex_offset();
        REQUIRE(gvm_positions.size() >= 16);
        const auto vs = cube_vertices(L);
        for(SizeT v = 0; v < 8; ++v)
        {
            const Vector3 expected = A0b * vs[v] + t0b;
            for(int a = 0; a < 3; ++a)
                T61_CHECK(c,
                          rel_err(gvm_positions[offset + v](a), expected(a)) <= 1e-12);
        }
    }

    // ---- the two externally-owned buffer exports ---------------------------
    // `backend::BufferView::handle()` is a caller-owned *host* address (that is
    // what the pybind frontend passes); see the deviation note in
    // affine_body_state_accessor_feature.cpp.
    {
        REQUIRE(T61ABDProbe::abd != nullptr);
        auto qs    = read_vec12(T61ABDProbe::abd->qs());
        auto q_vs  = read_vec12(T61ABDProbe::abd->q_vs());
        REQUIRE(qs.size() == 2);

        std::vector<Matrix4x4> trans_out(2, Matrix4x4::Zero());
        uipc::backend::BufferView trans_view{
            reinterpret_cast<uipc::backend::HandleT>(trans_out.data()),
            0,
            2u,
            sizeof(Matrix4x4),
            sizeof(Matrix4x4),
            "luisa"};
        feature->copy_transform_to(trans_view, 0, 2u);
        for(int i = 0; i < 2; ++i)
        {
            const auto expected = gg::q_to_transform(qs[i]);
            for(int a = 0; a < 4; ++a)
                for(int b = 0; b < 4; ++b)
                    T61_CHECK(c, std::abs(trans_out[i](a, b) - expected(a, b)) <= 1e-14);
        }

        std::vector<Matrix4x4> vel_out(2, Matrix4x4::Zero());
        uipc::backend::BufferView vel_view{
            reinterpret_cast<uipc::backend::HandleT>(vel_out.data()),
            0,
            2u,
            sizeof(Matrix4x4),
            sizeof(Matrix4x4),
            "luisa"};
        feature->copy_velocity_to(vel_view, 0, 2u);
        for(int i = 0; i < 2; ++i)
        {
            const auto expected = gg::q_v_to_transform_v(q_vs[i]);
            for(int a = 0; a < 4; ++a)
                for(int b = 0; b < 4; ++b)
                    T61_CHECK(c, std::abs(vel_out[i](a, b) - expected(a, b)) <= 1e-14);
        }

        // a *sub-range* export (the offset/count path)
        std::vector<Matrix4x4> half(1, Matrix4x4::Zero());
        uipc::backend::BufferView half_view{
            reinterpret_cast<uipc::backend::HandleT>(half.data()),
            0,
            1u,
            sizeof(Matrix4x4),
            sizeof(Matrix4x4),
            "luisa"};
        feature->copy_transform_to(half_view, 1, 1u);
        const auto expected1 = gg::q_to_transform(qs[1]);
        for(int a = 0; a < 4; ++a)
            for(int b = 0; b < 4; ++b)
                T61_CHECK(c, std::abs(half[0](a, b) - expected1(a, b)) <= 1e-14);
    }

    std::printf("[t61] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  5) a real scene: one ABD body, the reporters, one advance() under BDF1
// ===========================================================================
TEST_CASE("abd_base_scene", "[luisa][abd_base]")
{
    test_runtime();
    t61::reset();
    CaseScope scope{"abd_base_scene"};

    const Float L     = 0.6;
    const Float rho   = 800.0;
    const Float kappa = 5.0e7;
    const auto  o     = box_moments(L, rho);

    const Matrix3x3 A = Eigen::AngleAxisd(0.35, Vector3{0.0, 1.0, 0.0}.normalized())
                            .toRotationMatrix();
    const Vector3 t{0.0, 0.4, 0.0};

    Json config = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = false;
    config["integrator"]["type"]     = std::string{"bdf1"};
    Scene scene{config};

    auto object = scene.objects().create("abd_base_scene");
    auto body   = make_box_body(L, rho, kappa, 1, {make_transform(A, t)});
    object->geometries().create(body);

    Engine engine{"luisa", workspace_path("uipc_luisa_abd_base_scene")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    auto& c = scope.counter;

    // the BDF1 flag exists for this scene (that is what "a BDF1 scene" means; the
    // ABD BDF1 integrator itself is T6.2's)
    REQUIRE(T61ABDProbe::bdf1_flag != nullptr);

    // The animator slot is T6.1b's: `AffineBodyAnimator` has no body (and no
    // SimSystemCreator) in T6.1a, and this scene has no Constraint constitution, so
    // neither `GlobalAnimator` nor `AffineBodyAnimator` exists.  This is the
    // *not-implemented* signal for the batch that owns them, not an expectation of
    // the final behaviour.
        T61_CHECK(c, T61ABDProbe::global_animator == nullptr);
        // T6.1b landed the ABD animator: it is *registered* now, and its `find<>`
        // result is the registered-but-invalidated object (T5.3's finding), so the
        // "is it in play?" question is asked with `is_valid()` -- in this scene
        // there is no Constraint constitution, hence no `GlobalAnimator`, so the
        // animator is invalidated by its own `require<>`.
        T61_CHECK(c, (T61ABDProbe::animator == nullptr
                      || T61ABDProbe::animator->is_valid() == false));

    REQUIRE(T61ABDProbe::abd != nullptr);
    auto& abd = *T61ABDProbe::abd;

    // exactly one ABD body of 8 vertices
    T61_CHECK(c, t61::rec().body_count == 1);
    T61_CHECK(c, t61::rec().vertex_count == 8);
    REQUIRE(T61ABDProbe::vertex_reporter != nullptr);
    REQUIRE(T61ABDProbe::body_reporter != nullptr);
    T61_CHECK(c, T61ABDProbe::vertex_reporter->vertex_count() == 8);
    T61_CHECK(c, T61ABDProbe::body_reporter->body_count() == 1);

    // The ABD reporters own exactly this scene's 8 vertices / 1 body, but the
    // *shared binary* has other test files' reporters registered too (they report
    // this scene's geometry as well), so the global arrays are longer than this
    // scene's contribution and every assertion below is made on this reporter's
    // own slice (cookbook §7.14 rule 6).  "Nothing else is disturbed" is therefore
    // checked as: the ABD slice holds exactly the model's values, the scene's one
    // geometry is untouched, and the ABD state is unchanged across advance().
    REQUIRE(T61ABDProbe::vertex_manager != nullptr);
    REQUIRE(T61ABDProbe::body_manager != nullptr);
    auto& vertex_manager = *T61ABDProbe::vertex_manager;
    auto& body_manager   = *T61ABDProbe::body_manager;

    // The shared binary has other test files' reporters registered, so the global
    // arrays are *longer* than this scene's contribution: every assertion below is
    // made on this reporter's own slice (cookbook §7.14 rule 6).
    const auto vertex_offset = T61ABDProbe::vertex_reporter->vertex_offset();
    const auto body_offset   = T61ABDProbe::body_reporter->body_offset();
    T61_CHECK(c, vertex_offset >= 0);
    T61_CHECK(c, body_offset >= 0);

    {
        auto gvm_positions = read_vec3(vertex_manager.positions());
        auto gvm_body_ids  = read_index(vertex_manager.body_ids());
        auto gvm_rest      = read_vec3(vertex_manager.rest_positions());
        const auto v0      = static_cast<SizeT>(vertex_offset);
        REQUIRE(gvm_positions.size() >= v0 + 8u);
        REQUIRE(gvm_body_ids.size() >= v0 + 8u);
        REQUIRE(gvm_rest.size() >= v0 + 8u);
        // this scene's one body is at our slice of the body manager's arrays
        auto gbm_coindices = read_index(body_manager.coindices());
        REQUIRE(gbm_coindices.size() >= static_cast<SizeT>(body_offset) + 1u);
        T61_CHECK(c, gbm_coindices[static_cast<SizeT>(body_offset)] == 0);

        const auto vs = cube_vertices(L);
        auto       qs = read_vec12(abd.qs());
        REQUIRE(qs.size() == 1);
        // q is the scene's transform
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(qs[0](i), make_q(A, t)(i)) <= 1e-14);
        for(SizeT v = 0; v < 8; ++v)
        {
            // the reported position is p = A x + t (the vertex reporter's kernel)
            const Vector3 expected = A * vs[v] + t;
            for(int a = 0; a < 3; ++a)
                T61_CHECK(c, rel_err(gvm_positions[v0 + v](a), expected(a)) <= 1e-13);
            // the rest position is the geometry's own vertex
            for(int a = 0; a < 3; ++a)
                T61_CHECK(c, rel_err(gvm_rest[v0 + v](a), vs[v](a)) <= 1e-14);
            // every vertex of our slice belongs to our body 0
            T61_CHECK(c, gvm_body_ids[v0 + v] == body_offset);
        }
    }

    // the surface reporter reported the cube's boundary (6 faces x 2 triangles),
    // modelled as the boundary of the six-tet decomposition itself
    {
        REQUIRE(T61ABDProbe::surface_manager != nullptr);
        auto& surface_manager = *T61ABDProbe::surface_manager;
        auto  surf_vertices   = read_index(surface_manager.surf_vertices());
        auto  surf_triangles  = read_vec3i(surface_manager.surf_triangles());

        const auto [expected_vertices, expected_triangles] =
            boundary_oracle(vertex_offset);
        T61_CHECK(c, expected_vertices.size() == 8);
        T61_CHECK(c, expected_triangles.size() == 12);

        // the reported arrays are global, so first isolate this reporter's slice:
        // its vertices are exactly [vertex_offset, vertex_offset + 8) and its
        // triangles are the ones whose three vertices all lie in that range.
        std::vector<IndexT> ours_vertices;
        for(auto v : surf_vertices)
        {
            if(v >= vertex_offset && v < vertex_offset + 8)
                ours_vertices.push_back(v);
        }
        std::sort(ours_vertices.begin(), ours_vertices.end());
        ours_vertices.erase(std::unique(ours_vertices.begin(), ours_vertices.end()),
                            ours_vertices.end());
        T61_CHECK(c, ours_vertices == expected_vertices);

        std::vector<Vector3i> ours_triangles;
        for(auto& t : surf_triangles)
        {
            bool mine = true;
            for(int k = 0; k < 3; ++k)
                mine = mine && (t(k) >= vertex_offset && t(k) < vertex_offset + 8);
            if(mine)
                ours_triangles.push_back(t);
        }
        T61_CHECK(c, same_triangles(canonical(ours_triangles),
                                    canonical(expected_triangles)));
    }

    // the external-force manager exists and its buffers are zeroed (the
    // clear() half of the T4.2 lifecycle; the constraints that would fill them are
    // T6.4's)
    {
        REQUIRE(T61ABDProbe::external_force_manager != nullptr);
        auto forces = read_vec12(abd.body_external_forces());
        auto accs   = read_vec12(abd.body_external_force_accs());
        REQUIRE(forces.size() == 1);
        REQUIRE(accs.size() == 1);
        for(int i = 0; i < 12; ++i)
        {
            T61_CHECK(c, forces[0](i) == 0.0);
            T61_CHECK(c, accs[0](i) == 0.0);
        }
    }

    // ---- one advance(): the scene stays valid and the ABD state is the scene's
    {
        auto qs_before = read_vec12(abd.qs());
        world.advance();
        REQUIRE(world.is_valid());
        world.sync();
        auto qs_after = read_vec12(abd.qs());
        REQUIRE(qs_after.size() == 1);

        // T6.1a has no ABD time integrator (T6.2): q may only be the one the scene
        // put there.  This is a *bug detector*, not an expectation of the final
        // behaviour -- when T6.2 lands, this assertion moves to the ABD dynamics
        // test file.
        for(int i = 0; i < 12; ++i)
            T61_CHECK(c, rel_err(qs_after[0](i), qs_before[0](i)) <= 1e-15);

        // the reported positions still match the scene (the vertex reporter ran
        // again and re-derived them from q)
        auto       gvm_positions = read_vec3(vertex_manager.positions());
        const auto v0            = static_cast<SizeT>(vertex_offset);
        REQUIRE(gvm_positions.size() >= v0 + 8u);
        const auto vs = cube_vertices(L);
        for(SizeT v = 0; v < 8; ++v)
        {
            const Vector3 expected = A * vs[v] + t;
            for(int a = 0; a < 3; ++a)
                T61_CHECK(c, rel_err(gvm_positions[v0 + v](a), expected(a)) <= 1e-13);
        }
    }

    // the scene's geometries are still there and untouched
    {
        auto visitor   = uipc::backend::WorldVisitor{world};
        auto geo_slots = visitor.scene().geometries();
        REQUIRE(geo_slots.size() == 1);
        auto sc = geo_slots[0]->geometry().as<SimplicialComplex>();
        REQUIRE(sc != nullptr);
        T61_CHECK(c, sc->vertices().size() == 8);
        T61_CHECK(c, sc->tetrahedra().size() == 6);
        T61_CHECK(c, sc->instances().size() == 1);
        T61_CHECK(c, sc->dim() == 3);
        // and the frontend meta is the one measured before the run
        auto vol_attr = sc->meta().find<Float>(builtin::volume);
        auto m_attr   = sc->meta().find<Float>(builtin::abd_mass);
        REQUIRE((vol_attr && m_attr));
        T61_CHECK(c, rel_err(vol_attr->view().front(), o.volume) <= 1e-14);
        T61_CHECK(c, rel_err(m_attr->view().front(), o.m) <= 1e-14);
    }

    std::printf("[t61] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  6) device == host for the whole ported ABD routine surface
// ===========================================================================
TEST_CASE("abd_base_device_vs_host", "[luisa][abd_base]")
{
    test_runtime();
    CaseScope scope{"abd_base_device_vs_host"};

    std::vector<double> host;
    abd_round<double>(
        [&](int i, double value)
        {
            if(static_cast<int>(host.size()) <= i)
                host.resize(i + 1, 0.0);
            host[i] = value;
        });
    REQUIRE(host.size() == static_cast<size_t>(kABDDumpCount));

    // ---- the host side's semantic checks (the same numbers the device dumps) --
    {
        auto& c = scope.counter;

        // the hand-derived dump inputs: the mass triples are the box moments, and
        // the dump of `abd_dyadic_mass_to_mat` is the definition-derived matrix
        const auto o  = box_moments(0.5, 1000.0);
        const auto M  = mass_matrix_oracle(o.m, o.mx, o.mxx);
        const auto M2 = mass_matrix_oracle(box_moments(0.2, 250.0).m,
                                           box_moments(0.2, 250.0).mx,
                                           box_moments(0.2, 250.0).mxx);
        // the dump's mass triple is the box's (L = 0.5, rho = 1000 -> m = 125)
        T61_CHECK(c,
                  rel_err(gg::abd_dyadic_mass_mass(
                              gg::make_abd_dyadic_mass(o.m, o.mx, o.mxx)),
                          o.m)
                      <= 1e-14);

        // `torque_to_F`'s near-singular guard is a *select* in the port (RULE M2):
        // a collapsed affine matrix must produce exactly zero
        {
            lps::LVec<double, 12> q_singular;
            for(int i = 0; i < 12; ++i)
                q_singular(i) = 0.0;
            q_singular(0) = 0.1;
            const auto F = gg::torque_to_F<double>(
                3.0, lv(Vector3{0.0, 0.0, 1.0}), q_singular);
            T61_CHECK(c, lps::squaredNorm(F) == 0.0);
            // and a healthy matrix must not produce zero
            const auto F_ok =
                gg::torque_to_F<double>(3.0, lv(Vector3{0.0, 0.0, 1.0}), fixed_q<double>());
            T61_CHECK(c, lps::squaredNorm(F_ok) > 0.0);
        }

        // kappa*volume*shape_energy must equal the operator form for the dump input
        const auto   q_dump = fixed_q<double>();
        const double E_sum  = gg::shape_energy<double>(q_dump);
        const double E_op   = (A_oracle(to_eigen12(q_dump))
                               * A_oracle(to_eigen12(q_dump)).transpose()
                               - Matrix3x3::Identity())
                                  .squaredNorm();
        T61_CHECK(c, rel_err(E_sum, E_op) <= 1e-14);

        // the inversion of the box mass matrix, checked against M^-1 M = I
        // explicit type: `auto` here would bind Eigen's `Inverse<>` *expression*,
        // which references the temporary matrix it was built from (dangling).
        const Matrix12x12 Minv =
            gg::abd_dyadic_mass_to_mat_eigen(gg::make_abd_dyadic_mass(o.m, o.mx, o.mxx))
                .inverse();
        const Matrix12x12 I = Minv * M;
        for(int a = 0; a < 12; ++a)
            for(int b = 0; b < 12; ++b)
                T61_CHECK(c, std::abs(I(a, b) - (a == b ? 1.0 : 0.0)) <= 1e-11);
        (void)M2;
    }

    // ---- the device run -----------------------------------------------------
    lud::LDeviceBuffer<double> device(kABDDumpCount);
    gg::frame_stream().record(abd_dump_kernel()(device.view()).dispatch(1u));
    auto device_host = read_device(device.cview());
    REQUIRE(device_host.size() == host.size());

    int    bitwise_equal = 0;
    double worst_abs     = 0.0;
    int    worst_idx     = -1;
    for(int i = 0; i < kABDDumpCount; ++i)
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
    std::printf("[t61] device-vs-host: %d/%d bitwise equal, worst abs %.3e (entry %d)\n",
                bitwise_equal,
                kABDDumpCount,
                worst_abs,
                worst_idx);

    // The arithmetic is the same code on both sides (one templated routine), so
    // the requirement is bitwise (the T2.1/T5.1 convention); if a future backend
    // contraction (FMA) makes it too strict, the bound printed above is the
    // fallback.
    {
        auto& c = scope.counter;
        T61_CHECK(c, bitwise_equal == kABDDumpCount);
        T61_CHECK(c, worst_abs <= 1e-12);
    }

    std::printf("[t61] %s: %d checks\n", scope.name.c_str(), scope.counter.checks);
}

// ===========================================================================
//  the per-case assertion summary (brief §6.3)
// ===========================================================================
TEST_CASE("abd_base_summary", "[luisa][abd_base]")
{
    std::printf("[t61] ---------- per-case assertion counts ----------\n");
    int total = 0;
    for(auto& [name, n] : g_case_counts)
    {
        std::printf("[t61] %-32s %6d\n", name.c_str(), n);
        total += n;
    }
    std::printf("[t61] %-32s %6d\n", "TOTAL", total);
    CHECK(!g_case_counts.empty());
    CHECK(total > 0);
}
