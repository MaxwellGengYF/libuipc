/********************************************************************
 * @file luisa_fem_constitution_a_test.cpp
 * @brief T5.2 (batch a) — the 3D volumetric FEM constitutions (ARAP + stable
 *        Neo-Hookean) of the LuisaCompute backend.
 *
 * Every oracle in this file is derived independently *here*: the reference backend
 * is CUDA-only and cannot be host-included, so nothing is copied from it.  The
 * oracles are
 *   (a) hand-derived closed forms for a single tetrahedron (energy, gradient and,
 *       where it is short, the Hessian) — the derivations are in the comments,
 *   (b) central finite differences (h = 1e-6) *with respect to the 12 dofs of the
 *       element* — i.e. through `fem::F`/`dFdx`/`project_F_gradient`/
 *       `project_F_hessian_block`, not on a hand-fed F — on randomized admissible
 *       states,
 *   (c) structural invariants (symmetry, positive semidefiniteness, the analytic
 *       eigensystem behind the SNK1 projection),
 *   (d) a *host* unit test of the ported `DoubletVectorAssembler` /
 *       `TripletMatrixAssembler` index semantics (the upper-triangular swap),
 *   (e) the device-vs-host equivalence of the ported per-element routines
 *       (S = double vs S = lcrt::Var<double> inside a Kernel1D), and
 *   (f) a real scene through the public Engine/World API, where the *real*
 *       constitution's `do_init`/`do_compute_energy`/`do_compute_gradient_hessian`
 *       run on the real FEM device buffers and the assembled doublets/triplets are
 *       compared against the same hand-written host model.
 *
 * Coverage map:
 *  1) ARAP closed forms ........................... fem_constitution_a_arap_closed_forms
 *  2) SN(H/SNK1/SMT) closed forms ................. fem_constitution_a_snh_closed_forms
 *  3) finite differences + symmetry + PSD ......... fem_constitution_a_finite_difference
 *  4) degenerates pinned from the reference ....... fem_constitution_a_degenerate_inputs
 *  5) assembler index semantics (host) ............ fem_constitution_a_assembler
 *  6) device == host for the element routines ..... fem_constitution_a_device_vs_host
 *  7) real scene: init + energy + gradient/Hessian .. fem_constitution_a_scene
 *
 * Shared-binary rule (cookbook §7.14 rule 6): every system registered by every test
 * file is built for every scene, and `luisa_fem_base_test.cpp` registers a *test*
 * dim-3 constitution whose uid is ARAP's (9).  The method attributes a (dim, uid)
 * geometry group to the *first* system registered with that uid, so the ARAP scene
 * below detects whether the group is ours and says so (the unique-uid
 * StableNeoHookean scene carries the strong end-to-end checks).  The probe system
 * registered here only *reads* the built systems, so it cannot change another test
 * file's expectations.
 *********************************************************************/

#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"

#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/arap.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/elastic_moduli.h>

#include <type_define.h>
#include <frame_stream.h>
#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_elastics.h>
#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/arap_3d.h>
#include <finite_element/constitutions/stable_neo_hookean_3d.h>
#include <utils/matrix_assembler.h>
#include <luisa_device/linear/linear.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <random>
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

namespace
{
namespace sym_arap = uipc::backend::luisa::sym::arap_3d;
namespace sym_snh = uipc::backend::luisa::sym::stable_neo_hookean_3d;
namespace sym_simp = uipc::backend::luisa::sym::simplified_stable_neo_hookean_3d;
namespace snk1 = uipc::backend::luisa::snk1;

// ---------------------------------------------------------------------------
//  Eigen <-> numerics-layer shims (the T5.1 convention): the hand-derived oracles
//  are written in Eigen notation while the ported routines work on the numerics
//  layer's types.
// ---------------------------------------------------------------------------
[[nodiscard]] inline lps::LVec<double, 3> lv(const Vector3& v) noexcept
{
    return lps::LVec<double, 3>{v(0), v(1), v(2)};
}

[[nodiscard]] inline lps::LMat<double, 3, 3> lm(const Matrix3x3& m) noexcept
{
    lps::LMat<double, 3, 3> out;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] inline Matrix3x3 to_eigen(const lps::LMat<double, 3, 3>& m) noexcept
{
    Matrix3x3 out;
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] inline lps::LVec<double, 9> lv9(const lps::LMat<double, 3, 3>& m) noexcept
{
    return uipc::backend::luisa::flatten(m);
}

[[nodiscard]] inline lps::LVec<double, 12> lv12(const Vector12& v) noexcept
{
    lps::LVec<double, 12> out;
    for(int i = 0; i < 12; ++i)
        out(i) = v(i);
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

[[nodiscard]] inline Vector12 to_eigen(const lps::LVec<double, 12>& v) noexcept
{
    Vector12 out;
    for(int i = 0; i < 12; ++i)
        out(i) = v(i);
    return out;
}

[[nodiscard]] inline Matrix12x12 to_eigen(const lps::LMat<double, 12, 12>& m) noexcept
{
    Matrix12x12 out;
    for(int c = 0; c < 12; ++c)
        for(int r = 0; r < 12; ++r)
            out(r, c) = m(r, c);
    return out;
}

[[nodiscard]] inline Vector9 to_eigen(const lps::LVec<double, 9>& v) noexcept
{
    Vector9 out;
    for(int i = 0; i < 9; ++i)
        out(i) = v(i);
    return out;
}

[[nodiscard]] inline Matrix9x9 to_eigen(const lps::LMat<double, 9, 9>& m) noexcept
{
    Matrix9x9 out;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
            out(r, c) = m(r, c);
    return out;
}

constexpr double kFdH = 1e-6;

double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

/// `max |entry|` of any Eigen matrix/vector *expression*.
template <typename Derived>
double max_abs(const Eigen::MatrixBase<Derived>& m)
{
    return m.cwiseAbs().maxCoeff();
}

/// Smallest eigenvalue of a symmetric 12x12 (Eigen's self-adjoint solver).
double min_eigenvalue(const Matrix12x12& m)
{
    Eigen::SelfAdjointEigenSolver<Matrix12x12> es(m);
    return es.eigenvalues()(0);
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

// ---------------------------------------------------------------------------
//  hand-derived tetrahedron oracle
//
//  Reference ("unit") tet: X0 = (0,0,0), X1 = (1,0,0), X2 = (0,1,0), X3 = (0,0,1)
//  =>  Dm = Ds = I, Dm^-1 = I, rest volume V = det(Ds)/6 = 1/6, and the method's
//  shape gradients are g_1..g_3 = the columns of Dm^-1 = I with
//  g_0 = -(g_1 + g_2 + g_3):
//      g_0 = (-1,-1,-1), g_1 = (1,0,0), g_2 = (0,1,0), g_3 = (0,0,1).
// ---------------------------------------------------------------------------
struct TetDesc
{
    Vector3 X0, X1, X2, X3;  // material configuration
    Vector3 x0, x1, x2, x3;  // deformed configuration
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

/// `x -> F x + t` applied to the material configuration.
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

std::array<Vector3, 4> unit_tet_shape_gradients()
{
    return {Vector3{-1.0, -1.0, -1.0},
            Vector3{1.0, 0.0, 0.0},
            Vector3{0.0, 1.0, 0.0},
            Vector3{0.0, 0.0, 1.0}};
}

// ---------------------------------------------------------------------------
//  the ported per-element routines, called exactly like the kernels do
// ---------------------------------------------------------------------------
std::array<Vector3, 4> tet_x(const TetDesc& t)
{
    return {t.x0, t.x1, t.x2, t.x3};
}

lps::LMat<double, 3, 3> dm_inv(const TetDesc& t)
{
    return fem::Dm_inv(lv(t.X0), lv(t.X1), lv(t.X2), lv(t.X3));
}

double rest_volume(const TetDesc& t)
{
    return lps::determinant(fem::Ds(lv(t.X0), lv(t.X1), lv(t.X2), lv(t.X3))) / 6.0;
}

double arap_energy(const std::array<Vector3, 4>& x,
                   const lps::LMat<double, 3, 3>& DmInv,
                   double                         kappa,
                   double                         dt,
                   double                         volume)
{
    return gg::arap_3d_detail::tet_energy<double>(
        kappa, dt, volume, lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3]), DmInv);
}

Vector12 arap_gradient(const std::array<Vector3, 4>& x,
                       const lps::LMat<double, 3, 3>& DmInv,
                       double                         kappa,
                       double                         dt,
                       double                         volume)
{
    return to_eigen(gg::arap_3d_detail::tet_gradient<double>(
        kappa, dt, volume, lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3]), DmInv));
}

Matrix12x12 arap_hessian(const std::array<Vector3, 4>& x,
                         const lps::LMat<double, 3, 3>& DmInv,
                         double                         kappa,
                         double                         dt,
                         double                         volume)
{
    return to_eigen(gg::arap_3d_detail::tet_hessian<double>(
        kappa, dt, volume, lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3]), DmInv));
}

double snh_energy(const std::array<Vector3, 4>& x,
                  const lps::LMat<double, 3, 3>& DmInv,
                  double                         mu,
                  double                         lambda,
                  double                         dt,
                  double                         volume)
{
    return gg::snh_3d_detail::tet_energy<double>(
        mu, lambda, dt, volume, lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3]), DmInv);
}

Vector12 snh_gradient(const std::array<Vector3, 4>& x,
                      const lps::LMat<double, 3, 3>& DmInv,
                      double                         mu,
                      double                         lambda,
                      double                         dt,
                      double                         volume)
{
    return to_eigen(gg::snh_3d_detail::tet_gradient<double>(
        mu, lambda, dt, volume, lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3]), DmInv));
}

Matrix12x12 snh_hessian(const std::array<Vector3, 4>& x,
                        const lps::LMat<double, 3, 3>& DmInv,
                        double                         mu,
                        double                         lambda,
                        double                         dt,
                        double                         volume)
{
    return to_eigen(gg::snh_3d_detail::tet_hessian<double>(
        mu, lambda, dt, volume, lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3]), DmInv));
}

/// Perturb one dof (dof = 3 * vertex + component) of the deformed configuration.
TetDesc perturb(const TetDesc& t, int dof, double d)
{
    TetDesc out = t;
    auto    x   = tet_x(out);
    x[dof / 3](dof % 3) += d;
    out.x0 = x[0];
    out.x1 = x[1];
    out.x2 = x[2];
    out.x3 = x[3];
    return out;
}

/// Central-difference gradient of an element energy w.r.t. the 12 dofs.
template <typename Eval>
Vector12 fd_gradient(const Eval& eval, const TetDesc& t)
{
    Vector12 g;
    for(int k = 0; k < 12; ++k)
    {
        const double ep = eval(perturb(t, k, kFdH));
        const double em = eval(perturb(t, k, -kFdH));
        g(k)            = (ep - em) / (2 * kFdH);
    }
    return g;
}

/// Central-difference Hessian from an analytic gradient.
template <typename Grad>
Matrix12x12 fd_hessian(const Grad& grad, const TetDesc& t)
{
    Matrix12x12 H;
    for(int k = 0; k < 12; ++k)
    {
        const Vector12 gp = grad(perturb(t, k, kFdH));
        const Vector12 gm = grad(perturb(t, k, -kFdH));
        H.col(k)          = (gp - gm) / (2 * kFdH);
    }
    return 0.5 * (H + H.transpose());
}

// --- finite differences on F (`sym::*` siblings take F directly) -------------
template <typename Fn>
Vector9 fd_gradient_F(const Fn& f, const lps::LMat<double, 3, 3>& F)
{
    Vector9 g;
    for(int k = 0; k < 9; ++k)
    {
        const int r = k % 3, c = k / 3;
        auto      Fp = F, Fm = F;
        Fp(r, c) += kFdH;
        Fm(r, c) -= kFdH;
        g(k) = (f(Fp) - f(Fm)) / (2 * kFdH);
    }
    return g;
}

template <typename Grad>
Matrix9x9 fd_hessian_F(const Grad& grad, const lps::LMat<double, 3, 3>& F)
{
    Matrix9x9 H;
    for(int k = 0; k < 9; ++k)
    {
        const int r = k % 3, c = k / 3;
        auto      Fp = F, Fm = F;
        Fp(r, c) += kFdH;
        Fm(r, c) -= kFdH;
        const Vector9 gp = to_eigen(grad(Fp));
        const Vector9 gm = to_eigen(grad(Fm));
        H.col(k)         = (gp - gm) / (2 * kFdH);
    }
    return 0.5 * (H + H.transpose());
}

// ---------------------------------------------------------------------------
//  deterministic admissible random states
//
//  "Admissible" = a well-conditioned rest tet (finite Dm^-1) and a deformed state
//  that is a rotation of the rest shape plus a moderate affine strain
//  (|strain| <= 0.25), which keeps the local Hessians inside their
//  positive-semidefinite branch (the test asserts that).
// ---------------------------------------------------------------------------
std::mt19937& rng()
{
    static std::mt19937 gen{20240521u};
    return gen;
}

double rand_uniform(double lo, double hi)
{
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng());
}

TetDesc random_tet()
{
    TetDesc t;
    t.X0 = Vector3{0.0, 0.0, 0.0};
    t.X1 = Vector3{1.0, 0.15, -0.1} * rand_uniform(0.8, 1.3);
    t.X2 = Vector3{-0.1, 1.0, 0.2} * rand_uniform(0.8, 1.3);
    t.X3 = Vector3{0.12, -0.18, 1.0} * rand_uniform(0.8, 1.3);

    Vector3 axis{rand_uniform(-1.0, 1.0), rand_uniform(-1.0, 1.0), rand_uniform(-1.0, 1.0)};
    if(axis.norm() < 1e-6)
        axis = Vector3{1.0, 0.0, 0.0};

    Matrix3x3 strain = Matrix3x3::Zero();
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            strain(r, c) = rand_uniform(-0.25, 0.25);

    const Matrix3x3 F = rotation_matrix(axis, rand_uniform(-3.14159, 3.14159))
                        * (Matrix3x3::Identity() + strain);
    return affine_tet(t, F, Vector3{0.3, -0.4, 0.5});
}

// ---------------------------------------------------------------------------
//  the assembled model: doublets/triplets exactly as the ported assemblers write
//  them (the host oracle for the scene test)
// ---------------------------------------------------------------------------
struct AssembledModel
{
    std::vector<double>  energies;
    std::vector<int32_t> doublet_indices;
    std::vector<double>  doublet_values;
    std::vector<int32_t> triplet_rows;
    std::vector<int32_t> triplet_cols;
    std::vector<double>  triplet_values;
};

struct ElementResult
{
    double       energy = 0.0;
    lps::LVec<double, 12> gradient;
    lps::LMat<double, 12, 12> hessian;
};

/// `include_hessian = false` models the `gradient_only` path (the reference's
/// kernel returns before writing any triplet).
template <typename ElemFn>
AssembledModel build_model(SizeT                        n,
                           const std::vector<Vector4i>& tets,
                           bool                         include_hessian,
                           ElemFn&&                     elem)
{
    AssembledModel m;
    m.energies.resize(n);
    m.doublet_indices.resize(n * 4);
    m.doublet_values.resize(n * 12);
    if(include_hessian)
    {
        m.triplet_rows.resize(n * 10);
        m.triplet_cols.resize(n * 10);
        m.triplet_values.resize(n * 90);
    }

    for(SizeT e = 0; e < n; ++e)
    {
        const Vector4i      tet = tets[e];
        const ElementResult res = elem(e, tet);
        m.energies[e]           = res.energy;

        // `DoubletVectorAssembler::segment<4>(4e).write(tet, G12)`
        for(int i = 0; i < 4; ++i)
        {
            m.doublet_indices[4 * e + i] = tet(i);
            for(int c = 0; c < 3; ++c)
                m.doublet_values[(4 * e + i) * 3 + c] = res.gradient(3 * i + c);
        }

        if(!include_hessian)
            continue;

        // `TripletMatrixAssembler::half_block<4>(10e).write(tet, H12)` -- the block
        // belongs to the pair sorted by global index (the reference's `upper_LR`)
        int k = 0;
        for(int i = 0; i < 4; ++i)
            for(int j = i; j < 4; ++j, ++k)
            {
                const bool keep  = tet(i) < tet(j);
                const int  left  = keep ? i : j;
                const int  right = keep ? j : i;
                m.triplet_rows[10 * e + k] = tet(left);
                m.triplet_cols[10 * e + k] = tet(right);
                for(int c = 0; c < 3; ++c)
                    for(int rr = 0; rr < 3; ++rr)
                        m.triplet_values[(10 * e + k) * 9 + c * 3 + rr] =
                            res.hessian(3 * left + rr, 3 * right + c);
            }
    }
    return m;
}

// ---------------------------------------------------------------------------
//  upload helpers (host Eigen -> LC buffers)
// ---------------------------------------------------------------------------
template <typename T, typename U, typename Conv>
lud::LDeviceBuffer<T> upload(const std::vector<U>& src, Conv&& conv)
{
    std::vector<T> host;
    host.reserve(src.size());
    for(auto& v : src)
        host.push_back(conv(v));
    lud::LDeviceBuffer<T> buffer;
    buffer.copy_from(span<const T>{host});
    return buffer;
}

template <typename T>
lud::LDeviceBuffer<T> upload(const std::vector<T>& src)
{
    lud::LDeviceBuffer<T> buffer;
    buffer.copy_from(span<const T>{src});
    return buffer;
}
}  // namespace

// ---------------------------------------------------------------------------
//  the shared-binary probe (see the file comment)
// ---------------------------------------------------------------------------
class FEMConstitutionAProbe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::FiniteElementMethod*        fem      = nullptr;
    inline static gg::FiniteElementElastics*      elastics = nullptr;
    inline static gg::ARAP3D*                     arap     = nullptr;
    inline static gg::StableNeoHookean3D*         snh      = nullptr;

  private:
    void do_build() override
    {
        fem      = &require<gg::FiniteElementMethod>();
        elastics = &require<gg::FiniteElementElastics>();

        // `find<T>({.exact = false})` returns the registered system of type T when
        // the scene's constitution uids made the engine build it (else nullptr).
        arap = find<gg::ARAP3D>({.exact = false});
        snh  = find<gg::StableNeoHookean3D>({.exact = false});
    }
};
REGISTER_SIM_SYSTEM(FEMConstitutionAProbe);

namespace
{
// ---------------------------------------------------------------------------
//  the scene: two well-separated unit tets (rest volume 1/6, Dm^-1 = I each)
// ---------------------------------------------------------------------------
struct TetSceneDesc
{
    std::vector<Vector3>  positions;
    std::vector<Vector4i> tets;
    std::vector<double>   params;   // per-tet kappa (ARAP) or mu (SNH)
    std::vector<double>   params2;  // per-tet lambda (SNH)
    double                density = 1000.0;
};

TetSceneDesc two_unit_tets()
{
    TetSceneDesc d;
    d.positions = {{0.0, 0.0, 0.0},
                   {1.0, 0.0, 0.0},
                   {0.0, 1.0, 0.0},
                   {0.0, 0.0, 1.0},
                   {4.0, 0.0, 0.0},
                   {5.0, 0.0, 0.0},
                   {4.0, 1.0, 0.0},
                   {4.0, 0.0, 1.0}};
    d.tets      = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    return d;
}

Scene build_arap_scene(const TetSceneDesc& desc)
{
    auto config = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = false;

    Scene scene{config};
    auto  object = scene.objects().create("t52a_arap");

    ARAP arap;
    auto mesh = tetmesh(span<const Vector3>{desc.positions}, span<const Vector4i>{desc.tets});
    label_surface(mesh);
    arap.apply_to(mesh, desc.params[0], desc.density);
    {
        // distinct per-tet kappa (the constitution must read them per primitive)
        auto kappa = mesh.tetrahedra().find<Float>("kappa");
        REQUIRE(kappa != nullptr);
        auto kappa_view = view(*kappa);
        for(SizeT i = 0; i < kappa_view.size(); ++i)
            kappa_view[i] = desc.params[i];
    }
    object->geometries().create(mesh);
    return scene;
}

Scene build_snh_scene(const TetSceneDesc& desc)
{
    auto config = Scene::default_config();
    config["sanity_check"]["enable"] = false;
    config["contact"]["enable"]      = false;

    Scene scene{config};
    auto  object = scene.objects().create("t52a_snh");

    StableNeoHookean snh;
    auto mesh = tetmesh(span<const Vector3>{desc.positions}, span<const Vector4i>{desc.tets});
    label_surface(mesh);
    snh.apply_to(mesh, ElasticModuli::youngs_poisson(1.0e6, 0.3), desc.density);
    {
        // distinct per-tet (mu, lambda) -- the frontend writes both attributes, the
        // constitution reads them verbatim
        auto mu     = mesh.tetrahedra().find<Float>("mu");
        auto lambda = mesh.tetrahedra().find<Float>("lambda");
        REQUIRE(mu != nullptr);
        REQUIRE(lambda != nullptr);
        auto mu_view     = view(*mu);
        auto lambda_view = view(*lambda);
        for(SizeT i = 0; i < mu_view.size(); ++i)
        {
            mu_view[i]     = desc.params[i];
            lambda_view[i] = desc.params2[i];
        }
    }
    object->geometries().create(mesh);
    return scene;
}
}  // namespace

// ---------------------------------------------------------------------------
//  1) ARAP: hand-derived closed forms
//
//  ARAP energy:  E(F) = kappa * v * ||F - R||_F^2 with R = U V^T from the signed
//  SVD (U, V proper rotations).  Since ||F - R||^2 = sum_i (sigma_i - 1)^2,
//    * F = Q (any proper rotation): sigma = (1,1,1), R = Q  =>  E = 0 and
//      dE/dF = 2 (F - R) = 0,
//    * F = s I (uniform stretch, s > 0): R = I  =>  E = 3 kappa v (s - 1)^2 and
//      dE/dF = 2 kappa v (s - 1) I; with the unit tet's shape gradients the element
//      gradient is G_i = 2 kappa v (s - 1) * g_i.
//  The element routines scale `kappa * dt * dt`, so the expected energy is
//  `kappa dt^2 v 3 (s-1)^2`.
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_a_arap_closed_forms", "[luisa][fem][constitution_a]")
{
    test_runtime();

    const double kappa = 1.3;
    const double dt    = 1.0;
    const auto   rest  = unit_tet();
    const double V     = rest_volume(rest);
    CHECK(rel_err(V, 1.0 / 6.0) <= 1e-15);

    const auto DmInv = dm_inv(rest);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(std::abs(DmInv(r, c) - (r == c ? 1.0 : 0.0)) <= 1e-15);

    const auto shape_grads = unit_tet_shape_gradients();

    // ---- F = I: a rigid (rest) state has zero energy and zero gradient ---------
    {
        const auto x = tet_x(rest);
        CHECK(std::abs(arap_energy(x, DmInv, kappa, dt, V)) <= 1e-30);
        CHECK(max_abs(arap_gradient(x, DmInv, kappa, dt, V)) <= 1e-15);
    }

    // ---- F = Q: an arbitrary rotation has zero energy and zero gradient --------
    {
        const Matrix3x3 Q = rotation_matrix(Vector3{1.0, 2.0, -3.0}, 0.7);
        const auto      t = affine_tet(rest, Q, Vector3{2.0, -1.0, 0.5});
        const auto      x = tet_x(t);
        CHECK(std::abs(arap_energy(x, DmInv, kappa, dt, V)) <= 1e-30);
        CHECK(max_abs(arap_gradient(x, DmInv, kappa, dt, V)) <= 1e-15);

        // finite, symmetric and PSD, with the 6 rigid-body modes as its nullspace
        const Matrix12x12 H = arap_hessian(x, DmInv, kappa, dt, V);
        CHECK(max_abs(H - H.transpose()) <= 1e-12);
        CHECK(min_eigenvalue(H) >= -1e-10);
        CHECK(rel_err(min_eigenvalue(H), 0.0) <= 1e-6);
    }

    // ---- F = s I: the analytic kappa * sum (sigma_i - 1)^2 law ----------------
    {
        const double s      = 1.7;
        const auto   t      = affine_tet(rest, Matrix3x3::Identity() * s, Vector3::Zero());
        const auto   x      = tet_x(t);
        const double expect = kappa * dt * dt * V * 3.0 * (s - 1.0) * (s - 1.0);
        CHECK(rel_err(arap_energy(x, DmInv, kappa, dt, V), expect) <= 1e-14);

        const double c = 2.0 * kappa * V * (s - 1.0);
        Vector12     G_expect;
        for(int i = 0; i < 4; ++i)
            for(int r = 0; r < 3; ++r)
                G_expect(3 * i + r) = c * shape_grads[i](r);
        CHECK(max_abs(arap_gradient(x, DmInv, kappa, dt, V) - G_expect) <= 1e-12);

        const Matrix12x12 H = arap_hessian(x, DmInv, kappa, dt, V);
        CHECK(max_abs(H - H.transpose()) <= 1e-12);
        CHECK(min_eigenvalue(H) >= -1e-10);
    }

    // ---- scaling: quadratic in dt, linear in kappa and in the rest volume -----
    {
        const auto   t  = affine_tet(rest, Matrix3x3::Identity() * 1.4, Vector3::Zero());
        const auto   x  = tet_x(t);
        const double E1 = arap_energy(x, DmInv, kappa, 1.0, V);
        CHECK(rel_err(arap_energy(x, DmInv, kappa, 2.0, V), 4.0 * E1) <= 1e-14);
        CHECK(rel_err(arap_energy(x, DmInv, 2.0 * kappa, 1.0, V), 2.0 * E1) <= 1e-14);
        CHECK(rel_err(arap_energy(x, DmInv, kappa, 1.0, 2.0 * V), 2.0 * E1) <= 1e-14);
    }

    // ---- the reference's `sym::arap_3d` surface, called like the .cu does -----
    {
        const auto   F   = lm(Matrix3x3::Identity() * 1.7);
        const double kt2 = kappa * dt * dt;  // the reference passes `kappas * dt * dt`

        double E = 0.0;
        sym_arap::E<double>(E, kt2, V, F);
        CHECK(rel_err(E, kt2 * V * 3.0 * 0.49) <= 1e-14);

        lps::LVec<double, 9> dEdF;
        sym_arap::dEdF<double>(dEdF, kt2, V, F);
        const double c = 2.0 * kt2 * V * 0.7;
        for(int i = 0; i < 9; ++i)
            CHECK(rel_err(dEdF(i), c * ((i % 4 == 0) ? 1.0 : 0.0)) <= 1e-14);

        lps::LMat<double, 9, 9> ddEddF;
        sym_arap::ddEddF<double>(ddEddF, kt2, V, F);
        // E = kappa v sum (sigma_i - 1)^2, so dE/dF = 2kv(F-R) and the Hessian must
        // be the derivative of that: compare against FD of dEdF
        auto grad_F = [&](const lps::LMat<double, 3, 3>& Fm)
        {
            lps::LVec<double, 9> g;
            sym_arap::dEdF<double>(g, kt2, V, Fm);
            return g;
        };
        const Matrix9x9 h_fd = fd_hessian_F(grad_F, F);
        CHECK(rel_err(max_abs(to_eigen(ddEddF) - h_fd), 0.0) <= 1e-8);
    }
}

// ---------------------------------------------------------------------------
//  2) The stable Neo-Hookean variants
//
//  SNK1 (the constitution's variant), unit volume:
//      Psi = 0.5 mu (Ic - 3) + 0.5 lambda (J - 1 - mu/lambda)^2,
//      Ic = |F|^2, J = det F.  Hence
//    * F = I or F = Q: Ic = 3, J = 1 => Psi = 0.5 mu^2 / lambda (a constant) and
//      dPsi/dF = mu F + (lambda (J-1) - mu) cof(F) = 0 for a rotation,
//    * F = s I: J = s^3, Ic = 3 s^2 =>
//      Psi = 0.5 (mu (3 s^2 - 3) + lambda (s^3 - 1 - mu/lambda)^2),
//      dPsi/dF = (mu s + (lambda (s^3 - 1) - mu) s^2) I.
//  `sym::stable_neo_hookean_3d` (the reference's unreferenced SMT sibling):
//      E = 0.5 lambda (J - alpha)^2 + 0.5 mu (Ic - 3) - 0.5 mu log(Ic + 1),
//      alpha = 1 + 0.75 mu / lambda
//      dE/dF = mu (1 - 1/(Ic+1)) F + (lambda (J - 1 - 0.75 mu/lambda)) cof(F)
//  `sym::simplified_stable_neo_hookean_3d` (the transcribed SymEigen output):
//      E = 0.5 lambda (J-1)^2 - mu (J-1) + 0.5 mu (Ic - 3) + mu^2 / lambda^2.
//  Both siblings are checked against central differences of *their own* energy, so
//  the transcriptions are verified rather than assumed.
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_a_snh_closed_forms", "[luisa][fem][constitution_a]")
{
    test_runtime();

    const double mu     = 1.7;
    const double lambda = 3.3;
    const double dt     = 1.0;
    const auto   rest   = unit_tet();
    const double V      = rest_volume(rest);
    const auto   DmInv  = dm_inv(rest);
    const auto   shape_grads = unit_tet_shape_gradients();

    // ---- F = I and F = Q: constant energy, zero gradient ----------------------
    {
        const auto x = tet_x(rest);
        CHECK(rel_err(snh_energy(x, DmInv, mu, lambda, dt, V),
                      0.5 * mu * mu / lambda * dt * dt * V)
              <= 1e-14);
        CHECK(max_abs(snh_gradient(x, DmInv, mu, lambda, dt, V)) <= 1e-15);

        const Matrix3x3 Q = rotation_matrix(Vector3{-1.0, 0.5, 2.0}, -1.1);
        const auto      t = affine_tet(rest, Q, Vector3{0.1, 0.2, 0.3});
        const auto      y = tet_x(t);
        CHECK(rel_err(snh_energy(y, DmInv, mu, lambda, dt, V),
                      0.5 * mu * mu / lambda * dt * dt * V)
              <= 1e-14);
        CHECK(max_abs(snh_gradient(y, DmInv, mu, lambda, dt, V)) <= 1e-13);
    }

    // ---- F = s I: the hand-derived stretch law -------------------------------
    {
        const double s = 1.7;
        const auto   t = affine_tet(rest, Matrix3x3::Identity() * s, Vector3::Zero());
        const auto   x = tet_x(t);

        const double psi = 0.5
                           * (mu * (3.0 * s * s - 3.0)
                              + lambda * std::pow(s * s * s - 1.0 - mu / lambda, 2.0));
        CHECK(rel_err(snh_energy(x, DmInv, mu, lambda, dt, V), psi * dt * dt * V) <= 1e-14);

        const double dpsi = mu * s + (lambda * (s * s * s - 1.0) - mu) * s * s;
        const Vector12 G  = snh_gradient(x, DmInv, mu, lambda, dt, V);
        for(int i = 0; i < 4; ++i)
            for(int r = 0; r < 3; ++r)
                CHECK(rel_err(G(3 * i + r), dt * dt * V * dpsi * shape_grads[i](r)) <= 1e-13);

        const Matrix12x12 H = snh_hessian(x, DmInv, mu, lambda, dt, V);
        CHECK(max_abs(H - H.transpose()) <= 1e-12);
        CHECK(min_eigenvalue(H) >= -1e-10);
    }

    // ---- the SNK1 surface, called like the .cu does --------------------------
    {
        const auto F    = lm(Matrix3x3::Identity() * 1.7);
        const double s  = 1.7;
        double       E  = 0.0;
        snk1::E<double>(E, mu, lambda, F);
        const double psi = 0.5
                           * (mu * (3.0 * s * s - 3.0)
                              + lambda * std::pow(s * s * s - 1.0 - mu / lambda, 2.0));
        CHECK(rel_err(E, psi) <= 1e-14);

        lps::LMat<double, 3, 3> dEdF;
        snk1::dEdF<double>(dEdF, mu, lambda, F);
        const double c = mu * s + (lambda * (s * s * s - 1.0) - mu) * s * s;
        for(int r = 0; r < 3; ++r)
            for(int col = 0; col < 3; ++col)
                CHECK(rel_err(dEdF(r, col), c * (r == col ? 1.0 : 0.0)) <= 1e-14);

        // the analytically SPD-projected Hessian is the *eigenvalue-clamped* true
        // Hessian: at this uniform stretch the true 9x9 Hessian is already
        // indefinite, so the projection is active and the spectra must be
        // `max(eig(true), 0)`
        auto energy = [&](const lps::LMat<double, 3, 3>& Fm)
        {
            double R = 0.0;
            snk1::E<double>(R, mu, lambda, Fm);
            return R;
        };
        auto grad = [&](const lps::LMat<double, 3, 3>& Fm)
        {
            lps::LMat<double, 3, 3> g;
            snk1::dEdF<double>(g, mu, lambda, Fm);
            return lv9(g);
        };
        const Matrix9x9 h_fd = fd_hessian_F(grad, F);
        lps::LMat<double, 9, 9> H9;
        snk1::ddEddF_spd<double>(H9, mu, lambda, F);
        Eigen::SelfAdjointEigenSolver<Matrix9x9> es_ref(to_eigen(H9)), es_fd(h_fd);
        CHECK((es_ref.eigenvalues() - es_fd.eigenvalues().cwiseMax(0.0))
                  .cwiseAbs()
                  .maxCoeff()
              <= 1e-6);
        // the projected Hessian is PSD by construction
        CHECK(es_ref.eigenvalues()(0) >= -1e-9);
    }

    // ---- sym::stable_neo_hookean_3d: E / dEdVecF / ddEddVecF vs FD -----------
    {
        double worst_g = 0.0, worst_h = 0.0;
        for(int trial = 0; trial < 20; ++trial)
        {
            const TetDesc t = random_tet();
            const auto    x = tet_x(t);
            const auto    F = fem::Ds(lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3])) * DmInv;

            auto energy = [&](const lps::LMat<double, 3, 3>& Fm)
            {
                double R = 0.0;
                sym_snh::E<double>(R, mu, lambda, Fm);
                return R;
            };
            auto grad = [&](const lps::LMat<double, 3, 3>& Fm)
            {
                lps::LMat<double, 3, 3> g;
                sym_snh::dEdVecF<double>(g, mu, lambda, Fm);
                return lv9(g);
            };

            // the hand-derived gradient: mu (1 - 1/(Ic+1)) F + (lambda (J-1-0.75mu/lam)) cof(F)
            const double J   = lps::determinant(F);
            const double Ic  = gg::ddot(F, F);
            const auto   cof = fem::dJdF(F);
            const lps::LMat<double, 3, 3> g_expect =
                F * lps::detail::mul(mu, lps::detail::sub(1.0, lps::detail::div(1.0, lps::detail::add(Ic, 1.0))))
                + cof * (lps::detail::mul(lambda, lps::detail::sub(lps::detail::sub(J, 1.0),
                                                                  lps::detail::div(lps::detail::mul(0.75, mu), lambda))));

            lps::LMat<double, 3, 3> g;
            sym_snh::dEdVecF<double>(g, mu, lambda, F);
            for(int r = 0; r < 3; ++r)
                for(int c = 0; c < 3; ++c)
                    CHECK(rel_err(g(r, c), g_expect(r, c)) <= 1e-13);

            // the closed form of E for F = s I is checked above; here FD validates
            // the transcribed PEPF entries and the f*hat blocks of the Hessian
            const Vector9 g_fd = fd_gradient_F(energy, F);
            worst_g = std::max(worst_g, rel_err(max_abs(to_eigen(lv9(g)) - g_fd), 0.0));

            lps::LMat<double, 9, 9> H9;
            sym_snh::ddEddVecF<double>(H9, mu, lambda, F);
            const Matrix9x9 h_fd = fd_hessian_F(grad, F);
            worst_h = std::max(worst_h, rel_err(max_abs(to_eigen(H9) - h_fd), 0.0));
        }
        std::printf("[t52a] sym::stable_neo_hookean_3d vs FD: grad rel %.3e  hess rel %.3e\n",
                    worst_g, worst_h);
        CHECK(worst_g <= 1e-6);
        CHECK(worst_h <= 1e-6);
    }

    // ---- sym::simplified_stable_neo_hookean_3d (the transcribed SMT output) ---
    {
        double worst_g = 0.0, worst_h = 0.0;
        for(int trial = 0; trial < 20; ++trial)
        {
            const TetDesc t = random_tet();
            const auto    x = tet_x(t);
            const auto    F = fem::Ds(lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3])) * DmInv;
            const Vector9 vecF = to_eigen(lv9(F));

            auto energy = [&](const lps::LMat<double, 3, 3>& Fm)
            {
                double R = 0.0;
                sym_simp::E<double>(R, mu, lambda, lv9(Fm));
                return R;
            };
            auto grad = [&](const lps::LMat<double, 3, 3>& Fm)
            {
                lps::LVec<double, 9> g;
                sym_simp::dEdVecF<double>(g, mu, lambda, lv9(Fm));
                return g;
            };

            // hand-derived closed form (the file's own LaTeX):
            //   E = 0.5 lambda (J-1)^2 - mu (J-1) + 0.5 mu (Ic-3) + mu^2/lambda^2
            const double J  = lps::determinant(F);
            const double Ic = gg::ddot(F, F);
            const double e_expect =
                0.5 * lambda * (J - 1.0) * (J - 1.0) - mu * (J - 1.0) + 0.5 * mu * (Ic - 3.0)
                + mu * mu / (lambda * lambda);
            double e = 0.0;
            sym_simp::E<double>(e, mu, lambda, lv9(F));
            CHECK(rel_err(e, e_expect) <= 1e-13);

            lps::LVec<double, 9> g;
            sym_simp::dEdVecF<double>(g, mu, lambda, lv9(F));
            worst_g = std::max(worst_g, rel_err(max_abs(to_eigen(g) - fd_gradient_F(energy, F)), 0.0));

            lps::LMat<double, 9, 9> H9;
            sym_simp::ddEddVecF<double>(H9, mu, lambda, lv9(F));
            worst_h = std::max(worst_h, rel_err(max_abs(to_eigen(H9) - fd_hessian_F(grad, F)), 0.0));
            (void)vecF;
        }
        std::printf("[t52a] sym::simplified_stable_neo_hookean_3d vs FD: grad rel %.3e"
                    "  hess rel %.3e\n",
                    worst_g, worst_h);
        CHECK(worst_g <= 1e-6);
        CHECK(worst_h <= 1e-6);
    }
}

// ---------------------------------------------------------------------------
//  3) finite differences of the ported per-element routines (w.r.t. the 12 dofs)
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_a_finite_difference", "[luisa][fem][constitution_a]")
{
    test_runtime();

    const double kappa = 0.9;
    const double mu = 1.7, lambda = 3.3;
    const double dt = 1.0;

    double arap_g_worst = 0.0, arap_h_worst = 0.0, arap_sym = 0.0, arap_min_eig = 0.0;
    double snh_g_worst = 0.0, snh_h_worst = 0.0, snh_sym = 0.0, snh_min_eig = 0.0;
    // the 9x9 (=F-space) Hessian of the *energy*, which is where both constitutions
    // apply their projection: `make_spd` (ARAP) resp. `ddEddF_spd` (SNK1)
    double arap9 = 0.0, snh9 = 0.0;
    int    projected_arap = 0, projected_snh = 0;

    for(int trial = 0; trial < 40; ++trial)
    {
        const TetDesc t    = random_tet();
        const double  V    = rest_volume(t);
        const auto    DmIn = dm_inv(t);

        // F of this affine state (the 9x9 oracles below differentiate w.r.t. F)
        const auto x = tet_x(t);
        const auto F = fem::Ds(lv(x[0]), lv(x[1]), lv(x[2]), lv(x[3])) * DmIn;

        // --- ARAP -------------------------------------------------------------
        {
            auto energy = [&](const TetDesc& p)
            { return arap_energy(tet_x(p), DmIn, kappa, dt, V); };
            auto grad = [&](const TetDesc& p)
            { return arap_gradient(tet_x(p), DmIn, kappa, dt, V); };

            arap_g_worst = std::max(arap_g_worst,
                                    rel_err(max_abs(arap_gradient(x, DmIn, kappa, dt, V)
                                                    - fd_gradient(energy, t)),
                                            0.0));
            const Matrix12x12 H_ref = arap_hessian(x, DmIn, kappa, dt, V);
            const Matrix12x12 H_fd  = fd_hessian(grad, t);
            arap_sym     = std::max(arap_sym, max_abs(H_ref - H_ref.transpose()));
            arap_min_eig = std::min(arap_min_eig, min_eigenvalue(H_ref));

            // `ARAP_Hessian` is the true Hessian of E; `make_spd` clamps it, so the
            // analytic element Hessian equals the FD one exactly where the true
            // Hessian is positive semidefinite (the projection is then inactive).
            if(min_eigenvalue(H_fd) >= -1e-8)
                arap_h_worst = std::max(arap_h_worst,
                                        rel_err(max_abs(H_ref - H_fd), 0.0));
            else
                ++projected_arap;

            // the projected 9x9 spectrum is max(eig(true 9x9 Hessian), 0)
            auto grad_F = [&](const lps::LMat<double, 3, 3>& Fm)
            {
                lps::LVec<double, 9> g;
                sym_arap::dEdF<double>(g, kappa * dt * dt, V, Fm);
                return g;
            };
            const Matrix9x9 h9_fd = fd_hessian_F(grad_F, F);
            lps::LMat<double, 9, 9> h9_an;
            sym_arap::ddEddF<double>(h9_an, kappa * dt * dt, V, F);
            lps::make_spd_inplace(h9_an);
            Eigen::SelfAdjointEigenSolver<Matrix9x9> es_an(to_eigen(h9_an)), es_fd9(h9_fd);
            arap9 = std::max(arap9,
                             (es_an.eigenvalues() - es_fd9.eigenvalues().cwiseMax(0.0))
                                 .cwiseAbs()
                                 .maxCoeff());
        }

        // --- stable Neo-Hookean (SNK1) ---------------------------------------
        {
            auto energy = [&](const TetDesc& p)
            { return snh_energy(tet_x(p), DmIn, mu, lambda, dt, V); };
            auto grad = [&](const TetDesc& p)
            { return snh_gradient(tet_x(p), DmIn, mu, lambda, dt, V); };

            snh_g_worst = std::max(snh_g_worst,
                                   rel_err(max_abs(snh_gradient(x, DmIn, mu, lambda, dt, V)
                                                   - fd_gradient(energy, t)),
                                           0.0));
            const Matrix12x12 H_ref = snh_hessian(x, DmIn, mu, lambda, dt, V);
            const Matrix12x12 H_fd  = fd_hessian(grad, t);
            snh_sym     = std::max(snh_sym, max_abs(H_ref - H_ref.transpose()));
            snh_min_eig = std::min(snh_min_eig, min_eigenvalue(H_ref));

            if(min_eigenvalue(H_fd) >= -1e-8)
                snh_h_worst = std::max(snh_h_worst, rel_err(max_abs(H_ref - H_fd), 0.0));
            else
                ++projected_snh;

            // the analytic, SPD-projected 9x9 eigen-system: its spectrum must be
            // max(eig(true 9x9 Hessian), 0)
            auto grad_F = [&](const lps::LMat<double, 3, 3>& Fm)
            {
                lps::LMat<double, 3, 3> g;
                snk1::dEdF<double>(g, mu, lambda, Fm);
                return lv9(g);
            };
            const Matrix9x9 h9_fd = fd_hessian_F(grad_F, F);
            lps::LMat<double, 9, 9> h9_an;
            snk1::ddEddF_spd<double>(h9_an, mu, lambda, F);
            Eigen::SelfAdjointEigenSolver<Matrix9x9> es_an(to_eigen(h9_an)), es_fd9(h9_fd);
            snh9 = std::max(snh9,
                            (es_an.eigenvalues() - es_fd9.eigenvalues().cwiseMax(0.0))
                                .cwiseAbs()
                                .maxCoeff());
        }
    }

    std::printf("[t52a] ARAP dof-FD: grad rel %.3e  hess rel %.3e  asym %.3e  min eig %.3e"
                "  (9x9 clamp %.3e, projected %d/40)\n",
                arap_g_worst, arap_h_worst, arap_sym, arap_min_eig, arap9, projected_arap);
    std::printf("[t52a] SNK1 dof-FD: grad rel %.3e  hess rel %.3e  asym %.3e  min eig %.3e"
                "  (9x9 clamp %.3e, projected %d/40)\n",
                snh_g_worst, snh_h_worst, snh_sym, snh_min_eig, snh9, projected_snh);

    // the brief's bounds are grad rel < 1e-4 and Hessian rel < 1e-3 (h = 1e-6);
    // the comparisons above are the PSD states, where the reference's projection is
    // inactive, so the analytic Hessian *is* the true one
    CHECK(arap_g_worst <= 1e-6);
    CHECK(arap_h_worst <= 1e-6);
    CHECK(snh_g_worst <= 1e-6);
    CHECK(snh_h_worst <= 1e-6);
    // in the (more strained) states where the true Hessian is indefinite, both
    // constitutions must produce exactly the eigenvalue-clamped one
    CHECK(arap9 <= 1e-6);
    CHECK(snh9 <= 1e-6);
    CHECK(arap_sym <= 1e-10);
    CHECK(snh_sym <= 1e-10);
    CHECK(arap_min_eig >= -1e-8);
    CHECK(snh_min_eig >= -1e-8);
}

// ---------------------------------------------------------------------------
//  4) degenerate states — pinned from the reference source, not assumed
//
//  Neither constitution checks its input (the only volume check in the reference
//  lives in the FEM method's Dm^-1/rest-volume kernel, see T5.1), so the behaviour
//  is whatever the formulas do:
//    * `ARAP_Hessian` divides by (sigma_i + sigma_j): a zero-volume state (all
//      sigma = 0) or a reflection (sigma_1 + sigma_2 = 0) makes the *Hessian*
//      non-finite, while E and dE/dF stay finite,
//    * the SNK1 and the SMT energies are polynomial in F (plus a log), so they are
//      finite on every well-defined input, and the SNK1 Hessian is clamped and
//      stays PSD even for a collapsed element.
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_a_degenerate_inputs", "[luisa][fem][constitution_a]")
{
    test_runtime();

    const double kappa = 0.9, mu = 1.7, lambda = 3.3, dt = 1.0;
    const auto   rest  = unit_tet();
    const double V     = rest_volume(rest);
    const auto   DmInv = dm_inv(rest);

    struct Degenerate
    {
        const char* name;
        Matrix3x3   F;
        bool        arap_hessian_finite;
    };

    const std::vector<Degenerate> cases{
        {"zero-volume (F = 0)", Matrix3x3::Zero(), false},
        {"singular (sigma = 1,1,0)",
         (Matrix3x3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0).finished(),
         true},
        {"inverted (F = -I)", Matrix3x3::Identity() * -1.0, false},
        {"reflection (diag(1,1,-1))",
         (Matrix3x3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0).finished(),
         false},
    };

    for(const auto& c : cases)
    {
        const auto t = affine_tet(rest, c.F, Vector3::Zero());
        const auto x = tet_x(t);
        INFO(c.name);

        // ---- ARAP -----------------------------------------------------------
        CHECK(std::isfinite(arap_energy(x, DmInv, kappa, dt, V)));

        lps::LVec<double, 9> dEdF;
        sym_arap::dEdF<double>(dEdF, kappa * dt * dt, V, lm(c.F));
        for(int i = 0; i < 9; ++i)
            CHECK(std::isfinite(dEdF(i)));

        lps::LMat<double, 9, 9> H9;
        sym_arap::ddEddF<double>(H9, kappa * dt * dt, V, lm(c.F));
        bool finite = true;
        for(int r = 0; r < 9; ++r)
            for(int cc = 0; cc < 9; ++cc)
                finite = finite && std::isfinite(H9(r, cc));
        CHECK(finite == c.arap_hessian_finite);
        // the projected element Hessian therefore either is finite or carries the
        // non-finite values through `make_spd` -- never silently wrong
        const Matrix12x12 H12 = arap_hessian(x, DmInv, kappa, dt, V);
        if(c.arap_hessian_finite)
        {
            for(int r = 0; r < 12; ++r)
                for(int cc = 0; cc < 12; ++cc)
                    CHECK(std::isfinite(H12(r, cc)));
        }
        else
        {
            CHECK(!std::isfinite(max_abs(H12)));
        }

        // ---- SNK1 -----------------------------------------------------------
        CHECK(std::isfinite(snh_energy(x, DmInv, mu, lambda, dt, V)));
        const Vector12 G = snh_gradient(x, DmInv, mu, lambda, dt, V);
        for(int i = 0; i < 12; ++i)
            CHECK(std::isfinite(G(i)));
        const Matrix12x12 H = snh_hessian(x, DmInv, mu, lambda, dt, V);
        bool h_finite = true;
        for(int r = 0; r < 12; ++r)
            for(int cc = 0; cc < 12; ++cc)
                h_finite = h_finite && std::isfinite(H(r, cc));
        CHECK(h_finite);
        CHECK(min_eigenvalue(H) >= -1e-8);

        // ---- the SMT sibling ------------------------------------------------
        const auto vecF = lv9(lm(c.F));
        double     e    = 0.0;
        lps::LVec<double, 9> g;
        lps::LMat<double, 9, 9> Hm;
        sym_simp::E<double>(e, mu, lambda, vecF);
        sym_simp::dEdVecF<double>(g, mu, lambda, vecF);
        sym_simp::ddEddVecF<double>(Hm, mu, lambda, vecF);
        CHECK(std::isfinite(e));
        for(int i = 0; i < 9; ++i)
            CHECK(std::isfinite(g(i)));
        for(int r = 0; r < 9; ++r)
            for(int cc = 0; cc < 9; ++cc)
                CHECK(std::isfinite(Hm(r, cc)));
    }
}

// ---------------------------------------------------------------------------
//  5) the ported assemblers on host memory (`storage::memory`)
//
//  `TripletMatrixAssembler::half_block` writes the block of the pair *sorted by
//  global index* (the reference's `upper_LR`), and `segment<4>` writes row `tet(i)`
//  at slot `4e + i`.  A hand-written host model of both is compared entry by entry,
//  including a fully descending tet (every pair swaps).
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_a_assembler", "[luisa][fem][constitution_a]")
{
    test_runtime();

    using HostDoublet = lla::DoubletVectorViewT<false, double, 3, lla::storage::memory>;
    using HostTriplet = lla::TripletMatrixViewT<false, double, 3, 3, lla::storage::memory>;

    constexpr SizeT n = 3;
    // every (i, j) pair order, including a fully descending tet
    const std::vector<Vector4i> tets{{0, 1, 2, 3}, {3, 1, 2, 0}, {5, 4, 7, 6}};

    std::vector<Vector12>    G(n);
    std::vector<Matrix12x12> H(n);
    for(SizeT e = 0; e < n; ++e)
    {
        for(int i = 0; i < 12; ++i)
            G[e](i) = 10.0 * e + i;
        for(int c = 0; c < 12; ++c)
            for(int r = 0; r < 12; ++r)
                H[e](r, c) = 100.0 * e + 10.0 * r + c;
    }

    std::vector<int32_t> d_indices(n * 4, -1);
    std::vector<double>  d_values(n * 12, 0.0);
    std::vector<int32_t> t_rows(n * 10, -1);
    std::vector<int32_t> t_cols(n * 10, -1);
    std::vector<double>  t_values(n * 90, 0.0);

    const SizeT vertex_count = 8;
    HostDoublet doublets(n * 4, n * 4, d_indices.data(), d_values.data());
    HostTriplet triplets(vertex_count, vertex_count, n * 10, t_rows.data(), t_cols.data(),
                         t_values.data());

    gg::DoubletVectorAssembler DVA{doublets};
    gg::TripletMatrixAssembler TMA{triplets};

    for(SizeT e = 0; e < n; ++e)
    {
        lps::LVec<IndexT, 4> tet;
        for(int i = 0; i < 4; ++i)
            tet(i) = tets[e](i);

        DVA.segment<4>(static_cast<IndexT>(4 * e)).write(tet, lv12(G[e]));
        TMA.half_block<4>(static_cast<IndexT>(10 * e)).write(tet, lm12(H[e]));
    }

    for(SizeT e = 0; e < n; ++e)
    {
        const Vector4i tet = tets[e];
        for(int i = 0; i < 4; ++i)
        {
            CHECK(d_indices[4 * e + i] == tet(i));
            for(int c = 0; c < 3; ++c)
                CHECK(rel_err(d_values[(4 * e + i) * 3 + c], G[e](3 * i + c)) <= 1e-15);
        }
        int k = 0;
        for(int i = 0; i < 4; ++i)
            for(int j = i; j < 4; ++j, ++k)
            {
                const bool keep  = tet(i) < tet(j);
                const int  left  = keep ? i : j;
                const int  right = keep ? j : i;
                CHECK(t_rows[10 * e + k] == tet(left));
                CHECK(t_cols[10 * e + k] == tet(right));
                for(int c = 0; c < 3; ++c)
                    for(int r = 0; r < 3; ++r)
                        CHECK(rel_err(t_values[(10 * e + k) * 9 + c * 3 + r],
                                      H[e](3 * left + r, 3 * right + c))
                              <= 1e-15);
            }
    }
}

namespace
{
// ---------------------------------------------------------------------------
//  the device kernels of test 6: the same per-element routines, instantiated for
//  S = lcrt::Var<double> inside a kernel (cookbook §7.4)
// ---------------------------------------------------------------------------
auto& arap_element_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<gg::LVector4i>,
                                   lcrt::Buffer<gg::LVector3>,
                                   lcrt::Buffer<gg::LMatrix3x3>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   double,
                                   uint32_t>(
            [](lcrt::BufferVar<gg::LVector4i>  tets,
               lcrt::BufferVar<gg::LVector3>   xs,
               lcrt::BufferVar<gg::LMatrix3x3> Dm_invs,
               lcrt::BufferVar<double>     kappas,
               lcrt::BufferVar<double>     volumes,
               lcrt::BufferVar<double>     out_energies,
               lcrt::BufferVar<double>     out_gradients,
               lcrt::BufferVar<double>     out_hessians,
               lcrt::Var<double>           dt,
               lcrt::Var<uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto tet    = gg::load_vec4i(tets->read(i));
                    auto Dm_inv = gg::load_mat3x3(Dm_invs->read(i));

                    auto x0 = gg::load_vec3(xs->read(tet(0)));
                    auto x1 = gg::load_vec3(xs->read(tet(1)));
                    auto x2 = gg::load_vec3(xs->read(tet(2)));
                    auto x3 = gg::load_vec3(xs->read(tet(3)));

                    auto kappa = kappas->read(i);
                    auto v     = volumes->read(i);

                    auto E = gg::arap_3d_detail::tet_energy<lcrt::Var<double>>(
                        kappa, dt, v, x0, x1, x2, x3, Dm_inv);
                    out_energies->write(i, E);

                    auto G = gg::arap_3d_detail::tet_gradient<lcrt::Var<double>>(
                        kappa, dt, v, x0, x1, x2, x3, Dm_inv);
                    for(int c = 0; c < 12; ++c)
                        out_gradients->write(i * 12u + c, G(c));

                    auto H = gg::arap_3d_detail::tet_hessian<lcrt::Var<double>>(
                        kappa, dt, v, x0, x1, x2, x3, Dm_inv);
                    for(int r = 0; r < 12; ++r)
                        for(int c = 0; c < 12; ++c)
                            out_hessians->write(i * 144u + r * 12u + c, H(r, c));
                };
            });
    return k;
}

auto& snh_element_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<gg::LVector4i>,
                                   lcrt::Buffer<gg::LVector3>,
                                   lcrt::Buffer<gg::LMatrix3x3>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   lcrt::Buffer<double>,
                                   double,
                                   uint32_t>(
            [](lcrt::BufferVar<gg::LVector4i>  tets,
               lcrt::BufferVar<gg::LVector3>   xs,
               lcrt::BufferVar<gg::LMatrix3x3> Dm_invs,
               lcrt::BufferVar<double>     mus,
               lcrt::BufferVar<double>     lambdas,
               lcrt::BufferVar<double>     volumes,
               lcrt::BufferVar<double>     out_energies,
               lcrt::BufferVar<double>     out_gradients,
               lcrt::BufferVar<double>     out_hessians,
               lcrt::Var<double>           dt,
               lcrt::Var<uint32_t>         n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto tet    = gg::load_vec4i(tets->read(i));
                    auto Dm_inv = gg::load_mat3x3(Dm_invs->read(i));

                    auto x0 = gg::load_vec3(xs->read(tet(0)));
                    auto x1 = gg::load_vec3(xs->read(tet(1)));
                    auto x2 = gg::load_vec3(xs->read(tet(2)));
                    auto x3 = gg::load_vec3(xs->read(tet(3)));

                    auto mu     = mus->read(i);
                    auto lambda = lambdas->read(i);
                    auto v      = volumes->read(i);

                    auto E = gg::snh_3d_detail::tet_energy<lcrt::Var<double>>(
                        mu, lambda, dt, v, x0, x1, x2, x3, Dm_inv);
                    out_energies->write(i, E);

                    auto G = gg::snh_3d_detail::tet_gradient<lcrt::Var<double>>(
                        mu, lambda, dt, v, x0, x1, x2, x3, Dm_inv);
                    for(int c = 0; c < 12; ++c)
                        out_gradients->write(i * 12u + c, G(c));

                    auto H = gg::snh_3d_detail::tet_hessian<lcrt::Var<double>>(
                        mu, lambda, dt, v, x0, x1, x2, x3, Dm_inv);
                    for(int r = 0; r < 12; ++r)
                        for(int c = 0; c < 12; ++c)
                            out_hessians->write(i * 144u + r * 12u + c, H(r, c));
                };
            });
    return k;
}

/// One batch of elements + its dense device outputs.
struct Batch
{
    SizeT n = 0;
    std::vector<Vector4i>  tets;
    std::vector<Vector3>   xs;      // 4 vertices per element
    std::vector<Matrix3x3> Dm_inv;  // per element
    std::vector<double>    volume;  // per element
    std::vector<double>    p0;      // kappa / mu
    std::vector<double>    p1;      // lambda (unused for ARAP)
    double                 dt = 1.0;
};

Batch make_batch(int count)
{
    Batch b;
    b.n = static_cast<SizeT>(count);
    for(int e = 0; e < count; ++e)
    {
        const TetDesc t = random_tet();
        b.tets.push_back(Vector4i{4 * e, 4 * e + 1, 4 * e + 2, 4 * e + 3});
        for(auto& x : tet_x(t))
            b.xs.push_back(x);
        b.Dm_inv.push_back(to_eigen(dm_inv(t)));
        b.volume.push_back(rest_volume(t));
        b.p0.push_back(rand_uniform(0.5, 2.0));
        b.p1.push_back(rand_uniform(1.0, 4.0));
    }
    return b;
}

double worst_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    double worst = 0.0;
    for(size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

/// Run one batch on the device, return (energies, gradients, hessians) flat.
struct BatchResult
{
    std::vector<double> energies;
    std::vector<double> gradients;
    std::vector<double> hessians;
};

BatchResult run_batch(const Batch& b, bool arap)
{
    auto tets_dev = upload<gg::LVector4i>(
        b.tets, [](const Vector4i& v) { return gg::to_luisa(v); });
    auto xs_dev = upload<gg::LVector3>(
        b.xs, [](const Vector3& v) { return gg::to_luisa(v); });
    auto dm_dev = upload<gg::LMatrix3x3>(
        b.Dm_inv, [](const Matrix3x3& v) { return gg::to_luisa(v); });
    auto p0_dev = upload(b.p0);
    auto p1_dev = upload(b.p1);
    auto v_dev  = upload(b.volume);

    lud::LDeviceBuffer<double> out_e(b.n);
    lud::LDeviceBuffer<double> out_g(b.n * 12);
    lud::LDeviceBuffer<double> out_h(b.n * 144);

    const auto n32 = static_cast<uint32_t>(b.n);
    if(arap)
    {
        auto& k = arap_element_kernel();
        gg::frame_stream().record(
            k(tets_dev.view(), xs_dev.view(), dm_dev.view(), p0_dev.view(), v_dev.view(),
              out_e.view(), out_g.view(), out_h.view(), b.dt, n32)
                .dispatch(n32));
    }
    else
    {
        auto& k = snh_element_kernel();
        gg::frame_stream().record(
            k(tets_dev.view(), xs_dev.view(), dm_dev.view(), p0_dev.view(), p1_dev.view(),
              v_dev.view(), out_e.view(), out_g.view(), out_h.view(), b.dt, n32)
                .dispatch(n32));
    }
    gg::frame_stream().synchronize();

    BatchResult r;
    r.energies  = read_device(out_e.cview());
    r.gradients = read_device(out_g.cview());
    r.hessians  = read_device(out_h.cview());
    return r;
}
}  // namespace

// ---------------------------------------------------------------------------
//  6) device vs host: the *same* generic element routines, S = double vs
//     S = lcrt::Var<double>
// ---------------------------------------------------------------------------
TEST_CASE("fem_constitution_a_device_vs_host", "[luisa][fem][constitution_a]")
{
    test_runtime();

    const int count = 16;
    const Batch b   = make_batch(count);

    // ---- ARAP --------------------------------------------------------------
    {
        const BatchResult dev = run_batch(b, /*arap=*/true);

        std::vector<double> host_e(b.n);
        std::vector<double> host_g(b.n * 12);
        std::vector<double> host_h(b.n * 144);
        for(SizeT e = 0; e < b.n; ++e)
        {
            std::array<Vector3, 4> x;
            for(int i = 0; i < 4; ++i)
                x[i] = b.xs[4 * e + i];
            const auto DmIn = lm(b.Dm_inv[e]);

            host_e[e] = arap_energy(x, DmIn, b.p0[e], b.dt, b.volume[e]);
            const Vector12 G = arap_gradient(x, DmIn, b.p0[e], b.dt, b.volume[e]);
            for(int i = 0; i < 12; ++i)
                host_g[e * 12 + i] = G(i);
            const Matrix12x12 H = arap_hessian(x, DmIn, b.p0[e], b.dt, b.volume[e]);
            for(int r = 0; r < 12; ++r)
                for(int c = 0; c < 12; ++c)
                    host_h[(e * 12 + r) * 12 + c] = H(r, c);
        }

        const double de = worst_abs_diff(dev.energies, host_e);
        const double dg = worst_abs_diff(dev.gradients, host_g);
        const double dh = worst_abs_diff(dev.hessians, host_h);
        std::printf("[t52a] ARAP device vs host: E %.3e  G %.3e  H %.3e\n", de, dg, dh);
        // The device and host paths run the same generic routine, but the DSL
        // instantiation lowers `sqrt`/`pow` to `lcrt::*` instead of `std::*`, so the
        // last bits may differ; the bound is T5.1's documented 1e-12 (measured below).
        CHECK(de <= 1e-12);
        CHECK(dg <= 1e-12);
        CHECK(dh <= 1e-12);
    }

    // ---- stable Neo-Hookean ------------------------------------------------
    {
        const BatchResult dev = run_batch(b, /*arap=*/false);

        std::vector<double> host_e(b.n);
        std::vector<double> host_g(b.n * 12);
        std::vector<double> host_h(b.n * 144);
        for(SizeT e = 0; e < b.n; ++e)
        {
            std::array<Vector3, 4> x;
            for(int i = 0; i < 4; ++i)
                x[i] = b.xs[4 * e + i];
            const auto DmIn = lm(b.Dm_inv[e]);

            host_e[e] = snh_energy(x, DmIn, b.p0[e], b.p1[e], b.dt, b.volume[e]);
            const Vector12 G = snh_gradient(x, DmIn, b.p0[e], b.p1[e], b.dt, b.volume[e]);
            for(int i = 0; i < 12; ++i)
                host_g[e * 12 + i] = G(i);
            const Matrix12x12 H =
                snh_hessian(x, DmIn, b.p0[e], b.p1[e], b.dt, b.volume[e]);
            for(int r = 0; r < 12; ++r)
                for(int c = 0; c < 12; ++c)
                    host_h[(e * 12 + r) * 12 + c] = H(r, c);
        }

        const double de = worst_abs_diff(dev.energies, host_e);
        const double dg = worst_abs_diff(dev.gradients, host_g);
        const double dh = worst_abs_diff(dev.hessians, host_h);
        std::printf("[t52a] SNK1 device vs host: E %.3e  G %.3e  H %.3e\n", de, dg, dh);
        CHECK(de <= 1e-12);
        CHECK(dg <= 1e-12);
        CHECK(dh <= 1e-12);
    }
}

// ---------------------------------------------------------------------------
//  7) a real scene through the public Engine/World API
//
//  The probe's `find<>` gives us the *real* ARAP3D / StableNeoHookean3D systems.
//  Their `do_init` has already read the scene attributes (per-tet kappa / mu /
//  lambda), so the test
//    * checks those values on the host AND in the uploaded device buffers,
//    * builds the constitution's own `ComputeEnergyInfo` /
//      `ComputeGradientHessianInfo` (public in the reference's class layout) over
//      *scratch* buffers, calls the real `do_compute_energy` /
//      `do_compute_gradient_hessian` -- i.e. the real kernels, on the real method
//      buffers (`indices`/`xs`/`Dm_invs`/`rest_volumes`),
//    * compares the results against the hand-written host model, including the
//      assembled doublet rows/values and triplet (row, col, block) triples, and
//    * exercises the `gradient_only` path (no triplet may be touched).
//
//  The single-element *energy* oracle is the same generic routine the kernel
//  calls, so this case verifies the plumbing (init/attribute extraction/slices/
//  view offsets/assembler) rather than re-deriving the math (section 1-3 does).
// ---------------------------------------------------------------------------
namespace
{
struct SliceScratch
{
    SizeT n = 0;
    lud::LDeviceBuffer<double>  energies;
    lud::LDeviceBuffer<int32_t> d_indices;
    lud::LDeviceBuffer<double>  d_values;
    lud::LDeviceBuffer<int32_t> t_rows;
    lud::LDeviceBuffer<int32_t> t_cols;
    lud::LDeviceBuffer<double>  t_values;
};

SliceScratch make_scratch(SizeT n)
{
    SliceScratch s;
    s.n = n;
    s.energies.resize(n);
    s.d_indices.resize(4 * n);
    s.d_values.resize(12 * n);
    s.t_rows.resize(10 * n);
    s.t_cols.resize(10 * n);
    s.t_values.resize(90 * n);
    // sentinels: a `gradient_only` call must not touch the triplet buffers
    const std::vector<int32_t> r(10 * n, -7);
    const std::vector<int32_t> c(10 * n, -7);
    const std::vector<double>  v(90 * n, -7.0);
    s.t_rows.copy_from(span<const int32_t>{r});
    s.t_cols.copy_from(span<const int32_t>{c});
    s.t_values.copy_from(span<const double>{v});
    return s;
}

/// The doublet/triplet views over the scratch buffers -- the layout the elastics
/// will hand to a constitution: its own slice at offset 0, global vertex indices as
/// block rows/cols (the reference's convention).
lla::LDoubletVectorView<Float, 3> doublet_view(SliceScratch& s)
{
    const auto n = static_cast<int32_t>(4 * s.n);
    return lla::LDoubletVectorView<Float, 3>{n, 0, n, n, 0, n, s.d_indices.view(),
                                             s.d_values.view()};
}

lla::LTripletMatrixView<Float, 3, 3> triplet_view(SliceScratch& s, int32_t vertex_count)
{
    const auto n = static_cast<int32_t>(10 * s.n);
    return lla::LTripletMatrixView<Float, 3, 3>{
        vertex_count,
        vertex_count,
        0,
        n,
        n,
        lla::make_index_pair(0, 0),
        lla::make_index_pair(vertex_count, vertex_count),
        s.t_rows.view(),
        s.t_cols.view(),
        s.t_values.view()};
}

std::vector<Vector3> read_vec3(const lud::LCBufferView<gg::LVector3>& view)
{
    std::vector<Vector3> host;
    for(auto& v : read_device(view))
        host.push_back(gg::to_eigen(v));
    return host;
}

std::vector<Vector4i> read_vec4i(const lud::LCBufferView<gg::LVector4i>& view)
{
    std::vector<Vector4i> host;
    for(auto& v : read_device(view))
        host.push_back(gg::to_eigen(v));
    return host;
}

std::vector<Matrix3x3> read_mat3(const lud::LCBufferView<gg::LMatrix3x3>& view)
{
    std::vector<Matrix3x3> host;
    for(auto& v : read_device(view))
        host.push_back(gg::to_eigen(v));
    return host;
}

/// The weak shapes of the `sym::*::tet_*` routines (a local alias so `run_pipeline`
/// can take the *function pointer*, not a re-implementation).
using EnergyFn = double (*)(double, double, double, const lps::LVec<double, 3>&,
                            const lps::LVec<double, 3>&, const lps::LVec<double, 3>&,
                            const lps::LVec<double, 3>&, const lps::LMat<double, 3, 3>&);

/// Wall-clock milliseconds (the first dispatch of a kernel also JIT-compiles it,
/// which is worth measuring: LC's nvrtc backend inlines the whole generic routine).
double now_ms()
{
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

template <typename Constitution, typename ElemFn>
void run_pipeline(Constitution&                   constitution,
                  gg::FEM3DConstitution*          self,
                  const char*                     what,
                  SizeT                           index_in_dim,
                  SizeT                           n,
                  const std::vector<Vector4i>&    tets,
                  double                          dt,
                  ElemFn&&                        elem)
{
    auto scratch = make_scratch(n);

    // ---- the constitution's own slice of the method's buffers ----------------
    gg::FEM3DConstitution::ComputeEnergyInfo energy_info{
        self, index_in_dim, dt, scratch.energies.view()};

    REQUIRE(energy_info.indices().size() == n);
    REQUIRE(energy_info.Dm_invs().size() == n);
    REQUIRE(energy_info.rest_volumes().size() == n);
    CHECK(energy_info.constitution_info().primitive_count == n);
    CHECK(energy_info.dt() == dt);
    CHECK(energy_info.xs().size() >= 4 * n);

    // ---- energy (the real kernel) -------------------------------------------
    const auto t0 = now_ms();
    constitution.do_compute_energy(energy_info);
    const auto t1 = now_ms();
    const auto energies = read_device(scratch.energies.cview());
    REQUIRE(energies.size() == n);

    // ---- gradient + Hessian (the real kernel, the real assemblers) ----------
    auto gradients = doublet_view(scratch);
    auto hessians  = triplet_view(scratch,
                                  static_cast<int32_t>(energy_info.xs().size()));

    gg::FEM3DConstitution::ComputeGradientHessianInfo gh_info{
        self, index_in_dim, false, dt, gradients, hessians};
    constitution.do_compute_gradient_hessian(gh_info);
    const auto t2 = now_ms();
    std::printf("[t52a] %s: n=%llu  energy dispatch %.0f ms  gradient+hessian dispatch"
                " %.0f ms (JIT included)\n",
                what,
                static_cast<unsigned long long>(n),
                t1 - t0,
                t2 - t1);

    const auto d_indices = read_device(scratch.d_indices.cview());
    const auto d_values  = read_device(scratch.d_values.cview());
    const auto t_rows    = read_device(scratch.t_rows.cview());
    const auto t_cols    = read_device(scratch.t_cols.cview());
    const auto t_values  = read_device(scratch.t_values.cview());

    // ---- the hand-written host model ---------------------------------------
    const AssembledModel oracle =
        build_model(n, tets, /*include_hessian=*/true, elem);

    double dev_e = 0.0, dev_g = 0.0, dev_h = 0.0;
    for(SizeT e = 0; e < n; ++e)
    {
        dev_e = std::max(dev_e, std::abs(energies[e] - oracle.energies[e]));
        for(int i = 0; i < 4; ++i)
        {
            CHECK(d_indices[4 * e + i] == oracle.doublet_indices[4 * e + i]);
            for(int c = 0; c < 3; ++c)
            {
                const double a = d_values[(4 * e + i) * 3 + c];
                const double b = oracle.doublet_values[(4 * e + i) * 3 + c];
                dev_g          = std::max(dev_g, std::abs(a - b));
                CHECK(rel_err(a, b) <= 1e-12);
            }
        }
        for(int k = 0; k < 10; ++k)
        {
            CHECK(t_rows[10 * e + k] == oracle.triplet_rows[10 * e + k]);
            CHECK(t_cols[10 * e + k] == oracle.triplet_cols[10 * e + k]);
            for(int i = 0; i < 9; ++i)
            {
                const double a = t_values[(10 * e + k) * 9 + i];
                const double b = oracle.triplet_values[(10 * e + k) * 9 + i];
                dev_h          = std::max(dev_h, std::abs(a - b));
                CHECK(rel_err(a, b) <= 1e-12);
            }
        }
    }
    std::printf("[t52a] %s device(constitution kernel) vs host(generic routine):"
                " E %.3e  G %.3e  H %.3e\n",
                what, dev_e, dev_g, dev_h);
    CHECK(rel_err(dev_e, 0.0) <= 1e-12);
    CHECK(rel_err(dev_g, 0.0) <= 1e-12);
    CHECK(rel_err(dev_h, 0.0) <= 1e-12);

    // ---- `gradient_only` must not touch the Hessian buffers -----------------
    {
        auto scratch2  = make_scratch(n);
        auto g_only    = doublet_view(scratch2);
        auto h_only    = triplet_view(scratch2,
                                      static_cast<int32_t>(energy_info.xs().size()));
        gg::FEM3DConstitution::ComputeGradientHessianInfo gh_only{
            self, index_in_dim, true, dt, g_only, h_only};
        constitution.do_compute_gradient_hessian(gh_only);

        const auto g = read_device(scratch2.d_values.cview());
        REQUIRE(g.size() == d_values.size());
        for(SizeT i = 0; i < g.size(); ++i)
            CHECK(rel_err(g[i], d_values[i]) <= 1e-15);
        for(auto v : read_device(scratch2.t_values.cview()))
            CHECK(v == -7.0);  // the sentinel survives
        for(auto v : read_device(scratch2.t_rows.cview()))
            CHECK(v == -7);
    }
}
}  // namespace

TEST_CASE("fem_constitution_a_scene", "[luisa][fem][constitution_a]")
{
    test_runtime();

    const auto desc = two_unit_tets();

    // =======================================================================
    //  A) StableNeoHookean (uid 10 -- unique in this test binary)
    // =======================================================================
    {
        auto d    = desc;
        d.params  = {1.7, 2.9};
        d.params2 = {3.3, 0.8};

        const auto workspace = workspace_path("uipc_luisa_fem_constitution_a_snh");
        Engine     engine{"luisa", workspace};
        World      world{engine};
        auto       scene = build_snh_scene(d);
        world.init(scene);
        REQUIRE(world.is_valid());

        REQUIRE(FEMConstitutionAProbe::snh != nullptr);
        auto& snh = *FEMConstitutionAProbe::snh;
        CHECK(snh.uid() == 10);
        CHECK(snh.dim() == 3);

        // ---- do_init read the scene's per-tet (mu, lambda) ------------------
        REQUIRE(snh.h_mus.size() == d.tets.size());
        REQUIRE(snh.h_lambdas.size() == d.tets.size());
        for(SizeT i = 0; i < d.tets.size(); ++i)
        {
            CHECK(rel_err(snh.h_mus[i], d.params[i]) <= 1e-15);
            CHECK(rel_err(snh.h_lambdas[i], d.params2[i]) <= 1e-15);
        }
        const auto dev_mus     = read_device(snh.mus.cview());
        const auto dev_lambdas = read_device(snh.lambdas.cview());
        REQUIRE(dev_mus.size() == d.tets.size());
        for(SizeT i = 0; i < d.tets.size(); ++i)
        {
            CHECK(rel_err(dev_mus[i], d.params[i]) <= 1e-15);
            CHECK(rel_err(dev_lambdas[i], d.params2[i]) <= 1e-15);
        }

        // ---- the method's own buffers (read through the constitution) --------
        gg::FEM3DConstitution::ComputeEnergyInfo probe_info{&snh, 0, 1.0, {}};
        const auto tets  = read_vec4i(probe_info.indices());
        const auto xbars = read_vec3(probe_info.x_bars());
        const auto xs    = read_vec3(probe_info.xs());
        const auto DmInv = read_mat3(probe_info.Dm_invs());
        const auto vols  = read_device(probe_info.rest_volumes());
        REQUIRE(tets.size() == d.tets.size());
        REQUIRE(DmInv.size() == d.tets.size());
        REQUIRE(vols.size() == d.tets.size());
        for(SizeT e = 0; e < d.tets.size(); ++e)
        {
            for(int c = 0; c < 4; ++c)
                CHECK(tets[e](c) == d.tets[e](c));
            // the two scene tets are unit tets: V = 1/6 and Dm^-1 = I
            CHECK(rel_err(vols[e], 1.0 / 6.0) <= 1e-12);
            for(int c = 0; c < 3; ++c)
                for(int r = 0; r < 3; ++r)
                    CHECK(std::abs(DmInv[e](r, c) - (r == c ? 1.0 : 0.0)) <= 1e-12);
        }
        // no dynamics in T5.2: xs == x_bars (the scene's rest state)
        REQUIRE(xs.size() == xbars.size());
        for(SizeT i = 0; i < xs.size(); ++i)
            CHECK(rel_err((xs[i] - xbars[i]).norm(), 0.0) <= 1e-12);

        // ---- drive the real kernels -----------------------------------------
        auto elem = [&](SizeT e, const Vector4i& tet) -> ElementResult
        {
            std::array<Vector3, 4> x;
            for(int i = 0; i < 4; ++i)
                x[i] = xs[static_cast<SizeT>(tet(i))];
            ElementResult r;
            r.energy   = snh_energy(x, lm(DmInv[e]), dev_mus[e], dev_lambdas[e], 1.0,
                                    vols[e]);
            r.gradient = lv12(snh_gradient(x, lm(DmInv[e]), dev_mus[e], dev_lambdas[e],
                                           1.0, vols[e]));
            r.hessian  = lm12(snh_hessian(x, lm(DmInv[e]), dev_mus[e], dev_lambdas[e],
                                          1.0, vols[e]));
            return r;
        };

        run_pipeline(snh, &snh, "StableNeoHookean3D", 0, d.tets.size(), tets, 1.0, elem);
    }

    // =======================================================================
    //  B) ARAP (uid 9)
    //
    //  `luisa_fem_base_test.cpp` registers a *test* dim-3 constitution with uid 9
    //  (T51FEM3DConstitution).  The method hands a (dim, uid) geometry group to the
    //  first system registered with that uid, so this case reports which one got it
    //  instead of silently asserting the wrong slice.
    // =======================================================================
    {
        auto d   = desc;
        d.params = {1.3, 2.7};
        d.params2 = {0.0, 0.0};

        const auto workspace = workspace_path("uipc_luisa_fem_constitution_a_arap");
        Engine     engine{"luisa", workspace};
        World      world{engine};
        auto       scene = build_arap_scene(d);
        world.init(scene);
        REQUIRE(world.is_valid());

        REQUIRE(FEMConstitutionAProbe::arap != nullptr);
        auto& arap = *FEMConstitutionAProbe::arap;
        CHECK(arap.uid() == 9);
        CHECK(arap.dim() == 3);

        gg::FEM3DConstitution::ComputeEnergyInfo probe_info{&arap, 0, 1.0, {}};
        // Among two dim-3 constitutions with the *same* uid the method hands the
        // geometry group to the first registered one (index 0), so index 0 is the
        // group that owns the geometry; whether that is *this* system is visible in
        // `do_init`'s own output (`h_kappas` is sized from the constitution's own
        // group).
        const auto n = d.tets.size();
        if(arap.h_kappas.size() != n)
        {
            // the (3, 9) group belongs to the other test file's probe constitution
            // (see the class comment); the ARAP-specific end-to-end checks cannot run
            WARN("ARAP3D did not own the (dim 3, uid 9) geometry group in this binary: "
                 "luisa_fem_base_test.cpp's T51FEM3DConstitution registered uid 9 first");
        }
        else
        {
            // ---- do_init read the scene's per-tet kappa ---------------------
            for(SizeT i = 0; i < n; ++i)
                CHECK(rel_err(arap.h_kappas[i], d.params[i]) <= 1e-15);
            const auto dev_kappas = read_device(arap.kappas.cview());
            REQUIRE(dev_kappas.size() == n);
            for(SizeT i = 0; i < n; ++i)
                CHECK(rel_err(dev_kappas[i], d.params[i]) <= 1e-15);
            const auto tets  = read_vec4i(probe_info.indices());
            const auto xs    = read_vec3(probe_info.xs());
            const auto xbars = read_vec3(probe_info.x_bars());
            const auto DmInv = read_mat3(probe_info.Dm_invs());
            const auto vols  = read_device(probe_info.rest_volumes());
            REQUIRE(tets.size() == n);
            REQUIRE(DmInv.size() == n);
            REQUIRE(vols.size() == n);
            REQUIRE(DmInv.size() == n);
            REQUIRE(vols.size() == n);
            for(SizeT e = 0; e < n; ++e)
            {
                for(int c = 0; c < 4; ++c)
                    CHECK(tets[e](c) == d.tets[e](c));
                CHECK(rel_err(vols[e], 1.0 / 6.0) <= 1e-12);
                for(int c = 0; c < 3; ++c)
                    for(int r = 0; r < 3; ++r)
                        CHECK(std::abs(DmInv[e](r, c) - (r == c ? 1.0 : 0.0)) <= 1e-12);
            }
            REQUIRE(xs.size() == xbars.size());
            for(SizeT i = 0; i < xs.size(); ++i)
                CHECK(rel_err((xs[i] - xbars[i]).norm(), 0.0) <= 1e-12);

            auto elem = [&](SizeT e, const Vector4i& tet) -> ElementResult
            {
                std::array<Vector3, 4> x;
                for(int i = 0; i < 4; ++i)
                    x[i] = xs[static_cast<SizeT>(tet(i))];
                ElementResult r;
                r.energy   = arap_energy(x, lm(DmInv[e]), dev_kappas[e], 1.0, vols[e]);
                r.gradient = lv12(arap_gradient(x, lm(DmInv[e]), dev_kappas[e], 1.0,
                                                vols[e]));
                r.hessian  = lm12(arap_hessian(x, lm(DmInv[e]), dev_kappas[e], 1.0,
                                               vols[e]));
                return r;
            };

            run_pipeline(arap, &arap, "ARAP3D", 0, n, tets, 1.0, elem);
        }
    }
}