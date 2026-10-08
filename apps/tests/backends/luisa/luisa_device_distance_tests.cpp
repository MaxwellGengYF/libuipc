/********************************************************************
 * @file luisa_device_distance_tests.cpp
 * @brief T2.3 — tests for the ported distance / CCD primitives
 *        (`src/backends/luisa/luisa_device/math/distance/`).
 *
 * Layers, mirroring the plan's acceptance criteria:
 *
 *  1. **Reference equivalence (host).**  The reference
 *     `src/backends/cuda/utils/distance/details/*.inl` are pure Eigen and compile
 *     on the host, so they are included directly into this TU and compared against
 *     the port instantiated with `T = double` over >= 10 000 random configurations
 *     per primitive — including near-degenerate ones (coincident / colinear points,
 *     zero-length edges, nearly parallel edges).  Two extra oracles are added: a
 *     central finite difference of the ported `_distance2` for the gradient, and
 *     one of the ported gradient for the Hessian.
 *     The reference `distance_flagged.h` is *not* includable here (it pulls
 *     `<cuda_tool/cuda_tool.h>` and `src/backends/cuda` is not on this target's
 *     include path); the flag-dispatched primitives are validated against the
 *     reference's own per-dim layouts (`kPeFromPt`/`kPeFromEe`, transcribed into
 *     `flagged.h`) plus the reference-verified canonical primitives, and against
 *     hand-derived flag patterns.
 *     The four `*_ccd` loops are compared against a *direct host transcription of
 *     the reference's `while(true)` + early-return loop* (same inputs, same
 *     `max_iter`) — that is exactly what the predicated port must reproduce.
 *  2. **Flag / CCD behaviour (host).**
 *  3. **Device smoke.**  ONE generic routine is instantiated twice — `S = double`
 *     (host reference) and `S = lcrt::Var<double>` (inside a Kernel1D) — dumping
 *     every primitive's outputs into a buffer; the two must agree to 1e-8 relative
 *     (T2.2's bound: same algorithm, NVRTC vs cl.exe FMA contraction).
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/math/distance/distance.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <vector>

namespace lmath = uipc::backend::luisa::luisa_device::math;
namespace dist  = uipc::backend::luisa::luisa_device::math::distance;
namespace lcrt  = ::luisa::compute;

using lmath::LMat;
using lmath::LVec;

// ---------------------------------------------------------------------------
//  Reference implementation (in-tree only)
// ---------------------------------------------------------------------------
// Eigen is only available in the in-tree build, and the standalone harness must
// still compile — hence the guard on <Eigen/Core> *and* the standalone define.
#if defined(__has_include) && __has_include(<Eigen/Core>) && !defined(UIPC_LUISA_TESTS_STANDALONE)
#define UIPC_GENERIC
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <uipc/common/type_define.h>

// The reference `details/ccd.inl` uses the `Eigen::Array3<T>` / `Array4<T>`
// spellings, which Eigen 3.4 does not provide.
namespace Eigen
{
template <class T>
using Array3 = Array<T, 3, 1>;
template <class T>
using Array4 = Array<T, 4, 1>;
}// namespace Eigen

// The reference splits declarations (wrapper .h) from definitions (.inl); pulling in
// the `.inl` files directly — as the T2.3 brief asks — means the one forward
// reference inside `edge_edge_mollifier.inl` (`need_mollify` calls
// `edge_edge_cross_norm2`, which the reference wrapper declares beforehand) has to
// be declared here, exactly as the reference wrapper does.
namespace uipc::backend::cuda::distance
{
template <typename T>
UIPC_GENERIC void edge_edge_cross_norm2(const Eigen::Vector<T, 3>& ea0,
                                        const Eigen::Vector<T, 3>& ea1,
                                        const Eigen::Vector<T, 3>& eb0,
                                        const Eigen::Vector<T, 3>& eb1,
                                        T&                         result);
}// namespace uipc::backend::cuda::distance

#include "../../../../src/backends/cuda/utils/distance/details/point_point.inl"
#include "../../../../src/backends/cuda/utils/distance/details/point_edge.inl"
#include "../../../../src/backends/cuda/utils/distance/details/point_triangle.inl"
#include "../../../../src/backends/cuda/utils/distance/details/edge_edge.inl"
#include "../../../../src/backends/cuda/utils/distance/details/edge_edge_mollifier.inl"
#include "../../../../src/backends/cuda/utils/distance/details/ccd.inl"

namespace ref = uipc::backend::cuda::distance;
#define UIPC_HAVE_DISTANCE_REFERENCE 1
#endif

namespace
{
// ---------------------------------------------------------------------------
//  Small test utilities
// ---------------------------------------------------------------------------
struct Lcg
{
    uint64_t s = 0x2f1e3d4c5b6a7981ULL;
    double   next() noexcept
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11) & 0x1fffffULL) / 1048576.0 - 1.0;  // [-1, 1)
    }
    double unit() noexcept { return 0.5 + 0.5 * next(); }  // (0, 1)
};

/// T2.2's comparison: relative with an absolute floor at 1.  `NaN`/`inf` compare
/// equal to themselves (a degenerate configuration legitimately produces the same
/// non-finite value on both sides).
bool close(double a, double b, double tol)
{
    if(std::isnan(a) || std::isnan(b))
        return std::isnan(a) && std::isnan(b);
    if(std::isinf(a) || std::isinf(b))
        return a == b;
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

/// |a - b| with the non-finite convention above; a mismatch of finiteness is
/// reported as `inf`.
double diff(double a, double b)
{
    if(!std::isfinite(a) || !std::isfinite(b))
        return close(a, b, 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
    return std::abs(a - b);
}

double max_abs(const double* a, int n)
{
    double m = 0.0;
    for(int i = 0; i < n; ++i)
        if(std::isfinite(a[i]))
            m = std::max(m, std::abs(a[i]));
    return m;
}

double max_abs_diff(const double* a, const double* b, int n)
{
    double worst = 0.0;
    for(int i = 0; i < n; ++i) worst = std::max(worst, diff(a[i], b[i]));
    return worst;
}

template <typename T, int N>
inline void to_flat(const LVec<T, N>& v, double* out)
{
    for(int i = 0; i < N; ++i) out[i] = static_cast<double>(v(i));
}

template <typename T, int R, int C>
inline void to_flat(const LMat<T, R, C>& m, double* out)
{
    int k = 0;
    for(int c = 0; c < C; ++c)
        for(int r = 0; r < R; ++r) out[k++] = static_cast<double>(m(r, c));
}

template <typename T, int N>
inline LVec<T, N> from_flat(const double* in)
{
    LVec<T, N> v;
    for(int i = 0; i < N; ++i) v(i) = static_cast<T>(in[i]);
    return v;
}

inline double dot3(const LVec<double, 3>& a, const LVec<double, 3>& b) noexcept
{
    return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
}

/// A random vector in `[-scale, scale)` per component.
inline LVec<double, 3> rnd_vec(Lcg& rng, double scale = 1.0)
{
    return LVec<double, 3>{scale * rng.next(), scale * rng.next(), scale * rng.next()};
}

/// Random configurations per primitive (the plan's `>= 10 000`).
constexpr int kRandomIters = 10000;

#if defined(UIPC_HAVE_DISTANCE_REFERENCE)
inline uipc::Vector3 to_eigen(const LVec<double, 3>& v) noexcept
{
    return uipc::Vector3{v(0), v(1), v(2)};
}

/// Flatten an Eigen vector / matrix (column-major, like `to_flat`).
template <typename E>
inline void eigen_flat(const E& e, double* out)
{
    for(int i = 0; i < static_cast<int>(e.size()); ++i) out[i] = e(i);
}
#endif
}// namespace

// ===========================================================================
//  1. Reference equivalence (host): point-point / point-edge / point-triangle /
//     edge-edge, with gradient, Hessian and finite-difference oracles
// ===========================================================================
#if defined(UIPC_HAVE_DISTANCE_REFERENCE)
namespace
{
/// One primitive family, described by callbacks so that the driver (and its
/// reporting) is written once for all four.
struct Family
{
    const char* name = "";
    int         npoints = 0;
    int         gsize = 0;
    int         hsize = 0;
    std::function<void(int, Lcg&, std::vector<LVec<double, 3>>&)>    gen;
    std::function<bool(const std::vector<LVec<double, 3>>&)>         well_scaled;
    std::function<void(const std::vector<LVec<double, 3>>&, double&)> port_d2;
    std::function<void(const std::vector<LVec<double, 3>>&, double*)> port_grad, port_hess;
    std::function<void(const std::vector<LVec<double, 3>>&, double&)> ref_d2;
    std::function<void(const std::vector<LVec<double, 3>>&, double*)> ref_grad, ref_hess;
};

/// Central finite difference of a `_distance2` -> gradient oracle.
void fd_gradient(const Family& f, const std::vector<LVec<double, 3>>& X, const double* analytic, double h, double& worst)
{
    auto      Xp = X, Xm = X;
    const int n = 3 * f.npoints;
    for(int j = 0; j < n; ++j)
    {
        const int k = j / 3, c = j % 3;
        Xp[k](c) += h;
        Xm[k](c) -= h;
        double dp = 0.0, dm = 0.0;
        f.port_d2(Xp, dp);
        f.port_d2(Xm, dm);
        Xp[k](c) -= h;
        Xm[k](c) += h;
        const double fd = (dp - dm) / (2.0 * h);
        worst           = std::max(worst, std::abs(fd - analytic[j]) / std::max(1.0, std::abs(fd)));
    }
}

/// Central finite difference of a `_gradient` -> Hessian oracle.
void fd_hessian(const Family& f, const std::vector<LVec<double, 3>>& X, const double* analytic, double h, double& worst)
{
    auto      Xp = X, Xm = X;
    const int n = 3 * f.npoints;
    std::vector<double> gp(static_cast<std::size_t>(n)), gm(static_cast<std::size_t>(n));
    for(int j = 0; j < n; ++j)
    {
        const int k = j / 3, c = j % 3;
        Xp[k](c) += h;
        Xm[k](c) -= h;
        f.port_grad(Xp, gp.data());
        f.port_grad(Xm, gm.data());
        Xp[k](c) -= h;
        Xm[k](c) += h;
        for(int r = 0; r < n; ++r)  // d g_r / d x_j  ==  H(r, j) (column-major flat)
        {
            const double fd = (gp[static_cast<std::size_t>(r)] - gm[static_cast<std::size_t>(r)]) / (2.0 * h);
            worst           = std::max(worst,
                                       std::abs(fd - analytic[static_cast<std::size_t>(j) * n + r])
                                           / std::max(1.0, std::abs(fd)));
        }
    }
}

void run_family(const Family& f)
{
    Lcg                         rng;
    std::vector<LVec<double, 3>> X(static_cast<std::size_t>(f.npoints));
    std::vector<double> gp(static_cast<std::size_t>(f.gsize)), hp(static_cast<std::size_t>(f.hsize));
    std::vector<double> gr(static_cast<std::size_t>(f.gsize)), hr(static_cast<std::size_t>(f.hsize));

    double worst[8] = {};  // d2, g, h (absolute) | d2, g, h (relative) | fd_g, fd_h
    for(int it = 0; it < kRandomIters; ++it)
    {
        f.gen(it, rng, X);

        double d2p = 0.0, d2r = 0.0;
        f.port_d2(X, d2p);
        f.ref_d2(X, d2r);
        worst[0] = std::max(worst[0], diff(d2p, d2r));
        worst[3] = std::max(worst[3], diff(d2p, d2r) / std::max(1.0, std::abs(d2r)));

        f.port_grad(X, gp.data());
        f.ref_grad(X, gr.data());
        worst[1] = std::max(worst[1], max_abs_diff(gp.data(), gr.data(), f.gsize));
        worst[4] = std::max(worst[4], max_abs_diff(gp.data(), gr.data(), f.gsize) / std::max(1.0, max_abs(gr.data(), f.gsize)));

        f.port_hess(X, hp.data());
        f.ref_hess(X, hr.data());
        worst[2] = std::max(worst[2], max_abs_diff(hp.data(), hr.data(), f.hsize));
        worst[5] = std::max(worst[5], max_abs_diff(hp.data(), hr.data(), f.hsize) / std::max(1.0, max_abs(hr.data(), f.hsize)));

        if(f.well_scaled(X))
        {
            fd_gradient(f, X, gp.data(), 1e-6, worst[6]);
            fd_hessian(f, X, hp.data(), 1e-6, worst[7]);
        }
    }

    std::printf("    [%s] reference: dist2 |e| %.3e (rel %.3e) | grad |e| %.3e (rel %.3e) | "
                "hessian |e| %.3e (rel %.3e)\n",
                f.name,
                worst[0],
                worst[3],
                worst[1],
                worst[4],
                worst[2],
                worst[5]);
    std::printf("    [%s] finite difference: grad rel %.3e | hessian rel %.3e\n", f.name, worst[6], worst[7]);

    CHECK(worst[0] <= 1e-5);  // dist2, absolute
    CHECK(worst[1] <= 1e-5);  // gradient, absolute
    CHECK(worst[2] <= 1e-5);  // Hessian, absolute
    CHECK(worst[5] <= 1e-5);  // Hessian, relative
    CHECK(worst[6] <= 1e-5);  // gradient vs FD
    CHECK(worst[7] <= 1e-5);  // Hessian vs FD
}

// --- the four families ------------------------------------------------------
Family make_point_point()
{
    Family f;
    f.name    = "point_point";
    f.npoints = 2;
    f.gsize   = 6;
    f.hsize   = 36;
    f.gen     = [](int it, Lcg& rng, std::vector<LVec<double, 3>>& X)
    {
        X[0] = rnd_vec(rng);
        X[1] = X[0] + rnd_vec(rng);
        if(it % 5 == 0)
            X[1] = X[0];                             // coincident points
        else if(it % 5 == 1)
            X[1] = X[0] + rnd_vec(rng, 1e-13);       // nearly coincident
    };
    f.well_scaled = [](const std::vector<LVec<double, 3>>& X)
    { return lmath::norm(X[1] - X[0]) > 0.25; };
    f.port_d2 = [](const std::vector<LVec<double, 3>>& X, double& d)
    { dist::point_point_distance2(X[0], X[1], d); };
    f.port_grad = [](const std::vector<LVec<double, 3>>& X, double* g)
    {
        LVec<double, 6> gg;
        dist::point_point_distance2_gradient(X[0], X[1], gg);
        to_flat(gg, g);
    };
    f.port_hess = [](const std::vector<LVec<double, 3>>& X, double* h)
    {
        LMat<double, 6, 6> hh;
        dist::point_point_distance2_hessian(X[0], X[1], hh);
        to_flat(hh, h);
    };
    f.ref_d2 = [](const std::vector<LVec<double, 3>>& X, double& d)
    { ref::point_point_distance2(to_eigen(X[0]), to_eigen(X[1]), d); };
    f.ref_grad = [](const std::vector<LVec<double, 3>>& X, double* g)
    {
        Eigen::Matrix<double, 6, 1> gg;
        ref::point_point_distance2_gradient(to_eigen(X[0]), to_eigen(X[1]), gg);
        eigen_flat(gg, g);
    };
    f.ref_hess = [](const std::vector<LVec<double, 3>>& X, double* h)
    {
        Eigen::Matrix<double, 6, 6> hh;
        ref::point_point_distance2_hessian(to_eigen(X[0]), to_eigen(X[1]), hh);
        eigen_flat(hh, h);
    };
    return f;
}

Family make_point_edge()
{
    Family f;
    f.name    = "point_edge";
    f.npoints = 3;
    f.gsize   = 9;
    f.hsize   = 81;
    f.gen     = [](int it, Lcg& rng, std::vector<LVec<double, 3>>& X)
    {
        X[0] = rnd_vec(rng);                 // p
        X[1] = rnd_vec(rng);                 // e0
        X[2] = X[1] + rnd_vec(rng);          // e1
        if(it % 7 == 0)
            X[2] = X[1];                                                 // zero-length edge
        else if(it % 7 == 1)
            X[2] = X[1] + rnd_vec(rng);                                  // (rewritten below)
        else if(it % 7 == 2)
            X[0] = X[1];                                                 // p == e0
        else if(it % 7 == 3)
            X[2] = X[1] + rnd_vec(rng, 1e-12);                            // nearly zero-length
        if(it % 7 == 1)  // colinear with the edge
        {
            const double t = 2.0 * rng.unit() - 0.5;
            X[0]           = X[1] + (X[2] - X[1]) * t;
        }
    };
    f.well_scaled = [](const std::vector<LVec<double, 3>>& X)
    {
        const LVec<double, 3> e = X[2] - X[1];
        const LVec<double, 3> c = lmath::cross(e, X[0] - X[1]);
        return lmath::norm(e) > 0.25 && lmath::norm(c) > 0.05;
    };
    f.port_d2 = [](const std::vector<LVec<double, 3>>& X, double& d)
    { dist::point_edge_distance2(X[0], X[1], X[2], d); };
    f.port_grad = [](const std::vector<LVec<double, 3>>& X, double* g)
    {
        LVec<double, 9> gg;
        dist::point_edge_distance2_gradient(X[0], X[1], X[2], gg);
        to_flat(gg, g);
    };
    f.port_hess = [](const std::vector<LVec<double, 3>>& X, double* h)
    {
        LMat<double, 9, 9> hh;
        dist::point_edge_distance2_hessian(X[0], X[1], X[2], hh);
        to_flat(hh, h);
    };
    f.ref_d2 = [](const std::vector<LVec<double, 3>>& X, double& d)
    { ref::point_edge_distance2(to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), d); };
    f.ref_grad = [](const std::vector<LVec<double, 3>>& X, double* g)
    {
        Eigen::Matrix<double, 9, 1> gg;
        ref::point_edge_distance2_gradient(to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), gg);
        eigen_flat(gg, g);
    };
    f.ref_hess = [](const std::vector<LVec<double, 3>>& X, double* h)
    {
        Eigen::Matrix<double, 9, 9> hh;
        ref::point_edge_distance2_hessian(to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), hh);
        eigen_flat(hh, h);
    };
    return f;
}

Family make_point_triangle()
{
    Family f;
    f.name    = "point_triangle";
    f.npoints = 4;
    f.gsize   = 12;
    f.hsize   = 144;
    f.gen     = [](int it, Lcg& rng, std::vector<LVec<double, 3>>& X)
    {
        X[0] = rnd_vec(rng);                 // p
        X[1] = rnd_vec(rng);                 // t0
        X[2] = X[1] + rnd_vec(rng);          // t1
        X[3] = X[1] + rnd_vec(rng);          // t2
        if(it % 9 == 0)
            X[3] = X[1];                                    // degenerate triangle
        else if(it % 9 == 1)                                // colinear
            X[3] = X[1] + (X[2] - X[1]) * (2.0 * rng.unit() - 0.5);
        else if(it % 9 == 2)
            X[0] = X[1];                                    // p == t0
        else if(it % 9 == 3)                                // tiny triangle
        {
            X[2] = X[1] + rnd_vec(rng, 1e-11);
            X[3] = X[1] + rnd_vec(rng, 1e-11);
        }
        else if(it % 9 == 4)
            X[0] = X[1] + 1e-12 * rnd_vec(rng);              // p nearly on t0
    };
    f.well_scaled = [](const std::vector<LVec<double, 3>>& X)
    {
        const LVec<double, 3> n = lmath::cross(X[2] - X[1], X[3] - X[1]);
        const double          a = lmath::norm(n);
        return a > 0.25 && std::abs(dot3(X[0] - X[1], n)) / a > 0.25;
    };
    f.port_d2 = [](const std::vector<LVec<double, 3>>& X, double& d)
    { dist::point_triangle_distance2(X[0], X[1], X[2], X[3], d); };
    f.port_grad = [](const std::vector<LVec<double, 3>>& X, double* g)
    {
        LVec<double, 12> gg;
        dist::point_triangle_distance2_gradient(X[0], X[1], X[2], X[3], gg);
        to_flat(gg, g);
    };
    f.port_hess = [](const std::vector<LVec<double, 3>>& X, double* h)
    {
        LMat<double, 12, 12> hh;
        dist::point_triangle_distance2_hessian(X[0], X[1], X[2], X[3], hh);
        to_flat(hh, h);
    };
    f.ref_d2 = [](const std::vector<LVec<double, 3>>& X, double& d)
    {
        ref::point_triangle_distance2(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), d);
    };
    f.ref_grad = [](const std::vector<LVec<double, 3>>& X, double* g)
    {
        Eigen::Matrix<double, 12, 1> gg;
        ref::point_triangle_distance2_gradient(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), gg);
        eigen_flat(gg, g);
    };
    f.ref_hess = [](const std::vector<LVec<double, 3>>& X, double* h)
    {
        Eigen::Matrix<double, 12, 12> hh;
        ref::point_triangle_distance2_hessian(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), hh);
        eigen_flat(hh, h);
    };
    return f;
}

Family make_edge_edge()
{
    Family f;
    f.name    = "edge_edge";
    f.npoints = 4;
    f.gsize   = 12;
    f.hsize   = 144;
    f.gen     = [](int it, Lcg& rng, std::vector<LVec<double, 3>>& X)
    {
        X[0] = rnd_vec(rng);                 // ea0
        X[1] = X[0] + rnd_vec(rng);          // ea1
        X[2] = rnd_vec(rng);                 // eb0
        X[3] = X[2] + rnd_vec(rng);          // eb1
        const LVec<double, 3> u = X[1] - X[0];
        if(it % 9 == 0)
            X[3] = X[2] + u * (2.0 * rng.unit() - 0.5);   // parallel
        else if(it % 9 == 1)
            X[3] = X[2] + u * (2.0 * rng.unit() - 0.5) + rnd_vec(rng, 1e-13);  // nearly
        else if(it % 9 == 2)
            X[1] = X[0];                                  // zero-length a
        else if(it % 9 == 3)
            X[3] = X[2];                                  // zero-length b
        else if(it % 9 == 4)
        {
            X[1] = X[0];                                  // both degenerate: PP
            X[3] = X[2];
        }
        else if(it % 9 == 5)
            X[3] = X[2] + u * (2.0 * rng.unit() - 0.5) + rnd_vec(rng, 1e-9);  // near-parallel
    };
    f.well_scaled = [](const std::vector<LVec<double, 3>>& X)
    {
        const LVec<double, 3> u = X[1] - X[0];
        const LVec<double, 3> v = X[3] - X[2];
        return lmath::norm(u) > 0.25 && lmath::norm(v) > 0.25 && lmath::norm(lmath::cross(u, v)) > 0.05;
    };
    f.port_d2 = [](const std::vector<LVec<double, 3>>& X, double& d)
    { dist::edge_edge_distance2(X[0], X[1], X[2], X[3], d); };
    f.port_grad = [](const std::vector<LVec<double, 3>>& X, double* g)
    {
        LVec<double, 12> gg;
        dist::edge_edge_distance2_gradient(X[0], X[1], X[2], X[3], gg);
        to_flat(gg, g);
    };
    f.port_hess = [](const std::vector<LVec<double, 3>>& X, double* h)
    {
        LMat<double, 12, 12> hh;
        dist::edge_edge_distance2_hessian(X[0], X[1], X[2], X[3], hh);
        to_flat(hh, h);
    };
    f.ref_d2 = [](const std::vector<LVec<double, 3>>& X, double& d)
    {
        ref::edge_edge_distance2(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), d);
    };
    f.ref_grad = [](const std::vector<LVec<double, 3>>& X, double* g)
    {
        Eigen::Matrix<double, 12, 1> gg;
        ref::edge_edge_distance2_gradient(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), gg);
        eigen_flat(gg, g);
    };
    f.ref_hess = [](const std::vector<LVec<double, 3>>& X, double* h)
    {
        Eigen::Matrix<double, 12, 12> hh;
        ref::edge_edge_distance2_hessian(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), hh);
        eigen_flat(hh, h);
    };
    return f;
}
}// namespace

TEST_CASE("distance_host_simplex_reference", "[luisa][distance][reference]")
{
    run_family(make_point_point());
    run_family(make_point_edge());
    run_family(make_point_triangle());
    run_family(make_edge_edge());
}
#endif  // UIPC_HAVE_DISTANCE_REFERENCE

// ===========================================================================
//  2. Reference equivalence (host): the edge-edge mollifier
// ===========================================================================
#if defined(UIPC_HAVE_DISTANCE_REFERENCE)
namespace
{
void run_mollifier()
{
    Lcg rng;
    // worst errors:        threshold | need mismatches | cross_norm2 value/grad/hess
    //                      | mollifier value/grad/hess | FD grad/hess
    double worst[10] = {};
    int    true_count = 0, false_count = 0;

    std::vector<LVec<double, 3>> X(4);
    double gp[12], hp[144], gr[12], hr[144];

    for(int it = 0; it < kRandomIters; ++it)
    {
        X[0] = rnd_vec(rng);                 // ea0
        X[1] = X[0] + rnd_vec(rng);          // ea1
        X[2] = rnd_vec(rng);                 // eb0
        X[3] = X[2] + rnd_vec(rng);          // eb1
        const LVec<double, 3> u = X[1] - X[0];
        if(it % 4 == 0)
            X[3] = X[2] + u * (2.0 * rng.unit() - 0.5);                        // parallel
        else if(it % 4 == 1)
            X[3] = X[2] + u * (2.0 * rng.unit() - 0.5) + rnd_vec(rng, 1e-12);  // nearly
        else if(it % 4 == 2)
            X[3] = X[2] + u * (2.0 * rng.unit() - 0.5) + rnd_vec(rng, 1e-6);

        // --- threshold (from the rest configuration) -------------------------
        const double coeff = 0.01 + rng.unit();
        double       thr_p = 0.0, thr_r = 0.0;
        dist::edge_edge_mollifier_threshold(X[0], X[1], X[2], X[3], coeff, thr_p);
        ref::edge_edge_mollifier_threshold(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), coeff, thr_r);
        worst[0] = std::max(worst[0], diff(thr_p, thr_r));

        // --- cross norm^2: value, gradient, Hessian ---------------------------
        double cn2_p = 0.0, cn2_r = 0.0;
        dist::edge_edge_cross_norm2(X[0], X[1], X[2], X[3], cn2_p);
        ref::edge_edge_cross_norm2(to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), cn2_r);
        worst[1] = std::max(worst[1], diff(cn2_p, cn2_r));

        LVec<double, 12>     g12;
        LMat<double, 12, 12> h12;
        Eigen::Matrix<double, 12, 1>  g12r;
        Eigen::Matrix<double, 12, 12> h12r;
        dist::edge_edge_cross_norm2_gradient(X[0], X[1], X[2], X[3], g12);
        dist::edge_edge_cross_norm2_hessian(X[0], X[1], X[2], X[3], h12);
        ref::edge_edge_cross_norm2_gradient(to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), g12r);
        ref::edge_edge_cross_norm2_hessian(to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), h12r);
        to_flat(g12, gp);
        to_flat(h12, hp);
        eigen_flat(g12r, gr);
        eigen_flat(h12r, hr);
        worst[2] = std::max(worst[2], max_abs_diff(gp, gr, 12));
        worst[3] = std::max(worst[3], max_abs_diff(hp, hr, 144));

        // --- the mollifier itself: a threshold chosen to straddle cn2 twice ----
        const double eps_x = cn2_p * (1.0 / 8.0 + 4.0 * rng.unit());

        const bool need_p = dist::need_mollify(X[0], X[1], X[2], X[3], eps_x);
        const bool need_r = ref::need_mollify(to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), eps_x);
        CHECK(need_p == need_r);

        double e_p = 0.0, e_r = 0.0;
        dist::edge_edge_mollifier(X[0], X[1], X[2], X[3], eps_x, e_p);
        ref::edge_edge_mollifier(to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), eps_x, e_r);
        worst[4] = std::max(worst[4], diff(e_p, e_r));

        dist::edge_edge_mollifier_gradient(X[0], X[1], X[2], X[3], eps_x, g12);
        dist::edge_edge_mollifier_hessian(X[0], X[1], X[2], X[3], eps_x, h12);
        ref::edge_edge_mollifier_gradient(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), eps_x, g12r);
        ref::edge_edge_mollifier_hessian(
            to_eigen(X[0]), to_eigen(X[1]), to_eigen(X[2]), to_eigen(X[3]), eps_x, h12r);
        to_flat(g12, gp);
        to_flat(h12, hp);
        eigen_flat(g12r, gr);
        eigen_flat(h12r, hr);
        worst[5] = std::max(worst[5], max_abs_diff(gp, gr, 12));
        worst[6] = std::max(worst[6], max_abs_diff(hp, hr, 144));

        if(need_p)
            ++true_count;
        else
            ++false_count;

        // Finite differences, only well inside one branch and well scaled (the
        // reference's mollifier is C0 but not C1 at cross^2 == eps_x by design, and
        // a near-parallel pair drives 1/eps_x up far enough that the difference
        // quotient is no longer meaningful).
        const LVec<double, 3> du = X[1] - X[0];
        const LVec<double, 3> dv = X[3] - X[2];
        const bool well_scaled = lmath::norm(du) > 0.25 && lmath::norm(dv) > 0.25
                                 && cn2_p > 0.01  // the cross must be well conditioned
                                 && std::abs(cn2_p - eps_x) > 0.25 * eps_x;
        if(well_scaled)
        {
            auto d2 = [&](const std::vector<LVec<double, 3>>& Y, double& out)
            { dist::edge_edge_mollifier(Y[0], Y[1], Y[2], Y[3], eps_x, out); };
            auto dg = [&](const std::vector<LVec<double, 3>>& Y, double* out)
            {
                LVec<double, 12> g;
                dist::edge_edge_mollifier_gradient(Y[0], Y[1], Y[2], Y[3], eps_x, g);
                to_flat(g, out);
            };
            auto Xc = X;
            for(int j = 0; j < 12; ++j)
            {
                const int k = j / 3, c = j % 3;
                auto      Xp = Xc, Xm = Xc;
                Xp[k](c) += 1e-6;
                Xm[k](c) -= 1e-6;
                double fp = 0.0, fm = 0.0;
                d2(Xp, fp);
                d2(Xm, fm);
                const double fd = (fp - fm) / 2e-6;
                worst[7] = std::max(worst[7], std::abs(fd - gp[j]) / std::max(1.0, std::abs(fd)));

                std::vector<double> gpp(12), gmm(12);
                dg(Xp, gpp.data());
                dg(Xm, gmm.data());
                for(int r = 0; r < 12; ++r)
                {
                    const double fd2 = (gpp[static_cast<std::size_t>(r)] - gmm[static_cast<std::size_t>(r)]) / 2e-6;
                    worst[8] = std::max(worst[8],
                                        std::abs(fd2 - hp[static_cast<std::size_t>(j) * 12 + r])
                                            / std::max(1.0, std::abs(fd2)));
                }
            }
        }
    }

    std::printf("    [mollifier] reference: threshold %.3e | cross_norm2 %.3e (grad %.3e, hessian %.3e) | "
                "mollifier %.3e (grad %.3e, hessian %.3e)\n",
                worst[0],
                worst[1],
                worst[2],
                worst[3],
                worst[4],
                worst[5],
                worst[6]);
    std::printf("    [mollifier] finite difference: grad rel %.3e | hessian rel %.3e | "
                "needs-mollify true/false = %d/%d\n",
                worst[7],
                worst[8],
                true_count,
                false_count);

    CHECK(worst[0] <= 1e-5);
    CHECK(worst[1] <= 1e-5);
    CHECK(worst[2] <= 1e-5);
    CHECK(worst[3] <= 1e-5);
    CHECK(worst[4] <= 1e-5);
    CHECK(worst[5] <= 1e-5);
    CHECK(worst[6] <= 1e-5);
    CHECK(worst[7] <= 1e-5);
    CHECK(worst[8] <= 1e-5);
    CHECK((true_count > 100 && false_count > 100));  // both branches are exercised
}
}// namespace

TEST_CASE("distance_host_mollifier_reference", "[luisa][distance][reference]")
{
    run_mollifier();
}

// ===========================================================================
//  3. Reference equivalence (host): the CCD broad phases
// ===========================================================================
namespace
{
void run_broad_phases()
{
    Lcg rng;
    int mismatches = 0;
    int hits       = 0;  // how often the filters report "still close"
    int checked    = 0;

    for(int it = 0; it < kRandomIters; ++it)
    {
        const LVec<double, 3> a = rnd_vec(rng), b = rnd_vec(rng), c = rnd_vec(rng), d = rnd_vec(rng);
        const LVec<double, 3> da = rnd_vec(rng, 0.5), db = rnd_vec(rng, 0.5), dc = rnd_vec(rng, 0.5), dd = rnd_vec(rng, 0.5);
        // a third of the cases are "touching" (the boxes overlap after the sweep)
        const bool tight = (it % 3 == 0);
        const LVec<double, 3> offset = tight ? rnd_vec(rng, 0.05) : rnd_vec(rng, 2.0);
        const double          dist   = tight ? 0.5 : 0.01;

        const LVec<double, 3> a2 = a + offset;
        const LVec<double, 3> b2 = b + offset;
        const LVec<double, 3> c2 = c + offset;
        const LVec<double, 3> d2 = d + offset;

        // point-edge (static) and point-triangle (static)
        {
            const bool p = dist::point_edge_cd_broadphase(a, b, c, dist);
            const bool r = ref::point_edge_cd_broadphase(to_eigen(a), to_eigen(b), to_eigen(c), dist);
            mismatches += (p != r) ? 1 : 0;
            hits += p ? 1 : 0;
            ++checked;
        }
        {
            const bool p = dist::point_triangle_cd_broadphase(a, b, c, d, dist);
            const bool r = ref::point_triangle_cd_broadphase(
                to_eigen(a), to_eigen(b), to_eigen(c), to_eigen(d), dist);
            mismatches += (p != r) ? 1 : 0;
            hits += p ? 1 : 0;
            ++checked;
        }
        {
            const bool p = dist::edge_edge_cd_broadphase(a, b, c, d, dist);
            const bool r = ref::edge_edge_cd_broadphase(
                to_eigen(a), to_eigen(b), to_eigen(c), to_eigen(d), dist);
            mismatches += (p != r) ? 1 : 0;
            hits += p ? 1 : 0;
            ++checked;
        }
        // swept variants
        {
            const bool p = dist::point_triangle_ccd_broadphase(
                a, b, c, d, da, db, dc, dd, dist);
            const bool r = ref::point_triangle_ccd_broadphase(to_eigen(a),
                                                              to_eigen(b),
                                                              to_eigen(c),
                                                              to_eigen(d),
                                                              to_eigen(da),
                                                              to_eigen(db),
                                                              to_eigen(dc),
                                                              to_eigen(dd),
                                                              dist);
            mismatches += (p != r) ? 1 : 0;
            hits += p ? 1 : 0;
            ++checked;
        }
        {
            const bool p = dist::edge_edge_ccd_broadphase(
                a, b, c, d, da, db, dc, dd, dist);
            const bool r = ref::edge_edge_ccd_broadphase(to_eigen(a),
                                                         to_eigen(b),
                                                         to_eigen(c),
                                                         to_eigen(d),
                                                         to_eigen(da),
                                                         to_eigen(db),
                                                         to_eigen(dc),
                                                         to_eigen(dd),
                                                         dist);
            mismatches += (p != r) ? 1 : 0;
            hits += p ? 1 : 0;
            ++checked;
        }
        {
            const bool p = dist::point_edge_ccd_broadphase(a, b, c, da, db, dc, dist);
            const bool r = ref::point_edge_ccd_broadphase(
                to_eigen(a), to_eigen(b), to_eigen(c), to_eigen(da), to_eigen(db), to_eigen(dc), dist);
            mismatches += (p != r) ? 1 : 0;
            hits += p ? 1 : 0;
            ++checked;
        }
        {
            const bool p = dist::point_point_ccd_broadphase(a2, b2, da, db, dist);
            const bool r = ref::point_point_ccd_broadphase(
                to_eigen(a2), to_eigen(b2), to_eigen(da), to_eigen(db), dist);
            mismatches += (p != r) ? 1 : 0;
            hits += p ? 1 : 0;
            ++checked;
        }
        // 2-D swept point-edge
        {
            const LVec<double, 2> p2{a(0), a(1)}, e0{b(0), b(1)}, e1{c(0), c(1)};
            const LVec<double, 2> dp2{da(0), da(1)}, de02{db(0), db(1)}, de12{dc(0), dc(1)};
            const bool            p = dist::point_edge_ccd_broadphase(p2, e0, e1, dp2, de02, de12, dist);
            const bool            r = ref::point_edge_ccd_broadphase(
                Eigen::Vector<double, 2>{p2(0), p2(1)},
                Eigen::Vector<double, 2>{e0(0), e0(1)},
                Eigen::Vector<double, 2>{e1(0), e1(1)},
                Eigen::Vector<double, 2>{dp2(0), dp2(1)},
                Eigen::Vector<double, 2>{de02(0), de02(1)},
                Eigen::Vector<double, 2>{de12(0), de12(1)},
                dist);
            mismatches += (p != r) ? 1 : 0;
            hits += p ? 1 : 0;
            ++checked;
        }
    }

    std::printf("    [ccd broad phases] %d comparisons, %d mismatches (%d 'close' results)\n",
                checked,
                mismatches,
                hits);
    CHECK(mismatches == 0);
    CHECK(hits > 0);
    CHECK((checked - hits) > 0);
}
}// namespace

TEST_CASE("distance_host_ccd_broadphase_reference", "[luisa][distance][reference]")
{
    run_broad_phases();
}
#endif  // UIPC_HAVE_DISTANCE_REFERENCE

// ===========================================================================
//  4. The four conservative-advancement (additive CCD) routines vs a direct
//     host transcription of the reference's loop
// ===========================================================================
#if defined(UIPC_HAVE_DISTANCE_REFERENCE)
namespace
{
// The reference's `*_ccd` bodies cannot be instantiated directly (they call the
// reference's flag constructors, which live in `distance_flagged.h` — not
// includable here, see the file header).  These are therefore *verbatim
// transcriptions of the reference's loop structure* (real `while(true)` with
// data-dependent early returns) over the port's flag-verified canonical
// primitives: exactly the control flow the predicated port has to reproduce.
bool ref_point_triangle_ccd(LVec<double, 3> p,
                            LVec<double, 3> t0,
                            LVec<double, 3> t1,
                            LVec<double, 3> t2,
                            LVec<double, 3> dp,
                            LVec<double, 3> dt0,
                            LVec<double, 3> dt1,
                            LVec<double, 3> dt2,
                            double          eta,
                            double          thickness,
                            int             max_iter,
                            double&         toc)
{
    auto mov = (dt0 + dt1 + dt2 + dp) / 4.0;
    dt0      = dt0 - mov;
    dt1      = dt1 - mov;
    dt2      = dt2 - mov;
    dp       = dp - mov;
    auto disp = lmath::make_vec<double, 3>([&](int i)
                                           {
                                               return i == 0   ? lmath::squaredNorm(dt0)
                                                      : i == 1 ? lmath::squaredNorm(dt1)
                                                               : lmath::squaredNorm(dt2);
                                           });
    const double maxDispMag = dp.norm() + std::sqrt(lmath::maxCoeff(disp));

    if(maxDispMag <= 0.0)
        return false;

    double dist2_cur = 0.0;
    auto   flag      = dist::point_triangle_distance_flag<double>(p, t0, t1, t2);
    dist::point_triangle_distance2(flag, p, t0, t1, t2, dist2_cur);
    double dist_cur = std::sqrt(dist2_cur);
    double gap      = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    double toc_prev = toc;
    toc             = 0.0;
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }
        const double tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness)
                                     / ((dist_cur + thickness) * maxDispMag);
        p  = p + tocLowerBound * dp;
        t0 = t0 + tocLowerBound * dt0;
        t1 = t1 + tocLowerBound * dt1;
        t2 = t2 + tocLowerBound * dt2;
        flag = dist::point_triangle_distance_flag<double>(p, t0, t1, t2);
        dist::point_triangle_distance2(flag, p, t0, t1, t2, dist2_cur);
        dist_cur = std::sqrt(dist2_cur);
        if(toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
            break;
        toc += tocLowerBound;
        if(toc > toc_prev)
            return false;
    }
    return true;
}

bool ref_edge_edge_ccd(LVec<double, 3> ea0,
                       LVec<double, 3> ea1,
                       LVec<double, 3> eb0,
                       LVec<double, 3> eb1,
                       LVec<double, 3> dea0,
                       LVec<double, 3> dea1,
                       LVec<double, 3> deb0,
                       LVec<double, 3> deb1,
                       double          eta,
                       double          thickness,
                       int             max_iter,
                       double&         toc)
{
    auto mov = (dea0 + dea1 + deb0 + deb1) / 4.0;
    dea0     = dea0 - mov;
    dea1     = dea1 - mov;
    deb0     = deb0 - mov;
    deb1     = deb1 - mov;
    const double maxDispMag = std::sqrt(std::max(lmath::squaredNorm(dea0), lmath::squaredNorm(dea1)))
                              + std::sqrt(std::max(lmath::squaredNorm(deb0), lmath::squaredNorm(deb1)));
    if(maxDispMag == 0.0)
        return false;

    double dist2_cur = 0.0;
    auto   flag      = dist::edge_edge_distance_flag<double>(ea0, ea1, eb0, eb1);
    dist::edge_edge_distance2(flag, ea0, ea1, eb0, eb1, dist2_cur);
    double dFunc = dist2_cur - thickness * thickness;
    if(dFunc <= 0.0)
    {
        const double d0 = lmath::squaredNorm(ea0 - eb0);
        const double d1 = lmath::squaredNorm(ea0 - eb1);
        const double d2 = lmath::squaredNorm(ea1 - eb0);
        const double d3 = lmath::squaredNorm(ea1 - eb1);
        dist2_cur       = std::min(std::min(d0, d1), std::min(d2, d3));
        dFunc           = dist2_cur - thickness * thickness;
    }
    double dist_cur = std::sqrt(dist2_cur);
    double gap      = eta * dFunc / (dist_cur + thickness);
    double toc_prev = toc;
    toc             = 0.0;
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }
        const double tocLowerBound = (1 - eta) * dFunc / ((dist_cur + thickness) * maxDispMag);
        ea0 = ea0 + tocLowerBound * dea0;
        ea1 = ea1 + tocLowerBound * dea1;
        eb0 = eb0 + tocLowerBound * deb0;
        eb1 = eb1 + tocLowerBound * deb1;
        flag = dist::edge_edge_distance_flag<double>(ea0, ea1, eb0, eb1);
        dist::edge_edge_distance2(flag, ea0, ea1, eb0, eb1, dist2_cur);
        dFunc = dist2_cur - thickness * thickness;
        if(dFunc <= 0.0)
        {
            const double d0 = lmath::squaredNorm(ea0 - eb0);
            const double d1 = lmath::squaredNorm(ea0 - eb1);
            const double d2 = lmath::squaredNorm(ea1 - eb0);
            const double d3 = lmath::squaredNorm(ea1 - eb1);
            dist2_cur       = std::min(std::min(d0, d1), std::min(d2, d3));
            dFunc           = dist2_cur - thickness * thickness;
        }
        dist_cur = std::sqrt(dist2_cur);
        if(toc && (dFunc / (dist_cur + thickness) < gap))
            break;
        toc += tocLowerBound;
        if(toc > toc_prev)
            return false;
    }
    return true;
}

bool ref_point_edge_ccd(LVec<double, 3> p,
                        LVec<double, 3> e0,
                        LVec<double, 3> e1,
                        LVec<double, 3> dp,
                        LVec<double, 3> de0,
                        LVec<double, 3> de1,
                        double          eta,
                        double          thickness,
                        int             max_iter,
                        double&         toc)
{
    auto mov = (dp + de0 + de1) / 3.0;
    de0      = de0 - mov;
    de1      = de1 - mov;
    dp       = dp - mov;
    const double maxDispMag =
        dp.norm() + std::sqrt(std::max(lmath::squaredNorm(de0), lmath::squaredNorm(de1)));
    if(maxDispMag == 0.0)
        return false;

    double dist2_cur = 0.0;
    auto   flag      = dist::point_edge_distance_flag<double>(p, e0, e1);
    dist::point_edge_distance2(flag, p, e0, e1, dist2_cur);
    double dist_cur = std::sqrt(dist2_cur);
    double gap      = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    double toc_prev = toc;
    toc             = 0.0;
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }
        const double tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness)
                                     / ((dist_cur + thickness) * maxDispMag);
        p  = p + tocLowerBound * dp;
        e0 = e0 + tocLowerBound * de0;
        e1 = e1 + tocLowerBound * de1;
        flag = dist::point_edge_distance_flag<double>(p, e0, e1);
        dist::point_edge_distance2(flag, p, e0, e1, dist2_cur);
        dist_cur = std::sqrt(dist2_cur);
        if(toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
            break;
        toc += tocLowerBound;
        if(toc > toc_prev)
            return false;
    }
    return true;
}

bool ref_point_point_ccd(LVec<double, 3> p0,
                         LVec<double, 3> p1,
                         LVec<double, 3> dp0,
                         LVec<double, 3> dp1,
                         double          eta,
                         double          thickness,
                         int             max_iter,
                         double&         toc)
{
    auto mov = (dp0 + dp1) / 2.0;
    dp1      = dp1 - mov;
    dp0      = dp0 - mov;
    const double maxDispMag = dp0.norm() + dp1.norm();
    if(maxDispMag == 0.0)
        return false;

    double dist2_cur = 0.0;
    auto   flag      = dist::point_point_distance_flag<double>(p0, p1);
    dist::point_point_distance2(flag, p0, p1, dist2_cur);
    double dist_cur = std::sqrt(dist2_cur);
    double gap      = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    double toc_prev = toc;
    toc             = 0.0;
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }
        const double tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness)
                                     / ((dist_cur + thickness) * maxDispMag);
        p0 = p0 + tocLowerBound * dp0;
        p1 = p1 + tocLowerBound * dp1;
        flag = dist::point_point_distance_flag<double>(p0, p1);
        dist::point_point_distance2(flag, p0, p1, dist2_cur);
        dist_cur = std::sqrt(dist2_cur);
        if(toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
            break;
        toc += tocLowerBound;
        if(toc > toc_prev)
            return false;
    }
    return true;
}

void run_ccd_loops()
{
    static const int kMaxIters[] = {0, 1, 2, 3, 5, 10, 25, 100};

    Lcg rng;
    int mismatches = 0, hits = 0, misses = 0, tocs_compared = 0;
    double worst_toc = 0.0;

    const double kTocIn = 1.0;  // the reference's caller passes `large_enough_toi`

    for(int it = 0; it < kRandomIters; ++it)
    {
        const double eta       = 0.1 + 0.8 * rng.unit();
        const double thickness = 0.001 + 0.05 * rng.unit();

        // --- point / triangle pair moving into contact ------------------------
        {
            const LVec<double, 3> t0 = rnd_vec(rng, 0.5);
            const LVec<double, 3> u  = rnd_vec(rng, 0.5) + LVec<double, 3>{0.4, 0.0, 0.0};
            const LVec<double, 3> v  = rnd_vec(rng, 0.5) + LVec<double, 3>{0.0, 0.4, 0.0};
            const LVec<double, 3> t1 = t0 + u, t2 = t0 + v;
            const LVec<double, 3> n  = lmath::normalized(lmath::cross(u, v));
            const LVec<double, 3> centroid = (t0 + t1 + t2) / 3.0;
            const double gap_size = thickness + 0.02 + 0.6 * rng.unit();
            const LVec<double, 3> p = centroid + n * gap_size;
            // 0.3 means "dawdles and stops above the surface", >1 overshoots
            const double         speed = 0.3 + 2.0 * rng.unit();
            const LVec<double, 3> dp   = (n * (-gap_size)) * speed;
            const LVec<double, 3> dt0  = rnd_vec(rng, 0.05);
            const LVec<double, 3> dt1  = rnd_vec(rng, 0.05);
            const LVec<double, 3> dt2  = rnd_vec(rng, 0.05);

            for(int mi : kMaxIters)
            {
                double toc_p = kTocIn, toc_r = kTocIn;
                const bool found_p = dist::point_triangle_ccd(
                    p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness, mi, toc_p);
                const bool found_r =
                    ref_point_triangle_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness, mi, toc_r);
                if(found_p != found_r)
                {
                    ++mismatches;
                }
                else
                {
                    ++tocs_compared;
                    worst_toc = std::max(worst_toc, std::abs(toc_p - toc_r) / std::max(1.0, std::abs(toc_r)));
                }
                found_p ? ++hits : ++misses;
            }
        }

        // --- edge / edge pair -------------------------------------------------
        {
            const LVec<double, 3> ea0 = LVec<double, 3>{-0.5, 0.0, 0.0} + rnd_vec(rng, 0.2);
            const LVec<double, 3> ea1 = LVec<double, 3>{0.5, 0.0, 0.0} + rnd_vec(rng, 0.2);
            const double          gap_size = thickness + 0.02 + 0.6 * rng.unit();
            const LVec<double, 3> eb0 = LVec<double, 3>{0.0, gap_size, -0.25} + rnd_vec(rng, 0.2);
            const LVec<double, 3> eb1 = LVec<double, 3>{0.0, gap_size, 0.25} + rnd_vec(rng, 0.2);
            const double          speed = 0.3 + 2.0 * rng.unit();
            const LVec<double, 3> closing{0.0, -gap_size * speed, 0.0};
            const LVec<double, 3> da0 = rnd_vec(rng, 0.05), da1 = rnd_vec(rng, 0.05);
            const LVec<double, 3> db0 = closing, db1 = closing;

            for(int mi : kMaxIters)
            {
                double toc_p = kTocIn, toc_r = kTocIn;
                const bool found_p = dist::edge_edge_ccd(
                    ea0, ea1, eb0, eb1, da0, da1, db0, db1, eta, thickness, mi, toc_p);
                const bool found_r = ref_edge_edge_ccd(
                    ea0, ea1, eb0, eb1, da0, da1, db0, db1, eta, thickness, mi, toc_r);
                if(found_p != found_r)
                {
                    ++mismatches;
                }
                else
                {
                    ++tocs_compared;
                    worst_toc = std::max(worst_toc, std::abs(toc_p - toc_r) / std::max(1.0, std::abs(toc_r)));
                }
                found_p ? ++hits : ++misses;
            }
        }

        // --- point / edge -----------------------------------------------------
        {
            const LVec<double, 3> e0 = rnd_vec(rng, 0.5);
            const LVec<double, 3> e1 = e0 + rnd_vec(rng, 0.5) + LVec<double, 3>{0.4, 0.0, 0.0};
            const double          gap_size = thickness + 0.02 + 0.6 * rng.unit();
            const LVec<double, 3> p        = (e0 + e1) / 2.0 + LVec<double, 3>{0.0, gap_size, 0.0};
            const double          speed    = 0.3 + 2.0 * rng.unit();
            const LVec<double, 3> dp{0.0, -gap_size * speed, 0.0};
            const LVec<double, 3> de0 = rnd_vec(rng, 0.05), de1 = rnd_vec(rng, 0.05);

            for(int mi : kMaxIters)
            {
                double toc_p = kTocIn, toc_r = kTocIn;
                const bool found_p =
                    dist::point_edge_ccd(p, e0, e1, dp, de0, de1, eta, thickness, mi, toc_p);
                const bool found_r =
                    ref_point_edge_ccd(p, e0, e1, dp, de0, de1, eta, thickness, mi, toc_r);
                if(found_p != found_r)
                {
                    ++mismatches;
                }
                else
                {
                    ++tocs_compared;
                    worst_toc = std::max(worst_toc, std::abs(toc_p - toc_r) / std::max(1.0, std::abs(toc_r)));
                }
                found_p ? ++hits : ++misses;
            }
        }

        // --- point / point ----------------------------------------------------
        {
            const double          gap_size = thickness + 0.02 + 0.6 * rng.unit();
            const LVec<double, 3> p0       = rnd_vec(rng, 0.5);
            const LVec<double, 3> p1       = p0 + LVec<double, 3>{0.0, gap_size, 0.0};
            const double          speed    = 0.3 + 2.0 * rng.unit();
            const LVec<double, 3> dp0{0.0, gap_size * speed * 0.5, 0.0};
            const LVec<double, 3> dp1{0.0, -gap_size * speed * 0.5, 0.0};

            for(int mi : kMaxIters)
            {
                double toc_p = kTocIn, toc_r = kTocIn;
                const bool found_p = dist::point_point_ccd(p0, p1, dp0, dp1, eta, thickness, mi, toc_p);
                const bool found_r = ref_point_point_ccd(p0, p1, dp0, dp1, eta, thickness, mi, toc_r);
                if(found_p != found_r)
                {
                    ++mismatches;
                }
                else
                {
                    ++tocs_compared;
                    worst_toc = std::max(worst_toc, std::abs(toc_p - toc_r) / std::max(1.0, std::abs(toc_r)));
                }
                found_p ? ++hits : ++misses;
            }
        }
    }

    std::printf("    [ccd loops] %d found / %d not found, %d mismatches vs the reference loop, "
                "worst |toc| rel %.3e\n",
                hits,
                misses,
                mismatches,
                worst_toc);
    CHECK(mismatches == 0);
    CHECK(hits > 1000);
    CHECK(misses > 1000);
    CHECK(worst_toc <= 1e-12);

    // A dedicated "nothing moves" case: the reference returns false *before*
    // touching `toc`, which is what the port must reproduce.
    {
        const LVec<double, 3> p{0.0, 0.5, 0.0}, t0{0.0, 0.0, 0.0}, t1{1.0, 0.0, 0.0}, t2{0.0, 1.0, 0.0};
        double                toc_p = 0.75, toc_r = 0.75;
        const bool            found_p = dist::point_triangle_ccd(p, t0, t1, t2, LVec<double, 3>{}, LVec<double, 3>{}, LVec<double, 3>{}, LVec<double, 3>{}, 0.5, 0.01, 10, toc_p);
        const bool            found_r = ref_point_triangle_ccd(p, t0, t1, t2, LVec<double, 3>{}, LVec<double, 3>{}, LVec<double, 3>{}, LVec<double, 3>{}, 0.5, 0.01, 10, toc_r);
        CHECK(found_p == found_r);
        CHECK(found_p == false);
        CHECK(toc_p == toc_r);  // untouched
        CHECK(toc_p == 0.75);
    }

    // The unbounded form (`max_iter < 0`): the port uses a fixed host bound
    // (`detail::kCcdUnboundedIterations`); on a converging configuration both must
    // land on the same time of impact.
    {
        const LVec<double, 3> p{0.0, 0.6, 0.0}, t0{0.0, 0.0, 0.0}, t1{1.0, 0.0, 0.0}, t2{0.0, 1.0, 0.0};
        const LVec<double, 3> dp{0.0, -0.5, 0.0};
        double                toc_p = 1.0, toc_r = 1.0;
        const bool            found_p = dist::point_triangle_ccd(p, t0, t1, t2, dp, LVec<double, 3>{}, LVec<double, 3>{}, LVec<double, 3>{}, 0.5, 0.01, -1, toc_p);
        const bool            found_r = ref_point_triangle_ccd(p, t0, t1, t2, dp, LVec<double, 3>{}, LVec<double, 3>{}, LVec<double, 3>{}, 0.5, 0.01, -1, toc_r);
        CHECK(found_p == found_r);
        CHECK(close(toc_p, toc_r, 1e-12));
    }
}
}// namespace

TEST_CASE("distance_host_ccd_loop_reference", "[luisa][distance][reference]")
{
    run_ccd_loops();
}
#endif  // UIPC_HAVE_DISTANCE_REFERENCE

// ===========================================================================
//  5. Flag constructors and the flag-dispatched primitives (host behaviour)
// ===========================================================================
namespace
{
using Flag3 = dist::FlagVec<double, 3>;
using Flag4 = dist::FlagVec<double, 4>;

std::string flag_to_string(const int* f, int n)
{
    std::string s = "{";
    for(int i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(f[i]);
    return s + "}";
}

template <int N>
bool flag_equals(const dist::FlagVec<double, N>& got, const int* want)
{
    for(int i = 0; i < N; ++i)
        if(got(i) != want[i])
            return false;
    return true;
}

/// `CHECK` plus the two flag vectors in the diagnostic (the standalone harness has
/// no Catch2 `INFO`, so the context is printed directly).
template <int N>
void check_flag(const char* what, const dist::FlagVec<double, N>& got, const int* want)
{
    int g[N];
    for(int i = 0; i < N; ++i) g[i] = got(i);
    if(!flag_equals(got, want))
        std::printf("    [%s] flag mismatch: got %s, want %s\n",
                    what,
                    flag_to_string(g, N).c_str(),
                    flag_to_string(want, N).c_str());
    CHECK(flag_equals(got, want));
}

/// The reference's per-dim layout, restated independently of `flagged.h` (it is a
/// re-reading of `distance_flagged.h`'s `pp_from_*` / `pe_from_pt` / `pe_from_ee`):
///   dim 2  -> the two active indices, ascending;
///   dim 3, point-triangle -> `p` stays the point unless `p` is the inactive one;
///            then the smallest active triangle vertex is the point and the other
///            two are the edge (ascending);
///   dim 3, edge-edge -> the point is the active vertex whose partner is inactive,
///            the other two (ascending) are the edge;
///   dim 4  -> all four.
void pt_layout(const Flag4& f, int dim, int* out)
{
    if(dim == 2)
    {
        int k = 0;
        for(int i = 0; i < 4; ++i)
            if(f(i) != 0) out[k++] = i;
    }
    else if(dim == 3)
    {
        int inactive = 0;
        for(int i = 0; i < 4; ++i)
            if(f(i) == 0) inactive = i;
        if(inactive == 0)
        {
            out[0] = 1;
            out[1] = 2;
            out[2] = 3;
        }
        else
        {
            out[0] = 0;  // the point is still `p`
            int k = 1;
            for(int i = 1; i < 4; ++i)
                if(f(i) != 0) out[k++] = i;
        }
    }
    else
    {
        out[0] = 0;
        out[1] = 1;
        out[2] = 2;
        out[3] = 3;
    }
}

void ee_layout(const Flag4& f, int dim, int* out)
{
    if(dim == 2)
    {
        int k = 0;
        for(int i = 0; i < 4; ++i)
            if(f(i) != 0) out[k++] = i;
    }
    else if(dim == 3)
    {
        int inactive = 0;
        for(int i = 0; i < 4; ++i)
            if(f(i) == 0) inactive = i;
        // the free endpoint of the degenerate edge comes first
        if(inactive == 0)
        {
            out[0] = 1;
            out[1] = 2;
            out[2] = 3;
        }
        else if(inactive == 1)
        {
            out[0] = 0;
            out[1] = 2;
            out[2] = 3;
        }
        else if(inactive == 2)
        {
            out[0] = 3;
            out[1] = 0;
            out[2] = 1;
        }
        else
        {
            out[0] = 2;
            out[1] = 0;
            out[2] = 1;
        }
    }
    else
    {
        out[0] = 0;
        out[1] = 1;
        out[2] = 2;
        out[3] = 3;
    }
}
}// namespace

TEST_CASE("distance_flagged_degenerate_edge_edge_layout", "[luisa][distance][flagged]")
{
    // Regression guard: at `dim == 3`, `degenerate_edge_edge` must return
    // `pe_from_ee`'s **direct** indices into {ea0, ea1, eb0, eb1} — the layout
    // `ee_layout` restates above — and not the ascending active ranks that
    // `active_offsets` produces.  Only the all-ascending flag ({1,1,1,0}, inactive 3)
    // agreed; the other three used to come back permuted, e.g. active {0,1,3}
    // (inactive 2, table row {3,0,1}) was reported as {2,0,1}, so the point of the
    // point-edge pair became eb0 instead of the free endpoint eb1.  The
    // `edge_edge_distance2` dispatch does not go through this helper (it indexes
    // `kPeFromEe` directly), but the trajectory-filter kernels do
    // (`simplex_trajectory_filter_kernels.h:1379`), which is why the distance tests
    // alone stayed green.
    for(int inactive = 0; inactive < 4; ++inactive)
    {
        int bits[4] = {1, 1, 1, 1};
        bits[inactive] = 0;
        const Flag4 flag = dist::detail::flag4<int>(bits[0], bits[1], bits[2], bits[3]);

        LVec<int, 4> offsets{0, 0, 0, 0};
        const int    dim = dist::degenerate_edge_edge(flag, offsets);
        CHECK(dim == 3);

        int want[4] = {0, 0, 0, 0};
        ee_layout(flag, 3, want);
        for(int s = 0; s < 3; ++s)
        {
            if(offsets(s) != want[s])
                std::printf(" [ee degenerate] inactive %d: offsets(%d) = %d, want %d\n",
                            inactive,
                            s,
                            offsets(s),
                            want[s]);
            CHECK(offsets(s) == want[s]);
        }
    }

    // `dim == 2` keeps `pp_from_ee`'s ascending active pair.
    {
        const Flag4  flag = dist::detail::flag4<int>(1, 0, 1, 0);
        LVec<int, 4> offsets{0, 0, 0, 0};
        const int    dim = dist::degenerate_edge_edge(flag, offsets);
        CHECK(dim == 2);
        CHECK(offsets(0) == 0);
        CHECK(offsets(1) == 2);
    }
}

TEST_CASE("distance_flagged_host_flags", "[luisa][distance][flagged]")
{
    // --- point-point: a pair is always fully active --------------------------
    {
        LVec<double, 3> p0{0.0, 0.0, 0.0}, p1{1.0, 2.0, 3.0};
        const int       want[2] = {1, 1};
        check_flag("point_point_distance_flag", dist::point_point_distance_flag<double>(p0, p1), want);
    }

    // --- point-edge: ratio < 1 / ratio > 0, and the degenerate fallback -------
    {
        const LVec<double, 3> e0{0.0, 0.0, 0.0}, e1{1.0, 0.0, 0.0};
        struct Case
        {
            LVec<double, 3> p;
            int             want[3];
        };
        const Case cases[] = {
            {{0.5, 1.0, 0.0}, {1, 1, 1}},   // projection interior  -> the whole edge
            {{2.0, 1.0, 0.0}, {1, 0, 1}},   // beyond e1            -> PE(p, e1)
            {{-1.0, 1.0, 0.0}, {1, 1, 0}},  // before e0            -> PE(p, e0)
            {{0.5, 0.0, 0.0}, {1, 1, 1}},   // on the edge
            {{1.0, 0.5, 0.0}, {1, 0, 1}},   // at e1
        };
        for(const auto& c : cases)
            check_flag("point_edge_distance_flag",
                       dist::point_edge_distance_flag<double>(c.p, e0, e1),
                       c.want);
        // zero-length edge: fall back to point-point against the nearer endpoint
        const LVec<double, 3> z{1.0, 0.0, 0.0};
        const int             want_deg[3] = {1, 1, 0};
        check_flag("point_edge_distance_flag (zero-length)",
                   dist::point_edge_distance_flag<double>(LVec<double, 3>{2.0, 0.0, 0.0}, z, z),
                   want_deg);
    }

    // --- point-triangle: all seven terminal cases ---------------------------
    {
        const LVec<double, 3> t0{0.0, 0.0, 0.0}, t1{1.0, 0.0, 0.0}, t2{0.0, 1.0, 0.0};
        struct Case
        {
            LVec<double, 3> p;
            int             want[4];
            const char*     what;
        };
        const Case cases[] = {
            {{0.25, 0.25, 1.0}, {1, 1, 1, 1}, "PT (projection inside)"},
            {{0.5, 0.5, 1.0}, {1, 0, 1, 1}, "PE t1t2 (on the hypotenuse)"},
            {{0.25, -0.5, 0.0}, {1, 1, 1, 0}, "PE t0t1"},
            {{-0.5, 0.25, 0.0}, {1, 1, 0, 1}, "PE t2t0"},
            {{1.5, 0.25, 0.0}, {1, 0, 1, 0}, "PP t1"},
            {{-1.0, -1.0, 0.0}, {1, 1, 0, 0}, "PP t0"},
            {{0.25, 1.5, 0.0}, {1, 0, 0, 1}, "PP t2"},
        };
        for(const auto& c : cases)
            check_flag(c.what,
                       dist::point_triangle_distance_flag<double>(c.p, t0, t1, t2),
                       c.want);
    }

    // --- edge-edge: general and near-parallel (typed-candidate) paths --------
    {
        struct Case
        {
            LVec<double, 3> ea0, ea1, eb0, eb1;
            int             want[4];
            const char*     what;
        };
        const Case cases[] = {
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {1, 0, 1, 0}, "parallel (typed PP candidate)"},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.25, 1.0, 0.0}, {1.25, 1.0, 0.0}, {0, 1, 1, 1}, "parallel, offset (PE)"},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, -1.0, 0.0}, {0.5, 1.0, 0.0}, {0, 1, 1, 1}, "crossing (PE)"},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.1, 0.1, 0.0}, {2.0, 3.0, 0.0}, {1, 1, 1, 0}, "a0 near b0 (PE)"},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-2.0, -2.0, 0.0}, {1.05, 0.05, 0.0}, {0, 1, 1, 1}, "b1 near a1 (PE)"},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.2, 0.5, 0.5}, {0.8, 0.5, -0.5}, {1, 1, 1, 1}, "skew (full EE)"},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0, 1, 1, 0}, "collinear disjoint (PP)"},
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {1.5, 0.0, 0.0}, {0, 1, 1, 1}, "collinear overlapping (PE)"},
        };
        for(const auto& c : cases)
            check_flag(c.what,
                       dist::edge_edge_distance_flag<double>(c.ea0, c.ea1, c.eb0, c.eb1),
                       c.want);
    }
}

TEST_CASE("distance_flagged_host_dispatch", "[luisa][distance][flagged]")
{
    // The flag-dispatched primitives must agree with the *reference-verified*
    // canonical primitives applied to the layout this test derives independently
    // (`pt_layout` / `ee_layout`, a re-reading of `distance_flagged.h`), and every
    // flag must have 2, 3 or 4 active entries.
    Lcg rng;

    auto scatter_vec = [](const double* canon, const int* out, int n, double* dst)
    {
        for(int i = 0; i < 12; ++i) dst[i] = 0.0;
        for(int s = 0; s < n; ++s)
            for(int r = 0; r < 3; ++r) dst[3 * out[s] + r] = canon[3 * s + r];
    };
    auto scatter_mat = [](const double* canon, const int* out, int n, double* dst)
    {
        constexpr int N = 4;
        for(int i = 0; i < 12 * 12; ++i) dst[i] = 0.0;
        for(int s = 0; s < n; ++s)
            for(int t = 0; t < n; ++t)
                for(int r = 0; r < 3; ++r)
                    for(int c = 0; c < 3; ++c)
                        dst[(3 * out[t] + c) * 12 + (3 * out[s] + r)] =
                            canon[(3 * t + c) * (3 * n) + (3 * s + r)];
    };

    double worst[6] = {};     // PT: d2 / grad / hessian, then EE: d2 / grad / hessian
    int    counts[2][3] = {};  // [point-triangle, edge-edge][dim 2, 3, 4]

    double canon_g[12], canon_h[144], want_g[12], want_h[144], got_buf[144];
    std::vector<LVec<double, 3>> X(4);

    for(int it = 0; it < kRandomIters; ++it)
    {
        X[0] = rnd_vec(rng);                 // p   / ea0
        X[1] = rnd_vec(rng);                 // t0  / ea1
        X[2] = X[1] + rnd_vec(rng);          // t1  / eb0
        X[3] = X[1] + rnd_vec(rng);          // t2  / eb1
        if(it % 4 == 0)                      // cluster (near-degenerate) configuration
        {
            X[1] = X[0] + rnd_vec(rng, 0.5);
            X[2] = X[0] + rnd_vec(rng, 0.5);
            X[3] = X[0] + rnd_vec(rng, 0.5);
        }
        else if(it % 4 == 1)
            X[3] = X[2] + rnd_vec(rng, 0.01);  // near-degenerate second simplex
        else if(it % 4 == 2)
            X[3] = X[1];                       // exactly degenerate

        // ---------------- point-triangle -----------------------------------
        {
            const Flag4 flag = dist::point_triangle_distance_flag<double>(X[0], X[1], X[2], X[3]);
            int         act  = 0;
            for(int i = 0; i < 4; ++i) act += (flag(i) != 0) ? 1 : 0;
            CHECK((act >= 2 && act <= 4));
            if(act < 2 || act > 4)
                continue;
            counts[0][act - 2] += 1;

            int out[4];
            pt_layout(flag, act, out);

            double got_d2 = 0.0, want_d2 = 0.0;
            dist::point_triangle_distance2(flag, X[0], X[1], X[2], X[3], got_d2);
            if(act == 2)
                dist::point_point_distance2(X[out[0]], X[out[1]], want_d2);
            else if(act == 3)
                dist::point_edge_distance2(X[out[0]], X[out[1]], X[out[2]], want_d2);
            else
                dist::point_triangle_distance2(X[0], X[1], X[2], X[3], want_d2);
            worst[0] = std::max(worst[0], diff(got_d2, want_d2));

            LVec<double, 12> got_g;
            dist::point_triangle_distance2_gradient(flag, X[0], X[1], X[2], X[3], got_g);
            for(int i = 0; i < 12; ++i) canon_g[i] = 0.0;
            if(act == 2)
            {
                LVec<double, 6> g6;
                dist::point_point_distance2_gradient(X[out[0]], X[out[1]], g6);
                to_flat(g6, canon_g);
            }
            else if(act == 3)
            {
                LVec<double, 9> g9;
                dist::point_edge_distance2_gradient(X[out[0]], X[out[1]], X[out[2]], g9);
                to_flat(g9, canon_g);
            }
            else
            {
                LVec<double, 12> g12;
                dist::point_triangle_distance2_gradient(X[0], X[1], X[2], X[3], g12);
                to_flat(g12, canon_g);
            }
            scatter_vec(canon_g, out, act, want_g);
            to_flat(got_g, got_buf);
            worst[1] = std::max(worst[1], max_abs_diff(got_buf, want_g, 12));

            LMat<double, 12, 12> got_h;
            dist::point_triangle_distance2_hessian(flag, X[0], X[1], X[2], X[3], got_h);
            for(int i = 0; i < 144; ++i) canon_h[i] = 0.0;
            if(act == 2)
            {
                LMat<double, 6, 6> h6;
                dist::point_point_distance2_hessian(X[out[0]], X[out[1]], h6);
                to_flat(h6, canon_h);
            }
            else if(act == 3)
            {
                LMat<double, 9, 9> h9;
                dist::point_edge_distance2_hessian(X[out[0]], X[out[1]], X[out[2]], h9);
                to_flat(h9, canon_h);
            }
            else
            {
                LMat<double, 12, 12> h12;
                dist::point_triangle_distance2_hessian(X[0], X[1], X[2], X[3], h12);
                to_flat(h12, canon_h);
            }
            scatter_mat(canon_h, out, act, want_h);
            to_flat(got_h, got_buf);
            worst[2] = std::max(worst[2], max_abs_diff(got_buf, want_h, 144));
        }

        // ---------------- edge-edge ----------------------------------------
        {
            const Flag4 flag = dist::edge_edge_distance_flag<double>(X[0], X[1], X[2], X[3]);
            int         act  = 0;
            for(int i = 0; i < 4; ++i) act += (flag(i) != 0) ? 1 : 0;
            CHECK((act >= 2 && act <= 4));
            if(act < 2 || act > 4)
                continue;
            counts[1][act - 2] += 1;

            int out[4];
            ee_layout(flag, act, out);

            double got_d2 = 0.0, want_d2 = 0.0;
            dist::edge_edge_distance2(flag, X[0], X[1], X[2], X[3], got_d2);
            if(act == 2)
                dist::point_point_distance2(X[out[0]], X[out[1]], want_d2);
            else if(act == 3)
                dist::point_edge_distance2(X[out[0]], X[out[1]], X[out[2]], want_d2);
            else
                dist::edge_edge_distance2(X[0], X[1], X[2], X[3], want_d2);
            worst[3] = std::max(worst[3], diff(got_d2, want_d2));

            LVec<double, 12> got_g;
            dist::edge_edge_distance2_gradient(flag, X[0], X[1], X[2], X[3], got_g);
            for(int i = 0; i < 12; ++i) canon_g[i] = 0.0;
            if(act == 2)
            {
                LVec<double, 6> g6;
                dist::point_point_distance2_gradient(X[out[0]], X[out[1]], g6);
                to_flat(g6, canon_g);
            }
            else if(act == 3)
            {
                LVec<double, 9> g9;
                dist::point_edge_distance2_gradient(X[out[0]], X[out[1]], X[out[2]], g9);
                to_flat(g9, canon_g);
            }
            else
            {
                LVec<double, 12> g12;
                dist::edge_edge_distance2_gradient(X[0], X[1], X[2], X[3], g12);
                to_flat(g12, canon_g);
            }
            scatter_vec(canon_g, out, act, want_g);
            to_flat(got_g, got_buf);
            worst[4] = std::max(worst[4], max_abs_diff(got_buf, want_g, 12));

            LMat<double, 12, 12> got_h;
            dist::edge_edge_distance2_hessian(flag, X[0], X[1], X[2], X[3], got_h);
            for(int i = 0; i < 144; ++i) canon_h[i] = 0.0;
            if(act == 2)
            {
                LMat<double, 6, 6> h6;
                dist::point_point_distance2_hessian(X[out[0]], X[out[1]], h6);
                to_flat(h6, canon_h);
            }
            else if(act == 3)
            {
                LMat<double, 9, 9> h9;
                dist::point_edge_distance2_hessian(X[out[0]], X[out[1]], X[out[2]], h9);
                to_flat(h9, canon_h);
            }
            else
            {
                LMat<double, 12, 12> h12;
                dist::edge_edge_distance2_hessian(X[0], X[1], X[2], X[3], h12);
                to_flat(h12, canon_h);
            }
            scatter_mat(canon_h, out, act, want_h);
            to_flat(got_h, got_buf);
            worst[5] = std::max(worst[5], max_abs_diff(got_buf, want_h, 144));
        }
    }

    std::printf("    [flagged dispatch] PT d2 %.3e grad %.3e hessian %.3e | EE d2 %.3e grad %.3e hessian %.3e\n",
                worst[0],
                worst[1],
                worst[2],
                worst[3],
                worst[4],
                worst[5]);
    std::printf("    [flagged dispatch] dim counts: PT %d/%d/%d, EE %d/%d/%d (dim 2/3/4)\n",
                counts[0][0],
                counts[0][1],
                counts[0][2],
                counts[1][0],
                counts[1][1],
                counts[1][2]);

    for(int i = 0; i < 6; ++i) CHECK(worst[i] <= 1e-12);
    // every dim must occur for both families
    for(int d = 0; d < 3; ++d)
    {
        CHECK(counts[0][d] > 0);
        CHECK(counts[1][d] > 0);
    }
}

// ===========================================================================
//  6. Device smoke: ONE generic routine, instantiated as `double` (host) and
//     `lcrt::Var<double>` (kernel); the two must agree
// ===========================================================================
namespace
{
// Per element input: 7 points (p, e0, e1, t0, t1, t2, q) followed by their 7
// displacements, followed by two scalars (mollifier scale, eta).
constexpr int kPoints      = 7;
constexpr int kInputSize   = kPoints * 3 * 2 + 3;
constexpr int kMollScaleAt = kPoints * 3 * 2;
constexpr int kEtaAt       = kMollScaleAt + 1;
constexpr int kThickAt     = kEtaAt + 1;

// Output layout (in dump order):
//   PP  d2, g(6), h(36)                                       43
//   PE  d2, g(9), h(81)                                       91
//   PT  d2, g(12), h(144)                                    157
//   EE  d2, g(12), h(144)                                    157
//   mollifier: threshold, e, g(12), h(144), need_mollify         159
//   CCD: PT (found, toc), EE, PE, PP                             8
//   broad phases                                                 8
//   flagged PT: flag(4), d2, g(12), h(144)                     161
//   flagged EE: flag(4), d2, g(12), h(144)                     161
constexpr int kOutSize = 43 + 91 + 157 + 157 + 159 + 8 + 8 + 161 + 161;

constexpr int kDevN     = 8;
constexpr int kCcdIters = 4;  // a small budget: the loop semantics are covered by
                              // distance_host_ccd_loop_reference

template <typename V, typename Put>
void put_vec(const V& v, Put&& put)
{
    for(int i = 0; i < lmath::vec_dim_v<V>; ++i) put(v(i));
}

template <typename M, typename Put>
void put_mat(const M& m, Put&& put)
{
    for(int c = 0; c < lmath::mat_cols_v<M>; ++c)
        for(int r = 0; r < lmath::mat_rows_v<M>; ++r) put(m(r, c));
}

/// 0/1 of a predicate (`bool` on the host, a DSL boolean expression on the device).
template <typename S, typename P>
S as_scalar(const P& pred)
{
    return dist::detail::select_scalar(lmath::zero_of<S>(), lmath::one_of<S>(), pred);
}

/// ONE generic computation for every primitive of the layer.
///
/// `load(j)` reads the j-th input scalar, `put(value)` appends the next output
/// scalar.  Instantiated with `S = double` outside a kernel and with
/// `S = lcrt::Var<double>` inside one, the two must agree (the dump order is
/// identical by construction).
template <typename S, typename Load, typename Put>
void dist_calc(const Load& load, Put&& put, S eta, S thickness)
{
    const auto P = [&](int k) noexcept
    { return LVec<S, 3>{load(3 * k + 0), load(3 * k + 1), load(3 * k + 2)}; };
    const auto D = [&](int k) noexcept
    {
        const int base = kPoints * 3;
        return LVec<S, 3>{load(base + 3 * k + 0), load(base + 3 * k + 1), load(base + 3 * k + 2)};
    };
    // p, e0, e1, t0, t1, t2, q
    const LVec<S, 3> p = P(0), e0 = P(1), e1 = P(2), t0 = P(3), t1 = P(4), t2 = P(5), q = P(6);
    const LVec<S, 3> dp = D(0), de0 = D(1), de1 = D(2), dt0 = D(3), dt1 = D(4), dt2 = D(5), dq = D(6);

    // --- point / point -------------------------------------------------------
    {
        S d2;
        dist::point_point_distance2(p, q, d2);
        put(d2);
        LVec<S, 6> g;
        dist::point_point_distance2_gradient(p, q, g);
        put_vec(g, put);
        LMat<S, 6, 6> h;
        dist::point_point_distance2_hessian(p, q, h);
        put_mat(h, put);
    }
    // --- point / edge --------------------------------------------------------
    {
        S d2;
        dist::point_edge_distance2(p, e0, e1, d2);
        put(d2);
        LVec<S, 9> g;
        dist::point_edge_distance2_gradient(p, e0, e1, g);
        put_vec(g, put);
        LMat<S, 9, 9> h;
        dist::point_edge_distance2_hessian(p, e0, e1, h);
        put_mat(h, put);
    }
    // --- point / triangle ----------------------------------------------------
    {
        S d2;
        dist::point_triangle_distance2(p, t0, t1, t2, d2);
        put(d2);
        LVec<S, 12> g;
        dist::point_triangle_distance2_gradient(p, t0, t1, t2, g);
        put_vec(g, put);
        LMat<S, 12, 12> h;
        dist::point_triangle_distance2_hessian(p, t0, t1, t2, h);
        put_mat(h, put);
    }
    // --- edge / edge ---------------------------------------------------------
    {
        S d2;
        dist::edge_edge_distance2(e0, e1, t0, t1, d2);
        put(d2);
        LVec<S, 12> g;
        dist::edge_edge_distance2_gradient(e0, e1, t0, t1, g);
        put_vec(g, put);
        LMat<S, 12, 12> h;
        dist::edge_edge_distance2_hessian(e0, e1, t0, t1, h);
        put_mat(h, put);
    }
    // --- mollifier -----------------------------------------------------------
    {
        S thr;
        dist::edge_edge_mollifier_threshold(e0, e1, t0, t1, 0.5, thr);
        put(thr);
        // scale the threshold per element so that both mollifier branches occur
        S eps_x = lmath::detail::mul(thr, load(kMollScaleAt));
        S e;
        dist::edge_edge_mollifier(e0, e1, t0, t1, eps_x, e);
        put(e);
        LVec<S, 12> g;
        dist::edge_edge_mollifier_gradient(e0, e1, t0, t1, eps_x, g);
        put_vec(g, put);
        LMat<S, 12, 12> h;
        dist::edge_edge_mollifier_hessian(e0, e1, t0, t1, eps_x, h);
        put_mat(h, put);
        put(as_scalar<S>(dist::need_mollify<S>(e0, e1, t0, t1, eps_x)));
    }
    // --- additive CCD --------------------------------------------------------
    {
        S toc = lmath::make_scalar<S>(1.0);
        auto found = dist::point_triangle_ccd<S>(
            p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness, kCcdIters, toc);
        put(as_scalar<S>(found));
        put(toc);
    }
    {
        S toc = lmath::make_scalar<S>(1.0);
        auto found = dist::edge_edge_ccd<S>(
            e0, e1, t0, t1, de0, de1, dt0, dt1, eta, thickness, kCcdIters, toc);
        put(as_scalar<S>(found));
        put(toc);
    }
    {
        S toc = lmath::make_scalar<S>(1.0);
        auto found = dist::point_edge_ccd<S>(
            p, e0, e1, dp, de0, de1, eta, thickness, kCcdIters, toc);
        put(as_scalar<S>(found));
        put(toc);
    }
    {
        S toc = lmath::make_scalar<S>(1.0);
        auto found = dist::point_point_ccd<S>(p, q, dp, dq, eta, thickness, kCcdIters, toc);
        put(as_scalar<S>(found));
        put(toc);
    }
    // --- broad phases --------------------------------------------------------
    {
        put(as_scalar<S>(dist::point_edge_cd_broadphase<S>(p, e0, e1, thickness)));
        put(as_scalar<S>(dist::point_triangle_cd_broadphase<S>(p, t0, t1, t2, thickness)));
        put(as_scalar<S>(dist::edge_edge_cd_broadphase<S>(e0, e1, t0, t1, thickness)));
        put(as_scalar<S>(dist::point_triangle_ccd_broadphase<S>(p, t0, t1, t2, dp, dt0, dt1, dt2, thickness)));
        put(as_scalar<S>(dist::edge_edge_ccd_broadphase<S>(e0, e1, t0, t1, de0, de1, dt0, dt1, thickness)));
        put(as_scalar<S>(dist::point_edge_ccd_broadphase<S>(p, e0, e1, dp, de0, de1, thickness)));
        put(as_scalar<S>(dist::point_point_ccd_broadphase<S>(p, q, dp, dq, thickness)));
        const LVec<S, 2> p2{p(0), p(1)}, e20{e0(0), e0(1)}, e21{e1(0), e1(1)};
        const LVec<S, 2> dp2{dp(0), dp(1)}, de02{de0(0), de0(1)}, de12{de1(0), de1(1)};
        put(as_scalar<S>(dist::point_edge_ccd_broadphase<S>(p2, e20, e21, dp2, de02, de12, thickness)));
    }
    // --- flag-dispatched point / triangle ------------------------------------
    {
        auto flag = dist::point_triangle_distance_flag<S>(p, t0, t1, t2);
        for(int i = 0; i < 4; ++i) put(as_scalar<S>(flag(i) != 0));
        S d2;
        dist::point_triangle_distance2(flag, p, t0, t1, t2, d2);
        put(d2);
        LVec<S, 12> g;
        dist::point_triangle_distance2_gradient(flag, p, t0, t1, t2, g);
        put_vec(g, put);
        LMat<S, 12, 12> h;
        dist::point_triangle_distance2_hessian(flag, p, t0, t1, t2, h);
        put_mat(h, put);
    }
    // --- flag-dispatched edge / edge -----------------------------------------
    {
        auto flag = dist::edge_edge_distance_flag<S>(e0, e1, t0, t1);
        for(int i = 0; i < 4; ++i) put(as_scalar<S>(flag(i) != 0));
        S d2;
        dist::edge_edge_distance2(flag, e0, e1, t0, t1, d2);
        put(d2);
        LVec<S, 12> g;
        dist::edge_edge_distance2_gradient(flag, e0, e1, t0, t1, g);
        put_vec(g, put);
        LMat<S, 12, 12> h;
        dist::edge_edge_distance2_hessian(flag, e0, e1, t0, t1, h);
        put_mat(h, put);
    }
}

/// Deterministic per-element configuration, mixing generic and near-degenerate
/// geometry.  Used for the host list *and* uploaded for the kernel, so both sides
/// see identical inputs.
void fill_input(Lcg& rng, int elem, std::vector<double>& in)
{
    const std::size_t base = static_cast<std::size_t>(elem) * kInputSize;
    auto              set  = [&](int k, double x, double y, double z)
    { in[base + 3 * k + 0] = x, in[base + 3 * k + 1] = y, in[base + 3 * k + 2] = z; };
    auto setd = [&](int k, double x, double y, double z)
    {
        const int b = kPoints * 3;
        in[base + b + 3 * k + 0] = x, in[base + b + 3 * k + 1] = y, in[base + b + 3 * k + 2] = z;
    };

    // a triangle around the origin, a probe point above it, a second point
    const double t0x = 0.1 * rng.next(), t0y = 0.1 * rng.next(), t0z = 0.1 * rng.next();
    const double ux = 1.0 + 0.2 * rng.next(), uy = 0.1 * rng.next(), uz = 0.1 * rng.next();
    const double vx = 0.1 * rng.next(), vy = 1.0 + 0.2 * rng.next(), vz = 0.1 * rng.next();
    double       gap = 0.4 + 0.3 * rng.unit();

    // edge e0-e1 (nearly parallel to the triangle's first edge for odd elements)
    const double ex = 1.0 + 0.2 * rng.next(), ey = 0.05 * rng.next(), ez = 0.05 * rng.next();

    set(0, t0x, t0y, t0z + gap);                        // p
    set(1, -0.5 + 0.1 * rng.next(), 0.1 * rng.next(), 0.0);  // e0
    set(2, 0.5 + 0.1 * rng.next(), 0.1 * rng.next(), 0.0);   // e1
    set(3, t0x, t0y, t0z);                              // t0
    set(4, t0x + ux, t0y + uy, t0z + uz);               // t1
    set(5, t0x + vx, t0y + vy, t0z + vz);               // t2
    set(6, 0.3 * rng.next(), -0.7 - 0.2 * rng.unit(), 0.2 * rng.next());  // q

    // displacements: the probe falls towards the triangle, the second edge closes
    setd(0, 0.0, -gap * (0.5 + rng.unit()), 0.0);
    setd(1, 0.02 * rng.next(), 0.0, 0.02 * rng.next());
    setd(2, 0.02 * rng.next(), 0.0, 0.02 * rng.next());
    setd(3, 0.02 * rng.next(), 0.0, 0.02 * rng.next());
    setd(4, 0.02 * rng.next(), 0.0, 0.02 * rng.next());
    setd(5, 0.02 * rng.next(), 0.0, 0.02 * rng.next());
    setd(6, 0.0, 0.3 * rng.unit(), 0.0);

    // the two scalars
    in[base + kMollScaleAt] = 0.05 + 2.0 * rng.unit();
    in[base + kEtaAt]       = 0.2 + 0.6 * rng.unit();
    in[base + kThickAt]     = 0.01;

    switch(elem)
    {
        case 1:  // nearly coincident probe and vertex
            set(0, t0x + 1e-9, t0y + 1e-9, t0z);
            break;
        case 2:  // very short edge (still well conditioned: the flag's ratio test
                 // and the near-degenerate paths are exercised, no 0/0)
            set(2, in[base + 3] + 0.01, in[base + 4], in[base + 5]);
            break;
        case 3:  // thin (nearly colinear) triangle
            set(5, t0x + 0.5 * ux, t0y + 0.5 * uy, t0z + 0.5 * uz + 0.01);
            break;
        case 4:  // parallel edges
            set(3, 0.0, 0.0, 0.0);
            set(4, ex, ey, ez);
            break;
        case 5:  // nearly parallel edges
            set(3, 0.0, 0.0, 0.0);
            set(4, ex, ey, ez);
            setd(3, 1e-12, 0.0, 0.0);
            break;
        case 6:  // nothing moves (the CCD early-out, `toc` must stay untouched)
            setd(0, 0.0, 0.0, 0.0);
            setd(1, 0.0, 0.0, 0.0);
            setd(2, 0.0, 0.0, 0.0);
            setd(3, 0.0, 0.0, 0.0);
            setd(4, 0.0, 0.0, 0.0);
            setd(5, 0.0, 0.0, 0.0);
            setd(6, 0.0, 0.0, 0.0);
            break;
        default:  // moving apart
            setd(0, 0.0, gap * (0.5 + rng.unit()), 0.0);
            break;
    }
}
}// namespace

TEST_CASE("distance_device_smoke", "[luisa][distance][device]")
{
    test_runtime();

    Lcg                 rng;
    std::vector<double> host_in(static_cast<std::size_t>(kDevN) * kInputSize);
    for(int i = 0; i < kDevN; ++i) fill_input(rng, i, host_in);

    // host reference: the same routine, `S = double`
    std::vector<double> host_out(static_cast<std::size_t>(kDevN) * kOutSize, 0.0);
    for(int e = 0; e < kDevN; ++e)
    {
        const std::size_t base = static_cast<std::size_t>(e) * kInputSize;
        const auto        load = [&](int j) noexcept { return host_in[base + static_cast<std::size_t>(j)]; };
        uint32_t          k    = 0;
        const auto        put  = [&](const auto& v) noexcept
        { host_out[static_cast<std::size_t>(e) * kOutSize + k] = static_cast<double>(v), ++k; };
        dist_calc<double>(load, put, host_in[base + kEtaAt], host_in[base + kThickAt]);
    }

    lud::LDeviceBuffer<double> dev_in, dev_out;
    dev_in.resize(host_in.size());
    dev_out.resize(static_cast<std::size_t>(kDevN) * kOutSize);
    dev_in.copy_from(luisa::span<const double>{host_in});

    static auto kernel = lud::compile1<lcrt::Buffer<double>, lcrt::Buffer<double>, uint32_t>(
        [](lcrt::BufferVar<double> out, lcrt::BufferVar<double> in, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                const auto load = [&](int j) noexcept
                { return in->read(i * static_cast<uint32_t>(kInputSize) + static_cast<uint32_t>(j)); };
                uint32_t   k   = 0;
                const auto put = [&](const auto& v) noexcept
                { out->write(i * static_cast<uint32_t>(kOutSize) + k, v), ++k; };
                dist_calc<lcrt::Var<double>>(load, put, load(kEtaAt), load(kThickAt));
            };
        });

    lud::main_stream() << kernel(dev_out.view(), dev_in.view(), kDevN).dispatch(kDevN)
                       << lcrt::synchronize();

    std::vector<double> got(dev_out.size(), 0.0);
    dev_out.view().copy_to(luisa::span{got});
    lud::main_stream().synchronize();

    uint32_t bad = 0u;
    for(int e = 0; e < kDevN; ++e)
    {
        for(int k = 0; k < kOutSize; ++k)
        {
            const std::size_t idx = static_cast<std::size_t>(e) * kOutSize + k;
            if(!close(got[idx], host_out[idx], 1e-8))
            {
                if(bad < 10u)
                    std::printf(" [device] elem %d slot %d gpu=% .17g cpu=% .17g\n",
                                e,
                                k,
                                got[idx],
                                host_out[idx]);
                ++bad;
            }
        }
    }
    std::printf("    [device smoke] %d outputs per element, %u mismatches (1e-8 relative)\n",
                kOutSize,
                bad);
    CHECK(bad == 0u);
}
