/********************************************************************
 * @file luisa_device_matrix_tests.cpp
 * @brief T2.2 — tests for determinant/inverse/make_spd (matrix_ops.h),
 *        Jacobi eigen decomposition (evd.h) and the 3x3 SVD/polar
 *        decomposition (svd.h).
 *
 * Structure mirrors luisa_device_math_tests.cpp:
 *  - host semantics + parity against Eigen (the plan's acceptance criteria:
 *    `make_spd` output must have min eigenvalue >= -1e-8 on 200 random symmetric
 *    matrices, and the SVD must reproduce known polar decompositions);
 *  - `mat_calc<S>` is ONE generic routine instantiated with `S = double` (host
 *    reference) and `S = lcrt::Var<double>` (GPU); the two must agree.
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/math/svd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace lmath = uipc::backend::luisa::luisa_device::math;
namespace lcrt  = ::luisa::compute;

using lmath::LMat;
using lmath::LVec;

namespace
{
struct Lcg
{
    uint64_t s = 0x0fedcba987654321ULL;
    double   next() noexcept
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11) & 0x1fffffULL) / 1048576.0 - 1.0;  // [-1, 1)
    }
    double positive() noexcept { return 0.5 + 0.5 * std::abs(next()); }
};

bool close(double a, double b, double tol)
{
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

template <int R, int C>
LMat<double, R, C> random_mat(Lcg& rng)
{
    return lmath::make_mat<double, R, C>([&](int, int) { return rng.next(); });
}

/// Symmetric with the given (per-index) scale; `well_conditioned` keeps it PSD-ish.
template <int N>
LMat<double, N, N> random_symmetric(Lcg& rng)
{
    auto M = random_mat<N, N>(rng);
    return M + M.transpose();
}
}// namespace

TEST_CASE("matrix_ops_host", "[luisa][matrix]")
{
    Lcg rng;
    // 2x2 / 3x3 / 4x4 determinant + inverse
    {
        LMat<double, 2, 2> A = random_mat<2, 2>(rng);
        A(0, 0) += 3.0;  // keep it well conditioned
        A(1, 1) += 3.0;
        CHECK(close(lmath::determinant(A), A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0), 1e-12));
        auto I = A * lmath::inverse(A);
        CHECK(close(I(0, 0), 1.0, 1e-12));
        CHECK(close(I(1, 1), 1.0, 1e-12));
        CHECK(close(I(0, 1), 0.0, 1e-12));
    }
    {
        LMat<double, 3, 3> A = random_mat<3, 3>(rng);
        for(int i = 0; i < 3; ++i) A(i, i) += 3.0;
        auto inv = lmath::inverse(A);
        auto I   = A * inv;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                CHECK(close(I(r, c), r == c ? 1.0 : 0.0, 1e-11));
        // member spellings agree with the free functions
        auto I2 = A.inverse();
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c) CHECK(close(I2(r, c), inv(r, c), 1e-14));
        CHECK(close(A.determinant(), lmath::determinant(A), 1e-14));
    }
    {
        LMat<double, 4, 4> A = random_mat<4, 4>(rng);
        for(int i = 0; i < 4; ++i) A(i, i) += 4.0;
        auto   I     = A * lmath::inverse(A);
        double worst = 0.0;
        for(int r = 0; r < 4; ++r)
            for(int c = 0; c < 4; ++c)
                worst = std::max(worst, std::abs(I(r, c) - (r == c ? 1.0 : 0.0)));
        std::printf("    [inverse4x4] worst |A A^-1 - I| = %.3e\n", worst);
        CHECK(worst < 1e-12);
    }
}

namespace
{
/// Eigenvalues of a symmetric matrix, plus the reconstruction V diag(l) V^T.
template <int N>
void check_evd(const LMat<double, N, N>& A0, double tol)
{
    LVec<double, N>    values;
    LMat<double, N, N> vectors;
    lmath::jacobi_evd(A0, values, vectors, /*sort=*/true);

    bool ordered = true;
    for(int i = 1; i < N; ++i) ordered = ordered && values(i - 1) <= values(i);
    CHECK(ordered);

    auto D = values.asDiagonal();
    auto R = vectors * D * vectors.transpose();
    for(int r = 0; r < N; ++r)
        for(int c = 0; c < N; ++c) CHECK(close(R(r, c), A0(r, c), tol));

    auto G = vectors.transpose() * vectors;
    for(int r = 0; r < N; ++r)
        for(int c = 0; c < N; ++c) CHECK(close(G(r, c), r == c ? 1.0 : 0.0, 1e-10));
}
}// namespace

TEST_CASE("evd_host", "[luisa][matrix]")
{
    Lcg rng;
    check_evd(random_symmetric<3>(rng), 1e-11);
    check_evd(random_symmetric<6>(rng), 1e-9);
    check_evd(random_symmetric<9>(rng), 1e-8);
}

TEST_CASE("make_spd_host", "[luisa][matrix]")
{
    Lcg rng;
    // 200 random symmetric matrices: the projected matrix must be PSD, and the
    // projection must be the identity on already-PSD input.
    double worst_min_eigen = 1e300;
    double worst_frobenius  = 0.0;
    for(int iter = 0; iter < 200; ++iter)
    {
        LMat<double, 3, 3> H = random_symmetric<3>(rng);
        auto               P = lmath::make_spd(H);

        // symmetry
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                CHECK(close(P(r, c), P(c, r), 1e-12));

        // min eigenvalue of the result, via Jacobi on the result itself
        LVec<double, 3>    vals;
        LMat<double, 3, 3> vecs;
        lmath::jacobi_evd(P, vals, vecs, /*sort=*/true);
        worst_min_eigen = std::min(worst_min_eigen, vals(0));

        // the projection is idempotent
        auto PP = lmath::make_spd(P);
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                worst_frobenius = std::max(worst_frobenius, std::abs(PP(r, c) - P(r, c)));
    }
    std::printf("    [make_spd] worst min eigenvalue over 200 random symmetric 3x3 = %.3e\n",
                worst_min_eigen);
    std::printf("    [make_spd] worst |P(P(H)) - P(H)| = %.3e\n", worst_frobenius);
    CHECK(worst_min_eigen >= -1e-8);
    CHECK(worst_frobenius < 1e-10);

    // PSD input is already fixed: use M M^T
    auto M = random_mat<3, 3>(rng);
    auto G = M * M.transpose();
    auto P = lmath::make_spd(G);
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            CHECK(close(P(r, c), G(r, c), 1e-10));

    // evd_jacobi's `sort` flag is exercised with the default (sorted) too
    LVec<double, 3>    vals;
    LMat<double, 3, 3> vecs;
    lmath::evd(G, vals, vecs);
    CHECK((vals(0) <= vals(1) && vals(1) <= vals(2)));
}

TEST_CASE("svd_host", "[luisa][matrix]")
{
    Lcg rng;
    for(int iter = 0; iter < 50; ++iter)
    {
        LMat<double, 3, 3> A = random_mat<3, 3>(rng);
        LMat<double, 3, 3> U, V;
        LVec<double, 3>    sigma;
        lmath::svd(A, U, sigma, V);

        // singular values are non-negative, sorted descending
        CHECK(sigma(0) >= sigma(1) - 1e-14);
        CHECK(sigma(1) >= sigma(2) - 1e-14);

        // U and V are proper rotations
        CHECK(close(lmath::determinant(U), 1.0, 1e-12));
        CHECK(close(lmath::determinant(V), 1.0, 1e-12));
        auto UtU = U.transpose() * U;
        auto VtV = V.transpose() * V;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
            {
                CHECK(close(UtU(r, c), r == c ? 1.0 : 0.0, 1e-10));
                CHECK(close(VtV(r, c), r == c ? 1.0 : 0.0, 1e-10));
            }

        // reconstruction A == U diag(sigma) V^T
        auto R = U * sigma.asDiagonal() * V.transpose();
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                CHECK(close(R(r, c), A(r, c), 1e-10));

        // the reference's spelling must be identical
        LMat<double, 3, 3> U2, V2;
        LVec<double, 3>    s2;
        lmath::qr_svd(A, s2, U2, V2);
        for(int i = 0; i < 3; ++i) CHECK(close(s2(i), sigma(i), 1e-14));
    }

    // A reflection (det < 0) keeps U, V proper and pushes the sign into sigma(2)
    LMat<double, 3, 3> F = LMat<double, 3, 3>::Identity();
    F(2, 2)               = -2.0;
    LMat<double, 3, 3> U, V;
    LVec<double, 3>    sigma;
    lmath::svd(F, U, sigma, V);
    CHECK(close(lmath::determinant(U), 1.0, 1e-12));
    CHECK(close(lmath::determinant(V), 1.0, 1e-12));
    CHECK(sigma(2) < 0.0);
    // The reflection is absorbed into the *smallest* singular value, so the
    // magnitudes are still the singular values of diag(1, 1, -2) = (2, 1, 1).
    CHECK(close(std::abs(sigma(0)), 2.0, 1e-12));
    CHECK(close(std::abs(sigma(2)), 1.0, 1e-12));
    auto Fr = U * sigma.asDiagonal() * V.transpose();
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c) CHECK(close(Fr(r, c), F(r, c), 1e-12));
}

TEST_CASE("pd_host", "[luisa][matrix]")
{
    Lcg rng;
    for(int iter = 0; iter < 50; ++iter)
    {
        // positive determinant, so the polar decomposition is unique
        LMat<double, 3, 3> A = random_mat<3, 3>(rng);
        for(int i = 0; i < 3; ++i) A(i, i) += 3.0;
        if(lmath::determinant(A) < 0.0)
            for(int r = 0; r < 3; ++r) A(r, 0) = -A(r, 0);

        LMat<double, 3, 3> R, S;
        lmath::pd(A, R, S);

        CHECK(close(lmath::determinant(R), 1.0, 1e-10));
        auto RtR = R.transpose() * R;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
            {
                CHECK(close(RtR(r, c), r == c ? 1.0 : 0.0, 1e-10));
                CHECK(close(S(r, c), S(c, r), 1e-10));  // S symmetric
            }

        auto RS = R * S;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                CHECK(close(RS(r, c), A(r, c), 1e-10));

        // S == sqrt(A^T A)
        auto AtA = A.transpose() * A;
        auto S2  = S * S;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                CHECK(close(S2(r, c), AtA(r, c), 1e-10));
    }
}

// Guarded on the *libuipc* header (not just Eigen): the standalone harness links
// no libuipc, so this block is skipped there.
#if __has_include(<uipc/common/type_define.h>)
// Eigen/Dense pulls in the decompositions (SelfAdjointEigenSolver, JacobiSVD);
// Eigen/Geometry covers cross/asDiagonal.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <uipc/common/type_define.h>

TEST_CASE("matrix_eigen_parity", "[luisa][matrix]")
{
    constexpr double tol = 1e-6;
    Lcg              rng;

    // --- determinant / inverse ---------------------------------------------
    {
        uipc::Matrix2x2 e;
        LMat<double, 2, 2> m = random_mat<2, 2>(rng);
        m(0, 0) += 3.0;
        m(1, 1) += 3.0;
        for(int c = 0; c < 2; ++c)
            for(int r = 0; r < 2; ++r) e(r, c) = m(r, c);
        CHECK(close(lmath::determinant(m), e.determinant(), tol));
        auto im = lmath::inverse(m);
        auto ie = e.inverse();
        for(int c = 0; c < 2; ++c)
            for(int r = 0; r < 2; ++r) CHECK(close(im(r, c), ie(r, c), tol));
    }
    {
        LMat<double, 3, 3> m = random_mat<3, 3>(rng);
        for(int i = 0; i < 3; ++i) m(i, i) += 3.0;
        uipc::Matrix3x3 e;
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r) e(r, c) = m(r, c);
        CHECK(close(lmath::determinant(m), e.determinant(), tol));
        auto im = lmath::inverse(m);
        auto ie = e.inverse();
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r) CHECK(close(im(r, c), ie(r, c), tol));
    }
    {
        LMat<double, 4, 4> m = random_mat<4, 4>(rng);
        for(int i = 0; i < 4; ++i) m(i, i) += 4.0;
        uipc::Matrix4x4 e;
        for(int c = 0; c < 4; ++c)
            for(int r = 0; r < 4; ++r) e(r, c) = m(r, c);
        CHECK(close(lmath::determinant(m), e.determinant(), tol));
        auto im = lmath::inverse(m);
        auto ie = e.inverse();
        for(int c = 0; c < 4; ++c)
            for(int r = 0; r < 4; ++r) CHECK(close(im(r, c), ie(r, c), tol));
    }

    // --- symmetric eigen decomposition -------------------------------------
    {
        LMat<double, 9, 9> m = random_symmetric<9>(rng);
        uipc::Matrix9x9    e;
        for(int c = 0; c < 9; ++c)
            for(int r = 0; r < 9; ++r) e(r, c) = m(r, c);

        Eigen::SelfAdjointEigenSolver<uipc::Matrix9x9> solver(e);

        LVec<double, 9>    values;
        LMat<double, 9, 9> vectors;
        lmath::jacobi_evd(m, values, vectors, /*sort=*/true);

        for(int i = 0; i < 9; ++i)
            CHECK(close(values(i), solver.eigenvalues()(i), tol));

        // Eigenvectors are only defined up to sign; compare the absolute overlap.
        for(int i = 0; i < 9; ++i)
        {
            double acc = 0.0;
            for(int r = 0; r < 9; ++r) acc += vectors(r, i) * solver.eigenvectors()(r, i);
            CHECK(close(std::abs(acc), 1.0, 1e-6));
        }
    }

    // --- SVD / polar decomposition -----------------------------------------
    for(int iter = 0; iter < 20; ++iter)
    {
        LMat<double, 3, 3> A = random_mat<3, 3>(rng);
        for(int i = 0; i < 3; ++i) A(i, i) += 3.0;
        if(lmath::determinant(A) < 0.0)
            for(int r = 0; r < 3; ++r) A(r, 0) = -A(r, 0);

        uipc::Matrix3x3 e;
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r) e(r, c) = A(r, c);

        Eigen::JacobiSVD<uipc::Matrix3x3> svd(e, Eigen::ComputeFullU | Eigen::ComputeFullV);

        LMat<double, 3, 3> U, V;
        LVec<double, 3>    sigma;
        lmath::svd(A, U, sigma, V);

        // singular values match (both sorted descending; ours keeps sigma(2)
        // possibly negative, so compare magnitudes)
        for(int i = 0; i < 3; ++i)
            CHECK(close(std::abs(sigma(i)), svd.singularValues()(i), tol));

        // polar decomposition vs Eigen's
        auto eR = svd.matrixU() * svd.matrixV().transpose();
        auto eSigma = svd.singularValues();
        auto eS = svd.matrixV() * eSigma.asDiagonal() * svd.matrixV().transpose();

        LMat<double, 3, 3> R, S;
        lmath::pd(A, R, S);
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
            {
                CHECK(close(R(r, c), eR(r, c), tol));
                CHECK(close(S(r, c), eS(r, c), tol));
            }
    }
}
#endif// __has_include(<uipc/common/type_define.h>)

// ---------------------------------------------------------------------------
//  Device smoke test: one generic routine, instantiated for host and device
// ---------------------------------------------------------------------------
namespace
{
constexpr uint32_t kDevN   = 32u;
constexpr uint32_t kDevOut = 64u;  // doubles per element

/// Shared generic computation, instantiated with `S = double` (host reference)
/// and `S = lcrt::Var<double>` (kernel). Every branch-free routine in the
/// numerics layer is exercised here.
template <typename S>
struct MatResult
{
    lmath::LMat<S, 3, 3> spd;      // make_spd(sym)
    lmath::LMat<S, 3, 3> inv;      // inverse(A)
    lmath::LMat<S, 3, 3> U, V;     // SVD
    lmath::LMat<S, 3, 3> R, Sq;    // polar
    lmath::LVec<S, 3>    sigma;
    S                    det;
    lmath::LVec<S, 3>    eig;      // sorted eigenvalues of sym
};

template <typename S>
MatResult<S> mat_calc(const lmath::LMat<S, 3, 3>& A)
{
    MatResult<S> out;

    auto sym = A + A.transpose();

    out.spd = lmath::make_spd(sym);

    out.inv = lmath::inverse(A);
    out.det = lmath::determinant(A);

    lmath::LMat<S, 3, 3> vecs;
    lmath::jacobi_evd(sym, out.eig, vecs, /*sort=*/true);

    lmath::svd(A, out.U, out.sigma, out.V);
    lmath::pd(A, out.R, out.Sq);

    return out;
}

std::vector<double> mat_reference(const std::vector<double>& in)
{
    std::vector<double> out(kDevN * kDevOut, 0.0);
    for(uint32_t i = 0; i < kDevN; ++i)
    {
        LMat<double, 3, 3> A;
        for(int k = 0; k < 9; ++k) A.data[static_cast<std::size_t>(k)] = in[i * 9u + k];
        auto  o = mat_calc<double>(A);
        auto* d = out.data() + i * kDevOut;
        for(int k = 0; k < 9; ++k)
        {
            d[0 + k]  = o.spd.data[static_cast<std::size_t>(k)];
            d[9 + k]  = o.inv.data[static_cast<std::size_t>(k)];
            d[18 + k] = o.U.data[static_cast<std::size_t>(k)];
            d[27 + k] = o.V.data[static_cast<std::size_t>(k)];
            d[36 + k] = o.R.data[static_cast<std::size_t>(k)];
            d[45 + k] = o.Sq.data[static_cast<std::size_t>(k)];
        }
        for(int k = 0; k < 3; ++k)
        {
            d[54 + k] = o.sigma(k);
            d[57 + k] = o.eig(k);
        }
        d[60] = o.det;
    }
    return out;
}
}// namespace

TEST_CASE("matrix_device_smoke", "[luisa][matrix][device]")
{
    test_runtime();

    std::vector<double> host_in(kDevN * 9u);
    {
        Lcg rng;
        for(uint32_t i = 0; i < kDevN; ++i)
            for(int k = 0; k < 9; ++k) host_in[i * 9u + k] = rng.next();
        // keep the matrices well conditioned (the reference does the same)
        for(uint32_t i = 0; i < kDevN; ++i)
            for(int k = 0; k < 3; ++k) host_in[i * 9u + k * 3u + k] += 3.0;
    }

    lud::LDeviceBuffer<double> dev_in, dev_out;
    dev_in.resize(host_in.size());
    dev_out.resize(kDevN * kDevOut);
    dev_in.copy_from(luisa::span<const double>{host_in});

    using DMat3 = lmath::LMat<lcrt::Var<double>, 3, 3>;

    static auto kernel = lud::compile1<lcrt::Buffer<double>, lcrt::Buffer<double>, uint32_t>(
        [](lcrt::BufferVar<double> out, lcrt::BufferVar<double> in, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n)
            {
                DMat3 A;
                for(int k = 0; k < 9; ++k)
                    A.data[static_cast<std::size_t>(k)] = in->read(i * 9u + static_cast<uint32_t>(k));

                auto o = mat_calc<lcrt::Var<double>>(A);

                for(int k = 0; k < 9; ++k)
                {
                    auto kk = static_cast<uint32_t>(k);
                    out->write(i * 64u + 0u + kk, o.spd.data[static_cast<std::size_t>(k)]);
                    out->write(i * 64u + 9u + kk, o.inv.data[static_cast<std::size_t>(k)]);
                    out->write(i * 64u + 18u + kk, o.U.data[static_cast<std::size_t>(k)]);
                    out->write(i * 64u + 27u + kk, o.V.data[static_cast<std::size_t>(k)]);
                    out->write(i * 64u + 36u + kk, o.R.data[static_cast<std::size_t>(k)]);
                    out->write(i * 64u + 45u + kk, o.Sq.data[static_cast<std::size_t>(k)]);
                }
                for(int k = 0; k < 3; ++k)
                {
                    auto kk = static_cast<uint32_t>(k);
                    out->write(i * 64u + 54u + kk, o.sigma(k));
                    out->write(i * 64u + 57u + kk, o.eig(k));
                }
                out->write(i * 64u + 60u, o.det);
            };
        });

    lud::main_stream() << kernel(dev_out.view(), dev_in.view(), kDevN).dispatch(kDevN)
                       << lcrt::synchronize();

    std::vector<double> got(dev_out.size(), 0.0);
    dev_out.view().copy_to(luisa::span{got});
    lud::main_stream().synchronize();

    const auto want = mat_reference(host_in);

    uint32_t bad = 0u;
    for(uint32_t i = 0; i < kDevN; ++i)
    {
        for(uint32_t k = 0; k < 61u; ++k)  // 0..60 are written
        {
            const double g = got[i * kDevOut + k];
            const double w = want[i * kDevOut + k];
            // Same routine, two compilers: NVRTC and cl.exe differ at the
            // 1e-10 level through FMA contraction, so 1e-8 is the meaningful
            // bound for "identical algorithms, different back ends".
            if(!close(g, w, 1e-8))
            {
                if(bad < 10u)
                    std::printf("    [device] elem %u slot %u gpu=% .17g cpu=% .17g\n", i, k, g, w);
                ++bad;
            }
        }
    }
    CHECK(bad == 0u);
}
