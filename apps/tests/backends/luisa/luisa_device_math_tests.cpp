/********************************************************************
 * @file luisa_device_math_tests.cpp
 * @brief T2.1 — tests for the generic math layer (math/vec.h, math/mat.h).
 *
 * Three groups:
 *  - host semantics: storage layout, strided views (segment/block/col/row/
 *    diagonal), assignment-through-views, geometry helpers;
 *  - host parity with Eigen (the plan's acceptance criterion: 1e-6 on
 *    representative Vector9 / Matrix9x9 operations). Only compiled when Eigen is
 *    reachable, so the standalone harness still builds without vcpkg;
 *  - a DSL smoke kernel: `smoke_calc<S>` is one generic routine instantiated with
 *    `S = double` (host reference) and `S = lcrt::Var<double>` (GPU). The GPU
 *    output must match the host evaluation bit-for-bit — this is the first unit
 *    of the M9 equivalence harness, and it exercises exactly the patterns the
 *    ported constitutive code uses (default-construct + `segment<>` assignment,
 *    outer product, `block<>` assembly).
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/math/mat.h>

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
// Deterministic pseudo-random stream so host and device see identical inputs.
struct Lcg
{
    uint64_t s = 0x123456789abcdefULL;
    double   next() noexcept
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11) & 0x1fffffULL) / 1048576.0 - 1.0;  // [-1, 1)
    }
};

bool close(double a, double b, double tol = 1e-9)
{
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

template <int N>
LVec<double, N> make_vec(Lcg& rng)
{
    return lmath::make_vec<double, N>([&](int) { return rng.next(); });
}

template <int R, int C>
LMat<double, R, C> make_mat(Lcg& rng)
{
    return lmath::make_mat<double, R, C>([&](int, int) { return rng.next(); });
}
}// namespace

TEST_CASE("math_host_layout", "[luisa][math]")
{
    // Column-major storage, row-major constructor (Eigen-compatible):
    //   m(r, c) == data[c * R + r]  and  Matrix(a..i) fills a..i row by row.
    LMat<double, 3, 3> m{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    CHECK((close(m(0, 0), 1.0)));
    CHECK((close(m(0, 1), 2.0)));
    CHECK((close(m(0, 2), 3.0)));
    CHECK((close(m(1, 0), 4.0)));
    CHECK((close(m(1, 2), 6.0)));
    CHECK((close(m(2, 0), 7.0)));
    CHECK((close(m(2, 2), 9.0)));
    CHECK((close(m.data[0], 1.0)));  // (0,0)
    CHECK((close(m.data[1], 4.0)));  // (1,0) -- column-major
    CHECK((close(m.data[2], 7.0)));  // (2,0) -- column-major
    CHECK((close(m.data[3], 2.0)));  // (0,1) -- second column

    auto m9 = LMat<double, 9, 9>::Identity();
    CHECK((close(m9(4, 4), 1.0)));
    CHECK((close(m9(4, 5), 0.0)));
    CHECK((close(m9.data[4 * 9 + 4], 1.0)));
    CHECK((close(m9.data[4 * 9 + 5], 0.0)));

    // Vectors are zero-initialised by default (unlike Eigen), which is what the
    // DSL needs: a default-constructed Var is a zeroed local.
    LVec<double, 3> v;
    CHECK((close(v(0), 0.0) && close(v(1), 0.0) && close(v(2), 0.0)));
    CHECK((close(LVec<double, 3>::Ones().sum(), 3.0)));
    CHECK((close(LVec<double, 3>::Unit<2>()(2), 1.0)));
    CHECK((close(LVec<double, 3>::Unit<2>()(0), 0.0)));
    CHECK((close(LVec<double, 3>::Map(&v.data[0])(1), 0.0)));
}

TEST_CASE("math_host_views", "[luisa][math]")
{
    LVec<double, 12> q = LVec<double, 12>::Zero();
    LVec<double, 3>  a{1.0, 2.0, 3.0};
    LVec<double, 3>  b{4.0, 5.0, 6.0};

    // segment views write through to the parent (the `Vector9 ret; ret.segment<3>(i) = ..`
    // idiom of the ported constitutive functions).
    auto a1 = q.segment<3>(3);
    a1      = a;
    auto a2 = q.segment<3>(6);
    a2      = b;
    CHECK((close(q(3), 1.0) && close(q(5), 3.0)));
    CHECK((close(q(6), 4.0) && close(q(8), 6.0)));

    // a const parent yields a copy, so nothing is aliased
    const LVec<double, 12>& cq = q;
    auto                    s  = cq.segment<3>(3);
    s(0)                       = 99.0;
    CHECK((close(cq(3), 1.0)));

    auto h = q.head<3>();
    h      = a;
    CHECK((close(q(0), 1.0) && close(q(2), 3.0)));
    auto t = q.tail<2>();
    t      = LVec<double, 2>{7.0, 8.0};
    CHECK((close(q(10), 7.0) && close(q(11), 8.0)));

    // matrix blocks / rows / cols / diagonal
    Lcg                rng;
    LMat<double, 9, 9> H   = LMat<double, 9, 9>::Zero();
    auto               blk = make_mat<3, 3>(rng);
    H.block<3, 3>(0, 3)    = blk;
    CHECK((close(H(0, 3), blk(0, 0))));
    CHECK((close(H(2, 5), blk(2, 2))));
    CHECK((close(H(0, 0), 0.0)));
    CHECK((close(H(6, 6), 0.0)));

    auto col = H.col(3);
    col      = LVec<double, 9>::Ones();
    CHECK((close(H(4, 3), 1.0)));

    auto row = H.row(1);
    row      = LVec<double, 9>::Constant(5.0);
    CHECK((close(H(1, 7), 5.0)));

    auto dg = H.diagonal();
    dg      = LVec<double, 9>::Constant(3.0);
    CHECK((close(H(6, 6), 3.0)));
    CHECK((close(H(6, 7), 0.0)));

    // `.array() += scalar` through a view (used by the contact code)
    LMat<double, 2, 2> S = LMat<double, 2, 2>::Zero();
    S.diagonal().array() += 2.5;
    CHECK((close(S(0, 0), 2.5) && close(S(1, 1), 2.5) && close(S(0, 1), 0.0)));
}

TEST_CASE("math_host_ops", "[luisa][math]")
{
    Lcg  rng;
    auto a = make_vec<3>(rng);
    auto b = make_vec<3>(rng);

    CHECK((close(a.dot(b), a(0) * b(0) + a(1) * b(1) + a(2) * b(2))));
    auto c = lmath::cross(a, b);
    CHECK((close(c(0), a(1) * b(2) - a(2) * b(1))));
    CHECK((close(c(1), a(2) * b(0) - a(0) * b(2))));
    CHECK((close(c(2), a(0) * b(1) - a(1) * b(0))));

    auto n = a.norm();
    CHECK((close(n, std::sqrt(a.dot(a)), 1e-12)));
    auto u = a.normalized();
    CHECK((close(u.norm(), 1.0, 1e-12)));

    // outer product `a * b.transpose()` and matrix-vector product
    auto M  = a * b.transpose();
    CHECK((close(M(1, 2), a(1) * b(2))));
    auto Mv = M * a;
    CHECK((close(Mv(1), a(1) * a.dot(b), 1e-12)));

    // Column-matrix -> vector conversion (Eigen allows it implicitly)
    LVec<double, 3> colvec = M.col(0);
    CHECK((close(colvec(2), M(2, 0))));

    // Array (element-wise) algebra
    auto e = a.array() * b.array();
    CHECK((close(e(2), a(2) * b(2))));
    auto f = a.array() / b.array();
    CHECK((close(f(1), a(1) / b(1))));
    auto g = a.array() * 2.0;
    CHECK((close(g(0), a(0) * 2.0)));
    auto d = (a - b).array().abs().max();
    CHECK((close(d, std::max({std::abs(a(0) - b(0)), std::abs(a(1) - b(1)), std::abs(a(2) - b(2))}), 1e-12)));

    LMat<double, 3, 3> T = M;
    CHECK((close(T.trace(), M(0, 0) + M(1, 1) + M(2, 2))));
    LMat<double, 3, 3> D = a.asDiagonal();
    CHECK((close(D(1, 1), a(1)) && close(D(1, 0), 0.0)));
    CHECK((close(D.norm(), std::sqrt(a(0) * a(0) + a(1) * a(1) + a(2) * a(2)), 1e-12)));
}

// Guarded on the *libuipc* header (not just Eigen): the standalone harness links
// no libuipc, so this block is skipped there and only the Eigen-free cases run.
#if __has_include(<uipc/common/type_define.h>)
// `cross` / `asDiagonal` live in Eigen/Geometry (Eigen/Core alone would leave it
// declared but undefined).
#include <Eigen/Geometry>
#include <uipc/common/type_define.h>

namespace
{
template <typename L, typename E>
void expect_vec_equal(const L& l, const E& e, double tol, const char* what)
{
    bool ok = true;
    for(int i = 0; i < static_cast<int>(e.size()); ++i)
        ok = ok && close(l(i), e(i), tol);
    if(!ok)
    {
        std::printf("    [%s] vector mismatch\n", what);
        for(int i = 0; i < static_cast<int>(e.size()); ++i)
            std::printf("      [%d] l=% .17g e=% .17g\n", i, l(i), e(i));
    }
    CHECK(ok);
}

template <typename L, typename E>
void expect_mat_equal(const L& l, const E& e, double tol, const char* what)
{
    bool ok = true;
    for(int c = 0; c < static_cast<int>(e.cols()); ++c)
        for(int r = 0; r < static_cast<int>(e.rows()); ++r)
            ok = ok && close(l(r, c), e(r, c), tol);
    if(!ok)
    {
        std::printf("    [%s] matrix mismatch\n", what);
        for(int r = 0; r < static_cast<int>(e.rows()); ++r)
            for(int c = 0; c < static_cast<int>(e.cols()); ++c)
                std::printf("      [%d,%d] l=% .17g e=% .17g\n", r, c, l(r, c), e(r, c));
    }
    CHECK(ok);
}
}// namespace

TEST_CASE("math_host_eigen_parity", "[luisa][math]")
{
    constexpr double tol = 1e-6;

    Lcg            rng;
    LVec<double, 9> a = make_vec<9>(rng);
    LVec<double, 9> b = make_vec<9>(rng);
    uipc::Vector9   ea;
    uipc::Vector9   eb;
    for(int i = 0; i < 9; ++i)
    {
        ea(i) = a(i);
        eb(i) = b(i);
    }

    // --- Vector9 -----------------------------------------------------------
    expect_vec_equal(a + b, ea + eb, tol, "v + v");
    expect_vec_equal(a - b, ea - eb, tol, "v - v");
    expect_vec_equal(a * 2.5, ea * 2.5, tol, "v * s");
    expect_vec_equal(-a, -ea, tol, "-v");
    expect_vec_equal(a / 3.0, ea / 3.0, tol, "v / s");

    CHECK((close(a.dot(b), ea.dot(eb), tol)));
    CHECK((close(a.squaredNorm(), ea.squaredNorm(), tol)));
    CHECK((close(a.norm(), ea.norm(), tol)));
    expect_vec_equal(a.normalized(), ea.normalized(), tol, "normalized()");
    expect_vec_equal(a.array() * b.array(), (ea.array() * eb.array()).matrix(), tol, "array product");
    expect_vec_equal(a.segment<3>(3), ea.segment<3>(3), tol, "segment<3>(3)");
    expect_vec_equal(a.tail<4>(), ea.tail<4>(), tol, "tail<4>");
    expect_vec_equal(a.head<4>(), ea.head<4>(), tol, "head<4>");
    CHECK((close(a.maxCoeff(), ea.maxCoeff(), tol)));
    CHECK((close(a.minCoeff(), ea.minCoeff(), tol)));

    LVec<double, 3>    v3{a(0), a(1), a(2)};
    LVec<double, 3>    w3{b(0), b(1), b(2)};
    uipc::Vector3      e3{a(0), a(1), a(2)};
    uipc::Vector3      f3{b(0), b(1), b(2)};
    expect_vec_equal(lmath::cross(v3, w3), e3.cross(f3), tol, "cross");
    expect_vec_equal(v3.asDiagonal().diagonal(), e3.asDiagonal().diagonal(), tol, "asDiagonal");

    // --- Matrix9x9 ---------------------------------------------------------
    LMat<double, 9, 9> A = make_mat<9, 9>(rng);
    LMat<double, 9, 9> B = make_mat<9, 9>(rng);
    uipc::Matrix9x9    eA;
    uipc::Matrix9x9    eB;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
        {
            eA(r, c) = A(r, c);
            eB(r, c) = B(r, c);
        }

    expect_mat_equal(A + B, eA + eB, tol, "M + M");
    expect_mat_equal(A - B, eA - eB, tol, "M - M");
    expect_mat_equal(A * 1.25, eA * 1.25, tol, "M * s");
    expect_mat_equal(A.transpose(), eA.transpose(), tol, "transpose");
    expect_mat_equal(A.block<3, 3>(3, 6), eA.block<3, 3>(3, 6), tol, "block<3,3>(3,6)");
    expect_mat_equal(A * B, eA * eB, tol, "M * M");
    expect_vec_equal(A * a, eA * ea, tol, "M * v");
    expect_mat_equal(a * b.transpose(), ea * eb.transpose(), tol, "outer product");
    CHECK((close(A.trace(), eA.trace(), tol)));
    expect_vec_equal(A.diagonal(), eA.diagonal(), tol, "diagonal");
    expect_vec_equal(A.col(4), eA.col(4), tol, "col(4)");
    expect_vec_equal(A.row(4), eA.row(4), tol, "row(4)");
    expect_mat_equal(A.array() * B.array(), (eA.array() * eB.array()).matrix(), tol, "array product");

    // The symmetric 12x12 block structure of the affine-body hessians.
    LMat<double, 12, 12> S = make_mat<12, 12>(rng);
    S                      = S + S.transpose();
    CHECK((close(S(3, 7), S(7, 3))));
    CHECK((close(S.block<3, 3>(0, 0)(1, 2), S(1, 2))));

    LVec<double, 12>    q  = make_vec<12>(rng);
    uipc::Vector12      eq;
    for(int i = 0; i < 12; ++i) eq(i) = q(i);
    LMat<double, 9, 9>  H  = LMat<double, 9, 9>::Zero();
    uipc::Matrix9x9     eH = uipc::Matrix9x9::Zero();
    H.block<3, 3>(0, 0)    = (q.segment<3>(0) * q.segment<3>(3).transpose());
    eH.block<3, 3>(0, 0)   = (eq.segment<3>(0) * eq.segment<3>(3).transpose());
    expect_mat_equal(H, eH, tol, "H.block = outer(segment)");
}
#endif// __has_include(<uipc/common/type_define.h>)

// ---------------------------------------------------------------------------
//  Device smoke test: one generic routine, instantiated for host and device
// ---------------------------------------------------------------------------
namespace
{
constexpr uint32_t kDeviceN   = 64u;
constexpr uint32_t kDeviceOut = 24u;  // doubles written per element

/// Slot layout shared by the host reference and the kernel below.
/// (Plain `uint32_t` constants, not an unscoped enum: LC's `Expr<uint>` operators
/// do not accept an unnamed-enum operand.)
constexpr uint32_t kSlotQ    = 0u;   // 0..11  q
constexpr uint32_t kSlotC    = 12u;  // 12..14 cross(a,b)
constexpr uint32_t kSlotR    = 15u;  // 15..17 (a b^T) a
constexpr uint32_t kSlotD    = 18u;  // dot(a,b)
constexpr uint32_t kSlotNm   = 19u;  // |a+b|
constexpr uint32_t kSlotNrm0 = 20u;  // (a/|a| * |a|)(0)
constexpr uint32_t kSlotH00  = 21u;  // H(0,3) with H.block<3,3>(0,3) = a b^T

/**
 * @brief The shared generic computation.
 *
 * Instantiated with `S = double` (host reference) and `S = lcrt::Var<double>`
 * (inside the kernel). Both instantiations must agree exactly.
 */
template <typename S>
struct SmokeResult
{
    lmath::LVec<S, 12> q;
    lmath::LVec<S, 3>  c;
    lmath::LVec<S, 3>  r;
    S                  d;
    S                  nm;
    S                  nrm0;
    S                  h00;
};

template <typename S>
SmokeResult<S> smoke_calc(const lmath::LVec<S, 3>& a, const lmath::LVec<S, 3>& b)
{
    SmokeResult<S> out;

    // default-construct + assign through strided views (`Vector9 ret; ret.segment<3>(0) = ..`)
    out.q  = lmath::LVec<S, 12>::Zero();
    auto q0 = out.q.template segment<3>(0);
    q0      = a;
    auto q1 = out.q.template segment<3>(3);
    q1      = b;
    auto q2 = out.q.template segment<3>(6);
    q2      = a + b;
    auto q3 = out.q.template segment<3>(9);
    q3      = lmath::cross(a, b);

    out.c = lmath::cross(a, b);

    auto M = a * b.transpose();  // outer product
    out.r  = M * a;              // matrix * vector

    out.d  = lmath::dot(a, b);
    out.nm = (a + b).norm();

    auto un   = a.normalized() * a.norm();
    out.nrm0  = un(0);

    // 9x9 tile assembly + a 3x3 block read-back
    lmath::LMat<S, 9, 9> H         = lmath::LMat<S, 9, 9>::Zero();
    H.template block<3, 3>(0, 3)   = M;
    out.h00                        = H(0, 3);

    return out;
}

std::vector<double> smoke_reference(const std::vector<double>& in_a, const std::vector<double>& in_b)
{
    std::vector<double> out(kDeviceN * kDeviceOut, 0.0);
    for(uint32_t i = 0; i < kDeviceN; ++i)
    {
        LVec<double, 3> a{in_a[i * 3u + 0u], in_a[i * 3u + 1u], in_a[i * 3u + 2u]};
        LVec<double, 3> b{in_b[i * 3u + 0u], in_b[i * 3u + 1u], in_b[i * 3u + 2u]};
        auto            o = smoke_calc<double>(a, b);
        auto*           d = out.data() + i * kDeviceOut;
        for(int k = 0; k < 12; ++k) d[kSlotQ + k] = o.q(k);
        for(int k = 0; k < 3; ++k)
        {
            d[kSlotC + k] = o.c(k);
            d[kSlotR + k] = o.r(k);
        }
        d[kSlotD]    = o.d;
        d[kSlotNm]   = o.nm;
        d[kSlotNrm0] = o.nrm0;
        d[kSlotH00]  = o.h00;
    }
    return out;
}
}// namespace

TEST_CASE("math_device_smoke", "[luisa][math][device]")
{
    test_runtime();

    std::vector<double> host_a(kDeviceN * 3u);
    std::vector<double> host_b(kDeviceN * 3u);
    {
        Lcg rng;
        for(auto& v : host_a) v = rng.next();
        for(auto& v : host_b) v = rng.next();
    }

    lud::LDeviceBuffer<double> dev_a, dev_b, dev_out;
    dev_a.resize(host_a.size());
    dev_b.resize(host_b.size());
    dev_out.resize(kDeviceN * kDeviceOut);
    dev_a.copy_from(luisa::span<const double>{host_a});
    dev_b.copy_from(luisa::span<const double>{host_b});

    using DVec3 = lmath::LVec<lcrt::Var<double>, 3>;

    static auto kernel =
        lud::compile1<lcrt::Buffer<double>, lcrt::Buffer<double>, lcrt::Buffer<double>, uint32_t>(
            [](lcrt::BufferVar<double> out, lcrt::BufferVar<double> in_a, lcrt::BufferVar<double> in_b,
               lcrt::Var<uint32_t> n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if (i < n)
                {
                    DVec3 a{in_a->read(i * 3u + 0u), in_a->read(i * 3u + 1u), in_a->read(i * 3u + 2u)};
                    DVec3 b{in_b->read(i * 3u + 0u), in_b->read(i * 3u + 1u), in_b->read(i * 3u + 2u)};

                    // the very same generic routine the host reference calls
                    auto o = smoke_calc<lcrt::Var<double>>(a, b);

                    for(int k = 0; k < 12; ++k)
                        out->write(i * 24u + static_cast<uint32_t>(kSlotQ + k), o.q(k));
                    for(int k = 0; k < 3; ++k)
                    {
                        out->write(i * 24u + static_cast<uint32_t>(kSlotC + k), o.c(k));
                        out->write(i * 24u + static_cast<uint32_t>(kSlotR + k), o.r(k));
                    }
                    out->write(i * 24u + kSlotD, o.d);
                    out->write(i * 24u + kSlotNm, o.nm);
                    out->write(i * 24u + kSlotNrm0, o.nrm0);
                    out->write(i * 24u + kSlotH00, o.h00);
                };
            });

    lud::main_stream() << kernel(dev_out.view(), dev_a.view(), dev_b.view(), kDeviceN).dispatch(kDeviceN)
                       << lcrt::synchronize();

    std::vector<double> got(dev_out.size(), 0.0);
    dev_out.view().copy_to(luisa::span{got});
    lud::main_stream().synchronize();

    const auto want = smoke_reference(host_a, host_b);

    uint32_t bad = 0u;
    for(uint32_t i = 0; i < kDeviceN; ++i)
    {
        for(uint32_t k = 0; k < kDeviceOut; ++k)
        {
            const double g = got[i * kDeviceOut + k];
            const double w = want[i * kDeviceOut + k];
            if(!close(g, w, 1e-12))
            {
                if(bad < 8u)
                    std::printf("    [device] elem %u slot %u gpu=% .17g cpu=% .17g\n", i, k, g, w);
                ++bad;
            }
        }
    }
    CHECK(bad == 0u);

    // Spot-check that the kernel really produced the expected structure:
    // H(0,3) == M(0,0) == a(0) * b(0) for element 0.
    CHECK((close(got[0 * kDeviceOut + kSlotH00], host_a[0] * host_b[0], 1e-12)));
    auto known = smoke_calc<double>(LVec<double, 3>::Ones(), LVec<double, 3>::Ones());
    CHECK((close(known.nm, std::sqrt(12.0), 1e-12)));
}

// ---------------------------------------------------------------------------
// Device double-precision accuracy of the transcendental builtins.
//
// Motivated by T5.2c: LC's `cuda_builtin/cuda_device_math.h` maps the *double*
// overloads to their float counterparts (logf/expf/acosf/... and powf for a
// non-integral exponent), and LC's **AST** codegen path does emit them
// (`cuda_codegen_ast.cpp` maps CallOp::LOG* to `lc_log*`).  This build uses the
// **LLVM** codegen path instead, which lowers the same ops to libdevice's double
// functions, so this test pins the observed behaviour: if a future LC upgrade
// switches codegen, the assertions below fail loudly instead of silently
// degrading every ported constitutive routine to ~1e-8 relative accuracy.
// (See docs/luisa_porting_cookbook.md sections 1.4/1.5.)
// ---------------------------------------------------------------------------
TEST_CASE("device_double_math_accuracy", "[luisa][device]")
{
    test_runtime();

    std::vector<double> in{0.5, 1.3, 2.7, 0.9};
    constexpr uint32_t  K = 5;
    const uint32_t      n = static_cast<uint32_t>(in.size());
    std::vector<double> out(in.size() * K, 0.0);

    lud::LDeviceBuffer<double> dev_in, dev_out;
    dev_in.copy_from(luisa::span{in});
    dev_out.resize(in.size() * K);
    dev_out.view().fill(0.0);

    static auto kernel = lud::compile1<lcrt::Buffer<double>, lcrt::Buffer<double>, uint32_t>(
        [](lcrt::BufferVar<double> x, lcrt::BufferVar<double> o, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto v = x->read(i);
                o->write(i * 5u + 0u, lcrt::log(v));
                o->write(i * 5u + 1u, lcrt::pow(v, 0.5));
                o->write(i * 5u + 2u, lcrt::exp(v));
                o->write(i * 5u + 3u, lcrt::acos(v));
                o->write(i * 5u + 4u, lcrt::sin(v));
            };
        });

    lud::main_stream() << kernel(dev_in.view(), dev_out.view(), n).dispatch(n)
                       << lcrt::synchronize();
    dev_out.view().copy_to(luisa::span{out});
    lud::main_stream().synchronize();

    // Double precision is ~1e-16 and the float fallback is ~1e-7..1e-8, so a 1e-13
    // bound separates the two unambiguously.  acos is undefined outside [-1, 1], so
    // the out-of-domain inputs must compare as "both NaN" rather than by a ratio.
    for(size_t i = 0; i < in.size(); ++i)
    {
        const double v = in[i];
        const double host[K] = {std::log(v), std::pow(v, 0.5), std::exp(v), std::acos(v), std::sin(v)};
        for(uint32_t k = 0; k < K; ++k)
        {
            const double got = out[i * K + k];
            if(std::isnan(host[k]) || std::isnan(got))
            {
                CHECK((std::isnan(host[k]) == std::isnan(got)));
                continue;
            }
            const double rel = std::abs(got - host[k]) / std::max(1e-300, std::abs(host[k]));
            if(!(rel <= 1e-13))
                std::printf(" [mathacc] i=%zu k=%u x=%.17g dev=%.17g host=%.17g rel=%.3e\n",
                            i, k, v, got, host[k], rel);
            CHECK(rel <= 1e-13);
        }
    }
}
