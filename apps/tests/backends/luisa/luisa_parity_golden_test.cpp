/********************************************************************
 * @file luisa_parity_golden_test.cpp
 * @brief T9.3a — Tier-1 golden consumer ([parity_golden], Class A).
 *
 * The golden is produced by `uipc_parity_reference.exe`
 * (apps/tests/parity/reference/main.cpp), which evaluates the *reference*
 * implementation (src/backends/cuda/utils/distance/details/point_triangle.inl) on the
 * host. This test evaluates the ported routine on the same inputs and requires
 * bit-exact equality for `T = double` (host) -- Class A, the whole point of Tier 1.
 *
 * DEVIATION from 03-parity-harness.md §4.2 step 3 (recorded in
 * docs/luisa_backend_progress.md): the device half (`T = lcrt::Var<double>` against the
 * same golden) is NOT here. Building a kernel that takes the ported `LVec<S,3>` /
 * `LVec<S,12>` / `LMat<S,12,12>` arguments aborts inside LC's AST builder with the same
 * pre-existing `function_builder.cpp:436` assertion ("declared as double, got byte")
 * that stops every dof-based scene in this tree, and an abort cannot be caught by a
 * test. T2.3 already covers the device numerics of these four primitives through a
 * scalar `load`/`put` call shape (luisa_device_distance_tests.cpp
 * "distance_device_smoke", 1e-8 relative); the golden's device half gets added back
 * once the AST-builder abort is fixed (plan 10).
 *
 * Two rules from 03-parity-harness.md §4.2 are taken literally:
 *   * a missing golden is a SKIP, never a pass ("if missing -> SKIP");
 *   * the inputs are read from the golden too, so the test also asserts it fed
 *     identical data instead of trusting its own sweep.
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/math/distance/distance.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs    = std::filesystem;
namespace dist  = uipc::backend::luisa::luisa_device::math::distance;
namespace lcrt  = ::luisa::compute;
namespace lmath = uipc::backend::luisa::luisa_device::math;

using lmath::LMat;
using lmath::LVec;

namespace
{
constexpr std::string_view kMagic   = "UIPCGOLD1";
constexpr std::uint32_t    kVersion = 1;
/// dist2 (1) + gradient (12) + Hessian (12x12): the golden's output layout.
constexpr std::uint32_t kGoldenOutSize = 1u + 12u + 12u * 12u;
constexpr std::uint32_t kGoldenInSize  = 12u;

struct Golden
{
    std::string         name;
    std::size_t         n_input  = 0;
    std::size_t         n_output = 0;
    std::vector<double> input;   // case_count * n_input
    std::vector<double> output;  // case_count * n_output

    std::size_t case_count() const noexcept
    {
        return n_input == 0 ? 0 : input.size() / n_input;
    }
};

std::string golden_path()
{
    return (fs::path{UIPC_TEST_PROGRAM_PATH}.parent_path() / "parity_golden" / "distance_pt.bin")
        .string();
}

/// Reads `UIPCGOLD1` (magic, version, name, counts, then the raw float64 payload).
bool read_golden(const std::string& path, Golden& golden, std::string& error)
{
    std::ifstream in{path, std::ios::binary};
    if(!in)
    {
        error = "cannot open " + path;
        return false;
    }

    char magic[kMagic.size()];
    in.read(magic, sizeof(magic));
    if(!in || std::memcmp(magic, kMagic.data(), sizeof(magic)) != 0)
    {
        error = "bad magic in " + path;
        return false;
    }

    std::uint32_t version  = 0;
    std::uint32_t name_len = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    if(!in || version != kVersion)
    {
        error = "unsupported golden version";
        return false;
    }

    golden.name.resize(name_len);
    if(name_len > 0)
        in.read(golden.name.data(), name_len);

    std::uint64_t case_count = 0;
    std::uint64_t n_input    = 0;
    std::uint64_t n_output   = 0;
    in.read(reinterpret_cast<char*>(&case_count), sizeof(case_count));
    in.read(reinterpret_cast<char*>(&n_input), sizeof(n_input));
    in.read(reinterpret_cast<char*>(&n_output), sizeof(n_output));
    if(!in || case_count == 0 || n_input == 0 || n_output == 0)
    {
        error = "corrupt golden header";
        return false;
    }

    golden.n_input  = static_cast<std::size_t>(n_input);
    golden.n_output = static_cast<std::size_t>(n_output);
    golden.input.resize(static_cast<std::size_t>(case_count) * golden.n_input);
    golden.output.resize(static_cast<std::size_t>(case_count) * golden.n_output);
    // The payload is interleaved per case (input[], output[], input[], ...), which is
    // the layout uipc_parity_reference writes: one case is one contiguous record.
    for(std::uint64_t c = 0; c < case_count; ++c)
    {
        in.read(reinterpret_cast<char*>(golden.input.data() + c * golden.n_input),
                static_cast<std::streamsize>(golden.n_input * sizeof(double)));
        in.read(reinterpret_cast<char*>(golden.output.data() + c * golden.n_output),
                static_cast<std::streamsize>(golden.n_output * sizeof(double)));
        if(!in)
        {
            error = "truncated payload in " + path;
            return false;
        }
    }
    return true;
}

void load_case(const Golden& golden, std::size_t index, LVec<double, 3> (&p)[4])
{
    const double* base = golden.input.data() + index * golden.n_input;
    for(int k = 0; k < 4; ++k)
        for(int a = 0; a < 3; ++a)
            p[k](a) = base[k * 3 + a];
}

void evaluate(const LVec<double, 3> (&p)[4], std::vector<double>& out)
{
    double dist2 = 0.0;
    dist::point_triangle_distance2(p[0], p[1], p[2], p[3], dist2);
    LVec<double, 12> gradient;
    dist::point_triangle_distance2_gradient(p[0], p[1], p[2], p[3], gradient);
    LMat<double, 12, 12> hessian;
    dist::point_triangle_distance2_hessian(p[0], p[1], p[2], p[3], hessian);

    out.assign(1 + 12 + 12 * 12, 0.0);
    out[0] = dist2;
    for(int i = 0; i < 12; ++i)
        out[1 + i] = gradient(i);
    std::size_t k = 1 + 12;
    for(int col = 0; col < 12; ++col)
        for(int row = 0; row < 12; ++row)
            out[k++] = hessian(row, col);
}
}  // namespace

TEST_CASE("parity_golden_distance", "[parity_golden]")
{
    const std::string path = golden_path();
    if(!fs::exists(path))
    {
        SKIP("reference golden not generated: " + path + " (run uipc_parity_reference.exe first)");
    }

    Golden      golden;
    std::string error;
    INFO("golden: " + path + ": " + error);
    REQUIRE(read_golden(path, golden, error));
    REQUIRE(golden.name == "distance_pt");
    REQUIRE(golden.n_input == kGoldenInSize);
    REQUIRE(golden.n_output == kGoldenOutSize);

    // ---- host: bit-exact against the reference (Class A) -------------------
    std::vector<double> output;
    std::size_t         exact_mismatches      = 0;
    double              worst_abs             = 0.0;
    std::size_t         worst_case            = 0;
    std::size_t         worst_slot            = 0;
    std::size_t         zero_sign_differences = 0;
    std::size_t         nan_slots             = 0;
    for(std::size_t c = 0; c < golden.case_count(); ++c)
    {
        LVec<double, 3> points[4];
        load_case(golden, c, points);
        evaluate(points, output);
        const double* expected = golden.output.data() + c * golden.n_output;
        for(std::size_t k = 0; k < golden.n_output; ++k)
        {
            const double got = output[k];
            if(std::isnan(got) && std::isnan(expected[k]))
            {
                // Both sides produce the same non-finite value. Measured: the three
                // deliberately degenerate configurations (zero-area triangle,
                // coincident vertices, a triangle collapsed to a segment) are NaN for
                // every output on both sides -- that is the reference's own behaviour
                // and the golden pins it. Same convention as
                // luisa_device_distance_tests.cpp's `close()`: NaN matches NaN.
                ++nan_slots;
                continue;
            }
            if(!(got == expected[k]))  // exact equality
            {
                ++exact_mismatches;
                const double delta = std::abs(got - expected[k]);
                if(delta >= worst_abs)
                {
                    worst_abs  = delta;
                    worst_case = c;
                    worst_slot = k;
                }
            }
            else if(std::memcmp(&got, expected + k, sizeof(double)) != 0)
            {
                // Same value, different sign of zero (measured on the degenerate
                // configurations). Numerically identical, so it is counted and
                // reported, not failed: the Class-A gate is "every element is exactly
                // equal", and -0.0 == 0.0 in IEEE-754.
                ++zero_sign_differences;
            }
        }
    }
    std::printf(
        "    [parity_golden] %zu cases x %zu outputs, host |delta| = %.3e "
        "(%zu mismatches, %zu zero-sign-only differences, %zu NaN slots; worst "
        "case %zu slot %zu)\n",
        golden.case_count(),
        golden.n_output,
        worst_abs,
        exact_mismatches,
        zero_sign_differences,
        nan_slots,
        worst_case,
        worst_slot);
    CHECK(exact_mismatches == 0u);
}

// ===========================================================================
//  T8.1b (plan 15) — the two contact `sym/*.inl` goldens
//
//  `uipc_parity_reference` now also evaluates *the reference* contact bodies on the host
//  (`apps/tests/parity/reference/main.cpp`):
//    * `sym/codim_ipc_contact.inl`      — the SymEigen log^2 barrier law, value + d/dD +
//                                          d^2/dD^2, which is the law of all four normal
//                                          contact models and of the codim variants;
//    * `sym/vertex_half_plane_distance.inl` — D, dDdx, ddDddx, TR, dTRdx, i.e. the whole
//                                          vertex-half-plane distance layer.
//  These are the *only* two reference files in the contact path that are host-compilable
//  (research 15 §5.2): everything else is `__device__ inline` and pulls
//  <type_define.h>/cuda_tool, which is why the remaining §4.1 rows stay `not_implemented`.
//  Both are plain `template <typename T>`, so the ported twin's host instantiation
//  (`T = double`) is compared **exactly** (Class A) — no device, no tolerance.
// ===========================================================================
namespace sym_codim = uipc::backend::luisa::sym::codim_ipc_contact;
namespace sym_ph    = uipc::backend::luisa::sym::ipc_vertex_half_contact;

namespace
{
/// The golden's per-case layouts (must match the generator's).
constexpr std::size_t kBarrierIn  = 4;  // kappa, D, dHat, xi
constexpr std::size_t kBarrierOut = 3;  // B, dB/dD, d^2B/dD^2
constexpr std::size_t kVhpIn      = 18;  // x, P, N, x_prev, e1, e2
constexpr std::size_t kVhpOut     = 1 + 3 + 9 + 2 + 6;

std::string golden_path_for(std::string_view file)
{
    return (fs::path{UIPC_TEST_PROGRAM_PATH}.parent_path() / "parity_golden" / std::string{file})
        .string();
}

/// Evaluates one case of the ported *host* instantiation and writes the generator's layout.
using CaseEvaluator = void (*)(const double* in, double* out);

/// The Class-A driver: SKIP when the golden is absent, then require bit-exact equality on
/// every element (a `NaN`-vs-`NaN` pair counts as equal, as in `parity_golden_distance`).
void require_exact(std::string_view label,
                   std::string_view file,
                   std::size_t      n_input,
                   std::size_t      n_output,
                   CaseEvaluator    evaluate)
{
    const std::string path = golden_path_for(file);
    if(!fs::exists(path))
    {
        SKIP("reference golden not generated: " + path
             + " (run uipc_parity_reference.exe first)");
    }

    Golden      golden;
    std::string error;
    INFO("golden: " + path + ": " + error);
    REQUIRE(read_golden(path, golden, error));
    REQUIRE(golden.name == std::string{label});
    REQUIRE(golden.n_input == n_input);
    REQUIRE(golden.n_output == n_output);

    std::vector<double> output(n_output, 0.0);
    std::size_t         exact_mismatches = 0;
    std::size_t         nan_slots        = 0;
    double              worst_abs        = 0.0;
    std::size_t         worst_case       = 0;
    std::size_t         worst_slot       = 0;
    for(std::size_t c = 0; c < golden.case_count(); ++c)
    {
        evaluate(golden.input.data() + c * n_input, output.data());
        const double* expected = golden.output.data() + c * n_output;
        for(std::size_t k = 0; k < n_output; ++k)
        {
            const double got = output[k];
            if(std::isnan(got) && std::isnan(expected[k]))
            {
                ++nan_slots;
                continue;
            }
            if(!(got == expected[k]))
            {
                ++exact_mismatches;
                const double delta = std::abs(got - expected[k]);
                if(delta >= worst_abs)
                {
                    worst_abs  = delta;
                    worst_case = c;
                    worst_slot = k;
                }
            }
        }
    }
    std::printf(" [parity_golden_ipc] %s: %zu cases x %zu outputs, host |delta| = %.3e "
                "(%zu mismatches, %zu NaN slots; worst case %zu slot %zu)\n",
                std::string{label}.c_str(),
                golden.case_count(),
                golden.n_output,
                worst_abs,
                exact_mismatches,
                nan_slots,
                worst_case,
                worst_slot);
    CHECK(exact_mismatches == 0u);
}

void evaluate_barrier(const double* in, double* out)
{
    const double kappa = in[0], D = in[1], dHat = in[2], xi = in[3];
    double       B = 0.0, d1 = 0.0, d2 = 0.0;
    sym_codim::KappaBarrier(B, kappa, D, dHat, xi);
    sym_codim::dKappaBarrierdD(d1, kappa, D, dHat, xi);
    sym_codim::ddKappaBarrierddD(d2, kappa, D, dHat, xi);
    out[0] = B;
    out[1] = d1;
    out[2] = d2;
}

void evaluate_vhp(const double* in, double* out)
{
    LVec<double, 3> v[6];
    for(int k = 0; k < 6; ++k)
        for(int a = 0; a < 3; ++a)
            v[k](a) = in[3 * k + a];
    const auto& x  = v[0];
    const auto& P  = v[1];
    const auto& N  = v[2];
    const auto& xp = v[3];
    const auto& e1 = v[4];
    const auto& e2 = v[5];

    double D = 0.0;
    sym_ph::HalfPlaneD(D, x, P, N);
    LVec<double, 3> g;
    sym_ph::dHalfPlaneDdx(g, x, P, N);
    LMat<double, 3, 3> H;
    sym_ph::ddHalfPlaneDddx(H, x, P, N);
    LVec<double, 2> tr;
    sym_ph::TR(tr, x, xp, e1, e2);
    LMat<double, 2, 3> J;
    sym_ph::dTRdx(J, x, xp, e1, e2);

    std::size_t k = 0;
    out[k++] = D;
    for(int i = 0; i < 3; ++i)
        out[k++] = g(i);
    for(int col = 0; col < 3; ++col)  // column-major, as the generator writes it
        for(int row = 0; row < 3; ++row)
            out[k++] = H(row, col);
    out[k++] = tr(0);
    out[k++] = tr(1);
    for(int row = 0; row < 2; ++row)
        for(int col = 0; col < 3; ++col)
            out[k++] = J(row, col);
}
}  // namespace

TEST_CASE("parity_golden_codim_barrier", "[parity_golden_ipc]")
{
    require_exact("codim_barrier", "codim_barrier.bin", kBarrierIn, kBarrierOut,
                  evaluate_barrier);
}

TEST_CASE("parity_golden_vhp_distance", "[parity_golden_ipc]")
{
    require_exact("vhp_distance", "vhp_distance.bin", kVhpIn, kVhpOut, evaluate_vhp);
}
