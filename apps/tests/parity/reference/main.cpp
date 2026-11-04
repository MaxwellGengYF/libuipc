/********************************************************************
 * @file reference/main.cpp
 * @brief T9.3a — uipc_parity_reference: the Tier-1 golden generator.
 *
 * A golden is a binary record of what the *reference* (CUDA-side, read-only)
 * implementation produces on the host for fixed inputs, so a Luisa-side test can
 * assert bit-exact equality without a GPU and without the CUDA backend being
 * buildable (03-parity-harness.md §4).
 *
 * Format `UIPCGOLD1` (all little-endian, float64 payload):
 *
 *   char[9]  magic "UIPCGOLD1"
 *   u32      version (1)
 *   u32      name_len
 *   char[]   case name, name_len bytes, no NUL
 *   u64      case_count
 *   u64      n_input   (doubles per case)
 *   u64      n_output  (doubles per case)
 *   per case: double[n_input], double[n_output]
 *
 * Inputs are written too, so the consumer can assert it fed identical data
 * (03-parity-harness.md §4.1).
 *
 * Only one case is authorable in this tree: `distance_pt`. Its reference
 * (`src/backends/cuda/utils/distance/details/point_triangle.inl`) is pure Eigen and
 * host-compilable, and the ported twin exists
 * (src/backends/luisa/luisa_device/math/distance/point_triangle.h). The other §4.1
 * cases are registered as `not_implemented` with the plan that owns them — the
 * contact/AL/stitch/pFpx headers spell `__device__` inline and are not host
 * compilable (research §1), so writing a case for them here would be a lie.
 *
 * NOTE: this binary includes the reference headers *directly* and never links the
 * CUDA backend: no nvcc is involved, which is why it builds in a --backend_cuda=n
 * configuration.
 *********************************************************************/
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

// ---------------------------------------------------------------------------
//  The reference implementation, evaluated on the host.
//  `UIPC_GENERIC` is empty in a host TU (src/backends/cuda/type_define.h:30-34), so
//  the __device__-free distance .inl files compile as plain C++.
// ---------------------------------------------------------------------------
#define UIPC_GENERIC
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <uipc/common/type_define.h>

#include "../../../../src/backends/cuda/utils/distance/details/point_triangle.inl"

namespace ref = uipc::backend::cuda::distance;

namespace
{
namespace fs = std::filesystem;

constexpr std::string_view kMagic   = "UIPCGOLD1";
constexpr std::uint32_t    kVersion = 1;

// ---------------------------------------------------------------------------
//  case registry
// ---------------------------------------------------------------------------
struct GoldenEntry
{
    std::string_view name;
    /// The plan that owns the case (03-parity-harness.md §4.1).
    int  plan        = 0;
    bool implemented = false;
    /// Why it is not implemented here (empty for the implemented case).
    std::string_view note = {};
};

const std::vector<GoldenEntry>& registry()
{
    static const std::vector<GoldenEntry> entries = {
        {"distance_pt", 21, true, ""},
        {"ipc_pt_energy",
         15,
         false,
         "ipc_simplex_normal_contact_function.h does not exist; the reference is "
         "codim_ipc_*_contact_function.h, which is inline __device__"},
        {"ipc_pt_friction", 15, false, "inline __device__, not host-compilable"},
        {"ipc_vhp_normal", 15, false, "inline __device__, not host-compilable"},
        {"ipc_vhp_friction", 15, false, "inline __device__, not host-compilable"},
        {"codim_pe", 15, false, "inline __device__, not host-compilable"},
        {"codim_pp", 15, false, "inline __device__, not host-compilable"},
        {"pfpx_pt",
         16,
         false,
         "host-compilable (all UIPC_GENERIC); the golden gets written by plan 16, "
         "the ported family has no consumer yet"},
        {"al_pt", 17, false, "host-compilable (0 __device__); needs the active-set port of plan 17"},
        {"stitch_vv", 19, false, "needs inter_primitive_effect_system (plan 19)"},
        {"abd_fem_block", 18, false, "the block formula lives in a .cu (plan 18)"},
        {"mas_cluster_inverse", 13, false, "finite_element/mas_preconditioner_engine.* (plan 13)"},
    };
    return entries;
}

// ---------------------------------------------------------------------------
//  deterministic input sweep
// ---------------------------------------------------------------------------
struct Lcg
{
    std::uint64_t s = 0x2f1e3d4c5b6a7981ull;
    double        next() noexcept
    {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((s >> 11) & 0x1fffffull) / 1048576.0 - 1.0;
    }
};

using V3 = Eigen::Matrix<double, 3, 1>;

V3 random_vec(Lcg& rng, double scale = 1.0)
{
    return V3{scale * rng.next(), scale * rng.next(), scale * rng.next()};
}

/// One point-triangle configuration: (p, t0, t1, t2).
struct PtCase
{
    V3 p, t0, t1, t2;
};

/// The sweep: 32 well-scaled random configurations plus 8 deliberately degenerate
/// ones (each exercises a different branch of the closest-point classification).
std::vector<PtCase> pt_cases()
{
    std::vector<PtCase> cases;
    Lcg                 rng;

    for(int i = 0; i < 32; ++i)
        cases.push_back(
            PtCase{random_vec(rng), random_vec(rng), random_vec(rng), random_vec(rng)});

    const V3 a{-1.0, 0.0, 0.0};
    const V3 b{1.0, 0.0, 0.0};
    const V3 c{0.0, 1.0, 0.0};
    cases.push_back(PtCase{{0.0, 0.0, 1.0}, a, b, c});  // vertex/p inside region
    cases.push_back(PtCase{{0.3, 0.3, 0.0}, a, b, c});  // in the plane, inside
    cases.push_back(PtCase{{5.0, 5.0, 5.0}, a, b, c});  // far outside, vertex branch
    cases.push_back(PtCase{a, a, b, c});                // p coincides with t0
    cases.push_back(PtCase{{0.0, 0.0, 0.0}, a, b, {-1.0, 0.0, 0.0}});  // zero-area triangle
    cases.push_back(PtCase{{0.0, 0.0, 0.0}, a, a, c});  // two coincident vertices
    cases.push_back(PtCase{{0.0, 1e-9, 1e-9}, a, b, c});  // near-coplanar
    cases.push_back(PtCase{
        {0.0, 0.0, 0.0}, {1e-12, 0.0, 0.0}, {2e-12, 0.0, 0.0}, {3e-12, 0.0, 0.0}});  // collapsed to a segment
    return cases;
}

// ---------------------------------------------------------------------------
//  the distance_pt golden
// ---------------------------------------------------------------------------
constexpr std::size_t kPtInput  = 12;                // p, t0, t1, t2
constexpr std::size_t kPtOutput = 1 + 12 + 12 * 12;  // dist2, gradient, Hessian

void evaluate_pt(const PtCase& c, double* in, double* out)
{
    for(int i = 0; i < 3; ++i)
    {
        in[i]     = c.p(i);
        in[3 + i] = c.t0(i);
        in[6 + i] = c.t1(i);
        in[9 + i] = c.t2(i);
    }

    double dist2 = 0.0;
    ref::point_triangle_distance2(c.p, c.t0, c.t1, c.t2, dist2);

    Eigen::Matrix<double, 12, 1> grad;
    grad.setZero();
    ref::point_triangle_distance2_gradient(c.p, c.t0, c.t1, c.t2, grad);

    Eigen::Matrix<double, 12, 12> hessian;
    hessian.setZero();
    ref::point_triangle_distance2_hessian(c.p, c.t0, c.t1, c.t2, hessian);

    out[0] = dist2;
    for(int i = 0; i < 12; ++i)
        out[1 + i] = grad(i);
    // Column-major, the storage order of both Eigen and the port's LMat, so the
    // consumer's to_flat() (c outer, r inner) reads the same layout.
    std::size_t k = 1 + 12;
    for(int col = 0; col < 12; ++col)
        for(int row = 0; row < 12; ++row)
            out[k++] = hessian(row, col);
}

// ---------------------------------------------------------------------------
//  binary writer
// ---------------------------------------------------------------------------
void write_raw(std::ofstream& out, const void* data, std::size_t bytes)
{
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if(!out)
        throw std::runtime_error{"failed to write golden file"};
}

void write_u32(std::ofstream& out, std::uint32_t value)
{
    write_raw(out, &value, sizeof(value));
}

void write_u64(std::ofstream& out, std::uint64_t value)
{
    write_raw(out, &value, sizeof(value));
}

void write_golden(const fs::path& path, std::string_view name, const std::vector<PtCase>& cases)
{
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    if(!out)
        throw std::runtime_error{"cannot open golden file '" + path.string() + "'"};

    write_raw(out, kMagic.data(), kMagic.size());
    write_u32(out, kVersion);
    write_u32(out, static_cast<std::uint32_t>(name.size()));
    write_raw(out, name.data(), name.size());
    write_u64(out, static_cast<std::uint64_t>(cases.size()));
    write_u64(out, kPtInput);
    write_u64(out, kPtOutput);

    std::vector<double> input(kPtInput);
    std::vector<double> output(kPtOutput);
    for(const auto& c : cases)
    {
        evaluate_pt(c, input.data(), output.data());
        write_raw(out, input.data(), input.size() * sizeof(double));
        write_raw(out, output.data(), output.size() * sizeof(double));
    }
    out.flush();
    if(!out)
        throw std::runtime_error{"failed to flush golden file '" + path.string() + "'"};
}

std::string program_path()
{
#if defined(_WIN32)
    char  buffer[MAX_PATH];
    DWORD n = ::GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if(n > 0 && n < MAX_PATH)
        return std::string{buffer, n};
    return {};
#elif defined(__linux__)
    std::error_code ec;
    auto            resolved = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string{} : resolved.string();
#else
    return {};
#endif
}

constexpr std::string_view kUsage =
    R"(usage: uipc_parity_reference [options]

    --case=NAME   write only this golden (default: every implemented case)
    --out=DIR     golden directory (default: <exe dir>/parity_golden)
    --list        print the case registry (implemented and not implemented)
    --help        print this message

exit codes: 0 ok, 2 usage, 3 the requested case is not implemented in this tree
)";
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string case_name;
        std::string out_dir;
        bool        list = false;

        for(int i = 1; i < argc; ++i)
        {
            const std::string_view arg =
                (argv[i] != nullptr) ? std::string_view{argv[i]} : std::string_view{};
            if(arg == "--help" || arg == "-h")
            {
                std::printf("%s", std::string{kUsage}.c_str());
                return 0;
            }
            if(arg == "--list")
                list = true;
            else if(arg.rfind("--case=", 0) == 0)
                case_name = std::string{arg.substr(7)};
            else if(arg.rfind("--out=", 0) == 0)
                out_dir = std::string{arg.substr(6)};
            else
            {
                std::fprintf(stderr,
                             "unknown argument '%s'\n%s",
                             std::string{arg}.c_str(),
                             std::string{kUsage}.c_str());
                return 2;
            }
        }

        if(list)
        {
            for(const auto& entry : registry())
            {
                std::printf("%-20s plan %-2d %-15s %s\n",
                            std::string{entry.name}.c_str(),
                            entry.plan,
                            entry.implemented ? "implemented" : "not_implemented",
                            std::string{entry.note}.c_str());
            }
            return 0;
        }

        const fs::path golden_dir =
            out_dir.empty() ?
                (fs::path{program_path()}.parent_path() / "parity_golden") :
                fs::path{out_dir};
        std::error_code ec;
        fs::create_directories(golden_dir, ec);
        if(ec)
            throw std::runtime_error{"cannot create golden directory '"
                                     + golden_dir.string() + "': " + ec.message()};

        int written = 0;
        for(const auto& entry : registry())
        {
            if(!case_name.empty() && entry.name != case_name)
                continue;
            if(!entry.implemented)
            {
                // Only an explicit request for a not-implemented case is exit 3; a
                // plain "write everything" run skips them and reports them on stdout.
                if(!case_name.empty())
                {
                    std::fprintf(stderr,
                                 "not implemented: golden '%s' is owned by plan %d (%s)\n",
                                 std::string{entry.name}.c_str(),
                                 entry.plan,
                                 std::string{entry.note}.c_str());
                    return 3;
                }
                std::printf("skipped %s (plan %d, not_implemented)\n",
                            std::string{entry.name}.c_str(),
                            entry.plan);
                continue;
            }
            if(entry.name != "distance_pt")
                throw std::runtime_error{"no generator for golden '"
                                         + std::string{entry.name} + "'"};

            const auto cases = pt_cases();
            const auto path  = golden_dir / (std::string{entry.name} + ".bin");
            write_golden(path, entry.name, cases);
            std::printf("wrote %s (%zu cases, %zu inputs, %zu outputs each)\n",
                        path.string().c_str(),
                        cases.size(),
                        kPtInput,
                        kPtOutput);
            ++written;
        }

        if(!case_name.empty() && written == 0)
        {
            std::fprintf(stderr,
                         "unknown golden case '%s' (see --list)\n",
                         case_name.c_str());
            return 2;
        }
        std::fflush(stdout);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
