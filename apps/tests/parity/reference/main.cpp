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
 * Three cases are authorable in this tree: `distance_pt` (the point-triangle distance /
 * gradient / Hessian of
 * `src/backends/cuda/utils/distance/details/point_triangle.inl`, pure Eigen) and, from
 * T8.1b, `codim_barrier` + `vhp_distance` — the two `contact_models/sym/*.inl` bodies, which
 * are plain `template <typename T>` + std::pow/Eigen and compile in a host TU once the
 * `__host__`/`__device__` markers are neutralised.  Together they pin the law of all four
 * normal contact models (the codim variants) and of the vertex–half-plane pair.  Every other
 * §4.1 case is registered as `not_implemented` with the plan that owns it: the remaining
 * contact/AL/stitch/pFpx headers spell `__device__` inline and pull
 * <type_define.h>/cuda_tool, so writing a case for them here would be a lie (research 15
 * §5.2).
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

// ---------------------------------------------------------------------------
//  The contact `sym/*.inl` bodies: the only other host-compilable reference files in
//  the contact path (research 15 §5.2 -- everything else in contact_models/ is
//  `__device__ inline` and pulls <type_define.h>/cuda_tool).  Both are plain
//  `template <typename T>` with Eigen/std::pow and no device intrinsics, so a host TU
//  can evaluate them directly; the two CUDA markers are neutralised here (the
//  `-D__host__= -D__device__=` shim of 03-parity-harness §4.1, spelled in-source so
//  both build systems see the same thing).
// ---------------------------------------------------------------------------
#if !defined(__host__)
#define __host__
#endif
#if !defined(__device__)
#define __device__
#endif

#include "../../../../src/backends/cuda/contact_system/contact_models/sym/codim_ipc_contact.inl"
#include "../../../../src/backends/cuda/contact_system/contact_models/sym/vertex_half_plane_distance.inl"

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
        {"codim_barrier",
         15,
         true,
         "sym/codim_ipc_contact.inl: the log^2 barrier law, value + d/dD + d^2/dD^2 "
         "(host-compilable with the __host__/__device__ shim)"},
        {"vhp_distance",
         15,
         true,
         "sym/vertex_half_plane_distance.inl: D, dDdx, ddDddx, TR, dTRdx "
         "(host-compilable with the __host__/__device__ shim)"},
        {"ipc_pt_energy",
         15,
         false,
         "the reference PT/EE/PE/PP math is codim_ipc_simplex_normal_contact_function.h, "
         "which is inline __device__ and pulls <type_define.h>; covered instead by the "
         "host instantiation of the ported twin (t81b_barrier_law / t81b_sym_expressions)"},
        {"ipc_pt_friction", 15, false, "inline __device__, not host-compilable"},
        {"ipc_vhp_normal",
         15,
         false,
         "the half-plane *barrier* chain is inline __device__; its distance layer is the "
         "`vhp_distance` golden next to this one"},
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

std::vector<double> pt_inputs()
{
    std::vector<double> in;
    for(const auto& c : pt_cases())
    {
        for(int i = 0; i < 3; ++i)
        {
            in.push_back(c.p(i));
            in.push_back(c.t0(i));
            in.push_back(c.t1(i));
            in.push_back(c.t2(i));
        }
    }
    return in;
}

void evaluate_pt(const double* in, double* out)
{
    V3 p, t0, t1, t2;
    for(int i = 0; i < 3; ++i)
    {
        p(i)  = in[i];
        t0(i) = in[3 + i];
        t1(i) = in[6 + i];
        t2(i) = in[9 + i];
    }

    double dist2 = 0.0;
    ref::point_triangle_distance2(p, t0, t1, t2, dist2);

    Eigen::Matrix<double, 12, 1> grad;
    grad.setZero();
    ref::point_triangle_distance2_gradient(p, t0, t1, t2, grad);

    Eigen::Matrix<double, 12, 12> hessian;
    hessian.setZero();
    ref::point_triangle_distance2_hessian(p, t0, t1, t2, hessian);

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
//  the codim_barrier golden: the SymEigen log^2 barrier law with its first and second
//  D-derivative (`sym/codim_ipc_contact.inl`, the dispatcher plus both families)
// ---------------------------------------------------------------------------
constexpr std::size_t kBarrierInput  = 4;  // kappa, D, dHat, xi
constexpr std::size_t kBarrierOutput = 3;  // B, dB/dD, d^2B/dD^2

/// The same sweep `t81b_barrier_law` drives through the device kernel: three toggles of the
/// dispatcher's `xi` branch (0 / the thickness forms), two `dHat`, three `kappa` signs and four
/// positions inside the active band `(xi^2, (dHat+xi)^2)`.
std::vector<double> barrier_inputs()
{
    std::vector<double> in;
    for(double xi : {0.0, 0.01, 0.005})
        for(double dHat : {0.1, 0.05})
            for(double kappa : {1.0, 8.0e5, -3.0})
            {
                const double lo = xi * xi, hi = (dHat + xi) * (dHat + xi);
                for(double frac : {0.1, 0.4, 0.75, 0.99})
                {
                    const double D = lo + frac * (hi - lo);
                    in.insert(in.end(), {kappa, D, dHat, xi});
                }
            }
    return in;
}

void evaluate_barrier(const double* in, double* out)
{
    const double kappa = in[0], D = in[1], dHat = in[2], xi = in[3];
    double       B = 0.0, d1 = 0.0, d2 = 0.0;
    KappaBarrier(B, kappa, D, dHat, xi);
    dKappaBarrierdD(d1, kappa, D, dHat, xi);
    ddKappaBarrierddD(d2, kappa, D, dHat, xi);
    out[0] = B;
    out[1] = d1;
    out[2] = d2;
}

// ---------------------------------------------------------------------------
//  the vhp_distance golden: `sym/vertex_half_plane_distance.inl` -- the signed distance,
//  its gradient/Hessian and the tangential relative displacement with its Jacobian
// ---------------------------------------------------------------------------
constexpr std::size_t kVhpInput  = 18;                // x, P, N, x_prev, e1, e2
constexpr std::size_t kVhpOutput = 1 + 3 + 9 + 2 + 6;  // D, dDdx, ddDddx, TR, dTRdx

std::vector<double> vhp_inputs()
{
    std::vector<double> in;
    Lcg                 rng;

    auto push = [&in](const V3& v)
    { in.insert(in.end(), {v(0), v(1), v(2)}); };

    auto emit = [&](const V3& x, const V3& P, const V3& N, const V3& xp)
    {
        // an orthonormal tangent basis, the way `compute_tan_basis` derives it
        const V3 trial = std::abs(N.dot(V3::UnitX())) > 0.9 ? V3::UnitZ() : V3::UnitX();
        const V3 e1    = trial.cross(N).normalized();
        const V3 e2    = N.cross(e1);
        push(x);
        push(P);
        push(N);
        push(xp);
        push(e1);
        push(e2);
    };

    for(int i = 0; i < 24; ++i)
    {
        const V3 x  = random_vec(rng);
        const V3 P  = random_vec(rng);
        const V3 N  = random_vec(rng).normalized();
        const V3 xp = x + 0.05 * random_vec(rng);
        emit(x, P, N, xp);
    }
    // degenerate-but-finite: zero tangent motion, a plane through the point, and the same
    // configuration twice (determinism)
    emit(V3::Zero(), V3::Zero(), V3::UnitZ(), V3::Zero());
    emit(V3{0.0, 0.0, 0.25}, V3{1.0, 2.0, 0.25}, V3::UnitZ(), V3{0.0, 0.0, 0.25});
    emit(V3{-1.0, 0.5, 0.0}, V3{1.0, 2.0, 0.0}, V3{0.6, 0.8, 0.0}, V3{-1.0, 0.5, 0.0});
    return in;
}

void evaluate_vhp(const double* in, double* out)
{
    V3 x, P, N, xp, e1, e2;
    auto load = [&in](int k, V3& v)
    {
        for(int i = 0; i < 3; ++i)
            v(i) = in[3 * k + i];
    };
    load(0, x);
    load(1, P);
    load(2, N);
    load(3, xp);
    load(4, e1);
    load(5, e2);

    double D = 0.0;
    HalfPlaneD(D, x, P, N);
    Eigen::Matrix<double, 3, 1> g;
    g.setZero();
    dHalfPlaneDdx(g, x, P, N);
    Eigen::Matrix<double, 3, 3> H;
    H.setZero();
    ddHalfPlaneDddx(H, x, P, N);
    Eigen::Matrix<double, 2, 1> tr;
    tr.setZero();
    TR(tr, x, xp, e1, e2);
    Eigen::Matrix<double, 2, 3> J;
    J.setZero();
    dTRdx(J, x, xp, e1, e2);

    std::size_t k = 0;
    out[k++] = D;
    for(int i = 0; i < 3; ++i)
        out[k++] = g(i);
    for(int col = 0; col < 3; ++col)  // column-major, as the port's LMat storage
        for(int row = 0; row < 3; ++row)
            out[k++] = H(row, col);
    out[k++] = tr(0);
    out[k++] = tr(1);
    for(int row = 0; row < 2; ++row)  // row-major: `dTRdx` is a 2x3 Jacobian read as J(r, c)
        for(int col = 0; col < 3; ++col)
            out[k++] = J(row, col);
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

using Evaluator = void (*)(const double* in, double* out);

/// One case family: the flat input sweep plus the reference evaluation of one case.
struct Generator
{
    std::string_view    name;
    std::size_t         n_input  = 0;
    std::size_t         n_output = 0;
    std::vector<double> (*inputs)() = nullptr;
    Evaluator           evaluate     = nullptr;
};

const Generator* find_generator(std::string_view name)
{
    static const std::vector<Generator> gens = {
        {"distance_pt", kPtInput, kPtOutput, pt_inputs, evaluate_pt},
        {"codim_barrier", kBarrierInput, kBarrierOutput, barrier_inputs, evaluate_barrier},
        {"vhp_distance", kVhpInput, kVhpOutput, vhp_inputs, evaluate_vhp},
    };
    for(const auto& g : gens)
        if(g.name == name)
            return &g;
    return nullptr;
}

void write_golden(const fs::path& path, const Generator& gen, const std::vector<double>& inputs)
{
    const std::size_t case_count = inputs.size() / gen.n_input;
    if(case_count * gen.n_input != inputs.size())
        throw std::runtime_error{"input sweep is not a whole number of cases"};

    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    if(!out)
        throw std::runtime_error{"cannot open golden file '" + path.string() + "'"};

    write_raw(out, kMagic.data(), kMagic.size());
    write_u32(out, kVersion);
    write_u32(out, static_cast<std::uint32_t>(gen.name.size()));
    write_raw(out, gen.name.data(), gen.name.size());
    write_u64(out, static_cast<std::uint64_t>(case_count));
    write_u64(out, gen.n_input);
    write_u64(out, gen.n_output);

    std::vector<double> output(gen.n_output);
    for(std::size_t c = 0; c < case_count; ++c)
    {
        const double* input = inputs.data() + c * gen.n_input;
        gen.evaluate(input, output.data());
        write_raw(out, input, gen.n_input * sizeof(double));
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
            const auto* gen = find_generator(entry.name);
            if(gen == nullptr)
                throw std::runtime_error{"no generator for golden '"
                                         + std::string{entry.name} + "'"};

            const auto  inputs = gen->inputs();
            const auto  path   = golden_dir / (std::string{entry.name} + ".bin");
            write_golden(path, *gen, inputs);
            std::printf("wrote %s (%zu cases, %zu inputs, %zu outputs each)\n",
                        path.string().c_str(),
                        inputs.size() / gen->n_input,
                        gen->n_input,
                        gen->n_output);
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
