/********************************************************************
 * @file ab_main.cpp
 * @brief A/B measurement vehicle for `optimize_plan.md` OP-A1 - "open and commit the
 * frame command list" (acceptance M-O1/M-O2).
 *
 * One process, one scene, N frames through the *whole* engine, not just the device:
 * build a procedural affine-body cube -> `World::init` -> `advance()/sync()` per
 * frame. The only thing the two runs differ in is `config["luisa"]["frame_batching"]`,
 * which the Luisa backend reads in `init_device()` (src/backends/luisa/sim_engine.cpp:
 * 148) to decide whether a frame is one opened-and-committed command list or one
 * submission per command. The same binary therefore holds scene, frames and clock
 * constant, which is what makes the pair comparable:
 *
 *      uipc_engine_ab --batching=1 --json
 *      uipc_engine_ab --batching=0 --json
 *
 * Scene provenance: this is the *minimal* form of TEST_CASE("abd_base_scene") in
 * apps/tests/backends/luisa/luisa_abd_base_test.cpp:2144 - one procedural six-tet
 * cube with the frontend `AffineBodyConstitution`, no asset file, no `AssetDir`, no
 * contact, no sanity check. One deliberate difference: that case re-labels its bodies
 * to a test-local mock constitution uid so it can assert on its own slice of a
 * shared binary; a benchmark wants the production path, so the real uid-1
 * constitution is used here.
 *
 * Deliberately independent of apps/benchmarks/common (the Tier-0 harness): the Tier-0
 * cases measure a *device* operation, this measures an *engine frame*.
 *********************************************************************/

#include <uipc/uipc.h>
#include <uipc/backend/buffer_view.h>
#include <uipc/backend/visitors/world_visitor.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/logger.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/core/affine_body_state_accessor_feature.h>
#include <uipc/geometry/simplicial_complex.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

using namespace uipc;

namespace
{
// ---------------------------------------------------------------------------
//  command line
// ---------------------------------------------------------------------------
constexpr std::string_view kOptionsHelp =
    R"(
  --batching=0|1  Luisa frame command-list batching (default 1)
  --frames=N      timed frames, N > 0 (default 30)
  --warmup=N      untimed frames before the timed ones (default 3)
  --out=PATH      also write the JSON result to PATH
  --json          print only the result JSON on stdout
  --help          print this message
)";

/// The binary name is the build target's business, so it is taken from `argv[0]`
/// rather than hardcoded here.
std::string usage_text(std::string_view program)
{
    return "usage: " + std::string{program} + " [options]\n" + std::string{kOptionsHelp};
}

/// Bad command line: distinct from a runtime failure because the contract asks for a
/// usage message and exit code 2 (a broken scene or a missing backend exits 1).
class UsageError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

struct Options
{
    bool        batching  = true;
    SizeT       frames    = 30;
    SizeT       warmup    = 3;
    std::string out_path  = {};
    bool        json_only = false;
    bool        help      = false;
};

/// `--name=value` -> true and `value` set; false when `arg` is not `--name=...`.
bool take_value(std::string_view arg, std::string_view name, std::string_view& value)
{
    if(arg.size() <= name.size() || arg.substr(0, name.size()) != name)
        return false;
    if(arg[name.size()] != '=')
        return false;
    value = arg.substr(name.size() + 1);
    return true;
}

SizeT parse_size(std::string_view text, std::string_view option)
{
    SizeT       value        = 0;
    const auto* first        = text.data();
    const auto* last         = text.data() + text.size();
    const auto [stop, error] = std::from_chars(first, last, value);
    if(error != std::errc{} || stop != last)
    {
        throw UsageError{std::string{option} + " expects a non-negative integer, got '"
                         + std::string{text} + "'"};
    }
    return value;
}

bool parse_batching(std::string_view text)
{
    if(text == "1")
        return true;
    if(text == "0")
        return false;
    throw UsageError{std::string{"--batching expects 0 or 1, got '"}
                     + std::string{text} + "'"};
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for(int i = 1; i < argc; ++i)
    {
        const std::string_view arg{argv[i]};
        std::string_view       value;

        if(arg == "--help" || arg == "-h")
        {
            options.help = true;
            return options;
        }
        if(arg == "--json")
        {
            options.json_only = true;
            continue;
        }
        if(take_value(arg, "--batching", value))
        {
            options.batching = parse_batching(value);
            continue;
        }
        if(take_value(arg, "--frames", value))
        {
            options.frames = parse_size(value, "--frames");
            if(options.frames == 0)
                throw UsageError{"--frames must be greater than 0"};
            continue;
        }
        if(take_value(arg, "--warmup", value))
        {
            options.warmup = parse_size(value, "--warmup");
            continue;
        }
        if(take_value(arg, "--out", value))
        {
            if(value.empty())
                throw UsageError{"--out expects a file path"};
            options.out_path = std::string{value};
            continue;
        }
        throw UsageError{"unknown argument '" + std::string{arg} + "'"};
    }
    return options;
}

// ---------------------------------------------------------------------------
//  scene: one procedural affine-body cube
//
//  Copied from luisa_abd_base_test.cpp:393-409 (cube_tets/cube_vertices) and
//  :1031-1072 (make_box_body), minus the asset loading and minus the mock uid.
// ---------------------------------------------------------------------------
/// The six-tet decomposition of the cube [0,L]^3: every tet is a monotone path from
/// v0 to v7 with determinant +L^3, i.e. the decomposition integrates the box exactly.
std::array<Vector4i, 6> cube_tets() noexcept
{
    return {{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}}};
}

/// Vertex index = x + 2y + 4z (luisa_abd_base_test.cpp:398).
std::array<Vector3, 8> cube_vertices(Float L) noexcept
{
    std::array<Vector3, 8> vertices;
    for(int i = 0; i < 8; ++i)
    {
        const Float x = (i & 1) ? L : 0.0;
        const Float y = (i & 2) ? L : 0.0;
        const Float z = (i & 4) ? L : 0.0;
        vertices[i]   = Vector3{x, y, z};
    }
    return vertices;
}

Matrix4x4 make_transform(const Matrix3x3& A, const Vector3& t) noexcept
{
    Matrix4x4 transform         = Matrix4x4::Identity();
    transform.block<3, 3>(0, 0) = A;
    transform.block<3, 1>(0, 3) = t;
    return transform;
}

geometry::SimplicialComplex make_abd_cube(Float L, Float density, Float kappa, const Matrix4x4& transform)
{
    const auto vertices = cube_vertices(L);
    const auto tets     = cube_tets();

    auto sc = geometry::tetmesh(span<const Vector3>{vertices}, span<const Vector4i>{tets});
    // `label_surface` writes `is_surf`, which the affine-body surface reporter reads;
    // it is a host-side labelling pass and does not depend on the backend.
    geometry::label_surface(sc);
    sc.instances().resize(1);

    constitution::AffineBodyConstitution abd;
    abd.apply_to(sc, kappa, density);

    // The instance transform carries the deformed affine state; gravity then acts on
    // it every frame, so the state digest below is not the trivial rest state.
    view(sc.transforms())[0] = transform;
    return sc;
}

// ---------------------------------------------------------------------------
//  engine / world plumbing
// ---------------------------------------------------------------------------
void check_engine(core::World& world, core::Engine& engine, std::string_view stage)
{
    // The engine records statuses instead of throwing (the World logs them and goes
    // invalid), so a benchmark that ignored them could happily time a dead world.
    if(engine.status().has_error())
    {
        throw std::runtime_error{std::string{stage} + ": the luisa engine reported an error: "
                                 + engine.status().to_json().dump()};
    }
    if(!world.is_valid())
    {
        throw std::runtime_error{std::string{stage} + ": the world is invalid (see the engine log above)"};
    }
}

std::string make_workspace()
{
    // Same workspace shape as the Luisa tests (luisa_abd_base_test.cpp:147): a
    // per-scene directory under the temp dir. The two A/B runs share it; it only
    // holds the backend's own scratch (nothing is dumped into it here).
    const auto dir = std::filesystem::temp_directory_path() / "uipc_engine_ab_abd_cube";
    std::filesystem::create_directories(dir);
    return dir.string() + "/";
}

// ---------------------------------------------------------------------------
//  state checksum
//
//  The digest must be *independent of batching*: submission order is an execution
//  detail, so two runs that batched differently must produce the same bits. It is
//  taken over the simulated vertex positions rather than over timings, so a batching
//  change that silently altered the physics cannot pass as a speed-up.
// ---------------------------------------------------------------------------
constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime       = 1099511628211ull;

std::uint64_t hash_double(std::uint64_t seed, Float value) noexcept
{
    // FNV-1a over the raw IEEE-754 bytes: bit-exact by construction, so `-0.0` vs
    // `0.0` (a genuine difference) does not cancel the way a floating-point sum of
    // the coordinates would.
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    std::uint64_t       hash = seed;
    for(int byte = 0; byte < 8; ++byte)
    {
        hash ^= (bits >> (8 * byte)) & 0xFFull;
        hash *= kFnvPrime;
    }
    return hash;
}

/// The current affine transform of the first body.
///
/// The scene's own `transform` attribute is what the *frontend* wrote at build time;
/// the state the last frame produced lives in the backend. The backend is asked
/// first (public `AffineBodyStateAccessorFeature`, the same read-back the Luisa
/// tests use at luisa_abd_base_test.cpp:2001) and the attribute is the fallback for a
/// backend that publishes no affine-body state.
Matrix4x4 first_instance_transform(core::World& world, geometry::SimplicialComplex& sc)
{
    if(auto feature = world.features().find<core::AffineBodyStateAccessorFeature>();
       feature && feature->body_count() > 0)
    {
        std::array<Matrix4x4, 1> transform{Matrix4x4::Identity()};
        // The Luisa overrider writes into a *host* buffer
        // (affine_body_state_accessor_feature.cpp, do_copy_transform_to), so a plain
        // std::array is a valid `BufferView` destination for the "luisa" backend.
        backend::BufferView buffer{reinterpret_cast<backend::HandleT>(transform.data()),
                                   0,
                                   transform.size(),
                                   sizeof(Matrix4x4),
                                   sizeof(Matrix4x4),
                                   "luisa"};
        feature->copy_transform_to(buffer, 0, transform.size());
        return transform[0];
    }

    auto transform_attr = sc.instances().find<Matrix4x4>(builtin::transform);
    if(transform_attr == nullptr || transform_attr->view().empty())
        throw std::runtime_error{"the first geometry carries no instance transform"};
    // Say it out loud: this transform is correct before the first frame and stale
    // after it (this backend writes `q` back into the geometry only on dump), so the
    // digest is a weak guard in this mode. On stderr, because stdout is the JSON.
    std::cerr << "warning: no affine-body state accessor, the checksum falls back to "
                 "the frontend's instance transform (stale after the first frame)\n";
    return transform_attr->view()[0];
}

/// Checksum definition (reproducible across processes, batching-independent):
///   digest = FNV-1a( p_0.x, p_0.y, p_0.z, p_1.x, ... ) over the vertices of the
///            *first* scene geometry in geometry order, with p = A x + t,
///   checksum = static_cast<double>(digest >> 11).
/// `x` is the vertex stored in the geometry (the material configuration), `A`/`t` are
/// the frame's affine state; p = A x + t is the affine-body vertex law
/// (luisa_abd_base_test.cpp:2247 asserts the same identity against the device
/// buffers). Truncating to 53 bits makes the value exactly representable as a
/// `double`, so two processes' printed `checksum` compares bit-for-bit.
double state_checksum(core::World& world)
{
    auto visitor    = backend::WorldVisitor{world};
    auto geometries = visitor.scene().geometries();
    if(geometries.empty())
        throw std::runtime_error{"the scene has no geometry to checksum"};

    auto* sc = geometries[0]->geometry().as<geometry::SimplicialComplex>();
    if(sc == nullptr)
        throw std::runtime_error{"the first scene geometry is not a SimplicialComplex"};

    // find<> hands back a shared_ptr<AttributeSlot<T>>, not a raw pointer.
    auto position_attr = sc->vertices().find<Vector3>(builtin::position);
    if(!position_attr)
        throw std::runtime_error{"the first geometry has no `position` attribute"};
    const auto positions = position_attr->view();

    const Matrix4x4 transform = first_instance_transform(world, *sc);
    const Matrix3x3 A         = transform.block<3, 3>(0, 0);
    const Vector3   t         = transform.block<3, 1>(0, 3);

    std::uint64_t digest = kFnvOffsetBasis;
    for(const Vector3& x : positions)
    {
        const Vector3 p = A * x + t;
        for(int axis = 0; axis < 3; ++axis)
            digest = hash_double(digest, p(axis));
    }
    return static_cast<double>(digest >> 11);
}

// ---------------------------------------------------------------------------
//  statistics over the timed frames
// ---------------------------------------------------------------------------
struct FrameSummary
{
    double median_ms = 0.0;
    double mean_ms   = 0.0;
    double min_ms    = 0.0;
    double max_ms    = 0.0;
    double p95_ms    = 0.0;
    double cv_pct    = 0.0;
};

FrameSummary summarize(std::vector<double> samples)
{
    FrameSummary summary;
    if(samples.empty())
        return summary;

    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();

    double total = 0.0;
    for(double sample : samples)
        total += sample;

    summary.mean_ms   = total / static_cast<double>(n);
    summary.min_ms    = samples.front();
    summary.max_ms    = samples.back();
    summary.median_ms = (n % 2 == 1) ? samples[n / 2] :
                                       0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    // Nearest-rank P95 on the sorted copy, as specified by the report contract.
    const auto p95_index = static_cast<SizeT>(0.95 * static_cast<double>(n - 1));
    summary.p95_ms = samples[p95_index];

    double squared = 0.0;
    for(double sample : samples)
    {
        const double delta = sample - summary.mean_ms;
        squared += delta * delta;
    }
    const double stddev = std::sqrt(squared / static_cast<double>(n));
    summary.cv_pct = (summary.mean_ms > 0.0) ? (stddev / summary.mean_ms * 100.0) : 0.0;
    return summary;
}

/// Milliseconds are reported at microsecond resolution: anything finer is below the
/// clock noise on every platform this runs on, and it keeps two runs' JSON diffable.
double round6(double value) noexcept
{
    return std::round(value * 1.0e6) / 1.0e6;
}

// ---------------------------------------------------------------------------
//  reporting
// ---------------------------------------------------------------------------
Json make_result(const Options&             options,
                 const FrameSummary&        summary,
                 const std::vector<double>& round_ms,
                 double                     checksum,
                 const Json&                last_frame_stats)
{
    // The key *set* is the contract, not the key order: `uipc::Json` is
    // `nlohmann::json`, whose object type is an ordered-by-name map.
    Json result                = Json::object();
    result["schema"]           = "uipc-engine-ab/v1";
    result["backend"]          = "luisa";
    result["scene"]            = "abd_cube";
    result["frame_batching"]   = options.batching;
    result["frames"]           = options.frames;
    result["warmup"]           = options.warmup;
    result["median_ms"]        = round6(summary.median_ms);
    result["mean_ms"]          = round6(summary.mean_ms);
    result["min_ms"]           = round6(summary.min_ms);
    result["max_ms"]           = round6(summary.max_ms);
    result["p95_ms"]           = round6(summary.p95_ms);
    result["cv_pct"]           = round6(summary.cv_pct);
    result["checksum"]         = checksum;
    result["round_ms"]         = Json::array();
    result["last_frame_stats"] = last_frame_stats;
    for(double millis : round_ms)
        result["round_ms"].push_back(round6(millis));
    return result;
}

std::int64_t integer_or(const Json& object, std::string_view key) noexcept
{
    // The frame-stats schema is allowed to grow/shrink (i.e. `counters` may be
    // absent), so every read is a lookup with a fallback, never `operator[]`.
    const auto it = object.find(std::string{key});
    if(it == object.end() || !it->is_number())
        return 0;
    return it->get<std::int64_t>();
}

void print_human(std::ostream& os, const Json& result)
{
    const auto line = [&os](std::string_view key, const Json& value)
    { os << key << ": " << value.dump() << '\n'; };

    line("schema", result.at("schema"));
    line("backend", result.at("backend"));
    line("scene", result.at("scene"));
    line("frame_batching", result.at("frame_batching"));
    line("frames", result.at("frames"));
    line("warmup", result.at("warmup"));
    line("median_ms", result.at("median_ms"));
    line("mean_ms", result.at("mean_ms"));
    line("min_ms", result.at("min_ms"));
    line("max_ms", result.at("max_ms"));
    line("p95_ms", result.at("p95_ms"));
    line("cv_pct", result.at("cv_pct"));
    line("checksum", result.at("checksum"));

    // The batching effect must be readable without opening the JSON: one frame's
    // submission counters, spelled out at the top level.
    const Json& stats    = result.at("last_frame_stats");
    Json        counters = Json::object();
    if(const auto it = stats.find("counters"); it != stats.end() && it->is_object())
        counters = *it;

    const auto commits        = integer_or(counters, "commits");
    const auto commands       = integer_or(counters, "commands");
    const auto direct_submits = integer_or(counters, "direct_submits");
    os << "frame_commits: " << commits << '\n';
    os << "frame_commands: " << commands << '\n';
    os << "frame_direct_submits: " << direct_submits << '\n';

    // One opened-and-committed list costs O(1) commits per frame regardless of the
    // command count; the unbatched baseline pays one per command. An engine that
    // publishes no counters reports 0, never NaN.
    const double per_1000 = (commands > 0) ? 1000.0 * static_cast<double>(commits)
                                                 / static_cast<double>(commands) :
                                             0.0;
    os << "commits_per_1000_commands: " << std::fixed << std::setprecision(3)
       << per_1000 << '\n';
}

// ---------------------------------------------------------------------------
//  driver
// ---------------------------------------------------------------------------
int run(const Options& options)
{
    // spdlog's console sink is *stdout* (src/core/common/logger.cpp:20) and the Luisa
    // backend logs one `info` line per frame (sim_engine.cpp:421). Both would break
    // the single-JSON-object contract of `--json`, and the per-frame formatting would
    // land inside the timed window.
    logger::set_level(Logger::Level::err);

    // ---- config ----
    // Two configs, because they are two different contracts: the *engine* config is
    // free-form (`luisa/*`) and is what the backend reads, the *scene* config is
    // schema-validated and rejects unknown keys (scene_default_config.cpp:516).
    Json engine_config                       = core::Engine::default_config();
    engine_config["luisa"]["frame_batching"] = options.batching;

    Json scene_config                       = core::Scene::default_config();
    scene_config["dt"]                      = Float{0.01};
    scene_config["gravity"]                 = Vector3{0.0, -9.8, 0.0};
    scene_config["integrator"]["type"]      = std::string{"bdf1"};
    scene_config["contact"]["enable"]       = IndexT{0};
    scene_config["contact"]["constitution"] = std::string{"ipc"};
    scene_config["sanity_check"]["enable"]  = IndexT{0};

    // ---- scene ----
    // Fixed constants: the A/B must not depend on anything but the batching switch.
    constexpr Float kCubeSize  = 0.6;
    constexpr Float kDensity   = 800.0;
    constexpr Float kKappa     = 5.0e7;
    constexpr Float kTiltAngle = 0.35;

    // A tilted, lifted cube (the transform of luisa_abd_base_test.cpp:2155-2157, with
    // the rotation about +y written out so no Eigen/Geometry include is needed): the
    // affine state is not the identity, so the vertex reporter and the time
    // integrator have real work and the checksum is not the rest configuration.
    const Float cos_tilt = std::cos(kTiltAngle);
    const Float sin_tilt = std::sin(kTiltAngle);

    Matrix3x3 tilt = Matrix3x3::Identity();
    tilt(0, 0)     = cos_tilt;
    tilt(0, 2)     = sin_tilt;
    tilt(2, 0)     = -sin_tilt;
    tilt(2, 2)     = cos_tilt;

    core::Scene scene{scene_config};
    auto        object = scene.objects().create("abd_cube");
    object->geometries().create(make_abd_cube(
        kCubeSize, kDensity, kKappa, make_transform(tilt, Vector3{0.0, 0.4, 0.0})));

    // ---- engine ----
    const std::string workspace = make_workspace();
    core::Engine      engine{"luisa", workspace, engine_config};
    core::World       world{engine};

    world.init(scene);
    check_engine(world, engine, "World::init");

    // ---- warm-up ----
    // The first frames JIT the LC pipelines, allocate device buffers and fault in
    // pages; both A/B arms pay that equally, so charging it to the median would only
    // dilute the difference the run is measuring.
    Json last_frame_stats = Json::object();
    for(SizeT frame = 0; frame < options.warmup; ++frame)
    {
        world.advance();
        world.sync();
        last_frame_stats = engine.frame_stats();
        check_engine(world, engine, "warm-up frame");
    }

    // ---- timed frames ----
    // `advance()` records the frame's work and `sync()` is what makes it observable
    // (the Luisa backend's do_sync commits the frame list and waits,
    // sim_engine.cpp:512), so the pair is the frame as an application sees it.
    using clock = std::chrono::steady_clock;

    std::vector<double> round_ms;
    round_ms.reserve(options.frames);
    for(SizeT frame = 0; frame < options.frames; ++frame)
    {
        const auto start = clock::now();
        world.advance();
        world.sync();
        const auto stop = clock::now();

        round_ms.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());

        // Outside the timed window: `frame_stats()` builds a JSON object.
        last_frame_stats = engine.frame_stats();
        check_engine(world, engine, "timed frame");
    }

    // ---- state ----
    // `retrieve()` is the step that hands the simulated state back to the
    // application (include/uipc/core/world.h:38). It is outside the timed window
    // because the timed unit is the frame, not the read-back.
    world.retrieve();
    check_engine(world, engine, "retrieve");
    const double checksum = state_checksum(world);

    // ---- report ----
    const auto summary = summarize(round_ms);
    const auto result = make_result(options, summary, round_ms, checksum, last_frame_stats);

    if(!options.out_path.empty())
    {
        std::ofstream out{options.out_path, std::ios::binary | std::ios::trunc};
        if(!out)
            throw std::runtime_error{"cannot open --out file '" + options.out_path + "'"};
        out << result.dump() << '\n';
        out.flush();
        if(!out)
            throw std::runtime_error{"failed to write --out file '" + options.out_path + "'"};
    }

    if(!options.json_only)
    {
        print_human(std::cout, result);
        std::cout << std::flush;
    }
    // Flush explicitly and through the C stdio layer: the LC runtime is torn down
    // from static destructors after `main` returns, and an unflushed result line
    // is lost when stdout is a console (observed: empty stdout, exit code 0).
    std::cout << result.dump() << std::endl;
    std::fflush(stdout);
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    const std::string_view program = (argc > 0 && argv[0] != nullptr) ?
                                         std::string_view{argv[0]} :
                                         std::string_view{"?"};

    try
    {
        const auto options = parse_options(argc, argv);
        if(options.help)
        {
            std::cout << usage_text(program) << std::flush;
            return 0;
        }
        return run(options);
    }
    catch(const UsageError& e)
    {
        std::cerr << "error: " << e.what() << "\n\n"
                  << usage_text(program) << std::flush;
        return 2;
    }
    catch(const std::exception& e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
