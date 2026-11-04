/********************************************************************
 * @file main.cpp
 * @brief T9.3a — uipc_parity_runner: one (scene, backend) leg of the parity harness.
 *
 * The runner is deliberately the *only* thing in the harness that constructs an
 * Engine, and it constructs exactly one: the backend is a CLI argument, the module is
 * loaded by name by core::Engine, so one process can never hold two backends (a second
 * Engine in one process breaks the LC stream — AGENT_ENV_BRIEF/plan §8 trap 1). The
 * comparison therefore always happens between two processes, driven by
 * scripts/parity_run.py, and never inside this binary.
 *
 * Exit codes (03-parity-harness.md §3.2, research §3 D8):
 *   0 ok · 2 usage · 3 backend unavailable (SKIP, not a pass) ·
 *   4 simulation error (engine.status().has_error() or an exception) ·
 *   5 non-finite state detected
 *********************************************************************/
#include <parity_artifact.h>
#include <parity_checksum.h>
#include <parity_probe.h>
#include <parity_scene.h>

#include <uipc/builtin/attribute_name.h>
#include <uipc/common/logger.h>
#include <uipc/common/type_define.h>
#include <uipc/core/engine.h>
#include <uipc/core/scene.h>
#include <uipc/core/world.h>

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace
{
namespace fs = std::filesystem;
using namespace uipc;
using namespace uipc::parity;

// ---------------------------------------------------------------------------
//  exit codes (research §3 D8)
// ---------------------------------------------------------------------------
constexpr int kExitOk    = 0;
constexpr int kExitUsage = 2;
constexpr int kExitUnavailable = 3;  // SKIP: the reference backend is not built here
constexpr int kExitSimError  = 4;
constexpr int kExitNonFinite = 5;

// ---------------------------------------------------------------------------
//  command line
// ---------------------------------------------------------------------------
constexpr std::string_view kOptionsHelp =
    R"(
    --scene=NAME            scene registry entry (see --list)
    --backend=NAME          backend module to load: luisa | none | cuda
    --out=PATH              artifact root; the leg writes into PATH/NAME/backend
                            (default: output/parity/<yyyy-mm-dd>/<scene>)
    --frames=N              recorded frames, N > 0 (default 3)
    --warmup=N              unrecorded frames before them (default 0)
    --dt=SECONDS            scene timestep (default 0.01)
    --dump-linear-system    set extras/debug/dump_linear_system=1 and copy
                            A.mtx/b.mtx/x.mtx from the workspace when present
    --list                  print the scene registry and exit
    --help                  print this message
  )";

/// The binary name is the build target's business.
std::string usage_text(std::string_view program)
{
    return "usage: " + std::string{program} + " --scene=NAME --backend=NAME [options]\n"
           + std::string{kOptionsHelp};
}

class UsageError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

struct Options
{
    std::string scene              = {};
    std::string backend            = {};
    std::string out                = {};
    SizeT       frames             = 3;
    SizeT       warmup             = 0;
    double      dt                 = 0.01;
    bool        dump_linear_system = false;
    bool        list               = false;
    bool        help               = false;
};

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
        throw UsageError{"--" + std::string{option} + " expects a non-negative integer, got '"
                         + std::string{text} + "'"};
    return value;
}

double parse_double(std::string_view text, std::string_view option)
{
    double      value        = 0.0;
    const auto* first        = text.data();
    const auto* last         = text.data() + text.size();
    const auto [stop, error] = std::from_chars(first, last, value);
    if(error != std::errc{} || stop != last || !(value > 0.0))
        throw UsageError{"--" + std::string{option} + " expects a positive number, got '"
                         + std::string{text} + "'"};
    return value;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for(int i = 1; i < argc; ++i)
    {
        const std::string_view arg =
            (argv[i] != nullptr) ? std::string_view{argv[i]} : std::string_view{};
        std::string_view value;
        if(arg == "--help" || arg == "-h")
            options.help = true;
        else if(arg == "--list")
            options.list = true;
        else if(arg == "--dump-linear-system")
            options.dump_linear_system = true;
        else if(take_value(arg, "--scene", value))
            options.scene = value;
        else if(take_value(arg, "--backend", value))
            options.backend = value;
        else if(take_value(arg, "--out", value))
            options.out = value;
        else if(take_value(arg, "--frames", value))
            options.frames = parse_size(value, "frames");
        else if(take_value(arg, "--warmup", value))
            options.warmup = parse_size(value, "warmup");
        else if(take_value(arg, "--dt", value))
            options.dt = parse_double(value, "dt");
        else
            throw UsageError{"unknown argument '" + std::string{arg} + "'"};
    }
    return options;
}

// ---------------------------------------------------------------------------
//  process/exe plumbing
// ---------------------------------------------------------------------------
/// Absolute path of this executable (the backend module is resolved next to it, the
/// same way core::Engine resolves `uipc_backend_<name>.dll`).
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

std::string module_name(std::string_view backend)
{
    return "uipc_backend_" + std::string{backend} + ".dll";
}

Json file_stamp(const fs::path& path)
{
    Json            stamp = Json::object();
    std::error_code ec;
    if(!fs::exists(path, ec))
    {
        stamp["path"]  = path.string();
        stamp["state"] = "absent";
        return stamp;
    }
    stamp["path"]   = path.string();
    stamp["state"]  = "present";
    stamp["size"]   = static_cast<std::int64_t>(fs::file_size(path, ec));
    const auto time = fs::last_write_time(path, ec);
    if(!ec)
    {
        // Seconds since the file-clock epoch: portable, no local-time zone in the
        // artifact (a stale DLL/mtime difference is what has to be visible).
        stamp["mtime_s"] = static_cast<std::int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch())
                .count());
    }
    return stamp;
}

std::string today()
{
    const auto        now = std::chrono::system_clock::now();
    const std::time_t t   = std::chrono::system_clock::to_time_t(now);
    std::tm           tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
    return buffer;
}

// ---------------------------------------------------------------------------
//  engine plumbing
// ---------------------------------------------------------------------------
void require_healthy(core::World& world, core::Engine& engine, std::string_view stage)
{
    if(engine.status().has_error())
    {
        throw std::runtime_error{std::string{stage} + ": the engine reported an error: "
                                 + engine.status().to_json().dump()};
    }
    if(!world.is_valid())
    {
        throw std::runtime_error{std::string{stage} + ": the world is invalid (see the engine log)"};
    }
}

Json scene_config_for(const Options& options)
{
    // The scene config is schema-validated and rejects unknown keys
    // (src/core/core/scene_default_config.cpp), so only registered keys appear here.
    Json config                       = core::Scene::default_config();
    config["dt"]                      = static_cast<Float>(options.dt);
    config["gravity"]                 = Vector3{0.0, -9.8, 0.0};
    config["integrator"]["type"]      = std::string{"bdf1"};
    config["contact"]["enable"]       = IndexT{0};
    config["contact"]["constitution"] = std::string{"ipc"};
    config["sanity_check"]["enable"]  = IndexT{0};
    if(options.dump_linear_system)
        config["extras"]["debug"]["dump_linear_system"] = IndexT{1};
    return config;
}

/// `--dump-linear-system` is a copy-if-present path, never a dumper of our own:
/// the CUDA writer is unreachable in a build without nvcc and the Luisa reader is
/// still commented out (research §2). A missing file is recorded as "absent", which
/// the comparison driver treats as "this gate did not run" instead of "identical".
Json copy_linear_system(const fs::path& workspace, const fs::path& out_dir)
{
    Json result = Json::object();
    for(const char* name : {"A.mtx", "b.mtx", "x.mtx"})
    {
        std::error_code ec;
        const auto      source = workspace / name;
        if(!fs::exists(source, ec))
        {
            result[name] = "absent";
            continue;
        }
        const auto target = out_dir / name;
        fs::copy_file(source, target, fs::copy_options::overwrite_existing, ec);
        result[name] = ec ? "copy_failed" : "present";
    }
    return result;
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
            std::printf("%s", usage_text(program).c_str());
            return kExitOk;
        }

        if(options.list)
        {
            for(const auto& entry : scenes())
            {
                std::printf("%-16s %s\n",
                            std::string{entry.name}.c_str(),
                            std::string{entry.provenance}.c_str());
            }
            return kExitOk;
        }

        if(options.scene.empty() || options.backend.empty() || options.frames == 0)
            throw UsageError{"--scene, --backend and a positive --frames are required"};

        const auto* scene_entry = find_scene(options.scene);
        if(scene_entry == nullptr)
            throw UsageError{"unknown scene '" + options.scene + "' (see --list)"};

        const fs::path exe    = fs::path{program_path()}.parent_path();
        const fs::path module = exe / module_name(options.backend);

        const std::string out_root =
            options.out.empty() ?
                (fs::path{"output"} / "parity" / today() / options.scene).string() :
                options.out;
        const fs::path artifact_dir = fs::path{out_root} / options.backend;

        // ---- SKIP contract (research §3 D7): no module, no run, exit 3 ----------
        {
            std::error_code ec;
            if(!fs::exists(module, ec))
            {
                const std::string reason = module_name(options.backend)
                                           + " not found next to " + exe.string();
                ArtifactWriter::write_skip(artifact_dir.string(), options.backend, reason);
                std::fprintf(stderr,
                             "SKIP backend=%s scene=%s reason=%s\n",
                             options.backend.c_str(),
                             options.scene.c_str(),
                             reason.c_str());
                return kExitUnavailable;
            }
        }

        ArtifactWriter writer{artifact_dir.string()};

        Json manifest          = Json::object();
        manifest["schema"]     = "uipc-parity/1";
        manifest["backend"]    = options.backend;
        manifest["scene"]      = options.scene;
        manifest["provenance"] = std::string{scene_entry->provenance};
        manifest["frames"]     = options.frames;
        manifest["warmup"]     = options.warmup;
        manifest["dt"]         = options.dt;

        const Json description        = scene_entry->describe();
        manifest["scene_description"] = description;
        manifest["scene_hash"] =
            checksum_value(hash_string(kFnvOffsetBasis, description.dump()));

        Json host = Json::object();
#if defined(_WIN32)
        host["os"] = "windows";
#elif defined(__linux__)
        host["os"] = "linux";
#else
        host["os"] = "unknown";
#endif
        host["exe"]        = program_path();
        manifest["host"]   = std::move(host);
        manifest["module"] = file_stamp(module);
        {
            // A stale LC runtime next to the executable silently changes the numbers
            // (cookbook §1.5); its stamp makes that visible in the artifact. The md5
            // of the backend module is added by scripts/parity_run.py.
            Json lc            = Json::object();
            lc["runtime"]      = file_stamp(exe / "luisa-core.dll");
            lc["program_path"] = program_path();
            manifest["lc"]     = std::move(lc);
        }

        int  exit_code = kExitOk;
        Json frame_ms  = Json::array();
        try
        {
            // ---- scene + engine ------------------------------------------------
            logger::set_level(Logger::Level::info);

            const Json scene_config  = scene_config_for(options);
            manifest["scene_config"] = scene_config;

            core::Scene scene{scene_config};
            scene_entry->build(scene);

            const fs::path workspace = artifact_dir / "workspace";
            fs::create_directories(workspace);

            Json engine_config        = core::Engine::default_config();
            manifest["engine_config"] = engine_config;

            core::Engine engine{options.backend, workspace.string(), engine_config};
            core::World world{engine};

            world.init(scene);
            require_healthy(world, engine, "World::init");
            std::printf("init ok (scene=%s backend=%s)\n",
                        options.scene.c_str(),
                        options.backend.c_str());
            std::fflush(stdout);

            // The "both legs fed identical data" gate: the initial probe set is
            // recorded, so a scene that silently differs between two legs cannot even
            // get to the first frame with matching artifacts.
            manifest["input_hash"] = arrays_checksum(collect_probes(world));

            for(SizeT frame = 0; frame < options.warmup; ++frame)
            {
                world.advance();
                world.sync();
                require_healthy(world, engine, "warm-up frame");
            }

            for(SizeT frame = 0; frame < options.frames; ++frame)
            {
                const auto start = std::chrono::steady_clock::now();
                world.advance();
                world.sync();
                const auto stop = std::chrono::steady_clock::now();
                require_healthy(world, engine, "frame " + std::to_string(frame));

                const auto arrays = collect_probes(world);
                if(!all_finite(arrays))
                {
                    std::fprintf(stderr,
                                 "non-finite state detected after frame %zu (scene=%s "
                                 "backend=%s)\n",
                                 frame,
                                 options.scene.c_str(),
                                 options.backend.c_str());
                    exit_code = kExitNonFinite;
                    break;
                }

                const double millis =
                    std::chrono::duration<double, std::milli>(stop - start).count();
                frame_ms.push_back(millis);

                const double checksum =
                    writer.write_frame(frame, arrays, frame_scalars(engine));
                std::printf("frame %zu checksum %.17g (%.1f ms)\n", frame, checksum, millis);
            }

            if(options.dump_linear_system)
                manifest["linear_system"] = copy_linear_system(workspace, artifact_dir);
            else
            {
                Json absent               = Json::object();
                absent["A.mtx"]           = "disabled";
                absent["b.mtx"]           = "disabled";
                absent["x.mtx"]           = "disabled";
                manifest["linear_system"] = std::move(absent);
            }
        }
        catch(const std::exception& e)
        {
            std::fprintf(stderr,
                         "simulation error (scene=%s backend=%s): %s\n",
                         options.scene.c_str(),
                         options.backend.c_str(),
                         e.what());
            exit_code = kExitSimError;
        }

        // Energies: the scene key `line_search/report_energy` logs a block to the
        // logger (src/backends/cuda/line_search/line_searcher.cu:103-119), it never
        // writes `energy.txt`; the ported backend writes neither. Recorded as
        // "absent" so the comparison driver knows the gate did not run.
        manifest["energies"]  = "absent";
        manifest["frame_ms"]  = frame_ms;
        manifest["exit_code"] = exit_code;

        writer.add_to_manifest(manifest);
        writer.write_manifest(manifest);
        std::printf("artifact %s (exit_code %d)\n", artifact_dir.string().c_str(), exit_code);
        std::fflush(stdout);
        return exit_code;
    }
    catch(const UsageError& e)
    {
        std::fprintf(stderr, "%s\n%s", e.what(), usage_text(program).c_str());
        return kExitUsage;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return kExitSimError;
    }
}
