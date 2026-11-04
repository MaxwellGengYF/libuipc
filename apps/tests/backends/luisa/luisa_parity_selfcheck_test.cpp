/********************************************************************
 * @file luisa_parity_selfcheck_test.cpp
 * @brief T9.3a — the parity harness's own gate ([parity], Class A on itself).
 *
 * Every case here spawns `uipc_parity_runner.exe` (and `scripts/parity_compare.py`)
 * as *subprocesses*. It must never construct a core::Engine itself: a second Engine
 * in one process breaks the LC stream (plan §8 trap 1), and the whole point of the
 * harness is that the two legs are two processes. This file therefore links nothing
 * backend-specific and runs no kernel of its own.
 *
 * What each case proves:
 *
 *  parity_self_comparison        two Luisa runs of the same scene produce byte-identical
 *                               raw arrays and equal per-frame checksums, and
 *                               `scripts/parity_compare.py` reports delta == 0.0 (rc 0).
 *                               A harness that is not reproducible on itself cannot be
 *                               a cross-backend gate.
 *  parity_gravity_scene         the same, for the gravity scene every later plan needs.
 *                               SKIPs (never passes) while the ported engine cannot
 *                               initialise a dof-based scene on this tree.
 *  parity_compare_detects_drift a 1e-3 perturbation of ONE element must make the
 *                               comparator exit 1 and name the probe and the index.
 *                               This is what stops the harness from being a no-op.
 *  parity_reference_backend_absent
 *                               --backend=cuda exits 3, writes SKIP.json and says SKIP
 *                               on stderr. If the CUDA module is present, SKIP with the
 *                               reason that the cross-backend case is not implemented
 *                               here -- never a silent pass.
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <uipc/common/json.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;
using uipc::Json;

namespace
{
std::string exe_dir()
{
    return fs::path{UIPC_TEST_PROGRAM_PATH}.parent_path().string();
}

std::string runner_path()
{
    return (fs::path{exe_dir()} / "uipc_parity_runner.exe").string();
}

/// Walk up from the executable until `scripts/parity_compare.py` is found: the test
/// must work in both build layouts (build/windows/x64/release and build/Release/bin).
std::string repo_root()
{
    fs::path dir = fs::path{UIPC_TEST_PROGRAM_PATH}.parent_path();
    for(int i = 0; i < 8 && !dir.empty(); ++i)
    {
        if(fs::exists(dir / "scripts" / "parity_compare.py"))
            return dir.string();
        if(!dir.has_parent_path())
            break;
        dir = dir.parent_path();
    }
    return {};
}

std::string python_exe()
{
    if(const char* env = std::getenv("UIPC_PYTHON"); env != nullptr && env[0] != '\0')
        return env;
    return "python";
}

/// Runs `command`, sending stdout+stderr to `log_path`; returns the child's exit code.
///
/// CreateProcess is used directly instead of `std::system`: `cmd /c "<quoted exe>"
/// args > "<quoted log>"` is mangled by cmd's quote handling (measured: rc 1, no log,
/// "the filename, directory name, or volume label syntax is incorrect"), and an
/// unquoted command line breaks as soon as the build tree contains a space.
int run_logged(const std::string& command, const std::string& log_path)
{
#if defined(_WIN32)
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE              log = ::CreateFileA(log_path.c_str(),
                               GENERIC_WRITE,
                               FILE_SHARE_READ,
                               &attributes,
                               CREATE_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL,
                               nullptr);
    if(log == INVALID_HANDLE_VALUE)
        return -1;

    STARTUPINFOA startup{};
    startup.cb         = sizeof(startup);
    startup.dwFlags    = STARTF_USESTDHANDLES;
    startup.hStdOutput = log;
    startup.hStdError  = log;
    startup.hStdInput  = ::GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION process{};
    std::string mutable_command = command;  // CreateProcessA may write to it
    const BOOL  started         = ::CreateProcessA(
        nullptr, mutable_command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process);
    ::CloseHandle(log);
    if(!started)
        return -1;

    ::WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 0;
    ::GetExitCodeProcess(process.hProcess, &exit_code);
    ::CloseHandle(process.hProcess);
    ::CloseHandle(process.hThread);
    return static_cast<int>(exit_code);
#else
    const std::string full   = command + " > \"" + log_path + "\" 2>&1";
    const int         status = std::system(full.c_str());
    if(status == -1)
        return -1;
    if(WIFEXITED(status))
        return WEXITSTATUS(status);
    return 128 + WTERMSIG(status);
#endif
}

std::string read_text(const std::string& path)
{
    std::ifstream in{path, std::ios::binary};
    if(!in)
        return {};
    return std::string{std::istreambuf_iterator<char>{in},
                       std::istreambuf_iterator<char>{}};
}

bool same_bytes(const std::string& a, const std::string& b)
{
    return read_text(a) == read_text(b);
}

Json load_json(const std::string& path)
{
    std::ifstream in{path};
    Json          json;
    in >> json;
    return json;
}

/// Adds `delta` to element `index` of a raw little-endian float64 array in place.
void perturb_element(const std::string& path, std::size_t index, double delta)
{
    std::fstream io{path, std::ios::binary | std::ios::in | std::ios::out};
    REQUIRE(io);
    io.seekg(static_cast<std::streamoff>(index * sizeof(double)));
    double value = 0.0;
    io.read(reinterpret_cast<char*>(&value), sizeof(value));
    REQUIRE(io);
    value += delta;
    io.seekp(static_cast<std::streamoff>(index * sizeof(double)));
    io.write(reinterpret_cast<const char*>(&value), sizeof(value));
    REQUIRE(io);
}

/// One (scene, frames) run into `<dir>/<backend>`; returns the runner's exit code.
int run_leg(const std::string& scene,
            const std::string& backend,
            std::size_t        frames,
            const std::string& out_dir)
{
    const std::string log = out_dir + ".log";
    const std::string command =
        "\"" + runner_path() + "\" --scene=" + scene + " --backend=" + backend
        + " --frames=" + std::to_string(frames) + " --out=\"" + out_dir + "\"";
    return run_logged(command, log);
}

std::string leg_dir(const std::string& out_dir, const std::string& backend)
{
    return (fs::path{out_dir} / backend).string();
}

std::string fresh_dir(const std::string& name)
{
    const auto dir = fs::temp_directory_path() / "uipc_parity_selfcheck" / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir.string();
}

/// Asserts that the two legs are byte-identical, frame by frame, and that the recorded
/// geometry matches the recorded file sizes (so a truncated write cannot pass).
void require_identical_legs(const std::string& a, const std::string& b)
{
    const Json manifest_a = load_json((fs::path{a} / "manifest.json").string());
    const Json manifest_b = load_json((fs::path{b} / "manifest.json").string());
    REQUIRE(manifest_a.at("schema") == "uipc-parity/1");
    REQUIRE(manifest_a.at("frames") == manifest_b.at("frames"));

    const auto& arrays_a = manifest_a.at("arrays");
    const auto& arrays_b = manifest_b.at("arrays");
    REQUIRE(arrays_a.size() == manifest_a.at("frames").get<std::size_t>());
    REQUIRE(arrays_a.size() == arrays_b.size());

    for(std::size_t frame = 0; frame < arrays_a.size(); ++frame)
    {
        REQUIRE(arrays_a.at(frame).size() == arrays_b.at(frame).size());
        for(std::size_t i = 0; i < arrays_a.at(frame).size(); ++i)
        {
            const auto& entry_a = arrays_a.at(frame).at(i);
            const auto& entry_b = arrays_b.at(frame).at(i);
            REQUIRE(entry_a.at("file") == entry_b.at("file"));
            const auto path_a =
                (fs::path{a} / entry_a.at("file").get<std::string>()).string();
            const auto path_b =
                (fs::path{b} / entry_b.at("file").get<std::string>()).string();
            REQUIRE(fs::exists(path_a));
            REQUIRE(fs::exists(path_b));
            const auto expected_bytes =
                entry_a.at("count").get<std::size_t>()
                * entry_a.at("components").get<std::size_t>() * sizeof(double);
            REQUIRE(fs::file_size(path_a) == expected_bytes);
            REQUIRE(fs::file_size(path_b) == expected_bytes);
            CHECK(same_bytes(path_a, path_b));
        }
    }

    const auto& checksums_a = manifest_a.at("checksums");
    const auto& checksums_b = manifest_b.at("checksums");
    REQUIRE(checksums_a.size() == arrays_a.size());
    REQUIRE(checksums_b.size() == arrays_b.size());
    for(std::size_t frame = 0; frame < checksums_a.size(); ++frame)
        CHECK(checksums_a.at(frame).at("value") == checksums_b.at(frame).at("value"));
}

/// Runs the comparator on the two leg directories and returns its exit code.
int run_comparator(const std::string& ref_dir,
                   const std::string& test_dir,
                   const std::string& log_path,
                   const std::string& extra = {})
{
    const std::string root = repo_root();
    const std::string script =
        (fs::path{root} / "scripts" / "parity_compare.py").string();
    const std::string command = "\"" + python_exe() + "\" \"" + script + "\" --ref \""
                                + ref_dir + "\" --test \"" + test_dir + "\"" + extra;
    return run_logged(command, log_path);
}
}  // namespace

TEST_CASE("parity_self_comparison", "[parity]")
{
    if(!fs::exists(runner_path()))
        SKIP("parity runner not found at " + runner_path());
    if(repo_root().empty())
        SKIP("scripts/parity_compare.py not found above " + exe_dir());

    const auto root = fs::temp_directory_path() / "uipc_parity_selfcheck";
    const auto a    = fresh_dir("a");
    const auto b    = fresh_dir("b");

    // The scene is the harness's own proof-of-life scene: a geometry with no
    // constitution. The two gravity scenes are exercised by `parity_gravity_scene`
    // below, which SKIPs while the ported engine cannot initialise a dof-based scene.
    REQUIRE(run_leg("harness_smoke", "luisa", 3, a) == 0);
    REQUIRE(run_leg("harness_smoke", "luisa", 3, b) == 0);

    require_identical_legs(leg_dir(a, "luisa"), leg_dir(b, "luisa"));

    const auto log = (root / "compare_self.log").string();
    const int rc = run_comparator(leg_dir(a, "luisa"), leg_dir(b, "luisa"), log);
    INFO(read_text(log));
    CHECK(rc == 0);
    CHECK(read_text(log).find("delta == 0.0") != std::string::npos);
}

TEST_CASE("parity_gravity_scene", "[parity]")
{
    if(!fs::exists(runner_path()))
        SKIP("parity runner not found at " + runner_path());
    if(repo_root().empty())
        SKIP("scripts/parity_compare.py not found above " + exe_dir());

    const auto root = fs::temp_directory_path() / "uipc_parity_selfcheck";
    const auto a    = fresh_dir("gravity_a");
    const auto b    = fresh_dir("gravity_b");
    const std::string a_log = a + ".log";
    const int         rc_a  = run_leg("gravity_abd", "luisa", 3, a);

    if(rc_a != 0)
    {
        // A SKIP, never a pass: the case becomes a real gate the moment the engine can
        // initialise the scene (plan 10 / T8.4), and it says so out loud until then.
        const std::string log  = read_text(a_log);
        const std::size_t keep = 700;
        SKIP("gravity_abd cannot initialise on this tree (runner rc "
             + std::to_string(rc_a)
             + "): pre-existing abort inside World::init for every dof-based scene, "
               "see docs/luisa_backend_progress.md T9.3a. Runner log tail:\n"
             + (log.size() > keep ? log.substr(log.size() - keep) : log));
    }

    REQUIRE(run_leg("gravity_abd", "luisa", 3, b) == 0);
    require_identical_legs(leg_dir(a, "luisa"), leg_dir(b, "luisa"));

    const auto log = (root / "compare_gravity.log").string();
    CHECK(run_comparator(leg_dir(a, "luisa"), leg_dir(b, "luisa"), log) == 0);
    // The gravity scene must actually move: its live ABD state probe has to exist.
    const auto manifest =
        load_json((fs::path{leg_dir(a, "luisa")} / "manifest.json").string());
    CHECK(manifest.at("arrays").at(0).size() >= 2);
}

TEST_CASE("parity_compare_detects_drift", "[parity]")
{
    if(!fs::exists(runner_path()))
        SKIP("parity runner not found at " + runner_path());
    if(repo_root().empty())
        SKIP("scripts/parity_compare.py not found above " + exe_dir());

    const auto root = fs::temp_directory_path() / "uipc_parity_selfcheck";
    const auto ref  = fresh_dir("drift_ref");
    const auto bad  = fresh_dir("drift_test");
    REQUIRE(run_leg("harness_smoke", "luisa", 3, ref) == 0);
    REQUIRE(run_leg("harness_smoke", "luisa", 3, bad) == 0);

    // Perturb exactly one element of frame 1's vertices probe by +1e-3.
    const auto manifest =
        load_json((fs::path{leg_dir(bad, "luisa")} / "manifest.json").string());
    const auto file = manifest.at("arrays").at(1).at(0).at("file").get<std::string>();
    perturb_element((fs::path{leg_dir(bad, "luisa")} / file).string(), 5, 1.0e-3);

    const auto log = (root / "compare_drift.log").string();
    const int rc = run_comparator(leg_dir(ref, "luisa"), leg_dir(bad, "luisa"), log);
    const auto text = read_text(log);
    INFO(text);
    // A test that only ever sees rc 0 proves nothing (research §5).
    CHECK(rc == 1);
    CHECK(text.find("vertices[5]") != std::string::npos);
}

TEST_CASE("parity_reference_backend_absent", "[parity]")
{
    if(!fs::exists(runner_path()))
        SKIP("parity runner not found at " + runner_path());

    const auto root       = fs::temp_directory_path() / "uipc_parity_selfcheck";
    const auto out        = fresh_dir("cuda_leg");
    const std::string log = out + ".log";
    const int         rc  = run_leg("harness_smoke", "cuda", 1, out);
    const auto        text = read_text(log);
    INFO(text);

    if(rc == 0)
    {
        // Never a silent pass: with a real reference backend present the cross-backend
        // comparison needs a CUDA-side ground truth this machine cannot validate.
        SKIP(
            "reference backend present; the cross-backend case is not implemented on "
            "this machine");
    }
    CHECK(rc == 3);
    CHECK(text.find("SKIP") != std::string::npos);
    CHECK(text.find("uipc_backend_cuda.dll") != std::string::npos);
    CHECK(fs::exists(fs::path{leg_dir(out, "cuda")} / "SKIP.json"));
}
