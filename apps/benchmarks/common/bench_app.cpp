/********************************************************************
 * @file bench_app.cpp
 * @brief The shared skeleton of every `uipc_bench_<tier>_<backend>` binary
 *        (bench_plan.md §6.1): argument parsing, the environment header, the case
 *        loop, the report and the cache census.
 *
 * The tier binary supplies two things and nothing else: its case list and its
 * device (`AppDeviceHooks`).  Tier 0 passes `BenchDevice` (submission), Tier 1 passes
 * `BenchKernels` (kernels), and both get an identical record schema -- which is what
 * makes a Tier-0 row and a Tier-1 row comparable in `summary.csv`.
 *********************************************************************/
#include <common/bench_app.h>
#include <common/bench_env.h>
#include <common/bench_report.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace uipc::benchmark
{

namespace
{
void usage(const char* exe)
{
    std::printf(
        "usage: %s [options]\n"
        "  --list                     list the registered cases and exit\n"
        "  --case <id>                run one case (prefix match allowed)\n"
        "  --tier <t0|t1|...>         run one tier\n"
        "  --out <dir>                output directory (default output/benchmark-runs)\n"
        "  --rounds <n>               measured rounds (default 3, §7.1)\n"
        "  --warmup <n>               discarded rounds (default 1)\n"
        "  --dispatches <n>           H0.1/H0.2 dispatch count (default 100000)\n"
        "  --graph-nodes <n>          nodes per captured graph (default 1000)\n"
        "  --graph-replays <n>        graph replays per round (default 20)\n"
        "  --sync-pairs <n>           H0.4 dispatch+sync pairs (default 1000)\n"
        "  --threads <a,b,c>          H0.5 thread counts (default 1000,100000,1000000,10000000)\n"
        "  --per-frame <n>            dispatches/frame for the graph-economics line "
        "(default 383)\n"
        "  --extra <text>             free-form text stored in the record header\n",
        exe);
}

std::vector<std::uint32_t> parse_list(const char* s)
{
    std::vector<std::uint32_t> out;
    std::string                cur;
    for(const char* p = s;; ++p)
    {
        if(*p == ',' || *p == '\0')
        {
            if(!cur.empty())
                out.push_back(static_cast<std::uint32_t>(std::stoul(cur)));
            cur.clear();
            if(*p == '\0')
                break;
            continue;
        }
        cur.push_back(*p);
    }
    return out;
}
}// namespace

int bench_app_main(int argc, char** argv, const std::vector<CaseDef>& cases, const AppDeviceHooks& hooks)
{
    Options options;
    options.program_path = argc > 0 ? argv[0] : "";
    for(int i = 0; i < argc; ++i)
        options.command_line += std::string{i == 0 ? "" : " "} + argv[i];

    for(int i = 1; i < argc; ++i)
    {
        const std::string a    = argv[i];
        auto              next = [&](const char* what) -> const char*
        {
            if(i + 1 >= argc)
            {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if(a == "--list")
            options.list_only = true;
        else if(a == "--help" || a == "-h")
        {
            usage(options.program_path.c_str());
            return 0;
        }
        else if(a == "--case")
            options.only_case = next("--case");
        else if(a == "--tier")
            options.only_tier = next("--tier");
        else if(a == "--out")
            options.out_dir = next("--out");
        else if(a == "--rounds")
            options.rounds = static_cast<std::uint32_t>(std::stoul(next("--rounds")));
        else if(a == "--warmup")
            options.warmup = static_cast<std::uint32_t>(std::stoul(next("--warmup")));
        else if(a == "--dispatches")
            options.h01_dispatches = std::stoull(next("--dispatches"));
        else if(a == "--graph-nodes")
            options.graph_nodes =
                static_cast<std::uint32_t>(std::stoul(next("--graph-nodes")));
        else if(a == "--graph-replays")
            options.graph_replays =
                static_cast<std::uint32_t>(std::stoul(next("--graph-replays")));
        else if(a == "--sync-pairs")
            options.sync_pairs =
                static_cast<std::uint32_t>(std::stoul(next("--sync-pairs")));
        else if(a == "--threads")
            options.thread_scale = parse_list(next("--threads"));
        else if(a == "--per-frame")
            options.dispatches_per_frame =
                static_cast<std::uint32_t>(std::stoul(next("--per-frame")));
        else if(a == "--extra")
            options.extra_environment = next("--extra");
        else
        {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(options.program_path.c_str());
            return 2;
        }
    }

    auto matches = [&](const CaseDef& c)
    {
        if(!options.only_tier.empty() && options.only_tier != c.tier)
            return false;
        if(!options.only_case.empty()
           && std::strncmp(c.id, options.only_case.c_str(), options.only_case.size()) != 0)
            return false;
        return true;
    };

    if(options.list_only)
    {
        for(const auto& c : cases)
            std::printf("%-24s %-3s %s\n", c.id, c.tier, c.scenario);
        return 0;
    }

    // device is the tier's (AppDeviceHooks)
    hooks.init(options.program_path);

    Environment env;
    fill_environment_base(env, options.program_path);
    const auto& info    = hooks.info();
    env.backend         = info.backend;
    env.device          = info.device_name;
    env.runtime_version = info.runtime_version;
    env.device_config   = info.extra;
    env.extra           = options.extra_environment;
    env.command_line    = options.command_line;
    // Q4 needs the LC revision next to the module hash: LC numerics change between
    // revisions (cookbook §1.6), so a record without it cannot be reproduced.
    if(info.backend == "luisa")
        env.lc_revision = info.runtime_version;
    if(const char* prefix = std::getenv("UIPC_LC_PREFIX"))
        env.lc_prefix = prefix;

    // The backend module identity (§6.4): the sha256 is what stops a stale DLL from
    // being mislabelled, which is the "stale LC runtime silently changes numbers"
    // risk of the plan's risk table.
    const fs::path exe_dir =
        fs::path{env.executable.empty() ? options.program_path : env.executable}.parent_path();
    const fs::path dll = exe_dir / ("uipc_backend_" + env.backend + ".dll");
    if(fs::exists(dll))
    {
        env.backend_dll        = dll.string();
        env.backend_dll_sha256 = sha256_file(dll.string());
    }
    else
    {
        env.backend_dll = "(not found next to the executable: " + dll.string() + ")";
    }

    BenchReport report{env, options.out_dir};

    std::printf("[bench] backend=%s device=%s runtime=%s\n",
                env.backend.c_str(),
                env.device.c_str(),
                env.runtime_version.c_str());
    std::printf("[bench] dll=%s sha256=%s\n",
                env.backend_dll.c_str(),
                env.backend_dll_sha256.c_str());
    std::printf(
        "[bench] cache dir=%s entries=%llu (%.2f MiB) -- P6: never compare a cold "
        "and a warm run\n",
        env.cache_dir.c_str(),
        static_cast<unsigned long long>(env.cache_entries),
        static_cast<double>(env.cache_bytes) / (1024.0 * 1024.0));
    std::fflush(stdout);

    const auto exe_before = env.cache_entries;

    int ran = 0;
    for(const auto& c : cases)
    {
        if(!matches(c))
            continue;
        std::printf("\n[bench] === %s: %s ===\n", c.id, c.scenario);
        std::fflush(stdout);
        CaseContext ctx{hooks.object(), report, options};
        c.run(ctx);
        ++ran;
    }

    report.summary();
    std::fflush(stdout);

    // P6: the cache census after the run proves whether the run JIT-compiled. An
    // entry-count change across a run means the next run cannot be compared to it.
    const auto after = cache_census(env.executable);
    std::printf("[bench] cache after run: %llu entries (before: %llu)%s\n",
                static_cast<unsigned long long>(after.entries),
                static_cast<unsigned long long>(exe_before),
                after.entries == exe_before ? " -- warm, comparable" :
                                              " -- COLD RUN, do not compare");
    if(ran == 0)
        std::fprintf(stderr, "[bench] no case matched the selector\n");

    hooks.shutdown();
    return ran == 0 ? 1 : 0;
}

}  // namespace uipc::benchmark
