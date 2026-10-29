#pragma once
/********************************************************************
 * @file bench_case.h
 * @brief Case registry and run options (bench_plan.md §6.3: "case specification
 *        is data, not code").
 *
 * A case is one measured scenario. The registry is filled by the tier files
 * (`tier0/t0_cases.cpp`, later `tier1/`, `tier2/`), and main() walks it. The
 * same registry source is compiled into the CUDA and the Luisa binary; only the
 * BenchDevice implementation behind `BenchDevice` differs (§6.1).
 *********************************************************************/
#include <cstdint>
#include <string>
#include <vector>

#include <common/bench_device.h>
#include <common/bench_report.h>

namespace uipc::benchmark
{
struct Options
{
    std::string   out_dir = "output/benchmark-runs";
    std::uint32_t rounds  = 3;  ///< §7.1: 3 rounds minimum
    std::uint32_t warmup  = 1;

    /// H0.1/H0.2: 100 000 tiny dispatches (§4.1).
    std::uint64_t h01_dispatches = 100000;
    /// H0.3: nodes per captured graph and replays per measurement.
    std::uint32_t graph_nodes   = 1000;
    std::uint32_t graph_replays = 20;
    /// H0.4: number of dispatch+sync pairs.
    std::uint32_t sync_pairs = 1000;
    /// H0.5 thread counts (1 K .. 10 M).
    std::vector<std::uint32_t> thread_scale{1000u, 100000u, 1000000u, 10000000u};
    /// §4.1 "graph economics": dispatches_per_frame comes from the §6.5 counters
    /// once a frame pipeline exists; until then it is the launch-site count of
    /// the backend (§2.3: 383 on CUDA, 389 record() / 360 dispatch() on Luisa).
    std::uint32_t dispatches_per_frame = 383;

    // Measured per-dispatch costs carried between cases so that G1b can print the
    // three ceilings (plain / batched / graph) side by side once they are known.
    double plain_us_per_dispatch   = 0.0;
    double batched_us_per_dispatch = 0.0;
    double graph_us_per_dispatch   = 0.0;

    bool        list_only = false;
    std::string only_case;     ///< "" = every case in the registry
    std::string only_tier;     ///< "" = every tier
    std::string program_path;  ///< argv[0] (LC needs it)
    std::string command_line;
    std::string extra_environment;  ///< free-form note stored in the record header
};

/**
 * @brief One case's context.  `device` is **type-erased** so that every tier shares one
 * registry type -- and therefore one shared `main`, one report loop and one record
 * schema: Tier 0 passes a `BenchDevice*`, Tier 1 a `BenchKernels*`.  A case casts it
 * back once, in its first line (`auto& device = ctx.as<BenchDevice>();`), which is the
 * price of not duplicating the harness per tier.
 */
struct CaseContext
{
    void*        device = nullptr;
    BenchReport& report;
    Options&     options;

    template <typename T>
    [[nodiscard]] T& as() const noexcept
    {
        return *static_cast<T*>(device);
    }
};

using CaseFn = void (*)(CaseContext&);

struct CaseDef
{
    const char* id;
    const char* tier;
    const char* scenario;
    CaseFn      run;
};

/// Every registered case, in execution order (cheap cases first, per §4.2).
[[nodiscard]] std::vector<CaseDef> const& case_registry();

/// Helpers shared by the tier files.
/// Assert the device buffer equals `expected` element-wise; records the verdict in
/// `record` (§6.2: correctness before timing).
void check_all_equal(BenchDevice& device,
                     BufferHandle buffer,
                     std::size_t  count,
                     float        expected,
                     CaseRecord&  record);

/// Sum of a device buffer (host-side reference).
double sum_host(BenchDevice& device, BufferHandle buffer, std::size_t count);

/// The §4.1 "graph economics" line, printed once per case from the measured
/// us/dispatch, us/graph-dispatch and the dispatches-per-frame assumption.
void print_graph_economics(const char*   case_id,
                           double        us_per_dispatch,
                           double        us_per_graph_dispatch,
                           std::uint32_t dispatches_per_frame);
}// namespace uipc::benchmark
