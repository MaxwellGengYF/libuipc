#pragma once
/********************************************************************
 * @file bench_report.h
 * @brief Result records and the schema of bench_plan.md §6.4 (uipc-bench/v1).
 *
 * Two artefacts per run, both written next to the data of the existing runner
 * (output/benchmark-runs/):
 *
 *   bench-<backend>-<runid>.csv          one row per case, flushed immediately
 *                                        (a fatal CUDA case must not lose the file)
 *   <case>-<backend>-<runid>.json        one JSON object per case, written when
 *                                        that case finishes
 *
 * The mandatory attribution fields of §6.4 (cache, lc_revision,
 * backend_dll_sha256, runs, agreement_pct) come from `Environment` + `Timing`.
 *********************************************************************/
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <common/bench_harness.h>

namespace uipc::benchmark
{
/// Everything that must be identical for two records to be comparable (§7.2).
struct Environment
{
    std::string   schema = "uipc-bench/v1";
    std::string   run_id;
    std::string   timestamp;
    std::string   backend;
    std::string   backend_dll;
    std::string   backend_dll_sha256;
    std::string   lc_revision;
    std::string   lc_prefix;
    std::string   device;
    std::string   driver;
    std::string   runtime_version;
    std::uint64_t cache_entries = 0;
    std::uint64_t cache_bytes   = 0;
    std::string   cache_dir;
    std::string   executable;
    std::string   git_commit;
    std::string   command_line;
    std::string   device_config;  ///< LC DeviceConfig flags in force
    std::string   extra;          ///< free-form, e.g. "graph-nodes=1000"
};

/// One measured case, one backend (§6.4).
struct CaseRecord
{
    std::string id;         ///< "H0.1", "K1.spmv.rbk_sym", "S5.pcg"
    std::string tier;       ///< "T0".."T3"
    std::string scenario;   ///< human-readable one line
    std::string work_unit;  ///< "dispatch", "triplet", "pair", "body", ...

    std::uint64_t work_units = 0;  ///< total work units in one rep
    Timing        timing;

    double per_call_us      = 0.0;  ///< one rep
    double per_unit_us      = 0.0;  ///< one work unit
    double units_per_second = 0.0;  ///< throughput
    double bytes_per_second = 0.0;  ///< when `bytes_per_unit` is set
    double bytes_per_unit   = 0.0;

    double      checksum          = 0.0;
    double      expected_checksum = 0.0;
    bool        checksum_ok       = true;
    std::string checksum_note;

    // §6.5 counters as observed around the measured window.
    std::uint64_t dispatches           = 0;
    std::uint64_t commits              = 0;
    std::uint64_t syncs                = 0;
    double        sync_ms              = 0.0;
    std::uint64_t h2d_bytes            = 0;
    std::uint64_t d2h_bytes            = 0;
    std::uint64_t graph_captures       = 0;
    std::uint64_t graph_instantiates   = 0;
    std::uint64_t graph_updates        = 0;
    std::uint64_t graph_launches       = 0;
    std::uint64_t graph_refusals       = 0;
    std::uint64_t graph_rebuilds       = 0;
    double        graph_instantiate_ms = 0.0;
    double        graph_update_ms      = 0.0;

    std::vector<std::string> flags;  ///< submit_bound / sync_bound / kernel_bound / ...
    std::string notes;

    /// Derived: commit / sync / dispatch per rep, the numbers §9 G1 asks for.
    [[nodiscard]] double dispatches_per_rep() const noexcept
    {
        return timing.reps == 0u ?
                   0.0 :
                   static_cast<double>(dispatches) / static_cast<double>(timing.reps);
    }
    [[nodiscard]] double syncs_per_rep() const noexcept
    {
        return timing.reps == 0u ?
                   0.0 :
                   static_cast<double>(syncs) / static_cast<double>(timing.reps);
    }
};

namespace detail
{
    std::string json_escape(std::string_view s);
}

/// Writes the CSV row stream and the per-case JSON records.
class BenchReport
{
  public:
    BenchReport(Environment env, std::string out_dir);
    ~BenchReport();
    BenchReport(const BenchReport&)            = delete;
    BenchReport& operator=(const BenchReport&) = delete;

    /// Append one case: the CSV row is flushed before this returns, the JSON
    /// record is written when the case's record is complete (same call).
    void add(const CaseRecord& record);

    /// Print a one-screen summary of everything added so far.
    void summary() const;

    [[nodiscard]] const Environment& environment() const noexcept
    {
        return m_env;
    }
    [[nodiscard]] const std::string& csv_path() const noexcept
    {
        return m_csv_path;
    }

  private:
    Environment             m_env;
    std::string             m_out_dir;
    std::string             m_csv_path;
    std::FILE*              m_csv = nullptr;
    std::vector<CaseRecord> m_records;
    bool                    m_header_written = false;
};
}// namespace uipc::benchmark
