/********************************************************************
 * @file bench_report.cpp
 * @brief CSV row stream + per-case JSON, schema uipc-bench/v1 (bench_plan.md §6.4).
 *********************************************************************/
#include <common/bench_report.h>

#include <filesystem>
#include <string>

namespace uipc::benchmark
{
namespace fs = std::filesystem;

namespace detail
{
    std::string json_escape(std::string_view s)
    {
        std::string out;
        out.reserve(s.size() + 8u);
        for(char c : s)
        {
            switch(c)
            {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    if(static_cast<unsigned char>(c) < 0x20u)
                    {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                        out += buf;
                    }
                    else
                    {
                        out.push_back(c);
                    }
            }
        }
        return out;
    }
}// namespace detail

namespace
{
    void write_json_array(std::FILE* f, const char* key, const std::vector<double>& values)
    {
        std::fprintf(f, ",\"%s\":[", key);
        for(std::size_t i = 0; i < values.size(); ++i)
            std::fprintf(f, "%s%.6f", i == 0u ? "" : ",", values[i]);
        std::fprintf(f, "]");
    }

    void write_string_array(std::FILE* f, const char* key, const std::vector<std::string>& values)
    {
        std::fprintf(f, ",\"%s\":[", key);
        for(std::size_t i = 0; i < values.size(); ++i)
            std::fprintf(f,
                         "%s\"%s\"",
                         i == 0u ? "" : ",",
                         detail::json_escape(values[i]).c_str());
        std::fprintf(f, "]");
    }
}// namespace

BenchReport::BenchReport(Environment env, std::string out_dir)
    : m_env{std::move(env)}
    , m_out_dir{std::move(out_dir)}
{
    std::error_code ec;
    fs::create_directories(m_out_dir, ec);
    m_csv_path =
        (fs::path{m_out_dir} / ("bench-" + m_env.backend + "-" + m_env.run_id + ".csv"))
            .string();
    m_csv = std::fopen(m_csv_path.c_str(), "wb");
}

BenchReport::~BenchReport()
{
    if(m_csv)
    {
        std::fclose(m_csv);
        m_csv = nullptr;
    }
}

void BenchReport::add(const CaseRecord& record)
{
    m_records.push_back(record);

    if(m_csv)
    {
        if(!m_header_written)
        {
            std::fprintf(m_csv,
                         "case,tier,backend,scenario,work_unit,work_units,reps,rounds,"
                         "median_ms,p95_ms,min_ms,max_ms,cov_pct,per_call_us,per_unit_us,"
                         "units_per_s,bytes_per_s,checksum_ok,dispatches,commits,syncs,sync_ms,"
                         "h2d_bytes,d2h_bytes,graph_captures,graph_instantiates,graph_launches,"
                         "graph_refusals,graph_instantiate_ms,graph_update_ms,flags,notes\n");
            m_header_written = true;
        }
        std::fprintf(m_csv,
                     "%s,%s,%s,%s,%s,%llu,%llu,%u,%.6f,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%.3f,"
                     "%s,%llu,%llu,%llu,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%.6f,%.6f,",
                     record.id.c_str(),
                     record.tier.c_str(),
                     m_env.backend.c_str(),
                     record.scenario.c_str(),
                     record.work_unit.c_str(),
                     static_cast<unsigned long long>(record.work_units),
                     static_cast<unsigned long long>(record.timing.reps),
                     record.timing.rounds,
                     record.timing.median_ms,
                     record.timing.p95_ms,
                     record.timing.min_ms,
                     record.timing.max_ms,
                     record.timing.cov_pct,
                     record.per_call_us,
                     record.per_unit_us,
                     record.units_per_second,
                     record.bytes_per_second,
                     record.checksum_ok ? "1" : "0",
                     static_cast<unsigned long long>(record.dispatches),
                     static_cast<unsigned long long>(record.commits),
                     static_cast<unsigned long long>(record.syncs),
                     record.sync_ms,
                     static_cast<unsigned long long>(record.h2d_bytes),
                     static_cast<unsigned long long>(record.d2h_bytes),
                     static_cast<unsigned long long>(record.graph_captures),
                     static_cast<unsigned long long>(record.graph_instantiates),
                     static_cast<unsigned long long>(record.graph_launches),
                     static_cast<unsigned long long>(record.graph_refusals),
                     record.graph_instantiate_ms,
                     record.graph_update_ms);
        std::string flags;
        for(std::size_t i = 0; i < record.flags.size(); ++i)
            flags += (i == 0u ? "" : "|") + record.flags[i];
        std::fprintf(m_csv,
                     "\"%s\",\"%s\"\n",
                     detail::json_escape(flags).c_str(),
                     detail::json_escape(record.notes).c_str());
        // §6.2 "row flushed incrementally": a CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES in
        // a later case must not lose the rows already measured.
        std::fflush(m_csv);
    }

    // Per-case JSON: written when the case finishes, so a crash in case n+1 keeps
    // the records of case 0..n.
    const auto json_path =
        (fs::path{m_out_dir} / (record.id + "-" + m_env.backend + "-" + m_env.run_id + ".json"))
            .string();
    if(std::FILE* j = std::fopen(json_path.c_str(), "wb"))
    {
        std::fprintf(j, "{\n\"schema\":\"%s\"", m_env.schema.c_str());
        std::fprintf(j, ",\n\"run_id\":\"%s\"", m_env.run_id.c_str());
        std::fprintf(j, ",\n\"timestamp\":\"%s\"", m_env.timestamp.c_str());
        std::fprintf(j, ",\n\"git_commit\":\"%s\"", m_env.git_commit.c_str());
        std::fprintf(j, ",\n\"backend\":\"%s\"", m_env.backend.c_str());
        std::fprintf(j, ",\n\"case\":\"%s\"", record.id.c_str());
        std::fprintf(j, ",\n\"tier\":\"%s\"", record.tier.c_str());
        std::fprintf(j,
                     ",\n\"scenario\":\"%s\"",
                     detail::json_escape(record.scenario).c_str());
        // ---- environment block (§6.4 mandatory fields) ----
        std::fprintf(j, ",\n\"environment\":{");
        std::fprintf(j, "\"device\":\"%s\"", m_env.device.c_str());
        std::fprintf(j, ",\"driver\":\"%s\"", m_env.driver.c_str());
        std::fprintf(j, ",\"runtime_version\":\"%s\"", m_env.runtime_version.c_str());
        std::fprintf(j,
                     ",\"backend_dll\":\"%s\"",
                     detail::json_escape(m_env.backend_dll).c_str());
        std::fprintf(j,
                     ",\"backend_dll_sha256\":\"%s\"",
                     m_env.backend_dll_sha256.c_str());
        std::fprintf(j, ",\"lc_revision\":\"%s\"", m_env.lc_revision.c_str());
        std::fprintf(j,
                     ",\"lc_prefix\":\"%s\"",
                     detail::json_escape(m_env.lc_prefix).c_str());
        std::fprintf(j, ",\"cache\":{");
        std::fprintf(j, "\"dir\":\"%s\"", detail::json_escape(m_env.cache_dir).c_str());
        std::fprintf(j,
                     ",\"entries\":%llu",
                     static_cast<unsigned long long>(m_env.cache_entries));
        std::fprintf(j,
                     ",\"mib\":%.2f}",
                     static_cast<double>(m_env.cache_bytes) / (1024.0 * 1024.0));
        std::fprintf(j,
                     ",\"device_config\":\"%s\"",
                     detail::json_escape(m_env.device_config).c_str());
        std::fprintf(j,
                     ",\"executable\":\"%s\"",
                     detail::json_escape(m_env.executable).c_str());
        std::fprintf(j,
                     ",\"command_line\":\"%s\"",
                     detail::json_escape(m_env.command_line).c_str());
        std::fprintf(j, ",\"extra\":\"%s\"}", detail::json_escape(m_env.extra).c_str());
        // ---- measurement block ----
        std::fprintf(j, ",\n\"work_unit\":\"%s\"", record.work_unit.c_str());
        std::fprintf(j,
                     ",\n\"work_units\":%llu",
                     static_cast<unsigned long long>(record.work_units));
        std::fprintf(j,
                     ",\n\"reps\":%llu",
                     static_cast<unsigned long long>(record.timing.reps));
        std::fprintf(j, ",\n\"rounds\":%u", record.timing.rounds);
        std::fprintf(j, ",\n\"warmup\":%u", record.timing.warmup);
        std::fprintf(j, ",\n\"sync_per_rep\":%s", record.timing.sync_per_rep ? "true" : "false");
        write_json_array(j, "round_ms", record.timing.round_ms);
        std::fprintf(j, ",\n\"median_ms\":%.6f", record.timing.median_ms);
        std::fprintf(j, ",\n\"p95_ms\":%.6f", record.timing.p95_ms);
        std::fprintf(j, ",\n\"min_ms\":%.6f", record.timing.min_ms);
        std::fprintf(j, ",\n\"max_ms\":%.6f", record.timing.max_ms);
        std::fprintf(j, ",\n\"agreement_pct\":%.3f", record.timing.cov_pct);
        std::fprintf(j, ",\n\"per_call_us\":%.6f", record.per_call_us);
        std::fprintf(j, ",\n\"per_unit_us\":%.6f", record.per_unit_us);
        std::fprintf(j, ",\n\"units_per_second\":%.3f", record.units_per_second);
        std::fprintf(j, ",\n\"bytes_per_second\":%.3f", record.bytes_per_second);
        std::fprintf(j, ",\n\"checksum\":%.6f", record.checksum);
        std::fprintf(j, ",\n\"expected_checksum\":%.6f", record.expected_checksum);
        std::fprintf(j, ",\n\"checksum_ok\":%s", record.checksum_ok ? "true" : "false");
        if(!record.checksum_note.empty())
            std::fprintf(j,
                         ",\n\"checksum_note\":\"%s\"",
                         detail::json_escape(record.checksum_note).c_str());
        // ---- counters block (§6.5) ----
        std::fprintf(j, ",\n\"counters\":{");
        std::fprintf(j,
                     "\"dispatches\":%llu,\"commits\":%llu,\"syncs\":%llu,\"sync_ms\":%.6f",
                     static_cast<unsigned long long>(record.dispatches),
                     static_cast<unsigned long long>(record.commits),
                     static_cast<unsigned long long>(record.syncs),
                     record.sync_ms);
        std::fprintf(j,
                     ",\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"dispatches_per_rep\":%.3f,"
                     "\"syncs_per_rep\":%.3f",
                     static_cast<unsigned long long>(record.h2d_bytes),
                     static_cast<unsigned long long>(record.d2h_bytes),
                     record.dispatches_per_rep(),
                     record.syncs_per_rep());
        std::fprintf(j,
                     ",\"graph_captures\":%llu,\"graph_instantiates\":%llu,"
                     "\"graph_updates\":%llu,\"graph_launches\":%llu,\"graph_rebuilds\":%llu,"
                     "\"graph_refusals\":%llu,\"graph_instantiate_ms\":%.6f,"
                     "\"graph_update_ms\":%.6f}",
                     static_cast<unsigned long long>(record.graph_captures),
                     static_cast<unsigned long long>(record.graph_instantiates),
                     static_cast<unsigned long long>(record.graph_updates),
                     static_cast<unsigned long long>(record.graph_launches),
                     static_cast<unsigned long long>(record.graph_rebuilds),
                     static_cast<unsigned long long>(record.graph_refusals),
                     record.graph_instantiate_ms,
                     record.graph_update_ms);
        // The newline belongs to the writer (every other field starts with one); passing it
        // in the key used to emit a malformed object.
        write_string_array(j, "flags", record.flags);
        if(!record.notes.empty())
            std::fprintf(j,
                         ",\n\"notes\":\"%s\"",
                         detail::json_escape(record.notes).c_str());
        std::fprintf(j, "\n}\n");
        std::fclose(j);
    }

    std::printf("[bench] %-22s %-7s median %10.3f ms  %10.3f us/%s",
                record.id.c_str(),
                record.tier.c_str(),
                record.timing.median_ms,
                record.per_unit_us,
                record.work_unit.c_str());
    if(!record.checksum_ok)
        std::printf("  CHECKSUM FAILED");
    if(!record.timing.stable())
        std::printf("  UNSTABLE(cov=%.2f%%)", record.timing.cov_pct);
    std::printf("\n");
    std::fflush(stdout);
}

void BenchReport::summary() const
{
    if(m_records.empty())
        return;
    std::printf("\n=== %s (%s) ===\n", m_env.backend.c_str(), m_env.device.c_str());
    std::printf("%-22s %-14s %12s %12s %10s %8s\n", "case", "work_unit", "median_ms", "us/unit", "units/s", "cov%");
    for(const auto& r : m_records)
    {
        std::printf("%-22s %-14s %12.3f %12.3f %10.3g %8.2f\n",
                    r.id.c_str(),
                    r.work_unit.c_str(),
                    r.timing.median_ms,
                    r.per_unit_us,
                    r.units_per_second,
                    r.timing.cov_pct);
    }
    std::printf("csv: %s\n", m_csv_path.c_str());
}
}// namespace uipc::benchmark
