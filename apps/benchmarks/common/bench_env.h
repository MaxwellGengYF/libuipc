#pragma once
/********************************************************************
 * @file bench_env.h
 * @brief Environment discovery for the record header (§6.4, §7.2, P6).
 *
 * The four facts that make two records comparable-or-not are collected here:
 * the backend module identity (path + sha256), the LC revision, the PTX cache
 * census (<exe>/.cache entries and MiB, cookbook §7.37 rule 8) and the GPU /
 * driver pair.
 *********************************************************************/
#include <cstdint>
#include <string>
#include <string_view>

#include <common/bench_report.h>

namespace uipc::benchmark
{
/// "2026-10-28T14:03:11" in local time.
[[nodiscard]] std::string now_timestamp();

/// "20261028-140311" + 4 random-ish digits, used in file names.
[[nodiscard]] std::string make_run_id();

/// Absolute path of the running executable.
[[nodiscard]] std::string executable_path();

/// Lower-case hex SHA-256 of a file; empty string when the file cannot be read.
[[nodiscard]] std::string sha256_file(std::string_view path);

/// SHA-256 of a byte range (exposed so the harness can self-check one value).
[[nodiscard]] std::string sha256_bytes(const void* data, std::size_t size);

struct CacheCensus
{
    std::string   dir;
    std::uint64_t entries = 0;
    std::uint64_t bytes   = 0;
};

/// Census of the LC PTX cache that sits next to the executable(<exe>/.cache).
[[nodiscard]] CacheCensus cache_census(std::string_view exe_path);

/// Run a command and return its trimmed first line ("" on failure).
[[nodiscard]] std::string run_capture(std::string_view command);

/// Fill the machine-specific half of an Environment (everything but `backend`).
void fill_environment_base(Environment& env, std::string_view program_path);
}// namespace uipc::benchmark
