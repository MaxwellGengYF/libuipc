#pragma once
/********************************************************************
 * @file parity_artifact.h
 * @brief T9.3a — the parity artifact writer (manifest.json + raw arrays).
 *
 * Layout (03-parity-harness.md §3.3, research §3 D4). Everything lives under
 * `<out>/<backend>/`:
 *
 *   manifest.json                  every array named and shaped, one checksum per frame
 *   scalars.jsonl                  one `{"frame":i,"stats":{...}}` line per frame
 *   frame_<NNN>_<probe>[_g<k>].f64 raw little-endian float64, no compression
 *   SKIP.json                      only when the backend module is missing
 *
 * Format rules: raw little-endian float64 only (both legs run on the same machine in
 * the same CI job, so no endianness negotiation and no compression); arrays are named
 * and shaped in the manifest so a reader never guesses geometry order; one 53-bit
 * truncated FNV-1a digest per frame makes "identical or not" a fast gate before any
 * tolerance work.
 *********************************************************************/
#include <parity_probe.h>

#include <uipc/common/json.h>
#include <uipc/common/type_define.h>

#include <string>
#include <string_view>
#include <vector>

namespace uipc::parity
{
/// The per-frame checksum of one probe set: FNV-1a over (probe name, geometry index,
/// element count, every element) of every block, truncated to 53 bits. The runner
/// also uses it on the *initial* state, which is the "both legs fed identical data"
/// gate recorded as `input_hash`.
double arrays_checksum(const std::vector<ArrayBlock>& arrays) noexcept;

class ArtifactWriter
{
  public:
    /// Creates `dir` (and its parents). Throws on failure.
    explicit ArtifactWriter(std::string dir);

    const std::string& dir() const noexcept { return m_dir; }

    /// Writes the frame's raw arrays and its scalars.jsonl line; returns the frame
    /// checksum (`checksum_value()` of the FNV-1a digest over every array element).
    double write_frame(SizeT frame, const std::vector<ArrayBlock>& arrays, const Json& scalars);

    /// The manifest fields this writer owns: `arrays`, `checksums`, `scalars`.
    void add_to_manifest(Json& manifest) const;

    /// Writes `manifest.json` into dir() (indented, so a human can diff two runs).
    void write_manifest(const Json& manifest) const;

    /// Writes `SKIP.json` into `dir` (the research §3 D7 skip contract).
    static void write_skip(const std::string& dir, std::string_view backend, std::string_view reason);

  private:
    std::string m_dir;
    Json        m_arrays        = Json::array();
    Json        m_checksums     = Json::array();
    SizeT       m_scalar_frames = 0;
};
}  // namespace uipc::parity
