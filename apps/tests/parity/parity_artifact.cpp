/********************************************************************
 * @file parity_artifact.cpp
 * @brief T9.3a — manifest + raw array writer (see parity_artifact.h).
 *********************************************************************/
#include <parity_artifact.h>

#include <parity_checksum.h>

#include <bit>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace uipc::parity
{
namespace
{
    namespace fs = std::filesystem;

    static_assert(std::endian::native == std::endian::little,
                  "the parity artifact format is raw little-endian float64 only "
                  "(03-parity-harness.md §3.3); this build target is big-endian");

    /// `frame_000_vertices_g0.f64` / `frame_000_transforms.f64`.
    std::string array_file_name(SizeT frame, const ArrayBlock& block)
    {
        std::ostringstream os;
        os << "frame_" << std::setw(3) << std::setfill('0') << frame << '_'
           << block.probe;
        if(block.geometry >= 0)
            os << "_g" << block.geometry;
        os << ".f64";
        return os.str();
    }

    void write_doubles(const fs::path& path, const std::vector<double>& data)
    {
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        if(!out)
            throw std::runtime_error{"cannot open parity array file '" + path.string() + "'"};
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(double)));
        out.flush();
        if(!out)
            throw std::runtime_error{"failed to write parity array file '"
                                     + path.string() + "'"};
    }

    void write_text(const fs::path& path, std::string_view text)
    {
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        if(!out)
            throw std::runtime_error{"cannot open parity file '" + path.string() + "'"};
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.flush();
        if(!out)
            throw std::runtime_error{"failed to write parity file '" + path.string() + "'"};
    }
}  // namespace

double arrays_checksum(const std::vector<ArrayBlock>& arrays) noexcept
{
    std::uint64_t digest = kFnvOffsetBasis;
    for(const auto& block : arrays)
    {
        // The block order, name, geometry index and element count are part of the
        // digest: two runs whose probe set differs cannot produce the same checksum.
        digest = hash_string(digest, block.probe);
        digest = hash_int(digest, block.geometry);
        digest = hash_int(digest, static_cast<std::int64_t>(block.data.size()));
        for(double value : block.data)
            digest = hash_double(digest, value);
    }
    return checksum_value(digest);
}

ArtifactWriter::ArtifactWriter(std::string dir)
    : m_dir{std::move(dir)}
{
    std::error_code ec;
    fs::create_directories(fs::path{m_dir}, ec);
    if(ec)
        throw std::runtime_error{"cannot create parity output directory '"
                                 + m_dir + "': " + ec.message()};
    // Start from an empty scalars file: a second run into the same directory must not
    // append to the first one's frames.
    write_text(fs::path{m_dir} / "scalars.jsonl", "");
}

double ArtifactWriter::write_frame(SizeT                          frame,
                                   const std::vector<ArrayBlock>& arrays,
                                   const Json&                    scalars)
{
    Json array_json = Json::array();

    for(const auto& block : arrays)
    {
        const auto file = array_file_name(frame, block);
        write_doubles(fs::path{m_dir} / file, block.data);

        Json entry          = Json::object();
        entry["probe"]      = block.probe;
        entry["geometry"]   = block.geometry;
        entry["dtype"]      = "f64";
        entry["components"] = block.components;
        entry["count"] = block.data.size() / static_cast<SizeT>(block.components);
        entry["file"] = file;
        array_json.push_back(std::move(entry));
    }

    const double checksum = arrays_checksum(arrays);

    m_arrays.push_back(std::move(array_json));
    Json checksum_entry     = Json::object();
    checksum_entry["frame"] = frame;
    checksum_entry["value"] = checksum;
    m_checksums.push_back(std::move(checksum_entry));

    Json line     = Json::object();
    line["frame"] = frame;
    line["stats"] = scalars;
    std::ofstream out{fs::path{m_dir} / "scalars.jsonl", std::ios::binary | std::ios::app};
    if(!out)
        throw std::runtime_error{"cannot append to scalars.jsonl in '" + m_dir + "'"};
    out << line.dump() << '\n';
    out.flush();
    if(!out)
        throw std::runtime_error{"failed to append to scalars.jsonl in '" + m_dir + "'"};
    ++m_scalar_frames;

    return checksum;
}

void ArtifactWriter::add_to_manifest(Json& manifest) const
{
    manifest["arrays"]    = m_arrays;
    manifest["checksums"] = m_checksums;
    Json scalars          = Json::object();
    scalars["file"]       = "scalars.jsonl";
    scalars["frames"]     = m_scalar_frames;
    manifest["scalars"]   = std::move(scalars);
}

void ArtifactWriter::write_manifest(const Json& manifest) const
{
    write_text(fs::path{m_dir} / "manifest.json", manifest.dump(2) + "\n");
}

void ArtifactWriter::write_skip(const std::string& dir, std::string_view backend, std::string_view reason)
{
    std::error_code ec;
    fs::create_directories(fs::path{dir}, ec);
    Json skip       = Json::object();
    skip["schema"]  = "uipc-parity-skip/1";
    skip["status"]  = "skipped";
    skip["backend"] = std::string{backend};
    skip["reason"]  = std::string{reason};
    write_text(fs::path{dir} / "SKIP.json", skip.dump(2) + "\n");
}
}  // namespace uipc::parity
