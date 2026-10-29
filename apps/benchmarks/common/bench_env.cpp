/********************************************************************
 * @file bench_env.cpp
 * @brief Environment discovery + the SHA-256 used to fingerprint a backend DLL.
 *********************************************************************/
#include <common/bench_env.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace uipc::benchmark
{
namespace fs = std::filesystem;

// ---- SHA-256 (FIPS 180-4) ------------------------------------------------
// Self-contained on purpose: the harness must not pull a crypto dependency into
// a benchmark binary, and the value it produces is checked against
// `certutil -hashfile` in benchmarks/README.md before it is ever quoted.
namespace
{
    constexpr std::array<std::uint32_t, 64> kK = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    constexpr std::uint32_t rotr(std::uint32_t x, std::uint32_t n) noexcept
    {
        return (x >> n) | (x << (32u - n));
    }

    class Sha256
    {
      public:
        void update(const void* data, std::size_t size)
        {
            const auto* p = static_cast<const unsigned char*>(data);
            for(std::size_t i = 0; i < size; ++i)
            {
                m_block[m_block_len++] = p[i];
                if(m_block_len == 64u)
                {
                    transform(m_block.data());
                    m_total += 512u;
                    m_block_len = 0u;
                }
            }
        }

        [[nodiscard]] std::string final_hex()
        {
            const std::uint64_t bits =
                m_total + static_cast<std::uint64_t>(m_block_len) * 8u;
            // padding: 0x80 then zeros then the 64-bit big-endian length
            std::array<unsigned char, 128> tail{};
            std::size_t                    n = m_block_len;
            for(std::size_t i = 0; i < n; ++i)
                tail[i] = m_block[i];
            tail[n++]               = 0x80u;
            const std::size_t total = (n <= 56u) ? 64u : 128u;
            for(std::size_t i = 0; i < 8u; ++i)
                tail[total - 1u - i] =
                    static_cast<unsigned char>((bits >> (8u * i)) & 0xffu);
            for(std::size_t off = 0; off < total; off += 64u)
                transform(tail.data() + off);

            static constexpr char kHex[] = "0123456789abcdef";
            std::string           out;
            out.reserve(64u);
            for(std::uint32_t word : m_state)
            {
                for(int shift = 28; shift >= 0; shift -= 4)
                    out.push_back(kHex[(word >> shift) & 0xfu]);
            }
            return out;
        }

      private:
        void transform(const unsigned char* block)
        {
            std::array<std::uint32_t, 64> w{};
            for(std::size_t i = 0; i < 16u; ++i)
            {
                w[i] = (static_cast<std::uint32_t>(block[i * 4u]) << 24)
                       | (static_cast<std::uint32_t>(block[i * 4u + 1u]) << 16)
                       | (static_cast<std::uint32_t>(block[i * 4u + 2u]) << 8)
                       | static_cast<std::uint32_t>(block[i * 4u + 3u]);
            }
            for(std::size_t i = 16u; i < 64u; ++i)
            {
                const std::uint32_t s0 = rotr(w[i - 15u], 7u) ^ rotr(w[i - 15u], 18u)
                                         ^ (w[i - 15u] >> 3u);
                const std::uint32_t s1 =
                    rotr(w[i - 2u], 17u) ^ rotr(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
                w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
            }
            std::uint32_t a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3];
            std::uint32_t e = m_state[4], f = m_state[5], g = m_state[6], h = m_state[7];
            for(std::size_t i = 0; i < 64u; ++i)
            {
                const std::uint32_t s1 = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
                const std::uint32_t ch = (e & f) ^ ((~e) & g);
                const std::uint32_t t1 = h + s1 + ch + kK[i] + w[i];
                const std::uint32_t s0 = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
                const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                const std::uint32_t t2  = s0 + maj;
                h                       = g;
                g                       = f;
                f                       = e;
                e                       = d + t1;
                d                       = c;
                c                       = b;
                b                       = a;
                a                       = t1 + t2;
            }
            m_state[0] += a;
            m_state[1] += b;
            m_state[2] += c;
            m_state[3] += d;
            m_state[4] += e;
            m_state[5] += f;
            m_state[6] += g;
            m_state[7] += h;
        }

        std::array<std::uint32_t, 8> m_state{
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::array<unsigned char, 64> m_block{};
        std::size_t                   m_block_len = 0u;
        std::uint64_t                 m_total     = 0u;
    };
}// namespace

std::string sha256_bytes(const void* data, std::size_t size)
{
    Sha256 h;
    h.update(data, size);
    return h.final_hex();
}

std::string sha256_file(std::string_view path)
{
    std::ifstream in(std::string{path}, std::ios::binary);
    if(!in)
        return {};
    Sha256            h;
    std::vector<char> buffer(1u << 16);
    while(in)
    {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto got = static_cast<std::size_t>(in.gcount());
        if(got == 0u)
            break;
        h.update(buffer.data(), got);
    }
    return h.final_hex();
}

// ---- clock / ids ---------------------------------------------------------
std::string now_timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm    tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

std::string make_run_id()
{
    const auto now = std::chrono::system_clock::now();
    const auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm    tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    std::random_device                      rd;
    std::uniform_int_distribution<unsigned> dist(0u, 9999u);
    char                                    tail[8];
    std::snprintf(tail, sizeof(tail), "%04u", dist(rd));
    return std::string{buf} + "-" + tail;
}

std::string executable_path()
{
#if defined(_WIN32)
    char buf[MAX_PATH * 4];
    DWORD len = GetModuleFileNameA(nullptr, buf, static_cast<DWORD>(sizeof(buf)));
    if(len == 0 || len >= sizeof(buf))
        return {};
    return std::string{buf, len};
#else
    char    buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if(len <= 0)
        return {};
    buf[len] = '\0';
    return std::string{buf, static_cast<std::size_t>(len)};
#endif
}

CacheCensus cache_census(std::string_view exe_path)
{
    CacheCensus census;
    if(exe_path.empty())
        return census;
    const fs::path dir = fs::path{std::string{exe_path}}.parent_path() / ".cache";
    census.dir = dir.string();
    std::error_code ec;
    if(!fs::is_directory(dir, ec))
        return census;
    for(const auto& entry : fs::recursive_directory_iterator(dir, ec))
    {
        if(ec)
            break;
        if(!entry.is_regular_file(ec))
            continue;
        ++census.entries;
        census.bytes += entry.file_size(ec);
    }
    return census;
}

std::string run_capture(std::string_view command)
{
#if defined(_WIN32)
    std::FILE* pipe = _popen(std::string{command}.c_str(), "r");
#else
    std::FILE* pipe = popen(std::string{command}.c_str(), "r");
#endif
    if(!pipe)
        return {};
    std::string out;
    char        buf[512];
    while(std::fgets(buf, sizeof(buf), pipe) != nullptr)
    {
        out += buf;
        if(!out.empty() && out.back() == '\n')
            break;  // first line only: every use here is a one-line fact
    }
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    while(!out.empty() && (out.back() == '\n' || out.back() == '\r'))
        out.pop_back();
    return out;
}

void fill_environment_base(Environment& env, std::string_view program_path)
{
    const std::string exe =
        program_path.empty() ? executable_path() : std::string{program_path};
    env.executable = exe;
    env.run_id     = make_run_id();
    env.timestamp  = now_timestamp();
    env.git_commit = run_capture("git -C \"" + fs::path{exe}.parent_path().string()
                                 + "\" rev-parse HEAD 2>nul");
    if(env.git_commit.empty())
        env.git_commit = run_capture("git rev-parse HEAD");
    env.driver = run_capture("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>nul");

    const auto census = cache_census(exe);
    env.cache_dir     = census.dir;
    env.cache_entries = census.entries;
    env.cache_bytes   = census.bytes;
}
}// namespace uipc::benchmark
