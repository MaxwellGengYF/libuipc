#pragma once
/********************************************************************
 * @file parity_checksum.h
 * @brief T9.3a — per-frame state digest of the parity harness.
 *
 * This is a *copy* of the checksum in apps/benchmarks/engine/ab_main.cpp:275-291
 * (kFnvOffsetBasis / kFnvPrime / hash_double), not a re-derivation: the whole point
 * of the digest is that the engine A/B driver and the parity harness agree on what
 * "the same state" means, so the two definitions must stay line-for-line identical.
 * Only the two additions marked "parity:" below are new.
 *
 * Contract (see 03-parity-harness.md §3.3):
 *   * byte-wise FNV-1a over the raw IEEE-754 bit pattern of every dumped double, so
 *     `-0.0` vs `0.0` (a genuine difference) cannot cancel the way a floating-point
 *     sum of the coordinates would;
 *   * the digest is truncated to 53 bits so the printed value is exactly
 *     representable as a `double` and two processes' manifests compare bit-for-bit.
 *********************************************************************/
#include <bit>
#include <cstdint>
#include <string_view>

namespace uipc::parity
{
inline constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
inline constexpr std::uint64_t kFnvPrime       = 1099511628211ull;

/// FNV-1a over the eight bytes of `value`'s IEEE-754 representation, little-endian
/// byte order (ab_main.cpp:278-291).
inline std::uint64_t hash_double(std::uint64_t seed, double value) noexcept
{
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    std::uint64_t       hash = seed;
    for(int byte = 0; byte < 8; ++byte)
    {
        hash ^= (bits >> (8 * byte)) & 0xFFull;
        hash *= kFnvPrime;
    }
    return hash;
}

/// parity: the same digest over the bytes of a string (the scene description, so a
/// scene-registry drift between two legs is visible in the manifest).
inline std::uint64_t hash_string(std::uint64_t seed, std::string_view text) noexcept
{
    std::uint64_t hash = seed;
    for(char c : text)
    {
        hash ^= static_cast<unsigned char>(c);
        hash *= kFnvPrime;
    }
    return hash;
}

/// parity: an integral value folded in as its 8 little-endian bytes.
inline std::uint64_t hash_int(std::uint64_t seed, std::int64_t value) noexcept
{
    std::uint64_t hash = seed;
    const auto    bits = static_cast<std::uint64_t>(value);
    for(int byte = 0; byte < 8; ++byte)
    {
        hash ^= (bits >> (8 * byte)) & 0xFFull;
        hash *= kFnvPrime;
    }
    return hash;
}

/// The value recorded in the manifest: `static_cast<double>(digest >> 11)`
/// (ab_main.cpp:367), i.e. 53 significant bits, exactly representable.
inline double checksum_value(std::uint64_t digest) noexcept
{
    return static_cast<double>(digest >> 11);
}
}  // namespace uipc::parity
