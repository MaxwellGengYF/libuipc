#pragma once
/********************************************************************
 * @file parity_probe.h
 * @brief T9.3a — backend-agnostic probe extraction (public uipc API only).
 *
 * Every probe is a flat array of `double` plus the shape that describes it, so the
 * artifact reader never guesses geometry order (03-parity-harness.md §3.3). The set
 * of probes a run produces depends on what the backend *publishes*, not on which
 * backend it is:
 *
 *   vertices      — `builtin::position` of every scene geometry, in creation order
 *   transforms    — live affine-body state (AffineBodyStateAccessorFeature)
 *   velocities    — live affine-body velocity (same feature)
 *   fe_positions  — live finite-element positions (FiniteElementStateAccessorFeature)
 *   fe_velocities — live finite-element velocity (same feature)
 *
 * The state is read through the two `*StateAccessorFeature`s (create_geometry() +
 * copy_to()), never through `copy_transform_to(raw_pointer)`: the Luisa overrider
 * writes to a host buffer while the CUDA one reinterprets the handle as a device
 * pointer (research §3 D5), and never through the geometry's own `transform`
 * attribute, which is stale after the first frame until `dump()` (plan §8 trap 2).
 *********************************************************************/
#include <uipc/common/json.h>

#include <string>
#include <vector>

namespace uipc::core
{
class Engine;
class World;
}  // namespace uipc::core

namespace uipc::parity
{
/// One named, shaped per-frame array of `double`.
struct ArrayBlock
{
    std::string probe = {};
    /// Scene-geometry index for geometry-shaped probes, -1 otherwise.
    int geometry = -1;
    /// 3 for Vector3, 16 for Matrix4x4.
    int                 components = 1;
    std::vector<double> data       = {};
};

/// Every probe the backend publishes for the current world state, in a fixed order
/// (vertices by geometry index, then transforms, velocities, fe_positions,
/// fe_velocities). The order is what the per-frame checksum is taken over, so it is
/// part of the artifact contract.
std::vector<ArrayBlock> collect_probes(core::World& world);

/// The scalars probe: `Engine::frame_stats()` of the frame that just finished.
Json frame_scalars(const core::Engine& engine);

/// True when every element of every block is finite. The runner turns a false into
/// exit code 5 instead of writing an artifact full of NaN.
bool all_finite(const std::vector<ArrayBlock>& blocks) noexcept;
}  // namespace uipc::parity
