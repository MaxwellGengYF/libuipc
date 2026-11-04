#pragma once
/********************************************************************
 * @file parity_scene.h
 * @brief T9.3a — the backend-agnostic scene registry of the parity harness.
 *
 * One scene is one function that populates a `core::Scene`. There is no `#ifdef`,
 * no backend name and no asset file anywhere in this file: the backend is a CLI
 * argument of the runner (03-parity-harness.md §3.1), so the same translation unit
 * produces the CUDA leg and the Luisa leg. Each entry also carries
 *
 *   * `provenance` — which apps/tests/sim_case/*.cpp (or benchmark) case the body is
 *     a re-statement of, so a scene can never be invented from scratch;
 *   * `describe()` — a canonical JSON description of every constant of the scene
 *     (materials, geometry counts, transforms, config overrides). The runner hashes
 *     it into the manifest's `scene_hash`, which is what makes a registry drift
 *     (plan §8 trap "CUDA and Luisa scenes silently differ") a visible artifact
 *     difference instead of a silent physics difference.
 *********************************************************************/
#include <uipc/common/json.h>
#include <uipc/common/span.h>
#include <string_view>

namespace uipc::core
{
class Scene;
}

namespace uipc::parity
{
struct SceneEntry
{
    /// The `--scene=` value.
    std::string_view name;
    /// Where the body comes from (file:line of the case it re-states).
    std::string_view provenance;
    /// Populates the scene. Must not touch the backend or the filesystem.
    void (*build)(core::Scene& scene) = nullptr;
    /// Canonical description of every constant the body uses (see file header).
    Json (*describe)() = nullptr;
};

/// The registry, in name order.
span<const SceneEntry> scenes() noexcept;

/// nullptr when the name is unknown.
const SceneEntry* find_scene(std::string_view name) noexcept;
}  // namespace uipc::parity
