#pragma once
/********************************************************************
 * @file distance_system/distance_diagnoser.h
 * @brief S9' (T8.5-2) — the distance diagnoser (port of
 * src/backends/cuda/distance_system/distance_diagnoser.h, 72 lines).
 *
 * A pure query service, not part of the solve chain: `DistanceDiagnoser` inserts
 * one `core::DistanceDiagnoserFeature` at build time (`do_build`) and answers the
 * four distance queries the frontend exposes through
 * `scene.features().find<core::DistanceDiagnoserFeature>()`.  It registers no
 * action and `require<>`s no other system, exactly like the reference (which
 * overrides only `do_build`, .cu:61-66), so it cannot perturb a simulation.
 *
 * HOST EVALUATION — the one design decision of this port.  The four queries are
 * evaluated as **host code** over host spans with `T = double`: no LC kernel, no
 * stream, no device buffers.  Rationale:
 *  * every pair is computed independently from host-visible inputs (no atomics,
 *    no reduction, no order dependence), so the place of evaluation cannot change
 *    a value — Class A (bit-exactness) is retained;
 *  * the ported math layer is host-evaluable by design
 *    (`luisa_device/math/vec.h`: `T = double` -> host evaluation), and every
 *    primitive the diagnoser needs is already instantiated on the host by green
 *    tests (`apps/tests/backends/luisa/luisa_device_distance_tests.cpp`);
 *  * the ported sanity checkers already evaluate the reference's arithmetic as
 *    host code over host spans (`sanity_check/simplicial_surface_distance_check.cpp`);
 *  * a kernel path would need `Vector12`/`Matrix12x12` buffer element types plus
 *    a cold NVRTC build of two 12x12 chains (~11-12 min, cookbook §7.36 rule 3)
 *    for a diagnostic API whose CUDA counterpart is a straight-line per-pair
 *    evaluation anyway.
 *
 * The reference's device->host traffic disappears with that: what the CUDA kernel
 * writes into `DeviceBuffer<T>` and `detail::copy_back` copies into
 * `R.instances()`, the port writes into a host `vector<T>` of the same dimension
 * and copies back through the same `copy_back` call list, so the produced
 * attribute set (names, types, write order, per-entry defaults) is the
 * reference's own — see the .cpp's `copy_back` note for the one correction the
 * audit turned up (there are no config keys: the six `dist2/*`, `e_k/*`,
 * `barrier/*` names are *attribute* names, always written).
 *********************************************************************/
#include <sim_system.h>
#include <uipc/core/distance_diagnoser_feature.h>

namespace uipc::backend::luisa
{
class DistanceDiagnoser;

class DistanceDiagnoserFeatureOverrider final : public core::DistanceDiagnoserFeatureOverrider
{
  public:
    DistanceDiagnoserFeatureOverrider(DistanceDiagnoser* diagnoser);

  private:
    void do_compute_point_triangle_distance(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& points,
        const geometry::SimplicialComplex& triangles) override;

    void do_compute_edge_edge_distance(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& edges_a,
        const geometry::SimplicialComplex& edges_b,
        const geometry::SimplicialComplex& rest_edges_a,
        const geometry::SimplicialComplex& rest_edges_b) override;

    void do_compute_point_edge_distance(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& points,
        const geometry::SimplicialComplex& edges) override;

    void do_compute_point_point_distance(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& points_a,
        const geometry::SimplicialComplex& points_b) override;

    SimSystemSlot<DistanceDiagnoser> m_diagnoser;
};

class DistanceDiagnoser final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    void compute_point_triangle_distance(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& points,
        const geometry::SimplicialComplex& triangles);

    void compute_edge_edge_distance(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& edges_a,
        const geometry::SimplicialComplex& edges_b,
        const geometry::SimplicialComplex& rest_edges_a,
        const geometry::SimplicialComplex& rest_edges_b);

    void compute_point_edge_distance(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& points,
        const geometry::SimplicialComplex& edges);

    void compute_point_point_distance(
        geometry::Geometry&                R,
        const geometry::SimplicialComplex& points_a,
        const geometry::SimplicialComplex& points_b);

  private:
    friend class DistanceDiagnoserFeatureOverrider;

    void do_build() override;
};
}  // namespace uipc::backend::luisa
