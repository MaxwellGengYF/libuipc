#pragma once
/********************************************************************
 * @file finite_element_external_vertex_force.h
 * @brief T5.4b — the per-vertex external-force reporter (port of
 *        src/backends/cuda/finite_element/finite_element_external_vertex_force.{h,cu}).
 *
 * A `FiniteElementExternalForceReporter` (uid 671) that requires
 * `FiniteElementExternalVertexForceConstraint` and scatter-**adds** the
 * constraint's `forces()` into `FiniteElementMethod::vertex_external_forces` at
 * `constraint->vertex_ids()`.
 *
 * ---------------------------------------------------------------------------
 * THE VECTOR-VALUED ATOMIC ADD (the one genuinely hard part of T5.4b)
 * ---------------------------------------------------------------------------
 * The reference's kernel is
 *
 *     eigen::atomic_add(forces(vid), vertex_forces(i));
 *
 * an atomic add of a `Vector3` into an **AoS** `BufferView<Vector3>`.  LC derives
 * an `AtomicRef<LVector3>` for the `LUISA_STRUCT` element type, but it exposes the
 * struct's `std::array<double,3>` member as an `AtomicRef<std::array<double,3>>`,
 * which has no `fetch_add` (cookbook §7.18 rule 10) — so the reference's statement
 * has no direct DSL spelling.
 *
 * The port's answer is the *scalar alias* of the same bytes: LC's
 * `BufferView<T>::as<U>()` reinterprets a buffer view as another element type
 * (handle + byte offset preserved, element stride `sizeof(U)`), and the LVector3
 * buffer's element stride is exactly `sizeof(LVector3)` (24 B = 3 x 8), so
 * `forces.native().as<double>()` is the flat `double[3 * n]` view of the AoS
 * buffer and the reference's add becomes the three per-component atomic adds
 *
 *     forces_flat.atomic(3 * vid + c).fetch_add(f(c)),  c = 0, 1, 2
 *
 * which is what `finite_element_external_vertex_force_detail::scatter_add_vec3`
 * does.  This keeps the reference's *atomicity* (an unbounded number of writers may
 * target the same vertex — the duplicate-vertex case below) instead of falling back
 * to a read-modify-write, which would need a single-writer proof.  `do_step`
 * asserts the stride contract before dispatching; the unit test measures it.
 *
 * The port keeps the reference's `UID = 671` (a *different* registry from the
 * constraint's 671 and from the frontend constitution's 671 — cookbook §7.20).
 *********************************************************************/
#include <finite_element/finite_element_external_force_reporter.h>
#include <finite_element/constraints/finite_element_external_vertex_force_constraint.h>
#include <finite_element/finite_element_method.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace finite_element_external_vertex_force_detail
{
/**
 * @brief `eigen::atomic_add(forces(vid), vertex_forces(i))` over the flat scalar
 *        alias of the AoS force buffer.
 *
 * One generic implementation over the view flavour and the scalar type (the
 * cookbook §7.4 recipe): the shipped kernel passes a *device* dense-vector view over
 * the `double` alias of `vertex_external_forces`, the unit test passes a
 * `storage::memory` one over a host `double[]` (whose `atomic_add_at` is the plain
 * read-modify-write — bitwise-equal arithmetic, see the device-vs-host case).
 *
 * `f` is the already-materialised `LVec<...,3>` of the entry (`load_vec3` on the
 * device), `vid` the target vertex id.
 */
template <typename ScalarView, typename Vid, typename V>
void scatter_add_vec3(ScalarView forces_scalar, const Vid& vid, const V& f) noexcept
{
    for(int c = 0; c < 3; ++c)
    {
        (void)forces_scalar.atomic_add(vid * 3 + c, f(c));
    }
}
}  // namespace finite_element_external_vertex_force_detail

/**
 * @brief Scatter-add external forces from constraint buffers into the
 *        per-vertex external force buffer on FiniteElementMethod.
 */
class FiniteElementExternalVertexForce final : public FiniteElementExternalForceReporter
{
  public:
    static constexpr U64 UID = 671;

    using FiniteElementExternalForceReporter::FiniteElementExternalForceReporter;

    SimSystemSlot<FiniteElementMethod>                         finite_element_method;
    SimSystemSlot<FiniteElementExternalVertexForceConstraint>  constraint;

    void do_build(BuildInfo& info) override;

    U64 get_uid() const noexcept override { return UID; }

    void do_init() override {}

    void do_step(ExternalForceInfo& info) override;
};
}  // namespace uipc::backend::luisa
