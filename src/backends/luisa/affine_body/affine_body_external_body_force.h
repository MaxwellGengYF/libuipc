#pragma once
/********************************************************************
 * @file affine_body/affine_body_external_body_force.h
 * @brief T6.4b — the ABD per-body external-force *reporter* (uid 666), the ABD
 *        sibling of the ported FEM `FiniteElementExternalVertexForce` (uid 671).
 *
 * REFERENCE: src/backends/cuda/affine_body/affine_body_external_body_force.{h,cu}
 * (the `.h` is not part of the reference tree: the reference declares this class in
 * the `.cu`, the T5.4b file-split deviation the rest of this batch also reports).
 *
 * A `AffineBodyExternalForceReporter` that requires `AffineBodyDynamics` and the
 * paired `AffineBodyExternalBodyForceConstraint` (both uid 666 -- *different*
 * registries, see the constraint's header) and scatter-**adds** the constraint's
 * `forces()` into the dynamics' per-body `body_id_to_external_force` at
 * `constraint->body_ids()`.  The external-force *manager* then turns that into
 * `a_ext = M^{-1} F` (and gravity/`external_kinetic` are handled by the dynamics
 * itself), which is how the forces reach `do_predict_dof` without any energy term.
 *
 * ---------------------------------------------------------------------------
 * THE VECTOR-VALUED ATOMIC ADD (the same genuinely hard part as T5.4b's 671)
 * ---------------------------------------------------------------------------
 * The reference's kernel is
 *
 *     eigen::atomic_add(forces(body_id), body_forces(i));
 *
 * an atomic add of a `Vector12` into an **AoS** `BufferView<Vector12>`.  LC derives
 * an `AtomicRef<LVector12>` for the `LUISA_STRUCT` element type, but it exposes the
 * struct's `std::array<double,12>` member as an `AtomicRef<std::array<double,12>>`,
 * which has no `fetch_add` (cookbook §7.18 rule 10) -- so the reference's statement
 * has no direct DSL spelling.
 *
 * The port's answer is the *scalar alias* of the same bytes: LC's
 * `BufferView<T>::as<U>()` reinterprets a buffer view as another element type (handle
 * + byte offset preserved, element stride `sizeof(U)`), and the LVector12 buffer's
 * element stride is exactly `sizeof(LVector12)` (96 B = 12 x 8), so
 * `forces.native().as<double>()` is the flat `double[12 * n]` view of the AoS buffer
 * and the reference's add becomes the twelve per-component atomic adds
 *
 *     forces_flat.atomic(12 * body_id + c).fetch_add(F(c)),  c = 0, 1, ..., 11
 *
 * which is what `affine_body_external_body_force_detail::scatter_add_vec12` does
 * (the revolute/prismatic joint external-force reporters spell the same helper; this
 * batch's copy is local so that its include closure is the reference's own).  This
 * keeps the reference's *atomicity* (an unbounded number of writers may target the
 * same body -- the duplicate-body_id case the unit test drives) instead of falling
 * back to a read-modify-write.  `do_step` asserts the stride contract before
 * dispatching; the unit test measures it.
 *********************************************************************/
#include <affine_body/affine_body_external_force_reporter.h>
#include <affine_body/constraints/affine_body_external_body_force_constraint.h>
#include <affine_body/affine_body_dynamics.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace affine_body_external_body_force_detail
{
/**
 * @brief `eigen::atomic_add(forces(body_id), body_forces(i))` over the flat scalar
 *        alias of the AoS force buffer.
 *
 * One generic implementation over the view flavour and the scalar type (the cookbook
 * §7.4 recipe): the shipped kernel passes a *device* dense-vector view over the
 * `double` alias of `body_id_to_external_force`, the unit test passes a
 * `storage::memory` one over a host `double[]` (whose `atomic_add` is the plain
 * read-modify-write -- the same arithmetic, see the device-vs-host case).
 */
template <typename ScalarView, typename Body, typename V>
inline void scatter_add_vec12(ScalarView forces_scalar, const Body& body, const V& F) noexcept
{
    for(int c = 0; c < 12; ++c)
    {
        (void)forces_scalar.atomic_add(body * 12 + c, F(c));
    }
}
}  // namespace affine_body_external_body_force_detail

/**
 * @brief Get external forces from the constraint and apply them to affine bodies.
 *
 * This reporter adds forces to the AffineBodyDynamics system's per-body external
 * force buffer.
 *
 * This is the "body force" implementation -- forces are applied directly to bodies.
 * Future implementations like AffineBodyExternalVertexForce may apply forces to
 * vertices.
 */
class AffineBodyExternalBodyForce final : public AffineBodyExternalForceReporter
{
  public:
    static constexpr U64 UID = 666;  // Same UID as the constraint and the constitution

    using AffineBodyExternalForceReporter::AffineBodyExternalForceReporter;

    SimSystemSlot<AffineBodyDynamics>                    affine_body_dynamics;
    SimSystemSlot<AffineBodyExternalBodyForceConstraint> constraint;

    void do_build(BuildInfo& info) override;

    U64 get_uid() const noexcept override { return UID; }

    void do_init() override
    {
        // Nothing to do
    }

    void do_step(ExternalForceInfo& info) override;
};
}  // namespace uipc::backend::luisa
