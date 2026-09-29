/********************************************************************
 * @file affine_body/constitutions/affine_body_prismatic_joint_external_force.cpp
 * @brief T6.3b2 — the prismatic joint's external-force constraint + reporter: one
 *        kernel, the out-of-line members and the two registrations (reference:
 *        src/backends/cuda/affine_body/constitutions/affine_body_prismatic_joint.cu,
 *        classes `AffineBodyPrismaticJointExternalForceConstraint` and
 *        `AffineBodyPrismaticJointExternalForce`).
 *
 * The classes are declared in `..._external_force.h` (the file-split deviation,
 * reported there).  The 667 "collision" between the constraint uid and the reporter
 * uid is the reference's own and is documented in that header.
 *
 * THE ATOMIC SCATTER (the T5.4b `finite_element_external_vertex_force` idiom): the
 * reference's
 *
 *     eigen::atomic_add(external_forces(bids(0)), F_i);
 *
 * is a CUDA atomic add of a 12-vector into an AoS `Vector12` element.  The port has
 * no atomic on an `LVector12` element, so the kernel takes the *scalar alias* of the
 * same buffer — `LBufferView<LVector12>::native().as<double>()`, which keeps the
 * handle and byte offset and uses element stride `sizeof(double)` — and performs the
 * twelve per-component `fetch_add`s the CUDA codegen of the reference's call would
 * emit.  `do_step` asserts the stride contract (`stride == sizeof(LVector12)`) before
 * dispatching, so the alias is exact; the unit test measures it.
 ********************************************************************/
#include <affine_body/constitutions/affine_body_prismatic_joint_external_force.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace
{
using DenseDeviceView = lla::LDenseVectorDeviceView<lcrt::Var<double>>;

/// `eigen::atomic_add(external_forces(body), F12)`: twelve atomic adds on the scalar
/// alias (see the file comment).
template <typename ScalarView, typename Body, typename V>
inline void scatter_add_vec12(ScalarView   external_forces_scalar,
                              const Body&  body,
                              const V&     F) noexcept
{
    for(int c = 0; c < 12; ++c)
    {
        (void)external_forces_scalar.atomic_add(body * 12 + c, F(c));
    }
}

/// `affine_body_prismatic_joint_external_force_step_kernel`.
///
/// The first argument is the scalar alias of the dynamics' per-body
/// `external_force` window (`Buffer<double>` in the shader prototype, dispatched with
/// `.native().as<double>()`).
auto& prismatic_joint_external_force_step_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,   // external_forces (scalar alias)
                                                lcrt::Buffer<LVector2i>,  // body_ids
                                                lcrt::Buffer<double>,     // forces
                                                lcrt::Buffer<LVector6>,   // rest_tangents
                                                lcrt::Buffer<int32_t>,    // constrained_flags
                                                lcrt::Buffer<LVector12>,  // qs
                                                uint32_t,  // n (the constrained-pair count)
                                                uint32_t>(  // body_count (the external-forces window size)
        [](lcrt::BufferVar<double>    external_forces_flat,
           lcrt::BufferVar<LVector2i> body_ids,
           lcrt::BufferVar<double>    forces,
           lcrt::BufferVar<LVector6>  rest_tangents,
           lcrt::BufferVar<int32_t>   constrained_flags,
           lcrt::BufferVar<LVector12> qs,
           lcrt::Var<uint32_t>        n,
           lcrt::Var<uint32_t>        body_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // reference: `if(constrained_flags(i) == 0) return;`
                $if(constrained_flags->read(i) != 0)
                {
                    auto bids = load_vec2i(body_ids->read(i));
                    auto f    = forces->read(i);

                    auto t_bar = load_vec6(rest_tangents->read(i));

                    auto q_i = load_vec12(qs->read(bids(0)));
                    auto q_j = load_vec12(qs->read(bids(1)));

                    // ABDJacobi JT[2] = {ABDJacobi{t_bar.segment<3>(0)},
                    //                    ABDJacobi{t_bar.segment<3>(3)}};
                    const ABDJacobiT<lcrt::Var<double>> JT[2] = {
                        ABDJacobiT<lcrt::Var<double>>{.x_bar = t_bar.template segment<3>(0)},
                        ABDJacobiT<lcrt::Var<double>>{.x_bar = t_bar.template segment<3>(3)}};

                    lps::LVec<lcrt::Var<double>, 3> t_i;
                    lps::LVec<lcrt::Var<double>, 3> t_j;

                    // Vector3 t_i = JT[0].vec_x(q_i); Vector3 t_j = JT[1].vec_x(q_j);
                    // t_j = 0.5 * (t_i + t_j); t_i = -t_j;
                    abd_prismatic_joint_external_force_detail::symmetrized_axes<
                        lcrt::Var<double>>(t_i, t_j, JT[0], JT[1], q_i, q_j);

                    lps::LVec<lcrt::Var<double>, 12> F_i;
                    lps::LVec<lcrt::Var<double>, 12> F_j;
                    abd_prismatic_joint_external_force_detail::force_pair<
                        lcrt::Var<double>>(f, t_i, t_j, F_i, F_j);

                    // NOTE (deviation from the FE sibling, which could use its own
                    // `n`): the external-force window is indexed by *body*, while `n`
                    // counts the driving pairs, so the dense view's extent is the
                    // dynamics' `body_count`, passed in explicitly.
                    auto external_forces = DenseDeviceView::from_handles(
                        external_forces_flat, body_count * 12u);

                    scatter_add_vec12(external_forces, bids(0), F_i);
                    scatter_add_vec12(external_forces, bids(1), F_j);
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyPrismaticJointExternalForceConstraint);
REGISTER_SIM_SYSTEM(AffineBodyPrismaticJointExternalForce);

// ---------------------------------------------------------------------------
//  AffineBodyPrismaticJointExternalForceConstraint
// ---------------------------------------------------------------------------
void AffineBodyPrismaticJointExternalForceConstraint::do_build(BuildInfo& info)
{
    (void)info;
    prismatic_joint = require<AffineBodyPrismaticJoint>();
}

U64 AffineBodyPrismaticJointExternalForceConstraint::get_uid() const noexcept
{
    return ConstraintUID;
}

void AffineBodyPrismaticJointExternalForceConstraint::do_init(
    InterAffineBodyAnimator::FilteredInfo& info)
{
    auto geo_slots = world().scene().geometries();

    info.for_each(
        geo_slots,
        [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
        {
            (void)I;
            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc,
                        "AffineBodyPrismaticJointExternalForceConstraint: geometry must be SimplicialComplex");

            auto is_constrained = sc->edges().find<IndexT>("external_force/is_constrained");
            UIPC_ASSERT(is_constrained,
                        "AffineBodyPrismaticJointExternalForceConstraint: Geometry must have 'external_force/is_constrained' attribute on `edges`");
            auto is_constrained_view = is_constrained->view();

            auto external_force = sc->edges().find<Float>("external_force");
            UIPC_ASSERT(external_force,
                        "AffineBodyPrismaticJointExternalForceConstraint: Geometry must have 'external_force' attribute on `edges`");
            auto external_force_view = external_force->view();

            for(SizeT i = 0; i < sc->edges().size(); ++i)
            {
                h_is_constrained.push_back(is_constrained_view[i]);
                h_forces.push_back(external_force_view[i]);
            }
        });

    SizeT N = h_forces.size();
    if(N > 0)
    {
        forces.resize(h_forces.size());
        forces.view().copy_from(span<const Float>{h_forces});
        is_constrained.resize(h_is_constrained.size());
        is_constrained.view().copy_from(span<const IndexT>{h_is_constrained});
    }
}

void AffineBodyPrismaticJointExternalForceConstraint::do_step(
    InterAffineBodyAnimator::FilteredInfo& info)
{
    auto geo_slots = world().scene().geometries();

    SizeT offset = 0;
    info.for_each(
        geo_slots,
        [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
        {
            (void)I;
            auto sc = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc,
                        "AffineBodyPrismaticJointExternalForceConstraint: geometry must be SimplicialComplex");

            auto is_constrained = sc->edges().find<IndexT>("external_force/is_constrained");
            UIPC_ASSERT(is_constrained,
                        "AffineBodyPrismaticJointExternalForceConstraint: Geometry must have 'external_force/is_constrained' attribute on `edges`");
            auto is_constrained_view = is_constrained->view();

            auto external_force = sc->edges().find<Float>("external_force");
            UIPC_ASSERT(external_force,
                        "AffineBodyPrismaticJointExternalForceConstraint: Geometry must have 'external_force' attribute on `edges`");
            auto external_force_view = external_force->view();

            auto Es = sc->edges().topo().view();
            for(auto&& [i, e] : enumerate(Es))
            {
                (void)e;
                h_is_constrained[offset] = is_constrained_view[i];
                h_forces[offset]         = external_force_view[i];
                ++offset;
            }
        });

    SizeT N = h_forces.size();
    if(N > 0)
    {
        // (the ported `copy_from` of an empty span is not the reference's no-op
        // cudaMemcpy, so the empty case is guarded -- see the T6.3b2 notes)
        is_constrained.view().copy_from(span<const IndexT>{h_is_constrained});
        forces.view().copy_from(span<const Float>{h_forces});
    }
}

void AffineBodyPrismaticJointExternalForceConstraint::do_report_extent(
    InterAffineBodyAnimator::ReportExtentInfo& info)
{
    info.energy_count(0);
    info.gradient_count(0);
    if(!info.gradient_only())
        info.hessian_count(0);
}

void AffineBodyPrismaticJointExternalForceConstraint::do_compute_energy(
    InterAffineBodyAnimator::ComputeEnergyInfo& info)
{
    (void)info;
}

void AffineBodyPrismaticJointExternalForceConstraint::do_compute_gradient_hessian(
    InterAffineBodyAnimator::GradientHessianInfo& info)
{
    (void)info;
}

// ---------------------------------------------------------------------------
//  AffineBodyPrismaticJointExternalForce (the reporter)
// ---------------------------------------------------------------------------
void AffineBodyPrismaticJointExternalForce::do_build(BuildInfo& info)
{
    (void)info;
    constraint = require<AffineBodyPrismaticJointExternalForceConstraint>();
}

U64 AffineBodyPrismaticJointExternalForce::get_uid() const noexcept
{
    return ReporterUID;
}

void AffineBodyPrismaticJointExternalForce::do_init() {}

void AffineBodyPrismaticJointExternalForce::do_step(ExternalForceInfo& info)
{
    SizeT force_count = constraint->forces.size();
    if(force_count == 0)
        return;

    auto abd = constraint->prismatic_joint->affine_body_dynamics;

    int n = (int)force_count;

    auto external_forces = info.external_forces();

    // the scalar alias contract (see the file comment): the flat index of component c
    // of body b is `12 * b + c`, exact iff the element stride is sizeof(LVector12).
    auto native = external_forces.native();
    UIPC_ASSERT(native.stride() == sizeof(LVector12),
                "AffineBodyPrismaticJointExternalForce: the scalar alias of the external-force "
                "buffer needs element_stride == sizeof(LVector12) == {} B, measured {} B",
                sizeof(LVector12),
                native.stride());

    auto& k = prismatic_joint_external_force_step_kernel();
    frame_stream().record(k(native.as<double>(),
                            constraint->prismatic_joint->body_ids.cview(),
                            constraint->forces.cview(),
                            constraint->prismatic_joint->rest_ts.cview(),
                            constraint->is_constrained.cview(),
                            abd->qs(),
                            static_cast<uint32_t>(n),
                            static_cast<uint32_t>(external_forces.size()))
                            .dispatch(n));
}
}  // namespace uipc::backend::luisa
