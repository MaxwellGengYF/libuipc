/********************************************************************
 * @file affine_body/constraints/soft_transform_constraint.cpp
 * @brief T6.4b — the ABD soft transform constraint: its three kernels, the
 *        out-of-line members and the system registration (reference:
 *        src/backends/cuda/affine_body/constraints/soft_transform_constraint.cu).
 *
 * The class (and the reference's `compute_constraint_mass` law) is declared in
 * `soft_transform_constraint.h` (the file-split deviation, reported there).
 *
 * THE THREE DEVICE PREDICATES, and why they are selects:
 *  * `if(is_fixed(i)) { E = 0.0; } else { ... }` — the energy;
 *  * `if(is_fixed(i)) { G.setZero(); } else { G = M * dq; }` — the gradient;
 *  * `if(is_fixed(i)) { Z.setZero(); } else { ... }` — the Hessian.
 * `is_fixed` is device data (the dynamics' per-body flag), so each becomes a
 * `scalar_detail::select_scalar` / `select_container` over the already-computed
 * candidate (RULE M2).  This is the *guarded* spelling — the same one the FEM sibling
 * uses, unlike the kinetic systems'.
 *
 * `lerp(q_prev, aim, substep_ratio)` is the reference's `cuda/animator/utils.h`
 * function, i.e. the **clamped** one — `animator::lerp` in this port (cookbook §7.14
 * rule 4: `lps::lerp` is unclamped and found by ADL for the same arguments).
 *
 * THE ONE SPELLING DEVIATION (documented, not a behaviour change): the reference's
 * energy reads `E = 0.5 * dq.transpose() * M * dq`, which Eigen evaluates as
 * `((dq^T M) dq)`; the port writes `0.5 * lps::dot(dq, M * dq)`, i.e. `dq^T (M dq)`.
 * The two are the same bilinear form (M is symmetric — `M_cm` and the dyadic mass are)
 * and differ only in the *summation order*, which is the same deviation the T6.1a BDF
 * kinetics make; the layer has no `row-vector * matrix` product, and `lps::dot` is the
 * port's single spelling of `x^T M x` (see the BDF1 kinetic's own note).
 ********************************************************************/
#include <affine_body/constraints/soft_transform_constraint.h>
#include <affine_body/utils.h>  // transform_to_q (the reference's `affine_body/utils.h`)
#include <animator/utils.h>     // animator::lerp (the reference's clamped `lerp`)
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/zip.h>
#include <kernel_cout.h>  // LC_LOG
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace
{
namespace Detail = soft_transform_constraint_detail;

using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 12>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 12, 12>;

/// The shared head of the three kernels: the constraint-local entry `I`, its global
/// body id `i`, the substep aim state `dq` and the constraint mass `M`.
struct EntryState
{
    lcrt::Var<IndexT>               i;
    lps::LVec<lcrt::Var<double>, 12> dq;
    lps::LMat<lcrt::Var<double>, 12, 12> M;
    lcrt::Var<int32_t>              is_fixed;
};

/// The reference's `Vector12 q = qs(i); q_prev = q_prevs(i); q_aim = lerp(...); dq =
/// q - q_aim; s = strength_ratios(I); M = compute_constraint_mass(...)` plus the
/// fixed flag — one loader for the three kernels.
[[nodiscard]] inline EntryState load_entry(
    const lcrt::Var<uint32_t>&             I,
    const lcrt::BufferVar<int32_t>&        indices,
    const lcrt::BufferVar<LVector12>&      qs,
    const lcrt::BufferVar<LVector12>&      q_prevs,
    const lcrt::BufferVar<LVector12>&      aim_transforms,
    const lcrt::BufferVar<LVector2>&       strength_ratios,
    const lcrt::BufferVar<ABDJacobiDyadicMass>& body_masses,
    const lcrt::BufferVar<int32_t>&        is_fixed,
    const lcrt::Var<double>&               substep_ratio) noexcept
{
    EntryState e;
    e.i        = indices.read(I);
    e.is_fixed = is_fixed.read(e.i);

    const auto q      = load_vec12(qs.read(e.i));
    const auto q_prev = load_vec12(q_prevs.read(e.i));
    const auto q_aim  = animator::lerp(q_prev, load_vec12(aim_transforms.read(I)),
                                      substep_ratio);
    const auto s      = load_vec2(strength_ratios.read(I));

    e.dq = q - q_aim;
    e.M  = Detail::compute_constraint_mass(load_dyadic_mass(body_masses.read(e.i)), s(0), s(1));
    return e;
}
}  // namespace

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `soft_transform_constraint_compute_energy_kernel`
auto& soft_transform_constraint_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<double,  // substep_ratio
                                                lcrt::Buffer<int32_t>,  // indices
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector12>,  // aim_transforms
                                                lcrt::Buffer<LVector2>,   // strength_ratios
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<double>,               // energies
                                                lcrt::Buffer<int32_t>,              // is_fixed
                                                uint32_t>(                          // n
        [](lcrt::Var<double>           substep_ratio,
           lcrt::BufferVar<int32_t>   indices,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::BufferVar<LVector12> aim_transforms,
           lcrt::BufferVar<LVector2>  strength_ratios,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
           lcrt::BufferVar<double>    energies,
           lcrt::BufferVar<int32_t>   is_fixed,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                const auto e = load_entry(I, indices, qs, q_prevs, aim_transforms,
                                          strength_ratios, body_masses, is_fixed,
                                          substep_ratio);
                // reference: `E = 0.5 * dq.transpose() * M * dq;` with `is_fixed` -> 0
                energies.write(I,
                               scalar_detail::select_scalar(0.5 * lps::dot(e.dq, e.M * e.dq),
                                                            lcrt::Var<double>{0.0},
                                                            e.is_fixed != 0));
            };
        });
    return k;
}

/// `soft_transform_constraint_compute_gradient_hessian_k1_kernel`
auto& soft_transform_constraint_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<double,  // substep_ratio
                                                lcrt::Buffer<int32_t>,  // indices
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector12>,  // aim_transforms
                                                lcrt::Buffer<LVector2>,   // strength_ratios
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<int32_t>,  // G indices
                                                lcrt::Buffer<double>,   // G values
                                                uint32_t,               // G total segments
                                                int32_t,  // G doublet index offset
                                                int32_t,  // G doublet count
                                                int32_t,  // G total doublet count
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t>(              // n
        [](lcrt::Var<double>           substep_ratio,
           lcrt::BufferVar<int32_t>   indices,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::BufferVar<LVector12> aim_transforms,
           lcrt::BufferVar<LVector2>  strength_ratios,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
           lcrt::BufferVar<int32_t>   G_indices,
           lcrt::BufferVar<double>    G_values,
           lcrt::Var<uint32_t>        G_total_segments,
           lcrt::Var<int32_t>         G_doublet_index_offset,
           lcrt::Var<int32_t>         G_doublet_count,
           lcrt::Var<int32_t>         G_total_doublet_count,
           lcrt::BufferVar<int32_t>   is_fixed,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                const auto e = load_entry(I, indices, qs, q_prevs, aim_transforms,
                                          strength_ratios, body_masses, is_fixed,
                                          substep_ratio);
                const auto zero = lps::make_vec<lcrt::Var<double>, 12>(
                    [](int) noexcept { return lcrt::Var<double>{0.0}; });
                const auto G = scalar_detail::select_container(e.M * e.dq, zero,
                                                               e.is_fixed != 0);

                // reference: `gradients(I).write(i, G);` — the *segment* index is the
                // global body id and the doublet's own index is `I` (the view's
                // `doublet_index_offset` was carried in from the animator's slice).
                auto D = DoubletDeviceView::from_handles_full(G_indices,
                                                              G_values,
                                                              G_total_segments,
                                                              G_doublet_index_offset,
                                                              G_doublet_count,
                                                              G_total_doublet_count,
                                                              std::int32_t{0},
                                                              G_total_segments);
                D(I).write(e.i, G);
            };
        });
    return k;
}

/// `soft_transform_constraint_compute_gradient_hessian_k2_kernel`
auto& soft_transform_constraint_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // indices
                                                lcrt::Buffer<LVector2>,  // strength_ratios
                                                lcrt::Buffer<ABDJacobiDyadicMass>,  // body_masses
                                                lcrt::Buffer<int32_t>,  // H row indices
                                                lcrt::Buffer<int32_t>,  // H col indices
                                                lcrt::Buffer<double>,   // H values
                                                int32_t,                // H total block rows
                                                int32_t,                // H total block cols
                                                int32_t,  // H triplet index offset
                                                int32_t,  // H triplet count
                                                int32_t,  // H total triplet count
                                                int32_t,  // H submatrix offset x
                                                int32_t,  // H submatrix offset y
                                                int32_t,  // H submatrix extent x
                                                int32_t,  // H submatrix extent y
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t>(              // n
        [](lcrt::BufferVar<int32_t>   indices,
           lcrt::BufferVar<LVector2>  strength_ratios,
           lcrt::BufferVar<ABDJacobiDyadicMass> body_masses,
           lcrt::BufferVar<int32_t>   H_row_indices,
           lcrt::BufferVar<int32_t>   H_col_indices,
           lcrt::BufferVar<double>    H_values,
           lcrt::Var<int32_t>         H_total_rows,
           lcrt::Var<int32_t>         H_total_cols,
           lcrt::Var<int32_t>         H_triplet_index_offset,
           lcrt::Var<int32_t>         H_triplet_count,
           lcrt::Var<int32_t>         H_total_triplet_count,
           lcrt::Var<int32_t>         H_sm_offset_x,
           lcrt::Var<int32_t>         H_sm_offset_y,
           lcrt::Var<int32_t>         H_sm_extent_x,
           lcrt::Var<int32_t>         H_sm_extent_y,
           lcrt::BufferVar<int32_t>   is_fixed,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                const auto i = indices.read(I);
                const auto s = load_vec2(strength_ratios.read(I));

                lps::LMat<lcrt::Var<double>, 12, 12> M;
                const auto zero = lps::make_mat<lcrt::Var<double>, 12, 12>(
                    [](int, int) noexcept { return lcrt::Var<double>{0.0}; });
                // reference: `if(is_fixed(i)) { Z.setZero(); } else { M =
                // compute_constraint_mass(...); }`
                {
                    const auto general = Detail::compute_constraint_mass(load_dyadic_mass(body_masses.read(i)), s(0), s(1));
                    M = scalar_detail::select_container(general, zero,
                                                        is_fixed.read(i) != 0);
                }

                auto T = TripletDeviceView::from_handles_full(H_row_indices,
                                                              H_col_indices,
                                                              H_values,
                                                              H_total_rows,
                                                              H_total_cols,
                                                              H_triplet_index_offset,
                                                              H_triplet_count,
                                                              H_total_triplet_count,
                                                              H_sm_offset_x,
                                                              H_sm_offset_y,
                                                              H_sm_extent_x,
                                                              H_sm_extent_y);
                T(I).write(i, i, M);
            };
        });
    return k;
}

REGISTER_SIM_SYSTEM(SoftTransformConstraint);

U64 SoftTransformConstraint::get_uid() const noexcept
{
    return SoftTransformConstraintUID;
}

void SoftTransformConstraint::do_build(BuildInfo& info)
{
    (void)info;
}

void SoftTransformConstraint::do_init(AffineBodyAnimator::FilteredInfo& info)
{
    do_step(info);  // do the same thing as do_step
}

void SoftTransformConstraint::do_step(AffineBodyAnimator::FilteredInfo& info)
{
    using ForEachInfo = AffineBodyDynamics::ForEachInfo;

    auto geo_slots = world().scene().geometries();

    // clear
    h_constrained_bodies.clear();
    h_aim_transforms.clear();
    h_strength_ratios.clear();

    IndexT current_body_offset = 0;
    info.for_each(
        geo_slots,
        [&](geometry::SimplicialComplex& sc)
        {
            auto body_offset = sc.meta().find<IndexT>(builtin::backend_abd_body_offset);
            UIPC_ASSERT(body_offset,
                        "SoftTransformConstraint: `backend_abd_body_offset` attribute not found");
            current_body_offset = body_offset->view().front();

            auto is_constrained = sc.instances().find<IndexT>(builtin::is_constrained);
            UIPC_ASSERT(is_constrained,
                        "SoftTransformConstraint: `is_constrained` attribute not found");
            auto aim_transform = sc.instances().find<Matrix4x4>(builtin::aim_transform);
            UIPC_ASSERT(aim_transform,
                        "SoftTransformConstraint: `aim_transform` attribute not found");
            auto strength_ratio = sc.instances().find<Vector2>("strength_ratio");
            UIPC_ASSERT(strength_ratio,
                        "SoftTransformConstraint: `strength_ratio` attribute not found");

            return zip(is_constrained->view(),
                       aim_transform->view(),
                       strength_ratio->view());
        },
        [&](const ForEachInfo& I, auto&& values)
        {
            SizeT bI = I.local_index() + current_body_offset;

            auto&& [is_constrained, aim_transform, strength_ratio] = values;

            if(is_constrained)
            {
                h_constrained_bodies.push_back(static_cast<IndexT>(bI));
                Vector12 q = transform_to_q(aim_transform);
                h_aim_transforms.push_back(q);
                h_strength_ratios.push_back(strength_ratio);
                UIPC_ASSERT(strength_ratio(0) >= 0.0 && strength_ratio(1) >= 0.0,
                            "Strength ratios must be non-negative, but got ({}, {})",
                            strength_ratio(0),
                            strength_ratio(1));
            }
        });

    // NOTE (port necessity, the T6.3b1 idiom): the reference's `copy_from` is a plain
    // cudaMemcpy, which is a no-op for an empty source; an LC `copy_from` of a
    // zero-size span is not, so each upload is guarded.  The values are identical.
    constrained_bodies.resize(h_constrained_bodies.size());
    if(!h_constrained_bodies.empty())
    {
        constrained_bodies.view().copy_from(span<const IndexT>{h_constrained_bodies});
    }

    // the host `Vector12`/`Vector2` (Eigen) -> device `LVector12`/`LVector2` crossings
    // are element-wise (the T6.3b2 rule-15 trap: the ABD header's `to_luisa(Vector12)`
    // is invisible to the global `copy_host_to_device`'s dependent unqualified call).
    {
        std::vector<LVector12> staging;
        staging.reserve(h_aim_transforms.size());
        for(auto& v : h_aim_transforms)
            staging.push_back(to_luisa(v));
        aim_transforms.resize(staging.size());
        if(!staging.empty())
        {
            aim_transforms.view().copy_from(span<const LVector12>{staging});
        }
    }
    {
        std::vector<LVector2> staging;
        staging.reserve(h_strength_ratios.size());
        for(auto& v : h_strength_ratios)
            staging.push_back(to_luisa(v));
        strength_ratios.resize(staging.size());
        if(!staging.empty())
        {
            strength_ratios.view().copy_from(span<const LVector2>{staging});
        }
    }
}

void SoftTransformConstraint::do_report_extent(AffineBodyAnimator::ReportExtentInfo& info)
{
    info.energy_count(h_constrained_bodies.size());
    info.gradient_count(h_constrained_bodies.size());
    if(info.gradient_only())
        return;

    info.hessian_count(h_constrained_bodies.size());
}

void SoftTransformConstraint::do_compute_energy(AffineBodyAnimator::ComputeEnergyInfo& info)
{
    int n = static_cast<int>(constrained_bodies.size());
    if(n > 0)
    {
        auto& k = soft_transform_constraint_energy_kernel();
        frame_stream().record(k(info.substep_ratio(),
                                constrained_bodies.cview(),
                                info.qs(),
                                info.q_prevs(),
                                aim_transforms.cview(),
                                strength_ratios.cview(),
                                info.body_masses(),
                                info.energies(),
                                info.is_fixed(),
                                static_cast<uint32_t>(n))
                              .dispatch(n));
    }
}

void SoftTransformConstraint::do_compute_gradient_hessian(
    AffineBodyAnimator::ComputeGradientHessianInfo& info)
{
    int n = static_cast<int>(constrained_bodies.size());
    if(n > 0)
    {
        auto gradients = info.gradients();

        auto& k = soft_transform_constraint_gradient_kernel();
        frame_stream().record(
            k(info.substep_ratio(),
              constrained_bodies.cview(),
              info.qs(),
              info.q_prevs(),
              aim_transforms.cview(),
              strength_ratios.cview(),
              info.body_masses(),
              gradients.indices(),
              gradients.values(),
              static_cast<uint32_t>(gradients.total_extent()),
              static_cast<std::int32_t>(gradients.doublet_index_offset()),
              static_cast<std::int32_t>(gradients.doublet_count()),
              static_cast<std::int32_t>(gradients.total_doublet_count()),
              info.is_fixed(),
              static_cast<uint32_t>(n))
                .dispatch(n));
    }

    if(info.gradient_only())
        return;

    if(n > 0)
    {
        auto hessians = info.hessians();

        auto& k = soft_transform_constraint_hessian_kernel();
        frame_stream().record(
            k(constrained_bodies.cview(),
              strength_ratios.cview(),
              info.body_masses(),
              hessians.row_indices(),
              hessians.col_indices(),
              hessians.values(),
              static_cast<std::int32_t>(hessians.total_rows()),
              static_cast<std::int32_t>(hessians.total_cols()),
              static_cast<std::int32_t>(hessians.triplet_index_offset()),
              static_cast<std::int32_t>(hessians.triplet_count()),
              static_cast<std::int32_t>(hessians.total_triplet_count()),
              static_cast<std::int32_t>(hessians.submatrix_offset().x),
              static_cast<std::int32_t>(hessians.submatrix_offset().y),
              static_cast<std::int32_t>(hessians.extent().x),
              static_cast<std::int32_t>(hessians.extent().y),
              info.is_fixed(),
              static_cast<uint32_t>(n))
                .dispatch(n));
    }
}
}  // namespace uipc::backend::luisa
