#include <affine_body/affine_body_constraint.h>
#include <affine_body/utils.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/zip.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
inline Matrix12x12 compute_constraint_mass(const ABDJacobiDyadicMass& mass,
                                           Float translation_strength,
                                           Float rotation_strength)
{
    Matrix12x12 M = mass.to_mat();
    Float cross_term_strength = std::sqrt(translation_strength * rotation_strength);
    M.block<3, 3>(0, 0) *= translation_strength;
    M.block<3, 9>(0, 3) *= cross_term_strength;
    M.block<9, 3>(3, 0) *= cross_term_strength;
    M.block<9, 9>(3, 3) *= rotation_strength;
    return M;
}

class SoftTransformConstraint final : public AffineBodyConstraint
{
    static constexpr U64 SoftTransformConstraintUID = 16ull;

  public:
    using AffineBodyConstraint::AffineBodyConstraint;

    vector<IndexT>   h_constrained_bodies;
    vector<Vector12> h_aim_transforms;
    vector<Vector2>  h_strength_ratios;

    luisa::compute::Buffer<IndexT>   constrained_bodies;
    luisa::compute::Buffer<Vector12> aim_transforms;
    luisa::compute::Buffer<Vector2>  strength_ratios;

    virtual void do_build(BuildInfo& info) override {}

    virtual U64 get_uid() const noexcept override
    {
        return SoftTransformConstraintUID;
    }

    void do_init(AffineBodyAnimator::FilteredInfo& info) override
    {
        do_step(info);  // do the same thing as do_step
    }

    void do_step(AffineBodyAnimator::FilteredInfo& info) override
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
                current_body_offset = body_offset->view().front();

                auto is_constrained = sc.instances().find<IndexT>(builtin::is_constrained);
                auto aim_transform = sc.instances().find<Matrix4x4>(builtin::aim_transform);
                auto strength_ratio = sc.instances().find<Vector2>("strength_ratio");

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
                    h_constrained_bodies.push_back(bI);
                    Vector12 q = transform_to_q(aim_transform);
                    h_aim_transforms.push_back(q);
                    h_strength_ratios.push_back(strength_ratio);
                    UIPC_ASSERT(strength_ratio(0) >= 0.0 && strength_ratio(1) >= 0.0,
                                "Strength ratios must be non-negative, but got ({}, {})",
                                strength_ratio(0),
                                strength_ratio(1));
                }
            });

        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();

        constrained_bodies = device.create_buffer<IndexT>(h_constrained_bodies.size());
        constrained_bodies.view().copy_from(h_constrained_bodies.data());

        aim_transforms = device.create_buffer<Vector12>(h_aim_transforms.size());
        aim_transforms.view().copy_from(h_aim_transforms.data());

        strength_ratios = device.create_buffer<Vector2>(h_strength_ratios.size());
        strength_ratios.view().copy_from(h_strength_ratios.data());
    }

    void do_report_extent(AffineBodyAnimator::ReportExtentInfo& info) override
    {
        info.energy_count(h_constrained_bodies.size());
        info.gradient_count(h_constrained_bodies.size());
        if(info.gradient_only())
            return;

        info.hessian_count(h_constrained_bodies.size());
    }

    void do_compute_energy(AffineBodyAnimator::ComputeEnergyInfo& info) override
    {
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto qs_view = info.qs();
        auto q_prevs_view = info.q_prevs();
        auto body_masses_view = info.body_masses();
        auto is_fixed_view = info.is_fixed();
        auto energies_view = info.energies();

        SizeT constraint_count = constrained_bodies.size();

        Kernel1D compute_energy_kernel = [&](BufferVar<IndexT> constrained_bodies,
                                              BufferVar<Vector12> aim_transforms,
                                              BufferVar<Vector2> strength_ratios,
                                              BufferVar<Vector12> qs,
                                              BufferVar<Vector12> q_prevs,
                                              BufferVar<ABDJacobiDyadicMass> body_masses,
                                              BufferVar<IndexT> is_fixed,
                                              Float substep_ratio,
                                              BufferVar<Float> energies) {
            auto I = dispatch_id().x;
            $if(I < constraint_count) {
                IndexT i = constrained_bodies.read(I);

                $if(is_fixed.read(i) != 0) {
                    energies.write(I, 0.0f);
                }
                $else {
                    Vector12 q = qs.read(i);
                    Vector12 q_prev = q_prevs.read(i);
                    Vector12 q_aim = aim_transforms.read(I);

                    // lerp: q_aim = q_prev + substep_ratio * (q_aim - q_prev)
                    Vector12 q_target;
                    for(int k = 0; k < 12; ++k) {
                        q_target[k] = q_prev[k] + substep_ratio * (q_aim[k] - q_prev[k]);
                    }

                    Vector12 dq;
                    for(int k = 0; k < 12; ++k) {
                        dq[k] = q[k] - q_target[k];
                    }

                    Vector2 s = strength_ratios.read(I);

                    ABDJacobiDyadicMass mass = body_masses.read(i);
                    Matrix12x12 M = compute_constraint_mass(mass, s[0], s[1]);

                    // E = 0.5 * dq^T * M * dq
                    Float E = 0.0f;
                    for(int r = 0; r < 12; ++r) {
                        Float row_sum = 0.0f;
                        for(int c = 0; c < 12; ++c) {
                            row_sum += M(r, c) * dq[c];
                        }
                        E += dq[r] * row_sum;
                    }
                    E *= 0.5f;

                    energies.write(I, E);
                };
            };
        };

        auto kernel = device.compile(compute_energy_kernel);
        stream << kernel(constrained_bodies.view(),
                         aim_transforms.view(),
                         strength_ratios.view(),
                         qs_view,
                         q_prevs_view,
                         body_masses_view,
                         is_fixed_view,
                         info.substep_ratio(),
                         energies_view)
                      .dispatch(constraint_count);
    }

    void do_compute_gradient_hessian(AffineBodyAnimator::ComputeGradientHessianInfo& info) override
    {
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto qs_view = info.qs();
        auto q_prevs_view = info.q_prevs();
        auto body_masses_view = info.body_masses();
        auto is_fixed_view = info.is_fixed();
        auto gradients_view = info.gradients();
        auto gradient_indices_view = info.gradient_indices();
        auto hessians_view = info.hessians();
        auto hessian_row_indices_view = info.hessian_row_indices();
        auto hessian_col_indices_view = info.hessian_col_indices();
        bool gradient_only = info.gradient_only();

        SizeT constraint_count = constrained_bodies.size();

        Kernel1D compute_gradient_hessian_kernel = [&](BufferVar<IndexT> constrained_bodies,
                                                       BufferVar<Vector12> aim_transforms,
                                                       BufferVar<Vector2> strength_ratios,
                                                       BufferVar<Vector12> qs,
                                                       BufferVar<Vector12> q_prevs,
                                                       BufferVar<ABDJacobiDyadicMass> body_masses,
                                                       BufferVar<IndexT> is_fixed,
                                                       Float substep_ratio,
                                                       BufferVar<Vector12> gradients,
                                                       BufferVar<IndexT> gradient_indices,
                                                       BufferVar<Matrix12x12> hessians,
                                                       BufferVar<IndexT> hessian_row_indices,
                                                       BufferVar<IndexT> hessian_col_indices,
                                                       Bool grad_only) {
            auto I = dispatch_id().x;
            $if(I < constraint_count) {
                IndexT i = constrained_bodies.read(I);

                Vector12 G;
                Matrix12x12 M;

                $if(is_fixed.read(i) != 0) {
                    // Set to zero
                    for(int k = 0; k < 12; ++k) {
                        G[k] = 0.0f;
                    }
                    for(int r = 0; r < 12; ++r) {
                        for(int c = 0; c < 12; ++c) {
                            M(r, c) = 0.0f;
                        }
                    }
                }
                $else {
                    Vector12 q = qs.read(i);
                    Vector12 q_prev = q_prevs.read(i);
                    Vector12 q_aim = aim_transforms.read(I);

                    // lerp: q_target = q_prev + substep_ratio * (q_aim - q_prev)
                    Vector12 q_target;
                    for(int k = 0; k < 12; ++k) {
                        q_target[k] = q_prev[k] + substep_ratio * (q_aim[k] - q_prev[k]);
                    }

                    Vector12 dq;
                    for(int k = 0; k < 12; ++k) {
                        dq[k] = q[k] - q_target[k];
                    }

                    Vector2 s = strength_ratios.read(I);

                    ABDJacobiDyadicMass mass = body_masses.read(i);
                    M = compute_constraint_mass(mass, s[0], s[1]);

                    // G = M * dq
                    for(int r = 0; r < 12; ++r) {
                        Float row_sum = 0.0f;
                        for(int c = 0; c < 12; ++c) {
                            row_sum += M(r, c) * dq[c];
                        }
                        G[r] = row_sum;
                    }
                };

                // Write gradient
                gradient_indices.write(I, i);
                gradients.write(I, G);

                $if(!grad_only) {
                    // Write hessian (diagonal block)
                    hessian_row_indices.write(I, i);
                    hessian_col_indices.write(I, i);
                    hessians.write(I, M);
                };
            };
        };

        auto kernel = device.compile(compute_gradient_hessian_kernel);
        stream << kernel(constrained_bodies.view(),
                         aim_transforms.view(),
                         strength_ratios.view(),
                         qs_view,
                         q_prevs_view,
                         body_masses_view,
                         is_fixed_view,
                         info.substep_ratio(),
                         gradients_view,
                         gradient_indices_view,
                         hessians_view,
                         hessian_row_indices_view,
                         hessian_col_indices_view,
                         gradient_only)
                      .dispatch(constraint_count);
    }
};

REGISTER_SIM_SYSTEM(SoftTransformConstraint);
}  // namespace uipc::backend::luisa
