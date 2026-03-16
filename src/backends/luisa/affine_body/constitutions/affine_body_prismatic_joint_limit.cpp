#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constraints/external_articulation_constraint_function.h>
#include <affine_body/constitutions/joint_limit_penalty.h>
#include <affine_body/utils.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class AffineBodyPrismaticJointLimit final : public InterAffineBodyConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 669;
    static constexpr U64   JointUID        = 20;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    using Vector24    = Vector<Float, 24>;
    using Matrix24x24 = Matrix<Float, 24, 24>;

    vector<Vector2i> h_body_ids;
    vector<Vector6>  h_l_basis;
    vector<Vector6>  h_r_basis;
    vector<Vector24> h_ref_qs;
    vector<Float>    h_lowers;
    vector<Float>    h_uppers;
    vector<Float>    h_strengths;

    luisa::compute::Buffer<Vector2i> body_ids;
    luisa::compute::Buffer<Vector6>  l_basis;
    luisa::compute::Buffer<Vector6>  r_basis;
    luisa::compute::Buffer<Vector24> ref_qs;
    luisa::compute::Buffer<Float>    lowers;
    luisa::compute::Buffer<Float>    uppers;
    luisa::compute::Buffer<Float>    strengths;

    static auto get_prismatic_basis(const geometry::SimplicialComplex* L,
                                    IndexT                             L_inst_id,
                                    const geometry::SimplicialComplex* R,
                                    IndexT                             R_inst_id,
                                    const geometry::SimplicialComplex* joint_mesh,
                                    IndexT                             joint_index)
    {
        auto topo_view = joint_mesh->edges().topo().view();
        auto pos_view  = joint_mesh->positions().view();

        Vector2i e = topo_view[joint_index];
        Vector3  t = pos_view[e[1]] - pos_view[e[0]];
        UIPC_ASSERT(t.squaredNorm() > 0.0,
                    "AffineBodyPrismaticJointLimit: joint edge {} has zero length; cannot compute prismatic basis",
                    joint_index);
        t          = t.normalized();
        Vector3 c  = pos_view[e[0]];

        auto compute_ct_bar = [&](const geometry::SimplicialComplex* geo, IndexT inst_id) -> Vector6
        {
            UIPC_ASSERT(geo, "AffineBodyPrismaticJointLimit: geometry is null when computing basis");

            const Matrix4x4& trans = geo->transforms().view()[inst_id];
            Transform        T{trans};
            Matrix3x3        inv_rot = T.rotation().inverse();
            Vector6          ct_bar;
            ct_bar.segment<3>(0) = T.inverse() * c;
            ct_bar.segment<3>(3) = inv_rot * t;
            return ct_bar;
        };

        Vector6 L_ct_bar = compute_ct_bar(L, L_inst_id);
        Vector6 R_ct_bar = compute_ct_bar(R, R_inst_id);
        return std::tuple{L_ct_bar, R_ct_bar};
    }

    void do_build(BuildInfo& info) override {}

    void do_init(FilteredInfo& info) override
    {
        auto geo_slots = world().scene().geometries();

        h_body_ids.clear();
        h_l_basis.clear();
        h_r_basis.clear();
        h_ref_qs.clear();
        h_lowers.clear();
        h_uppers.clear();
        h_strengths.clear();

        info.for_each(
            geo_slots,
            [&](geometry::Geometry& geo)
            {
                auto uid = geo.meta().find<U64>(builtin::constitution_uid);
                UIPC_ASSERT(uid && uid->view()[0] == JointUID,
                            "AffineBodyPrismaticJointLimit must be attached on base prismatic joint geometry (UID={})",
                            JointUID);

                auto sc = geo.as<geometry::SimplicialComplex>();
                UIPC_ASSERT(sc, "AffineBodyPrismaticJointLimit geometry must be SimplicialComplex");

                auto geo_ids_attr = sc->edges().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids_attr,
                            "AffineBodyPrismaticJointLimit requires `geo_ids` attribute on edges");
                auto geo_ids = geo_ids_attr->view();

                auto inst_ids_attr = sc->edges().find<Vector2i>("inst_ids");
                UIPC_ASSERT(inst_ids_attr,
                            "AffineBodyPrismaticJointLimit requires `inst_ids` attribute on edges");
                auto inst_ids = inst_ids_attr->view();

                auto lower_attr = sc->edges().find<Float>("limit/lower");
                UIPC_ASSERT(lower_attr,
                            "AffineBodyPrismaticJointLimit requires `limit/lower` attribute on edges");
                auto lower_view = lower_attr->view();

                auto upper_attr = sc->edges().find<Float>("limit/upper");
                UIPC_ASSERT(upper_attr,
                            "AffineBodyPrismaticJointLimit requires `limit/upper` attribute on edges");
                auto upper_view = upper_attr->view();

                auto strength_attr = sc->edges().find<Float>("limit/strength");
                UIPC_ASSERT(strength_attr,
                            "AffineBodyPrismaticJointLimit requires `limit/strength` attribute on edges");
                auto strength_view = strength_attr->view();

                auto edges = sc->edges().topo().view();
                for(auto&& [i, e] : enumerate(edges))
                {
                    Vector2i geo_id  = geo_ids[i];
                    Vector2i inst_id = inst_ids[i];

                    auto* left_sc  = info.body_geo(geo_slots, geo_id[0]);
                    auto* right_sc = info.body_geo(geo_slots, geo_id[1]);

                    UIPC_ASSERT(inst_id[0] >= 0
                                    && inst_id[0] < static_cast<IndexT>(left_sc->instances().size()),
                                "AffineBodyPrismaticJointLimit: left instance ID {} out of range [0, {})",
                                inst_id[0],
                                left_sc->instances().size());
                    UIPC_ASSERT(inst_id[1] >= 0
                                    && inst_id[1] < static_cast<IndexT>(right_sc->instances().size()),
                                "AffineBodyPrismaticJointLimit: right instance ID {} out of range [0, {})",
                                inst_id[1],
                                right_sc->instances().size());

                    Vector2i bid = {
                        info.body_id(geo_id[0], inst_id[0]),
                        info.body_id(geo_id[1], inst_id[1]),
                    };

                    auto [lb, rb] = get_prismatic_basis(
                        left_sc, inst_id[0], right_sc, inst_id[1], sc, i);

                    Vector24 ref;
                    ref.segment<12>(0) =
                        transform_to_q(left_sc->transforms().view()[inst_id[0]]);
                    ref.segment<12>(12) =
                        transform_to_q(right_sc->transforms().view()[inst_id[1]]);

                    h_body_ids.push_back(bid);
                    h_l_basis.push_back(lb);
                    h_r_basis.push_back(rb);
                    h_ref_qs.push_back(ref);
                    UIPC_ASSERT(lower_view[i] <= upper_view[i],
                                "AffineBodyPrismaticJointLimit: requires `limit/lower <= limit/upper` on edge {}, but got lower={} upper={}",
                                i,
                                lower_view[i],
                                upper_view[i]);
                    h_lowers.push_back(lower_view[i]);
                    h_uppers.push_back(upper_view[i]);
                    h_strengths.push_back(strength_view[i]);
                }
            });

        // Create device buffers and copy data
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        body_ids = device.create_buffer<Vector2i>(h_body_ids.size());
        l_basis = device.create_buffer<Vector6>(h_l_basis.size());
        r_basis = device.create_buffer<Vector6>(h_r_basis.size());
        ref_qs = device.create_buffer<Vector24>(h_ref_qs.size());
        lowers = device.create_buffer<Float>(h_lowers.size());
        uppers = device.create_buffer<Float>(h_uppers.size());
        strengths = device.create_buffer<Float>(h_strengths.size());

        body_ids.view().copy_from(h_body_ids.data());
        l_basis.view().copy_from(h_l_basis.data());
        r_basis.view().copy_from(h_r_basis.data());
        ref_qs.view().copy_from(h_ref_qs.data());
        lowers.view().copy_from(h_lowers.data());
        uppers.view().copy_from(h_uppers.data());
        strengths.view().copy_from(h_strengths.data());
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(body_ids.size());
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace EPJ = sym::external_prismatic_joint_constraint;
        
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto body_ids_view = body_ids.view();
        auto l_basis_view = l_basis.view();
        auto r_basis_view = r_basis.view();
        auto ref_qs_view = ref_qs.view();
        auto lowers_view = lowers.view();
        auto uppers_view = uppers.view();
        auto strengths_view = strengths.view();
        auto qs_view = info.qs();
        auto q_prevs_view = info.q_prevs();
        auto energies_view = info.energies();

        SizeT joint_count = body_ids.size();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector2i> body_ids,
                                              BufferVar<Vector6> l_basis,
                                              BufferVar<Vector6> r_basis,
                                              BufferVar<Vector24> ref_qs,
                                              BufferVar<Float> lowers,
                                              BufferVar<Float> uppers,
                                              BufferVar<Float> strengths,
                                              BufferVar<Vector12> qs,
                                              BufferVar<Vector12> q_prevs,
                                              BufferVar<Float> Es) {
            auto I = dispatch_id().x;
            $if(I < joint_count) {
                Vector2i bid = body_ids.read(I);

                Vector6 lb = l_basis.read(I);
                Vector6 rb = r_basis.read(I);
                Vector24 ref_q = ref_qs.read(I);

                Vector12 qk      = qs.read(bid[0]);
                Vector12 ql      = qs.read(bid[1]);
                Vector12 q_prevk = q_prevs.read(bid[0]);
                Vector12 q_prevl = q_prevs.read(bid[1]);
                
                // Extract q_refk and q_refl from ref_q
                Vector12 q_refk, q_refl;
                for(int k = 0; k < 12; ++k) {
                    q_refk[k] = ref_q[k];
                    q_refl[k] = ref_q[12 + k];
                }

                Float theta_prev = 0.0f;
                EPJ::DeltaTheta<Float>(
                    theta_prev, lb, q_prevk, q_refk, rb, q_prevl, q_refl);

                Float delta = 0.0f;
                EPJ::DeltaTheta<Float>(delta, lb, qk, q_prevk, rb, ql, q_prevl);

                Float x        = theta_prev + delta;
                Float lower    = lowers.read(I);
                Float upper    = uppers.read(I);
                Float strength = strengths.read(I);

                Float E = joint_limit::eval_penalty_energy<Float>(
                    x, lower, upper, strength);

                Es.write(I, E);
            };
        };

        auto kernel = device.compile(compute_energy_kernel);
        stream << kernel(body_ids_view,
                         l_basis_view,
                         r_basis_view,
                         ref_qs_view,
                         lowers_view,
                         uppers_view,
                         strengths_view,
                         qs_view,
                         q_prevs_view,
                         energies_view)
                      .dispatch(joint_count);
    }

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override
    {
        info.gradient_count(2 * body_ids.size());
        if(info.gradient_only())
            return;

        info.hessian_count(HalfHessianSize * body_ids.size());
    }

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace EPJ = sym::external_prismatic_joint_constraint;
        
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto body_ids_view = body_ids.view();
        auto l_basis_view = l_basis.view();
        auto r_basis_view = r_basis.view();
        auto ref_qs_view = ref_qs.view();
        auto lowers_view = lowers.view();
        auto uppers_view = uppers.view();
        auto strengths_view = strengths.view();
        auto qs_view = info.qs();
        auto q_prevs_view = info.q_prevs();
        auto gradients_view = info.gradients();
        auto hessians_view = info.hessians();
        bool gradient_only = info.gradient_only();

        SizeT joint_count = body_ids.size();

        // The views from info.gradients() and info.hessians() use BufferView<const T>
        // but the constitutions need to write to them. We use a workaround by
        // creating mutable buffer views from the underlying buffers.
        
        // Get the underlying buffers by creating new views with const_cast
        // This assumes the buffers are actually mutable
        auto G_indices = luisa::compute::BufferView<luisa::uint>(
            const_cast<luisa::compute::Buffer<luisa::uint>&>(
                *reinterpret_cast<const luisa::compute::Buffer<luisa::uint>*>(
                    &gradients_view.indices)),
            gradients_view.indices.offset(),
            gradients_view.indices.size());
        auto G_values = luisa::compute::BufferView<Vector12>(
            const_cast<luisa::compute::Buffer<Vector12>&>(
                *reinterpret_cast<const luisa::compute::Buffer<Vector12>*>(
                    &gradients_view.values)),
            gradients_view.values.offset(),
            gradients_view.values.size());
        
        auto H_row_indices = luisa::compute::BufferView<luisa::uint>(
            const_cast<luisa::compute::Buffer<luisa::uint>&>(
                *reinterpret_cast<const luisa::compute::Buffer<luisa::uint>*>(
                    &hessians_view.row_indices)),
            hessians_view.row_indices.offset(),
            hessians_view.row_indices.size());
        auto H_col_indices = luisa::compute::BufferView<luisa::uint>(
            const_cast<luisa::compute::Buffer<luisa::uint>&>(
                *reinterpret_cast<const luisa::compute::Buffer<luisa::uint>*>(
                    &hessians_view.col_indices)),
            hessians_view.col_indices.offset(),
            hessians_view.col_indices.size());
        auto H_values = luisa::compute::BufferView<Matrix12x12>(
            const_cast<luisa::compute::Buffer<Matrix12x12>&>(
                *reinterpret_cast<const luisa::compute::Buffer<Matrix12x12>*>(
                    &hessians_view.values)),
            hessians_view.values.offset(),
            hessians_view.values.size());

        Kernel1D compute_gradient_hessian_kernel = [&](BufferVar<Vector2i> body_ids,
                                                       BufferVar<Vector6> l_basis,
                                                       BufferVar<Vector6> r_basis,
                                                       BufferVar<Vector24> ref_qs,
                                                       BufferVar<Float> lowers,
                                                       BufferVar<Float> uppers,
                                                       BufferVar<Float> strengths,
                                                       BufferVar<Vector12> qs,
                                                       BufferVar<Vector12> q_prevs,
                                                       BufferVar<Vector12> G12s_values,
                                                       BufferVar<luisa::uint> G12s_indices,
                                                       BufferVar<Matrix12x12> H12x12s_values,
                                                       BufferVar<luisa::uint> H12x12s_row_indices,
                                                       BufferVar<luisa::uint> H12x12s_col_indices,
                                                       Bool gradient_only) {
            auto I = dispatch_id().x;
            $if(I < joint_count) {
                Vector2i bid = body_ids.read(I);

                Vector6 lb = l_basis.read(I);
                Vector6 rb = r_basis.read(I);
                Vector24 ref_q = ref_qs.read(I);

                Vector12 qk      = qs.read(bid[0]);
                Vector12 ql      = qs.read(bid[1]);
                Vector12 q_prevk = q_prevs.read(bid[0]);
                Vector12 q_prevl = q_prevs.read(bid[1]);
                
                // Extract q_refk and q_refl from ref_q
                Vector12 q_refk, q_refl;
                for(int k = 0; k < 12; ++k) {
                    q_refk[k] = ref_q[k];
                    q_refl[k] = ref_q[12 + k];
                }

                Float theta_prev = 0.0f;
                EPJ::DeltaTheta<Float>(
                    theta_prev, lb, q_prevk, q_refk, rb, q_prevl, q_refl);

                Float delta = 0.0f;
                EPJ::DeltaTheta<Float>(delta, lb, qk, q_prevk, rb, ql, q_prevl);

                Float x        = theta_prev + delta;
                Float lower    = lowers.read(I);
                Float upper    = uppers.read(I);
                Float strength = strengths.read(I);

                Float dE_dx   = 0.0f;
                Float d2E_dx2 = 0.0f;
                joint_limit::eval_penalty_derivatives<Float>(
                    x, lower, upper, strength, dE_dx, d2E_dx2);

                Vector24 dx_dq;
                EPJ::dDeltaTheta_dQ<Float>(dx_dq, lb, qk, q_prevk, rb, ql, q_prevl);

                Vector24 G;
                for(int k = 0; k < 24; ++k) {
                    G[k] = dE_dx * dx_dq[k];
                }

                // Write gradients using DoubletVectorAssembler pattern
                // Each joint writes 2 gradients (one for each body)
                IndexT offset = 2 * I;
                for(int ii = 0; ii < 2; ++ii) {
                    IndexT dst = (ii == 0) ? bid[0] : bid[1];
                    G12s_indices.write(offset + ii, dst);
                    
                    // Write Vector12 for this body
                    Vector12 G12;
                    for(int k = 0; k < 12; ++k) {
                        G12[k] = G[ii * 12 + k];
                    }
                    G12s_values.write(offset + ii, G12);
                }

                $if(!gradient_only) {
                    // Compute Hessian: d2E_dx2 * (dx_dq * dx_dq.transpose())
                    Matrix24x24 H;
                    for(int i = 0; i < 24; ++i) {
                        for(int j = 0; j < 24; ++j) {
                            H(i, j) = d2E_dx2 * dx_dq[i] * dx_dq[j];
                        }
                    }

                    $if(dE_dx != 0.0f) {
                        Vector12 F;
                        Vector12 F_prev;
                        EPJ::F<Float>(F, lb, qk, rb, ql);
                        EPJ::F<Float>(F_prev, lb, q_prevk, rb, q_prevl);

                        Matrix12x12 ddx_ddF;
                        EPJ::ddDeltaTheta_ddF(ddx_ddF, F, F_prev);

                        Matrix12x12 H_F;
                        for(int i = 0; i < 12; ++i) {
                            for(int j = 0; j < 12; ++j) {
                                H_F(i, j) = dE_dx * ddx_ddF(i, j);
                            }
                        }

                        Matrix24x24 JT_H_J;
                        EPJ::JT_H_J<Float>(JT_H_J, H_F, lb, rb, lb, rb);
                        
                        for(int i = 0; i < 24; ++i) {
                            for(int j = 0; j < 24; ++j) {
                                H(i, j) += JT_H_J(i, j);
                            }
                        }
                    };

                    // Write hessians using TripletMatrixAssembler pattern (half block)
                    // HalfHessianSize = 3 for 2 bodies (2*(2+1)/2 = 3)
                    // Blocks: (0,0), (0,1), (1,1)
                    IndexT h_offset = HalfHessianSize * I;
                    
                    // Block (0,0): bid[0], bid[0]
                    H12x12s_row_indices.write(h_offset, bid[0]);
                    H12x12s_col_indices.write(h_offset, bid[0]);
                    Matrix12x12 H_00;
                    for(int i = 0; i < 12; ++i) {
                        for(int j = 0; j < 12; ++j) {
                            H_00(i, j) = H(i, j);
                        }
                    }
                    H12x12s_values.write(h_offset, H_00);
                    
                    // Block (0,1): bid[0], bid[1] (or swapped for upper triangular)
                    IndexT L = (bid[0] < bid[1]) ? 0 : 1;
                    IndexT R = (bid[0] < bid[1]) ? 1 : 0;
                    IndexT row_L = (L == 0) ? bid[0] : bid[1];
                    IndexT col_R = (R == 0) ? bid[0] : bid[1];
                    H12x12s_row_indices.write(h_offset + 1, row_L);
                    H12x12s_col_indices.write(h_offset + 1, col_R);
                    
                    Matrix12x12 H_LR;
                    for(int i = 0; i < 12; ++i) {
                        for(int j = 0; j < 12; ++j) {
                            if(L == 0) {
                                H_LR(i, j) = H(i, 12 + j);  // (0,1) block
                            } else {
                                H_LR(i, j) = H(12 + i, j);  // (1,0) block
                            }
                        }
                    }
                    H12x12s_values.write(h_offset + 1, H_LR);
                    
                    // Block (1,1): bid[1], bid[1]
                    H12x12s_row_indices.write(h_offset + 2, bid[1]);
                    H12x12s_col_indices.write(h_offset + 2, bid[1]);
                    Matrix12x12 H_11;
                    for(int i = 0; i < 12; ++i) {
                        for(int j = 0; j < 12; ++j) {
                            H_11(i, j) = H(12 + i, 12 + j);
                        }
                    }
                    H12x12s_values.write(h_offset + 2, H_11);
                };
            };
        };

        auto kernel = device.compile(compute_gradient_hessian_kernel);
        
        stream << kernel(body_ids_view,
                         l_basis_view,
                         r_basis_view,
                         ref_qs_view,
                         lowers_view,
                         uppers_view,
                         strengths_view,
                         qs_view,
                         q_prevs_view,
                         G_values,
                         G_indices,
                         H_values,
                         H_row_indices,
                         H_col_indices,
                         gradient_only)
                      .dispatch(joint_count);
    }

    U64 get_uid() const noexcept override { return ConstitutionUID; }
};

REGISTER_SIM_SYSTEM(AffineBodyPrismaticJointLimit);
}  // namespace uipc::backend::luisa
