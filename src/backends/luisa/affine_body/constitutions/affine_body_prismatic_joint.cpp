#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constitutions/affine_body_prismatic_joint_function.h>
#include <time_integrator/time_integrator.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/offset_count_collection.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <uipc/common/enumerate.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class AffineBodyPrismaticJoint final : public InterAffineBodyConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 20;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;
    using InterAffineBodyConstitution::InterAffineBodyConstitution;


    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    // [    body0   |   body1    ]
    luisa::compute::Buffer<Vector2i> body_ids;
    luisa::compute::Buffer<Vector6>  rest_cs;  // c_bar
    luisa::compute::Buffer<Vector6>  rest_ts;  // t_bar
    luisa::compute::Buffer<Vector6>  rest_ns;  // n_bar
    luisa::compute::Buffer<Vector6>  rest_bs;  // b_bar
    luisa::compute::Buffer<Float>    strength_ratios;

    vector<Vector2i> h_body_ids;
    vector<Vector6>  h_rest_cs;
    vector<Vector6>  h_rest_ts;
    vector<Vector6>  h_rest_ns;
    vector<Vector6>  h_rest_bs;
    vector<Float>    h_strength_ratios;

    using Vector24    = Vector<Float, 24>;
    using Matrix24x24 = Matrix<Float, 24, 24>;

    void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = require<AffineBodyDynamics>();
    }

    void do_init(FilteredInfo& info) override
    {
        auto geo_slots = world().scene().geometries();

        list<Vector2i> body_ids_list;
        list<Vector6>  rest_c_list;
        list<Vector6>  rest_t_list;
        list<Vector6>  rest_n_list;
        list<Vector6>  rest_b_list;
        list<Float>    strength_ratio_list;

        info.for_each(
            geo_slots,
            [&](geometry::Geometry& geo)
            {
                auto uid = geo.meta().find<U64>(builtin::constitution_uid);
                U64  uid_value = uid->view()[0];
                UIPC_ASSERT(uid_value == ConstitutionUID,
                            "AffineBodyPrismaticJoint: Geometry constitution UID mismatch");

                auto sc = geo.as<geometry::SimplicialComplex>();

                auto geo_ids = sc->edges().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids, "AffineBodyPrismaticJoint: Geometry must have 'geo_ids' attribute on `edges`");
                auto geo_ids_view = geo_ids->view();

                auto inst_ids = sc->edges().find<Vector2i>("inst_ids");
                UIPC_ASSERT(inst_ids, "AffineBodyPrismaticJoint: Geometry must have 'inst_ids' attribute on `edges`");
                auto inst_ids_view = inst_ids->view();

                auto strength_ratio = sc->edges().find<Float>("strength_ratio");
                UIPC_ASSERT(strength_ratio, "AffineBodyPrismaticJoint: Geometry must have 'strength_ratio' attribute on `edges`");
                auto strength_ratio_view = strength_ratio->view();

                auto Normal = [&](const Vector3& W) -> Vector3
                {
                    Vector3 ref = std::abs(W.dot(Vector3(1, 0, 0))) < 0.9 ?
                                      Vector3(1, 0, 0) :
                                      Vector3(0, 1, 0);

                    Vector3 U = ref.cross(W).normalized();
                    Vector3 V = W.cross(U).normalized();

                    return V;
                };

                auto Es = sc->edges().topo().view();
                auto Ps = sc->positions().view();
                for(auto&& [i, e] : enumerate(Es))
                {
                    Vector3 P0 = Ps[e[0]];
                    Vector3 P1 = Ps[e[1]];
                    UIPC_ASSERT((P0 - P1).squaredNorm() > 0,
                                "AffineBodyPrismaticJoint: Edge positions must not be too close");

                    Vector2i geo_id  = geo_ids_view[i];
                    Vector2i inst_id = inst_ids_view[i];

                    Vector2i body_ids = {info.body_id(geo_id(0), inst_id(0)),
                                         info.body_id(geo_id(1), inst_id(1))};
                    body_ids_list.push_back(body_ids);

                    auto left_sc  = info.body_geo(geo_slots, geo_id(0));
                    auto right_sc = info.body_geo(geo_slots, geo_id(1));

                    UIPC_ASSERT(inst_id(0) >= 0
                                    && inst_id(0) < static_cast<IndexT>(
                                           left_sc->instances().size()),
                                "AffineBodyPrismaticJoint: Left instance ID {} is out of range [0, {})",
                                inst_id(0),
                                left_sc->instances().size());
                    UIPC_ASSERT(inst_id(1) >= 0
                                    && inst_id(1) < static_cast<IndexT>(
                                           right_sc->instances().size()),
                                "AffineBodyPrismaticJoint: Right instance ID {} is out of range [0, {})",
                                inst_id(1),
                                right_sc->instances().size());

                    Transform LT{left_sc->transforms().view()[inst_id(0)]};
                    Transform RT{right_sc->transforms().view()[inst_id(1)]};

                    Vector3 tangent   = (P1 - P0).normalized();
                    Vector3 normal    = Normal(tangent);
                    Vector3 bitangent = normal.cross(tangent).normalized();

                    Vector6 rest_position_c;
                    rest_position_c.segment<3>(0) = LT.inverse() * P0;  // ci_bar
                    rest_position_c.segment<3>(3) = RT.inverse() * P0;  // cj_bar
                    rest_c_list.push_back(rest_position_c);

                    Matrix3x3 LT_rotation = LT.rotation();
                    Matrix3x3 RT_rotation = RT.rotation();

                    Vector6 rest_vec_t;
                    rest_vec_t.segment<3>(0) = LT_rotation.inverse() * tangent;  // ti_bar
                    rest_vec_t.segment<3>(3) = RT_rotation.inverse() * tangent;  // tj_bar
                    rest_t_list.push_back(rest_vec_t);

                    Vector6 rest_vec_n;
                    rest_vec_n.segment<3>(0) = LT_rotation.inverse() * normal;  // ni_bar
                    rest_vec_n.segment<3>(3) = RT_rotation.inverse() * normal;  // nj_bar
                    rest_n_list.push_back(rest_vec_n);

                    Vector6 rest_vec_b;
                    rest_vec_b.segment<3>(0) = LT_rotation.inverse() * bitangent;  // bi_bar
                    rest_vec_b.segment<3>(3) = RT_rotation.inverse() * bitangent;  // bj_bar
                    rest_b_list.push_back(rest_vec_b);
                }

                std::ranges::copy(strength_ratio_view,
                                  std::back_inserter(strength_ratio_list));
            });

        h_body_ids.resize(body_ids_list.size());
        std::ranges::move(body_ids_list, h_body_ids.begin());

        h_rest_cs.resize(rest_c_list.size());
        std::ranges::move(rest_c_list, h_rest_cs.begin());

        h_rest_ts.resize(rest_t_list.size());
        std::ranges::move(rest_t_list, h_rest_ts.begin());

        h_rest_ns.resize(rest_n_list.size());
        std::ranges::move(rest_n_list, h_rest_ns.begin());

        h_rest_bs.resize(rest_b_list.size());
        std::ranges::move(rest_b_list, h_rest_bs.begin());

        h_strength_ratios.resize(strength_ratio_list.size());
        std::ranges::move(strength_ratio_list, h_strength_ratios.begin());

        // Create device buffers
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        body_ids = device.create_buffer<Vector2i>(h_body_ids.size());
        rest_cs = device.create_buffer<Vector6>(h_rest_cs.size());
        rest_ts = device.create_buffer<Vector6>(h_rest_ts.size());
        rest_ns = device.create_buffer<Vector6>(h_rest_ns.size());
        rest_bs = device.create_buffer<Vector6>(h_rest_bs.size());
        strength_ratios = device.create_buffer<Float>(h_strength_ratios.size());

        // Copy data to device
        body_ids.view().copy_from(h_body_ids.data());
        rest_cs.view().copy_from(h_rest_cs.data());
        rest_ts.view().copy_from(h_rest_ts.data());
        rest_ns.view().copy_from(h_rest_ns.data());
        rest_bs.view().copy_from(h_rest_bs.data());
        strength_ratios.view().copy_from(h_strength_ratios.data());
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(body_ids.size());  // one energy per joint
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace PJ = sym::affine_body_prismatic_joint;
        
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto body_ids_view = body_ids.view();
        auto rest_cs_view = rest_cs.view();
        auto rest_ts_view = rest_ts.view();
        auto rest_ns_view = rest_ns.view();
        auto rest_bs_view = rest_bs.view();
        auto strength_ratios_view = strength_ratios.view();
        auto body_masses_view = info.body_masses();
        auto qs_view = info.qs();
        auto energies_view = info.energies();

        SizeT joint_count = body_ids.size();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector2i> body_ids,
                                              BufferVar<Vector6> rest_cs,
                                              BufferVar<Vector6> rest_ts,
                                              BufferVar<Vector6> rest_ns,
                                              BufferVar<Vector6> rest_bs,
                                              BufferVar<Float> strength_ratios,
                                              BufferVar<ABDJacobiDyadicMass> body_masses,
                                              BufferVar<Vector12> qs,
                                              BufferVar<Float> Es) {
            auto I = dispatch_id().x;
            $if(I < joint_count) {
                Vector2i bids = body_ids.read(I);

                Float kappa = strength_ratios.read(I)
                              * (body_masses.read(bids[0]).mass()
                                 + body_masses.read(bids[1]).mass());

                Vector12 qi = qs.read(bids[0]);
                Vector12 qj = qs.read(bids[1]);

                Vector6 rest_c = rest_cs.read(I);
                Vector6 rest_t = rest_ts.read(I);
                Vector6 rest_n = rest_ns.read(I);
                Vector6 rest_b = rest_bs.read(I);

                // Create Frame F01
                Vector9 F01;
                // Extract segments manually
                Vector3 ci_bar, cj_bar, ti_bar, tj_bar;
                for(int k = 0; k < 3; ++k) {
                    ci_bar[k] = rest_c[k];
                    cj_bar[k] = rest_c[3 + k];
                    ti_bar[k] = rest_t[k];
                    tj_bar[k] = rest_t[3 + k];
                }
                
                PJ::F01<Float>(F01,
                               ci_bar,
                               ti_bar,
                               qi,
                               cj_bar,
                               tj_bar,
                               qj);

                Float E01;
                PJ::E01(E01, kappa, F01);

                Vector3 ni_bar, bi_bar, nj_bar, bj_bar;
                for(int k = 0; k < 3; ++k) {
                    ni_bar[k] = rest_n[k];
                    bi_bar[k] = rest_b[k];
                    nj_bar[k] = rest_n[3 + k];
                    bj_bar[k] = rest_b[3 + k];
                }

                Float E23;
                PJ::E23<Float>(E23,
                               kappa,
                               ni_bar,
                               bi_bar,
                               qi,
                               nj_bar,
                               bj_bar,
                               qj);

                Es.write(I, E01 + E23);
            };
        };

        auto kernel = device.compile(compute_energy_kernel);
        stream << kernel(body_ids_view,
                         rest_cs_view,
                         rest_ts_view,
                         rest_ns_view,
                         rest_bs_view,
                         strength_ratios_view,
                         body_masses_view,
                         qs_view,
                         energies_view)
                      .dispatch(joint_count);
    }

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override
    {
        info.gradient_count(2 * body_ids.size());  // each joint has 2 * Vector12 gradients

        if(info.gradient_only())
            return;

        info.hessian_count(HalfHessianSize * body_ids.size());  // each joint has HalfHessianSize * Matrix12x12 hessians
    }

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace PJ = sym::affine_body_prismatic_joint;
        
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto body_ids_view = body_ids.view();
        auto rest_cs_view = rest_cs.view();
        auto rest_ts_view = rest_ts.view();
        auto rest_ns_view = rest_ns.view();
        auto rest_bs_view = rest_bs.view();
        auto strength_ratios_view = strength_ratios.view();
        auto body_masses_view = info.body_masses();
        auto qs_view = info.qs();
        auto gradients_view = info.gradients();
        auto hessians_view = info.hessians();
        bool gradient_only = info.gradient_only();

        SizeT joint_count = body_ids.size();

        // The views from info.gradients() and info.hessians() use BufferView<const T>
        // but the constitutions need to write to them. We use a workaround by
        // creating mutable buffer views from the underlying buffers.
        
        // Note: This is a temporary workaround. The proper fix is to modify
        // InterAffineBodyConstitutionManager to use mutable view types.
        
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
                                                       BufferVar<Vector6> rest_cs,
                                                       BufferVar<Vector6> rest_ts,
                                                       BufferVar<Vector6> rest_ns,
                                                       BufferVar<Vector6> rest_bs,
                                                       BufferVar<Float> strength_ratios,
                                                       BufferVar<ABDJacobiDyadicMass> body_masses,
                                                       BufferVar<Vector12> qs,
                                                       BufferVar<Vector12> G12s_values,
                                                       BufferVar<luisa::uint> G12s_indices,
                                                       BufferVar<Matrix12x12> H12x12s_values,
                                                       BufferVar<luisa::uint> H12x12s_row_indices,
                                                       BufferVar<luisa::uint> H12x12s_col_indices,
                                                       Bool gradient_only) {
            auto I = dispatch_id().x;
            $if(I < joint_count) {
                Vector2i bids = body_ids.read(I);

                Float kappa = strength_ratios.read(I)
                              * (body_masses.read(bids[0]).mass()
                                 + body_masses.read(bids[1]).mass());

                Vector12 qi = qs.read(bids[0]);
                Vector12 qj = qs.read(bids[1]);

                Vector6 rest_c = rest_cs.read(I);
                Vector6 rest_t = rest_ts.read(I);
                Vector6 rest_n = rest_ns.read(I);
                Vector6 rest_b = rest_bs.read(I);

                // Extract segments manually
                Vector3 ci_bar, cj_bar, ti_bar, tj_bar;
                for(int k = 0; k < 3; ++k) {
                    ci_bar[k] = rest_c[k];
                    cj_bar[k] = rest_c[3 + k];
                    ti_bar[k] = rest_t[k];
                    tj_bar[k] = rest_t[3 + k];
                }

                // Create Frame F01
                Vector9 F01;
                PJ::F01<Float>(F01,
                               ci_bar,
                               ti_bar,
                               qi,
                               cj_bar,
                               tj_bar,
                               qj);

                // Get Gradient based on Frame F01
                Vector9 G01;
                PJ::dE01dF01(G01, kappa, F01);
                Vector24 J01T_G01;
                PJ::J01T_G01<Float>(J01T_G01,
                                    G01,
                                    ci_bar,
                                    ti_bar,
                                    cj_bar,
                                    tj_bar);

                // Get Gradient based on [qi,qj] directly
                Vector3 ni_bar, bi_bar, nj_bar, bj_bar;
                for(int k = 0; k < 3; ++k) {
                    ni_bar[k] = rest_n[k];
                    bi_bar[k] = rest_b[k];
                    nj_bar[k] = rest_n[3 + k];
                    bj_bar[k] = rest_b[3 + k];
                }

                Vector24 dE23dQ;
                PJ::dE23dQ<Float>(dE23dQ,
                                  kappa,
                                  ni_bar,
                                  bi_bar,
                                  qi,
                                  nj_bar,
                                  bj_bar,
                                  qj);

                Vector24 G;
                for(int k = 0; k < 24; ++k) {
                    G[k] = J01T_G01[k] + dE23dQ[k];
                }

                // Write gradients using DoubletVectorAssembler pattern
                // Each joint writes 2 gradients (one for each body)
                IndexT offset = 2 * I;
                for(int ii = 0; ii < 2; ++ii) {
                    IndexT dst = (ii == 0) ? bids[0] : bids[1];
                    G12s_indices.write(offset + ii, dst);
                    
                    // Write Vector12 for this body
                    Vector12 G12;
                    for(int k = 0; k < 12; ++k) {
                        G12[k] = G[ii * 12 + k];
                    }
                    G12s_values.write(offset + ii, G12);
                }

                $if(!gradient_only) {
                    // Get Hessian based on Frame F01
                    Matrix9x9 H01;
                    PJ::ddE01ddF01(H01, kappa, F01);

                    // Ensure H01 is SPD
                    make_spd(H01);

                    Matrix24x24 J01T_H01_J01;
                    PJ::J01T_H01_J01<Float>(J01T_H01_J01,
                                            H01,
                                            ci_bar,
                                            ti_bar,
                                            cj_bar,
                                            tj_bar);

                    // Get Hessian based on [qi,qj] directly
                    Matrix24x24 ddE23ddQ;
                    PJ::ddE23ddQ<Float>(ddE23ddQ,
                                        kappa,
                                        ni_bar,
                                        bi_bar,
                                        qi,
                                        nj_bar,
                                        bj_bar,
                                        qj);

                    Matrix24x24 H;
                    for(int i = 0; i < 24; ++i) {
                        for(int j = 0; j < 24; ++j) {
                            H(i, j) = J01T_H01_J01(i, j) + ddE23ddQ(i, j);
                        }
                    }

                    // Write hessians using TripletMatrixAssembler pattern (half block)
                    // HalfHessianSize = 3 for 2 bodies (2*(2+1)/2 = 3)
                    // Blocks: (0,0), (0,1), (1,1)
                    IndexT h_offset = HalfHessianSize * I;
                    
                    // Block (0,0): bids[0], bids[0]
                    H12x12s_row_indices.write(h_offset, bids[0]);
                    H12x12s_col_indices.write(h_offset, bids[0]);
                    Matrix12x12 H_00;
                    for(int i = 0; i < 12; ++i) {
                        for(int j = 0; j < 12; ++j) {
                            H_00(i, j) = H(i, j);
                        }
                    }
                    H12x12s_values.write(h_offset, H_00);
                    
                    // Block (0,1): bids[0], bids[1] (or swapped for upper triangular)
                    IndexT L = (bids[0] < bids[1]) ? 0 : 1;
                    IndexT R = (bids[0] < bids[1]) ? 1 : 0;
                    IndexT row_L = (L == 0) ? bids[0] : bids[1];
                    IndexT col_R = (R == 0) ? bids[0] : bids[1];
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
                    
                    // Block (1,1): bids[1], bids[1]
                    H12x12s_row_indices.write(h_offset + 2, bids[1]);
                    H12x12s_col_indices.write(h_offset + 2, bids[1]);
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
                         rest_cs_view,
                         rest_ts_view,
                         rest_ns_view,
                         rest_bs_view,
                         strength_ratios_view,
                         body_masses_view,
                         qs_view,
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
REGISTER_SIM_SYSTEM(AffineBodyPrismaticJoint);
}  // namespace uipc::backend::luisa
