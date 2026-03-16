#include <affine_body/inter_affine_body_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/constitutions/affine_body_revolute_joint_function.h>
#include <uipc/common/enumerate.h>
#include <utils/make_spd.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
static constexpr U64          ConstitutionUID = 18;
class AffineBodyRevoluteJoint final : public InterAffineBodyConstitution
{
  public:
    using InterAffineBodyConstitution::InterAffineBodyConstitution;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    vector<Vector2i> h_body_ids;
    // [    body0   |   body1    ]
    // [    x0, x1  |   x2, x3   ]
    vector<Vector12> h_rest_positions;
    vector<Float>    h_strength_ratio;

    luisa::compute::Buffer<Vector2i> body_ids;
    luisa::compute::Buffer<Vector12> rest_positions;
    luisa::compute::Buffer<Float>    strength_ratio;


    void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = require<AffineBodyDynamics>();
    }

    void do_init(FilteredInfo& info) override
    {
        auto geo_slots = world().scene().geometries();

        list<Vector2i> body_ids_list;
        list<Vector12> rest_positions_list;
        list<Float>    strength_ratio_list;

        info.for_each(
            geo_slots,
            [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
            {
                auto uid = geo.meta().find<U64>(builtin::constitution_uid);
                U64  uid_value = uid->view()[0];
                UIPC_ASSERT(uid_value == ConstitutionUID,
                            "AffineBodyRevoluteJoint: Geometry constitution UID mismatch");

                auto joint_geo_id = I.geo_info().geo_id;

                auto sc = geo.as<geometry::SimplicialComplex>();
                UIPC_ASSERT(sc, "AffineBodyRevoluteJoint: Geometry must be a simplicial complex");

                auto geo_ids = sc->edges().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids, "AffineBodyRevoluteJoint: Geometry must have 'geo_ids' attribute on `edges`");
                auto geo_ids_view = geo_ids->view();

                auto inst_ids = sc->edges().find<Vector2i>("inst_ids");
                UIPC_ASSERT(inst_ids, "AffineBodyRevoluteJoint: Geometry must have 'inst_ids' attribute on `edges`");
                auto inst_ids_view = inst_ids->view();

                auto strength_ratio = sc->edges().find<Float>("strength_ratio");
                UIPC_ASSERT(strength_ratio, "AffineBodyRevoluteJoint: Geometry must have 'strength_ratio' attribute on `edges`");
                auto strength_ratio_view = strength_ratio->view();

                auto Es = sc->edges().topo().view();
                auto Ps = sc->positions().view();
                for(auto&& [i, e] : enumerate(Es))
                {
                    Vector2i geo_id  = geo_ids_view[i];
                    Vector2i inst_id = inst_ids_view[i];

                    Vector3 P0  = Ps[e[0]];
                    Vector3 P1  = Ps[e[1]];
                    Vector3 mid = (P0 + P1) / 2;
                    Vector3 Dir = (P1 - P0);

                    UIPC_ASSERT(Dir.norm() > 1e-12,
                                R"(AffineBodyRevoluteJoint: Edge with zero length detected,
Joint GeometryID = {},
LinkGeoIDs       = ({}, {}),
LinkInstIDs      = ({}, {}),
Edge             = ({}, {}))",
                                joint_geo_id,
                                geo_id(0),
                                geo_id(1),
                                inst_id(0),
                                inst_id(1),
                                e(0),
                                e(1));

                    Vector3 HalfAxis = Dir.normalized() / 2;

                    // Re-define P0 and P1 to be symmetric around the mid-point
                    P0 = mid - HalfAxis;
                    P1 = mid + HalfAxis;

                    Vector2i body_ids = {info.body_id(geo_id(0), inst_id(0)),
                                         info.body_id(geo_id(1), inst_id(1))};
                    body_ids_list.push_back(body_ids);

                    auto left_sc  = info.body_geo(geo_slots, geo_id(0));
                    auto right_sc = info.body_geo(geo_slots, geo_id(1));

                    UIPC_ASSERT(inst_id(0) >= 0
                                    && inst_id(0) < static_cast<IndexT>(
                                           left_sc->instances().size()),
                                "AffineBodyRevoluteJoint: Left instance ID {} is out of range [0, {})",
                                inst_id(0),
                                left_sc->instances().size());
                    UIPC_ASSERT(inst_id(1) >= 0
                                    && inst_id(1) < static_cast<IndexT>(
                                           right_sc->instances().size()),
                                "AffineBodyRevoluteJoint: Right instance ID {} is out of range [0, {})",
                                inst_id(1),
                                right_sc->instances().size());

                    Transform LT{left_sc->transforms().view()[inst_id(0)]};
                    Transform RT{right_sc->transforms().view()[inst_id(1)]};

                    Vector12 rest_pos;
                    rest_pos.segment<3>(0) = LT.inverse() * P0;  // x0_bar
                    rest_pos.segment<3>(3) = LT.inverse() * P1;  // x1_bar

                    rest_pos.segment<3>(6) = RT.inverse() * P0;  // x2_bar
                    rest_pos.segment<3>(9) = RT.inverse() * P1;  // x3_bar
                    rest_positions_list.push_back(rest_pos);
                }

                std::ranges::copy(strength_ratio_view,
                                  std::back_inserter(strength_ratio_list));
            });

        h_body_ids.resize(body_ids_list.size());
        std::ranges::move(body_ids_list, h_body_ids.begin());

        h_rest_positions.resize(rest_positions_list.size());
        std::ranges::move(rest_positions_list, h_rest_positions.begin());

        h_strength_ratio.resize(strength_ratio_list.size());
        std::ranges::move(strength_ratio_list, h_strength_ratio.begin());

        // Create device buffers and copy data
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        body_ids = device.create_buffer<Vector2i>(h_body_ids.size());
        rest_positions = device.create_buffer<Vector12>(h_rest_positions.size());
        strength_ratio = device.create_buffer<Float>(h_strength_ratio.size());

        body_ids.view().copy_from(h_body_ids.data());
        rest_positions.view().copy_from(h_rest_positions.data());
        strength_ratio.view().copy_from(h_strength_ratio.data());
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(body_ids.size());  // one energy per joint
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace RJ = sym::affine_body_revolute_joint;
        
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto body_ids_view = body_ids.view();
        auto rest_positions_view = rest_positions.view();
        auto strength_ratio_view = strength_ratio.view();
        auto body_masses_view = info.body_masses();
        auto qs_view = info.qs();
        auto energies_view = info.energies();

        SizeT joint_count = body_ids.size();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector2i> body_ids,
                                              BufferVar<Vector12> rest_positions,
                                              BufferVar<Float> strength_ratio,
                                              BufferVar<ABDJacobiDyadicMass> body_masses,
                                              BufferVar<Vector12> qs,
                                              BufferVar<Float> Es) {
            auto I = dispatch_id().x;
            $if(I < joint_count) {
                Vector2i bids = body_ids.read(I);

                Float kappa = strength_ratio.read(I)
                              * (body_masses.read(bids[0]).mass()
                                 + body_masses.read(bids[1]).mass());

                Vector12 X_bar = rest_positions.read(I);

                Vector12 q_i = qs.read(bids[0]);
                Vector12 q_j = qs.read(bids[1]);

                // Extract x_bar values manually
                float3 x_bar[4];
                for(int k = 0; k < 3; ++k) {
                    x_bar[0][k] = X_bar[k];
                    x_bar[1][k] = X_bar[3 + k];
                    x_bar[2][k] = X_bar[6 + k];
                    x_bar[3][k] = X_bar[9 + k];
                }

                ABDJacobi Js[4] = {ABDJacobi{x_bar[0]},
                                   ABDJacobi{x_bar[1]},
                                   ABDJacobi{x_bar[2]},
                                   ABDJacobi{x_bar[3]}};

                float3 X[4];
                X[0] = Js[0].point_x(q_i);
                X[1] = Js[1].point_x(q_i);
                X[2] = Js[2].point_x(q_j);
                X[3] = Js[3].point_x(q_j);

                float3 D02 = X[0] - X[2];
                float3 D13 = X[1] - X[3];

                Float E0 = luisa::dot(D02, D02);
                Float E1 = luisa::dot(D13, D13);

                // energy = 1/2 * kappa * (||x0 - x2||^2 + ||x1 - x3||^2)
                Es.write(I, kappa / 2 * (E0 + E1));
            };
        };

        auto kernel = device.compile(compute_energy_kernel);
        stream << kernel(body_ids_view,
                         rest_positions_view,
                         strength_ratio_view,
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
        namespace RJ = sym::affine_body_revolute_joint;
        
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto body_ids_view = body_ids.view();
        auto rest_positions_view = rest_positions.view();
        auto strength_ratio_view = strength_ratio.view();
        auto body_masses_view = info.body_masses();
        auto qs_view = info.qs();
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
                                                       BufferVar<Vector12> rest_positions,
                                                       BufferVar<Float> strength_ratio,
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
                Vector12 X_bar = rest_positions.read(I);

                Vector12 q_i = qs.read(bids[0]);
                Vector12 q_j = qs.read(bids[1]);

                // Extract x_bar values manually
                float3 x_bar[4];
                for(int k = 0; k < 3; ++k) {
                    x_bar[0][k] = X_bar[k];
                    x_bar[1][k] = X_bar[3 + k];
                    x_bar[2][k] = X_bar[6 + k];
                    x_bar[3][k] = X_bar[9 + k];
                }

                ABDJacobi Js[4] = {ABDJacobi{x_bar[0]},
                                   ABDJacobi{x_bar[1]},
                                   ABDJacobi{x_bar[2]},
                                   ABDJacobi{x_bar[3]}};

                float3 X[4];
                X[0] = Js[0].point_x(q_i);
                X[1] = Js[1].point_x(q_i);
                X[2] = Js[2].point_x(q_j);
                X[3] = Js[3].point_x(q_j);

                float3 D02 = X[0] - X[2];
                float3 D13 = X[1] - X[3];

                Float K = strength_ratio.read(I)
                          * (body_masses.read(bids[0]).mass()
                             + body_masses.read(bids[1]).mass());

                // Fill Body Gradient:
                // G = 0.5 * kappa * (J0^T * (x0 - x2) + J1^T * (x1 - x3))
                {
                    Vector12 G_i = K * (Js[0].T() * D02 + Js[1].T() * D13);
                    G12s_indices.write(2 * I + 0, bids[0]);
                    G12s_values.write(2 * I + 0, G_i);
                }
                {
                    // G = 0.5 * kappa * (J2^T * (x2 - x0) + J3^T * (x3 - x1))
                    Vector12 G_j = K * (Js[2].T() * (-D02) + Js[3].T() * (-D13));
                    G12s_indices.write(2 * I + 1, bids[1]);
                    G12s_values.write(2 * I + 1, G_j);
                }

                $if(!gradient_only) {
                    // Fill Body Hessian:
                    // Block (0,0): bids[0], bids[0]
                    {
                        Matrix12x12 H_ii;
                        RJ::Hess(H_ii,
                                 K,
                                 x_bar[0],
                                 x_bar[0],
                                 x_bar[1],
                                 x_bar[1]);
                        H12x12s_row_indices.write(HalfHessianSize * I + 0, bids[0]);
                        H12x12s_col_indices.write(HalfHessianSize * I + 0, bids[0]);
                        H12x12s_values.write(HalfHessianSize * I + 0, H_ii);
                    }
                    // Block (0,1) or (1,0): off-diagonal
                    {
                        Matrix12x12 H;
                        IndexT row_L, col_R;
                        RJ::Hess(H,
                                 -K,
                                 x_bar[0],
                                 x_bar[2],
                                 x_bar[1],
                                 x_bar[3]);
                        if(bids[0] < bids[1]) {
                            row_L = bids[0];
                            col_R = bids[1];
                        } else {
                            // Transpose for upper triangular storage
                            row_L = bids[1];
                            col_R = bids[0];
                            // Transpose H in place
                            Matrix12x12 H_T;
                            for(int i = 0; i < 12; ++i) {
                                for(int j = 0; j < 12; ++j) {
                                    H_T(i, j) = H(j, i);
                                }
                            }
                            H = H_T;
                        }
                        H12x12s_row_indices.write(HalfHessianSize * I + 1, row_L);
                        H12x12s_col_indices.write(HalfHessianSize * I + 1, col_R);
                        H12x12s_values.write(HalfHessianSize * I + 1, H);
                    }
                    // Block (1,1): bids[1], bids[1]
                    {
                        Matrix12x12 H_jj;
                        RJ::Hess(H_jj,
                                 K,
                                 x_bar[2],
                                 x_bar[2],
                                 x_bar[3],
                                 x_bar[3]);
                        H12x12s_row_indices.write(HalfHessianSize * I + 2, bids[1]);
                        H12x12s_col_indices.write(HalfHessianSize * I + 2, bids[1]);
                        H12x12s_values.write(HalfHessianSize * I + 2, H_jj);
                    }
                };
            };
        };

        auto kernel = device.compile(compute_gradient_hessian_kernel);
        
        stream << kernel(body_ids_view,
                         rest_positions_view,
                         strength_ratio_view,
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

REGISTER_SIM_SYSTEM(AffineBodyRevoluteJoint);
}  // namespace uipc::backend::luisa
