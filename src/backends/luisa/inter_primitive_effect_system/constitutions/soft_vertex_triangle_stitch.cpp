#include <inter_primitive_effect_system/inter_primitive_constitution.h>
#include <inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch_function.h>
#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <Eigen/LU>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class SoftVertexTriangleStitch : public InterPrimitiveConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 30;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using InterPrimitiveConstitution::InterPrimitiveConstitution;

    U64 get_uid() const noexcept override { return ConstitutionUID; }

    void do_build(BuildInfo& info) override {}

    luisa::compute::Buffer<Vector4i>  topos;
    luisa::compute::Buffer<Float>     mus;
    luisa::compute::Buffer<Float>     lambdas;
    luisa::compute::Buffer<Matrix3x3> Dm_invs;
    luisa::compute::Buffer<Float>     rest_volumes;

    vector<Vector4i>  h_topos;
    vector<Float>     h_mus;
    vector<Float>     h_lambdas;
    vector<Matrix3x3> h_Dm_invs;
    vector<Float>     h_rest_volumes;

    luisa::compute::BufferView<const Float>              energies;
    MutableDoubletVectorView<Float, 3>    gradients;
    MutableTripletMatrixView<Float, 3, 3> hessians;

    void do_init(FilteredInfo& info) override
    {
        list<Vector4i>  topo_buffer;
        list<Float>     mu_buffer;
        list<Float>     lambda_buffer;
        list<Matrix3x3> Dm_inv_buffer;
        list<Float>     rest_volume_buffer;

        auto geo_slots    = world().scene().geometries();
        using ForEachInfo = InterPrimitiveConstitutionManager::ForEachInfo;
        info.for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::Geometry& geo)
            {
                auto topo = geo.instances().find<Vector4i>(builtin::topo);
                UIPC_ASSERT(topo, "SoftVertexTriangleStitch requires attribute `topo` on instances()");
                auto geo_ids = geo.meta().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids, "SoftVertexTriangleStitch requires attribute `geo_ids` on meta()");
                auto mu_slot = geo.instances().find<Float>("mu");
                UIPC_ASSERT(mu_slot, "SoftVertexTriangleStitch requires attribute `mu` on instances()");
                auto lambda_slot = geo.instances().find<Float>("lambda");
                UIPC_ASSERT(lambda_slot, "SoftVertexTriangleStitch requires attribute `lambda` on instances()");

                Vector2i ids         = geo_ids->view()[0];
                auto     l_slot      = info.geo_slot(ids[0]);
                auto     r_slot      = info.geo_slot(ids[1]);
                auto     l_rest_slot = info.rest_geo_slot(ids[0]);
                auto     r_rest_slot = info.rest_geo_slot(ids[1]);
                UIPC_ASSERT(l_rest_slot,
                            "SoftVertexTriangleStitch requires rest geometry for slot id {}",
                            ids[0]);
                UIPC_ASSERT(r_rest_slot,
                            "SoftVertexTriangleStitch requires rest geometry for slot id {}",
                            ids[1]);

                auto l_geo = l_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(l_geo,
                            "SoftVertexTriangleStitch requires simplicial complex geometry, but got {} ({})",
                            l_slot->geometry().type(),
                            l_slot->id());
                auto r_geo = r_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(r_geo,
                            "SoftVertexTriangleStitch requires simplicial complex geometry, but got {} ({})",
                            r_slot->geometry().type(),
                            r_slot->id());
                auto l_rest_geo =
                    l_rest_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(l_rest_geo,
                            "SoftVertexTriangleStitch requires rest simplicial complex for id {}",
                            ids[0]);
                auto r_rest_geo =
                    r_rest_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(r_rest_geo,
                            "SoftVertexTriangleStitch requires rest simplicial complex for id {}",
                            ids[1]);

                auto l_offset = l_geo->meta().find<IndexT>(builtin::global_vertex_offset);
                UIPC_ASSERT(l_offset,
                            "SoftVertexTriangleStitch requires attribute `global_vertex_offset` on meta() of geometry {} ({})",
                            l_slot->geometry().type(),
                            l_slot->id());
                IndexT l_offset_v = l_offset->view()[0];
                auto r_offset = r_geo->meta().find<IndexT>(builtin::global_vertex_offset);
                UIPC_ASSERT(r_offset,
                            "SoftVertexTriangleStitch requires attribute `global_vertex_offset` on meta() of geometry {} ({})",
                            r_slot->geometry().type(),
                            r_slot->id());
                IndexT r_offset_v = r_offset->view()[0];

                auto rest0_pos = l_rest_geo->positions().view();
                auto rest1_pos = r_rest_geo->positions().view();

                Transform l_transform(l_rest_geo->transforms().view()[0]);
                Transform r_transform(r_rest_geo->transforms().view()[0]);

                auto min_sep_slot = geo.instances().find<Float>("min_separate_distance");
                UIPC_ASSERT(min_sep_slot, "SoftVertexTriangleStitch requires per-instance attribute `min_separate_distance`");
                auto min_sep_view = min_sep_slot->view();

                auto  topo_view   = topo->view();
                auto  mu_view     = mu_slot->view();
                auto  lambda_view = lambda_slot->view();
                SizeT n           = topo_view.size();

                for(SizeT i = 0; i < n; ++i)
                {
                    const Vector4i& t = topo_view[i];
                    IndexT  v_id = t(0), tri0 = t(1), tri1 = t(2), tri2 = t(3);
                    Vector3 x0 = l_transform * rest0_pos[v_id];
                    Vector3 x1 = r_transform * rest1_pos[tri0];
                    Vector3 x2 = r_transform * rest1_pos[tri1];
                    Vector3 x3 = r_transform * rest1_pos[tri2];

                    Float   d  = min_sep_view[i];
                    Vector3 e1 = x2 - x1, e2 = x3 - x1;
                    Vector3 normal = e1.cross(e2);
                    constexpr Float geo_degeneracy_tol = 1e-12;

                    Float   nrm    = normal.norm();
                    UIPC_ASSERT(nrm >= geo_degeneracy_tol,
                                "SoftVertexTriangleStitch: triangle ({},{},{}) is degenerate",
                                tri0,
                                tri1,
                                tri2);
                    normal /= nrm;

                    Float signed_dist = normal.dot(x0 - x1);
                    if(std::abs(signed_dist) < d)
                    {
                        Float sign = (signed_dist >= 0) ? 1.0 : -1.0;
                        x0         = x0 + (sign * d - signed_dist) * normal;
                    }

                    Matrix3x3 Dm;
                    Dm.col(0)      = x1 - x0;
                    Dm.col(1)      = x2 - x0;
                    Dm.col(2)      = x3 - x0;
                    Float rest_vol = (1.0 / 6.0) * std::abs(Dm.determinant());

                    topo_buffer.push_back(Vector4i{t(0) + l_offset_v,
                                                   t(1) + r_offset_v,
                                                   t(2) + r_offset_v,
                                                   t(3) + r_offset_v});
                    mu_buffer.push_back(mu_view[i]);
                    lambda_buffer.push_back(lambda_view[i]);
                    Dm_inv_buffer.push_back(Dm.inverse());
                    rest_volume_buffer.push_back(rest_vol);
                }
            });

        h_topos.assign(topo_buffer.begin(), topo_buffer.end());
        h_mus.assign(mu_buffer.begin(), mu_buffer.end());
        h_lambdas.assign(lambda_buffer.begin(), lambda_buffer.end());
        h_Dm_invs.assign(Dm_inv_buffer.begin(), Dm_inv_buffer.end());
        h_rest_volumes.assign(rest_volume_buffer.begin(), rest_volume_buffer.end());

        auto& device = engine().device();
        topos        = device.create_buffer<Vector4i>(h_topos.size());
        mus          = device.create_buffer<Float>(h_mus.size());
        lambdas      = device.create_buffer<Float>(h_lambdas.size());
        Dm_invs      = device.create_buffer<Matrix3x3>(h_Dm_invs.size());
        rest_volumes = device.create_buffer<Float>(h_rest_volumes.size());

        topos.copy_from(h_topos.data());
        mus.copy_from(h_mus.data());
        lambdas.copy_from(h_lambdas.data());
        Dm_invs.copy_from(h_Dm_invs.data());
        rest_volumes.copy_from(h_rest_volumes.data());
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(topos.size());
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace SVTS = sym::soft_vertex_triangle_stitch;

        energies = info.energies();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector4i> topos_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Float> mus_buf,
                                             BufferVar<Float> lambdas_buf,
                                             BufferVar<Matrix3x3> Dm_invs_buf,
                                             BufferVar<Float> rest_vols_buf,
                                             BufferVar<Float> Es_buf,
                                             Float dt_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < topos_buf.size())
            {
                const Vector4i tet = topos_buf.read(I);
                Vector3 x0 = xs_buf.read(tet(0));
                Vector3 x1 = xs_buf.read(tet(1));
                Vector3 x2 = xs_buf.read(tet(2));
                Vector3 x3 = xs_buf.read(tet(3));

                Matrix3x3 F    = fem::F(x0, x1, x2, x3, Dm_invs_buf.read(I));
                Vector9   VecF = flatten(F);

                Float E_val;
                SVTS::E(E_val, mus_buf.read(I), lambdas_buf.read(I), VecF);
                Es_buf.write(I, E_val * dt_val * dt_val * rest_vols_buf.read(I));
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(topos.view(),
                                    info.positions(),
                                    mus.view(),
                                    lambdas.view(),
                                    Dm_invs.view(),
                                    rest_volumes.view(),
                                    info.energies(),
                                    info.dt())
                                .dispatch(topos.size());
    }

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override
    {
        info.gradient_count(StencilSize * topos.size());

        if(info.gradient_only())
            return;

        info.hessian_count(HalfHessianSize * topos.size());
    }

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace SVTS = sym::soft_vertex_triangle_stitch;

        gradients = info.gradients();
        hessians  = info.hessians();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Vector4i> topos_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Float> mus_buf,
                                                BufferVar<Float> lambdas_buf,
                                                BufferVar<Matrix3x3> Dm_invs_buf,
                                                BufferVar<Float> rest_vols_buf,
                                                BufferVar<Float> G3s_buf,
                                                BufferVar<Matrix3x3> H3x3s_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < topos_buf.size())
            {
                const Vector4i tet = topos_buf.read(I);
                Vector3 x0 = xs_buf.read(tet(0));
                Vector3 x1 = xs_buf.read(tet(1));
                Vector3 x2 = xs_buf.read(tet(2));
                Vector3 x3 = xs_buf.read(tet(3));

                Matrix3x3 F    = fem::F(x0, x1, x2, x3, Dm_invs_buf.read(I));
                Vector9   VecF = flatten(F);

                Float Vdt2 = rest_vols_buf.read(I) * dt_val * dt_val;

                Vector9 dEdVecF;
                SVTS::dEdVecF(dEdVecF, mus_buf.read(I), lambdas_buf.read(I), VecF);
                dEdVecF *= Vdt2;

                Matrix9x12 dFdx = fem::dFdx(Dm_invs_buf.read(I));
                Vector12   G    = dFdx.transpose() * dEdVecF;

                // Write gradient using atomic operations
                for(int k = 0; k < 4; ++k)
                {
                    Float3 gk = make_float3(G(3 * k + 0), G(3 * k + 1), G(3 * k + 2));
                    G3s_buf.atomic(tet(k)).x.fetch_add(gk.x);
                    G3s_buf.atomic(tet(k)).y.fetch_add(gk.y);
                    G3s_buf.atomic(tet(k)).z.fetch_add(gk.z);
                }

                $if(!gradient_only_val)
                {
                    Matrix9x9 ddEddVecF;
                    SVTS::ddEddVecF(ddEddVecF, mus_buf.read(I), lambdas_buf.read(I), VecF);
                    ddEddVecF *= Vdt2;
                    make_spd(ddEddVecF);
                    Matrix12x12 H = dFdx.transpose() * ddEddVecF * dFdx;

                    // Write Hessian blocks (4x4 block structure, only upper triangle)
                    // H is 12x12, we write 3x3 blocks
                    int block_idx = I * HalfHessianSize;
                    for(int i = 0; i < 4; ++i)
                    {
                        for(int j = i; j < 4; ++j)
                        {
                            Matrix3x3 H_ij;
                            for(int r = 0; r < 3; ++r)
                                for(int c = 0; c < 3; ++c)
                                    H_ij(r, c) = H(3 * i + r, 3 * j + c);
                            H3x3s_buf.write(block_idx++, H_ij);
                        }
                    }
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_grad_hess_kernel);
        engine().stream() << shader(topos.view(),
                                    info.positions(),
                                    mus.view(),
                                    lambdas.view(),
                                    Dm_invs.view(),
                                    rest_volumes.view(),
                                    info.gradients().buffer(),
                                    info.hessians().buffer(),
                                    info.dt(),
                                    info.gradient_only())
                                .dispatch(topos.size());
    }
};

REGISTER_SIM_SYSTEM(SoftVertexTriangleStitch);
}  // namespace uipc::backend::luisa
