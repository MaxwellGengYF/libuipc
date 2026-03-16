#include <inter_primitive_effect_system/inter_primitive_constitution.h>
#include <inter_primitive_effect_system/constitutions/soft_vertex_edge_stitch_function.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <Eigen/Dense>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class SoftVertexEdgeStitch : public InterPrimitiveConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 29;
    static constexpr SizeT StencilSize     = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using InterPrimitiveConstitution::InterPrimitiveConstitution;

    U64 get_uid() const noexcept override { return ConstitutionUID; }

    void do_build(BuildInfo& info) override {}

    luisa::compute::Buffer<Vector3i>  topos;
    luisa::compute::Buffer<Float>     mus;
    luisa::compute::Buffer<Float>     lambdas;
    luisa::compute::Buffer<Matrix2x2> inv_Bs;
    luisa::compute::Buffer<Float>     rest_areas;
    luisa::compute::Buffer<Float>     thicknesses;

    vector<Vector3i>  h_topos;
    vector<Float>     h_mus;
    vector<Float>     h_lambdas;
    vector<Matrix2x2> h_inv_Bs;
    vector<Float>     h_rest_areas;
    vector<Float>     h_thicknesses;

    luisa::compute::BufferView<const Float>              energies;
    MutableDoubletVectorView<Float, 3>    gradients;
    MutableTripletMatrixView<Float, 3, 3> hessians;

    void do_init(FilteredInfo& info) override
    {
        list<Vector3i>  topo_buffer;
        list<Float>     mu_buffer;
        list<Float>     lambda_buffer;
        list<Matrix2x2> inv_B_buffer;
        list<Float>     rest_area_buffer;
        list<Float>     thickness_buffer;

        auto geo_slots    = world().scene().geometries();
        using ForEachInfo = InterPrimitiveConstitutionManager::ForEachInfo;
        info.for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::Geometry& geo)
            {
                auto topo = geo.instances().find<Vector3i>(builtin::topo);
                UIPC_ASSERT(topo,
                            "SoftVertexEdgeStitch requires attribute `topo` on instances()");
                auto geo_ids = geo.meta().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids,
                            "SoftVertexEdgeStitch requires attribute `geo_ids` on meta()");
                auto mu_slot = geo.instances().find<Float>("mu");
                UIPC_ASSERT(mu_slot,
                            "SoftVertexEdgeStitch requires attribute `mu` on instances()");
                auto lambda_slot = geo.instances().find<Float>("lambda");
                UIPC_ASSERT(lambda_slot,
                            "SoftVertexEdgeStitch requires attribute `lambda` on instances()");
                auto thick_slot = geo.instances().find<Float>("thickness");
                UIPC_ASSERT(thick_slot,
                            "SoftVertexEdgeStitch requires attribute `thickness` on instances()");
                auto min_sep_slot = geo.instances().find<Float>("min_separate_distance");
                UIPC_ASSERT(min_sep_slot,
                            "SoftVertexEdgeStitch requires attribute `min_separate_distance` on instances()");

                Vector2i ids         = geo_ids->view()[0];
                auto     l_slot      = info.geo_slot(ids[0]);
                auto     r_slot      = info.geo_slot(ids[1]);
                auto     l_rest_slot = info.rest_geo_slot(ids[0]);
                auto     r_rest_slot = info.rest_geo_slot(ids[1]);
                UIPC_ASSERT(l_rest_slot,
                            "SoftVertexEdgeStitch requires rest geometry for slot id {}",
                            ids[0]);
                UIPC_ASSERT(r_rest_slot,
                            "SoftVertexEdgeStitch requires rest geometry for slot id {}",
                            ids[1]);

                auto l_geo = l_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(l_geo,
                            "SoftVertexEdgeStitch requires simplicial complex, but got {} ({})",
                            l_slot->geometry().type(),
                            l_slot->id());
                auto r_geo = r_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(r_geo,
                            "SoftVertexEdgeStitch requires simplicial complex, but got {} ({})",
                            r_slot->geometry().type(),
                            r_slot->id());
                auto l_rest_geo =
                    l_rest_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(l_rest_geo,
                            "SoftVertexEdgeStitch requires rest simplicial complex for id {}",
                            ids[0]);
                auto r_rest_geo =
                    r_rest_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(r_rest_geo,
                            "SoftVertexEdgeStitch requires rest simplicial complex for id {}",
                            ids[1]);

                auto l_offset =
                    l_geo->meta().find<IndexT>(builtin::global_vertex_offset);
                UIPC_ASSERT(l_offset,
                            "SoftVertexEdgeStitch requires `global_vertex_offset` on meta() of geometry {} ({})",
                            l_slot->geometry().type(),
                            l_slot->id());
                IndexT l_offset_v = l_offset->view()[0];
                auto   r_offset =
                    r_geo->meta().find<IndexT>(builtin::global_vertex_offset);
                UIPC_ASSERT(r_offset,
                            "SoftVertexEdgeStitch requires `global_vertex_offset` on meta() of geometry {} ({})",
                            r_slot->geometry().type(),
                            r_slot->id());
                IndexT r_offset_v = r_offset->view()[0];

                auto rest0_pos    = l_rest_geo->positions().view();
                auto rest1_pos    = r_rest_geo->positions().view();

                Transform l_transform(l_rest_geo->transforms().view()[0]);
                Transform r_transform(r_rest_geo->transforms().view()[0]);

                auto topo_view    = topo->view();
                auto mu_view      = mu_slot->view();
                auto lambda_view  = lambda_slot->view();
                auto thick_view   = thick_slot->view();
                auto min_sep_view = min_sep_slot->view();
                SizeT n           = topo_view.size();

                for(SizeT i = 0; i < n; ++i)
                {
                    const Vector3i& t = topo_view[i];
                    IndexT  v_id = t(0), e0 = t(1), e1 = t(2);
                    Vector3 x0 = l_transform * rest0_pos[v_id];
                    Vector3 x1 = r_transform * rest1_pos[e0];
                    Vector3 x2 = r_transform * rest1_pos[e1];

                    constexpr Float geo_degeneracy_tol = 1e-12;

                    Float   d    = min_sep_view[i];
                    Vector3 edge = x2 - x1;
                    Float   edge_len = edge.norm();
                    UIPC_ASSERT(edge_len >= geo_degeneracy_tol,
                                "SoftVertexEdgeStitch: edge ({},{}) is degenerate",
                                e0, e1);
                    Vector3 edge_dir = edge / edge_len;

                    // project vertex onto edge line to get closest point
                    Float   proj = edge_dir.dot(x0 - x1);
                    proj         = std::clamp(proj, Float(0), edge_len);
                    Vector3 closest = x1 + proj * edge_dir;
                    Vector3 ve_dir  = x0 - closest;
                    Float   ve_dist = ve_dir.norm();

                    if(ve_dist < d)
                    {
                        if(ve_dist < geo_degeneracy_tol)
                        {
                            // collinear: pick arbitrary perpendicular
                            Vector3 arbitrary =
                                std::abs(edge_dir(0)) < 0.9
                                    ? Vector3{1, 0, 0}
                                    : Vector3{0, 1, 0};
                            ve_dir = edge_dir.cross(arbitrary);
                            ve_dir.normalize();
                        }
                        else
                        {
                            ve_dir /= ve_dist;
                        }
                        x0 = closest + d * ve_dir;
                    }

                    // compute rest metric B and its inverse
                    Vector3   e01 = x1 - x0;
                    Vector3   e02 = x2 - x0;
                    Matrix2x2 B;
                    B(0, 0) = e01.dot(e01);
                    B(0, 1) = e01.dot(e02);
                    B(1, 0) = B(0, 1);
                    B(1, 1) = e02.dot(e02);
                    Matrix2x2 IB = B.inverse();

                    // rest area = 0.5 * |e01 x e02|
                    Float rest_area =
                        0.5 * e01.cross(e02).norm();

                    topo_buffer.push_back(Vector3i{t(0) + l_offset_v,
                                                   t(1) + r_offset_v,
                                                   t(2) + r_offset_v});
                    mu_buffer.push_back(mu_view[i]);
                    lambda_buffer.push_back(lambda_view[i]);
                    inv_B_buffer.push_back(IB);
                    rest_area_buffer.push_back(rest_area);
                    thickness_buffer.push_back(thick_view[i]);
                }
            });

        h_topos.assign(topo_buffer.begin(), topo_buffer.end());
        h_mus.assign(mu_buffer.begin(), mu_buffer.end());
        h_lambdas.assign(lambda_buffer.begin(), lambda_buffer.end());
        h_inv_Bs.assign(inv_B_buffer.begin(), inv_B_buffer.end());
        h_rest_areas.assign(rest_area_buffer.begin(), rest_area_buffer.end());
        h_thicknesses.assign(thickness_buffer.begin(), thickness_buffer.end());

        auto& device = engine().device();
        topos        = device.create_buffer<Vector3i>(h_topos.size());
        mus          = device.create_buffer<Float>(h_mus.size());
        lambdas      = device.create_buffer<Float>(h_lambdas.size());
        inv_Bs       = device.create_buffer<Matrix2x2>(h_inv_Bs.size());
        rest_areas   = device.create_buffer<Float>(h_rest_areas.size());
        thicknesses  = device.create_buffer<Float>(h_thicknesses.size());

        topos.copy_from(h_topos.data());
        mus.copy_from(h_mus.data());
        lambdas.copy_from(h_lambdas.data());
        inv_Bs.copy_from(h_inv_Bs.data());
        rest_areas.copy_from(h_rest_areas.data());
        thicknesses.copy_from(h_thicknesses.data());
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(topos.size());
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace NH = sym::soft_vertex_edge_stitch;

        energies = info.energies();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector3i> topos_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Float> mus_buf,
                                             BufferVar<Float> lambdas_buf,
                                             BufferVar<Matrix2x2> IBs_buf,
                                             BufferVar<Float> rest_areas_buf,
                                             BufferVar<Float> thicknesses_buf,
                                             BufferVar<Float> Es_buf,
                                             Float dt_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < topos_buf.size())
            {
                const Vector3i tri = topos_buf.read(I);
                Vector9 X;
                for(int k = 0; k < 3; ++k)
                    X.segment<3>(3 * k) = xs_buf.read(tri(k));

                Float E_val;
                NH::E(E_val, lambdas_buf.read(I), mus_buf.read(I), X, IBs_buf.read(I));

                Float Vdt2 = rest_areas_buf.read(I) * 2 * thicknesses_buf.read(I) * dt_val * dt_val;
                Es_buf.write(I, E_val * Vdt2);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(topos.view(),
                                    info.positions(),
                                    mus.view(),
                                    lambdas.view(),
                                    inv_Bs.view(),
                                    rest_areas.view(),
                                    thicknesses.view(),
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
        namespace NH = sym::soft_vertex_edge_stitch;

        gradients = info.gradients();
        hessians  = info.hessians();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Vector3i> topos_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Float> mus_buf,
                                                BufferVar<Float> lambdas_buf,
                                                BufferVar<Matrix2x2> IBs_buf,
                                                BufferVar<Float> rest_areas_buf,
                                                BufferVar<Float> thicknesses_buf,
                                                BufferVar<Float> G3s_buf,
                                                BufferVar<Matrix3x3> H3x3s_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < topos_buf.size())
            {
                const Vector3i tri = topos_buf.read(I);
                Vector9 X;
                for(int k = 0; k < 3; ++k)
                    X.segment<3>(3 * k) = xs_buf.read(tri(k));

                Float Vdt2 = rest_areas_buf.read(I) * 2 * thicknesses_buf.read(I) * dt_val * dt_val;

                Vector9 G;
                NH::dEdX(G, lambdas_buf.read(I), mus_buf.read(I), X, IBs_buf.read(I));
                G *= Vdt2;

                // Write gradient using atomic operations
                for(int k = 0; k < 3; ++k)
                {
                    Float3 gk = make_float3(G(3 * k + 0), G(3 * k + 1), G(3 * k + 2));
                    G3s_buf.atomic(tri(k)).x.fetch_add(gk.x);
                    G3s_buf.atomic(tri(k)).y.fetch_add(gk.y);
                    G3s_buf.atomic(tri(k)).z.fetch_add(gk.z);
                }

                $if(!gradient_only_val)
                {
                    Matrix9x9 H;
                    NH::ddEddX(H, lambdas_buf.read(I), mus_buf.read(I), X, IBs_buf.read(I));
                    make_spd(H);
                    H *= Vdt2;

                    // Write Hessian blocks
                    // H is 9x9, we need to write 3x3 blocks for each (i,j) pair
                    int block_idx = I * HalfHessianSize;
                    for(int i = 0; i < 3; ++i)
                    {
                        for(int j = i; j < 3; ++j)
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
                                    inv_Bs.view(),
                                    rest_areas.view(),
                                    thicknesses.view(),
                                    info.gradients().buffer(),
                                    info.hessians().buffer(),
                                    info.dt(),
                                    info.gradient_only())
                                .dispatch(topos.size());
    }
};

REGISTER_SIM_SYSTEM(SoftVertexEdgeStitch);
}  // namespace uipc::backend::luisa
