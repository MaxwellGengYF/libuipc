#include <inter_primitive_effect_system/inter_primitive_constitution.h>
#include <inter_primitive_effect_system/constitutions/soft_vertex_stitch_function.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class SoftVertexStitch : public InterPrimitiveConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 22;
    static constexpr SizeT StencilSize     = 2;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using InterPrimitiveConstitution::InterPrimitiveConstitution;

    U64 get_uid() const noexcept override { return ConstitutionUID; }

    void do_build(BuildInfo& info) override {}

    luisa::compute::Buffer<Vector2i> topos;
    luisa::compute::Buffer<Float>    kappas;
    luisa::compute::Buffer<Float>    rest_lengths;

    vector<Vector2i> h_topos;
    vector<Float>    h_kappas;
    vector<Float>    h_rest_lengths;

    luisa::compute::BufferView<const Float>              energies;
    MutableDoubletVectorView<Float, 3>    gradients;
    MutableTripletMatrixView<Float, 3, 3> hessians;

    void do_init(FilteredInfo& info) override
    {
        list<Vector2i> topo_buffer;
        list<Float>    kappa_buffer;
        list<Float>    rest_length_buffer;

        auto geo_slots    = world().scene().geometries();
        using ForEachInfo = InterPrimitiveConstitutionManager::ForEachInfo;
        info.for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::Geometry& geo)
            {
                auto topo = geo.instances().find<Vector2i>(builtin::topo);
                UIPC_ASSERT(topo, "SoftVertexStitch requires attribute `topo` on instances()");
                auto geo_ids = geo.meta().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids, "SoftVertexStitch requires attribute `geo_ids` on meta()");
                auto kappa = geo.instances().find<Float>("kappa");
                UIPC_ASSERT(kappa, "SoftVertexStitch requires attribute `kappa` on instances()");
                auto rest_length = geo.instances().find<Float>("rest_length");
                UIPC_ASSERT(rest_length, "SoftVertexStitch requires attribute `rest_length` on instances()");
                Vector2i ids = geo_ids->view()[0];

                auto l_slot = info.geo_slot(ids[0]);
                auto r_slot = info.geo_slot(ids[1]);

                auto l_geo = l_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(l_geo,
                            "SoftVertexStitch requires simplicial complex geometry, but yours {} ({})",
                            l_slot->geometry().type(),
                            l_slot->id());
                auto r_geo = r_slot->geometry().as<geometry::SimplicialComplex>();
                UIPC_ASSERT(r_geo,
                            "SoftVertexStitch requires simplicial complex geometry, but yours {} ({})",
                            r_slot->geometry().type(),
                            r_slot->id());

                auto l_offset = l_geo->meta().find<IndexT>(builtin::global_vertex_offset);
                UIPC_ASSERT(l_offset,
                            "SoftVertexStitch requires attribute `global_vertex_offset` on meta() of geometry {} ({})",
                            l_slot->geometry().type(),
                            l_slot->id());
                IndexT l_offset_v = l_offset->view()[0];
                auto r_offset = r_geo->meta().find<IndexT>(builtin::global_vertex_offset);
                UIPC_ASSERT(r_offset,
                            "SoftVertexStitch requires attribute `global_vertex_offset` on meta() of geometry {} ({})",
                            r_slot->geometry().type(),
                            r_slot->id());
                IndexT r_offset_v = r_offset->view()[0];

                auto topo_view = topo->view();
                for(auto& v : topo_view)
                {
                    topo_buffer.push_back(Vector2i{v[0] + l_offset_v, v[1] + r_offset_v});
                }

                auto kappa_view = kappa->view();
                for(auto kappa : kappa_view)
                {
                    kappa_buffer.push_back(kappa);
                }

                auto rest_length_view = rest_length->view();
                for(auto rl : rest_length_view)
                {
                    UIPC_ASSERT(rl >= 0, "rest_length must be non-negative");
                    rest_length_buffer.push_back(rl);
                }
            });

        h_topos.resize(topo_buffer.size());
        std::ranges::move(topo_buffer, h_topos.begin());

        h_kappas.resize(kappa_buffer.size());
        std::ranges::move(kappa_buffer, h_kappas.begin());

        h_rest_lengths.resize(rest_length_buffer.size());
        std::ranges::move(rest_length_buffer, h_rest_lengths.begin());

        auto& device = engine().device();
        topos        = device.create_buffer<Vector2i>(h_topos.size());
        kappas       = device.create_buffer<Float>(h_kappas.size());
        rest_lengths = device.create_buffer<Float>(h_rest_lengths.size());

        topos.copy_from(h_topos.data());
        kappas.copy_from(h_kappas.data());
        rest_lengths.copy_from(h_rest_lengths.data());
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(topos.size());
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        energies = info.energies();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector2i> topos_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Float> kappas_buf,
                                             BufferVar<Float> rest_lengths_buf,
                                             BufferVar<Float> Es_buf,
                                             Float dt_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < topos_buf.size())
            {
                const Vector2i PP  = topos_buf.read(I);
                Float          Kt2 = kappas_buf.read(I) * dt_val * dt_val;
                Float          L0  = rest_lengths_buf.read(I);
                Vector3        x0  = xs_buf.read(PP[0]);
                Vector3        x1  = xs_buf.read(PP[1]);
                Vector3        dx  = x0 - x1;
                $if(L0 == 0.0)
                {
                    // Harmonic energy: E = 0.5 * k * ||dx||^2
                    Es_buf.write(I, 0.5 * Kt2 * dx.squaredNorm());
                }
                $else
                {
                    Float dist = dx.norm();
                    Float diff = dist - L0;
                    Es_buf.write(I, 0.5 * Kt2 * diff * diff);
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(topos.view(),
                                    info.positions(),
                                    kappas.view(),
                                    rest_lengths.view(),
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
        namespace SVS = sym::soft_vertex_stitch;

        gradients = info.gradients();
        hessians  = info.hessians();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Vector2i> topos_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Float> kappas_buf,
                                                BufferVar<Float> rest_lengths_buf,
                                                BufferVar<Float> G3s_buf,
                                                BufferVar<Matrix3x3> H3x3s_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < topos_buf.size())
            {
                const Vector2i PP = topos_buf.read(I);
                Vector6 X;
                X.segment<3>(0) = xs_buf.read(PP[0]);
                X.segment<3>(3) = xs_buf.read(PP[1]);

                Float   Kt2 = kappas_buf.read(I) * dt_val * dt_val;
                Float   L0  = rest_lengths_buf.read(I);
                Vector3 dx  = X.head<3>() - X.tail<3>();

                Vector6   G;
                Matrix6x6 H;

                $if(L0 == 0.0)
                {
                    // Harmonic energy: E = 0.5 * k * ||dx||^2
                    // G = k * [dx; -dx],  H = k * [[I,-I],[-I,I]]
                    G.head<3>() = Kt2 * dx;
                    G.tail<3>() = -Kt2 * dx;

                    $if(!gradient_only_val)
                    {
                        Matrix3x3 blk = Kt2 * Matrix3x3::Identity();
                        H.block<3, 3>(0, 0) = blk;
                        H.block<3, 3>(0, 3) = -blk;
                        H.block<3, 3>(3, 0) = -blk;
                        H.block<3, 3>(3, 3) = blk;
                    };
                }
                $else
                {
                    SVS::dEdX(G, Kt2, X, L0);
                    $if(!gradient_only_val)
                    {
                        SVS::ddEddX(H, Kt2, X, L0);
                    };
                };

                // Write gradient using atomic operations
                Float3 g0 = make_float3(G(0), G(1), G(2));
                Float3 g1 = make_float3(G(3), G(4), G(5));
                G3s_buf.atomic(PP[0]).x.fetch_add(g0.x);
                G3s_buf.atomic(PP[0]).y.fetch_add(g0.y);
                G3s_buf.atomic(PP[0]).z.fetch_add(g0.z);
                G3s_buf.atomic(PP[1]).x.fetch_add(g1.x);
                G3s_buf.atomic(PP[1]).y.fetch_add(g1.y);
                G3s_buf.atomic(PP[1]).z.fetch_add(g1.z);

                $if(!gradient_only_val)
                {
                    make_spd(H);

                    // Write Hessian blocks (2x2 block structure, only upper triangle)
                    // H is 6x6, we write 3x3 blocks
                    // Block (0,0)
                    Matrix3x3 H_00;
                    for(int r = 0; r < 3; ++r)
                        for(int c = 0; c < 3; ++c)
                            H_00(r, c) = H(r, c);
                    H3x3s_buf.write(I * HalfHessianSize + 0, H_00);

                    // Block (0,1)
                    Matrix3x3 H_01;
                    for(int r = 0; r < 3; ++r)
                        for(int c = 0; c < 3; ++c)
                            H_01(r, c) = H(r, 3 + c);
                    H3x3s_buf.write(I * HalfHessianSize + 1, H_01);

                    // Block (1,1)
                    Matrix3x3 H_11;
                    for(int r = 0; r < 3; ++r)
                        for(int c = 0; c < 3; ++c)
                            H_11(r, c) = H(3 + r, 3 + c);
                    H3x3s_buf.write(I * HalfHessianSize + 2, H_11);
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_grad_hess_kernel);
        engine().stream() << shader(topos.view(),
                                    info.positions(),
                                    kappas.view(),
                                    rest_lengths.view(),
                                    info.gradients().buffer(),
                                    info.hessians().buffer(),
                                    info.dt(),
                                    info.gradient_only())
                                .dispatch(topos.size());
    }
};

REGISTER_SIM_SYSTEM(SoftVertexStitch);
}  // namespace uipc::backend::luisa
