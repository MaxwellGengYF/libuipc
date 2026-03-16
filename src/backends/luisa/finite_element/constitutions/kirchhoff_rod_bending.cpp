#include <finite_element/finite_element_extra_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <finite_element/constitutions/kirchhoff_rod_bending_function.h>
#include <finite_element/matrix_utils.h>
#include <utils/matrix_assembler.h>
#include <luisa/luisa-compute.h>
#include <numbers>
#include <array>

namespace uipc::backend::luisa
{
class KirchhoffRodBending final : public FiniteElementExtraConstitution
{
    static constexpr U64   KirchhoffRodBendingUID = 15;
    static constexpr SizeT StencilSize            = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;
    using Base = FiniteElementExtraConstitution;

  public:
    using Base::Base;
    U64 get_uid() const noexcept override { return KirchhoffRodBendingUID; }

    vector<Vector3i> h_hinges;
    vector<Float>    h_bending_stiffness;

    luisa::compute::Buffer<Vector3i> hinges;
    luisa::compute::Buffer<Float>    bending_stiffnesses;

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        auto geo_slots    = world().scene().geometries();

        list<Vector3i> hinge_list;  // X0, X1, X2
        list<Float>    bending_stiffness_list;

        info.for_each(  //
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                unordered_map<IndexT, set<IndexT>> hinge_map;  // Vertex -> Connected Vertices

                auto vertex_offset =
                    sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
                auto vertex_offset_v = vertex_offset->view().front();

                auto edges = sc.edges().topo().view();

                for(auto e : edges)
                {
                    auto v0 = e[0];
                    auto v1 = e[1];

                    hinge_map[v0].insert(v1);
                    hinge_map[v1].insert(v0);
                }

                auto bending_stiffnesses = sc.vertices().find<Float>("bending_stiffness");
                UIPC_ASSERT(bending_stiffnesses, "Bending stiffness not found, why?");

                auto bs_view = bending_stiffnesses->view();

                for(auto& [v, connected] : hinge_map)
                {
                    auto bs = bs_view[v];

                    if(connected.size() < 2)  // Not a hinge
                        continue;

                    for(auto v1 : connected)
                        for(auto v2 : connected)
                        {
                            if(v1 >= v2)  // Avoid duplicate
                                continue;

                            hinge_list.push_back({vertex_offset_v + v1,
                                                  vertex_offset_v + v,  // center vertex
                                                  vertex_offset_v + v2});
                            bending_stiffness_list.push_back(bs);
                        }
                }
            });

        // Setup data
        h_hinges.resize(hinge_list.size());
        h_bending_stiffness.resize(hinge_list.size());
        std::ranges::move(hinge_list, h_hinges.begin());
        std::ranges::move(bending_stiffness_list, h_bending_stiffness.begin());

        // Copy to device
        auto& device = engine().device();

        hinges = device.create_buffer<Vector3i>(h_hinges.size());
        hinges.copy_from(h_hinges.data());

        bending_stiffnesses = device.create_buffer<Float>(h_bending_stiffness.size());
        bending_stiffnesses.copy_from(h_bending_stiffness.data());
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(hinges.size());                 // Each hinge has 1 energy
        info.gradient_count(hinges.size() * StencilSize);  // Each hinge has 3 vertices

        if(info.gradient_only())
            return;

        info.hessian_count(hinges.size() * HalfHessianSize);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace KRB = sym::kirchhoff_rod_bending;

        constexpr Float Pi = std::numbers::pi;

        auto hinges_view            = hinges.view();
        auto bending_stiffnesses_view = bending_stiffnesses.view();
        auto thicknesses            = info.thicknesses();
        auto xs                     = info.xs();
        auto x_bars                 = info.x_bars();
        auto energies               = info.energies();
        Float dt                    = info.dt();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector3i> hinges_buf,
                                             BufferVar<Float> bending_stiffnesses_buf,
                                             BufferVar<Float> thicknesses_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Vector3> x_bars_buf,
                                             BufferVar<Float> energies_buf,
                                             Float dt_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < hinges_buf.size())
            {
                Vector3i hinge = hinges_buf.read(I);
                Float    k     = bending_stiffnesses_buf.read(I) * dt_val * dt_val;
                Float    r     = thicknesses_buf.read(I);

                Vector3 x0 = xs_buf.read(hinge[0]);
                Vector3 x1 = xs_buf.read(hinge[1]);
                Vector3 x2 = xs_buf.read(hinge[2]);

                Vector3 x0_bar = x_bars_buf.read(hinge[0]);
                Vector3 x1_bar = x_bars_buf.read(hinge[1]);
                Vector3 x2_bar = x_bars_buf.read(hinge[2]);

                // Rest length of the two edges
                Float L0 = length(x1_bar - x0_bar) + length(x2_bar - x1_bar);

                std::array<Float, 9> X;
                X[0] = x0.x;
                X[1] = x0.y;
                X[2] = x0.z;
                X[3] = x1.x;
                X[4] = x1.y;
                X[5] = x1.z;
                X[6] = x2.x;
                X[7] = x2.y;
                X[8] = x2.z;

                Float E_val;
                KRB::E(E_val, k, X, L0, r, Pi);

                energies_buf.write(I, E_val);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(hinges_view,
                                    bending_stiffnesses_view,
                                    thicknesses,
                                    xs,
                                    x_bars,
                                    energies,
                                    dt)
                                .dispatch(hinges_view.size());
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace KRB = sym::kirchhoff_rod_bending;

        constexpr Float Pi = std::numbers::pi;

        auto hinges_view            = hinges.view();
        auto bending_stiffnesses_view = bending_stiffnesses.view();
        auto thicknesses            = info.thicknesses();
        auto xs                     = info.xs();
        auto x_bars                 = info.x_bars();
        auto gradients              = info.gradients();
        auto hessians               = info.hessians();
        Float dt                    = info.dt();
        Bool gradient_only          = info.gradient_only();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Vector3i> hinges_buf,
                                                BufferVar<Float> bending_stiffnesses_buf,
                                                BufferVar<Float> thicknesses_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Vector3> x_bars_buf,
                                                BufferVar<Float> gradients_buf,
                                                BufferVar<Float> hessians_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < hinges_buf.size())
            {
                Vector3i hinge = hinges_buf.read(I);
                Float    k     = bending_stiffnesses_buf.read(I);
                Float    r     = thicknesses_buf.read(I);

                Vector3 x0 = xs_buf.read(hinge[0]);
                Vector3 x1 = xs_buf.read(hinge[1]);
                Vector3 x2 = xs_buf.read(hinge[2]);

                Vector3 x0_bar = x_bars_buf.read(hinge[0]);
                Vector3 x1_bar = x_bars_buf.read(hinge[1]);
                Vector3 x2_bar = x_bars_buf.read(hinge[2]);

                // Rest length of the two edges
                Float L0 = length(x1_bar - x0_bar) + length(x2_bar - x1_bar);

                std::array<Float, 9> X;
                X[0] = x0.x;
                X[1] = x0.y;
                X[2] = x0.z;
                X[3] = x1.x;
                X[4] = x1.y;
                X[5] = x1.z;
                X[6] = x2.x;
                X[7] = x2.y;
                X[8] = x2.z;

                Float dt2 = dt_val * dt_val;

                std::array<Float, 9> G;
                KRB::dEdX(G, k, X, L0, r, Pi);
                for(int i = 0; i < 9; ++i)
                    G[i] *= dt2;

                // Write gradient using DoubletVectorAssembler
                for(int i = 0; i < 3; ++i)
                {
                    for(int j = 0; j < 3; ++j)
                    {
                        auto atomic_grad = gradients_buf->atomic(hinge[i] * 3 + j);
                        atomic_grad.fetch_add(G[i * 3 + j]);
                    }
                }

                $if(!gradient_only_val)
                {
                    std::array<std::array<Float, 9>, 9> H;
                    KRB::ddEddX(H, k, X, L0, r, Pi);

                    for(int i = 0; i < 9; ++i)
                        for(int j = 0; j < 9; ++j)
                            H[i][j] *= dt2;
                    
                    // Make Hessian positive semi-definite
                    H = clamp_to_spd(H);

                    // Write Hessian using TripletMatrixAssembler - half block only
                    IndexT hess_offset = I * HalfHessianSize;
                    
                    // For 3 vertices, we have 6 upper triangular blocks (3 diagonal + 3 off-diagonal)
                    // Block indices: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
                    IndexT block_idx = hess_offset;
                    
                    for(int ii = 0; ii < 3; ++ii)
                    {
                        for(int jj = ii; jj < 3; ++jj)
                        {
                            // Extract 3x3 block from the 9x9 matrix
                            for(int r = 0; r < 3; ++r)
                            {
                                for(int c = 0; c < 3; ++c)
                                {
                                    Float val = H[ii * 3 + r][jj * 3 + c];
                                    hessians_buf->write(block_idx * 9 + r * 3 + c, val);
                                }
                            }
                            block_idx++;
                        }
                    }
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_grad_hess_kernel);
        engine().stream() << shader(hinges_view,
                                    bending_stiffnesses_view,
                                    thicknesses,
                                    xs,
                                    x_bars,
                                    gradients,
                                    hessians,
                                    dt,
                                    gradient_only)
                                .dispatch(hinges_view.size());
    }
};

REGISTER_SIM_SYSTEM(KirchhoffRodBending);
}  // namespace uipc::backend::luisa
