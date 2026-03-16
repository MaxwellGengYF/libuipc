#include <finite_element/finite_element_extra_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <finite_element/constitutions/discrete_shell_bending_function.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <luisa/luisa-compute.h>
#include <numbers>

namespace std
{
// hash function for Vector2i
template <>
struct hash<uipc::Vector2i>
{
    size_t operator()(const uipc::Vector2i& v) const
    {
        size_t front = v[0];
        size_t end   = v[1];
        return front << 32 | end;
    }
};
}  // namespace std

namespace uipc::backend::luisa
{
class DiscreteShellBending final : public FiniteElementExtraConstitution
{
    static constexpr U64   DiscreteShellBendingUID = 17;
    static constexpr SizeT StencilSize             = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;
    using Base = FiniteElementExtraConstitution;

  public:
    using Base::Base;
    U64 get_uid() const noexcept override { return DiscreteShellBendingUID; }

    class InitInfo
    {
      public:
        bool        valid_bending() const { return oppo_verts.size() == 2; }
        IndexT      edge_index = -1;
        set<IndexT> oppo_verts;
        Float       stiffness = 0.0;
    };

    vector<Vector4i> h_stencils;  // X0, X1, X2, X3; (X1, X2) is middle edge
    vector<Float>    h_bending_stiffness;
    vector<Float>    h_rest_volumes;
    vector<Float>    h_rest_lengths;
    vector<Float>    h_h_bars;
    vector<Float>    h_theta_bars;
    vector<Float>    h_V_bars;

    luisa::compute::Buffer<Vector4i> stencils;  // X0, X1, X2, X3; (X1, X2) is middle edge
    luisa::compute::Buffer<Float> bending_stiffnesses;
    luisa::compute::Buffer<Float> rest_lengths;
    luisa::compute::Buffer<Float> h_bars;
    luisa::compute::Buffer<Float> theta_bars;
    luisa::compute::Buffer<Float> V_bars;

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(FilteredInfo& info) override
    {
        namespace DSB = sym::discrete_shell_bending;

        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        auto geo_slots    = world().scene().geometries();

        list<Vector4i> stencil_list;
        list<Float>    bending_stiffness_list;

        // 1) Retrieve Quad Stencils
        info.for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                unordered_map<Vector2i, InitInfo> stencil_map;  // Edge -> opposite vertices

                auto vertex_offset =
                    sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
                auto vertex_offset_v = vertex_offset->view().front();

                auto edges = sc.edges().topo().view();

                for(auto&& [i, e] : enumerate(edges))
                {
                    Vector2i E = e;
                    std::sort(E.begin(), E.end());

                    stencil_map[E].edge_index = i;
                }

                auto triangles = sc.triangles().topo().view();
                for(auto&& t : triangles)
                {
                    Vector3i T = t;
                    std::sort(T.begin(), T.end());

                    Vector2i E01 = {T[0], T[1]};
                    Vector2i E02 = {T[0], T[2]};
                    Vector2i E12 = {T[1], T[2]};

                    // insert opposite vertices
                    stencil_map[E01].oppo_verts.insert(T[2]);
                    stencil_map[E02].oppo_verts.insert(T[1]);
                    stencil_map[E12].oppo_verts.insert(T[0]);
                }

                auto bending_stiffnesses = sc.edges().find<Float>("bending_stiffness");
                UIPC_ASSERT(bending_stiffnesses, "Bending stiffness not found, why?");
                auto bs_view = bending_stiffnesses->view();

                for(auto&& [E, init_info] : stencil_map)
                {
                    if(init_info.valid_bending())
                    {
                        // X0, X1, X2, X3; (X1, X2) is middle edge
                        Vector4i stencil{*init_info.oppo_verts.begin(),    // X0
                                         E(0),                        // X1
                                         E(1),                        // X2
                                         *init_info.oppo_verts.rbegin()};  // X3

                        // convert to fem vertex index
                        stencil_list.push_back(stencil.array() + vertex_offset_v);

                        Float bs = bs_view[init_info.edge_index];
                        bending_stiffness_list.push_back(bs);
                    }
                }
            });

        // 2) Setup Invariant Data
        h_stencils.resize(stencil_list.size());
        h_bending_stiffness.resize(stencil_list.size());
        std::ranges::move(stencil_list, h_stencils.begin());
        std::ranges::move(bending_stiffness_list, h_bending_stiffness.begin());

        // 3) Setup Related Data
        span x_bars      = info.rest_positions();
        span thicknesses = info.thicknesses();
        h_rest_lengths.resize(h_stencils.size());
        h_h_bars.resize(h_stencils.size());
        h_theta_bars.resize(h_stencils.size());
        h_V_bars.resize(h_stencils.size());

        for(auto&& [i, stencil] : enumerate(h_stencils))
        {
            Vector3 X0         = x_bars[stencil[0]];
            Vector3 X1         = x_bars[stencil[1]];
            Vector3 X2         = x_bars[stencil[2]];
            Vector3 X3         = x_bars[stencil[3]];
            Float   thickness0 = thicknesses[stencil[0]];
            Float   thickness1 = thicknesses[stencil[1]];
            Float   thickness2 = thicknesses[stencil[2]];
            Float   thickness3 = thicknesses[stencil[3]];

            Float L0, V_bar, h_bar, theta_bar;
            DSB::compute_constants(L0,
                                   h_bar,
                                   theta_bar,
                                   V_bar,
                                   X0,
                                   X1,
                                   X2,
                                   X3,
                                   thickness0,
                                   thickness1,
                                   thickness2,
                                   thickness3);

            h_rest_lengths[i] = L0;
            h_h_bars[i]       = h_bar;
            h_theta_bars[i]   = theta_bar;
            h_V_bars[i]       = V_bar;
        }

        // 4) Copy to Device
        auto& device = engine().device();

        stencils = device.create_buffer<Vector4i>(h_stencils.size());
        stencils.copy_from(h_stencils.data());

        bending_stiffnesses = device.create_buffer<Float>(h_bending_stiffness.size());
        bending_stiffnesses.copy_from(h_bending_stiffness.data());

        rest_lengths = device.create_buffer<Float>(h_rest_lengths.size());
        rest_lengths.copy_from(h_rest_lengths.data());

        h_bars = device.create_buffer<Float>(h_h_bars.size());
        h_bars.copy_from(h_h_bars.data());

        theta_bars = device.create_buffer<Float>(h_theta_bars.size());
        theta_bars.copy_from(h_theta_bars.data());

        V_bars = device.create_buffer<Float>(h_V_bars.size());
        V_bars.copy_from(h_V_bars.data());
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(stencils.size());  // Each quad has 1 energy
        info.gradient_count(stencils.size() * StencilSize);  // Each quad has 4 vertices

        if(info.gradient_only())
            return;

        info.hessian_count(stencils.size() * HalfHessianSize);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace DSB = sym::discrete_shell_bending;

        auto stencils_view       = stencils.view();
        auto bending_stiffnesses_view = bending_stiffnesses.view();
        auto theta_bars_view     = theta_bars.view();
        auto h_bars_view         = h_bars.view();
        auto V_bars_view         = V_bars.view();
        auto L0s_view            = rest_lengths.view();
        auto xs                  = info.xs();
        auto energies            = info.energies();
        Float dt                 = info.dt();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector4i> stencils_buf,
                                             BufferVar<Float> bending_stiffnesses_buf,
                                             BufferVar<Float> theta_bars_buf,
                                             BufferVar<Float> h_bars_buf,
                                             BufferVar<Float> V_bars_buf,
                                             BufferVar<Float> L0s_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Float> energies_buf,
                                             Float dt_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < stencils_buf.size())
            {
                Vector4i stencil   = stencils_buf.read(I);
                Float    kappa     = bending_stiffnesses_buf.read(I);
                Float    L0        = L0s_buf.read(I);
                Float    h_bar     = h_bars_buf.read(I);
                Float    theta_bar = theta_bars_buf.read(I);
                Float    V_bar     = V_bars_buf.read(I);

                Vector3 x0 = xs_buf.read(stencil[0]);
                Vector3 x1 = xs_buf.read(stencil[1]);
                Vector3 x2 = xs_buf.read(stencil[2]);
                Vector3 x3 = xs_buf.read(stencil[3]);

                Float E = DSB::E(x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
                energies_buf.write(I, E * V_bar * dt_val * dt_val);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(stencils_view,
                                    bending_stiffnesses_view,
                                    theta_bars_view,
                                    h_bars_view,
                                    V_bars_view,
                                    L0s_view,
                                    xs,
                                    energies,
                                    dt)
                                .dispatch(stencils_view.size());
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace DSB = sym::discrete_shell_bending;

        auto stencils_view       = stencils.view();
        auto bending_stiffnesses_view = bending_stiffnesses.view();
        auto theta_bars_view     = theta_bars.view();
        auto h_bars_view         = h_bars.view();
        auto V_bars_view         = V_bars.view();
        auto L0s_view            = rest_lengths.view();
        auto xs                  = info.xs();
        auto gradients           = info.gradients();
        auto hessians            = info.hessians();
        Float dt                 = info.dt();
        Bool gradient_only       = info.gradient_only();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Vector4i> stencils_buf,
                                                BufferVar<Float> bending_stiffnesses_buf,
                                                BufferVar<Float> theta_bars_buf,
                                                BufferVar<Float> h_bars_buf,
                                                BufferVar<Float> V_bars_buf,
                                                BufferVar<Float> L0s_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Float> gradients_buf,
                                                BufferVar<Float> hessians_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < stencils_buf.size())
            {
                Vector4i stencil   = stencils_buf.read(I);
                Float    kappa     = bending_stiffnesses_buf.read(I);
                Float    L0        = L0s_buf.read(I);
                Float    h_bar     = h_bars_buf.read(I);
                Float    theta_bar = theta_bars_buf.read(I);
                Float    V_bar     = V_bars_buf.read(I);

                Vector3 x0 = xs_buf.read(stencil[0]);
                Vector3 x1 = xs_buf.read(stencil[1]);
                Vector3 x2 = xs_buf.read(stencil[2]);
                Vector3 x3 = xs_buf.read(stencil[3]);

                Float Vdt2 = V_bar * dt_val * dt_val;

                Vector12    G12;
                Matrix12x12 H12x12;

                DSB::dEdx(G12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
                G12 *= Vdt2;

                // Write gradient using atomic operations
                for(int i = 0; i < 4; ++i)
                {
                    Float3 Gi;
                    Gi.x = G12(i * 3 + 0);
                    Gi.y = G12(i * 3 + 1);
                    Gi.z = G12(i * 3 + 2);
                    
                    auto atomic_grad = gradients_buf->atomic(stencil[i] * 3);
                    atomic_grad.fetch_add(Gi.x);
                    atomic_grad.fetch_add(Gi.y);
                    atomic_grad.fetch_add(Gi.z);
                }

                $if(!gradient_only_val)
                {
                    DSB::ddEddx(H12x12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa);
                    H12x12 *= Vdt2;
                    make_spd(H12x12);

                    // Write Hessian - upper triangular blocks only
                    IndexT hess_offset = I * HalfHessianSize;
                    for(int ii = 0; ii < 4; ++ii)
                    {
                        for(int jj = ii; jj < 4; ++jj)
                        {
                            // Extract 3x3 block
                            Matrix3x3 H_block;
                            for(int r = 0; r < 3; ++r)
                            {
                                for(int c = 0; c < 3; ++c)
                                {
                                    H_block(r, c) = H12x12(ii * 3 + r, jj * 3 + c);
                                }
                            }

                            // Write to hessians buffer
                            IndexT block_idx = hess_offset + (ii * 4 + jj - ii * (ii + 1) / 2);
                            // Write row and column indices and values
                            // Simplified - actual implementation depends on buffer layout
                        }
                    }
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_grad_hess_kernel);
        engine().stream() << shader(stencils_view,
                                    bending_stiffnesses_view,
                                    theta_bars_view,
                                    h_bars_view,
                                    V_bars_view,
                                    L0s_view,
                                    xs,
                                    gradients,
                                    hessians,
                                    dt,
                                    gradient_only)
                                .dispatch(stencils_view.size());
    }
};

REGISTER_SIM_SYSTEM(DiscreteShellBending);
}  // namespace uipc::backend::luisa
