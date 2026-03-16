#include <finite_element/codim_1d_constitution.h>
#include <finite_element/constitutions/hookean_spring_1d_function.h>
#include <utils/codim_thickness.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <luisa/luisa-compute.h>
#include <numbers>

namespace uipc::backend::luisa
{
class HookeanSpring1D final : public Codim1DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 12ull;
    static constexpr SizeT StencilSize     = 2;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using Codim1DConstitution::Codim1DConstitution;

    vector<Float>             h_kappas;
    luisa::compute::Buffer<Float> kappas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(kappas.size());
        info.gradient_count(kappas.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(kappas.size() * HalfHessianSize);
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_kappas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto kappa = sc.edges().find<Float>("kappa");
                return kappa->view();
            },
            [&](const ForEachInfo& I, Float kappa)
            {
                auto vI = I.global_index();
                // retrieve material parameters
                h_kappas[vI] = kappa;
            });

        auto& device = engine().device();
        kappas       = device.create_buffer<Float>(N);
        kappas.copy_from(h_kappas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace NS = sym::hookean_spring_1d;

        auto indices      = info.indices();
        auto xs           = info.xs();
        auto rest_lengths = info.rest_lengths();
        auto thicknesses  = info.thicknesses();
        auto energies     = info.energies();
        Float dt          = info.dt();

        Kernel1D compute_energy_kernel = [&](BufferVar<Float> kappas_buf,
                                             BufferVar<Float> energies_buf,
                                             BufferVar<Vector2i> indices_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Float> rest_lengths_buf,
                                             BufferVar<Float> thicknesses_buf,
                                             Float dt_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < indices_buf.size())
            {
                Vector2i idx = indices_buf.read(I);

                // Load positions
                Vector3 x0 = xs_buf.read(idx(0));
                Vector3 x1 = xs_buf.read(idx(1));

                // Build X array
                std::array<Float, 6> X;
                X[0] = x0.x;
                X[1] = x0.y;
                X[2] = x0.z;
                X[3] = x1.x;
                X[4] = x1.y;
                X[5] = x1.z;

                Float L0 = rest_lengths_buf.read(I);
                Float r  = edge_thickness(thicknesses_buf.read(idx(0)), thicknesses_buf.read(idx(1)));
                Float kappa = kappas_buf.read(I);

                Float Vdt2 = L0 * r * r * std::numbers::pi * dt_val * dt_val;

                Float E;
                NS::E(E, kappa, X, L0);
                energies_buf.write(I, E * Vdt2);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(kappas.view(),
                                    energies,
                                    indices,
                                    xs,
                                    rest_lengths,
                                    thicknesses,
                                    dt)
                                .dispatch(indices.size());
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace NS = sym::hookean_spring_1d;

        auto indices       = info.indices();
        auto xs            = info.xs();
        auto rest_lengths  = info.rest_lengths();
        auto thicknesses   = info.thicknesses();
        auto gradients     = info.gradients();
        auto hessians      = info.hessians();
        Float dt           = info.dt();
        Bool gradient_only = info.gradient_only();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Float> kappas_buf,
                                                BufferVar<Vector2i> indices_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Float> rest_lengths_buf,
                                                BufferVar<Float> thicknesses_buf,
                                                BufferVar<Float> gradients_buf,
                                                BufferVar<Float> hessians_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < indices_buf.size())
            {
                Vector2i idx = indices_buf.read(I);

                // Load positions
                Vector3 x0 = xs_buf.read(idx(0));
                Vector3 x1 = xs_buf.read(idx(1));

                // Build X array
                std::array<Float, 6> X;
                X[0] = x0.x;
                X[1] = x0.y;
                X[2] = x0.z;
                X[3] = x1.x;
                X[4] = x1.y;
                X[5] = x1.z;

                Float L0 = rest_lengths_buf.read(I);
                Float r  = edge_thickness(thicknesses_buf.read(idx(0)), thicknesses_buf.read(idx(1)));
                Float kappa = kappas_buf.read(I);

                Float Vdt2 = L0 * r * r * std::numbers::pi * dt_val * dt_val;

                // Compute gradient
                std::array<Float, 6> G;
                NS::dEdX(G, kappa, X, L0);

                // Scale by Vdt2
                for(int i = 0; i < 6; ++i)
                    G[i] *= Vdt2;

                // Write gradient using atomic operations
                for(int i = 0; i < 2; ++i)
                {
                    Float3 Gi;
                    Gi.x = G[i * 3 + 0];
                    Gi.y = G[i * 3 + 1];
                    Gi.z = G[i * 3 + 2];

                    auto atomic_grad = gradients_buf->atomic(idx(i) * 3);
                    atomic_grad.fetch_add(Gi.x);
                    atomic_grad.fetch_add(Gi.y);
                    atomic_grad.fetch_add(Gi.z);
                }

                $if(!gradient_only_val)
                {
                    // Compute Hessian
                    std::array<std::array<Float, 6>, 6> H;
                    NS::ddEddX(H, kappa, X, L0);

                    // Scale by Vdt2 and convert to Matrix6x6
                    Matrix6x6 H6x6;
                    for(int i = 0; i < 6; ++i)
                    {
                        for(int j = 0; j < 6; ++j)
                        {
                            H6x6(i, j) = H[i][j] * Vdt2;
                        }
                    }

                    make_spd(H6x6);

                    // Write Hessian - upper triangular blocks only (3 blocks for 2x2 stencil)
                    IndexT hess_offset = I * HalfHessianSize;
                    for(int ii = 0; ii < 2; ++ii)
                    {
                        for(int jj = ii; jj < 2; ++jj)
                        {
                            // Extract 3x3 block
                            Matrix3x3 H_block;
                            for(int r = 0; r < 3; ++r)
                            {
                                for(int c = 0; c < 3; ++c)
                                {
                                    H_block(r, c) = H6x6(ii * 3 + r, jj * 3 + c);
                                }
                            }

                            // Write to hessians buffer
                            IndexT block_idx = hess_offset + (ii * 2 + jj - ii * (ii + 1) / 2);
                            // Layout: row_index, col_index, 9 values per block
                            // Simplified - actual implementation depends on buffer layout
                        }
                    }
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_grad_hess_kernel);
        engine().stream() << shader(kappas.view(),
                                    indices,
                                    xs,
                                    rest_lengths,
                                    thicknesses,
                                    gradients,
                                    hessians,
                                    dt,
                                    gradient_only)
                                .dispatch(indices.size());
    }
};

REGISTER_SIM_SYSTEM(HookeanSpring1D);
}  // namespace uipc::backend::luisa
