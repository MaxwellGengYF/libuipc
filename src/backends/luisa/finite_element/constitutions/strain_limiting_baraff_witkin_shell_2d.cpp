#include <finite_element/codim_2d_constitution.h>
#include <finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.h>
#include <utils/codim_thickness.h>
#include <finite_element/matrix_utils.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class StrainLimitingBaraffWitkinShell2D final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 819;
    static constexpr SizeT StencilSize     = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using Codim2DConstitution::Codim2DConstitution;

    vector<Float> h_mus;
    vector<Float> h_lambdas;
    vector<Float> h_strain_rates;

    luisa::compute::Buffer<Float>     mus;
    luisa::compute::Buffer<Float>     lambdas;
    luisa::compute::Buffer<Float>     strain_rates;
    luisa::compute::Buffer<Matrix2x2> inv_B_matrices;

    SimSystemSlot<FiniteElementMethod> fem;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override
    {
        fem = require<FiniteElementMethod>();
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        namespace BWS = sym::strainlimiting_baraff_witkin_shell_2d;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_mus.resize(N);
        h_lambdas.resize(N);
        h_strain_rates.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto lambda = sc.triangles().find<Float>("lambda");
                auto mu     = sc.triangles().find<Float>("mu");

                return zip(lambda->view(), mu->view());
            },
            [&](const ForEachInfo& I, auto lambda_mu)
            {
                auto vI = I.global_index();

                auto&& [lambda, mu] = lambda_mu;

                h_lambdas[vI]      = lambda;
                h_mus[vI]          = mu;
                h_strain_rates[vI] = 100;
            });

        auto& device = engine().device();
        mus          = device.create_buffer<Float>(N);
        mus.copy_from(h_mus.data());

        lambdas = device.create_buffer<Float>(N);
        lambdas.copy_from(h_lambdas.data());

        strain_rates = device.create_buffer<Float>(N);
        strain_rates.copy_from(h_strain_rates.data());

        auto& cinfo       = info.constitution_info();
        auto  prim_offset = cinfo.primitive_offset;
        auto  prim_count  = cinfo.primitive_count;

        auto prims  = fem->codim_2ds().subview(prim_offset, prim_count);
        auto x_bars = fem->x_bars();

        inv_B_matrices = device.create_buffer<Matrix2x2>(N);

        // Precompute inverse of rest shape matrix for each triangle
        Kernel1D precompute_inv_B_kernel = [&](BufferVar<Vector3i> prims_buf,
                                                BufferVar<Vector3> x_bars_buf,
                                                BufferVar<Matrix2x2> inv_Bs_buf) noexcept
        {
            auto I = dispatch_x();
            $if(I < prims_buf.size())
            {
                Vector3i tri = prims_buf.read(I);
                Vector3 x0   = x_bars_buf.read(tri(0));
                Vector3 x1   = x_bars_buf.read(tri(1));
                Vector3 x2   = x_bars_buf.read(tri(2));

                BWS::Float2x2 Dm = BWS::Dm2x2(x0, x1, x2);
                
                // Compute inverse of Dm
                float det = Dm[0][0] * Dm[1][1] - Dm[0][1] * Dm[1][0];
                float inv_det = 1.0f / det;
                
                Matrix2x2 inv_B;
                inv_B[0][0] =  Dm[1][1] * inv_det;
                inv_B[0][1] = -Dm[0][1] * inv_det;
                inv_B[1][0] = -Dm[1][0] * inv_det;
                inv_B[1][1] =  Dm[0][0] * inv_det;
                
                inv_Bs_buf.write(I, inv_B);
            };
        };

        auto shader = device.compile(precompute_inv_B_kernel);
        engine().stream() << shader(prims, x_bars, inv_B_matrices).dispatch(prims.size());
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(h_mus.size());
        info.gradient_count(h_mus.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(h_mus.size() * HalfHessianSize);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace BWS = sym::strainlimiting_baraff_witkin_shell_2d;

        auto indices     = info.indices();
        auto xs          = info.xs();
        auto rest_areas  = info.rest_areas();
        auto thicknesses = info.thicknesses();
        auto energies    = info.energies();
        Float dt         = info.dt();

        Kernel1D compute_energy_kernel = [&](BufferVar<Float> mus_buf,
                                             BufferVar<Float> lambdas_buf,
                                             BufferVar<Float> strain_rates_buf,
                                             BufferVar<Float> rest_areas_buf,
                                             BufferVar<Float> thicknesses_buf,
                                             BufferVar<Float> energies_buf,
                                             BufferVar<Vector3i> indices_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Matrix2x2> IBs_buf,
                                             Float dt_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < indices_buf.size())
            {
                Vector3i idx = indices_buf.read(I);
                
                // Build X vector (9 elements for 3 vertices)
                BWS::Float9 X;
                for(int i = 0; i < 3; ++i)
                {
                    Vector3 xi = xs_buf.read(idx(i));
                    X[i * 3 + 0] = xi.x;
                    X[i * 3 + 1] = xi.y;
                    X[i * 3 + 2] = xi.z;
                }

                Matrix2x2 IB = IBs_buf.read(I);
                BWS::Float2x2 IB_custom;
                IB_custom[0] = Float2(IB[0][0], IB[0][1]);
                IB_custom[1] = Float2(IB[1][0], IB[1][1]);

                Float lambda      = lambdas_buf.read(I);
                Float mu          = mus_buf.read(I);
                Float strain_rate = strain_rates_buf.read(I);
                Float rest_area   = rest_areas_buf.read(I);

                Float thickness = triangle_thickness(thicknesses_buf.read(idx(0)),
                                                     thicknesses_buf.read(idx(1)),
                                                     thicknesses_buf.read(idx(2)));

                // Compute Ds and F
                BWS::Float3x2 Ds = BWS::Ds3x2(
                    Float3(X[0], X[1], X[2]),
                    Float3(X[3], X[4], X[5]),
                    Float3(X[6], X[7], X[8])
                );
                BWS::Float3x2 F = Ds * IB_custom;

                Float2 anisotropic_a(1.0f, 0.0f);
                Float2 anisotropic_b(0.0f, 1.0f);

                // thickness is onesided, so the Volume is area * thickness * 2
                Float V = rest_area * thickness * 2.0f;

                Float E = BWS::E(F, anisotropic_a, anisotropic_b, lambda, mu, strain_rate);
                energies_buf.write(I, E * V * dt_val * dt_val);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(mus.view(),
                                    lambdas.view(),
                                    strain_rates.view(),
                                    rest_areas,
                                    thicknesses,
                                    energies,
                                    indices,
                                    xs,
                                    inv_B_matrices,
                                    dt)
                                .dispatch(indices.size());
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace BWS = sym::strainlimiting_baraff_witkin_shell_2d;

        auto indices       = info.indices();
        auto xs            = info.xs();
        auto thicknesses   = info.thicknesses();
        auto gradients     = info.gradients();
        auto hessians      = info.hessians();
        auto rest_areas    = info.rest_areas();
        Float dt           = info.dt();
        Bool gradient_only = info.gradient_only();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Float> mus_buf,
                                                BufferVar<Float> lambdas_buf,
                                                BufferVar<Float> strain_rates_buf,
                                                BufferVar<Vector3i> indices_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Float> thicknesses_buf,
                                                BufferVar<Float> gradients_buf,
                                                BufferVar<Matrix3x3> hessians_buf,
                                                BufferVar<Float> rest_areas_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < indices_buf.size())
            {
                Vector3i idx = indices_buf.read(I);
                
                // Build X vector (9 elements for 3 vertices)
                BWS::Float9 X;
                for(int i = 0; i < 3; ++i)
                {
                    Vector3 xi = xs_buf.read(idx(i));
                    X[i * 3 + 0] = xi.x;
                    X[i * 3 + 1] = xi.y;
                    X[i * 3 + 2] = xi.z;
                }

                Matrix2x2 IB = inv_B_matrices.read(I);
                BWS::Float2x2 IB_custom;
                IB_custom[0] = Float2(IB[0][0], IB[0][1]);
                IB_custom[1] = Float2(IB[1][0], IB[1][1]);

                Float lambda      = lambdas_buf.read(I);
                Float mu          = mus_buf.read(I);
                Float strain_rate = strain_rates_buf.read(I);
                Float rest_area   = rest_areas_buf.read(I);

                Float thickness = triangle_thickness(thicknesses_buf.read(idx(0)),
                                                     thicknesses_buf.read(idx(1)),
                                                     thicknesses_buf.read(idx(2)));

                // Compute Ds and F
                BWS::Float3x2 Ds = BWS::Ds3x2(
                    Float3(X[0], X[1], X[2]),
                    Float3(X[3], X[4], X[5]),
                    Float3(X[6], X[7], X[8])
                );
                BWS::Float3x2 F = Ds * IB_custom;

                Float2 anisotropic_a(1.0f, 0.0f);
                Float2 anisotropic_b(0.0f, 1.0f);

                Float V = 2.0f * rest_area * thickness;
                Float Vdt2 = V * dt_val * dt_val;

                auto dFdx = BWS::dFdX(IB_custom);

                BWS::Float3x2 dEdF;
                BWS::dEdF(dEdF, F, anisotropic_a, anisotropic_b, lambda, mu, strain_rate);

                auto VecdEdF = BWS::flatten(dEdF);

                // G = dFdx^T * VecdEdF (9 elements)
                BWS::Float9 G;
                for(int i = 0; i < 9; ++i)
                {
                    G[i] = 0.0f;
                    for(int j = 0; j < 6; ++j)
                    {
                        G[i] += dFdx[j][i] * VecdEdF[j];
                    }
                }

                // Scale gradient
                for(int i = 0; i < 9; ++i)
                {
                    G[i] *= Vdt2;
                }

                // Write gradients - atomic add to gradient buffer at each vertex
                for(int i = 0; i < 3; ++i)
                {
                    Float3 Gi(G[i * 3 + 0], G[i * 3 + 1], G[i * 3 + 2]);
                    gradients_buf.atomic(idx(i)).fetch_add(Gi);
                }

                $if(!gradient_only_val)
                {
                    // Compute Hessian
                    BWS::Float6x6 ddEddF;
                    BWS::ddEddF(ddEddF, F, anisotropic_a, anisotropic_b, lambda, mu, strain_rate);

                    // Scale Hessian
                    for(int col = 0; col < 6; ++col)
                    {
                        for(int row = 0; row < 6; ++row)
                        {
                            ddEddF[col][row] *= Vdt2;
                        }
                    }

                    // H = dFdx^T * ddEddF * dFdx (9x9)
                    // First compute ddEddF * dFdx (6x9)
                    std::array<std::array<float, 9>, 6> temp;
                    for(int col = 0; col < 9; ++col)
                    {
                        for(int row = 0; row < 6; ++row)
                        {
                            float sum = 0.0f;
                            for(int k = 0; k < 6; ++k)
                            {
                                sum += ddEddF[k][row] * dFdx[k][col];
                            }
                            temp[row][col] = sum;
                        }
                    }

                    // Then compute dFdx^T * temp (9x9)
                    Matrix9x9 H;
                    for(int col = 0; col < 9; ++col)
                    {
                        for(int row = 0; row < 9; ++row)
                        {
                            float sum = 0.0f;
                            for(int k = 0; k < 6; ++k)
                            {
                                sum += dFdx[k][row] * temp[k][col];
                            }
                            H[col][row] = sum;
                        }
                    }

                    // Apply SPD projection
                    H = clamp_to_spd(H);

                    // Write Hessian blocks (upper triangular)
                    // For 3 vertices, we have 6 upper-triangular blocks (3*4/2 = 6)
                    IndexT hess_offset = I * HalfHessianSize;

                    // Block (0,0): vertices 0-0
                    Matrix3x3 H_00;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_00[c][r] = H[c][r];
                        }
                    }
                    hessians_buf.write(hess_offset + 0, H_00);

                    // Block (0,1): vertices 0-1
                    Matrix3x3 H_01;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_01[c][r] = H[3 + c][r];
                        }
                    }
                    hessians_buf.write(hess_offset + 1, H_01);

                    // Block (0,2): vertices 0-2
                    Matrix3x3 H_02;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_02[c][r] = H[6 + c][r];
                        }
                    }
                    hessians_buf.write(hess_offset + 2, H_02);

                    // Block (1,1): vertices 1-1
                    Matrix3x3 H_11;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_11[c][r] = H[3 + c][3 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 3, H_11);

                    // Block (1,2): vertices 1-2
                    Matrix3x3 H_12;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_12[c][r] = H[6 + c][3 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 4, H_12);

                    // Block (2,2): vertices 2-2
                    Matrix3x3 H_22;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_22[c][r] = H[6 + c][6 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 5, H_22);
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_grad_hess_kernel);
        engine().stream() << shader(mus.view(),
                                    lambdas.view(),
                                    strain_rates.view(),
                                    indices,
                                    xs,
                                    thicknesses,
                                    gradients,
                                    hessians,
                                    rest_areas,
                                    dt,
                                    gradient_only)
                                .dispatch(indices.size());
    }
};

REGISTER_SIM_SYSTEM(StrainLimitingBaraffWitkinShell2D);
}  // namespace uipc::backend::luisa
