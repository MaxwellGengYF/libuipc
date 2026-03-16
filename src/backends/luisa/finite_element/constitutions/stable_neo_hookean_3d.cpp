#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>
#include <finite_element/fem_utils.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class StableNeoHookean3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 10;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_mus;
    vector<Float> h_lambdas;

    luisa::compute::Buffer<Float> mus;
    luisa::compute::Buffer<Float> lambdas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(mus.size());
        info.gradient_count(mus.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(mus.size() * HalfHessianSize);
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_mus.resize(N);
        h_lambdas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto mu     = sc.tetrahedra().find<Float>("mu");
                auto lambda = sc.tetrahedra().find<Float>("lambda");

                return zip(mu->view(), lambda->view());
            },
            [&](const ForEachInfo& I, auto mu_and_lambda)
            {
                auto&& [mu, lambda] = mu_and_lambda;

                auto vI = I.global_index();

                h_mus[vI]     = mu;
                h_lambdas[vI] = lambda;
            });

        auto& device = engine().device();
        mus          = device.create_buffer<Float>(N);
        mus.copy_from(h_mus.data());

        lambdas = device.create_buffer<Float>(N);
        lambdas.copy_from(h_lambdas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace SNH = sym::stable_neo_hookean_3d;

        auto indices  = info.indices();
        auto xs       = info.xs();
        auto Dm_invs  = info.Dm_invs();
        auto volumes  = info.rest_volumes();
        auto energies = info.energies();
        Float dt      = info.dt();

        Kernel1D compute_energy_kernel = [&](BufferVar<Float> mus_buf,
                                             BufferVar<Float> lambdas_buf,
                                             BufferVar<Float> energies_buf,
                                             BufferVar<Vector4i> indices_buf,
                                             BufferVar<Vector3> xs_buf,
                                             BufferVar<Matrix3x3> Dm_invs_buf,
                                             BufferVar<Float> volumes_buf,
                                             Float dt_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < indices_buf.size())
            {
                Vector4i tet     = indices_buf.read(I);
                Matrix3x3 Dm_inv = Dm_invs_buf.read(I);
                Float mu         = mus_buf.read(I);
                Float lambda     = lambdas_buf.read(I);

                Vector3 x0 = xs_buf.read(tet(0));
                Vector3 x1 = xs_buf.read(tet(1));
                Vector3 x2 = xs_buf.read(tet(2));
                Vector3 x3 = xs_buf.read(tet(3));

                Matrix3x3 F = fem::F(x0, x1, x2, x3, Dm_inv);

                Float E;
                SNH::E(E, mu, lambda, F);
                E *= dt_val * dt_val * volumes_buf.read(I);
                energies_buf.write(I, E);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(mus.view(),
                                    lambdas.view(),
                                    energies,
                                    indices,
                                    xs,
                                    Dm_invs,
                                    volumes,
                                    dt)
                                .dispatch(indices.size());
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace SNH = sym::stable_neo_hookean_3d;

        auto indices       = info.indices();
        auto xs            = info.xs();
        auto Dm_invs       = info.Dm_invs();
        auto volumes       = info.rest_volumes();
        auto gradients     = info.gradients();
        auto hessians      = info.hessians();
        Float dt           = info.dt();
        Bool gradient_only = info.gradient_only();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Float> mus_buf,
                                                BufferVar<Float> lambdas_buf,
                                                BufferVar<Vector4i> indices_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Matrix3x3> Dm_invs_buf,
                                                BufferVar<Float> volumes_buf,
                                                BufferVar<Float> gradients_buf,
                                                BufferVar<Matrix3x3> hessians_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < indices_buf.size())
            {
                Vector4i tet     = indices_buf.read(I);
                Matrix3x3 Dm_inv = Dm_invs_buf.read(I);
                Float mu         = mus_buf.read(I);
                Float lambda     = lambdas_buf.read(I);

                Vector3 x0 = xs_buf.read(tet(0));
                Vector3 x1 = xs_buf.read(tet(1));
                Vector3 x2 = xs_buf.read(tet(2));
                Vector3 x3 = xs_buf.read(tet(3));

                Matrix3x3 F = fem::F(x0, x1, x2, x3, Dm_inv);

                Float Vdt2 = volumes_buf.read(I) * dt_val * dt_val;

                Matrix3x3 dEdF;
                SNH::dEdVecF(dEdF, mu, lambda, F);

                // Flatten dEdF to Vector9
                Vector9 VecdEdF;
                for(int col = 0; col < 3; ++col)
                {
                    for(int row = 0; row < 3; ++row)
                    {
                        VecdEdF[col * 3 + row] = dEdF[col][row];
                    }
                }
                VecdEdF *= Vdt2;

                Matrix9x12 dFdx = fem::dFdx(Dm_inv);
                Vector12   G    = dFdx.transpose() * VecdEdF;

                // Write gradients - atomic add to gradient buffer at each vertex
                for(int i = 0; i < 4; ++i)
                {
                    Float3 Gi;
                    Gi.x = G[i * 3 + 0];
                    Gi.y = G[i * 3 + 1];
                    Gi.z = G[i * 3 + 2];

                    gradients_buf.atomic(tet(i)).fetch_add(Gi);
                }

                $if(!gradient_only_val)
                {
                    // Compute Hessian
                    luisa::Matrix<Float, 9> ddEddF;
                    SNH::ddEddVecF(ddEddF, mu, lambda, F);

                    // Scale Hessian
                    for(int col = 0; col < 9; ++col)
                    {
                        for(int row = 0; row < 9; ++row)
                        {
                            ddEddF[col][row] *= Vdt2;
                        }
                    }

                    // Convert to Matrix9x9 for SPD projection
                    Matrix9x9 ddEddF_array;
                    for(int col = 0; col < 9; ++col)
                    {
                        for(int row = 0; row < 9; ++row)
                        {
                            ddEddF_array[col][row] = ddEddF[col][row];
                        }
                    }

                    // Apply SPD projection
                    ddEddF_array = clamp_to_spd(ddEddF_array);

                    // Convert back to luisa::Matrix<Float, 9>
                    for(int col = 0; col < 9; ++col)
                    {
                        for(int row = 0; row < 9; ++row)
                        {
                            ddEddF[col][row] = ddEddF_array[col][row];
                        }
                    }

                    // Compute H = dFdx^T * ddEddF * dFdx
                    // dFdx is 9x12, ddEddF is 9x9, result is 12x12
                    // First compute ddEddF * dFdx (9x12)
                    luisa::Matrix<Float, 9, 12> temp;
                    for(int col = 0; col < 12; ++col)
                    {
                        for(int row = 0; row < 9; ++row)
                        {
                            Float sum = 0.0f;
                            for(int k = 0; k < 9; ++k)
                            {
                                sum += ddEddF[k][row] * dFdx[k][col];
                            }
                            temp[col][row] = sum;  // temp is column-major
                        }
                    }

                    // Then compute dFdx^T * temp (12x12)
                    // H[i][j] = sum_k dFdx[k][i] * temp[j][k]
                    Matrix12x12 H;
                    for(int col = 0; col < 12; ++col)
                    {
                        for(int row = 0; row < 12; ++row)
                        {
                            Float sum = 0.0f;
                            for(int k = 0; k < 9; ++k)
                            {
                                sum += dFdx[k][row] * temp[col][k];
                            }
                            H[col][row] = sum;
                        }
                    }

                    // Write Hessian blocks (upper triangular)
                    // For 4 vertices, we have 10 upper-triangular blocks (4*5/2 = 10)
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

                    // Block (0,3): vertices 0-3
                    Matrix3x3 H_03;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_03[c][r] = H[9 + c][r];
                        }
                    }
                    hessians_buf.write(hess_offset + 3, H_03);

                    // Block (1,1): vertices 1-1
                    Matrix3x3 H_11;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_11[c][r] = H[3 + c][3 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 4, H_11);

                    // Block (1,2): vertices 1-2
                    Matrix3x3 H_12;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_12[c][r] = H[6 + c][3 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 5, H_12);

                    // Block (1,3): vertices 1-3
                    Matrix3x3 H_13;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_13[c][r] = H[9 + c][3 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 6, H_13);

                    // Block (2,2): vertices 2-2
                    Matrix3x3 H_22;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_22[c][r] = H[6 + c][6 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 7, H_22);

                    // Block (2,3): vertices 2-3
                    Matrix3x3 H_23;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_23[c][r] = H[9 + c][6 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 8, H_23);

                    // Block (3,3): vertices 3-3
                    Matrix3x3 H_33;
                    for(int r = 0; r < 3; ++r)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            H_33[c][r] = H[9 + c][9 + r];
                        }
                    }
                    hessians_buf.write(hess_offset + 9, H_33);
                };
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_grad_hess_kernel);
        engine().stream() << shader(mus.view(),
                                    lambdas.view(),
                                    indices,
                                    xs,
                                    Dm_invs,
                                    volumes,
                                    gradients,
                                    hessians,
                                    dt,
                                    gradient_only)
                                .dispatch(indices.size());
    }
};

REGISTER_SIM_SYSTEM(StableNeoHookean3D);
}  // namespace uipc::backend::luisa
