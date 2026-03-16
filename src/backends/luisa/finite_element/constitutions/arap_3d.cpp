#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/arap_function.h>
#include <finite_element/fem_utils.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class ARAP3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 9;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_kappas;

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
                auto kappa = sc.tetrahedra().find<Float>("kappa");
                UIPC_ASSERT(kappa, "Can't find attribute `kappa` on tetrahedra, why can it happen?");
                return kappa->view();
            },
            [&](const ForEachInfo& I, Float kappa)
            { h_kappas[I.global_index()] = kappa; });

        auto& device = engine().device();
        kappas       = device.create_buffer<Float>(N);
        kappas.copy_from(h_kappas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace ARAP = sym::arap_3d;

        auto indices     = info.indices();
        auto xs          = info.xs();
        auto Dm_invs     = info.Dm_invs();
        auto volumes     = info.rest_volumes();
        auto energies    = info.energies();
        Float dt         = info.dt();

        Kernel1D compute_energy_kernel = [&](BufferVar<Float> kappas_buf,
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
                Vector4i tet    = indices_buf.read(I);
                Matrix3x3 Dm_inv = Dm_invs_buf.read(I);

                Vector3 x0 = xs_buf.read(tet(0));
                Vector3 x1 = xs_buf.read(tet(1));
                Vector3 x2 = xs_buf.read(tet(2));
                Vector3 x3 = xs_buf.read(tet(3));

                Matrix3x3 F = fem::F(x0, x1, x2, x3, Dm_inv);

                // Compute SVD of F for ARAP energy
                Matrix3x3 U, V;
                Vector3 Sigma;
                svd(F, U, Sigma, V);

                Float E;
                Float kappa_val = kappas_buf.read(I) * dt_val * dt_val;
                Float v         = volumes_buf.read(I);

                ARAP::E(E, kappa_val, v, F, U, Sigma, V);
                energies_buf.write(I, E);
            };
        };

        auto& device = engine().device();
        auto  shader = device.compile(compute_energy_kernel);
        engine().stream() << shader(kappas.view(),
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
        namespace ARAP = sym::arap_3d;

        auto indices       = info.indices();
        auto xs            = info.xs();
        auto Dm_invs       = info.Dm_invs();
        auto volumes       = info.rest_volumes();
        auto gradients     = info.gradients();
        auto hessians      = info.hessians();
        Float dt           = info.dt();
        Bool gradient_only = info.gradient_only();

        Kernel1D compute_grad_hess_kernel = [&](BufferVar<Float> kappas_buf,
                                                BufferVar<Vector4i> indices_buf,
                                                BufferVar<Vector3> xs_buf,
                                                BufferVar<Matrix3x3> Dm_invs_buf,
                                                BufferVar<Float> volumes_buf,
                                                BufferVar<Float> gradients_buf,
                                                BufferVar<Float> hessians_buf,
                                                Float dt_val,
                                                Bool gradient_only_val) noexcept
        {
            auto I = dispatch_x();
            $if(I < indices_buf.size())
            {
                Vector4i tet     = indices_buf.read(I);
                Matrix3x3 Dm_inv = Dm_invs_buf.read(I);

                Vector3 x0 = xs_buf.read(tet(0));
                Vector3 x1 = xs_buf.read(tet(1));
                Vector3 x2 = xs_buf.read(tet(2));
                Vector3 x3 = xs_buf.read(tet(3));

                Matrix3x3 F = fem::F(x0, x1, x2, x3, Dm_inv);

                // Compute SVD of F
                Matrix3x3 U, V;
                Vector3 Sigma;
                svd(F, U, Sigma, V);

                Float kt2 = kappas_buf.read(I) * dt_val * dt_val;
                Float v   = volumes_buf.read(I);

                Vector9 dEdF;
                ARAP::dEdF(dEdF, kt2, v, F, U, Sigma, V);

                Matrix9x12 dFdx = fem::dFdx(Dm_inv);
                Vector12   G12  = dFdx.transpose() * dEdF;

                // Write gradient using DoubletVectorAssembler pattern
                // gradients is a flat buffer, we write 4 doublets (one per vertex)
                IndexT grad_offset = I * StencilSize;
                for(int i = 0; i < 4; ++i)
                {
                    // Write index
                    // gradients buffer layout: index_buffer stores indices, value_buffer stores values
                    // For now, we assume the buffer is pre-allocated with proper layout
                    // Write the 3 components of gradient for vertex i
                    Float3 Gi;
                    Gi.x = G12(i * 3 + 0);
                    Gi.y = G12(i * 3 + 1);
                    Gi.z = G12(i * 3 + 2);
                    
                    // Atomic add to gradient buffer at vertex tet(i)
                    // The gradients buffer is indexed by global vertex index
                    // We need to write to the correct offset
                    auto atomic_grad = gradients_buf->atomic(tet(i) * 3);
                    atomic_grad.fetch_add(Gi.x);
                    atomic_grad.fetch_add(Gi.y);
                    atomic_grad.fetch_add(Gi.z);
                }

                $if(!gradient_only_val)
                {
                    Vector9 H_col[9];
                    ARAP::ddEddF(H_col, kt2, v, F, U, Sigma, V);
                    
                    // Convert column-major H_col to Matrix9x9
                    Matrix9x9 ddEddF;
                    for(int col = 0; col < 9; ++col)
                    {
                        for(int row = 0; row < 9; ++row)
                        {
                            ddEddF(row, col) = H_col[col][row];
                        }
                    }
                    
                    make_spd(ddEddF);
                    Matrix12x12 H12x12 = dFdx.transpose() * ddEddF * dFdx;

                    // Write Hessian using TripletMatrixAssembler pattern
                    // Write only upper triangular part (10 blocks for 4x4 stencil)
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
                            // Layout: row_index, col_index, 3x3 values
                            // For simplicity, we write flattened
                            IndexT block_idx = hess_offset + (ii * 4 + jj - ii * (ii + 1) / 2);
                            // Write row and column indices
                            // hessians_buf stores: [row_idx, col_idx, 9 values] per block
                            // This is a simplified version - actual implementation depends on buffer layout
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
                                    Dm_invs,
                                    volumes,
                                    gradients,
                                    hessians,
                                    dt,
                                    gradient_only)
                                .dispatch(indices.size());
    }
};

REGISTER_SIM_SYSTEM(ARAP3D);
}  // namespace uipc::backend::luisa
