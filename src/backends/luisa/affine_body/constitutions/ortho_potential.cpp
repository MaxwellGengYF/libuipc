#include <sim_system.h>
#include <affine_body/constitutions/ortho_potential_function.h>
#include <affine_body/affine_body_constitution.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/common/enumerate.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
class OrthoPotential final : public AffineBodyConstitution
{
  public:
    static constexpr U64 ConstitutionUID = 1ull;

    using AffineBodyConstitution::AffineBodyConstitution;

    vector<Float> h_kappas;

    luisa::compute::Buffer<Float> kappas;

    virtual void do_build(AffineBodyConstitution::BuildInfo& info) override {}

    U64 get_uid() const override { return ConstitutionUID; }

    void do_init(AffineBodyDynamics::FilteredInfo& info) override
    {
        using ForEachInfo = AffineBodyDynamics::ForEachInfo;

        // find out constitution coefficients
        h_kappas.resize(info.body_count());
        auto geo_slots = world().scene().geometries();

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc)
            { return sc.instances().find<Float>("kappa")->view(); },
            [&](const ForEachInfo& I, Float kappa)
            {
                auto bodyI      = I.global_index();
                h_kappas[bodyI] = kappa;
            });

        _build_on_device();
    }

    void _build_on_device()
    {
        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        kappas = device.create_buffer<Float>(h_kappas.size());
        kappas.view().copy_from(h_kappas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        namespace AOP = sym::abd_ortho_potential;

        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto qs_view = info.qs();
        auto volumes_view = info.volumes();
        auto energies_view = info.energies();
        auto kappas_view = kappas.view();
        Float dt = info.dt();
        SizeT body_count = qs_view.size();

        Kernel1D compute_energy_kernel = [&](BufferVar<Vector12> qs,
                                              BufferVar<Float> volumes,
                                              BufferVar<Float> kappas,
                                              BufferVar<Float> shape_energies) {
            auto i = dispatch_id().x;
            $if(i < body_count) {
                Vector12 q = qs.read(i);
                Float volume = volumes.read(i);
                Float kappa = kappas.read(i);

                Float Vdt2 = volume * dt * dt;

                // Convert Vector12 to Array<float, 12> for ortho_potential function
                AOP::Array<float, 12> q_array;
                for(int k = 0; k < 12; ++k) {
                    q_array[k] = q[k];
                }

                Float E;
                AOP::E(E, kappa, q_array);

                shape_energies.write(i, E * Vdt2);
            };
        };

        auto kernel = device.compile(compute_energy_kernel);
        stream << kernel(qs_view, volumes_view, kappas_view, energies_view)
                      .dispatch(body_count);
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        namespace AOP = sym::abd_ortho_potential;

        auto& device = static_cast<SimEngine&>(world().sim_engine()).device();
        auto& stream = static_cast<SimEngine&>(world().sim_engine()).compute_stream();

        auto qs_view = info.qs();
        auto volumes_view = info.volumes();
        auto gradients_view = info.gradients();
        auto hessians_view = info.hessians();
        auto kappas_view = kappas.view();
        Float dt = info.dt();
        bool gradient_only = info.gradient_only();
        SizeT N = qs_view.size();

        Kernel1D compute_gradient_hessian_kernel = [&](BufferVar<Vector12> qs,
                                                       BufferVar<Float> volumes,
                                                       BufferVar<Float> kappas,
                                                       BufferVar<Vector12> shape_gradients,
                                                       BufferVar<Matrix12x12> body_hessians,
                                                       Bool gradient_only) {
            auto i = dispatch_id().x;
            $if(i < N) {
                Matrix12x12 H = Matrix12x12::zeros();  // Zero matrix
                Vector12    G = Vector12::zeros();     // Zero vector

                Vector12 q = qs.read(i);
                Float kappa = kappas.read(i);
                Float volume = volumes.read(i);

                Float Vdt2 = volume * dt * dt;

                // Convert Vector12 to Array<float, 12> for ortho_potential function
                AOP::Array<float, 12> q_array;
                for(int k = 0; k < 12; ++k) {
                    q_array[k] = q[k];
                }

                // Compute gradient (9 elements for the deformation gradient part)
                AOP::Array<float, 9> G9;
                AOP::dEdq(G9, kappa, q_array);
                
                // Place gradient in positions 3-11 (skip translation part 0-2)
                for(int k = 0; k < 9; ++k) {
                    G[3 + k] = G9[k] * Vdt2;
                }
                shape_gradients.write(i, G);

                $if(!gradient_only) {
                    // Compute Hessian (9x9 for the deformation gradient part)
                    AOP::Matrix9x9 H9x9;
                    AOP::ddEddq(H9x9, kappa, q_array);
                    
                    // Place Hessian in block (3,3) to (11,11)
                    Matrix12x12 H12 = Matrix12x12::zeros();
                    
                    // Copy 9x9 block to positions (3,3)
                    for(int row = 0; row < 9; ++row) {
                        for(int col = 0; col < 9; ++col) {
                            H12(3 + row, 3 + col) = H9x9[row][col] * Vdt2;
                        }
                    }
                    
                    body_hessians.write(i, H12);
                };
            };
        };

        auto kernel = device.compile(compute_gradient_hessian_kernel);
        stream << kernel(qs_view,
                         volumes_view,
                         kappas_view,
                         gradients_view,
                         hessians_view,
                         gradient_only)
                      .dispatch(N);
    }
};

REGISTER_SIM_SYSTEM(OrthoPotential);
}  // namespace uipc::backend::luisa
