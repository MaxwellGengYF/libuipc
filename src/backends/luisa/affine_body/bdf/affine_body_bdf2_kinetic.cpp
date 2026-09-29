#include <affine_body/bdf/affine_body_bdf2_kinetic.h>
#include <frame_stream.h>

/***************************************************************
 * T6.2 — AffineBodyBDF2Kinetic: the two kernels, their cached shaders and the system
 * registration (reference:
 * src/backends/cuda/affine_body/bdf/affine_body_bdf2_kinetic.cu).
 *
 * Identical structure to the BDF1 sibling; the only difference is that both kernels
 * take the `inv_beta = 9/4` scaling as a *scalar argument* (the reference passes its
 * `static constexpr Float inv_beta` at the call site, which is what makes the value
 * observable from the device side and assertable from the test).
 *
 * The gradient/Hessian stages write the dense per-body buffers T6.1b's
 * `ComputeGradientHessianInfo` hands over, and `if(gradient_only) return;` becomes
 * `$if(gradient_only == 0u)` (RULE M2), exactly like the BDF1 sibling.
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace
{
/// `AffineBodyBDF2Kinetic_do_compute_energy_kernel`
auto& bdf2_kinetic_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<int32_t>,              // is_fixed
                                   lcrt::Buffer<int32_t>,              // ext_kinetic
                                   lcrt::Buffer<LVector12>,            // qs
                                   lcrt::Buffer<LVector12>,            // q_tildes
                                   lcrt::Buffer<ABDJacobiDyadicMass>,  // masses
                                   lcrt::Buffer<double>,               // Ks
                                   double,                             // inv_beta
                                   uint32_t>(                          // n
            [](lcrt::BufferVar<int32_t>              is_fixed,
               lcrt::BufferVar<int32_t>              ext_kinetic,
               lcrt::BufferVar<LVector12>            qs,
               lcrt::BufferVar<LVector12>            q_tildes,
               lcrt::BufferVar<ABDJacobiDyadicMass>  masses,
               lcrt::BufferVar<double>               Ks,
               lcrt::Var<double>                     inv_beta,
               lcrt::Var<uint32_t>                   n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto K = bdf2_kinetic_detail::energy(load_vec12(qs->read(i)),
                                                         load_vec12(q_tildes->read(i)),
                                                         load_dyadic_mass(masses->read(i)),
                                                         inv_beta,
                                                         is_fixed->read(i),
                                                         ext_kinetic->read(i));
                    Ks->write(i, K);
                };
            });
    return k;
}

/// `AffineBodyBDF2Kinetic_do_compute_gradient_hessian_kernel`
auto& bdf2_kinetic_gradient_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<int32_t>,              // is_fixed
                                   lcrt::Buffer<LVector12>,            // qs
                                   lcrt::Buffer<LVector12>,            // q_tildes
                                   lcrt::Buffer<ABDJacobiDyadicMass>,  // masses
                                   lcrt::Buffer<LVector12>,            // gradients
                                   lcrt::Buffer<LMatrix12x12>,         // hessians
                                   uint32_t,                           // gradient_only
                                   double,                             // inv_beta
                                   uint32_t>(                          // n
            [](lcrt::BufferVar<int32_t>              is_fixed,
               lcrt::BufferVar<LVector12>            qs,
               lcrt::BufferVar<LVector12>            q_tildes,
               lcrt::BufferVar<ABDJacobiDyadicMass>  masses,
               lcrt::BufferVar<LVector12>            gradients,
               lcrt::BufferVar<LMatrix12x12>         hessians,
               lcrt::Var<uint32_t>                   gradient_only,
               lcrt::Var<double>                     inv_beta,
               lcrt::Var<uint32_t>                   n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    const auto m   = load_dyadic_mass(masses->read(i));
                    const auto q   = load_vec12(qs->read(i));
                    const auto q_t = load_vec12(q_tildes->read(i));

                    auto G = bdf2_kinetic_detail::gradient(
                        q, q_t, m, inv_beta, is_fixed->read(i));

                    lcrt::Var<LVector12> g_out;
                    store_vec12(g_out, G);
                    gradients->write(i, g_out);

                    $if(gradient_only == 0u)
                    {
                        auto H = bdf2_kinetic_detail::hessian(m, inv_beta);
                        lcrt::Var<LMatrix12x12> h_out;
                        store_mat12x12(h_out, H);
                        hessians->write(i, h_out);
                    };
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyBDF2Kinetic);

void AffineBodyBDF2Kinetic::do_build(BuildInfo& info)
{
    (void)info;
    // need BDF2 flag for BDF2 time integration
    require<BDF2Flag>();
}

void AffineBodyBDF2Kinetic::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n > 0)
    {
        auto& k = bdf2_kinetic_energy_kernel();
        frame_stream().record(k(info.is_fixed(),
                                info.external_kinetic(),
                                info.qs(),
                                info.q_tildes(),
                                info.masses(),
                                info.energies(),
                                inv_beta,
                                n)
                                  .dispatch(n));
    }
}

void AffineBodyBDF2Kinetic::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n == 0)
        return;

    auto& k = bdf2_kinetic_gradient_hessian_kernel();
    frame_stream().record(k(info.is_fixed(),
                            info.qs(),
                            info.q_tildes(),
                            info.masses(),
                            info.gradients(),
                            info.hessians(),
                            info.gradient_only() ? 1u : 0u,
                            inv_beta,
                            n)
                              .dispatch(n));
}
}  // namespace uipc::backend::luisa
