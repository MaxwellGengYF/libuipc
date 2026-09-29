#include <affine_body/bdf/affine_body_bdf1_kinetic.h>
#include <frame_stream.h>

/***************************************************************
 * T6.2 — AffineBodyBDF1Kinetic: the two kernels, their cached shaders and the
 * system registration (reference:
 * src/backends/cuda/affine_body/bdf/affine_body_bdf1_kinetic.cu).
 *
 * The gradient and Hessian stages write the *dense per-body* buffers T6.1b's
 * `ABDLinearSubsystem::ComputeGradientHessianInfo` hands over
 * (`lud::LBufferView<LVector12>` / `lud::LBufferView<LMatrix12x12>`, see cookbook
 * §7.22 rule 6) -- `gradients->write(i, G12)` and `hessians->write(i, H12)` are
 * literally the reference's `gradients(i) = G` / `hessians(i) = M.to_mat()`.  No
 * device doublet/triplet view has to be rebuilt here (that is the FEM kernel's
 * idiom); a 12-vector is loaded with `load_vec12` and stored with `store_vec12`
 * (affine_body/type_define.h).
 *
 * The reference's `if(gradient_only) return;` -- which skips the Hessian stage --
 * becomes `$if(gradient_only == 0u)` (RULE M2): a device boolean may not branch.
 * Unlike the FEM kinetic there is no empty-Hessian-view hazard to guard on the host
 * side, because the ABD subsystem passes a dense per-body buffer in both modes.
 *
 * DEVIATION (reported): the reference's `..._compute_gradient_hessian_kernel` takes
 * a `q_prevs` view, binds `const auto& q_prev = q_prevs(i)` and never reads it.  The
 * port drops the argument (an unused kernel parameter is dead weight and the
 * warning-free requirement forbids keeping a named one); the FEM twin's kernel has no
 * such parameter either.
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace
{
/// `AffineBodyBDF1Kinetic_do_compute_energy_kernel`
auto& bdf1_kinetic_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<int32_t>,              // is_fixed
                                   lcrt::Buffer<int32_t>,              // ext_kinetic
                                   lcrt::Buffer<LVector12>,            // qs
                                   lcrt::Buffer<LVector12>,            // q_tildes
                                   lcrt::Buffer<ABDJacobiDyadicMass>,  // masses
                                   lcrt::Buffer<double>,               // Ks
                                   uint32_t>(                          // n
            [](lcrt::BufferVar<int32_t>              is_fixed,
               lcrt::BufferVar<int32_t>              ext_kinetic,
               lcrt::BufferVar<LVector12>            qs,
               lcrt::BufferVar<LVector12>            q_tildes,
               lcrt::BufferVar<ABDJacobiDyadicMass>  masses,
               lcrt::BufferVar<double>               Ks,
               lcrt::Var<uint32_t>                   n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto K = bdf1_kinetic_detail::energy(load_vec12(qs->read(i)),
                                                         load_vec12(q_tildes->read(i)),
                                                         load_dyadic_mass(masses->read(i)),
                                                         is_fixed->read(i),
                                                         ext_kinetic->read(i));
                    Ks->write(i, K);
                };
            });
    return k;
}

/// `AffineBodyBDF1Kinetic_do_compute_gradient_hessian_kernel`
auto& bdf1_kinetic_gradient_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<int32_t>,              // is_fixed
                                   lcrt::Buffer<LVector12>,            // qs
                                   lcrt::Buffer<LVector12>,            // q_tildes
                                   lcrt::Buffer<ABDJacobiDyadicMass>,  // masses
                                   lcrt::Buffer<LVector12>,            // gradients
                                   lcrt::Buffer<LMatrix12x12>,         // hessians
                                   uint32_t,                           // gradient_only
                                   uint32_t>(                          // n
            [](lcrt::BufferVar<int32_t>              is_fixed,
               lcrt::BufferVar<LVector12>            qs,
               lcrt::BufferVar<LVector12>            q_tildes,
               lcrt::BufferVar<ABDJacobiDyadicMass>  masses,
               lcrt::BufferVar<LVector12>            gradients,
               lcrt::BufferVar<LMatrix12x12>         hessians,
               lcrt::Var<uint32_t>                   gradient_only,
               lcrt::Var<uint32_t>                   n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    // The buffers are indexed by body id, so the dispatch index is
                    // the element index (no int32 cast/layout view is needed here,
                    // unlike the FEM kernel's doublet/triplet views).
                    const auto m   = load_dyadic_mass(masses->read(i));
                    const auto q   = load_vec12(qs->read(i));
                    const auto q_t = load_vec12(q_tildes->read(i));

                    auto G = bdf1_kinetic_detail::gradient(q, q_t, m, is_fixed->read(i));

                    lcrt::Var<LVector12> g_out;
                    store_vec12(g_out, G);
                    gradients->write(i, g_out);

                    $if(gradient_only == 0u)
                    {
                        auto H = bdf1_kinetic_detail::hessian(m);
                        lcrt::Var<LMatrix12x12> h_out;
                        store_mat12x12(h_out, H);
                        hessians->write(i, h_out);
                    };
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyBDF1Kinetic);

void AffineBodyBDF1Kinetic::do_build(BuildInfo& info)
{
    (void)info;
    // need BDF1 flag for BDF1 time integration
    require<BDF1Flag>();
}

void AffineBodyBDF1Kinetic::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n > 0)
    {
        auto& k = bdf1_kinetic_energy_kernel();
        frame_stream().record(k(info.is_fixed(),
                                info.external_kinetic(),
                                info.qs(),
                                info.q_tildes(),
                                info.masses(),
                                info.energies(),
                                n)
                                  .dispatch(n));
    }
}

void AffineBodyBDF1Kinetic::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n == 0)
        return;

    auto& k = bdf1_kinetic_gradient_hessian_kernel();
    frame_stream().record(k(info.is_fixed(),
                            info.qs(),
                            info.q_tildes(),
                            info.masses(),
                            info.gradients(),
                            info.hessians(),
                            info.gradient_only() ? 1u : 0u,
                            n)
                              .dispatch(n));
}
}  // namespace uipc::backend::luisa
