/********************************************************************
 * @file affine_body/constitutions/ortho_potential.cpp
 * @brief T6.3a — the ABD orthogonal-potential constitution: the two kernels, their
 *        cached shaders, the out-of-line members and the system registration
 *        (reference:
 *        src/backends/cuda/affine_body/constitutions/ortho_potential.cu).
 *
 * The class itself is declared in `ortho_potential.h`; the kernel mapping (windowed
 * `LCBufferView` arguments bound at `base + offset_bytes`, the dense per-body
 * gradient/Hessian buffers, `$if(gradient_only == 0u)`) is the one documented in
 * `arap.cpp` and the reference's own `.cu`.
 *
 * The one thing this file has that `arap.cpp` does not is the reference's
 * `make_spd(H9x9, ...)` clamp; it lives in
 * `abd_ortho_potential_detail::hessian_9x9_spd` (ortho_potential_function.h) so the
 * kernel and the unit test run the same statements.
 ********************************************************************/
#include <affine_body/constitutions/ortho_potential.h>
#include <frame_stream.h>

namespace uipc::backend::luisa
{
namespace
{
// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `ortho_potential_compute_energy_kernel`
auto& ortho_potential_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,     // shape_energies
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<double>,     // kappas
                                                lcrt::Buffer<double>,     // volumes
                                                double,                   // dt
                                                uint32_t>(                // n
        [](lcrt::BufferVar<double>     shape_energies,
           lcrt::BufferVar<LVector12>  qs,
           lcrt::BufferVar<double>     kappas,
           lcrt::BufferVar<double>     volumes,
           lcrt::Var<double>           dt,
           lcrt::Var<uint32_t>         n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto E = abd_ortho_potential_detail::body_energy<lcrt::Var<double>>(
                    kappas->read(i), dt, volumes->read(i), load_vec12(qs->read(i)));
                shape_energies->write(i, E);
            };
        });
    return k;
}

/// `ortho_potential_compute_gradient_hessian_kernel`
auto& ortho_potential_gradient_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector12>,     // qs
                                                lcrt::Buffer<double>,        // volumes
                                                lcrt::Buffer<LVector12>,     // gradients
                                                lcrt::Buffer<LMatrix12x12>,  // body_hessian
                                                lcrt::Buffer<double>,        // kappas
                                                double,                      // dt
                                                uint32_t,                    // gradient_only
                                                uint32_t>(                   // n
        [](lcrt::BufferVar<LVector12>     qs,
           lcrt::BufferVar<double>        volumes,
           lcrt::BufferVar<LVector12>     gradients,
           lcrt::BufferVar<LMatrix12x12>  body_hessian,
           lcrt::BufferVar<double>        kappas,
           lcrt::Var<double>              dt,
           lcrt::Var<uint32_t>            gradient_only,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto kappa  = kappas->read(i);
                auto volume = volumes->read(i);
                auto q      = load_vec12(qs->read(i));

                auto G = abd_ortho_potential_detail::body_gradient<lcrt::Var<double>>(
                    kappa, dt, volume, q);

                lcrt::Var<LVector12> g_out;
                store_vec12(g_out, G);
                gradients->write(i, g_out);

                // reference: `if(gradient_only) return;`
                $if(gradient_only == 0u)
                {
                    // the reference's `make_spd` is inside this routine
                    auto H = abd_ortho_potential_detail::body_hessian<lcrt::Var<double>>(
                        kappa, dt, volume, q);

                    lcrt::Var<LMatrix12x12> h_out;
                    store_mat12x12(h_out, H);
                    body_hessian->write(i, h_out);
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(OrthoPotential);

U64 OrthoPotential::get_uid() const noexcept
{
    return ConstitutionUID;
}

void OrthoPotential::do_build(BuildInfo& info)
{
    (void)info;
    // the reference's body is empty (the base's `do_build()` does the work)
}

void OrthoPotential::do_init(AffineBodyDynamics::FilteredInfo& info)
{
    using ForEachInfo = AffineBodyDynamics::ForEachInfo;

    // find out constitution coefficients
    h_kappas.resize(info.body_count());

    auto geo_slots = world().scene().geometries();

    info.for_each(
        geo_slots,
        [](geometry::SimplicialComplex& sc)
        { return sc.instances().find<Float>("kappa")->view(); },
        [&](const ForEachInfo& I, Float kappa) { h_kappas[I.global_index()] = kappa; });

    kappas.resize(h_kappas.size());
    kappas.view().copy_from(span<const Float>{h_kappas});
}

void OrthoPotential::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n > 0)
    {
        auto& k = ortho_potential_energy_kernel();
        frame_stream().record(k(info.energies(),
                                info.qs(),
                                kappas.cview(),
                                info.volumes(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}

void OrthoPotential::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n == 0)
        return;

    auto& k = ortho_potential_gradient_hessian_kernel();
    frame_stream().record(k(info.qs(),
                            info.volumes(),
                            info.gradients(),
                            info.hessians(),
                            kappas.cview(),
                            info.dt(),
                            info.gradient_only() ? 1u : 0u,
                            n)
                            .dispatch(n));
}
}  // namespace uipc::backend::luisa
