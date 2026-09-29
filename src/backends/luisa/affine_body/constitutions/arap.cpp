/********************************************************************
 * @file affine_body/constitutions/arap.cpp
 * @brief T6.3a — the ABD ARAP constitution: the two kernels, their cached shaders,
 *        the out-of-line members and the system registration (reference:
 *        src/backends/cuda/affine_body/constitutions/arap.cu).
 *
 * The class itself is declared in `arap.h` (see the deviation note there).
 *
 * Kernel mapping (cookbook §7.9/§7.10/§7.13, the T6.1b/T6.2 idiom):
 *   * `cuda_tool::CBufferView<Vector12>` -> `lud::LCBufferView<LVector12>`; a
 *     *windowed* view (`info.qs()`, `info.volumes()`) is passed as a kernel
 *     argument and LC binds it at `base + offset_bytes` (verified in LC's CUDA
 *     backend: `CUDABuffer::binding(offset, size)` -> `device_address() + offset`,
 *     §7.17 rule 1), so the per-body index arithmetic is identical to the
 *     reference's.
 *   * the gradient/Hessian buffers the linear subsystem hands over are the *dense
 *     per-body* `lud::LBufferView<LVector12>` / `<LMatrix12x12>` (§7.22 rule 6), so
 *     `gradients->write(i, G12)` / `hessians->write(i, H12)` are literally the
 *     reference's `gradients(i) = G` / `hessians(i) = H`; a 12-vector/12x12 is
 *     loaded with `load_vec12`/`store_vec12`/`store_mat12x12`
 *     (affine_body/type_define.h).
 *   * the per-body math is the *same* generic routine the unit test instantiates
 *     (`abd_arap_detail::{body_energy,body_gradient,body_hessian}`).
 *
 * The reference's `if(gradient_only) return;` becomes `$if(gradient_only == 0u)`
 * (RULE M2): a device predicate cannot drive host control flow.  There is no
 * empty-Hessian-view host guard (unlike the FEM constitutions): the ABD subsystem
 * passes a dense per-body buffer in *both* modes (the T6.2 finding, §7.22 rule 6).
 *
 * NO `make_spd`: the reference's `arap.cu` includes `utils/make_spd.h` but never
 * calls it (measured: the only call sites in `src/backends/cuda/affine_body/
 * constitutions/` are `ortho_potential.cu:66` and the four joints).  The port
 * therefore drops the vestigial include and ships the *unclamped* exact Hessian.
 ********************************************************************/
#include <affine_body/constitutions/arap.h>
#include <frame_stream.h>

namespace uipc::backend::luisa
{
namespace
{
// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// `arap_compute_energy_kernel`
auto& arap_energy_kernel()
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
                auto E = abd_arap_detail::body_energy<lcrt::Var<double>>(
                    kappas->read(i), dt, volumes->read(i), load_vec12(qs->read(i)));
                shape_energies->write(i, E);
            };
        });
    return k;
}

/// `arap_compute_gradient_hessian_kernel`
auto& arap_gradient_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector12>,     // qs
                                                lcrt::Buffer<double>,        // volumes
                                                lcrt::Buffer<LVector12>,     // gradients
                                                lcrt::Buffer<LMatrix12x12>,  // hessians
                                                lcrt::Buffer<double>,        // kappas
                                                double,                      // dt
                                                uint32_t,                    // gradient_only
                                                uint32_t>(                   // n
        [](lcrt::BufferVar<LVector12>     qs,
           lcrt::BufferVar<double>        volumes,
           lcrt::BufferVar<LVector12>     gradients,
           lcrt::BufferVar<LMatrix12x12>  hessians,
           lcrt::BufferVar<double>        kappas,
           lcrt::Var<double>              dt,
           lcrt::Var<uint32_t>            gradient_only,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto kappa   = kappas->read(i);
                auto volume  = volumes->read(i);
                auto q       = load_vec12(qs->read(i));

                auto G = abd_arap_detail::body_gradient<lcrt::Var<double>>(kappa, dt, volume, q);

                lcrt::Var<LVector12> g_out;
                store_vec12(g_out, G);
                gradients->write(i, g_out);

                // reference: `if(gradient_only) return;`
                $if(gradient_only == 0u)
                {
                    auto H = abd_arap_detail::body_hessian<lcrt::Var<double>>(kappa, dt, volume, q);

                    lcrt::Var<LMatrix12x12> h_out;
                    store_mat12x12(h_out, H);
                    hessians->write(i, h_out);
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(ARAP);

U64 ARAP::get_uid() const noexcept
{
    return ConstitutionUID;
}

void ARAP::do_build(BuildInfo& info)
{
    (void)info;
    // the reference's body is empty (the base's `do_build()` does the work)
}

void ARAP::do_init(AffineBodyDynamics::FilteredInfo& info)
{
    using ForEachInfo = AffineBodyDynamics::ForEachInfo;

    // find out constitution coefficients
    h_kappas.resize(info.body_count());

    auto geo_slots = world().scene().geometries();

    // the reference's getter, verbatim (the frontend's `apply_to` always writes the
    // per-instance `kappa` attribute for a geometry of this uid)
    info.for_each(
        geo_slots,
        [](geometry::SimplicialComplex& sc)
        { return sc.instances().find<Float>("kappa")->view(); },
        [&](const ForEachInfo& I, Float kappa) { h_kappas[I.global_index()] = kappa; });

    kappas.resize(h_kappas.size());
    kappas.view().copy_from(span<const Float>{h_kappas});
}

void ARAP::do_compute_energy(ComputeEnergyInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n > 0)
    {
        auto& k = arap_energy_kernel();
        frame_stream().record(k(info.energies(),
                                info.qs(),
                                kappas.cview(),
                                info.volumes(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}

void ARAP::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n == 0)
        return;

    auto& k = arap_gradient_hessian_kernel();
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
