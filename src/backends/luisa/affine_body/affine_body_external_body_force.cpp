#include <affine_body/affine_body_external_body_force.h>
#include <frame_stream.h>

/*************************************************************************************************
* T6.4b — AffineBodyExternalBodyForce (port of
* src/backends/cuda/affine_body/affine_body_external_body_force.cu).
*
* Read the scope note in affine_body_external_body_force.h first: it explains the
* scalar-alias spelling of the reference's `eigen::atomic_add(forces(body_id), F)`.
*
*   reference                                port
*   ---------------------------------------  ------------------------------------------------
*   forces(body_id) (AoS BufferView<Vector12>) forces.native().as<double>() (flat double[12n])
*   eigen::atomic_add(forces(body_id), F)      scatter_add_vec12(flat_view, body_id, F)
*   12 x `atomicAdd(&forces[12*body_id + c], F[c])`  the same twelve atomic adds (that is what
*                                              the CUDA codegen of the LC fetch_add emits)
*
* `n` counts the *force entries* while the dense view's extent is the dynamics'
* `body_count` (the buffer is one Vector12 per body), so the body count travels as its
* own kernel argument -- the T6.3b3 revolute external-force reporter's convention.
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using DenseDeviceView = lla::LDenseVectorDeviceView<lcrt::Var<double>>;

/// `affine_body_external_body_force_do_step_kernel`.
///
/// The first argument is the *scalar alias* of the dynamics' AoS force buffer: the
/// shader prototype is `Buffer<double>` while the dispatch passes
/// `LBufferView<LVector12>::native().as<double>()` (same handle and byte offset,
/// element stride 8, size = 12 x the body count).  `do_step` asserts the stride
/// contract that makes the alias exact.
auto& affine_body_external_body_force_step_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,        // forces (scalar alias)
                                                lcrt::Buffer<IndexT>,        // body_ids
                                                lcrt::Buffer<LVector12>,     // body_forces
                                                uint32_t,                    // n (the force count)
                                                uint32_t>(                   // body_count
        [](lcrt::BufferVar<double>      forces_flat,
           lcrt::BufferVar<IndexT>      body_ids,
           lcrt::BufferVar<LVector12>   body_forces,
           lcrt::Var<uint32_t>          n,
           lcrt::Var<uint32_t>          body_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto body_id = body_ids.read(i);
                auto f       = load_vec12(body_forces.read(i));

                auto forces = DenseDeviceView::from_handles(forces_flat, body_count * 12u);
                affine_body_external_body_force_detail::scatter_add_vec12(forces, body_id, f);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyExternalBodyForce);

void AffineBodyExternalBodyForce::do_build(BuildInfo& info)
{
    (void)info;
    affine_body_dynamics = require<AffineBodyDynamics>();
    constraint           = require<AffineBodyExternalBodyForceConstraint>();
}

void AffineBodyExternalBodyForce::do_step(ExternalForceInfo& info)
{
    SizeT force_count = constraint->forces().size();

    auto n = static_cast<uint32_t>(force_count);
    if(n == 0)
    {
        return;
    }

    auto forces = info.external_forces();

    // The scalar alias contract: `BufferView<LVector12>::as<double>()` addresses the
    // same bytes with element stride sizeof(double) and size 12 x the body count, so
    // the flat index of component c of body b is 12 * b + c.  This holds iff the
    // LVector12 buffer's element stride is exactly sizeof(LVector12) (96 B = 12 x 8),
    // i.e. LC did not pad the registered struct.
    auto native = forces.native();
    UIPC_ASSERT(native.stride() == sizeof(LVector12),
                "AffineBodyExternalBodyForce: the scalar alias of the external-force buffer "
                "needs element_stride == sizeof(LVector12) == {} B, measured {} B",
                sizeof(LVector12),
                native.stride());

    auto& k = affine_body_external_body_force_step_kernel();
    frame_stream().record(k(native.as<double>(),
                            constraint->body_ids(),
                            constraint->forces(),
                            n,
                            // the dense view's extent is the *body* count, i.e. the
                            // window's own size (the T6.3b3 convention)
                            static_cast<uint32_t>(forces.size()))
                          .dispatch(n));
}
}  // namespace uipc::backend::luisa
