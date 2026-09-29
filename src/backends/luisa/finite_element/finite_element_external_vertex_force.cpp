#include <finite_element/finite_element_external_vertex_force.h>
#include <frame_stream.h>

/*************************************************************************************************
* T5.4b — FiniteElementExternalVertexForce (port of
* src/backends/cuda/finite_element/finite_element_external_vertex_force.cu).
*
* Read the scope note in finite_element_external_vertex_force.h first: it explains
* the scalar-alias spelling of the reference's `eigen::atomic_add(forces(vid), F)`.
*
*   reference                                port
*   ---------------------------------------  ------------------------------------------------
*   forces(vid) (AoS BufferView<Vector3>)     forces.native().as<double>() (flat double[3n])
*   eigen::atomic_add(forces(vid), F)         scatter_add_vec3(flat_view, vid, F)
*   3 x `atomicAdd(&forces[3*vid + c], F[c])` the same three atomic adds (that is what
*                                             the CUDA codegen of the LC fetch_add emits)
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using DenseDeviceView = lla::LDenseVectorDeviceView<lcrt::Var<double>>;

/// `FiniteElementExternalVertexForce_do_step_kernel`.
///
/// The first argument is the *scalar alias* of the method's AoS force buffer: the
/// shader prototype is `Buffer<double>` while the dispatch passes
/// `LBufferView<LVector3>::native().as<double>()` (same handle and byte offset,
/// element stride 8, size = 3 x the vertex count).  `do_step` asserts the stride
/// contract that makes the alias exact.
auto& finite_element_external_vertex_force_step_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,        // forces (scalar alias)
                                                lcrt::Buffer<std::int32_t>,  // vertex_ids
                                                lcrt::Buffer<LVector3>,      // vertex_forces
                                                uint32_t>(                   // n
        [](lcrt::BufferVar<double>       forces_flat,
           lcrt::BufferVar<std::int32_t> vertex_ids,
           lcrt::BufferVar<LVector3>     vertex_forces,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto vid = vertex_ids->read(i);
                auto f   = load_vec3(vertex_forces->read(i));

                auto forces = DenseDeviceView::from_handles(forces_flat, n * 3u);
                finite_element_external_vertex_force_detail::scatter_add_vec3(forces, vid, f);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FiniteElementExternalVertexForce);

void FiniteElementExternalVertexForce::do_build(BuildInfo& info)
{
    (void)info;
    finite_element_method = require<FiniteElementMethod>();
    constraint            = require<FiniteElementExternalVertexForceConstraint>();
}

void FiniteElementExternalVertexForce::do_step(ExternalForceInfo& info)
{
    auto n = static_cast<uint32_t>(constraint->forces().size());
    if(n == 0)
    {
        return;
    }

    auto forces = info.external_forces();

    // The scalar alias contract: `BufferView<LVector3>::as<double>()` addresses the
    // same bytes with element stride sizeof(double) and size 3 x the vector count,
    // so the flat index of component c of vertex v is 3 * v + c.  This holds iff the
    // LVector3 buffer's element stride is exactly sizeof(LVector3) (24 B = 3 x 8),
    // i.e. LC did not pad the registered struct.
    auto native = forces.native();
    UIPC_ASSERT(native.stride() == sizeof(LVector3),
                "FiniteElementExternalVertexForce: the scalar alias of the external-force buffer "
                "needs element_stride == sizeof(LVector3) == {} B, measured {} B",
                sizeof(LVector3),
                native.stride());

    auto& k = finite_element_external_vertex_force_step_kernel();
    frame_stream().record(k(native.as<double>(),
                            constraint->vertex_ids(),
                            constraint->forces(),
                            n)
                              .dispatch(n));
}
}  // namespace uipc::backend::luisa
