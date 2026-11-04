/********************************************************************
 * @file contact_models/ipc_vertex_half_plane_normal_contact.cpp
 * @brief T8.1b — `IPCVertexHalfPlaneNormalContact`: the concrete IPC vertex/half-plane normal
 * contact model (port of
 * src/backends/cuda/contact_system/contact_models/ipc_vertex_half_plane_normal_contact.cu,
 * 163 reference lines).
 *
 * One energy kernel + one assemble kernel over the filter's `PHs()` pairs
 * (`{vertex, half_plane_instance}`).  The half-plane instance's *vertex* index is
 * `HI + half_plane_vertex_offset()` (`HalfPlaneVertexReporter::vertex_offset()`) — that is how
 * the plane's contact element id is reached in the manager's per-vertex `contact_element_ids`
 * array, and it is the only place this model differs structurally from the simplex one.
 * Because there is a single pair per thread, the reference writes one doublet
 * (`Grad(I).write(vI, G)`) and one triplet (`Hess(I).write(vI, vI, H)`) with the views'
 * `operator()` — the port builds the device views with `from_handles_full` from the flat slice
 * handles and does the same.
 * `gradient_only` travels as the reference's `bool` argument; on the device it is a predicate,
 * so the port passes a `uint32` and branches with `$if` (RULE M2).
 *********************************************************************/
#include <contact_system/vertex_half_plane_normal_contact.h>
#include <implicit_geometry/half_plane.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <kernel_cout.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace (utils/make_spd.h)
#include <pipeline/ipc_pipeline_flag.h>

namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

/// `do_compute_energy_kernel`.
auto& do_compute_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<Float>,
                                   lcrt::Buffer<LVector2i>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   std::int32_t,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   std::uint32_t>(
            [](lcrt::BufferVar<Float>        Es,
               lcrt::BufferVar<LVector2i>    PHs,
               lcrt::BufferVar<LVector3>     plane_positions,
               lcrt::BufferVar<LVector3>     plane_normals,
               lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::Var<std::int32_t>       half_plane_vertex_offset,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<Float>              dt,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    using namespace sym::ipc_vertex_half_contact;

                    auto PH = load_vec2i(PHs->read(I));

                    auto vI = PH(0);
                    auto HI = PH(1);

                    auto d_hat = d_hats->read(vI);

                    auto v = load_vec3(Ps->read(vI));
                    auto P = load_vec3(plane_positions->read(HI));
                    auto N = load_vec3(plane_normals->read(HI));

                    auto kt2 = lps::detail::mul(
                        lps::detail::mul(
                            table->read(contact_ids->read(vI) * table_size
                                        + contact_ids->read(HI + half_plane_vertex_offset))
                                .kappa,
                            dt),
                        dt);

                    auto thickness = thicknesses->read(vI);

                    Es->write(I, PH_barrier_energy(kt2, d_hat, thickness, v, P, N));
                };
            });
    return k;
}

/// `do_assemble_kernel`.
auto& do_assemble_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<Float>,
                                   std::uint32_t,
                                   lcrt::Buffer<LVector2i>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   lcrt::Buffer<IndexT>,
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   lcrt::Buffer<Float>,
                                   std::int32_t,
                                   Float,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t,
                                   std::uint32_t>(
            [](lcrt::BufferVar<IndexT>       Grad_indices,
               lcrt::BufferVar<Float>        Grad_values,
               lcrt::Var<std::uint32_t>      Grad_count,
               lcrt::BufferVar<IndexT>       Hess_rows,
               lcrt::BufferVar<IndexT>       Hess_cols,
               lcrt::BufferVar<Float>        Hess_values,
               lcrt::Var<std::uint32_t>      Hess_count,
               lcrt::BufferVar<LVector2i>    PHs,
               lcrt::BufferVar<LVector3>     plane_positions,
               lcrt::BufferVar<LVector3>     plane_normals,
               lcrt::BufferVar<ContactCoeff> table,
               lcrt::Var<std::int32_t>       table_size,
               lcrt::BufferVar<IndexT>       contact_ids,
               lcrt::BufferVar<LVector3>     Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<std::int32_t>       half_plane_vertex_offset,
               lcrt::Var<Float>              dt,
               lcrt::Var<std::uint32_t>      gradient_only,
               lcrt::Var<std::uint32_t>      n_segments,
               lcrt::Var<std::uint32_t>      G_offset,
               lcrt::Var<std::uint32_t>      G_total,
               lcrt::Var<std::uint32_t>      H_offset,
               lcrt::Var<std::uint32_t>      H_total,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    using namespace sym::ipc_vertex_half_contact;

                    auto PH = load_vec2i(PHs->read(I));

                    auto vI = PH(0);
                    auto HI = PH(1);

                    auto v = load_vec3(Ps->read(vI));
                    auto P = load_vec3(plane_positions->read(HI));
                    auto N = load_vec3(plane_normals->read(HI));

                    auto d_hat = d_hats->read(vI);

                    auto kt2 = lps::detail::mul(
                        lps::detail::mul(
                            table->read(contact_ids->read(vI) * table_size
                                        + contact_ids->read(HI + half_plane_vertex_offset))
                                .kappa,
                            dt),
                        dt);

                    auto thickness = thicknesses->read(vI);

                    auto G   = lps::LVec<lcrt::Var<Float>, 3>();
                    auto D_G = lla::LDoubletVectorDeviceView<lcrt::Var<Float>, 3>::
                        from_handles_full(Grad_indices,
                                          Grad_values,
                                          n_segments,
                                          G_offset,
                                          Grad_count,
                                          G_total,
                                          0u,
                                          n_segments);
                    $if(gradient_only != 0u)
                    {
                        PH_barrier_gradient(G, kt2, d_hat, thickness, v, P, N);
                        D_G(I).write(vI, G);
                    }
                    $else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 3, 3>();
                        PH_barrier_gradient_hessian(G, H, kt2, d_hat, thickness, v, P, N);
                        D_G(I).write(vI, G);
                        auto T_H = lla::LTripletMatrixDeviceView<lcrt::Var<Float>, 3, 3>::
                            from_handles_full(Hess_rows,
                                              Hess_cols,
                                              Hess_values,
                                              n_segments,
                                              n_segments,
                                              H_offset,
                                              Hess_count,
                                              H_total,
                                              0u,
                                              0u,
                                              n_segments,
                                              n_segments);
                        T_H(I).write(vI, vI, H);
                    };
                };
            });
    return k;
}

}  // namespace

class IPCVertexHalfPlaneNormalContact final : public VertexHalfPlaneNormalContact
{
  public:
    using VertexHalfPlaneNormalContact::VertexHalfPlaneNormalContact;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();
        half_plane = &require<HalfPlane>();
    }

    virtual void do_compute_energy(EnergyInfo& info)
    {
        if(info.PHs().size() > 0)
        {
            auto& k = do_compute_energy_kernel();
            auto  n = static_cast<std::uint32_t>(info.PHs().size());
            frame_stream()
                .record(k(info.energies(),
                          info.PHs(),
                          half_plane->positions(),
                          half_plane->normals(),
                          info.contact_tabular(),
                          static_cast<std::int32_t>(info.contact_tabular_size()),
                          info.contact_element_ids(),
                          info.positions(),
                          info.thicknesses(),
                          static_cast<std::int32_t>(info.half_plane_vertex_offset()),
                          info.d_hats(),
                          info.dt(),
                          n)
                             .dispatch(n));
        }
    }

    virtual void do_assemble(ContactInfo& info)
    {
        if(info.PHs().size())
        {
            auto& k = do_assemble_kernel();
            auto  n = static_cast<std::uint32_t>(info.PHs().size());
            frame_stream()
                .record(k(info.gradients().indices(),
                          info.gradients().values(),
                          static_cast<std::uint32_t>(info.gradients().doublet_count()),
                          info.hessians().row_indices(),
                          info.hessians().col_indices(),
                          info.hessians().values(),
                          static_cast<std::uint32_t>(info.hessians().triplet_count()),
                          info.PHs(),
                          half_plane->positions(),
                          half_plane->normals(),
                          info.contact_tabular(),
                          static_cast<std::int32_t>(info.contact_tabular_size()),
                          info.contact_element_ids(),
                          info.positions(),
                          info.thicknesses(),
                          info.d_hats(),
                          static_cast<std::int32_t>(info.half_plane_vertex_offset()),
                          info.dt(),
                          info.gradient_only() ? 1u : 0u,
                          static_cast<std::uint32_t>(info.gradients().total_extent()),
                          static_cast<std::uint32_t>(info.gradients().doublet_index_offset()),
                          static_cast<std::uint32_t>(info.gradients().total_doublet_count()),
                          static_cast<std::uint32_t>(info.hessians().triplet_index_offset()),
                          static_cast<std::uint32_t>(info.hessians().total_triplet_count()),
                          n)
                             .dispatch(n));
        }
    }

    HalfPlane* half_plane = nullptr;
};

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneNormalContact);
}  // namespace uipc::backend::luisa
