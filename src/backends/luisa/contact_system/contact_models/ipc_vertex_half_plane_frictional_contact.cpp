/********************************************************************
 * @file contact_models/ipc_vertex_half_plane_frictional_contact.cpp
 * @brief T8.1b — `IPCVertexHalfPlaneFrictionalContact`: the concrete IPC vertex/half-plane
 * frictional contact model (port of
 * src/backends/cuda/contact_system/contact_models/ipc_vertex_half_plane_frictional_contact.cu,
 * 189 reference lines).
 *
 * The frictional twin of ipc_vertex_half_plane_normal_contact.cpp: the same two kernels over
 * `friction_PHs()`, but the pair's *friction* energy / gradient / Hessian
 * (`PH_friction_energy{,_gradient,_gradient_hessian}` of
 * `ipc_vertex_half_plane_contact_function.h`) with the previous-step vertex position
 * (`info.prev_positions()`), the pair's `mu` from the contact table and `eps_vh =
 * eps_velocity() * dt`.  The reference's `coeff = table(...)` reads *both* members at once; the
 * port reads the flat table entry once and takes `.kappa`/`.mu` from it, exactly like the
 * reference's single read.  The `3x3` Hessian is `make_spd`-projected after the friction
 * Hessian, as in the reference.
 *********************************************************************/
#include <contact_system/vertex_half_plane_frictional_contact.h>
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

/// The reference's `cuda::make_spd(H)` on the 3x3 friction Hessian: the port's fixed-sweep
/// `lps::make_spd_inplace` (the reference's `utils/make_spd.h` loop runs an eigensolver; the
/// T5.2 measurement — `hs1d_detail::kMakeSpdSweeps` — is the port's 8).
inline constexpr int kMakeSpdSweeps = 8;

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
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   Float,
                                   lcrt::Buffer<Float>,
                                   std::int32_t,
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
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::Var<Float>              eps_v,
               lcrt::BufferVar<Float>        d_hats,
               lcrt::Var<std::int32_t>       half_plane_vertex_offset,
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

                    auto v      = load_vec3(Ps->read(vI));
                    auto prev_v = load_vec3(prev_Ps->read(vI));
                    auto P      = load_vec3(plane_positions->read(HI));
                    auto N      = load_vec3(plane_normals->read(HI));

                    auto d_hat = d_hats->read(vI);

                    auto coeff = table->read(
                        contact_ids->read(vI) * table_size
                        + contact_ids->read(HI + half_plane_vertex_offset));
                    auto kt2 = lps::detail::mul(lps::detail::mul(coeff.kappa, dt), dt);
                    auto mu  = coeff.mu;

                    auto thickness = thicknesses->read(vI);

                    Es->write(I,
                              PH_friction_energy(kt2,
                                                 d_hat,
                                                 thickness,
                                                 mu,
                                                 lps::detail::mul(eps_v, dt),
                                                 prev_v,
                                                 v,
                                                 P,
                                                 N));
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
                                   lcrt::Buffer<LVector3>,
                                   lcrt::Buffer<Float>,
                                   Float,
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
               lcrt::BufferVar<LVector3>     prev_Ps,
               lcrt::BufferVar<Float>        thicknesses,
               lcrt::Var<Float>              eps_v,
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

                    auto v      = load_vec3(Ps->read(vI));
                    auto prev_v = load_vec3(prev_Ps->read(vI));
                    auto P      = load_vec3(plane_positions->read(HI));
                    auto N      = load_vec3(plane_normals->read(HI));

                    auto d_hat = d_hats->read(vI);

                    auto coeff = table->read(
                        contact_ids->read(vI) * table_size
                        + contact_ids->read(HI + half_plane_vertex_offset));
                    auto kt2 = lps::detail::mul(lps::detail::mul(coeff.kappa, dt), dt);
                    auto mu  = coeff.mu;

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
                        PH_friction_gradient(
                            G, kt2, d_hat, thickness, mu, lps::detail::mul(eps_v, dt), prev_v, v, P, N);
                        D_G(I).write(vI, G);
                    }
                    $else
                    {
                        auto H = lps::LMat<lcrt::Var<Float>, 3, 3>();
                        PH_friction_gradient_hessian(
                            G, H, kt2, d_hat, thickness, mu, lps::detail::mul(eps_v, dt), prev_v, v, P, N);
                        lps::make_spd_inplace(H, kMakeSpdSweeps);
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

class IPCVertexHalfPlaneFrictionalContact final : public VertexHalfPlaneFrictionalContact
{
  public:
    using VertexHalfPlaneFrictionalContact::VertexHalfPlaneFrictionalContact;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();
        half_plane = &require<HalfPlane>();
    }

    virtual void do_compute_energy(EnergyInfo& info)
    {
        if(info.friction_PHs().size() > 0)
        {
            auto& k = do_compute_energy_kernel();
            auto  n = static_cast<std::uint32_t>(info.friction_PHs().size());
            frame_stream()
                .record(k(info.energies(),
                          info.friction_PHs(),
                          half_plane->positions(),
                          half_plane->normals(),
                          info.contact_tabular(),
                          static_cast<std::int32_t>(info.contact_tabular_size()),
                          info.contact_element_ids(),
                          info.positions(),
                          info.prev_positions(),
                          info.thicknesses(),
                          info.eps_velocity(),
                          info.d_hats(),
                          static_cast<std::int32_t>(info.half_plane_vertex_offset()),
                          info.dt(),
                          n)
                             .dispatch(n));
        }
    }

    virtual void do_assemble(ContactInfo& info)
    {
        if(info.friction_PHs().size())
        {
            auto& k = do_assemble_kernel();
            auto  n = static_cast<std::uint32_t>(info.friction_PHs().size());
            frame_stream()
                .record(k(info.gradients().indices(),
                          info.gradients().values(),
                          static_cast<std::uint32_t>(info.gradients().doublet_count()),
                          info.hessians().row_indices(),
                          info.hessians().col_indices(),
                          info.hessians().values(),
                          static_cast<std::uint32_t>(info.hessians().triplet_count()),
                          info.friction_PHs(),
                          half_plane->positions(),
                          half_plane->normals(),
                          info.contact_tabular(),
                          static_cast<std::int32_t>(info.contact_tabular_size()),
                          info.contact_element_ids(),
                          info.positions(),
                          info.prev_positions(),
                          info.thicknesses(),
                          info.eps_velocity(),
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

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneFrictionalContact);
}  // namespace uipc::backend::luisa
