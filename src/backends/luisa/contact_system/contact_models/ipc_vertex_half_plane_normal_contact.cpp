#include <contact_system/vertex_half_plane_normal_contact.h>
#include <implicit_geometry/half_plane.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <utils/make_spd.h>

namespace uipc::backend::luisa
{
using namespace sym::ipc_vertex_half_contact;

class IPCVertexHalfPlaneNormalContact final : public VertexHalfPlaneNormalContact
{
  public:
    using VertexHalfPlaneNormalContact::VertexHalfPlaneNormalContact;

    void do_build(BuildInfo& info) override
    {
        auto constitution =
            world().scene().config().find<std::string>("contact/constitution");
        if(constitution->view()[0] != "ipc")
        {
            throw SimSystemException("Constitution is not IPC");
        }

        half_plane = &require<HalfPlane>();
    }

    void do_compute_energy(EnergyInfo& info) override
    {
        // Get counts
        SizeT ph_count = info.PHs().size();

        if(ph_count == 0)
            return;

        // Get device
        auto& engine = world().sim_engine();
        auto& device = static_cast<SimEngine&>(engine).device();
        auto  stream = static_cast<SimEngine&>(engine).compute_stream();

        // Get buffer views
        auto contact_tabular_view = info.contact_tabular();
        auto positions_view       = info.positions();
        auto thicknesses_view     = info.thicknesses();
        auto d_hats_view          = info.d_hats();
        auto contact_element_ids_view = info.contact_element_ids();
        auto PHs_view             = info.PHs();
        auto energies_view        = info.energies();
        auto plane_positions_view = half_plane->positions();
        auto plane_normals_view   = half_plane->normals();

        Float dt = info.dt();
        IndexT half_plane_vertex_offset = info.half_plane_vertex_offset();

        // PH barrier energy kernel
        Kernel1D ph_energy_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                        BufferVar<Vector3> positions,
                                        BufferVar<Float> thicknesses,
                                        BufferVar<Float> d_hats,
                                        BufferVar<IndexT> contact_element_ids,
                                        BufferVar<Vector2i> PHs,
                                        BufferVar<Vector3> plane_positions,
                                        BufferVar<Vector3> plane_normals,
                                        BufferVar<Float> energies,
                                        Float dt,
                                        IndexT half_plane_vertex_offset)
        {
            auto I = dispatch_x();
            $if(I < ph_count)
            {
                Vector2i PH = PHs.read(I);

                IndexT vI = PH[0];
                IndexT HI = PH[1];

                Float d_hat = d_hats.read(vI);

                Vector3 v = positions.read(vI);
                Vector3 P = plane_positions.read(HI);
                Vector3 N = plane_normals.read(HI);

                ContactCoeff coeff = contact_tabular.read(
                    contact_element_ids.read(vI) * contact_tabular.size() + 
                    contact_element_ids.read(HI + half_plane_vertex_offset));
                Float kt2 = coeff.kappa * dt * dt;

                Float thickness = thicknesses.read(vI);

                Float E = PH_barrier_energy(kt2, d_hat, thickness, v, P, N);
                energies.write(I, E);
            };
        };

        auto shader = device.compile(ph_energy_kernel);
        stream << shader(contact_tabular_view,
                         positions_view,
                         thicknesses_view,
                         d_hats_view,
                         contact_element_ids_view,
                         PHs_view,
                         plane_positions_view,
                         plane_normals_view,
                         energies_view,
                         dt,
                         half_plane_vertex_offset)
                      .dispatch(ph_count);
    }

    void do_assemble(ContactInfo& info) override
    {
        SizeT ph_count = info.PHs().size();

        if(ph_count == 0)
            return;

        // Get device
        auto& engine = world().sim_engine();
        auto& device = static_cast<SimEngine&>(engine).device();
        auto  stream = static_cast<SimEngine&>(engine).compute_stream();

        // Get buffer views
        auto contact_tabular_view = info.contact_tabular();
        auto positions_view       = info.positions();
        auto thicknesses_view     = info.thicknesses();
        auto d_hats_view          = info.d_hats();
        auto contact_element_ids_view = info.contact_element_ids();
        auto PHs_view             = info.PHs();
        auto gradients_view       = info.gradients();
        auto hessians_view        = info.hessians();
        auto plane_positions_view = half_plane->positions();
        auto plane_normals_view   = half_plane->normals();

        Float dt = info.dt();
        IndexT half_plane_vertex_offset = info.half_plane_vertex_offset();
        bool gradient_only = info.gradient_only();

        // PH barrier assembly kernel
        // Gradient is stored as (index, x, y, z) tuples in Float array: 4 floats per entry
        // Hessian is stored as (row, col, 3x3 matrix) in Float array: 11 floats per entry (2 indices + 9 matrix elements)
        Kernel1D ph_assemble_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                          BufferVar<Vector3> positions,
                                          BufferVar<Float> thicknesses,
                                          BufferVar<Float> d_hats,
                                          BufferVar<IndexT> contact_element_ids,
                                          BufferVar<Vector2i> PHs,
                                          BufferVar<Float> gradients,
                                          BufferVar<Float> hessians,
                                          BufferVar<Vector3> plane_positions,
                                          BufferVar<Vector3> plane_normals,
                                          Float dt,
                                          IndexT half_plane_vertex_offset,
                                          Bool grad_only)
        {
            auto I = dispatch_x();
            $if(I < ph_count)
            {
                Vector2i PH = PHs.read(I);

                IndexT vI = PH[0];
                IndexT HI = PH[1];

                Vector3 v = positions.read(vI);
                Vector3 P = plane_positions.read(HI);
                Vector3 N = plane_normals.read(HI);

                Float d_hat = d_hats.read(vI);

                ContactCoeff coeff = contact_tabular.read(
                    contact_element_ids.read(vI) * contact_tabular.size() + 
                    contact_element_ids.read(HI + half_plane_vertex_offset));
                Float kt2 = coeff.kappa * dt * dt;

                Float thickness = thicknesses.read(vI);

                Vector3 G;
                $if(grad_only)
                {
                    PH_barrier_gradient(G, kt2, d_hat, thickness, v, P, N);
                    
                    // Write gradient: (index, x, y, z) at position I * 4
                    IndexT base_idx = I * 4;
                    gradients.write(base_idx + 0, cast<Float>(vI));
                    gradients.write(base_idx + 1, G.x);
                    gradients.write(base_idx + 2, G.y);
                    gradients.write(base_idx + 3, G.z);
                }
                $else
                {
                    Matrix3x3 H;
                    PH_barrier_gradient_hessian(G, H, kt2, d_hat, thickness, v, P, N);
                    make_spd(H);
                    
                    // Write gradient: (index, x, y, z) at position I * 4
                    IndexT grad_base = I * 4;
                    gradients.write(grad_base + 0, cast<Float>(vI));
                    gradients.write(grad_base + 1, G.x);
                    gradients.write(grad_base + 2, G.y);
                    gradients.write(grad_base + 3, G.z);
                    
                    // Write Hessian: (row, col, 3x3 matrix) at position I * 11
                    IndexT hess_base = I * 11;
                    hessians.write(hess_base + 0, cast<Float>(vI));  // row
                    hessians.write(hess_base + 1, cast<Float>(vI));  // col (diagonal)
                    // Matrix in row-major order
                    hessians.write(hess_base + 2, H[0][0]);
                    hessians.write(hess_base + 3, H[0][1]);
                    hessians.write(hess_base + 4, H[0][2]);
                    hessians.write(hess_base + 5, H[1][0]);
                    hessians.write(hess_base + 6, H[1][1]);
                    hessians.write(hess_base + 7, H[1][2]);
                    hessians.write(hess_base + 8, H[2][0]);
                    hessians.write(hess_base + 9, H[2][1]);
                    hessians.write(hess_base + 10, H[2][2]);
                };
            };
        };

        auto shader = device.compile(ph_assemble_kernel);
        stream << shader(contact_tabular_view,
                         positions_view,
                         thicknesses_view,
                         d_hats_view,
                         contact_element_ids_view,
                         PHs_view,
                         gradients_view,
                         hessians_view,
                         plane_positions_view,
                         plane_normals_view,
                         dt,
                         half_plane_vertex_offset,
                         gradient_only)
                      .dispatch(ph_count);
    }

    HalfPlane* half_plane = nullptr;
};

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneNormalContact);
}  // namespace uipc::backend::luisa
