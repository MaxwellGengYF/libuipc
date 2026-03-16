#include <contact_system/contact_models/ipc_simplex_frictional_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <contact_system/simplex_frictional_contact.h>
#include <utils/codim_thickness.h>
#include <utils/primitive_d_hat.h>
#include <utils/distance/edge_edge.h>

namespace uipc::backend::luisa
{
using namespace sym::codim_ipc_contact;

class IPCSimplexFrictionalContact final : public SimplexFrictionalContact
{
public:
    using SimplexFrictionalContact::SimplexFrictionalContact;

    void do_compute_energy(EnergyInfo& info) override
    {
        // Get counts
        SizeT pt_count = info.friction_PTs().size();
        SizeT ee_count = info.friction_EEs().size();
        SizeT pe_count = info.friction_PEs().size();
        SizeT pp_count = info.friction_PPs().size();
        SizeT total_count = pt_count + ee_count + pe_count + pp_count;

        if(total_count == 0)
            return;

        // Get device
        auto& engine = world().sim_engine();
        auto& device = static_cast<SimEngine&>(engine).device();
        auto  stream = static_cast<SimEngine&>(engine).compute_stream();

        // Get buffer views
        auto contact_tabular_view = info.contact_tabular();
        auto positions_view       = info.positions();
        auto prev_positions_view  = info.prev_positions();
        auto rest_positions_view  = info.rest_positions();
        auto thicknesses_view     = info.thicknesses();
        auto d_hats_view          = info.d_hats();
        auto contact_element_ids_view = info.contact_element_ids();

        // PT data
        auto PTs_view       = info.friction_PTs();
        auto PT_energies_view = info.friction_PT_energies();

        // EE data
        auto EEs_view       = info.friction_EEs();
        auto EE_energies_view = info.friction_EE_energies();

        // PE data
        auto PEs_view       = info.friction_PEs();
        auto PE_energies_view = info.friction_PE_energies();

        // PP data
        auto PPs_view       = info.friction_PPs();
        auto PP_energies_view = info.friction_PP_energies();

        Float dt    = info.dt();
        Float eps_v = info.eps_velocity();

        // Offsets for fused kernel
        IndexT ee_offset = pt_count;
        IndexT pe_offset = ee_offset + ee_count;
        IndexT pp_offset = pe_offset + pe_count;

        // Energy kernel
        Kernel1D energy_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                     BufferVar<Float3> positions,
                                     BufferVar<Float3> prev_positions,
                                     BufferVar<Float3> rest_positions,
                                     BufferVar<Float> thicknesses,
                                     BufferVar<Float> d_hats,
                                     BufferVar<IndexT> contact_element_ids,
                                     BufferVar<Vector4i> PTs,
                                     BufferVar<Float> PT_energies,
                                     BufferVar<Vector4i> EEs,
                                     BufferVar<Float> EE_energies,
                                     BufferVar<Vector3i> PEs,
                                     BufferVar<Float> PE_energies,
                                     BufferVar<Vector2i> PPs,
                                     BufferVar<Float> PP_energies,
                                     Float dt,
                                     Float eps_v,
                                     UInt pt_count,
                                     UInt ee_offset,
                                     UInt pe_offset,
                                     UInt pp_offset)
        {
            auto idx = dispatch_x();

            // PT contact
            if(idx < pt_count)
            {
                auto i = idx;
                auto PT = PTs.read(i);
                Vector4i cids = {contact_element_ids.read(PT[0]), contact_element_ids.read(PT[1]),
                                 contact_element_ids.read(PT[2]), contact_element_ids.read(PT[3])};
                auto coeff = PT_contact_coeff(contact_tabular, cids);
                Float kt2 = coeff.kappa * dt * dt;
                Float mu = coeff.mu;

                auto prev_P  = prev_positions.read(PT[0]);
                auto prev_T0 = prev_positions.read(PT[1]);
                auto prev_T1 = prev_positions.read(PT[2]);
                auto prev_T2 = prev_positions.read(PT[3]);

                auto P  = positions.read(PT[0]);
                auto T0 = positions.read(PT[1]);
                auto T1 = positions.read(PT[2]);
                auto T2 = positions.read(PT[3]);

                Float thickness_P  = thicknesses.read(PT[0]);
                Float thickness_T0 = thicknesses.read(PT[1]);
                Float thickness_T1 = thicknesses.read(PT[2]);
                Float thickness_T2 = thicknesses.read(PT[3]);
                Float thickness = PT_thickness(thickness_P, thickness_T0, thickness_T1, thickness_T2);

                Float d_hat_P  = d_hats.read(PT[0]);
                Float d_hat_T0 = d_hats.read(PT[1]);
                Float d_hat_T1 = d_hats.read(PT[2]);
                Float d_hat_T2 = d_hats.read(PT[3]);
                Float d_hat = PT_d_hat(d_hat_P, d_hat_T0, d_hat_T1, d_hat_T2);

                Float eps_vh = eps_v * dt;

                auto E = PT_friction_energy(
                    kt2, d_hat, thickness, mu, eps_vh,
                    Float3{prev_P.x, prev_P.y, prev_P.z},
                    Float3{prev_T0.x, prev_T0.y, prev_T0.z},
                    Float3{prev_T1.x, prev_T1.y, prev_T1.z},
                    Float3{prev_T2.x, prev_T2.y, prev_T2.z},
                    Float3{P.x, P.y, P.z},
                    Float3{T0.x, T0.y, T0.z},
                    Float3{T1.x, T1.y, T1.z},
                    Float3{T2.x, T2.y, T2.z});

                PT_energies.write(i, E);
            }
            // EE contact
            else if(idx < ee_offset)
            {
                auto i = idx - ee_offset;
                auto EE = EEs.read(i);
                Vector4i cids = {contact_element_ids.read(EE[0]), contact_element_ids.read(EE[1]),
                                 contact_element_ids.read(EE[2]), contact_element_ids.read(EE[3])};
                auto coeff = EE_contact_coeff(contact_tabular, cids);
                Float kt2 = coeff.kappa * dt * dt;
                Float mu = coeff.mu;

                auto rest_Ea0 = rest_positions.read(EE[0]);
                auto rest_Ea1 = rest_positions.read(EE[1]);
                auto rest_Eb0 = rest_positions.read(EE[2]);
                auto rest_Eb1 = rest_positions.read(EE[3]);

                auto prev_Ea0 = prev_positions.read(EE[0]);
                auto prev_Ea1 = prev_positions.read(EE[1]);
                auto prev_Eb0 = prev_positions.read(EE[2]);
                auto prev_Eb1 = prev_positions.read(EE[3]);

                auto Ea0 = positions.read(EE[0]);
                auto Ea1 = positions.read(EE[1]);
                auto Eb0 = positions.read(EE[2]);
                auto Eb1 = positions.read(EE[3]);

                Float thickness_Ea0 = thicknesses.read(EE[0]);
                Float thickness_Ea1 = thicknesses.read(EE[1]);
                Float thickness_Eb0 = thicknesses.read(EE[2]);
                Float thickness_Eb1 = thicknesses.read(EE[3]);
                Float thickness = EE_thickness(thickness_Ea0, thickness_Ea1, thickness_Eb0, thickness_Eb1);

                Float d_hat_Ea0 = d_hats.read(EE[0]);
                Float d_hat_Ea1 = d_hats.read(EE[1]);
                Float d_hat_Eb0 = d_hats.read(EE[2]);
                Float d_hat_Eb1 = d_hats.read(EE[3]);
                Float d_hat = EE_d_hat(d_hat_Ea0, d_hat_Ea1, d_hat_Eb0, d_hat_Eb1);

                Float eps_vh = eps_v * dt;

                // Check mollifier
                Float eps_x;
                distance::edge_edge_mollifier_threshold(
                    Float3{rest_Ea0.x, rest_Ea0.y, rest_Ea0.z},
                    Float3{rest_Ea1.x, rest_Ea1.y, rest_Ea1.z},
                    Float3{rest_Eb0.x, rest_Eb0.y, rest_Eb0.z},
                    Float3{rest_Eb1.x, rest_Eb1.y, rest_Eb1.z},
                    1e-3f, eps_x);
                bool mollified = distance::need_mollify(
                    Float3{prev_Ea0.x, prev_Ea0.y, prev_Ea0.z},
                    Float3{prev_Ea1.x, prev_Ea1.y, prev_Ea1.z},
                    Float3{prev_Eb0.x, prev_Eb0.y, prev_Eb0.z},
                    Float3{prev_Eb1.x, prev_Eb1.y, prev_Eb1.z},
                    eps_x);

                Float E = 0.0f;
                if(!mollified)
                {
                    E = EE_friction_energy(
                        kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_Ea0.x, prev_Ea0.y, prev_Ea0.z},
                        Float3{prev_Ea1.x, prev_Ea1.y, prev_Ea1.z},
                        Float3{prev_Eb0.x, prev_Eb0.y, prev_Eb0.z},
                        Float3{prev_Eb1.x, prev_Eb1.y, prev_Eb1.z},
                        Float3{Ea0.x, Ea0.y, Ea0.z},
                        Float3{Ea1.x, Ea1.y, Ea1.z},
                        Float3{Eb0.x, Eb0.y, Eb0.z},
                        Float3{Eb1.x, Eb1.y, Eb1.z});
                }

                EE_energies.write(i, E);
            }
            // PE contact
            else if(idx < pe_offset)
            {
                auto i = idx - pe_offset;
                auto PE = PEs.read(i);
                Vector3i cids = {contact_element_ids.read(PE[0]), contact_element_ids.read(PE[1]), 
                                 contact_element_ids.read(PE[2])};
                auto coeff = PE_contact_coeff(contact_tabular, cids);
                Float kt2 = coeff.kappa * dt * dt;
                Float mu = coeff.mu;

                auto prev_P  = prev_positions.read(PE[0]);
                auto prev_E0 = prev_positions.read(PE[1]);
                auto prev_E1 = prev_positions.read(PE[2]);

                auto P  = positions.read(PE[0]);
                auto E0 = positions.read(PE[1]);
                auto E1 = positions.read(PE[2]);

                Float thickness_P  = thicknesses.read(PE[0]);
                Float thickness_E0 = thicknesses.read(PE[1]);
                Float thickness_E1 = thicknesses.read(PE[2]);
                Float thickness = PE_thickness(thickness_P, thickness_E0, thickness_E1);

                Float d_hat_P  = d_hats.read(PE[0]);
                Float d_hat_E0 = d_hats.read(PE[1]);
                Float d_hat_E1 = d_hats.read(PE[2]);
                Float d_hat = PE_d_hat(d_hat_P, d_hat_E0, d_hat_E1);

                Float eps_vh = eps_v * dt;

                auto E = PE_friction_energy(
                    kt2, d_hat, thickness, mu, eps_vh,
                    Float3{prev_P.x, prev_P.y, prev_P.z},
                    Float3{prev_E0.x, prev_E0.y, prev_E0.z},
                    Float3{prev_E1.x, prev_E1.y, prev_E1.z},
                    Float3{P.x, P.y, P.z},
                    Float3{E0.x, E0.y, E0.z},
                    Float3{E1.x, E1.y, E1.z});

                PE_energies.write(i, E);
            }
            // PP contact
            else if(idx < pp_offset)
            {
                auto i = idx - pp_offset;
                auto PP = PPs.read(i);
                Vector2i cids = {contact_element_ids.read(PP[0]), contact_element_ids.read(PP[1])};
                auto coeff = PP_contact_coeff(contact_tabular, cids);
                Float kt2 = coeff.kappa * dt * dt;
                Float mu = coeff.mu;

                auto prev_P0 = prev_positions.read(PP[0]);
                auto prev_P1 = prev_positions.read(PP[1]);

                auto P0 = positions.read(PP[0]);
                auto P1 = positions.read(PP[1]);

                Float thickness_P0 = thicknesses.read(PP[0]);
                Float thickness_P1 = thicknesses.read(PP[1]);
                Float thickness = PP_thickness(thickness_P0, thickness_P1);

                Float d_hat_P0 = d_hats.read(PP[0]);
                Float d_hat_P1 = d_hats.read(PP[1]);
                Float d_hat = PP_d_hat(d_hat_P0, d_hat_P1);

                Float eps_vh = eps_v * dt;

                auto E = PP_friction_energy(
                    kt2, d_hat, thickness, mu, eps_vh,
                    Float3{prev_P0.x, prev_P0.y, prev_P0.z},
                    Float3{prev_P1.x, prev_P1.y, prev_P1.z},
                    Float3{P0.x, P0.y, P0.z},
                    Float3{P1.x, P1.y, P1.z});

                PP_energies.write(i, E);
            }
        };

        auto shader = device.compile(energy_kernel);
        stream << shader(contact_tabular_view,
                         positions_view,
                         prev_positions_view,
                         rest_positions_view,
                         thicknesses_view,
                         d_hats_view,
                         contact_element_ids_view,
                         PTs_view,
                         PT_energies_view,
                         EEs_view,
                         EE_energies_view,
                         PEs_view,
                         PE_energies_view,
                         PPs_view,
                         PP_energies_view,
                         dt,
                         eps_v,
                         static_cast<UInt>(pt_count),
                         static_cast<UInt>(ee_offset),
                         static_cast<UInt>(pe_offset),
                         static_cast<UInt>(pp_offset))
                      .dispatch(total_count);
    }

    void do_assemble(ContactInfo& info) override
    {
        // Get counts
        SizeT pt_count = info.friction_PTs().size();
        SizeT ee_count = info.friction_EEs().size();
        SizeT pe_count = info.friction_PEs().size();
        SizeT pp_count = info.friction_PPs().size();
        SizeT total_count = pt_count + ee_count + pe_count + pp_count;

        if(total_count == 0)
            return;

        // Get device
        auto& engine = world().sim_engine();
        auto& device = static_cast<SimEngine&>(engine).device();
        auto  stream = static_cast<SimEngine&>(engine).compute_stream();

        // Get buffer views
        auto contact_tabular_view = info.contact_tabular();
        auto positions_view       = info.positions();
        auto prev_positions_view  = info.prev_positions();
        auto rest_positions_view  = info.rest_positions();
        auto thicknesses_view     = info.thicknesses();
        auto d_hats_view          = info.d_hats();
        auto contact_element_ids_view = info.contact_element_ids();

        // PT data
        auto PTs_view       = info.friction_PTs();
        auto PT_gradients_view = info.friction_PT_gradients();
        auto PT_hessians_view = info.friction_PT_hessians();

        // EE data
        auto EEs_view       = info.friction_EEs();
        auto EE_gradients_view = info.friction_EE_gradients();
        auto EE_hessians_view = info.friction_EE_hessians();

        // PE data
        auto PEs_view       = info.friction_PEs();
        auto PE_gradients_view = info.friction_PE_gradients();
        auto PE_hessians_view = info.friction_PE_hessians();

        // PP data
        auto PPs_view       = info.friction_PPs();
        auto PP_gradients_view = info.friction_PP_gradients();
        auto PP_hessians_view = info.friction_PP_hessians();

        Float dt    = info.dt();
        Float eps_v = info.eps_velocity();
        bool gradient_only = info.gradient_only();

        // Offsets for fused kernel
        IndexT ee_offset = pt_count;
        IndexT pe_offset = ee_offset + ee_count;
        IndexT pp_offset = pe_offset + pe_count;

        // Assemble kernel - write directly to output buffers
        Kernel1D assemble_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                       BufferVar<Float3> positions,
                                       BufferVar<Float3> prev_positions,
                                       BufferVar<Float3> rest_positions,
                                       BufferVar<Float> thicknesses,
                                       BufferVar<Float> d_hats,
                                       BufferVar<IndexT> contact_element_ids,
                                       BufferVar<Vector4i> PTs,
                                       BufferVar<DoubletVector3> PT_grads,
                                       BufferVar<TripletMatrix3> PT_hess,
                                       BufferVar<Vector4i> EEs,
                                       BufferVar<DoubletVector3> EE_grads,
                                       BufferVar<TripletMatrix3> EE_hess,
                                       BufferVar<Vector3i> PEs,
                                       BufferVar<DoubletVector3> PE_grads,
                                       BufferVar<TripletMatrix3> PE_hess,
                                       BufferVar<Vector2i> PPs,
                                       BufferVar<DoubletVector3> PP_grads,
                                       BufferVar<TripletMatrix3> PP_hess,
                                       Float dt,
                                       Float eps_v,
                                       Bool grad_only,
                                       UInt pt_count,
                                       UInt ee_offset,
                                       UInt pe_offset,
                                       UInt pp_offset)
        {
            auto idx = dispatch_x();

            // PT contact
            if(idx < pt_count)
            {
                auto i = idx;
                auto PT = PTs.read(i);
                Vector4i cids = {contact_element_ids.read(PT[0]), contact_element_ids.read(PT[1]),
                                 contact_element_ids.read(PT[2]), contact_element_ids.read(PT[3])};
                auto coeff = PT_contact_coeff(contact_tabular, cids);
                Float kt2 = coeff.kappa * dt * dt;
                Float mu = coeff.mu;

                auto prev_P  = prev_positions.read(PT[0]);
                auto prev_T0 = prev_positions.read(PT[1]);
                auto prev_T1 = prev_positions.read(PT[2]);
                auto prev_T2 = prev_positions.read(PT[3]);

                auto P  = positions.read(PT[0]);
                auto T0 = positions.read(PT[1]);
                auto T1 = positions.read(PT[2]);
                auto T2 = positions.read(PT[3]);

                Float thickness_P  = thicknesses.read(PT[0]);
                Float thickness_T0 = thicknesses.read(PT[1]);
                Float thickness_T1 = thicknesses.read(PT[2]);
                Float thickness_T2 = thicknesses.read(PT[3]);
                Float thickness = PT_thickness(thickness_P, thickness_T0, thickness_T1, thickness_T2);

                Float d_hat_P  = d_hats.read(PT[0]);
                Float d_hat_T0 = d_hats.read(PT[1]);
                Float d_hat_T1 = d_hats.read(PT[2]);
                Float d_hat_T2 = d_hats.read(PT[3]);
                Float d_hat = PT_d_hat(d_hat_P, d_hat_T0, d_hat_T1, d_hat_T2);

                Float eps_vh = eps_v * dt;

                Vector12 G;
                Matrix12x12 H;
                
                if(grad_only)
                {
                    PT_friction_gradient(G, kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_P.x, prev_P.y, prev_P.z},
                        Float3{prev_T0.x, prev_T0.y, prev_T0.z},
                        Float3{prev_T1.x, prev_T1.y, prev_T1.z},
                        Float3{prev_T2.x, prev_T2.y, prev_T2.z},
                        Float3{P.x, P.y, P.z},
                        Float3{T0.x, T0.y, T0.z},
                        Float3{T1.x, T1.y, T1.z},
                        Float3{T2.x, T2.y, T2.z});
                }
                else
                {
                    PT_friction_gradient_hessian(G, H, kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_P.x, prev_P.y, prev_P.z},
                        Float3{prev_T0.x, prev_T0.y, prev_T0.z},
                        Float3{prev_T1.x, prev_T1.y, prev_T1.z},
                        Float3{prev_T2.x, prev_T2.y, prev_T2.z},
                        Float3{P.x, P.y, P.z},
                        Float3{T0.x, T0.y, T0.z},
                        Float3{T1.x, T1.y, T1.z},
                        Float3{T2.x, T2.y, T2.z});
                    
                    // Make SPD
                    friction::make_spd(H);
                }

                // Write gradient
                $for(jj, 4) {
                    DoubletVector3 grad;
                    grad.index = PT[jj];
                    grad.value = Vector3{G[jj * 3 + 0], G[jj * 3 + 1], G[jj * 3 + 2]};
                    PT_grads.write(i * 4 + jj, grad);
                };

                // Write Hessian
                $if(!grad_only) {
                    $for(jj, 4) {
                        $for(kk, 4) {
                            if(jj <= kk) {
                                // Compute linear index for upper triangular
                                IndexT block_idx;
                                if(jj == 0) {
                                    block_idx = kk;
                                } else if(jj == 1) {
                                    block_idx = 4 + (kk - 1);
                                } else if(jj == 2) {
                                    block_idx = 7 + (kk - 2);
                                } else {
                                    block_idx = 9;
                                }
                                
                                TripletMatrix3 hess;
                                hess.row = PT[jj];
                                hess.col = PT[kk];
                                for(int r = 0; r < 3; ++r) {
                                    for(int c = 0; c < 3; ++c) {
                                        hess.value[r][c] = H[jj * 3 + r][kk * 3 + c];
                                    }
                                }
                                PT_hess.write(i * 10 + block_idx, hess);
                            }
                        };
                    };
                };
            }
            // EE contact
            else if(idx < ee_offset)
            {
                auto i = idx - ee_offset;
                auto EE = EEs.read(i);
                Vector4i cids = {contact_element_ids.read(EE[0]), contact_element_ids.read(EE[1]),
                                 contact_element_ids.read(EE[2]), contact_element_ids.read(EE[3])};
                auto coeff = EE_contact_coeff(contact_tabular, cids);
                Float kt2 = coeff.kappa * dt * dt;
                Float mu = coeff.mu;

                auto rest_Ea0 = rest_positions.read(EE[0]);
                auto rest_Ea1 = rest_positions.read(EE[1]);
                auto rest_Eb0 = rest_positions.read(EE[2]);
                auto rest_Eb1 = rest_positions.read(EE[3]);

                auto prev_Ea0 = prev_positions.read(EE[0]);
                auto prev_Ea1 = prev_positions.read(EE[1]);
                auto prev_Eb0 = prev_positions.read(EE[2]);
                auto prev_Eb1 = prev_positions.read(EE[3]);

                auto Ea0 = positions.read(EE[0]);
                auto Ea1 = positions.read(EE[1]);
                auto Eb0 = positions.read(EE[2]);
                auto Eb1 = positions.read(EE[3]);

                Float thickness_Ea0 = thicknesses.read(EE[0]);
                Float thickness_Ea1 = thicknesses.read(EE[1]);
                Float thickness_Eb0 = thicknesses.read(EE[2]);
                Float thickness_Eb1 = thicknesses.read(EE[3]);
                Float thickness = EE_thickness(thickness_Ea0, thickness_Ea1, thickness_Eb0, thickness_Eb1);

                Float d_hat_Ea0 = d_hats.read(EE[0]);
                Float d_hat_Ea1 = d_hats.read(EE[1]);
                Float d_hat_Eb0 = d_hats.read(EE[2]);
                Float d_hat_Eb1 = d_hats.read(EE[3]);
                Float d_hat = EE_d_hat(d_hat_Ea0, d_hat_Ea1, d_hat_Eb0, d_hat_Eb1);

                Float eps_vh = eps_v * dt;

                // Check mollifier
                Float eps_x;
                distance::edge_edge_mollifier_threshold(
                    Float3{rest_Ea0.x, rest_Ea0.y, rest_Ea0.z},
                    Float3{rest_Ea1.x, rest_Ea1.y, rest_Ea1.z},
                    Float3{rest_Eb0.x, rest_Eb0.y, rest_Eb0.z},
                    Float3{rest_Eb1.x, rest_Eb1.y, rest_Eb1.z},
                    1e-3f, eps_x);
                bool mollified = distance::need_mollify(
                    Float3{prev_Ea0.x, prev_Ea0.y, prev_Ea0.z},
                    Float3{prev_Ea1.x, prev_Ea1.y, prev_Ea1.z},
                    Float3{prev_Eb0.x, prev_Eb0.y, prev_Eb0.z},
                    Float3{prev_Eb1.x, prev_Eb1.y, prev_Eb1.z},
                    eps_x);

                Vector12 G;
                Matrix12x12 H;
                
                if(mollified)
                {
                    G = Vector12::Zero();
                    H = Matrix12x12::Zero();
                }
                else if(grad_only)
                {
                    EE_friction_gradient(G, kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_Ea0.x, prev_Ea0.y, prev_Ea0.z},
                        Float3{prev_Ea1.x, prev_Ea1.y, prev_Ea1.z},
                        Float3{prev_Eb0.x, prev_Eb0.y, prev_Eb0.z},
                        Float3{prev_Eb1.x, prev_Eb1.y, prev_Eb1.z},
                        Float3{Ea0.x, Ea0.y, Ea0.z},
                        Float3{Ea1.x, Ea1.y, Ea1.z},
                        Float3{Eb0.x, Eb0.y, Eb0.z},
                        Float3{Eb1.x, Eb1.y, Eb1.z});
                }
                else
                {
                    EE_friction_gradient_hessian(G, H, kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_Ea0.x, prev_Ea0.y, prev_Ea0.z},
                        Float3{prev_Ea1.x, prev_Ea1.y, prev_Ea1.z},
                        Float3{prev_Eb0.x, prev_Eb0.y, prev_Eb0.z},
                        Float3{prev_Eb1.x, prev_Eb1.y, prev_Eb1.z},
                        Float3{Ea0.x, Ea0.y, Ea0.z},
                        Float3{Ea1.x, Ea1.y, Ea1.z},
                        Float3{Eb0.x, Eb0.y, Eb0.z},
                        Float3{Eb1.x, Eb1.y, Eb1.z});
                    friction::make_spd(H);
                }

                // Write gradient
                $for(jj, 4) {
                    DoubletVector3 grad;
                    grad.index = EE[jj];
                    grad.value = Vector3{G[jj * 3 + 0], G[jj * 3 + 1], G[jj * 3 + 2]};
                    EE_grads.write(i * 4 + jj, grad);
                };

                // Write Hessian
                $if(!grad_only) {
                    $for(jj, 4) {
                        $for(kk, 4) {
                            if(jj <= kk) {
                                IndexT block_idx;
                                if(jj == 0) {
                                    block_idx = kk;
                                } else if(jj == 1) {
                                    block_idx = 4 + (kk - 1);
                                } else if(jj == 2) {
                                    block_idx = 7 + (kk - 2);
                                } else {
                                    block_idx = 9;
                                }
                                
                                TripletMatrix3 hess;
                                hess.row = EE[jj];
                                hess.col = EE[kk];
                                for(int r = 0; r < 3; ++r) {
                                    for(int c = 0; c < 3; ++c) {
                                        hess.value[r][c] = H[jj * 3 + r][kk * 3 + c];
                                    }
                                }
                                EE_hess.write(i * 10 + block_idx, hess);
                            }
                        };
                    };
                };
            }
            // PE contact
            else if(idx < pe_offset)
            {
                auto i = idx - pe_offset;
                auto PE = PEs.read(i);
                Vector3i cids = {contact_element_ids.read(PE[0]), contact_element_ids.read(PE[1]), 
                                 contact_element_ids.read(PE[2])};
                auto coeff = PE_contact_coeff(contact_tabular, cids);
                Float kt2 = coeff.kappa * dt * dt;
                Float mu = coeff.mu;

                auto prev_P  = prev_positions.read(PE[0]);
                auto prev_E0 = prev_positions.read(PE[1]);
                auto prev_E1 = prev_positions.read(PE[2]);

                auto P  = positions.read(PE[0]);
                auto E0 = positions.read(PE[1]);
                auto E1 = positions.read(PE[2]);

                Float thickness_P  = thicknesses.read(PE[0]);
                Float thickness_E0 = thicknesses.read(PE[1]);
                Float thickness_E1 = thicknesses.read(PE[2]);
                Float thickness = PE_thickness(thickness_P, thickness_E0, thickness_E1);

                Float d_hat_P  = d_hats.read(PE[0]);
                Float d_hat_E0 = d_hats.read(PE[1]);
                Float d_hat_E1 = d_hats.read(PE[2]);
                Float d_hat = PE_d_hat(d_hat_P, d_hat_E0, d_hat_E1);

                Float eps_vh = eps_v * dt;

                Vector9 G;
                Matrix9x9 H;
                
                if(grad_only)
                {
                    PE_friction_gradient(G, kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_P.x, prev_P.y, prev_P.z},
                        Float3{prev_E0.x, prev_E0.y, prev_E0.z},
                        Float3{prev_E1.x, prev_E1.y, prev_E1.z},
                        Float3{P.x, P.y, P.z},
                        Float3{E0.x, E0.y, E0.z},
                        Float3{E1.x, E1.y, E1.z});
                }
                else
                {
                    PE_friction_gradient_hessian(G, H, kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_P.x, prev_P.y, prev_P.z},
                        Float3{prev_E0.x, prev_E0.y, prev_E0.z},
                        Float3{prev_E1.x, prev_E1.y, prev_E1.z},
                        Float3{P.x, P.y, P.z},
                        Float3{E0.x, E0.y, E0.z},
                        Float3{E1.x, E1.y, E1.z});
                    friction::make_spd(H);
                }

                // Write gradient
                $for(jj, 3) {
                    DoubletVector3 grad;
                    grad.index = PE[jj];
                    grad.value = Vector3{G[jj * 3 + 0], G[jj * 3 + 1], G[jj * 3 + 2]};
                    PE_grads.write(i * 3 + jj, grad);
                };

                // Write Hessian
                $if(!grad_only) {
                    $for(jj, 3) {
                        $for(kk, 3) {
                            if(jj <= kk) {
                                IndexT block_idx;
                                if(jj == 0) {
                                    block_idx = kk;
                                } else if(jj == 1) {
                                    block_idx = 3 + (kk - 1);
                                } else {
                                    block_idx = 5;
                                }
                                
                                TripletMatrix3 hess;
                                hess.row = PE[jj];
                                hess.col = PE[kk];
                                for(int r = 0; r < 3; ++r) {
                                    for(int c = 0; c < 3; ++c) {
                                        hess.value[r][c] = H[jj * 3 + r][kk * 3 + c];
                                    }
                                }
                                PE_hess.write(i * 6 + block_idx, hess);
                            }
                        };
                    };
                };
            }
            // PP contact
            else if(idx < pp_offset)
            {
                auto i = idx - pp_offset;
                auto PP = PPs.read(i);
                Vector2i cids = {contact_element_ids.read(PP[0]), contact_element_ids.read(PP[1])};
                auto coeff = PP_contact_coeff(contact_tabular, cids);
                Float kt2 = coeff.kappa * dt * dt;
                Float mu = coeff.mu;

                auto prev_P0 = prev_positions.read(PP[0]);
                auto prev_P1 = prev_positions.read(PP[1]);

                auto P0 = positions.read(PP[0]);
                auto P1 = positions.read(PP[1]);

                Float thickness_P0 = thicknesses.read(PP[0]);
                Float thickness_P1 = thicknesses.read(PP[1]);
                Float thickness = PP_thickness(thickness_P0, thickness_P1);

                Float d_hat_P0 = d_hats.read(PP[0]);
                Float d_hat_P1 = d_hats.read(PP[1]);
                Float d_hat = PP_d_hat(d_hat_P0, d_hat_P1);

                Float eps_vh = eps_v * dt;

                Vector6 G;
                Matrix6x6 H;
                
                if(grad_only)
                {
                    PP_friction_gradient(G, kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_P0.x, prev_P0.y, prev_P0.z},
                        Float3{prev_P1.x, prev_P1.y, prev_P1.z},
                        Float3{P0.x, P0.y, P0.z},
                        Float3{P1.x, P1.y, P1.z});
                }
                else
                {
                    PP_friction_gradient_hessian(G, H, kt2, d_hat, thickness, mu, eps_vh,
                        Float3{prev_P0.x, prev_P0.y, prev_P0.z},
                        Float3{prev_P1.x, prev_P1.y, prev_P1.z},
                        Float3{P0.x, P0.y, P0.z},
                        Float3{P1.x, P1.y, P1.z});
                    friction::make_spd(H);
                }

                // Write gradient
                $for(jj, 2) {
                    DoubletVector3 grad;
                    grad.index = PP[jj];
                    grad.value = Vector3{G[jj * 3 + 0], G[jj * 3 + 1], G[jj * 3 + 2]};
                    PP_grads.write(i * 2 + jj, grad);
                };

                // Write Hessian
                $if(!grad_only) {
                    $for(jj, 2) {
                        $for(kk, 2) {
                            if(jj <= kk) {
                                IndexT block_idx = (jj == 0) ? kk : 2;
                                
                                TripletMatrix3 hess;
                                hess.row = PP[jj];
                                hess.col = PP[kk];
                                for(int r = 0; r < 3; ++r) {
                                    for(int c = 0; c < 3; ++c) {
                                        hess.value[r][c] = H[jj * 3 + r][kk * 3 + c];
                                    }
                                }
                                PP_hess.write(i * 3 + block_idx, hess);
                            }
                        };
                    };
                };
            }
        };

        auto shader = device.compile(assemble_kernel);
        stream << shader(contact_tabular_view,
                         positions_view,
                         prev_positions_view,
                         rest_positions_view,
                         thicknesses_view,
                         d_hats_view,
                         contact_element_ids_view,
                         PTs_view,
                         PT_gradients_view,
                         PT_hessians_view,
                         EEs_view,
                         EE_gradients_view,
                         EE_hessians_view,
                         PEs_view,
                         PE_gradients_view,
                         PE_hessians_view,
                         PPs_view,
                         PP_gradients_view,
                         PP_hessians_view,
                         dt,
                         eps_v,
                         gradient_only,
                         static_cast<UInt>(pt_count),
                         static_cast<UInt>(ee_offset),
                         static_cast<UInt>(pe_offset),
                         static_cast<UInt>(pp_offset))
                      .dispatch(total_count);
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexFrictionalContact);
}  // namespace uipc::backend::luisa
