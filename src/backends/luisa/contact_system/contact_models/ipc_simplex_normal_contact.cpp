#include <contact_system/contact_models/ipc_simplex_normal_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <contact_system/simplex_normal_contact.h>
#include <utils/codim_thickness.h>
#include <utils/primitive_d_hat.h>
#include <utils/distance/distance_flagged.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>

namespace uipc::backend::luisa
{
using namespace sym::codim_ipc_simplex_contact;

class IPCSimplexNormalContact final : public SimplexNormalContact
{
public:
    using SimplexNormalContact::SimplexNormalContact;

    void do_build(BuildInfo& info) override
    {
        auto constitution =
            world().scene().config().find<std::string>("contact/constitution");
        if(constitution->view()[0] != "ipc")
        {
            throw SimSystemException("Constitution is not IPC");
        }
    }

    void do_compute_energy(EnergyInfo& info) override
    {
        // Get counts
        SizeT pt_count = info.PTs().size();
        SizeT ee_count = info.EEs().size();
        SizeT pe_count = info.PEs().size();
        SizeT pp_count = info.PPs().size();

        // Get device
        auto& engine = world().sim_engine();
        auto& device = static_cast<SimEngine&>(engine).device();
        auto  stream = static_cast<SimEngine&>(engine).compute_stream();

        // Get buffer views
        auto contact_tabular_view = info.contact_tabular();
        auto positions_view       = info.positions();
        auto rest_positions_view  = info.rest_positions();
        auto thicknesses_view     = info.thicknesses();
        auto d_hats_view          = info.d_hats();
        auto contact_element_ids_view = info.contact_element_ids();

        // PT data
        auto PTs_view       = info.PTs();
        auto PT_energies_view = info.PT_energies();

        // EE data
        auto EEs_view       = info.EEs();
        auto EE_energies_view = info.EE_energies();

        // PE data
        auto PEs_view       = info.PEs();
        auto PE_energies_view = info.PE_energies();

        // PP data
        auto PPs_view       = info.PPs();
        auto PP_energies_view = info.PP_energies();

        Float dt = info.dt();

        // PT energy kernel
        if(pt_count > 0)
        {
            Kernel1D pt_energy_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                            BufferVar<Vector3> positions,
                                            BufferVar<Float> thicknesses,
                                            BufferVar<Float> d_hats,
                                            BufferVar<IndexT> contact_element_ids,
                                            BufferVar<Vector4i> PTs,
                                            BufferVar<Float> PT_energies,
                                            Float dt)
            {
                auto i = dispatch_x();
                $if(i < pt_count)
                {
                    auto PT = PTs.read(i);
                    Vector4i cids = {contact_element_ids.read(PT[0]), contact_element_ids.read(PT[1]),
                                     contact_element_ids.read(PT[2]), contact_element_ids.read(PT[3])};
                    Float kt2 = PT_kappa(contact_tabular, cids) * dt * dt;

                    auto P  = positions.read(PT[0]);
                    auto T0 = positions.read(PT[1]);
                    auto T1 = positions.read(PT[2]);
                    auto T2 = positions.read(PT[3]);

                    Float thickness = PT_thickness(thicknesses.read(PT[0]),
                                                   thicknesses.read(PT[1]),
                                                   thicknesses.read(PT[2]),
                                                   thicknesses.read(PT[3]));

                    Float d_hat = PT_d_hat(d_hats.read(PT[0]), d_hats.read(PT[1]),
                                           d_hats.read(PT[2]), d_hats.read(PT[3]));

                    Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);

                    $if constexpr(RUNTIME_CHECK)
                    {
                        Float D;
                        distance::point_triangle_distance2(flag, P, T0, T1, T2, D);
                        Vector2 range = D_range(thickness, d_hat);
                        // Note: LC_ASSERT not available in device code, skip assertion
                    };

                    Float E = PT_barrier_energy(flag, kt2, d_hat, thickness, P, T0, T1, T2);
                    PT_energies.write(i, E);
                };
            };

            auto shader = device.compile(pt_energy_kernel);
            stream << shader(contact_tabular_view,
                             positions_view,
                             thicknesses_view,
                             d_hats_view,
                             contact_element_ids_view,
                             PTs_view,
                             PT_energies_view,
                             dt)
                          .dispatch(pt_count);
        }

        // EE energy kernel
        if(ee_count > 0)
        {
            Kernel1D ee_energy_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                            BufferVar<Vector3> positions,
                                            BufferVar<Vector3> rest_positions,
                                            BufferVar<Float> thicknesses,
                                            BufferVar<Float> d_hats,
                                            BufferVar<IndexT> contact_element_ids,
                                            BufferVar<Vector4i> EEs,
                                            BufferVar<Float> EE_energies,
                                            Float dt)
            {
                auto i = dispatch_x();
                $if(i < ee_count)
                {
                    auto EE = EEs.read(i);
                    Vector4i cids = {contact_element_ids.read(EE[0]), contact_element_ids.read(EE[1]),
                                     contact_element_ids.read(EE[2]), contact_element_ids.read(EE[3])};
                    Float kt2 = EE_kappa(contact_tabular, cids) * dt * dt;

                    auto E0 = positions.read(EE[0]);
                    auto E1 = positions.read(EE[1]);
                    auto E2 = positions.read(EE[2]);
                    auto E3 = positions.read(EE[3]);

                    auto t0_Ea0 = rest_positions.read(EE[0]);
                    auto t0_Ea1 = rest_positions.read(EE[1]);
                    auto t0_Eb0 = rest_positions.read(EE[2]);
                    auto t0_Eb1 = rest_positions.read(EE[3]);

                    Float thickness = EE_thickness(thicknesses.read(EE[0]),
                                                   thicknesses.read(EE[1]),
                                                   thicknesses.read(EE[2]),
                                                   thicknesses.read(EE[3]));

                    Float d_hat = EE_d_hat(d_hats.read(EE[0]), d_hats.read(EE[1]),
                                           d_hats.read(EE[2]), d_hats.read(EE[3]));

                    Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                    $if constexpr(RUNTIME_CHECK)
                    {
                        Float D;
                        distance::edge_edge_distance2(flag, E0, E1, E2, E3, D);
                        Vector2 range = D_range(thickness, d_hat);
                        // Note: LC_ASSERT not available in device code, skip assertion
                    };

                    Float E = mollified_EE_barrier_energy(flag, kt2, d_hat, thickness,
                                                          t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1,
                                                          E0, E1, E2, E3);
                    EE_energies.write(i, E);
                };
            };

            auto shader = device.compile(ee_energy_kernel);
            stream << shader(contact_tabular_view,
                             positions_view,
                             rest_positions_view,
                             thicknesses_view,
                             d_hats_view,
                             contact_element_ids_view,
                             EEs_view,
                             EE_energies_view,
                             dt)
                          .dispatch(ee_count);
        }

        // PE energy kernel
        if(pe_count > 0)
        {
            Kernel1D pe_energy_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                            BufferVar<Vector3> positions,
                                            BufferVar<Float> thicknesses,
                                            BufferVar<Float> d_hats,
                                            BufferVar<IndexT> contact_element_ids,
                                            BufferVar<Vector3i> PEs,
                                            BufferVar<Float> PE_energies,
                                            Float dt)
            {
                auto i = dispatch_x();
                $if(i < pe_count)
                {
                    auto PE = PEs.read(i);
                    Vector3i cids = {contact_element_ids.read(PE[0]),
                                     contact_element_ids.read(PE[1]),
                                     contact_element_ids.read(PE[2])};
                    Float kt2 = PE_kappa(contact_tabular, cids) * dt * dt;

                    auto P  = positions.read(PE[0]);
                    auto E0 = positions.read(PE[1]);
                    auto E1 = positions.read(PE[2]);

                    Float thickness = PE_thickness(thicknesses.read(PE[0]),
                                                   thicknesses.read(PE[1]),
                                                   thicknesses.read(PE[2]));

                    Float d_hat = PE_d_hat(d_hats.read(PE[0]), d_hats.read(PE[1]),
                                           d_hats.read(PE[2]));

                    Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                    $if constexpr(RUNTIME_CHECK)
                    {
                        Float D;
                        distance::point_edge_distance2(flag, P, E0, E1, D);
                        Vector2 range = D_range(thickness, d_hat);
                        // Note: LC_ASSERT not available in device code, skip assertion
                    };

                    Float E = PE_barrier_energy(flag, kt2, d_hat, thickness, P, E0, E1);
                    PE_energies.write(i, E);
                };
            };

            auto shader = device.compile(pe_energy_kernel);
            stream << shader(contact_tabular_view,
                             positions_view,
                             thicknesses_view,
                             d_hats_view,
                             contact_element_ids_view,
                             PEs_view,
                             PE_energies_view,
                             dt)
                          .dispatch(pe_count);
        }

        // PP energy kernel
        if(pp_count > 0)
        {
            Kernel1D pp_energy_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                            BufferVar<Vector3> positions,
                                            BufferVar<Float> thicknesses,
                                            BufferVar<Float> d_hats,
                                            BufferVar<IndexT> contact_element_ids,
                                            BufferVar<Vector2i> PPs,
                                            BufferVar<Float> PP_energies,
                                            Float dt)
            {
                auto i = dispatch_x();
                $if(i < pp_count)
                {
                    auto PP = PPs.read(i);
                    Vector2i cids = {contact_element_ids.read(PP[0]), contact_element_ids.read(PP[1])};
                    Float kt2 = PP_kappa(contact_tabular, cids) * dt * dt;

                    auto Pa = positions.read(PP[0]);
                    auto Pb = positions.read(PP[1]);

                    Float thickness = PP_thickness(thicknesses.read(PP[0]),
                                                   thicknesses.read(PP[1]));

                    Float d_hat = PP_d_hat(d_hats.read(PP[0]), d_hats.read(PP[1]));

                    Vector2i flag = distance::point_point_distance_flag(Pa, Pb);

                    $if constexpr(RUNTIME_CHECK)
                    {
                        Float D;
                        distance::point_point_distance2(flag, Pa, Pb, D);
                        Vector2 range = D_range(thickness, d_hat);
                        // Note: LC_ASSERT not available in device code, skip assertion
                    };

                    Float E = PP_barrier_energy(flag, kt2, d_hat, thickness, Pa, Pb);
                    PP_energies.write(i, E);
                };
            };

            auto shader = device.compile(pp_energy_kernel);
            stream << shader(contact_tabular_view,
                             positions_view,
                             thicknesses_view,
                             d_hats_view,
                             contact_element_ids_view,
                             PPs_view,
                             PP_energies_view,
                             dt)
                          .dispatch(pp_count);
        }
    }

    void do_assemble(ContactInfo& info) override
    {
        // Fused kernel: PT + EE + PE + PP in one launch using offset-based dispatch.
        // Reduces 4 kernel launches to 1, improving GPU occupancy by providing
        // more threads in a single launch.
        auto pt_count = (IndexT)info.PTs().size();
        auto ee_count = (IndexT)info.EEs().size();
        auto pe_count = (IndexT)info.PEs().size();
        auto pp_count = (IndexT)info.PPs().size();
        auto total    = pt_count + ee_count + pe_count + pp_count;

        if(total == 0)
            return;

        // Get device
        auto& engine = world().sim_engine();
        auto& device = static_cast<SimEngine&>(engine).device();
        auto  stream = static_cast<SimEngine&>(engine).compute_stream();

        IndexT ee_offset = pt_count;
        IndexT pe_offset = ee_offset + ee_count;
        IndexT pp_offset = pe_offset + pe_count;

        bool gradient_only = info.gradient_only();

        // Get buffer views
        auto contact_tabular_view = info.contact_tabular();
        auto positions_view       = info.positions();
        auto rest_positions_view  = info.rest_positions();
        auto thicknesses_view     = info.thicknesses();
        auto d_hats_view          = info.d_hats();
        auto contact_element_ids_view = info.contact_element_ids();

        // PT data
        auto PTs_view = info.PTs();
        auto PT_gradients_view = info.PT_gradients();
        auto PT_hessians_view = info.PT_hessians();

        // EE data
        auto EEs_view = info.EEs();
        auto EE_gradients_view = info.EE_gradients();
        auto EE_hessians_view = info.EE_hessians();

        // PE data
        auto PEs_view = info.PEs();
        auto PE_gradients_view = info.PE_gradients();
        auto PE_hessians_view = info.PE_hessians();

        // PP data
        auto PPs_view = info.PPs();
        auto PP_gradients_view = info.PP_gradients();
        auto PP_hessians_view = info.PP_hessians();

        Float dt = info.dt();

        // Fused assembly kernel
        Kernel1D assemble_kernel = [&](BufferVar<ContactCoeff> contact_tabular,
                                       BufferVar<Vector3> positions,
                                       BufferVar<Vector3> rest_positions,
                                       BufferVar<Float> thicknesses,
                                       BufferVar<Float> d_hats,
                                       BufferVar<IndexT> contact_element_ids,
                                       // PT
                                       BufferVar<Vector4i> PTs,
                                       BufferVar<Doublet<Vector3>> PT_grads,
                                       BufferVar<Triplet<Matrix3>> PT_hess,
                                       // EE
                                       BufferVar<Vector4i> EEs,
                                       BufferVar<Doublet<Vector3>> EE_grads,
                                       BufferVar<Triplet<Matrix3>> EE_hess,
                                       // PE
                                       BufferVar<Vector3i> PEs,
                                       BufferVar<Doublet<Vector3>> PE_grads,
                                       BufferVar<Triplet<Matrix3>> PE_hess,
                                       // PP
                                       BufferVar<Vector2i> PPs,
                                       BufferVar<Doublet<Vector3>> PP_grads,
                                       BufferVar<Triplet<Matrix3>> PP_hess,
                                       Float dt,
                                       Bool grad_only,
                                       UInt pt_count,
                                       UInt ee_offset,
                                       UInt pe_offset,
                                       UInt pp_offset)
        {
            auto idx = dispatch_x();

            // PT contact
            $if(idx < ee_offset)
            {
                auto i = idx;
                auto PT = PTs.read(i);
                Vector4i cids = {contact_element_ids.read(PT[0]), contact_element_ids.read(PT[1]),
                                 contact_element_ids.read(PT[2]), contact_element_ids.read(PT[3])};
                Float kt2 = PT_kappa(contact_tabular, cids) * dt * dt;

                auto P  = positions.read(PT[0]);
                auto T0 = positions.read(PT[1]);
                auto T1 = positions.read(PT[2]);
                auto T2 = positions.read(PT[3]);

                Float thickness = PT_thickness(thicknesses.read(PT[0]),
                                               thicknesses.read(PT[1]),
                                               thicknesses.read(PT[2]),
                                               thicknesses.read(PT[3]));
                Float d_hat = PT_d_hat(d_hats.read(PT[0]), d_hats.read(PT[1]),
                                       d_hats.read(PT[2]), d_hats.read(PT[3]));
                Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);

                Vector12 G;
                $if(grad_only)
                {
                    PT_barrier_gradient(G, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                }
                $else
                {
                    Matrix12x12 H;
                    PT_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                    make_spd(H);

                    // Write Hessian - upper triangular only
                    $for(ii, 4) {
                        $for(jj, ii, 4) {
                            IndexT block_idx = i * PTHalfHessianSize + ii * 4 + jj - ii * (ii + 1) / 2;
                            Triplet<Matrix3> hess;
                            hess.row = PT[ii];
                            hess.col = PT[jj];
                            for(int r = 0; r < 3; ++r) {
                                for(int c = 0; c < 3; ++c) {
                                    hess.value[r][c] = H[ii * 3 + r][jj * 3 + c];
                                }
                            }
                            PT_hess.write(block_idx, hess);
                        };
                    };
                };

                // Write gradient
                $for(ii, 4) {
                    Doublet<Vector3> grad;
                    grad.index = PT[ii];
                    grad.value = Vector3{G[ii * 3 + 0], G[ii * 3 + 1], G[ii * 3 + 2]};
                    PT_grads.write(i * 4 + ii, grad);
                };
            }
            // EE contact
            $elif(idx < pe_offset)
            {
                auto i = idx - ee_offset;
                auto EE = EEs.read(i);
                Vector4i cids = {contact_element_ids.read(EE[0]), contact_element_ids.read(EE[1]),
                                 contact_element_ids.read(EE[2]), contact_element_ids.read(EE[3])};
                Float kt2 = EE_kappa(contact_tabular, cids) * dt * dt;

                auto E0 = positions.read(EE[0]);
                auto E1 = positions.read(EE[1]);
                auto E2 = positions.read(EE[2]);
                auto E3 = positions.read(EE[3]);

                auto t0_Ea0 = rest_positions.read(EE[0]);
                auto t0_Ea1 = rest_positions.read(EE[1]);
                auto t0_Eb0 = rest_positions.read(EE[2]);
                auto t0_Eb1 = rest_positions.read(EE[3]);

                Float thickness = EE_thickness(thicknesses.read(EE[0]),
                                               thicknesses.read(EE[1]),
                                               thicknesses.read(EE[2]),
                                               thicknesses.read(EE[3]));
                Float d_hat = EE_d_hat(d_hats.read(EE[0]), d_hats.read(EE[1]),
                                       d_hats.read(EE[2]), d_hats.read(EE[3]));
                Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                Vector12 G;
                $if(grad_only)
                {
                    mollified_EE_barrier_gradient(G, flag, kt2, d_hat, thickness,
                                                  t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1,
                                                  E0, E1, E2, E3);
                }
                $else
                {
                    Matrix12x12 H;
                    mollified_EE_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness,
                                                          t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1,
                                                          E0, E1, E2, E3);
                    make_spd(H);

                    // Write Hessian - upper triangular only
                    $for(ii, 4) {
                        $for(jj, ii, 4) {
                            IndexT block_idx = i * EEHalfHessianSize + ii * 4 + jj - ii * (ii + 1) / 2;
                            Triplet<Matrix3> hess;
                            hess.row = EE[ii];
                            hess.col = EE[jj];
                            for(int r = 0; r < 3; ++r) {
                                for(int c = 0; c < 3; ++c) {
                                    hess.value[r][c] = H[ii * 3 + r][jj * 3 + c];
                                }
                            }
                            EE_hess.write(block_idx, hess);
                        };
                    };
                };

                // Write gradient
                $for(ii, 4) {
                    Doublet<Vector3> grad;
                    grad.index = EE[ii];
                    grad.value = Vector3{G[ii * 3 + 0], G[ii * 3 + 1], G[ii * 3 + 2]};
                    EE_grads.write(i * 4 + ii, grad);
                };
            }
            // PE contact
            $elif(idx < pp_offset)
            {
                auto i = idx - pe_offset;
                auto PE = PEs.read(i);
                Vector3i cids = {contact_element_ids.read(PE[0]),
                                 contact_element_ids.read(PE[1]),
                                 contact_element_ids.read(PE[2])};
                Float kt2 = PE_kappa(contact_tabular, cids) * dt * dt;

                auto P  = positions.read(PE[0]);
                auto E0 = positions.read(PE[1]);
                auto E1 = positions.read(PE[2]);

                Float thickness = PE_thickness(thicknesses.read(PE[0]),
                                               thicknesses.read(PE[1]),
                                               thicknesses.read(PE[2]));
                Float d_hat = PE_d_hat(d_hats.read(PE[0]), d_hats.read(PE[1]),
                                       d_hats.read(PE[2]));
                Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                Vector9 G;
                $if(grad_only)
                {
                    PE_barrier_gradient(G, flag, kt2, d_hat, thickness, P, E0, E1);
                }
                $else
                {
                    Matrix9x9 H;
                    PE_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P, E0, E1);
                    make_spd(H);

                    // Write Hessian - upper triangular only
                    $for(ii, 3) {
                        $for(jj, ii, 3) {
                            IndexT block_idx = i * PEHalfHessianSize + ii * 3 + jj - ii * (ii + 1) / 2;
                            Triplet<Matrix3> hess;
                            hess.row = PE[ii];
                            hess.col = PE[jj];
                            for(int r = 0; r < 3; ++r) {
                                for(int c = 0; c < 3; ++c) {
                                    hess.value[r][c] = H[ii * 3 + r][jj * 3 + c];
                                }
                            }
                            PE_hess.write(block_idx, hess);
                        };
                    };
                };

                // Write gradient
                $for(ii, 3) {
                    Doublet<Vector3> grad;
                    grad.index = PE[ii];
                    grad.value = Vector3{G[ii * 3 + 0], G[ii * 3 + 1], G[ii * 3 + 2]};
                    PE_grads.write(i * 3 + ii, grad);
                };
            }
            // PP contact
            $else
            {
                auto i = idx - pp_offset;
                auto PP = PPs.read(i);
                Vector2i cids = {contact_element_ids.read(PP[0]), contact_element_ids.read(PP[1])};
                Float kt2 = PP_kappa(contact_tabular, cids) * dt * dt;

                auto P0 = positions.read(PP[0]);
                auto P1 = positions.read(PP[1]);

                Float thickness = PP_thickness(thicknesses.read(PP[0]),
                                               thicknesses.read(PP[1]));
                Float d_hat = PP_d_hat(d_hats.read(PP[0]), d_hats.read(PP[1]));
                Vector2i flag = distance::point_point_distance_flag(P0, P1);

                Vector6 G;
                $if(grad_only)
                {
                    PP_barrier_gradient(G, flag, kt2, d_hat, thickness, P0, P1);
                }
                $else
                {
                    Matrix6x6 H;
                    PP_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P0, P1);
                    make_spd(H);

                    // Write Hessian - upper triangular only
                    $for(ii, 2) {
                        $for(jj, ii, 2) {
                            IndexT block_idx = i * PPHalfHessianSize + ii * 2 + jj - ii * (ii + 1) / 2;
                            Triplet<Matrix3> hess;
                            hess.row = PP[ii];
                            hess.col = PP[jj];
                            for(int r = 0; r < 3; ++r) {
                                for(int c = 0; c < 3; ++c) {
                                    hess.value[r][c] = H[ii * 3 + r][jj * 3 + c];
                                }
                            }
                            PP_hess.write(block_idx, hess);
                        };
                    };
                };

                // Write gradient
                $for(ii, 2) {
                    Doublet<Vector3> grad;
                    grad.index = PP[ii];
                    grad.value = Vector3{G[ii * 3 + 0], G[ii * 3 + 1], G[ii * 3 + 2]};
                    PP_grads.write(i * 2 + ii, grad);
                };
            };
        };

        auto shader = device.compile(assemble_kernel);
        stream << shader(contact_tabular_view,
                         positions_view,
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
                         gradient_only,
                         static_cast<UInt>(pt_count),
                         static_cast<UInt>(ee_offset),
                         static_cast<UInt>(pe_offset),
                         static_cast<UInt>(pp_offset))
                      .dispatch(total);
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexNormalContact);
}  // namespace uipc::backend::luisa
