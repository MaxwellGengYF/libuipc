#include <backends/luisa/finite_element/constitutions/neo_hookean_shell_2d.h>
#include <backends/luisa/finite_element/finite_element_animator.h>
#include <backends/luisa/finite_element/finite_element_method.h>
#include <backends/luisa/utils/codim_thickness.h>
#include <finite_element/matrix_utils.h>
#include <backends/luisa/finite_element/constitutions/neo_hookean_shell_2d_function.h>

namespace uipc::backend::luisa
{
class NeoHookeanShell2D::Impl
{
  public:
    Impl(NeoHookeanShell2D* ptr) noexcept
        : m_constitution(ptr)
    {
    }

    void init(FiniteElementAnimator::ScopedInitState& state)
    {
        using namespace luisa;
        using namespace luisa::compute;

        // Get FEM and constitution info
        auto& fem       = m_constitution->m_finite_element_method;
        auto  codim_dim = m_constitution->m_codim_dimension;
        auto  uid       = m_constitution->m_uid;

        // Get primitive info
        auto  geo_slots = state.geo_slots();
        auto& prim_info = fem->codim_primitive_infos(codim_dim);

        // Count elements
        SizeT N = 0;
        for(auto& info : prim_info)
        {
            if(info.constitution_uid == uid)
            {
                N += info.primitive_count;
            }
        }

        // Initialize material parameters
        h_lambdas.resize(N);
        h_mus.resize(N);

        // Initialize inverse B matrices (2x2 for triangles)
        inv_B_matrices = device_buffer<Matrix2x2>(N);

        SizeT offset = 0;
        for(auto& info : prim_info)
        {
            if(info.constitution_uid != uid)
                continue;

            auto  geo_slot = geo_slots[info.geo_slot_index];
            auto& geo      = geo_slot->geometry();

            // Get rest positions and indices
            auto rest_positions = geo.attrs().find<Float3>(builtin::rest_position);
            auto indices        = geo.topos().find<Triangle>(builtin::triangle);

            UIPC_ASSERT(rest_positions, "Rest positions not found");
            UIPC_ASSERT(indices, "Triangle indices not found");

            auto rest_pos_view = rest_positions->view();
            auto index_view    = indices->view();

            // Process each triangle
            for(SizeT i = 0; i < info.primitive_count; ++i)
            {
                auto tri = index_view[i];

                // Get rest positions
                Float3 x0_bar = rest_pos_view[tri[0]];
                Float3 x1_bar = rest_pos_view[tri[1]];
                Float3 x2_bar = rest_pos_view[tri[2]];

                // Compute rest shape matrix B
                Matrix2x2 B;
                Float3    e0_bar = x1_bar - x0_bar;
                Float3    e1_bar = x2_bar - x0_bar;

                // B = [e0_bar · e0_bar, e0_bar · e1_bar]
                //     [e1_bar · e0_bar, e1_bar · e1_bar]
                B[0][0] = dot(e0_bar, e0_bar);
                B[0][1] = dot(e0_bar, e1_bar);
                B[1][0] = B[0][1];
                B[1][1] = dot(e1_bar, e1_bar);

                // Compute inverse B
                Matrix2x2 IB = inverse(B);

                // Store inverse B matrix
                inv_B_matrices->write(offset + i, IB);

                // Get material parameters from geometry
                auto lambda_attr = geo.attrs().find<Float>(builtin::lambda);
                auto mu_attr     = geo.attrs().find<Float>(builtin::mu);

                if(lambda_attr && mu_attr)
                {
                    h_lambdas[offset + i] = lambda_attr->view()[i];
                    h_mus[offset + i]     = mu_attr->view()[i];
                }
                else
                {
                    // Default values
                    h_lambdas[offset + i] = 1000.0f;
                    h_mus[offset + i]     = 1000.0f;
                }
            }

            offset += info.primitive_count;
        }

        // Upload material parameters to device
        lambdas = device_buffer<Float>(N);
        mus     = device_buffer<Float>(N);
        lambdas->copy_from(h_lambdas.data());
        mus->copy_from(h_mus.data());
    }

    void compute_energy(ComputeEnergyInfo& info)
    {
        namespace NH = sym::neo_hookean_shell_2d;
        using namespace luisa;
        using namespace luisa::compute;

        auto& fem       = m_constitution->m_finite_element_method;
        auto  codim_dim = m_constitution->m_codim_dimension;
        auto  uid       = m_constitution->m_uid;

        auto& prim_info = fem->codim_primitive_infos(codim_dim);

        SizeT N = 0;
        for(auto& info : prim_info)
        {
            if(info.constitution_uid == uid)
            {
                N += info.primitive_count;
            }
        }

        if(N == 0)
            return;

        auto  thicknesses = fem->thicknesses().view();
        auto  rest_areas  = fem->rest_areas(codim_dim).view();
        auto  energies    = info.energies().view();
        auto  xs          = info.xs().view();
        auto  indices     = fem->codim_primitive_indices(codim_dim).view();

        auto kernel = device().compile<1>(
            [thicknesses, rest_areas, energies, xs, indices, lambdas = lambdas->view(),
             mus = mus->view(), inv_Bs = inv_B_matrices->view()](Int i) mutable
            {
                // Get triangle indices
                UInt4 idx = indices.read(i);

                // Get current positions
                Float3 x0 = xs.read(idx[0]);
                Float3 x1 = xs.read(idx[1]);
                Float3 x2 = xs.read(idx[2]);

                // Get thickness (shell is one-sided, multiply by 2 for volume)
                Float thickness = triangle_thickness(thicknesses.read(idx[0]),
                                                     thicknesses.read(idx[1]),
                                                     thicknesses.read(idx[2]));
                Float h = 2.0f * thickness;

                // Get rest area
                Float rest_area = rest_areas.read(idx[3]);

                // Get material parameters
                Float lambda = lambdas.read(i);
                Float mu     = mus.read(i);

                // Get inverse B matrix
                Matrix2x2 IB = inv_Bs.read(i);

                // Build X vector (9 elements: x0, y0, z0, x1, y1, z1, x2, y2, z2)
                std::array<Float, 9> X;
                X[0] = x0.x;
                X[1] = x0.y;
                X[2] = x0.z;
                X[3] = x1.x;
                X[4] = x1.y;
                X[5] = x1.z;
                X[6] = x2.x;
                X[7] = x2.y;
                X[8] = x2.z;

                // Compute energy
                Float E_val;
                NH::E(E_val, lambda, mu, X, IB);

                // Energy = E * rest_area * h
                Float energy = E_val * rest_area * h;

                energies.write(idx[3], energy);
            });

        shader(kernel, N).dispatch(N);
    }

    void compute_gradient_hessian(ComputeGradientHessianInfo& info)
    {
        namespace NH = sym::neo_hookean_shell_2d;
        using namespace luisa;
        using namespace luisa::compute;

        auto& fem       = m_constitution->m_finite_element_method;
        auto  codim_dim = m_constitution->m_codim_dimension;
        auto  uid       = m_constitution->m_uid;

        auto& prim_info = fem->codim_primitive_infos(codim_dim);

        SizeT N = 0;
        for(auto& info : prim_info)
        {
            if(info.constitution_uid == uid)
            {
                N += info.primitive_count;
            }
        }

        if(N == 0)
            return;

        auto thicknesses = fem->thicknesses().view();
        auto rest_areas  = fem->rest_areas(codim_dim).view();
        auto gradients   = info.gradients().view();
        auto hessians    = info.hessians().view();
        auto xs          = info.xs().view();
        auto indices     = fem->codim_primitive_indices(codim_dim).view();

        auto kernel = device().compile<1>(
            [thicknesses, rest_areas, gradients, hessians, xs, indices,
             lambdas = lambdas->view(), mus = mus->view(),
             inv_Bs = inv_B_matrices->view(), gradient_only = info.gradient_only()](Int i) mutable
            {
                // Get triangle indices
                UInt4 idx = indices.read(i);

                // Get current positions
                Float3 x0 = xs.read(idx[0]);
                Float3 x1 = xs.read(idx[1]);
                Float3 x2 = xs.read(idx[2]);

                // Get thickness (shell is one-sided, multiply by 2 for volume)
                Float thickness = triangle_thickness(thicknesses.read(idx[0]),
                                                     thicknesses.read(idx[1]),
                                                     thicknesses.read(idx[2]));
                Float h = 2.0f * thickness;

                // Get rest area
                Float rest_area = rest_areas.read(idx[3]);

                // Get material parameters
                Float lambda = lambdas.read(i);
                Float mu     = mus.read(i);

                // Get inverse B matrix
                Matrix2x2 IB = inv_Bs.read(i);

                // Build X vector (9 elements: x0, y0, z0, x1, y1, z1, x2, y2, z2)
                std::array<Float, 9> X;
                X[0] = x0.x;
                X[1] = x0.y;
                X[2] = x0.z;
                X[3] = x1.x;
                X[4] = x1.y;
                X[5] = x1.z;
                X[6] = x2.x;
                X[7] = x2.y;
                X[8] = x2.z;

                // Compute gradient (9 elements)
                std::array<Float, 9> G;
                NH::dEdX(G, lambda, mu, X, IB);

                // Scale gradient by rest_area * h
                Float scale = rest_area * h;
                for(int j = 0; j < 9; ++j)
                {
                    G[j] *= scale;
                }

                // Write gradients for each vertex
                Float3 g0(G[0], G[1], G[2]);
                Float3 g1(G[3], G[4], G[5]);
                Float3 g2(G[6], G[7], G[8]);

                gradients.atomic(idx[0]).fetch_add(g0);
                gradients.atomic(idx[1]).fetch_add(g1);
                gradients.atomic(idx[2]).fetch_add(g2);

                // Compute Hessian if needed
                if(!gradient_only)
                {
                    // Compute Hessian (9x9 matrix)
                    std::array<std::array<Float, 9>, 9> H;
                    NH::ddEddX(H, lambda, mu, X, IB);

                    // Scale Hessian by rest_area * h
                    for(int j = 0; j < 9; ++j)
                    {
                        for(int k = 0; k < 9; ++k)
                        {
                            H[j][k] *= scale;
                        }
                    }

                    // Apply SPD projection
                    auto H_spd = clamp_to_spd(H);

                    // Write Hessian blocks (6 upper-triangular blocks for 3 vertices)
                    // Block (0,0): vertices 0-0
                    Float3x3 H_00(H_spd[0][0], H_spd[0][1], H_spd[0][2],
                                  H_spd[1][0], H_spd[1][1], H_spd[1][2],
                                  H_spd[2][0], H_spd[2][1], H_spd[2][2]);

                    // Block (0,1): vertices 0-1
                    Float3x3 H_01(H_spd[0][3], H_spd[0][4], H_spd[0][5],
                                  H_spd[1][3], H_spd[1][4], H_spd[1][5],
                                  H_spd[2][3], H_spd[2][4], H_spd[2][5]);

                    // Block (0,2): vertices 0-2
                    Float3x3 H_02(H_spd[0][6], H_spd[0][7], H_spd[0][8],
                                  H_spd[1][6], H_spd[1][7], H_spd[1][8],
                                  H_spd[2][6], H_spd[2][7], H_spd[2][8]);

                    // Block (1,1): vertices 1-1
                    Float3x3 H_11(H_spd[3][3], H_spd[3][4], H_spd[3][5],
                                  H_spd[4][3], H_spd[4][4], H_spd[4][5],
                                  H_spd[5][3], H_spd[5][4], H_spd[5][5]);

                    // Block (1,2): vertices 1-2
                    Float3x3 H_12(H_spd[3][6], H_spd[3][7], H_spd[3][8],
                                  H_spd[4][6], H_spd[4][7], H_spd[4][8],
                                  H_spd[5][6], H_spd[5][7], H_spd[5][8]);

                    // Block (2,2): vertices 2-2
                    Float3x3 H_22(H_spd[6][6], H_spd[6][7], H_spd[6][8],
                                  H_spd[7][6], H_spd[7][7], H_spd[7][8],
                                  H_spd[8][6], H_spd[8][7], H_spd[8][8]);

                    // Write Hessian blocks
                    hessians.write(idx[3] * 6 + 0, H_00);
                    hessians.write(idx[3] * 6 + 1, H_01);
                    hessians.write(idx[3] * 6 + 2, H_02);
                    hessians.write(idx[3] * 6 + 3, H_11);
                    hessians.write(idx[3] * 6 + 4, H_12);
                    hessians.write(idx[3] * 6 + 5, H_22);
                }
            });

        shader(kernel, N).dispatch(N);
    }

    void step(StepInfo& info)
    {
        // No specific step logic needed for Neo-Hookean shell
    }

    NeoHookeanShell2D* m_constitution;

    // Host material parameters
    vector<Float> h_lambdas;
    vector<Float> h_mus;

    // Device material parameters
    luisa::compute::Buffer<Float>      lambdas;
    luisa::compute::Buffer<Float>      mus;
    luisa::compute::Buffer<Matrix2x2>  inv_B_matrices;
};

std::string_view NeoHookeanShell2D::get_name() const noexcept
{
    return "NeoHookeanShell2D";
}

void NeoHookeanShell2D::do_init(FiniteElementAnimator::ScopedInitState& state)
{
    if(!m_impl)
        m_impl = luisa::make_unique<Impl>(this);
    m_impl->init(state);
}

void NeoHookeanShell2D::do_compute_energy(ComputeEnergyInfo& info)
{
    m_impl->compute_energy(info);
}

void NeoHookeanShell2D::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    m_impl->compute_gradient_hessian(info);
}

void NeoHookeanShell2D::do_step(StepInfo& info)
{
    m_impl->step(info);
}
}  // namespace uipc::backend::luisa
