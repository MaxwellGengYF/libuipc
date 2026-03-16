#include <finite_element/finite_element_constraint.h>
#include <finite_element/finite_element_animator.h>
#include <finite_element/finite_element_method.h>
#include <luisa/runtime/buffer.h>

namespace uipc::backend::luisa
{
class SoftPositionConstraint final : public FiniteElementConstraint
{
  public:
    using FiniteElementConstraint::FiniteElementConstraint;

    static constexpr SizeT ConstraintUID = 14;

    class Impl
    {
      public:
        Impl(SoftPositionConstraint* constraint)
            : m_constraint(constraint)
        {
        }

        void init(FiniteElementAnimator::FilteredInfo& info)
        {
            using namespace luisa::compute;
            auto& fem = m_constraint->m_finite_element_method;

            // Collect constrained vertices from all animated geometries
            vector<IndexT> h_indices;
            vector<Vector3> h_aim_positions;
            vector<Float> h_strength_ratios;

            for(auto& geo_info : info.anim_geo_infos())
            {
                auto  geo_slot = m_constraint->world().scene().geometries()[geo_info.geo_slot_index];
                auto& geo      = geo_slot->geometry();

                if(geo_info.dim_uid.uid == ConstraintUID)
                {
                    auto  vertex_count = geo_info.vertex_count;
                    auto  vertex_offset = geo_info.vertex_offset;
                    auto& is_constrained = geo.attrs().find<IndexT>(builtin::is_constrained);
                    auto& aim_position = geo.attrs().find<Vector3>(builtin::aim_position);
                    auto& strength_ratio = geo.attrs().find<Float>(builtin::strength_ratio);

                    if(!is_constrained)
                        continue;

                    auto is_constrained_view = is_constrained->view();
                    auto aim_position_view = aim_position->view();
                    auto strength_ratio_view = strength_ratio ? strength_ratio->view() : span<const Float>{};

                    for(SizeT i = 0; i < vertex_count; ++i)
                    {
                        if(is_constrained_view[i])
                        {
                            h_indices.push_back(vertex_offset + i);
                            h_aim_positions.push_back(aim_position_view[i]);
                            h_strength_ratios.push_back(strength_ratio ? strength_ratio_view[i] : 1.0f);
                        }
                    }
                }
            }

            // Create device buffers
            SizeT count = h_indices.size();
            constrained_count = count;

            if(count > 0)
            {
                auto& device = fem->m_impl.engine().device();
                constrained_vertices = device.create_buffer<IndexT>(count);
                aim_positions = device.create_buffer<Vector3>(count);
                strength_ratios = device.create_buffer<Float>(count);

                // Copy data to device
                auto& stream = fem->m_impl.engine().stream();
                stream << constrained_vertices.copy_from(h_indices.data())
                       << aim_positions.copy_from(h_aim_positions.data())
                       << strength_ratios.copy_from(h_strength_ratios.data());
            }
        }

        void step(FiniteElementAnimator::FilteredInfo& info)
        {
            // Update aim positions if needed (for animated constraints)
            using namespace luisa::compute;
            auto& fem = m_constraint->m_finite_element_method;

            if(constrained_count == 0)
                return;

            vector<Vector3> h_aim_positions;
            bool need_update = false;

            for(auto& geo_info : info.anim_geo_infos())
            {
                auto  geo_slot = m_constraint->world().scene().geometries()[geo_info.geo_slot_index];
                auto& geo      = geo_slot->geometry();

                if(geo_info.dim_uid.uid == ConstraintUID)
                {
                    auto  vertex_count = geo_info.vertex_count;
                    auto  vertex_offset = geo_info.vertex_offset;
                    auto& is_constrained = geo.attrs().find<IndexT>(builtin::is_constrained);
                    auto& aim_position = geo.attrs().find<Vector3>(builtin::aim_position);

                    if(!is_constrained)
                        continue;

                    auto is_constrained_view = is_constrained->view();
                    auto aim_position_view = aim_position->view();

                    for(SizeT i = 0; i < vertex_count; ++i)
                    {
                        if(is_constrained_view[i])
                        {
                            h_aim_positions.push_back(aim_position_view[i]);
                            need_update = true;
                        }
                    }
                }
            }

            if(need_update && !h_aim_positions.empty())
            {
                auto& stream = fem->m_impl.engine().stream();
                stream << aim_positions.copy_from(h_aim_positions.data());
            }
        }

        void report_extent(FiniteElementAnimator::ReportExtentInfo& info)
        {
            info.gradient_count(constrained_count);
            if(!info.gradient_only())
                info.hessian_count(constrained_count);
        }

        void compute_energy(FiniteElementAnimator::ComputeEnergyInfo& info)
        {
            using namespace luisa::compute;

            if(constrained_count == 0)
                return;

            auto& fem = m_constraint->m_finite_element_method;
            auto& stream = fem->m_impl.engine().stream();
            auto& device = fem->m_impl.engine().device();

            auto energies = info.energies();
            auto xs = info.xs();
            auto x_prevs = info.x_prevs();
            auto masses = info.masses();
            auto substep_ratio = info.substep_ratio();

            // Kernel to compute soft position constraint energy
            // E = 0.5 * s * m * |x - aim_x|^2
            Kernel1D compute_energy_kernel = [&](
                BufferVar<IndexT> indices,
                BufferVar<Vector3> aim_pos,
                BufferVar<Float> strength,
                BufferVar<Vector3> xs_buf,
                BufferVar<Vector3> x_prevs_buf,
                BufferVar<Float> masses_buf,
                BufferVar<Float> energies_buf,
                Var<Float> substep_ratio_var) noexcept
            {
                auto I = dispatch_x();
                $if(I < constrained_count)
                {
                    auto i = indices.read(I);
                    auto x = xs_buf.read(i);
                    auto x_prev = x_prevs_buf.read(i);
                    auto aim = aim_pos.read(I);
                    auto s = strength.read(I);
                    auto m = masses_buf.read(i);

                    // Interpolate aim position based on substep ratio
                    auto aim_x = lerp(x_prev, aim, substep_ratio_var);
                    auto dx = x - aim_x;
                    auto E = 0.5f * s * m * dot(dx, dx);

                    energies_buf.write(I, E);
                };
            };

            auto shader = device.compile(compute_energy_kernel);
            stream << shader(constrained_vertices, aim_positions, strength_ratios,
                            xs, x_prevs, masses, energies, substep_ratio)
                      .dispatch(constrained_count);
        }

        void compute_gradient_hessian(FiniteElementAnimator::ComputeGradientHessianInfo& info)
        {
            using namespace luisa::compute;

            if(constrained_count == 0)
                return;

            auto& fem = m_constraint->m_finite_element_method;
            auto& stream = fem->m_impl.engine().stream();
            auto& device = fem->m_impl.engine().device();

            auto gradients = info.gradients();
            auto hessians = info.hessians();
            auto xs = info.xs();
            auto x_prevs = info.x_prevs();
            auto masses = info.masses();
            auto is_fixed = info.is_fixed();
            auto substep_ratio = info.substep_ratio();

            // Kernel to compute gradient and hessian
            // G = s * m * (x - aim_x)
            // H = s * m * I
            Kernel1D compute_grad_hess_kernel = [&](
                BufferVar<IndexT> indices,
                BufferVar<Vector3> aim_pos,
                BufferVar<Float> strength,
                BufferVar<Vector3> xs_buf,
                BufferVar<Vector3> x_prevs_buf,
                BufferVar<Float> masses_buf,
                BufferVar<IndexT> is_fixed_buf,
                BufferVar<luisa::uint> grad_indices,
                BufferVar<Vector3> grad_values,
                BufferVar<luisa::uint> hess_row_indices,
                BufferVar<luisa::uint> hess_col_indices,
                BufferVar<Matrix3x3> hess_values,
                Var<Float> substep_ratio_var,
                Var<IndexT> hess_count) noexcept
            {
                auto I = dispatch_x();
                $if(I < constrained_count)
                {
                    auto i = indices.read(I);
                    auto x = xs_buf.read(i);
                    auto x_prev = x_prevs_buf.read(i);
                    auto aim = aim_pos.read(I);
                    auto s = strength.read(I);
                    auto m = masses_buf.read(i);
                    auto fixed = is_fixed_buf.read(i);

                    // Interpolate aim position based on substep ratio
                    auto aim_x = lerp(x_prev, aim, substep_ratio_var);
                    auto dx = x - aim_x;

                    // Gradient: G = s * m * (x - aim_x)
                    Vector3 G = s * m * dx;

                    // If vertex is fixed, gradient is zero
                    $if(fixed != 0)
                    {
                        G = Vector3{0.0f, 0.0f, 0.0f};
                    };

                    // Write gradient doublet
                    grad_indices.write(I, cast<luisa::uint>(i));
                    grad_values.write(I, G);

                    // Hessian: H = s * m * I (only if not gradient_only)
                    $if(hess_count > 0)
                    {
                        Matrix3x3 H = s * m * Matrix3x3::identity();

                        // If vertex is fixed, hessian is zero
                        $if(fixed != 0)
                        {
                            H = Matrix3x3::zero();
                        };

                        // Write hessian triplet (diagonal block)
                        hess_row_indices.write(I, cast<luisa::uint>(i));
                        hess_col_indices.write(I, cast<luisa::uint>(i));
                        hess_values.write(I, H);
                    };
                };
            };

            auto shader = device.compile(compute_grad_hess_kernel);
            stream << shader(constrained_vertices, aim_positions, strength_ratios,
                            xs, x_prevs, masses, is_fixed,
                            gradients.indices, gradients.values,
                            hessians.row_indices, hessians.col_indices, hessians.values,
                            substep_ratio, hessians.count)
                      .dispatch(constrained_count);
        }

        SoftPositionConstraint* m_constraint;
        S<FiniteElementMethod>  m_finite_element_method;

        luisa::compute::Buffer<IndexT> constrained_vertices;
        luisa::compute::Buffer<Vector3> aim_positions;
        luisa::compute::Buffer<Float> strength_ratios;
        SizeT constrained_count = 0;
    };

  protected:
    void do_build(BuildInfo& info) override
    {
        m_impl.m_finite_element_method = &require<FiniteElementMethod>();
    }

    U64 get_uid() const noexcept override { return ConstraintUID; }

    void do_init(FiniteElementAnimator::FilteredInfo& info) override { m_impl.init(info); }
    void do_step(FiniteElementAnimator::FilteredInfo& info) override { m_impl.step(info); }
    void do_report_extent(FiniteElementAnimator::ReportExtentInfo& info) override { m_impl.report_extent(info); }
    void do_compute_energy(FiniteElementAnimator::ComputeEnergyInfo& info) override { m_impl.compute_energy(info); }
    void do_compute_gradient_hessian(FiniteElementAnimator::ComputeGradientHessianInfo& info) override { m_impl.compute_gradient_hessian(info); }

  private:
    Impl m_impl{this};
};

REGISTER_SIM_SYSTEM(SoftPositionConstraint);
}  // namespace uipc::backend::luisa
