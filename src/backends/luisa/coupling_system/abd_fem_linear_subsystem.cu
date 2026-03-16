#include <linear_system/off_diag_linear_subsystem.h>
#include <coupling_system/abd_fem_dytopo_effect_receiver.h>
#include <coupling_system/fem_abd_dytopo_effect_receiver.h>
#include <affine_body/abd_linear_subsystem.h>
#include <finite_element/fem_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <affine_body/affine_body_dynamics.h>
#include <finite_element/finite_element_method.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <kernel_cout.h>
#include <utils/matrix_unpacker.h>

namespace uipc::backend::luisa
{
class ABDFEMLinearSubsystem final : public OffDiagLinearSubsystem
{
  public:
    using OffDiagLinearSubsystem::OffDiagLinearSubsystem;

    SimSystemSlot<GlobalLinearSystem> global_linear_system;

    SimSystemSlot<ABDFEMDyTopoEffectReceiver> abd_fem_dytopo_effect_receiver;

    SimSystemSlot<ABDLinearSubsystem> abd_linear_subsystem;
    SimSystemSlot<FEMLinearSubsystem> fem_linear_subsystem;

    SimSystemSlot<AffineBodyDynamics>  affine_body_dynamics;
    SimSystemSlot<FiniteElementMethod> finite_element_method;

    SimSystemSlot<AffineBodyVertexReporter>    affine_body_vertex_reporter;
    SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;

    virtual void do_build(BuildInfo& info) override
    {
        global_linear_system = require<GlobalLinearSystem>();

        abd_fem_dytopo_effect_receiver = require<ABDFEMDyTopoEffectReceiver>();

        abd_linear_subsystem = require<ABDLinearSubsystem>();
        fem_linear_subsystem = require<FEMLinearSubsystem>();

        affine_body_dynamics  = require<AffineBodyDynamics>();
        finite_element_method = require<FiniteElementMethod>();

        affine_body_vertex_reporter    = require<AffineBodyVertexReporter>();
        finite_element_vertex_reporter = require<FiniteElementVertexReporter>();

        info.connect(abd_linear_subsystem.view(), fem_linear_subsystem.view());
    }

    virtual void report_extent(GlobalLinearSystem::OffDiagExtentInfo& info) override
    {
        if(!abd_fem_dytopo_effect_receiver)
        {
            info.extent(0, 0);
            return;
        }

        // ABD-FEM Hessian: H12x3
        auto abd_fem_dytopo_effect_count =
            abd_fem_dytopo_effect_receiver->hessians().count;
        auto abd_fem_H3x3_count = abd_fem_dytopo_effect_count * 4;

        info.extent(abd_fem_H3x3_count, 0);
    }

    virtual void assemble(GlobalLinearSystem::OffDiagInfo& info) override
    {
        auto count = abd_fem_dytopo_effect_receiver->hessians().count;

        if(count > 0)
        {
            // Get device and stream for kernel dispatch
            auto& engine = this->world().sim_engine();
            auto& device = static_cast<SimEngine&>(engine).device();
            auto& stream = static_cast<SimEngine&>(engine).compute_stream();

            // Create kernel for assembling off-diagonal Hessian blocks
            // ABD-FEM coupling: transforms 3x3 vertex Hessians to 12x3 body Hessians
            Kernel1D assemble_kernel = [&](
                BufferView<IndexT> v2b,
                BufferView<ABDJacobi> Js,
                BufferView<IndexT> body_is_fixed,
                BufferView<IndexT> vertex_is_fixed,
                BufferView<TripletMatrixUnpacker<Float, 3>::TripletEntry> L_triplets,
                BufferView<const IndexT> abd_fem_rows,
                BufferView<const IndexT> abd_fem_cols,
                BufferView<const Matrix3x3> abd_fem_values,
                IndexT abd_point_offset,
                IndexT fem_point_offset) noexcept
            {
                auto I = dispatch_id().x;
                
                // global vertex indices
                IndexT gI_abd_v = abd_fem_rows[I];
                IndexT gJ_fem_v = abd_fem_cols[I];
                Matrix3x3 H3x3 = abd_fem_values[I];

                // Debug assertion - in debug builds, use device_log for diagnostics
                // Note: LuisaCompute doesn't have direct assert in kernels, 
                // but we assume ABD vertices are before FEM vertices

                IndexT I_abd_v = gI_abd_v - abd_point_offset;
                IndexT J_fem_v = gJ_fem_v - fem_point_offset;

                IndexT body_id = v2b[I_abd_v];

                ABDJacobi J = Js[I_abd_v];

                IndexT I4 = 4 * I;

                Matrix12x3 H = J.to_mat().transpose() * H3x3;

                if(body_is_fixed[body_id] || vertex_is_fixed[J_fem_v])
                    H = Matrix12x3::zeros();

                // Use TripletMatrixUnpacker to write the 4x1 block of 3x3 matrices
                // L(I4 + k) writes to (4*body_id + k, J_fem_v)
                TripletMatrixUnpacker<Float, 3> TMU{L_triplets, 4 * count};
                
                // Write 4 triplets for the 12x3 = 4x(3x3) block
                for(int k = 0; k < 4; ++k)
                {
                    auto& entry = L_triplets[I4 + k];
                    entry.row = 4 * body_id + k;
                    entry.col = J_fem_v;
                    entry.value = H.block<3, 3>(3 * k, 0);
                }
            };

            auto shader = device.compile(assemble_kernel);

            // Get the triplet matrix view for lr_hessian
            auto lr_hessian = info.lr_hessian();
            
            // Dispatch the kernel
            stream << shader(
                affine_body_dynamics->v2b(),
                affine_body_dynamics->Js(),
                affine_body_dynamics->body_is_fixed(),
                finite_element_method->is_fixed(),
                lr_hessian.values,
                abd_fem_dytopo_effect_receiver->hessians().row_indices,
                abd_fem_dytopo_effect_receiver->hessians().col_indices,
                abd_fem_dytopo_effect_receiver->hessians().values,
                affine_body_vertex_reporter->vertex_offset(),
                finite_element_vertex_reporter->vertex_offset()
            ).dispatch(count);
        }
    }
};

REGISTER_SIM_SYSTEM(ABDFEMLinearSubsystem);
}  // namespace uipc::backend::luisa
